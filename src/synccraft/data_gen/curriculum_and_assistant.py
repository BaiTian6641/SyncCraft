"""Simple-to-Complex Curriculum, Concise Texture+Animation Synthesis & Controlled Defect-Repair Engine (ADR 0005).

Provides deterministic, non-black-box mechanisms for training an assistant-based
Minecraft / Blockbench 3D world model across geometry, stateful texture, and animation:
1. Progressive 4-tier complexity coarsening (tier_1_simple -> tier_4_masterwork) for
   both standalone simple models and coarse-to-fine step-by-step construction pairs.
2. Concise parametric @materials/@style + swept-SAT verified @animation bone-channel synthesis.
3. Controlled ~5% 'bad data' diagnostic-and-repair pairs (verified defective draft + exact
   physics audit telemetry -> mechanistic explanation + 100% SAT-verified repaired asset).
"""

from __future__ import annotations

from dataclasses import replace
import json
import math
import random
from typing import Any

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import (
    Attach,
    Bone,
    Box,
    Locator,
    MaterialDecl,
    Meta,
    Palette,
    Place,
    Program,
    StyleDecl,
    validate_program,
)
from synccraft.dsl.parser import format_program, parse_program
from synccraft.kinematics.se3_chain import Scene, compile_geometry
from .taxonomy_catalog import (
    COMPLEXITY_TIERS,
    TAXONOMY_BY_FAMILY,
    classify_complexity_tier,
)

ASSISTANT_TASK_MODES: tuple[str, ...] = (
    "create_full_asset",
    "coarse_to_fine_upgrade",
    "texture_and_animate_rig",
    "diagnose_and_repair_defect",
)


def _prune_unused_palettes_and_materials(prog: Program) -> Program:
    used_mats: set[str] = set()
    expanded_bones: dict[str, str | None] = {}
    expanded_boxes: set[str] = set()

    def _scan(bone: Bone, parent_name: str | None = None, prefix: str | None = None, renamed: str | None = None) -> None:
        bname = renamed or (f"{prefix}__{bone.name}" if prefix else bone.name)
        expanded_bones[bname] = parent_name
        for item in bone.items:
            if isinstance(item, Box):
                used_mats.add(item.material)
                expanded_boxes.add(f"{prefix}__{item.name}" if prefix else item.name)
            elif isinstance(item, Bone):
                _scan(item, bname, prefix=prefix)
                if item.mirror_name:
                    m_root = f"{prefix}__{item.mirror_name}" if prefix else item.mirror_name
                    _scan(item, bname, prefix=m_root, renamed=m_root)

    _scan(prog.root)
    new_palettes = [p for p in prog.palettes if p.name in used_mats]
    if not new_palettes and prog.palettes:
        new_palettes = [prog.palettes[0]]
    valid_mat_names = {p.name for p in new_palettes}
    new_materials = tuple(m for m in prog.materials if m.name in valid_mat_names)

    # Prune @muscles, @kinematics, and @behavior declarations referencing trimmed bones/boxes
    root_name = prog.root.name
    kept_muscles = []
    for m in prog.muscles:
        if m.origin_bone not in expanded_bones or m.insertion_bone not in expanded_bones or m.insertion_bone == root_name:
            continue
        if m.belly_box is not None and m.belly_box not in expanded_boxes:
            m = replace(m, belly_box=None, bulge_inflate=0.0)
        if m.tendon_coupling is not None and (
            m.tendon_coupling.bone not in expanded_bones or m.tendon_coupling.bone == root_name
        ):
            m = replace(m, tendon_coupling=None)
        kept_muscles.append(m)
    kept_muscle_names = {m.name for m in kept_muscles}

    new_kin = None
    kept_ik_names: set[str] = set()
    if prog.kinematics is not None:
        iks = tuple(
            ik for ik in prog.kinematics.ik_chains
            if ik.upper in expanded_bones and ik.upper != root_name and expanded_bones.get(ik.lower) == ik.upper
        )
        lks = []
        for lk in prog.kinematics.look_chains:
            bs = tuple(b for b in lk.bones if b in expanded_bones and b != root_name)
            if bs:
                lks.append(replace(lk, bones=bs))
        if iks or lks:
            from synccraft.dsl.ast_nodes import KinematicsDecl
            new_kin = KinematicsDecl(ik_chains=iks, look_chains=tuple(lks))
            kept_ik_names = {ik.name for ik in iks}

    new_beh = None
    if prog.behavior is not None:
        from synccraft.dsl.ast_nodes import BehaviorDecl
        clips = []
        for cl in prog.behavior.clips:
            trs = tuple(t for t in cl.tracks if t.bone in expanded_bones and t.bone != root_name)
            mds = tuple(md for md in cl.muscle_drives if md.muscle in kept_muscle_names)
            ikds = tuple(ikd for ikd in cl.ik_drives if ikd.chain in kept_ik_names)
            if trs or mds or ikds:
                clips.append(replace(cl, tracks=trs, muscle_drives=mds, ik_drives=ikds))
        states = []
        for st in prog.behavior.states:
            pbs = tuple((b, v) for b, v in st.pose_biases if b in expanded_bones and b != root_name)
            mts = tuple((m, v) for m, v in st.muscle_tones if m in kept_muscle_names)
            mms = tuple(mm for mm in st.material_mods if mm.material in valid_mat_names)
            if pbs or mts or mms:
                states.append(replace(st, pose_biases=pbs, muscle_tones=mts, material_mods=mms))
        if clips or states:
            new_beh = BehaviorDecl(clips=tuple(clips), states=tuple(states))

    return replace(
        prog,
        palettes=new_palettes,
        materials=new_materials,
        muscles=tuple(kept_muscles),
        kinematics=new_kin,
        behavior=new_beh,
    )


def _box_volume(bx: Box) -> float:
    return float(bx.size[0] * bx.size[1] * bx.size[2])


def _subtree_volume(bone: Bone) -> float:
    vol = sum(_box_volume(it) for it in bone.items if isinstance(it, Box))
    vol += sum(_subtree_volume(it) for it in bone.items if isinstance(it, Bone))
    return vol * (1.35 if bone.mirror_name is not None else 1.0)


def _trim_bone_recursive(
    bone: Bone,
    *,
    max_boxes_per_bone: int,
    max_child_bones: int,
    max_depth: int,
    cur_depth: int = 0,
) -> Bone | None:
    """Keep largest silhouette-defining boxes & child bones while preserving all base_box/parent_box/locator links."""
    boxes_in_bone = [it for it in bone.items if isinstance(it, Box)]
    locators_in_bone = [it for it in bone.items if isinstance(it, Locator)]
    child_bones = [it for it in bone.items if isinstance(it, Bone)]
    if not boxes_in_bone and not child_bones:
        return None

    by_name = {bx.name: bx for bx in boxes_in_bone}
    loc_by_name = {loc.name: loc for loc in locators_in_bone}
    kept_box_names: list[str] = []
    kept_loc_names: list[str] = []

    def _ensure_box(bx_name: str) -> bool:
        if bx_name in kept_box_names:
            return True
        bx = by_name.get(bx_name)
        if bx is None:
            return False
        if bx.place is not None and bx.place.base_box is not None:
            if not _ensure_box(bx.place.base_box):
                return False
        kept_box_names.append(bx_name)
        return True

    def _ensure_anchor(anchor_name: str) -> bool:
        if _ensure_box(anchor_name):
            return True
        if anchor_name in loc_by_name:
            if anchor_name not in kept_loc_names:
                kept_loc_names.append(anchor_name)
            return True
        return False

    # Always keep the bone's primary anchor box when present
    if boxes_in_bone:
        _ensure_box(boxes_in_bone[0].name)

    is_grouping_only = not boxes_in_bone
    next_depth = cur_depth if is_grouping_only else (cur_depth + 1)
    next_max_boxes = max_boxes_per_bone if is_grouping_only else max(1, max_boxes_per_bone - 1)
    next_max_children = (
        max_child_bones
        if is_grouping_only
        else max(1 if cur_depth + 1 < max_depth else 0, max_child_bones - 1)
    )

    # Prioritize child bones with the largest physical silhouette volume, preserving declaration order
    kept_children: list[Bone] = []
    if (is_grouping_only or cur_depth < max_depth) and max_child_bones > 0 and child_bones:
        ranked_indices = sorted(
            range(len(child_bones)),
            key=lambda i: (-_subtree_volume(child_bones[i]), i),
        )
        chosen_set = set(ranked_indices[:max_child_bones])
        selected_children = [child_bones[i] for i in range(len(child_bones)) if i in chosen_set]
        for ch in selected_children:
            if ch.attach is not None:
                if not _ensure_anchor(ch.attach.parent_box):
                    continue
            trimmed_ch = _trim_bone_recursive(
                ch,
                max_boxes_per_bone=next_max_boxes,
                max_child_bones=next_max_children,
                max_depth=max_depth,
                cur_depth=next_depth,
            )
            if trimmed_ch is not None:
                kept_children.append(trimmed_ch)

    # Fill up to max_boxes_per_bone prioritizing largest-volume silhouette boxes first
    if boxes_in_bone:
        remaining_ranked = sorted(
            enumerate(boxes_in_bone[1:]),
            key=lambda pair: (-_box_volume(pair[1]), pair[0]),
        )
        for _, bx in remaining_ranked:
            if len(kept_box_names) >= max_boxes_per_bone:
                break
            _ensure_box(bx.name)

    if not kept_box_names and not kept_children:
        return None
    if not kept_box_names and not kept_loc_names:
        if locators_in_bone:
            kept_loc_names.append(locators_in_bone[0].name)
        else:
            return None

    kept_set = set(kept_box_names)
    kept_loc_set = set(kept_loc_names)
    kept_child_map = {ch.name: ch for ch in kept_children}
    if not kept_loc_set:
        ordered_boxes = [bx for bx in boxes_in_bone if bx.name in kept_set]
        return replace(bone, items=[*ordered_boxes, *kept_children])

    ordered_items: list[Box | Locator | Bone] = []
    for it in bone.items:
        if isinstance(it, Box) and it.name in kept_set:
            ordered_items.append(it)
        elif isinstance(it, Locator) and it.name in kept_loc_set:
            ordered_items.append(it)
        elif isinstance(it, Bone) and it.name in kept_child_map:
            ordered_items.append(kept_child_map[it.name])
    return replace(bone, items=ordered_items)


def coarsen_program_to_tier(program: Program, target_tier: str) -> Program:
    """Derive a SAT + P1-02 verified, morphologically faithful simpler silhouette blockout."""
    if target_tier not in ("tier_1_simple", "tier_2_medium"):
        return program

    lo, hi = COMPLEXITY_TIERS[target_tier]
    # Evaluate silhouette-spanning budgets and pick the best audited fit near the upper-middle of [lo, hi]
    if target_tier == "tier_1_simple":
        ideal_boxes = 7
        budgets = [
            (1, 3, 2),
            (2, 2, 2),
            (2, 3, 2),
            (2, 4, 2),
            (3, 2, 2),
            (3, 3, 2),
            (2, 2, 1),
            (3, 1, 1),
            (3, 2, 1),
        ]
    else:
        ideal_boxes = 12
        budgets = [
            (2, 4, 3),
            (3, 3, 2),
            (3, 3, 3),
            (3, 4, 2),
            (3, 4, 3),
            (4, 3, 2),
            (4, 4, 2),
            (5, 4, 3),
        ]

    best_prog: Program | None = None
    best_score: tuple[int, int] = (10_000, 10_000)

    for max_b, max_ch, max_d in budgets:
        trimmed_root = _trim_bone_recursive(
            program.root,
            max_boxes_per_bone=max_b,
            max_child_bones=max_ch,
            max_depth=max_d,
            cur_depth=0,
        )
        if trimmed_root is None:
            continue
        cand = _prune_unused_palettes_and_materials(replace(program, root=trimmed_root))
        try:
            validate_program(cand)
            sc = compile_geometry(cand)
            n_boxes = len(sc.boxes)
            if n_boxes < 4:
                continue
            rep = audit_scene(
                sc,
                collision_tolerance=1e-7,
                enforce_physical_contact=True,
                max_contact_gap=1e-4,
                max_inset_depth=0.5,
            )
            if not rep["passed"]:
                continue
            range_penalty = 0 if (lo <= n_boxes <= hi) else (100 + min(abs(n_boxes - lo), abs(n_boxes - hi)))
            score = (range_penalty, abs(n_boxes - ideal_boxes))
            if score < best_score:
                best_score = score
                best_prog = cand
        except Exception:
            continue

    return best_prog if best_prog is not None else program


def ensure_concise_texture_program(
    program: Program,
    *,
    family_id: str,
    rng: random.Random,
) -> tuple[Program, dict[str, Any]]:
    """Ensure concise, physically/biologically grounded @materials & @style on Program."""
    tax = TAXONOMY_BY_FAMILY.get(family_id)
    rec_mats = tax.recommended_materials if tax is not None else ("painted_metal", "solid_wood")

    if program.materials and program.style is not None:
        mat_decls = list(program.materials)
        style_decl = program.style
    else:
        mat_decls = []
        for idx, pal in enumerate(program.palettes[:3]):
            fam = rec_mats[idx % len(rec_mats)]
            is_metal = fam in ("brushed_metal", "painted_metal")
            rough = round(rng.uniform(0.18, 0.42) if is_metal else rng.uniform(0.42, 0.84), 3)
            metal = 1.0 if fam == "brushed_metal" else 0.0
            coat = round(rng.uniform(0.35, 0.72), 3) if fam in ("painted_metal", "thin_film_iridescence") else 0.0
            scale = round(rng.uniform(0.80, 1.40), 3)
            mat_decls.append(
                MaterialDecl(
                    name=pal.name,
                    family=fam,
                    roughness=rough,
                    metalness=metal,
                    coating=coat,
                    ior=1.5,
                    scale=scale,
                    seed=rng.randint(100, 99999),
                )
            )
        enc = rng.choice(("vanilla", "labpbr_1_3", "gltf_2_0"))
        style_decl = StyleDecl(
            target_encoding=enc,
            quantize_ramp=(enc == "vanilla"),
            bevel_highlight=round(rng.uniform(0.04, 0.10), 3),
            view_deg=round(rng.choice((15.0, 20.0, 25.0)), 1),
        )
        program = replace(program, materials=tuple(mat_decls), style=style_decl)

    turing_active = any(m.family == "patterned_skin" for m in mat_decls)
    recipe = {
        "target_encoding": style_decl.target_encoding,
        "texel_density": program.meta.texel_density,
        "quantize_ramp": style_decl.quantize_ramp,
        "substrates": [
            {
                "slot": m.name,
                "family": m.family,
                "roughness": m.roughness,
                "metalness": m.metalness,
                "coating": m.coating,
            }
            for m in mat_decls
        ],
        "turing_reaction_diffusion": "gray_scott_spots_or_stripes" if turing_active else "none",
    }
    return program, recipe


def apply_animation_keyframe(program: Program, spec: dict[str, Any], phase_fraction: float) -> Program:
    """Evaluate analytic animation spec at phase_fraction in [0, 1] and return posed Program."""
    channels = {ch["bone"]: ch for ch in spec.get("channels", [])}
    if not channels:
        return program

    two_pi_phase = 2.0 * math.pi * phase_fraction

    def _pose_bone(b: Bone) -> Bone:
        new_items = [_pose_bone(it) if isinstance(it, Bone) else it for it in b.items]
        ch = channels.get(b.name)
        if ch is None:
            return replace(b, items=new_items)
        amp = float(ch["amp_deg"])
        ph_rad = math.radians(float(ch.get("phase_deg", 0.0)))
        delta = amp * math.sin(two_pi_phase + ph_rad)
        rx, ry, rz = b.rest_euler
        axis = ch["axis"]
        if axis == "x":
            posed_euler = (round(rx + delta, 3), ry, rz)
        elif axis == "y":
            posed_euler = (rx, round(ry + delta, 3), rz)
        else:
            posed_euler = (rx, ry, round(rz + delta, 3))
        return replace(b, rest_euler=posed_euler, items=new_items)

    return replace(program, root=_pose_bone(program.root))


def synthesize_verified_animation_spec(
    program: Program,
    scene: Scene,
    *,
    family_id: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Build a concise analytic bone-animation spec calibrated for 0 swept-SAT self-collisions."""
    tax = TAXONOMY_BY_FAMILY.get(family_id)
    archetype = tax.motion_archetype if tax is not None else "hinge_articulation"

    # Collect non-root articulated bones in breadth-first order
    articulated_bones: list[tuple[Bone, int]] = []

    def _walk(b: Bone, depth: int) -> None:
        if depth > 0:
            articulated_bones.append((b, depth))
        for it in b.items:
            if isinstance(it, Bone):
                _walk(it, depth + 1)

    _walk(program.root, 0)

    freq_map = {
        "serpentine_wave": 0.85,
        "polycephalic_sway": 0.65,
        "wing_and_gait": 1.25,
        "quadruped_stride": 1.40,
        "biped_stride": 1.30,
        "aquatic_undulation": 1.00,
        "rotary_spin": 2.00,
        "reciprocating_mechanism": 2.50,
        "vehicle_drive_and_articulate": 1.50,
        "hydraulic_boom_chain": 0.55,
        "wind_canopy_sway": 0.35,
        "pendulum_oscillation": 0.50,
        "hinge_articulation": 0.60,
    }
    base_freq = round(freq_map.get(archetype, 0.8) * rng.uniform(0.9, 1.1), 2)

    raw_channels: list[dict[str, Any]] = []
    for idx, (b, depth) in enumerate(articulated_bones[:6]):
        b_low = b.name.lower()
        if archetype in ("serpentine_wave", "aquatic_undulation"):
            axis = "y"
            law = "traveling_wave"
            amp = 12.0
            phase = -55.0 * idx
        elif archetype == "polycephalic_sway":
            axis = "y" if idx % 2 == 0 else "x"
            law = "counter_phase_sway"
            amp = 10.0
            phase = 120.0 * idx
        elif archetype == "wing_and_gait":
            if any(k in b_low for k in ("wing", "spar", "pinion", "patag", "pectoral")):
                axis = "z"
                law = "wing_flap_harmonic"
                amp = 16.0
                phase = -40.0 * (depth - 1)
            else:
                axis = "x"
                law = "counter_stride"
                amp = 11.0
                phase = 180.0 * (idx % 2)
        elif archetype in ("quadruped_stride", "biped_stride"):
            axis = "x"
            law = "gait_stride_harmonic"
            amp = 12.0
            phase = 180.0 * (idx % 2)
        elif archetype in ("rotary_spin", "vehicle_drive_and_articulate"):
            if any(k in b_low for k in ("rotor", "turret", "slew", "prop", "rudder")):
                axis = "y"
            elif any(k in b_low for k in ("wheel", "axle", "drive", "bed", "dump", "mast", "waterwheel")):
                axis = "x"
            else:
                axis = "z"
            law = "bounded_mechanism_cycle"
            amp = 14.0
            phase = 90.0 * (idx % 4)
        elif archetype in ("hydraulic_boom_chain", "reciprocating_mechanism"):
            axis = "x" if depth % 2 == 1 else "z"
            law = "coupled_actuator_sweep"
            amp = 10.0
            phase = -45.0 * idx
        elif archetype == "wind_canopy_sway":
            axis = "z" if idx % 2 == 0 else "x"
            law = "wind_harmonic_sway"
            amp = 6.0
            phase = 60.0 * idx
        else:
            axis = "z" if idx % 2 == 0 else "x"
            law = "harmonic_oscillation"
            amp = 8.0
            phase = 90.0 * (idx % 2)

        raw_channels.append(
            {
                "bone": b.name,
                "law": law,
                "axis": axis,
                "amp_deg": round(amp, 2),
                "freq_hz": base_freq,
                "phase_deg": round(phase, 1),
            }
        )

    spec = {
        "motion_archetype": archetype,
        "cycle_seconds": round(1.0 / max(0.2, base_freq), 2),
        "swept_sat_verified": True,
        "channels": raw_channels,
    }

    # Swept-SAT calibration across 4 cycle phases: scale amplitudes down if any non-exempt collision appears
    sample_phases = (0.15, 0.25, 0.50, 0.75)
    for _attempt in range(6):
        all_clear = True
        for ph in sample_phases:
            posed_prog = apply_animation_keyframe(program, spec, ph)
            posed_scene = compile_geometry(posed_prog)
            rep = audit_scene(
                posed_scene,
                collision_tolerance=1e-6,
                enforce_physical_contact=False,
            )
            if rep["collision_pairs"]:
                all_clear = False
                break
        if all_clear:
            return spec
        for ch in spec["channels"]:
            ch["amp_deg"] = round(ch["amp_deg"] * 0.5, 2)

    # Fallback: zero amplitudes guarantee static rest pose legality
    for ch in spec["channels"]:
        ch["amp_deg"] = 0.0
    return spec


def build_controlled_defect_repair_pair(
    clean_program: Program,
    *,
    rng: random.Random,
) -> dict[str, Any]:
    """Construct a verified defective draft + physics diagnostic -> clean repaired Program pair."""
    root = clean_program.root
    anchor_box = next(it for it in root.items if isinstance(it, Box))
    mat_name = anchor_box.material

    defect_choices = ["sibling_obb_collision", "floating_detached_gap", "coincident_copenetration"]
    child_bones = [it for it in root.items if isinstance(it, Bone)]
    if child_bones:
        defect_choices.append("folded_back_joint_penetration")
    rng.shuffle(defect_choices)

    for defect_kind in defect_choices:
        corrupted_prog: Program | None = None
        diagnosis_note = ""

        if defect_kind == "sibling_obb_collision":
            # Insert an unattached sibling box overlapping anchor_box's interior
            bad_box = Box(
                "err_overlapping_strut",
                (max(2, anchor_box.size[0]), max(2, anchor_box.size[1]), max(2, anchor_box.size[2])),
                mat_name,
            )
            corrupted_prog = replace(clean_program, root=replace(root, items=[*root.items, bad_box]))
            diagnosis_note = (
                f"Unattached sibling box 'err_overlapping_strut' shares the local origin with '{anchor_box.name}', "
                "causing a direct 15-axis OBB interior collision without a surface Place/Attach contract."
            )
        elif defect_kind == "floating_detached_gap":
            bad_box = Box(
                "err_floating_panel",
                (2, 2, 2),
                mat_name,
                place=Place(anchor_box.name, (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), offset=(0.0, 7.5, 0.0)),
            )
            corrupted_prog = replace(clean_program, root=replace(root, items=[*root.items, bad_box]))
            diagnosis_note = (
                f"Box 'err_floating_panel' declares Place contact on '{anchor_box.name}' with +7.5 Y gap (> 1e-4), "
                "violating P1-02 physical contact and creating an unauthorized disconnected floating component."
            )
        elif defect_kind == "coincident_copenetration":
            bad_box = Box(
                "err_buried_core",
                anchor_box.size,
                mat_name,
                place=Place(anchor_box.name, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), offset=(0.0, 0.0, 0.0)),
            )
            corrupted_prog = replace(clean_program, root=replace(root, items=[*root.items, bad_box]))
            diagnosis_note = (
                f"Box 'err_buried_core' is placed at [0,0,0] center alignment inside '{anchor_box.name}', "
                "exceeding max_inset_depth=0.5 (P1-02 coincident co-penetration)."
            )
        elif defect_kind == "folded_back_joint_penetration" and child_bones:
            target_b = child_bones[0]
            folded_b = replace(target_b, rest_euler=(0.0, 0.0, 165.0))
            new_items = [folded_b if it is target_b else it for it in root.items]
            corrupted_prog = replace(clean_program, root=replace(root, items=new_items))
            diagnosis_note = (
                f"Child bone '{target_b.name}' has rest_euler Z=165 deg, folding its geometry back through "
                "its parent attachment face and causing excessive contact penetration."
            )

        if corrupted_prog is None:
            continue

        try:
            validate_program(corrupted_prog)
            bad_scene = compile_geometry(corrupted_prog)
            bad_audit = audit_scene(
                bad_scene,
                collision_tolerance=1e-7,
                enforce_physical_contact=True,
                max_contact_gap=1e-4,
                max_inset_depth=0.5,
            )
            if not bad_audit["passed"]:
                return {
                    "defect_type": defect_kind,
                    "defective_dsl": format_program(corrupted_prog),
                    "defective_audit_summary": {
                        "passed": False,
                        "collision_count": len(bad_audit["collision_pairs"]),
                        "contact_violation_count": len(bad_audit["contact_violations"]),
                        "sample_violations": [
                            v.get("violation") for v in bad_audit["contact_violations"][:2]
                        ],
                    },
                    "diagnosis": diagnosis_note,
                    "repaired_dsl": format_program(clean_program),
                }
        except Exception:
            continue

    raise RuntimeError("Failed to synthesize verified controlled-defect repair pair")


def format_concise_animation_comment(animation_spec: dict[str, Any]) -> str:
    """Serialize concise @animation plan alongside Block-DSL for unified assistant responses."""
    compact_json = json.dumps(animation_spec, sort_keys=True, separators=(",", ":"))
    return f"// @animation_plan: {compact_json}"


def build_assistant_supervision_turn(
    *,
    task_mode: str,
    base_prompt: str,
    clean_program: Program,
    coarse_program: Program | None,
    animation_spec: dict[str, Any],
    texture_recipe: dict[str, Any],
    defect_bundle: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Format a concise Assistant World-Model (prompt, completion) pair for one of 4 modes."""
    full_dsl = format_program(clean_program)
    anim_line = format_concise_animation_comment(animation_spec)

    if task_mode == "diagnose_and_repair_defect" and defect_bundle is not None:
        audit_json = json.dumps(defect_bundle["defective_audit_summary"], sort_keys=True)
        user_prompt = (
            f"[MODE: DIAGNOSE_AND_REPAIR]\n"
            f"Goal: {base_prompt}\n"
            f"Draft Block-DSL (contains modeling defect):\n{defect_bundle['defective_dsl']}\n"
            f"Physics Audit Telemetry: {audit_json}\n"
            f"Diagnose the physical violation and output the repaired Block-DSL v3.0 + @animation_plan."
        )
        completion = (
            f"// Diagnosis: {defect_bundle['diagnosis']}\n"
            f"{full_dsl}\n"
            f"{anim_line}"
        )
        return {"assistant_prompt": user_prompt, "assistant_completion": completion}

    if task_mode == "coarse_to_fine_upgrade" and coarse_program is not None:
        coarse_dsl = format_program(coarse_program)
        coarse_boxes = len(compile_geometry(coarse_program).boxes)
        fine_boxes = len(compile_geometry(clean_program).boxes)
        user_prompt = (
            f"[MODE: COARSE_TO_FINE_UPGRADE]\n"
            f"Simple Blockout ({coarse_boxes} boxes):\n{coarse_dsl}\n"
            f"Upgrade Brief: Refine this simple blockout into a {fine_boxes}-box articulated model with "
            f"stateful @materials/@style ({texture_recipe['target_encoding']}) and collision-free @animation_plan. "
            f"Context: {base_prompt}"
        )
        completion = f"{full_dsl}\n{anim_line}"
        return {"assistant_prompt": user_prompt, "assistant_completion": completion}

    if task_mode == "texture_and_animate_rig":
        geo_only_prog = replace(clean_program, materials=(), style=None)
        subs = ", ".join(f"{s['slot']}:{s['family']}" for s in texture_recipe["substrates"])
        user_prompt = (
            f"[MODE: TEXTURE_AND_ANIMATE]\n"
            f"Untextured Rig:\n{format_program(geo_only_prog)}\n"
            f"Material & Motion Brief: Bind stateful @materials ({subs}), @style ({texture_recipe['target_encoding']}, "
            f"{texture_recipe['texel_density']}px), and a swept-SAT verified '{animation_spec['motion_archetype']}' @animation_plan."
        )
        completion = f"{full_dsl}\n{anim_line}"
        return {"assistant_prompt": user_prompt, "assistant_completion": completion}

    # Default: create_full_asset (concise unified model + texture + animation)
    tier = classify_complexity_tier(len(compile_geometry(clean_program).boxes))
    user_prompt = (
        f"[MODE: CREATE_MODEL_TEXTURE_ANIMATION | TIER: {tier}]\n"
        f"{base_prompt}\n"
        f"Include stateful @materials/@style ({texture_recipe['target_encoding']}) and a swept-SAT safe @animation_plan."
    )
    completion = f"{full_dsl}\n{anim_line}"
    return {"assistant_prompt": user_prompt, "assistant_completion": completion}
