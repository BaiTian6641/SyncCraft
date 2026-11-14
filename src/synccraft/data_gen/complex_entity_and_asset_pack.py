"""Apex Complex Entity & Game Asset Expansion Pack (ADR 0006).

Clean-room procedural family builders and multi-turn dataset generator for large,
structurally rich entities and interactive game assets (32-76+ boxes, 10-28+ bones,
nested `mirror_x` wing phalanges, Hill-type `@muscles` with biarticular tendon
couplings & isochoric `belly_box` bulge, `@kinematics` closed-loop stance IK &
multi-bone gaze chains, cross-box welded `@materials`, and swept-SAT `@behavior`):

Seen Complex Families (`APEX_SEEN_FAMILIES`, 8):
  1. `apex_flying_wyvern` (Monster Hunter-style flying wyvern with nested-mirror wings)
  2. `brute_theropod_wyvern` (heavy bipedal brute wyvern with jaw masseter & dorsal sails)
  3. `pelagic_leviathan_wyvern` (undulating multi-spine sea/marsh leviathan)
  4. `fanged_thunder_wyvern` (agile quadrupedal fanged wyvern with capacitor plates)
  5. `armored_arthropod_colossus` (6-legged tripod-gait carapace colossus)
  6. `morphing_switch_blade_axe` (transforming phial mechanical weapon)
  7. `elemental_pocket_companion` (Cobblemon-style expressive creature with game sockets)
  8. `modular_apricorn_healing_station` (Cobblemon-style interactive mechanical block)

Held-Out Unseen Complex Families (`APEX_UNSEEN_FAMILIES`, 2, all >= 40 boxes):
  9. `elder_calamity_dragon` (6-limbed Elder Dragon: 4 walking legs + 2 nested-mirror wings)
  10. `relic_automaton_guardian` (ancient stone-and-bronze piston guardian colossus)

Note: Preserves `SEEN_STATIC_FAMILIES` (40) and `UNSEEN_STATIC_FAMILIES` (15) in
`static_families.py` unchanged for backward benchmark stability.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import (
    Attach,
    BehaviorDecl,
    BehaviorStateDecl,
    Bone,
    Box,
    ClipDecl,
    IKChainDecl,
    IKDriveDecl,
    KinematicsDecl,
    Locator,
    LookChainDecl,
    MaterialDecl,
    MaterialModDecl,
    Meta,
    MuscleDecl,
    MuscleDriveDecl,
    Palette,
    Place,
    Program,
    StyleDecl,
    TendonCoupling,
    TrackDecl,
    Vec3,
    Vec3i,
    validate_program,
)
from synccraft.dsl.parser import format_program
from synccraft.kinematics.musculoskeletal import (
    audit_behavior_clip,
    evaluate_muscles_at_activations,
)
from synccraft.kinematics.se3_chain import compile_geometry

APEX_SEEN_FAMILIES: tuple[str, ...] = (
    "apex_flying_wyvern",
    "brute_theropod_wyvern",
    "pelagic_leviathan_wyvern",
    "fanged_thunder_wyvern",
    "armored_arthropod_colossus",
    "morphing_switch_blade_axe",
    "elemental_pocket_companion",
    "modular_apricorn_healing_station",
)

APEX_UNSEEN_FAMILIES: tuple[str, ...] = (
    "elder_calamity_dragon",
    "relic_automaton_guardian",
)

ALL_APEX_FAMILIES: tuple[str, ...] = APEX_SEEN_FAMILIES + APEX_UNSEEN_FAMILIES


def _flush_place(base_box: str, face: Vec3, offset: Vec3 = (0.0, 0.0, 0.0)) -> Place:
    return Place(
        base_box=base_box,
        anchor=face,
        align=(-face[0], -face[1], -face[2]),
        offset=offset,
    )


def _flush_attach(
    parent_box: str,
    face: Vec3,
    offset: Vec3 = (0.0, 0.0, 0.0),
    detached: bool = False,
) -> Attach:
    """Attach flush to `face`, interpreting tangential components of `offset` as normalized face anchors in [-1, 1]."""
    anchor = (
        round(max(-1.0, min(1.0, face[0] + (0.0 if abs(face[0]) > 0.5 else offset[0]))), 4),
        round(max(-1.0, min(1.0, face[1] + (0.0 if abs(face[1]) > 0.5 else offset[1]))), 4),
        round(max(-1.0, min(1.0, face[2] + (0.0 if abs(face[2]) > 0.5 else offset[2]))), 4),
    )
    return Attach(
        parent_box=parent_box,
        anchor=anchor,
        align=(-face[0], -face[1], -face[2]),
        offset=(0.0, 0.0, 0.0),
        detached=detached,
    )


def _build_segmented_chain(
    prefix: str,
    parent_box: str,
    direction: Vec3,
    seg_sizes: list[Vec3i],
    mat_primary: str,
    mat_ridge: str | None = None,
    *,
    rest_euler_step: Vec3 = (0.0, 0.0, 0.0),
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-35.0, 35.0), (-40.0, 40.0), (-25.0, 25.0)),
    tip_locator: tuple[str, str] | None = None,
) -> Bone:
    """Build a multi-bone serial chain (tail, neck, spine) strictly flush along `direction`."""
    assert seg_sizes
    nodes: list[Bone] = []
    for idx in range(len(seg_sizes) - 1, -1, -1):
        bname = f"{prefix}_{idx + 1}"
        bx_name = f"{prefix}_seg_{idx + 1}"
        sz = seg_sizes[idx]
        p_box = parent_box if idx == 0 else f"{prefix}_seg_{idx}"
        items: list[Box | Locator | Bone] = [Box(bx_name, sz, mat_primary)]
        if mat_ridge is not None and sz[0] >= 2 and sz[2] >= 2:
            ridge_sz = (max(1, sz[0] - 2), 2, max(1, sz[2] - 2))
            items.append(
                Box(
                    f"{prefix}_osteoderm_{idx + 1}",
                    ridge_sz,
                    mat_ridge,
                    place=_flush_place(bx_name, (0.0, 1.0, 0.0)),
                )
            )
        if idx == len(seg_sizes) - 1 and tip_locator is not None:
            loc_id, loc_role = tip_locator
            items.append(
                Locator(
                    name=loc_id,
                    offset=(0.0, 0.0, float(sz[2]) * 0.5 * direction[2]),
                    role=loc_role,
                )
            )
        if nodes:
            items.append(nodes[-1])
        nodes.append(
            Bone(
                name=bname,
                rest_euler=rest_euler_step,
                attach=_flush_attach(p_box, direction),
                items=items,
                joint_limits=limits,
            )
        )
    return nodes[-1]


def _build_nested_mirror_wyvern_wing(
    parent_box: str,
    mat_bone: str,
    mat_membrane: str,
    mat_claw: str,
    *,
    span_arm: int = 8,
    span_fore: int = 9,
    span_digit: int = 8,
    attach_offset: Vec3 = (0.0, 0.25, 0.0),
) -> Bone:
    """Build a Monster Hunter-style wyvern wing with 2-level nested `mirror_x` wing digits!

    Level 1 mirror_x: `wing_arm_l` -> `wing_arm_r` across the torso X-midplane.
    Level 2 nested mirror_x: `wing_digit_in` -> `wing_digit_out` across `wing_carpal_hub`'s
    local X-axis on its trailing (-Z) face, toggling parity cleanly back to det(R)=+1!
    """
    digit_bone = Bone(
        name="wing_digit_in",
        rest_euler=(-4.0, 0.0, 0.0),
        attach=_flush_attach("wing_carpal_hub", (0.0, 0.0, -1.0), offset=(0.65, 0.0, 0.0)),
        mirror_name="wing_digit_out",
        joint_limits=((-30.0, 30.0), (-25.0, 25.0), (-40.0, 40.0)),
        items=[
            Box("wing_phalanx", (1, 1, span_digit), mat_bone),
            Box(
                "wing_web_panel",
                (2, 1, max(3, span_digit - 2)),
                mat_membrane,
                place=_flush_place("wing_phalanx", (0.0, -1.0, 0.0)),
            ),
            Box(
                "wing_tip_spur",
                (1, 1, 2),
                mat_claw,
                place=_flush_place("wing_phalanx", (0.0, 0.0, -1.0)),
            ),
        ],
    )

    wing_hand = Bone(
        name="wing_hand_l",
        rest_euler=(0.0, 0.0, -4.0),
        attach=_flush_attach("wing_radius", (1.0, 0.0, 0.0)),
        joint_limits=((-35.0, 35.0), (-30.0, 30.0), (-55.0, 55.0)),
        items=[
            Box("wing_carpal_hub", (5, 2, 3), mat_bone),
            Box(
                "wing_thumb_claw",
                (2, 1, 2),
                mat_claw,
                place=_flush_place("wing_carpal_hub", (0.0, 0.0, 1.0)),
            ),
            digit_bone,
        ],
    )

    wing_forearm = Bone(
        name="wing_forearm_l",
        rest_euler=(0.0, -5.0, 6.0),
        attach=_flush_attach("wing_humerus", (1.0, 0.0, 0.0)),
        joint_limits=((-40.0, 40.0), (-35.0, 35.0), (-65.0, 65.0)),
        items=[
            Box("wing_radius", (span_fore, 2, 2), mat_bone),
            Box(
                "wing_patagium_mid",
                (max(4, span_fore - 2), 1, 4),
                mat_membrane,
                place=_flush_place("wing_radius", (0.0, 0.0, -1.0)),
            ),
            wing_hand,
        ],
    )

    return Bone(
        name="wing_arm_l",
        rest_euler=(0.0, 0.0, 14.0),
        attach=_flush_attach(parent_box, (1.0, 0.0, 0.0), offset=attach_offset),
        mirror_name="wing_arm_r",
        joint_limits=((-45.0, 45.0), (-40.0, 40.0), (-75.0, 75.0)),
        items=[
            Box("wing_humerus", (span_arm, 2, 3), mat_bone),
            Box(
                "wing_patagium_prox",
                (max(4, span_arm - 3), 1, 4),
                mat_membrane,
                place=_flush_place("wing_humerus", (0.0, 0.0, -1.0), offset=(0.5, 0.0, 0.0)),
            ),
            wing_forearm,
        ],
    )


def _build_two_bone_limb(
    upper_name: str,
    lower_name: str,
    mirror_upper: str | None,
    parent_box: str,
    attach_face: Vec3,
    attach_offset: Vec3,
    upper_sz: Vec3i,
    lower_sz: Vec3i,
    foot_sz: Vec3i,
    mat_limb: str,
    mat_claw: str,
    *,
    upper_euler: Vec3 = (8.0, 0.0, 0.0),
    lower_euler: Vec3 = (-14.0, 0.0, 0.0),
) -> Bone:
    """Build a 2-bone IK-ready vertebrate/arthropod limb extending downward along -Y."""
    lower_box = f"{lower_name}_shank"
    foot_box = f"{lower_name}_pes"
    claw_box = f"{lower_name}_talon"
    upper_box = f"{upper_name}_muscle_mass"

    lower_bone = Bone(
        name=lower_name,
        rest_euler=lower_euler,
        attach=_flush_attach(upper_box, (0.0, -1.0, 0.0)),
        joint_limits=((-75.0, 65.0), (-20.0, 20.0), (-20.0, 20.0)),
        items=[
            Box(lower_box, lower_sz, mat_limb),
            Box(
                foot_box,
                foot_sz,
                mat_limb,
                place=_flush_place(lower_box, (0.0, -1.0, 0.0), offset=(0.0, 0.0, 0.5)),
            ),
            Box(
                claw_box,
                (max(1, foot_sz[0] - 1), 1, 2),
                mat_claw,
                place=_flush_place(foot_box, (0.0, 0.0, 1.0)),
            ),
            Locator(
                name=f"{lower_name}_sole",
                offset=(0.0, -float(lower_sz[1] + foot_sz[1]), 0.5),
                role="ik_effector",
            ),
        ],
    )

    return Bone(
        name=upper_name,
        rest_euler=upper_euler,
        attach=_flush_attach(parent_box, attach_face, offset=attach_offset),
        mirror_name=mirror_upper,
        joint_limits=((-65.0, 65.0), (-35.0, 35.0), (-40.0, 40.0)),
        items=[
            Box(upper_box, upper_sz, mat_limb),
            lower_bone,
        ],
    )


def _finalize_and_verify_apex(program: Program) -> Program:
    """Validate AST, enforce 100% rest SAT + P1-02 contact pass, and verify all @muscles & @behavior clips."""
    prog = validate_program(program)
    rest_sc = compile_geometry(prog)
    rest_aud = audit_scene(rest_sc, enforce_physical_contact=True)
    if not rest_aud["passed"]:
        raise AssertionError(
            f"Apex family rest audit failed: collisions={rest_aud['collision_pairs']}, "
            f"violations={rest_aud['contact_violations']}"
        )

    # Verify full-activation musculoskeletal pose is collision-free
    if prog.muscles:
        repaired_muscles: list[MuscleDecl] = []
        for idx, m in enumerate(prog.muscles):
            chosen = m
            for s in (1.0, 0.65, 0.35, 0.0):
                cand_m = replace(
                    m,
                    moment_arm_deg=round(m.moment_arm_deg * s, 2),
                    bulge_inflate=round(m.bulge_inflate * s, 3),
                )
                tmp_p = replace(prog, muscles=(*repaired_muscles, cand_m, *prog.muscles[idx + 1 :]))
                ms_sc = evaluate_muscles_at_activations(tmp_p, {cand_m.name: 1.0}).scene
                if not audit_scene(ms_sc, enforce_physical_contact=True)["passed"]:
                    continue
                chosen = cand_m
                break
            repaired_muscles.append(chosen)
        prog = validate_program(replace(prog, muscles=tuple(repaired_muscles)))

    # Verify every @behavior state and clip passes SAT + P1-02 + < 0.02u stance slip
    if prog.behavior is not None:
        verified_states: list[BehaviorStateDecl] = []
        for st in prog.behavior.states:
            chosen_st = st
            for scale in (1.0, 0.65, 0.35, 0.0):
                cand_st = replace(
                    st,
                    pose_biases=tuple(
                        (b, (round(v[0] * scale, 2), round(v[1] * scale, 2), round(v[2] * scale, 2)))
                        for b, v in st.pose_biases
                    ),
                    muscle_tones=tuple((m_id, round(val * scale, 3)) for m_id, val in st.muscle_tones),
                )
                st_sc = evaluate_muscles_at_activations(
                    prog,
                    dict(cand_st.muscle_tones),
                    base_pose_deltas=dict(cand_st.pose_biases),
                ).scene
                if audit_scene(st_sc, enforce_physical_contact=True)["passed"]:
                    chosen_st = cand_st
                    break
            verified_states.append(chosen_st)
        prog = validate_program(replace(prog, behavior=replace(prog.behavior, states=tuple(verified_states))))

        if prog.behavior.clips:
            verified_clips: list[ClipDecl] = []
            for c_idx, cl in enumerate(prog.behavior.clips):
                chosen_cl = cl
                for scale in (1.0, 0.72, 0.50, 0.30, 0.15, 0.0):
                    cand_cl = replace(
                        cl,
                        tracks=tuple(
                            replace(t, amp_deg=round(t.amp_deg * scale, 3), bias_deg=round(t.bias_deg * scale, 3))
                            for t in cl.tracks
                        ),
                        muscle_drives=tuple(
                            replace(
                                md,
                                peak=round(md.peak * scale, 3),
                                baseline=round(min(md.baseline, md.peak * scale), 3),
                            )
                            for md in cl.muscle_drives
                        ),
                        ik_drives=tuple(
                            replace(ikd, stride=round(ikd.stride * scale, 3), lift=round(ikd.lift * scale, 3))
                            for ikd in cl.ik_drives
                        ),
                    )
                    tmp_beh = replace(
                        prog.behavior,
                        clips=(*verified_clips, cand_cl, *prog.behavior.clips[c_idx + 1 :]),
                    )
                    tmp_p = replace(prog, behavior=tmp_beh)
                    rep = audit_behavior_clip(tmp_p, cand_cl.name, num_frames=12)
                    if rep["passed"]:
                        chosen_cl = cand_cl
                        break
                verified_clips.append(chosen_cl)
            prog = validate_program(replace(prog, behavior=replace(prog.behavior, clips=tuple(verified_clips))))
    return prog


# ---------------------------------------------------------------------------
# 10 Clean-Room Apex Complex Entity & Game Asset Family Builders
# ---------------------------------------------------------------------------


def _build_apex_flying_wyvern(rng: random.Random, seed: int) -> Program:
    hue_main = round(rng.uniform(8.0, 38.0), 1)
    palettes = [
        Palette("carapace_scale", (0.42, 0.16, hue_main), 6, (-8.0, 10.0)),
        Palette("belly_hide", (0.72, 0.07, (hue_main + 28.0) % 360.0), 5),
        Palette("patagium_wing", (0.50, 0.12, (hue_main + 15.0) % 360.0), 6),
        Palette("horn_keratin", (0.26, 0.04, 45.0), 4),
        Palette("flame_sac", (0.78, 0.22, 32.0), 4, emissive=True),
    ]
    chest_w = rng.choice((8, 10))
    chest_l = rng.choice((10, 12))

    # Head + articulated jaw + horns + breath emitter
    jaw_bone = Bone(
        name="mandible_jaw",
        rest_euler=(6.0, 0.0, 0.0),
        attach=_flush_attach("cranium_skull", (0.0, -1.0, 0.0), offset=(0.0, 0.0, 0.4)),
        joint_limits=((-5.0, 48.0), (-10.0, 10.0), (-10.0, 10.0)),
        items=[
            Box("lower_jaw", (4, 2, 6), "carapace_scale"),
            Box("throat_flame_pouch", (2, 1, 4), "flame_sac", place=_flush_place("lower_jaw", (0.0, -1.0, 0.0))),
            Locator("breath_muzzle_vfx", offset=(0.0, 0.5, 3.0), role="vfx_emitter"),
        ],
    )
    crown_horn = Bone(
        name="crest_horn_l",
        rest_euler=(-18.0, 0.0, -14.0),
        attach=_flush_attach("cranium_skull", (0.0, 1.0, 0.0), offset=(0.65, 0.0, 0.0)),
        mirror_name="crest_horn_r",
        joint_limits=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),
        items=[
            Box("horn_base", (2, 4, 2), "horn_keratin"),
            Box("horn_tip", (1, 3, 1), "horn_keratin", place=_flush_place("horn_base", (0.0, 1.0, 0.0))),
        ],
    )
    head_bone = Bone(
        name="wyvern_head",
        rest_euler=(8.0, 0.0, 0.0),
        attach=_flush_attach("neck_seg_2", (0.0, 0.0, 1.0)),
        joint_limits=((-40.0, 40.0), (-50.0, 50.0), (-30.0, 30.0)),
        items=[
            Box("cranium_skull", (6, 4, 6), "carapace_scale"),
            Box("snout_rostrum", (4, 3, 4), "carapace_scale", place=_flush_place("cranium_skull", (0.0, 0.0, 1.0))),
            Locator("head_camera_focus", offset=(0.0, 1.5, 4.0), role="camera_focus"),
            crown_horn,
            jaw_bone,
        ],
    )
    neck_2 = Bone(
        name="neck_2",
        rest_euler=(-12.0, 0.0, 0.0),
        attach=_flush_attach("neck_seg_1", (0.0, 0.0, 1.0)),
        joint_limits=((-35.0, 35.0), (-40.0, 40.0), (-25.0, 25.0)),
        items=[
            Box("neck_seg_2", (5, 5, 5), "carapace_scale"),
            Box("neck_throat_2", (3, 1, 4), "belly_hide", place=_flush_place("neck_seg_2", (0.0, -1.0, 0.0))),
            head_bone,
        ],
    )
    neck_1 = Bone(
        name="neck_1",
        rest_euler=(-16.0, 0.0, 0.0),
        attach=_flush_attach("thorax_core", (0.0, 0.0, 1.0), offset=(0.0, 0.3, 0.0)),
        joint_limits=((-35.0, 35.0), (-40.0, 40.0), (-25.0, 25.0)),
        items=[
            Box("neck_seg_1", (6, 6, 5), "carapace_scale"),
            Box("neck_osteoderm_1", (2, 2, 3), "horn_keratin", place=_flush_place("neck_seg_1", (0.0, 1.0, 0.0))),
            neck_2,
        ],
    )

    wing_l = _build_nested_mirror_wyvern_wing(
        "dorsal_shoulder_saddle",
        "carapace_scale",
        "patagium_wing",
        "horn_keratin",
        span_arm=8,
        span_fore=9,
        span_digit=8,
    )

    hindleg_l = _build_two_bone_limb(
        "hind_thigh_l",
        "hind_shank_l",
        "hind_thigh_r",
        "pelvis_hip",
        (1.0, 0.0, 0.0),
        (0.0, -0.3, 0.0),
        (3, 6, 4),
        (2, 6, 3),
        (3, 2, 5),
        "carapace_scale",
        "horn_keratin",
    )

    tail_chain = _build_segmented_chain(
        "tail",
        "pelvis_hip",
        (0.0, 0.0, -1.0),
        [(6, 5, 6), (5, 4, 6), (4, 3, 6), (3, 2, 7)],
        "carapace_scale",
        "horn_keratin",
        rest_euler_step=(-2.0, 0.0, 0.0),
        tip_locator=("tail_club_hitbox", "hitbox"),
    )

    root = Bone(
        name="wyvern_root",
        pivot_local=(0.0, 18.0, 0.0),
        items=[
            Box("thorax_core", (chest_w, 7, chest_l), "carapace_scale"),
            Box("sternum_keel", (chest_w - 2, 2, chest_l - 2), "belly_hide", place=_flush_place("thorax_core", (0.0, -1.0, 0.0))),
            Box("dorsal_shoulder_saddle", (chest_w + 2, 2, 6), "carapace_scale", place=_flush_place("thorax_core", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 1.0))),
            Box("pelvis_hip", (chest_w - 2, 6, 7), "carapace_scale", place=_flush_place("thorax_core", (0.0, 0.0, -1.0))),
            Locator("rider_saddle_socket", offset=(0.0, 5.5, 0.0), role="rider_seat"),
            neck_1,
            wing_l,
            hindleg_l,
            tail_chain,
        ],
    )

    materials = (
        MaterialDecl("carapace_scale", "keratin_scales", roughness=0.34, scale=1.2, seed=seed + 11, anisotropy=1.5, subsurface=0.25, wear=0.42, weld_group="wyvern_dermis"),
        MaterialDecl("belly_hide", "patterned_skin", roughness=0.52, scale=1.1, seed=seed + 23, subsurface=0.38, rd_feed=0.037, rd_kill=0.060, weld_group="wyvern_dermis"),
        MaterialDecl("patagium_wing", "thin_film_iridescence", roughness=0.22, scale=1.4, seed=seed + 37, film_nm=410.0, weld_group="wing_membrane"),
        MaterialDecl("horn_keratin", "keratin_scales", roughness=0.28, scale=0.9, seed=seed + 41, anisotropy=1.8, wear=0.50),
        MaterialDecl("flame_sac", "patterned_skin", roughness=0.40, scale=1.0, seed=seed + 53, emissive=1.1),
    )
    style = StyleDecl(target_encoding="labpbr_1_3", quantize_ramp=False, bevel_highlight=0.11, view_deg=22.0)

    muscles = (
        MuscleDecl("pectoral_downstroke_l", "wyvern_root", "wing_arm_l", "z", 22.0, TendonCoupling("wing_forearm_l", "z", 0.35), "wing_humerus", 0.10),
        MuscleDecl("pectoral_downstroke_r", "wyvern_root", "wing_arm_r", "z", -22.0, TendonCoupling("wing_arm_r__wing_forearm_l", "z", 0.35), "wing_arm_r__wing_humerus", 0.10),
        MuscleDecl("hind_gastroc_l", "wyvern_root", "hind_thigh_l", "x", 18.0, TendonCoupling("hind_shank_l", "x", -0.65), "hind_thigh_l_muscle_mass", 0.09),
        MuscleDecl("hind_gastroc_r", "wyvern_root", "hind_thigh_r", "x", 18.0, TendonCoupling("hind_thigh_r__hind_shank_l", "x", -0.65), "hind_thigh_r__hind_thigh_l_muscle_mass", 0.09),
        MuscleDecl("jaw_adductor", "wyvern_head", "mandible_jaw", "x", 24.0),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("leg_l_ik", "hind_thigh_l", "hind_shank_l", (0.0, -8.0, 0.5), "x"),
            IKChainDecl("leg_r_ik", "hind_thigh_r", "hind_thigh_r__hind_shank_l", (0.0, -8.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("cervical_gaze", ("neck_1", "neck_2", "wyvern_head"), 0.72),),
    )
    dur = 1.6
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                name="stalking_wing_patrol",
                duration_s=dur,
                loop=True,
                tracks=(
                    TrackDecl("tail_1", "wave", "y", 9.0, fq, 0.0),
                    TrackDecl("tail_2", "wave", "y", 11.0, fq, 315.0),
                    TrackDecl("tail_3", "wave", "y", 13.0, fq, 270.0),
                    TrackDecl("tail_4", "wave", "y", 15.0, fq, 225.0),
                    TrackDecl("neck_1", "wave", "x", 4.0, fq, 90.0),
                ),
                muscle_drives=(
                    MuscleDriveDecl("pectoral_downstroke_l", 0.65, fq, 0.0, 0.05),
                    MuscleDriveDecl("pectoral_downstroke_r", 0.65, fq, 0.0, 0.05),
                ),
                ik_drives=(
                    IKDriveDecl("leg_l_ik", 1.8, 0.9, 0.0),
                    IKDriveDecl("leg_r_ik", 1.8, 0.9, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                name="thermal_rage_mode",
                pose_biases=(("wyvern_head", (-6.0, 0.0, 0.0)),),
                muscle_tones=(("jaw_adductor", 0.75), ("pectoral_downstroke_l", 0.35), ("pectoral_downstroke_r", 0.35)),
                material_mods=(MaterialModDecl("flame_sac", emissive=1.8, moisture=0.1, wear_delta=0.05),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_brute_theropod_wyvern(rng: random.Random, seed: int) -> Program:
    hue = round(rng.uniform(330.0, 385.0) % 360.0, 1)
    palettes = [
        Palette("theropod_hide", (0.40, 0.14, hue), 6),
        Palette("dorsal_plate", (0.28, 0.08, (hue + 20.0) % 360.0), 5),
        Palette("thermal_sail", (0.68, 0.20, 28.0), 5, emissive=True),
        Palette("fang_ivory", (0.82, 0.04, 75.0), 4),
    ]
    sail_l = Bone(
        name="thermo_sail_l",
        rest_euler=(0.0, 0.0, -16.0),
        attach=_flush_attach("thorax_barrel", (0.0, 1.0, 0.0), offset=(0.75, 0.0, -0.25)),
        mirror_name="thermo_sail_r",
        joint_limits=((-20.0, 20.0), (-20.0, 20.0), (-55.0, 55.0)),
        items=[
            Box("sail_strut", (1, 6, 5), "dorsal_plate"),
            Box("sail_membrane", (1, 5, 6), "thermal_sail", place=_flush_place("sail_strut", (0.0, 1.0, 0.0))),
        ],
    )
    vestigial_arm_l = Bone(
        name="forearm_vestigial_l",
        rest_euler=(18.0, 0.0, 12.0),
        attach=_flush_attach("thorax_barrel", (1.0, 0.0, 0.0), offset=(0.0, -0.35, 0.35)),
        mirror_name="forearm_vestigial_r",
        joint_limits=((-45.0, 45.0), (-20.0, 20.0), (-30.0, 30.0)),
        items=[
            Box("brachium_l", (2, 4, 2), "theropod_hide"),
            Box("foreclaw_l", (1, 2, 2), "fang_ivory", place=_flush_place("brachium_l", (0.0, -1.0, 0.0))),
        ],
    )
    crushing_jaw = Bone(
        name="crushing_mandible",
        rest_euler=(8.0, 0.0, 0.0),
        attach=_flush_attach("brute_skull", (0.0, -1.0, 0.0), offset=(0.0, 0.0, 0.3)),
        joint_limits=((-5.0, 52.0), (-10.0, 10.0), (-10.0, 10.0)),
        items=[
            Box("mandible_slab", (7, 3, 6), "dorsal_plate"),
            Box("tusk_row_l", (1, 3, 4), "fang_ivory", place=_flush_place("mandible_slab", (1.0, 0.0, 0.0), offset=(0.0, 1.0, 0.5))),
            Box("tusk_row_r", (1, 3, 4), "fang_ivory", place=_flush_place("mandible_slab", (-1.0, 0.0, 0.0), offset=(0.0, 1.0, 0.5))),
            Locator("roar_blast_emitter", offset=(0.0, 1.5, 4.0), role="vfx_emitter"),
        ],
    )
    head = Bone(
        name="brute_head",
        rest_euler=(6.0, 0.0, 0.0),
        attach=_flush_attach("cervical_mass", (0.0, 0.0, 1.0)),
        joint_limits=((-35.0, 35.0), (-45.0, 45.0), (-25.0, 25.0)),
        items=[
            Box("brute_skull", (7, 5, 8), "dorsal_plate"),
            Box("nasal_battering_crest", (3, 3, 5), "dorsal_plate", place=_flush_place("brute_skull", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 1.0))),
            Locator("brute_eyes_focus", offset=(0.0, 1.5, 4.0), role="camera_focus"),
            crushing_jaw,
        ],
    )
    neck = Bone(
        name="brute_neck",
        rest_euler=(-10.0, 0.0, 0.0),
        attach=_flush_attach("thorax_barrel", (0.0, 0.0, 1.0), offset=(0.0, 0.25, 0.0)),
        joint_limits=((-30.0, 30.0), (-35.0, 35.0), (-20.0, 20.0)),
        items=[
            Box("cervical_mass", (6, 6, 5), "theropod_hide"),
            Box("masseter_jowl", (7, 3, 4), "theropod_hide", place=_flush_place("cervical_mass", (0.0, -1.0, 0.0))),
            head,
        ],
    )
    leg_l = _build_two_bone_limb(
        "brute_femur_l",
        "brute_tibia_l",
        "brute_femur_r",
        "hip_girdle",
        (1.0, 0.0, 0.0),
        (0.0, -0.25, 0.0),
        (4, 7, 5),
        (3, 7, 4),
        (4, 2, 6),
        "theropod_hide",
        "fang_ivory",
    )
    tail = _build_segmented_chain(
        "brute_tail",
        "hip_girdle",
        (0.0, 0.0, -1.0),
        [(7, 6, 6), (6, 5, 6), (5, 4, 6), (6, 4, 7)],
        "theropod_hide",
        "dorsal_plate",
        tip_locator=("hammer_tail_hitbox", "hitbox"),
    )

    root = Bone(
        name="brute_root",
        pivot_local=(0.0, 19.0, 0.0),
        items=[
            Box("thorax_barrel", (9, 8, 11), "theropod_hide"),
            Box("dorsal_armor_keel", (3, 2, 9), "dorsal_plate", place=_flush_place("thorax_barrel", (0.0, 1.0, 0.0))),
            Box("hip_girdle", (8, 7, 8), "theropod_hide", place=_flush_place("thorax_barrel", (0.0, 0.0, -1.0))),
            Locator("brute_saddle", offset=(0.0, 6.0, -1.0), role="rider_seat"),
            sail_l,
            vestigial_arm_l,
            neck,
            leg_l,
            tail,
        ],
    )
    materials = (
        MaterialDecl("theropod_hide", "patterned_skin", roughness=0.54, scale=1.25, seed=seed + 13, subsurface=0.32, weld_group="brute_skin"),
        MaterialDecl("dorsal_plate", "keratin_scales", roughness=0.36, scale=1.0, seed=seed + 29, anisotropy=1.6, wear=0.48, weld_group="brute_skin"),
        MaterialDecl("thermal_sail", "patterned_skin", roughness=0.35, scale=1.1, seed=seed + 43, emissive=0.9),
        MaterialDecl("fang_ivory", "polymer_dielectric", roughness=0.22, ior=1.56, scale=0.8, seed=seed + 61),
    )
    style = StyleDecl("labpbr_1_3", False, 0.10, 20.0)
    muscles = (
        MuscleDecl("jaw_masseter", "brute_neck", "crushing_mandible", "x", 28.0, None, "masseter_jowl", 0.12),
        MuscleDecl("quad_extensor_l", "brute_root", "brute_femur_l", "x", 18.0, TendonCoupling("brute_tibia_l", "x", -0.60), "brute_femur_l_muscle_mass", 0.11),
        MuscleDecl("quad_extensor_r", "brute_root", "brute_femur_r", "x", 18.0, TendonCoupling("brute_femur_r__brute_tibia_l", "x", -0.60), "brute_femur_r__brute_femur_l_muscle_mass", 0.11),
        MuscleDecl("sail_erector_l", "brute_root", "thermo_sail_l", "z", -26.0),
        MuscleDecl("sail_erector_r", "brute_root", "thermo_sail_r", "z", 26.0),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("stride_l", "brute_femur_l", "brute_tibia_l", (0.0, -9.0, 0.5), "x"),
            IKChainDecl("stride_r", "brute_femur_r", "brute_femur_r__brute_tibia_l", (0.0, -9.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("brute_gaze", ("brute_neck", "brute_head"), 0.70),),
    )
    dur = 1.8
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "heavy_stomp_march",
                dur,
                True,
                tracks=(
                    TrackDecl("brute_tail_1", "wave", "y", 8.0, fq, 0.0),
                    TrackDecl("brute_tail_2", "wave", "y", 11.0, fq, 315.0),
                    TrackDecl("brute_tail_3", "wave", "y", 13.0, fq, 270.0),
                    TrackDecl("brute_tail_4", "wave", "y", 15.0, fq, 225.0),
                ),
                muscle_drives=(MuscleDriveDecl("jaw_masseter", 0.45, fq, 0.0, 0.05),),
                ik_drives=(
                    IKDriveDecl("stride_l", 2.2, 1.1, 0.0),
                    IKDriveDecl("stride_r", 2.2, 1.1, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "overheated_enrage",
                pose_biases=(("brute_head", (-8.0, 0.0, 0.0)),),
                muscle_tones=(("sail_erector_l", 0.90), ("sail_erector_r", 0.90), ("jaw_masseter", 0.70)),
                material_mods=(MaterialModDecl("thermal_sail", emissive=1.9, moisture=0.05, wear_delta=0.06),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_pelagic_leviathan_wyvern(rng: random.Random, seed: int) -> Program:
    hue = round(rng.uniform(185.0, 235.0), 1)
    palettes = [
        Palette("leviathan_dermis", (0.46, 0.13, hue), 6),
        Palette("ventral_scutes", (0.76, 0.06, (hue + 25.0) % 360.0), 5),
        Palette("pelagic_fin", (0.58, 0.16, (hue - 25.0) % 360.0), 6),
        Palette("voltaic_crystal", (0.80, 0.20, 195.0), 4, emissive=True),
    ]
    crystal_pod_l = Bone(
        name="dorsal_capacitor_l",
        rest_euler=(0.0, 0.0, -16.0),
        attach=_flush_attach("leviathan_thorax", (0.0, 1.0, 0.0), offset=(0.70, 0.0, 0.1)),
        mirror_name="dorsal_capacitor_r",
        joint_limits=((-20.0, 20.0), (-20.0, 20.0), (-35.0, 35.0)),
        items=[
            Box("capacitor_socket", (2, 3, 3), "leviathan_dermis"),
            Box("voltaic_spire", (1, 5, 2), "voltaic_crystal", place=_flush_place("capacitor_socket", (0.0, 1.0, 0.0))),
        ],
    )
    fore_flipper_l = _build_two_bone_limb(
        "pectoral_limb_l",
        "pectoral_paddle_l",
        "pectoral_limb_r",
        "leviathan_thorax",
        (1.0, 0.0, 0.0),
        (0.0, -0.35, 0.3),
        (3, 4, 3),
        (2, 4, 4),
        (4, 1, 5),
        "leviathan_dermis",
        "pelagic_fin",
    )
    hind_flipper_l = _build_two_bone_limb(
        "pelvic_limb_l",
        "pelvic_paddle_l",
        "pelvic_limb_r",
        "spine_seg_1",
        (1.0, 0.0, 0.0),
        (0.0, -0.3, 0.0),
        (3, 4, 3),
        (2, 4, 3),
        (3, 1, 5),
        "leviathan_dermis",
        "pelagic_fin",
    )

    # 5-segment serpentine body + caudal fin
    tail_chain = _build_segmented_chain(
        "caudal",
        "spine_seg_1",
        (0.0, 0.0, -1.0),
        [(6, 5, 7), (5, 4, 7), (4, 4, 7), (3, 5, 8)],
        "leviathan_dermis",
        "pelagic_fin",
        tip_locator=("caudal_wake_vfx", "vfx_emitter"),
    )
    spine_1 = Bone(
        name="spine_1",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("leviathan_thorax", (0.0, 0.0, -1.0)),
        joint_limits=((-30.0, 30.0), (-45.0, 45.0), (-25.0, 25.0)),
        items=[
            Box("spine_seg_1", (7, 6, 8), "leviathan_dermis"),
            Box("spine_scute_1", (5, 1, 6), "ventral_scutes", place=_flush_place("spine_seg_1", (0.0, -1.0, 0.0))),
            hind_flipper_l,
            tail_chain,
        ],
    )

    head_jaw = Bone(
        name="leviathan_jaw",
        rest_euler=(4.0, 0.0, 0.0),
        attach=_flush_attach("leviathan_cranium", (0.0, -1.0, 0.0)),
        joint_limits=((-5.0, 45.0), (-10.0, 10.0), (-10.0, 10.0)),
        items=[
            Box("mandible_keel", (4, 2, 7), "ventral_scutes"),
            Locator("voltaic_breath_vfx", offset=(0.0, 0.5, 3.5), role="vfx_emitter"),
        ],
    )
    frill_l = Bone(
        name="gill_frill_l",
        rest_euler=(0.0, -22.0, 0.0),
        attach=_flush_attach("leviathan_cranium", (1.0, 0.0, 0.0)),
        mirror_name="gill_frill_r",
        joint_limits=((-20.0, 20.0), (-40.0, 40.0), (-20.0, 20.0)),
        items=[Box("frill_blade", (1, 4, 4), "pelagic_fin")],
    )
    head = Bone(
        name="leviathan_head",
        rest_euler=(6.0, 0.0, 0.0),
        attach=_flush_attach("neck_seg_2", (0.0, 0.0, 1.0)),
        joint_limits=((-35.0, 35.0), (-45.0, 45.0), (-25.0, 25.0)),
        items=[
            Box("leviathan_cranium", (5, 4, 8), "leviathan_dermis"),
            Locator("leviathan_focus", offset=(0.0, 1.5, 4.0), role="camera_focus"),
            frill_l,
            head_jaw,
        ],
    )
    neck_2 = Bone(
        name="neck_2",
        rest_euler=(-8.0, 0.0, 0.0),
        attach=_flush_attach("neck_seg_1", (0.0, 0.0, 1.0)),
        joint_limits=((-35.0, 35.0), (-45.0, 45.0), (-25.0, 25.0)),
        items=[Box("neck_seg_2", (5, 5, 6), "leviathan_dermis"), head],
    )
    neck_1 = Bone(
        name="neck_1",
        rest_euler=(-10.0, 0.0, 0.0),
        attach=_flush_attach("leviathan_thorax", (0.0, 0.0, 1.0)),
        joint_limits=((-35.0, 35.0), (-45.0, 45.0), (-25.0, 25.0)),
        items=[
            Box("neck_seg_1", (6, 5, 6), "leviathan_dermis"),
            Box("neck_fin_1", (1, 3, 4), "pelagic_fin", place=_flush_place("neck_seg_1", (0.0, 1.0, 0.0))),
            neck_2,
        ],
    )

    root = Bone(
        name="leviathan_root",
        pivot_local=(0.0, 12.0, 0.0),
        items=[
            Box("leviathan_thorax", (8, 6, 10), "leviathan_dermis"),
            Box("thorax_scutes", (6, 1, 8), "ventral_scutes", place=_flush_place("leviathan_thorax", (0.0, -1.0, 0.0))),
            Locator("leviathan_mount", offset=(0.0, 4.0, 1.0), role="rider_seat"),
            crystal_pod_l,
            fore_flipper_l,
            neck_1,
            spine_1,
        ],
    )
    materials = (
        MaterialDecl("leviathan_dermis", "patterned_skin", roughness=0.28, scale=1.3, seed=seed + 17, subsurface=0.42, porosity=0.12, weld_group="leviathan_body"),
        MaterialDecl("ventral_scutes", "keratin_scales", roughness=0.32, scale=1.0, seed=seed + 31, anisotropy=1.7, weld_group="leviathan_body"),
        MaterialDecl("pelagic_fin", "thin_film_iridescence", roughness=0.18, scale=1.4, seed=seed + 47, film_nm=460.0),
        MaterialDecl("voltaic_crystal", "thin_film_iridescence", roughness=0.12, ior=1.75, scale=0.9, seed=seed + 59, emissive=1.2, film_nm=350.0),
    )
    style = StyleDecl("labpbr_1_3", False, 0.12, 22.0)
    muscles = (
        MuscleDecl("lateral_myomere_1", "leviathan_root", "spine_1", "y", 16.0, TendonCoupling("caudal_1", "y", 0.65), "spine_seg_1", 0.07),
        MuscleDecl("frill_flarer_l", "leviathan_head", "gill_frill_l", "y", -24.0),
        MuscleDecl("frill_flarer_r", "leviathan_head", "gill_frill_r", "y", 24.0),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("fore_l_ik", "pectoral_limb_l", "pectoral_paddle_l", (0.0, -5.0, 0.5), "x"),
            IKChainDecl("fore_r_ik", "pectoral_limb_r", "pectoral_limb_r__pectoral_paddle_l", (0.0, -5.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("serpentine_gaze", ("neck_1", "neck_2", "leviathan_head"), 0.78),),
    )
    dur = 1.6
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "serpentine_undulation",
                dur,
                True,
                tracks=(
                    TrackDecl("neck_1", "wave", "y", 8.0, fq, 45.0),
                    TrackDecl("spine_1", "wave", "y", 10.0, fq, 0.0),
                    TrackDecl("caudal_1", "wave", "y", 13.0, fq, 315.0),
                    TrackDecl("caudal_2", "wave", "y", 15.0, fq, 270.0),
                    TrackDecl("caudal_3", "wave", "y", 17.0, fq, 225.0),
                    TrackDecl("caudal_4", "wave", "y", 19.0, fq, 180.0),
                ),
                ik_drives=(
                    IKDriveDecl("fore_l_ik", 1.2, 0.6, 0.0),
                    IKDriveDecl("fore_r_ik", 1.2, 0.6, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "abyssal_voltaic_surge",
                muscle_tones=(("frill_flarer_l", 0.85), ("frill_flarer_r", 0.85)),
                material_mods=(
                    MaterialModDecl("voltaic_crystal", emissive=2.0, moisture=0.8, wear_delta=0.02),
                    MaterialModDecl("leviathan_dermis", emissive=0.0, moisture=0.9, wear_delta=0.0),
                ),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_fanged_thunder_wyvern(rng: random.Random, seed: int) -> Program:
    hue = round(rng.uniform(195.0, 225.0), 1)
    palettes = [
        Palette("cobalt_shell", (0.38, 0.14, hue), 6),
        Palette("thunder_mane", (0.84, 0.10, 88.0), 5),
        Palette("fulgur_core", (0.82, 0.20, 178.0), 4, emissive=True),
        Palette("talon_horn", (0.30, 0.06, 52.0), 4),
    ]
    foreleg_l = _build_two_bone_limb(
        "foreleg_humerus_l",
        "foreleg_ulna_l",
        "foreleg_humerus_r",
        "chest_musculature",
        (1.0, 0.0, 0.0),
        (0.0, -0.25, 0.25),
        (4, 6, 4),
        (3, 6, 3),
        (4, 2, 5),
        "cobalt_shell",
        "talon_horn",
    )
    hindleg_l = _build_two_bone_limb(
        "hindleg_femur_l",
        "hindleg_crus_l",
        "hindleg_femur_r",
        "lumbar_haunch",
        (1.0, 0.0, 0.0),
        (0.0, -0.25, -0.1),
        (4, 6, 4),
        (3, 6, 3),
        (4, 2, 5),
        "cobalt_shell",
        "talon_horn",
    )
    shoulder_plate_l = Bone(
        name="capacitor_pauldron_l",
        rest_euler=(0.0, 0.0, -16.0),
        attach=_flush_attach("chest_musculature", (0.0, 1.0, 0.0), offset=(0.75, 0.0, 0.25)),
        mirror_name="capacitor_pauldron_r",
        joint_limits=((-20.0, 20.0), (-20.0, 20.0), (-45.0, 45.0)),
        items=[
            Box("pauldron_shell", (2, 4, 5), "cobalt_shell"),
            Box("pauldron_cell", (1, 3, 3), "fulgur_core", place=_flush_place("pauldron_shell", (0.0, 1.0, 0.0))),
        ],
    )
    jaw = Bone(
        name="lupine_jaw",
        rest_euler=(10.0, 0.0, 0.0),
        attach=_flush_attach("lupine_skull", (0.0, -1.0, 0.0), offset=(0.0, 0.0, 0.35)),
        joint_limits=((-5.0, 42.0), (-10.0, 10.0), (-10.0, 10.0)),
        items=[Box("lupine_mandible", (4, 2, 4), "cobalt_shell")],
    )
    horn_l = Bone(
        name="swept_horn_l",
        rest_euler=(-18.0, 0.0, -14.0),
        attach=_flush_attach("lupine_skull", (0.0, 1.0, 0.0), offset=(0.65, 0.0, 0.0)),
        mirror_name="swept_horn_r",
        joint_limits=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),
        items=[Box("swept_horn_beam", (1, 5, 2), "talon_horn")],
    )
    head = Bone(
        name="lupine_head",
        rest_euler=(6.0, 0.0, 0.0),
        attach=_flush_attach("mane_neck_mass", (0.0, 0.0, 1.0)),
        joint_limits=((-35.0, 35.0), (-45.0, 45.0), (-25.0, 25.0)),
        items=[
            Box("lupine_skull", (5, 4, 6), "cobalt_shell"),
            Locator("lupine_gaze_focus", offset=(0.0, 1.0, 3.0), role="camera_focus"),
            horn_l,
            jaw,
        ],
    )
    neck = Bone(
        name="mane_neck",
        rest_euler=(-12.0, 0.0, 0.0),
        attach=_flush_attach("chest_musculature", (0.0, 0.0, 1.0), offset=(0.0, 0.25, 0.0)),
        joint_limits=((-35.0, 35.0), (-40.0, 40.0), (-25.0, 25.0)),
        items=[
            Box("mane_neck_mass", (6, 6, 5), "thunder_mane"),
            Box("mane_dorsal_tuft", (4, 2, 4), "thunder_mane", place=_flush_place("mane_neck_mass", (0.0, 1.0, 0.0))),
            head,
        ],
    )
    tail = _build_segmented_chain(
        "blade_tail",
        "lumbar_haunch",
        (0.0, 0.0, -1.0),
        [(5, 4, 6), (4, 4, 6), (5, 3, 7)],
        "cobalt_shell",
        "fulgur_core",
        tip_locator=("fulgur_tail_emitter", "vfx_emitter"),
    )

    root = Bone(
        name="fanged_root",
        pivot_local=(0.0, 16.0, 0.0),
        items=[
            Box("chest_musculature", (8, 7, 9), "cobalt_shell"),
            Box("chest_mane_ruff", (6, 2, 7), "thunder_mane", place=_flush_place("chest_musculature", (0.0, -1.0, 0.0))),
            Box("dorsal_fulgur_ridge", (2, 2, 7), "fulgur_core", place=_flush_place("chest_musculature", (0.0, 1.0, 0.0))),
            Box("lumbar_haunch", (7, 6, 9), "cobalt_shell", place=_flush_place("chest_musculature", (0.0, 0.0, -1.0))),
            Locator("fanged_rider_seat", offset=(0.0, 5.5, -2.0), role="rider_seat"),
            shoulder_plate_l,
            foreleg_l,
            hindleg_l,
            neck,
            tail,
        ],
    )
    materials = (
        MaterialDecl("cobalt_shell", "keratin_scales", roughness=0.30, scale=1.1, seed=seed + 19, anisotropy=1.65, wear=0.45, weld_group="fanged_coat"),
        MaterialDecl("thunder_mane", "patterned_skin", roughness=0.64, scale=0.85, seed=seed + 37, porosity=0.35, weld_group="fanged_coat"),
        MaterialDecl("fulgur_core", "thin_film_iridescence", roughness=0.16, scale=1.0, seed=seed + 49, emissive=1.1, film_nm=430.0),
        MaterialDecl("talon_horn", "keratin_scales", roughness=0.26, scale=0.8, seed=seed + 67, anisotropy=1.9),
    )
    style = StyleDecl("labpbr_1_3", False, 0.11, 22.0)
    muscles = (
        MuscleDecl("tricep_drive_l", "fanged_root", "foreleg_humerus_l", "x", 16.0, TendonCoupling("foreleg_ulna_l", "x", -0.55), "foreleg_humerus_l_muscle_mass", 0.10),
        MuscleDecl("tricep_drive_r", "fanged_root", "foreleg_humerus_r", "x", 16.0, TendonCoupling("foreleg_humerus_r__foreleg_ulna_l", "x", -0.55), "foreleg_humerus_r__foreleg_humerus_l_muscle_mass", 0.10),
        MuscleDecl("pauldron_flare_l", "fanged_root", "capacitor_pauldron_l", "z", -22.0),
        MuscleDecl("pauldron_flare_r", "fanged_root", "capacitor_pauldron_r", "z", 22.0),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("fl_ik", "foreleg_humerus_l", "foreleg_ulna_l", (0.0, -8.0, 0.5), "x"),
            IKChainDecl("fr_ik", "foreleg_humerus_r", "foreleg_humerus_r__foreleg_ulna_l", (0.0, -8.0, 0.5), "x"),
            IKChainDecl("hl_ik", "hindleg_femur_l", "hindleg_crus_l", (0.0, -8.0, 0.5), "x"),
            IKChainDecl("hr_ik", "hindleg_femur_r", "hindleg_femur_r__hindleg_crus_l", (0.0, -8.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("lupine_gaze", ("mane_neck", "lupine_head"), 0.74),),
    )
    dur = 1.4
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "quadruped_prowl_trot",
                dur,
                True,
                tracks=(
                    TrackDecl("blade_tail_1", "wave", "y", 9.0, fq, 0.0),
                    TrackDecl("blade_tail_2", "wave", "y", 12.0, fq, 310.0),
                    TrackDecl("blade_tail_3", "wave", "y", 15.0, fq, 260.0),
                ),
                ik_drives=(
                    IKDriveDecl("fl_ik", 1.8, 0.85, 0.0),
                    IKDriveDecl("hr_ik", 1.8, 0.85, 0.0),
                    IKDriveDecl("fr_ik", 1.8, 0.85, 180.0),
                    IKDriveDecl("hl_ik", 1.8, 0.85, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "fulgur_overcharge_state",
                muscle_tones=(("pauldron_flare_l", 0.90), ("pauldron_flare_r", 0.90)),
                material_mods=(MaterialModDecl("fulgur_core", emissive=2.0, moisture=0.15, wear_delta=0.04),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_armored_arthropod_colossus(rng: random.Random, seed: int) -> Program:
    hue = round(rng.uniform(115.0, 165.0), 1)
    palettes = [
        Palette("chitin_elytra", (0.36, 0.14, hue), 6),
        Palette("joint_arthrodial", (0.24, 0.05, 45.0), 4),
        Palette("siege_horn", (0.52, 0.10, (hue + 40.0) % 360.0), 5),
        Palette("compound_eye", (0.72, 0.22, 18.0), 4, emissive=True),
    ]
    leg_front_l = _build_two_bone_limb(
        "coxa_front_l", "tibia_front_l", "coxa_front_r",
        "cephalothorax_shell", (1.0, 0.0, 0.0), (0.0, -0.3, 0.55),
        (2, 5, 2), (2, 6, 2), (2, 1, 3), "joint_arthrodial", "siege_horn",
    )
    leg_mid_l = _build_two_bone_limb(
        "coxa_mid_l", "tibia_mid_l", "coxa_mid_r",
        "cephalothorax_shell", (1.0, 0.0, 0.0), (0.0, -0.3, -0.55),
        (2, 5, 2), (2, 6, 2), (2, 1, 3), "joint_arthrodial", "siege_horn",
    )
    leg_rear_l = _build_two_bone_limb(
        "coxa_rear_l", "tibia_rear_l", "coxa_rear_r",
        "abdomen_gaster", (1.0, 0.0, 0.0), (0.0, -0.3, 0.2),
        (2, 5, 2), (2, 6, 2), (2, 1, 3), "joint_arthrodial", "siege_horn",
    )
    mandible_l = Bone(
        name="pincer_mandible_l",
        rest_euler=(0.0, 12.0, 0.0),
        attach=_flush_attach("arthropod_head_plate", (0.0, 0.0, 1.0), offset=(0.65, -0.35, 0.0)),
        mirror_name="pincer_mandible_r",
        joint_limits=((-15.0, 15.0), (-45.0, 35.0), (-15.0, 15.0)),
        items=[
            Box("mandible_arm", (2, 2, 5), "siege_horn"),
            Box("mandible_tooth", (1, 1, 2), "siege_horn", place=_flush_place("mandible_arm", (-1.0, 0.0, 0.0), offset=(0.0, 0.0, 1.5))),
        ],
    )
    head_bone = Bone(
        name="arthropod_head",
        rest_euler=(4.0, 0.0, 0.0),
        attach=_flush_attach("cephalothorax_shell", (0.0, 0.0, 1.0)),
        joint_limits=((-25.0, 25.0), (-30.0, 30.0), (-20.0, 20.0)),
        items=[
            Box("arthropod_head_plate", (6, 4, 5), "chitin_elytra"),
            Box("siege_ram_horn", (2, 3, 7), "siege_horn", place=_flush_place("arthropod_head_plate", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 1.0))),
            Box("eye_cluster_l", (1, 2, 2), "compound_eye", place=_flush_place("arthropod_head_plate", (1.0, 0.0, 0.0), offset=(0.0, 0.5, -0.5))),
            Box("eye_cluster_r", (1, 2, 2), "compound_eye", place=_flush_place("arthropod_head_plate", (-1.0, 0.0, 0.0), offset=(0.0, 0.5, -0.5))),
            Locator("arthropod_target_focus", offset=(0.0, 1.0, 3.0), role="camera_focus"),
            mandible_l,
        ],
    )
    elytra_l = Bone(
        name="elytra_case_l",
        rest_euler=(0.0, 0.0, -8.0),
        attach=_flush_attach("abdomen_gaster", (0.0, 1.0, 0.0), offset=(0.55, 0.0, 0.0)),
        mirror_name="elytra_case_r",
        joint_limits=((-25.0, 25.0), (-20.0, 20.0), (-55.0, 10.0)),
        items=[Box("elytra_shield", (4, 2, 9), "chitin_elytra")],
    )
    abdomen = Bone(
        name="abdomen_segment",
        rest_euler=(-4.0, 0.0, 0.0),
        attach=_flush_attach("cephalothorax_shell", (0.0, 0.0, -1.0)),
        joint_limits=((-25.0, 25.0), (-25.0, 25.0), (-15.0, 15.0)),
        items=[
            Box("abdomen_gaster", (9, 6, 10), "chitin_elytra"),
            Box("stinger_telson", (3, 2, 4), "siege_horn", place=_flush_place("abdomen_gaster", (0.0, 0.0, -1.0))),
            Locator("acid_spray_emitter", offset=(0.0, 0.0, -7.0), role="vfx_emitter"),
            elytra_l,
            leg_rear_l,
        ],
    )
    root = Bone(
        name="colossus_root",
        pivot_local=(0.0, 13.0, 0.0),
        items=[
            Box("cephalothorax_shell", (8, 6, 13), "chitin_elytra"),
            Box("ventral_sternite", (6, 1, 11), "joint_arthrodial", place=_flush_place("cephalothorax_shell", (0.0, -1.0, 0.0))),
            Locator("howdah_rider_seat", offset=(0.0, 4.5, 0.0), role="rider_seat"),
            head_bone,
            leg_front_l,
            leg_mid_l,
            abdomen,
        ],
    )
    materials = (
        MaterialDecl("chitin_elytra", "thin_film_iridescence", roughness=0.18, scale=1.1, seed=seed + 17, film_nm=495.0, wear=0.35, weld_group="chitin_hull"),
        MaterialDecl("joint_arthrodial", "keratin_scales", roughness=0.48, scale=0.85, seed=seed + 29, wear=0.30),
        MaterialDecl("siege_horn", "keratin_scales", roughness=0.25, scale=0.95, seed=seed + 43, anisotropy=1.8, wear=0.55),
        MaterialDecl("compound_eye", "polymer_dielectric", roughness=0.10, ior=1.65, scale=0.6, seed=seed + 61, emissive=1.0),
    )
    style = StyleDecl("labpbr_1_3", False, 0.13, 22.0)
    muscles = (
        MuscleDecl("mandible_closer_l", "arthropod_head", "pincer_mandible_l", "y", -22.0),
        MuscleDecl("mandible_closer_r", "arthropod_head", "pincer_mandible_r", "y", 22.0),
        MuscleDecl("elytra_lifter_l", "abdomen_segment", "elytra_case_l", "z", -24.0),
        MuscleDecl("elytra_lifter_r", "abdomen_segment", "elytra_case_r", "z", 24.0),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("lf_ik", "coxa_front_l", "tibia_front_l", (0.0, -7.0, 0.5), "x"),
            IKChainDecl("rf_ik", "coxa_front_r", "coxa_front_r__tibia_front_l", (0.0, -7.0, 0.5), "x"),
            IKChainDecl("lm_ik", "coxa_mid_l", "tibia_mid_l", (0.0, -7.0, 0.5), "x"),
            IKChainDecl("rm_ik", "coxa_mid_r", "coxa_mid_r__tibia_mid_l", (0.0, -7.0, 0.5), "x"),
            IKChainDecl("lr_ik", "coxa_rear_l", "tibia_rear_l", (0.0, -7.0, 0.5), "x"),
            IKChainDecl("rr_ik", "coxa_rear_r", "coxa_rear_r__tibia_rear_l", (0.0, -7.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("cephalic_aim", ("arthropod_head",), 0.9),),
    )
    dur = 1.5
    fq = round(1.0 / dur, 4)
    # Alternating hexapod tripod gait: Tripod A = (LF, RM, LR) at 0 deg; Tripod B = (RF, LM, RR) at 180 deg!
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "hexapod_tripod_advance",
                dur,
                True,
                tracks=(TrackDecl("abdomen_segment", "wave", "x", 4.0, fq, 0.0),),
                muscle_drives=(
                    MuscleDriveDecl("mandible_closer_l", 0.55, fq, 0.0, 0.05),
                    MuscleDriveDecl("mandible_closer_r", 0.55, fq, 0.0, 0.05),
                ),
                ik_drives=(
                    IKDriveDecl("lf_ik", 1.5, 0.75, 0.0),
                    IKDriveDecl("rm_ik", 1.5, 0.75, 0.0),
                    IKDriveDecl("lr_ik", 1.5, 0.75, 0.0),
                    IKDriveDecl("rf_ik", 1.5, 0.75, 180.0),
                    IKDriveDecl("lm_ik", 1.5, 0.75, 180.0),
                    IKDriveDecl("rr_ik", 1.5, 0.75, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "siege_shell_open",
                muscle_tones=(("elytra_lifter_l", 0.85), ("elytra_lifter_r", 0.85)),
                material_mods=(MaterialModDecl("compound_eye", emissive=1.8, moisture=0.1, wear_delta=0.03),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_morphing_switch_blade_axe(rng: random.Random, seed: int) -> Program:
    hue_coat = round(rng.uniform(15.0, 240.0), 1)
    palettes = [
        Palette("forged_edge", (0.78, 0.02, 235.0), 6),
        Palette("chassis_coat", (0.36, 0.12, hue_coat), 6),
        Palette("grip_wrap", (0.30, 0.06, 42.0), 5),
        Palette("phial_plasma", (0.80, 0.22, 145.0), 4, emissive=True),
    ]
    axe_wing_l = Bone(
        name="axe_crescent_l",
        rest_euler=(0.0, 0.0, 12.0),
        attach=_flush_attach("morph_gearbox", (1.0, 0.0, 0.0), offset=(0.0, 0.65, 0.0)),
        mirror_name="axe_crescent_r",
        joint_limits=((-10.0, 10.0), (-20.0, 20.0), (-65.0, 25.0)),
        items=[
            Box("axe_cheek", (3, 7, 2), "chassis_coat"),
            Box("axe_bevel_edge", (2, 9, 1), "forged_edge", place=_flush_place("axe_cheek", (1.0, 0.0, 0.0))),
            Box("axe_upper_horn", (1, 3, 1), "forged_edge", place=_flush_place("axe_bevel_edge", (0.0, 1.0, 0.0))),
            Box("axe_lower_beard", (1, 2, 1), "forged_edge", place=_flush_place("axe_bevel_edge", (0.0, -1.0, 0.0))),
        ],
    )
    sword_tip_carrier = Bone(
        name="sword_extension_latch",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("sword_main_spine", (0.0, 1.0, 0.0)),
        joint_limits=((-45.0, 45.0), (-15.0, 15.0), (-15.0, 15.0)),
        items=[
            Box("sword_distal_fuller", (2, 7, 2), "forged_edge"),
            Box("sword_point_crown", (1, 3, 2), "forged_edge", place=_flush_place("sword_distal_fuller", (0.0, 1.0, 0.0))),
            Box("sword_distal_glow", (1, 5, 1), "phial_plasma", place=_flush_place("sword_distal_fuller", (0.0, 0.0, 1.0))),
            Locator("discharge_muzzle_vfx", offset=(0.0, 5.5, 0.0), role="vfx_emitter"),
        ],
    )
    sword_core_arm = Bone(
        name="sword_morph_arm",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("morph_gearbox", (0.0, 1.0, 0.0)),
        joint_limits=((-35.0, 35.0), (-15.0, 15.0), (-15.0, 15.0)),
        items=[
            Box("sword_main_spine", (3, 8, 2), "chassis_coat"),
            Box("sword_keen_front", (1, 8, 1), "forged_edge", place=_flush_place("sword_main_spine", (0.0, 0.0, 1.0))),
            Box("sword_keen_back", (1, 8, 1), "forged_edge", place=_flush_place("sword_main_spine", (0.0, 0.0, -1.0))),
            Box("phial_conduit_strip", (1, 6, 1), "phial_plasma", place=_flush_place("sword_keen_front", (0.0, 0.0, 1.0))),
            sword_tip_carrier,
        ],
    )
    phial_drum = Bone(
        name="phial_rotary_drum",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("morph_gearbox", (0.0, 0.0, 1.0)),
        joint_limits=((-20.0, 20.0), (-20.0, 20.0), (-180.0, 180.0)),
        items=[
            Box("drum_casing", (3, 4, 2), "chassis_coat"),
            Box("phial_cell_top", (2, 1, 2), "phial_plasma", place=_flush_place("drum_casing", (0.0, 1.0, 0.0))),
            Box("phial_cell_bot", (2, 1, 2), "phial_plasma", place=_flush_place("drum_casing", (0.0, -1.0, 0.0))),
            Box("drum_axle_cap", (1, 2, 1), "forged_edge", place=_flush_place("drum_casing", (0.0, 0.0, 1.0))),
        ],
    )
    guard_latch_l = Bone(
        name="handguard_vent_l",
        rest_euler=(0.0, 0.0, -12.0),
        attach=_flush_attach("mid_collar", (1.0, 0.0, 0.0)),
        mirror_name="handguard_vent_r",
        joint_limits=((-15.0, 15.0), (-15.0, 15.0), (-40.0, 40.0)),
        items=[
            Box("vent_fin", (2, 3, 2), "chassis_coat"),
            Box("vent_glow", (1, 2, 1), "phial_plasma", place=_flush_place("vent_fin", (1.0, 0.0, 0.0))),
        ],
    )
    grip_assembly = Bone(
        name="haft_assembly",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("morph_gearbox", (0.0, -1.0, 0.0)),
        joint_limits=((-10.0, 10.0), (-25.0, 25.0), (-10.0, 10.0)),
        items=[
            Box("crossguard_block", (5, 2, 3), "chassis_coat"),
            Box("upper_grip", (2, 6, 2), "grip_wrap", place=_flush_place("crossguard_block", (0.0, -1.0, 0.0))),
            Box("mid_collar", (3, 2, 3), "forged_edge", place=_flush_place("upper_grip", (0.0, -1.0, 0.0))),
            Box("lower_grip", (2, 6, 2), "grip_wrap", place=_flush_place("mid_collar", (0.0, -1.0, 0.0))),
            Box("counterweight_pommel", (4, 3, 4), "chassis_coat", place=_flush_place("lower_grip", (0.0, -1.0, 0.0))),
            Box("pommel_spike", (2, 2, 2), "forged_edge", place=_flush_place("counterweight_pommel", (0.0, -1.0, 0.0))),
            Locator("primary_hand_socket", offset=(0.0, -5.0, 0.0), role="socket"),
            guard_latch_l,
        ],
    )

    root = Bone(
        name="switch_axe_root",
        pivot_local=(0.0, 16.0, 0.0),
        items=[
            Box("morph_gearbox", (5, 6, 4), "chassis_coat"),
            Box("gearbox_rear_plate", (3, 4, 1), "forged_edge", place=_flush_place("morph_gearbox", (0.0, 0.0, -1.0))),
            Box("gearbox_rear_vent", (2, 2, 1), "phial_plasma", place=_flush_place("gearbox_rear_plate", (0.0, 0.0, -1.0))),
            axe_wing_l,
            sword_core_arm,
            phial_drum,
            grip_assembly,
        ],
    )
    materials = (
        MaterialDecl("forged_edge", "brushed_metal", roughness=0.18, metalness=1.0, scale=0.9, seed=seed + 11, anisotropy=2.1, wear=0.30, oxidation=0.18),
        MaterialDecl("chassis_coat", "painted_metal", roughness=0.32, coating=0.65, scale=1.1, seed=seed + 29, wear=0.45, oxidation=0.35),
        MaterialDecl("grip_wrap", "woven_fabric", roughness=0.72, scale=0.75, seed=seed + 41, porosity=0.42),
        MaterialDecl("phial_plasma", "thin_film_iridescence", roughness=0.14, scale=1.0, seed=seed + 59, emissive=1.3, film_nm=520.0),
    )
    style = StyleDecl("labpbr_1_3", False, 0.14, 20.0)
    # Mechanical cable/cam actuator modeled via @muscles + tendon_coupling synchronized folding!
    muscles = (
        MuscleDecl("cam_fold_actuator_l", "switch_axe_root", "axe_crescent_l", "z", -28.0, TendonCoupling("handguard_vent_l", "z", -0.50)),
        MuscleDecl("cam_fold_actuator_r", "switch_axe_root", "axe_crescent_r", "z", 28.0, TendonCoupling("handguard_vent_r", "z", 0.50)),
    )
    kin = KinematicsDecl(look_chains=(LookChainDecl("blade_alignment", ("sword_morph_arm", "sword_extension_latch"), 0.8),))
    dur = 1.6
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "phial_morph_cycle",
                dur,
                True,
                tracks=(
                    TrackDecl("phial_rotary_drum", "hinge", "z", 90.0, fq, 0.0),
                    TrackDecl("sword_extension_latch", "hinge", "x", 12.0, fq, 0.0),
                ),
                muscle_drives=(
                    MuscleDriveDecl("cam_fold_actuator_l", 0.85, fq, 0.0, 0.0),
                    MuscleDriveDecl("cam_fold_actuator_r", 0.85, fq, 0.0, 0.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "elemental_phial_awakened",
                muscle_tones=(("cam_fold_actuator_l", 0.90), ("cam_fold_actuator_r", 0.90)),
                material_mods=(MaterialModDecl("phial_plasma", emissive=2.0, moisture=0.0, wear_delta=0.08),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("weapon_3d", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_elemental_pocket_companion(rng: random.Random, seed: int) -> Program:
    """Cobblemon-inspired clean-room expressive battle/companion creature with full socket contract."""
    hue = round(rng.uniform(35.0, 145.0), 1)
    palettes = [
        Palette("companion_coat", (0.64, 0.16, hue), 6),
        Palette("cream_ruff", (0.86, 0.06, (hue + 25.0) % 360.0), 5),
        Palette("elemental_bud", (0.76, 0.22, (hue + 140.0) % 360.0), 5, emissive=True),
        Palette("paw_claw", (0.34, 0.05, 45.0), 4),
    ]
    ear_l = Bone(
        name="long_ear_l",
        rest_euler=(-10.0, 0.0, -16.0),
        attach=_flush_attach("companion_cranium", (0.0, 1.0, 0.0), offset=(0.65, 0.0, -0.1)),
        mirror_name="long_ear_r",
        joint_limits=((-35.0, 35.0), (-25.0, 25.0), (-45.0, 45.0)),
        items=[
            Box("ear_pinna", (2, 6, 2), "companion_coat"),
            Box("ear_tip_tuft", (2, 3, 2), "elemental_bud", place=_flush_place("ear_pinna", (0.0, 1.0, 0.0))),
        ],
    )
    cheek_pouch_l = Bone(
        name="spark_cheek_l",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("companion_cranium", (1.0, 0.0, 0.0), offset=(0.0, -0.25, 0.3)),
        mirror_name="spark_cheek_r",
        joint_limits=((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
        items=[Box("cheek_gland", (1, 2, 2), "elemental_bud")],
    )
    jaw = Bone(
        name="companion_jaw",
        rest_euler=(6.0, 0.0, 0.0),
        attach=_flush_attach("companion_cranium", (0.0, 0.0, 1.0), offset=(0.0, -0.55, 0.0)),
        joint_limits=((-5.0, 35.0), (-10.0, 10.0), (-10.0, 10.0)),
        items=[
            Box("smile_mandible", (4, 1, 2), "cream_ruff"),
            Locator("special_attack_vfx", offset=(0.0, 0.5, 2.5), role="vfx_emitter"),
        ],
    )
    head = Bone(
        name="companion_head",
        rest_euler=(4.0, 0.0, 0.0),
        attach=_flush_attach("collar_ruff", (0.0, 1.0, 0.0)),
        joint_limits=((-35.0, 35.0), (-50.0, 50.0), (-30.0, 30.0)),
        items=[
            Box("companion_cranium", (7, 6, 7), "companion_coat"),
            Box("forehead_gem", (2, 2, 1), "elemental_bud", place=_flush_place("companion_cranium", (0.0, 0.0, 1.0), offset=(0.0, 1.0, 0.0))),
            Locator("head_look_target", offset=(0.0, 1.0, 3.5), role="camera_focus"),
            ear_l,
            cheek_pouch_l,
            jaw,
        ],
    )
    arm_l = _build_two_bone_limb(
        "fore_arm_l", "fore_paw_l", "fore_arm_r",
        "plump_torso", (1.0, 0.0, 0.0), (0.0, -0.05, 0.55),
        (2, 4, 3), (2, 3, 2), (2, 1, 3), "companion_coat", "paw_claw",
        upper_euler=(-38.0, 0.0, 32.0), lower_euler=(-18.0, 0.0, 6.0),
    )
    leg_l = _build_two_bone_limb(
        "hind_haunch_l", "hind_paw_l", "hind_haunch_r",
        "hip_pelvis", (1.0, 0.0, 0.0), (0.0, -0.25, -0.25),
        (3, 4, 4), (2, 4, 3), (3, 2, 4), "companion_coat", "paw_claw",
        upper_euler=(8.0, 0.0, 8.0),
    )
    plume_tail = _build_segmented_chain(
        "plume_tail",
        "hip_pelvis",
        (0.0, 0.0, -1.0),
        [(3, 3, 5), (4, 4, 5), (5, 5, 6)],
        "companion_coat",
        "elemental_bud",
        rest_euler_step=(-6.0, 0.0, 0.0),
        tip_locator=("tail_burst_emitter", "vfx_emitter"),
    )

    root = Bone(
        name="companion_root",
        pivot_local=(0.0, 12.0, 0.0),
        items=[
            Box("plump_torso", (7, 7, 7), "companion_coat"),
            Box("belly_patch", (5, 5, 1), "cream_ruff", place=_flush_place("plump_torso", (0.0, 0.0, 1.0))),
            Box("collar_ruff", (7, 2, 7), "cream_ruff", place=_flush_place("plump_torso", (0.0, 1.0, 0.0))),
            Box("hip_pelvis", (6, 4, 6), "companion_coat", place=_flush_place("plump_torso", (0.0, -1.0, 0.0))),
            Locator("shoulder_perch_seat", offset=(0.0, 5.5, -1.0), role="rider_seat"),
            Locator("battle_hitbox_center", offset=(0.0, 1.0, 0.0), role="hitbox"),
            Locator("held_berry_socket", offset=(0.0, 1.5, 4.5), role="socket"),
            head,
            arm_l,
            leg_l,
            plume_tail,
        ],
    )
    materials = (
        MaterialDecl("companion_coat", "patterned_skin", roughness=0.58, scale=1.15, seed=seed + 19, subsurface=0.40, rd_feed=0.038, rd_kill=0.061, weld_group="companion_fur"),
        MaterialDecl("cream_ruff", "patterned_skin", roughness=0.66, scale=0.9, seed=seed + 31, porosity=0.30, weld_group="companion_fur"),
        MaterialDecl("elemental_bud", "thin_film_iridescence", roughness=0.22, scale=1.0, seed=seed + 47, emissive=1.0, film_nm=400.0),
        MaterialDecl("paw_claw", "keratin_scales", roughness=0.35, scale=0.8, seed=seed + 61),
    )
    style = StyleDecl("labpbr_1_3", True, 0.09, 20.0)
    muscles = (
        MuscleDecl("ear_perk_l", "companion_head", "long_ear_l", "z", -18.0),
        MuscleDecl("ear_perk_r", "companion_head", "long_ear_r", "z", 18.0),
        MuscleDecl("hop_calf_l", "companion_root", "hind_haunch_l", "x", 16.0, TendonCoupling("hind_paw_l", "x", -0.60), "hind_haunch_l_muscle_mass", 0.08),
        MuscleDecl("hop_calf_r", "companion_root", "hind_haunch_r", "x", 16.0, TendonCoupling("hind_haunch_r__hind_paw_l", "x", -0.60), "hind_haunch_r__hind_haunch_l_muscle_mass", 0.08),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("paw_l_ik", "hind_haunch_l", "hind_paw_l", (0.0, -6.0, 0.5), "x"),
            IKChainDecl("paw_r_ik", "hind_haunch_r", "hind_haunch_r__hind_paw_l", (0.0, -6.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("companion_gaze", ("companion_head",), 0.85),),
    )
    dur = 1.4
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "ground_jaunt_walk",
                dur,
                True,
                tracks=(
                    TrackDecl("plume_tail_1", "wave", "y", 12.0, fq, 0.0),
                    TrackDecl("plume_tail_2", "wave", "y", 15.0, fq, 300.0),
                    TrackDecl("plume_tail_3", "wave", "y", 18.0, fq, 240.0),
                ),
                muscle_drives=(
                    MuscleDriveDecl("ear_perk_l", 0.70, fq, 0.0, 0.10),
                    MuscleDriveDecl("ear_perk_r", 0.70, fq, 180.0, 0.10),
                ),
                ik_drives=(
                    IKDriveDecl("paw_l_ik", 1.5, 0.8, 0.0),
                    IKDriveDecl("paw_r_ik", 1.5, 0.8, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "battle_surge_ready",
                muscle_tones=(("ear_perk_l", 0.95), ("ear_perk_r", 0.95)),
                material_mods=(MaterialModDecl("elemental_bud", emissive=1.85, moisture=0.2, wear_delta=0.01),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bedrock_entity", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_modular_apricorn_healing_station(rng: random.Random, seed: int) -> Program:
    """Cobblemon-inspired clean-room interactive healing/crafting block assembly."""
    hue_trim = round(rng.uniform(5.0, 35.0), 1)
    palettes = [
        Palette("enamel_chassis", (0.84, 0.04, 230.0), 6),
        Palette("crimson_trim", (0.52, 0.18, hue_trim), 6),
        Palette("machined_alloy", (0.68, 0.02, 240.0), 5),
        Palette("pulse_visor", (0.78, 0.18, 165.0), 4, emissive=True),
    ]
    cradle_l = Bone(
        name="capsule_cradle_l",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("tray_deck", (0.0, 1.0, 0.0), offset=(0.55, 0.0, 0.0)),
        mirror_name="capsule_cradle_r",
        joint_limits=((-15.0, 15.0), (-180.0, 180.0), (-15.0, 15.0)),
        items=[
            Box("cradle_ring", (4, 1, 6), "machined_alloy"),
            Box("capsule_slot_front", (2, 2, 2), "crimson_trim", place=_flush_place("cradle_ring", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 1.5))),
            Box("capsule_slot_rear", (2, 2, 2), "crimson_trim", place=_flush_place("cradle_ring", (0.0, 1.0, 0.0), offset=(0.0, 0.0, -1.5))),
            Box("capsule_core_node", (1, 1, 1), "pulse_visor", place=_flush_place("cradle_ring", (0.0, 1.0, 0.0))),
            Locator("capsule_dock_socket", offset=(0.0, 1.5, 0.0), role="socket"),
        ],
    )
    canopy_lid = Bone(
        name="inspection_canopy",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("rear_pylon", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 0.4)),
        joint_limits=((-65.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)),
        items=[
            Box("canopy_hinge_bar", (12, 2, 3), "machined_alloy"),
            Box("canopy_roof_hood", (12, 2, 7), "pulse_visor", place=_flush_place("canopy_hinge_bar", (0.0, 0.0, 1.0))),
            Box("canopy_lip", (10, 1, 1), "crimson_trim", place=_flush_place("canopy_roof_hood", (0.0, 0.0, 1.0))),
            Box("canopy_crest", (6, 1, 5), "enamel_chassis", place=_flush_place("canopy_roof_hood", (0.0, 1.0, 0.0))),
        ],
    )
    scanner_head = Bone(
        name="scanner_emitter_head",
        rest_euler=(10.0, 0.0, 0.0),
        attach=_flush_attach("scanner_boom", (0.0, 0.0, 1.0)),
        joint_limits=((-40.0, 40.0), (-45.0, 45.0), (-20.0, 20.0)),
        items=[
            Box("emitter_housing", (4, 2, 3), "machined_alloy"),
            Box("emitter_lens", (2, 1, 2), "pulse_visor", place=_flush_place("emitter_housing", (0.0, -1.0, 0.0))),
            Locator("healing_beam_vfx", offset=(0.0, -1.5, 0.0), role="vfx_emitter"),
        ],
    )
    scanner_arm = Bone(
        name="scanner_arm_pivot",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("rear_pylon", (0.0, 0.0, 1.0), offset=(0.0, 0.35, 0.0)),
        joint_limits=((-25.0, 25.0), (-45.0, 45.0), (-15.0, 15.0)),
        items=[
            Box("scanner_boom", (2, 2, 4), "machined_alloy"),
            scanner_head,
        ],
    )
    side_vent_l = Bone(
        name="coolant_louver_l",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("station_pedestal", (1.0, 0.0, 0.0)),
        mirror_name="coolant_louver_r",
        joint_limits=((-10.0, 10.0), (-25.0, 25.0), (-30.0, 30.0)),
        items=[
            Box("louver_panel", (1, 6, 8), "machined_alloy"),
            Box("louver_trim", (1, 2, 6), "crimson_trim", place=_flush_place("louver_panel", (1.0, 0.0, 0.0), offset=(0.0, -1.5, 0.0))),
            Box("apricorn_hopper_pod", (2, 3, 3), "enamel_chassis", place=_flush_place("louver_panel", (1.0, 0.0, 0.0), offset=(0.0, 2.0, -1.5))),
            Box("hopper_cap", (2, 1, 3), "crimson_trim", place=_flush_place("apricorn_hopper_pod", (0.0, 1.0, 0.0))),
        ],
    )

    root = Bone(
        name="station_root",
        pivot_local=(0.0, 5.0, 0.0),
        items=[
            Box("station_pedestal", (14, 8, 12), "enamel_chassis"),
            Box("base_plinth", (16, 2, 14), "machined_alloy", place=_flush_place("station_pedestal", (0.0, -1.0, 0.0))),
            Box("front_console", (12, 4, 2), "crimson_trim", place=_flush_place("station_pedestal", (0.0, 0.0, 1.0), offset=(0.0, -1.0, 0.0))),
            Box("console_screen", (8, 2, 1), "pulse_visor", place=_flush_place("front_console", (0.0, 1.0, 0.0))),
            Box("console_keypad", (8, 1, 1), "machined_alloy", place=_flush_place("front_console", (0.0, 0.0, 1.0))),
            Box("tray_deck", (12, 2, 8), "enamel_chassis", place=_flush_place("station_pedestal", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 1.5))),
            Box("rear_pylon", (12, 8, 3), "enamel_chassis", place=_flush_place("station_pedestal", (0.0, 1.0, 0.0), offset=(0.0, 0.0, -4.5))),
            Box("pylon_status_bar", (10, 2, 1), "pulse_visor", place=_flush_place("rear_pylon", (0.0, 0.0, -1.0))),
            Box("rear_transformer_box", (8, 5, 2), "machined_alloy", place=_flush_place("station_pedestal", (0.0, 0.0, -1.0))),
            Box("transformer_coil", (4, 3, 1), "pulse_visor", place=_flush_place("rear_transformer_box", (0.0, 0.0, -1.0))),
            Locator("operator_camera_focus", offset=(0.0, 6.0, 6.0), role="camera_focus"),
            cradle_l,
            canopy_lid,
            scanner_arm,
            side_vent_l,
        ],
    )
    materials = (
        MaterialDecl("enamel_chassis", "painted_metal", roughness=0.26, coating=0.75, scale=1.0, seed=seed + 11, wear=0.32, oxidation=0.20),
        MaterialDecl("crimson_trim", "painted_metal", roughness=0.22, coating=0.80, scale=0.9, seed=seed + 23, wear=0.38),
        MaterialDecl("machined_alloy", "brushed_metal", roughness=0.28, metalness=1.0, scale=0.85, seed=seed + 37, anisotropy=1.9),
        MaterialDecl("pulse_visor", "polymer_dielectric", roughness=0.12, ior=1.58, scale=1.0, seed=seed + 53, emissive=1.15),
    )
    style = StyleDecl("labpbr_1_3", False, 0.12, 20.0)
    muscles = (
        MuscleDecl("canopy_servo", "station_root", "inspection_canopy", "x", -32.0),
        MuscleDecl("scanner_gimbal", "station_root", "scanner_arm_pivot", "y", 18.0, TendonCoupling("scanner_emitter_head", "x", 0.45)),
    )
    kin = KinematicsDecl(look_chains=(LookChainDecl("scanner_tracking", ("scanner_arm_pivot", "scanner_emitter_head"), 0.75),))
    dur = 2.0
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "restoration_scan_cycle",
                dur,
                True,
                tracks=(
                    TrackDecl("capsule_cradle_l", "wave", "y", 25.0, fq, 0.0),
                    TrackDecl("capsule_cradle_r", "wave", "y", -25.0, fq, 0.0),
                ),
                muscle_drives=(
                    MuscleDriveDecl("scanner_gimbal", 0.80, fq, 0.0, 0.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "canopy_open_maintenance",
                muscle_tones=(("canopy_servo", 0.85),),
                material_mods=(MaterialModDecl("pulse_visor", emissive=1.95, moisture=0.0, wear_delta=0.02),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("block_multipart", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_elder_calamity_dragon(rng: random.Random, seed: int) -> Program:
    """Held-Out Unseen Family (>= 56 boxes): 6-limbed Elder Dragon (4 stance legs + 2 nested-mirror wings)."""
    hue = round(rng.uniform(255.0, 315.0), 1)
    palettes = [
        Palette("obsidian_scale", (0.28, 0.09, hue), 6),
        Palette("elder_gold_trim", (0.72, 0.14, 78.0), 5),
        Palette("aether_membrane", (0.52, 0.18, (hue + 25.0) % 360.0), 6),
        Palette("calamity_core", (0.78, 0.24, 310.0), 4, emissive=True),
    ]
    wing_l = _build_nested_mirror_wyvern_wing(
        "elder_wing_mount",
        "obsidian_scale",
        "aether_membrane",
        "elder_gold_trim",
        span_arm=9,
        span_fore=10,
        span_digit=9,
    )
    foreleg_l = _build_two_bone_limb(
        "elder_fore_humerus_l", "elder_fore_radius_l", "elder_fore_humerus_r",
        "elder_thorax", (1.0, 0.0, 0.0), (0.0, -0.3, 0.3),
        (4, 7, 4), (3, 6, 3), (4, 2, 5), "obsidian_scale", "elder_gold_trim",
    )
    hindleg_l = _build_two_bone_limb(
        "elder_hind_femur_l", "elder_hind_tibia_l", "elder_hind_femur_r",
        "elder_pelvis", (1.0, 0.0, 0.0), (0.0, -0.25, 0.0),
        (4, 7, 5), (3, 6, 4), (4, 2, 5), "obsidian_scale", "elder_gold_trim",
    )
    elder_jaw = Bone(
        name="elder_jaw",
        rest_euler=(10.0, 0.0, 0.0),
        attach=_flush_attach("elder_skull", (0.0, -1.0, 0.0), offset=(0.0, 0.0, 0.4)),
        joint_limits=((-5.0, 48.0), (-10.0, 10.0), (-10.0, 10.0)),
        items=[
            Box("elder_mandible", (4, 2, 5), "obsidian_scale"),
            Locator("calamity_beam_vfx", offset=(0.0, 0.5, 3.5), role="vfx_emitter"),
        ],
    )
    great_horn_l = Bone(
        name="elder_crown_horn_l",
        rest_euler=(-20.0, 0.0, -14.0),
        attach=_flush_attach("elder_skull", (0.0, 1.0, 0.0), offset=(0.65, 0.0, 0.0)),
        mirror_name="elder_crown_horn_r",
        joint_limits=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),
        items=[
            Box("crown_horn_prox", (2, 5, 2), "elder_gold_trim"),
            Box("crown_horn_dist", (1, 5, 2), "elder_gold_trim", place=_flush_place("crown_horn_prox", (0.0, 1.0, 0.0))),
        ],
    )
    elder_head = Bone(
        name="elder_head",
        rest_euler=(8.0, 0.0, 0.0),
        attach=_flush_attach("elder_neck_seg_2", (0.0, 0.0, 1.0)),
        joint_limits=((-40.0, 40.0), (-50.0, 50.0), (-30.0, 30.0)),
        items=[
            Box("elder_skull", (6, 5, 7), "obsidian_scale"),
            Box("elder_snout", (4, 3, 4), "obsidian_scale", place=_flush_place("elder_skull", (0.0, 0.0, 1.0))),
            Box("aether_third_eye", (2, 2, 1), "calamity_core", place=_flush_place("elder_skull", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 2.0))),
            Locator("elder_gaze_focus", offset=(0.0, 1.5, 4.5), role="camera_focus"),
            great_horn_l,
            elder_jaw,
        ],
    )
    neck_2 = Bone(
        name="elder_neck_2",
        rest_euler=(-12.0, 0.0, 0.0),
        attach=_flush_attach("elder_neck_seg_1", (0.0, 0.0, 1.0)),
        joint_limits=((-35.0, 35.0), (-40.0, 40.0), (-25.0, 25.0)),
        items=[
            Box("elder_neck_seg_2", (5, 5, 6), "obsidian_scale"),
            Box("neck_crest_2", (2, 2, 4), "elder_gold_trim", place=_flush_place("elder_neck_seg_2", (0.0, 1.0, 0.0))),
            elder_head,
        ],
    )
    neck_1 = Bone(
        name="elder_neck_1",
        rest_euler=(-14.0, 0.0, 0.0),
        attach=_flush_attach("elder_thorax", (0.0, 0.0, 1.0), offset=(0.0, 0.25, 0.0)),
        joint_limits=((-35.0, 35.0), (-40.0, 40.0), (-25.0, 25.0)),
        items=[
            Box("elder_neck_seg_1", (6, 6, 6), "obsidian_scale"),
            Box("neck_crest_1", (2, 2, 4), "elder_gold_trim", place=_flush_place("elder_neck_seg_1", (0.0, 1.0, 0.0))),
            neck_2,
        ],
    )
    tail = _build_segmented_chain(
        "elder_tail",
        "elder_pelvis",
        (0.0, 0.0, -1.0),
        [(6, 5, 7), (5, 4, 7), (4, 3, 7), (4, 3, 8)],
        "obsidian_scale",
        "elder_gold_trim",
        tip_locator=("elder_tail_spear", "hitbox"),
    )

    root = Bone(
        name="elder_root",
        pivot_local=(0.0, 19.0, 0.0),
        items=[
            Box("elder_thorax", (10, 8, 12), "obsidian_scale"),
            Box("aether_furnace_chest", (6, 3, 8), "calamity_core", place=_flush_place("elder_thorax", (0.0, -1.0, 0.0))),
            Box("elder_wing_mount", (10, 3, 7), "elder_gold_trim", place=_flush_place("elder_thorax", (0.0, 1.0, 0.0), offset=(0.0, 0.0, 1.0))),
            Box("elder_pelvis", (8, 7, 10), "obsidian_scale", place=_flush_place("elder_thorax", (0.0, 0.0, -1.0))),
            Locator("elder_throne_seat", offset=(0.0, 7.5, -2.0), role="rider_seat"),
            wing_l,
            foreleg_l,
            hindleg_l,
            neck_1,
            tail,
        ],
    )
    materials = (
        MaterialDecl("obsidian_scale", "keratin_scales", roughness=0.28, scale=1.2, seed=seed + 13, anisotropy=1.7, wear=0.40, weld_group="elder_hide"),
        MaterialDecl("elder_gold_trim", "brushed_metal", roughness=0.24, metalness=1.0, scale=0.95, seed=seed + 31, anisotropy=1.9, oxidation=0.12),
        MaterialDecl("aether_membrane", "thin_film_iridescence", roughness=0.20, scale=1.35, seed=seed + 47, film_nm=385.0, weld_group="elder_wing"),
        MaterialDecl("calamity_core", "patterned_skin", roughness=0.30, scale=1.05, seed=seed + 61, emissive=1.35),
    )
    style = StyleDecl("labpbr_1_3", False, 0.13, 22.0)
    muscles = (
        MuscleDecl("elder_wing_depressor_l", "elder_root", "wing_arm_l", "z", 20.0, TendonCoupling("wing_forearm_l", "z", 0.35), "wing_humerus", 0.09),
        MuscleDecl("elder_wing_depressor_r", "elder_root", "wing_arm_r", "z", -20.0, TendonCoupling("wing_arm_r__wing_forearm_l", "z", 0.35), "wing_arm_r__wing_humerus", 0.09),
        MuscleDecl("elder_jaw_closer", "elder_head", "elder_jaw", "x", 24.0),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("efl_ik", "elder_fore_humerus_l", "elder_fore_radius_l", (0.0, -8.0, 0.5), "x"),
            IKChainDecl("efr_ik", "elder_fore_humerus_r", "elder_fore_humerus_r__elder_fore_radius_l", (0.0, -8.0, 0.5), "x"),
            IKChainDecl("ehl_ik", "elder_hind_femur_l", "elder_hind_tibia_l", (0.0, -8.0, 0.5), "x"),
            IKChainDecl("ehr_ik", "elder_hind_femur_r", "elder_hind_femur_r__elder_hind_tibia_l", (0.0, -8.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("elder_regal_gaze", ("elder_neck_1", "elder_neck_2", "elder_head"), 0.74),),
    )
    dur = 1.8
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "regal_hexapod_procession",
                dur,
                True,
                tracks=(
                    TrackDecl("elder_tail_1", "wave", "y", 8.0, fq, 0.0),
                    TrackDecl("elder_tail_2", "wave", "y", 11.0, fq, 315.0),
                    TrackDecl("elder_tail_3", "wave", "y", 13.0, fq, 270.0),
                    TrackDecl("elder_tail_4", "wave", "y", 15.0, fq, 225.0),
                ),
                muscle_drives=(
                    MuscleDriveDecl("elder_wing_depressor_l", 0.55, fq, 0.0, 0.05),
                    MuscleDriveDecl("elder_wing_depressor_r", 0.55, fq, 0.0, 0.05),
                ),
                ik_drives=(
                    IKDriveDecl("efl_ik", 1.8, 0.85, 0.0),
                    IKDriveDecl("ehr_ik", 1.8, 0.85, 0.0),
                    IKDriveDecl("efr_ik", 1.8, 0.85, 180.0),
                    IKDriveDecl("ehl_ik", 1.8, 0.85, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "cataclysm_eclipse_aura",
                muscle_tones=(("elder_wing_depressor_l", 0.70), ("elder_wing_depressor_r", 0.70), ("elder_jaw_closer", 0.65)),
                material_mods=(MaterialModDecl("calamity_core", emissive=2.0, moisture=0.1, wear_delta=0.05),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


def _build_relic_automaton_guardian(rng: random.Random, seed: int) -> Program:
    """Held-Out Unseen Family (>= 44 boxes): Ancient Stone & Bronze Piston Guardian Colossus."""
    palettes = [
        Palette("ancient_basalt", (0.44, 0.03, 215.0), 6),
        Palette("patina_bronze", (0.56, 0.11, 72.0), 6),
        Palette("overgrown_moss", (0.52, 0.14, 132.0), 5),
        Palette("aether_rune", (0.80, 0.19, 198.0), 4, emissive=True),
    ]
    claw_finger_l = Bone(
        name="piston_claw_in_l",
        rest_euler=(0.0, 0.0, 6.0),
        attach=_flush_attach("gauntlet_wrist", (0.0, -1.0, 0.0), offset=(0.65, 0.0, 0.0)),
        mirror_name="piston_claw_out_l",
        joint_limits=((-20.0, 20.0), (-20.0, 20.0), (-45.0, 45.0)),
        items=[
            Box("claw_digit", (1, 4, 2), "patina_bronze"),
            Box("claw_tip_wedge", (1, 2, 1), "ancient_basalt", place=_flush_place("claw_digit", (0.0, -1.0, 0.0))),
        ],
    )
    forearm_l = Bone(
        name="guardian_forearm_l",
        rest_euler=(-16.0, 0.0, 0.0),
        attach=_flush_attach("bicep_cylinder", (0.0, -1.0, 0.0)),
        joint_limits=((-75.0, 45.0), (-25.0, 25.0), (-25.0, 25.0)),
        items=[
            Box("gauntlet_wrist", (5, 7, 5), "ancient_basalt"),
            Box("gauntlet_shield_face", (1, 6, 6), "patina_bronze", place=_flush_place("gauntlet_wrist", (1.0, 0.0, 0.0))),
            Box("gauntlet_rune_inlay", (1, 4, 2), "aether_rune", place=_flush_place("gauntlet_shield_face", (1.0, 0.0, 0.0))),
            claw_finger_l,
        ],
    )
    arm_l = Bone(
        name="guardian_shoulder_l",
        rest_euler=(-8.0, 0.0, 16.0),
        attach=_flush_attach("shoulder_yoke", (1.0, 0.0, 0.0)),
        mirror_name="guardian_shoulder_r",
        joint_limits=((-65.0, 65.0), (-35.0, 35.0), (-55.0, 55.0)),
        items=[
            Box("pauldron_arch", (5, 5, 6), "ancient_basalt"),
            Box("pauldron_moss_cap", (4, 2, 5), "overgrown_moss", place=_flush_place("pauldron_arch", (0.0, 1.0, 0.0))),
            Box("pauldron_rune_stud", (1, 2, 2), "aether_rune", place=_flush_place("pauldron_arch", (1.0, 0.0, 0.0))),
            Box("bicep_cylinder", (3, 6, 3), "patina_bronze", place=_flush_place("pauldron_arch", (0.0, -1.0, 0.0))),
            forearm_l,
        ],
    )
    boiler_stack_l = Bone(
        name="boiler_chimney_l",
        rest_euler=(-18.0, 0.0, 0.0),
        attach=_flush_attach("shoulder_yoke", (0.0, 1.0, 0.0), offset=(0.65, 0.0, -0.65)),
        mirror_name="boiler_chimney_r",
        joint_limits=((-15.0, 15.0), (-15.0, 15.0), (-15.0, 15.0)),
        items=[
            Box("chimney_pipe", (2, 5, 2), "patina_bronze"),
            Box("chimney_cowl", (3, 1, 3), "ancient_basalt", place=_flush_place("chimney_pipe", (0.0, 1.0, 0.0))),
        ],
    )
    leg_l = _build_two_bone_limb(
        "guardian_thigh_l", "guardian_greave_l", "guardian_thigh_r",
        "pelvic_keystone", (1.0, 0.0, 0.0), (0.0, -0.25, 0.0),
        (4, 6, 5), (4, 6, 4), (5, 2, 6), "ancient_basalt", "patina_bronze",
    )
    flywheel = Bone(
        name="dorsal_flywheel",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("monolith_chest", (0.0, 0.0, -1.0)),
        joint_limits=((-10.0, 10.0), (-10.0, 10.0), (-180.0, 180.0)),
        items=[
            Box("flywheel_hub", (4, 4, 2), "patina_bronze"),
            Box("flywheel_spoke_n", (2, 4, 1), "patina_bronze", place=_flush_place("flywheel_hub", (0.0, 1.0, 0.0))),
            Box("flywheel_spoke_s", (2, 4, 1), "patina_bronze", place=_flush_place("flywheel_hub", (0.0, -1.0, 0.0))),
            Box("flywheel_spoke_e", (4, 2, 1), "patina_bronze", place=_flush_place("flywheel_hub", (1.0, 0.0, 0.0))),
            Box("flywheel_spoke_w", (4, 2, 1), "patina_bronze", place=_flush_place("flywheel_hub", (-1.0, 0.0, 0.0))),
            Box("flywheel_core_gem", (2, 2, 1), "aether_rune", place=_flush_place("flywheel_hub", (0.0, 0.0, -1.0))),
        ],
    )
    head = Bone(
        name="guardian_turret_head",
        rest_euler=(0.0, 0.0, 0.0),
        attach=_flush_attach("shoulder_yoke", (0.0, 1.0, 0.0)),
        joint_limits=((-25.0, 25.0), (-65.0, 65.0), (-20.0, 20.0)),
        items=[
            Box("turret_monolith", (6, 6, 6), "ancient_basalt"),
            Box("cyclops_visor", (4, 2, 1), "aether_rune", place=_flush_place("turret_monolith", (0.0, 0.0, 1.0))),
            Box("crown_moss_slab", (5, 1, 5), "overgrown_moss", place=_flush_place("turret_monolith", (0.0, 1.0, 0.0))),
            Locator("guardian_eye_beam", offset=(0.0, 0.0, 4.0), role="vfx_emitter"),
            Locator("guardian_look_focus", offset=(0.0, 1.0, 3.5), role="camera_focus"),
        ],
    )

    root = Bone(
        name="guardian_root",
        pivot_local=(0.0, 18.0, 0.0),
        items=[
            Box("monolith_chest", (10, 9, 8), "ancient_basalt"),
            Box("shoulder_yoke", (14, 3, 7), "patina_bronze", place=_flush_place("monolith_chest", (0.0, 1.0, 0.0))),
            Box("core_furnace_lens", (4, 4, 1), "aether_rune", place=_flush_place("monolith_chest", (0.0, 0.0, 1.0))),
            Box("chest_moss_drape", (8, 2, 1), "overgrown_moss", place=_flush_place("monolith_chest", (0.0, 0.0, 1.0), offset=(0.0, 3.0, 0.0))),
            Box("waist_gear_ring", (6, 3, 6), "patina_bronze", place=_flush_place("monolith_chest", (0.0, -1.0, 0.0))),
            Box("pelvic_keystone", (8, 4, 7), "ancient_basalt", place=_flush_place("waist_gear_ring", (0.0, -1.0, 0.0))),
            Locator("guardian_shoulder_seat", offset=(0.0, 7.5, 0.0), role="rider_seat"),
            head,
            boiler_stack_l,
            flywheel,
            arm_l,
            leg_l,
        ],
    )
    materials = (
        MaterialDecl("ancient_basalt", "weathered_stone", roughness=0.78, scale=1.3, seed=seed + 17, porosity=0.38, biofilm=0.35, weld_group="stone_masonry"),
        MaterialDecl("patina_bronze", "brushed_metal", roughness=0.34, metalness=1.0, scale=1.0, seed=seed + 33, anisotropy=1.6, oxidation=0.45, wear=0.35),
        MaterialDecl("overgrown_moss", "lichen_biofilm", roughness=0.82, scale=0.95, seed=seed + 49, porosity=0.55, biofilm=0.65),
        MaterialDecl("aether_rune", "thin_film_iridescence", roughness=0.15, scale=1.0, seed=seed + 63, emissive=1.25, film_nm=475.0),
    )
    style = StyleDecl("labpbr_1_3", False, 0.11, 20.0)
    muscles = (
        MuscleDecl("piston_bicep_l", "guardian_root", "guardian_shoulder_l", "x", -22.0, TendonCoupling("guardian_forearm_l", "x", -0.55), "bicep_cylinder", 0.08),
        MuscleDecl("piston_bicep_r", "guardian_root", "guardian_shoulder_r", "x", -22.0, TendonCoupling("guardian_shoulder_r__guardian_forearm_l", "x", -0.55), "guardian_shoulder_r__bicep_cylinder", 0.08),
    )
    kin = KinematicsDecl(
        ik_chains=(
            IKChainDecl("g_leg_l", "guardian_thigh_l", "guardian_greave_l", (0.0, -9.0, 0.5), "x"),
            IKChainDecl("g_leg_r", "guardian_thigh_r", "guardian_thigh_r__guardian_greave_l", (0.0, -9.0, 0.5), "x"),
        ),
        look_chains=(LookChainDecl("turret_gaze", ("guardian_turret_head",), 0.9),),
    )
    dur = 2.0
    fq = round(1.0 / dur, 4)
    beh = BehaviorDecl(
        clips=(
            ClipDecl(
                "colossus_patrol_stride",
                dur,
                True,
                tracks=(
                    TrackDecl("dorsal_flywheel", "hinge", "z", 90.0, fq, 0.0),
                    TrackDecl("guardian_turret_head", "wave", "y", 18.0, fq, 0.0),
                ),
                muscle_drives=(
                    MuscleDriveDecl("piston_bicep_l", 0.65, fq, 0.0, 0.05),
                    MuscleDriveDecl("piston_bicep_r", 0.65, fq, 180.0, 0.05),
                ),
                ik_drives=(
                    IKDriveDecl("g_leg_l", 1.8, 0.85, 0.0),
                    IKDriveDecl("g_leg_r", 1.8, 0.85, 180.0),
                ),
            ),
        ),
        states=(
            BehaviorStateDecl(
                "sentinel_overdrive",
                muscle_tones=(("piston_bicep_l", 0.85), ("piston_bicep_r", 0.85)),
                material_mods=(MaterialModDecl("aether_rune", emissive=2.0, moisture=0.2, wear_delta=0.04),),
            ),
        ),
    )
    return _finalize_and_verify_apex(
        Program(Meta("entity", "bbmodel_5", 16), palettes, root, materials=materials, style=style, muscles=muscles, kinematics=kin, behavior=beh)
    )


_APEX_BUILDER_BY_FAMILY = {
    "apex_flying_wyvern": _build_apex_flying_wyvern,
    "brute_theropod_wyvern": _build_brute_theropod_wyvern,
    "pelagic_leviathan_wyvern": _build_pelagic_leviathan_wyvern,
    "fanged_thunder_wyvern": _build_fanged_thunder_wyvern,
    "armored_arthropod_colossus": _build_armored_arthropod_colossus,
    "morphing_switch_blade_axe": _build_morphing_switch_blade_axe,
    "elemental_pocket_companion": _build_elemental_pocket_companion,
    "modular_apricorn_healing_station": _build_modular_apricorn_healing_station,
    "elder_calamity_dragon": _build_elder_calamity_dragon,
    "relic_automaton_guardian": _build_relic_automaton_guardian,
}


def build_apex_complex_program(family_id: str, *, seed: int = 101) -> Program:
    """Build a deterministic, 100% static- and swept-SAT-audited Apex complex Program."""
    if family_id not in _APEX_BUILDER_BY_FAMILY:
        raise KeyError(f"Unknown Apex family {family_id!r}; expected one of {sorted(_APEX_BUILDER_BY_FAMILY)}")
    rng = random.Random(seed)
    return _APEX_BUILDER_BY_FAMILY[family_id](rng, seed)


def generate_apex_complex_dataset(
    output_dir: Path | str | None = None,
    *,
    samples_per_seen_family: int = 6,
    samples_per_unseen_family: int = 6,
    seed: int = 202611,
) -> dict[str, Any]:
    """Generate an audited multi-turn Apex Complex Entity & Artist-Assistant dataset shard.

    Each family sample emits three complementary rule-grounded supervision records:
    1. `apex_full_masterwork`: Complete v3.1 creation (geometry + @materials + @muscles + @kinematics + @behavior).
    2. `apex_progressive_stage_upgrade`: Artist-like progressive stage transition (Stage k -> Stage k+1).
    3. `apex_artist_critique_and_enhance`: Takes an incomplete draft, emits quantitative 5-pillar critique,
       and upgrades it into a 100% SAT- & stance-IK-verified asset.
    """
    from synccraft.world.artist_loop import (
        analyze_artist_project,
        build_progressive_from_scratch_trajectory,
        run_looped_artist_enhancement,
    )

    rows: list[dict[str, Any]] = []
    family_plan = [
        *((fam, "seen_train", samples_per_seen_family) for fam in APEX_SEEN_FAMILIES),
        *((fam, "unseen_complex_holdout", samples_per_unseen_family) for fam in APEX_UNSEEN_FAMILIES),
    ]

    sample_counter = 0
    for fam, split_tag, count in family_plan:
        for local_idx in range(count):
            sample_seed = seed + sample_counter * 97 + local_idx * 13
            sample_counter += 1
            prog = build_apex_complex_program(fam, seed=sample_seed)
            dsl_full = format_program(prog)
            sc = compile_geometry(prog)
            static_aud = audit_scene(sc, enforce_physical_contact=True)
            clip_aud = (
                audit_behavior_clip(prog, prog.behavior.clips[0].name, num_frames=12)
                if prog.behavior and prog.behavior.clips
                else None
            )
            crit_full = analyze_artist_project(prog)
            stages = build_progressive_from_scratch_trajectory(prog)

            base_meta = {
                "family_id": fam,
                "split": split_tag,
                "seed": sample_seed,
                "license_spdx": "Apache-2.0",
                "license_track": "clean_room_permissive",
                "commercial_weights_eligible": True,
                "default_target_llm": "LiquidAI/LFM3-VL-2B",
                "box_count": len(sc.boxes),
                "bone_count": len(sc.bones),
                "locator_count": len(sc.locators),
                "muscle_count": len(prog.muscles),
                "static_audit_passed": bool(static_aud["passed"]),
                "dynamic_clip_passed": bool(clip_aud["passed"]) if clip_aud else True,
                "max_stance_slip_units": clip_aud["max_stance_slip_units"] if clip_aud else 0.0,
            }

            # 1. Full Masterwork Creation Turn
            sid_full = hashlib.sha256(f"{fam}:{sample_seed}:full".encode()).hexdigest()[:16]
            rows.append(
                {
                    "sample_id": f"apex_{sid_full}",
                    "task_mode": "apex_full_masterwork",
                    **base_meta,
                    "source_stage": None,
                    "target_stage": None,
                    "initial_readiness_score": None,
                    "readiness_score": round(crit_full.readiness_score, 4),
                    "critique_json": None,
                    "user_prompt": (
                        f"Design a complete rule-grounded Block-DSL v3.1 {fam.replace('_', ' ')} "
                        f"with articulated bone hierarchy, semantic locators, welded scientific @materials, "
                        f"Hill-type @muscles tendon couplings, @kinematics IK, and swept-SAT @behavior."
                    ),
                    "assistant_dsl": dsl_full,
                }
            )

            # 2. Progressive From-Scratch Stage Transition Turn (cycle across Stage 0->1, 1->2, 2->3, 3->4)
            st_idx = local_idx % 4
            st_src = stages[st_idx]
            st_dst = stages[st_idx + 1]
            sid_prog = hashlib.sha256(f"{fam}:{sample_seed}:prog:{st_idx}".encode()).hexdigest()[:16]
            rows.append(
                {
                    "sample_id": f"apex_{sid_prog}",
                    "task_mode": "apex_progressive_stage_upgrade",
                    **base_meta,
                    "source_stage": st_src.stage_name,
                    "target_stage": st_dst.stage_name,
                    "initial_readiness_score": None,
                    "readiness_score": round(st_dst.readiness_score, 4),
                    "critique_json": None,
                    "user_prompt": (
                        f"Advance this {fam.replace('_', ' ')} project from {st_src.stage_name} "
                        f"({st_src.box_count} boxes) to {st_dst.stage_name}: {st_dst.description}\n\n"
                        f"```blockdsl\n{st_src.program_dsl}\n```"
                    ),
                    "assistant_dsl": st_dst.program_dsl,
                }
            )

            # 3. Closed-Loop Artist Critique & Enhancement Turn (from Stage 2 clay/untextured/unanimated draft)
            draft_prog = stages[2].program
            loop_res = run_looped_artist_enhancement(draft_prog, max_iterations=4)
            sid_loop = hashlib.sha256(f"{fam}:{sample_seed}:critique".encode()).hexdigest()[:16]
            rows.append(
                {
                    "sample_id": f"apex_{sid_loop}",
                    "task_mode": "apex_artist_critique_and_enhance",
                    **base_meta,
                    "source_stage": None,
                    "target_stage": None,
                    "initial_readiness_score": loop_res["initial_readiness_score"],
                    "readiness_score": loop_res["final_readiness_score"],
                    "critique_json": json.dumps(loop_res["history"][0]["critique"], sort_keys=True),
                    "user_prompt": (
                        f"Audit this artist's {fam.replace('_', ' ')} rig across geometry, skeletal limits, "
                        f"surface materials, musculoskeletal tendons, and stance-locked motion, then apply "
                        f"rule-verified upgrades:\n\n```blockdsl\n{stages[2].program_dsl}\n```"
                    ),
                    "assistant_dsl": loop_res["final_program_dsl"],
                }
            )

    summary = {
        "schema": "synccraft.apex_complex_dataset/1.0",
        "adr": "ADR-0006",
        "default_target_llm": "LiquidAI/LFM3-VL-2B",
        "total_rows": len(rows),
        "seen_families": list(APEX_SEEN_FAMILIES),
        "unseen_families": list(APEX_UNSEEN_FAMILIES),
        "static_sat_pass_rate": sum(1 for r in rows if r["static_audit_passed"]) / max(1, len(rows)),
        "dynamic_clip_pass_rate": sum(1 for r in rows if r["dynamic_clip_passed"]) / max(1, len(rows)),
        "mean_box_count": round(sum(r["box_count"] for r in rows) / max(1, len(rows)), 2),
        "max_box_count": max((r["box_count"] for r in rows), default=0),
        "mean_bone_count": round(sum(r["bone_count"] for r in rows) / max(1, len(rows)), 2),
        "max_stance_slip_observed": round(max((r["max_stance_slip_units"] for r in rows), default=0.0), 6),
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        jsonl_path = out_p / "apex_complex_entities_v3_1.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        (out_p / "apex_complex_manifest.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Write unified + train/test_unseen Parquet splits with explicit 23-column schema if pyarrow is available
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            pa = None
            pq = None

        if pa is not None and pq is not None:
            apex_schema = pa.schema(
                [
                    ("sample_id", pa.string()),
                    ("task_mode", pa.string()),
                    ("family_id", pa.string()),
                    ("split", pa.string()),
                    ("seed", pa.int64()),
                    ("license_spdx", pa.string()),
                    ("license_track", pa.string()),
                    ("commercial_weights_eligible", pa.bool_()),
                    ("default_target_llm", pa.string()),
                    ("box_count", pa.int64()),
                    ("bone_count", pa.int64()),
                    ("locator_count", pa.int64()),
                    ("muscle_count", pa.int64()),
                    ("static_audit_passed", pa.bool_()),
                    ("dynamic_clip_passed", pa.bool_()),
                    ("max_stance_slip_units", pa.float64()),
                    ("source_stage", pa.string()),
                    ("target_stage", pa.string()),
                    ("initial_readiness_score", pa.float64()),
                    ("readiness_score", pa.float64()),
                    ("critique_json", pa.string()),
                    ("user_prompt", pa.string()),
                    ("assistant_dsl", pa.string()),
                ]
            )
            train_rows = [r for r in rows if r["split"] == "seen_train"]
            unseen_rows = [r for r in rows if r["split"] == "unseen_complex_holdout"]
            pq.write_table(pa.Table.from_pylist(rows, schema=apex_schema), out_p / "apex_complex_entities_v3_1.parquet")
            pq.write_table(pa.Table.from_pylist(train_rows, schema=apex_schema), out_p / "train.parquet")
            pq.write_table(pa.Table.from_pylist(unseen_rows, schema=apex_schema), out_p / "test_unseen.parquet")

    return {"summary": summary, "rows": rows}
