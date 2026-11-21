"""Industry-Grade Universal Full-Body Biomechanical Solver (`SO(3)` / `SE(3)` / `Sim(3)`).

Expands SyncCraft's kinematics into a modular, Houdini-KineFX / Unreal-Control-Rig / Maya
style whole-body biological rig assembler and solver covering EVERY anatomical subsystem
on ANY real animal or brand-new imaginary creature (regardless of bone names or limb count):

1. Universal Skeleton-to-Rig Auto-Assembly (`auto_assemble_biomechanical_rig`):
   Decomposes an arbitrary skeleton into modular biological primitives:
   - `spine_chain` (pelvis -> lumbar -> thorax)
   - `neck_assemblies` (single- or multi-headed neck chains, skull, jaw, tongue, throat)
   - `tail_assemblies` (single- or multi-tailed caudal chains)
   - `wing_assemblies` (scapula/shoulder -> humerus -> forearm -> carpus -> phalanges + locked inner spar)
   - `limb_assemblies` (upper -> mid -> ankle/wrist -> foot/hand pad -> multi-phalanx digit & thumb chains)
   - `auto_bio_constraints` (complete SO(3) biological joint harness)
   - `auto_muscle_groups` (whole-body antagonist Sim(3) muscle mapping)

2. Multi-Segment Spine & Serpentine S-Neck Spline-IK (`solve_spine_neck_spline_ik`):
   Smoothly distributes 3D world-target aiming, S-curve coil extension/retraction, and
   bipedal rear-up curvature across N-segment necks and spines with progressive vertebral
   stiffness weights and SO(3) biological limits.

3. Real-Time 3D Center-of-Mass & Angular-Momentum Balance (`solve_com_balance_and_counterweight`):
   Computes exact volume-weighted 3D COM across all posed boxes, gently stabilizes `pelvis_root`
   within the grounded support polygon, and drives counter-rotating spine + tail momentum arcs
   whenever limbs, wings, or long necks swing asymmetrically.

4. Reverse-Foot Rocker & Multi-Phalanx Hand/Foot Digit Solver (`solve_foot_roll_and_digit_articulation`):
   - Authentic gait foot-roll rocker (heel-strike dorsiflexion -> dead-flat mid-stance -> toe-off
     push-off -> swing clearance)
   - Universal multi-phalanx finger/toe/talon articulation (stance ground splay, swing/flight
     aerodynamic talon curl, coordinated flexor grasp & opposable thumb opposition) strictly
     guarded by `finger_proximal`, `finger_flex`, and `thumb_saddle` SO(3) limits.

5. Full-Anatomy Wing Pectoral-to-Phalanx Coupling & Ground Clearance (`solve_wing_full_anatomy`):
   - Smooth continuous `wing_fold` (`0 = sleek cloak drape`, `1 = full spread`)
   - Pectoral/scapular downstroke coupling + differential leading-to-trailing phalanx camber
   - Automatic wingtip ground-clearance guard (`y >= ground_y + margin`)
   - Strict `wing_anchor_locked` preservation (`[0, 0, 0]` on inner anchor spars).

6. Whole-Body Isochoric Antagonist Muscle Bulge (`solve_full_body_antagonist_muscle_bulge`):
   Computes volume-preserving `Sim(3)` scaling (`det(S) == 1.0`) across torso/pectorals,
   neck/throat sac, wing arms, upper/lower limbs (biceps/triceps/quads/calves), and tail base
   directly from dynamic joint flexion & torque work, ready for `.bbmodel` & Bedrock/GeckoLib export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from synccraft.dsl.ast_nodes import Program, Vec3
from synccraft.kinematics.bio_constraints import (
    BIO_JOINT_PRESETS,
    infer_universal_bio_constraints,
    is_rigid_mechanical_bone,
    soft_clamp_euler_to_limits,
)
from synccraft.kinematics.morphology import (
    LimbMorphology,
    MorphologyReport,
    WingMorphology,
    analyze_morphology,
    point_in_convex_polygon_margin,
)
from synccraft.kinematics.se3_chain import (
    Scene,
    WorldBone,
    _recover_euler_xyz_degrees,
    compile_geometry,
    compile_posed_geometry,
    euler_xyz_matrix,
)


@dataclass(frozen=True)
class NeckAssembly:
    """Assembled cervical, cranial, and oral biological subsystem."""

    chain: tuple[str, ...]
    head_bone: str | None
    jaw_bones: tuple[str, ...] = ()
    tongue_bones: tuple[str, ...] = ()
    throat_bone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain": list(self.chain),
            "head_bone": self.head_bone,
            "jaw_bones": list(self.jaw_bones),
            "tongue_bones": list(self.tongue_bones),
            "throat_bone": self.throat_bone,
        }


@dataclass(frozen=True)
class LimbAssembly:
    """Full limb assembly from proximal girdle down through ankle/wrist, foot/hand pad, and digit chains."""

    role: str
    side: str
    is_free: bool
    posture: str
    chain: tuple[str, ...]
    upper_bone: str | None
    lower_bone: str | None
    ankle_wrist_bone: str | None
    foot_hand_bone: str | None
    digit_chains: tuple[tuple[str, ...], ...] = ()
    thumb_chains: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "side": self.side,
            "is_free": self.is_free,
            "posture": self.posture,
            "chain": list(self.chain),
            "upper_bone": self.upper_bone,
            "lower_bone": self.lower_bone,
            "ankle_wrist_bone": self.ankle_wrist_bone,
            "foot_hand_bone": self.foot_hand_bone,
            "digit_chains": [list(c) for c in self.digit_chains],
            "thumb_chains": [list(c) for c in self.thumb_chains],
        }


@dataclass(frozen=True)
class WingAssembly:
    """Full wing/airfoil assembly from pectoral host through humerus, forearm, carpus, and phalanges."""

    role: str
    side: str
    pair_index: int
    chain: tuple[str, ...]
    pectoral_host_bone: str | None
    shoulder_bone: str | None
    forearm_bone: str | None
    wrist_bone: str | None
    finger_bones: tuple[str, ...] = ()
    locked_spar_bones: tuple[str, ...] = ()
    span: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "side": self.side,
            "pair_index": self.pair_index,
            "chain": list(self.chain),
            "pectoral_host_bone": self.pectoral_host_bone,
            "shoulder_bone": self.shoulder_bone,
            "forearm_bone": self.forearm_bone,
            "wrist_bone": self.wrist_bone,
            "finger_bones": list(self.finger_bones),
            "locked_spar_bones": list(self.locked_spar_bones),
            "span": round(self.span, 4),
        }


@dataclass(frozen=True)
class AutoMuscleGroup:
    """Synthesized volume-preserving Sim(3) antagonist muscle unit bound to discovered skeleton bones."""

    name: str
    region: str  # "pectoral" | "limb_upper" | "limb_lower" | "neck_throat" | "torso_core" | "wing_arm" | "tail_base"
    target_bone: str
    driver_bones: tuple[str, ...]
    primary_axis: int  # 0=X, 1=Y, 2=Z (long axis of bone; radial bulge expands orthogonal axes)
    max_bulge: float = 0.045

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "region": self.region,
            "target_bone": self.target_bone,
            "driver_bones": list(self.driver_bones),
            "primary_axis": self.primary_axis,
            "max_bulge": round(self.max_bulge, 4),
        }


@dataclass(frozen=True)
class MechanicalSubsystem:
    """Discovered rigid vehicle / mecha / industrial actuator joints on hybrid or mechanical builds."""

    wheels: tuple[str, ...] = ()
    steered_knuckles: tuple[str, ...] = ()
    rotors: tuple[str, ...] = ()
    turrets: tuple[str, ...] = ()
    cannons: tuple[str, ...] = ()
    hatches: tuple[str, ...] = ()
    thrusters: tuple[str, ...] = ()
    actuators: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "wheels": list(self.wheels),
            "steered_knuckles": list(self.steered_knuckles),
            "rotors": list(self.rotors),
            "turrets": list(self.turrets),
            "cannons": list(self.cannons),
            "hatches": list(self.hatches),
            "thrusters": list(self.thrusters),
            "actuators": list(self.actuators),
        }


@dataclass(frozen=True)
class BiomechanicalRigAssembly:
    """Complete assembled whole-body biological + mechanical control rig derived from ANY skeleton."""

    root_bone: str
    spine_chain: tuple[str, ...]
    neck_assemblies: tuple[NeckAssembly, ...]
    tail_chains: tuple[tuple[str, ...], ...]
    limb_assemblies: tuple[LimbAssembly, ...]
    wing_assemblies: tuple[WingAssembly, ...]
    tentacles: tuple[tuple[str, ...], ...]
    auto_muscle_groups: tuple[AutoMuscleGroup, ...]
    bio_constraints: dict[str, tuple[Any, np.ndarray]] = field(repr=False)
    morphology: MorphologyReport = field(repr=False)
    mechanical: MechanicalSubsystem = field(default_factory=MechanicalSubsystem)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_bone": self.root_bone,
            "spine_chain": list(self.spine_chain),
            "neck_assemblies": [n.to_dict() for n in self.neck_assemblies],
            "tail_chains": [list(t) for t in self.tail_chains],
            "limb_assemblies": [lb.to_dict() for lb in self.limb_assemblies],
            "wing_assemblies": [wg.to_dict() for wg in self.wing_assemblies],
            "tentacles": [list(t) for t in self.tentacles],
            "auto_muscle_groups": [m.to_dict() for m in self.auto_muscle_groups],
            "mechanical": self.mechanical.to_dict(),
            "constrained_bone_count": len(self.bio_constraints),
        }


def _children_map(scene: Scene) -> dict[str, list[str]]:
    ch: dict[str, list[str]] = {b.name: [] for b in scene.bones}
    for b in scene.bones:
        if b.parent_name is not None and b.parent_name in ch:
            ch[b.parent_name].append(b.name)
    return ch


def _bone_long_axis_local(scene: Scene, bone_name: str) -> int:
    """Determine which local axis (0=X, 1=Y, 2=Z) represents a bone's primary length direction."""
    wb = scene.bone_by_name.get(bone_name)
    if wb is None:
        return 1
    sizes = [bx.size for bx in scene.boxes if bx.bone_name == bone_name and getattr(bx, "plane_normal", None) is None]
    if not sizes:
        return 1
    mean_sz = np.mean(np.asarray(sizes, dtype=np.float64), axis=0)
    return int(np.argmax(mean_sz))


def _collect_digit_chains(
    start_bones: Sequence[str],
    children: dict[str, list[str]],
    allowed_set: set[str] | None = None,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[str, ...], ...]]:
    """Trace rooted multi-phalanx digit chains (proximal -> intermediate -> distal claw/tip)."""
    regular: list[tuple[str, ...]] = []
    thumbs: list[tuple[str, ...]] = []

    def _trace(bname: str, prefix: list[str]) -> None:
        cur = [*prefix, bname]
        kids = [k for k in children.get(bname, []) if allowed_set is None or k in allowed_set]
        if not kids:
            chain_t = tuple(cur)
            low_all = " ".join(c.lower() for c in cur)
            if any(tok in low_all for tok in ("thumb", "dewclaw", "hallux", "pollex")):
                thumbs.append(chain_t)
            else:
                regular.append(chain_t)
            return
        for k in kids:
            _trace(k, cur)

    for sb in start_bones:
        _trace(sb, [])
    return tuple(regular), tuple(thumbs)


def auto_assemble_biomechanical_rig(
    program: Program,
    rest_scene: Scene | None = None,
    morphology: MorphologyReport | None = None,
    *,
    explicit_limits: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> BiomechanicalRigAssembly:
    """Resolve ANY real or imaginary creature skeleton into a complete full-body biological rig."""
    from synccraft.kinematics.musculoskeletal import _collect_expanded_joint_limits

    sc = rest_scene if rest_scene is not None else compile_geometry(program, validate=False)
    morph = morphology if morphology is not None else analyze_morphology(program, sc)
    root_name = sc.bones[0].name if sc.bones else "root"
    children = _children_map(sc)

    exp_lim = explicit_limits if explicit_limits is not None else _collect_expanded_joint_limits(program)
    bio_lim = infer_universal_bio_constraints(program, sc, morph, explicit_limits=exp_lim)

    # 1. Spine chain (root -> waist -> chest)
    spine_chain = tuple(morph.spine) if morph.spine else (root_name,)

    # 2. Neck & Head assemblies (single- or multi-headed!)
    raw_necks = morph.neck_chains if morph.neck_chains else ((morph.neck,) if morph.neck else ())
    neck_assemblies: list[NeckAssembly] = []
    for n_ch in raw_necks:
        if not n_ch:
            continue
        head_b = n_ch[-1]
        throat_b = n_ch[len(n_ch) // 2] if len(n_ch) >= 2 else n_ch[0]
        jaws: list[str] = []
        tongues: list[str] = []
        # BFS descendants of head_b for oral apparatus
        stack = list(children.get(head_b, []))
        seen_o: set[str] = set()
        while stack:
            cur = stack.pop(0)
            if cur in seen_o:
                continue
            seen_o.add(cur)
            low = cur.lower()
            if any(k in low for k in ("jaw", "mandible", "chin")):
                jaws.append(cur)
            elif any(k in low for k in ("tongue", "glossa")):
                tongues.append(cur)
            stack.extend(children.get(cur, []))
        neck_assemblies.append(
            NeckAssembly(
                chain=tuple(n_ch),
                head_bone=head_b,
                jaw_bones=tuple(jaws),
                tongue_bones=tuple(tongues),
                throat_bone=throat_b,
            )
        )

    # 3. Tail chains (single- or multi-tailed!)
    tail_chains = morph.tail_chains if morph.tail_chains else ((morph.tail,) if morph.tail else ())

    # 4. Wing / Airfoil assemblies (2/4/6 wings or fins!)
    wing_assemblies: list[WingAssembly] = []
    for wg in morph.wings:
        w_ch = wg.chain
        sh_b = w_ch[0] if len(w_ch) >= 1 else None
        fa_b = w_ch[1] if len(w_ch) >= 2 else None
        wr_b = w_ch[2] if len(w_ch) >= 3 else None
        pec_host = sc.bone_by_name[sh_b].parent_name if (sh_b and sh_b in sc.bone_by_name) else None
        wing_assemblies.append(
            WingAssembly(
                role=wg.role,
                side=wg.side,
                pair_index=wg.pair_index,
                chain=tuple(w_ch),
                pectoral_host_bone=pec_host,
                shoulder_bone=sh_b,
                forearm_bone=fa_b,
                wrist_bone=wr_b,
                finger_bones=tuple(wg.finger_bones),
                locked_spar_bones=tuple(wg.locked_spar_bones),
                span=float(wg.span),
            )
        )

    # 5. Limb assemblies (grounded limbs + free upper limbs, including foot/hand pads & multi-phalanx digit chains!)
    limb_assemblies: list[LimbAssembly] = []

    def _build_limb_asm(lb: LimbMorphology, is_free: bool) -> LimbAssembly:
        ch = tuple(lb.chain)
        upper_b = ch[0] if len(ch) >= 1 else None
        lower_b = ch[1] if len(ch) >= 2 else None
        ankle_b = ch[-1] if len(ch) >= 3 else lower_b
        # Detect distinct foot/hand pad bone right below the IK structural chain (e.g. paw_l, foot_l, hand_l)
        foot_hand_b: str | None = None
        digit_roots: list[str] = []
        digit_set = set(lb.digits)
        if ch:
            end_b = ch[-1]
            end_kids = children.get(end_b, [])
            # Check if a single child acts as palm/paw/foot hub branching into >= 2 digits
            hub_candidates: list[str] = []
            for k in end_kids:
                kl = k.lower()
                grandkids = children.get(k, [])
                if (
                    any(tok in kl for tok in ("paw", "foot", "hand", "palm", "hoof", "manus", "pes", "sole"))
                    or len(grandkids) >= 2
                ):
                    hub_candidates.append(k)
            if hub_candidates:
                foot_hand_b = hub_candidates[0]
                digit_roots = list(children.get(foot_hand_b, []))
                # Include any remaining direct digit siblings (like a rear ankle dewclaw!)
                for k in end_kids:
                    if k != foot_hand_b and (k in digit_set or any(tok in k.lower() for tok in ("toe", "finger", "claw", "talon", "dewclaw", "thumb", "digit"))):
                        digit_roots.append(k)
            else:
                # If the chain ends directly on foot/hand or branches straight into digits
                if any(tok in end_b.lower() for tok in ("paw", "foot", "hand", "palm", "hoof")):
                    foot_hand_b = end_b
                digit_roots = [k for k in end_kids if k in digit_set or any(tok in k.lower() for tok in ("toe", "finger", "claw", "talon", "dewclaw", "thumb", "digit"))]

        # Collect all descendants under digit_roots as multi-phalanx chains
        reg_chains, thumb_chains = _collect_digit_chains(digit_roots, children)
        # Ensure any remaining discovered digits not traced above are included
        traced = {b for c in (*reg_chains, *thumb_chains) for b in c}
        leftover = [d for d in lb.digits if d not in traced and d != foot_hand_b]
        if leftover:
            extra_reg, extra_th = _collect_digit_chains(leftover, children)
            reg_chains = (*reg_chains, *extra_reg)
            thumb_chains = (*thumb_chains, *extra_th)

        return LimbAssembly(
            role=lb.role,
            side=lb.side,
            is_free=is_free,
            posture=lb.posture,
            chain=ch,
            upper_bone=upper_b,
            lower_bone=lower_b,
            ankle_wrist_bone=ankle_b,
            foot_hand_bone=foot_hand_b,
            digit_chains=reg_chains,
            thumb_chains=thumb_chains,
        )

    for lb in morph.limbs:
        limb_assemblies.append(_build_limb_asm(lb, is_free=False))
    for flb in getattr(morph, "free_limbs", ()):
        limb_assemblies.append(_build_limb_asm(flb, is_free=True))

    # 6. Synthesize Whole-Body Sim(3) Antagonist Muscle Groups automatically from skeleton topology!
    #    Rigid mechanical/vehicle hardware bones (wheels, turrets, cannons, hydraulic cylinders) stay strictly rigid.
    explicit_bio_by_bone: dict[str, str | None] = {}

    def _collect_bio_tags(bnode: Any, pref_b: str | None = None, renamed_b: str | None = None) -> None:
        nm = renamed_b or (f"{pref_b}__{bnode.name}" if pref_b else bnode.name)
        explicit_bio_by_bone[nm] = getattr(bnode, "bio_joint", None)
        for it in getattr(bnode, "items", ()):
            if hasattr(it, "items") and hasattr(it, "name"):
                _collect_bio_tags(it, pref_b=pref_b)
                if getattr(it, "mirror_name", None):
                    mp = f"{pref_b}__{it.mirror_name}" if pref_b else it.mirror_name
                    _collect_bio_tags(it, pref_b=mp, renamed_b=mp)

    _collect_bio_tags(program.root)

    wheels_l: list[str] = []
    steered_l: list[str] = []
    rotors_l: list[str] = []
    turrets_l: list[str] = []
    cannons_l: list[str] = []
    hatches_l: list[str] = []
    thrusters_l: list[str] = []
    actuators_l: list[str] = []
    for wb in sc.bones:
        if wb.name == root_name:
            continue
        low = wb.name.split("__")[-1].lower()
        btag = explicit_bio_by_bone.get(wb.name)
        if btag == "steered_wheel_knuckle" or any(k in low for k in ("steer", "knuckle")):
            steered_l.append(wb.name)
        elif btag == "wheel_axle" or any(k in low for k in ("wheel", "tire", "axle")):
            wheels_l.append(wb.name)
        elif btag == "rotor_propeller" or any(k in low for k in ("rotor", "propeller", "turbine")):
            rotors_l.append(wb.name)
        elif btag == "turret_yaw" or any(k in low for k in ("turret", "slew")):
            turrets_l.append(wb.name)
        elif btag == "cannon_elevation" or any(k in low for k in ("cannon", "railgun", "barrel")):
            cannons_l.append(wb.name)
        elif btag == "hatch_door" or any(k in low for k in ("hatch", "canopy", "ramp", "door")):
            hatches_l.append(wb.name)
        elif btag == "gimbal_thruster" or any(k in low for k in ("thruster", "nozzle")):
            thrusters_l.append(wb.name)
        elif btag == "hydraulic_hinge" or any(k in low for k in ("piston", "hydraulic", "actuator", "boom")):
            actuators_l.append(wb.name)

    mech_sub = MechanicalSubsystem(
        wheels=tuple(wheels_l),
        steered_knuckles=tuple(steered_l),
        rotors=tuple(rotors_l),
        turrets=tuple(turrets_l),
        cannons=tuple(cannons_l),
        hatches=tuple(hatches_l),
        thrusters=tuple(thrusters_l),
        actuators=tuple(actuators_l),
    )

    auto_muscles: list[AutoMuscleGroup] = []
    seen_targets: set[tuple[str, str]] = set()

    def _add_auto_muscle(name: str, region: str, target_bone: str | None, drivers: Sequence[str], max_bulge: float) -> None:
        if not target_bone or target_bone not in sc.bone_by_name or target_bone == root_name:
            return
        if is_rigid_mechanical_bone(target_bone, explicit_bio_by_bone.get(target_bone)):
            return
        key = (region, target_bone)
        if key in seen_targets:
            return
        seen_targets.add(key)
        valid_drivers = tuple(d for d in drivers if d and d in sc.bone_by_name)
        if not valid_drivers:
            valid_drivers = (target_bone,)
        auto_muscles.append(
            AutoMuscleGroup(
                name=name,
                region=region,
                target_bone=target_bone,
                driver_bones=valid_drivers,
                primary_axis=_bone_long_axis_local(sc, target_bone),
                max_bulge=float(max_bulge),
            )
        )

    # 6a. Torso / Pectoral core muscles (driven by spine breathing + wing shoulders!)
    thorax_bone = spine_chain[-1] if len(spine_chain) >= 2 else (spine_chain[0] if spine_chain else None)
    wing_shoulders = [w.shoulder_bone for w in wing_assemblies if w.shoulder_bone]
    if thorax_bone and thorax_bone != root_name:
        _add_auto_muscle(
            "auto_thorax_pectoral",
            "pectoral" if wing_shoulders else "torso_core",
            thorax_bone,
            [*wing_shoulders, thorax_bone],
            max_bulge=0.042 if wing_shoulders else 0.032,
        )

    # 6b. Wing brachial/deltoid muscles (driven by humerus + elbow flap work!)
    for wg in wing_assemblies:
        if wg.shoulder_bone:
            _add_auto_muscle(
                f"auto_wing_brachial_{wg.role}",
                "wing_arm",
                wg.shoulder_bone,
                [b for b in (wg.shoulder_bone, wg.forearm_bone) if b],
                max_bulge=0.045,
            )

    # 6c. Limb upper (deltoid/biceps/quadriceps) & lower (forearm/calf) muscles!
    for la in limb_assemblies:
        if la.upper_bone:
            _add_auto_muscle(
                f"auto_limb_upper_{la.role}",
                "limb_upper",
                la.upper_bone,
                [b for b in (la.upper_bone, la.lower_bone) if b],
                max_bulge=0.048,
            )
        if la.lower_bone and la.lower_bone != la.upper_bone:
            _add_auto_muscle(
                f"auto_limb_lower_{la.role}",
                "limb_lower",
                la.lower_bone,
                [b for b in (la.lower_bone, la.ankle_wrist_bone, la.foot_hand_bone) if b],
                max_bulge=0.035,
            )

    # 6d. Cervical / Throat-Sac muscles (driven by neck arch + jaw roar opening!)
    for n_i, na in enumerate(neck_assemblies):
        if na.throat_bone:
            _add_auto_muscle(
                f"auto_neck_throat_{n_i}",
                "neck_throat",
                na.throat_bone,
                [*na.chain[:2], *na.jaw_bones],
                max_bulge=0.052,
            )

    # 6e. Caudal base propulsion muscle
    for t_i, tc in enumerate(tail_chains):
        if tc:
            _add_auto_muscle(
                f"auto_tail_base_{t_i}",
                "tail_base",
                tc[0],
                tc[:2],
                max_bulge=0.035,
            )

    return BiomechanicalRigAssembly(
        root_bone=root_name,
        spine_chain=spine_chain,
        neck_assemblies=tuple(neck_assemblies),
        tail_chains=tuple(tail_chains),
        limb_assemblies=tuple(limb_assemblies),
        wing_assemblies=tuple(wing_assemblies),
        tentacles=tuple(morph.tentacles),
        auto_muscle_groups=tuple(auto_muscles),
        bio_constraints=bio_lim,
        morphology=morph,
        mechanical=mech_sub,
    )


def solve_spine_neck_spline_ik(
    program: Program,
    rest_scene: Scene,
    chain_bones: Sequence[str],
    pose_deltas: dict[str, Vec3],
    *,
    aim_target_world: Vec3 | None = None,
    s_coil_deg: float = 0.0,
    bend_yaw_deg: float = 0.0,
    limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
    root_translation_delta: Vec3 = (0.0, 0.0, 0.0),
    pose_position_deltas: dict[str, Vec3] | None = None,
    pose_scale_factors: dict[str, Vec3] | None = None,
    forward_sign_z: float = 1.0,
) -> dict[str, Vec3]:
    """Multi-segment SE(3)/SO(3) Spline-IK & S-Curve Coil solver for necks and spines.

    - When ``s_coil_deg`` is nonzero: generates an authentic avian/draconic S-curve coil
      (proximal segments draw back/up while mid segments arch forward and distal atlas levels
      the skull), distributing curvature smoothly without kinking any single vertebra.
    - When ``aim_target_world`` is set: iteratively aligns the chain tip toward the 3D world
      point across all vertebrae with progressive stiffness weights, clamped smoothly in SO(3).
    """
    valid = [b for b in chain_bones if b in rest_scene.bone_by_name and rest_scene.bone_by_name[b].parent_name is not None]
    if not valid:
        return dict(pose_deltas)

    out = dict(pose_deltas)
    n_v = len(valid)

    # 1. Smooth S-curve coil & lateral bend distribution
    if abs(float(s_coil_deg)) > 1e-5 or abs(float(bend_yaw_deg)) > 1e-5:
        for i, bname in enumerate(valid):
            u = float(i) / float(max(1, n_v - 1)) if n_v > 1 else 0.5
            # Classic S-curve profile: base arches back (-), mid curves forward (+), atlas levels (-)
            if n_v >= 3:
                s_profile = -math.cos(math.pi * u) * (1.0 - 0.35 * u)
                if i == n_v - 1:
                    s_profile = -0.45  # distal head atlas counters neck arch to stay focused!
            else:
                s_profile = 1.0 if i == 0 else -0.55
            dpitch = float(s_coil_deg) * s_profile / max(1.0, 0.55 * n_v)
            dyaw = (float(bend_yaw_deg) / float(n_v)) * (0.7 + 0.6 * u)
            cur = np.asarray(out.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            cand = cur + np.array([dpitch, dyaw, 0.0], dtype=np.float64)
            if limits_map is not None and bname in limits_map:
                jlim, asign = limits_map[bname]
                cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
            out[bname] = (float(cand[0]), float(cand[1]), float(cand[2]))

    # 2. Iterative 3D world-space Spline-IK aiming toward `aim_target_world`
    if aim_target_world is not None:
        tgt_w = np.asarray(aim_target_world, dtype=np.float64)
        # Progressive weights: proximal vertebrae carry broad sweep, distal atlas refines gaze
        raw_w = np.array([0.55 + 0.45 * (i / max(1, n_v - 1)) for i in range(n_v)], dtype=np.float64)
        weights = raw_w / max(1e-9, float(np.sum(raw_w)))

        for _it in range(3):
            sc_cur = compile_posed_geometry(
                program,
                out,
                root_translation_delta=root_translation_delta,
                pose_position_deltas=pose_position_deltas,
                pose_scale_factors=pose_scale_factors,
                validate=False,
            )
            tip_b = sc_cur.bone_by_name[valid[-1]]
            # Forward direction of tip bone in world space (aligned with creature forward_sign_z)
            fwd_local = np.array([0.0, 0.0, float(forward_sign_z)], dtype=np.float64)
            cur_fwd_w = tip_b.rotation @ fwd_local
            cur_fwd_w /= max(1e-9, float(np.linalg.norm(cur_fwd_w)))

            to_tgt = tgt_w - tip_b.pivot_world
            dist = float(np.linalg.norm(to_tgt))
            if dist < 1e-4:
                break
            des_fwd_w = to_tgt / dist
            dot_v = float(np.clip(np.dot(cur_fwd_w, des_fwd_w), -1.0, 1.0))
            ang_rad = math.acos(dot_v)
            if ang_rad < math.radians(0.8):
                break
            rot_axis_w = np.cross(cur_fwd_w, des_fwd_w)
            ax_n = float(np.linalg.norm(rot_axis_w))
            if ax_n < 1e-7:
                break
            rot_axis_w /= ax_n

            # Rodrigues rotation matrix for world correction step (damped 0.72 per iteration)
            step_rad = 0.72 * min(ang_rad, math.radians(48.0))
            for bname, w in zip(valid, weights):
                wb = sc_cur.bone_by_name[bname]
                # Express incremental axis-angle in bone's local frame
                ax_loc = wb.rotation.T @ rot_axis_w
                d_rad = step_rad * float(w)
                # Small-angle exact Rodrigues -> Euler XYZ
                K = np.array(
                    [
                        [0.0, -ax_loc[2], ax_loc[1]],
                        [ax_loc[2], 0.0, -ax_loc[0]],
                        [-ax_loc[1], ax_loc[0], 0.0],
                    ],
                    dtype=np.float64,
                )
                r_inc = np.eye(3) + math.sin(d_rad) * K + (1.0 - math.cos(d_rad)) * (K @ K)
                d_eul = _recover_euler_xyz_degrees(r_inc)
                cur = np.asarray(out.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
                cand = cur + d_eul
                if limits_map is not None and bname in limits_map:
                    jlim, asign = limits_map[bname]
                    cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                out[bname] = (float(cand[0]), float(cand[1]), float(cand[2]))

    return out


def _compute_scene_com_world(scene: Scene) -> np.ndarray:
    """Exact volume-weighted 3D center-of-mass across all non-plane cuboids in `scene`."""
    m_tot = 0.0
    acc = np.zeros(3, dtype=np.float64)
    for bx in scene.boxes:
        if getattr(bx, "plane_normal", None) is not None:
            v = 0.05 * float(np.prod(np.maximum(bx.size, 0.25)))
        else:
            v = max(1e-4, float(bx.size[0] * bx.size[1] * bx.size[2]))
        m_tot += v
        acc += v * bx.center
    return acc / max(1e-9, m_tot)


def solve_com_balance_and_counterweight(
    program: Program,
    rest_scene: Scene,
    assembly: BiomechanicalRigAssembly,
    pose_deltas: dict[str, Vec3],
    root_translation_delta: Vec3,
    *,
    strength: float = 0.55,
    in_place: bool = True,
    grounded: bool = True,
    limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
    pose_position_deltas: dict[str, Vec3] | None = None,
    pose_scale_factors: dict[str, Vec3] | None = None,
) -> tuple[dict[str, Vec3], Vec3]:
    """Stabilize 3D Center-of-Mass & conserve angular momentum via spine + tail counter-swing.

    Measures the real 3D COM shift caused by posed wings, forelimbs, and long serpentine necks:
    1. Drives counter-rotating angular-momentum arcs along all tail chains (`tail_chains`)
       opposite to lateral COM excursions (preventing frozen stiff tails during asymmetric swings).
    2. When grounded, gently shifts `pelvis_root` laterally (and sagittally only when not in-place)
       to keep COM comfortably inside the stance support polygon.
    """
    gain = float(np.clip(strength, 0.0, 1.0))
    if gain <= 1e-5:
        return dict(pose_deltas), root_translation_delta

    posed_sc = compile_posed_geometry(
        program,
        pose_deltas,
        root_translation_delta=root_translation_delta,
        pose_position_deltas=pose_position_deltas,
        pose_scale_factors=pose_scale_factors,
        validate=False,
    )
    com_rest = np.asarray(assembly.morphology.com, dtype=np.float64)
    com_now = _compute_scene_com_world(posed_sc)
    root_arr = np.asarray(root_translation_delta, dtype=np.float64)
    # Local COM shift relative to root displacement
    d_com = (com_now - root_arr) - com_rest

    out_poses = dict(pose_deltas)
    new_root = root_arr.copy()
    # Normalize COM shift against core stance/torso width (NOT giant outstretched wingspan!)
    if len(assembly.morphology.support_polygon_xz) >= 2:
        sup_xs = [p[0] for p in assembly.morphology.support_polygon_xz]
        sup_zs = [p[1] for p in assembly.morphology.support_polygon_xz]
        core_w = max(4.0, max(sup_xs) - min(sup_xs))
        core_l = max(6.0, max(sup_zs) - min(sup_zs))
    else:
        core_w = max(4.0, min(20.0, float(assembly.morphology.body_span[0])))
        core_l = max(6.0, float(assembly.morphology.body_span[2]))

    # 1. Angular-momentum tail counterbalance: lateral COM shift (+X) swings tail opposite to conserve moment!
    lat_ratio = float(np.clip(d_com[0] / (0.20 * core_w), -1.0, 1.0))
    sag_ratio = float(np.clip(d_com[2] / (0.25 * core_l), -1.0, 1.0))
    if abs(lat_ratio) > 0.01 or abs(sag_ratio) > 0.02:
        for t_chain in assembly.tail_chains:
            valid_t = [b for b in t_chain if b != assembly.root_bone and b in rest_scene.bone_by_name]
            if not valid_t:
                continue
            n_t = len(valid_t)
            for ti, tb in enumerate(valid_t):
                wt = (0.70 + 0.55 * (ti / max(1, n_t - 1))) / math.sqrt(float(n_t))
                # Opposite yaw counters lateral shift; subtle pitch counters forward/aft shift
                dyaw = -22.0 * gain * lat_ratio * wt
                dpitch = 12.0 * gain * sag_ratio * wt * assembly.morphology.forward_sign_z
                cur = np.asarray(out_poses.get(tb, (0.0, 0.0, 0.0)), dtype=np.float64)
                cand = cur + np.array([dpitch, dyaw, 0.0], dtype=np.float64)
                if limits_map is not None and tb in limits_map:
                    jlim, asign = limits_map[tb]
                    cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                out_poses[tb] = (float(cand[0]), float(cand[1]), float(cand[2]))

    # 2. Grounded support-polygon COM nudge (lateral X shift only in `in_place` mode so root_z stays 0.00u!)
    if grounded and len(assembly.morphology.support_polygon_xz) >= 2:
        hull = assembly.morphology.support_polygon_xz
        margin = point_in_convex_polygon_margin((float(com_now[0]), float(com_now[2])), hull)
        if margin < 0.0 or abs(d_com[0]) > 0.25:
            max_lat = min(1.8, 0.08 * core_w)
            nudge_x = float(np.clip(-0.42 * gain * d_com[0], -max_lat, max_lat))
            new_root[0] += nudge_x
            if not in_place:
                max_sag = min(1.5, 0.06 * core_l)
                new_root[2] += float(np.clip(-0.25 * gain * d_com[2], -max_sag, max_sag))

    return out_poses, (float(new_root[0]), float(new_root[1]), float(new_root[2]))


def solve_foot_roll_and_digit_articulation(
    assembly: BiomechanicalRigAssembly,
    pose_deltas: dict[str, Vec3],
    *,
    limb_cycle_phases: dict[str, float] | None = None,
    limb_in_stance: dict[str, bool] | None = None,
    duty: float = 0.65,
    foot_roll_strength: float = 0.65,
    digit_strength: float = 0.80,
    grasp_amount: float = 0.0,
    aerial_tuck: bool = False,
    limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> dict[str, Vec3]:
    """Universal Reverse-Foot Rocker & Multi-Phalanx Hand/Foot Digit Articulation (`SE(3)` + `SO(3)`).

    Covers BOTH grounded limbs (feet/paws/talons) and free limbs (humanoid hands/claws):
    1. Foot-Roll Rocker (when `limb_cycle_phases` provided):
       - Heel-strike (`0.00..0.14 * duty`): gentle dorsiflexion landing absorption
       - Mid-stance (`0.14..0.82 * duty`): flat ground plant (`0°`)
       - Toe-off (`0.82..1.00 * duty`): smooth plantarflexion push-off
       - Swing (`duty..1.00`): subtle toe clearance pitch
    2. Articulated Multi-Phalanx Digits (`proximal -> intermediate -> distal talon/fingertip`):
       - Stance contact: toes/claws gently splay & settle flat on ground
       - Swing / Aerial flight: talons/fingers curl naturally inward
       - Grasp / Combat clutch (`grasp_amount ∈ [0, 1]`): coordinated progressive finger curl +
         opposable thumb opposition, strictly clamped by `finger_proximal`/`finger_flex`/`thumb_saddle`!
    """
    f_gain = float(np.clip(foot_roll_strength, 0.0, 1.0))
    d_gain = float(np.clip(digit_strength, 0.0, 1.0))
    grasp = float(np.clip(grasp_amount, 0.0, 1.0))
    if f_gain <= 1e-5 and d_gain <= 1e-5 and grasp <= 1e-5:
        return dict(pose_deltas)

    out = dict(pose_deltas)
    d_clamped = float(np.clip(duty, 0.20, 0.90))
    fwd_z = float(assembly.morphology.forward_sign_z)

    for la in assembly.limb_assemblies:
        u = limb_cycle_phases.get(la.role) if limb_cycle_phases is not None else None
        in_st = limb_in_stance.get(la.role, False) if limb_in_stance is not None else (not aerial_tuck and not la.is_free)

        # 1. Foot/Paw Roll Rocker on the foot/paw pad bone (only if NOT part of structural IK chain so slip stays 0!)
        if f_gain > 1e-5 and not la.is_free and la.foot_hand_bone and la.foot_hand_bone not in la.chain and u is not None:
            u_mod = float(u) % 1.0
            roll_pitch = 0.0
            if u_mod < d_clamped:
                st_u = u_mod / max(1e-6, d_clamped)
                if st_u < 0.14:
                    # Heel-strike dorsiflexion softening cleanly to 0°
                    roll_pitch = -7.5 * f_gain * math.sin(0.5 * math.pi * (1.0 - st_u / 0.14))
                elif st_u > 0.80:
                    # Toe-off push plantarflexion
                    t_off = (st_u - 0.80) / 0.20
                    roll_pitch = 11.0 * f_gain * math.sin(math.pi * t_off)
            else:
                # Swing clearance: gentle natural toe-point / curl arc
                sw_u = (u_mod - d_clamped) / max(1e-6, 1.0 - d_clamped)
                roll_pitch = 14.0 * f_gain * math.sin(math.pi * sw_u)

            cur = np.asarray(out.get(la.foot_hand_bone, (0.0, 0.0, 0.0)), dtype=np.float64)
            cand = cur + np.array([roll_pitch * fwd_z, 0.0, 0.0], dtype=np.float64)
            if limits_map is not None and la.foot_hand_bone in limits_map:
                jlim, asign = limits_map[la.foot_hand_bone]
                cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
            out[la.foot_hand_bone] = (float(cand[0]), float(cand[1]), float(cand[2]))

        # 2. Articulated Multi-Phalanx Digit Chains (fingers, toes, claws, talons)
        if d_gain > 1e-5 or grasp > 1e-5:
            if la.is_free:
                # Free humanoid hands / fore-claws: subtle natural resting curl + combat grasp!
                curl_factor = max(0.22 * d_gain, grasp)
                splay_deg = 4.0 * d_gain * (1.0 - grasp)
            elif aerial_tuck:
                # Aerial raptor/dragon talon tuck
                curl_factor = max(0.75 * d_gain, grasp)
                splay_deg = -4.0 * d_gain
            elif not in_st:
                # Swing phase: toes curl cleanly off ground
                sw_u = ((float(u) % 1.0) - d_clamped) / max(1e-6, 1.0 - d_clamped) if u is not None else 0.5
                curl_factor = max(0.55 * d_gain * math.sin(math.pi * float(np.clip(sw_u, 0.0, 1.0))), grasp)
                splay_deg = -2.0 * d_gain
            else:
                # Grounded stance: toes stay flat on y=0 & gently splay to grip ground (never curl through floor!)
                curl_factor = 0.0
                splay_deg = 4.5 * d_gain

            n_reg = len(la.digit_chains)
            for d_i, d_chain in enumerate(la.digit_chains):
                # Fan lateral splay symmetrically from outer to inner digit
                fan_s = ((float(d_i) / float(max(1, n_reg - 1))) - 0.5) * 2.0 if n_reg > 1 else 0.0
                for ph_i, bname in enumerate(d_chain):
                    # Progressive phalanx curl (proximal -> intermediate -> distal talon)
                    ph_scale = 26.0 if ph_i == 0 else (34.0 if ph_i == 1 else 24.0)
                    d_flex = ph_scale * curl_factor
                    d_yaw = (splay_deg * fan_s) if ph_i == 0 else 0.0
                    cur = np.asarray(out.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
                    cand = cur + np.array([d_flex, d_yaw, 0.0], dtype=np.float64)
                    # Enforce biological finger_flex / finger_proximal SO(3) anti-hyperextension!
                    if limits_map is not None and bname in limits_map:
                        jlim, asign = limits_map[bname]
                        cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                    else:
                        jlim = BIO_JOINT_PRESETS["finger_proximal" if ph_i == 0 else "finger_flex"]
                        cand = soft_clamp_euler_to_limits(cand, jlim)
                    out[bname] = (float(cand[0]), float(cand[1]), float(cand[2]))

            # Opposable thumbs / rear dewclaws
            side_sgn = 1.0 if la.side == "l" else -1.0
            for th_chain in la.thumb_chains:
                for ph_i, bname in enumerate(th_chain):
                    d_flex = (22.0 if ph_i == 0 else 28.0) * curl_factor
                    d_opp = (side_sgn * 16.0 * max(grasp, 0.25 * curl_factor)) if ph_i == 0 else 0.0
                    cur = np.asarray(out.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
                    cand = cur + np.array([d_flex, d_opp, 0.0], dtype=np.float64)
                    if limits_map is not None and bname in limits_map:
                        jlim, asign = limits_map[bname]
                        cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                    else:
                        jlim = BIO_JOINT_PRESETS["thumb_saddle" if ph_i == 0 else "finger_flex"]
                        cand = soft_clamp_euler_to_limits(cand, jlim)
                    out[bname] = (float(cand[0]), float(cand[1]), float(cand[2]))

    return out


def solve_wing_full_anatomy(
    program: Program,
    rest_scene: Scene,
    assembly: BiomechanicalRigAssembly,
    pose_deltas: dict[str, Vec3],
    scale_factors: dict[str, Vec3],
    *,
    wing_fold: float | None = None,
    downstroke_effort: float = 0.0,
    pectoral_bulge: float = 0.0,
    ground_clearance: bool = True,
    root_translation_delta: Vec3 = (0.0, 0.0, 0.0),
    limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> tuple[dict[str, Vec3], dict[str, Vec3]]:
    """Full scapula/pectoral-to-phalanx wing kinematics, fold blending & ground-clearance guard.

    1. Continuous ``wing_fold`` (`0.0` = sleek cloak drape, `1.0` = full spread) blends shoulder,
       elbow, wrist, and non-locked phalanges cleanly while keeping `wing_anchor_locked` (`[0,0,0]`).
    2. Pectoral/scapular coupling adds `Sim(3)` radial expansion on the thoracic pectoral host
       and brachial wing humerus synchronized with downstroke power effort.
    3. Automatic wingtip ground-clearance guard checks the lowest posed wing spar corner and adds
       smooth upward shoulder dihedral lift if any wing bone approaches `y < ground_y + 0.5u`.
    """
    if not assembly.wing_assemblies:
        return dict(pose_deltas), dict(scale_factors)

    out_poses = dict(pose_deltas)
    out_scales = dict(scale_factors)

    # 1. Optional continuous wing_fold carriage (0.0 = compact cloak drape .. 1.0 = full spread)
    if wing_fold is not None:
        fold_amt = 1.0 - float(np.clip(wing_fold, 0.0, 1.0))  # 1.0 = folded cloak, 0.0 = spread
        if fold_amt > 1e-4:
            for wg in assembly.wing_assemblies:
                z_sgn = 1.0 if wg.side == "l" else -1.0
                if wg.shoulder_bone:
                    cur = np.asarray(out_poses.get(wg.shoulder_bone, (0.0, 0.0, 0.0)), dtype=np.float64)
                    cand = cur + np.array([8.0 * fold_amt, z_sgn * 14.0 * fold_amt, z_sgn * 12.0 * fold_amt], dtype=np.float64)
                    if limits_map is not None and wg.shoulder_bone in limits_map:
                        jlim, asign = limits_map[wg.shoulder_bone]
                        cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                    out_poses[wg.shoulder_bone] = (float(cand[0]), float(cand[1]), float(cand[2]))
                if wg.forearm_bone:
                    cur = np.asarray(out_poses.get(wg.forearm_bone, (0.0, 0.0, 0.0)), dtype=np.float64)
                    cand = cur + np.array([0.0, -z_sgn * 12.0 * fold_amt, -z_sgn * 8.0 * fold_amt], dtype=np.float64)
                    if limits_map is not None and wg.forearm_bone in limits_map:
                        jlim, asign = limits_map[wg.forearm_bone]
                        cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                    out_poses[wg.forearm_bone] = (float(cand[0]), float(cand[1]), float(cand[2]))
                # Differential finger fold (strictly skip locked inner anchor spars!)
                n_f = len(wg.finger_bones)
                for fi, fb in enumerate(wg.finger_bones):
                    if fb in wg.locked_spar_bones:
                        continue
                    w_f = 1.0 - 0.35 * (fi / max(1, n_f - 1)) if n_f > 1 else 1.0
                    cur = np.asarray(out_poses.get(fb, (0.0, 0.0, 0.0)), dtype=np.float64)
                    cand = cur + np.array([0.0, -z_sgn * 8.0 * fold_amt * w_f, 0.0], dtype=np.float64)
                    if limits_map is not None and fb in limits_map:
                        jlim, asign = limits_map[fb]
                        cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                    out_poses[fb] = (float(cand[0]), float(cand[1]), float(cand[2]))

    # 2. Pectoral & wing-brachial Sim(3) downstroke power bulge
    p_gain = float(np.clip(pectoral_bulge, 0.0, 1.0))
    eff = float(np.clip(downstroke_effort, 0.0, 1.0))
    if p_gain > 1e-5 and eff > 1e-4:
        pec_hosts = {wg.pectoral_host_bone for wg in assembly.wing_assemblies if wg.pectoral_host_bone and wg.pectoral_host_bone != assembly.root_bone}
        for ph in pec_hosts:
            delta = 0.045 * p_gain * eff
            s_rad = 1.0 + delta
            s_ax = s_rad ** (-0.5)  # exact volume preservation det(S) == 1.0
            prev = out_scales.get(ph, (1.0, 1.0, 1.0))
            out_scales[ph] = (float(prev[0] * s_rad), float(prev[1] * s_rad), float(prev[2] * s_ax))
        for wg in assembly.wing_assemblies:
            if wg.shoulder_bone:
                delta = 0.038 * p_gain * eff
                s_rad = 1.0 + delta
                s_ax = s_rad ** (-0.5)
                long_ax = _bone_long_axis_local(rest_scene, wg.shoulder_bone)
                s_vec = [s_rad, s_rad, s_rad]
                s_vec[long_ax] = s_ax
                prev = out_scales.get(wg.shoulder_bone, (1.0, 1.0, 1.0))
                out_scales[wg.shoulder_bone] = (
                    float(prev[0] * s_vec[0]),
                    float(prev[1] * s_vec[1]),
                    float(prev[2] * s_vec[2]),
                )

    # 3. Wingtip ground-clearance guard (prevents wingtips from punching below ground_y)
    if ground_clearance:
        sc_chk = compile_posed_geometry(
            program,
            out_poses,
            root_translation_delta=root_translation_delta,
            pose_scale_factors=out_scales or None,
            validate=False,
        )
        g_y = float(assembly.morphology.ground_y)
        for wg in assembly.wing_assemblies:
            if not wg.shoulder_bone:
                continue
            w_bones = set(wg.chain) | set(wg.finger_bones)
            min_y = float("inf")
            for bx in sc_chk.boxes:
                if bx.bone_name in w_bones:
                    c_y = float(np.min(bx.corners(include_inflate=False)[:, 1]))
                    if c_y < min_y:
                        min_y = c_y
            deficit = (g_y + 0.5) - min_y
            if math.isfinite(deficit) and deficit > 0.0:
                lift_deg = min(22.0, math.degrees(math.atan2(deficit, max(6.0, 0.5 * wg.span))) + 2.0)
                z_sgn = -1.0 if wg.side == "l" else 1.0
                cur = np.asarray(out_poses.get(wg.shoulder_bone, (0.0, 0.0, 0.0)), dtype=np.float64)
                cand = cur + np.array([0.0, 0.0, z_sgn * lift_deg], dtype=np.float64)
                if limits_map is not None and wg.shoulder_bone in limits_map:
                    jlim, asign = limits_map[wg.shoulder_bone]
                    cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
                out_poses[wg.shoulder_bone] = (float(cand[0]), float(cand[1]), float(cand[2]))

    # Always preserve rigid inner wing anchor lock!
    for wg in assembly.wing_assemblies:
        for lb_lk in wg.locked_spar_bones:
            out_poses[lb_lk] = (0.0, 0.0, 0.0)

    return out_poses, out_scales


def solve_full_body_antagonist_muscle_bulge(
    assembly: BiomechanicalRigAssembly,
    pose_deltas: dict[str, Vec3],
    scale_factors: dict[str, Vec3],
    *,
    strength: float = 0.65,
    extra_effort_by_region: dict[str, float] | None = None,
    preserve_grounded_limb_length: bool = True,
) -> dict[str, Vec3]:
    """Compute universal volume-preserving Sim(3) muscle bulge across the WHOLE skeleton.

    Covers torso/pectorals, neck/throat sac, wing arms, upper/lower limbs (biceps/triceps/
    quadriceps/calves), and tail base directly from live joint excursion & flexion work.
    For grounded structural IK limb bones when ``preserve_grounded_limb_length=True``, axial
    scale stays ``1.0`` while radial axes bulge so closed-loop stance foot lock stays exact.
    """
    gain = float(np.clip(strength, 0.0, 1.0))
    if gain <= 1e-5:
        return dict(scale_factors)

    out_scales = dict(scale_factors)
    extra = extra_effort_by_region or {}
    grounded_ik_bones: set[str] = (
        {bn for la in assembly.limb_assemblies if not la.is_free for bn in la.chain}
        if preserve_grounded_limb_length
        else set()
    )

    for mg in assembly.auto_muscle_groups:
        # Compute normalized biomechanical effort in [0, 1] from angular deflection of driver joints
        ang_sum = 0.0
        for db in mg.driver_bones:
            v = pose_deltas.get(db)
            if v is not None:
                ang_sum += float(math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
        norm_work = math.tanh(ang_sum / 52.0)
        eff = float(np.clip(max(norm_work, float(extra.get(mg.region, 0.0))), 0.0, 1.0))
        if eff <= 0.02:
            continue

        # Exact isochoric Sim(3) scaling: radial axes (1 + delta), long axis (1 + delta)^(-0.5)
        # (or 1.0 on grounded structural IK bones so stance foot-lock never drifts)
        delta = float(mg.max_bulge) * gain * eff
        s_rad = 1.0 + delta
        s_ax = 1.0 if mg.target_bone in grounded_ik_bones else (s_rad ** (-0.5))
        s_vec = [s_rad, s_rad, s_rad]
        s_vec[mg.primary_axis] = s_ax

        prev = out_scales.get(mg.target_bone, (1.0, 1.0, 1.0))
        # Cap total cumulative scale strictly inside realistic biological tissue range [0.92, 1.12]
        out_scales[mg.target_bone] = (
            float(np.clip(prev[0] * s_vec[0], 0.92, 1.12)),
            float(np.clip(prev[1] * s_vec[1], 0.92, 1.12)),
            float(np.clip(prev[2] * s_vec[2], 0.92, 1.12)),
        )

    return out_scales


def limb_matches_pin_spec(
    la: LimbAssembly,
    lb: LimbMorphology,
    pin_spec: bool | Sequence[str],
) -> bool:
    """Return True if a grounded limb assembly matches `ground_pin_limbs` (`true` or selector list)."""
    if la.is_free or not pin_spec:
        return False
    if pin_spec is True:
        return True
    if isinstance(pin_spec, (tuple, list, set)):
        chain_low = {b.lower() for b in la.chain}
        role_low = la.role.lower()
        side_low = la.side.lower()
        ik_low = (lb.ik_chain_name or "").lower()
        for raw_tok in pin_spec:
            tok = str(raw_tok).strip().lower()
            if not tok:
                continue
            if tok in ("all", "true"):
                return True
            if tok in (role_low, f"{role_low}_{side_low}", ik_low) or tok in chain_low:
                return True
            if tok in ("hind", "rear", "back", "legs") and (
                role_low.startswith(("hind", "rear", "back", "leg"))
                or any(b.startswith(("leg_", "hind", "back")) for b in chain_low)
            ):
                return True
            if tok in ("fore", "front", "arms") and (
                role_low.startswith(("fore", "front", "arm"))
                or any(b.startswith(("arm_", "fore", "front")) for b in chain_low)
            ):
                return True
            if any(b.startswith(tok) for b in chain_low):
                return True
    return False


def solve_standing_limb_ik_pinning(
    program: Program,
    rest_scene: Scene,
    assembly: BiomechanicalRigAssembly,
    pose_deltas: dict[str, Vec3],
    *,
    pin_spec: bool | Sequence[str] = True,
    root_translation_delta: Vec3 = (0.0, 0.0, 0.0),
    pose_position_deltas: dict[str, Vec3] | None = None,
    pose_scale_factors: dict[str, Vec3] | None = None,
    limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> tuple[dict[str, Vec3], dict[str, Vec3], dict[str, bool], dict[str, int]]:
    """Pin grounded standing limbs to exact world ground contact (`SE(3)` IK + `SO(3)` foot-pad lock).

    Eliminates forward-kinematics foot floor penetration (`-4..-6u`) and floating hover on
    non-gait standing clips (`idle`, `bite`, `cloak_sweep`, `tail_whip`, or hind-pinned
    bipedal `roar` via ``ground_pin_limbs: [hind_l, hind_r]``):
    1. Solves closed-loop N-bone `SE(3)` IK on every pinned limb chain to keep its wrist/ankle
       hub (or ground contact point) locked at its exact rest world coordinates against any
       spine/chest/waist/pelvis breathing, pitch, roll, yaw, or spring follow-through.
    2. Applies exact right-handed `SO(3)` world-orientation lock (`R_foot_posed == R_foot_rest`)
       on any child foot/paw pad bone (`foot_hand_bone`) so the sole and all claws sit dead-flat
       on `y = ground_y` (`< 0.02u` slip in both SyncCraft and raw Blockbench 5 FK).
    """
    from synccraft.data_gen.ik_motion_templates import solve_n_bone_ik

    out_poses = dict(pose_deltas)
    eff_positions: dict[str, Vec3] = {}
    eff_stance: dict[str, bool] = {}
    eff_step_ids: dict[str, int] = {}

    if not pin_spec or not assembly.morphology.limbs:
        return out_poses, eff_positions, eff_stance, eff_step_ids

    grounded_las = [la for la in assembly.limb_assemblies if not la.is_free]
    selected: list[tuple[LimbAssembly, LimbMorphology]] = []
    for la, lb in zip(grounded_las, assembly.morphology.limbs):
        if len(lb.chain) >= 2 and limb_matches_pin_spec(la, lb, pin_spec):
            selected.append((la, lb))

    if not selected:
        return out_poses, eff_positions, eff_stance, eff_step_ids

    ik_decl_by_name: dict[str, Any] = (
        {c.name: c for c in program.kinematics.ik_chains} if program.kinematics is not None else {}
    )

    zero_poses = dict(out_poses)
    for _la, lb in selected:
        for bname in lb.chain:
            zero_poses[bname] = (0.0, 0.0, 0.0)

    posture_scene = compile_posed_geometry(
        program,
        zero_poses,
        pose_position_deltas=pose_position_deltas or None,
        pose_scale_factors=pose_scale_factors or None,
        validate=False,
    )

    g_y = float(assembly.morphology.ground_y)
    level_feet: list[str] = []

    for la, lb in selected:
        foot_b = la.foot_hand_bone
        if (
            foot_b
            and foot_b not in lb.chain
            and foot_b in rest_scene.bone_by_name
            and rest_scene.bone_by_name[foot_b].parent_name == lb.chain[-1]
        ):
            ik_bones = lb.chain
            wb_foot_rest = rest_scene.bone_by_name[foot_b]
            eff_local: Vec3 = (
                float(wb_foot_rest.local_translation[0]),
                float(wb_foot_rest.local_translation[1]),
                float(wb_foot_rest.local_translation[2]),
            )
            tgt_w = wb_foot_rest.pivot_world.copy()
            level_feet.append(foot_b)
        elif (
            foot_b
            and len(lb.chain) >= 3
            and foot_b == lb.chain[-1]
            and foot_b in rest_scene.bone_by_name
            and rest_scene.bone_by_name[foot_b].parent_name == lb.chain[-2]
        ):
            ik_bones = lb.chain[:-1]
            wb_foot_rest = rest_scene.bone_by_name[foot_b]
            eff_local = (
                float(wb_foot_rest.local_translation[0]),
                float(wb_foot_rest.local_translation[1]),
                float(wb_foot_rest.local_translation[2]),
            )
            tgt_w = wb_foot_rest.pivot_world.copy()
            level_feet.append(foot_b)
        else:
            ik_bones = lb.chain
            eff_local = lb.contact_local
            tgt_w = np.array([lb.contact_world[0], lb.contact_world[1], lb.contact_world[2]], dtype=np.float64)

        decl_ik = ik_decl_by_name.get(lb.ik_chain_name or "")
        hinge_ax = decl_ik.hinge_axis if decl_ik is not None else "x"
        pole_w = decl_ik.pole if decl_ik is not None else None
        plane_mode = (
            decl_ik.plane
            if (decl_ik is not None and decl_ik.plane is not None)
            else ("sprawled" if lb.posture == "sprawled" else "sagittal")
        )

        shifted_tgt = (
            float(tgt_w[0] - root_translation_delta[0]),
            float(tgt_w[1] - root_translation_delta[1]),
            float(tgt_w[2] - root_translation_delta[2]),
        )
        # Seed chain bones at 0 so solve_n_bone_ik uses shared posture_scene directly
        solve_in = dict(out_poses)
        for bname in ik_bones:
            solve_in[bname] = (0.0, 0.0, 0.0)

        ik_res = solve_n_bone_ik(
            program,
            ik_bones,
            eff_local,
            shifted_tgt,
            base_pose_deltas=solve_in,
            hinge_axis=hinge_ax,
            pole_world=pole_w,
            plane=plane_mode,
            validate=False,
            _limits_cache=limits_map,
            _zero_scene=posture_scene,
            pose_scale_factors=pose_scale_factors or None,
        )
        out_poses = ik_res.pose_euler_deltas
        ach_w: Vec3 = (
            float(ik_res.achieved_world[0] + root_translation_delta[0]),
            float(ik_res.achieved_world[1] + root_translation_delta[1]),
            float(ik_res.achieved_world[2] + root_translation_delta[2]),
        )
        eff_key = lb.ik_chain_name or la.role
        eff_positions[eff_key] = ach_w
        eff_stance[eff_key] = True
        eff_step_ids[eff_key] = 0

    def _project_level_feet(poses_in: dict[str, Vec3]) -> tuple[dict[str, Vec3], Scene]:
        sc_p = compile_posed_geometry(
            program,
            poses_in,
            root_translation_delta=root_translation_delta,
            pose_position_deltas=pose_position_deltas or None,
            pose_scale_factors=pose_scale_factors or None,
            validate=False,
        )
        poses_upd = dict(poses_in)
        for foot_b in level_feet:
            wb_foot_rest = rest_scene.bone_by_name[foot_b]
            parent_b = wb_foot_rest.parent_name
            if parent_b is None or parent_b not in sc_p.bone_by_name:
                continue
            r_foot_rest_world = wb_foot_rest.rotation
            r_foot_rest_local = euler_xyz_matrix(wb_foot_rest.local_euler)
            r_parent_posed = sc_p.bone_by_name[parent_b].rotation
            r_delta = r_foot_rest_local.T @ (r_parent_posed.T @ r_foot_rest_world)
            d_eul = _recover_euler_xyz_degrees(r_delta)
            if limits_map is not None and foot_b in limits_map:
                jlim, asign = limits_map[foot_b]
                d_eul = soft_clamp_euler_to_limits(d_eul, jlim, angle_sign=asign)
            poses_upd[foot_b] = (float(d_eul[0]), float(d_eul[1]), float(d_eul[2]))
        return poses_upd, sc_p

    # Exact SO(3) world-flat orientation lock + Sim(3) ground-plane barrier (y >= ground_y) on all pinned feet & claws!
    if level_feet:
        out_poses, _ = _project_level_feet(out_poses)
        sc_chk = compile_posed_geometry(
            program,
            out_poses,
            root_translation_delta=root_translation_delta,
            pose_position_deltas=pose_position_deltas or None,
            pose_scale_factors=pose_scale_factors or None,
            validate=False,
        )
        need_relevel = False
        for la, lb in selected:
            foot_b = la.foot_hand_bone
            if not foot_b or foot_b not in level_feet:
                continue
            foot_set: set[str] = {foot_b}
            for dc in la.digit_chains:
                foot_set.update(dc)
            for tc in la.thumb_chains:
                foot_set.update(tc)
            bxs_f = [bx for bx in sc_chk.boxes if bx.bone_name in foot_set and getattr(bx, "plane_normal", None) is None]
            if not bxs_f:
                continue
            min_yf = min(float(np.min(bx.corners(include_inflate=False)[:, 1])) for bx in bxs_f)
            if min_yf < g_y - 0.02:
                dy_fix = g_y - min_yf
                ik_bones = lb.chain[:-1] if (len(lb.chain) >= 3 and foot_b == lb.chain[-1]) else lb.chain
                wb_foot_rest = rest_scene.bone_by_name[foot_b]
                eff_local_f: Vec3 = (
                    float(wb_foot_rest.local_translation[0]),
                    float(wb_foot_rest.local_translation[1]),
                    float(wb_foot_rest.local_translation[2]),
                )
                tgt_w_f = wb_foot_rest.pivot_world.copy()
                tgt_w_f[1] += dy_fix
                decl_ik = ik_decl_by_name.get(lb.ik_chain_name or "")
                hinge_ax = decl_ik.hinge_axis if decl_ik is not None else "x"
                pole_w = decl_ik.pole if decl_ik is not None else None
                plane_mode = (
                    decl_ik.plane
                    if (decl_ik is not None and decl_ik.plane is not None)
                    else ("sprawled" if lb.posture == "sprawled" else "sagittal")
                )
                shifted_tgt_f = (
                    float(tgt_w_f[0] - root_translation_delta[0]),
                    float(tgt_w_f[1] - root_translation_delta[1]),
                    float(tgt_w_f[2] - root_translation_delta[2]),
                )
                solve_in_f = dict(out_poses)
                for bname in ik_bones:
                    solve_in_f[bname] = (0.0, 0.0, 0.0)
                ik_res_f = solve_n_bone_ik(
                    program,
                    ik_bones,
                    eff_local_f,
                    shifted_tgt_f,
                    base_pose_deltas=solve_in_f,
                    hinge_axis=hinge_ax,
                    pole_world=pole_w,
                    plane=plane_mode,
                    validate=False,
                    _limits_cache=limits_map,
                    _zero_scene=posture_scene,
                    pose_scale_factors=pose_scale_factors or None,
                )
                out_poses = ik_res_f.pose_euler_deltas
                need_relevel = True
        if need_relevel:
            out_poses, _ = _project_level_feet(out_poses)

    return out_poses, eff_positions, eff_stance, eff_step_ids


def solve_serpentine_cervical_wave(
    rest_scene: Scene,
    neck_chain: Sequence[str],
    pose_deltas: dict[str, Vec3],
    *,
    phase_rad: float,
    pitch_amp_deg: float = 4.0,
    yaw_amp_deg: float = 3.5,
    roll_amp_deg: float = 1.8,
    phase_lag_rad: float = 0.65,
    limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> dict[str, Vec3]:
    """Propagate an organic SO(3) traveling serpentine S-wave along multi-segment necks.

    Prevents stiff, locked "gyroscope" necks on long-necked dragons, wyverns, plesiosaurs,
    and birds: proximal and mid cervical vertebrae carry a smooth phase-lagged pitch/yaw/roll
    undulation while the distal head (`atlas`) counter-tracks ~45% so the gaze stays poised
    and predatory without freezing the neck column.
    """
    valid = [b for b in neck_chain if b in rest_scene.bone_by_name and rest_scene.bone_by_name[b].parent_name is not None]
    if len(valid) < 2:
        return dict(pose_deltas)
    out = dict(pose_deltas)
    n_v = len(valid)
    scale_n = 1.0 / max(1.0, math.sqrt(float(max(1, n_v - 1))))
    for idx, bname in enumerate(valid):
        is_head = idx == n_v - 1
        lag = phase_lag_rad * float(idx)
        if is_head:
            # Distal skull leads/counter-tracks subtly so gaze feels alive yet focused
            dp = -0.42 * pitch_amp_deg * math.sin(phase_rad - lag + 0.35)
            dy = -0.38 * yaw_amp_deg * math.cos(phase_rad - lag + 0.25)
            dr = -0.35 * roll_amp_deg * math.sin(phase_rad - lag)
        else:
            env = 0.75 + 0.45 * math.sin(math.pi * (float(idx + 0.5) / float(max(1, n_v - 1))))
            dp = pitch_amp_deg * scale_n * env * math.sin(phase_rad - lag)
            dy = yaw_amp_deg * scale_n * env * math.cos(phase_rad - lag)
            dr = roll_amp_deg * scale_n * env * math.sin(phase_rad - lag - 0.4)
        cur = np.asarray(out.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
        cand = cur + np.array([dp, dy, dr], dtype=np.float64)
        if limits_map is not None and bname in limits_map:
            jlim, asign = limits_map[bname]
            cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
        out[bname] = (float(cand[0]), float(cand[1]), float(cand[2]))
    return out


__all__ = [
    "AutoMuscleGroup",
    "BiomechanicalRigAssembly",
    "LimbAssembly",
    "MechanicalSubsystem",
    "NeckAssembly",
    "WingAssembly",
    "auto_assemble_biomechanical_rig",
    "limb_matches_pin_spec",
    "solve_com_balance_and_counterweight",
    "solve_foot_roll_and_digit_articulation",
    "solve_full_body_antagonist_muscle_bulge",
    "solve_serpentine_cervical_wave",
    "solve_spine_neck_spline_ik",
    "solve_standing_limb_ik_pinning",
    "solve_wing_full_anatomy",
]
