"""Universal Biological SO(3) Joint Constraints, Auto-Inference & Gaze Stabilization.

Provides:
1. Named anatomical SO(3) joint presets (`BIO_JOINT_PRESETS`) covering humanoids,
   hands/digits, quadrupeds, digitigrades, arachnids, serpents, fish, birds, and
   winged dragons/imaginary creatures.
2. Smooth C1 elastic-barrier soft clamping (`soft_clamp_euler_to_limits`) near
   anatomical stops so joints decelerate organically into tissue resistance without
   ever exceeding biological limits.
3. Universal geometry-first constraint auto-inference (`infer_universal_bio_constraints`):
   derives physically valid biological SO(3) limits directly from ANY skeleton's
   3D bone topology, segment lengths, and rest-pose bend geometry — even for brand-new
   imaginary creatures with arbitrary bone names.
4. Exact SO(3) vestibulo-ocular head/neck world stabilization
   (`solve_vestibulo_ocular_stabilization`) supporting both single- and multi-headed
   creatures.
5. Skeleton + muscle capability resolution (`analyze_motion_capabilities`): inspects
   the model's own anatomy to determine what locomotion and actuation abilities it
   supports (biped/quadruped/hexapod/arachnid gait, serpentine/aquatic undulation,
   2/4/6-wing flight/hover/glide, arm counter-swing, tentacle/tail waves, muscle drives).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synccraft.kinematics.se3_chain import (
    Scene,
    _angle_sign_from_reflection,
    _recover_euler_xyz_degrees,
    compile_geometry,
    compile_posed_geometry,
)

# Canonical anatomical SO(3) joint presets (degrees, right-handed local XYZ Euler:
# X = pitch / flex-extend, Y = yaw / twist-sweep, Z = roll / abduct-adduct).
# Every range satisfies -180 <= min <= 0 <= max <= 180 to stay compatible with
# SyncCraft's strict rest-pose origin contract.
BIO_JOINT_PRESETS: dict[str, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {
    # --- Humanoid / Hand / Digit joints ---
    "finger_flex": ((-5.0, 85.0), (-8.0, 8.0), (-8.0, 8.0)),
    "finger_proximal": ((-10.0, 90.0), (-15.0, 15.0), (-8.0, 8.0)),
    "thumb_saddle": ((-25.0, 55.0), (-30.0, 30.0), (-20.0, 20.0)),
    # --- Limb joints (Humanoid, Quadruped, Digitigrade, Arthropod, Dragon) ---
    "hinge_elbow": ((-125.0, 8.0), (-8.0, 8.0), (-8.0, 8.0)),
    "elbow_sprawled": ((-75.0, 120.0), (-42.0, 42.0), (-42.0, 42.0)),
    "hinge_knee": ((-8.0, 125.0), (-8.0, 8.0), (-8.0, 8.0)),
    "hinge_hock": ((-115.0, 15.0), (-8.0, 8.0), (-8.0, 8.0)),
    "wrist_ankle": ((-50.0, 55.0), (-22.0, 22.0), (-22.0, 22.0)),
    "ball_shoulder": ((-85.0, 75.0), (-45.0, 45.0), (-75.0, 45.0)),
    "ball_hip": ((-75.0, 65.0), (-35.0, 35.0), (-40.0, 35.0)),
    "arthropod_coxa": ((-45.0, 45.0), (-55.0, 55.0), (-45.0, 45.0)),
    "arthropod_hinge": ((-105.0, 95.0), (-22.0, 22.0), (-95.0, 95.0)),
    # --- Axial, Aquatic & Cranial joints ---
    "spine_vertebra": ((-24.0, 24.0), (-26.0, 26.0), (-16.0, 16.0)),
    "serpent_vertebra": ((-22.0, 22.0), (-42.0, 42.0), (-18.0, 18.0)),
    "neck_vertebra": ((-32.0, 32.0), (-35.0, 35.0), (-20.0, 20.0)),
    "head_atlas": ((-38.0, 38.0), (-45.0, 45.0), (-24.0, 24.0)),
    "tail_vertebra": ((-30.0, 30.0), (-38.0, 38.0), (-20.0, 20.0)),
    "tentacle_segment": ((-45.0, 45.0), (-45.0, 45.0), (-35.0, 35.0)),
    "jaw_hinge": ((-4.0, 62.0), (-4.0, 4.0), (-3.0, 3.0)),
    "ear_pinna": ((-22.0, 22.0), (-32.0, 32.0), (-24.0, 24.0)),
    # --- Winged / Aerodynamic / Hydrodynamic Fin joints ---
    "wing_shoulder": ((-55.0, 65.0), (-50.0, 55.0), (-68.0, 75.0)),
    "wing_elbow": ((-24.0, 24.0), (-32.0, 42.0), (-32.0, 35.0)),
    "wing_wrist": ((-26.0, 26.0), (-42.0, 45.0), (-36.0, 36.0)),
    "wing_finger": ((-20.0, 26.0), (-30.0, 32.0), (-24.0, 24.0)),
    "wing_anchor_locked": ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),
    "elytron_hinge": ((-45.0, 45.0), (-35.0, 35.0), (-65.0, 65.0)),
    "arthropod_chelicera": ((-34.0, 22.0), (-18.0, 18.0), (-22.0, 22.0)),
    "arthropod_pedipalp": ((-36.0, 28.0), (-28.0, 28.0), (-24.0, 24.0)),
    "pelagic_bell": ((-25.0, 25.0), (-25.0, 25.0), (-25.0, 25.0)),
    "fin_root": ((-40.0, 40.0), (-45.0, 45.0), (-55.0, 55.0)),
    "fin_ray": ((-25.0, 25.0), (-30.0, 30.0), (-25.0, 25.0)),
    # --- Mechanical / Vehicle / Industrial Mecha joints ---
    "wheel_axle": ((float("-inf"), float("inf")), (0.0, 0.0), (0.0, 0.0)),
    "steered_wheel_knuckle": ((-5.0, 5.0), (-38.0, 38.0), (-4.0, 4.0)),
    "rotor_propeller": ((0.0, 0.0), (float("-inf"), float("inf")), (0.0, 0.0)),
    "turret_yaw": ((0.0, 0.0), (-180.0, 180.0), (0.0, 0.0)),
    "cannon_elevation": ((-15.0, 65.0), (0.0, 0.0), (0.0, 0.0)),
    "hydraulic_hinge": ((-145.0, 145.0), (0.0, 0.0), (0.0, 0.0)),
    "hatch_door": ((0.0, 95.0), (0.0, 0.0), (0.0, 0.0)),
    "gimbal_thruster": ((-30.0, 30.0), (-30.0, 30.0), (-15.0, 15.0)),
}

MECHANICAL_JOINT_PRESET_NAMES: frozenset[str] = frozenset(
    {
        "wheel_axle",
        "steered_wheel_knuckle",
        "rotor_propeller",
        "turret_yaw",
        "cannon_elevation",
        "hydraulic_hinge",
        "hatch_door",
        "gimbal_thruster",
    }
)


def is_rigid_mechanical_bone(bone_name: str, bio_joint: str | None = None) -> bool:
    """Return True if a bone represents rigid mechanical/vehicle hardware (exempt from soft Sim(3) muscle bulge)."""
    if bio_joint is not None and bio_joint in MECHANICAL_JOINT_PRESET_NAMES:
        return True
    low = bone_name.lower()
    return any(
        tok in low
        for tok in (
            "wheel",
            "tire",
            "axle",
            "turret",
            "cannon",
            "railgun",
            "barrel",
            "rotor",
            "propeller",
            "turbine",
            "piston",
            "hydraulic",
            "actuator",
            "chassis",
            "hull",
            "hatch",
            "canopy",
            "thruster",
            "nozzle",
        )
    )

BIO_JOINT_NAMES: frozenset[str] = frozenset(BIO_JOINT_PRESETS.keys())


def resolve_bone_joint_limits(bone: Any) -> Any:
    """Return explicit ``bone.joint_limits`` if set, else resolve rest-pose-aware ``bone.bio_joint`` preset."""
    jlim = getattr(bone, "joint_limits", None)
    if jlim is not None:
        return jlim
    bio = getattr(bone, "bio_joint", None)
    if bio is not None and bio in BIO_JOINT_PRESETS:
        base = BIO_JOINT_PRESETS[bio]
        if bio in ("hinge_knee", "hinge_elbow", "hinge_hock"):
            rest_e = getattr(bone, "rest_euler", None)
            if rest_e is not None and len(rest_e) >= 1:
                rx = float(rest_e[0])
                lo, hi = float(base[0][0]), float(base[0][1])
                if rx > 10.0:
                    lo = min(lo, -min(rx, 85.0))
                elif rx < -10.0:
                    hi = max(hi, min(-rx, 85.0))
                return ((lo, hi), base[1], base[2])
        return base
    return None


def _soft_clamp_scalar(val: float, lo: float, hi: float, margin_frac: float = 0.18) -> float:
    """C1-smooth elastic barrier clamping ``val`` strictly within ``[lo, hi]``.

    Inside the central linear zone the output equals ``val`` identically. Approaching
    either anatomical stop, a ``tanh`` barrier smoothly decelerates the joint into
    biological tissue resistance without ever exceeding ``[lo, hi]``.
    """
    if not math.isfinite(lo) or not math.isfinite(hi):
        return float(val)
    span = hi - lo
    if span <= 1e-9:
        return float(lo)
    margin = min(0.35 * span, max(0.5, float(margin_frac) * span))
    if margin >= 0.49 * span:
        margin = 0.45 * span
    inner_hi = hi - margin
    inner_lo = lo + margin
    v = float(val)
    if v > inner_hi:
        excess = v - inner_hi
        return float(min(hi, inner_hi + margin * math.tanh(excess / max(1e-6, margin))))
    if v < inner_lo:
        excess = inner_lo - v
        return float(max(lo, inner_lo - margin * math.tanh(excess / max(1e-6, margin))))
    return v


def soft_clamp_euler_to_limits(
    delta_euler: np.ndarray | tuple[float, float, float],
    joint_limits: Any,
    *,
    angle_sign: np.ndarray | None = None,
    margin_frac: float = 0.18,
) -> np.ndarray:
    """Apply smooth biological SO(3) barrier clamping across all 3 local axes."""
    eff_sign = angle_sign if angle_sign is not None else np.ones(3, dtype=np.float64)
    arr = np.asarray(delta_euler, dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    for idx in range(3):
        lo, hi = float(joint_limits[idx][0]), float(joint_limits[idx][1])
        if float(eff_sign[idx]) < 0.0:
            lo, hi = -hi, -lo
        out[idx] = _soft_clamp_scalar(float(arr[idx]), lo, hi, margin_frac=margin_frac)
    return out


def _infer_hinge_limit_from_rest_euler(
    rest_euler: np.ndarray,
    *,
    max_flex: float = 115.0,
    max_hyper: float = 12.0,
    max_total_flex: float = 158.0,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Infer rest-pose-aware 1-DOF hinge direction & extension headroom directly from rest bend.

    Works universally across plantigrade knees, deep triple-segment digitigrade Z-folds
    (e.g., +132 deg dragon/wolf knee or -104 deg hock), elbows, reverse-jointed alien/bird
    limbs, folded arthropod legs, and folding mechanical excavator booms:
    - Gently bent rest joints (e.g., +15 deg) block backward hyperextension past 0 deg while
      allowing deep flexion.
    - Already-deeply-folded joints (e.g., +132 deg Z-fold knee or -104 deg Z-fold hock)
      automatically receive full straightening headroom back to 0 deg (without hyperextending
      past straight!) while capping further over-folding at `max_total_flex` (~158 deg).
    """
    rx = float(rest_euler[0])
    if rx > 5.0:
        # Already bent +X by rx: allow straightening back toward 0 deg (-(rx + max_hyper)),
        # and allow remaining flexion up to max_total_flex without folding back through parent.
        ext_headroom = min(145.0, rx) + max_hyper if rx >= 25.0 else max_hyper
        flex_headroom = min(max_flex, max(20.0, max_total_flex - rx))
        x_lim = (-ext_headroom, flex_headroom)
    elif rx < -5.0:
        # Already bent -X by |rx|: allow straightening back toward 0 deg (+(|rx| + max_hyper)),
        # and cap further -X flexion at max_total_flex.
        abs_rx = -rx
        ext_headroom = min(145.0, abs_rx) + max_hyper if abs_rx >= 25.0 else max_hyper
        flex_headroom = min(max_flex, max(20.0, max_total_flex - abs_rx))
        x_lim = (-flex_headroom, ext_headroom)
    else:
        # Neutral straight rest joint -> symmetric moderate biological/mechanical hinge
        x_lim = (-95.0, 95.0)
    # Anatomical 1-DOF sagittal hinge purity: restrict sideways knee/elbow/hock yaw & roll to +-8 deg
    # so Z-fold shins/ankles and elbows never bow outward like frog legs while hips absorb 3D tracking!
    return (x_lim, (-8.0, 8.0), (-8.0, 8.0))


def infer_universal_bio_constraints(
    program: Any,
    rest_scene: Scene | None = None,
    morphology: Any | None = None,
    *,
    explicit_limits: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> dict[str, tuple[Any, np.ndarray]]:
    """Derive biological SO(3) joint limits for EVERY bone on ANY creature rig.

    Explicitly authored ``joint_limits`` or ``bio_joint`` presets are ALWAYS preserved
    untouched. For any remaining unconstrained bone, limits are inferred purely from
    3D morphology, chain topology, and rest-pose bend geometry — ensuring universal
    coverage across real animals, humanoids, arthropods, aquatic creatures, and
    brand-new imaginary creatures regardless of bone naming conventions.
    """
    from synccraft.kinematics.morphology import analyze_morphology

    sc = rest_scene if rest_scene is not None else compile_geometry(program, validate=False)
    morph = morphology if morphology is not None else analyze_morphology(program, sc)
    root_name = sc.bones[0].name

    out: dict[str, tuple[Any, np.ndarray]] = dict(explicit_limits or {})
    # Compute reflection angle_sign for every bone from scene.mirrored
    sign_by_bone: dict[str, np.ndarray] = {
        b.name: (np.array((1.0, -1.0, -1.0), dtype=np.float64) if b.mirrored else np.ones(3, dtype=np.float64))
        for b in sc.bones
    }

    def _assign_if_missing(bname: str, preset_or_lim: Any) -> None:
        if bname == root_name or bname in out or bname not in sc.bone_by_name:
            return
        lim = BIO_JOINT_PRESETS[preset_or_lim] if isinstance(preset_or_lim, str) else preset_or_lim
        # When assigning inferred limits directly to an expanded bone, store limits in
        # that bone's own frame with angle_sign=(1,1,1) unless mirrored preset symmetry applies.
        out[bname] = (lim, sign_by_bone.get(bname, np.ones(3, dtype=np.float64)))

    # 0. Mechanical / Vehicle / Mecha hardware joints (wheels, steering knuckles, turrets, cannons, hatches, thrusters)
    for wb in sc.bones:
        if wb.name == root_name or wb.name in out:
            continue
        low = wb.name.lower()
        if any(k in low for k in ("steer", "knuckle")):
            _assign_if_missing(wb.name, "steered_wheel_knuckle")
        elif any(k in low for k in ("wheel", "tire", "axle")):
            _assign_if_missing(wb.name, "wheel_axle")
        elif any(k in low for k in ("rotor", "propeller", "turbine")):
            _assign_if_missing(wb.name, "rotor_propeller")
        elif any(k in low for k in ("turret", "slew")):
            _assign_if_missing(wb.name, "turret_yaw")
        elif any(k in low for k in ("cannon", "railgun", "barrel")):
            _assign_if_missing(wb.name, "cannon_elevation")
        elif any(k in low for k in ("hatch", "canopy", "ramp", "door")):
            _assign_if_missing(wb.name, "hatch_door")
        elif any(k in low for k in ("thruster", "nozzle")):
            _assign_if_missing(wb.name, "gimbal_thruster")
        elif any(k in low for k in ("piston", "hydraulic", "actuator", "boom")):
            _assign_if_missing(wb.name, "hydraulic_hinge")

    # 1. Grounded & free limbs (any count: biped, quadruped, hexapod, arachnid, alien)
    all_limbs = (*morph.limbs, *getattr(morph, "free_limbs", ()))
    for lb in all_limbs:
        chain = lb.chain
        n_c = len(chain)
        is_arthropod = len(morph.limbs) >= 6 and lb.posture == "sprawled"
        for idx, bname in enumerate(chain):
            if bname in out or bname == root_name:
                continue
            wb = sc.bone_by_name[bname]
            if idx == 0:
                _assign_if_missing(bname, "arthropod_coxa" if is_arthropod else ("ball_shoulder" if lb.role.startswith(("fore", "arm")) else "ball_hip"))
            elif idx == n_c - 1 and n_c >= 3:
                if abs(float(wb.local_euler[0])) > 45.0:
                    # Deeply folded distal metatarsus/hock or arthropod shaft -> rest-aware hinge!
                    hinge_lim = _infer_hinge_limit_from_rest_euler(wb.local_euler)
                    out[bname] = (hinge_lim, np.ones(3, dtype=np.float64))
                else:
                    _assign_if_missing(bname, "wrist_ankle")
            else:
                if is_arthropod:
                    _assign_if_missing(bname, "arthropod_hinge")
                else:
                    # Geometry-first hinge direction inference from rest_euler
                    hinge_lim = _infer_hinge_limit_from_rest_euler(wb.local_euler)
                    out[bname] = (hinge_lim, np.ones(3, dtype=np.float64))
        # Terminal digits (fingers / toes / claws) -> prevent backward hyperextension!
        for d_bname in getattr(lb, "digits", ()):
            _assign_if_missing(d_bname, "finger_flex")

    # 2. Wings / Airfoils / Flippers (any count: 2, 4, 6 wings or aquatic fins)
    for wg in morph.wings:
        for locked_b in getattr(wg, "locked_spar_bones", ()):
            _assign_if_missing(locked_b, "wing_anchor_locked")
        w_chain = wg.chain
        for idx, bname in enumerate(w_chain):
            if idx == 0:
                _assign_if_missing(bname, "wing_shoulder")
            elif idx == 1:
                _assign_if_missing(bname, "wing_elbow")
            else:
                _assign_if_missing(bname, "wing_wrist")
        for f_bname in getattr(wg, "finger_bones", ()):
            _assign_if_missing(f_bname, "wing_finger")

    # 3. Spine / Serpentine axial chain
    spine_preset = "serpent_vertebra" if len(morph.limbs) == 0 else "spine_vertebra"
    for bname in morph.spine:
        _assign_if_missing(bname, spine_preset)

    # 4. Neck & Head chains (single- or multi-headed creatures)
    neck_chains = getattr(morph, "neck_chains", ()) or ((morph.neck,) if morph.neck else ())
    for n_chain in neck_chains:
        for idx, bname in enumerate(n_chain):
            _assign_if_missing(bname, "head_atlas" if idx == len(n_chain) - 1 else "neck_vertebra")

    # 5. Tail chains (single- or multi-tailed creatures)
    tail_chains = getattr(morph, "tail_chains", ()) or ((morph.tail,) if morph.tail else ())
    for t_chain in tail_chains:
        for bname in t_chain:
            _assign_if_missing(bname, "tail_vertebra")

    # 6. Tentacles / Secondary flexible appendages
    for tent_chain in getattr(morph, "tentacles", ()):
        for bname in tent_chain:
            _assign_if_missing(bname, "tentacle_segment")

    # 7. Remaining uncategorized child bones (jaws, ears, antennae, fins, crests, digits)
    for wb in sc.bones:
        if wb.name == root_name or wb.name in out:
            continue
        low = wb.name.lower()
        if any(k in low for k in ("jaw", "mandible", "chin", "mouth")):
            _assign_if_missing(wb.name, "jaw_hinge")
        elif any(k in low for k in ("paw", "foot", "hand", "palm", "hoof", "ankle", "wrist", "manus", "pes")):
            _assign_if_missing(wb.name, "wrist_ankle")
        elif any(k in low for k in ("finger", "toe", "thumb", "digit", "claw", "talon")):
            _assign_if_missing(wb.name, "finger_flex")
        elif any(k in low for k in ("ear", "pinna", "frill", "ruff", "collar")):
            _assign_if_missing(wb.name, "ear_pinna")
        elif any(k in low for k in ("fin", "flipper")):
            _assign_if_missing(wb.name, "fin_ray")
        else:
            # Conservative universal biological joint envelope
            _assign_if_missing(wb.name, "spine_vertebra")

    return out


def analyze_motion_capabilities(
    program: Any,
    rest_scene: Scene | None = None,
) -> dict[str, Any]:
    """Resolve what locomotion & biomechanical animation capabilities a rig physically supports.

    Works universally on any skeleton + ``@muscles`` definition — from standard animals
    and humanoids to arbitrary multi-limbed, multi-winged, multi-headed, tentacled, or
    legless imaginary creatures.
    """
    from synccraft.kinematics.morphology import analyze_morphology

    sc = rest_scene if rest_scene is not None else compile_geometry(program, validate=False)
    morph = analyze_morphology(program, sc)
    n_limbs = len(morph.limbs)
    n_free_limbs = len(getattr(morph, "free_limbs", ()))
    n_wings = len(morph.wings)
    n_necks = len(getattr(morph, "neck_chains", ()) or ((morph.neck,) if morph.neck else ()))
    n_tails = len(getattr(morph, "tail_chains", ()) or ((morph.tail,) if morph.tail else ()))
    n_tentacles = len(getattr(morph, "tentacles", ()))
    axial_len = len(morph.spine) + len(morph.tail)

    wheel_bones = [
        b.name
        for b in sc.bones
        if any(k in b.name.split("__")[-1].lower() for k in ("wheel", "tire", "axle"))
        and not any(k in b.name.split("__")[-1].lower() for k in ("steer", "knuckle"))
    ]
    turret_bones = [b.name for b in sc.bones if any(k in b.name.lower() for k in ("turret", "slew"))]
    rotor_bones = [b.name for b in sc.bones if any(k in b.name.lower() for k in ("rotor", "propeller", "turbine"))]
    is_mech_or_vehicle = bool(wheel_bones or turret_bones or rotor_bones or any(is_rigid_mechanical_bone(b.name) for b in sc.bones))

    wheel_bones_set = set(wheel_bones)
    non_wheel_limbs = [
        lb
        for lb in morph.limbs
        if not any(b in wheel_bones_set or is_rigid_mechanical_bone(b) for b in lb.chain)
    ]
    n_bio_limbs = len(non_wheel_limbs)

    if wheel_bones and n_bio_limbs == 0:
        recommended_gait = None
        locomotion_class = "wheeled_vehicle"
    elif n_bio_limbs == 0:
        recommended_gait = "serpentine_wave" if axial_len >= 3 else None
        locomotion_class = "limbless_undulatory" if axial_len >= 3 else "stationary_or_aerial"
    elif n_bio_limbs == 2:
        recommended_gait = "biped_alternate"
        locomotion_class = "biped"
    elif n_bio_limbs == 4:
        any_sprawled = any(lb.posture == "sprawled" for lb in non_wheel_limbs)
        recommended_gait = "sprawl_diagonal" if any_sprawled else "lateral_sequence"
        locomotion_class = "quadruped_sprawled" if any_sprawled else "quadruped"
    elif n_bio_limbs == 6:
        recommended_gait = "hexapod_tripod"
        locomotion_class = "hexapod"
    else:
        recommended_gait = "metachronal_wave"
        locomotion_class = f"polypod_{n_bio_limbs}_limbs"

    wing_pairs = n_wings // 2
    can_fly = n_wings >= 2
    recommended_flight_modes: list[str] = []
    if can_fly:
        recommended_flight_modes = ["hover", "cruise", "glide", "soar", "mantle"]
    elif n_wings == 1:
        recommended_flight_modes = ["glide"]

    muscles_info = [
        {
            "name": m.name,
            "origin_bone": m.origin_bone,
            "insertion_bone": m.insertion_bone,
            "action_axis": m.action_axis,
            "has_tendon_coupling": m.tendon_coupling is not None,
            "has_sim3_bulge": bool(getattr(m, "scale_bulge", 0.0) > 0.0 or m.bulge_inflate > 0.0),
        }
        for m in getattr(program, "muscles", ())
    ]

    return {
        "locomotion_class": locomotion_class,
        "grounded_limb_count": n_limbs,
        "free_limb_count": n_free_limbs,
        "wing_or_airfoil_count": n_wings,
        "wing_pair_count": wing_pairs,
        "neck_head_chain_count": n_necks,
        "tail_chain_count": n_tails,
        "tentacle_chain_count": n_tentacles,
        "wheel_count": len(wheel_bones),
        "turret_count": len(turret_bones),
        "rotor_count": len(rotor_bones),
        "is_mechanical_or_vehicle": is_mech_or_vehicle,
        "can_walk_or_run": n_limbs >= 2,
        "can_drive_wheels": len(wheel_bones) >= 2,
        "can_slither_or_swim": axial_len >= 3,
        "can_fly_or_hover": can_fly or len(rotor_bones) >= 1,
        "can_arm_counter_swing": n_free_limbs >= 2,
        "can_stabilize_gaze": n_necks >= 1,
        "recommended_gait_pattern": recommended_gait,
        "recommended_flight_modes": recommended_flight_modes,
        "declared_muscles": muscles_info,
        "bone_count": len(sc.bones),
    }


def solve_vestibulo_ocular_stabilization(
    program: Any,
    rest_scene: Scene,
    neck_bones: tuple[str, ...] | list[str],
    pose_deltas: dict[str, tuple[float, float, float]],
    *,
    strength: float = 0.80,
    weight_decay: float = 0.72,
    limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
    root_translation_delta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    pose_position_deltas: dict[str, tuple[float, float, float]] | None = None,
    pose_scale_factors: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """Counter-rotate a neck/head chain in exact SO(3) to keep gaze level against torso motion.

    Models the avian/reptilian/mammalian vestibulo-ocular & cervico-collic reflex: when
    the spine and chest pitch, yaw, or roll during locomotion or wingbeats, the neck/head
    chain distributes a smooth counter-rotation (weighted distally toward the skull via
    ``weight_decay``) so the creature's head holds a steady gaze lock on the horizon.
    """
    gain = float(np.clip(strength, 0.0, 1.0))
    if gain <= 1e-5 or not neck_bones:
        return dict(pose_deltas)

    chain = [
        b
        for b in neck_bones
        if b in rest_scene.bone_by_name and rest_scene.bone_by_name[b].parent_name is not None
    ]
    if not chain:
        return dict(pose_deltas)

    posed_pre = compile_posed_geometry(
        program,
        pose_deltas,
        root_translation_delta=root_translation_delta,
        pose_position_deltas=pose_position_deltas,
        pose_scale_factors=pose_scale_factors,
        validate=False,
    )
    first_b = chain[0]
    parent_name = rest_scene.bone_by_name[first_b].parent_name
    if parent_name is None or parent_name not in posed_pre.bone_by_name:
        return dict(pose_deltas)

    r_parent_rest = rest_scene.bone_by_name[parent_name].rotation
    r_parent_posed = posed_pre.bone_by_name[parent_name].rotation
    r_counter_world = r_parent_rest @ r_parent_posed.T
    if np.allclose(r_counter_world, np.eye(3), atol=1e-6):
        return dict(pose_deltas)

    # Smooth C1 acrobatic disengagement: during inverted somersaults, backflips, or
    # frontflips (torso deviation > 55 deg), vestibulo-ocular horizon stabilization
    # smoothly attenuates to 0 by 95 deg so _recover_euler_xyz_degrees never hits
    # gimbal lock or the 180 deg antipodal branch-cut snap.
    cos_dev = float(np.clip(0.5 * (float(np.trace(r_counter_world)) - 1.0), -1.0, 1.0))
    dev_deg = math.degrees(math.acos(cos_dev))
    if dev_deg >= 95.0:
        return dict(pose_deltas)
    if dev_deg > 55.0:
        u = (dev_deg - 55.0) / 40.0
        gain *= 1.0 - (3.0 * u * u - 2.0 * u * u * u)
        if gain <= 1e-5:
            return dict(pose_deltas)

    n_c = len(chain)
    raw_w = np.array([float(weight_decay) ** (n_c - 1 - i) for i in range(n_c)], dtype=np.float64)
    weights = raw_w / max(1e-9, float(np.sum(raw_w)))

    out = dict(pose_deltas)
    for b_name, w in zip(chain, weights):
        b_rest = rest_scene.bone_by_name[b_name]
        # Map world counter-rotation into this bone's local frame cleanly
        r_loc_corr = b_rest.rotation.T @ r_counter_world @ b_rest.rotation
        corr_deg = _recover_euler_xyz_degrees(r_loc_corr) * (gain * float(w))
        cur = np.asarray(out.get(b_name, (0.0, 0.0, 0.0)), dtype=np.float64)
        cand = cur + corr_deg
        if limits_map is not None and b_name in limits_map:
            jlim, asign = limits_map[b_name]
            cand = soft_clamp_euler_to_limits(cand, jlim, angle_sign=asign)
        out[b_name] = (float(cand[0]), float(cand[1]), float(cand[2]))

    return out
