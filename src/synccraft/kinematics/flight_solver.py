"""Universal Aerodynamic & Hydrodynamic Airfoil / Flight Solver (`SO(3)` / `SE(3)` / `Sim(3)`).

Synthesizes biologically grounded wing-beat, hovering, cruising, gliding, grounded wing-mantle,
and aquatic flipper-swimming animations for ANY winged or finned creature rig (2-wing birds,
bats, wyverns, dragons; 4-wing butterflies, dragonflies, angels; 6-wing seraphs; or finned
aquatic creatures) purely from `@morphology` geometry and `@muscles` / `SO(3)` constraints.

Key physical invariants:
1. Minecraft / Blockbench Entity-Space (`in_place: true`): NEVER bakes artificial macro
   altitude (`+35u`) or macro forward flight distance into bone positions. Emits ONLY
   authentic local anti-phase aerodynamic lift COM recoil (`±lift_recoil_u`) plus exact
   matched entity velocity in `synccraft_root_motion`.
2. Geometry-First Bilateral Axis Calibration: derives downstroke, fore-aft sweep, and
   chordwise pronation axes directly from each wing's 3D angular Jacobian so bilateral
   pairs stay world-symmetric regardless of bind-pose bone orientation.
3. Asymmetric Power/Recovery Figure-8 Stroke: fast pronated extended downstroke followed
   by folded, feathered (supinated) recovery upstroke with proximal-to-distal phase lag
   (`shoulder -> elbow -> wrist -> membrane fingers`).
4. Membrane Continuity Protection: holds medial locked anchor spars (`locked_spar_bones`,
   e.g. forearm-mounted `wing_l_f4`) strictly at `[0, 0, 0]` to prevent inner-wing splitting.
5. Full Unified Biomechanics: integrates Zajac/Hill `@muscles` + `Sim(3)` muscle bulge,
   vestibulo-ocular `head_stabilize` across all neck chains, multi-tail/tentacle waves,
   damped `spring` secondary inertia, and grounded world IK in `mantle` mode.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

import numpy as np

from synccraft.dsl.ast_nodes import ClipDecl, FlightDecl, IKChainDecl, Program, SpringDecl, Vec3
from synccraft.kinematics.bio_constraints import (
    infer_universal_bio_constraints,
    soft_clamp_euler_to_limits,
    solve_vestibulo_ocular_stabilization,
)
from synccraft.kinematics.morphology import MorphologyReport, WingMorphology, analyze_morphology
from synccraft.kinematics.se3_chain import (
    Scene,
    _recover_euler_xyz_degrees,
    compile_geometry,
    compile_posed_geometry,
    euler_xyz_matrix,
)


def _asymmetric_wing_phase(phi: float, downstroke_frac: float = 0.44) -> float:
    """Warp linear cycle phase ``phi in [0, 1)`` so downstroke occupies ``downstroke_frac``.

    Maps:
    - ``phi = 0.0`` (top of stroke / upstroke apex) -> ``theta = 0.0``
    - ``phi = downstroke_frac`` (bottom of power downstroke) -> ``theta = pi``
    - ``phi = 1.0`` (return to top of stroke) -> ``theta = 2 * pi``

    Uses a C1-smooth piecewise cosine-blended warp so derivatives are continuous across
    stroke reversals with zero loop jump.
    """
    u = phi % 1.0
    d = float(np.clip(downstroke_frac, 0.30, 0.65))
    # Smooth C1 harmonic phase warp: theta = 2*pi*u + k * sin(2*pi*u)
    # At u = d, we want 2*pi*d + k*sin(2*pi*d) = pi => k = pi*(1 - 2*d) / sin(2*pi*d)
    sin_2pid = math.sin(2.0 * math.pi * d)
    if abs(sin_2pid) < 1e-6:
        k_warp = 0.0
    else:
        k_warp = float(np.clip(math.pi * (1.0 - 2.0 * d) / sin_2pid, -0.75, 0.75))
    return 2.0 * math.pi * u + k_warp * math.sin(2.0 * math.pi * u)


def _calibrate_wing_local_axes(
    rest_scene: Scene,
    wing: WingMorphology,
    forward_sign_z: float,
) -> dict[str, tuple[int, float]]:
    """Derive exact local rotation axis & sign for (downstroke, fwd_sweep, pronation) from 3D Jacobian."""
    chain = wing.chain
    if not chain or chain[0] not in rest_scene.bone_by_name:
        return {"down": (2, 1.0), "fwd": (1, 1.0), "twist": (0, 1.0)}

    root_b = rest_scene.bone_by_name[chain[0]]
    # Measure displacement of the distal spar pivot (wrist or tip box center)
    if len(chain) >= 2 and chain[-1] in rest_scene.bone_by_name:
        tip_w = rest_scene.bone_by_name[chain[-1]].pivot_world
    else:
        bxs = [bx for bx in rest_scene.boxes if bx.bone_name == chain[0]]
        tip_w = bxs[0].center if bxs else (root_b.pivot_world + np.array([1.0 if wing.side == "r" else -1.0, 0.0, 0.0]))
    r_vec = tip_w - root_b.pivot_world
    if float(np.linalg.norm(r_vec)) < 1e-4:
        r_vec = np.array([-10.0 if wing.side == "l" else 10.0, 2.0, 0.0], dtype=np.float64)

    # Instantaneous world linear velocity of tip for +1 rad around each local axis i
    jac = np.zeros((3, 3), dtype=np.float64)
    for ax_i in range(3):
        axis_w = root_b.rotation[:, ax_i]
        jac[:, ax_i] = np.cross(axis_w, r_vec)

    # 1. Downstroke axis: strongest vertical |J[1, i]|, sign chosen so +angle moves tip DOWN (-Y)
    down_ax = int(np.argmax(np.abs(jac[1, :])))
    down_sign = -1.0 if jac[1, down_ax] > 0.0 else 1.0
    if abs(jac[1, down_ax]) < 1e-4:
        down_ax, down_sign = 2, (1.0 if wing.side == "l" else -1.0)

    # 2. Fore-aft sweep axis: among remaining axes, strongest |J[2, i]|, sign moves tip FORWARD (+fwd_z)
    rem_axes = [i for i in range(3) if i != down_ax]
    fwd_ax = max(rem_axes, key=lambda i: abs(jac[2, i]))
    fwd_sign = 1.0 if (jac[2, fwd_ax] * forward_sign_z) >= 0.0 else -1.0
    if abs(jac[2, fwd_ax]) < 1e-4:
        fwd_ax, fwd_sign = 1, (1.0 if wing.side == "l" else -1.0)

    # 3. Chordwise pronation / twist axis: the remaining spanwise axis (usually X)
    twist_ax = [i for i in range(3) if i not in (down_ax, fwd_ax)][0]
    twist_sign = 1.0

    # Measure bind-pose world elevation angle above horizontal so already-elevated dihedral wings
    # NEVER over-lift past the dorsal midline and clap into their mirrored twin at upstroke apex!
    horiz_r = max(1e-4, math.hypot(float(r_vec[0]), float(r_vec[2])))
    rest_elev_deg = math.degrees(math.atan2(float(r_vec[1]), horiz_r))

    return {
        "down": (down_ax, down_sign),
        "fwd": (fwd_ax, fwd_sign),
        "twist": (twist_ax, twist_sign),
        "rest_elev_deg": (0, float(rest_elev_deg)),
    }


def _project_world_aero_to_bone_euler(
    rest_scene: Scene,
    bone_name: str,
    side: str,
    forward_sign_z: float,
    down_deg: float,
    fwd_deg: float,
    twist_deg: float,
    tip_ref_world: np.ndarray | None = None,
) -> np.ndarray:
    """Project orthogonal world aerodynamic angles `(down_deg, fwd_deg, twist_deg)` into bone's local `SO(3)` Euler delta.

    Uses the Lie-algebra `so(3)` -> `SO(3)` Rodrigues exponential map around the bone's actual 3D
    spanwise vector so steeply pitched cathedral wings (e.g. `-61 deg` rest pitch) and fanned distal
    fingers (`-52 deg` yaw) beat cleanly up/down and forward/aft in world space without axis skew!
    """
    wb = rest_scene.bone_by_name[bone_name]
    b_piv = wb.pivot_world
    if tip_ref_world is not None and float(np.linalg.norm(tip_ref_world - b_piv)) > 1e-3:
        r_span = tip_ref_world - b_piv
    else:
        bxs = [bx for bx in rest_scene.boxes if bx.bone_name == bone_name and getattr(bx, "plane_normal", None) is None]
        if bxs:
            # Farthest box center or corner along outward span
            best_v = bxs[0].center - b_piv
            for bx in bxs:
                for cr in bx.corners(include_inflate=False):
                    dv = cr - b_piv
                    if float(np.linalg.norm(dv)) > float(np.linalg.norm(best_v)):
                        best_v = dv
            r_span = best_v
        else:
            r_span = np.array([-1.0 if side == "l" else 1.0, 0.0, 0.0], dtype=np.float64)

    # Project span vector primarily into horizontal-outward component so world lift & sweep stay orthogonal
    out_sgn = -1.0 if side == "l" else 1.0
    u_span = np.asarray(r_span, dtype=np.float64)
    n_s = float(np.linalg.norm(u_span))
    if n_s < 1e-5:
        u_span = np.array([out_sgn, 0.0, 0.0], dtype=np.float64)
    else:
        u_span = u_span / n_s

    # World unit axis rotating `u_span` toward `-e_y` (downstroke): omega x u_span = -e_y => omega = u_span x (-e_y)
    e_down = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    w_down = np.cross(u_span, e_down)
    nd = float(np.linalg.norm(w_down))
    w_down = (w_down / nd) if nd > 1e-5 else np.array([0.0, 0.0, -out_sgn], dtype=np.float64)

    # World unit axis rotating `u_span` toward `+forward_sign_z * e_z`: omega = u_span x (fwd_z * e_z)
    e_fwd = np.array([0.0, 0.0, float(forward_sign_z)], dtype=np.float64)
    w_fwd = np.cross(u_span, e_fwd)
    nf = float(np.linalg.norm(w_fwd))
    w_fwd = (w_fwd / nf) if nf > 1e-5 else np.array([0.0, -out_sgn * float(forward_sign_z), 0.0], dtype=np.float64)

    # Spanwise pronation twist (leading edge pitches down on downstroke)
    w_twist = -out_sgn * float(forward_sign_z) * u_span

    omega_w = (
        math.radians(float(down_deg)) * w_down
        + math.radians(float(fwd_deg)) * w_fwd
        + math.radians(float(twist_deg)) * w_twist
    )
    ang = float(np.linalg.norm(omega_w))
    if ang < 1e-9:
        return np.zeros(3, dtype=np.float64)

    omega_loc = wb.rotation.T @ omega_w
    k_axis = omega_loc / ang
    K = np.array(
        [
            [0.0, -k_axis[2], k_axis[1]],
            [k_axis[2], 0.0, -k_axis[0]],
            [-k_axis[1], k_axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    r_delta = np.eye(3, dtype=np.float64) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)
    return np.asarray(_recover_euler_xyz_degrees(r_delta), dtype=np.float64)


def _apply_muscle_scale_bulge(
    program: Program,
    rest_scene: Scene,
    activations: dict[str, float],
    scale_factors: dict[str, Vec3],
) -> None:
    """Convert declared `@muscles` ``scale_bulge`` activations into volume-preserving Sim(3) bone scales."""
    for mus in getattr(program, "muscles", ()):
        sb = float(getattr(mus, "scale_bulge", 0.0))
        if sb <= 1e-6:
            continue
        act = float(np.clip(activations.get(mus.name, 0.0), 0.0, 1.0))
        if act <= 1e-6:
            continue
        target_bone = mus.insertion_bone
        if mus.belly_box is not None and mus.belly_box in rest_scene.box_by_name:
            target_bone = rest_scene.box_by_name[mus.belly_box].bone_name
        # Volume-preserving Sim(3) bulge: expand laterally orthogonal to primary axis while slightly contracting along it
        delta = sb * act
        if mus.action_axis == "x":
            s_vec = (1.0 + delta, 1.0 + 0.35 * delta, 1.0 - 0.30 * delta)
        elif mus.action_axis == "y":
            s_vec = (1.0 + delta, 1.0 - 0.30 * delta, 1.0 + delta)
        else:
            s_vec = (1.0 + 0.35 * delta, 1.0 + delta, 1.0 + delta)
        prev = scale_factors.get(target_bone, (1.0, 1.0, 1.0))
        scale_factors[target_bone] = (
            float(prev[0] * s_vec[0]),
            float(prev[1] * s_vec[1]),
            float(prev[2] * s_vec[2]),
        )


_JET_MODES: frozenset[str] = frozenset({"jet_hover", "jet_cruise", "jet_boost"})


def _compute_flight_velocity_u_s(
    flight: FlightDecl,
    morph: MorphologyReport,
    duration_s: float,
) -> float:
    """Compute matched Minecraft entity velocity (units/s) for cruise/glide/swim_fins/jet modes."""
    mode = flight.mode
    if mode in ("hover", "mantle", "jet_hover"):
        return 0.0
    wing_span = max((w.span for w in morph.wings), default=max(12.0, morph.body_span[0]))
    if mode == "cruise":
        return float(max(16.0, 1.35 * wing_span * max(0.4, flight.flap_hz)))
    if mode == "jet_cruise":
        return float(max(36.0, 2.60 * wing_span))
    if mode == "jet_boost":
        return float(max(24.0, 1.85 * wing_span * max(0.5, flight.flap_hz)))
    if mode in ("glide", "soar"):
        return float(max(20.0, 1.60 * wing_span))
    if mode == "takeoff_burst":
        return float(max(12.0, 0.85 * wing_span * max(0.5, flight.flap_hz)))
    if mode == "swim_fins":
        return float(max(14.0, 1.15 * morph.body_span[2] * max(0.4, flight.flap_hz)))
    return 0.0


def evaluate_flight_clip(
    program: Program,
    clip: ClipDecl,
    *,
    num_frames: int = 12,
    active_state: str | None = None,
    sample_phases: Sequence[float] | None = None,
    _phase_cache: dict[float, Any] | None = None,
) -> list[Any]:
    """Synthesize exact SO(3)/SE(3)/Sim(3) aerodynamic flight, hover, glide, mantle, or swim_fins frames."""
    from synccraft.data_gen.ik_motion_templates import solve_n_bone_ik
    from synccraft.kinematics.musculoskeletal import (
        BehaviorFrameSample,
        _batch_sample_clip_layer_deltas,
        _clamp_pose_vec,
        _collect_expanded_joint_limits,
        _compute_muscle_bulge_and_telemetry,
        _evaluate_track_angle,
        _integrate_spring_follow_through,
        evaluate_keyframe_track,
        evaluate_muscles_at_activations,
    )

    if not getattr(clip, "flights", ()):
        raise ValueError(f"Clip {clip.name!r} has no flight declaration")
    flight = clip.flights[0]

    rest_scene = compile_geometry(program, validate=False)
    morph = analyze_morphology(program, rest_scene)
    explicit_limits = _collect_expanded_joint_limits(program)
    auto_rig = bool(getattr(clip, "auto_rig", False))
    use_auto_bio = bool(flight.auto_bio_limits or getattr(clip, "auto_bio_limits", False) or auto_rig)
    digit_g = float(getattr(flight, "digit_articulation", 0.0)) or (0.80 if auto_rig else 0.0)
    com_g = max(float(getattr(flight, "com_balance", 0.0)), float(getattr(clip, "com_balance", 0.0))) or (0.50 if auto_rig else 0.0)
    antag_g = max(float(getattr(flight, "antagonist_bulge", 0.0)), float(getattr(clip, "antagonist_bulge", 0.0))) or (0.65 if auto_rig else 0.0)
    pec_g = float(getattr(flight, "pectoral_bulge", 0.0)) or (0.70 if auto_rig else 0.0)
    ground_clr = bool(getattr(flight, "ground_clearance", True))
    neck_aim_tgt = getattr(clip, "neck_aim", None)
    spine_coil_deg = float(getattr(clip, "spine_coil", 0.0))
    digit_grasp_g = float(getattr(clip, "digit_grasp", 0.0))
    need_rig_asm = bool(
        auto_rig
        or digit_g > 1e-4
        or com_g > 1e-4
        or antag_g > 1e-4
        or pec_g > 1e-4
        or neck_aim_tgt is not None
        or abs(spine_coil_deg) > 1e-4
        or digit_grasp_g > 1e-4
    )
    rig_asm = None
    if need_rig_asm:
        from synccraft.kinematics.whole_body_solver import auto_assemble_biomechanical_rig

        rig_asm = auto_assemble_biomechanical_rig(program, rest_scene, morph, explicit_limits=explicit_limits)
        limits_map = rig_asm.bio_constraints if use_auto_bio else explicit_limits
    else:
        limits_map = (
            infer_universal_bio_constraints(program, rest_scene, morph, explicit_limits=explicit_limits)
            if use_auto_bio
            else explicit_limits
        )

    state_pose_bias: dict[str, Vec3] = {}
    state_muscle_tone: dict[str, float] = {}
    if active_state is not None and program.behavior is not None:
        st_map = {s.name: s for s in program.behavior.states}
        if active_state not in st_map:
            raise KeyError(f"Unknown @behavior state {active_state!r}")
        st = st_map[active_state]
        state_pose_bias = dict(st.pose_biases)
        state_muscle_tone = dict(st.muscle_tones)

    if sample_phases is not None:
        phases = np.asarray(list(sample_phases), dtype=np.float64)
    else:
        n_f = max(2, int(num_frames))
        phases = np.linspace(0.0, 1.0, n_f, endpoint=not clip.loop, dtype=np.float64)

    root_bone_name = rest_scene.bones[0].name
    fwd_z = morph.forward_sign_z
    dur_s = max(1e-4, float(clip.duration_s))

    # Quantize flap cycles to exact integer when looping so loop closure is exact (< 0.001 deg)
    raw_cycles = float(flight.flap_hz) * dur_s
    n_cycles = max(1, int(round(raw_cycles))) if clip.loop else max(0.25, raw_cycles)

    # Pre-calibrate wing axes for every discovered wing/fin
    wing_by_role: dict[str, WingMorphology] = {w.role: w for w in morph.wings}
    wing_axes: dict[str, dict[str, tuple[int, float]]] = {
        w.role: _calibrate_wing_local_axes(rest_scene, w, fwd_z) for w in morph.wings
    }
    # Pair lookup so mirrored right wings receive exact X-reflected Euler symmetry
    left_role_for_pair: dict[int, str] = {w.pair_index: w.role for w in morph.wings if w.side == "l"}

    # Mode-specific aerodynamic / jet-thruster shaping factors
    mode = flight.mode
    is_jet = mode in _JET_MODES
    gimbal_deg = float(getattr(flight, "gimbal_deg", 3.5))
    bank_deg = float(getattr(flight, "bank_deg", 0.0))
    if mode == "hover":
        stroke_scale, sweep_scale, pron_scale, fold_scale = 1.0, 1.0, 1.0, 1.0
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 1.0, 1.0, 5.5
        down_frac = 0.44
        dihedral_bias_deg = -6.0  # slightly lifted neutral hover plane
    elif mode == "cruise":
        stroke_scale, sweep_scale, pron_scale, fold_scale = 1.05, 0.65, 0.85, 0.80
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 0.72, 0.22, 3.2
        down_frac = 0.46
        dihedral_bias_deg = -4.0
    elif mode == "jet_hover":
        # VTOL multi-thruster hover: ZERO wing paddling/flapping; pure thruster suspension + nozzle micro-gimbal!
        stroke_scale, sweep_scale, pron_scale, fold_scale = 0.0, 0.0, 0.0, 0.0
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 1.0, 1.0, 1.4
        down_frac = 0.50
        dihedral_bias_deg = 0.0
    elif mode == "jet_cruise":
        # Supersonic / fighter-jet cruise: ZERO wing flapping; swept-back thruster stability + subtle aero bank!
        stroke_scale, sweep_scale, pron_scale, fold_scale = 0.0, 0.0, 0.0, 0.0
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 0.28, 1.0, 0.7
        down_frac = 0.50
        dihedral_bias_deg = 0.0
    elif mode == "jet_boost":
        # Impulsive VTOL mid-air rocket-hop / afterburner surge: ZERO wing flapping; sharp thrust recoil & tremor!
        stroke_scale, sweep_scale, pron_scale, fold_scale = 0.0, 0.0, 0.0, 0.0
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 1.25, 1.0, 2.2
        down_frac = 0.42
        dihedral_bias_deg = 0.0
    elif mode in ("glide", "soar"):
        stroke_scale, sweep_scale, pron_scale, fold_scale = 0.22, 0.18, 0.35, 0.18
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 0.32, 0.18, 1.6
        down_frac = 0.50
        dihedral_bias_deg = -10.0  # majestic open dihedral sail
    elif mode == "takeoff_burst":
        stroke_scale, sweep_scale, pron_scale, fold_scale = 1.28, 1.15, 1.15, 1.15
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 1.25, 1.15, 7.0
        down_frac = 0.42
        dihedral_bias_deg = -8.0
    elif mode == "mantle":
        stroke_scale, sweep_scale, pron_scale, fold_scale = 0.78, 0.65, 0.60, 0.55
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 0.25, 0.35, 3.0
        down_frac = 0.46
        dihedral_bias_deg = -14.0  # high intimidating wing arch
    else:  # swim_fins
        stroke_scale, sweep_scale, pron_scale, fold_scale = 0.85, 1.10, 0.90, 0.55
        recoil_scale, pitch_bias_scale, pitch_osc_deg = 0.35, 0.10, 2.5
        down_frac = 0.48
        dihedral_bias_deg = 0.0

    # Entity-space matched velocity metadata (in_place keeps animation local!)
    spd_u_s = _compute_flight_velocity_u_s(flight, morph, dur_s)
    rm_vel: Vec3 = (0.0, 0.0, fwd_z * spd_u_s) if abs(spd_u_s) > 1e-6 else (0.0, 0.0, 0.0)

    # Collect declared secondary springs (exclude wing/limb structural bones if ground-locked)
    all_springs: list[SpringDecl] = []
    seen_sp: set[str] = set()
    for sp_src in (
        *(program.kinematics.springs if program.kinematics is not None else ()),
        *(program.behavior.springs if program.behavior is not None else ()),
    ):
        if sp_src.bone not in seen_sp:
            seen_sp.add(sp_src.bone)
            all_springs.append(sp_src)

    ik_decl_by_name: dict[str, IKChainDecl] = (
        {c.name: c for c in program.kinematics.ik_chains} if program.kinematics is not None else {}
    )
    axis_idx = {"x": 0, "y": 1, "z": 2}
    all_times_s = [float(p) * dur_s for p in phases]
    batched_layers = [
        _batch_sample_clip_layer_deltas(program, lyr.source_clip, all_times_s, float(lyr.weight), parent_clip=clip)
        for lyr in getattr(clip, "layers", ())
    ]

    # Identify all neck, tail, and wing-related muscles for automatic neural recruitment
    wing_bone_set = {bn for w in morph.wings for bn in (*w.chain, *w.finger_bones)}

    times_list: list[float] = []
    phases_list: list[float] = []
    root_deltas_list: list[Vec3] = []
    acts_list: list[dict[str, float]] = []
    pos_deltas_list: list[dict[str, Vec3]] = []
    scale_factors_list: list[dict[str, Vec3]] = []
    pre_spring_poses_list: list[dict[str, Vec3]] = []
    eff_pos_list: list[dict[str, Vec3]] = []
    eff_stance_list: list[dict[str, bool]] = []
    eff_step_list: list[dict[str, int]] = []

    for f_idx, phi_raw in enumerate(phases):
        phi_clip = float(phi_raw)
        t_s = phi_clip * dur_s
        cyc_prog = phi_clip * float(n_cycles)
        cyc_phi = cyc_prog % 1.0
        times_list.append(t_s)
        phases_list.append(phi_clip)

        # Asymmetric aerodynamic stroke angle (or uniform jet-propulsion harmonic angle)
        u_jet = 2.0 * math.pi * cyc_phi
        theta = _asymmetric_wing_phase(cyc_phi, downstroke_frac=down_frac)
        in_place_active = bool(flight.in_place or getattr(clip, "in_place", False))
        # Anti-phase aerodynamic lift COM recoil (or VTOL jet suspension + 3x rocket-exhaust micro-tremor)
        # In-place contract: ZERO huge +35u altitude offset; pure local COM heave around y=0!
        if mode == "jet_hover":
            lift_recoil_y = float(flight.lift_recoil_u) * recoil_scale * (math.sin(u_jet) + 0.14 * math.sin(3.0 * u_jet))
        elif mode == "jet_cruise":
            lift_recoil_y = float(flight.lift_recoil_u) * recoil_scale * (math.sin(u_jet) + 0.12 * math.sin(3.0 * u_jet))
        elif mode == "jet_boost":
            lift_recoil_y = float(flight.lift_recoil_u) * recoil_scale * (math.sin(theta - 0.25 * math.pi) + 0.16 * math.sin(3.0 * theta))
        elif mode == "mantle":
            # Grounded mantle squats/rises subtly without lifting feet off the floor
            lift_recoil_y = -abs(float(flight.lift_recoil_u) * 0.22) * (0.5 - 0.5 * math.cos(theta))
        else:
            lift_recoil_y = float(flight.lift_recoil_u) * recoil_scale * math.sin(theta - 0.35 * math.pi)
        # Fore-aft aerodynamic surge strictly gated off when in_place=True or in jet VTOL modes!
        surge_z = (
            0.0
            if (in_place_active or is_jet)
            else ((0.30 * float(flight.lift_recoil_u) * recoil_scale * math.cos(theta)) if mode in ("hover", "takeoff_burst") else 0.0)
        )
        adv_z = 0.0 if in_place_active else (fwd_z * spd_u_s * t_s)
        root_delta: Vec3 = (0.0, float(lift_recoil_y), float(adv_z + fwd_z * surge_z))
        root_deltas_list.append(root_delta)

        pose_map: dict[str, np.ndarray] = {
            b: np.asarray(v, dtype=np.float64).copy() for b, v in state_pose_bias.items()
        }
        frame_pos_deltas: dict[str, Vec3] = {}
        frame_scale_factors: dict[str, Vec3] = {}

        # 1. Explicit tracks, keyframes & additive layers
        for tr in clip.tracks:
            ang = _evaluate_track_angle(tr.law, t_s, dur_s, tr.amp_deg, tr.freq_hz, tr.phase_deg, tr.bias_deg)
            vec = pose_map.setdefault(tr.bone, np.zeros(3, dtype=np.float64))
            vec[axis_idx[tr.axis]] += ang
        for kf in clip.keyframes:
            kv = evaluate_keyframe_track(kf, t_s)
            if kf.channel == "rotation":
                vec = pose_map.setdefault(kf.bone, np.zeros(3, dtype=np.float64))
                vec += np.asarray(kv, dtype=np.float64)
            elif kf.channel == "position":
                pp = frame_pos_deltas.get(kf.bone, (0.0, 0.0, 0.0))
                frame_pos_deltas[kf.bone] = (pp[0] + kv[0], pp[1] + kv[1], pp[2] + kv[2])
            elif kf.channel == "scale":
                ps = frame_scale_factors.get(kf.bone, (1.0, 1.0, 1.0))
                frame_scale_factors[kf.bone] = (ps[0] * kv[0], ps[1] * kv[1], ps[2] * kv[2])
        for lyr_batch in batched_layers:
            l_eul, l_pos = lyr_batch[f_idx]
            for lb_n, lv in l_eul.items():
                vec = pose_map.setdefault(lb_n, np.zeros(3, dtype=np.float64))
                vec += np.asarray(lv, dtype=np.float64)
            for lb_n, lpv in l_pos.items():
                pp = frame_pos_deltas.get(lb_n, (0.0, 0.0, 0.0))
                frame_pos_deltas[lb_n] = (pp[0] + lpv[0], pp[1] + lpv[1], pp[2] + lpv[2])

        # 2. Axial torso pitch & aerodynamic / jet-thruster counter-recoil
        pitch_bias = -float(flight.body_pitch_deg) * pitch_bias_scale * fwd_z
        # During power downstroke (or jet thrust pulse), thrust pitches chest slightly upward
        pitch_dyn = -pitch_osc_deg * math.sin((u_jet if is_jet else theta) - 0.15 * math.pi) * fwd_z
        non_root_spine = [b for b in morph.spine if b != root_bone_name]
        if non_root_spine:
            # Share base hover body attitude across root (45%) + spine vertebrae (55%)
            rv = pose_map.setdefault(root_bone_name, np.zeros(3, dtype=np.float64))
            rv[0] += 0.45 * pitch_bias + 0.35 * pitch_dyn
            if is_jet and abs(bank_deg) > 1e-4:
                rv[2] += bank_deg * math.sin(u_jet)
            per_sp_bias = 0.55 * pitch_bias / float(len(non_root_spine))
            per_sp_dyn = 0.65 * pitch_dyn / float(len(non_root_spine))
            for s_i, sb in enumerate(non_root_spine):
                sv = pose_map.setdefault(sb, np.zeros(3, dtype=np.float64))
                sv[0] += per_sp_bias + per_sp_dyn * math.cos(0.25 * s_i)
                if mode == "swim_fins":
                    sv[1] += 7.0 * math.sin(theta - 0.55 * s_i)
        else:
            rv = pose_map.setdefault(root_bone_name, np.zeros(3, dtype=np.float64))
            rv[0] += pitch_bias + pitch_dyn
            if is_jet and abs(bank_deg) > 1e-4:
                rv[2] += bank_deg * math.sin(u_jet)

        # 3. Universal Multi-Pair Aerodynamic Wing / Airfoil / Flipper OR Multi-Nozzle Jet-Thruster Synthesis
        # Compute canonical left-wing deltas per pair first, then mirror-reflect for right pair!
        pair_left_deltas: dict[tuple[int, int], np.ndarray] = {}
        pair_left_finger_deltas: dict[tuple[int, int], np.ndarray] = {}

        # Sort so side='l' runs before side='r' for exact mirror pairing
        ordered_wings = sorted(morph.wings, key=lambda w: (w.pair_index, 0 if w.side == "l" else (1 if w.side == "r" else 2)))
        for wg in ordered_wings:
            pair_lag_rad = math.radians(float(flight.pair_phase_deg) * wg.pair_index)
            phi_w = (cyc_phi - (float(flight.pair_phase_deg) * wg.pair_index) / 360.0) % 1.0
            th_w = _asymmetric_wing_phase(phi_w, downstroke_frac=down_frac)
            ax_info = wing_axes[wg.role]
            down_ax, down_sgn = ax_info["down"]
            fwd_ax, fwd_sgn = ax_info["fwd"]
            twist_ax, twist_sgn = ax_info["twist"]

            # Enforce v25 Anti-Split Wing Membrane Law: locked forearm spars stay strictly [0, 0, 0]
            for lb_lock in wg.locked_spar_bones:
                pose_map[lb_lock] = np.zeros(3, dtype=np.float64)

            is_mirrored_right = (
                wg.side == "r"
                and wg.mirror_of is not None
                and wg.pair_index in left_role_for_pair
                and wg.chain
                and rest_scene.bone_by_name[wg.chain[0]].mirrored
            )

            wing_tip_ref = rest_scene.bone_by_name[wg.chain[-1]].pivot_world if len(wg.chain) >= 2 else None
            if is_jet:
                # VTOL / Supersonic Jet-Thruster Propulsion: ZERO bird-flapping or forward sweep bias!
                # Synthesizes crisp high-frequency thrust-recoil micro-trim on shoulder/nacelle + independent
                # phased nozzle gimbal vectoring across every engine ray in `finger_bones` (e.g. 3 thrusters/wing).
                u_w = 2.0 * math.pi * phi_w
                tremor_gain = 1.25 if mode == "jet_boost" else (0.75 if mode == "jet_cruise" else 1.0)
                if wg.chain:
                    b0 = wg.chain[0]
                    if is_mirrored_right and (wg.pair_index, 0) in pair_left_deltas:
                        d_l = pair_left_deltas[(wg.pair_index, 0)]
                        d0 = np.array([d_l[0], -d_l[1], -d_l[2]], dtype=np.float64)
                    else:
                        sh_pitch = 0.35 * gimbal_deg * tremor_gain * math.sin(2.0 * u_w)
                        sh_yaw = 0.22 * gimbal_deg * tremor_gain * math.cos(u_w)
                        d0 = _project_world_aero_to_bone_euler(
                            rest_scene, b0, wg.side, fwd_z, sh_pitch, sh_yaw, 0.0, tip_ref_world=wing_tip_ref
                        )
                        if wg.side == "l":
                            pair_left_deltas[(wg.pair_index, 0)] = d0.copy()
                    pose_map.setdefault(b0, np.zeros(3, dtype=np.float64))[:] += d0
                if len(wg.chain) >= 2:
                    b1 = wg.chain[1]
                    if is_mirrored_right and (wg.pair_index, 1) in pair_left_deltas:
                        d_l = pair_left_deltas[(wg.pair_index, 1)]
                        d1 = np.array([d_l[0], -d_l[1], -d_l[2]], dtype=np.float64)
                    else:
                        nac_pitch = 0.55 * gimbal_deg * tremor_gain * (0.65 * math.sin(u_w - 0.35) + 0.35 * math.sin(3.0 * u_w))
                        nac_yaw = 0.30 * gimbal_deg * tremor_gain * math.cos(2.0 * u_w)
                        b1_tip = rest_scene.bone_by_name[wg.chain[2]].pivot_world if len(wg.chain) >= 3 else None
                        d1 = _project_world_aero_to_bone_euler(
                            rest_scene, b1, wg.side, fwd_z, nac_pitch, nac_yaw, 0.0, tip_ref_world=b1_tip
                        )
                        if wg.side == "l":
                            pair_left_deltas[(wg.pair_index, 1)] = d1.copy()
                    pose_map.setdefault(b1, np.zeros(3, dtype=np.float64))[:] += d1
                n_eng = max(1, len(wg.finger_bones))
                for fi, fb in enumerate(wg.finger_bones):
                    if fb in wg.locked_spar_bones:
                        continue
                    if is_mirrored_right and (wg.pair_index, fi) in pair_left_finger_deltas:
                        df_l = pair_left_finger_deltas[(wg.pair_index, fi)]
                        df = np.array([df_l[0], -df_l[1], -df_l[2]], dtype=np.float64)
                    else:
                        # Independent phased gimbal micro-vectoring per thruster engine (exact integer harmonics!)
                        eng_ph = u_w + (2.0 * math.pi * fi / float(n_eng))
                        eng_pitch = gimbal_deg * tremor_gain * (0.62 * math.sin(eng_ph) + 0.38 * math.sin(3.0 * u_w + 1.05 * fi))
                        eng_yaw = gimbal_deg * tremor_gain * 0.42 * math.cos(eng_ph)
                        df = _project_world_aero_to_bone_euler(
                            rest_scene, fb, wg.side, fwd_z, eng_pitch, eng_yaw, 0.0
                        )
                        if wg.side == "l":
                            pair_left_finger_deltas[(wg.pair_index, fi)] = df.copy()
                    pose_map.setdefault(fb, np.zeros(3, dtype=np.float64))[:] += df
                continue

            # A. Shoulder / Wing Root (chain[0]): True SO(3) World-Frame Figure-8 Elliptical Aero-Stroke!
            if wg.chain:
                b0 = wg.chain[0]
                if is_mirrored_right and (wg.pair_index, 0) in pair_left_deltas:
                    d_l = pair_left_deltas[(wg.pair_index, 0)]
                    d0 = np.array([d_l[0], -d_l[1], -d_l[2]], dtype=np.float64)
                else:
                    rest_elev = float(ax_info.get("rest_elev_deg", (0, 0.0))[1])
                    max_up_lift = max(4.0, 56.0 - rest_elev)
                    raw_amp = float(flight.stroke_deg) * stroke_scale
                    up_amp = min(raw_amp, max_up_lift)
                    down_amp = raw_amp
                    s_wave = -math.cos(th_w)
                    raw_flap = dihedral_bias_deg + (down_amp * s_wave if s_wave >= 0.0 else up_amp * s_wave)
                    flap_val = max(-max_up_lift, raw_flap)
                    base_fwd_b = 14.0 if float(wg.span) >= 75.0 else 6.0
                    fwd_bias = base_fwd_b if mode in ("hover", "takeoff_burst", "mantle") else (0.75 * base_fwd_b if mode in ("cruise", "glide", "soar") else 0.0)
                    sweep_val = fwd_bias + float(flight.sweep_deg) * sweep_scale * (math.sin(th_w) + 0.18 * math.sin(2.0 * th_w))
                    pron_val = float(flight.pronation_deg) * pron_scale * math.sin(th_w)
                    d0 = _project_world_aero_to_bone_euler(
                        rest_scene, b0, wg.side, fwd_z, flap_val, sweep_val, pron_val, tip_ref_world=wing_tip_ref
                    )
                    if wg.side == "l":
                        pair_left_deltas[(wg.pair_index, 0)] = d0.copy()
                vec0 = pose_map.setdefault(b0, np.zeros(3, dtype=np.float64))
                vec0 += d0

            # B. Elbow / Forearm (chain[1]): SO(3) Aerodynamic Upstroke Fold + Phase-Lagged Downstroke Extension!
            if len(wg.chain) >= 2:
                b1 = wg.chain[1]
                if is_mirrored_right and (wg.pair_index, 1) in pair_left_deltas:
                    d_l = pair_left_deltas[(wg.pair_index, 1)]
                    d1 = np.array([d_l[0], -d_l[1], -d_l[2]], dtype=np.float64)
                else:
                    th_elbow = th_w - math.radians(34.0)
                    upstroke_tuck = max(0.0, -math.sin(th_w)) ** 1.35
                    elbow_lag_flap = 0.58 * float(flight.elbow_fold_deg) * fold_scale * (-math.cos(th_elbow))
                    elbow_fold_sweep = -float(flight.elbow_fold_deg) * fold_scale * (0.75 * upstroke_tuck - 0.22 * max(0.0, math.sin(th_w)))
                    b1_tip = rest_scene.bone_by_name[wg.chain[2]].pivot_world if len(wg.chain) >= 3 else None
                    d1 = _project_world_aero_to_bone_euler(
                        rest_scene, b1, wg.side, fwd_z, elbow_lag_flap, elbow_fold_sweep, 0.0, tip_ref_world=b1_tip
                    )
                    if wg.side == "l":
                        pair_left_deltas[(wg.pair_index, 1)] = d1.copy()
                vec1 = pose_map.setdefault(b1, np.zeros(3, dtype=np.float64))
                vec1 += d1

            # C. Wrist / Hand-Wing (chain[2]): SO(3) Distal Whip Propagation (`~62 deg` lag) + Camber Cup!
            if len(wg.chain) >= 3:
                b2 = wg.chain[2]
                if is_mirrored_right and (wg.pair_index, 2) in pair_left_deltas:
                    d_l = pair_left_deltas[(wg.pair_index, 2)]
                    d2 = np.array([d_l[0], -d_l[1], -d_l[2]], dtype=np.float64)
                else:
                    th_wrist = th_w - math.radians(62.0)
                    upstroke_feath = max(0.0, -math.sin(th_wrist))
                    wrist_whip = float(flight.wrist_lag_deg) * stroke_scale * (-math.cos(th_wrist))
                    wrist_camber = float(flight.wrist_lag_deg) * 0.45 * fold_scale * (0.35 * math.sin(th_wrist) - 0.65 * upstroke_feath)
                    d2 = _project_world_aero_to_bone_euler(
                        rest_scene, b2, wg.side, fwd_z, wrist_whip, wrist_camber, 0.0
                    )
                    if wg.side == "l":
                        pair_left_deltas[(wg.pair_index, 2)] = d2.copy()
                vec2 = pose_map.setdefault(b2, np.zeros(3, dtype=np.float64))
                vec2 += d2

            # D. Distal Radiating Membrane Fingers (f1, f2, f3): Progressive SO(3) Fan & Tip Whip!
            for fi, fb in enumerate(wg.finger_bones):
                if is_mirrored_right and (wg.pair_index, fi) in pair_left_finger_deltas:
                    df_l = pair_left_finger_deltas[(wg.pair_index, fi)]
                    df = np.array([df_l[0], -df_l[1], -df_l[2]], dtype=np.float64)
                else:
                    th_f = th_w - math.radians(74.0 + 12.0 * fi)
                    fan_spread = float(flight.finger_fan_deg) * (1.0 - 0.15 * fi) * math.sin(th_f)
                    tip_whip = float(flight.finger_fan_deg) * 0.80 * (-math.cos(th_f))
                    df = _project_world_aero_to_bone_euler(
                        rest_scene, fb, wg.side, fwd_z, tip_whip, fan_spread, 0.0
                    )
                    if wg.side == "l":
                        pair_left_finger_deltas[(wg.pair_index, fi)] = df.copy()
                vecf = pose_map.setdefault(fb, np.zeros(3, dtype=np.float64))
                vecf += df

        # 3b. Supple Phase-Delayed Aerial Serpentine Neck Wave (synchronized with wingbeat recoil!)
        neck_chains_air = morph.neck_chains if morph.neck_chains else ((morph.neck,) if morph.neck else ())
        for nc_i, n_chain_a in enumerate(neck_chains_air):
            valid_na = [nb for nb in n_chain_a if nb != root_bone_name]
            if len(valid_na) >= 2:
                n_na = len(valid_na)
                na_pitch_amp = max(2.5, 0.65 * pitch_osc_deg + 2.2) / math.sqrt(float(n_na))
                na_yaw_amp = (2.4 if mode != "glide" else 3.2) / math.sqrt(float(n_na))
                for n_i, nb in enumerate(valid_na):
                    nv = pose_map.setdefault(nb, np.zeros(3, dtype=np.float64))
                    lag_na = 0.55 * float(n_i) + 0.35 * float(nc_i)
                    if n_na >= 3 and n_i == n_na - 1:
                        nv[0] += -0.45 * na_pitch_amp * math.sin(theta - lag_na)
                        nv[1] += -0.40 * na_yaw_amp * math.cos(theta - lag_na)
                    else:
                        env_na = 0.80 + 0.40 * math.sin(math.pi * (float(n_i + 0.5) / float(max(1, n_na))))
                        nv[0] += na_pitch_amp * env_na * math.sin(theta - lag_na - 0.35 * math.pi)
                        nv[1] += na_yaw_amp * env_na * math.cos(theta - lag_na)

        # 4. Jacobian-Verified SO(3) Upward Aerial Limb Tuck (lifts wrists/ankles UPWARD toward torso, NEVER into floor!)
        tuck = float(np.clip(flight.limb_tuck, 0.0, 1.0))
        if mode != "mantle" and tuck > 1e-4:
            for lb_i, lb in enumerate(morph.limbs):
                phase_lb = (u_jet if is_jet else theta) - 0.45 * math.pi - 0.25 * (1 if lb.role.startswith("hind") else 0)
                dangle_osc = (0.85 if is_jet else 3.5) * math.sin(phase_lb)
                is_fore = lb.role.startswith("fore")
                root_lb = rest_scene.bone_by_name[lb.chain[0]]
                eff_pt = np.asarray(lb.contact_world, dtype=np.float64)
                r_limb = eff_pt - root_lb.pivot_world
                j_z_limb = np.cross(root_lb.rotation[:, 2], r_limb)
                outward_x = 1.0 if float(root_lb.pivot_world[0]) >= 0.0 else -1.0
                abduct_z_sign = 1.0 if (float(j_z_limb[0]) * outward_x) >= 0.0 else -1.0
                for seg_i, bname in enumerate(lb.chain):
                    wb = rest_scene.bone_by_name[bname]
                    lv = pose_map.setdefault(bname, np.zeros(3, dtype=np.float64))
                    r_seg = eff_pt - wb.pivot_world
                    j_x_seg = np.cross(wb.rotation[:, 0], r_seg)
                    # Exact SO(3) Jacobian sign that lifts distal foot UPWARD (+Y) in world space!
                    lift_x_sign = 1.0 if float(j_x_seg[1]) >= 0.0 else -1.0
                    if seg_i == 0:
                        lv[2] += abduct_z_sign * (8.0 if is_fore else -3.0) * tuck
                        up_deg = (22.0 if is_fore else 29.0) * tuck
                        if mode in ("cruise", "glide", "soar", "jet_cruise", "jet_boost"):
                            up_deg = (38.0 if is_fore else 34.0) * tuck
                        lv[0] += lift_x_sign * up_deg + dangle_osc
                    elif seg_i == 1 and len(lb.chain) >= 4:
                        # Digitigrade shin is already deeply flexed in rest pose: gentle tuck avoids calf-hamstring jam!
                        lv[0] += lift_x_sign * 8.0 * tuck + 0.4 * dangle_osc
                    elif seg_i == 1 or (seg_i == 2 and len(lb.chain) >= 4):
                        lv[0] += lift_x_sign * (28.0 if is_fore else 26.0) * tuck + 0.5 * dangle_osc
                    else:
                        # Relaxed raptor/dragon wrist/foot curl (streamlined landing gear retraction!)
                        lv[0] += (14.0 if is_jet else 10.0) * tuck + 0.3 * dangle_osc

        # 5. Aerodynamic Counter-Balancing Tail Chains & Tentacles (ALL discovered chains!)
        tail_chains = morph.tail_chains if morph.tail_chains else ((morph.tail,) if morph.tail else ())
        for tc_idx, t_chain in enumerate(tail_chains):
            if not t_chain:
                continue
            amp_base = float(flight.tail_counter_deg) / max(1.0, math.sqrt(float(len(t_chain))))
            tc_phase_off = 0.65 * tc_idx
            for t_i, tb in enumerate(t_chain):
                if tb == root_bone_name:
                    continue
                tv = pose_map.setdefault(tb, np.zeros(3, dtype=np.float64))
                lag = 0.55 * (t_i + 1)
                if is_jet:
                    # Taut supersonic / VTOL aerodynamic stabilizer fin trim + counter-rudder on bank!
                    tv[0] += 0.45 * amp_base * (1.0 + 0.10 * t_i) * math.sin(u_jet - 0.35 * lag)
                    tv[1] += (0.35 * amp_base * math.sin(u_jet - 0.40 * lag + tc_phase_off)) - (
                        0.22 * bank_deg / max(1.0, float(len(t_chain))) * math.sin(u_jet)
                    )
                else:
                    # Pitch rudder wave counters chest recoil; graceful S-curve yaw swish adds aerial life
                    tv[0] += (4.5 * pitch_bias_scale + amp_base * (1.0 + 0.14 * t_i) * math.sin(theta - lag - 0.5 * math.pi))
                    tv[1] += 0.55 * amp_base * (1.0 + 0.16 * t_i) * math.sin(theta - lag + tc_phase_off)

        for tent_idx, tent in enumerate(morph.tentacles):
            for s_i, sb in enumerate(tent):
                if sb == root_bone_name:
                    continue
                sv = pose_map.setdefault(sb, np.zeros(3, dtype=np.float64))
                sv[0] += 10.0 * math.sin(theta - 0.6 * s_i + 0.8 * tent_idx)
                sv[1] += 8.0 * math.cos(theta - 0.6 * s_i + 0.8 * tent_idx)

        # 6. Neural Muscle Activations (explicit muscle_drives + automatic pectoral downstroke recruitment!)
        # In jet VTOL/cruise/boost modes, NEVER auto-recruit flapping wing muscles (wings are rigid jet-thruster pods!).
        acts = dict(state_muscle_tone)
        if is_jet:
            downstroke_effort = float(np.clip(0.55 + 0.35 * math.sin(u_jet), 0.0, 1.0))
        else:
            downstroke_effort = float(np.clip(0.5 + 0.5 * math.sin(theta - 0.1 * math.pi), 0.0, 1.0))
            for mus in getattr(program, "muscles", ()):
                if mus.insertion_bone in wing_bone_set or "wing" in mus.name.lower() or "pect" in mus.name.lower() or "chest" in mus.name.lower():
                    acts[mus.name] = max(acts.get(mus.name, 0.0), 0.15 + 0.85 * downstroke_effort)
        for md in clip.muscle_drives:
            u_m = (t_s * md.freq_hz + md.phase_deg / 360.0) % 1.0
            val = md.baseline + (md.peak - md.baseline) * (0.5 - 0.5 * math.cos(2.0 * math.pi * u_m))
            acts[md.muscle] = float(np.clip(max(acts.get(md.muscle, 0.0), val), 0.0, 1.0))
        acts_list.append(acts)

        # Apply Sim(3) volume-preserving muscle scale bulge
        _apply_muscle_scale_bulge(program, rest_scene, acts, frame_scale_factors)

        # Soft biological SO(3) limit clamping on primary pose
        for b_k in list(pose_map):
            if b_k in limits_map:
                jlim, asign = limits_map[b_k]
                pose_map[b_k] = soft_clamp_euler_to_limits(pose_map[b_k], jlim, angle_sign=asign)

        base_pose_tuples: dict[str, Vec3] = {
            b: (float(v[0]), float(v[1]), float(v[2])) for b, v in pose_map.items()
        }

        # 7. Zajac/Hill @muscles equilibrium + biarticular tendon coupling (only on non-wing muscles unless driven)
        if program.muscles and acts:
            ms_res = evaluate_muscles_at_activations(
                program,
                acts,
                base_pose_deltas=base_pose_tuples,
                root_translation_delta=root_delta,
                pose_position_deltas=frame_pos_deltas or None,
                pose_scale_factors=frame_scale_factors or None,
                validate=False,
                _rest_scene=rest_scene,
                _limits_map=limits_map,
                _skip_scene=True,
            )
            cur_poses = dict(ms_res.pose_euler_deltas)
            # Re-protect locked wing membrane anchor spars after any muscle coupling
            for wg in morph.wings:
                for lb_lock in wg.locked_spar_bones:
                    cur_poses[lb_lock] = (0.0, 0.0, 0.0)
        else:
            cur_poses = base_pose_tuples

        # 8. Vestibulo-Ocular Gaze Stabilization (on >= 4-bone serpentine necks, stabilize only distal atlas/head!)
        stab_strength = max(float(flight.head_stabilize), float(getattr(clip, "head_stabilize", 0.0)))
        if stab_strength > 1e-4:
            neck_chains_all = morph.neck_chains if morph.neck_chains else ((morph.neck,) if morph.neck else ())
            if program.kinematics is not None and program.kinematics.look_chains:
                for lkc in program.kinematics.look_chains:
                    stab_bones = lkc.bones[-2:] if len(lkc.bones) >= 4 else lkc.bones
                    eff_stab = min(0.38, stab_strength) if len(lkc.bones) >= 4 else stab_strength
                    cur_poses = solve_vestibulo_ocular_stabilization(
                        program,
                        rest_scene,
                        stab_bones,
                        cur_poses,
                        strength=eff_stab,
                        weight_decay=float(lkc.weight_decay),
                        limits_map=limits_map,
                        root_translation_delta=root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                    )
            else:
                for n_ch in neck_chains_all:
                    stab_bones = n_ch[-2:] if len(n_ch) >= 4 else n_ch
                    eff_stab = min(0.38, stab_strength) if len(n_ch) >= 4 else stab_strength
                    cur_poses = solve_vestibulo_ocular_stabilization(
                        program,
                        rest_scene,
                        stab_bones,
                        cur_poses,
                        strength=eff_stab,
                        limits_map=limits_map,
                        root_translation_delta=root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                    )

        # 9. In grounded `mantle` mode, pin all grounded feet cleanly to ground_y via world SE(3) IK!
        eff_positions: dict[str, Vec3] = {}
        eff_stance: dict[str, bool] = {}
        eff_step_ids: dict[str, int] = {}
        if mode == "mantle" and morph.limbs:
            for lb in morph.limbs:
                tgt_w = np.array([lb.contact_world[0], morph.ground_y, lb.contact_world[2]], dtype=np.float64)
                decl_ik = ik_decl_by_name.get(lb.ik_chain_name or "")
                hinge_ax = decl_ik.hinge_axis if decl_ik is not None else "x"
                pole_w = decl_ik.pole if decl_ik is not None else None
                plane_mode = (
                    decl_ik.plane
                    if (decl_ik is not None and decl_ik.plane is not None)
                    else ("sprawled" if lb.posture == "sprawled" else "sagittal")
                )
                shifted_tgt = (
                    float(tgt_w[0] - root_delta[0]),
                    float(tgt_w[1] - root_delta[1]),
                    float(tgt_w[2] - root_delta[2]),
                )
                ik_res = solve_n_bone_ik(
                    program,
                    lb.chain,
                    lb.contact_local,
                    shifted_tgt,
                    base_pose_deltas=cur_poses,
                    hinge_axis=hinge_ax,
                    pole_world=pole_w,
                    plane=plane_mode,
                    max_iters=32,
                    tol=1e-4,
                    validate=False,
                    _limits_cache=limits_map,
                    pose_scale_factors=frame_scale_factors or None,
                )
                cur_poses = ik_res.pose_euler_deltas
                eff_positions[lb.role] = (
                    float(ik_res.achieved_world[0] + root_delta[0]),
                    float(ik_res.achieved_world[1] + root_delta[1]),
                    float(ik_res.achieved_world[2] + root_delta[2]),
                )
                eff_stance[lb.role] = True
                eff_step_ids[lb.role] = 0

        # 9b. Full-Body Wing Pectoral Bulge, Spline-IK Neck, Aerial Talon Curl & Whole-Body Sim(3) Bulge
        if rig_asm is not None:
            from synccraft.kinematics.whole_body_solver import (
                solve_com_balance_and_counterweight,
                solve_foot_roll_and_digit_articulation,
                solve_full_body_antagonist_muscle_bulge,
                solve_spine_neck_spline_ik,
                solve_wing_full_anatomy,
            )

            if pec_g > 1e-4 or (ground_clr and mode in ("mantle", "takeoff_burst")):
                cur_poses, frame_scale_factors = solve_wing_full_anatomy(
                    program,
                    rest_scene,
                    rig_asm,
                    cur_poses,
                    frame_scale_factors,
                    downstroke_effort=downstroke_effort,
                    pectoral_bulge=pec_g,
                    ground_clearance=(ground_clr and mode in ("mantle", "takeoff_burst")),
                    root_translation_delta=root_delta,
                    limits_map=limits_map,
                )
            if neck_aim_tgt is not None or abs(spine_coil_deg) > 1e-4:
                for na in rig_asm.neck_assemblies:
                    cur_poses = solve_spine_neck_spline_ik(
                        program,
                        rest_scene,
                        na.chain,
                        cur_poses,
                        aim_target_world=neck_aim_tgt,
                        s_coil_deg=spine_coil_deg,
                        limits_map=limits_map,
                        root_translation_delta=root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                        forward_sign_z=fwd_z,
                    )
            if com_g > 1e-4:
                cur_poses, root_delta = solve_com_balance_and_counterweight(
                    program,
                    rest_scene,
                    rig_asm,
                    cur_poses,
                    root_delta,
                    strength=com_g,
                    in_place=bool(flight.in_place or getattr(clip, "in_place", False)),
                    grounded=(mode == "mantle"),
                    limits_map=limits_map,
                    pose_position_deltas=frame_pos_deltas or None,
                    pose_scale_factors=frame_scale_factors or None,
                )
                root_deltas_list[-1] = root_delta
            if digit_g > 1e-4 or digit_grasp_g > 1e-4:
                cur_poses = solve_foot_roll_and_digit_articulation(
                    rig_asm,
                    cur_poses,
                    limb_in_stance=eff_stance if mode == "mantle" else None,
                    foot_roll_strength=0.0,
                    digit_strength=digit_g,
                    grasp_amount=digit_grasp_g,
                    aerial_tuck=(mode != "mantle"),
                    limits_map=limits_map,
                )
            if antag_g > 1e-4:
                frame_scale_factors = solve_full_body_antagonist_muscle_bulge(
                    rig_asm,
                    cur_poses,
                    frame_scale_factors,
                    strength=antag_g,
                    extra_effort_by_region={"pectoral": downstroke_effort, "wing_arm": downstroke_effort},
                    preserve_grounded_limb_length=(mode == "mantle"),
                )
            if mode == "mantle" and morph.limbs:
                from synccraft.kinematics.whole_body_solver import solve_standing_limb_ik_pinning

                cur_poses, pin_pos, pin_st, pin_sid = solve_standing_limb_ik_pinning(
                    program,
                    rest_scene,
                    rig_asm,
                    cur_poses,
                    pin_spec=True,
                    root_translation_delta=root_delta,
                    pose_position_deltas=frame_pos_deltas or None,
                    pose_scale_factors=frame_scale_factors or None,
                    limits_map=limits_map,
                )
                eff_positions.update(pin_pos)
                eff_stance.update(pin_st)
                eff_step_ids.update(pin_sid)

        pos_deltas_list.append(frame_pos_deltas)
        scale_factors_list.append(frame_scale_factors)
        pre_spring_poses_list.append(cur_poses)
        eff_pos_list.append(eff_positions)
        eff_stance_list.append(eff_stance)
        eff_step_list.append(eff_step_ids)

    # 10. Implicit-Euler Secondary Spring Follow-Through (ears, tail tips, cape ruffs — excluding locked wing spars & mantle legs)
    protected_bones = {bn for w in morph.wings for bn in w.locked_spar_bones}
    if mode == "mantle":
        protected_bones.update(bn for lb in morph.limbs for bn in lb.chain)
    active_springs = [sp for sp in all_springs if sp.bone not in protected_bones]
    if active_springs and len(phases) >= 2:
        final_poses_list = _integrate_spring_follow_through(
            program,
            rest_scene,
            active_springs,
            times_list,
            clip.loop,
            dur_s,
            pre_spring_poses_list,
            root_deltas_list,
            limits_map,
        )
    else:
        final_poses_list = pre_spring_poses_list

    # 11. Final exact SE(3) + Sim(3) scene compilation & isochoric muscle bulge
    samples: list[BehaviorFrameSample] = []
    for f_idx in range(len(phases)):
        cur_poses = final_poses_list[f_idx]
        root_delta = root_deltas_list[f_idx]
        acts = acts_list[f_idx]
        f_pos = pos_deltas_list[f_idx]
        f_scl = scale_factors_list[f_idx]

        kin_scene = compile_posed_geometry(
            program,
            cur_poses,
            root_translation_delta=root_delta,
            pose_position_deltas=f_pos or None,
            pose_scale_factors=f_scl or None,
            validate=False,
        )
        frame_inflate: dict[str, float] = {}
        if program.muscles:
            frame_inflate, _ = _compute_muscle_bulge_and_telemetry(
                rest_scene, kin_scene, program.muscles, acts, cur_poses
            )
        final_scene = (
            compile_posed_geometry(
                program,
                cur_poses,
                box_inflate_deltas=frame_inflate,
                root_translation_delta=root_delta,
                pose_position_deltas=f_pos or None,
                pose_scale_factors=f_scl or None,
                validate=False,
            )
            if frame_inflate
            else kin_scene
        )
        samples.append(
            BehaviorFrameSample(
                frame_index=f_idx,
                time_s=times_list[f_idx],
                phase=phases_list[f_idx],
                pose_euler_deltas=cur_poses,
                box_inflate_deltas=frame_inflate,
                root_translation_delta=root_delta,
                muscle_activations=acts,
                effector_world_positions=eff_pos_list[f_idx],
                effector_in_stance=eff_stance_list[f_idx],
                effector_stance_step_id=eff_step_list[f_idx],
                scene=final_scene,
                pose_position_deltas=f_pos,
                pose_scale_factors=f_scl,
                root_motion_velocity=rm_vel,
            )
        )
    return samples


def audit_flight_clip_contracts(
    program: Program,
    clip: ClipDecl,
    frames: list[Any],
    *,
    precomputed_collision_events: int | None = None,
) -> dict[str, Any]:
    """Verify 6 universal aerodynamic flight contracts (in_place, bilateral symmetry, SO(3) limits, loop, SAT, lift recoil)."""
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.dsl.ast_nodes import _is_continuous_limit_pair
    from synccraft.kinematics.musculoskeletal import _collect_expanded_joint_limits

    flight = clip.flights[0]
    rest_scene = compile_geometry(program, validate=False)
    morph = analyze_morphology(program, rest_scene)
    explicit_limits = _collect_expanded_joint_limits(program)
    use_auto_bio = bool(flight.auto_bio_limits or getattr(clip, "auto_bio_limits", False) or getattr(clip, "auto_rig", False))
    limits_map = (
        infer_universal_bio_constraints(program, rest_scene, morph, explicit_limits=explicit_limits)
        if use_auto_bio
        else explicit_limits
    )

    # 1. In-place entity-space contract: no macro altitude (+35u) or macro horizontal drift when in_place=True
    max_root_y = max((abs(float(fr.root_translation_delta[1])) for fr in frames), default=0.0)
    max_root_z = max((abs(float(fr.root_translation_delta[2])) for fr in frames), default=0.0)
    in_place_active = bool(flight.in_place or getattr(clip, "in_place", False))
    max_allowed_y = max(1.0, float(flight.lift_recoil_u) * 1.65 + 0.5)
    max_allowed_z = max(1.0, float(flight.lift_recoil_u) * 0.65 + 0.5)
    in_place_ok = True if not in_place_active else (max_root_y <= max_allowed_y and max_root_z <= max_allowed_z)

    # 2. Bilateral wing tip symmetry in root body-local frame (handles level flight AND banked fighter-jet turns!)
    max_asym_u = 0.0
    wing_by_role = {w.role: w for w in morph.wings}
    root_b_name = rest_scene.bones[0].name
    for wg in morph.wings:
        if wg.mirror_of and wg.mirror_of in wing_by_role and wg.chain and wing_by_role[wg.mirror_of].chain:
            tip_r = wg.chain[-1]
            tip_l = wing_by_role[wg.mirror_of].chain[-1]
            for fr in frames:
                rb = fr.scene.bone_by_name[root_b_name]
                pr_loc = rb.rotation.T @ (fr.scene.bone_by_name[tip_r].pivot_world - rb.pivot_world)
                pl_loc = rb.rotation.T @ (fr.scene.bone_by_name[tip_l].pivot_world - rb.pivot_world)
                asym = float(math.hypot(pr_loc[1] - pl_loc[1], pr_loc[2] - pl_loc[2]))
                max_asym_u = max(max_asym_u, asym)
    symmetry_ok = max_asym_u <= 0.5

    # 3. SO(3) biological joint limits
    max_limit_excess_deg = 0.0
    for fr in frames:
        for bname, eul in fr.pose_euler_deltas.items():
            if bname not in limits_map:
                continue
            jlim, ang_sign = limits_map[bname]
            for ax_i in range(3):
                if _is_continuous_limit_pair(jlim[ax_i]):
                    continue
                lo_raw, hi_raw = float(jlim[ax_i][0]), float(jlim[ax_i][1])
                if float(ang_sign[ax_i]) < 0.0:
                    lo_raw, hi_raw = -hi_raw, -lo_raw
                val = float(eul[ax_i])
                if val < lo_raw:
                    max_limit_excess_deg = max(max_limit_excess_deg, lo_raw - val)
                elif val > hi_raw:
                    max_limit_excess_deg = max(max_limit_excess_deg, val - hi_raw)
    joint_limits_ok = max_limit_excess_deg <= 1e-2

    # 4. Clipping (15-axis OBB SAT)
    if precomputed_collision_events is not None:
        clip_collisions = int(precomputed_collision_events)
    else:
        clip_collisions = 0
        for fr in frames:
            aud = audit_scene(fr.scene, enforce_physical_contact=False)
            clip_collisions += len(aud["collision_pairs"])
    clipping_ok = clip_collisions == 0

    # 5. Loop closure
    max_loop_jump_deg = 0.0
    if clip.loop and len(frames) >= 2:
        endpoints = evaluate_flight_clip(program, clip, sample_phases=[0.0, 1.0])
        p0, p1 = endpoints[0].pose_euler_deltas, endpoints[1].pose_euler_deltas
        for bname in set(p0) | set(p1):
            v0 = np.asarray(p0.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            v1 = np.asarray(p1.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            max_loop_jump_deg = max(max_loop_jump_deg, float(np.max(np.abs(v1 - v0))))
    loop_ok = max_loop_jump_deg <= 0.75

    # 6. Locked inner-membrane anchor compliance
    max_locked_deg = 0.0
    for wg in morph.wings:
        for lb_lock in wg.locked_spar_bones:
            for fr in frames:
                v = fr.pose_euler_deltas.get(lb_lock, (0.0, 0.0, 0.0))
                max_locked_deg = max(max_locked_deg, max(abs(v[0]), abs(v[1]), abs(v[2])))
    membrane_lock_ok = max_locked_deg <= 1e-4

    checks = {
        "in_place_entity_space": in_place_ok,
        "bilateral_symmetry": symmetry_ok,
        "joint_limits": joint_limits_ok,
        "clipping": clipping_ok,
        "loop": loop_ok,
        "membrane_anchor_lock": membrane_lock_ok,
    }
    upb = float(getattr(program.meta, "units_per_block", 16))
    rm_v = frames[0].root_motion_velocity if frames else (0.0, 0.0, 0.0)
    spd_u = float(np.linalg.norm(np.asarray(rm_v, dtype=np.float64)))
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "mode": flight.mode,
        "wing_count": len(morph.wings),
        "in_place": in_place_active,
        "max_root_y_units": round(max_root_y, 4),
        "max_root_z_units": round(max_root_z, 4),
        "max_bilateral_asymmetry_units": round(max_asym_u, 5),
        "max_joint_limit_excess_deg": round(max_limit_excess_deg, 5),
        "max_loop_jump_deg": round(max_loop_jump_deg, 5),
        "max_locked_spar_deg": round(max_locked_deg, 5),
        "root_motion": {
            "in_place": in_place_active,
            "speed_units_per_s": round(spd_u, 4),
            "speed_blocks_per_s": round(spd_u / max(1.0, upb), 4),
            "velocity_vec": [round(float(c), 4) for c in rm_v],
        },
    }
