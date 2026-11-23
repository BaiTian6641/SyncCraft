"""Dimensionless Froude-Scaled Locomotion Solver & 8-Rule Biomechanical Gait Verifier (ADR 0011).

Expands compact `gait <name> { ... }` declarations against derived `@morphology` roles
using Alexander & Jayes (1983) dynamic similarity (`v = sqrt(Fr * g * h_hip)`), automatic
quadruped / biped / sprawled / hexapod phase projection, ancestor spine-motion-compensated
closed-loop world IK (`< 0.02u` stance slip), root-motion velocity synthesis, and an
8-check biomechanical verifier (`foot_slip`, `reach`, `joint_limits`, `clipping`, `balance`,
`ground`, `loop`, `speed_match`).
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from synccraft.dsl.ast_nodes import ClipDecl, GaitDecl, IKChainDecl, Program, SpringDecl, Vec3, _is_continuous_limit_pair
from synccraft.kinematics.bio_constraints import (
    infer_universal_bio_constraints,
    soft_clamp_euler_to_limits,
    solve_vestibulo_ocular_stabilization,
)
from synccraft.kinematics.flight_solver import _apply_muscle_scale_bulge
from synccraft.kinematics.morphology import (
    LimbMorphology,
    MorphologyReport,
    analyze_morphology,
    _convex_hull_2d,
    point_in_convex_polygon_margin,
)
from synccraft.kinematics.se3_chain import (
    _recover_euler_xyz_degrees,
    compile_geometry,
    compile_posed_geometry,
    euler_xyz_matrix,
)

if TYPE_CHECKING:
    from synccraft.kinematics.musculoskeletal import BehaviorFrameSample


def _default_gravity_units_s2(program: Program) -> float:
    if getattr(program.meta, "gravity", None) is not None:
        return float(program.meta.gravity)
    return 9.81 * float(getattr(program.meta, "units_per_block", 16))


def resolve_limb_phases(gait: GaitDecl, morph: MorphologyReport) -> dict[str, float]:
    """Resolve normalized `[0, 1)` touchdown phases for every ground-contacting limb role."""
    explicit = {rk: float(rv) % 1.0 for rk, rv in gait.phase}
    roles = [lb.role for lb in morph.limbs]
    has_mid = any(r.startswith("mid") for r in roles)
    has_fore = any(r.startswith("fore") for r in roles)

    pat = gait.pattern
    defaults: dict[str, float] = {}
    if pat in ("serpentine_wave", "swim_undulate") or not roles:
        return {}
    if pat == "metachronal_wave" or len(roles) >= 8:
        # Universal metachronal traveling wave for ANY polypod (6, 8, 10, 12+ legs: spiders, centipedes, aliens!)
        left_r = [lb.role for lb in morph.limbs if lb.side == "l"]
        right_r = [lb.role for lb in morph.limbs if lb.side == "r"]
        n_tiers = max(1, max(len(left_r), len(right_r)))
        for idx_l, rl in enumerate(reversed(left_r)):
            defaults[rl] = (idx_l / float(n_tiers)) % 1.0
        for idx_r, rr in enumerate(reversed(right_r)):
            defaults[rr] = (0.5 + idx_r / float(n_tiers)) % 1.0
    elif not has_fore and not has_mid:
        # Biped / standing wyvern / avian (`hind_l`, `hind_r`)
        defaults = {"hind_l": 0.0, "hind_r": 0.50}
    elif has_mid:
        # Hexapod / multipod (`fore_*`, `mid_*`, `hind_*`)
        if pat in ("diagonal_pairs", "hexapod_tripod", "sprawl_diagonal", "trot"):
            defaults = {
                "fore_l": 0.0,
                "mid_r": 0.0,
                "hind_l": 0.0,
                "fore_r": 0.50,
                "mid_l": 0.50,
                "hind_r": 0.50,
            }
        else:
            # Metachronal wave walk
            defaults = {
                "hind_l": 0.0,
                "mid_l": 1.0 / 6.0,
                "fore_l": 2.0 / 6.0,
                "hind_r": 0.50,
                "mid_r": 4.0 / 6.0,
                "fore_r": 5.0 / 6.0,
            }
    else:
        # Quadruped (`fore_l`, `fore_r`, `hind_l`, `hind_r`)
        if pat in ("diagonal_pairs", "sprawl_diagonal"):
            defaults = {"fore_l": 0.0, "hind_r": 0.0, "fore_r": 0.50, "hind_l": 0.50}
        elif pat == "rotary_gallop":
            defaults = {"hind_l": 0.0, "hind_r": 0.14, "fore_r": 0.50, "fore_l": 0.64}
        elif pat == "transverse_gallop":
            defaults = {"hind_l": 0.0, "hind_r": 0.14, "fore_l": 0.50, "fore_r": 0.64}
        elif pat == "biped_alternate":
            defaults = {"fore_l": 0.0, "hind_r": 0.0, "fore_r": 0.50, "hind_l": 0.50}
        else:
            # Default Hildebrand lateral-sequence walk
            defaults = {"hind_l": 0.0, "fore_l": 0.25, "hind_r": 0.50, "fore_r": 0.75}

    out: dict[str, float] = {}
    for idx, lb in enumerate(morph.limbs):
        if lb.role in explicit:
            out[lb.role] = explicit[lb.role]
        elif lb.role in defaults:
            out[lb.role] = defaults[lb.role]
        else:
            out[lb.role] = (0.5 * (idx % 2) + 0.15 * (idx // 2)) % 1.0
    return out


def compute_froude_kinematics(
    program: Program,
    gait: GaitDecl,
    morph: MorphologyReport,
    duration_s: float,
) -> dict[str, Any]:
    """Compute Froude speed, safe stride length, crouch bias, and integer cycle count."""
    g_u = _default_gravity_units_s2(program)
    if not morph.limbs or gait.pattern in ("serpentine_wave", "swim_undulate"):
        # Limbless / aquatic undulatory propulsion (serpents, eels, fish, wyrms, leviathans!)
        body_len = max(4.0, float(morph.body_span[2]))
        fr = max(1e-4, float(gait.froude))
        v_und = math.sqrt(fr * g_u * max(2.0, 0.35 * body_len))
        s_und = 0.55 * body_len * (fr ** 0.25)
        raw_c = (v_und * max(0.1, duration_s)) / max(1.0, s_und)
        n_cyc = max(1, min(6, int(round(raw_c))))
        period_s = max(0.1, float(duration_s) / float(n_cyc))
        stride_u = v_und * period_s
        return {
            "gravity_u_s2": g_u,
            "mean_hip_height": max(1.0, 0.25 * float(morph.body_span[1])),
            "froude_target_speed": v_und,
            "stride_length": stride_u,
            "cycles_in_clip": n_cyc,
            "cycle_period_s": period_s,
            "achieved_speed_u_s": stride_u / period_s,
            "crouch_y": 0.0,
            "lift_height": 0.0,
            "heave_amp": (0.04 * float(morph.body_span[1])) if gait.pattern == "swim_undulate" else 0.0,
        }

    mean_h = float(np.mean([lb.hip_height for lb in morph.limbs]))
    fr = max(1e-4, float(gait.froude))
    v_froude = math.sqrt(fr * g_u * max(0.5, mean_h))

    # Alexander & Jayes (1983): relative stride length S / h ≈ 2.3 * Fr^0.30
    s_alexander = 2.3 * mean_h * (fr ** 0.30)

    # Compute vertical posture bias:
    # - Straight erect limbs (>= 90% reach) must crouch slightly so feet reach stride extremes cleanly.
    # - Deeply crouched/broad-chested limbs (<= 84% reach) have generous extension reserve and benefit from
    #   subtle athletic lift (+0.65 * heave_amp) so the ribcage stays cleanly clear above elbows/knees!
    rest_ratios = [
        math.hypot(
            lb.contact_world[0] - lb.hip_world[0],
            math.hypot(lb.contact_world[1] - lb.hip_world[1], lb.contact_world[2] - lb.hip_world[2]),
        )
        / max(1e-6, float(lb.reach[1]))
        for lb in morph.limbs
    ]
    needs_crouch = any(lb.posture == "erect" and r >= 0.90 for lb, r in zip(morph.limbs, rest_ratios))
    heave_amp = float(gait.heave_ratio) * mean_h * (0.8 + 0.4 * float(gait.weight))
    if needs_crouch:
        crouch_y = -0.12 * mean_h * (0.75 + 0.4 * float(gait.weight))
    elif rest_ratios and max(rest_ratios) <= 0.89 and all(lb.posture != "erect" for lb in morph.limbs):
        # Cap athletic upward lift strictly inside every limb's 0.90*reach spherical headroom
        safe_lifts: list[float] = []
        for lb in morph.limbs:
            dx_r = float(lb.contact_world[0] - lb.hip_world[0])
            dy_r = abs(float(lb.contact_world[1] - lb.hip_world[1]))
            dz_r = float(lb.contact_world[2] - lb.hip_world[2])
            r_cap = 0.91 * float(lb.reach[1])
            y_avail_sq = r_cap * r_cap - dx_r * dx_r - dz_r * dz_r
            if y_avail_sq > dy_r * dy_r:
                safe_lifts.append(max(0.0, math.sqrt(y_avail_sq) - dy_r - heave_amp))
            else:
                safe_lifts.append(0.0)
        crouch_y = min(1.05 * heave_amp, min(safe_lifts) if safe_lifts else 0.0)
    else:
        crouch_y = -0.025 * mean_h

    # Bound stride by every limb's 3D IK reach envelope during stance + heave excursion
    duty = float(gait.duty)
    safe_strides: list[float] = []
    for lb in morph.limbs:
        dx = float(lb.contact_world[0] - lb.hip_world[0])
        dy_worst = abs(float(lb.contact_world[1] - (lb.hip_world[1] + crouch_y + heave_amp)))
        r_max = 0.95 * float(lb.reach[1])
        horiz_budget_sq = max(0.0, r_max * r_max - dy_worst * dy_worst - dx * dx)
        max_half_stance_z = 0.84 * math.sqrt(horiz_budget_sq)
        # Account for rest Z offset of foot relative to hip (never exceed actual horizontal IK reach)
        dz_rest = abs(float(lb.contact_world[2] - lb.hip_world[2]))
        avail_half_z = min(max_half_stance_z, max(0.18 * lb.hip_height, max_half_stance_z - 0.30 * dz_rest)) if max_half_stance_z > 0.25 else max(0.12 * lb.hip_height, max_half_stance_z)
        safe_s = (2.0 * avail_half_z) / max(0.25, duty)
        safe_strides.append(safe_s)

    # Bound stride by ipsilateral inter-limb Z spacing: in body frame each foot stays within ±0.5*duty*stride
    for i_a, lb_a in enumerate(morph.limbs):
        for lb_b in morph.limbs[i_a + 1 :]:
            if lb_a.side == lb_b.side and abs(float(lb_a.contact_world[0] - lb_b.contact_world[0])) < 2.5:
                dz_pair = abs(float(lb_a.contact_world[2] - lb_b.contact_world[2]))
                if dz_pair > 0.5:
                    safe_strides.append(max(0.45 * dz_pair, (dz_pair - 1.45) / max(0.25, duty)))

    max_safe_stride = min(safe_strides) if safe_strides else s_alexander
    s_target = min(s_alexander, max_safe_stride)

    # Integer cycle count inside `duration_s` preserves exact loop periodicity AND scales step frequency with h^-1/2
    raw_cycles = (v_froude * max(0.1, duration_s)) / max(0.25, s_target)
    cycles_in_clip = max(1, min(6, int(round(raw_cycles))))
    cycle_period_s = max(0.1, float(duration_s) / float(cycles_in_clip))
    stride_length = min(max_safe_stride, v_froude * cycle_period_s)
    achieved_speed = stride_length / cycle_period_s

    lift_h = min(
        0.42 * min(lb.hip_height for lb in morph.limbs),
        float(gait.lift_ratio) * mean_h * (0.85 + 0.35 * float(gait.snap)),
    )

    return {
        "gravity_u_s2": g_u,
        "mean_hip_height": mean_h,
        "froude_target_speed": v_froude,
        "stride_length": stride_length,
        "cycles_in_clip": cycles_in_clip,
        "cycle_period_s": cycle_period_s,
        "achieved_speed_u_s": achieved_speed,
        "crouch_y": crouch_y,
        "lift_height": lift_h,
        "heave_amp": heave_amp,
    }


def evaluate_gait_clip(
    program: Program,
    clip: ClipDecl,
    *,
    num_frames: int = 12,
    active_state: str | None = None,
    sample_phases: Sequence[float] | None = None,
    _phase_cache: dict[Any, Any] | None = None,
) -> list[BehaviorFrameSample]:
    """Evaluate a clip containing a `gait` declaration into exact posed SE(3) `BehaviorFrameSample`s."""
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

    gait = clip.gaits[0]
    auto_rig = bool(getattr(clip, "auto_rig", False))
    in_place_mode = bool(getattr(gait, "in_place", False) or getattr(clip, "in_place", False))
    use_auto_bio = bool(getattr(gait, "auto_bio_limits", False) or getattr(clip, "auto_bio_limits", False) or auto_rig)
    stab_strength = max(
        float(getattr(gait, "head_stabilize", 0.0)),
        float(getattr(clip, "head_stabilize", 0.0)),
        0.70 if auto_rig else 0.0,
    )
    foot_roll_g = float(getattr(gait, "foot_roll", 0.0)) or (0.65 if auto_rig else 0.0)
    digit_g = float(getattr(gait, "digit_articulation", 0.0)) or (0.80 if auto_rig else 0.0)
    com_g = max(float(getattr(gait, "com_balance", 0.0)), float(getattr(clip, "com_balance", 0.0))) or (0.55 if auto_rig else 0.0)
    antag_g = max(float(getattr(gait, "antagonist_bulge", 0.0)), float(getattr(clip, "antagonist_bulge", 0.0))) or (0.65 if auto_rig else 0.0)
    wing_fold_g = getattr(gait, "wing_fold", None)
    neck_aim_tgt = getattr(clip, "neck_aim", None)
    spine_coil_deg = float(getattr(clip, "spine_coil", 0.0))
    digit_grasp_g = float(getattr(clip, "digit_grasp", 0.0))
    need_rig_asm = bool(
        auto_rig
        or foot_roll_g > 1e-4
        or digit_g > 1e-4
        or com_g > 1e-4
        or antag_g > 1e-4
        or wing_fold_g is not None
        or neck_aim_tgt is not None
        or abs(spine_coil_deg) > 1e-4
        or digit_grasp_g > 1e-4
    )

    if _phase_cache is not None and "__setup__" in _phase_cache:
        rest_scene, morph, limits_map, limb_phases, kin, rig_asm = _phase_cache["__setup__"]
    else:
        rest_scene = compile_geometry(program, validate=False)
        morph = analyze_morphology(program, rest_scene)
        explicit_limits = _collect_expanded_joint_limits(program)
        limits_map = (
            infer_universal_bio_constraints(program, rest_scene, morph, explicit_limits=explicit_limits)
            if use_auto_bio
            else explicit_limits
        )
        limb_phases = resolve_limb_phases(gait, morph)
        kin = compute_froude_kinematics(program, gait, morph, float(clip.duration_s))
        rig_asm = None
        if need_rig_asm:
            from synccraft.kinematics.whole_body_solver import auto_assemble_biomechanical_rig

            rig_asm = auto_assemble_biomechanical_rig(program, rest_scene, morph, explicit_limits=explicit_limits)
            if use_auto_bio:
                limits_map = rig_asm.bio_constraints
        if _phase_cache is not None:
            _phase_cache["__setup__"] = (rest_scene, morph, limits_map, limb_phases, kin, rig_asm)

    is_undulatory = (gait.pattern in ("serpentine_wave", "swim_undulate")) or (len(morph.limbs) == 0)

    state_pose_bias: dict[str, Vec3] = {}
    state_muscle_tone: dict[str, float] = {}
    if active_state is not None and program.behavior is not None:
        st_map = {s.name: s for s in program.behavior.states}
        if active_state in st_map:
            state_pose_bias = dict(st_map[active_state].pose_biases)
            state_muscle_tone = dict(st_map[active_state].muscle_tones)

    if sample_phases is not None:
        phases = np.asarray(list(sample_phases), dtype=np.float64)
    else:
        n_f = max(2, int(num_frames))
        phases = np.linspace(0.0, 1.0, n_f, endpoint=not clip.loop, dtype=np.float64)

    fwd_z = morph.forward_sign_z
    stride_len = float(kin["stride_length"])
    n_cycles = int(kin["cycles_in_clip"])
    crouch_y = float(kin["crouch_y"])
    heave_amp = float(kin["heave_amp"])
    lift_h = float(kin["lift_height"])
    duty = float(gait.duty)
    spd_u_s = float(kin["achieved_speed_u_s"])
    # In-place clips STILL emit exact matched Minecraft entity velocity in root_motion_velocity!
    rm_vel: Vec3 = (0.0, 0.0, fwd_z * spd_u_s) if (gait.export_root_motion or in_place_mode) else (0.0, 0.0, 0.0)

    ik_decl_by_name: dict[str, IKChainDecl] = (
        {c.name: c for c in program.kinematics.ik_chains} if program.kinematics is not None else {}
    )
    axis_idx = {"x": 0, "y": 1, "z": 2}
    root_bone_name = rest_scene.bones[0].name

    # Secondary springs (exclude grounded limb IK chains and locked wing anchor spars)
    all_springs: list[SpringDecl] = []
    seen_sp_b: set[str] = set()
    limb_bone_set = {bn for lb in morph.limbs for bn in lb.chain}
    locked_wing_set = {bn for wg in morph.wings for bn in wg.locked_spar_bones}
    for sp_src in (
        *(program.kinematics.springs if program.kinematics is not None else ()),
        *(program.behavior.springs if program.behavior is not None else ()),
    ):
        if sp_src.bone not in seen_sp_b and sp_src.bone not in limb_bone_set and sp_src.bone not in locked_wing_set:
            seen_sp_b.add(sp_src.bone)
            all_springs.append(sp_src)

    # Pre-batch secondary motion `layers` once across all phases
    all_times_s = [float(p) * float(clip.duration_s) for p in phases]
    batched_layers = [
        _batch_sample_clip_layer_deltas(program, lyr.source_clip, all_times_s, float(lyr.weight), parent_clip=clip)
        for lyr in getattr(clip, "layers", ())
    ]

    # When springs are active, collect pre-spring axial/muscle poses across all frames first
    use_spring_pass = bool(all_springs) and len(phases) >= 2
    axial_poses_list: list[dict[str, Vec3]] = []
    exported_root_list: list[Vec3] = []
    virtual_root_list: list[Vec3] = []
    acts_list: list[dict[str, float]] = []
    pos_deltas_list: list[dict[str, Vec3]] = []
    scale_factors_list: list[dict[str, Vec3]] = []
    times_list: list[float] = []
    phases_list: list[float] = []
    cyc_prog_list: list[float] = []

    for f_idx, phi_raw in enumerate(phases):
        phi_clip = float(phi_raw)
        phi_key = round(phi_clip, 8)
        if not use_spring_pass and _phase_cache is not None and phi_key in _phase_cache:
            # Fast cache hit when no cross-frame spring integration is required
            cached_s = _phase_cache[phi_key]
            axial_poses_list.append(cached_s)  # sentinel handled below
            exported_root_list.append(cached_s.root_translation_delta)
            virtual_root_list.append(cached_s.root_translation_delta)
            acts_list.append(cached_s.muscle_activations)
            pos_deltas_list.append(cached_s.pose_position_deltas)
            scale_factors_list.append(cached_s.pose_scale_factors)
            times_list.append(cached_s.time_s)
            phases_list.append(phi_clip)
            cyc_prog_list.append(phi_clip * float(n_cycles))
            continue

        t_s = phi_clip * float(clip.duration_s)
        cyc_prog = phi_clip * float(n_cycles)
        cyc_phi = cyc_prog % 1.0
        times_list.append(t_s)
        phases_list.append(phi_clip)
        cyc_prog_list.append(cyc_prog)

        # 1. Root motion: steady virtual advance + crouch + 2x-per-stride vertical CoM heave
        heave_y = crouch_y - heave_amp * math.cos(4.0 * math.pi * cyc_phi)
        adv_z = fwd_z * cyc_prog * stride_len
        virtual_root_delta: Vec3 = (0.0, float(heave_y), float(adv_z))
        # Minecraft / Blockbench in-place contract: strip macro adv_z from exported bone position!
        exported_root_delta: Vec3 = (0.0, float(heave_y), 0.0) if in_place_mode else virtual_root_delta
        exported_root_list.append(exported_root_delta)
        virtual_root_list.append(virtual_root_delta)

        # 2. Axial posture (spine wave, torso pitch, neck counter-bob, multi-tail/tentacle/free-limb waves)
        pose_map: dict[str, np.ndarray] = {
            b: np.asarray(v, dtype=np.float64).copy() for b, v in state_pose_bias.items()
        }

        for tr in clip.tracks:
            ang = _evaluate_track_angle(
                tr.law, t_s, clip.duration_s, tr.amp_deg, tr.freq_hz, tr.phase_deg, tr.bias_deg
            )
            vec = pose_map.setdefault(tr.bone, np.zeros(3, dtype=np.float64))
            vec[axis_idx[tr.axis]] += ang

        frame_pos_deltas: dict[str, Vec3] = {}
        frame_scale_factors: dict[str, Vec3] = {}
        for kf in clip.keyframes:
            kv = evaluate_keyframe_track(kf, t_s)
            if kf.channel == "rotation":
                vec = pose_map.setdefault(kf.bone, np.zeros(3, dtype=np.float64))
                vec += np.asarray(kv, dtype=np.float64)
            elif kf.channel == "position":
                prev_p = frame_pos_deltas.get(kf.bone, (0.0, 0.0, 0.0))
                frame_pos_deltas[kf.bone] = (prev_p[0] + kv[0], prev_p[1] + kv[1], prev_p[2] + kv[2])
            elif kf.channel == "scale":
                prev_s = frame_scale_factors.get(kf.bone, (1.0, 1.0, 1.0))
                frame_scale_factors[kf.bone] = (prev_s[0] * kv[0], prev_s[1] * kv[1], prev_s[2] * kv[2])

        # Apply additive secondary motion layers BEFORE world limb IK so feet stay planted (< 0.02u slip)
        for lyr_batch in batched_layers:
            l_eul, l_pos = lyr_batch[f_idx]
            for lb_n, lv in l_eul.items():
                vec = pose_map.setdefault(lb_n, np.zeros(3, dtype=np.float64))
                vec += np.asarray(lv, dtype=np.float64)
            for lb_n, lpv in l_pos.items():
                prev_p = frame_pos_deltas.get(lb_n, (0.0, 0.0, 0.0))
                frame_pos_deltas[lb_n] = (prev_p[0] + lpv[0], prev_p[1] + lpv[1], prev_p[2] + lpv[2])

        two_pi = 2.0 * math.pi * cyc_phi
        non_root_spine = [b for b in morph.spine if b != root_bone_name]
        if is_undulatory:
            # Limbless serpentine / aquatic traveling sinusoidal wave down entire spine + tail!
            axial_wave_chain = [*non_root_spine, *[b for b in morph.tail if b not in non_root_spine and b != root_bone_name]]
            base_amp = max(8.0, float(gait.spine_wave_deg) * 3.2) / max(1.0, math.sqrt(float(max(1, len(axial_wave_chain)))))
            for w_i, wb_n in enumerate(axial_wave_chain):
                wv = pose_map.setdefault(wb_n, np.zeros(3, dtype=np.float64))
                env = 0.65 + 0.20 * w_i
                lag = 0.72 * w_i
                wv[1] += base_amp * env * math.sin(two_pi - lag)
                if gait.pattern == "swim_undulate":
                    wv[0] += 0.20 * float(gait.pitch_deg) * math.sin(two_pi - lag)
        elif non_root_spine:
            per_sp_wave = float(gait.spine_wave_deg) / float(len(non_root_spine))
            per_sp_pitch = float(gait.pitch_deg) / float(len(non_root_spine))
            for s_i, sb in enumerate(non_root_spine):
                sv = pose_map.setdefault(sb, np.zeros(3, dtype=np.float64))
                sv[0] += per_sp_pitch * math.sin(2.0 * two_pi)
                sv[1] += per_sp_wave * math.sin(two_pi - 0.4 * s_i)
        elif abs(float(gait.pitch_deg)) > 1e-6:
            rv = pose_map.setdefault(root_bone_name, np.zeros(3, dtype=np.float64))
            rv[0] += 0.5 * float(gait.pitch_deg) * math.sin(2.0 * two_pi)

        # Supple phase-delayed serpentine neck wave (active on ALL multi-bone necks so long dragon necks never freeze!)
        neck_chains_wave = morph.neck_chains if morph.neck_chains else ((morph.neck,) if morph.neck else ())
        for nc_i, n_chain_w in enumerate(neck_chains_wave):
            valid_nk = [nb for nb in n_chain_w if nb != root_bone_name]
            if not valid_nk:
                continue
            n_nk = len(valid_nk)
            nk_pitch_amp = max(2.2, 1.1 * abs(float(gait.pitch_deg)) + 1.8) / math.sqrt(float(n_nk))
            nk_yaw_amp = max(2.0, 0.95 * abs(float(gait.spine_wave_deg))) / math.sqrt(float(n_nk))
            for n_i, nb in enumerate(valid_nk):
                nv = pose_map.setdefault(nb, np.zeros(3, dtype=np.float64))
                lag_n = 0.58 * float(n_i) + 0.4 * float(nc_i)
                if n_nk >= 3 and n_i == n_nk - 1:
                    # Distal head/atlas counter-tracks subtly to maintain predatory focus
                    nv[0] += -0.45 * nk_pitch_amp * math.sin(2.0 * two_pi - lag_n + 0.3)
                    nv[1] += -0.40 * nk_yaw_amp * math.sin(two_pi - lag_n + 0.25)
                else:
                    env_n = 0.80 + 0.35 * math.sin(math.pi * (float(n_i + 0.5) / float(max(1, n_nk))))
                    nv[0] += -nk_pitch_amp * env_n * math.sin(2.0 * two_pi - lag_n)
                    nv[1] += nk_yaw_amp * env_n * math.sin(two_pi - lag_n)
                    nv[2] += 0.35 * nk_yaw_amp * env_n * math.cos(two_pi - lag_n)

        # Traveling caudal wave down ALL discovered tail chains (supports single- and multi-tailed creatures!)
        tail_chains_all = morph.tail_chains if morph.tail_chains else ((morph.tail,) if morph.tail else ())
        if not is_undulatory:
            for tc_i, t_chain in enumerate(tail_chains_all):
                if not t_chain:
                    continue
                base_tail_amp = max(2.5, 1.4 * float(gait.spine_wave_deg)) / math.sqrt(float(len(t_chain)))
                tc_off = 0.55 * tc_i
                for t_i, tb in enumerate(t_chain):
                    if tb == root_bone_name:
                        continue
                    tv = pose_map.setdefault(tb, np.zeros(3, dtype=np.float64))
                    lag_rad = 2.0 * math.pi * float(gait.tail_lag) * (t_i + 1)
                    tv[1] += base_tail_amp * (1.0 + 0.12 * t_i) * math.sin(two_pi - lag_rad + tc_off)

        # Tentacle secondary traveling waves (octopus / kraken / alien tendrils)
        for tent_i, tent_chain in enumerate(morph.tentacles):
            for s_i, sb in enumerate(tent_chain):
                if sb == root_bone_name:
                    continue
                sv = pose_map.setdefault(sb, np.zeros(3, dtype=np.float64))
                sv[0] += 7.5 * math.sin(two_pi - 0.55 * s_i + 0.7 * tent_i)
                sv[1] += 6.5 * math.cos(two_pi - 0.55 * s_i + 0.7 * tent_i)

        # Automatic anti-phase counter-swing for free non-grounded upper limbs (suppress on <0.35x vestigial paws!)
        mean_gnd_len = (
            sum(float(lb.length) for lb in morph.limbs) / max(1, len(morph.limbs))
            if morph.limbs
            else 1.0
        )
        for flb in morph.free_limbs:
            is_vestigial = (float(flb.length) / max(1e-3, mean_gnd_len)) < 0.35
            opp_role = "hind_r" if flb.side == "l" else "hind_l"
            arm_phase = limb_phases.get(opp_role, 0.5 if flb.side == "l" else 0.0)
            arm_th = 2.0 * math.pi * ((cyc_phi + arm_phase) % 1.0)
            swing_amp = 4.0 if is_vestigial else min(32.0, 12.0 + 28.0 * math.sqrt(max(0.05, float(gait.froude))))
            for seg_i, bname in enumerate(flb.chain):
                av = pose_map.setdefault(bname, np.zeros(3, dtype=np.float64))
                if seg_i == 0:
                    av[0] += -swing_amp * math.cos(arm_th) * fwd_z
                elif seg_i == 1 and not is_vestigial:
                    # Natural forward elbow flexion on forward arm swing
                    av[0] += (-12.0 - 14.0 * max(0.0, -math.cos(arm_th))) * fwd_z

        # Subtle sympathetic wing-cloak settling bounce during grounded walk/run (keeps locked spars at [0,0,0]!)
        if morph.wings and not is_undulatory and abs(float(gait.heave_ratio)) > 1e-4:
            for wg in morph.wings:
                if wg.chain:
                    w0 = wg.chain[0]
                    wv0 = pose_map.setdefault(w0, np.zeros(3, dtype=np.float64))
                    z_sgn = 1.0 if wg.side == "l" else -1.0
                    wv0[2] += z_sgn * 3.0 * math.sin(2.0 * two_pi - 0.35)
                for lb_lk in wg.locked_spar_bones:
                    pose_map[lb_lk] = np.zeros(3, dtype=np.float64)

        # Neural muscle excitation curves + Zajac/Hill muscle & tendon coupling + Sim(3) bulge
        acts = dict(state_muscle_tone)
        for md in clip.muscle_drives:
            u_m = (t_s * md.freq_hz + md.phase_deg / 360.0) % 1.0
            val = md.baseline + (md.peak - md.baseline) * (0.5 - 0.5 * math.cos(2.0 * math.pi * u_m))
            acts[md.muscle] = float(np.clip(max(acts.get(md.muscle, 0.0), val), 0.0, 1.0))
        acts_list.append(acts)
        _apply_muscle_scale_bulge(program, rest_scene, acts, frame_scale_factors)

        for b_k in list(pose_map):
            if use_auto_bio and b_k in limits_map:
                jlim, asign = limits_map[b_k]
                pose_map[b_k] = soft_clamp_euler_to_limits(pose_map[b_k], jlim, angle_sign=asign)
            else:
                pose_map[b_k] = _clamp_pose_vec(b_k, pose_map[b_k], limits_map)

        cur_poses: dict[str, Vec3] = {
            b: (float(v[0]), float(v[1]), float(v[2])) for b, v in pose_map.items()
        }

        if program.muscles and acts:
            ms_res = evaluate_muscles_at_activations(
                program,
                acts,
                base_pose_deltas=cur_poses,
                root_translation_delta=exported_root_delta,
                pose_position_deltas=frame_pos_deltas or None,
                pose_scale_factors=frame_scale_factors or None,
                validate=False,
                _rest_scene=rest_scene,
                _limits_map=limits_map,
                _skip_scene=True,
            )
            cur_poses = dict(ms_res.pose_euler_deltas)
            for wg in morph.wings:
                for lb_lk in wg.locked_spar_bones:
                    cur_poses[lb_lk] = (0.0, 0.0, 0.0)

        # Vestibulo-ocular gaze lock: on long serpentine necks (>= 4 bones), stabilize ONLY the distal
        # atlas/head (at capped gentle gain <= 0.38) so proximal neck vertebrae stay supple and vibrant!
        if stab_strength > 1e-4:
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
                        root_translation_delta=exported_root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                    )
            else:
                neck_chains_all = morph.neck_chains if morph.neck_chains else ((morph.neck,) if morph.neck else ())
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
                        root_translation_delta=exported_root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                    )

        # Whole-body wing fold, serpentine S-neck Spline-IK & COM/tail counterweight BEFORE limb IK!
        if rig_asm is not None:
            from synccraft.kinematics.whole_body_solver import (
                solve_com_balance_and_counterweight,
                solve_spine_neck_spline_ik,
                solve_wing_full_anatomy,
            )

            if wing_fold_g is not None:
                cur_poses, frame_scale_factors = solve_wing_full_anatomy(
                    program,
                    rest_scene,
                    rig_asm,
                    cur_poses,
                    frame_scale_factors,
                    wing_fold=float(wing_fold_g),
                    ground_clearance=True,
                    root_translation_delta=exported_root_delta,
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
                        root_translation_delta=exported_root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                        forward_sign_z=fwd_z,
                    )
            if com_g > 1e-4:
                cur_poses, balanced_exp_root = solve_com_balance_and_counterweight(
                    program,
                    rest_scene,
                    rig_asm,
                    cur_poses,
                    exported_root_delta,
                    strength=com_g,
                    in_place=in_place_mode,
                    grounded=not is_undulatory,
                    limits_map=limits_map,
                    pose_position_deltas=frame_pos_deltas or None,
                    pose_scale_factors=frame_scale_factors or None,
                )
                dx_b = balanced_exp_root[0] - exported_root_delta[0]
                exported_root_delta = balanced_exp_root
                virtual_root_delta = (virtual_root_delta[0] + dx_b, virtual_root_delta[1], virtual_root_delta[2])
                exported_root_list[-1] = exported_root_delta
                virtual_root_list[-1] = virtual_root_delta

        axial_poses_list.append(cur_poses)
        pos_deltas_list.append(frame_pos_deltas)
        scale_factors_list.append(frame_scale_factors)

    # 3. Implicit-Euler Secondary Spring Follow-Through BEFORE world limb IK so grounded feet never slip!
    if use_spring_pass:
        axial_poses_list = _integrate_spring_follow_through(
            program,
            rest_scene,
            all_springs,
            times_list,
            clip.loop,
            float(clip.duration_s),
            axial_poses_list,
            exported_root_list,
            limits_map,
        )

    samples: list[BehaviorFrameSample] = []
    stride_vec = np.array([0.0, 0.0, fwd_z * stride_len], dtype=np.float64)

    # Precompute SE(3)/SO(3) wrist-ankle split metadata + foot/digit rest corner offsets for exact ground barrier!
    # Applies uniformly to BOTH auto-discovered limbs AND explicit @kinematics ik_chains (whether the ik_chain
    # terminates at the foot/paw bone OR at the distal forearm/ankle bone directly above la_m.foot_hand_bone)!
    la_by_role = {la.role: la for la in rig_asm.limb_assemblies} if rig_asm is not None else {}
    foot_split_info: dict[str, dict[str, Any]] = {}
    for lb_m in (morph.limbs if not is_undulatory else ()):
        la_m = la_by_role.get(lb_m.role)
        foot_cand: str | None = None
        prox_bones: tuple[str, ...] = lb_m.chain
        if len(lb_m.chain) >= 3:
            last_b = lb_m.chain[-1]
            if (
                last_b in rest_scene.bone_by_name
                and rest_scene.bone_by_name[last_b].parent_name == lb_m.chain[-2]
                and (
                    (la_m is not None and la_m.foot_hand_bone == last_b)
                    or any(k in last_b.lower() for k in ("foot", "paw", "hand", "talon", "hoof", "pad", "tars"))
                )
            ):
                foot_cand = last_b
                prox_bones = lb_m.chain[:-1]
        if (
            foot_cand is None
            and la_m is not None
            and la_m.foot_hand_bone is not None
            and la_m.foot_hand_bone in rest_scene.bone_by_name
            and rest_scene.bone_by_name[la_m.foot_hand_bone].parent_name == lb_m.chain[-1]
        ):
            foot_cand = la_m.foot_hand_bone
            prox_bones = lb_m.chain
        if foot_cand is not None and len(prox_bones) >= 2:
            wb_f = rest_scene.bone_by_name[foot_cand]
            r_fw = wb_f.rotation
            r_floc = euler_xyz_matrix(wb_f.local_euler)
            sole_w = np.array([lb_m.contact_world[0], morph.ground_y, lb_m.contact_world[2]], dtype=np.float64) - wb_f.pivot_world
            sole_floc = r_fw.T @ sole_w
            f_bones = {foot_cand, *lb_m.digits, *(la_m.digit_chains[0] if (la_m and la_m.digit_chains) else ())}
            if la_m is not None:
                for dc in la_m.digit_chains:
                    f_bones.update(dc)
                for tc in la_m.thumb_chains:
                    f_bones.update(tc)
            f_bxs = [bx for bx in rest_scene.boxes if bx.bone_name in f_bones and getattr(bx, "plane_normal", None) is None]
            if f_bxs:
                all_fc = np.vstack([bx.corners(include_inflate=False) for bx in f_bxs]) - wb_f.pivot_world
                fc_local = all_fc @ r_fw  # N x 3 in foot_b local frame
            else:
                fc_local = sole_floc.reshape(1, 3)
            # Measure true proximal leg chain length (hip -> ... -> wrist/ankle) for dynamic reach-crouch safety!
            prox_len = 0.0
            for idx_p in range(1, len(prox_bones)):
                prox_len += float(
                    np.linalg.norm(
                        rest_scene.bone_by_name[prox_bones[idx_p]].pivot_world
                        - rest_scene.bone_by_name[prox_bones[idx_p - 1]].pivot_world
                    )
                )
            prox_len += float(np.linalg.norm(wb_f.local_translation))
            foot_split_info[lb_m.role] = {
                "ik_bones": prox_bones,
                "foot_bone": foot_cand,
                "wrist_eff_local": (
                    float(wb_f.local_translation[0]),
                    float(wb_f.local_translation[1]),
                    float(wb_f.local_translation[2]),
                ),
                "r_foot_rest_w": r_fw,
                "r_foot_rest_loc": r_floc,
                "sole_rest_w": sole_w,
                "sole_local_foot": sole_floc,
                "corners_local_foot": fc_local,
                "prox_max_reach": 0.96 * max(0.5, prox_len),
            }

    for f_idx, phi_clip in enumerate(phases_list):
        entry = axial_poses_list[f_idx]
        if isinstance(entry, BehaviorFrameSample):
            samples.append(dataclasses.replace(entry, frame_index=f_idx))
            continue

        cur_poses = dict(entry)
        t_s = times_list[f_idx]
        cyc_prog = cyc_prog_list[f_idx]
        exported_root_delta = exported_root_list[f_idx]
        virtual_root_delta = virtual_root_list[f_idx]
        frame_pos_deltas = pos_deltas_list[f_idx]
        frame_scale_factors = scale_factors_list[f_idx]
        acts = acts_list[f_idx]

        # 4. Closed-loop world-planted SE(3) limb IK + SO(3) world-sagittal reverse-foot rocker!
        active_limbs = () if is_undulatory else morph.limbs
        posture_scene = (
            compile_posed_geometry(
                program,
                cur_poses,
                pose_position_deltas=frame_pos_deltas or None,
                pose_scale_factors=frame_scale_factors or None,
                validate=False,
            )
            if active_limbs
            else None
        )
        eff_positions: dict[str, Vec3] = {}
        eff_stance: dict[str, bool] = {}
        eff_step_ids: dict[str, int] = {}
        limb_u_map: dict[str, float] = {}
        desired_foot_world_rot: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}

        # Pre-plan targets + dynamic SE(3) torso suspension dip when bold strides approach forelimb/hindlimb max reach!
        planned_limb_solves: list[dict[str, Any]] = []
        extra_crouch_y = 0.0
        for lb in active_limbs:
            phase_off = limb_phases.get(lb.role, 0.0)
            eff_rest_w = np.array(
                [lb.contact_world[0], morph.ground_y, lb.contact_world[2]],
                dtype=np.float64,
            )
            if stride_len <= 1e-6:
                tgt_w = eff_rest_w.copy()
                in_stance = True
                step_id = 0
                u_val = 0.25
                s_swing = 0.0
            else:
                raw_p = cyc_prog + phase_off
                wrap_k = math.floor(raw_p)
                u_val = raw_p - wrap_k
                stance_anchor = eff_rest_w + (wrap_k - phase_off + 0.5 * duty) * stride_vec
                if u_val < duty:
                    tgt_w = stance_anchor.copy()
                    in_stance = True
                    step_id = int(wrap_k)
                    s_swing = 0.0
                else:
                    s_swing = (u_val - duty) / max(1e-6, 1.0 - duty)
                    snap_g = 0.35 * float(gait.snap)
                    s_warped = (1.0 - snap_g) * s_swing + snap_g * (0.5 - 0.5 * math.cos(math.pi * s_swing))
                    sigma = 0.5 - 0.5 * math.cos(math.pi * s_warped)
                    lift_y = lift_h * (math.sin(math.pi * s_swing) ** 2)
                    tgt_w = stance_anchor + sigma * stride_vec + np.array([0.0, lift_y, 0.0], dtype=np.float64)
                    in_stance = False
                    step_id = -1
            limb_u_map[lb.role] = u_val

            finfo = foot_split_info.get(lb.role)
            if finfo is not None:
                r_fw_rest = finfo["r_foot_rest_w"]
                if in_stance:
                    # Exact flat SO(3) world sole lock across stance: ZERO sideways roll/yaw wobble & 0.0000u slip!
                    r_foot_w = r_fw_rest
                    wrist_tgt_w = tgt_w - finfo["sole_rest_w"]
                    sole_offset_now = finfo["sole_rest_w"]
                else:
                    # Pure world-sagittal SO(3) foot-roll arc: toe-off plantarflexion -> mid-swing arch -> heel-first entry
                    rock_deg = (
                        float(foot_roll_g)
                        * fwd_z
                        * (15.0 * math.sin(math.pi * s_swing) * math.cos(0.55 * math.pi * s_swing) - 5.0 * (math.sin(math.pi * s_swing) ** 2))
                    )
                    r_x_world = euler_xyz_matrix((rock_deg, 0.0, 0.0))
                    r_foot_w = r_x_world @ r_fw_rest
                    sole_offset_now = r_foot_w @ finfo["sole_local_foot"]
                    wrist_tgt_w = tgt_w - sole_offset_now
                    # Hard SE(3) ground-plane barrier: no heel or claw tip may dip below morph.ground_y during swing!
                    rot_corners_y = (finfo["corners_local_foot"] @ r_foot_w.T)[:, 1]
                    min_dy = float(np.min(rot_corners_y))
                    min_wrist_y = float(morph.ground_y) - min_dy + 0.25 * lift_h * math.sin(math.pi * s_swing)
                    if wrist_tgt_w[1] < min_wrist_y:
                        wrist_tgt_w[1] = min_wrist_y

                if posture_scene is not None:
                    hip_w_cur = posture_scene.bone_by_name[finfo["ik_bones"][0]].pivot_world + np.asarray(virtual_root_delta, dtype=np.float64)
                    dxz = math.hypot(float(wrist_tgt_w[0] - hip_w_cur[0]), float(wrist_tgt_w[2] - hip_w_cur[2]))
                    dy_cur = float(hip_w_cur[1] - wrist_tgt_w[1])
                    max_r = float(finfo["prox_max_reach"])
                    dist_cur = math.hypot(dxz, dy_cur)
                    if dist_cur > max_r and dy_cur > 1.0:
                        safe_dy = math.sqrt(max(0.25, max_r * max_r - min(dxz, 0.96 * max_r) ** 2))
                        need_drop = max(0.0, dy_cur - safe_dy + 0.15)
                        extra_crouch_y = max(extra_crouch_y, min(need_drop, 0.28 * lb.hip_height))

                planned_limb_solves.append(
                    {
                        "lb": lb,
                        "in_stance": in_stance,
                        "step_id": step_id,
                        "ik_bones": finfo["ik_bones"],
                        "eff_local": finfo["wrist_eff_local"],
                        "tgt_w": wrist_tgt_w,
                        "sole_offset_now": sole_offset_now,
                        "foot_bone": finfo["foot_bone"],
                        "r_foot_w": r_foot_w,
                        "r_foot_rest_loc": finfo["r_foot_rest_loc"],
                    }
                )
            else:
                planned_limb_solves.append(
                    {
                        "lb": lb,
                        "in_stance": in_stance,
                        "step_id": step_id,
                        "ik_bones": lb.chain,
                        "eff_local": lb.contact_local,
                        "tgt_w": tgt_w,
                        "sole_offset_now": np.zeros(3, dtype=np.float64),
                        "foot_bone": None,
                        "r_foot_w": None,
                        "r_foot_rest_loc": None,
                    }
                )

        if extra_crouch_y > 1e-4:
            exported_root_delta = (exported_root_delta[0], exported_root_delta[1] - extra_crouch_y, exported_root_delta[2])
            virtual_root_delta = (virtual_root_delta[0], virtual_root_delta[1] - extra_crouch_y, virtual_root_delta[2])
            exported_root_list[f_idx] = exported_root_delta
            virtual_root_list[f_idx] = virtual_root_delta

        for pls in planned_limb_solves:
            lb = pls["lb"]
            decl_ik = ik_decl_by_name.get(lb.ik_chain_name or "")
            hinge_ax = decl_ik.hinge_axis if decl_ik is not None else "x"
            pole_w = decl_ik.pole if decl_ik is not None else None
            plane_mode = (
                decl_ik.plane
                if (decl_ik is not None and decl_ik.plane is not None)
                else ("sprawled" if lb.posture == "sprawled" else "sagittal")
            )
            tgt_w = pls["tgt_w"]
            shifted_tgt = (
                float(tgt_w[0] - virtual_root_delta[0]),
                float(tgt_w[1] - virtual_root_delta[1]),
                float(tgt_w[2] - virtual_root_delta[2]),
            )
            ik_res = solve_n_bone_ik(
                program,
                pls["ik_bones"],
                pls["eff_local"],
                shifted_tgt,
                base_pose_deltas=cur_poses,
                hinge_axis=hinge_ax,
                pole_world=pole_w,
                plane=plane_mode,
                validate=False,
                _limits_cache=limits_map,
                _zero_scene=posture_scene,
                pose_scale_factors=frame_scale_factors or None,
            )
            cur_poses = ik_res.pose_euler_deltas
            sole_off = pls["sole_offset_now"]
            ach_w = (
                float(ik_res.achieved_world[0] + virtual_root_delta[0] + sole_off[0]),
                float(ik_res.achieved_world[1] + virtual_root_delta[1] + sole_off[1]),
                float(ik_res.achieved_world[2] + virtual_root_delta[2] + sole_off[2]),
            )
            eff_positions[lb.role] = ach_w
            eff_stance[lb.role] = pls["in_stance"]
            eff_step_ids[lb.role] = pls["step_id"]
            if pls["foot_bone"] is not None:
                desired_foot_world_rot[pls["foot_bone"]] = (
                    pls["foot_bone"],
                    pls["r_foot_w"],
                    pls["r_foot_rest_loc"],
                )

        # Project exact SO(3) world-sagittal foot orientation onto split foot/paw bones!
        if desired_foot_world_rot:
            sc_legs = compile_posed_geometry(
                program,
                cur_poses,
                root_translation_delta=exported_root_delta,
                pose_position_deltas=frame_pos_deltas or None,
                pose_scale_factors=frame_scale_factors or None,
                validate=False,
            )
            for foot_b, (_fb, r_fw_tgt, r_floc_rest) in desired_foot_world_rot.items():
                parent_b = rest_scene.bone_by_name[foot_b].parent_name
                if parent_b is None or parent_b not in sc_legs.bone_by_name:
                    continue
                r_parent_posed = sc_legs.bone_by_name[parent_b].rotation
                r_delta = r_floc_rest.T @ (r_parent_posed.T @ r_fw_tgt)
                d_eul = _recover_euler_xyz_degrees(r_delta)
                if limits_map is not None and foot_b in limits_map:
                    jlim, asign = limits_map[foot_b]
                    d_eul = soft_clamp_euler_to_limits(d_eul, jlim, angle_sign=asign)
                cur_poses[foot_b] = (float(d_eul[0]), float(d_eul[1]), float(d_eul[2]))

        # 5. Multi-Phalanx Digit Articulation (swing claw curl) & Whole-Body Sim(3) Antagonist Muscle Bulge
        if rig_asm is not None:
            from synccraft.kinematics.whole_body_solver import (
                solve_foot_roll_and_digit_articulation,
                solve_full_body_antagonist_muscle_bulge,
            )

            # Foot-roll rocker is already solved in exact world SO(3) when desired_foot_world_rot is active!
            eff_foot_roll_g = 0.0 if desired_foot_world_rot else foot_roll_g
            if eff_foot_roll_g > 1e-4 or digit_g > 1e-4 or digit_grasp_g > 1e-4:
                cur_poses = solve_foot_roll_and_digit_articulation(
                    rig_asm,
                    cur_poses,
                    limb_cycle_phases=limb_u_map if limb_u_map else None,
                    limb_in_stance=eff_stance if eff_stance else None,
                    duty=duty,
                    foot_roll_strength=eff_foot_roll_g,
                    digit_strength=digit_g,
                    grasp_amount=digit_grasp_g,
                    aerial_tuck=False,
                    limits_map=limits_map,
                )
            if antag_g > 1e-4:
                frame_scale_factors = solve_full_body_antagonist_muscle_bulge(
                    rig_asm,
                    cur_poses,
                    frame_scale_factors,
                    strength=antag_g,
                    preserve_grounded_limb_length=True,
                )

        kin_scene = compile_posed_geometry(
            program,
            cur_poses,
            root_translation_delta=exported_root_delta,
            pose_position_deltas=frame_pos_deltas or None,
            pose_scale_factors=frame_scale_factors or None,
            validate=False,
        )
        frame_inflate: dict[str, float] = {}
        if program.muscles and acts:
            frame_inflate, _ = _compute_muscle_bulge_and_telemetry(
                rest_scene, kin_scene, program.muscles, acts, cur_poses
            )
        final_scene = (
            compile_posed_geometry(
                program,
                cur_poses,
                box_inflate_deltas=frame_inflate,
                root_translation_delta=exported_root_delta,
                pose_position_deltas=frame_pos_deltas or None,
                pose_scale_factors=frame_scale_factors or None,
                validate=False,
            )
            if frame_inflate
            else kin_scene
        )
        sample_obj = BehaviorFrameSample(
            frame_index=f_idx,
            time_s=t_s,
            phase=phi_clip,
            pose_euler_deltas=cur_poses,
            box_inflate_deltas=frame_inflate,
            root_translation_delta=exported_root_delta,
            muscle_activations=acts,
            effector_world_positions=eff_positions,
            effector_in_stance=eff_stance,
            effector_stance_step_id=eff_step_ids,
            scene=final_scene,
            pose_position_deltas=frame_pos_deltas,
            pose_scale_factors=frame_scale_factors,
            root_motion_velocity=rm_vel,
        )
        if not use_spring_pass and _phase_cache is not None:
            _phase_cache[round(phi_clip, 8)] = sample_obj
        samples.append(sample_obj)

    return samples


def audit_gait_clip_contracts(
    program: Program,
    clip: ClipDecl,
    frames: list[BehaviorFrameSample],
    *,
    max_stance_slip: float = 0.02,
    precomputed_collision_events: int | None = None,
) -> dict[str, Any]:
    """Run all 8 Gate C biomechanical gait checks on sampled `BehaviorFrameSample`s."""
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.kinematics.musculoskeletal import _collect_expanded_joint_limits

    gait = clip.gaits[0]
    in_place_mode = bool(getattr(gait, "in_place", False) or getattr(clip, "in_place", False))
    use_auto_bio = bool(getattr(gait, "auto_bio_limits", False) or getattr(clip, "auto_bio_limits", False))
    rest_scene = compile_geometry(program)
    morph = analyze_morphology(program, rest_scene)
    kin = compute_froude_kinematics(program, gait, morph, float(clip.duration_s))
    explicit_limits = _collect_expanded_joint_limits(program)
    limits_map = (
        infer_universal_bio_constraints(program, rest_scene, morph, explicit_limits=explicit_limits)
        if use_auto_bio
        else explicit_limits
    )
    fwd_z = morph.forward_sign_z
    n_cycles = int(kin["cycles_in_clip"])
    stride_len = float(kin["stride_length"])

    def _virtual_adv_vec(fr: BehaviorFrameSample) -> np.ndarray:
        if not in_place_mode:
            return np.zeros(3, dtype=np.float64)
        return np.array([0.0, 0.0, fwd_z * float(fr.phase) * float(n_cycles) * stride_len], dtype=np.float64)

    # 1. Foot slip (`<= 0.02u`)
    measured_slip = 0.0
    for idx in range(1, len(frames)):
        f0, f1 = frames[idx - 1], frames[idx]
        for role, in_st in f1.effector_in_stance.items():
            sid1 = f1.effector_stance_step_id.get(role, -1)
            sid0 = f0.effector_stance_step_id.get(role, -2)
            if in_st and f0.effector_in_stance.get(role, False) and sid1 >= 0 and sid1 == sid0:
                p0 = f0.effector_world_positions[role]
                p1 = f1.effector_world_positions[role]
                measured_slip = max(measured_slip, math.hypot(p1[0] - p0[0], p1[2] - p0[2]))
    foot_slip_ok = measured_slip <= max_stance_slip

    # 2. Reach (hip-to-effector distance within limb reach envelope)
    max_reach_ratio = 0.0
    for fr in frames:
        v_adv = _virtual_adv_vec(fr)
        for lb in morph.limbs:
            if lb.role not in fr.effector_world_positions:
                continue
            hip_p = fr.scene.bone_by_name[lb.chain[0]].pivot_world + v_adv
            eff_p = np.asarray(fr.effector_world_positions[lb.role], dtype=np.float64)
            dist = float(np.linalg.norm(eff_p - hip_p))
            max_reach_ratio = max(max_reach_ratio, dist / max(1e-6, lb.length))
    reach_ok = max_reach_ratio <= 1.005

    # 3. Joint limits
    max_limit_excess_deg = 0.0
    for fr in frames:
        for bname, eul in fr.pose_euler_deltas.items():
            if bname not in limits_map:
                continue
            jlim, ang_sign = limits_map[bname]
            if jlim is None:
                continue
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
    joint_limits_ok = max_limit_excess_deg <= 1e-3

    # 4. Clipping (15-axis OBB SAT across all frames)
    if precomputed_collision_events is not None:
        clip_collisions = int(precomputed_collision_events)
    else:
        clip_collisions = 0
        for fr in frames:
            aud = audit_scene(fr.scene, enforce_physical_contact=False)
            clip_collisions += len(aud["collision_pairs"])
    clipping_ok = clip_collisions == 0

    # 5. Balance (CoM projection vs active stance polygon or dynamic gait corridor)
    min_balance_margin = float("inf")
    body_w = max(1.0, morph.body_span[0])
    for fr in frames:
        # Compute posed CoM
        m_tot = 0.0
        c_acc = np.zeros(3, dtype=np.float64)
        for bx in fr.scene.boxes:
            if getattr(bx, "plane_normal", None) is not None:
                continue
            v = float(bx.size[0] * bx.size[1] * bx.size[2])
            m_tot += v
            c_acc += v * bx.center
        com_p = (c_acc / max(1e-9, m_tot)) + _virtual_adv_vec(fr)
        stance_pts = [
            (fr.effector_world_positions[r][0], fr.effector_world_positions[r][2])
            for r, in_st in fr.effector_in_stance.items()
            if in_st and r in fr.effector_world_positions
        ]
        if len(stance_pts) >= 3 and float(gait.froude) <= 0.30:
            hull = _convex_hull_2d(stance_pts)
            margin = point_in_convex_polygon_margin((float(com_p[0]), float(com_p[2])), hull)
            min_balance_margin = min(min_balance_margin, margin)
        elif morph.limbs:
            # Dynamic gait / biped: lateral CoM stays within outer foot corridor
            all_xs = [fr.effector_world_positions[lb.role][0] for lb in morph.limbs if lb.role in fr.effector_world_positions]
            if all_xs:
                lat_Span = max(1.0, max(all_xs) - min(all_xs))
                lat_center = 0.5 * (max(all_xs) + min(all_xs))
                margin = 0.65 * lat_Span - abs(float(com_p[0]) - lat_center)
                min_balance_margin = min(min_balance_margin, margin)
    if not math.isfinite(min_balance_margin):
        min_balance_margin = 0.0
    balance_tol = -0.42 * body_w
    balance_ok = min_balance_margin >= balance_tol

    # 6. Ground plane invariant (no foot penetrates below ground_y - 0.10u; stance feet sit on ground_y)
    min_foot_rel_y = float("inf")
    max_stance_float_y = 0.0
    for fr in frames:
        for role, pos_w in fr.effector_world_positions.items():
            rel_y = float(pos_w[1]) - morph.ground_y
            min_foot_rel_y = min(min_foot_rel_y, rel_y)
            if fr.effector_in_stance.get(role, False):
                max_stance_float_y = max(max_stance_float_y, abs(rel_y))
    if not math.isfinite(min_foot_rel_y):
        min_foot_rel_y = 0.0
    ground_ok = (min_foot_rel_y >= -0.10) and (max_stance_float_y <= 0.15)

    # 7. Loop closure (pose at phase=0 matches pose at phase=1 for looping gaits)
    max_loop_jump_deg = 0.0
    if clip.loop:
        endpoints = evaluate_gait_clip(program, clip, sample_phases=[0.0, 1.0])
        p_start = endpoints[0].pose_euler_deltas
        p_end = endpoints[1].pose_euler_deltas
        for bname in set(p_start) | set(p_end):
            v0 = np.asarray(p_start.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            v1 = np.asarray(p_end.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            max_loop_jump_deg = max(max_loop_jump_deg, float(np.max(np.abs(v1 - v0))))
    loop_ok = max_loop_jump_deg <= 0.5

    # 8. Speed match (root forward velocity matches stride * cycles / duration_s; in_place also verifies 0 macro Z-drift)
    expected_disp = float(kin["cycles_in_clip"]) * float(kin["stride_length"])
    if in_place_mode:
        rm_v = frames[0].root_motion_velocity if frames else (0.0, 0.0, 0.0)
        measured_disp = float(np.linalg.norm(np.asarray(rm_v, dtype=np.float64))) * float(clip.duration_s)
        max_exported_root_z = max((abs(float(fr.root_translation_delta[2])) for fr in frames), default=0.0)
        rel_speed_err = abs(measured_disp - expected_disp) / max(1e-6, expected_disp) if expected_disp > 1e-6 else 0.0
        speed_match_ok = (rel_speed_err <= 0.02) and (max_exported_root_z <= 0.05)
    else:
        if len(frames) >= 2 and abs(frames[-1].phase - frames[0].phase) > 1e-6:
            d_phi = float(frames[-1].phase - frames[0].phase)
            d_root_z = abs(float(frames[-1].root_translation_delta[2] - frames[0].root_translation_delta[2]))
            measured_disp = d_root_z / d_phi
        else:
            measured_disp = expected_disp
        rel_speed_err = abs(measured_disp - expected_disp) / max(1e-6, expected_disp) if expected_disp > 1e-6 else 0.0
        speed_match_ok = rel_speed_err <= 0.02

    checks = {
        "foot_slip": foot_slip_ok,
        "reach": reach_ok,
        "joint_limits": joint_limits_ok,
        "clipping": clipping_ok,
        "balance": balance_ok,
        "ground": ground_ok,
        "loop": loop_ok,
        "speed_match": speed_match_ok,
    }
    passed = all(checks.values())
    upb = float(getattr(program.meta, "units_per_block", 16))
    spd_u = float(kin["achieved_speed_u_s"])
    return {
        "passed": passed,
        "checks": checks,
        "froude": float(gait.froude),
        "duty": float(gait.duty),
        "pattern": gait.pattern,
        "mean_hip_height": round(float(kin["mean_hip_height"]), 4),
        "stride_length": round(float(kin["stride_length"]), 4),
        "cycles_in_clip": int(kin["cycles_in_clip"]),
        "max_stance_slip_units": round(measured_slip, 6),
        "max_reach_ratio": round(max_reach_ratio, 4),
        "max_joint_limit_excess_deg": round(max_limit_excess_deg, 5),
        "min_balance_margin": round(min_balance_margin, 4),
        "min_foot_rel_ground_y": round(min_foot_rel_y, 4),
        "max_stance_float_y": round(max_stance_float_y, 4),
        "max_loop_jump_deg": round(max_loop_jump_deg, 5),
        "relative_speed_error": round(rel_speed_err, 6),
        "root_motion": {
            "in_place": in_place_mode,
            "speed_units_per_s": round(spd_u, 4),
            "speed_blocks_per_s": round(spd_u / max(1.0, upb), 4),
            "velocity_vec": [0.0, 0.0, round(morph.forward_sign_z * spd_u, 4)],
            "displacement_per_cycle": [0.0, 0.0, round(morph.forward_sign_z * expected_disp, 4)],
        },
    }
