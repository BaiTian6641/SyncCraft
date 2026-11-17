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

from synccraft.dsl.ast_nodes import ClipDecl, GaitDecl, IKChainDecl, Program, Vec3, _is_continuous_limit_pair
from synccraft.kinematics.morphology import (
    LimbMorphology,
    MorphologyReport,
    analyze_morphology,
    _convex_hull_2d,
    point_in_convex_polygon_margin,
)
from synccraft.kinematics.se3_chain import compile_geometry, compile_posed_geometry

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
    if not has_fore and not has_mid:
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
    if not morph.limbs:
        return {
            "gravity_u_s2": g_u,
            "mean_hip_height": 1.0,
            "froude_target_speed": 0.0,
            "stride_length": 0.0,
            "cycles_in_clip": 1,
            "cycle_period_s": max(0.1, duration_s),
            "achieved_speed_u_s": 0.0,
            "crouch_y": 0.0,
            "lift_height": 0.0,
            "heave_amp": 0.0,
        }

    mean_h = float(np.mean([lb.hip_height for lb in morph.limbs]))
    fr = max(1e-4, float(gait.froude))
    v_froude = math.sqrt(fr * g_u * max(0.5, mean_h))

    # Alexander & Jayes (1983): relative stride length S / h ≈ 2.3 * Fr^0.30
    s_alexander = 2.3 * mean_h * (fr ** 0.30)

    # Compute crouch bias so bind-pose legs stay safely flexed without saturating knee flexion limits
    needs_crouch = any(
        math.hypot(
            lb.contact_world[0] - lb.hip_world[0],
            math.hypot(lb.contact_world[1] - lb.hip_world[1], lb.contact_world[2] - lb.hip_world[2]),
        )
        >= 0.90 * lb.reach[1]
        for lb in morph.limbs
    )
    crouch_y = -0.12 * mean_h * (0.75 + 0.4 * float(gait.weight)) if needs_crouch else -0.04 * mean_h

    # Bound stride by every limb's 3D IK reach envelope during stance + heave excursion
    heave_amp = float(gait.heave_ratio) * mean_h * (0.8 + 0.4 * float(gait.weight))
    duty = float(gait.duty)
    safe_strides: list[float] = []
    for lb in morph.limbs:
        dx = float(lb.contact_world[0] - lb.hip_world[0])
        dy_worst = abs(float(lb.contact_world[1] - (lb.hip_world[1] + crouch_y + heave_amp)))
        r_max = 0.96 * float(lb.reach[1])
        horiz_budget_sq = max(0.0, r_max * r_max - dy_worst * dy_worst - dx * dx)
        max_half_stance_z = 0.86 * math.sqrt(horiz_budget_sq)
        # Account for rest Z offset of foot relative to hip
        dz_rest = abs(float(lb.contact_world[2] - lb.hip_world[2]))
        avail_half_z = max(0.18 * lb.hip_height, max_half_stance_z - 0.30 * dz_rest)
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
        _evaluate_track_angle,
        evaluate_keyframe_track,
    )

    gait = clip.gaits[0]
    if _phase_cache is not None and "__setup__" in _phase_cache:
        rest_scene, morph, limits_map, limb_phases, kin = _phase_cache["__setup__"]
    else:
        rest_scene = compile_geometry(program, validate=False)
        morph = analyze_morphology(program, rest_scene)
        limits_map = _collect_expanded_joint_limits(program)
        limb_phases = resolve_limb_phases(gait, morph)
        kin = compute_froude_kinematics(program, gait, morph, float(clip.duration_s))
        if _phase_cache is not None:
            _phase_cache["__setup__"] = (rest_scene, morph, limits_map, limb_phases, kin)

    state_pose_bias: dict[str, Vec3] = {}
    if active_state is not None and program.behavior is not None:
        st_map = {s.name: s for s in program.behavior.states}
        if active_state in st_map:
            state_pose_bias = dict(st_map[active_state].pose_biases)

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
    rm_vel: Vec3 = (0.0, 0.0, fwd_z * spd_u_s) if gait.export_root_motion else (0.0, 0.0, 0.0)

    ik_decl_by_name: dict[str, IKChainDecl] = (
        {c.name: c for c in program.kinematics.ik_chains} if program.kinematics is not None else {}
    )
    axis_idx = {"x": 0, "y": 1, "z": 2}
    root_bone_name = rest_scene.bones[0].name

    # Pre-batch secondary motion `layers` once across all phases
    all_times_s = [float(p) * float(clip.duration_s) for p in phases]
    batched_layers = [
        _batch_sample_clip_layer_deltas(program, lyr.source_clip, all_times_s, float(lyr.weight), parent_clip=clip)
        for lyr in getattr(clip, "layers", ())
    ]

    samples: list[BehaviorFrameSample] = []
    for f_idx, phi_raw in enumerate(phases):
        phi_clip = float(phi_raw)
        phi_key = round(phi_clip, 8)
        if _phase_cache is not None and phi_key in _phase_cache:
            samples.append(dataclasses.replace(_phase_cache[phi_key], frame_index=f_idx))
            continue
        t_s = phi_clip * float(clip.duration_s)
        cyc_prog = phi_clip * float(n_cycles)
        cyc_phi = cyc_prog % 1.0

        # 1. Root motion: steady forward advance + crouch + 2x-per-stride vertical CoM heave
        heave_y = crouch_y - heave_amp * math.cos(4.0 * math.pi * cyc_phi)
        adv_z = fwd_z * cyc_prog * stride_len
        root_delta: Vec3 = (0.0, float(heave_y), float(adv_z))

        # 2. Axial posture (spine wave, torso pitch, neck counter-bob, phase-lagged tail wave)
        pose_map: dict[str, np.ndarray] = {
            b: np.asarray(v, dtype=np.float64).copy() for b, v in state_pose_bias.items()
        }

        # Apply any secondary explicit tracks / keyframes overlaid in the same clip
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

        # Root / spine pitch & lateral S-wave (scaled safely within joint_limits)
        two_pi = 2.0 * math.pi * cyc_phi
        non_root_spine = [b for b in morph.spine if b != root_bone_name]
        if non_root_spine:
            per_sp_wave = float(gait.spine_wave_deg) / float(len(non_root_spine))
            per_sp_pitch = float(gait.pitch_deg) / float(len(non_root_spine))
            for s_i, sb in enumerate(non_root_spine):
                sv = pose_map.setdefault(sb, np.zeros(3, dtype=np.float64))
                sv[0] += per_sp_pitch * math.sin(2.0 * two_pi)
                sv[1] += per_sp_wave * math.sin(two_pi - 0.4 * s_i)
        elif abs(float(gait.pitch_deg)) > 1e-6:
            rv = pose_map.setdefault(root_bone_name, np.zeros(3, dtype=np.float64))
            rv[0] += 0.5 * float(gait.pitch_deg) * math.sin(2.0 * two_pi)

        # Neck counter-stabilization bob
        if morph.neck:
            per_nk = -0.6 * float(gait.pitch_deg) / float(len(morph.neck))
            for n_i, nb in enumerate(morph.neck):
                if nb == root_bone_name:
                    continue
                nv = pose_map.setdefault(nb, np.zeros(3, dtype=np.float64))
                nv[0] += per_nk * math.sin(2.0 * two_pi - 0.25 * n_i)

        # Traveling caudal wave down the tail chain
        if morph.tail:
            base_tail_amp = max(2.5, 1.4 * float(gait.spine_wave_deg)) / math.sqrt(float(len(morph.tail)))
            for t_i, tb in enumerate(morph.tail):
                if tb == root_bone_name:
                    continue
                tv = pose_map.setdefault(tb, np.zeros(3, dtype=np.float64))
                lag_rad = 2.0 * math.pi * float(gait.tail_lag) * (t_i + 1)
                tv[1] += base_tail_amp * (1.0 + 0.12 * t_i) * math.sin(two_pi - lag_rad)

        for b_k in list(pose_map):
            pose_map[b_k] = _clamp_pose_vec(b_k, pose_map[b_k], limits_map)

        cur_poses: dict[str, Vec3] = {
            b: (float(v[0]), float(v[1]), float(v[2])) for b, v in pose_map.items()
        }

        # 3. Closed-loop world-planted limb IK solved AFTER axial spine/torso motion (compensates spine motion!)
        posture_scene = (
            compile_posed_geometry(
                program,
                cur_poses,
                pose_position_deltas=frame_pos_deltas or None,
                pose_scale_factors=frame_scale_factors or None,
                validate=False,
            )
            if morph.limbs and not frame_scale_factors
            else None
        )
        eff_positions: dict[str, Vec3] = {}
        eff_stance: dict[str, bool] = {}
        eff_step_ids: dict[str, int] = {}
        stride_vec = np.array([0.0, 0.0, fwd_z * stride_len], dtype=np.float64)

        for lb in morph.limbs:
            phase_off = limb_phases.get(lb.role, 0.0)
            eff_rest_w = np.array(
                [lb.contact_world[0], morph.ground_y, lb.contact_world[2]],
                dtype=np.float64,
            )

            if stride_len <= 1e-6:
                tgt_w = eff_rest_w.copy()
                in_stance = True
                step_id = 0
            else:
                raw_p = cyc_prog + phase_off
                wrap_k = math.floor(raw_p)
                u = raw_p - wrap_k
                stance_anchor = eff_rest_w + (wrap_k - phase_off + 0.5 * duty) * stride_vec
                if u < duty:
                    tgt_w = stance_anchor.copy()
                    in_stance = True
                    step_id = int(wrap_k)
                else:
                    s = (u - duty) / max(1e-6, 1.0 - duty)
                    # Snap sharpens swing acceleration without creating boundary velocity jumps
                    snap_g = 0.35 * float(gait.snap)
                    s_warped = (1.0 - snap_g) * s + snap_g * (0.5 - 0.5 * math.cos(math.pi * s))
                    sigma = 0.5 - 0.5 * math.cos(math.pi * s_warped)
                    lift_y = lift_h * (math.sin(math.pi * s) ** 2)
                    tgt_w = stance_anchor + sigma * stride_vec + np.array([0.0, lift_y, 0.0], dtype=np.float64)
                    in_stance = False
                    step_id = -1

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
                validate=False,
                _limits_cache=limits_map,
                _zero_scene=posture_scene,
            )
            cur_poses = ik_res.pose_euler_deltas
            ach_w = (
                float(ik_res.achieved_world[0] + root_delta[0]),
                float(ik_res.achieved_world[1] + root_delta[1]),
                float(ik_res.achieved_world[2] + root_delta[2]),
            )
            eff_positions[lb.role] = ach_w
            eff_stance[lb.role] = in_stance
            eff_step_ids[lb.role] = step_id

        final_scene = compile_posed_geometry(
            program,
            cur_poses,
            root_translation_delta=root_delta,
            pose_position_deltas=frame_pos_deltas or None,
            pose_scale_factors=frame_scale_factors or None,
            validate=False,
        )
        sample_obj = BehaviorFrameSample(
            frame_index=f_idx,
            time_s=t_s,
            phase=phi_clip,
            pose_euler_deltas=cur_poses,
            box_inflate_deltas={},
            root_translation_delta=root_delta,
            muscle_activations={},
            effector_world_positions=eff_positions,
            effector_in_stance=eff_stance,
            effector_stance_step_id=eff_step_ids,
            scene=final_scene,
            pose_position_deltas=frame_pos_deltas,
            pose_scale_factors=frame_scale_factors,
            root_motion_velocity=rm_vel,
        )
        if _phase_cache is not None:
            _phase_cache[phi_key] = sample_obj
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
    rest_scene = compile_geometry(program)
    morph = analyze_morphology(program, rest_scene)
    kin = compute_froude_kinematics(program, gait, morph, float(clip.duration_s))
    limits_map = _collect_expanded_joint_limits(program)

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
        for lb in morph.limbs:
            if lb.role not in fr.effector_world_positions:
                continue
            hip_p = fr.scene.bone_by_name[lb.chain[0]].pivot_world
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
        com_p = c_acc / max(1e-9, m_tot)
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

    # 8. Speed match (root forward velocity matches stride * cycles / duration_s)
    expected_disp = float(kin["cycles_in_clip"]) * float(kin["stride_length"])
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
            "speed_units_per_s": round(spd_u, 4),
            "speed_blocks_per_s": round(spd_u / max(1.0, upb), 4),
            "velocity_vec": [0.0, 0.0, round(morph.forward_sign_z * spd_u, 4)],
            "displacement_per_cycle": [0.0, 0.0, round(morph.forward_sign_z * expected_disp, 4)],
        },
    }
