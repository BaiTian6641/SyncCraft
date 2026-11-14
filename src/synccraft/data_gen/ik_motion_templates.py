"""Analytical + damped-Jacobian 2-bone IK and parameterized joint trajectories (P1-03d).

Shares SyncCraft's exact right-handed SE(3) kinematics (`compile_posed_geometry`).
Note: Textual `@motion` DSL grammar and full quadruped stance-foot lock benchmarks
remain tracked under ADR 0003 / P3-01; this module provides typed trajectory and
IK operators for action-conditioned WorldState evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from synccraft.dsl.ast_nodes import Program, Vec3
from synccraft.kinematics.se3_chain import compile_posed_geometry


@dataclass(frozen=True)
class TwoBoneIKResult:
    """Result of solving 2-bone limb IK to a world-space target."""

    pose_euler_deltas: dict[str, Vec3]
    achieved_world: Vec3
    target_world: Vec3
    clamped_target_world: Vec3
    reachable: bool
    position_error: float
    iterations: int


def _effector_world(
    program: Program,
    pose_deltas: dict[str, Vec3],
    lower_bone_name: str,
    effector_local: np.ndarray,
) -> tuple[np.ndarray, Any]:
    scene = compile_posed_geometry(program, pose_deltas)
    lower = scene.bone_by_name[lower_bone_name]
    eff_w = lower.pivot_world + lower.rotation @ effector_local
    return eff_w, scene


def solve_two_bone_ik(
    program: Program,
    upper_bone: str,
    lower_bone: str,
    effector_local: Vec3,
    target_world: Vec3,
    *,
    base_pose_deltas: dict[str, Vec3] | None = None,
    hinge_axis: str = "x",
    max_iters: int = 28,
    tol: float = 1e-5,
) -> TwoBoneIKResult:
    """Solve 2-bone limb IK (3-DOF upper joint + 1-DOF lower hinge) in exact SE(3).

    Uses Law-of-Cosines reachability clamping + damped Levenberg-Marquardt updates
    through ``compile_posed_geometry`` so mirror_x reflections and arbitrary rest_euler
    frames agree to machine precision.
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}.get(hinge_axis)
    if axis_idx is None:
        raise ValueError(f"hinge_axis must be 'x', 'y', or 'z', got {hinge_axis!r}")

    eff_loc = np.asarray(effector_local, dtype=np.float64)
    tgt_w = np.asarray(target_world, dtype=np.float64)

    poses: dict[str, Vec3] = dict(base_pose_deltas or {})
    poses.setdefault(upper_bone, (0.0, 0.0, 0.0))
    poses.setdefault(lower_bone, (0.0, 0.0, 0.0))

    init_scene = compile_posed_geometry(program, poses)
    if upper_bone not in init_scene.bone_by_name or lower_bone not in init_scene.bone_by_name:
        raise KeyError(f"Unknown IK bone {upper_bone!r} or {lower_bone!r}")
    up_b = init_scene.bone_by_name[upper_bone]
    lo_b = init_scene.bone_by_name[lower_bone]
    if lo_b.parent_name != upper_bone:
        raise ValueError(f"lower_bone {lower_bone!r} must be a direct child of upper_bone {upper_bone!r}")

    root_p = up_b.pivot_world.copy()
    l1 = float(np.linalg.norm(lo_b.local_translation))
    l2 = float(np.linalg.norm(eff_loc))
    max_reach = max(1e-6, l1 + l2 - 1e-4)
    min_reach = abs(l1 - l2) + 1e-4

    delta_vec = tgt_w - root_p
    dist = float(np.linalg.norm(delta_vec))
    reachable = (min_reach - 1e-3) <= dist <= (l1 + l2 + 1e-3)
    if dist > max_reach and dist > 1e-9:
        clamped_tgt = root_p + delta_vec * (max_reach / dist)
    elif dist < min_reach and dist > 1e-9:
        clamped_tgt = root_p + delta_vec * (min_reach / dist)
    elif dist <= 1e-9:
        clamped_tgt = root_p + np.array([0.0, -min_reach, 0.0], dtype=np.float64)
    else:
        clamped_tgt = tgt_w.copy()

    # Analytical Law-of-Cosines warmstart for elbow/knee flexion angle
    d_clamp = float(np.linalg.norm(clamped_tgt - root_p))
    if l1 > 1e-6 and l2 > 1e-6:
        cos_knee = float(np.clip((d_clamp * d_clamp - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0))
        knee_deg = math.degrees(math.acos(cos_knee))
    else:
        knee_deg = 0.0

    # Parameter vector q = [upper_rx, upper_ry, upper_rz, lower_hinge]
    u0 = poses[upper_bone]
    l0 = poses[lower_bone]
    best_q: np.ndarray | None = None
    best_err = float("inf")

    for sign_try in (1.0, -1.0):
        q = np.array([u0[0], u0[1], u0[2], l0[axis_idx] + sign_try * knee_deg * 0.5], dtype=np.float64)
        lam = 1e-2
        eps = 1e-4
        iters_used = 0

        def _apply_q(vec_q: np.ndarray) -> dict[str, Vec3]:
            d = dict(poses)
            d[upper_bone] = (float(vec_q[0]), float(vec_q[1]), float(vec_q[2]))
            lo_list = list(l0)
            lo_list[axis_idx] = float(vec_q[3])
            d[lower_bone] = (lo_list[0], lo_list[1], lo_list[2])
            return d

        cur_eff, _ = _effector_world(program, _apply_q(q), lower_bone, eff_loc)
        cur_err = float(np.linalg.norm(cur_eff - clamped_tgt))

        for it in range(1, max_iters + 1):
            iters_used = it
            if cur_err <= tol:
                break
            jac = np.zeros((3, 4), dtype=np.float64)
            for j in range(4):
                q_plus = q.copy()
                q_plus[j] += eps
                eff_plus, _ = _effector_world(program, _apply_q(q_plus), lower_bone, eff_loc)
                jac[:, j] = (eff_plus - cur_eff) / eps

            residual = clamped_tgt - cur_eff
            lhs = jac.T @ jac + lam * np.eye(4, dtype=np.float64)
            dq = np.linalg.solve(lhs, jac.T @ residual)
            # Limit single-step angle jump for smooth convergence
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > 25.0:
                dq *= 25.0 / dq_norm

            q_cand = q + dq
            cand_eff, _ = _effector_world(program, _apply_q(q_cand), lower_bone, eff_loc)
            cand_err = float(np.linalg.norm(cand_eff - clamped_tgt))
            if cand_err < cur_err:
                q = q_cand
                cur_eff = cand_eff
                cur_err = cand_err
                lam = max(1e-6, lam * 0.5)
            else:
                lam = min(1e2, lam * 4.0)

        if cur_err < best_err:
            best_err = cur_err
            best_q = q.copy()
        if best_err <= tol:
            break

    assert best_q is not None
    final_poses = dict(poses)
    final_poses[upper_bone] = (float(best_q[0]), float(best_q[1]), float(best_q[2]))
    lo_final = list(l0)
    lo_final[axis_idx] = float(best_q[3])
    final_poses[lower_bone] = (lo_final[0], lo_final[1], lo_final[2])
    achieved_w, _ = _effector_world(program, final_poses, lower_bone, eff_loc)

    return TwoBoneIKResult(
        pose_euler_deltas=final_poses,
        achieved_world=(float(achieved_w[0]), float(achieved_w[1]), float(achieved_w[2])),
        target_world=(float(tgt_w[0]), float(tgt_w[1]), float(tgt_w[2])),
        clamped_target_world=(float(clamped_tgt[0]), float(clamped_tgt[1]), float(clamped_tgt[2])),
        reachable=reachable,
        position_error=float(np.linalg.norm(achieved_w - clamped_tgt)),
        iterations=iters_used,
    )


def sample_joint_trajectory(
    keyframes: list[tuple[float, dict[str, Vec3]]],
    num_samples: int,
) -> list[tuple[float, dict[str, Vec3]]]:
    """Piecewise smooth-step (cosine-eased) interpolation across pose keyframes."""
    if not keyframes:
        raise ValueError("keyframes must be non-empty")
    if num_samples < 2:
        return [keyframes[-1]]

    sorted_kf = sorted(keyframes, key=lambda item: float(item[0]))
    t_start = float(sorted_kf[0][0])
    t_end = float(sorted_kf[-1][0])
    all_bones = sorted({b for _, pose in sorted_kf for b in pose})

    times = np.linspace(t_start, t_end, num_samples, dtype=np.float64)
    samples: list[tuple[float, dict[str, Vec3]]] = []

    for t_val in times:
        if t_val <= t_start or len(sorted_kf) == 1:
            pose_out = {b: sorted_kf[0][1].get(b, (0.0, 0.0, 0.0)) for b in all_bones}
            samples.append((float(t_val), pose_out))
            continue
        if t_val >= t_end:
            pose_out = {b: sorted_kf[-1][1].get(b, (0.0, 0.0, 0.0)) for b in all_bones}
            samples.append((float(t_val), pose_out))
            continue

        seg_idx = 0
        for i in range(len(sorted_kf) - 1):
            if sorted_kf[i][0] <= t_val <= sorted_kf[i + 1][0]:
                seg_idx = i
                break
        t0, p0 = sorted_kf[seg_idx]
        t1, p1 = sorted_kf[seg_idx + 1]
        alpha = (t_val - t0) / max(1e-9, t1 - t0)
        smooth_a = 0.5 - 0.5 * math.cos(math.pi * alpha)
        pose_interp: dict[str, Vec3] = {}
        for b in all_bones:
            v0 = np.asarray(p0.get(b, (0.0, 0.0, 0.0)), dtype=np.float64)
            v1 = np.asarray(p1.get(b, (0.0, 0.0, 0.0)), dtype=np.float64)
            vm = (1.0 - smooth_a) * v0 + smooth_a * v1
            pose_interp[b] = (float(vm[0]), float(vm[1]), float(vm[2]))
        samples.append((float(t_val), pose_interp))

    return samples


def generate_ik_clip(
    program: Program,
    upper_bone: str,
    lower_bone: str,
    effector_local: Vec3,
    waypoints_world: list[tuple[float, Vec3]],
    *,
    hinge_axis: str = "x",
    num_frames: int = 16,
) -> dict[str, Any]:
    """Solve 2-bone IK across world-space effector waypoints and sample a trajectory clip."""
    if not waypoints_world:
        raise ValueError("waypoints_world must contain at least one (time, target_xyz) tuple")
    ik_keyframes: list[tuple[float, dict[str, Vec3]]] = []
    max_ik_err = 0.0
    all_reachable = True
    cur_pose: dict[str, Vec3] = {}

    for t_wp, tgt in waypoints_world:
        res = solve_two_bone_ik(
            program,
            upper_bone,
            lower_bone,
            effector_local,
            tgt,
            base_pose_deltas=cur_pose,
            hinge_axis=hinge_axis,
        )
        cur_pose = res.pose_euler_deltas
        max_ik_err = max(max_ik_err, res.position_error)
        all_reachable = all_reachable and res.reachable
        ik_keyframes.append((float(t_wp), cur_pose))

    frames = sample_joint_trajectory(ik_keyframes, num_samples=num_frames)
    return {
        "upper_bone": upper_bone,
        "lower_bone": lower_bone,
        "frames": frames,
        "max_ik_clamped_error": max_ik_err,
        "all_waypoints_reachable": all_reachable,
        "scope_note": "Typed P1-03d trajectory clip; @motion text grammar remains deferred per ADR 0003.",
    }
