"""Analytical + damped-Jacobian 2-bone and N-bone IK & joint trajectories (P1-03d / v3.2).

Shares SyncCraft's exact right-handed SE(3) kinematics (`compile_posed_geometry`).
Generalizes IK solving to arbitrary contiguous N >= 2 bone chains with optional
`pole_world` plane steering and exact `joint_limits` clamping, while preserving the
fast Law-of-Cosines + 4-DOF solver when `len(bones) == 2` and `pole_world is None`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from synccraft.dsl.ast_nodes import Bone, Program, RadialSpec, Vec3
from synccraft.kinematics.se3_chain import (
    _angle_sign_from_reflection,
    _clamp_delta_to_joint_limits,
    compile_posed_geometry,
    euler_xyz_matrix,
)


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


@dataclass(frozen=True)
class NBoneIKResult:
    """Result of solving generalized N-bone (N >= 2) chain IK to a world-space target."""

    bones: tuple[str, ...]
    pose_euler_deltas: dict[str, Vec3]
    achieved_world: Vec3
    target_world: Vec3
    clamped_target_world: Vec3
    reachable: bool
    position_error: float
    iterations: int
    pole_world: Vec3 | None = None


def _effector_world(
    program: Program,
    pose_deltas: dict[str, Vec3],
    tip_bone_name: str,
    effector_local: np.ndarray,
    *,
    validate: bool = True,
) -> tuple[np.ndarray, Any]:
    scene = compile_posed_geometry(program, pose_deltas, validate=validate)
    tip = scene.bone_by_name[tip_bone_name]
    eff_w = tip.pivot_world + tip.rotation @ effector_local
    return eff_w, scene


def _collect_expanded_limits_and_signs(program: Program) -> dict[str, tuple[Any, np.ndarray]]:
    """Map expanded bone name -> (joint_limits, angle_sign) across mirror_x, mirror(axis=z), and radial."""
    out: dict[str, tuple[Any, np.ndarray]] = {}

    def _walk(
        node: Bone,
        *,
        sign: np.ndarray,
        prefix: str | None = None,
        renamed: str | None = None,
    ) -> None:
        exp_name = renamed or (f"{prefix}__{node.name}" if prefix else node.name)
        out[exp_name] = (node.joint_limits, _angle_sign_from_reflection(sign))
        for ch in node.items:
            if isinstance(ch, Bone):
                _walk(ch, sign=sign, prefix=prefix)
                m_flip = (
                    np.array((-1.0, 1.0, 1.0), dtype=np.float64)
                    if ch.mirror_axis == "x"
                    else np.array((1.0, 1.0, -1.0), dtype=np.float64)
                )
                if ch.mirror_name:
                    mp = f"{prefix}__{ch.mirror_name}" if prefix else ch.mirror_name
                    _walk(ch, sign=sign * m_flip, prefix=mp, renamed=mp)
                if isinstance(ch.radial, RadialSpec):
                    for k in range(1, ch.radial.count):
                        rn = f"{ch.name}_r{k}"
                        rp = f"{prefix}__{rn}" if prefix else rn
                        _walk(ch, sign=sign, prefix=rp, renamed=rp)
                        if ch.mirror_name:
                            rmn = f"{ch.mirror_name}_r{k}"
                            rmp = f"{prefix}__{rmn}" if prefix else rmn
                            _walk(ch, sign=sign * m_flip, prefix=rmp, renamed=rmp)

    _walk(program.root, sign=np.ones(3, dtype=np.float64))
    return out


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
    validate: bool = True,
    _limits_cache: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> TwoBoneIKResult:
    """Solve 2-bone limb IK (3-DOF upper joint + 1-DOF lower hinge) in exact SE(3).

    Uses Law-of-Cosines reachability clamping + damped Levenberg-Marquardt updates
    through ``compile_posed_geometry`` so mirror reflections and arbitrary rest_euler
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

    init_scene = compile_posed_geometry(program, poses, validate=validate)
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

    d_clamp = float(np.linalg.norm(clamped_tgt - root_p))
    if l1 > 1e-6 and l2 > 1e-6:
        cos_knee = float(np.clip((d_clamp * d_clamp - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0))
        knee_deg = math.degrees(math.acos(cos_knee))
    else:
        knee_deg = 0.0

    if poses[upper_bone] == (0.0, 0.0, 0.0) and poses[lower_bone] == (0.0, 0.0, 0.0):
        zero_scene = init_scene
    else:
        zero_poses = dict(poses)
        zero_poses[upper_bone] = (0.0, 0.0, 0.0)
        zero_poses[lower_bone] = (0.0, 0.0, 0.0)
        zero_scene = compile_posed_geometry(program, zero_poses, validate=False)
    up_0 = zero_scene.bone_by_name[upper_bone]
    lo_0 = zero_scene.bone_by_name[lower_bone]
    r_up_0 = up_0.rotation
    r_lo_local_rest = r_up_0.T @ lo_0.rotation
    t_lo = lo_0.local_translation

    limits_by_expanded = _limits_cache if _limits_cache is not None else _collect_expanded_limits_and_signs(program)
    up_lim, up_asign = limits_by_expanded.get(upper_bone, (None, np.ones(3)))
    lo_lim, lo_asign = limits_by_expanded.get(lower_bone, (None, np.ones(3)))

    u0 = poses[upper_bone]
    l0 = poses[lower_bone]
    best_q: np.ndarray | None = None
    best_err = float("inf")

    def _fast_eff(vec_q: np.ndarray) -> np.ndarray:
        du = vec_q[:3]
        if up_lim is not None:
            du = _clamp_delta_to_joint_limits(du, up_lim, angle_sign=up_asign)
        dl = np.array([l0[0], l0[1], l0[2]], dtype=np.float64)
        dl[axis_idx] = float(vec_q[3])
        if lo_lim is not None:
            dl = _clamp_delta_to_joint_limits(dl, lo_lim, angle_sign=lo_asign)
        r_u = r_up_0 @ euler_xyz_matrix(du)
        p_l = root_p + r_u @ t_lo
        r_l = r_u @ r_lo_local_rest @ euler_xyz_matrix(dl)
        return p_l + r_l @ eff_loc

    iters_used = 0
    for sign_try in (1.0, -1.0):
        q = np.array([u0[0], u0[1], u0[2], l0[axis_idx] + sign_try * knee_deg * 0.5], dtype=np.float64)
        lam = 1e-2
        eps = 1e-4

        cur_eff = _fast_eff(q)
        cur_err = float(np.linalg.norm(cur_eff - clamped_tgt))

        for it in range(1, max_iters + 1):
            iters_used = it
            if cur_err <= tol:
                break
            jac = np.zeros((3, 4), dtype=np.float64)
            for j in range(4):
                q_plus = q.copy()
                q_plus[j] += eps
                eff_plus = _fast_eff(q_plus)
                jac[:, j] = (eff_plus - cur_eff) / eps

            residual = clamped_tgt - cur_eff
            lhs = jac.T @ jac + lam * np.eye(4, dtype=np.float64)
            dq = np.linalg.solve(lhs, jac.T @ residual)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > 25.0:
                dq *= 25.0 / dq_norm

            q_cand = q + dq
            cand_eff = _fast_eff(q_cand)
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
    du_final = best_q[:3]
    if up_lim is not None:
        du_final = _clamp_delta_to_joint_limits(du_final, up_lim, angle_sign=up_asign)
    final_poses[upper_bone] = (float(du_final[0]), float(du_final[1]), float(du_final[2]))
    lo_final = np.array([l0[0], l0[1], l0[2]], dtype=np.float64)
    lo_final[axis_idx] = float(best_q[3])
    if lo_lim is not None:
        lo_final = _clamp_delta_to_joint_limits(lo_final, lo_lim, angle_sign=lo_asign)
    final_poses[lower_bone] = (float(lo_final[0]), float(lo_final[1]), float(lo_final[2]))
    achieved_w, _ = _effector_world(program, final_poses, lower_bone, eff_loc, validate=False)

    return TwoBoneIKResult(
        pose_euler_deltas=final_poses,
        achieved_world=(float(achieved_w[0]), float(achieved_w[1]), float(achieved_w[2])),
        target_world=(float(tgt_w[0]), float(tgt_w[1]), float(tgt_w[2])),
        clamped_target_world=(float(clamped_tgt[0]), float(clamped_tgt[1]), float(clamped_tgt[2])),
        reachable=reachable,
        position_error=float(np.linalg.norm(achieved_w - clamped_tgt)),
        iterations=iters_used,
    )


def solve_n_bone_ik(
    program: Program,
    bones: tuple[str, ...] | list[str],
    effector_local: Vec3,
    target_world: Vec3,
    *,
    base_pose_deltas: dict[str, Vec3] | None = None,
    hinge_axis: str = "x",
    pole_world: Vec3 | None = None,
    plane: str | None = None,
    max_iters: int = 40,
    tol: float = 1e-5,
    validate: bool = True,
    _limits_cache: dict[str, tuple[Any, np.ndarray]] | None = None,
    _zero_scene: Any | None = None,
) -> NBoneIKResult:
    """Solve generalized N-bone (N >= 2) chain IK in exact SE(3) with joint_limits and pole steering.

    Delegates directly to fast ``solve_two_bone_ik`` when ``len(bones) == 2`` and
    ``pole_world is None``. For ``N >= 3`` (or when ``pole_world`` is provided), runs
    damped Levenberg-Marquardt with nullspace pole-plane steering across the exact
    chained SO(3) bone frames and verifies final world effector coordinates via
    ``compile_posed_geometry``.
    """
    chain_bones = tuple(str(b) for b in bones)
    n_bones = len(chain_bones)
    if n_bones < 2:
        raise ValueError(f"solve_n_bone_ik requires at least 2 bones, got {n_bones}")

    axis_idx = {"x": 0, "y": 1, "z": 2}.get(hinge_axis)
    if axis_idx is None:
        raise ValueError(f"hinge_axis must be 'x', 'y', or 'z', got {hinge_axis!r}")

    plane_mode = plane or "free"
    if plane_mode not in ("free", "sagittal", "sprawled"):
        raise ValueError(f"plane must be 'free', 'sagittal', or 'sprawled', got {plane!r}")

    if n_bones == 2 and pole_world is None and plane_mode == "free":
        res2 = solve_two_bone_ik(
            program,
            chain_bones[0],
            chain_bones[1],
            effector_local,
            target_world,
            base_pose_deltas=base_pose_deltas,
            hinge_axis=hinge_axis,
            max_iters=max_iters,
            tol=tol,
            validate=validate,
            _limits_cache=_limits_cache,
        )
        return NBoneIKResult(
            bones=chain_bones,
            pose_euler_deltas=res2.pose_euler_deltas,
            achieved_world=res2.achieved_world,
            target_world=res2.target_world,
            clamped_target_world=res2.clamped_target_world,
            reachable=res2.reachable,
            position_error=res2.position_error,
            iterations=res2.iterations,
            pole_world=None,
        )

    eff_loc = np.asarray(effector_local, dtype=np.float64)
    tgt_w = np.asarray(target_world, dtype=np.float64)

    poses: dict[str, Vec3] = dict(base_pose_deltas or {})
    for b in chain_bones:
        poses.setdefault(b, (0.0, 0.0, 0.0))

    if _zero_scene is not None and all(poses[b] == (0.0, 0.0, 0.0) for b in chain_bones):
        init_scene = _zero_scene
        zero_scene = _zero_scene
    else:
        init_scene = compile_posed_geometry(program, poses, validate=validate)
        if all(poses[b] == (0.0, 0.0, 0.0) for b in chain_bones):
            zero_scene = init_scene
        else:
            zero_poses = dict(poses)
            for b in chain_bones:
                zero_poses[b] = (0.0, 0.0, 0.0)
            zero_scene = compile_posed_geometry(program, zero_poses, validate=False)
    for idx_b, b_name in enumerate(chain_bones):
        if b_name not in init_scene.bone_by_name:
            raise KeyError(f"Unknown IK bone {b_name!r}")
        if idx_b > 0 and init_scene.bone_by_name[b_name].parent_name != chain_bones[idx_b - 1]:
            raise ValueError(
                f"IK chain bone {b_name!r} must be a direct child of {chain_bones[idx_b - 1]!r}"
            )

    root_b0 = zero_scene.bone_by_name[chain_bones[0]]
    root_p = root_b0.pivot_world.copy()
    r_0_base = root_b0.rotation.copy()

    t_local: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
    r_local_rest: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    seg_lengths: list[float] = []
    for i in range(1, n_bones):
        bi = zero_scene.bone_by_name[chain_bones[i]]
        bip = zero_scene.bone_by_name[chain_bones[i - 1]]
        tl = bi.local_translation.copy()
        t_local.append(tl)
        r_local_rest.append(bip.rotation.T @ bi.rotation)
        seg_lengths.append(float(np.linalg.norm(tl)))
    seg_lengths.append(float(np.linalg.norm(eff_loc)))

    total_len = sum(seg_lengths)
    max_seg = max(seg_lengths) if seg_lengths else 0.0
    max_reach = max(1e-6, total_len - 1e-4)
    min_reach = max(0.0, 2.0 * max_seg - total_len) + 1e-4

    delta_vec = tgt_w - root_p
    dist = float(np.linalg.norm(delta_vec))
    reachable = (min_reach - 1e-3) <= dist <= (total_len + 1e-3)
    if dist > max_reach and dist > 1e-9:
        clamped_tgt = root_p + delta_vec * (max_reach / dist)
    elif dist < min_reach and dist > 1e-9:
        clamped_tgt = root_p + delta_vec * (min_reach / dist)
    elif dist <= 1e-9:
        clamped_tgt = root_p + np.array([0.0, -max(min_reach, 1e-3), 0.0], dtype=np.float64)
    else:
        clamped_tgt = tgt_w.copy()

    limits_by_expanded = _limits_cache if _limits_cache is not None else _collect_expanded_limits_and_signs(program)
    chain_limits = [limits_by_expanded.get(b, (None, np.ones(3))) for b in chain_bones]

    eff_pole_world = pole_world
    if eff_pole_world is None and plane_mode != "free":
        rest_mid = zero_scene.bone_by_name[chain_bones[1]].pivot_world
        rest_tip_b = zero_scene.bone_by_name[chain_bones[-1]]
        rest_eff = rest_tip_b.pivot_world + rest_tip_b.rotation @ eff_loc
        if plane_mode == "sagittal":
            dz_rest = float(rest_mid[2] - 0.5 * (root_p[2] + rest_eff[2]))
            fwd_s = 1.0 if dz_rest >= -1e-3 else -1.0
            eff_pole_world = (
                float(root_p[0]),
                float(root_p[1] + 0.25 * total_len),
                float(root_p[2] + fwd_s * max(16.0, 1.5 * total_len)),
            )
        elif plane_mode == "sprawled":
            lat_s = 1.0 if float(root_p[0]) >= 0.0 else -1.0
            if abs(float(root_p[0])) < 1e-3 and abs(float(rest_mid[0])) >= 1e-3:
                lat_s = 1.0 if float(rest_mid[0]) >= 0.0 else -1.0
            eff_pole_world = (
                float(root_p[0] + lat_s * max(18.0, 2.0 * total_len)),
                float(root_p[1] + 0.65 * total_len),
                float(root_p[2]),
            )

    pole_active = False
    n_pole = np.zeros(3, dtype=np.float64)
    b_plane = np.zeros(3, dtype=np.float64)
    chord_mid = root_p + 0.5 * (clamped_tgt - root_p)
    p_mid_ideal = chord_mid.copy()
    desired_bulge = 0.0
    pole_tuple: Vec3 | None = None
    if eff_pole_world is not None:
        pole_w = np.asarray(eff_pole_world, dtype=np.float64)
        pole_tuple = (float(pole_w[0]), float(pole_w[1]), float(pole_w[2]))
        chord = clamped_tgt - root_p
        chord_len = float(np.linalg.norm(chord))
        u_hat = chord / chord_len if chord_len > 1e-6 else np.array([0.0, -1.0, 0.0], dtype=np.float64)
        pole_rel = pole_w - root_p
        pole_perp = pole_rel - float(np.dot(pole_rel, u_hat)) * u_hat
        perp_norm = float(np.linalg.norm(pole_perp))
        if perp_norm > 1e-6:
            pole_active = True
            n_pole = pole_perp / perp_norm
            b_plane = np.cross(u_hat, n_pole)
            slack_sq = max(0.0, total_len * total_len - chord_len * chord_len)
            desired_bulge = max(0.04 * total_len, 0.30 * math.sqrt(slack_sq))
            if n_bones == 2:
                l1_s, l2_s = seg_lengths[0], seg_lengths[1]
                a_along = (l1_s * l1_s - l2_s * l2_s + chord_len * chord_len) / (2.0 * max(1e-6, chord_len))
                a_along = float(np.clip(a_along, -l1_s, l1_s))
                h_perp = math.sqrt(max(0.0, l1_s * l1_s - a_along * a_along))
                p_mid_ideal = root_p + a_along * u_hat + h_perp * n_pole
            else:
                p_mid_ideal = chord_mid + desired_bulge * n_pole

    n_dof = 3 * n_bones

    def _clamp_q(vec_q: np.ndarray) -> np.ndarray:
        out = vec_q.copy()
        for i in range(n_bones):
            jlim, asign = chain_limits[i]
            if jlim is not None:
                out[3 * i : 3 * i + 3] = _clamp_delta_to_joint_limits(
                    out[3 * i : 3 * i + 3], jlim, angle_sign=asign
                )
        return out

    def _fast_chain(vec_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        qc = _clamp_q(vec_q)
        p_cur = root_p
        r_cur = r_0_base @ euler_xyz_matrix(qc[0:3])
        mid_sum = np.zeros(3, dtype=np.float64)
        for i in range(1, n_bones):
            p_cur = p_cur + r_cur @ t_local[i]
            mid_sum = mid_sum + p_cur
            r_cur = r_cur @ r_local_rest[i] @ euler_xyz_matrix(qc[3 * i : 3 * i + 3])
        eff_pt = p_cur + r_cur @ eff_loc
        mid_pt = mid_sum / float(max(1, n_bones - 1))
        return eff_pt, mid_pt

    q_base = np.zeros(n_dof, dtype=np.float64)
    for i, b in enumerate(chain_bones):
        pb = poses[b]
        q_base[3 * i : 3 * i + 3] = (float(pb[0]), float(pb[1]), float(pb[2]))

    def _secondary_cost(vec_q: np.ndarray, mid_pt: np.ndarray) -> float:
        dq_b = vec_q - q_base
        reg = 2e-5 * float(np.dot(dq_b, dq_b))
        if plane_mode == "sagittal":
            for bi_k in range(1, n_bones):
                for ax_k in range(3):
                    if ax_k != axis_idx:
                        reg += 4e-3 * (float(vec_q[3 * bi_k + ax_k]) ** 2)
        if not pole_active:
            return reg
        diff_mid = mid_pt - p_mid_ideal
        wrong_side = max(0.0, -float(np.dot(mid_pt - chord_mid, n_pole)))
        return reg + 0.5 * float(np.dot(diff_mid, diff_mid)) + 4.0 * wrong_side * wrong_side

    d_clamp = float(np.linalg.norm(clamped_tgt - root_p))
    slack_ratio = max(0.0, 1.0 - d_clamp / max(1e-6, total_len))
    arch_deg = 32.0 * math.sqrt(slack_ratio)

    best_q: np.ndarray | None = None
    best_score = float("inf")
    best_err = float("inf")
    iters_used = 0
    eps = 1e-4

    # Smooth single-trajectory pole-plane Gauss-Newton + nullspace polish (zero branch jumps)
    if pole_active:
        q = _clamp_q(q_base.copy())
        lam_s1 = 1e-2
        w_mid = 0.35
        reg_diag = np.full(n_dof, 1e-4, dtype=np.float64)
        if plane_mode == "sagittal":
            for bi_k in range(1, n_bones):
                for ax_k in range(3):
                    if ax_k != axis_idx:
                        reg_diag[3 * bi_k + ax_k] = 8e-3

        cur_eff, cur_mid = _fast_chain(q)
        cur_obj = float(np.sum((cur_eff - clamped_tgt) ** 2)) + (w_mid * w_mid) * float(
            np.sum((cur_mid - p_mid_ideal) ** 2)
        ) + float(np.dot(reg_diag, (q - q_base) ** 2))

        s1_iters = min(18, max_iters)
        for it in range(1, s1_iters + 1):
            iters_used = it
            if float(np.linalg.norm(cur_eff - clamped_tgt)) <= tol and it >= 6:
                break
            jac6 = np.zeros((6, n_dof), dtype=np.float64)
            for j in range(n_dof):
                q_plus = q.copy()
                q_plus[j] += eps
                eff_p, mid_p = _fast_chain(q_plus)
                jac6[0:3, j] = (eff_p - cur_eff) / eps
                jac6[3:6, j] = w_mid * (mid_p - cur_mid) / eps
            res6 = np.concatenate((clamped_tgt - cur_eff, w_mid * (p_mid_ideal - cur_mid)))
            lhs = jac6.T @ jac6 + np.diag(reg_diag + lam_s1)
            rhs = jac6.T @ res6 - reg_diag * (q - q_base)
            dq = np.linalg.solve(lhs, rhs)
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > 25.0:
                dq *= 25.0 / dq_norm
            q_cand = _clamp_q(q + dq)
            c_eff, c_mid = _fast_chain(q_cand)
            c_obj = float(np.sum((c_eff - clamped_tgt) ** 2)) + (w_mid * w_mid) * float(
                np.sum((c_mid - p_mid_ideal) ** 2)
            ) + float(np.dot(reg_diag, (q_cand - q_base) ** 2))
            if c_obj < cur_obj:
                q, cur_eff, cur_mid, cur_obj = q_cand, c_eff, c_mid, c_obj
                lam_s1 = max(1e-6, lam_s1 * 0.5)
            else:
                lam_s1 = min(1e2, lam_s1 * 4.0)

        # Stage 2: Exact primary end-effector convergence with nullspace pole retention
        lam_s2 = 1e-3
        cur_err = float(np.linalg.norm(cur_eff - clamped_tgt))
        cur_sec = _secondary_cost(q, cur_mid)
        for it2 in range(1, max(28, max_iters - s1_iters) + 1):
            if cur_err <= tol:
                break
            iters_used += 1
            jac = np.zeros((3, n_dof), dtype=np.float64)
            use_null = it2 <= 6
            grad_sec = np.zeros(n_dof, dtype=np.float64)
            for j in range(n_dof):
                q_plus = q.copy()
                q_plus[j] += eps
                eff_p, mid_p = _fast_chain(q_plus)
                jac[:, j] = (eff_p - cur_eff) / eps
                if use_null:
                    grad_sec[j] = (_secondary_cost(q_plus, mid_p) - cur_sec) / eps
            residual = clamped_tgt - cur_eff
            jj_t = jac @ jac.T + lam_s2 * np.eye(3, dtype=np.float64)
            j_pinv = jac.T @ np.linalg.inv(jj_t)
            dq_prim = j_pinv @ residual
            if use_null:
                null_proj = np.eye(n_dof, dtype=np.float64) - j_pinv @ jac
                dq_null = null_proj @ (-2.0 * grad_sec)
                dq_null_n = float(np.linalg.norm(dq_null))
                if dq_null_n > 3.0:
                    dq_null *= 3.0 / dq_null_n
                dq = dq_prim + dq_null
            else:
                dq = dq_prim
            dq_n = float(np.linalg.norm(dq))
            if dq_n > 20.0:
                dq *= 20.0 / dq_n
            q_cand = _clamp_q(q + dq)
            c_eff, c_mid = _fast_chain(q_cand)
            c_err = float(np.linalg.norm(c_eff - clamped_tgt))
            if c_err < cur_err:
                q, cur_eff, cur_mid, cur_err = q_cand, c_eff, c_mid, c_err
                cur_sec = _secondary_cost(q, cur_mid)
                lam_s2 = max(1e-6, lam_s2 * 0.5)
            else:
                q_cand_p = _clamp_q(q + dq_prim * min(1.0, 15.0 / max(1e-9, float(np.linalg.norm(dq_prim)))))
                p_eff, p_mid = _fast_chain(q_cand_p)
                p_err = float(np.linalg.norm(p_eff - clamped_tgt))
                if p_err < cur_err:
                    q, cur_eff, cur_mid, cur_err = q_cand_p, p_eff, p_mid, p_err
                    cur_sec = _secondary_cost(q, cur_mid)
                    lam_s2 = max(1e-6, lam_s2 * 0.7)
                else:
                    lam_s2 = min(1e2, lam_s2 * 4.0)

        best_q = q.copy()
        best_err = cur_err
        best_score = cur_err + 0.05 * cur_sec

    warmstarts = [] if (best_q is not None and best_err <= max(0.05, 0.08 * total_len)) else [1.0, -1.0]

    for sign_try in warmstarts:
        q = q_base.copy()
        for i in range(1, n_bones):
            q[3 * i + axis_idx] += sign_try * arch_deg
        q = _clamp_q(q)
        lam = 1e-2

        cur_eff, cur_mid = _fast_chain(q)
        cur_err = float(np.linalg.norm(cur_eff - clamped_tgt))
        cur_sec = _secondary_cost(q, cur_mid)

        for it in range(1, max_iters + 1):
            iters_used = it
            if cur_err <= tol and (not pole_active or it >= 12):
                break
            jac = np.zeros((3, n_dof), dtype=np.float64)
            grad_sec = np.zeros(n_dof, dtype=np.float64)
            for j in range(n_dof):
                q_plus = q.copy()
                q_plus[j] += eps
                eff_p, mid_p = _fast_chain(q_plus)
                jac[:, j] = (eff_p - cur_eff) / eps
                if pole_active:
                    sec_p = _secondary_cost(q_plus, mid_p)
                    grad_sec[j] = (sec_p - cur_sec) / eps
                else:
                    grad_sec[j] = 2e-6 * q[j]

            residual = clamped_tgt - cur_eff
            jj_t = jac @ jac.T + lam * np.eye(3, dtype=np.float64)
            j_pinv = jac.T @ np.linalg.inv(jj_t)
            dq_prim = j_pinv @ residual

            null_proj = np.eye(n_dof, dtype=np.float64) - j_pinv @ jac
            sec_gain = 18.0 if pole_active else 1.0
            dq_null = null_proj @ (-sec_gain * grad_sec)
            dq_null_norm = float(np.linalg.norm(dq_null))
            if dq_null_norm > 8.0:
                dq_null *= 8.0 / dq_null_norm

            dq = dq_prim + dq_null
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > 25.0:
                dq *= 25.0 / dq_norm

            q_cand = _clamp_q(q + dq)
            cand_eff, cand_mid = _fast_chain(q_cand)
            cand_err = float(np.linalg.norm(cand_eff - clamped_tgt))
            cand_sec = _secondary_cost(q_cand, cand_mid)

            improved = (cand_err < cur_err - 1e-9) or (
                cand_err <= max(tol, cur_err + 1e-6) and cand_sec < cur_sec - 1e-9
            )
            if improved:
                q = q_cand
                cur_eff = cand_eff
                cur_mid = cand_mid
                cur_err = cand_err
                cur_sec = cand_sec
                lam = max(1e-6, lam * 0.5)
            else:
                # Fallback pure primary step without nullspace perturbation
                q_cand_p = _clamp_q(q + dq_prim * min(1.0, 20.0 / max(1e-9, float(np.linalg.norm(dq_prim)))))
                p_eff, p_mid = _fast_chain(q_cand_p)
                p_err = float(np.linalg.norm(p_eff - clamped_tgt))
                if p_err < cur_err:
                    q = q_cand_p
                    cur_eff = p_eff
                    cur_mid = p_mid
                    cur_err = p_err
                    cur_sec = _secondary_cost(q, cur_mid)
                    lam = max(1e-6, lam * 0.7)
                else:
                    lam = min(1e2, lam * 4.0)

        combo_score = cur_err + (0.05 * cur_sec if pole_active else 1e-5 * cur_sec)
        if combo_score < best_score:
            best_score = combo_score
            best_err = cur_err
            best_q = q.copy()
        if best_err <= tol and not pole_active:
            break

    assert best_q is not None
    q_final = _clamp_q(best_q)
    final_poses = dict(poses)
    for i, b in enumerate(chain_bones):
        final_poses[b] = (float(q_final[3 * i]), float(q_final[3 * i + 1]), float(q_final[3 * i + 2]))
    if _zero_scene is not None and not validate:
        achieved_w, _ = _fast_chain(q_final)
    else:
        achieved_w, _ = _effector_world(program, final_poses, chain_bones[-1], eff_loc, validate=False)

    return NBoneIKResult(
        bones=chain_bones,
        pose_euler_deltas=final_poses,
        achieved_world=(float(achieved_w[0]), float(achieved_w[1]), float(achieved_w[2])),
        target_world=(float(tgt_w[0]), float(tgt_w[1]), float(tgt_w[2])),
        clamped_target_world=(float(clamped_tgt[0]), float(clamped_tgt[1]), float(clamped_tgt[2])),
        reachable=reachable,
        position_error=float(np.linalg.norm(achieved_w - clamped_tgt)),
        iterations=iters_used,
        pole_world=pole_tuple,
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
