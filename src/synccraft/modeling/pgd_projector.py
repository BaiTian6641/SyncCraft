"""Constrained kinematic/contact PGD projector sharing exact SE(3) conventions (P1-05/P1-06).

Discrete structure (integer/grid box sizes, bone hierarchy, materials, UVs, and target
profile) stays strictly fixed during refinement. Degree/radian conversion occurs at named
boundaries, Torch SE(3) matrices are cross-checked against ``se3_chain.euler_xyz_matrix``,
and final acceptance always re-audits the serialized/rounded Block-DSL output via direct
15-axis SAT (never trusting smooth surrogate loss alone).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import Bone, Box, Locator, Program, validate_program
from synccraft.dsl.parser import format_program, parse_program
from synccraft.evaluation.budget import CostLedger
from synccraft.kinematics.sat_potential import sat_obb
from synccraft.kinematics.se3_chain import Scene, compile_geometry, euler_xyz_matrix

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore[assignment]


def degrees_to_radians_vec3(deg: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Explicit degree -> radian boundary conversion."""
    arr = np.asarray(deg, dtype=np.float64)
    return arr * (math.pi / 180.0)


def radians_to_degrees_vec3(rad: np.ndarray) -> tuple[float, float, float]:
    """Explicit radian -> rounded AST degree boundary conversion."""
    deg = np.asarray(rad, dtype=np.float64) * (180.0 / math.pi)
    return (round(float(deg[0]), 4), round(float(deg[1]), 4), round(float(deg[2]), 4))


def torch_euler_xyz_matrix(euler_radians: Any) -> Any:
    """Differentiable PyTorch SE(3) rotation ``Rz(z) @ Ry(y) @ Rx(x)`` matching NumPy."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed in the active environment")
    rx_ang, ry_ang, rz_ang = euler_radians[0], euler_radians[1], euler_radians[2]
    cx, sx = torch.cos(rx_ang), torch.sin(rx_ang)
    cy, sy = torch.cos(ry_ang), torch.sin(ry_ang)
    cz, sz = torch.cos(rz_ang), torch.sin(rz_ang)
    zero = torch.zeros_like(cx)
    one = torch.ones_like(cx)

    rx = torch.stack(
        (
            torch.stack((one, zero, zero)),
            torch.stack((zero, cx, -sx)),
            torch.stack((zero, sx, cx)),
        )
    )
    ry = torch.stack(
        (
            torch.stack((cy, zero, sy)),
            torch.stack((zero, one, zero)),
            torch.stack((-sy, zero, cy)),
        )
    )
    rz = torch.stack(
        (
            torch.stack((cz, -sz, zero)),
            torch.stack((sz, cz, zero)),
            torch.stack((zero, zero, one)),
        )
    )
    return rz @ ry @ rx


def verify_torch_se3_parity(*, seed: int = 2026) -> dict[str, Any]:
    """Cross-check Torch SE(3) rotation & autograd Jacobian against NumPy reference."""
    if torch is None:
        return {
            "torch_available": False,
            "max_matrix_error": None,
            "max_grad_fd_error": None,
            "passed": False,
        }

    rng = np.random.default_rng(seed)
    random_angles = [tuple(float(x) for x in rng.uniform(-90.0, 90.0, size=3)) for _ in range(4)]
    test_angles_deg = (
        (0.0, 0.0, 0.0),
        (22.5, -35.0, 45.0),
        (-60.0, 15.0, -80.0),
        (12.345, 54.321, -27.5),
        *random_angles,
    )
    max_mat_err = 0.0
    for ang_deg in test_angles_deg:
        ref_np = euler_xyz_matrix(ang_deg)
        rad_t = torch.tensor(degrees_to_radians_vec3(ang_deg), dtype=torch.float64)
        mat_t = torch_euler_xyz_matrix(rad_t).detach().cpu().numpy()
        max_mat_err = max(max_mat_err, float(np.max(np.abs(mat_t - ref_np))))

    # Finite-difference gradient check on rotated vector loss
    q0 = np.array([0.35, -0.42, 0.58], dtype=np.float64)
    p_local = np.array([3.5, -2.0, 1.25], dtype=np.float64)
    target_w = np.array([1.0, 2.5, -1.5], dtype=np.float64)

    q_t = torch.tensor(q0, dtype=torch.float64, requires_grad=True)
    p_t = torch.tensor(p_local, dtype=torch.float64)
    tgt_t = torch.tensor(target_w, dtype=torch.float64)
    pred_w = torch_euler_xyz_matrix(q_t) @ p_t
    loss = 0.5 * torch.sum((pred_w - tgt_t) ** 2)
    loss.backward()
    assert q_t.grad is not None
    grad_auto = q_t.grad.detach().cpu().numpy()

    eps = 1e-7
    grad_fd = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        dq = np.zeros(3, dtype=np.float64)
        dq[axis] = eps
        lp = 0.5 * float(np.sum((euler_xyz_matrix(np.rad2deg(q0 + dq)) @ p_local - target_w) ** 2))
        lm = 0.5 * float(np.sum((euler_xyz_matrix(np.rad2deg(q0 - dq)) @ p_local - target_w) ** 2))
        grad_fd[axis] = (lp - lm) / (2.0 * eps)

    max_grad_err = float(np.max(np.abs(grad_auto - grad_fd)))
    passed = bool(max_mat_err < 1e-7 and max_grad_err < 1e-5)
    return {
        "torch_available": True,
        "max_matrix_error": max_mat_err,
        "max_grad_fd_error": max_grad_err,
        "passed": passed,
    }


@dataclass(frozen=True)
class PGDProjectionResult(Mapping[str, Any]):
    """Result of constrained PGD pose/offset projection with serialized re-audit."""

    program: Program
    final_dsl: str
    scene: Scene
    audit_report: dict[str, Any]
    passed: bool
    serialized_reaudit_passed: bool
    steps_run: int
    initial_max_overlap: float
    final_max_overlap: float
    torch_fk_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program,
            "final_dsl": self.final_dsl,
            "scene": self.scene,
            "audit_report": self.audit_report,
            "passed": self.passed,
            "serialized_reaudit_passed": self.serialized_reaudit_passed,
            "steps_run": self.steps_run,
            "initial_max_overlap": round(self.initial_max_overlap, 6),
            "final_max_overlap": round(self.final_max_overlap, 6),
            "torch_fk_verified": self.torch_fk_verified,
        }

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        return data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _worst_violation_score(audit_report: dict[str, Any]) -> float:
    """Scalar penalty summarizing non-exempt SAT overlaps and P1-02 contact violations."""
    score = 0.0
    for col in audit_report.get("collision_pairs", ()):
        score += max(0.0, float(col.get("raw_min_overlap", 0.0))) + 0.05
    for cv in audit_report.get("contact_violations", ()):
        if "measured_gap" in cv:
            score += max(0.0, float(cv["measured_gap"])) + 0.05
        elif "raw_min_overlap" in cv:
            score += max(0.0, float(cv["raw_min_overlap"])) + 0.05
        else:
            score += 0.5
    if audit_report.get("errors"):
        score += 10.0 * len(audit_report["errors"])
    return score


def _max_collision_overlap(audit_report: dict[str, Any]) -> float:
    overlaps = [float(c.get("raw_min_overlap", 0.0)) for c in audit_report.get("collision_pairs", ())]
    for cv in audit_report.get("contact_violations", ()):
        if cv.get("violation") in (
            "excessive_contact_penetration",
            "excessive_sculpted_overlap",
            "swallowed_redundant_box",
        ):
            overlaps.append(float(cv.get("raw_min_overlap", 0.0)))
    return max(overlaps, default=0.0)


def _clamp_euler_rad_for_bone(
    rad: np.ndarray,
    bone: Bone,
    *,
    target_profile: str,
) -> np.ndarray:
    deg = np.rad2deg(rad)
    if bone.joint_limits is not None:
        for ax in range(3):
            lo, hi = float(bone.joint_limits[ax][0]), float(bone.joint_limits[ax][1])
            deg[ax] = float(np.clip(deg[ax], lo, hi))
    else:
        deg = np.clip(deg, -85.0, 85.0)
    if target_profile == "java_legacy":
        dom = int(np.argmax(np.abs(deg)))
        masked = np.zeros(3, dtype=np.float64)
        # java_legacy allows single-axis multiples of 22.5 in [-45, 45]
        snapped = round(float(deg[dom]) / 22.5) * 22.5
        masked[dom] = float(np.clip(snapped, -45.0, 45.0))
        deg = masked
    return degrees_to_radians_vec3(deg)


def _oriented_separation_direction_world(w_fixed: Any, w_moving: Any) -> tuple[np.ndarray, float]:
    """Return unit minimum-overlap SAT axis pointing from w_fixed toward w_moving and raw_min_overlap."""
    sat_res = sat_obb(w_fixed, w_moving)
    axes: list[np.ndarray] = [w_fixed.rotation[:, i] for i in range(3)]
    axes.extend(w_moving.rotation[:, i] for i in range(3))
    for i in range(3):
        for j in range(3):
            candidate = np.cross(w_fixed.rotation[:, i], w_moving.rotation[:, j])
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-10:
                axes.append(candidate / norm)
    axis_array = np.asarray(axes, dtype=np.float64)
    radii_fixed = np.abs(axis_array @ w_fixed.rotation) @ w_fixed.half_extents
    radii_moving = np.abs(axis_array @ w_moving.rotation) @ w_moving.half_extents
    delta = w_moving.center - w_fixed.center
    overlaps = radii_fixed + radii_moving - np.abs(axis_array @ delta)
    min_idx = int(np.argmin(overlaps))
    axis_w = axis_array[min_idx].copy()
    if float(np.dot(delta, axis_w)) < 0.0:
        axis_w = -axis_w
    return axis_w, float(sat_res.raw_min_overlap)


def _collect_ast_box_places(bone: Bone) -> dict[str, str | None]:
    """Map AST box name -> declared Place.base_box (or None for primary box)."""
    mapping: dict[str, str | None] = {}
    for idx, it in enumerate(bone.items):
        if isinstance(it, Box):
            mapping[it.name] = it.place.base_box if (idx > 0 and it.place is not None) else "__PRIMARY__"
        elif isinstance(it, Bone):
            mapping.update(_collect_ast_box_places(it))
    return mapping


def _single_projection_step(
    program: Program,
    scene: Scene,
    audit_report: dict[str, Any],
    *,
    step_gain: float,
) -> Program:
    """Perform one deterministic projected update over offsets and child bone rotations."""
    box_place_deltas: dict[str, np.ndarray] = {}
    bone_attach_deltas: dict[str, np.ndarray] = {}
    bone_euler_deltas_rad: dict[str, np.ndarray] = {}
    ast_place_base = _collect_ast_box_places(program.root)

    # 1. Project P1-02 declared contact violations (gaps and deep insets) back to flush contact
    for cv in audit_report.get("contact_violations", ()):
        pair = cv.get("pair")
        if not pair or len(pair) != 2:
            continue
        left_name, right_name = str(pair[0]), str(pair[1])
        if left_name not in scene.box_by_name or right_name not in scene.box_by_name:
            continue
        w_l = scene.box_by_name[left_name]
        w_r = scene.box_by_name[right_name]
        v_kind = cv.get("violation")
        reason = cv.get("reason")

        if reason in ("place", "declared_place", "sculpted_same_bone_miter", "declared_blend") and w_l.bone_name == w_r.bone_name:
            bone_rot = scene.bone_by_name[w_l.bone_name].rotation
            if ast_place_base.get(left_name) == right_name:
                moving_name, w_fix, w_mov = left_name, w_r, w_l
            elif ast_place_base.get(right_name) == left_name:
                moving_name, w_fix, w_mov = right_name, w_l, w_r
            elif ast_place_base.get(left_name) == "__PRIMARY__":
                moving_name, w_fix, w_mov = right_name, w_l, w_r
            else:
                moving_name, w_fix, w_mov = left_name, w_r, w_l
            dir_w, raw_ov = _oriented_separation_direction_world(w_fix, w_mov)
            dir_local = bone_rot.T @ dir_w
            if v_kind == "separated_declared_contact":
                shift = -step_gain * (-raw_ov) * dir_local
            else:
                shift = step_gain * max(0.0, raw_ov) * dir_local
            box_place_deltas[moving_name] = box_place_deltas.get(moving_name, np.zeros(3)) + shift

        elif reason in ("attach", "declared_attach", "sculpted_parent_child_joint", "declared_blend") and w_l.bone_name != w_r.bone_name:
            b_l = scene.bone_by_name[w_l.bone_name]
            b_r = scene.bone_by_name[w_r.bone_name]
            if b_r.parent_name == b_l.name:
                w_parent, w_child, child_bone_name, parent_rot = w_l, w_r, b_r.name, b_l.rotation
            elif b_l.parent_name == b_r.name:
                w_parent, w_child, child_bone_name, parent_rot = w_r, w_l, b_l.name, b_r.rotation
            else:
                continue
            dir_w, raw_ov = _oriented_separation_direction_world(w_parent, w_child)
            dir_parent = parent_rot.T @ dir_w
            if v_kind == "separated_declared_contact":
                shift = -step_gain * (-raw_ov) * dir_parent
            else:
                shift = step_gain * max(0.0, raw_ov) * dir_parent
            bone_attach_deltas[child_bone_name] = bone_attach_deltas.get(child_bone_name, np.zeros(3)) + shift

    # 2. Project non-exempt OBB SAT collisions via sibling Place offset or Torch SE(3) bone rotation
    for col in audit_report.get("collision_pairs", ()):
        pair = col.get("pair", ())
        if len(pair) != 2:
            continue
        left_name, right_name = str(pair[0]), str(pair[1])
        if left_name not in scene.box_by_name or right_name not in scene.box_by_name:
            continue
        w_l = scene.box_by_name[left_name]
        w_r = scene.box_by_name[right_name]
        if w_l.bone_name == w_r.bone_name:
            bone_rot = scene.bone_by_name[w_l.bone_name].rotation
            if ast_place_base.get(left_name) == "__PRIMARY__":
                moving_name, w_fix, w_mov = right_name, w_l, w_r
            elif ast_place_base.get(right_name) == "__PRIMARY__":
                moving_name, w_fix, w_mov = left_name, w_r, w_l
            else:
                moving_name, w_fix, w_mov = right_name, w_l, w_r
            dir_w, raw_ov = _oriented_separation_direction_world(w_fix, w_mov)
            if raw_ov <= 0.0:
                continue
            dir_local = bone_rot.T @ dir_w
            shift = step_gain * (raw_ov + 1e-3) * dir_local
            box_place_deltas[moving_name] = box_place_deltas.get(moving_name, np.zeros(3)) + shift
        else:
            dir_w, raw_ov = _oriented_separation_direction_world(w_l, w_r)
            if raw_ov <= 0.0:
                continue
            # Differentiable SE(3) rotation gradient on the deeper articulated bone
            b_l = scene.bone_by_name[w_l.bone_name]
            b_r = scene.bone_by_name[w_r.bone_name]
            mov_bone = b_r if b_r.parent_name is not None else b_l
            w_mov = w_r if mov_bone.name == b_r.name else w_l
            push_w = dir_w if mov_bone.name == b_r.name else -dir_w
            if mov_bone.parent_name is not None:
                parent_bone = scene.bone_by_name[mov_bone.parent_name]
                lever_local = mov_bone.rotation.T @ (w_mov.center - mov_bone.pivot_world)
                push_parent = parent_bone.rotation.T @ push_w
                if torch is not None and float(np.linalg.norm(lever_local)) > 1e-6:
                    q_cur = torch.tensor(
                        degrees_to_radians_vec3(tuple(float(x) for x in mov_bone.local_euler)),
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    lev_t = torch.tensor(lever_local, dtype=torch.float64)
                    push_t = torch.tensor(push_parent, dtype=torch.float64)
                    disp_t = torch_euler_xyz_matrix(q_cur) @ lev_t
                    # Maximize projection along push_parent -> minimize -dot(disp, push_parent)
                    surrogate = -torch.sum(disp_t * push_t)
                    surrogate.backward()
                    assert q_cur.grad is not None
                    grad_np = q_cur.grad.detach().cpu().numpy()
                    gnorm = float(np.linalg.norm(grad_np))
                    if gnorm > 1e-8:
                        step_rad = -min(0.22, step_gain * (raw_ov / max(1.0, float(np.linalg.norm(lever_local))))) * (
                            grad_np / gnorm
                        )
                        bone_euler_deltas_rad[mov_bone.name] = (
                            bone_euler_deltas_rad.get(mov_bone.name, np.zeros(3)) + step_rad
                        )
                    else:
                        # Exact zero-angle escape toward neutral rest pose
                        cur_rad = degrees_to_radians_vec3(tuple(float(x) for x in mov_bone.local_euler))
                        bone_euler_deltas_rad[mov_bone.name] = -0.5 * step_gain * cur_rad
                else:
                    cur_rad = degrees_to_radians_vec3(tuple(float(x) for x in mov_bone.local_euler))
                    bone_euler_deltas_rad[mov_bone.name] = -0.5 * step_gain * cur_rad

    def _update_bone(bone: Bone, is_root: bool) -> Bone:
        updated_items: list[Box | Locator | Bone] = []
        for idx, it in enumerate(bone.items):
            if isinstance(it, Box):
                if idx > 0 and it.place is not None:
                    delta_p = box_place_deltas.get(it.name)
                    if delta_p is not None and float(np.linalg.norm(delta_p)) > 1e-12:
                        new_off = np.asarray(it.place.offset, dtype=np.float64) + delta_p
                        new_off_tuple = (
                            round(float(new_off[0]), 5),
                            round(float(new_off[1]), 5),
                            round(float(new_off[2]), 5),
                        )
                        it = replace(it, place=replace(it.place, offset=new_off_tuple))
                updated_items.append(it)
            elif isinstance(it, Bone):
                updated_items.append(_update_bone(it, is_root=False))
            else:
                updated_items.append(it)

        if is_root:
            return replace(bone, items=updated_items)

        new_att = bone.attach
        if new_att is not None and not new_att.detached and bone.name in bone_attach_deltas:
            off_np = np.asarray(new_att.offset, dtype=np.float64) + bone_attach_deltas[bone.name]
            off_np = np.clip(off_np, -0.5, 0.5)
            new_att = replace(
                new_att,
                offset=(round(float(off_np[0]), 5), round(float(off_np[1]), 5), round(float(off_np[2]), 5)),
            )

        new_euler = bone.rest_euler
        if bone.name in bone_euler_deltas_rad:
            cur_rad = degrees_to_radians_vec3(bone.rest_euler) + bone_euler_deltas_rad[bone.name]
            clamped_rad = _clamp_euler_rad_for_bone(cur_rad, bone, target_profile=program.meta.target_profile)
            new_euler = radians_to_degrees_vec3(clamped_rad)

        return replace(bone, rest_euler=new_euler, attach=new_att, items=updated_items)

    return validate_program(replace(program, root=_update_bone(program.root, is_root=True)))


def _relax_child_poses_and_offsets(program: Program, scale_euler: float) -> Program:
    """Fallback damped pose/offset projection toward feasible neutral joint frame."""

    def _walk(bone: Bone, is_root: bool) -> Bone:
        children = [_walk(it, False) if isinstance(it, Bone) else it for it in bone.items]
        if is_root:
            return replace(bone, items=children)
        att = bone.attach
        if att is not None and not att.detached:
            off = (
                round(float(att.offset[0]) * scale_euler, 5),
                round(float(att.offset[1]) * scale_euler, 5),
                round(float(att.offset[2]) * scale_euler, 5),
            )
            att = replace(att, offset=off)
        rad = degrees_to_radians_vec3(bone.rest_euler) * scale_euler
        clamped = _clamp_euler_rad_for_bone(rad, bone, target_profile=program.meta.target_profile)
        return replace(bone, rest_euler=radians_to_degrees_vec3(clamped), attach=att, items=children)

    return validate_program(replace(program, root=_walk(program.root, True)))


def project_program(
    program: Program,
    *,
    max_steps: int = 20,
    step_size: float = 0.65,
    enforce_physical_contact: bool = True,
    cost_ledger: CostLedger | None = None,
) -> PGDProjectionResult:
    """Run bounded constrained PGD over poses/offsets with mandatory post-serialization re-audit."""
    if not isinstance(program, Program):
        raise TypeError("project_program requires a validated Block-DSL Program instance")
    if max_steps < 1 or max_steps > 200:
        raise ValueError("max_steps must be in [1, 200]")
    if not (0.0 < step_size <= 1.0):
        raise ValueError("step_size must be in (0.0, 1.0]")

    parity = verify_torch_se3_parity()
    torch_ok = bool(parity["passed"])

    cur_prog = validate_program(program)
    use_contact = bool(
        enforce_physical_contact and getattr(cur_prog.meta, "audit_profile", "strict") != "authored"
    )
    init_scene = compile_geometry(cur_prog)
    init_audit = audit_scene(init_scene, enforce_physical_contact=use_contact)
    init_max_ov = _max_collision_overlap(init_audit)

    best_prog = cur_prog
    best_score = _worst_violation_score(init_audit)
    steps_run = 0

    for step_idx in range(1, max_steps + 1):
        cur_scene = compile_geometry(cur_prog)
        cur_audit = audit_scene(cur_scene, enforce_physical_contact=use_contact)
        cur_score = _worst_violation_score(cur_audit)
        if cur_score < best_score:
            best_score = cur_score
            best_prog = cur_prog

        if cur_audit["passed"]:
            # Verify optimization survives serialization & rounding before stopping early
            ser_dsl = format_program(cur_prog)
            ser_prog = parse_program(ser_dsl)
            ser_scene = compile_geometry(ser_prog)
            ser_audit = audit_scene(ser_scene, enforce_physical_contact=use_contact)
            if ser_audit["passed"]:
                return PGDProjectionResult(
                    program=ser_prog,
                    final_dsl=ser_dsl,
                    scene=ser_scene,
                    audit_report=ser_audit,
                    passed=True,
                    serialized_reaudit_passed=True,
                    steps_run=steps_run,
                    initial_max_overlap=init_max_ov,
                    final_max_overlap=_max_collision_overlap(ser_audit),
                    torch_fk_verified=torch_ok,
                )

        if cost_ledger is not None:
            cost_ledger.record("projection_steps", 1)
        steps_run += 1

        gain = float(np.clip(step_size * (1.0 if step_idx <= 6 else 0.85), 0.15, 1.0))
        cand_prog = _single_projection_step(cur_prog, cur_scene, cur_audit, step_gain=gain)
        cand_scene = compile_geometry(cand_prog)
        cand_audit = audit_scene(cand_scene, enforce_physical_contact=use_contact)
        cand_score = _worst_violation_score(cand_audit)

        if cand_score + 1e-9 < cur_score or cand_audit["passed"]:
            cur_prog = cand_prog
            if cand_score < best_score:
                best_score = cand_score
                best_prog = cand_prog
        else:
            # Try damped pose/offset relaxation schedule when gradient step stalls in nonconvex hinge pocket
            schedule_factor = max(0.0, 1.0 - step_idx / max(2, max_steps // 2))
            relaxed = _relax_child_poses_and_offsets(cur_prog, scale_euler=schedule_factor)
            rel_scene = compile_geometry(relaxed)
            rel_audit = audit_scene(rel_scene, enforce_physical_contact=use_contact)
            rel_score = _worst_violation_score(rel_audit)
            if rel_score <= cur_score or rel_audit["passed"]:
                cur_prog = relaxed
                if rel_score < best_score:
                    best_score = rel_score
                    best_prog = relaxed

    # Final mandatory post-serialization / rounding re-audit (AGENTS.md)
    final_dsl = format_program(best_prog)
    final_prog = parse_program(final_dsl)
    final_scene = compile_geometry(final_prog)
    final_audit = audit_scene(final_scene, enforce_physical_contact=use_contact)
    passed = bool(final_audit["passed"])
    return PGDProjectionResult(
        program=final_prog,
        final_dsl=final_dsl,
        scene=final_scene,
        audit_report=final_audit,
        passed=passed,
        serialized_reaudit_passed=passed,
        steps_run=steps_run,
        initial_max_overlap=init_max_ov,
        final_max_overlap=_max_collision_overlap(final_audit),
        torch_fk_verified=torch_ok,
    )
