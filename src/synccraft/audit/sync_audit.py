"""Deterministic static geometry audit with explicit P1-02 contact & connectivity contracts."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synccraft.kinematics.sat_potential import _validate_box, sat_obb
from synccraft.kinematics.se3_chain import Scene, WorldBox


def _graph_components(graph: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    components: list[list[str]] = []
    for initial in sorted(graph):
        if initial in seen:
            continue
        pending, component = [initial], []
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            pending.extend(sorted(graph[current] - seen))
        components.append(sorted(component))
    return components


def _components(scene: Scene) -> list[list[str]]:
    graph = {box.name: set() for box in scene.boxes}
    for left, right in scene.adjacency:
        if left in graph and right in graph:
            graph[left].add(right)
            graph[right].add(left)
    return _graph_components(graph)


def _count_authorized_detached_bones(scene: Scene) -> int:
    """Count box-bearing bones lacking an attach edge to an ancestor box beyond the primary root component."""
    if not scene.bones or not scene.boxes:
        return 0
    bone_by_name = {bone.name: bone for bone in scene.bones}
    boxes_by_bone: dict[str, set[str]] = {bone.name: set() for bone in scene.bones}
    for box in scene.boxes:
        if box.bone_name in boxes_by_bone:
            boxes_by_bone[box.bone_name].add(box.name)

    independent_box_roots = 0
    for bone in scene.bones:
        child_boxes = boxes_by_bone.get(bone.name, set())
        if not child_boxes:
            continue
        ancestor_boxes: set[str] = set()
        curr_parent = bone.parent_name
        while curr_parent is not None and curr_parent in bone_by_name:
            ancestor_boxes |= boxes_by_bone.get(curr_parent, set())
            curr_parent = bone_by_name[curr_parent].parent_name
        if not ancestor_boxes:
            independent_box_roots += 1
            continue
        has_attach_edge = False
        for pair, reason in scene.adjacency_reasons.items():
            if reason != "attach":
                continue
            a, b = pair
            if (a in ancestor_boxes and b in child_boxes) or (b in ancestor_boxes and a in child_boxes):
                has_attach_edge = True
                break
        if not has_attach_edge:
            independent_box_roots += 1
    return max(0, independent_box_roots - 1)


def _evaluate_declared_contact(
    left: WorldBox,
    right: WorldBox,
    pair: tuple[str, str],
    reason: str,
    raw_min_overlap: float,
    *,
    collision_tolerance: float,
    max_contact_gap: float,
    max_inset_depth: float,
) -> dict[str, Any] | None:
    """Verify P1-02 physical contact bounds on an explicit attach/place pair."""
    if raw_min_overlap < -max_contact_gap:
        return {
            "pair": list(pair),
            "reason": reason,
            "violation": "separated_declared_contact",
            "raw_min_overlap": raw_min_overlap,
            "measured_gap": -raw_min_overlap,
            "max_allowed_gap": max_contact_gap,
        }

    rel_rot = left.rotation.T @ right.rotation
    cos_theta = float(np.clip((np.trace(rel_rot) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cos_theta)
    theta_deg = math.degrees(theta)

    inflate_allowance = float(left.inflate + right.inflate)
    hinge_allowance = 0.0
    # Hinge corner swing is permitted only for non-folding acute/moderate joint angles (<= 60 deg).
    if 1e-5 < theta <= math.radians(60.0):
        t_min = min(float(np.median(left.size)), float(np.median(right.size)))
        hinge_allowance = 0.5 * math.sqrt(3.0) * t_min * math.sin(theta)

    max_allowed_penetration = max(max_inset_depth, hinge_allowance) + inflate_allowance + collision_tolerance

    # Reject any joint where either box's center is buried deeper than max_inset_depth inside the partner OBB.
    delta_in_left = np.abs(left.rotation.T @ (right.center - left.center))
    delta_in_right = np.abs(right.rotation.T @ (left.center - right.center))
    right_center_buried = bool(np.all(left.size * 0.5 - delta_in_left > max_inset_depth + collision_tolerance))
    left_center_buried = bool(np.all(right.size * 0.5 - delta_in_right > max_inset_depth + collision_tolerance))
    center_buried = right_center_buried or left_center_buried

    if raw_min_overlap > max_allowed_penetration or center_buried:
        return {
            "pair": list(pair),
            "reason": reason,
            "violation": "excessive_contact_penetration",
            "raw_min_overlap": raw_min_overlap,
            "max_allowed_penetration": max_allowed_penetration,
            "relative_angle_deg": theta_deg,
            "center_buried": center_buried,
        }
    return None


def audit_scene(
    scene: Scene,
    target_profile: str | None = None,
    *,
    collision_tolerance: float = 1e-7,
    enforce_physical_contact: bool = True,
    max_contact_gap: float = 1e-4,
    max_inset_depth: float = 0.5,
) -> dict[str, Any]:
    """Audit non-exempt OBB collisions plus P1-02 physical contact & connectivity contracts.

    Uses an exact world-AABB broadphase prefilter (mathematically lossless under
    isotropic sqrt(3) support expansion) before exact 15-axis OBB SAT evaluation.
    """
    collisions: list[dict[str, Any]] = []
    declared_contacts: list[dict[str, Any]] = []
    contact_violations: list[dict[str, Any]] = []
    errors: list[str] = []
    evaluated_pairs = 0

    if (
        not np.isfinite((collision_tolerance, max_contact_gap, max_inset_depth)).all()
        or collision_tolerance < 0
        or max_contact_gap < 0
        or max_inset_depth < 0
    ):
        raise ValueError("Audit tolerances must be finite and nonnegative")

    boxes = scene.boxes
    n_boxes = len(boxes)
    if n_boxes == 0:
        errors.append("Scene has no boxes")

    valid_mask = [False] * n_boxes
    world_mins = np.zeros((n_boxes, 3), dtype=np.float64)
    world_maxs = np.zeros((n_boxes, 3), dtype=np.float64)

    for idx, box in enumerate(boxes):
        try:
            _validate_box(box)
            valid_mask[idx] = True
            world_half = np.abs(box.rotation) @ box.half_extents
            world_mins[idx] = box.center - world_half
            world_maxs[idx] = box.center + world_half
        except ValueError as exc:
            errors.append(f"{box.name}: {exc}")

    physical_graph: dict[str, set[str]] = {box.name: set() for box in boxes}
    # Exact isotropic OBB support margin along world axes so neither collisions nor
    # flush contacts within max_contact_gap can ever be skipped by broadphase.
    broadphase_margin = math.sqrt(3.0) * max(collision_tolerance, max_contact_gap) + 1e-6

    for i in range(n_boxes):
        if not valid_mask[i]:
            continue
        left = boxes[i]
        min_i = world_mins[i]
        max_i = world_maxs[i]
        for j in range(i + 1, n_boxes):
            if not valid_mask[j]:
                continue
            right = boxes[j]
            pair = (left.name, right.name) if left.name <= right.name else (right.name, left.name)
            is_declared = pair in scene.adjacency

            if not is_declared:
                evaluated_pairs += 1
                min_j = world_mins[j]
                max_j = world_maxs[j]
                if (
                    min_i[0] - max_j[0] > broadphase_margin
                    or min_j[0] - max_i[0] > broadphase_margin
                    or min_i[1] - max_j[1] > broadphase_margin
                    or min_j[1] - max_i[1] > broadphase_margin
                    or min_i[2] - max_j[2] > broadphase_margin
                    or min_j[2] - max_i[2] > broadphase_margin
                ):
                    continue

            try:
                result = sat_obb(left, right, collision_tolerance=collision_tolerance)
            except ValueError as exc:
                errors.append(f"{left.name} / {right.name}: {exc}")
                continue

            entry = {"pair": list(pair), **result.to_dict()}
            if is_declared:
                reason = scene.adjacency_reasons.get(pair, "declared")
                entry["reason"] = reason
                declared_contacts.append(entry)
                violation = _evaluate_declared_contact(
                    left,
                    right,
                    pair,
                    reason,
                    result.raw_min_overlap,
                    collision_tolerance=collision_tolerance,
                    max_contact_gap=max_contact_gap,
                    max_inset_depth=max_inset_depth,
                )
                if violation is not None:
                    contact_violations.append(violation)
                else:
                    physical_graph[left.name].add(right.name)
                    physical_graph[right.name].add(left.name)
            else:
                if result.collision:
                    collisions.append(entry)
                elif result.raw_min_overlap >= -max_contact_gap:
                    physical_graph[left.name].add(right.name)
                    physical_graph[right.name].add(left.name)

    physical_components = _graph_components(physical_graph) if n_boxes > 0 else []
    authorized_detached = _count_authorized_detached_bones(scene)
    max_allowed_components = (1 + authorized_detached) if n_boxes > 0 else 0
    if n_boxes > 0 and not errors and len(physical_components) > max_allowed_components:
        contact_violations.append(
            {
                "violation": "disconnected_physical_components",
                "component_count": len(physical_components),
                "max_allowed_components": max_allowed_components,
                "authorized_detached_bones": authorized_detached,
                "components": physical_components,
            }
        )

    passed = (
        not collisions
        and (not contact_violations if enforce_physical_contact else True)
        and not errors
    )
    return {
        "schema": "synccraft.audit/0.2",
        "scope": "starter_static_geometry",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "complete_for_release": False,
        "target_profile": target_profile or scene.meta.target_profile,
        "box_count": n_boxes,
        "bone_count": len(scene.bones),
        "evaluated_non_exempt_pairs": evaluated_pairs,
        "collision_tolerance": collision_tolerance,
        "contact_policy": {
            "version": "p1-02/1.0",
            "enforced": enforce_physical_contact,
            "max_contact_gap": max_contact_gap,
            "max_inset_depth": max_inset_depth,
            "authorized_detached_bones": authorized_detached,
        },
        "collision_pairs": collisions,
        "declared_contact_pairs": declared_contacts,
        "contact_violations": contact_violations,
        "connected_components": _components(scene),
        "physical_connected_components": physical_components,
        "errors": errors,
        "coverage_gaps": [
            "No exact OBB intersection-volume measurement",
            "No Steve/Alex grip, display, UV, texture-seam, or dynamic foot-skating audit",
            "Engine import, schema conformance, and target-specific bounds require backend acceptance",
            "No differentiable gradients, constrained PGD, or learned-refiner validation",
        ],
    }
