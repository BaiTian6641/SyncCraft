"""Deterministic static geometry audit with explicit P1-02 contact, M4 hinge bounds,
zero-thickness coplanar Z-fighting checks, and versioned ``strict`` | ``authored`` profiles."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synccraft.kinematics.sat_potential import _validate_box, sat_obb
from synccraft.kinematics.se3_chain import Scene, WorldBox

# Explicit M4 physical upper bounds for acute hinge corner swing under strict P1-02 audit.
MAX_HINGE_ANGLE_DEG: float = 60.0
MAX_HINGE_ALLOWANCE_UNITS: float = 2.5


def compute_delta_hinge(
    theta_rad: float,
    left_size: np.ndarray,
    right_size: np.ndarray,
    *,
    max_hinge_angle_deg: float = MAX_HINGE_ANGLE_DEG,
    max_hinge_allowance: float = MAX_HINGE_ALLOWANCE_UNITS,
) -> float:
    """Bounded acute hinge corner-swing allowance ``delta_hinge(theta)`` (M4 / P1-02).

    When two boxes share an explicit ``attach``/``joint``/``place`` hinge at relative SO(3)
    angle ``theta`` in ``(0, pi/3]`` (``<= 60 deg``), a flush joint centered on the face
    midpoint swings its cross-sectional corner into the partner along the hinge plane by
    ``0.5 * sqrt(3) * t_min * sin(theta)``, where ``t_min`` is the smaller median positive
    cross-section thickness of the two primitives.

    Per M4, ``delta_hinge(theta)`` is strictly bounded above by ``max_hinge_allowance``
    (``2.5`` model units) so thick torso blocks can never excuse deep burial, and vanishes
    identically at ``theta == 0`` (parallel boxes) and for ``theta > 60 deg`` (acute folding).
    Center burial deeper than ``max_inset_depth`` (``0.5u``) is always rejected separately.
    """
    max_rad = math.radians(max_hinge_angle_deg)
    if not (1e-5 < theta_rad <= max_rad):
        return 0.0
    left_pos = left_size[left_size > 1e-9]
    right_pos = right_size[right_size > 1e-9]
    if left_pos.size == 0 or right_pos.size == 0:
        return 0.0
    t_min = min(float(np.median(left_pos)), float(np.median(right_pos)))
    raw_swing = 0.5 * math.sqrt(3.0) * t_min * math.sin(theta_rad)
    return float(min(max_hinge_allowance, raw_swing))


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
    """Count box-bearing bones lacking an attach/joint edge to an ancestor box beyond the primary root component."""
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
            if reason not in ("attach", "joint"):
                continue
            a, b = pair
            if (a in ancestor_boxes and b in child_boxes) or (b in ancestor_boxes and a in child_boxes):
                has_attach_edge = True
                break
        if not has_attach_edge:
            independent_box_roots += 1
    return max(0, independent_box_roots - 1)


def _face_name_for_local_axis(axis_idx: int, sign_dir: int) -> str:
    if axis_idx == 0:
        return "east" if sign_dir > 0 else "west"
    if axis_idx == 1:
        return "up" if sign_dir > 0 else "down"
    return "south" if sign_dir > 0 else "north"


def _in_plane_overlap_positive(
    center_a: np.ndarray,
    rot_a: np.ndarray,
    ext_a: np.ndarray,
    tan_axes_a: tuple[int, int],
    center_b: np.ndarray,
    rot_b: np.ndarray,
    ext_b: np.ndarray,
    tan_axes_b: tuple[int, int],
    plane_normal_world: np.ndarray,
    area_tol: float,
) -> float:
    """Return minimum 2D in-plane separating-axis overlap (> area_tol iff positive 2D area overlap)."""
    candidate_dirs: list[np.ndarray] = []
    for ax_i in tan_axes_a:
        v = rot_a[:, ax_i]
        v_proj = v - float(np.dot(v, plane_normal_world)) * plane_normal_world
        nrm = float(np.linalg.norm(v_proj))
        if nrm > 1e-8:
            candidate_dirs.append(v_proj / nrm)
    for ax_j in tan_axes_b:
        v = rot_b[:, ax_j]
        v_proj = v - float(np.dot(v, plane_normal_world)) * plane_normal_world
        nrm = float(np.linalg.norm(v_proj))
        if nrm > 1e-8:
            candidate_dirs.append(v_proj / nrm)
    if not candidate_dirs:
        return 0.0

    delta = center_b - center_a
    min_ov = float("inf")
    for u in candidate_dirs:
        rad_a = sum(abs(float(np.dot(u, rot_a[:, i]))) * float(ext_a[i]) for i in tan_axes_a)
        rad_b = sum(abs(float(np.dot(u, rot_b[:, j]))) * float(ext_b[j]) for j in tan_axes_b)
        dist = abs(float(np.dot(u, delta)))
        ov = rad_a + rad_b - dist
        if ov <= area_tol:
            return 0.0
        min_ov = min(min_ov, ov)
    return min_ov if math.isfinite(min_ov) else 0.0


def _detect_coplanar_z_fighting(
    left: WorldBox,
    right: WorldBox,
    *,
    collision_tolerance: float,
) -> dict[str, Any] | None:
    """Detect exact coplanar Z-fighting between planar quads, plane-on-box faces, or duplicate 3D boxes."""
    z_tol = max(collision_tolerance * 10.0, 1e-5)
    area_tol = max(collision_tolerance * 100.0, 1e-4)
    axis_lookup = {"x": 0, "y": 1, "z": 2}

    lp = getattr(left, "plane_normal", None)
    rp = getattr(right, "plane_normal", None)

    # Case 1: Two planar quads lying on the exact same world plane with positive 2D overlap
    if lp in axis_lookup and rp in axis_lookup:
        nl_idx = axis_lookup[lp]
        nr_idx = axis_lookup[rp]
        n_left = left.rotation[:, nl_idx]
        n_right = right.rotation[:, nr_idx]
        if abs(float(np.dot(n_left, n_right))) >= 1.0 - 1e-6:
            if abs(float(left.half_extents[nl_idx]) - float(right.half_extents[nr_idx])) <= z_tol:
                normal_sep = abs(float(np.dot(right.center - left.center, n_left)))
                if normal_sep <= z_tol:
                    tan_l = tuple(i for i in range(3) if i != nl_idx)
                    tan_r = tuple(j for j in range(3) if j != nr_idx)
                    ov_2d = _in_plane_overlap_positive(
                        left.center,
                        left.rotation,
                        left.half_extents,
                        tan_l,  # type: ignore[arg-type]
                        right.center,
                        right.rotation,
                        right.half_extents,
                        tan_r,  # type: ignore[arg-type]
                        n_left,
                        area_tol,
                    )
                    if ov_2d > area_tol:
                        pair = sorted((left.name, right.name))
                        return {
                            "pair": pair,
                            "violation": "coplanar_z_fighting",
                            "kind": "plane_plane_coplanar",
                            "in_plane_min_overlap": ov_2d,
                        }
        return None

    # Case 2: One planar quad lying flat on a non-culled outer face of a 3D box
    if (lp in axis_lookup) != (rp in axis_lookup):
        plane_box = left if lp in axis_lookup else right
        solid_box = right if lp in axis_lookup else left
        np_idx = axis_lookup[getattr(plane_box, "plane_normal")]
        n_plane = plane_box.rotation[:, np_idx]
        h_plane_n = float(plane_box.half_extents[np_idx])
        tan_p = tuple(i for i in range(3) if i != np_idx)

        for s_ax in range(3):
            n_solid_ax = solid_box.rotation[:, s_ax]
            if abs(float(np.dot(n_plane, n_solid_ax))) < 1.0 - 1e-6:
                continue
            tan_s = tuple(j for j in range(3) if j != s_ax)
            for sign_dir in (-1, 1):
                face_name = _face_name_for_local_axis(s_ax, sign_dir)
                if face_name in solid_box.cullface:
                    continue
                face_center = solid_box.center + float(sign_dir) * float(solid_box.half_extents[s_ax]) * n_solid_ax
                # Check if either rendered surface of plane_box coincides with face_center
                dist_center = abs(float(np.dot(plane_box.center - face_center, n_plane)))
                if abs(dist_center - h_plane_n) <= z_tol:
                    ov_2d = _in_plane_overlap_positive(
                        plane_box.center,
                        plane_box.rotation,
                        plane_box.half_extents,
                        tan_p,  # type: ignore[arg-type]
                        face_center,
                        solid_box.rotation,
                        solid_box.half_extents,
                        tan_s,  # type: ignore[arg-type]
                        n_plane,
                        area_tol,
                    )
                    if ov_2d > area_tol:
                        pair = sorted((left.name, right.name))
                        return {
                            "pair": pair,
                            "violation": "coplanar_z_fighting",
                            "kind": "plane_on_box_face_coplanar",
                            "in_plane_min_overlap": ov_2d,
                        }
        return None

    # Case 3: Duplicate coincident 6-face-coplanar 3D boxes
    if np.linalg.norm(left.center - right.center) <= z_tol:
        # Match rotated axes up to permutation/sign
        gram = np.abs(left.rotation.T @ right.rotation)
        if np.all(np.max(gram, axis=1) >= 1.0 - 1e-6):
            perm = np.argmax(gram, axis=1)
            if len(set(perm.tolist())) == 3:
                matched_ext = right.half_extents[perm]
                if np.all(np.abs(left.half_extents - matched_ext) <= z_tol):
                    if set(left.cullface) != set(
                        ("north", "south", "east", "west", "up", "down")
                    ) and set(right.cullface) != set(("north", "south", "east", "west", "up", "down")):
                        pair = sorted((left.name, right.name))
                        return {
                            "pair": pair,
                            "violation": "coplanar_z_fighting",
                            "kind": "coincident_duplicate_boxes",
                            "in_plane_min_overlap": float(np.min(2.0 * left.half_extents)),
                        }
    return None


def _authored_bone_exemption_reason(
    left_bone: str,
    right_bone: str,
    bone_parent_map: dict[str, str | None],
) -> str | None:
    """Return explicit authored-rig joint exemption category, or None if bones are non-adjacent."""
    if left_bone not in bone_parent_map or right_bone not in bone_parent_map:
        return None
    if left_bone == right_bone:
        return "authored_same_bone"
    parent_l = bone_parent_map[left_bone]
    parent_r = bone_parent_map[right_bone]
    if parent_l == right_bone or parent_r == left_bone:
        return "authored_parent_child"
    if parent_l is not None and parent_l == parent_r:
        return "authored_sibling_joint"
    return None


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
    """Verify strict P1-02 physical contact bounds and M4 hinge cap on an explicit attach/place pair."""
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
    hinge_allowance = compute_delta_hinge(theta, left.size, right.size)
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
            "delta_hinge_allowance": hinge_allowance,
            "relative_angle_deg": theta_deg,
            "center_buried": center_buried,
        }
    return None


def audit_scene(
    scene: Scene,
    target_profile: str | None = None,
    *,
    audit_profile: str | None = None,
    collision_tolerance: float = 1e-7,
    enforce_physical_contact: bool = True,
    max_contact_gap: float = 1e-4,
    max_inset_depth: float = 0.5,
) -> dict[str, Any]:
    """Audit non-exempt OBB collisions, coplanar Z-fighting, and P1-02 contact contracts.

    Profiles:
    - ``"strict"`` (default): Enforces full P1-02 ``<= 0.5u`` inset + bounded M4
      ``delta_hinge(theta) <= 2.5u``, ``<= 1e-4`` contact gap, physical connectivity,
      exact coplanar Z-fighting rejection, and zero parent/sibling collision immunity.
    - ``"authored"``: Honors ``scene.meta.audit_profile == "authored"`` for imported/real
      authored rigs by exempting declared contacts, same-bone, parent-child, and sibling
      joint overlaps from SAT failure (reporting every exemption explicitly in
      ``authored_exempt_pairs``), while still enforcing finite/orthonormal OBB bounds,
      exact coplanar Z-fighting rejection, and non-adjacent bone SAT collision rejection.
    """
    resolved_profile = audit_profile or str(getattr(scene.meta, "audit_profile", "strict"))
    if resolved_profile not in ("strict", "authored"):
        raise ValueError(f"Unknown audit_profile {resolved_profile!r}; expected 'strict' or 'authored'")
    is_authored = resolved_profile == "authored"

    collisions: list[dict[str, Any]] = []
    declared_contacts: list[dict[str, Any]] = []
    authored_exempt_pairs: list[dict[str, Any]] = []
    contact_violations: list[dict[str, Any]] = []
    z_fighting_pairs: list[dict[str, Any]] = []
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

    bone_parent_map: dict[str, str | None] = {bone.name: bone.parent_name for bone in scene.bones}

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
    broadphase_margin = math.sqrt(3.0) * max(collision_tolerance, max_contact_gap, 1e-4) + 1e-6

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

            z_fight = _detect_coplanar_z_fighting(left, right, collision_tolerance=collision_tolerance)

            entry = {"pair": list(pair), **result.to_dict()}
            if is_declared:
                reason = scene.adjacency_reasons.get(pair, "declared")
                entry["reason"] = reason
                declared_contacts.append(entry)
                if is_authored:
                    physical_graph[left.name].add(right.name)
                    physical_graph[right.name].add(left.name)
                    if result.collision and z_fight is None:
                        authored_exempt_pairs.append(
                            {**entry, "exemption_reason": f"declared_{reason}"}
                        )
                else:
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
                    elif z_fight is None:
                        physical_graph[left.name].add(right.name)
                        physical_graph[right.name].add(left.name)
                if z_fight is not None:
                    z_fighting_pairs.append(z_fight)
                    contact_violations.append(z_fight)
            else:
                if z_fight is not None:
                    z_fighting_pairs.append(z_fight)
                    contact_violations.append(z_fight)
                if result.collision:
                    auth_reason = (
                        _authored_bone_exemption_reason(left.bone_name, right.bone_name, bone_parent_map)
                        if is_authored
                        else None
                    )
                    if auth_reason is not None and z_fight is None:
                        authored_exempt_pairs.append({**entry, "exemption_reason": auth_reason})
                        physical_graph[left.name].add(right.name)
                        physical_graph[right.name].add(left.name)
                    else:
                        collisions.append(entry)
                elif result.raw_min_overlap >= -max_contact_gap and z_fight is None:
                    physical_graph[left.name].add(right.name)
                    physical_graph[right.name].add(left.name)

    physical_components = _graph_components(physical_graph) if n_boxes > 0 else []
    authorized_detached = _count_authorized_detached_bones(scene)
    max_allowed_components = (1 + authorized_detached) if n_boxes > 0 else 0
    if not is_authored and n_boxes > 0 and not errors and len(physical_components) > max_allowed_components:
        contact_violations.append(
            {
                "violation": "disconnected_physical_components",
                "component_count": len(physical_components),
                "max_allowed_components": max_allowed_components,
                "authorized_detached_bones": authorized_detached,
                "components": physical_components,
            }
        )

    if enforce_physical_contact:
        contact_ok = (not z_fighting_pairs) if is_authored else (not contact_violations and not z_fighting_pairs)
    else:
        contact_ok = True
    passed = (
        not collisions
        and contact_ok
        and not errors
    )
    return {
        "schema": "synccraft.audit/0.3",
        "scope": "starter_static_geometry",
        "audit_profile": resolved_profile,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "complete_for_release": False,
        "target_profile": target_profile or scene.meta.target_profile,
        "box_count": n_boxes,
        "bone_count": len(scene.bones),
        "evaluated_non_exempt_pairs": evaluated_pairs,
        "collision_tolerance": collision_tolerance,
        "contact_policy": {
            "version": "p1-02/1.1",
            "audit_profile": resolved_profile,
            "enforced": enforce_physical_contact and not is_authored,
            "max_contact_gap": max_contact_gap,
            "max_inset_depth": max_inset_depth,
            "max_hinge_angle_deg": MAX_HINGE_ANGLE_DEG,
            "max_hinge_allowance": MAX_HINGE_ALLOWANCE_UNITS,
            "authorized_detached_bones": authorized_detached,
        },
        "collision_pairs": collisions,
        "declared_contact_pairs": declared_contacts,
        "authored_exempt_pairs": authored_exempt_pairs,
        "z_fighting_pairs": z_fighting_pairs,
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
