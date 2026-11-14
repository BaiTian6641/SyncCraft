"""Deterministic starter audit with explicit coverage and contact exclusions."""

from __future__ import annotations

from itertools import combinations
from typing import Any

from synccraft.kinematics.se3_chain import Scene
from synccraft.kinematics.sat_potential import sat_obb


def _components(scene: Scene) -> list[list[str]]:
    graph = {box.name: set() for box in scene.boxes}
    for left, right in scene.adjacency:
        graph[left].add(right)
        graph[right].add(left)
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


def audit_scene(
    scene: Scene,
    target_profile: str | None = None,
    *,
    collision_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Audit non-exempt collisions, without claiming full asset acceptance.

    Only explicit attach-primary and place-base relations are excluded. Their
    measured penetrations remain visible. Disconnected declared-contact graph
    components are reported, not treated as physical gap or support proofs.
    """
    collisions: list[dict[str, Any]] = []
    declared_contacts: list[dict[str, Any]] = []
    errors: list[str] = []
    evaluated_pairs = 0
    for left, right in combinations(scene.boxes, 2):
        pair = tuple(sorted((left.name, right.name)))
        try:
            result = sat_obb(left, right, collision_tolerance=collision_tolerance)
        except ValueError as exc:
            errors.append(f"{left.name} / {right.name}: {exc}")
            continue
        entry = {"pair": list(pair), **result.to_dict()}
        if pair in scene.adjacency:
            entry["reason"] = scene.adjacency_reasons[pair]
            declared_contacts.append(entry)
        else:
            evaluated_pairs += 1
            if result.collision:
                collisions.append(entry)
    # A single invalid box still needs validation even when there are no pairs.
    if len(scene.boxes) == 1:
        try:
            sat_obb(scene.boxes[0], scene.boxes[0], collision_tolerance=collision_tolerance)
        except ValueError as exc:
            errors.append(f"{scene.boxes[0].name}: {exc}")
    if not scene.boxes:
        errors.append("Scene has no boxes")
    passed = not collisions and not errors
    return {
        "schema": "synccraft.audit/0.1",
        "scope": "starter_static_geometry",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "complete_for_release": False,
        "target_profile": target_profile or scene.meta.target_profile,
        "box_count": len(scene.boxes),
        "bone_count": len(scene.bones),
        "evaluated_non_exempt_pairs": evaluated_pairs,
        "collision_tolerance": collision_tolerance,
        "collision_pairs": collisions,
        "declared_contact_pairs": declared_contacts,
        "connected_components": _components(scene),
        "errors": errors,
        "coverage_gaps": [
            "No exact OBB intersection-volume measurement",
            "Declared contacts are excluded; deep penetration and gaps need separate validation",
            "Connectivity is a declared-edge graph, not a physical support or floating-part proof",
            "No Steve/Alex grip, display, UV, texture-seam, or dynamic foot-skating audit",
            "Engine import, schema conformance, and target-specific bounds require backend acceptance",
            "No differentiable gradients, constrained PGD, or learned-refiner validation",
        ],
    }
