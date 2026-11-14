"""Static Java block/item model JSON; no resource-pack or item dispatch wrapper."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .common import FACES, ExportError, euler_zyx_degrees, face_uv, reflected_cullfaces, validate_atlas, validate_scene_for_export, vector


def _legacy_rotation(rotation: np.ndarray) -> tuple[str, float] | None:
    """Validate the fully composed world rotation, never local Euler fields."""
    if np.allclose(rotation, np.eye(3), rtol=0, atol=1e-8):
        return None
    for index, axis in enumerate("xyz"):
        for angle in (-45.0, -22.5, 22.5, 45.0):
            rad = np.deg2rad(angle)
            c, s = np.cos(rad), np.sin(rad)
            if index == 0:
                candidate = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            elif index == 1:
                candidate = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                candidate = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            if np.allclose(rotation, candidate, rtol=0, atol=1e-8):
                return axis, angle
    raise ExportError("java_legacy requires a fully flattened single-axis rotation in {-45,-22.5,0,22.5,45}")


def export_java(
    scene: Any, *, texture_id: str = "synccraft:block/atlas", atlas: dict | None = None,
    profile: str | None = None,
) -> dict:
    """Flatten static OBBs into Java element JSON without rounding geometry.

    Modern uses Mojang's x/y/z element rotation fields (1.21.11). Legacy is the
    intentionally conservative v3 profile. Both enforce the plan's model-space
    [-16,32] envelope, including rotated corners, in addition to from/to bounds.
    """
    validate_scene_for_export(scene)
    target = profile or scene.meta.target_profile
    if target not in ("java_modern", "java_legacy"):
        raise ExportError("Java export requires explicit java_modern or java_legacy profile")
    if scene.meta.archetype == "entity":
        raise ExportError("vanilla Java block/item model JSON cannot encode custom entity geometry")
    if not re.fullmatch(r"[a-z0-9_.-]+:[a-z0-9_./-]+", texture_id):
        raise ExportError("texture_id must be a lowercase namespace:path resource location")
    validate_atlas(atlas, scene, embedded=False)
    elements = []
    for box in scene.boxes:
        half = np.asarray(box.size) / 2 + box.inflate
        from_ = np.asarray(box.center) - half
        to = np.asarray(box.center) + half
        all_coordinates = np.concatenate((from_, to, np.asarray(box.corners()).reshape(-1)))
        if not np.isfinite(all_coordinates).all() or (all_coordinates < -16 - 1e-8).any() or (all_coordinates > 32 + 1e-8).any():
            raise ExportError(f"box {box.name} exceeds the starter's [-16,32] model-space bounds")
        faces = {}
        for face in FACES:
            item = {"uv": face_uv(atlas, box.name, face, normalized=True), "texture": "#atlas"}
            if face in reflected_cullfaces(box):
                # Mojang specifies that face names stay unchanged by element rotation.
                item["cullface"] = face
            faces[face] = item
        element = {"from": vector(from_), "to": vector(to), "faces": faces}
        if target == "java_legacy":
            legacy = _legacy_rotation(np.asarray(box.rotation))
            if legacy is not None:
                axis, angle = legacy
                element["rotation"] = {"origin": vector(box.center), "axis": axis, "angle": angle, "rescale": False}
        elif not np.allclose(box.rotation, np.eye(3), rtol=0, atol=1e-10):
            x, y, z = euler_zyx_degrees(box.rotation)
            element["rotation"] = {"origin": vector(box.center), "x": x, "y": y, "z": z, "rescale": False}
        elements.append(element)
    return {"textures": {"atlas": texture_id, "particle": "#atlas"}, "elements": elements}
