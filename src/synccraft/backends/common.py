"""Small shared export helpers, with no renderer or model dependencies."""

from __future__ import annotations

import math
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

FACES = ("north", "south", "east", "west", "up", "down")


class ExportError(ValueError):
    """The requested export cannot faithfully represent the supplied scene."""


def stable_uuid(kind: str, name: str, asset_name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"synccraft:v3:{asset_name}:{kind}:{name}"))


def validate_scene_for_export(scene: Any) -> None:
    """Prevent audit/export disagreement for directly constructed Scene callers."""
    from synccraft.kinematics.se3_chain import euler_xyz_matrix

    if not scene.bones or not scene.boxes:
        raise ExportError("an export scene must contain bones and boxes")
    names: set[str] = set()
    bones: dict[str, Any] = {}

    def vec3(value: Any) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ExportError("scene vectors must contain three finite numbers")
        return result

    def claim(name: str) -> None:
        if not isinstance(name, str) or not name or name in names:
            raise ExportError("scene names must be nonempty and globally unique")
        names.add(name)

    for bone in scene.bones:
        claim(bone.name)
        translation = vec3(bone.local_translation)
        local_rotation = euler_xyz_matrix(vec3(bone.local_euler))
        if bone.parent_name is None:
            expected_pivot, expected_rotation = translation, local_rotation
        else:
            if bone.parent_name not in bones:
                raise ExportError("bones must be in parent-before-child order")
            parent = bones[bone.parent_name]
            expected_pivot = parent.pivot_world + parent.rotation @ translation
            expected_rotation = parent.rotation @ local_rotation
        if not np.allclose(vec3(bone.pivot_world), expected_pivot, atol=1e-8, rtol=0):
            raise ExportError(f"bone {bone.name}: local/world pivot disagreement")
        euler_zyx_degrees(bone.rotation)  # proper orthonormal rotation check
        if not np.allclose(bone.rotation, expected_rotation, atol=1e-8, rtol=0):
            raise ExportError(f"bone {bone.name}: local/world rotation disagreement")
        bones[bone.name] = bone
    size_step = float(getattr(getattr(scene, "meta", None), "size_step", 1.0))
    if size_step <= 0.0 or not math.isfinite(size_step):
        raise ExportError("scene meta.size_step must be finite and positive")

    for box in scene.boxes:
        claim(box.name)
        if box.bone_name not in bones:
            raise ExportError(f"box {box.name}: missing bone")
        size = vec3(box.size)
        plane_normal = getattr(box, "plane_normal", None)
        if plane_normal is not None:
            if plane_normal not in ("x", "y", "z"):
                raise ExportError(f"box {box.name}: invalid plane_normal {plane_normal!r}")
            p_idx = {"x": 0, "y": 1, "z": 2}[plane_normal]
            if abs(float(size[p_idx])) > 1e-9:
                raise ExportError(f"box {box.name}: plane_normal={plane_normal!r} requires zero size on that axis")
            other = [float(size[i]) for i in range(3) if i != p_idx]
            if any(v <= 0.0 for v in other):
                raise ExportError(f"box {box.name}: planar tangents must have positive size")
            if abs(size_step - 1.0) <= 1e-12:
                if any(not math.isclose(v, round(v), abs_tol=1e-9) for v in other):
                    raise ExportError(f"box {box.name}: size must contain positive integers when size_step == 1.0")
            else:
                if any(not math.isclose(v / size_step, round(v / size_step), abs_tol=1e-7) for v in other):
                    raise ExportError(f"box {box.name}: size must align to size_step={size_step}")
        elif abs(size_step - 1.0) <= 1e-12:
            if (size <= 0).any() or not np.equal(size, np.floor(size)).all():
                raise ExportError(f"box {box.name}: size must contain positive integers")
        else:
            if (size <= 0).any() or not np.allclose(size / size_step, np.rint(size / size_step), atol=1e-7, rtol=0):
                raise ExportError(f"box {box.name}: size must be positive multiples of size_step={size_step}")

        audit_profile = str(getattr(scene.meta, "audit_profile", "strict"))
        if not isinstance(box.inflate, (int, float)) or not math.isfinite(box.inflate):
            raise ExportError(f"box {box.name}: inflate must be finite and nonnegative")
        if box.inflate < 0:
            if audit_profile != "authored":
                raise ExportError(f"box {box.name}: inflate must be finite and nonnegative")
            non_planar = [float(size[i]) for i in range(3) if plane_normal is None or i != {"x": 0, "y": 1, "z": 2}[plane_normal]]
            if any(0.5 * v + float(box.inflate) <= 0.0 for v in non_planar):
                raise ExportError(f"box {box.name}: negative inflate collapses non-planar extents")
        bone = bones[box.bone_name]
        expected_center = bone.pivot_world + bone.rotation @ vec3(box.local_center)
        if not np.allclose(vec3(box.center), expected_center, atol=1e-8, rtol=0):
            raise ExportError(f"box {box.name}: local/world center disagreement")
        euler_zyx_degrees(box.rotation)
        raw_local_euler = getattr(box, "local_euler", None)
        if raw_local_euler is not None and np.any(np.abs(np.asarray(raw_local_euler, dtype=np.float64)) > 1e-9):
            expected_box_rot = bone.rotation @ euler_xyz_matrix(vec3(raw_local_euler))
            if not np.allclose(box.rotation, expected_box_rot, atol=1e-8, rtol=0):
                raise ExportError(f"box {box.name}: rotation disagrees with bone @ local_euler")
        elif not np.allclose(box.rotation, bone.rotation, atol=1e-8, rtol=0):
            raise ExportError(f"box {box.name}: rotation disagrees with its bone")
        if any(face not in FACES for face in box.cullface):
            raise ExportError(f"box {box.name}: invalid cullface")


def reflected_cullfaces(box: Any) -> tuple[str, ...]:
    """Mirror authored X face semantics along with the reflected geometry."""
    exchange = {"east": "west", "west": "east"}
    return tuple(exchange.get(face, face) if box.mirrored else face for face in box.cullface)


def vector(values: Any) -> list[float]:
    numbers = np.asarray(values, dtype=np.float64)
    if numbers.ndim != 1 or not np.isfinite(numbers).all():
        raise ExportError("export vectors must contain finite numbers")
    return [0.0 if abs(float(x)) < 5e-11 else round(float(x), 10) for x in numbers]


def validate_atlas(atlas: dict | None, scene: Any, *, embedded: bool) -> tuple[int, int]:
    if atlas is None:
        return 16, 16
    width, height = atlas.get("width"), atlas.get("height")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ExportError("atlas width and height must be positive integers")
    if embedded and not str(atlas.get("data_url", "")).startswith("data:image/png;base64,"):
        raise ExportError("bbmodel atlas requires an embedded PNG data_url")
    for box in scene.boxes:
        for face in FACES:
            try:
                uv = atlas["faces"][box.name][face]
            except (KeyError, TypeError) as exc:
                raise ExportError(f"atlas is missing {box.name}.{face}") from exc
            if len(uv) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in uv):
                raise ExportError(f"invalid UV rectangle for {box.name}.{face}")
            if any(x < 0 or x > bound for x, bound in zip(uv, (width, height, width, height))):
                raise ExportError(f"UV rectangle outside atlas for {box.name}.{face}")
    return width, height


def face_uv(atlas: dict | None, box_name: str, face: str, *, normalized: bool) -> list[float]:
    if atlas is None:
        return [0.0, 0.0, 16.0, 16.0]
    uv = np.asarray(atlas["faces"][box_name][face], dtype=np.float64)
    if normalized:
        uv = uv * np.array([16 / atlas["width"], 16 / atlas["height"]] * 2)
    return vector(uv)


def euler_zyx_degrees(rotation: Any) -> list[float]:
    """Recover x,y,z in Rz(z) @ Ry(y) @ Rx(x), including gimbal lock."""
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ExportError("rotation must be a finite 3x3 matrix")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0) or not math.isclose(float(np.linalg.det(r)), 1, abs_tol=1e-8):
        raise ExportError("rotation must be a proper orthonormal matrix")
    y = math.asin(float(np.clip(-r[2, 0], -1, 1)))
    if abs(math.cos(y)) > 1e-8:
        x = math.atan2(r[2, 1], r[2, 2])
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        z = 0.0
    return vector(np.rad2deg([x, y, z]))
