"""NumPy reference implementation of Block-DSL v3 rigid transforms.

All public AST Euler angles are degrees. Column vectors use Rz @ Ry @ Rx.
The differentiable PyTorch implementation is a future milestone; these routines
are deliberately not described as differentiable or used to claim PGD results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from synccraft.dsl.ast_nodes import Bone, Box, Program, validate_program


def _vec(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("Expected a finite three-component vector")
    return array


def euler_xyz_matrix(euler_degrees: Any) -> np.ndarray:
    """Return a proper rotation, applying X then Y then Z to column vectors."""
    x, y, z = np.deg2rad(_vec(euler_degrees))
    cx, cy, cz = np.cos((x, y, z))
    sx, sy, sz = np.sin((x, y, z))
    rx = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=float)
    ry = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=float)
    rz = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


@dataclass(frozen=True)
class WorldBox:
    name: str
    bone_name: str
    center: np.ndarray
    rotation: np.ndarray
    size: np.ndarray
    inflate: float
    local_center: np.ndarray
    material: str
    cullface: tuple[str, ...] = ()
    mirrored: bool = False

    @property
    def half_extents(self) -> np.ndarray:
        return self.size / 2.0 + self.inflate

    def corners(self, *, include_inflate: bool = True) -> np.ndarray:
        extents = self.half_extents if include_inflate else self.size / 2.0
        signs = np.asarray(tuple(product((-1.0, 1.0), repeat=3)))
        return (signs * extents) @ self.rotation.T + self.center

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "bone_name": self.bone_name,
            "center": self.center.tolist(), "rotation": self.rotation.tolist(),
            "size": self.size.tolist(), "inflate": self.inflate,
            "local_center": self.local_center.tolist(), "material": self.material,
            "cullface": list(self.cullface), "mirrored": self.mirrored,
        }


@dataclass(frozen=True)
class WorldBone:
    name: str
    parent_name: str | None
    pivot_world: np.ndarray
    rotation: np.ndarray
    local_translation: np.ndarray
    local_euler: np.ndarray
    mirrored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "parent_name": self.parent_name,
            "pivot_world": self.pivot_world.tolist(), "rotation": self.rotation.tolist(),
            "local_translation": self.local_translation.tolist(),
            "local_euler": self.local_euler.tolist(), "mirrored": self.mirrored,
        }


@dataclass(frozen=True)
class Scene:
    boxes: tuple[WorldBox, ...]
    bones: tuple[WorldBone, ...]
    adjacency: frozenset[tuple[str, str]]
    meta: Any
    adjacency_reasons: dict[tuple[str, str], str] = field(default_factory=dict)

    @property
    def box_by_name(self) -> dict[str, WorldBox]:
        return {box.name: box for box in self.boxes}

    @property
    def bone_by_name(self) -> dict[str, WorldBone]:
        return {bone.name: bone for bone in self.bones}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "synccraft.scene/0.1",
            "coordinate_system": "right_handed_y_up_column_vectors",
            "angle_units": "degrees",
            "meta": {
                "archetype": self.meta.archetype,
                "target_profile": self.meta.target_profile,
                "texel_density": self.meta.texel_density,
            },
            "boxes": [box.to_dict() for box in self.boxes],
            "bones": [bone.to_dict() for bone in self.bones],
            "adjacency": [
                {"pair": list(pair), "reason": self.adjacency_reasons[pair]}
                for pair in sorted(self.adjacency)
            ],
        }


def compile_geometry(program: Program) -> Scene:
    """Expand validated boxes, explicit placements, child bones, and mirrors.

    ``mirror_x(right)`` creates a sibling reflected across the parent box's
    local-X center plane. Descendant names use ``right__original_name``.
    Nested mirror declarations are rejected by the starter parser and here too.
    Dimensions remain fixed integers; inflation only changes physical extents.
    """
    # Direct AST callers must get the same semantic guarantees as text callers.
    # In particular, NumPy conversion alone would accept booleans as dimensions.
    validate_program(program)
    boxes: list[WorldBox] = []
    bones: list[WorldBone] = []
    adjacency: dict[tuple[str, str], str] = {}
    names: set[str] = set()
    reflect = np.array((-1.0, 1.0, 1.0))
    angle_reflect = np.array((1.0, -1.0, -1.0))

    def reserve(name: str) -> None:
        if name in names:
            raise ValueError(f"Duplicate expanded geometry identifier: {name}")
        names.add(name)

    def edge(left: str, right: str, reason: str) -> None:
        adjacency[tuple(sorted((left, right)))] = reason

    def walk(
        bone: Bone,
        parent: WorldBone | None,
        parent_local_boxes: dict[str, WorldBox],
        *,
        mirrored: bool = False,
        prefix: str | None = None,
        renamed_root: str | None = None,
    ) -> None:
        name = renamed_root or (f"{prefix}__{bone.name}" if prefix else bone.name)
        reserve(name)
        sign = reflect if mirrored else np.ones(3)
        euler = _vec(bone.rest_euler) * (angle_reflect if mirrored else 1.0)
        local_rotation = euler_xyz_matrix(euler)
        if parent is None:
            if bone.attach is not None:
                raise ValueError("The root bone cannot have an attachment")
            translation = _vec(bone.pivot_local)
            world_pivot = translation.copy()
            world_rotation = local_rotation
            align = np.zeros(3)
            attachment_base = None
        else:
            if bone.attach is None:
                raise ValueError(f"Child bone {bone.name} requires attach(...)")
            spec = bone.attach
            try:
                attachment_base = parent_local_boxes[spec.parent_box]
            except KeyError as exc:
                raise ValueError(
                    f"{bone.name}: parent_box {spec.parent_box!r} is not an earlier box in the parent bone"
                ) from exc
            translation = (
                attachment_base.local_center
                + 0.5 * attachment_base.size * _vec(spec.anchor) * sign
                + _vec(spec.offset) * sign
            )
            world_pivot = parent.pivot_world + parent.rotation @ translation
            world_rotation = parent.rotation @ local_rotation
            align = _vec(spec.align) * sign
        compiled_bone = WorldBone(
            name, parent.name if parent else None, world_pivot, world_rotation,
            translation, euler, mirrored,
        )
        bones.append(compiled_bone)
        local_boxes: dict[str, WorldBox] = {}
        primary: WorldBox | None = None
        for item in bone.items:
            if isinstance(item, Box):
                box_name = f"{prefix}__{item.name}" if prefix else item.name
                reserve(box_name)
                size = _vec(item.size)
                if np.any(size <= 0) or np.any(size != np.floor(size)):
                    raise ValueError(f"{item.name}: size must contain positive integers")
                if not np.isfinite(item.inflate) or np.any(size / 2 + item.inflate <= 0):
                    raise ValueError(f"{item.name}: inflation creates invalid extents")
                if item.place is None:
                    if primary is not None:
                        raise ValueError(f"{item.name}: subsequent boxes require place(...)")
                    local_center = -0.5 * size * align
                else:
                    spec = item.place
                    base = local_boxes.get(spec.base_box) if spec.base_box else None
                    if spec.base_box and base is None:
                        raise ValueError(f"{item.name}: place base_box must be an earlier box in the same bone")
                    if base is None:
                        if np.any(_vec(spec.anchor) != 0):
                            raise ValueError("place without base_box requires anchor=[0,0,0]")
                        reference = np.zeros(3)
                    else:
                        reference = base.local_center + 0.5 * base.size * _vec(spec.anchor) * sign
                    local_center = reference - 0.5 * size * _vec(spec.align) * sign + _vec(spec.offset) * sign
                compiled_box = WorldBox(
                    box_name, name, world_pivot + world_rotation @ local_center,
                    world_rotation.copy(), size, float(item.inflate), local_center,
                    item.material, tuple(item.cullface), mirrored,
                )
                boxes.append(compiled_box)
                local_boxes[item.name] = compiled_box
                if primary is None:
                    primary = compiled_box
                    if attachment_base is not None and not bone.attach.detached:
                        edge(attachment_base.name, box_name, "attach")
                if item.place is not None and item.place.base_box:
                    edge(local_boxes[item.place.base_box].name, box_name, "place")
            elif isinstance(item, Bone):
                if mirrored and item.mirror_name:
                    raise ValueError("Nested mirror_x declarations are not supported in the starter")
                walk(item, compiled_bone, local_boxes, mirrored=mirrored, prefix=prefix)
                if item.mirror_name:
                    if prefix is not None:
                        raise ValueError("Nested mirror_x declarations are not supported in the starter")
                    walk(
                        item, compiled_bone, local_boxes, mirrored=True,
                        prefix=item.mirror_name, renamed_root=item.mirror_name,
                    )
            else:
                raise ValueError(f"Unsupported geometry node: {type(item).__name__}")
        if primary is None:
            raise ValueError(f"Bone {bone.name} must contain at least one box in the starter")

    walk(program.root, None, {})
    return Scene(tuple(boxes), tuple(bones), frozenset(adjacency), program.meta, adjacency)
