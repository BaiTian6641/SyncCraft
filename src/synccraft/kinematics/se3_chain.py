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

from synccraft.dsl.ast_nodes import Bone, Box, Locator, Program, validate_program


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
class WorldLocator:
    name: str
    bone_name: str
    role: str
    position_world: np.ndarray
    rotation: np.ndarray
    local_offset: np.ndarray
    mirrored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bone_name": self.bone_name,
            "role": self.role,
            "position_world": self.position_world.tolist(),
            "rotation": self.rotation.tolist(),
            "local_offset": self.local_offset.tolist(),
            "mirrored": self.mirrored,
        }


@dataclass(frozen=True)
class Scene:
    boxes: tuple[WorldBox, ...]
    bones: tuple[WorldBone, ...]
    adjacency: frozenset[tuple[str, str]]
    meta: Any
    adjacency_reasons: dict[tuple[str, str], str] = field(default_factory=dict)
    locators: tuple[WorldLocator, ...] = ()

    @property
    def box_by_name(self) -> dict[str, WorldBox]:
        return {box.name: box for box in self.boxes}

    @property
    def bone_by_name(self) -> dict[str, WorldBone]:
        return {bone.name: bone for bone in self.bones}

    @property
    def locator_by_name(self) -> dict[str, WorldLocator]:
        return {loc.name: loc for loc in self.locators}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        if self.locators:
            payload["locators"] = [loc.to_dict() for loc in self.locators]
        return payload


def _recover_euler_xyz_degrees(rotation: np.ndarray) -> np.ndarray:
    """Recover degrees (x, y, z) for R = Rz(z) @ Ry(y) @ Rx(x)."""
    y = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(np.cos(y)) > 1e-8:
        x = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        z = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        x = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
        z = 0.0
    return np.rad2deg(np.array((x, y, z), dtype=np.float64))


def _clamp_delta_to_joint_limits(
    delta_euler: np.ndarray,
    joint_limits: Any,
    *,
    mirrored: bool,
) -> np.ndarray:
    rx_lo, rx_hi = float(joint_limits[0][0]), float(joint_limits[0][1])
    ry_lo, ry_hi = float(joint_limits[1][0]), float(joint_limits[1][1])
    rz_lo, rz_hi = float(joint_limits[2][0]), float(joint_limits[2][1])
    if mirrored:
        ry_lo, ry_hi = -ry_hi, -ry_lo
        rz_lo, rz_hi = -rz_hi, -rz_lo
    return np.array(
        (
            float(np.clip(delta_euler[0], rx_lo, rx_hi)),
            float(np.clip(delta_euler[1], ry_lo, ry_hi)),
            float(np.clip(delta_euler[2], rz_lo, rz_hi)),
        ),
        dtype=np.float64,
    )


def compile_geometry(
    program: Program,
    *,
    pose_euler_deltas: dict[str, Any] | None = None,
    box_inflate_deltas: dict[str, float] | None = None,
    root_translation_delta: Any | None = None,
    clamp_joint_limits: bool = True,
) -> Scene:
    """Expand validated boxes, locators, explicit placements, child bones, and mirrors.

    ``mirror_x(right)`` creates a sibling reflected across the parent's local-X
    plane with exact O(3) parity toggling at arbitrary nesting depth.
    Descendant names use ``right__original_name`` (and ``outer__inner__name`` for
    nested mirrors). Integer box sizes remain strictly immutable; ``inflate`` and
    optional ``box_inflate_deltas`` adjust physical clearances/muscle bulges only.
    """
    validate_program(program)
    boxes: list[WorldBox] = []
    bones: list[WorldBone] = []
    locators: list[WorldLocator] = []
    adjacency: dict[tuple[str, str], str] = {}
    names: set[str] = set()
    reflect = np.array((-1.0, 1.0, 1.0))
    angle_reflect = np.array((1.0, -1.0, -1.0))
    root_delta_vec = _vec(root_translation_delta) if root_translation_delta is not None else np.zeros(3)

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
        parent_local_locators: dict[str, WorldLocator],
        inherited_attachment_box: WorldBox | None,
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
        if pose_euler_deltas and name in pose_euler_deltas:
            delta_euler = _vec(pose_euler_deltas[name])
            if clamp_joint_limits and bone.joint_limits is not None:
                delta_euler = _clamp_delta_to_joint_limits(
                    delta_euler, bone.joint_limits, mirrored=mirrored
                )
            local_rotation = local_rotation @ euler_xyz_matrix(delta_euler)
            euler = _recover_euler_xyz_degrees(local_rotation)
        if parent is None:
            if bone.attach is not None:
                raise ValueError("The root bone cannot have an attachment")
            translation = _vec(bone.pivot_local) + root_delta_vec
            world_pivot = translation.copy()
            world_rotation = local_rotation
            align = np.zeros(3)
            attachment_base: WorldBox | None = None
        else:
            if bone.attach is None:
                raise ValueError(f"Child bone {bone.name} requires attach(...)")
            spec = bone.attach
            if spec.parent_box in parent_local_boxes:
                attachment_base = parent_local_boxes[spec.parent_box]
                translation = (
                    attachment_base.local_center
                    + 0.5 * attachment_base.size * _vec(spec.anchor) * sign
                    + _vec(spec.offset) * sign
                )
            elif spec.parent_box in parent_local_locators:
                if np.any(_vec(spec.anchor) != 0):
                    raise ValueError(
                        f"{bone.name}: attaching to locator {spec.parent_box!r} requires anchor=[0,0,0]"
                    )
                loc_base = parent_local_locators[spec.parent_box]
                translation = loc_base.local_offset + _vec(spec.offset) * sign
                attachment_base = None if spec.detached else inherited_attachment_box
            else:
                raise ValueError(
                    f"{bone.name}: parent_box {spec.parent_box!r} is not an earlier box or locator in the parent bone"
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
        local_locators: dict[str, WorldLocator] = {}
        primary: WorldBox | None = None
        for item in bone.items:
            if isinstance(item, Box):
                box_name = f"{prefix}__{item.name}" if prefix else item.name
                reserve(box_name)
                size = _vec(item.size)
                if np.any(size <= 0) or np.any(size != np.floor(size)):
                    raise ValueError(f"{item.name}: size must contain positive integers")
                extra_inflate = float(box_inflate_deltas.get(box_name, 0.0)) if box_inflate_deltas else 0.0
                effective_inflate = float(item.inflate) + extra_inflate
                if not np.isfinite(effective_inflate) or np.any(size / 2 + effective_inflate <= 0):
                    raise ValueError(f"{item.name}: inflation creates invalid extents")
                if item.place is None:
                    if primary is not None:
                        raise ValueError(f"{item.name}: subsequent boxes require place(...)")
                    local_center = -0.5 * size * align
                else:
                    spec = item.place
                    if spec.base_box:
                        if spec.base_box in local_boxes:
                            base = local_boxes[spec.base_box]
                            reference = base.local_center + 0.5 * base.size * _vec(spec.anchor) * sign
                        elif spec.base_box in local_locators:
                            if np.any(_vec(spec.anchor) != 0):
                                raise ValueError(
                                    f"{item.name}: place relative to locator {spec.base_box!r} requires anchor=[0,0,0]"
                                )
                            reference = local_locators[spec.base_box].local_offset
                        else:
                            raise ValueError(
                                f"{item.name}: place base_box must be an earlier box or locator in the same bone"
                            )
                    else:
                        if np.any(_vec(spec.anchor) != 0):
                            raise ValueError("place without base_box requires anchor=[0,0,0]")
                        reference = np.zeros(3)
                    local_center = reference - 0.5 * size * _vec(spec.align) * sign + _vec(spec.offset) * sign
                compiled_box = WorldBox(
                    box_name, name, world_pivot + world_rotation @ local_center,
                    world_rotation.copy(), size, effective_inflate, local_center,
                    item.material, tuple(item.cullface), mirrored,
                )
                boxes.append(compiled_box)
                local_boxes[item.name] = compiled_box
                if primary is None:
                    primary = compiled_box
                    if attachment_base is not None and not bone.attach.detached:
                        edge(attachment_base.name, box_name, "attach")
                if item.place is not None and item.place.base_box and item.place.base_box in local_boxes:
                    edge(local_boxes[item.place.base_box].name, box_name, "place")
            elif isinstance(item, Locator):
                loc_name = f"{prefix}__{item.name}" if prefix else item.name
                reserve(loc_name)
                local_offset = _vec(item.offset) * sign
                compiled_loc = WorldLocator(
                    name=loc_name,
                    bone_name=name,
                    role=item.role,
                    position_world=world_pivot + world_rotation @ local_offset,
                    rotation=world_rotation.copy(),
                    local_offset=local_offset,
                    mirrored=mirrored,
                )
                locators.append(compiled_loc)
                local_locators[item.name] = compiled_loc
            elif isinstance(item, Bone):
                child_inherited = primary if primary is not None else attachment_base
                walk(
                    item,
                    compiled_bone,
                    local_boxes,
                    local_locators,
                    child_inherited,
                    mirrored=mirrored,
                    prefix=prefix,
                )
                if item.mirror_name:
                    m_prefix = f"{prefix}__{item.mirror_name}" if prefix else item.mirror_name
                    walk(
                        item,
                        compiled_bone,
                        local_boxes,
                        local_locators,
                        child_inherited,
                        mirrored=not mirrored,
                        prefix=m_prefix,
                        renamed_root=m_prefix,
                    )
            else:
                raise ValueError(f"Unsupported geometry node: {type(item).__name__}")
        if primary is None and not local_locators:
            raise ValueError(f"Bone {bone.name} must contain at least one box or locator")

    walk(program.root, None, {}, {}, None)
    return Scene(
        tuple(boxes),
        tuple(bones),
        frozenset(adjacency),
        program.meta,
        adjacency,
        tuple(locators),
    )


def compile_posed_geometry(
    program: Program,
    pose_euler_deltas: dict[str, Any] | None = None,
    *,
    box_inflate_deltas: dict[str, float] | None = None,
    root_translation_delta: Any | None = None,
    clamp_joint_limits: bool = True,
) -> Scene:
    """Compile a posed Scene sharing the exact SE(3) chain with rest compilation."""
    if pose_euler_deltas or box_inflate_deltas:
        valid_bones: set[str] = set()
        valid_boxes: set[str] = set()

        def _collect(node: Bone, prefix: str | None = None, renamed: str | None = None) -> None:
            expanded = renamed or (f"{prefix}__{node.name}" if prefix else node.name)
            valid_bones.add(expanded)
            valid_bones.add(node.name)
            for child in node.items:
                if isinstance(child, Box):
                    valid_boxes.add(f"{prefix}__{child.name}" if prefix else child.name)
                elif isinstance(child, Bone):
                    _collect(child, prefix=prefix)
                    if child.mirror_name:
                        m_prefix = f"{prefix}__{child.mirror_name}" if prefix else child.mirror_name
                        _collect(child, prefix=m_prefix, renamed=m_prefix)

        _collect(program.root)
        if pose_euler_deltas:
            unknown = sorted(set(pose_euler_deltas) - valid_bones)
            if unknown:
                raise ValueError(f"Unknown bone(s) in pose_euler_deltas: {unknown}")
        if box_inflate_deltas:
            unknown_boxes = sorted(set(box_inflate_deltas) - valid_boxes)
            if unknown_boxes:
                raise ValueError(f"Unknown box(es) in box_inflate_deltas: {unknown_boxes}")
    return compile_geometry(
        program,
        pose_euler_deltas=pose_euler_deltas,
        box_inflate_deltas=box_inflate_deltas,
        root_translation_delta=root_translation_delta,
        clamp_joint_limits=clamp_joint_limits,
    )
