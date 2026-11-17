"""NumPy reference implementation of Block-DSL v3.2 rigid transforms.

All public AST Euler angles are degrees. Column vectors use Rz @ Ry @ Rx.
Supports versioned ``size_step`` grids, zero-thickness planar quads (``plane_normal``),
per-box local ``rotate`` & ``pivot`` without extra bone synthesis, ``joint(...)``
attachments, ``mirror(axis=x|z)`` with exact O(3) -> SO(3) parity, ``radial(count=K, axis=...)``
cyclic expansions, and optional keyframe ``pose_position_deltas`` / ``pose_scale_factors``.
The differentiable PyTorch implementation is a separate module; these routines
are deliberately not described as differentiable or used to claim PGD results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from synccraft.dsl.ast_nodes import Bone, Box, Locator, Program, RadialSpec, validate_program


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


def _angle_sign_from_reflection(sign: np.ndarray) -> np.ndarray:
    """Return proper SO(3) Euler angle parity for diagonal reflection diag(sx, sy, sz).

    For S = diag(sx, sy, sz) with s_i in {-1, +1},
    S @ R_xyz(rx, ry, rz) @ S == R_xyz(sy*sz*rx, sx*sz*ry, sx*sy*rz) with det == +1.
    """
    sx, sy, sz = float(sign[0]), float(sign[1]), float(sign[2])
    return np.array((sy * sz, sx * sz, sx * sy), dtype=np.float64)


def _stored_cullfaces(cullfaces: tuple[str, ...], sign: np.ndarray, mirrored: bool) -> tuple[str, ...]:
    """Store cullface directions compatible with backends.common.reflected_cullfaces(box).

    ``reflected_cullfaces`` swaps east/west whenever ``box.mirrored`` is True.
    For X-mirrors ``(sign[0] < 0) == mirrored``, so ``cullfaces`` is stored verbatim;
    for Z-mirrors ``north``/``south`` swap and ``east``/``west`` pre-compensate ``mirrored``.
    """
    if not cullfaces:
        return ()
    flip_x = (float(sign[0]) < 0.0) != mirrored
    flip_y = float(sign[1]) < 0.0
    flip_z = float(sign[2]) < 0.0
    if not (flip_x or flip_y or flip_z):
        return cullfaces
    out: list[str] = []
    for face in cullfaces:
        f = face
        if flip_x:
            if f == "east":
                f = "west"
            elif f == "west":
                f = "east"
        if flip_y:
            if f == "up":
                f = "down"
            elif f == "down":
                f = "up"
        if flip_z:
            if f == "south":
                f = "north"
            elif f == "north":
                f = "south"
        out.append(f)
    return tuple(out)


def _stored_uv_faces(
    uv_faces: dict[str, Any],
    sign: np.ndarray,
) -> dict[str, Any]:
    """Reflect explicit per-face UV rectangles across X/Y/Z mirror planes."""
    if not uv_faces:
        return {}
    sx, sy, sz = float(sign[0]) < 0.0, float(sign[1]) < 0.0, float(sign[2]) < 0.0
    if not (sx or sy or sz):
        return dict(uv_faces)
    cur = dict(uv_faces)
    if sx:
        nxt: dict[str, Any] = {}
        swap_x = {"east": "west", "west": "east"}
        for f, r in cur.items():
            df = swap_x.get(f, f)
            u0, v0, u1, v1 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            rot = int(r[4]) if len(r) == 5 else 0
            if rot == 0:
                nxt[df] = (u1, v0, u0, v1)
            elif rot == 180:
                nxt[df] = (u1, v0, u0, v1, 180)
            else:
                nxt[df] = (u0, v1, u1, v0, 360 - rot)
        cur = nxt
    if sy:
        nxt = {}
        swap_y = {"up": "down", "down": "up"}
        for f, r in cur.items():
            df = swap_y.get(f, f)
            u0, v0, u1, v1 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            rot = int(r[4]) if len(r) == 5 else 0
            nxt[df] = (u0, v1, u1, v0) if rot == 0 else (u0, v1, u1, v0, (360 - rot) % 360)
        cur = nxt
    if sz:
        nxt = {}
        swap_z = {"north": "south", "south": "north"}
        for f, r in cur.items():
            df = swap_z.get(f, f)
            u0, v0, u1, v1 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
            rot = int(r[4]) if len(r) == 5 else 0
            if f in ("up", "down"):
                nxt[df] = (u0, v1, u1, v0) if rot == 0 else (u0, v1, u1, v0, (360 - rot) % 360)
            else:
                nxt[df] = (u1, v0, u0, v1) if rot == 0 else (u1, v0, u0, v1, (360 - rot) % 360)
        cur = nxt
    return cur


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
    local_euler: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    local_pivot: np.ndarray | None = None
    unrotated_local_center: np.ndarray | None = None
    plane_normal: str | None = None
    uv: tuple[float, float] | None = None
    uv_faces: dict[str, tuple[float, ...]] = field(default_factory=dict)
    texture: str | None = None
    blend_with: tuple[str, ...] = ()
    uv_share_key: str | None = None
    component_rest_pos: np.ndarray | None = None
    component_rest_rot: np.ndarray | None = None
    instance_name: str | None = None
    edge: Any = None
    layer: int = 0
    prism_group: str | None = None

    @property
    def half_extents(self) -> np.ndarray:
        return np.maximum(0.0, self.size / 2.0 + self.inflate)

    def corners(self, *, include_inflate: bool = True) -> np.ndarray:
        extents = self.half_extents if include_inflate else self.size / 2.0
        signs = np.asarray(tuple(product((-1.0, 1.0), repeat=3)))
        return (signs * extents) @ self.rotation.T + self.center

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "bone_name": self.bone_name,
            "center": self.center.tolist(),
            "rotation": self.rotation.tolist(),
            "size": self.size.tolist(),
            "inflate": self.inflate,
            "local_center": self.local_center.tolist(),
            "material": self.material,
            "cullface": list(self.cullface),
            "mirrored": self.mirrored,
        }
        if np.any(np.abs(self.local_euler) > 1e-12):
            payload["local_euler"] = self.local_euler.tolist()
        if self.local_pivot is not None:
            payload["local_pivot"] = self.local_pivot.tolist()
        if self.plane_normal is not None:
            payload["plane_normal"] = self.plane_normal
        if self.uv is not None:
            payload["uv"] = list(self.uv)
        if self.uv_faces:
            payload["uv_faces"] = {k: list(v) for k, v in self.uv_faces.items()}
        if self.texture is not None:
            payload["texture"] = self.texture
        if self.blend_with:
            payload["blend_with"] = list(self.blend_with)
        if self.uv_share_key is not None:
            payload["uv_share_key"] = self.uv_share_key
        if self.instance_name is not None:
            payload["instance_name"] = self.instance_name
        return payload


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
            "name": self.name,
            "parent_name": self.parent_name,
            "pivot_world": self.pivot_world.tolist(),
            "rotation": self.rotation.tolist(),
            "local_translation": self.local_translation.tolist(),
            "local_euler": self.local_euler.tolist(),
            "mirrored": self.mirrored,
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
    local_euler: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "bone_name": self.bone_name,
            "role": self.role,
            "position_world": self.position_world.tolist(),
            "rotation": self.rotation.tolist(),
            "local_offset": self.local_offset.tolist(),
            "mirrored": self.mirrored,
        }
        if np.any(np.abs(self.local_euler) > 1e-12):
            payload["local_euler"] = self.local_euler.tolist()
        return payload


@dataclass(frozen=True)
class Scene:
    boxes: tuple[WorldBox, ...]
    bones: tuple[WorldBone, ...]
    adjacency: frozenset[tuple[str, str]]
    meta: Any
    adjacency_reasons: dict[tuple[str, str], str] = field(default_factory=dict)
    locators: tuple[WorldLocator, ...] = ()
    blend_pairs: frozenset[tuple[str, str]] = frozenset()
    layout_contracts: Any = None

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
        meta_dict: dict[str, Any] = {
            "archetype": self.meta.archetype,
            "target_profile": self.meta.target_profile,
            "texel_density": self.meta.texel_density,
        }
        if getattr(self.meta, "size_step", 1.0) != 1.0:
            meta_dict["size_step"] = float(self.meta.size_step)
        if getattr(self.meta, "audit_profile", "strict") != "strict":
            meta_dict["audit_profile"] = str(self.meta.audit_profile)
        payload: dict[str, Any] = {
            "schema": "synccraft.scene/0.1",
            "coordinate_system": "right_handed_y_up_column_vectors",
            "angle_units": "degrees",
            "meta": meta_dict,
            "boxes": [box.to_dict() for box in self.boxes],
            "bones": [bone.to_dict() for bone in self.bones],
            "adjacency": [
                {"pair": list(pair), "reason": self.adjacency_reasons[pair]}
                for pair in sorted(self.adjacency)
            ],
        }
        if self.locators:
            payload["locators"] = [loc.to_dict() for loc in self.locators]
        if self.blend_pairs:
            payload["blend_pairs"] = [list(pair) for pair in sorted(self.blend_pairs)]
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
    mirrored: bool = False,
    angle_sign: np.ndarray | None = None,
) -> np.ndarray:
    eff_sign = (
        angle_sign
        if angle_sign is not None
        else (np.array((1.0, -1.0, -1.0), dtype=np.float64) if mirrored else np.ones(3, dtype=np.float64))
    )
    clamped = np.zeros(3, dtype=np.float64)
    for idx in range(3):
        lo, hi = float(joint_limits[idx][0]), float(joint_limits[idx][1])
        if float(eff_sign[idx]) < 0.0:
            lo, hi = -hi, -lo
        clamped[idx] = float(np.clip(delta_euler[idx], lo, hi))
    return clamped


def _validate_physical_size(
    size: np.ndarray,
    size_step: float,
    plane_normal: str | None,
    effective_inflate: float,
    box_name: str,
) -> None:
    if not np.isfinite(effective_inflate):
        raise ValueError(f"{box_name}: inflation creates invalid extents")
    if plane_normal is None:
        if size_step == 1.0:
            if np.any(size <= 0) or np.any(size != np.floor(size)):
                raise ValueError(f"{box_name}: size must contain positive integers")
        else:
            steps = np.round(size / size_step)
            if np.any(size <= 0) or np.any(steps < 1) or not np.allclose(steps * size_step, size, atol=1e-9, rtol=0):
                raise ValueError(f"{box_name}: size must align to positive multiples of size_step={size_step}")
        if np.any(size / 2.0 + effective_inflate <= 0.0):
            raise ValueError(f"{box_name}: inflation creates invalid extents")
    else:
        n_idx = {"x": 0, "y": 1, "z": 2}[plane_normal]
        if abs(float(size[n_idx])) > 1e-12:
            raise ValueError(f"{box_name}: planar box normal axis {plane_normal!r} must have 0 thickness")
        for idx in range(3):
            if idx == n_idx:
                continue
            comp = float(size[idx])
            if size_step == 1.0:
                if comp <= 0.0 or comp != float(np.floor(comp)):
                    raise ValueError(f"{box_name}: planar tangential sizes must be positive integers")
            else:
                steps_c = round(comp / size_step)
                if comp <= 0.0 or steps_c < 1 or abs(steps_c * size_step - comp) > 1e-9:
                    raise ValueError(f"{box_name}: planar size {comp} not aligned to size_step={size_step}")
            if comp / 2.0 + effective_inflate <= 0.0:
                raise ValueError(f"{box_name}: inflation creates invalid tangential extents")


def _radial_rotation_matrix(axis: str, theta_deg: float, parent_angle_sign: np.ndarray) -> np.ndarray:
    if axis == "x":
        eul = np.array((theta_deg, 0.0, 0.0), dtype=np.float64)
    elif axis == "y":
        eul = np.array((0.0, theta_deg, 0.0), dtype=np.float64)
    else:
        eul = np.array((0.0, 0.0, theta_deg), dtype=np.float64)
    return euler_xyz_matrix(eul * parent_angle_sign)


def compile_geometry(
    program: Program,
    *,
    pose_euler_deltas: dict[str, Any] | None = None,
    box_inflate_deltas: dict[str, float] | None = None,
    root_translation_delta: Any | None = None,
    pose_position_deltas: dict[str, Any] | None = None,
    pose_scale_factors: dict[str, Any] | None = None,
    clamp_joint_limits: bool = True,
    validate: bool = True,
) -> Scene:
    """Expand validated boxes, planes, locators, placements, child bones, mirrors, and radials.

    ``mirror_x(right)`` and ``mirror(axis=x|z)`` create siblings reflected across the
    parent's local X or Z plane with exact O(3) -> SO(3) parity toggling at arbitrary
    nesting depth. ``radial(count=K, axis=...)`` expands K-fold cyclic siblings around
    the parent axis. Per-box ``rotate`` and ``pivot`` compile directly into ``WorldBox``
    orientations without synthesizing extra bones.
    """
    if validate:
        validate_program(program)
    size_step = float(getattr(program.meta, "size_step", 1.0))
    boxes: list[WorldBox] = []
    bones: list[WorldBone] = []
    locators: list[WorldLocator] = []
    adjacency: dict[tuple[str, str], str] = {}
    pending_blends: list[tuple[str | None, int, tuple[str, ...]]] = []
    pending_contacts: list[tuple[str | None, int, tuple[str, ...]]] = []
    names: set[str] = set()
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
        sign: np.ndarray,
        parent_scale: np.ndarray,
        prefix: str | None = None,
        renamed_root: str | None = None,
        radial_rot: np.ndarray | None = None,
    ) -> None:
        name = renamed_root or (f"{prefix}__{bone.name}" if prefix else bone.name)
        reserve(name)
        mirrored = bool(float(sign[0] * sign[1] * sign[2]) < 0.0)
        angle_sign = _angle_sign_from_reflection(sign)

        local_scale = np.ones(3, dtype=np.float64)
        if pose_scale_factors and name in pose_scale_factors:
            local_scale = _vec(pose_scale_factors[name])
            if np.any(local_scale <= 0.0):
                raise ValueError(f"{name}: pose_scale_factors must be strictly positive")
        cumulative_scale = parent_scale * local_scale

        euler = _vec(bone.rest_euler) * angle_sign
        local_rotation = euler_xyz_matrix(euler)
        if pose_euler_deltas and name in pose_euler_deltas:
            delta_euler = _vec(pose_euler_deltas[name])
            if clamp_joint_limits and bone.joint_limits is not None:
                delta_euler = _clamp_delta_to_joint_limits(
                    delta_euler, bone.joint_limits, angle_sign=angle_sign
                )
            local_rotation = local_rotation @ euler_xyz_matrix(delta_euler)
            euler = _recover_euler_xyz_degrees(local_rotation)

        if radial_rot is not None:
            local_rotation = radial_rot @ local_rotation
            euler = _recover_euler_xyz_degrees(local_rotation)

        pos_delta = np.zeros(3, dtype=np.float64)
        if pose_position_deltas and name in pose_position_deltas:
            pos_delta = _vec(pose_position_deltas[name]) * sign

        if parent is None:
            if bone.attach is not None:
                raise ValueError("The root bone cannot have an attachment")
            translation = _vec(bone.pivot_local) + root_delta_vec + pos_delta
            world_pivot = translation.copy()
            world_rotation = local_rotation
            align = np.zeros(3)
            attachment_base: WorldBox | None = None
        else:
            if bone.attach is None:
                raise ValueError(f"Child bone {bone.name} requires attach(...) or joint(...)")
            spec = bone.attach
            if spec.parent_box in parent_local_boxes:
                attachment_base = parent_local_boxes[spec.parent_box]
                r_parent_box = euler_xyz_matrix(attachment_base.local_euler)
                seat_scale = 1.0 - float(getattr(spec, "seat", 0.0))
                rel_attach = (
                    r_parent_box @ (0.5 * attachment_base.size * (_vec(spec.anchor) * sign * seat_scale))
                    + _vec(spec.offset) * sign * parent_scale
                )
                if radial_rot is not None:
                    rel_attach = radial_rot @ rel_attach
                translation = attachment_base.local_center + rel_attach + pos_delta * parent_scale
            elif spec.parent_box in parent_local_locators:
                if np.any(_vec(spec.anchor) != 0):
                    raise ValueError(
                        f"{bone.name}: attaching to locator {spec.parent_box!r} requires anchor=[0,0,0]"
                    )
                loc_base = parent_local_locators[spec.parent_box]
                rel_loc = loc_base.local_offset + _vec(spec.offset) * sign * parent_scale
                if radial_rot is not None:
                    rel_loc = radial_rot @ rel_loc
                translation = rel_loc + pos_delta * parent_scale
                attachment_base = None if spec.detached else inherited_attachment_box
            else:
                raise ValueError(
                    f"{bone.name}: parent_box {spec.parent_box!r} is not an earlier box or locator in the parent bone"
                )
            world_pivot = parent.pivot_world + parent.rotation @ translation
            world_rotation = parent.rotation @ local_rotation
            align = _vec(spec.align) * sign

        compiled_bone = WorldBone(
            name,
            parent.name if parent else None,
            world_pivot,
            world_rotation,
            translation,
            euler,
            mirrored,
        )
        bones.append(compiled_bone)
        local_boxes: dict[str, WorldBox] = {}
        local_locators: dict[str, WorldLocator] = {}
        primary: WorldBox | None = None

        for item in bone.items:
            if isinstance(item, Box):
                box_name = f"{prefix}__{item.name}" if prefix else item.name
                reserve(box_name)
                raw_size = _vec(item.size)
                extra_inflate = float(box_inflate_deltas.get(box_name, 0.0)) if box_inflate_deltas else 0.0
                effective_inflate = float(item.inflate) + extra_inflate
                _validate_physical_size(raw_size, size_step, item.plane_normal, effective_inflate, item.name)
                size = raw_size * cumulative_scale

                if item.explicit_local_center is not None:
                    unrot_local_center = _vec(item.explicit_local_center) * sign * cumulative_scale
                elif item.place is None:
                    if primary is not None:
                        raise ValueError(f"{item.name}: subsequent boxes require place(...)")
                    unrot_local_center = -0.5 * size * align
                else:
                    spec = item.place
                    if spec.base_box:
                        if spec.base_box in local_boxes:
                            base = local_boxes[spec.base_box]
                            r_base_local = euler_xyz_matrix(base.local_euler)
                            reference = base.local_center + r_base_local @ (
                                0.5 * base.size * (_vec(spec.anchor) * sign)
                            )
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
                    unrot_local_center = (
                        reference
                        - 0.5 * size * _vec(spec.align) * sign
                        + _vec(spec.offset) * sign * cumulative_scale
                    )

                box_local_euler = _vec(item.rotate) * angle_sign
                r_box_local = euler_xyz_matrix(box_local_euler)
                if item.pivot is not None:
                    box_local_pivot = _vec(item.pivot) * sign * cumulative_scale
                    local_center = box_local_pivot + r_box_local @ (unrot_local_center - box_local_pivot)
                else:
                    box_local_pivot = None
                    local_center = unrot_local_center

                item_layer = int(getattr(item, "layer", 0))
                if item.plane_normal is not None and item_layer != 0:
                    n_idx = {"x": 0, "y": 1, "z": 2}[item.plane_normal]
                    layer_vec = np.zeros(3, dtype=np.float64)
                    layer_vec[n_idx] = 0.25 * float(item_layer) * float(sign[n_idx])
                    local_center = local_center + (r_box_local @ layer_vec) * cumulative_scale
                    if item.pivot is not None:
                        unrot_local_center = unrot_local_center + layer_vec * cumulative_scale
                    else:
                        unrot_local_center = local_center

                box_world_center = world_pivot + world_rotation @ local_center
                box_world_rot = world_rotation @ r_box_local
                comp_rest_pos = (
                    _vec(item.component_rest_pos) * sign
                    if item.component_rest_pos is not None
                    else None
                )
                comp_rest_rot = (
                    euler_xyz_matrix(_vec(item.component_rest_euler) * angle_sign)
                    if item.component_rest_euler is not None
                    else None
                )
                eff_uv_share = (
                    None
                    if (mirrored and item.uv_share_key is not None)
                    else item.uv_share_key
                )
                compiled_box = WorldBox(
                    box_name,
                    name,
                    box_world_center,
                    box_world_rot,
                    size,
                    effective_inflate,
                    local_center,
                    item.material,
                    _stored_cullfaces(tuple(item.cullface), sign, mirrored),
                    mirrored,
                    local_euler=box_local_euler,
                    local_pivot=box_local_pivot,
                    unrotated_local_center=unrot_local_center,
                    plane_normal=item.plane_normal,
                    uv=item.uv,
                    uv_faces=_stored_uv_faces(item.uv_faces_dict, sign),
                    texture=item.texture,
                    uv_share_key=eff_uv_share,
                    component_rest_pos=comp_rest_pos,
                    component_rest_rot=comp_rest_rot,
                    instance_name=f"{prefix}__{item.instance_name}" if (prefix and item.instance_name) else item.instance_name,
                    edge=getattr(item, "edge", None),
                    layer=item_layer,
                    prism_group=f"{prefix}__{item.prism_group}" if (prefix and getattr(item, "prism_group", None)) else getattr(item, "prism_group", None),
                )
                box_idx = len(boxes)
                boxes.append(compiled_box)
                if item.blend_with:
                    pending_blends.append((prefix, box_idx, item.blend_with))
                if item.explicit_contact_with:
                    pending_contacts.append((prefix, box_idx, item.explicit_contact_with))
                local_boxes[item.name] = compiled_box
                if primary is None:
                    primary = compiled_box
                    if attachment_base is not None and bone.attach is not None and not bone.attach.detached:
                        reason = "joint" if bone.attach.joint_mode else "attach"
                        edge(attachment_base.name, box_name, reason)
                if item.place is not None and item.place.base_box and item.place.base_box in local_boxes:
                    edge(local_boxes[item.place.base_box].name, box_name, "place")
            elif isinstance(item, Locator):
                loc_name = f"{prefix}__{item.name}" if prefix else item.name
                reserve(loc_name)
                local_offset = _vec(item.offset) * sign * cumulative_scale
                loc_euler = _vec(item.rot) * angle_sign
                r_loc_local = euler_xyz_matrix(loc_euler)
                compiled_loc = WorldLocator(
                    name=loc_name,
                    bone_name=name,
                    role=item.role,
                    position_world=world_pivot + world_rotation @ local_offset,
                    rotation=world_rotation @ r_loc_local,
                    local_offset=local_offset,
                    mirrored=mirrored,
                    local_euler=loc_euler,
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
                    sign=sign,
                    parent_scale=cumulative_scale,
                    prefix=prefix,
                )
                m_flip = (
                    np.array((-1.0, 1.0, 1.0), dtype=np.float64)
                    if item.mirror_axis == "x"
                    else np.array((1.0, 1.0, -1.0), dtype=np.float64)
                )
                if item.mirror_name:
                    m_prefix = f"{prefix}__{item.mirror_name}" if prefix else item.mirror_name
                    walk(
                        item,
                        compiled_bone,
                        local_boxes,
                        local_locators,
                        child_inherited,
                        sign=sign * m_flip,
                        parent_scale=cumulative_scale,
                        prefix=m_prefix,
                        renamed_root=m_prefix,
                    )
                if isinstance(item.radial, RadialSpec):
                    for k in range(1, item.radial.count):
                        theta_k = 360.0 * float(k) / float(item.radial.count)
                        r_rad = _radial_rotation_matrix(item.radial.axis, theta_k, angle_sign)
                        r_base_name = f"{item.name}_r{k}"
                        r_prefix = f"{prefix}__{r_base_name}" if prefix else r_base_name
                        walk(
                            item,
                            compiled_bone,
                            local_boxes,
                            local_locators,
                            child_inherited,
                            sign=sign,
                            parent_scale=cumulative_scale,
                            prefix=r_prefix,
                            renamed_root=r_prefix,
                            radial_rot=r_rad,
                        )
                        if item.mirror_name:
                            rm_base_name = f"{item.mirror_name}_r{k}"
                            rm_prefix = f"{prefix}__{rm_base_name}" if prefix else rm_base_name
                            walk(
                                item,
                                compiled_bone,
                                local_boxes,
                                local_locators,
                                child_inherited,
                                sign=sign * m_flip,
                                parent_scale=cumulative_scale,
                                prefix=rm_prefix,
                                renamed_root=rm_prefix,
                                radial_rot=r_rad,
                            )
            else:
                raise ValueError(f"Unsupported geometry node: {type(item).__name__}")
        if primary is None and not local_locators:
            raise ValueError(f"Bone {bone.name} must contain at least one box or locator")

    walk(
        program.root,
        None,
        {},
        {},
        None,
        sign=np.ones(3, dtype=np.float64),
        parent_scale=np.ones(3, dtype=np.float64),
    )
    blend_pairs: set[tuple[str, str]] = set()
    if pending_blends:
        import re as _re

        radial_counts: dict[str, int] = {}

        def _collect_radial_counts(node: Bone, pref_r: str | None = None) -> None:
            for ch in node.items:
                if isinstance(ch, Bone):
                    if isinstance(ch.radial, RadialSpec):
                        base_r = f"{pref_r}__{ch.name}" if pref_r else ch.name
                        radial_counts[base_r] = ch.radial.count
                    _collect_radial_counts(ch, pref_r)
                    if ch.mirror_name:
                        mp = f"{pref_r}__{ch.mirror_name}" if pref_r else ch.mirror_name
                        _collect_radial_counts(ch, mp)

        _collect_radial_counts(program.root)

        for pref, b_idx, raw_targets in pending_blends:
            src_box = boxes[b_idx]
            resolved_targets: list[str] = []
            for tgt in raw_targets:
                cand = f"{pref}__{tgt}" if (pref and f"{pref}__{tgt}" in names) else tgt
                if pref and cand == tgt:
                    m_src = _re.match(r"^(.*)_r(\d+)$", pref)
                    if m_src and m_src.group(1) in radial_counts:
                        r_root = m_src.group(1)
                        k_cur = int(m_src.group(2))
                        r_cnt = radial_counts[r_root]
                        leaf_r = r_root.split("__")[-1]
                        m_tgt = _re.match(rf"^{_re.escape(leaf_r)}_r(\d+)__(.+)$", tgt)
                        if m_tgt:
                            k_next = (k_cur + int(m_tgt.group(1))) % r_cnt
                            outer_p = r_root[: -(len(leaf_r) + 2)] if "__" in r_root else None
                            if k_next == 0:
                                cand = f"{outer_p}__{m_tgt.group(2)}" if outer_p else m_tgt.group(2)
                            else:
                                cand = f"{r_root}_r{k_next}__{m_tgt.group(2)}"
                    else:
                        leaf_p = pref.split("__")[-1]
                        if tgt.startswith(f"{leaf_p}__"):
                            outer_p = pref[: -(len(leaf_p) + 2)] if "__" in pref else None
                            base_item = tgt[len(leaf_p) + 2 :]
                            cand_inv = f"{outer_p}__{base_item}" if outer_p else base_item
                            if cand_inv in names:
                                cand = cand_inv
                if cand in names and cand != src_box.name:
                    resolved_targets.append(cand)
                    blend_pairs.add((src_box.name, cand) if src_box.name <= cand else (cand, src_box.name))
            if resolved_targets:
                object.__setattr__(src_box, "blend_with", tuple(resolved_targets))
    if pending_contacts:
        box_names_set = {b.name for b in boxes}
        for pref, b_idx, raw_contacts in pending_contacts:
            src_box = boxes[b_idx]
            for tgt in raw_contacts:
                cand = f"{pref}__{tgt}" if (pref and f"{pref}__{tgt}" in box_names_set) else tgt
                if cand in box_names_set and cand != src_box.name:
                    edge(src_box.name, cand, "connect")
    layout_contracts = (
        getattr(program.assembly_spec, "layout", None)
        if getattr(program, "assembly_spec", None) is not None
        else None
    )
    return Scene(
        tuple(boxes),
        tuple(bones),
        frozenset(adjacency),
        program.meta,
        adjacency,
        tuple(locators),
        frozenset(blend_pairs),
        layout_contracts=layout_contracts,
    )


def compile_posed_geometry(
    program: Program,
    pose_euler_deltas: dict[str, Any] | None = None,
    *,
    box_inflate_deltas: dict[str, float] | None = None,
    root_translation_delta: Any | None = None,
    pose_position_deltas: dict[str, Any] | None = None,
    pose_scale_factors: dict[str, Any] | None = None,
    clamp_joint_limits: bool = True,
    validate: bool = True,
) -> Scene:
    """Compile a posed Scene sharing the exact SE(3) chain with rest compilation."""
    if validate and (pose_euler_deltas or box_inflate_deltas or pose_position_deltas or pose_scale_factors):
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
                    if isinstance(child.radial, RadialSpec):
                        for k in range(1, child.radial.count):
                            r_name = f"{child.name}_r{k}"
                            r_prefix = f"{prefix}__{r_name}" if prefix else r_name
                            _collect(child, prefix=r_prefix, renamed=r_prefix)
                            if child.mirror_name:
                                rm_name = f"{child.mirror_name}_r{k}"
                                rm_prefix = f"{prefix}__{rm_name}" if prefix else rm_name
                                _collect(child, prefix=rm_prefix, renamed=rm_prefix)

        _collect(program.root)
        if pose_euler_deltas:
            unknown = sorted(set(pose_euler_deltas) - valid_bones)
            if unknown:
                raise ValueError(f"Unknown bone(s) in pose_euler_deltas: {unknown}")
        if pose_position_deltas:
            unknown_pos = sorted(set(pose_position_deltas) - valid_bones)
            if unknown_pos:
                raise ValueError(f"Unknown bone(s) in pose_position_deltas: {unknown_pos}")
        if pose_scale_factors:
            unknown_sc = sorted(set(pose_scale_factors) - valid_bones)
            if unknown_sc:
                raise ValueError(f"Unknown bone(s) in pose_scale_factors: {unknown_sc}")
        if box_inflate_deltas:
            unknown_boxes = sorted(set(box_inflate_deltas) - valid_boxes)
            if unknown_boxes:
                raise ValueError(f"Unknown box(es) in box_inflate_deltas: {unknown_boxes}")
    return compile_geometry(
        program,
        pose_euler_deltas=pose_euler_deltas,
        box_inflate_deltas=box_inflate_deltas,
        root_translation_delta=root_translation_delta,
        pose_position_deltas=pose_position_deltas,
        pose_scale_factors=pose_scale_factors,
        clamp_joint_limits=clamp_joint_limits,
        validate=validate,
    )
