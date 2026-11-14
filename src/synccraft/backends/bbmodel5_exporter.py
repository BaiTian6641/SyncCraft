"""Blockbench 5.0 static Generic Model exporter.

The format uses separate groups and an outliner UUID tree. This is a tested
serialization contract, not a claim of verified Blockbench UI import. See
docs/COMPATIBILITY.md for sources and the remaining import gate.
"""

from __future__ import annotations

import base64
import struct
from typing import Any
import zlib

import numpy as np

from .common import FACES, ExportError, face_uv, reflected_cullfaces, stable_uuid, validate_atlas, validate_scene_for_export, vector


def _white_atlas(scene: Any) -> dict:
    """An original 1px white fallback so omitted atlases still show visible faces."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff\xff")) + chunk(b"IEND", b"")
    return {
        "width": 1, "height": 1,
        "data_url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        "faces": {box.name: {face: [0, 0, 1, 1] for face in FACES} for box in scene.boxes},
    }


def export_bbmodel(scene: Any, *, name: str = "SyncCraft asset", atlas: dict | None = None) -> dict:
    """Return JSON-safe .bbmodel data, preserving the static bone hierarchy.

    ``atlas`` is optional: width, height, data_url and faces[box][face] pixel UVs.
    Omission creates a white 1px texture; it is not material or cavity shading.
    Group origins are cumulative *unrotated* local translations. Using world
    pivots directly would apply ancestor rotations twice on import.
    """
    validate_scene_for_export(scene)
    if atlas is None:
        atlas = _white_atlas(scene)
    width, height = validate_atlas(atlas, scene, embedded=True)
    origins: dict[str, np.ndarray] = {}
    nodes: dict[str, dict] = {}
    groups: list[dict] = []
    outliner: list[dict] = []
    for bone in scene.bones:
        if bone.name in origins:
            raise ExportError(f"duplicate bone name {bone.name}")
        if bone.parent_name is not None and bone.parent_name not in origins:
            raise ExportError("bones must be in parent-before-child order")
        parent_origin = origins[bone.parent_name] if bone.parent_name is not None else np.zeros(3)
        origin = parent_origin + np.asarray(bone.local_translation)
        origins[bone.name] = origin
        uid = stable_uuid("bone", bone.name, name)
        groups.append({
            "name": bone.name, "uuid": uid,
            "origin": vector(origin), "rotation": vector(bone.local_euler),
            "export": True, "visibility": True,
        })
        node = {"uuid": uid, "isOpen": True, "children": []}
        nodes[bone.name] = node
        if bone.parent_name is None:
            outliner.append(node)
        else:
            nodes[bone.parent_name]["children"].append(node)
    elements: list[dict] = []
    for box in scene.boxes:
        if box.bone_name not in origins:
            raise ExportError(f"box {box.name} references a missing bone")
        center = origins[box.bone_name] + np.asarray(box.local_center)
        half_size = np.asarray(box.size) / 2
        uid = stable_uuid("box", box.name, name)
        faces = {}
        for face in FACES:
            face_data = {"uv": face_uv(atlas, box.name, face, normalized=False), "texture": 0 if atlas is not None else None}
            if face in reflected_cullfaces(box):
                face_data["cullface"] = face
            faces[face] = face_data
        elements.append({
            "name": box.name, "type": "cube", "uuid": uid,
            "from": vector(center - half_size), "to": vector(center + half_size),
            "origin": vector(origins[box.bone_name]), "rotation": [0.0, 0.0, 0.0],
            "inflate": float(box.inflate), "box_uv": False, "autouv": 0,
            "faces": faces, "export": True, "visibility": True,
        })
        nodes[box.bone_name]["children"].append(uid)
    textures = []
    if atlas is not None:
        textures.append({
            "name": "atlas.png", "id": "0", "uuid": stable_uuid("texture", "atlas", name),
            "width": width, "height": height, "uv_width": width, "uv_height": height,
            "source": atlas["data_url"], "internal": True,
            "render_mode": "default", "render_sides": "auto",
        })
    return {
        "meta": {"format_version": "5.0", "model_format": "free", "box_uv": False},
        "name": name, "resolution": {"width": width, "height": height},
        "elements": elements, "groups": groups, "outliner": outliner,
        "textures": textures,
    }
