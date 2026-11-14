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


def _serialize_bbmodel_animations(
    program: Any,
    scene: Any,
    *,
    asset_name: str,
    num_animation_frames: int,
) -> list[dict[str, Any]]:
    from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip

    n_frames = max(4, int(num_animation_frames))
    root_bone_name = scene.bones[0].name
    animations_out: list[dict[str, Any]] = []

    for clip in program.behavior.clips:
        samples = evaluate_behavior_clip(program, clip.name, num_frames=n_frames)
        animators: dict[str, dict[str, Any]] = {}

        # Root position keyframes when non-zero
        if any(any(abs(float(c)) > 1e-5 for c in s.root_translation_delta) for s in samples):
            root_uid = stable_uuid("bone", root_bone_name, asset_name)
            pos_kfs: list[dict[str, Any]] = []
            for k_i, s in enumerate(samples):
                tx, ty, tz = s.root_translation_delta
                pos_kfs.append({
                    "channel": "position",
                    "data_points": [{"x": round(float(tx), 4), "y": round(float(ty), 4), "z": round(float(tz), 4)}],
                    "uuid": stable_uuid("kf_pos", f"{clip.name}:{root_bone_name}:{k_i}", asset_name),
                    "time": round(float(s.time_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            animators.setdefault(
                root_uid,
                {"name": root_bone_name, "type": "bone", "keyframes": []},
            )["keyframes"].extend(pos_kfs)

        for bone in scene.bones:
            bname = bone.name
            if bname == root_bone_name:
                continue
            frame_vecs = [s.pose_euler_deltas.get(bname, (0.0, 0.0, 0.0)) for s in samples]
            if max(max(abs(v[0]), abs(v[1]), abs(v[2])) for v in frame_vecs) < 1e-4:
                continue
            b_uid = stable_uuid("bone", bname, asset_name)
            rot_kfs: list[dict[str, Any]] = []
            for k_i, (s, vec_d) in enumerate(zip(samples, frame_vecs)):
                rot_kfs.append({
                    "channel": "rotation",
                    "data_points": [{
                        "x": round(float(vec_d[0]), 4),
                        "y": round(float(vec_d[1]), 4),
                        "z": round(float(vec_d[2]), 4),
                    }],
                    "uuid": stable_uuid("kf_rot", f"{clip.name}:{bname}:{k_i}", asset_name),
                    "time": round(float(s.time_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            if clip.loop:
                v0 = frame_vecs[0]
                rot_kfs.append({
                    "channel": "rotation",
                    "data_points": [{
                        "x": round(float(v0[0]), 4),
                        "y": round(float(v0[1]), 4),
                        "z": round(float(v0[2]), 4),
                    }],
                    "uuid": stable_uuid("kf_rot", f"{clip.name}:{bname}:wrap", asset_name),
                    "time": round(float(clip.duration_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            animators.setdefault(
                b_uid,
                {"name": bname, "type": "bone", "keyframes": []},
            )["keyframes"].extend(rot_kfs)

        animations_out.append({
            "uuid": stable_uuid("animation", clip.name, asset_name),
            "name": clip.name,
            "loop": "loop" if clip.loop else "once",
            "override": False,
            "length": round(float(clip.duration_s), 4),
            "snapping": 24,
            "selected": False,
            "animators": animators,
        })

    return animations_out


def export_bbmodel(
    scene: Any,
    *,
    name: str = "SyncCraft asset",
    atlas: dict | None = None,
    program: Any | None = None,
    num_animation_frames: int = 12,
) -> dict:
    """Return JSON-safe .bbmodel data, preserving the bone hierarchy (+ optional @behavior animations).

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
    doc: dict[str, Any] = {
        "meta": {"format_version": "5.0", "model_format": "free", "box_uv": False},
        "name": name, "resolution": {"width": width, "height": height},
        "elements": elements, "groups": groups, "outliner": outliner,
        "textures": textures,
    }
    if program is not None and getattr(program, "behavior", None) is not None and program.behavior.clips:
        doc["animations"] = _serialize_bbmodel_animations(
            program,
            scene,
            asset_name=name,
            num_animation_frames=num_animation_frames,
        )
    return doc
