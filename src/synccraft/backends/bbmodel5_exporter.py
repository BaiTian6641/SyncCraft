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
    from synccraft.dsl.ast_nodes import validate_program
    from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip
    from .bedrock_gecko_exporter import _extract_so3_euler_keyframe_deltas

    validate_program(program)
    n_frames = max(4, int(num_animation_frames))
    root_bone_name = scene.bones[0].name
    animations_out: list[dict[str, Any]] = []

    for clip in program.behavior.clips:
        samples = evaluate_behavior_clip(program, clip.name, num_frames=n_frames)
        animators: dict[str, dict[str, Any]] = {}

        explicit_kf_channels: set[tuple[str, str]] = set()
        has_procedural_solvers = bool(
            getattr(clip, "auto_rig", False)
            or getattr(clip, "ground_pin_limbs", False)
            or getattr(clip, "flights", ())
            or getattr(clip, "swims", ())
            or getattr(clip, "gaits", ())
            or getattr(clip, "ik_drives", ())
            or getattr(clip, "muscle_drives", ())
            or getattr(clip, "layers", ())
            or float(getattr(clip, "com_balance", 0.0)) > 1e-5
            or float(getattr(clip, "spine_coil", 0.0)) != 0.0
            or getattr(clip, "neck_aim", None) is not None
            or float(getattr(clip, "head_stabilize", 0.0)) > 1e-5
            or getattr(program.behavior, "springs", ())
            or (getattr(program, "kinematics", None) is not None and getattr(program.kinematics, "springs", ()))
        )
        if not has_procedural_solvers:
            for kf_tr in getattr(clip, "keyframes", ()):
                explicit_kf_channels.add((kf_tr.bone, kf_tr.channel))
                b_uid = stable_uuid("bone", kf_tr.bone, asset_name)
                ch_kfs: list[dict[str, Any]] = []
                interp = "catmullrom" if getattr(kf_tr, "interpolation", "linear") == "catmullrom" else "linear"
                for k_i, (t_s, vec_v) in enumerate(kf_tr.keys):
                    ch_kfs.append({
                        "channel": kf_tr.channel,
                        "data_points": [{
                            "x": round(float(vec_v[0]), 4),
                            "y": round(float(vec_v[1]), 4),
                            "z": round(float(vec_v[2]), 4),
                        }],
                        "uuid": stable_uuid(f"kf_{kf_tr.channel}", f"{clip.name}:{kf_tr.bone}:{k_i}", asset_name),
                        "time": round(float(t_s), 4),
                        "color": -1,
                        "interpolation": interp,
                    })
                if ch_kfs:
                    animators.setdefault(
                        b_uid,
                        {"name": kf_tr.bone, "type": "bone", "keyframes": []},
                    )["keyframes"].extend(ch_kfs)

        # Root position keyframes when non-zero
        is_in_place = bool(
            getattr(clip, "in_place", False)
            or any(getattr(g, "in_place", False) for g in getattr(clip, "gaits", ()))
            or any(getattr(fl, "in_place", False) for fl in getattr(clip, "flights", ()))
            or any(getattr(sw, "in_place", False) for sw in getattr(clip, "swims", ()))
        )
        has_macro_root_drift = (
            not is_in_place
            and (any(abs(d.stride) > 1e-6 for d in clip.ik_drives) or bool(getattr(clip, "gaits", ())))
        )
        def _root_total_pos(s: Any) -> tuple[float, float, float]:
            r = s.root_translation_delta
            p = s.pose_position_deltas.get(root_bone_name, (0.0, 0.0, 0.0))
            return (float(r[0]) + float(p[0]), float(r[1]) + float(p[1]), float(r[2]) + float(p[2]))

        if (root_bone_name, "position") not in explicit_kf_channels and any(
            any(abs(c) > 1e-5 for c in _root_total_pos(s)) for s in samples
        ):
            root_uid = stable_uuid("bone", root_bone_name, asset_name)
            pos_kfs: list[dict[str, Any]] = []
            for k_i, s in enumerate(samples):
                tx, ty, tz = _root_total_pos(s)
                pos_kfs.append({
                    "channel": "position",
                    "data_points": [{"x": round(float(tx), 4), "y": round(float(ty), 4), "z": round(float(tz), 4)}],
                    "uuid": stable_uuid("kf_pos", f"{clip.name}:{root_bone_name}:{k_i}", asset_name),
                    "time": round(float(s.time_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            if clip.loop and not has_macro_root_drift:
                tx0, ty0, tz0 = _root_total_pos(samples[0])
                pos_kfs.append({
                    "channel": "position",
                    "data_points": [{"x": round(float(tx0), 4), "y": round(float(ty0), 4), "z": round(float(tz0), 4)}],
                    "uuid": stable_uuid("kf_pos", f"{clip.name}:{root_bone_name}:wrap", asset_name),
                    "time": round(float(clip.duration_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            animators.setdefault(
                root_uid,
                {"name": root_bone_name, "type": "bone", "keyframes": []},
            )["keyframes"].extend(pos_kfs)

        tracks_by_bone: dict[str, list[Any]] = {}
        for tr in clip.tracks:
            tracks_by_bone.setdefault(tr.bone, []).append(tr)

        for bone in scene.bones:
            bname = bone.name
            if (bname, "rotation") in explicit_kf_channels:
                continue
            frame_vecs = _extract_so3_euler_keyframe_deltas(samples, bone)
            if max(max(abs(v[0]), abs(v[1]), abs(v[2])) for v in frame_vecs) < 1e-4:
                continue
            b_uid = stable_uuid("bone", bname, asset_name)
            b_tracks = tracks_by_bone.get(bname, [])
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
                v_last = np.asarray(frame_vecs[-1], dtype=np.float64)
                if any(tr.law == "spin" for tr in b_tracks) and len(frame_vecs) >= 2:
                    v_wrap = v_last + (v_last - np.asarray(frame_vecs[-2], dtype=np.float64))
                else:
                    v0_arr = np.asarray(frame_vecs[0], dtype=np.float64)
                    v_wrap = v_last + (((v0_arr - v_last) + 180.0) % 360.0 - 180.0)
                rot_kfs.append({
                    "channel": "rotation",
                    "data_points": [{
                        "x": round(float(v_wrap[0]), 4),
                        "y": round(float(v_wrap[1]), 4),
                        "z": round(float(v_wrap[2]), 4),
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

        # Export non-root prismatic / slide position channels (ADR 0008)
        for bone in scene.bones:
            bname = bone.name
            if bname == root_bone_name or (bname, "position") in explicit_kf_channels:
                continue
            if not any(
                bname in s.pose_position_deltas
                and any(abs(float(c)) > 1e-5 for c in s.pose_position_deltas[bname])
                for s in samples
            ):
                continue
            b_uid = stable_uuid("bone", bname, asset_name)
            x_sign = -1.0 if bone.mirrored else 1.0
            p_kfs: list[dict[str, Any]] = []
            for k_i, s in enumerate(samples):
                dx, dy, dz = s.pose_position_deltas.get(bname, (0.0, 0.0, 0.0))
                p_kfs.append({
                    "channel": "position",
                    "data_points": [{"x": round(x_sign * float(dx), 4), "y": round(float(dy), 4), "z": round(float(dz), 4)}],
                    "uuid": stable_uuid("kf_pos", f"{clip.name}:{bname}:{k_i}", asset_name),
                    "time": round(float(s.time_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            if clip.loop:
                dx0, dy0, dz0 = samples[0].pose_position_deltas.get(bname, (0.0, 0.0, 0.0))
                p_kfs.append({
                    "channel": "position",
                    "data_points": [{"x": round(x_sign * float(dx0), 4), "y": round(float(dy0), 4), "z": round(float(dz0), 4)}],
                    "uuid": stable_uuid("kf_pos", f"{clip.name}:{bname}:wrap", asset_name),
                    "time": round(float(clip.duration_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            animators.setdefault(
                b_uid,
                {"name": bname, "type": "bone", "keyframes": []},
            )["keyframes"].extend(p_kfs)

        # Export Sim(3) bone scale channels (e.g. volume-preserving muscle bulge / breathing)
        for bone in scene.bones:
            bname = bone.name
            if (bname, "scale") in explicit_kf_channels:
                continue
            if not any(
                bname in s.pose_scale_factors
                and any(abs(float(c) - 1.0) > 1e-4 for c in s.pose_scale_factors[bname])
                for s in samples
            ):
                continue
            b_uid = stable_uuid("bone", bname, asset_name)
            s_kfs: list[dict[str, Any]] = []
            for k_i, s in enumerate(samples):
                sx, sy, sz = s.pose_scale_factors.get(bname, (1.0, 1.0, 1.0))
                s_kfs.append({
                    "channel": "scale",
                    "data_points": [{"x": round(float(sx), 4), "y": round(float(sy), 4), "z": round(float(sz), 4)}],
                    "uuid": stable_uuid("kf_scl", f"{clip.name}:{bname}:{k_i}", asset_name),
                    "time": round(float(s.time_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            if clip.loop:
                sx0, sy0, sz0 = samples[0].pose_scale_factors.get(bname, (1.0, 1.0, 1.0))
                s_kfs.append({
                    "channel": "scale",
                    "data_points": [{"x": round(float(sx0), 4), "y": round(float(sy0), 4), "z": round(float(sz0), 4)}],
                    "uuid": stable_uuid("kf_scl", f"{clip.name}:{bname}:wrap", asset_name),
                    "time": round(float(clip.duration_s), 4),
                    "color": -1,
                    "interpolation": "linear",
                })
            animators.setdefault(
                b_uid,
                {"name": bname, "type": "bone", "keyframes": []},
            )["keyframes"].extend(s_kfs)

        if getattr(clip, "effects", ()):
            from synccraft.kinematics.musculoskeletal import resolve_clip_effect_events

            eff_events = resolve_clip_effect_events(program, clip)
            eff_kfs: list[dict[str, Any]] = []
            for idx_e, ev in enumerate(eff_events):
                t_e = round(float(ev["time_s"]), 4)
                if ev.get("sound"):
                    eff_kfs.append(
                        {
                            "channel": "sound",
                            "data_points": [{"effect": ev["sound"], "file": ""}],
                            "uuid": stable_uuid("bb_eff_snd", f"{clip.name}:{idx_e}", asset_name),
                            "time": t_e,
                            "color": -1,
                            "interpolation": "linear",
                        }
                    )
                if ev.get("particle"):
                    eff_kfs.append(
                        {
                            "channel": "particle",
                            "data_points": [{"effect": ev["particle"], "locator": ev.get("locator") or "", "script": ""}],
                            "uuid": stable_uuid("bb_eff_prt", f"{clip.name}:{idx_e}", asset_name),
                            "time": t_e,
                            "color": -1,
                            "interpolation": "linear",
                        }
                    )
                if ev.get("instruction"):
                    eff_kfs.append(
                        {
                            "channel": "timeline",
                            "data_points": [{"script": ev["instruction"]}],
                            "uuid": stable_uuid("bb_eff_ins", f"{clip.name}:{idx_e}", asset_name),
                            "time": t_e,
                            "color": -1,
                            "interpolation": "linear",
                        }
                    )
            if eff_kfs:
                animators["effects"] = {"name": "Effects", "type": "effect", "keyframes": eff_kfs}

        anim_dict_bb: dict[str, Any] = {
            "uuid": stable_uuid("animation", clip.name, asset_name),
            "name": clip.name,
            "loop": "loop" if clip.loop else "once",
            "override": False,
            "length": round(float(clip.duration_s), 4),
            "snapping": 24,
            "selected": False,
            "animators": animators,
        }
        if samples and (is_in_place or any(abs(float(c)) > 1e-6 for c in getattr(samples[0], "root_motion_velocity", (0.0, 0.0, 0.0)))):
            rm_v = samples[0].root_motion_velocity
            upb = float(getattr(program.meta, "units_per_block", 16))
            spd_u = float(np.linalg.norm(np.asarray(rm_v, dtype=np.float64)))
            anim_dict_bb["synccraft_root_motion"] = {
                "in_place": is_in_place,
                "speed_units_per_s": round(spd_u, 4),
                "speed_blocks_per_s": round(spd_u / max(1.0, upb), 4),
                "velocity_vec": [round(float(rm_v[0]), 4), round(float(rm_v[1]), 4), round(float(rm_v[2]), 4)],
            }
        animations_out.append(anim_dict_bb)

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
    from synccraft.kinematics.se3_chain import euler_xyz_matrix

    elements: list[dict] = []
    for box in scene.boxes:
        if box.bone_name not in origins:
            raise ExportError(f"box {box.name} references a missing bone")
        bone_origin = origins[box.bone_name]
        half_size = np.asarray(box.size, dtype=np.float64) / 2.0
        raw_loc_euler = getattr(box, "local_euler", None)
        has_loc_rot = raw_loc_euler is not None and np.any(np.abs(np.asarray(raw_loc_euler, dtype=np.float64)) > 1e-9)
        if has_loc_rot:
            loc_e = np.asarray(raw_loc_euler, dtype=np.float64)
            r_loc = euler_xyz_matrix((float(loc_e[0]), float(loc_e[1]), float(loc_e[2])))
            raw_lpiv = getattr(box, "local_pivot", None)
            c_piv_bone = (
                np.asarray(raw_lpiv, dtype=np.float64)
                if raw_lpiv is not None
                else np.asarray(box.local_center, dtype=np.float64)
            )
            raw_unrot = getattr(box, "unrotated_local_center", None)
            c_unrot_bone = (
                np.asarray(raw_unrot, dtype=np.float64)
                if raw_unrot is not None
                else (c_piv_bone + r_loc.T @ (np.asarray(box.local_center, dtype=np.float64) - c_piv_bone))
            )
            el_from = bone_origin + c_unrot_bone - half_size
            el_to = bone_origin + c_unrot_bone + half_size
            el_origin = vector(bone_origin + c_piv_bone)
            el_rot = vector(loc_e)
        else:
            center = bone_origin + np.asarray(box.local_center, dtype=np.float64)
            el_from = center - half_size
            el_to = center + half_size
            el_origin = vector(bone_origin)
            el_rot = [0.0, 0.0, 0.0]

        uid = stable_uuid("box", box.name, name)
        faces = {}
        box_uv_faces = getattr(box, "uv_faces", None) or {}
        for face in FACES:
            uv_rot = 0
            if box_uv_faces and width == 1 and height == 1:
                if face not in box_uv_faces:
                    faces[face] = {"uv": [0.0, 0.0, 0.0, 0.0], "texture": None}
                    continue
                raw_rect = box_uv_faces[face]
                uv_rect = list(raw_rect[:4])
                if len(raw_rect) == 5:
                    uv_rot = int(raw_rect[4])
            else:
                uv_rect = face_uv(atlas, box.name, face, normalized=False)
            face_data: dict[str, Any] = {"uv": uv_rect, "texture": 0 if atlas is not None else None}
            if uv_rot:
                face_data["rotation"] = uv_rot
            if face in reflected_cullfaces(box):
                face_data["cullface"] = face
            faces[face] = face_data
        elements.append({
            "name": box.name, "type": "cube", "uuid": uid,
            "from": vector(el_from), "to": vector(el_to),
            "origin": el_origin, "rotation": el_rot,
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
    if program is not None and getattr(program, "source_map", None) is not None:
        doc["synccraft_source_map"] = program.source_map.to_dict()
    return doc
