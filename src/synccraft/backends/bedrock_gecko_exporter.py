"""Bedrock Entity 1.12.0 / 1.8.0 (Cobblemon-compatible) and GeckoLib 4 Exporters.

Serializes validated ``Scene`` hierarchies (and optional ``Program`` ``@behavior``
musculoskeletal + stance-IK clips) into:
- ``geometry``: ``format_version: "1.12.0"`` ``minecraft:geometry`` JSON
- ``animations``: ``format_version: "1.8.0"`` ``animations`` JSON (baked keyframes
  or ``query.anim_time`` Molang channels)

Enforces strict target-profile matching, cumulative unrotated bone pivots,
semantic 3D ``locators`` export, and an independent post-serialization 8-corner
world vertex re-audit (``max_err <= 1e-6``). Does NOT claim live Minecraft or
Cobblemon GUI client import verification (``runtime_import_verified: False``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from synccraft.kinematics.se3_chain import euler_xyz_matrix

from .common import (
    FACES,
    ExportError,
    face_uv,
    validate_atlas,
    validate_scene_for_export,
    vector,
)

_CORNER_SIGNS = np.asarray(
    (
        (-1.0, -1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0),
    ),
    dtype=np.float64,
)


def _infer_locator_role_hint(raw_name: str) -> str:
    low = raw_name.lower()
    if any(tok in low for tok in ("seat", "mount", "rider", "saddle")):
        return "rider_seat"
    if any(tok in low for tok in ("eye", "head", "camera", "look", "focus", "gaze")):
        return "camera_focus"
    if any(tok in low for tok in ("target", "hitbox", "hurtbox")):
        return "hitbox"
    if any(tok in low for tok in ("foot", "paw", "talon", "hand", "claw", "ik", "effector", "toe", "hoof")):
        return "ik_effector"
    if any(tok in low for tok in ("effect", "special", "breath", "muzzle", "flame", "beam", "vfx", "emitter")):
        return "vfx_emitter"
    return "socket"


def _reaudit_serialized_bedrock_geometry(geo_doc: dict[str, Any], scene: Any) -> float:
    """Independently reconstruct 8-corner world vertices from serialized .geo.json and verify < 1e-6 drift."""
    bones_json = geo_doc["minecraft:geometry"][0]["bones"]
    piv_by_bone: dict[str, np.ndarray] = {}
    world_piv_by_bone: dict[str, np.ndarray] = {}
    world_rot_by_bone: dict[str, np.ndarray] = {}

    reconstructed_list: list[np.ndarray] = []
    for rec in bones_json:
        bname = str(rec["name"])
        parent_name = rec.get("parent")
        piv = np.asarray(rec.get("pivot", [0.0, 0.0, 0.0]), dtype=np.float64)
        rot_e = tuple(float(x) for x in rec.get("rotation", [0.0, 0.0, 0.0]))
        piv_by_bone[bname] = piv
        if parent_name is None:
            w_piv = piv.copy()
            w_rot = euler_xyz_matrix(rot_e)
        else:
            p_piv = piv_by_bone[str(parent_name)]
            pw_piv = world_piv_by_bone[str(parent_name)]
            pw_rot = world_rot_by_bone[str(parent_name)]
            w_piv = pw_piv + pw_rot @ (piv - p_piv)
            w_rot = pw_rot @ euler_xyz_matrix(rot_e)
        world_piv_by_bone[bname] = w_piv
        world_rot_by_bone[bname] = w_rot

        for cube in rec.get("cubes") or []:
            origin = np.asarray(cube["origin"], dtype=np.float64)
            sz = np.asarray(cube["size"], dtype=np.float64)
            ctr = origin + 0.5 * sz
            loc_c = ctr - piv
            w_ctr = w_piv + w_rot @ loc_c
            corners = (_CORNER_SIGNS * (sz * 0.5)) @ w_rot.T + w_ctr
            reconstructed_list.append(corners)

    # Match scene boxes in bone emission order
    expected_boxes_ordered: list[Any] = []
    boxes_by_bone: dict[str, list[Any]] = {b.name: [] for b in scene.bones}
    for bx in scene.boxes:
        boxes_by_bone[bx.bone_name].append(bx)
    for b in scene.bones:
        expected_boxes_ordered.extend(boxes_by_bone[b.name])

    if len(reconstructed_list) != len(expected_boxes_ordered):
        raise ExportError(
            f"Serialized cube count {len(reconstructed_list)} disagrees with scene box count {len(expected_boxes_ordered)}"
        )

    max_err = 0.0
    for recon_c, w_box in zip(reconstructed_list, expected_boxes_ordered):
        exp_c = w_box.corners(include_inflate=False)
        err = float(np.max(np.linalg.norm(recon_c - exp_c, axis=-1)))
        max_err = max(max_err, err)

    if max_err > 1e-6:
        raise ExportError(f"Post-serialization 8-corner re-audit failed: max_err={max_err:.8g} > 1e-6")
    return max_err


def _serialize_behavior_animations(
    program: Any,
    scene: Any,
    *,
    short_asset_id: str,
    num_animation_frames: int,
    animation_mode: str,
) -> dict[str, Any] | None:
    if program is None or getattr(program, "behavior", None) is None or not program.behavior.clips:
        return None
    if animation_mode not in ("keyframes", "molang"):
        raise ExportError(f"Unsupported animation_mode {animation_mode!r}; expected 'keyframes' or 'molang'")

    from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip

    n_frames = max(4, int(num_animation_frames))
    root_bone_name = scene.bones[0].name

    # Identify bones touched by muscles, tendon couplings, or IK chains so molang mode bakes those accurately
    coupled_bones: set[str] = set()
    for m in getattr(program, "muscles", ()):
        coupled_bones.add(m.insertion_bone)
        if m.tendon_coupling is not None:
            coupled_bones.add(m.tendon_coupling.bone)
    if getattr(program, "kinematics", None) is not None:
        for ikc in program.kinematics.ik_chains:
            coupled_bones.add(ikc.upper)
            coupled_bones.add(ikc.lower)

    animations_dict: dict[str, Any] = {}
    axis_order = ("x", "y", "z")

    for clip in program.behavior.clips:
        samples = evaluate_behavior_clip(program, clip.name, num_frames=n_frames)
        bones_payload: dict[str, dict[str, Any]] = {}

        # Root translation keyframes when forward gait / heave is non-zero
        if any(math.hypot(s.root_translation_delta[0], math.hypot(s.root_translation_delta[1], s.root_translation_delta[2])) > 1e-5 for s in samples):
            pos_kfs: dict[str, list[float]] = {}
            for s in samples:
                t_key = f"{round(float(s.time_s), 4):.4f}".rstrip("0").rstrip(".")
                if "." not in t_key:
                    t_key = f"{t_key}.0"
                pos_kfs[t_key] = vector([round(float(x), 4) for x in s.root_translation_delta])
            if clip.loop and not any(abs(d.stride) > 1e-6 for d in clip.ik_drives):
                end_key = f"{round(float(clip.duration_s), 4):.4f}".rstrip("0").rstrip(".")
                if "." not in end_key:
                    end_key = f"{end_key}.0"
                pos_kfs[end_key] = vector([round(float(x), 4) for x in samples[0].root_translation_delta])
            bones_payload.setdefault(root_bone_name, {})["position"] = pos_kfs

        tracks_by_bone: dict[str, list[Any]] = {}
        for tr in clip.tracks:
            tracks_by_bone.setdefault(tr.bone, []).append(tr)

        for bone in scene.bones:
            bname = bone.name
            if bname == root_bone_name:
                continue

            b_tracks = tracks_by_bone.get(bname, [])
            can_emit_pure_molang = (
                animation_mode == "molang"
                and bool(b_tracks)
                and bname not in coupled_bones
                and all(tr.law == "wave" for tr in b_tracks)
                and len({tr.axis for tr in b_tracks}) == len(b_tracks)
            )
            if can_emit_pure_molang:
                molang_vec: list[Any] = [0.0, 0.0, 0.0]
                for tr in b_tracks:
                    ax_i = axis_order.index(tr.axis)
                    deg_per_s = round(float(tr.freq_hz) * 360.0, 4)
                    if abs(tr.amp_deg) < 1e-6:
                        molang_vec[ax_i] = round(float(tr.bias_deg), 4)
                    else:
                        phase_part = ""
                        if abs(tr.phase_deg) > 1e-6:
                            sign_s = "+" if tr.phase_deg >= 0 else "-"
                            phase_part = f" {sign_s} {round(abs(float(tr.phase_deg)), 4)}"
                        core = f"math.sin(query.anim_time * {deg_per_s}{phase_part}) * {round(float(tr.amp_deg), 4)}"
                        if abs(tr.bias_deg) > 1e-6:
                            b_sign = "+" if tr.bias_deg >= 0 else "-"
                            core = f"{core} {b_sign} {round(abs(float(tr.bias_deg)), 4)}"
                        molang_vec[ax_i] = core
                bones_payload.setdefault(bname, {})["rotation"] = molang_vec
                continue

            # Baked exact SE(3) / Hill-muscle / stance-IK rotation keyframes
            frame_vecs = [s.pose_euler_deltas.get(bname, (0.0, 0.0, 0.0)) for s in samples]
            max_abs = max(max(abs(v[0]), abs(v[1]), abs(v[2])) for v in frame_vecs)
            if max_abs < 1e-4:
                continue

            rot_kfs: dict[str, list[float]] = {}
            for s, vec_d in zip(samples, frame_vecs):
                t_key = f"{round(float(s.time_s), 4):.4f}".rstrip("0").rstrip(".")
                if "." not in t_key:
                    t_key = f"{t_key}.0"
                rot_kfs[t_key] = vector([round(float(vec_d[0]), 4), round(float(vec_d[1]), 4), round(float(vec_d[2]), 4)])

            if clip.loop:
                end_key = f"{round(float(clip.duration_s), 4):.4f}".rstrip("0").rstrip(".")
                if "." not in end_key:
                    end_key = f"{end_key}.0"
                v0 = frame_vecs[0]
                rot_kfs[end_key] = vector([round(float(v0[0]), 4), round(float(v0[1]), 4), round(float(v0[2]), 4)])

            bones_payload.setdefault(bname, {})["rotation"] = rot_kfs

        full_clip_key = clip.name if clip.name.startswith("animation.") else f"animation.{short_asset_id}.{clip.name}"
        animations_dict[full_clip_key] = {
            "loop": bool(clip.loop),
            "animation_length": round(float(clip.duration_s), 4),
            "bones": bones_payload,
        }

    return {
        "format_version": "1.8.0",
        "animations": animations_dict,
    }


def _export_bedrock_or_gecko_common(
    scene: Any,
    *,
    expected_profile: str,
    program: Any | None,
    atlas: dict[str, Any] | None,
    identifier: str,
    profile: str | None,
    num_animation_frames: int,
    animation_mode: str,
) -> dict[str, Any]:
    validate_scene_for_export(scene)
    target = profile or getattr(getattr(scene, "meta", None), "target_profile", None)
    if target != expected_profile:
        raise ExportError(
            f"Exporter for {expected_profile!r} rejected scene with target_profile={target!r}; "
            f"set meta.target_profile={expected_profile!r} or pass explicit profile={expected_profile!r}"
        )

    if atlas is not None:
        tex_w, tex_h = validate_atlas(atlas, scene, embedded=False)
    else:
        tex_w, tex_h = 64, 64

    origins: dict[str, np.ndarray] = {}
    for bone in scene.bones:
        parent_origin = (
            origins[bone.parent_name]
            if bone.parent_name is not None
            else np.zeros(3, dtype=np.float64)
        )
        origins[bone.name] = parent_origin + np.asarray(bone.local_translation, dtype=np.float64)

    boxes_by_bone: dict[str, list[Any]] = {b.name: [] for b in scene.bones}
    for box in scene.boxes:
        boxes_by_bone[box.bone_name].append(box)

    locs_by_bone: dict[str, list[Any]] = {b.name: [] for b in scene.bones}
    for loc in getattr(scene, "locators", ()):
        if loc.bone_name in locs_by_bone:
            locs_by_bone[loc.bone_name].append(loc)

    used_loc_keys: set[str] = set()
    bones_out: list[dict[str, Any]] = []

    for bone in scene.bones:
        b_entry: dict[str, Any] = {"name": bone.name}
        if bone.parent_name is not None:
            b_entry["parent"] = bone.parent_name
        b_entry["pivot"] = vector(origins[bone.name])
        rot_vec = vector(bone.local_euler)
        if any(abs(x) > 1e-9 for x in rot_vec):
            b_entry["rotation"] = rot_vec

        cubes_out: list[dict[str, Any]] = []
        for box in boxes_by_bone[bone.name]:
            ctr = origins[bone.name] + np.asarray(box.local_center, dtype=np.float64)
            half = 0.5 * np.asarray(box.size, dtype=np.float64)
            cube_entry: dict[str, Any] = {
                "origin": vector(ctr - half),
                "size": [int(box.size[0]), int(box.size[1]), int(box.size[2])],
            }
            if float(box.inflate) > 0.0:
                cube_entry["inflate"] = round(float(box.inflate), 6)
            if atlas is not None:
                uv_faces: dict[str, dict[str, list[float]]] = {}
                for face in FACES:
                    u0, v0, u1, v1 = face_uv(atlas, box.name, face, normalized=False)
                    uv_faces[face] = {
                        "uv": vector([u0, v0]),
                        "uv_size": vector([u1 - u0, v1 - v0]),
                    }
                cube_entry["uv"] = uv_faces
            else:
                cube_entry["uv"] = [0, 0]
            cubes_out.append(cube_entry)

        if cubes_out:
            b_entry["cubes"] = cubes_out

        loc_dict: dict[str, Any] = {}
        for loc in locs_by_bone[bone.name]:
            is_synthetic_pivot = (
                loc.role == "socket"
                and np.allclose(loc.local_offset, 0.0, atol=1e-9)
                and (loc.name == f"{bone.name}_pivot" or loc.name == "bedrock_origin")
            )
            if is_synthetic_pivot:
                continue
            base_key = loc.name[4:] if (loc.name.startswith("loc_") and len(loc.name) > 4) else loc.name
            cand_key = base_key
            suffix = 2
            while cand_key in used_loc_keys:
                cand_key = f"{base_key}_{suffix}"
                suffix += 1
            used_loc_keys.add(cand_key)

            pos_vec = vector(origins[bone.name] + np.asarray(loc.local_offset, dtype=np.float64))
            if _infer_locator_role_hint(cand_key) == loc.role:
                loc_dict[cand_key] = pos_vec
            else:
                loc_dict[cand_key] = {"offset": pos_vec, "role": loc.role}

        if loc_dict:
            b_entry["locators"] = loc_dict

        bones_out.append(b_entry)

    all_corners = np.concatenate([bx.corners(include_inflate=True) for bx in scene.boxes], axis=0)
    vis_w = round(max(1.0, float(2.0 * np.max(np.abs(all_corners[:, [0, 2]]))) / 16.0 + 0.5), 3)
    min_y_u, max_y_u = float(np.min(all_corners[:, 1])), float(np.max(all_corners[:, 1]))
    vis_h = round(max(1.0, (max_y_u - min_y_u) / 16.0 + 0.5), 3)
    vis_off_y = round(0.5 * (min_y_u + max_y_u) / 16.0, 3)

    geo_id = identifier if identifier.startswith("geometry.") else f"geometry.synccraft.{identifier}"
    geometry_doc: dict[str, Any] = {
        "format_version": "1.12.0",
        "minecraft:geometry": [
            {
                "description": {
                    "identifier": geo_id,
                    "texture_width": tex_w,
                    "texture_height": tex_h,
                    "visible_bounds_width": vis_w,
                    "visible_bounds_height": vis_h,
                    "visible_bounds_offset": [0.0, vis_off_y, 0.0],
                },
                "bones": bones_out,
            }
        ],
    }

    max_err = _reaudit_serialized_bedrock_geometry(geometry_doc, scene)
    short_id = geo_id.split(".")[-1] or "entity"
    animations_doc = _serialize_behavior_animations(
        program,
        scene,
        short_asset_id=short_id,
        num_animation_frames=num_animation_frames,
        animation_mode=animation_mode,
    )

    return {
        "target_profile": expected_profile,
        "runtime_import_verified": False,
        "serialized_vertex_max_error": round(float(max_err), 12),
        "geometry": geometry_doc,
        "animations": animations_doc,
    }


def export_bedrock(
    scene: Any,
    *,
    program: Any | None = None,
    atlas: dict[str, Any] | None = None,
    identifier: str = "geometry.synccraft.entity",
    profile: str | None = None,
    num_animation_frames: int = 12,
    animation_mode: str = "keyframes",
) -> dict[str, Any]:
    """Export Bedrock 1.12.0 .geo.json (+ 1.8.0 .animation.json when ``program`` has ``@behavior``)."""
    return _export_bedrock_or_gecko_common(
        scene,
        expected_profile="bedrock_entity",
        program=program,
        atlas=atlas,
        identifier=identifier,
        profile=profile,
        num_animation_frames=num_animation_frames,
        animation_mode=animation_mode,
    )


def export_geckolib(
    scene: Any,
    *,
    program: Any | None = None,
    atlas: dict[str, Any] | None = None,
    identifier: str = "geometry.synccraft.geckolib_asset",
    profile: str | None = None,
    num_animation_frames: int = 12,
    animation_mode: str = "keyframes",
) -> dict[str, Any]:
    """Export GeckoLib 4 1.12.0 .geo.json (+ 1.8.0 .animation.json when ``program`` has ``@behavior``)."""
    return _export_bedrock_or_gecko_common(
        scene,
        expected_profile="geckolib",
        program=program,
        atlas=atlas,
        identifier=identifier,
        profile=profile,
        num_animation_frames=num_animation_frames,
        animation_mode=animation_mode,
    )
