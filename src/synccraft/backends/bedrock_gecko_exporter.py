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

_MX_REFLECT = np.diag((-1.0, 1.0, 1.0))

# SyncCraft corner signs s_sync = (sx, sy, sz) map under M_x reflection to Bedrock
# local corner signs (-sx, sy, sz): M_x (w_ctr_sync + W_sync (s_sync * h)) == w_ctr_bed + W_bed (s_bed * h).
_BEDROCK_CORNER_SIGNS_FOR_SYNC_ORDER = np.asarray(
    tuple((-float(s[0]), float(s[1]), float(s[2])) for s in _CORNER_SIGNS),
    dtype=np.float64,
)


def sync_point_to_bedrock(pt_sync: Any) -> np.ndarray:
    """Convert right-handed SyncCraft point/pivot/locator/position (x, y, z) to Bedrock (-x, y, z)."""
    p = np.asarray(pt_sync, dtype=np.float64)
    return np.array((-float(p[0]), float(p[1]), float(p[2])), dtype=np.float64)


def bedrock_point_to_sync(pt_bedrock: Any) -> np.ndarray:
    """Convert Bedrock point/pivot/locator/position (bx, by, bz) to right-handed SyncCraft (-bx, by, bz)."""
    p = np.asarray(pt_bedrock, dtype=np.float64)
    return np.array((-float(p[0]), float(p[1]), float(p[2])), dtype=np.float64)


def sync_cube_from_to_bedrock_origin(from_sync: Any, size_xyz: Any) -> np.ndarray:
    """Convert SyncCraft cube min-corner ``from_sync`` and ``size_xyz`` to Bedrock cube ``origin`` (C5).

    Blockbench / Bedrock 1.12.0 negates the X interval ``[from_x, from_x + size_x]`` into
    ``[-(from_x + size_x), -from_x]``, so ``bedrock_origin_x = -(sync_from_x + size_x)``.
    """
    fr = np.asarray(from_sync, dtype=np.float64)
    sz = np.asarray(size_xyz, dtype=np.float64)
    return np.array((-(float(fr[0]) + float(sz[0])), float(fr[1]), float(fr[2])), dtype=np.float64)


def bedrock_origin_to_sync_cube_from(origin_bedrock: Any, size_xyz: Any) -> np.ndarray:
    """Invert Bedrock 1.12.0 cube ``origin`` back to SyncCraft cube min-corner ``sync_from`` (C5)."""
    ob = np.asarray(origin_bedrock, dtype=np.float64)
    sz = np.asarray(size_xyz, dtype=np.float64)
    return np.array((-(float(ob[0]) + float(sz[0])), float(ob[1]), float(ob[2])), dtype=np.float64)


def bedrock_rotation_matrix(euler_bedrock_deg: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Evaluate Bedrock 1.12.0 / 1.8.0 left-handed bone/cube rotation matrix R_bed(bx, by, bz).

    In Blockbench's Bedrock codec, right-handed ZYX/intrinsic-XYZ Euler ``(rx, ry, rz)`` maps to
    Bedrock ``(bx, by, bz) = (-rx, -ry, rz)``. Under X-reflection ``M_x = diag(-1, 1, 1)``:
        M_x @ euler_xyz_matrix((rx, ry, rz)) @ M_x == euler_xyz_matrix((rx, -ry, -rz))
                                                   == euler_xyz_matrix((-bx, by, -bz)).
    """
    bx, by, bz = (float(x) for x in euler_bedrock_deg)
    return euler_xyz_matrix((-bx, by, -bz))


def sync_euler_to_bedrock(euler_sync_deg: Any, *, verify_o3: bool = True) -> tuple[float, float, float]:
    """Conjugate right-handed SyncCraft XYZ Euler angles across X-reflection into Bedrock (-rx, -ry, rz)."""
    rx, ry, rz = (float(x) for x in euler_sync_deg)
    bed = (-rx, -ry, rz)
    if verify_o3:
        r_sync = euler_xyz_matrix((rx, ry, rz))
        r_conj = _MX_REFLECT @ r_sync @ _MX_REFLECT
        r_bed = bedrock_rotation_matrix(bed)
        if not np.allclose(r_conj, r_bed, atol=1e-12, rtol=0.0):
            raise ExportError("O(3) X-reflection conjugation mismatch in sync_euler_to_bedrock")
    return bed


def bedrock_euler_to_sync(euler_bedrock_deg: Any, *, verify_o3: bool = True) -> tuple[float, float, float]:
    """Conjugate Bedrock 1.12.0 / 1.8.0 Euler angles (bx, by, bz) back to SyncCraft (-bx, -by, bz)."""
    bx, by, bz = (float(x) for x in euler_bedrock_deg)
    sync_e = (-bx, -by, bz)
    if verify_o3:
        r_bed = bedrock_rotation_matrix((bx, by, bz))
        r_sync = euler_xyz_matrix(sync_e)
        if not np.allclose(_MX_REFLECT @ r_bed @ _MX_REFLECT, r_sync, atol=1e-12, rtol=0.0):
            raise ExportError("O(3) X-reflection conjugation mismatch in bedrock_euler_to_sync")
    return sync_e


def _format_size_value(val: float) -> int | float:
    f = float(val)
    r = round(f)
    return int(r) if abs(f - r) <= 1e-9 else round(f, 6)


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
    """Independently evaluate Bedrock 1.12.0 left-handed kinematics on serialized .geo.json and verify < 1e-6 drift."""
    bones_json = geo_doc["minecraft:geometry"][0]["bones"]
    piv_by_bone: dict[str, np.ndarray] = {}
    world_piv_by_bone: dict[str, np.ndarray] = {}
    world_rot_by_bone: dict[str, np.ndarray] = {}

    reconstructed_list: list[tuple[np.ndarray, np.ndarray]] = []
    for rec in bones_json:
        bname = str(rec["name"])
        parent_name = rec.get("parent")
        piv_bed = np.asarray(rec.get("pivot", [0.0, 0.0, 0.0]), dtype=np.float64)
        rot_bed = tuple(float(x) for x in rec.get("rotation", [0.0, 0.0, 0.0]))
        piv_by_bone[bname] = piv_bed
        r_local_bed = bedrock_rotation_matrix(rot_bed)
        if parent_name is None:
            w_piv_bed = piv_bed.copy()
            w_rot_bed = r_local_bed
        else:
            p_piv_bed = piv_by_bone[str(parent_name)]
            pw_piv_bed = world_piv_by_bone[str(parent_name)]
            pw_rot_bed = world_rot_by_bone[str(parent_name)]
            w_piv_bed = pw_piv_bed + pw_rot_bed @ (piv_bed - p_piv_bed)
            w_rot_bed = pw_rot_bed @ r_local_bed
        world_piv_by_bone[bname] = w_piv_bed
        world_rot_by_bone[bname] = w_rot_bed

        for cube in rec.get("cubes") or []:
            origin_bed = np.asarray(cube["origin"], dtype=np.float64)
            sz = np.asarray(cube["size"], dtype=np.float64)
            inf = float(cube.get("inflate", 0.0))
            ctr_bed = origin_bed + 0.5 * sz
            if "rotation" in cube and any(abs(float(x)) > 1e-9 for x in cube["rotation"]):
                c_rot_bed = tuple(float(x) for x in cube["rotation"])
                c_piv_bed = np.asarray(cube.get("pivot", ctr_bed.tolist()), dtype=np.float64)
                r_cube_bed = bedrock_rotation_matrix(c_rot_bed)
                w_cube_rot_bed = w_rot_bed @ r_cube_bed
                w_ctr_bed = w_piv_bed + w_rot_bed @ (c_piv_bed - piv_bed) + w_cube_rot_bed @ (ctr_bed - c_piv_bed)
            else:
                w_cube_rot_bed = w_rot_bed
                w_ctr_bed = w_piv_bed + w_rot_bed @ (ctr_bed - piv_bed)

            corners_bed_raw = (_BEDROCK_CORNER_SIGNS_FOR_SYNC_ORDER * (sz * 0.5)) @ w_cube_rot_bed.T + w_ctr_bed
            corners_bed_inf = (_BEDROCK_CORNER_SIGNS_FOR_SYNC_ORDER * (sz * 0.5 + inf)) @ w_cube_rot_bed.T + w_ctr_bed
            # Map Bedrock world space (-X, +Y, +Z) back to SyncCraft right-handed world space (+X, +Y, +Z)
            corners_sync_raw = corners_bed_raw @ _MX_REFLECT
            corners_sync_inf = corners_bed_inf @ _MX_REFLECT
            reconstructed_list.append((corners_sync_raw, corners_sync_inf))

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
    for (recon_raw, recon_inf), w_box in zip(reconstructed_list, expected_boxes_ordered):
        exp_raw = w_box.corners(include_inflate=False)
        exp_inf = w_box.corners(include_inflate=True)
        err_raw = float(np.max(np.linalg.norm(recon_raw - exp_raw, axis=-1)))
        err_inf = float(np.max(np.linalg.norm(recon_inf - exp_inf, axis=-1)))
        max_err = max(max_err, err_raw, err_inf)

    if max_err > 1e-6:
        raise ExportError(f"Post-serialization 8-corner re-audit failed: max_err={max_err:.8g} > 1e-6")
    return max_err


def _extract_so3_euler_keyframe_deltas(samples: list[Any], rest_bone: Any) -> list[tuple[float, float, float]]:
    """Recover exact SO(3) matrix-composed Euler deltas ``e_posed - e_rest`` with continuous frame unwrapping.

    Bedrock/GeckoLib/Blockbench evaluate animated bone orientation as ``R_xyz(e_rest + kf)``.
    Using ``posed_bone.local_euler - rest_bone.local_euler`` guarantees
    ``R_xyz(e_rest + kf) == R_rest @ R_delta_clamped`` (preserving joint_limits and avoiding
    Euler-sum cross-axis distortion on rotated bind bones).
    """
    bname = rest_bone.name
    rest_e = np.asarray(rest_bone.local_euler, dtype=np.float64)
    prev = np.zeros(3, dtype=np.float64)
    out: list[tuple[float, float, float]] = []
    for s in samples:
        raw_delta_req = s.pose_euler_deltas.get(bname)
        if raw_delta_req is None or np.allclose(raw_delta_req, 0.0, atol=1e-12):
            cur = prev + (((-prev) + 180.0) % 360.0 - 180.0)
        else:
            posed_b = s.scene.bone_by_name[bname]
            req_arr = np.asarray(raw_delta_req, dtype=np.float64)
            parent_rot = s.scene.bone_by_name[posed_b.parent_name].rotation if posed_b.parent_name else np.eye(3)
            local_rot = parent_rot.T @ posed_b.rotation
            if np.allclose(euler_xyz_matrix(rest_e + req_arr), local_rot, atol=1e-7):
                cur = req_arr.copy()
            else:
                posed_e = np.asarray(posed_b.local_euler, dtype=np.float64)
                raw_d = posed_e - rest_e
                cur = prev + (((raw_d - prev) + 180.0) % 360.0 - 180.0)
        prev = cur
        out.append((float(cur[0]), float(cur[1]), float(cur[2])))
    return out


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
    from synccraft.dsl.ast_nodes import validate_program

    validate_program(program)
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
            bones_seq = getattr(ikc, "bones", ()) or ()
            if bones_seq:
                coupled_bones.update(bones_seq)
            if getattr(ikc, "upper", ""):
                coupled_bones.add(ikc.upper)
            if getattr(ikc, "lower", ""):
                coupled_bones.add(ikc.lower)
        for sp in getattr(program.kinematics, "springs", ()):
            coupled_bones.add(sp.bone)
    for sp in getattr(program.behavior, "springs", ()):
        coupled_bones.add(sp.bone)

    animations_dict: dict[str, Any] = {}
    axis_order = ("x", "y", "z")

    def _fmt_time_key(t_val: float) -> str:
        t_key = f"{round(float(t_val), 4):.4f}".rstrip("0").rstrip(".")
        return f"{t_key}.0" if "." not in t_key else t_key

    for clip in program.behavior.clips:
        samples = evaluate_behavior_clip(program, clip.name, num_frames=n_frames)
        bones_payload: dict[str, dict[str, Any]] = {}
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
                ch_dict: dict[str, Any] = {}
                is_catmull = getattr(kf_tr, "interpolation", "linear") == "catmullrom"
                for t_s, vec_v in kf_tr.keys:
                    if kf_tr.channel == "rotation":
                        bed_v = sync_euler_to_bedrock(vec_v)
                    elif kf_tr.channel == "position":
                        p_b = sync_point_to_bedrock(vec_v)
                        bed_v = (float(p_b[0]), float(p_b[1]), float(p_b[2]))
                    else:
                        bed_v = (float(vec_v[0]), float(vec_v[1]), float(vec_v[2]))
                    vec_fmt = vector([round(bed_v[0], 4), round(bed_v[1], 4), round(bed_v[2], 4)])
                    ch_dict[_fmt_time_key(t_s)] = (
                        {"post": vec_fmt, "lerp_mode": "catmullrom"} if is_catmull else vec_fmt
                    )
                if ch_dict:
                    bones_payload.setdefault(kf_tr.bone, {})[kf_tr.channel] = ch_dict

        # Root translation keyframes when forward gait / heave is non-zero (Bedrock negates position X!)
        is_in_place = bool(
            getattr(clip, "in_place", False)
            or any(getattr(g, "in_place", False) for g in getattr(clip, "gaits", ()))
            or any(getattr(fl, "in_place", False) for fl in getattr(clip, "flights", ()))
            or any(getattr(sw, "in_place", False) for sw in getattr(clip, "swims", ()))
        )
        has_forward_locomotion = (
            not is_in_place
            and (any(abs(d.stride) > 1e-6 for d in clip.ik_drives) or bool(getattr(clip, "gaits", ())))
        )
        def _root_total_pos(s: Any) -> tuple[float, float, float]:
            r = s.root_translation_delta
            p = s.pose_position_deltas.get(root_bone_name, (0.0, 0.0, 0.0))
            return (float(r[0]) + float(p[0]), float(r[1]) + float(p[1]), float(r[2]) + float(p[2]))

        if (root_bone_name, "position") not in explicit_kf_channels and any(
            math.hypot(_root_total_pos(s)[0], math.hypot(_root_total_pos(s)[1], _root_total_pos(s)[2])) > 1e-5
            for s in samples
        ):
            pos_kfs: dict[str, list[float]] = {}
            for s in samples:
                p_bed = sync_point_to_bedrock(_root_total_pos(s))
                pos_kfs[_fmt_time_key(s.time_s)] = vector([round(float(x), 4) for x in p_bed])
            if clip.loop and not has_forward_locomotion:
                p0_bed = sync_point_to_bedrock(_root_total_pos(samples[0]))
                pos_kfs[_fmt_time_key(clip.duration_s)] = vector([round(float(x), 4) for x in p0_bed])
            bones_payload.setdefault(root_bone_name, {})["position"] = pos_kfs

        tracks_by_bone: dict[str, list[Any]] = {}
        for tr in clip.tracks:
            tracks_by_bone.setdefault(tr.bone, []).append(tr)

        for bone in scene.bones:
            bname = bone.name
            if (bname, "rotation") in explicit_kf_channels:
                continue

            b_tracks = tracks_by_bone.get(bname, [])
            can_emit_pure_molang = (
                animation_mode == "molang"
                and bool(b_tracks)
                and bname not in coupled_bones
                and all(tr.law in ("wave", "spin") for tr in b_tracks)
                and len({tr.axis for tr in b_tracks}) == len(b_tracks)
            )
            if can_emit_pure_molang:
                molang_vec: list[Any] = [0.0, 0.0, 0.0]
                for tr in b_tracks:
                    ax_i = axis_order.index(tr.axis)
                    # Bedrock 1.8.0 negates X and Y rotation angles relative to right-handed SyncCraft (C5)
                    axis_sign = -1.0 if tr.axis in ("x", "y") else 1.0
                    amp_bed = axis_sign * float(tr.amp_deg)
                    bias_bed = axis_sign * float(tr.bias_deg)
                    if tr.law == "spin":
                        rate_deg_s = round(amp_bed * float(tr.freq_hz), 4)
                        const_off = round(bias_bed + amp_bed * (float(tr.phase_deg) / 360.0), 4)
                        if abs(rate_deg_s) < 1e-6:
                            molang_vec[ax_i] = const_off
                        else:
                            core = f"query.anim_time * {rate_deg_s}"
                            if abs(const_off) > 1e-6:
                                c_sign = "+" if const_off >= 0 else "-"
                                core = f"{core} {c_sign} {round(abs(const_off), 4)}"
                            molang_vec[ax_i] = core
                        continue
                    deg_per_s = round(float(tr.freq_hz) * 360.0, 4)
                    if abs(amp_bed) < 1e-6:
                        molang_vec[ax_i] = round(bias_bed, 4)
                    else:
                        phase_part = ""
                        if abs(tr.phase_deg) > 1e-6:
                            sign_s = "+" if tr.phase_deg >= 0 else "-"
                            phase_part = f" {sign_s} {round(abs(float(tr.phase_deg)), 4)}"
                        core = f"math.sin(query.anim_time * {deg_per_s}{phase_part}) * {round(amp_bed, 4)}"
                        if abs(bias_bed) > 1e-6:
                            b_sign = "+" if bias_bed >= 0 else "-"
                            core = f"{core} {b_sign} {round(abs(bias_bed), 4)}"
                        molang_vec[ax_i] = core
                bones_payload.setdefault(bname, {})["rotation"] = molang_vec
                continue

            # Baked exact SE(3) / Hill-muscle / stance-IK rotation keyframes (Bedrock negates X & Y!)
            frame_vecs = _extract_so3_euler_keyframe_deltas(samples, bone)
            max_abs = max(max(abs(v[0]), abs(v[1]), abs(v[2])) for v in frame_vecs)
            if max_abs < 1e-4:
                continue

            rot_kfs: dict[str, list[float]] = {}
            for s, vec_d in zip(samples, frame_vecs):
                bed_d = sync_euler_to_bedrock(vec_d, verify_o3=False)
                rot_kfs[_fmt_time_key(s.time_s)] = vector([round(bed_d[0], 4), round(bed_d[1], 4), round(bed_d[2], 4)])

            if clip.loop:
                v_last = np.asarray(frame_vecs[-1], dtype=np.float64)
                if any(tr.law == "spin" for tr in b_tracks) and len(frame_vecs) >= 2:
                    v_wrap = v_last + (v_last - np.asarray(frame_vecs[-2], dtype=np.float64))
                else:
                    v0_arr = np.asarray(frame_vecs[0], dtype=np.float64)
                    v_wrap = v_last + (((v0_arr - v_last) + 180.0) % 360.0 - 180.0)
                bed_wrap = sync_euler_to_bedrock(v_wrap, verify_o3=False)
                rot_kfs[_fmt_time_key(clip.duration_s)] = vector([round(bed_wrap[0], 4), round(bed_wrap[1], 4), round(bed_wrap[2], 4)])

            bones_payload.setdefault(bname, {})["rotation"] = rot_kfs

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
            pos_kfs_b: dict[str, list[float]] = {}
            x_sign = -1.0 if bone.mirrored else 1.0
            for s in samples:
                dx, dy, dz = s.pose_position_deltas.get(bname, (0.0, 0.0, 0.0))
                p_bed = sync_point_to_bedrock((x_sign * float(dx), float(dy), float(dz)))
                pos_kfs_b[_fmt_time_key(s.time_s)] = vector([round(float(x), 4) for x in p_bed])
            if clip.loop:
                dx0, dy0, dz0 = samples[0].pose_position_deltas.get(bname, (0.0, 0.0, 0.0))
                p0_bed = sync_point_to_bedrock((x_sign * float(dx0), float(dy0), float(dz0)))
                pos_kfs_b[_fmt_time_key(clip.duration_s)] = vector([round(float(x), 4) for x in p0_bed])
            bones_payload.setdefault(bname, {})["position"] = pos_kfs_b

        # Export Sim(3) bone scale channels (volume-preserving muscle bulge / breathing)
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
            scl_kfs_b: dict[str, list[float]] = {}
            for s in samples:
                sx, sy, sz = s.pose_scale_factors.get(bname, (1.0, 1.0, 1.0))
                scl_kfs_b[_fmt_time_key(s.time_s)] = vector([round(float(sx), 4), round(float(sy), 4), round(float(sz), 4)])
            if clip.loop:
                sx0, sy0, sz0 = samples[0].pose_scale_factors.get(bname, (1.0, 1.0, 1.0))
                scl_kfs_b[_fmt_time_key(clip.duration_s)] = vector([round(float(sx0), 4), round(float(sy0), 4), round(float(sz0), 4)])
            bones_payload.setdefault(bname, {})["scale"] = scl_kfs_b

        full_clip_key = clip.name if clip.name.startswith("animation.") else f"animation.{short_asset_id}.{clip.name}"
        clip_entry: dict[str, Any] = {
            "loop": bool(clip.loop),
            "animation_length": round(float(clip.duration_s), 4),
            "bones": bones_payload,
        }
        if getattr(clip, "effects", ()):
            from synccraft.kinematics.musculoskeletal import resolve_clip_effect_events

            eff_events = resolve_clip_effect_events(program, clip)
            sound_fx: dict[str, Any] = {}
            particle_fx: dict[str, Any] = {}
            timeline_fx: dict[str, Any] = {}
            for ev in eff_events:
                tk = _fmt_time_key(float(ev["time_s"]))
                if ev.get("sound"):
                    entry_s = {"effect": ev["sound"]}
                    if tk in sound_fx:
                        cur_s = sound_fx[tk]
                        sound_fx[tk] = [*cur_s, entry_s] if isinstance(cur_s, list) else [cur_s, entry_s]
                    else:
                        sound_fx[tk] = entry_s
                if ev.get("particle"):
                    entry_p: dict[str, Any] = {"effect": ev["particle"]}
                    if ev.get("locator"):
                        entry_p["locator"] = ev["locator"]
                    if tk in particle_fx:
                        cur_p = particle_fx[tk]
                        particle_fx[tk] = [*cur_p, entry_p] if isinstance(cur_p, list) else [cur_p, entry_p]
                    else:
                        particle_fx[tk] = entry_p
                if ev.get("instruction"):
                    cur_i = timeline_fx.get(tk, [])
                    timeline_fx[tk] = [*cur_i, f"{ev['instruction']};"]
            if sound_fx:
                clip_entry["sound_effects"] = sound_fx
            if particle_fx:
                clip_entry["particle_effects"] = particle_fx
            if timeline_fx:
                clip_entry["timeline"] = timeline_fx
        if samples and (is_in_place or any(abs(float(c)) > 1e-6 for c in getattr(samples[0], "root_motion_velocity", (0.0, 0.0, 0.0)))):
            rm_v = samples[0].root_motion_velocity
            upb = float(getattr(program.meta, "units_per_block", 16))
            spd_u = math.hypot(rm_v[0], math.hypot(rm_v[1], rm_v[2]))
            clip_entry["synccraft_root_motion"] = {
                "in_place": is_in_place,
                "speed_units_per_s": round(spd_u, 4),
                "speed_blocks_per_s": round(spd_u / max(1.0, upb), 4),
                "velocity_vec": [round(float(rm_v[0]), 4), round(float(rm_v[1]), 4), round(float(rm_v[2]), 4)],
                "displacement_per_cycle": [
                    round(float(rm_v[0]) * float(clip.duration_s), 4),
                    round(float(rm_v[1]) * float(clip.duration_s), 4),
                    round(float(rm_v[2]) * float(clip.duration_s), 4),
                ],
            }
        animations_dict[full_clip_key] = clip_entry

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
    elif program is not None and getattr(program, "textures", ()):
        t0 = program.textures[0]
        tex_w, tex_h = int(t0.width), int(t0.height)
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
        b_entry["pivot"] = vector(sync_point_to_bedrock(origins[bone.name]))
        rot_bed = vector(sync_euler_to_bedrock(bone.local_euler))
        if any(abs(x) > 1e-9 for x in rot_bed):
            b_entry["rotation"] = rot_bed

        cubes_out: list[dict[str, Any]] = []
        for box in boxes_by_bone[bone.name]:
            bone_origin = origins[bone.name]
            sz_arr = np.asarray(box.size, dtype=np.float64)
            half = 0.5 * sz_arr
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
                sync_from = bone_origin + c_unrot_bone - half
                sync_cpiv = bone_origin + c_piv_bone
            else:
                sync_ctr = bone_origin + np.asarray(box.local_center, dtype=np.float64)
                sync_from = sync_ctr - half
                sync_cpiv = None
                loc_e = None

            cube_entry: dict[str, Any] = {
                "origin": vector(sync_cube_from_to_bedrock_origin(sync_from, sz_arr)),
                "size": [_format_size_value(box.size[0]), _format_size_value(box.size[1]), _format_size_value(box.size[2])],
            }
            if has_loc_rot and sync_cpiv is not None and loc_e is not None:
                cube_entry["pivot"] = vector(sync_point_to_bedrock(sync_cpiv))
                cube_entry["rotation"] = vector(sync_euler_to_bedrock(loc_e))
            if abs(float(box.inflate)) > 1e-9:
                cube_entry["inflate"] = round(float(box.inflate), 6)
            if atlas is not None:
                uv_faces: dict[str, dict[str, list[float]]] = {}
                for face in FACES:
                    u0, v0, u1, v1 = face_uv(atlas, box.name, face, normalized=False)
                    # Native per-face UVs use the Blockbench convention. Bedrock
                    # reverses both chart axes on up/down faces; preserve the
                    # painted image when the emitted geometry is imported.
                    if face in ("up", "down"):
                        u0, v0, u1, v1 = u1, v1, u0, v0
                    uv_faces[face] = {
                        "uv": vector([u0, v0]),
                        "uv_size": vector([u1 - u0, v1 - v0]),
                    }
                cube_entry["uv"] = uv_faces
            elif getattr(box, "uv_faces", None):
                uv_faces_ast: dict[str, dict[str, Any]] = {}
                for face, rect in box.uv_faces.items():
                    u0, v0, u1, v1 = rect[:4]
                    if face in ("up", "down"):
                        u0, v0, u1, v1 = u1, v1, u0, v0
                    f_spec: dict[str, Any] = {
                        "uv": vector([u0, v0]),
                        "uv_size": vector([u1 - u0, v1 - v0]),
                    }
                    if len(rect) == 5 and int(rect[4]) != 0:
                        f_spec["uv_rotation"] = int(rect[4])
                    uv_faces_ast[face] = f_spec
                cube_entry["uv"] = uv_faces_ast
            elif getattr(box, "uv", None) is not None:
                bu, bv = box.uv
                cube_entry["uv"] = [_format_size_value(bu), _format_size_value(bv)]
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

            pos_vec = vector(sync_point_to_bedrock(origins[bone.name] + np.asarray(loc.local_offset, dtype=np.float64)))
            raw_le = getattr(loc, "local_euler", None)
            has_lrot = raw_le is not None and np.any(np.abs(np.asarray(raw_le, dtype=np.float64)) > 1e-9)
            if has_lrot:
                rot_loc_bed = vector(sync_euler_to_bedrock(raw_le))
                loc_obj: dict[str, Any] = {"offset": pos_vec, "rotation": rot_loc_bed}
                if _infer_locator_role_hint(cand_key) != loc.role:
                    loc_obj["role"] = loc.role
                loc_dict[cand_key] = loc_obj
            elif _infer_locator_role_hint(cand_key) == loc.role:
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

    out_bundle: dict[str, Any] = {
        "target_profile": expected_profile,
        "runtime_import_verified": False,
        "serialized_vertex_max_error": round(float(max_err), 12),
        "geometry": geometry_doc,
        "animations": animations_doc,
    }
    if program is not None and getattr(program, "source_map", None) is not None:
        out_bundle["source_map"] = program.source_map.to_dict()
    return out_bundle


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


def export_bedrock_attachable(
    scene: Any,
    *,
    program: Any | None = None,
    atlas: dict[str, Any] | None = None,
    attachable_identifier: str = "synccraft:custom_attachable",
    texture_path: str = "textures/entity/attachable",
    slot: str = "mainhand",
    num_animation_frames: int = 12,
) -> dict[str, Any]:
    """Export a Bedrock 1.10.0 ``minecraft:attachable`` bundle for held/worn submodels (ADR 0008)."""
    short_name = attachable_identifier.split(":")[-1] or "attachable"
    geo_id = f"geometry.synccraft.{short_name}"
    core = _export_bedrock_or_gecko_common(
        scene,
        expected_profile="bedrock_entity",
        program=program,
        atlas=atlas,
        identifier=geo_id,
        profile="bedrock_entity",
        num_animation_frames=num_animation_frames,
        animation_mode="keyframes",
    )
    anim_keys = list((core.get("animations") or {}).get("animations", {}).keys())
    anim_map = {k.split(".")[-1]: k for k in anim_keys}
    attachable_doc: dict[str, Any] = {
        "format_version": "1.10.0",
        "minecraft:attachable": {
            "description": {
                "identifier": attachable_identifier,
                "materials": {"default": "entity_alphatest", "enchanted": "entity_alphatest_glint"},
                "textures": {"default": texture_path, "enchanted": "textures/misc/enchanted_item_glint"},
                "geometry": {"default": geo_id},
                "animations": anim_map,
                "scripts": {"animate": list(anim_map.keys())} if anim_map else {},
                "render_controllers": ["controller.render.item_default"],
                "synccraft_slot_hint": slot,
            }
        },
    }
    return {
        **core,
        "attachable": attachable_doc,
    }
