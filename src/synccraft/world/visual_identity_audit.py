"""Independent Raw-Source vs Converted Block-DSL v3.2 Visual-Identity Audit & Proof Renderer.

Provides an independent cross-representation visual oracle that renders raw source
files (``.bbmodel``, Minecraft Java ``.json``, Bedrock ``.geo.json``, and raw indexed
triangle meshes in ``glTF 2.0``) directly without going through Block-DSL AST parsing
or ``compile_geometry``, and compares pixel-by-pixel against converted Block-DSL v3.2
programs (PSNR dB, RGB MAE, Silhouette IoU, and 4-column visual proof sheets).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from synccraft.backends.common import FACES
from synccraft.data_gen.batch_pipeline import (
    _build_camera_basis,
    _compile_program_atlas_rgba,
    _rasterize_scene_with_atlas,
)
from synccraft.dsl.ast_nodes import Program
from synccraft.dsl.parser import format_program, parse_program
from synccraft.kinematics.se3_chain import Scene, compile_geometry, euler_xyz_matrix
from synccraft.surface.domain import FACE_AXES_BOX, face_uv_extents_units
from synccraft.world.inverse_inference import (
    IngestedAssetRecord,
    _clamp_authored_inflate,
    _decode_gltf_image_by_index,
    _gltf_node_local_mat4,
    _java_element_geometry,
    _mojang_auto_uv_16,
    _read_gltf_accessor,
)


def _decode_png_bytes(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"), dtype=np.uint8)


def _decode_data_url_png(src: str) -> np.ndarray:
    _, _, b64 = src.partition(",")
    return _decode_png_bytes(base64.b64decode(b64))


def _raw_bedrock_cube_uv(
    uv_raw: Any,
    dims: tuple[float, float, float],
    *,
    mirror: bool = False,
) -> dict[str, list[float]]:
    if isinstance(uv_raw, dict):
        out_f: dict[str, list[float]] = {}
        for f in FACES:
            fd_f = uv_raw.get(f)
            if fd_f is None or not isinstance(fd_f, dict):
                continue
            if "uv" in fd_f:
                u0, v0 = (float(x) for x in fd_f["uv"])
                wu_def, hv_def = face_uv_extents_units(dims, f)
                su, sv = (float(x) for x in fd_f.get("uv_size", [wu_def, hv_def]))
                rect_f = [u0, v0, u0 + su, v0 + sv]
            else:
                wu, hv = face_uv_extents_units(dims, f)
                rect_f = [0.0, 0.0, float(wu), float(hv)]
            rot_uv = int(fd_f.get("uv_rotation", fd_f.get("rotation", 0)) or 0) % 360
            if rot_uv in (90, 180, 270):
                rect_f.append(float(rot_uv))
            out_f[f] = rect_f
        return out_f
    pair = uv_raw if isinstance(uv_raw, (list, tuple)) and len(uv_raw) >= 2 else [0.0, 0.0]
    ub, vb = float(pair[0]), float(pair[1])
    sx = max(0, int(math.floor(float(dims[0]) + 1e-9)))
    sy = max(0, int(math.floor(float(dims[1]) + 1e-9)))
    sz = max(0, int(math.floor(float(dims[2]) + 1e-9)))
    base_net: dict[str, list[float]] = {
        "up": [ub + sz, vb, ub + sz + sx, vb + sz],
        "down": [ub + sz + sx, vb, ub + sz + 2 * sx, vb + sz],
        "east": [ub, vb + sz, ub + sz, vb + sz + sy],
        "north": [ub + sz, vb + sz, ub + sz + sx, vb + sz + sy],
        "west": [ub + sz + sx, vb + sz, ub + 2 * sz + sx, vb + sz + sy],
        "south": [ub + 2 * sz + sx, vb + sz, ub + 2 * sz + 2 * sx, vb + sz + sy],
    }
    if mirror:
        swap_ew = {"east": "west", "west": "east"}
        return {swap_ew.get(fn, fn): [r[2], r[1], r[0], r[3]] for fn, r in base_net.items()}
    return base_net


@dataclass(frozen=True)
class _RawBox:
    name: str
    center: np.ndarray
    half_extents: np.ndarray
    rotation: np.ndarray
    uv_faces: dict[str, tuple[float, ...]] | None = None


@dataclass(frozen=True)
class _RawScene:
    boxes: tuple[_RawBox, ...]


@dataclass(frozen=True)
class _RawGltfTriangleBatch:
    v0: np.ndarray  # (T, 3) world
    v1: np.ndarray  # (T, 3) world
    v2: np.ndarray  # (T, 3) world
    uv0: np.ndarray  # (T, 2)
    uv1: np.ndarray  # (T, 2)
    uv2: np.ndarray  # (T, 2)
    c0: np.ndarray  # (T, 4) float64 in [0, 1]
    c1: np.ndarray  # (T, 4) float64 in [0, 1]
    c2: np.ndarray  # (T, 4) float64 in [0, 1]
    texture_rgba: np.ndarray | None
    double_sided: bool
    det_sign: float


@dataclass(frozen=True)
class RawSourceRenderBundle:
    format_kind: str
    raw_scene: _RawScene | None
    raw_uv_map: dict[str, dict[str, list[float]]]
    raw_atlas_rgba: np.ndarray
    gltf_batches: tuple[_RawGltfTriangleBatch, ...] = ()
    world_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float64))


def _corners_of_obb(center: np.ndarray, half_extents: np.ndarray, rot: np.ndarray) -> np.ndarray:
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return center[None, :] + (signs * half_extents[None, :]) @ rot.T


def _camera_bounds_from_points(
    pts_world: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[float, float, float, float]:
    if pts_world.size == 0:
        return (-8.0, 8.0, -8.0, 8.0)
    cam_right, cam_up, _ = _build_camera_basis(azimuth_deg, elevation_deg)
    xs = pts_world @ cam_right
    ys = pts_world @ cam_up
    return (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))


def extract_raw_source_render_bundle(
    payload: dict[str, Any],
    format_kind: str,
    *,
    atlas_png: bytes | None = None,
    java_textures_png: dict[str, bytes] | None = None,
) -> RawSourceRenderBundle:
    """Build an independent raw-source renderable bundle directly from raw source JSON/glTF."""
    external_rgba = _decode_png_bytes(atlas_png) if atlas_png is not None else None

    if format_kind == "bbmodel":
        return _extract_raw_bbmodel(payload, external_rgba)
    if format_kind == "java_json":
        java_rgba_map: dict[str, np.ndarray] = {}
        if java_textures_png:
            for k, b_png in java_textures_png.items():
                if b_png:
                    try:
                        java_rgba_map[str(k)] = _decode_png_bytes(b_png)
                    except Exception:
                        continue
        return _extract_raw_java(payload, external_rgba, java_rgba_map)
    if format_kind == "bedrock_geojson":
        return _extract_raw_bedrock(payload, external_rgba)
    if format_kind == "gltf_2":
        return _extract_raw_gltf_triangles(payload, external_rgba)
    raise ValueError(f"Unsupported format_kind for raw visual audit: {format_kind!r}")


def _extract_raw_bbmodel(
    doc: dict[str, Any],
    external_rgba: np.ndarray | None,
) -> RawSourceRenderBundle:
    res = doc.get("resolution") or {"width": 16, "height": 16}
    doc_w = max(1, int(res.get("width", 16)))
    doc_h = max(1, int(res.get("height", 16)))

    textures = doc.get("textures") or []
    tex_key_to_idx: dict[Any, int] = {}
    decoded_by_idx: dict[int, np.ndarray] = {}
    if isinstance(textures, list):
        for ti, te in enumerate(textures):
            if not isinstance(te, dict):
                continue
            tex_key_to_idx[ti] = ti
            tex_key_to_idx[str(ti)] = ti
            tex_key_to_idx[f"#{ti}"] = ti
            if te.get("id") is not None:
                tex_key_to_idx[str(te["id"])] = ti
                tex_key_to_idx[f"#{te['id']}"] = ti
            if te.get("uuid") is not None:
                tex_key_to_idx[str(te["uuid"])] = ti
            src = str(te.get("source") or "")
            if src.startswith("data:image/"):
                try:
                    decoded_by_idx[ti] = _decode_data_url_png(src)
                except Exception:
                    pass

    raw_elems = doc.get("elements") or []
    has_visible_cubes = any(
        isinstance(e, dict)
        and str(e.get("type", "cube")) == "cube"
        and "from" in e
        and "to" in e
        and e.get("export") is not False
        and e.get("visibility") is not False
        for e in raw_elems
    )
    elems_by_uuid: dict[str, dict[str, Any]] = {}
    referenced_tex: set[int] = set()
    obs_max_uv_by_ti: dict[int, tuple[float, float]] = {}
    per_tex_uv_mode = bool(doc.get("per_texture_uv_size") or (doc.get("meta") or {}).get("per_texture_uv_size"))
    for idx_e, el in enumerate(raw_elems):
        if not isinstance(el, dict):
            continue
        if str(el.get("type", "cube")) != "cube":
            continue
        if el.get("export") is False or (has_visible_cubes and el.get("visibility") is False):
            continue
        if "from" not in el or "to" not in el:
            continue
        uid = str(el.get("uuid") or f"raw_el_{idx_e}")
        elems_by_uuid[uid] = el
        fd = el.get("faces") or {}
        if isinstance(fd, dict) and fd:
            for f in FACES:
                ff = fd.get(f)
                if isinstance(ff, dict):
                    tk = ff.get("texture", 0 if ("uv" in ff and "texture" not in ff) else None)
                    if tk is None:
                        continue
                    ti_f = tex_key_to_idx.get(tk, tex_key_to_idx.get(str(tk), 0))
                    referenced_tex.add(ti_f)
                    uv_r = ff.get("uv")
                    if isinstance(uv_r, (list, tuple)) and len(uv_r) >= 4:
                        mu = max(abs(float(uv_r[0])), abs(float(uv_r[2])))
                        mv = max(abs(float(uv_r[1])), abs(float(uv_r[3])))
                        pu, pv = obs_max_uv_by_ti.get(ti_f, (0.0, 0.0))
                        obs_max_uv_by_ti[ti_f] = (max(pu, mu), max(pv, mv))
        elif textures:
            referenced_tex.add(0)

    def _uv_space(ti: int, fb_w: int, fb_h: int) -> tuple[float, float]:
        obs_u, obs_v = obs_max_uv_by_ti.get(ti, (0.0, 0.0))
        if isinstance(textures, list) and 0 <= ti < len(textures) and isinstance(textures[ti], dict):
            td = textures[ti]
            uw = float(td.get("uv_width") or 0.0)
            uh = float(td.get("uv_height") or 0.0)
            if uw > 0.0 and uh > 0.0 and (per_tex_uv_mode or (obs_u <= uw * 1.01 and obs_v <= uh * 1.01)):
                return max(1.0, uw), max(1.0, uh)
        if doc_w > 0 and doc_h > 0 and (obs_u <= doc_w * 1.01 and obs_v <= doc_h * 1.01):
            return max(1.0, float(doc_w)), max(1.0, float(doc_h))
        return max(1.0, float(fb_w or doc_w)), max(1.0, float(fb_h or doc_h))

    active_tis = sorted(ti for ti in referenced_tex if ti in decoded_by_idx)
    if not active_tis and decoded_by_idx:
        active_tis = [min(decoded_by_idx)]

    tex_xform: dict[int, tuple[float, float, float, float]] = {}
    if len(active_tis) == 1:
        ti0 = active_tis[0]
        raw_atlas = decoded_by_idx[ti0]
        ph, pw = raw_atlas.shape[:2]
        uw, uh = _uv_space(ti0, pw, ph)
        sx, sy = pw / uw, ph / uh
        for ti in referenced_tex | {ti0}:
            tex_xform[ti] = (sx, sy, 0.0, 0.0)
    elif len(active_tis) > 1:
        cw = max(decoded_by_idx[ti].shape[1] for ti in active_tis)
        ch = sum(decoded_by_idx[ti].shape[0] for ti in active_tis)
        raw_atlas = np.zeros((ch, cw, 4), dtype=np.uint8)
        vy = 0
        for ti in active_tis:
            arr = decoded_by_idx[ti]
            ah, aw = arr.shape[:2]
            raw_atlas[vy : vy + ah, :aw] = arr
            uw, uh = _uv_space(ti, aw, ah)
            tex_xform[ti] = (aw / uw, ah / uh, 0.0, float(vy))
            vy += ah
    elif external_rgba is not None:
        raw_atlas = external_rgba
        ph, pw = raw_atlas.shape[:2]
        uw, uh = _uv_space(0, pw, ph)
        for ti in referenced_tex | {0}:
            tex_xform[ti] = (pw / uw, ph / uh, 0.0, 0.0)
    else:
        raw_atlas = np.full((doc_h, doc_w, 4), 200, dtype=np.uint8)
        raw_atlas[:, :, 3] = 255

    groups_by_uuid: dict[str, dict[str, Any]] = {}
    for g in doc.get("groups") or []:
        if isinstance(g, dict) and g.get("uuid"):
            groups_by_uuid[str(g["uuid"])] = g

    def _register_inline_groups(nodes: list[Any]) -> None:
        for n in nodes:
            if isinstance(n, dict):
                uid = str(n.get("uuid") or "")
                if uid and uid not in groups_by_uuid and ("origin" in n or "rotation" in n or "name" in n):
                    groups_by_uuid[uid] = n
                if isinstance(n.get("children"), list):
                    _register_inline_groups(n["children"])

    outliner = doc.get("outliner") or []
    if isinstance(outliner, list):
        _register_inline_groups(outliner)

    raw_boxes: list[_RawBox] = []
    raw_uv_map: dict[str, dict[str, list[float]]] = {}
    visited_el_uuids: set[str] = set()
    all_corners: list[np.ndarray] = []

    def _emit_raw_cube(uid: str, el: dict[str, Any], parent_piv_w: np.ndarray, parent_rot_w: np.ndarray, parent_orig: np.ndarray) -> None:
        visited_el_uuids.add(uid)
        lo_raw = np.asarray(el["from"], dtype=np.float64)
        hi_raw = np.asarray(el["to"], dtype=np.float64)
        lo = np.minimum(lo_raw, hi_raw)
        hi = np.maximum(lo_raw, hi_raw)
        span = np.maximum(0.0, hi - lo)
        raw_inf = float(el.get("inflate", 0.0) or 0.0)
        inf = _clamp_authored_inflate(raw_inf, span)
        he = np.maximum(0.0, 0.5 * span + inf)
        ctr_local = 0.5 * (lo + hi)
        r_raw = el.get("rotation") or [0.0, 0.0, 0.0]
        r_tup = (float(r_raw[0]), float(r_raw[1]), float(r_raw[2]))
        if any(abs(a) > 1e-8 for a in r_tup):
            el_orig = np.asarray(el.get("origin", ctr_local.tolist()), dtype=np.float64)
            r_el = euler_xyz_matrix(r_tup)
            w_rot = parent_rot_w @ r_el
            w_ctr = parent_piv_w + parent_rot_w @ (el_orig - parent_orig) + w_rot @ (ctr_local - el_orig)
        else:
            w_rot = parent_rot_w
            w_ctr = parent_piv_w + parent_rot_w @ (ctr_local - parent_orig)

        bname = f"raw_bb_{len(raw_boxes)}_{uid[:8]}"
        f_dict = el.get("faces") or {}
        f_uv: dict[str, list[float]] = {}
        box_uv_mode = bool(doc.get("meta", {}).get("box_uv", False)) and not any(
            isinstance(f_dict.get(fn), dict) and "uv" in f_dict[fn] for fn in FACES
        )
        if box_uv_mode:
            uv_off = el.get("uv_offset") or [0, 0]
            u0_b, v0_b = float(uv_off[0]), float(uv_off[1])
            mir = bool(el.get("mirror_uv", False))
            bed_uv = _raw_bedrock_cube_uv(
                [u0_b, v0_b],
                (float(span[0]), float(span[1]), float(span[2])),
                mirror=mir,
            )
            sx, sy, ox, oy = tex_xform.get(0, (1.0, 1.0, 0.0, 0.0))
            for fn, r_v in bed_uv.items():
                f_uv[fn] = [r_v[0] * sx + ox, r_v[1] * sy + oy, r_v[2] * sx + ox, r_v[3] * sy + oy]
        elif isinstance(f_dict, dict):
            has_explicit_face_tex = any(
                isinstance(v_fd, dict) and "texture" in v_fd and v_fd["texture"] is not None
                for v_fd in f_dict.values()
            )
            for fn in FACES:
                if fn not in f_dict:
                    continue
                fd = f_dict[fn]
                if not isinstance(fd, dict):
                    continue
                if fd.get("enabled") is False:
                    continue
                if "texture" in fd and fd["texture"] is None:
                    continue
                if has_explicit_face_tex and "texture" not in fd:
                    continue
                uv_r = fd.get("uv")
                if not (isinstance(uv_r, (list, tuple)) and len(uv_r) == 4):
                    continue
                tk = fd.get("texture", 0)
                ti = tex_key_to_idx.get(tk, tex_key_to_idx.get(str(tk), 0))
                sx, sy, ox, oy = tex_xform.get(ti, (1.0, 1.0, 0.0, 0.0))
                rect = [
                    float(uv_r[0]) * sx + ox,
                    float(uv_r[1]) * sy + oy,
                    float(uv_r[2]) * sx + ox,
                    float(uv_r[3]) * sy + oy,
                ]
                f_rot = int(fd.get("rotation", 0) or 0) % 360
                if f_rot in (90, 180, 270):
                    rect.append(float(f_rot))
                f_uv[fn] = rect

        raw_uv_map[bname] = f_uv
        raw_boxes.append(_RawBox(name=bname, center=w_ctr, half_extents=he, rotation=w_rot))
        all_corners.append(_corners_of_obb(w_ctr, he, w_rot))

    def _mark_subtree_visited(node: Any) -> None:
        if isinstance(node, str):
            visited_el_uuids.add(node)
        elif isinstance(node, dict):
            uid_m = str(node.get("uuid") or "")
            if uid_m:
                visited_el_uuids.add(uid_m)
            for ch_m in node.get("children") or []:
                _mark_subtree_visited(ch_m)

    has_visible_outliner = any(
        (isinstance(top, str) and top in elems_by_uuid)
        or (
            isinstance(top, dict)
            and top.get("export") is not False
            and top.get("visibility") is not False
            and groups_by_uuid.get(str(top.get("uuid") or ""), top).get("export") is not False
            and groups_by_uuid.get(str(top.get("uuid") or ""), top).get("visibility") is not False
        )
        for top in outliner
    )

    def _walk_node(node: Any, p_piv_w: np.ndarray, p_rot_w: np.ndarray, p_orig: np.ndarray) -> None:
        if isinstance(node, str):
            if node in elems_by_uuid:
                _emit_raw_cube(node, elems_by_uuid[node], p_piv_w, p_rot_w, p_orig)
            return
        if not isinstance(node, dict):
            return
        if node.get("export") is False or (has_visible_outliner and node.get("visibility") is False):
            _mark_subtree_visited(node)
            return
        uid = str(node.get("uuid") or "")
        if uid in elems_by_uuid and "children" not in node:
            _emit_raw_cube(uid, elems_by_uuid[uid], p_piv_w, p_rot_w, p_orig)
            return
        grp = groups_by_uuid.get(uid, node)
        if grp.get("export") is False or (has_visible_outliner and grp.get("visibility") is False):
            _mark_subtree_visited(node)
            return
        g_orig = np.asarray(grp.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
        g_rot_raw = grp.get("rotation") or [0.0, 0.0, 0.0]
        g_rot = euler_xyz_matrix((float(g_rot_raw[0]), float(g_rot_raw[1]), float(g_rot_raw[2])))
        w_piv = p_piv_w + p_rot_w @ (g_orig - p_orig)
        w_rot = p_rot_w @ g_rot
        for ch in node.get("children") or []:
            _walk_node(ch, w_piv, w_rot, g_orig)

    z3 = np.zeros(3, dtype=np.float64)
    eye3 = np.eye(3, dtype=np.float64)
    for top in outliner:
        _walk_node(top, z3, eye3, z3)
    for uid, el in elems_by_uuid.items():
        if uid not in visited_el_uuids:
            _emit_raw_cube(uid, el, z3, eye3, z3)

    pts = np.concatenate(all_corners, axis=0) if all_corners else np.zeros((0, 3), dtype=np.float64)
    return RawSourceRenderBundle(
        format_kind="bbmodel",
        raw_scene=_RawScene(boxes=tuple(raw_boxes)),
        raw_uv_map=raw_uv_map,
        raw_atlas_rgba=raw_atlas,
        world_points=pts,
    )


def _extract_raw_java(
    doc: dict[str, Any],
    external_rgba: np.ndarray | None,
    java_textures_rgba: dict[str, np.ndarray],
) -> RawSourceRenderBundle:
    ts = doc.get("texture_size") or [16, 16]
    uv_w_base = max(1.0, float(ts[0]))
    uv_h_base = max(1.0, float(ts[1]))
    tex_dict = doc.get("textures") if isinstance(doc.get("textures"), dict) else {}

    def _res_key(raw_ref: str) -> str:
        cur = str(raw_ref or "").strip()
        if cur.startswith("#"):
            cur = cur[1:]
        for _ in range(8):
            nxt = tex_dict.get(cur)
            if isinstance(nxt, str) and nxt.startswith("#"):
                cur = nxt[1:]
            else:
                break
        return cur

    elements = [e for e in (doc.get("elements") or []) if isinstance(e, dict) and "from" in e and "to" in e]
    used_keys: list[str] = []
    for el in elements:
        fd = el.get("faces") or {}
        if isinstance(fd, dict):
            for fn in FACES:
                ff = fd.get(fn)
                if isinstance(ff, dict) and ff.get("texture"):
                    rk = _res_key(str(ff["texture"]))
                    if rk in java_textures_rgba and rk not in used_keys:
                        used_keys.append(rk)

    key_xform: dict[str, tuple[float, float, float, float]] = {}
    if len(used_keys) > 1:
        cw = max(java_textures_rgba[k].shape[1] for k in used_keys)
        ch = sum(java_textures_rgba[k].shape[0] for k in used_keys)
        raw_atlas = np.zeros((ch, cw, 4), dtype=np.uint8)
        vy = 0
        for k in used_keys:
            arr = java_textures_rgba[k]
            ah, aw = arr.shape[:2]
            raw_atlas[vy : vy + ah, :aw] = arr
            key_xform[k] = (aw / 16.0, ah / 16.0, 0.0, float(vy))
            vy += ah
    elif len(used_keys) == 1:
        k0 = used_keys[0]
        raw_atlas = java_textures_rgba[k0]
        ah, aw = raw_atlas.shape[:2]
        key_xform[k0] = (aw / 16.0, ah / 16.0, 0.0, 0.0)
    elif external_rgba is not None:
        raw_atlas = external_rgba
        ah, aw = raw_atlas.shape[:2]
        key_xform["__default__"] = (aw / 16.0, ah / 16.0, 0.0, 0.0)
    else:
        raw_atlas = np.full((int(uv_h_base), int(uv_w_base), 4), 200, dtype=np.uint8)
        raw_atlas[:, :, 3] = 255
        key_xform["__default__"] = (uv_w_base / 16.0, uv_h_base / 16.0, 0.0, 0.0)

    default_xf = next(iter(key_xform.values()))
    raw_boxes: list[_RawBox] = []
    raw_uv_map: dict[str, dict[str, list[float]]] = {}
    all_corners: list[np.ndarray] = []

    for idx_e, el in enumerate(elements):
        lo_j, hi_j, _, span_scaled, ctr_geom, rot_euler, rot_origin, has_rot = _java_element_geometry(el)
        if has_rot:
            r_mat = euler_xyz_matrix(rot_euler)
            w_ctr = rot_origin + r_mat @ (ctr_geom - rot_origin)
        else:
            r_mat = np.eye(3, dtype=np.float64)
            w_ctr = ctr_geom
        he = 0.5 * span_scaled
        bname = f"raw_java_{idx_e}"
        fd = el.get("faces") or {}
        f_uv: dict[str, list[float]] = {}
        if isinstance(fd, dict):
            for fn in FACES:
                if fn not in fd or not isinstance(fd[fn], dict):
                    continue
                ff = fd[fn]
                t_ref = ff.get("texture")
                if t_ref is None or str(t_ref).strip() == "":
                    continue
                rk = _res_key(str(t_ref))
                sx, sy, ox, oy = key_xform.get(rk, default_xf)
                uv_r = ff.get("uv")
                if isinstance(uv_r, (list, tuple)) and len(uv_r) == 4:
                    u0_n, v0_n, u1_n, v1_n = (
                        float(uv_r[0]),
                        float(uv_r[1]),
                        float(uv_r[2]),
                        float(uv_r[3]),
                    )
                else:
                    u0_n, v0_n, u1_n, v1_n = _mojang_auto_uv_16(lo_j, hi_j, fn)
                rect = [u0_n * sx + ox, v0_n * sy + oy, u1_n * sx + ox, v1_n * sy + oy]
                f_rot = int(ff.get("rotation", 0) or 0) % 360
                if f_rot in (90, 180, 270):
                    rect.append(float(f_rot))
                f_uv[fn] = rect
        raw_uv_map[bname] = f_uv
        raw_boxes.append(_RawBox(name=bname, center=w_ctr, half_extents=he, rotation=r_mat))
        all_corners.append(_corners_of_obb(w_ctr, he, r_mat))

    pts = np.concatenate(all_corners, axis=0) if all_corners else np.zeros((0, 3), dtype=np.float64)
    return RawSourceRenderBundle(
        format_kind="java_json",
        raw_scene=_RawScene(boxes=tuple(raw_boxes)),
        raw_uv_map=raw_uv_map,
        raw_atlas_rgba=raw_atlas,
        world_points=pts,
    )


def _extract_raw_bedrock(
    doc: dict[str, Any],
    external_rgba: np.ndarray | None,
) -> RawSourceRenderBundle:
    geos = doc.get("minecraft:geometry")
    if isinstance(geos, list) and geos:
        g0 = geos[0]
        desc = g0.get("description") or {}
        tw = max(1, int(desc.get("texture_width", 64)))
        th = max(1, int(desc.get("texture_height", 64)))
        bones = g0.get("bones") or []
    else:
        k0 = next((k for k in doc if k.startswith("geometry.")), None)
        g0 = doc[k0] if k0 else {}
        tw = max(1, int(g0.get("texturewidth", 64)))
        th = max(1, int(g0.get("textureheight", 64)))
        bones = g0.get("bones") or []

    if external_rgba is not None:
        raw_atlas = external_rgba
        sx = raw_atlas.shape[1] / float(tw)
        sy = raw_atlas.shape[0] / float(th)
    else:
        raw_atlas = np.full((th, tw, 4), 200, dtype=np.uint8)
        raw_atlas[:, :, 3] = 255
        sx, sy = 1.0, 1.0

    bone_map = {str(b["name"]): b for b in bones if isinstance(b, dict) and b.get("name")}
    world_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _bone_world(bname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if bname in world_cache:
            return world_cache[bname]
        b = bone_map[bname]
        piv_b = b.get("pivot") or [0.0, 0.0, 0.0]
        piv_rh = np.array([-float(piv_b[0]), float(piv_b[1]), float(piv_b[2])], dtype=np.float64)
        rot_b = b.get("rotation") or [0.0, 0.0, 0.0]
        r_loc = euler_xyz_matrix((-float(rot_b[0]), -float(rot_b[1]), float(rot_b[2])))
        pname = b.get("parent")
        if pname and str(pname) in bone_map:
            p_piv_w, p_rot_w, p_piv_bind = _bone_world(str(pname))
            w_piv = p_piv_w + p_rot_w @ (piv_rh - p_piv_bind)
            w_rot = p_rot_w @ r_loc
        else:
            w_piv = piv_rh
            w_rot = r_loc
        world_cache[bname] = (w_piv, w_rot, piv_rh)
        return w_piv, w_rot, piv_rh

    raw_boxes: list[_RawBox] = []
    raw_uv_map: dict[str, dict[str, list[float]]] = {}
    all_corners: list[np.ndarray] = []

    for bname, b in bone_map.items():
        w_piv, w_rot, piv_bind = _bone_world(bname)
        b_mir = bool(b.get("mirror", False))
        for ci, c in enumerate(b.get("cubes") or []):
            if not isinstance(c, dict) or "origin" not in c or "size" not in c:
                continue
            orig_b = np.asarray(c["origin"], dtype=np.float64)
            sz = np.maximum(0.0, np.asarray(c["size"], dtype=np.float64))
            lo_rh = np.array([-(orig_b[0] + sz[0]), orig_b[1], orig_b[2]], dtype=np.float64)
            ctr_bind = lo_rh + 0.5 * sz
            inf = _clamp_authored_inflate(float(c.get("inflate", b.get("inflate", 0.0)) or 0.0), sz)
            he = np.maximum(0.0, 0.5 * sz + inf)
            c_rot_b = c.get("rotation")
            if isinstance(c_rot_b, (list, tuple)) and len(c_rot_b) == 3 and any(abs(float(x)) > 1e-8 for x in c_rot_b):
                cp_b = c.get("pivot") or [-(ctr_bind[0]), ctr_bind[1], ctr_bind[2]]
                cp_rh = np.array([-float(cp_b[0]), float(cp_b[1]), float(cp_b[2])], dtype=np.float64)
                r_c = euler_xyz_matrix((-float(c_rot_b[0]), -float(c_rot_b[1]), float(c_rot_b[2])))
                cw_rot = w_rot @ r_c
                cw_ctr = w_piv + w_rot @ (cp_rh - piv_bind) + cw_rot @ (ctr_bind - cp_rh)
            else:
                cw_rot = w_rot
                cw_ctr = w_piv + w_rot @ (ctr_bind - piv_bind)

            c_mir = bool(c.get("mirror", b_mir))
            uv_d = _raw_bedrock_cube_uv(
                c.get("uv"),
                (float(sz[0]), float(sz[1]), float(sz[2])),
                mirror=c_mir,
            )
            scaled_uv: dict[str, list[float]] = {}
            for fn, rv in uv_d.items():
                entry = [rv[0] * sx, rv[1] * sy, rv[2] * sx, rv[3] * sy]
                if len(rv) >= 5:
                    entry.append(rv[4])
                scaled_uv[fn] = entry

            box_id = f"raw_bed_{len(raw_boxes)}_{ci}"
            raw_uv_map[box_id] = scaled_uv
            raw_boxes.append(_RawBox(name=box_id, center=cw_ctr, half_extents=he, rotation=cw_rot))
            all_corners.append(_corners_of_obb(cw_ctr, he, cw_rot))

    pts = np.concatenate(all_corners, axis=0) if all_corners else np.zeros((0, 3), dtype=np.float64)
    return RawSourceRenderBundle(
        format_kind="bedrock_geojson",
        raw_scene=_RawScene(boxes=tuple(raw_boxes)),
        raw_uv_map=raw_uv_map,
        raw_atlas_rgba=raw_atlas,
        world_points=pts,
    )


def _extract_raw_gltf_triangles(
    bundle: dict[str, Any],
    external_rgba: np.ndarray | None,
) -> RawSourceRenderBundle:
    doc = bundle.get("gltf") if isinstance(bundle.get("gltf"), dict) else bundle
    buffers_b = bundle.get("buffers") or bundle.get("buffers_bytes") or {}
    images_b = bundle.get("images") or bundle.get("images_bytes") or {}

    # Decode glTF textures
    tex_images: dict[int, np.ndarray] = {}
    gltf_textures = doc.get("textures") or []
    gltf_images = doc.get("images") or []
    for ti, t_entry in enumerate(gltf_textures):
        if not isinstance(t_entry, dict):
            continue
        src_idx = t_entry.get("source")
        if src_idx is None or not (0 <= int(src_idx) < len(gltf_images)):
            continue
        arr_t, _ = _decode_gltf_image_by_index(doc, buffers_b, images_b, int(src_idx))
        if arr_t is not None:
            tex_images[ti] = arr_t

    nodes = doc.get("nodes") or []
    meshes = doc.get("meshes") or []
    materials = doc.get("materials") or []
    scenes = doc.get("scenes") or []
    # Strip Sketchfab viewer Z-up wrapper nodes (GLTF_SceneRootNode / RootNode / Sketchfab_model)
    # so raw glTF evaluation uses the exact same Y-up artist model root as _decompile_gltf.
    name_to_idx = {str(n.get("name") or ""): i for i, n in enumerate(nodes) if isinstance(n, dict)}
    if "GLTF_SceneRootNode" in name_to_idx:
        root_indices = [int(i) for i in (nodes[name_to_idx["GLTF_SceneRootNode"]].get("children") or [])]
    elif "RootNode" in name_to_idx:
        root_indices = [int(i) for i in (nodes[name_to_idx["RootNode"]].get("children") or [])]
    elif "Sketchfab_model" in name_to_idx:
        sf_children = [int(i) for i in (nodes[name_to_idx["Sketchfab_model"]].get("children") or [])]
        if len(sf_children) == 1 and nodes[sf_children[0]].get("mesh") is None and nodes[sf_children[0]].get("children"):
            c0 = sf_children[0]
            m_sf = _gltf_node_local_mat4(nodes[name_to_idx["Sketchfab_model"]])[:3, :3]
            m_c0 = _gltf_node_local_mat4(nodes[c0])[:3, :3]
            prod = m_sf @ m_c0
            s_est = float(np.linalg.norm(prod[:, 0]))
            if s_est > 1e-9 and np.allclose(prod / s_est, np.eye(3), atol=1e-2):
                root_indices = [int(i) for i in (nodes[c0].get("children") or [])]
            else:
                root_indices = sf_children
        else:
            root_indices = sf_children
    elif scenes and isinstance(scenes[0], dict) and scenes[0].get("nodes"):
        root_indices = [int(i) for i in scenes[0]["nodes"]]
    else:
        child_set = {int(c) for n in nodes if isinstance(n, dict) for c in (n.get("children") or [])}
        root_indices = [i for i in range(len(nodes)) if i not in child_set]

    batches: list[_RawGltfTriangleBatch] = []
    all_pts: list[np.ndarray] = []

    def _walk_gltf_node(n_idx: int, parent_m: np.ndarray) -> None:
        if not (0 <= n_idx < len(nodes)):
            return
        nd = nodes[n_idx]
        if not isinstance(nd, dict):
            return
        m_world = parent_m @ _gltf_node_local_mat4(nd)
        det3 = float(np.linalg.det(m_world[:3, :3]))
        det_sign = -1.0 if det3 < 0.0 else 1.0

        m_idx = nd.get("mesh")
        if m_idx is not None and 0 <= int(m_idx) < len(meshes):
            mesh_d = meshes[int(m_idx)]
            for prim in mesh_d.get("primitives") or []:
                if not isinstance(prim, dict) or int(prim.get("mode", 4)) != 4:
                    continue
                attrs = prim.get("attributes") or {}
                if "POSITION" not in attrs:
                    continue
                pos = _read_gltf_accessor(doc, buffers_b, int(attrs["POSITION"])).astype(np.float64)
                if pos.ndim != 2 or pos.shape[1] < 3 or len(pos) < 3:
                    continue
                pos = pos[:, :3]
                if prim.get("indices") is not None:
                    idx_arr = _read_gltf_accessor(doc, buffers_b, int(prim["indices"])).reshape(-1).astype(np.int64)
                else:
                    idx_arr = np.arange(len(pos), dtype=np.int64)
                n_tri = len(idx_arr) // 3
                if n_tri == 0:
                    continue
                tris = idx_arr[: n_tri * 3].reshape(n_tri, 3)

                # Transform vertices to world coordinates (then convert meters->Blockbench 16x if needed? Wait: check scale!)
                pos_h = np.concatenate([pos, np.ones((len(pos), 1), dtype=np.float64)], axis=1)
                pos_w = (pos_h @ m_world.T)[:, :3]
                all_pts.append(pos_w)

                if "TEXCOORD_0" in attrs:
                    uvs = _read_gltf_accessor(doc, buffers_b, int(attrs["TEXCOORD_0"])).astype(np.float64)[:, :2]
                else:
                    uvs = np.zeros((len(pos), 2), dtype=np.float64)

                base_factor = np.ones(4, dtype=np.float64)
                tex_arr: np.ndarray | None = None
                double_sided = True
                mat_idx = prim.get("material")
                if mat_idx is not None and 0 <= int(mat_idx) < len(materials):
                    mat_d = materials[int(mat_idx)]
                    double_sided = bool(mat_d.get("doubleSided", True))
                    pbr = mat_d.get("pbrMetallicRoughness") or {}
                    bf = pbr.get("baseColorFactor")
                    if isinstance(bf, (list, tuple)) and len(bf) >= 3:
                        base_factor[:3] = [float(bf[0]), float(bf[1]), float(bf[2])]
                        if len(bf) >= 4:
                            base_factor[3] = float(bf[3])
                    bct = pbr.get("baseColorTexture")
                    if isinstance(bct, dict) and bct.get("index") is not None:
                        tex_arr = tex_images.get(int(bct["index"]))
                if tex_arr is None and external_rgba is not None and mat_idx is None:
                    tex_arr = external_rgba

                if "COLOR_0" in attrs:
                    vcol = _read_gltf_accessor(doc, buffers_b, int(attrs["COLOR_0"])).astype(np.float64)
                    if vcol.max() > 1.05:
                        vcol = vcol / (65535.0 if vcol.max() > 256.0 else 255.0)
                    if vcol.shape[1] == 3:
                        vcol = np.concatenate([vcol, np.ones((len(vcol), 1), dtype=np.float64)], axis=1)
                    else:
                        vcol = vcol[:, :4]
                    # glTF COLOR_0 & baseColorFactor are linear; convert linear factor -> sRGB multiplier when untextured
                    vcol_rgba = np.clip(vcol * base_factor[None, :], 0.0, 1.0)
                else:
                    vcol_rgba = np.broadcast_to(base_factor[None, :], (len(pos), 4)).copy()

                batches.append(
                    _RawGltfTriangleBatch(
                        v0=pos_w[tris[:, 0]],
                        v1=pos_w[tris[:, 1]],
                        v2=pos_w[tris[:, 2]],
                        uv0=uvs[tris[:, 0]],
                        uv1=uvs[tris[:, 1]],
                        uv2=uvs[tris[:, 2]],
                        c0=vcol_rgba[tris[:, 0]],
                        c1=vcol_rgba[tris[:, 1]],
                        c2=vcol_rgba[tris[:, 2]],
                        texture_rgba=tex_arr,
                        double_sided=double_sided,
                        det_sign=det_sign,
                    )
                )

        for ch_i in nd.get("children") or []:
            _walk_gltf_node(int(ch_i), m_world)

    eye4 = np.eye(4, dtype=np.float64)
    for ri in root_indices:
        _walk_gltf_node(ri, eye4)

    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3), dtype=np.float64)
    fb_atlas = next((b.texture_rgba for b in batches if b.texture_rgba is not None), None)
    if fb_atlas is None:
        fb_atlas = np.full((16, 16, 4), 220, dtype=np.uint8)
    return RawSourceRenderBundle(
        format_kind="gltf_2",
        raw_scene=None,
        raw_uv_map={},
        raw_atlas_rgba=fb_atlas,
        gltf_batches=tuple(batches),
        world_points=pts,
    )


def _linear_rgb_to_srgb_float(lin: np.ndarray) -> np.ndarray:
    c = np.clip(lin, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 1e-12), 1.0 / 2.4) - 0.055)


def _rasterize_raw_gltf_triangles(
    bundle: RawSourceRenderBundle,
    *,
    canvas_size: tuple[int, int] = (220, 220),
    azimuth_deg: float = 36.0,
    elevation_deg: float = 24.0,
    bg_rgb: tuple[int, int, int] = (24, 28, 38),
    camera_bounds: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Direct orthographic Z-buffered barycentric triangle rasterizer for raw glTF 2.0 meshes."""
    w_img, h_img = canvas_size
    cam_right, cam_up, cam_fwd = _build_camera_basis(azimuth_deg, elevation_deg)

    if camera_bounds is not None:
        min_x, max_x, min_y, max_y = camera_bounds
    else:
        min_x, max_x, min_y, max_y = _camera_bounds_from_points(bundle.world_points, azimuth_deg, elevation_deg)

    span_x = max(1e-3, max_x - min_x)
    span_y = max(1e-3, max_y - min_y)
    margin = 0.84
    scale = min((w_img * margin) / span_x, (h_img * margin) / span_y)
    cx_cam = 0.5 * (min_x + max_x)
    cy_cam = 0.5 * (min_y + max_y)

    color_buf = np.empty((h_img, w_img, 3), dtype=np.uint8)
    color_buf[:, :, 0] = bg_rgb[0]
    color_buf[:, :, 1] = bg_rgb[1]
    color_buf[:, :, 2] = bg_rgb[2]
    for gy in range(0, h_img, 22):
        color_buf[gy, :] = (31, 36, 48)
    for gx in range(0, w_img, 22):
        color_buf[:, gx] = (31, 36, 48)

    z_buf = np.full((h_img, w_img), -1e18, dtype=np.float64)
    translucent_queue: list[tuple[float, int, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for batch in bundle.gltf_batches:
        tex = batch.texture_rgba
        n_tris = len(batch.v0)
        for ti in range(n_tris):
            p0 = batch.v0[ti]
            p1 = batch.v1[ti]
            p2 = batch.v2[ti]

            n_vec = np.cross(p1 - p0, p2 - p0) * batch.det_sign
            n_len = float(np.linalg.norm(n_vec))
            if n_len <= 1e-11:
                continue
            n_world = n_vec / n_len
            ndotv = float(np.dot(n_world, cam_fwd))
            if ndotv <= 1e-6:
                if batch.double_sided and ndotv < -1e-6:
                    ndotv = -ndotv
                else:
                    continue

            x0 = (float(np.dot(p0, cam_right)) - cx_cam) * scale + 0.5 * w_img
            y0 = -(float(np.dot(p0, cam_up)) - cy_cam) * scale + 0.5 * h_img
            z0 = float(np.dot(p0, cam_fwd))

            x1 = (float(np.dot(p1, cam_right)) - cx_cam) * scale + 0.5 * w_img
            y1 = -(float(np.dot(p1, cam_up)) - cy_cam) * scale + 0.5 * h_img
            z1 = float(np.dot(p1, cam_fwd))

            x2 = (float(np.dot(p2, cam_right)) - cx_cam) * scale + 0.5 * w_img
            y2 = -(float(np.dot(p2, cam_up)) - cy_cam) * scale + 0.5 * h_img
            z2 = float(np.dot(p2, cam_fwd))

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-5:
                continue
            inv_den = 1.0 / denom

            px_min = max(0, int(math.floor(min(x0, x1, x2))))
            px_max = min(w_img - 1, int(math.ceil(max(x0, x1, x2))))
            py_min = max(0, int(math.floor(min(y0, y1, y2))))
            py_max = min(h_img - 1, int(math.ceil(max(y0, y1, y2))))
            if px_min > px_max or py_min > py_max:
                continue

            gx, gy = np.meshgrid(
                np.arange(px_min, px_max + 1, dtype=np.float64) + 0.5,
                np.arange(py_min, py_max + 1, dtype=np.float64) + 0.5,
            )
            w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) * inv_den
            w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) * inv_den
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside):
                continue

            c_lin = (
                w0[..., None] * batch.c0[ti]
                + w1[..., None] * batch.c1[ti]
                + w2[..., None] * batch.c2[ti]
            )
            c_srgb_mult = _linear_rgb_to_srgb_float(c_lin[..., :3])
            c_alpha_mult = np.clip(c_lin[..., 3], 0.0, 1.0)

            if tex is not None:
                th, tw = tex.shape[:2]
                u_interp = w0 * batch.uv0[ti, 0] + w1 * batch.uv1[ti, 0] + w2 * batch.uv2[ti, 0]
                v_interp = w0 * batch.uv0[ti, 1] + w1 * batch.uv1[ti, 1] + w2 * batch.uv2[ti, 1]
                # glTF REPEAT wrap with exact pixel-center clamping inside [0, 1] islands
                u_wrap = np.where((u_interp >= -1e-4) & (u_interp <= 1.0001), np.clip(u_interp, 1e-5, 1.0 - 1e-5), np.mod(u_interp, 1.0))
                v_wrap = np.where((v_interp >= -1e-4) & (v_interp <= 1.0001), np.clip(v_interp, 1e-5, 1.0 - 1e-5), np.mod(v_interp, 1.0))
                uu = np.clip(np.floor(u_wrap * tw).astype(np.int32), 0, tw - 1)
                vv = np.clip(np.floor(v_wrap * th).astype(np.int32), 0, th - 1)
                sampled_rgba = tex[vv, uu]
                rgb_base = sampled_rgba[..., :3].astype(np.float64) * c_srgb_mult
                alpha = np.clip(sampled_rgba[..., 3].astype(np.float64) * c_alpha_mult, 0.0, 255.0).astype(np.uint8)
            else:
                rgb_base = c_srgb_mult * 255.0
                alpha = np.clip(c_alpha_mult * 255.0, 0.0, 255.0).astype(np.uint8)

            shade = 0.78 + 0.26 * ndotv
            shaded = np.clip(rgb_base * shade, 0.0, 255.0).astype(np.uint8)
            z_pix = w0 * z0 + w1 * z1 + w2 * z2
            sub_z = z_buf[py_min : py_max + 1, px_min : px_max + 1]

            opaque_mask = inside & (alpha >= 200) & (z_pix > sub_z)
            if np.any(opaque_mask):
                sub_z[opaque_mask] = z_pix[opaque_mask]
                sub_col = color_buf[py_min : py_max + 1, px_min : px_max + 1]
                sub_col[opaque_mask] = shaded[opaque_mask]

            trans_cand = inside & (alpha >= 20) & (alpha < 200)
            if np.any(trans_cand):
                translucent_queue.append(
                    (float(np.mean(z_pix[trans_cand])), py_min, py_max, px_min, px_max, trans_cand, z_pix, shaded, alpha)
                )

    if translucent_queue:
        translucent_queue.sort(key=lambda it: it[0])
        for _, py_min, py_max, px_min, px_max, trans_cand, z_pix, shaded, alpha in translucent_queue:
            sub_z = z_buf[py_min : py_max + 1, px_min : px_max + 1]
            vis_t = trans_cand & (z_pix > sub_z)
            if not np.any(vis_t):
                continue
            sub_col = color_buf[py_min : py_max + 1, px_min : px_max + 1]
            a_f = (alpha[vis_t].astype(np.float64) / 255.0)[:, None]
            sub_col[vis_t] = np.clip(
                shaded[vis_t].astype(np.float64) * a_f + sub_col[vis_t].astype(np.float64) * (1.0 - a_f),
                0.0,
                255.0,
            ).astype(np.uint8)

    return Image.fromarray(color_buf, mode="RGB")


def rasterize_raw_source_bundle(
    bundle: RawSourceRenderBundle,
    *,
    canvas_size: tuple[int, int] = (220, 220),
    azimuth_deg: float = 36.0,
    elevation_deg: float = 24.0,
    camera_bounds: tuple[float, float, float, float] | None = None,
    draw_contours: bool = False,
) -> Image.Image:
    """Render raw source bundle from the specified orthographic camera."""
    if bundle.format_kind == "gltf_2":
        return _rasterize_raw_gltf_triangles(
            bundle,
            canvas_size=canvas_size,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            camera_bounds=camera_bounds,
        )
    assert bundle.raw_scene is not None
    return _rasterize_scene_with_atlas(
        bundle.raw_scene,  # type: ignore[arg-type]
        {"faces": bundle.raw_uv_map},
        bundle.raw_atlas_rgba,
        canvas_size=canvas_size,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        camera_bounds=camera_bounds,
        draw_contours=draw_contours,
    )


def _scene_world_points(scene: Scene) -> np.ndarray:
    if not scene.boxes:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate([_corners_of_obb(b.center, b.half_extents, b.rotation) for b in scene.boxes], axis=0)


def _compute_image_pair_metrics(
    img_ref: Image.Image,
    img_cand: Image.Image,
    bg_img: Image.Image,
) -> dict[str, float]:
    a = np.asarray(img_ref, dtype=np.float64)
    b = np.asarray(img_cand, dtype=np.float64)
    bg = np.asarray(bg_img, dtype=np.float64)

    fg_a = np.any(np.abs(a - bg) > 1.5, axis=-1)
    fg_b = np.any(np.abs(b - bg) > 1.5, axis=-1)
    union = fg_a | fg_b
    inter = fg_a & fg_b
    iou = float(inter.sum() / union.sum()) if np.any(union) else 1.0

    diff = np.abs(a - b)
    if np.any(union):
        mse_fg = float(np.mean((a[union] - b[union]) ** 2))
        mae_fg = float(np.mean(diff[union]))
    else:
        mse_fg = float(np.mean((a - b) ** 2))
        mae_fg = float(np.mean(diff))

    psnr = 60.0 if mse_fg <= 0.065 else min(60.0, 10.0 * math.log10((255.0 ** 2) / mse_fg))
    return {
        "psnr_db": round(float(psnr), 2),
        "mae_rgb": round(float(mae_fg), 3),
        "silhouette_iou": round(float(iou), 4),
    }


def _make_empty_studio_bg(canvas_size: tuple[int, int], bg_rgb: tuple[int, int, int] = (24, 28, 38)) -> Image.Image:
    w_img, h_img = canvas_size
    buf = np.empty((h_img, w_img, 3), dtype=np.uint8)
    buf[:, :, 0] = bg_rgb[0]
    buf[:, :, 1] = bg_rgb[1]
    buf[:, :, 2] = bg_rgb[2]
    for gy in range(0, h_img, 22):
        buf[gy, :] = (31, 36, 48)
    for gx in range(0, w_img, 22):
        buf[:, gx] = (31, 36, 48)
    return Image.fromarray(buf, mode="RGB")


def audit_converted_asset_visual_identity(
    payload: dict[str, Any],
    format_kind: str,
    rec: IngestedAssetRecord,
    enriched_program: Program,
    *,
    atlas_png: bytes | None = None,
    java_textures_png: dict[str, bytes] | None = None,
    canvas_size: tuple[int, int] = (220, 220),
) -> tuple[dict[str, Any], tuple[Image.Image, Image.Image, Image.Image, Image.Image]]:
    """Audit visual identity between raw source and converted Block-DSL v3.2 + render 4-panel proof."""
    raw_bundle = extract_raw_source_render_bundle(
        payload,
        format_kind,
        atlas_png=atlas_png,
        java_textures_png=java_textures_png,
    )
    reparsed = parse_program(format_program(enriched_program))
    conv_scene = compile_geometry(reparsed)

    # Note on glTF scale: _decompile_gltf scales meter-sized meshes to Blockbench units;
    # camera_bounds auto-fits each representation to identical orthographic framing when scales differ,
    # or shares exact world bounds when world extents match within 1%!
    conv_pts = _scene_world_points(conv_scene)
    raw_pts = raw_bundle.world_points
    raw_span = float(np.ptp(raw_pts, axis=0).max()) if raw_pts.size else 1.0
    conv_span = float(np.ptp(conv_pts, axis=0).max()) if conv_pts.size else 1.0
    same_world_scale = abs(raw_span - conv_span) <= 0.02 * max(1.0, conv_span)

    conv_atlas = (
        rec.resolved_atlas_rgba
        if (rec.has_source_texture and rec.resolved_atlas_rgba is not None)
        else raw_bundle.raw_atlas_rgba
    )
    conv_uv_faces: dict[str, dict[str, list[float]]] = {}
    standalone_uv_faces: dict[str, dict[str, list[float]]] = {}
    for wb in conv_scene.boxes:
        if wb.uv_faces:
            standalone_uv_faces[wb.name] = {fn: [float(x) for x in rv] for fn, rv in wb.uv_faces.items()}
        if wb.name in rec.resolved_uv_map:
            conv_uv_faces[wb.name] = rec.resolved_uv_map[wb.name]
        elif wb.name in standalone_uv_faces:
            conv_uv_faces[wb.name] = standalone_uv_faces[wb.name]

    bg_img = _make_empty_studio_bg(canvas_size)
    view_metrics: list[dict[str, float]] = []
    standalone_psnrs: list[float] = []
    front_raw_img: Image.Image | None = None
    front_conv_img: Image.Image | None = None

    for v_idx, (azim, elev) in enumerate(((36.0, 24.0), (-144.0, 22.0))):
        if same_world_scale and raw_pts.size and conv_pts.size:
            joint_pts = np.concatenate([raw_pts, conv_pts], axis=0)
            b_raw = _camera_bounds_from_points(joint_pts, azim, elev)
            b_conv = b_raw
        else:
            b_raw = _camera_bounds_from_points(raw_pts, azim, elev)
            b_conv = _camera_bounds_from_points(conv_pts, azim, elev)

        img_raw = rasterize_raw_source_bundle(
            raw_bundle,
            canvas_size=canvas_size,
            azimuth_deg=azim,
            elevation_deg=elev,
            camera_bounds=b_raw,
            draw_contours=False,
        )
        img_conv = _rasterize_scene_with_atlas(
            conv_scene,
            {"faces": conv_uv_faces},
            conv_atlas,
            canvas_size=canvas_size,
            azimuth_deg=azim,
            elevation_deg=elev,
            camera_bounds=b_conv,
            draw_contours=False,
        )
        m = _compute_image_pair_metrics(img_raw, img_conv, bg_img)
        view_metrics.append(m)

        img_standalone = _rasterize_scene_with_atlas(
            conv_scene,
            {"faces": standalone_uv_faces},
            conv_atlas,
            canvas_size=canvas_size,
            azimuth_deg=azim,
            elevation_deg=elev,
            camera_bounds=b_conv,
            draw_contours=False,
        )
        m_st = _compute_image_pair_metrics(img_raw, img_standalone, bg_img)
        standalone_psnrs.append(m_st["psnr_db"])

        if v_idx == 0:
            front_raw_img = img_raw
            front_conv_img = img_conv

    assert front_raw_img is not None and front_conv_img is not None

    # Panel 3: Self-contained procedural multi-material Block-DSL (@palette + @materials + @style)
    proc_desc, proc_rgba = _compile_program_atlas_rgba(reparsed, seed=101, subdivisions_per_unit=4)
    # Strip authored uv_faces temporarily so _rasterize_scene_with_atlas samples proc_desc's packed UV islands!
    proc_boxes = [
        _RawBox(name=wb.name, center=wb.center, half_extents=wb.half_extents, rotation=wb.rotation, uv_faces=None)
        for wb in conv_scene.boxes
    ]
    proc_img = _rasterize_scene_with_atlas(
        _RawScene(boxes=tuple(proc_boxes)),  # type: ignore[arg-type]
        proc_desc,
        proc_rgba,
        canvas_size=canvas_size,
        azimuth_deg=36.0,
        elevation_deg=24.0,
        draw_contours=True,
    )

    # Panel 4: Animated @behavior pose (with artist PNG if available, else procedural) OR back-3/4 alt view
    posed_scene = conv_scene
    is_animated_pose = False
    if reparsed.behavior and reparsed.behavior.clips:
        try:
            from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip

            best_clip = max(
                reparsed.behavior.clips,
                key=lambda c: len(c.tracks) + len(c.keyframes) + len(c.muscle_drives) + len(c.ik_drives),
            )
            frames = evaluate_behavior_clip(reparsed, best_clip.name, num_frames=6)
            best_disp = -1.0
            for fr in frames[1:]:
                disp = float(
                    sum(
                        np.linalg.norm(fr.scene.box_by_name[b.name].center - b.center)
                        + 4.0 * np.linalg.norm(fr.scene.box_by_name[b.name].rotation - b.rotation)
                        for b in conv_scene.boxes
                        if b.name in fr.scene.box_by_name
                    )
                )
                if disp > best_disp:
                    best_disp = disp
                    posed_scene = fr.scene
            is_animated_pose = best_disp > 1e-4
        except Exception:
            posed_scene = conv_scene

    if rec.has_source_texture:
        pose_img = _rasterize_scene_with_atlas(
            posed_scene,
            {"faces": conv_uv_faces},
            conv_atlas,
            canvas_size=canvas_size,
            azimuth_deg=-34.0 if is_animated_pose else -138.0,
            elevation_deg=22.0,
            draw_contours=False,
        )
    else:
        posed_proc_boxes = [
            _RawBox(name=wb.name, center=wb.center, half_extents=wb.half_extents, rotation=wb.rotation, uv_faces=None)
            for wb in posed_scene.boxes
        ]
        pose_img = _rasterize_scene_with_atlas(
            _RawScene(boxes=tuple(posed_proc_boxes)),  # type: ignore[arg-type]
            proc_desc,
            proc_rgba,
            canvas_size=canvas_size,
            azimuth_deg=-34.0 if is_animated_pose else -138.0,
            elevation_deg=22.0,
            draw_contours=True,
        )

    metrics = {
        "psnr_db": round(float(np.mean([m["psnr_db"] for m in view_metrics])), 2),
        "mae_rgb": round(float(np.mean([m["mae_rgb"] for m in view_metrics])), 3),
        "silhouette_iou": round(float(np.mean([m["silhouette_iou"] for m in view_metrics])), 4),
        "standalone_dsl_psnr_db": round(float(np.mean(standalone_psnrs)), 2),
        "vertex_reconstruction_error": round(float(rec.vertex_reconstruction_error), 8),
        "has_source_texture": bool(rec.has_source_texture),
        "material_slots": len(reparsed.palettes),
        "is_animated_pose": bool(is_animated_pose),
    }
    return metrics, (front_raw_img, front_conv_img, proc_img, pose_img)


def render_visual_identity_proof_sheet(
    proof_cards: list[dict[str, Any]],
    out_png_path: Path | str,
    *,
    title: str = "SyncCraft v3.2 Visual Identity Proof -- Raw Source vs Converted Block-DSL vs @materials vs Pose",
) -> Path:
    """Render a 4-column visual-identity proof sheet comparing raw source assets to converted Block-DSL v3.2."""
    out_path = Path(out_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = max(1, len(proof_cards))
    panel_w, panel_h = 220, 220
    card_w = 4 * panel_w + 5 * 12  # 940 px
    card_h = panel_h + 74  # 294 px
    pad = 16
    header_h = 58
    sheet_w = pad * 2 + card_w
    sheet_h = header_h + pad + n_rows * (card_h + pad)

    sheet = Image.new("RGB", (sheet_w, sheet_h), (14, 17, 24))
    draw = ImageDraw.Draw(sheet)

    draw.rectangle((0, 0, sheet_w, header_h), fill=(22, 27, 39))
    draw.text((18, 12), title.encode("ascii", "replace").decode("ascii"), fill=(236, 243, 255))
    mean_psnr = float(np.mean([c["metrics"]["psnr_db"] for c in proof_cards])) if proof_cards else 0.0
    mean_iou = float(np.mean([c["metrics"]["silhouette_iou"] for c in proof_cards])) if proof_cards else 0.0
    draw.text(
        (18, 33),
        f"Assets audited on sheet: {len(proof_cards)} | Mean Raw-vs-BlockDSL PSNR: {mean_psnr:.2f} dB | Mean Silhouette IoU: {mean_iou:.4f}",
        fill=(146, 220, 182),
    )

    for idx, card in enumerate(proof_cards):
        x0 = pad
        y0 = header_h + pad + idx * (card_h + pad)
        m = card["metrics"]
        p_raw, p_conv, p_proc, p_pose = card["panels"]

        ok_col = (72, 168, 122) if m["silhouette_iou"] >= 0.96 else (218, 152, 64)
        draw.rectangle((x0, y0, x0 + card_w, y0 + card_h), fill=(20, 24, 34), outline=ok_col, width=2)

        hdr = (
            f"[{card['platform']}] {card['title']}  --  by {card['author']}  "
            f"({card['format_kind']}, {card['box_count']} boxes, {card['bone_count']} bones, {card['clip_count']} clips)"
        )
        draw.text((x0 + 12, y0 + 8), hdr[:128].encode("ascii", "replace").decode("ascii"), fill=(242, 246, 255))

        sub = (
            f"Visual Identity: PSNR={m['psnr_db']:.2f} dB (standalone DSL={m['standalone_dsl_psnr_db']:.2f} dB) | "
            f"MAE={m['mae_rgb']:.2f}/255 | Silhouette IoU={m['silhouette_iou']:.4f} | "
            f"VertErr={m['vertex_reconstruction_error']:.6f} | @materials slots={m['material_slots']}"
        )
        draw.text((x0 + 12, y0 + 25), sub.encode("ascii", "replace").decode("ascii"), fill=(154, 214, 184))

        vy = y0 + 46
        captions = [
            "1. Raw Source + Artist PNG" if m["has_source_texture"] else "1. Raw Source Geometry",
            "2. Block-DSL v3.2 + Artist PNG" if m["has_source_texture"] else "2. Block-DSL v3.2 Geometry",
            f"3. Block-DSL @materials ({m['material_slots']} slots)",
            "4. Block-DSL @behavior Pose" if m["is_animated_pose"] else "4. Block-DSL Back-3/4 View",
        ]
        cap_cols = [(190, 210, 240), (138, 236, 188), (240, 208, 152), (186, 196, 248)]
        for ci, (im_p, cap, ccol) in enumerate(zip((p_raw, p_conv, p_proc, p_pose), captions, cap_cols)):
            px = x0 + 12 + ci * (panel_w + 12)
            sheet.paste(im_p.resize((panel_w, panel_h), resample=Image.Resampling.NEAREST), (px, vy))
            draw.rectangle((px, vy, px + panel_w, vy + 18), fill=(16, 20, 28))
            draw.text((px + 5, vy + 3), cap, fill=ccol)

    sheet.save(out_path, format="PNG")
    return out_path
