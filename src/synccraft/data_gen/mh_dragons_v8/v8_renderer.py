"""src/synccraft/data_gen/mh_dragons_v8/v8_renderer.py

Fast Vectorized Perspective Z-Buffer Studio Renderer for x32 Blockbench Assets.

Matches the exact outward-face winding, RGBA alpha-cutout (`alpha >= 128`),
Nearest-Neighbor texel filtering, ACES Filmic tone mapping, soft cast floor
shadows, and ZYX bone Euler animation evaluation of `viewer/index.html` and
Blockbench — with zero wireframe toy outlines and zero external Blender dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .v8_painter import FACES


def _uniform_filter_2d(arr: np.ndarray, size: int = 5) -> np.ndarray:
    """Fast separable 2D box blur with edge replication (pure NumPy)."""
    if size <= 1:
        return arr.copy()
    r = size // 2
    pad_y = np.pad(arr, ((r, size - 1 - r), (0, 0)), mode="edge")
    cs_y = np.cumsum(np.pad(pad_y, ((1, 0), (0, 0)), mode="constant"), axis=0)
    blur_y = (cs_y[size:] - cs_y[:-size]) / float(size)
    pad_x = np.pad(blur_y, ((0, 0), (r, size - 1 - r)), mode="edge")
    cs_x = np.cumsum(np.pad(pad_x, ((0, 0), (1, 0)), mode="constant"), axis=1)
    return (cs_x[:, size:] - cs_x[:, :-size]) / float(size)


def _rot_xyz_deg(euler_deg: Sequence[float]) -> np.ndarray:
    """Right-handed column-vector matrix Rz @ Ry @ Rx matching SyncCraft & Three.js order='ZYX'."""
    rx, ry, rz = [math.radians(float(a)) for a in euler_deg]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mz @ my @ mx


def _sample_track_keys(keys: list[list[Any]], t: float, duration: float, loop: bool) -> np.ndarray:
    if not keys:
        return np.zeros(3, dtype=np.float64)
    if duration > 1e-9:
        t = (t % duration) if loop else min(max(0.0, t), duration)
    if t <= float(keys[0][0]):
        return np.asarray(keys[0][1], dtype=np.float64)
    if t >= float(keys[-1][0]):
        return np.asarray(keys[-1][1], dtype=np.float64)
    for i in range(len(keys) - 1):
        t0, v0 = float(keys[i][0]), np.asarray(keys[i][1], dtype=np.float64)
        t1, v1 = float(keys[i + 1][0]), np.asarray(keys[i + 1][1], dtype=np.float64)
        if t0 <= t <= t1:
            span = max(1e-9, t1 - t0)
            u = (t - t0) / span
            # Smooth Hermite / Catmull-Rom tangent interpolation when >= 4 keys
            p_prev = np.asarray(keys[max(0, i - 1)][1], dtype=np.float64)
            p_next = np.asarray(keys[min(len(keys) - 1, i + 2)][1], dtype=np.float64)
            u2, u3 = u * u, u * u * u
            return 0.5 * (
                (2.0 * v0)
                + (-p_prev + v1) * u
                + (2.0 * p_prev - 5.0 * v0 + 4.0 * v1 - p_next) * u2
                + (-p_prev + 3.0 * v0 - 3.0 * v1 + p_next) * u3
            )
    return np.asarray(keys[-1][1], dtype=np.float64)


def compute_bone_world_transforms(
    bones: list[dict[str, Any]],
    clip: dict[str, Any] | None = None,
    t_sec: float = 0.0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return `{bone_name: (R_world_3x3, t_world_3)}` for rest or animated pose `t_sec`."""
    rot_deltas: dict[str, np.ndarray] = {}
    pos_deltas: dict[str, np.ndarray] = {}
    if clip is not None:
        dur = float(clip.get("duration", 1.0))
        loop = bool(clip.get("loop", False))
        for tr in clip.get("tracks", []):
            bname = tr["bone"]
            val = _sample_track_keys(tr["keys"], t_sec, dur, loop)
            if tr.get("channel", "rotation") == "position":
                pos_deltas[bname] = val
            elif tr.get("channel") == "rotation":
                rot_deltas[bname] = val

    by_name = {b["name"]: b for b in bones}
    memo: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _solve(name: str) -> tuple[np.ndarray, np.ndarray]:
        if name in memo:
            return memo[name]
        b = by_name[name]
        piv = np.asarray(b["pivot"], dtype=np.float64)
        r_loc = _rot_xyz_deg(rot_deltas[name]) if name in rot_deltas else np.eye(3, dtype=np.float64)
        dp = pos_deltas.get(name, np.zeros(3, dtype=np.float64))
        parent = b.get("parent")
        if parent is None or parent not in by_name:
            # World point x transforms as: piv + dp + r_loc @ (x - piv) = r_loc @ x + (piv + dp - r_loc @ piv)
            r_w = r_loc
            t_w = piv + dp - r_loc @ piv
        else:
            r_p, t_p = _solve(parent)
            # Local bone frame at rest pivot `piv`: x' = r_p @ (piv + dp + r_loc @ (x - piv)) + t_p
            r_w = r_p @ r_loc
            t_w = r_p @ (piv + dp - r_loc @ piv) + t_p
        memo[name] = (r_w, t_w)
        return r_w, t_w

    for b in bones:
        _solve(b["name"])
    return memo


def _face_png_uv_corners(face: str, rect: Sequence[float]) -> np.ndarray:
    u0, v0, u1, v1 = [float(x) for x in rect[:4]]
    corners = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    if face in ("north", "east"):
        corners = [(1.0 - u, v) for u, v in corners]
    elif face == "up":
        corners = [(u, 1.0 - v) for u, v in corners]
    return np.array([[u0 + u * (u1 - u0), v0 + v * (v1 - v0)] for u, v in corners], dtype=np.float64)


def _aces_filmic(rgb_lin: np.ndarray) -> np.ndarray:
    x = np.maximum(0.0, rgb_lin * 1.12)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    # Linear to sRGB
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * np.power(np.maximum(mapped, 1e-12), 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def _compute_Focus_bounds(
    posed_quads: list[tuple[np.ndarray, np.ndarray, bool, float]],
    head_only: bool = False,
    head_bone_names: set[str] | None = None,
    quad_bones: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pts_list: list[np.ndarray] = []
    if head_only and head_bone_names and quad_bones:
        for (verts, _, _, _), bname in zip(posed_quads, quad_bones):
            if bname in head_bone_names:
                pts_list.append(verts)
    if not pts_list:
        pts_list = [q[0] for q in posed_quads]
    all_pts = np.concatenate(pts_list, axis=0)
    return all_pts.min(axis=0), all_pts.max(axis=0)


def build_natural_attitude_clip(doc: dict[str, Any]) -> dict[str, Any]:
    """`blockbench-sculpted-style` Line 319: Posing the bind pose with subtle living attitude
    (slight head pitch/yaw, jaw snarl, curved tail sway, and poised wing lift) so static
    hero/front renders read alive rather than stiff T-poses.
    """
    bnames = {b["name"] for b in doc["bones"]}
    tracks: list[dict[str, Any]] = []

    def _rot(bname: str, rx: float, ry: float, rz: float) -> None:
        if bname in bnames:
            tracks.append({"bone": bname, "channel": "rotation", "keys": [(0.0, [rx, ry, rz]), (1.0, [rx, ry, rz])]})

    _rot("neck_1", -3.0, 3.5, 0.0)
    _rot("neck_2", -2.5, 4.0, -1.5)
    _rot("head", 4.5, 5.5, -2.5)
    _rot("head_cranium", 4.5, 5.5, -2.5)
    _rot("jaw", 3.5, 0.0, 0.0)
    for ti in range(1, 9):
        _rot(f"tail_{ti}", 1.8 if ti <= 3 else -2.2, 3.8 + 0.6 * ti, 0.0)
    _rot("wing_l", 0.0, -4.0, 4.5)
    _rot("wing_r", 0.0, 3.0, -2.5)
    return {"duration": 1.0, "loop": True, "tracks": tracks}


def _select_focus_bones(view: str, bones: list[dict[str, Any]]) -> set[str]:
    low_view = view.lower()
    if low_view.startswith("head"):
        keys = ("head", "skull", "jaw", "mandible", "neck_2", "neck_3", "horn", "feeler", "hood", "crest", "antenna", "rail_horn")
    elif low_view.startswith("limbs"):
        keys = ("fore_", "hind_", "leg", "shin", "thigh", "paw", "arm")
    elif low_view.startswith("tail"):
        keys = ("tail_", "pelvis")
    else:
        return set()
    return {b["name"] for b in bones if any(k in b["name"].lower() for k in keys)}


def render_studio_frame(
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    atlas_rgba: np.ndarray,
    *,
    emissive_rgba: np.ndarray | None = None,
    view: str = "hero",
    clip: dict[str, Any] | None = None,
    t_sec: float = 0.0,
    width: int = 1200,
    height: int = 900,
    clay: bool = False,
    silhouette: bool = False,
    draw_ground_shadow: bool = True,
) -> Image.Image:
    """Fast Nearest-Neighbor x32 Blockbench studio & regional renderer with crisp cel shading."""
    bones = doc["bones"]
    eff_clip = clip
    is_regional = any(view.startswith(p) for p in ("head", "limbs", "tail"))
    if eff_clip is None and view in ("hero", "front", "back_34") and not silhouette and not is_regional:
        eff_clip = build_natural_attitude_clip(doc)
    tf_map = compute_bone_world_transforms(bones, clip=eff_clip, t_sec=t_sec)
    mats = doc.get("materials", {})
    tex_h, tex_w = atlas_rgba.shape[:2]

    posed_quads: list[tuple[np.ndarray, np.ndarray, bool, float]] = []
    quad_bones: list[str] = []
    for b in boxes:
        bname = b["bone"]
        r_w, t_w = tf_map[bname]
        hidden = set(b.get("hidden_faces", ()))
        is_plane = any(s == 0.0 for s in b["size"])
        for face in FACES:
            if face in hidden:
                continue
            fverts = np.asarray(b["face_vertices"][face], dtype=np.float64) @ r_w.T + t_w
            uv_px = _face_png_uv_corners(face, b["uv"][face])
            mat_key = b.get("face_materials", {}).get(face) or b["material"]
            if is_plane and face in ("west", "down", "north") and b.get("back_material"):
                mat_key = b["back_material"]
            emissive = float(mats.get(mat_key, {}).get("emissive", 0.0))
            posed_quads.append((fverts, uv_px, is_plane, emissive))
            quad_bones.append(bname)

    focus_bones = _select_focus_bones(view, bones)

    # Camera presets (+Z is front snout, +X is monster left, +Y is up)
    if view == "hero":
        az_deg, el_deg, fov_deg, fill_frac = 38.0, 20.0, 32.0, 0.84
    elif view == "front":
        az_deg, el_deg, fov_deg, fill_frac = 8.0, 14.0, 32.0, 0.84
    elif view == "back_34":
        az_deg, el_deg, fov_deg, fill_frac = 142.0, 22.0, 32.0, 0.84
    elif view == "side":
        az_deg, el_deg, fov_deg, fill_frac = 86.0, 12.0, 30.0, 0.86
    elif view == "top":
        az_deg, el_deg, fov_deg, fill_frac = 18.0, 68.0, 32.0, 0.84
    elif view == "head_hero":
        az_deg, el_deg, fov_deg, fill_frac = 36.0, 14.0, 28.0, 0.82
    elif view == "head_side":
        az_deg, el_deg, fov_deg, fill_frac = 84.0, 8.0, 28.0, 0.82
    elif view == "head_front":
        az_deg, el_deg, fov_deg, fill_frac = 4.0, 10.0, 28.0, 0.82
    elif view in ("limbs_hero", "tail_hero"):
        az_deg, el_deg, fov_deg, fill_frac = 42.0, 18.0, 30.0, 0.80
    elif view in ("limbs_side", "tail_side"):
        az_deg, el_deg, fov_deg, fill_frac = 86.0, 12.0, 30.0, 0.82
    elif view in ("limbs_front", "tail_front"):
        az_deg, el_deg, fov_deg, fill_frac = 8.0, 14.0, 30.0, 0.82
    else:
        az_deg, el_deg, fov_deg, fill_frac = 38.0, 20.0, 32.0, 0.84

    az, el = math.radians(az_deg), math.radians(el_deg)
    cam_dir = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)], dtype=np.float64)
    aspect = width / max(1.0, float(height))
    half_tan_y = math.tan(math.radians(fov_deg * 0.5))
    half_tan_x = half_tan_y * aspect

    fwd = -cam_dir / np.linalg.norm(cam_dir)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    focus_pts_list: list[np.ndarray] = []
    if focus_bones:
        for (verts, _, _, _), bname in zip(posed_quads, quad_bones):
            if bname in focus_bones:
                focus_pts_list.append(verts)
    if not focus_pts_list:
        focus_pts_list = [q[0] for q in posed_quads]
    focus_pts = np.concatenate(focus_pts_list, axis=0)
    px_cam = focus_pts @ right
    py_cam = focus_pts @ up
    pz_cam = focus_pts @ (-fwd)
    target = (
        right * (0.5 * (px_cam.min() + px_cam.max()))
        + up * (0.5 * (py_cam.min() + py_cam.max()))
        + (-fwd) * (0.5 * (pz_cam.min() + pz_cam.max()))
    )
    rel_x = np.abs(focus_pts @ right - float(target @ right))
    rel_y = np.abs(focus_pts @ up - float(target @ up))
    rel_d = (focus_pts - target[None, :]) @ cam_dir
    req_dist_x = np.max(rel_x / (half_tan_x * fill_frac) + rel_d)
    req_dist_y = np.max(rel_y / (half_tan_y * fill_frac) + rel_d)
    cam_dist = max(14.0, float(max(req_dist_x, req_dist_y)))
    cam_pos = target + cam_dir * cam_dist

    # Studio slate gradient backdrop (matching `Lagiacrus, reconstructed` Page 1)
    yy_bg, xx_bg = np.mgrid[0:height, 0:width]
    ry_n = (yy_bg / max(1.0, height - 1.0)) - 0.45
    rx_n = (xx_bg / max(1.0, width - 1.0)) - 0.50
    vign = np.clip(1.0 - 0.50 * (rx_n * rx_n + ry_n * ry_n), 0.42, 1.0)
    bg_lin = np.dstack((0.036 * vign, 0.052 * vign, 0.082 * vign)).astype(np.float64)

    full_min_y = min(float(q[0][:, 1].min()) for q in posed_quads)
    floor_y = min(0.0, full_min_y)
    key_dir = np.array([0.46, 0.78, 0.42], dtype=np.float64)
    key_dir /= np.linalg.norm(key_dir)
    fill_dir = np.array([-0.55, 0.35, -0.45], dtype=np.float64)
    fill_dir /= np.linalg.norm(fill_dir)

    if draw_ground_shadow and not is_regional and not silhouette and cam_pos[1] > floor_y + 1.0:
        ndc_x = ((xx_bg + 0.5) / width * 2.0 - 1.0) * half_tan_x
        ndc_y = (1.0 - (yy_bg + 0.5) / height * 2.0) * half_tan_y
        ray_y = fwd[1] + right[1] * ndc_x + up[1] * ndc_y
        hit_mask = ray_y < -1e-4
        t_hit = np.where(hit_mask, (floor_y - cam_pos[1]) / np.where(hit_mask, ray_y, -1.0), 0.0)
        hit_x = cam_pos[0] + t_hit * (fwd[0] + right[0] * ndc_x + up[0] * ndc_y)
        hit_z = cam_pos[2] + t_hit * (fwd[2] + right[2] * ndc_x + up[2] * ndc_y)

        q_centers = np.array([q[0].mean(axis=0) for q in posed_quads], dtype=np.float64)
        h_above = np.maximum(0.0, q_centers[:, 1] - floor_y)
        sh_x = q_centers[:, 0] - h_above * (key_dir[0] / key_dir[1])
        sh_z = q_centers[:, 2] - h_above * (key_dir[2] / key_dir[1])
        sx_lo, sx_hi = float(sh_x.min()) - 14.0, float(sh_x.max()) + 14.0
        sz_lo, sz_hi = float(sh_z.min()) - 14.0, float(sh_z.max()) + 14.0
        grid_res = 96
        gx = np.clip(((sh_x - sx_lo) / max(1.0, sx_hi - sx_lo) * (grid_res - 1)).astype(int), 0, grid_res - 1)
        gz = np.clip(((sh_z - sz_lo) / max(1.0, sz_hi - sz_lo) * (grid_res - 1)).astype(int), 0, grid_res - 1)
        sh_map = np.zeros((grid_res, grid_res), dtype=np.float64)
        np.add.at(sh_map, (gz, gx), 1.0)
        sh_map = _uniform_filter_2d(np.minimum(1.0, sh_map * 0.45), size=7)
        sh_map = _uniform_filter_2d(sh_map, size=5)

        grid_mask = hit_mask & (np.abs(hit_x) <= 96.0) & (np.abs(hit_z) <= 96.0)
        if np.any(grid_mask):
            gx_m = np.abs(((hit_x[grid_mask] + 8.0) % 16.0) - 8.0)
            gz_m = np.abs(((hit_z[grid_mask] + 8.0) % 16.0) - 8.0)
            line_m = (gx_m > 7.65) | (gz_m > 7.65)
            fade_g = np.exp(-0.00009 * (np.maximum(hit_x[grid_mask] ** 2, hit_z[grid_mask] ** 2)))
            bg_lin[grid_mask] += (line_m * 0.016 * fade_g)[:, None]

        u_fl = (hit_x - sx_lo) / max(1.0, sx_hi - sx_lo) * (grid_res - 1)
        v_fl = (hit_z - sz_lo) / max(1.0, sz_hi - sz_lo) * (grid_res - 1)
        in_box = hit_mask & (u_fl >= 0) & (u_fl < grid_res - 1) & (v_fl >= 0) & (v_fl < grid_res - 1)
        if np.any(in_box):
            ui = np.clip(u_fl[in_box].astype(int), 0, grid_res - 1)
            vi = np.clip(v_fl[in_box].astype(int), 0, grid_res - 1)
            occ = sh_map[vi, ui]
            dist_fade = np.exp(-0.00014 * (t_hit[in_box] ** 2))
            floor_tint = 1.0 - 0.42 * occ * dist_fade
            bg_lin[in_box] *= floor_tint[:, None]

    z_buf = np.full((height, width), np.inf, dtype=np.float64)
    color_lin = bg_lin.copy()
    glow_lin = np.zeros((height, width, 3), dtype=np.float64)

    atlas_srgb = atlas_rgba[..., :3].astype(np.float64) / 255.0
    atlas_lin = np.where(atlas_srgb <= 0.04045, atlas_srgb / 12.92, np.power((atlas_srgb + 0.055) / 1.055, 2.4))
    atlas_alpha = atlas_rgba[..., 3]
    em_alpha_tex = emissive_rgba[..., 3].astype(np.float64) / 255.0 if (emissive_rgba is not None and emissive_rgba.shape[:2] == atlas_rgba.shape[:2]) else None
    clay_lin = np.array([0.22, 0.27, 0.34], dtype=np.float64)

    # Pre-filter & sort visible quads front-to-back for fast early-z culling!
    col_c = np.arange(width, dtype=np.float64) + 0.5
    row_c = np.arange(height, dtype=np.float64) + 0.5
    near_z = 1.5
    visible_work: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for fverts, uv_px, is_plane, emissive in posed_quads:
        e1 = fverts[1] - fverts[0]
        e2 = fverts[2] - fverts[0]
        n_raw = np.cross(e1, e2)
        n_len = float(np.linalg.norm(n_raw))
        if n_len < 1e-8:
            continue
        normal = n_raw / n_len
        to_cam = cam_pos - fverts[0]
        facing = float(np.dot(normal, to_cam))
        if facing <= 0.0:
            continue

        rel = fverts - cam_pos[None, :]
        cz = rel @ fwd
        if np.any(cz <= near_z):
            continue
        cx = rel @ right
        cy = rel @ up
        sx = (cx / (cz * half_tan_x) * 0.5 + 0.5) * width
        sy = (0.5 - cy / (cz * half_tan_y) * 0.5) * height

        ndotl = max(0.0, float(np.dot(normal, key_dir)))
        ndotfill = max(0.0, float(np.dot(normal, fill_dir)))
        hemi = 0.50 + 0.50 * float(normal[1])
        view_vec = to_cam / max(1e-8, float(np.linalg.norm(to_cam)))
        rim = pow(max(0.0, 1.0 - abs(float(np.dot(normal, view_vec)))), 2.8) * 0.10
        shade_rgb = (
            np.array([0.52, 0.56, 0.64]) * (0.56 + 0.28 * hemi)
            + np.array([1.02, 0.96, 0.88]) * (0.50 * ndotl)
            + np.array([0.62, 0.76, 0.96]) * (0.18 * ndotfill + rim)
        )
        visible_work.append((float(cz.min()), sx, sy, cz, uv_px, shade_rgb, emissive))

    visible_work.sort(key=lambda item: item[0])

    for _, sx, sy, cz, uv_px, shade_rgb, emissive in visible_work:
        u_lo, u_hi = min(uv_px[:, 0]), max(uv_px[:, 0])
        v_lo, v_hi = min(uv_px[:, 1]), max(uv_px[:, 1])
        for tri in ((0, 1, 2), (0, 2, 3)):
            i0, i1, i2 = tri
            x0, y0, inv_z0 = sx[i0], sy[i0], 1.0 / cz[i0]
            x1, y1, inv_z1 = sx[i1], sy[i1], 1.0 / cz[i1]
            x2, y2, inv_z2 = sx[i2], sy[i2], 1.0 / cz[i2]

            px_min = max(0, int(math.floor(min(x0, x1, x2))))
            px_max = min(width - 1, int(math.ceil(max(x0, x1, x2))))
            py_min = max(0, int(math.floor(min(y0, y1, y2))))
            py_max = min(height - 1, int(math.ceil(max(y0, y1, y2))))
            if px_min > px_max or py_min > py_max:
                continue

            z_slice = z_buf[py_min : py_max + 1, px_min : px_max + 1]
            tri_min_z = min(cz[i0], cz[i1], cz[i2])
            if float(z_slice.max()) < tri_min_z - 1e-4:
                continue

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-8:
                continue
            inv_den = 1.0 / denom

            px_c = col_c[px_min : px_max + 1][None, :]
            py_c = row_c[py_min : py_max + 1][:, None]

            w0 = ((y1 - y2) * (px_c - x2) + (x2 - x1) * (py_c - y2)) * inv_den
            w1 = ((y2 - y0) * (px_c - x2) + (x0 - x2) * (py_c - y2)) * inv_den
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
            if not np.any(inside):
                continue

            inv_z = w0 * inv_z0 + w1 * inv_z1 + w2 * inv_z2
            z_cam = 1.0 / np.maximum(1e-8, inv_z)
            vis = inside & (z_cam < z_slice - 1e-4)
            if not np.any(vis):
                continue

            w0v, w1v, w2v, zv = w0[vis], w1[vis], w2[vis], z_cam[vis]
            u_p = (w0v * (uv_px[i0, 0] * inv_z0) + w1v * (uv_px[i1, 0] * inv_z1) + w2v * (uv_px[i2, 0] * inv_z2)) * zv
            v_p = (w0v * (uv_px[i0, 1] * inv_z0) + w1v * (uv_px[i1, 1] * inv_z1) + w2v * (uv_px[i2, 1] * inv_z2)) * zv

            tx = np.clip(np.floor(np.clip(u_p, u_lo + 1e-4, u_hi - 1e-4)).astype(int), 0, tex_w - 1)
            ty = np.clip(np.floor(np.clip(v_p, v_lo + 1e-4, v_hi - 1e-4)).astype(int), 0, tex_h - 1)

            alpha_vals = atlas_alpha[ty, tx]
            opaque = alpha_vals >= 128
            if not np.any(opaque):
                continue

            vis_idx_y, vis_idx_x = np.nonzero(vis)
            oy = vis_idx_y[opaque]
            ox = vis_idx_x[opaque]
            tx_op = tx[opaque]
            ty_op = ty[opaque]

            z_slice[oy, ox] = zv[opaque]
            if silhouette:
                continue
            base_c = clay_lin[None, :] if clay else atlas_lin[ty_op, tx_op]
            lit_c = base_c * shade_rgb[None, :]
            em_px = np.maximum(emissive, em_alpha_tex[ty_op, tx_op]) if em_alpha_tex is not None else np.full(len(tx_op), emissive, dtype=np.float64)
            if not clay and np.any(em_px > 0.25):
                em_boost = base_c * (em_px[:, None] * 0.65)
                lit_c = lit_c + em_boost
                glow_lin[py_min + oy, px_min + ox] = base_c * np.maximum(0.0, em_px[:, None] - 0.45) * 0.55
            else:
                glow_lin[py_min + oy, px_min + ox] = 0.0
            color_lin[py_min + oy, px_min + ox] = lit_c

    fg_mask = np.isfinite(z_buf)
    if silhouette:
        sil_u8 = np.full((height, width, 3), (228, 232, 240), dtype=np.uint8)
        sil_u8[fg_mask] = (18, 23, 32)
        return Image.fromarray(sil_u8, mode="RGB")

    # Crisp depth cavity ambient occlusion highlighting armor overlap seams
    if np.any(fg_mask):
        z_safe = np.where(fg_mask, z_buf, float(z_buf[fg_mask].max()) + 20.0)
        z_blur = _uniform_filter_2d(z_safe, size=5)
        cavity = np.clip((z_safe - z_blur) * 0.085, 0.0, 0.22)
        ao_factor = np.where(fg_mask, 1.0 - cavity, 1.0)
        color_lin *= ao_factor[:, :, None]

    # Tight controlled focal glow ONLY around bright emissive gems/eyes (ZERO fog washout!)
    if not clay and float(np.max(glow_lin)) > 0.04:
        bloom = np.zeros_like(glow_lin)
        for ch_i in range(3):
            b1 = _uniform_filter_2d(glow_lin[:, :, ch_i], size=5)
            bloom[:, :, ch_i] = 0.22 * b1
        color_lin += bloom

    out_u8 = _aces_filmic(color_lin)
    return Image.fromarray(out_u8, mode="RGB")


def render_stage_and_regional_sheet(
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    atlas_rgba: np.ndarray,
    *,
    emissive_rgba: np.ndarray | None = None,
    stage1_doc: dict[str, Any] | None = None,
    stage1_boxes: list[dict[str, Any]] | None = None,
    output_path: Path,
) -> Path:
    """Render a 1920x1040 Stage-Progression & Regional Crop Visual Inspection Sheet (matching PDF Pages 3-5):
      Row 1: [1] Stage 1 Primary Mass | [2] Full 3/4 Back View | [3] Head Hero | [4] Head Side | [5] Head Front
      Row 2: [6] Limbs Hero | [7] Limbs Front | [8] Tail Hero | [9] Tail Side | [10] Silhouette + Enlarged Swatch Strip
    """
    cw, ch = 1920, 1020
    sheet = Image.new("RGB", (cw, ch), (12, 17, 23))
    draw = ImageDraw.Draw(sheet)

    def _lf(sz: int, bold: bool = False):
        cands = (
            ["DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]
            if bold
            else ["DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
        )
        for fp in cands:
            try:
                return ImageFont.truetype(fp, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    f_title = _lf(22, True)
    f_sub = _lf(14, False)
    f_lbl = _lf(12, True)

    n_total = len(boxes)
    n_planes = sum(1 for b in boxes if any(s == 0.0 for s in b["size"]))
    n_cuboids = n_total - n_planes
    tex_h, tex_w = atlas_rgba.shape[:2]

    draw.rectangle((0, 0, cw, 68), fill=(18, 25, 35))
    draw.text((20, 12), f"{doc['title'].upper()} - REGIONAL & STAGE INSPECTION SHEET", fill=(238, 244, 250), font=f_title)
    draw.text(
        (20, 42),
        f"Elements: {n_total} ({n_cuboids} cuboids + {n_planes} planes)  |  Bones: {len(doc['bones'])}  |  Atlas: {tex_w}x{tex_h}",
        fill=(152, 200, 232),
        font=f_sub,
    )

    pw, ph = 360, 410
    p_s1 = (
        render_studio_frame(stage1_doc, stage1_boxes, np.full((64, 64, 4), 255, dtype=np.uint8), view="hero", width=pw * 2, height=ph * 2, clay=True).resize((pw, ph), Image.Resampling.LANCZOS)
        if (stage1_doc and stage1_boxes)
        else render_studio_frame(doc, boxes, atlas_rgba, view="hero", width=pw * 2, height=ph * 2, clay=True).resize((pw, ph), Image.Resampling.LANCZOS)
    )
    p_back = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="back_34", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_h1 = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="head_hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_h2 = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="head_side", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_h3 = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="head_front", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)

    p_l1 = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="limbs_hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_l2 = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="limbs_front", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_t1 = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="tail_hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_t2 = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="tail_side", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)

    # Panel 10: Silhouette Check + Enlarged Material Swatch Preview from Atlas
    p_misc = Image.new("RGB", (pw, ph), (20, 26, 36))
    p_sil = render_studio_frame(doc, boxes, atlas_rgba, view="side", width=pw * 2, height=220 * 2, silhouette=True, draw_ground_shadow=False).resize((pw - 16, 210), Image.Resampling.LANCZOS)
    p_misc.paste(p_sil, (8, 8))
    atlas_im = Image.fromarray(atlas_rgba).convert("RGB")
    swatch_crop = atlas_im.crop((0, 0, min(atlas_im.width, 192), min(atlas_im.height, 96))).resize((pw - 16, 172), Image.Resampling.NEAREST)
    p_misc.paste(swatch_crop, (8, 228))

    rows = [
        [
            ("1. STAGE 1 PRIMARY MASS (CLAY)", p_s1),
            ("2. REAR 3/4 VIEW", p_back),
            ("3. HEAD HERO CLOSE-UP", p_h1),
            ("4. HEAD SIDE PROFILE", p_h2),
            ("5. HEAD FRONT SYMMETRY", p_h3),
        ],
        [
            ("6. LIMBS & CLAWS HERO", p_l1),
            ("7. LIMBS FRONT STANCE", p_l2),
            ("8. TAIL & DORSAL HERO", p_t1),
            ("9. TAIL SIDE CURVE", p_t2),
            ("10. SILHOUETTE + x32 ATLAS SWATCHES", p_misc),
        ],
    ]
    for r_i, row in enumerate(rows):
        y0 = 84 + r_i * (ph + 46)
        for c_i, (lbl, im) in enumerate(row):
            x0 = 20 + c_i * (pw + 20)
            draw.rounded_rectangle((x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 24), radius=5, fill=(20, 28, 38), outline=(46, 64, 88))
            sheet.paste(im, (x0, y0))
            draw.text((x0 + 8, y0 + ph + 4), lbl, fill=(196, 222, 245), font=f_lbl)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)
    return output_path


def render_monster_studio_review_sheet(
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    atlas_rgba: np.ndarray,
    clips: dict[str, dict[str, Any]],
    *,
    emissive_rgba: np.ndarray | None = None,
    ref_board_or_dir: Path | None,
    output_path: Path,
    signature_clip_names: Sequence[str] = (),
) -> Path:
    """Render an 8-panel 1920x1040 high-resolution Studio Review Sheet:
    Row 1: [1] Multi-Source Reference Grid (2x2) | [2] 3/4 Studio Hero | [3] Side Profile + Silhouette Check | [4] Head & Snout Macro
    Row 2: [5] Dorsal & Multi-Bank Wing Span | [6..8] 3 Peak Dynamic Monster Hunter Action Poses!
    """
    cw, ch = 1920, 1040
    sheet = Image.new("RGB", (cw, ch), (12, 17, 23))
    draw = ImageDraw.Draw(sheet)
    def _load_font(sz: int, bold: bool = False) -> Any:
        cands = (
            [
                "DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
            if bold
            else [
                "DejaVuSans.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
            ]
        )
        for fp in cands:
            try:
                return ImageFont.truetype(fp, sz)
            except OSError:
                continue
        return ImageFont.load_default()

    f_title = _load_font(22, bold=True)
    f_sub = _load_font(14, bold=False)
    f_lbl = _load_font(13, bold=True)

    n_total = len(boxes)
    n_planes = sum(1 for b in boxes if any(s == 0.0 for s in b["size"]))
    n_cuboids = n_total - n_planes
    tex_h, tex_w = atlas_rgba.shape[:2]

    draw.rectangle((0, 0, cw, 74), fill=(18, 25, 35))
    draw.text(
        (24, 14),
        f"{doc['title'].upper()} - {doc.get('title_epithet', '')}   [{doc['species_class']} | Sculpted Blockbench Style]",
        fill=(238, 244, 250),
        font=f_title,
    )
    tpu_lbl = f"{16 * int(doc.get('texels_per_unit', 1))}px/Block"
    draw.text(
        (24, 46),
        f"3D Cuboids: {n_cuboids} ({100.0 * n_cuboids / max(1, n_total):.0f}%)  |  Cutout Cards: {n_planes}  |  "
        f"Anatomical Bones: {len(doc['bones'])}  |  {tpu_lbl} Cel Atlas: {tex_w}x{tex_h}  |  "
        f"Authored Vigorous Clips: {len(clips)}",
        fill=(152, 200, 232),
        font=f_sub,
    )

    pw, ph = 450, 420

    # Panel 1: Wide Multi-Source Internet Reference Grid (2x2 across official renders, game screenshots, 3D & fan art!)
    ref_panel = Image.new("RGB", (pw, ph), (22, 28, 38))
    if ref_board_or_dir is not None:
        imgs: list[Image.Image] = []
        gal = ref_board_or_dir / "mh_gallery" / doc["id"]
        if gal.is_dir():
            for p in sorted(gal.glob("*.png"))[:2]:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
        wide_dir = ref_board_or_dir / "mh_wide_refs" / doc["id"]
        if wide_dir.is_dir():
            wide_cands = sorted(p for p in wide_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            # Lead with official lead/turnaround/screenshot renders cleanly
            for p in wide_cands[:4]:
                if len(imgs) >= 4:
                    break
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    pass
        if not imgs:
            for wb_cand in (
                ref_board_or_dir / f"{doc['id']}_wide_board.jpg",
                ref_board_or_dir / "mh_wide_boards" / f"{doc['id']}_wide_board.jpg",
                ref_board_or_dir / "mh_official_renders" / f"{doc['id']}.png",
            ):
                if wb_cand.is_file():
                    try:
                        wb_im = Image.open(wb_cand).convert("RGB")
                        if "_wide_board" in wb_cand.name and wb_im.width >= 1200 and wb_im.height >= 800:
                            # Crop top-row hero tiles `[1],[2],[5],[7]` cleanly out of the 4x4 wide contact sheet!
                            bw, bh = wb_im.width, wb_im.height
                            y0_r1, y1_r1 = int(bh * 0.045), int(bh * 0.255)
                            y0_r2, y1_r2 = int(bh * 0.285), int(bh * 0.495)
                            imgs.extend(
                                [
                                    wb_im.crop((int(bw * 0.015), y0_r1, int(bw * 0.245), y1_r1)),
                                    wb_im.crop((int(bw * 0.260), y0_r1, int(bw * 0.490), y1_r1)),
                                    wb_im.crop((int(bw * 0.015), y0_r2, int(bw * 0.245), y1_r2)),
                                    wb_im.crop((int(bw * 0.505), y0_r2, int(bw * 0.740), y1_r2)),
                                ]
                            )
                        else:
                            imgs.append(wb_im)
                        break
                    except Exception:
                        pass
        if len(imgs) >= 4:
            cw_s, ch_s = (pw - 18) // 2, (ph - 18) // 2
            for idx, im in enumerate(imgs[:4]):
                rr, cc = idx // 2, idx % 2
                sc = min(cw_s / im.width, ch_s / im.height)
                nw, nh = max(1, int(im.width * sc)), max(1, int(im.height * sc))
                slot = Image.new("RGB", (cw_s, ch_s), (236, 240, 246))
                slot.paste(im.resize((nw, nh), Image.Resampling.LANCZOS), ((cw_s - nw) // 2, (ch_s - nh) // 2))
                ref_panel.paste(slot, (6 + cc * (cw_s + 6), 6 + rr * (ch_s + 6)))
        elif len(imgs) >= 2:
            hh = (ph - 18) // 2
            for idx, im in enumerate(imgs[:2]):
                sc = min((pw - 12) / im.width, hh / im.height)
                nw, nh = max(1, int(im.width * sc)), max(1, int(im.height * sc))
                slot = Image.new("RGB", (pw - 12, hh), (236, 240, 246))
                slot.paste(im.resize((nw, nh), Image.Resampling.LANCZOS), ((pw - 12 - nw) // 2, (hh - nh) // 2))
                ref_panel.paste(slot, (6, 6 + idx * (hh + 6)))
        elif len(imgs) == 1:
            im = imgs[0]
            sc = min((pw - 12) / im.width, (ph - 12) / im.height)
            nw, nh = max(1, int(im.width * sc)), max(1, int(im.height * sc))
            ref_panel.paste(im.resize((nw, nh), Image.Resampling.LANCZOS), ((pw - nw) // 2, (ph - nh) // 2))

    p_hero = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_side = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="side", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    # `blockbench-sculpted-style` Line 82: Flat Dark Silhouette Readability Check inset on Side Profile!
    iw, ih = 136, 98
    p_sil = render_studio_frame(doc, boxes, atlas_rgba, view="side", width=iw * 2, height=ih * 2, silhouette=True, draw_ground_shadow=False).resize((iw, ih), Image.Resampling.LANCZOS)
    sd_draw = ImageDraw.Draw(p_side)
    sd_draw.rectangle((pw - iw - 10, ph - ih - 10, pw - 7, ph - 7), outline=(92, 128, 168), width=2)
    p_side.paste(p_sil, (pw - iw - 8, ph - ih - 8))
    p_head = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="head_hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_top = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="top", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)

    # Pick 3 vigorous action clips (roar + 2 signature attacks) at their peak excursion frame
    cand_clips = list(signature_clip_names)
    for k in clips:
        if k not in cand_clips and k not in ("idle", "alert", "hurt", "walk"):
            cand_clips.append(k)
    chosen_clips = (cand_clips + list(clips.keys()))[:3]

    action_panels: list[tuple[str, Image.Image]] = []
    for c_idx, cname in enumerate(chosen_clips):
        cdata = clips[cname]
        dur = float(cdata.get("duration", 2.0))
        # Find peak keyframe timestamp
        best_t = dur * 0.48
        best_mag = -1.0
        for sample_i in range(1, 16):
            ts = dur * (sample_i / 16.0)
            mag = 0.0
            for tr in cdata.get("tracks", []):
                v = _sample_track_keys(tr["keys"], ts, dur, bool(cdata.get("loop", False)))
                mag += float(np.sum(np.abs(v)))
            if mag > best_mag:
                best_mag = mag
                best_t = ts
        v_mode = "hero" if c_idx != 1 else "front"
        im_act = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view=v_mode, clip=cdata, t_sec=best_t, width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
        action_panels.append((f"{6 + c_idx}. ACTION: {cname.upper()} (t={best_t:.2f}s)", im_act))

    row1 = [
        ("1. MULTI-SOURCE RENDERS, 3D & ART REFS", ref_panel),
        ("2. 3/4 SCULPTED HERO (x32 CALM CEL)", p_hero),
        ("3. SIDE PROFILE + SILHOUETTE CHECK", p_side),
        ("4. SCULPTED SNOUT, FANGS & HORNS MACRO", p_head),
    ]
    row2 = [("5. DORSAL & FAN-LAYERED WING ARCHITECTURE", p_top)] + action_panels

    for r_i, row in enumerate((row1, row2)):
        y0 = 92 + r_i * (ph + 52)
        for c_i, (lbl, im) in enumerate(row):
            x0 = 24 + c_i * (pw + 24)
            draw.rounded_rectangle((x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 26), radius=6, fill=(20, 28, 38), outline=(46, 64, 88))
            sheet.paste(im, (x0, y0))
            draw.text((x0 + 10, y0 + ph + 5), lbl, fill=(232, 185, 135) if r_i == 1 and c_i >= 1 else (196, 222, 245), font=f_lbl)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    return output_path


def render_animated_clip_gif(
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    atlas_rgba: np.ndarray,
    clip: dict[str, Any],
    output_gif_path: Path,
    *,
    emissive_rgba: np.ndarray | None = None,
    fps: int = 16,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Render an actual multi-frame animated GIF of a vigorous Monster Hunter signature attack."""
    dur = float(clip.get("duration", 2.0))
    n_frames = max(16, int(round(dur * fps)))
    frames: list[Image.Image] = []
    for i in range(n_frames):
        t_s = dur * (i / n_frames)
        fr = render_studio_frame(doc, boxes, atlas_rgba, emissive_rgba=emissive_rgba, view="hero", clip=clip, t_sec=t_s, width=width * 2, height=height * 2).resize((width, height), Image.Resampling.LANCZOS)
        frames.append(fr)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(30, int(round(1000.0 / fps))),
        loop=0,
        optimize=False,
    )
    return output_gif_path
