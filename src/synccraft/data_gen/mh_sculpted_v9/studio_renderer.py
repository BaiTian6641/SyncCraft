"""Fast Shared-SE(3) Perspective Z-Buffer Studio Renderer for Native Block-DSL v4.1 Scenes.

Renders directly from SyncCraft's compiled `Scene` (`compile_geometry(program)`) and
animated `BehaviorFrameSample.scene` (`evaluate_behavior_clip(program, clip_name)`),
guaranteeing 100% geometric & kinematic agreement with `audit_scene`, `export_bbmodel`,
and `export_bedrock`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synccraft.dsl.ast_nodes import Program
from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip
from synccraft.kinematics.se3_chain import WorldBox, Scene

from .artisan_painter import FACES, PLANE_PRIMARY_FACES


def _uniform_filter_2d(arr: np.ndarray, size: int = 5) -> np.ndarray:
    if size <= 1:
        return arr.copy()
    r = size // 2
    pad_y = np.pad(arr, ((r, size - 1 - r), (0, 0)), mode="edge")
    cs_y = np.cumsum(np.pad(pad_y, ((1, 0), (0, 0)), mode="constant"), axis=0)
    blur_y = (cs_y[size:] - cs_y[:-size]) / float(size)
    pad_x = np.pad(blur_y, ((0, 0), (r, size - 1 - r)), mode="edge")
    cs_x = np.cumsum(np.pad(pad_x, ((0, 0), (1, 0)), mode="constant"), axis=1)
    return (cs_x[:, size:] - cs_x[:, :-size]) / float(size)


def _aces_filmic_u8(rgb_lin: np.ndarray) -> np.ndarray:
    x = np.maximum(0.0, rgb_lin * 1.15)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    srgb = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * np.power(np.maximum(mapped, 1e-12), 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def _box_outward_quads_world(box: WorldBox) -> dict[str, np.ndarray]:
    pnorm = getattr(box, "plane_normal", None)
    inf = float(getattr(box, "inflate", 0.0))
    hx = 0.0 if pnorm == "x" else (0.5 * float(box.size[0]) + inf)
    hy = 0.0 if pnorm == "y" else (0.5 * float(box.size[1]) + inf)
    hz = 0.0 if pnorm == "z" else (0.5 * float(box.size[2]) + inf)
    local_faces = {
        "north": np.array([[+hx, -hy, -hz], [-hx, -hy, -hz], [-hx, +hy, -hz], [+hx, +hy, -hz]], dtype=np.float64),
        "south": np.array([[-hx, -hy, +hz], [+hx, -hy, +hz], [+hx, +hy, +hz], [-hx, +hy, +hz]], dtype=np.float64),
        "east": np.array([[+hx, -hy, +hz], [+hx, -hy, -hz], [+hx, +hy, -hz], [+hx, +hy, +hz]], dtype=np.float64),
        "west": np.array([[-hx, -hy, -hz], [-hx, -hy, +hz], [-hx, +hy, +hz], [-hx, +hy, -hz]], dtype=np.float64),
        "up": np.array([[-hx, +hy, +hz], [+hx, +hy, +hz], [+hx, +hy, -hz], [-hx, +hy, -hz]], dtype=np.float64),
        "down": np.array([[-hx, -hy, -hz], [+hx, -hy, -hz], [+hx, -hy, +hz], [-hx, -hy, +hz]], dtype=np.float64),
    }
    if pnorm in PLANE_PRIMARY_FACES:
        active = PLANE_PRIMARY_FACES[pnorm]
        return {f: local_faces[f] @ box.rotation.T + box.center[None, :] for f in active}
    return {f: v @ box.rotation.T + box.center[None, :] for f, v in local_faces.items()}


def _face_uv_corners(rect: Sequence[float]) -> np.ndarray:
    u0, v0, u1, v1 = [float(x) for x in rect[:4]]
    return np.array([[u0, v1], [u1, v1], [u1, v0], [u0, v0]], dtype=np.float64)


def _load_font(sz: int, bold: bool = False) -> Any:
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


def render_scene_frame(
    scene: Scene,
    atlas_faces: dict[str, dict[str, list[int]]] | None,
    atlas_rgba: np.ndarray | None,
    *,
    emissive_rgba: np.ndarray | None = None,
    program: Program | None = None,
    view: str = "hero",
    width: int = 1200,
    height: int = 900,
    clay: bool = False,
    silhouette: bool = False,
    draw_ground_shadow: bool = True,
) -> Image.Image:
    """Render a SyncCraft `Scene` from any studio or regional camera."""
    em_materials: set[str] = set()
    if program is not None:
        for p in program.palettes:
            if p.emissive:
                em_materials.add(p.name)
        for m in program.materials:
            if float(m.emissive) > 0.15:
                em_materials.add(m.name)

    if atlas_rgba is None or atlas_faces is None:
        clay = True
        tex_h, tex_w = 4, 4
        atlas_rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
    else:
        tex_h, tex_w = atlas_rgba.shape[:2]

    posed_quads: list[tuple[np.ndarray, np.ndarray, float, str]] = []
    for box in scene.boxes:
        qmap = _box_outward_quads_world(box)
        em_val = 0.85 if box.material in em_materials else 0.0
        for face, fverts in qmap.items():
            rect = atlas_faces[box.name][face] if (atlas_faces and box.name in atlas_faces) else [0, 0, 2, 2]
            uv_px = _face_uv_corners(rect)
            posed_quads.append((fverts, uv_px, em_val, box.bone_name))

    is_head_macro = view.startswith("head")
    head_keys = ("head", "skull", "cranium", "snout", "muzzle", "jaw", "mandible", "beak", "visor", "horn", "antler", "feeler", "hood", "crest", "vane", "tiara", "rail")
    focus_bones = {
        b.name for b in scene.bones if any(k in b.name.lower() for k in head_keys)
    } if is_head_macro else set()

    if view == "hero":
        az_deg, el_deg, fov_deg, fill_frac = 38.0, 19.0, 31.0, 0.85
    elif view == "front":
        az_deg, el_deg, fov_deg, fill_frac = 8.0, 14.0, 31.0, 0.84
    elif view == "back_34":
        az_deg, el_deg, fov_deg, fill_frac = 142.0, 21.0, 31.0, 0.84
    elif view == "side":
        az_deg, el_deg, fov_deg, fill_frac = 88.0, 10.0, 29.0, 0.86
    elif view == "top":
        az_deg, el_deg, fov_deg, fill_frac = 16.0, 68.0, 32.0, 0.85
    elif view == "head_hero":
        az_deg, el_deg, fov_deg, fill_frac = 36.0, 13.0, 27.0, 0.82
    elif view == "head_side":
        az_deg, el_deg, fov_deg, fill_frac = 86.0, 7.0, 27.0, 0.82
    else:
        az_deg, el_deg, fov_deg, fill_frac = 38.0, 19.0, 31.0, 0.85

    az, el = math.radians(az_deg), math.radians(el_deg)
    cam_dir = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)], dtype=np.float64)
    aspect = width / max(1.0, float(height))
    half_tan_y = math.tan(math.radians(fov_deg * 0.5))
    half_tan_x = half_tan_y * aspect

    fwd = -cam_dir / np.linalg.norm(cam_dir)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(right) < 1e-6 else (right / np.linalg.norm(right))
    up = np.cross(right, fwd)

    focus_list = [q[0] for q in posed_quads if (not focus_bones or q[3] in focus_bones)]
    if not focus_list:
        focus_list = [q[0] for q in posed_quads]
    focus_pts = np.concatenate(focus_list, axis=0)

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

    yy_bg, xx_bg = np.mgrid[0:height, 0:width]
    ry_n = (yy_bg / max(1.0, height - 1.0)) - 0.45
    rx_n = (xx_bg / max(1.0, width - 1.0)) - 0.50
    vign = np.clip(1.0 - 0.52 * (rx_n * rx_n + ry_n * ry_n), 0.40, 1.0)
    bg_lin = np.dstack((0.032 * vign, 0.046 * vign, 0.074 * vign)).astype(np.float64)

    full_min_y = min(float(q[0][:, 1].min()) for q in posed_quads) if posed_quads else 0.0
    floor_y = min(0.0, full_min_y)
    key_dir = np.array([0.44, 0.78, 0.44], dtype=np.float64)
    key_dir /= np.linalg.norm(key_dir)
    fill_dir = np.array([-0.55, 0.36, -0.45], dtype=np.float64)
    fill_dir /= np.linalg.norm(fill_dir)

    if draw_ground_shadow and not is_head_macro and not silhouette and cam_pos[1] > floor_y + 1.0 and posed_quads:
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
        sh_map = _uniform_filter_2d(_uniform_filter_2d(np.minimum(1.0, sh_map * 0.45), size=7), size=5)

        grid_mask = hit_mask & (np.abs(hit_x) <= 96.0) & (np.abs(hit_z) <= 96.0)
        if np.any(grid_mask):
            gx_m = np.abs(((hit_x[grid_mask] + 8.0) % 16.0) - 8.0)
            gz_m = np.abs(((hit_z[grid_mask] + 8.0) % 16.0) - 8.0)
            line_m = (gx_m > 7.65) | (gz_m > 7.65)
            fade_g = np.exp(-0.00009 * (np.maximum(hit_x[grid_mask] ** 2, hit_z[grid_mask] ** 2)))
            bg_lin[grid_mask] += (line_m * 0.015 * fade_g)[:, None]

        u_fl = (hit_x - sx_lo) / max(1.0, sx_hi - sx_lo) * (grid_res - 1)
        v_fl = (hit_z - sz_lo) / max(1.0, sz_hi - sz_lo) * (grid_res - 1)
        in_box = hit_mask & (u_fl >= 0) & (u_fl < grid_res - 1) & (v_fl >= 0) & (v_fl < grid_res - 1)
        if np.any(in_box):
            ui = np.clip(u_fl[in_box].astype(int), 0, grid_res - 1)
            vi = np.clip(v_fl[in_box].astype(int), 0, grid_res - 1)
            occ = sh_map[vi, ui]
            dist_fade = np.exp(-0.00014 * (t_hit[in_box] ** 2))
            bg_lin[in_box] *= (1.0 - 0.44 * occ * dist_fade)[:, None]

    z_buf = np.full((height, width), np.inf, dtype=np.float64)
    color_lin = bg_lin.copy()
    glow_lin = np.zeros((height, width, 3), dtype=np.float64)

    atlas_srgb = atlas_rgba[..., :3].astype(np.float64) / 255.0
    atlas_lin = np.where(atlas_srgb <= 0.04045, atlas_srgb / 12.92, np.power((atlas_srgb + 0.055) / 1.055, 2.4))
    atlas_alpha = atlas_rgba[..., 3]
    em_alpha_tex = (
        emissive_rgba[..., 3].astype(np.float64) / 255.0
        if (emissive_rgba is not None and emissive_rgba.shape[:2] == atlas_rgba.shape[:2])
        else None
    )
    clay_lin = np.array([0.26, 0.32, 0.40], dtype=np.float64)

    col_c = np.arange(width, dtype=np.float64) + 0.5
    row_c = np.arange(height, dtype=np.float64) + 0.5
    near_z = 1.5
    visible_work: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []

    for fverts, uv_px, em_val, _ in posed_quads:
        e1 = fverts[1] - fverts[0]
        e2 = fverts[2] - fverts[0]
        n_raw = np.cross(e1, e2)
        n_len = float(np.linalg.norm(n_raw))
        if n_len < 1e-8:
            continue
        normal = n_raw / n_len
        to_cam = cam_pos - fverts[0]
        if float(np.dot(normal, to_cam)) <= 0.0:
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
        rim = pow(max(0.0, 1.0 - abs(float(np.dot(normal, view_vec)))), 2.6) * 0.14
        # Balanced studio lighting so painted Blockbench colors stay vibrant & crisp
        shade_rgb = (
            np.array([0.56, 0.59, 0.66]) * (0.58 + 0.28 * hemi)
            + np.array([1.02, 0.97, 0.90]) * (0.48 * ndotl)
            + np.array([0.64, 0.78, 0.98]) * (0.18 * ndotfill + rim)
        )
        visible_work.append((float(cz.min()), sx, sy, cz, uv_px, shade_rgb, em_val))

    visible_work.sort(key=lambda item: item[0])

    for _, sx, sy, cz, uv_px, shade_rgb, em_val in visible_work:
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

            opaque = np.ones(len(tx), dtype=bool) if clay else (atlas_alpha[ty, tx] >= 128)
            if not np.any(opaque):
                continue

            vis_idx_y, vis_idx_x = np.nonzero(vis)
            oy, ox = vis_idx_y[opaque], vis_idx_x[opaque]
            tx_op, ty_op = tx[opaque], ty[opaque]

            z_slice[oy, ox] = zv[opaque]
            if silhouette:
                continue
            base_c = clay_lin[None, :] if clay else atlas_lin[ty_op, tx_op]
            lit_c = base_c * shade_rgb[None, :]
            em_px = (
                np.maximum(em_val, em_alpha_tex[ty_op, tx_op])
                if (not clay and em_alpha_tex is not None)
                else np.full(len(tx_op), 0.0 if clay else em_val, dtype=np.float64)
            )
            if not clay and np.any(em_px > 0.20):
                lit_c = lit_c + base_c * (em_px[:, None] * 0.72)
                glow_lin[py_min + oy, px_min + ox] = base_c * np.maximum(0.0, em_px[:, None] - 0.35) * 0.65
            else:
                glow_lin[py_min + oy, px_min + ox] = 0.0
            color_lin[py_min + oy, px_min + ox] = lit_c

    fg_mask = np.isfinite(z_buf)
    if silhouette:
        sil_u8 = np.full((height, width, 3), (232, 236, 244), dtype=np.uint8)
        sil_u8[fg_mask] = (16, 20, 28)
        return Image.fromarray(sil_u8, mode="RGB")

    if np.any(fg_mask):
        z_safe = np.where(fg_mask, z_buf, float(z_buf[fg_mask].max()) + 20.0)
        z_blur = _uniform_filter_2d(z_safe, size=5)
        cavity = np.clip((z_safe - z_blur) * 0.08, 0.0, 0.20)
        color_lin *= np.where(fg_mask, 1.0 - cavity, 1.0)[:, :, None]

    if np.any(glow_lin > 1e-4):
        for ch_i in range(3):
            glow_lin[..., ch_i] = _uniform_filter_2d(_uniform_filter_2d(glow_lin[..., ch_i], size=9), size=7)
        color_lin = color_lin + glow_lin * 0.85

    return Image.fromarray(_aces_filmic_u8(color_lin), mode="RGB")


def _pick_peak_clip_frame(program: Program, clip_name: str, samples_count: int = 15) -> tuple[Scene, float]:
    samples = evaluate_behavior_clip(program, clip_name, num_frames=samples_count)
    best_s = samples[len(samples) // 2]
    best_mag = -1.0
    for s in samples[1:-1]:
        rot_mag = sum(abs(v[0]) + abs(v[1]) + abs(v[2]) for v in s.pose_euler_deltas.values())
        pos_mag = sum(abs(c) * 3.5 for c in s.root_translation_delta)
        tot = rot_mag + pos_mag
        if tot > best_mag:
            best_mag = tot
            best_s = s
    return best_s.scene, float(best_s.time_s)


def render_stage_progression_sheet(
    title: str,
    stage1_scene: Scene | None,
    stage2_scene: Scene | None,
    final_scene: Scene,
    atlas_faces: dict[str, dict[str, list[int]]],
    atlas_rgba: np.ndarray,
    *,
    emissive_rgba: np.ndarray | None = None,
    program: Program | None = None,
    style_score: float = 0.0,
    output_path: Path,
) -> Path:
    """Render an 8-panel Stage-by-Stage & 4-View Silhouette Visual Check Sheet (`SKILL.md` §2)."""
    cw, ch = 1880, 1020
    sheet = Image.new("RGB", (cw, ch), (12, 17, 24))
    draw = ImageDraw.Draw(sheet)
    f_h1 = _load_font(21, bold=True)
    f_sub = _load_font(14, bold=False)
    f_lbl = _load_font(13, bold=True)

    s1_cnt = len(stage1_scene.boxes) if stage1_scene is not None else 0
    s2_cnt = len(stage2_scene.boxes) if stage2_scene is not None else 0
    s3_cnt = len(final_scene.boxes)

    draw.rectangle((0, 0, cw, 72), fill=(18, 25, 36))
    draw.text(
        (24, 14),
        f"{title.upper()} — PROGRESSIVE BLOCK-DSL v4.1 SCULPTOR STAGES & SILHOUETTE AUDIT",
        fill=(240, 245, 252),
        font=f_h1,
    )
    draw.text(
        (24, 44),
        f"Stage 1 Primary Masses: {s1_cnt} elems  ->  Stage 2 Secondary Anatomy: {s2_cnt} elems  ->  "
        f"Stage 3 Textured Masterwork: {s3_cnt} elems ({len(final_scene.bones)} bones)  |  "
        f"Sculpted Style Anchor Score: {style_score:.3f}",
        fill=(146, 202, 238),
        font=f_sub,
    )

    pw, ph = 440, 405
    p1 = render_scene_frame(stage1_scene or final_scene, None, None, view="hero", width=pw * 2, height=ph * 2, clay=True).resize((pw, ph), Image.Resampling.LANCZOS)
    p2 = render_scene_frame(stage2_scene or final_scene, None, None, view="hero", width=pw * 2, height=ph * 2, clay=True).resize((pw, ph), Image.Resampling.LANCZOS)
    p3 = render_scene_frame(final_scene, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view="hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p4 = render_scene_frame(final_scene, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view="head_side", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)

    s_side = render_scene_frame(final_scene, None, None, view="side", width=pw * 2, height=ph * 2, silhouette=True).resize((pw, ph), Image.Resampling.LANCZOS)
    s_front = render_scene_frame(final_scene, None, None, view="front", width=pw * 2, height=ph * 2, silhouette=True).resize((pw, ph), Image.Resampling.LANCZOS)
    s_top = render_scene_frame(final_scene, None, None, view="top", width=pw * 2, height=ph * 2, silhouette=True).resize((pw, ph), Image.Resampling.LANCZOS)
    p_back = render_scene_frame(final_scene, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view="back_34", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)

    rows = [
        [
            (f"1. STAGE 1 BLOCKOUT ({s1_cnt} primary masses)", p1),
            (f"2. STAGE 2 SECONDARY ANATOMY ({s2_cnt} elems)", p2),
            (f"3. STAGE 3 VIBRANT MASTERWORK ({s3_cnt} elems)", p3),
            ("4. REGIONAL CHECK: HEAD & SNOUT PROFILE", p4),
        ],
        [
            ("5. SILHOUETTE CHECK: SIDE GESTURE", s_side),
            ("6. SILHOUETTE CHECK: FRONT STANCE", s_front),
            ("7. SILHOUETTE CHECK: TOP WING/FIN SPAN", s_top),
            ("8. REAR 3/4 DORSAL & TAIL FLOW", p_back),
        ],
    ]
    for r_i, row in enumerate(rows):
        y0 = 90 + r_i * (ph + 50)
        for c_i, (lbl, im) in enumerate(row):
            x0 = 24 + c_i * (pw + 24)
            draw.rounded_rectangle((x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 26), radius=6, fill=(20, 28, 38), outline=(48, 66, 90))
            sheet.paste(im, (x0, y0))
            draw.text((x0 + 10, y0 + ph + 5), lbl, fill=(198, 224, 246), font=f_lbl)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    return output_path


def render_monster_masterwork_review_sheet(
    *,
    title: str,
    title_epithet: str,
    species_class: str,
    program: Program,
    scene: Scene,
    atlas_faces: dict[str, dict[str, list[int]]],
    atlas_rgba: np.ndarray,
    emissive_rgba: np.ndarray | None,
    reference_image_path: Path | None,
    style_score: float,
    output_path: Path,
    signature_clip_names: Sequence[str] = (),
) -> Path:
    """Render high-res 1920x1040 8-panel Studio Review Sheet comparing Reference vs Model & Vigorous Actions."""
    cw, ch = 1920, 1040
    sheet = Image.new("RGB", (cw, ch), (12, 17, 23))
    draw = ImageDraw.Draw(sheet)
    f_title = _load_font(22, bold=True)
    f_sub = _load_font(14, bold=False)
    f_lbl = _load_font(13, bold=True)

    n_total = len(scene.boxes)
    n_planes = sum(1 for b in scene.boxes if getattr(b, "plane_normal", None) is not None or float(np.min(b.size)) <= 0.0)
    n_cuboids = n_total - n_planes
    tex_h, tex_w = atlas_rgba.shape[:2]
    n_clips = len(program.behavior.clips) if program.behavior is not None else 0

    draw.rectangle((0, 0, cw, 74), fill=(18, 25, 35))
    draw.text(
        (24, 14),
        f"{title.upper()} - {title_epithet.strip()}   [{species_class} | Block-DSL v4.1 Sculpted Masterwork]",
        fill=(240, 246, 252),
        font=f_title,
    )
    draw.text(
        (24, 46),
        f"3D Cuboids: {n_cuboids} ({100.0 * n_cuboids / max(1, n_total):.0f}%)  |  Cutout Planes: {n_planes}  |  "
        f"Bones: {len(scene.bones)}  |  32px/Block Vibrant Cel Atlas: {tex_w}x{tex_h}  |  "
        f"Vigorous Clips: {n_clips}  |  Style Anchor: {style_score:.3f}",
        fill=(152, 204, 238),
        font=f_sub,
    )

    pw, ph = 450, 420
    ref_panel = Image.new("RGB", (pw, ph), (22, 28, 38))
    if reference_image_path is not None and reference_image_path.is_file():
        im = Image.open(reference_image_path).convert("RGB")
        sc = min(pw / im.width, ph / im.height)
        nw, nh = max(1, int(im.width * sc)), max(1, int(im.height * sc))
        ref_panel.paste(im.resize((nw, nh), Image.Resampling.LANCZOS), ((pw - nw) // 2, (ph - nh) // 2))

    # Render Hero with subtle living idle poise if `idle` exists, otherwise rest scene
    hero_sc = scene
    if program.behavior is not None and any(c.name == "idle" for c in program.behavior.clips):
        idle_samps = evaluate_behavior_clip(program, "idle", sample_phases=[0.28])
        if idle_samps:
            hero_sc = idle_samps[0].scene

    p_hero = render_scene_frame(hero_sc, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view="hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_side = render_scene_frame(scene, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view="side", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)

    iw, ih = 140, 100
    p_sil = render_scene_frame(scene, None, None, view="side", width=iw * 2, height=ih * 2, silhouette=True, draw_ground_shadow=False).resize((iw, ih), Image.Resampling.LANCZOS)
    sd_draw = ImageDraw.Draw(p_side)
    sd_draw.rectangle((pw - iw - 10, ph - ih - 10, pw - 7, ph - 7), outline=(96, 134, 174), width=2)
    p_side.paste(p_sil, (pw - iw - 8, ph - ih - 8))

    p_head = render_scene_frame(scene, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view="head_hero", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
    p_top = render_scene_frame(scene, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view="top", width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)

    all_clip_names = [c.name for c in program.behavior.clips] if program.behavior is not None else []
    cands: list[str] = [c for c in signature_clip_names if c in all_clip_names]
    for c in all_clip_names:
        if c not in cands and c not in ("idle", "walk", "enraged_taunt"):
            cands.append(c)
    for c in all_clip_names:
        if c not in cands:
            cands.append(c)
    chosen = cands[:3]

    action_panels: list[tuple[str, Image.Image]] = []
    for idx, cname in enumerate(chosen):
        act_sc, t_pk = _pick_peak_clip_frame(program, cname)
        v_mode = "hero" if idx != 1 else "front"
        im_act = render_scene_frame(act_sc, atlas_faces, atlas_rgba, emissive_rgba=emissive_rgba, program=program, view=v_mode, width=pw * 2, height=ph * 2).resize((pw, ph), Image.Resampling.LANCZOS)
        action_panels.append((f"{6 + idx}. ACTION: {cname.upper()} (t={t_pk:.2f}s)", im_act))
    while len(action_panels) < 3:
        action_panels.append((f"{6 + len(action_panels)}. POSE STUDY", p_hero.copy()))

    row1 = [
        ("1. MULTI-SOURCE RENDERS, 3D & ART REFS", ref_panel),
        ("2. 3/4 SCULPTED HERO (x32 VIBRANT CEL)", p_hero),
        ("3. SIDE PROFILE + SILHOUETTE CHECK", p_side),
        ("4. SCULPTED SNOUT, FANGS & HORNS MACRO", p_head),
    ]
    row2 = [("5. DORSAL & NON-UNI-SHEET WING/FIN SPAN", p_top)] + action_panels

    for r_i, row in enumerate((row1, row2)):
        y0 = 92 + r_i * (ph + 52)
        for c_i, (lbl, im) in enumerate(row):
            x0 = 24 + c_i * (pw + 24)
            draw.rounded_rectangle((x0 - 2, y0 - 2, x0 + pw + 2, y0 + ph + 26), radius=6, fill=(20, 28, 38), outline=(46, 64, 88))
            sheet.paste(im, (x0, y0))
            draw.text((x0 + 10, y0 + ph + 5), lbl, fill=(238, 190, 132) if r_i == 1 and c_i >= 1 else (196, 222, 245), font=f_lbl)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    return output_path


def render_animated_action_gif(
    program: Program,
    clip_name: str,
    atlas_faces: dict[str, dict[str, list[int]]],
    atlas_rgba: np.ndarray,
    output_gif_path: Path,
    *,
    emissive_rgba: np.ndarray | None = None,
    fps: int = 16,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Render a smooth multi-frame animated GIF of a vigorous Monster Hunter `@behavior` clip."""
    clip = next(c for c in program.behavior.clips if c.name == clip_name)
    n_frames = max(16, min(36, int(round(float(clip.duration_s) * fps))))
    samples = evaluate_behavior_clip(program, clip_name, num_frames=n_frames)
    frames: list[Image.Image] = []
    for s in samples:
        fr = render_scene_frame(
            s.scene,
            atlas_faces,
            atlas_rgba,
            emissive_rgba=emissive_rgba,
            program=program,
            view="hero",
            width=width * 2,
            height=height * 2,
        ).resize((width, height), Image.Resampling.LANCZOS)
        frames.append(fr)
    output_gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(35, int(round(1000.0 / fps))),
        loop=0,
        optimize=False,
    )
    return output_gif_path
