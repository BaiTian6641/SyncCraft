#!/usr/bin/env python3
"""Standard reusable Blockbench Studio Rendering, Reference Prep, Pixel-Relief Telemetry & Verification Kit (v39.4).

Import this module directly in any creature/asset build script instead of rewriting camera renderers,
turnaround grids, open-jaw close-ups, A2 filmstrips, reference downloaders, BB5 wing-continuity checks,
or craft telemetry from scratch.

Quick usage:
    import sys
    sys.path.insert(0, "/Users/tianrui6641/BlockBench-Skill/skills/blockbench-sculpted-style/scripts")
    import studio_render_kit as kit
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

SKILL_SCRIPTS_DIR = Path(__file__).resolve().parent
SKILL_ROOT_DIR = SKILL_SCRIPTS_DIR.parent


def _discover_synccraft_v20() -> Path:
    candidates = [
        SKILL_ROOT_DIR.parent.parent / "synccraft_v20",
        SKILL_ROOT_DIR.parent.parent,
        Path("/Users/tianrui6641/BlockBench-Skill/synccraft_v20"),
    ]
    for c in candidates:
        if (c / "src" / "synccraft").is_dir():
            return c
    return Path("/Users/tianrui6641/BlockBench-Skill/synccraft_v20")


SYNCCRAFT_V20_ROOT = _discover_synccraft_v20()


def ensure_synccraft_v20() -> Path:
    """Ensure synccraft_v20/src is first on sys.path and strip any read-only _synccraft_ro paths."""
    sys.path[:] = [p for p in sys.path if "_synccraft_ro" not in p]
    src = str(SYNCCRAFT_V20_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return SYNCCRAFT_V20_ROOT


ensure_synccraft_v20()

import numpy as np
from PIL import Image, ImageDraw

from synccraft.data_gen.mh_sculpted_v9.studio_renderer import (
    _box_outward_quads_world,
    _face_uv_corners,
    _load_font,
)
from synccraft.kinematics.musculoskeletal import evaluate_behavior_clip
from synccraft.kinematics.se3_chain import Scene, compile_geometry


# ---------------------------------------------------------------------------
# 1. Official Reference Downloader & Clean Neutral-Matte Converter
# ---------------------------------------------------------------------------

_BLOCKED_DOMAINS = ("fandom.com", "wikia.nocookie.net", "wikia.com")


def fetch_reference_images(
    urls_by_name: dict[str, str],
    out_dir: str | Path,
    *,
    convert_to_png: bool = True,
    matte_rgb: tuple[int, int, int] = (128, 128, 128),
    timeout_s: float = 25.0,
) -> dict[str, Path]:
    """Download non-Fandom official reference images cleanly and convert WebP/RGBA to matte-composited PNG."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36"}
    for name, url in urls_by_name.items():
        low = url.lower()
        if any(b in low for b in _BLOCKED_DOMAINS):
            raise ValueError(f"Blocked low-res hotlink domain in reference URL: {url}")
        ext = Path(url.split("?")[0]).suffix.lower() or ".png"
        raw_path = out_path / f"{name}{ext}"
        png_path = out_path / f"{name}.png"
        if not raw_path.exists() and not (convert_to_png and png_path.exists()):
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw_path.write_bytes(resp.read())
        if convert_to_png:
            src_file = raw_path if raw_path.exists() else png_path
            im = Image.open(src_file)
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (*matte_rgb, 255))
                im = Image.alpha_composite(bg, rgba).convert("RGB")
            else:
                im = im.convert("RGB")
            im.save(png_path, format="PNG")
            if raw_path != png_path and raw_path.exists():
                raw_path.unlink()
            saved[name] = png_path
        else:
            saved[name] = raw_path
    return saved


# ---------------------------------------------------------------------------
# 2. Core ACES-Filmic / Unlit / Silhouette Software Renderer
# ---------------------------------------------------------------------------

def _aces_u8(rgb_lin: np.ndarray) -> np.ndarray:
    x = np.maximum(0.0, rgb_lin * 1.12)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    srgb = np.where(
        mapped <= 0.0031308,
        12.92 * mapped,
        1.055 * np.power(np.maximum(mapped, 1e-12), 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def _unpack_atlas(atlas: dict[str, Any] | None) -> tuple[dict | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if not atlas:
        return None, None, None, None, None
    faces = atlas.get("faces") or atlas.get("uv_faces")
    albedo = atlas.get("albedo_rgba")
    if albedo is None:
        albedo = atlas.get("rgba")
    emissive = atlas.get("emissive_rgba")
    normal_n = atlas.get("normal_rgba")
    if normal_n is None:
        normal_n = atlas.get("labpbr_n")
    specular_s = atlas.get("specular_rgba")
    if specular_s is None:
        specular_s = atlas.get("labpbr_s")
    return faces, albedo, emissive, normal_n, specular_s


def render_view(
    scene: Scene,
    atlas_faces: dict[str, dict[str, list[int]]] | None = None,
    atlas_rgba: np.ndarray | None = None,
    *,
    emissive_rgba: np.ndarray | None = None,
    normal_rgba: np.ndarray | None = None,
    specular_rgba: np.ndarray | None = None,
    atlas: dict[str, Any] | None = None,
    az_deg: float = -36.0,
    el_deg: float = 16.0,
    fov_deg: float = 28.0,
    fill_frac: float = 0.84,
    width: int = 640,
    height: int = 480,
    focus_bone_filter: Callable[[str], bool] | None = None,
    hide_bone_filter: Callable[[str], bool] | None = None,
    box_filter: Callable[[Any], bool] | None = None,
    clay: bool = False,
    unlit: bool = False,
    silhouette: bool = False,
    draw_ground: bool = True,
    bg_rgb: tuple[float, float, float] = (0.045, 0.055, 0.085),
    material_colors: dict[str, tuple[int, int, int]] | None = None,
) -> Image.Image:
    """Render a crisp studio view of a SyncCraft Scene (LabPBR GGX, textured, clay, unlit, or silhouette)."""
    if atlas is not None and atlas_rgba is None:
        atlas_faces, atlas_rgba, em2, nrm2, spc2 = _unpack_atlas(atlas)
        if emissive_rgba is None:
            emissive_rgba = em2
        if normal_rgba is None:
            normal_rgba = nrm2
        if specular_rgba is None:
            specular_rgba = spc2

    use_atlas = atlas_rgba is not None and atlas_faces is not None and not clay and not silhouette
    tex_h, tex_w = atlas_rgba.shape[:2] if use_atlas else (2, 2)

    quads: list[tuple[np.ndarray, np.ndarray, str, str, str]] = []
    for box in scene.boxes:
        if hide_bone_filter is not None and hide_bone_filter(box.bone_name):
            continue
        if box_filter is not None and not box_filter(box):
            continue
        qmap = _box_outward_quads_world(box)
        for face, fverts in qmap.items():
            rect = atlas_faces[box.name][face] if (use_atlas and box.name in atlas_faces and face in atlas_faces[box.name]) else [0, 0, 1, 1]
            uv_px = _face_uv_corners(rect)
            quads.append((fverts, uv_px, box.bone_name, box.material, box.name))

    if not quads:
        return Image.new("RGB", (width, height), (36, 44, 58))

    focus_list = [q[0] for q in quads if (focus_bone_filter is None or focus_bone_filter(q[2]))]
    if not focus_list:
        focus_list = [q[0] for q in quads]
    focus_pts = np.concatenate(focus_list, axis=0)

    az, el = math.radians(az_deg), math.radians(el_deg)
    cam_dir = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)], dtype=np.float64)
    fwd = -cam_dir / np.linalg.norm(cam_dir)
    world_up = np.array([0.0, 0.0, -1.0]) if abs(el_deg) > 84.0 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    aspect = width / max(1.0, float(height))
    half_tan_y = math.tan(math.radians(fov_deg * 0.5))
    half_tan_x = half_tan_y * aspect

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
    req_dist = max(
        float(np.max(rel_x / (half_tan_x * fill_frac) + rel_d)),
        float(np.max(rel_y / (half_tan_y * fill_frac) + rel_d)),
        8.0,
    )
    cam_pos = target + cam_dir * req_dist

    if silhouette:
        img_lin = np.full((height, width, 3), 0.94, dtype=np.float64)
    else:
        img_lin = np.zeros((height, width, 3), dtype=np.float64)
        img_lin[:, :, 0], img_lin[:, :, 1], img_lin[:, :, 2] = bg_rgb

    if draw_ground and not silhouette and focus_bone_filter is None and cam_pos[1] > 0.5:
        yy, xx = np.mgrid[0:height, 0:width]
        ndc_x = ((xx + 0.5) / width * 2.0 - 1.0) * half_tan_x
        ndc_y = (1.0 - (yy + 0.5) / height * 2.0) * half_tan_y
        ray_x = fwd[0] + right[0] * ndc_x + up[0] * ndc_y
        ray_y = fwd[1] + right[1] * ndc_x + up[1] * ndc_y
        ray_z = fwd[2] + right[2] * ndc_x + up[2] * ndc_y
        mask = ray_y < -1e-4
        t_hit = np.where(mask, (0.0 - cam_pos[1]) / np.where(mask, ray_y, -1.0), -1.0)
        gx = cam_pos[0] + t_hit * ray_x
        gz = cam_pos[2] + t_hit * ray_z
        dist2 = (gx - target[0]) ** 2 + (gz - target[2]) ** 2
        rad = max(110.0, req_dist * 0.9)
        in_floor = mask & (t_hit > 0.0) & (dist2 < rad * rad)
        if np.any(in_floor):
            bx = np.floor(gx / 16.0).astype(np.int64)
            bz = np.floor(gz / 16.0).astype(np.int64)
            chk = ((bx + bz) & 1) == 0
            fade = np.clip(1.0 - np.sqrt(dist2) / rad, 0.0, 1.0)
            for c_i, b_c in enumerate(bg_rgb):
                img_lin[:, :, c_i] = np.where(
                    in_floor,
                    np.where(chk, b_c + 0.048 * fade, b_c + 0.022 * fade),
                    img_lin[:, :, c_i],
                )

    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    key_dir = np.array([0.40, 0.78, 0.48], dtype=np.float64)
    key_dir /= np.linalg.norm(key_dir)
    fill_dir = np.array([-0.55, 0.36, -0.42], dtype=np.float64)
    fill_dir /= np.linalg.norm(fill_dir)

    if use_atlas:
        albedo_f = atlas_rgba[:, :, :3].astype(np.float64) / 255.0
        albedo_lin = np.where(
            albedo_f <= 0.04045,
            albedo_f / 12.92,
            np.power((albedo_f + 0.055) / 1.055, 2.4),
        )
        alpha_u8 = atlas_rgba[:, :, 3]
        em_lin = None
        if emissive_rgba is not None:
            ef = emissive_rgba[:, :, :3].astype(np.float64) / 255.0
            em_lin = np.where(ef <= 0.04045, ef / 12.92, np.power((ef + 0.055) / 1.055, 2.4))
        use_pbr = (normal_rgba is not None or specular_rgba is not None) and not unlit and not silhouette
        if use_pbr:
            nrm_xy_ao = normal_rgba.astype(np.float64) / 255.0 if normal_rgba is not None else None
            spc_f = specular_rgba.astype(np.float64) if specular_rgba is not None else None
    else:
        use_pbr = False

    for fverts, uv_px, bname, mat, box_name in quads:
        rel = fverts - cam_pos[None, :]
        cz = rel @ fwd
        if np.any(cz <= 0.1):
            continue
        cx = rel @ right
        cy = rel @ up
        sx = ((cx / (cz * half_tan_x)) * 0.5 + 0.5) * width
        sy = (0.5 - (cy / (cz * half_tan_y)) * 0.5) * height

        e1 = fverts[1] - fverts[0]
        e2 = fverts[2] - fverts[0]
        e3 = fverts[3] - fverts[0]
        n = np.cross(e1, e2)
        nlen = float(np.linalg.norm(n))
        if nlen < 1e-8:
            continue
        n /= nlen
        face_ctr = fverts.mean(axis=0)
        view_dir = cam_pos - face_ctr
        vlen = float(np.linalg.norm(view_dir))
        view_dir = view_dir / max(1e-8, vlen)
        if float(np.dot(n, view_dir)) < 0.0:
            n = -n
        # Per-quad orthonormal tangent basis (T along +u, B along -v/OpenGL +Y) for true LabPBR normal mapping!
        t_vec = e1 / max(1e-8, float(np.linalg.norm(e1)))
        b_vec = -e3 / max(1e-8, float(np.linalg.norm(e3)))
        if float(uv_px[1, 0] - uv_px[0, 0]) < 0.0:
            t_vec = -t_vec
        half_k = key_dir + view_dir
        half_k /= max(1e-8, float(np.linalg.norm(half_k)))

        if silhouette or unlit:
            shade = 1.0
        else:
            nd_k = max(0.0, float(np.dot(n, key_dir)))
            nd_f = max(0.0, float(np.dot(n, fill_dir)))
            up_b = 0.5 + 0.5 * float(n[1])
            shade = 0.36 + 0.60 * nd_k + 0.18 * nd_f + 0.12 * up_b

        if not use_atlas and not silhouette:
            if material_colors and mat in material_colors:
                r8, g8, b8 = material_colors[mat]
                base_col = np.array([r8 / 255.0, g8 / 255.0, b8 / 255.0], dtype=np.float64) ** 2.2
            else:
                h_seed = abs(hash(mat)) % 360
                base_col = np.array([
                    0.42 + 0.18 * math.sin(math.radians(h_seed)),
                    0.46 + 0.16 * math.sin(math.radians(h_seed + 120)),
                    0.50 + 0.16 * math.sin(math.radians(h_seed + 240)),
                ], dtype=np.float64)

        for i0, i1, i2 in ((0, 1, 2), (0, 2, 3)):
            x0, y0, z0 = sx[i0], sy[i0], cz[i0]
            x1, y1, z1 = sx[i1], sy[i1], cz[i1]
            x2, y2, z2 = sx[i2], sy[i2], cz[i2]
            xmin = max(0, int(math.floor(min(x0, x1, x2))))
            xmax = min(width - 1, int(math.ceil(max(x0, x1, x2))))
            ymin = max(0, int(math.floor(min(y0, y1, y2))))
            ymax = min(height - 1, int(math.ceil(max(y0, y1, y2))))
            if xmin > xmax or ymin > ymax:
                continue
            area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
            if abs(area) < 1e-6:
                continue
            inv_a = 1.0 / area
            xs = np.arange(xmin, xmax + 1, dtype=np.float64) + 0.5
            ys = np.arange(ymin, ymax + 1, dtype=np.float64) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            w0 = ((x1 - gx) * (y2 - gy) - (y1 - gy) * (x2 - gx)) * inv_a
            w1 = ((x2 - gx) * (y0 - gy) - (y2 - gy) * (x0 - gx)) * inv_a
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside):
                continue

            inv_z = w0 / z0 + w1 / z1 + w2 / z2
            z_pix = 1.0 / np.maximum(inv_z, 1e-8)
            sub_z = zbuf[ymin : ymax + 1, xmin : xmax + 1]
            vis = inside & (z_pix < sub_z)
            if not np.any(vis):
                continue

            if use_atlas:
                u_p = (w0 * (uv_px[i0, 0] / z0) + w1 * (uv_px[i1, 0] / z1) + w2 * (uv_px[i2, 0] / z2)) * z_pix
                v_p = (w0 * (uv_px[i0, 1] / z0) + w1 * (uv_px[i1, 1] / z1) + w2 * (uv_px[i2, 1] / z2)) * z_pix
                ui = np.clip(np.floor(u_p).astype(np.int64), 0, tex_w - 1)
                vi = np.clip(np.floor(v_p).astype(np.int64), 0, tex_h - 1)
                vis = vis & (alpha_u8[vi, ui] >= 128)
                if not np.any(vis):
                    continue
                sub_z[vis] = z_pix[vis]
                vy, ux = vi[vis], ui[vis]
                base_c = albedo_lin[vy, ux]
                if use_pbr:
                    # True per-pixel Tangent-Space Normal Map + LabPBR 1.3 GGX Metallic/Roughness/AO/SSS!
                    if nrm_xy_ao is not None:
                        ns = nrm_xy_ao[vy, ux]
                        tnx = ns[:, 0] * 2.0 - 1.0
                        tny = ns[:, 1] * 2.0 - 1.0
                        tnz = np.sqrt(np.clip(1.0 - tnx * tnx - tny * tny, 0.05, 1.0))
                        wn = tnx[:, None] * t_vec[None, :] + tny[:, None] * b_vec[None, :] + tnz[:, None] * n[None, :]
                        wn = wn / np.maximum(1e-8, np.linalg.norm(wn, axis=-1, keepdims=True))
                        ao_p = np.clip(0.55 + 0.45 * ns[:, 2:3], 0.45, 1.0)
                    else:
                        wn = np.broadcast_to(n[None, :], (vy.shape[0], 3))
                        ao_p = 1.0

                    if spc_f is not None:
                        ss = spc_f[vy, ux]
                        rough = np.clip(1.0 - ss[:, 0:1] / 255.0, 0.08, 0.96)
                        is_met = (ss[:, 1:2] >= 229.5).astype(np.float64)
                        f0_d = np.clip(ss[:, 1:2] / 255.0, 0.03, 0.16)
                        sss_v = np.where(ss[:, 2:3] >= 65.0, (ss[:, 2:3] - 65.0) / 190.0, 0.0)
                        em_s = np.where(ss[:, 3:4] < 254.5, ss[:, 3:4] / 254.0, 0.0)
                    else:
                        rough, is_met, f0_d, sss_v, em_s = 0.50, 0.0, 0.04, 0.0, 0.0

                    ndl_k = np.clip(wn @ key_dir, 0.0, 1.0)[:, None]
                    ndl_f = np.clip(wn @ fill_dir, 0.0, 1.0)[:, None]
                    ndh_k = np.clip(wn @ half_k, 0.0, 1.0)[:, None]
                    ndv = np.clip(wn @ view_dir, 0.05, 1.0)[:, None]
                    up_b = (0.5 + 0.5 * wn[:, 1:2])

                    # GGX specular lobe + Schlick Fresnel
                    a2 = (rough * rough) ** 2
                    denom = (ndh_k * ndh_k * (a2 - 1.0) + 1.0)
                    D_ggx = a2 / (math.pi * denom * denom + 1e-5)
                    F0 = f0_d * (1.0 - is_met) + base_c * is_met
                    F = F0 + (1.0 - F0) * ((1.0 - ndv) ** 5)
                    spec_k = np.minimum(2.8, D_ggx * F * 0.42 * ndl_k / (ndv + 0.15))
                    # Metallic sky/ground reflection sheen
                    env_r = np.array([0.45, 0.52, 0.68]) * up_b + np.array([0.24, 0.20, 0.26]) * (1.0 - up_b)
                    spec_env = F * env_r * (1.1 - 0.65 * rough) * (0.45 + 1.15 * is_met)
                    # Subsurface scattering wrap for translucent wings/gems
                    wrap_k = np.clip(((wn @ key_dir)[:, None] + 0.5) / 1.5, 0.0, 1.0)
                    sss_glow = sss_v * (1.0 - is_met) * base_c * wrap_k * 0.45

                    diff_l = (0.34 + 0.62 * ndl_k + 0.20 * ndl_f + 0.14 * up_b) * (1.0 - 0.55 * is_met)
                    col = (base_c * diff_l + spec_k + spec_env + sss_glow) * ao_p
                    if em_lin is not None:
                        col = col + em_lin[vy, ux] * 1.45
                    elif np.any(em_s > 0.01):
                        col = col + base_c * em_s * 1.35
                else:
                    col = base_c * shade
                    if em_lin is not None and not unlit:
                        col = col + em_lin[vy, ux] * 1.45
                img_lin[ymin : ymax + 1, xmin : xmax + 1][vis] = col
            elif silhouette:
                sub_z[vis] = z_pix[vis]
                img_lin[ymin : ymax + 1, xmin : xmax + 1][vis] = np.array([0.05, 0.06, 0.08])
            else:
                sub_z[vis] = z_pix[vis]
                img_lin[ymin : ymax + 1, xmin : xmax + 1][vis] = base_col * shade

    if silhouette:
        out = np.clip(np.rint(img_lin * 255.0), 0, 255).astype(np.uint8)
    elif unlit:
        srgb = np.where(
            img_lin <= 0.0031308,
            12.92 * img_lin,
            1.055 * np.power(np.maximum(img_lin, 1e-12), 1.0 / 2.4) - 0.055,
        )
        out = np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)
    else:
        out = _aces_u8(img_lin)
    return Image.fromarray(out, mode="RGB")


def make_grid(tiles: list[tuple[str, Image.Image]], cols: int, title: str = "") -> Image.Image:
    """Assemble labeled tiles into a clean studio contact sheet."""
    tw, th = tiles[0][1].size
    rows = math.ceil(len(tiles) / cols)
    hdr = 40 if title else 0
    canvas = Image.new("RGB", (cols * tw, hdr + rows * th), (16, 20, 28))
    draw = ImageDraw.Draw(canvas)
    f_hdr = _load_font(20, bold=True)
    f_lbl = _load_font(14, bold=True)
    if title:
        draw.text((14, 9), title, fill=(235, 242, 252), font=f_hdr)
    for idx, (lbl, im) in enumerate(tiles):
        r, c = divmod(idx, cols)
        x0, y0 = c * tw, hdr + r * th
        canvas.paste(im, (x0, y0))
        bw = min(tw - 12, max(90, len(lbl) * 8 + 18))
        draw.rectangle([x0 + 6, y0 + 6, x0 + 6 + bw, y0 + 27], fill=(10, 14, 20))
        draw.text((x0 + 12, y0 + 8), lbl, fill=(236, 244, 252), font=f_lbl)
    return canvas


# ---------------------------------------------------------------------------
# 3. Standard Studio Sheet Generators (8-Panel, Close-Up, T1 Squint, Variants, A2)
# ---------------------------------------------------------------------------

def render_8panel_turnaround(
    scene: Scene,
    out_path: str | Path,
    *,
    atlas: dict[str, Any] | None = None,
    title: str = "Studio 8-Panel Turnaround",
    material_colors: dict[str, tuple[int, int, int]] | None = None,
    width: int = 560,
    height: int = 440,
    has_wings: bool = False,
) -> Path:
    """Render the standard 8-panel turnaround (Hero 3/4, Front, Side, Right 3/4, Rear 3/4, Top, Rear/No-Wing, Silhouette)."""
    kw = dict(atlas=atlas, material_colors=material_colors, width=width, height=height)
    is_wing = lambda bn: "wing" in bn.lower()
    p7 = (
        ("No-Wing Side (-90, 4)", render_view(scene, az_deg=-90, el_deg=4, hide_bone_filter=is_wing, **kw))
        if has_wings
        else ("Rear Profile (180, 10)", render_view(scene, az_deg=180, el_deg=10, **kw))
    )
    tiles = [
        ("Hero 3/4 (-36, 16)", render_view(scene, az_deg=-36, el_deg=16, **kw)),
        ("Dead-Front (0, 6)", render_view(scene, az_deg=0, el_deg=6, **kw)),
        ("Left Profile (-90, 4)", render_view(scene, az_deg=-90, el_deg=4, **kw)),
        ("Right 3/4 (+38, 16)", render_view(scene, az_deg=38, el_deg=16, **kw)),
        ("Rear 3/4 (-142, 18)", render_view(scene, az_deg=-142, el_deg=18, **kw)),
        ("Top-Down Dorsal (0, 86)", render_view(scene, az_deg=0, el_deg=86, **kw)),
        p7,
        ("Side Silhouette (-90, 4)", render_view(scene, az_deg=-90, el_deg=4, silhouette=True, draw_ground=False, width=width, height=height)),
    ]
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    make_grid(tiles, cols=4, title=title).save(out_p)
    return out_p


def render_head_closeup_sheet(
    program: Any,
    scene: Scene,
    out_path: str | Path,
    *,
    atlas: dict[str, Any] | None = None,
    title: str = "Studio Close-Up Gate (Head, Flush Closed Bite, Open Jaw & Foot Splay)",
    material_colors: dict[str, tuple[int, int, int]] | None = None,
    jaw_bone: str = "jaw",
    jaw_open_deg: float = 36.0,
    head_filter: Callable[[str], bool] | None = None,
    foot_filter: Callable[[str], bool] | None = None,
    extra_tiles: list[tuple[str, Image.Image]] | None = None,
    width: int = 560,
    height: int = 440,
) -> Path:
    """Render standard 6-panel close-up gate verifying eyes, flush closed bite, open jaw hinge/teeth, and top-down foot splay."""
    head_fn = head_filter or (lambda bn: any(k in bn.lower() for k in ("head", "jaw", "snout", "beak", "brow", "cheek", "crest", "eye", "horn", "ear", "tongue", "dental", "frill")))
    foot_fn = foot_filter or (lambda bn: any(k in bn.lower() for k in ("foot_l", "paw_l", "toe", "claw", "hoof_l")) and "__" not in bn.lower() and "_r" not in bn.lower())
    hide_wing = lambda bn: "wing" in bn.lower()
    kw = dict(atlas=atlas, material_colors=material_colors, width=width, height=height, draw_ground=False)

    try:
        open_scene = compile_geometry(program, pose_euler_deltas={jaw_bone: (jaw_open_deg, 0.0, 0.0)})
    except Exception:
        open_scene = scene

    tiles = [
        ("Head Dead-Front (0, 4)", render_view(scene, az_deg=0, el_deg=4, focus_bone_filter=head_fn, hide_bone_filter=hide_wing, **kw)),
        ("Head Hero 3/4 (-36, 10)", render_view(scene, az_deg=-36, el_deg=10, focus_bone_filter=head_fn, hide_bone_filter=hide_wing, **kw)),
        ("Head Side Closed (-90, 2)", render_view(scene, az_deg=-90, el_deg=2, focus_bone_filter=head_fn, hide_bone_filter=hide_wing, **kw)),
        (f"Open Jaw +{int(jaw_open_deg)} deg (-48, -4)", render_view(open_scene, az_deg=-48, el_deg=-4, focus_bone_filter=head_fn, hide_bone_filter=hide_wing, **kw)),
        ("Top-Down Foot Splay (0, 86)", render_view(scene, az_deg=0, el_deg=86, focus_bone_filter=foot_fn, hide_bone_filter=lambda bn: not foot_fn(bn), **kw)),
    ]
    if extra_tiles:
        tiles.extend(extra_tiles)
    else:
        tail_fn = lambda bn: "tail" in bn.lower()
        tiles.append(("Tail & Rear Detail (-130, 20)", render_view(scene, az_deg=-130, el_deg=20, focus_bone_filter=tail_fn, hide_bone_filter=hide_wing, **kw)))

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    make_grid(tiles, cols=3, title=title).save(out_p)
    return out_p


def render_t1_unlit_squint(
    scene: Scene,
    out_path: str | Path,
    atlas: dict[str, Any],
    *,
    title: str = "T1 Unlit Macro Values & 25% Squint Gate",
    width: int = 560,
    height: int = 440,
) -> Path:
    """Render unlit macro value panels plus a true 25% downsampled nearest-upsampled squint panel."""
    kw = dict(atlas=atlas, unlit=True, width=width, height=height)
    hero = render_view(scene, az_deg=-36, el_deg=16, **kw)
    squint = hero.resize((max(16, width // 4), max(16, height // 4)), Image.Resampling.BILINEAR).resize((width, height), Image.Resampling.NEAREST)
    tiles = [
        ("T1 Unlit Hero 3/4", hero),
        ("T1 Unlit Dead-Front", render_view(scene, az_deg=0, el_deg=6, **kw)),
        ("T1 Unlit Pure Side", render_view(scene, az_deg=-90, el_deg=4, **kw)),
        ("25% Squint Readability Gate", squint),
    ]
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    make_grid(tiles, cols=2, title=title).save(out_p)
    return out_p


def render_variants_showcase(
    variants: list[tuple[str, Scene, dict[str, Any]]],
    out_path: str | Path,
    *,
    title: str = "Multi-Variant T2 Showcase",
    views: list[tuple[str, float, float]] | None = None,
    width: int = 560,
    height: int = 440,
) -> Path:
    """Render side-by-side comparison rows across multiple species/variants."""
    vlist = views or [("Hero 3/4", -36.0, 16.0), ("Side Profile", -90.0, 4.0)]
    tiles: list[tuple[str, Image.Image]] = []
    for v_name, az, el in vlist:
        for label, scn, atl in variants:
            im = render_view(scn, atlas=atl, az_deg=az, el_deg=el, width=width, height=height)
            tiles.append((f"{label} — {v_name}", im))
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    make_grid(tiles, cols=len(variants), title=title).save(out_p)
    return out_p


def render_a2_filmstrip(
    program: Any,
    clip_specs: list[tuple[str, float, float]],
    out_path: str | Path,
    atlas: dict[str, Any],
    *,
    title: str = "A2 Biomechanical Motion Filmstrip (Full T2 Textures)",
    phases: Iterable[int] = (0, 5, 9, 13),
    num_frames: int = 16,
    width: int = 460,
    height: int = 345,
) -> Path:
    """Render full-texture multi-clip A2 motion filmstrips (len(phases) columns per clip row)."""
    ph_list = list(phases)
    tiles: list[tuple[str, Image.Image]] = []
    for cname, az, el in clip_specs:
        samps = evaluate_behavior_clip(program, cname, num_frames=num_frames)
        for fi in ph_list:
            smp = samps[min(fi, len(samps) - 1)]
            pct = int(round(100.0 * fi / max(1, num_frames - 1)))
            im = render_view(smp.scene, atlas=atlas, az_deg=az, el_deg=el, width=width, height=height, fill_frac=0.80)
            tiles.append((f"{cname} ({pct}%)", im))
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    make_grid(tiles, cols=len(ph_list), title=title).save(out_p)
    return out_p


# ---------------------------------------------------------------------------
# 4. v39.3 Pixel-Relief Texture Telemetry (Anti-Flat-Gradient + Anti-Noise!)
# ---------------------------------------------------------------------------

def compute_atlas_metrics(
    atlas_rgba: np.ndarray,
    atlas_faces: dict[str, dict[str, list[int]]],
    emissive_rgba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure within-face lum_step, jump>25% (must be 4.5%-14.0% under v39.3!), and glow_area_pct."""
    rgb = atlas_rgba[:, :, :3].astype(np.float64)
    alpha = atlas_rgba[:, :, 3] >= 128
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    diffs: list[np.ndarray] = []
    seen_rects: set[tuple[int, int, int, int]] = set()
    for bmap in atlas_faces.values():
        for rect in bmap.values():
            x0, y0, x1, y1 = int(min(rect[0], rect[2])), int(min(rect[1], rect[3])), int(max(rect[0], rect[2])), int(max(rect[1], rect[3]))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            key = (x0, y0, x1, y1)
            if key in seen_rects:
                continue
            seen_rects.add(key)
            sub_l = lum[y0:y1, x0:x1]
            sub_a = alpha[y0:y1, x0:x1]
            dh = np.abs(sub_l[:, 1:] - sub_l[:, :-1])[sub_a[:, 1:] & sub_a[:, :-1]]
            dv = np.abs(sub_l[1:, :] - sub_l[:-1, :])[sub_a[1:, :] & sub_a[:-1, :]]
            if dh.size:
                diffs.append(dh)
            if dv.size:
                diffs.append(dv)

    if diffs:
        all_d = np.concatenate(diffs)
        lum_step = float(np.mean(all_d))
        jump25_pct = float(100.0 * np.mean(all_d > 25.0))
    else:
        lum_step, jump25_pct = 0.0, 0.0

    opaque_px = max(1, int(np.sum(alpha)))
    glow_pct = 0.0
    if emissive_rgba is not None:
        em_max = np.max(emissive_rgba[:, :, :3], axis=2)
        glow_px = int(np.sum((em_max > 18) & alpha))
        glow_pct = float(100.0 * glow_px / opaque_px)

    band_ok = (4.5 <= jump25_pct <= 14.0) and (5.0 <= lum_step <= 8.5) and (glow_pct <= 4.0)
    verdict = "PASS_V39_3"
    if jump25_pct < 4.5:
        verdict = "FAIL_FLAT_PLASTIC_GRADIENT (jump>25 < 4.5%! Carve crisp 2px pixel clusters & 1px overlap crevices!)"
    elif jump25_pct > 18.0:
        verdict = "FAIL_NOISY_STATIC (jump>25 > 18%! Organize into deliberate 2px clusters!)"
    elif lum_step < 5.0:
        verdict = "WARN_LOW_CONTRAST (lum_step < 5.0; deepen bevel/shadow value separation!)"
    elif glow_pct > 4.0:
        verdict = "FAIL_EXCESS_GLOW (glow_pct > 4.0%!)"

    return {
        "lum_step": round(lum_step, 2),
        "jump25_pct": round(jump25_pct, 2),
        "glow_pct": round(glow_pct, 3),
        "v39_3_band_ok": band_ok,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 5. Automated Gate Runner & Swarm Overview Screenshot Helper
# ---------------------------------------------------------------------------

def run_full_verification(
    *,
    dsl_path: str | Path | None = None,
    bb_path: str | Path,
    geo_path: str | Path | None = None,
    tier: str = "standard",
    stance: str = "auto",
) -> dict[str, Any]:
    """Run construct_check.py across all provided formats plus verify_bbmodel_ground.py."""
    cc_script = str(SKILL_SCRIPTS_DIR / "construct_check.py")
    vg_script = str(SKILL_SCRIPTS_DIR / "verify_bbmodel_ground.py")
    report: dict[str, Any] = {"all_zero_warn": True, "ground_pass": False, "formats": {}}

    checks = []
    if dsl_path:
        checks.append(("blockdsl", [str(SYNCCRAFT_V20_ROOT), str(dsl_path)]))
    checks.append(("bbmodel", [str(bb_path)]))
    if geo_path:
        checks.append(("geo_json", [str(geo_path)]))

    for label, extra in checks:
        res = subprocess.run([sys.executable, cc_script, "--tier", tier] + extra, capture_output=True, text=True)
        out = res.stdout + res.stderr
        warns = [ln.strip() for ln in out.splitlines() if "[WARN" in ln]
        fails = [ln.strip() for ln in out.splitlines() if "[FAIL" in ln]
        if res.returncode != 0 or warns or fails:
            report["all_zero_warn"] = False
        report["formats"][label] = {"rc": res.returncode, "fails": fails, "warns": warns}

    vg_cmd = [sys.executable, vg_script, str(bb_path)]
    if stance and stance != "auto":
        vg_cmd.extend(["--stance", stance])
    vg = subprocess.run(vg_cmd, capture_output=True, text=True)
    vg_out = vg.stdout + vg.stderr
    report["ground_pass"] = (vg.returncode == 0) and ("PASS=True" in vg_out)
    report["ground_summary"] = "\n".join(vg_out.splitlines()[:8])
    return report


def stage_swarm_screenshot(
    src_png: str | Path,
    worker_brain_dir: str | Path,
    caption: str,
    dest_name: str | None = None,
) -> str:
    """Copy a milestone render into the worker's own brain directory and return the Swarm UI Screenshot line."""
    src = Path(src_png)
    bdir = Path(worker_brain_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    dst = bdir / (dest_name or src.name)
    shutil.copy2(src, dst)
    words = caption.strip().split()[:10]
    return f"Screenshot: {dst.resolve()} | {' '.join(words)}"


# ---------------------------------------------------------------------------
# 6. Dynamic Blockbench 5 (.bbmodel) World-Space Wing/Joint Continuity Audit (v39.4)
# ---------------------------------------------------------------------------

def _bb5_euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _bb5_eval_channel(kfs: list[dict[str, Any]], t: float, default: float = 0.0) -> np.ndarray:
    if not kfs:
        return np.full(3, default, dtype=np.float64)
    pts = sorted(kfs, key=lambda k: float(k["time"]))
    if t <= float(pts[0]["time"]):
        dp = pts[0]["data_points"][0]
        return np.array([float(dp["x"]), float(dp["y"]), float(dp["z"])], dtype=np.float64)
    if t >= float(pts[-1]["time"]):
        dp = pts[-1]["data_points"][0]
        return np.array([float(dp["x"]), float(dp["y"]), float(dp["z"])], dtype=np.float64)
    for i in range(len(pts) - 1):
        t0, t1 = float(pts[i]["time"]), float(pts[i + 1]["time"])
        if t0 <= t <= t1:
            alpha = (t - t0) / max(1e-9, t1 - t0)
            d0, d1 = pts[i]["data_points"][0], pts[i + 1]["data_points"][0]
            v0 = np.array([float(d0["x"]), float(d0["y"]), float(d0["z"])], dtype=np.float64)
            v1 = np.array([float(d1["x"]), float(d1["y"]), float(d1["z"])], dtype=np.float64)
            return v0 * (1.0 - alpha) + v1 * alpha
    return np.full(3, default, dtype=np.float64)


def _bb5_element_obb(
    el: dict[str, Any],
    bone_origin: np.ndarray,
    parent_R: np.ndarray,
    parent_t: np.ndarray,
    parent_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f0 = np.array(el["from"], dtype=np.float64)
    f1 = np.array(el["to"], dtype=np.float64)
    loc_ctr = 0.5 * (f0 + f1)
    loc_half = 0.5 * np.abs(f1 - f0)
    rot = el.get("rotation")
    if rot and any(abs(float(r)) > 1e-6 for r in rot):
        org = np.array(el.get("origin", loc_ctr.tolist()), dtype=np.float64)
        R_el = _bb5_euler_xyz_deg(float(rot[0]), float(rot[1]), float(rot[2]))
        loc_ctr = org + R_el @ (loc_ctr - org)
    else:
        R_el = np.eye(3, dtype=np.float64)
    corners_raw = (
        np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float64,
        )
        * loc_half[None, :]
    )
    corners_el = corners_raw @ R_el.T + loc_ctr[None, :]
    corners_scaled = (corners_el - bone_origin[None, :]) * parent_s[None, :] + bone_origin[None, :]
    wc = corners_scaled @ parent_R.T + parent_t[None, :]
    w_ctr = np.mean(wc, axis=0)
    v_x, v_y, v_z = wc[4] - wc[0], wc[2] - wc[0], wc[1] - wc[0]
    hx = max(1e-6, 0.5 * float(np.linalg.norm(v_x)))
    hy = max(1e-6, 0.5 * float(np.linalg.norm(v_y)))
    hz = max(1e-6, 0.5 * float(np.linalg.norm(v_z)))
    ax = np.column_stack([v_x / (2.0 * hx), v_y / (2.0 * hy), v_z / (2.0 * hz)])
    return w_ctr, ax, np.array([hx, hy, hz], dtype=np.float64)


def _bb5_obb_signed_overlap(
    obb_a: tuple[np.ndarray, np.ndarray, np.ndarray],
    obb_b: tuple[np.ndarray, np.ndarray, np.ndarray],
    cutoff: float = -float("inf"),
) -> float:
    """Exact 15-axis SAT signed overlap (> 0 = physical intersection, < 0 = negative air gap).

    When `cutoff` is provided, short-circuits cleanly if the exact overlap cannot exceed `cutoff`.
    """
    ca, Ra, ha = obb_a
    cb, Rb, hb = obb_b
    d = cb - ca
    # Exact bounding-sphere upper bound: overlap <= (|ha| + |hb|) - |d|
    dist_c = float(np.linalg.norm(d))
    if (float(np.linalg.norm(ha)) + float(np.linalg.norm(hb)) - dist_c) <= cutoff:
        return cutoff
    min_ov = float("inf")
    for k in range(3):
        for u in (Ra[:, k], Rb[:, k]):
            ra = float(abs(u @ Ra[:, 0]) * ha[0] + abs(u @ Ra[:, 1]) * ha[1] + abs(u @ Ra[:, 2]) * ha[2])
            rb = float(abs(u @ Rb[:, 0]) * hb[0] + abs(u @ Rb[:, 1]) * hb[1] + abs(u @ Rb[:, 2]) * hb[2])
            ov = (ra + rb) - abs(float(d @ u))
            if ov < min_ov:
                min_ov = ov
                if min_ov <= cutoff:
                    return float(min_ov)
    for i in range(3):
        ri = Ra[:, i]
        for j in range(3):
            cr = np.cross(ri, Rb[:, j])
            n = float(np.linalg.norm(cr))
            if n > 1e-6:
                u = cr / n
                ra = float(abs(u @ Ra[:, 0]) * ha[0] + abs(u @ Ra[:, 1]) * ha[1] + abs(u @ Ra[:, 2]) * ha[2])
                rb = float(abs(u @ Rb[:, 0]) * hb[0] + abs(u @ Rb[:, 1]) * hb[1] + abs(u @ Rb[:, 2]) * hb[2])
                ov = (ra + rb) - abs(float(d @ u))
                if ov < min_ov:
                    min_ov = ov
                    if min_ov <= cutoff:
                        return float(min_ov)
    return float(min_ov)


def evaluate_bb5_world_obbs(
    bb: dict[str, Any],
    anim_pos: dict[str, np.ndarray] | None = None,
    anim_rot: dict[str, np.ndarray] | None = None,
    anim_scl: dict[str, np.ndarray] | None = None,
    only_elems: set[str] | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, list[str]], list[tuple[str, str]]]:
    """Evaluate element OBBs in Blockbench 5 FK world space (with Sim(3) scale, rotation, position).

    Returns `(obbs_by_elem_name, bone_to_elem_names, parent_child_bone_pairs)`.
    """
    el_by_uuid = {e["uuid"]: e for e in bb.get("elements", [])}
    grp_by_uuid = {g["uuid"]: g for g in bb.get("groups", [])}
    apos = anim_pos or {}
    arot = anim_rot or {}
    ascl = anim_scl or {}
    obbs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    bone_elems: dict[str, list[str]] = {}
    edges: list[tuple[str, str]] = []

    def walk(
        nodes: list[Any],
        parent_name: str | None,
        parent_R: np.ndarray,
        parent_t: np.ndarray,
        parent_s: np.ndarray,
        bone_origin: np.ndarray,
    ) -> None:
        for nd in nodes:
            if isinstance(nd, str):
                el = el_by_uuid.get(nd)
                if el is not None:
                    en = str(el.get("name", nd))
                    if only_elems is None or en in only_elems:
                        obbs[en] = _bb5_element_obb(el, bone_origin, parent_R, parent_t, parent_s)
                    if parent_name is not None:
                        bone_elems.setdefault(parent_name, []).append(en)
            elif isinstance(nd, dict):
                guid = str(nd.get("uuid", ""))
                g = grp_by_uuid.get(guid, nd)
                gname = str(g.get("name", guid))
                if parent_name is not None:
                    edges.append((parent_name, gname))
                org = np.array(g.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
                rest_r = g.get("rotation", [0.0, 0.0, 0.0])
                dpos = apos.get(guid, np.zeros(3, dtype=np.float64))
                drot = arot.get(guid, np.zeros(3, dtype=np.float64))
                dscl = ascl.get(guid, np.ones(3, dtype=np.float64))
                R_bone = _bb5_euler_xyz_deg(
                    float(rest_r[0]) + float(drot[0]),
                    float(rest_r[1]) + float(drot[1]),
                    float(rest_r[2]) + float(drot[2]),
                )
                cur_s = parent_s * dscl
                new_R = parent_R @ R_bone
                scaled_rel = bone_origin + (org + dpos - bone_origin) * parent_s
                new_t = parent_R @ (scaled_rel - R_bone @ org) + parent_t
                walk(nd.get("children", []), gname, new_R, new_t, cur_s, org)

    walk(bb.get("outliner", []), None, np.eye(3), np.zeros(3), np.ones(3), np.zeros(3))
    return obbs, bone_elems, edges


def verify_bb5_wing_continuity(
    bb_path_or_dict: str | Path | dict[str, Any],
    *,
    custom_pairs: list[tuple[str, list[str], list[str]]] | None = None,
    bone_filter: Callable[[str, str], bool] | None = None,
    samples_per_clip: int = 25,
    min_overlap_threshold: float = -0.02,
) -> dict[str, Any]:
    """Verify dynamic Blockbench 5 world-space OBB continuity across Rest + EVERY animated clip.

    Catches detached front wing claws, floating `wing_l_f4` spars, or disconnected elbow/wrist/patagium
    seams that static single-pose checks can miss during flight, folding, or thrust inversions.

    Args:
        bb_path_or_dict: Path to `.bbmodel` or parsed `.bbmodel` dict.
        custom_pairs: Optional explicit `[(label, elems_a, elems_b), ...]` checks. When omitted,
            auto-discovers all `(parent_bone, child_bone)` wing/patagium/claw pairs that carry elements.
        bone_filter: Optional `(parent_bone, child_bone) -> bool` predicate (defaults to wing/patagium/claw edges).
        samples_per_clip: Number of evenly spaced timestamps sampled per animation clip (default 25).
        min_overlap_threshold: Minimum allowed signed overlap in Blockbench units (default `-0.02u`).
    """
    bb = (
        json.loads(Path(bb_path_or_dict).read_text(encoding="utf-8"))
        if not isinstance(bb_path_or_dict, dict)
        else bb_path_or_dict
    )
    rest_obbs, bone_elems, edges = evaluate_bb5_world_obbs(bb)

    if custom_pairs is not None:
        check_pairs = list(custom_pairs)
    else:
        if bone_filter is None:
            _wing_tokens = ("wing", "patag", "alula", "pinion", "thruster", "nacelle", "elytr")
            def bone_filter(p: str, c: str) -> bool:
                pl, cl = p.lower(), c.lower()
                return any(t in cl for t in _wing_tokens) or any(t in pl for t in _wing_tokens)
        check_pairs = []
        for pb, cb in edges:
            if bone_filter(pb, cb) and bone_elems.get(pb) and bone_elems.get(cb):
                check_pairs.append((f"{pb}->{cb}", list(bone_elems[pb]), list(bone_elems[cb])))

    needed_elems: set[str] = set()
    for _, ga, gb in check_pairs:
        needed_elems.update(ga)
        needed_elems.update(gb)

    min_by_label: dict[str, tuple[float, str, float]] = {}
    checks_total = 0

    def _eval_pose(obbs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], clip_name: str, t_s: float) -> None:
        nonlocal checks_total
        for label, ga, gb in check_pairs:
            valid_a = [obbs[a] for a in ga if a in obbs]
            valid_b = [obbs[b] for b in gb if b in obbs]
            if not valid_a or not valid_b:
                continue
            best = -float("inf")
            for oa in valid_a:
                for ob in valid_b:
                    ov = _bb5_obb_signed_overlap(oa, ob, cutoff=best)
                    if ov > best:
                        best = ov
            checks_total += 1
            prev = min_by_label.get(label)
            if prev is None or best < prev[0]:
                min_by_label[label] = (best, clip_name, t_s)

    _eval_pose(rest_obbs, "REST", 0.0)

    for anim in bb.get("animations", []):
        cname = str(anim.get("name", "unnamed"))
        length = max(0.01, float(anim.get("length", 1.0)))
        animators = anim.get("animators", {}) or {}
        # Pre-split channels once per animation clip
        split_ch: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
        for guid, adata in animators.items():
            if not isinstance(adata, dict):
                continue
            kfs = adata.get("keyframes", []) or []
            pk = [k for k in kfs if k.get("channel") == "position"]
            rk = [k for k in kfs if k.get("channel") == "rotation"]
            sk = [k for k in kfs if k.get("channel") == "scale"]
            if pk or rk or sk:
                split_ch[guid] = (pk, rk, sk)
        n_samp = max(2, int(samples_per_clip))
        for si in range(n_samp):
            t_s = length * (si / max(1, n_samp - 1))
            apos: dict[str, np.ndarray] = {}
            arot: dict[str, np.ndarray] = {}
            ascl: dict[str, np.ndarray] = {}
            for guid, (pk, rk, sk) in split_ch.items():
                if pk:
                    apos[guid] = _bb5_eval_channel(pk, t_s, 0.0)
                if rk:
                    arot[guid] = _bb5_eval_channel(rk, t_s, 0.0)
                if sk:
                    ascl[guid] = _bb5_eval_channel(sk, t_s, 1.0)
            obbs, _, _ = evaluate_bb5_world_obbs(bb, apos, arot, ascl, only_elems=needed_elems)
            _eval_pose(obbs, cname, t_s)

    failures = {
        lbl: {"min_overlap_u": round(ov, 4), "clip": cl, "t_s": round(ts, 3)}
        for lbl, (ov, cl, ts) in min_by_label.items()
        if ov < min_overlap_threshold
    }
    worst_label, (worst_ov, worst_clip, worst_t) = (
        min(min_by_label.items(), key=lambda kv: kv[1][0])
        if min_by_label
        else ("NONE", (0.0, "REST", 0.0))
    )
    return {
        "passed": len(failures) == 0,
        "pairs_checked": len(min_by_label),
        "checks_total": checks_total,
        "min_overlap_u": round(float(worst_ov), 4),
        "worst_joint": worst_label,
        "worst_clip": worst_clip,
        "worst_t_s": round(float(worst_t), 3),
        "failures": failures,
    }
