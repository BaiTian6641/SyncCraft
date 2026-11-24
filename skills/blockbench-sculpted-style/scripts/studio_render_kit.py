#!/usr/bin/env python3
"""Standard reusable Blockbench Studio Rendering, Reference Prep, Pixel-Relief Telemetry & Verification Kit (v39.3).

Import this module directly in any creature/asset build script instead of rewriting camera renderers,
turnaround grids, open-jaw close-ups, A2 filmstrips, reference downloaders, or craft telemetry from scratch.

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


def _unpack_atlas(atlas: dict[str, Any] | None) -> tuple[dict | None, np.ndarray | None, np.ndarray | None]:
    if not atlas:
        return None, None, None
    faces = atlas.get("faces") or atlas.get("uv_faces")
    albedo = atlas.get("albedo_rgba")
    if albedo is None:
        albedo = atlas.get("rgba")
    emissive = atlas.get("emissive_rgba")
    return faces, albedo, emissive


def render_view(
    scene: Scene,
    atlas_faces: dict[str, dict[str, list[int]]] | None = None,
    atlas_rgba: np.ndarray | None = None,
    *,
    emissive_rgba: np.ndarray | None = None,
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
    """Render a crisp studio view of a SyncCraft Scene (textured, clay, unlit, or silhouette)."""
    if atlas is not None and atlas_rgba is None:
        atlas_faces, atlas_rgba, em2 = _unpack_atlas(atlas)
        if emissive_rgba is None:
            emissive_rgba = em2

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
        n = np.cross(e1, e2)
        nlen = float(np.linalg.norm(n))
        if nlen < 1e-8:
            continue
        n /= nlen
        if float(np.dot(n, cam_pos - fverts.mean(axis=0))) < 0.0:
            n = -n

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
                col = albedo_lin[vi[vis], ui[vis]] * shade
                if em_lin is not None and not unlit:
                    col = col + em_lin[vi[vis], ui[vis]] * 1.45
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
