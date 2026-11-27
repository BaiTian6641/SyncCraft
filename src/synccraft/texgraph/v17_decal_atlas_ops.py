"""SyncCraft TexGraph v17 (`SyncCraft Studio v53.0`) Pro Engines:

Grounded in our 3-Track Gemini Deep Research Audit (`STUDIO_V17_DEEP_RESEARCH_ARCHITECTURE.md`):
1. **Mip-Safe Pixel-Art Voronoi / Jump-Flooding Dilation (`mip_safe_dilate_op`)**:
   Exact pure-NumPy vectorized Jump-Flooding Algorithm (`JFA+1`) providing blur-free nearest-seed
   UV island & alpha-cutout edge bleeding across arbitrary mip radii (`bleed_px=4..16` or infinite),
   straight-alpha RGB bleed (zero dark cutout halos!), tangent-normal renormalization with flat
   `(128,128,255)` background, LabPBR sentinel preservation, and interactive mip-level preview (`0..4`).
2. **3D Camera-Projected Decal / Stencil Stamps (`decal_stamp_op`)**:
   Substance-Painter-style planar/camera projection directly onto 3D `.bbmodel` UV space via baked
   `world_pos` & `world_normal` with zero seams across UV islands, backface culling + smoothstep
   normal-angle cutoff, exact vectorized Kay-Kajiya Ray-vs-OBB front-most `t_first` occlusion culling
   (zero bleed-through onto back limbs!), 8 built-in procedural PBR stencils + custom PNG support,
   and multi-stamp stacking (`stamps_json`).
3. **Multi-Material Asset Atlas Baker (`bake_multi_asset_atlas`)**:
   Combines multiple `.bbmodel` models/textures into a single MaxRects-BSSF packed atlas (`pot`
   Power-of-Two for UE5/Unity/glTF or `tight` multiple-of-16 for Blockbench/Minecraft) with mip-safe
   Voronoi padding and exact `.bbmodel` face UV remapping.
4. **Canonical One-Click 6-Engine Export Presets (`ENGINE_EXPORT_PRESETS`, `export_engine_preset`)**:
   Spec-accurate channel packing & normal Y-axis convention (DirectX Y-down for UE5 & LabPBR 1.3;
   OpenGL Y-up for Unity, MaterialX & glTF; Unity HDRP neutral `B=255` Detail Mask + `_HeightMap`;
   Bedrock RTX/Vibrant Visuals `1.21.30` `_mers`; MaterialX `1.39` `open_pbr_surface` / `1.38` `standard_surface`).
"""
from __future__ import annotations

import base64
import copy
import io
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from .pbr import BAKE_LIGHTING_PRESETS, bake_pbr_to_albedo, height_to_normal_map, pack_labpbr_1_3
from .pipeline import material_make
from .pixel_art_ops import blend_layers, hex_to_srgb, rgba_float_to_u8


# ===========================================================================
# 1. Pure-NumPy Vectorized Jump-Flooding (`JFA+1`) Mip-Safe UV Dilation
# ===========================================================================


def _voronoi_nearest_seed_jfa(valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact-grade nearest-seed Voronoi indices & Euclidean distances in pure vectorized NumPy (JFA+1).

    Returns ``(nearest_y, nearest_x, dist_px)`` of shape ``(H, W)`` without needing SciPy.
    """
    mask = np.asarray(valid_mask, dtype=bool)
    H, W = mask.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.int32)
    if not np.any(mask):
        return yy, xx, np.full((H, W), 1e9, dtype=np.float32)
    if np.all(mask):
        return yy, xx, np.zeros((H, W), dtype=np.float32)

    seed_y = np.where(mask, yy, -1).astype(np.int32)
    seed_x = np.where(mask, xx, -1).astype(np.int32)
    best_d2 = np.where(mask, 0.0, 1e18).astype(np.float32)

    max_dim = max(H, W)
    step = 1
    while step * 2 < max_dim:
        step *= 2
    steps: list[int] = []
    s = step
    while s >= 1:
        steps.append(s)
        s //= 2
    steps.append(1)  # JFA+1 refinement pass eliminates rare 1-px Voronoi artifacts!

    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    yy_f = yy.astype(np.float32)
    xx_f = xx.astype(np.float32)

    for st in steps:
        for dy, dx in offsets:
            oy = dy * st
            ox = dx * st
            # Slice overlap without wrapping across opposite canvas borders
            y_src0 = max(0, -oy)
            y_src1 = min(H, H - oy)
            x_src0 = max(0, -ox)
            x_src1 = min(W, W - ox)
            if y_src0 >= y_src1 or x_src0 >= x_src1:
                continue
            y_dst0 = y_src0 + oy
            y_dst1 = y_src1 + oy
            x_dst0 = x_src0 + ox
            x_dst1 = x_src1 + ox

            cand_y = seed_y[y_src0:y_src1, x_src0:x_src1]
            cand_x = seed_x[y_src0:y_src1, x_src0:x_src1]
            has_seed = cand_y >= 0
            if not np.any(has_seed):
                continue

            dy_c = yy_f[y_dst0:y_dst1, x_dst0:x_dst1] - cand_y.astype(np.float32)
            dx_c = xx_f[y_dst0:y_dst1, x_dst0:x_dst1] - cand_x.astype(np.float32)
            d2 = dy_c * dy_c + dx_c * dx_c

            cur_d2 = best_d2[y_dst0:y_dst1, x_dst0:x_dst1]
            better = has_seed & (d2 < cur_d2)
            if np.any(better):
                dst_sy = seed_y[y_dst0:y_dst1, x_dst0:x_dst1]
                dst_sx = seed_x[y_dst0:y_dst1, x_dst0:x_dst1]
                dst_sy[better] = cand_y[better]
                dst_sx[better] = cand_x[better]
                cur_d2[better] = d2[better]

    seed_y = np.clip(seed_y, 0, H - 1)
    seed_x = np.clip(seed_x, 0, W - 1)
    return seed_y, seed_x, np.sqrt(np.maximum(0.0, best_d2)).astype(np.float32)


def mip_safe_dilate_rgba(
    rgba: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    bleed_px: int = 8,
    alpha_cutoff: float = 0.02,
    preserve_alpha: bool = True,
) -> np.ndarray:
    """Straight-alpha mip-safe Voronoi RGB dilation eliminating dark cutout halos & mipmap UV seam bleeding."""
    arr = np.asarray(rgba)
    is_u8 = arr.dtype == np.uint8
    f = arr.astype(np.float32) / 255.0 if is_u8 else arr.astype(np.float32).copy()
    if f.ndim == 2:
        f = np.stack((f, f, f, np.ones_like(f)), axis=-1)
    elif f.shape[2] == 3:
        f = np.concatenate((f, np.ones(f.shape[:2] + (1,), dtype=np.float32)), axis=-1)

    seed_mask = (f[..., 3] > float(alpha_cutoff)) if valid_mask is None else (np.asarray(valid_mask, dtype=bool) & (f[..., 3] > float(alpha_cutoff)))
    if not np.any(seed_mask) or np.all(seed_mask):
        return arr.copy()

    ny, nx, dist = _voronoi_nearest_seed_jfa(seed_mask)
    fill_mask = (~seed_mask) if int(bleed_px) <= 0 else ((~seed_mask) & (dist <= float(bleed_px) + 1e-4))
    out = f.copy()
    out[fill_mask, :3] = f[ny[fill_mask], nx[fill_mask], :3]
    if not preserve_alpha:
        out[fill_mask, 3] = f[ny[fill_mask], nx[fill_mask], 3]
    if is_u8:
        return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def mip_safe_dilate_normal(
    normal_rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bleed_px: int = 8,
) -> np.ndarray:
    """Nearest-seed Voronoi dilation for tangent-space normal maps with renormalization & flat (128,128,255) void."""
    arr = np.asarray(normal_rgb)
    is_u8 = arr.dtype == np.uint8
    f = arr.astype(np.float32) / 255.0 if is_u8 else arr.astype(np.float32).copy()
    has_alpha = f.ndim == 3 and f.shape[2] == 4
    rgb = f[..., :3].copy()
    seed = np.asarray(valid_mask, dtype=bool)
    if not np.any(seed):
        rgb[..., 0] = 0.5
        rgb[..., 1] = 0.5
        rgb[..., 2] = 1.0
    else:
        ny, nx, dist = _voronoi_nearest_seed_jfa(seed)
        fill = (~seed) if int(bleed_px) <= 0 else ((~seed) & (dist <= float(bleed_px) + 1e-4))
        void = (~seed) & (~fill)
        rgb[fill] = rgb[ny[fill], nx[fill]]
        rgb[void] = np.array([0.5, 0.5, 1.0], dtype=np.float32)
        # Renormalize tangent vectors to unit length
        nv = rgb * 2.0 - 1.0
        nlen = np.maximum(1e-5, np.linalg.norm(nv, axis=-1, keepdims=True))
        rgb = np.clip((nv / nlen) * 0.5 + 0.5, 0.0, 1.0)

    out = np.concatenate((rgb, f[..., 3:4]), axis=-1) if has_alpha else rgb
    if is_u8:
        return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    return out.astype(np.float32)


def mip_safe_dilate_scalar(
    field: np.ndarray,
    valid_mask: np.ndarray,
    *,
    bleed_px: int = 8,
    void_value: float | None = None,
) -> np.ndarray:
    """Linear-space nearest-seed Voronoi dilation for Roughness / Metallic / Height / AO / LabPBR channels."""
    arr = np.asarray(field)
    is_u8 = arr.dtype == np.uint8
    out = arr.copy()
    seed = np.asarray(valid_mask, dtype=bool)
    if not np.any(seed) or np.all(seed):
        return out
    ny, nx, dist = _voronoi_nearest_seed_jfa(seed)
    fill = (~seed) if int(bleed_px) <= 0 else ((~seed) & (dist <= float(bleed_px) + 1e-4))
    out[fill] = arr[ny[fill], nx[fill]]
    if void_value is not None:
        void = (~seed) & (~fill)
        out[void] = int(round(void_value * 255.0)) if is_u8 else float(void_value)
    return out


def _simulate_mip_preview(rgba: np.ndarray, level: int) -> np.ndarray:
    """Box-downsample by 2^level and nearest-upsample back so artists can inspect mipmap seam safety!"""
    lvl = max(0, min(4, int(level)))
    if lvl == 0:
        return rgba
    H, W = rgba.shape[:2]
    scale = 1 << lvl
    mh, mw = max(1, H // scale), max(1, W // scale)
    u8 = rgba_float_to_u8(rgba)
    im = Image.fromarray(u8, mode="RGBA")
    small = im.resize((mw, mh), resample=Image.Resampling.BOX)
    back = small.resize((W, H), resample=Image.Resampling.NEAREST)
    return np.asarray(back, dtype=np.float32) / 255.0


def mip_safe_dilate_op(
    src: dict[str, np.ndarray] | np.ndarray,
    *,
    mesh_context: Any = None,
    bleed_px: int = 8,
    alpha_cutoff: float = 0.02,
    preserve_alpha: bool = True,
    mip_preview_level: int = 0,
) -> dict[str, np.ndarray] | np.ndarray:
    """TexGraph v17 node (`mip_safe_dilate`): exact JFA+1 mip-safe Voronoi edge dilation for RGBA or MATERIAL buses."""
    b_px = max(0, min(64, int(bleed_px)))
    imask = mesh_context.island_mask if (mesh_context is not None and hasattr(mesh_context, "island_mask")) else None

    if isinstance(src, dict) and "albedo" in src:
        alb = np.asarray(src["albedo"], dtype=np.float32)
        seed = (imask & (alb[..., 3] > float(alpha_cutoff))) if imask is not None else (alb[..., 3] > float(alpha_cutoff))
        alb_d = mip_safe_dilate_rgba(alb, valid_mask=imask, bleed_px=b_px, alpha_cutoff=alpha_cutoff, preserve_alpha=preserve_alpha)
        if int(mip_preview_level) > 0:
            alb_d = _simulate_mip_preview(alb_d, int(mip_preview_level))
        out = dict(src)
        out["albedo"] = alb_d
        for sk in ("height", "roughness", "metallic", "ao", "shadow", "thickness", "sss", "porosity"):
            if sk in src and src[sk] is not None:
                out[sk] = mip_safe_dilate_scalar(np.asarray(src[sk], dtype=np.float32), seed, bleed_px=b_px)
        if "normal" in src and src["normal"] is not None:
            out["normal"] = mip_safe_dilate_normal(np.asarray(src["normal"], dtype=np.float32), seed, bleed_px=b_px)
        if "emissive" in src and src["emissive"] is not None:
            out["emissive"] = mip_safe_dilate_rgba(
                np.asarray(src["emissive"], dtype=np.float32),
                valid_mask=seed,
                bleed_px=b_px,
                alpha_cutoff=0.01,
                preserve_alpha=True,
            )
        return out

    arr = np.asarray(src, dtype=np.float32)
    if arr.ndim == 2:
        seed = imask if imask is not None else (arr > float(alpha_cutoff))
        return mip_safe_dilate_scalar(arr, seed, bleed_px=b_px)
    dil = mip_safe_dilate_rgba(arr, valid_mask=imask, bleed_px=b_px, alpha_cutoff=alpha_cutoff, preserve_alpha=preserve_alpha)
    if int(mip_preview_level) > 0:
        dil = _simulate_mip_preview(dil, int(mip_preview_level))
    return dil


# ===========================================================================
# 2. 3D Camera-Projected PBR Decal / Stencil Stamps (`decal_stamp_op`)
# ===========================================================================

DECAL_STENCILS: dict[str, dict[str, Any]] = {
    "chrono_sigil": {
        "title": "Chronos Astrolabe Sigil",
        "color": "#F0C85A",
        "emissive_color": "#38F2FF",
        "roughness": 0.18,
        "metallic": 0.94,
        "height_delta": 0.28,
        "emissive_strength": 0.75,
    },
    "amethyst_crest": {
        "title": "Amethyst Crystal Crest",
        "color": "#B564FF",
        "emissive_color": "#C084FC",
        "roughness": 0.12,
        "metallic": 0.35,
        "height_delta": 0.32,
        "emissive_strength": 0.85,
    },
    "dragon_claw_slash": {
        "title": "Triple Claw Battle Scar",
        "color": "#261B2E",
        "emissive_color": "#000000",
        "roughness": 0.88,
        "metallic": 0.12,
        "height_delta": -0.35,
        "emissive_strength": 0.0,
    },
    "rivet_plate_seam": {
        "title": "Forged Steel Rivet Strip",
        "color": "#A3B1C6",
        "emissive_color": "#000000",
        "roughness": 0.24,
        "metallic": 0.92,
        "height_delta": 0.30,
        "emissive_strength": 0.0,
    },
    "arcane_rune_ring": {
        "title": "Arcane Cyan Rune Ring",
        "color": "#38F2FF",
        "emissive_color": "#38F2FF",
        "roughness": 0.15,
        "metallic": 0.15,
        "height_delta": 0.22,
        "emissive_strength": 1.0,
    },
    "weather_scorch": {
        "title": "Battle Scorch & Soot",
        "color": "#1A181C",
        "emissive_color": "#FF6B2B",
        "roughness": 0.94,
        "metallic": 0.04,
        "height_delta": -0.14,
        "emissive_strength": 0.18,
    },
    "guild_star": {
        "title": "Artificer 8-Point Star",
        "color": "#F5D06B",
        "emissive_color": "#FFE79A",
        "roughness": 0.16,
        "metallic": 0.96,
        "height_delta": 0.30,
        "emissive_strength": 0.35,
    },
    "moss_lichen_patch": {
        "title": "Stylized Moss & Lichen",
        "color": "#4C8C4A",
        "emissive_color": "#000000",
        "roughness": 0.90,
        "metallic": 0.0,
        "height_delta": 0.18,
        "emissive_strength": 0.0,
    },
}


def render_decal_stencil(
    stencil_id: str,
    size: int = 96,
    *,
    tint_color: str | None = None,
    custom_data_url: str = "",
) -> dict[str, np.ndarray]:
    """Generate a procedural multi-channel PBR stencil (albedo RGBA, height delta, emissive mask) or decode custom PNG."""
    S = max(16, min(256, int(size)))
    spec = DECAL_STENCILS.get(stencil_id, DECAL_STENCILS["chrono_sigil"])
    tint = np.array(hex_to_srgb(tint_color or spec["color"]), dtype=np.float32)
    em_col = np.array(hex_to_srgb(spec["emissive_color"]), dtype=np.float32)

    if custom_data_url and "," in custom_data_url:
        try:
            raw = base64.b64decode(custom_data_url.split(",", 1)[1])
            im = Image.open(io.BytesIO(raw)).convert("RGBA").resize((S, S), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            return {
                "albedo": arr,
                "height_mask": arr[..., 3] * luma,
                "emissive_mask": arr[..., 3] * np.clip((luma - 0.55) / 0.45, 0.0, 1.0),
                "emissive_rgb": arr[..., :3],
            }
        except Exception:
            pass

    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    u = (xx + 0.5) / float(S) - 0.5
    v = (yy + 0.5) / float(S) - 0.5
    r = np.hypot(u, v)
    ang = np.arctan2(v, u)

    alpha = np.zeros((S, S), dtype=np.float32)
    em_mask = np.zeros((S, S), dtype=np.float32)
    shade = np.ones((S, S), dtype=np.float32)
    h_mask = np.zeros((S, S), dtype=np.float32)

    if stencil_id == "chrono_sigil":
        ring_outer = np.clip(1.0 - np.abs(r - 0.37) / 0.045, 0.0, 1.0)
        ring_inner = np.clip(1.0 - np.abs(r - 0.20) / 0.032, 0.0, 1.0)
        spokes = np.clip(1.0 - np.abs(np.sin(ang * 4.0)) / 0.16, 0.0, 1.0) * (r <= 0.38) * (r >= 0.18)
        diamond = np.clip(1.0 - (np.abs(u) + np.abs(v)) / 0.14, 0.0, 1.0)
        alpha = np.clip(np.maximum.reduce([ring_outer, ring_inner, spokes * 0.9, diamond]), 0.0, 1.0)
        em_mask = np.clip(ring_inner * 0.85 + diamond, 0.0, 1.0)
        h_mask = alpha
        shade = 0.78 + 0.35 * np.clip(1.0 - np.abs(r - 0.37) / 0.045, 0.0, 1.0)
    elif stencil_id == "amethyst_crest":
        d_hex = np.maximum(np.abs(u) * 1.15, np.abs(u) * 0.575 + np.abs(v) * 0.95)
        outer = np.clip((0.40 - d_hex) / 0.04, 0.0, 1.0)
        inner_cut = np.clip((0.26 - d_hex) / 0.035, 0.0, 1.0)
        core = np.clip((0.14 - (np.abs(u) + np.abs(v) * 0.75)) / 0.03, 0.0, 1.0)
        alpha = np.clip(np.maximum(outer * (1.0 - inner_cut * 0.75), core), 0.0, 1.0)
        em_mask = np.clip(core + (outer - inner_cut) * 0.4, 0.0, 1.0)
        h_mask = np.clip(outer * (1.0 - d_hex * 1.8) + core * 0.6, 0.0, 1.0)
        shade = 0.75 + 0.45 * (u + 0.5 - v * 0.5)
    elif stencil_id == "dragon_claw_slash":
        # Three diagonal tapered claw slashes
        for off, length in ((-0.14, 0.38), (0.0, 0.44), (0.14, 0.36)):
            lu = (u * 0.76 + v * 0.65)
            lv = (-u * 0.65 + v * 0.76) - off
            taper = np.clip(1.0 - (lu / length) ** 2, 0.0, 1.0)
            w_s = 0.045 * taper + 0.004
            slash = np.clip(1.0 - np.abs(lv) / np.maximum(0.004, w_s), 0.0, 1.0) * (taper > 0.02)
            alpha = np.maximum(alpha, slash)
        h_mask = alpha
        shade = 0.65 + 0.35 * (1.0 - alpha)
    elif stencil_id == "rivet_plate_seam":
        band = np.clip((0.14 - np.abs(v)) / 0.025, 0.0, 1.0) * (np.abs(u) <= 0.45)
        seam = np.clip((0.022 - np.abs(v)) / 0.012, 0.0, 1.0) * (np.abs(u) <= 0.45)
        rivets = np.zeros_like(alpha)
        for rx in (-0.34, -0.17, 0.0, 0.17, 0.34):
            for ry in (-0.075, 0.075):
                rivets = np.maximum(rivets, np.clip((0.036 - np.hypot(u - rx, v - ry)) / 0.014, 0.0, 1.0))
        alpha = np.clip(np.maximum(band * 0.85, rivets), 0.0, 1.0)
        h_mask = np.clip(band * 0.5 + rivets * 0.9 - seam * 0.6, 0.0, 1.0)
        shade = np.clip(0.85 + rivets * 0.35 - seam * 0.45, 0.25, 1.2)
    elif stencil_id == "arcane_rune_ring":
        r1 = np.clip(1.0 - np.abs(r - 0.39) / 0.030, 0.0, 1.0)
        r2 = np.clip(1.0 - np.abs(r - 0.27) / 0.024, 0.0, 1.0)
        glyphs = ((np.sin(ang * 12.0) > 0.15) & (r > 0.28) & (r < 0.38)).astype(np.float32) * 0.9
        tri = np.clip(1.0 - np.abs(np.cos(ang * 3.0) * r - 0.12) / 0.026, 0.0, 1.0) * (r < 0.26)
        alpha = np.clip(np.maximum.reduce([r1, r2, glyphs, tri]), 0.0, 1.0)
        em_mask = alpha
        h_mask = alpha
    elif stencil_id == "weather_scorch":
        n_ang = 0.06 * np.sin(ang * 5.0) + 0.04 * np.cos(ang * 9.0 + 1.1)
        alpha = np.clip((0.38 + n_ang - r) / 0.22, 0.0, 1.0) ** 1.4
        em_mask = np.clip((0.14 - r) / 0.14, 0.0, 1.0) * 0.7
        h_mask = alpha
        shade = 0.55 + 0.45 * (r / 0.4)
    elif stencil_id == "guild_star":
        star_r = 0.19 + 0.22 * (np.abs(np.cos(ang * 4.0)) ** 2.2)
        alpha = np.clip((star_r - r) / 0.032, 0.0, 1.0)
        inner = np.clip((0.08 - r) / 0.025, 0.0, 1.0)
        h_mask = np.clip(alpha * (1.0 - r / np.maximum(0.05, star_r)) + inner * 0.5, 0.0, 1.0)
        em_mask = inner
        shade = 0.78 + 0.40 * h_mask
    else:  # moss_lichen_patch
        blobs = np.zeros_like(alpha)
        for bx, by, br in ((-0.14, -0.06, 0.16), (0.06, -0.10, 0.14), (0.14, 0.08, 0.15), (-0.04, 0.12, 0.17), (0.0, 0.0, 0.19)):
            blobs = np.maximum(blobs, np.clip((br - np.hypot(u - bx, v - by)) / 0.04, 0.0, 1.0))
        tex = 0.8 + 0.2 * np.sin(xx * 0.9) * np.cos(yy * 0.9)
        alpha = np.clip(blobs, 0.0, 1.0)
        h_mask = alpha * tex
        shade = 0.72 + 0.35 * tex

    rgb = np.clip(tint[None, None, :] * shade[..., None], 0.0, 1.0)
    # Blend emissive highlight color into albedo where glowing
    if float(em_mask.max()) > 0.01 and float(spec.get("emissive_strength", 0.0)) > 0.01:
        rgb = np.clip(rgb * (1.0 - em_mask[..., None] * 0.45) + em_col[None, None, :] * (em_mask[..., None] * 0.45), 0.0, 1.0)
    alb = np.concatenate((rgb, alpha[..., None]), axis=-1).astype(np.float32)
    em_rgb = np.broadcast_to(em_col[None, None, :], (S, S, 3)).astype(np.float32).copy()
    return {
        "albedo": alb,
        "height_mask": h_mask.astype(np.float32),
        "emissive_mask": em_mask.astype(np.float32),
        "emissive_rgb": em_rgb,
    }


def _bilinear_sample_2d(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized bilinear sample of (S, S[, C]) image at normalized (u, v) in [0, 1]."""
    S_h, S_w = img.shape[:2]
    fx = np.clip(u, 0.0, 1.0) * (S_w - 1)
    fy = np.clip(v, 0.0, 1.0) * (S_h - 1)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(S_w - 1, x0 + 1)
    y1 = np.minimum(S_h - 1, y0 + 1)
    tx = (fx - x0).astype(np.float32)
    ty = (fy - y0).astype(np.float32)
    if img.ndim == 3:
        tx = tx[..., None]
        ty = ty[..., None]
    c00 = img[y0, x0]
    c10 = img[y0, x1]
    c01 = img[y1, x0]
    c11 = img[y1, x1]
    return (c00 * (1.0 - tx) + c10 * tx) * (1.0 - ty) + (c01 * (1.0 - tx) + c11 * tx) * ty


def _ray_obb_first_hit_t(
    origins: np.ndarray,
    dirs: np.ndarray,
    obb_centers: np.ndarray,
    obb_RTs: np.ndarray,
    obb_halfs: np.ndarray,
) -> np.ndarray:
    """Exact vectorized Kay-Kajiya ray-vs-K-OBBs front-most hit distance `t_first` per ray (N,)."""
    N = origins.shape[0]
    K = obb_centers.shape[0]
    if N == 0 or K == 0:
        return np.full((N,), np.inf, dtype=np.float32)

    half_pad = np.maximum(obb_halfs.astype(np.float32), 1e-3)  # prevents zero-thickness plane miss
    t_first = np.full((N,), np.inf, dtype=np.float32)
    chunk = 1024
    c_f = obb_centers.astype(np.float32)
    rt_f = obb_RTs.astype(np.float32)

    for i0 in range(0, N, chunk):
        i1 = min(N, i0 + chunk)
        o_c = origins[i0:i1].astype(np.float32)
        d_c = dirs[i0:i1].astype(np.float32)
        rel = o_c[:, None, :] - c_f[None, :, :]  # (B, K, 3)
        ro = np.einsum("kij,bkj->bki", rt_f, rel)
        rd = np.einsum("kij,bj->bki", rt_f, d_c)

        par = np.abs(rd) < 1e-6
        safe_rd = np.where(par, 1.0, rd)
        t0 = (-half_pad[None, :, :] - ro) / safe_rd
        t1 = (half_pad[None, :, :] - ro) / safe_rd
        tmin_a = np.minimum(t0, t1)
        tmax_a = np.maximum(t0, t1)

        out_par = par & ((ro < -half_pad[None, :, :]) | (ro > half_pad[None, :, :]))
        tmin_a = np.where(par, -1e18, tmin_a)
        tmax_a = np.where(par, 1e18, tmax_a)

        t_enter = np.max(tmin_a, axis=-1)
        t_exit = np.min(tmax_a, axis=-1)
        valid = (~np.any(out_par, axis=-1)) & (t_exit >= t_enter) & (t_exit >= 0.0)
        cand_t = np.where(valid, np.maximum(0.0, t_enter), np.inf)
        t_first[i0:i1] = np.min(cand_t, axis=-1)

    return t_first


def _apply_single_decal_stamp(
    alb: np.ndarray,
    h_ch: np.ndarray,
    r_ch: np.ndarray,
    m_ch: np.ndarray,
    em_ch: np.ndarray,
    stamp: dict[str, Any],
    *,
    mesh_context: Any = None,
) -> bool:
    """Project one 3D (or 2D fallback) decal stamp in-place onto PBR channels. Returns True if height modified."""
    H, W = alb.shape[:2]
    stencil_id = str(stamp.get("stencil") or "chrono_sigil")
    spec = DECAL_STENCILS.get(stencil_id, DECAL_STENCILS["chrono_sigil"])
    tint = stamp.get("tint_color") or spec["color"]
    custom_url = str(stamp.get("custom_data_url") or "")
    st_maps = render_decal_stencil(stencil_id, size=96, tint_color=tint, custom_data_url=custom_url)

    opacity = float(np.clip(stamp.get("opacity", 1.0), 0.0, 1.0))
    if opacity <= 1e-4:
        return False
    roll_rad = math.radians(float(stamp.get("roll_deg", 0.0)))
    scale_u = max(0.02, float(stamp.get("scale_u", 10.0)))
    scale_v = max(0.02, float(stamp.get("scale_v", scale_u)))
    max_ang = max(5.0, min(90.0, float(stamp.get("max_angle_deg", 72.0))))
    fade_ang = max(0.0, min(max_ang - 1.0, float(stamp.get("angle_fade_deg", 20.0))))
    use_occ = bool(stamp.get("occlusion_test", True))
    bmode = str(stamp.get("blend_mode") or "normal").lower()

    rough_v = float(np.clip(stamp.get("roughness", spec["roughness"]), 0.04, 0.98))
    metal_v = float(np.clip(stamp.get("metallic", spec["metallic"]), 0.0, 1.0))
    h_delta = float(stamp.get("height_delta", spec["height_delta"])) * float(stamp.get("height_strength", 1.0))
    em_str = float(np.clip(stamp.get("emissive_strength", spec["emissive_strength"]), 0.0, 1.5))

    if mesh_context is not None and hasattr(mesh_context, "world_pos"):
        imask = np.asarray(mesh_context.island_mask, dtype=bool)
        wpos = np.asarray(mesh_context.world_pos, dtype=np.float32)
        wnrm = np.asarray(mesh_context.world_normal, dtype=np.float32)

        # Projector basis in .bbmodel world space
        fwd = np.asarray(stamp.get("forward") or [0.0, 0.0, -1.0], dtype=np.float32)
        fn = float(np.linalg.norm(fwd))
        fwd = fwd / fn if fn > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)

        up_in = np.asarray(stamp.get("up") or [0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(up_in, fwd))) > 0.98:
            up_in = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, up_in)
        rn = float(np.linalg.norm(right))
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up = np.cross(right, fwd)
        up = up / max(1e-6, float(np.linalg.norm(up)))

        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        r_ax = cr * right + sr * up
        u_ax = -sr * right + cr * up

        if stamp.get("origin") is not None:
            origin = np.asarray(stamp["origin"], dtype=np.float32)
        else:
            valid_pts = wpos[imask]
            center_m = valid_pts.mean(axis=0) if valid_pts.size else np.array([0.0, 12.0, 0.0], dtype=np.float32)
            origin = center_m - fwd * 24.0

        rel = wpos - origin[None, None, :]
        xp = np.einsum("hwc,c->hw", rel, r_ax)
        yp = np.einsum("hwc,c->hw", rel, u_ax)
        zp = np.einsum("hwc,c->hw", rel, fwd)

        u = xp / scale_u + 0.5
        v = 0.5 - yp / scale_v
        in_bounds = imask & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)

        # Normal-angle cutoff + backface culling: dot(world_normal, -forward)
        cos_a = np.einsum("hwc,c->hw", wnrm, -fwd)
        c_min = math.cos(math.radians(max_ang))
        c_fade = math.cos(math.radians(max(0.0, max_ang - fade_ang)))
        if fade_ang <= 1e-3:
            w_ang = (cos_a >= c_min).astype(np.float32)
        else:
            t_a = np.clip((cos_a - c_min) / max(1e-5, c_fade - c_min), 0.0, 1.0)
            w_ang = (t_a * t_a * (3.0 - 2.0 * t_a)).astype(np.float32)
        w_ang = np.where(cos_a > 0.0, w_ang, 0.0)

        # Optional depth slab bounds
        z_min = float(stamp.get("depth_min", -1e6))
        z_max = float(stamp.get("depth_max", 1e6))
        w_depth = ((zp >= z_min) & (zp <= z_max)).astype(np.float32)

        cand = in_bounds & (w_ang > 1e-4) & (w_depth > 1e-4)
        w_occ = np.ones((H, W), dtype=np.float32)
        if (
            use_occ
            and np.any(cand)
            and getattr(mesh_context, "obb_centers", None) is not None
            and len(mesh_context.obb_centers) > 0
        ):
            cy, cx = np.nonzero(cand)
            pts_c = wpos[cy, cx]
            zp_c = zp[cy, cx]
            # Shoot orthographic projector ray starting behind the projector plane along +fwd
            ray_start_z = float(np.min(zp_c)) - 4.0
            ray_o = pts_c - (zp_c - ray_start_z)[:, None] * fwd[None, :]
            ray_d = np.broadcast_to(fwd[None, :], ray_o.shape)
            t_first = _ray_obb_first_hit_t(
                ray_o,
                ray_d,
                mesh_context.obb_centers,
                mesh_context.obb_RTs,
                mesh_context.obb_halfs,
            )
            ray_dist = (zp_c - ray_start_z).astype(np.float32)
            visible = ray_dist <= (t_first + 0.15)
            w_occ[cy, cx] = visible.astype(np.float32)

        proj_w = in_bounds.astype(np.float32) * w_ang * w_depth * w_occ
    else:
        # 2D fallback planar stamp on UV canvas
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cu = float(stamp.get("center_u", 0.5))
        cv = float(stamp.get("center_v", 0.5))
        su = max(0.05, float(stamp.get("scale_2d", 0.45)))
        du = (xx / max(1.0, W - 1.0) - cu) / su
        dv = (yy / max(1.0, H - 1.0) - cv) / su
        cr, sr = math.cos(roll_rad), math.sin(roll_rad)
        u = cr * du + sr * dv + 0.5
        v = -sr * du + cr * dv + 0.5
        in_bounds = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        proj_w = in_bounds.astype(np.float32)

    if float(proj_w.max()) <= 1e-4:
        return False

    samp_alb = _bilinear_sample_2d(st_maps["albedo"], u, v)
    samp_h = _bilinear_sample_2d(st_maps["height_mask"], u, v)
    samp_em = _bilinear_sample_2d(st_maps["emissive_mask"], u, v)
    samp_em_rgb = _bilinear_sample_2d(st_maps["emissive_rgb"], u, v)

    w = np.clip(samp_alb[..., 3] * proj_w * opacity, 0.0, 1.0)
    if float(w.max()) <= 1e-4:
        return False

    if bmode == "normal":
        alb[..., :3] = alb[..., :3] * (1.0 - w[..., None]) + samp_alb[..., :3] * w[..., None]
        alb[..., 3] = np.clip(np.maximum(alb[..., 3], w), 0.0, 1.0)
    else:
        blended = blend_layers(alb, samp_alb, mode=bmode, opacity=opacity)
        alb[..., :3] = alb[..., :3] * (1.0 - w[..., None]) + blended[..., :3] * w[..., None]
        alb[..., 3] = np.clip(np.maximum(alb[..., 3], w), 0.0, 1.0)

    r_ch[...] = r_ch * (1.0 - w) + rough_v * w
    m_ch[...] = m_ch * (1.0 - w) + metal_v * w
    h_ch[...] = np.clip(h_ch + h_delta * samp_h * proj_w * opacity, 0.0, 1.0)

    if em_str > 0.005:
        we = np.clip(samp_em * proj_w * opacity * em_str, 0.0, 1.0)
        em_ch[..., :3] = np.clip(em_ch[..., :3] * (1.0 - we[..., None]) + samp_em_rgb * we[..., None], 0.0, 1.0)
        em_ch[..., 3] = np.clip(np.maximum(em_ch[..., 3], we), 0.0, 1.0)
    return abs(h_delta) > 1e-4


def decal_stamp_op(
    height: int,
    width: int,
    *,
    base: dict[str, np.ndarray] | np.ndarray | None = None,
    mesh_context: Any = None,
    stencil: str = "chrono_sigil",
    tint_color: str = "#F0C85A",
    scale_u: float = 10.0,
    roll_deg: float = 0.0,
    max_angle_deg: float = 72.0,
    angle_fade_deg: float = 20.0,
    opacity: float = 0.95,
    occlusion_test: bool = True,
    height_strength: float = 1.0,
    emissive_strength: float = 0.75,
    stamps_json: Any = "",
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Composite one or more 3D camera-projected PBR decal/stencil stamps cleanly onto a MATERIAL bus."""
    H, W = int(height), int(width)
    if isinstance(base, dict) and "albedo" in base:
        alb = np.asarray(base["albedo"], dtype=np.float32).copy()
        h_ch = np.asarray(base.get("height", np.full((H, W), 0.5, dtype=np.float32)), dtype=np.float32).copy()
        r_ch = np.asarray(base.get("roughness", np.full((H, W), 0.55, dtype=np.float32)), dtype=np.float32).copy()
        m_ch = np.asarray(base.get("metallic", np.zeros((H, W), dtype=np.float32)), dtype=np.float32).copy()
        ao_ch = np.asarray(base.get("ao", np.ones((H, W), dtype=np.float32)), dtype=np.float32).copy()
        em_ch = (
            np.asarray(base["emissive"], dtype=np.float32).copy()
            if "emissive" in base and base["emissive"] is not None
            else np.zeros((H, W, 4), dtype=np.float32)
        )
        sh_ch = base.get("shadow")
        bn_ch = base.get("bent_normal")
        th_ch = base.get("thickness")
        sss_ch = base.get("sss")
        por_ch = base.get("porosity")
        nrm_in = base.get("normal")
    elif isinstance(base, np.ndarray):
        arr = np.asarray(base, dtype=np.float32)
        if arr.ndim == 2:
            alb = np.stack((arr, arr, arr, np.ones_like(arr)), axis=-1)
        elif arr.shape[2] == 3:
            alb = np.concatenate((arr, np.ones((H, W, 1), dtype=np.float32)), axis=-1)
        else:
            alb = arr.copy()
        h_ch = np.full((H, W), 0.5, dtype=np.float32)
        r_ch = np.full((H, W), 0.55, dtype=np.float32)
        m_ch = np.zeros((H, W), dtype=np.float32)
        ao_ch = np.ones((H, W), dtype=np.float32)
        em_ch = np.zeros((H, W, 4), dtype=np.float32)
        sh_ch = bn_ch = th_ch = sss_ch = por_ch = nrm_in = None
    else:
        alb = np.zeros((H, W, 4), dtype=np.float32)
        alb[..., :3] = np.array([0.20, 0.22, 0.28], dtype=np.float32)
        alb[..., 3] = 1.0
        h_ch = np.full((H, W), 0.5, dtype=np.float32)
        r_ch = np.full((H, W), 0.55, dtype=np.float32)
        m_ch = np.zeros((H, W), dtype=np.float32)
        ao_ch = np.ones((H, W), dtype=np.float32)
        em_ch = np.zeros((H, W, 4), dtype=np.float32)
        sh_ch = bn_ch = th_ch = sss_ch = por_ch = nrm_in = None

    stamps: list[dict[str, Any]] = []
    if isinstance(stamps_json, list):
        stamps = [s for s in stamps_json if isinstance(s, dict)]
    elif isinstance(stamps_json, str) and stamps_json.strip():
        try:
            parsed = json.loads(stamps_json)
            if isinstance(parsed, list):
                stamps = [s for s in parsed if isinstance(s, dict)]
        except Exception:
            stamps = []

    if not stamps:
        stamps = [
            {
                "stencil": stencil,
                "tint_color": tint_color,
                "scale_u": scale_u,
                "scale_v": scale_u,
                "roll_deg": roll_deg,
                "max_angle_deg": max_angle_deg,
                "angle_fade_deg": angle_fade_deg,
                "opacity": opacity,
                "occlusion_test": occlusion_test,
                "height_strength": height_strength,
                "emissive_strength": emissive_strength,
            }
        ]

    any_h = False
    for st in stamps:
        if st.get("visible", True) is False:
            continue
        if _apply_single_decal_stamp(alb, h_ch, r_ch, m_ch, em_ch, st, mesh_context=mesh_context):
            any_h = True

    if any_h or nrm_in is None:
        nrm_h = height_to_normal_map(h_ch, strength=2.6, tileable=tileable)
        if nrm_in is not None and any_h:
            nrm_out = np.clip(0.55 * nrm_h + 0.45 * np.asarray(nrm_in, dtype=np.float32), 0.0, 1.0)
        else:
            nrm_out = nrm_h
    else:
        nrm_out = np.asarray(nrm_in, dtype=np.float32)

    return material_make(
        np.clip(alb, 0.0, 1.0),
        height=np.clip(h_ch, 0.0, 1.0),
        normal=nrm_out,
        roughness=np.clip(r_ch, 0.04, 0.98),
        metallic=np.clip(m_ch, 0.0, 1.0),
        ao=np.clip(ao_ch, 0.0, 1.0),
        emissive=em_ch if float(em_ch[..., 3].max()) > 1e-4 else None,
        shadow=sh_ch,
        bent_normal=bn_ch,
        thickness=th_ch,
        sss=sss_ch,
        porosity=por_ch,
    )


# ===========================================================================
# 3. Multi-Material Asset Atlas Baker (MaxRects-BSSF + Mip-Safe Dilation)
# ===========================================================================


def _next_pot(v: int) -> int:
    p = 16
    while p < v:
        p *= 2
    return p


def _ceil16(v: int) -> int:
    return max(16, int(math.ceil(v / 16.0)) * 16)


def _maxrects_bssf_pack(
    rects: list[tuple[str, int, int]],
    *,
    canvas_mode: str = "pot",
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """Deterministic MaxRects Best-Short-Side-Fit packer returning `(atlas_w, atlas_h, {id: (x, y)})`."""
    if not rects:
        return 64, 64, {}
    ordered = sorted(rects, key=lambda r: (max(r[1], r[2]), r[1] * r[2]), reverse=True)
    total_area = sum(w * h for _, w, h in ordered)
    max_w = max(w for _, w, _ in ordered)
    max_h = max(h for _, _, h in ordered)

    side = max(max_w, max_h, int(math.ceil(math.sqrt(total_area * 1.18))))
    w_try = _next_pot(side) if canvas_mode == "pot" else _ceil16(side)
    h_try = w_try

    for _attempt in range(12):
        free: list[tuple[int, int, int, int]] = [(0, 0, w_try, h_try)]
        placed: dict[str, tuple[int, int]] = {}
        ok = True
        for rid, rw, rh in ordered:
            best_idx = -1
            best_score = (10**9, 10**9)
            best_pos = (0, 0)
            for idx, (fx, fy, fw, fh) in enumerate(free):
                if rw <= fw and rh <= fh:
                    short_side = min(fw - rw, fh - rh)
                    long_side = max(fw - rw, fh - rh)
                    sc = (short_side, long_side)
                    if sc < best_score:
                        best_score = sc
                        best_idx = idx
                        best_pos = (fx, fy)
            if best_idx < 0:
                ok = False
                break
            px, py = best_pos
            placed[rid] = (px, py)
            # Split overlapping free rectangles
            new_free: list[tuple[int, int, int, int]] = []
            for fx, fy, fw, fh in free:
                if px >= fx + fw or px + rw <= fx or py >= fy + fh or py + rh <= fy:
                    new_free.append((fx, fy, fw, fh))
                    continue
                if px > fx:
                    new_free.append((fx, fy, px - fx, fh))
                if px + rw < fx + fw:
                    new_free.append((px + rw, fy, (fx + fw) - (px + rw), fh))
                if py > fy:
                    new_free.append((fx, fy, fw, py - fy))
                if py + rh < fy + fh:
                    new_free.append((fx, py + rh, fw, (fy + fh) - (py + rh)))
            # Prune fully contained free rects
            pruned: list[tuple[int, int, int, int]] = []
            for i, a in enumerate(new_free):
                ax0, ay0, aw, ah = a
                contained = False
                for j, b in enumerate(new_free):
                    if i == j:
                        continue
                    bx0, by0, bw, bh = b
                    if ax0 >= bx0 and ay0 >= by0 and ax0 + aw <= bx0 + bw and ay0 + ah <= by0 + bh:
                        contained = True
                        break
                if not contained and aw > 0 and ah > 0:
                    pruned.append(a)
            free = pruned
        if ok:
            used_w = max(placed[rid][0] + rw for rid, rw, _ in ordered)
            used_h = max(placed[rid][1] + rh for rid, _, rh in ordered)
            fin_w = _next_pot(used_w) if canvas_mode == "pot" else _ceil16(used_w)
            fin_h = _next_pot(used_h) if canvas_mode == "pot" else _ceil16(used_h)
            return fin_w, fin_h, placed
        if canvas_mode == "pot":
            if w_try <= h_try:
                w_try *= 2
            else:
                h_try *= 2
        else:
            w_try = _ceil16(int(w_try * 1.25) + 16)
            h_try = _ceil16(int(h_try * 1.25) + 16)
    raise RuntimeError("MaxRects-BSSF could not fit atlas rectangles within bounds")


def bake_multi_asset_atlas(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    stem: str = "synccraft_atlas",
    bleed_px: int = 4,
    canvas_mode: str = "pot",  # "pot" | "tight"
    max_assets: int = 16,
) -> dict[str, Any]:
    """Pack multiple `.bbmodel`/`.texdsl` assets into a shared mip-safe PBR atlas + remapped `.bbmodel` files."""
    from .mesh_baker import bake_bbmodel_maps
    from .texdsl import parse_texdsl
    from .v16_pro_ops import scan_batch_folder

    t0 = time.perf_counter()
    assets = scan_batch_folder(input_dir, max_assets=max_assets)
    if not assets:
        raise FileNotFoundError(f"No .bbmodel or .texdsl assets found in {input_dir}")

    pad = max(0, min(16, int(bleed_px)))
    evaluated: list[dict[str, Any]] = []
    rects: list[tuple[str, int, int]] = []

    for a in assets:
        bb_doc = None
        mc = None
        if a["bbmodel_path"]:
            bb_doc = json.loads(Path(a["bbmodel_path"]).read_text(encoding="utf-8"))
            mc = bake_bbmodel_maps(bb_doc)
        if a["texdsl_path"]:
            g = parse_texdsl(Path(a["texdsl_path"]).read_text(encoding="utf-8"), mesh_context=mc)
        elif mc is not None:
            from .studio_server import StudioSession
            g = StudioSession._build_bbmodel_starter_graph(bb_doc or {}, mc)
            g.name = a["stem"]
        else:
            continue
        if mc is not None:
            g.attach_mesh(mc)
        ev = g.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
        pm = ev.pbr_maps or pack_labpbr_1_3(ev.output)
        H_a, W_a = pm["albedo"].shape[:2]
        seed_m = mc.island_mask if mc is not None else (pm["albedo"][..., 3] > 5)
        evaluated.append(
            {
                "id": a["id"],
                "stem": a["stem"],
                "w": W_a,
                "h": H_a,
                "albedo": pm["albedo"],
                "normal": pm["normal_rgb"],
                "orm": pm["orm_u8"],
                "emissive": rgba_float_to_u8(ev.emissive) if ev.emissive is not None else np.zeros((H_a, W_a, 4), dtype=np.uint8),
                "seed_mask": seed_m,
                "bbmodel": bb_doc,
            }
        )
        rects.append((a["id"], W_a + 2 * pad, H_a + 2 * pad))

    atlas_w, atlas_h, placed = _maxrects_bssf_pack(rects, canvas_mode=canvas_mode)
    alb_atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    nrm_atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    nrm_atlas[..., 0] = 128
    nrm_atlas[..., 1] = 128
    nrm_atlas[..., 2] = 255
    nrm_atlas[..., 3] = 255
    orm_atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    orm_atlas[..., 0] = 255
    orm_atlas[..., 1] = 165
    orm_atlas[..., 3] = 255
    em_atlas = np.zeros((atlas_h, atlas_w, 4), dtype=np.uint8)
    em_atlas[..., 3] = 255
    valid_atlas = np.zeros((atlas_h, atlas_w), dtype=bool)

    entries: list[dict[str, Any]] = []
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for item in evaluated:
        px, py = placed[item["id"]]
        dx, dy = px + pad, py + pad
        w_i, h_i = item["w"], item["h"]
        alb_atlas[dy : dy + h_i, dx : dx + w_i] = item["albedo"]
        nrm_atlas[dy : dy + h_i, dx : dx + w_i] = item["normal"]
        orm_atlas[dy : dy + h_i, dx : dx + w_i] = item["orm"]
        em_atlas[dy : dy + h_i, dx : dx + w_i] = item["emissive"]
        valid_atlas[dy : dy + h_i, dx : dx + w_i] = item["seed_mask"]
        entries.append(
            {
                "id": item["id"],
                "stem": item["stem"],
                "x": dx,
                "y": dy,
                "w": w_i,
                "h": h_i,
                "uv_rect": [round(dx / atlas_w, 5), round(dy / atlas_h, 5), round((dx + w_i) / atlas_w, 5), round((dy + h_i) / atlas_h, 5)],
            }
        )

    # Apply mip-safe Voronoi edge dilation across the whole atlas!
    if pad > 0:
        alb_atlas = mip_safe_dilate_rgba(alb_atlas, valid_mask=valid_atlas, bleed_px=pad, preserve_alpha=True)
        nrm_atlas = mip_safe_dilate_normal(nrm_atlas, valid_atlas, bleed_px=pad)
        orm_atlas = np.stack(
            (
                mip_safe_dilate_scalar(orm_atlas[..., 0], valid_atlas, bleed_px=pad),
                mip_safe_dilate_scalar(orm_atlas[..., 1], valid_atlas, bleed_px=pad),
                mip_safe_dilate_scalar(orm_atlas[..., 2], valid_atlas, bleed_px=pad),
                orm_atlas[..., 3],
            ),
            axis=-1,
        )

    # Save atlas PBR maps
    files: dict[str, str] = {}
    for key, fname, arr in (
        ("albedo", f"{stem}_albedo.png", alb_atlas),
        ("normal", f"{stem}_normal.png", nrm_atlas),
        ("orm", f"{stem}_orm.png", orm_atlas),
        ("emissive", f"{stem}_emissive.png", em_atlas),
    ):
        fp = out_root / fname
        Image.fromarray(arr, mode="RGBA").save(fp, format="PNG", optimize=True)
        files[key] = str(fp)

    # Encode atlas albedo data URL once for remapped .bbmodel files
    buf = io.BytesIO()
    Image.fromarray(alb_atlas, mode="RGBA").save(buf, format="PNG", optimize=True)
    atlas_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    remapped_models: list[str] = []
    for item, ent in zip(evaluated, entries):
        if item["bbmodel"] is None:
            continue
        doc = copy.deepcopy(item["bbmodel"])
        old_res = doc.get("resolution") or {}
        old_w = max(1.0, float(old_res.get("width", item["w"])))
        old_h = max(1.0, float(old_res.get("height", item["h"])))
        sx = float(item["w"]) / old_w
        sy = float(item["h"]) / old_h
        dx, dy = float(ent["x"]), float(ent["y"])
        doc["resolution"] = {"width": atlas_w, "height": atlas_h}
        doc["textures"] = [
            {
                "name": f"{stem}_albedo.png",
                "id": "0",
                "width": atlas_w,
                "height": atlas_h,
                "uv_width": atlas_w,
                "uv_height": atlas_h,
                "internal": True,
                "source": atlas_url,
            }
        ]
        for elem in doc.get("elements") or []:
            if not isinstance(elem, dict):
                continue
            for finfo in (elem.get("faces") or {}).values():
                if not isinstance(finfo, dict) or not finfo.get("uv") or len(finfo["uv"]) < 4:
                    continue
                u0, v0, u1, v1 = finfo["uv"][:4]
                finfo["uv"] = [
                    round(float(u0) * sx + dx, 4),
                    round(float(v0) * sy + dy, 4),
                    round(float(u1) * sx + dx, 4),
                    round(float(v1) * sy + dy, 4),
                ]
                finfo["texture"] = 0
        out_bb = out_root / f"{item['stem']}_atlas.bbmodel"
        out_bb.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        remapped_models.append(str(out_bb))

    fill_ratio = round(sum(e["w"] * e["h"] for e in entries) / max(1.0, float(atlas_w * atlas_h)), 4)
    manifest = {
        "schema": "synccraft.atlas/1.0",
        "ok": True,
        "stem": stem,
        "atlas_width": atlas_w,
        "atlas_height": atlas_h,
        "canvas_mode": canvas_mode,
        "bleed_px": pad,
        "fill_ratio": fill_ratio,
        "asset_count": len(entries),
        "entries": entries,
        "files": files,
        "remapped_bbmodels": remapped_models,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    mf_path = out_root / f"{stem}_atlas_manifest.json"
    mf_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(mf_path)
    return manifest


# ===========================================================================
# 4. Canonical One-Click 6-Engine PBR Export Presets (Track 2 Spec Parity!)
# ===========================================================================

ENGINE_EXPORT_PRESETS: list[dict[str, Any]] = [
    {
        "id": "full_pbr_v2",
        "title": "SyncCraft Full PBR v2.0 + MaterialX",
        "engine": "Universal Multi-Engine",
        "normal_convention": "OpenGL (Y-up)",
        "description": "Exports all 12 Special PBR channels, LabPBR 1.3, Bedrock MER, UE5 ORM, Unity MaskMap, MaterialX 1.39, .texdsl & .texgraph.json.",
        "channels": [
            {"file": "<stem>.png", "slots": ["RGB: Albedo (sRGB)", "A: Opacity"]},
            {"file": "<stem>_normal.png", "slots": ["RGB: Tangent Normal (OpenGL)"]},
            {"file": "<stem>_orm.png", "slots": ["R: AO", "G: Roughness", "B: Metallic"]},
            {"file": "<stem>.mtlx", "slots": ["ASWF MaterialX 1.39 OpenPBR"]},
        ],
    },
    {
        "id": "unreal_ue5",
        "title": "Unreal Engine 5 (ORM + DirectX Normal)",
        "engine": "Unreal Engine 5 / Substrate",
        "normal_convention": "DirectX (Y-down)",
        "description": "Native UE5 texture naming (T_<stem>_D/N/ORM/E) with DirectX Y-down normal map and sRGB-off ORM mask.",
        "channels": [
            {"file": "T_<stem>_D.png", "slots": ["RGB: BaseColor (sRGB)", "A: Opacity"]},
            {"file": "T_<stem>_N.png", "slots": ["RGB: Normal (DirectX Y-down)"]},
            {"file": "T_<stem>_ORM.png", "slots": ["R: AO", "G: Roughness", "B: Metallic"]},
            {"file": "T_<stem>_E.png", "slots": ["RGB: Emissive (sRGB)"]},
        ],
    },
    {
        "id": "unity_hdrp",
        "title": "Unity HDRP / URP (MaskMap + OpenGL Normal)",
        "engine": "Unity HDRP / URP",
        "normal_convention": "OpenGL (Y-up)",
        "description": "Spec-accurate Unity HDRP MaskMap (R=Metallic, G=AO, B=Neutral Detail 255, A=Smoothness) + separate HeightMap & OpenGL Normal.",
        "channels": [
            {"file": "<stem>_Albedo.png", "slots": ["RGB: BaseMap (sRGB)", "A: Alpha"]},
            {"file": "<stem>_Normal.png", "slots": ["RGB: Normal (OpenGL Y-up)"]},
            {"file": "<stem>_MaskMap.png", "slots": ["R: Metallic", "G: AO", "B: Detail (255)", "A: Smoothness"]},
            {"file": "<stem>_HeightMap.png", "slots": ["R: Height / POM"]},
        ],
    },
    {
        "id": "java_labpbr",
        "title": "Minecraft Java LabPBR 1.3 (Iris / Optifine)",
        "engine": "Minecraft Java LabPBR 1.3",
        "normal_convention": "DirectX (Y-down)",
        "description": "shaderLABS LabPBR 1.3 compliant trio with DirectX Y-down _n.G, POM Height in _n.A [1..255], and Emissive in _s.A [0..254, 255=none].",
        "channels": [
            {"file": "<stem>.png", "slots": ["RGB: Albedo (sRGB)", "A: Cutout Alpha"]},
            {"file": "<stem>_n.png", "slots": ["R: Nx", "G: Ny (DirectX)", "B: AO", "A: Height 1..255"]},
            {"file": "<stem>_s.png", "slots": ["R: Smooth", "G: F0/MetalID", "B: Por/SSS", "A: Emissive"]},
        ],
    },
    {
        "id": "bedrock_rtx",
        "title": "Minecraft Bedrock RTX / Vibrant Visuals 1.21.30",
        "engine": "Minecraft Bedrock RTX",
        "normal_convention": "OpenGL (Y-up)",
        "description": "Bedrock 1.21.30 texture_set.json + _mer.png + _mers.png (Subsurface in Alpha) + _normal.png.",
        "channels": [
            {"file": "<stem>.png", "slots": ["RGB: Color (sRGB)", "A: Alpha"]},
            {"file": "<stem>_mers.png", "slots": ["R: Metal", "G: Emissive", "B: Roughness", "A: SSS"]},
            {"file": "<stem>_normal.png", "slots": ["RGB: Tangent Normal"]},
            {"file": "<stem>.texture_set.json", "slots": ["format_version: 1.21.30"]},
        ],
    },
    {
        "id": "maya_materialx",
        "title": "Autodesk Maya LookdevX / OpenUSD (MaterialX 1.39)",
        "engine": "Maya LookdevX / Karma / USD",
        "normal_convention": "OpenGL (Y-up)",
        "description": "ASWF MaterialX 1.39 OpenPBR Surface (.mtlx) + mip-dilated OpenGL PBR PNG maps ready for Arnold, Karma & USD.",
        "channels": [
            {"file": "<stem>.mtlx", "slots": ["MaterialX 1.39 open_pbr_surface"]},
            {"file": "<stem>.png", "slots": ["RGB: Base Color (sRGB)"]},
            {"file": "<stem>_normal.png", "slots": ["RGB: Normal (OpenGL Y-up)"]},
            {"file": "<stem>_roughness/_metallic/_ao.png", "slots": ["Linear Scalar Maps"]},
        ],
    },
    {
        "id": "blockbench_baked",
        "title": "Blockbench 5 / Hytale Stylized Pre-Baked",
        "engine": "Blockbench / Vanilla Minecraft / Hytale",
        "normal_convention": "Pre-Baked Studio Lighting",
        "description": "Mip-safe Voronoi-dilated pre-baked studio albedo PNG + ready-to-open _baked.bbmodel with embedded texture.",
        "channels": [
            {"file": "<stem>_baked.png", "slots": ["RGB: Pre-Baked PBR + AO + Emissive", "A: Alpha"]},
            {"file": "<stem>_baked.bbmodel", "slots": ["Embedded Base64 Texture"]},
        ],
    },
]


def _flip_normal_green_u8(nrm_u8: np.ndarray) -> np.ndarray:
    """Convert OpenGL Y-up tangent normal u8 -> DirectX Y-down (flipping G = 255 - G)."""
    out = np.asarray(nrm_u8, dtype=np.uint8).copy()
    out[..., 1] = 255 - out[..., 1]
    return out


def export_engine_preset(
    graph: Any,
    preset_id: str,
    *,
    stem: str | None = None,
    out_dir: str | Path | None = None,
    lighting_preset: str = "studio_hero",
    bleed_px: int = 4,
    bbmodel_doc: dict[str, Any] | None = None,
    shader_model: str = "open_pbr_surface",
    coat_weight: float = 0.15,
) -> dict[str, Any]:
    """Export an engine-tailored PBR bundle adhering strictly to Track 2 primary specs + mip-safe Voronoi dilation."""
    from .syncmat_and_mtlx import export_materialx_xml
    from .texdsl import export_texdsl

    st = str(stem or getattr(graph, "name", None) or "synccraft_asset")
    pid = str(preset_id or "full_pbr_v2").strip().lower()
    if lighting_preset not in BAKE_LIGHTING_PRESETS:
        lighting_preset = "studio_hero"
    ev = graph.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
    pm = copy.deepcopy(ev.pbr_maps or pack_labpbr_1_3(ev.output))
    mc = getattr(graph, "mesh_context", None)
    seed_m = mc.island_mask if (mc is not None and hasattr(mc, "island_mask")) else (pm["albedo"][..., 3] > 5)

    # Apply mip-safe Voronoi dilation if requested (protecting alpha cutouts & normal unit vectors!)
    b_px = max(0, min(32, int(bleed_px)))
    if b_px > 0 and np.any(seed_m) and not np.all(seed_m):
        pm["albedo"] = mip_safe_dilate_rgba(pm["albedo"], valid_mask=seed_m, bleed_px=b_px, preserve_alpha=True)
        pm["normal_rgb"] = mip_safe_dilate_normal(pm["normal_rgb"], seed_m, bleed_px=b_px)
        for ch_k in ("roughness_u8", "metallic_u8", "height_u8", "ao_u8"):
            if ch_k in pm:
                dil_c = mip_safe_dilate_scalar(pm[ch_k][..., 0], seed_m, bleed_px=b_px)
                pm[ch_k] = np.stack((dil_c, dil_c, dil_c, pm[ch_k][..., 3]), axis=-1)
        if "orm_u8" in pm:
            pm["orm_u8"] = np.stack(
                (
                    mip_safe_dilate_scalar(pm["orm_u8"][..., 0], seed_m, bleed_px=b_px),
                    mip_safe_dilate_scalar(pm["orm_u8"][..., 1], seed_m, bleed_px=b_px),
                    mip_safe_dilate_scalar(pm["orm_u8"][..., 2], seed_m, bleed_px=b_px),
                    pm["orm_u8"][..., 3],
                ),
                axis=-1,
            )

    em_u8 = (
        rgba_float_to_u8(ev.emissive)
        if ev.emissive is not None
        else np.concatenate((np.zeros(pm["albedo"].shape[:2] + (3,), dtype=np.uint8), np.full(pm["albedo"].shape[:2] + (1,), 255, dtype=np.uint8)), axis=-1)
    )

    # Spec-accurate derived engine maps (Track 2 Audit)
    nrm_dx = _flip_normal_green_u8(pm["normal_rgb"])
    labpbr_n_dx = _flip_normal_green_u8(pm["labpbr_n"])
    # Unity HDRP MaskMap: R=Metallic, G=AO, B=DetailMask (255 neutral!), A=Smoothness
    ones_u8 = np.full(pm["albedo"].shape[:2], 255, dtype=np.uint8)
    smooth_u8 = (255 - pm["roughness_u8"][..., 0]).astype(np.uint8)
    unity_hdrp_maskmap = np.stack((pm["metallic_u8"][..., 0], pm["ao_u8"][..., 0], ones_u8, smooth_u8), axis=-1)
    # Bedrock 1.21.30 MERS: R=Metallic, G=Emissive, B=Roughness, A=Subsurface
    sss_u8 = (
        np.clip(np.rint(np.asarray(ev.pbr_channels["sss"], dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
        if (ev.pbr_channels and "sss" in ev.pbr_channels and ev.pbr_channels["sss"] is not None)
        else np.zeros(pm["albedo"].shape[:2], dtype=np.uint8)
    )
    bedrock_mers = np.stack((pm["bedrock_mer"][..., 0], pm["bedrock_mer"][..., 1], pm["bedrock_mer"][..., 2], sss_u8), axis=-1)

    png_files: dict[str, np.ndarray] = {}
    text_files: dict[str, str] = {}

    if pid == "unreal_ue5":
        png_files[f"T_{st}_D.png"] = pm["albedo"]
        png_files[f"T_{st}_N.png"] = nrm_dx
        png_files[f"T_{st}_ORM.png"] = pm["orm_u8"]
        png_files[f"T_{st}_E.png"] = em_u8
    elif pid == "unity_hdrp":
        png_files[f"{st}_Albedo.png"] = pm["albedo"]
        png_files[f"{st}_Normal.png"] = pm["normal_rgb"]
        png_files[f"{st}_MaskMap.png"] = unity_hdrp_maskmap
        png_files[f"{st}_HeightMap.png"] = pm["height_u8"]
        if ev.emissive is not None:
            png_files[f"{st}_Emissive.png"] = em_u8
    elif pid == "java_labpbr":
        png_files[f"{st}.png"] = pm["albedo"]
        png_files[f"{st}_n.png"] = labpbr_n_dx
        png_files[f"{st}_s.png"] = pm["labpbr_s"]
    elif pid == "bedrock_rtx":
        png_files[f"{st}.png"] = pm["albedo"]
        png_files[f"{st}_mer.png"] = pm["bedrock_mer"]
        png_files[f"{st}_mers.png"] = bedrock_mers
        png_files[f"{st}_normal.png"] = pm["normal_rgb"]
        ts_doc = {
            "format_version": "1.21.30",
            "minecraft:texture_set": {
                "color": st,
                "metalness_emissive_roughness": f"{st}_mer",
                "metalness_emissive_roughness_subsurface": f"{st}_mers",
                "normal": f"{st}_normal",
            },
        }
        text_files[f"{st}.texture_set.json"] = json.dumps(ts_doc, indent=2) + "\n"
    elif pid == "maya_materialx":
        png_files[f"{st}.png"] = pm["albedo"]
        png_files[f"{st}_normal.png"] = pm["normal_rgb"]
        png_files[f"{st}_roughness.png"] = pm["roughness_u8"]
        png_files[f"{st}_metallic.png"] = pm["metallic_u8"]
        png_files[f"{st}_height.png"] = pm["height_u8"]
        png_files[f"{st}_ao.png"] = pm["ao_u8"]
        has_em = bool(ev.emissive is not None and float(np.max(ev.emissive[..., 3])) > 0.05)
        if has_em:
            png_files[f"{st}_emissive.png"] = em_u8
        text_files[f"{st}.mtlx"] = export_materialx_xml(
            st,
            shader_model=shader_model,
            has_emissive=has_em,
            coat_weight=coat_weight,
        )
    elif pid == "blockbench_baked":
        baked = bake_pbr_to_albedo(pm, preset=lighting_preset, emissive_rgba=ev.emissive)
        if baked.dtype != np.uint8:
            baked = rgba_float_to_u8(baked)
        if b_px > 0 and np.any(seed_m) and not np.all(seed_m):
            baked = mip_safe_dilate_rgba(baked, valid_mask=seed_m, bleed_px=b_px, preserve_alpha=True)
            if baked.dtype != np.uint8:
                baked = rgba_float_to_u8(baked)
        png_files[f"{st}_baked.png"] = baked
        if bbmodel_doc is not None:
            doc = copy.deepcopy(bbmodel_doc)
            buf = io.BytesIO()
            Image.fromarray(baked, mode="RGBA").save(buf, format="PNG", optimize=True)
            b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            h_b, w_b = baked.shape[:2]
            doc["textures"] = [
                {
                    "name": f"{st}_baked.png",
                    "id": "0",
                    "width": w_b,
                    "height": h_b,
                    "uv_width": w_b,
                    "uv_height": h_b,
                    "internal": True,
                    "source": b64,
                }
            ]
            text_files[f"{st}_baked.bbmodel"] = json.dumps(doc, separators=(",", ":"))
    else:  # full_pbr_v2
        png_files[f"{st}.png"] = pm["albedo"]
        png_files[f"{st}_normal.png"] = pm["normal_rgb"]
        png_files[f"{st}_normal_dx.png"] = nrm_dx
        png_files[f"{st}_orm.png"] = pm["orm_u8"]
        png_files[f"{st}_maskmap.png"] = unity_hdrp_maskmap
        png_files[f"{st}_n.png"] = labpbr_n_dx
        png_files[f"{st}_s.png"] = pm["labpbr_s"]
        png_files[f"{st}_mer.png"] = pm["bedrock_mer"]
        png_files[f"{st}_height.png"] = pm["height_u8"]
        png_files[f"{st}_roughness.png"] = pm["roughness_u8"]
        png_files[f"{st}_metallic.png"] = pm["metallic_u8"]
        png_files[f"{st}_ao.png"] = pm["ao_u8"]
        if ev.emissive is not None:
            png_files[f"{st}_emissive.png"] = em_u8
        baked = bake_pbr_to_albedo(pm, preset=lighting_preset, emissive_rgba=ev.emissive)
        if baked.dtype != np.uint8:
            baked = rgba_float_to_u8(baked)
        png_files[f"{st}_baked.png"] = baked
        text_files[f"{st}.mtlx"] = export_materialx_xml(st, shader_model=shader_model, coat_weight=coat_weight)
        text_files[f"{st}.texdsl"] = export_texdsl(graph)
        text_files[f"{st}.texgraph.json"] = json.dumps(graph.to_recipe(), indent=2) + "\n"

    # Write to disk or return browser download bundle
    if out_dir:
        d = Path(out_dir).expanduser().resolve()
        d.mkdir(parents=True, exist_ok=True)
        written: dict[str, str] = {}
        for fname, arr in png_files.items():
            fp = d / fname
            Image.fromarray(arr, mode="RGBA").save(fp, format="PNG", optimize=True)
            written[fname] = str(fp)
        for fname, txt in text_files.items():
            fp = d / fname
            fp.write_text(txt, encoding="utf-8")
            written[fname] = str(fp)
        return {
            "ok": True,
            "mode": "disk",
            "preset_id": pid,
            "stem": st,
            "bleed_px": b_px,
            "out_dir": str(d),
            "files": written,
            "file_names": sorted(written.keys()),
        }

    embedded: dict[str, str] = {}
    for fname, arr in png_files.items():
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG", optimize=True)
        embedded[fname] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    bundle_manifest = {
        "schema": "synccraft.engine_preset/1.0",
        "preset_id": pid,
        "stem": st,
        "bleed_px": b_px,
        "text_files": text_files,
        "embedded_pngs": embedded,
    }
    return {
        "ok": True,
        "mode": "download",
        "preset_id": pid,
        "stem": st,
        "bleed_px": b_px,
        "file_names": sorted(list(png_files.keys()) + list(text_files.keys())),
        "text_files": text_files,
        "embedded_pngs": embedded,
        "bundle_filename": f"{st}.{pid}.synccraft.json",
        "bundle_json": json.dumps(bundle_manifest, indent=2) + "\n",
    }
