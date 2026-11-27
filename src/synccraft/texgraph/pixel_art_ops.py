"""Expert pixel-art shading, OKLCH hue-shifted ramps, OKLab quantization, Selout, blending & flipbook export."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .generators_noise import bayer_matrix


# ---------------------------------------------------------------------------
# Perceptual OKLCH <-> OKLab <-> Linear sRGB <-> sRGB [0..1] Color Math
# ---------------------------------------------------------------------------


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(lin, dtype=np.float32), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1.0 / 2.4) - 0.055).astype(np.float32)


def oklch_to_srgb_float(lch: np.ndarray) -> np.ndarray:
    """Convert OKLCH (..., 3) -> sRGB float32 in [0, 1] with smooth chroma clamping."""
    arr = np.asarray(lch, dtype=np.float64)
    L = np.clip(arr[..., 0], 0.0, 1.0)
    C = np.maximum(0.0, arr[..., 1])
    H = np.deg2rad(arr[..., 2])
    a = C * np.cos(H)
    b = C * np.sin(H)
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    lin = np.stack((r, g, bl), axis=-1)
    return linear_to_srgb(lin)


def srgb_to_oklab(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB float32 [0, 1] (..., 3) -> perceptual OKLab (..., 3)."""
    lin = srgb_to_linear(srgb).astype(np.float64)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l = np.cbrt(np.maximum(0.0, 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b))
    m = np.cbrt(np.maximum(0.0, 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b))
    s = np.cbrt(np.maximum(0.0, 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b))
    L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return np.stack((L, A, B), axis=-1).astype(np.float32)


def hex_to_srgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def build_hue_shifted_palette(
    base_lch: tuple[float, float, float],
    *,
    steps: int = 6,
    l_dark_delta: float = -0.26,
    l_light_delta: float = 0.24,
    hue_shift: tuple[float, float] = (-22.0, 18.0),
    chroma_shadow_boost: float = 0.24,
    chroma_highlight_taper: float = 0.16,
) -> np.ndarray:
    """Build an expert pixel-art hue-shifted OKLCH ramp (N, 3) in sRGB [0, 1].

    Shadows deepen with richer chroma and cool hue shift; highlights brighten cleanly
    with warm hue shift and slight chroma taper so surfaces NEVER look muddy or plastic!
    """
    n = max(2, int(steps))
    L0, C0, H0 = map(float, base_lch)
    sh_deg, hi_deg = float(hue_shift[0]), float(hue_shift[1])
    t = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    L = np.where(t < 0.0, L0 + (-t) * l_dark_delta, L0 + t * l_light_delta)
    L = np.clip(L, 0.05, 0.97)
    C = np.where(
        t < 0.0,
        C0 * (1.0 + (-t) * chroma_shadow_boost * (1.0 - 0.4 * (-t))),
        C0 * (1.0 - t * chroma_highlight_taper),
    )
    C = np.clip(C, 0.0, 0.32)
    H = (H0 + np.where(t < 0.0, (-t) * sh_deg, t * hi_deg)) % 360.0
    return oklch_to_srgb_float(np.stack((L, C, H), axis=-1))


# ---------------------------------------------------------------------------
# Shading, Hue-Shifted Ramp Mapping, Quantization, Bevel & Selout Nodes
# ---------------------------------------------------------------------------


def levels_remap(
    field: np.ndarray,
    *,
    in_min: float = 0.0,
    in_max: float = 1.0,
    gamma: float = 1.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
    invert: bool = False,
) -> np.ndarray:
    """Photoshop/Pixel-Composer style Levels / Gamma / Invert remap."""
    f = np.asarray(field, dtype=np.float32)
    norm = np.clip((f - float(in_min)) / max(1e-5, float(in_max) - float(in_min)), 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    if abs(float(gamma) - 1.0) > 1e-4:
        norm = np.power(norm, 1.0 / max(1e-4, float(gamma)))
    return (float(out_min) + norm * (float(out_max) - float(out_min))).astype(np.float32)


def directional_shade(
    height_field: np.ndarray,
    *,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    edge_bevel: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    light_dir: tuple[float, float, float] = (0.38, 0.84, 0.38),
    bump_strength: float = 1.25,
    ao_strength: float = 0.55,
    bevel_strength: float = 0.22,
    cavity_strength: float = 0.45,
    rim_strength: float = 0.14,
    spec_strength: float = 0.18,
    spec_power: float = 14.0,
    tileable: bool = False,
) -> np.ndarray:
    """Combine height relief, 3D normals, baked 3D AO, exposed edge bevels & joint cavities -> shade [0, 1]."""
    h = np.asarray(height_field, dtype=np.float32)
    H, W = h.shape[:2]

    # Tangent-space micro-normal from Sobel height gradient
    mode = "wrap" if tileable else "edge"
    p = np.pad(h, 1, mode=mode)
    gx = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
    gy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
    bs = float(bump_strength)
    nx = -gx * bs
    ny = -gy * bs
    nz = np.ones_like(h)
    inv_l = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-12)
    nx, ny, nz = nx * inv_l, ny * inv_l, nz * inv_l

    ld = np.asarray(light_dir, dtype=np.float32)
    ld = ld / max(1e-6, float(np.linalg.norm(ld)))

    # Micro-relief directional lighting (top-left studio key light in UV space)
    key_2d = np.array([-0.55, -0.68, 0.48], dtype=np.float32)
    key_2d /= float(np.linalg.norm(key_2d))
    ndotl_2d = np.clip(nx * key_2d[0] + ny * key_2d[1] + nz * key_2d[2], 0.0, 1.0)

    if world_normal is not None:
        wn = np.asarray(world_normal, dtype=np.float32)
        ndotl_3d = np.clip(wn[..., 0] * ld[0] + wn[..., 1] * ld[1] + wn[..., 2] * ld[2], 0.0, 1.0)
        # Hemisphere sky bounce so undersides shift gracefully without turning pitch-black
        hemi = 0.5 + 0.5 * wn[..., 1]
        base_lit = 0.18 + 0.42 * ndotl_3d + 0.22 * hemi + 0.26 * (ndotl_2d - 0.45)
        # Rim glint on upward/grazing edges
        rim = np.power(np.clip(1.0 - np.abs(wn[..., 2]), 0.0, 1.0), 2.5) * np.clip(wn[..., 1] + 0.4, 0.0, 1.0) * float(rim_strength)
    else:
        base_lit = 0.22 + 0.72 * ndotl_2d + 0.14 * (h - 0.5)
        rim = np.zeros_like(h)

    # Specular crisp highlight along micro-ridges
    half_v = key_2d + np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_v /= float(np.linalg.norm(half_v))
    spec = np.power(np.clip(nx * half_v[0] + ny * half_v[1] + nz * half_v[2], 0.0, 1.0), float(spec_power)) * float(spec_strength)

    shade = base_lit + rim + spec

    if baked_ao is not None:
        ao = np.clip(np.asarray(baked_ao, dtype=np.float32), 0.0, 1.0)
        ao_fac = (1.0 - float(ao_strength)) + float(ao_strength) * ao
        shade = shade * ao_fac

    if joint_cavity is not None:
        cav = np.clip(np.asarray(joint_cavity, dtype=np.float32), 0.0, 1.0)
        shade = shade * (1.0 - float(cavity_strength) * cav)

    if edge_bevel is not None:
        bev = np.clip(np.asarray(edge_bevel, dtype=np.float32), 0.0, 1.0)
        shade = shade + float(bevel_strength) * bev

    return np.clip(shade, 0.0, 1.0).astype(np.float32)


def pixel_bevel_emboss(
    height_field: np.ndarray,
    *,
    strength: float = 0.35,
    light_angle_deg: float = 135.0,
    tileable: bool = False,
) -> np.ndarray:
    """Crisp 1-pixel directional pixel-art bevel/emboss delta centered around 0.5."""
    h = np.asarray(height_field, dtype=np.float32)
    rad = math.radians(float(light_angle_deg))
    dx = int(round(math.cos(rad)))
    dy = int(round(-math.sin(rad)))
    mode = "wrap" if tileable else "edge"
    p = np.pad(h, 1, mode=mode)
    shifted = p[1 + dy : 1 + dy + h.shape[0], 1 + dx : 1 + dx + h.shape[1]]
    diff = (h - shifted) * float(strength)
    return np.clip(0.5 + diff, 0.0, 1.0).astype(np.float32)


def hue_shifted_ramp(
    field: np.ndarray,
    *,
    base_lch: tuple[float, float, float] = (0.45, 0.08, 250.0),
    steps: int = 6,
    l_dark_delta: float = -0.26,
    l_light_delta: float = 0.24,
    hue_shift: tuple[float, float] = (-22.0, 18.0),
    colors: list[str] | None = None,
    quantize: bool = True,
    dither: str = "none",
    dither_strength: float = 0.16,
    alpha_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map scalar shade/height field (H, W) -> RGBA float32 [0, 1] using a hue-shifted OKLCH ramp (or custom hex ramp)."""
    f = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    H, W = f.shape[:2]

    if colors:
        ramp = np.array([hex_to_srgb(c) for c in colors], dtype=np.float32)
    else:
        ramp = build_hue_shifted_palette(
            base_lch,
            steps=steps,
            l_dark_delta=l_dark_delta,
            l_light_delta=l_light_delta,
            hue_shift=hue_shift,
        )
    n = ramp.shape[0]

    val = f
    if dither in ("bayer2", "bayer4", "bayer8"):
        sz = {"bayer2": 2, "bayer4": 4, "bayer8": 8}[dither]
        bm = bayer_matrix(H, W, size=sz) - 0.5
        val = np.clip(val + bm * (float(dither_strength) / max(1.0, float(n - 1))), 0.0, 1.0)
    elif dither == "clustered":
        # Diagonal 2-px stepped pixel-art clusters (avoids noisy checkerboard stipple!)
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cl = (((yy + xx // 2) % 2).astype(np.float32) - 0.5)
        val = np.clip(val + cl * (float(dither_strength) / max(1.0, float(n - 1))), 0.0, 1.0)

    pos = val * float(n - 1)
    if quantize:
        idx = np.clip(np.rint(pos).astype(np.int64), 0, n - 1)
        rgb = ramp[idx]
    else:
        i0 = np.clip(np.floor(pos).astype(np.int64), 0, n - 1)
        i1 = np.clip(i0 + 1, 0, n - 1)
        frac = (pos - i0)[..., None]
        rgb = ramp[i0] * (1.0 - frac) + ramp[i1] * frac

    alpha = np.ones((H, W, 1), dtype=np.float32) if alpha_mask is None else np.clip(np.asarray(alpha_mask, dtype=np.float32), 0.0, 1.0)[..., None]
    return np.concatenate((rgb.astype(np.float32), alpha), axis=-1)


def palette_quantize(
    rgba: np.ndarray,
    *,
    palette_hex: list[str] | None = None,
    palette_rgb: np.ndarray | None = None,
    dither: str = "none",
    dither_strength: float = 0.14,
) -> np.ndarray:
    """Quantize RGBA image [0, 1] to an artist palette via perceptual OKLab nearest-neighbor."""
    img = np.asarray(rgba, dtype=np.float32)
    H, W = img.shape[:2]
    rgb = img[..., :3].copy()
    alpha = img[..., 3:4] if img.shape[2] == 4 else np.ones((H, W, 1), dtype=np.float32)

    if palette_hex:
        pal = np.array([hex_to_srgb(c) for c in palette_hex], dtype=np.float32)
    elif palette_rgb is not None:
        pal = np.asarray(palette_rgb, dtype=np.float32)
    else:
        raise ValueError("palette_quantize requires palette_hex or palette_rgb")

    if dither in ("bayer2", "bayer4", "bayer8"):
        sz = {"bayer2": 2, "bayer4": 4, "bayer8": 8}[dither]
        bm = (bayer_matrix(H, W, size=sz) - 0.5) * float(dither_strength)
        rgb = np.clip(rgb + bm[..., None], 0.0, 1.0)

    pal_lab = srgb_to_oklab(pal)  # (K, 3)

    if dither == "floyd":
        # Per-pixel Floyd-Steinberg error diffusion in sRGB matched via OKLab
        out_rgb = rgb.copy()
        st = float(dither_strength) * 2.5
        for y in range(H):
            row_lab = srgb_to_oklab(out_rgb[y])
            for x in range(W):
                cur_c = out_rgb[y, x]
                lab = srgb_to_oklab(cur_c[None, :])[0]
                best = int(np.argmin(np.sum((pal_lab - lab[None, :]) ** 2, axis=-1)))
                q_c = pal[best]
                err = (cur_c - q_c) * st
                out_rgb[y, x] = q_c
                if x + 1 < W:
                    out_rgb[y, x + 1] = np.clip(out_rgb[y, x + 1] + err * (7.0 / 16.0), 0.0, 1.0)
                if y + 1 < H:
                    if x > 0:
                        out_rgb[y + 1, x - 1] = np.clip(out_rgb[y + 1, x - 1] + err * (3.0 / 16.0), 0.0, 1.0)
                    out_rgb[y + 1, x] = np.clip(out_rgb[y + 1, x] + err * (5.0 / 16.0), 0.0, 1.0)
                    if x + 1 < W:
                        out_rgb[y + 1, x + 1] = np.clip(out_rgb[y + 1, x + 1] + err * (1.0 / 16.0), 0.0, 1.0)
        return np.concatenate((out_rgb, alpha), axis=-1).astype(np.float32)

    px_lab = srgb_to_oklab(rgb)  # (H, W, 3)
    diff = px_lab[..., None, :] - pal_lab[None, None, :, :]
    best = np.argmin(np.sum(diff * diff, axis=-1), axis=-1)
    q_rgb = pal[best]
    return np.concatenate((q_rgb, alpha), axis=-1).astype(np.float32)


def selout_outline(
    rgba: np.ndarray,
    *,
    darken: float = 0.52,
    cool_tint: tuple[float, float, float] = (0.04, 0.02, 0.12),
    edge_threshold: float = 0.22,
    silhouette: bool = True,
    interior: bool = True,
    tileable: bool = False,
) -> np.ndarray:
    """Selective Outlining (Selout) — classic pro pixel-art tinted edge definition.

    Adds rich dark hue-shifted outlines along alpha silhouettes and sharp internal
    tonal boundaries without harsh flat-black borders.
    """
    img = np.asarray(rgba, dtype=np.float32).copy()
    H, W = img.shape[:2]
    rgb = img[..., :3]
    alpha = img[..., 3] if img.shape[2] == 4 else np.ones((H, W), dtype=np.float32)
    mode = "wrap" if tileable else "constant"

    edge_mask = np.zeros((H, W), dtype=np.float32)

    if silhouette:
        op = (alpha > 0.5).astype(np.float32)
        p_a = np.pad(op, 1, mode=mode, constant_values=0.0)
        nb_min = np.minimum.reduce([p_a[:-2, 1:-1], p_a[2:, 1:-1], p_a[1:-1, :-2], p_a[1:-1, 2:]])
        sil_edge = (op > 0.5) & (nb_min < 0.5)
        edge_mask = np.where(sil_edge, 1.0, edge_mask)

    if interior:
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        p_l = np.pad(lum, 1, mode="wrap" if tileable else "edge")
        grad = np.maximum(np.abs(p_l[1:-1, 2:] - lum), np.abs(p_l[2:, 1:-1] - lum))
        int_edge = (grad > float(edge_threshold)) & (alpha > 0.5)
        edge_mask = np.where(int_edge, np.maximum(edge_mask, 0.72), edge_mask)

    tint = np.array(cool_tint, dtype=np.float32)
    dark_col = np.clip(rgb * (1.0 - float(darken)) + tint * float(darken) * 0.45, 0.0, 1.0)
    out_rgb = rgb * (1.0 - edge_mask[..., None]) + dark_col * edge_mask[..., None]
    if img.shape[2] == 4:
        return np.concatenate((out_rgb, alpha[..., None]), axis=-1).astype(np.float32)
    return out_rgb.astype(np.float32)


def blend_layers(
    base: np.ndarray,
    top: np.ndarray,
    *,
    mode: str = "normal",
    opacity: float = 1.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Studio compositing blend node supporting both scalar (H,W) and RGBA (H,W,4) layers."""
    a = np.asarray(base, dtype=np.float32)
    b = np.asarray(top, dtype=np.float32)
    if a.ndim == 2 and b.ndim == 3:
        a = np.concatenate((np.repeat(a[..., None], 3, axis=-1), np.ones((*a.shape, 1), dtype=np.float32)), axis=-1)
    elif b.ndim == 2 and a.ndim == 3:
        b = np.concatenate((np.repeat(b[..., None], 3, axis=-1), np.ones((*b.shape, 1), dtype=np.float32)), axis=-1)

    is_rgba = (a.ndim == 3 and a.shape[2] == 4)
    a_c = a[..., :3] if is_rgba else a
    b_c = b[..., :3] if is_rgba else b

    if mode == "multiply":
        res = a_c * b_c
    elif mode == "screen":
        res = 1.0 - (1.0 - a_c) * (1.0 - b_c)
    elif mode == "overlay":
        res = np.where(a_c < 0.5, 2.0 * a_c * b_c, 1.0 - 2.0 * (1.0 - a_c) * (1.0 - b_c))
    elif mode == "soft_light":
        res = (1.0 - 2.0 * b_c) * a_c * a_c + 2.0 * b_c * a_c
    elif mode == "hard_light":
        res = np.where(b_c < 0.5, 2.0 * a_c * b_c, 1.0 - 2.0 * (1.0 - a_c) * (1.0 - b_c))
    elif mode == "add":
        res = np.clip(a_c + b_c, 0.0, 1.0)
    elif mode == "subtract":
        res = np.clip(a_c - b_c, 0.0, 1.0)
    elif mode == "max":
        res = np.maximum(a_c, b_c)
    elif mode == "min":
        res = np.minimum(a_c, b_c)
    else:
        res = b_c

    w = np.full(a.shape[:2], float(opacity), dtype=np.float32)
    if mask is not None:
        w = w * np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if is_rgba:
        w_eff = w * b[..., 3]
        out_rgb = a_c * (1.0 - w_eff[..., None]) + res * w_eff[..., None]
        out_a = np.clip(a[..., 3] + w_eff * (1.0 - a[..., 3]), 0.0, 1.0)
        return np.concatenate((np.clip(out_rgb, 0.0, 1.0), out_a[..., None]), axis=-1).astype(np.float32)

    if a.ndim == 3:
        return np.clip(a_c * (1.0 - w[..., None]) + res * w[..., None], 0.0, 1.0).astype(np.float32)
    return np.clip(a_c * (1.0 - w) + res * w, 0.0, 1.0).astype(np.float32)


def emissive_extract(
    albedo_rgba: np.ndarray,
    emissive_mask: np.ndarray,
    *,
    emissive_color_rgba: np.ndarray | None = None,
    glow_bleed_px: int = 2,
    glow_strength: float = 0.35,
    tileable: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite emissive glow cleanly onto albedo and return paired (albedo_rgba, emissive_rgba)."""
    alb = np.asarray(albedo_rgba, dtype=np.float32).copy()
    mask = np.clip(np.asarray(emissive_mask, dtype=np.float32), 0.0, 1.0)
    H, W = alb.shape[:2]

    if emissive_color_rgba is not None:
        ec = np.asarray(emissive_color_rgba, dtype=np.float32)[..., :3]
    else:
        ec = alb[..., :3]

    # Soft multi-pixel bloom bleed onto adjacent albedo texels (never blurs crisp core!)
    bleed_rgb = np.zeros((H, W, 3), dtype=np.float32)
    r_max = max(0, int(glow_bleed_px))
    if r_max > 0 and float(mask.max()) > 0.05:
        src_rgb = ec * mask[..., None]
        mode = "wrap" if tileable else "constant"
        p_src = np.pad(src_rgb, ((r_max, r_max), (r_max, r_max), (0, 0)), mode=mode)
        for dy in range(-r_max, r_max + 1):
            for dx in range(-r_max, r_max + 1):
                if dx == 0 and dy == 0:
                    continue
                d = math.hypot(dx, dy)
                if d > r_max + 0.5:
                    continue
                w = ((1.0 - d / (r_max + 1.0)) ** 2) * float(glow_strength)
                nb = p_src[r_max + dy : r_max + dy + H, r_max + dx : r_max + dx + W]
                bleed_rgb = np.maximum(bleed_rgb, nb * w)

    out_alb = alb.copy()
    out_alb[..., :3] = np.clip(
        alb[..., :3] * (1.0 - mask[..., None]) + ec * mask[..., None] + bleed_rgb * (1.0 - mask[..., None]),
        0.0,
        1.0,
    )
    emis_out = np.zeros((H, W, 4), dtype=np.float32)
    emis_out[..., :3] = np.clip(ec * mask[..., None] + bleed_rgb * 0.45, 0.0, 1.0)
    emis_out[..., 3] = np.clip(mask + np.max(bleed_rgb, axis=-1) * 0.4, 0.0, 1.0)
    return out_alb.astype(np.float32), emis_out.astype(np.float32)


def rgba_float_to_u8(rgba: np.ndarray) -> np.ndarray:
    a = np.asarray(rgba, dtype=np.float32)
    if a.ndim == 2:
        g = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
        return np.stack((g, g, g, np.full_like(g, 255)), axis=-1)
    if a.shape[2] == 3:
        a = np.concatenate((a, np.ones((*a.shape[:2], 1), dtype=np.float32)), axis=-1)
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


def encode_png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba_float_to_u8(arr), mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def export_flipbook(
    frames_rgba: list[np.ndarray],
    *,
    out_png_path: Path | str | None = None,
    frametime: int = 2,
    interpolate: bool = False,
    out_gif_path: Path | str | None = None,
    gif_scale: int = 4,
) -> dict[str, Any]:
    """Export looping animation frames as a vertical Minecraft/Blockbench flipbook PNG + .mcmeta + GIF."""
    if not frames_rgba:
        raise ValueError("export_flipbook requires at least one frame")
    u8_frames = [rgba_float_to_u8(f) for f in frames_rgba]
    h, w = u8_frames[0].shape[:2]
    strip = np.concatenate(u8_frames, axis=0)

    png_buf = io.BytesIO()
    Image.fromarray(strip, mode="RGBA").save(png_buf, format="PNG", optimize=True)
    png_bytes = png_buf.getvalue()

    mcmeta = {
        "animation": {
            "frametime": max(1, int(frametime)),
            "interpolate": bool(interpolate),
            "frames": list(range(len(u8_frames))),
        }
    }

    if out_png_path is not None:
        p = Path(out_png_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(png_bytes)
        Path(f"{p}.mcmeta").write_text(json.dumps(mcmeta, indent=2) + "\n", encoding="utf-8")

    gif_bytes = b""
    if out_gif_path is not None:
        sc = max(1, int(gif_scale))
        pil_frames = [
            Image.fromarray(fr, mode="RGBA").resize((w * sc, h * sc), resample=Image.Resampling.NEAREST)
            for fr in u8_frames
        ]
        gbuf = io.BytesIO()
        duration_ms = max(40, int(frametime) * 50)
        pil_frames[0].save(
            gbuf,
            format="GIF",
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
        )
        gif_bytes = gbuf.getvalue()
        gp = Path(out_gif_path)
        gp.parent.mkdir(parents=True, exist_ok=True)
        gp.write_bytes(gif_bytes)

    return {
        "frame_width": w,
        "frame_height": h,
        "num_frames": len(u8_frames),
        "strip_height": h * len(u8_frames),
        "mcmeta": mcmeta,
        "png_bytes": png_bytes,
        "gif_bytes": gif_bytes,
    }


def vibrance_grade(
    rgba: np.ndarray,
    *,
    vibrance: float = 0.22,
    contrast: float = 1.06,
    lift: float = 0.015,
    shadow_cool: float = 0.05,
) -> np.ndarray:
    """Chroma-preserving perceptual vibrance & clean contrast grade (eliminates muddy/dirty midtones).

    Boosts muted midtone saturation in OKLab while protecting already-vibrant colors from
    clipping, lifts crushed shadows cleanly, and adds subtle jewel-cool shadow depth.
    """
    img = np.asarray(rgba, dtype=np.float32)
    rgb = np.clip(img[..., :3], 0.0, 1.0)
    alpha = img[..., 3:4] if img.shape[2] == 4 else np.ones((*img.shape[:2], 1), dtype=np.float32)

    lab = srgb_to_oklab(rgb).astype(np.float64)
    L = lab[..., 0]
    A = lab[..., 1]
    B = lab[..., 2]
    C = np.sqrt(A * A + B * B)
    H = np.arctan2(B, A)

    # Selective vibrance: boosts muted tones strongly, tapers smoothly for vivid colors
    vib_scale = 1.0 + float(vibrance) * np.clip(1.0 - C / 0.24, 0.15, 1.0)
    C_new = np.clip(C * vib_scale, 0.0, 0.34)
    L_new = np.clip((L - 0.5) * float(contrast) + 0.5 + float(lift), 0.04, 0.99)

    # Subtle cool indigo tint in deep shadows (never murky grey-brown)
    sh_w = np.clip((0.36 - L_new) / 0.36, 0.0, 1.0) * float(shadow_cool)
    A_new = C_new * np.cos(H) + sh_w * 0.018
    B_new = C_new * np.sin(H) - sh_w * 0.055

    l_ = (L_new + 0.3963377774 * A_new + 0.2158037573 * B_new) ** 3
    m_ = (L_new - 0.1055613458 * A_new - 0.0638541728 * B_new) ** 3
    s_ = (L_new - 0.0894841775 * A_new - 1.2914855480 * B_new) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    srgb = linear_to_srgb(np.stack((r, g, bl), axis=-1))
    return np.concatenate((srgb, alpha), axis=-1).astype(np.float32)


def paint_creature_eye(
    local_u: np.ndarray,
    local_v: np.ndarray,
    *,
    side_sign: np.ndarray | None = None,
    iris_inner: str = "#FEF08A",
    iris_mid: str = "#22D3EE",
    iris_outer: str = "#0284C7",
    limbal: str = "#1E1B4B",
    sclera: str = "#0F172A",
    pupil: str = "#070514",
    catchlight: str = "#FFFFFF",
    pupil_shape: str = "slit",
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize a crisp 7-layer studio pixel-art dragon/creature eye on local face UV [0..1]^2.

    Automatically mirrors gaze direction horizontally when ``side_sign > 0`` (+X right eye)
    versus ``side_sign < 0`` (-X left eye) so BOTH eyes stare forward with 100% symmetry!
    Layers: (1) Dark orbital rim, (2) Upper-lid brow shadow, (3) Shaded sclera corners,
    (4) Dark limbal ring, (5) Starburst radial iris + lower refraction crescent,
    (6) Razor-sharp vertical slit or round pupil, (7) Crisp wet specular catchlight.

    Returns ``(eye_rgba, eye_emissive_mask)``.
    """
    u = np.asarray(local_u, dtype=np.float32)
    v = np.asarray(local_v, dtype=np.float32)
    # West face (+u -> +Z anterior nose); East face (+u -> -Z posterior)!
    # Mirror u on right-side (+X) eyes so u_fwd=1.0 ALWAYS points anteriorly toward the snout!
    if side_sign is not None:
        u_fwd = np.where(np.asarray(side_sign, dtype=np.float32) > 0.0, 1.0 - u, u)
    else:
        u_fwd = u

    H, W = u.shape[:2]
    c_rim = np.array(hex_to_srgb(pupil), dtype=np.float32)
    c_scl = np.array(hex_to_srgb(sclera), dtype=np.float32)
    c_lim = np.array(hex_to_srgb(limbal), dtype=np.float32)
    c_out = np.array(hex_to_srgb(iris_outer), dtype=np.float32)
    c_mid = np.array(hex_to_srgb(iris_mid), dtype=np.float32)
    c_in = np.array(hex_to_srgb(iris_inner), dtype=np.float32)
    c_pup = np.array(hex_to_srgb(pupil), dtype=np.float32)
    c_cat = np.array(hex_to_srgb(catchlight), dtype=np.float32)

    # Slightly forward-shifted fierce predator gaze (u_c = 0.56 toward snout, v_c = 0.52)
    du = (u_fwd - 0.56) * 1.38
    dv = (v - 0.52) * 1.15
    rad = np.sqrt(du * du + dv * dv)

    # 1. Start with deep orbital socket / shaded sclera
    rgb = np.broadcast_to(c_scl, (H, W, 3)).copy()

    # 2. Dark limbal ring (0.34 < rad <= 0.45)
    limbal_m = (rad <= 0.46)
    rgb = np.where(limbal_m[..., None], c_lim, rgb)

    # 3. Radial starburst iris (rad <= 0.37)
    iris_m = (rad <= 0.38)
    ang = np.arctan2(dv, du)
    ray = 0.5 + 0.5 * np.cos(ang * 10.0)
    t_ir = np.clip(1.0 - rad / 0.38, 0.0, 1.0)
    # Outer -> mid -> luminous inner ring
    iris_col = np.where(
        (t_ir < 0.48)[..., None],
        c_out * (1.0 - (t_ir / 0.48)[..., None]) + c_mid * (t_ir / 0.48)[..., None],
        c_mid * (1.0 - ((t_ir - 0.48) / 0.52)[..., None]) + c_in * ((t_ir - 0.48) / 0.52)[..., None],
    )
    # Subtle starburst facet brightening + lower crescent refraction glow!
    crescent = (dv > 0.08) & iris_m & (rad > 0.14) & (rad < 0.34)
    iris_col = np.clip(iris_col * (0.88 + 0.22 * ray[..., None]) + np.where(crescent[..., None], 0.16, 0.0), 0.0, 1.0)
    rgb = np.where(iris_m[..., None], iris_col, rgb)

    # 4. Crisp dragon vertical slit (or round predator) pupil
    if pupil_shape == "round":
        pup_m = (rad <= 0.17)
    else:
        # Tapered vertical almond slit (narrower at top/bottom, bold center)
        slit_w = 0.115 * np.clip(1.0 - (dv / 0.33) ** 2, 0.0, 1.0)
        pup_m = (np.abs(du) <= slit_w) & (np.abs(dv) <= 0.31)
    rgb = np.where(pup_m[..., None], c_pup, rgb)

    # 5. Upper-lid fierce brow shadow across top 22% of eye face
    brow_sh = np.clip((0.24 - v) / 0.24, 0.0, 1.0)
    rgb = rgb * (1.0 - 0.58 * brow_sh[..., None])

    # 6. Outer 1-band orbital dark rim along top/bottom/rear edge
    rim_m = (v < 0.09) | (v > 0.91) | (u_fwd < 0.07) | (u_fwd > 0.94)
    rgb = np.where(rim_m[..., None], c_rim, rgb)

    # 7. Crisp wet white specular catchlight (upper-anterior quadrant, stays sharp on both eyes!)
    cat_main = ((u_fwd - 0.67) ** 2 + (v - 0.34) ** 2) <= 0.078 ** 2
    cat_sub = ((u_fwd - 0.42) ** 2 + (v - 0.67) ** 2) <= 0.042 ** 2
    rgb = np.where(cat_main[..., None], c_cat, rgb)
    rgb = np.where(cat_sub[..., None], np.clip(c_in * 0.5 + c_cat * 0.5, 0.0, 1.0), rgb)

    # Selective emissive mask: iris & catchlight glow vividly while pupil, limbal ring & rim stay dark!
    em_mask = np.where(iris_m & (~pup_m) & (~rim_m), 0.78 * (1.0 - 0.45 * brow_sh), 0.0)
    em_mask = np.where(cat_main, 1.0, em_mask).astype(np.float32)

    rgba = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    return rgba, em_mask


def wing_scallop_alpha(
    local_u: np.ndarray,
    local_v: np.ndarray,
    *,
    arches: int = 2,
    depth: float = 0.28,
    rim_width: float = 0.085,
    side_taper: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve sweeping Gothic bat/dragon scalloped arches into thin wing membrane planes (`up`/`down` faces).

    For Blockbench `up`/`down` faces on wing sail panels:
      - ``local_v = 1.0`` (up) or ``local_v = 0.0`` (down) corresponds to the leading/trailing Z edge.
      - Because Blockbench `up` maps `+v -> +Z` (leading spar!) and `-v -> -Z` (trailing edge!),
        we carve graceful concave parabolic scalloped arches into the trailing `-Z` edge (`v_trail`)
        while keeping the leading spar edge 100% solid and attached to the bone!

    Returns ``(hard_alpha_mask, trailing_rim_mask)`` in float32 [0, 1].
    """
    u = np.clip(np.asarray(local_u, dtype=np.float32), 0.0, 1.0)
    v = np.clip(np.asarray(local_v, dtype=np.float32), 0.0, 1.0)
    k = max(1, int(arches))
    # Phase along span u: scalloped arch peaks (pointed dragon tips) at u=0, 1/k, 2/k, ..., 1
    arch_u = (u * k) % 1.0
    # Parabolic concave arch reaching deepest cutout `depth` at mid-bay (arch_u = 0.5)
    scallop_cut = float(depth) * (1.0 - (2.0 * arch_u - 1.0) ** 2)
    # Slight tapered curve near outer tip edge (u -> 0) for swept wingtip silhouette
    tip_cut = float(side_taper) * np.clip(1.0 - u * 3.5, 0.0, 1.0)
    cut_y = scallop_cut + tip_cut

    # Distance inward from the scalloped trailing edge (v >= cut_y is solid membrane!)
    dist_in = v - cut_y
    alpha = (dist_in >= 0.0).astype(np.float32)
    rim = ((dist_in >= 0.0) & (dist_in <= float(rim_width))).astype(np.float32)
    return alpha, rim


def measure_painterly_metrics(
    albedo_u8: np.ndarray,
    mesh_ctx: Any | None = None,
) -> dict[str, float]:
    """Measure SKILL.md §10.2 studio pixel-art interior patch smoothness & deliberate relief metrics.

    Evaluates horizontal & vertical neighbor luminance steps strictly INSIDE opaque UV islands
    (excluding 1 px island boundaries and eye islands):
    - ``mean_lum_step``: ideal studio painterly Blockbench range is ``4.0 - 8.5`` (on 0-255);
      lower than ~3.0 reads as flat plastic airbrush, higher than ~11.0 reads as rock-hard noisy grit.
    - ``jump_gt25_pct``: percentage of interior neighbor pairs jumping > 25 luminance; ideal is
      ``4.0% - 14.0%`` (calm >= 60% base fields with deliberate sculpted clusters & bevels).
    - ``calm_interior_pct``: percentage of interior neighbor pairs with smooth step <= 10.
    """
    arr = np.asarray(albedo_u8)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3] > 128 if arr.shape[-1] >= 4 else np.ones(arr.shape[:2], dtype=bool)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    steps_list: list[np.ndarray] = []
    if mesh_ctx is not None and hasattr(mesh_ctx, "islands"):
        for isl in mesh_ctx.islands:
            ename = str(getattr(isl, "elem_name", "")).lower()
            if "eye_" in ename or isl.w < 6 or isl.h < 6:
                continue
            x0 = int(isl.x0) + 1
            y0 = int(isl.y0) + 1
            x1 = int(isl.x0 + isl.w) - 1
            y1 = int(isl.y0 + isl.h) - 1
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            sub_l = lum[y0:y1, x0:x1]
            sub_a = alpha[y0:y1, x0:x1]
            mh = sub_a[:, 1:] & sub_a[:, :-1]
            mv = sub_a[1:, :] & sub_a[:-1, :]
            if np.any(mh):
                steps_list.append(np.abs(sub_l[:, 1:] - sub_l[:, :-1])[mh])
            if np.any(mv):
                steps_list.append(np.abs(sub_l[1:, :] - sub_l[:-1, :])[mv])
    else:
        mh = alpha[:, 1:] & alpha[:, :-1]
        mv = alpha[1:, :] & alpha[:-1, :]
        if np.any(mh):
            steps_list.append(np.abs(lum[:, 1:] - lum[:, :-1])[mh])
        if np.any(mv):
            steps_list.append(np.abs(lum[1:, :] - lum[:-1, :])[mv])

    if not steps_list:
        return {"mean_lum_step": 0.0, "jump_gt25_pct": 0.0, "calm_interior_pct": 100.0, "pairs_sampled": 0.0}
    all_s = np.concatenate(steps_list)
    return {
        "mean_lum_step": round(float(all_s.mean()), 2),
        "jump_gt25_pct": round(100.0 * float((all_s > 25.0).mean()), 2),
        "calm_interior_pct": round(100.0 * float((all_s <= 10.0).mean()), 2),
        "pairs_sampled": float(all_s.size),
    }
