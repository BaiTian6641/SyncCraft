"""Pixel-Composer-Parity Transform, Distort, Filter, SDF, Normal-Blend & Color Grading Operations.

All pure vectorized NumPy (zero ML / SciPy bloat):
- Color & Grading: ``curves_remap``, ``color_balance``, ``oklch_adjust``, ``posterize_soft``
- Transform & Distort: ``tile_transform``, ``polar_warp``, ``twirl_distort``, ``wave_distort``
- Filters, SDF & PBR Helpers: ``gaussian_blur``, ``directional_blur``, ``unsharp_mask``,
  ``sobel_edges``, ``sdf_from_mask``, ``morph_op``, ``normal_blend_rnm``, ``ao_from_height``, ``bloom_glow``
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .generators_gradients import oklab_to_srgb
from .generators_noise import uv_grid
from .pixel_art_ops import srgb_to_oklab


def _sample_bilinear(arr: np.ndarray, x: np.ndarray, y: np.ndarray, *, tileable: bool = False) -> np.ndarray:
    H, W = arr.shape[:2]
    if tileable:
        x0 = np.floor(x).astype(np.int64) % W
        y0 = np.floor(y).astype(np.int64) % H
        x1 = (x0 + 1) % W
        y1 = (y0 + 1) % H
    else:
        x_c = np.clip(x, 0.0, W - 1.0001)
        y_c = np.clip(y, 0.0, H - 1.0001)
        x0 = np.floor(x_c).astype(np.int64)
        y0 = np.floor(y_c).astype(np.int64)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        x, y = x_c, y_c
    fx = (x - np.floor(x)).astype(np.float32)
    fy = (y - np.floor(y)).astype(np.float32)
    if arr.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    top = arr[y0, x0] * (1.0 - fx) + arr[y0, x1] * fx
    bot = arr[y1, x0] * (1.0 - fx) + arr[y1, x1] * fx
    return (top * (1.0 - fy) + bot * fy).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. Color & Studio Grading Nodes
# ---------------------------------------------------------------------------


def curves_remap(field: np.ndarray, points: Sequence[tuple[float, float] | list[float]]) -> np.ndarray:
    """Piecewise smooth tone curve remap (Photoshop / Pixel Composer `Curve` node)."""
    arr = np.asarray(field, dtype=np.float32)
    pts = sorted([(float(p[0]), float(p[1])) for p in points], key=lambda p: p[0])
    if len(pts) < 2:
        return arr
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:
        out = arr.copy()
        out[..., :3] = np.interp(np.clip(arr[..., :3], 0.0, 1.0), xs, ys).astype(np.float32)
        return np.clip(out, 0.0, 1.0)
    return np.clip(np.interp(np.clip(arr, 0.0, 1.0), xs, ys), 0.0, 1.0).astype(np.float32)


def color_balance(
    rgba: np.ndarray,
    *,
    shadows: tuple[float, float, float] = (0.0, 0.0, 0.0),
    midtones: tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlights: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Cohesive Shadows / Midtones / Highlights studio color balance (unifies atmospheric mood like Image 4!)."""
    img = np.asarray(rgba, dtype=np.float32).copy()
    rgb = img[..., :3]
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
    w_sh = np.clip(1.0 - lum * 2.0, 0.0, 1.0) ** 1.5
    w_hi = np.clip(lum * 2.0 - 1.0, 0.0, 1.0) ** 1.5
    w_mid = np.clip(1.0 - np.abs(lum - 0.5) * 2.0, 0.0, 1.0)
    delta = (
        w_sh * np.array(shadows, dtype=np.float32)[None, None, :]
        + w_mid * np.array(midtones, dtype=np.float32)[None, None, :]
        + w_hi * np.array(highlights, dtype=np.float32)[None, None, :]
    )
    img[..., :3] = np.clip(rgb + delta, 0.0, 1.0)
    return img.astype(np.float32)


def oklch_adjust(
    rgba: np.ndarray,
    *,
    hue_shift_deg: float = 0.0,
    chroma_scale: float = 1.0,
    lightness_offset: float = 0.0,
) -> np.ndarray:
    """Perceptual OKLCH Hue / Chroma / Lightness adjustment node."""
    img = np.asarray(rgba, dtype=np.float32)
    alpha = img[..., 3:4] if img.shape[2] == 4 else np.ones((*img.shape[:2], 1), dtype=np.float32)
    lab = srgb_to_oklab(img[..., :3]).astype(np.float64)
    L = np.clip(lab[..., 0] + float(lightness_offset), 0.0, 1.0)
    C = np.clip(np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2) * float(chroma_scale), 0.0, 0.36)
    H = np.arctan2(lab[..., 2], lab[..., 1]) + math.radians(float(hue_shift_deg))
    out_rgb = oklab_to_srgb(np.stack((L, C * np.cos(H), C * np.sin(H)), axis=-1))
    return np.concatenate((out_rgb, alpha), axis=-1).astype(np.float32)


def posterize_soft(rgba: np.ndarray, *, steps: int = 6, blend: float = 0.55) -> np.ndarray:
    """Smooth-to-pixel hybrid posterization (``blend=0.0`` pure smooth, ``blend=1.0`` crisp stepped bands)."""
    img = np.asarray(rgba, dtype=np.float32)
    n = max(2, int(steps))
    b = float(np.clip(blend, 0.0, 1.0))
    if img.ndim == 3 and img.shape[2] == 4:
        q = np.rint(img[..., :3] * (n - 1)) / float(n - 1)
        out = img.copy()
        out[..., :3] = img[..., :3] * (1.0 - b) + q * b
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    q = np.rint(img * (n - 1)) / float(n - 1)
    return np.clip(img * (1.0 - b) + q * b, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Transform & Distortion Nodes
# ---------------------------------------------------------------------------


def tile_transform(
    src: np.ndarray,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotate_deg: float = 0.0,
    tileable: bool = True,
) -> np.ndarray:
    """2D Affine scale / offset / rotation transform node."""
    H, W = src.shape[:2]
    u, v = uv_grid(H, W)
    du = (u - 0.5) / max(1e-4, float(scale_x))
    dv = (v - 0.5) / max(1e-4, float(scale_y))
    rad = math.radians(-float(rotate_deg))
    ca, sa = math.cos(rad), math.sin(rad)
    su = du * ca - dv * sa + 0.5 - float(offset_x)
    sv = du * sa + dv * ca + 0.5 - float(offset_y)
    return _sample_bilinear(src, su * W - 0.5, sv * H - 0.5, tileable=tileable)


def polar_warp(src: np.ndarray, *, to_polar: bool = True, tileable: bool = True) -> np.ndarray:
    """Convert Cartesian <-> Polar coordinates (ideal for circular portals, shields, runes & eyes!)."""
    H, W = src.shape[:2]
    u, v = uv_grid(H, W)
    if to_polar:
        dx, dy = (u - 0.5) * 2.0, (v - 0.5) * 2.0
        ang = (np.arctan2(dy, dx) / (2.0 * math.pi)) % 1.0
        rad = np.clip(np.sqrt(dx * dx + dy * dy), 0.0, 1.0)
        return _sample_bilinear(src, ang * W - 0.5, rad * H - 0.5, tileable=tileable)
    ang = u * 2.0 * math.pi
    r = v * 0.5
    su = 0.5 + r * np.cos(ang)
    sv = 0.5 + r * np.sin(ang)
    return _sample_bilinear(src, su * W - 0.5, sv * H - 0.5, tileable=tileable)


def twirl_distort(
    src: np.ndarray,
    *,
    strength_deg: float = 90.0,
    radius: float = 0.5,
    cx: float = 0.5,
    cy: float = 0.5,
    tileable: bool = False,
) -> np.ndarray:
    """Smooth vortex twirl distortion node."""
    H, W = src.shape[:2]
    u, v = uv_grid(H, W)
    dx, dy = u - float(cx), v - float(cy)
    dist = np.sqrt(dx * dx + dy * dy)
    fac = np.clip(1.0 - dist / max(1e-4, float(radius)), 0.0, 1.0)
    ang = math.radians(float(strength_deg)) * fac * fac
    ca, sa = np.cos(ang), np.sin(ang)
    su = float(cx) + dx * ca - dy * sa
    sv = float(cy) + dx * sa + dy * ca
    return _sample_bilinear(src, su * W - 0.5, sv * H - 0.5, tileable=tileable)


def wave_distort(
    src: np.ndarray,
    *,
    amp_x: float = 0.04,
    amp_y: float = 0.0,
    freq_x: float = 3.0,
    freq_y: float = 3.0,
    phase: float = 0.0,
    tileable: bool = True,
) -> np.ndarray:
    """Seamless sinusoidal wave displacement (supports animated water/plasma/cloth ripple loops!)."""
    H, W = src.shape[:2]
    u, v = uv_grid(H, W)
    ph = float(phase) * 2.0 * math.pi
    su = u + float(amp_x) * np.sin(2.0 * math.pi * float(freq_y) * v + ph)
    sv = v + float(amp_y) * np.sin(2.0 * math.pi * float(freq_x) * u + ph)
    return _sample_bilinear(src, su * W - 0.5, sv * H - 0.5, tileable=tileable)


# ---------------------------------------------------------------------------
# 3. Separable Blur, Sharpen, Edge, SDF, Morphology, Normal-Blend & Height AO
# ---------------------------------------------------------------------------


def gaussian_blur(src: np.ndarray, *, radius: int = 2, sigma: float = 0.0, tileable: bool = False) -> np.ndarray:
    """Fast separable 2D Gaussian blur for smooth glow, shadows & soft gradients."""
    arr = np.asarray(src, dtype=np.float32)
    r = max(1, int(radius))
    sig = float(sigma) if float(sigma) > 1e-4 else max(0.6, r * 0.5)
    ks = np.arange(-r, r + 1, dtype=np.float32)
    kern = np.exp(-0.5 * (ks / sig) ** 2)
    kern /= float(kern.sum())
    mode = "wrap" if tileable else "edge"

    def _conv1d(a: np.ndarray, axis: int) -> np.ndarray:
        pw = [(0, 0)] * a.ndim
        pw[axis] = (r, r)
        p = np.pad(a, pw, mode=mode)
        out = np.zeros_like(a)
        for i, w_k in enumerate(kern):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += p[tuple(sl)] * w_k
        return out

    return _conv1d(_conv1d(arr, 1), 0).astype(np.float32)


def directional_blur(src: np.ndarray, *, angle_deg: float = 0.0, length: int = 4, tileable: bool = False) -> np.ndarray:
    """Motion / painterly directional blur along any angle."""
    arr = np.asarray(src, dtype=np.float32)
    H, W = arr.shape[:2]
    n = max(2, int(length))
    rad = math.radians(float(angle_deg))
    dx, dy = math.cos(rad), math.sin(rad)
    u, v = uv_grid(H, W)
    acc = np.zeros_like(arr)
    for k in range(n):
        t = (k / float(n - 1) - 0.5) * float(n)
        acc += _sample_bilinear(arr, u * W - 0.5 + dx * t, v * H - 0.5 + dy * t, tileable=tileable)
    return (acc / float(n)).astype(np.float32)


def unsharp_mask(src: np.ndarray, *, radius: int = 1, amount: float = 0.65, tileable: bool = False) -> np.ndarray:
    """Crisp pixel-art unsharp mask edge enhancement."""
    arr = np.asarray(src, dtype=np.float32)
    blur = gaussian_blur(arr, radius=radius, tileable=tileable)
    if arr.ndim == 3 and arr.shape[2] == 4:
        out = arr.copy()
        out[..., :3] = np.clip(arr[..., :3] + (arr[..., :3] - blur[..., :3]) * float(amount), 0.0, 1.0)
        return out
    return np.clip(arr + (arr - blur) * float(amount), 0.0, 1.0).astype(np.float32)


def sobel_edges(src: np.ndarray, *, strength: float = 1.5, tileable: bool = False) -> np.ndarray:
    """Crisp Sobel edge magnitude detector [0, 1]."""
    arr = np.asarray(src, dtype=np.float32)
    if arr.ndim == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    p = np.pad(arr, 1, mode="wrap" if tileable else "edge")
    gx = (p[:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2.0 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.clip(np.sqrt(gx * gx + gy * gy) * 0.5 * float(strength), 0.0, 1.0).astype(np.float32)


def sdf_from_mask(mask: np.ndarray, *, max_dist_px: float = 12.0, invert: bool = False) -> np.ndarray:
    """Fast vectorized 2-pass Chamfer Signed Distance Field [0, 1] from any binary/soft mask.

    Powers silky inner bevels, soft bone-contact shadows, and smooth feathered halos!
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    inside = (m <= 0.5) if invert else (m > 0.5)
    H, W = inside.shape
    big = float(max_dist_px) * 2.0 + 10.0
    d = np.where(inside, 0.0, big).astype(np.float32)
    # Vectorized multi-step Chamfer propagation (exact up to max_dist_px)
    steps = max(2, int(math.ceil(max_dist_px)))
    for _ in range(steps):
        p = np.pad(d, 1, mode="edge")
        d = np.minimum.reduce([
            d,
            p[:-2, 1:-1] + 1.0,
            p[2:, 1:-1] + 1.0,
            p[1:-1, :-2] + 1.0,
            p[1:-1, 2:] + 1.0,
            p[:-2, :-2] + 1.4142,
            p[:-2, 2:] + 1.4142,
            p[2:, :-2] + 1.4142,
            p[2:, 2:] + 1.4142,
        ])
    return np.clip(1.0 - d / max(1.0, float(max_dist_px)), 0.0, 1.0).astype(np.float32)


def morph_op(src: np.ndarray, *, op: str = "dilate", radius: int = 1, tileable: bool = False) -> np.ndarray:
    """Morphological dilate / erode / open / close filter on scalar or RGBA."""
    arr = np.asarray(src, dtype=np.float32)
    r = max(1, int(radius))
    mode = "wrap" if tileable else "edge"

    def _step(a: np.ndarray, is_max: bool) -> np.ndarray:
        pw = ((1, 1), (1, 1), (0, 0)) if a.ndim == 3 else ((1, 1), (1, 1))
        p = np.pad(a, pw, mode=mode)
        fn = np.maximum.reduce if is_max else np.minimum.reduce
        return fn([
            p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:],
            p[:-2, :-2], p[:-2, 2:], p[2:, :-2], p[2:, 2:], a,
        ])

    out = arr
    if op == "erode":
        for _ in range(r):
            out = _step(out, False)
    elif op == "open":
        for _ in range(r):
            out = _step(out, False)
        for _ in range(r):
            out = _step(out, True)
    elif op == "close":
        for _ in range(r):
            out = _step(out, True)
        for _ in range(r):
            out = _step(out, False)
    else:  # dilate
        for _ in range(r):
            out = _step(out, True)
    return out.astype(np.float32)


def normal_blend_rnm(base_nrm: np.ndarray, detail_nrm: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    """Reoriented Normal Mapping (RNM) — blends macro + micro tangent-space normal maps accurately!"""
    t = np.asarray(base_nrm, dtype=np.float32)[..., :3] * 2.0 - 1.0
    u = np.asarray(detail_nrm, dtype=np.float32)[..., :3] * 2.0 - 1.0
    s = float(strength)
    u = np.stack((u[..., 0] * s, u[..., 1] * s, u[..., 2]), axis=-1)
    u = u / np.sqrt(np.sum(u * u, axis=-1, keepdims=True) + 1e-12)
    t_mod = np.stack((t[..., 0], t[..., 1], t[..., 2] + 1.0), axis=-1)
    u_mod = np.stack((-u[..., 0], -u[..., 1], u[..., 2]), axis=-1)
    dot = np.sum(t_mod * u_mod, axis=-1, keepdims=True)
    r = t_mod * dot / np.maximum(1e-6, t_mod[..., 2:3]) - u_mod
    r = r / np.sqrt(np.sum(r * r, axis=-1, keepdims=True) + 1e-12)
    return np.clip(r * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)


def ao_from_height(height_field: np.ndarray, *, radius: int = 3, strength: float = 1.0, tileable: bool = False) -> np.ndarray:
    """Fast 8-direction Horizon-Based Ambient Occlusion (HBAO) synthesized purely from a 2D heightfield."""
    h = np.asarray(height_field, dtype=np.float32)
    if h.ndim == 3:
        h = h[..., 0]
    H, W = h.shape
    r = max(1, int(radius))
    mode = "wrap" if tileable else "edge"
    p = np.pad(h, r, mode=mode)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    occ = np.zeros_like(h)
    for dx, dy in dirs:
        max_slope = np.zeros_like(h)
        step_d = math.hypot(dx, dy)
        for s in range(1, r + 1):
            nb = p[r + dy * s : r + dy * s + H, r + dx * s : r + dx * s + W]
            slope = np.maximum(0.0, (nb - h) / (s * step_d * 0.35))
            max_slope = np.maximum(max_slope, slope)
        occ += max_slope / (1.0 + max_slope)
    ao = 1.0 - (occ / 8.0) * float(strength) * 1.45
    return np.clip(ao, 0.15, 1.0).astype(np.float32)


def bloom_glow(rgba: np.ndarray, *, threshold: float = 0.72, radius: int = 3, intensity: float = 0.55, tileable: bool = False) -> np.ndarray:
    """Multi-radius crisp additive studio bloom glow for luminous crystals, runes & eyes."""
    img = np.asarray(rgba, dtype=np.float32).copy()
    rgb = img[..., :3]
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    bright = rgb * np.clip((lum - float(threshold)) / max(1e-4, 1.0 - float(threshold)), 0.0, 1.0)[..., None]
    b1 = gaussian_blur(bright, radius=max(1, int(radius)), tileable=tileable)
    b2 = gaussian_blur(bright, radius=max(2, int(radius) * 2), tileable=tileable)
    img[..., :3] = np.clip(rgb + (0.65 * b1 + 0.35 * b2) * float(intensity), 0.0, 1.0)
    return img.astype(np.float32)
