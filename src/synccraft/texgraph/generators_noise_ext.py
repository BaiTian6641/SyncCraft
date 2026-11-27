"""Extended Industry-Standard Procedural Noise Generators (`TexGraph v13`).

Brings Pixel Composer + Substance 3D Designer noise library parity in pure vectorized NumPy:
- ``simplex_noise``     : Isotropic 2D Simplex gradient noise (zero axis bias, time-loopable)
- ``white_noise``       : Deterministic hash grain / salt-and-pepper / film grain
- ``blue_noise``        : High-frequency blue-noise / Bayer void-cluster stipple mask
- ``bubble_noise``      : Pixel Composer `Noise Bubble` — circular soft/rimmed bubbles & foam
- ``crystal_noise``     : Pixel Composer `Noise Cristal` / Shard — faceted Voronoi crystal planes
- ``scratch_noise``     : Pixel Composer `Noise Scratch` — micro-scratches & directional tool marks
- ``strand_noise``      : Pixel Composer `Noise Strand` — flowing anisotropic fiber/hair strands
- ``caustic_noise``     : Pixel Composer `Caustic` — animated dual-wave optical caustics
- ``fold_marble_noise`` : Pixel Composer `Fold Noise` — turbulent folded marble & stratified veins
- ``curl_flow_noise``   : Divergence-free 2D curl flow field magnitude & wisps (time-loopable)
- ``grunge_blotch``     : Substance Designer multi-scale weathered dirt/rust/patina grunge map
- ``starfield_sparkle`` : Animated twinkling starfield / crystal glint / dust sparkle
- ``plasma_wave``       : Animated multi-harmonic iridescent plasma / chronal wave field

Every single generator strictly accepts ``seed: int`` so artists can adjust & randomize seeds live!
"""
from __future__ import annotations

import math
import numpy as np

from .generators_noise import (
    _hash2i,
    fbm_noise,
    perlin_noise,
    sample_field_bilinear,
    uv_grid,
    value_noise,
    voronoi_noise,
)


def simplex_noise(
    height: int,
    width: int,
    *,
    scale: float = 8.0,
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    loop_speed: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Isotropic 2D Simplex gradient noise in [0, 1] (periodically wrapped on torus when tileable)."""
    if tileable:
        # Blend two 60-degree offset periodic gradient lattices for isotropic simplex look + exact seamless wrap
        s = max(1, int(round(scale)))
        n1 = perlin_noise(
            height, width, scale=s, seed=seed + 11, tileable=True,
            phase=phase, loop_speed=loop_speed, offset_x=offset_x, offset_y=offset_y,
        )
        n2 = perlin_noise(
            height, width, scale=max(1, s), seed=seed + 79, tileable=True,
            phase=phase, loop_speed=-loop_speed, offset_x=offset_x + 0.37, offset_y=offset_y + 0.61,
        )
        # Rotate sampling coordinates by 45 deg using periodic lattice indices (integer-harmonic wrap)
        u, v = uv_grid(height, width)
        ang = float(phase) * float(loop_speed) * 2.0 * math.pi
        u_rot = (u + v) % 1.0
        v_rot = (u - v) % 1.0
        n3 = sample_field_bilinear(n2, u_rot, v_rot, tileable=True)
        out = 0.55 * n1 + 0.45 * n3
        if abs(ang) > 1e-6:
            out = np.clip(out + 0.04 * np.sin(2.0 * math.pi * (u * s + v * s) + ang), 0.0, 1.0)
        lo, hi = float(np.percentile(out, 2)), float(np.percentile(out, 98))
        return np.clip((out - lo) / max(1e-4, hi - lo), 0.0, 1.0).astype(np.float32)

    # Exact Gustavson 2D Simplex Noise when non-tileable
    u, v = uv_grid(height, width)
    x = u * float(scale) + float(offset_x) * float(scale)
    y = v * float(scale) + float(offset_y) * float(scale)

    F2 = 0.5 * (math.sqrt(3.0) - 1.0)
    G2 = (3.0 - math.sqrt(3.0)) / 6.0

    s_skew = (x + y) * F2
    i = np.floor(x + s_skew).astype(np.int64)
    j = np.floor(y + s_skew).astype(np.int64)
    t = (i + j) * G2
    x0 = x - (i - t)
    y0 = y - (j - t)

    i1 = (x0 > y0).astype(np.int64)
    j1 = 1 - i1

    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2

    ang_shift = float(phase) * float(loop_speed) * 2.0 * math.pi

    def _contrib(ii: np.ndarray, jj: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        t0 = 0.5 - dx * dx - dy * dy
        mask = t0 > 0.0
        t0c = np.where(mask, t0, 0.0)
        ang = _hash2i(ii, jj, seed + 313) * np.float32(2.0 * math.pi) + np.float32(ang_shift)
        g = np.cos(ang) * dx + np.sin(ang) * dy
        return (t0c ** 4) * g

    n = _contrib(i, j, x0, y0) + _contrib(i + i1, j + j1, x1, y1) + _contrib(i + 1, j + 1, x2, y2)
    return np.clip(n * 35.0 + 0.5, 0.0, 1.0).astype(np.float32)


def white_noise(
    height: int,
    width: int,
    *,
    density: float = 1.0,
    contrast: float = 1.0,
    monochrome: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Crisp per-pixel deterministic hash grain / white noise (SCALAR or RGBA when monochrome=False)."""
    yy, xx = np.mgrid[0:height, 0:width]
    r = _hash2i(xx, yy, seed + 17)
    d = float(np.clip(density, 0.02, 1.0))
    if d < 0.999:
        keep = _hash2i(xx, yy, seed + 97) < d
        r = np.where(keep, r, 0.0)
    c = float(contrast)
    r = np.clip((r - 0.5) * c + 0.5, 0.0, 1.0).astype(np.float32)
    if monochrome:
        return r
    g = np.clip((_hash2i(xx, yy, seed + 251) - 0.5) * c + 0.5, 0.0, 1.0).astype(np.float32)
    b = np.clip((_hash2i(xx, yy, seed + 509) - 0.5) * c + 0.5, 0.0, 1.0).astype(np.float32)
    if d < 0.999:
        keep = _hash2i(xx, yy, seed + 97) < d
        g = np.where(keep, g, 0.0)
        b = np.where(keep, b, 0.0)
    a = np.ones((height, width), dtype=np.float32)
    return np.stack((r, g, b, a), axis=-1)


def blue_noise(
    height: int,
    width: int,
    *,
    scale: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """High-frequency blue-noise mask (R2 quasi-random + 8x8 Bayer harmonic) ideal for pixel dithering."""
    bayer8 = np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    ) / 64.0
    sc = max(0.25, float(scale))
    yy, xx = np.mgrid[0:height, 0:width]
    ox = (int(seed) * 7) % 8
    oy = (int(seed) * 13) % 8
    sx = np.floor(xx / sc).astype(np.int64)
    sy = np.floor(yy / sc).astype(np.int64)
    b_val = bayer8[(sy + oy) % 8, (sx + ox) % 8]
    # Add golden-ratio quasi-random R2 phase keyed to seed for rich non-repeating blue spectrum
    phi1 = 0.7548776662
    phi2 = 0.5698402910
    r2 = (0.5 + phi1 * (sx + seed * 3) + phi2 * (sy + seed * 5)) % 1.0
    # High-pass hash difference to suppress low-frequency clumping
    w0 = _hash2i(sx, sy, seed + 41)
    w_avg = 0.25 * (
        _hash2i(sx - 1, sy, seed + 41)
        + _hash2i(sx + 1, sy, seed + 41)
        + _hash2i(sx, sy - 1, seed + 41)
        + _hash2i(sx, sy + 1, seed + 41)
    )
    hp = np.clip(0.5 + (w0 - w_avg), 0.0, 1.0)
    return np.clip(0.48 * b_val + 0.28 * r2 + 0.24 * hp, 0.0, 1.0).astype(np.float32)


def bubble_noise(
    height: int,
    width: int,
    *,
    scale: float = 6.0,
    jitter: float = 0.85,
    rim_width: float = 0.28,
    style: str = "dome",
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    loop_speed: float = 0.0,
) -> np.ndarray:
    """Pixel Composer `Noise Bubble` — circular soap/foam/pore domes or thin rims."""
    d1 = voronoi_noise(
        height, width, scale=scale, jitter=jitter, output="f1",
        seed=seed, tileable=tileable, phase=phase, loop_speed=loop_speed,
    )
    cid = voronoi_noise(
        height, width, scale=scale, jitter=jitter, output="cell_id",
        seed=seed, tileable=tileable, phase=phase, loop_speed=loop_speed,
    )
    radius = 0.28 + 0.26 * cid
    r_norm = np.clip(d1 / np.maximum(0.08, radius), 0.0, 1.5)
    if style == "rim":
        rw = max(0.06, float(rim_width))
        ring = np.exp(-(((r_norm - (1.0 - rw * 0.5)) / (rw * 0.5)) ** 2))
        return np.clip(ring, 0.0, 1.0).astype(np.float32)
    if style == "flat":
        return (r_norm < 0.92).astype(np.float32) * (0.55 + 0.45 * cid)
    # Default: spherical dome bubble
    dome = np.sqrt(np.clip(1.0 - r_norm * r_norm, 0.0, 1.0))
    return dome.astype(np.float32)


def crystal_noise(
    height: int,
    width: int,
    *,
    scale: float = 6.0,
    sharpness: float = 2.4,
    facet_contrast: float = 0.72,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Pixel Composer `Noise Cristal` / Shard — crisp angular faceted crystal planes."""
    cid = voronoi_noise(height, width, scale=scale, jitter=0.92, output="cell_id", seed=seed, tileable=tileable)
    edge = voronoi_noise(height, width, scale=scale, jitter=0.92, output="edge", seed=seed, tileable=tileable)
    f1 = voronoi_noise(height, width, scale=scale, jitter=0.92, output="f1", seed=seed, tileable=tileable)
    # Angular facet slope per Voronoi cell
    u, v = uv_grid(height, width)
    ang = cid * np.float32(2.0 * math.pi)
    plane = 0.5 + 0.5 * (np.cos(ang) * (u * scale % 1.0 - 0.5) + np.sin(ang) * (v * scale % 1.0 - 0.5))
    sh = max(0.5, float(sharpness))
    bevel = np.clip(edge * sh, 0.0, 1.0)
    fc = float(np.clip(facet_contrast, 0.0, 1.0))
    facet = (1.0 - fc) * (1.0 - f1) + fc * (0.62 * cid + 0.38 * plane)
    return np.clip(facet * (0.35 + 0.65 * bevel), 0.0, 1.0).astype(np.float32)


def scratch_noise(
    height: int,
    width: int,
    *,
    count: int = 36,
    length: float = 0.28,
    width_px: float = 1.0,
    angle_deg: float = -28.0,
    angle_jitter: float = 35.0,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Pixel Composer `Noise Scratch` — crisp micro-scratches and worn metallic tool marks."""
    u, v = uv_grid(height, width)
    rng = np.random.default_rng(int(seed) + 419)
    n = max(1, min(200, int(count)))
    acc = np.zeros((height, width), dtype=np.float32)
    half_w = max(0.35, float(width_px)) / float(max(8, min(height, width)))
    base_len = max(0.03, float(length))

    for _ in range(n):
        cx = float(rng.uniform(0.0, 1.0))
        cy = float(rng.uniform(0.0, 1.0))
        ang = math.radians(float(angle_deg) + float(rng.uniform(-angle_jitter, angle_jitter)))
        ln = base_len * float(rng.uniform(0.45, 1.35))
        inten = float(rng.uniform(0.45, 1.0))
        dx = u - cx
        dy = v - cy
        if tileable:
            dx = dx - np.round(dx)
            dy = dy - np.round(dy)
        ca, sa = math.cos(ang), math.sin(ang)
        along = dx * ca + dy * sa
        perp = np.abs(-dx * sa + dy * ca)
        half_l = 0.5 * ln
        # Taper ends cleanly into razor-sharp scratch tips
        t_end = np.clip(1.0 - np.abs(along) / max(1e-4, half_l), 0.0, 1.0)
        local_w = half_w * (0.35 + 0.65 * t_end)
        stroke = np.clip(1.0 - perp / np.maximum(1e-4, local_w), 0.0, 1.0) * np.sqrt(t_end) * inten
        acc = np.maximum(acc, stroke.astype(np.float32))
    return np.clip(acc, 0.0, 1.0).astype(np.float32)


def strand_noise(
    height: int,
    width: int,
    *,
    scale: float = 12.0,
    flow_strength: float = 0.14,
    angle_deg: float = 0.0,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Pixel Composer `Noise Strand` — flowing directional fiber, hair, fur & fabric strands."""
    u, v = uv_grid(height, width)
    th = math.radians(float(angle_deg))
    # Rotate periodic coordinates gently via shear/projection
    s_along = max(1, int(round(scale * 0.28)))
    s_across = max(2, int(round(scale * 2.2)))
    warp = (fbm_noise(height, width, scale=max(2.0, scale * 0.35), octaves=3, seed=seed + 59, tileable=tileable) - 0.5) * float(flow_strength)
    ca, sa = math.cos(th), math.sin(th)
    if abs(sa) < 0.35:
        uu = (u + warp * 0.35) % 1.0
        vv = (v + warp) % 1.0
        base = value_noise(height, width, scale=s_along, scale_y=s_across, seed=seed + 13, tileable=tileable)
        hi = value_noise(height, width, scale=max(2, s_along * 2), scale_y=s_across * 2, seed=seed + 71, tileable=tileable)
        res = 0.65 * sample_field_bilinear(base, uu, vv, tileable=tileable) + 0.35 * sample_field_bilinear(hi, uu, vv, tileable=tileable)
    else:
        uu = (u + warp) % 1.0
        vv = (v + warp * 0.35) % 1.0
        base = value_noise(height, width, scale=s_across, scale_y=s_along, seed=seed + 13, tileable=tileable)
        hi = value_noise(height, width, scale=s_across * 2, scale_y=max(2, s_along * 2), seed=seed + 71, tileable=tileable)
        res = 0.65 * sample_field_bilinear(base, uu, vv, tileable=tileable) + 0.35 * sample_field_bilinear(hi, uu, vv, tileable=tileable)
    lo, hi_v = float(np.percentile(res, 2)), float(np.percentile(res, 98))
    return np.clip((res - lo) / max(1e-4, hi_v - lo), 0.0, 1.0).astype(np.float32)


def caustic_noise(
    height: int,
    width: int,
    *,
    scale: float = 5.0,
    sharpness: float = 2.8,
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    loop_speed: float = 1.0,
) -> np.ndarray:
    """Pixel Composer `Caustic` — animated bright optical caustic web (water, glass, crystal, arcane)."""
    v1 = voronoi_noise(
        height, width, scale=scale, jitter=0.9, output="f1",
        seed=seed + 17, tileable=tileable, phase=phase, loop_speed=loop_speed,
    )
    v2 = voronoi_noise(
        height, width, scale=max(2.0, scale * 1.25), jitter=0.9, output="f1",
        seed=seed + 83, tileable=tileable, phase=phase + 0.33, loop_speed=-loop_speed,
    )
    web = np.minimum(v1, v2) * 0.55 + np.abs(v1 - v2) * 0.45
    sh = max(0.8, float(sharpness))
    bright = np.power(np.clip(1.0 - np.abs(v1 - v2) * 1.85, 0.0, 1.0), sh)
    ridge = np.power(np.clip(web * 1.45, 0.0, 1.0), sh * 0.75)
    return np.clip(0.58 * bright + 0.52 * ridge, 0.0, 1.0).astype(np.float32)


def fold_marble_noise(
    height: int,
    width: int,
    *,
    scale: float = 4.0,
    veins: float = 5.0,
    turbulence: float = 2.6,
    angle_deg: float = 32.0,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Pixel Composer `Fold Noise` / Marble Veins — sine-folded stratified mineral veins."""
    u, v = uv_grid(height, width)
    th = math.radians(float(angle_deg))
    # Snap wavevector to integers when tileable for exact wrap!
    kx = float(veins) * math.cos(th)
    ky = float(veins) * math.sin(th)
    if tileable:
        kx = round(kx)
        ky = round(ky)
        if kx == 0 and ky == 0:
            kx = max(1.0, round(veins))
    turb = fbm_noise(height, width, scale=scale, octaves=4, seed=seed + 29, tileable=tileable)
    arg = 2.0 * math.pi * (kx * u + ky * v) + float(turbulence) * (turb * 2.0 - 1.0) * math.pi
    val = 0.5 + 0.5 * np.sin(arg)
    # Add sharp secondary mineral fracture vein
    sharp_vein = np.power(1.0 - np.abs(np.sin(arg * 0.5)), 4.0)
    return np.clip(0.62 * val + 0.38 * sharp_vein, 0.0, 1.0).astype(np.float32)


def curl_flow_noise(
    height: int,
    width: int,
    *,
    scale: float = 5.0,
    octaves: int = 3,
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    loop_speed: float = 1.0,
) -> np.ndarray:
    """Divergence-free 2D curl-flow wisp noise (ideal for smoke, swirling magic, brushed vortexes)."""
    pot = fbm_noise(
        height, width, scale=scale, octaves=octaves, seed=seed + 101,
        tileable=tileable, phase=phase, loop_speed=loop_speed,
    )
    mode = "wrap" if tileable else "edge"
    p = np.pad(pot, 1, mode=mode)
    # Curl of scalar potential psi: (dpsi/dy, -dpsi/dx)
    dy = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
    dx = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
    mag = np.sqrt(dx * dx + dy * dy + 1e-12)
    # Advect a second fbm layer along the curl field for rich wispy streaks
    u, v = uv_grid(height, width)
    u2 = (u + dy * 1.8) % 1.0
    v2 = (v - dx * 1.8) % 1.0
    adv = sample_field_bilinear(
        fbm_noise(height, width, scale=scale * 1.5, octaves=max(2, octaves), seed=seed + 233, tileable=tileable, phase=phase, loop_speed=loop_speed),
        u2, v2, tileable=tileable,
    )
    lo, hi = float(np.percentile(mag, 2)), float(np.percentile(mag, 98))
    mag_n = np.clip((mag - lo) / max(1e-5, hi - lo), 0.0, 1.0)
    return np.clip(0.55 * adv + 0.45 * mag_n, 0.0, 1.0).astype(np.float32)


def grunge_blotch(
    height: int,
    width: int,
    *,
    scale: float = 5.0,
    contrast: float = 1.45,
    speckle: float = 0.28,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Substance Designer Grunge Map — layered macro blotches, cracks & fine weathered speckles."""
    macro = fbm_noise(height, width, scale=scale, octaves=4, seed=seed + 7, tileable=tileable)
    ridge = fbm_noise(height, width, scale=max(2.0, scale * 1.8), octaves=3, mode="ridged", seed=seed + 61, tileable=tileable)
    micro = value_noise(height, width, scale=max(8.0, scale * 4.5), seed=seed + 149, tileable=tileable)
    sp = float(np.clip(speckle, 0.0, 1.0))
    mix = macro * (0.62 + 0.38 * ridge) * (1.0 - sp * 0.45) + micro * (sp * 0.55)
    c = max(0.2, float(contrast))
    return np.clip((mix - 0.45) * c + 0.5, 0.0, 1.0).astype(np.float32)


def starfield_sparkle(
    height: int,
    width: int,
    *,
    density: float = 0.08,
    twinkle_speed: float = 1.0,
    cross_glint: bool = True,
    seed: int = 0,
    phase: float = 0.0,
) -> np.ndarray:
    """Animated twinkling starfield / gemstone glint / chronal sparkle noise."""
    yy, xx = np.mgrid[0:height, 0:width]
    h0 = _hash2i(xx, yy, seed + 37)
    d = float(np.clip(density, 0.005, 0.45))
    is_star = h0 < d
    base_b = np.where(is_star, 0.45 + 0.55 * _hash2i(xx, yy, seed + 113), 0.0).astype(np.float32)
    ph_off = _hash2i(xx, yy, seed + 229) * np.float32(2.0 * math.pi)
    twinkle = 0.55 + 0.45 * np.sin(ph_off + np.float32(float(phase) * float(twinkle_speed) * 2.0 * math.pi))
    stars = base_b * twinkle
    if cross_glint:
        # Crisp 1px 4-way star sparkle arms on brightest stars
        bright = np.where(stars > 0.72, stars * 0.48, 0.0).astype(np.float32)
        p = np.pad(bright, 1, mode="wrap")
        arms = np.maximum.reduce((p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]))
        stars = np.maximum(stars, arms)
    return np.clip(stars, 0.0, 1.0).astype(np.float32)


def plasma_wave(
    height: int,
    width: int,
    *,
    freq: float = 3.0,
    distortion: float = 1.4,
    speed: float = 1.0,
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
) -> np.ndarray:
    """Animated multi-harmonic plasma / chronal energy interference field (exact periodic loop)."""
    u, v = uv_grid(height, width)
    f = max(1, int(round(freq))) if tileable else max(0.5, float(freq))
    t = float(phase) * float(speed) * 2.0 * math.pi
    s1 = float(seed % 97) * 0.13
    s2 = float((seed * 7) % 89) * 0.17
    w1 = np.sin(2.0 * math.pi * f * u + t + s1)
    w2 = np.sin(2.0 * math.pi * f * v - t + s2)
    w3 = np.sin(2.0 * math.pi * f * (u + v) + t + s1 * 0.5)
    warp = (fbm_noise(height, width, scale=max(2.0, f * 1.5), octaves=3, seed=seed + 41, tileable=tileable, phase=phase, loop_speed=speed) - 0.5) * float(distortion) * math.pi
    w4 = np.cos(2.0 * math.pi * f * (u - v) + warp - t)
    out = 0.5 + 0.125 * (w1 + w2 + w3 + w4)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
