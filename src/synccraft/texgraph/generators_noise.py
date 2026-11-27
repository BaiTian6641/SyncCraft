"""Pure-NumPy seamless 2D & 3D rest-space noise, Voronoi, Gabor, warp & dither generators.

All 2D generators support exact periodic tiling (`tileable=True`) so block textures wrap
seamlessly across edges, plus smooth periodic `phase` modulation for looping flipbooks.
"""
from __future__ import annotations

import math
import numpy as np


def _hash2i(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic uint32 spatial hash mapped to float32 in [0, 1)."""
    h = (
        (ix.astype(np.int64) * 374761393)
        ^ (iy.astype(np.int64) * 668265263)
        ^ (int(seed) * 2654435761)
    ) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return (h.astype(np.float32) / np.float32(4294967296.0))


def _hash3i(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    h = (
        (ix.astype(np.int64) * 73856093)
        ^ (iy.astype(np.int64) * 19349663)
        ^ (iz.astype(np.int64) * 83492791)
        ^ (int(seed) * 2654435761)
    ) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return (h.astype(np.float32) / np.float32(4294967296.0))


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def uv_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return pixel-centered (u, v) float32 coordinate grids in [0, 1)."""
    u = (np.arange(width, dtype=np.float32) + 0.5) / float(max(1, width))
    v = (np.arange(height, dtype=np.float32) + 0.5) / float(max(1, height))
    return np.meshgrid(u, v)


def value_noise(
    height: int,
    width: int,
    *,
    scale: float = 8.0,
    scale_y: float | None = None,
    seed: int = 0,
    tileable: bool = True,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Smooth quintic 2D value noise in [0, 1], optionally exact-periodic."""
    sx = max(1, int(round(scale))) if tileable else max(1e-4, float(scale))
    sy = max(1, int(round(scale_y if scale_y is not None else scale))) if tileable else max(1e-4, float(scale_y if scale_y is not None else scale))
    u, v = uv_grid(height, width)
    x = u * sx + float(offset_x) * sx
    y = v * sy + float(offset_y) * sy
    ix0 = np.floor(x).astype(np.int64)
    iy0 = np.floor(y).astype(np.int64)
    fx = (x - ix0).astype(np.float32)
    fy = (y - iy0).astype(np.float32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    if tileable:
        ix0 = ix0 % int(sx)
        ix1 = ix1 % int(sx)
        iy0 = iy0 % int(sy)
        iy1 = iy1 % int(sy)
    ux, uy = _fade(fx), _fade(fy)
    c00 = _hash2i(ix0, iy0, seed)
    c10 = _hash2i(ix1, iy0, seed)
    c01 = _hash2i(ix0, iy1, seed)
    c11 = _hash2i(ix1, iy1, seed)
    a = c00 + ux * (c10 - c00)
    b = c01 + ux * (c11 - c01)
    return (a + uy * (b - a)).astype(np.float32)


def perlin_noise(
    height: int,
    width: int,
    *,
    scale: float = 8.0,
    scale_y: float | None = None,
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    loop_speed: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Gradient 2D Perlin noise normalized to [0, 1] with exact periodic wrap and optional phase rotation."""
    sx = max(1, int(round(scale))) if tileable else max(1e-4, float(scale))
    sy = max(1, int(round(scale_y if scale_y is not None else scale))) if tileable else max(1e-4, float(scale_y if scale_y is not None else scale))
    u, v = uv_grid(height, width)
    x = u * sx + float(offset_x) * sx
    y = v * sy + float(offset_y) * sy

    ix0 = np.floor(x).astype(np.int64)
    iy0 = np.floor(y).astype(np.int64)
    fx = (x - ix0).astype(np.float32)
    fy = (y - iy0).astype(np.float32)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    if tileable:
        ix0 = ix0 % int(sx)
        ix1 = ix1 % int(sx)
        iy0 = iy0 % int(sy)
        iy1 = iy1 % int(sy)

    angle_shift = float(phase) * float(loop_speed) * 2.0 * math.pi

    def _grad_dot(gx_i: np.ndarray, gy_i: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
        ang = _hash2i(gx_i, gy_i, seed) * np.float32(2.0 * math.pi) + np.float32(angle_shift)
        return np.cos(ang) * dx + np.sin(ang) * dy

    n00 = _grad_dot(ix0, iy0, fx, fy)
    n10 = _grad_dot(ix1, iy0, fx - 1.0, fy)
    n01 = _grad_dot(ix0, iy1, fx, fy - 1.0)
    n11 = _grad_dot(ix1, iy1, fx - 1.0, fy - 1.0)

    ux, uy = _fade(fx), _fade(fy)
    nx0 = n00 + ux * (n10 - n00)
    nx1 = n01 + ux * (n11 - n01)
    val = nx0 + uy * (nx1 - nx0)
    return np.clip(val * np.float32(0.7071) + np.float32(0.5), 0.0, 1.0).astype(np.float32)


def fbm_noise(
    height: int,
    width: int,
    *,
    scale: float = 6.0,
    scale_y: float | None = None,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    mode: str = "fbm",
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    loop_speed: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """Multi-octave noise supporting 'fbm', 'ridged', 'billow', and 'turbulence' modes."""
    acc = np.zeros((height, width), dtype=np.float32)
    amp = 1.0
    norm = 0.0
    freq_x = float(scale)
    freq_y = float(scale_y if scale_y is not None else scale)
    lac = max(1, int(round(lacunarity))) if tileable else float(lacunarity)

    for oc in range(max(1, int(octaves))):
        n = perlin_noise(
            height,
            width,
            scale=freq_x,
            scale_y=freq_y,
            seed=seed + oc * 131,
            tileable=tileable,
            phase=phase,
            loop_speed=loop_speed * (1 if oc % 2 == 0 else -1),
            offset_x=offset_x,
            offset_y=offset_y,
        )
        if mode == "ridged":
            v = 1.0 - np.abs(2.0 * n - 1.0)
            v = v * v
        elif mode == "billow":
            v = np.abs(2.0 * n - 1.0)
        elif mode == "turbulence":
            v = np.abs(2.0 * n - 1.0)
        else:
            v = n
        acc += np.float32(amp) * v
        norm += amp
        amp *= float(persistence)
        freq_x *= lac
        freq_y *= lac

    return np.clip(acc / max(1e-6, norm), 0.0, 1.0).astype(np.float32)


def voronoi_noise(
    height: int,
    width: int,
    *,
    scale: float = 6.0,
    scale_y: float | None = None,
    jitter: float = 0.85,
    metric: str = "euclidean",
    output: str = "f1",
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    loop_speed: float = 0.0,
) -> np.ndarray:
    """Seamless 2D Voronoi/Worley cellular generator.

    ``output`` in ``{"f1", "f2", "edge", "cracks", "cell_id", "bevel"}``.
    ``metric`` in ``{"euclidean", "manhattan", "chebyshev"}``.
    """
    sx = max(1, int(round(scale))) if tileable else max(1e-4, float(scale))
    sy = max(1, int(round(scale_y if scale_y is not None else scale))) if tileable else max(1e-4, float(scale_y if scale_y is not None else scale))
    u, v = uv_grid(height, width)
    x = u * sx
    y = v * sy
    bx = np.floor(x).astype(np.int64)
    by = np.floor(y).astype(np.int64)
    jit = float(np.clip(jitter, 0.0, 1.0))

    d1 = np.full((height, width), 1e9, dtype=np.float32)
    d2 = np.full((height, width), 1e9, dtype=np.float32)
    best_id = np.zeros((height, width), dtype=np.float32)

    ang_phase = float(phase) * float(loop_speed) * 2.0 * math.pi

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            cx = bx + dx
            cy = by + dy
            hx = cx % int(sx) if tileable else cx
            hy = cy % int(sy) if tileable else cy
            rx = _hash2i(hx, hy, seed + 17)
            ry = _hash2i(hx, hy, seed + 53)
            if abs(ang_phase) > 1e-6:
                orbit_a = _hash2i(hx, hy, seed + 97) * np.float32(2.0 * math.pi) + np.float32(ang_phase)
                px = cx.astype(np.float32) + 0.5 + jit * 0.42 * np.cos(orbit_a)
                py = cy.astype(np.float32) + 0.5 + jit * 0.42 * np.sin(orbit_a)
            else:
                px = cx.astype(np.float32) + 0.5 + (rx - 0.5) * jit
                py = cy.astype(np.float32) + 0.5 + (ry - 0.5) * jit

            ddx = np.abs(x - px)
            ddy = np.abs(y - py)
            if metric == "manhattan":
                dist = ddx + ddy
            elif metric == "chebyshev":
                dist = np.maximum(ddx, ddy)
            else:
                dist = np.sqrt(ddx * ddx + ddy * ddy + np.float32(1e-12))

            cid = _hash2i(hx, hy, seed + 193)
            less1 = dist < d1
            d2 = np.where(less1, d1, np.minimum(d2, dist))
            best_id = np.where(less1, cid, best_id)
            d1 = np.where(less1, dist, d1)

    if output == "f2":
        return np.clip(d2 / 1.4, 0.0, 1.0).astype(np.float32)
    if output == "f2_minus_f1":
        return np.clip(d2 - d1, 0.0, 1.0).astype(np.float32)
    if output == "edge":
        return np.clip((d2 - d1) * 1.4, 0.0, 1.0).astype(np.float32)
    if output == "cracks":
        edge = np.clip((d2 - d1) * 2.2, 0.0, 1.0)
        return (1.0 - edge).astype(np.float32)
    if output == "bevel":
        return np.clip((d2 - d1) * 3.5, 0.0, 1.0).astype(np.float32)
    if output == "cell_id":
        return best_id.astype(np.float32)
    return np.clip(d1, 0.0, 1.0).astype(np.float32)


def gabor_noise(
    height: int,
    width: int,
    *,
    scale: float = 12.0,
    angle_deg: float = 0.0,
    anisotropy: float = 3.5,
    octaves: int = 3,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Directional anisotropic Gabor-like grain (ideal for fur, feathers, brushed metal, wood)."""
    u, v = uv_grid(height, width)
    th = math.radians(float(angle_deg))
    c, s = math.cos(th), math.sin(th)
    # Periodic directional harmonic blend when tileable (snap wavevectors to integers!)
    acc = np.zeros((height, width), dtype=np.float32)
    norm = 0.0
    rng = np.random.default_rng(seed + 701)
    for oc in range(max(1, int(octaves))):
        freq = float(scale) * (2.0 ** oc)
        amp = 0.55 ** oc
        n_kernels = 6
        layer = np.zeros((height, width), dtype=np.float32)
        for _ in range(n_kernels):
            jitter_ang = rng.normal(0.0, 0.25 / max(1.0, float(anisotropy)))
            a = th + jitter_ang
            kx = math.cos(a) * freq
            ky = math.sin(a) * freq
            if tileable:
                kx = round(kx)
                ky = round(ky)
                if kx == 0 and ky == 0:
                    kx = max(1, int(round(freq)))
            ph = float(rng.uniform(0.0, 2.0 * math.pi))
            layer += np.cos(2.0 * math.pi * (kx * u + ky * v) + ph).astype(np.float32)
        layer /= float(n_kernels)
        # Modulate envelope with anisotropic low-freq value noise so grain breaks organically
        env = value_noise(
            height,
            width,
            scale=max(2, int(round(scale * 0.4 * (2**oc)))),
            scale_y=max(2, int(round(scale * 0.4 * float(anisotropy) * (2**oc)))),
            seed=seed + oc * 43 + 9,
            tileable=tileable,
        )
        acc += np.float32(amp) * (0.5 + 0.5 * layer) * (0.65 + 0.7 * env)
        norm += amp
    out = acc / max(1e-6, norm)
    # Normalize robustly into [0, 1]
    lo, hi = float(np.percentile(out, 2)), float(np.percentile(out, 98))
    return np.clip((out - lo) / max(1e-5, hi - lo), 0.0, 1.0).astype(np.float32)


def sample_field_bilinear(field: np.ndarray, u: np.ndarray, v: np.ndarray, *, tileable: bool = True) -> np.ndarray:
    """Vectorized bilinear sampler for 2D scalar (H,W) or multi-channel (H,W,C) arrays."""
    h, w = field.shape[:2]
    x = u * float(w) - 0.5
    y = v * float(h) - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)
    fy = (y - y0).astype(np.float32)
    x1 = x0 + 1
    y1 = y0 + 1
    if tileable:
        x0 = x0 % w
        x1 = x1 % w
        y0 = y0 % h
        y1 = y1 % h
    else:
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
    if field.ndim == 3:
        fx = fx[..., None]
        fy = fy[..., None]
    c00 = field[y0, x0]
    c10 = field[y0, x1]
    c01 = field[y1, x0]
    c11 = field[y1, x1]
    return ((c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy).astype(np.float32)


def domain_warp(
    field: np.ndarray,
    *,
    strength: float = 0.08,
    scale: float = 4.0,
    octaves: int = 3,
    mode: str = "fbm",
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    flow_speed: float = 0.0,
) -> np.ndarray:
    """Domain-warp or curl-warp any scalar or RGBA image seamlessly."""
    h, w = field.shape[:2]
    u, v = uv_grid(h, w)
    if mode == "curl":
        pot = fbm_noise(h, w, scale=scale, octaves=octaves, seed=seed, tileable=tileable, phase=phase, loop_speed=flow_speed)
        # Divergence-free 2D curl: (dP/dy, -dP/dx)
        dp_dy = (np.roll(pot, -1, axis=0) - np.roll(pot, 1, axis=0)) * 0.5 * h * 0.08
        dp_dx = (np.roll(pot, -1, axis=1) - np.roll(pot, 1, axis=1)) * 0.5 * w * 0.08
        du = dp_dy * float(strength)
        dv = -dp_dx * float(strength)
    else:
        wx = fbm_noise(h, w, scale=scale, octaves=octaves, seed=seed + 13, tileable=tileable, phase=phase, loop_speed=flow_speed) - 0.5
        wy = fbm_noise(h, w, scale=scale, octaves=octaves, seed=seed + 79, tileable=tileable, phase=phase, loop_speed=flow_speed) - 0.5
        du = wx * float(strength) * 2.0
        dv = wy * float(strength) * 2.0
    return sample_field_bilinear(field, u + du, v + dv, tileable=tileable)


def bayer_matrix(height: int, width: int, *, size: int = 4) -> np.ndarray:
    """Tiled normalized Bayer ordered-dither threshold matrix in [0, 1)."""
    if size <= 2:
        m = np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0
    elif size <= 4:
        m = np.array(
            [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ],
            dtype=np.float32,
        ) / 16.0
    else:
        b4 = np.array(
            [
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ],
            dtype=np.float32,
        )
        m = np.block([[4 * b4, 4 * b4 + 2], [4 * b4 + 3, 4 * b4 + 1]]) / 64.0
    mh, mw = m.shape
    return np.tile(m, (height // mh + 1, width // mw + 1))[:height, :width].astype(np.float32)


def noise_3d_field(
    pos_xyz: np.ndarray,
    *,
    scale: float = 0.25,
    octaves: int = 3,
    mode: str = "fbm",
    seed: int = 0,
    phase: float = 0.0,
    flow_vec: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Seamless 3D world-space procedural noise evaluated directly on MeshBaker `world_pos` (H,W,3).

    Eliminates UV seam discontinuities completely across adjacent 3D faces!
    """
    p = np.asarray(pos_xyz, dtype=np.float32) * np.float32(scale)
    if any(abs(c) > 1e-6 for c in flow_vec):
        p = p + np.array(flow_vec, dtype=np.float32) * np.float32(phase)

    def _val3(coords: np.ndarray, s: int) -> np.ndarray:
        fl = np.floor(coords)
        ix = fl[..., 0].astype(np.int64)
        iy = fl[..., 1].astype(np.int64)
        iz = fl[..., 2].astype(np.int64)
        fx = (coords[..., 0] - fl[..., 0]).astype(np.float32)
        fy = (coords[..., 1] - fl[..., 1]).astype(np.float32)
        fz = (coords[..., 2] - fl[..., 2]).astype(np.float32)
        ux, uy, uz = _fade(fx), _fade(fy), _fade(fz)
        c000 = _hash3i(ix, iy, iz, s)
        c100 = _hash3i(ix + 1, iy, iz, s)
        c010 = _hash3i(ix, iy + 1, iz, s)
        c110 = _hash3i(ix + 1, iy + 1, iz, s)
        c001 = _hash3i(ix, iy, iz + 1, s)
        c101 = _hash3i(ix + 1, iy, iz + 1, s)
        c011 = _hash3i(ix, iy + 1, iz + 1, s)
        c111 = _hash3i(ix + 1, iy + 1, iz + 1, s)
        x00 = c000 + ux * (c100 - c000)
        x10 = c010 + ux * (c110 - c010)
        x01 = c001 + ux * (c101 - c001)
        x11 = c011 + ux * (c111 - c011)
        y0 = x00 + uy * (x10 - x00)
        y1 = x01 + uy * (x11 - x01)
        return (y0 + uz * (y1 - y0)).astype(np.float32)

    if mode in ("voronoi_f1", "voronoi_edge", "voronoi_cracks"):
        fl = np.floor(p)
        bx = fl[..., 0].astype(np.int64)
        by = fl[..., 1].astype(np.int64)
        bz = fl[..., 2].astype(np.int64)
        d1 = np.full(p.shape[:-1], 1e9, dtype=np.float32)
        d2 = np.full(p.shape[:-1], 1e9, dtype=np.float32)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    cx, cy, cz = bx + dx, by + dy, bz + dz
                    px = cx.astype(np.float32) + 0.1 + 0.8 * _hash3i(cx, cy, cz, seed + 11)
                    py = cy.astype(np.float32) + 0.1 + 0.8 * _hash3i(cx, cy, cz, seed + 29)
                    pz = cz.astype(np.float32) + 0.1 + 0.8 * _hash3i(cx, cy, cz, seed + 53)
                    dist2 = (p[..., 0] - px) ** 2 + (p[..., 1] - py) ** 2 + (p[..., 2] - pz) ** 2
                    less1 = dist2 < d1
                    d2 = np.where(less1, d1, np.minimum(d2, dist2))
                    d1 = np.where(less1, dist2, d1)
        r1 = np.sqrt(np.maximum(0.0, d1))
        r2 = np.sqrt(np.maximum(0.0, d2))
        if mode == "voronoi_f1":
            return np.clip(r1, 0.0, 1.0).astype(np.float32)
        edge = np.clip((r2 - r1) * 1.8, 0.0, 1.0)
        return (1.0 - edge if mode == "voronoi_cracks" else edge).astype(np.float32)

    acc = np.zeros(p.shape[:-1], dtype=np.float32)
    amp = 1.0
    norm = 0.0
    freq = 1.0
    for oc in range(max(1, int(octaves))):
        n = _val3(p * np.float32(freq), seed + oc * 107)
        if mode == "ridged":
            v = 1.0 - np.abs(2.0 * n - 1.0)
            v = v * v
        elif mode == "billow":
            v = np.abs(2.0 * n - 1.0)
        else:
            v = n
        acc += np.float32(amp) * v
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return np.clip(acc / max(1e-6, norm), 0.0, 1.0).astype(np.float32)
