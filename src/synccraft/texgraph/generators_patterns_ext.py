"""Extended Industry-Standard Geometric & Ornamental Pattern Generators (`TexGraph v13`).

Completes Pixel Composer + Substance 3D Designer pattern parity in pure vectorized NumPy:
- ``stripes_pattern``    : Directional stripes, bands & wavy ribbons
- ``checker_tartan``     : Checkerboard, Gingham & Scottish Tartan weave
- ``polka_rivets``       : Rivets, studs, bolts & halftone domes (staggered/aligned)
- ``weave_chainmail``    : Cloth plain weave, twill, basketweave & interlocking chainmail rings
- ``clockwork_gears``    : Animated steampunk/artificer interlocking gears & astrolabe cogs
- ``circuit_runes``      : Procedural PCB traces, arcane rune glyphs & hex conduits
- ``panel_frame_trim``   : Beveled architectural/armor border frame, corner brackets & rivets
- ``polygon_star``       : Regular polygons, heraldic stars & crests
- ``crosshatch_engrave`` : Fine metal engraving, crosshatching & guilloche lines
- ``crystal_facets``     : Cut gemstone / amethyst prism facets with crisp bevels
- ``maze_labyrinth``     : Pixel Composer `Maze` — seeded orthogonal/diagonal labyrinth walls
- ``scallop_shingles``   : Art-Deco / fish-scale / roof-shingle overlapping fan arches
"""
from __future__ import annotations

import math
import numpy as np

from .generators_noise import _hash2i, fbm_noise, uv_grid, voronoi_noise


def stripes_pattern(
    height: int,
    width: int,
    *,
    count: int = 8,
    angle_deg: float = 0.0,
    ratio: float = 0.5,
    wave_amp: float = 0.0,
    softness: float = 0.08,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Clean periodic stripes / bands / wavy ribbons."""
    u, v = uv_grid(height, width)
    th = math.radians(float(angle_deg))
    n = max(1, int(count))
    kx = n * math.cos(th)
    ky = n * math.sin(th)
    if tileable:
        kx = round(kx)
        ky = round(ky)
        if kx == 0 and ky == 0:
            kx = n
    shift = (int(seed) % 100) * 0.01
    coord = kx * u + ky * v + shift
    if abs(float(wave_amp)) > 1e-4:
        coord = coord + float(wave_amp) * np.sin(2.0 * math.pi * (ky * u - kx * v + shift * 3.0))
    frac = coord - np.floor(coord)
    r = float(np.clip(ratio, 0.05, 0.95))
    s = max(0.01, float(softness))
    # Smooth periodic pulse centered at [0, r]
    d = np.minimum(frac, 1.0 - frac)
    edge = np.clip((r - frac) / s, 0.0, 1.0) * np.clip(frac / s, 0.0, 1.0)
    if s <= 0.02:
        return (frac < r).astype(np.float32)
    return np.clip(edge, 0.0, 1.0).astype(np.float32)


def checker_tartan(
    height: int,
    width: int,
    *,
    cols: int = 8,
    rows: int = 8,
    style: str = "checker",
    seed: int = 0,
) -> np.ndarray:
    """Checkerboard, Gingham, or Scottish Tartan plaid scalar field."""
    u, v = uv_grid(height, width)
    c, r = max(1, int(cols)), max(1, int(rows))
    ox = (int(seed) % 7) / float(max(2, c * 2))
    oy = ((int(seed) * 3) % 7) / float(max(2, r * 2))
    gx = (u + ox) * c
    gy = (v + oy) * r
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)
    fx = gx - ix
    fy = gy - iy
    if style == "gingham":
        sx = (ix % 2 == 0).astype(np.float32)
        sy = (iy % 2 == 0).astype(np.float32)
        return (0.20 + 0.40 * sx + 0.40 * sy).astype(np.float32)
    if style == "tartan":
        band_x = (fx < 0.45).astype(np.float32) * 0.55 + ((fx > 0.62) & (fx < 0.72)).astype(np.float32) * 0.95
        band_y = (fy < 0.45).astype(np.float32) * 0.55 + ((fy > 0.62) & (fy < 0.72)).astype(np.float32) * 0.95
        twill = (((np.floor(u * c * 4).astype(np.int64) + np.floor(v * r * 4).astype(np.int64)) % 2) == 0).astype(np.float32)
        return np.clip(0.15 + 0.45 * np.where(twill > 0.5, band_x, band_y) + 0.35 * np.maximum(band_x, band_y), 0.0, 1.0).astype(np.float32)
    return (((ix + iy) % 2) == 0).astype(np.float32)


def polka_rivets(
    height: int,
    width: int,
    *,
    cols: int = 6,
    rows: int = 6,
    radius: float = 0.28,
    stagger: bool = True,
    style: str = "dome",
    seed: int = 0,
) -> np.ndarray:
    """Rivets, armor studs, hex bolts, and halftone polka dots with height relief."""
    u, v = uv_grid(height, width)
    c, r = max(1, int(cols)), max(1, int(rows))
    gy = v * r
    iy = np.floor(gy).astype(np.int64)
    gx = u * c + (0.5 * (iy % 2) if stagger else 0.0)
    ix = np.floor(gx).astype(np.int64)
    fx = (gx - ix) - 0.5
    fy = (gy - iy) - 0.5
    rad = max(0.05, min(0.49, float(radius)))
    cid = _hash2i(ix % c, iy % r, seed + 53)
    var = 0.88 + 0.12 * cid

    if style == "hex_bolt":
        ax, ay = np.abs(fx), np.abs(fy)
        d = np.maximum(ax, 0.5 * ax + 0.8660254 * ay)
        h = np.clip((rad - d) / max(0.03, rad * 0.35), 0.0, 1.0)
        return (h * var).astype(np.float32)
    d = np.sqrt(fx * fx + fy * fy + 1e-12)
    if style == "ring":
        ring = np.clip(1.0 - np.abs(d - rad * 0.78) / max(0.03, rad * 0.28), 0.0, 1.0)
        return (ring * var).astype(np.float32)
    if style == "flat":
        return (d <= rad).astype(np.float32) * var
    # Spherical dome rivet
    rn = np.clip(d / rad, 0.0, 1.2)
    dome = np.sqrt(np.clip(1.0 - rn * rn, 0.0, 1.0))
    return (dome * var).astype(np.float32)


def weave_chainmail(
    height: int,
    width: int,
    *,
    cols: int = 8,
    rows: int = 8,
    style: str = "weave",
    thickness: float = 0.34,
    seed: int = 0,
) -> np.ndarray:
    """Interlocking cloth weave, diagonal twill, or medieval chainmail rings."""
    u, v = uv_grid(height, width)
    c, r = max(1, int(cols)), max(1, int(rows))
    th = float(np.clip(thickness, 0.12, 0.48))

    if style == "chainmail":
        # Two staggered layers of toroidal rings
        def _rings(ox: float, oy: float) -> np.ndarray:
            gx = (u * c + ox) % 1.0 - 0.5
            gy = (v * r + oy) % 1.0 - 0.5
            d = np.sqrt(gx * gx + gy * gy + 1e-12)
            ring_r = 0.38
            tube = max(0.05, th * 0.42)
            prof = np.sqrt(np.clip(1.0 - ((d - ring_r) / tube) ** 2, 0.0, 1.0))
            # Over/under shading along ring angle
            shade = 0.76 + 0.24 * np.sin(np.arctan2(gy, gx) * 2.0 + 0.8)
            return (prof * shade).astype(np.float32)

        r1 = _rings(0.0, 0.0)
        r2 = _rings(0.5, 0.5)
        jitter = 0.92 + 0.08 * _hash2i(np.floor(u * c).astype(np.int64), np.floor(v * r).astype(np.int64), seed + 17)
        return np.clip(np.maximum(r1, r2 * 0.94) * jitter, 0.0, 1.0).astype(np.float32)

    gx = u * c
    gy = v * r
    ix = np.floor(gx).astype(np.int64)
    iy = np.floor(gy).astype(np.int64)
    fx = gx - ix
    fy = gy - iy
    warp_over = (((ix + iy) % 2) == 0) if style != "twill" else (((ix + iy) % 3) == 0)
    # Cross-section arch across strand + sinusoidal over-under elevation along strand
    wx = np.sqrt(np.clip(1.0 - ((np.abs(fx - 0.5)) / max(0.08, th)) ** 2, 0.0, 1.0))
    wy = np.sqrt(np.clip(1.0 - ((np.abs(fy - 0.5)) / max(0.08, th)) ** 2, 0.0, 1.0))
    arch_x = 0.65 + 0.35 * np.sin(fy * math.pi)
    arch_y = 0.65 + 0.35 * np.sin(fx * math.pi)
    hx = wx * np.where(warp_over, arch_x, 0.52 * arch_x)
    hy = wy * np.where(warp_over, 0.52 * arch_y, arch_y)
    var = 0.90 + 0.10 * _hash2i(ix % c, iy % r, seed + 31)
    return np.clip(np.maximum(hx, hy) * var, 0.0, 1.0).astype(np.float32)


def clockwork_gears(
    height: int,
    width: int,
    *,
    teeth: int = 12,
    rings: int = 2,
    style: str = "single",
    rotate_speed: float = 1.0,
    seed: int = 0,
    phase: float = 0.0,
) -> np.ndarray:
    """Animated steampunk / artificer clockwork gears, cogs & astrolabe wheels."""
    u, v = uv_grid(height, width)
    t_rot = float(phase) * float(rotate_speed) * 2.0 * math.pi + (int(seed) % 37) * 0.17
    nt = max(5, min(32, int(teeth)))

    def _gear(cx: float, cy: float, r_outer: float, n_t: int, dir_sign: float) -> np.ndarray:
        dx = u - cx
        dy = v - cy
        r = np.sqrt(dx * dx + dy * dy + 1e-12)
        ang = np.arctan2(dy, dx) + t_rot * dir_sign
        tooth_wave = np.clip(np.cos(ang * n_t) * 2.4 + 0.2, -1.0, 1.0)
        r_edge = r_outer + tooth_wave * (r_outer * 0.12)
        rim = (r <= r_edge) & (r >= r_outer * 0.62)
        hub = (r <= r_outer * 0.32) & (r >= r_outer * 0.11)
        spokes_n = max(3, min(8, int(rings) + 3))
        spoke = (r <= r_outer * 0.68) & (r >= r_outer * 0.28) & (np.abs(np.sin(ang * spokes_n * 0.5)) < 0.22)
        solid = (rim | hub | spoke).astype(np.float32)
        # Bevel highlight toward outer rim and hub center
        bevel = np.clip(1.0 - np.abs(r - r_outer * 0.80) / max(0.04, r_outer * 0.28), 0.55, 1.0)
        return (solid * bevel).astype(np.float32)

    if style == "cluster":
        g1 = _gear(0.36, 0.38, 0.27, nt, 1.0)
        g2 = _gear(0.74, 0.66, 0.22, max(6, nt - 3), -1.0 * nt / max(6, nt - 3))
        g3 = _gear(0.24, 0.78, 0.16, max(5, nt // 2), -1.4)
        return np.clip(np.maximum.reduce((g1, g2 * 0.92, g3 * 0.85)), 0.0, 1.0).astype(np.float32)
    return _gear(0.5, 0.5, 0.39, nt, 1.0)


def circuit_runes(
    height: int,
    width: int,
    *,
    scale: int = 8,
    density: float = 0.55,
    style: str = "circuit",
    seed: int = 0,
) -> np.ndarray:
    """Procedural PCB circuit traces, arcane rune grids, or chronal glyph conduits."""
    u, v = uv_grid(height, width)
    n = max(3, min(32, int(scale)))
    gx = u * n
    gy = v * n
    ix = np.floor(gx).astype(np.int64) % n
    iy = np.floor(gy).astype(np.int64) % n
    fx = gx - np.floor(gx)
    fy = gy - np.floor(gy)
    h1 = _hash2i(ix, iy, seed + 19)
    h2 = _hash2i(ix, iy, seed + 83)
    dens = float(np.clip(density, 0.1, 0.95))
    active = h1 < dens
    w = 0.13

    if style == "runes":
        # Angular glyph strokes inside each cell
        vert = (np.abs(fx - 0.5) < w) & (fy > 0.12) & (fy < 0.88)
        horiz = (np.abs(fy - 0.5) < w) & (fx > 0.15) & (fx < 0.85) & (h2 > 0.35)
        diag1 = (np.abs((fx - 0.5) - (fy - 0.5)) < w * 1.15) & (h2 > 0.52) & (fy > 0.18) & (fy < 0.82)
        diag2 = (np.abs((fx - 0.5) + (fy - 0.5)) < w * 1.15) & (h2 < 0.48) & (fy > 0.18) & (fy < 0.82)
        glyph = (vert | horiz | diag1 | diag2) & active
        return glyph.astype(np.float32)

    # Orthogonal + 45-deg PCB traces + circular via pads
    seg_h = (np.abs(fy - 0.5) < w) & ((h2 < 0.55) | (fx < 0.55))
    seg_v = (np.abs(fx - 0.5) < w) & ((h2 >= 0.35) | (fy < 0.55))
    d_pad = np.sqrt((fx - 0.5) ** 2 + (fy - 0.5) ** 2 + 1e-12)
    pad = (d_pad < 0.25) & (d_pad > 0.09) & (h2 > 0.62)
    trace = ((seg_h | seg_v) & active) | (pad & active)
    return trace.astype(np.float32)


def panel_frame_trim(
    height: int,
    width: int,
    *,
    border_width: float = 0.14,
    corner_size: float = 0.24,
    style: str = "bevel_frame",
    rivets: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Architectural/armor panel border trim, beveled corner brackets & edge rivets."""
    _ = seed
    u, v = uv_grid(height, width)
    du = np.minimum(u, 1.0 - u)
    dv = np.minimum(v, 1.0 - v)
    d_edge = np.minimum(du, dv)
    bw = float(np.clip(border_width, 0.03, 0.42))
    cs = float(np.clip(corner_size, 0.05, 0.48))

    in_border = d_edge <= bw
    bevel = np.clip(1.0 - np.abs(d_edge - bw * 0.5) / max(0.015, bw * 0.5), 0.0, 1.0)
    h = np.where(in_border, 0.45 + 0.55 * bevel, 0.22).astype(np.float32)

    if style == "brackets":
        corner = (du <= cs) & (dv <= cs) & in_border
        h = np.where(corner, 0.65 + 0.35 * bevel, 0.18).astype(np.float32)
    elif style == "double_trim":
        inner = (d_edge >= bw * 1.35) & (d_edge <= bw * 1.75)
        h = np.where(inner, 0.78, h).astype(np.float32)

    if rivets:
        # 4 crisp corner rivet studs
        rc = bw * 0.52
        for rx in (rc, 1.0 - rc):
            for ry in (rc, 1.0 - rc):
                dr = np.sqrt((u - rx) ** 2 + (v - ry) ** 2 + 1e-12)
                stud = np.clip(1.0 - dr / max(0.02, bw * 0.34), 0.0, 1.0)
                h = np.maximum(h, (0.55 + 0.45 * stud) * (stud > 0.05))
    return np.clip(h, 0.0, 1.0).astype(np.float32)


def polygon_star(
    height: int,
    width: int,
    *,
    sides: int = 6,
    inner_radius: float = 0.5,
    radius: float = 0.38,
    rotation_deg: float = 0.0,
    repeat: int = 1,
    bevel: float = 0.22,
    seed: int = 0,
) -> np.ndarray:
    """Regular polygon, heraldic star, or tiled emblem SDF heightfield."""
    _ = seed
    u, v = uv_grid(height, width)
    rep = max(1, min(16, int(repeat)))
    gx = (u * rep) % 1.0 - 0.5
    gy = (v * rep) % 1.0 - 0.5
    r = np.sqrt(gx * gx + gy * gy + 1e-12)
    n = max(3, min(16, int(sides)))
    ang = np.arctan2(gy, gx) + math.radians(float(rotation_deg))
    sector = 2.0 * math.pi / float(n)
    a_loc = np.abs(((ang + sector * 0.5) % sector) - sector * 0.5)
    # Interpolate outer vertex radius (a_loc=0) and inner valley radius (a_loc=sector/2)
    t_ang = a_loc / (sector * 0.5)
    ir = float(np.clip(inner_radius, 0.15, 1.0))
    # For regular polygon (ir >= 0.99), use exact cos(pi/n)/cos(a_loc)
    r_poly = float(radius) * math.cos(math.pi / float(n)) / np.maximum(0.25, np.cos(a_loc))
    r_star = float(radius) * ((1.0 - t_ang) + t_ang * ir)
    r_bound = r_poly if ir >= 0.98 else r_star
    sdf = r_bound - r
    bv = max(0.01, float(bevel) * float(radius))
    return np.clip(sdf / bv, 0.0, 1.0).astype(np.float32)


def crosshatch_engrave(
    height: int,
    width: int,
    *,
    freq: float = 12.0,
    angle_deg: float = 45.0,
    cross: bool = True,
    weight: float = 0.26,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Fine crosshatching, armor etching, and guilloche engraving grooves."""
    s1 = stripes_pattern(
        height, width, count=max(2, int(round(freq))), angle_deg=angle_deg,
        ratio=float(np.clip(weight, 0.08, 0.65)), softness=0.06, seed=seed, tileable=tileable,
    )
    if not cross:
        return (1.0 - s1 * 0.75).astype(np.float32)
    s2 = stripes_pattern(
        height, width, count=max(2, int(round(freq))), angle_deg=angle_deg + 90.0,
        ratio=float(np.clip(weight, 0.08, 0.65)), softness=0.06, seed=seed + 19, tileable=tileable,
    )
    grooves = np.maximum(s1, s2 * 0.88)
    return np.clip(1.0 - grooves * 0.72, 0.0, 1.0).astype(np.float32)


def crystal_facets(
    height: int,
    width: int,
    *,
    scale: float = 5.0,
    facet_contrast: float = 0.78,
    rim_width: float = 0.14,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Cut gemstone / faceted amethyst prism height & specular facet pattern."""
    edge = voronoi_noise(height, width, scale=scale, jitter=0.95, output="edge", seed=seed + 3, tileable=tileable)
    cid = voronoi_noise(height, width, scale=scale, jitter=0.95, output="cell_id", seed=seed + 3, tileable=tileable)
    f1 = voronoi_noise(height, width, scale=scale, jitter=0.95, output="f1", seed=seed + 3, tileable=tileable)
    rw = max(0.03, float(rim_width))
    ridge = np.clip(1.0 - edge / rw, 0.0, 1.0)
    fc = float(np.clip(facet_contrast, 0.0, 1.0))
    body = (1.0 - fc) * (1.0 - f1) + fc * cid
    return np.clip(body * 0.82 + ridge * 0.35, 0.0, 1.0).astype(np.float32)


def maze_labyrinth(
    height: int,
    width: int,
    *,
    cols: int = 8,
    rows: int = 8,
    wall_width: float = 0.28,
    style: str = "orthogonal",
    seed: int = 0,
) -> np.ndarray:
    """Pixel Composer `Maze` — seeded orthogonal or diagonal labyrinth walls."""
    u, v = uv_grid(height, width)
    c, r = max(2, int(cols)), max(2, int(rows))
    gx = u * c
    gy = v * r
    ix = np.floor(gx).astype(np.int64) % c
    iy = np.floor(gy).astype(np.int64) % r
    fx = gx - np.floor(gx)
    fy = gy - np.floor(gy)
    h = _hash2i(ix, iy, seed + 131)
    ww = float(np.clip(wall_width, 0.10, 0.45))

    if style == "diagonal":
        d1 = np.abs(fx - fy)
        d2 = np.abs(fx + fy - 1.0)
        d = np.where(h < 0.5, d1, d2)
        return np.clip(1.0 - d / max(0.05, ww), 0.0, 1.0).astype(np.float32)

    # Binary-tree orthogonal labyrinth: each cell carves either North or East passage!
    wall_n = (fy < ww) & (h >= 0.5)
    wall_w = (fx < ww) & (h < 0.5)
    pillar = (fx < ww) & (fy < ww)
    wall = (wall_n | wall_w | pillar).astype(np.float32)
    return wall


def scallop_shingles(
    height: int,
    width: int,
    *,
    cols: int = 6,
    rows: int = 8,
    overlap: float = 0.45,
    bevel: float = 0.18,
    seed: int = 0,
) -> np.ndarray:
    """Overlapping Art-Deco scallops, roof shingles, or rounded armor scales."""
    u, v = uv_grid(height, width)
    c, r = max(1, int(cols)), max(1, int(rows))
    out = np.zeros((height, width), dtype=np.float32)
    bv = max(0.04, float(bevel))
    ov = float(np.clip(overlap, 0.15, 0.85))
    # Evaluate 2 vertically staggered overlapping rows so lower rows tuck cleanly beneath upper rows
    for row_off in (0, -1):
        gy = v * r - row_off
        iy = np.floor(gy).astype(np.int64)
        gx = u * c + 0.5 * (iy % 2)
        ix = np.floor(gx).astype(np.int64)
        fx = (gx - ix) - 0.5
        fy = gy - iy
        d = np.sqrt(fx * fx + (fy / (0.65 + ov * 0.55)) ** 2 + 1e-12)
        in_tile = (fy >= 0.0) & (d <= 0.52)
        prof = np.clip((0.52 - d) / bv, 0.0, 1.0) * (0.45 + 0.55 * np.clip(fy, 0.0, 1.0))
        cid = 0.85 + 0.15 * _hash2i(ix % c, iy % r, seed + 67)
        layer = np.where(in_tile, prof * cid, 0.0).astype(np.float32)
        out = np.maximum(out, layer)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
