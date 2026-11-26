"""Structured geometric, architectural, ornamental & biological pattern generators.

Covers: square grid, brick (running/herringbone/basketweave/stacked), hex honeycomb,
triangle facets, diamond/chevron/argyle, Truchet scrollwork, Voronoi mosaic/stained-glass,
imbricated dragon/fish scales, feather barbs, fur tufts, radial astrolabe rings, and
animated neon/lightning/circuit energy veins.
"""
from __future__ import annotations

import math
import numpy as np

from .generators_noise import _hash2i, domain_warp, fbm_noise, uv_grid, voronoi_noise


def grid_pattern(
    height: int,
    width: int,
    *,
    cols: int = 4,
    rows: int = 4,
    mortar: float = 0.08,
    bevel: float = 0.14,
    seed: int = 0,
    output: str = "height",
) -> np.ndarray:
    """Clean tileable rectangular grid with beveled tiles and recessed mortar."""
    u, v = uv_grid(height, width)
    gx = u * max(1, int(cols))
    gy = v * max(1, int(rows))
    ix = np.floor(gx).astype(np.int64) % max(1, int(cols))
    iy = np.floor(gy).astype(np.int64) % max(1, int(rows))
    fx = gx - np.floor(gx)
    fy = gy - np.floor(gy)

    edge_d = np.minimum(np.minimum(fx, 1.0 - fx), np.minimum(fy, 1.0 - fy))
    m_half = float(mortar) * 0.5
    if output == "mask":
        return (edge_d >= m_half).astype(np.float32)
    if output == "cell_id":
        return _hash2i(ix, iy, seed + 19)
    h_prof = np.clip((edge_d - m_half) / max(1e-4, float(bevel)), 0.0, 1.0)
    cell_var = 0.85 + 0.15 * _hash2i(ix, iy, seed + 37)
    return (h_prof * cell_var).astype(np.float32)


def brick_pattern(
    height: int,
    width: int,
    *,
    cols: int = 4,
    rows: int = 8,
    style: str = "running",
    offset: float = 0.5,
    mortar: float = 0.08,
    bevel: float = 0.16,
    chip_strength: float = 0.12,
    seed: int = 0,
    output: str = "height",
) -> np.ndarray:
    """Tileable masonry bricks: 'running', 'stacked', 'herringbone', or 'basketweave'."""
    u, v = uv_grid(height, width)
    c = max(1, int(cols))
    r = max(1, int(rows))

    if style == "herringbone":
        # Seamless 2x2 repeating herringbone unit cell
        n = max(1, c)
        gx = u * (n * 2)
        gy = v * (n * 2)
        ix = np.floor(gx).astype(np.int64)
        iy = np.floor(gy).astype(np.int64)
        fx = gx - ix
        fy = gy - iy
        diag = (ix + iy) % 2
        # Horizontal brick pair vs vertical brick pair
        bx_id = np.where(diag == 0, ix // 2, ix)
        by_id = np.where(diag == 0, iy, iy // 2)
        lu = np.where(diag == 0, ((ix % 2) + fx) * 0.5, fx)
        lv = np.where(diag == 0, fy, ((iy % 2) + fy) * 0.5)
        edge_d = np.minimum(np.minimum(lu, 1.0 - lu) * 2.0, np.minimum(lv, 1.0 - lv))
        cid = _hash2i(bx_id % (n * 2), by_id % (n * 2), seed + 71)
    elif style == "basketweave":
        n = max(1, c)
        gx = u * (n * 2)
        gy = v * (n * 2)
        ix = np.floor(gx).astype(np.int64)
        iy = np.floor(gy).astype(np.int64)
        fx = gx - ix
        fy = gy - iy
        horiz = ((ix + iy) % 2) == 0
        # Split each square block into 2 parallel planks
        lu = np.where(horiz, fx, (fx * 2.0) % 1.0)
        lv = np.where(horiz, (fy * 2.0) % 1.0, fy)
        sub_id = np.where(horiz, np.floor(fy * 2.0).astype(np.int64), np.floor(fx * 2.0).astype(np.int64))
        edge_d = np.minimum(np.minimum(lu, 1.0 - lu), np.minimum(lv, 1.0 - lv))
        cid = _hash2i((ix % (n * 2)) * 3 + sub_id, iy % (n * 2), seed + 89)
    else:
        gy = v * r
        iy = np.floor(gy).astype(np.int64)
        row_shift = (iy % 2) * (float(offset) if style != "stacked" else 0.0)
        gx = u * c + row_shift
        ix = np.floor(gx).astype(np.int64) % c
        fx = gx - np.floor(gx)
        fy = gy - iy
        # Keep horizontal and vertical mortar thicknesses optically equal
        ratio = float(r) / float(c)
        dx = np.minimum(fx, 1.0 - fx) * ratio
        dy = np.minimum(fy, 1.0 - fy)
        edge_d = np.minimum(dx, dy)
        cid = _hash2i(ix, iy % r, seed + 41)

    if chip_strength > 1e-4:
        chip = (fbm_noise(height, width, scale=max(4, c * 2), octaves=3, seed=seed + 211, tileable=True) - 0.5) * float(chip_strength)
        edge_d = edge_d + chip

    m_half = float(mortar) * 0.5
    if output == "mask":
        return (edge_d >= m_half).astype(np.float32)
    if output == "mortar":
        return (edge_d < m_half).astype(np.float32)
    if output == "cell_id":
        return cid.astype(np.float32)
    h_prof = np.clip((edge_d - m_half) / max(1e-4, float(bevel)), 0.0, 1.0)
    # Slight per-brick height relief + subtle top-to-bottom brick face slope
    return np.clip(h_prof * (0.78 + 0.22 * cid), 0.0, 1.0).astype(np.float32)


def hex_grid_pattern(
    height: int,
    width: int,
    *,
    cols: int = 4,
    mortar: float = 0.08,
    bevel: float = 0.14,
    seed: int = 0,
    output: str = "height",
) -> np.ndarray:
    """Exact seamless periodic hexagonal honeycomb pattern."""
    # Seamless tileable hex lattice using dual-offset rectangular grid (aspect sqrt(3))
    c = max(1, int(cols))
    r = max(1, int(round(c / math.sqrt(3.0))))
    u, v = uv_grid(height, width)

    # Dual offset grids A (0,0) and B (0.5, 0.5) in stretched hex metric
    sx = float(c)
    sy = float(r) * math.sqrt(3.0)
    px = u * sx
    py = v * sy

    # Grid A cell centers at (i + 0.5, j*sqrt(3) + 0.5*sqrt(3))
    def _nearest_grid(ox: float, oy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gx = px - ox
        gy = py / math.sqrt(3.0) - oy
        ix = np.floor(gx).astype(np.int64)
        iy = np.floor(gy).astype(np.int64)
        fx = (gx - ix) - 0.5
        fy = ((gy - iy) - 0.5) * math.sqrt(3.0)
        # Hex distance metric (max of 3 axial projections)
        ax = np.abs(fx)
        ay = np.abs(fy)
        hex_d = np.maximum(ax, 0.5 * ax + 0.8660254 * ay)
        euc_d = np.sqrt(fx * fx + fy * fy + 1e-12)
        return hex_d, euc_d, ix % c, iy % r, fx

    ha, ea, ixa, iya, _ = _nearest_grid(0.0, 0.0)
    hb, eb, ixb, iyb, _ = _nearest_grid(0.5, 0.5)

    use_a = ea < eb
    d_hex = np.where(use_a, ha, hb)
    cid = np.where(use_a, _hash2i(ixa, iya, seed + 13), _hash2i(ixb + 1000, iyb + 1000, seed + 29))

    # Distance from hex boundary (boundary is at d_hex == 0.5)
    edge_dist = np.clip(0.5 - d_hex, 0.0, 0.5) * 2.0
    if output == "edge":
        return np.clip(1.0 - edge_dist / max(1e-4, float(mortar)), 0.0, 1.0).astype(np.float32)
    if output == "cell_id":
        return cid.astype(np.float32)
    if output == "radial":
        return np.clip(1.0 - d_hex * 2.0, 0.0, 1.0).astype(np.float32)
    h_prof = np.clip((edge_dist - float(mortar)) / max(1e-4, float(bevel)), 0.0, 1.0)
    return (h_prof * (0.80 + 0.20 * cid)).astype(np.float32)


def tri_grid_pattern(
    height: int,
    width: int,
    *,
    cols: int = 6,
    mortar: float = 0.06,
    bevel: float = 0.14,
    seed: int = 0,
    output: str = "height",
) -> np.ndarray:
    """Seamless triangular / faceted crystal lattice."""
    u, v = uv_grid(height, width)
    c = max(1, int(cols))
    gx = u * c
    gy = v * c
    ix = np.floor(gx).astype(np.int64) % c
    iy = np.floor(gy).astype(np.int64) % c
    fx = gx - np.floor(gx)
    fy = gy - np.floor(gy)

    # Alternate diagonal orientation per checker square for symmetrical diamond-triangle facets
    flip = ((ix + iy) % 2) == 1
    lu = np.where(flip, 1.0 - fx, fx)
    upper = lu + fy < 1.0
    diag_dist = np.abs(lu + fy - 1.0) * 0.7071
    edge_d = np.where(
        upper,
        np.minimum(np.minimum(lu, fy), diag_dist),
        np.minimum(np.minimum(1.0 - lu, 1.0 - fy), diag_dist),
    )
    cid = _hash2i(ix * 2 + upper.astype(np.int64), iy, seed + 61)
    if output == "edge":
        return (edge_d < float(mortar)).astype(np.float32)
    if output == "cell_id":
        return cid.astype(np.float32)
    h_prof = np.clip((edge_d - float(mortar)) / max(1e-4, float(bevel)), 0.0, 1.0)
    return (h_prof * (0.75 + 0.25 * cid)).astype(np.float32)


def diamond_chevron_pattern(
    height: int,
    width: int,
    *,
    cols: int = 6,
    rows: int = 6,
    style: str = "diamond",
    mortar: float = 0.08,
    bevel: float = 0.14,
    seed: int = 0,
    output: str = "height",
) -> np.ndarray:
    """Diamond, Argyle, or Chevron repeating patterns."""
    u, v = uv_grid(height, width)
    c, r = max(1, int(cols)), max(1, int(rows))
    if style == "chevron":
        gx = (u * c) % 1.0
        zig = np.abs(2.0 * gx - 1.0)
        gy = (v * r + zig * 0.5)
        iy = np.floor(gy).astype(np.int64) % r
        fy = gy - np.floor(gy)
        edge_d = np.minimum(fy, 1.0 - fy)
        cid = _hash2i(np.zeros_like(iy), iy, seed + 101)
    else:
        # Seamless diamond / argyle via 45-deg rotated coordinates
        a = u * c + v * r
        b = u * c - v * r
        ia = np.floor(a).astype(np.int64) % (c + r)
        ib = np.floor(b).astype(np.int64) % (c + r)
        fa = a - np.floor(a)
        fb = b - np.floor(b)
        edge_d = np.minimum(np.minimum(fa, 1.0 - fa), np.minimum(fb, 1.0 - fb))
        cid = _hash2i(ia, ib, seed + 109)
        if style == "argyle":
            # Add thin crisp argyle cross-stitch highlight lines
            cross = np.minimum(np.abs(fa - 0.5), np.abs(fb - 0.5))
            stitch = np.clip(1.0 - cross / 0.06, 0.0, 1.0)
            if output == "edge":
                return np.maximum((edge_d < float(mortar)).astype(np.float32), stitch).astype(np.float32)

    if output == "edge":
        return (edge_d < float(mortar)).astype(np.float32)
    if output == "cell_id":
        return cid.astype(np.float32)
    h_prof = np.clip((edge_d - float(mortar)) / max(1e-4, float(bevel)), 0.0, 1.0)
    return (h_prof * (0.78 + 0.22 * cid)).astype(np.float32)


def truchet_pattern(
    height: int,
    width: int,
    *,
    cols: int = 6,
    style: str = "arcs",
    thickness: float = 0.22,
    seed: int = 0,
) -> np.ndarray:
    """Interlocking Truchet tile scrollwork / astrolabe filigree pattern."""
    u, v = uv_grid(height, width)
    c = max(1, int(cols))
    gx = u * c
    gy = v * c
    ix = np.floor(gx).astype(np.int64) % c
    iy = np.floor(gy).astype(np.int64) % c
    fx = gx - np.floor(gx)
    fy = gy - np.floor(gy)
    flip = _hash2i(ix, iy, seed + 151) > 0.5
    lu = np.where(flip, 1.0 - fx, fx)

    if style == "diagonal":
        d1 = np.abs(lu - fy) * 0.7071
        d = d1
    else:
        # Quarter-circle arcs centered at (0,0) and (1,1) with radius 0.5
        r0 = np.abs(np.sqrt(lu * lu + fy * fy) - 0.5)
        r1 = np.abs(np.sqrt((1.0 - lu) ** 2 + (1.0 - fy) ** 2) - 0.5)
        d = np.minimum(r0, r1)
    half_t = max(0.02, float(thickness) * 0.5)
    return np.clip(1.0 - d / half_t, 0.0, 1.0).astype(np.float32)


def mosaic_pattern(
    height: int,
    width: int,
    *,
    scale: float = 7.0,
    jitter: float = 0.85,
    grout: float = 0.12,
    bevel: float = 0.16,
    seed: int = 0,
    tileable: bool = True,
    output: str = "height",
) -> np.ndarray:
    """Voronoi cobblestone / stained-glass mosaic with crisp beveled tiles and dark grout."""
    edge = voronoi_noise(height, width, scale=scale, jitter=jitter, output="edge", seed=seed, tileable=tileable)
    cid = voronoi_noise(height, width, scale=scale, jitter=jitter, output="cell_id", seed=seed, tileable=tileable)
    if output == "cell_id":
        return cid
    if output == "grout":
        return (edge < float(grout)).astype(np.float32)
    h_prof = np.clip((edge - float(grout)) / max(1e-4, float(bevel)), 0.0, 1.0)
    return (h_prof * (0.72 + 0.28 * cid)).astype(np.float32)


def imbricated_scales_pattern(
    height: int,
    width: int,
    *,
    cols: int = 8,
    rows: int = 10,
    shape: str = "dragon",
    keel: float = 0.35,
    seed: int = 0,
    output: str = "height",
) -> np.ndarray:
    """Overlapping imbricated dragon / reptile / fish scales with cast lip shadow & central keel.

    Each scale overlaps the row below it, creating authentic layered 3D biological armor relief.
    """
    u, v = uv_grid(height, width)
    c = max(1, int(cols))
    r = max(1, int(rows))

    # Evaluate current row AND the overlapping row above it (rows overlap vertically!)
    def _eval_row(row_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gy = v * r - row_offset * 0.5
        iy = np.floor(gy).astype(np.int64)
        stagger = (iy % 2) * 0.5
        gx = u * c + stagger
        ix = np.floor(gx).astype(np.int64) % c
        fx = (gx - np.floor(gx)) - 0.5  # [-0.5, 0.5]
        fy = gy - iy  # [0, 1], 0 = root (top), 1 = overlapping lip tip (bottom)

        if shape == "fish":
            # Rounded semicircular arc
            rad = np.sqrt((fx * 2.0) ** 2 + (fy * 1.15) ** 2 + 1e-12)
            inside = rad <= 1.0
            rim_d = np.clip(1.0 - rad, 0.0, 1.0)
        elif shape == "diamond":
            rad = np.abs(fx) * 2.0 + np.abs(fy - 0.35) * 1.35
            inside = rad <= 1.0
            rim_d = np.clip(1.0 - rad, 0.0, 1.0)
        elif shape == "shield":
            # Broad armored osteoderm scute
            rad = np.maximum(np.abs(fx) * 2.1, np.abs(fy - 0.4) * 1.45) + 0.25 * (fx * fx * 4.0)
            inside = rad <= 1.0
            rim_d = np.clip(1.0 - rad, 0.0, 1.0)
        else:
            # Pointed gothic dragon scale (ogive tip)
            ax = np.abs(fx) * 2.0
            tip_taper = np.clip(1.0 - fy * 0.72, 0.15, 1.0)
            rad = np.sqrt((ax / tip_taper) ** 2 * 0.72 + (fy * 1.05) ** 2 + 1e-12)
            inside = (rad <= 1.0) & (fy >= 0.0) & (fy <= 1.0)
            rim_d = np.clip(1.0 - rad, 0.0, 1.0)

        cid = _hash2i(ix, iy % r, seed + 307)
        # Central raised keel ridge + domed body + crisp rim bevel + dark overlap shadow near top root
        keel_ridge = float(keel) * np.clip(1.0 - np.abs(fx) * 6.0, 0.0, 1.0) * (1.0 - 0.3 * fy)
        dome = np.sqrt(np.clip(rim_d, 0.0, 1.0)) * 0.65
        root_ao = np.clip(fy * 1.65, 0.22, 1.0)
        lip_highlight = np.clip(1.0 - np.abs(rim_d - 0.12) * 8.0, 0.0, 1.0) * 0.18 * fy
        h_val = np.clip((dome + keel_ridge + lip_highlight) * root_ao * (0.85 + 0.15 * cid), 0.0, 1.0)
        return inside, h_val, cid, rim_d

    in0, h0, cid0, rim0 = _eval_row(0)
    in1, h1, cid1, rim1 = _eval_row(1)

    # Row 1 (upper overlapping scale) takes precedence where inside
    h_final = np.where(in1, h1, np.where(in0, h0, 0.08))
    cid_final = np.where(in1, cid1, cid0)
    rim_final = np.where(in1, rim1, rim0)

    if output == "cell_id":
        return cid_final.astype(np.float32)
    if output == "edge":
        return np.clip(1.0 - rim_final * 4.5, 0.0, 1.0).astype(np.float32)
    return np.clip(h_final, 0.0, 1.0).astype(np.float32)


def feather_barbs_pattern(
    height: int,
    width: int,
    *,
    cols: int = 4,
    barb_freq: float = 28.0,
    barb_angle_deg: float = 32.0,
    rachis_width: float = 0.10,
    seed: int = 0,
) -> np.ndarray:
    """Structured feather vane with central quill rachis and angled parallel barbules."""
    u, v = uv_grid(height, width)
    c = max(1, int(cols))
    gx = u * c
    ix = np.floor(gx).astype(np.int64) % c
    fx = (gx - np.floor(gx)) - 0.5  # [-0.5, 0.5], 0 = central quill
    ax = np.abs(fx)

    slope = math.tan(math.radians(float(barb_angle_deg)))
    barb_coord = (v * float(barb_freq) - ax * slope * float(barb_freq) * 0.35)
    barbs = 0.5 + 0.5 * np.sin(2.0 * math.pi * barb_coord)
    # Soft organic notch breaks across barbules
    notch = fbm_noise(height, width, scale=max(4, c * 3), octaves=3, seed=seed + 419, tileable=True)
    vane = (0.55 + 0.45 * barbs) * (0.75 + 0.25 * notch) * np.clip(1.0 - ax * 1.4, 0.2, 1.0)
    # Raised central quill rachis
    rachis = np.clip(1.0 - ax / max(0.02, float(rachis_width) * 0.5), 0.0, 1.0)
    cid = _hash2i(ix, np.zeros_like(ix), seed + 431)
    return np.clip(np.maximum(vane * (0.85 + 0.15 * cid), rachis * 0.95), 0.0, 1.0).astype(np.float32)


def fur_tufts_pattern(
    height: int,
    width: int,
    *,
    cols: int = 10,
    rows: int = 8,
    droop: float = 0.35,
    seed: int = 0,
) -> np.ndarray:
    """Overlapping painterly V-shaped fur/plumage clumps with root-to-tip shading."""
    u, v = uv_grid(height, width)
    c, r = max(1, int(cols)), max(1, int(rows))
    warp = (fbm_noise(height, width, scale=max(3, c // 2), octaves=2, seed=seed + 503, tileable=True) - 0.5) * float(droop)
    gx = (u + warp * 0.15) * c
    gy = v * r
    iy = np.floor(gy).astype(np.int64)
    stagger = (iy % 2) * 0.5
    gx = gx + stagger
    ix = np.floor(gx).astype(np.int64) % c
    fx = (gx - np.floor(gx)) - 0.5
    fy = gy - iy

    # Tapered shaggy tuft silhouette
    width_taper = np.clip(1.0 - fy * 0.85, 0.08, 1.0)
    tuft = np.clip(1.0 - (np.abs(fx) * 2.0 / width_taper), 0.0, 1.0)
    # Root dark AO -> bright mid-strand -> crisp tip
    strand = np.sqrt(tuft) * (0.35 + 0.65 * fy)
    cid = _hash2i(ix, iy % r, seed + 521)
    return np.clip(strand * (0.75 + 0.25 * cid), 0.0, 1.0).astype(np.float32)


def radial_rings_pattern(
    height: int,
    width: int,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
    freq: float = 6.0,
    spokes: int = 0,
    phase: float = 0.0,
    loop_speed: float = 0.0,
) -> np.ndarray:
    """Concentric astrolabe rings and optional radial spokes (animatable via `phase`)."""
    u, v = uv_grid(height, width)
    dx, dy = u - float(cx), v - float(cy)
    r = np.sqrt(dx * dx + dy * dy)
    rings = 0.5 + 0.5 * np.cos(2.0 * math.pi * (r * float(freq) - float(phase) * float(loop_speed)))
    if int(spokes) > 0:
        ang = np.arctan2(dy, dx)
        sp = 0.5 + 0.5 * np.cos(int(spokes) * ang + 2.0 * math.pi * float(phase) * float(loop_speed))
        return np.clip(np.maximum(rings * 0.8, sp * 0.7), 0.0, 1.0).astype(np.float32)
    return rings.astype(np.float32)


def energy_veins_pattern(
    height: int,
    width: int,
    *,
    scale: float = 5.0,
    thickness: float = 0.16,
    warp_strength: float = 0.10,
    style: str = "lightning",
    seed: int = 0,
    tileable: bool = True,
    phase: float = 0.0,
    flow_speed: float = 1.0,
) -> np.ndarray:
    """Glowing neon / dragon-gas / magma / circuit energy veins with smooth periodic pulse."""
    if style == "circuit":
        # Crisp orthogonal Manhattan Voronoi conduit traces
        cr = voronoi_noise(height, width, scale=scale, jitter=0.55, metric="manhattan", output="cracks", seed=seed, tileable=tileable)
        base = np.clip((cr - (1.0 - float(thickness) * 1.4)) / max(1e-4, float(thickness) * 1.4), 0.0, 1.0)
    elif style == "nebula":
        f = fbm_noise(height, width, scale=scale, octaves=4, mode="ridged", seed=seed, tileable=tileable, phase=phase, loop_speed=flow_speed)
        base = domain_warp(f, strength=warp_strength, scale=max(2.0, scale * 0.6), mode="curl", seed=seed + 31, tileable=tileable, phase=phase, flow_speed=flow_speed)
        base = np.power(np.clip(base, 0.0, 1.0), 2.2)
    else:
        # Sharp branching plasma / lightning cracks
        cr = voronoi_noise(height, width, scale=scale, jitter=0.90, metric="euclidean", output="cracks", seed=seed, tileable=tileable)
        if warp_strength > 1e-4:
            cr = domain_warp(cr, strength=warp_strength, scale=max(2.0, scale * 0.8), mode="fbm", seed=seed + 47, tileable=tileable, phase=phase, flow_speed=flow_speed)
        thr = max(0.02, 1.0 - float(thickness) * 1.5)
        base = np.clip((cr - thr) / max(1e-4, 1.0 - thr), 0.0, 1.0)

    # Periodic traveling energy wave along veins for seamless animated loops
    if abs(float(flow_speed)) > 1e-6:
        u, v = uv_grid(height, width)
        wave_field = fbm_noise(height, width, scale=max(2, int(round(scale * 0.5))), octaves=2, seed=seed + 99, tileable=tileable)
        pulse = 0.62 + 0.38 * np.sin(2.0 * math.pi * (wave_field * 1.5 + u - float(phase) * float(flow_speed)))
        base = base * pulse

    return np.clip(base, 0.0, 1.0).astype(np.float32)
