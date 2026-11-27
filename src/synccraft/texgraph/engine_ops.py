"""Industry-Standard Procedural Texture Operators (Substance Designer + Pixel Composer Parity).

Pure-algorithmic NumPy operators powering `TexGraph v7` across Blocks, Items & Entities:
- ``sdf_shape`` & ``sdf_boolean``  : Analytical 2D signed-distance primitives & smooth booleans
                                     (icons, items, weapons, potions, shields, gems, leaves, scales, stones).
- ``tile_sampler``                 : Substance Tile Sampler / Pixel Composer Scatter & Splatter
                                     with grid/Poisson-jitter, scale/rotation/luminance jitter,
                                     external driver maps, and 100% seamless toroidal wrapping.
- ``flood_fill``                   : Fast connected-component island labeling -> per-island
                                     random grayscale, UV gradients, random tilt slopes & edge distance.
- ``slope_blur``                   : Substance Slope Blur (blur / min-erode / max-dilate along a
                                     secondary noise gradient for chipped stone, impasto & organic wear).
- ``histogram_scan`` / ``histogram_select`` / ``auto_levels`` : Precision tonal/mask slicing.
- ``curvature_from_height`` & ``edge_wear_mask``              : Multi-scale convex/concave curvature & wear.
- ``displace_uv`` & ``normal_light_2d``                       : Pixel Composer Displace & 2D Normal Light.
- ``directional_selout``                                      : Brandon James Greer directional selective outline.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .generators_noise import fbm_noise, uv_grid
from .ops_fx_color import gaussian_blur
from .pixel_art_ops import hex_to_srgb


# ---------------------------------------------------------------------------
# 1. Analytical 2D SDF Primitives & Booleans
# ---------------------------------------------------------------------------


def _rot2(px: np.ndarray, py: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    if abs(angle_deg) < 1e-6:
        return px, py
    rad = math.radians(float(angle_deg))
    c, s = math.cos(rad), math.sin(rad)
    return px * c - py * s, px * s + py * c


def _sd_box(px: np.ndarray, py: np.ndarray, bx: float, by: float) -> np.ndarray:
    qx = np.abs(px) - bx
    qy = np.abs(py) - by
    out = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
    inn = np.minimum(np.maximum(qx, qy), 0.0)
    return (out + inn).astype(np.float32)


def _sd_capsule(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float, r: float) -> np.ndarray:
    pax, pay = px - ax, py - ay
    bax, bay = bx - ax, by - ay
    denom = max(1e-8, bax * bax + bay * bay)
    h = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0)
    dx = pax - bax * h
    dy = pay - bay * h
    return (np.sqrt(dx * dx + dy * dy) - r).astype(np.float32)


def _sd_smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    if k <= 1e-5:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return (b * (1.0 - h) + a * h - k * h * (1.0 - h)).astype(np.float32)


def sdf_shape(
    height: int,
    width: int,
    *,
    shape: str = "rounded_box",
    center: tuple[float, float] = (0.5, 0.5),
    size: tuple[float, float] = (0.34, 0.34),
    corner_radius: float = 0.06,
    angle_deg: float = 0.0,
    sides: int = 6,
    inner_ratio: float = 0.48,
    bevel_width: float = 0.14,
    outline_width: float = 0.04,
) -> dict[str, np.ndarray]:
    """Analytical 2D Signed Distance Field shape generator (18+ primitives & item silhouettes).

    Supported ``shape`` values:
      - Geometric: ``circle``, ``box``, ``rounded_box``, ``polygon``, ``star``, ``capsule``,
        ``crescent``, ``teardrop``, ``gear``
      - Natural / Bio: ``leaf``, ``feather``, ``scale``
      - Minecraft Item Forges: ``sword``, ``axe``, ``pickaxe``, ``potion``, ``shield``, ``gem``, ``book``, ``apple``

    Returns multi-port dict: ``{"sdf", "mask", "bevel", "outline"}``.
    """
    u, v = uv_grid(height, width)
    px0 = (u - float(center[0])).astype(np.float32)
    py0 = (v - float(center[1])).astype(np.float32)
    px, py = _rot2(px0, py0, angle_deg)
    sx, sy = max(1e-3, float(size[0])), max(1e-3, float(size[1]))
    cr = float(np.clip(corner_radius, 0.0, min(sx, sy) * 0.95))

    sh = shape.lower()
    if sh == "circle":
        d = np.sqrt((px / sx) ** 2 + (py / sy) ** 2) * min(sx, sy) - min(sx, sy)
    elif sh == "box":
        d = _sd_box(px, py, sx, sy)
    elif sh == "rounded_box":
        d = _sd_box(px, py, max(1e-4, sx - cr), max(1e-4, sy - cr)) - cr
    elif sh == "capsule":
        d = _sd_capsule(px, py, 0.0, -sy * 0.65, 0.0, sy * 0.65, sx * 0.55)
    elif sh == "polygon":
        n = max(3, int(sides))
        ang = np.arctan2(py, px)
        r = np.sqrt(px * px + py * py)
        sec = 2.0 * math.pi / n
        a_loc = ((ang + math.pi) % sec) - sec * 0.5
        rad_n = min(sx, sy) * math.cos(sec * 0.5)
        d = r * np.cos(a_loc) - rad_n
    elif sh == "star":
        n = max(3, int(sides))
        ang = np.arctan2(py, px)
        r = np.sqrt(px * px + py * py)
        sec = 2.0 * math.pi / n
        a_loc = np.abs(((ang + math.pi) % sec) - sec * 0.5) / (sec * 0.5)
        r_bound = min(sx, sy) * (float(inner_ratio) + (1.0 - float(inner_ratio)) * (1.0 - a_loc))
        d = (r - r_bound) * 0.85
    elif sh == "crescent":
        r1 = min(sx, sy)
        d1 = np.sqrt(px * px + py * py) - r1
        d2 = np.sqrt((px - r1 * 0.42) ** 2 + (py + r1 * 0.14) ** 2) - r1 * 0.84
        d = np.maximum(d1, -d2)
    elif sh == "teardrop":
        # Smooth organic droplet/flame pointing upward (-y)
        ny = (py / sy)
        nx = (px / sx)
        w_taper = np.clip(0.55 + 0.45 * ny, 0.08, 1.0)
        d = (np.sqrt((nx / w_taper) ** 2 + (ny - 0.15) ** 2) - 0.78) * min(sx, sy)
    elif sh == "leaf":
        # Pointed botanical leaf (two intersecting circular arcs)
        nx = px / sx
        ny = py / sy
        c1 = np.sqrt((nx - 0.65) ** 2 + ny * ny) - 1.22
        c2 = np.sqrt((nx + 0.65) ** 2 + ny * ny) - 1.22
        d = np.maximum(c1, c2) * min(sx, sy)
    elif sh == "feather":
        nx = px / sx
        ny = py / sy
        body = (np.sqrt((nx / np.clip(0.95 - 0.35 * np.abs(ny), 0.2, 1.0)) ** 2 + (ny * 0.92) ** 2) - 0.82) * min(sx, sy)
        quill = _sd_capsule(px, py, 0.0, sy * 0.35, 0.0, sy * 1.15, sx * 0.12)
        d = np.minimum(body, quill)
    elif sh == "scale":
        # Rounded overlapping dragon/fish shield scale (pointed bottom tip)
        nx = px / sx
        ny = py / sy
        w_sc = np.clip(1.0 - 0.48 * np.maximum(0.0, ny) ** 1.4, 0.12, 1.0)
        d = (np.sqrt((nx / w_sc) ** 2 + (ny * 0.95) ** 2) - 0.88) * min(sx, sy)
    elif sh == "gear":
        n = max(6, int(sides))
        r = np.sqrt(px * px + py * py)
        ang = np.arctan2(py, px)
        tooth = 0.5 + 0.5 * np.cos(ang * n)
        outer = r - min(sx, sy) * (0.76 + 0.24 * np.clip((tooth - 0.35) * 4.0, 0.0, 1.0))
        hole = -(r - min(sx, sy) * 0.32)
        d = np.maximum(outer, hole)
    elif sh == "sword":
        # Classic diagonal Minecraft sword (bottom-left pommel -> top-right razor tip!)
        lx, ly = _rot2(px, py, -45.0)
        blade_w = np.clip(0.088 - np.maximum(0.0, (-ly - 0.25)) * 0.30, 0.015, 0.095)
        blade = np.maximum(np.abs(lx) - blade_w, np.maximum(-ly - 0.54, ly - 0.12))
        tip = (np.abs(lx) * 1.25 + (-ly - 0.45)) * 0.72
        blade = np.maximum(blade, tip)
        guard = _sd_box(lx, ly - 0.14, 0.21, 0.050) - 0.014
        grip = _sd_capsule(lx, ly, 0.0, 0.16, 0.0, 0.39, 0.042)
        pommel = (np.abs(lx) + np.abs(ly - 0.44)) * 0.72 - 0.062
        d = np.minimum(np.minimum(blade, guard), np.minimum(grip, pommel))
    elif sh == "axe":
        lx, ly = _rot2(px, py, -45.0)
        haft = _sd_capsule(lx, ly, 0.0, -0.36, 0.0, 0.44, 0.042)
        pommel = np.sqrt(lx * lx + (ly - 0.44) ** 2) - 0.060
        head_outer = np.sqrt((lx - 0.02) ** 2 + (ly + 0.24) ** 2) - 0.25
        head_clip = _sd_box(lx - 0.12, ly + 0.23, 0.18, 0.22)
        head_cut = -(np.sqrt((lx - 0.09) ** 2 + (ly + 0.02) ** 2) - 0.12)
        blade = np.maximum(np.maximum(head_outer, head_clip), head_cut)
        back_spike = _sd_box(lx + 0.08, ly + 0.24, 0.085, 0.060) - 0.012
        d = np.minimum(np.minimum(haft, pommel), np.minimum(blade, back_spike))
    elif sh == "pickaxe":
        lx, ly = _rot2(px, py, -45.0)
        haft = _sd_capsule(lx, ly, 0.0, -0.30, 0.0, 0.44, 0.044)
        # Bold curved crescent forged pick head (reads super clean at both 16x16 and 32x32!)
        arc_c = np.abs(np.sqrt(lx * lx + (ly - 0.16) ** 2) - 0.44) - 0.066
        arc_clip = np.maximum(ly + 0.04, np.abs(lx) - 0.35)
        pick_head = np.maximum(arc_c, arc_clip)
        collar = _sd_box(lx, ly + 0.27, 0.090, 0.070) - 0.012
        d = np.minimum(np.minimum(haft, collar), pick_head)
    elif sh == "potion":
        bulb = np.sqrt((px * 0.96) ** 2 + ((py - 0.11) * 1.05) ** 2) - 0.27
        base_flat = (py - 0.36)
        bulb = np.maximum(bulb, base_flat)
        neck = _sd_box(px, py + 0.17, 0.088, 0.12) - 0.015
        lip = _sd_box(px, py + 0.26, 0.130, 0.038) - 0.012
        cork = _sd_box(px, py + 0.34, 0.076, 0.056) - 0.010
        d = np.minimum(_sd_smooth_min(bulb, neck, 0.04), np.minimum(lip, cork))
    elif sh == "shield":
        nx = px / 0.32
        ny = (py + 0.02) / 0.38
        taper = np.where(ny > 0.0, np.clip(1.0 - 0.82 * (np.maximum(0.0, ny) ** 1.65), 0.04, 1.0), 1.0)
        top_arch = -0.10 * (1.0 - np.clip(nx * nx, 0.0, 1.0))
        d_x = np.abs(nx) - taper
        d_y = np.maximum(-(ny + 0.88 + top_arch), ny - 1.02)
        d = np.maximum(d_x, d_y) * 0.32
    elif sh == "gem":
        # Faceted emerald/starlight octagonal gemstone!
        nx = np.abs(px) / 0.26
        ny = np.abs(py) / 0.34
        d = (np.maximum(np.maximum(nx, ny), (nx + ny) * 0.72) - 1.0) * 0.28
    elif sh == "book":
        # Slightly angled isometric-feel spellbook with spine & clasp
        lx, ly = _rot2(px, py, -10.0)
        cover = _sd_box(lx, ly, 0.26, 0.32) - 0.035
        clasp = _sd_box(lx - 0.25, ly, 0.05, 0.075) - 0.012
        d = np.minimum(cover, clasp)
    elif sh == "apple":
        # Plump studio hearth apple + curved stem + top leaf!
        lobe_l = np.sqrt((px + 0.085) ** 2 + ((py - 0.06) * 0.96) ** 2) - 0.22
        lobe_r = np.sqrt((px - 0.085) ** 2 + ((py - 0.06) * 0.96) ** 2) - 0.22
        body = _sd_smooth_min(lobe_l, lobe_r, 0.08)
        dimple = -(np.sqrt(px * px + (py + 0.21) ** 2) - 0.09)
        body = np.maximum(body, dimple * 0.55)
        stem = _sd_capsule(px, py, 0.0, -0.14, 0.05, -0.34, 0.026)
        leaf = _sd_capsule(px, py, 0.04, -0.26, 0.19, -0.33, 0.042)
        d = np.minimum(body, np.minimum(stem, leaf))
    else:
        raise ValueError(f"Unknown sdf_shape: {shape!r}")

    sdf = d.astype(np.float32)
    mask = (sdf <= 0.0).astype(np.float32)
    bw = max(1e-4, float(bevel_width))
    bevel = np.clip(-sdf / bw, 0.0, 1.0) * mask
    ow = max(1e-4, float(outline_width))
    outline = ((sdf <= 0.0) & (sdf >= -ow)).astype(np.float32)
    return {
        "sdf": sdf,
        "mask": mask,
        "bevel": bevel.astype(np.float32),
        "outline": outline,
    }


def sdf_boolean(
    a: np.ndarray,
    b: np.ndarray,
    *,
    op: str = "smooth_union",
    smoothness: float = 0.04,
) -> dict[str, np.ndarray]:
    """Boolean combination of two 2D SDF fields (union, subtract, intersect, xor + smooth variants)."""
    fa = np.asarray(a, dtype=np.float32)
    fb = np.asarray(b, dtype=np.float32)
    k = max(0.0, float(smoothness))
    mode = op.lower()
    if mode == "union":
        d = np.minimum(fa, fb)
    elif mode == "subtract":
        d = np.maximum(fa, -fb)
    elif mode == "intersect":
        d = np.maximum(fa, fb)
    elif mode == "xor":
        d = np.maximum(np.minimum(fa, fb), -np.maximum(fa, fb))
    elif mode == "smooth_union":
        d = _sd_smooth_min(fa, fb, k)
    elif mode == "smooth_subtract":
        d = -_sd_smooth_min(-fa, fb, k)
    elif mode == "smooth_intersect":
        d = -_sd_smooth_min(-fa, -fb, k)
    else:
        raise ValueError(f"Unsupported sdf_boolean op: {op!r}")
    mask = (d <= 0.0).astype(np.float32)
    bevel = np.clip(-d / max(0.02, 0.14), 0.0, 1.0) * mask
    return {"sdf": d.astype(np.float32), "mask": mask, "bevel": bevel.astype(np.float32)}


# ---------------------------------------------------------------------------
# 2. Substance Tile Sampler / Pixel Composer Scatter & Splatter
# ---------------------------------------------------------------------------


def tile_sampler(
    height: int,
    width: int,
    stamp: np.ndarray | None = None,
    *,
    count_x: int = 5,
    count_y: int = 5,
    distribution: str = "jitter_grid",
    pos_jitter: float = 0.38,
    scale_base: float = 0.92,
    scale_jitter: float = 0.28,
    rot_base_deg: float = 0.0,
    rot_jitter_deg: float = 25.0,
    lum_jitter: float = 0.30,
    density: float = 1.0,
    density_map: np.ndarray | None = None,
    scale_map: np.ndarray | None = None,
    rot_map: np.ndarray | None = None,
    blend_mode: str = "max",
    tileable: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Industry-standard Substance Tile Sampler / Pixel Composer Scatter powerhouse.

    Stamps any scalar height/mask OR RGBA sprite across ``(height, width)`` with deterministic
    grid or jittered scatter, per-stamp scale/rotation/luminance variation, optional external
    density/scale/rotation driver maps, and **100% seamless toroidal wrapping** when ``tileable=True``!

    Returns dict: ``{"height", "mask", "id_map", "rgba"}``.
    """
    H, W = int(height), int(width)
    rng = np.random.default_rng(int(seed))

    if stamp is None:
        # Default organic rounded cobblestone/pebble height dome
        st = sdf_shape(32, 32, shape="rounded_box", size=(0.36, 0.32), corner_radius=0.14, bevel_width=0.22)
        stamp_arr = np.power(st["bevel"], 0.65).astype(np.float32)
        is_rgba = False
    else:
        stamp_arr = np.asarray(stamp, dtype=np.float32)
        is_rgba = stamp_arr.ndim == 3 and stamp_arr.shape[2] >= 3
        if is_rgba and stamp_arr.shape[2] == 3:
            stamp_arr = np.concatenate((stamp_arr, np.ones((*stamp_arr.shape[:2], 1), dtype=np.float32)), axis=-1)

    sh, sw = stamp_arr.shape[:2]
    nx = max(1, int(count_x))
    ny = max(1, int(count_y))

    out_h = np.zeros((H, W), dtype=np.float32)
    out_m = np.zeros((H, W), dtype=np.float32)
    out_id = np.zeros((H, W), dtype=np.float32)
    out_rgba = np.zeros((H, W, 4), dtype=np.float32)

    cell_w = W / float(nx)
    cell_h = H / float(ny)

    # Generate stamp centers in [0, W) x [0, H)
    pts: list[tuple[float, float]] = []
    if distribution == "random":
        total = max(1, nx * ny)
        for _ in range(total):
            pts.append((float(rng.random() * W), float(rng.random() * H)))
    else:
        stagger = 0.5 if distribution == "stagger_grid" else 0.0
        for iy in range(ny):
            for ix in range(nx):
                cx = (ix + 0.5 + (iy % 2) * stagger + (rng.random() - 0.5) * 2.0 * float(pos_jitter)) * cell_w
                cy = (iy + 0.5 + (rng.random() - 0.5) * 2.0 * float(pos_jitter)) * cell_h
                if tileable:
                    cx %= W
                    cy %= H
                pts.append((cx, cy))

    n_pts = max(1, len(pts))
    for idx, (cx, cy) in enumerate(pts):
        px_i = int(round(cx)) % W
        py_i = int(round(cy)) % H
        local_dens = float(density)
        if density_map is not None:
            local_dens *= float(np.clip(density_map[py_i, px_i], 0.0, 1.0))
        if rng.random() > local_dens:
            continue

        sc = float(scale_base) * (1.0 + (rng.random() - 0.5) * 2.0 * float(scale_jitter))
        if scale_map is not None:
            sc *= 0.4 + 1.2 * float(np.clip(scale_map[py_i, px_i], 0.0, 1.0))
        sc = max(0.15, sc)

        ang = float(rot_base_deg) + (rng.random() - 0.5) * 2.0 * float(rot_jitter_deg)
        if rot_map is not None:
            ang += (float(rot_map[py_i, px_i]) - 0.5) * 360.0
        lum = float(np.clip(1.0 + (rng.random() - 0.5) * 2.0 * float(lum_jitter), 0.25, 1.35))
        sid = float((idx + 1) / float(n_pts + 1))

        # Bounding patch in output pixels
        rad_x = max(1, int(math.ceil(cell_w * sc * 0.78)))
        rad_y = max(1, int(math.ceil(cell_h * sc * 0.78)))
        dy = np.arange(-rad_y, rad_y + 1, dtype=np.float32)
        dx = np.arange(-rad_x, rad_x + 1, dtype=np.float32)
        gy, gx = np.meshgrid(dy, dx, indexing="ij")

        # Inverse rotate & normalize into stamp coordinates [-0.5, 0.5]
        rad = math.radians(-ang)
        ca, sa = math.cos(rad), math.sin(rad)
        rx = (gx * ca - gy * sa) / max(1e-3, cell_w * sc)
        ry = (gx * sa + gy * ca) / max(1e-3, cell_h * sc)
        su = rx + 0.5
        sv = ry + 0.5
        valid = (su >= 0.0) & (su <= 1.0) & (sv >= 0.0) & (sv <= 1.0)
        if not np.any(valid):
            continue

        # Bilinear sample from stamp
        sx_f = np.clip(su * (sw - 1), 0.0, sw - 1.001)
        sy_f = np.clip(sv * (sh - 1), 0.0, sh - 1.001)
        x0 = np.floor(sx_f).astype(np.int32)
        y0 = np.floor(sy_f).astype(np.int32)
        x1 = np.minimum(x0 + 1, sw - 1)
        y1 = np.minimum(y0 + 1, sh - 1)
        tx = (sx_f - x0).astype(np.float32)
        ty = (sy_f - y0).astype(np.float32)

        ox = np.rint(cx + gx).astype(np.int32)
        oy = np.rint(cy + gy).astype(np.int32)
        if tileable:
            ox %= W
            oy %= H
            in_bounds = valid
        else:
            in_bounds = valid & (ox >= 0) & (ox < W) & (oy >= 0) & (oy < H)

        if not np.any(in_bounds):
            continue

        oy_v = oy[in_bounds]
        ox_v = ox[in_bounds]
        y0_v, y1_v, x0_v, x1_v = y0[in_bounds], y1[in_bounds], x0[in_bounds], x1[in_bounds]
        tx_v, ty_v = tx[in_bounds], ty[in_bounds]

        if is_rgba:
            c00 = stamp_arr[y0_v, x0_v]
            c10 = stamp_arr[y0_v, x1_v]
            c01 = stamp_arr[y1_v, x0_v]
            c11 = stamp_arr[y1_v, x1_v]
            txe, tye = tx_v[:, None], ty_v[:, None]
            samp = (c00 * (1 - txe) + c10 * txe) * (1 - tye) + (c01 * (1 - txe) + c11 * txe) * tye
            a_v = np.clip(samp[:, 3], 0.0, 1.0)
            rgb_v = np.clip(samp[:, :3] * lum, 0.0, 1.0)
            h_v = a_v * (0.299 * rgb_v[:, 0] + 0.587 * rgb_v[:, 1] + 0.114 * rgb_v[:, 2])
            if blend_mode == "alpha":
                cur = out_rgba[oy_v, ox_v]
                out_rgba[oy_v, ox_v, :3] = cur[:, :3] * (1.0 - a_v[:, None]) + rgb_v * a_v[:, None]
                out_rgba[oy_v, ox_v, 3] = np.clip(cur[:, 3] + a_v * (1.0 - cur[:, 3]), 0.0, 1.0)
                out_h[oy_v, ox_v] = np.maximum(out_h[oy_v, ox_v], h_v)
                out_m[oy_v, ox_v] = np.maximum(out_m[oy_v, ox_v], a_v)
                win = a_v > 0.25
                out_id[oy_v[win], ox_v[win]] = sid
            else:
                win = h_v > out_h[oy_v, ox_v]
                out_h[oy_v[win], ox_v[win]] = h_v[win]
                out_m[oy_v[win], ox_v[win]] = np.maximum(out_m[oy_v[win], ox_v[win]], a_v[win])
                out_id[oy_v[win], ox_v[win]] = sid
                out_rgba[oy_v[win], ox_v[win], :3] = rgb_v[win]
                out_rgba[oy_v[win], ox_v[win], 3] = a_v[win]
        else:
            s00 = stamp_arr[y0_v, x0_v]
            s10 = stamp_arr[y0_v, x1_v]
            s01 = stamp_arr[y1_v, x0_v]
            s11 = stamp_arr[y1_v, x1_v]
            samp = (s00 * (1 - tx_v) + s10 * tx_v) * (1 - ty_v) + (s01 * (1 - tx_v) + s11 * tx_v) * ty_v
            val = np.clip(samp * lum, 0.0, 1.0)
            if blend_mode == "add":
                out_h[oy_v, ox_v] = np.clip(out_h[oy_v, ox_v] + val, 0.0, 1.0)
                out_m[oy_v, ox_v] = np.maximum(out_m[oy_v, ox_v], (samp > 0.05).astype(np.float32))
                win = samp > 0.2
                out_id[oy_v[win], ox_v[win]] = sid
            else:
                win = val > out_h[oy_v, ox_v]
                out_h[oy_v[win], ox_v[win]] = val[win]
                out_m[oy_v[win], ox_v[win]] = np.maximum(out_m[oy_v[win], ox_v[win]], (samp[win] > 0.05).astype(np.float32))
                out_id[oy_v[win], ox_v[win]] = sid

    if not is_rgba:
        out_rgba[..., :3] = out_h[..., None]
        out_rgba[..., 3] = out_m
    return {
        "height": np.clip(out_h, 0.0, 1.0).astype(np.float32),
        "mask": np.clip(out_m, 0.0, 1.0).astype(np.float32),
        "id_map": out_id.astype(np.float32),
        "rgba": np.clip(out_rgba, 0.0, 1.0).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# 3. Substance Flood Fill (Connected-Component Island Attributes!)
# ---------------------------------------------------------------------------


def flood_fill(
    mask: np.ndarray,
    *,
    threshold: float = 0.25,
    tileable: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Substance Designer Flood Fill: labels every isolated tile/stone/plank/scale island and
    generates per-island attributes (works seamlessly across wrapped borders when ``tileable=True``!).

    Outputs:
      - ``island_id``   : normalized unique island ID in (0, 1] (0 in mortar/background)
      - ``random_gray`` : deterministic uniform random luminance per island in [0, 1]
      - ``gradient_u``  : local horizontal 0->1 ramp inside each island
      - ``gradient_v``  : local vertical 0->1 ramp inside each island
      - ``random_slope``: random directional tilt plane per island (makes cobbles/tiles catch varied light!)
      - ``edge_dist``   : normalized Manhattan/chamfer distance from each island's border [0, 1]
    """
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    H, W = m.shape
    solid = m >= float(threshold)

    # Fast pure-Python/NumPy Union-Find 4-connected labeling (handles up to 512x512 in a few ms!)
    N = H * W
    parent = np.arange(N, dtype=np.int32)
    solid_flat = solid.ravel()

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Vectorized neighbor pairs for speed
    ys, xs = np.nonzero(solid)
    if ys.size == 0:
        z = np.zeros((H, W), dtype=np.float32)
        return {"island_id": z, "random_gray": z, "gradient_u": z, "gradient_v": z, "random_slope": z, "edge_dist": z}

    # Horizontal & vertical edges
    r_mask = solid[:, :-1] & solid[:, 1:]
    ry, rx = np.nonzero(r_mask)
    for y, x in zip(ry.tolist(), rx.tolist()):
        _union(y * W + x, y * W + (x + 1))

    d_mask = solid[:-1, :] & solid[1:, :]
    dy, dx = np.nonzero(d_mask)
    for y, x in zip(dy.tolist(), dx.tolist()):
        _union(y * W + x, (y + 1) * W + x)

    if tileable:
        wrap_x = solid[:, 0] & solid[:, W - 1]
        for y in np.nonzero(wrap_x)[0].tolist():
            _union(y * W, y * W + (W - 1))
        wrap_y = solid[0, :] & solid[H - 1, :]
        for x in np.nonzero(wrap_y)[0].tolist():
            _union(x, (H - 1) * W + x)

    # Path compress all solid pixels & compact to contiguous 1..K labels
    labels = np.zeros(N, dtype=np.int32)
    root_map: dict[int, int] = {}
    nxt = 1
    for idx in (ys * W + xs).tolist():
        r = _find(idx)
        lid = root_map.get(r)
        if lid is None:
            lid = nxt
            root_map[r] = lid
            nxt += 1
        labels[idx] = lid
    lbl = labels.reshape(H, W)
    num_islands = nxt - 1

    # Deterministic per-island hash -> random gray & random tilt direction
    ids = np.arange(num_islands + 1, dtype=np.uint32)
    h1 = (ids * np.uint32(747796405) + np.uint32(seed * 2891336453 & 0xFFFFFFFF))
    h1 = ((h1 >> ((h1 >> np.uint32(28)) + np.uint32(4))) ^ h1) * np.uint32(277803737)
    h1 = h1 ^ (h1 >> np.uint32(22))
    rand_lut = (h1 & np.uint32(0xFFFF)).astype(np.float32) / 65535.0
    rand_lut[0] = 0.0

    h2 = ((ids ^ np.uint32(0x9E3779B9)) * np.uint32(2654435761) + np.uint32(seed * 1597334677 & 0xFFFFFFFF))
    ang_lut = ((h2 & np.uint32(0xFFFF)).astype(np.float32) / 65535.0) * (2.0 * math.pi)

    random_gray = rand_lut[lbl]
    island_id = (lbl.astype(np.float32) / float(max(1, num_islands)))

    # Per-island local bounding-box UV (unwrapped cleanly for toroidal islands!)
    grad_u = np.zeros((H, W), dtype=np.float32)
    grad_v = np.zeros((H, W), dtype=np.float32)
    rand_slope = np.zeros((H, W), dtype=np.float32)

    for k in range(1, num_islands + 1):
        ky, kx = np.nonzero(lbl == k)
        if ky.size == 0:
            continue
        # Unwrap across toroidal edge if island touches both 0 and W-1 / H-1
        x_f = kx.astype(np.float32)
        y_f = ky.astype(np.float32)
        if tileable and (kx.min() == 0 and kx.max() == W - 1):
            x_f = np.where(x_f < W * 0.5, x_f + W, x_f)
        if tileable and (ky.min() == 0 and ky.max() == H - 1):
            y_f = np.where(y_f < H * 0.5, y_f + H, y_f)
        x0, x1 = float(x_f.min()), float(x_f.max())
        y0, y1 = float(y_f.min()), float(y_f.max())
        lu = (x_f - x0) / max(1.0, x1 - x0)
        lv = (y_f - y0) / max(1.0, y1 - y0)
        grad_u[ky, kx] = lu
        grad_v[ky, kx] = lv
        ca, sa = math.cos(float(ang_lut[k])), math.sin(float(ang_lut[k]))
        sl = 0.5 + ((lu - 0.5) * ca + (lv - 0.5) * sa) * 0.85
        rand_slope[ky, kx] = np.clip(sl, 0.0, 1.0)

    # Fast iterative chamfer distance from island borders (pure NumPy!)
    dist = solid.astype(np.float32)
    cur = solid.copy()
    max_r = max(2, min(H, W) // 4)
    for _ in range(max_r):
        if tileable:
            nxt_c = cur & np.roll(cur, 1, axis=0) & np.roll(cur, -1, axis=0) & np.roll(cur, 1, axis=1) & np.roll(cur, -1, axis=1)
        else:
            p = np.pad(cur, 1, mode="constant", constant_values=False)
            nxt_c = cur & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        if not np.any(nxt_c):
            break
        dist += nxt_c.astype(np.float32)
        cur = nxt_c
    max_d = float(dist.max())
    edge_dist = (dist / max(1.0, max_d)).astype(np.float32)

    return {
        "island_id": island_id.astype(np.float32),
        "random_gray": random_gray.astype(np.float32),
        "gradient_u": grad_u,
        "gradient_v": grad_v,
        "random_slope": rand_slope,
        "edge_dist": edge_dist,
    }


# ---------------------------------------------------------------------------
# 4. Substance Slope Blur, Curvature, Edge Wear, Histogram Scan & Displace
# ---------------------------------------------------------------------------


def _sample_bilinear(arr: np.ndarray, sx: np.ndarray, sy: np.ndarray, tileable: bool) -> np.ndarray:
    H, W = arr.shape[:2]
    if tileable:
        sx_w = np.mod(sx, float(W))
        sy_w = np.mod(sy, float(H))
        x0 = np.floor(sx_w).astype(np.int32) % W
        y0 = np.floor(sy_w).astype(np.int32) % H
        x1 = (x0 + 1) % W
        y1 = (y0 + 1) % H
    else:
        sx_w = np.clip(sx, 0.0, float(W - 1))
        sy_w = np.clip(sy, 0.0, float(H - 1))
        x0 = np.floor(sx_w).astype(np.int32)
        y0 = np.floor(sy_w).astype(np.int32)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
    tx = (sx_w - np.floor(sx_w)).astype(np.float32)
    ty = (sy_w - np.floor(sy_w)).astype(np.float32)
    if arr.ndim == 3:
        tx = tx[..., None]
        ty = ty[..., None]
    c00 = arr[y0, x0]
    c10 = arr[y0, x1]
    c01 = arr[y1, x0]
    c11 = arr[y1, x1]
    return ((c00 * (1.0 - tx) + c10 * tx) * (1.0 - ty) + (c01 * (1.0 - tx) + c11 * tx) * ty).astype(np.float32)


def slope_blur(
    src: np.ndarray,
    slope_map: np.ndarray,
    *,
    intensity: float = 1.6,
    samples: int = 6,
    mode: str = "min",
    tileable: bool = True,
) -> np.ndarray:
    """Substance Designer Slope Blur: advects ``src`` along the gradient of ``slope_map``.

    - ``mode="min"``  : Organic chipped stone, worn corners, weathered rock & jagged bevels!
    - ``mode="max"``  : Swollen painterly impasto, lichen/moss bloom, organic clumps!
    - ``mode="blur"`` : Directional oil/pixel brushstroke smearing along flow lines!
    """
    a = np.asarray(src, dtype=np.float32)
    sm = np.asarray(slope_map, dtype=np.float32)
    if sm.ndim == 3:
        sm = 0.299 * sm[..., 0] + 0.587 * sm[..., 1] + 0.114 * sm[..., 2]
    H, W = a.shape[:2]

    # Gradient of slope map
    if tileable:
        gx = (np.roll(sm, -1, axis=1) - np.roll(sm, 1, axis=1)) * 0.5
        gy = (np.roll(sm, -1, axis=0) - np.roll(sm, 1, axis=0)) * 0.5
    else:
        gx = np.zeros_like(sm)
        gy = np.zeros_like(sm)
        gx[:, 1:-1] = (sm[:, 2:] - sm[:, :-2]) * 0.5
        gy[1:-1, :] = (sm[2:, :] - sm[:-2, :]) * 0.5

    n_steps = max(1, int(samples))
    step_scale = float(intensity) * max(1.0, min(H, W) / 32.0) / float(n_steps)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    cur_x, cur_y = xx.copy(), yy.copy()
    acc = a.copy()
    m = mode.lower()
    for _ in range(n_steps):
        vx = _sample_bilinear(gx, cur_x, cur_y, tileable)
        vy = _sample_bilinear(gy, cur_x, cur_y, tileable)
        cur_x = cur_x + vx * step_scale * 8.0
        cur_y = cur_y + vy * step_scale * 8.0
        samp = _sample_bilinear(a, cur_x, cur_y, tileable)
        if m == "min":
            acc = np.minimum(acc, samp)
        elif m == "max":
            acc = np.maximum(acc, samp)
        else:
            acc = acc + samp
    if m == "blur":
        acc = acc / float(n_steps + 1)
    return np.clip(acc, 0.0, 1.0).astype(np.float32)


def histogram_scan(
    src: np.ndarray,
    *,
    position: float = 0.5,
    contrast: float = 0.6,
    invert: bool = False,
) -> np.ndarray:
    """Substance Designer Histogram Scan: clean position + contrast mask isolation."""
    a = np.asarray(src, dtype=np.float32)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    pos = float(np.clip(position, 0.0, 1.0))
    con = float(np.clip(contrast, 0.0, 0.999))
    span = max(1e-4, 1.0 - con)
    lo = (1.0 - pos) * (1.0 - span)
    hi = lo + span
    out = np.clip((a - lo) / max(1e-5, hi - lo), 0.0, 1.0)
    if invert:
        out = 1.0 - out
    return out.astype(np.float32)


def histogram_select(
    src: np.ndarray,
    *,
    position: float = 0.5,
    range_width: float = 0.25,
    contrast: float = 0.5,
) -> np.ndarray:
    """Substance Designer Histogram Select: isolates a band centered at ``position``."""
    a = np.asarray(src, dtype=np.float32)
    if a.ndim == 3:
        a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    d = np.abs(a - float(position))
    rw = max(1e-4, float(range_width) * 0.5)
    soft = max(1e-4, rw * (1.0 - float(np.clip(contrast, 0.0, 0.99))))
    out = np.clip(1.0 - np.maximum(0.0, d - (rw - soft)) / max(1e-4, soft), 0.0, 1.0)
    return out.astype(np.float32)


def auto_levels(src: np.ndarray, *, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Normalizes grayscale range robustly using percentile endpoints."""
    a = np.asarray(src, dtype=np.float32)
    lo = float(np.percentile(a, low_pct))
    hi = float(np.percentile(a, high_pct))
    if hi - lo < 1e-6:
        return np.clip(a, 0.0, 1.0).astype(np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def curvature_from_height(
    height_map: np.ndarray,
    *,
    radius: int = 2,
    strength: float = 2.2,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Multi-scale surface curvature detector (convex edges > 0.5, concave crevices < 0.5)."""
    h = np.asarray(height_map, dtype=np.float32)
    if h.ndim == 3:
        h = h[..., 0]
    r = max(1, int(radius))
    blur_s = gaussian_blur(h, radius=r, tileable=tileable)
    blur_l = gaussian_blur(h, radius=max(r + 1, r * 2), tileable=tileable)
    diff = 0.65 * (h - blur_s) + 0.35 * (h - blur_l)
    curv = np.clip(0.5 + diff * float(strength) * 2.5, 0.0, 1.0).astype(np.float32)
    convex = np.clip((curv - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    concave = np.clip((0.5 - curv) * 2.0, 0.0, 1.0).astype(np.float32)
    return {"curvature": curv, "convex": convex, "concave": concave}


def edge_wear_mask(
    height_map: np.ndarray,
    *,
    wear_amount: float = 0.45,
    chip_scale: float = 8.0,
    grime_amount: float = 0.55,
    tileable: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Substance-grade procedural Edge Wear (chipped highlights) & Crevice Grime mask generator."""
    h = np.asarray(height_map, dtype=np.float32)
    if h.ndim == 3:
        h = h[..., 0]
    H, W = h.shape
    c = curvature_from_height(h, radius=max(1, min(H, W) // 24), strength=2.6, tileable=tileable)
    chip_n = fbm_noise(H, W, scale=max(2.0, float(chip_scale)), octaves=3, seed=seed + 17, tileable=tileable)
    wear_raw = np.clip(c["convex"] * (0.45 + 0.85 * chip_n), 0.0, 1.0)
    wear = histogram_scan(wear_raw, position=float(wear_amount), contrast=0.68)
    grime_raw = np.clip(c["concave"] * (0.55 + 0.65 * (1.0 - chip_n)), 0.0, 1.0)
    grime = histogram_scan(grime_raw, position=float(grime_amount), contrast=0.58)
    return {"wear": wear, "grime": grime, "curvature": c["curvature"]}


def displace_uv(
    src: np.ndarray,
    disp_map: np.ndarray,
    *,
    disp_y_map: np.ndarray | None = None,
    strength_x: float = 2.0,
    strength_y: float = 2.0,
    mid_value: float = 0.5,
    pixel_snap: bool = False,
    tileable: bool = True,
) -> np.ndarray:
    """Pixel Composer `Displace` node: warps pixels via a displacement map (Bilinear or crisp Pixel snap!)."""
    a = np.asarray(src, dtype=np.float32)
    dx_m = np.asarray(disp_map, dtype=np.float32)
    if dx_m.ndim == 3:
        dx_m = 0.299 * dx_m[..., 0] + 0.587 * dx_m[..., 1] + 0.114 * dx_m[..., 2]
    dy_m = dx_m if disp_y_map is None else np.asarray(disp_y_map, dtype=np.float32)
    if dy_m.ndim == 3:
        dy_m = 0.299 * dy_m[..., 0] + 0.587 * dy_m[..., 1] + 0.114 * dy_m[..., 2]

    H, W = a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = xx + (dx_m - float(mid_value)) * 2.0 * float(strength_x)
    sy = yy + (dy_m - float(mid_value)) * 2.0 * float(strength_y)
    if pixel_snap:
        sx = np.rint(sx)
        sy = np.rint(sy)
    return _sample_bilinear(a, sx, sy, tileable)


def normal_light_2d(
    albedo: np.ndarray,
    normal_rgb: np.ndarray,
    *,
    light_az_deg: float = -45.0,
    light_el_deg: float = 48.0,
    light_color: str = "#FFF7E6",
    ambient_color: str = "#2A3352",
    ambient_strength: float = 0.34,
    spec_strength: float = 0.22,
    shininess: float = 18.0,
    quantize_steps: int = 0,
) -> np.ndarray:
    """Pixel Composer `Normal Light`: crisp directional 2D pixel-art relighting from albedo + normal map."""
    alb = np.asarray(albedo, dtype=np.float32)
    if alb.ndim == 2:
        alb = np.stack((alb, alb, alb, np.ones_like(alb)), axis=-1)
    nrm = np.asarray(normal_rgb, dtype=np.float32)[..., :3] * 2.0 - 1.0
    n_len = np.maximum(1e-6, np.linalg.norm(nrm, axis=-1, keepdims=True))
    N = nrm / n_len

    az = math.radians(float(light_az_deg))
    el = math.radians(float(light_el_deg))
    L = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)], dtype=np.float32)
    L /= max(1e-6, float(np.linalg.norm(L)))
    V = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    Hv = L + V
    Hv /= max(1e-6, float(np.linalg.norm(Hv)))

    ndotl = np.clip(np.sum(N * L[None, None, :], axis=-1), 0.0, 1.0)
    if int(quantize_steps) >= 2:
        q = float(int(quantize_steps) - 1)
        ndotl = np.rint(ndotl * q) / q
    spec = np.power(np.clip(np.sum(N * Hv[None, None, :], axis=-1), 0.0, 1.0), float(shininess)) * float(spec_strength)

    lc = np.array(hex_to_srgb(light_color), dtype=np.float32)[None, None, :]
    ac = np.array(hex_to_srgb(ambient_color), dtype=np.float32)[None, None, :]
    amb = float(np.clip(ambient_strength, 0.0, 1.0))

    lit = alb[..., :3] * (ac * amb + lc * ((1.0 - amb) * 0.35 + 0.95 * ndotl[..., None])) + lc * spec[..., None]
    out = np.concatenate((np.clip(lit, 0.0, 1.0), alb[..., 3:4]), axis=-1)
    return out.astype(np.float32)


def directional_selout(
    rgba: np.ndarray,
    *,
    light_angle_deg: float = -135.0,
    lit_darken: float = 0.64,
    shadow_darken: float = 0.30,
    warm_tint: tuple[float, float, float] = (0.08, 0.04, -0.02),
    cool_tint: tuple[float, float, float] = (-0.02, 0.01, 0.14),
    interior_outline: bool = False,
    interior_threshold: float = 0.32,
) -> np.ndarray:
    """Brandon James Greer Directional Selective Outline (`selout`) for crisp Minecraft items & sprites!

    Instead of stamping a flat dead-black 1 px outline around the whole sprite, computes the local
    silhouette outward normal on every outer boundary pixel and hue-shifts/darkens it selectively:
      - Light-facing edges (top-left): brighter, slightly warmer version of adjacent fill (`lit_darken`)
      - Shadow-facing edges (bottom-right): deep, cooler hue-shifted shadow (`shadow_darken`)
    Guarantees 100% hard binary alpha `{0.0, 1.0}` with zero muddy half-alpha fringe!
    """
    arr = np.asarray(rgba, dtype=np.float32).copy()
    if arr.ndim == 2:
        arr = np.stack((arr, arr, arr, np.ones_like(arr)), axis=-1)
    H, W = arr.shape[:2]
    solid = arr[..., 3] >= 0.5
    if not np.any(solid):
        return arr

    # Pad to check 4-neighbors safely
    p = np.pad(solid, 1, mode="constant", constant_values=False)
    up = p[:-2, 1:-1]
    dn = p[2:, 1:-1]
    lf = p[1:-1, :-2]
    rt = p[1:-1, 2:]
    border = solid & ~(up & dn & lf & rt)

    # Approximate outward silhouette normal (points toward transparent void!)
    nx = (~rt).astype(np.float32) - (~lf).astype(np.float32)
    ny = (~dn).astype(np.float32) - (~up).astype(np.float32)
    nlen = np.maximum(1e-5, np.sqrt(nx * nx + ny * ny))
    nx /= nlen
    ny /= nlen

    rad = math.radians(float(light_angle_deg))
    lx, ly = math.cos(rad), math.sin(rad)
    # lit_w = 1 on light-facing silhouette edges, 0 on shadow-facing edges!
    lit_w = np.clip(0.5 + 0.5 * (nx * lx + ny * ly), 0.0, 1.0)

    rgb = arr[..., :3]
    wt = np.array(warm_tint, dtype=np.float32)[None, None, :]
    ct = np.array(cool_tint, dtype=np.float32)[None, None, :]

    sel_lit = np.clip(rgb * float(lit_darken) + wt * 0.5, 0.0, 1.0)
    sel_sh = np.clip(rgb * float(shadow_darken) + ct * 0.65, 0.0, 1.0)
    sel_rgb = sel_sh * (1.0 - lit_w[..., None]) + sel_lit * lit_w[..., None]

    out_rgb = np.where(border[..., None], sel_rgb, rgb)

    if interior_outline:
        lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        gx = np.abs(np.pad(lum, ((0, 0), (1, 1)), mode="edge")[:, 2:] - lum)
        gy = np.abs(np.pad(lum, ((1, 1), (0, 0)), mode="edge")[2:, :] - lum)
        inner_e = solid & (~border) & (np.maximum(gx, gy) >= float(interior_threshold))
        out_rgb = np.where(inner_e[..., None], np.clip(rgb * 0.68 + ct * 0.3, 0.0, 1.0), out_rgb)

    out = np.concatenate((np.clip(out_rgb, 0.0, 1.0), solid.astype(np.float32)[..., None]), axis=-1)
    return out.astype(np.float32)
