"""Smooth Gradient Path, Multi-Stop OKLab Color Ramp, Spline Ribbon & 3D Body-Flow Generators.

Pure vectorized NumPy implementation designed for studio Blockbench pixel-art craft:
- Silky smooth perceptual OKLab multi-stop gradients (zero muddy gray midpoints!)
- Smooth-to-pixel hybrid blending (`smooth_blend` slider from crisp pixel bands to silky smooth)
- Cosine palette gradients (Pixel Composer `Gradient Cos` parity)
- Bezier / Catmull-Rom spline path distance & along-path gradient ribbons
- Per-UV-island directional/rim smooth gradients & 3D creature body-flow gradients
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .generators_noise import bayer_matrix, uv_grid
from .pixel_art_ops import hex_to_srgb, linear_to_srgb, srgb_to_oklab


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert OKLab (..., 3) -> sRGB float32 [0, 1]."""
    arr = np.asarray(lab, dtype=np.float64)
    L, A, B = arr[..., 0], arr[..., 1], arr[..., 2]
    l_ = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3
    m_ = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3
    s_ = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    return linear_to_srgb(np.stack((r, g, bl), axis=-1))


def gradient_field(
    height: int,
    width: int,
    *,
    kind: str = "linear",
    angle_deg: float = 90.0,
    cx: float = 0.5,
    cy: float = 0.5,
    radius: float = 0.55,
    aspect: float = 1.0,
    repeat: float = 1.0,
    ping_pong: bool = False,
    ease: str = "linear",
) -> np.ndarray:
    """Generate a smooth scalar gradient field [0, 1] in linear, radial, conical, diamond, box, or reflected mode."""
    u, v = uv_grid(height, width)
    dx = (u - float(cx)) * float(aspect)
    dy = v - float(cy)

    if kind == "radial":
        t = 1.0 - np.sqrt(dx * dx + dy * dy) / max(1e-4, float(radius))
    elif kind == "radial_out":
        t = np.sqrt(dx * dx + dy * dy) / max(1e-4, float(radius))
    elif kind == "diamond":
        t = 1.0 - (np.abs(dx) + np.abs(dy)) / max(1e-4, float(radius) * 1.414)
    elif kind == "box":
        t = 1.0 - np.maximum(np.abs(dx), np.abs(dy)) / max(1e-4, float(radius))
    elif kind == "conical":
        ang = np.arctan2(dy, dx) + math.radians(float(angle_deg))
        t = (ang / (2.0 * math.pi)) % 1.0
    elif kind == "reflected":
        rad = math.radians(float(angle_deg))
        proj = dx * math.cos(rad) + dy * math.sin(rad)
        t = 1.0 - np.abs(proj) / max(1e-4, float(radius))
    else:  # "linear"
        rad = math.radians(float(angle_deg))
        proj = dx * math.cos(rad) + dy * math.sin(rad)
        t = proj + 0.5

    if float(repeat) != 1.0:
        t = t * float(repeat)
        if ping_pong:
            t = 1.0 - np.abs((t % 2.0) - 1.0)
        else:
            t = t % 1.0
    elif ping_pong:
        t = 1.0 - np.abs(np.clip(t, 0.0, 1.0) * 2.0 - 1.0)

    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    if ease == "smoothstep":
        t = t * t * (3.0 - 2.0 * t)
    elif ease == "smootherstep":
        t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    elif ease == "ease_in":
        t = t * t
    elif ease == "ease_out":
        t = 1.0 - (1.0 - t) ** 2
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def multi_stop_gradient(
    field: np.ndarray,
    stops: Sequence[tuple[float, str] | list[Any]],
    *,
    space: str = "oklab",
    smooth_blend: float = 1.0,
    quantize_steps: int = 0,
    dither: str = "none",
    dither_strength: float = 0.12,
    alpha_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map scalar field [0, 1] -> RGBA via arbitrary multi-stop perceptual OKLab or sRGB smooth gradient.

    Parameters
    ----------
    stops:
        Sequence of ``(position_0_to_1, "#RRGGBB")`` pairs (or plain list of hex strings, evenly spaced).
    space:
        ``"oklab"`` (default): perceptually uniform silky interpolation with zero muddy gray transitions.
        ``"srgb"``: standard sRGB linear segment interpolation.
    smooth_blend:
        When ``quantize_steps >= 2``, blends between quantized pixel-art bands (0.0) and pure smooth
        continuous gradient (1.0) — e.g. 0.45–0.75 achieves the iconic painterly-pixel Blockbench studio look!
    """
    f = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    H, W = f.shape[:2]

    parsed: list[tuple[float, tuple[float, float, float]]] = []
    if stops and isinstance(stops[0], str):
        n_s = len(stops)
        for i, hx in enumerate(stops):
            pos = i / float(max(1, n_s - 1))
            parsed.append((pos, hex_to_srgb(str(hx))))
    else:
        for item in stops:
            pos, hx = float(item[0]), str(item[1])
            parsed.append((pos, hex_to_srgb(hx)))
    if len(parsed) < 2:
        parsed = [(0.0, (0.08, 0.08, 0.14)), (1.0, (0.95, 0.96, 1.0))]
    parsed.sort(key=lambda x: x[0])

    pos_arr = np.array([p[0] for p in parsed], dtype=np.float32)
    rgb_arr = np.array([p[1] for p in parsed], dtype=np.float32)
    col_space = srgb_to_oklab(rgb_arr) if space == "oklab" else rgb_arr

    val = f
    q_n = int(quantize_steps)
    if q_n >= 2 and float(smooth_blend) < 0.999:
        v_d = val
        if dither in ("bayer2", "bayer4", "bayer8"):
            sz = {"bayer2": 2, "bayer4": 4, "bayer8": 8}[dither]
            bm = bayer_matrix(H, W, size=sz) - 0.5
            v_d = np.clip(val + bm * (float(dither_strength) / float(q_n - 1)), 0.0, 1.0)
        elif dither == "clustered":
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
            cl = (((yy + xx // 2) % 2).astype(np.float32) - 0.5)
            v_d = np.clip(val + cl * (float(dither_strength) / float(q_n - 1)), 0.0, 1.0)
        v_q = np.rint(v_d * float(q_n - 1)) / float(q_n - 1)
        sb = float(np.clip(smooth_blend, 0.0, 1.0))
        val = val * sb + v_q * (1.0 - sb)

    # Vectorized piecewise linear interpolation in OKLab or sRGB
    idx = np.clip(np.searchsorted(pos_arr, val, side="right") - 1, 0, len(pos_arr) - 2)
    p0 = pos_arr[idx]
    p1 = pos_arr[idx + 1]
    frac = np.clip((val - p0) / np.maximum(1e-6, p1 - p0), 0.0, 1.0)[..., None]
    interp = col_space[idx] * (1.0 - frac) + col_space[idx + 1] * frac
    rgb = oklab_to_srgb(interp) if space == "oklab" else np.clip(interp, 0.0, 1.0).astype(np.float32)

    alpha = np.ones((H, W, 1), dtype=np.float32) if alpha_mask is None else np.clip(np.asarray(alpha_mask, dtype=np.float32), 0.0, 1.0)[..., None]
    return np.concatenate((rgb, alpha), axis=-1).astype(np.float32)


def cosine_gradient(
    field: np.ndarray,
    *,
    a: tuple[float, float, float] = (0.50, 0.50, 0.50),
    b: tuple[float, float, float] = (0.50, 0.50, 0.50),
    c: tuple[float, float, float] = (1.00, 1.00, 1.00),
    d: tuple[float, float, float] = (0.00, 0.33, 0.67),
    phase: float = 0.0,
    alpha_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inigo Quilez / Pixel Composer `Gradient Cos` smooth procedural iridescent palette node."""
    t = np.asarray(field, dtype=np.float32)[..., None] + float(phase)
    va = np.array(a, dtype=np.float32)[None, None, :]
    vb = np.array(b, dtype=np.float32)[None, None, :]
    vc = np.array(c, dtype=np.float32)[None, None, :]
    vd = np.array(d, dtype=np.float32)[None, None, :]
    rgb = np.clip(va + vb * np.cos(2.0 * math.pi * (vc * t + vd)), 0.0, 1.0).astype(np.float32)
    H, W = field.shape[:2]
    alpha = np.ones((H, W, 1), dtype=np.float32) if alpha_mask is None else np.clip(np.asarray(alpha_mask, dtype=np.float32), 0.0, 1.0)[..., None]
    return np.concatenate((rgb, alpha), axis=-1).astype(np.float32)


def spline_path_gradient(
    height: int,
    width: int,
    points: Sequence[tuple[float, float] | list[float]],
    *,
    stroke_width: float = 0.12,
    samples_per_seg: int = 24,
    closed: bool = False,
    output: str = "ribbon",
    falloff: str = "smoothstep",
) -> np.ndarray:
    """Evaluate a smooth Catmull-Rom spline path in UV space [0, 1]^2.

    ``output``:
      - ``"ribbon"``: smooth 1.0 on spline core falling off over ``stroke_width`` to 0.0
      - ``"along"`` : normalized [0, 1] arc-length progress along the path (great for sweeping along-path color gradients!)
      - ``"sdf"``   : normalized unsigned distance field from the spline
    """
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 2:
        return np.zeros((height, width), dtype=np.float32)

    # Sample smooth Catmull-Rom polyline
    n = len(pts)
    poly: list[tuple[float, float]] = []
    seg_count = n if closed else (n - 1)
    sp = max(4, int(samples_per_seg))
    for i in range(seg_count):
        p0 = pts[(i - 1) % n] if closed else pts[max(0, i - 1)]
        p1 = pts[i % n]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if closed else pts[min(n - 1, i + 2)]
        for s in range(sp):
            t = s / float(sp)
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2.0 * p1[0]) + (-p0[0] + p2[0]) * t + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2 + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2.0 * p1[1]) + (-p0[1] + p2[1]) * t + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2 + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3)
            poly.append((x, y))
    if not closed:
        poly.append(pts[-1])
    else:
        poly.append(poly[0])

    P = np.asarray(poly, dtype=np.float32)  # (M, 2)
    seg_v = P[1:] - P[:-1]
    seg_len2 = np.maximum(1e-10, np.sum(seg_v * seg_v, axis=-1))
    seg_len = np.sqrt(seg_len2)
    cum_len = np.concatenate(([0.0], np.cumsum(seg_len)))
    tot_len = max(1e-6, float(cum_len[-1]))

    u, v = uv_grid(height, width)
    min_d2 = np.full((height, width), 1e9, dtype=np.float32)
    best_along = np.zeros((height, width), dtype=np.float32)

    for k in range(len(seg_v)):
        ax, ay = float(P[k, 0]), float(P[k, 1])
        vx, vy = float(seg_v[k, 0]), float(seg_v[k, 1])
        l2 = float(seg_len2[k])
        t = np.clip(((u - ax) * vx + (v - ay) * vy) / l2, 0.0, 1.0)
        px = ax + t * vx
        py = ay + t * vy
        d2 = (u - px) ** 2 + (v - py) ** 2
        closer = d2 < min_d2
        if np.any(closer):
            min_d2 = np.where(closer, d2, min_d2)
            best_along = np.where(closer, (cum_len[k] + t * seg_len[k]) / tot_len, best_along)

    dist = np.sqrt(min_d2)
    if output == "along":
        return np.clip(best_along, 0.0, 1.0).astype(np.float32)
    if output == "sdf":
        return np.clip(dist, 0.0, 1.0).astype(np.float32)

    w = max(1e-4, float(stroke_width))
    t_r = np.clip(1.0 - dist / w, 0.0, 1.0)
    if falloff == "smoothstep":
        t_r = t_r * t_r * (3.0 - 2.0 * t_r)
    elif falloff == "glow":
        t_r = np.power(t_r, 2.2)
    return t_r.astype(np.float32)


def per_island_gradient(
    local_u: np.ndarray,
    local_v: np.ndarray,
    *,
    direction: str = "top_to_bottom",
    curve: float = 1.0,
) -> np.ndarray:
    """Compute a smooth studio shading gradient independently across every UV cuboid face island.

    Directions:
      - ``"top_to_bottom"`` : bright top (1.0) fading smoothly down to shadowed bottom (0.0)
      - ``"bottom_to_top"`` : bright bottom to shadowed top
      - ``"left_to_right"`` / ``"diag"``
      - ``"center_out"``    : bright convex center fading smoothly toward face edges (fakes roundness!)
      - ``"edge_rim"``      : soft 1..2 px rim highlight hugging face perimeter
    """
    u = np.clip(np.asarray(local_u, dtype=np.float32), 0.0, 1.0)
    v = np.clip(np.asarray(local_v, dtype=np.float32), 0.0, 1.0)
    if direction == "bottom_to_top":
        val = v
    elif direction == "left_to_right":
        val = u
    elif direction == "diag":
        val = 0.5 * (u + (1.0 - v))
    elif direction == "center_out":
        d = np.sqrt(((u - 0.5) * 2.0) ** 2 + ((v - 0.5) * 2.0) ** 2)
        val = np.clip(1.0 - d * 0.72, 0.0, 1.0)
    elif direction == "edge_rim":
        edge_d = np.minimum(np.minimum(u, 1.0 - u), np.minimum(v, 1.0 - v))
        val = np.clip(1.0 - edge_d * 5.0, 0.0, 1.0)
    else:  # "top_to_bottom"
        val = 1.0 - v
    if abs(float(curve) - 1.0) > 1e-4:
        val = np.power(np.clip(val, 0.0, 1.0), float(curve))
    return np.clip(val, 0.0, 1.0).astype(np.float32)


def body_flow_gradient(
    world_pos_norm: np.ndarray,
    world_normal: np.ndarray,
    *,
    axis: str = "dorsal_ventral",
    bias: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute a silky continuous 3D creature-space gradient across all connected bones (zero seams!).

    Axes:
      - ``"dorsal_ventral"`` : combines 3D Y height + upward surface normal (classic countershading!)
      - ``"head_to_tail"``   : front (+Z) to rear (-Z) longitudinal creature gradient
      - ``"center_to_tips"`` : torso core (0.0) sweeping out to wingtips/horns/tail-tip (1.0)
      - ``"top_light_wrap"`` : smooth half-Lambert 3D studio key-light wrap (silky painterly base!)
    """
    pn = np.asarray(world_pos_norm, dtype=np.float32)
    wn = np.asarray(world_normal, dtype=np.float32)
    if axis == "head_to_tail":
        val = pn[..., 2]
    elif axis == "center_to_tips":
        dx = np.abs(pn[..., 0] * 2.0 - 1.0)
        dy = np.abs(pn[..., 1] - 0.45) * 1.4
        dz = np.abs(pn[..., 2] - 0.52) * 1.5
        val = np.clip(np.sqrt(dx * dx + dy * dy * 0.4 + dz * dz * 0.65), 0.0, 1.0)
    elif axis == "top_light_wrap":
        ld = np.array([0.32, 0.86, 0.39], dtype=np.float32)
        ld /= float(np.linalg.norm(ld))
        ndotl = np.sum(wn * ld[None, None, :], axis=-1)
        # Half-Lambert silky wrap (never crushes undersides to flat black!)
        val = np.clip((ndotl * 0.5 + 0.5) ** 1.35 * 0.78 + pn[..., 1] * 0.22, 0.0, 1.0)
    else:  # "dorsal_ventral"
        hemi = 0.5 + 0.5 * wn[..., 1]
        val = np.clip(0.55 * hemi + 0.45 * pn[..., 1], 0.0, 1.0)
    val = np.clip(val + float(bias), 0.0, 1.0)
    if abs(float(gamma) - 1.0) > 1e-4:
        val = np.power(val, float(gamma))
    return val.astype(np.float32)
