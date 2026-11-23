#!/usr/bin/env python3
"""
Visual Primitives & Native-Resolution Reference Preprocessor (v33).

Implements formal visual-spatial reasoning inspired by DeepSeek's "Thinking with Visual
Primitives" (points, bounding boxes, dashed skeleton chains, angle arcs, proportion
calipers) paired with native 1:1 lossless cropping and structural edge/contour filtering.

Designed as a general-purpose pipeline for ANY 3D Blockbench subject:
creatures, real animals, humanoids, vehicles, mechs, props, and architecture.

Dependencies: Python 3 + Pillow + NumPy (standard in studio/agent environments).

Subcommands:
  1. grid   <in_img> <out_png> [--step 0.1] [--dual-edges]
     Overlay 0.0-1.0 normalized coordinate grid at 100% NATIVE 1:1 pixel resolution
     (zero downsampling!). Optionally pairs native color + structural edge-map side by side.

  2. edges  <in_img> <out_png> [--mode ink|overlay|dual] [--sigma 1.0] [--gain 2.0]
     Extract crisp structural contours (Scharr/Sobel multi-scale gradient + subtle DoG)
     to expose silhouette breaks, eyelid rims, bevels, plate overlaps, and joint angles.

  3. zoom   <in_img> <out_dir> --boxes '<json>' [--min-side 480] [--with-edges]
     Export lossless native 1:1 regional crops (never compressed thumbnails!) plus
     matching structural edge-maps and edge overlays for focal regions (eyes, joints,
     tails/plates, accessories).

  4. trace  <in_img> <spec_json> <out_png> [--edge-blend 0.0] [--export-crops-dir <dir>]
     Render DeepSeek-style Visual Primitives at 100% native resolution:
       - Dashed skeleton centerlines (`- - -`) with joint nodes & automatic bend angles (deg)
       - Dotted proportion calipers (`· · ·`) with automatic segment-length ratios
       - Focal bounding boxes (`[x1, y1, x2, y2]`) with optional lossless native crop export
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


DEFAULT_CHAIN_COLORS = [
    (0, 235, 255),    # cyan: spine / primary axis
    (255, 110, 64),   # coral-orange: hindlimb / secondary chain
    (120, 255, 130),  # mint-green: forelimb / arm
    (255, 215, 64),   # gold: tail / crest / horn
    (220, 130, 255),  # violet: wing / accessory / turret
]


def _parse_color(val: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        return (int(val[0]), int(val[1]), int(val[2]))
    if isinstance(val, str):
        s = val.strip().lstrip("#")
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    return fallback


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _convolve_separable_1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        out = np.zeros_like(arr, dtype=np.float64)
        for i, k in enumerate(kernel):
            out += float(k) * padded[i : i + arr.shape[0], :]
        return out
    padded = np.pad(arr, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float64)
    for i, k in enumerate(kernel):
        out += float(k) * padded[:, i : i + arr.shape[1]]
    return out


def _gaussian_blur(gray: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.15:
        return gray.astype(np.float64)
    radius = max(1, int(math.ceil(2.5 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (xs / sigma) ** 2)
    k /= np.sum(k)
    tmp = _convolve_separable_1d(gray.astype(np.float64), k, axis=1)
    return _convolve_separable_1d(tmp, k, axis=0)


def _bilateral_smooth_rgb(
    rgb: np.ndarray, *, radius: int = 2, sigma_s: float = 1.5, sigma_r: float = 0.14
) -> np.ndarray:
    """Fast vectorized edge-preserving bilateral filter (melts fine grain while keeping sharp edges!)."""
    h, w, _ = rgb.shape
    padded = np.pad(rgb, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    acc = np.zeros_like(rgb, dtype=np.float64)
    w_sum = np.zeros((h, w, 1), dtype=np.float64)
    inv_2s2 = 0.5 / max(1e-6, sigma_s * sigma_s)
    inv_2r2 = 0.5 / max(1e-6, sigma_r * sigma_r)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = math.exp(-(dx * dx + dy * dy) * inv_2s2)
            patch = padded[radius + dy : radius + dy + h, radius + dx : radius + dx + w, :]
            diff2 = np.sum(np.square(patch - rgb), axis=-1, keepdims=True)
            wr = np.exp(-diff2 * inv_2r2)
            wt = ws * wr
            acc += patch * wt
            w_sum += wt
    return acc / np.maximum(w_sum, 1e-8)


def _nms_and_hysteresis(
    mag: np.ndarray, gx: np.ndarray, gy: np.ndarray, *, gain: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """True 8-neighbor Non-Maximum Suppression (1 px razor-thin ridges) + hysteresis contour linking."""
    angle = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0
    mp = np.pad(mag, ((1, 1), (1, 1)), mode="edge")
    # 4 quantized gradient-normal directions: 0 deg (E-W), 45 deg (NE-SW), 90 deg (N-S), 135 deg (NW-SE)
    q0 = (angle < 22.5) | (angle >= 157.5)
    q45 = (angle >= 22.5) & (angle < 67.5)
    q90 = (angle >= 67.5) & (angle < 112.5)
    q135 = (angle >= 112.5) & (angle < 157.5)

    n_a = np.where(
        q0, mp[1:-1, 2:],
        np.where(q45, mp[0:-2, 2:], np.where(q90, mp[0:-2, 1:-1], mp[0:-2, 0:-2])),
    )
    n_b = np.where(
        q0, mp[1:-1, 0:-2],
        np.where(q45, mp[2:, 0:-2], np.where(q90, mp[2:, 1:-1], mp[2:, 2:])),
    )
    nms = np.where((mag >= n_a) & (mag >= n_b), mag, 0.0)

    nz = nms[nms > 1e-5]
    if nz.size < 16:
        z = np.zeros_like(nms)
        return z, z
    p_hi = float(np.percentile(nz, 78.0))
    p_lo = float(np.percentile(nz, 50.0))
    scale = max(1e-5, float(np.percentile(nz, 97.5)))
    strength = np.clip((nms / scale) * (gain * 0.62), 0.0, 1.0)

    strong = nms >= max(1e-5, p_hi / max(0.55, gain * 0.52))
    weak = (nms >= max(1e-5, p_lo / max(0.55, gain * 0.52))) & (~strong)
    # 5-pass vectorized 8-neighbor hysteresis linking (connects weak contours touching strong ridges)
    linked = strong.copy()
    for _ in range(5):
        lp = np.pad(linked, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        nbr = (
            lp[0:-2, 0:-2] | lp[0:-2, 1:-1] | lp[0:-2, 2:]
            | lp[1:-1, 0:-2] | lp[1:-1, 2:]
            | lp[2:, 0:-2] | lp[2:, 1:-1] | lp[2:, 2:]
        )
        new_linked = linked | (weak & nbr)
        if np.array_equal(new_linked, linked):
            break
        linked = new_linked

    strong_map = np.where(strong, np.clip(0.55 + 0.45 * strength, 0.0, 1.0), 0.0)
    secondary_map = np.where(linked & (~strong), np.clip(0.32 + 0.48 * strength, 0.0, 0.85), 0.0)
    return strong_map, secondary_map


def compute_structural_edges(
    img: Image.Image, *, sigma: float = 0.9, gain: float = 2.0
) -> np.ndarray:
    """Return crisp [0, 1] 1 px Non-Max-Suppressed bilateral color-tensor structural edge map (ZERO blur!)."""
    strong_map, secondary_map = _compute_structural_edge_tiers(img, sigma=sigma, gain=gain)
    return np.clip(np.maximum(strong_map, secondary_map * 0.68), 0.0, 1.0)


def _compute_structural_edge_tiers(
    img: Image.Image, *, sigma: float = 0.9, gain: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (strong_1px_silhouettes, secondary_1px_detail_contours) via edge-preserving bilateral + color-tensor NMS."""
    rgb_raw = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
    # Edge-preserving bilateral pre-filter suppresses noisy fur/scale grain WITHOUT blurring silhouette edges!
    rgb = _bilateral_smooth_rgb(rgb_raw, radius=2, sigma_s=max(1.1, sigma * 1.4), sigma_r=0.14)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # Opponent chromatic channels so colored plates of similar luminance still yield razor-crisp edges
    rg = rgb[..., 0] - rgb[..., 1]
    by = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]

    k_d = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    k_s = np.array([3.0, 10.0, 3.0], dtype=np.float64) / 16.0

    def _scharr(ch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = _convolve_separable_1d(_convolve_separable_1d(ch, k_d, axis=1), k_s, axis=0)
        gy = _convolve_separable_1d(_convolve_separable_1d(ch, k_s, axis=1), k_d, axis=0)
        return gx, gy

    lx, ly = _scharr(lum)
    rgx, rgy = _scharr(rg)
    byx, byy = _scharr(by)

    # Color Structure Tensor (Di Zenzo multi-channel edge magnitude + dominant orientation)
    w_c = 0.52
    gxx = lx * lx + w_c * (rgx * rgx + byx * byx)
    gyy = ly * ly + w_c * (rgy * rgy + byy * byy)
    gxy = lx * ly + w_c * (rgx * rgy + byx * byy)
    trace = gxx + gyy
    disc = np.sqrt(np.maximum(0.0, np.square(gxx - gyy) + 4.0 * np.square(gxy)))
    mag = np.sqrt(np.maximum(0.0, 0.5 * (trace + disc)))
    theta = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)
    gx_dom = np.cos(theta)
    gy_dom = np.sin(theta)
    return _nms_and_hysteresis(mag, gx_dom, gy_dom, gain=gain)


def render_edge_image(
    img: Image.Image,
    *,
    mode: str = "color_ink",
    sigma: float = 0.9,
    gain: float = 2.0,
) -> Image.Image:
    strong, secondary = _compute_structural_edge_tiers(img, sigma=sigma, gain=gain)
    edges = np.clip(np.maximum(strong, secondary * 0.68), 0.0, 1.0)
    src = np.asarray(img.convert("RGB"), dtype=np.float64)
    if mode == "color_ink":
        # Crisp 1 px dark structural ink + cool slate secondary contours over clean 34% full-color anatomical wash!
        wash = 248.0 * 0.66 + src * 0.34
        out = wash.copy()
        sec_w = np.clip(secondary * 0.72, 0.0, 0.72)[..., None]
        sec_col = np.array([58.0, 78.0, 108.0])[None, None, :]
        out = out * (1.0 - sec_w) + sec_col * sec_w
        str_w = np.clip(strong * 0.96, 0.0, 0.96)[..., None]
        str_col = np.array([12.0, 16.0, 26.0])[None, None, :]
        out = out * (1.0 - str_w) + str_col * str_w
        return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8), mode="RGB")
    if mode == "ink":
        # Pure razor-crisp 1 px architectural ink lines on clean drafting white (ZERO blurry halos!)
        val = np.clip(np.round((1.0 - edges) * 250.0), 10, 250).astype(np.uint8)
        rgb = np.stack([val, val, np.clip(val.astype(np.int32) + 2, 0, 252).astype(np.uint8)], axis=-1)
        return Image.fromarray(rgb, mode="RGB")
    if mode == "overlay":
        # Crisp 1 px glowing cyan contours directly on slightly dimmed full-color native reference
        base = src * 0.76
        sec_col = np.array([255.0, 200.0, 70.0])[None, None, :]
        str_col = np.array([40.0, 245.0, 255.0])[None, None, :]
        w2 = np.clip(secondary * 0.65, 0.0, 0.65)[..., None]
        w1 = np.clip(strong * 0.95, 0.0, 0.95)[..., None]
        blended = base * (1.0 - w2) + sec_col * w2
        blended = blended * (1.0 - w1) + str_col * w1
        return Image.fromarray(np.clip(np.round(blended), 0, 255).astype(np.uint8), mode="RGB")
    if mode == "dual":
        ink = render_edge_image(img, mode="color_ink", sigma=sigma, gain=gain)
        gap = 8
        out = Image.new("RGB", (img.width * 2 + gap, img.height), (24, 28, 36))
        out.paste(img.convert("RGB"), (0, 0))
        out.paste(ink, (img.width + gap, 0))
        return out
    raise ValueError(f"Unknown edge mode {mode!r}; expected color_ink, ink, overlay, or dual")


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int = 3,
    dash: float = 12.0,
    gap: float = 7.0,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = math.hypot(dx, dy)
    if dist < 1e-4:
        return
    ux, uy = dx / dist, dy / dist
    pos = 0.0
    while pos < dist:
        end = min(dist, pos + dash)
        draw.line(
            [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * end, y0 + uy * end)],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def _draw_label_pill(
    draw: ImageDraw.ImageDraw,
    pos: tuple[float, float],
    text: str,
    *,
    fg: tuple[int, int, int],
    bg: tuple[int, int, int] = (14, 18, 26),
    font: ImageFont.ImageFont,
    pad: int = 3,
    bounds: tuple[int, int] | None = None,
    occupied: list[tuple[int, int, int, int]] | None = None,
) -> None:
    x0, y0 = int(round(pos[0])), int(round(pos[1]))
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = (tb[2] - tb[0]) + 2 * pad, (tb[3] - tb[1]) + 2 * pad

    def clamp(cx: int, cy: int) -> tuple[int, int]:
        if bounds is not None:
            bw, bh = bounds
            cx = max(pad + 2, min(bw - tw - pad - 2, cx))
            cy = max(pad + 2, min(bh - th - pad - 2, cy))
        return cx, cy

    candidates = [
        (x0, y0),
        (x0, y0 - th - 6),
        (x0, y0 + th + 6),
        (x0 - tw - 12, y0),
        (x0 - tw - 12, y0 - th - 6),
        (x0 + 12, y0 + th + 12),
    ]
    chosen_x, chosen_y = clamp(x0, y0)
    if occupied is not None:
        for cx, cy in candidates:
            cx, cy = clamp(cx, cy)
            rect = (cx - pad, cy - pad, cx + tw - pad, cy + th - pad)
            if not any(
                not (rect[2] < o[0] or rect[0] > o[2] or rect[3] < o[1] or rect[1] > o[3])
                for o in occupied
            ):
                chosen_x, chosen_y = cx, cy
                break
        occupied.append((chosen_x - pad, chosen_y - pad, chosen_x + tw - pad, chosen_y + th - pad))

    bbox = draw.textbbox((chosen_x, chosen_y), text, font=font)
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=bg,
        outline=fg,
        width=1,
    )
    draw.text((chosen_x, chosen_y), text, fill=fg, font=font)


def _draw_corner_box(
    draw: ImageDraw.ImageDraw,
    box_px: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    width: int = 3,
    label: str | None = None,
    font: ImageFont.ImageFont | None = None,
) -> None:
    x0, y0, x1, y1 = box_px
    _draw_dashed_line(draw, (x0, y0), (x1, y0), fill=color, width=max(2, width - 1), dash=10, gap=6)
    _draw_dashed_line(draw, (x1, y0), (x1, y1), fill=color, width=max(2, width - 1), dash=10, gap=6)
    _draw_dashed_line(draw, (x1, y1), (x0, y1), fill=color, width=max(2, width - 1), dash=10, gap=6)
    _draw_dashed_line(draw, (x0, y1), (x0, y0), fill=color, width=max(2, width - 1), dash=10, gap=6)
    corner = max(10, min(26, int(0.16 * min(x1 - x0, y1 - y0))))
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line([(cx, cy), (cx + sx * corner, cy)], fill=color, width=width + 1)
        draw.line([(cx, cy), (cx, cy + sy * corner)], fill=color, width=width + 1)
    if label and font is not None:
        _draw_label_pill(draw, (x0 + 4, max(4, y0 - 22)), label, fg=color, font=font)


def _autocrop_letterbox(img: Image.Image, tol: int = 12) -> Image.Image:
    """Trim uniform black/white letterbox margins so 0.0-1.0 coordinates frame the actual subject."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int32)
    h, w, _ = arr.shape
    if h < 16 or w < 16:
        return img
    # Detect border color from the 4 corners
    corners = np.array([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]])
    bg = np.median(corners, axis=0)
    diff = np.max(np.abs(arr - bg[None, None, :]), axis=-1)
    row_active = np.mean(diff > tol, axis=1) > 0.015
    col_active = np.mean(diff > tol, axis=0) > 0.015
    ys = np.where(row_active)[0]
    xs = np.where(col_active)[0]
    if ys.size < 8 or xs.size < 8:
        return img
    pad = 4
    y0, y1 = max(0, int(ys[0]) - pad), min(h, int(ys[-1]) + 1 + pad)
    x0, x1 = max(0, int(xs[0]) - pad), min(w, int(xs[-1]) + 1 + pad)
    return img.crop((x0, y0, x1, y1))


def render_grid_panel(
    src: Image.Image,
    *,
    step: float = 0.1,
    dual_edges: bool = True,
    include_clean_cutout: bool = False,
    sigma: float = 1.0,
    gain: float = 1.65,
) -> Image.Image:
    """Overlay a fine 0.0-1.0 coordinate grid (+ optional crisp 1 px NMS color-context edges & clean color cutout)."""
    w, h = src.size
    margin = 34
    base = Image.new("RGB", (w + margin, h + margin), (18, 22, 30))
    base.paste(src, (margin, margin))
    draw = ImageDraw.Draw(base, "RGBA")
    font = _load_font(max(11, min(16, int(min(w, h) * 0.022))))

    n_steps = max(2, int(round(1.0 / step)))
    for i in range(n_steps + 1):
        frac = min(1.0, i * step)
        gx = margin + int(round(frac * (w - 1)))
        gy = margin + int(round(frac * (h - 1)))
        is_mid = abs(frac - 0.5) < 1e-3
        line_col = (255, 220, 80, 150) if is_mid else (0, 230, 255, 110)
        draw.line([(gx, margin), (gx, margin + h)], fill=line_col, width=2 if is_mid else 1)
        draw.line([(margin, gy), (margin + w, gy)], fill=line_col, width=2 if is_mid else 1)
        lbl = f"{frac:.1f}"
        draw.text((max(2, gx - 10), 8), lbl, fill=(220, 240, 255, 255), font=font)
        draw.text((4, max(2, gy - 7)), lbl, fill=(220, 240, 255, 255), font=font)

    if not dual_edges and not include_clean_cutout:
        return base

    panes: list[Image.Image] = []
    if include_clean_cutout:
        # Panel 1: 100% Un-filtered, Un-gridded Clean Native Color Cutout (zero grid/filter occlusion!)
        clean_pane = Image.new("RGB", (w + margin, h + margin), (18, 22, 30))
        clean_pane.paste(src, (margin, margin))
        cdraw = ImageDraw.Draw(clean_pane)
        cdraw.text((margin + 4, 8), "NATIVE COLOR CUTOUT (UNFILTERED)", fill=(245, 228, 150), font=font)
        panes.append(clean_pane)
    panes.append(base)
    if dual_edges:
        # Crisp 1 px Non-Max-Suppressed Bilateral Color-Context Structural Contours (ZERO blurry gray sludge!)
        edge_rgb = render_edge_image(src, mode="color_ink", sigma=sigma, gain=gain)
        edge_pane = Image.new("RGB", (w + margin, h + margin), (18, 22, 30))
        edge_pane.paste(edge_rgb, (margin, margin))
        edraw = ImageDraw.Draw(edge_pane, "RGBA")
        for i in range(n_steps + 1):
            frac = min(1.0, i * step)
            gx = margin + int(round(frac * (w - 1)))
            gy = margin + int(round(frac * (h - 1)))
            is_mid = abs(frac - 0.5) < 1e-3
            line_col = (220, 120, 0, 125) if is_mid else (0, 130, 180, 82)
            edraw.line([(gx, margin), (gx, margin + h)], fill=line_col, width=1)
            edraw.line([(margin, gy), (margin + w, gy)], fill=line_col, width=1)
            lbl = f"{frac:.1f}"
            edraw.text((max(2, gx - 10), 8), lbl, fill=(220, 240, 255, 255), font=font)
            edraw.text((4, max(2, gy - 7)), lbl, fill=(220, 240, 255, 255), font=font)
        panes.append(edge_pane)

    gap = 8
    total_w = sum(p.width for p in panes) + gap * (len(panes) - 1)
    combined = Image.new("RGB", (total_w, base.height), (12, 15, 20))
    cur_x = 0
    for p in panes:
        combined.paste(p, (cur_x, 0))
        cur_x += p.width + gap
    return combined


def cmd_grid(args: argparse.Namespace) -> None:
    src = Image.open(args.input).convert("RGB")
    if getattr(args, "autocrop", False):
        src = _autocrop_letterbox(src)
    w, h = src.size
    out_img = render_grid_panel(
        src,
        step=float(args.step),
        dual_edges=bool(args.dual_edges),
        sigma=float(args.sigma),
        gain=float(args.gain),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path, format="PNG", compress_level=3)
    print(json.dumps({
        "command": "grid",
        "source_size": [w, h],
        "output_size": [out_img.width, out_img.height],
        "downsampled": False,
        "output": str(out_path),
    }, indent=2))


def cmd_edges(args: argparse.Namespace) -> None:
    src = Image.open(args.input).convert("RGB")
    out_img = render_edge_image(src, mode=args.mode, sigma=args.sigma, gain=args.gain)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path, format="PNG", compress_level=3)
    print(json.dumps({
        "command": "edges",
        "mode": args.mode,
        "source_size": [src.width, src.height],
        "output_size": [out_img.width, out_img.height],
        "downsampled": False,
        "output": str(out_path),
    }, indent=2))


def _export_native_crop(
    src: Image.Image,
    name: str,
    box_frac: list[float] | tuple[float, float, float, float],
    out_dir: Path,
    *,
    min_side: int = 0,
    with_edges: bool = True,
    with_grid: bool = True,
    grid_step: float = 0.1,
    sigma: float = 0.85,
    gain: float = 2.0,
) -> dict[str, Any]:
    w, h = src.size
    x0 = max(0, min(w - 1, int(round(float(box_frac[0]) * w))))
    y0 = max(0, min(h - 1, int(round(float(box_frac[1]) * h))))
    x1 = max(x0 + 1, min(w, int(round(float(box_frac[2]) * w))))
    y1 = max(y0 + 1, min(h, int(round(float(box_frac[3]) * h))))
    crop = src.crop((x0, y0, x1, y1))
    native_size = [crop.width, crop.height]

    # NEVER downsample! Only optionally upscale tiny focal crops (e.g. eye sockets) for crisp visual inspection.
    inspect_img = crop
    scale_used = 1.0
    if min_side > 0 and max(crop.width, crop.height) < min_side:
        scale_used = min(3.0, min_side / float(max(crop.width, crop.height)))
        inspect_img = crop.resize(
            (int(round(crop.width * scale_used)), int(round(crop.height * scale_used))),
            Image.Resampling.LANCZOS,
        )
        # Mild unsharp mask to keep pixel/contour crisper when inspecting tiny source regions
        inspect_img = inspect_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=85, threshold=2))

    safe_name = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)
    native_path = out_dir / f"{safe_name}_native.png"
    inspect_img.save(native_path, format="PNG", compress_level=3)
    entry: dict[str, Any] = {
        "name": name,
        "box_frac": [round(float(v), 4) for v in box_frac],
        "box_px": [x0, y0, x1, y1],
        "native_crop_px": native_size,
        "inspect_scale": round(scale_used, 3),
        "native_png": str(native_path),
    }
    if with_grid:
        # v35 Post-Zoom 3-Panel Multimodal Strip: [Clean Native Color Cutout | Post-Zoom 0.10 Grid | Crisp 1 px NMS Color-Context Contours]
        grid_path = out_dir / f"{safe_name}_zoom_grid_edges.png"
        grid_img = render_grid_panel(
            inspect_img,
            step=grid_step,
            dual_edges=with_edges,
            include_clean_cutout=True,
            sigma=sigma,
            gain=gain,
        )
        grid_img.save(grid_path, format="PNG", compress_level=3)
        entry["zoom_grid_edges_png"] = str(grid_path)
    elif with_edges:
        dual_path = out_dir / f"{safe_name}_native_and_edges.png"
        dual_img = render_edge_image(inspect_img, mode="dual", sigma=sigma, gain=gain)
        dual_img.save(dual_path, format="PNG", compress_level=3)
        entry["dual_edges_png"] = str(dual_path)
    return entry


def cmd_zoom(args: argparse.Namespace) -> None:
    src = Image.open(args.input).convert("RGB")
    if getattr(args, "autocrop", False):
        src = _autocrop_letterbox(src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    boxes_raw = Path(args.boxes).read_text() if Path(args.boxes).exists() else args.boxes
    boxes = json.loads(boxes_raw)
    if not isinstance(boxes, dict):
        raise SystemExit("--boxes must be a JSON object mapping region_name -> [x1, y1, x2, y2]")

    manifest = []
    for name, rect in boxes.items():
        manifest.append(
            _export_native_crop(
                src,
                name,
                rect,
                out_dir,
                min_side=int(args.min_side),
                with_edges=bool(args.with_edges),
                with_grid=bool(getattr(args, "with_grid", True)),
                grid_step=float(getattr(args, "step", 0.1)),
                sigma=float(args.sigma),
                gain=float(args.gain),
            )
        )
    print(json.dumps({"command": "zoom", "source_size": [src.width, src.height], "crops": manifest}, indent=2))


def cmd_trace(args: argparse.Namespace) -> None:
    src = Image.open(args.input).convert("RGB")
    if getattr(args, "autocrop", False):
        src = _autocrop_letterbox(src)
    w, h = src.size
    spec = json.loads(Path(args.spec).read_text())

    edge_blend = float(args.edge_blend)
    if edge_blend > 0.0:
        edge_ov = render_edge_image(src, mode="overlay", sigma=float(args.sigma), gain=float(args.gain))
        base_arr = (
            np.asarray(src, dtype=np.float64) * (1.0 - edge_blend)
            + np.asarray(edge_ov, dtype=np.float64) * edge_blend
        )
        work = Image.fromarray(np.clip(np.round(base_arr), 0, 255).astype(np.uint8), mode="RGB")
    else:
        work = src.copy()

    header_h = 46 if (spec.get("title") or spec.get("calipers")) else 0
    canvas = Image.new("RGB", (w, h + header_h), (14, 18, 26))
    canvas.paste(work, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(12, min(16, int(min(w, h) * 0.016))))
    small_font = _load_font(max(11, min(14, int(min(w, h) * 0.0135))))

    def to_px(xf: float, yf: float) -> tuple[float, float]:
        return (float(xf) * (w - 1), header_h + float(yf) * (h - 1))

    canvas_bounds = (canvas.width, canvas.height)
    occupied_pills: list[tuple[int, int, int, int]] = []
    if header_h > 0:
        occupied_pills.append((0, 0, canvas.width, header_h + 2))
    named_pts: dict[str, tuple[float, float]] = {}
    telemetry: dict[str, Any] = {
        "command": "trace",
        "source_size": [w, h],
        "chains": {},
        "calipers": {},
        "ratios": {},
        "boxes": [],
    }

    # 1. Focal Bounding Boxes (<box> primitives)
    crop_dir = Path(args.export_crops_dir) if args.export_crops_dir else None
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
    for idx, b in enumerate(spec.get("boxes", [])):
        bname = str(b.get("name", f"box_{idx}"))
        rect = b["rect"]
        col = _parse_color(b.get("color"), (255, 224, 96))
        x0, y0 = to_px(rect[0], rect[1])
        x1, y1 = to_px(rect[2], rect[3])
        _draw_corner_box(
            draw,
            (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
            color=col,
            width=3,
            label=f"[{bname}]",
            font=small_font,
        )
        if crop_dir is not None:
            telemetry["boxes"].append(
                _export_native_crop(
                    src,
                    bname,
                    rect,
                    crop_dir,
                    min_side=int(args.min_side),
                    with_edges=True,
                    sigma=float(args.sigma),
                    gain=float(args.gain),
                )
            )
        else:
            telemetry["boxes"].append({"name": bname, "rect": rect})

    # 2. Dashed Skeleton Centerline Chains (`- - -`) & Joint Waypoints (<point>)
    for c_idx, ch in enumerate(spec.get("chains", [])):
        cname = str(ch.get("name", f"chain_{c_idx}"))
        col = _parse_color(ch.get("color"), DEFAULT_CHAIN_COLORS[c_idx % len(DEFAULT_CHAIN_COLORS)])
        pts_spec = ch.get("points", [])
        pts_px: list[tuple[str, float, float]] = []
        for p_idx, item in enumerate(pts_spec):
            if len(item) == 3:
                plbl, xf, yf = str(item[0]), float(item[1]), float(item[2])
            else:
                plbl, xf, yf = f"{cname}_{p_idx}", float(item[0]), float(item[1])
            px, py = to_px(xf, yf)
            pts_px.append((plbl, px, py))
            named_pts[plbl] = (px, py)

        seg_lens: list[float] = []
        joint_angles: dict[str, float] = {}
        signed_turns_deg: dict[str, float] = {}
        for i in range(len(pts_px) - 1):
            p0 = (pts_px[i][1], pts_px[i][2])
            p1 = (pts_px[i + 1][1], pts_px[i + 1][2])
            # Dark shadow stroke underneath for contrast on both bright and dark images
            _draw_dashed_line(draw, p0, p1, fill=(10, 12, 18), width=5, dash=12, gap=7)
            _draw_dashed_line(draw, p0, p1, fill=col, width=3, dash=12, gap=7)
            seg_lens.append(math.hypot(p1[0] - p0[0], p1[1] - p0[1]))

        # Compute signed 2D curvature concavity relative to chain horizontal flow (invariant to left/right facing!).
        # In world-up coordinates (y_up = -y_img): positive signed bend = CONCAVE_UP (U), negative = CONCAVE_DOWN (arch).
        dx_total = (pts_px[-1][1] - pts_px[0][1]) if len(pts_px) >= 2 else 1.0
        flow_sign = 1.0 if dx_total >= 0.0 else -1.0
        turn_vals: list[float] = []

        for i, (plbl, px, py) in enumerate(pts_px):
            r = 5
            draw.ellipse([px - r - 1, py - r - 1, px + r + 1, py + r + 1], fill=(12, 14, 20), outline=col, width=2)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255))
            tag = plbl
            if ch.get("measure_angles", True) and 0 < i < len(pts_px) - 1:
                v_prev = np.array([pts_px[i][1] - pts_px[i - 1][1], -(pts_px[i][2] - pts_px[i - 1][2])], dtype=np.float64)
                v_next = np.array([pts_px[i + 1][1] - pts_px[i][1], -(pts_px[i + 1][2] - pts_px[i][2])], dtype=np.float64)
                n1, n2 = float(np.linalg.norm(v_prev)), float(np.linalg.norm(v_next))
                if n1 > 1e-4 and n2 > 1e-4:
                    # Interior angle between incoming (-v_prev) and outgoing (+v_next)
                    cos_int = float(np.clip(np.dot(-v_prev, v_next) / (n1 * n2), -1.0, 1.0))
                    ang_deg = math.degrees(math.acos(cos_int))
                    joint_angles[plbl] = round(ang_deg, 1)
                    # Signed deflection from v_prev -> v_next (+ = curves UP, - = arches DOWN)
                    cross_z = float(v_prev[0] * v_next[1] - v_prev[1] * v_next[0]) * flow_sign
                    dot_d = float(np.dot(v_prev, v_next))
                    turn_deg = math.degrees(math.atan2(cross_z, dot_d))
                    signed_turns_deg[plbl] = round(turn_deg, 1)
                    turn_vals.append(turn_deg)
                    tag = f"{plbl} ({ang_deg:.0f}°)"
            _draw_label_pill(
                draw,
                (px + 8, py - 10),
                tag,
                fg=col,
                font=small_font,
                bounds=canvas_bounds,
                occupied=occupied_pills,
            )

        if turn_vals:
            pos_turns = [v for v in turn_vals if v > 5.0]
            neg_turns = [v for v in turn_vals if v < -5.0]
            if pos_turns and neg_turns:
                curv_cls = "S_INFLECTION (~)"
            elif pos_turns:
                curv_cls = "CONCAVE_UP (U)"
            elif neg_turns:
                curv_cls = "CONCAVE_DOWN (arch)"
            else:
                curv_cls = "STRAIGHT (--)"
        else:
            curv_cls = "SEGMENT"

        if ch.get("label_concavity", False) and len(pts_px) >= 3:
            mid_i = len(pts_px) // 2
            _draw_label_pill(
                draw,
                (pts_px[mid_i][1] + 10, pts_px[mid_i][2] - 26),
                f"{cname}: {curv_cls}",
                fg=col,
                font=small_font,
                bounds=canvas_bounds,
                occupied=occupied_pills,
            )

        telemetry["chains"][cname] = {
            "segment_lengths_px": [round(s, 1) for s in seg_lens],
            "total_length_px": round(sum(seg_lens), 1),
            "joint_angles_deg": joint_angles,
            "signed_deflections_deg": signed_turns_deg,
            "curvature_concavity": curv_cls,
        }

    # 3. Dotted Proportion Calipers (`· · ·`)
    for cal in spec.get("calipers", []):
        cal_name = str(cal["name"])
        col = _parse_color(cal.get("color"), (255, 235, 110))
        a = cal["a"]
        b = cal["b"]
        pa = named_pts[a] if isinstance(a, str) else to_px(a[0], a[1])
        pb = named_pts[b] if isinstance(b, str) else to_px(b[0], b[1])
        _draw_dashed_line(draw, pa, pb, fill=(10, 12, 18), width=4, dash=4, gap=6)
        _draw_dashed_line(draw, pa, pb, fill=col, width=2, dash=4, gap=6)
        dist_px = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        telemetry["calipers"][cal_name] = round(dist_px, 1)
        mid = ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5)
        _draw_label_pill(
            draw,
            (mid[0] + 4, mid[1] - 8),
            f"{cal_name}:{dist_px:.0f}px",
            fg=col,
            font=small_font,
            bounds=canvas_bounds,
            occupied=occupied_pills,
        )

    # 4. Automatic Ratio Calipers (compare chain totals or caliper lengths)
    metric_lookup: dict[str, float] = {}
    for k, v in telemetry["chains"].items():
        metric_lookup[k] = float(v["total_length_px"])
    for k, v in telemetry["calipers"].items():
        metric_lookup[k] = float(v)

    ratio_strs: list[str] = []
    for rspec in spec.get("ratios", []):
        rname = str(rspec["name"])
        num = metric_lookup.get(str(rspec["num"]), 0.0)
        den = metric_lookup.get(str(rspec["den"]), 0.0)
        if den > 1e-6:
            val = num / den
            telemetry["ratios"][rname] = round(val, 3)
            ratio_strs.append(f"{rname}={val:.2f}x")

    if header_h > 0:
        title = str(spec.get("title", "VISUAL PRIMITIVES STRUCTURAL TRACE (NATIVE 1:1 RESOLUTION)"))
        if ratio_strs:
            title = f"{title}   |   " + "  ".join(ratio_strs)
        draw.text((12, 13), title, fill=(130, 245, 255), font=font)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", compress_level=3)
    telemetry["output"] = str(out_path)
    telemetry["downsampled"] = False
    print(json.dumps(telemetry, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual Primitives & Native-Resolution Reference Preprocessor (v33)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_grid = sub.add_parser("grid", help="Overlay 0.0-1.0 coordinate grid at native 1:1 resolution.")
    p_grid.add_argument("input")
    p_grid.add_argument("output")
    p_grid.add_argument("--step", type=float, default=0.1)
    p_grid.add_argument("--dual-edges", action="store_true")
    p_grid.add_argument("--autocrop", action="store_true")
    p_grid.add_argument("--sigma", type=float, default=1.0)
    p_grid.add_argument("--gain", type=float, default=1.65)
    p_grid.set_defaults(func=cmd_grid)

    p_edges = sub.add_parser("edges", help="Extract structural edges/contours at native 1:1 resolution.")
    p_edges.add_argument("input")
    p_edges.add_argument("output")
    p_edges.add_argument("--mode", choices=("color_ink", "ink", "overlay", "dual"), default="color_ink")
    p_edges.add_argument("--sigma", type=float, default=1.0)
    p_edges.add_argument("--gain", type=float, default=1.65)
    p_edges.set_defaults(func=cmd_edges)

    p_zoom = sub.add_parser("zoom", help="Zoom FIRST into regions, THEN overlay fine grid + structural outline/edge trace.")
    p_zoom.add_argument("input")
    p_zoom.add_argument("out_dir")
    p_zoom.add_argument("--boxes", required=True, help="JSON string or path mapping name -> [x1,y1,x2,y2]")
    p_zoom.add_argument("--autocrop", action="store_true")
    p_zoom.add_argument("--min-side", type=int, default=480)
    p_zoom.add_argument("--with-edges", action="store_true", default=True)
    p_zoom.add_argument("--with-grid", action="store_true", default=True)
    p_zoom.add_argument("--step", type=float, default=0.1)
    p_zoom.add_argument("--sigma", type=float, default=0.9)
    p_zoom.add_argument("--gain", type=float, default=1.65)
    p_zoom.set_defaults(func=cmd_zoom)

    p_trace = sub.add_parser("trace", help="Render DeepSeek Visual Primitives (dashed skeleton, angles, calipers, boxes).")
    p_trace.add_argument("input")
    p_trace.add_argument("spec")
    p_trace.add_argument("output")
    p_trace.add_argument("--edge-blend", type=float, default=0.25)
    p_trace.add_argument("--export-crops-dir", default=None)
    p_trace.add_argument("--autocrop", action="store_true")
    p_trace.add_argument("--min-side", type=int, default=480)
    p_trace.add_argument("--sigma", type=float, default=1.0)
    p_trace.add_argument("--gain", type=float, default=1.65)
    p_trace.set_defaults(func=cmd_trace)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
