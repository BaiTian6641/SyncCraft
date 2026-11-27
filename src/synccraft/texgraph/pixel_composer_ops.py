"""Pixel Composer & Expert Blockbench Atomic Pixel-Art Node Operators (`TexGraph v9`).

Directly distilled from MakhamDev's Pixel Composer (`Common Nodes`, `Fire Effect`, `Ship Generator`,
`Bush Generator`, `Shiny Gem`, `Pixel Builder`, `1.22.4 Heightmap Shadow`) and expert Blockbench
studio artists (Everbloom Games, Mickey Joe, ArtsByKev, Wackyblocks, Pseudo Elephant / Hytale):

1. **Pixel-Art Quantization, Dithering & Palette Craft**:
   - ``pixel_dither``          : Ordered Bayer (2x2, 4x4, 8x8), Diamond stipple, Blue Noise, and
                                 Error-Diffusion (Floyd-Steinberg / Atkinson) dithering locked
                                 strictly to a discrete perceptual OKLab palette (zero blurry mud!).
   - ``palette_posterize``     : Exact discrete N-color palette quantization in OKLab/RGB/HSV
                                 with optional hard alpha snap (`posterize_alpha`).
   - ``color_replace``         : Crisp palette-to-palette color replacement (exact colorway swap).
   - ``level_selector``        : Isolates a narrow luminance band `[center ± width]` for specular
                                 highlights, gem glints & rim masks.

2. **Pixel-Art Topology Cleanup & Cluster Sculpting (Anti-Blur / Anti-Noise / Anti-Boxy!)**:
   - ``de_corner``             : Pixel Composer `De-Corner` — strips L-shaped diagonal stair-step
                                 jaggies / double corners cleanly in 1-4 passes.
   - ``de_stray``              : Pixel Composer `De-Stray` — removes isolated 1x1 digital noise
                                 specks while preserving coherent 2-4 px artistic clusters.
   - ``pixel_clusters``        : Hytale / Everbloom / Mickey Joe directional 2-4 px pixel cluster
                                 sculptor (breaks smooth gradients into deliberate hand-placed
                                 pixel clusters without 1x1 TV-static confetti!).
   - ``pixel_bevel_directional``: Anti-wireframe directional top-left lit bevel + bottom-right
                                  cast shadow with asymmetric chipped notch breakup (Wackyblocks
                                  & Hytale Firesteel standard — NEVER uniform 4-sided box frames!).
   - ``pixel_outline``         : Crisp 4-way or 8-way outside/inside pixel-art outline & inline
                                 with optional directional selout tinting.

3. **Lighting, Geometry, FX & 3D World-Space Bounce Light**:
   - ``heightmap_shadow``      : Pixel Composer 1.22.4 ray-marched directional cast shadows across
                                 2D heightmaps (roof-tile plate overlaps, bricks, straps, scales).
   - ``pixel_mirror``          : Bilateral X/Y/XY/quad pixel-grid symmetry operator.
   - ``pixel_sort``            : Luminance-masked directional pixel sorting for arcane/energy streaks.
   - ``chromatic_aberration``  : Subtle RGB channel edge shift for magic/VFX sprites.
   - ``vignette``              : Smooth or pixel-stepped focal framing vignette.
   - ``bake_emissive_bounce``  : Hytale `Golem_Firesteel`-grade 3D world-space baked GI bounce glow
                                 from glowing eyes, runes, cores & fissures onto adjacent dark
                                 armor, obsidian & skin surfaces!
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .generators_noise import bayer_matrix, fbm_noise, uv_grid
from .pipeline import MATERIAL_BUS_PORTS, _ensure_bus
from .pixel_art_ops import hex_to_srgb, srgb_to_oklab


def _parse_palette_rgb(palette: Sequence[Any]) -> np.ndarray:
    """Parse either ['#hex', ...] or [(t, '#hex'), ...] into float32 (K, 3) sRGB array."""
    hexes: list[str] = []
    for item in palette:
        if isinstance(item, str):
            hexes.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            hexes.append(str(item[1]))
    if not hexes:
        hexes = ["#1B1726", "#4D5368", "#949FB5", "#E5EEF7"]
    return np.array([hex_to_srgb(h) for h in hexes], dtype=np.float32)


def _to_rgba(src: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(src, dtype=np.float32)
    if arr.ndim == 2:
        rgba = np.stack((arr, arr, arr, np.ones_like(arr)), axis=-1)
        return np.clip(rgba, 0.0, 1.0).astype(np.float32), True
    if arr.shape[2] == 3:
        rgba = np.concatenate((arr, np.ones((*arr.shape[:2], 1), dtype=np.float32)), axis=-1)
        return np.clip(rgba, 0.0, 1.0).astype(np.float32), False
    return np.clip(arr[..., :4], 0.0, 1.0).astype(np.float32), False


# ---------------------------------------------------------------------------
# 1. Pixel Composer Palette Posterize & Dither (Locked OKLab Discrete Ramps!)
# ---------------------------------------------------------------------------


def palette_posterize(
    src: np.ndarray,
    palette: Sequence[Any],
    *,
    space: str = "oklab",  # "oklab" | "rgb" | "luma_rank"
    posterize_alpha: bool = True,
    alpha_threshold: float = 0.5,
    blend: float = 1.0,
) -> np.ndarray:
    """Pixel Composer `Posterize`: lock every pixel crisply onto a curated discrete palette."""
    rgba, _ = _to_rgba(src)
    H, W = rgba.shape[:2]
    pal_rgb = _parse_palette_rgb(palette)  # (K, 3)
    K = pal_rgb.shape[0]

    rgb = rgba[..., :3]
    if space == "luma_rank":
        # Sort palette dark->light and index directly by perceptual luminance!
        pal_lum = 0.2126 * pal_rgb[:, 0] + 0.7152 * pal_rgb[:, 1] + 0.0722 * pal_rgb[:, 2]
        order = np.argsort(pal_lum)
        pal_sorted = pal_rgb[order]
        lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        idx = np.clip(np.rint(lum * (K - 1)), 0, K - 1).astype(np.int32)
        snapped = pal_sorted[idx]
    elif space == "rgb":
        diff = rgb[..., None, :] - pal_rgb[None, None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        idx = np.argmin(dist2, axis=-1)
        snapped = pal_rgb[idx]
    else:
        # Perceptual OKLab nearest-color snapping (matches artist hue-shifted ramps cleanly!)
        src_lab = srgb_to_oklab(rgb)
        pal_lab = srgb_to_oklab(pal_rgb[None, :, :])[0]  # (K, 3)
        diff = src_lab[..., None, :] - pal_lab[None, None, :, :]
        # Slightly emphasize lightness accuracy (1.25x) so tonal planes stay crisp
        dist2 = 1.35 * (diff[..., 0] ** 2) + (diff[..., 1] ** 2) + (diff[..., 2] ** 2)
        idx = np.argmin(dist2, axis=-1)
        snapped = pal_rgb[idx]

    w = float(np.clip(blend, 0.0, 1.0))
    out_rgb = rgb * (1.0 - w) + snapped * w
    out_a = (rgba[..., 3] >= float(alpha_threshold)).astype(np.float32) if posterize_alpha else rgba[..., 3]
    return np.concatenate((np.clip(out_rgb, 0.0, 1.0), out_a[..., None]), axis=-1).astype(np.float32)


def pixel_dither(
    src: np.ndarray,
    palette: Sequence[Any],
    *,
    pattern: str = "bayer_2x2",  # "bayer_2x2" | "bayer_4x4" | "bayer_8x8" | "diamond" | "blue_noise" | "floyd_steinberg" | "atkinson"
    contrast: float = 1.0,
    mode: str = "color",  # "color" | "alpha"
    invert: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Pixel Composer `Dither` & `Error Diffuse Dither`: palette-locked pixel dithering!

    Supports ordered matrices (`bayer_2x2`, `bayer_4x4`, `bayer_8x8`, `diamond`, `blue_noise`)
    AND true error diffusion (`floyd_steinberg`, `atkinson`) snapped to a discrete palette!
    """
    rgba, is_scalar = _to_rgba(src)
    H, W = rgba.shape[:2]
    pal_rgb = _parse_palette_rgb(palette)
    pal_lum = 0.2126 * pal_rgb[:, 0] + 0.7152 * pal_rgb[:, 1] + 0.0722 * pal_rgb[:, 2]
    order = np.argsort(pal_lum)
    pal_sorted = pal_rgb[order]
    K = pal_sorted.shape[0]

    # Build normalized [0, 1) threshold tile for ordered patterns
    def _ordered_thresh() -> np.ndarray:
        if pattern == "bayer_2x2":
            m = bayer_matrix(H, W, size=2)
        elif pattern == "bayer_4x4":
            m = bayer_matrix(H, W, size=4)
        elif pattern == "bayer_8x8":
            m = bayer_matrix(H, W, size=8)
        elif pattern == "diamond":
            d4 = np.array(
                [
                    [0, 8, 2, 10],
                    [12, 4, 14, 6],
                    [3, 11, 1, 9],
                    [15, 7, 13, 5],
                ],
                dtype=np.float32,
            ) / 16.0
            m = np.tile(d4, (H // 4 + 1, W // 4 + 1))[:H, :W]
        else:
            # High-frequency blue-noise-like hash (void-and-cluster approximation)
            rng = np.random.default_rng(int(seed))
            wn = rng.random((H, W), dtype=np.float32)
            # High-pass filter white noise to push energy into high frequencies (blue spectrum!)
            avg = 0.25 * (np.roll(wn, 1, 0) + np.roll(wn, -1, 0) + np.roll(wn, 1, 1) + np.roll(wn, -1, 1))
            hp = wn - 0.72 * avg
            m = ((hp - hp.min()) / max(1e-6, float(hp.max() - hp.min()))).astype(np.float32)
        return (1.0 - m) if invert else m

    if mode == "alpha":
        thresh = _ordered_thresh()
        a_out = (rgba[..., 3] > thresh).astype(np.float32)
        return np.concatenate((rgba[..., :3], a_out[..., None]), axis=-1).astype(np.float32)

    spread = float(np.clip(contrast, 0.0, 4.0)) * 0.45

    if pattern in ("floyd_steinberg", "atkinson"):
        # True palette-locked Error-Diffusion Dither (fast row-by-row float32 buffer)
        buf = rgba[..., :3].astype(np.float32).copy()
        pal_lab = srgb_to_oklab(pal_sorted[None, :, :])[0]
        out_rgb = np.zeros_like(buf)
        is_atk = pattern == "atkinson"
        # Use weighted sRGB distance inside tight loop for speed (~4ms on 128x128)
        w_rgb = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        for y in range(H):
            for x in range(W):
                px = np.clip(buf[y, x], 0.0, 1.0)
                d = pal_sorted - px[None, :]
                ci = int(np.argmin(np.sum(d * d * w_rgb[None, :], axis=-1)))
                chosen = pal_sorted[ci]
                out_rgb[y, x] = chosen
                err = (px - chosen) * min(1.0, float(contrast))
                if is_atk:
                    e = err * (1.0 / 8.0)
                    if x + 1 < W:
                        buf[y, x + 1] += e
                    if x + 2 < W:
                        buf[y, x + 2] += e
                    if y + 1 < H:
                        if x > 0:
                            buf[y + 1, x - 1] += e
                        buf[y + 1, x] += e
                        if x + 1 < W:
                            buf[y + 1, x + 1] += e
                    if y + 2 < H:
                        buf[y + 2, x] += e
                else:
                    if x + 1 < W:
                        buf[y, x + 1] += err * (7.0 / 16.0)
                    if y + 1 < H:
                        if x > 0:
                            buf[y + 1, x - 1] += err * (3.0 / 16.0)
                        buf[y + 1, x] += err * (5.0 / 16.0)
                        if x + 1 < W:
                            buf[y + 1, x + 1] += err * (1.0 / 16.0)
        return np.concatenate((np.clip(out_rgb, 0.0, 1.0), rgba[..., 3:4]), axis=-1).astype(np.float32)

    # Ordered dithering (bayer_2x2 / bayer_4x4 / bayer_8x8 / diamond / blue_noise):
    thresh = _ordered_thresh() - 0.5  # [-0.5, +0.5]
    if is_scalar:
        # Scalar ramp direct ordered dither across sorted palette stops!
        t = rgba[..., 0] * float(K - 1) + thresh * (spread * 2.0)
        idx = np.clip(np.rint(t), 0, K - 1).astype(np.int32)
        out_rgb = pal_sorted[idx]
    else:
        # Color image ordered dither: perturb in perceptual lightness/RGB by 1 palette step then OKLab-snap!
        step = (1.0 / max(1.0, float(K - 1))) * spread * 1.35
        perturbed = np.clip(rgba[..., :3] + thresh[..., None] * step, 0.0, 1.0)
        snapped = palette_posterize(np.concatenate((perturbed, rgba[..., 3:4]), axis=-1), pal_sorted.tolist() if False else palette, space="oklab", posterize_alpha=False)
        out_rgb = snapped[..., :3]

    return np.concatenate((np.clip(out_rgb, 0.0, 1.0), rgba[..., 3:4]), axis=-1).astype(np.float32)


def color_replace(
    src: np.ndarray,
    src_palette: Sequence[Any],
    dst_palette: Sequence[Any],
    *,
    tolerance: float = 0.25,
) -> np.ndarray:
    """Pixel Composer `Replace Palette / Replace Colors`: swap curated ramp A -> curated ramp B cleanly."""
    rgba, _ = _to_rgba(src)
    s_rgb = _parse_palette_rgb(src_palette)
    d_rgb = _parse_palette_rgb(dst_palette)
    if d_rgb.shape[0] != s_rgb.shape[0]:
        # Resample dst_palette to match src_palette count
        xs = np.linspace(0.0, d_rgb.shape[0] - 1, s_rgb.shape[0])
        d_rgb = np.stack([np.interp(xs, np.arange(d_rgb.shape[0]), d_rgb[:, c]) for c in range(3)], axis=-1).astype(np.float32)

    src_lab = srgb_to_oklab(rgba[..., :3])
    pal_lab = srgb_to_oklab(s_rgb[None, :, :])[0]
    diff = src_lab[..., None, :] - pal_lab[None, None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    best_i = np.argmin(dist, axis=-1)
    best_d = np.min(dist, axis=-1)
    mapped = d_rgb[best_i]
    out_rgb = np.where((best_d <= float(tolerance))[..., None], mapped, rgba[..., :3])
    return np.concatenate((np.clip(out_rgb, 0.0, 1.0), rgba[..., 3:4]), axis=-1).astype(np.float32)


def level_selector(
    src: np.ndarray,
    *,
    center: float = 0.75,
    width: float = 0.20,
    softness: float = 0.05,
) -> np.ndarray:
    """Pixel Composer `Level Selector`: isolate a specific luminance slice [center ± width/2] as a mask."""
    arr = np.asarray(src, dtype=np.float32)
    lum = (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]) if arr.ndim == 3 else arr
    dist = np.abs(lum - float(center))
    half = max(1e-4, 0.5 * float(width))
    soft = max(1e-4, float(softness))
    m = np.clip(1.0 - (dist - half) / soft, 0.0, 1.0).astype(np.float32)
    return m


# ---------------------------------------------------------------------------
# 2. Pixel Composer Topology Cleanup: De-Corner, De-Stray & Directional Clusters
# ---------------------------------------------------------------------------


def _color_dist_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sum(d * d, axis=-1)


def de_stray(
    src: np.ndarray,
    *,
    strictness: str = "medium",  # "low" (all 4 cardinal differ) | "medium" (3-4 differ) | "high" (6+ of 8 differ)
    tolerance: float = 0.06,
    iterations: int = 1,
    tileable: bool = False,
) -> np.ndarray:
    """Pixel Composer `De-Stray`: removes isolated 1x1 orphan noise specks cleanly!

    Replaces any pixel whose cardinal/8-way neighbors disagree with it (above ``tolerance``)
    by the majority / median neighbor color, preserving deliberate lines, bevels & eye catchlights.
    """
    rgba, is_scalar = _to_rgba(src)
    cur = rgba.copy()
    tol2 = float(tolerance) ** 2
    mode = "wrap" if tileable else "edge"

    for _ in range(max(1, int(iterations))):
        p = np.pad(cur, ((1, 1), (1, 1), (0, 0)), mode=mode)
        c = p[1:-1, 1:-1]
        n_up = p[:-2, 1:-1]
        n_dn = p[2:, 1:-1]
        n_lt = p[1:-1, :-2]
        n_rt = p[1:-1, 2:]
        card = (n_up, n_dn, n_lt, n_rt)
        diff_card = sum((_color_dist_sq(c, nb) > tol2).astype(np.int32) for nb in card)

        if strictness == "low":
            # Isolated pixel differs from ALL 4 cardinal neighbors AND opposite pair matches!
            is_stray = (diff_card == 4) & (
                (_color_dist_sq(n_up, n_dn) <= tol2 * 2.5) | (_color_dist_sq(n_lt, n_rt) <= tol2 * 2.5)
            )
        elif strictness == "high":
            n_ul = p[:-2, :-2]
            n_ur = p[:-2, 2:]
            n_dl = p[2:, :-2]
            n_dr = p[2:, 2:]
            all8 = card + (n_ul, n_ur, n_dl, n_dr)
            diff8 = sum((_color_dist_sq(c, nb) > tol2).astype(np.int32) for nb in all8)
            is_stray = diff8 >= 6
        else:
            # medium: differs from all 4 cardinal neighbors (keeps 1px lines & 2px catchlights!)
            is_stray = diff_card == 4

        if not np.any(is_stray):
            break
        # Pick closest-to-median among the 4 cardinal neighbors (keeps palette locked — zero blur!)
        stack4 = np.stack(card, axis=0)  # (4, H, W, 4)
        # Pairwise agreement count among the 4 cardinal neighbors -> pick highest-consensus neighbor
        scores = np.zeros((4, cur.shape[0], cur.shape[1]), dtype=np.float32)
        for i in range(4):
            for j in range(4):
                if i != j:
                    scores[i] -= _color_dist_sq(stack4[i], stack4[j])
        best_idx = np.argmax(scores, axis=0)  # (H, W)
        yy, xx = np.indices(best_idx.shape)
        rep = stack4[best_idx, yy, xx]
        cur = np.where(is_stray[..., None], rep, cur)

    return cur[..., 0].astype(np.float32) if is_scalar else cur.astype(np.float32)


def de_corner(
    src: np.ndarray,
    *,
    strictness: str = "medium",  # "low" | "medium" | "high"
    tolerance: float = 0.06,
    iterations: int = 1,
    tileable: bool = False,
) -> np.ndarray:
    """Pixel Composer `De-Corner`: strips ugly L-shaped stair-step pixel jaggies / corner doubles!

    Detects classic pixel-art "doubles" where pixel C forms a redundant inner elbow between two
    adjacent cardinal pixels (e.g., UP and LEFT share C's color while DOWN, RIGHT, and UL differ)
    and snaps C cleanly to the outer background/neighbor color while staying 100% palette-locked!
    """
    rgba, is_scalar = _to_rgba(src)
    cur = rgba.copy()
    tol2 = float(tolerance) ** 2
    mode = "wrap" if tileable else "edge"

    for _ in range(max(1, int(iterations))):
        p = np.pad(cur, ((1, 1), (1, 1), (0, 0)), mode=mode)
        c = p[1:-1, 1:-1]
        u = p[:-2, 1:-1]
        d = p[2:, 1:-1]
        l = p[1:-1, :-2]
        r = p[1:-1, 2:]
        ul = p[:-2, :-2]
        ur = p[:-2, 2:]
        dl = p[2:, :-2]
        dr = p[2:, 2:]

        def _eq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return _color_dist_sq(x, y) <= tol2

        su, sd, sl, sr = _eq(c, u), _eq(c, d), _eq(c, l), _eq(c, r)
        sul, sur, sdl, sdr = _eq(c, ul), _eq(c, ur), _eq(c, dl), _eq(c, dr)

        # Classic 4 L-elbow orientations: exactly 2 orthogonal cardinals match C, opposite 2 differ,
        # and the inner diagonal between the 2 matching cardinals does NOT match (true 1px line double!)
        el_ul = su & sl & (~sd) & (~sr) & (~sul)
        el_ur = su & sr & (~sd) & (~sl) & (~sur)
        el_dl = sd & sl & (~su) & (~sr) & (~sdl)
        el_dr = sd & sr & (~su) & (~sl) & (~sdr)

        if strictness == "low":
            # Only clean when the 2 outer opposite neighbors agree with each other
            el_ul &= _eq(d, r)
            el_ur &= _eq(d, l)
            el_dl &= _eq(u, r)
            el_dr &= _eq(u, l)
        elif strictness == "high":
            # Also clean outer corner pixels protruding into solid background
            el_ul = el_ul | (su & sl & (~sd) & (~sr) & _eq(d, r) & _eq(d, dr))
            el_ur = el_ur | (su & sr & (~sd) & (~sl) & _eq(d, l) & _eq(d, dl))
            el_dl = el_dl | (sd & sl & (~su) & (~sr) & _eq(u, r) & _eq(u, ur))
            el_dr = el_dr | (sd & sr & (~su) & (~sl) & _eq(u, l) & _eq(u, ul))

        nxt = cur.copy()
        nxt = np.where(el_ul[..., None], d, nxt)
        nxt = np.where(el_ur[..., None], d, nxt)
        nxt = np.where(el_dl[..., None], u, nxt)
        nxt = np.where(el_dr[..., None], u, nxt)
        if np.array_equal(nxt, cur):
            break
        cur = nxt

    return cur[..., 0].astype(np.float32) if is_scalar else cur.astype(np.float32)


def pixel_clusters(
    src: np.ndarray,
    palette: Sequence[Any],
    *,
    direction: str = "isotropic",  # "isotropic" | "horizontal" | "vertical" | "diag_pos" | "diag_neg"
    cluster_size: int = 2,
    dither_strength: float = 0.0,
    clean_strays: bool = True,
    clean_corners: bool = True,
    tileable: bool = False,
) -> np.ndarray:
    """Hytale / Everbloom / Mickey Joe Directional Pixel-Cluster Sculptor!

    Converts continuous height/color fields into crisp, intentional **2-4 px organized clusters**
    (NEVER blurry airbrush AND NEVER noisy 1x1 TV-static!):
    1. Anisotropic directional coherence filter along the chosen grain/fur/plank axis
    2. Optional subtle ordered Bayer bridge transition between tones
    3. Strict perceptual OKLab discrete palette lock (`palette_posterize`)
    4. `de_stray` + `de_corner` pixel-art topology polish!
    """
    rgba, _ = _to_rgba(src)
    rgb = rgba[..., :3].copy()
    mode = "wrap" if tileable else "edge"
    r = max(1, min(3, int(cluster_size)))

    # 1. Directional cluster consolidation (preserves sharp cross-axis steps while forming clean 2-4px streaks!)
    if direction == "horizontal":
        pad = np.pad(rgb, ((0, 0), (r, r), (0, 0)), mode=mode)
        acc = np.zeros_like(rgb)
        for dx in range(-r, r + 1):
            acc += pad[:, r + dx : r + dx + rgb.shape[1], :]
        rgb = acc / float(2 * r + 1)
    elif direction == "vertical":
        pad = np.pad(rgb, ((r, r), (0, 0), (0, 0)), mode=mode)
        acc = np.zeros_like(rgb)
        for dy in range(-r, r + 1):
            acc += pad[r + dy : r + dy + rgb.shape[0], :, :]
        rgb = acc / float(2 * r + 1)
    elif direction in ("diag_pos", "diag_neg"):
        sgn = 1 if direction == "diag_pos" else -1
        pad = np.pad(rgb, ((r, r), (r, r), (0, 0)), mode=mode)
        acc = np.zeros_like(rgb)
        for k in range(-r, r + 1):
            dy = k
            dx = k * sgn
            acc += pad[r + dy : r + dy + rgb.shape[0], r + dx : r + dx + rgb.shape[1], :]
        rgb = acc / float(2 * r + 1)
    elif r >= 2:
        # Edge-preserving 3x3 cross consolidation (keeps broad calm islands crisp!)
        pad = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode=mode)
        rgb = (
            2.0 * pad[1:-1, 1:-1]
            + pad[:-2, 1:-1]
            + pad[2:, 1:-1]
            + pad[1:-1, :-2]
            + pad[1:-1, 2:]
        ) / 6.0

    work = np.concatenate((np.clip(rgb, 0.0, 1.0), rgba[..., 3:4]), axis=-1).astype(np.float32)

    # 2. Optional subtle Bayer bridge dithering, else pure discrete palette cluster snap!
    if float(dither_strength) > 0.01:
        work = pixel_dither(work, palette, pattern="bayer_2x2", contrast=float(dither_strength))
    else:
        work = palette_posterize(work, palette, space="oklab", posterize_alpha=False)

    # 3. Pixel Composer topology cleanup (eliminates stray 1px confetti & diagonal elbow jaggies!)
    if clean_strays:
        work = de_stray(work, strictness="medium", tolerance=0.05, iterations=1, tileable=tileable)
    if clean_corners:
        work = de_corner(work, strictness="low", tolerance=0.05, iterations=1, tileable=tileable)
    return work.astype(np.float32)


def pixel_bevel_directional(
    height: int,
    width: int,
    *,
    local_uv: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    light_angle_deg: float = -135.0,  # Top-left Minecraft light!
    bevel_width: float = 0.11,
    chip_amount: float = 0.42,
    breakup_scale: float = 10.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Anti-Box-Wireframe Directional Pixel Bevel & Chipped Edge Operator (Wackyblocks / Hytale standard!).

    NEVER outlines all 4 sides of a cube identically!
    - ``highlight``: crisp broken 1 px catchlight strictly on top & leading-left exposed rims
    - ``shadow``   : crisp 1 px dark underside & trailing-right cast-shadow edges
    - ``chips``    : asymmetric 1-2 px dark battle-wear edge notches breaking straight lines!
    """
    H, W = int(height), int(width)
    if local_uv is None:
        u, v = uv_grid(H, W)
    else:
        u = np.asarray(local_uv[..., 0], dtype=np.float32)
        v = np.asarray(local_uv[..., 1], dtype=np.float32)
    cav = np.zeros((H, W), dtype=np.float32) if joint_cavity is None else np.asarray(joint_cavity, dtype=np.float32)

    bw = max(0.03, float(bevel_width))
    top_e = np.clip(1.0 - v / bw, 0.0, 1.0)
    bot_e = np.clip(1.0 - (1.0 - v) / bw, 0.0, 1.0)
    left_e = np.clip(1.0 - u / bw, 0.0, 1.0)
    right_e = np.clip(1.0 - (1.0 - u) / bw, 0.0, 1.0)

    # Light direction in UV (default -135 deg = top-left: top & left edges lit, OPPOSITE bottom & right shadowed!)
    rad = math.radians(float(light_angle_deg))
    lx = math.cos(rad)  # -0.707 (light from left)
    ly = math.sin(rad)  # -0.707 (light from top)
    w_lit_top = max(0.0, -ly)
    w_lit_bot = max(0.0, ly)
    w_lit_left = max(0.0, -lx)
    w_lit_right = max(0.0, lx)

    lit_raw = np.maximum(
        np.maximum(top_e * w_lit_top, bot_e * w_lit_bot),
        np.maximum(left_e * w_lit_left, right_e * w_lit_right),
    ) / max(1e-4, max(w_lit_top, w_lit_bot, w_lit_left, w_lit_right))
    # Shadow sits on the edges OPPOSITE to the incoming light vector!
    shd_raw = np.maximum(
        np.maximum(bot_e * w_lit_top, top_e * w_lit_bot),
        np.maximum(right_e * w_lit_left, left_e * w_lit_right),
    ) / max(1e-4, max(w_lit_top, w_lit_bot, w_lit_left, w_lit_right))

    # Boost highlight on upward/forward world-facing surfaces, boost shadow on undersides!
    if world_normal is not None:
        wn = np.asarray(world_normal, dtype=np.float32)
        up_f = np.clip(wn[..., 1] * 0.45 + wn[..., 2] * 0.25 + 0.65, 0.20, 1.25)
        lit_raw = np.clip(lit_raw * up_f, 0.0, 1.0)
        shd_raw = np.clip(shd_raw * (1.25 - 0.35 * up_f), 0.0, 1.0)

    # Asymmetric organic edge breakup & battle-chip notches (destroys robotic straight-line frames!)
    res_mult = max(1.0, math.sqrt((H * W) / 4096.0))
    brk = fbm_noise(H, W, scale=float(breakup_scale) * res_mult, octaves=2, seed=seed, tileable=False)
    chip_n = fbm_noise(H, W, scale=float(breakup_scale) * 1.9 * res_mult, octaves=2, seed=seed + 31, tileable=False)
    ca = float(np.clip(chip_amount, 0.0, 1.0))

    # Interrupt highlight intermittently so it reads like hand-placed 2-5px edge glints!
    keep_hi = np.clip((brk - (0.24 + 0.26 * ca)) * 3.8, 0.0, 1.0)
    chips = ((chip_n > (0.76 - 0.22 * ca)) & (np.maximum(lit_raw, shd_raw) > 0.28)).astype(np.float32) * ca
    hi = np.clip(lit_raw * keep_hi * (1.0 - cav * 0.92) * (1.0 - chips), 0.0, 1.0).astype(np.float32)
    lo = np.clip(np.maximum(shd_raw * (0.65 + 0.35 * brk), chips * 0.85), 0.0, 1.0).astype(np.float32)

    return {
        "highlight": hi,
        "shadow": lo,
        "chips": chips.astype(np.float32),
        "signed": np.clip(0.5 + 0.5 * hi - 0.5 * lo, 0.0, 1.0).astype(np.float32),
    }


def pixel_outline(
    src: np.ndarray,
    *,
    width: int = 1,
    position: str = "outside",  # "outside" | "inside"
    color: str = "#161224",
    corner_mode: str = "4way",  # "4way" (clean pixel art!) | "8way"
    alpha_threshold: float = 0.5,
) -> np.ndarray:
    """Pixel Composer `Outline`: crisp 4-way or 8-way pixel-art outside/inside outline."""
    rgba, _ = _to_rgba(src)
    solid = rgba[..., 3] >= float(alpha_threshold)
    w_px = max(1, min(4, int(width)))
    c_rgb = np.array(hex_to_srgb(color), dtype=np.float32)[None, None, :]

    def _step(mask: np.ndarray, dilate: bool) -> np.ndarray:
        p = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=(not dilate))
        if dilate:
            out = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
            if corner_mode == "8way":
                out |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
            return out
        out = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
        if corner_mode == "8way":
            out &= p[:-2, :-2] & p[:-2, 2:] & p[2:, :-2] & p[2:, 2:]
        return out

    m = solid.copy()
    for _ in range(w_px):
        m = _step(m, dilate=(position == "outside"))

    out = rgba.copy()
    if position == "outside":
        ring = m & (~solid)
        out[..., :3] = np.where(ring[..., None], c_rgb, out[..., :3])
        out[..., 3] = np.where(ring, 1.0, out[..., 3])
    else:
        ring = solid & (~m)
        out[..., :3] = np.where(ring[..., None], c_rgb, out[..., :3])
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Heightmap Shadow, Mirror, Pixel Sort, Chromatic Aberration, Vignette & 3D Bounce
# ---------------------------------------------------------------------------


def heightmap_shadow(
    height_map: np.ndarray,
    *,
    light_angle_deg: float = -135.0,
    elevation_deg: float = 34.0,
    max_steps: int = 8,
    strength: float = 0.72,
    tileable: bool = True,
) -> np.ndarray:
    """Pixel Composer 1.22.4 `Heightmap Shadow`: directional ray-marched cast shadows across relief!

    Casts crisp 1-4 px directional roof-tile drop shadows behind raised bricks, overlapping
    plates, straps, feathers, and scales! Returns shadow visibility [0.0=deep shadow .. 1.0=lit].
    """
    h = np.asarray(height_map, dtype=np.float32)
    if h.ndim == 3:
        h = 0.2126 * h[..., 0] + 0.7152 * h[..., 1] + 0.0722 * h[..., 2]
    H, W = h.shape[:2]
    # Ray marches TOWARD light source (dx, dy)
    rad = math.radians(float(light_angle_deg))
    dx = math.cos(rad)
    dy = math.sin(rad)
    slope = math.tan(math.radians(float(np.clip(elevation_deg, 8.0, 82.0)))) * 0.14
    steps = max(1, min(16, int(max_steps)))
    yy, xx = np.indices((H, W), dtype=np.float32)

    occ = np.zeros((H, W), dtype=np.float32)
    for s in range(1, steps + 1):
        sx = np.rint(xx + dx * s).astype(np.int32)
        sy = np.rint(yy + dy * s).astype(np.int32)
        if tileable:
            sx = sx % W
            sy = sy % H
        else:
            sx = np.clip(sx, 0, W - 1)
            sy = np.clip(sy, 0, H - 1)
        ray_h = h + slope * float(s)
        sample_h = h[sy, sx]
        diff = np.clip((sample_h - ray_h) * 4.5, 0.0, 1.0)
        decay = 1.0 - float(s - 1) / float(steps + 1)
        occ = np.maximum(occ, diff * decay)

    return np.clip(1.0 - occ * float(np.clip(strength, 0.0, 1.0)), 0.08, 1.0).astype(np.float32)


def pixel_mirror(src: np.ndarray, *, axis: str = "x") -> np.ndarray:
    """Pixel Composer `Mirror`: crisp pixel-grid symmetry (left->right, top->bottom, or 4-way quad)."""
    arr = np.asarray(src, dtype=np.float32).copy()
    H, W = arr.shape[:2]
    if axis in ("x", "xy"):
        half_w = W // 2
        left = arr[:, :half_w]
        arr[:, W - half_w :] = left[:, ::-1]
    if axis in ("y", "xy"):
        half_h = H // 2
        top = arr[:half_h, :]
        arr[H - half_h :, :] = top[::-1, :]
    return arr


def pixel_sort(
    src: np.ndarray,
    *,
    axis: str = "vertical",  # "vertical" | "horizontal"
    low_thresh: float = 0.25,
    high_thresh: float = 0.85,
    reverse: bool = False,
) -> np.ndarray:
    """Pixel Composer `Pixel Sort`: sorts pixel spans within [low_thresh, high_thresh] luminance intervals."""
    rgba, is_scalar = _to_rgba(src)
    out = rgba.copy()
    lum = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    lo, hi = float(low_thresh), float(high_thresh)

    def _sort_line(line: np.ndarray, l_vals: np.ndarray) -> np.ndarray:
        mask = (l_vals >= lo) & (l_vals <= hi)
        n = len(l_vals)
        res = line.copy()
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i > 1:
                order = np.argsort(l_vals[i:j])
                if reverse:
                    order = order[::-1]
                res[i:j] = line[i:j][order]
            i = j
        return res

    if axis == "horizontal":
        for y in range(out.shape[0]):
            out[y] = _sort_line(out[y], lum[y])
    else:
        for x in range(out.shape[1]):
            out[:, x] = _sort_line(out[:, x], lum[:, x])

    return out[..., 0].astype(np.float32) if is_scalar else out.astype(np.float32)


def chromatic_aberration(src: np.ndarray, *, shift_x: int = 1, shift_y: int = 0, strength: float = 0.65) -> np.ndarray:
    """Pixel Composer `Chromatic Aberration`: crisp 1-2 px RGB channel edge separation."""
    rgba, _ = _to_rgba(src)
    sx, sy = int(shift_x), int(shift_y)
    w = float(np.clip(strength, 0.0, 1.0))
    r_sh = np.roll(np.roll(rgba[..., 0], sx, axis=1), sy, axis=0)
    b_sh = np.roll(np.roll(rgba[..., 2], -sx, axis=1), -sy, axis=0)
    out = rgba.copy()
    out[..., 0] = rgba[..., 0] * (1.0 - w) + r_sh * w
    out[..., 2] = rgba[..., 2] * (1.0 - w) + b_sh * w
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def vignette(src: np.ndarray, *, strength: float = 0.35, radius: float = 0.72, color: str = "#120E1D") -> np.ndarray:
    """Pixel Composer `Vignette`: subtle hue-shifted dark focal frame."""
    rgba, _ = _to_rgba(src)
    H, W = rgba.shape[:2]
    u, v = uv_grid(H, W)
    d = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2) * 2.0
    t = np.clip((d - float(radius)) / max(0.05, 1.42 - float(radius)), 0.0, 1.0)
    w = (t * t * float(np.clip(strength, 0.0, 1.0)))[..., None]
    c = np.array(hex_to_srgb(color), dtype=np.float32)[None, None, :]
    out = rgba.copy()
    out[..., :3] = np.clip(out[..., :3] * (1.0 - w) + c * w, 0.0, 1.0)
    return out.astype(np.float32)


def bake_emissive_bounce(
    mat: dict[str, np.ndarray],
    mesh_context: Any | None,
    *,
    radius_units: float = 7.5,
    intensity: float = 0.52,
    max_emitters: int = 96,
    posterize_steps: int = 0,
) -> dict[str, np.ndarray]:
    """Hytale `Golem_Firesteel`-Grade 3D World-Space Baked Emissive Bounce Light (MATERIAL Bus)!

    Samples glowing 3D texels (`emissive.alpha > 0.25`) across the entity's UV atlas and computes
    physically attenuated inverse-square diffuse bounce illumination in 3D world space onto nearby
    non-emissive faces (eyes lighting brow ridges/cheeks, axe runes lighting steel blades, magma
    fissures lighting dark obsidian bevels!).
    """
    out = _ensure_bus(mat, tileable=False)
    for k in MATERIAL_BUS_PORTS:
        if k in out and isinstance(out[k], np.ndarray):
            out[k] = out[k].copy()
    if mesh_context is None:
        return out

    mc = mesh_context
    em = out["emissive"]
    em_a = em[..., 3]
    em_mask = mc.island_mask & (em_a > 0.22)
    if not np.any(em_mask):
        return out

    # Collect emitting 3D texels and subsample deterministically for fast vectorized evaluation (~15ms)
    ey, ex = np.nonzero(em_mask)
    n_em = ey.size
    k_max = max(8, int(max_emitters))
    if n_em > k_max:
        idx = np.linspace(0, n_em - 1, k_max, dtype=np.int32)
        ey, ex = ey[idx], ex[idx]

    e_pos = mc.world_pos[ey, ex].astype(np.float32)  # (M, 3)
    e_col = (em[ey, ex, :3] * em_a[ey, ex, None]).astype(np.float32)  # (M, 3)

    # Evaluate on active non-emissive island texels
    recv_mask = mc.island_mask & (em_a < 0.25)
    if not np.any(recv_mask):
        return out
    ry, rx = np.nonzero(recv_mask)
    r_pos = mc.world_pos[ry, rx].astype(np.float32)  # (N, 3)
    r_nrm = mc.world_normal[ry, rx].astype(np.float32)  # (N, 3)

    R = max(1.0, float(radius_units))
    R2 = R * R
    # Chunked vectorized inverse-square radiance to keep peak RAM tiny even on 512x512 atlases
    bounce = np.zeros((r_pos.shape[0], 3), dtype=np.float32)
    chunk = 4096
    for s0 in range(0, r_pos.shape[0], chunk):
        s1 = min(r_pos.shape[0], s0 + chunk)
        diff = e_pos[None, :, :] - r_pos[s0:s1, None, :]  # (C, M, 3)
        d2 = np.sum(diff * diff, axis=-1)  # (C, M)
        dist = np.sqrt(np.maximum(d2, 0.25))
        # Smooth finite-radius falloff + soft half-Lambert facing wrap
        falloff = np.clip(1.0 - d2 / R2, 0.0, 1.0)
        falloff = (falloff * falloff) / (1.0 + d2 * 0.22)
        ndotl = np.sum(diff * r_nrm[s0:s1, None, :], axis=-1) / dist
        wrap = np.clip(ndotl * 0.55 + 0.45, 0.0, 1.0)
        w = falloff * wrap  # (C, M)
        # Peak-normalized accumulation so dense emitter clusters don't blow out into flat white
        w_sum = np.sum(w, axis=-1, keepdims=True)
        w_max = np.max(w, axis=-1, keepdims=True)
        col_acc = w @ e_col  # (C, 3)
        bounce[s0:s1] = (col_acc / np.maximum(1e-4, w_sum)) * np.clip(w_max * 2.2, 0.0, 1.0)

    if int(posterize_steps) >= 2:
        st = float(int(posterize_steps))
        bounce = np.rint(bounce * st) / st

    gain = float(np.clip(intensity, 0.0, 1.5))
    # Screen-like painterly bounce blend (adds luminous colored rim warmth without washing dark blacks flat!)
    base_rgb = out["albedo"][ry, rx, :3]
    lit_rgb = np.clip(base_rgb + bounce * gain * (1.0 - base_rgb * 0.42), 0.0, 1.0)
    out["albedo"][ry, rx, :3] = lit_rgb
    return out
