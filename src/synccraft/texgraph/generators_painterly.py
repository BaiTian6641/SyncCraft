"""Master Studio Painterly & Stylized Blockbench Texture Generators.

Distilled directly from expert Blockbench studio art & YouTube master workflows
(ArtsByKev, Everbloom, Brandon James Greer, Vijay Dragon Master) and studio creature/world references:
- ``smooth_wing_membrane``      : Backlit translucent smooth-gradient dragon wing membrane with bone-proximity AO,
                                  graceful tension folds, and hand-pixelled concave/tattered trailing edges!
- ``gradient_eye_paint``        : Non-standard luminous multi-stop gradient eye (fierce glowing dragon eyes with
                                  optional subtle ember/slit modes, 100% mirror-symmetric L/R!).
- ``directional_brushstrokes``  : Soft overlapping hand-painted directional studio brushwork.
- ``layered_scale_plates``      : Bold, calm chunky Blockbench armor/scale plates with 1px lit rim + dark under-seam.
- ``temple_carved_stone``       : Clean sunlit stepped ruin sandstone masonry + carved relief glyphs.
- ``stylized_wood_planks``      : Warm studio timber planks with smooth grain & clean joinery seams.
- ``stylized_foliage_clusters`` : Crisp 3-tone painterly leaf canopy clusters (zero static noise!).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .generators_gradients import multi_stop_gradient
from .generators_noise import fbm_noise, uv_grid, value_noise
from .pixel_art_ops import hex_to_srgb


def directional_brushstrokes(
    height: int,
    width: int,
    *,
    angle_deg: float = 0.0,
    scale_along: float = 3.5,
    scale_across: float = 16.0,
    contrast: float = 0.35,
    seed: int = 42,
    tileable: bool = False,
) -> np.ndarray:
    """Long, smooth, overlapping painterly directional brushstrokes (replaces noisy static with hand-painted flow)."""
    u, v = uv_grid(height, width)
    rad = math.radians(float(angle_deg))
    ca, sa = math.cos(rad), math.sin(rad)
    ur = (u - 0.5) * ca - (v - 0.5) * sa + 0.5
    vr = (u - 0.5) * sa + (v - 0.5) * ca + 0.5

    # Two overlapping elongated brushstroke layers (macro wash + crisp brush bristle)
    n1 = fbm_noise(height, width, scale=max(1.5, float(scale_along)), scale_y=max(2.0, float(scale_across)), octaves=3, persistence=0.48, seed=seed, tileable=tileable)
    n2 = value_noise(height, width, scale=max(2.0, float(scale_along) * 1.8), scale_y=max(3.0, float(scale_across) * 1.6), seed=seed + 97, tileable=tileable)

    # Rotate sampling slightly if angle_deg != 0
    if abs(float(angle_deg)) > 1.0:
        yy = np.clip(np.rint(vr * (height - 1)).astype(np.int64), 0, height - 1)
        xx = np.clip(np.rint(ur * (width - 1)).astype(np.int64), 0, width - 1)
        if tileable:
            yy = np.rint(vr * height).astype(np.int64) % height
            xx = np.rint(ur * width).astype(np.int64) % width
        s1 = n1[yy, xx]
        s2 = n2[yy, xx]
    else:
        s1, s2 = n1, n2

    stroke = 0.5 + ((0.65 * s1 + 0.35 * s2) - 0.5) * (1.2 + float(contrast) * 1.4)
    return np.clip(stroke, 0.0, 1.0).astype(np.float32)


def smooth_wing_membrane(
    local_u: np.ndarray,
    local_v: np.ndarray,
    *,
    palette_stops: Sequence[tuple[float, str]] = (
        (0.00, "#190E33"),
        (0.28, "#3B1D6E"),
        (0.58, "#7C3AED"),
        (0.84, "#C084FC"),
        (1.00, "#A5F3FC"),
    ),
    rim_color: str = "#67E8F9",
    arch_depth: float = 0.24,
    arches: int = 1,
    tatter_strength: float = 0.06,
    fold_count: float = 5.0,
    fold_strength: float = 0.14,
    glow_center_strength: float = 0.85,
    bone_shadow_width: float = 0.24,
    rim_width: float = 0.055,
    smooth_blend: float = 0.72,
    quantize_steps: int = 8,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Master Studio Dragon Wing Membrane Generator (silky backlit translucency + bone AO + organic tattered arch).

    Matches high-craft Blockbench studio wings:
      - Rich dark hue-shifted bone contact shadow along leading spar (v->1) & lateral finger struts (u->0,1)
      - Silky smooth luminous translucent glow pool in the center of each wing bay (80% calm!)
      - Graceful low-contrast radial tension folds & painterly strokes sweeping from wrist to trailing edge
      - Organic concave bat/dragon arch with hand-pixelled subtle edge notches (zero busy checkerboard noise!)

    Returns dict: ``{"rgba", "alpha", "rim_mask", "glow_mask", "height"}``.
    """
    u = np.clip(np.asarray(local_u, dtype=np.float32), 0.0, 1.0)
    v = np.clip(np.asarray(local_v, dtype=np.float32), 0.0, 1.0)
    H, W = u.shape[:2]

    # 1. Single sweeping concave bat-wing arch (or N arches) + subtle organic pixel-notches at trailing edge
    k = max(1, int(arches))
    au = (u * k) % 1.0
    arch = float(arch_depth) * np.power(1.0 - (2.0 * au - 1.0) ** 2, 1.15)
    # Deterministic organic small edge notches (like classic tattered dragon wing tips)
    notch_n = value_noise(H, W, scale=max(8.0, W / 18.0), scale_y=2.0, seed=seed + 19, tileable=False)
    # Only notch the very edge, keeping arch silhouette unmistakably graceful
    notches = np.where(notch_n > 0.72, (notch_n - 0.72) * 3.5 * float(tatter_strength), 0.0)
    cut_v = np.clip(arch + notches, 0.0, 0.58)

    dist_from_trail = v - cut_v
    alpha = (dist_from_trail >= 0.0).astype(np.float32)
    rim_mask = ((dist_from_trail >= 0.0) & (dist_from_trail <= float(rim_width))).astype(np.float32)

    # 2. Bone-proximity contact shadow gradient (dark along leading spar v=1 & lateral spars u=0,1)
    bw = max(0.05, float(bone_shadow_width))
    bone_lead = np.clip((1.0 - v) / bw, 0.0, 1.0)
    bone_side = np.clip(np.minimum(u, 1.0 - u) / (bw * 0.85), 0.0, 1.0)
    trail_fade = np.clip(dist_from_trail / (bw * 0.75), 0.0, 1.0)
    bone_open = np.power(bone_lead * bone_side, 0.65) * (0.45 + 0.55 * trail_fade)

    # 3. Luminous backlit translucent core pool (elliptical center slightly toward inner bay)
    du = (u - 0.50) * 1.75
    dv = (v - 0.54) * 1.95
    core_pool = np.clip(1.0 - np.sqrt(du * du + dv * dv), 0.0, 1.0)
    core_pool = core_pool * core_pool * (3.0 - 2.0 * core_pool) * float(glow_center_strength)

    # 4. Graceful radial tension folds + silky painterly brushwork sweeping toward trailing edge
    ang = np.arctan2((1.05 - v), (u - 0.5) * 1.4)
    folds = 0.5 + 0.5 * np.sin(ang * float(fold_count) * math.pi + 0.7 * np.sin(u * 6.0))
    brush = directional_brushstrokes(H, W, angle_deg=14.0, scale_along=3.0, scale_across=14.0, contrast=0.22, seed=seed + 51)
    fold_delta = (folds - 0.5) * float(fold_strength) + (brush - 0.5) * 0.12

    # Combine into smooth continuous tone field [0, 1] (calm, luminous, zero harsh static!)
    tone = np.clip(0.14 + 0.44 * bone_open + 0.42 * core_pool + fold_delta, 0.0, 1.0).astype(np.float32)
    height_f = np.clip(0.35 + 0.45 * (1.0 - bone_open) + 0.22 * (folds - 0.5), 0.0, 1.0).astype(np.float32)

    # Smooth-to-pixel hybrid OKLab multi-stop colorization!
    rgba = multi_stop_gradient(
        tone,
        palette_stops,
        space="oklab",
        smooth_blend=smooth_blend,
        quantize_steps=quantize_steps,
        dither="clustered",
        dither_strength=0.06,
        alpha_mask=alpha,
    )

    # Subtle trailing edge rim tint (blended smoothly, never garish solid sticker)
    rc = np.array(hex_to_srgb(rim_color), dtype=np.float32)
    rim_w = (rim_mask * 0.58)[..., None]
    rgba[..., :3] = np.clip(rgba[..., :3] * (1.0 - rim_w) + rc[None, None, :] * rim_w, 0.0, 1.0)

    # Soft emissive mask strongest in translucent core & delicate trailing rim
    glow_mask = np.clip((core_pool * 0.52 + rim_mask * 0.65) * alpha, 0.0, 1.0).astype(np.float32)
    return {
        "rgba": rgba.astype(np.float32),
        "alpha": alpha,
        "rim_mask": rim_mask,
        "glow_mask": glow_mask,
        "height": height_f,
    }


def gradient_eye_paint(
    local_u: np.ndarray,
    local_v: np.ndarray,
    *,
    side_sign: np.ndarray | None = None,
    stops: Sequence[tuple[float, str]] = (
        (0.00, "#0E4C92"),
        (0.28, "#06B6D4"),
        (0.58, "#67E8F9"),
        (0.82, "#FEF08A"),
        (1.00, "#FFFFFF"),
    ),
    brow_shadow: float = 0.42,
    angle_deg: float = -22.0,
    style: str = "gradient_glow",
    rim_color: str = "#0B1026",
) -> tuple[np.ndarray, np.ndarray]:
    """Non-standard Studio Luminous Gradient Eye (zero cartoon pupil — fierce mythical dragon gaze!).

    Matches the sleek radiant glowing gradient eyes seen in pro Blockbench dragons (Images 1–4)
    and Aethelion v2's fan-favorite non-standard gradient eye — with 100% L/R mirror symmetry!
    ``side_sign`` automatically flips horizontal U on right-side (+X) eyes so the radiant bright
    core ALWAYS sweeps identically forward toward the snout on BOTH left and right eyes!

    Returns ``(eye_rgba, eye_emissive_mask)``.
    """
    u = np.clip(np.asarray(local_u, dtype=np.float32), 0.0, 1.0)
    v = np.clip(np.asarray(local_v, dtype=np.float32), 0.0, 1.0)
    if side_sign is not None:
        u_fwd = np.where(np.asarray(side_sign, dtype=np.float32) > 0.0, 1.0 - u, u)
    else:
        u_fwd = u

    rad = math.radians(float(angle_deg))
    # Smooth diagonal gradient sweeping from deep outer-posterior corner to radiant anterior-lower starlight core!
    diag = np.clip(0.50 + (u_fwd - 0.50) * math.cos(rad) + (v - 0.46) * math.sin(rad), 0.0, 1.0)

    if style == "ember_slit":
        # Fierce glowing hot central ember band fading into dark crimson corners
        core = np.clip(1.0 - np.abs(v - 0.52) * 2.4 - np.abs(u_fwd - 0.58) * 1.15, 0.0, 1.0)
        t = np.clip(0.25 * diag + 0.85 * core, 0.0, 1.0)
    else:  # "gradient_glow" (non-standard sleek luminous gradient eye!)
        # Combine sweeping diagonal gradient + soft luminous anterior hotspot + subtle sleek bottom glow
        hot = np.clip(1.0 - np.sqrt(((u_fwd - 0.68) * 1.55) ** 2 + ((v - 0.54) * 1.75) ** 2), 0.0, 1.0)
        t = np.clip(0.58 * diag + 0.52 * (hot * hot * (3.0 - 2.0 * hot)), 0.0, 1.0)

    rgba = multi_stop_gradient(t, stops, space="oklab", smooth_blend=0.88, quantize_steps=8)

    # Subtle top brow hood shadow (gives fierce scowl depth without breaking the smooth gradient!)
    top_sh = np.clip((0.20 - v) / 0.20, 0.0, 1.0) * float(brow_shadow)
    # Hairline 1-band dark rear/top socket edge so eye sits recessed under brow
    rim_c = np.array(hex_to_srgb(rim_color), dtype=np.float32)
    edge_fade = np.clip(np.maximum((0.08 - v) / 0.08, (0.07 - u_fwd) / 0.07), 0.0, 1.0) * 0.65
    rgba[..., :3] = rgba[..., :3] * (1.0 - top_sh[..., None]) * (1.0 - edge_fade[..., None]) + rim_c[None, None, :] * edge_fade[..., None]

    # Full luminous emissive across the glowing eye gradient (tapered subtly at brow shadow)
    em_mask = np.clip((0.55 + 0.45 * t) * (1.0 - top_sh * 0.85) * (1.0 - edge_fade), 0.0, 1.0).astype(np.float32)
    return rgba.astype(np.float32), em_mask


def layered_scale_plates(
    height: int,
    width: int,
    *,
    cols: int = 8,
    rows: int = 12,
    stagger: float = 0.5,
    gap: float = 0.08,
    top_rim: float = 0.22,
    variance: float = 0.20,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Chunky, calm, deliberate Blockbench overlapping scale/armor plates (like Images 2 & 3).

    Produces clean rectangular/beveled staggered armor plates with a crisp sunlit top edge,
    calm smooth plate body, and dark under-plate drop shadow (ZERO noisy pixel confetti!).
    """
    u, v = uv_grid(height, width)
    c = max(2, int(cols))
    r = max(2, int(rows))
    gv = v * r
    row_i = np.floor(gv).astype(np.int64)
    fv = gv - row_i

    gu = u * c + (row_i % 2) * float(stagger)
    col_i = np.floor(gu).astype(np.int64)
    fu = gu - col_i

    # Deterministic per-plate subtle tonal variation (makes each scale feel hand-painted!)
    h_hash = ((col_i * 73856093) ^ (row_i * 19349663) ^ (int(seed) * 83492791)) & 0xFFFF
    plate_id = (h_hash.astype(np.float32) / 65535.0)

    g = float(gap)
    in_u = np.clip(np.minimum(fu, 1.0 - fu) / max(1e-4, g), 0.0, 1.0)
    in_v = np.clip(np.minimum(fv, 1.0 - fv) / max(1e-4, g * 1.2), 0.0, 1.0)
    plate_mask = np.minimum(in_u, in_v)

    # Smooth top-to-bottom dome gradient across each plate + crisp sunlit top highlight + dark cast shadow at bottom
    top_hi = np.clip(1.0 - fv / max(0.08, float(top_rim)), 0.0, 1.0) * 0.26
    bot_sh = np.clip((fv - (1.0 - g * 2.2)) / max(0.08, g * 2.2), 0.0, 1.0) * 0.34
    body_grad = 0.56 + (0.5 - fv) * 0.16 + (plate_id - 0.5) * float(variance)

    h_val = np.clip((body_grad + top_hi - bot_sh) * (0.28 + 0.72 * plate_mask), 0.0, 1.0).astype(np.float32)
    seam = np.clip(1.0 - plate_mask, 0.0, 1.0).astype(np.float32)
    return {"height": h_val, "seam": seam, "plate_id": plate_id.astype(np.float32)}


def temple_carved_stone(
    height: int,
    width: int,
    *,
    cols: int = 2,
    rows: int = 4,
    carve_motif: bool = True,
    seed: int = 108,
) -> dict[str, np.ndarray]:
    """Clean sunlit Mayan/Ruin carved ashlar stone masonry (inspired directly by Reference Image 6!)."""
    u, v = uv_grid(height, width)
    base = layered_scale_plates(height, width, cols=cols, rows=rows, stagger=0.5, gap=0.07, top_rim=0.18, variance=0.16, seed=seed)
    h = base["height"].copy()

    # Optional crisp recessed rectangular Mayan spiral / step-fret glyph border on alternate blocks
    motif_m = np.zeros((height, width), dtype=np.float32)
    if carve_motif:
        dx = np.abs((u * cols) % 1.0 - 0.5)
        dy = np.abs((v * rows) % 1.0 - 0.5)
        outer_ring = (np.maximum(dx, dy) > 0.24) & (np.maximum(dx, dy) < 0.34)
        inner_sq = (dx < 0.12) & (dy < 0.12)
        motif_m = (outer_ring | inner_sq).astype(np.float32) * (1.0 - base["seam"])
        h = np.clip(h - motif_m * 0.24, 0.0, 1.0)

    # Subtle smooth limestone grain
    grain = fbm_noise(height, width, scale=5.0, octaves=2, seed=seed + 13, tileable=True)
    h = np.clip(h + (grain - 0.5) * 0.10, 0.0, 1.0).astype(np.float32)
    return {"height": h, "seam": base["seam"], "motif": motif_m}


def stylized_wood_planks(
    height: int,
    width: int,
    *,
    planks: int = 4,
    seed: int = 202,
) -> dict[str, np.ndarray]:
    """Warm, clean studio Blockbench wood planks with silky flowing grain (inspired by Reference Image 7!)."""
    u, v = uv_grid(height, width)
    p = max(2, int(planks))
    gv = v * p
    row_i = np.floor(gv).astype(np.int64)
    fv = gv - row_i

    # Plank seams + subtle end-joint vertical notch
    seam_h = (fv < 0.08) | (fv > 0.94)
    joint_u = ((row_i * 0.37 + 0.25) % 1.0)
    seam_v = np.abs(((u - joint_u + 0.5) % 1.0) - 0.5) < 0.025
    seam = (seam_h | seam_v).astype(np.float32)

    # Smooth elongated wood grain lines + per-plank warm tone shift
    grain = directional_brushstrokes(height, width, angle_deg=0.0, scale_along=2.5, scale_across=18.0, contrast=0.30, seed=seed, tileable=True)
    plank_var = (((row_i * 2654435761) & 0xFFFF).astype(np.float32) / 65535.0 - 0.5) * 0.14
    top_hi = np.clip(1.0 - fv / 0.20, 0.0, 1.0) * 0.18
    h = np.clip((0.52 + plank_var + (grain - 0.5) * 0.36 + top_hi) * (1.0 - 0.72 * seam), 0.0, 1.0).astype(np.float32)
    return {"height": h, "seam": seam}


def stylized_foliage_clusters(
    height: int,
    width: int,
    *,
    cols: int = 5,
    rows: int = 5,
    seed: int = 303,
) -> dict[str, np.ndarray]:
    """Crisp, clean 3-tone studio leaf canopy clusters (inspired directly by Reference Image 7 trees!)."""
    u, v = uv_grid(height, width)
    # Rounded overlapping leaf-clump lobes with clean top sunlit gradient
    n1 = fbm_noise(height, width, scale=float(cols), scale_y=float(rows), octaves=2, mode="billow", seed=seed, tileable=True)
    n2 = value_noise(height, width, scale=float(cols) * 2.0, scale_y=float(rows) * 2.0, seed=seed + 41, tileable=True)
    clump = np.clip(0.65 * n1 + 0.35 * n2, 0.0, 1.0)
    # Clean top-down canopy sunlight gradient
    sun_grad = np.clip(1.0 - v * 0.35 + (clump - 0.5) * 0.65, 0.0, 1.0)
    h = np.clip(0.55 * clump + 0.45 * sun_grad, 0.0, 1.0).astype(np.float32)
    return {"height": h, "clump": clump.astype(np.float32)}
