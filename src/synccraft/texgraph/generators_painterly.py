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


def sculpted_pinion_wing(
    span_u: np.ndarray,
    trail_v: np.ndarray,
    *,
    face_is_down: np.ndarray | None = None,
    wave: np.ndarray | None = None,
    vanes: int = 5,
    sweep: float = 0.84,
    envelope_style: str = "primary",
    blade_stops: Sequence[tuple[float, str]] = (
        (0.00, "#0A091E"),
        (0.24, "#171A45"),
        (0.52, "#283D7E"),
        (0.78, "#4D7EC7"),
        (0.93, "#99CEFA"),
        (1.00, "#E3F6FF"),
    ),
    gold_stops: Sequence[tuple[float, str]] = (
        (0.00, "#753E14"),
        (0.45, "#D99527"),
        (0.82, "#F9DA70"),
        (1.00, "#FFF9D9"),
    ),
    neon_stops: Sequence[tuple[float, str]] = (
        (0.00, "#0284C7"),
        (0.45, "#22D3EE"),
        (0.82, "#A5F3FC"),
        (1.00, "#FFFFFF"),
    ),
    gold_arc: bool = True,
    quill_glow_strength: float = 0.92,
    smooth_blend: float = 0.70,
    quantize_steps: int = 8,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Master Studio Sculpted Aerodynamic Pinion-Blade Wing Generator (Full Multi-Port `MATERIAL` Bus).

    Carves crisp 100% hard-alpha overlapping pointed pinion-feather blades along ``span_u`` (0=root -> 1=tip)
    and ``trail_v`` (0=leading spar -> 1=trailing tips, 100% aligned top & bottom!). Combines:
    - Silky smooth-to-pixel OKLab metallic obsidian-sapphire pinion facets with dark overlap shadow & razor honed tips
    - Subtle directional feather-barb micro-relief & slight ventral darkening on underside (``face_is_down``)
    - Continuous brushed antique champagne-gold astrolabe anchor arc near spar root (LabPBR Metallic 0.94!)
    - Sleek recessed glowing Twin-Spectrum energy quill filaments framed by dark obsidian bezels + soft optical bounce halo

    Returns a complete `MATERIAL` bus dict:
      ``{"albedo", "alpha", "height", "normal", "roughness", "metallic", "ao", "emissive", "sss", "porosity", "glow_mask", "gold_mask", "blade_mask"}``
    """
    from .pbr import height_to_normal_map

    u = np.clip(np.asarray(span_u, dtype=np.float32), 0.0, 1.0)
    v = np.clip(np.asarray(trail_v, dtype=np.float32), 0.0, 1.0)
    H, W = u.shape[:2]
    is_dn = (
        np.clip(np.asarray(face_is_down, dtype=np.float32), 0.0, 1.0)
        if face_is_down is not None
        else np.zeros((H, W), dtype=np.float32)
    )

    n_vanes = max(2, int(vanes))
    vane_coord = u * float(n_vanes) - v * float(sweep)
    vane_idx = np.floor(vane_coord)
    vane_frac = (vane_coord - vane_idx).astype(np.float32)

    # Graceful aerodynamic trailing silhouette envelopes (tapered near carpal wrist so adjacent bays NEVER collide!)
    if envelope_style == "distal_tip":
        # Long razor outer primary wingtip tapering sleekly to a needle lance at u=1
        env = 0.94 - 0.56 * np.power(np.clip(u - 0.08, 0.0, 1.0), 1.32) - 0.16 * np.clip(0.18 - u, 0.0, 1.0)
    elif envelope_style == "inner_curtain":
        # Sweeping inner covert/tertiary fan, fuller in middle, tucked smoothly near body & elbow
        env = 0.90 - 0.38 * np.power(np.abs(u - 0.48) * 1.65, 1.35)
    elif envelope_style == "forearm_covert":
        # Sleek secondary flight pinions trailing cleanly behind the forearm spar
        env = 0.88 - 0.42 * np.power(np.abs(u - 0.52) * 1.75, 1.40)
    else:  # "primary": wedge-tapered at carpal root (u=0), sweeping deeply toward outer span (u=0.65..0.85)
        root_taper = np.clip(0.38 + 0.62 * np.power(np.clip(u / 0.34, 0.0, 1.0), 0.72), 0.38, 1.0)
        tip_taper = 1.0 - 0.24 * np.power(np.clip((u - 0.68) / 0.32, 0.0, 1.0), 1.45)
        env = 0.96 * root_taper * tip_taper

    # Crisp pointed aerodynamic lance-tip profile per pinion feather
    blade_point = 1.0 - 0.38 * np.where(
        vane_frac < 0.64,
        (0.64 - vane_frac) / 0.64,
        (vane_frac - 0.64) / 0.36,
    )
    max_v = np.clip(env * blade_point, 0.08, 0.99)
    edge_mask = (u >= 0.015 + 0.09 * v) & (u <= 0.985 - 0.12 * np.power(v, 1.25))
    solid = (v <= max_v) & edge_mask
    alpha = solid.astype(np.float32)

    # 1. Silky sculptural obsidian pinion blade shading & height relief
    # Smooth gradient across each blade: dark overlap shadow on trailing-left seam (vane_frac -> 0),
    # calm obsidian-sapphire facet across middle, honed cerulean-silver spine ridge along leading edge (vane_frac -> 0.88)
    overlap_sh = np.clip(vane_frac / 0.22, 0.0, 1.0)
    ridge_hi = np.clip((vane_frac - 0.58) / 0.34, 0.0, 1.0)
    ridge_hi = ridge_hi * ridge_hi * (3.0 - 2.0 * ridge_hi)
    root_ao = np.clip(v / 0.20, 0.0, 1.0)
    tip_sheen = np.clip(1.0 - np.maximum(0.0, max_v - v) / 0.12, 0.0, 1.0) * np.clip((v - 0.25) / 0.25, 0.0, 1.0)
    # Subtle micro-barb striations angled gently along each blade
    barbs = 0.5 + 0.5 * np.sin((v * 26.0 - vane_frac * 9.0) * math.pi)
    brush = directional_brushstrokes(H, W, angle_deg=-24.0, scale_along=3.0, scale_across=16.0, contrast=0.18, seed=seed + 11)

    # Ventral underside naturally reads 14% deeper so top vs underside wing contrast feels physically grounded
    dn_drop = 1.0 - 0.14 * is_dn
    blade_tone = (
        (0.11 + 0.30 * overlap_sh + 0.28 * ridge_hi + 0.18 * tip_sheen)
        * (0.44 + 0.56 * root_ao)
        + (barbs - 0.5) * 0.045
        + (brush - 0.5) * 0.055
    ) * dn_drop
    # Razor honed silver-blue tip rim ONLY along trailing feather points (clean restrained highlight!)
    near_tip = solid & ((max_v - v) <= 0.036) & (v > 0.28)
    blade_tone = np.where(near_tip, np.maximum(blade_tone, 0.78 - 0.10 * is_dn), blade_tone)
    blade_tone = np.clip(blade_tone, 0.0, 1.0).astype(np.float32)

    blade_rgba = multi_stop_gradient(
        blade_tone,
        blade_stops,
        space="oklab",
        smooth_blend=smooth_blend,
        quantize_steps=quantize_steps,
        dither="clustered",
        dither_strength=0.05,
        alpha_mask=alpha,
    )

    # Sculpted bevel heightfield per blade (stepped feather overlap + proud central ridge + beveled tip)
    blade_h = np.clip(
        (0.28 + 0.42 * overlap_sh + 0.24 * ridge_hi + 0.06 * barbs) * (0.65 + 0.35 * root_ao) * alpha,
        0.0,
        1.0,
    ).astype(np.float32)

    # 2. Continuous brushed antique champagne-gold astrolabe anchor arc near spar root!
    arc_v = 0.18 + 0.055 * np.sin(u * math.pi)
    arc_d = np.abs(v - arc_v)
    gold_band = (solid & (arc_d <= 0.036)).astype(np.float32) if gold_arc else np.zeros((H, W), dtype=np.float32)
    gold_shadow = (solid & (v > arc_v + 0.036) & (v <= arc_v + 0.068)).astype(np.float32) if gold_arc else np.zeros((H, W), dtype=np.float32)
    gold_tone = np.clip((0.68 - (arc_d / 0.040) * 0.32 + 0.16 * np.sin(u * 12.0)) * dn_drop, 0.0, 1.0).astype(np.float32)
    gold_rgba = multi_stop_gradient(gold_tone, gold_stops, space="oklab", smooth_blend=0.78, quantize_steps=6, alpha_mask=alpha)

    comp_rgb = blade_rgba[..., :3].copy()
    # Cast dark bevel line directly under the gold arc
    comp_rgb *= (1.0 - gold_shadow[..., None] * 0.42)
    comp_rgb = comp_rgb * (1.0 - gold_band[..., None]) + gold_rgba[..., :3] * gold_band[..., None]

    # 3. Sleek recessed glowing Twin-Spectrum Neon Quill Filaments framed by dark obsidian bezels!
    quill_dist = np.abs(vane_frac - 0.56)
    in_quill_len = solid & (v >= 0.26) & (v <= max_v - 0.11)
    slit_bezel = (in_quill_len & (quill_dist <= 0.125)).astype(np.float32)
    slit_halo = np.clip(1.0 - np.maximum(0.0, quill_dist - 0.035) / 0.065, 0.0, 1.0) * in_quill_len.astype(np.float32)
    slit_core = np.clip(1.0 - quill_dist / 0.044, 0.0, 1.0) * (solid & (v >= 0.28) & (v <= max_v - 0.13)).astype(np.float32)
    hot_core = np.clip(1.0 - quill_dist / 0.020, 0.0, 1.0) * (solid & (v >= 0.35) & (v <= max_v - 0.20)).astype(np.float32)

    # Optional animated traveling pulse along the quill filament!
    if wave is not None:
        w_mod = np.clip(0.68 + 0.32 * np.asarray(wave, dtype=np.float32), 0.55, 1.0)
    else:
        w_mod = np.ones((H, W), dtype=np.float32)

    neon_driver = np.clip(0.32 + 0.48 * slit_core + 0.30 * hot_core - 0.14 * np.clip(v - 0.4, 0.0, 0.6), 0.0, 1.0).astype(np.float32)
    neon_rgba = multi_stop_gradient(neon_driver, neon_stops, space="oklab", smooth_blend=0.95, alpha_mask=alpha)

    # Dark recessed obsidian bezel channel around each glowing slit (makes the glow razor-clean!)
    bezel_only = np.clip(slit_bezel - slit_halo, 0.0, 1.0)
    comp_rgb *= (1.0 - bezel_only[..., None] * 0.62)
    # Soft optical bounce halo on the recessed obsidian bevel walls
    halo_w = np.clip(slit_halo * 0.55 * float(quill_glow_strength), 0.0, 0.72)[..., None]
    comp_rgb = comp_rgb * (1.0 - halo_w) + neon_rgba[..., :3] * halo_w
    # Crisp luminous slit core
    core_w = np.clip((slit_core * 0.92 + hot_core * 0.40) * w_mod * float(quill_glow_strength), 0.0, 1.0)[..., None]
    comp_rgb = comp_rgb * (1.0 - core_w) + neon_rgba[..., :3] * core_w

    albedo = np.concatenate((np.clip(comp_rgb, 0.0, 1.0), alpha[..., None]), axis=-1).astype(np.float32)
    glow_mask = np.clip((slit_core * 0.92 + hot_core * 0.35 + slit_halo * 0.18) * w_mod * float(quill_glow_strength) * alpha, 0.0, 1.0).astype(np.float32)
    emissive = np.concatenate((neon_rgba[..., :3] * glow_mask[..., None], (glow_mask > 0.04).astype(np.float32)[..., None] * alpha[..., None]), axis=-1).astype(np.float32)

    # 4. Physical PBR channels for the pinion wing!
    # Gold arc protrudes cleanly (+0.18 height), quill channel recesses cleanly (-0.22 height)!
    height_f = np.clip(blade_h + gold_band * 0.22 - slit_bezel * 0.20, 0.0, 1.0).astype(np.float32)
    normal_rgb = height_to_normal_map(height_f, strength=2.6, tileable=False)
    # Metallic: 0.94 on champagne-gold astrolabe arc, 0.26 on honed obsidian blades, 0.0 on recessed energy slits
    metallic = np.clip((0.26 * (1.0 - slit_bezel) + 0.68 * gold_band) * alpha, 0.0, 1.0).astype(np.float32)
    # Roughness: 0.17 polished gold, 0.26 glossy obsidian facets, 0.12 energy lens glass
    roughness = np.clip(0.28 * (1.0 - gold_band) * (1.0 - slit_core) + 0.17 * gold_band + 0.12 * slit_core, 0.08, 1.0).astype(np.float32)
    ao = np.clip((0.55 + 0.45 * overlap_sh) * (0.62 + 0.38 * root_ao) * (1.0 - gold_shadow * 0.35), 0.18, 1.0).astype(np.float32)
    sss = np.clip((0.18 * tip_sheen + 0.45 * slit_halo) * alpha, 0.0, 1.0).astype(np.float32)
    porosity = np.zeros((H, W), dtype=np.float32)

    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "alpha": alpha,
        "height": height_f,
        "normal": normal_rgb,
        "roughness": roughness,
        "metallic": metallic,
        "ao": ao,
        "emissive": emissive,
        "sss": sss,
        "porosity": porosity,
        "glow_mask": glow_mask,
        "gold_mask": gold_band,
        "blade_mask": alpha,
    }
