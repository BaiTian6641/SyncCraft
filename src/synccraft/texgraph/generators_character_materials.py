"""Part-Aware Semantic Entity Detailing & Character/Orc Multi-Material PBR Kit (`TexGraph v8`).

Distilled directly from our 16-video Blockbench master timelapse study (Voxel Spawns,
TugkanDeMan, BearSpirit08, Mickey Joe, ArtsByKev, Longridr, Captain Atlas, Everbloom Games):

1. **Dedicated Character & Gear PBR Material Generators (Full `MATERIAL` Bus)**:
   - ``orc_flesh_skin``       : Hue-shifted olive/emerald orc/goblin/monster skin
                                (warm yellow-green brow/knuckle highlights -> emerald midtones
                                -> cool teal-violet shadows, warm reddish subcutaneous blush,
                                healed pale battle scars, optional tribal war paint).
   - ``stitched_leather``     : Warm desaturated brown/umber leather with lower specular contrast
                                than steel, concentric stepped rim bevels, local-UV pixel
                                stitch dots & worn fold creases.
   - ``forged_battle_metal``  : Cool charcoal-indigo deep shadows (`#1E1824`), slate-steel armor
                                plates, high-contrast 1 px edge bevels, battle scratches,
                                corner rivets & optional cavity rust.
   - ``woven_battle_cloth``   : Muted crimson/linen/olive woven fabric with soft 2-3 px vertical
                                drape folds, hem trim, or diagonal twisted rope bindings.
   - ``keratin_horn_bone``    : Dark desaturated keratin root/gumline collar stepping into
                                striated warm ivory/ochre distal tips (tusks, horns, teeth,
                                beaks, claws, hooves — never stark `#FFFFFF`).

2. **Part-Aware Semantic Entity Landmark Compositor (`paint_entity_eyes_and_parts`)**:
   - Solves the #1 procedural entity texturing pitfall: **NEVER paints uniform fur/scales/stone
     over eyes, sockets, noses, beaks, mouths, tongues, teeth/tusks, claws/hooves/horns,
     inner ears, translucent fins, or glowing runes!**
   - Automatically detects each eye's primary exposed face (`front` / `lateral` / `both` via
     `eye_facing="auto"`), paints a 1-2 px dark orbital socket on non-facing sides, supports
     both crisp studio creature eyes and luminous non-standard gradient eyes, PLUS smooth
     **animated `eye_blink` (0..1)** and **animated `eye_glow` / `rune_glow` (0..1)** drivers!
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .generators_gradients import multi_stop_gradient
from .generators_noise import fbm_noise, noise_3d_field, uv_grid, voronoi_noise
from .generators_painterly import directional_brushstrokes, gradient_eye_paint
from .pbr import height_to_normal_map
from .pipeline import MATERIAL_BUS_PORTS, _ensure_bus, material_make
from .pixel_art_ops import hex_to_srgb, paint_creature_eye
from .pixel_composer_ops import (
    de_corner,
    de_stray,
    heightmap_shadow,
    palette_posterize,
    pixel_bevel_directional,
    pixel_clusters,
)


def _normalize_stops(stops: Sequence[Any]) -> tuple[tuple[float, str], ...]:
    """Accept either [(t, '#hex'), ...] or bare ['#hex1', '#hex2', ...] lists."""
    seq = list(stops)
    if not seq:
        return ((0.0, "#202028"), (1.0, "#E0E0E8"))
    if isinstance(seq[0], str):
        n = len(seq)
        if n == 1:
            return ((0.0, seq[0]), (1.0, seq[0]))
        return tuple((float(i) / float(n - 1), str(c)) for i, c in enumerate(seq))
    return tuple((float(p[0]), str(p[1])) for p in seq)


def _mesh_or_fallback(
    H: int,
    W: int,
    *,
    world_pos_norm: np.ndarray | None,
    world_normal: np.ndarray | None,
    baked_ao: np.ndarray | None,
    joint_cavity: np.ndarray | None,
    edge_bevel: np.ndarray | None,
    local_uv: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, v = uv_grid(H, W)
    if world_pos_norm is None:
        wpn = np.stack((u, 1.0 - v, np.full_like(u, 0.5)), axis=-1).astype(np.float32)
    else:
        wpn = np.asarray(world_pos_norm, dtype=np.float32)
    if world_normal is None:
        wn = np.stack((np.zeros_like(u), np.full_like(u, 0.72), np.full_like(u, 0.69)), axis=-1).astype(np.float32)
    else:
        wn = np.asarray(world_normal, dtype=np.float32)
    ao = np.ones((H, W), dtype=np.float32) if baked_ao is None else np.asarray(baked_ao, dtype=np.float32)
    cav = np.zeros((H, W), dtype=np.float32) if joint_cavity is None else np.asarray(joint_cavity, dtype=np.float32)
    if edge_bevel is None:
        dx = np.minimum(u, 1.0 - u)
        dy = np.minimum(v, 1.0 - v)
        bev = np.clip(1.0 - np.minimum(dx, dy) / 0.14, 0.0, 1.0).astype(np.float32)
    else:
        bev = np.asarray(edge_bevel, dtype=np.float32)
    luv = np.stack((u, v), axis=-1).astype(np.float32) if local_uv is None else np.asarray(local_uv, dtype=np.float32)
    return wpn, wn, ao, cav, bev, luv


# ---------------------------------------------------------------------------
# 1. Orc / Goblin / Troll / Monster Flesh Skin (MATERIAL Bus)
# ---------------------------------------------------------------------------


def orc_flesh_skin(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    edge_bevel: np.ndarray | None = None,
    local_uv: np.ndarray | None = None,
    stops: Sequence[Any] = (
        (0.00, "#1D2A24"),  # Everbloom Troll / BearSpirit08 deep cool sage-teal shadow
        (0.24, "#314835"),  # earthy forest shadow
        (0.50, "#4F6E43"),  # calm studio sage-olive midtone (NEVER neon candy lime!)
        (0.75, "#749257"),  # warm sunlit moss plane
        (0.90, "#9DB872"),  # warm yellow-olive forehead/brow/knuckle peak
        (1.00, "#C7D896"),  # crisp warm highlight cluster
    ),
    subsurface_color: str = "#8E4B3E",
    subsurface_strength: float = 0.22,
    scar_amount: float = 0.28,
    scar_color: str = "#C7B299",
    warpaint_color: str | None = None,
    warpaint_amount: float = 0.0,
    muscle_contrast: float = 0.36,
    roughness_base: float = 0.62,
    sss_base: float = 0.38,
    smooth_blend: float = 0.22,
    quantize_steps: int = 6,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Studio Hue-Shifted Orc / Goblin / Troll Skin MaterialBus (Everbloom / BearSpirit08 discipline).

    - Calm, earthy sage-olive hue-shifted palette (zero neon-lime airbrush!)
    - Crisp 2-4 px organized muscle-plane pixel clusters (`pixel_clusters` + `de_stray` + `de_corner`)
    - Anti-wireframe directional top-left warm catchlight & cool underside occlusion
    - Subtle warm reddish-brown subcutaneous blush, pale battle scars & optional tribal war paint!
    """
    H, W = int(height), int(width)
    wpn, wn, ao, cav, bev, luv = _mesh_or_fallback(
        H, W,
        world_pos_norm=world_pos_norm,
        world_normal=world_normal,
        baked_ao=baked_ao,
        joint_cavity=joint_cavity,
        edge_bevel=edge_bevel,
        local_uv=local_uv,
    )
    res_mult = max(1.0, math.sqrt((H * W) / 4096.0))
    pal = _normalize_stops(stops)
    pal_hex = [c for _, c in pal]

    # 3D rest-space broad muscle clusters + gentle directional skin texture (calm 65% base!)
    macro_3d = noise_3d_field(wpn, scale=4.0, octaves=3, mode="fbm", seed=seed)
    muscle_ridge = 1.0 - np.abs(noise_3d_field(wpn, scale=7.2, octaves=2, mode="fbm", seed=seed + 19) * 2.0 - 1.0)
    brush = directional_brushstrokes(
        H, W,
        angle_deg=-14.0,
        scale_along=3.2 * res_mult,
        scale_across=12.0 * res_mult,
        contrast=0.18,
        seed=seed + 37,
        tileable=False,
    )
    dir_bev = pixel_bevel_directional(
        H, W, local_uv=luv, world_normal=wn, joint_cavity=cav,
        light_angle_deg=-135.0, bevel_width=0.10, chip_amount=0.35, breakup_scale=9.0, seed=seed + 7,
    )

    # Top-front directional form planes + crisp directional edge accents (never 4-sided box frames!)
    top_light = np.clip(wn[..., 1] * 0.48 + wn[..., 2] * 0.24 + 0.54, 0.0, 1.0)
    isl_v = 1.0 - luv[..., 1]
    h_map = np.clip(
        0.48
        + 0.18 * (macro_3d - 0.5)
        + 0.16 * float(muscle_contrast) * (muscle_ridge - 0.5)
        + 0.10 * (brush - 0.5)
        + 0.12 * dir_bev["highlight"]
        - 0.14 * dir_bev["shadow"]
        - 0.24 * cav,
        0.05,
        0.98,
    ).astype(np.float32)

    tone = np.clip(
        0.16
        + 0.34 * top_light
        + 0.14 * isl_v
        + 0.24 * (h_map - 0.46)
        + 0.18 * (ao - 0.65)
        + 0.14 * dir_bev["highlight"]
        - 0.16 * dir_bev["shadow"]
        - 0.24 * cav,
        0.0,
        1.0,
    ).astype(np.float32)

    # Crisp palette-locked painterly clusters (breaks smooth airbrush into studio hand-placed pixel islands!)
    raw_alb = multi_stop_gradient(
        tone,
        pal,
        space="oklab",
        smooth_blend=float(smooth_blend),
        quantize_steps=int(quantize_steps),
        dither="none",
        dither_strength=0.0,
    )
    clustered = pixel_clusters(
        raw_alb,
        pal_hex,
        direction="isotropic",
        cluster_size=2,
        dither_strength=0.0,
        clean_strays=True,
        clean_corners=True,
    )
    # Blend 80% crisp pixel clusters + 20% smooth bridge (exact Hytale / Everbloom sweet spot!)
    rgb = (clustered[..., :3] * 0.80 + raw_alb[..., :3] * 0.20).astype(np.float32)

    # Subcutaneous warm blush along exposed convex ridges (brow, nose bridge, cheekbones, knuckles!)
    sss_n = noise_3d_field(wpn, scale=5.5, octaves=2, mode="fbm", seed=seed + 61)
    sss_m = np.clip(
        (0.55 * dir_bev["highlight"] + 0.45 * np.clip(wn[..., 2] * 0.5 + 0.5, 0.0, 1.0))
        * np.clip((sss_n - 0.40) * 2.4, 0.0, 1.0)
        * ao,
        0.0,
        1.0,
    ).astype(np.float32)
    s_w = sss_m * float(np.clip(subsurface_strength, 0.0, 0.55))
    c_sub = np.array(hex_to_srgb(subsurface_color), dtype=np.float32)[None, None, :]
    rgb = rgb * (1.0 - s_w[..., None]) + c_sub * s_w[..., None]

    # Healed diagonal battle scars (crisp 1-2 px slash snapped onto pixel grid!)
    scar_m = np.zeros((H, W), dtype=np.float32)
    sc_amt = float(np.clip(scar_amount, 0.0, 1.0))
    if sc_amt > 0.02:
        slash_coord = wpn[..., 0] * 3.8 + wpn[..., 1] * 2.9 - wpn[..., 2] * 2.2
        slash_wave = np.abs(np.sin(slash_coord * math.pi * 1.35 + macro_3d * 1.2))
        scar_patch = noise_3d_field(wpn, scale=2.6, octaves=2, mode="fbm", seed=seed + 83)
        scar_thresh = 0.042 + 0.040 * sc_amt
        scar_gate = (scar_patch > (0.68 - 0.20 * sc_amt)).astype(np.float32)
        scar_m = ((slash_wave < scar_thresh).astype(np.float32) * scar_gate * (1.0 - cav * 0.8)).astype(np.float32)
        c_scar = np.array(hex_to_srgb(scar_color), dtype=np.float32)[None, None, :]
        rgb = rgb * (1.0 - scar_m[..., None] * 0.65) + c_scar * (scar_m[..., None] * 0.65)
        h_map = np.clip(h_map - scar_m * 0.16, 0.04, 0.98).astype(np.float32)

    # Optional crisp tribal war paint (hard-edged brushed cluster bands across brow/cheeks/shoulders!)
    wp_amt = float(np.clip(warpaint_amount, 0.0, 1.0))
    if warpaint_color and wp_amt > 0.02:
        wp_band = np.sin((wpn[..., 1] * 14.0 + wpn[..., 0] * 5.5 + (macro_3d - 0.5) * 1.6) * math.pi)
        wp_zone = ((wpn[..., 1] > 0.55) & (wpn[..., 2] > 0.42)).astype(np.float32)
        wp_edge = ((wp_band > (0.76 - 0.26 * wp_amt)).astype(np.float32)) * wp_zone
        c_wp = np.array(hex_to_srgb(warpaint_color), dtype=np.float32)[None, None, :]
        rgb = rgb * (1.0 - wp_edge[..., None] * 0.78) + c_wp * (wp_edge[..., None] * 0.78)

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    rough = np.clip(float(roughness_base) + 0.10 * cav - 0.08 * dir_bev["highlight"] - 0.08 * scar_m, 0.40, 0.86).astype(np.float32)
    sss = np.clip(float(sss_base) * (0.65 + 0.55 * sss_m) * ao, 0.05, 0.85).astype(np.float32)
    out = material_make(
        albedo,
        height=h_map,
        normal=height_to_normal_map(h_map, strength=2.2, tileable=False),
        roughness=rough,
        metallic=0.0,
        ao=np.clip(ao * (1.0 - 0.25 * cav), 0.14, 1.0),
        sss=sss,
        porosity=0.22,
    )
    out["scar_mask"] = scar_m
    out["subsurface_mask"] = sss_m
    return out


# ---------------------------------------------------------------------------
# 2. Stitched Leather & Harness Straps (MATERIAL Bus — Anti-Boxy Directional Craft!)
# ---------------------------------------------------------------------------


def stitched_leather(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    edge_bevel: np.ndarray | None = None,
    local_uv: np.ndarray | None = None,
    stops: Sequence[Any] = (
        (0.00, "#241614"),  # deep cool umber crease
        (0.28, "#4A2C1E"),  # rich dark saddle shadow
        (0.58, "#72472C"),  # warm chestnut leather body
        (0.84, "#9C6D47"),  # worn tan outer catchlight
        (1.00, "#C4996E"),  # subtle burnished bevel
    ),
    stitch_color: str = "#D9C198",
    stitch_density: float = 1.0,
    wear: float = 0.32,
    roughness_base: float = 0.64,
    smooth_blend: float = 0.22,
    quantize_steps: int = 5,
    seed: int = 107,
) -> dict[str, np.ndarray]:
    """Studio Stitched Leather MaterialBus (BearSpirit08 / Voxel Spawns / Everbloom discipline).

    Anti-wireframe directional top-left tan bevel + bottom-right umber shadow, calm 2-4 px leather
    clusters, crisp local-UV stitch dashes & worn creases — ZERO robotic 4-sided box frames!
    """
    H, W = int(height), int(width)
    wpn, wn, ao, cav, bev, luv = _mesh_or_fallback(
        H, W,
        world_pos_norm=world_pos_norm,
        world_normal=world_normal,
        baked_ao=baked_ao,
        joint_cavity=joint_cavity,
        edge_bevel=edge_bevel,
        local_uv=local_uv,
    )
    res_mult = max(1.0, math.sqrt((H * W) / 4096.0))
    pal = _normalize_stops(stops)
    pal_hex = [c for _, c in pal]

    grain = fbm_noise(H, W, scale=11.0 * res_mult, octaves=2, seed=seed, tileable=False)
    creases = 1.0 - np.abs(fbm_noise(H, W, scale=5.0 * res_mult, octaves=2, seed=seed + 23, tileable=False) * 2.0 - 1.0)
    crease_dip = np.power(creases, 3.2) * float(np.clip(wear, 0.0, 1.0))

    dir_bev = pixel_bevel_directional(
        H, W, local_uv=luv, world_normal=wn, joint_cavity=cav,
        light_angle_deg=-135.0, bevel_width=0.11, chip_amount=0.36, breakup_scale=9.5, seed=seed + 11,
    )

    lu, lv = luv[..., 0], luv[..., 1]
    sd = float(np.clip(stitch_density, 0.0, 2.0))
    stitch_m = np.zeros((H, W), dtype=np.float32)
    if sd > 0.05:
        freq_s = 6.5 * sd
        on_h = (np.abs(lv - 0.16) < 0.042) | (np.abs(lv - 0.84) < 0.042)
        dash_h = (np.sin(lu * freq_s * 2.0 * math.pi) > 0.42) & (lu > 0.14) & (lu < 0.86)
        stitch_m = (on_h & dash_h).astype(np.float32) * np.clip(1.0 - cav * 1.1, 0.0, 1.0)

    top_light = np.clip(wn[..., 1] * 0.40 + wn[..., 2] * 0.22 + 0.56, 0.0, 1.0)
    h_map = np.clip(
        0.52
        + 0.10 * (grain - 0.5)
        + 0.14 * dir_bev["highlight"]
        - 0.16 * dir_bev["shadow"]
        - 0.14 * crease_dip
        + 0.15 * stitch_m
        - 0.22 * cav,
        0.06,
        0.96,
    ).astype(np.float32)

    tone = np.clip(
        0.24
        + 0.28 * top_light
        + 0.16 * dir_bev["highlight"]
        - 0.18 * dir_bev["shadow"]
        + 0.10 * (grain - 0.5)
        + 0.16 * (ao - 0.65)
        - 0.16 * crease_dip
        - 0.20 * cav,
        0.0,
        0.94,
    ).astype(np.float32)

    raw_alb = multi_stop_gradient(tone, pal, space="oklab", smooth_blend=float(smooth_blend), quantize_steps=int(quantize_steps))
    cl = pixel_clusters(raw_alb, pal_hex, direction="horizontal", cluster_size=2, clean_strays=True, clean_corners=True)
    rgb = (cl[..., :3] * 0.82 + raw_alb[..., :3] * 0.18).astype(np.float32)
    if sd > 0.05:
        c_st = np.array(hex_to_srgb(stitch_color), dtype=np.float32)[None, None, :]
        rgb = rgb * (1.0 - stitch_m[..., None] * 0.72) + c_st * (stitch_m[..., None] * 0.72)

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    rough = np.clip(float(roughness_base) - 0.08 * dir_bev["highlight"] + 0.12 * crease_dip + 0.10 * cav, 0.44, 0.88).astype(np.float32)
    out = material_make(
        albedo,
        height=h_map,
        normal=height_to_normal_map(h_map, strength=2.2, tileable=False),
        roughness=rough,
        metallic=0.0,
        ao=np.clip(ao * (1.0 - 0.24 * cav - 0.14 * dir_bev["shadow"]), 0.14, 1.0),
        porosity=0.35,
    )
    out["stitch_mask"] = stitch_m
    return out


# ---------------------------------------------------------------------------
# 3. Forged Battle Metal & Chipped Steel Armor (Wackyblocks / Hytale Standard!)
# ---------------------------------------------------------------------------


def forged_battle_metal(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    edge_bevel: np.ndarray | None = None,
    local_uv: np.ndarray | None = None,
    stops: Sequence[Any] = (
        (0.00, "#1D1B2A"),  # Wackyblocks / TugkanDeMan cool charcoal-indigo shadow (#393d4b family)
        (0.24, "#333748"),  # dark forged slate base
        (0.52, "#51586B"),  # calm studio steel body (65% plate interior!)
        (0.78, "#838FA4"),  # brushed diagonal specular band
        (0.93, "#C2D1DE"),  # razor broken 1 px top-left bevel (#c7d6d5 Wackyblocks catchlight!)
        (1.00, "#ECF3FA"),  # pinpoint corner glint
    ),
    scratches: float = 0.45,
    rivets: float = 0.50,
    rust: float = 0.18,
    rust_color: str = "#8C4222",
    roughness_base: float = 0.26,
    metallic_base: float = 0.88,
    smooth_blend: float = 0.20,
    quantize_steps: int = 6,
    seed: int = 211,
) -> dict[str, np.ndarray]:
    """Studio Forged Steel & Battle Armor MaterialBus (Wackyblocks / Hytale / TugkanDeMan standard!).

    - **ZERO uniform 4-sided box outlines!** Uses `pixel_bevel_directional` for asymmetrically
      broken 1 px top-left catchlights, cool dark bottom-right cast shadows & chipped battle notches!
    - Calm 65%+ brushed slate plate interiors (`pixel_clusters` diagonal streaks) punctuated by
      deliberate diagonal scratches, crisp rivets & cavity rust!
    """
    H, W = int(height), int(width)
    wpn, wn, ao, cav, bev, luv = _mesh_or_fallback(
        H, W,
        world_pos_norm=world_pos_norm,
        world_normal=world_normal,
        baked_ao=baked_ao,
        joint_cavity=joint_cavity,
        edge_bevel=edge_bevel,
        local_uv=local_uv,
    )
    res_mult = max(1.0, math.sqrt((H * W) / 4096.0))
    pal = _normalize_stops(stops)
    pal_hex = [c for _, c in pal]

    hammer = voronoi_noise(H, W, scale=7.5 * res_mult, jitter=0.68, output="f1", seed=seed, tileable=False)
    brush = directional_brushstrokes(
        H, W,
        angle_deg=32.0,
        scale_along=2.8 * res_mult,
        scale_across=14.0 * res_mult,
        contrast=0.22,
        seed=seed + 17,
        tileable=False,
    )

    # Anti-wireframe directional chipped metallic edge bevels (Wackyblocks / Hytale Firesteel craft!)
    dir_bev = pixel_bevel_directional(
        H, W, local_uv=luv, world_normal=wn, joint_cavity=cav,
        light_angle_deg=-135.0, bevel_width=0.10, chip_amount=0.46, breakup_scale=11.0, seed=seed + 5,
    )

    lu, lv = luv[..., 0], luv[..., 1]
    # Clean bold diagonal brushed steel highlight band (like Hytale Chest metal straps!)
    diag_band = np.exp(-(((lu * 0.70 + lv * 0.70) - 0.52) / 0.16) ** 2)

    # Crisp diagonal battle scratches (bright glint beside narrow dark gouge)
    sc_amt = float(np.clip(scratches, 0.0, 1.0))
    scratch_hi = np.zeros((H, W), dtype=np.float32)
    scratch_lo = np.zeros((H, W), dtype=np.float32)
    if sc_amt > 0.02:
        s_n = fbm_noise(H, W, scale=6.5 * res_mult, octaves=2, seed=seed + 43, tileable=False)
        diag_s = np.sin((wpn[..., 0] * 16.0 - wpn[..., 1] * 13.0 + wpn[..., 2] * 9.0 + s_n * 2.2) * math.pi)
        gate = (s_n > (0.68 - 0.18 * sc_amt)).astype(np.float32)
        scratch_hi = (diag_s > (0.95 - 0.04 * sc_amt)).astype(np.float32) * gate
        scratch_lo = (diag_s < -(0.95 - 0.04 * sc_amt)).astype(np.float32) * gate

    # Subtle recessed rivets near corners (only where not buried in cavities)
    riv_amt = float(np.clip(rivets, 0.0, 1.0))
    riv_hi = np.zeros((H, W), dtype=np.float32)
    riv_ring = np.zeros((H, W), dtype=np.float32)
    if riv_amt > 0.05:
        for cx, cy in ((0.18, 0.18), (0.82, 0.18), (0.18, 0.82), (0.82, 0.82)):
            dr = np.sqrt((lu - cx) ** 2 + (lv - cy) ** 2)
            riv_hi = np.maximum(riv_hi, (dr < 0.042).astype(np.float32))
            riv_ring = np.maximum(riv_ring, ((dr >= 0.042) & (dr < 0.072)).astype(np.float32))
        riv_hi *= riv_amt * (1.0 - cav)
        riv_ring *= riv_amt * (1.0 - cav)

    top_light = np.clip(wn[..., 1] * 0.44 + wn[..., 2] * 0.24 + 0.50, 0.0, 1.0)
    h_map = np.clip(
        0.54
        - 0.09 * hammer
        + 0.09 * (brush - 0.5)
        + 0.22 * dir_bev["highlight"]
        - 0.22 * dir_bev["shadow"]
        - 0.24 * dir_bev["chips"]
        - 0.20 * scratch_lo
        + 0.18 * riv_hi
        - 0.14 * riv_ring
        - 0.24 * cav,
        0.04,
        0.98,
    ).astype(np.float32)

    tone = np.clip(
        0.16
        + 0.30 * top_light
        + 0.22 * diag_band * ao
        + 0.34 * dir_bev["highlight"]
        - 0.26 * dir_bev["shadow"]
        - 0.30 * dir_bev["chips"]
        + 0.10 * (brush - 0.5)
        - 0.10 * hammer
        + 0.26 * scratch_hi
        - 0.24 * scratch_lo
        + 0.22 * riv_hi
        - 0.20 * riv_ring
        - 0.28 * cav,
        0.0,
        1.0,
    ).astype(np.float32)

    raw_alb = multi_stop_gradient(tone, pal, space="oklab", smooth_blend=float(smooth_blend), quantize_steps=int(quantize_steps))
    cl = pixel_clusters(raw_alb, pal_hex, direction="diag_pos", cluster_size=2, clean_strays=True, clean_corners=True)
    # Keep razor 1 px directional bevel catchlights & rivets crisp ON TOP of calm clustered steel plates!
    rgb = (cl[..., :3] * 0.84 + raw_alb[..., :3] * 0.16).astype(np.float32)
    c_hi = np.array(hex_to_srgb(pal_hex[-2]), dtype=np.float32)[None, None, :]
    c_sh = np.array(hex_to_srgb(pal_hex[0]), dtype=np.float32)[None, None, :]
    hi_px = (dir_bev["highlight"] > 0.52).astype(np.float32)[..., None]
    chip_px = (dir_bev["chips"] > 0.25).astype(np.float32)[..., None]
    rgb = rgb * (1.0 - hi_px * 0.75) + c_hi * (hi_px * 0.75)
    rgb = rgb * (1.0 - chip_px * 0.80) + c_sh * (chip_px * 0.80)

    # Optional oxidized rust tucked into cavities & chipped notches (Wackyblocks rust accent!)
    rust_m = np.zeros((H, W), dtype=np.float32)
    r_amt = float(np.clip(rust, 0.0, 1.0))
    if r_amt > 0.02:
        r_n = noise_3d_field(wpn, scale=7.0, octaves=2, mode="fbm", seed=seed + 79)
        rust_raw = (0.55 * cav + 0.45 * dir_bev["chips"] + 0.30 * (1.0 - ao)) * r_n * (1.8 + 1.4 * r_amt)
        rust_m = (rust_raw > 0.42).astype(np.float32) * r_amt
        c_rust = np.array(hex_to_srgb(rust_color), dtype=np.float32)[None, None, :]
        rgb = rgb * (1.0 - rust_m[..., None] * 0.82) + c_rust * (rust_m[..., None] * 0.82)

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    rough = np.clip(float(roughness_base) - 0.10 * dir_bev["highlight"] - 0.08 * scratch_hi + 0.52 * rust_m + 0.18 * cav, 0.10, 0.90).astype(np.float32)
    metal = np.clip(float(metallic_base) * (1.0 - 0.85 * rust_m), 0.10, 0.98).astype(np.float32)
    out = material_make(
        albedo,
        height=h_map,
        normal=height_to_normal_map(h_map, strength=2.6, tileable=False),
        roughness=rough,
        metallic=metal,
        ao=np.clip(ao * (1.0 - 0.25 * cav - 0.18 * riv_ring - 0.15 * dir_bev["shadow"]), 0.12, 1.0),
    )
    out["rust_mask"] = rust_m
    out["scratch_mask"] = np.clip(scratch_hi + scratch_lo, 0.0, 1.0).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 4. Woven Battle Cloth & Twisted Rope (MATERIAL Bus — Crisp Vertical Folds!)
# ---------------------------------------------------------------------------


def woven_battle_cloth(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    edge_bevel: np.ndarray | None = None,
    local_uv: np.ndarray | None = None,
    stops: Sequence[Any] = (
        (0.00, "#240C16"),  # deep plum-wine cloth fold shadow
        (0.30, "#511824"),  # dark battle crimson
        (0.60, "#882730"),  # muted studio crimson midtone
        (0.85, "#B84448"),  # warm sunlit fold crest
        (1.00, "#DF786E"),  # dusty warm edge catchlight
    ),
    style: str = "drape",  # "drape" | "rope"
    fold_contrast: float = 0.44,
    trim_color: str | None = "#C99A4E",
    trim_width: float = 0.12,
    roughness_base: float = 0.86,
    smooth_blend: float = 0.22,
    quantize_steps: int = 5,
    seed: int = 307,
) -> dict[str, np.ndarray]:
    """Studio Woven Cloth / Tabard / Loincloth / Rope Binding MaterialBus (Voxel Spawns craft).

    Crisp 2-3 px vertical cloth drape fold clusters (`pixel_clusters`) with gentle value steps,
    embroidered hem trim band & diagonal rope-twist mode!
    """
    H, W = int(height), int(width)
    wpn, wn, ao, cav, bev, luv = _mesh_or_fallback(
        H, W,
        world_pos_norm=world_pos_norm,
        world_normal=world_normal,
        baked_ao=baked_ao,
        joint_cavity=joint_cavity,
        edge_bevel=edge_bevel,
        local_uv=local_uv,
    )
    res_mult = max(1.0, math.sqrt((H * W) / 4096.0))
    pal = _normalize_stops(stops)
    pal_hex = [c for _, c in pal]
    lu, lv = luv[..., 0], luv[..., 1]

    if style == "rope":
        strand = 0.5 + 0.5 * np.sin((lu * 7.5 + lv * 7.5) * 2.0 * math.pi)
        fiber = fbm_noise(H, W, scale=16.0 * res_mult, octaves=2, seed=seed, tileable=False)
        fold_h = np.clip(0.30 + 0.55 * strand + 0.15 * (fiber - 0.5), 0.0, 1.0).astype(np.float32)
    else:
        n_warp = fbm_noise(H, W, scale=3.2 * res_mult, octaves=2, seed=seed + 11, tileable=False)
        fold_wave = 0.5 + 0.5 * np.sin((lu * 3.5 + (n_warp - 0.5) * 0.40) * 2.0 * math.pi)
        fold_h = np.clip(0.50 + float(fold_contrast) * (fold_wave - 0.5), 0.08, 0.94).astype(np.float32)

    top_light = np.clip(wn[..., 1] * 0.36 + wn[..., 2] * 0.24 + 0.56, 0.0, 1.0)
    hem_wear = np.clip((lv - 0.80) / 0.20, 0.0, 1.0) * (1.0 if style != "rope" else 0.0)
    tone = np.clip(
        0.16
        + 0.28 * top_light
        + 0.38 * fold_h
        + 0.16 * (ao - 0.65)
        - 0.16 * hem_wear
        - 0.20 * cav,
        0.0,
        1.0,
    ).astype(np.float32)

    raw_alb = multi_stop_gradient(tone, pal, space="oklab", smooth_blend=float(smooth_blend), quantize_steps=int(quantize_steps))
    cl = pixel_clusters(raw_alb, pal_hex, direction=("diag_pos" if style == "rope" else "vertical"), cluster_size=2, clean_strays=True, clean_corners=True)
    rgb = (cl[..., :3] * 0.85 + raw_alb[..., :3] * 0.15).astype(np.float32)

    if trim_color and style != "rope" and float(trim_width) > 0.01:
        tw = float(np.clip(trim_width, 0.03, 0.25))
        trim_m = ((lv >= (1.0 - tw - 0.06)) & (lv <= (1.0 - 0.05))).astype(np.float32)
        c_tr = np.array(hex_to_srgb(trim_color), dtype=np.float32)[None, None, :]
        tr_shade = (0.74 + 0.30 * fold_h)[..., None]
        rgb = rgb * (1.0 - trim_m[..., None] * 0.86) + np.clip(c_tr * tr_shade, 0.0, 1.0) * (trim_m[..., None] * 0.86)

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    rough = np.clip(float(roughness_base) + 0.06 * cav + 0.04 * hem_wear, 0.72, 0.96).astype(np.float32)
    return material_make(
        albedo,
        height=fold_h,
        normal=height_to_normal_map(fold_h, strength=1.7, tileable=False),
        roughness=rough,
        metallic=0.0,
        ao=np.clip(ao * (1.0 - 0.20 * cav), 0.16, 1.0),
        porosity=0.65,
    )


# ---------------------------------------------------------------------------
# 5. Keratin, Horn, Ivory Tusks, Teeth, Claws & Hooves (MATERIAL Bus)
# ---------------------------------------------------------------------------


def keratin_horn_bone(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    edge_bevel: np.ndarray | None = None,
    local_uv: np.ndarray | None = None,
    stops: Sequence[Any] = (
        (0.00, "#352826"),  # dark desaturated keratin root / gumline collar
        (0.28, "#725B4B"),  # warm taupe shaft base
        (0.60, "#B89F7D"),  # aged bone midtone
        (0.86, "#E3D4B4"),  # warm ivory shaft
        (1.00, "#F5ECD8"),  # clean ivory tip (never harsh #FFFFFF!)
    ),
    orientation: str = "tip_up",  # "tip_up" | "tip_down" | "tip_out"
    ring_count: float = 5.0,
    striation_strength: float = 0.22,
    roughness_base: float = 0.42,
    smooth_blend: float = 0.64,
    quantize_steps: int = 7,
    seed: int = 409,
) -> dict[str, np.ndarray]:
    """Studio Keratin / Ivory / Bone MaterialBus (Longridr / ArtsByKev standard).

    Smooth dark-rooted-to-warm-ivory gradient with subtle transverse growth rings and
    longitudinal striations — ideal for Orc tusks, fangs, dragon horns, beaks, talons & hooves!
    """
    H, W = int(height), int(width)
    wpn, wn, ao, cav, bev, luv = _mesh_or_fallback(
        H, W,
        world_pos_norm=world_pos_norm,
        world_normal=world_normal,
        baked_ao=baked_ao,
        joint_cavity=joint_cavity,
        edge_bevel=edge_bevel,
        local_uv=local_uv,
    )
    pal = _normalize_stops(stops)
    lu, lv = luv[..., 0], luv[..., 1]

    if orientation == "tip_down":
        along = lv
    elif orientation == "tip_out":
        along = np.maximum(np.abs(lu - 0.5) * 2.0, 1.0 - lv)
    else:
        along = 1.0 - lv  # tip_up (lv=0 is top tip, lv=1 is dark gumline/root!)

    # Transverse keratin growth rings + fine longitudinal grain striations
    rings = 0.5 + 0.5 * np.sin(along * float(ring_count) * 2.0 * math.pi)
    grain = np.sin(lu * 18.0 * math.pi + noise_3d_field(wpn, scale=8.0, octaves=2, seed=seed) * 2.0) * 0.5 + 0.5
    top_light = np.clip(wn[..., 1] * 0.35 + wn[..., 2] * 0.25 + 0.55, 0.0, 1.0)

    h_map = np.clip(
        0.38
        + 0.28 * along
        + 0.14 * (rings - 0.5)
        + float(striation_strength) * 0.4 * (grain - 0.5)
        + 0.12 * bev
        - 0.20 * cav,
        0.05,
        0.96,
    ).astype(np.float32)

    tone = np.clip(
        0.10
        + 0.56 * np.power(np.clip(along, 0.0, 1.0), 0.78)
        + 0.18 * (top_light - 0.45)
        + 0.10 * (rings - 0.5)
        + float(striation_strength) * 0.32 * (grain - 0.5)
        + 0.12 * bev
        - 0.28 * cav,
        0.0,
        1.0,
    ).astype(np.float32)

    raw_alb = multi_stop_gradient(
        tone,
        pal,
        space="oklab",
        smooth_blend=min(0.32, float(smooth_blend)),
        quantize_steps=int(quantize_steps),
        dither="none",
        dither_strength=0.0,
    )
    pal_hex = [c for _, c in pal]
    cl = pixel_clusters(raw_alb, pal_hex, direction="vertical", cluster_size=2, clean_strays=True, clean_corners=True)
    rgb = (cl[..., :3] * 0.78 + raw_alb[..., :3] * 0.22).astype(np.float32)
    alb = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    rough = np.clip(float(roughness_base) - 0.12 * along + 0.14 * cav, 0.24, 0.72).astype(np.float32)
    return material_make(
        alb,
        height=h_map,
        normal=height_to_normal_map(h_map, strength=2.2, tileable=False),
        roughness=rough,
        metallic=0.0,
        ao=np.clip(ao * (1.0 - 0.24 * cav), 0.16, 1.0),
        sss=0.14,
    )


# ---------------------------------------------------------------------------
# 6. Part-Aware Semantic Entity Landmark Compositor (NEVER Paint Over Eyes/Parts!)
# ---------------------------------------------------------------------------


def _select_eye_facing_mask(
    mc: Any,
    eye_pattern: str,
    eye_facing: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(all_eye_mask, iris_face_mask, iris_u_fwd)`` for all elements matching ``eye_pattern``.

    Automatically selects the right visible face(s) per eye box:
    - ``eye_facing='front'``   -> ``south`` (+Z, standard SyncCraft/Blockbench forward!)
    - ``eye_facing='lateral'`` -> ``west`` (x<0) / ``east`` (x>0)
    - ``eye_facing='both'``    -> both ``south`` and outward lateral
    - ``eye_facing='auto'``    -> inspects each eye box's UV island aspect ratio & exposure!
    """
    H, W = mc.height, mc.width
    all_eye = mc.match_elements_mask(eye_pattern) > 0.5
    if not np.any(all_eye):
        z = np.zeros((H, W), dtype=np.float32)
        return z, z, mc.local_uv[..., 0].astype(np.float32)

    from synccraft.shading.uv_packer import FACES
    f_south = FACES.index("south")
    f_north = FACES.index("north")
    f_east = FACES.index("east")
    f_west = FACES.index("west")

    iris_face = np.zeros((H, W), dtype=bool)
    u_fwd = mc.local_uv[..., 0].astype(np.float32).copy()

    # Group matched islands by element index
    elem_ids = np.unique(mc.elem_id_map[all_eye])
    by_elem: dict[int, dict[str, Any]] = {}
    for isl in mc.islands:
        if isl.elem_idx in elem_ids:
            by_elem.setdefault(isl.elem_idx, {})[isl.face] = isl

    for eidx in elem_ids.tolist():
        em = mc.elem_id_map == eidx
        cx = float(mc.elem_centers_x[eidx]) if eidx < len(mc.elem_centers_x) else 0.0
        lat_name = "west" if cx < 0.0 else "east"
        lat_idx = f_west if cx < 0.0 else f_east
        fmap = by_elem.get(eidx, {})

        def _face_area_and_exp(fname: str) -> tuple[float, float]:
            isl = fmap.get(fname)
            if isl is None:
                return 0.0, 0.0
            sl_ao = mc.baked_ao[isl.y0 : isl.y0 + isl.h, isl.x0 : isl.x0 + isl.w]
            sl_cav = mc.joint_cavity[isl.y0 : isl.y0 + isl.h, isl.x0 : isl.x0 + isl.w]
            exp = float(np.mean(sl_ao * (1.0 - 0.55 * sl_cav))) if sl_ao.size else 0.0
            return float(isl.w * isl.h), exp

        chosen: list[int] = []
        if eye_facing == "front":
            chosen = [f_south]
        elif eye_facing == "north":
            chosen = [f_north]
        elif eye_facing == "lateral":
            chosen = [lat_idx]
        elif eye_facing == "both":
            chosen = [f_south, lat_idx]
        else:
            # Smart auto detection: compare south (+Z) vs outward lateral (west/east)
            s_area, s_exp = _face_area_and_exp("south")
            l_area, l_exp = _face_area_and_exp(lat_name)
            # Check island widths (south width = X extent, lateral width = Z extent)
            s_w = float(fmap["south"].w) if "south" in fmap else 0.0
            l_w = float(fmap[lat_name].w) if lat_name in fmap else 0.0
            if s_area > 0 and l_area > 0:
                if s_w >= l_w * 1.25 or (s_exp > l_exp + 0.16 and s_w >= l_w * 0.85):
                    chosen = [f_south]
                elif l_w >= s_w * 1.25 or (l_exp > s_exp + 0.16 and l_w >= s_w * 0.85):
                    chosen = [lat_idx]
                else:
                    # Roughly square/protruding eye box (owls, frogs, wolves, bugs): crisp on BOTH front & side!
                    chosen = [f_south, lat_idx]
            elif s_area > 0:
                chosen = [f_south]
            elif l_area > 0:
                chosen = [lat_idx]
            else:
                chosen = [f_south, f_north, lat_idx]

        for fidx in chosen:
            fm = em & (mc.face_id_map == fidx)
            iris_face |= fm
            if fidx == f_east:
                # East face (+u -> -Z): mirror horizontally so u_fwd=1 always points anteriorly (+Z)!
                u_fwd = np.where(fm, 1.0 - mc.local_uv[..., 0], u_fwd)
            elif fidx == f_south and cx > 0.0:
                # Right-side front eye (+X): mirror horizontally for exact L/R facial symmetry!
                u_fwd = np.where(fm, 1.0 - mc.local_uv[..., 0], u_fwd)

    return all_eye.astype(np.float32), iris_face.astype(np.float32), u_fwd.astype(np.float32)


def _overlay_bus_region(
    out: dict[str, np.ndarray],
    mask: np.ndarray,
    *,
    albedo_rgb: np.ndarray,
    height: np.ndarray | float | None = None,
    roughness: np.ndarray | float | None = None,
    metallic: np.ndarray | float | None = None,
    ao: np.ndarray | float | None = None,
    sss: np.ndarray | float | None = None,
    emissive_rgba: np.ndarray | None = None,
) -> None:
    """In-place composite of a semantic part mask cleanly over an existing `MATERIAL` bus."""
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if not np.any(m > 1e-4):
        return
    m3 = m[..., None]
    out["albedo"][..., :3] = out["albedo"][..., :3] * (1.0 - m3) + np.clip(albedo_rgb, 0.0, 1.0) * m3
    if height is not None:
        hv = np.asarray(height, dtype=np.float32)
        out["height"] = out["height"] * (1.0 - m) + hv * m
    if roughness is not None:
        rv = np.asarray(roughness, dtype=np.float32)
        out["roughness"] = out["roughness"] * (1.0 - m) + rv * m
    if metallic is not None:
        mv = np.asarray(metallic, dtype=np.float32)
        out["metallic"] = out["metallic"] * (1.0 - m) + mv * m
    if ao is not None:
        av = np.asarray(ao, dtype=np.float32)
        out["ao"] = out["ao"] * (1.0 - m) + av * m
    if sss is not None:
        sv = np.asarray(sss, dtype=np.float32)
        out["sss"] = out["sss"] * (1.0 - m) + sv * m
    if emissive_rgba is not None:
        em = np.asarray(emissive_rgba, dtype=np.float32)
        out["emissive"][..., :3] = out["emissive"][..., :3] * (1.0 - m3) + em[..., :3] * m3
        out["emissive"][..., 3] = out["emissive"][..., 3] * (1.0 - m) + em[..., 3] * m


def paint_entity_eyes_and_parts(
    base_mat: dict[str, np.ndarray],
    mesh_context: Any | None,
    *,
    # Eye controls (protected LAST so nothing ever paints over eyes!)
    eye_pattern: str = r"(^|/|__)(eye(_[lr]|\b)|eyeball|pupil|iris)",
    eye_facing: str = "auto",  # "auto" | "front" | "lateral" | "both"
    eye_style: str = "creature",  # "creature" | "gradient"
    eye_iris: str = "#F59E0B",
    eye_iris_inner: str = "#FEF08A",
    eye_iris_outer: str = "#B45309",
    eye_sclera: str = "#16121C",
    eye_pupil: str = "#09060E",
    eye_pupil_shape: str = "round",  # "round" | "slit"
    eye_glow: float = 0.35,
    eye_blink: float = 0.0,
    eyelid_color: str = "#2C382A",
    # Recessed orbital socket & brow ridge
    socket_pattern: str = r"(eye_?socket|orbital_(rim|hood|frame)|eyelid|brow_ridge)",
    socket_darken: float = 0.48,
    # Wet glossy nose / snout tip
    nose_pattern: str = r"(rhinarium|nostril|(^|/|__)nose(_tip)?\b|snout_tip)",
    nose_color: str = "#231B22",
    # Keratin beak / bill
    beak_pattern: str = r"(beak|bill)",
    beak_stops: Sequence[Any] = ("#3A322D", "#8E6F43", "#D7B46A", "#F0DFB0"),
    # Dark plum mouth interior & coral tongue (ArtsByKev / Longridr discipline!)
    mouth_pattern: str = r"(mouth_throat|palate|throat_back|gum|jaw_inner)",
    mouth_color: str = "#2A1422",
    tongue_pattern: str = r"tongue",
    tongue_color: str = "#AD4C5C",
    # Ivory teeth, fangs & tusks
    tooth_pattern: str = r"(tooth|teeth|fang|tusk)",
    tusk_root: str = "#42332B",
    tusk_tip: str = "#EFE4C8",
    # Keratin claws, talons, hooves, horns, antlers, spikes & stingers
    claw_pattern: str = r"(claw|talon|hoof|horn|antler|spike|spur|stinger|cercus|mandible_blade|scythe_blade)",
    claw_root: str = "#2E2426",
    claw_tip: str = "#D4C5A9",
    # Inner ear coral blush & dark paw pads
    ear_inner_pattern: str = r"(ear_inner|ear_scoop)",
    ear_inner_color: str = "#B6686C",
    paw_pad_pattern: str = r"(paw_pad|foot_pad)",
    paw_pad_color: str = "#2F2428",
    # Translucent backlit fish fins / amphibian webs
    fin_pattern: str = r"(_fin(_slab)?|caudal_(lo|up)|dorsal_sail_web|(foot|paw)_web)",
    fin_base_color: str = "#1C6B82",
    fin_tip_color: str = "#7CE8DF",
    # Glowing emissive runes / cores / gems (animatable via rune_glow!)
    rune_pattern: str = r"(rune|ember|magma_core|gem|crystal|orb|core_glow|lantern)",
    rune_color: str = "#FF7A21",
    rune_core_color: str = "#FFF5C2",
    rune_glow: float = 0.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Part-Aware Semantic Entity Detailing Composite Node (Full `MATERIAL` Bus).

    Automatically separates and hand-shades EVERY anatomical landmark on a 3D entity using its
    `MeshBakeContext` so body fur, scales, stone, or skin NEVER paint over:
    1. Recessed orbital sockets & overhanging brow shadows
    2. Dark glossy wet noses, paw pads & muted coral-pink inner ears
    3. Translucent ray-striated fins & amphibian webs
    4. Dark desaturated plum mouth interiors + coral tongues
    5. Dark-rooted warm ivory teeth, fangs & tusks
    6. Dark-rooted keratin beaks, claws, talons, hooves, horns & spikes
    7. Glowing emissive runes / gems / cores (animatable via `rune_glow`)
    8. **PROTECTED LAST**: Crisp studio eyes (auto front/lateral face selection, recessed orbital
       socket on non-facing sides, multi-layer iris/pupil/catchlight OR non-standard gradient eye,
       animated `eye_blink` & `eye_glow`!)
    """
    out = _ensure_bus(base_mat, tileable=False)
    # Deep copy mutable arrays so upstream cached bus is untouched
    for k in MATERIAL_BUS_PORTS:
        if k in out and isinstance(out[k], np.ndarray):
            out[k] = out[k].copy()
    H, W = out["albedo"].shape[:2]

    if mesh_context is None:
        return out

    mc = mesh_context
    wpn, wn, ao, cav, bev, luv = (
        mc.world_pos_norm,
        mc.world_normal,
        mc.baked_ao,
        mc.joint_cavity,
        mc.edge_bevel,
        mc.local_uv,
    )
    lu, lv = luv[..., 0], luv[..., 1]
    top_light = np.clip(wn[..., 1] * 0.45 + wn[..., 2] * 0.25 + 0.52, 0.0, 1.0)
    eye_all_m, iris_face_m, eye_u_fwd = _select_eye_facing_mask(mc, eye_pattern, eye_facing) if eye_pattern else (
        np.zeros((H, W), dtype=np.float32), np.zeros((H, W), dtype=np.float32), lu
    )
    not_eye = (1.0 - eye_all_m).astype(np.float32)

    # 1. Recessed orbital sockets, eyelids & overhanging brow shadows
    if socket_pattern:
        sock_m = mc.match_elements_mask(socket_pattern) * not_eye
        if np.any(sock_m > 0.01):
            dk = float(np.clip(socket_darken, 0.0, 0.85))
            # Hue-shifted deep cool orbital shadow (rich dark plum-slate, never flat muddy grey)
            sock_rgb = out["albedo"][..., :3] * (1.0 - dk) + np.array([0.07, 0.05, 0.11], dtype=np.float32)[None, None, :] * dk
            # Subtle upper catchlight on brow ridge top face only
            brow_top = np.clip((wn[..., 1] - 0.55) * 2.2, 0.0, 1.0) * sock_m
            sock_rgb = np.clip(sock_rgb + brow_top[..., None] * 0.12, 0.0, 1.0)
            _overlay_bus_region(out, sock_m, albedo_rgb=sock_rgb, ao=np.clip(out["ao"] * (1.0 - 0.42 * dk), 0.10, 1.0))

    # 2. Translucent ray-striated fish fins & amphibian webs
    if fin_pattern:
        fin_m = mc.match_elements_mask(fin_pattern) * not_eye
        if np.any(fin_m > 0.01):
            ray = 0.5 + 0.5 * np.sin((lu * 12.0 + lv * 3.0) * math.pi)
            grad = np.clip(0.35 * (1.0 - lv) + 0.45 * lu + 0.20 * ray, 0.0, 1.0)
            c0 = np.array(hex_to_srgb(fin_base_color), dtype=np.float32)[None, None, :]
            c1 = np.array(hex_to_srgb(fin_tip_color), dtype=np.float32)[None, None, :]
            fin_rgb = np.clip((c0 * (1.0 - grad[..., None]) + c1 * grad[..., None]) * (0.72 + 0.36 * top_light[..., None]), 0.0, 1.0)
            _overlay_bus_region(
                out, fin_m,
                albedo_rgb=fin_rgb,
                height=0.35 + 0.28 * ray,
                roughness=0.26,
                metallic=0.05,
                sss=0.68,
            )

    # 3. Inner ear coral-pink blush framed by dark maroon rim + dark paw pads
    if ear_inner_pattern:
        ear_m = mc.match_elements_mask(ear_inner_pattern) * not_eye
        if np.any(ear_m > 0.01):
            d_e = np.minimum(np.minimum(lu, 1.0 - lu), np.minimum(lv, 1.0 - lv))
            rim = np.clip(1.0 - d_e / 0.16, 0.0, 1.0)
            c_in = np.array(hex_to_srgb(ear_inner_color), dtype=np.float32)[None, None, :]
            c_rim = c_in * np.array([0.46, 0.34, 0.42], dtype=np.float32)[None, None, :]
            ear_rgb = np.clip((c_in * (1.0 - rim[..., None]) + c_rim * rim[..., None]) * (0.76 + 0.28 * ao[..., None]), 0.0, 1.0)
            _overlay_bus_region(out, ear_m, albedo_rgb=ear_rgb, roughness=0.58, sss=0.42)

    if paw_pad_pattern:
        pad_m = mc.match_elements_mask(paw_pad_pattern) * not_eye
        if np.any(pad_m > 0.01):
            c_pad = np.array(hex_to_srgb(paw_pad_color), dtype=np.float32)[None, None, :]
            pad_rgb = np.clip(c_pad * (0.72 + 0.34 * top_light[..., None] + 0.14 * bev[..., None]), 0.0, 1.0)
            _overlay_bus_region(out, pad_m, albedo_rgb=pad_rgb, roughness=0.52, height=0.44 + 0.12 * bev)

    # 4. Dark desaturated plum mouth interior (NEVER #000000!) + warm rose-coral tongue
    if mouth_pattern:
        mth_m = mc.match_elements_mask(mouth_pattern) * not_eye
        if np.any(mth_m > 0.01):
            c_mth = np.array(hex_to_srgb(mouth_color), dtype=np.float32)[None, None, :]
            mth_rgb = np.clip(c_mth * (0.65 + 0.45 * ao[..., None]), 0.0, 1.0)
            _overlay_bus_region(out, mth_m, albedo_rgb=mth_rgb, roughness=0.28, ao=np.clip(ao * 0.55, 0.10, 1.0))

    if tongue_pattern:
        tng_m = mc.match_elements_mask(tongue_pattern) * not_eye
        if np.any(tng_m > 0.01):
            c_tng = np.array(hex_to_srgb(tongue_color), dtype=np.float32)[None, None, :]
            groove = np.exp(-((lu - 0.5) / 0.10) ** 2) * 0.20
            tng_rgb = np.clip(c_tng * (0.76 + 0.32 * top_light[..., None] - groove[..., None]), 0.0, 1.0)
            _overlay_bus_region(out, tng_m, albedo_rgb=tng_rgb, roughness=0.24, sss=0.50, height=0.56 - groove)

    # 5. Wet glossy dark nose / rhinarium / snout tip
    if nose_pattern:
        nose_m = mc.match_elements_mask(nose_pattern) * not_eye
        if np.any(nose_m > 0.01):
            c_ns = np.array(hex_to_srgb(nose_color), dtype=np.float32)[None, None, :]
            # Subtle wet specular shine on top plane + twin dark nostril pits on front
            spec_dot = (np.clip(wn[..., 1], 0.0, 1.0) * np.clip(1.0 - np.abs(lu - 0.5) * 2.4, 0.0, 1.0)) * 0.28
            nostril = (
                ((lu - 0.30) ** 2 + (lv - 0.58) ** 2 < 0.016)
                | ((lu - 0.70) ** 2 + (lv - 0.58) ** 2 < 0.016)
            ).astype(np.float32) * np.clip(wn[..., 2], 0.0, 1.0)
            ns_rgb = np.clip(c_ns * (0.72 + 0.38 * top_light[..., None]) + spec_dot[..., None] - nostril[..., None] * 0.16, 0.02, 1.0)
            _overlay_bus_region(out, nose_m, albedo_rgb=ns_rgb, roughness=0.18, height=0.62 - 0.25 * nostril)

    # 6. Keratin beak / bill
    if beak_pattern:
        bk_m = mc.match_elements_mask(beak_pattern) * not_eye
        if np.any(bk_m > 0.01):
            bk_bus = keratin_horn_bone(
                H, W,
                world_pos_norm=wpn, world_normal=wn, baked_ao=ao, joint_cavity=cav, edge_bevel=bev, local_uv=luv,
                stops=beak_stops, orientation="tip_down", ring_count=3.0, roughness_base=0.34, seed=seed + 5,
            )
            _overlay_bus_region(
                out, bk_m,
                albedo_rgb=bk_bus["albedo"][..., :3],
                height=bk_bus["height"],
                roughness=bk_bus["roughness"],
                metallic=0.0,
                ao=bk_bus["ao"],
            )

    # 7. Keratin claws, talons, hooves, horns, antlers, spikes & stingers
    if claw_pattern:
        clw_m = mc.match_elements_mask(claw_pattern) * not_eye
        if np.any(clw_m > 0.01):
            clw_bus = keratin_horn_bone(
                H, W,
                world_pos_norm=wpn, world_normal=wn, baked_ao=ao, joint_cavity=cav, edge_bevel=bev, local_uv=luv,
                stops=(claw_root, "#5E4C42", "#9A846D", claw_tip),
                orientation="tip_down",
                ring_count=4.0,
                roughness_base=0.38,
                seed=seed + 11,
            )
            _overlay_bus_region(
                out, clw_m,
                albedo_rgb=clw_bus["albedo"][..., :3],
                height=clw_bus["height"],
                roughness=clw_bus["roughness"],
                metallic=0.04,
                ao=clw_bus["ao"],
            )

    # 8. Warm ivory teeth, fangs & tusks (dark gumline collar -> striated warm ivory tip)
    if tooth_pattern:
        tth_m = mc.match_elements_mask(tooth_pattern) * not_eye
        if np.any(tth_m > 0.01):
            tth_bus = keratin_horn_bone(
                H, W,
                world_pos_norm=wpn, world_normal=wn, baked_ao=ao, joint_cavity=cav, edge_bevel=bev, local_uv=luv,
                stops=(tusk_root, "#9E896D", "#D8C9A8", tusk_tip),
                orientation="tip_up",
                ring_count=3.0,
                striation_strength=0.18,
                roughness_base=0.32,
                seed=seed + 19,
            )
            _overlay_bus_region(
                out, tth_m,
                albedo_rgb=tth_bus["albedo"][..., :3],
                height=tth_bus["height"],
                roughness=tth_bus["roughness"],
                metallic=0.0,
                ao=tth_bus["ao"],
            )

    # 9. Glowing runes / cores / gems / magma embers (animatable via rune_glow!)
    if rune_pattern and float(rune_glow) > 0.01:
        rn_m = mc.match_elements_mask(rune_pattern) * not_eye
        if np.any(rn_m > 0.01):
            rg = float(np.clip(rune_glow, 0.0, 1.5))
            d_c = np.sqrt((lu - 0.5) ** 2 + (lv - 0.5) ** 2) * 2.0
            core_w = np.clip(1.0 - d_c * 0.85, 0.0, 1.0)
            c_out = np.array(hex_to_srgb(rune_color), dtype=np.float32)[None, None, :]
            c_core = np.array(hex_to_srgb(rune_core_color), dtype=np.float32)[None, None, :]
            rn_rgb = np.clip((c_out * (1.0 - core_w[..., None]) + c_core * core_w[..., None]) * (0.65 + 0.45 * rg), 0.0, 1.0)
            em_a = np.clip(rn_m * (0.45 + 0.55 * core_w) * min(1.0, rg), 0.0, 1.0)
            rn_em = np.concatenate((rn_rgb, em_a[..., None]), axis=-1).astype(np.float32)
            _overlay_bus_region(
                out, rn_m,
                albedo_rgb=rn_rgb,
                height=0.72 + 0.16 * core_w,
                roughness=0.14,
                metallic=0.10,
                ao=1.0,
                emissive_rgba=rn_em,
            )

    # 10. PROTECTED LAST: Eyes! Dark orbital socket on non-facing sides + crisp eye + animated blink & glow!
    if np.any(eye_all_m > 0.5):
        c_sock = np.array(hex_to_srgb(eye_sclera), dtype=np.float32)[None, None, :]
        # Paint non-facing sides of the eye box in dark recessed socket tone
        non_face = np.clip(eye_all_m - iris_face_m, 0.0, 1.0)
        if np.any(non_face > 0.01):
            _overlay_bus_region(
                out, non_face,
                albedo_rgb=np.broadcast_to(c_sock * 0.78, (H, W, 3)),
                roughness=0.55,
                metallic=0.0,
                ao=0.35,
                emissive_rgba=np.zeros((H, W, 4), dtype=np.float32),
            )

        if np.any(iris_face_m > 0.01):
            if eye_style == "gradient":
                e_rgba, e_em_mask = gradient_eye_paint(
                    eye_u_fwd,
                    lv,
                    side_sign=None,  # already mirror-normalized per face!
                    stops=((0.0, eye_sclera), (0.30, eye_iris_outer), (0.68, eye_iris), (0.94, eye_iris_inner), (1.0, "#FFFFFF")),
                    brow_shadow=0.48,
                    style="gradient_glow",
                    rim_color=eye_sclera,
                )
            else:
                e_rgba, e_em_mask = paint_creature_eye(
                    eye_u_fwd,
                    lv,
                    side_sign=None,  # already mirror-normalized per face!
                    iris_inner=eye_iris_inner,
                    iris_mid=eye_iris,
                    iris_outer=eye_iris_outer,
                    limbal=eye_sclera,
                    sclera=eye_sclera,
                    pupil=eye_pupil,
                    catchlight="#FFFFFF",
                    pupil_shape=eye_pupil_shape,
                )

            eye_rgb = e_rgba[..., :3].copy()
            eg = float(np.clip(eye_glow, 0.0, 1.5))
            # Subtle luminance boost on glowing eyes so animated pulse visibly breathes in albedo too!
            eye_rgb = np.clip(eye_rgb * (0.82 + 0.26 * eg * e_em_mask[..., None]), 0.0, 1.0)
            em_str = np.clip(e_em_mask * min(1.0, eg), 0.0, 1.0)

            # Animated eyelid blink (0.0 = wide open -> 1.0 = fully closed with dark lash seam!)
            blk = float(np.clip(eye_blink, 0.0, 1.0))
            lid_m = np.zeros((H, W), dtype=np.float32)
            if blk > 0.02:
                upper_cut = 0.08 + 0.68 * blk
                lower_cut = 0.94 - 0.24 * blk
                lid_m = ((lv <= upper_cut) | (lv >= lower_cut)).astype(np.float32)
                lash_line = ((np.abs(lv - upper_cut) < 0.07) | (np.abs(lv - lower_cut) < 0.05)).astype(np.float32)
                c_lid = np.array(hex_to_srgb(eyelid_color), dtype=np.float32)[None, None, :]
                lid_rgb = np.clip(c_lid * (0.78 + 0.26 * (1.0 - lv[..., None])) * (1.0 - 0.62 * lash_line[..., None]), 0.0, 1.0)
                eye_rgb = eye_rgb * (1.0 - lid_m[..., None]) + lid_rgb * lid_m[..., None]
                em_str = em_str * (1.0 - lid_m)

            eye_em = np.concatenate((eye_rgb, (em_str * iris_face_m)[..., None]), axis=-1).astype(np.float32)
            eye_rough = np.where(lid_m > 0.5, 0.64, 0.08).astype(np.float32)
            _overlay_bus_region(
                out, iris_face_m,
                albedo_rgb=eye_rgb,
                height=np.where(lid_m > 0.5, 0.55, 0.78).astype(np.float32),
                roughness=eye_rough,
                metallic=0.0,
                ao=1.0,
                emissive_rgba=eye_em,
            )

    # Recompute clean tangent-space normals on modified relief
    out["normal"] = height_to_normal_map(out["height"], strength=2.4, tileable=False)
    out["eye_mask"] = iris_face_m
    return out
