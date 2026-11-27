"""Full-Spectrum Biological, Natural & World-Block PBR Material Generators (`TexGraph v7`).

Benchmarked directly against YouTube & studio pixel-art / Blockbench references:
- **World Blocks (100% Seamless Toroidal `MATERIAL`-Bus Generators)**:
  * ``cobblestone_masonry``   : Flood-Fill + Slope-Blur + Curvature Edge Wear cobble/ashlar
                                (ArtsByKev 16x16 & 32x32 seamless stone craft).
  * ``bark_and_timber``       : Furrowed organic tree bark with Slope-Blur splits & crevice moss.
  * ``crystalline_ore_vein``  : Chipped host stone + faceted sparkling mineral ore crystals.
  * ``caustic_water_surface`` : Seamless looping animated water caustics & pixel foam crests.
- **Real-Life Animals & Imaginary Creatures (3D-Mesh-Aware & 2D-Swatch `MATERIAL`-Bus Generators)**:
  * ``organic_fur_pelt``      : Directional 2-4px tufted mammal fur locks + jagged countershading
                                + markings (`solid`, `saddle`, `tabby`, `spots`, `dappled`)
                                (Everbloom Husky Wolf standard — zero dirt speckle!).
  * ``layered_plumage``       : Overlapping scalloped bird/griffin/phoenix feathers + central quill
                                + barred/speckled/iridescent markings.
  * ``chitin_carapace``       : Segmented arthropod/scorpion/beetle plates + cool plum shadows
                                + Fresnel iridescence (Mickey Joe Scorpion standard!).
  * ``aquatic_scales_skin``   : Shimmering fish cycloid scales OR moist porous dart-frog/koi skin
                                with glossy wet specular PBR roughness.
  * ``elemental_golem_surface``: Flood-Fill + Slope-Blur fractured golem rock/obsidian/ice slabs
                                 with chipped edge wear & recessed glowing energy fissures!
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .engine_ops import (
    curvature_from_height,
    edge_wear_mask,
    flood_fill,
    histogram_scan,
    sdf_shape,
    slope_blur,
    tile_sampler,
)
from .generators_gradients import multi_stop_gradient
from .generators_noise import fbm_noise, uv_grid, voronoi_noise
from .generators_painterly import directional_brushstrokes, organic_domain_warp_3d
from .ops_fx_color import oklch_adjust
from .pbr import height_to_normal_map
from .pixel_art_ops import hex_to_srgb
from .pixel_composer_ops import heightmap_shadow, pixel_clusters


# ---------------------------------------------------------------------------
# 1. World Block Seamless PBR Material Generators
# ---------------------------------------------------------------------------


def cobblestone_masonry(
    height: int,
    width: int,
    *,
    cols: int = 4,
    rows: int = 4,
    stone_stops: Sequence[tuple[float, str]] = (
        (0.00, "#3A373C"),  # cool deep slate-violet mortar shadow
        (0.24, "#615D5D"),  # cool recessed stone shadow
        (0.52, "#8B847E"),  # balanced warm-neutral cobble midtone
        (0.78, "#B7A89A"),  # warm sunlit top-left edge (ArtsByKev #b7a59a!)
        (1.00, "#DCD1C4"),  # crisp mineral catchlight
    ),
    mortar_color: str = "#2F2C33",
    moss_amount: float = 0.0,
    wear_strength: float = 0.35,
    smooth_blend: float = 0.42,
    quantize_steps: int = 6,
    tileable: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Seamless Minecraft-Grade Cobblestone & Ashlar Masonry (`Flood Fill + Slope Blur + Edge Wear`).

    Produces bold, readable 3-6 px organic rounded stone clusters with warm top-left lit bevels,
    cool shadowed crevices, per-stone height & tilt diversity, and ZERO 1x1 noisy static!
    """
    H, W = int(height), int(width)
    # 1. Voronoi cellular stones with clean organic rounding (seamless toroidal!)
    scale_x = max(2.0, float(cols))
    scale_y = max(2.0, float(rows))
    f2m1 = voronoi_noise(H, W, scale=scale_x, scale_y=scale_y, jitter=0.68, output="f2_minus_f1", seed=seed, tileable=tileable)
    # Isolate bold interlocking stone bodies separated by crisp mortar channels
    mortar_thresh = 0.075 if min(H, W) <= 20 else 0.068
    stone_mask = (f2m1 > mortar_thresh).astype(np.float32)

    # 2. Flood Fill per-stone identity (unique height + unique light-catching slope per cobble!)
    ff = flood_fill(stone_mask, threshold=0.5, tileable=tileable, seed=seed + 13)
    # Smooth rounded pillow dome per stone from F2-F1 + chamfer distance
    dome = np.clip((f2m1 - mortar_thresh) / max(0.14, 0.34 - mortar_thresh), 0.0, 1.0)
    dome = np.power(dome, 0.52)

    base_h = np.clip(
        stone_mask * (0.36 + 0.38 * dome + 0.22 * (ff["random_gray"] - 0.5) + 0.16 * (ff["random_slope"] - 0.5)),
        0.0,
        1.0,
    ).astype(np.float32)

    # 3. Substance Slope Blur chipping (breaks straight Voronoi polygons into hand-knapped cobbles!)
    chip_n = fbm_noise(H, W, scale=max(3.0, scale_x * 1.4), octaves=2, seed=seed + 37, tileable=tileable)
    sb_intensity = 0.40 if min(H, W) <= 20 else 0.78
    chipped_h = slope_blur(base_h, chip_n, intensity=sb_intensity * (0.4 + 0.6 * float(wear_strength)), samples=3, mode="min", tileable=tileable)
    stone_h = np.clip(0.64 * base_h + 0.36 * chipped_h, 0.0, 1.0).astype(np.float32)

    # 4. Top-left directional bevel lighting (ArtsByKev signature warm top-left vs cool bottom-right!)
    if tileable:
        gx = np.roll(stone_h, -1, axis=1) - np.roll(stone_h, 1, axis=1)
        gy = np.roll(stone_h, -1, axis=0) - np.roll(stone_h, 1, axis=0)
    else:
        gx = np.gradient(stone_h, axis=1)
        gy = np.gradient(stone_h, axis=0)
    # Light from top-left (-0.7, -0.7): top & left rims of each stone rise downward/rightward -> warm bright catchlight!
    tl_lit = np.clip(0.5 + (gx * 0.75 + gy * 0.85) * 1.25, 0.0, 1.0)

    ew = edge_wear_mask(stone_h, wear_amount=0.45, chip_scale=scale_x * 1.8, grime_amount=0.52, tileable=tileable, seed=seed + 53)
    mortar_m = (1.0 - np.clip(stone_h / 0.24, 0.0, 1.0)).astype(np.float32)

    tone = np.clip(
        0.18
        + 0.46 * stone_h
        + 0.28 * (tl_lit - 0.5)
        + 0.14 * ew["wear"] * float(wear_strength)
        + 0.12 * (ff["random_gray"] - 0.5) * stone_mask
        - 0.28 * mortar_m,
        0.0,
        1.0,
    ).astype(np.float32)

    rgba = multi_stop_gradient(
        tone,
        stone_stops,
        space="oklab",
        smooth_blend=max(0.48, float(smooth_blend)),
        quantize_steps=quantize_steps,
        dither="none",
        dither_strength=0.0,
    )
    mc = np.array(hex_to_srgb(mortar_color), dtype=np.float32)[None, None, :]
    rgb = rgba[..., :3] * (1.0 - mortar_m[..., None] * 0.56) + mc * (mortar_m[..., None] * 0.56)

    # Optional crevice moss tucked organically inside mortar seams!
    ma = float(np.clip(moss_amount, 0.0, 1.0))
    moss_m = np.zeros((H, W), dtype=np.float32)
    if ma > 0.01:
        moss_n = fbm_noise(H, W, scale=max(3.0, scale_x * 1.2), octaves=2, seed=seed + 79, tileable=tileable)
        moss_m = histogram_scan(np.clip(mortar_m * 0.75 + ew["grime"] * 0.45, 0.0, 1.0) * moss_n, position=ma, contrast=0.72)
        moss_rgba = multi_stop_gradient(
            np.clip(0.35 + 0.55 * tl_lit + 0.2 * (moss_n - 0.5), 0.0, 1.0),
            ((0.0, "#1E4620"), (0.48, "#4B8B3B"), (0.85, "#8BC34A"), (1.0, "#CBEB7B")),
            space="oklab",
            smooth_blend=0.45,
            quantize_steps=5,
        )
        rgb = rgb * (1.0 - moss_m[..., None]) + moss_rgba[..., :3] * moss_m[..., None]
        stone_h = np.clip(stone_h + moss_m * 0.14, 0.0, 1.0)

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    normal_rgb = height_to_normal_map(stone_h, strength=2.6, tileable=tileable)
    roughness = np.clip(0.76 + 0.16 * mortar_m - 0.14 * ew["wear"] + 0.12 * moss_m, 0.35, 0.96).astype(np.float32)
    ao = np.clip(1.0 - 0.65 * mortar_m - 0.20 * ew["grime"], 0.18, 1.0).astype(np.float32)
    porosity = np.clip(0.42 + 0.35 * mortar_m + 0.35 * moss_m, 0.0, 1.0).astype(np.float32)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": stone_h,
        "normal": normal_rgb,
        "roughness": roughness,
        "metallic": np.zeros((H, W), dtype=np.float32),
        "ao": ao,
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
        "sss": (moss_m * 0.32).astype(np.float32),
        "porosity": porosity,
        "mortar_mask": mortar_m,
    }


def bark_and_timber(
    height: int,
    width: int,
    *,
    bark_stops: Sequence[tuple[float, str]] = (
        (0.00, "#24171A"),  # deep plum-espresso bark fissure
        (0.28, "#472C24"),  # rich umber shadow bark
        (0.58, "#724A34"),  # warm oak bark ridge
        (0.84, "#9E7250"),  # sunlit weathered ridge highlight
        (1.00, "#C9A17A"),  # dry silvered bark edge
    ),
    furrow_count: float = 5.0,
    moss_amount: float = 0.20,
    tileable: bool = True,
    seed: int = 109,
) -> dict[str, np.ndarray]:
    """Organic furrowed Minecraft oak bark with calm vertical ridges & crevice lichen/moss (100% seamless!)."""
    H, W = int(height), int(width)
    u, v = uv_grid(H, W)
    # Broad, calm, seamless vertical bark strips warped softly by low-frequency organic flow (ZERO speckle!)
    fc = max(3, int(round(furrow_count)))
    warp_x = (fbm_noise(H, W, scale=2.0, scale_y=2.0, octaves=2, seed=seed + 19, tileable=tileable) - 0.5) * 0.28
    phase_x = (u + warp_x) * fc
    strip_wave = 0.5 + 0.5 * np.cos(phase_x * 2.0 * math.pi)
    strip_domes = np.power(strip_wave, 0.72)
    grain = fbm_noise(H, W, scale=float(fc) * 1.5, scale_y=2.0, octaves=2, seed=seed + 5, tileable=tileable)
    # Gentle staggered horizontal bark knots/breaks (soft, not harsh cuts!)
    knot_v = fbm_noise(H, W, scale=2.0, scale_y=max(2.0, float(fc) * 0.8), octaves=2, seed=seed + 41, tileable=tileable)
    bark_raw = np.clip(0.62 * strip_domes + 0.26 * grain + 0.12 * knot_v, 0.0, 1.0).astype(np.float32)
    bark_h = slope_blur(bark_raw, grain, intensity=0.45 if min(H, W) <= 20 else 0.72, samples=3, mode="blur", tileable=tileable)

    curv = curvature_from_height(bark_h, radius=1, strength=1.5, tileable=tileable)
    gx = (np.roll(bark_h, -1, axis=1) - np.roll(bark_h, 1, axis=1)) if tileable else np.gradient(bark_h, axis=1)
    tone = np.clip(0.18 + 0.64 * bark_h + 0.18 * gx + 0.08 * curv["convex"] - 0.10 * curv["concave"], 0.0, 1.0).astype(np.float32)

    rgba = multi_stop_gradient(tone, bark_stops, space="oklab", smooth_blend=0.58, quantize_steps=6, dither="none")
    rgb = rgba[..., :3]

    ma = float(np.clip(moss_amount, 0.0, 1.0))
    moss_m = np.zeros((H, W), dtype=np.float32)
    if ma > 0.01:
        mn = fbm_noise(H, W, scale=3.0, octaves=2, seed=seed + 67, tileable=tileable)
        crevice = np.clip(1.0 - strip_domes / 0.42, 0.0, 1.0)
        moss_m = histogram_scan(np.clip(crevice * mn, 0.0, 1.0), position=ma + 0.18, contrast=0.68)
        mcol = multi_stop_gradient(np.clip(0.38 + 0.52 * mn, 0.0, 1.0), ((0.0, "#244C28"), (0.6, "#5C9A3C"), (1.0, "#A9D464")), space="oklab", smooth_blend=0.55, quantize_steps=4)
        rgb = rgb * (1.0 - moss_m[..., None]) + mcol[..., :3] * moss_m[..., None]

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": bark_h,
        "normal": height_to_normal_map(bark_h, strength=2.4, tileable=tileable),
        "roughness": np.clip(0.84 - 0.12 * curv["convex"] + 0.08 * moss_m, 0.55, 0.96).astype(np.float32),
        "metallic": np.zeros((H, W), dtype=np.float32),
        "ao": np.clip(0.45 + 0.55 * bark_h, 0.22, 1.0).astype(np.float32),
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
        "sss": (moss_m * 0.25).astype(np.float32),
        "porosity": np.clip(0.60 + 0.25 * moss_m, 0.0, 1.0).astype(np.float32),
    }


def crystalline_ore_vein(
    height: int,
    width: int,
    *,
    host_stops: Sequence[tuple[float, str]] = (
        (0.00, "#1D1D29"),
        (0.32, "#353748"),
        (0.66, "#575B70"),
        (1.00, "#868CA3"),
    ),
    ore_stops: Sequence[tuple[float, str]] = (
        (0.00, "#0E4C75"),
        (0.35, "#06B6D4"),
        (0.72, "#67E8F9"),
        (1.00, "#F0FDFF"),
    ),
    ore_density: float = 0.55,
    emissive_strength: float = 0.65,
    tileable: bool = True,
    seed: int = 211,
) -> dict[str, np.ndarray]:
    """Chipped deepslate/basalt host stone embedded with sharp faceted mineral crystal ore clusters!"""
    H, W = int(height), int(width)
    # 1. Host stone via cobblestone_masonry (subtle dark deepslate slabs)
    host = cobblestone_masonry(H, W, cols=3, rows=3, stone_stops=host_stops, mortar_color="#14141D", wear_strength=0.45, smooth_blend=0.42, quantize_steps=5, tileable=tileable, seed=seed)

    # 2. Faceted gem stamp scattered via Tile Sampler (100% seamless!)
    gem_sdf = sdf_shape(16, 16, shape="gem", size=(0.28, 0.34), bevel_width=0.22)
    gem_stamp = np.clip(gem_sdf["bevel"] * 1.15, 0.0, 1.0)
    sc = tile_sampler(
        H, W, gem_stamp,
        count_x=3, count_y=3,
        distribution="stagger_grid",
        pos_jitter=0.28,
        scale_base=0.56,
        scale_jitter=0.24,
        rot_jitter_deg=35.0,
        lum_jitter=0.18,
        density=float(np.clip(ore_density, 0.15, 1.0)),
        tileable=tileable,
        seed=seed + 31,
    )
    ore_m = (sc["height"] > 0.22).astype(np.float32)
    # Dark socket border hugging each crystal cluster so gems pop in 16x16 & 32x32!
    if tileable:
        ore_dil = np.maximum.reduce([ore_m, np.roll(ore_m, 1, 0), np.roll(ore_m, -1, 0), np.roll(ore_m, 1, 1), np.roll(ore_m, -1, 1)])
    else:
        p = np.pad(ore_m, 1, mode="edge")
        ore_dil = np.maximum.reduce([ore_m, p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
    socket = np.clip(ore_dil - ore_m, 0.0, 1.0)

    # Crisp faceted specular crystal shading (top-left bright facet, bottom-right deep hue!)
    oh = sc["height"]
    gx = (np.roll(oh, -1, axis=1) - np.roll(oh, 1, axis=1)) if tileable else np.gradient(oh, axis=1)
    gy = (np.roll(oh, -1, axis=0) - np.roll(oh, 1, axis=0)) if tileable else np.gradient(oh, axis=0)
    facet_tone = np.clip(0.32 + 0.56 * oh + 0.34 * (gx + gy), 0.0, 1.0).astype(np.float32)
    ore_rgba = multi_stop_gradient(facet_tone, ore_stops, space="oklab", smooth_blend=0.42, quantize_steps=5, dither="none")

    rgb = host["albedo"][..., :3] * (1.0 - socket[..., None] * 0.34)
    rgb = rgb * (1.0 - ore_m[..., None]) + ore_rgba[..., :3] * ore_m[..., None]
    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)

    tot_h = np.clip(host["height"] * (1.0 - ore_m) + (0.62 + 0.38 * oh) * ore_m - 0.14 * socket, 0.0, 1.0).astype(np.float32)
    em_w = np.clip((facet_tone - 0.42) * 1.75 * ore_m * float(emissive_strength), 0.0, 1.0)
    emissive = np.concatenate((ore_rgba[..., :3] * em_w[..., None], (em_w > 0.05).astype(np.float32)[..., None]), axis=-1).astype(np.float32)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": tot_h,
        "normal": height_to_normal_map(tot_h, strength=2.9, tileable=tileable),
        "roughness": np.clip(host["roughness"] * (1.0 - ore_m) + 0.12 * ore_m, 0.08, 0.95).astype(np.float32),
        "metallic": np.clip(0.65 * ore_m, 0.0, 1.0).astype(np.float32),
        "ao": np.clip(host["ao"] * (1.0 - 0.32 * socket), 0.16, 1.0).astype(np.float32),
        "emissive": emissive,
        "sss": (0.45 * ore_m).astype(np.float32),
        "porosity": (host["porosity"] * (1.0 - ore_m)).astype(np.float32),
        "ore_mask": ore_m,
    }


def caustic_water_surface(
    height: int,
    width: int,
    *,
    water_stops: Sequence[tuple[float, str]] = (
        (0.00, "#0C356A"),
        (0.32, "#1565A8"),
        (0.64, "#279EFF"),
        (0.88, "#7DD3FC"),
        (1.00, "#E0F7FF"),
    ),
    scale: float = 4.0,
    phase: float = 0.0,
    tileable: bool = True,
    seed: int = 313,
) -> dict[str, np.ndarray]:
    """Seamless looping animated water surface with dual-layer Voronoi caustics & sparkling pixel ripples."""
    H, W = int(height), int(width)
    # Seamless circular scroll shift along phase
    s_x = int(round(phase * W))
    s_y = int(round(phase * H))
    c1 = voronoi_noise(H, W, scale=max(2.0, float(scale)), output="edge", phase=phase, loop_speed=1.0, seed=seed, tileable=tileable)
    c2 = voronoi_noise(H, W, scale=max(2.0, float(scale) * 1.25), output="edge", phase=phase, loop_speed=1.0, seed=seed + 29, tileable=tileable)
    c1_s = np.roll(np.roll(c1, s_x, axis=1), s_y, axis=0) if tileable else c1
    c2_s = np.roll(np.roll(c2, -s_x, axis=1), s_y, axis=0) if tileable else c2
    rip = fbm_noise(H, W, scale=max(2.0, float(scale) * 0.8), octaves=2, phase=phase, loop_speed=1.0, seed=seed + 53, tileable=tileable)

    caustic = np.clip(0.55 * c1_s + 0.45 * c2_s, 0.0, 1.0)
    h = np.clip(0.30 + 0.45 * rip + 0.35 * np.power(caustic, 1.4), 0.0, 1.0).astype(np.float32)
    tone = np.clip(0.22 + 0.38 * rip + 0.50 * np.power(caustic, 1.6), 0.0, 1.0).astype(np.float32)
    rgba = multi_stop_gradient(tone, water_stops, space="oklab", smooth_blend=0.58, quantize_steps=6, dither="none")
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": rgba,
        "height": h,
        "normal": height_to_normal_map(h, strength=1.8, tileable=tileable),
        "roughness": np.full((H, W), 0.10, dtype=np.float32),
        "metallic": np.zeros((H, W), dtype=np.float32),
        "ao": np.full((H, W), 0.94, dtype=np.float32),
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
        "sss": np.full((H, W), 0.55, dtype=np.float32),
        "porosity": np.zeros((H, W), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# 2. Real-Life Animal & Imaginary Creature Biological Material Generators
# ---------------------------------------------------------------------------


def _default_mesh_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback synthetic spherical/cylindrical 3D fields so bio generators work BOTH on 3D meshes AND 2D swatches!"""
    u, v = uv_grid(height, width)
    # Gentle curved torso/limb profile so 2D material swatches look sculptural and grounded
    wpn = np.stack((u, 1.0 - v, np.full_like(u, 0.5)), axis=-1).astype(np.float32)
    nx = (u - 0.5) * 1.15
    ny = (0.5 - v) * 1.25
    nz = np.sqrt(np.clip(1.0 - np.minimum(0.95, nx * nx + ny * ny), 0.05, 1.0))
    wn = np.stack((nx, ny, nz), axis=-1).astype(np.float32)
    wn /= np.maximum(1e-6, np.linalg.norm(wn, axis=-1, keepdims=True))
    ao = np.clip(0.74 + 0.26 * (1.0 - v), 0.0, 1.0).astype(np.float32)
    cav = np.zeros((height, width), dtype=np.float32)
    luv = np.stack((u, v), axis=-1).astype(np.float32)
    return wpn, wn, ao, cav, luv


def organic_fur_pelt(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    local_uv: np.ndarray | None = None,
    coat_stops: Sequence[tuple[float, str]] = (
        (0.00, "#241918"),  # deep warm-plum espresso undercoat shadow (Everbloom Husky!)
        (0.28, "#4A362F"),  # rich dark walnut dorsal fur
        (0.58, "#7D6154"),  # warm taupe midtone tufts
        (0.84, "#B59C8D"),  # soft sunlit coat highlight
        (1.00, "#E3D6CC"),  # delicate sunlit fur tips
    ),
    belly_stops: Sequence[tuple[float, str]] = (
        (0.00, "#756761"),
        (0.35, "#A8988F"),
        (0.72, "#D5C9C0"),
        (1.00, "#F5EFEA"),
    ),
    marking_stops: Sequence[tuple[float, str]] = (
        (0.00, "#1B1416"),
        (0.50, "#3A2B29"),
        (1.00, "#685049"),
    ),
    pattern: str = "saddle",  # "solid" | "saddle" | "tabby" | "spots" | "dappled"
    tuft_scale: float = 12.0,
    countershade_height: float = 0.44,
    smooth_blend: float = 0.48,
    quantize_steps: int = 6,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Real-Life Mammal Fur & Beast Pelt Generator (Everbloom Husky / Fox / Wolf / Big Cat Standard!).

    Key craft laws (zero static speckle!):
    - Directional layered 2-4 px tufted hair locks flowing down gravity/anatomy
    - Jagged organic tuft border between dorsal saddle & pale cream underbelly/chest/legs
    - Warm-to-cool hue-shifted undercoat AO & soft velvety PBR sheen
    """
    H, W = int(height), int(width)
    if world_pos_norm is None or world_normal is None or baked_ao is None or joint_cavity is None or local_uv is None:
        wpn, wn, ao, cav, luv = _default_mesh_fields(H, W)
    else:
        wpn = np.asarray(world_pos_norm, dtype=np.float32)
        wn = np.asarray(world_normal, dtype=np.float32)
        ao = np.clip(np.asarray(baked_ao, dtype=np.float32), 0.0, 1.0)
        cav = np.clip(np.asarray(joint_cavity, dtype=np.float32), 0.0, 1.0)
        luv = np.asarray(local_uv, dtype=np.float32)

    w3 = organic_domain_warp_3d(wpn, scale=3.2, warp_strength=0.45, octaves=3, seed=seed + 7)
    # Resolution-independent atlas scaling so 64x64 swatches AND 384x256 3D entity atlases share crisp 2-4px fur clusters!
    res_mult = max(1.0, math.sqrt((H * W) / (64.0 * 64.0)))
    ts = max(4.0, float(tuft_scale) * res_mult)
    tuft_a = fbm_noise(H, W, scale=ts * 1.25, scale_y=max(2.0, ts * 0.30), octaves=2, seed=seed + 19, tileable=False)
    tuft_v = fbm_noise(H, W, scale=max(6.0, ts * 1.45), scale_y=max(2.0, ts * 0.26), octaves=2, seed=seed + 53, tileable=False)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    clump_pitch = max(5.0, min(H, W) / max(5.0, ts * 0.65))
    shingle_phase = ((yy + (tuft_a - 0.5) * clump_pitch * 2.2) / clump_pitch) % 1.0
    tuft_overlap_sh = np.clip(1.0 - shingle_phase / 0.22, 0.0, 1.0) * (0.40 + 0.60 * tuft_v) * 0.85
    tuft_tip_hi = np.clip((shingle_phase - 0.66) / 0.34, 0.0, 1.0) * (0.40 + 0.60 * tuft_a) * 0.88

    # Volumetric top-down directional lighting
    key_l = np.array([0.32, 0.86, 0.40], dtype=np.float32)
    key_l /= np.linalg.norm(key_l)
    ndotl = np.clip(np.sum(wn * key_l[None, None, :], axis=-1) * 0.5 + 0.5, 0.0, 1.0)

    # Smooth painterly vertical fur clumps
    fur_h = np.clip(0.38 + 0.32 * tuft_v + 0.16 * tuft_tip_hi - 0.20 * tuft_overlap_sh + 0.14 * (w3["mottle"] - 0.5), 0.0, 1.0).astype(np.float32)
    fur_tone = np.clip(
        0.22 + 0.42 * ndotl + 0.22 * (fur_h - 0.48) + 0.12 * (w3["mottle"] - 0.48) - 0.18 * tuft_overlap_sh - 0.30 * cav - 0.22 * (1.0 - ao),
        0.0,
        1.0,
    ).astype(np.float32)

    # Crisp Everbloom-grade vertical pixel-clustered fur locks (80% stepped clusters + 20% smooth bridge!)
    sb_fur = min(0.28, float(smooth_blend))
    coat_raw = multi_stop_gradient(fur_tone, coat_stops, space="oklab", smooth_blend=sb_fur, quantize_steps=quantize_steps, dither="none")
    coat_cl = pixel_clusters(coat_raw, [c for _, c in coat_stops], direction="vertical", cluster_size=2, clean_strays=True, clean_corners=True)
    coat_rgba = coat_cl * 0.80 + coat_raw * 0.20

    belly_raw = multi_stop_gradient(np.clip(fur_tone + 0.08, 0.0, 1.0), belly_stops, space="oklab", smooth_blend=sb_fur, quantize_steps=quantize_steps, dither="none")
    belly_cl = pixel_clusters(belly_raw, [c for _, c in belly_stops], direction="vertical", cluster_size=2, clean_strays=True, clean_corners=True)
    belly_rgba = belly_cl * 0.80 + belly_raw * 0.20

    # Jagged fur-tuft countershading boundary (Everbloom Husky signature: crisp stepped cream belly, chest & paws!)
    elev = wpn[..., 1] * 0.72 + (wn[..., 1] * 0.5 + 0.5) * 0.28
    jagged_edge = (tuft_v - 0.5) * 0.24 + (w3["flow"] - 0.5) * 0.14
    belly_mask = np.clip((float(countershade_height) - (elev + jagged_edge)) / 0.055, 0.0, 1.0).astype(np.float32)

    rgb = coat_rgba[..., :3] * (1.0 - belly_mask[..., None]) + belly_rgba[..., :3] * belly_mask[..., None]

    # Species coat markings (tabby stripes, organic snow-leopard rosettes, dorsal saddle, or dappled!)
    pat = pattern.lower()
    mark_m = np.zeros((H, W), dtype=np.float32)
    if pat == "tabby":
        stripe_coord = wpn[..., 2] * 7.5 + wpn[..., 0] * 3.0 + (w3["warp_x"] * 0.42)
        s_wave = 0.5 + 0.5 * np.sin(stripe_coord * math.pi * 2.0 + tuft_a * 1.4)
        mark_m = histogram_scan(s_wave, position=0.38, contrast=0.58) * (1.0 - belly_mask * 0.85)
    elif pat == "spots":
        # Authentic soft-edged big-cat rosettes (hollow broken ring with softer taupe inner fill, NEVER hard black polka dots!)
        sp = voronoi_noise(H, W, scale=max(3.5, ts * 0.42), jitter=0.84, output="f1", seed=seed + 83, tileable=False)
        sp_w = np.clip(sp + (w3["warp_x"] * 0.09) + (tuft_v - 0.5) * 0.12, 0.0, 1.0)
        ring = np.exp(-(((sp_w - 0.24) / 0.095) ** 2)) * (0.55 + 0.45 * tuft_a)
        inner = np.clip(1.0 - sp_w / 0.22, 0.0, 1.0) * 0.36
        mark_m = np.clip(np.maximum(ring * 0.78, inner), 0.0, 0.82) * (1.0 - belly_mask * 0.85)
    elif pat == "dappled":
        mark_m = histogram_scan(w3["mottle"] * (0.7 + 0.5 * tuft_a), position=0.45, contrast=0.52) * (1.0 - belly_mask * 0.7)
    elif pat == "saddle":
        dorsal_band = np.clip((elev + jagged_edge - (float(countershade_height) + 0.12)) / 0.14, 0.0, 1.0)
        mark_m = dorsal_band * 0.55

    if float(mark_m.max()) > 0.01:
        mark_raw = multi_stop_gradient(fur_tone, marking_stops, space="oklab", smooth_blend=sb_fur, quantize_steps=quantize_steps, dither="none")
        mark_cl = pixel_clusters(mark_raw, [c for _, c in marking_stops], direction="vertical", cluster_size=2, clean_strays=True, clean_corners=True)
        mark_rgba = mark_cl * 0.80 + mark_raw * 0.20
        rgb = rgb * (1.0 - mark_m[..., None]) + mark_rgba[..., :3] * mark_m[..., None]

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": fur_h,
        "normal": height_to_normal_map(fur_h, strength=1.9, tileable=False),
        "roughness": np.clip(0.80 + 0.12 * tuft_overlap_sh - 0.10 * tuft_tip_hi, 0.58, 0.96).astype(np.float32),
        "metallic": np.zeros((H, W), dtype=np.float32),
        "ao": np.clip(ao * (1.0 - 0.28 * tuft_overlap_sh) * (1.0 - 0.25 * cav), 0.16, 1.0).astype(np.float32),
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
        "sss": np.full((H, W), 0.16, dtype=np.float32),
        "porosity": np.full((H, W), 0.45, dtype=np.float32),
        "belly_mask": belly_mask,
    }


def layered_plumage(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    feather_stops: Sequence[tuple[float, str]] = (
        (0.00, "#2B2834"),  # cool slate-indigo under-feather shadow
        (0.28, "#6E656C"),  # soft warm taupe-gray covert
        (0.58, "#B4A89F"),  # luminous owl/eagle mid plumage
        (0.84, "#E2D9CE"),  # sunlit creamy feather tip
        (1.00, "#FAF6F0"),  # crisp quill & rim catchlight
    ),
    accent_stops: Sequence[tuple[float, str]] = (
        (0.00, "#1C1826"),
        (0.55, "#4E3D3A"),
        (1.00, "#9E7A56"),
    ),
    cols: int = 8,
    rows: int = 10,
    marking: str = "barred",  # "solid" | "barred" | "speckled" | "iridescent"
    iridescence: float = 0.18,
    smooth_blend: float = 0.50,
    quantize_steps: int = 6,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Avian Plumage & Feathered Creature MaterialBus (Owls, Eagles, Parrots, Griffins, Phoenixes!).

    Generates staggered overlapping U-scallop feathers with a bright central rachis quill,
    diagonal micro-barb shimmer, dark under-overlap cast shadow, and optional barred/speckled/iridescent tips!
    """
    H, W = int(height), int(width)
    if world_pos_norm is None or world_normal is None or baked_ao is None or joint_cavity is None:
        wpn, wn, ao, cav, _ = _default_mesh_fields(H, W)
    else:
        wpn = np.asarray(world_pos_norm, dtype=np.float32)
        wn = np.asarray(world_normal, dtype=np.float32)
        ao = np.clip(np.asarray(baked_ao, dtype=np.float32), 0.0, 1.0)
        cav = np.clip(np.asarray(joint_cavity, dtype=np.float32), 0.0, 1.0)

    w3 = organic_domain_warp_3d(wpn, scale=3.0, warp_strength=0.32, octaves=2, seed=seed + 11)
    u, v = uv_grid(H, W)
    # Organic feather pitch with gentle distortion
    gx_b = u * max(3.0, float(cols)) + w3["warp_x"] * 0.24
    gy_b = v * max(3.0, float(rows)) + w3["warp_y"] * 0.24

    row0 = np.floor(gy_b).astype(np.int32)
    fy0 = gy_b - row0.astype(np.float32)
    gx0 = gx_b + (row0 & 1).astype(np.float32) * 0.5
    col0 = np.floor(gx0).astype(np.int32)
    fx0 = gx0 - col0.astype(np.float32)

    row_u = row0 - 1
    gx_u = gx_b + (row_u & 1).astype(np.float32) * 0.5
    col_u = np.floor(gx_u).astype(np.int32)
    fx_u = gx_u - col_u.astype(np.float32)

    dx_u = 2.0 * np.abs(fx_u - 0.5)
    arch_y = (0.48 * (1.0 - dx_u ** 1.85)).astype(np.float32)
    in_up = fy0 < arch_y

    sx = np.where(in_up, (fx_u - 0.5) * 2.0, (fx0 - 0.5) * 2.0).astype(np.float32)
    sy = np.where(in_up, np.clip((1.0 + fy0) / 1.48, 0.0, 1.0), np.clip((fy0 - arch_y) / np.maximum(0.25, 1.0 + 0.48 * (1.0 - np.abs(sx) ** 1.85) - arch_y), 0.0, 1.0)).astype(np.float32)
    sid_x = np.where(in_up, col_u, col0)
    sid_y = np.where(in_up, row_u, row0)

    # Soft velvety under-feather cast shadow + subtle warm central quill + calm feather fan body (>=68% calm!)
    dist_sh = fy0 - arch_y
    overlap_sh = np.where((~in_up) & (dist_sh >= 0.0) & (dist_sh < 0.28), np.clip(1.0 - dist_sh / 0.28, 0.0, 1.0) ** 1.45, 0.0).astype(np.float32) * 0.65
    quill = np.exp(-((sx / 0.24) ** 2)) * np.clip(1.0 - sy * 0.65, 0.15, 1.0) * 0.55
    fan_dome = np.clip(1.0 - 0.35 * (sx * sx) - 0.22 * ((sy - 0.55) ** 2), 0.35, 1.0).astype(np.float32)
    tip_rim = np.clip((sy - 0.68) / 0.32, 0.0, 1.0) * np.clip(1.0 - np.abs(sx) * 0.55, 0.0, 1.0)

    h_int = (sid_x.astype(np.uint32) * np.uint32(73856093)) ^ (sid_y.astype(np.uint32) * np.uint32(19349663)) ^ np.uint32(seed * 1597334677 & 0xFFFFFFFF)
    f_var = ((h_int & np.uint32(0xFFFF)).astype(np.float32) / 65535.0 - 0.5) * 0.12

    key_l = np.array([0.32, 0.86, 0.40], dtype=np.float32)
    key_l /= np.linalg.norm(key_l)
    ndotl = np.clip(np.sum(wn * key_l[None, None, :], axis=-1) * 0.5 + 0.5, 0.0, 1.0)

    f_h = np.clip(0.36 + 0.28 * fan_dome + 0.12 * quill + 0.12 * tip_rim - 0.30 * overlap_sh + 0.08 * (w3["mottle"] - 0.5), 0.0, 1.0).astype(np.float32)
    # Ray-marched roof-tile feather cast shadow (Pixel Composer 1.22.4 Heightmap Shadow -> crisp overlapping scallops!)
    hm_sh = heightmap_shadow(f_h, light_angle_deg=-115.0, elevation_deg=40.0, max_steps=3, strength=0.58, tileable=False)
    tone = np.clip(
        0.22 + 0.32 * ndotl + 0.24 * (f_h - 0.44) + f_var + 0.14 * tip_rim + 0.08 * quill - 0.26 * overlap_sh - 0.22 * (1.0 - hm_sh) - 0.24 * cav - 0.16 * (1.0 - ao),
        0.0,
        1.0,
    ).astype(np.float32)

    sb_pl = min(0.28, float(smooth_blend))
    f_pal_hex = [c for _, c in feather_stops]
    base_raw = multi_stop_gradient(tone, feather_stops, space="oklab", smooth_blend=sb_pl, quantize_steps=max(5, quantize_steps), dither="none")
    base_cl = pixel_clusters(base_raw, f_pal_hex, direction="vertical", cluster_size=2, clean_strays=True, clean_corners=True)
    rgb = (base_cl[..., :3] * 0.82 + base_raw[..., :3] * 0.18).astype(np.float32)

    mk = marking.lower()
    if mk == "barred":
        # Crisp single painterly crescent band across mid-tip of each feather (zero harsh zebra stripes!)
        bar = np.exp(-(((sy - 0.58) / 0.16) ** 2)) * np.clip(1.0 - np.abs(sx) * 0.45, 0.2, 1.0) * 0.58
        acc_raw = multi_stop_gradient(tone, accent_stops, space="oklab", smooth_blend=sb_pl, quantize_steps=quantize_steps, dither="none")
        acc_cl = pixel_clusters(acc_raw, [c for _, c in accent_stops], direction="horizontal", cluster_size=2, clean_strays=True, clean_corners=True)
        rgb = rgb * (1.0 - bar[..., None]) + (acc_cl[..., :3] * 0.82 + acc_raw[..., :3] * 0.18) * bar[..., None]
    elif mk == "speckled":
        dot = np.exp(-(((sx * 1.5) ** 2 + ((sy - 0.64) * 2.1) ** 2) / 0.20)) * 0.64
        acc_raw = multi_stop_gradient(tone, accent_stops, space="oklab", smooth_blend=sb_pl, quantize_steps=quantize_steps, dither="none")
        acc_cl = pixel_clusters(acc_raw, [c for _, c in accent_stops], direction="vertical", cluster_size=2, clean_strays=True, clean_corners=True)
        rgb = rgb * (1.0 - dot[..., None]) + (acc_cl[..., :3] * 0.82 + acc_raw[..., :3] * 0.18) * dot[..., None]
    elif mk == "solid":
        # Even pale/solid breast plumage gets crisp dark scalloped feather-tip arches so chest panels NEVER look blank/flat!
        scallop_rim = (overlap_sh > 0.26).astype(np.float32) * 0.42
        c_deep = np.array(hex_to_srgb(f_pal_hex[0]), dtype=np.float32)[None, None, :]
        rgb = rgb * (1.0 - scallop_rim[..., None]) + c_deep * scallop_rim[..., None]

    ir = float(np.clip(iridescence, 0.0, 1.0))
    if ir > 0.01 or mk == "iridescent":
        shift_rgba = oklch_adjust(np.concatenate((rgb, np.ones((H, W, 1), dtype=np.float32)), axis=-1), hue_shift_deg=34.0, chroma_scale=1.22, lightness_offset=0.03)
        ir_w = np.clip(tip_rim * (0.45 + 0.55 * ndotl) * max(ir, 0.45 if mk == "iridescent" else 0.0), 0.0, 0.85)[..., None]
        rgb = rgb * (1.0 - ir_w) + shift_rgba[..., :3] * ir_w

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": f_h,
        "normal": height_to_normal_map(f_h, strength=2.2, tileable=False),
        "roughness": np.clip(0.56 - 0.16 * tip_rim - 0.10 * quill + 0.18 * overlap_sh, 0.24, 0.90).astype(np.float32),
        "metallic": np.clip(ir * 0.45 * tip_rim, 0.0, 1.0).astype(np.float32),
        "ao": np.clip(ao * (1.0 - 0.35 * overlap_sh) * (1.0 - 0.25 * cav), 0.16, 1.0).astype(np.float32),
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
        "sss": np.clip(0.24 * tip_rim, 0.0, 1.0).astype(np.float32),
        "porosity": np.full((H, W), 0.18, dtype=np.float32),
    }


def chitin_carapace(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    chitin_stops: Sequence[tuple[float, str]] = (
        (0.00, "#1E1822"),  # Mickey Joe cool plum-black crevice (#322b2e family!)
        (0.26, "#382D35"),  # smoked aubergine-slate shadow chitin
        (0.54, "#5E4C4B"),  # warm dark earth-bronze armored carapace
        (0.80, "#8E7364"),  # sunlit weathered ochre chitin dome
        (1.00, "#CBB299"),  # crisp glossy chitin specular ridge
    ),
    segments: int = 6,
    iridescence: float = 0.26,
    glossiness: float = 0.72,
    smooth_blend: float = 0.56,
    quantize_steps: int = 6,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Segmented Insect / Scorpion / Beetle / Void Arthropod Chitin Carapace (`MATERIAL` Bus).

    Benchmarked directly against Mickey Joe's Scorpion tutorial (`step_57_shot_1.png`):
    curved overlapping dorsal armor plates, cool plum-violet recesses, subtle dimpled pores,
    and glossy iridescent specular ridges!
    """
    H, W = int(height), int(width)
    if world_pos_norm is None or world_normal is None or baked_ao is None or joint_cavity is None:
        wpn, wn, ao, cav, _ = _default_mesh_fields(H, W)
    else:
        wpn = np.asarray(world_pos_norm, dtype=np.float32)
        wn = np.asarray(world_normal, dtype=np.float32)
        ao = np.clip(np.asarray(baked_ao, dtype=np.float32), 0.0, 1.0)
        cav = np.clip(np.asarray(joint_cavity, dtype=np.float32), 0.0, 1.0)

    w3 = organic_domain_warp_3d(wpn, scale=2.8, warp_strength=0.35, octaves=2, seed=seed + 13)
    u, v = uv_grid(H, W)
    # Resolution-independent chitin plate pitch so both 64x64 swatches & 272px insect atlases show crisp armored segments!
    res_mult_c = max(1.0, H / 64.0)
    eff_segs = max(2.0, float(segments) * (0.55 + 0.45 * res_mult_c))
    seg_coord = v * eff_segs + 0.16 * np.cos((u - 0.5) * math.pi * 2.0) + 0.12 * w3["warp_y"]
    seg_f = seg_coord % 1.0
    seg_seam = np.clip(1.0 - seg_f / 0.18, 0.0, 1.0)
    seg_rim = np.clip((seg_f - 0.76) / 0.24, 0.0, 1.0) * (1.0 - seg_seam)
    seg_dome = np.sin(np.clip(seg_f, 0.0, 1.0) * math.pi) ** 0.75

    # Subtle chitin micro-dimpling + organic studio brushwork scaled to atlas resolution
    dimple = fbm_noise(H, W, scale=14.0 * max(1.0, math.sqrt((H * W) / 4096.0)), octaves=2, seed=seed + 41, tileable=False)
    ridge = np.exp(-(((u - 0.5) * 2.2) ** 2)) * 0.24

    key_l = np.array([0.34, 0.85, 0.42], dtype=np.float32)
    key_l /= np.linalg.norm(key_l)
    ndotl = np.clip(np.sum(wn * key_l[None, None, :], axis=-1) * 0.5 + 0.5, 0.0, 1.0)

    ch_h = np.clip(0.32 + 0.44 * seg_dome + 0.14 * seg_rim + 0.12 * ridge + 0.06 * (dimple - 0.5) - 0.38 * seg_seam, 0.0, 1.0).astype(np.float32)
    # Mickey Joe Scorpion roof-tile ray-marched shadow under each overlapping dorsal segment!
    ch_sh = heightmap_shadow(ch_h, light_angle_deg=-120.0, elevation_deg=38.0, max_steps=4, strength=0.68, tileable=False)
    tone = np.clip(
        0.18 + 0.38 * ndotl + 0.26 * (ch_h - 0.45) + 0.18 * seg_rim * ndotl + 0.10 * (w3["mottle"] - 0.5) - 0.34 * seg_seam - 0.22 * (1.0 - ch_sh) - 0.28 * cav - 0.20 * (1.0 - ao),
        0.0,
        1.0,
    ).astype(np.float32)

    sb_ch = min(0.28, float(smooth_blend))
    ch_hex = [c for _, c in chitin_stops]
    base_raw = multi_stop_gradient(tone, chitin_stops, space="oklab", smooth_blend=sb_ch, quantize_steps=quantize_steps, dither="none")
    base_cl = pixel_clusters(base_raw, ch_hex, direction="horizontal", cluster_size=2, clean_strays=True, clean_corners=True)
    base_rgba = base_cl * 0.82 + base_raw * 0.18
    rgb = base_rgba[..., :3]

    ir = float(np.clip(iridescence, 0.0, 1.0))
    if ir > 0.01:
        fresnel = np.power(np.clip(1.0 - np.abs(wn[..., 2]), 0.0, 1.0), 1.8)
        ir_rgba = oklch_adjust(base_rgba, hue_shift_deg=-42.0, chroma_scale=1.35, lightness_offset=0.04)
        w_ir = np.clip((fresnel * 0.6 + seg_rim * 0.4) * ir, 0.0, 0.72)[..., None]
        rgb = rgb * (1.0 - w_ir) + ir_rgba[..., :3] * w_ir

    gl = float(np.clip(glossiness, 0.1, 0.95))
    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": ch_h,
        "normal": height_to_normal_map(ch_h, strength=2.6, tileable=False),
        "roughness": np.clip((1.0 - gl) + 0.38 * seg_seam - 0.14 * seg_rim, 0.12, 0.90).astype(np.float32),
        "metallic": np.clip(0.32 * (0.4 + 0.6 * ir) * (1.0 - seg_seam), 0.0, 1.0).astype(np.float32),
        "ao": np.clip(ao * (1.0 - 0.42 * seg_seam) * (1.0 - 0.28 * cav), 0.14, 1.0).astype(np.float32),
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
        "sss": np.full((H, W), 0.06, dtype=np.float32),
        "porosity": np.full((H, W), 0.05, dtype=np.float32),
    }


def aquatic_scales_skin(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    base_stops: Sequence[tuple[float, str]] = (
        (0.00, "#0A284B"),
        (0.28, "#0E6491"),
        (0.58, "#1FB3C8"),
        (0.84, "#7CE8E2"),
        (1.00, "#E4FFF9"),
    ),
    spot_stops: Sequence[tuple[float, str]] = (
        (0.00, "#7C1D18"),
        (0.45, "#EA580C"),
        (0.82, "#FACC15"),
        (1.00, "#FEF9C3"),
    ),
    mode: str = "fish_scales",  # "fish_scales" | "amphibian_moist" | "koi_calico"
    scale_cols: int = 10,
    scale_rows: int = 12,
    wetness: float = 0.82,
    smooth_blend: float = 0.62,
    quantize_steps: int = 7,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Aquatic Fish Scales & Moist Amphibian Skin (`MATERIAL` Bus — Koi, Reef Fish, Dart Frogs, Leviathans!)."""
    H, W = int(height), int(width)
    if world_pos_norm is None or world_normal is None or baked_ao is None or joint_cavity is None:
        wpn, wn, ao, cav, _ = _default_mesh_fields(H, W)
    else:
        wpn = np.asarray(world_pos_norm, dtype=np.float32)
        wn = np.asarray(world_normal, dtype=np.float32)
        ao = np.clip(np.asarray(baked_ao, dtype=np.float32), 0.0, 1.0)
        cav = np.clip(np.asarray(joint_cavity, dtype=np.float32), 0.0, 1.0)

    w3 = organic_domain_warp_3d(wpn, scale=3.0, warp_strength=0.40, octaves=3, seed=seed + 17)
    u, v = uv_grid(H, W)
    key_l = np.array([0.30, 0.88, 0.38], dtype=np.float32)
    key_l /= np.linalg.norm(key_l)
    ndotl = np.clip(np.sum(wn * key_l[None, None, :], axis=-1) * 0.5 + 0.5, 0.0, 1.0)
    wet = float(np.clip(wetness, 0.1, 1.0))

    md = mode.lower()
    if md == "fish_scales":
        # Shimmering overlapping cycloid fish scales + iridescent lateral line!
        gx = u * max(4.0, float(scale_cols)) + w3["warp_x"] * 0.18
        gy = v * max(4.0, float(scale_rows)) + w3["warp_y"] * 0.18
        r0 = np.floor(gy).astype(np.int32)
        fy = gy - r0.astype(np.float32)
        fx = (gx + (r0 & 1).astype(np.float32) * 0.5) % 1.0
        dist_c = np.sqrt(((fx - 0.5) * 1.8) ** 2 + ((fy - 0.35) * 1.4) ** 2)
        sc_rim = np.clip(1.0 - np.abs(dist_c - 0.72) / 0.18, 0.0, 1.0)
        sc_dome = np.clip(1.0 - dist_c * 0.65, 0.0, 1.0)
        lat_line = np.exp(-(((v - 0.48 + 0.05 * w3["warp_y"]) / 0.07) ** 2))
        surf_h = np.clip(0.35 + 0.42 * sc_dome + 0.18 * sc_rim, 0.0, 1.0).astype(np.float32)
        sc_sh = heightmap_shadow(surf_h, light_angle_deg=-130.0, elevation_deg=42.0, max_steps=3, strength=0.55, tileable=False)
        tone = np.clip(0.18 + 0.42 * ndotl + 0.24 * (surf_h - 0.48) + 0.18 * lat_line + 0.14 * sc_rim * ndotl - 0.18 * (1.0 - sc_sh) - 0.24 * cav, 0.0, 1.0).astype(np.float32)
        sb_aq = min(0.30, float(smooth_blend))
        base_raw = multi_stop_gradient(tone, base_stops, space="oklab", smooth_blend=sb_aq, quantize_steps=quantize_steps, dither="none")
        base_cl = pixel_clusters(base_raw, [c for _, c in base_stops], direction="diag_pos", cluster_size=2, clean_strays=True, clean_corners=True)
        rgb = (base_cl[..., :3] * 0.80 + base_raw[..., :3] * 0.20).astype(np.float32)
        if float(lat_line.max()) > 0.1:
            accent_raw = multi_stop_gradient(tone, spot_stops, space="oklab", smooth_blend=sb_aq, quantize_steps=quantize_steps, dither="none")
            accent_cl = pixel_clusters(accent_raw, [c for _, c in spot_stops], direction="diag_pos", cluster_size=2, clean_strays=True, clean_corners=True)
            rgb = rgb * (1.0 - lat_line[..., None] * 0.48) + (accent_cl[..., :3] * 0.80 + accent_raw[..., :3] * 0.20) * (lat_line[..., None] * 0.48)
    else:
        # Smooth moist amphibian/koi skin with resolution-independent organic blotches & crisp pixel clusters!
        res_m_a = max(1.0, math.sqrt((H * W) / 4096.0))
        pores = fbm_noise(H, W, scale=12.0 * res_m_a, octaves=2, seed=seed + 53, tileable=False)
        blotch_n = fbm_noise(H, W, scale=4.2 * res_m_a, octaves=2, seed=seed + 71, tileable=False)
        if md == "koi_calico":
            patch_m = histogram_scan(0.65 * w3["mottle"] + 0.35 * blotch_n, position=0.46, contrast=0.72)
        else:
            # Domain-warped organic irregular poison-dart-frog blotches (NEVER rigid circles!)
            sp = voronoi_noise(H, W, scale=4.2 * res_m_a, jitter=0.85, output="f1", seed=seed + 71, tileable=False)
            sp_w = np.clip(sp + 0.14 * w3["warp_x"] + 0.10 * (blotch_n - 0.5), 0.0, 1.0)
            patch_m = histogram_scan(np.clip(1.0 - sp_w / 0.42, 0.0, 1.0), position=0.50, contrast=0.72)
        surf_h = np.clip(0.42 + 0.24 * w3["mottle"] + 0.18 * patch_m + 0.14 * (pores - 0.5), 0.0, 1.0).astype(np.float32)
        tone = np.clip(0.20 + 0.46 * ndotl + 0.16 * (w3["mottle"] - 0.5) + 0.12 * (pores - 0.5) + 0.18 * (ndotl ** 4) * wet - 0.25 * cav, 0.0, 1.0).astype(np.float32)
        base_raw = multi_stop_gradient(tone, base_stops, space="oklab", smooth_blend=0.28, quantize_steps=6, dither="none")
        base_cl = pixel_clusters(base_raw, [c for _, c in base_stops], direction="omni", cluster_size=2, clean_strays=True, clean_corners=True)
        spot_raw = multi_stop_gradient(tone, spot_stops, space="oklab", smooth_blend=0.28, quantize_steps=6, dither="none")
        spot_cl = pixel_clusters(spot_raw, [c for _, c in spot_stops], direction="omni", cluster_size=2, clean_strays=True, clean_corners=True)
        rgb = (base_cl[..., :3] * 0.78 + base_raw[..., :3] * 0.22) * (1.0 - patch_m[..., None]) + (spot_cl[..., :3] * 0.78 + spot_raw[..., :3] * 0.22) * patch_m[..., None]

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": surf_h,
        "normal": height_to_normal_map(surf_h, strength=1.8, tileable=False),
        "roughness": np.clip(0.42 - 0.28 * wet + 0.10 * (1.0 - surf_h), 0.10, 0.65).astype(np.float32),
        "metallic": np.clip(0.28 if md == "fish_scales" else 0.04, 0.0, 1.0) * np.ones((H, W), dtype=np.float32),
        "ao": np.clip(ao * (1.0 - 0.22 * cav), 0.18, 1.0).astype(np.float32),
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
        "sss": np.full((H, W), 0.34, dtype=np.float32),
        "porosity": np.full((H, W), 0.08, dtype=np.float32),
    }


def elemental_golem_surface(
    height: int,
    width: int,
    *,
    world_pos_norm: np.ndarray | None = None,
    world_normal: np.ndarray | None = None,
    baked_ao: np.ndarray | None = None,
    joint_cavity: np.ndarray | None = None,
    rock_stops: Sequence[tuple[float, str]] = (
        (0.00, "#151320"),
        (0.28, "#2C2B3D"),
        (0.58, "#4E5066"),
        (0.84, "#7B829E"),
        (1.00, "#BAC6DE"),
    ),
    fissure_stops: Sequence[tuple[float, str]] = (
        (0.00, "#7F1D1D"),
        (0.38, "#EA580C"),
        (0.74, "#FACC15"),
        (1.00, "#FFFBEB"),
    ),
    slab_scale: float = 4.5,
    glow_strength: float = 0.85,
    smooth_blend: float = 0.48,
    quantize_steps: int = 6,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Fantasy Construct / Elemental Golem MaterialBus (`Flood Fill + Slope Blur + Glowing Crevice Veins`)."""
    H, W = int(height), int(width)
    if world_pos_norm is None or world_normal is None or baked_ao is None or joint_cavity is None:
        wpn, wn, ao, cav, _ = _default_mesh_fields(H, W)
    else:
        wpn = np.asarray(world_pos_norm, dtype=np.float32)
        wn = np.asarray(world_normal, dtype=np.float32)
        ao = np.clip(np.asarray(baked_ao, dtype=np.float32), 0.0, 1.0)
        cav = np.clip(np.asarray(joint_cavity, dtype=np.float32), 0.0, 1.0)

    # Fractured tectonic rock slabs via TRUE Voronoi F2-F1 + Flood Fill + Slope Blur chipping!
    f2m1 = voronoi_noise(H, W, scale=max(2.0, float(slab_scale)), jitter=0.72, output="f2_minus_f1", seed=seed, tileable=False)
    slab_m = (f2m1 > 0.082).astype(np.float32)
    ff = flood_fill(slab_m, threshold=0.5, tileable=False, seed=seed + 19)
    chip_n = fbm_noise(H, W, scale=max(3.0, float(slab_scale) * 1.6), octaves=2, seed=seed + 41, tileable=False)

    dome = np.clip((f2m1 - 0.082) / 0.26, 0.0, 1.0) ** 0.55
    raw_h = np.clip(slab_m * (0.36 + 0.34 * dome + 0.20 * (ff["random_gray"] - 0.5) + 0.16 * (ff["random_slope"] - 0.5)), 0.0, 1.0).astype(np.float32)
    rock_h = slope_blur(raw_h, chip_n, intensity=0.85, samples=3, mode="min", tileable=False)
    ew = edge_wear_mask(rock_h, wear_amount=0.44, chip_scale=float(slab_scale) * 1.8, grime_amount=0.52, tileable=False, seed=seed + 61)

    key_l = np.array([0.32, 0.85, 0.42], dtype=np.float32)
    key_l /= np.linalg.norm(key_l)
    ndotl = np.clip(np.sum(wn * key_l[None, None, :], axis=-1) * 0.5 + 0.5, 0.0, 1.0)
    rock_sh = heightmap_shadow(rock_h, light_angle_deg=-135.0, elevation_deg=36.0, max_steps=4, strength=0.70, tileable=False)
    tone = np.clip(0.20 + 0.38 * ndotl + 0.30 * (rock_h - 0.40) + 0.26 * ew["wear"] - 0.22 * (1.0 - rock_sh) - 0.26 * cav - 0.18 * (1.0 - ao), 0.0, 1.0).astype(np.float32)
    rock_raw = multi_stop_gradient(tone, rock_stops, space="oklab", smooth_blend=min(0.26, float(smooth_blend)), quantize_steps=quantize_steps, dither="none")
    rock_cl = pixel_clusters(rock_raw, [c for _, c in rock_stops], direction="omni", cluster_size=2, clean_strays=True, clean_corners=True)
    rock_rgba = rock_cl * 0.82 + rock_raw * 0.18

    # Focused glowing elemental fissures strictly tucked inside deep rock crevices!
    cracks = np.clip((0.105 - f2m1) / 0.105, 0.0, 1.0) * (0.55 + 0.45 * chip_n)
    glow_m = np.clip(np.power(cracks, 1.15) * float(glow_strength), 0.0, 1.0).astype(np.float32)
    fiss_rgba = multi_stop_gradient(np.clip(0.25 + 0.75 * glow_m, 0.0, 1.0), fissure_stops, space="oklab", smooth_blend=0.88, quantize_steps=6, dither="none")

    rgb = rock_rgba[..., :3] * (1.0 - glow_m[..., None]) + fiss_rgba[..., :3] * glow_m[..., None]
    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    emissive = np.concatenate((fiss_rgba[..., :3] * glow_m[..., None], (glow_m > 0.08).astype(np.float32)[..., None]), axis=-1).astype(np.float32)

    return {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": albedo,
        "height": rock_h,
        "normal": height_to_normal_map(rock_h, strength=2.9, tileable=False),
        "roughness": np.clip(0.74 - 0.18 * ew["wear"] - 0.45 * glow_m, 0.16, 0.94).astype(np.float32),
        "metallic": np.clip(0.22 * ew["wear"] * (1.0 - glow_m), 0.0, 1.0).astype(np.float32),
        "ao": np.clip(ao * (0.45 + 0.55 * slab_m) * (1.0 - 0.25 * cav), 0.16, 1.0).astype(np.float32),
        "emissive": emissive,
        "sss": (glow_m * 0.40).astype(np.float32),
        "porosity": np.clip(0.45 * (1.0 - glow_m), 0.0, 1.0).astype(np.float32),
        "fissure_mask": glow_m,
    }
