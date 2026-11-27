"""Industry-Standard PBR Material Library, Material Bus Sequencer & Dynamic FX (`TexGraph v13`).

Builds a full commercial Substance Designer / Unreal / Unity PBR material palette:
- **Metals (5):** `mat_brass`, `mat_bronze_copper`, `mat_gold_precious`, `mat_steel_iron`, `mat_chrome_titanium`
- **Dielectrics & Translucent Glass/Crystal (7):** `mat_plastic`, `mat_painted_enamel`,
  `mat_ceramic_porcelain`, `mat_rubber_grip`, `mat_glass`, `mat_crystal_gem`, `mat_ice_frost`
- **Painterly & Bio PBR (5):** `mat_dragon_scales_pbr`, `mat_marble_veined`,
  `mat_obsidian_volcanic`, `mat_velvet_silk`, `mat_glowing_runes_pbr`
- **Material Bus Sequencer & Normal Ops (8):** `material_split`, `material_tweak`,
  `material_region_apply`, `material_weathering`, `normal_from_height_multi`,
  `normal_bevel`, `normal_to_curvature`, `normal_strength`
- **Dynamic Animation FX (3):** `anim_shimmer`, `anim_flicker_pulse`, `anim_flow_warp`
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .generators_noise import fbm_noise, gabor_noise, sample_field_bilinear, uv_grid, value_noise, voronoi_noise
from .generators_noise_ext import (
    caustic_noise,
    crystal_noise,
    fold_marble_noise,
    grunge_blotch,
    scratch_noise,
    strand_noise,
)
from .generators_patterns_ext import circuit_runes, crystal_facets, polka_rivets, scallop_shingles, weave_chainmail
from .mesh_baker import MeshBakeContext
from .mesh_mask_ops import _box_blur_2d, mask_smart_crevice_dirt, mask_smart_edge_wear, mask_world_direction
from .ops_fx_color import ao_from_height, normal_blend_rnm, oklch_adjust
from .pbr import height_to_normal_map
from .pipeline import _ensure_bus, _to_scalar_2d, is_material_bus, material_blend, material_height_blend, material_make
from .pixel_art_ops import hex_to_srgb
from .region_ops import resolve_part_mask


def _hex_rgb(hx: str, fallback: tuple[float, float, float] = (0.7, 0.7, 0.7)) -> np.ndarray:
    try:
        r, g, b = hex_to_srgb(str(hx))
        return np.array([r, g, b], dtype=np.float32)
    except Exception:
        return np.array(fallback, dtype=np.float32)


def _tint_albedo(
    val: np.ndarray,
    base_hex: str,
    *,
    dark_mul: float = 0.52,
    bright_mul: float = 1.24,
    alpha: np.ndarray | float = 1.0,
) -> np.ndarray:
    rgb0 = _hex_rgb(base_hex)
    v = np.clip(val, 0.0, 1.0)[..., None]
    mul = dark_mul + (bright_mul - dark_mul) * v
    rgb = np.clip(rgb0[None, None, :] * mul, 0.0, 1.0)
    if isinstance(alpha, (int, float)):
        a = np.full(val.shape + (1,), float(np.clip(alpha, 0.0, 1.0)), dtype=np.float32)
    else:
        a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)[..., None]
    return np.concatenate((rgb, a), axis=-1).astype(np.float32)


# ===========================================================================
# 1. METALS (PBR)
# ===========================================================================


def mat_brass(
    height: int,
    width: int,
    *,
    base_color: str = "#D4A84A",
    patina_color: str = "#4D5D48",
    roughness: float = 0.26,
    brushed_strength: float = 0.55,
    patina_amount: float = 0.22,
    wear_amount: float = 0.30,
    scratches: float = 0.35,
    brushed_angle: float = 0.0,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Physically calibrated Brushed / Aged Brass PBR MATERIAL (metallic ~0.92, brushed anisotropy, patina & scratches)."""
    grain = gabor_noise(height, width, scale=14.0, angle_deg=brushed_angle, anisotropy=4.5, octaves=3, seed=seed + 11, tileable=tileable)
    macro = fbm_noise(height, width, scale=4.0, octaves=3, seed=seed + 37, tileable=tileable)
    scr = scratch_noise(height, width, count=int(18 + 55 * scratches), length=0.25, angle_deg=brushed_angle - 22.0, angle_jitter=32.0, seed=seed + 71, tileable=tileable)
    pat_n = grunge_blotch(height, width, scale=5.0, contrast=1.55, speckle=0.35, seed=seed + 113, tileable=tileable)

    pat_m = np.clip((pat_n - (1.0 - float(patina_amount) * 1.15)) * 2.6, 0.0, 1.0).astype(np.float32)
    bs = float(np.clip(brushed_strength, 0.0, 1.0))
    h = np.clip(0.54 + (grain - 0.5) * 0.24 * bs + (macro - 0.5) * 0.14 - scr * 0.22 * float(scratches) - pat_m * 0.14, 0.0, 1.0)

    alb = _tint_albedo(0.50 + (grain - 0.5) * 0.22 * bs + (macro - 0.5) * 0.22 + scr * 0.26 * float(wear_amount), base_color, dark_mul=0.58, bright_mul=1.22)
    p_rgb = _hex_rgb(patina_color)
    alb[..., :3] = alb[..., :3] * (1.0 - pat_m[..., None] * 0.85) + p_rgb[None, None, :] * (pat_m[..., None] * 0.85)

    r_base = float(np.clip(roughness, 0.06, 0.92))
    rough = np.clip(r_base + (1.0 - grain) * 0.10 * bs + pat_m * 0.44 - scr * 0.10 * float(scratches), 0.06, 0.96)
    metal = np.clip(0.94 - pat_m * 0.72, 0.08, 1.0)
    ao = np.clip(1.0 - pat_m * 0.24 - scr * 0.10, 0.45, 1.0)
    return material_make(alb, height=h, roughness=rough, metallic=metal, ao=ao, normal_strength=2.2, tileable=tileable)


def mat_bronze_copper(
    height: int,
    width: int,
    *,
    metal_preset: str = "bronze",
    verdigris: float = 0.28,
    hammer_scale: float = 7.0,
    roughness: float = 0.30,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Forged Bronze, Raw Copper, or Rose Gold with hammered dimples and verdigris teal oxidation."""
    presets = {
        "bronze": ("#A87442", "#478A78"),
        "copper": ("#CD7B54", "#3E9985"),
        "rose_gold": ("#DC9E8A", "#6B5E5B"),
    }
    base_hex, verd_hex = presets.get(str(metal_preset).lower(), presets["bronze"])
    dents = voronoi_noise(height, width, scale=hammer_scale, jitter=0.88, output="f1", seed=seed + 19, tileable=tileable)
    gr = grunge_blotch(height, width, scale=5.0, contrast=1.5, speckle=0.3, seed=seed + 53, tileable=tileable)
    v_mask = np.clip((gr * (0.55 + 0.45 * dents) - (1.0 - float(verdigris) * 1.25)) * 2.8, 0.0, 1.0).astype(np.float32)

    h = np.clip(0.62 - dents * 0.32 - v_mask * 0.12, 0.0, 1.0)
    alb = _tint_albedo(0.62 - dents * 0.34, base_hex, dark_mul=0.52, bright_mul=1.25)
    v_rgb = _hex_rgb(verd_hex)
    alb[..., :3] = alb[..., :3] * (1.0 - v_mask[..., None] * 0.88) + v_rgb[None, None, :] * (v_mask[..., None] * 0.88)
    rough = np.clip(float(roughness) + dents * 0.14 + v_mask * 0.52, 0.08, 0.96)
    metal = np.clip(0.92 - v_mask * 0.85, 0.0, 1.0)
    ao = np.clip(1.0 - dents * 0.18 - v_mask * 0.20, 0.45, 1.0)
    return material_make(alb, height=h, roughness=rough, metallic=metal, ao=ao, normal_strength=2.5, tileable=tileable)


def mat_gold_precious(
    height: int,
    width: int,
    *,
    base_color: str = "#F2C64D",
    finish: str = "polished",
    roughness: float = 0.18,
    tarnish: float = 0.12,
    engrave_scale: float = 8.0,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Polished, Hammered, or Antique Engraved Gold PBR MATERIAL."""
    macro = fbm_noise(height, width, scale=4.0, octaves=3, seed=seed + 7, tileable=tileable)
    if finish == "hammered":
        tex = 1.0 - voronoi_noise(height, width, scale=engrave_scale, jitter=0.9, output="f1", seed=seed + 29, tileable=tileable) * 0.65
    elif finish == "antique":
        tex = 0.55 + 0.45 * fold_marble_noise(height, width, scale=engrave_scale * 0.5, veins=4.0, turbulence=1.8, seed=seed + 41, tileable=tileable)
    else:
        tex = 0.65 + 0.35 * gabor_noise(height, width, scale=16.0, angle_deg=15.0, anisotropy=4.0, seed=seed + 13, tileable=tileable)

    tarn = np.clip((grunge_blotch(height, width, scale=5.0, seed=seed + 89, tileable=tileable) - (1.0 - float(tarnish) * 1.3)) * 2.5, 0.0, 1.0)
    h = np.clip(0.55 + (tex - 0.5) * 0.32 - tarn * 0.12, 0.0, 1.0)
    alb = _tint_albedo(0.58 + (tex - 0.5) * 0.26 + (macro - 0.5) * 0.14 - tarn * 0.35, base_color, dark_mul=0.58, bright_mul=1.18)
    rough = np.clip(float(roughness) + (1.0 - tex) * 0.12 + tarn * 0.40, 0.05, 0.90)
    metal = np.clip(0.98 - tarn * 0.45, 0.3, 1.0)
    return material_make(alb, height=h, roughness=rough, metallic=metal, ao=np.clip(1.0 - tarn * 0.25, 0.5, 1.0), normal_strength=2.0, tileable=tileable)


def mat_steel_iron(
    height: int,
    width: int,
    *,
    alloy: str = "steel",
    roughness: float = 0.28,
    rust_amount: float = 0.15,
    brushed_angle: float = 0.0,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Brushed Steel, Forged Iron, Damascus Wave Steel, or Dark Gunmetal PBR MATERIAL."""
    alloys = {
        "steel": ("#B8C2CE", 0.92),
        "iron": ("#7A808A", 0.86),
        "damascus": ("#9DA7B4", 0.93),
        "gunmetal": ("#4A515E", 0.88),
    }
    hex_c, met0 = alloys.get(str(alloy).lower(), alloys["steel"])
    if alloy == "damascus":
        pat = fold_marble_noise(height, width, scale=5.0, veins=7.0, turbulence=2.8, angle_deg=brushed_angle + 25.0, seed=seed + 17, tileable=tileable)
    elif alloy == "iron":
        pat = 0.55 * fbm_noise(height, width, scale=8.0, octaves=4, seed=seed + 23, tileable=tileable) + 0.45 * (1.0 - voronoi_noise(height, width, scale=9.0, output="f1", seed=seed + 31, tileable=tileable))
    else:
        pat = gabor_noise(height, width, scale=15.0, angle_deg=brushed_angle, anisotropy=5.0, octaves=3, seed=seed + 13, tileable=tileable)

    scr = scratch_noise(height, width, count=28, length=0.22, angle_deg=brushed_angle, angle_jitter=40.0, seed=seed + 67, tileable=tileable)
    rust_n = grunge_blotch(height, width, scale=5.5, contrast=1.6, speckle=0.4, seed=seed + 109, tileable=tileable)
    rust_m = np.clip((rust_n - (1.0 - float(rust_amount) * 1.25)) * 2.7, 0.0, 1.0).astype(np.float32)

    h = np.clip(0.55 + (pat - 0.5) * 0.25 - scr * 0.18 - rust_m * 0.18, 0.0, 1.0)
    alb = _tint_albedo(0.52 + (pat - 0.5) * 0.34 + scr * 0.16, hex_c, dark_mul=0.50, bright_mul=1.22)
    rust_rgb = np.array([0.52, 0.22, 0.09], dtype=np.float32)
    alb[..., :3] = alb[..., :3] * (1.0 - rust_m[..., None] * 0.90) + rust_rgb[None, None, :] * (rust_m[..., None] * 0.90)
    rough = np.clip(float(roughness) + (1.0 - pat) * 0.14 + rust_m * 0.58, 0.08, 0.98)
    metal = np.clip(met0 - rust_m * 0.85, 0.0, 1.0)
    return material_make(alb, height=h, roughness=rough, metallic=metal, ao=np.clip(1.0 - rust_m * 0.28, 0.4, 1.0), normal_strength=2.3, tileable=tileable)


def mat_chrome_titanium(
    height: int,
    width: int,
    *,
    base_color: str = "#DCE4EE",
    anodize_amount: float = 0.18,
    roughness: float = 0.09,
    micro_grain: float = 0.12,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Mirror Chrome / Anodized Titanium conductor with ultra-low roughness & optional heat iridescence."""
    grain = gabor_noise(height, width, scale=18.0, angle_deg=0.0, anisotropy=5.0, octaves=2, seed=seed + 5, tileable=tileable)
    wave = fbm_noise(height, width, scale=3.0, octaves=3, seed=seed + 37, tileable=tileable)
    alb = _tint_albedo(0.70 + (grain - 0.5) * 0.16 * float(micro_grain), base_color, dark_mul=0.72, bright_mul=1.12)
    aa = float(np.clip(anodize_amount, 0.0, 1.0))
    if aa > 1e-3:
        irid_r = 0.55 + 0.45 * np.cos(6.283 * (wave + 0.00))
        irid_g = 0.55 + 0.45 * np.cos(6.283 * (wave + 0.33))
        irid_b = 0.55 + 0.45 * np.cos(6.283 * (wave + 0.67))
        irid = np.stack((irid_r, irid_g, irid_b), axis=-1)
        alb[..., :3] = alb[..., :3] * (1.0 - aa * 0.55) + irid * (aa * 0.55)
    h = np.clip(0.5 + (grain - 0.5) * 0.12 * float(micro_grain), 0.0, 1.0)
    rough = np.clip(float(roughness) + (grain - 0.5) * 0.06 * float(micro_grain), 0.04, 0.65)
    return material_make(alb, height=h, roughness=rough, metallic=0.98, ao=1.0, normal_strength=1.4, tileable=tileable)


# ===========================================================================
# 2. DIELECTRICS, PLASTICS, GLASS & TRANSPARENT CRYSTALS (PBR)
# ===========================================================================


def mat_plastic(
    height: int,
    width: int,
    *,
    color: str = "#3A7BD5",
    finish: str = "satin",
    roughness: float = 0.35,
    grain_amount: float = 0.18,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Matte, Satin, or Glossy Injection-Molded Plastic PBR MATERIAL (metallic=0.0, micro-pebble grain)."""
    pebble = value_noise(height, width, scale=22.0, seed=seed + 11, tileable=tileable)
    macro = fbm_noise(height, width, scale=3.5, octaves=2, seed=seed + 47, tileable=tileable)
    ga = float(np.clip(grain_amount, 0.0, 1.0))
    fin_bias = {"glossy": -0.18, "satin": 0.0, "matte": 0.24}.get(str(finish).lower(), 0.0)
    h = np.clip(0.5 + (pebble - 0.5) * 0.22 * ga + (macro - 0.5) * 0.06, 0.0, 1.0)
    alb = _tint_albedo(0.62 + (macro - 0.5) * 0.12 + (pebble - 0.5) * 0.08 * ga, color, dark_mul=0.72, bright_mul=1.14)
    rough = np.clip(float(roughness) + fin_bias + (pebble - 0.5) * 0.12 * ga, 0.06, 0.94)
    return material_make(alb, height=h, roughness=rough, metallic=0.0, ao=1.0, normal_strength=1.5 * ga + 0.4, tileable=tileable)


def mat_painted_enamel(
    height: int,
    width: int,
    *,
    paint_color: str = "#B9382C",
    undercoat_color: str = "#A8B2BE",
    chip_amount: float = 0.25,
    roughness: float = 0.28,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Painted Enamel / Lacquer over Metallic Undercoat with realistic peeled paint chips."""
    gr = grunge_blotch(height, width, scale=5.5, contrast=1.7, speckle=0.4, seed=seed + 19, tileable=tileable)
    scr = scratch_noise(height, width, count=22, length=0.20, angle_deg=-35.0, angle_jitter=55.0, seed=seed + 73, tileable=tileable)
    ca = float(np.clip(chip_amount, 0.0, 1.0))
    chip = np.clip(((gr - (1.0 - ca * 1.15)) * 4.5 + scr * ca * 1.5), 0.0, 1.0).astype(np.float32)
    paint_alb = _tint_albedo(0.60 + (gr - 0.5) * 0.16, paint_color, dark_mul=0.68, bright_mul=1.14)
    metal_alb = _tint_albedo(0.60 + (gr - 0.5) * 0.22, undercoat_color, dark_mul=0.58, bright_mul=1.22)

    alb = paint_alb * (1.0 - chip[..., None]) + metal_alb * chip[..., None]
    h = np.clip(0.68 - chip * 0.34, 0.0, 1.0)
    rough = np.clip(float(roughness) * (1.0 - chip) + 0.32 * chip + (gr - 0.5) * 0.08, 0.08, 0.92)
    metal = np.clip(chip * 0.88, 0.0, 1.0)
    return material_make(alb, height=h, roughness=rough, metallic=metal, ao=np.clip(1.0 - chip * 0.16, 0.5, 1.0), normal_strength=2.6, tileable=tileable)


def mat_ceramic_porcelain(
    height: int,
    width: int,
    *,
    glaze_color: str = "#EAE5DC",
    crack_color: str = "#6B6158",
    crackle_amount: float = 0.28,
    roughness: float = 0.14,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Glazed Ceramic / Porcelain with fine kiln crazing cracks and smooth specular gloss."""
    cracks = voronoi_noise(height, width, scale=7.0, jitter=0.92, output="cracks", seed=seed + 23, tileable=tileable)
    macro = fbm_noise(height, width, scale=3.0, octaves=3, seed=seed + 61, tileable=tileable)
    cm = np.clip((cracks - 0.72) * 3.8 * float(crackle_amount), 0.0, 1.0).astype(np.float32)
    alb = _tint_albedo(0.72 + (macro - 0.5) * 0.14, glaze_color, dark_mul=0.78, bright_mul=1.08)
    c_rgb = _hex_rgb(crack_color)
    alb[..., :3] = alb[..., :3] * (1.0 - cm[..., None]) + c_rgb[None, None, :] * cm[..., None]
    h = np.clip(0.65 + (macro - 0.5) * 0.08 - cm * 0.35, 0.0, 1.0)
    rough = np.clip(float(roughness) + cm * 0.52, 0.06, 0.92)
    return material_make(alb, height=h, roughness=rough, metallic=0.0, ao=np.clip(1.0 - cm * 0.25, 0.5, 1.0), normal_strength=2.2, tileable=tileable)


def mat_rubber_grip(
    height: int,
    width: int,
    *,
    color: str = "#2A2E35",
    tread: str = "diamond",
    roughness: float = 0.78,
    wear: float = 0.20,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Vulcanized Grip Rubber with diamond, pebble, or ribbed tactile relief and matte roughness."""
    if tread == "pebble":
        pat = polka_rivets(height, width, cols=10, rows=10, radius=0.34, stagger=True, style="dome", seed=seed + 13)
    elif tread == "ribbed":
        u, v = uv_grid(height, width)
        pat = (0.5 + 0.5 * np.sin(v * 2.0 * math.pi * 12.0)).astype(np.float32)
    else:
        u, v = uv_grid(height, width)
        a = np.abs(((u + v) * 8.0) % 1.0 - 0.5) * 2.0
        b = np.abs(((u - v) * 8.0) % 1.0 - 0.5) * 2.0
        pat = np.clip(1.0 - np.maximum(a, b) * 1.15, 0.0, 1.0).astype(np.float32)
    gr = fbm_noise(height, width, scale=12.0, octaves=3, seed=seed + 41, tileable=tileable)
    h = np.clip(0.28 + pat * 0.58 + (gr - 0.5) * 0.08, 0.0, 1.0)
    alb = _tint_albedo(0.45 + pat * 0.30 + gr * 0.22 * float(wear), color, dark_mul=0.62, bright_mul=1.35)
    rough = np.clip(float(roughness) - pat * 0.10 * float(wear) + (gr - 0.5) * 0.08, 0.42, 0.98)
    return material_make(alb, height=h, roughness=rough, metallic=0.0, ao=np.clip(0.75 + pat * 0.25, 0.5, 1.0), normal_strength=3.0, tileable=tileable)


def mat_glass(
    height: int,
    width: int,
    *,
    tint_color: str = "#9CE6FF",
    opacity: float = 0.38,
    frost_roughness: float = 0.08,
    fresnel_rim: float = 0.72,
    style: str = "clear",
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Transparent Clear / Frosted / Stained / Rippled Glass PBR MATERIAL with real alpha translucency & Fresnel sheen."""
    u, v = uv_grid(height, width)
    du = np.minimum(u, 1.0 - u)
    dv = np.minimum(v, 1.0 - v)
    edge = np.clip(1.0 - np.minimum(du, dv) / 0.16, 0.0, 1.0)
    diag_glint = np.exp(-(((u + v - 0.72) / 0.10) ** 2)) * 0.45 + np.exp(-(((u + v - 1.18) / 0.06) ** 2)) * 0.25

    if style == "rippled":
        wave = caustic_noise(height, width, scale=4.5, sharpness=1.8, seed=seed + 17, tileable=tileable)
        h = np.clip(0.45 + wave * 0.35, 0.0, 1.0)
    elif style == "stained":
        lead = voronoi_noise(height, width, scale=5.0, jitter=0.85, output="cracks", seed=seed + 31, tileable=tileable)
        cid = voronoi_noise(height, width, scale=5.0, jitter=0.85, output="cell_id", seed=seed + 31, tileable=tileable)
        h = np.clip(0.55 + cid * 0.20 - (lead > 0.78).astype(np.float32) * 0.40, 0.0, 1.0)
        diag_glint = diag_glint + (cid - 0.5) * 0.35
    elif style == "frosted":
        frost = value_noise(height, width, scale=28.0, seed=seed + 43, tileable=tileable)
        h = np.clip(0.5 + (frost - 0.5) * 0.24, 0.0, 1.0)
    else:
        subtle = fbm_noise(height, width, scale=3.0, octaves=2, seed=seed + 9, tileable=tileable)
        h = np.clip(0.5 + (subtle - 0.5) * 0.08 + edge * 0.12, 0.0, 1.0)

    op = float(np.clip(opacity, 0.08, 0.98))
    alpha = np.clip(op + edge * 0.34 * float(fresnel_rim) + diag_glint * 0.22, 0.10, 0.96).astype(np.float32)
    alb = _tint_albedo(0.62 + diag_glint * 0.48 + edge * 0.25 * float(fresnel_rim), tint_color, dark_mul=0.55, bright_mul=1.42, alpha=alpha)
    if style == "stained":
        lead_m = (voronoi_noise(height, width, scale=5.0, jitter=0.85, output="cracks", seed=seed + 31, tileable=tileable) > 0.80).astype(np.float32)
        alb[..., :3] = alb[..., :3] * (1.0 - lead_m[..., None] * 0.85) + 0.14 * lead_m[..., None]
        alb[..., 3] = np.maximum(alb[..., 3], lead_m * 0.98)

    r_add = 0.26 if style == "frosted" else 0.0
    rough = np.clip(float(frost_roughness) + r_add + (h - 0.5) * 0.08, 0.04, 0.85)
    return material_make(alb, height=h, roughness=rough, metallic=0.05, ao=1.0, sss=0.45, normal_strength=1.8, tileable=tileable)


def mat_crystal_gem(
    height: int,
    width: int,
    *,
    crystal_color: str = "#A855F7",
    core_glow_color: str = "#38F2FF",
    opacity: float = 0.72,
    facet_scale: float = 5.0,
    glow_strength: float = 0.45,
    roughness: float = 0.10,
    seed: int = 42,
    tileable: bool = True,
    phase: float = 0.0,
) -> dict[str, np.ndarray]:
    """Transparent Faceted Crystal / Amethyst / Gemstone PBR MATERIAL with internal refraction, glow & alpha translucency!"""
    facets = crystal_facets(height, width, scale=facet_scale, facet_contrast=0.80, rim_width=0.14, seed=seed + 13, tileable=tileable)
    caus = caustic_noise(height, width, scale=max(2.0, facet_scale * 0.85), sharpness=2.4, seed=seed + 59, tileable=tileable, phase=phase, loop_speed=1.0)
    edge = voronoi_noise(height, width, scale=facet_scale, jitter=0.95, output="edge", seed=seed + 16, tileable=tileable)
    rim = np.clip(1.0 - edge / 0.14, 0.0, 1.0)

    pulse = 0.82 + 0.18 * math.sin(float(phase) * 2.0 * math.pi)
    op = float(np.clip(opacity, 0.18, 1.0))
    alpha = np.clip(op + rim * 0.24 + facets * 0.10, 0.20, 0.98).astype(np.float32)

    c_rgb = _hex_rgb(crystal_color)
    g_rgb = _hex_rgb(core_glow_color)
    gs = float(np.clip(glow_strength, 0.0, 1.5)) * pulse

    rgb = (
        c_rgb[None, None, :] * (0.42 + 0.78 * facets[..., None])
        + g_rgb[None, None, :] * (caus[..., None] * 0.38 * gs + rim[..., None] * 0.28)
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    alb = np.concatenate((rgb, alpha[..., None]), axis=-1).astype(np.float32)

    em_mask = np.clip(caus * 0.65 + rim * 0.45, 0.0, 1.0) * min(1.0, gs)
    em_rgb = np.clip((c_rgb * 0.45 + g_rgb * 0.55)[None, None, :] * em_mask[..., None], 0.0, 1.0)
    em = np.concatenate((em_rgb, em_mask[..., None]), axis=-1).astype(np.float32)

    h = np.clip(0.25 + facets * 0.65 + rim * 0.12, 0.0, 1.0)
    rough = np.clip(float(roughness) + (1.0 - facets) * 0.10, 0.04, 0.55)
    return material_make(alb, height=h, roughness=rough, metallic=0.14, ao=np.clip(0.72 + facets * 0.28, 0.5, 1.0), emissive=em, sss=0.72, normal_strength=3.0, tileable=tileable)


def mat_ice_frost(
    height: int,
    width: int,
    *,
    ice_color: str = "#8AD8FF",
    opacity: float = 0.68,
    crack_density: float = 6.0,
    frost_amount: float = 0.35,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Translucent Cracked Glacier Ice & White Frost Rime PBR MATERIAL."""
    cracks = voronoi_noise(height, width, scale=crack_density, jitter=0.9, output="cracks", seed=seed + 19, tileable=tileable)
    cr_m = np.clip((cracks - 0.62) * 3.0, 0.0, 1.0)
    frost = np.clip((fbm_noise(height, width, scale=8.0, octaves=4, seed=seed + 53, tileable=tileable) - (0.72 - float(frost_amount) * 0.55)) * 2.8, 0.0, 1.0)
    alpha = np.clip(float(opacity) + cr_m * 0.25 + frost * 0.28, 0.20, 0.98).astype(np.float32)
    alb = _tint_albedo(0.58 + cr_m * 0.38 + frost * 0.42, ice_color, dark_mul=0.52, bright_mul=1.42, alpha=alpha)
    alb[..., :3] = np.clip(alb[..., :3] * (1.0 - frost[..., None] * 0.45) + frost[..., None] * 0.45, 0.0, 1.0)
    h = np.clip(0.60 - cr_m * 0.32 + frost * 0.16, 0.0, 1.0)
    rough = np.clip(0.08 + frost * 0.62 + cr_m * 0.24, 0.05, 0.88)
    return material_make(alb, height=h, roughness=rough, metallic=0.02, ao=1.0, sss=0.68, normal_strength=2.6, tileable=tileable)


# ===========================================================================
# 3. PAINTERLY & BIO PBR EXPANSION
# ===========================================================================


def mat_dragon_scales_pbr(
    height: int,
    width: int,
    *,
    scale_color: str = "#7A6FA8",
    rim_color: str = "#E8EEFF",
    cols: int = 6,
    rows: int = 8,
    iridescence: float = 0.25,
    roughness: float = 0.38,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Sculpted Overlapping Dragon Scales PBR MATERIAL with lit rim catchlights & optional pearlescent sheen."""
    h = scallop_shingles(height, width, cols=cols, rows=rows, overlap=0.52, bevel=0.22, seed=seed + 17)
    micro = fbm_noise(height, width, scale=max(4.0, cols * 1.5), octaves=3, seed=seed + 71, tileable=tileable)
    h_tot = np.clip(h * 0.80 + micro * 0.20, 0.0, 1.0)
    rim = np.clip((h - 0.62) * 2.8, 0.0, 1.0)
    alb = _tint_albedo(0.24 + h_tot * 0.76, scale_color, dark_mul=0.38, bright_mul=1.22)
    r_rgb = _hex_rgb(rim_color)
    alb[..., :3] = np.clip(alb[..., :3] * (1.0 - rim[..., None] * 0.55) + r_rgb[None, None, :] * (rim[..., None] * 0.55), 0.0, 1.0)
    ir = float(np.clip(iridescence, 0.0, 1.0))
    if ir > 1e-3:
        shift = np.stack((0.5 + 0.5 * np.cos(micro * 6.28), 0.5 + 0.5 * np.cos(micro * 6.28 + 2.1), 0.5 + 0.5 * np.cos(micro * 6.28 + 4.2)), axis=-1)
        alb[..., :3] = np.clip(alb[..., :3] * (1.0 - ir * 0.28 * rim[..., None]) + shift * (ir * 0.28 * rim[..., None]), 0.0, 1.0)
    rough = np.clip(float(roughness) + (1.0 - h_tot) * 0.26, 0.12, 0.92)
    ao = np.clip(0.45 + h_tot * 0.55, 0.35, 1.0)
    return material_make(alb, height=h_tot, roughness=rough, metallic=0.14 + ir * 0.22, ao=ao, normal_strength=2.8, tileable=tileable)


def mat_marble_veined(
    height: int,
    width: int,
    *,
    stone_color: str = "#EAE6DF",
    vein_color: str = "#524D5C",
    vein_scale: float = 4.0,
    roughness: float = 0.16,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Polished Veined Marble / Jade / Alabaster PBR MATERIAL."""
    vein = fold_marble_noise(height, width, scale=vein_scale, veins=4.0, turbulence=2.8, angle_deg=35.0, seed=seed + 13, tileable=tileable)
    sharp = np.clip((vein - 0.58) * 2.8, 0.0, 1.0)
    s_rgb = _hex_rgb(stone_color)
    v_rgb = _hex_rgb(vein_color)
    rgb = np.clip(s_rgb[None, None, :] * (0.86 + 0.18 * (1.0 - vein[..., None])) * (1.0 - sharp[..., None] * 0.78) + v_rgb[None, None, :] * (sharp[..., None] * 0.78), 0.0, 1.0)
    alb = np.concatenate((rgb, np.ones((height, width, 1), dtype=np.float32)), axis=-1)
    h = np.clip(0.62 - sharp * 0.18, 0.0, 1.0)
    rough = np.clip(float(roughness) + sharp * 0.22, 0.06, 0.85)
    return material_make(alb, height=h, roughness=rough, metallic=0.02, ao=1.0, normal_strength=1.6, tileable=tileable)


def mat_obsidian_volcanic(
    height: int,
    width: int,
    *,
    obsidian_color: str = "#1D192B",
    fissure_color: str = "#FF5A1F",
    fissure_glow: float = 0.55,
    roughness: float = 0.18,
    seed: int = 42,
    tileable: bool = True,
    phase: float = 0.0,
) -> dict[str, np.ndarray]:
    """Conchoidal Fractured Volcanic Obsidian with optional animated magma/chronal fissures."""
    cr = crystal_noise(height, width, scale=6.0, sharpness=2.6, facet_contrast=0.65, seed=seed + 29, tileable=tileable)
    cracks = voronoi_noise(height, width, scale=5.0, jitter=0.92, output="cracks", seed=seed + 83, tileable=tileable)
    fis = np.clip((cracks - 0.76) * 4.5, 0.0, 1.0).astype(np.float32)
    pulse = 0.76 + 0.24 * math.sin(float(phase) * 2.0 * math.pi)
    fg = float(np.clip(fissure_glow, 0.0, 1.5)) * pulse

    alb = _tint_albedo(0.35 + cr * 0.65, obsidian_color, dark_mul=0.45, bright_mul=1.55)
    f_rgb = _hex_rgb(fissure_color)
    alb[..., :3] = np.clip(alb[..., :3] * (1.0 - fis[..., None] * min(1.0, fg)) + f_rgb[None, None, :] * (fis[..., None] * min(1.0, fg)), 0.0, 1.0)
    em_mask = np.clip(fis * min(1.0, fg), 0.0, 1.0)
    em = np.concatenate((f_rgb[None, None, :] * em_mask[..., None], em_mask[..., None]), axis=-1).astype(np.float32)
    h = np.clip(0.35 + cr * 0.55 - fis * 0.35, 0.0, 1.0)
    rough = np.clip(float(roughness) + (1.0 - cr) * 0.18 + fis * 0.35, 0.06, 0.90)
    return material_make(alb, height=h, roughness=rough, metallic=0.12, ao=np.clip(0.70 + cr * 0.30, 0.4, 1.0), emissive=em, normal_strength=2.9, tileable=tileable)


def mat_velvet_silk(
    height: int,
    width: int,
    *,
    cloth_color: str = "#3E266A",
    sheen_color: str = "#C09CFF",
    weave_scale: int = 12,
    fold_strength: float = 0.45,
    roughness: float = 0.72,
    seed: int = 42,
    tileable: bool = True,
) -> dict[str, np.ndarray]:
    """Royal Velvet / Silk / Arcane Cloth with soft drape folds, fine textile weave & rim sheen."""
    wv = weave_chainmail(height, width, cols=weave_scale, rows=weave_scale, style="weave", thickness=0.38, seed=seed + 7)
    folds = fold_marble_noise(height, width, scale=2.5, veins=2.0, turbulence=1.3, angle_deg=78.0, seed=seed + 41, tileable=tileable)
    fs = float(np.clip(fold_strength, 0.0, 1.0))
    h = np.clip(0.5 + (folds - 0.5) * 0.55 * fs + (wv - 0.5) * 0.18, 0.0, 1.0)
    alb = _tint_albedo(0.38 + h * 0.62, cloth_color, dark_mul=0.45, bright_mul=1.22)
    sh_rgb = _hex_rgb(sheen_color)
    sh_m = np.clip((folds - 0.58) * 2.2, 0.0, 1.0) * 0.42
    alb[..., :3] = np.clip(alb[..., :3] * (1.0 - sh_m[..., None]) + sh_rgb[None, None, :] * sh_m[..., None], 0.0, 1.0)
    rough = np.clip(float(roughness) - sh_m * 0.25 + (1.0 - wv) * 0.10, 0.25, 0.96)
    return material_make(alb, height=h, roughness=rough, metallic=0.0, ao=np.clip(0.65 + h * 0.35, 0.45, 1.0), normal_strength=2.2, tileable=tileable)


def mat_glowing_runes_pbr(
    height: int,
    width: int,
    *,
    base_color: str = "#2A2B3D",
    rune_color: str = "#38F2FF",
    rune_scale: int = 6,
    pulse_speed: float = 1.0,
    glow_strength: float = 0.90,
    seed: int = 42,
    tileable: bool = True,
    phase: float = 0.0,
) -> dict[str, np.ndarray]:
    """Arcane / Chronal Etched Glowing Runes PBR MATERIAL with animated energy pulse & carved stone base."""
    runes = circuit_runes(height, width, scale=rune_scale, density=0.62, style="runes", seed=seed + 19)
    stone = fbm_noise(height, width, scale=6.0, octaves=4, seed=seed + 53, tileable=tileable)
    pulse = 0.68 + 0.32 * math.sin(float(phase) * float(pulse_speed) * 2.0 * math.pi)
    gs = float(np.clip(glow_strength, 0.0, 1.5)) * pulse
    halo = _box_blur_2d(runes, 1, tileable=tileable) * 0.55
    glow_m = np.clip(np.maximum(runes, halo) * min(1.0, gs), 0.0, 1.0)

    alb = _tint_albedo(0.38 + stone * 0.62, base_color, dark_mul=0.52, bright_mul=1.22)
    r_rgb = _hex_rgb(rune_color)
    alb[..., :3] = np.clip(alb[..., :3] * (1.0 - glow_m[..., None] * 0.78) + r_rgb[None, None, :] * (glow_m[..., None] * 1.15), 0.0, 1.0)
    em = np.concatenate((r_rgb[None, None, :] * glow_m[..., None], glow_m[..., None]), axis=-1).astype(np.float32)
    h = np.clip(0.58 + (stone - 0.5) * 0.24 - runes * 0.32, 0.0, 1.0)
    rough = np.clip(0.62 * (1.0 - runes) + 0.14 * runes, 0.10, 0.92)
    return material_make(alb, height=h, roughness=rough, metallic=0.15 * (1.0 - runes), ao=np.clip(0.78 + stone * 0.22, 0.5, 1.0), emissive=em, normal_strength=2.6, tileable=tileable)


# ===========================================================================
# 4. MATERIAL BUS SEQUENCER & PBR NORMAL UTILITIES
# ===========================================================================


def material_split(mat: dict[str, np.ndarray] | np.ndarray, *, tileable: bool = False) -> dict[str, np.ndarray]:
    """Unpack any PBR MATERIAL bus into explicit individually wired channels (including `opacity`)."""
    b = _ensure_bus(mat, tileable=tileable)
    out = dict(b)
    out["opacity"] = np.clip(b["albedo"][..., 3], 0.0, 1.0).astype(np.float32)
    return out


def material_tweak(
    mat: dict[str, np.ndarray] | np.ndarray,
    *,
    hue_shift_deg: float = 0.0,
    saturation: float = 1.0,
    brightness: float = 0.0,
    roughness_offset: float = 0.0,
    roughness_scale: float = 1.0,
    metallic_offset: float = 0.0,
    normal_strength: float = 1.0,
    opacity_scale: float = 1.0,
    emissive_boost: float = 1.0,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Non-destructive Substance/Unreal PBR Material Tweaker (HSV, Roughness, Metallic, Normal, Opacity, Emissive)."""
    b = _ensure_bus(mat, tileable=tileable)
    alb = b["albedo"].copy()
    if abs(float(hue_shift_deg)) > 1e-3 or abs(float(saturation) - 1.0) > 1e-3 or abs(float(brightness)) > 1e-3:
        alb = oklch_adjust(alb, hue_shift_deg=float(hue_shift_deg), chroma_scale=float(saturation), lightness_offset=float(brightness))
    alb[..., 3] = np.clip(alb[..., 3] * float(opacity_scale), 0.0, 1.0)

    rough = np.clip(b["roughness"] * float(roughness_scale) + float(roughness_offset), 0.04, 1.0).astype(np.float32)
    metal = np.clip(b["metallic"] + float(metallic_offset), 0.0, 1.0).astype(np.float32)
    nrm = normal_strength_op(b["normal"], strength=float(normal_strength), flip_y=False)
    em = b["emissive"].copy()
    eb = max(0.0, float(emissive_boost))
    em[..., :3] = np.clip(em[..., :3] * eb, 0.0, 1.0)
    em[..., 3] = np.clip(em[..., 3] * min(1.0, eb), 0.0, 1.0)

    return material_make(
        alb, height=b["height"], normal=nrm, roughness=rough, metallic=metal,
        ao=b["ao"], shadow=b["shadow"], bent_normal=b["bent_normal"], thickness=b["thickness"],
        emissive=em, sss=b["sss"], porosity=b["porosity"], tileable=tileable,
    )


def material_region_apply(
    base: dict[str, np.ndarray] | np.ndarray,
    top: dict[str, np.ndarray] | np.ndarray,
    *,
    mask: np.ndarray | None = None,
    mesh_ctx: MeshBakeContext | None = None,
    part_pattern: str = "",
    face_filter: str = "all",
    opacity: float = 1.0,
    use_height_blend: bool = False,
    height_contrast: float = 6.0,
    seam_ao_strength: float = 0.22,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Apply a PBR `top` MATERIAL strictly onto a masked UV/3D region over `base` (Brass/Glass/Crystal layering!)."""
    b = _ensure_bus(base, tileable=tileable)
    t = _ensure_bus(top, tileable=tileable)
    H, W = b["albedo"].shape[:2]

    if mask is not None:
        m = _to_scalar_2d(mask, H, W, 1.0)
        if str(part_pattern or "").strip() or (str(face_filter or "all").lower() != "all"):
            m = m * resolve_part_mask(H, W, mesh_ctx, part_pattern=part_pattern, face_filter=face_filter)
    else:
        m = resolve_part_mask(H, W, mesh_ctx, part_pattern=part_pattern, face_filter=face_filter)

    m = np.clip(m * float(opacity), 0.0, 1.0).astype(np.float32)
    if use_height_blend:
        return material_height_blend(b, t, mask=m, contrast=float(height_contrast), seam_ao_strength=float(seam_ao_strength), tileable=tileable)

    out = material_blend(b, t, mask=m, opacity=1.0, normal_mode="lerp", tileable=tileable)
    if float(seam_ao_strength) > 1e-4:
        blur_m = _box_blur_2d(m, 1, tileable=tileable)
        seam = np.clip(4.0 * blur_m * (1.0 - blur_m), 0.0, 1.0)
        s_ao = float(np.clip(seam_ao_strength, 0.0, 0.7))
        out["ao"] = np.clip(out["ao"] * (1.0 - seam * s_ao), 0.08, 1.0)
        out["albedo"][..., :3] = np.clip(out["albedo"][..., :3] * (1.0 - seam[..., None] * s_ao * 0.45), 0.0, 1.0)
    return out


def material_weathering(
    mat: dict[str, np.ndarray] | np.ndarray,
    *,
    mesh_ctx: MeshBakeContext | None = None,
    preset: str = "edge_wear",
    amount: float = 0.42,
    color: str = "#DCE4EE",
    roughness_target: float = 0.65,
    metallic_target: float = 0.85,
    seed: int = 42,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """One-node Substance-style Weathering FX (`edge_wear`, `patina`, `rust`, `dust`, `moss`) over any MATERIAL."""
    b = _ensure_bus(mat, tileable=tileable)
    H, W = b["albedo"].shape[:2]
    pr = str(preset or "edge_wear").strip().lower()

    if pr == "dust":
        m = mask_world_direction(H, W, mesh_ctx, direction="top_up", threshold=0.45, softness=0.30, noise_breakup=0.35, seed=seed)
        def_col, def_r, def_m = "#C2B9A8", 0.88, 0.0
    elif pr == "moss":
        m_top = mask_world_direction(H, W, mesh_ctx, direction="top_up", threshold=0.38, softness=0.28, noise_breakup=0.40, seed=seed)
        m_cr = mask_smart_crevice_dirt(H, W, mesh_ctx, dirt_amount=0.65, seed=seed + 13)
        m = np.clip(np.maximum(m_top * 0.85, m_cr * 0.75), 0.0, 1.0)
        def_col, def_r, def_m = "#4B7A3B", 0.85, 0.0
    elif pr in ("patina", "rust"):
        m = mask_smart_crevice_dirt(H, W, mesh_ctx, dirt_amount=0.62, seed=seed + 29)
        def_col = "#438A78" if pr == "patina" else "#8A3C1B"
        def_r, def_m = 0.82, 0.05
    else:
        m = mask_smart_edge_wear(H, W, mesh_ctx, wear_amount=0.60, seed=seed + 41)
        def_col, def_r, def_m = "#DCE4EE", 0.24, 0.90

    w = np.clip(m * float(amount) * 1.35, 0.0, 1.0).astype(np.float32)
    c_hex = color if (str(color or "").startswith("#") and str(color).upper() != "#DCE4EE" or pr == "edge_wear") else def_col
    w_rgb = _hex_rgb(c_hex)
    r_tgt = float(roughness_target) if pr == "edge_wear" else def_r
    m_tgt = float(metallic_target) if pr == "edge_wear" else def_m

    alb = b["albedo"].copy()
    alb[..., :3] = np.clip(alb[..., :3] * (1.0 - w[..., None]) + w_rgb[None, None, :] * w[..., None], 0.0, 1.0)
    rough = np.clip(b["roughness"] * (1.0 - w) + r_tgt * w, 0.05, 0.98)
    metal = np.clip(b["metallic"] * (1.0 - w) + m_tgt * w, 0.0, 1.0)
    h = np.clip(b["height"] + (0.08 * w if pr in ("moss", "dust") else -0.06 * w), 0.0, 1.0)
    return material_make(
        alb, height=h, roughness=rough, metallic=metal, ao=b["ao"],
        shadow=b["shadow"], bent_normal=b["bent_normal"], thickness=b["thickness"],
        emissive=b["emissive"], sss=b["sss"], porosity=b["porosity"], tileable=tileable,
    )


def normal_from_height_multi(
    height_field: np.ndarray,
    *,
    macro_strength: float = 2.0,
    detail_strength: float = 2.6,
    convention: str = "opengl",
    tileable: bool = True,
) -> np.ndarray:
    """Multi-scale Scharr PBR Normal Generator blending broad blurred macro slopes + crisp micro detail."""
    h = np.asarray(height_field, dtype=np.float32)
    if h.ndim == 3:
        h = 0.299 * h[..., 0] + 0.587 * h[..., 1] + 0.114 * h[..., 2]
    h_macro = _box_blur_2d(h, 2, tileable=tileable)
    n_macro = height_to_normal_map(h_macro, strength=float(macro_strength), tileable=tileable, convention=convention)
    n_detail = height_to_normal_map(h, strength=float(detail_strength), tileable=tileable, convention=convention)
    return normal_blend_rnm(n_macro, n_detail, strength=1.0)


def normal_bevel(
    mask_or_height: np.ndarray,
    *,
    bevel_px: int = 2,
    strength: float = 3.2,
    roundness: float = 0.65,
    tileable: bool = True,
) -> np.ndarray:
    """Convert any hard mask or height island into a smoothly rounded beveled tangent-space NORMAL map."""
    h = np.asarray(mask_or_height, dtype=np.float32)
    if h.ndim == 3:
        h = h[..., 0]
    b = _box_blur_2d(np.clip(h, 0.0, 1.0), max(1, int(bevel_px)), tileable=tileable)
    rnd = float(np.clip(roundness, 0.1, 1.0))
    shaped = np.power(np.clip(b, 0.0, 1.0), 1.0 - 0.55 * rnd)
    return height_to_normal_map(shaped, strength=float(strength), tileable=tileable, convention="opengl")


def normal_to_curvature(
    normal_rgb: np.ndarray,
    *,
    strength: float = 2.0,
    tileable: bool = True,
) -> np.ndarray:
    """Extract convex (>0.5) / concave (<0.5) curvature directly from any tangent-space NORMAL map."""
    n = np.asarray(normal_rgb, dtype=np.float32)[..., :3] * 2.0 - 1.0
    mode = "wrap" if tileable else "edge"
    px = np.pad(n[..., 0], 1, mode=mode)
    py = np.pad(n[..., 1], 1, mode=mode)
    dnx_dx = (px[1:-1, 2:] - px[1:-1, :-2]) * 0.5
    dny_dy = (py[:-2, 1:-1] - py[2:, 1:-1]) * 0.5
    div = dnx_dx + dny_dy
    return np.clip(0.5 + div * float(strength) * 0.5, 0.0, 1.0).astype(np.float32)


def normal_strength_op(
    normal_rgb: np.ndarray,
    *,
    strength: float = 1.0,
    flip_y: bool = False,
) -> np.ndarray:
    """Adjust tangent-space normal intensity and optionally flip Green channel (OpenGL <-> DirectX)."""
    n = np.asarray(normal_rgb, dtype=np.float32)[..., :3] * 2.0 - 1.0
    nx = n[..., 0] * float(strength)
    ny = n[..., 1] * float(strength) * (-1.0 if flip_y else 1.0)
    nz = np.maximum(0.05, n[..., 2])
    inv_l = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-12)
    out = np.stack((nx * inv_l * 0.5 + 0.5, ny * inv_l * 0.5 + 0.5, nz * inv_l * 0.5 + 0.5), axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# 5. DYNAMIC ANIMATION FX (LIVE TIME-AWARE!)
# ===========================================================================


def anim_shimmer(
    src: dict[str, np.ndarray] | np.ndarray,
    *,
    angle_deg: float = -35.0,
    speed: float = 1.0,
    width: float = 0.16,
    intensity: float = 0.55,
    color: str = "#FFF7D6",
    phase: float = 0.0,
    tileable: bool = True,
) -> dict[str, np.ndarray] | np.ndarray:
    """Sweeping animated metallic/crystal specular shimmer glint across RGBA or full MATERIAL bus."""
    is_bus = is_material_bus(src)
    base = src["albedo"] if is_bus else np.asarray(src, dtype=np.float32)
    H, W = base.shape[:2]
    u, v = uv_grid(H, W)
    th = math.radians(float(angle_deg))
    proj = (u * math.cos(th) + v * math.sin(th) - float(phase) * float(speed)) % 1.0
    d = np.minimum(proj, 1.0 - proj)
    bw = max(0.02, float(width))
    glint = np.exp(-((d / bw) ** 2)).astype(np.float32) * float(np.clip(intensity, 0.0, 1.5))
    c_rgb = _hex_rgb(color)

    if is_bus:
        b = _ensure_bus(src, tileable=tileable)
        alb = b["albedo"].copy()
        alb[..., :3] = np.clip(alb[..., :3] + c_rgb[None, None, :] * glint[..., None] * 0.65, 0.0, 1.0)
        em = b["emissive"].copy()
        em[..., :3] = np.clip(em[..., :3] + c_rgb[None, None, :] * glint[..., None], 0.0, 1.0)
        em[..., 3] = np.clip(np.maximum(em[..., 3], glint), 0.0, 1.0)
        return material_make(
            alb, height=b["height"], normal=b["normal"], roughness=b["roughness"], metallic=b["metallic"],
            ao=b["ao"], shadow=b["shadow"], bent_normal=b["bent_normal"], thickness=b["thickness"],
            emissive=em, sss=b["sss"], porosity=b["porosity"], tileable=tileable,
        )

    if base.ndim == 2:
        return np.clip(base + glint, 0.0, 1.0).astype(np.float32)
    out = base.copy()
    out[..., :3] = np.clip(out[..., :3] + c_rgb[None, None, :] * glint[..., None], 0.0, 1.0)
    return out


def anim_flicker_pulse(
    src: dict[str, np.ndarray] | np.ndarray,
    *,
    min_val: float = 0.55,
    max_val: float = 1.25,
    freq: float = 2.0,
    irregularity: float = 0.35,
    seed: int = 0,
    phase: float = 0.0,
    tileable: bool = True,
) -> dict[str, np.ndarray] | np.ndarray:
    """Organic multi-harmonic animated fire/arcane/crystal flicker pulse."""
    t = float(phase) * 2.0 * math.pi
    s = float(seed % 53) * 0.21
    irr = float(np.clip(irregularity, 0.0, 1.0))
    f = max(1.0, round(float(freq)))
    w = (
        0.5
        + 0.32 * math.sin(f * t + s)
        + 0.12 * irr * math.sin(f * 3.0 * t + s * 2.1)
        + 0.06 * irr * math.cos(f * 5.0 * t - s * 1.4)
    )
    gain = float(min_val) + float(max_val - min_val) * float(np.clip(w, 0.0, 1.0))

    if is_material_bus(src):
        b = _ensure_bus(src, tileable=tileable)
        em = b["emissive"].copy()
        em[..., :3] = np.clip(em[..., :3] * gain, 0.0, 1.0)
        alb = b["albedo"].copy()
        alb[..., :3] = np.clip(alb[..., :3] * (0.75 + 0.25 * gain), 0.0, 1.0)
        return material_make(
            alb, height=b["height"], normal=b["normal"], roughness=b["roughness"], metallic=b["metallic"],
            ao=b["ao"], shadow=b["shadow"], bent_normal=b["bent_normal"], thickness=b["thickness"],
            emissive=em, sss=b["sss"], porosity=b["porosity"], tileable=tileable,
        )
    arr = np.asarray(src, dtype=np.float32)
    out = arr.copy()
    if out.ndim == 3 and out.shape[2] == 4:
        out[..., :3] = np.clip(out[..., :3] * gain, 0.0, 1.0)
    else:
        out = np.clip(out * gain, 0.0, 1.0)
    return out.astype(np.float32)


def anim_flow_warp(
    src: dict[str, np.ndarray] | np.ndarray,
    *,
    strength: float = 0.06,
    scale: float = 4.0,
    speed: float = 1.0,
    angle_deg: float = 0.0,
    seed: int = 0,
    phase: float = 0.0,
    tileable: bool = True,
) -> dict[str, np.ndarray] | np.ndarray:
    """Seamless looping animated UV ripple / water / heat-haze distortion warp."""
    is_bus = is_material_bus(src)
    ref = src["albedo"] if is_bus else np.asarray(src, dtype=np.float32)
    H, W = ref.shape[:2]
    u, v = uv_grid(H, W)
    dx = fbm_noise(H, W, scale=scale, octaves=3, seed=seed + 11, tileable=tileable, phase=phase, loop_speed=speed) - 0.5
    dy = fbm_noise(H, W, scale=scale, octaves=3, seed=seed + 67, tileable=tileable, phase=phase, loop_speed=-speed) - 0.5
    th = math.radians(float(angle_deg))
    st = float(strength)
    drift_u = float(phase) * float(speed) * 0.12 * math.cos(th)
    drift_v = float(phase) * float(speed) * 0.12 * math.sin(th)
    u2 = (u + dx * st + drift_u) % 1.0
    v2 = (v + dy * st + drift_v) % 1.0

    if is_bus:
        b = _ensure_bus(src, tileable=tileable)
        return material_make(
            sample_field_bilinear(b["albedo"], u2, v2, tileable=tileable),
            height=sample_field_bilinear(b["height"], u2, v2, tileable=tileable),
            normal=sample_field_bilinear(b["normal"], u2, v2, tileable=tileable),
            roughness=sample_field_bilinear(b["roughness"], u2, v2, tileable=tileable),
            metallic=sample_field_bilinear(b["metallic"], u2, v2, tileable=tileable),
            ao=sample_field_bilinear(b["ao"], u2, v2, tileable=tileable),
            emissive=sample_field_bilinear(b["emissive"], u2, v2, tileable=tileable),
            sss=sample_field_bilinear(b["sss"], u2, v2, tileable=tileable),
            tileable=tileable,
        )
    return sample_field_bilinear(ref, u2, v2, tileable=tileable).astype(np.float32)
