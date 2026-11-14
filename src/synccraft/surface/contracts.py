"""Typed contracts for P1-03a stateful surface-material programs and world states.

Separates four distinct representations outside ``uv_packer.py``:
1. ``MaterialProgram``: substrate family, coatings, grain/strata axes, RD/biofilm
   parameters, artist stamps/overrides, and deterministic seeds.
2. ``PersistentSurfaceState``: rest-space fields (moisture, oxidation, coating
   damage, sliding wear, biofilm, reaction-diffusion U/V) with explicit geometry
   remapping/reset policies.
3. ``OpticalFields``: linear-space base color, micro-height, tangent-space normal,
   perceptual roughness (alpha = perceptual_roughness**2), metalness, distinct
   dielectric F0, porosity, subsurface, opacity, emissive, posed cavity AO, and
   intrinsic thin-film thickness/IOR.
4. ``StyleConfig``: OKLCH palette ramp settings, pixel-art stylization, reference
   view angle for static thin-film bakes, lighting, and target encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from typing import Any

import numpy as np

from synccraft.dsl.ast_nodes import FACES, Palette, Program, Vec3

MATERIAL_FAMILIES: frozenset[str] = frozenset({
    "painted_metal",
    "brushed_metal",
    "polymer_dielectric",
    "woven_fabric",
    "solid_wood",
    "weathered_stone",
    "keratin_scales",
    "patterned_skin",
    "lichen_biofilm",
    "thin_film_iridescence",
})

FIDELITY_BY_FAMILY: dict[str, str] = {
    "painted_metal": "layered_conductor_dielectric_with_reduced_wear_oxidation",
    "brushed_metal": "anisotropic_conductor_with_reduced_oxidation",
    "polymer_dielectric": "dielectric_microfacet_subsurface",
    "woven_fabric": "orthotropic_yarn_weave_microfacet",
    "solid_wood": "rest_3d_orthotropic_growth_ring_solid",
    "weathered_stone": "rest_3d_voronoi_strata_solid_with_reduced_patina",
    "keratin_scales": "imbricated_cellular_plate_microfacet",
    "patterned_skin": "supported_pattern_mechanism_gray_scott_rd",
    "lichen_biofilm": "exposure_moisture_gated_colonization_reduced_model",
    "thin_film_iridescence": "single_layer_airy_interference_optics",
}

SYMMETRY_MODES: frozenset[str] = frozenset({"independent_seed", "mirrored"})
STAMP_SHAPES: frozenset[str] = frozenset({"rect", "disc", "stripe_u", "stripe_v", "cross"})
TARGET_ENCODINGS: frozenset[str] = frozenset({"vanilla", "labpbr_1_3", "gltf_2_0"})


# ---------------------------------------------------------------------------
# Perceptual OKLCH <-> Linear sRGB <-> Display sRGB Color Space Utilities
# ---------------------------------------------------------------------------


def oklch_to_linear_srgb(lch: Vec3 | np.ndarray) -> np.ndarray:
    """Convert OKLCH (L in [0,1], C >= 0, H in degrees) to Linear sRGB in [0,1]."""
    arr = np.asarray(lch, dtype=np.float64)
    lightness = arr[..., 0]
    chroma = arr[..., 1]
    hue_rad = np.deg2rad(arr[..., 2])
    a = chroma * np.cos(hue_rad)
    b = chroma * np.sin(hue_rad)

    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3

    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    b_lin = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    out = np.stack((r, g, b_lin), axis=-1)
    return np.clip(out, 0.0, 1.0)


def linear_srgb_to_oklch(rgb_linear: np.ndarray) -> np.ndarray:
    """Convert Linear sRGB [..., 3] in [0,1] to OKLCH (L, C, H_degrees)."""
    rgb = np.clip(np.asarray(rgb_linear, dtype=np.float64), 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    l_ = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m_ = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s_ = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_cbrt = np.cbrt(np.maximum(0.0, l_))
    m_cbrt = np.cbrt(np.maximum(0.0, m_))
    s_cbrt = np.cbrt(np.maximum(0.0, s_))

    L = 0.2104542553 * l_cbrt + 0.7936177850 * m_cbrt - 0.0040720468 * s_cbrt
    A = 1.9779984951 * l_cbrt - 2.4285922050 * m_cbrt + 0.4505937099 * s_cbrt
    B = 0.0259040371 * l_cbrt + 0.7827717662 * m_cbrt - 0.8086757660 * s_cbrt

    C = np.hypot(A, B)
    H = np.rad2deg(np.arctan2(B, A)) % 360.0
    return np.stack((np.clip(L, 0.0, 1.0), C, H), axis=-1)


def linear_srgb_to_oklab(rgb_linear: np.ndarray) -> np.ndarray:
    """Convert Linear sRGB [..., 3] to Cartesian Oklab (L, a, b)."""
    rgb = np.clip(np.asarray(rgb_linear, dtype=np.float64), 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l_ = np.cbrt(np.maximum(0.0, 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b))
    m_ = np.cbrt(np.maximum(0.0, 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b))
    s_ = np.cbrt(np.maximum(0.0, 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b))
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return np.stack((L, A, B), axis=-1)


def linear_to_srgb_u8(rgb_linear: np.ndarray) -> np.ndarray:
    """Encode Linear sRGB [..., 3] in [0,1] to uint8 display sRGB [0..255]."""
    v = np.clip(np.asarray(rgb_linear, dtype=np.float64), 0.0, 1.0)
    encoded = np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.power(np.maximum(v, 1e-12), 1.0 / 2.4) - 0.055)
    return np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)


def srgb_u8_to_linear(rgb_u8: np.ndarray) -> np.ndarray:
    """Decode uint8 display sRGB [..., 3] to Linear sRGB in [0,1]."""
    v = np.clip(np.asarray(rgb_u8, dtype=np.float64) / 255.0, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def ior_to_dielectric_f0(ior: float) -> float:
    """Fresnel normal-incidence reflectance F0 = ((n - 1) / (n + 1))^2."""
    n = max(1.0, float(ior))
    ratio = (n - 1.0) / (n + 1.0)
    return float(ratio * ratio)


# ---------------------------------------------------------------------------
# 1. Material Program Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialSpec:
    """Authoring specification for one physical/biological material process."""

    name: str
    family: str = "painted_metal"
    substrate_lch: Vec3 = (0.72, 0.02, 240.0)
    secondary_lch: Vec3 = (0.42, 0.08, 45.0)
    coating_lch: Vec3 = (0.48, 0.14, 235.0)
    oxide_lch: Vec3 = (0.44, 0.11, 42.0)
    coating_thickness: float = 0.65
    perceptual_roughness: float = 0.35
    substrate_roughness: float = 0.22
    metalness: float = 0.0
    substrate_metalness: float = 1.0
    dielectric_ior: float = 1.50
    porosity: float = 0.08
    subsurface: float = 0.0
    emissive_strength: float = 0.0
    grain_axis: Vec3 = (0.0, 1.0, 0.0)
    feature_scale: float = 1.0
    anisotropy: float = 1.0
    # Reaction-diffusion parameters (physical units: model_units^2 / unit_time)
    rd_du: float = 0.08
    rd_dv: float = 0.04
    rd_feed: float = 0.036
    rd_kill: float = 0.062
    rd_initial_time: float = 22.0
    countershade_strength: float = 0.25
    extremity_darkening: float = 0.12
    # Thin-film optical parameters
    film_thickness_nm: float = 380.0
    film_ior: float = 1.45
    # Reduced-order state susceptibility coefficients
    wear_susceptibility: float = 0.35
    oxidation_rate: float = 0.25
    biofilm_affinity: float = 0.20
    # Continuity & symmetry controls
    symmetry_mode: str = "independent_seed"
    weld_group: str | None = None
    labpbr_metal_code: int = 230  # 230=Iron, 231=Gold, 234=Copper, 255=Albedo-tinted metal
    seed: int = 101

    def __post_init__(self) -> None:
        if self.family not in MATERIAL_FAMILIES:
            raise ValueError(f"Unknown material family {self.family!r}; expected one of {sorted(MATERIAL_FAMILIES)}")
        if self.symmetry_mode not in SYMMETRY_MODES:
            raise ValueError(f"Unknown symmetry_mode {self.symmetry_mode!r}")
        if not (0.02 <= self.perceptual_roughness <= 1.0) or not (0.02 <= self.substrate_roughness <= 1.0):
            raise ValueError("perceptual_roughness and substrate_roughness must lie in [0.02, 1.0]")
        if not (0.0 <= self.metalness <= 1.0) or not (0.0 <= self.substrate_metalness <= 1.0):
            raise ValueError("metalness values must lie in [0.0, 1.0]")
        if self.dielectric_ior < 1.0 or self.film_ior < 1.0:
            raise ValueError("IOR must be >= 1.0")
        if self.feature_scale <= 0.0:
            raise ValueError("feature_scale must be > 0")

    @property
    def fidelity_label(self) -> str:
        return FIDELITY_BY_FAMILY[self.family]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "fidelity_label": self.fidelity_label,
            "substrate_lch": list(self.substrate_lch),
            "secondary_lch": list(self.secondary_lch),
            "coating_lch": list(self.coating_lch),
            "oxide_lch": list(self.oxide_lch),
            "coating_thickness": self.coating_thickness,
            "perceptual_roughness": self.perceptual_roughness,
            "substrate_roughness": self.substrate_roughness,
            "metalness": self.metalness,
            "substrate_metalness": self.substrate_metalness,
            "dielectric_ior": self.dielectric_ior,
            "porosity": self.porosity,
            "subsurface": self.subsurface,
            "emissive_strength": self.emissive_strength,
            "grain_axis": list(self.grain_axis),
            "feature_scale": self.feature_scale,
            "anisotropy": self.anisotropy,
            "rd_du": self.rd_du,
            "rd_dv": self.rd_dv,
            "rd_feed": self.rd_feed,
            "rd_kill": self.rd_kill,
            "rd_initial_time": self.rd_initial_time,
            "countershade_strength": self.countershade_strength,
            "extremity_darkening": self.extremity_darkening,
            "film_thickness_nm": self.film_thickness_nm,
            "film_ior": self.film_ior,
            "wear_susceptibility": self.wear_susceptibility,
            "oxidation_rate": self.oxidation_rate,
            "biofilm_affinity": self.biofilm_affinity,
            "symmetry_mode": self.symmetry_mode,
            "weld_group": self.weld_group,
            "labpbr_metal_code": self.labpbr_metal_code,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialSpec":
        payload = {k: v for k, v in data.items() if k != "fidelity_label"}
        for vec_key in ("substrate_lch", "secondary_lch", "coating_lch", "oxide_lch", "grain_axis"):
            if vec_key in payload and isinstance(payload[vec_key], list):
                payload[vec_key] = tuple(float(x) for x in payload[vec_key])
        return cls(**payload)


@dataclass(frozen=True)
class StampOverride:
    """Resolution-independent artist marking/stamp in rest face coordinates (model units)."""

    box_name: str
    face: str
    u_min: float
    v_min: float
    u_max: float
    v_max: float
    lch: Vec3
    shape: str = "rect"
    perceptual_roughness: float | None = None
    metalness: float | None = None
    emissive: float | None = None
    height_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.face not in FACES:
            raise ValueError(f"Invalid stamp face {self.face!r}")
        if self.shape not in STAMP_SHAPES:
            raise ValueError(f"Invalid stamp shape {self.shape!r}")
        if self.u_max <= self.u_min or self.v_max <= self.v_min:
            raise ValueError("Stamp u_max > u_min and v_max > v_min required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_name": self.box_name,
            "face": self.face,
            "u_min": self.u_min,
            "v_min": self.v_min,
            "u_max": self.u_max,
            "v_max": self.v_max,
            "lch": list(self.lch),
            "shape": self.shape,
            "perceptual_roughness": self.perceptual_roughness,
            "metalness": self.metalness,
            "emissive": self.emissive,
            "height_delta": self.height_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StampOverride":
        payload = dict(data)
        payload["lch"] = tuple(float(x) for x in payload["lch"])
        return cls(**payload)


@dataclass(frozen=True)
class MaterialProgram:
    """Complete material program mapping Program palette identifiers to physical specs."""

    materials: dict[str, MaterialSpec]
    stamps: tuple[StampOverride, ...] = ()

    @classmethod
    def default_from_palettes(cls, palettes: list[Palette], seed: int = 101) -> "MaterialProgram":
        specs: dict[str, MaterialSpec] = {}
        for idx, pal in enumerate(palettes):
            specs[pal.name] = MaterialSpec(
                name=pal.name,
                family="polymer_dielectric",
                substrate_lch=pal.base_lch,
                secondary_lch=pal.base_lch,
                coating_lch=pal.base_lch,
                coating_thickness=0.0,
                perceptual_roughness=0.55,
                metalness=0.0,
                emissive_strength=1.0 if pal.emissive else 0.0,
                seed=seed + idx * 37,
            )
        return cls(materials=specs)

    @classmethod
    def from_program(cls, program: Program, seed: int = 101) -> "MaterialProgram":
        """Build MaterialProgram from Program palettes + optional @materials AST declarations."""
        base = cls.default_from_palettes(program.palettes, seed=seed)
        if not program.materials:
            return base
        pal_map = {p.name: p for p in program.palettes}
        specs = dict(base.materials)
        for mdecl in program.materials:
            pal = pal_map[mdecl.name]
            l_val, c_val, h_val = pal.base_lch
            sec_lch = (max(0.05, l_val * 0.65), min(0.35, c_val * 1.2 + 0.03), (h_val + pal.hue_shift[0]) % 360.0)
            specs[mdecl.name] = MaterialSpec(
                name=mdecl.name,
                family=mdecl.family,
                substrate_lch=(0.74, 0.02, 240.0) if mdecl.family == "painted_metal" else pal.base_lch,
                secondary_lch=sec_lch,
                coating_lch=pal.base_lch,
                coating_thickness=mdecl.coating,
                perceptual_roughness=mdecl.roughness,
                metalness=mdecl.metalness,
                substrate_metalness=1.0 if mdecl.family in ("painted_metal", "brushed_metal") else mdecl.metalness,
                dielectric_ior=mdecl.ior,
                porosity=mdecl.porosity,
                subsurface=mdecl.subsurface,
                feature_scale=mdecl.scale,
                anisotropy=mdecl.anisotropy,
                rd_feed=mdecl.rd_feed,
                rd_kill=mdecl.rd_kill,
                film_thickness_nm=mdecl.film_nm,
                wear_susceptibility=mdecl.wear,
                oxidation_rate=mdecl.oxidation,
                biofilm_affinity=mdecl.biofilm,
                emissive_strength=max(mdecl.emissive, 1.0 if pal.emissive else 0.0),
                weld_group=mdecl.weld_group,
                seed=mdecl.seed,
            )
        return cls(materials=specs)

    def spec_for(self, material_name: str, fallback_palette: Palette | None = None) -> MaterialSpec:
        if material_name in self.materials:
            return self.materials[material_name]
        if fallback_palette is not None:
            return MaterialSpec(
                name=material_name,
                family="polymer_dielectric",
                substrate_lch=fallback_palette.base_lch,
                secondary_lch=fallback_palette.base_lch,
                coating_lch=fallback_palette.base_lch,
                coating_thickness=0.0,
                perceptual_roughness=0.55,
                metalness=0.0,
                emissive_strength=1.0 if fallback_palette.emissive else 0.0,
            )
        raise KeyError(f"Material {material_name!r} not found in MaterialProgram")

    def with_material(self, spec: MaterialSpec) -> "MaterialProgram":
        updated = dict(self.materials)
        updated[spec.name] = spec
        return replace(self, materials=updated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "materials": {k: v.to_dict() for k, v in sorted(self.materials.items())},
            "stamps": [s.to_dict() for s in self.stamps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MaterialProgram":
        mats = {k: MaterialSpec.from_dict(v) for k, v in data.get("materials", {}).items()}
        stamps = tuple(StampOverride.from_dict(s) for s in data.get("stamps", []))
        return cls(materials=mats, stamps=stamps)


# ---------------------------------------------------------------------------
# 2. Persistent Surface State (Rest-Space Fields)
# ---------------------------------------------------------------------------

FaceKey = tuple[str, str]  # (box_name, face_name)


@dataclass
class FacePersistentPatch:
    """Mutable or copy-on-write rest-space state arrays for one (box, face)."""

    moisture: np.ndarray  # [0, 1]
    oxidation: np.ndarray  # [0, 1]
    coating_damage: np.ndarray  # [0, +inf) abraded thickness
    wear: np.ndarray  # [0, +inf) cumulative Archard wear
    biofilm: np.ndarray  # [0, 1] lichen/biofilm colonization
    rd_u: np.ndarray  # [0, 1] activator/substrate concentration U
    rd_v: np.ndarray  # [0, 1] inhibitor/morphogen concentration V
    rd_initialized: bool = False

    @classmethod
    def zeros(cls, shape: tuple[int, int]) -> "FacePersistentPatch":
        return cls(
            moisture=np.zeros(shape, dtype=np.float64),
            oxidation=np.zeros(shape, dtype=np.float64),
            coating_damage=np.zeros(shape, dtype=np.float64),
            wear=np.zeros(shape, dtype=np.float64),
            biofilm=np.zeros(shape, dtype=np.float64),
            rd_u=np.ones(shape, dtype=np.float64),
            rd_v=np.zeros(shape, dtype=np.float64),
            rd_initialized=False,
        )

    def copy(self) -> "FacePersistentPatch":
        return FacePersistentPatch(
            moisture=self.moisture.copy(),
            oxidation=self.oxidation.copy(),
            coating_damage=self.coating_damage.copy(),
            wear=self.wear.copy(),
            biofilm=self.biofilm.copy(),
            rd_u=self.rd_u.copy(),
            rd_v=self.rd_v.copy(),
            rd_initialized=bool(self.rd_initialized),
        )


@dataclass
class PersistentSurfaceState:
    """Persistent surface state across all (box, face) patches in rest coordinates."""

    subdivisions_per_unit: int
    patches: dict[FaceKey, FacePersistentPatch] = field(default_factory=dict)

    def copy(self) -> "PersistentSurfaceState":
        return PersistentSurfaceState(
            subdivisions_per_unit=self.subdivisions_per_unit,
            patches={k: v.copy() for k, v in self.patches.items()},
        )

    def summary_stats(self) -> dict[str, Any]:
        """Compact per-box summary suitable for LLM World-Model prompts and diffs."""
        by_box: dict[str, dict[str, float]] = {}
        for (box_name, _), patch in sorted(self.patches.items()):
            acc = by_box.setdefault(
                box_name,
                {"mean_wear": 0.0, "max_wear": 0.0, "mean_oxidation": 0.0, "mean_moisture": 0.0, "mean_biofilm": 0.0, "faces": 0.0},
            )
            acc["mean_wear"] += float(patch.wear.mean())
            acc["max_wear"] = max(acc["max_wear"], float(patch.wear.max()))
            acc["mean_oxidation"] += float(patch.oxidation.mean())
            acc["mean_moisture"] += float(patch.moisture.mean())
            acc["mean_biofilm"] += float(patch.biofilm.mean())
            acc["faces"] += 1.0
        for stats in by_box.values():
            cnt = max(1.0, stats.pop("faces"))
            stats["mean_wear"] = round(stats["mean_wear"] / cnt, 5)
            stats["max_wear"] = round(stats["max_wear"], 5)
            stats["mean_oxidation"] = round(stats["mean_oxidation"] / cnt, 5)
            stats["mean_moisture"] = round(stats["mean_moisture"] / cnt, 5)
            stats["mean_biofilm"] = round(stats["mean_biofilm"] / cnt, 5)
        return by_box

    def state_sha256(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"subdiv:{self.subdivisions_per_unit}\n".encode("utf-8"))
        for key in sorted(self.patches):
            p = self.patches[key]
            hasher.update(f"{key[0]}:{key[1]}:{int(p.rd_initialized)}:".encode("utf-8"))
            for arr in (p.moisture, p.oxidation, p.coating_damage, p.wear, p.biofilm, p.rd_u, p.rd_v):
                hasher.update(np.round(arr, 8).tobytes())
        return hasher.hexdigest()


# ---------------------------------------------------------------------------
# 3. Intrinsic Optical Fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceOpticalPatch:
    """Solver- or pixel-resolution intrinsic physical optical fields for one face."""

    base_color_linear: np.ndarray  # (H, W, 3) linear sRGB in [0, 1]
    height: np.ndarray  # (H, W) micro-relief in model units
    tangent_normal: np.ndarray  # (H, W, 3) unit normal in (T_right, B_up, N_out)
    perceptual_roughness: np.ndarray  # (H, W) alpha_p in [0.02, 1.0]; GGX alpha = alpha_p^2
    metalness: np.ndarray  # (H, W) in [0, 1]
    dielectric_f0: np.ndarray  # (H, W) in [0, 0.25], distinct from metalness
    porosity: np.ndarray  # (H, W) in [0, 1]
    subsurface: np.ndarray  # (H, W) in [0, 1]
    opacity: np.ndarray  # (H, W) in [0, 1]
    emissive: np.ndarray  # (H, W) in [0, 1]
    cavity_ao: np.ndarray  # (H, W) posed visibility in [0, 1]
    iridescence_thickness_nm: np.ndarray  # (H, W) >= 0 (0 = no thin-film)
    iridescence_ior: np.ndarray  # (H, W) >= 1.0
    labpbr_metal_code: int = 230


@dataclass(frozen=True)
class OpticalFields:
    """Intrinsic optical fields across all (box, face) patches."""

    subdivisions_per_unit: float
    patches: dict[FaceKey, FaceOpticalPatch]


# ---------------------------------------------------------------------------
# 4. Style / Export Settings & World State
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StyleConfig:
    """Stylization, lighting, and target encoding configuration."""

    quantize_to_palette_ramp: bool = True
    bevel_highlight_strength: float = 0.08
    vanilla_reference_view_deg: float = 20.0
    target_encoding: str = "vanilla"

    def __post_init__(self) -> None:
        if self.target_encoding not in TARGET_ENCODINGS:
            raise ValueError(f"Unsupported target_encoding {self.target_encoding!r}")

    @classmethod
    def from_program(cls, program: Program) -> "StyleConfig":
        if program.style is None:
            return cls()
        s = program.style
        return cls(
            quantize_to_palette_ramp=s.quantize_ramp,
            bevel_highlight_strength=s.bevel_highlight,
            vanilla_reference_view_deg=s.view_deg,
            target_encoding=s.target_encoding,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantize_to_palette_ramp": self.quantize_to_palette_ramp,
            "bevel_highlight_strength": self.bevel_highlight_strength,
            "vanilla_reference_view_deg": self.vanilla_reference_view_deg,
            "target_encoding": self.target_encoding,
        }


@dataclass(frozen=True)
class EnvironmentState:
    """External world & illumination conditions E_t."""

    humidity: float = 0.35
    oxygen_activity: float = 0.21
    temperature_c: float = 20.0
    spore_load: float = 0.0
    light_dir_world: Vec3 = (0.45, 0.75, 0.48)
    ambient_intensity: float = 0.42
    sun_intensity: float = 0.78

    def normalized_light_dir(self) -> np.ndarray:
        v = np.asarray(self.light_dir_world, dtype=np.float64)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else np.array((0.0, 1.0, 0.0), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "humidity": self.humidity,
            "oxygen_activity": self.oxygen_activity,
            "temperature_c": self.temperature_c,
            "spore_load": self.spore_load,
            "light_dir_world": list(self.light_dir_world),
            "ambient_intensity": self.ambient_intensity,
            "sun_intensity": self.sun_intensity,
        }


@dataclass
class WorldState:
    """Unified stateful world representation S_t = (Geometry, Pose, Materials, SurfaceState, Style, Env)."""

    program: Program
    material_program: MaterialProgram
    surface_state: PersistentSurfaceState
    pose_euler_deltas: dict[str, Vec3] = field(default_factory=dict)
    style_config: StyleConfig = field(default_factory=StyleConfig)
    environment: EnvironmentState = field(default_factory=EnvironmentState)
    time_sec: float = 0.0
    step_index: int = 0

    def compact_summary(self) -> dict[str, Any]:
        """Structured summary for LLM World-Model conditioning and trajectory logs."""
        return {
            "step_index": self.step_index,
            "time_sec": round(self.time_sec, 4),
            "archetype": self.program.meta.archetype,
            "texel_density": self.program.meta.texel_density,
            "pose_euler_deltas": {k: [round(x, 3) for x in v] for k, v in sorted(self.pose_euler_deltas.items())},
            "materials": {
                k: {
                    "family": v.family,
                    "coating_thickness": v.coating_thickness,
                    "perceptual_roughness": v.perceptual_roughness,
                    "metalness": v.metalness,
                }
                for k, v in sorted(self.material_program.materials.items())
            },
            "environment": self.environment.to_dict(),
            "surface_state_summary": self.surface_state.summary_stats(),
            "surface_state_sha256": self.surface_state.state_sha256(),
        }

    def to_json_metadata(self) -> str:
        return json.dumps(self.compact_summary(), sort_keys=True, indent=2)
