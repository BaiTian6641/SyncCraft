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

FIDELITY_TIERS: frozenset[str] = frozenset({
    "physical_law_validated",
    "reduced_order_qualitative",
    "stylistic",
})

FIDELITY_BY_FAMILY: dict[str, str] = {
    # Exact optical laws validated against analytical/quadrature references (M5/M6):
    # - Cook-Torrance GGX microfacet + Height-Correlated Smith G2 + Fresnel F0(n)
    #   (Walter et al., EGSR 2007; Heitz, JCGT 2014)
    # - Single-layer Airy interference Δ = 2 n_film d cos(θ_t) + λ/2 (Born & Wolf 1999, §7.6)
    "polymer_dielectric": "physical_law_validated",
    "thin_film_iridescence": "physical_law_validated",
    # Reduced-order continuum / reaction-diffusion / cellular material models (M5/M6):
    "painted_metal": "reduced_order_qualitative",
    "brushed_metal": "reduced_order_qualitative",
    "woven_fabric": "reduced_order_qualitative",
    "solid_wood": "reduced_order_qualitative",
    "weathered_stone": "reduced_order_qualitative",
    "keratin_scales": "reduced_order_qualitative",
    "patterned_skin": "reduced_order_qualitative",
    "lichen_biofilm": "reduced_order_qualitative",
}

# Literature dielectric refractive indices n (at ~589 nm sodium D-line) and exact
# normal-incidence Fresnel reflectance F0 = ((n - 1) / (n + 1))^2:
DIELECTRIC_IOR_REFERENCE_TABLE: dict[str, dict[str, float | str]] = {
    "water_20c": {
        "ior": 1.333,
        "f0": round(((1.333 - 1.0) / (1.333 + 1.0)) ** 2, 6),
        "source": "Hale & Querry, Appl. Opt. 12(3):555-563 (1973)",
    },
    "fused_silica_sio2": {
        "ior": 1.458,
        "f0": round(((1.458 - 1.0) / (1.458 + 1.0)) ** 2, 6),
        "source": "Malitson, J. Opt. Soc. Am. 55(10):1205-1209 (1965)",
    },
    "acrylic_pmma_clearcoat": {
        "ior": 1.500,
        "f0": round(((1.500 - 1.0) / (1.500 + 1.0)) ** 2, 6),
        "source": "Beadie et al., Appl. Opt. 54(31):F139-F143 (2015); Burley, Disney Principled BRDF (2012)",
    },
    "crown_silicate_glass": {
        "ior": 1.520,
        "f0": round(((1.520 - 1.0) / (1.520 + 1.0)) ** 2, 6),
        "source": "Born & Wolf, Principles of Optics, 7th ed., §1.5 (1999)",
    },
    "alpha_keratin_scale": {
        "ior": 1.550,
        "f0": round(((1.550 - 1.0) / (1.550 + 1.0)) ** 2, 6),
        "source": "Leertouwer, Wilts & Stavenga, J. R. Soc. Interface 8(65):1687-1694 (2011)",
    },
    "arthropod_chitin_cuticle": {
        "ior": 1.560,
        "f0": round(((1.560 - 1.0) / (1.560 + 1.0)) ** 2, 6),
        "source": "Leertouwer et al., J. R. Soc. Interface 8(65):1687-1694 (2011)",
    },
}

# Dimensional normalization of Archard's adhesive/abrasive sliding wear law
# (Archard, J. F., "Contact and Rubbing of Flat Surfaces", J. Appl. Phys. 24(8):981-988, 1953):
#   ΔV = K * (F_N * Δs) / H   ==>   wear depth Δh = (K / H) * P_N * Δs
# With SyncCraft length unit 1 u = 1/16 m = 0.0625 m, reference polymer/oxide topcoat
# thickness h_coat,0 = 75 μm (7.5e-5 m), nominal pad contact pressure P_0 = 10 MPa (1.0e7 Pa),
# and reference specific wear rate K_ref / H_ref = 5.4e-11 Pa^-1:
#   c_rub = (K_ref / H_ref) * P_0 * (1 u) / h_coat,0
#         = (5.4e-11 Pa^-1 * 1.0e7 Pa * 0.0625 m) / 7.5e-5 m = 0.45
# For cyclic conformal joint-hinge clearance sliding (AnimateClip), sinusoidal Hertzian
# pressure distribution over a partial hinge arc gives effective engagement factor η_hinge ≈ 0.489,
# yielding c_swept = c_rub * η_hinge ≈ 0.22.
ARCHARD_SWEPT_CONTACT_COUPLING: float = 0.22
ARCHARD_DIRECT_RUBBING_COUPLING: float = 0.45
ARCHARD_WEAR_NORMALIZATION: dict[str, float | str] = {
    "model_unit_meters": 0.0625,
    "reference_coating_thickness_m": 7.5e-5,
    "reference_contact_pressure_pa": 1.0e7,
    "reference_specific_wear_rate_inv_pa": 5.4e-11,
    "direct_rubbing_coupling": ARCHARD_DIRECT_RUBBING_COUPLING,
    "swept_hinge_engagement_factor": round(ARCHARD_SWEPT_CONTACT_COUPLING / ARCHARD_DIRECT_RUBBING_COUPLING, 6),
    "swept_contact_coupling": ARCHARD_SWEPT_CONTACT_COUPLING,
    "source": "Archard, J. F., J. Appl. Phys. 24(8):981-988 (1953)",
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
    substrate_lch: Vec3 = (0.72, 0.02, 240.0)  # OKLCH (L in [0,1], C >= 0, H in deg)
    secondary_lch: Vec3 = (0.42, 0.08, 45.0)
    coating_lch: Vec3 = (0.48, 0.14, 235.0)
    oxide_lch: Vec3 = (0.44, 0.11, 42.0)
    coating_thickness: float = 0.65  # dimensionless in [0,1]; normalized to h_coat,0 = 75 μm topcoat
    perceptual_roughness: float = 0.35  # dimensionless α_p in [0.02,1]; GGX α = α_p^2 (Burley 2012)
    substrate_roughness: float = 0.22  # dimensionless α_p in [0.02,1] of underlying metal/mineral
    metalness: float = 0.0  # dimensionless conductor weight in [0,1]
    substrate_metalness: float = 1.0  # dimensionless substrate conductor weight in [0,1]
    dielectric_ior: float = 1.50  # n at 589 nm (PMMA/polyurethane clearcoat: n=1.50 -> F0=0.040; Beadie 2015)
    porosity: float = 0.08  # connected void fraction φ in [0,1] (wet darkening: Lekner & Dorf 1988)
    subsurface: float = 0.0  # dimensionless wrap-scattering weight in [0,1]
    emissive_strength: float = 0.0  # normalized self-emission in [0,1]
    grain_axis: Vec3 = (0.0, 1.0, 0.0)  # bone-local orthotropic fiber/strata unit direction
    feature_scale: float = 1.0  # inverse spatial wavelength in u^-1 (1 u = 0.0625 m)
    anisotropy: float = 1.0  # directional diffusivity/groove elongation ratio (>= 1 along grain_axis)
    # Gray-Scott reaction-diffusion parameters (Pearson, Science 261:189-192, 1993):
    # D_u, D_v in u^2 / time_unit (D_u : D_v = 2 : 1); F, k in 1 / time_unit.
    rd_du: float = 0.08
    rd_dv: float = 0.04
    rd_feed: float = 0.036
    rd_kill: float = 0.062
    rd_initial_time: float = 22.0  # dimensionless reaction time units
    countershade_strength: float = 0.25  # Thayer (1896) / Rowland (2009) dorsoventral gradient in [0,1]
    extremity_darkening: float = 0.12  # distal axial melanin/acral darkening in [0,1]
    # Single-layer Airy thin-film optical parameters (Born & Wolf 1999, §7.6):
    film_thickness_nm: float = 380.0  # physical film thickness d in nanometers (nm)
    film_ior: float = 1.45  # n_film (hydrated silica/lipid epicuticle ~1.45-1.46; Malitson 1965)
    # Reduced-order aging susceptibility coefficients:
    wear_susceptibility: float = 0.35  # dimensionless Archard (1953) susceptibility; scaled by 0.22/0.45 couplings
    oxidation_rate: float = 0.25  # normalized Wagner/Tammann parabolic-logarithmic oxide rate in [0,1]
    biofilm_affinity: float = 0.20  # moisture/insolation colonization rate in [0,1]
    # Continuity & symmetry controls
    symmetry_mode: str = "independent_seed"
    weld_group: str | None = None
    coat: str | None = None
    labpbr_metal_code: int = 230  # LabPBR 1.3 R-channel: 230=Iron, 231=Gold, 234=Copper, 255=Albedo-tinted metal
    seed: int = 101

    def __post_init__(self) -> None:
        if self.family not in MATERIAL_FAMILIES:
            raise ValueError(f"Unknown material family {self.family!r}; expected one of {sorted(MATERIAL_FAMILIES)}")
        if FIDELITY_BY_FAMILY.get(self.family) not in FIDELITY_TIERS:
            raise ValueError(f"Family {self.family!r} has unstandardized fidelity tier")
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
            "coat": self.coat,
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
        """Build MaterialProgram from Program palettes + optional @materials and @stamps AST declarations."""
        base = cls.default_from_palettes(program.palettes, seed=seed)
        pal_map = {p.name: p for p in program.palettes}
        specs = dict(base.materials)
        for mdecl in program.materials:
            pal = pal_map[mdecl.name]
            l_val, c_val, h_val = pal.base_lch
            sec_lch = (max(0.05, l_val * 0.65), min(0.35, c_val * 1.2 + 0.03), (h_val + pal.hue_shift[0]) % 360.0)
            coat_name = getattr(mdecl, "coat", None)
            rd_f = float(mdecl.rd_feed)
            rd_k = float(mdecl.rd_kill)
            rd_t = 22.0
            aniso = float(mdecl.anisotropy)
            if coat_name == "plain":
                rd_f, rd_k, rd_t = 0.090, 0.059, 0.0
            elif coat_name == "spotted":
                rd_f, rd_k = 0.030, 0.062
            elif coat_name == "striped":
                rd_f, rd_k = 0.042, 0.060
                aniso = max(aniso, 3.2)
            elif coat_name in ("banded_limb", "scale_rows"):
                rd_t = 0.0
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
                anisotropy=aniso,
                rd_feed=rd_f,
                rd_kill=rd_k,
                rd_initial_time=rd_t,
                film_thickness_nm=mdecl.film_nm,
                wear_susceptibility=mdecl.wear,
                oxidation_rate=mdecl.oxidation,
                biofilm_affinity=mdecl.biofilm,
                emissive_strength=max(mdecl.emissive, 1.0 if pal.emissive else 0.0),
                weld_group=mdecl.weld_group,
                coat=coat_name,
                seed=mdecl.seed,
            )

        compiled_stamps: list[StampOverride] = []
        if getattr(program, "stamps", ()):
            from synccraft.kinematics.se3_chain import compile_geometry
            from synccraft.surface.domain import face_uv_extents_units

            rest_sc = compile_geometry(program)
            box_by_name = rest_sc.box_by_name
            upb = float(getattr(program.meta, "units_per_block", 16))
            px_to_u = upb / float(max(1, program.meta.texel_density))
            face_alias_map = {
                "front": "south",
                "back": "north",
                "side": "west",
                "top": "up",
                "bottom": "down",
            }
            opp_side_face = {"west": "east", "east": "west"}

            for stp in program.stamps:
                if stp.target_box not in box_by_name or stp.color not in pal_map:
                    continue
                bx = box_by_name[stp.target_box]
                pal_s = pal_map[stp.color]
                f_resolved = face_alias_map.get(stp.face, stp.face)
                if f_resolved not in FACES:
                    continue
                wu, hv = face_uv_extents_units(bx.size, f_resolved)
                wu = max(px_to_u, float(wu))
                hv = max(px_to_u, float(hv))

                # `size` is specified in texel pixels (or model units when fractional); convert cleanly
                sw_u = min(wu * 0.95, max(px_to_u * 0.8, float(stp.size[0]) * px_to_u))
                sh_v = min(hv * 0.95, max(px_to_u * 0.8, float(stp.size[1]) * px_to_u))

                raw_u, raw_v = float(stp.at[0]), float(stp.at[1])
                base_cu = (raw_u * wu) if (0.0 <= raw_u <= 1.0 and wu > 1.2) else min(wu - 0.5 * sw_u, max(0.5 * sw_u, raw_u))
                base_cv = (raw_v * hv) if (0.0 <= raw_v <= 1.0 and hv > 1.2) else min(hv - 0.5 * sh_v, max(0.5 * sh_v, raw_v))
                em_val = max(float(stp.emissive), 1.0 if pal_s.emissive else 0.0)
                em_opt = em_val if em_val > 0.0 else None

                n_rows = max(1, int(stp.rows))
                step_u = float(stp.spacing) * px_to_u if float(stp.spacing) > 0.0 else (sw_u * 1.6)
                for r_idx in range(n_rows):
                    cu_i = base_cu + (r_idx - 0.5 * (n_rows - 1)) * step_u
                    u0 = max(0.0, min(wu - 1e-4, cu_i - 0.5 * sw_u))
                    u1 = max(u0 + 1e-3, min(wu, cu_i + 0.5 * sw_u))
                    v0 = max(0.0, min(hv - 1e-4, base_cv - 0.5 * sh_v))
                    v1 = max(v0 + 1e-3, min(hv, base_cv + 0.5 * sh_v))
                    compiled_stamps.append(
                        StampOverride(
                            box_name=bx.name,
                            face=f_resolved,
                            u_min=u0,
                            v_min=v0,
                            u_max=u1,
                            v_max=v1,
                            lch=pal_s.base_lch,
                            shape=stp.shape,
                            emissive=em_opt,
                        )
                    )
                    if stp.mirror:
                        # 1. Opposite lateral face on centerline boxes (`west` <-> `east`)
                        if f_resolved in opp_side_face and abs(float(bx.center[0])) <= 0.5:
                            compiled_stamps.append(
                                StampOverride(
                                    box_name=bx.name,
                                    face=opp_side_face[f_resolved],
                                    u_min=max(0.0, wu - u1),
                                    v_min=v0,
                                    u_max=max(max(0.0, wu - u1) + 1e-3, wu - u0),
                                    v_max=v1,
                                    lch=pal_s.base_lch,
                                    shape=stp.shape,
                                    emissive=em_opt,
                                )
                            )
                        # 2. Bilateral mirror on north/south/up/down of a centerline box
                        elif f_resolved in ("north", "south", "up", "down") and abs(float(bx.center[0])) <= 0.5 and abs(cu_i - 0.5 * wu) > 1e-3:
                            mu0 = max(0.0, wu - u1)
                            mu1 = max(mu0 + 1e-3, wu - u0)
                            compiled_stamps.append(
                                StampOverride(
                                    box_name=bx.name,
                                    face=f_resolved,
                                    u_min=mu0,
                                    v_min=v0,
                                    u_max=mu1,
                                    v_max=v1,
                                    lch=pal_s.base_lch,
                                    shape=stp.shape,
                                    emissive=em_opt,
                                )
                            )
                        # 3. Mirrored twin box on bilateral limbs/cheeks
                        for cand_b in rest_sc.boxes:
                            if cand_b.name != bx.name and cand_b.mirrored != bx.mirrored and (
                                cand_b.name.endswith(f"__{bx.name}") or bx.name.endswith(f"__{cand_b.name}")
                            ):
                                mf = opp_side_face.get(f_resolved, f_resolved)
                                compiled_stamps.append(
                                    StampOverride(
                                        box_name=cand_b.name,
                                        face=mf,
                                        u_min=max(0.0, wu - u1),
                                        v_min=v0,
                                        u_max=max(max(0.0, wu - u1) + 1e-3, wu - u0),
                                        v_max=v1,
                                        lch=pal_s.base_lch,
                                        shape=stp.shape,
                                        emissive=em_opt,
                                    )
                                )

        return cls(materials=specs, stamps=tuple(compiled_stamps))

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
    shade_mode: str = "physical"
    banded_steps: int = 4
    banded_direction: str = "down"
    banded_cluster: str = "diagonal"
    banded_top_highlight: bool = True
    banded_contact_shadow: bool = True
    glow_bleed: int = 0

    def __post_init__(self) -> None:
        if self.target_encoding not in TARGET_ENCODINGS:
            raise ValueError(f"Unsupported target_encoding {self.target_encoding!r}")
        if self.shade_mode not in ("physical", "banded"):
            raise ValueError(f"Unsupported shade_mode {self.shade_mode!r}")

    @property
    def fidelity_label(self) -> str:
        return "stylistic"

    @classmethod
    def from_program(cls, program: Program) -> "StyleConfig":
        if program.style is None:
            return cls()
        s = program.style
        bd = getattr(s, "banded", None)
        return cls(
            quantize_to_palette_ramp=s.quantize_ramp,
            bevel_highlight_strength=s.bevel_highlight,
            vanilla_reference_view_deg=s.view_deg,
            target_encoding=s.target_encoding,
            shade_mode=getattr(s, "shade_mode", "physical"),
            banded_steps=int(bd.steps) if bd is not None else 4,
            banded_direction=str(bd.direction) if bd is not None else "down",
            banded_cluster=str(bd.cluster) if bd is not None else "diagonal",
            banded_top_highlight=bool(bd.top_highlight) if bd is not None else True,
            banded_contact_shadow=bool(bd.contact_shadow) if bd is not None else True,
            glow_bleed=int(getattr(s, "glow_bleed", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity_label": self.fidelity_label,
            "quantize_to_palette_ramp": self.quantize_to_palette_ramp,
            "bevel_highlight_strength": self.bevel_highlight_strength,
            "vanilla_reference_view_deg": self.vanilla_reference_view_deg,
            "target_encoding": self.target_encoding,
            "shade_mode": self.shade_mode,
            "banded_steps": self.banded_steps,
            "banded_direction": self.banded_direction,
            "banded_cluster": self.banded_cluster,
            "banded_top_highlight": self.banded_top_highlight,
            "banded_contact_shadow": self.banded_contact_shadow,
            "glow_bleed": self.glow_bleed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StyleConfig":
        payload = {k: v for k, v in data.items() if k != "fidelity_label"}
        return cls(**payload)


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
