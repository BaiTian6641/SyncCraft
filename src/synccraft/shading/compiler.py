"""P1-03c Intrinsic OpticalFields -> Pixel-Art / Stylized Vanilla Compiler.

Implements the explicit four-stage pipeline:
Material processes -> Intrinsic OpticalFields -> Pixel-art decisions -> Target export.

Key invariants:
- Palette ramps are authored in perceptual OKLCH; all lighting calculations
  (Cook-Torrance GGX specular, Lambertian diffuse, posed cavity AO, Airy thin-film)
  occur strictly in linear sRGB color space.
- Perceptual roughness ``alpha_p`` maps to GGX microfacet roughness ``alpha = alpha_p**2``.
- ``metalness`` and ``dielectric_f0`` remain distinct in Schlick Fresnel blending.
- Simulation resolution (``SurfaceDomain.subdivisions_per_unit``) is decoupled from
  target pixel density (``Meta.texel_density``) via exact physical-area box integration.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from synccraft.dsl.ast_nodes import Palette, Program
from synccraft.kinematics.se3_chain import Scene
from synccraft.surface.contracts import (
    EnvironmentState,
    FaceKey,
    FaceOpticalPatch,
    MaterialProgram,
    OpticalFields,
    PersistentSurfaceState,
    StyleConfig,
    WorldState,
    linear_srgb_to_oklab,
    linear_to_srgb_u8,
    oklch_to_linear_srgb,
)
from synccraft.surface.domain import (
    PosedFaceKinematics,
    SurfaceDomain,
    build_surface_domain,
    evaluate_posed_surface_kinematics,
    initialize_surface_state,
)
from synccraft.surface.processes import evaluate_airy_thin_film_linear_rgb, evaluate_optical_fields
from .uv_packer import allocate_uv_layout


def _interval_overlap_matrix(n_out: int, n_in: int, extent_units: float) -> np.ndarray:
    """Exact area-preserving 1D box-filter matrix W of shape (n_out, n_in).

    Each row i averages input cells j weighted by physical interval overlap on [0, extent_units].
    """
    out_edges = np.linspace(0.0, extent_units, n_out + 1, dtype=np.float64)
    in_edges = np.linspace(0.0, extent_units, n_in + 1, dtype=np.float64)
    lo = np.maximum(out_edges[:-1, None], in_edges[None, :-1])
    hi = np.minimum(out_edges[1:, None], in_edges[None, 1:])
    overlap = np.maximum(0.0, hi - lo)
    out_widths = np.maximum(1e-12, out_edges[1:] - out_edges[:-1])[:, None]
    return overlap / out_widths


def _box_downsample_2d(field_2d: np.ndarray, w_row: np.ndarray, w_col: np.ndarray) -> np.ndarray:
    if field_2d.ndim == 2:
        return w_row @ field_2d @ w_col.T
    # (H, W, C)
    return np.einsum("ab,bcd,ec->aed", w_row, field_2d, w_col, optimize=True)


def downsample_optical_fields(
    optical_fields: OpticalFields,
    domain: SurfaceDomain,
    texel_density: int,
) -> OpticalFields:
    """Area-integrate solver-resolution OpticalFields to integer pixel resolution."""
    density = texel_density / 16.0
    out_patches: dict[FaceKey, FaceOpticalPatch] = {}

    is_authored = getattr(domain.rest_scene.meta, "audit_profile", "strict") == "authored"
    for fk, patch in optical_fields.patches.items():
        grid = domain.faces[fk]
        w_is_int = math.isclose(float(grid.width_units), round(float(grid.width_units)), abs_tol=1e-9) and float(grid.width_units) > 0.0
        h_is_int = math.isclose(float(grid.height_units), round(float(grid.height_units)), abs_tol=1e-9) and float(grid.height_units) > 0.0
        fw = float(grid.width_units) * density
        fh = float(grid.height_units) * density
        if not is_authored and (
            (w_is_int and not math.isclose(fw, round(fw), abs_tol=1e-7))
            or (h_is_int and not math.isclose(fh, round(fh), abs_tol=1e-7))
        ):
            raise ValueError(
                f"texel_density={texel_density} produces fractional texels ({fw}x{fh}) on {fk}; integer atlas required"
            )
        w_px = max(1, int(round(fw)))
        h_px = max(1, int(round(fh)))
        in_h, in_w = patch.height.shape

        if (h_px, w_px) == (in_h, in_w):
            out_patches[fk] = patch
            continue

        eff_h_u = max(1e-6, float(grid.height_units))
        eff_w_u = max(1e-6, float(grid.width_units))
        w_row = _interval_overlap_matrix(h_px, in_h, eff_h_u)
        w_col = _interval_overlap_matrix(w_px, in_w, eff_w_u)

        nrm = _box_downsample_2d(patch.tangent_normal, w_row, w_col)
        nrm_len = np.maximum(1e-12, np.linalg.norm(nrm, axis=-1, keepdims=True))
        nrm_unit = nrm / nrm_len

        out_patches[fk] = FaceOpticalPatch(
            base_color_linear=np.clip(_box_downsample_2d(patch.base_color_linear, w_row, w_col), 0.0, 1.0),
            height=_box_downsample_2d(patch.height, w_row, w_col),
            tangent_normal=nrm_unit,
            perceptual_roughness=np.clip(_box_downsample_2d(patch.perceptual_roughness, w_row, w_col), 0.02, 1.0),
            metalness=np.clip(_box_downsample_2d(patch.metalness, w_row, w_col), 0.0, 1.0),
            dielectric_f0=np.clip(_box_downsample_2d(patch.dielectric_f0, w_row, w_col), 0.0, 0.25),
            porosity=np.clip(_box_downsample_2d(patch.porosity, w_row, w_col), 0.0, 1.0),
            subsurface=np.clip(_box_downsample_2d(patch.subsurface, w_row, w_col), 0.0, 1.0),
            opacity=np.clip(_box_downsample_2d(patch.opacity, w_row, w_col), 0.0, 1.0),
            emissive=np.clip(_box_downsample_2d(patch.emissive, w_row, w_col), 0.0, 1.0),
            cavity_ao=np.clip(_box_downsample_2d(patch.cavity_ao, w_row, w_col), 0.0, 1.0),
            iridescence_thickness_nm=np.maximum(0.0, _box_downsample_2d(patch.iridescence_thickness_nm, w_row, w_col)),
            iridescence_ior=np.maximum(1.0, _box_downsample_2d(patch.iridescence_ior, w_row, w_col)),
            labpbr_metal_code=patch.labpbr_metal_code,
        )

    return OpticalFields(subdivisions_per_unit=density, patches=out_patches)


def build_oklch_palette_ramp(palette: Palette, extra_steps: int = 0) -> np.ndarray:
    """Author a perceptual OKLCH color ramp and return Linear sRGB swatches (N, 3)."""
    n_steps = max(2, int(palette.steps) + int(extra_steps))
    base_l, base_c, base_h = palette.base_lch
    sh_shift, hi_shift = palette.hue_shift

    t = np.linspace(-1.0, 1.0, n_steps, dtype=np.float64)
    l_span_lo = max(0.06, base_l - 0.30)
    l_span_hi = min(0.96, base_l + 0.26)
    lightness = np.where(t < 0.0, base_l + t * (base_l - l_span_lo), base_l + t * (l_span_hi - base_l))
    chroma = np.maximum(0.0, base_c * (1.0 - 0.22 * (t * t)))
    hue = (base_h + np.where(t < 0.0, (-t) * sh_shift, t * hi_shift)) % 360.0
    lch = np.stack((lightness, chroma, hue), axis=-1)
    return oklch_to_linear_srgb(lch)


def quantize_linear_rgb_to_swatches(rgb_linear: np.ndarray, swatches_linear: np.ndarray) -> np.ndarray:
    """Quantize linear sRGB pixels (H, W, 3) to nearest perceptual Oklab swatch."""
    px_lab = linear_srgb_to_oklab(rgb_linear)
    sw_lab = linear_srgb_to_oklab(swatches_linear)  # (K, 3)
    diff = px_lab[..., None, :] - sw_lab[None, None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    best = np.argmin(dist2, axis=-1)
    return swatches_linear[best]


def evaluate_cook_torrance_ggx_brdf(
    wi: np.ndarray,
    wo: np.ndarray,
    normal: np.ndarray,
    *,
    perceptual_roughness: np.ndarray | float,
    base_color_linear: np.ndarray | tuple[float, float, float] = (1.0, 1.0, 1.0),
    metalness: np.ndarray | float = 0.0,
    dielectric_f0: np.ndarray | float = 0.04,
    iridescence_thickness_nm: np.ndarray | float = 0.0,
    iridescence_ior: np.ndarray | float = 1.45,
) -> np.ndarray:
    """Evaluate reciprocal, white-furnace energy-conserving Cook-Torrance GGX BRDF in sr^-1.

    Formulation (Walter et al., EGSR 2007; Heitz, JCGT 3(2):32-91, 2014;
    Ashikhmin & Shirley, J. Graph. Tools 5(2):25-32, 2000; Kelemen & Szirmay-Kalos 2001):
    - Specular microfacet lobe: f_s(ω_i, ω_o) = F(ω_o · h) D_GGX(h) V_Smith(ω_i, ω_o)
      with α = α_p^2, D_GGX(h) = α^2 / (π ((n·h)^2 (α^2 - 1) + 1)^2), and exact
      Height-Correlated Smith visibility V = G_2 / (4 (n·ω_i)(n·ω_o)).
    - Reciprocal Fresnel-coupled diffuse lobe:
      f_d(ω_i, ω_o) = (28 / (23π)) (1 - m)(1 - F_0) (1 - (1 - 0.5(n·ω_i))^5)
                      (1 - (1 - 0.5(n·ω_o))^5) ρ_base.

    Guarantees:
    1. Exact Helmholtz reciprocity: f_r(ω_i, ω_o) == f_r(ω_o, ω_i) to machine precision.
    2. White-furnace passivity: ∫_{Ω+} f_r(ω_i, ω_o) (n·ω_i) dω_i <= 1.0 for all α_p in [0.02, 1]
       and all view cosines μ_o in (0, 1].
    """
    wi_a = np.asarray(wi, dtype=np.float64)
    wo_a = np.asarray(wo, dtype=np.float64)
    n_a = np.asarray(normal, dtype=np.float64)

    wi_u = wi_a / np.maximum(1e-12, np.linalg.norm(wi_a, axis=-1, keepdims=True))
    wo_u = wo_a / np.maximum(1e-12, np.linalg.norm(wo_a, axis=-1, keepdims=True))
    n_u = n_a / np.maximum(1e-12, np.linalg.norm(n_a, axis=-1, keepdims=True))

    ndoti_raw = np.sum(n_u * wi_u, axis=-1)
    ndoto_raw = np.sum(n_u * wo_u, axis=-1)
    active = (ndoti_raw > 0.0) & (ndoto_raw > 0.0)

    ndoti = np.clip(ndoti_raw, 1e-8, 1.0)
    ndoto = np.clip(ndoto_raw, 1e-8, 1.0)

    h_raw = wi_u + wo_u
    h_norm = np.linalg.norm(h_raw, axis=-1, keepdims=True)
    h_u = np.where(h_norm > 1e-12, h_raw / np.maximum(1e-12, h_norm), n_u)

    ndoth = np.clip(np.sum(n_u * h_u, axis=-1), 0.0, 1.0)
    # Symmetric half-angle cosine: for unit vectors wi_u and wo_u, wi_u·h_u == wo_u·h_u identically.
    hdoto = np.clip(0.5 * (np.sum(wo_u * h_u, axis=-1) + np.sum(wi_u * h_u, axis=-1)), 0.0, 1.0)

    rp = np.asarray(perceptual_roughness, dtype=np.float64)
    alpha = np.maximum(0.0004, rp * rp)
    alpha2 = alpha * alpha

    denom_d = ndoth * ndoth * (alpha2 - 1.0) + 1.0
    d_ggx = alpha2 / np.maximum(1e-24, math.pi * denom_d * denom_d)

    ggx_o = ndoti * np.sqrt(ndoto * ndoto * (1.0 - alpha2) + alpha2)
    ggx_i = ndoto * np.sqrt(ndoti * ndoti * (1.0 - alpha2) + alpha2)
    vis_smith = 0.5 / np.maximum(1e-15, ggx_o + ggx_i)

    base_rgb = np.clip(np.asarray(base_color_linear, dtype=np.float64), 0.0, 1.0)
    m = np.clip(np.asarray(metalness, dtype=np.float64), 0.0, 1.0)[..., None]
    f0_d = np.clip(np.asarray(dielectric_f0, dtype=np.float64), 0.0, 1.0)[..., None]
    f0 = (1.0 - m) * f0_d + m * base_rgb
    f_schlick = f0 + (1.0 - f0) * np.power(1.0 - hdoto[..., None], 5.0)

    irid_d = np.asarray(iridescence_thickness_nm, dtype=np.float64)
    if np.any(irid_d > 1.0):
        # Symmetric half-angle incidence cos(θ_d) = ω_o · h == ω_i · h preserves exact reciprocity
        irid_rgb = evaluate_airy_thin_film_linear_rgb(
            irid_d,
            iridescence_ior,
            hdoto,
            substrate_rgb_lin=f0,
        )
        irid_mask = np.clip(irid_d / 80.0, 0.0, 1.0)[..., None]
        f_schlick = f_schlick * (1.0 - irid_mask) + irid_rgb * irid_mask

    f_spec = (d_ggx * vis_smith)[..., None] * f_schlick
    trans_i = 1.0 - np.power(1.0 - 0.5 * ndoti, 5.0)
    trans_o = 1.0 - np.power(1.0 - 0.5 * ndoto, 5.0)
    f_diff = (
        (28.0 / (23.0 * math.pi))
        * (1.0 - m)
        * (1.0 - f0)
        * (trans_i * trans_o)[..., None]
        * base_rgb
    )
    total = f_diff + f_spec
    return np.where(active[..., None], total, 0.0)


def compute_ggx_hemispherical_reflectance(
    mu_o: float,
    perceptual_roughness: float,
    *,
    metalness: float = 1.0,
    base_color_linear: tuple[float, float, float] = (1.0, 1.0, 1.0),
    dielectric_f0: float = 0.04,
    n_quad: int = 64,
) -> np.ndarray:
    """Integrate directional-hemispherical reflectance ρ_hd(μ_o) = ∫_{Ω+} f_r(ω_i, ω_o) (n·ω_i) dω_i.

    Uses exact GGX half-vector Gauss-Legendre importance quadrature on u in [0, 1)
    (where D_GGX(h) cos(θ_h) dω_h = du dφ_h / (2π), eliminating delta-peak aliasing at
    low roughnesses α_p >= 0.02) for the specular lobe, and Gauss-Legendre hemisphere
    quadrature for the smooth Ashikhmin-Shirley diffuse lobe.
    """
    mu_o_c = float(np.clip(mu_o, 1e-4, 1.0))
    sin_o = math.sqrt(max(0.0, 1.0 - mu_o_c * mu_o_c))
    wo = np.array((sin_o, 0.0, mu_o_c), dtype=np.float64)
    normal = np.array((0.0, 0.0, 1.0), dtype=np.float64)

    alpha = max(0.02, float(perceptual_roughness)) ** 2
    alpha2 = alpha * alpha

    u_nodes, u_weights = np.polynomial.legendre.leggauss(int(n_quad))
    u = 0.5 * (u_nodes + 1.0)
    wu = 0.5 * u_weights
    n_phi = max(64, 2 * int(n_quad))
    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False, dtype=np.float64)
    w_phi_norm = 1.0 / float(n_phi)  # dφ / (2π)

    cos2_th = (1.0 - u) / (1.0 + (alpha2 - 1.0) * u)
    cos_th = np.sqrt(np.clip(cos2_th, 1e-15, 1.0))
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - cos2_th))

    hx = sin_th[:, None] * np.cos(phi[None, :])
    hy = sin_th[:, None] * np.sin(phi[None, :])
    hz = np.broadcast_to(cos_th[:, None], hx.shape)

    vdoth = wo[0] * hx + wo[1] * hy + wo[2] * hz
    wix = 2.0 * vdoth * hx - wo[0]
    wiy = 2.0 * vdoth * hy - wo[1]
    wiz = 2.0 * vdoth * hz - wo[2]
    wi_spec = np.stack((wix, wiy, wiz), axis=-1)

    valid = (vdoth > 0.0) & (wiz > 0.0)
    # Isolate pure specular vs pure diffuse cleanly:
    denom_d = cos2_th * (alpha2 - 1.0) + 1.0
    d_ggx = (alpha2 / np.maximum(1e-24, math.pi * denom_d * denom_d))[:, None]
    # Pure diffuse BRDF at wi_spec has dielectric_f0=f0, metalness=m, minus specular:
    m_clamped = float(np.clip(metalness, 0.0, 1.0))
    base_arr = np.clip(np.asarray(base_color_linear, dtype=np.float64), 0.0, 1.0)
    f0_rgb = (1.0 - m_clamped) * float(np.clip(dielectric_f0, 0.0, 1.0)) + m_clamped * base_arr
    # Evaluate specular BRDF via evaluate_cook_torrance_ggx_brdf with metalness=1.0 and base_color=f0_rgb!
    f_spec_vals = evaluate_cook_torrance_ggx_brdf(
        wi_spec,
        np.broadcast_to(wo, wi_spec.shape),
        np.broadcast_to(normal, wi_spec.shape),
        perceptual_roughness=float(perceptual_roughness),
        base_color_linear=tuple(float(c) for c in f0_rgb),
        metalness=1.0,
    )
    # Since du * (dφ / 2π) = D_GGX(h) cos(θ_h) dω_h and dω_i = 4 (ω_o·h) dω_h,
    # f_spec * (n·ω_i) dω_i = (f_spec / D_GGX(h)) * (n·ω_i) * (4 (ω_o·h) / cos(θ_h)) * du * (dφ / 2π)
    jacobian_weight = np.where(
        valid,
        (wiz * 4.0 * vdoth) / np.maximum(1e-24, d_ggx * cos_th[:, None]),
        0.0,
    )
    rho_spec = np.sum(f_spec_vals * jacobian_weight[..., None] * wu[:, None, None], axis=(0, 1)) * w_phi_norm

    if m_clamped >= 1.0 - 1e-12:
        return rho_spec

    # Smooth diffuse lobe hemispherical integral via Gauss-Legendre on μ_i in [0, 1]
    mu_i = 0.5 * (u_nodes + 1.0)
    w_mui = 0.5 * u_weights
    sin_i = np.sqrt(np.maximum(0.0, 1.0 - mu_i * mu_i))
    wi_d = np.stack(
        (
            sin_i[:, None] * np.cos(phi[None, :]),
            sin_i[:, None] * np.sin(phi[None, :]),
            np.broadcast_to(mu_i[:, None], (len(mu_i), n_phi)),
        ),
        axis=-1,
    )
    f_tot_d = evaluate_cook_torrance_ggx_brdf(
        wi_d,
        np.broadcast_to(wo, wi_d.shape),
        np.broadcast_to(normal, wi_d.shape),
        perceptual_roughness=float(perceptual_roughness),
        base_color_linear=base_color_linear,
        metalness=m_clamped,
        dielectric_f0=float(dielectric_f0),
    )
    f_spec_d = evaluate_cook_torrance_ggx_brdf(
        wi_d,
        np.broadcast_to(wo, wi_d.shape),
        np.broadcast_to(normal, wi_d.shape),
        perceptual_roughness=float(perceptual_roughness),
        base_color_linear=tuple(float(c) for c in f0_rgb),
        metalness=1.0,
    )
    f_diff_vals = np.maximum(0.0, f_tot_d - f_spec_d)
    rho_diff = np.sum(f_diff_vals * mu_i[:, None, None] * w_mui[:, None, None], axis=(0, 1)) * (2.0 * math.pi * w_phi_norm)
    return rho_spec + rho_diff


def shade_patch_ggx_linear(
    patch: FaceOpticalPatch,
    grid_t_right_box: np.ndarray,
    grid_b_up_box: np.ndarray,
    grid_n_out_box: np.ndarray,
    box_rotation_world: np.ndarray,
    edge_distance_px: np.ndarray,
    environment: EnvironmentState,
    style: StyleConfig,
    *,
    view_dir_world: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate linear-space Cook-Torrance GGX + posed cavity AO + Airy thin-film."""
    rot = np.asarray(box_rotation_world, dtype=np.float64)
    t_w = rot @ grid_t_right_box
    b_w = rot @ grid_b_up_box
    n_geom_w = rot @ grid_n_out_box

    tn = patch.tangent_normal
    n_world = (
        tn[..., 0:1] * t_w
        + tn[..., 1:2] * b_w
        + tn[..., 2:3] * n_geom_w
    )
    n_world = n_world / np.maximum(1e-12, np.linalg.norm(n_world, axis=-1, keepdims=True))

    l_dir = environment.normalized_light_dir()
    if view_dir_world is not None:
        v_dir = np.asarray(view_dir_world, dtype=np.float64)
        v_dir = v_dir / max(1e-12, float(np.linalg.norm(v_dir)))
        v_grid = np.broadcast_to(v_dir, n_world.shape)
    else:
        # Static stylized bake uses documented reference view angle relative to surface normal + light azimuth
        ref_rad = math.radians(style.vanilla_reference_view_deg)
        tangent_toward_light = l_dir - np.einsum("ijk,k->ij", n_world, l_dir)[..., None] * n_world
        tl_norm = np.linalg.norm(tangent_toward_light, axis=-1, keepdims=True)
        fallback_t = np.broadcast_to(t_w, n_world.shape)
        unit_tangent = np.where(tl_norm > 1e-6, tangent_toward_light / np.maximum(1e-6, tl_norm), fallback_t)
        v_grid = math.cos(ref_rad) * n_world + math.sin(ref_rad) * unit_tangent
        v_grid = v_grid / np.maximum(1e-12, np.linalg.norm(v_grid, axis=-1, keepdims=True))

    l_grid = np.broadcast_to(l_dir, n_world.shape)
    ndotl = np.clip(np.einsum("ijk,k->ij", n_world, l_dir), 0.0, 1.0)
    ndotv = np.clip(np.sum(n_world * v_grid, axis=-1), 1e-4, 1.0)

    brdf_rgb = evaluate_cook_torrance_ggx_brdf(
        l_grid,
        v_grid,
        n_world,
        perceptual_roughness=patch.perceptual_roughness,
        base_color_linear=patch.base_color_linear,
        metalness=patch.metalness,
        dielectric_f0=patch.dielectric_f0,
        iridescence_thickness_nm=patch.iridescence_thickness_nm,
        iridescence_ior=patch.iridescence_ior,
    )
    # Direct exitant radiance with white-Lambertian-normalized sun irradiance E_sun = π * sun_intensity
    direct_lobe = (math.pi * environment.sun_intensity) * brdf_rgb * ndotl[..., None]

    # Distinct dielectric F0 vs conductor albedo reflectance for ambient specular tint
    m = patch.metalness[..., None]
    f0_diel = np.repeat(patch.dielectric_f0[..., None], 3, axis=-1)
    f0 = (1.0 - m) * f0_diel + m * patch.base_color_linear
    f_ambient = f0 + (1.0 - f0) * np.power(1.0 - ndotv[..., None], 5.0)
    if np.any(patch.iridescence_thickness_nm > 1.0):
        cos_theta = ndotv if view_dir_world is not None else np.full_like(ndotv, math.cos(math.radians(style.vanilla_reference_view_deg)))
        irid_amb = evaluate_airy_thin_film_linear_rgb(
            patch.iridescence_thickness_nm,
            patch.iridescence_ior,
            cos_theta,
            substrate_rgb_lin=f0,
        )
        irid_mask = np.clip(patch.iridescence_thickness_nm / 80.0, 0.0, 1.0)[..., None]
        f_ambient = f_ambient * (1.0 - irid_mask) + irid_amb * irid_mask

    # Hemisphere ambient weighted by posed cavity AO + subtle environment reflection for conductors
    hemi_up = 0.5 + 0.5 * n_world[..., 1:2]
    ambient_diff = (1.0 - 0.65 * m) * patch.base_color_linear
    ambient_spec = 0.65 * m * f_ambient * (1.0 - 0.45 * patch.perceptual_roughness[..., None])
    ambient_lobe = (ambient_diff + ambient_spec) * patch.cavity_ao[..., None] * environment.ambient_intensity * (0.75 + 0.35 * hemi_up)

    # Controlled Minecraft bevel highlight near exposed outward edges
    bevel_mask = np.clip(1.0 - edge_distance_px / 0.35, 0.0, 1.0) * patch.cavity_ao * (0.4 + 0.6 * ndotl)
    bevel_lobe = style.bevel_highlight_strength * bevel_mask[..., None] * (0.5 * patch.base_color_linear + 0.5)

    emissive_lobe = patch.emissive[..., None] * np.maximum(patch.base_color_linear, 0.08) * 1.15
    radiance = ambient_lobe + direct_lobe + bevel_lobe + emissive_lobe
    # Soft shoulder tone compression above 1.0 to preserve highlight hue in linear space
    compressed = radiance / (1.0 + 0.25 * np.maximum(0.0, radiance - 0.85))
    return np.clip(compressed, 0.0, 1.0)


def write_padded_island(canvas: np.ndarray, x: int, y: int, island: np.ndarray, padding: int) -> None:
    """Blit ``island`` (H, W, C) at (x, y) and edge-clamp into its ``padding`` border."""
    h, w = island.shape[:2]
    canvas[y : y + h, x : x + w] = island
    if padding > 0:
        for p in range(1, padding + 1):
            canvas[y - p, x : x + w] = island[0]
            canvas[y + h - 1 + p, x : x + w] = island[-1]
            canvas[y - p : y + h + p, x - p] = canvas[y - p : y + h + p, x]
            canvas[y - p : y + h + p, x + w - 1 + p] = canvas[y - p : y + h + p, x + w - 1]


def encode_rgba_png(rgba_u8: np.ndarray) -> bytes:
    img = Image.fromarray(rgba_u8, mode="RGBA")
    stream = io.BytesIO()
    img.save(stream, format="PNG", optimize=False)
    return stream.getvalue()


def evaluate_world_optical_pipeline(
    world_state: WorldState,
    *,
    subdivisions_per_unit: int | None = None,
) -> tuple[SurfaceDomain, Scene, dict[FaceKey, PosedFaceKinematics], OpticalFields, OpticalFields]:
    """Run P1-03a/b/c stages returning (domain, posed_scene, posed_kin, solver_optics, pixel_optics)."""
    subdiv = subdivisions_per_unit or max(4, world_state.surface_state.subdivisions_per_unit)
    domain = build_surface_domain(
        world_state.program,
        world_state.material_program,
        subdivisions_per_unit=subdiv,
    )
    if set(world_state.surface_state.patches.keys()) != set(domain.faces.keys()) or world_state.surface_state.subdivisions_per_unit != subdiv:
        from synccraft.surface.domain import remap_surface_state

        world_state.surface_state = remap_surface_state(world_state.surface_state, domain)

    posed_scene, posed_kin = evaluate_posed_surface_kinematics(
        domain,
        world_state.program,
        world_state.pose_euler_deltas,
    )
    solver_optics = evaluate_optical_fields(
        world_state.program,
        domain,
        world_state.material_program,
        world_state.surface_state,
        posed_kin,
    )
    pixel_optics = downsample_optical_fields(
        solver_optics,
        domain,
        world_state.program.meta.texel_density,
    )
    return domain, posed_scene, posed_kin, solver_optics, pixel_optics


def compile_vanilla_atlas(
    world_state: WorldState,
    *,
    uv_layout: dict[str, Any] | None = None,
    max_width: int = 512,
    padding: int = 1,
    box_order: list[str] | None = None,
    face_order: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Compile WorldState into a stylized Vanilla RGBA PNG atlas and descriptor."""
    domain, posed_scene, _, _, pixel_optics = evaluate_world_optical_pipeline(world_state)
    layout = uv_layout or allocate_uv_layout(
        world_state.program,
        posed_scene,
        max_width=max_width,
        padding=padding,
        box_order=box_order,
        face_order=face_order,
    )
    width, height = int(layout["width"]), int(layout["height"])
    pad = int(layout.get("padding", padding))

    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    posed_boxes = posed_scene.box_by_name
    palettes = {p.name: p for p in world_state.program.palettes}

    # Build adaptive swatch tables per material when palette quantization is enabled
    swatch_bank: dict[str, np.ndarray] = {}
    if world_state.style_config.quantize_to_palette_ramp:
        for mat_name, pal in palettes.items():
            spec = world_state.material_program.spec_for(mat_name, pal)
            sw_list = [build_oklch_palette_ramp(pal, extra_steps=4)]
            for extra_lch in (spec.substrate_lch, spec.secondary_lch, spec.coating_lch, spec.oxide_lch):
                extra_pal = Palette(mat_name, extra_lch, max(4, pal.steps), pal.hue_shift, pal.emissive)
                sw_list.append(build_oklch_palette_ramp(extra_pal, extra_steps=2))
            swatch_bank[mat_name] = np.concatenate(sw_list, axis=0)

    density = world_state.program.meta.texel_density / 16.0
    dx_px = 1.0 / density

    for fk, patch in pixel_optics.patches.items():
        box_name, face = fk
        u0, v0, u1, v1 = layout["faces"][box_name][face]
        h_px, w_px = int(v1 - v0), int(u1 - u0)
        grid = domain.faces[fk]
        pbox = posed_boxes[box_name]

        u_px = (np.arange(w_px, dtype=np.float64) + 0.5) * dx_px
        v_px = (np.arange(h_px, dtype=np.float64) + 0.5) * dx_px
        ug, vg = np.meshgrid(u_px, v_px)
        eff_w_u = max(dx_px, float(grid.width_units))
        eff_h_u = max(dx_px, float(grid.height_units))
        edge_d = np.maximum(
            0.0,
            np.minimum(np.minimum(ug, eff_w_u - ug), np.minimum(vg, eff_h_u - vg)),
        )

        shaded_lin = shade_patch_ggx_linear(
            patch,
            grid.tangent_right_box,
            grid.bitangent_up_box,
            grid.normal_box,
            pbox.rotation,
            edge_d,
            world_state.environment,
            world_state.style_config,
        )

        if world_state.style_config.quantize_to_palette_ramp and grid.material_name in swatch_bank:
            # Preserve unquantized stamp overrides or blend high-step perceptual ramp
            swatches = swatch_bank[grid.material_name]
            has_stamp = any(s.box_name == box_name and s.face == face for s in world_state.material_program.stamps)
            if not has_stamp and spec_allows_ramp(world_state.material_program, grid.material_name):
                shaded_lin = quantize_linear_rgb_to_swatches(shaded_lin, swatches)

        rgb_u8 = linear_to_srgb_u8(shaded_lin)
        alpha_u8 = np.rint(np.clip(patch.opacity, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None]
        island_rgba = np.concatenate((rgb_u8, alpha_u8), axis=-1)
        write_padded_island(canvas, int(u0), int(v0), island_rgba, pad)

    png_bytes = encode_rgba_png(canvas)
    descriptor = {
        "width": width,
        "height": height,
        "padding": pad,
        "faces": layout["faces"],
        "data_url": "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii"),
        "mode": "stateful_vanilla_ggx_oklch_bake",
        "vanilla_reference_view_deg": world_state.style_config.vanilla_reference_view_deg,
        "limitations_note": (
            "Static RGB atlas bake evaluates GGX specular and thin-film interference at "
            f"vanilla_reference_view_deg={world_state.style_config.vanilla_reference_view_deg}; "
            "dynamic view-dependent highlights require LabPBR 1.3 or glTF 2.0 PBR targets."
        ),
    }
    return descriptor, png_bytes


def spec_allows_ramp(material_program: MaterialProgram, material_name: str) -> bool:
    if material_name in material_program.materials:
        return material_program.materials[material_name].family != "thin_film_iridescence"
    return True
