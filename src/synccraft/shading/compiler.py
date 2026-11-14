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

    for fk, patch in optical_fields.patches.items():
        grid = domain.faces[fk]
        fw = grid.width_units * density
        fh = grid.height_units * density
        if not math.isclose(fw, round(fw)) or not math.isclose(fh, round(fh)):
            raise ValueError(
                f"texel_density={texel_density} produces fractional texels ({fw}x{fh}) on {fk}; integer atlas required"
            )
        w_px = max(1, int(round(fw)))
        h_px = max(1, int(round(fh)))
        in_h, in_w = patch.height.shape

        if (h_px, w_px) == (in_h, in_w):
            out_patches[fk] = patch
            continue

        w_row = _interval_overlap_matrix(h_px, in_h, float(grid.height_units))
        w_col = _interval_overlap_matrix(w_px, in_w, float(grid.width_units))

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

    h_vec = v_grid + l_dir
    h_vec = h_vec / np.maximum(1e-12, np.linalg.norm(h_vec, axis=-1, keepdims=True))

    ndotl = np.clip(np.einsum("ijk,k->ij", n_world, l_dir), 0.0, 1.0)
    ndotv = np.clip(np.sum(n_world * v_grid, axis=-1), 1e-4, 1.0)
    ndoth = np.clip(np.sum(n_world * h_vec, axis=-1), 0.0, 1.0)
    vdoth = np.clip(np.sum(v_grid * h_vec, axis=-1), 0.0, 1.0)

    # GGX microfacet roughness convention: alpha = perceptual_roughness^2
    alpha = np.maximum(0.0016, patch.perceptual_roughness * patch.perceptual_roughness)
    alpha2 = alpha * alpha

    denom_d = ndoth * ndoth * (alpha2 - 1.0) + 1.0
    d_ggx = alpha2 / np.maximum(1e-8, math.pi * denom_d * denom_d)

    # Height-Correlated Smith G2
    ggx_v = ndotl * np.sqrt(ndotv * ndotv * (1.0 - alpha2) + alpha2)
    ggx_l = ndotv * np.sqrt(ndotl * ndotl * (1.0 - alpha2) + alpha2)
    vis_smith = 0.5 / np.maximum(1e-6, ggx_v + ggx_l)

    # Distinct dielectric F0 vs conductor albedo reflectance
    m = patch.metalness[..., None]
    f0_diel = np.repeat(patch.dielectric_f0[..., None], 3, axis=-1)
    f0 = (1.0 - m) * f0_diel + m * patch.base_color_linear
    f_schlick = f0 + (1.0 - f0) * np.power(1.0 - vdoth[..., None], 5.0)

    if np.any(patch.iridescence_thickness_nm > 1.0):
        cos_theta = ndotv if view_dir_world is not None else np.full_like(ndotv, math.cos(math.radians(style.vanilla_reference_view_deg)))
        irid_rgb = evaluate_airy_thin_film_linear_rgb(
            patch.iridescence_thickness_nm,
            patch.iridescence_ior,
            cos_theta,
            substrate_rgb_lin=f0,
        )
        irid_mask = np.clip(patch.iridescence_thickness_nm / 80.0, 0.0, 1.0)[..., None]
        f_schlick = f_schlick * (1.0 - irid_mask) + irid_rgb * irid_mask

    spec_lobe = (d_ggx * vis_smith)[..., None] * f_schlick * ndotl[..., None] * environment.sun_intensity
    kd = (1.0 - m) * (1.0 - f_schlick)
    diffuse_lobe = kd * patch.base_color_linear * (ndotl[..., None] * environment.sun_intensity)

    # Hemisphere ambient weighted by posed cavity AO + subtle environment reflection for conductors
    hemi_up = 0.5 + 0.5 * n_world[..., 1:2]
    ambient_diff = (1.0 - 0.65 * m) * patch.base_color_linear
    ambient_spec = 0.65 * m * f_schlick * (1.0 - 0.45 * patch.perceptual_roughness[..., None])
    ambient_lobe = (ambient_diff + ambient_spec) * patch.cavity_ao[..., None] * environment.ambient_intensity * (0.75 + 0.35 * hemi_up)

    # Controlled Minecraft bevel highlight near exposed outward edges
    bevel_mask = np.clip(1.0 - edge_distance_px / 0.35, 0.0, 1.0) * patch.cavity_ao * (0.4 + 0.6 * ndotl)
    bevel_lobe = style.bevel_highlight_strength * bevel_mask[..., None] * (0.5 * patch.base_color_linear + 0.5)

    emissive_lobe = patch.emissive[..., None] * np.maximum(patch.base_color_linear, 0.08) * 1.15
    radiance = ambient_lobe + diffuse_lobe + spec_lobe + bevel_lobe + emissive_lobe
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
        edge_d = np.minimum(np.minimum(ug, grid.width_units - ug), np.minimum(vg, grid.height_units - vg))

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
