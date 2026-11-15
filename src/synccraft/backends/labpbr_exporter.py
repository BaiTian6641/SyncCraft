"""Minecraft LabPBR 1.3 Multi-Texture Exporter (P1-03c).

Reference specification:
- https://shaderlabs.org/wiki/LabPBR_Material_Standard (Version 1.3)
- https://shaderlabs.org/wiki/LabPBR_Implementation_Requirements

Channel layout (kept strictly separate from glTF 2.0):
- ``atlas.png``: Intrinsic sRGB albedo (RGB) + linear opacity (A). No baked sun lighting.
- ``atlas_n.png``:
  - R: +X tangent normal (left-to-right in UV) mapped [-1, 1] -> [0, 255]
  - G: +Y bitangent normal (bottom-to-top in UV, OpenGL green-up) mapped [-1, 1] -> [0, 255]
  - B: Posed ambient occlusion [0..255] (shader reconstructs Nz = sqrt(1 - Nx^2 - Ny^2))
  - A: Parallax height map [1..255] (0 is strictly avoided; LabPBR reserves 0 for POM cutout)
- ``atlas_s.png``:
  - R: Perceptual smoothness (1 - perceptual_roughness) in [0..255]
  - G: Dielectric linear F0 in [0..229], or LabPBR conductor metal codes [230..255]
  - B: Porosity in [0..64], or Subsurface Scattering (SSS) in [65..255]
  - A: Emissive strength in [0..254]; A = 255 strictly denotes NO emission in LabPBR 1.3.

Note: Serialization and channel-invariant auditing do not claim live Iris/OptiFine shader
pack rendering until verified in an external Minecraft runtime; several LabPBR channels
(porosity/SSS, POM, hardcoded metalness) remain optional across shader implementations.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

import numpy as np

from synccraft.backends.common import validate_scene_for_export
from synccraft.shading.compiler import (
    encode_rgba_png,
    evaluate_world_optical_pipeline,
    write_padded_island,
)
from synccraft.shading.uv_packer import allocate_uv_layout
from synccraft.surface.contracts import WorldState, linear_to_srgb_u8


def export_labpbr_1_3(
    world_state: WorldState,
    *,
    uv_layout: dict[str, Any] | None = None,
    max_width: int | None = None,
    padding: int = 1,
) -> dict[str, Any]:
    """Export ``WorldState`` to LabPBR 1.3 PNG byte streams and audited descriptor."""
    domain, posed_scene, _, _, pixel_optics = evaluate_world_optical_pipeline(world_state)
    validate_scene_for_export(posed_scene)

    layout = uv_layout or allocate_uv_layout(
        world_state.program,
        posed_scene,
        max_width=max_width,
        padding=padding,
    )
    width, height = int(layout["width"]), int(layout["height"])
    pad = int(layout.get("padding", padding))

    albedo_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    # Default _n background keeps flat normal (128, 128), AO=255, POM height=128 (>= 1 everywhere!)
    normal_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    normal_canvas[..., 0] = 128
    normal_canvas[..., 1] = 128
    normal_canvas[..., 2] = 255
    normal_canvas[..., 3] = 128
    # Default _s background has smoothness=0, F0=10, porosity=0, emissive=255 (255 = NO emission!)
    specular_canvas = np.zeros((height, width, 4), dtype=np.uint8)
    specular_canvas[..., 1] = 10
    specular_canvas[..., 3] = 255

    for fk, patch in pixel_optics.patches.items():
        box_name, face = fk
        u0, v0, u1, v1 = layout["faces"][box_name][face]
        h_px, w_px = int(v1 - v0), int(u1 - u0)

        # 1. Intrinsic Albedo (atlas.png)
        alb_rgb = linear_to_srgb_u8(patch.base_color_linear)
        alb_a = np.rint(np.clip(patch.opacity, 0.0, 1.0) * 255.0).astype(np.uint8)[..., None]
        alb_island = np.concatenate((alb_rgb, alb_a), axis=-1)
        write_padded_island(albedo_canvas, int(u0), int(v0), alb_island, pad)

        # 2. Normal + AO + POM Height (atlas_n.png)
        tn = patch.tangent_normal
        nx_u8 = np.rint(np.clip(tn[..., 0] * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
        ny_u8 = np.rint(np.clip(tn[..., 1] * 0.5 + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
        ao_u8 = np.rint(np.clip(patch.cavity_ao, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Map micro-height [-0.125, +0.125] model units into [1, 255]
        h_u8 = np.clip(np.rint(128.0 + (patch.height / 0.125) * 127.0), 1, 255).astype(np.uint8)
        nrm_island = np.stack((nx_u8, ny_u8, ao_u8, h_u8), axis=-1)
        write_padded_island(normal_canvas, int(u0), int(v0), nrm_island, pad)

        # 3. Specular (atlas_s.png)
        smooth_u8 = np.rint(np.clip(1.0 - patch.perceptual_roughness, 0.0, 1.0) * 255.0).astype(np.uint8)

        is_conductor = patch.metalness >= 0.5
        diel_g = np.clip(np.rint(patch.dielectric_f0 * 255.0), 0, 229).astype(np.uint8)
        metal_g = np.full((h_px, w_px), int(np.clip(patch.labpbr_metal_code, 230, 255)), dtype=np.uint8)
        f0_metal_u8 = np.where(is_conductor, metal_g, diel_g)

        use_sss = (patch.subsurface > 0.05) & (patch.subsurface >= patch.porosity)
        sss_b = np.clip(np.rint(65.0 + np.clip(patch.subsurface, 0.0, 1.0) * 190.0), 65, 255).astype(np.uint8)
        poros_b = np.clip(np.rint(np.clip(patch.porosity, 0.0, 1.0) * 64.0), 0, 64).astype(np.uint8)
        poros_sss_u8 = np.where(use_sss, sss_b, poros_b)

        is_emissive = patch.emissive > 1e-4
        emis_val = np.clip(np.rint(np.clip(patch.emissive, 0.0, 1.0) * 254.0), 1, 254).astype(np.uint8)
        emis_u8 = np.where(is_emissive, emis_val, np.uint8(255))

        spec_island = np.stack((smooth_u8, f0_metal_u8, poros_sss_u8, emis_u8), axis=-1)
        write_padded_island(specular_canvas, int(u0), int(v0), spec_island, pad)

    atlas_png = encode_rgba_png(albedo_canvas)
    atlas_n_png = encode_rgba_png(normal_canvas)
    atlas_s_png = encode_rgba_png(specular_canvas)

    return {
        "standard": "LabPBR_1.3",
        "width": width,
        "height": height,
        "padding": pad,
        "faces": layout["faces"],
        "atlas_png": atlas_png,
        "atlas_n_png": atlas_n_png,
        "atlas_s_png": atlas_s_png,
        "atlas_data_url": "data:image/png;base64," + base64.b64encode(atlas_png).decode("ascii"),
        "arrays": {
            "atlas": albedo_canvas,
            "atlas_n": normal_canvas,
            "atlas_s": specular_canvas,
        },
        "sha256": {
            "atlas.png": hashlib.sha256(atlas_png).hexdigest(),
            "atlas_n.png": hashlib.sha256(atlas_n_png).hexdigest(),
            "atlas_s.png": hashlib.sha256(atlas_s_png).hexdigest(),
        },
        "channel_audit": {
            "normal_alpha_min": int(normal_canvas[..., 3].min()),
            "normal_alpha_max": int(normal_canvas[..., 3].max()),
            "specular_g_conductor_pixels": int(np.sum(specular_canvas[..., 1] >= 230)),
            "specular_g_dielectric_pixels": int(np.sum(specular_canvas[..., 1] <= 229)),
            "specular_alpha_emissive_pixels": int(np.sum(specular_canvas[..., 3] <= 254)),
            "specular_alpha_non_emissive_pixels": int(np.sum(specular_canvas[..., 3] == 255)),
        },
        "verification_status": "serialized_and_channel_audited_not_runtime_shader_verified",
    }
