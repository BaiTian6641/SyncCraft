"""Deterministic UV atlas allocation and P1-03c stateful Pixel-PBR shading compiler."""

from .cavity_shader import shade_asset
from .compiler import (
    build_oklch_palette_ramp,
    compile_vanilla_atlas,
    compute_ggx_hemispherical_reflectance,
    downsample_optical_fields,
    evaluate_cook_torrance_ggx_brdf,
    evaluate_world_optical_pipeline,
    quantize_linear_rgb_to_swatches,
    shade_patch_ggx_linear,
)
from .uv_packer import allocate_uv_layout, build_atlas, oklch_to_srgb

__all__ = [
    "allocate_uv_layout",
    "build_atlas",
    "build_oklch_palette_ramp",
    "compile_vanilla_atlas",
    "compute_ggx_hemispherical_reflectance",
    "downsample_optical_fields",
    "evaluate_cook_torrance_ggx_brdf",
    "evaluate_world_optical_pipeline",
    "oklch_to_srgb",
    "quantize_linear_rgb_to_swatches",
    "shade_asset",
    "shade_patch_ggx_linear",
]
