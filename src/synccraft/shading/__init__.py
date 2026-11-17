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
from .preserve_and_layer import (
    alpha_over_rgba,
    decode_rgba_png,
    encode_rgba_png,
    extract_painted_palette_profile,
    generate_oklch_texture_variant,
    preserve_and_layer_painted_texture,
    rebake_selected_faces,
    rgba_to_data_url,
)
from .uv_packer import allocate_uv_layout, build_atlas, oklch_to_srgb

__all__ = [
    "allocate_uv_layout",
    "alpha_over_rgba",
    "build_atlas",
    "build_oklch_palette_ramp",
    "compile_vanilla_atlas",
    "compute_ggx_hemispherical_reflectance",
    "decode_rgba_png",
    "downsample_optical_fields",
    "encode_rgba_png",
    "evaluate_cook_torrance_ggx_brdf",
    "evaluate_world_optical_pipeline",
    "extract_painted_palette_profile",
    "generate_oklch_texture_variant",
    "oklch_to_srgb",
    "preserve_and_layer_painted_texture",
    "quantize_linear_rgb_to_swatches",
    "rebake_selected_faces",
    "rgba_to_data_url",
    "shade_asset",
    "shade_patch_ggx_linear",
]
