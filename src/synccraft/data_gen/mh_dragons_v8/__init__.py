"""SyncCraft Block-DSL v4.0 Sculpted Blockbench Monster Hunter Studio Engine & 22-Specimen Registry."""

from __future__ import annotations

from .dsl_primitives import RigCanvas, lerp3
from .registry import ALL_V8_MONSTER_BUILDERS, CANONICAL_22_ORDER
from .species import SPECIES_V8_BUILDERS
from .v8_animator import build_monster_12_clips
from .v8_compiler import (
    compile_and_export_v8_monster,
    emit_stage1_coarse_blockout_dsl,
    emit_v4_assembly_dsl,
    prepare_v4_assembly_blueprint,
)
from .v8_painter import bake_x32_atlas, prepare_elements_and_pack_uv
from .v8_renderer import (
    render_animated_clip_gif,
    render_monster_studio_review_sheet,
    render_stage_and_regional_sheet,
    render_studio_frame,
)
from .v8_viewer import build_v8_offline_viewer

__all__ = [
    "ALL_V8_MONSTER_BUILDERS",
    "CANONICAL_22_ORDER",
    "RigCanvas",
    "SPECIES_V8_BUILDERS",
    "bake_x32_atlas",
    "build_monster_12_clips",
    "build_v8_offline_viewer",
    "compile_and_export_v8_monster",
    "emit_stage1_coarse_blockout_dsl",
    "emit_v4_assembly_dsl",
    "lerp3",
    "prepare_elements_and_pack_uv",
    "prepare_v4_assembly_blueprint",
    "render_animated_clip_gif",
    "render_monster_studio_review_sheet",
    "render_stage_and_regional_sheet",
    "render_studio_frame",
]
