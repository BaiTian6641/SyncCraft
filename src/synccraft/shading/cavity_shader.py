"""Production stateful surface/cavity/GGX shader entry point (P1-03c)."""

from __future__ import annotations

from typing import Any

from synccraft.dsl.ast_nodes import Program
from synccraft.surface.contracts import (
    EnvironmentState,
    MaterialProgram,
    PersistentSurfaceState,
    StyleConfig,
    WorldState,
)
from synccraft.surface.domain import build_surface_domain, initialize_surface_state
from .compiler import compile_vanilla_atlas


def shade_asset(
    program_or_world: Program | WorldState,
    *,
    material_program: MaterialProgram | None = None,
    surface_state: PersistentSurfaceState | None = None,
    pose_euler_deltas: dict[str, tuple[float, float, float]] | None = None,
    style_config: StyleConfig | None = None,
    environment: EnvironmentState | None = None,
    uv_layout: dict[str, Any] | None = None,
    max_width: int = 512,
    padding: int = 1,
) -> tuple[dict[str, Any], bytes]:
    """Compile a Program or WorldState into a shaded Vanilla RGBA atlas + PNG bytes."""
    if isinstance(program_or_world, WorldState):
        ws = program_or_world
    elif isinstance(program_or_world, Program):
        mat_prog = material_program or MaterialProgram.from_program(program_or_world)
        domain = build_surface_domain(program_or_world, mat_prog)
        st = surface_state or initialize_surface_state(domain)
        ws = WorldState(
            program=program_or_world,
            material_program=mat_prog,
            surface_state=st,
            pose_euler_deltas=dict(pose_euler_deltas or {}),
            style_config=style_config or StyleConfig.from_program(program_or_world),
            environment=environment or EnvironmentState(),
        )
    else:
        raise TypeError("shade_asset expects a Program or WorldState")

    return compile_vanilla_atlas(
        ws,
        uv_layout=uv_layout,
        max_width=max_width,
        padding=padding,
    )
