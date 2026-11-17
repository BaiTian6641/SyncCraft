"""Monster Hunter Block-DSL v4.1 Hand-Authored Progressive Sculpted Masterworks (V9)."""

from .catalog import CANONICAL_22_BY_ID, CANONICAL_22_ORDER, CANONICAL_22_SPECS, MonsterSpec, get_monster_spec
from .studio_compiler import DEFAULT_DSL_ROOT, inspect_or_compile_monster

__all__ = [
    "CANONICAL_22_BY_ID",
    "CANONICAL_22_ORDER",
    "CANONICAL_22_SPECS",
    "DEFAULT_DSL_ROOT",
    "MonsterSpec",
    "get_monster_spec",
    "inspect_or_compile_monster",
]
