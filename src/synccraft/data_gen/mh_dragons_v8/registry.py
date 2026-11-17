"""src/synccraft/data_gen/mh_dragons_v8/registry.py

Unified 22-Specimen Registry across the Bespoke Per-Species Sculpted Block-DSL Modules.
"""

from __future__ import annotations

from typing import Any, Callable

from .species import SPECIES_V8_BUILDERS
from .v8_viewer import CANONICAL_22_ORDER

ALL_V8_MONSTER_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = dict(SPECIES_V8_BUILDERS)

missing = [mid for mid in CANONICAL_22_ORDER if mid not in ALL_V8_MONSTER_BUILDERS]
if missing:
    raise RuntimeError(f"Registry missing expected sculpted monsters: {missing}")

__all__ = ["ALL_V8_MONSTER_BUILDERS", "CANONICAL_22_ORDER"]
