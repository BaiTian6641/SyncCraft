"""All 22 bespoke Monster Hunter Block-DSL v4.0/v4.1 sculpted species builders."""

from __future__ import annotations

from typing import Any, Callable

from .arkveld import build_arkveld
from .chaotic_gore_magala import build_chaotic_gore_magala
from .crimson_glow_valstrax import build_crimson_glow_valstrax
from .fatalis import build_fatalis, build_white_fatalis
from .gore_magala import build_gore_magala
from .jin_dahaad import build_jin_dahaad
from .lagiacrus import (
    build_abyssal_lagiacrus,
    build_ivory_lagiacrus,
    build_lagiacrus,
)
from .legiana import build_legiana, build_shrieking_legiana
from .mizutsune import build_mizutsune
from .rathalos import (
    build_azure_rathalos,
    build_rathalos,
    build_silver_rathalos,
)
from .rey_dau import build_rey_dau
from .ruiner_nergigante import build_ruiner_nergigante
from .stygian_zinogre import build_stygian_zinogre
from .velkhana import build_velkhana
from .zinogre import build_zinogre
from .zoh_shia import build_zoh_shia

SPECIES_V8_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "rathalos": build_rathalos,
    "azure_rathalos": build_azure_rathalos,
    "silver_rathalos": build_silver_rathalos,
    "legiana": build_legiana,
    "shrieking_legiana": build_shrieking_legiana,
    "rey_dau": build_rey_dau,
    "arkveld": build_arkveld,
    "zinogre": build_zinogre,
    "stygian_zinogre": build_stygian_zinogre,
    "gore_magala": build_gore_magala,
    "chaotic_gore_magala": build_chaotic_gore_magala,
    "ruiner_nergigante": build_ruiner_nergigante,
    "crimson_glow_valstrax": build_crimson_glow_valstrax,
    "fatalis": build_fatalis,
    "white_fatalis": build_white_fatalis,
    "velkhana": build_velkhana,
    "zoh_shia": build_zoh_shia,
    "lagiacrus": build_lagiacrus,
    "ivory_lagiacrus": build_ivory_lagiacrus,
    "abyssal_lagiacrus": build_abyssal_lagiacrus,
    "mizutsune": build_mizutsune,
    "jin_dahaad": build_jin_dahaad,
}

__all__ = [
    "SPECIES_V8_BUILDERS",
    "build_arkveld",
    "build_azure_rathalos",
    "build_chaotic_gore_magala",
    "build_crimson_glow_valstrax",
    "build_fatalis",
    "build_gore_magala",
    "build_ivory_lagiacrus",
    "build_abyssal_lagiacrus",
    "build_jin_dahaad",
    "build_lagiacrus",
    "build_legiana",
    "build_mizutsune",
    "build_rathalos",
    "build_rey_dau",
    "build_ruiner_nergigante",
    "build_shrieking_legiana",
    "build_silver_rathalos",
    "build_stygian_zinogre",
    "build_velkhana",
    "build_white_fatalis",
    "build_zinogre",
    "build_zoh_shia",
]
