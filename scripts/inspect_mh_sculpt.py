#!/usr/bin/env python3
"""CLI for Sculptor Iterative Stage Compilation, Sculpted Style Audit, & Visual Inspection.

Usage:
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/inspect_mh_sculpt.py zinogre --stage 1
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/inspect_mh_sculpt.py zinogre --stage 2
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/inspect_mh_sculpt.py zinogre --stage 3 --gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from synccraft.data_gen.mh_sculpted_v9.studio_compiler import DEFAULT_DSL_ROOT, inspect_or_compile_monster


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("monster_id", help="Canonical Monster Hunter specimen ID (e.g. zinogre, rathalos)")
    p.add_argument("--stage", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--dsl-root", type=Path, default=DEFAULT_DSL_ROOT)
    p.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts" / "Monster_Hunter_BlockDSL_Masterworks_v9")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--gif", action="store_true")
    p.add_argument("--allow-audit-fail", action="store_true", help="Write render previews even when audit fails so sculptor can inspect overlaps")
    args = p.parse_args()

    res = inspect_or_compile_monster(
        args.monster_id,
        dsl_root=args.dsl_root,
        output_pack_root=args.output_root,
        stage=args.stage,
        render_previews=not args.no_render,
        render_gif=args.gif,
        strict_audit=not args.allow_audit_fail,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
