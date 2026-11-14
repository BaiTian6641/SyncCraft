#!/usr/bin/env python3
"""Compile the original starter examples into runs/demo; no model downloads."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from synccraft.cli import compile_file

results = []
for source in sorted((ROOT / "examples").glob("*.blockdsl")):
    results.append(compile_file(source, ROOT / "runs" / "demo" / source.stem, overwrite=True))
print(json.dumps(results, indent=2))
