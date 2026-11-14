#!/usr/bin/env python3
"""Verify all bundled release files against the package's SHA-256 manifest."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
errors = []
for entry in manifest["files"]:
    rel = Path(entry["path"])
    if rel.is_absolute() or ".." in rel.parts:
        errors.append(f"unsafe manifest path: {rel}")
        continue
    path = root / rel
    if not path.is_file():
        errors.append(f"missing: {rel}")
    elif hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        errors.append(f"changed: {rel}")
print(json.dumps({"checked_files": len(manifest["files"]), "errors": errors}, indent=2))
raise SystemExit(bool(errors))
