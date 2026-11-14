"""Local-only CLI for the implemented static starter slice."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from synccraft import __version__


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _atomic_write(path, value):
    data = value.encode("utf-8") if isinstance(value, str) else value
    fd, temp = tempfile.mkstemp(prefix=".synccraft-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def load_scene(path, target=None):
    from synccraft.dsl import parse_program, validate_program
    from synccraft.kinematics.se3_chain import compile_geometry
    program = parse_program(Path(path).read_text(encoding="utf-8"))
    if target:
        program = replace(program, meta=replace(program.meta, target_profile=target))
        validate_program(program)
    return program, compile_geometry(program)


def compile_file(path, out, *, target=None, overwrite=False, allow_audit_failures=False):
    from synccraft.audit.sync_audit import audit_scene
    from synccraft.audit.offscreen_renderer import render_four_views
    from synccraft.backends.bbmodel5_exporter import export_bbmodel
    from synccraft.backends.java_exporter import export_java
    from synccraft.dsl import format_program
    from synccraft.shading.uv_packer import build_atlas

    program, scene = load_scene(path, target)
    profile = program.meta.target_profile
    if profile not in ("bbmodel_5", "java_modern", "java_legacy"):
        raise NotImplementedError(f"{profile} export is task P3-02; this starter supports static BB/Java")
    report = audit_scene(scene)
    if not report["passed"] and not allow_audit_failures:
        print(_json(report), file=sys.stderr)
        raise ValueError("geometry audit failed; no export written (use --allow-audit-failures for diagnosis)")
    atlas, png = build_atlas(program, scene)
    bbmodel = export_bbmodel(scene, name=Path(path).stem, atlas=atlas)
    java = export_java(scene, profile=profile, atlas=atlas) if profile.startswith("java_") else None
    output = Path(out)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError("output directory is not empty; use another directory or --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "asset.blockdsl": format_program(program), "ast.json": _json(asdict(program)),
        "scene.json": _json(scene.to_dict()), "audit.json": _json(report),
        "asset.bbmodel": _json(bbmodel), "atlas.png": png,
        "atlas.json": _json({k:v for k,v in atlas.items() if k != "data_url"}),
        "build_info.json": _json({
            "synccraft_version": __version__, "profile": profile,
            "scope": "starter_static_boxes", "engine_import_verified": False,
            "complete_for_release": False,
            "diagnostic_material_mode": "base_OKLCH_color_only",
            "warnings": [
                "Palette steps/hue_shift/emissive are preserved in AST but not rendered as ramps or emission.",
                "Preview uses CPU painter sorting, not ModernGL or the target engine.",
                "Java output, when present, is an individual model, not an installable resource pack.",
                "Audit coverage gaps and exempt contact pairs must be reviewed before production use.",
            ]}),
    }
    if java is not None:
        payloads["asset.java.json"] = _json(java)
    for filename, data in payloads.items():
        _atomic_write(output / filename, data)
    # Remove only stale generated Java output when explicitly overwriting for another target.
    if overwrite and java is None:
        (output / "asset.java.json").unlink(missing_ok=True)
    render_four_views(program, scene, output / "preview.png")
    return {"output": str(output.resolve()), "boxes": len(scene.boxes),
            "bones": len(scene.bones), "audit_passed": report["passed"],
            "scope": report["scope"], "engine_import_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="synccraft", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("doctor", help="report local CPU dependencies; no network requests")
    validate = subs.add_parser("validate", help="parse, perform FK, and run the limited static audit")
    validate.add_argument("input", type=Path)
    validate.add_argument("--target", choices=["bbmodel_5", "java_modern", "java_legacy"])
    compile_parser = subs.add_parser("compile", help="write model, audit, atlas, and CPU preview")
    compile_parser.add_argument("input", type=Path)
    compile_parser.add_argument("--out", type=Path, required=True)
    compile_parser.add_argument("--target", choices=["bbmodel_5", "java_modern", "java_legacy"])
    compile_parser.add_argument("--overwrite", action="store_true")
    compile_parser.add_argument("--allow-audit-failures", action="store_true")
    gate = subs.add_parser("gate-g2", help="calculate the conditional refiner branch from recorded evidence")
    gate.add_argument("evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            versions = {}
            for name in ("numpy", "Pillow", "torch", "transformers", "lark", "xgrammar", "moderngl"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = None
            print(_json({"python": sys.version.split()[0], "platform": platform.platform(),
                         "synccraft": __version__, "packages": versions,
                         "gpu_training_verified": False}))
        elif args.command == "validate":
            from synccraft.audit.sync_audit import audit_scene
            _, scene = load_scene(args.input, args.target)
            result = audit_scene(scene)
            print(_json(result))
            return 0 if result["passed"] else 2
        elif args.command == "compile":
            print(_json(compile_file(args.input, args.out, target=args.target,
                                    overwrite=args.overwrite,
                                    allow_audit_failures=args.allow_audit_failures)))
        elif args.command == "gate-g2":
            from synccraft.evaluation.gate_g2 import decide_gate
            result = decide_gate(json.loads(args.evidence.read_text(encoding="utf-8")))
            print(_json(result))
            return 2 if result["decision"] == "blocked" else 0
        return 0
    except (ValueError, OSError, NotImplementedError, RecursionError) as error:
        print(_json({"error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
