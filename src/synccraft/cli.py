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
    bb_srv = subs.add_parser(
        "serve-blockbench",
        help="run the 127.0.0.1 HTTP bridge for the SyncCraft Blockbench 5 plugin",
    )
    bb_srv.add_argument("--host", default="127.0.0.1")
    bb_srv.add_argument("--port", type=int, default=18765)

    studio_cmd = subs.add_parser(
        "studio",
        help="launch the interactive SyncCraft Studio 5-pane Node-Graph & 3D PBR Workstation UI",
    )
    studio_cmd.add_argument("--host", default="127.0.0.1")
    studio_cmd.add_argument("--port", type=int, default=18790)
    studio_cmd.add_argument("--sample", default="kaelith_dragonborn_3d", help="initial Studio project sample or preset name")
    studio_cmd.add_argument("--size", type=int, default=64, help="default canvas resolution for procedural blocks/items")
    studio_cmd.add_argument("--open", action="store_true", help="open default browser automatically")

    # Parametric Pixel Composer & Semantic UV v2 CLI
    tg = subs.add_parser(
        "texgraph",
        help="Parametric Pixel Composer (render recipes/presets, animate flipbooks, repack .bbmodel UVs, HTML inspector)",
    )
    tg_subs = tg.add_subparsers(dest="tg_cmd", required=True)
    tg_render = tg_subs.add_parser("render", help="render a .texdsl program, .texgraph.json recipe, or built-in preset")
    tg_render.add_argument("source", help="path to .texdsl program, .texgraph.json recipe, or preset:<name>")
    tg_render.add_argument("-o", "--out", type=Path, required=True, help="output PNG path")
    tg_render.add_argument("--bbmodel", type=Path, help="optional .bbmodel to bake 3D maps from and embed texture into")
    tg_render.add_argument("--out-bbmodel", type=Path, help="optional path to write updated .bbmodel")
    tg_render.add_argument("--frames", type=int, default=None, help="number of flipbook frames (>1 writes .mcmeta strip; defaults to @texture frames)")
    tg_render.add_argument("--frametime", type=int, default=None)
    tg_render.add_argument("--gif", type=Path, help="optional animated GIF preview path")
    tg_render.add_argument("--inspector", type=Path, help="optional interactive HTML node-graph inspector output")
    tg_render.add_argument("--size", type=int, default=32, help="canvas size for built-in block/item presets")
    tg_render.add_argument("--pbr", action="store_true", help="export full LabPBR 1.3 (_n, _s), Bedrock RTX (_mer) & standard PBR maps")
    tg_render.add_argument("--multipass-sheet", type=Path, help="optional PNG path for Multi-Pass Pipeline Progression Contact Sheet")

    tg_repack = tg_subs.add_parser("repack-uv", help="repack an existing .bbmodel into semantic anatomical UV partitions")
    tg_repack.add_argument("input", type=Path)
    tg_repack.add_argument("-o", "--out", type=Path, required=True)
    tg_repack.add_argument("--mirror", choices=["adaptive", "all", "none"], default="adaptive")
    tg_repack.add_argument("--canvas", choices=["tight", "pot"], default="tight", help="atlas dimension mode: flexible 16px-aligned tight or power-of-two")
    tg_repack.add_argument("--padding", type=int, default=1)
    tg_repack.add_argument("--zone-map", type=Path, help="optional PNG path for visual UV Zone Diagnostic Map")

    tg_bake = tg_subs.add_parser("bake", help="export Baked Studio Albedo (_baked.png), .synccraft.pbr.json manifest & optional _baked.bbmodel")
    tg_bake.add_argument("source", help="path to .texdsl program, .texgraph.json recipe, or preset:<name>")
    tg_bake.add_argument("-o", "--out-dir", type=Path, required=True, help="output directory for baked PBR package")
    tg_bake.add_argument("--stem", help="output file stem (defaults to graph name)")
    tg_bake.add_argument("--preset", default="studio_hero", choices=["studio_hero", "dramatic_rim", "high_noon", "moonlit_obsidian"])
    tg_bake.add_argument("--bbmodel", type=Path, help="optional .bbmodel to bake 3D maps from and write <stem>_baked.bbmodel")
    tg_bake.add_argument("--size", type=int, default=32)
    tg_bake.add_argument("--embed-manifest", action="store_true", help="embed PBR maps as data URLs inside .synccraft.pbr.json for single-file Blockbench import")

    tg_subs.add_parser("presets", help="list available built-in TexGraph presets")
    tg_subs.add_parser("schema", help="emit UI-ready typed JSON schema for all TexGraph nodes")

    tg_val = tg_subs.add_parser("validate", help="statically validate a .texdsl program, .texgraph.json recipe, or preset DAG")
    tg_val.add_argument("source", help="path to .texdsl program, .texgraph.json recipe, or preset:<name>")
    tg_val.add_argument("--size", type=int, default=32)

    tg_qa = tg_subs.add_parser("qa", help="evaluate multi-domain texture quality (block/item/entity) & node profile")
    tg_qa.add_argument("source", help="path to .texdsl program, .texgraph.json recipe, or preset:<name>")
    tg_qa.add_argument("--domain", choices=["block", "item", "entity"], default=None)
    tg_qa.add_argument("--bbmodel", type=Path, help="optional .bbmodel to bake 3D maps from for entity QA")
    tg_qa.add_argument("--size", type=int, default=32)

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
        elif args.command == "serve-blockbench":
            from synccraft.backends.blockbench_plugin_bridge import serve_blockbench_bridge
            serve_blockbench_bridge(host=args.host, port=args.port)
            return 0
        elif args.command == "studio":
            from synccraft.texgraph.studio_server import serve_studio
            serve_studio(
                host=args.host,
                port=args.port,
                initial_sample=args.sample,
                size=args.size,
                open_browser=bool(args.open),
            )
            return 0
        elif args.command == "texgraph":
            from synccraft.shading.uv_packer import repack_bbmodel_uvs
            from synccraft.texgraph import (
                PRESET_REGISTRY,
                TexGraph,
                bake_bbmodel_maps,
                encode_png,
                export_html_inspector,
            )

            def _load_tg(src: str, *, size: int = 32, mesh_ctx=None):
                if src.startswith("preset:"):
                    pname = src.split(":", 1)[1]
                    if pname not in PRESET_REGISTRY:
                        raise ValueError(f"Unknown preset {pname!r}; choices: {sorted(PRESET_REGISTRY)}")
                    return PRESET_REGISTRY[pname](size=size)
                return TexGraph.load(src, mesh_context=mesh_ctx)

            if args.tg_cmd == "bake":
                mesh_ctx = None
                bb_doc = None
                if args.bbmodel:
                    bb_doc = json.loads(args.bbmodel.read_text(encoding="utf-8"))
                    mesh_ctx = bake_bbmodel_maps(bb_doc)
                graph = _load_tg(args.source, size=args.size, mesh_ctx=mesh_ctx)
                if mesh_ctx is not None:
                    graph.attach_mesh(mesh_ctx)
                baked = graph.export_baked_pbr(
                    args.out_dir,
                    stem=args.stem or graph.name,
                    preset=args.preset,
                    bbmodel_doc=bb_doc,
                    recipe_path=args.source if not args.source.startswith("preset:") else None,
                    embed_maps_in_manifest=bool(args.embed_manifest),
                )
                print(_json({"mode": "baked_pbr", "preset": args.preset, "files": baked["files"]}))
                return 0
            if args.tg_cmd == "presets":
                from synccraft.texgraph.pipeline import SUBGRAPH_REGISTRY
                print(_json({"presets": sorted(PRESET_REGISTRY.keys()), "subgraphs": sorted(SUBGRAPH_REGISTRY.keys())}))
                return 0
            if args.tg_cmd == "schema":
                from synccraft.texgraph.registry import export_node_schema_json
                print(_json(export_node_schema_json()))
                return 0
            if args.tg_cmd == "validate":
                graph = _load_tg(args.source, size=args.size)
                rep = graph.validate_graph()
                print(_json(rep.to_dict()))
                return 0 if rep.ok else 2
            if args.tg_cmd == "qa":
                from synccraft.texgraph.quality import evaluate_texture_quality
                mesh_ctx = None
                if args.bbmodel:
                    bb_doc = json.loads(args.bbmodel.read_text(encoding="utf-8"))
                    mesh_ctx = bake_bbmodel_maps(bb_doc)
                graph = _load_tg(args.source, size=args.size, mesh_ctx=mesh_ctx)
                if mesh_ctx is not None:
                    graph.attach_mesh(mesh_ctx)
                vrep = graph.validate_graph()
                ev = graph.evaluate(build_pbr=True)
                dom = args.domain or getattr(graph, "domain", None) or ("entity" if mesh_ctx is not None else "block")
                qa = evaluate_texture_quality(
                    ev.output,
                    domain=dom,
                    tileable=graph.tileable,
                    mesh_ctx=graph.mesh_context,
                    pbr_channels=ev.pbr_channels,
                    pbr_maps=ev.pbr_maps,
                )
                print(_json({"validation": vrep.to_dict(), "quality": qa.to_dict(), "profile": graph.profile()}))
                return 0 if (vrep.ok and qa.passed) else 2
            if args.tg_cmd == "repack-uv":
                doc = json.loads(args.input.read_text(encoding="utf-8"))
                new_doc, rep, zpng = repack_bbmodel_uvs(
                    doc,
                    padding=args.padding,
                    mirror_lr_uv=args.mirror,
                    canvas_mode=args.canvas,
                )
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(new_doc, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
                if args.zone_map:
                    args.zone_map.parent.mkdir(parents=True, exist_ok=True)
                    args.zone_map.write_bytes(zpng)
                print(_json({"before": rep["before"], "after": rep["after"], "metrics": rep["metrics"]}))
                return 0
            if args.tg_cmd == "render":
                mesh_ctx = None
                bb_doc = None
                if args.bbmodel:
                    bb_doc = json.loads(args.bbmodel.read_text(encoding="utf-8"))
                    mesh_ctx = bake_bbmodel_maps(bb_doc)
                graph = _load_tg(args.source, size=args.size, mesh_ctx=mesh_ctx)
                if mesh_ctx is not None:
                    graph.attach_mesh(mesh_ctx)
                anim_cfg = getattr(graph, "animation_config", {}) or {}
                eff_frames = int(args.frames if args.frames is not None else anim_cfg.get("frames", 1))
                eff_frametime = int(args.frametime if args.frametime is not None else anim_cfg.get("frametime", 2))
                eff_interp = bool(anim_cfg.get("interpolate", False))
                eff_sched = anim_cfg.get("frames_schedule")
                frames_list = None
                if eff_frames > 1:
                    if args.pbr:
                        fb = graph.render_pbr_flipbook(
                            n_frames=eff_frames,
                            out_dir=args.out.parent,
                            stem=args.out.stem,
                            frametime=eff_frametime,
                            interpolate=eff_interp,
                            frames_schedule=eff_sched,
                        )
                        frames_list = fb["pbr_lit_frames"]
                        summary = {"mode": "pbr_flipbook", "frames": fb["num_frames"], "out_dir": str(args.out.parent), "pbr_lit_gif": fb["pbr_lit_gif"]}
                    else:
                        fb = graph.render_flipbook(
                            n_frames=eff_frames,
                            frametime=eff_frametime,
                            interpolate=eff_interp,
                            frames_schedule=eff_sched,
                            out_png_path=args.out,
                            out_gif_path=args.gif,
                        )
                        frames_list = [graph.evaluate(phase=k / float(eff_frames)).output for k in range(eff_frames)]
                        summary = {"mode": "flipbook", "frames": fb["num_frames"], "out": str(args.out)}
                else:
                    ev = graph.evaluate(phase=0.0, capture_intermediates=bool(args.inspector), build_pbr=bool(args.pbr or args.inspector))
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    if args.pbr:
                        bundle = graph.export_pbr_bundle(args.out.parent, stem=args.out.stem)
                        summary = {"mode": "static_pbr", "width": graph.width, "height": graph.height, "files": bundle["files"], "nodes": len(graph.node_order)}
                    else:
                        args.out.write_bytes(encode_png(ev.output))
                        if ev.emissive is not None:
                            em_path = args.out.with_name(f"{args.out.stem}_emissive.png")
                            em_path.write_bytes(encode_png(ev.emissive))
                        summary = {"mode": "static", "width": graph.width, "height": graph.height, "out": str(args.out), "nodes": len(graph.node_order)}
                    if bb_doc is not None and args.out_bbmodel:
                        new_bb, _, _ = graph.apply_to_bbmodel(bb_doc)
                        args.out_bbmodel.parent.mkdir(parents=True, exist_ok=True)
                        args.out_bbmodel.write_text(json.dumps(new_bb, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
                if args.inspector:
                    export_html_inspector(graph, args.inspector, flipbook_frames=frames_list)
                    summary["inspector"] = str(args.inspector)
                if args.multipass_sheet:
                    mp, _ = graph.render_multipass_sheet(args.multipass_sheet)
                    summary["multipass_sheet"] = str(args.multipass_sheet)
                    summary["passes"] = [{"name": ps.name, "title": ps.title, "nodes": ps.node_count, "ms": ps.time_ms} for ps in mp.passes]
                print(_json(summary))
                return 0
        return 0
    except (ValueError, OSError, NotImplementedError, RecursionError) as error:
        print(_json({"error": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
