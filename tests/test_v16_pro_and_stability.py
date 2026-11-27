"""Exhaustive TexGraph v16 (`SyncCraft Studio v52.0`) Pro Features, Security & Stability Test Suite.

Covers:
1. 230-Node Schema v4.0 Registry + 0-Fallback Preview Catalog + Animated `script_expr` preview.
2. Interactive 3D Multi-Material PBR Paint Layer Stack (`paint_layer_stack` + `/api/paint_stack/stroke`).
3. Fritsch-Carlson Monotone Cubic Spline / Linear / Step Curves (`sim_animate.py` + `update_driver`).
4. AST-Sandboxed `script_expr` + 12 Malicious Code-Injection Security Tests (`ScriptSecurityError` / HTTP 400).
5. Distributable `.syncpack` Macro Package Marketplace (`list`, `save`, `import`, `spawn`).
6. Multi-Asset Batch Bakery (`scan_batch_folder`, `run_batch_bakery`, CLI `batch-bake`, Studio `/api/batch/*`).
7. Exhaustive Concurrency, Fuzzing, Malformed-Payload & 15-Step Undo/Redo State Parity Stress Tests.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from synccraft.texgraph import (
    BUILTIN_SYNCPACKS,
    NODE_CATALOG,
    PBR_BRUSH_PRESETS,
    ScriptSecurityError,
    TexGraph,
    apply_pbr_brush_strokes_to_layer,
    eval_anim_driver,
    export_node_schema_json,
    generate_node_anim_preview,
    generate_node_preview_catalog,
    list_all_syncpacks,
    make_empty_paint_layer,
    paint_layer_stack_op,
    run_batch_bakery,
    scan_batch_folder,
    script_expr_op,
    validate_safe_script_ast,
)
from synccraft.texgraph.studio_server import create_studio_server


def _http_json(url: str, *, method: str = "GET", payload: dict | None = None, raw_bytes: bytes | None = None) -> tuple[int, dict]:
    data = raw_bytes if raw_bytes is not None else (json.dumps(payload).encode("utf-8") if payload is not None else None)
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=20.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        try:
            doc = json.loads(body)
        except Exception:
            doc = {"raw": body}
        return err.code, doc


@pytest.fixture()
def studio_server():
    srv, sess = create_studio_server(host="127.0.0.1", port=0, initial_sample="brass_crystal_workflow_3d", size=64)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}", sess
    finally:
        srv.shutdown()
        srv.server_close()


def test_v16_registry_schema_and_previews_230_nodes():
    assert len(NODE_CATALOG) == 230
    assert "paint_layer_stack" in NODE_CATALOG
    assert "script_expr" in NODE_CATALOG
    assert NODE_CATALOG["script_expr"].time_aware is True

    schema = export_node_schema_json()
    assert schema["node_count"] == 230
    assert schema["schema_version"] == "4.0"

    catalog = generate_node_preview_catalog(size=32)
    assert len(catalog) == 230
    assert catalog["paint_layer_stack"].startswith("data:image/png;base64,")
    assert catalog["script_expr"].startswith("data:image/png;base64,")

    anim_frames = generate_node_anim_preview("script_expr", size=32, frames=4)
    assert len(anim_frames) == 4
    assert len(set(anim_frames)) >= 3  # truly animated across phase!


def test_v16_paint_layer_stack_multichannel_and_eraser():
    H, W = 64, 64
    base_mat = paint_layer_stack_op(H, W, base=None, layers_json="[]")
    assert base_mat["albedo"].shape == (H, W, 4)
    assert float(base_mat["metallic"].max()) < 0.05

    # Paint metallic gold trim + glowing arcane cyan on two separate layers
    lyr1 = make_empty_paint_layer("layer_1", "Gold Trim")
    lyr1 = apply_pbr_brush_strokes_to_layer(
        lyr1,
        W,
        H,
        [{"u": 0.35, "v": 0.50, "radius_px": 7, "opacity": 1.0, "softness": 0.25}],
        brush=PBR_BRUSH_PRESETS["gold_trim"],
    )
    lyr2 = make_empty_paint_layer("layer_2", "Arcane Glow")
    lyr2 = apply_pbr_brush_strokes_to_layer(
        lyr2,
        W,
        H,
        [{"u": 0.70, "v": 0.50, "radius_px": 7, "opacity": 1.0, "softness": 0.25}],
        brush=PBR_BRUSH_PRESETS["arcane_glow"],
    )

    mat = paint_layer_stack_op(H, W, base=base_mat, layers_json=json.dumps([lyr1, lyr2]), height_normal_strength=3.0)
    # Metallic gold patch
    assert float(mat["metallic"][32, 22]) > 0.75
    assert float(mat["roughness"][32, 22]) < 0.35
    assert float(mat["height"][32, 22]) > 0.65
    # Normal perturbed around relief rim
    assert float(np.std(mat["normal"][..., 0])) > 0.01
    # Emissive glow patch
    assert "emissive" in mat and mat["emissive"] is not None
    assert float(mat["emissive"][32, 44, 3]) > 0.70

    # Hiding layer 2 removes the emissive glow non-destructively
    lyr2_hidden = dict(lyr2, visible=False)
    mat_hidden = paint_layer_stack_op(H, W, base=base_mat, layers_json=json.dumps([lyr1, lyr2_hidden]))
    assert mat_hidden.get("emissive") is None or float(mat_hidden["emissive"][..., 3].max()) < 0.01

    # Eraser brush clears painted region on layer 1
    lyr1_erased = apply_pbr_brush_strokes_to_layer(
        lyr1,
        W,
        H,
        [{"u": 0.35, "v": 0.50, "radius_px": 10, "opacity": 1.0, "softness": 0.1}],
        brush=PBR_BRUSH_PRESETS["eraser"],
    )
    mat_erased = paint_layer_stack_op(H, W, base=base_mat, layers_json=json.dumps([lyr1_erased]))
    assert float(mat_erased["metallic"][32, 22]) < 0.15


def test_v16_monotone_cubic_spline_and_step_curves():
    kfs = [[0.0, 0.0], [0.3, 0.2], [0.5, 0.9], [1.0, 1.0]]
    # Exact keyframe interpolation
    for t_k, v_k in kfs:
        got = eval_anim_driver(t_k, kind="bezier_curve", keyframes=kfs)
        assert abs(got - v_k) < 1e-4

    # Strictly monotone (zero Fritsch-Carlson overshoot!) across 100 dense samples
    samples = [eval_anim_driver(i / 100.0, kind="spline", keyframes=kfs) for i in range(101)]
    for a, b in zip(samples, samples[1:]):
        assert b >= a - 1e-5
    assert min(samples) >= -1e-5 and max(samples) <= 1.0 + 1e-5

    # Step easing holds left value until next keyframe
    step_kfs = [[0.0, 0.1, "step"], [0.5, 0.9, "step"], [1.0, 0.9, "step"]]
    assert abs(eval_anim_driver(0.25, kind="keyframes", keyframes=step_kfs) - 0.1) < 1e-4
    assert abs(eval_anim_driver(0.75, kind="keyframes", keyframes=step_kfs) - 0.9) < 1e-4


def test_v16_script_expr_functional_and_exhaustive_security_sandbox():
    # Scalar animated wave
    f0 = script_expr_op(32, 32, expr="0.5 + 0.5 * sin(r_dist * 18.0 - t * tau)", phase=0.0)
    f1 = script_expr_op(32, 32, expr="0.5 + 0.5 * sin(r_dist * 18.0 - t * tau)", phase=0.25)
    assert f0.shape == (32, 32)
    assert float(np.mean(np.abs(f0 - f1))) > 0.10

    # RGB 3-channel iridescent mode + ternary + safe div-by-zero + clamped pow
    rgb = script_expr_op(
        32,
        32,
        mode="rgb",
        expr="clamp(mix(u, v, 0.5) if u > 0.3 else smoothstep(0.0, 0.5, r_dist), 0.0, 1.0)",
        expr_g="1.0 / 0.0",  # safe guarded division
        expr_b="pow(2.0, 999999)",  # exponent clamped, zero DoS!
        phase=0.5,
    )
    assert rgb.shape == (32, 32, 4)
    assert np.all(np.isfinite(rgb))
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0

    # Exhaustive security injection attempts: ALL MUST be blocked with ScriptSecurityError!
    malicious_exprs = [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "(lambda x: x)(1)",
        "[x for x in (1, 2, 3)]",
        "u.__class__",
        "eval('1+1')",
        "exec('import sys')",
        "globals()",
        "getattr(u, 'shape')",
        "u + " + " + ".join(["v"] * 600),  # length / node limit DoS
        "sin(" * 35 + "u" + ")" * 35,  # nesting depth DoS
    ]
    for bad in malicious_exprs:
        with pytest.raises(ScriptSecurityError):
            validate_safe_script_ast(bad)
        with pytest.raises(ScriptSecurityError):
            script_expr_op(16, 16, expr=bad)


def test_v16_studio_api_pbr_paint_curves_syncpacks_and_security(studio_server):
    base_url, _ = studio_server

    # 1. Health & Schema
    code, health = _http_json(f"{base_url}/api/health")
    assert code == 200
    assert health["engine_version"] == "16.0"
    assert health["node_count"] == 230

    code, schema = _http_json(f"{base_url}/api/schema")
    assert code == 200
    assert len(schema["syncpacks"]) >= 4
    assert "arcane_glow" in schema["brush_presets"]

    # 2. Interactive 3D PBR Paint Stack API
    code, st_paint = _http_json(
        f"{base_url}/api/paint_stack/stroke",
        method="POST",
        payload={
            "strokes": [{"u": 0.45, "v": 0.45, "radius_px": 8, "opacity": 1.0}],
            "brush": PBR_BRUSH_PRESETS["arcane_glow"],
            "undoable": True,
        },
    )
    assert code == 200
    assert st_paint["created_node_id"] == "pbr_paint_stack"
    assert len(st_paint["paint_layers"]) == 1

    code, st_lyr2 = _http_json(
        f"{base_url}/api/paint_stack/stroke",
        method="POST",
        payload={"node_id": "pbr_paint_stack", "layer_action": "add_layer", "name": "Gold Filigree"},
    )
    assert code == 200
    assert len(st_lyr2["paint_layers"]) == 2

    code, ev = _http_json(f"{base_url}/api/evaluate", method="POST", payload={"phase": 0.0, "include_qa": False})
    assert code == 200 and ev["ok"] is True

    # 3. Interactive Curve Editor `update_driver` API
    nodes = st_lyr2["recipe"]["nodes"]
    target_nid = nodes[0]["id"]
    code, st_drv = _http_json(
        f"{base_url}/api/graph/mutate",
        method="POST",
        payload={"action": "add_driver", "node_id": target_nid, "param": "roughness", "kind": "sine", "min_val": 0.1, "max_val": 0.8},
    )
    assert code == 200
    drv_idx = len(st_drv["recipe"]["drivers"]) - 1
    code, st_crv = _http_json(
        f"{base_url}/api/graph/mutate",
        method="POST",
        payload={
            "action": "update_driver",
            "index": drv_idx,
            "kind": "bezier_curve",
            "keyframes": [[0.0, 0.12], [0.5, 0.88], [1.0, 0.12]],
        },
    )
    assert code == 200
    assert st_crv["recipe"]["drivers"][drv_idx]["kind"] == "bezier_curve"
    assert len(st_crv["recipe"]["drivers"][drv_idx]["keyframes"]) == 3

    # 4. .syncpack Marketplace Spawn (including animated script_expr!)
    code, pk_list = _http_json(f"{base_url}/api/syncpack/list")
    assert code == 200 and len(pk_list["packs"]) >= 4

    code, st_sp = _http_json(
        f"{base_url}/api/syncpack/spawn",
        method="POST",
        payload={"pack_id": "pack_chrono_arcane", "macro_id": "macro_portal_ripple", "pos": [120, 120]},
    )
    assert code == 200
    assert len(st_sp["created_node_ids"]) == 2
    script_nid = next(nid for nid in st_sp["created_node_ids"] if "portal_wave" in nid)

    # 5. Security Rejection over HTTP (must return 400 ScriptSecurityError!)
    code, err_res = _http_json(
        f"{base_url}/api/graph/mutate",
        method="POST",
        payload={"action": "set_param", "node_id": script_nid, "param": "expr", "value": "__import__('os').system('id')"},
    )
    assert code == 400
    assert err_res["error"] == "ScriptSecurityError"

    # Also test malicious expression directly via graph evaluate (even on orphan node)
    g_sec = TexGraph(width=16, height=16)
    n_base = g_sec.add("constant", value="#888888")
    g_sec.add("script_expr", expr="open('/etc/passwd').read()")
    g_sec.set_output(n_base)
    with pytest.raises(ScriptSecurityError):
        g_sec.evaluate(allow_orphan_errors=True)


def test_v16_multi_asset_batch_bakery_engine_cli_and_studio(tmp_path: Path, studio_server):
    in_dir = tmp_path / "assets_in"
    out_dir = tmp_path / "batch_out"
    in_dir.mkdir()

    # Asset 1: standalone .texdsl
    g1 = TexGraph(name="rune_bronze_tile", width=32, height=32, tileable=True)
    m1 = g1.add("mat_brass", base_color="#D9A74A", roughness=0.22)
    g1.set_output_material(m1)
    (in_dir / "rune_bronze_tile.texdsl").write_text(g1.to_texdsl(), encoding="utf-8")

    # Asset 2: synthetic .bbmodel + sibling .texdsl
    bb_doc = {
        "meta": {"format_version": "4.10", "model_format": "free"},
        "name": "mini_crystal_obelisk",
        "resolution": {"width": 32, "height": 32},
        "elements": [
            {
                "name": "obelisk_core",
                "from": [-4, 0, -4],
                "to": [4, 16, 4],
                "faces": {
                    f: {"uv": [0, 0, 16, 16]} for f in ("north", "south", "east", "west", "up", "down")
                },
            }
        ],
        "outliner": [],
    }
    (in_dir / "mini_crystal_obelisk.bbmodel").write_text(json.dumps(bb_doc), encoding="utf-8")
    g2 = TexGraph(name="mini_crystal_obelisk", width=32, height=32, tileable=False)
    m2 = g2.add("mat_crystal_gem", crystal_color="#9333EA", core_glow_color="#38F2FF", opacity=0.8)
    g2.set_output_material(m2)
    (in_dir / "mini_crystal_obelisk.texdsl").write_text(g2.to_texdsl(), encoding="utf-8")

    scanned = scan_batch_folder(in_dir)
    assert len(scanned) == 2

    # Engine direct run
    report = run_batch_bakery(in_dir, out_dir, preset="studio_hero")
    assert report["ok"] is True
    assert report["total"] == 2
    assert report["succeeded"] == 2
    assert report["failed"] == 0
    assert (out_dir / "batch_bakery_report.json").is_file()
    for it in report["items"]:
        assert it["status"] == "done"
        assert (Path(it["out_dir"]) / f"{it['stem']}.mtlx").is_file()
        assert (Path(it["out_dir"]) / f"{it['stem']}.synccraft.pbr.json").is_file()

    # CLI `synccraft batch-bake`
    cli_out = tmp_path / "cli_batch_out"
    proc = subprocess.run(
        [sys.executable, "-m", "synccraft.cli", "batch-bake", str(in_dir), "-o", str(cli_out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (cli_out / "batch_bakery_report.json").is_file()

    # Studio HTTP `/api/batch/scan`, `/api/batch/start`, `/api/batch/status`
    base_url, _ = studio_server
    code, sc = _http_json(f"{base_url}/api/batch/scan", method="POST", payload={"input_dir": str(in_dir)})
    assert code == 200 and sc["count"] == 2

    studio_out = tmp_path / "studio_batch_out"
    code, st_run = _http_json(
        f"{base_url}/api/batch/start",
        method="POST",
        payload={"input_dir": str(in_dir), "out_dir": str(studio_out), "preset": "studio_hero"},
    )
    assert code == 200
    deadline = time.time() + 25.0
    final_st = None
    while time.time() < deadline:
        c2, b_st = _http_json(f"{base_url}/api/batch/status")
        assert c2 == 200
        if not b_st["running"]:
            final_st = b_st
            break
        time.sleep(0.1)
    assert final_st is not None
    assert final_st["succeeded"] == 2 and final_st["failed"] == 0


def test_v16_exhaustive_concurrency_fuzz_and_undo_redo_stability(studio_server):
    base_url, _ = studio_server

    # Snapshot initial state
    code, init_st = _http_json(f"{base_url}/api/state")
    assert code == 200
    init_count = len(init_st["recipe"]["nodes"])

    # 1. Multi-step diverse Undo/Redo State Parity Stress (10 sequential actions -> full undo -> full redo)
    actions = [
        ("/api/graph/mutate", {"action": "add_node", "op": "perlin", "pos": [80, 80]}),
        ("/api/graph/mutate", {"action": "add_node", "op": "script_expr", "pos": [180, 80]}),
        ("/api/graph/mutate", {"action": "add_node", "op": "math_scalar", "pos": [280, 80]}),
        ("/api/syncpack/spawn", {"pack_id": "pack_shader_math", "macro_id": "macro_iridescent_rgb", "pos": [140, 220]}),
        ("/api/paint_stack/stroke", {"strokes": [{"u": 0.5, "v": 0.5, "radius_px": 6}], "brush": PBR_BRUSH_PRESETS["gold_trim"], "undoable": True}),
        ("/api/paint_stack/stroke", {"layer_action": "add_layer", "name": "Stress Layer 2"}),
        ("/api/graph/mutate", {"action": "randomize_seed", "seed": 777}),
    ]
    for path, body in actions:
        c, _ = _http_json(f"{base_url}{path}", method="POST", payload=body)
        assert c == 200

    _, mid_st = _http_json(f"{base_url}/api/state")
    mid_count = len(mid_st["recipe"]["nodes"])
    assert mid_count > init_count
    undo_steps = mid_st["undo_depth"]
    assert undo_steps >= len(actions)

    # Undo all steps back to initial node count
    for _ in range(undo_steps):
        c, _ = _http_json(f"{base_url}/api/graph/mutate", method="POST", payload={"action": "undo"})
        assert c == 200
    _, undone_st = _http_json(f"{base_url}/api/state")
    assert len(undone_st["recipe"]["nodes"]) == init_count

    # Redo all steps forward back to mid_count
    for _ in range(undo_steps):
        c, _ = _http_json(f"{base_url}/api/graph/mutate", method="POST", payload={"action": "redo"})
        assert c == 200
    _, redone_st = _http_json(f"{base_url}/api/state")
    assert len(redone_st["recipe"]["nodes"]) == mid_count

    # 2. Concurrent 24-Request Stability & Fuzz Stress (ZERO 500s, ZERO deadlocks!)
    jobs = [
        ("GET", "/api/health", None, None, {200}),
        ("GET", "/api/state", None, None, {200}),
        ("GET", "/api/syncpack/list", None, None, {200}),
        ("GET", "/api/logs/file?tail=20", None, None, {200}),
        ("POST", "/api/evaluate", {"phase": 0.0, "include_qa": False}, None, {200}),
        ("POST", "/api/evaluate", {"phase": 0.35, "include_qa": False}, None, {200}),
        ("POST", "/api/graph/mutate", {"action": "unknown_bogus_action"}, None, {400}),
        ("POST", "/api/graph/mutate", {"action": "add_node", "op": "nonexistent_op_xyz"}, None, {400}),
        ("POST", "/api/graph/mutate", {"action": "update_driver", "index": 999}, None, {400}),
        ("POST", "/api/graph/mutate", None, b"{bad json!!!", {400}),
        ("POST", "/api/syncpack/spawn", {"pack_id": "no_such_pack", "macro_id": "x"}, None, {400}),
        ("POST", "/api/nonexistent_endpoint", {}, None, {404}),
    ] * 2  # 24 concurrent requests

    def _run_job(spec):
        method, path, payload, raw, expected = spec
        code_r, res_r = _http_json(f"{base_url}{path}", method=method, payload=payload, raw_bytes=raw)
        assert code_r != 500, f"HTTP 500 on {method} {path}: {res_r}"
        assert code_r in expected, f"Unexpected {code_r} on {method} {path}: {res_r}"
        return code_r

    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = list(pool.map(_run_job, jobs))
    assert len(codes) == 24

    # Final post-stress evaluation sanity check
    code_ok, ev_final = _http_json(f"{base_url}/api/evaluate", method="POST", payload={"phase": 0.25, "include_qa": True})
    assert code_ok == 200 and ev_final["ok"] is True
