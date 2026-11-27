"""Comprehensive test suite for SyncCraft Studio v51.0 / TexGraph v15 Commercial Workstation.

Verifies:
1. 228-node registry (100% Schema v4.0) + all 15 Blender-style Math/Vector/Mapping/Utility & `paint_mask` ops
2. Real procedural node previews for all 17 new v15 ops
3. Nested Compound Subgraph Collapse (`group_nodes_into_subgraph`), recursive evaluation & lossless `ungroup_subgraph_node`
4. Connected Block Tileset (CTM 47-Tile & Wang 16-Tile) batch generator + OptiFine `.properties` & disk export
5. Live HTTP Studio v15 APIs:
   - `POST /api/evaluate_anim_sequence` (GPU 30-60 FPS animation ring buffer)
   - `POST /api/paint_mask/stroke` (3D/2D brush painting auto-spawn, paint & clear)
   - `group_nodes` -> `enter_group` (breadcrumbs) -> `exit_group` -> `ungroup_node` & `insert_node_on_wire`
   - `POST /api/export/ctm` (`ctm47` & `wang16`)
   - `POST /api/export/turntable_gif` (GIF89a multi-frame encoding)
   - `GET /api/bridge/latest` (Blockbench 5 Live Link snapshot + CORS header)
   - `extract_bbmodel_webgl_mesh` `uv_rects` for 2D UV Wireframe overlay
"""
from __future__ import annotations

import base64
import io
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from synccraft.texgraph import (
    CTM_47_MASKS,
    NODE_CATALOG,
    TexGraph,
    WANG_16_MASKS,
    create_studio_server,
    export_node_schema_json,
    extract_bbmodel_webgl_mesh,
    generate_ctm_tileset,
    generate_node_preview_catalog,
)
from synccraft.texgraph.studio_server import build_sample_monolith_bbmodel


def _http_json(url: str, method: str = "GET", payload: dict | None = None) -> tuple[dict, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        return json.loads(resp.read().decode("utf-8")), dict(resp.headers.items())


def test_registry_228_nodes_and_blender_math_ops() -> None:
    schema = export_node_schema_json()
    assert schema["node_count"] >= 228
    assert schema["param_schema_coverage_pct"] == 100.0

    new_ops = {
        "math_scalar", "map_range", "clamp_value", "mix_blend",
        "rgb_combine", "rgb_separate", "vector_math", "uv_mapping",
        "tex_coord", "float_constant", "color_constant", "invert_field",
        "fresnel_weight", "checker_basic", "paint_mask", "group_input", "subgraph_group",
    }
    assert new_ops <= set(NODE_CATALOG.keys())

    # Evaluate a chained Blender-style shader math pipeline
    g = TexGraph(width=32, height=32, seed=42, tileable=True, name="blender_math_chain")
    tc = g.add("tex_coord", id="tc", coord_type="radial_dist")
    fc = g.add("float_constant", id="fc", value=0.6)
    ms = g.add("math_scalar", id="ms", inputs={"a": tc, "b": fc}, math_op="multiply")
    mr = g.add("map_range", id="mr", inputs={"value": ms}, from_min=0.0, from_max=0.6, to_min=0.1, to_max=0.95, interpolation="smoothstep")
    cl = g.add("clamp_value", id="cl", inputs={"value": mr}, min_val=0.15, max_val=0.85)
    inv = g.add("invert_field", id="inv", inputs={"src": cl}, fac=1.0)
    chk = g.add("checker_basic", id="chk", scale_x=4, scale_y=4)
    c_const = g.add("color_constant", id="cc", color="#38BDF8")
    mx = g.add("mix_blend", id="mx", inputs={"a": chk, "b": c_const, "factor": inv}, blend_mode="overlay")
    sep = g.add("rgb_separate", id="sep", inputs={"color": mx}, channel="luma")
    comb = g.add("rgb_combine", id="comb", inputs={"r": sep, "g": inv, "b": cl}, mode="rgb")
    uv_m = g.add("uv_mapping", id="uvm", inputs={"src": comb}, rotation_deg=25.0, scale_x=1.5, scale_y=1.5)
    fr = g.add("fresnel_weight", id="fr", ior=1.5, power=2.5)
    vm = g.add("vector_math", id="vm", vec_op="normalize")
    pm = g.add("paint_mask", id="pm", base_value=0.4)
    g.set_output(uv_m)

    assert g.validate_graph().ok
    ev = g.evaluate(phase=0.0, capture_intermediates=True, build_pbr=False)
    assert ev.output.shape == (32, 32, 4)
    assert float(np.std(ev.output[..., :3])) > 0.02
    assert ev.intermediates[fr].shape == (32, 32)
    assert ev.intermediates[vm].shape == (32, 32, 3)
    assert np.allclose(ev.intermediates[pm], 0.4, atol=1e-2)

    previews = generate_node_preview_catalog(size=32)
    for op in new_ops:
        assert op in previews and previews[op].startswith("data:image/png;base64,")


def test_compound_subgraph_group_and_ungroup_lossless() -> None:
    g = TexGraph(width=32, height=32, seed=42, tileable=True, name="group_test")
    n1 = g.add("perlin", id="n1", scale=5.0, seed=42)
    m1 = g.add("map_range", id="m1", inputs={"value": n1}, from_min=0.1, from_max=0.9, interpolation="smoothstep")
    inv = g.add("invert_field", id="inv", inputs={"src": m1}, fac=1.0)
    col = g.add("multi_stop_gradient", id="col", inputs={"src": inv}, stops=["#121826", "#38BDF8", "#F8FAFC"])
    g.set_output(col)

    before = g.evaluate(phase=0.0, build_pbr=False).output.copy()

    gid = g.group_nodes_into_subgraph([m1, inv], title="Remap + Invert")
    assert gid in g.nodes and g.nodes[gid].op == "subgraph_group"
    assert m1 not in g.nodes and inv not in g.nodes
    after_group = g.evaluate(phase=0.0, build_pbr=False).output
    assert np.allclose(before, after_group, atol=1e-4)

    restored = g.ungroup_subgraph_node(gid)
    assert len(restored) == 2
    assert gid not in g.nodes
    after_ungroup = g.evaluate(phase=0.0, build_pbr=False).output
    assert np.allclose(before, after_ungroup, atol=1e-4)


def test_ctm_47_and_wang_16_tileset_generator(tmp_path: Path) -> None:
    assert len(CTM_47_MASKS) == 47
    assert len(WANG_16_MASKS) == 16

    base = np.full((16, 16, 4), [0.35, 0.42, 0.56, 1.0], dtype=np.float32)
    res47 = generate_ctm_tileset(base, mode="ctm47", style="metal_frame", edge_width_px=2, out_dir=tmp_path / "ctm47", stem="chronal_stone")
    assert res47["ok"] and res47["tile_count"] == 47
    assert res47["sheet_u8"].shape == (4 * 16, 12 * 16, 4)
    assert "method=ctm" in res47["optifine_properties"]
    assert "tiles=0-46" in res47["optifine_properties"]
    assert (tmp_path / "ctm47" / "chronal_stone_ctm47_sheet.png").is_file()
    assert (tmp_path / "ctm47" / "tiles" / "46.png").is_file()

    res16 = generate_ctm_tileset(base, mode="wang16", style="moss_overhang", edge_width_px=2)
    assert res16["ok"] and res16["tile_count"] == 16
    assert res16["sheet_u8"].shape == (4 * 16, 4 * 16, 4)


def test_webgl_mesh_uv_rects_present() -> None:
    mesh = extract_bbmodel_webgl_mesh(build_sample_monolith_bbmodel())
    assert "uv_rects" in mesh and len(mesh["uv_rects"]) >= 12
    assert all(len(r) == 4 and r[2] >= r[0] and r[3] >= r[1] for r in mesh["uv_rects"])


def test_studio_v15_http_api_end_to_end() -> None:
    server, _ = create_studio_server(host="127.0.0.1", port=0, initial_sample="brass_crystal_workflow_3d", size=32)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        # 1. Pre-baked 30-60 FPS animation ring sequence
        seq, _ = _http_json(f"{base}/api/evaluate_anim_sequence", "POST", {"frames": 6, "channel": "pbr_lit", "max_dim": 128})
        assert seq["ok"] and seq["frames"] == 6 and len(seq["sequence"]) == 6
        assert seq["sequence"][0]["albedo"].startswith("data:image/png;base64,")
        assert isinstance(seq["sequence"][0]["thumbs"], dict) and len(seq["sequence"][0]["thumbs"]) >= 1

        # 2. Direct 3D/2D Brush Mask Painting (auto-spawn + stroke + clear)
        st_p, _ = _http_json(
            f"{base}/api/paint_mask/stroke",
            "POST",
            {"strokes": [{"u": 0.5, "v": 0.5, "radius_px": 8, "value": 1.0}], "undoable": True},
        )
        pm_id = st_p["painted_node_id"]
        assert any(n["id"] == pm_id and n["op"] == "paint_mask" for n in st_p["recipe"]["nodes"])
        pm_node = next(n for n in st_p["recipe"]["nodes"] if n["id"] == pm_id)
        assert pm_node["params"]["mask_data_url"].startswith("data:image/png;base64,")

        # 3. Compound Subgraph Group -> Enter (Breadcrumbs) -> Exit -> Ungroup
        st_cur, _ = _http_json(f"{base}/api/state")
        pair = [st_cur["recipe"]["nodes"][0]["id"], st_cur["recipe"]["nodes"][1]["id"]]
        st_grp, _ = _http_json(f"{base}/api/graph/mutate", "POST", {"action": "group_nodes", "ids": pair, "title": "Mask Pair"})
        gid = st_grp["created_node_id"]
        assert any(n["id"] == gid and n["op"] == "subgraph_group" for n in st_grp["recipe"]["nodes"])

        st_in, _ = _http_json(f"{base}/api/graph/mutate", "POST", {"action": "enter_group", "node_id": gid})
        assert st_in["in_subgraph"] is True and len(st_in["breadcrumbs"]) == 2

        st_out, _ = _http_json(f"{base}/api/graph/mutate", "POST", {"action": "exit_group", "level": 0})
        assert st_out["in_subgraph"] is False and len(st_out["breadcrumbs"]) == 1

        st_un, _ = _http_json(f"{base}/api/graph/mutate", "POST", {"action": "ungroup_node", "node_id": gid})
        assert not any(n["id"] == gid for n in st_un["recipe"]["nodes"])

        # 4. CTM 47-Tile HTTP Export
        ctm, _ = _http_json(f"{base}/api/export/ctm", "POST", {"mode": "ctm47", "stem": "brass_tile"})
        assert ctm["ok"] and ctm["tile_count"] == 47 and ctm["sheet_data_url"].startswith("data:image/png;base64,")

        # 5. 360° Turntable GIF Encoder
        dummy_frames = []
        for k in range(4):
            im = Image.new("RGBA", (32, 32), (40 + k * 40, 80, 180, 255))
            b = io.BytesIO()
            im.save(b, format="PNG")
            dummy_frames.append("data:image/png;base64," + base64.b64encode(b.getvalue()).decode("ascii"))
        gif_res, _ = _http_json(f"{base}/api/export/turntable_gif", "POST", {"frames": dummy_frames, "fps": 12, "stem": "demo"})
        assert gif_res["ok"] and gif_res["frame_count"] == 4 and gif_res["gif_data_url"].startswith("data:image/gif;base64,")
        gif_bytes = base64.b64decode(gif_res["gif_data_url"].split(",", 1)[1])
        assert gif_bytes[:4] == b"GIF8"

        # 6. Blockbench 5 Live Link Bridge Snapshot + CORS
        _http_json(f"{base}/api/evaluate", "POST", {"phase": 0.0, "include_qa": False})
        br, hdrs = _http_json(f"{base}/api/bridge/latest?since=-1")
        assert br["ok"] and br["changed"] is True and br["rev"] >= 1
        assert "albedo" in br["channels"]
        assert hdrs.get("Access-Control-Allow-Origin") == "*"

        # 7. Persistent Studio File Log API
        lg, _ = _http_json(f"{base}/api/logs/file?tail=50")
        assert lg["ok"] and lg["path"].endswith("synccraft_studio.log") and lg["line_count"] >= 1
        hl, _ = _http_json(f"{base}/api/health")
        assert hl["ok"] and hl["engine_version"] == "15.1" and hl["workspace_epoch"] >= 1
    finally:
        server.shutdown()
        server.server_close()


def test_v15_1_loading_integrity_epoch_and_cache_isolation(tmp_path: Path) -> None:
    """Verify strict 1-to-1 project loading, zero cross-project mesh/texture bleed, monotonic workspace_epoch & mesh bake cache parity."""
    from synccraft.texgraph.mesh_baker import bake_bbmodel_maps, clear_mesh_bake_cache
    from synccraft.texgraph.studio_server import StudioSession, build_sample_monolith_bbmodel

    # 1. Mesh bake SHA-256 fingerprint + LRU & Disk .npz cache parity
    bb = build_sample_monolith_bbmodel()
    clear_mesh_bake_cache(clear_disk=False)
    mc1 = bake_bbmodel_maps(bb, use_cache=True)
    assert len(mc1.content_fingerprint) == 32
    mc2 = bake_bbmodel_maps(bb, use_cache=True)
    assert mc2.bake_cache_hit is True and mc2.content_fingerprint == mc1.content_fingerprint
    assert np.array_equal(mc1.baked_ao, mc2.baked_ao) and np.array_equal(mc1.baked_shadow, mc2.baked_shadow)

    # 2. Strict 1-to-1 loading: standalone .texdsl MUST detach previous project's 3D .bbmodel unless keep_mesh=True!
    sess = StudioSession(initial_sample="brass_crystal_workflow_3d", size=32)
    ep0 = sess._workspace_epoch
    assert sess.mesh_ctx is not None and sess.bbmodel_doc is not None

    dsl_code = """
@texture { name: "pure_2d_block" size: [32, 32] seed: 7 tileable: true }
@pass main "Main" {
  n = perlin(scale=4.0, seed=7)
  c = multi_stop_gradient(n, stops=["#111827", "#38BDF8", "#F8FAFC"])
}
@output albedo(c)
"""
    st_2d = sess.load_project({"kind": "texdsl", "code": dsl_code, "filename": "pure_2d_block.texdsl"})
    assert st_2d["workspace_epoch"] > ep0
    assert st_2d["has_mesh"] is False and st_2d["mesh"] is None
    assert st_2d["recipe"]["width"] == 32 and st_2d["recipe"]["height"] == 32
    ev_2d = sess.evaluate({"phase": 0.0, "include_qa": False})
    assert ev_2d["workspace_epoch"] == st_2d["workspace_epoch"]
    assert ev_2d["width"] == 32 and ev_2d["height"] == 32

    # 3. Standalone .bbmodel 1-to-1 project load preserves exact UV dims & builds clean starter graph
    st_bb = sess.load_project({"kind": "bbmodel_project", "bbmodel": bb, "filename": "monolith.bbmodel"})
    assert st_bb["workspace_epoch"] > st_2d["workspace_epoch"]
    assert st_bb["has_mesh"] is True and st_bb["mesh_fingerprint"] == mc1.content_fingerprint
    assert st_bb["recipe"]["width"] == mc1.width and st_bb["recipe"]["height"] == mc1.height

    # 4. Disk file 1-to-1 sibling auto-pairing (.texdsl + sibling .bbmodel)
    (tmp_path / "paired.bbmodel").write_text(json.dumps(bb), encoding="utf-8")
    (tmp_path / "paired.texdsl").write_text(dsl_code, encoding="utf-8")
    st_disk = sess.load_project({"kind": "disk_file", "path": str(tmp_path / "paired.texdsl")})
    assert st_disk["has_mesh"] is True and st_disk["mesh_fingerprint"] == mc1.content_fingerprint
    assert st_disk["workspace_epoch"] > st_bb["workspace_epoch"]

    # 5. Evaluating twice with same state hits channel PNG cache & increments monotonic eval_seq
    ev_a = sess.evaluate({"phase": 0.0, "include_qa": False})
    ev_b = sess.evaluate({"phase": 0.0, "include_qa": False})
    assert ev_b["eval_seq"] > ev_a["eval_seq"]
    assert ev_a["channels"]["albedo"] == ev_b["channels"]["albedo"]
