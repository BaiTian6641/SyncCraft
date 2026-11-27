"""Comprehensive test suite for SyncCraft Studio v50.0 / TexGraph v14 Commercial Features.

Verifies:
1. 211-node registry + `reroute` zero-overhead pass-through (RGBA, SCALAR, MASK, full MATERIAL bus)
2. Comment Frames (`TexGraph.frames`) serialization & `create_frame` / `update_frame` / `delete_frame`
3. Live Animated Preview Swatch Sheet (`generate_anim_preview_sheet` + `/api/schema/preview_anim_sheet`)
4. Reusable `.syncmat` Subgraph/Material Presets (`BUILTIN_SYNCMATS`, extract, insert, save, HTTP API)
5. ASWF MaterialX 1.38 / OpenPBR & Autodesk Standard Surface XML export (`.mtlx`, XML-validated)
6. GPU Color-ID 3D Picking metadata (`pick_ids` + `elem_meta`) & 2D UV Picking (`/api/mesh/pick_uv`)
7. Live Dynamic Thumbnail Evaluation Mode (`include_thumbs_mode="dynamic"`) & multi-node bypass/reroute
8. Upgraded Kaelith Dragonborn `.texdsl` featuring `mat_brass`, `mat_velvet_silk`, `mat_crystal_gem`
"""
from __future__ import annotations

import json
import threading
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from synccraft.texgraph import (
    NODE_CATALOG,
    TexGraph,
    bake_bbmodel_maps,
    create_studio_server,
    extract_bbmodel_webgl_mesh,
    generate_anim_preview_sheet,
    load_texdsl,
)
from synccraft.texgraph.quality import evaluate_texture_quality
from synccraft.texgraph.studio_server import build_sample_monolith_bbmodel
from synccraft.texgraph.syncmat_and_mtlx import (
    BUILTIN_SYNCMATS,
    export_materialx_xml,
    extract_syncmat_from_graph,
    insert_syncmat_into_graph,
    list_all_syncmats,
    save_user_syncmat,
)


def _http_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_registry_has_211_nodes_and_reroute_passthrough() -> None:
    assert len(NODE_CATALOG) >= 211
    assert "reroute" in NODE_CATALOG
    assert NODE_CATALOG["reroute"].category == "Compositing"

    # Test reroute passing a full 12-channel MATERIAL bus without losing channels
    g = TexGraph(width=32, height=32, seed=42, tileable=True, name="reroute_bus_test")
    g.add("mat_brass", id="brass", seed=17)
    g.add("reroute", id="dot1", inputs={"in": "brass"})
    g.add("material_weathering", id="aged", inputs={"mat": "dot1"}, preset="patina", amount=0.3)
    g.set_output_material("aged")
    assert g.validate_graph().ok
    ev = g.evaluate(phase=0.0, build_pbr=True)
    assert ev.output.shape == (32, 32, 4)
    assert ev.pbr_maps is not None and "metallic_u8" in ev.pbr_maps
    assert float(np.mean(ev.pbr_maps["metallic_u8"])) > 100.0


def test_frames_serialization_roundtrip() -> None:
    g = TexGraph(width=32, height=32, seed=7, name="frames_test")
    g.add("perlin_noise", id="n1", scale=4.0, seed=7)
    g.add("color_ramp", id="c1", inputs={"scalar": "n1"}, stops=["#111122", "#AABBFF"])
    g.set_output("c1")
    g.frames.append({"id": "fr_1", "title": "Base Noise", "color": "#2563EB", "node_ids": ["n1", "c1"]})

    rec = g.to_recipe()
    assert len(rec["frames"]) == 1
    assert rec["frames"][0]["title"] == "Base Noise"

    g2 = TexGraph.from_recipe(rec)
    assert len(g2.frames) == 1
    assert g2.frames[0]["node_ids"] == ["n1", "c1"]
    assert g2.frames[0]["color"] == "#2563EB"


def test_anim_preview_sheet_generation() -> None:
    sheet = generate_anim_preview_sheet(size=24, frames=4)
    assert len(sheet) >= 12
    for op in ("anim_flicker_pulse", "anim_shimmer", "mat_crystal_gem", "caustic_noise"):
        assert op in sheet, f"Missing animated op {op!r} from anim preview sheet"
        assert len(sheet[op]) == 4
        assert all(u.startswith("data:image/png;base64,") for u in sheet[op])
        # Verify frames actually vary visually (not frozen duplicates)
        assert len(set(sheet[op])) >= 2


def test_syncmat_extract_insert_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import synccraft.texgraph.syncmat_and_mtlx as sm_mod

    monkeypatch.setattr(sm_mod, "USER_SYNCMAT_DIR", tmp_path)
    assert len(BUILTIN_SYNCMATS) >= 6
    ids = set(BUILTIN_SYNCMATS.keys())
    assert {"aged_chronal_brass", "amethyst_chronal_crystal", "frosted_arcane_glass"} <= ids

    g = TexGraph(width=32, height=32, seed=42, name="syncmat_host")
    g.add("perlin_noise", id="base_n", scale=5.0)
    g.set_output("base_n")

    preset = BUILTIN_SYNCMATS["aged_chronal_brass"]
    created, out_id, frame = insert_syncmat_into_graph(g, preset, pos=(120.0, 80.0), wrap_frame=True)
    assert len(created) == len(preset["nodes"])
    assert out_id in g.nodes
    assert frame is not None and frame in g.frames
    # Re-insert a second copy — must not collide with existing node IDs!
    created2, out_id2, _ = insert_syncmat_into_graph(g, preset, pos=(420.0, 80.0), wrap_frame=True)
    assert set(created).isdisjoint(set(created2))
    assert out_id2 != out_id

    extracted = extract_syncmat_from_graph(g, node_ids=created, name="My Brass Copy")
    assert extracted["schema"] == "synccraft.syncmat/1.0"
    assert len(extracted["nodes"]) == len(created)

    saved_file = save_user_syncmat(extracted)
    assert saved_file.is_file()
    listed = list_all_syncmats()
    assert any(p["id"] == extracted["id"] and not p["builtin"] for p in listed)


@pytest.mark.parametrize("shader_model", ["open_pbr_surface", "standard_surface"])
def test_materialx_138_xml_valid(shader_model: str) -> None:
    xml_str = export_materialx_xml(
        "kaelith_dragonborn",
        shader_model=shader_model,
        has_emissive=True,
        has_opacity=True,
        coat_weight=0.35,
        transmission_weight=0.40,
        subsurface_weight=0.25,
    )
    root = ET.fromstring(xml_str)
    assert root.tag == "materialx"
    assert root.attrib.get("version") in ("1.38", "1.39")
    assert root.find("surfacematerial") is not None
    assert root.find(shader_model) is not None
    img_nodes = root.findall(".//image")
    assert len(img_nodes) >= 6
    assert "kaelith_dragonborn.png" in xml_str
    assert "kaelith_dragonborn_normal.png" in xml_str


def test_webgl_mesh_pick_ids_and_elem_meta() -> None:
    bb = build_sample_monolith_bbmodel()
    mesh = extract_bbmodel_webgl_mesh(bb)
    assert mesh["vertex_count"] > 0
    assert len(mesh["pick_ids"]) == mesh["vertex_count"]
    assert len(mesh["elem_meta"]) == len(bb["elements"])
    assert min(mesh["pick_ids"]) >= 1
    assert max(mesh["pick_ids"]) <= len(mesh["elem_meta"])
    assert all("short" in m and "name" in m and "id" in m for m in mesh["elem_meta"])


def test_studio_v14_http_api_end_to_end() -> None:
    server, sess = create_studio_server(host="127.0.0.1", port=0, initial_sample="kaelith_dragonborn_3d", size=64)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        # 1. Animated Swatch Sheet
        sheet_res = _http_json(f"{base}/api/schema/preview_anim_sheet?size=24&frames=4")
        assert sheet_res["ok"] and sheet_res["count"] >= 12

        # 2. 2D UV Picking on a guaranteed occupied UV island pixel
        mc = sess.mesh_ctx
        assert mc is not None
        ys, xs = np.where(mc.elem_id_map >= 0)
        assert len(ys) > 0
        u = (float(xs[0]) + 0.5) / mc.width
        v = (float(ys[0]) + 0.5) / mc.height
        pick_res = _http_json(f"{base}/api/mesh/pick_uv?u={u:.5f}&v={v:.5f}")
        assert pick_res["ok"] and pick_res["hit"] is True
        assert isinstance(pick_res["short"], str) and len(pick_res["short"]) > 0

        # 3. Comment Frame create -> update -> delete + Multi-Bypass + Reroute insertion
        st0 = _http_json(f"{base}/api/state")
        nodes0 = st0["recipe"]["nodes"]
        assert len(nodes0) >= 4
        pair = [nodes0[0]["id"], nodes0[1]["id"]]

        st_fr = _http_json(
            f"{base}/api/graph/mutate",
            "POST",
            {"action": "create_frame", "node_ids": pair, "title": "Test Group", "color": "#0F766E"},
        )
        fid = st_fr["created_frame_id"]
        assert any(f["id"] == fid and f["title"] == "Test Group" for f in st_fr["recipe"]["frames"])

        st_fr2 = _http_json(
            f"{base}/api/graph/mutate",
            "POST",
            {"action": "update_frame", "frame_id": fid, "title": "Renamed Group", "color": "#7E22CE"},
        )
        assert any(f["id"] == fid and f["title"] == "Renamed Group" and f["color"] == "#7E22CE" for f in st_fr2["recipe"]["frames"])

        st_fr3 = _http_json(
            f"{base}/api/graph/mutate",
            "POST",
            {"action": "delete_frame", "frame_id": fid},
        )
        assert not any(f["id"] == fid for f in st_fr3["recipe"].get("frames", []))

        # Find any connected wire and insert a reroute dot on it
        wired_target = next(((n["id"], next(iter(n["inputs"].keys()))) for n in nodes0 if n.get("inputs")), None)
        assert wired_target is not None
        st_rr = _http_json(
            f"{base}/api/graph/mutate",
            "POST",
            {"action": "insert_reroute", "dst_node": wired_target[0], "dst_port": wired_target[1], "pos": [180, 120]},
        )
        dot_id = st_rr["created_node_id"]
        assert any(n["id"] == dot_id and n["op"] == "reroute" for n in st_rr["recipe"]["nodes"])

        st_bp = _http_json(
            f"{base}/api/graph/mutate",
            "POST",
            {"action": "toggle_bypass_multi", "ids": pair},
        )
        by_id = {n["id"]: n for n in st_bp["recipe"]["nodes"]}
        assert by_id[pair[0]]["bypassed"] and by_id[pair[1]]["bypassed"]
        # Undo bypass
        _http_json(f"{base}/api/graph/mutate", "POST", {"action": "undo"})

        # 4. Dynamic-only thumbnail evaluation mode
        ev_dyn = _http_json(
            f"{base}/api/evaluate",
            "POST",
            {"phase": 0.25, "include_thumbs_mode": "dynamic", "include_qa": False},
        )
        assert ev_dyn["ok"] is True
        assert isinstance(ev_dyn["thumbs"], dict)

        # 5. .syncmat list, save (in-memory) & insert
        sm_list = _http_json(f"{base}/api/syncmat/list")
        assert sm_list["ok"] and len(sm_list["presets"]) >= 6
        sm_save = _http_json(
            f"{base}/api/syncmat/save",
            "POST",
            {"node_ids": pair, "name": "Smoke Preset", "persist": False},
        )
        assert sm_save["ok"] and sm_save["preset"]["name"] == "Smoke Preset"
        sm_ins = _http_json(
            f"{base}/api/syncmat/insert",
            "POST",
            {"preset_id": "amethyst_chronal_crystal", "pos": [300, 200], "wrap_frame": True},
        )
        assert len(sm_ins["created_node_ids"]) >= 2
        assert sm_ins["created_frame"] is not None

        # 6. MaterialX .mtlx export & full PBR v2.0 download bundle including .mtlx
        mtlx = _http_json(
            f"{base}/api/export/materialx",
            "POST",
            {"stem": "kaelith_test", "shader_model": "open_pbr_surface", "coat_weight": 0.3},
        )
        assert mtlx["ok"] and mtlx["filename"] == "kaelith_test.mtlx"
        ET.fromstring(mtlx["xml"])

        exp = _http_json(
            f"{base}/api/export",
            "POST",
            {"stem": "kaelith_pkg", "preset": "studio_hero"},
        )
        assert exp["ok"] and "kaelith_pkg.mtlx" in exp["downloads"]
    finally:
        server.shutdown()
        server.server_close()


def test_kaelith_upgraded_v14_texdsl_showcases_commercial_pbr_nodes() -> None:
    base = Path("/Users/tianrui6641/BlockBench-Skill/test_flights/texgraph_showcase/kaelith_chrono_dragonborn")
    tex_path = base / "kaelith_chrono_dragonborn.texdsl"
    bb_path = base / "kaelith_chrono_dragonborn.bbmodel"
    if not tex_path.is_file() or not bb_path.is_file():
        pytest.skip("Kaelith showcase files not present in workspace")

    bb_doc = json.loads(bb_path.read_text(encoding="utf-8"))
    mc = bake_bbmodel_maps(bb_doc)
    g = load_texdsl(tex_path, mesh_context=mc)
    ops = {n.op for n in g.nodes.values()}
    assert {"mat_velvet_silk", "mat_brass", "material_weathering", "mat_crystal_gem", "material_region_apply"} <= ops
    assert g.validate_graph().ok
    ev = g.evaluate(phase=0.0, build_pbr=True)
    qa = evaluate_texture_quality(
        ev.output,
        domain="entity",
        tileable=False,
        mesh_ctx=mc,
        pbr_channels=ev.pbr_channels,
        pbr_maps=ev.pbr_maps,
    )
    assert qa.passed and qa.score >= 95.0
