"""End-to-end verification tests for TexGraph v13 / SyncCraft Studio Commercial Upgrade.

Covers:
1. Full 210-node registry with 100% parameter coverage & universal adjustable `seed` on all Noise/Pattern/Material/Sim nodes
2. Zero preview fallbacks across all 210 nodes + real multi-frame animated previews (`/api/schema/preview_anim`)
3. Deterministic seed sensitivity across noises, patterns & PBR materials
4. 3D Model Spatial, Direction, UV-Box, Distance & Substance Smart Masks (`mesh_mask_ops.py`)
5. Full Brass / Plastic / Glass / Transparent Crystal PBR Material Sequencer & `brass_crystal_workflow_3d`
6. Live Studio HTTP endpoints (`mesh_parts`, `duplicate_nodes`, `randomize_seed`, `disconnect_all`, `preview_anim`)
"""
from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from synccraft.texgraph.core import TexGraph
from synccraft.texgraph.mesh_baker import bake_bbmodel_maps
from synccraft.texgraph.node_previews import (
    _eval_single_node_preview,
    _fallback_swatch,
    _get_refs_and_mesh,
    generate_node_anim_preview,
    generate_node_preview_catalog,
)
from synccraft.texgraph.registry import NODE_CATALOG, export_node_schema_json
from synccraft.texgraph.studio_server import (
    build_sample_monolith_bbmodel,
    build_studio_sample,
    create_studio_server,
)


def test_210_node_registry_and_universal_seed_coverage() -> None:
    schema = export_node_schema_json()
    assert schema["node_count"] >= 210
    assert schema["param_schema_coverage_pct"] == 100.0

    # Every Noise node must expose an adjustable `seed` parameter!
    noise_ops = [op for op, m in NODE_CATALOG.items() if m.category == "Noise"]
    assert len(noise_ops) >= 20
    for op in noise_ops:
        pnames = {p.name for p in NODE_CATALOG[op].params}
        assert "seed" in pnames, f"Noise op {op!r} missing adjustable seed param!"

    # Every Metal and Dielectric/Glass PBR material must expose `seed`!
    for cat in ("Materials — Metal (PBR)", "Materials — Dielectric & Glass (PBR)", "Geometric Patterns"):
        ops = [op for op, m in NODE_CATALOG.items() if m.category == cat]
        assert len(ops) >= 5
        for op in ops:
            if op == "polygon_star":
                continue
            pnames = {p.name for p in NODE_CATALOG[op].params}
            assert "seed" in pnames, f"Op {op!r} in {cat!r} missing seed param!"


def test_zero_preview_fallbacks_across_all_210_nodes_and_live_anim_frames() -> None:
    prevs = generate_node_preview_catalog(40)
    assert len(prevs) == len(NODE_CATALOG)
    refs, mc = _get_refs_and_mesh(40)
    fallbacks = [
        op
        for op, m in NODE_CATALOG.items()
        if _eval_single_node_preview(op, 40, refs, mc) == _fallback_swatch(m.category, 40)
    ]
    assert fallbacks == [], f"Unexpected fallback previews: {fallbacks}"

    # Verify live multi-frame animated previews actually move!
    for op in ("caustic_noise", "plasma_wave", "clockwork_gears", "mat_crystal_gem", "anim_shimmer", "perlin"):
        frames = generate_node_anim_preview(op, size=40, frames=6)
        assert len(frames) == 6
        assert len(set(frames)) >= 4, f"Animated preview for {op!r} was static!"


def test_seed_sensitivity_on_noises_patterns_and_pbr_materials() -> None:
    ops = [
        "value_noise",
        "perlin",
        "fbm",
        "voronoi",
        "gabor",
        "simplex_noise",
        "white_noise",
        "blue_noise",
        "bubble_noise",
        "crystal_noise",
        "scratch_noise",
        "strand_noise",
        "fold_marble_noise",
        "grunge_blotch",
        "maze_labyrinth",
        "mat_brass",
        "mat_bronze_copper",
        "mat_crystal_gem",
        "mat_dragon_scales_pbr",
    ]
    for op in ops:
        g1 = TexGraph(32, 32, seed=1)
        g1.set_output(g1.add(op, seed=11))
        o1 = g1.evaluate().output

        g2 = TexGraph(32, 32, seed=1)
        g2.set_output(g2.add(op, seed=8888))
        o2 = g2.evaluate().output
        diff = float(np.abs(o1.astype(np.float32) - o2.astype(np.float32)).mean())
        assert diff > 0.005, f"{op!r} did not respond to seed change (diff={diff:.5f})"


def test_3d_model_masks_and_brass_glass_crystal_pbr_sequence() -> None:
    bb = build_sample_monolith_bbmodel()
    mc = bake_bbmodel_maps(bb)
    g = TexGraph(mc.width, mc.height, tileable=False, seed=42, mesh_context=mc)

    m_part = g.add("mask_uv_part", part_pattern="cornice|plinth")
    m_pos = g.add("mask_world_position", axis="y_height", range_min=0.4, range_max=1.0)
    m_dir = g.add("mask_world_direction", direction="top_up", threshold=0.4, softness=0.2)
    m_box = g.add("mask_uv_box", mode="island_interior", border_px=1.0)
    m_dist = g.add("mask_3d_distance", shape="sphere", cx=0.5, cy=0.7, cz=0.5, radius=0.45)
    m_wear = g.add("mask_smart_edge_wear", wear_amount=0.6, seed=19)
    m_dirt = g.add("mask_smart_crevice_dirt", dirt_amount=0.6, seed=23)
    m_comb = g.add("mask_combine", inputs={"mask_a": m_part, "mask_b": m_dir}, mode="union")

    base_plastic = g.add("mat_plastic", color="#2B3648", finish="matte", seed=5)
    brass = g.add("mat_brass", base_color="#D9AE4E", patina_amount=0.25, seed=77)
    brass_tweak = g.add("material_tweak", inputs={"mat": brass}, saturation=1.1, roughness_offset=-0.04)
    glass = g.add("mat_glass", tint_color="#9CE6FF", opacity=0.42, style="clear", seed=9)
    crys = g.add("mat_crystal_gem", crystal_color="#A855F7", opacity=0.68, seed=12)

    s1 = g.add("material_region_apply", inputs={"base": base_plastic, "top": brass_tweak, "mask": m_comb})
    s2 = g.add("material_region_apply", inputs={"base": s1, "top": crys}, part_pattern="crest_fin")
    sh = g.add("anim_shimmer", inputs={"src": s2}, intensity=0.55)
    g.set_output_material(sh)

    ev0 = g.evaluate(phase=0.0, capture_intermediates=True, build_pbr=True)
    ev1 = g.evaluate(phase=0.35, build_pbr=True)

    # All 8 mask ops produce valid non-empty masks in [0, 1]
    for mid in (m_part, m_pos, m_dir, m_box, m_dist, m_wear, m_dirt, m_comb):
        arr = ev0.intermediates[mid]
        assert arr.min() >= -1e-5 and arr.max() <= 1.0001
        assert arr.mean() > 0.01

    # Glass and crystal preserve true alpha translucency (< 0.95)
    assert float(ev0.intermediates[glass]["albedo"][..., 3].mean()) < 0.75
    assert float(ev0.output[..., 3].min()) < 0.95

    # Live animation changes across timeline phases!
    anim_diff = float(np.abs(ev0.output.astype(np.float32) - ev1.output.astype(np.float32)).mean())
    assert anim_diff > 0.002

    # Flagship Brass & Crystal workflow sample evaluates cleanly and animates!
    g_bc, _ = build_studio_sample("brass_crystal_workflow_3d")
    b0 = g_bc.evaluate(phase=0.0, build_pbr=True)
    b1 = g_bc.evaluate(phase=0.35, build_pbr=True)
    assert float(np.abs(b0.output.astype(np.float32) - b1.output.astype(np.float32)).mean()) > 0.002


def test_studio_server_v13_endpoints_and_mutations() -> None:
    srv, _ = create_studio_server(host="127.0.0.1", port=0, initial_sample="brass_crystal_workflow_3d")
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"

    def _get(path: str) -> dict:
        with urllib.request.urlopen(base + path, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        prev = _get("/api/schema/previews?size=40")
        assert prev["count"] >= 210
        assert "caustic_noise" in prev["time_aware_ops"]

        anim = _get("/api/schema/preview_anim?op=caustic_noise&size=40&frames=6")
        assert anim["ok"] is True and anim["frames"] == 6 and len(set(anim["images"])) >= 4

        st = _get("/api/state")
        assert st["has_mesh"] is True
        assert st["mesh_parts"] is not None
        assert "cornice_canopy" in st["mesh_parts"]["elements"]
        n0 = len(st["recipe"]["nodes"])

        # Add Perlin noise, randomize its seed, duplicate it, disconnect it, and delete it!
        st_add = _post("/api/graph/mutate", {"action": "add_node", "op": "perlin", "pos": [120, 120]})
        nid = st_add["created_node_id"]
        assert len(st_add["recipe"]["nodes"]) == n0 + 1

        st_rnd = _post("/api/graph/mutate", {"action": "randomize_seed", "node_id": nid, "seed": 54321})
        n_obj = next(x for x in st_rnd["recipe"]["nodes"] if x["id"] == nid)
        assert n_obj["params"]["seed"] == 54321

        st_dup = _post("/api/graph/mutate", {"action": "duplicate_nodes", "ids": [nid]})
        dup_id = st_dup["created_node_id"]
        assert len(st_dup["recipe"]["nodes"]) == n0 + 2

        st_disc = _post("/api/graph/mutate", {"action": "disconnect_all", "node_id": dup_id})
        assert len(st_disc["recipe"]["nodes"]) == n0 + 2

        st_del = _post("/api/graph/mutate", {"action": "delete_nodes", "ids": [nid, dup_id]})
        assert len(st_del["recipe"]["nodes"]) == n0
    finally:
        srv.shutdown()
        srv.server_close()
