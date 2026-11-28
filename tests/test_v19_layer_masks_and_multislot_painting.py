"""Exhaustive tests for SyncCraft TexGraph v19.1 (`SyncCraft Studio v55.1` Track B):
Non-Destructive 3D Layer Mask Stack (Smart Fill Layers + Curvature/AO/World-Normal/Color-ID/Slot
Generators + Manual White/Black Paint Masks) & Multi-Texture `.bbmodel` Slot-Isolated Painting.
"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.request

import numpy as np

from synccraft.texgraph import (
    LAYER_MASK_GENERATORS,
    TexGraph,
    apply_mask_strokes_to_layer,
    bake_bbmodel_maps,
    compute_layer_smart_mask,
    make_fill_paint_layer,
)
from synccraft.texgraph.studio_server import build_sample_monolith_bbmodel, create_studio_server
from synccraft.texgraph.v16_pro_ops import (
    apply_pbr_brush_strokes_to_layer,
    make_empty_paint_layer,
    paint_layer_stack_op,
)


def _http_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _make_two_slot_paint_bbmodel() -> dict:
    """Two-texture `.bbmodel` with disjoint UV islands for Slot 0 (`body.png`) and Slot 1 (`crystal.png`)."""
    faces_0 = {f: {"uv": [2, 2, 26, 26], "texture": 0} for f in ("north", "east", "south", "west", "up", "down")}
    faces_1 = {f: {"uv": [36, 36, 60, 60], "texture": 1} for f in ("north", "east", "south", "west", "up", "down")}
    return {
        "meta": {"format_version": "4.10", "model_format": "bedrock", "box_uv": False},
        "name": "dual_slot_canvas",
        "resolution": {"width": 64, "height": 64},
        "textures": [
            {"id": "0", "uuid": "t0", "name": "body.png", "width": 64, "height": 64, "uv_width": 64, "uv_height": 64},
            {"id": "1", "uuid": "t1", "name": "crystal.png", "width": 64, "height": 64, "uv_width": 64, "uv_height": 64},
        ],
        "elements": [
            {"name": "body_plate", "uuid": "e0", "from": [-6, 0, -4], "to": [6, 12, 4], "origin": [0, 0, 0], "faces": faces_0},
            {"name": "crystal_spire", "uuid": "e1", "from": [-3, 12, -3], "to": [3, 20, 3], "origin": [0, 12, 0], "faces": faces_1},
        ],
        "outliner": [
            {"name": "body_bone", "uuid": "b0", "origin": [0, 0, 0], "children": ["e0"]},
            {"name": "crystal_bone", "uuid": "b1", "origin": [0, 12, 0], "children": ["e1"]},
        ],
    }


class TestV19LayerMasksAndMultiSlotPainting(unittest.TestCase):
    def test_v19_1_all_smart_mask_generators_3d_and_2d(self) -> None:
        self.assertEqual(len(LAYER_MASK_GENERATORS), 7)
        bb = build_sample_monolith_bbmodel()
        mc = bake_bbmodel_maps(bb)
        H, W = mc.height, mc.width

        for gen in LAYER_MASK_GENERATORS:
            gid = gen["id"]
            lyr = make_empty_paint_layer("l0", "Test", mask_mode=gid, mask_balance=0.55, mask_contrast=0.5)
            m3d = compute_layer_smart_mask(lyr, H, W, mesh_context=mc)
            self.assertEqual(m3d.shape, (H, W))
            self.assertTrue(np.isfinite(m3d).all())
            self.assertGreaterEqual(float(m3d.min()), 0.0)
            self.assertLessEqual(float(m3d.max()), 1.0)

            # 2D fallback (no mesh_context) must also run cleanly
            m2d = compute_layer_smart_mask(lyr, 32, 32, mesh_context=None)
            self.assertEqual(m2d.shape, (32, 32))

        # World Normal Up (+Y) vs Down (-Y) directional verification
        up_m = compute_layer_smart_mask(
            make_empty_paint_layer("u", "Up", mask_mode="world_normal_up", mask_balance=0.55, mask_contrast=0.7),
            H, W, mesh_context=mc,
        )
        up_faces = mc.island_mask & (mc.world_normal[..., 1] > 0.7)
        dn_faces = mc.island_mask & (mc.world_normal[..., 1] < -0.7)
        if np.any(up_faces) and np.any(dn_faces):
            self.assertGreater(float(up_m[up_faces].mean()), 0.8)
            self.assertLess(float(up_m[dn_faces].mean()), 0.1)

        # Invert flag strictly complements smart mask
        up_inv = compute_layer_smart_mask(
            make_empty_paint_layer("ui", "UpInv", mask_mode="world_normal_up", mask_balance=0.55, mask_contrast=0.7, mask_invert=True),
            H, W, mesh_context=mc,
        )
        self.assertLess(float(np.max(np.abs((up_m + up_inv) - 1.0))), 1e-5)

    def test_v19_1_smart_fill_layer_and_non_destructive_paint_mask(self) -> None:
        bb = build_sample_monolith_bbmodel()
        mc = bake_bbmodel_maps(bb)
        H, W = mc.height, mc.width

        base_alb = np.full((H, W, 4), [0.15, 0.16, 0.20, 1.0], dtype=np.float32)
        base_mat = {
            "albedo": base_alb,
            "height": np.full((H, W), 0.5, dtype=np.float32),
            "roughness": np.full((H, W), 0.80, dtype=np.float32),
            "metallic": np.zeros((H, W), dtype=np.float32),
            "ao": np.ones((H, W), dtype=np.float32),
        }

        fill_lyr = make_fill_paint_layer(
            "fill_gold",
            "Gold Edge Wear",
            preset="gold_trim",
            mask_mode="curvature_edges",
            mask_balance=0.62,
            mask_contrast=0.55,
            opacity=1.0,
        )
        res_fill = paint_layer_stack_op(H, W, base=base_mat, layers_json=[fill_lyr], mesh_context=mc)
        # Exposed edges gain gold metallic > 0.3 while flat interior keeps lower metallic
        self.assertGreater(float(res_fill["metallic"][mc.island_mask].max()), 0.35)
        self.assertGreater(float(res_fill["albedo"][mc.island_mask, 0].max()), 0.45)

        # Non-destructive Black Paint Mask stroke erases visibility at (0.5, 0.5) without touching fill_material!
        masked_reveal = apply_mask_strokes_to_layer(fill_lyr, W, H, [{"u": 0.5, "v": 0.5, "radius_px": 8}], erase=False)
        m_rev = compute_layer_smart_mask(masked_reveal, H, W, mesh_context=mc)
        self.assertGreater(float(m_rev[H // 2, W // 2]), 0.9)

        masked_hide = apply_mask_strokes_to_layer(masked_reveal, W, H, [{"u": 0.5, "v": 0.5, "radius_px": 8}], erase=True)
        m_hid = compute_layer_smart_mask(masked_hide, H, W, mesh_context=mc)
        self.assertLess(float(m_hid[H // 2, W // 2]), 0.05)

        # Verify TexGraph node passes `self.mesh_context` straight into `paint_layer_stack_op`
        g = TexGraph(width=W, height=H, seed=5)
        g.attach_mesh(mc)
        g.add("fbm", id="n0", scale=5.0)
        g.add("material_make", id="mat0", inputs={"albedo": "n0"}, roughness=0.75, metallic=0.0)
        g.add("paint_layer_stack", id="stack0", inputs={"base": "mat0"}, layers_json=json.dumps([fill_lyr]))
        g.set_output_material("stack0")
        ev = g.evaluate()
        self.assertGreater(float(ev.pbr_channels["metallic"][mc.island_mask].max()), 0.35)

    def test_v19_1_multi_slot_isolated_painting_and_color_id_regex(self) -> None:
        bb = _make_two_slot_paint_bbmodel()
        mc = bake_bbmodel_maps(bb, use_cache=False)
        H, W = mc.height, mc.width

        # Verify bake_bbmodel_maps populated disjoint slot_masks (0 & 1)
        self.assertIn(0, mc.slot_masks)
        self.assertIn(1, mc.slot_masks)
        s0 = mc.slot_masks[0] > 0.5
        s1 = mc.slot_masks[1] > 0.5
        self.assertTrue(np.any(s0))
        self.assertTrue(np.any(s1))
        self.assertFalse(np.any(s0 & s1))

        # 1. Color-ID Part regex isolates ONLY `crystal_spire` (Slot 1 region), zero hit on `body_plate`!
        cid_lyr = make_fill_paint_layer(
            "cid_crys",
            "Crystal Glow",
            preset="arcane_glow",
            mask_mode="color_id_part",
            mask_part_pattern="crystal",
        )
        m_cid = compute_layer_smart_mask(cid_lyr, H, W, mesh_context=mc)
        self.assertGreater(float(m_cid[s1].mean()), 0.9)
        self.assertLess(float(m_cid[s0].max()), 0.01)

        # 2. Multi-slot Fill Layer targeting Slot 1 modifies ONLY Slot 1 pixels, leaving Slot 0 untouched!
        base_mat = {
            "albedo": np.full((H, W, 4), [0.10, 0.10, 0.12, 1.0], dtype=np.float32),
            "height": np.full((H, W), 0.5, dtype=np.float32),
            "roughness": np.full((H, W), 0.80, dtype=np.float32),
            "metallic": np.zeros((H, W), dtype=np.float32),
            "ao": np.ones((H, W), dtype=np.float32),
        }
        slot1_fill = make_fill_paint_layer("s1_fill", "Slot 1 Gold", preset="gold_trim", mask_mode="none", target_slot=1, opacity=1.0)
        out_s1 = paint_layer_stack_op(H, W, base=base_mat, layers_json=[slot1_fill], mesh_context=mc)
        self.assertGreater(float(out_s1["metallic"][s1].mean()), 0.85)
        self.assertLess(float(out_s1["metallic"][s0].max()), 1e-4)

        # 3. Huge brush stroke spanning the entire canvas on a Slot 0 layer NEVER bleeds into Slot 1!
        slot0_paint = make_empty_paint_layer("s0_pnt", "Slot 0 Paint", target_slot=0)
        painted = apply_pbr_brush_strokes_to_layer(
            slot0_paint,
            W,
            H,
            [{"u": 0.5, "v": 0.5, "radius_px": 50.0, "opacity": 1.0}],
            brush={"preset": "gold_trim"},
            slot_mask=mc.slot_masks[0],
        )
        out_p0 = paint_layer_stack_op(H, W, base=base_mat, layers_json=[painted], mesh_context=mc)
        self.assertGreater(float(out_p0["metallic"][s0].mean()), 0.7)
        self.assertLess(float(out_p0["metallic"][s1].max()), 1e-4)

    def test_v19_1_studio_http_smart_mask_and_multislot_api(self) -> None:
        srv, _sess = create_studio_server(initial_sample="monolith_v10_3d", port=0)
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            health = _http_json(f"{base}/api/health")
            self.assertEqual(health["engine_version"], "19.1")

            schema = _http_json(f"{base}/api/schema")
            self.assertEqual(len(schema["layer_mask_generators"]), 7)

            # Load dual-slot model into live Studio session
            _http_json(f"{base}/api/mesh/load", {"bbmodel": _make_two_slot_paint_bbmodel()})

            # Add Smart-Masked Fill Layer via HTTP
            st1 = _http_json(
                f"{base}/api/paint_stack/stroke",
                {
                    "layer_action": "add_fill_layer",
                    "name": "Slot 1 Crystal Gold",
                    "preset": "gold_trim",
                    "mask_mode": "curvature_edges",
                    "target_slot": 1,
                },
            )
            self.assertEqual(len(st1["paint_layers"]), 2)
            fill_l = st1["paint_layers"][-1]
            self.assertEqual(fill_l["layer_type"], "fill")
            self.assertEqual(fill_l["mask_mode"], "curvature_edges")
            self.assertEqual(fill_l["target_slot"], 1)

            # Update smart mask to Color-ID Part + Invert
            st2 = _http_json(
                f"{base}/api/paint_stack/stroke",
                {
                    "layer_id": fill_l["id"],
                    "layer_action": "set_mask",
                    "mask_mode": "color_id_part",
                    "mask_part_pattern": "crystal",
                    "mask_invert": True,
                },
            )
            upd = [x for x in st2["paint_layers"] if x["id"] == fill_l["id"]][0]
            self.assertEqual(upd["mask_mode"], "color_id_part")
            self.assertTrue(upd["mask_invert"])
            self.assertEqual(upd["mask_part_pattern"], "crystal")

            # Paint non-destructive mask stroke (`edit_target: "mask"`) and clear mask
            st3 = _http_json(
                f"{base}/api/paint_stack/stroke",
                {
                    "layer_id": fill_l["id"],
                    "edit_target": "mask",
                    "strokes": [{"u": 0.75, "v": 0.75, "radius_px": 6}],
                    "brush": {"erase": False},
                },
            )
            m_painted = [x for x in st3["paint_layers"] if x["id"] == fill_l["id"]][0]
            self.assertTrue(m_painted["has_mask_paint"])

            st4 = _http_json(
                f"{base}/api/paint_stack/stroke",
                {"layer_id": fill_l["id"], "layer_action": "clear_mask"},
            )
            m_cleared = [x for x in st4["paint_layers"] if x["id"] == fill_l["id"]][0]
            self.assertFalse(m_cleared["has_mask_paint"])
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
