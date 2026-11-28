"""Exhaustive end-to-end tests for SyncCraft TexGraph v18 (`SyncCraft Studio v54.0`):
- Track A: Interactive 3D Decal Transform Gizmo previews & Custom PNG Alpha Stencil Projection
- Track B: Cuboid Auto-LOD Mesh Decimation (`LOD0 -> LOD1 -> LOD2`, CLI `lod-bake`, Studio preview/export)
           & Smart Mesh Baker Layers (`mesh_bake_layer` & `vertex_color_bake`, 234 total typed nodes)
- Track C: Multi-Texture `.bbmodel` Slot Switching & Live Blockbench 5 Skeletal Pose Sync over SSE Bridge
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from synccraft.cli import main as cli_main
from synccraft.texgraph import (
    SMART_BAKER_PRESETS,
    TexGraph,
    bake_bbmodel_maps,
    build_slot_uv_mask,
    decimate_bbmodel_lod,
    export_lod_bundle,
    export_node_schema_json,
    extract_bbmodel_texture_slots,
    generate_bbmodel_lod_chain,
    generate_node_preview_catalog,
    mesh_bake_layer_op,
    render_decal_stencil,
    resolve_face_texture_slot,
    vertex_color_bake_op,
)
from synccraft.texgraph.studio_server import (
    build_sample_monolith_bbmodel,
    create_studio_server,
    extract_bbmodel_webgl_mesh,
)
from synccraft.texgraph.v17_decal_atlas_ops import decal_stamp_op


def _make_custom_png_data_url(size: int = 32) -> str:
    """Create a crisp deterministic magenta-cyan alpha badge PNG data URL."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    cx = (size - 1) * 0.5
    dist = np.hypot(xx - cx, yy - cx)
    ring = (dist >= size * 0.18) & (dist <= size * 0.42)
    core = dist < size * 0.16
    img[ring] = [255, 48, 210, 255]
    img[core] = [40, 240, 255, 255]
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _make_two_slot_bbmodel() -> dict:
    """Build a spec-accurate 2-texture Blockbench 5 `.bbmodel` with one `texture: null` culled face."""
    def _faces(tex_idx: int, cull_down: bool = False) -> dict:
        out = {}
        for f in ("north", "east", "south", "west", "up", "down"):
            if cull_down and f == "down":
                out[f] = {"uv": [0, 0, 8, 8], "texture": None}
            else:
                u0 = 0 if tex_idx == 0 else 16
                out[f] = {"uv": [u0, 0, u0 + 12, 12], "texture": tex_idx}
        return out

    return {
        "meta": {"format_version": "4.10", "model_format": "bedrock", "box_uv": False},
        "name": "dual_slot_relic",
        "resolution": {"width": 64, "height": 64},
        "textures": [
            {"id": "0", "uuid": "tex-body-0", "name": "body.png", "width": 64, "height": 64, "uv_width": 64, "uv_height": 64},
            {"id": "1", "uuid": "tex-crystal-1", "name": "crystal.png", "width": 32, "height": 32, "uv_width": 32, "uv_height": 32},
        ],
        "elements": [
            {
                "name": "body_torso",
                "uuid": "el-body-1",
                "from": [-6, 0, -4],
                "to": [6, 14, 4],
                "origin": [0, 0, 0],
                "faces": _faces(0, cull_down=True),
            },
            {
                "name": "crystal_core",
                "uuid": "el-crys-2",
                "from": [-3, 14, -3],
                "to": [3, 22, 3],
                "origin": [0, 14, 0],
                "faces": _faces(1, cull_down=False),
            },
        ],
        "outliner": [
            {
                "name": "root",
                "uuid": "bone-root",
                "origin": [0, 0, 0],
                "children": [
                    "el-body-1",
                    {
                        "name": "crown_bone",
                        "uuid": "bone-crown",
                        "origin": [0, 14, 0],
                        "children": ["el-crys-2"],
                    },
                ],
            }
        ],
    }


def _http_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


class TestV18GizmoLodBakersMultiTexBridge(unittest.TestCase):
    def test_v18_schema_234_nodes_and_previews(self) -> None:
        schema = export_node_schema_json()
        self.assertEqual(schema["node_count"], 234)
        self.assertEqual(schema["schema_version"], "4.0")
        ops = {n["op"]: n for cat in schema["nodes_by_category"].values() for n in cat}
        self.assertIn("mesh_bake_layer", ops)
        self.assertIn("vertex_color_bake", ops)
        self.assertEqual(ops["mesh_bake_layer"]["output_type"], "MATERIAL")
        self.assertEqual(ops["vertex_color_bake"]["output_type"], "RGBA")

        previews = generate_node_preview_catalog(size=40)
        self.assertEqual(len(previews), 234)
        for op in ("mesh_bake_layer", "vertex_color_bake"):
            self.assertTrue(previews[op].startswith("data:image/png;base64,"), f"Invalid preview for {op}")

    def test_v18_mesh_bake_layer_presets_3d_and_2d(self) -> None:
        bb = build_sample_monolith_bbmodel()
        mc = bake_bbmodel_maps(bb)
        H, W = mc.height, mc.width

        base_alb = np.zeros((H, W, 4), dtype=np.float32)
        base_alb[..., :3] = [0.22, 0.24, 0.30]
        base_alb[..., 3] = 1.0
        base_mat = {
            "albedo": base_alb,
            "height": np.full((H, W), 0.5, dtype=np.float32),
            "roughness": np.full((H, W), 0.72, dtype=np.float32),
            "metallic": np.full((H, W), 0.05, dtype=np.float32),
            "ao": np.ones((H, W), dtype=np.float32),
        }

        self.assertEqual(len(SMART_BAKER_PRESETS), 6)
        for pid in SMART_BAKER_PRESETS:
            out3d = mesh_bake_layer_op(H, W, base=base_mat, mesh_context=mc, preset=pid)
            self.assertEqual(out3d["albedo"].shape, (H, W, 4))
            self.assertEqual(out3d["metallic"].shape, (H, W))
            self.assertEqual(out3d["roughness"].shape, (H, W))
            self.assertTrue(np.isfinite(out3d["albedo"]).all())

            # 2D fallback (no mesh_context) must also run cleanly
            out2d = mesh_bake_layer_op(32, 32, preset=pid)
            self.assertEqual(out2d["albedo"].shape, (32, 32, 4))

        # Verify alpine_snow_dust brightens upward (+Y) faces more than downward (-Y) faces
        snow = mesh_bake_layer_op(H, W, base=base_mat, mesh_context=mc, preset="alpine_snow_dust")
        up_mask = mc.island_mask & (mc.world_normal[..., 1] > 0.7)
        dn_mask = mc.island_mask & (mc.world_normal[..., 1] < -0.7)
        if np.any(up_mask) and np.any(dn_mask):
            up_luma = float(snow["albedo"][up_mask, :3].mean())
            dn_luma = float(snow["albedo"][dn_mask, :3].mean())
            self.assertGreater(up_luma, dn_luma + 0.15)

        # Verify edge_wear_metal increases metallic on exposed mesh islands
        wear = mesh_bake_layer_op(H, W, base=base_mat, mesh_context=mc, preset="edge_wear_metal")
        self.assertGreater(float(wear["metallic"][mc.island_mask].max()), 0.25)

        # Verify TexGraph node execution for both v18 nodes
        g = TexGraph(width=W, height=H, seed=7)
        g.attach_mesh(mc)
        g.add("fbm", id="n0", scale=6.0)
        g.add("material_make", id="mat0", inputs={"albedo": "n0"}, roughness=0.7, metallic=0.05)
        g.add("mesh_bake_layer", id="smart_wear", inputs={"base": "mat0"}, preset="edge_wear_metal")
        g.add("vertex_color_bake", id="cid", mode="part_id")
        g.set_output_material("smart_wear")
        ev = g.evaluate(capture_intermediates=True)
        self.assertEqual(ev.output.shape, (H, W, 4))
        self.assertEqual(ev.intermediates["cid"].shape, (H, W, 4))

    def test_v18_vertex_color_bake_all_modes(self) -> None:
        bb = build_sample_monolith_bbmodel()
        mc = bake_bbmodel_maps(bb)
        H, W = mc.height, mc.width

        modes = (
            "part_id",
            "bone_id",
            "zone_id",
            "curvature_diverging",
            "world_normal_rgb",
            "world_pos_rgb",
            "ao_cavity_rgb",
        )
        for m in modes:
            out = vertex_color_bake_op(H, W, mesh_context=mc, mode=m)
            self.assertEqual(out.shape, (H, W, 4))
            self.assertTrue(np.isfinite(out).all())
            out2d = vertex_color_bake_op(32, 32, mode=m)
            self.assertEqual(out2d.shape, (32, 32, 4))

        # part_id should produce multiple distinct island colors on multi-element monolith
        pid_rgba = vertex_color_bake_op(H, W, mesh_context=mc, mode="part_id")
        u8_cols = (pid_rgba[mc.island_mask, :3] * 255.0).round().astype(np.uint8)
        uniq = np.unique(u8_cols.reshape(-1, 3), axis=0)
        self.assertGreaterEqual(len(uniq), 3)

        # Soft-tint blend mode preserves base color influence
        base = np.full((H, W, 4), [0.1, 0.2, 0.3, 1.0], dtype=np.float32)
        blended = vertex_color_bake_op(H, W, base=base, mesh_context=mc, mode="part_id", tint_strength=0.35)
        self.assertFalse(np.allclose(blended[..., :3], pid_rgba[..., :3]))

    def test_v18_auto_lod_decimation_export_and_cli(self) -> None:
        # Build a rich 24-box hierarchical skeleton model (plus test Kaelith if present in workspace)
        elems = []
        leaf_children = []
        for i in range(24):
            uid = f"box_{i}"
            sz = 8.0 if i < 6 else (4.0 if i < 14 else 1.5)
            name = f"torso_core_{i}" if i < 4 else (f"limb_plate_{i}" if i < 14 else f"micro_rivet_{i}")
            elems.append(
                {
                    "name": name,
                    "uuid": uid,
                    "from": [i * 2.0, 0.0, 0.0],
                    "to": [i * 2.0 + sz, sz, sz],
                    "origin": [i * 2.0, 0.0, 0.0],
                    "faces": {
                        f: {"uv": [0, 0, 8, 8], "texture": 0}
                        for f in ("north", "east", "south", "west", "up", "down")
                    },
                }
            )
            leaf_children.append(uid)

        synth_bb = {
            "meta": {"format_version": "4.10", "model_format": "bedrock"},
            "name": "chrono_golem",
            "resolution": {"width": 64, "height": 64},
            "textures": [{"id": "0", "name": "golem.png", "width": 64, "height": 64}],
            "elements": elems,
            "outliner": [
                {
                    "name": "root",
                    "uuid": "b_root",
                    "origin": [0, 0, 0],
                    "children": [
                        {"name": "body", "uuid": "b_body", "origin": [0, 0, 0], "children": leaf_children[:10]},
                        {
                            "name": "arm",
                            "uuid": "b_arm",
                            "origin": [0, 0, 0],
                            "children": [
                                {
                                    "name": "wrist",
                                    "uuid": "b_wrist",
                                    "origin": [0, 0, 0],
                                    "children": [
                                        {"name": "finger_tip", "uuid": "b_finger", "origin": [0, 0, 0], "children": leaf_children[10:]}
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
            "animations": [{"name": "idle", "length": 1.0, "loop": "loop", "animators": {}}],
        }

        chain = generate_bbmodel_lod_chain(synth_bb, lod1_ratio=0.58, lod2_ratio=0.35)
        self.assertEqual(chain["lod0"]["box_count"], 24)
        self.assertGreater(chain["lod0"]["box_count"], chain["lod1"]["box_count"])
        self.assertGreater(chain["lod1"]["box_count"], chain["lod2"]["box_count"])
        self.assertGreater(chain["lod1"]["reduction_pct"], 25.0)
        self.assertGreater(chain["lod2"]["reduction_pct"], 50.0)
        self.assertTrue(chain["lod2"]["bbmodel"]["outliner"])
        self.assertEqual(len(chain["lod2"]["bbmodel"]["animations"]), 1)

        # Also verify flagship Kaelith Chrono-Dragonborn if available in workspace
        kaelith_path = (
            Path(__file__).resolve().parents[2]
            / "MH-blockbench-assets"
            / "models"
            / "kaelith_chrono_dragonborn"
            / "kaelith_chrono_dragonborn.bbmodel"
        )
        if kaelith_path.is_file():
            k_doc = json.loads(kaelith_path.read_text(encoding="utf-8"))
            k_chain = generate_bbmodel_lod_chain(k_doc, lod1_ratio=0.58, lod2_ratio=0.35)
            self.assertGreater(k_chain["lod0"]["box_count"], k_chain["lod1"]["box_count"])
            self.assertGreater(k_chain["lod1"]["box_count"], k_chain["lod2"]["box_count"])
            self.assertGreaterEqual(k_chain["lod1"]["reduction_pct"], 35.0)
            self.assertGreaterEqual(k_chain["lod2"]["reduction_pct"], 55.0)
            self.assertTrue(k_chain["lod2"]["bbmodel"].get("animations"))

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "lod_out"
            mf = export_lod_bundle(synth_bb, out_dir, stem="chrono_golem", bleed_px=4)
            self.assertTrue((out_dir / "chrono_golem_lod0.bbmodel").is_file())
            self.assertTrue((out_dir / "chrono_golem_lod1.bbmodel").is_file())
            self.assertTrue((out_dir / "chrono_golem_lod2.bbmodel").is_file())
            self.assertTrue((out_dir / "chrono_golem_lod_manifest.json").is_file())
            self.assertEqual(mf["levels"][0]["box_count"], 24)

            # Verify CLI `synccraft lod-bake`
            bb_file = Path(td) / "in.bbmodel"
            bb_file.write_text(json.dumps(synth_bb), encoding="utf-8")
            cli_out = Path(td) / "cli_lod"
            rc = cli_main(["lod-bake", str(bb_file), "-o", str(cli_out), "--stem", "cli_golem", "--bleed", "2"])
            self.assertEqual(rc, 0)
            self.assertTrue((cli_out / "cli_golem_lod_manifest.json").is_file())
            self.assertTrue((cli_out / "cli_golem_lod2.bbmodel").is_file())

    def test_v18_multi_texture_slot_extraction_and_isolation(self) -> None:
        bb = _make_two_slot_bbmodel()
        slots = extract_bbmodel_texture_slots(bb)
        self.assertEqual(len(slots), 2)
        self.assertEqual(slots[0]["name"], "body.png")
        self.assertEqual(slots[0]["face_count"], 5)  # 6 faces minus 1 culled (`texture: null`)
        self.assertEqual(slots[1]["name"], "crystal.png")
        self.assertEqual(slots[1]["face_count"], 6)

        self.assertIsNone(resolve_face_texture_slot(None, bb["textures"]))
        self.assertEqual(resolve_face_texture_slot("tex-crystal-1", bb["textures"]), 1)

        mesh = extract_bbmodel_webgl_mesh(bb)
        self.assertEqual(len(mesh["texture_slots"]), 2)
        # 11 visible faces * 6 vertices per face = 66 vertices
        self.assertEqual(len(mesh["tex_slot_ids"]), 66)
        self.assertEqual(set(mesh["tex_slot_ids"]), {0.0, 1.0})

        m0 = build_slot_uv_mask(bb, 0, 64, 64)
        m1 = build_slot_uv_mask(bb, 1, 64, 64)
        self.assertGreater(float(m0.sum()), 0.0)
        self.assertGreater(float(m1.sum()), 0.0)
        # Slot 0 UVs are [0..12] and Slot 1 UVs are [16..28] (scaled to 32..56 on 64px canvas) -> disjoint!
        self.assertEqual(float((m0 * m1).sum()), 0.0)

    def test_v18_custom_png_decal_stencil(self) -> None:
        custom_url = _make_custom_png_data_url(32)
        st_maps = render_decal_stencil("chrono_sigil", size=32, custom_data_url=custom_url)
        self.assertEqual(st_maps["albedo"].shape, (32, 32, 4))
        # Center pixel of our custom badge is bright cyan (R~0.16, G~0.94, B~1.0)
        ctr = st_maps["albedo"][16, 16]
        self.assertGreater(float(ctr[3]), 0.9)
        self.assertGreater(float(ctr[2]), 0.85)
        self.assertLess(float(ctr[0]), 0.30)

        # Verify decal_stamp_op composites custom PNG stencil cleanly in 2D and 3D
        stamps = [{"stencil": "chrono_sigil", "custom_data_url": custom_url, "scale_u": 12.0, "opacity": 1.0}]
        out2d = decal_stamp_op(64, 64, stamps_json=json.dumps(stamps))
        self.assertGreater(float(out2d["albedo"][32, 32, 2]), 0.7)

        bb = build_sample_monolith_bbmodel()
        mc = bake_bbmodel_maps(bb)
        out3d = decal_stamp_op(mc.height, mc.width, mesh_context=mc, stamps_json=json.dumps(stamps))
        self.assertEqual(out3d["albedo"].shape, (mc.height, mc.width, 4))
        self.assertGreater(float(out3d["albedo"][..., 2].max()), 0.7)

    def test_v18_studio_http_endpoints_and_bridge_pose_sync(self) -> None:
        srv, _sess = create_studio_server(initial_sample="monolith_v10_3d", port=0)
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            health = _http_json(f"{base}/api/health")
            self.assertEqual(health["engine_version"], "18.0")
            self.assertEqual(health["node_count"], 234)

            schema = _http_json(f"{base}/api/schema")
            self.assertEqual(len(schema["decal_stencil_previews"]), 8)
            self.assertEqual(len(schema["smart_baker_presets"]), 6)
            self.assertTrue(all(v.startswith("data:image/png;base64,") for v in schema["decal_stencil_previews"].values()))

            # Load dual-slot `.bbmodel` and verify texture slot switching
            st_loaded = _http_json(f"{base}/api/mesh/load", {"bbmodel": _make_two_slot_bbmodel()})
            self.assertEqual(len(st_loaded["texture_slots"]), 2)
            self.assertEqual(st_loaded["active_texture_slot"], -1)

            st_slot = _http_json(f"{base}/api/mesh/texture_slot", {"slot": 1})
            self.assertEqual(st_slot["active_texture_slot"], 1)

            # Switch back to monolith and verify Auto-LOD preview + export endpoints
            _http_json(f"{base}/api/mesh/load", {"sample": "temple_monolith"})
            lod1 = _http_json(f"{base}/api/lod/preview", {"level": "lod1"})
            self.assertTrue(lod1["ok"])
            self.assertEqual(lod1["active_lod"], "lod1")
            self.assertEqual(len(lod1["levels"]), 3)
            self.assertIn("positions", lod1["mesh"])

            with tempfile.TemporaryDirectory() as td:
                exp = _http_json(f"{base}/api/lod/export", {"stem": "monolith_test", "out_dir": td, "bleed_px": 2})
                self.assertTrue(Path(exp["manifest_path"]).is_file())
                self.assertTrue((Path(td) / "monolith_test_lod2.bbmodel").is_file())

            # Custom PNG decal stamp via HTTP API
            custom_url = _make_custom_png_data_url(32)
            st_dec = _http_json(
                f"{base}/api/decal_stamp/apply",
                {
                    "action": "add",
                    "stamp": {
                        "stencil": "chrono_sigil",
                        "custom_data_url": custom_url,
                        "custom_name": "my_badge.png",
                        "scale_u": 11.0,
                        "roll_deg": 25,
                    },
                },
            )
            self.assertEqual(st_dec["stamp_count"], 1)

            # Live BB5 Skeletal Pose Sync over SSE bridge
            pose_res = _http_json(
                f"{base}/api/bridge/pose",
                {"clip": "idle_breath", "time": 1.35, "playing": True, "speed": 1.5},
            )
            self.assertTrue(pose_res["ok"])
            self.assertEqual(pose_res["pose"]["clip"], "idle_breath")
            self.assertAlmostEqual(pose_res["pose"]["time"], 1.35, places=3)

            latest = _http_json(f"{base}/api/bridge/latest?since=-1")
            self.assertTrue(latest["changed"])
            self.assertEqual(latest["pose"]["clip"], "idle_breath")
            self.assertAlmostEqual(latest["pose"]["time"], 1.35, places=3)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_v18_blockbench5_plugin_v2_1_runtime_jsc(self) -> None:
        plugin_js = (
            Path(__file__).resolve().parents[2]
            / "MH-blockbench-assets"
            / "plugins"
            / "synccraft_pbr_rebuilder"
            / "synccraft_pbr_rebuilder.js"
        )
        if not plugin_js.is_file():
            self.skipTest("MH-blockbench-assets sibling repo not present")
        js_bin = shutil.which("node") or (
            "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
            if Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc").is_file()
            else None
        )
        if not js_bin:
            self.skipTest("Neither node nor jsc available")
        is_node = Path(js_bin).name == "node"
        harness = (
            f"var _p = {json.dumps(str(plugin_js))};\n"
            + ("var P = require(_p);\n" if is_node else "var module = { exports: {} }; load(_p); var P = module.exports;\n")
            + """
var actions = [];
function FakeAction(id, cfg) { this.id = id; this.cfg = cfg; this.delete = function(){ this.deleted = true; }; }
var tex0 = { name: "body.png", selected: false, src: "", material: { needsUpdate: false }, select: function(){ this.selected = true; }, updateSource: function(u){ this.src = u; }, updateChangesAfterEdit: function(){} };
var tex1 = { name: "crystal.png", selected: false, src: "", material: { needsUpdate: false }, select: function(){ this.selected = true; }, updateSource: function(u){ this.src = u; }, updateChangesAfterEdit: function(){} };
var updatedFacesTex = null;
var previewCalls = 0, defaultPoseCalls = 0, timelineT = -1, selectedAnim = null;
var animIdle = { name: "idle_breath", select: function(){ selectedAnim = this.name; } };
var grpHead = {
  name: "head",
  mesh: {
    fix_rotation: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0, set: function(x,y,z){ this.x=x; this.y=y; this.z=z; } },
    fix_position: { x: 0, y: 0, z: 0 },
    position: { x: 0, y: 0, z: 0, set: function(x,y,z){ this.x=x; this.y=y; this.z=z; } }
  }
};
var rt = {
  Action: FakeAction,
  MenuBar: { addAction: function(a){ actions.push(a); } },
  Undo: { initEdit: function(){}, finishEdit: function(){} },
  Blockbench: { showQuickMessage: function(){} },
  Texture: { selected: tex0, all: [tex0, tex1] },
  Cube: { all: [] },
  Canvas: { scene: { add: function(){}, remove: function(){} }, updateAllFaces: function(t){ updatedFacesTex = t; } },
  Animation: { all: [animIdle], selected: null },
  Timeline: { time: 0, setTime: function(t){ timelineT = t; this.time = t; } },
  Animator: { preview: function(){ previewCalls++; }, showDefaultPose: function(){ defaultPoseCalls++; } },
  Group: { all: [grpHead] }
};
var ctrl = P.createPbrRebuilderController(rt);
ctrl.onload();
if (actions.length < 6) throw new Error("Expected >=6 v2.1.0 BB5 actions, got " + actions.length);

// 1. Multi-texture slot 1 + live skeletal pose sync
ctrl.applyLiveStudioPayload({
  rev: 7,
  texture_slot: 1,
  channels: { baked: "data:image/png;base64,AAAA" },
  pose: { clip: "idle_breath", time: 1.25, playing: false, bone_overrides: { head: { rot: [15, 30, 0] } } }
});
if (!tex1.selected || tex1.src !== "data:image/png;base64,AAAA") throw new Error("Slot 1 targeting failed");
if (updatedFacesTex !== tex1) throw new Error("Canvas.updateAllFaces not called on target slot");
if (selectedAnim !== "idle_breath" || Math.abs(timelineT - 1.25) > 1e-4 || previewCalls < 1) throw new Error("Live pose sync failed");
if (Math.abs(grpHead.mesh.rotation.y - (30 * Math.PI / 180)) > 1e-4) throw new Error("Bone override failed");

// 2. Bind Pose reset
ctrl.applyLiveStudioPayload({
  rev: 8,
  pose: { clip: null, time: 0 }
});
if (defaultPoseCalls < 1) throw new Error("Bind pose reset did not call Animator.showDefaultPose");
ctrl.onunload();
(typeof print === "function" ? print : console.log)("BB5_V2_1_OK");
"""
        )
        r = subprocess.run([js_bin, "-e", harness], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, f"BB5 v2.1.0 plugin test failed: {r.stderr or r.stdout}")
        self.assertIn("BB5_V2_1_OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
