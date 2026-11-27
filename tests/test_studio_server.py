"""End-to-end backend, API, incremental caching, security & export tests for SyncCraft Studio (`v48.0`)."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from synccraft.texgraph.studio_server import create_studio_server


def _http_req(base_url: str, path: str, *, method: str = "GET", body: dict | None = None) -> tuple[int, bytes, dict]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read(), dict(err.headers)


class TestSyncCraftStudioServer(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.session = create_studio_server(host="127.0.0.1", port=0, initial_sample="monolith_v10_3d", size=64)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def test_static_assets_security_and_health_schema(self) -> None:
        # 1. HTML + JS + CSS static assets serve cleanly
        for path, marker in (
            ("/", b"SyncCraft Studio"),
            ("/assets/studio.css", b"--bg-canvas"),
            ("/assets/studio_gl3d.js", b"StudioGL3DViewport"),
            ("/assets/studio_graph.js", b"StudioGraphCanvas"),
            ("/assets/studio_app.js", b"apiPost"),
        ):
            code, raw, _ = _http_req(self.base_url, path)
            self.assertEqual(code, 200, f"Failed GET {path}")
            self.assertIn(marker, raw)

        # 2. Path traversal blocked strictly
        code, _, _ = _http_req(self.base_url, "/assets/../../cli.py")
        self.assertEqual(code, 404)

        # 3. Health & Schema v4.0
        code, raw, _ = _http_req(self.base_url, "/api/health")
        self.assertEqual(code, 200)
        h = json.loads(raw)
        self.assertTrue(h["ok"])
        self.assertEqual(h["schema_version"], "4.0")
        self.assertGreaterEqual(h["node_count"], 144)

        code, raw, _ = _http_req(self.base_url, "/api/schema")
        self.assertEqual(code, 200)
        s = json.loads(raw)
        self.assertEqual(s["schema"]["schema_version"], "4.0")
        self.assertEqual(s["schema"]["param_schema_coverage_pct"], 100.0)
        self.assertGreaterEqual(len(s["samples"]), 5)
        self.assertGreaterEqual(len(s["presets"]), 20)

    def test_live_3d_monolith_evaluation_cache_mutations_and_export(self) -> None:
        # 1. Initial 3D Monolith state + baked WebGL mesh + live .texdsl
        code, raw, _ = _http_req(self.base_url, "/api/state")
        self.assertEqual(code, 200)
        st = json.loads(raw)
        self.assertTrue(st["has_mesh"])
        self.assertGreater(st["mesh"]["vertex_count"], 100)
        self.assertGreater(st["mesh_stats"]["seam_pairs"], 1000)
        self.assertIn("@texture", st["texdsl"])

        # 2. Full 12-channel PBR evaluation + QA scorecard
        code, raw, _ = _http_req(self.base_url, "/api/evaluate", method="POST", body={"phase": 0.0, "include_thumbs": True, "include_qa": True})
        self.assertEqual(code, 200)
        ev1 = json.loads(raw)
        self.assertTrue(ev1["ok"])
        for ch in ("pbr_lit", "baked", "albedo", "normal", "shadow", "bent_normal", "thickness", "ao", "roughness", "metallic", "height", "orm", "maskmap", "bedrock_mer", "labpbr_n", "labpbr_s", "uv_seams"):
            self.assertIn(ch, ev1["channels"], f"Missing PBR channel: {ch}")
            self.assertTrue(ev1["channels"][ch].startswith("data:image/png;base64,"))
        self.assertGreaterEqual(len(ev1["thumbs"]), 5)
        self.assertGreaterEqual(ev1["qa"]["score"], 75.0)

        # 3. Tweaking downstream node (`monolith_final`) gets incremental cache hits on untouched upstream nodes!
        code, raw, _ = _http_req(
            self.base_url,
            "/api/graph/mutate",
            method="POST",
            body={"action": "set_param", "node_id": "monolith_final", "param": "vibrance", "value": 0.25},
        )
        self.assertEqual(code, 200)
        st_mod = json.loads(raw)
        self.assertEqual(st_mod["undo_depth"], 1)

        code, raw, _ = _http_req(self.base_url, "/api/evaluate", method="POST", body={"phase": 0.0})
        self.assertEqual(code, 200)
        ev2 = json.loads(raw)
        self.assertTrue(ev2["ok"])
        self.assertGreaterEqual(ev2["cache_hits"], 4, "Untouched upstream nodes must hit content-addressed cache!")

        # 4. Add node auto-wired from output + bypass + auto_layout + undo/redo
        code, raw, _ = _http_req(
            self.base_url,
            "/api/graph/mutate",
            method="POST",
            body={"action": "add_node", "op": "material_grade", "connect_from": {"node": "monolith_final", "port": "mat"}, "pos": [1240, 120]},
        )
        self.assertEqual(code, 200)
        st_add = json.loads(raw)
        new_id = st_add["created_node_id"]
        self.assertTrue(any(n["id"] == new_id for n in st_add["recipe"]["nodes"]))

        # Cycle rejection guard!
        code, raw, _ = _http_req(
            self.base_url,
            "/api/graph/mutate",
            method="POST",
            body={"action": "connect", "src_node": new_id, "src_port": "mat", "dst_node": "monolith_final", "dst_port": "mat"},
        )
        self.assertEqual(code, 400)
        self.assertIn("cycle", json.loads(raw)["message"].lower())

        # Bypass toggle
        code, raw, _ = _http_req(self.base_url, "/api/graph/mutate", method="POST", body={"action": "toggle_bypass", "node_id": new_id})
        self.assertEqual(code, 200)
        self.assertTrue(any(n["id"] == new_id and n.get("bypassed") for n in json.loads(raw)["recipe"]["nodes"]))

        # Auto-layout
        code, raw, _ = _http_req(self.base_url, "/api/graph/mutate", method="POST", body={"action": "auto_layout"})
        self.assertEqual(code, 200)

        # Undo & Redo
        code, raw, _ = _http_req(self.base_url, "/api/graph/mutate", method="POST", body={"action": "undo"})
        self.assertEqual(code, 200)
        self.assertGreaterEqual(json.loads(raw)["redo_depth"], 1)
        code, raw, _ = _http_req(self.base_url, "/api/graph/mutate", method="POST", body={"action": "redo"})
        self.assertEqual(code, 200)

        # 5. Live Bidirectional .texdsl compile + syntax error reporting
        code, raw, _ = _http_req(
            self.base_url,
            "/api/graph/mutate",
            method="POST",
            body={"action": "apply_texdsl", "code": "@texture broken syntax {{{\n"},
        )
        self.assertEqual(code, 400)

        # 6. Full Multi-Engine PBR v2.0 Export (Disk + Download mode)
        with tempfile.TemporaryDirectory() as tmp:
            code, raw, _ = _http_req(
                self.base_url,
                "/api/export",
                method="POST",
                body={"stem": "studio_monolith", "preset": "studio_hero", "out_dir": tmp},
            )
            self.assertEqual(code, 200)
            exp = json.loads(raw)
            self.assertTrue(exp["ok"])
            self.assertEqual(exp["manifest"]["schema"], "synccraft.pbr.bundle/2.0")
            for k in ("baked_albedo", "baked_bbmodel", "orm", "unity_maskmap", "shadow", "bent_normal", "thickness", "labpbr_n", "labpbr_s", "bedrock_mer", "texdsl", "texgraph_json"):
                self.assertIn(k, exp["files"])
                self.assertTrue(Path(exp["files"][k]).is_file())

        # 7. Unconnected freshly spawned node must NOT crash live Studio evaluation!
        code, raw, _ = _http_req(
            self.base_url,
            "/api/graph/mutate",
            method="POST",
            body={"action": "add_node", "op": "pixel_dither", "pos": [900, 300]},
        )
        self.assertEqual(code, 200)
        code, raw, _ = _http_req(self.base_url, "/api/evaluate", method="POST", body={"phase": 0.0})
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(raw)["ok"])

        # 8. All flagship Studio projects load & evaluate cleanly (zero TypeErrors / stale buffers!)
        for sid, sz in (("connected_block_wall_v10", 64), ("arcane_ember_vfx", 32), ("temple_carved_stone", 32)):
            code, raw, _ = _http_req(self.base_url, "/api/load", method="POST", body={"kind": "sample", "id": sid, "size": sz})
            self.assertEqual(code, 200, f"Failed load {sid}")
            code, raw, _ = _http_req(self.base_url, "/api/evaluate", method="POST", body={"phase": 0.25})
            self.assertEqual(code, 200, f"Failed evaluate {sid}: {raw!r}")
            ev_s = json.loads(raw)
            self.assertTrue(ev_s["ok"])
            self.assertIn("pbr_lit", ev_s["channels"])
            self.assertIn("orm", ev_s["channels"])


if __name__ == "__main__":
    unittest.main()
