"""Comprehensive tests for Milestone v12 (`SyncCraft v49.0`):
- Pure-algorithmic `extract_style_dna` (presets + real PIL image + base64 data URL)
- AI Co-Pilot `diagnose_asset` & `apply_surgical_patch` with strict `artist_overrides` lock protection
- Unified `.syncproj` manifest round-trip & Batch Chronal Variant Factory (`generate_variant_pack`)
- Kaelith Chrono-Amethyst Artificer Dragonborn 3D Studio flagship sample
- All 5 new Studio REST API endpoints (`/api/ai/diagnose`, `/api/ai/patch`, `/api/style/extract`, `/api/variants/apply`, `/api/syncproj/export`)
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from synccraft.texgraph import (
    BUILT_IN_STYLE_PRESETS,
    DEFAULT_CHRONO_VARIANTS,
    SyncProject,
    TexGraph,
    apply_surgical_patch,
    apply_variant_to_graph,
    diagnose_asset,
    extract_style_dna,
    generate_variant_pack,
)
from synccraft.texgraph.studio_server import STUDIO_SAMPLES, build_studio_sample, create_studio_server


def _http_req(base_url: str, path: str, *, method: str = "GET", body: dict | None = None) -> tuple[int, bytes]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


class TestV12AICopilotAndSyncProj(unittest.TestCase):
    def test_extract_style_dna_presets_and_real_image(self) -> None:
        # 1. Built-in D&D / Chronal presets
        for pid in ("chronos_amethyst", "solar_aeon", "void_eclipse", "frost_singularity"):
            self.assertIn(pid, BUILT_IN_STYLE_PRESETS)
            dna = extract_style_dna(f"preset:{pid}", num_stops=6, name=pid)
            self.assertEqual(len(dna["stops"]), 6)
            self.assertTrue(all(s.startswith("#") and len(s) == 7 for s in dna["stops"]))
            self.assertIn("stats", dna)
            self.assertIn("texdsl_palette_block", dna)

        # 2. Pure-algorithmic extraction from real synthetic reference image (PIL + base64 data URL)
        arr = np.zeros((48, 48, 3), dtype=np.uint8)
        arr[:16, :, :] = (24, 18, 52)     # Deep slate-indigo shadow
        arr[16:32, :, :] = (120, 68, 190) # Royal amethyst midtone
        arr[32:, :, :] = (240, 200, 110)  # Celestial gold highlight
        im = Image.fromarray(arr, mode="RGB")

        dna_pil = extract_style_dna(im, num_stops=5, name="synth_chrono")
        self.assertEqual(len(dna_pil["stops"]), 5)
        self.assertEqual(dna_pil["name"], "synth_chrono")

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        dna_url = extract_style_dna(data_url, num_stops=5, name="url_chrono")
        self.assertEqual(dna_url["stops"], dna_pil["stops"])

    def test_diagnose_and_surgical_patch_preserves_artist_overrides(self) -> None:
        g, _ = build_studio_sample("monolith_v10_3d", size=64)
        rep = diagnose_asset(g, g.mesh_context, phase=0.0)
        self.assertIn("qa", rep)
        self.assertIn("findings", rep)
        self.assertGreaterEqual(len(rep["findings"]), 1)
        self.assertIn("patch_ops", rep["findings"][0])

        # Simulate a human artist locking a specific parameter on `monolith_final`
        g.nodes["monolith_final"].params["vibrance"] = 0.42
        locks = {"monolith_final.vibrance": 0.42}

        # AI attempts to overwrite BOTH the locked param AND an unlocked param (`contrast`)
        res = apply_surgical_patch(
            g,
            [
                {"op": "set_param", "node_id": "monolith_final", "param": "vibrance", "value": 0.05},
                {"op": "set_param", "node_id": "monolith_final", "param": "contrast", "value": 1.12},
            ],
            artist_overrides=locks,
            preserve_artist_overrides=True,
        )
        self.assertTrue(res["ok"])
        self.assertEqual(res["applied_count"], 1)
        preserved_keys = {f"{o['node_id']}.{o['param']}" for o in res["preserved_overrides"]}
        self.assertIn("monolith_final.vibrance", preserved_keys)
        self.assertAlmostEqual(float(g.nodes["monolith_final"].params["vibrance"]), 0.42, places=4)
        self.assertAlmostEqual(float(g.nodes["monolith_final"].params["contrast"]), 1.12, places=4)

    def test_syncproj_roundtrip_and_batch_variant_pack(self) -> None:
        g, bb = build_studio_sample("kaelith_dragonborn_3d", size=64)
        self.assertIsNotNone(bb)

        # Lock an artist parameter and verify it survives variant morphing!
        target_node = "graded" if "graded" in g.nodes else next(iter(g.nodes))
        locks = {f"{target_node}.contrast": 1.11}
        g_solar = apply_variant_to_graph(g, DEFAULT_CHRONO_VARIANTS["solar_aeon"], artist_overrides=locks)
        self.assertAlmostEqual(float(g_solar.nodes[target_node].params["contrast"]), 1.11, places=4)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            proj = SyncProject(
                name="kaelith_test",
                blockdsl_path="kaelith_chrono_dragonborn.blockdsl",
                bbmodel_path="kaelith_chrono_dragonborn.bbmodel",
                texdsl_path="kaelith_chrono_dragonborn.texdsl",
                artist_overrides=locks,
            )
            saved = proj.save(tmp_p / "kaelith.syncproj")
            loaded = SyncProject.load(saved)
            self.assertEqual(loaded.name, "kaelith_test")
            self.assertEqual(loaded.artist_overrides, locks)
            self.assertIn("solar_aeon", loaded.variants)

            # Generate 2-variant batch pack with QA gate!
            subset = {k: DEFAULT_CHRONO_VARIANTS[k] for k in ("chronos_prime", "void_eclipse")}
            summary = generate_variant_pack(g, subset, tmp_p / "variants", bbmodel_doc=bb, artist_overrides=locks, qa_gate=75.0)
            self.assertEqual(summary["variant_count"], 2)
            self.assertTrue(summary["all_passed"])
            for v in summary["variants"]:
                self.assertTrue(v["qa_passed"])
                self.assertTrue(Path(v["baked_png"]).is_file())
                self.assertTrue(Path(v["baked_bbmodel"]).is_file())

    def test_studio_v12_endpoints_with_kaelith_dragonborn(self) -> None:
        # Confirm Kaelith Chrono-Dragonborn replaced Orc as flagship sample #1!
        self.assertEqual(STUDIO_SAMPLES[0]["id"], "kaelith_dragonborn_3d")

        server, sess = create_studio_server(host="127.0.0.1", port=0, initial_sample="kaelith_dragonborn_3d", size=64)
        port = server.server_address[1]
        base = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            # 1. Schema exposes v12 style_presets & variants
            code, raw = _http_req(base, "/api/schema")
            self.assertEqual(code, 200)
            sch = json.loads(raw)
            self.assertGreaterEqual(len(sch["style_presets"]), 4)
            self.assertEqual(len(sch["variants"]), 4)

            # 2. Initial state loads Kaelith 3D Dragonborn mesh cleanly
            code, raw = _http_req(base, "/api/state")
            self.assertEqual(code, 200)
            st = json.loads(raw)
            self.assertTrue(st["has_mesh"])
            self.assertGreaterEqual(st["mesh_stats"]["elements"], 100)

            # 3. Artist sets a param -> recorded in artist_overrides lock map
            node_id = "graded" if "graded" in sess.graph.nodes else next(iter(sess.graph.nodes))
            code, raw = _http_req(
                base,
                "/api/graph/mutate",
                method="POST",
                body={"action": "set_param", "node_id": node_id, "param": "vibrance", "value": 0.31},
            )
            self.assertEqual(code, 200)
            st_locked = json.loads(raw)
            self.assertAlmostEqual(float(st_locked["artist_overrides"][f"{node_id}.vibrance"]), 0.31, places=4)

            # 4. /api/ai/diagnose & /api/ai/patch preserve artist lock!
            code, raw = _http_req(base, "/api/ai/diagnose", method="POST", body={"phase": 0.0})
            self.assertEqual(code, 200)
            diag = json.loads(raw)
            self.assertGreaterEqual(diag["qa"]["score"], 90.0)

            code, raw = _http_req(
                base,
                "/api/ai/patch",
                method="POST",
                body={
                    "patch_ops": [{"op": "set_param", "node_id": node_id, "param": "vibrance", "value": 0.01}],
                    "preserve_artist_overrides": True,
                },
            )
            self.assertEqual(code, 200)
            pres = json.loads(raw)
            preserved_keys = {f"{o['node_id']}.{o['param']}" for o in pres["patch_result"]["preserved_overrides"]}
            self.assertIn(f"{node_id}.vibrance", preserved_keys)
            self.assertAlmostEqual(float(sess.graph.nodes[node_id].params["vibrance"]), 0.31, places=4)

            # 5. /api/style/extract
            code, raw = _http_req(base, "/api/style/extract", method="POST", body={"source": "preset:solar_aeon", "apply": False})
            self.assertEqual(code, 200)
            dna_resp = json.loads(raw)
            self.assertTrue(dna_resp["ok"])
            self.assertEqual(len(dna_resp["dna"]["stops"]), 6)

            # 6. /api/variants/apply live 3D morph (artist lock still preserved!)
            code, raw = _http_req(base, "/api/variants/apply", method="POST", body={"variant_id": "void_eclipse"})
            self.assertEqual(code, 200)
            vresp = json.loads(raw)
            self.assertEqual(vresp["variant"], "void_eclipse")
            self.assertEqual(vresp["state"]["active_variant"], "void_eclipse")
            self.assertAlmostEqual(float(sess.graph.nodes[node_id].params["vibrance"]), 0.31, places=4)

            # 7. /api/syncproj/export
            code, raw = _http_req(base, "/api/syncproj/export", method="POST", body={"stem": "kaelith_dragonborn"})
            self.assertEqual(code, 200)
            sresp = json.loads(raw)
            self.assertEqual(sresp["manifest"]["schema"], "synccraft.project/1.0")
            self.assertIn(f"{node_id}.vibrance", sresp["manifest"]["artist_overrides"])

            # 8. Clear artist locks
            code, raw = _http_req(base, "/api/graph/mutate", method="POST", body={"action": "clear_artist_overrides"})
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(raw)["artist_overrides"], {})
        finally:
            server.shutdown()
            server.server_close()
            th.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
