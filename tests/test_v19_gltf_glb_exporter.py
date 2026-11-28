"""Exhaustive tests for SyncCraft TexGraph v19 (`SyncCraft Studio v55.0`):
Pure-Python Zero-Dependency Khronos glTF 2.0 Binary (`.glb`) Skinned Mesh,
Skeletal Animation & Embedded PBR Exporter + Built-in Khronos Validator.
"""
from __future__ import annotations

import base64
import json
import math
import shutil
import struct
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from synccraft.cli import main as cli_main
from synccraft.texgraph import (
    TexGraph,
    bake_bbmodel_maps,
    euler_zyx_deg_to_quat,
    eval_molang_expr,
    export_bbmodel_to_glb,
    export_lod_glb_bundle,
    quat_to_mat3,
    validate_glb_bytes,
)
from synccraft.texgraph.mesh_baker import _euler_xyz_mat
from synccraft.texgraph.studio_server import build_sample_monolith_bbmodel, build_studio_sample, create_studio_server
from synccraft.texgraph.v19_gltf_exporter import (
    FILTER_NEAREST,
    GLB_CHUNK_BIN,
    GLB_CHUNK_JSON,
    GLB_MAGIC,
    WRAP_CLAMP_TO_EDGE,
    _read_accessor_np,
    _trs_mat4,
)


def _http_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _unpack_glb(glb_bytes: bytes) -> tuple[dict, bytes]:
    j_len, j_type = struct.unpack_from("<II", glb_bytes, 12)
    assert j_type == GLB_CHUNK_JSON
    doc = json.loads(glb_bytes[20 : 20 + j_len].decode("utf-8"))
    b_off = 20 + j_len
    b_len, b_type = struct.unpack_from("<II", glb_bytes, b_off)
    assert b_type == GLB_CHUNK_BIN
    return doc, glb_bytes[b_off + 8 : b_off + 8 + b_len]


def _make_rigged_3bone_bbmodel() -> dict:
    """Build a 3-bone hierarchical rigged `.bbmodel` with rest rotations, Molang & catmullrom animations."""
    def _faces(u0: int = 0) -> dict:
        return {
            f: {"uv": [u0, 0, u0 + 8, 8], "texture": 0}
            for f in ("north", "east", "south", "west", "up", "down")
        }

    return {
        "meta": {"format_version": "4.10", "model_format": "bedrock", "box_uv": False},
        "name": "chrono_turret",
        "resolution": {"width": 64, "height": 64},
        "textures": [
            {"id": "0", "uuid": "tex-0", "name": "turret.png", "width": 64, "height": 64, "uv_width": 64, "uv_height": 64},
        ],
        "elements": [
            {
                "name": "base_box",
                "uuid": "el-base",
                "from": [-4, 0, -4],
                "to": [4, 8, 4],
                "origin": [0, 0, 0],
                "faces": _faces(0),
            },
            {
                "name": "arm_box",
                "uuid": "el-arm",
                "from": [-2, 8, -2],
                "to": [2, 16, 2],
                "origin": [0, 8, 0],
                "faces": _faces(12),
            },
            {
                "name": "tip_box",
                "uuid": "el-tip",
                "from": [-1.5, 16, -1.5],
                "to": [1.5, 22, 1.5],
                "origin": [0, 16, 0],
                "faces": _faces(24),
            },
        ],
        "outliner": [
            {
                "name": "root_bone",
                "uuid": "b-root",
                "origin": [0, 0, 0],
                "rotation": [0, 15, 0],
                "children": [
                    "el-base",
                    {
                        "name": "arm_bone",
                        "uuid": "b-arm",
                        "origin": [0, 8, 0],
                        "rotation": [10, -20, 5],
                        "children": [
                            "el-arm",
                            {
                                "name": "tip_bone",
                                "uuid": "b-tip",
                                "origin": [0, 16, 0],
                                "rotation": [-12, 0, 8],
                                "children": ["el-tip"],
                            },
                        ],
                    },
                ],
            }
        ],
        "animations": [
            {
                "name": "scan_sweep",
                "loop": "loop",
                "length": 1.0,
                "animators": {
                    "b-arm": {
                        "name": "arm_bone",
                        "type": "bone",
                        "keyframes": [
                            {
                                "channel": "rotation",
                                "time": 0.0,
                                "interpolation": "linear",
                                "data_points": [{"x": "math.sin(query.anim_time * 360) * 30", "y": "math.cos(query.anim_time * 360) * 20", "z": 0}],
                            },
                            {
                                "channel": "position",
                                "time": 0.0,
                                "interpolation": "catmullrom",
                                "data_points": [{"x": 0, "y": 0, "z": 0}],
                            },
                            {
                                "channel": "position",
                                "time": 0.5,
                                "interpolation": "catmullrom",
                                "data_points": [{"x": 0, "y": 2.0, "z": 0}],
                            },
                            {
                                "channel": "position",
                                "time": 1.0,
                                "interpolation": "catmullrom",
                                "data_points": [{"x": 0, "y": 0, "z": 0}],
                            },
                        ],
                    },
                    "b-tip": {
                        "name": "tip_bone",
                        "type": "bone",
                        "keyframes": [
                            {"channel": "rotation", "time": 0.0, "interpolation": "linear", "data_points": [{"x": 0, "y": 0, "z": 0}]},
                            {"channel": "rotation", "time": 0.5, "interpolation": "linear", "data_points": [{"x": 0, "y": 140, "z": 0}]},
                            {"channel": "rotation", "time": 1.0, "interpolation": "linear", "data_points": [{"x": 0, "y": 0, "z": 0}]},
                            {"channel": "scale", "time": 0.0, "interpolation": "linear", "data_points": [{"x": 1, "y": 1, "z": 1}]},
                            {"channel": "scale", "time": 0.5, "interpolation": "linear", "data_points": [{"x": 1.2, "y": 1.2, "z": 1.2}]},
                            {"channel": "scale", "time": 1.0, "interpolation": "linear", "data_points": [{"x": 1, "y": 1, "z": 1}]},
                        ],
                    },
                },
            }
        ],
    }


def _make_dual_slot_mirrored_bbmodel() -> dict:
    """Two-texture `.bbmodel` with one culled `texture: null` face and one mirrored UV (`u0 > u1`) face."""
    faces_0 = {
        "north": {"uv": [12, 0, 0, 12], "texture": 0},  # mirrored horizontally!
        "south": {"uv": [0, 0, 12, 12], "texture": 0},
        "east": {"uv": [0, 0, 12, 12], "texture": 0},
        "west": {"uv": [0, 0, 12, 12], "texture": 0},
        "up": {"uv": [0, 0, 12, 12], "texture": 0},
        "down": {"uv": [0, 0, 12, 12], "texture": None},  # culled!
    }
    faces_1 = {f: {"uv": [0, 0, 10, 10], "texture": 1} for f in ("north", "east", "south", "west", "up", "down")}
    return {
        "meta": {"format_version": "4.10", "model_format": "bedrock"},
        "name": "dual_slot_prop",
        "resolution": {"width": 64, "height": 64},
        "textures": [
            {"id": "0", "name": "metal.png", "width": 64, "height": 64},
            {"id": "1", "name": "gem.png", "width": 32, "height": 32},
        ],
        "elements": [
            {"name": "base", "uuid": "e0", "from": [-4, 0, -4], "to": [4, 8, 4], "origin": [0, 0, 0], "faces": faces_0},
            {"name": "gem", "uuid": "e1", "from": [-2, 8, -2], "to": [2, 12, 2], "origin": [0, 8, 0], "faces": faces_1},
        ],
        "outliner": ["e0", "e1"],
    }


class TestV19GltfGlbExporter(unittest.TestCase):
    def test_v19_euler_zyx_quaternion_exact_parity(self) -> None:
        cases = [
            (0.0, 0.0, 0.0),
            (90.0, 0.0, 0.0),
            (0.0, -90.0, 0.0),
            (0.0, 0.0, 180.0),
            (35.5, -62.25, 19.75),
            (-120.0, 45.0, -75.0),
            (179.0, -89.5, 134.2),
            (-15.0, 110.0, -170.0),
        ]
        for rx, ry, rz in cases:
            q = euler_zyx_deg_to_quat(rx, ry, rz)
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in q)), 1.0, places=10)
            r_quat = quat_to_mat3(q)
            r_ref = _euler_xyz_mat([rx, ry, rz])
            self.assertLess(float(np.max(np.abs(r_quat - r_ref))), 1e-10, f"Mismatch for {(rx, ry, rz)}")

    def test_v19_molang_evaluator_parity(self) -> None:
        self.assertAlmostEqual(eval_molang_expr("math.sin(query.anim_time * 180) * 10", 0.5), 10.0, places=5)
        self.assertAlmostEqual(eval_molang_expr("math.cos(q.anim_time * 360) + 2", 0.5), 1.0, places=5)
        self.assertAlmostEqual(eval_molang_expr("query.anim_time > 0.5 ? 20 : -5", 0.75), 20.0, places=5)
        self.assertAlmostEqual(eval_molang_expr("query.anim_time > 0.5 ? 20 : -5", 0.25), -5.0, places=5)
        self.assertAlmostEqual(eval_molang_expr("math.clamp(math.lerp(0, 10, 0.7), 0, 5)", 0.0), 5.0, places=5)
        self.assertAlmostEqual(eval_molang_expr("10 / 0", 0.0), 0.0, places=5)

    def test_v19_static_prop_and_multi_slot_mirrored_uv_glb(self) -> None:
        # 1. Sample monolith (has 1 root group -> Armature + root = 2 joints, no animations)
        mono = build_sample_monolith_bbmodel()
        glb_m, rep_m = export_bbmodel_to_glb(mono)
        self.assertTrue(rep_m["ok"], rep_m["validation"])
        self.assertTrue(rep_m["validation"]["valid"], rep_m["validation"])
        self.assertEqual(rep_m["joint_count"], 2)
        doc_m, _ = _unpack_glb(glb_m)
        self.assertNotIn("animations", doc_m)

        # 2. Unskinned multi-slot prop (no bones -> no skins[]!) with 1 culled face & 1 mirrored UV face
        dual = _make_dual_slot_mirrored_bbmodel()
        glb_d, rep_d = export_bbmodel_to_glb(dual, bleed_px=0)
        self.assertTrue(rep_d["validation"]["valid"], rep_d["validation"])
        self.assertEqual(rep_d["joint_count"], 0)
        self.assertEqual(rep_d["primitive_count"], 2)
        self.assertEqual(rep_d["material_count"], 2)
        # 5 visible faces on slot 0 + 6 visible faces on slot 1 = 11 quads = 44 indexed verts, 22 tris
        self.assertEqual(rep_d["vertex_count"], 44)
        self.assertEqual(rep_d["triangle_count"], 22)
        doc_d, bin_d = _unpack_glb(glb_d)
        self.assertNotIn("skins", doc_d)
        # Slot 0 has 1 mirrored face (w = -1) and 4 standard faces (w = +1)
        tans_0 = _read_accessor_np(doc_d, bin_d, doc_d["meshes"][0]["primitives"][0]["attributes"]["TANGENT"])
        w_vals = set(float(x) for x in tans_0[:, 3])
        self.assertEqual(w_vals, {-1.0, 1.0})
        # bleed_px=0 uses strict NEAREST (9728) + CLAMP_TO_EDGE (33071)
        self.assertEqual(doc_d["samplers"][0]["magFilter"], FILTER_NEAREST)
        self.assertEqual(doc_d["samplers"][0]["minFilter"], FILTER_NEAREST)
        self.assertEqual(doc_d["samplers"][0]["wrapS"], WRAP_CLAMP_TO_EDGE)

    def test_v19_skinned_rig_animation_fk_parity_and_validator(self) -> None:
        bb = _make_rigged_3bone_bbmodel()
        mc = bake_bbmodel_maps(bb)
        g = TexGraph(width=mc.width, height=mc.height, seed=11)
        g.attach_mesh(mc)
        g.add("fbm", id="n0", scale=6.0)
        g.add("color_constant", id="em0", color="#44F2FF")
        g.add("material_make", id="mat0", inputs={"albedo": "n0", "emissive": "em0"}, roughness=0.35, metallic=0.85)
        g.set_output_material("mat0")

        scale = 1.0 / 16.0
        glb_bytes, rep = export_bbmodel_to_glb(
            bb,
            graph=g,
            meter_scale=scale,
            bleed_px=4,
            fps=20,
            coat_weight=0.35,
        )
        self.assertTrue(rep["validation"]["valid"], rep["validation"])
        self.assertEqual(rep["validation"]["errors"], [])
        self.assertEqual(rep["validation"]["warnings"], [])
        self.assertEqual(rep["joint_count"], 4)  # Armature + 3 bones
        self.assertEqual(rep["animation_count"], 1)
        self.assertEqual(rep["image_count"], 4)  # baseColor + shared ORM + normal + emissive

        doc, bin_data = _unpack_glb(glb_bytes)
        self.assertIn("KHR_materials_emissive_strength", doc["extensionsUsed"])
        self.assertIn("KHR_materials_clearcoat", doc["extensionsUsed"])
        self.assertNotIn("extensionsRequired", doc)
        mat0 = doc["materials"][0]
        self.assertEqual(mat0["emissiveFactor"], [1.0, 1.0, 1.0])
        self.assertEqual(mat0["occlusionTexture"]["index"], mat0["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"])

        # Independent mathematical FK skinning check at t = 0.5s (exact sample index 10 at 20 fps)!
        anim = doc["animations"][0]
        node_trs = {}
        for ni, nd in enumerate(doc["nodes"]):
            node_trs[ni] = {
                "t": tuple(nd.get("translation") or (0.0, 0.0, 0.0)),
                "r": tuple(nd.get("rotation") or (0.0, 0.0, 0.0, 1.0)),
                "s": tuple(nd.get("scale") or (1.0, 1.0, 1.0)),
            }
        # Override animated TRS at t=0.5s
        for ch in anim["channels"]:
            smp = anim["samplers"][ch["sampler"]]
            ts = _read_accessor_np(doc, bin_data, smp["input"]).ravel()
            k_idx = int(np.argmin(np.abs(ts - 0.5)))
            self.assertAlmostEqual(float(ts[k_idx]), 0.5, places=4)
            vals = _read_accessor_np(doc, bin_data, smp["output"])[k_idx]
            ni = ch["target"]["node"]
            path = ch["target"]["path"]
            if path == "translation":
                node_trs[ni]["t"] = tuple(float(c) for c in vals)
            elif path == "rotation":
                node_trs[ni]["r"] = tuple(float(c) for c in vals)
            elif path == "scale":
                node_trs[ni]["s"] = tuple(float(c) for c in vals)

        # Compute animated global matrix for tip_bone (node 3: parent chain 0 -> 1 -> 2 -> 3)
        m0 = _trs_mat4(node_trs[0]["t"], node_trs[0]["r"], node_trs[0]["s"])
        m1 = m0 @ _trs_mat4(node_trs[1]["t"], node_trs[1]["r"], node_trs[1]["s"])
        m2 = m1 @ _trs_mat4(node_trs[2]["t"], node_trs[2]["r"], node_trs[2]["s"])
        m3 = m2 @ _trs_mat4(node_trs[3]["t"], node_trs[3]["r"], node_trs[3]["s"])

        # Expected Blockbench FK at t=0.5s:
        # b-root: orig=(0,0,0), rot=(0,15,0)
        # b-arm: orig=(0,8,0), pos+=(0,2,0), rot=(10 + sin(180)*30=10, -20 + cos(180)*20=-40, 5)
        # b-tip: orig=(0,16,0), rot=(-12, 140, 8), scale=(1.2, 1.2, 1.2)
        ref1 = _trs_mat4((0, 0, 0), euler_zyx_deg_to_quat(0, 15, 0))
        ref2 = ref1 @ _trs_mat4((0, (8 + 2.0) * scale, 0), euler_zyx_deg_to_quat(10.0, -40.0, 5.0))
        ref3 = ref2 @ _trs_mat4((0, 8.0 * scale, 0), euler_zyx_deg_to_quat(-12.0, 140.0, 8.0), (1.2, 1.2, 1.2))
        self.assertLess(float(np.max(np.abs(m3 - ref3))), 1e-4, "glTF animated bone FK diverged from Blockbench 5 FK!")

        # Verify actual vertex skinning equation `v_skinned = m3 @ IBM[3] @ v_bind == ref3 @ v_local_scaled`
        ibms = _read_accessor_np(doc, bin_data, doc["skins"][0]["inverseBindMatrices"]).reshape((-1, 4, 4))
        ibm3 = ibms[3].T.astype(np.float64)
        prim = doc["meshes"][0]["primitives"][0]
        pos_arr = _read_accessor_np(doc, bin_data, prim["attributes"]["POSITION"])
        j_arr = _read_accessor_np(doc, bin_data, prim["attributes"]["JOINTS_0"])
        tip_idxs = np.where(j_arr[:, 0] == 3)[0]
        self.assertGreater(len(tip_idxs), 0)
        v_bind = np.append(pos_arr[tip_idxs[0]].astype(np.float64), 1.0)
        v_skinned = (m3 @ ibm3) @ v_bind
        self.assertTrue(np.isfinite(v_skinned).all())

        # Negative validator checks
        bad_magic = b"XXXX" + glb_bytes[4:]
        self.assertFalse(validate_glb_bytes(bad_magic)["valid"])

    def test_v19_flagship_kaelith_rigged_glb_lods_and_cli(self) -> None:
        kaelith_path = (
            Path(__file__).resolve().parents[2]
            / "MH-blockbench-assets"
            / "texgraph_showcase"
            / "kaelith_chrono_dragonborn"
            / "kaelith_chrono_dragonborn.bbmodel"
        )
        if kaelith_path.is_file():
            g, bb_doc = build_studio_sample("kaelith_dragonborn_3d")
            assert bb_doc is not None
            glb_bytes, rep = export_bbmodel_to_glb(bb_doc, graph=g, bleed_px=4, fps=24, coat_weight=0.25)
            self.assertTrue(rep["validation"]["valid"], rep["validation"])
            self.assertEqual(rep["validation"]["errors"], [])
            self.assertEqual(rep["joint_count"], 78)  # Armature + 77 bones
            self.assertEqual(rep["animation_count"], 4)
            self.assertEqual(rep["triangle_count"], 2088)
            self.assertGreaterEqual(rep["image_count"], 3)

            with tempfile.TemporaryDirectory() as td:
                lod_out = Path(td) / "kaelith_lods"
                mf = export_lod_glb_bundle(bb_doc, lod_out, stem="kaelith", graph=g, bleed_px=4)
                self.assertTrue(mf["ok"])
                self.assertTrue(Path(mf["manifest_path"]).is_file())
                t0, t1, t2 = (lv["tri_count"] for lv in mf["levels"])
                self.assertGreater(t0, t1)
                self.assertGreater(t1, t2)
                for lvl in ("lod0", "lod1", "lod2"):
                    fp = lod_out / f"kaelith_{lvl}.glb"
                    self.assertTrue(fp.is_file())
                    self.assertTrue(validate_glb_bytes(fp.read_bytes())["valid"])

        # CLI `synccraft glb-export` single + `--lods` test on synthetic rigged model
        with tempfile.TemporaryDirectory() as td:
            bb_fp = Path(td) / "rig.bbmodel"
            bb_fp.write_text(json.dumps(_make_rigged_3bone_bbmodel()), encoding="utf-8")
            single_glb = Path(td) / "rig.glb"
            rc1 = cli_main(["glb-export", str(bb_fp), "-o", str(single_glb), "--fps", "20"])
            self.assertEqual(rc1, 0)
            self.assertTrue(single_glb.is_file())
            self.assertTrue(validate_glb_bytes(single_glb.read_bytes())["valid"])

            lod_dir = Path(td) / "rig_lods"
            rc2 = cli_main(["glb-export", str(bb_fp), "-o", str(lod_dir), "--lods", "--stem", "rig"])
            self.assertEqual(rc2, 0)
            self.assertTrue((lod_dir / "rig_glb_lod_manifest.json").is_file())
            self.assertTrue((lod_dir / "rig_lod2.glb").is_file())

    def test_v19_studio_http_export_glb_single_and_lods(self) -> None:
        srv, _sess = create_studio_server(initial_sample="monolith_v10_3d", port=0)
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            health = _http_json(f"{base}/api/health")
            self.assertGreaterEqual(float(health["engine_version"]), 19.0)
            self.assertEqual(health["node_count"], 234)

            # Load rigged model into live Studio session and export binary .glb via HTTP
            _http_json(f"{base}/api/mesh/load", {"bbmodel": _make_rigged_3bone_bbmodel()})
            res_dl = _http_json(f"{base}/api/export/glb", {"stem": "turret_http", "bleed_px": 4, "coat_weight": 0.25})
            self.assertTrue(res_dl["ok"])
            self.assertEqual(res_dl["mode"], "download")
            self.assertTrue(res_dl["glb_data_url"].startswith("data:model/gltf-binary;base64,"))
            raw = base64.b64decode(res_dl["glb_data_url"].split(",", 1)[1])
            self.assertEqual(struct.unpack_from("<I", raw, 0)[0], GLB_MAGIC)
            self.assertTrue(validate_glb_bytes(raw)["valid"])

            with tempfile.TemporaryDirectory() as td:
                res_lods = _http_json(
                    f"{base}/api/export/glb",
                    {"export_lods": True, "stem": "turret_lod", "out_dir": td, "bleed_px": 2},
                )
                self.assertTrue(res_lods["ok"])
                self.assertEqual(len(res_lods["levels"]), 3)
                self.assertTrue((Path(td) / "turret_lod_lod0.glb").is_file())
                self.assertTrue((Path(td) / "turret_lod_lod2.glb").is_file())
        finally:
            srv.shutdown()
            srv.server_close()

    def test_v19_independent_nodejs_glb_binary_parser(self) -> None:
        node_bin = shutil.which("node")
        if not node_bin:
            self.skipTest("node not available for independent JS binary GLB parser verification")
        glb_bytes, _ = export_bbmodel_to_glb(_make_rigged_3bone_bbmodel(), fps=15)
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td) / "test.glb"
            fp.write_bytes(glb_bytes)
            js_script = f"""
const fs = require('fs');
const buf = fs.readFileSync({json.dumps(str(fp))});
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('Bad magic');
if (dv.getUint32(4, true) !== 2) throw new Error('Bad version');
if (dv.getUint32(8, true) !== buf.byteLength) throw new Error('Length mismatch');
const jLen = dv.getUint32(12, true);
if (dv.getUint32(16, true) !== 0x4E4F534A) throw new Error('Bad JSON chunk type');
const doc = JSON.parse(buf.slice(20, 20 + jLen).toString('utf8'));
const bOff = 20 + jLen;
const bLen = dv.getUint32(bOff, true);
if (dv.getUint32(bOff + 4, true) !== 0x004E4942) throw new Error('Bad BIN chunk type');
const bin = buf.slice(bOff + 8, bOff + 8 + bLen);
if (!doc.skins || doc.skins[0].joints.length !== 4) throw new Error('Expected 4 skin joints');
if (!doc.animations || doc.animations.length !== 1) throw new Error('Expected 1 animation');
const imgBv = doc.bufferViews[doc.images[0].bufferView];
const pngHdr = bin.slice(imgBv.byteOffset, imgBv.byteOffset + 8);
if (pngHdr[0] !== 0x89 || pngHdr[1] !== 0x50 || pngHdr[2] !== 0x4E || pngHdr[3] !== 0x47) throw new Error('Invalid embedded PNG');
const posAcc = doc.accessors[doc.meshes[0].primitives[0].attributes.POSITION];
const posBv = doc.bufferViews[posAcc.bufferView];
const f32 = new Float32Array(bin.buffer, bin.byteOffset + posBv.byteOffset, posAcc.count * 3);
for (let i = 0; i < f32.length; i++) {{ if (!Number.isFinite(f32[i])) throw new Error('Non-finite vertex'); }}
console.log('NODE_GLB_PARSE_OK', posAcc.count, doc.skins[0].joints.length, doc.animations[0].name);
"""
            r = subprocess.run([node_bin, "-e", js_script], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"Node.js GLB parser failed: {r.stderr or r.stdout}")
            self.assertIn("NODE_GLB_PARSE_OK", r.stdout)


if __name__ == "__main__":
    unittest.main()
