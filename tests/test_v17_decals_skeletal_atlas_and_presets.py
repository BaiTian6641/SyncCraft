"""Milestone v17 Verification Suite:
- 232 Typed Nodes (Schema v4.0) + Zero Fallback Swatches
- Mip-Safe Voronoi/JFA Dilation (Straight-Alpha RGB + Unit Tangent Normal + Mip Preview)
- 3D Camera-Projected Decal / Stencil Stamps (Exact Kay-Kajiya OBB Occlusion & Backface Culling)
- Blockbench 5.0 Skeletal Hierarchy & Live Animation Extraction (Kaelith 4 Clips + Rest-Pose FK Parity)
- Multi-Material Shared Atlas Baker (MaxRects-BSSF PoT Packing, Mip Bleed, Remapped .bbmodel UVs & CLI)
- Canonical One-Click 6-Engine Export Presets (UE5 DX Normal, Unity HDRP B=255 MaskMap, LabPBR 1.3 DX _n,
  Bedrock 1.21.30 _mers, MaterialX 1.39 OpenPBR / 1.38 StandardSurface, Blockbench Baked)
- Live Studio v17 HTTP Endpoints (/api/health, /api/schema, /api/decal_stamp/apply, /api/export/preset, /api/atlas/bake)
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from synccraft.texgraph import (
    DECAL_STENCILS,
    ENGINE_EXPORT_PRESETS,
    NODE_CATALOG,
    TexGraph,
    bake_bbmodel_maps,
    bake_multi_asset_atlas,
    build_sample_monolith_bbmodel,
    create_studio_server,
    decal_stamp_op,
    export_engine_preset,
    export_node_schema_json,
    extract_bbmodel_webgl_mesh,
    generate_node_preview_catalog,
    mip_safe_dilate_normal,
    mip_safe_dilate_op,
    mip_safe_dilate_rgba,
)
from synccraft.texgraph.mesh_baker import _euler_xyz_mat
from synccraft.texgraph.studio_server import _find_kaelith_dragonborn_paths


def _make_two_occluding_boxes_bbmodel() -> dict:
    """Two boxes aligned on Z: Front box at z=[4, 8], Rear box at z=[-8, -4], with disjoint UVs on 64x64."""
    return {
        "name": "decal_occlusion_fixture",
        "resolution": {"width": 64, "height": 64},
        "elements": [
            {
                "uuid": "front-box",
                "name": "front_plate",
                "from": [-6, 0, 4],
                "to": [6, 12, 8],
                "origin": [0, 6, 6],
                "rotation": [0, 0, 0],
                "faces": {
                    "south": {"uv": [2, 2, 26, 26], "texture": 0},   # +Z facing camera
                    "north": {"uv": [34, 2, 58, 26], "texture": 0},  # -Z facing away (backface)
                },
            },
            {
                "uuid": "rear-box",
                "name": "rear_plate",
                "from": [-6, 0, -8],
                "to": [6, 12, -4],
                "origin": [0, 6, -6],
                "rotation": [0, 0, 0],
                "faces": {
                    "south": {"uv": [2, 34, 26, 58], "texture": 0},  # +Z facing camera BUT occluded by front_plate!
                    "north": {"uv": [34, 34, 58, 58], "texture": 0},
                },
            },
        ],
        "outliner": ["front-box", "rear-box"],
    }


def _http_json(url: str, body: dict | None = None) -> dict:
    if body is None:
        with urllib.request.urlopen(url, timeout=15) as resp:
            assert resp.status == 200
            return json.loads(resp.read().decode("utf-8"))
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode("utf-8"))


def test_v17_node_registry_232_schema_and_previews() -> None:
    assert len(NODE_CATALOG) == 232
    assert "decal_stamp" in NODE_CATALOG
    assert "mip_safe_dilate" in NODE_CATALOG

    schema = export_node_schema_json()
    assert schema["node_count"] == 232
    assert schema["schema_version"] == "4.0"

    previews = generate_node_preview_catalog(size=32)
    assert len(previews) == 232
    assert previews["decal_stamp"].startswith("data:image/png;base64,")
    assert previews["mip_safe_dilate"].startswith("data:image/png;base64,")


def test_v17_mip_safe_voronoi_dilation_rgb_normal_and_mip_preview() -> None:
    H, W = 64, 64
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[20:44, 20:44] = (235, 195, 70, 255)
    seed_m = rgba[..., 3] > 0

    dil = mip_safe_dilate_rgba(rgba, valid_mask=seed_m, bleed_px=8, preserve_alpha=True)
    # Interior preserved identically
    assert np.array_equal(dil[20:44, 20:44], rgba[20:44, 20:44])
    # Straight-alpha preserved: exterior pixels remain transparent (alpha == 0)
    assert int(dil[16, 30, 3]) == 0
    # RGB cleanly dilated outward to eliminate dark mipmap seams
    assert int(dil[16, 30, 0]) == 235 and int(dil[16, 30, 1]) == 195 and int(dil[16, 30, 2]) == 70
    # Outside bleed_px stays black
    assert int(dil[4, 4, 0]) == 0

    # Tangent Normal Voronoi Dilation keeps unit-length normals + flat (128,128,255) void
    nrm = np.zeros((H, W, 4), dtype=np.uint8)
    nrm[..., :3] = (128, 128, 255)
    nrm[..., 3] = 255
    nrm[20:44, 20:44, :3] = (180, 140, 240)
    dil_n = mip_safe_dilate_normal(nrm, seed_m, bleed_px=6)
    # Dilated pixel near border inherits tilted normal, normalized to unit length
    vec = (dil_n[17, 30, :3].astype(np.float32) / 127.5) - 1.0
    norm_len = float(np.linalg.norm(vec))
    assert 0.94 <= norm_len <= 1.06
    assert dil_n[17, 30, 0] > 165
    # Far void stays neutral flat (128, 128, 255)
    assert tuple(int(c) for c in dil_n[2, 2, :3]) == (128, 128, 255)

    # Graph-level mip_safe_dilate node + mip_preview_level=2 check
    g = TexGraph(64, 64, tileable=False, seed=7, name="dilate_test")
    b = g.add("constant", id="col", value="#3A5878")
    d = g.add("mip_safe_dilate", id="dil", inputs={"src": b}, bleed_px=6, mip_preview_level=2)
    g.set_output(d)
    ev = g.evaluate(build_pbr=False)
    assert ev.output.shape == (64, 64, 4)
    assert float(ev.output.min()) >= 0.0 and float(ev.output.max()) <= 1.0

    # Direct op smoke check
    direct = mip_safe_dilate_op(rgba.astype(np.float32) / 255.0, bleed_px=6, preserve_alpha=True)
    assert direct.shape == (64, 64, 4)
    assert direct[16, 30, 0] > 0.85 and direct[16, 30, 3] == 0.0


def test_v17_3d_decal_stamp_front_obb_occlusion_and_backface_culling() -> None:
    bb = _make_two_occluding_boxes_bbmodel()
    mc = bake_bbmodel_maps(bb)
    assert mc.obb_centers is not None and mc.obb_centers.shape == (2, 3)
    assert mc.obb_RTs is not None and mc.obb_RTs.shape == (2, 3, 3)
    assert mc.obb_halfs is not None and mc.obb_halfs.shape == (2, 3)

    H, W = 64, 64
    base_mat = {
        "albedo": np.full((H, W, 4), [0.15, 0.18, 0.24, 1.0], dtype=np.float32),
        "height": np.full((H, W), 0.5, dtype=np.float32),
        "roughness": np.full((H, W), 0.65, dtype=np.float32),
        "metallic": np.zeros((H, W), dtype=np.float32),
        "ao": np.ones((H, W), dtype=np.float32),
        "emissive": np.zeros((H, W, 4), dtype=np.float32),
    }

    stamp = {
        "stencil": "chrono_sigil",
        "tint_color": "#FF22AA",
        "origin": [0.0, 6.0, 20.0],
        "forward": [0.0, 0.0, -1.0],
        "up": [0.0, 1.0, 0.0],
        "scale_u": 10.0,
        "scale_v": 10.0,
        "max_angle_deg": 75.0,
        "opacity": 1.0,
        "occlusion_test": True,
    }
    res_occ = decal_stamp_op(H, W, base=base_mat, mesh_context=mc, stamps_json=json.dumps([stamp]))
    alb_occ = res_occ["albedo"]

    # Front plate +Z face (UV [2..26, 2..26]) MUST receive bright magenta decal
    front_south_diff = float(np.max(np.abs(alb_occ[4:24, 4:24, 0] - 0.15)))
    assert front_south_diff > 0.35, f"Expected front face to be stamped, diff={front_south_diff}"

    # Front plate -Z face (UV [34..58, 2..26]) faces away -> MUST NOT be stamped (backface culling)
    front_north_diff = float(np.max(np.abs(alb_occ[4:24, 36:56, 0] - 0.15)))
    assert front_north_diff < 1e-4, f"Expected backface to remain clean, diff={front_north_diff}"

    # Rear plate +Z face (UV [2..26, 34..58]) is occluded behind front_plate -> MUST NOT be stamped!
    rear_south_diff = float(np.max(np.abs(alb_occ[36:56, 4:24, 0] - 0.15)))
    assert rear_south_diff < 1e-4, f"Expected occluded rear box to remain clean, diff={rear_south_diff}"

    # With occlusion_test=False, rear plate +Z face SHOULD be stamped
    stamp_no_occ = dict(stamp, occlusion_test=False)
    res_no_occ = decal_stamp_op(H, W, base=base_mat, mesh_context=mc, stamps_json=json.dumps([stamp_no_occ]))
    rear_no_occ_diff = float(np.max(np.abs(res_no_occ["albedo"][36:56, 4:24, 0] - 0.15)))
    assert rear_no_occ_diff > 0.35

    # Verify all 8 curated stencils execute cleanly
    assert len(DECAL_STENCILS) == 8
    for sid in DECAL_STENCILS:
        st_i = dict(stamp, stencil=sid)
        out_i = decal_stamp_op(H, W, base=base_mat, mesh_context=mc, stamps_json=json.dumps([st_i]))
        assert float(np.max(np.abs(out_i["albedo"][4:24, 4:24, :3] - base_mat["albedo"][4:24, 4:24, :3]))) > 0.05


def test_v17_kaelith_skeleton_and_animations_extraction() -> None:
    found = _find_kaelith_dragonborn_paths()
    assert found is not None, "Kaelith .bbmodel must exist in workspace"
    bb_doc = json.loads(found[0].read_text(encoding="utf-8"))
    mesh = extract_bbmodel_webgl_mesh(bb_doc)

    bones = mesh["bones"]
    anims = mesh["animations"]
    assert len(bones) >= 30
    assert len(mesh["local_positions"]) == len(mesh["positions"])
    assert len(mesh["local_normals"]) == len(mesh["normals"])
    assert len(mesh["local_tangents"]) == len(mesh["tangents"])
    assert len(mesh["bone_ids"]) == mesh["vertex_count"]

    # Topological ordering: parent index is always < child bone id
    for b in bones:
        assert b["parent"] < b["id"]

    # All 4 Kaelith clips extracted cleanly
    clip_names = {a["name"] for a in anims}
    for expected in ("idle_chronal", "walk_stride", "cast_time_stop", "artificer_inspect"):
        assert expected in clip_names, f"Missing Kaelith clip {expected} in {clip_names}"
    for a in anims:
        assert a["length"] > 0.2
        assert len(a["tracks"]) >= 10

    # Verify rest-pose FK from bones + local_positions reproduces static world positions exactly!
    world_M: list[np.ndarray] = []
    for b in bones:
        p_idx = b["parent"]
        p_orig = np.array(bones[p_idx]["origin"], dtype=np.float64) if p_idx >= 0 else np.zeros(3)
        delta = np.array(b["origin"], dtype=np.float64) - p_orig
        R = _euler_xyz_mat(b["rest_rot"])
        M_loc = np.eye(4, dtype=np.float64)
        M_loc[:3, :3] = R
        M_loc[:3, 3] = delta
        world_M.append(world_M[p_idx] @ M_loc if p_idx >= 0 else M_loc)

    pos = np.asarray(mesh["positions"], dtype=np.float64).reshape(-1, 3)
    lpos = np.asarray(mesh["local_positions"], dtype=np.float64).reshape(-1, 3)
    bids = mesh["bone_ids"]
    max_err = 0.0
    for vi in range(0, len(bids), 17):
        bi = bids[vi]
        if bi >= 0:
            p4 = world_M[bi] @ np.array([lpos[vi, 0], lpos[vi, 1], lpos[vi, 2], 1.0])
            err = float(np.linalg.norm(p4[:3] - pos[vi]))
            max_err = max(max_err, err)
    assert max_err < 1e-2, f"Rest-pose FK mismatch: max_err={max_err}"


def test_v17_multi_asset_atlas_baker_and_cli(tmp_path: Path) -> None:
    in_dir = tmp_path / "assets_in"
    out_dir = tmp_path / "atlas_out"
    in_dir.mkdir()

    # Asset 1: Sculpted Temple Monolith
    bb1 = build_sample_monolith_bbmodel()
    (in_dir / "temple_monolith.bbmodel").write_text(json.dumps(bb1), encoding="utf-8")
    # Asset 2: Two-box fixture
    bb2 = _make_two_occluding_boxes_bbmodel()
    (in_dir / "decal_fixture.bbmodel").write_text(json.dumps(bb2), encoding="utf-8")

    mf = bake_multi_asset_atlas(in_dir, out_dir, stem="props_shared", bleed_px=4, canvas_mode="pot")
    assert mf["ok"] is True
    assert mf["asset_count"] == 2
    aw, ah = mf["atlas_width"], mf["atlas_height"]
    # Exact Power-of-Two verification
    assert (aw & (aw - 1)) == 0 and (ah & (ah - 1)) == 0
    assert mf["fill_ratio"] > 0.1
    assert len(mf["remapped_bbmodels"]) == 2
    for key in ("albedo", "normal", "orm", "emissive"):
        fp = Path(mf["files"][key])
        assert fp.is_file()
        with Image.open(fp) as im:
            assert im.size == (aw, ah)

    # Every remapped .bbmodel face UV fits within [0..aw, 0..ah]
    for bb_path_s in mf["remapped_bbmodels"]:
        doc = json.loads(Path(bb_path_s).read_text(encoding="utf-8"))
        assert doc["resolution"] == {"width": aw, "height": ah}
        assert doc["textures"][0]["source"].startswith("data:image/png;base64,")
        for elem in doc["elements"]:
            for finfo in (elem.get("faces") or {}).values():
                u0, v0, u1, v1 = finfo["uv"][:4]
                assert 0.0 <= min(u0, u1) <= max(u0, u1) <= aw + 0.01
                assert 0.0 <= min(v0, v1) <= max(v0, v1) <= ah + 0.01

    # CLI smoke test (`synccraft atlas-bake`)
    cli_out = tmp_path / "atlas_cli_out"
    repo_src = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "synccraft.cli",
            "atlas-bake",
            str(in_dir),
            "--out",
            str(cli_out),
            "--stem",
            "cli_atlas",
            "--bleed",
            "4",
            "--canvas",
            "tight",
        ],
        env={"PYTHONPATH": repo_src},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, f"CLI failed: {proc.stderr}"
    assert (cli_out / "cli_atlas_albedo.png").is_file()
    assert (cli_out / "cli_atlas_atlas_manifest.json").is_file()


def test_v17_six_engine_export_presets_spec_parity(tmp_path: Path) -> None:
    assert len(ENGINE_EXPORT_PRESETS) == 7
    bb = build_sample_monolith_bbmodel()
    mc = bake_bbmodel_maps(bb)
    g = TexGraph(mc.width, mc.height, tileable=False, seed=42, name="monolith_spec", mesh_context=mc)
    g.domain = "entity"
    n1 = g.add("region_base_fill", id="base", stops=["#2B3242", "#526384", "#98A9C8", "#E6EEF8"])
    n2 = g.add("region_restrained_noise", id="detail", inputs={"mat": n1}, pattern="stone_chisel", amplitude=0.18)
    n3 = g.add("region_bevel_and_ao", id="bevel", inputs={"mat": n2})
    g.set_output_material(n3, normal_strength=2.6)

    # 1. Unreal Engine 5 (DirectX Normal + ORM)
    ue_dir = tmp_path / "ue5"
    ue = export_engine_preset(g, "unreal_ue5", stem="hero", out_dir=ue_dir, bleed_px=4)
    assert {"T_hero_D.png", "T_hero_N.png", "T_hero_ORM.png", "T_hero_E.png"}.issubset(set(ue["file_names"]))

    # 2. Unity HDRP (OpenGL Normal + MaskMap B=255 Neutral Detail Mask + HeightMap)
    un_dir = tmp_path / "unity"
    un = export_engine_preset(g, "unity_hdrp", stem="hero", out_dir=un_dir, bleed_px=4)
    assert {"hero_Albedo.png", "hero_Normal.png", "hero_MaskMap.png", "hero_HeightMap.png"}.issubset(set(un["file_names"]))
    mask_arr = np.asarray(Image.open(un_dir / "hero_MaskMap.png").convert("RGBA"))
    assert int(mask_arr[..., 2].min()) == 255 and int(mask_arr[..., 2].max()) == 255

    # Compare UE5 DirectX Normal G vs Unity OpenGL Normal G -> exact 255 - G inversion!
    ue_n = np.asarray(Image.open(ue_dir / "T_hero_N.png").convert("RGBA"))
    un_n = np.asarray(Image.open(un_dir / "hero_Normal.png").convert("RGBA"))
    assert np.array_equal(ue_n[..., 1], (255 - un_n[..., 1]).astype(np.uint8))

    # 3. Minecraft Java LabPBR 1.3 (DirectX _n.G + _s)
    lp_dir = tmp_path / "labpbr"
    lp = export_engine_preset(g, "java_labpbr", stem="hero", out_dir=lp_dir, bleed_px=4)
    assert {"hero.png", "hero_n.png", "hero_s.png"}.issubset(set(lp["file_names"]))
    lp_n = np.asarray(Image.open(lp_dir / "hero_n.png").convert("RGBA"))
    assert int(lp_n[..., 3].min()) >= 1  # LabPBR POM height never 0

    # 4. Minecraft Bedrock RTX / Vibrant Visuals 1.21.30 (_mers + texture_set.json)
    br_dir = tmp_path / "bedrock"
    br = export_engine_preset(g, "bedrock_rtx", stem="hero", out_dir=br_dir, bleed_px=4)
    assert {"hero.png", "hero_mer.png", "hero_mers.png", "hero_normal.png", "hero.texture_set.json"}.issubset(set(br["file_names"]))
    ts = json.loads((br_dir / "hero.texture_set.json").read_text(encoding="utf-8"))
    assert ts["format_version"] == "1.21.30"
    assert ts["minecraft:texture_set"]["metalness_emissive_roughness_subsurface"] == "hero_mers"

    # 5. Maya LookdevX MaterialX 1.39 (open_pbr_surface) & 1.38 (standard_surface)
    mx_dir = tmp_path / "mtlx"
    mx139 = export_engine_preset(g, "maya_materialx", stem="hero", out_dir=mx_dir, shader_model="open_pbr_surface")
    xml_139 = (mx_dir / "hero.mtlx").read_text(encoding="utf-8")
    assert 'version="1.39"' in xml_139 and "open_pbr_surface" in xml_139
    mx138_dir = tmp_path / "mtlx138"
    export_engine_preset(g, "maya_materialx", stem="hero", out_dir=mx138_dir, shader_model="standard_surface")
    xml_138 = (mx138_dir / "hero.mtlx").read_text(encoding="utf-8")
    assert 'version="1.38"' in xml_138 and "standard_surface" in xml_138

    # 6. Blockbench Pre-Baked
    bb_dir = tmp_path / "bb_baked"
    bb_res = export_engine_preset(g, "blockbench_baked", stem="hero", out_dir=bb_dir, bbmodel_doc=bb)
    assert {"hero_baked.png", "hero_baked.bbmodel"}.issubset(set(bb_res["file_names"]))

    # 7. Full PBR v2.0 (browser download mode)
    dl = export_engine_preset(g, "full_pbr_v2", stem="hero", out_dir=None, bleed_px=4)
    assert dl["mode"] == "download"
    assert dl["bundle_filename"] == "hero.full_pbr_v2.synccraft.json"
    assert "hero.mtlx" in dl["text_files"]


def test_v17_studio_http_endpoints_decals_presets_and_atlas(tmp_path: Path) -> None:
    server, _sess = create_studio_server(host="127.0.0.1", port=0, initial_sample="kaelith_dragonborn_3d")
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"
    try:
        health = _http_json(f"{base}/api/health")
        assert health["ok"] is True
        assert health["engine_version"] == "17.0"
        assert health["node_count"] == 232

        schema = _http_json(f"{base}/api/schema")
        assert len(schema["decal_stencils"]) == 8
        assert len(schema["export_presets"]) == 7

        state0 = _http_json(f"{base}/api/state")
        assert state0["mesh"] is not None
        assert len(state0["mesh"]["bones"]) >= 30
        assert len(state0["mesh"]["animations"]) >= 4

        # 3D Decal Stamp API: add -> add -> undo_last -> clear
        st1 = _http_json(
            f"{base}/api/decal_stamp/apply",
            {
                "action": "add",
                "stamp": {
                    "stencil": "chrono_sigil",
                    "tint_color": "#F0C85A",
                    "origin": [0.0, 22.0, 36.0],
                    "forward": [0.0, 0.0, -1.0],
                    "up": [0.0, 1.0, 0.0],
                    "scale_u": 12.0,
                    "opacity": 0.95,
                },
            },
        )
        assert st1["stamp_count"] == 1
        assert st1["created_node_id"] is not None
        decal_nid = st1["decal_node_id"]

        ev1 = _http_json(f"{base}/api/evaluate", {"include_thumbs": False, "include_qa": False})
        assert ev1["ok"] is True and ev1["channels"]["albedo"].startswith("data:image/png;base64,")

        st2 = _http_json(
            f"{base}/api/decal_stamp/apply",
            {
                "node_id": decal_nid,
                "action": "add",
                "stamp": {"stencil": "arcane_rune_ring", "origin": [0.0, 24.0, 36.0], "forward": [0.0, 0.0, -1.0], "scale_u": 10.0},
            },
        )
        assert st2["stamp_count"] == 2

        st3 = _http_json(f"{base}/api/decal_stamp/apply", {"node_id": decal_nid, "action": "undo_last"})
        assert st3["stamp_count"] == 1

        st4 = _http_json(f"{base}/api/decal_stamp/apply", {"node_id": decal_nid, "action": "clear"})
        assert st4["stamp_count"] == 0

        # Engine Export Preset API
        exp_ue = _http_json(
            f"{base}/api/export/preset",
            {"preset_id": "unreal_ue5", "stem": "kaelith", "bleed_px": 4, "out_dir": str(tmp_path / "ue_http")},
        )
        assert exp_ue["ok"] is True and "T_kaelith_N.png" in exp_ue["file_names"]

        exp_un = _http_json(
            f"{base}/api/export/preset",
            {"preset_id": "unity_hdrp", "stem": "kaelith", "bleed_px": 4},
        )
        assert exp_un["ok"] is True and "kaelith_MaskMap.png" in exp_un["file_names"]

        # Multi-Asset Atlas Bake API
        in_d = tmp_path / "atlas_http_in"
        in_d.mkdir()
        (in_d / "monolith.bbmodel").write_text(json.dumps(build_sample_monolith_bbmodel()), encoding="utf-8")
        (in_d / "fixture.bbmodel").write_text(json.dumps(_make_two_occluding_boxes_bbmodel()), encoding="utf-8")
        atl = _http_json(
            f"{base}/api/atlas/bake",
            {
                "input_dir": str(in_d),
                "out_dir": str(tmp_path / "atlas_http_out"),
                "stem": "http_atlas",
                "bleed_px": 4,
                "canvas_mode": "pot",
            },
        )
        assert atl["ok"] is True and atl["asset_count"] == 2
        assert atl["albedo_preview_url"].startswith("data:image/png;base64,")
    finally:
        server.shutdown()
        server.server_close()
