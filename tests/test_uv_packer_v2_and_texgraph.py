"""Comprehensive unit tests for Semantic Anatomical UV Packer v2 and `synccraft.texgraph` Parametric Pixel Composer."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from synccraft.backends import export_bbmodel
from synccraft.cli import main as cli_main
from synccraft.dsl import parse_program
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.shading.uv_packer import (
    allocate_uv_layout,
    classify_anatomical_zone,
    measure_uv_utilization,
    pack_semantic_uv_atlas,
    render_uv_zone_map,
    repack_bbmodel_uvs,
)
from synccraft.texgraph import (
    TexGraph,
    anisotropic_kuwahara,
    bake_bbmodel_maps,
    bilateral_denoise,
    build_molten_scales_block,
    build_mossy_carved_brick,
    build_neon_hex_block,
    build_prism_blade_item,
    cellular_automata,
    export_html_inspector,
    gray_scott_turing,
    morphological_despeckle,
    non_local_means_denoise,
    palette_quantize,
    pixel_cluster_mode,
    selout_outline,
    total_variation_denoise,
)

_CREATURE_DSL = """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 32, size_step: 0.5, audit_profile: sculpted }
@palette {
  obsidian: ramp(base_lch=[0.24, 0.055, 268], steps=5, hue_shift=[-18, 28])
  gold: ramp(base_lch=[0.68, 0.095, 82], steps=5, hue_shift=[-24, 18])
  neon: ramp(base_lch=[0.82, 0.16, 202], steps=4, emissive=true)
}
@rig {
  root bone pelvis_root {
    pivot_local: [0, 12, 0]
    box torso_core { size: [10, 8, 14], material: obsidian }
    bone head joint(parent=torso_core, at=[0, 2, 7], align=[0, 0, 0]) {
      box skull { size: [6, 5, 7], material: obsidian }
      box eye_l place(anchor=[0, 0, 0], align=[0, 0, 0], offset=[-3, 1, 2]) { size: [1, 1.5, 2], material: neon, uv_scale: 2 }
    }
    bone wing_l_upper joint(parent=torso_core, at=[-5, 3, 0], align=[0, 0, 0]) mirror_x(wing_r_upper) {
      box wing_spar { size: [14, 2, 3], material: obsidian }
      plane wing_sail place(anchor=[0, 0, 0], align=[0, 0, 0], offset=[-7, 0, -5]) { size: [14, 10], normal: y, material: obsidian }
    }
    bone leg_l_thigh joint(parent=torso_core, at=[-4, -4, -3], align=[0, 0, 0]) mirror_x(leg_r_thigh) {
      box thigh_box { size: [3, 8, 4], material: obsidian }
      box shin_gold place(anchor=[0, 0, 0], align=[0, 0, 0], offset=[0, -6, 0]) { size: [2.5, 6, 3], material: gold }
    }
    bone tail_1 joint(parent=torso_core, at=[0, 0, -7], align=[0, 0, 0]) {
      box tail_seg { size: [4, 4, 10], material: obsidian }
    }
  }
}
"""


class TestUVPackerV2(unittest.TestCase):
    def test_anatomical_zone_classifier(self) -> None:
        self.assertEqual(classify_anatomical_zone("wing_l_f1", "f1_prox_spar"), "wings")
        self.assertEqual(classify_anatomical_zone("head", "eye_l", is_emissive=True), "head")
        self.assertEqual(classify_anatomical_zone("chest", "chest_gravity_cyan", is_emissive=True), "emissive")
        self.assertEqual(classify_anatomical_zone("leg_l_shin", "shin_core"), "hindlimbs")
        self.assertEqual(classify_anatomical_zone("arm_l_fore", "arm_vambrace_plate"), "forelimbs")
        self.assertEqual(classify_anatomical_zone("tail_plume_r_1", "plume_l_2_vane"), "tail")

    def test_semantic_packer_adaptive_mirror_cull_and_bbmodel_repack(self) -> None:
        prog = parse_program(_CREATURE_DSL)
        scene = compile_geometry(prog)

        legacy = allocate_uv_layout(prog, scene, packing_mode="legacy")
        sem = pack_semantic_uv_atlas(prog, scene, padding=1, mirror_lr_uv="adaptive", cull_zero_area=True)
        m = sem["metrics"]

        # Zero-area plane side edges (4 per plane) must be culled
        self.assertGreaterEqual(m["culled_zero_area_faces"], 4)
        # Adaptive L/R mirroring must share wings & hindlimbs while keeping head/torso/tail unique
        self.assertGreater(m["mirrored_shared_faces"], 0)
        self.assertIn("wing_r_upper__wing_spar", sem["mirror_source_map"])
        self.assertIn("leg_r_thigh__thigh_box", sem["mirror_source_map"])
        self.assertNotIn("skull", sem["mirror_source_map"])

        # Mirrored slave faces have flipped u0 > u1 referencing master island
        u0, v0, u1, v1 = sem["faces"]["wing_r_upper__wing_spar"]["north"]
        self.assertGreater(u0, u1)
        self.assertLess(v0, v1)

        # Atlas must be smaller or higher padded utilization than legacy
        self.assertLessEqual(sem["width"] * sem["height"], legacy["width"] * legacy["height"])
        self.assertGreater(m["padded_utilization_pct"], 30.0)

        # Zone map PNG renders cleanly
        zpng = render_uv_zone_map(sem)
        self.assertTrue(zpng.startswith(b"\x89PNG"))

        # Standalone .bbmodel repacker works cleanly on exported legacy .bbmodel
        atlas_leg = {
            "width": legacy["width"],
            "height": legacy["height"],
            "faces": legacy["faces"],
            "rgba": np.full((legacy["height"], legacy["width"], 4), 180, dtype=np.uint8),
        }
        bb = export_bbmodel(scene, name="test_creature", atlas=atlas_leg, program=prog)
        repacked, report, zpng2 = repack_bbmodel_uvs(bb, padding=1, mirror_lr_uv="adaptive")
        self.assertTrue(zpng2.startswith(b"\x89PNG"))
        self.assertGreater(report["metrics"]["mirrored_shared_faces"], 0)
        self.assertEqual(repacked["resolution"]["width"], report["width"])
        self.assertEqual(repacked["resolution"]["height"], report["height"])
        util = measure_uv_utilization(repacked)
        self.assertGreater(util["utilization_pct"], 0.0)


class TestTexGraphEngine(unittest.TestCase):
    def test_all_noise_pattern_sim_denoise_and_recipe_roundtrip(self) -> None:
        g = TexGraph(32, 32, tileable=True, seed=77, name="full_ops_smoke")
        n_v = g.value_noise(scale=4.0)
        n_p = g.perlin(scale=4.0, loop_speed=1.0)
        n_f = g.fbm(scale=4.0, octaves=3, mode="ridged")
        n_vo = g.voronoi(scale=4.0, output="cracks")
        n_g = g.gabor(scale=8.0, angle_deg=35.0)
        w = g.domain_warp(n_f, strength=0.06, mode="curl")
        p_gr = g.grid(cols=4, rows=4)
        p_br = g.brick(cols=2, rows=4, style="herringbone")
        p_hx = g.hex_grid(cols=4)
        p_tr = g.tri_grid(cols=4)
        p_di = g.diamond(cols=4, rows=4, style="chevron")
        p_tu = g.truchet(cols=4)
        p_mo = g.mosaic(scale=5.0)
        p_sc = g.scales(cols=4, rows=4, shape="dragon")
        p_fe = g.feathers(cols=2)
        p_fu = g.fur(cols=4, rows=4)
        p_ra = g.radial_rings(freq=4.0, spokes=6)
        p_ve = g.energy_veins(scale=4.0, style="circuit", flow_speed=1.0)
        s_gs = g.gray_scott(preset="spots", steps=40)
        s_ca = g.cellular_automata(preset="cave", steps=3)
        s_it = g.iterate(p_br, operation="erode_weather", steps=2)
        s_wd = g.weathering_drip(p_sc, cavity=n_vo, streak_length=3)

        # Combine several signals and exercise every algorithmic denoiser
        mix = g.blend(g.blend(p_sc, s_gs, mode="soft_light", opacity=0.4), g.blend(n_v, n_p, opacity=0.5), mode="overlay", opacity=0.25)
        mix = g.blend(mix, g.blend(w, g.blend(n_g, g.blend(p_gr, g.blend(p_hx, g.blend(p_tr, g.blend(p_di, g.blend(p_tu, g.blend(p_mo, g.blend(p_fe, g.blend(p_fu, g.blend(p_ra, g.blend(s_ca, g.blend(s_it, s_wd, opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.2)
        f1 = g.kuwahara(mix, radius=1, sharpness=8.0)
        f2 = g.bilateral(f1, radius=1)
        f3 = g.tv_denoise(f2, weight=0.05, iterations=6)
        f4 = g.nlm_denoise(f3, patch_radius=1, search_radius=2)
        sh = g.shade(f4, use_mesh=False)
        col = g.hue_ramp(sh, base_lch=(0.45, 0.08, 220.0), steps=6, dither="bayer4")
        col_q = g.quantize(col, palette_hex=["#14182E", "#263459", "#46668E", "#7FA1C3", "#C9DFEE"])
        col_c = g.cluster_mode(g.despeckle(col_q))
        col_s = g.selout(col_c, silhouette=False, interior=True)
        alb, em = g.emissive(col_s, p_ve, glow_bleed_px=1)
        g.set_output(alb, emissive_node=em)

        ev0 = g.evaluate(phase=0.0, capture_intermediates=True)
        self.assertEqual(ev0.output.shape, (32, 32, 4))
        self.assertIsNotNone(ev0.emissive)
        self.assertTrue(np.isfinite(ev0.output).all())

        # Declarative JSON recipe round-trip must reproduce bit-exact output!
        recipe = g.to_recipe()
        g2 = TexGraph.from_recipe(recipe)
        ev_rt = g2.evaluate(phase=0.0)
        np.testing.assert_allclose(ev0.output, ev_rt.output, atol=1e-6)

        # Flipbook looping animation changes across phase and outputs .mcmeta
        ev_half = g.evaluate(phase=0.5)
        self.assertGreater(float(np.abs(ev0.output - ev_half.output).mean()), 1e-4)
        fb = g.render_flipbook(n_frames=4, frametime=2)
        self.assertEqual(fb["num_frames"], 4)
        self.assertEqual(fb["strip_height"], 128)
        self.assertEqual(fb["mcmeta"]["animation"]["frametime"], 2)

    def test_algorithmic_denoisers_reduce_speckle_noise(self) -> None:
        rng = np.random.default_rng(123)
        clean = np.zeros((32, 32, 4), dtype=np.float32)
        clean[:, :16, :3] = (0.2, 0.3, 0.5)
        clean[:, 16:, :3] = (0.8, 0.7, 0.5)
        clean[..., 3] = 1.0
        noisy = clean.copy()
        noisy[..., :3] = np.clip(noisy[..., :3] + rng.normal(0.0, 0.12, (32, 32, 3)).astype(np.float32), 0.0, 1.0)
        base_err = float(np.mean(np.abs(noisy[..., :3] - clean[..., :3])))

        for fn in (
            lambda x: anisotropic_kuwahara(x, radius=2, sharpness=8.0),
            lambda x: bilateral_denoise(x, radius=2, sigma_r=0.20),
            lambda x: total_variation_denoise(x, weight=0.12, iterations=12),
            lambda x: non_local_means_denoise(x, patch_radius=1, search_radius=3, h_param=0.15),
        ):
            out = fn(noisy)
            err = float(np.mean(np.abs(out[..., :3] - clean[..., :3])))
            self.assertLess(err, base_err * 0.75)

        # Orphan single-pixel despeckle
        speck = clean.copy()
        speck[8, 8, :3] = (1.0, 0.0, 1.0)
        fixed = morphological_despeckle(speck, threshold=0.2)
        np.testing.assert_allclose(fixed[8, 8, :3], clean[8, 8, :3], atol=1e-4)

        # Direct helper calls smoke cleanly
        self.assertEqual(gray_scott_turing(16, 16, steps=20).shape, (16, 16))
        self.assertEqual(cellular_automata(16, 16, steps=3).shape, (16, 16))
        self.assertEqual(pixel_cluster_mode(clean).shape, (32, 32, 4))
        self.assertEqual(selout_outline(clean).shape, (32, 32, 4))
        self.assertEqual(palette_quantize(clean, palette_hex=["#334D80", "#CCB380"], dither="floyd").shape, (32, 32, 4))

    def test_3d_mesh_baker_ao_and_presets_and_cli(self) -> None:
        prog = parse_program(_CREATURE_DSL)
        scene = compile_geometry(prog)
        sem = pack_semantic_uv_atlas(prog, scene, padding=1, mirror_lr_uv="adaptive")
        atlas = {
            "width": sem["width"],
            "height": sem["height"],
            "faces": sem["faces"],
            "rgba": np.full((sem["height"], sem["width"], 4), 200, dtype=np.uint8),
        }
        bb = export_bbmodel(scene, name="bake_test", atlas=atlas, program=prog)
        ctx = bake_bbmodel_maps(bb)
        self.assertEqual((ctx.height, ctx.width), (sem["height"], sem["width"]))
        self.assertGreater(int(ctx.island_mask.sum()), 100)
        # Baked AO must show genuine multi-box darkening near joints (min < 0.85, max > 0.90)
        ao_active = ctx.baked_ao[ctx.island_mask]
        self.assertLess(float(ao_active.min()), 0.85)
        self.assertGreater(float(ao_active.max()), 0.90)
        self.assertGreater(float(ctx.zone_mask("wings").sum()), 0.0)
        self.assertGreater(float(ctx.zone_mask("head").sum()), 0.0)

        # Presets + HTML Inspector + CLI smoke test
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            for builder in (build_neon_hex_block, build_mossy_carved_brick, build_molten_scales_block, build_prism_blade_item):
                g = builder(size=32)
                ev = g.evaluate(phase=0.0, capture_intermediates=True)
                self.assertEqual(ev.output.shape, (32, 32, 4))
            html_out = export_html_inspector(build_neon_hex_block(32), td / "inspector.html")
            self.assertTrue(html_out.is_file())
            html_txt = html_out.read_text(encoding="utf-8")
            self.assertIn("SyncCraft TexGraph Studio", html_txt)
            self.assertIn("LabPBR 1.3", html_txt)

            out_png = td / "hex.png"
            rc = cli_main(["texgraph", "render", "preset:neon_hex_conduit", "-o", str(out_png), "--frames", "4", "--size", "32", "--pbr"])
            self.assertEqual(rc, 0)
            self.assertTrue((td / "hex_flipbook.png").is_file())
            self.assertTrue((td / "hex_n_flipbook.png").is_file())
            self.assertTrue((td / "hex_s_flipbook.png").is_file())
            self.assertTrue((td / "hex_mer_flipbook.png").is_file())
            self.assertTrue((td / "hex_flipbook.png.mcmeta").is_file())

    def test_full_pbr_bundle_labpbr_invariants_eyes_and_wing_scallop(self) -> None:
        from synccraft.texgraph import paint_creature_eye, wing_scallop_alpha

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            g = build_neon_hex_block(32)
            # Round-trip v2 recipe with PBR channels
            g_rt = TexGraph.from_recipe(g.to_recipe())
            self.assertIn("height", g_rt.pbr_nodes)
            self.assertIn("roughness", g_rt.pbr_nodes)
            self.assertIn("metallic", g_rt.pbr_nodes)

            res = g_rt.export_pbr_bundle(td, stem="hex_pbr")
            ev = res["eval"]
            self.assertIsNotNone(ev.pbr_maps)
            self.assertIsNotNone(ev.pbr_preview)
            n_u8 = ev.pbr_maps["labpbr_n"]
            s_u8 = ev.pbr_maps["labpbr_s"]
            # LabPBR 1.3 spec invariants: _n alpha height NEVER 0; _s emissive uses 255 for non-emissive
            self.assertGreaterEqual(int(n_u8[..., 3].min()), 1)
            self.assertIn(255, s_u8[..., 3])
            self.assertTrue((td / "hex_pbr_n.png").is_file())
            self.assertTrue((td / "hex_pbr_s.png").is_file())
            self.assertTrue((td / "hex_pbr_mer.png").is_file())
            self.assertTrue((td / "hex_pbr.texture_set.json").is_file())
            self.assertTrue((td / "hex_pbr_pbr_sheet.png").is_file())

        # Symmetric 7-layer dragon eye & wing scallop cutouts
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 16, dtype=np.float32), np.linspace(0.0, 1.0, 24, dtype=np.float32), indexing="ij")
        eye_l, em_l = paint_creature_eye(xx, yy, side_sign=np.full_like(xx, -1.0))
        eye_r, em_r = paint_creature_eye(1.0 - xx, yy, side_sign=np.full_like(xx, 1.0))
        np.testing.assert_allclose(eye_l, eye_r, atol=1e-5)
        np.testing.assert_allclose(em_l, em_r, atol=1e-5)
        # Eye must have dark slit pupil AND bright iris/catchlight!
        self.assertLess(float(eye_l[..., :3].min()), 0.1)
        self.assertGreater(float(eye_l[..., :3].max()), 0.95)

        sc_a, sc_r = wing_scallop_alpha(xx, yy, arches=2, depth=0.30)
        self.assertGreater(float((sc_a < 0.5).mean()), 0.10)
        self.assertGreater(float((sc_a > 0.5).mean()), 0.55)
        self.assertGreater(float((sc_r > 0.5).sum()), 10.0)

    def test_v3_smooth_gradient_path_painterly_fx_and_ui_node_schema(self) -> None:
        from synccraft.texgraph import (
            NODE_CATALOG,
            build_dragon_wing_membrane_tile,
            build_stylized_wood_and_foliage,
            build_temple_carved_stone,
            export_node_schema_json,
            gradient_eye_paint,
        )

        # 1. Verify UI-ready typed node registry & CLI schema export
        schema = export_node_schema_json()
        self.assertGreaterEqual(schema["node_count"], 65)
        self.assertIn("Gradients & Paths", schema["categories"])
        self.assertIn("Painterly & Bio", schema["categories"])
        self.assertIn("PBR & Normal", schema["categories"])
        self.assertEqual(cli_main(["texgraph", "schema"]), 0)

        # 2. Exercise smooth gradient paths, painterly nodes, FX, normal blend, SDF, HBAO, exposed params & groups
        g = TexGraph(32, 32, tileable=True, seed=99, name="v3_full_pipeline")
        gf = g.gradient_field("radial", radius=0.65, ease="smoothstep", id="rad_g")
        sp = g.spline_path([(0.1, 0.2), (0.5, 0.8), (0.9, 0.2)], stroke_width=0.14, output="ribbon", id="spl_rib")
        br = g.brushstrokes(angle_deg=20.0, id="br_1")
        pl = g.layered_plates(cols=4, rows=4, id="pl_1")
        ts = g.temple_stone(cols=2, rows=4, id="ts_1")
        wp = g.wood_planks(planks=4, id="wp_1")
        fc = g.foliage_clusters(cols=4, rows=4, id="fc_1")
        pi = g.per_island_gradient("top_to_bottom", id="pi_1")
        bf = g.body_flow_gradient("top_light_wrap", id="bf_1")

        # Transform, SDF, morphology, blur, sharpen, edge, HBAO, normal blend
        w_tr = g.tile_transform(pl, scale_x=1.0, rotate_deg=0.0, id="tr_1")
        w_tw = g.twirl(gf, strength_deg=30.0, id="tw_1")
        w_wv = g.wave(br, amp_x=0.03, id="wv_1")
        w_pl = g.polar_warp(gf, to_polar=True, id="pol_1")
        m_sdf = g.sdf(sp, max_dist_px=6.0, id="sdf_1")
        m_mo = g.morph(sp, op="dilate", radius=1, id="mo_1")
        m_ed = g.sobel_edges(pl, id="ed_1")
        m_ao = g.ao_from_height(pl, radius=2, id="hbao_1")
        n_base = g.height_to_normal(pl, strength=2.0, id="n_base")
        n_det = g.height_to_normal(ts, strength=1.2, id="n_det")
        n_mix = g.normal_blend(n_base, n_det, strength=0.7, id="n_mix")

        h_comb = g.blend(g.blend(w_tr, g.blend(w_tw, g.blend(w_wv, g.blend(w_pl, g.blend(m_sdf, g.blend(m_mo, g.blend(m_ed, g.blend(wp, g.blend(fc, g.blend(pi, bf, opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.3), opacity=0.25), sp, mode="screen", opacity=0.35, id="h_comb")
        h_crv = g.curves(g.blur(h_comb, radius=1, id="bl_1"), [(0.0, 0.05), (0.5, 0.55), (1.0, 0.98)], id="crv_1")
        h_db = g.directional_blur(h_crv, angle_deg=30.0, length=3, id="db_1")
        c_ms = g.multi_stop_gradient(h_db, stops=[(0.0, "#190E33"), (0.5, "#6D28D9"), (1.0, "#67E8F9")], smooth_blend=0.7, quantize_steps=6, id="ms_col")
        c_cos = g.cosine_gradient(gf, id="cos_col")
        c_mix = g.blend(c_ms, c_cos, opacity=0.2, id="c_mix")
        c_cb = g.color_balance(c_mix, shadows=(-0.02, 0.0, 0.04), highlights=(0.03, 0.02, 0.0), id="cb_1")
        c_ok = g.oklch_adjust(c_cb, hue_shift_deg=6.0, chroma_scale=1.1, id="ok_1")
        c_po = g.posterize_soft(c_ok, steps=8, blend=0.4, id="po_1")
        c_sh = g.sharpen(c_po, amount=0.3, id="sh_1")
        c_bl = g.bloom(c_sh, threshold=0.7, radius=2, intensity=0.4, id="blm_1")

        g.set_output(c_bl, height_node=h_comb, normal_node=n_mix, ao_node=m_ao)
        g.expose_param("smoothness", "ms_col", "smooth_blend", min_val=0.0, max_val=1.0)
        g.add_group("Color Grading", ["ms_col", "cb_1", "ok_1", "blm_1"], color="#1E293B")

        ev0 = g.evaluate(phase=0.0, build_pbr=True)
        self.assertEqual(ev0.output.shape, (32, 32, 4))
        self.assertTrue(np.isfinite(ev0.output).all())
        self.assertIsNotNone(ev0.pbr_maps)

        # Round-trip v2.1 recipe with exposed_params & groups + set_exposed override
        g_rt = TexGraph.from_recipe(g.to_recipe())
        self.assertIn("smoothness", g_rt.exposed_params)
        self.assertEqual(len(g_rt.groups), 1)
        g_rt.set_exposed("smoothness", 0.2)
        ev_mod = g_rt.evaluate(phase=0.0)
        self.assertEqual(ev_mod.output.shape, (32, 32, 4))

        # 3. Verify non-standard luminous gradient eye L/R mirror symmetry & new studio presets
        yy, xx = np.meshgrid(np.linspace(0.0, 1.0, 16, dtype=np.float32), np.linspace(0.0, 1.0, 24, dtype=np.float32), indexing="ij")
        ge_l, gem_l = gradient_eye_paint(xx, yy, side_sign=np.full_like(xx, -1.0))
        ge_r, gem_r = gradient_eye_paint(1.0 - xx, yy, side_sign=np.full_like(xx, 1.0))
        np.testing.assert_allclose(ge_l, ge_r, atol=1e-5)
        np.testing.assert_allclose(gem_l, gem_r, atol=1e-5)

        for builder in (build_temple_carved_stone, build_stylized_wood_and_foliage, build_dragon_wing_membrane_tile):
            ev_p = builder(size=32).evaluate(build_pbr=True)
            self.assertEqual(ev_p.output.shape, (32, 32, 4))
            self.assertIsNotNone(ev_p.pbr_maps)

    def test_v4_multipass_material_bus_height_blend_subgraphs_and_pinion_wings(self) -> None:
        from synccraft.texgraph import (
            SUBGRAPH_REGISTRY,
            build_temple_carved_stone,
            export_node_schema_json,
            is_material_bus,
            sculpted_pinion_wing,
        )

        # 1. Verify schema v3.1 multi-port, MATERIAL bus, multipass, drivers & subgraphs metadata
        schema = export_node_schema_json()
        self.assertEqual(schema["schema_version"], "3.1")
        self.assertTrue(schema["supports_multi_port"])
        self.assertTrue(schema["supports_material_bus"])
        self.assertTrue(schema["supports_multipass"])
        self.assertTrue(schema["supports_param_drivers"])
        self.assertTrue(schema["supports_generate_iterate_simulate_animate"])
        self.assertIn("Material Bus (PBR)", schema["categories"])
        self.assertGreaterEqual(schema["subgraph_count"], 7)
        self.assertIn("sculpted_pinion_wing_pbr", SUBGRAPH_REGISTRY)
        self.assertIn("dragon_armor_plates_pbr", SUBGRAPH_REGISTRY)

        # 2. Multi-pass Temple Carved Stone with Substance Height-Blend & v3.0/3.1 lossless JSON round-trip!
        g = build_temple_carved_stone(size=32)
        self.assertEqual(len(g.passes), 4)
        self.assertIsNotNone(g.output_material_node)
        rec = g.to_recipe()
        self.assertIn(rec["version"], ("3.0", "3.1"))
        self.assertEqual(len(rec["passes"]), 4)
        self.assertEqual(rec["output_material"], g.output_material_node)

        g_rt = TexGraph.from_recipe(rec)
        self.assertEqual(len(g_rt.passes), 4)
        self.assertEqual(g_rt.output_material_node, g.output_material_node)

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            mp, mp_img = g_rt.render_multipass_sheet(td / "temple_mp.png", tile_size=128)
            self.assertEqual(len(mp.passes), 4)
            self.assertTrue((td / "temple_mp.png").is_file())
            self.assertGreater(mp_img.width, 200)
            # Every pass snapshot must carry an evaluated preview array
            for ps in mp.passes:
                self.assertIsNotNone(ps.preview_rgba)
                self.assertGreaterEqual(ps.node_count, 1)
            # HTML Inspector supports multi-pass tabs + MATERIAL bus cards cleanly
            html_out = export_html_inspector(g_rt, td / "temple_mp_inspector.html")
            html_txt = html_out.read_text(encoding="utf-8")
            self.assertIn("passBar", html_txt)
            self.assertIn("MATERIAL OUT", html_txt)
            # CLI --multipass-sheet smoke test
            rc = cli_main([
                "texgraph", "render", "preset:temple_carved_stone",
                "-o", str(td / "stone.png"), "--size", "32", "--pbr",
                "--multipass-sheet", str(td / "stone_mp.png"),
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((td / "stone_mp.png").is_file())

        # 3. Direct multi-output socket routing & Substance material_height_blend physics test
        gm = TexGraph(32, 32, tileable=True, seed=7, name="height_blend_unit")
        with gm.pass_("p1", title="Base") as p1:
            h_low = gm.constant(0.25, id="h_low")
            h_hi = gm.constant(0.85, id="h_hi")
            m_base = gm.material_make(gm.constant("#222244", id="c_b"), height=h_low, roughness=0.7, metallic=0.0, id="m_base")
            m_top = gm.material_make(gm.constant("#EAB308", id="c_t"), height=h_hi, roughness=0.15, metallic=0.95, id="m_top")
            p1.preview_ref = f"{m_base}.albedo"
        with gm.pass_("p2", title="HeightBlend") as p2:
            grad_m = gm.gradient("horizontal", id="mask_h")
            hb = gm.material_height_blend(m_base, m_top, mask=grad_m, height_offset=0.1, contrast=8.0, seam_ao_strength=0.3, id="hb_mat")
            p2.preview_ref = f"{hb}.albedo"
        gm.set_output_material(hb)
        ev_m = gm.evaluate(capture_intermediates=True, build_pbr=True)
        self.assertTrue(is_material_bus(ev_m.intermediates[hb]))
        # High-relief metallic top must poke cleanly over low-relief base on right side & respect mask=0 on far left!
        self.assertLess(float(ev_m.pbr_channels["metallic"][:, 0].mean()), 0.05)
        self.assertGreater(float(ev_m.pbr_channels["metallic"][:, -1].mean()), 0.90)
        self.assertGreater(float(ev_m.intermediates[hb]["seam_mask"].max()), 0.25)

        # 4. Wing top/bottom orientation alignment (`wing_trail_v` identical on up & down) + `sculpted_pinion_wing`!
        prog = parse_program(_CREATURE_DSL)
        scene = compile_geometry(prog)
        sem = pack_semantic_uv_atlas(prog, scene, padding=1, mirror_lr_uv="adaptive")
        atlas = {
            "width": sem["width"],
            "height": sem["height"],
            "faces": sem["faces"],
            "rgba": np.full((sem["height"], sem["width"], 4), 200, dtype=np.uint8),
        }
        bb = export_bbmodel(scene, name="wing_orient_test", atlas=atlas, program=prog)
        ctx = bake_bbmodel_maps(bb)
        self.assertEqual(ctx.wing_span_u.shape, (ctx.height, ctx.width))
        self.assertEqual(ctx.wing_trail_v.shape, (ctx.height, ctx.width))
        self.assertGreater(float(ctx.face_is_down.sum()), 0.0)

        # Sculpted Pinion Wing multi-port bus invariants
        uu, vv = np.meshgrid(np.linspace(0.0, 1.0, 48, dtype=np.float32), np.linspace(0.0, 1.0, 32, dtype=np.float32))
        pw = sculpted_pinion_wing(uu, vv, vanes=5, envelope_style="primary", gold_arc=True)
        self.assertTrue(is_material_bus(pw))
        self.assertEqual(pw["albedo"].shape, (32, 48, 4))
        self.assertEqual(pw["normal"].shape, (32, 48, 3))
        # Must carve clean hard alpha cutout AND paint high-metallic gold arc AND focused glowing quills!
        self.assertGreater(float((pw["alpha"] < 0.5).mean()), 0.15)
        self.assertGreater(float((pw["alpha"] > 0.5).mean()), 0.40)
        self.assertGreater(float(pw["metallic"].max()), 0.85)
        self.assertGreater(float(pw["glow_mask"].max()), 0.60)

    def test_v5_generate_iterate_simulate_animate_baked_pbr_and_plugin(self) -> None:
        import shutil
        import subprocess
        from synccraft.backends.blockbench_plugin_bridge import BlockbenchBridgeDispatcher
        from synccraft.texgraph import (
            SUBGRAPH_REGISTRY,
            anatomical_muscle_scales,
            build_arcane_ember_vfx_block,
            colossal_dragon_membrane,
            eval_anim_driver,
            export_node_schema_json,
            feedback_step_sequence,
            fluid_smoke_sim_field,
            is_material_bus,
            iterate_inline,
            particle_sim_field,
        )

        # 1. Schema v3.1 + Generate/Iterate/Simulate/Animate ops & subgraphs
        schema = export_node_schema_json()
        self.assertEqual(schema["schema_version"], "3.1")
        self.assertTrue(schema["supports_param_drivers"])
        self.assertTrue(schema["supports_generate_iterate_simulate_animate"])
        all_ops = {n["op"] for items in schema["nodes_by_category"].values() for n in items}
        for op_name in (
            "anim_driver",
            "iterate_inline",
            "feedback_loop",
            "particle_sim",
            "fluid_smoke_sim",
            "anatomical_muscle_scales",
            "colossal_dragon_membrane",
        ):
            self.assertIn(op_name, all_ops)
        self.assertIn("anatomical_muscle_scales_pbr", SUBGRAPH_REGISTRY)
        self.assertIn("colossal_dragon_membrane_pbr", SUBGRAPH_REGISTRY)

        # 2. Seamless Timeline Animation Drivers (wavetables, circular looping wiggler, eased keyframes)
        w0 = eval_anim_driver(0.0, kind="wiggler", min_val=0.2, max_val=0.9, octaves=3, seed=11)
        w1 = eval_anim_driver(1.0, kind="wiggler", min_val=0.2, max_val=0.9, octaves=3, seed=11)
        self.assertAlmostEqual(w0, w1, places=5)
        self.assertGreaterEqual(w0, 0.2 - 1e-5)
        self.assertLessEqual(w0, 0.9 + 1e-5)
        kf_mid = eval_anim_driver(0.5, kind="keyframes", keyframes=[[0.0, 0.0, "smoothstep"], [0.5, 1.0, "smoothstep"], [1.0, 0.0]])
        self.assertAlmostEqual(kf_mid, 1.0, places=5)

        # 3. Iterate Inline + Stateful Feedback Loop + Looping Particles + 2D Fluid Smoke Sim
        base = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
        it_out = iterate_inline(base, mode="warp_cascade", steps=3, strength=0.3, decay=0.65, seed=7)
        self.assertEqual(it_out.shape, (32, 32))
        fb_seq = feedback_step_sequence(lambda ph: np.roll(base, int(ph * 4), axis=1), num_frames=4, pre_roll=2, decay=0.75, drift_x_px=1.0, blur_radius=1)
        self.assertEqual(len(fb_seq), 4)
        self.assertEqual(fb_seq[0].shape, (32, 32))
        ps = particle_sim_field(32, 32, count=24, phase=0.5, trail_steps=4, seed=9)
        self.assertEqual(ps["density"].shape, (32, 32))
        self.assertEqual(ps["rgba"].shape, (32, 32, 4))
        self.assertGreater(float(ps["density"].max()), 0.25)
        fs = fluid_smoke_sim_field(32, 32, steps=8, phase=0.25, seed=5)
        self.assertEqual(fs["density"].shape, (32, 32))
        self.assertEqual(fs["temperature"].shape, (32, 32))
        self.assertGreater(float(fs["density"].max()), 0.20)

        # 4. Arcane Ember VFX 4-Pass Recipe v3.1 Driver Round-Trip + Baked PBR Format + CLI `bake`
        g_vfx = build_arcane_ember_vfx_block(size=32)
        self.assertEqual(len(g_vfx.passes), 4)
        self.assertGreaterEqual(len(g_vfx.param_drivers), 1)
        rec = g_vfx.to_recipe()
        self.assertEqual(rec["version"], "3.1")
        self.assertIn("drivers", rec)
        self.assertGreaterEqual(len(rec["drivers"]), 1)
        g_rt = TexGraph.from_recipe(rec)
        self.assertEqual(len(g_rt.param_drivers), len(g_vfx.param_drivers))

        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            pkg = g_rt.export_baked_pbr(td, stem="ember_test", preset="studio_hero", embed_maps_in_manifest=True)
            self.assertTrue((td / "ember_test_baked.png").is_file())
            self.assertTrue((td / "ember_test.synccraft.pbr.json").is_file())
            manifest = json.loads((td / "ember_test.synccraft.pbr.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "synccraft.pbr.bundle/1.0")
            self.assertIn("embedded_data_urls", manifest)
            self.assertIn("labpbr_n", manifest["embedded_data_urls"])
            self.assertIn("bedrock_mer", manifest["embedded_data_urls"])
            self.assertIn("baked", manifest["embedded_data_urls"])

            # CLI `synccraft texgraph bake` smoke test
            rc = cli_main([
                "texgraph", "bake", "preset:arcane_ember_vfx",
                "--out-dir", str(td / "cli_bake"), "--stem", "vfx_cli", "--size", "32", "--preset", "dramatic_rim", "--embed-manifest",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue((td / "cli_bake" / "vfx_cli_baked.png").is_file())
            self.assertTrue((td / "cli_bake" / "vfx_cli.synccraft.pbr.json").is_file())

        # 5. Colossal Dragon Membrane & 3D Muscle-Fitted Scales + Wing Bay Channels
        uu, vv = np.meshgrid(np.linspace(0.0, 1.0, 48, dtype=np.float32), np.linspace(0.0, 1.0, 48, dtype=np.float32))
        cm = colossal_dragon_membrane(uu, vv, tendon_ribs=3, backlit_sss=0.88)
        self.assertTrue(is_material_bus(cm))
        # Fingertip anchoring: u=0.01 and u=0.99 stay solid all the way near v=0.95 while mid-arch scoops cleanly!
        self.assertGreater(float(cm["alpha"][42, 1]), 0.5)
        self.assertGreater(float(cm["alpha"][42, 46]), 0.5)
        self.assertLess(float(cm["alpha"][46, 24]), 0.5)
        self.assertGreater(float(cm["tendon_mask"].max()), 0.70)
        self.assertGreater(float(cm["sss"].max()), 0.65)

        prog = parse_program(_CREATURE_DSL)
        scene = compile_geometry(prog)
        sem = pack_semantic_uv_atlas(prog, scene, padding=1, mirror_lr_uv="adaptive")
        atlas = {"width": sem["width"], "height": sem["height"], "faces": sem["faces"], "rgba": np.full((sem["height"], sem["width"], 4), 200, dtype=np.uint8)}
        bb = export_bbmodel(scene, name="muscle_test", atlas=atlas, program=prog)
        ctx = bake_bbmodel_maps(bb)
        self.assertEqual(ctx.wing_bay_u.shape, (ctx.height, ctx.width))
        self.assertEqual(ctx.wing_bay_mask.shape, (ctx.height, ctx.width))
        ms = anatomical_muscle_scales(ctx.world_pos_norm, ctx.world_normal, ctx.baked_ao, ctx.edge_bevel, ctx.joint_cavity, ctx.local_uv)
        self.assertTrue(is_material_bus(ms))
        self.assertEqual(ms["albedo"].shape, (ctx.height, ctx.width, 4))

        # 6. Bridge `pbr_rebuild` Endpoint
        disp = BlockbenchBridgeDispatcher()
        res = disp.dispatch("pbr_rebuild", {"preset": "arcane_ember_vfx", "size": 32, "lighting_preset": "studio_hero", "embed_maps": True})
        self.assertTrue(res["ok"])
        self.assertTrue(res["baked_data_url"].startswith("data:image/png;base64,"))
        self.assertEqual(res["manifest"]["schema"], "synccraft.pbr.bundle/1.0")

        # 7. Blockbench 5 PBR Rebuilder Plugin JavaScript Runtime Verification (`node` or macOS JavaScriptCore `jsc`)
        plugin_js = Path(__file__).resolve().parents[1] / "editors" / "blockbench-pbr-rebuilder" / "synccraft_pbr_rebuilder.js"
        self.assertTrue(plugin_js.is_file())
        js_bin = shutil.which("node") or ("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc" if Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc").is_file() else None)
        if js_bin:
            is_node = Path(js_bin).name == "node"
            harness = (
                f"var _p = {json.dumps(str(plugin_js))};\n"
                + ("var P = require(_p);\n" if is_node else "var module = { exports: {} }; load(_p); var P = module.exports;\n")
                + """
if (!P || !P.BAKE_PRESETS || !P.unpackPbrChannels || !P.bakePbrPixels || !P.createPbrRebuilderController) throw new Error("Missing exports");
var W = 4, H = 4, N = W * H;
var alb = new Uint8ClampedArray(N * 4);
var nMap = new Uint8ClampedArray(N * 4);
var sMap = new Uint8ClampedArray(N * 4);
var merMap = new Uint8ClampedArray(N * 4);
for (var i = 0; i < N; i++) {
  alb[i*4]=80; alb[i*4+1]=120; alb[i*4+2]=180; alb[i*4+3]=255;
  nMap[i*4]=145; nMap[i*4+1]=112; nMap[i*4+2]=240; nMap[i*4+3]=255;
  sMap[i*4]=210; sMap[i*4+1]=238; sMap[i*4+2]=40; sMap[i*4+3]=128;
  merMap[i*4]=200; merMap[i*4+1]=50; merMap[i*4+2]=70; merMap[i*4+3]=255;
}
var ch1 = P.unpackPbrChannels(W, H, { labpbrN: nMap, labpbrS: sMap });
if (!(ch1.metallic[0] > 0.8 && ch1.roughness[0] < 0.25 && ch1.emissive[0] > 0.4)) throw new Error("LabPBR unpack failed");
var ch2 = P.unpackPbrChannels(W, H, { bedrockMer: merMap });
if (!(ch2.metallic[0] > 0.7 && ch2.emissive[0] > 0.15 && ch2.roughness[0] < 0.35)) throw new Error("MER unpack failed");
var baked = P.bakePbrPixels(W, H, alb, { labpbrN: nMap, labpbrS: sMap }, { preset: "studio_hero" });
if (baked.length !== N * 4 || baked[3] !== 255) throw new Error("bakePbrPixels failed");
var actions = [], undoStart = 0, undoDone = 0;
function FakeAction(id, cfg) { this.id = id; this.cfg = cfg; this.deleted = false; this.delete = function(){ this.deleted = true; }; }
var fakeMat = { map: { image: { width: W, height: H }, needsUpdate: false } };
var fakeMesh = { material: fakeMat };
var fakeTex = { name: "dragon", width: W, height: H, ctx: { getImageData: function(){ return { data: new Uint8ClampedArray(alb) }; }, putImageData: function(){} }, updateChangesAfterEdit: function(){ this.edited = true; } };
var rt = {
  Action: FakeAction,
  MenuBar: { addAction: function(a){ actions.push(a); } },
  Undo: { initEdit: function(){ undoStart++; }, finishEdit: function(){ undoDone++; } },
  Blockbench: { showQuickMessage: function(){} },
  Texture: { selected: fakeTex, all: [fakeTex] },
  Cube: { all: [{ mesh: fakeMesh }] },
  Canvas: { scene: { add: function(){}, remove: function(){} } }
};
var ctrl = P.createPbrRebuilderController(rt);
ctrl.onload();
if (actions.length !== 4) throw new Error("Expected 4 Blockbench actions, got " + actions.length);
ctrl.loadManifestObject({ format: "synccraft.pbr.bundle", stem: "dragon", width: W, height: H, maps: { labpbr_n: "d_n.png" } });
ctrl.onunload();
if (!actions[0].deleted) throw new Error("onunload did not clean up actions");
"""
            )
            cmd = [js_bin, "-e", harness]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"Plugin JS runtime test failed: {r.stderr or r.stdout}")

    def test_v6_organic_painterly_dual_layer_wings_and_metrics(self) -> None:
        from synccraft.texgraph import (
            anatomical_muscle_scales,
            colossal_dragon_membrane,
            export_node_schema_json,
            measure_painterly_metrics,
            organic_domain_warp_3d,
            painterly_brush_field,
        )

        # 1. Registry & schema expose organic_domain_warp_3d and painterly_brush_field
        schema = export_node_schema_json()
        all_ops = {n["op"] for items in schema["nodes_by_category"].values() for n in items}
        self.assertIn("organic_domain_warp_3d", all_ops)
        self.assertIn("painterly_brush_field", all_ops)

        # 2. 0.5u dual-sided solid wing slab: uv_packer auto-culls thin side rims while preserving full up/down islands!
        dual_wing_dsl = """
        @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 32, size_step: 0.5, audit_profile: sculpted }
        @palette {
          obsidian: ramp(base_lch=[0.24, 0.055, 268], steps=5, hue_shift=[-18, 28])
        }
        @rig {
          root bone pelvis_root {
            pivot_local: [0, 12, 0]
            box torso_core { size: [12, 12, 18], material: obsidian }
            bone wing_l_upper joint(parent=torso_core, at=[-6, 4, 0], align=[0, 0, 0]) {
              box wing_spar { size: [30, 3, 4], material: obsidian }
              box wing_bay_1 place(base_box=wing_spar, anchor=[0, 0.2, -0.4], align=[0, 0, 0.9]) {
                size: [28, 0.5, 24], material: obsidian
              }
            }
          }
        }
        """
        prog = parse_program(dual_wing_dsl)
        scene = compile_geometry(prog)
        sem = pack_semantic_uv_atlas(prog, scene, padding=1, mirror_lr_uv="adaptive", canvas_mode="tight")
        bay_faces = sem["faces"]["wing_bay_1"]
        # Side rims of 0.5u membrane slab MUST be culled to [0,0,0,0], while BOTH up (dorsal) & down (ventral) have full area!
        for rim_f in ("north", "south", "east", "west"):
            self.assertEqual(bay_faces[rim_f], [0.0, 0.0, 0.0, 0.0], f"Expected culled thin rim on {rim_f}")
        for sheet_f in ("up", "down"):
            uv = bay_faces[sheet_f]
            self.assertGreater(abs(uv[2] - uv[0]), 10.0)
            self.assertGreater(abs(uv[3] - uv[1]), 10.0)
        # Dual-layer sheets MUST occupy distinct UV regions (not aliased onto the same pixels!)
        self.assertNotEqual(bay_faces["up"], bay_faces["down"])

        # 3. True Dual-Layer Dorsal vs Ventral `colossal_dragon_membrane`: distinct RGB palettes, 100% matched alpha!
        uu, vv = np.meshgrid(np.linspace(0.0, 1.0, 48, dtype=np.float32), np.linspace(0.0, 1.0, 48, dtype=np.float32))
        dorsal_stops = [(0.0, "#0A0C22"), (0.5, "#1E4280"), (1.0, "#C7F2FF")]
        ventral_stops = [(0.0, "#1C121F"), (0.5, "#C46C46"), (1.0, "#FFF0C8")]
        m_up = colossal_dragon_membrane(
            uu, vv, face_is_down=np.zeros_like(uu),
            dorsal_palette_stops=dorsal_stops, palette_stops=ventral_stops,
            dorsal_tendon_color="#D4A438", tendon_color="#251624",
            seed=77,
        )
        m_dn = colossal_dragon_membrane(
            uu, vv, face_is_down=np.ones_like(uu),
            dorsal_palette_stops=dorsal_stops, palette_stops=ventral_stops,
            dorsal_tendon_color="#D4A438", tendon_color="#251624",
            seed=77,
        )
        # 100.0% pixel-identical hard alpha silhouette between dorsal & ventral sheets!
        np.testing.assert_array_equal(m_up["alpha"], m_dn["alpha"])
        # Distinct dorsal vs ventral art (clearly separated mean RGB!)
        solid = m_up["alpha"] > 0.5
        rgb_diff = float(np.abs(m_up["albedo"][solid, :3] - m_dn["albedo"][solid, :3]).mean())
        self.assertGreater(rgb_diff, 0.12)
        # Ventral sheet gets stronger backlit SSS than protective dorsal sheet!
        self.assertGreater(float(m_dn["sss"][solid].mean()), float(m_up["sss"][solid].mean()))

        # 4. Organic 3D Domain Warp + Painterly Brush Field + Atmospheric Material Grade + Painterly QC Metrics
        atlas = {"width": sem["width"], "height": sem["height"], "faces": sem["faces"], "rgba": np.full((sem["height"], sem["width"], 4), 200, dtype=np.uint8)}
        bb = export_bbmodel(scene, name="organic_v6_test", atlas=atlas, program=prog)
        ctx = bake_bbmodel_maps(bb)

        w3 = organic_domain_warp_3d(ctx.world_pos_norm, scale=3.5, warp_strength=0.45, seed=19)
        self.assertEqual(set(w3.keys()), {"mottle", "warp_x", "warp_y", "flow"})
        self.assertTrue(np.isfinite(w3["mottle"]).all())

        pb = painterly_brush_field(ctx.height, ctx.width, world_pos_norm=ctx.world_pos_norm, organic_distortion=0.62, seed=23)
        self.assertEqual(set(pb.keys()), {"painterly", "brush", "mottle", "hue_drift"})
        self.assertGreater(float(pb["painterly"].std()), 0.04)

        g = TexGraph(ctx.width, ctx.height, mesh_context=ctx, seed=42, name="organic_v6_graph")
        w3_n = g.organic_domain_warp_3d(scale=3.5, warp_strength=0.45, id="w3")
        pb_n = g.painterly_brush(stroke_scale=10.0, organic_distortion=0.6, id="pb")
        ms = g.anatomical_muscle_scales(plate_cols=16, plate_rows=16, organic_distortion=0.65, softness=0.5, id="ms")
        graded = g.material_grade(
            ms,
            ao_bake=g.mesh_channel("baked_ao", id="ao"),
            atmosphere_strength=0.22,
            organic_wash_strength=0.10,
            vibrance=0.18,
            id="graded",
        )
        g.set_output_material(graded)
        ev = g.evaluate(build_pbr=True)
        self.assertEqual(ev.output.shape, (ctx.height, ctx.width, 4))
        self.assertTrue(np.isfinite(ev.output).all())
        self.assertIn(w3_n, g.nodes)
        self.assertIn(pb_n, g.nodes)

        metrics = measure_painterly_metrics(ev.pbr_maps["albedo"], mesh_ctx=ctx)
        self.assertIn("mean_lum_step", metrics)
        self.assertIn("jump_gt25_pct", metrics)
        self.assertIn("calm_interior_pct", metrics)
        self.assertGreater(metrics["mean_lum_step"], 0.5)
        self.assertLess(metrics["jump_gt25_pct"], 35.0)


if __name__ == "__main__":
    unittest.main()
