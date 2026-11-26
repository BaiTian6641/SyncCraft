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


if __name__ == "__main__":
    unittest.main()
