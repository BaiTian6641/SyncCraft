"""Comprehensive verification suite for ADR 0013:
1. Additive clip `layer` & stance-touchdown `effect` statements (grammar, parser, cycle check, `< 0.02u` slip, Bedrock/GeckoLib/.bbmodel export)
2. Phase 3 painted texture `preserve_and_layer_painted_texture`, OKLCH variants, and `rebake_selected_faces`
3. Inverse gait fitting (`invert_clip_to_gait` on both root-motion & in-place treadmill clips), `search_or_repair_gait`, and `retarget_clip_across_models`
4. 5-task proposer supervision dataset, `body_plan_group` + `source_author` split isolation, and Gate D baseline harness (`proposer_tasks.py`)
5. Blockbench 5 plugin (`editors/blockbench-synccraft/synccraft_assistant.js`) + live `127.0.0.1` HTTP bridge (`blockbench_plugin_bridge.py`)
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request

from lark import Lark
import numpy as np

from synccraft.backends import (
    export_bbmodel,
    export_bedrock,
    export_geckolib,
    make_blockbench_http_server,
)
from synccraft.dsl import (
    DSLValidationError,
    GaitDecl,
    format_program,
    parse_program,
)
from synccraft.evaluation.budget import CostLedger
from synccraft.kinematics import (
    analyze_morphology,
    audit_behavior_clip,
    compile_geometry,
    invert_clip_to_gait,
    retarget_clip_across_models,
    search_or_repair_gait,
)
from synccraft.shading import (
    preserve_and_layer_painted_texture,
    rebake_selected_faces,
    rgba_to_data_url,
)
from synccraft.shading.uv_packer import build_atlas
from synccraft.world import (
    PROPOSER_TASK_TYPES,
    build_proposer_supervision_dataset,
    decompile_bbmodel_animations_to_behavior,
    evaluate_gate_d_proposer,
    ingest_artist_project_to_program,
    style_phrase_to_gait_preset,
    verify_body_plan_and_author_split_isolation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = REPO_ROOT / "src" / "synccraft" / "dsl" / "grammar.lark"
PLUGIN_JS_PATH = REPO_ROOT / "editors" / "blockbench-synccraft" / "synccraft_assistant.js"
JSC_PATH = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")


def _fox_quadruped_dsl(extra_behavior: str = "") -> str:
    beh_section = f"@behavior {{\n{extra_behavior}\n}}" if extra_behavior else ""
    return f"""
@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  coat_orange: ramp(base_lch=[0.58, 0.14, 45], steps=5, hue_shift=[-16, 16])
  paw_dark: ramp(base_lch=[0.26, 0.04, 255], steps=4, hue_shift=[-10, 10])
  amber_glow: ramp(base_lch=[0.84, 0.16, 75], steps=3, emissive=true)
}}
@rig {{
  root bone torso {{
    pivot_local: [0, 8.35, 0]
    box chest {{ size: [4, 4, 6], material: coat_orange }}
    bone lumbar attach(parent_box=chest, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.25) {{
      joint_limits: [[-18, 18], [-22, 22], [-12, 12]]
      box hips {{ size: [3.5, 3.5, 5], material: coat_orange }}
      bone hind_leg_l attach(parent_box=hips, anchor=[-1, -0.65, -0.5], align=[1, 1, 0]) mirror_x(hind_leg_r) {{
        joint_limits: [[-60, 60], [-25, 25], [-25, 25]]
        box femur {{ size: [1.5, 3.5, 2], material: coat_orange }}
        bone shin attach(parent_box=femur, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          rest_euler: [10, 0, 0]
          joint_limits: [[-55, 55], [-20, 20], [-20, 20]]
          box tibia {{ size: [1.5, 3.5, 1.5], material: paw_dark }}
        }}
      }}
      bone tail_1 attach(parent_box=hips, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.25) {{
        joint_limits: [[-20, 20], [-28, 28], [-15, 15]]
        box caudal_1 {{ size: [1.5, 1.5, 4.5], material: coat_orange }}
      }}
    }}
    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.65, 0.6], align=[1, 1, 0]) mirror_x(fore_leg_r) {{
      joint_limits: [[-60, 60], [-25, 25], [-25, 25]]
      box humerus {{ size: [1.5, 3.5, 2], material: coat_orange }}
      bone forearm attach(parent_box=humerus, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        rest_euler: [-10, 0, 0]
        joint_limits: [[-55, 55], [-20, 20], [-20, 20]]
        box radius {{ size: [1.5, 3.5, 1.5], material: paw_dark }}
      }}
    }}
    bone neck attach(parent_box=chest, anchor=[0, 0.35, 1], align=[0, 0, -1], seat=0.25) {{
      rest_euler: [-15, 0, 0]
      joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
      box neck_seg {{ size: [2, 2, 3.5], material: coat_orange }}
      bone skull attach(parent_box=neck_seg, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.2) {{
        rest_euler: [12, 0, 0]
        joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
        box snout {{ size: [2.5, 2, 3.5], material: coat_orange }}
      }}
    }}
  }}
}}
@materials {{
  coat_orange: process(family=patterned_skin, coat=mottled, wear=0.45)
  paw_dark: process(family=keratin_scales, wear=0.35)
}}
@style {{
  target_encoding: vanilla,
  shade: banded(steps=4, direction=down, cluster=diagonal),
  glow_bleed: 1
}}
@stamps {{
  stamp eye on snout.east {{ at: [0.35, 0.42], size: [0.22, 0.18], color: amber_glow, emissive: 0.9, mirror: true }}
}}
{beh_section}
"""


class TestADR0013Pipeline(unittest.TestCase):
    def test_additive_clip_layers_and_touchdown_effects(self) -> None:
        dsl = _fox_quadruped_dsl(
            """
  clip breathe {
    duration_s: 2.4
    track lumbar: wave(axis=x, amp_deg=2.2, freq_hz=0.416667)
    track neck: wave(axis=x, amp_deg=-1.8, freq_hz=0.416667, phase_deg=35)
  }
  clip prowl {
    duration_s: 1.6
    gait stalk_gait {
      pattern: lateral_sequence,
      froude: 0.16,
      duty: 0.68,
      lift_ratio: 0.11
    }
    layer breathe: weight=0.85
    effect touchdown: { sound: "mob.fox.step", particle: "synccraft:paw_dust" }
    effect 0.40: { instruction: "ear_twitch_cue" }
  }
"""
        )
        lark_parser = Lark(GRAMMAR_PATH.read_text(encoding="utf-8"), start="start")
        lark_parser.parse(dsl)

        prog = parse_program(dsl)
        formatted = format_program(prog)
        lark_parser.parse(formatted)
        roundtrip = parse_program(formatted)
        prowl_clip = [c for c in roundtrip.behavior.clips if c.name == "prowl"][0]
        self.assertEqual(len(prowl_clip.layers), 1)
        self.assertEqual(prowl_clip.layers[0].source_clip, "breathe")
        self.assertAlmostEqual(prowl_clip.layers[0].weight, 0.85, places=4)
        self.assertEqual(len(prowl_clip.effects), 2)

        # Audit must pass with `< 0.02u` stance slip despite active breathing spine pitch!
        aud = audit_behavior_clip(roundtrip, "prowl", num_frames=16)
        self.assertTrue(aud["passed"], msg=f"Layered gait audit failed: {aud}")
        self.assertLessEqual(aud["max_stance_slip_units"], 0.02)

        # Cyclic `layer` graph must raise `DSLValidationError`
        cyclic_dsl = _fox_quadruped_dsl(
            """
  clip a { duration_s: 1.0 layer b: weight=0.5 }
  clip b { duration_s: 1.0 layer a: weight=0.5 }
"""
        )
        with self.assertRaises(DSLValidationError):
            parse_program(cyclic_dsl)

        # Native effect export across Bedrock, GeckoLib, and Blockbench `.bbmodel`
        scene = compile_geometry(roundtrip)
        atlas, _ = build_atlas(roundtrip, scene)
        bed = export_bedrock(scene, program=roundtrip, identifier="geometry.synccraft.fox", atlas=atlas)
        prowl_bed = bed["animations"]["animations"]["animation.fox.prowl"]
        self.assertIn("sound_effects", prowl_bed)
        self.assertIn("particle_effects", prowl_bed)
        self.assertIn("timeline", prowl_bed)
        self.assertGreaterEqual(len(prowl_bed["sound_effects"]), 4)

        gecko_prog = replace(roundtrip, meta=replace(roundtrip.meta, target_profile="geckolib"))
        gecko_scene = compile_geometry(gecko_prog)
        gecko = export_geckolib(gecko_scene, program=gecko_prog, identifier="geometry.synccraft.fox", atlas=atlas)
        prowl_gecko = gecko["animations"]["animations"]["animation.fox.prowl"]
        self.assertIn("sound_effects", prowl_gecko)

        bb_prog = replace(roundtrip, meta=replace(roundtrip.meta, target_profile="bbmodel_5"))
        bb_scene = compile_geometry(bb_prog)
        bb = export_bbmodel(bb_scene, name="fox", atlas=atlas, program=bb_prog)
        prowl_bb = [a for a in bb["animations"] if a["name"] == "prowl"][0]
        self.assertIn("effects", prowl_bb["animators"])
        fx_kfs = prowl_bb["animators"]["effects"]["keyframes"]
        channels = {kf["channel"] for kf in fx_kfs}
        self.assertTrue({"sound", "particle", "timeline"}.issubset(channels))

    def test_phase3_preserve_and_layer_variants_and_selective_face_rebake(self) -> None:
        prog = parse_program(_fox_quadruped_dsl())
        scene = compile_geometry(prog)
        atlas, _ = build_atlas(prog, scene)
        w, h = int(atlas["width"]), int(atlas["height"])

        # Construct a synthetic hand-painted RGBA sheet with:
        # - 1-px dark pixel-art outlines (`RGB = (18, 18, 22), A = 255`)
        # - Saturated chromatic body pixels
        # - Binary alpha cutout region (`A = 0` outside faces)
        # - One intentionally blank unpainted face (`hips.up` set to `A = 0`)
        painted = np.zeros((h, w, 4), dtype=np.uint8)
        for box_name, f_dict in atlas["faces"].items():
            for face_name, rect in f_dict.items():
                x0, y0, x1, y1 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
                if box_name == "hips" and face_name == "up":
                    continue  # Leave unpainted (`A = 0`) to test `fill_blank_faces_only`!
                if x1 <= x0 or y1 <= y0:
                    continue
                painted[y0:y1, x0:x1] = [196, 98, 44, 255]
                painted[y0, x0:x1] = [18, 18, 22, 255]  # Dark achromatic outline row
        painted_snapshot = painted.copy()

        res = preserve_and_layer_painted_texture(
            prog,
            scene,
            atlas,
            painted,
            wear_strength=0.45,
            glow_strength=0.5,
            variant_hue_shift_deg=110.0,
            variant_chroma_scale=1.15,
        )
        # 1. Original hand-painted buffer & returned `original_rgba` are 100% untouched (`0` pixels modified)
        self.assertEqual(res["original_pixels_changed"], 0)
        self.assertTrue(np.array_equal(painted, painted_snapshot))
        self.assertTrue(np.array_equal(res["original_rgba"], painted_snapshot))

        # 2. Blank `hips.up` face was filled ONLY on `A == 0` pixels
        blank_filled_px = int(np.sum(res["blank_fill_rgba"][:, :, 3] > 0))
        self.assertGreater(blank_filled_px, 0)
        already_painted_mask = painted_snapshot[:, :, 3] > 0
        self.assertEqual(int(np.sum(res["blank_fill_rgba"][:, :, 3][already_painted_mask])), 0)

        # 3. Separate non-destructive wear & glow layers exist
        self.assertGreater(int(np.sum(res["wear_layer_rgba"][:, :, 3] > 0)), 0)
        self.assertGreater(int(np.sum(res["glow_layer_rgba"][:, :, 3] > 0)), 0)
        self.assertGreaterEqual(res["palette_profile"]["painted_face_count"], 1)

        # 4. Perceptual OKLCH variant rotates chromatic pixels while preserving dark outlines & binary alpha cutouts
        var_rgba = res["variant_rgba"]
        self.assertIsNotNone(var_rgba)
        self.assertTrue(np.array_equal(var_rgba[:, :, 3], painted_snapshot[:, :, 3]))
        outline_mask = (
            (painted_snapshot[:, :, 0] == 18)
            & (painted_snapshot[:, :, 1] == 18)
            & (painted_snapshot[:, :, 2] == 22)
        )
        self.assertTrue(np.all(np.abs(var_rgba[outline_mask, :3].astype(int) - painted_snapshot[outline_mask, :3].astype(int)) <= 2))
        body_mask = (painted_snapshot[:, :, 0] == 196) & (painted_snapshot[:, :, 3] == 255)
        self.assertGreater(float(np.mean(np.abs(var_rgba[body_mask, :3].astype(float) - painted_snapshot[body_mask, :3].astype(float)))), 25.0)

        # 5. Selective single-face rebake modifies ONLY `chest.up` (`0` non-target pixels touched!)
        rebaked = rebake_selected_faces(
            prog,
            atlas,
            painted_snapshot,
            [("chest", "up")],
        )
        self.assertEqual(rebaked["updated_face_count"], 1)
        self.assertEqual(rebaked["non_target_pixels_changed"], 0)

    def test_inverse_gait_fitting_treadmill_and_search_and_retargeting(self) -> None:
        # A. Invert a root-motion diagonal-pair trot clip
        trot_dsl = _fox_quadruped_dsl(
            """
  clip trot {
    duration_s: 1.2
    gait fox_trot {
      pattern: diagonal_pairs,
      froude: 0.38,
      duty: 0.54,
      export_root_motion: true
    }
  }
"""
        )
        trot_prog = parse_program(trot_dsl)
        inv_rm = invert_clip_to_gait(trot_prog, "trot")
        self.assertFalse(inv_rm.was_in_place_treadmill)
        self.assertEqual(inv_rm.gait.pattern, "diagonal_pairs")
        self.assertAlmostEqual(inv_rm.gait.duty, 0.54, delta=0.14)
        self.assertGreater(inv_rm.stride_length_units, 2.0)

        # B. Invert an IN-PLACE treadmill keyframed walk clip (`0` root translation!)
        treadmill_dsl = _fox_quadruped_dsl(
            """
  clip artist_treadmill_walk {
    duration_s: 1.6
    key fore_leg_l.rotation { 0.0: [-24, 0, 0], 0.4: [0, 0, 0], 0.8: [24, 0, 0], 1.2: [0, 0, 0], 1.6: [-24, 0, 0] }
    key hind_leg_r.rotation { 0.0: [-24, 0, 0], 0.4: [0, 0, 0], 0.8: [24, 0, 0], 1.2: [0, 0, 0], 1.6: [-24, 0, 0] }
    key fore_leg_r.rotation { 0.0: [24, 0, 0], 0.4: [0, 0, 0], 0.8: [-24, 0, 0], 1.2: [0, 0, 0], 1.6: [24, 0, 0] }
    key hind_leg_l.rotation { 0.0: [24, 0, 0], 0.4: [0, 0, 0], 0.8: [-24, 0, 0], 1.2: [0, 0, 0], 1.6: [24, 0, 0] }
  }
"""
        )
        tm_prog = parse_program(treadmill_dsl)
        inv_tm = invert_clip_to_gait(tm_prog, "artist_treadmill_walk")
        self.assertTrue(inv_tm.was_in_place_treadmill)
        self.assertEqual(inv_tm.gait.pattern, "diagonal_pairs")
        self.assertGreater(inv_tm.stride_length_units, 1.5)
        self.assertGreater(inv_tm.gait.froude, 0.03)

        # C. Bounded deterministic gait repair (`search_or_repair_gait`) + `CostLedger`
        overdriven = GaitDecl(
            name="broken_gait",
            pattern="diagonal_pairs",
            froude=3.2,
            duty=0.32,
            lift_ratio=0.38,
            heave_ratio=0.16,
            pitch_deg=13.0,
            spine_wave_deg=20.0,
            snap=0.95,
        )
        ledger = CostLedger()
        repaired = search_or_repair_gait(
            trot_prog,
            overdriven,
            duration_s=1.2,
            clip_name="repaired_trot",
            max_evals=8,
            ledger=ledger,
        )
        self.assertTrue(repaired.passed, msg=f"Bounded gait search failed to repair: {repaired.audit_report}")
        self.assertGreaterEqual(ledger.compiler_calls, 1)
        self.assertEqual(ledger.initial_audit_calls, 1)

        # D. Cross-species retargeting (`fox` quadruped -> `raptor` biped & `beetle` hexapod)
        dataset = build_proposer_supervision_dataset()
        raptor_rec = [r for r in dataset if r.asset_id == "crest_raptor_holdout"][0]
        beetle_rec = [r for r in dataset if r.asset_id == "carapace_beetle_holdout"][0]
        raptor_prog = parse_program(raptor_rec.dsl_source)
        beetle_prog = parse_program(beetle_rec.dsl_source)

        ret_biped = retarget_clip_across_models(tm_prog, "artist_treadmill_walk", raptor_prog, new_clip_name="raptor_trot")
        self.assertTrue(ret_biped.passed)
        self.assertEqual(ret_biped.gait.pattern, "biped_alternate")

        ret_hex = retarget_clip_across_models(trot_prog, "trot", beetle_prog, new_clip_name="beetle_march")
        self.assertTrue(ret_hex.passed)
        self.assertEqual(ret_hex.gait.pattern, "hexapod_tripod")

    def test_five_proposer_tasks_split_isolation_and_gate_d_harness(self) -> None:
        records = build_proposer_supervision_dataset()
        present_tasks = {r.task_type for r in records}
        self.assertEqual(present_tasks, set(PROPOSER_TASK_TYPES))
        self.assertTrue(all(r.verified_by_solver for r in records))

        iso = verify_body_plan_and_author_split_isolation(records)
        self.assertTrue(iso["passed"])

        # Injecting body_plan_group leakage into `holdout_gate_d` MUST raise `ValueError`
        leaked = list(records)
        ho_idx = next(i for i, r in enumerate(leaked) if r.split == "holdout_gate_d")
        leaked[ho_idx] = replace(leaked[ho_idx], body_plan_group="canine_quadruped")
        with self.assertRaises(ValueError):
            verify_body_plan_and_author_split_isolation(leaked)

        # Run Gate D benchmark harness against an oracle-guided proposer vs fixed presets & search
        holdout_targets = {
            (r.prompt_payload["style_phrase"], len(r.prompt_payload["morphology"]["limbs"])): r.target_payload
            for r in records
            if r.split == "holdout_gate_d" and r.task_type == "propose_gait_from_style"
        }

        def _mock_proposer(prompt_payload: dict) -> dict:
            key = (prompt_payload["style_phrase"], len(prompt_payload["morphology"]["limbs"]))
            return dict(holdout_targets[key])

        report = evaluate_gate_d_proposer(records, _mock_proposer, max_search_evals=6)
        self.assertEqual(report["proposer"]["first_try_pass_rate"], 1.0)
        self.assertGreaterEqual(report["preset_plus_deterministic_search"]["post_search_pass_rate"], 0.75)
        self.assertGreater(report["proposer"]["ledger"]["accepted_tokens"], 0)

    def test_blockbench_plugin_live_localhost_bridge_and_js_controller(self) -> None:
        node_bin = shutil.which("node")
        has_jsc = JSC_PATH.exists()
        self.assertTrue(node_bin or has_jsc, "Expected either node or macOS JavaScriptCore (jsc) runtime")

        # Prepare a painted fox `.bbmodel` and a biped target `.bbmodel`
        fox_base = parse_program(_fox_quadruped_dsl())
        fox_prog = replace(fox_base, meta=replace(fox_base.meta, target_profile="bbmodel_5"))
        fox_scene = compile_geometry(fox_prog)
        fox_atlas, _ = build_atlas(fox_prog, fox_scene)
        fox_bbmodel = export_bbmodel(fox_scene, name="painted_fox", atlas=fox_atlas, program=fox_prog)

        records = build_proposer_supervision_dataset()
        raptor_dsl = [r for r in records if r.asset_id == "crest_raptor_holdout"][0].dsl_source
        raptor_raw = parse_program(raptor_dsl)
        raptor_prog = replace(raptor_raw, meta=replace(raptor_raw.meta, target_profile="bbmodel_5"))
        raptor_scene = compile_geometry(raptor_prog)
        raptor_atlas, _ = build_atlas(raptor_prog, raptor_scene)
        raptor_bbmodel = export_bbmodel(raptor_scene, name="crest_raptor", atlas=raptor_atlas, program=raptor_prog)

        def _http_call(base_url: str, endpoint: str, body: dict | None = None) -> dict:
            url = f"{base_url}/{endpoint.lstrip('/')}"
            if body is None:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))

        server = make_blockbench_http_server("127.0.0.1", 0)
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        try:
            # 1. Exercise ALL 7 HTTP endpoints live over `127.0.0.1`
            health = _http_call(base_url, "health")
            self.assertTrue(health["ok"])
            self.assertIn("generate_locomotion", health["actions"])

            live_analyze = _http_call(base_url, "analyze", {"bbmodel": fox_bbmodel})
            live_loc = _http_call(
                base_url,
                "generate_locomotion",
                {
                    "bbmodel": fox_bbmodel,
                    "style_phrase": "snappy alert trot",
                    "clip_name": "walk",
                    "duration_s": 1.4,
                    "footstep_sound": "mob.fox.step",
                },
            )
            self.assertTrue(live_loc["verified"])

            fox_with_walk_doc = dict(fox_bbmodel)
            fox_with_walk_doc["animations"] = [live_loc["animation"]]

            live_sec = _http_call(
                base_url,
                "secondary_motion",
                {
                    "bbmodel": fox_with_walk_doc,
                    "kind": "breathe",
                    "clip_name": "breathe_idle",
                    "base_clip_name": "walk",
                    "layer_weight": 0.65,
                },
            )
            self.assertTrue(live_sec["verified"])

            fox_with_layered_walk_doc = dict(fox_bbmodel)
            fox_with_layered_walk_doc["animations"] = [live_sec["animation"]]

            live_check = _http_call(base_url, "check", {"bbmodel": fox_with_layered_walk_doc, "clip_name": "walk"})
            self.assertTrue(live_check["passed"])

            live_ret = _http_call(
                base_url,
                "retarget",
                {
                    "source_bbmodel": fox_with_layered_walk_doc,
                    "clip_name": "walk",
                    "target_bbmodel": raptor_bbmodel,
                    "new_clip_name": "raptor_walk",
                },
            )
            self.assertTrue(live_ret["verified"])

            live_tex = _http_call(
                base_url,
                "texture_pass",
                {
                    "bbmodel": fox_bbmodel,
                    "mode": "preserve_and_layer",
                    "wear_amount": 0.4,
                    "hue_rotate_deg": 60.0,
                },
            )
            self.assertEqual(live_tex["original_pixels_changed"], 0)

            # 2. Execute `editors/blockbench-synccraft/synccraft_assistant.js` in native JS (`jsc` or `node`)
            bridge_fixtures = {
                "analyze": live_analyze,
                "generate_locomotion": live_loc,
                "secondary_motion": live_sec,
                "check": live_check,
                "retarget": live_ret,
                "texture_pass": live_tex,
            }
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                data_file = tmp_dir / "fixtures.json"
                data_file.write_text(
                    json.dumps(
                        {
                            "fox_bbmodel": fox_bbmodel,
                            "raptor_bbmodel": raptor_bbmodel,
                            "bridge_fixtures": bridge_fixtures,
                        }
                    ),
                    encoding="utf-8",
                )
                harness_js = tmp_dir / "run_plugin_test.js"
                plugin_path_json = json.dumps(str(PLUGIN_JS_PATH))
                data_path_json = json.dumps(str(data_file))

                if has_jsc and not node_bin:
                    preamble = f"""
const mod = {{ exports: {{}} }};
(new Function("module", "exports", readFile({plugin_path_json})))(mod, mod.exports);
const plugin = mod.exports;
const payload = JSON.parse(readFile({data_path_json}));
const outPrint = print;
"""
                    epilogue = "\nif (typeof drainMicrotasks === 'function') drainMicrotasks();\n"
                    cmd = [str(JSC_PATH), str(harness_js)]
                else:
                    preamble = f"""
const fs = require("fs");
const plugin = require({plugin_path_json});
const payload = JSON.parse(fs.readFileSync({data_path_json}, "utf8"));
const outPrint = (s) => process.stdout.write(s + "\\n");
"""
                    epilogue = "\n"
                    cmd = [str(node_bin), str(harness_js)]

                js_body = """
const undoSteps = [];
const messages = [];
const liveDoc = JSON.parse(JSON.stringify(payload.fox_bbmodel));

class MockAction {
  constructor(id, opts) { this.id = id; this.opts = opts; this.deleted = false; }
  delete() { this.deleted = true; }
}
class MockBoneAnimator {
  constructor(uuid, anim, name) { this.uuid = uuid; this.name = name; this.type = "bone"; this.keyframes = []; }
  addKeyframe(kf) { this.keyframes.push(JSON.parse(JSON.stringify(kf))); }
}
class MockEffectAnimator {
  constructor(anim) { this.name = "Effects"; this.type = "effect"; this.keyframes = []; }
  addKeyframe(kf) { this.keyframes.push(JSON.parse(JSON.stringify(kf))); }
}
class MockAnimation {
  constructor(opts) {
    this.name = opts.name;
    this.length = opts.length;
    this.loop = opts.loop || "loop";
    this.snapping = opts.snapping || 24;
    this.animators = {};
  }
  add() {
    if (!MockAnimation.all.includes(this)) MockAnimation.all.push(this);
  }
  select() { MockAnimation.selected = this; }
}
MockAnimation.all = [];
MockAnimation.selected = null;

class MockTexture {
  constructor(opts) { this.name = opts.name; this.width = opts.width; this.height = opts.height; this.source = ""; }
  fromDataURL(src) { this.source = src; }
  add() { MockTexture.all.push(this); }
}
MockTexture.all = (liveDoc.textures || []).map((t) => Object.assign(new MockTexture(t), { source: t.source }));
const origTex0Source = MockTexture.all[0].source;

const mockEnv = {
  Action: MockAction,
  MenuBar: { addAction() {} },
  Blockbench: { showMessageBox(m) { messages.push(m); } },
  Undo: {
    _open: 0,
    initEdit() { this._open += 1; },
    finishEdit(label) {
      if (this._open !== 1) throw new Error("Nested or unbalanced Undo step: " + this._open);
      this._open = 0;
      undoSteps.push(label);
    }
  },
  Animation: MockAnimation,
  BoneAnimator: MockBoneAnimator,
  EffectAnimator: MockEffectAnimator,
  Texture: MockTexture,
  Codecs: {
    project: {
      compile() {
        const copy = JSON.parse(JSON.stringify(liveDoc));
        copy.animations = MockAnimation.all.map((a) => ({
          name: a.name,
          length: a.length,
          loop: a.loop,
          snapping: a.snapping,
          animators: a.animators
        }));
        return copy;
      }
    }
  }
};

async function liveBackedFetch(url) {
  const endpoint = url.split("/").pop();
  const respData = payload.bridge_fixtures[endpoint];
  return {
    ok: true,
    status: 200,
    async json() { return JSON.parse(JSON.stringify(respData)); }
  };
}

let jsErr = null;
(async () => {
  const ctrl = plugin.createSyncCraftPluginController(mockEnv, {
    bridgeUrl: "http://127.0.0.1:18765",
    fetchImpl: liveBackedFetch
  });
  ctrl.onload();
  if (ctrl.registeredActions.length !== 6) throw new Error("Expected 6 actions");

  const ana = await ctrl.handlers.analyze();
  const loc = await ctrl.handlers.generateLocomotion({
    style_phrase: "snappy alert trot",
    clip_name: "walk",
    duration_s: 1.4,
    footstep_sound: "mob.fox.step"
  });
  const sec = await ctrl.handlers.addSecondaryMotion({
    kind: "breathe",
    clip_name: "breathe_idle",
    base_clip_name: "walk",
    layer_weight: 0.65
  });
  const chk = await ctrl.handlers.check({ clip_name: "walk" });

  const foxWithWalk = mockEnv.Codecs.project.compile();
  const ret = await ctrl.handlers.retarget({
    source_bbmodel: foxWithWalk,
    clip_name: "walk",
    target_bbmodel: payload.raptor_bbmodel,
    new_clip_name: "raptor_walk"
  });
  const tex = await ctrl.handlers.texturePass({
    mode: "preserve_and_layer",
    wear_amount: 0.4,
    hue_rotate_deg: 60.0
  });

  // Verify hard refusal on `verified: false` (0 Undo steps added!)
  const undoBeforeRefusal = undoSteps.length;
  const refused = plugin.applyVerifiedAnimationToBlockbench({ verified: false, animation: { name: "bad" } }, mockEnv);

  ctrl.onunload();
  if (ctrl.registeredActions.length !== 0) throw new Error("Expected onunload to clean up all registeredActions");

  outPrint(JSON.stringify({
    analyze_limbs: ana.morphology.limbs.length,
    loc_verified: loc.response.verified,
    loc_applied: loc.outcome.applied,
    sec_verified: sec.response.verified,
    check_passed: chk.passed,
    ret_verified: ret.response.verified,
    tex_orig_changed: tex.response.original_pixels_changed,
    orig_tex_preserved: MockTexture.all[0].source === origTex0Source,
    added_tex_count: tex.outcome.addedTextures.length,
    undo_steps: undoSteps,
    refusal_blocked: refused.applied === false && undoSteps.length === undoBeforeRefusal,
    exported_fox_bbmodel: foxWithWalk
  }));
})().catch((e) => { jsErr = e; });
"""
                harness_js.write_text(
                    preamble + js_body + epilogue + "\nif (jsErr) { throw jsErr; }\n",
                    encoding="utf-8",
                )
                proc = subprocess.run(cmd, text=True, capture_output=True, timeout=30, check=False)
                self.assertEqual(proc.returncode, 0, msg=f"JS plugin test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
                out = json.loads(proc.stdout.strip().splitlines()[-1])

            self.assertEqual(out["analyze_limbs"], 4)
            self.assertTrue(out["loc_verified"])
            self.assertTrue(out["loc_applied"])
            self.assertTrue(out["sec_verified"])
            self.assertTrue(out["check_passed"])
            self.assertTrue(out["ret_verified"])
            self.assertEqual(out["tex_orig_changed"], 0)
            self.assertTrue(out["orig_tex_preserved"])
            self.assertGreaterEqual(out["added_tex_count"], 2)
            self.assertTrue(out["refusal_blocked"])
            self.assertEqual(
                out["undo_steps"],
                [
                    "SyncCraft: Generate Locomotion",
                    "SyncCraft: Add Secondary Motion",
                    "SyncCraft: Retarget Animation",
                    "SyncCraft: Non-Destructive Texture Pass",
                ],
            )

            # Roundtrip the `.bbmodel` compiled out of our JS Blockbench runtime back into SyncCraft & re-audit!
            reimported = ingest_artist_project_to_program(out["exported_fox_bbmodel"], format_kind="bbmodel")
            self.assertIsNotNone(reimported.behavior)
            reaud = audit_behavior_clip(reimported, "walk", num_frames=16)
            self.assertTrue(reaud["passed"], msg=f"Roundtripped Blockbench animation failed re-audit: {reaud}")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
