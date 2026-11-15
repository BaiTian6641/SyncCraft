"""Comprehensive tests for P1-03d/e world transitions, 7 interventions, inverse inference, and datasets."""

from __future__ import annotations

import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

import numpy as np
from PIL import Image

from synccraft.backends import export_bbmodel, export_java, export_labpbr_1_3
from synccraft.data_gen.ik_motion_templates import generate_ik_clip, solve_two_bone_ik
from synccraft.dsl import Attach, Bone, Box, Locator, Meta, Palette, Place, Program, UnsupportedFeature, format_program, parse_program, validate_program
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.shading import compile_vanilla_atlas
from synccraft.surface import MaterialProgram, WorldState, build_surface_domain, initialize_surface_state
from synccraft.world import (
    LOCAL_WORKSPACE_LICENSE_POLICY,
    WORLD_CAPABILITY_LEVELS,
    ApplyRubbingAction,
    AssetProvenance,
    EditGeometryAction,
    EditMaterialAction,
    JointTrajectoryAction,
    LicenseGateError,
    NoOpEnvironmentAction,
    RepackUVAction,
    SetLightingAction,
    SetPoseAction,
    SetTexelDensityAction,
    TwoBoneIKAction,
    action_from_dict,
    analyze_artist_project,
    assign_multi_slot_materials_to_program,
    audit_converted_asset_visual_identity,
    build_articulated_mechanism_world,
    build_intervention_trajectory_dataset,
    build_unlabeled_workspace_provenance,
    build_world_capability_ladder_dataset,
    compute_ast_edit_size,
    decode_record_observation_image,
    evaluate_matched_budget_world_baselines,
    evaluate_world_capability_benchmark,
    ingest_artist_project_record,
    ingest_external_asset,
    ingest_gltf_archive,
    run_looped_artist_enhancement,
    run_seven_interventions_suite,
    score_world_capability_completion,
    step_world,
)


class TestWorldTransitionsAndInterventions(unittest.TestCase):
    def test_all_seven_causal_interventions_pass(self) -> None:
        suite = run_seven_interventions_suite()
        self.assertTrue(suite["all_passed"], msg=json.dumps(suite, indent=2))
        self.assertEqual(suite["passed_count"], 7)

    def test_two_bone_ik_exactness_and_clamping(self) -> None:
        ws = build_articulated_mechanism_world(seed=101)
        # Reachable target for upper_arm (len 6) + fore_arm (len 5) starting from shoulder at (3, 12, 0)
        res_ok = solve_two_bone_ik(
            ws.program,
            "upper_arm",
            "fore_arm",
            effector_local=(5.0, 0.0, 0.0),
            target_world=(11.5, 9.5, 1.5),
            hinge_axis="z",
        )
        self.assertTrue(res_ok.reachable)
        self.assertLess(res_ok.position_error, 1e-3)

        # Unreachable far target clamps cleanly
        res_far = solve_two_bone_ik(
            ws.program,
            "upper_arm",
            "fore_arm",
            effector_local=(5.0, 0.0, 0.0),
            target_world=(50.0, 12.0, 0.0),
            hinge_axis="z",
        )
        self.assertFalse(res_far.reachable)
        self.assertLess(res_far.position_error, 1e-2)

        clip = generate_ik_clip(
            ws.program,
            "upper_arm",
            "fore_arm",
            (5.0, 0.0, 0.0),
            [(0.0, (12.0, 11.0, 0.0)), (1.0, (10.5, 8.5, 1.0))],
            hinge_axis="z",
            num_frames=8,
        )
        self.assertEqual(len(clip["frames"]), 8)
        self.assertTrue(clip["all_waypoints_reachable"])

    def test_action_serialization_roundtrip_and_state_immutability(self) -> None:
        ws = build_articulated_mechanism_world(seed=101)
        orig_sha = ws.surface_state.state_sha256()

        actions = [
            NoOpEnvironmentAction(evolve_rd=False),
            SetPoseAction({"upper_arm": (0.0, 10.0, 15.0)}, substeps=4),
            JointTrajectoryAction(((0.0, {"upper_arm": (0.0, 0.0, 0.0)}), (1.0, {"upper_arm": (0.0, 0.0, 20.0)})), num_samples=6),
            TwoBoneIKAction("upper_arm", "fore_arm", (5.0, 0.0, 0.0), (11.5, 10.0, 0.0), hinge_axis="z", substeps=4),
            ApplyRubbingAction("housing", "north", 3.0, 2.0, radius=1.2),
            EditMaterialAction(ws.material_program.materials["painted_iron"]),
            EditGeometryAction(ws.program),
            SetTexelDensityAction(32),
            SetLightingAction((0.2, 0.9, 0.3), ambient_intensity=0.35, sun_intensity=0.85),
            RepackUVAction(max_width=256, padding=2),
        ]
        for act in actions:
            rebuilt = action_from_dict(act.to_dict())
            self.assertEqual(act.to_dict(), rebuilt.to_dict())
            _ = step_world(ws, rebuilt, dt=0.2)
            # Input state must never be mutated in place
            self.assertEqual(ws.surface_state.state_sha256(), orig_sha)

    def test_swept_sat_detects_mid_trajectory_collision_tunneling(self) -> None:
        # Arm starts at -35 deg (below obstacle) and sweeps to +35 deg (above obstacle), passing right through obstacle at 0 deg!
        prog = validate_program(
            Program(
                Meta("block_multipart", "bbmodel_5", 16),
                [Palette("m", (0.6, 0.05, 200.0), 4)],
                Bone(
                    "root",
                    pivot_local=(0.0, 8.0, 0.0),
                    items=[
                        Box("hub", (4, 4, 4), "m"),
                        Box("obstacle", (2, 2, 2), "m", place=Place("hub", (1, 0, 0), (-1, 0, 0), offset=(4.0, 0.0, 0.0))),
                        Bone(
                            "swing",
                            attach=Attach("hub", (1, 0, 0), (-1, 0, 0)),
                            rest_euler=(0.0, 0.0, -40.0),
                            items=[Box("blade", (8, 2, 2), "m")],
                        ),
                    ],
                ),
            )
        )
        mp = MaterialProgram.from_program(prog)
        ws = WorldState(prog, mp, initialize_surface_state(build_surface_domain(prog, mp)))
        # Sweep delta +80 deg around Z: starts at -40 deg (clear), ends at +40 deg (clear), collides with obstacle at +40 delta (0 deg world)!
        res = step_world(ws, SetPoseAction({"swing": (0.0, 0.0, 80.0)}, substeps=9), dt=0.0)
        self.assertFalse(res.swept_audit_passed)
        self.assertGreater(res.max_illegal_overlap_depth, 0.5)


class TestInverseInferenceAndTrajectoryDataset(unittest.TestCase):
    def test_license_gate_blocks_mojang_and_proprietary_assets(self) -> None:
        dummy_java = {
            "texture_size": [16, 16],
            "elements": [{"from": [4, 0, 4], "to": [12, 8, 12], "faces": {}}],
        }
        with self.assertRaises(LicenseGateError):
            ingest_external_asset(
                dummy_java,
                "java_json",
                AssetProvenance("bad_1", "https://example.org/x", "Author", "All-Rights-Reserved"),
            )
        with self.assertRaises(LicenseGateError):
            ingest_external_asset(
                dummy_java,
                "java_json",
                AssetProvenance("mojang_pig", "https://example.org/minecraft_jar", "Author", "CC0-1.0"),
            )

    def test_bbmodel_java_and_bedrock_ingestion_plus_labpbr_posterior(self) -> None:
        ws = build_articulated_mechanism_world(seed=101)
        scene = compile_geometry(ws.program)
        atlas_desc, _ = compile_vanilla_atlas(ws)
        bb_doc = export_bbmodel(scene, atlas=atlas_desc)
        lab = export_labpbr_1_3(ws, uv_layout=atlas_desc)

        prov = AssetProvenance("mech_arm_v1", "https://example.org/mech", "OpenContributor", "Apache-2.0")
        rec_rgb = ingest_external_asset(bb_doc, "bbmodel", prov, top_k=3)
        rec_pbr = ingest_external_asset(
            bb_doc,
            "bbmodel",
            prov,
            atlas_png=lab["atlas_png"],
            labpbr_n_png=lab["atlas_n_png"],
            labpbr_s_png=lab["atlas_s_png"],
            top_k=3,
        )
        self.assertGreater(rec_pbr.identifiability_score, rec_rgb.identifiability_score + 0.15)
        self.assertEqual(rec_pbr.material_hypotheses[0].spec.family, "painted_metal")
        self.assertGreaterEqual(len(rec_pbr.extracted_stamps), 1)

        # Java .json ingestion roundtrip
        prog_j = validate_program(
            Program(
                Meta("weapon_3d", "java_modern", 16),
                [Palette("oak", (0.58, 0.09, 64.0), 6)],
                Bone("r", pivot_local=(8.0, 4.0, 8.0), items=[Box("beam", (4, 8, 4), "oak")]),
            )
        )
        java_doc = export_java(compile_geometry(prog_j), profile="java_modern")
        rec_java = ingest_external_asset(java_doc, "java_json", AssetProvenance("oak_beam", "https://example.org/j", "Builder", "CC0-1.0"))
        self.assertLess(rec_java.vertex_reconstruction_error, 1e-6)

        # Bedrock .geo.json ingestion roundtrip
        bedrock_doc = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.custom_drone", "texture_width": 64, "texture_height": 64},
                    "bones": [
                        {"name": "body", "pivot": [0, 8, 0], "cubes": [{"origin": [-3, 5, -4], "size": [6, 6, 8], "uv": [0, 0]}]},
                        {"name": "wing_l", "parent": "body", "pivot": [-3, 10, 0], "rotation": [0, 0, 15], "cubes": [{"origin": [-9, 9, -2], "size": [6, 1, 4], "uv": [0, 16]}]},
                    ],
                }
            ],
        }
        rec_bed = ingest_external_asset(bedrock_doc, "bedrock_geojson", AssetProvenance("drone_01", "https://example.org/b", "Builder", "MIT"))
        self.assertEqual(len(compile_geometry(rec_bed.program).boxes), 2)

    def test_trajectory_dataset_compositional_holdout_and_matched_budget(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ds_a = build_intervention_trajectory_dataset(preset="smoke_24", seed=515, output_dir=Path(td) / "a")
            ds_b = build_world_capability_ladder_dataset(preset="smoke_24", seed=515, output_dir=Path(td) / "b")
            self.assertEqual(ds_a["manifest_sha256"], ds_b["manifest_sha256"])
            self.assertGreaterEqual(ds_a["total_trajectories"], 42)
            self.assertTrue(ds_a["zero_compositional_leakage"])
            self.assertEqual(set(ds_a["level_counts"].keys()), set(WORLD_CAPABILITY_LEVELS))

            # C1: Calling evaluate_matched_budget_world_baselines without model_completions MUST fail
            with self.assertRaises(TypeError):
                evaluate_matched_budget_world_baselines(ds_a)  # type: ignore[call-arg]

            # Empty completions must score 0.0 (never unconditional credit!)
            comp_empty = evaluate_matched_budget_world_baselines(ds_a, {})
            self.assertTrue(comp_empty["matched_budget_verified"])
            self.assertEqual(comp_empty["evaluated_model"]["core_levels_1_to_6_mean_score"], 0.0)
            self.assertEqual(comp_empty["evaluated_model"]["core_levels_1_to_6_exact_pass_rate"], 0.0)
            self.assertLess(comp_empty["gain_over_constant_prior"], 0.0)

            #Simulator ground-truth completions achieve 1.0 and beat constant-prior & NN baselines
            gt_completions = {
                str(r["sample_id"]): str(r["target_completion"])
                for r in ds_a["rows"]
                if r["split"] == "held_out_compositional_val"
            }
            comp_gt = evaluate_matched_budget_world_baselines(ds_a, gt_completions)
            self.assertEqual(comp_gt["evaluated_model"]["core_levels_1_to_6_exact_pass_rate"], 1.0)
            self.assertGreater(comp_gt["gain_over_constant_prior"], 0.15)
            self.assertGreater(comp_gt["gain_over_nearest_neighbor"], 0.10)
            self.assertTrue(comp_gt["simulator_oracle_upper_bound"]["is_oracle"])
            self.assertFalse(comp_gt["simulator_oracle_upper_bound"]["is_trained_model"])

    def test_c1_and_h12_seven_level_capability_benchmark_observations_and_probes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "ladder"
            ds = build_world_capability_ladder_dataset(preset="smoke_24", seed=2026, output_dir=out_dir)

            # Verify JSON serializability of all rows (notebook & JSONL contract)
            serialized = json.dumps(ds["rows"], sort_keys=True)
            self.assertGreater(len(serialized), 2000)
            self.assertTrue((out_dir / "world_capability_ladder.jsonl").is_file())

            # Zero Level-7 authored rigs in train split
            train_l7 = [r for r in ds["rows"] if r["split"] == "train" and r["level"] == "level_7_human_transfer"]
            self.assertEqual(len(train_l7), 0)

            # All 6 sub-levels explicitly represented in Level 7 Human Transfer
            l7_rows = [r for r in ds["rows"] if r["level"] == "level_7_human_transfer"]
            self.assertEqual({r["sub_level"] for r in l7_rows}, set(WORLD_CAPABILITY_LEVELS[:6]))
            for r in l7_rows:
                self.assertEqual(r["audit_profile"], "authored")

            # Level 5 multi-view PNG observations rendered & decodable
            l5_rows = [r for r in ds["rows"] if r.get("sub_level") == "level_5_invert_observation"]
            self.assertGreaterEqual(len(l5_rows), 2)
            for r in l5_rows:
                img = decode_record_observation_image(r)
                self.assertIsNotNone(img)
                arr = np.asarray(img)
                self.assertEqual(arr.ndim, 3)
                self.assertEqual(arr.shape[1], 4 * arr.shape[0])
                self.assertGreater(int((arr > 0).sum()), 100)
                self.assertTrue(Path(r["observation_image_path"]).is_file())
                self.assertTrue(Path(r["four_views_image_path"]).is_file())

            # Zero split-label or borrowed IP leakage in prompts/IDs
            banned = (
                "held-out",
                "holdout",
                "held_out",
                "unseen",
                "test_unseen",
                "seen_train",
                "validation",
                "apricorn",
                "cobblemon",
                "pokemon",
                "pokémon",
                "monster hunter",
                "mojang",
            )
            for r in ds["rows"]:
                blob = f"{r['sample_id']} {r['prompt']}".lower()
                for tok in banned:
                    self.assertNotIn(tok, blob)

            # Garbage completions MUST score 0.0 across every level
            eval_rows = [r for r in ds["rows"] if r["split"] == "held_out_compositional_val"]
            garbage_comps = {str(r["sample_id"]): "not valid json or dsl @@@" for r in eval_rows}
            bench_bad = evaluate_world_capability_benchmark(ds, garbage_comps, include_baselines=True)
            self.assertEqual(bench_bad["model_results"]["core_levels_1_to_6_mean_score"], 0.0)
            self.assertEqual(bench_bad["model_results"]["overall_parse_valid_rate"], 0.0)
            self.assertEqual(bench_bad["model_results"]["consistency_probes"]["consistency_rate"], 0.0)
            self.assertFalse(bench_bad["model_results"]["is_oracle"])
            self.assertTrue(bench_bad["model_results"]["is_trained_model"])

            # Verify Level 7 is reported strictly per sub-level and never collapsed
            l7_rep = bench_bad["baselines"]["simulator_oracle_upper_bound"]["level_7_human_transfer_by_sublevel"]
            self.assertEqual(set(l7_rep.keys()), set(WORLD_CAPABILITY_LEVELS[:6]))
            self.assertTrue(bench_bad["model_results"]["level_7_collapsed_average_forbidden"])
            self.assertNotIn("level_7_mean_score", bench_bad["model_results"])

            # Verify baselines: constant prior gets 0.0 on directional consistency probes, oracle gets 1.0
            prior_b = bench_bad["baselines"]["constant_prior_baseline"]
            nn_b = bench_bad["baselines"]["nearest_neighbor_train_retrieval"]
            oracle_b = bench_bad["baselines"]["simulator_oracle_upper_bound"]
            self.assertEqual(prior_b["consistency_probes"]["consistency_rate"], 0.0)
            self.assertEqual(oracle_b["consistency_probes"]["consistency_rate"], 1.0)
            self.assertEqual(oracle_b["core_levels_1_to_6_exact_pass_rate"], 1.0)
            self.assertFalse(prior_b["is_trained_model"])
            self.assertFalse(nn_b["is_trained_model"])
            self.assertTrue(oracle_b["is_oracle"])
            self.assertFalse(oracle_b["is_trained_model"])
            self.assertIn("NEVER report as a learned model score", oracle_b["warning"])

            # Verify inverted probe responses fail directional consistency even when parse-valid
            inverted_comps: dict[str, str] = {}
            probe_pairs: dict[str, dict[str, dict]] = {}
            for r in eval_rows:
                inverted_comps[str(r["sample_id"])] = str(r["target_completion"])
                if r.get("probe_pair_id") and r.get("probe_role"):
                    probe_pairs.setdefault(str(r["probe_pair_id"]), {})[str(r["probe_role"])] = r
            for grp in probe_pairs.values():
                b_id = str(grp["base"]["sample_id"])
                p_id = str(grp["perturbed"]["sample_id"])
                inverted_comps[b_id], inverted_comps[p_id] = inverted_comps[p_id], inverted_comps[b_id]
            bench_inv = evaluate_world_capability_benchmark(ds, inverted_comps, include_baselines=False)
            self.assertEqual(bench_inv["model_results"]["consistency_probes"]["consistency_rate"], 0.0)

            # Verify Level-6 AST edit-size penalty rewards minimal edits over bloated edits
            l6_row = next(r for r in ds["rows"] if r["level"] == "level_6_edit_and_repair")
            s_min = score_world_capability_completion(l6_row, str(l6_row["target_completion"]))
            ref_prog = parse_program(str(l6_row["target_completion"]))
            bloated_items = [
                *ref_prog.root.items,
                Locator("extra_unrequested_1", offset=(0.0, 9.0, 0.0), role="socket"),
                Locator("extra_unrequested_2", offset=(0.0, 10.0, 0.0), role="socket"),
                Locator("extra_unrequested_3", offset=(0.0, 11.0, 0.0), role="socket"),
                Locator("extra_unrequested_4", offset=(0.0, 12.0, 0.0), role="socket"),
            ]
            from dataclasses import replace as dc_replace

            bloated_prog = validate_program(dc_replace(ref_prog, root=dc_replace(ref_prog.root, items=bloated_items)))
            self.assertGreater(
                compute_ast_edit_size(parse_program(str(l6_row["ground_truth"]["input_dsl"])), bloated_prog),
                s_min["metrics"]["ast_edit_size"],
            )
            s_bloat = score_world_capability_completion(l6_row, format_program(bloated_prog))
            self.assertTrue(s_min["exact_pass"])
            self.assertFalse(s_bloat["exact_pass"])
            self.assertGreater(s_min["score"], s_bloat["score"])

    def test_h7_artist_loop_unlabeled_provenance_defaults_to_noassertion(self) -> None:
        default_prov = build_unlabeled_workspace_provenance("local_sketch_01")
        self.assertEqual(default_prov.license_spdx, "NOASSERTION")

        bedrock_doc = {
            "format_version": "1.12.0",
            "minecraft:geometry": [
                {
                    "description": {"identifier": "geometry.custom_construct", "texture_width": 64, "texture_height": 64},
                    "bones": [
                        {"name": "core", "pivot": [0, 6, 0], "cubes": [{"origin": [-3, 3, -3], "size": [6, 6, 6], "uv": [0, 0]}]},
                    ],
                }
            ],
        }
        # 1. Unlabeled local workspace input defaults to NOASSERTION and commercial_weights_eligible=False
        ing = ingest_artist_project_record(bedrock_doc, format_kind="bedrock_geojson")
        self.assertEqual(ing["license_spdx"], "NOASSERTION")
        self.assertNotEqual(ing["license_spdx"], "CC-BY-4.0")
        self.assertEqual(ing["license_policy"], LOCAL_WORKSPACE_LICENSE_POLICY)
        self.assertFalse(ing["commercial_weights_eligible"])

        critique = analyze_artist_project(bedrock_doc, format_kind="bedrock_geojson")
        self.assertGreaterEqual(critique.box_count, 1)

        loop_rep = run_looped_artist_enhancement(bedrock_doc, format_kind="bedrock_geojson", max_iterations=1)
        self.assertEqual(loop_rep["license_spdx"], "NOASSERTION")
        self.assertFalse(loop_rep["commercial_weights_eligible"])

        # 2. Requesting clean_room_permissive without explicit provenance MUST fail LicenseGateError
        with self.assertRaises(LicenseGateError):
            ingest_artist_project_record(
                bedrock_doc,
                format_kind="bedrock_geojson",
                license_policy="clean_room_permissive",
            )

        # 3. Verified permissive provenance opts into commercial_weights_eligible=True
        verified_prov = AssetProvenance("open_rig_01", "https://example.org/rig", "StudioArtist", "Apache-2.0")
        ing_ok = ingest_artist_project_record(
            bedrock_doc,
            format_kind="bedrock_geojson",
            provenance=verified_prov,
            license_policy="clean_room_permissive",
        )
        self.assertEqual(ing_ok["license_spdx"], "Apache-2.0")
        self.assertTrue(ing_ok["commercial_weights_eligible"])

        # 4. NoAI / do-not-train markers in local payload metadata are strictly blocked
        noai_doc = {**bedrock_doc, "credit": "Artist Name - NoAI / Do Not Train"}
        with self.assertRaises(LicenseGateError):
            ingest_artist_project_record(noai_doc, format_kind="bedrock_geojson")

    def test_h12_larger_presets_pilot_200_and_world_1200_zero_compositional_leakage(self) -> None:
        for preset_name, min_expected in (("pilot_200", 210), ("world_1200", 1200)):
            ds = build_world_capability_ladder_dataset(
                preset=preset_name,
                seed=2026,
                render_observations=False,
            )
            self.assertTrue(ds["zero_compositional_leakage"])
            self.assertGreaterEqual(ds["total_trajectories"], min_expected)
            train_triples = {r["compositional_triple"] for r in ds["rows"] if r["split"] == "train"}
            ood_triples = {
                r["compositional_triple"]
                for r in ds["rows"]
                if r["split"] == "held_out_compositional_val" and r["level"] != "level_7_human_transfer"
            }
            self.assertTrue(train_triples.isdisjoint(ood_triples))
            self.assertTrue(
                all(r["level"] != "level_7_human_transfer" for r in ds["rows"] if r["split"] == "train")
            )

    def test_bbmodel4_inline_outliner_locators_sharealike_and_gltf2_ingestion(self) -> None:
        # 1. Blockbench 4 inline outliner (no top-level 'groups') + locator element + decimal 0.01 size_step
        bb4_doc = {
            "meta": {"format_version": "4.10", "model_format": "free"},
            "resolution": {"width": 32, "height": 32},
            "elements": [
                {
                    "name": "torso_plate",
                    "from": [-2.48, 4.0, -1.5],
                    "to": [2.48, 10.0, 1.5],
                    "uuid": "el-cube-1",
                    "type": "cube",
                },
                {
                    "name": "seat_1",
                    "position": [0.0, 10.5, -0.5],
                    "uuid": "el-loc-1",
                    "type": "locator",
                },
                {
                    "name": "stray_zero_line",
                    "from": [0.0, 0.0, 0.0],
                    "to": [0.0, 4.0, 0.0],
                    "uuid": "el-deg-1",
                    "type": "cube",
                },
            ],
            "outliner": [
                {
                    "name": "mount_root",
                    "origin": [0.0, 4.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "uuid": "grp-inline-root",
                    "children": [
                        "el-cube-1",
                        "el-loc-1",
                        "el-deg-1",
                    ],
                }
            ],
        }
        prov_sa = AssetProvenance(
            "bb4_mount_01",
            "https://example.org/mount",
            "HumanArtist",
            "CC-BY-SA-4.0",
            attribution_notice="Mount by HumanArtist (CC-BY-SA-4.0)",
        )
        with self.assertRaises(LicenseGateError):
            ingest_external_asset(bb4_doc, "bbmodel", prov_sa, license_policy="clean_room_permissive")
        rec_sa = ingest_external_asset(bb4_doc, "bbmodel", prov_sa, license_policy="sharealike_ccbysa")
        self.assertEqual(rec_sa.license_track, "sharealike_ccbysa")
        self.assertFalse(rec_sa.commercial_weights_eligible)
        self.assertLess(rec_sa.vertex_reconstruction_error, 1e-6)
        self.assertEqual(rec_sa.program.meta.size_step, 0.01)
        sc_sa = compile_geometry(parse_program(format_program(rec_sa.program)))
        self.assertEqual(len(sc_sa.boxes), 1)
        self.assertEqual(len(sc_sa.locators), 1)
        self.assertEqual(sc_sa.locators[0].role, "rider_seat")

        # 2. AI-assisted quarantine gate
        prov_ai = AssetProvenance(
            "ai_assist_01",
            "https://example.org/ai_pack",
            "HybridCreator",
            "CC0-1.0",
            ai_assisted=True,
        )
        with self.assertRaises(LicenseGateError):
            ingest_external_asset(bb4_doc, "bbmodel", prov_ai, license_policy="clean_room_permissive")
        rec_ai = ingest_external_asset(bb4_doc, "bbmodel", prov_ai, license_policy="ai_assisted_quarantine")
        self.assertEqual(rec_ai.license_track, "ai_assisted_quarantine")
        self.assertFalse(rec_ai.commercial_weights_eligible)

        # 3. Free-mesh .bbmodel explicitly rejected with UnsupportedFeature
        mesh_bb = {
            "elements": [{"name": "sculpt", "type": "mesh", "vertices": {"v0": [0, 0, 0]}}],
            "outliner": [],
        }
        with self.assertRaises(UnsupportedFeature):
            ingest_external_asset(
                mesh_bb,
                "bbmodel",
                AssetProvenance("m1", "https://example.org/m", "Sculptor", "CC0-1.0"),
            )

        # 4. Sketchfab glTF 2.0 archive (.zip) cuboid + quaternion animation roundtrip & free-mesh rejection
        corners_m = np.array(
            [
                [-0.125, 0.0, -0.0625],
                [0.125, 0.0, -0.0625],
                [0.125, 0.25, -0.0625],
                [-0.125, 0.25, -0.0625],
                [-0.125, 0.0, 0.0625],
                [0.125, 0.0, 0.0625],
                [0.125, 0.25, 0.0625],
                [-0.125, 0.25, 0.0625],
            ],
            dtype=np.float32,
        )
        tris_idx = np.array(
            [
                0, 1, 2, 0, 2, 3,
                4, 6, 5, 4, 7, 6,
                0, 4, 5, 0, 5, 1,
                3, 2, 6, 3, 6, 7,
                1, 5, 6, 1, 6, 2,
                0, 3, 7, 0, 7, 4,
            ],
            dtype=np.uint16,
        )
        anim_t = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        s_ang = float(np.sin(np.deg2rad(15.0)))
        c_ang = float(np.cos(np.deg2rad(15.0)))
        anim_q = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [s_ang, 0.0, 0.0, c_ang],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        b0 = corners_m.tobytes()
        b1 = tris_idx.tobytes() + b"\x00\x00"  # 4-byte align
        b2 = anim_t.tobytes()
        b3 = anim_q.tobytes()
        scene_bin = b0 + b1 + b2 + b3

        png_buf = io.BytesIO()
        Image.fromarray(np.full((16, 16, 4), (90, 140, 200, 255), dtype=np.uint8)).save(png_buf, format="PNG")

        gltf_doc = {
            "asset": {"version": "2.0", "generator": "Blockbench glTF exporter"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [
                {"name": "Sketchfab_model", "children": [1], "matrix": [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1]},
                {"name": "GLTF_SceneRootNode", "children": [2], "matrix": [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1]},
                {"name": "body", "children": [3, 5]},
                {"name": "cube_0", "children": [4]},
                {"name": "Object_0", "mesh": 0},
                {"name": "arm_l", "translation": [0.1875, 0.25, 0.0], "children": [6]},
                {"name": "Object_1", "mesh": 0},
            ],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
            "textures": [{"source": 0}],
            "images": [{"uri": "textures/tex.png"}],
            "buffers": [{"uri": "scene.bin", "byteLength": len(scene_bin)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(b0)},
                {"buffer": 0, "byteOffset": len(b0), "byteLength": len(tris_idx.tobytes())},
                {"buffer": 0, "byteOffset": len(b0) + len(b1), "byteLength": len(b2)},
                {"buffer": 0, "byteOffset": len(b0) + len(b1) + len(b2), "byteLength": len(b3)},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 36, "type": "SCALAR"},
                {"bufferView": 2, "componentType": 5126, "count": 3, "type": "SCALAR"},
                {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC4"},
            ],
            "animations": [
                {
                    "name": "swing",
                    "samplers": [{"input": 2, "output": 3, "interpolation": "LINEAR"}],
                    "channels": [{"sampler": 0, "target": {"node": 5, "path": "rotation"}}],
                }
            ],
        }
        zip_mem = io.BytesIO()
        with zipfile.ZipFile(zip_mem, "w") as zf:
            zf.writestr("scene.gltf", json.dumps(gltf_doc))
            zf.writestr("scene.bin", scene_bin)
            zf.writestr("textures/tex.png", png_buf.getvalue())

        rec_gltf = ingest_gltf_archive(
            zip_mem.getvalue(),
            AssetProvenance("sf_mech_01", "https://sketchfab.com/3d-models/mech", "VoxelMaker", "CC-BY-4.0"),
            license_policy="clean_room_permissive",
        )
        self.assertLess(rec_gltf.vertex_reconstruction_error, 1e-6)
        sc_gltf = compile_geometry(parse_program(format_program(rec_gltf.program)))
        self.assertEqual(len(sc_gltf.boxes), 2)
        self.assertEqual(len(sc_gltf.bones), 2)
        self.assertIsNotNone(rec_gltf.program.behavior)
        assert rec_gltf.program.behavior is not None
        self.assertEqual(len(rec_gltf.program.behavior.clips), 1)

        # Free-mesh glTF (pyramid / 5 vertices, 6 triangles) must raise UnsupportedFeature
        pyr_pts = np.array(
            [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1], [0, 2, 0]],
            dtype=np.float32,
        )
        pyr_idx = np.array([0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4, 0, 2, 1, 0, 3, 2], dtype=np.uint16)
        pb0 = pyr_pts.tobytes()
        pb1 = pyr_idx.tobytes()
        gltf_pyr = {
            **gltf_doc,
            "animations": [],
            "buffers": [{"uri": "scene.bin", "byteLength": len(pb0) + len(pb1)}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": len(pb0)},
                {"buffer": 0, "byteOffset": len(pb0), "byteLength": len(pb1)},
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 5, "type": "VEC3"},
                {"bufferView": 1, "componentType": 5123, "count": 18, "type": "SCALAR"},
            ],
        }
        zip_pyr = io.BytesIO()
        with zipfile.ZipFile(zip_pyr, "w") as zf:
            zf.writestr("scene.gltf", json.dumps(gltf_pyr))
            zf.writestr("scene.bin", pb0 + pb1)
            zf.writestr("textures/tex.png", png_buf.getvalue())
        with self.assertRaises(UnsupportedFeature):
            ingest_gltf_archive(
                zip_pyr.getvalue(),
                AssetProvenance("sf_pyr_01", "https://sketchfab.com/3d-models/pyr", "Sculptor", "CC0-1.0"),
            )

    def test_visual_identity_multi_texture_uv_rotation_visibility_and_multi_slot_materials(self) -> None:
        import base64

        # 1. Mixed 64x64 + 16x16 multi-texture Blockbench 5.0 with negative inflate,
        #    swapped from>to, 90/180/270 deg UV rotation, omitted face, and hidden variant group
        tex0 = np.zeros((64, 64, 4), dtype=np.uint8)
        tex0[:, :] = (46, 98, 178, 255)  # cobalt metal body
        tex0[0:16, 0:16] = (230, 74, 58, 255)  # top-left quadrant red
        tex0[0:16, 16:32] = (62, 198, 112, 255)  # top-right quadrant green

        tex1 = np.zeros((16, 16, 4), dtype=np.uint8)
        tex1[:, :] = (232, 188, 64, 255)  # warm brass trim
        tex1[0:8, 0:8] = (250, 240, 180, 255)

        b0_png = io.BytesIO()
        Image.fromarray(tex0).save(b0_png, format="PNG")
        b1_png = io.BytesIO()
        Image.fromarray(tex1).save(b1_png, format="PNG")

        bb5_doc = {
            "meta": {"format_version": "5.0", "model_format": "free"},
            "resolution": {"width": 64, "height": 64},
            "textures": [
                {
                    "id": "10",
                    "name": "cobalt.png",
                    "width": 64,
                    "height": 64,
                    "uv_width": 64,
                    "uv_height": 64,
                    "source": "data:image/png;base64," + base64.b64encode(b0_png.getvalue()).decode("ascii"),
                },
                {
                    "id": "22",
                    "name": "brass.png",
                    "width": 16,
                    "height": 16,
                    "uv_width": 16,
                    "uv_height": 16,
                    "source": "data:image/png;base64," + base64.b64encode(b1_png.getvalue()).decode("ascii"),
                },
            ],
            "elements": [
                {
                    "name": "hull_box",
                    "uuid": "u-hull",
                    "type": "cube",
                    "from": [4.0, 2.0, -3.0],  # intentionally swapped X: from[0] > to[0]
                    "to": [-4.0, 10.0, 3.0],
                    "inflate": -0.25,
                    "faces": {
                        "north": {"uv": [0, 0, 32, 16], "texture": 0, "rotation": 90},
                        "south": {"uv": [0, 0, 32, 16], "texture": 0, "rotation": 180},
                        "east": {"uv": [0, 0, 16, 16], "texture": 0, "rotation": 270},
                        "west": {"uv": [16, 0, 32, 16], "texture": 0},
                        "up": {"uv": [0, 0, 16, 16], "texture": 0},
                        # "down" intentionally omitted (transparent face)
                    },
                },
                {
                    "name": "brass_cap",
                    "uuid": "u-cap",
                    "type": "cube",
                    "from": [-2.0, 9.75, -2.0],
                    "to": [2.0, 12.75, 2.0],
                    "faces": {
                        fn: {"uv": [0, 0, 16, 16], "texture": 1}
                        for fn in ("north", "south", "east", "west", "up", "down")
                    },
                },
                {
                    "name": "hidden_alt_wing",
                    "uuid": "u-hidden-el",
                    "type": "cube",
                    "visibility": False,
                    "from": [-12.0, 4.0, -2.0],
                    "to": [-5.0, 8.0, 2.0],
                    "faces": {fn: {"uv": [0, 0, 16, 16], "texture": 0} for fn in ("north", "south", "east", "west", "up", "down")},
                },
                {
                    "name": "hidden_group_cube",
                    "uuid": "u-hidden-grp-el",
                    "type": "cube",
                    "from": [5.0, 4.0, -2.0],
                    "to": [12.0, 8.0, 2.0],
                    "faces": {fn: {"uv": [0, 0, 16, 16], "texture": 1} for fn in ("north", "south", "east", "west", "up", "down")},
                },
            ],
            "outliner": [
                {
                    "name": "main_rig",
                    "uuid": "g-main",
                    "origin": [0.0, 2.0, 0.0],
                    "visibility": True,
                    "children": ["u-hull", "u-cap", "u-hidden-el"],
                },
                {
                    "name": "alt_variant_rig",
                    "uuid": "g-hidden",
                    "origin": [0.0, 2.0, 0.0],
                    "visibility": False,
                    "children": ["u-hidden-grp-el"],
                },
            ],
        }
        prov_bb5 = AssetProvenance("bb5_vis_01", "https://example.org/bb5", "ArtistA", "CC-BY-4.0")
        rec_bb5 = ingest_external_asset(bb5_doc, "bbmodel", prov_bb5, license_policy="clean_room_permissive")
        enriched_bb5 = assign_multi_slot_materials_to_program(
            rec_bb5.program,
            rec_bb5.resolved_uv_map,
            rec_bb5.resolved_atlas_rgba,
            rec_bb5.material_hypotheses[0].spec,
            has_source_texture=rec_bb5.has_source_texture,
            domain="industrial",
            max_slots=4,
            seed=42,
        )
        self.assertGreaterEqual(len(enriched_bb5.palettes), 2)
        self.assertEqual(len(compile_geometry(enriched_bb5).boxes), 2)  # both hidden cubes excluded!
        m_bb5, panels_bb5 = audit_converted_asset_visual_identity(
            bb5_doc,
            "bbmodel",
            rec_bb5,
            enriched_bb5,
            canvas_size=(160, 160),
        )
        self.assertEqual(len(panels_bb5), 4)
        self.assertGreaterEqual(m_bb5["psnr_db"], 50.0, msg=str(m_bb5))
        self.assertGreaterEqual(m_bb5["standalone_dsl_psnr_db"], 50.0, msg=str(m_bb5))
        self.assertGreaterEqual(m_bb5["silhouette_iou"], 0.999, msg=str(m_bb5))
        self.assertLess(m_bb5["vertex_reconstruction_error"], 1e-5)

        # 2. Multi-texture Java .json with rescale=True 22.5 deg rotation
        j_side = io.BytesIO()
        Image.fromarray(np.full((16, 16, 4), (168, 108, 62, 255), dtype=np.uint8)).save(j_side, format="PNG")
        j_top = io.BytesIO()
        Image.fromarray(np.full((16, 16, 4), (74, 156, 84, 255), dtype=np.uint8)).save(j_top, format="PNG")
        java_doc = {
            "textures": {"side": "custom:block/oak_side", "top": "custom:block/moss_top"},
            "elements": [
                {
                    "name": "angled_beam",
                    "from": [4.0, 0.0, 4.0],
                    "to": [12.0, 12.0, 12.0],
                    "rotation": {"origin": [8.0, 6.0, 8.0], "axis": "y", "angle": 22.5, "rescale": True},
                    "faces": {
                        "north": {"uv": [0, 0, 8, 12], "texture": "#side"},
                        "south": {"uv": [0, 0, 8, 12], "texture": "#side"},
                        "east": {"uv": [0, 0, 8, 12], "texture": "#side"},
                        "west": {"uv": [0, 0, 8, 12], "texture": "#side"},
                        "up": {"uv": [0, 0, 8, 8], "texture": "#top", "rotation": 90},
                        "down": {"uv": [0, 0, 8, 8], "texture": "#top"},
                    },
                }
            ],
        }
        java_tex_map = {"side": j_side.getvalue(), "top": j_top.getvalue()}
        rec_java = ingest_external_asset(
            java_doc,
            "java_json",
            AssetProvenance("java_vis_01", "https://example.org/java", "ArtistB", "CC0-1.0"),
            atlas_png=j_side.getvalue(),
            java_textures_png=java_tex_map,
            license_policy="clean_room_permissive",
        )
        enriched_java = assign_multi_slot_materials_to_program(
            rec_java.program,
            rec_java.resolved_uv_map,
            rec_java.resolved_atlas_rgba,
            rec_java.material_hypotheses[0].spec,
            has_source_texture=rec_java.has_source_texture,
            domain="architecture",
            seed=7,
        )
        m_java, _ = audit_converted_asset_visual_identity(
            java_doc,
            "java_json",
            rec_java,
            enriched_java,
            atlas_png=j_side.getvalue(),
            java_textures_png=java_tex_map,
            canvas_size=(160, 160),
        )
        self.assertGreaterEqual(m_java["psnr_db"], 55.0, msg=str(m_java))
        self.assertGreaterEqual(m_java["silhouette_iou"], 0.999, msg=str(m_java))


if __name__ == "__main__":
    unittest.main()
