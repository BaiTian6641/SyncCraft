"""Comprehensive tests for P1-03d/e world transitions, 7 interventions, inverse inference, and datasets."""

from __future__ import annotations

import json
import unittest

from synccraft.backends import export_bbmodel, export_java, export_labpbr_1_3
from synccraft.data_gen.ik_motion_templates import generate_ik_clip, solve_two_bone_ik
from synccraft.dsl import Attach, Bone, Box, Meta, Palette, Place, Program, validate_program
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.shading import compile_vanilla_atlas
from synccraft.surface import MaterialProgram, WorldState, build_surface_domain, initialize_surface_state
from synccraft.world import (
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
    build_articulated_mechanism_world,
    build_intervention_trajectory_dataset,
    evaluate_matched_budget_world_baselines,
    ingest_external_asset,
    run_seven_interventions_suite,
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
        ds_a = build_intervention_trajectory_dataset(preset="smoke_24", seed=515)
        ds_b = build_intervention_trajectory_dataset(preset="smoke_24", seed=515)
        self.assertEqual(ds_a["manifest_sha256"], ds_b["manifest_sha256"])
        self.assertEqual(ds_a["total_trajectories"], 24)
        self.assertTrue(ds_a["zero_compositional_leakage"])
        self.assertEqual(set(ds_a["capability_counts"].keys()), {"infer", "predict", "plan_repair"})

        comp = evaluate_matched_budget_world_baselines(ds_a)
        self.assertTrue(comp["matched_budget_verified"])
        self.assertGreater(comp["world_model_policy"]["accuracy"], comp["program_only_baseline"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
