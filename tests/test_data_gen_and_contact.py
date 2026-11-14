"""Tests for P1-02 physical contact/connectivity audit and P1-08 complex dataset generators."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

from synccraft.audit.sync_audit import audit_scene
from synccraft.backends.bbmodel5_exporter import export_bbmodel
from synccraft.backends.java_exporter import export_java
from synccraft.data_gen import (
    DATASET_PRESETS,
    MACRO_DOMAINS,
    MANDATORY_ROW_FIELDS,
    SEEN_STATIC_FAMILIES,
    UNSEEN_STATIC_FAMILIES,
    build_verified_dataset,
    verify_and_package_candidate,
)
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.sat_potential import sat_obb
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.shading.uv_packer import build_atlas


class TestP102PhysicalContactAudit(unittest.TestCase):
    """Verify P1-02 physical contact, co-penetration, and connectivity enforcement."""

    def test_numerical_notes_sec4_coincident_d0_rejected(self):
        """NUMERICAL_NOTES.md sec 4 fixture D=0: two coincident [20,20,20] boxes must fail P1-02."""
        dsl = """
        @meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.6, 0.02, 240.0], steps=4) }
        @rig {
          root bone root_node {
            pivot_local: [8.0, 8.0, 8.0]
            box base { size: [20, 20, 20], material: stone }
            box co_box place(base_box=base, anchor=[0, 0, 0], align=[0, 0, 0], offset=[0, 0, 0]) {
              size: [20, 20, 20], material: stone
            }
          }
        }
        """
        scene = compile_geometry(parse_program(dsl))
        report = audit_scene(scene)
        self.assertFalse(report["passed"])
        violations = [v["violation"] for v in report["contact_violations"]]
        self.assertIn("excessive_contact_penetration", violations)
        self.assertTrue(report["contact_violations"][0]["center_buried"])
        self.assertAlmostEqual(report["declared_contact_pairs"][0]["raw_min_overlap"], 20.0, places=6)

        # Legacy diagnostic opt-out remains explicit when enforce_physical_contact=False
        legacy = audit_scene(scene, enforce_physical_contact=False)
        self.assertTrue(legacy["passed"])
        self.assertGreater(len(legacy["contact_violations"]), 0)

    def test_numerical_notes_sec4_far_gap_d50_rejected(self):
        """NUMERICAL_NOTES.md sec 4 fixture D=50: 30-unit separated declared pair must fail P1-02."""
        dsl = """
        @meta { archetype: block_multipart, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.6, 0.02, 240.0], steps=4) }
        @rig {
          root bone root_node {
            pivot_local: [8.0, 8.0, 8.0]
            box base { size: [20, 20, 20], material: stone }
            box far_box place(base_box=base, anchor=[0, 0, 0], align=[0, 0, 0], offset=[50, 0, 0]) {
              size: [20, 20, 20], material: stone
            }
          }
        }
        """
        scene = compile_geometry(parse_program(dsl))
        report = audit_scene(scene)
        self.assertFalse(report["passed"])
        violations = {v["violation"] for v in report["contact_violations"]}
        self.assertIn("separated_declared_contact", violations)
        self.assertIn("disconnected_physical_components", violations)

    def test_folded_back_child_bone_interpenetrating_parent_rejected(self):
        """Surface anchor is valid on +X face, but 160-deg rotation folds child back into parent."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { steel: ramp(base_lch=[0.75, 0.02, 240.0], steps=4) }
        @rig {
          root bone hub_root {
            pivot_local: [8.0, 8.0, 8.0]
            box hub { size: [6, 6, 6], material: steel }
            bone folded_wing attach(parent_box=hub, anchor=[1, 0, 0], align=[-1, 0, 0]) {
              rest_euler: [0.0, 0.0, 160.0]
              box panel { size: [6, 2, 2], material: steel }
            }
          }
        }
        """
        scene = compile_geometry(parse_program(dsl))
        report = audit_scene(scene)
        self.assertFalse(report["passed"])
        self.assertEqual(report["contact_violations"][0]["violation"], "excessive_contact_penetration")

    def test_shallow_gem_inset_inflate_and_acute_hinge_accepted(self):
        """Intentional <=0.5 surface inset, inflate collar, and 22.5/45-deg hinges must pass P1-02."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette {
          steel: ramp(base_lch=[0.78, 0.02, 240.0], steps=4)
          ruby: ramp(base_lch=[0.65, 0.20, 25.0], steps=4, emissive=true)
        }
        @rig {
          root bone hilt {
            pivot_local: [8.0, 8.0, 8.0]
            box guard { size: [6, 2, 2], material: steel, inflate: 0.05 }
            box gem place(base_box=guard, anchor=[0, 0, 1], align=[0, 0, -1], offset=[0, 0, -0.25]) {
              size: [2, 1, 1], material: ruby
            }
            bone wing_l attach(parent_box=guard, anchor=[1, 0, 0], align=[-1, 0, 0]) mirror_x(wing_r) {
              rest_euler: [0.0, 0.0, 30.0]
              box quillon { size: [4, 1, 2], material: steel }
            }
          }
        }
        """
        scene = compile_geometry(parse_program(dsl))
        report = audit_scene(scene)
        self.assertTrue(report["passed"], f"Expected valid inset/hinge rig to pass: {report}")
        self.assertEqual(report["contact_violations"], [])
        self.assertEqual(len(report["physical_connected_components"]), 1)

    def test_authorized_detached_bone_vs_unauthorized_floating_box(self):
        """detached=true authorizes floating component; unattached origin-offset floating box fails."""
        authorized_dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { gold: ramp(base_lch=[0.80, 0.16, 85.0], steps=4, emissive=true) }
        @rig {
          root bone staff {
            pivot_local: [8.0, 8.0, 8.0]
            box shaft { size: [2, 8, 2], material: gold }
            bone halo attach(parent_box=shaft, anchor=[0, 1, 0], align=[0, -1, 0], offset=[0, 1.5, 0], detached=true) {
              box orb { size: [2, 2, 2], material: gold }
            }
          }
        }
        """
        auth_scene = compile_geometry(parse_program(authorized_dsl))
        auth_report = audit_scene(auth_scene)
        self.assertTrue(auth_report["passed"], f"Authorized detached bone should pass: {auth_report}")
        self.assertEqual(auth_report["contact_policy"]["authorized_detached_bones"], 1)
        self.assertEqual(len(auth_report["physical_connected_components"]), 2)

        unauthorized_dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { gold: ramp(base_lch=[0.80, 0.16, 85.0], steps=4) }
        @rig {
          root bone staff {
            pivot_local: [8.0, 8.0, 8.0]
            box shaft { size: [2, 8, 2], material: gold }
            box stray place(anchor=[0, 0, 0], align=[0, 0, 0], offset=[8.0, 0.0, 0.0]) {
              size: [2, 2, 2], material: gold
            }
          }
        }
        """
        unauth_scene = compile_geometry(parse_program(unauthorized_dsl))
        unauth_report = audit_scene(unauth_scene)
        self.assertFalse(unauth_report["passed"])
        self.assertEqual(unauth_report["contact_violations"][0]["violation"], "disconnected_physical_components")

    def test_world_aabb_broadphase_matches_exhaustive_sat_oracle(self):
        """Verify world-AABB broadphase skips zero true collisions or flush contacts."""
        for fam in ("straight_blade_sword", "tripod_telescope", "floating_ring_staff"):
            split = "test_unseen" if fam in UNSEEN_STATIC_FAMILIES else "train"
            row = verify_and_package_candidate(fam, split=split, variant_index=0)
            scene = compile_geometry(parse_program(row["canonical_dsl"]))
            report = audit_scene(scene)

            oracle_collisions = []
            for left, right in combinations(scene.boxes, 2):
                pair = tuple(sorted((left.name, right.name)))
                if pair in scene.adjacency:
                    continue
                res = sat_obb(left, right, collision_tolerance=1e-7)
                if res.collision:
                    oracle_collisions.append(list(pair))
            self.assertEqual([c["pair"] for c in report["collision_pairs"]], oracle_collisions)


class TestP108ComplexStaticGenerators(unittest.TestCase):
    """Verify P1-08-r3 / ADR 0005 55-family categorized generators, curriculum, and batch QC."""

    def test_all_55_families_meet_complexity_export_and_feature_contracts(self):
        """Every seen family (40) covers simple-to-complex tiers; every unseen family (15) has >=16 boxes."""
        self.assertEqual(len(SEEN_STATIC_FAMILIES), 40)
        self.assertEqual(len(UNSEEN_STATIC_FAMILIES), 15)
        observed_mirror = False
        observed_detached = False
        observed_java_profile = False
        observed_entity_body_plan = False
        observed_domains: set[str] = set()
        observed_tiers: set[str] = set()

        for fam in SEEN_STATIC_FAMILIES:
            for v_idx in range(5):
                row = verify_and_package_candidate(fam, split="train", variant_index=v_idx, run_exporter_check=(v_idx == 0))
                boxes = row["audit_version_and_results"]["boxes"]
                bones = row["audit_version_and_results"]["bones"]
                tier = row["generator_parameters"]["complexity_tier"]
                observed_tiers.add(tier)
                if v_idx < 3:
                    self.assertGreaterEqual(boxes, 12, f"{fam} v{v_idx} had only {boxes} boxes")
                    self.assertGreaterEqual(bones, 3, f"{fam} v{v_idx} had only {bones} bones")
                elif v_idx == 3:
                    self.assertGreaterEqual(boxes, 4, f"{fam} tier_1_simple v{v_idx} had {boxes} < 4 boxes")
                    self.assertLessEqual(boxes, 10, f"{fam} tier_1_simple v{v_idx} had {boxes} > 10 boxes")
                else:
                    self.assertGreaterEqual(boxes, 6, f"{fam} tier_2_medium v{v_idx} had {boxes} < 6 boxes")
                    self.assertLessEqual(boxes, 16, f"{fam} tier_2_medium v{v_idx} had {boxes} > 16 boxes")
                # Assistant completion (including trailing @animation_plan comment & repair header) must parse cleanly
                parsed_assist = parse_program(row["prompt_and_provenance"]["assistant_completion"])
                self.assertEqual(len(compile_geometry(parsed_assist).boxes), boxes)
                observed_domains.add(row["generator_parameters"]["domain"])
                if row["generator_parameters"]["has_mirror_x"]:
                    observed_mirror = True
                if row["generator_parameters"]["target_profile"] == "java_modern":
                    observed_java_profile = True
                if row["body_plan_id_if_dynamic"] is not None:
                    observed_entity_body_plan = True

        for fam in UNSEEN_STATIC_FAMILIES:
            for v_idx in range(3):
                row = verify_and_package_candidate(fam, split="test_unseen", variant_index=v_idx, run_exporter_check=(v_idx == 0))
                boxes = row["audit_version_and_results"]["boxes"]
                self.assertGreaterEqual(boxes, 16, f"Unseen {fam} v{v_idx} had {boxes} < 16 boxes")
                self.assertEqual(row["generator_parameters"]["quality_label"], "verified_clean")
                observed_domains.add(row["generator_parameters"]["domain"])
                if row["generator_parameters"]["has_detached_bone"]:
                    observed_detached = True
                prog = parse_program(row["canonical_dsl"])
                sc = compile_geometry(prog)
                atlas, _ = build_atlas(prog, sc)
                self.assertGreaterEqual(len(export_bbmodel(sc, name=fam, atlas=atlas)["elements"]), 16)
                if prog.meta.archetype != "entity":
                    self.assertGreaterEqual(len(export_java(sc, profile="java_modern", atlas=atlas)["elements"]), 16)

        self.assertEqual(observed_domains, set(MACRO_DOMAINS))
        self.assertEqual(observed_tiers, {"tier_1_simple", "tier_2_medium", "tier_3_complex", "tier_4_masterwork"})
        self.assertTrue(observed_mirror)
        self.assertTrue(observed_detached)
        self.assertTrue(observed_java_profile)
        self.assertTrue(observed_entity_body_plan)

    def test_pilot_200_preset_deduplication_isolation_diversity_and_uniform_schema(self):
        """Verify 260-row pilot_200 preset produces 100% unique hashes, 21 categories, 4 tiers, 5% repair mix, and uniform keys."""
        from synccraft.data_gen.taxonomy_catalog import ALL_CATEGORIES

        self.assertEqual(DATASET_PRESETS["full_60k"].total_seen, 60_000)
        self.assertEqual(DATASET_PRESETS["full_60k"].total_unseen, 1_200)
        self.assertEqual(DATASET_PRESETS["full_60k"].total_rows, 61_200)
        self.assertEqual(DATASET_PRESETS["full_75k"].total_seen, 75_000)
        self.assertEqual(DATASET_PRESETS["full_75k"].total_unseen, 2_400)
        self.assertEqual(DATASET_PRESETS["full_75k"].total_rows, 77_400)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "datasets"
            manifest = build_verified_dataset(out_dir, preset_name="pilot_200", max_workers=2)
            self.assertEqual(manifest["total_records"], 260)
            self.assertEqual(manifest["complexity_stats"]["unique_canonical_dsl_hashes"], 260)
            self.assertGreaterEqual(manifest["complexity_stats"]["min_boxes"], 4)
            self.assertLessEqual(manifest["complexity_stats"]["min_boxes"], 8)
            self.assertGreaterEqual(manifest["complexity_stats"]["max_boxes"], 25)

            self.assertEqual(len(ALL_CATEGORIES), 21)
            self.assertEqual(len(manifest["category_counts"]), len(ALL_CATEGORIES))
            self.assertEqual(
                set(manifest["complexity_tier_counts"].keys()),
                {"tier_1_simple", "tier_2_medium", "tier_3_complex", "tier_4_masterwork"},
            )
            self.assertEqual(
                set(manifest["assistant_task_mode_counts"].keys()),
                {"create_full_asset", "coarse_to_fine_upgrade", "texture_and_animate_rig", "diagnose_and_repair_defect"},
            )
            self.assertGreater(manifest["quality_label_counts"].get("controlled_defect_with_repair", 0), 0)
            self.assertIn("human_gathered_real_corpus", manifest["reserved_extension_tracks"])

            div = manifest["diversity_stats"]
            self.assertEqual(set(div["domain_counts"].keys()), set(MACRO_DOMAINS))
            self.assertEqual(len(div["naming_register_counts"]), 6)
            self.assertEqual(len(div["prompt_granularity_counts"]), 4)
            self.assertGreaterEqual(div["qd_shannon_entropy_bits"], 5.5)
            self.assertGreaterEqual(div["unique_ast_identifier_vocab_size"], 1500)
            self.assertGreaterEqual(div["stateful_materials_coverage_ratio"], 0.40)
            self.assertLessEqual(div["stateful_materials_coverage_ratio"], 0.85)

            rows = manifest["rows"]
            self.assertEqual(len(rows), 260)
            expected_gen_keys = set(rows[0]["generator_parameters"].keys())
            expected_audit_keys = set(rows[0]["audit_version_and_results"].keys())
            expected_prov_keys = set(rows[0]["prompt_and_provenance"].keys())

            for row in rows:
                for field in MANDATORY_ROW_FIELDS:
                    self.assertIn(field, row)
                self.assertEqual(set(row["generator_parameters"].keys()), expected_gen_keys)
                self.assertEqual(set(row["audit_version_and_results"].keys()), expected_audit_keys)
                self.assertEqual(set(row["prompt_and_provenance"].keys()), expected_prov_keys)
                if row["split"] in ("train", "val"):
                    self.assertIn(row["family_id"], SEEN_STATIC_FAMILIES)
                else:
                    self.assertIn(row["family_id"], UNSEEN_STATIC_FAMILIES)
                    self.assertGreaterEqual(row["audit_version_and_results"]["boxes"], 16)

    def test_batch_statistical_qc_and_visual_contact_sheet_pipeline(self):
        """Verify batch-by-batch SPC gates and 3D+2D visual inspection sheet rendering."""
        from PIL import Image
        from synccraft.data_gen.batch_pipeline import generate_dataset_batches

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "batched_ds"
            summary = generate_dataset_batches(
                out_dir,
                num_batches=2,
                train_per_seen_per_batch=1,
                val_per_seen_per_batch=1,
                test_per_unseen_per_batch=1,
                samples_to_render_per_batch=4,
            )
            self.assertTrue(summary["all_batches_accepted"])
            self.assertEqual(summary["total_records"], 2 * (40 * 2 + 15 * 1))
            self.assertEqual(len(summary["batches"]), 2)
            for b_rep in summary["batches"]:
                self.assertTrue(b_rep["batch_accepted"])
                self.assertEqual(b_rep["clean_sat_pass_rate"], 1.0)
                self.assertEqual(b_rep["animation_swept_sat_pass_rate"], 1.0)
                sheet_file = out_dir / b_rep["inspection_sheet"]
                self.assertTrue(sheet_file.is_file())
                with Image.open(sheet_file) as im:
                    self.assertGreater(im.size[0], 600)
                    self.assertGreater(im.size[1], 300)


if __name__ == "__main__":
    unittest.main()
