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
                # Comment-free assistant completion (using typed @behavior AST) must parse cleanly
                comp_text = row["prompt_and_provenance"]["assistant_completion"]
                self.assertNotIn("//", comp_text)
                self.assertNotIn("swept_sat_verified", comp_text)
                parsed_assist = parse_program(comp_text)
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
            self.assertEqual(
                set(summary["holdout_splits"]),
                {"test_unseen_params", "test_unseen", "test_compositional"},
            )
            for h_split in ("test_unseen_params", "test_unseen", "test_compositional"):
                self.assertGreater(summary["holdout_split_counts"][h_split], 0)
                self.assertTrue((out_dir / f"{h_split}.jsonl").is_file())
            self.assertIn("distinct_tree_shapes_by_family", summary)
            self.assertGreaterEqual(len(summary["distinct_tree_shapes_by_family"]), 55)


class TestCleanRoomZeroLeakageDiversityAndHoldouts(unittest.TestCase):
    """Strict regression guards for H1, H11, M2, M3, M13, M14, Dataset #6, and 3-holdout splits."""

    def test_zero_split_name_slug_or_borrowed_ip_leakage_across_all_65_families(self):
        """Every generated prompt across all 55 core + 10 Apex families must have 0 split/slug/IP leaks."""
        import re
        from synccraft.data_gen import (
            ALL_APEX_FAMILIES,
            APEX_SEEN_FAMILIES,
            APEX_UNSEEN_FAMILIES,
            generate_apex_complex_dataset,
        )
        from synccraft.data_gen.static_families import verify_and_package_compositional_candidate

        forbidden_pattern = re.compile(
            r"\b(?:held[-\s]?out|holdout|unseen|test_unseen|test_unseen_params|test_compositional|"
            r"seen_train|validation|apricorn|pok[eé]mon|cobblemon|monster\s+hunter)\b",
            re.IGNORECASE,
        )
        all_family_slugs = set(SEEN_STATIC_FAMILIES) | set(UNSEEN_STATIC_FAMILIES) | set(ALL_APEX_FAMILIES) | {
            "elemental_pocket_companion",
            "modular_apricorn_healing_station",
        }
        self.assertIn("elemental_companion_critter", APEX_SEEN_FAMILIES)
        self.assertIn("modular_botanical_healing_station", APEX_SEEN_FAMILIES)
        self.assertNotIn("elemental_pocket_companion", ALL_APEX_FAMILIES)
        self.assertNotIn("modular_apricorn_healing_station", ALL_APEX_FAMILIES)

        snake_case_slugs = {s for s in all_family_slugs if "_" in s}

        def _assert_clean_prompt(prompt_text: str, context: str) -> None:
            m = forbidden_pattern.search(prompt_text)
            if m is not None:
                self.fail(f"Forbidden leak {m.group(0)!r} in {context}: {prompt_text!r}")
            low = prompt_text.lower()
            for slug in snake_case_slugs:
                self.assertNotIn(slug.lower(), low, f"Raw family_id slug {slug!r} leaked in {context}: {prompt_text!r}")

        # 1. All 40 Seen + 15 Unseen core families across train, val, test_unseen, and test_unseen_params
        for fam in SEEN_STATIC_FAMILIES:
            for sp, v_idx in (("train", 0), ("train", 1), ("val", 0), ("test_unseen_params", 0)):
                row = verify_and_package_candidate(fam, split=sp, variant_index=v_idx)
                prov = row["prompt_and_provenance"]
                _assert_clean_prompt(prov["prompt"], f"{fam}/{sp}/v{v_idx}/prompt")
                _assert_clean_prompt(prov["assistant_prompt"], f"{fam}/{sp}/v{v_idx}/assistant_prompt")
                # Pinned specification check (Dataset #6): target box & bone counts pinned explicitly
                self.assertIn("boxes across", prov["prompt"])

        for fam in UNSEEN_STATIC_FAMILIES:
            for v_idx in (0, 1):
                row = verify_and_package_candidate(fam, split="test_unseen", variant_index=v_idx)
                prov = row["prompt_and_provenance"]
                _assert_clean_prompt(prov["prompt"], f"{fam}/test_unseen/v{v_idx}/prompt")
                _assert_clean_prompt(prov["assistant_prompt"], f"{fam}/test_unseen/v{v_idx}/assistant_prompt")
                self.assertIn("boxes across", prov["prompt"])

        # 2. Compositional holdout rows
        for p_idx in range(4):
            crow = verify_and_package_compositional_candidate(pairing_index=p_idx, variant_index=0)
            prov = crow["prompt_and_provenance"]
            _assert_clean_prompt(prov["prompt"], f"compositional/{p_idx}/prompt")
            _assert_clean_prompt(prov["assistant_prompt"], f"compositional/{p_idx}/assistant_prompt")

        # 3. All 10 Apex families (including progressive & critique turns + all 3 Apex holdouts)
        apex_bundle = generate_apex_complex_dataset(
            None,
            samples_per_seen_family=1,
            samples_per_unseen_family=1,
            seed=202699,
        )
        all_apex_rows = list(apex_bundle["rows"])
        for h_rows in apex_bundle["holdout_rows"].values():
            all_apex_rows.extend(h_rows)
        self.assertGreaterEqual(len(all_apex_rows), 30 + 24 + 12)
        for a_row in all_apex_rows:
            _assert_clean_prompt(a_row["user_prompt"], f"apex/{a_row['family_id']}/{a_row['split']}/{a_row['task_mode']}")
            self.assertNotIn("apricorn", a_row["assistant_dsl"].lower())
            self.assertNotIn("//", a_row["assistant_dsl"])
            self.assertIn("boxes across", a_row["user_prompt"])

    def test_comment_free_targets_and_typed_behavior_ast_supervision(self):
        """H11: Zero // @animation_plan or swept_sat_verified comments; articulated profiles carry @behavior AST."""
        observed_behavior_ast = 0
        for idx, fam in enumerate(SEEN_STATIC_FAMILIES[:16]):
            for v_idx in range(4):
                row = verify_and_package_candidate(fam, split="train", variant_index=idx * 4 + v_idx)
                comp = row["prompt_and_provenance"]["assistant_completion"]
                canon = row["canonical_dsl"]
                self.assertNotIn("//", comp)
                self.assertNotIn("//", canon)
                self.assertNotIn("@animation_plan", comp)
                self.assertNotIn("swept_sat_verified", comp)
                prog_comp = parse_program(comp)
                prog_canon = parse_program(canon)
                if prog_canon.meta.target_profile in ("bbmodel_5", "bedrock_entity", "geckolib"):
                    if prog_canon.behavior is not None and len(prog_canon.behavior.clips) > 0:
                        observed_behavior_ast += 1
                        self.assertIn("@behavior", comp)
                        self.assertIn("@behavior", canon)
                else:
                    self.assertIsNone(prog_comp.behavior)
                    self.assertIsNone(prog_canon.behavior)
        self.assertGreaterEqual(observed_behavior_ast, 20)

    def test_canonical_root_pivot_canonical_seeds_and_continuous_human_bridging(self):
        """M2 & M3: Root pivot_local == (0,0,0), canonical material seeds <= 16, continuous rest angles, offsets & detached."""
        from synccraft.data_gen import ALL_APEX_FAMILIES, build_apex_complex_program
        from synccraft.dsl.ast_nodes import Bone, Box

        discrete_24 = {round(k * 15.0, 4) for k in range(-24, 25)} | {22.5, -22.5, 67.5, -67.5}
        continuous_angles_seen = 0
        nonzero_offsets_seen = 0
        detached_seen = 0

        def _walk_bones(b: Bone, is_root: bool = False) -> None:
            nonlocal continuous_angles_seen, nonzero_offsets_seen, detached_seen
            if not is_root:
                for ang in b.rest_euler:
                    if abs(ang) > 1e-6 and round(ang, 4) not in discrete_24:
                        continuous_angles_seen += 1
                if b.attach is not None:
                    if b.attach.detached:
                        detached_seen += 1
                    if any(abs(v) > 1e-6 for v in b.attach.offset):
                        nonzero_offsets_seen += 1
            for it in b.items:
                if isinstance(it, Box) and it.place is not None:
                    if any(abs(v) > 1e-6 for v in it.place.offset):
                        nonzero_offsets_seen += 1
                elif isinstance(it, Bone):
                    _walk_bones(it, is_root=False)

        for fam in (*SEEN_STATIC_FAMILIES[:20], *UNSEEN_STATIC_FAMILIES[:8]):
            sp = "test_unseen" if fam in UNSEEN_STATIC_FAMILIES else "train"
            row = verify_and_package_candidate(fam, split=sp, variant_index=1)
            prog = parse_program(row["canonical_dsl"])
            self.assertEqual(prog.root.pivot_local, (0.0, 0.0, 0.0), f"{fam} had non-zero root pivot {prog.root.pivot_local}")
            if prog.materials:
                self.assertEqual(
                    {mat.seed for mat in prog.materials},
                    set(range(1, len(prog.materials) + 1)),
                )
                for mat in prog.materials:
                    self.assertLessEqual(mat.seed, 16)
            _walk_bones(prog.root, is_root=True)

        for fam in ALL_APEX_FAMILIES:
            aprog = build_apex_complex_program(fam, seed=202)
            self.assertEqual(aprog.root.pivot_local, (0.0, 0.0, 0.0), f"Apex {fam} had non-zero root pivot")
            self.assertEqual(
                {mat.seed for mat in aprog.materials},
                set(range(1, len(aprog.materials) + 1)),
            )
            for mat in aprog.materials:
                self.assertLessEqual(mat.seed, 16)
            _walk_bones(aprog.root, is_root=True)

        self.assertGreaterEqual(continuous_angles_seen, 25)
        self.assertGreaterEqual(nonzero_offsets_seen, 15)
        self.assertGreaterEqual(detached_seen, 3)

    def test_mechanical_crossbow_150_rows_yields_at_least_35_distinct_tree_shapes(self):
        """M14: 150 rows of `mechanical_crossbow` must produce >= 35 distinct bone-and-box tree shapes."""
        from synccraft.data_gen import compute_program_tree_shape

        distinct_shapes: set[str] = set()
        for v_idx in range(150):
            row = verify_and_package_candidate("mechanical_crossbow", split="train", variant_index=v_idx)
            self.assertTrue(row["audit_version_and_results"]["passed"])
            prog = parse_program(row["canonical_dsl"])
            sig = compute_program_tree_shape(prog)
            self.assertEqual(sig, row["generator_parameters"]["tree_shape_signature"])
            distinct_shapes.add(sig)

        self.assertGreaterEqual(
            len(distinct_shapes),
            35,
            f"mechanical_crossbow collapsed to {len(distinct_shapes)} < 35 tree shapes across 150 rows",
        )

    def test_three_holdout_splits_and_distinct_tree_shapes_in_manifest(self):
        """Verify `test_unseen_params`, `test_unseen`, `test_compositional` and `distinct_tree_shapes_by_family`."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "holdout_ds"
            manifest = build_verified_dataset(out_dir, preset_name="pilot_200", max_workers=2)
            self.assertEqual(
                set(manifest["holdout_splits"]),
                {"test_unseen_params", "test_unseen", "test_compositional"},
            )
            self.assertIn("distinct_tree_shapes_by_family", manifest)
            self.assertEqual(len(manifest["distinct_tree_shapes_by_family"]), 55)

            seen_train_hashes = {
                r["canonical_dsl_sha256"] for r in manifest["rows"] if r["split"] in ("train", "val")
            }
            for h_split in ("test_unseen_params", "test_unseen", "test_compositional"):
                self.assertGreater(manifest["holdout_split_counts"][h_split], 0)
                h_path = out_dir / f"{h_split}.jsonl"
                self.assertTrue(h_path.is_file())
                lines = [json.loads(line) for line in h_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertEqual(len(lines), manifest["holdout_split_counts"][h_split])
                for h_row in lines:
                    self.assertEqual(h_row["split"], h_split)
                    self.assertNotIn(h_row["canonical_dsl_sha256"], seen_train_hashes)
                    self.assertTrue(h_row["audit_version_and_results"]["passed"])
                    if h_split == "test_unseen_params":
                        self.assertIn(h_row["family_id"], SEEN_STATIC_FAMILIES)
                        self.assertEqual(h_row["generator_parameters"]["param_regime"], "heldout_extrapolation_v1")
                    elif h_split == "test_unseen":
                        self.assertIn(h_row["family_id"], UNSEEN_STATIC_FAMILIES)
                    elif h_split == "test_compositional":
                        self.assertIn(h_row["family_id"], SEEN_STATIC_FAMILIES)
                        self.assertTrue(h_row["generator_parameters"]["param_regime"].startswith("compositional:"))


if __name__ == "__main__":
    unittest.main()
