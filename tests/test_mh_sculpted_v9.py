"""Verification suite for ADR 0012 Sculpted Style Anchor & V9 Progressive Block-DSL v4.1 Masterworks."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from synccraft.audit.sculpted_style_evaluator import evaluate_sculpted_style_anchor
from synccraft.audit.sync_audit import audit_scene
from synccraft.backends.bedrock_gecko_exporter import export_bedrock
from synccraft.data_gen.mh_sculpted_v9.catalog import CANONICAL_22_ORDER, get_monster_spec
from synccraft.data_gen.mh_sculpted_v9.studio_compiler import DEFAULT_DSL_ROOT, inspect_or_compile_monster
from synccraft.dsl.ast_nodes import EdgeCutoutDecl
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.se3_chain import compile_geometry
from synccraft.shading.compiler import compute_plane_edge_alpha_mask

REPO_ROOT = Path(__file__).resolve().parents[1]
V9_PACK_ROOT = REPO_ROOT / "artifacts" / "Monster_Hunter_BlockDSL_Masterworks_v9"


class TestMonsterHunterSculptedV9Masterworks(unittest.TestCase):
    def test_01_sawtooth_width2_cutout_and_rotated_layered_plane_bedrock_parity(self) -> None:
        mask = compute_plane_edge_alpha_mask(
            10,
            12,
            EdgeCutoutDecl(mode="spikes", width=2, min_len=4, max_len=8, side="top", seed=21),
        )
        self.assertEqual(set(np.unique(mask).tolist()), {0, 255})
        # Alternating spike columns must differ rather than collapsing to a flat bar
        self.assertFalse(np.array_equal(mask[:, 0], mask[:, 1]))

        dsl = """
        @meta { dsl_version: "4.1", archetype: entity, target_profile: bbmodel_5, texel_density: 32, size_step: 0.5, audit_profile: sculpted }
        @palette { hide: ramp(base_lch=[0.50, 0.12, 200], steps=6, hue_shift=[20, -14]) }
        @rig {
          root bone root {
            pivot_local: [0, 4, 0]
            box core { size: [6, 8, 6], material: hide }
            plane tuft_a place(base_box=core, anchor=[0, 0.9, 0], align=[0, -0.8, 0]) {
              size: [5, 4], normal: z, material: hide, rotate: [22, -14, 10],
              edge: spikes(width=2, min_len=3, max_len=6, side=top, seed=7), layer: 2
            }
          }
        }
        """
        prog = parse_program(dsl)
        sc = compile_geometry(prog)
        self.assertTrue(audit_scene(sc)["passed"])
        bed = export_bedrock(sc, program=prog, profile="bedrock_entity")
        self.assertLessEqual(bed["serialized_vertex_max_error"], 1e-8)

    def test_02_sculpted_style_anchor_flags_doghouse_snout_and_unisheet_wings(self) -> None:
        bad_dsl = """
        @meta { dsl_version: "4.1", archetype: entity, target_profile: bbmodel_5, texel_density: 16, size_step: 0.5, audit_profile: sculpted }
        @palette { grey: ramp(base_lch=[0.5, 0.01, 200], steps=4, hue_shift=[0, 0]) }
        @rig {
          root bone root {
            pivot_local: [0, 3, 0]
            box skull { size: [6, 6, 6], material: grey }
            box snout_box place(base_box=skull, anchor=[0, 0.2, 1], align=[0, 0, -1]) {
              size: [6, 4, 6], material: grey
            }
            box jaw_box place(base_box=skull, anchor=[0, -0.6, 1], align=[0, 0, -1]) {
              size: [6, 2, 6], material: grey
            }
            bone wing_l attach(parent_box=skull, anchor=[1, 0.5, 0], align=[-1, 0, 0]) mirror_x(wing_r) {
              plane big_sheet { size: [14, 10], normal: y, material: grey }
            }
          }
        }
        """
        prog = parse_program(bad_dsl)
        sc = compile_geometry(prog)
        rep = evaluate_sculpted_style_anchor(prog, sc, require_wings_partitioned=True)
        self.assertFalse(rep.passed)
        diag_blob = " ".join(rep.diagnostics)
        self.assertIn("snout_doghouse_warning", diag_blob)
        self.assertIn("unisheet_wing_warning", diag_blob)

    def test_03_zinogre_progressive_exemplar_compiles_audits_and_exports_zero_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            man = inspect_or_compile_monster(
                "zinogre",
                dsl_root=DEFAULT_DSL_ROOT,
                output_pack_root=Path(tmp),
                stage=3,
                render_previews=False,
                render_gif=False,
                strict_audit=True,
            )
            self.assertEqual(man["id"], "zinogre")
            self.assertTrue(man["static_sculpted_audit_passed"])
            self.assertTrue(man["style_anchor_passed"])
            self.assertGreaterEqual(man["style_anchor_score"], 0.90)
            self.assertGreaterEqual(man["elements_total"], 245)
            self.assertGreater(man["stage2_boxes"], man["stage1_boxes"])
            self.assertGreater(man["elements_total"], man["stage2_boxes"])
            self.assertEqual(man["clip_count"], 12)
            self.assertEqual(man["serialized_bedrock_vertex_max_error"], 0.0)
            self.assertFalse(man["runtime_import_verified"])

    def test_04_all_authored_v9_monsters_have_distinct_progressive_stages_and_clean_splits(self) -> None:
        authored = [mid for mid in CANONICAL_22_ORDER if (DEFAULT_DSL_ROOT / mid / f"{mid}.blockdsl").is_file()]
        self.assertGreaterEqual(len(authored), 1)

        dsl_hashes: set[str] = set()
        for mid in authored:
            spec = get_monster_spec(mid)
            mdir = DEFAULT_DSL_ROOT / mid
            for fname in ("trait_sheet.md", "stage1_blockout.blockdsl", "stage2_secondary.blockdsl", f"{mid}.blockdsl"):
                self.assertTrue((mdir / fname).is_file(), f"{mid} missing {fname}")
            s3_txt = (mdir / f"{mid}.blockdsl").read_text(encoding="utf-8")
            self.assertNotIn(s3_txt, dsl_hashes, f"Duplicate Block-DSL source detected on {mid}")
            dsl_hashes.add(s3_txt)

            prog = parse_program(s3_txt)
            sc = compile_geometry(prog)
            aud = audit_scene(sc)
            self.assertTrue(aud["passed"], f"{mid} failed static sculpted audit: {aud['collision_pairs'][:3]} {aud['contact_violations'][:3]}")
            rep = evaluate_sculpted_style_anchor(prog, sc, require_wings_partitioned=spec.is_winged)
            self.assertTrue(rep.passed, f"{mid} failed style anchor ({rep.style_anchor_score}): {rep.diagnostics}")
            if spec.is_winged:
                self.assertGreaterEqual(rep.metrics["wing_bone_count"], 6, f"{mid} must partition wings across >= 6 wing bones")
            self.assertIsNotNone(prog.behavior, f"{mid} missing @behavior")
            self.assertGreaterEqual(len(prog.behavior.clips), 12, f"{mid} has < 12 clips")

        man_path = V9_PACK_ROOT / "dataset_manifest.json"
        if man_path.is_file():
            pack = json.loads(man_path.read_text(encoding="utf-8"))
            self.assertEqual(pack["monster_count"], 22)
            self.assertEqual(pack["total_animation_clips"], 264)
            self.assertEqual(pack["total_supervision_records"], 132)
            self.assertGreaterEqual(pack["mean_style_anchor_score"], 0.85)
            self.assertFalse(pack["commercial_weights_eligible"])
            self.assertFalse(pack["runtime_import_verified"])
            self.assertEqual(pack["license_track"], "research_nc_fan_reference")
            self.assertTrue((V9_PACK_ROOT / "previews" / "v9_collection_overview.jpg").is_file())
            self.assertTrue((V9_PACK_ROOT / "viewer" / "index.html").is_file())

            seen_families: dict[str, str] = {}
            tot_rows = 0
            for split_short in ("train", "val", "test_unseen"):
                pq_p = V9_PACK_ROOT / "parquet" / f"{split_short}.parquet"
                self.assertTrue(pq_p.is_file())
                rows = pq.read_table(pq_p).to_pylist()
                tot_rows += len(rows)
                for r in rows:
                    self.assertFalse(r["commercial_weights_eligible"])
                    fam = r["family_group_id"]
                    if fam in seen_families:
                        self.assertEqual(seen_families[fam], split_short)
                    seen_families[fam] = split_short
            self.assertEqual(tot_rows, 132)


if __name__ == "__main__":
    unittest.main()
