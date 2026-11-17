"""End-to-end verification suite for the 22-specimen V8 Monster Hunter Block-DSL v4.0 Hybrid Collection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pyarrow.parquet as pq

from synccraft.data_gen.mh_dragons_v8 import (
    ALL_V8_MONSTER_BUILDERS,
    CANONICAL_22_ORDER,
    compile_and_export_v8_monster,
)
from synccraft.dsl.assembly_ir import audit_assembly_contracts
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.se3_chain import compile_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = REPO_ROOT / "artifacts" / "Monster_Hunter_BlockDSL_Rebuild_v8"


class TestMonsterHunterV8HybridDataset(unittest.TestCase):
    def test_01_registry_has_all_22_canonical_and_variant_monsters(self) -> None:
        self.assertEqual(len(CANONICAL_22_ORDER), 22)
        self.assertEqual(set(ALL_V8_MONSTER_BUILDERS.keys()), set(CANONICAL_22_ORDER))

        for mid in CANONICAL_22_ORDER:
            doc = ALL_V8_MONSTER_BUILDERS[mid]()
            self.assertEqual(doc["id"], mid)
            elems = doc["elements"]
            n_total = len(elems)
            n_planes = sum(1 for e in elems if min(e["size"]) == 0.0)
            n_boxes = n_total - n_planes
            ratio = n_boxes / max(1, n_total)
            self.assertGreaterEqual(n_total, 260, f"{mid} element count {n_total} < 260")
            self.assertLessEqual(n_total, 1200, f"{mid} element count {n_total} > 1200")
            self.assertGreaterEqual(ratio, 0.80, f"{mid} 3D cuboid ratio {ratio:.2%} < 80%")
            self.assertGreaterEqual(len(doc["bones"]), 22, f"{mid} bone count too low")

            # Non-uni-sheet multi-bank wing contract for every winged wyvern/elder
            if any(b["name"] == "wing_l" for b in doc["bones"]):
                self.assertGreaterEqual(
                    len(doc.get("wing_articulation", [])),
                    6,
                    f"{mid} must define >= 6 non-uni-sheet articulated wing banks (3 per wing)",
                )

    def test_02_live_compile_and_export_parity_on_exemplar(self) -> None:
        doc = ALL_V8_MONSTER_BUILDERS["rathalos"]()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            man = compile_and_export_v8_monster(doc, out_dir, render_previews=False)
            self.assertEqual(man["id"], "rathalos")
            self.assertEqual(man["clip_count"], 12)
            self.assertGreaterEqual(man["cuboid_ratio"], 0.80)
            self.assertLess(man["compiler_transform_max_error"], 1e-4)
            self.assertLess(man["serialized_bedrock_vertex_max_error"], 1e-4)
            self.assertFalse(man["runtime_import_verified"])

            dsl_p = out_dir / "assets" / "rathalos" / "rathalos.blockdsl"
            s1_p = out_dir / "assets" / "rathalos" / "rathalos_stage1_blockout.blockdsl"
            bb_p = out_dir / "assets" / "rathalos" / "rathalos.bbmodel"
            geo_p = out_dir / "assets" / "rathalos" / "rathalos.geo.json"
            anim_p = out_dir / "assets" / "rathalos" / "rathalos.animation.json"
            tex_p = out_dir / "assets" / "rathalos" / "textures" / "albedo.png"
            for p in (dsl_p, s1_p, bb_p, geo_p, anim_p, tex_p):
                self.assertTrue(p.is_file(), f"Missing exported file {p}")

            prog = parse_program(dsl_p.read_text(encoding="utf-8"))
            sc = compile_geometry(prog)
            self.assertTrue(audit_assembly_contracts(prog, sc)["passed"])

    def test_03_packaged_v8_collection_and_family_splits(self) -> None:
        man_path = PACK_ROOT / "dataset_manifest.json"
        if not man_path.is_file():
            self.skipTest("Packaged V8 dataset artifact not yet built at artifacts/Monster_Hunter_BlockDSL_Rebuild_v8")

        pack = json.loads(man_path.read_text(encoding="utf-8"))
        self.assertEqual(pack["monster_count"], 22)
        self.assertEqual(pack["total_animation_clips"], 264)
        self.assertEqual(pack["total_supervision_records"], 110)
        self.assertGreaterEqual(pack["overall_cuboid_ratio"], 0.85)
        self.assertFalse(pack["commercial_weights_eligible"])
        self.assertFalse(pack["runtime_import_verified"])
        self.assertEqual(pack["license_track"], "research_nc_fan_reference")

        self.assertTrue((PACK_ROOT / "previews" / "v8_collection_overview.jpg").is_file())
        self.assertTrue((PACK_ROOT / "viewer" / "index.html").is_file())

        # Verify zero family-level leakage across train / val / test_unseen Parquet shards
        seen_families: dict[str, str] = {}
        total_rows = 0
        for split_short in ("train", "val", "test_unseen"):
            pq_p = PACK_ROOT / "parquet" / f"{split_short}.parquet"
            self.assertTrue(pq_p.is_file(), f"Missing {pq_p}")
            rows = pq.read_table(pq_p).to_pylist()
            self.assertGreater(len(rows), 0)
            total_rows += len(rows)
            for r in rows:
                self.assertFalse(r["commercial_weights_eligible"])
                fam = r["family_group_id"]
                if fam in seen_families:
                    self.assertEqual(
                        seen_families[fam],
                        split_short,
                        f"Family leakage detected for {fam}: {seen_families[fam]} vs {split_short}",
                    )
                seen_families[fam] = split_short
        self.assertEqual(total_rows, 110)


if __name__ == "__main__":
    unittest.main()
