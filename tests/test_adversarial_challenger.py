"""
Adversarial and Empirical Stress Test Suite for SyncCraft Colab Notebook Pipeline.

Challenger 2 Verification:
1. 13 mandatory manifest fields & strict schema/type validation
2. Strict 64-hex dataset_manifest_sha256 invariant & oracle recomputation
3. Unseen family isolation (strict exclusion of unseen families from train/val)
4. Gate G2 8 prerequisite checks & exact integer arithmetic 90% threshold (p * 10 < t * 9)
5. CostLedger monotonicity, non-negativity, and budget limits
6. Atomic checkpoint staging, crash-safety, and manifest hash integrity
"""

import copy
from fractions import Fraction
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, Dict, List, Set

import nbformat

from synccraft.evaluation.budget import CostLedger
from synccraft.evaluation.gate_g2 import REQUIRED_CHECKS, decide_gate

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "synccraft_end_to_end.ipynb"

SEEN_STATIC_FAMILIES: Set[str] = {
    "straight_blade_sword",
    "single_edge_blade",
    "battle_axe",
    "spear",
    "war_hammer",
    "simple_wood_bow",
    "furniture_table_chair_chest_cabinet",
    "gearbox",
    "weapon_3d",  # archetype family fallback from examples
}

UNSEEN_STATIC_FAMILIES: Set[str] = {
    "mechanical_crossbow",
    "hand_lantern",
    "tripod_telescope",
    "folding_windmill",
    "anvil_forge",
    "floating_ring_staff",
}

MANDATORY_MANIFEST_FIELDS = [
    "asset_id",
    "family_id",
    "body_plan_id_if_dynamic",
    "generator_revision",
    "generator_parameters",
    "canonical_dsl",
    "canonical_dsl_sha256",
    "prompt_and_provenance",
    "source_license_and_review_record",
    "split",
    "audit_version_and_results",
    "target_profile_and_version",
    "seed",
]


def execute_notebook_in_sandbox(temp_dir: Path) -> Dict[str, Any]:
    """Execute all cells of notebooks/synccraft_end_to_end.ipynb in a sandboxed directory."""
    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)

    orig_cwd = os.getcwd()
    orig_env = os.environ.copy()

    os.environ["SYNCCRAFT_SMOKE_TEST"] = "1"
    os.environ.pop("HF_TOKEN", None)

    # Mock Colab environment if not present
    if "google" not in sys.modules:
        google_mod = types.ModuleType("google")
        colab_mod = types.ModuleType("google.colab")
        userdata_mod = types.ModuleType("google.colab.userdata")
        userdata_mod.get = lambda key: os.environ.get(key)
        colab_mod.userdata = userdata_mod
        google_mod.colab = colab_mod
        sys.modules["google"] = google_mod
        sys.modules["google.colab"] = colab_mod
        sys.modules["google.colab.userdata"] = userdata_mod

    global_scope: Dict[str, Any] = {
        "__name__": "__main__",
        "__file__": str(NOTEBOOK_PATH),
    }

    try:
        os.chdir(temp_dir)
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            cleaned_lines = []
            for line in cell.source.splitlines():
                stripped = line.strip()
                if stripped.startswith("!") or stripped.startswith("%"):
                    cleaned_lines.append(f"# {line}")
                else:
                    cleaned_lines.append(line)
            cleaned_code = "\n".join(cleaned_lines)
            compiled = compile(cleaned_code, f"<notebook_cell_{idx}>", "exec")
            exec(compiled, global_scope)
    finally:
        os.chdir(orig_cwd)
        os.environ.clear()
        os.environ.update(orig_env)

    return global_scope


class TestChallengerAdversarialManifest(unittest.TestCase):
    """Area 1 & 2: Adversarial Stress Testing of Dataset Manifest & 64-hex SHA-256."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = Path(cls.temp_dir_obj.name)
        cls.global_scope = execute_notebook_in_sandbox(cls.temp_dir)
        cls.manifest_path = cls.temp_dir / "datasets" / "dataset_manifest.json"
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest_data = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir_obj.cleanup()

    def test_01_all_13_mandatory_fields_present_in_every_row(self):
        """Verify that all 13 mandatory fields exist in 100% of generated manifest rows."""
        rows = self.manifest_data.get("rows", [])
        self.assertGreater(len(rows), 0, "Manifest rows should not be empty")

        for i, row in enumerate(rows):
            for field in MANDATORY_MANIFEST_FIELDS:
                self.assertIn(
                    field,
                    row,
                    f"Row #{i} (asset_id={row.get('asset_id')}) missing mandatory field '{field}'",
                )
            self.assertEqual(
                len(MANDATORY_MANIFEST_FIELDS),
                13,
                "Requirement specifies exactly 13 mandatory manifest fields",
            )

    def test_02_strict_type_validation_on_all_fields(self):
        """Verify strict type invariants for all 13 fields across all rows."""
        rows = self.manifest_data.get("rows", [])
        for i, row in enumerate(rows):
            # 1. asset_id: non-empty string
            self.assertIsInstance(row["asset_id"], str, f"Row #{i} asset_id is not str")
            self.assertGreater(len(row["asset_id"]), 0, f"Row #{i} asset_id is empty")

            # 2. family_id: non-empty string
            self.assertIsInstance(row["family_id"], str, f"Row #{i} family_id is not str")
            self.assertGreater(len(row["family_id"]), 0, f"Row #{i} family_id is empty")

            # 3. body_plan_id_if_dynamic: Optional[str]
            self.assertTrue(
                row["body_plan_id_if_dynamic"] is None or isinstance(row["body_plan_id_if_dynamic"], str),
                f"Row #{i} body_plan_id_if_dynamic must be None or str",
            )

            # 4. generator_revision: non-empty string
            self.assertIsInstance(row["generator_revision"], str, f"Row #{i} generator_revision is not str")
            self.assertGreater(len(row["generator_revision"]), 0)

            # 5. generator_parameters: dict
            self.assertIsInstance(row["generator_parameters"], dict, f"Row #{i} generator_parameters is not dict")

            # 6. canonical_dsl: non-empty string
            self.assertIsInstance(row["canonical_dsl"], str, f"Row #{i} canonical_dsl is not str")
            self.assertGreater(len(row["canonical_dsl"]), 0)

            # 7. canonical_dsl_sha256: 64 hex characters
            self.assertIsInstance(row["canonical_dsl_sha256"], str)
            self.assertRegex(row["canonical_dsl_sha256"], r"^[a-fA-F0-9]{64}$")

            # 8. prompt_and_provenance: dict or str
            self.assertTrue(
                isinstance(row["prompt_and_provenance"], (dict, str)),
                f"Row #{i} prompt_and_provenance must be dict or str",
            )

            # 9. source_license_and_review_record: dict or str
            self.assertTrue(
                isinstance(row["source_license_and_review_record"], (dict, str)),
                f"Row #{i} source_license_and_review_record must be dict or str",
            )

            # 10. split: valid split category
            self.assertIn(row["split"], {"train", "val", "test_unseen"})

            # 11. audit_version_and_results: dict with passed=True
            self.assertIsInstance(row["audit_version_and_results"], dict)
            self.assertIn("passed", row["audit_version_and_results"])
            self.assertTrue(row["audit_version_and_results"]["passed"])

            # 12. target_profile_and_version: non-empty string
            self.assertIsInstance(row["target_profile_and_version"], str)
            self.assertGreater(len(row["target_profile_and_version"]), 0)

            # 13. seed: integer (not boolean)
            self.assertIsInstance(row["seed"], int)
            self.assertNotIsInstance(row["seed"], bool)

    def test_03_canonical_dsl_sha256_cryptographic_oracle_check(self):
        """Adversarial oracle check: recompute SHA-256 on canonical_dsl and verify exact match."""
        for i, row in enumerate(self.manifest_data.get("rows", [])):
            computed_hash = hashlib.sha256(row["canonical_dsl"].encode("utf-8")).hexdigest()
            self.assertEqual(
                computed_hash,
                row["canonical_dsl_sha256"],
                f"Cryptographic hash mismatch at row #{i} (asset_id={row['asset_id']})",
            )

    def test_04_dataset_manifest_sha256_strict_64_hex_invariant(self):
        """Verify top-level dataset_manifest_sha256 strictly matches 64 hex characters."""
        sha = self.manifest_data.get("dataset_manifest_sha256")
        self.assertIsInstance(sha, str)
        self.assertEqual(len(sha), 64, f"Hash length is {len(sha)}, expected exactly 64")
        self.assertRegex(sha, r"^[a-fA-F0-9]{64}$", "Hash must contain only hex characters")

    def test_05_dataset_manifest_sha256_reproducibility_oracle(self):
        """Verify top-level SHA-256 matches independent computation over manifest rows."""
        rows = self.manifest_data["rows"]
        expected_hash = hashlib.sha256(
            json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self.assertEqual(self.manifest_data["dataset_manifest_sha256"], expected_hash)

    def test_06_split_jsonl_files_match_manifest_rows(self):
        """Verify train.jsonl, val.jsonl, test_unseen.jsonl rows match manifest rows exactly."""
        datasets_dir = self.temp_dir / "datasets"
        jsonl_counts = 0
        for split_name in ("train", "val", "test_unseen"):
            p = datasets_dir / f"{split_name}.jsonl"
            self.assertTrue(p.is_file(), f"Split file missing: {p}")
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.assertEqual(r["split"], split_name)
                        for field in MANDATORY_MANIFEST_FIELDS:
                            self.assertIn(field, r)
                        jsonl_counts += 1
        self.assertEqual(jsonl_counts, len(self.manifest_data["rows"]))


class TestChallengerAdversarialFamilyIsolation(unittest.TestCase):
    """Area 3: Unseen Family Isolation Stress Testing."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = Path(cls.temp_dir_obj.name)
        cls.global_scope = execute_notebook_in_sandbox(cls.temp_dir)
        cls.manifest_path = cls.temp_dir / "datasets" / "dataset_manifest.json"
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest_data = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir_obj.cleanup()

    def test_01_unseen_families_strictly_excluded_from_train_and_val(self):
        """Assert zero leakage of unseen families into 'train' or 'val' splits."""
        rows = self.manifest_data.get("rows", [])
        train_rows = [r for r in rows if r["split"] == "train"]
        val_rows = [r for r in rows if r["split"] == "val"]
        test_rows = [r for r in rows if r["split"] == "test_unseen"]

        self.assertGreater(len(train_rows), 0, "Train split should have rows")
        self.assertGreater(len(test_rows), 0, "Test unseen split should have rows")

        for r in train_rows:
            self.assertNotIn(
                r["family_id"],
                UNSEEN_STATIC_FAMILIES,
                f"Data leakage detected! Unseen family '{r['family_id']}' found in train split (asset: {r['asset_id']})",
            )
            self.assertIn(
                r["family_id"],
                SEEN_STATIC_FAMILIES,
                f"Unknown family '{r['family_id']}' in train split",
            )

        for r in val_rows:
            self.assertNotIn(
                r["family_id"],
                UNSEEN_STATIC_FAMILIES,
                f"Data leakage detected! Unseen family '{r['family_id']}' found in val split (asset: {r['asset_id']})",
            )
            self.assertIn(
                r["family_id"],
                SEEN_STATIC_FAMILIES,
                f"Unknown family '{r['family_id']}' in val split",
            )

        for r in test_rows:
            self.assertIn(
                r["family_id"],
                UNSEEN_STATIC_FAMILIES,
                f"Held-out test row has seen family: '{r['family_id']}'",
            )

    def test_02_train_and_unseen_family_sets_strictly_disjoint(self):
        """Assert formal disjointness between train family set and unseen family set."""
        train_families = {r["family_id"] for r in self.manifest_data["rows"] if r["split"] == "train"}
        test_families = {r["family_id"] for r in self.manifest_data["rows"] if r["split"] == "test_unseen"}

        intersection = train_families.intersection(test_families)
        self.assertEqual(
            len(intersection),
            0,
            f"Isolation violation! Overlapping families between train and test: {intersection}",
        )
        self.assertTrue(train_families.isdisjoint(UNSEEN_STATIC_FAMILIES))

    def test_03_adversarial_leakage_detector_catches_synthetic_contamination(self):
        """Stress-test an isolation auditor against synthetic malicious contamination."""
        clean_rows = copy.deepcopy(self.manifest_data["rows"])
        contaminated_rows = copy.deepcopy(clean_rows)

        # Inject an unseen family into a train row
        contaminated_rows.append({
            "asset_id": "malicious_injected_sample",
            "family_id": "mechanical_crossbow",  # UNSEEN family
            "split": "train",                     # Contaminated split!
        })

        def audit_isolation(rows: List[Dict[str, Any]]) -> List[str]:
            violations = []
            for item in rows:
                if item.get("split") == "train" and item.get("family_id") in UNSEEN_STATIC_FAMILIES:
                    violations.append(f"Leakage: {item['asset_id']} ({item['family_id']}) in train")
            return violations

        self.assertEqual(len(audit_isolation(clean_rows)), 0)
        violations = audit_isolation(contaminated_rows)
        self.assertEqual(len(violations), 1)
        self.assertIn("malicious_injected_sample", violations[0])


class TestChallengerAdversarialGateG2(unittest.TestCase):
    """Area 4: Gate G2 Prerequisites & 90% Threshold Arithmetic Stress Testing."""

    def _get_valid_evidence(self) -> Dict[str, Any]:
        return {
            "full_static_audit_implemented": True,
            "heldout_split_frozen_before_training": True,
            "all_samples_have_at_least_16_parts": True,
            "repair_pgd_policy_preregistered": True,
            "matched_raw_budgets_enforced": True,
            "at_most_three_reaudits_per_sample": True,
            "failures_and_timeouts_counted": True,
            "baseline_implementations_verified": True,
            "dataset_manifest_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "evaluation_manifest_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            "total": 50,
            "passed": 45,  # 90.0% -> skip_learned_refiner
        }

    def test_01_all_8_prerequisites_individually_enforced(self):
        """Stress test: flipping any single prerequisite of the 8 must block the gate."""
        for check in REQUIRED_CHECKS:
            for invalid_val in (False, None, "True", 0, 1):
                evidence = self._get_valid_evidence()
                evidence[check] = invalid_val
                res = decide_gate(evidence)
                self.assertEqual(
                    res["decision"],
                    "blocked",
                    f"Check '{check}' with invalid value '{invalid_val}' was not blocked!",
                )
                self.assertFalse(res["learned_refiner_enabled"])
                self.assertIn(
                    check,
                    res["missing_or_invalid_evidence"],
                    f"Check '{check}' not reported in missing_or_invalid_evidence",
                )

            # Also test when check is missing entirely
            evidence = self._get_valid_evidence()
            del evidence[check]
            res = decide_gate(evidence)
            self.assertEqual(res["decision"], "blocked")
            self.assertIn(check, res["missing_or_invalid_evidence"])

    def test_02_manifest_sha256_adversarial_patterns_blocked(self):
        """Stress test malformed SHA-256 strings against decide_gate."""
        valid_evidence = self._get_valid_evidence()
        malformed_hashes = [
            "a" * 63,           # 63 hex (too short)
            "a" * 65,           # 65 hex (too long)
            "g" * 64,           # 64 non-hex ('g' invalid)
            "0" * 63 + "!",     # punctuation
            " " + "a" * 64,     # leading space
            "a" * 64 + "\n",    # trailing newline
            "",                 # empty string
            None,               # None
            1234567890,         # int
            ["a" * 64],         # list
        ]

        for bad_hash in malformed_hashes:
            for field in ("dataset_manifest_sha256", "evaluation_manifest_sha256"):
                evidence = copy.deepcopy(valid_evidence)
                evidence[field] = bad_hash
                res = decide_gate(evidence)
                self.assertEqual(
                    res["decision"],
                    "blocked",
                    f"Malformed hash '{bad_hash}' for field '{field}' was not blocked!",
                )
                self.assertIn(field, res["missing_or_invalid_evidence"])

    def test_03_sample_counts_boundaries_and_adversarial_types(self):
        """Stress test sample counts: non-int, negative, passed > total, zero total."""
        valid_evidence = self._get_valid_evidence()
        invalid_counts = [
            (0, 0),        # total <= 0
            (-1, 0),       # total negative
            (50, -1),      # passed negative
            (50, 51),      # passed > total
            (50.0, 45),    # float total
            (50, 45.0),    # float passed
            ("50", 45),    # str total
            (50, "45"),    # str passed
            (None, 45),    # None total
            (50, None),    # None passed
        ]

        for total, passed in invalid_counts:
            evidence = copy.deepcopy(valid_evidence)
            evidence["total"] = total
            evidence["passed"] = passed
            res = decide_gate(evidence)
            self.assertEqual(
                res["decision"],
                "blocked",
                f"Invalid count total={total}, passed={passed} was not blocked!",
            )
            self.assertIn("valid_sample_counts", res["missing_or_invalid_evidence"])

    def test_04_threshold_arithmetic_exact_90_percent_boundary(self):
        """
        Verify the exact 90% boundary arithmetic: p * 10 < t * 9
        Per AGENTS.md:
        - At or above 90%: skip_learned_refiner (enabled = False)
        - Strictly below 90%: eligible_for_learned_refiner (enabled = True)
        """
        evidence = self._get_valid_evidence()

        # Case 1: Exactly 90.0% (p=9, t=10) -> 9*10 < 10*9 is 90 < 90 (False) -> skip
        evidence["total"], evidence["passed"] = 10, 9
        res = decide_gate(evidence)
        self.assertFalse(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "skip_learned_refiner")
        self.assertEqual(res["pass_rate"], 0.9)

        # Case 2: Exactly 90.0% (p=45, t=50) -> 450 < 450 (False) -> skip
        evidence["total"], evidence["passed"] = 50, 45
        res = decide_gate(evidence)
        self.assertFalse(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "skip_learned_refiner")
        self.assertEqual(res["pass_rate"], 0.9)

        # Case 3: Exactly 90.0% (p=90, t=100) -> 900 < 900 (False) -> skip
        evidence["total"], evidence["passed"] = 100, 90
        res = decide_gate(evidence)
        self.assertFalse(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "skip_learned_refiner")

        # Case 4: Strictly below 90% (p=44, t=50 -> 88%) -> 440 < 450 (True) -> eligible
        evidence["total"], evidence["passed"] = 50, 44
        res = decide_gate(evidence)
        self.assertTrue(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "eligible_for_learned_refiner")
        self.assertEqual(res["pass_rate"], 0.88)

        # Case 5: Strictly below 90% (p=89, t=100 -> 89%) -> 890 < 900 (True) -> eligible
        evidence["total"], evidence["passed"] = 100, 89
        res = decide_gate(evidence)
        self.assertTrue(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "eligible_for_learned_refiner")

        # Case 6: Strictly above 90% (p=46, t=50 -> 92%) -> 460 < 450 (False) -> skip
        evidence["total"], evidence["passed"] = 50, 46
        res = decide_gate(evidence)
        self.assertFalse(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "skip_learned_refiner")

        # Case 7: 100% pass rate (p=50, t=50) -> skip
        evidence["total"], evidence["passed"] = 50, 50
        res = decide_gate(evidence)
        self.assertFalse(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "skip_learned_refiner")

        # Case 8: 0% pass rate (p=0, t=50) -> eligible
        evidence["total"], evidence["passed"] = 50, 0
        res = decide_gate(evidence)
        self.assertTrue(res["learned_refiner_enabled"])
        self.assertEqual(res["decision"], "eligible_for_learned_refiner")

    def test_05_integer_arithmetic_matches_exact_fractions_oracle(self):
        """Property test: integer formula (p*10 < t*9) matches Fraction(p, t) < 9/10 exactly."""
        threshold_fraction = Fraction(9, 10)
        test_totals = [1, 2, 5, 7, 10, 13, 20, 37, 50, 99, 100, 200, 333, 1000]

        for total in test_totals:
            for passed in range(total + 1):
                formula_enabled = (passed * 10 < total * 9)
                fraction_enabled = (Fraction(passed, total) < threshold_fraction)
                self.assertEqual(
                    formula_enabled,
                    fraction_enabled,
                    f"Discrepancy at passed={passed}, total={total}: formula={formula_enabled}, fraction={fraction_enabled}",
                )


class TestChallengerAdversarialCostLedger(unittest.TestCase):
    """Area 5: CostLedger Monotonicity and Non-Negativity Stress Testing."""

    def test_01_all_counters_initialized_to_zero_integers(self):
        """Verify initial counters are strictly zero integers."""
        ledger = CostLedger()
        for field in (
            "accepted_tokens",
            "discarded_tokens",
            "repair_tokens",
            "projection_steps",
            "compiler_calls",
            "initial_audit_calls",
            "reaudit_calls",
        ):
            val = getattr(ledger, field)
            self.assertEqual(val, 0)
            self.assertIsInstance(val, int)
            self.assertNotIsInstance(val, bool)

    def test_02_negative_or_float_initialization_rejected(self):
        """Verify that negative integers or floats in constructor are rejected."""
        with self.assertRaises(ValueError):
            CostLedger(accepted_tokens=-1)
        with self.assertRaises(ValueError):
            CostLedger(accepted_tokens=1.5)
        with self.assertRaises(ValueError):
            CostLedger(compiler_calls=-10)

    def test_03_frozen_immutability_enforced(self):
        """Verify direct attribute assignment raises FrozenInstanceError."""
        ledger = CostLedger()
        with self.assertRaises(FrozenInstanceError):
            ledger.accepted_tokens = 5

    def test_04_record_rejects_negative_amounts_strictly(self):
        """Verify that negative amounts in record() raise ValueError (no refunds)."""
        ledger = CostLedger()
        with self.assertRaises(ValueError) as ctx:
            ledger.record("accepted_tokens", -1)
        self.assertIn("rollback cannot refund cost", str(ctx.exception))

        with self.assertRaises(ValueError):
            ledger.record("compiler_calls", -100)

    def test_05_record_rejects_non_integer_amounts(self):
        """Verify record() rejects floats and strings."""
        ledger = CostLedger()
        with self.assertRaises(ValueError):
            ledger.record("accepted_tokens", 2.5)
        with self.assertRaises(ValueError):
            ledger.record("accepted_tokens", 0.0)
        with self.assertRaises(ValueError):
            ledger.record("accepted_tokens", "10")

    def test_06_record_rejects_unknown_counters(self):
        """Verify record() rejects undefined counter dimensions."""
        ledger = CostLedger()
        with self.assertRaises(ValueError) as ctx:
            ledger.record("uncalibrated_flops", 1000)
        self.assertIn("unknown operation counter", str(ctx.exception))

    def test_07_counters_are_strictly_monotonic_and_non_decreasing(self):
        """Verify sequence of valid record calls produces non-decreasing counters."""
        ledger = CostLedger()
        increments = [5, 12, 0, 8, 100, 3]
        prev = 0

        for inc in increments:
            ledger.record("accepted_tokens", inc)
            curr = ledger.accepted_tokens
            self.assertGreaterEqual(curr, prev, f"Counter decreased from {prev} to {curr}")
            self.assertGreaterEqual(curr, 0, f"Counter went negative: {curr}")
            prev = curr

        self.assertEqual(ledger.accepted_tokens, sum(increments))

    def test_08_within_budget_enforcement(self):
        """Verify budget limit compliance and boundary checking."""
        ledger = CostLedger()
        ledger.record("accepted_tokens", 50)
        ledger.record("compiler_calls", 10)

        # Within budget
        self.assertTrue(ledger.within({"accepted_tokens": 50, "compiler_calls": 10}))
        self.assertTrue(ledger.within({"accepted_tokens": 100, "compiler_calls": 20}))

        # Exceeds budget
        self.assertFalse(ledger.within({"accepted_tokens": 49}))
        self.assertFalse(ledger.within({"compiler_calls": 9}))

        # Invalid limit specifications
        with self.assertRaises(ValueError):
            ledger.within({"accepted_tokens": -1})
        with self.assertRaises(ValueError):
            ledger.within({"unknown_limit": 10})


class TestChallengerAdversarialAtomicCheckpointing(unittest.TestCase):
    """Area 6: Atomic Checkpoint Staging, Crash Safety & Hash Integrity Stress Testing."""

    def setUp(self):
        self.test_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.test_dir_obj.name)
        self.checkpoints_dir = self.test_dir / "checkpoints"
        self.datasets_dir = self.test_dir / "datasets"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy dataset manifest
        manifest_data = {"schema": "synccraft.dataset-manifest/0.1", "dataset_manifest_sha256": "e" * 64}
        (self.datasets_dir / "dataset_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    def tearDown(self):
        self.test_dir_obj.cleanup()

    def _save_atomic_checkpoint_under_test(
        self,
        step: int,
        manifest_sha: str,
        fail_at_step: str = None,
    ) -> Path:
        """Isolated implementation of save_atomic_checkpoint with fault-injection hooks."""
        final_dir = self.checkpoints_dir / f"checkpoint-{step}"
        temp_dir = Path(tempfile.mkdtemp(prefix=".tmp_ckpt_", dir=self.checkpoints_dir))

        try:
            if fail_at_step == "before_adapter":
                raise IOError("Simulated IO failure before adapter write")

            # 1. Save adapter configuration
            adapter_cfg = {"peft_type": "LORA", "r": 32, "lora_alpha": 64}
            (temp_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2), encoding="utf-8")

            if fail_at_step == "during_weights":
                raise MemoryError("Simulated OOM during weight serialization")

            # 2. Save model adapter weights
            (temp_dir / "adapter_model.bin").write_bytes(b"MOCK_WEIGHTS_BINARY_DATA")
            (temp_dir / "adapter_model.safetensors").write_bytes(b"LORA_SAFETENSORS_PAYLOAD")
            (temp_dir / "optimizer.pt").write_bytes(b"MOCK_OPTIMIZER_DATA")

            # 3. Save tokenizer metadata
            (temp_dir / "tokenizer_config.json").write_text(json.dumps({"cls": "MockTokenizer"}), encoding="utf-8")
            (temp_dir / "special_tokens_map.json").write_text(json.dumps({"eos": "<|end|>"}, indent=2), encoding="utf-8")

            # 4. Copy dataset manifest reference
            shutil.copy(self.datasets_dir / "dataset_manifest.json", temp_dir / "dataset_manifest.json")

            if fail_at_step == "before_manifest":
                raise RuntimeError("Simulated crash right before manifest commit")

            # 5. Generate checkpoint manifest with file SHA-256 hashes
            file_hashes = {}
            for f in sorted(temp_dir.iterdir()):
                if f.is_file():
                    file_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

            manifest_meta = {
                "step": step,
                "loss": 0.125,
                "dataset_manifest_sha256": manifest_sha,
                "file_hashes": file_hashes,
            }
            (temp_dir / "manifest.json").write_text(json.dumps(manifest_meta, indent=2), encoding="utf-8")

            # 6. Atomic directory move
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temp_dir.rename(final_dir)
            return final_dir
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_01_atomic_staging_in_same_filesystem_directory(self):
        """Verify temp staging dir prefix and parent directory match CHECKPOINTS_DIR."""
        ckpt_dir = self._save_atomic_checkpoint_under_test(step=10, manifest_sha="e" * 64)
        self.assertTrue(ckpt_dir.is_dir())
        self.assertEqual(ckpt_dir.parent, self.checkpoints_dir)

        # Verify no orphaned .tmp_ckpt_* directories remain
        orphaned = list(self.checkpoints_dir.glob(".tmp_ckpt_*"))
        self.assertEqual(len(orphaned), 0, f"Found leaked staging directories: {orphaned}")

    def test_02_crash_simulation_prevents_partial_checkpoint(self):
        """Adversarial crash injection: failure mid-write must leave NO corrupt checkpoint dir."""
        for fail_point in ("before_adapter", "during_weights", "before_manifest"):
            step = 99
            final_target = self.checkpoints_dir / f"checkpoint-{step}"

            with self.assertRaises((IOError, MemoryError, RuntimeError)):
                self._save_atomic_checkpoint_under_test(
                    step=step,
                    manifest_sha="e" * 64,
                    fail_at_step=fail_point,
                )

            # Target directory must NOT exist (no half-written corrupt checkpoints)
            self.assertFalse(
                final_target.exists(),
                f"Crash at '{fail_point}' leaked a partial directory at {final_target}!",
            )

            # Staging directory must be cleaned up in finally block
            orphaned = list(self.checkpoints_dir.glob(".tmp_ckpt_*"))
            self.assertEqual(
                len(orphaned),
                0,
                f"Crash at '{fail_point}' left behind orphaned staging dir: {orphaned}",
            )

    def test_03_file_hashes_in_manifest_match_disk_contents(self):
        """Verify file SHA-256 hashes recorded in manifest.json match disk files bit-for-bit."""
        ckpt_dir = self._save_atomic_checkpoint_under_test(step=20, manifest_sha="f" * 64)
        manifest_file = ckpt_dir / "manifest.json"
        self.assertTrue(manifest_file.is_file())

        with open(manifest_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("file_hashes", meta)
        file_hashes = meta["file_hashes"]

        for filename, expected_hash in file_hashes.items():
            disk_file = ckpt_dir / filename
            self.assertTrue(disk_file.is_file(), f"File {filename} listed in manifest missing on disk")
            actual_hash = hashlib.sha256(disk_file.read_bytes()).hexdigest()
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}",
            )

    def test_04_idempotent_overwrite_of_existing_checkpoint_step(self):
        """Verify saving over an existing checkpoint step succeeds cleanly without leaving temp dirs."""
        step = 42
        ckpt_1 = self._save_atomic_checkpoint_under_test(step=step, manifest_sha="1" * 64)
        self.assertTrue((ckpt_1 / "manifest.json").is_file())

        # Overwrite same step with updated manifest hash
        ckpt_2 = self._save_atomic_checkpoint_under_test(step=step, manifest_sha="2" * 64)
        self.assertEqual(ckpt_1, ckpt_2)
        with open(ckpt_2 / "manifest.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["dataset_manifest_sha256"], "2" * 64)

        orphaned = list(self.checkpoints_dir.glob(".tmp_ckpt_*"))
        self.assertEqual(len(orphaned), 0)


class TestChallengerNotebookEndToEndArtifactAudits(unittest.TestCase):
    """End-to-End Artifact Audits on Sandboxed Notebook Execution."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir_obj = tempfile.TemporaryDirectory()
        cls.temp_dir = Path(cls.temp_dir_obj.name)
        cls.global_scope = execute_notebook_in_sandbox(cls.temp_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir_obj.cleanup()

    def test_01_cost_ledger_in_executed_notebook_monotonic_and_populated(self):
        """Verify that CostLedger in the executed notebook was updated monotonically and accurately."""
        ledger = self.global_scope.get("cost_ledger")
        self.assertIsInstance(ledger, CostLedger)
        self.assertGreater(ledger.accepted_tokens, 0)
        self.assertEqual(ledger.compiler_calls, 6)
        self.assertEqual(ledger.initial_audit_calls, 6)
        self.assertEqual(ledger.discarded_tokens, 0)
        self.assertEqual(ledger.repair_tokens, 0)
        self.assertEqual(ledger.total_generated_tokens, ledger.accepted_tokens)

    def test_02_gate_g2_report_persisted_and_valid(self):
        """Verify runs/gate_g2_report.json is correctly saved and contains valid Gate G2 decision."""
        report_path = self.temp_dir / "runs" / "gate_g2_report.json"
        self.assertTrue(report_path.is_file(), f"Report file not found at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("evidence", data)
        self.assertIn("result", data)
        evidence = data["evidence"]
        result = data["result"]

        self.assertEqual(evidence["total"], 6)
        self.assertIsInstance(evidence["passed"], int)
        self.assertTrue(
            0 <= evidence["passed"] <= 6,
            f"evidence['passed'] ({evidence.get('passed')}) must be an integer between 0 and 6",
        )
        self.assertRegex(evidence["dataset_manifest_sha256"], r"^[a-fA-F0-9]{64}$")
        self.assertRegex(evidence["evaluation_manifest_sha256"], r"^[a-fA-F0-9]{64}$")

        # Dynamically validate genuine empirical outcomes against reference specification oracle
        expected_gate_result = decide_gate(evidence)
        self.assertIn(
            result["decision"],
            ("blocked", "skip_learned_refiner", "eligible_for_learned_refiner"),
            f"Gate G2 decision must be one of standard decisions, got: {result.get('decision')}",
        )
        self.assertEqual(
            result["decision"],
            expected_gate_result["decision"],
            f"Persisted decision {result.get('decision')} does not match decide_gate(evidence) oracle: {expected_gate_result['decision']}",
        )
        self.assertEqual(
            result["learned_refiner_enabled"],
            expected_gate_result["learned_refiner_enabled"],
            f"Persisted learned_refiner_enabled does not match oracle",
        )
        if "pass_rate" in expected_gate_result:
            self.assertAlmostEqual(
                result.get("pass_rate", -1.0),
                expected_gate_result["pass_rate"],
                places=5,
            )

    def test_03_all_six_unseen_families_represented_in_held_out_probing(self):
        """Verify probing prompts and audit results explicitly cover all 6 unseen static families."""
        probing_results = self.global_scope.get("probing_audit_results", [])
        probed_families = {r["family_id"] for r in probing_results}
        self.assertEqual(
            probed_families,
            UNSEEN_STATIC_FAMILIES,
            f"Probing did not cover exact set of 6 unseen families: {probed_families} vs {UNSEEN_STATIC_FAMILIES}",
        )



if __name__ == "__main__":
    unittest.main()
