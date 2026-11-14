"""
Adversarial Empirical Challenge Suite for SyncCraft Colab Notebook Pipeline.

Empirically tests edge cases and stress conditions:
- Group A: HF_TOKEN handling (None, empty, whitespace, mock failure, token redaction)
- Group B: Runtime environment & hardware mode detection (CPU vs GPU fallback, smoke test mode)
- Group C: 15-axis SAT collision boundary discrimination (0.0 flush, 1e-8 penetration, 1e-6 overlap)
- Group D: Degenerate cross-axis masking (norm <= 1e-10)
- Group E: Invalid Block-DSL inputs (missing root, negative box size, non-integers, missing meta, cycles)
- Group F: Full end-to-end headless execution under adversarial injections
"""

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List

import nbformat
import numpy as np

# Core SyncCraft imports
from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import DSLValidationError, UnsupportedFeature, validate_program
from synccraft.dsl.parser import DSLParseError, parse_program
from synccraft.kinematics.sat_potential import SATResult, sat_obb
from synccraft.kinematics.se3_chain import Scene, WorldBox, compile_geometry, euler_xyz_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "synccraft_end_to_end.ipynb"


class TestAdversarialHFTokenHandling(unittest.TestCase):
    """Group A: HF_TOKEN and Secrets Edge Cases."""

    def _extract_hf_helpers(self):
        """Extract get_hf_token and redact_token from notebook cell 3."""
        nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
        cell_src = nb.cells[4].source  # Cell 3 code
        scope: Dict[str, Any] = {"os": os, "sys": sys, "PROJECT_ROOT": REPO_ROOT}
        exec(cell_src, scope)
        return scope["get_hf_token"], scope["redact_token"]

    def test_token_none(self):
        """When HF_TOKEN is None or not in env, get_hf_token returns None and redact_token returns <none>."""
        orig = os.environ.get("HF_TOKEN")
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        try:
            get_hf_token, redact_token = self._extract_hf_helpers()
            token = get_hf_token()
            self.assertIsNone(token)
            self.assertEqual(redact_token(token), "<none>")
            self.assertEqual(redact_token(None), "<none>")
        finally:
            if orig is not None:
                os.environ["HF_TOKEN"] = orig

    def test_token_empty_string(self):
        """When HF_TOKEN is empty string, get_hf_token returns empty string or None, and redact_token returns <none>."""
        orig = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = ""
        try:
            get_hf_token, redact_token = self._extract_hf_helpers()
            token = get_hf_token()
            self.assertEqual(redact_token(token), "<none>")
        finally:
            if orig is not None:
                os.environ["HF_TOKEN"] = orig
            else:
                os.environ.pop("HF_TOKEN", None)

    def test_token_whitespace(self):
        """When HF_TOKEN is pure whitespace, redact_token redacts safely without leaking whitespace."""
        orig = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "   "
        try:
            get_hf_token, redact_token = self._extract_hf_helpers()
            token = get_hf_token()
            self.assertEqual(token, "   ")
            redacted = redact_token(token)
            # len("   ") is 3 <= 8, so it returns '***'
            self.assertEqual(redacted, "***")
        finally:
            if orig is not None:
                os.environ["HF_TOKEN"] = orig
            else:
                os.environ.pop("HF_TOKEN", None)

    def test_token_long_secret_redaction(self):
        """Tokens longer than 8 characters show only first 4 and last 4 characters."""
        _, redact_token = self._extract_hf_helpers()
        secret = "hf_1234567890abcdefghijklmnopqrstuvwxyz"
        redacted = redact_token(secret)
        self.assertEqual(redacted, "hf_1...wxyz")
        self.assertNotIn("1234567890abcdef", redacted)

    def test_colab_userdata_exception_fallback(self):
        """If google.colab.userdata.get raises an arbitrary Exception, fallback to os.environ works."""
        mock_google = types.ModuleType("google")
        mock_colab = types.ModuleType("google.colab")
        mock_userdata = types.ModuleType("google.colab.userdata")

        def broken_get(key):
            raise RuntimeError("Colab SecretNotFoundError simulation")

        mock_userdata.get = broken_get
        mock_colab.userdata = mock_userdata
        mock_google.colab = mock_colab

        orig_modules = sys.modules.copy()
        sys.modules["google"] = mock_google
        sys.modules["google.colab"] = mock_colab
        sys.modules["google.colab.userdata"] = mock_userdata

        orig_token = os.environ.get("HF_TOKEN")
        os.environ["HF_TOKEN"] = "hf_fallback_token_123456789"
        try:
            get_hf_token, _ = self._extract_hf_helpers()
            token = get_hf_token()
            self.assertEqual(token, "hf_fallback_token_123456789")
        finally:
            sys.modules.clear()
            sys.modules.update(orig_modules)
            if orig_token is not None:
                os.environ["HF_TOKEN"] = orig_token
            else:
                os.environ.pop("HF_TOKEN", None)


class TestAdversarialHardwareAndRuntime(unittest.TestCase):
    """Group B: CPU/GPU and Execution Mode Edge Cases."""

    def test_cpu_fallback_smoke_mode(self):
        """When CUDA is unavailable, SMOKE_TEST_MODE automatically activates."""
        nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
        cell_src = nb.cells[1].source  # Cell 1
        scope: Dict[str, Any] = {"os": os, "sys": sys, "platform": types.ModuleType("platform")}
        scope["platform"].python_version = lambda: "3.12.0"
        scope["platform"].platform = lambda: "TestOS"
        exec(cell_src, scope)

        self.assertIn("SMOKE_TEST_MODE", scope)
        self.assertTrue(scope["SMOKE_TEST_MODE"])


class TestAdversarialSATBoundaries(unittest.TestCase):
    """Group C: 15-Axis SAT Collision Boundaries."""

    def setUp(self):
        self.rot = np.eye(3, dtype=float)
        self.size = np.array([10.0, 10.0, 10.0], dtype=float)
        self.b_origin = WorldBox("b0", "root", np.array([0.0, 0.0, 0.0]), self.rot, self.size, 0.0, np.zeros(3), "mat")

    def test_exact_zero_flush_contact(self):
        """Exact flush contact: separation = 0.0 -> raw_min = 0.0 -> collision is FALSE, penetration is 0.0."""
        b_flush = WorldBox("b_flush", "root", np.array([10.0, 0.0, 0.0]), self.rot, self.size, 0.0, np.zeros(3), "mat")
        res = sat_obb(self.b_origin, b_flush)
        self.assertAlmostEqual(res.raw_min_overlap, 0.0, places=9)
        self.assertFalse(res.collision, "Exact flush contact must NOT be a collision")
        self.assertEqual(res.penetration_depth, 0.0)

    def test_tiny_positive_separation(self):
        """Tiny positive separation (+1e-8): raw_min = -1e-8 -> collision is FALSE."""
        b_sep = WorldBox("b_sep", "root", np.array([10.0 + 1e-8, 0.0, 0.0]), self.rot, self.size, 0.0, np.zeros(3), "mat")
        res = sat_obb(self.b_origin, b_sep)
        self.assertLess(res.raw_min_overlap, 0.0)
        self.assertFalse(res.collision)
        self.assertEqual(res.penetration_depth, 0.0)

    def test_tiny_penetration_below_default_tolerance(self):
        """
        Penetration of 1e-8 is less than default collision_tolerance (1e-7).
        With tolerance 1e-7: collision is FALSE (treated as numerical noise / within tolerance).
        With tolerance 1e-9: collision is TRUE.
        """
        b_pen_small = WorldBox("b_small", "root", np.array([10.0 - 1e-8, 0.0, 0.0]), self.rot, self.size, 0.0, np.zeros(3), "mat")
        
        # Default tolerance (1e-7)
        res_default = sat_obb(self.b_origin, b_pen_small, collision_tolerance=1e-7)
        self.assertAlmostEqual(res_default.raw_min_overlap, 1e-8, places=9)
        self.assertFalse(res_default.collision, "1e-8 penetration must not exceed 1e-7 tolerance")
        self.assertEqual(res_default.penetration_depth, 0.0)

        # Strict tolerance (1e-9)
        res_strict = sat_obb(self.b_origin, b_pen_small, collision_tolerance=1e-9)
        self.assertTrue(res_strict.collision, "1e-8 penetration exceeds 1e-9 strict tolerance")
        self.assertAlmostEqual(res_strict.penetration_depth, 1e-8, places=9)

    def test_penetration_above_default_tolerance(self):
        """Penetration of 1e-6 exceeds default collision_tolerance (1e-7) -> collision is TRUE."""
        b_pen_1e6 = WorldBox("b_1e6", "root", np.array([10.0 - 1e-6, 0.0, 0.0]), self.rot, self.size, 0.0, np.zeros(3), "mat")
        res = sat_obb(self.b_origin, b_pen_1e6, collision_tolerance=1e-7)
        self.assertTrue(res.collision, "1e-6 penetration must be flagged as collision")
        self.assertAlmostEqual(res.penetration_depth, 1e-6, places=8)

    def test_macro_penetration_point_two(self):
        """Macro penetration of 0.2 units: raw_min = 0.2, collision = TRUE, penetration = 0.2."""
        b_macro = WorldBox("b_macro", "root", np.array([9.8, 0.0, 0.0]), self.rot, self.size, 0.0, np.zeros(3), "mat")
        res = sat_obb(self.b_origin, b_macro)
        self.assertTrue(res.collision)
        self.assertAlmostEqual(res.penetration_depth, 0.2, places=6)

    def test_negative_box_extents_raise_value_error(self):
        """Negative or zero half_extents raise ValueError during box validation."""
        bad_box = WorldBox("bad", "root", np.zeros(3), self.rot, np.array([-1.0, 5.0, 5.0]), 0.0, np.zeros(3), "mat")
        with self.assertRaises(ValueError):
            sat_obb(self.b_origin, bad_box)

    def test_non_orthonormal_rotation_raises_value_error(self):
        """Non-orthonormal rotation matrices raise ValueError."""
        bad_rot = np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        bad_box = WorldBox("bad", "root", np.zeros(3), bad_rot, self.size, 0.0, np.zeros(3), "mat")
        with self.assertRaises(ValueError):
            sat_obb(self.b_origin, bad_box)


class TestAdversarialDegenerateCrossAxes(unittest.TestCase):
    """Group D: Degenerate Cross-Axis Masking."""

    def test_parallel_boxes_mask_all_three_degenerate_cross_axes(self):
        """
        When two boxes are aligned with identity rotation:
        The 3 parallel cross products u_i x u_i are [0, 0, 0] (norm = 0 <= 1e-10).
        These 3 degenerate axes are strictly masked out.
        The remaining 6 non-zero cross axes and 6 face normals are tested (12 total axes).
        """
        rot = np.eye(3, dtype=float)
        b1 = WorldBox("b1", "root", np.zeros(3), rot, np.array([4.0, 4.0, 4.0]), 0.0, np.zeros(3), "mat")
        b2 = WorldBox("b2", "root", np.array([8.0, 0.0, 0.0]), rot, np.array([4.0, 4.0, 4.0]), 0.0, np.zeros(3), "mat")

        res = sat_obb(b1, b2, cross_axis_tolerance=1e-10)
        self.assertEqual(res.tested_axes, 12, "Expected 12 tested axes when 3 degenerate cross axes are masked out")
        self.assertFalse(np.isnan(res.raw_min_overlap))
        self.assertFalse(np.isinf(res.raw_min_overlap))

    def test_sub_tolerance_cross_axis_masked_without_nan(self):
        """
        If a cross product has norm = 1e-11 <= 1e-10, it must be masked out,
        preventing catastrophic normalization (division by 1e-11) and NaN values.
        """
        rot1 = np.eye(3, dtype=float)
        eps = 1e-11
        rot2 = np.array([
            [np.cos(eps), -np.sin(eps), 0.0],
            [np.sin(eps),  np.cos(eps), 0.0],
            [0.0,          0.0,         1.0],
        ])
        u, _, vt = np.linalg.svd(rot2)
        rot2 = u @ vt

        b1 = WorldBox("b1", "root", np.zeros(3), rot1, np.array([2.0, 2.0, 2.0]), 0.0, np.zeros(3), "mat")
        b2 = WorldBox("b2", "root", np.array([10.0, 0.0, 0.0]), rot2, np.array([2.0, 2.0, 2.0]), 0.0, np.zeros(3), "mat")

        res = sat_obb(b1, b2, cross_axis_tolerance=1e-10)
        self.assertFalse(np.isnan(res.raw_min_overlap))
        self.assertFalse(np.isnan(res.penetration_depth))
        self.assertFalse(np.isnan(res.softmin_overlap))
        self.assertFalse(np.isinf(res.raw_min_overlap))
        self.assertLessEqual(res.tested_axes, 15)

    def test_super_tolerance_cross_axis_retained(self):
        """
        If a cross product has norm = 1e-8 > 1e-10, it is legitimately retained as a test axis.
        """
        eps = 1e-8
        rot1 = np.eye(3, dtype=float)
        rot2 = np.array([
            [np.cos(eps), -np.sin(eps), 0.0],
            [np.sin(eps),  np.cos(eps), 0.0],
            [0.0,          0.0,         1.0],
        ])
        u, _, vt = np.linalg.svd(rot2)
        rot2 = u @ vt

        b1 = WorldBox("b1", "root", np.zeros(3), rot1, np.array([2.0, 2.0, 2.0]), 0.0, np.zeros(3), "mat")
        b2 = WorldBox("b2", "root", np.array([10.0, 0.0, 0.0]), rot2, np.array([2.0, 2.0, 2.0]), 0.0, np.zeros(3), "mat")

        res = sat_obb(b1, b2, cross_axis_tolerance=1e-10)
        self.assertFalse(np.isnan(res.raw_min_overlap))
        self.assertGreater(res.tested_axes, 6)


class TestAdversarialDSLValidation(unittest.TestCase):
    """Group E: Invalid and Malicious Block-DSL Inputs."""

    def test_missing_root_bone(self):
        """DSL without root bone is rejected."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @rig {
          bone not_root {
            box blade { size: [2, 4, 1], material: stone }
          }
        }
        """
        with self.assertRaises(DSLParseError):
            parse_program(dsl)

    def test_negative_box_dimensions(self):
        """Negative box dimensions are rejected by semantic validation."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @rig {
          root bone base {
            pivot_local: [0, 0, 0]
            box blade { size: [-2, 4, 1], material: stone }
          }
        }
        """
        with self.assertRaises(DSLValidationError):
            parse_program(dsl)

    def test_zero_box_dimensions(self):
        """Zero box dimensions are rejected (sizes must be strictly positive integers)."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @rig {
          root bone base {
            pivot_local: [0, 0, 0]
            box blade { size: [0, 4, 1], material: stone }
          }
        }
        """
        with self.assertRaises(DSLValidationError):
            parse_program(dsl)

    def test_float_box_dimensions(self):
        """Float box dimensions (non-integers) are rejected."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @rig {
          root bone base {
            pivot_local: [0, 0, 0]
            box blade { size: [2.5, 4, 1], material: stone }
          }
        }
        """
        with self.assertRaises((DSLParseError, DSLValidationError)):
            parse_program(dsl)

    def test_missing_meta_header(self):
        """DSL without @meta header is rejected."""
        dsl = """
        @palette { stone: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @rig {
          root bone base {
            pivot_local: [0, 0, 0]
            box blade { size: [2, 4, 1], material: stone }
          }
        }
        """
        with self.assertRaises(DSLParseError):
            parse_program(dsl)

    def test_undefined_material_reference(self):
        """Referencing an undeclared material raises DSLValidationError."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { wood: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @rig {
          root bone base {
            pivot_local: [0, 0, 0]
            box blade { size: [2, 4, 1], material: obsidian_diamond }
          }
        }
        """
        with self.assertRaises(DSLValidationError):
            parse_program(dsl)

    def test_duplicate_box_names(self):
        """Duplicate box names raise DSLValidationError."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @rig {
          root bone base {
            pivot_local: [0, 0, 0]
            box piece { size: [2, 4, 1], material: stone }
            box piece place(base_box=piece, anchor=[0,1,0], align=[0,-1,0]) {
              size: [2, 4, 1], material: stone
            }
          }
        }
        """
        with self.assertRaises(DSLValidationError):
            parse_program(dsl)

    def test_unsupported_shade_block(self):
        """Programs with @shade are explicitly rejected as DSLParseError or UnsupportedFeature."""
        dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { stone: ramp(base_lch=[0.5, 0.0, 0.0], steps=2, emissive=false) }
        @shade { ambient_occlusion(radius=2) }
        @rig {
          root bone base {
            pivot_local: [0, 0, 0]
            box blade { size: [2, 4, 1], material: stone }
          }
        }
        """
        with self.assertRaises((DSLParseError, UnsupportedFeature)):
            parse_program(dsl)


class TestAdversarialEndToEndExecution(unittest.TestCase):
    """Group F: Headless In-Process Execution Under Adversarial Injections."""

    def test_headless_execution_with_whitespace_token(self):
        """
        Execute the notebook end-to-end headlessly with HF_TOKEN="   ".
        The notebook must handle whitespace token safely, fall back to local staging,
        and generate all required artifacts without unhandled exceptions.
        """
        nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            orig_cwd = os.getcwd()
            orig_env = os.environ.copy()

            os.environ["SYNCCRAFT_SMOKE_TEST"] = "1"
            os.environ["HF_TOKEN"] = "   "

            # Mock google.colab
            mock_modules: Dict[str, Any] = {}
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
                os.chdir(temp_path)
                code_cells = [cell.source for cell in nb.cells if cell.cell_type == "code"]
                for idx, code in enumerate(code_cells):
                    cleaned_lines = []
                    for line in code.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("!") or stripped.startswith("%"):
                            cleaned_lines.append(f"# {line}")
                        else:
                            cleaned_lines.append(line)
                    cleaned_code = "\n".join(cleaned_lines)
                    compiled = compile(cleaned_code, f"<cell_{idx}>", "exec")
                    exec(compiled, global_scope)

                # Verify dataset manifest exists
                manifest_file = temp_path / "datasets" / "dataset_manifest.json"
                self.assertTrue(manifest_file.is_file())

                # Verify Gate G2 report exists
                gate_report = temp_path / "runs" / "gate_g2_report.json"
                self.assertTrue(gate_report.is_file())

                # Verify published artifacts exist in local staging
                published_dirs = list((temp_path / "runs" / "published").glob("published_model_step_*"))
                self.assertGreater(len(published_dirs), 0)

            finally:
                os.chdir(orig_cwd)
                os.environ.clear()
                os.environ.update(orig_env)

    def test_candidate_filter_pipeline_filters_corrupted_candidates(self):
        """
        Test that when an adversarial candidate (e.g. invalid syntax, negative size, non-exempt sibling SAT collision)
        is in the candidate pool, the filtering loop rejects it cleanly into rejected_records.
        """
        from dataclasses import dataclass

        @dataclass
        class MockCandidate:
            asset_id: str
            family_id: str
            archetype: str
            dsl_source: str
            source_license: str = "Apache-2.0"
            split: str = "train"
            parameters: dict = None
            seed: int = 1

        valid_dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { iron: ramp(base_lch=[0.6, 0.02, 200.0], steps=3, emissive=false) }
        @rig {
          root bone handle {
            pivot_local: [0, 0, 0]
            box grip { size: [2, 6, 2], material: iron }
          }
        }
        """

        # Sibling collision: two sibling bones attaching to the same anchor without explicit mutual contract
        sibling_colliding_dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette { steel: ramp(base_lch=[0.75, 0.05, 240], steps=4) }
        @rig {
          root bone root {
            pivot_local: [8, 8, 8]
            box base { size: [6, 6, 6], material: steel }
            bone limb_1 attach(parent_box=base, anchor=[1,0,0], align=[-1,0,0]) {
              box b1 { size: [4, 4, 4], material: steel }
            }
            bone limb_2 attach(parent_box=base, anchor=[1,0,0], align=[-1,0,0]) {
              box b2 { size: [4, 4, 4], material: steel }
            }
          }
        }
        """

        corrupted_dsl = "@meta { archetype: weapon_3d } SYNTAX_ERROR %%%"

        candidates = [
            MockCandidate("valid_01", "straight_blade_sword", "weapon_3d", valid_dsl),
            MockCandidate("colliding_02", "straight_blade_sword", "weapon_3d", sibling_colliding_dsl),
            MockCandidate("corrupted_03", "straight_blade_sword", "weapon_3d", corrupted_dsl),
        ]

        accepted = []
        rejected = []

        for cand in candidates:
            try:
                prog = parse_program(cand.dsl_source)
                validate_program(prog)
                scene = compile_geometry(prog)
                audit_report = audit_scene(scene, collision_tolerance=1e-7)
                if not audit_report["passed"]:
                    rejected.append({"id": cand.asset_id, "reason": "sat_collision"})
                else:
                    accepted.append(cand.asset_id)
            except Exception as exc:
                rejected.append({"id": cand.asset_id, "reason": f"exception: {exc}"})

        self.assertEqual(accepted, ["valid_01"])
        self.assertEqual(len(rejected), 2)
        self.assertEqual(rejected[0]["id"], "colliding_02")
        self.assertEqual(rejected[0]["reason"], "sat_collision")
        self.assertEqual(rejected[1]["id"], "corrupted_03")
        self.assertIn("exception:", rejected[1]["reason"])


if __name__ == "__main__":
    unittest.main()
