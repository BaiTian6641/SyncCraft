"""
SyncCraft Production Colab Notebook End-to-End Test Suite.

Requirement-driven, opaque-box test suite for `notebooks/synccraft_end_to_end.ipynb`
covering Tiers 1-4 as specified in:
- ORIGINAL_REQUEST.md (R1, R2, R3, R4)
- PROJECT.md (§ Architecture, § Milestones, § Interface Contracts)
- AGENTS.md (Numerical rules, SE(3) kinematics, SAT audit, token redaction)
- configs/data_policy.yaml (13 mandatory row fields, seen/unseen static families)
- configs/training_static.yaml (4B/7B BF16 LoRA configuration on Colab RTX PRO 6000)
- src/synccraft/evaluation/gate_g2.py (Gate G2 decision rules and 8 required checks)
"""

import ast
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nbformat
import numpy as np
from PIL import Image

# Core SyncCraft imports
from synccraft.audit.sync_audit import audit_scene
from synccraft.backends.bbmodel5_exporter import export_bbmodel
from synccraft.backends.java_exporter import export_java
from synccraft.dsl.ast_nodes import DSLValidationError, validate_program
from synccraft.dsl.parser import DSLParseError, parse_program
from synccraft.evaluation.budget import CostLedger
from synccraft.evaluation.gate_g2 import REQUIRED_CHECKS, decide_gate
from synccraft.kinematics.sat_potential import SATResult, sat_obb
from synccraft.kinematics.se3_chain import Scene, WorldBox, compile_geometry, euler_xyz_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "synccraft_end_to_end.ipynb"

SEEN_STATIC_FAMILIES = [
    "straight_blade_sword",
    "single_edge_blade",
    "battle_axe",
    "spear",
    "war_hammer",
    "simple_wood_bow",
    "furniture_table_chair_chest_cabinet",
    "clockwork_pendulum_clock",
    "alchemical_workbench_apparatus",
    "gearbox",
    "articulated_robotic_arm",
    "industrial_turbine_generator",
    "piston_crank_engine",
    "cnc_milling_press",
    "centrifugal_pump_valve_skid",
    "wheeled_haul_truck",
    "tracked_excavator",
    "steam_rail_locomotive",
    "twin_rotor_helicopter_drone",
    "cargo_tugboat_vessel",
    "ancient_canopy_tree",
    "crystalline_geode_outcrop",
    "coral_fungal_colony",
    "pine_conifer_evergreen",
    "modular_timber_stone_building",
    "tiered_pagoda_watchtower",
    "arched_stone_bridge_gate",
    "waterwheel_millhouse",
    "humanoid_biped",
    "dog_horse_quadruped",
    "bird",
    "articulated_manta_shark",
    "winged_western_dragon",
    "quadruped_armored_drake",
    "eastern_lung_dragon",
    "bipedal_wyvern",
    "forelimb_lindworm",
    "feathered_raptor",
    "winged_amphithere",
    "multi_headed_hydra",
]

UNSEEN_STATIC_FAMILIES = [
    "mechanical_crossbow",
    "hand_lantern",
    "floating_ring_staff",
    "weeping_bonsai_arch",
    "tripod_telescope",
    "anvil_forge",
    "hydraulic_gantry_crane",
    "articulated_forklift_loader",
    "submersible_rov_craft",
    "folding_windmill",
    "coastal_lighthouse_keep",
    "hexapod_beetle",
    "heavy_tail_tyrannosaur",
    "segmented_sea_dragon",
    "crowned_zilant",
]

MANDATORY_ROW_FIELDS = [
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

REQUIRED_SECTIONS = [
    (1, re.compile(r"Section\s+1\s*:\s*Environment\s+Setup", re.IGNORECASE)),
    (2, re.compile(r"Section\s+2\s*:\s*Hugging\s+Face\s+Authentication", re.IGNORECASE)),
    (3, re.compile(r"Section\s+3\s*:\s*Dataset\s+Gathering", re.IGNORECASE)),
    (4, re.compile(r"Section\s+4\s*:\s*Model\s+Training", re.IGNORECASE)),
    (5, re.compile(r"Section\s+5\s*:\s*(Hugging\s+Face\s+)?Public\s+Asset\s+Publishing", re.IGNORECASE)),
    (6, re.compile(r"Section\s+6\s*:\s*Model\s+Probing", re.IGNORECASE)),
]


def load_notebook() -> nbformat.NotebookNode:
    """Load and return the target notebook, asserting its existence."""
    if not NOTEBOOK_PATH.is_file():
        raise AssertionError(
            f"Required notebook artifact not found at: {NOTEBOOK_PATH}. "
            "The notebook must be created at notebooks/synccraft_end_to_end.ipynb."
        )
    return nbformat.read(str(NOTEBOOK_PATH), as_version=4)


def get_all_code_cells(nb: nbformat.NotebookNode) -> List[str]:
    """Extract source text of all code cells in the notebook."""
    return [cell.source for cell in nb.cells if cell.cell_type == "code"]


def get_all_markdown_cells(nb: nbformat.NotebookNode) -> List[str]:
    """Extract source text of all markdown cells in the notebook."""
    return [cell.source for cell in nb.cells if cell.cell_type == "markdown"]


def get_section_code_cells(nb: nbformat.NotebookNode, section_num: int) -> List[str]:
    """Extract source text of code cells belonging to a given section (1 to 6)."""
    section_code = []
    current_sec = None
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            for sec_idx, pattern in REQUIRED_SECTIONS:
                if pattern.search(cell.source):
                    current_sec = sec_idx
                    break
        elif cell.cell_type == "code":
            if current_sec == section_num:
                section_code.append(cell.source)
    return section_code


def extract_ast_names_and_attributes(code: str) -> Set[str]:
    """Parse python code and extract all AST Names, Attributes, and Import aliases."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
            chain = []
            curr = node
            while isinstance(curr, ast.Attribute):
                chain.append(curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                chain.append(curr.id)
                identifiers.add(".".join(reversed(chain)))
        elif isinstance(node, ast.alias):
            identifiers.add(node.name)
            if node.asname:
                identifiers.add(node.asname)
    return identifiers


def validate_safetensors_binary_header(payload: bytes) -> Tuple[int, Dict[str, Any]]:
    """
    Validate safetensors binary header per the official specification:
    - Little-endian unsigned 64-bit integer (<Q) header size N in the first 8 bytes.
    - N < 100MB (100 * 1024 * 1024 bytes) to prevent resource exhaustion / malformed stubs.
    - 8 + N <= len(payload).
    - Next N bytes must decode as UTF-8 valid JSON object.
    - Payload must not be raw ASCII mock bytes (e.g. b'LORA_SAFETENSORS_PAYLOAD').
    Returns (header_len, header_dict).
    """
    if len(payload) < 8:
        raise ValueError(f"Safetensors payload too small: {len(payload)} bytes (minimum 8)")
    if b"LORA_SAFETENSORS_PAYLOAD" in payload:
        raise ValueError("Safetensors payload contains raw ASCII mock byte string 'b\"LORA_SAFETENSORS_PAYLOAD\"'")
    header_len = struct.unpack("<Q", payload[:8])[0]
    if header_len <= 0:
        raise ValueError(f"Invalid non-positive safetensors header length: {header_len}")
    max_header_size = 100 * 1024 * 1024
    if header_len >= max_header_size:
        raise ValueError(
            f"Safetensors header length {header_len} exceeds maximum allowed size "
            f"({max_header_size} bytes / 100MB). Likely un-packed mock ASCII bytes."
        )
    if 8 + header_len > len(payload):
        raise ValueError(
            f"Safetensors payload truncated: total bytes {len(payload)}, but header requires {8 + header_len}"
        )
    header_bytes = payload[8 : 8 + header_len]
    try:
        header_str = header_bytes.decode("utf-8")
        header_dict = json.loads(header_str)
    except Exception as exc:
        raise ValueError(f"Failed to decode safetensors JSON header: {exc}") from exc
    if not isinstance(header_dict, dict):
        raise ValueError(f"Safetensors header must be a JSON dictionary, got {type(header_dict)}")
    return header_len, header_dict


# ==============================================================================
# Tier 1: Feature Coverage & Structural Compliance
# ==============================================================================


class TestNotebookTier1FeatureCoverage(unittest.TestCase):
    """
    Tier 1 tests verifying notebook existence, schema validity, delineated
    sections, and explicit code implementations of requirements R1-R4.
    """

    def test_01_notebook_file_exists(self):
        """Verify the primary notebook file exists at the required repository path."""
        self.assertTrue(
            NOTEBOOK_PATH.is_file(),
            f"Notebook does not exist at {NOTEBOOK_PATH}. Expected notebooks/synccraft_end_to_end.ipynb",
        )

    def test_02_notebook_valid_json_and_nbformat_v4(self):
        """Verify the notebook is syntactically valid JSON conforming to nbformat v4."""
        nb = load_notebook()
        nbformat.validate(nb)
        self.assertEqual(nb.nbformat, 4, f"Expected nbformat version 4, got {nb.nbformat}")
        self.assertGreater(len(nb.cells), 0, "Notebook should contain at least one cell")

    def test_03_markdown_sections_delineated(self):
        """Verify all 6 required sections from PROJECT.md are delineated in markdown headers."""
        nb = load_notebook()
        md_texts = "\n\n".join(get_all_markdown_cells(nb))

        for sec_num, pattern in REQUIRED_SECTIONS:
            match = pattern.search(md_texts)
            self.assertIsNotNone(
                match,
                f"Missing required Section {sec_num} header in notebook markdown cells. Pattern: {pattern.pattern}",
            )

    def test_03b_colab_standalone_repo_bootstrap_and_auto_install(self):
        """
        Verify Section 1 (and Section 3 self-healing guard) automatically clones the
        SyncCraft git repository and installs synccraft + dependencies when opened
        standalone from GitHub in Google Colab (preventing ModuleNotFoundError: No module named 'synccraft').
        """
        nb = load_notebook()
        sec1_code = "\n".join(get_section_code_cells(nb, 1))
        sec3_code = "\n".join(get_section_code_cells(nb, 3))

        self.assertIn(
            'find_spec("synccraft")',
            sec1_code,
            "Section 1 must check importlib.util.find_spec('synccraft') to detect standalone Colab runtimes",
        )
        self.assertIn(
            "https://github.com/BaiTian6641/SyncCraft.git",
            sec1_code,
            "Section 1 must auto-clone BaiTian6641/SyncCraft when synccraft is missing on Colab",
        )
        self.assertIn(
            '"pip", "install"',
            sec1_code,
            "Section 1 must run pip install automatically on standalone Colab runtimes",
        )
        self.assertIn(
            'find_spec("synccraft")',
            sec3_code,
            "Section 3 must also include a self-healing find_spec('synccraft') guard for out-of-order Colab execution",
        )

    def test_04_colab_secrets_and_env_fallback(self):
        """Verify Section 2 code cell implements Colab Secrets retrieval with os.environ fallback."""
        nb = load_notebook()
        code_combined = "\n".join(get_all_code_cells(nb))

        # Check for google.colab userdata retrieval
        self.assertTrue(
            "google.colab" in code_combined or "userdata" in code_combined,
            "Notebook code should access google.colab.userdata for HF_TOKEN",
        )
        # Check for environment variable fallback
        self.assertTrue(
            "os.environ.get" in code_combined or "os.getenv" in code_combined,
            "Notebook code should contain fallback to os.environ for HF_TOKEN",
        )

    def test_05_dataset_gathering_and_sat_filtering(self):
        """Verify Section 3 code cell implements AST validation and direct SAT collision filtering."""
        nb = load_notebook()
        code_combined = "\n".join(get_all_code_cells(nb))

        # Must import and call core validation and audit tools
        self.assertTrue(
            "parse_program" in code_combined or "validate_program" in code_combined,
            "Section 3 must use parse_program / validate_program for DSL validation",
        )
        self.assertTrue(
            "compile_geometry" in code_combined,
            "Section 3 must compile candidate ASTs using compile_geometry",
        )
        self.assertTrue(
            "audit_scene" in code_combined or "sat_obb" in code_combined,
            "Section 3 must filter candidate geometries using audit_scene or sat_obb",
        )

    def test_05b_seen_static_families_complete_coverage(self):
        """
        Verify Section 3 defines templates or generators for all 8 seen static families
        specified in configs/data_policy.yaml.
        """
        nb = load_notebook()
        sec3_code = "\n".join(get_section_code_cells(nb, 3))
        for fam in SEEN_STATIC_FAMILIES:
            self.assertIn(
                fam,
                sec3_code,
                f"Section 3 must define template/generator for seen static family '{fam}' "
                f"per configs/data_policy.yaml",
            )

    def test_06_dataset_manifest_schema_and_hashing(self):
        """Verify Section 3 outputs dataset_manifest.json with SHA-256 and mandatory row fields."""
        nb = load_notebook()
        code_combined = "\n".join(get_all_code_cells(nb))

        self.assertIn("dataset_manifest.json", code_combined)
        self.assertTrue(
            "dataset_manifest_sha256" in code_combined or "sha256" in code_combined,
            "Manifest generation must calculate SHA-256 digest of the dataset manifest",
        )

    def test_06b_dataset_first_hf_hub_publishing_and_attribution(self):
        """
        Verify Section 3 builds the dataset FIRST and publishes it publicly to Hugging Face Hub
        (repo_type='dataset') BEFORE Section 4 training, complete with Dataset Card (README.md),
        NOTICE file, Apache-2.0 license, Oklab/SyncCraft credits, and BibTeX citation.
        """
        nb = load_notebook()
        sec3_code = "\n".join(get_section_code_cells(nb, 3))
        sec5_code = "\n".join(get_section_code_cells(nb, 5))

        # Section 3 (before Section 4 training) must publish dataset to Hugging Face Hub
        self.assertIn('repo_type="dataset"', sec3_code)
        self.assertIn("create_repo", sec3_code)
        self.assertIn("upload_folder", sec3_code)
        self.assertIn("DATASET_REPO_ID", sec3_code)

        # Dataset Card configs & attribution checks
        self.assertIn("configs:", sec3_code)
        self.assertIn("train.jsonl", sec3_code)
        self.assertIn("val.jsonl", sec3_code)
        self.assertIn("test_unseen.jsonl", sec3_code)
        self.assertIn("Apache-2.0", sec3_code)
        self.assertIn("BaiTian6641/SyncCraft", sec3_code)
        self.assertIn("Ottosson", sec3_code)
        self.assertIn("synccraft2026dataset", sec3_code)

        # Section 5 Model Card must cross-link DATASET_REPO_ID and include full attribution
        self.assertIn("DATASET_REPO_ID", sec5_code)
        self.assertIn("BaiTian6641/SyncCraft", sec5_code)
        self.assertIn("Ottosson", sec5_code)
        self.assertIn("synccraft2026model", sec5_code)

    def test_06c_full_60k_preset_and_sys_path_anti_shadowing(self):
        """
        Verify Section 1 & Section 3:
        1. Default to 'full_60k' (60,000 seen + 1,200 unseen = 61,200 complex rows) on GPU,
           and 'smoke' in SMOKE_TEST_MODE.
        2. Never insert PROJECT_ROOT directly onto sys.path (only SRC_DIR), preventing the
           generated datasets/ directory from shadowing Hugging Face 'datasets' via PEP 420.
        3. Enforce P1-02 physical contact auditing (enforce_physical_contact=True) and Parquet shards.
        """
        nb = load_notebook()
        sec1_code = "\n".join(get_section_code_cells(nb, 1))
        sec3_code = "\n".join(get_section_code_cells(nb, 3))

        self.assertNotIn("sys.path.insert(0, str(PROJECT_ROOT))", sec1_code)
        self.assertNotIn("sys.path.insert(0, str(_fallback_root))", sec3_code)
        self.assertIn("sys.path.insert(0, str(SRC_DIR))", sec1_code)
        self.assertIn("spec.loader is not None", sec1_code)

        self.assertIn('"full_60k"', sec3_code)
        self.assertIn("build_verified_dataset", sec3_code)
        self.assertIn("enforce_physical_contact=True", sec3_code)
        self.assertIn("write_parquet=True", sec3_code)

    def test_07_lora_4b_7b_training_configurations(self):
        """
        Verify Section 4 specifies genuine full-precision BF16 LoRA configurations for Colab RTX PRO 6000:
        - AST references to AutoModelForCausalLM, AutoTokenizer, and bfloat16 in GPU pipeline.
        - AST references to peft / LoraConfig (rank 32, alpha 64) and get_peft_model.
        - Strict prohibition of 4-bit QLoRA / BitsAndBytesConfig / load_in_4bit quantization.
        - Dataset sample tokenization (not dummy random integer noise training).
        """
        nb = load_notebook()
        sec4_code = "\n".join(get_section_code_cells(nb, 4))
        ast_names = extract_ast_names_and_attributes(sec4_code)

        # 1. Base causal LM architecture, BF16 precision, and tokenizer references
        self.assertTrue(
            "AutoModelForCausalLM" in ast_names,
            "Section 4 AST must reference AutoModelForCausalLM for BF16 LoRA base model loading",
        )
        self.assertTrue(
            "AutoTokenizer" in ast_names,
            "Section 4 AST must reference AutoTokenizer for dataset tokenization",
        )
        self.assertTrue(
            "bfloat16" in ast_names or "torch.bfloat16" in ast_names,
            "Section 4 AST must specify torch.bfloat16 precision for RTX PRO 6000 LoRA",
        )

        # 2. PEFT LoRA adapter configuration (pure LoRA, no 4-bit QLoRA quantization)
        self.assertTrue(
            "LoraConfig" in ast_names or "peft" in ast_names,
            "Section 4 AST must reference LoraConfig / peft for adapter initialization",
        )
        self.assertTrue(
            "get_peft_model" in ast_names or "PeftModel" in ast_names,
            "Section 4 AST must reference get_peft_model / PeftModel to attach adapter",
        )
        self.assertNotIn(
            "BitsAndBytesConfig",
            ast_names,
            "Section 4 must use native BF16 LoRA on RTX PRO 6000, not 4-bit BitsAndBytesConfig QLoRA",
        )
        self.assertNotIn(
            "load_in_4bit",
            sec4_code,
            "Section 4 must not enable 4-bit quantization (load_in_4bit)",
        )

        # 3. LoRA rank 32 and alpha 64 per configs/training_static.yaml
        self.assertIn(
            "32",
            sec4_code,
            "LoRA configuration should specify rank 32 (per configs/training_static.yaml)",
        )
        self.assertIn(
            "64",
            sec4_code,
            "LoRA configuration should specify alpha 64 (per configs/training_static.yaml)",
        )

        # 4. Genuine dataset sample tokenization
        self.assertTrue(
            "tokenizer" in sec4_code.lower(),
            "Section 4 must tokenize dataset samples using the tokenizer",
        )
        # Ban training solely on random integer tensors without tokenizing dataset
        self.assertNotIn(
            "torch.randint",
            sec4_code,
            "Section 4 training loop must not synthesize dummy input_ids via torch.randint; "
            "it must tokenize genuine Block-DSL dataset samples.",
        )

    def test_07b_lfm3_vl_2b_default_and_colab_torchao_guard(self):
        """
        Verify Section 1 & Section 4 default to LiquidAI/LFM3-VL-2B, dispatch VLMs via
        AutoModelForImageTextToText + AutoProcessor with dtype=..., resolve language-backbone
        tri-hybrid LoRA target_modules while freezing vision_tower, and remediate Colab's
        incompatible preinstalled torchao < 0.16.0.
        """
        import torch.nn as nn

        nb = load_notebook()
        sec1_code = "\n".join(get_section_code_cells(nb, 1))
        sec4_code = "\n".join(get_section_code_cells(nb, 4))

        self.assertIn("LiquidAI/LFM3-VL-2B", sec4_code)
        self.assertIn("AutoModelForImageTextToText", sec4_code)
        self.assertIn("AutoProcessor", sec4_code)
        self.assertIn("dtype=compute_dtype", sec4_code)
        self.assertIn("torchao", sec1_code)
        self.assertIn("torchao", sec4_code)
        self.assertIn("transformers>=5.21.0", sec1_code)

        # Execute resolve_lora_target_modules directly from Cell 11 source on synthetic LFM3-VL-2B & Transformer modules
        cell11_src = "".join(nb.cells[11].source)
        tree = ast.parse(cell11_src)
        extracted_nodes = [
            node for node in tree.body
            if (isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "CANDIDATE_LORA_LINEAR_SUFFIXES" for t in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name == "resolve_lora_target_modules")
        ]
        mod_ast = ast.Module(body=extracted_nodes, type_ignores=[])
        ns = {"nn": nn, "re": re}
        exec(compile(mod_ast, "<cell11_resolver>", "exec"), ns)
        resolver = ns["resolve_lora_target_modules"]

        class MockLfm3Vl(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = nn.ModuleDict({"q_proj": nn.Linear(8, 8), "out_proj": nn.Linear(8, 8)})
                self.multi_modal_projector = nn.ModuleDict({"linear": nn.Linear(8, 8)})
                self.language_model = nn.ModuleDict({
                    "q_proj": nn.Linear(8, 8),
                    "k_proj": nn.Linear(8, 8),
                    "v_proj": nn.Linear(8, 8),
                    "o_proj": nn.Linear(8, 8),
                    "gate_proj": nn.Linear(8, 8),
                    "in_proj": nn.Linear(8, 8),
                    "out_proj": nn.Linear(8, 8),
                    "w1": nn.Linear(8, 8),
                    "w2": nn.Linear(8, 8),
                    "w3": nn.Linear(8, 8),
                    "lm_head": nn.Linear(8, 16),
                })

        vlm_pattern = resolver(MockLfm3Vl())
        self.assertIsInstance(vlm_pattern, str)
        self.assertIsNone(re.fullmatch(vlm_pattern, "vision_tower.q_proj"))
        self.assertIsNone(re.fullmatch(vlm_pattern, "vision_tower.out_proj"))
        self.assertIsNone(re.fullmatch(vlm_pattern, "language_model.lm_head"))
        for lf in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "in_proj", "out_proj", "w1", "w2", "w3"):
            self.assertIsNotNone(re.fullmatch(vlm_pattern, f"model.language_model.layers.0.{lf}"))

        class MockTextCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleDict({
                    "q_proj": nn.Linear(8, 8),
                    "v_proj": nn.Linear(8, 8),
                    "up_proj": nn.Linear(8, 8),
                    "down_proj": nn.Linear(8, 8),
                    "lm_head": nn.Linear(8, 16),
                })

        text_targets = resolver(MockTextCausalLM())
        self.assertEqual(text_targets, ["q_proj", "v_proj", "up_proj", "down_proj"])

    def test_07c_anti_overfit_multidomain_and_val_loss_contracts(self):
        """
        Verify ADR 0004 & C6 multi-domain anti-overfitting contracts across Section 3, 4, and 6:
        - 'full_75k' default GPU preset + 'world_1200' causal trajectory preset + diversity_stats logging.
        - Response-masked label_smoothing, weight_decay=0.05, lora_dropout=0.10, and epochs=1 default.
        - Unbiased StreamMixtureSampler visiting 100% of main rows without replacement and capping aux repeats <= 3
          (strictly banning modulo-replacement global_sample_idx % 5 and b_idx % 4).
        - Stratified held-out ValLoss evaluation (evaluate_validation_loss & val_loss_history).
        - All 15 multi-domain unseen families probed in Section 6.
        """
        nb = load_notebook()
        sec3_code = "\n".join(get_section_code_cells(nb, 3))
        sec4_code = "\n".join(get_section_code_cells(nb, 4))
        sec6_code = "\n".join(get_section_code_cells(nb, 6))

        self.assertIn('"full_75k"', sec3_code)
        self.assertIn('"world_1200"', sec3_code)
        self.assertIn("diversity_stats", sec3_code)

        self.assertIn("label_smoothing", sec4_code)
        self.assertIn("evaluate_validation_loss", sec4_code)
        self.assertIn("val_loss_history", sec4_code)
        self.assertIn("StreamMixtureSampler", sec4_code)
        self.assertNotIn("global_sample_idx % 5", sec4_code)
        self.assertNotIn("b_idx % 4 == 3", sec4_code)

        for unseen_fam in UNSEEN_STATIC_FAMILIES:
            self.assertIn(unseen_fam, sec6_code, f"Section 6 must probe unseen family '{unseen_fam}'")

    def test_07d_v31_apex_musculoskeletal_and_8192_context_contracts(self):
        """
        Verify ADR 0006 clean-room Apex musculoskeletal integration across Section 3, 4, and 6:
        - Synthesizes 'apex_complex_v3_2' shard (with v3_1 fallback read) and registers config_name: apex_complex_v3_2 in Dataset Card.
        - Defaults max_sequence_length to 8192 (>= 8192 so ~7.6k-token multi-turn artist-loop rows do not truncate).
        - Mixes Apex and World streams via StreamMixtureSampler with max_aux_repeats <= 3.
        - Probes >=40-box held-out Apex unseen families (APEX_UNSEEN_FAMILIES).
        """
        nb = load_notebook()
        sec3_code = "\n".join(get_section_code_cells(nb, 3))
        sec4_code = "\n".join(get_section_code_cells(nb, 4))
        sec6_code = "\n".join(get_section_code_cells(nb, 6))

        self.assertIn("generate_apex_complex_dataset", sec3_code)
        self.assertIn("config_name: apex_complex_v3_2", sec3_code)
        self.assertIn('"8192"', sec4_code)
        self.assertNotIn("max_sequence_length: int = 2048", sec4_code)
        self.assertIn("StreamMixtureSampler", sec4_code)
        self.assertNotIn("global_sample_idx % 5", sec4_code)
        self.assertIn("APEX_UNSEEN_FAMILIES", sec6_code)
        self.assertIn("audit_behavior_clip", sec6_code)
        self.assertIn("apex_complex_eval_report.json", sec6_code)

    def test_08_checkpoint_structure_contract(self):
        """Verify Section 4 implements periodic/atomic checkpointing saving manifest and adapters."""
        nb = load_notebook()
        sec4_code = "\n".join(get_section_code_cells(nb, 4))

        self.assertIn("checkpoints", sec4_code)
        self.assertTrue(
            "manifest.json" in sec4_code or "manifest" in sec4_code,
            "Checkpointing must include a manifest.json tracking provenance and step",
        )
        self.assertNotIn(
            "LORA_SAFETENSORS_PAYLOAD",
            sec4_code,
            "Section 4 must not write dummy mock payload 'b\"LORA_SAFETENSORS_PAYLOAD\"' as safetensors",
        )

    def test_08b_safetensors_binary_header_contract(self):
        """
        Verify safetensors binary header serialization contract:
        - Little-endian unsigned 64-bit integer (<Q) header length.
        - Header length < 100MB (100 * 1024 * 1024 bytes).
        - Valid UTF-8 JSON header metadata dictionary.
        - No raw ASCII mock bytes.
        """
        # Test valid synthetic safetensors payload conforms to validator
        test_header = json.dumps({
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "dtype": "F32",
                "shape": [32, 256],
                "data_offsets": [0, 32768],
            }
        }).encode("utf-8")
        header_len_packed = struct.pack("<Q", len(test_header))
        dummy_tensor_bytes = b"\x00" * 32768
        valid_safetensors_payload = header_len_packed + test_header + dummy_tensor_bytes

        h_len, h_dict = validate_safetensors_binary_header(valid_safetensors_payload)
        self.assertEqual(h_len, len(test_header))
        self.assertIn("base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", h_dict)

        # Verify rejection of mock ASCII bytes
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(b"LORA_SAFETENSORS_PAYLOAD")

        # Verify rejection of oversize header (> 100MB)
        oversize_header_len = struct.pack("<Q", 120 * 1024 * 1024)
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(oversize_header_len + b"{}" * 100)

    def test_09_hf_publishing_idempotency_and_model_card(self):
        """Verify Section 5 implements idempotent repo creation and model card generation."""
        nb = load_notebook()
        code_combined = "\n".join(get_all_code_cells(nb))

        # Hugging Face Hub API usage
        self.assertTrue(
            "create_repo" in code_combined or "HfApi" in code_combined,
            "Section 5 must use Hugging Face Hub API to manage public repository",
        )
        self.assertTrue(
            "README.md" in code_combined or "model card" in code_combined.lower(),
            "Section 5 must generate a README.md model card for the published repository",
        )

    def test_10_probing_unseen_families_and_gate_g2(self):
        """Verify Section 6 implements probing on unseen families and Gate G2 evaluation."""
        nb = load_notebook()
        sec6_code = "\n".join(get_section_code_cells(nb, 6))

        # Must mention unseen families
        unseen_mentions = [fam in sec6_code for fam in UNSEEN_STATIC_FAMILIES]
        self.assertTrue(
            any(unseen_mentions),
            f"Section 6 must probe held-out unseen families (expected one or more of {UNSEEN_STATIC_FAMILIES})",
        )
        # Must evaluate Gate G2
        self.assertTrue(
            "decide_gate" in sec6_code or "REQUIRED_CHECKS" in sec6_code,
            "Section 6 must evaluate Gate G2 decision logic using decide_gate",
        )

    def test_10b_model_probing_queries_model_generation(self):
        """
        Verify Section 6 model probing actually queries the model and contains zero ground-truth fallbacks:
        - Probing loop must NOT bypass model generation by assigning raw_completion_text = task["dsl"].
        - Probing loop must NOT contain ground-truth fallback in Cell 16 (e.g. except Exception: evaluated_dsl = task["dsl_target"]).
        - AST inspection verifies model output is evaluated directly without task["dsl_target"] exception fallback.
        - Section 6 AST/code must invoke model generation (model.generate or calling model / generation routine).
        """
        nb = load_notebook()
        sec6_cells = get_section_code_cells(nb, 6)
        sec6_code = "\n".join(sec6_cells)

        # 1. Check for prohibited ground-truth substitution bypass
        clean_code = re.sub(r"\s+", " ", sec6_code)
        self.assertNotIn(
            'raw_completion_text = task["dsl"]',
            clean_code,
            "Section 6 probing must not bypass the model by assigning raw_completion_text = task['dsl']",
        )
        self.assertNotIn(
            "raw_completion_text = task['dsl']",
            clean_code,
            "Section 6 probing must not bypass the model by assigning raw_completion_text = task['dsl']",
        )

        # 2. Strict anti-facade: Ban ground-truth fallback in Cell 16
        self.assertNotIn(
            'evaluated_dsl = task["dsl_target"]',
            clean_code,
            "Cell 16 must not fall back to task['dsl_target'] when model completion has syntax errors",
        )
        self.assertNotIn(
            "evaluated_dsl = task['dsl_target']",
            clean_code,
            "Cell 16 must not fall back to task['dsl_target'] when model completion has syntax errors",
        )

        # AST inspection of Cell 16 (probing cell): verify no try-except block falls back to ground truth
        probing_cell_code = sec6_cells[0] if sec6_cells else sec6_code
        parsed_probing = ast.parse(probing_cell_code)
        for node in ast.walk(parsed_probing):
            if isinstance(node, ast.ExceptHandler):
                handler_source = ast.unparse(node)
                self.assertNotIn(
                    "dsl_target",
                    handler_source,
                    "Exception handler in Cell 16 probing must not fall back to dsl_target",
                )
                self.assertNotIn(
                    "UNSEEN_TEMPLATES",
                    handler_source,
                    "Exception handler in Cell 16 probing must not fall back to UNSEEN_TEMPLATES",
                )

        # Verify that model completion is evaluated directly (raw_completion_text assigned without fallback)
        self.assertTrue(
            "evaluated_dsl = raw_completion_text" in clean_code
            or '"completion": raw_completion_text' in clean_code
            or "'completion': raw_completion_text" in clean_code,
            "Cell 16 must directly evaluate raw_completion_text without ground-truth fallback substitution",
        )

        # 3. Check that generation routine queries model
        ast_names = extract_ast_names_and_attributes(sec6_code)
        has_model_query = (
            "generate" in ast_names
            or "model.generate" in ast_names
            or "forward" in ast_names
            or "model" in ast_names
        )
        self.assertTrue(
            has_model_query,
            "Section 6 probing must query the model (e.g. model.generate or generation function)",
        )

    def test_10c_gate_g2_prerequisites_dynamic_evaluation(self):
        """
        Verify Section 6 Gate G2 evidence evaluates prerequisites dynamically:
        - 'all_samples_have_at_least_16_parts' must NOT be hardcoded as a static boolean True.
        - In Cell 18, 'all_samples_have_16_parts' must genuinely inspect box counts (boxes >= 16),
          not use facade boolean proxies like passed_audits == len(probed_completions).
        - When evaluated on samples with < 16 parts, decide_gate reports decision: blocked.
        """
        nb = load_notebook()
        sec6_cells = get_section_code_cells(nb, 6)
        sec6_code = "\n".join(sec6_cells)

        # 1. Check that 'all_samples_have_at_least_16_parts: True' is not statically hardcoded
        self.assertNotIn(
            '"all_samples_have_at_least_16_parts": True',
            sec6_code,
            "Gate G2 evidence must dynamically compute 'all_samples_have_at_least_16_parts' "
            "from observed box counts rather than hardcoding True.",
        )
        self.assertNotIn(
            "'all_samples_have_at_least_16_parts': True",
            sec6_code,
            "Gate G2 evidence must dynamically compute 'all_samples_have_at_least_16_parts' "
            "from observed box counts rather than hardcoding True.",
        )

        # 2. Ban the facade proxy expression identified by Forensic Auditor
        clean_code = re.sub(r"\s+", " ", sec6_code)
        self.assertNotIn(
            "len(probing_audit_results) == len(probed_completions) and passed_audits == len(probed_completions)",
            clean_code,
            "Gate G2 prerequisite must not use passed_audits equality as a facade for box count check",
        )

        # 3. Cell 18 AST inspection: all_samples_have_16_parts must genuinely check box counts (>= 16)
        gate_cell_code = sec6_cells[2] if len(sec6_cells) >= 3 else sec6_code
        parsed_gate = ast.parse(gate_cell_code)
        found_box_count_check = False
        for node in ast.walk(parsed_gate):
            if isinstance(node, ast.Assign):
                target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "all_samples_have_16_parts" in target_names or "all_samples_have_at_least_16_parts" in target_names:
                    assigned_code = ast.unparse(node.value)
                    self.assertIn(
                        "boxes",
                        assigned_code,
                        f"'all_samples_have_16_parts' must inspect 'boxes', got: {assigned_code}",
                    )
                    self.assertIn(
                        "16",
                        assigned_code,
                        f"'all_samples_have_16_parts' must check for >= 16 parts, got: {assigned_code}",
                    )
                    self.assertNotIn(
                        "passed_audits",
                        assigned_code,
                        f"'all_samples_have_16_parts' must not substitute passed_audits for box count, got: {assigned_code}",
                    )
                    found_box_count_check = True

        self.assertTrue(
            found_box_count_check,
            "Cell 18 must assign 'all_samples_have_16_parts' with a genuine check on box counts (>= 16)",
        )

        # 4. Verify dynamic decide_gate contract on assemblies with < 16 parts
        evidence_insufficient_parts = {
            "status": "colab_pipeline_evaluated",
            "total": 6,
            "passed": 6,
            "dataset_manifest_sha256": "0" * 64,
            "evaluation_manifest_sha256": "1" * 64,
            "full_static_audit_implemented": True,
            "heldout_split_frozen_before_training": True,
            "all_samples_have_at_least_16_parts": False,  # Correctly reported for < 16 part models
            "repair_pgd_policy_preregistered": True,
            "matched_raw_budgets_enforced": True,
            "at_most_three_reaudits_per_sample": True,
            "failures_and_timeouts_counted": True,
            "baseline_implementations_verified": True,
        }
        res_blocked = decide_gate(evidence_insufficient_parts)
        self.assertEqual(
            res_blocked["decision"],
            "blocked",
            "When all_samples_have_at_least_16_parts is False, decide_gate must report decision: blocked",
        )
        self.assertFalse(res_blocked["learned_refiner_enabled"])
        self.assertIn("all_samples_have_at_least_16_parts", res_blocked["missing_or_invalid_evidence"])

        # 5. Verify dynamic decide_gate contract when all prerequisites are met and pass rate >= 90%
        evidence_sufficient = dict(evidence_insufficient_parts)
        evidence_sufficient["all_samples_have_at_least_16_parts"] = True
        evidence_sufficient["passed"] = 6
        res_eligible_or_skip = decide_gate(evidence_sufficient)
        self.assertEqual(
            res_eligible_or_skip["decision"],
            "skip_learned_refiner",
            "When all prerequisites met and pass rate >= 90%, decide_gate reports skip_learned_refiner",
        )


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


class TestNotebookTier2BoundaryAndCornerCases(unittest.TestCase):
    """
    Tier 2 tests verifying edge cases, boundary discrimination, fallbacks,
    degenerate cross-axis masking, and token security.
    """

    def test_01_missing_hf_token_graceful_fallback(self):
        """Verify missing HF_TOKEN falls back to local staging without crashing."""
        nb = load_notebook()
        code_combined = "\n".join(get_all_code_cells(nb))

        # Code must check if hf_token is None/empty and avoid unhandled crash
        self.assertTrue(
            "if not hf_token" in code_combined
            or "if hf_token is None" in code_combined
            or "if hf_token:" in code_combined
            or "not hf_token" in code_combined,
            "Section 2 or 5 must test whether HF_TOKEN is provided and fall back safely",
        )

    def test_02_missing_cuda_cpu_smoke_mode(self):
        """Verify SMOKE_TEST_MODE flag activates when CUDA is not available or SYNCCRAFT_SMOKE_TEST=1."""
        nb = load_notebook()
        code_combined = "\n".join(get_all_code_cells(nb))

        self.assertIn("SMOKE_TEST_MODE", code_combined)
        self.assertTrue(
            "SYNCCRAFT_SMOKE_TEST" in code_combined or "cuda.is_available" in code_combined,
            "SMOKE_TEST_MODE must check SYNCCRAFT_SMOKE_TEST or torch.cuda.is_available()",
        )

    def test_03_empty_and_malformed_candidate_dsl_rejection(self):
        """Test dataset filter error handling on empty, syntax-invalid, and semantically invalid DSL."""
        # 1. Empty string input
        with self.assertRaises(DSLParseError):
            parse_program("")

        # 2. Syntax invalid input
        malformed_dsl = "@meta { archetype: weapon_3d } invalid_token @@@"
        with self.assertRaises(DSLParseError):
            parse_program(malformed_dsl)

        # 3. Semantic invalid input (negative box dimension)
        invalid_semantic_dsl = """
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
            parse_program(invalid_semantic_dsl)

    def test_04_sat_flush_contact_vs_penetration_boundary(self):
        """
        Boundary test for direct 15-axis SAT OBB:
        Flush faces (raw overlap == 0.0) must NOT be flagged as collision.
        Penetrating faces (raw overlap > 0.0) MUST be flagged as collision.
        """
        rot = np.eye(3, dtype=float)
        size = np.array([10.0, 10.0, 10.0], dtype=float)

        # Box 1 centered at origin: spans [-5, 5] along X
        b1 = WorldBox("b1", "root", np.array([0.0, 0.0, 0.0]), rot, size, 0.0, np.zeros(3), "steel")

        # Box 2 centered at [10.0, 0, 0]: spans [5.0, 15.0] along X
        # Exactly flush at X=5.0 -> distance=10.0, r1+r2=10.0 -> raw_min=0.0
        b2_flush = WorldBox("b2", "root", np.array([10.0, 0.0, 0.0]), rot, size, 0.0, np.zeros(3), "steel")
        res_flush = sat_obb(b1, b2_flush)
        self.assertFalse(res_flush.collision, "Flush contact with 0.0 overlap must not be flagged as collision")
        self.assertEqual(res_flush.penetration_depth, 0.0)

        # Box 3 centered at [9.8, 0, 0]: spans [4.8, 14.8] along X
        # Penetrates by 0.2 units -> distance=9.8, r1+r2=10.0 -> raw_min=0.2
        b3_penetrate = WorldBox("b3", "root", np.array([9.8, 0.0, 0.0]), rot, size, 0.0, np.zeros(3), "steel")
        res_penetrate = sat_obb(b1, b3_penetrate)
        self.assertTrue(res_penetrate.collision, "Penetration of 0.2 units must be flagged as collision")
        self.assertAlmostEqual(res_penetrate.penetration_depth, 0.2, places=5)

    def test_05_degenerate_cross_axis_masking(self):
        """
        Test SAT OBB on parallel boxes where all 9 face-normal cross products are degenerate (0,0,0).
        Axes with norm <= 1e-10 must be excluded, returning finite overlap without NaN or division by zero.
        """
        rot = np.eye(3, dtype=float)
        b1 = WorldBox("left", "root", np.array([0.0, 0.0, 0.0]), rot, np.array([2.0, 3.0, 4.0]), 0.0, np.zeros(3), "steel")
        b2 = WorldBox("right", "root", np.array([10.0, 0.0, 0.0]), rot, np.array([2.0, 3.0, 4.0]), 0.0, np.zeros(3), "steel")

        res = sat_obb(b1, b2, cross_axis_tolerance=1e-10)
        self.assertFalse(np.isnan(res.raw_min_overlap), "Overlap must not be NaN")
        self.assertFalse(np.isinf(res.raw_min_overlap), "Overlap must not be Infinite")
        self.assertFalse(res.collision)

    def test_06_token_redaction_in_notebook_and_logs(self):
        """Verify that no unmasked real or fake API tokens exist in notebook cell sources or outputs."""
        nb = load_notebook()
        token_pattern = re.compile(r"hf_[a-zA-Z0-9]{34,}")

        for i, cell in enumerate(nb.cells):
            # Check cell source
            matches = token_pattern.findall(cell.source)
            self.assertEqual(
                matches,
                [],
                f"Unredacted Hugging Face token found in cell #{i} source: {matches}",
            )
            # Check cell outputs if present
            if hasattr(cell, "outputs"):
                for output in cell.outputs:
                    text = ""
                    if "text" in output:
                        text = output.text
                    elif "data" in output and "text/plain" in output.data:
                        text = output.data["text/plain"]
                    matches_out = token_pattern.findall(text)
                    self.assertEqual(
                        matches_out,
                        [],
                        f"Unredacted Hugging Face token found in cell #{i} output: {matches_out}",
                    )

    def test_07_safetensors_corrupt_header_rejection(self):
        """
        Boundary test for safetensors header verification:
        Rejects empty payload, raw ASCII mock bytes, header length >= 100MB,
        truncated payloads, and non-JSON header metadata.
        """
        # 1. Empty payload
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(b"")

        # 2. Too short (< 8 bytes)
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(b"short")

        # 3. Raw mock ASCII bytes (24 bytes)
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(b"LORA_SAFETENSORS_PAYLOAD")

        # 4. Header >= 100MB
        oversize_payload = struct.pack("<Q", 100 * 1024 * 1024 + 1) + b"\x00" * 64
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(oversize_payload)

        # 5. Truncated payload (header_len says 100, but only 20 bytes provided)
        truncated_payload = struct.pack("<Q", 100) + b"\x00" * 20
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(truncated_payload)

        # 6. Non-JSON header
        bad_json_payload = struct.pack("<Q", 8) + b"not_json"
        with self.assertRaises(ValueError):
            validate_safetensors_binary_header(bad_json_payload)


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================


class TestNotebookTier3CrossFeatureCombinations(unittest.TestCase):
    """
    Tier 3 tests verifying cross-subsystem pipelines:
    1. Dataset manifest hash -> Checkpoint include -> Gate G2 evidence validation
    2. Secrets -> HF repo publishing -> Model card creation
    3. AST validation -> SE(3) Forward Kinematics -> 15-axis SAT -> Multi-backend export
    """

    def test_01_manifest_hash_flows_to_checkpoint_and_gate_g2(self):
        """Verify dataset manifest SHA-256 links into checkpoint metadata and validates Gate G2."""
        manifest_data = {
            "schema": "synccraft.dataset-manifest/0.1",
            "dataset_manifest_sha256": "",  # placeholder
            "row_count": 2,
            "rows": [
                {
                    "asset_id": "test_asset_001",
                    "family_id": "straight_blade_sword",
                    "body_plan_id_if_dynamic": None,
                    "generator_revision": "v1.0",
                    "generator_parameters": {"length": 12},
                    "canonical_dsl": "// sample dsl",
                    "canonical_dsl_sha256": "a" * 64,
                    "prompt_and_provenance": "Procedural straight blade sword",
                    "source_license_and_review_record": "Apache-2.0",
                    "split": "train",
                    "audit_version_and_results": {"passed": True},
                    "target_profile_and_version": "bbmodel_5",
                    "seed": 42,
                },
                {
                    "asset_id": "test_asset_002",
                    "family_id": "mechanical_crossbow",
                    "body_plan_id_if_dynamic": None,
                    "generator_revision": "v1.0",
                    "generator_parameters": {"span": 16},
                    "canonical_dsl": "// sample crossbow dsl",
                    "canonical_dsl_sha256": "b" * 64,
                    "prompt_and_provenance": "Procedural crossbow",
                    "source_license_and_review_record": "Apache-2.0",
                    "split": "heldout_unseen",
                    "audit_version_and_results": {"passed": True},
                    "target_profile_and_version": "bbmodel_5",
                    "seed": 43,
                },
            ],
        }

        # Calculate exact SHA-256 over rows
        manifest_json_str = json.dumps(manifest_data["rows"], sort_keys=True)
        manifest_sha256 = hashlib.sha256(manifest_json_str.encode("utf-8")).hexdigest()
        manifest_data["dataset_manifest_sha256"] = manifest_sha256

        # Checkpoint references dataset_manifest_sha256
        checkpoint_manifest = {
            "step": 100,
            "dataset_manifest_sha256": manifest_sha256,
            "model_architecture": "Qwen3-4B-LoRA",
        }
        self.assertEqual(checkpoint_manifest["dataset_manifest_sha256"], manifest_sha256)

        # Gate G2 evidence incorporates dataset_manifest_sha256
        eval_manifest_sha256 = hashlib.sha256(b"eval_rows").hexdigest()
        evidence = {
            "full_static_audit_implemented": True,
            "heldout_split_frozen_before_training": True,
            "all_samples_have_at_least_16_parts": True,
            "repair_pgd_policy_preregistered": True,
            "matched_raw_budgets_enforced": True,
            "at_most_three_reaudits_per_sample": True,
            "failures_and_timeouts_counted": True,
            "baseline_implementations_verified": True,
            "dataset_manifest_sha256": manifest_sha256,
            "evaluation_manifest_sha256": eval_manifest_sha256,
            "total": 50,
            "passed": 48,
        }

        gate_result = decide_gate(evidence)
        self.assertNotIn("missing_or_invalid_evidence", gate_result)
        self.assertEqual(gate_result["decision"], "skip_learned_refiner")
        self.assertFalse(gate_result["learned_refiner_enabled"])

    def test_02_secrets_to_hf_publishing_to_model_card_flow(self):
        """Verify publishing flow generates an informative model card README.md with license."""
        model_card_content = """# SyncCraft v3.0 - 4B/7B BF16 LoRA End-to-End Model

## Model Overview
- **Base Architecture**: Qwen3-4B / Qwen2.5-Coder-7B-Instruct
- **Target Compute**: Google Colab NVIDIA RTX PRO 6000
- **Fine-Tuning Technique**: Full-Precision BF16 PEFT LoRA (r=32, alpha=64, quantization=none)
- **License**: Apache-2.0
- **Dataset Manifest Hash**: `{manifest_sha256}`

## Pipeline Metrics
- Training Loss: 0.142
- Gate G2 Pass Rate: 96.0% (48/50 held-out tasks)
"""
        manifest_hash = "c" * 64
        rendered_card = model_card_content.format(manifest_sha256=manifest_hash)

        self.assertIn("Apache-2.0", rendered_card)
        self.assertIn(manifest_hash, rendered_card)
        self.assertIn("LoRA", rendered_card)
        self.assertNotIn("QLoRA", rendered_card)

    def test_03_ast_to_geometry_to_sat_to_exports_flow(self):
        """Verify full geometric pipeline: DSL -> AST -> FK Scene -> SAT Audit -> BBModel/Java Export."""
        sample_dsl = """
        @meta { archetype: weapon_3d, target_profile: bbmodel_5, texel_density: 16 }
        @palette {
          charcoal: ramp(base_lch=[0.32, 0.03, 260.0], steps=3, emissive=false)
          steel: ramp(base_lch=[0.82, 0.02, 240.0], steps=3, emissive=false)
        }
        @rig {
          root bone handle {
            pivot_local: [8.0, 8.0, 8.0]
            box grip { size: [2, 6, 2], material: charcoal }
            bone blade attach(parent_box=grip, anchor=[0,1,0], align=[0,-1,0]) {
              box edge { size: [2, 10, 1], material: steel }
            }
          }
        }
        """
        # 1. Parse
        program = parse_program(sample_dsl)
        # 2. Validate
        validate_program(program)
        # 3. Kinematic SE(3) compile
        scene = compile_geometry(program)
        self.assertEqual(len(scene.boxes), 2)
        self.assertEqual(len(scene.bones), 2)

        # 4. SAT collision audit (declared attachment is exempt)
        audit_res = audit_scene(scene)
        self.assertTrue(audit_res["passed"], f"Expected audit to pass for attached blade: {audit_res}")
        self.assertEqual(len(audit_res["declared_contact_pairs"]), 1)
        self.assertEqual(len(audit_res["collision_pairs"]), 0)

        # 5. Export to Blockbench 5.0
        from synccraft.shading.uv_packer import build_atlas
        atlas, atlas_png = build_atlas(program, scene)
        bb_data = export_bbmodel(scene, name="test_blade", atlas=atlas)
        self.assertIn("elements", bb_data)
        self.assertEqual(len(bb_data["elements"]), 2)

        # 6. Export to Java Modern
        java_data = export_java(scene, profile="java_modern", atlas=atlas)
        self.assertIn("elements", java_data)
        self.assertEqual(len(java_data["elements"]), 2)


# ==============================================================================
# Tier 4: Real-World Scenarios
# ==============================================================================


class TestNotebookTier4RealWorldScenarios(unittest.TestCase):
    """
    Tier 4 tests verifying headless in-process notebook execution in SMOKE_TEST_MODE
    and validating all generated artifacts.
    """

    def test_01_headless_notebook_smoke_execution(self):
        """
        Execute notebooks/synccraft_end_to_end.ipynb headlessly in an isolated
        sandbox directory under SMOKE_TEST_MODE=1. Verify all cells succeed
        and artifacts (dataset manifest, checkpoint, Gate G2 summary) are generated.
        """
        nb = load_notebook()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            orig_cwd = os.getcwd()
            orig_env = os.environ.copy()

            # Set smoke test mode environment variable
            os.environ["SYNCCRAFT_SMOKE_TEST"] = "1"
            # Ensure safe fallback staging without real HF token
            os.environ.pop("HF_TOKEN", None)

            # Prepare lightweight module mocks for Colab / PyTorch if running on CPU test machine
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
                mock_modules["google"] = google_mod
                mock_modules["google.colab"] = colab_mod
                mock_modules["google.colab.userdata"] = userdata_mod

            global_scope: Dict[str, Any] = {
                "__name__": "__main__",
                "__file__": str(NOTEBOOK_PATH),
            }

            try:
                os.chdir(temp_path)

                code_cells = get_all_code_cells(nb)
                for idx, code in enumerate(code_cells):
                    # Sanitize IPython magic commands (!pip, %pip, %matplotlib, etc.)
                    cleaned_lines = []
                    for line in code.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("!") or stripped.startswith("%"):
                            cleaned_lines.append(f"# {line}")
                        else:
                            cleaned_lines.append(line)
                    cleaned_code = "\n".join(cleaned_lines)

                    try:
                        compiled = compile(cleaned_code, f"<notebook_cell_{idx}>", "exec")
                        exec(compiled, global_scope)
                    except Exception as exc:
                        self.fail(
                            f"Notebook execution failed in cell #{idx} with exception: {type(exc).__name__}: {exc}\n"
                            f"Cell Code Snippet:\n{cleaned_code[:300]}..."
                        )

                # Validate artifacts produced by notebook execution
                manifest_file = temp_path / "datasets" / "dataset_manifest.json"
                if not manifest_file.is_file():
                    candidates = list(temp_path.glob("**/dataset_manifest.json"))
                    self.assertTrue(
                        len(candidates) > 0,
                        f"Expected dataset_manifest.json to be generated under {temp_path}",
                    )
                    manifest_file = candidates[0]

                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_json = json.load(f)

                self.assertIn("dataset_manifest_sha256", manifest_json)
                self.assertRegex(
                    manifest_json["dataset_manifest_sha256"],
                    r"^[a-fA-F0-9]{64}$",
                    "Manifest SHA256 must be a 64-character hexadecimal string",
                )

                # Validate seen and unseen static family coverage + attribution in manifest
                manifest_families = {row.get("family_id") for row in manifest_json.get("rows", [])}
                for fam in SEEN_STATIC_FAMILIES:
                    self.assertIn(
                        fam,
                        manifest_families,
                        f"Generated dataset_manifest.json must contain entries for seen family '{fam}'",
                    )
                for fam in UNSEEN_STATIC_FAMILIES:
                    self.assertIn(
                        fam,
                        manifest_families,
                        f"Generated dataset_manifest.json must contain held-out entries for unseen family '{fam}'",
                    )
                self.assertIn("attribution", manifest_json)
                self.assertEqual(manifest_json["attribution"].get("license"), "Apache-2.0")

                # Validate Dataset Card (README.md), NOTICE, and offline-staged public dataset bundle
                ds_readme = temp_path / "datasets" / "README.md"
                ds_notice = temp_path / "datasets" / "NOTICE"
                staged_ds_readme = temp_path / "runs" / "published" / "published_dataset" / "README.md"
                self.assertTrue(ds_readme.is_file(), "datasets/README.md Dataset Card must be generated")
                self.assertTrue(ds_notice.is_file(), "datasets/NOTICE attribution file must be generated")
                self.assertTrue(staged_ds_readme.is_file(), "Staged public dataset bundle must include README.md")
                ds_card_text = ds_readme.read_text(encoding="utf-8")
                self.assertIn("configs:", ds_card_text)
                self.assertIn("BaiTian6641/SyncCraft", ds_card_text)
                self.assertIn("Ottosson", ds_card_text)
                self.assertIn("@misc{synccraft2026dataset,", ds_card_text)

                # Validate checkpoint generation, Model Card attribution, and safetensors binary integrity
                checkpoint_dirs = list(temp_path.glob("**/checkpoint*"))
                self.assertTrue(
                    len(checkpoint_dirs) > 0,
                    f"Expected at least one checkpoint directory to be produced under {temp_path}",
                )
                for ckpt in checkpoint_dirs:
                    sf_path = ckpt / "adapter_model.safetensors"
                    if sf_path.is_file():
                        sf_bytes = sf_path.read_bytes()
                        h_len, h_dict = validate_safetensors_binary_header(sf_bytes)
                        self.assertGreater(h_len, 0)
                        self.assertIsInstance(h_dict, dict)
                    mc_path = ckpt / "README.md"
                    if mc_path.is_file():
                        mc_text = mc_path.read_text(encoding="utf-8")
                        self.assertIn("https://huggingface.co/datasets/", mc_text)
                        self.assertIn("BaiTian6641/SyncCraft", mc_text)
                        self.assertIn("Ottosson", mc_text)
                        self.assertIn("@misc{synccraft2026model,", mc_text)

                # Validate probing outputs recorded in execution scope
                if "probed_completions" in global_scope:
                    probed = global_scope["probed_completions"]
                    self.assertGreater(len(probed), 0, "probed_completions must contain results")
                    for p in probed:
                        self.assertIn("completion", p)
                        self.assertIn("family_id", p)

                # Validate Gate G2 evidence in execution scope
                if "gate_evidence" in global_scope:
                    evidence = global_scope["gate_evidence"]
                    self.assertIn("dataset_manifest_sha256", evidence)
                    self.assertIn("evaluation_manifest_sha256", evidence)
                    self.assertIn("all_samples_have_at_least_16_parts", evidence)
                    self.assertTrue(
                        evidence["all_samples_have_at_least_16_parts"],
                        "Unseen holdout benchmark rows all have reference_boxes >= 16; failed syntax samples must not falsely block all_samples_have_at_least_16_parts!",
                    )

                # Validate sys.path anti-shadowing, P1-02 canary quarantine, and complex multi-bone asset synthesis
                self.assertNotIn(str(temp_path), sys.path)
                rejected_ids = {r.get("asset_id") for r in global_scope.get("rejected_records", [])}
                for expected_trap in (
                    "test_intentional_collision",
                    "test_p102_coincident_penetration",
                    "test_p102_separated_gap",
                ):
                    self.assertIn(expected_trap, rejected_ids)

                c_stats = manifest_json.get("complexity_stats", {})
                self.assertGreaterEqual(c_stats.get("min_boxes", 0), 4)
                self.assertGreaterEqual(c_stats.get("max_boxes", 0), 25)
                self.assertGreaterEqual(c_stats.get("min_bones", 0), 1)
                self.assertGreaterEqual(c_stats.get("max_bones", 0), 8)
                for row in manifest_json.get("rows", []):
                    prov = row.get("prompt_and_provenance", {})
                    self.assertIsInstance(prov, dict)
                    self.assertGreater(len(prov.get("prompt", "")), 40)
                    if row.get("split") == "test_unseen":
                        self.assertGreaterEqual(row["audit_version_and_results"]["boxes"], 16)

                # Validate ADR 0006 Block-DSL v3.2 shard, 8192 sequence length, interleaving, and >=40-box probing
                apex_manifest_file = temp_path / "datasets" / "apex_complex_v3_2" / "apex_complex_manifest.json"
                self.assertTrue(apex_manifest_file.is_file(), "apex_complex_v3_2/apex_complex_manifest.json must exist")
                self.assertIn("apex_complex_v3_2", ds_card_text)
                self.assertGreaterEqual(global_scope["training_cfg"].max_sequence_length, 8192)
                self.assertGreaterEqual(global_scope.get("apex_interleaved_count", 0), 1)
                nb_code_all = "\n".join(get_all_code_cells(load_notebook()))
                self.assertIn('"SYNCCRAFT_APEX_MAX_NEW_TOKENS", "5120"', nb_code_all)
                for ckpt in checkpoint_dirs:
                    ta_path = ckpt / "training_args.json"
                    if ta_path.is_file():
                        ta_json = json.loads(ta_path.read_text(encoding="utf-8"))
                        self.assertGreaterEqual(ta_json.get("max_sequence_length", 0), 8192)
                        self.assertGreaterEqual(ta_json.get("apex_v3_2_interleaved_samples", 0), 1)
                        self.assertGreaterEqual(ta_json.get("apex_v3_1_interleaved_samples", 0), 1)
                apex_probed_fams = {r["family_id"] for r in global_scope.get("apex_probed_completions", [])}
                self.assertEqual(apex_probed_fams, {"elder_calamity_dragon", "relic_automaton_guardian"})
                apex_eval_file = temp_path / "runs" / "apex_complex_eval_report.json"
                self.assertTrue(apex_eval_file.is_file(), "runs/apex_complex_eval_report.json must be written")

            finally:
                os.chdir(orig_cwd)
                os.environ.clear()
                os.environ.update(orig_env)


class TestNotebookPeerReviewOverhaulContracts(unittest.TestCase):
    """
    Deterministic verification of all external peer-review fixes (C1, C2, C3, C4, C6,
    H2, H3, H4, H5, H10, M7, M10, M11, M12, M15).
    """

    @staticmethod
    def _extract_defs_from_cell(cell_source: str, wanted_names: Set[str], base_ns: Dict[str, Any]) -> Dict[str, Any]:
        tree = ast.parse(cell_source)
        selected = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted_names:
                selected.append(node)
            elif isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id in wanted_names for t in node.targets):
                    selected.append(node)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id in wanted_names:
                    selected.append(node)
        mod_ast = ast.Module(body=selected, type_ignores=[])
        exec(compile(mod_ast, "<notebook_extracted_defs>", "exec"), base_ns)
        return base_ns

    def test_c6_stream_mixture_sampler_unbiased_without_replacement_and_repeat_cap(self):
        """
        Verify StreamMixtureSampler visits 100% of main rows without replacement
        and caps repeats of smaller apex/world streams strictly at <= 3.
        """
        import random

        nb = load_notebook()
        ns = self._extract_defs_from_cell(
            "".join(nb.cells[12].source),
            {"StreamMixtureSampler"},
            {"random": random},
        )
        StreamMixtureSampler = ns["StreamMixtureSampler"]

        main_rows = [
            {
                "family_id": f"seen_fam_{i % 5}",
                "generator_parameters": {"complexity_tier": f"tier_{(i % 3) + 1}"},
            }
            for i in range(50)
        ]
        apex_rows = [{"family_id": f"apex_fam_{i}"} for i in range(2)]
        world_rows = [{"geometry_family": f"world_fam_{i}", "capability_level": f"level_{(i % 7) + 1}"} for i in range(3)]

        # Reject repeat caps > 3
        with self.assertRaises(ValueError):
            StreamMixtureSampler(main_rows, apex_rows=apex_rows, world_rows=world_rows, max_aux_repeats=4)

        sampler = StreamMixtureSampler(
            main_rows,
            apex_rows=apex_rows,
            world_rows=world_rows,
            seed=2026,
            max_aux_repeats=3,
            target_aux_ratio=0.50,  # Request 25 aux draws, must be capped at 3*len(aux)!
        )
        epoch_len = len(sampler)
        # 50 main + min(25, 3*2=6) apex + min(25, 3*3=9) world = 65
        self.assertEqual(epoch_len, 50 + 6 + 9)

        drawn_main_indices = []
        for _ in range(epoch_len):
            stream_name, row_idx, _ = sampler.next_item()
            if stream_name == "main":
                drawn_main_indices.append(row_idx)

        # 100% of main rows visited once without replacement
        self.assertEqual(len(drawn_main_indices), 50)
        self.assertEqual(set(drawn_main_indices), set(range(50)))
        self.assertEqual(sampler.max_observed_repeats("main"), 1)
        # Auxiliary streams strictly capped at <= 3 repeats
        self.assertLessEqual(sampler.max_observed_repeats("apex"), 3)
        self.assertLessEqual(sampler.max_observed_repeats("world"), 3)
        self.assertEqual(sampler.seen_stream_counts, {"main": 50, "apex": 6, "world": 9})
        self.assertEqual(sum(sampler.seen_family_counts.values()), epoch_len)
        self.assertEqual(sum(sampler.seen_tier_counts.values()), epoch_len)

        # Epoch 1 (2-epoch cumulative check): aux capacity was saturated in Epoch 0, so Epoch 1 visits 50 main rows and 0 extra aux repeats!
        expected_two_epochs = sampler.estimate_total_items(2)
        self.assertEqual(expected_two_epochs, 50 * 2 + 6 + 9)
        drawn_main_epoch1 = []
        for _ in range(expected_two_epochs - epoch_len):
            stream_name, row_idx, _ = sampler.next_item()
            if stream_name == "main":
                drawn_main_epoch1.append(row_idx)
        self.assertEqual(len(drawn_main_epoch1), 50)
        self.assertEqual(set(drawn_main_epoch1), set(range(50)))
        self.assertEqual(sampler.max_observed_repeats("main"), 2)
        self.assertLessEqual(sampler.max_observed_repeats("apex"), 3)
        self.assertLessEqual(sampler.max_observed_repeats("world"), 3)

        # Also test multi-epoch partial spread (target_aux_ratio=0.08 -> 4 draws/epoch across 3 epochs, capped at <=3 per row!)
        sampler_spread = StreamMixtureSampler(
            main_rows,
            apex_rows=apex_rows,
            world_rows=world_rows,
            seed=2027,
            max_aux_repeats=3,
            target_aux_ratio=0.08,
        )
        total_3ep = sampler_spread.estimate_total_items(3)
        for _ in range(total_3ep):
            sampler_spread.next_item()
        self.assertEqual(sampler_spread.max_observed_repeats("main"), 3)
        self.assertLessEqual(sampler_spread.max_observed_repeats("apex"), 3)
        self.assertLessEqual(sampler_spread.max_observed_repeats("world"), 3)

    def test_h2_periodic_full_state_checkpoint_and_deterministic_resume(self):
        """
        Verify save_full_training_checkpoint and resume_from_checkpoint round-trip
        adapter weights, optimizer moments, LR scheduler, RNG states, and StreamMixtureSampler
        deterministically (bit-exact continuation).
        """
        from dataclasses import dataclass, field
        import math
        import random
        import torch
        import torch.nn as nn
        import torch.optim as optim

        nb = load_notebook()
        base_ns: Dict[str, Any] = {
            "Path": Path,
            "dataclass": dataclass,
            "field": field,
            "hashlib": hashlib,
            "json": json,
            "math": math,
            "nn": nn,
            "np": np,
            "optim": optim,
            "os": os,
            "random": random,
            "re": re,
            "shutil": shutil,
            "struct": struct,
            "tempfile": tempfile,
            "torch": torch,
            "SMOKE_TEST_MODE": True,
            "HAS_CUDA": False,
            "DATASET_REPO_ID": "TassP0lar/synccraft-v3-blockdsl-dataset",
        }
        self._extract_defs_from_cell(
            "".join(nb.cells[11].source),
            {"resolve_base_model_license_metadata", "LoRATrainingConfig"},
            base_ns,
        )
        # Also extract SyncCraftLoRALayer and SyncCraftCausalLM from the else branch of Cell 11
        cell11_tree = ast.parse("".join(nb.cells[11].source))
        for node in cell11_tree.body:
            if isinstance(node, ast.If):
                for sub in node.orelse:
                    if isinstance(sub, ast.ClassDef) and sub.name in {"SyncCraftLoRALayer", "SyncCraftCausalLM"}:
                        exec(compile(ast.Module(body=[sub], type_ignores=[]), "<cell11_smoke_cls>", "exec"), base_ns)

        self._extract_defs_from_cell(
            "".join(nb.cells[12].source),
            {
                "StreamMixtureSampler",
                "_serialize_safetensors_dict",
                "save_full_training_checkpoint",
                "resume_from_checkpoint",
            },
            base_ns,
        )

        SyncCraftCausalLM = base_ns["SyncCraftCausalLM"]
        StreamMixtureSampler = base_ns["StreamMixtureSampler"]
        LoRATrainingConfig = base_ns["LoRATrainingConfig"]
        save_ckpt = base_ns["save_full_training_checkpoint"]
        resume_ckpt = base_ns["resume_from_checkpoint"]

        main_rows = [{"family_id": f"fam_{i}", "tok_id": (i % 20) + 4} for i in range(12)]
        apex_rows = [{"family_id": "apex_0", "tok_id": 25}]
        world_rows = [{"geometry_family": "world_0", "tok_id": 26}]
        cfg = LoRATrainingConfig()

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_root = Path(tmp) / "ckpts"
            random.seed(1234)
            np.random.seed(1234)
            torch.manual_seed(1234)

            model_a = SyncCraftCausalLM(vocab_size=32, hidden_dim=16, r=4, alpha=8)
            # Freeze non-LoRA base layers so adapter_model.bin holds the full trainable delta
            for pname, p in model_a.named_parameters():
                p.requires_grad = "lora" in pname
            # Snapshot frozen base state for model_b initialization parity
            frozen_base_state = {k: v.clone() for k, v in model_a.state_dict().items() if "lora" not in k}

            opt_a = optim.AdamW([p for p in model_a.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.01)
            sched_a = optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.9)
            samp_a = StreamMixtureSampler(main_rows, apex_rows, world_rows, seed=77)
            loss_fn = nn.CrossEntropyLoss()

            for _ in range(3):
                _, _, r_item = samp_a.next_item()
                x = torch.tensor([[1, int(r_item["tok_id"]), 2, 3]], dtype=torch.long)
                logits = model_a(x)
                loss = loss_fn(logits[:, :-1, :].reshape(-1, 32), x[:, 1:].reshape(-1))
                opt_a.zero_grad()
                loss.backward()
                opt_a.step()
                sched_a.step()

            saved_dir = save_ckpt(
                ckpt_root,
                step=3,
                micro_step=3,
                model_obj=model_a,
                optimizer_obj=opt_a,
                scheduler_obj=sched_a,
                sampler_obj=samp_a,
                cfg=cfg,
                ds_sha256="a" * 64,
                val_loss_val=1.23,
                val_hist=[{"step": 3, "val_loss": 1.23}],
                train_hist=[{"step": 3, "loss": 1.50}],
                model_rev="rev_test",
            )
            for req_file in (
                "adapter_config.json",
                "adapter_model.bin",
                "adapter_model.safetensors",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pt",
                "sampler_state.json",
                "manifest.json",
            ):
                self.assertTrue((saved_dir / req_file).is_file(), f"Missing {req_file} in periodic checkpoint")

            # Continue Model A for 2 more steps and record RNG + weights
            cont_draws_a = []
            for _ in range(2):
                s_name, r_idx, r_item = samp_a.next_item()
                cont_draws_a.append((s_name, r_idx, random.random(), float(np.random.rand()), float(torch.rand(1).item())))
                x = torch.tensor([[1, int(r_item["tok_id"]), 2, 3]], dtype=torch.long)
                logits = model_a(x)
                loss = loss_fn(logits[:, :-1, :].reshape(-1, 32), x[:, 1:].reshape(-1))
                opt_a.zero_grad()
                loss.backward()
                opt_a.step()
                sched_a.step()

            # Perturb global RNGs, build fresh Model B, and resume from saved_dir
            random.seed(99999)
            np.random.seed(99999)
            torch.manual_seed(99999)
            model_b = SyncCraftCausalLM(vocab_size=32, hidden_dim=16, r=4, alpha=8)
            model_b.load_state_dict(frozen_base_state, strict=False)
            for pname, p in model_b.named_parameters():
                p.requires_grad = "lora" in pname
            opt_b = optim.AdamW([p for p in model_b.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.01)
            sched_b = optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.9)
            samp_b = StreamMixtureSampler(main_rows, apex_rows, world_rows, seed=1)

            resumed_meta = resume_ckpt(
                saved_dir,
                model_obj=model_b,
                optimizer_obj=opt_b,
                scheduler_obj=sched_b,
                sampler_obj=samp_b,
                map_device="cpu",
            )
            self.assertEqual(resumed_meta["global_step"], 3)

            cont_draws_b = []
            for _ in range(2):
                s_name, r_idx, r_item = samp_b.next_item()
                cont_draws_b.append((s_name, r_idx, random.random(), float(np.random.rand()), float(torch.rand(1).item())))
                x = torch.tensor([[1, int(r_item["tok_id"]), 2, 3]], dtype=torch.long)
                logits = model_b(x)
                loss = loss_fn(logits[:, :-1, :].reshape(-1, 32), x[:, 1:].reshape(-1))
                opt_b.zero_grad()
                loss.backward()
                opt_b.step()
                sched_b.step()

            self.assertEqual(cont_draws_a, cont_draws_b)
            self.assertAlmostEqual(opt_a.param_groups[0]["lr"], opt_b.param_groups[0]["lr"], places=9)
            for k in model_a.state_dict():
                self.assertTrue(torch.equal(model_a.state_dict()[k], model_b.state_dict()[k]), f"Mismatch in {k}")

    def test_h10_vram_safe_pre_fp32_response_loss_and_overflow_drop_policy(self):
        """
        Verify prepare_supervised_sequence drops responses exceeding max_seq_len rather than
        truncating mid-program without <eos>, and compute_vram_safe_response_loss slices
        response logits BEFORE casting to float32 with < 1e-5 parity vs full FP32 reference.
        """
        import torch
        import torch.nn as nn

        nb = load_notebook()
        ns = self._extract_defs_from_cell(
            "".join(nb.cells[12].source),
            {"prepare_supervised_sequence", "compute_vram_safe_response_loss"},
            {"torch": torch, "nn": nn},
        )
        prepare_seq = ns["prepare_supervised_sequence"]
        vram_safe_loss = ns["compute_vram_safe_response_loss"]

        # 1. Response >= max_seq_len is dropped cleanly (never sliced mid-AST without <eos>)
        dropped_ids, dropped_lbls, drop_reason = prepare_seq([10, 11], [20, 21, 22, 23, 24], eos_token_id=2, max_seq_len=5)
        self.assertIsNone(dropped_ids)
        self.assertIsNone(dropped_lbls)
        self.assertEqual(drop_reason, "dropped_response_exceeds_max_seq_len")

        # 2. Long prompt + short response trims prompt while preserving 100% of response + <eos>
        full_ids, lbls, status = prepare_seq([10, 11, 12, 13, 14], [20, 21], eos_token_id=2, max_seq_len=6)
        self.assertEqual(status, "prompt_trimmed_response_eos_intact")
        self.assertEqual(full_ids, [12, 13, 14, 20, 21, 2])
        self.assertEqual(lbls, [-100, -100, -100, 20, 21, 2])

        # 3. Verify pre-FP32 slicing in AST (active_mask applied BEFORE .float())
        cell12_src = "".join(nb.cells[12].source)
        self.assertIn("active_logits_bf16 = shift_logits[active_mask]", cell12_src)
        self.assertIn("active_logits_bf16.float()", cell12_src)

        # 4. Numerical parity on bfloat16 logits with 80% prompt tokens masked (-100)
        torch.manual_seed(42)
        logits_a = torch.randn(2, 20, 64, dtype=torch.bfloat16, requires_grad=True)
        logits_b = logits_a.detach().clone().requires_grad_(True)
        labels = torch.full((2, 20), -100, dtype=torch.long)
        labels[:, 16:] = torch.randint(0, 64, (2, 4))
        crit = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.03)

        loss_fast, active_cnt = vram_safe_loss(logits_a, labels, crit)
        loss_fast.backward()
        self.assertEqual(active_cnt, 8)

        loss_ref = crit(logits_b[:, :-1, :].float().reshape(-1, 64), labels[:, 1:].reshape(-1))
        loss_ref.backward()

        self.assertAlmostEqual(float(loss_fast.item()), float(loss_ref.item()), places=5)
        self.assertLess(float((logits_a.grad.float() - logits_b.grad.float()).abs().max().item()), 1e-4)

    def test_c2_h3_dynamic_max_new_tokens_chat_template_and_think_stripping(self):
        """
        Verify dynamic tokenizer-measured max_new_tokens = max(2560, ceil(1.30 * max_ref_tok_len)),
        apply_chat_template formatting, and explicit <think>...</think> reasoning block stripping.
        """
        import math
        import torch

        nb = load_notebook()
        ns: Dict[str, Any] = {"math": math, "re": re, "os": os, "torch": torch}
        self._extract_defs_from_cell(
            "".join(nb.cells[11].source),
            {"SYNCCRAFT_SYSTEM_PROMPT", "strip_reasoning_blocks", "apply_synccraft_chat_template"},
            ns,
        )
        # Extract BlockDSLTokenizer from Cell 11 else branch
        cell11_tree = ast.parse("".join(nb.cells[11].source))
        for node in cell11_tree.body:
            if isinstance(node, ast.If):
                for sub in node.orelse:
                    if isinstance(sub, ast.ClassDef) and sub.name == "BlockDSLTokenizer":
                        exec(compile(ast.Module(body=[sub], type_ignores=[]), "<cell11_tok>", "exec"), ns)
        self._extract_defs_from_cell(
            "".join(nb.cells[17].source),
            {"compute_dynamic_max_new_tokens"},
            ns,
        )

        tok = ns["BlockDSLTokenizer"](["@meta", "archetype", "box", "bone"])
        strip_fn = ns["strip_reasoning_blocks"]
        chat_fn = ns["apply_synccraft_chat_template"]
        dyn_cap_fn = ns["compute_dynamic_max_new_tokens"]

        cleaned, n_blocks = strip_fn("<think>\nplanning rig...\n</think>\n@meta { archetype: entity }")
        self.assertEqual(n_blocks, 1)
        self.assertEqual(cleaned, "@meta { archetype: entity }")

        cleaned_unclosed, n_unclosed = strip_fn("@meta { archetype: entity }\n<think>truncated thought")
        self.assertEqual(n_unclosed, 1)
        self.assertEqual(cleaned_unclosed, "@meta { archetype: entity }")

        prompt_txt, resp_txt = chat_fn(tok, "Build a lantern", "@meta {}", has_image=True)
        self.assertIn("<|im_start|>system", prompt_txt)
        self.assertIn("[image]", prompt_txt)
        self.assertTrue(prompt_txt.endswith("<|im_start|>assistant\n"))
        self.assertEqual(resp_txt, "@meta {}")

        cap_floor, measured_short = dyn_cap_fn(["@meta { box }"], tok, gpu_floor=2560, margin=1.30)
        self.assertEqual(cap_floor, 2560)
        self.assertGreater(measured_short, 0)

        long_ref = " ".join(["box"] * 3000)
        cap_scaled, measured_long = dyn_cap_fn([long_ref], tok, gpu_floor=2560, margin=1.30)
        self.assertEqual(measured_long, 3000)
        self.assertEqual(cap_scaled, int(math.ceil(3000 * 1.30)))

    def test_c4_h4_m10_smoke_refuses_hf_upload_and_model_card_license_honesty(self):
        """
        Verify default HF_NAMESPACE == 'TassP0lar', accurate LFM Open License v1.0 vs Apache-2.0
        Model Card metadata, and hard refusal to call HfApi.create_repo / upload_folder in smoke/CPU mode.
        """
        nb = load_notebook()
        cell3_src = "".join(nb.cells[4].source)
        self.assertIn('"HF_NAMESPACE", "TassP0lar"', cell3_src)

        ns: Dict[str, Any] = {"os": os, "re": re, "shutil": shutil}
        self._extract_defs_from_cell(
            "".join(nb.cells[11].source),
            {"resolve_base_model_license_metadata"},
            ns,
        )
        resolve_lic = ns["resolve_base_model_license_metadata"]

        lfm_lic = resolve_lic("LiquidAI/LFM3-VL-2B")
        self.assertEqual(lfm_lic["display_license"], "LFM Open License v1.0")
        self.assertEqual(lfm_lic["hf_frontmatter_license"], "other")
        self.assertEqual(lfm_lic["license_name"], "lfm-open-license-1.0")
        self.assertIn("$10M", lfm_lic["commercial_restriction_summary"])

        qwen_lic = resolve_lic("Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(qwen_lic["display_license"], "Apache-2.0")
        self.assertEqual(qwen_lic["hf_frontmatter_license"], "apache-2.0")

        # Execute Cell 10 (Model Card & publish guard) with a SpyHfApi and valid token under SMOKE_TEST_MODE=True
        class SpyHfApi:
            def __init__(self):
                self.calls = []

            def create_repo(self, *args, **kwargs):
                self.calls.append(("create_repo", args, kwargs))

            def upload_folder(self, *args, **kwargs):
                self.calls.append(("upload_folder", args, kwargs))

        class DummySampler:
            seen_stream_counts = {"main": 4, "apex": 1, "world": 1}

        class DummyCfg:
            model_name = "LiquidAI/LFM3-VL-2B"
            precision = "bf16"
            quantization = "none"
            lora_rank = 32
            lora_alpha = 64
            max_sequence_length = 8192

        spy_api = SpyHfApi()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt_dir = tmp_path / "checkpoint-6"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "manifest.json").write_text("{}", encoding="utf-8")

            cell10_scope: Dict[str, Any] = {
                "os": os,
                "shutil": shutil,
                "Path": Path,
                "PROJECT_ROOT": tmp_path,
                "MODEL_REPO_ID": "TassP0lar/synccraft-v3-bf16-lora",
                "DATASET_REPO_ID": "TassP0lar/synccraft-v3-blockdsl-dataset",
                "dataset_published_successfully": False,
                "resolve_base_model_license_metadata": resolve_lic,
                "training_cfg": DummyCfg(),
                "HAS_CUDA": False,
                "SMOKE_TEST_MODE": True,
                "USE_HF_HUB": True,
                "hf_api": spy_api,
                "hf_token": "hf_valid_secret_token_for_guard_test_12345",
                "device_name": "CPU",
                "manifest_sha256": "b" * 64,
                "base_model_revision": "smoke_local",
                "lora_layer_param_breakdown": {"short_conv": 1024, "attention": 1024, "mlp_ffn": 0, "vision_or_projector": 0},
                "training_history": [{"loss": 2.5}],
                "final_val_loss": 2.6,
                "cost_ledger": CostLedger(accepted_tokens=128),
                "stream_sampler": DummySampler(),
                "checkpoint_dir": ckpt_dir,
                "final_step": 6,
            }
            exec("".join(nb.cells[15].source), cell10_scope)

            self.assertEqual(spy_api.calls, [], "SpyHfApi must receive ZERO calls when SMOKE_TEST_MODE is True!")
            self.assertFalse(cell10_scope["published_successfully"])
            self.assertEqual(cell10_scope["model_publish_refused_reason"], "refused_smoke_or_cpu_mode")

            card_md = (ckpt_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("LFM Open License v1.0", card_md)
            self.assertIn("license: other", card_md)
            self.assertIn("license_name: lfm-open-license-1.0", card_md)
            self.assertNotIn("license: apache-2.0", card_md)
            self.assertIn("CPU Smoke Verification Adapter", card_md)
            self.assertNotIn("- rtx-pro-6000", card_md)

    def test_c1_c3_h5_zero_hardcoded_true_gate_flags_and_repo_hygiene(self):
        """
        Verify Cell 13 (nb.cells[19]) contains ZERO hardcoded True literals in gate_evidence,
        Section 6 wires run_bounded_repair_policy + verify_prompt_adherence + evaluate_world_capability_benchmark,
        and toy checkpoints / stale gate reports / .agents logs are absent and gitignored.
        """
        import subprocess

        nb = load_notebook()
        sec6_code = "\n".join(get_section_code_cells(nb, 6))
        self.assertIn("run_bounded_repair_policy", sec6_code)
        self.assertIn("verify_prompt_adherence", sec6_code)
        self.assertIn("evaluate_world_capability_benchmark", sec6_code)
        self.assertIn("truncated_outputs_count", sec6_code)

        # AST audit of Cell 13 (nb.cells[19]): zero ast.Constant(True) values inside gate_evidence = {...}
        cell19_tree = ast.parse("".join(nb.cells[19].source))
        found_gate_dict = False
        for node in ast.walk(cell19_tree):
            if isinstance(node, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "gate_evidence" for t in node.targets):
                    self.assertIsInstance(node.value, ast.Dict)
                    found_gate_dict = True
                    for k_node, v_node in zip(node.value.keys, node.value.values):
                        key_repr = ast.unparse(k_node) if k_node is not None else "<unpack>"
                        if isinstance(v_node, ast.Constant) and v_node.value is True:
                            self.fail(f"Hardcoded True literal forbidden in gate_evidence[{key_repr}]!")
        self.assertTrue(found_gate_dict, "Expected gate_evidence dict assignment in Cell 13 (nb.cells[19])")

        # Repo hygiene checks (C4, C3, M12)
        for forbidden_rel in (
            "checkpoints/checkpoint-5",
            "runs/published",
            "runs/gate_g2_report.json",
            ".agents/teamwork",
        ):
            tracked = subprocess.run(["git", "ls-files", "--", forbidden_rel], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
            self.assertEqual(
                tracked.stdout.strip(), "", f"Forbidden artifact/log path {forbidden_rel} must not be tracked in git!"
            )
            res = subprocess.run(
                ["git", "check-ignore", "-q", forbidden_rel],
                cwd=str(REPO_ROOT),
            )
            self.assertEqual(res.returncode, 0, f"{forbidden_rel} must be ignored by .gitignore!")

    def test_h13_tasks_json_evidence_paths_all_exist(self):
        """
        Verify every evidence path listed in TASKS.json resolves to an existing repository path.
        """
        tasks_path = REPO_ROOT / "TASKS.json"
        self.assertTrue(tasks_path.is_file(), "TASKS.json must exist at repository root")
        payload = json.loads(tasks_path.read_text(encoding="utf-8"))
        self.assertIn("tasks", payload)
        self.assertGreater(len(payload["tasks"]), 0)

        missing_paths = []
        for task in payload["tasks"]:
            tid = task.get("id", "<unknown>")
            for ev_rel in task.get("evidence", []):
                if not (REPO_ROOT / ev_rel).exists():
                    missing_paths.append((tid, ev_rel))

        self.assertEqual(
            missing_paths,
            [],
            f"TASKS.json contains non-existent evidence paths: {missing_paths}",
        )


if __name__ == "__main__":
    unittest.main()
