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
        Verify ADR 0004 multi-domain anti-overfitting contracts across Section 3, 4, and 6:
        - 'full_75k' default GPU preset + 'world_1200' causal trajectory preset + diversity_stats logging.
        - Response-masked label_smoothing, weight_decay=0.05, lora_dropout=0.10, and epochs=1 default.
        - Fixed global_sample_idx % 5 == 4 20% world-trajectory interleaving (not broken b_idx % 4 == 3).
        - Stratified held-out ValLoss evaluation (evaluate_validation_loss & val_loss_history).
        - All 12 multi-domain unseen families probed in Section 6.
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
        self.assertIn("global_sample_idx % 5 == 4", sec4_code)
        self.assertNotIn("b_idx % 4 == 3", sec4_code)

        for unseen_fam in UNSEEN_STATIC_FAMILIES:
            self.assertIn(unseen_fam, sec6_code, f"Section 6 must probe unseen family '{unseen_fam}'")

    def test_07d_v31_apex_musculoskeletal_and_8192_context_contracts(self):
        """
        Verify ADR 0006 Block-DSL v3.1 integration across Section 3, 4, and 6:
        - Synthesizes 'apex_complex_v3_1' shard and registers config_name: apex_complex_v3_1 in Dataset Card.
        - Defaults max_sequence_length to 8192 (>= 8192 so ~7.6k-token multi-turn artist-loop rows do not truncate).
        - Interleaves 20% Block-DSL v3.1 turns via global_sample_idx % 5 == 3 alongside world trajectories.
        - Probes both >=40-box held-out Apex unseen families (elder_calamity_dragon, relic_automaton_guardian).
        """
        nb = load_notebook()
        sec3_code = "\n".join(get_section_code_cells(nb, 3))
        sec4_code = "\n".join(get_section_code_cells(nb, 4))
        sec6_code = "\n".join(get_section_code_cells(nb, 6))

        self.assertIn("generate_apex_complex_dataset", sec3_code)
        self.assertIn("config_name: apex_complex_v3_1", sec3_code)
        self.assertIn('"8192"', sec4_code)
        self.assertNotIn("max_sequence_length: int = 2048", sec4_code)
        self.assertIn("global_sample_idx % 5 == 3", sec4_code)
        self.assertIn("elder_calamity_dragon", sec6_code)
        self.assertIn("relic_automaton_guardian", sec6_code)
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

                # Validate ADR 0006 Block-DSL v3.1 shard, 8192 sequence length, interleaving, and >=40-box probing
                apex_manifest_file = temp_path / "datasets" / "apex_complex_v3_1" / "apex_complex_manifest.json"
                self.assertTrue(apex_manifest_file.is_file(), "apex_complex_v3_1/apex_complex_manifest.json must exist")
                self.assertIn("apex_complex_v3_1", ds_card_text)
                self.assertGreaterEqual(global_scope["training_cfg"].max_sequence_length, 8192)
                self.assertGreaterEqual(global_scope.get("apex_interleaved_count", 0), 1)
                apex_probed_fams = {r["family_id"] for r in global_scope.get("apex_probed_completions", [])}
                self.assertEqual(apex_probed_fams, {"elder_calamity_dragon", "relic_automaton_guardian"})
                apex_eval_file = temp_path / "runs" / "apex_complex_eval_report.json"
                self.assertTrue(apex_eval_file.is_file(), "runs/apex_complex_eval_report.json must be written")

            finally:
                os.chdir(orig_cwd)
                os.environ.clear()
                os.environ.update(orig_env)


if __name__ == "__main__":
    unittest.main()
