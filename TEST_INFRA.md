# SyncCraft End-to-End Test Infrastructure Specification

**Document Version**: 2.0.0  
**Target Artifact**: `notebooks/synccraft_end_to_end.ipynb`  
**Test Suite Path**: `tests/test_notebook_e2e.py`  
**Integrity Mode**: Production Quality Gate / Zero-Facade Enforcement  
**Author**: Test Writer (`test_writer_e2e_2`)  
**Date**: 2026-11-14T04:22:00Z  

---

## 1. Overview and Anti-Facade Testing Philosophy

This document defines the hardened test infrastructure, execution harness, and verification standards for the SyncCraft Production Colab End-to-End Pipeline notebook (`notebooks/synccraft_end_to_end.ipynb`). 

The test strategy strictly follows **uncompromising, requirement-driven, opaque-box testing**:
- **Zero-Facade Enforcement**: The test suite employs deep AST structural analysis, binary header decoding, and execution state auditing to strictly disallow mock implementations, synthetic integer training on random noise, ground-truth substitution during probing, and fabricated gate attestation.
- **Progressive Testability**: Organized into hierarchical tiers (Tiers 1 through 4), verifying fundamental structural integrity, parameter boundaries, subsystem integration, and end-to-end headless execution.
- **Authoritative Mathematical & Structural Derivations**: Every expected value, threshold, and interface contract is derived from explicit specifications in `ORIGINAL_REQUEST.md`, `PROJECT.md`, `AGENTS.md`, `configs/data_policy.yaml`, `configs/training_static.yaml`, and `src/synccraft/evaluation/gate_g2.py`.

---

## 2. Authoritative Sources of Truth

Every expected value, threshold, and contract in the test suite is derived from explicit authoritative specifications:

| Requirement Area | Authoritative Source Document | Key Clauses / Invariants Enforced |
|---|---|---|
| **Pipeline Requirements** | `ORIGINAL_REQUEST.md` | R1 (Environment/Secrets), R2 (Dataset/Audit), R3 (4B/7B BF16 LoRA Training on Colab RTX PRO 6000), R4 (Probing/Gate G2) |
| **Interface Contracts** | `.agents/teamwork/PROJECT.md` § Interface Contracts | Delineated section headings, `SMOKE_TEST_MODE`, checkpoint layout, dataset manifest hash format |
| **Architectural Rules** | `AGENTS.md` | Shared SE(3) kinematics, direct 15-axis SAT OBB collision audit, degenerate cross-axis masking, token redaction |
| **Data Policy & Families** | `configs/data_policy.yaml` | 8 seen static families (`straight_blade_sword`, `single_edge_blade`, `battle_axe`, `spear`, `war_hammer`, `simple_wood_bow`, `furniture_table_chair_chest_cabinet`, `gearbox`), 6 unseen held-out families, 13 mandatory row manifest fields |
| **Training & Memory Config** | `configs/training_static.yaml` | $r=32$, $\alpha=64$, response-tokens loss masking, full-precision BF16 LoRA (`quantization: none`) on Colab RTX PRO 6000 for 4B & 7B, atomic checkpoints, genuine sample tokenization |
| **Safetensors Binary Spec** | Hugging Face / Safetensors Standard | 8-byte little-endian header length $N = \text{struct.unpack}("<Q", payload[:8])[0]$, $N < 100\text{ MB}$, valid JSON header dictionary, no raw ASCII mock payloads |
| **Gate G2 Decision Logic** | `src/synccraft/evaluation/gate_g2.py` | 8 mandatory checklist prerequisites, 64-hex SHA-256 validation, $\ge 16$ parts dynamic verification, 90% pass rate threshold |

---

## 3. Test Tier Architecture (25 Tests)

The expanded test suite in `tests/test_notebook_e2e.py` implements 25 comprehensive tests across four testing tiers:

```
+--------------------------------------------------------------------------------+
| Tier 4: Real-World Scenarios (1 Test)                                          |
| - test_01_headless_notebook_smoke_execution:                                   |
|   Headless in-process sandbox execution; validates dataset manifest (all 8     |
|   seen families present), atomic checkpoint directory, safetensors binary      |
|   header (<100MB unpack), probed completion records, dynamic Gate G2 evidence  |
+--------------------------------------------------------------------------------+
                                       ▲
+--------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Integration (3 Tests)                                    |
| - test_01_manifest_hash_flows_to_checkpoint_and_gate_g2                        |
| - test_02_secrets_to_hf_publishing_to_model_card_flow                          |
| - test_03_ast_to_geometry_to_sat_to_exports_flow                               |
+--------------------------------------------------------------------------------+
                                       ▲
+--------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases (7 Tests)                                      |
| - test_01_missing_hf_token_graceful_fallback                                   |
| - test_02_missing_cuda_cpu_smoke_mode                                          |
| - test_03_empty_and_malformed_candidate_dsl_rejection                          |
| - test_04_sat_flush_contact_vs_penetration_boundary                            |
| - test_05_degenerate_cross_axis_masking                                        |
| - test_06_token_redaction_in_notebook_and_logs                                 |
| - test_07_safetensors_corrupt_header_rejection: Rejects mock ASCII, header     |
|   length >= 100MB, truncated payloads, and non-JSON metadata                   |
+--------------------------------------------------------------------------------+
                                       ▲
+--------------------------------------------------------------------------------+
| Tier 1: Feature Coverage & Structural Compliance (14 Tests)                    |
| - test_01_notebook_file_exists                                                 |
| - test_02_notebook_valid_json_and_nbformat_v4                                  |
| - test_03_markdown_sections_delineated (Sections 1-6)                          |
| - test_04_colab_secrets_and_env_fallback                                       |
| - test_05_dataset_gathering_and_sat_filtering                                  |
| - test_05b_seen_static_families_complete_coverage: All 8 seen static families  |
| - test_06_dataset_manifest_schema_and_hashing                                  |
| - test_07_lora_4b_7b_training_configurations: AST verification of              |
|   AutoModelForCausalLM, AutoTokenizer, bfloat16, peft/LoraConfig/get_peft_model|
|   (no 4-bit QLoRA/BitsAndBytes), dataset sample tokenization, and strict       |
|   prohibition of torch.randint noise training                                  |
| - test_08_checkpoint_structure_contract: Manifest/adapters & ban on mock bytes |
| - test_08b_safetensors_binary_header_contract: struct.unpack("<Q", payload[:8])|
|   < 100MB, valid JSON, no b"LORA_SAFETENSORS_PAYLOAD"                          |
| - test_09_hf_publishing_idempotency_and_model_card                             |
| - test_10_probing_unseen_families_and_gate_g2                                  |
| - test_10b_model_probing_queries_model_generation: Ban on raw ground-truth     |
|   substitution task["dsl"]; AST check that probing queries model generation    |
| - test_10c_gate_g2_prerequisites_dynamic_evaluation: Ban on hardcoded         |
|   all_samples_have_at_least_16_parts: True; dynamic decide_gate blocking       |
+--------------------------------------------------------------------------------+
```

---

## 4. Specific Anti-Facade Verification Mechanics

### 4.1 AST Verification of GPU Pipeline & Dataset Ingestion
Tests parse notebook code cells using Python's `ast` module and inspect syntax trees:
- `AutoModelForCausalLM`: Must appear as an identifier/attribute in Section 4.
- `AutoTokenizer`: Must appear as an identifier/attribute in Section 4.
- `peft` / `LoraConfig` / `get_peft_model`: Must appear in the LoRA setup.
- Dataset Sample Tokenization: Must process dataset sample strings (`sample["canonical_dsl"]` / `sample["prompt"]`) through the tokenizer with label masking (`-100`).
- Random Noise Ban: Disallows `torch.randint` generation of fake token sequences in the training loop.

### 4.2 Safetensors Binary Header Decoding Specification
Per the Hugging Face Safetensors format specification:
1. The first 8 bytes represent an unsigned 64-bit little-endian integer (`<Q`), giving header length $N$.
2. Header Length Guard: $0 < N < 100 \times 1024 \times 1024$ (strictly less than 100MB).
3. Payload Boundary: $8 + N \le \text{len}(payload)$.
4. JSON Parse: The slice `payload[8 : 8 + N]` must decode as UTF-8 and parse into a valid JSON dictionary containing tensor names, dtypes, shapes, and data offsets.
5. Mock Payload Ban: Literal ASCII mock byte strings such as `b"LORA_SAFETENSORS_PAYLOAD"` fail both the length guard ($N = 5,062,419,124,724,059,980$ bytes $> 100$MB) and explicit content assertions.

### 4.3 Probing Model Generation Integrity
- Verifies that Section 6 probing calls the model's generation routine (`model.generate`, `model(...)`, or dedicated generation function).
- Rejects ground-truth substitution bypass: `raw_completion_text = task["dsl"]` is strictly disallowed.
- Verifies that completions audited by SAT originate from model outputs.

### 4.4 Dynamic Gate G2 Prerequisite Evaluation
- Falsely asserting `"all_samples_have_at_least_16_parts": True` when models have 3–4 parts is strictly prohibited.
- The test verifies that `all_samples_have_at_least_16_parts` is computed dynamically from scene box counts.
- Verifies that `decide_gate` yields `decision: blocked` when prerequisites are not met on non-dense assemblies.

### 4.5 All 8 Seen Static Families Coverage
- Enforces that Section 3 defines templates/generators for all 8 families declared in `configs/data_policy.yaml`:
  1. `straight_blade_sword`
  2. `single_edge_blade`
  3. `battle_axe`
  4. `spear`
  5. `war_hammer`
  6. `simple_wood_bow`
  7. `furniture_table_chair_chest_cabinet`
  8. `gearbox`
- Verifies that headless execution outputs manifest rows covering all 8 families.

---

## 5. How to Run the Tests

```bash
# Run the complete hardened E2E notebook test suite (25 tests):
.venv/bin/pytest tests/test_notebook_e2e.py -v

# Run with standard Python unittest:
.venv/bin/python -m unittest tests/test_notebook_e2e.py -v

# Run all repository tests:
.venv/bin/pytest
```

---

## 6. Escalation Protocol

When tests fail due to implementation code in `notebooks/synccraft_end_to_end.ipynb`:
1. The Test Writer documents the exact failing test, failure reason, and underlying requirement in `handoff.md` and `DISPATCH.md`.
2. The Test Writer does NOT modify implementation code (`notebooks/synccraft_end_to_end.ipynb`), adhering strictly to file ownership.
3. The finding is escalated to `worker_impl_2` to rework the notebook implementation until all 25 tests pass cleanly.
