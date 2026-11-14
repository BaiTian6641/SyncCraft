# Forensic Audit Report & Handoff — Iteration 3

**Work Product**: `/Users/tianrui6641/SyncCraft/notebooks/synccraft_end_to_end.ipynb`, `tests/test_notebook_e2e.py`, `tests/test_adversarial_challenger.py`  
**Auditor Archetype**: Forensic Auditor (`auditor_it3_1`)  
**Profile**: General Project (Integrity Forensics)  
**Integrity Mode**: Development Mode (per `ORIGINAL_REQUEST.md` § Integrity Mode)  
**Binary Verdict**: **CLEAN**  

---

## Executive Summary & Phase Results

| # | Check Name | Target Contract / Invariant | Status | Empirical Findings |
|---|---|---|---|---|
| 1 | **Safetensors Payload** | Ban on `b"LORA_SAFETENSORS_PAYLOAD"` & valid little-endian binary header | **PASS** | Valid uint64 header length (`<Q`, 222 bytes) + JSON header + offset buffers. Deserialized without error. No payload byte marker strings. |
| 2 | **QLoRA Training Pipeline** | Ban on `torch.randint` noise; authentic causal LM loss with response masking | **PASS** | Zero `torch.randint`; real causal LM next-token prediction on `canonical_dsl` tokens with prompt tokens masked to `-100`. |
| 3 | **Probing Evaluation** | Complete elimination of Cell 16 ground-truth fallback; direct evaluation of raw completions | **PASS** | `except Exception: evaluated_dsl = task["dsl_target"]` completely eliminated. `raw_completion_text` assigned directly to `evaluated_dsl`. 6/6 probed outputs honestly recorded with `syntax_valid: False, sat_collision_passed: False, boxes: 0`. |
| 4 | **Gate G2 Attestation** | Elimination of facade proxy expressions; dynamic evaluation of box counts ($\ge 16$) | **PASS** | `passed_audits == len(...)` proxy removed. Cell 18 dynamically evaluates `all(r.get("boxes", 0) >= 16 for r in probing_audit_results)`. Evaluates to `False` on invalid smoke outputs, producing honest Gate G2 decision: `decision: blocked`. |
| 5 | **Dense Assembly Integrity** | All 6 `UNSEEN_TEMPLATES` have $\ge 16$ boxes, valid AST, SE(3) compile, and 0 SAT collisions | **PASS** | All 6 held-out families (`mechanical_crossbow` [18], `hand_lantern` [18], `tripod_telescope` [18], `folding_windmill` [17], `anvil_forge` [17], `floating_ring_staff` [18]) compile SE(3) kinematic trees and pass 15-axis SAT with 0 collisions. |
| 6 | **Secrets & Security** | Token redaction & zero credential leaks | **PASS** | Zero credential leaks found across repository and notebook (regex scan returned 0); Colab Secrets integration with fallback and token masking (`redact_token()`) verified. |
| 7 | **Kinematics & SAT Audit** | Shared SE(3) kinematics, 15-axis SAT OBB, degenerate cross-axis masking | **PASS** | Direct 15-axis SAT OBB engine tests all 15 axes, masks degenerate cross-axes ($\le 10^{-10}$), respects contractual contacts without blanket parent/sibling exemptions. |
| 8 | **AGENTS.md Compliance** | Zero Mojang assets, zero `eval()`/`exec()` on untrusted Block-DSL strings | **PASS** | Zero Mojang assets; zero `eval()` or `exec()` on untrusted code (all Block-DSL parsed into typed AST via `parse_program`). |
| 9 | **Execution Validation** | Full automated execution of test suites and sanity scripts | **PASS** | `pytest tests/test_notebook_e2e.py`: 25/25 passed; `pytest tests/test_adversarial_challenger.py`: 29/29 passed; `pytest -v`: 130/130 passed; `scripts/check.py`: OK; `scripts/demo.py`: OK. |

---

## 1. Observation

### 1.1 Direct Tool Commands and Verbatim Results

#### Command 1: E2E Hardened Notebook Test Suite Execution
```bash
.venv/bin/pytest tests/test_notebook_e2e.py -v
```
**Output**:
```
============================= test session starts ==============================
platform darwin -- Python 3.12.16, pytest-9.2.0, pluggy-1.7.0 -- /Users/tianrui6641/SyncCraft/.venv/bin/python3.12
cachedir: .pytest_cache
rootdir: /Users/tianrui6641/SyncCraft
configfile: pyproject.toml
plugins: platformdirs-4.14.5
collecting ... collected 25 items

tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_01_notebook_file_exists PASSED [  4%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_02_notebook_valid_json_and_nbformat_v4 PASSED [  8%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_03_markdown_sections_delineated PASSED [ 12%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_04_colab_secrets_and_env_fallback PASSED [ 16%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_05_dataset_gathering_and_sat_filtering PASSED [ 20%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_05b_seen_static_families_complete_coverage PASSED [ 24%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_06_dataset_manifest_schema_and_hashing PASSED [ 28%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_07_qlora_4b_7b_training_configurations PASSED [ 32%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_08_checkpoint_structure_contract PASSED [ 36%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_08b_safetensors_binary_header_contract PASSED [ 40%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_09_hf_publishing_idempotency_and_model_card PASSED [ 44%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_10_probing_unseen_families_and_gate_g2 PASSED [ 48%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_10b_model_probing_queries_model_generation PASSED [ 52%]
tests/test_notebook_e2e.py::TestNotebookTier1FeatureCoverage::test_10c_gate_g2_prerequisites_dynamic_evaluation PASSED [ 56%]
tests/test_notebook_e2e.py::TestNotebookTier2BoundaryAndCornerCases::test_01_missing_hf_token_graceful_fallback PASSED [ 60%]
tests/test_notebook_e2e.py::TestNotebookTier2BoundaryAndCornerCases::test_02_missing_cuda_cpu_smoke_mode PASSED [ 64%]
tests/test_notebook_e2e.py::TestNotebookTier2BoundaryAndCornerCases::test_03_empty_and_malformed_candidate_dsl_rejection PASSED [ 68%]
tests/test_notebook_e2e.py::TestNotebookTier2BoundaryAndCornerCases::test_04_sat_flush_contact_vs_penetration_boundary PASSED [ 72%]
tests/test_notebook_e2e.py::TestNotebookTier2BoundaryAndCornerCases::test_05_degenerate_cross_axis_masking PASSED [ 76%]
tests/test_notebook_e2e.py::TestNotebookTier2BoundaryAndCornerCases::test_06_token_redaction_in_notebook_and_logs PASSED [ 80%]
tests/test_notebook_e2e.py::TestNotebookTier2BoundaryAndCornerCases::test_07_safetensors_corrupt_header_rejection PASSED [ 84%]
tests/test_notebook_e2e.py::TestNotebookTier3CrossFeatureCombinations::test_01_manifest_hash_flows_to_checkpoint_and_gate_g2 PASSED [ 88%]
tests/test_notebook_e2e.py::TestNotebookTier3CrossFeatureCombinations::test_02_secrets_to_hf_publishing_to_model_card_flow PASSED [ 92%]
tests/test_notebook_e2e.py::TestNotebookTier3CrossFeatureCombinations::test_03_ast_to_geometry_to_sat_to_exports_flow PASSED [ 96%]
tests/test_notebook_e2e.py::TestNotebookTier4RealWorldScenarios::test_01_headless_notebook_smoke_execution PASSED [100%]

============================== 25 passed in 2.43s ==============================
```

#### Command 2: Adversarial Challenger Test Suite Execution
```bash
.venv/bin/pytest tests/test_adversarial_challenger.py -v
```
**Output**: `29 passed in 2.51s` (Exit code 0).

#### Command 3: Full Repository Test Suite
```bash
.venv/bin/pytest -v
```
**Output**: `130 passed, 14 subtests passed in 3.70s` (Exit code 0).

#### Command 4: Project Sanity Check Script
```bash
.venv/bin/python scripts/check.py
```
**Output**: `Ran 130 tests in 3.519s; OK` (Exit code 0).

#### Command 5: Demo Script Execution
```bash
.venv/bin/python scripts/demo.py
```
**Output**: Exited 0 with verified outputs: `mirrored_wings` (5 boxes, 3 bones, audit_passed: true), `simple_blade` (4 boxes, 3 bones, audit_passed: true), `stacked_block` (3 boxes, 1 bone, audit_passed: true).

---

### 1.2 Direct Source Code and Artifact Observations

#### Observation 1.2.1: Safetensors Binary Deserialization (`checkpoints/checkpoint-5/adapter_model.safetensors`)
```python
import struct, json
with open("checkpoints/checkpoint-5/adapter_model.safetensors", "rb") as f:
    raw_len = f.read(8)
    header_len = struct.unpack("<Q", raw_len)[0]
    header_bytes = f.read(header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    rest = f.read()
```
**Empirical Result**:
- `header_len`: 222 bytes
- `header["__metadata__"]`: `{'format': 'pt', 'synccraft_version': '3.0'}`
- Tensors:
  - `lora_adapter.lora_A`: shape `[32, 128]`, dtype `F32`, data_offsets `[0, 16384]`
  - `lora_adapter.lora_B`: shape `[128, 32]`, dtype `F32`, data_offsets `[16384, 32768]`
- Total tensor data bytes: 32,768 (exact match for 8,192 float32 entries).
- Payload check: `b"LORA_SAFETENSORS_PAYLOAD"` was NOT found in the binary or header.

#### Observation 1.2.2: QLoRA Training Loop (`notebooks/synccraft_end_to_end.ipynb`, Cell 11)
- Prompt and response texts are derived from verified dataset records (`canonical_dsl`).
- Token IDs are constructed via `prompt_ids + response_ids`.
- Response masking is applied: `labels[:, :prompt_len] = -100`.
- Loss function: `criterion = nn.CrossEntropyLoss(ignore_index=-100)`.
- Loss computation:
  ```python
  shift_logits = logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous()
  loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
  ```
- Backpropagation and optimizer: `loss.backward()`, `clip_grad_norm_()`, `optimizer.step()`.
- Operation cost tracking: `cost_ledger.record("accepted_tokens", response_token_count)`.
- Zero occurrences of `torch.randint` across all cells.

#### Observation 1.2.3: Elimination of Probing Fallback Bypass (`notebooks/synccraft_end_to_end.ipynb`, Cell 16)
In Cell 16 (Cell 11 of the notebook), the legacy `try...except Exception: evaluated_dsl = task["dsl_target"]` has been completely deleted:
```python
    # Directly evaluate raw model completion without ground-truth fallback
    evaluated_dsl = raw_completion_text

    probed_completions.append({
        "family_id": task["family_id"],
        "archetype": task["archetype"],
        "prompt": task_prompt,
        "completion": evaluated_dsl,
    })
```
Empirical execution check confirms:
```
Family: mechanical_crossbow | Matches target: False | Length: 449
Family: hand_lantern        | Matches target: False | Length: 449
Family: tripod_telescope    | Matches target: False | Length: 449
Family: folding_windmill    | Matches target: False | Length: 449
Family: anvil_forge         | Matches target: False | Length: 449
Family: floating_ring_staff | Matches target: False | Length: 449
```
For 6/6 probed samples, the evaluated completion is the actual model-generated token sequence, not the hand-crafted ground truth.

#### Observation 1.2.4: Honest Probing Audit and Gate G2 Evaluation (`notebooks/synccraft_end_to_end.ipynb`, Cells 17 & 18)
In Cell 17, parse failures are caught and recorded honestly:
```python
    except Exception as exc:
        probing_audit_results.append({
            "family_id": item["family_id"],
            "syntax_valid": False,
            "sat_collision_passed": False,
            "boxes": 0,
            "error": str(exc),
        })
```
Output:
```
Probing Audit Summary:
  [FAIL] mechanical_crossbow      (Boxes: 0, Collisions: 0)
  [FAIL] hand_lantern             (Boxes: 0, Collisions: 0)
  [FAIL] tripod_telescope         (Boxes: 0, Collisions: 0)
  [FAIL] folding_windmill         (Boxes: 0, Collisions: 0)
  [FAIL] anvil_forge              (Boxes: 0, Collisions: 0)
  [FAIL] floating_ring_staff      (Boxes: 0, Collisions: 0)

Probing Audit Pass Rate: 0/6 (0.0%)
```
In Cell 18:
```python
# Dynamic evaluation of prerequisite checks from empirical probing audit results
all_samples_have_16_parts = (
    len(probing_audit_results) > 0 and
    all(r.get("boxes", 0) >= 16 for r in probing_audit_results)
)
```
Empirical gate decision output:
```
Total Held-Out Samples:      6
Passed First SAT Audit:      0
Pass Rate:                   0.00%
Gate G2 Decision:            BLOCKED
Learned Refiner Enabled:     False
Interpretation:              Engineering branch decision
Blocked Prerequisites:       ['all_samples_have_at_least_16_parts']
```
The facade proxy boolean `passed_audits == len(probed_completions)` has been completely removed. `all_samples_have_16_parts` dynamically evaluates to `False`, and `decide_gate` honestly blocks Gate G2.

#### Observation 1.2.5: Dense Assembly Invariant in `UNSEEN_TEMPLATES` (Cell 6)
Direct empirical evaluation of all 6 templates in `UNSEEN_TEMPLATES`:
```python
for fam, dsl in unseen.items():
    prog = parse_program(dsl)
    validate_program(prog)
    scene = compile_geometry(prog)
    audit = audit_scene(scene, collision_tolerance=1e-7)
    ...
```
Output:
```
Family: mechanical_crossbow      | Boxes: 18 | Bones:  3 | Contacts: 17 | Collisions: 0 | SAT Passed: True
Family: hand_lantern             | Boxes: 18 | Bones:  2 | Contacts: 17 | Collisions: 0 | SAT Passed: True
Family: tripod_telescope         | Boxes: 18 | Bones:  5 | Contacts: 17 | Collisions: 0 | SAT Passed: True
Family: folding_windmill         | Boxes: 17 | Bones:  2 | Contacts: 16 | Collisions: 0 | SAT Passed: True
Family: anvil_forge              | Boxes: 17 | Bones:  2 | Contacts: 16 | Collisions: 0 | SAT Passed: True
Family: floating_ring_staff      | Boxes: 18 | Bones:  2 | Contacts: 17 | Collisions: 0 | SAT Passed: True
```
All 6 held-out families have between 17 and 18 boxes ($\ge 16$), parse valid AST, compile SE(3) kinematic trees, and pass 15-axis SAT collision audits with zero collisions.

---

## 2. Logic Chain

1. **Iteration 2 Defect Remediation**:
   - In Iteration 2, the audit failed because Cell 16 caught parse errors and substituted ground-truth templates (`task["dsl_target"]`), and Cell 18 used a facade proxy (`passed_audits == len(...)`) to attest `True` for `all_samples_have_at_least_16_parts`.
   - Inspection of `notebooks/synccraft_end_to_end.ipynb` confirms both patterns have been completely eliminated.
   - Probed outputs are directly assigned from `raw_completion_text`. Untrained model outputs honestly fail syntax parsing, recording 0 boxes.
   - Gate G2 dynamically checks `all(r.get("boxes", 0) >= 16 for r in probing_audit_results)`, which evaluates to `False`, blocking Gate G2 cleanly per `AGENTS.md` rules.

2. **Test Suite Harmonization**:
   - `test_adversarial_challenger.py:822-845` was decoupled from hardcoded `skip_learned_refiner` and 100% pass rate expectations, comparing instead against `decide_gate(evidence)`.
   - `test_notebook_e2e.py` `test_10b` and `test_10c` were hardened with AST inspections that strictly verify that no exception handlers substitute ground truth and that box count checking is authentic.

3. **Phase 1 Mode-Agnostic Assessment**:
   - Hardcoded test outputs: None found.
   - Facade implementations: None found.
   - Fabricated verification outputs: None found.
   - Third-party delegation of core logic: None.

4. **Phase 2 Development Mode Flagging**:
   - Under Development Mode (specified in `ORIGINAL_REQUEST.md`), the work products satisfy all invariants.
   - All 9 forensic integrity checks pass with empirical evidence.

---

## 3. Caveats

1. **Hardware Execution Scope**: Probing in the local headless environment was executed in CPU smoke mode with 5 training steps on a miniature causal LM. The resulting completion failures (`boxes: 0`) and Gate G2 decision (`blocked`) are the expected, authentic behavior of an untrained smoke model. Convergence to valid 16-box DSL requires GPU training on large datasets.
2. **Bounds Handling in Exporter**: In Cell 7, `folding_windmill_heldout_s17` is quarantined into `rejected_records` during candidate dataset filtering because its canvas spar exceeds the `[-16, 32]` java-modern export bounds. This is correct behavior for the filtering pipeline, and `test_05` correctly verifies that invalid candidates are rejected while valid candidates are retained.

---

## 4. Conclusion

All integrity violations identified in Iteration 2 have been thoroughly and authentically remediated. There are no hardcoded test results, facade proxy booleans, ground-truth substitutions, or fabricated outputs. The notebook executes cleanly, all 130 tests pass across the entire suite, and all architectural and numerical invariants of `AGENTS.md` and `PROJECT.md` are upheld.

Final Binary Verdict: **CLEAN**

---

## 5. Verification Method

To independently verify this clean verdict:

1. **Verify Probing Completion Directness (No Ground-Truth Fallback)**:
   ```bash
   .venv/bin/python -c '
   import json
   nb = json.load(open("notebooks/synccraft_end_to_end.ipynb"))
   c16 = "".join(nb["cells"][16]["source"])
   assert "dsl_target" not in c16
   assert "evaluated_dsl = raw_completion_text" in c16
   print("Cell 16 verified: direct evaluation.")
   '
   ```

2. **Verify Dynamic Gate G2 Box Check**:
   ```bash
   .venv/bin/python -c '
   import json
   nb = json.load(open("notebooks/synccraft_end_to_end.ipynb"))
   c18 = "".join(nb["cells"][18]["source"])
   assert "all(r.get(\"boxes\", 0) >= 16 for r in probing_audit_results)" in c18
   assert "passed_audits == len(probed_completions)" not in c18
   print("Cell 18 verified: dynamic box count check.")
   '
   ```

3. **Verify Dense Assembly Templates in UNSEEN_TEMPLATES**:
   ```bash
   .venv/bin/python -c '
   import json
   from synccraft.dsl.parser import parse_program
   from synccraft.dsl.ast_nodes import validate_program
   from synccraft.kinematics.se3_chain import compile_geometry
   from synccraft.audit.sync_audit import audit_scene

   nb = json.load(open("notebooks/synccraft_end_to_end.ipynb"))
   scope = {}
   exec("".join(nb["cells"][6]["source"]), scope)
   for fam, dsl in scope["UNSEEN_TEMPLATES"].items():
       scene = compile_geometry(parse_program(dsl))
       rep = audit_scene(scene)
       assert len(scene.boxes) >= 16 and rep["passed"]
       print(f"{fam}: {len(scene.boxes)} boxes, SAT={rep[\"passed\"]}")
   '
   ```

4. **Run Full Test Suites**:
   ```bash
   .venv/bin/pytest tests/test_notebook_e2e.py -v
   .venv/bin/pytest tests/test_adversarial_challenger.py -v
   .venv/bin/pytest -v
   .venv/bin/python scripts/check.py
   .venv/bin/python scripts/demo.py
   ```
