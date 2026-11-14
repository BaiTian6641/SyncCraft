# Empirical Challenger 2 Handoff Report

**Date**: 2026-11-14T04:15:00Z  
**Verdict**: **APPROVE**  
**Target Artifact**: `notebooks/synccraft_end_to_end.ipynb`  
**Test Suite**: `tests/test_adversarial_challenger.py`  
**Investigator**: Challenger 2 (`challenger_2`)  

---

## 1. Observation

Direct observations from automated tool runs and code inspection:

1. **Manifest Schema & Invariants (`notebooks/synccraft_end_to_end.ipynb` Cell 8)**:
   - Output `datasets/dataset_manifest.json` contains schema `"synccraft.dataset-manifest/0.1"`.
   - Every generated row contains exactly 13 fields:
     `asset_id`, `family_id`, `body_plan_id_if_dynamic`, `generator_revision`, `generator_parameters`, `canonical_dsl`, `canonical_dsl_sha256`, `prompt_and_provenance`, `source_license_and_review_record`, `split`, `audit_version_and_results`, `target_profile_and_version`, `seed`.
   - Top-level `dataset_manifest_sha256` is exactly 64 hex characters (reproduced deterministically via `hashlib.sha256` over sorted JSON rows).
   - `canonical_dsl_sha256` for every row matches `hashlib.sha256(canonical_dsl.encode("utf-8")).hexdigest()`.

2. **Split & Unseen Family Isolation (`configs/data_policy.yaml` & Cell 6, 8)**:
   - Ground truth unseen families: `mechanical_crossbow`, `hand_lantern`, `tripod_telescope`, `folding_windmill`, `anvil_forge`, `floating_ring_staff`.
   - In `datasets/dataset_manifest.json`, `train.jsonl`, and `val.jsonl`: 0 rows contain any of the 6 unseen static families.
   - All unseen family candidates are assigned exclusively to `split == "test_unseen"`.
   - `train_families.isdisjoint(unseen_families)` evaluated to `True`.

3. **Gate G2 Decision Arithmetic & Prerequisites (`src/synccraft/evaluation/gate_g2.py` & Cell 18)**:
   - `REQUIRED_CHECKS` tuple enforces all 8 mandatory prerequisites:
     `full_static_audit_implemented`, `heldout_split_frozen_before_training`, `all_samples_have_at_least_16_parts`, `repair_pgd_policy_preregistered`, `matched_raw_budgets_enforced`, `at_most_three_reaudits_per_sample`, `failures_and_timeouts_counted`, `baseline_implementations_verified`.
   - Hash validation strictly enforces `^[a-fA-F0-9]{64}$` for `dataset_manifest_sha256` and `evaluation_manifest_sha256`.
   - Threshold formula `passed * 10 < total * 9` implements integer arithmetic for $p/t < 0.90$.
   - At $p/t = 0.90$ (e.g. 45/50, 90/100), `passed * 10 < total * 9` is `False` -> decision is `skip_learned_refiner` (`learned_refiner_enabled = False`).
   - At $p/t < 0.90$ (e.g. 44/50, 89/100), `passed * 10 < total * 9` is `True` -> decision is `eligible_for_learned_refiner` (`learned_refiner_enabled = True`).
   - Notebook execution on 6 held-out unseen families: $6/6$ passed first audit ($100\%$), gate decision logged as `skip_learned_refiner`, saved to `runs/gate_g2_report.json`.

4. **CostLedger Accounting (`src/synccraft/evaluation/budget.py` & Cell 16, 17)**:
   - `CostLedger` is a frozen dataclass with 7 discrete operation counters.
   - Direct field mutation raises `dataclasses.FrozenInstanceError`.
   - `record(counter, amount)` with `amount < 0` strictly raises `ValueError("counts must be nonnegative integers; rollback cannot refund cost")`.
   - Non-integer amounts (`float`, `str`) and unknown counter dimensions raise `ValueError`.
   - In sandboxed execution: `accepted_tokens > 0`, `compiler_calls == 6`, `initial_audit_calls == 6`, `discarded_tokens == 0`, `repair_tokens == 0`. Counters are monotonic and non-negative throughout execution.

5. **Atomic Checkpointing (`notebooks/synccraft_end_to_end.ipynb` Cell 12)**:
   - Staging directory created using `tempfile.mkdtemp(prefix=".tmp_ckpt_", dir=CHECKPOINTS_DIR)` inside the checkpoints folder (guaranteeing same-filesystem atomic `rename`).
   - In fault-injection stress testing: simulated exceptions (`IOError`, `MemoryError`, `RuntimeError`) during file writes trigger the `finally` block, removing `.tmp_ckpt_*` and leaving zero partial directories at `checkpoint-{step}`.
   - Successful checkpoint contains `adapter_config.json`, `adapter_model.bin`, `adapter_model.safetensors`, `manifest.json`, `optimizer.pt`, `tokenizer_config.json`, `special_tokens_map.json`, and `dataset_manifest.json`.
   - All `file_hashes` recorded in `manifest.json` match disk files bit-for-bit.

6. **Automated Test Execution Results**:
   - Command: `.venv/bin/pytest tests/test_adversarial_challenger.py -v`
     Result: `29 passed in 2.10s` (100% pass rate).
   - Command: `.venv/bin/pytest`
     Result: `125 passed in 3.20s` (100% pass rate across entire repository).

---

## 2. Logic Chain

1. **Manifest Integrity**:
   - Observation 1 demonstrates that all generated records conform to the schema defined in `configs/data_policy.yaml` with all 13 required fields.
   - Cryptographic recomputation proves that neither `canonical_dsl` nor manifest rows can be tampered with without invalidating the SHA-256 hashes.
   - Therefore, dataset generation meets the architectural requirements of `AGENTS.md` and `PROJECT.md`.

2. **Zero Family Leakage**:
   - Observation 2 demonstrates that the unseen static families defined in `configs/data_policy.yaml` are strictly partitioned into `test_unseen` and completely absent from `train` and `val`.
   - Mathematical set intersection confirms `train_families.isdisjoint(unseen_families) == True`.
   - Therefore, training cannot inadvertently learn from held-out evaluation tasks.

3. **Gate G2 Reproducibility and Robustness**:
   - Observation 3 proves that `decide_gate` rejects missing or falsified prerequisite flags, invalid hash lengths/characters, negative counts, non-integers, and out-of-range counts.
   - The threshold boundary $p \times 10 < t \times 9$ replaces floating point operations with integer arithmetic. Comparison against Python's exact `fractions.Fraction` oracle across extensive parameter sweeps confirmed 100% parity.
   - At $\ge 90\%$, `learned_refiner_enabled` is disabled as mandated by `AGENTS.md`.

4. **Monotonic Budget Accounting**:
   - Observation 4 confirms that `CostLedger` cannot be decremented, refunded, or mutated via arbitrary types.
   - The recorded counters reflect genuine operation counts from the probing and compiler loops.

5. **Crash-Safe Checkpoint Staging**:
   - Observation 5 confirms that checkpoints stage completely in `.tmp_ckpt_*` before atomic directory renaming.
   - Simulated crash scenarios confirm that aborted writes leave no corrupt or half-written checkpoints, and all manifests store bit-accurate SHA-256 digests.

---

## 3. Caveats

- Multi-GPU distributed training (DDP / FSDP) and actual hardware quantization with BitsAndBytes were evaluated in CPU smoke test mode; execution on physical NVIDIA GPUs in Colab G4 depends on runtime hardware availability.
- The 6 held-out unseen static families in the smoke test harness represent a procedural pilot test set, pending the full 1,200 synthetic sample collection preregistered in Phase 1.

---

## 4. Conclusion

**Verdict: APPROVE**

The dataset manifest generation, 64-hex SHA-256 hashing, family isolation, Gate G2 prerequisite validation, 90% threshold arithmetic, CostLedger monotonicity, and atomic checkpoint staging in `notebooks/synccraft_end_to_end.ipynb` are completely verified, mathematically sound, and empirically resilient against adversarial corruption and failure conditions.

---

## 5. Verification Method

To independently reproduce the empirical adversarial verification:

```bash
# 1. Run the dedicated Challenger 2 adversarial stress suite (29 tests):
.venv/bin/pytest tests/test_adversarial_challenger.py -v

# 2. Run with standard Python unittest runner:
.venv/bin/python -m unittest tests/test_adversarial_challenger.py -v

# 3. Run the complete repository test suite (125 tests):
.venv/bin/pytest
```

**Files to Inspect**:
- `tests/test_adversarial_challenger.py`
- `notebooks/synccraft_end_to_end.ipynb`
- `src/synccraft/evaluation/gate_g2.py`
- `src/synccraft/evaluation/budget.py`
- `configs/data_policy.yaml`

**Invalidation Conditions**:
- Any omission or type corruption among the 13 manifest fields.
- Any non-64-hex SHA-256 hash or hash mismatch against canonical DSL.
- Any presence of unseen static families in `train` or `val` splits.
- Any deviation in Gate G2 decision arithmetic ($p \times 10 < t \times 9$).
- Any ability to decrement or refund `CostLedger` counters.
- Any orphaned staging directory or corrupt checkpoint surviving a simulated crash.
