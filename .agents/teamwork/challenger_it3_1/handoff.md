# Handoff Report — Challenger Iteration 3 (Empirical Adversarial Verification)

**Author**: `challenger_it3_1` (Roles: critic, specialist)  
**Parent**: `352f5bd8-d48a-47d3-8e21-8d9aea7412d5`  
**Date**: 2026-11-14T05:40:00Z  
**Verdict**: **APPROVE**  
**Overall Risk Assessment**: **LOW**

---

## Challenge Summary

The empirical adversarial challenger has systematically investigated and stress-tested `notebooks/synccraft_end_to_end.ipynb` and its dense assembly templates following the remediation by `worker_impl_3`.

All forensic integrity defects reported in Iteration 2 have been completely eliminated:
1. **Dense Assembly Templates**: All 6 `UNSEEN_TEMPLATES` in Cell 6 now contain between 17 and 18 parts (all $\ge 16$), with strictly acyclic SE(3) kinematic trees, valid orthonormal bone transforms, positive volumes, and zero SAT collisions across tolerances from $0.0$ to $10^{-5}$.
2. **Probing Loop Integrity (Cell 16)**: Ground-truth fallback (`task["dsl_target"]`) has been completely removed. Raw model completions are directly assigned (`evaluated_dsl = raw_completion_text`) and recorded.
3. **Honest Auditing (Cell 17)**: Invalid model completions trigger parse/validation exceptions that are recorded honestly (`syntax_valid: False, sat_collision_passed: False, boxes: 0`), never masked or replaced.
4. **Dynamic Box Count Verification (Cell 18)**: Prerequisite `all_samples_have_16_parts` dynamically inspects empirical box counts (`all(r.get("boxes", 0) >= 16 for r in probing_audit_results)`). It strictly rejects sub-threshold completions (15 boxes), empty sets, mixed batches, and syntax failures.
5. **Full Test Suite & Pipeline Pass**: Headless smoke execution passes cleanly across 130 tests (`pytest tests/test_notebook_e2e.py`, `pytest tests/test_adversarial_challenger.py`, `pytest -v`, `scripts/check.py`, and `scripts/demo.py`).

---

## 1. Observation

### 1.1 Dense Assembly Templates in Cell 6 (`UNSEEN_TEMPLATES`)
We directly parsed, validated, compiled SE(3) kinematics, and audited all 6 unseen held-out templates:
- `mechanical_crossbow`: 18 boxes, 3 bones, 17 declared contacts, 0 SAT collisions (PASSED at tol $0.0, 10^{-9}, 10^{-7}, 10^{-5}$).
- `hand_lantern`: 18 boxes, 2 bones, 17 declared contacts, 0 SAT collisions (PASSED at tol $0.0, 10^{-9}, 10^{-7}, 10^{-5}$).
- `tripod_telescope`: 18 boxes, 5 bones, 17 declared contacts, 0 SAT collisions (PASSED at tol $0.0, 10^{-9}, 10^{-7}, 10^{-5}$).
- `folding_windmill`: 17 boxes, 2 bones, 16 declared contacts, 0 SAT collisions (PASSED at tol $0.0, 10^{-9}, 10^{-7}, 10^{-5}$).
- `anvil_forge`: 17 boxes, 2 bones, 16 declared contacts, 0 SAT collisions (PASSED at tol $0.0, 10^{-9}, 10^{-7}, 10^{-5}$).
- `floating_ring_staff`: 18 boxes, 2 bones, 17 declared contacts, 0 SAT collisions (PASSED at tol $0.0, 10^{-9}, 10^{-7}, 10^{-5}$).

All bone rotation matrices $R$ satisfy orthonormality $\|R^T R - I\| < 10^{-5}$ and $\det(R) = 1.0 \pm 10^{-5}$. All bone parent chains are acyclic. All box dimensions are strictly positive ($>0$), and all declared contacts have penetration depth $\le 10^{-7}$ and `collision: False`.

### 1.2 Probing Loop in Cell 16
Direct inspection of `notebooks/synccraft_end_to_end.ipynb` Cell 16:
- Lines 38–39:
  ```python
  # Directly evaluate raw model completion without ground-truth fallback
  evaluated_dsl = raw_completion_text
  ```
- Lines 41–46:
  ```python
  probed_completions.append({
      "family_id": task["family_id"],
      "archetype": task["archetype"],
      "prompt": task_prompt,
      "completion": evaluated_dsl,
  })
  ```
- `task["dsl_target"]` appears only in the metadata definition of `PROBING_PROMPTS` and is never accessed during generation, evaluation, or appending.

### 1.3 Geometric SAT Auditing in Cell 17
Direct inspection of Cell 17:
- Lines 28–34:
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
- Any parse, validation, or compilation error truthfully yields `boxes: 0` and `sat_collision_passed: False`.

### 1.4 Dynamic Gate G2 Prerequisites in Cell 18
Direct inspection of Cell 18:
- Lines 8–11:
  ```python
  all_samples_have_16_parts = (
      len(probing_audit_results) > 0 and
      all(r.get("boxes", 0) >= 16 for r in probing_audit_results)
  )
  ```
- Feeds `all_samples_have_at_least_16_parts: all_samples_have_16_parts` directly to `decide_gate`.
- The proxy expression `passed_audits == len(probed_completions)` has been deleted.

### 1.5 Test Suite Execution Results
Verbatim terminal outputs:
1. `.venv/bin/pytest tests/test_notebook_e2e.py -v`:
   `25 passed in 1.77s` (Exit code 0).
2. `.venv/bin/pytest tests/test_adversarial_challenger.py -v`:
   `29 passed in 2.41s` (Exit code 0).
3. `.venv/bin/pytest -v`:
   `130 passed, 14 subtests passed in 3.88s` (Exit code 0).
4. `.venv/bin/python scripts/check.py`:
   `Ran 130 tests in 3.426s; OK` (Exit code 0).
5. `.venv/bin/python scripts/demo.py`:
   `Exited with code 0; 3 models verified`.

---

## 2. Logic Chain

1. **Adversarial Test Suite 1: Arbitrary Invalid Model Completions**:
   - Injected 6 invalid inputs: python gibberish, broken DSL syntax, empty string, whitespace only, unknown tokens, and semantic missing-bone references.
   - Result: Cell 17 caught 100% of invalid completions honestly, recording `syntax_valid: False, sat_collision_passed: False, boxes: 0`.
   - Result: Cell 18 evaluated `all_samples_have_16_parts == False`.
   - Result: `decide_gate` reported `decision: blocked` with `missing_or_invalid_evidence: ['all_samples_have_at_least_16_parts']`.

2. **Adversarial Test Suite 2: Authentic 18-part Valid Completions**:
   - Fed the 6 authentic templates from `UNSEEN_TEMPLATES` into `probed_completions`.
   - Result: Cell 17 recorded `passed_audits = 6/6 (100%)`, all samples with 17 or 18 boxes.
   - Result: Cell 18 evaluated `all_samples_have_16_parts == True`.
   - Result: `decide_gate` reported `decision: skip_learned_refiner`, `learned_refiner_enabled: False` (since 100% $\ge 90\%$, satisfying the policy).

3. **Adversarial Test Suite 3: Boundary & Sub-Threshold Part Counts**:
   - Boundary test 3a: A valid program with exactly 15 boxes evaluated to `all_samples_have_16_parts == False`.
   - Boundary test 3b: A valid program with exactly 16 boxes evaluated to `all_samples_have_16_parts == True`.
   - Boundary test 3c: An empty results list evaluated to `all_samples_have_16_parts == False` due to the `len(...) > 0` guard.
   - Boundary test 3d: A mixed list of five 18-box models and one 15-box model evaluated to `all_samples_have_16_parts == False`.
   - Boundary test 3e: A mixed list of five 18-box models and one syntax error (`boxes: 0`) evaluated to `all_samples_have_16_parts == False`.

4. **Adversarial Test Suite 4: Collision Injection & Gate G2 Learned Refiner Activation**:
   - Injected a dense 19-box program containing an overlapping penetrating box.
   - Result: `audit_scene` detected 5 collision pairs; `report["passed"] == False`.
   - Result: Cell 17 recorded `boxes: 19`, `sat_collision_passed: False`, `passed_audits: 0`.
   - Result: Cell 18 computed `all_samples_have_16_parts == True`.
   - Result: `decide_gate` reported `decision: eligible_for_learned_refiner` and `learned_refiner_enabled: True` because prerequisites were met and the first-audit pass rate ($0\%$) fell below the $90\%$ threshold.

Conclusion from Logic Chain:
The system exhibits zero facade behavior. It behaves strictly as a deterministic, mathematically grounded verification pipeline that adheres to all SyncCraft v3.0 specifications and AGENTS.md requirements.

---

## 3. Caveats

1. **CPU Smoke Generation Quality**: In CPU smoke mode, the causal language model receives only 5 training steps on a micro-batch. Consequently, its probed completions are untrained token sequences that naturally fail DSL syntax parsing (`boxes: 0`). This is expected and required behavior in headless smoke testing: the pipeline truthfully detects this failure and reports Gate G2 as `BLOCKED`. Full model convergence and high pass rates require multi-epoch GPU training.
2. **Scope of Adversarial Challenge**: All testing was performed locally in the macOS CPU virtual environment (`.venv/bin/python3.12`) using direct AST, SE(3), SAT, and exporter oracles.

---

## 4. Conclusion

Verdict: **APPROVE**.

The remediation implemented in `notebooks/synccraft_end_to_end.ipynb` satisfies all technical and integrity requirements:
- All 6 held-out families have verified dense assembly templates ($\ge 16$ parts, acyclic kinematics, 0 SAT collisions).
- Probing evaluation evaluates raw completions with zero ground-truth substitution.
- Audit recording and Gate G2 prerequisite checks are dynamic, strict, and boundary-verified.
- All 130 tests across unit, integration, and adversarial suites pass without errors or regressions.

---

## 5. Verification Method

To independently verify the empirical results documented in this report:

1. **Verify UNSEEN_TEMPLATES Part Counts and SAT 0 Collisions**:
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
       prog = parse_program(dsl)
       validate_program(prog)
       scene = compile_geometry(prog)
       rep = audit_scene(scene, collision_tolerance=1e-7)
       assert len(scene.boxes) >= 16, f"{fam} has < 16 boxes"
       assert rep["passed"], f"{fam} collision detected"
       print(f"Verified {fam}: {len(scene.boxes)} boxes, passed SAT={rep[\"passed\"]}")
   '
   ```

2. **Verify Cell 16 and Cell 18 Dynamic Logic**:
   ```bash
   .venv/bin/python -c '
   import json
   nb = json.load(open("notebooks/synccraft_end_to_end.ipynb"))
   c16 = "".join(nb["cells"][16]["source"])
   c18 = "".join(nb["cells"][18]["source"])
   assert "evaluated_dsl = raw_completion_text" in c16
   assert "all(r.get(\"boxes\", 0) >= 16 for r in probing_audit_results)" in c18
   assert "passed_audits == len(probed_completions)" not in c18
   print("Verified Cell 16 & 18 AST integrity.")
   '
   ```

3. **Run Full Pytest Test Suites**:
   ```bash
   .venv/bin/pytest tests/test_notebook_e2e.py -v
   .venv/bin/pytest tests/test_adversarial_challenger.py -v
   .venv/bin/pytest -v
   .venv/bin/python scripts/check.py
   .venv/bin/python scripts/demo.py
   ```
