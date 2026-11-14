# Challenger 1 Task Assignment: Iteration 3 Adversarial Verification

## Objective
Empirically challenge `notebooks/synccraft_end_to_end.ipynb` and the dense assembly templates with stress testing, boundary analysis, and oracle comparisons.

## Inputs & Paths
- User Request: `/Users/tianrui6641/SyncCraft/.agents/teamwork/ORIGINAL_REQUEST.md`
- Project Document: `/Users/tianrui6641/SyncCraft/.agents/teamwork/PROJECT.md`
- Worker Handoff: `/Users/tianrui6641/SyncCraft/.agents/teamwork/worker_impl_3/handoff.md`
- State Folder: `/Users/tianrui6641/SyncCraft/.agents/teamwork/challenger_it3_1/`

## Tasks
1. Write and run stress harnesses on `UNSEEN_TEMPLATES` in Cell 6:
   - Verify every template has >= 16 parts, valid kinematics, zero SAT collisions.
2. Adversarially test Cell 16 and Cell 18:
   - Feed arbitrary invalid completions, verify Cell 17 records syntax error honestly, verify Cell 18 evaluates `all_samples_have_16_parts == False`.
   - Feed authentic 18-part valid completions, verify Cell 18 evaluates `all_samples_have_16_parts == True`.
3. Run test suites: `pytest tests/test_notebook_e2e.py -v`, `pytest tests/test_adversarial_challenger.py -v`, `pytest -v`.
4. Deliver verdict: **APPROVE** or **REQUEST_CHANGES** in `handoff.md`.

## 2026-11-14T05:32:32Z
You are challenger_it3_1. Read your task assignment at /Users/tianrui6641/SyncCraft/.agents/teamwork/challenger_it3_1/DISPATCH.md.
Also read:
- /Users/tianrui6641/SyncCraft/.agents/teamwork/ORIGINAL_REQUEST.md
- /Users/tianrui6641/SyncCraft/.agents/teamwork/PROJECT.md
- /Users/tianrui6641/SyncCraft/.agents/teamwork/worker_impl_3/handoff.md

State folder: /Users/tianrui6641/SyncCraft/.agents/teamwork/challenger_it3_1/
Empirically stress-test notebooks/synccraft_end_to_end.ipynb and the dense assembly templates:
1. Validate all 6 UNSEEN_TEMPLATES: >=16 boxes, valid kinematics, 15-axis SAT 0 collisions.
2. Adversarially test Cell 16 and Cell 18 with invalid vs valid completions.
3. Run test suites (.venv/bin/pytest tests/test_notebook_e2e.py -v, etc.).
Deliver your challenge report and verdict (APPROVE or REQUEST_CHANGES) in handoff.md, and notify via send_message.
