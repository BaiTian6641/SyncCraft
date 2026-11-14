# BRIEFING — 2026-11-14T03:49:00Z

## Mission
Investigate SyncCraft codebase for target model training (4B/7B QLoRA), checkpointing, HF hub integration, and probing/evaluation (held-out tasks, geometric & SAT collision audits) to inform `notebooks/synccraft_end_to_end.ipynb` design for Colab G4 instance.

## 🔒 My Identity
- Archetype: explorer
- Roles: survey, investigation, synthesis
- Working directory: /Users/tianrui6641/SyncCraft
- State folder: /Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_3
- Original parent: 70b41f22-995e-433f-8ebe-f08406d3f62e
- Milestone: Notebook training & probing architecture survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement source code
- Write only to state folder `/Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_3/`
- Target Colab G4 instance (16GB VRAM constraint, gradient checkpointing, batch sizes)
- Support headless verification / CPU fallback mode
- HF integration via Colab Secrets (`HF_TOKEN`)
- Adhere to `AGENTS.md` rules: typed program approach, SAT collision audit, explicit contract verification, discrete integer sizes, reproducible splits

## Current Parent
- Conversation ID: 70b41f22-995e-433f-8ebe-f08406d3f62e
- Updated: not yet

## Investigation State
- **Explored paths**:
  - `configs/training_static.yaml`, `configs/model_candidates.yaml`, `configs/evaluation.yaml`, `configs/data_policy.yaml`, `configs/gate_g2_evidence.json`, `configs/compute_budget.json`
  - `src/synccraft/modeling/colab_trainer.py`, `learned_refiner.py`, `pgd_projector.py`
  - `src/synccraft/evaluation/gate_g2.py`, `budget.py`
  - `src/synccraft/audit/sync_audit.py`, `src/synccraft/kinematics/sat_potential.py`, `se3_chain.py`
  - `src/synccraft/cli.py`, `scripts/check.py`, `scripts/demo.py`, `scripts/budget_estimate.py`, `scripts/bootstrap.sh`
  - `source/synccraft-v3-plan-extracted.txt` (sections 4.1, 4.3, 5.1, 5.2, 5.3, 6.2, 6.3)
- **Key findings**:
  1. Training config (`configs/training_static.yaml`) specifies 4B BF16 / 7B NF4 QLoRA (r=32, alpha=64, dropout 0.05, microbatch=2, grad_accum=64, eff_batch=128, seq_len=2048, response_tokens_only=true, gradient_checkpointing=true).
  2. Model candidates in `configs/model_candidates.yaml` focus on Apache-2.0 models: `Qwen2.5-Coder-7B-Instruct` (NF4 QLoRA on 16GB VRAM), `Qwen3-4B` / `Spark-X2.6-4B`.
  3. Codebase trainer `src/synccraft/modeling/colab_trainer.py` currently raises `NotImplementedError` for P1-09. Notebook will implement the self-contained execution loop.
  4. Checkpointing requires atomic manifest, 25-min interval, and artifacts: LoRA adapters, tokenizer, optimizer/scheduler state, and manifest with hashes.
  5. HF Hub integration connects via Colab Secrets `HF_TOKEN` (via `google.colab.userdata` with `os.environ` fallback).
  6. Probing & evaluation leverages `parse_program`, `validate_program`, `compile_geometry`, `audit_scene` (15-axis SAT OBB), and Gate G2 decision logic (`decide_gate`).
  7. Colab G4 instance (16GB T4/G4 VRAM): Requires NF4 4-bit quantization, gradient checkpointing, micro-batch size 1-2, PagedAdamW8bit/AdamW, and dual-mode execution (GPU full vs CPU headless fallback).
- **Unexplored areas**: None.

## Key Decisions Made
- Formulated concrete notebook architecture with 5 distinct sections and full headless fallback.

## Artifact Index
- `.agents/teamwork/explorer_survey_3/DISPATCH.md` — Dispatch prompt and instructions
- `.agents/teamwork/explorer_survey_3/BRIEFING.md` — Working memory and identity
- `.agents/teamwork/explorer_survey_3/progress.md` — Liveness heartbeat and step tracking
- `.agents/teamwork/explorer_survey_3/handoff.md` — Final handoff report
