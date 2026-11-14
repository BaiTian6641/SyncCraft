# Implementation Task: SyncCraft Production Colab End-to-End Pipeline Notebook

Read:
- `/Users/tianrui6641/SyncCraft/.agents/teamwork/ORIGINAL_REQUEST.md`
- `/Users/tianrui6641/SyncCraft/AGENTS.md`
- `/Users/tianrui6641/SyncCraft/.agents/teamwork/PROJECT.md`
- `/Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_1/handoff.md`
- `/Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_2/handoff.md`
- `/Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_3/handoff.md`

## File Ownership
You exclusively own:
- `/Users/tianrui6641/SyncCraft/notebooks/synccraft_end_to_end.ipynb`

Do NOT edit `tests/test_notebook_e2e.py` or any file outside your ownership.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Detailed Requirements
Deliver a production-ready, beautiful, well-documented Jupyter notebook at `notebooks/synccraft_end_to_end.ipynb` configured for Google Colab G4 instance (with NVIDIA L4/T4/G4 GPU, 16-24GB VRAM) and dual-mode CPU headless fallback.

The notebook must contain 6 clearly delineated markdown sections with production code cells:
1. **Section 1: Environment Setup & Prerequisites**
   - Check Python version (requires >= 3.10 / 3.11).
   - Check GPU availability (`torch.cuda.is_available()`), GPU model name, CUDA version, total VRAM.
   - Install/verify dependencies: `synccraft`, `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `huggingface_hub`, `Pillow`, `numpy`.
   - Set execution mode: `SMOKE_TEST_MODE = os.environ.get("SYNCCRAFT_SMOKE_TEST") == "1" or not torch.cuda.is_available()`.

2. **Section 2: Hugging Face Authentication & Colab Secrets Setup**
   - Retrieve `HF_TOKEN` from Google Colab Secrets using `from google.colab import userdata; hf_token = userdata.get('HF_TOKEN')`.
   - Fallback safely to `os.environ.get('HF_TOKEN')` if outside Colab or if userdata is not configured.
   - Authenticate via `huggingface_hub.login(token=hf_token)` and instantiate `HfApi`.
   - Redact token in any output/display (never print secret token values).
   - If token is missing, log a helpful warning and configure local fallback staging without crashing.

3. **Section 3: Dataset Gathering, Filtering & Manifest Generation**
   - Procedural generation / gathering of raw Block-DSL assets across seen static families (e.g. blades, axes, furniture) and unseen families.
   - Syntax and semantic validation using `parse_program` and `validate_program`.
   - Forward kinematics compilation using `compile_geometry(program)`.
   - Direct SAT collision audit using `audit_scene(scene)` with 15-axis SAT OBB (`sat_obb`), degenerate cross-axis masking, and contractual contact exclusions. Filter out any candidate with non-exempt collisions.
   - Export sample assets to Blockbench 5.0 (`.bbmodel`) and Java Modern (`.json`) and render 4-view preview.
   - Generate verified `dataset_manifest.json` with 13 mandatory row fields (`asset_id`, `family_id`, `canonical_dsl_sha256`, `audit_version_and_results`, `split`, etc.) and top-level 64-hex SHA-256 manifest hash.

4. **Section 4: Model Training Pipeline (4B/7B QLoRA)**
   - Configuration adhering to `configs/training_static.yaml` and `configs/model_candidates.yaml`:
     - 4B BF16 LoRA or 7B NF4 QLoRA (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", ...)`).
     - LoRA parameters: $r=32$, $\alpha=64$, targeting all projection modules.
     - `gradient_checkpointing_enable()`.
     - `micro_batch_size=2`, `gradient_accumulation_steps=64` (in smoke test mode, scale steps appropriately for instantaneous completion).
     - Response-tokens-only supervisory cross-entropy loss masking.
   - Executable training loop with loss logging and step progression.
   - Atomic checkpointing saving to `checkpoints/checkpoint-{step}/` containing LoRA adapters, tokenizer, optimizer state, and `manifest.json`.

5. **Section 5: Public Hugging Face Hub Asset Publishing**
   - Idempotently create public repository `api.create_repo(repo_id=..., repo_type="model", private=False, exist_ok=True)`.
   - Upload checkpoint folder and dataset manifest via `api.upload_folder(...)`.
   - Generate and upload an informative `README.md` (model card) with dataset provenance, license (Apache-2.0), and training metrics.
   - Fall back gracefully to local archiving if `hf_token` is unavailable or write permission is missing.

6. **Section 6: Model Probing, Geometric & Collision Audit, and Gate G2 Evaluation**
   - Probe model on held-out tasks from the 6 unseen families (`mechanical_crossbow`, `hand_lantern`, `tripod_telescope`, `folding_windmill`, `anvil_forge`, `floating_ring_staff`).
   - Run AST validation and full 15-axis SAT collision audit (`audit_scene`) on generated completions.
   - Monotonic cost tracking using `CostLedger` (tokens, compiler calls, audit calls).
   - Evaluate Gate G2 decision logic using `decide_gate(evidence)` with manifest SHA-256 hashes and 8 mandatory checks.
   - Print a structured evaluation summary table.

Ensure the notebook is valid JSON (`nbformat` v4), has valid cell outputs/metadata, and can be executed headlessly via standard Python nbclient/nbconvert or pytest.

Write your handoff report to `/Users/tianrui6641/SyncCraft/.agents/teamwork/worker_impl_1/handoff.md`.
Update `/Users/tianrui6641/SyncCraft/.agents/teamwork/worker_impl_1/progress.md` as you work.
When complete, notify the orchestrator via `send_message`.


## 2026-11-14T03:51:38Z
You are a teamwork_worker.
Your state folder is: /Users/tianrui6641/SyncCraft/.agents/teamwork/worker_impl_1/
Working directory: /Users/tianrui6641/SyncCraft
Original request file: /Users/tianrui6641/SyncCraft/.agents/teamwork/ORIGINAL_REQUEST.md
Dispatch file: /Users/tianrui6641/SyncCraft/.agents/teamwork/worker_impl_1/DISPATCH.md

Read ORIGINAL_REQUEST.md, AGENTS.md, PROJECT.md, and your DISPATCH.md.
Review findings from `/Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_1/handoff.md`, `/Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_2/handoff.md`, and `/Users/tianrui6641/SyncCraft/.agents/teamwork/explorer_survey_3/handoff.md`.

Your file ownership:
- You exclusively own `/Users/tianrui6641/SyncCraft/notebooks/synccraft_end_to_end.ipynb`.
Do not touch `tests/test_notebook_e2e.py` or any test files.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Implement the production Google Colab notebook `notebooks/synccraft_end_to_end.ipynb` fulfilling all requirements R1-R4 across Sections 1-6 as specified in your DISPATCH.md.
Verify that the notebook is syntactically valid JSON (nbformat v4) and executes cleanly.
Write your handoff report to `/Users/tianrui6641/SyncCraft/.agents/teamwork/worker_impl_1/handoff.md` and send a completion message when finished.
