# Project: SyncCraft Colab End-to-End Production Pipeline

## Architecture
- **Notebook Target**: `notebooks/synccraft_end_to_end.ipynb`
- **Target Hardware**: Google Colab G4 instance (NVIDIA L4/T4/G4 GPU, 16–24GB VRAM) and CPU headless execution mode.
- **Data Flow**:
  1. Environment & HF Secrets (`google.colab.userdata` with `os.environ` fallback, `huggingface_hub` login)
  2. Dataset Gathering, Geometry Validation, Direct SAT 15-Axis Collision Audit, and Manifest Generation (`dataset_manifest.json`)
  3. 4B/7B QLoRA Training Pipeline, Loss Masking, Gradient Checkpointing, Atomic Checkpoints, and Public HF Hub Publishing
  4. Model Probing on Held-Out Unseen Families, Geometric & Collision Audit (`audit_scene`), Multi-view Rendering, Format Export (Blockbench 5.0, Java Modern), and Gate G2 Verification (`decide_gate`).
- **Shared Interfaces**:
  - `synccraft.dsl.parser.parse_program`, `synccraft.dsl.ast_nodes.validate_program`
  - `synccraft.kinematics.se3_chain.compile_geometry`
  - `synccraft.audit.sync_audit.audit_scene`
  - `synccraft.evaluation.gate_g2.decide_gate`
  - `synccraft.evaluation.budget.CostLedger`

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Colab G4 Environment Setup & Prerequisites | Detect runtime, check GPU/VRAM, install and verify dependencies | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Hugging Face Authentication via Colab Secrets | Retrieve `HF_TOKEN` from `google.colab.userdata` with `os.environ` fallback, authenticate `huggingface_hub`, redact tokens | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Raw Asset Gathering & Procedural Pilot Generation | Collect built-in examples and procedural generators for seen/unseen families | M2 | ORIGINAL_REQUEST §R2 |
| 4 | Direct SAT Collision Audit Filtering | Filter candidate programs with 15-axis SAT OBB check, degenerate cross-axis masking, and contractual contact exclusions | M2 | ORIGINAL_REQUEST §R2 |
| 5 | Verified Dataset Manifest Generation | Produce verified `dataset_manifest.json` with 13 mandatory fields, SHA-256 hashes, and split tracking | M2 | ORIGINAL_REQUEST §R2 |
| 6 | 4B/7B QLoRA Training Pipeline | Configure BitsAndBytes NF4 / BF16 LoRA, response-tokens-only loss, gradient checkpointing, and execution loop | M3 | ORIGINAL_REQUEST §R3 |
| 7 | Atomic Model Checkpointing | Save adapters, tokenizer, optimizer state, scheduler, and `manifest.json` atomically | M3 | ORIGINAL_REQUEST §R3 |
| 8 | Hugging Face Public Asset Publishing | Push model checkpoints and dataset artifacts to public Hugging Face repository | M3 | ORIGINAL_REQUEST §R3 |
| 9 | Held-Out Probing & Audit Evaluation | Sample prompts from unseen families, probe completions, run AST validation and SAT collision audit | M4 | ORIGINAL_REQUEST §R4 |
| 10 | Export, Multi-View Rendering & Gate G2 Assessment | Export to Blockbench 5.0 and Java Modern, render 4-view preview, log metrics to `CostLedger`, evaluate `decide_gate` | M4 | ORIGINAL_REQUEST §R4 |
| 11 | Headless Automated Execution & E2E Validation | Ensure notebook executes end-to-end headlessly via `nbconvert` / `pytest` without unhandled exceptions | Final Milestone | ORIGINAL_REQUEST §Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Track | Requirement-driven opaque-box test suite (Tiers 1-4) in `tests/test_notebook_e2e.py` and `TEST_INFRA.md` | none | DONE |
| M1 | Environment & HF Secrets Setup | Colab G4 runtime detection, dependency configuration, and Colab Secrets HF auth | none | DONE |
| M2 | Dataset Gathering & SAT Filter Pipeline | Asset gathering, 15-axis SAT collision filter, and verified manifest output | M1 | DONE |
| M3 | 4B/7B QLoRA Training & HF Publishing | LoRA/QLoRA training loop, atomic checkpointing, and public HF asset publishing | M2 | DONE |
| M4 | Model Probing, Audit & Gate G2 | Probing on unseen families, geometric audit, Blockbench/Java export, and Gate G2 report | M3 | DONE |
| M5 | Final E2E Integration & Verification | 100% E2E test pass, adversarial coverage hardening, and final artifact packaging | E2E, M4 | DONE |

## Interface Contracts
### Notebook Cell Structure Contract
`notebooks/synccraft_end_to_end.ipynb` must be a valid Jupyter notebook containing:
- Delineated markdown headings:
  - `# Section 1: Environment Setup & Prerequisites`
  - `# Section 2: Hugging Face Authentication & Secrets Setup`
  - `# Section 3: Dataset Gathering, Filtering & Manifest Generation`
  - `# Section 4: Model Training (4B/7B QLoRA) & Checkpointing`
  - `# Section 5: Hugging Face Public Asset Publishing`
  - `# Section 6: Model Probing, Geometric & Collision Audit, and Evaluation`
- Execution parameters:
  - `SMOKE_TEST_MODE`: boolean flag, automatically set to `True` if `not torch.cuda.is_available()` or `os.environ.get("SYNCCRAFT_SMOKE_TEST") == "1"`.
  - When `SMOKE_TEST_MODE=True`, notebook executes all steps on CPU in under 30 seconds with 0 external network dependencies and 0 unhandled exceptions.
  - When `SMOKE_TEST_MODE=False` (on GPU Colab instance), executes full QLoRA / model pipeline.

### Checkpoint & Manifest Contract
- Checkpoints saved to `checkpoints/checkpoint-{step}/` containing `adapter_model.safetensors`, `adapter_config.json`, `manifest.json`.
- Dataset manifest saved to `datasets/dataset_manifest.json` containing `dataset_manifest_sha256` matching `^[a-fA-F0-9]{64}$`.

## Code Layout
- Notebook: `notebooks/synccraft_end_to_end.ipynb`
- Tests: `tests/test_notebook_e2e.py`
- Documentation / Test Infrastructure: `TEST_INFRA.md`, `TEST_READY.md`
- Core Source: `src/synccraft/` (read/imported by notebook)
