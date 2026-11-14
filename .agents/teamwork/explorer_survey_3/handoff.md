# Handoff Report: Model Training (4B/7B QLoRA), Checkpointing, Probing & Colab G4 Architecture

## 1. Observation

### 1.1 Training & Model Configuration Files
- **`configs/training_static.yaml`** (lines 1–41):
  - `schema: synccraft.training-intent/0.1`
  - `precision_intent: bf16_lora_4b_or_nf4_qlora_7b`
  - `max_sequence_length: 2048`
  - `micro_batch_size: 2`, `gradient_accumulation_steps: 64`, `effective_batch_size_single_gpu: 128`
  - `epochs: 2`
  - `lora`: `rank: 32`, `alpha: 64`, `dropout: 0.05`, `target_modules: resolve_from_actual_model`, `train_embeddings: false`, `expand_vocabulary: false`
  - `loss`: `response_tokens_only: true`, `chunked_fused_linear_cross_entropy: true`
  - `gradient_checkpointing: true`
  - `checkpoint`: `interval_minutes: 25`, `destination: local_until_remote_destination_authorized`, `atomic_complete_manifest: true`, `include: [adapters, refiner_if_enabled, optimizer, scheduler, scaler_if_used, rng_all_devices, sampler, shard_cursor, global_step, config_hash, model_revision, dataset_manifest]`
  - `targets_not_measurements`: `bf16_4b_peak_allocated_decimal_gb_less_than: 18.0`

- **`configs/model_candidates.yaml`** (lines 5–27):
  - Candidates:
    1. `Qwen3-4B`: Apache-2.0, 36-layer full attention, GQA, $d=2560$, ~8.05 GB BF16 / 2.6 GB NF4 static VRAM.
    2. `Qwen2.5-Coder-7B-Instruct`: Apache-2.0, 28-layer full attention, GQA, $d=3584$, vocab size 152,064, ~15.2 GB BF16 / 4.8 GB NF4 static VRAM, `suggested_memory_strategy: NF4_QLoRA_if_required`.
    3. `Spark-X2.6-4B`: Apache-2.0, 36 layers (27 sliding window + 9 full attention), $d=2560$, vocab size 131,072.

- **`src/synccraft/modeling/colab_trainer.py`** (lines 4–8):
  - Current implementation status:
    ```python
    def train(*args, **kwargs):
        raise NotImplementedError(
            "P1-09 requires verified model/data manifests, loss-kernel parity, actual "
            "VRAM profiling, and full-state atomic checkpoint/resume. No job was launched."
        )
    ```

### 1.2 Evaluation & Audit Codebase
- **`src/synccraft/audit/sync_audit.py`** (`audit_scene`):
  - Deterministic 15-axis SAT OBB intersection detection between box pairs using `sat_obb`.
  - Distinguishes declared contact pairs (`scene.adjacency`) from non-exempt collision pairs.
  - Returns `dict` with fields: `schema`, `scope`, `status`, `passed` (`bool`), `box_count`, `evaluated_non_exempt_pairs`, `collision_pairs`, `declared_contact_pairs`, `connected_components`, `coverage_gaps`.
  - Excludes declared contacts (`attach`, `place`) while recording penetrations; non-exempt overlaps trigger `passed = False`.

- **`src/synccraft/kinematics/sat_potential.py`** (`sat_obb`):
  - Evaluates up to 15 SAT axes (3 left faces + 3 right faces + 9 cross axes).
  - Explicitly filters degenerate cross-product axes: `norm > cross_axis_tolerance` ($10^{-10}$).
  - Computes `raw_min_overlap`, `penetration_depth`, `softmin_overlap`, `surrogate_energy`, and `collision = raw_min > collision_tolerance` ($10^{-7}$).

- **`src/synccraft/evaluation/gate_g2.py`** (`decide_gate`):
  - Validates Gate G2 evidence JSON against `REQUIRED_CHECKS`:
    `("full_static_audit_implemented", "heldout_split_frozen_before_training", "all_samples_have_at_least_16_parts", "repair_pgd_policy_preregistered", "matched_raw_budgets_enforced", "at_most_three_reaudits_per_sample", "failures_and_timeouts_counted", "baseline_implementations_verified")`.
  - Enforces 64-char hex SHA256 hashes for `dataset_manifest_sha256` and `evaluation_manifest_sha256`.
  - Evaluates threshold: `passed * 10 < total * 9` (strict 90% boundary). If $<90\%$, `eligible_for_learned_refiner`; if $\ge 90\%$, `skip_learned_refiner`; if missing evidence, `blocked`.

- **`src/synccraft/evaluation/budget.py`** (`CostLedger`):
  - Nonnegative monotonic counters: `accepted_tokens`, `discarded_tokens`, `repair_tokens`, `projection_steps`, `compiler_calls`, `initial_audit_calls`, `reaudit_calls`.
  - `total_generated_tokens = accepted_tokens + discarded_tokens + repair_tokens`.

- **`configs/data_policy.yaml`**:
  - Seen static families: `straight_blade_sword`, `single_edge_blade`, `battle_axe`, `spear`, `war_hammer`, `simple_wood_bow`, `furniture_table_chair_chest_cabinet`, `gearbox`.
  - Held-out unseen static families: `mechanical_crossbow`, `hand_lantern`, `tripod_telescope`, `folding_windmill`, `anvil_forge`, `floating_ring_staff`.

### 1.3 Hardware & Environment Requirements
- **Plan Document (`source/synccraft-v3-plan-extracted.txt`)**:
  - Section 5.1: `micro_batch=2`, `grad_accum=64`, `effective_batch=128`, `max_sequence_length=2048`.
  - Large vocabulary ($152,064$ tokens) logits tensor: $2 \times 2048 \times 152064 \times 4 \approx 2.5\text{ GB}$ per micro-batch if unchunked in FP32; `response_tokens_only` and chunked cross-entropy or gradient accumulation keeps peak VRAM below $16.8\text{ GB}$.
  - Target GPU: Colab G4 instance (AWS G4dn with 16GB T4 GPU or GCP G2 with 24GB L4 GPU). On a 16GB GPU (T4), 7B models MUST use 4-bit NormalFloat (`NF4`) quantization via BitsAndBytes.
  - Periodic checkpointing: 25-minute cadence, saving adapter weights, optimizer state, scheduler, random seed, and dataset cursor.

---

## 2. Logic Chain

### 2.1 Target Model Training Pipeline (Point 1)
1. **Model & Precision Selection**:
   - On a Colab G4/T4 instance (16GB VRAM), loading a 7B model (`Qwen/Qwen2.5-Coder-7B-Instruct`) in full 16-bit precision requires ~14–15 GB for base weights alone, which immediately causes an Out-Of-Memory (OOM) error once optimizer states and activations are allocated.
   - Therefore, the 7B configuration requires BitsAndBytes NF4 QLoRA:
     ```python
     bnb_config = BitsAndBytesConfig(
         load_in_4bit=True,
         bnb_4bit_quant_type="nf4",
         bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
         bnb_4bit_use_double_quant=True,
     )
     ```
     This compresses base model weights to ~4.5–5 GB VRAM.
   - For a 4B candidate or smaller (e.g. `Qwen3-4B` / `Qwen2.5-Coder-3B`), BF16 LoRA consumes ~8 GB base weights, leaving ~8 GB for activations and gradients.
2. **PEFT LoRA Configuration**:
   - Following `configs/training_static.yaml`:
     ```python
     peft_config = LoraConfig(
         r=32,
         lora_alpha=64,
         lora_dropout=0.05,
         bias="none",
         task_type="CAUSAL_LM",
         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
     )
     ```
   - Only adapter parameters (~30M–60M parameters, ~0.5–1.0% of total) are updated; base weights remain frozen.
3. **Loss Computation & Dataset Formatting**:
   - Input format: Instruction prompt specifying model generation task (archetype, target profile, texel density, description) + assistant completion containing the Block-DSL code.
   - Tokenization: Prompt tokens are assigned label `-100`, so cross-entropy loss is strictly calculated on response tokens (`response_tokens_only: true`).
   - Gradient Checkpointing: Must be enabled (`model.gradient_checkpointing_enable()`), trading ~20% compute overhead for an ~80% reduction in activation memory during the forward-backward pass.

### 2.2 Checkpointing, Loss Logging & Public HF Hub Integration (Point 2)
1. **Colab Secrets & Hugging Face Authentication**:
   - Colab provides secure credential storage via `google.colab.userdata`.
   - The integration retrieves `HF_TOKEN` via:
     ```python
     try:
         from google.colab import userdata
         hf_token = userdata.get('HF_TOKEN')
     except Exception:
         hf_token = os.environ.get('HF_TOKEN')
     ```
   - If present, authenticate via `huggingface_hub.login(token=hf_token)` and instantiate `api = HfApi(token=hf_token)`.
   - If absent, warn clearly and save artifacts locally without crashing, ensuring headless and offline workflows remain intact.
2. **Atomic Checkpointing & Artifact Structure**:
   - Checkpoint cadence: Every 25 minutes or every $N$ optimizer steps.
   - Atomic saving protocol: Write files to a temporary staging folder (`.tmp_ckpt/`), generate `manifest.json` with SHA-256 hashes, step count, loss value, and epoch, then atomically rename to `checkpoints/checkpoint-{step}/`.
   - Artifact package includes:
     - `adapter_model.safetensors` & `adapter_config.json` (LoRA weights)
     - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`
     - `trainer_state.pt` (optimizer state, scheduler state, RNG seeds)
     - `dataset_manifest.json` (SHA-256 reference and split lineage)
     - `manifest.json` (checkpoint metadata and integrity hash)
3. **Public Repository Upload**:
   - Create repo idempotently: `api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)`
   - Upload adapter directory: `api.upload_folder(folder_path=str(checkpoint_dir), repo_id=repo_id, repo_type="model", commit_message=f"SyncCraft checkpoint step {step}")`.

### 2.3 Probing and Evaluation Pipeline (Point 3)
1. **Held-Out Probing Task Execution**:
   - Prompts are sampled from the 6 held-out static families (`configs/data_policy.yaml`: `mechanical_crossbow`, `hand_lantern`, `tripod_telescope`, `folding_windmill`, `anvil_forge`, `floating_ring_staff`).
   - The model generates candidate Block-DSL text completions.
2. **Geometric & SAT Collision Audit**:
   - For each generated candidate:
     - Syntax Validation: `parse_program(text)` and `validate_program(ast)` to verify AST nodes, positive integer dimensions, and valid symbol references.
     - Kinematics Compilation: `compile_geometry(ast)` creates world-space OBBs (`WorldBox`) and computes parent-child transforms and adjacency.
     - Collision Audit: `audit_scene(scene, collision_tolerance=1e-7)` tests all non-exempt pairs using 15-axis SAT (`sat_obb`). It excludes declared contacts (`attach`, `place`) and records whether non-exempt collisions occurred.
3. **Metric Logging & Gate G2 Assessment**:
   - Metrics computed across the evaluation batch:
     - `syntax_and_symbol_pass_rate` ($\frac{\text{valid syntax}}{\text{total}}$)
     - `full_static_audit_pass_rate` ($\frac{\text{passed audit}}{\text{total}}$)
     - `non_exempt_collision_rate` ($\frac{\text{collision}}{\text{total}}$)
     - `CostLedger` totals (`total_generated_tokens`, `compiler_calls`, `audit_calls`)
   - Gate G2 Evaluation: Assemble `evidence` JSON with required checks and manifest SHA-256 hashes, then call `decide_gate(evidence)`. If all checks pass and parts $\ge 16$, determine whether pass rate is $\ge 90\%$ (`skip_learned_refiner`) or $<90\%$ (`eligible_for_learned_refiner`).

### 2.4 Structuring `notebooks/synccraft_end_to_end.ipynb` for Colab G4 & Headless Verification (Point 4)
1. **Five Well-Delineated Sections**:
   - **Section 1: Environment Setup & Prerequisites**
     - Verify Python version ($\ge 3.11$), check GPU availability (`torch.cuda.is_available()`), GPU name, VRAM.
     - Authenticate Hugging Face with Colab Secrets (`HF_TOKEN` from `google.colab.userdata` with fallback).
     - Install / verify dependencies (`synccraft`, `transformers`, `peft`, `bitsandbytes`, `accelerate`).
   - **Section 2: Dataset Gathering, Filtering & Manifest Generation**
     - Generate / load raw static asset programs from `examples/` and family generators.
     - Apply SAT collision audit filtering (`audit_scene`), filtering out overlapping samples.
     - Generate verified `dataset_manifest.json` containing SHA-256 hashes, splits (train/val/heldout), and metadata.
   - **Section 3: Model Training (4B/7B QLoRA)**
     - Configure BitsAndBytes NF4 quantization (or FP16/BF16 based on device), load tokenizer and base model.
     - Apply PEFT `LoraConfig(r=32, alpha=64, target_modules=...)` and enable gradient checkpointing.
     - Execute training loop with `micro_batch_size=2`, `gradient_accumulation_steps=64` (or scaled for quick demo/test), logging losses and GPU VRAM usage.
     - Save atomic checkpoints with `manifest.json`.
   - **Section 4: Hugging Face Publishing**
     - Check `HF_TOKEN`. If available, push checkpoint folder and dataset manifest to user's public HF model repository (`api.upload_folder`). If absent, log local artifact path.
   - **Section 5: Probing, Geometric SAT Audits & Gate G2 Evaluation**
     - Probe model on held-out tasks from unseen families.
     - Run `parse_program`, `compile_geometry`, and `audit_scene` on outputs.
     - Export model to Blockbench (`bbmodel_5`) and Java (`java_modern`).
     - Render multi-view previews (`render_four_views`).
     - Log evaluation metrics and execute Gate G2 decision logic (`decide_gate`).
2. **Dual-Mode Headless & CPU Fallback Architecture**:
   - The notebook must support automated headless verification (e.g. running via `nbconvert` or in CI):
     - Check `IN_SMOKE_TEST = os.environ.get("SYNCCRAFT_SMOKE_TEST") == "1" or not torch.cuda.is_available()`.
     - In full GPU mode: runs full QLoRA loading and training.
     - In smoke-test/CPU mode: uses a lightweight surrogate model (or a mock 2-step loop with dummy gradients), executes full dataset manifest creation, full SAT collision audit on DSL examples, full Gate G2 logic, and verifies checkpoint saving.
     - This guarantees the notebook executes end-to-end headlessly in $<30$ seconds without downloading 15 GB weights or requiring an active GPU, while having fully executable code for real Colab GPU runs.

---

## 3. Caveats
1. **Local vs Colab Python Environment**: The local development workstation currently uses system Python 3.9.6, whereas SyncCraft requires Python $\ge 3.11$ (`requires-python = ">=3.11"` in `pyproject.toml`). In Google Colab, default Python is 3.10+ / 3.11+, and PyTorch 2.x is standard. The notebook should explicitly assert Python $\ge 3.11$ in Section 1.
2. **Hardware Discrepancy (T4 vs L4 vs A100)**: AWS G4dn instances feature NVIDIA T4 GPUs (Turing, 16GB VRAM, compute capability 7.5), which lack native BF16 tensor core acceleration; GCP G2 instances feature NVIDIA L4 GPUs (Ada Lovelace, 24GB VRAM) which natively support BF16. The training script must dynamically select `torch.bfloat16` if `torch.cuda.is_bf16_supported()` else `torch.float16`.
3. **Repository Trainer State**: `src/synccraft/modeling/colab_trainer.py` is currently a stub with `raise NotImplementedError`. The notebook must encapsulate its own executable training loop adhering to `configs/training_static.yaml`.
4. **Hugging Face Hub Permissions**: Pushing to Hugging Face requires a write-capable token (`write` scope). If a read-only token is supplied in Colab Secrets, repository creation/upload will fail. The notebook must handle `HfHubHTTPError` gracefully and log informative troubleshooting guidance.

---

## 4. Conclusion
1. **Training Architecture**: The 4B/7B training stage in `notebooks/synccraft_end_to_end.ipynb` should implement BitsAndBytes NF4 quantization (`bnb_4bit_quant_type="nf4"`), PEFT LoRA ($r=32$, $\alpha=64$, targeting all linear projections), `gradient_checkpointing_enable()`, `micro_batch_size=2`, and `response_tokens_only` loss masking. On a 16GB Colab G4 instance, this limits peak VRAM to ~7–9 GB.
2. **Checkpointing & Publishing**: Checkpoints must follow atomic directory generation and include LoRA adapters, tokenizer configurations, optimizer state, and `manifest.json`. Publishing to Hugging Face should consume `HF_TOKEN` from Colab Secrets via `google.colab.userdata` with environment variable fallback.
3. **Probing & Evaluation**: Probing must draw from the 6 held-out families in `configs/data_policy.yaml`. Audit execution directly calls `parse_program`, `compile_geometry`, and `audit_scene` (15-axis SAT OBB), tracking metrics via `CostLedger` and evaluating the Gate G2 threshold via `decide_gate`.
4. **Notebook Structure**: The notebook must be organized into 5 clearly demarcated sections with a dual-mode switch (`SMOKE_TEST_MODE`) allowing instantaneous headless test execution via `nbconvert` / `pytest` while supporting full-scale GPU runs on Colab.

---

## 5. Verification Method

### 5.1 Verification Commands
To verify the findings and code paths independently:

1. **Verify Config Integrity**:
   ```bash
   python3 -c '
   import yaml, json
   training = yaml.safe_load(open("configs/training_static.yaml"))
   assert training["precision_intent"] == "bf16_lora_4b_or_nf4_qlora_7b"
   assert training["lora"]["rank"] == 32
   assert training["gradient_checkpointing"] is True
   models = yaml.safe_load(open("configs/model_candidates.yaml"))
   assert any(c["label"] == "Qwen2.5-Coder-7B-Instruct" for c in models["candidates"])
   print("Config verification passed.")
   '
   ```

2. **Verify SAT Collision Audit & Gate G2 Logic**:
   ```bash
   python3 -c '
   import json
   from synccraft.evaluation.gate_g2 import decide_gate
   evidence = json.load(open("configs/gate_g2_evidence.json"))
   result = decide_gate(evidence)
   assert result["decision"] == "blocked"
   print("Gate G2 verification passed.")
   '
   ```

3. **Verify Budget Cost Ledger**:
   ```bash
   python3 -c '
   from synccraft.evaluation.budget import CostLedger
   ledger = CostLedger()
   ledger.record("accepted_tokens", 50)
   ledger.record("discarded_tokens", 10)
   assert ledger.total_generated_tokens == 60
   print("CostLedger verification passed.")
   '
   ```

4. **Verify DSL Examples & Kinematics Audit**:
   ```bash
   python3 -c '
   from pathlib import Path
   from synccraft.dsl import parse_program
   from synccraft.kinematics.se3_chain import compile_geometry
   from synccraft.audit.sync_audit import audit_scene

   for p in Path("examples").glob("*.blockdsl"):
       prog = parse_program(p.read_text())
       scene = compile_geometry(prog)
       report = audit_scene(scene)
       assert report["passed"] is True, f"Failed on {p}"
   print("All starter examples passed SAT collision audit.")
   '
   ```

5. **Verify Headless Notebook Execution Harness**:
   - When `notebooks/synccraft_end_to_end.ipynb` is implemented, verify headless execution:
     ```bash
     SYNCCRAFT_SMOKE_TEST=1 jupyter nbconvert --to notebook --execute notebooks/synccraft_end_to_end.ipynb --output /tmp/executed.ipynb
     ```
     or via a dedicated pytest notebook runner in `tests/test_notebook.py`.

### 5.2 Invalidation Conditions
- If the model backbone is switched to a non-Apache-2.0 license (violating `configs/model_candidates.yaml` and `AGENTS.md`).
- If batch size and sequence length without NF4 quantization exceed 16 GB VRAM on a Colab G4 instance.
- If SAT collision audit is bypassed in favor of heuristic bounds checks or unverified smooth potentials.
- If Hugging Face authentication crashes when `HF_TOKEN` is unset in headless / CI environments.
