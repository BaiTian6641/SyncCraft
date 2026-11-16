# SyncCraft v3.2 — Symbolic & Physical 3D Block-Asset World Model

**A runnable, CPU-verified compiler, physics/surface world model, clean-room dataset generator, and Colab LoRA training/evaluation suite.**

The supplied v3.0 research plan (`source/synccraft-v3-plan.pdf`) and `ADR 0001`–`0006` define the system architecture: editable 3D block assets represented as typed `Block-DSL` programs, deterministic multi-target compilers, shared right-handed `SE(3)` kinematics, 15-axis OBB collision & physical contact auditing, stateful surface physics, musculoskeletal IK/locomotion, and measured Repair/PGD baselines.

Start with [START_HERE.md](START_HERE.md). Engineering governance is in [AGENTS.md](AGENTS.md), task status is in [TASKS.json](TASKS.json) and [docs/STATUS.md](docs/STATUS.md), and the full architectural review is in [docs/REPO_COMPREHENSIVE_REVIEW.md](docs/REPO_COMPREHENSIVE_REVIEW.md).

## Run locally (CPU)

Python 3.11 or newer is required (verified on Python 3.12 with NumPy 2.3+, Pillow 13+, PyTorch 2.2+, `pyarrow`, `nbformat`, and `lark`). No GPU, cloud account, or API key is required for the local test suite and CLI.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/check.py
python scripts/demo.py
```

For minimal core geometry/CLI dependencies only (`numpy` and `Pillow`), install `-r requirements-core.txt` and `-e . --no-deps`. For full Colab GPU training/evaluation dependencies (`transformers>=5.21.0`, `peft`, `torchao`, `datasets`, `huggingface_hub`), install `-e ".[ml]"`.

```bash
synccraft doctor
synccraft validate examples/simple_blade.blockdsl
synccraft compile examples/simple_blade.blockdsl --target java_modern --out runs/blade-java
synccraft compile examples/mirrored_wings.blockdsl --out runs/wings
```

Each compile writes a `.bbmodel`, canonical DSL, typed AST JSON, world-space scene, audit report, integer UV atlas, and four-view diagnostic preview. Java targets also write `asset.java.json`. Reuse an output directory with explicit `--overwrite`. A failed static audit prevents export by default.

## What works now (CPU-verified)

| Area | Implemented & CPU-verified scope |
| --- | --- |
| Block-DSL v3.2 | Typed AST, bounded zero-`eval` parser, compact canonical formatter, Lark grammar parity; `size_step`, `plane`, per-box `rotate`, `locator` sockets, `joint_limits`, arbitrary-depth nested `mirror_x`, `let`, `def`/`use`, `chain`, `mirror`/`radial`, `spring`, `@controller`, `@textures`, `@materials`, `@style`, `@muscles`, `@kinematics`, `@behavior` |
| Shared Kinematics & Collision | Shared NumPy & PyTorch `SE(3)` / `O(3)` chains (`det(R) = +1`), 15-axis OBB SAT with degenerate cross-axis masking, `P1-02` physical contact-patch audit & `authored` profile |
| Biomechanics & Motion | Reduced-order Zajac/Hill active force-length + passive parallel-elastic `@muscles`, biarticular tendon couplings, Box-UV-safe `belly_box` bulge, 2-bone IK + CCD chains, `< 0.02u` stance-slip `foot_cycle` gait, swept-pose SAT |
| Surface Physics & Pixel-PBR | Rest-space manifold Laplacian, CFL-substepped Gray–Scott reaction–diffusion, Archard wear & oxidation; linear-space Cook–Torrance GGX, Airy thin-film, OKLab/OKLCH ramps; Vanilla, LabPBR 1.3, and glTF 2.0 PBR compilers + CPU 3D ray-OBB renderer |
| Multi-Backend Exporters & Intake | Blockbench 5 `.bbmodel` (static + animated), Bedrock `1.12.0` `.geo.json` + `1.8.0` `.animation.json` (with exact left-handed X-reflection), GeckoLib, Java modern/legacy `.json`, zero-`eval` decompilers, and 5-pillar closed-loop artist assistant |
| Clean-Room Dataset & World Benchmark | `55` core procedural families + `10` clean-room Apex musculoskeletal families (`elemental_companion_critter`, `modular_botanical_healing_station`, etc.), zero split-name prompt leakage, 3 holdout splits (`test_unseen_params`, `test_unseen`, `test_compositional`), and 7-Level World Capability Ladder |
| Repair, Gate G2 & Colab Notebook | Bounded repair loop (`run_bounded_repair_policy`, `<= 3` re-audits with monotonic `CostLedger`), prompt-adherence verifier (`verify_prompt_adherence`), artifact-derived Gate G2 evaluator, and [notebooks/synccraft_end_to_end.ipynb](notebooks/synccraft_end_to_end.ipynb) (`StreamMixtureSampler`, VRAM-safe response loss, 20-min full-state checkpointing & resume, smoke-upload refusal) |
| VS Code Language Extension | [editors/vscode-blockdsl](editors/vscode-blockdsl): TextMate + Markdown code-fence highlighting, deterministic `mirror_x`/`radial`/`chain`/`def` expanded-symbol auto-completion, hover cards, snippets, go-to-definition, canonical formatting, live SE(3) + 15-axis SAT audit diagnostics, and zero-`npm` `.vsix` packager (`python scripts/package_vscode_extension.py`) |

## What still needs implementation or external validation

- **Unimplemented DSL / Decoder Milestones**: Polygon contour extrusion (`extrude`, `P1-01`), hand/grip ergonomics & display-slot solving (`@ergonomics`, `P1-04`), and live tokenizer XGrammar token masking with KV-cache rollback (`P1-07`) raise explicit `UnsupportedFeature` / `NotImplementedError` rather than returning silent approximations.
- **Pending GPU Execution (`P0-03`, `P0-04`, `P1-09`, `P1-10`)**: Candidate smoke checks, the comparable 8K 1-epoch bake-off (`LiquidAI/LFM3-VL-2B` vs `Qwen/Qwen2.5-Coder-7B-Instruct`), full Colab G4 NVIDIA RTX PRO 6000 (`~8.8 CU/hr`) BF16 LoRA training, and empirical unseen Gate G2 (`90%` threshold) evaluation have not yet been executed on GPU. [configs/gate_g2_evidence.json](configs/gate_g2_evidence.json) remains in `template_no_experiment_run` status, and the conditional learned refiner (`P2-01`/`P2-02`) stays gated on measured G2 `< 90%`.
- **Pending Live Engine / GUI Runtime Imports (`P0-02`, `P3-02`)**: Exporters pass numerical 8-corner post-serialization re-audits on CPU, **not yet live GUI/client imports** in Blockbench 5.0, Minecraft Java 1.21.11, Bedrock, or GeckoLib (`runtime_import_verified = False`).
- **External Benchmark (`P4-01`)**: `HandCraft-Bench-50` and external baseline comparisons remain pending.

See [docs/STATUS.md](docs/STATUS.md), [TASKS.json](TASKS.json), [docs/tasks/BACKLOG.md](docs/tasks/BACKLOG.md), and [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) before marking any phase complete.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/synccraft/` | Compiler, kinematics, SAT/contact audit, surface/optics, world simulator, data generators, modeling & exporters |
| `editors/vscode-blockdsl/` | Visual Studio Code grammar, semantic syntax highlighting, auto-completion, hover & live SAT audit extension |
| `notebooks/` | Standalone Colab G4 BF16 LoRA training, bounded repair, Gate G2 & 7-level World evaluation notebook |
| `examples/` | Original deterministic Block-DSL example assets; no imported game textures |
| `tests/` | Executable CPU regression, physics, dataset hygiene, adversarial & notebook E2E suites |
| `configs/` | Candidate models, training config (`max_seq_len: 8192`), Colab G4 budget, Gate G2 & dataset policies |
| `docs/` | Architecture, ADRs (`0001`–`0006`), geometry/compatibility contracts, status & comprehensive audit |
| `agent/` | Local-agent prompt and session reporting template |
| `artifacts/` | Validation summaries and recorded local CPU test evidence |
| `source/` | Unmodified supplied research plan plus extracted text and provenance |

## Licensing

Repository source code, clean-room procedural generators (`TassP0lar/synccraft-v3-blockdsl-dataset`), and original example definitions are released under `Apache-2.0`; see [LICENSE](LICENSE) and [NOTICE](NOTICE). External `CC-BY-NC-3.0` / `CC-BY-NC-4.0` community assets are strictly isolated under the non-commercial `research_nc` quarantine (`commercial_weights_eligible = False`), and any `NoAI` / `do-not-train` notice or unlabeled `NOASSERTION` provenance is blocked from commercial training.

Candidate base model licenses differ and govern trained LoRA derivatives (`configs/model_candidates.yaml`):
- **`LiquidAI/LFM3-VL-2B`**: Licensed under **`LFM Open License v1.0`**, which restricts free commercial use (including Derivatives / LoRA adapters) to entities below **`$10M USD` annual recurring revenue (ARR)**. Adapters trained on `LFM3-VL-2B` must never be labeled `Apache-2.0`.
- **`Qwen/Qwen2.5-Coder-7B-Instruct`**: Licensed under **`Apache-2.0`**, permitting unrestricted commercial adapter distribution when trained solely on `clean_room_permissive` data.
