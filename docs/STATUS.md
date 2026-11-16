# Project status and handoff boundary

Release: **0.1.0**, updated 2026-11-15 (`ADR 0007` sculpted partial-overlap & colossal entity extensions). Intent baseline: supplied SyncCraft v3.0
report + `ADR 0001`–`0007`. This is a scientifically grounded local CPU world-model, compiler, and clean-room dataset foundation,
not yet a GPU-trained model checkpoint or live-GUI-verified release (`TASKS.json`).

## Implemented and exercised on CPU in this environment

| Capability | Evidence and limits |
| --- | --- |
| Block-DSL v3.2 (`ADR 0004`, `ADR 0006`, `ADR 0007`) | Typed parsing, validation, compact canonical formatting (omitting default noise `offset=[0,0,0]`, `detached=false`, `inflate: 0.0`, `blend_with: []`), and Lark EBNF parity; `audit_profile: sculpted` + explicit `blend_with` surface-blending contracts (mirrored & cyclic `radial` involution), continuous `spin` mechanical track law (`[-1440, 1440]` deg), `size_step` fractional grids, `plane` zero-thickness quads, per-box `rotate`, `locator` sockets (`6` roles), `joint_limits`, arbitrary-depth nested `mirror_x` (`MAX_EXPANDED_MIRROR_NODES = 8192`), `let`, `def`/`use`, `chain`, `mirror`/`radial`, `spring`, `@controller`, `@textures`, `@materials` (`weld_group`), `@style`, `@muscles`, `@kinematics`, and `@behavior` |
| Shared NumPy & PyTorch SE(3) / O(3) & SAT | Right-handed `SE(3)` hierarchy (`det(R) = +1`), arbitrary-depth reflected subtrees, degenerate-axis-safe 15-axis OBB SAT cross-checked between NumPy and PyTorch, `strict` (`P1-02` physical contact-patch `>= 1e-4` area, Gate `G2` untouched), `sculpted` (`ADR 0007` explicit `sculpted_overlap_pairs`, anti-swallowing 8-corner OBB protrusion `> min(0.25, 0.15 * t_thin)`, bounded `<= 0.85 * t_med` overlap), and `authored` audit profiles, plus PGD sculpted-overlap projection and post-quantization 8-corner re-audit |
| Reduced-order Zajac/Hill musculoskeletal & IK (`ADR 0006`, `ADR 0007`) | Active force-length + passive parallel-elastic muscle-tendon equilibrium (`fidelity_label == "reduced_order_qualitative"`), agonist–antagonist co-contraction stiffness, biarticular `TendonCoupling` pantographs, Box-UV-safe `belly_box` `bulge_inflate`, continuous `spin` gear/wheel/rotor law with gimbal-safe `360°`/`720°` keyframe loop extrapolation and native Molang `query.anim_time`, 2-bone IK + CCD spine/tail chains, world-anchored $C^1$ `foot_cycle` stance IK (`< 0.02` units horizontal stance slip), multi-phase swept SAT `audit_behavior_clip`, and verified flagship `62`-box `Sculpted Elder Wyvern` + `74`-box `Colossal Walking Citadel` exemplars |
| Clean-room 55-family + 10-Apex dataset (`ADR 0005`, `ADR 0006`) | `55` core procedural families (`40` seen + `15` held-out unseen `>=16` boxes, including `>=35` distinct crossbow tree shapes) + `10` clean-room Apex families (`8` seen + `2` held-out unseen `>=40` boxes, neutral naming `elemental_companion_critter`, `modular_botanical_healing_station`), zero split-name prompt leakage, comment-free `@behavior` targets, `distinct_tree_shapes_by_family`, and 3 holdout splits (`test_unseen_params`, `test_unseen`, `test_compositional`) |
| Closed-loop artist assistant & dual-track license gate (`ADR 0006`) | Multi-format intake (`Block-DSL`, animated `.bbmodel`, Java `.json`, Bedrock `.geo.json` with exact left-handed X-reflection, `0`-thickness `plane`s, per-cube `rotate`, and zero-`eval` Molang/Catmull-Rom `.animation.json`), strict `clean_room_permissive` vs `research_nc` (`CC-BY-NC-3.0`/`4.0`) quarantine rejecting `CC-BY-ND`, `NoAI`/`do-not-train` notices, and defaulting unlabeled local inputs to `NOASSERTION`, plus 5-pillar critique & 5-stage progressive builder |
| Stateful surface & Pixel-PBR compiler (`ADR 0003`) | Separated `MaterialProgram`, `PersistentSurfaceState`, `OpticalFields`, `StyleConfig`; conservative manifold Laplacian & CFL Gray–Scott RD (`reduced_order_qualitative`); exact physical Cook–Torrance GGX (white-furnace `<= 1.0`, reciprocity) & Airy thin-film extrema (`physical_law_validated`), OKLab/OKLCH ramps; Vanilla, LabPBR 1.3, and glTF 2.0 exporters; Z-buffered 3D ray-OBB renderer `render_world_observation` |
| 7-Level World Capability Benchmark & Bounded Repair | Genuine `evaluate_world_capability_benchmark(records, model_completions)` across all 7 capability levels (`read_structure`, `predict_pose_collision`, `predict_motion_gait`, `predict_surface`, `invert_observation` with rendered 4-views, `edit_and_repair`, `human_transfer`) + parametric consistency probes + explicit `simulator_oracle_upper_bound`, `constant_prior_baseline`, and `nearest_neighbor_train_retrieval`; `run_bounded_repair_policy` (`<= 3` re-audits) and `verify_prompt_adherence` |
| Colab G4 E2E BF16 LoRA notebook | [notebooks/synccraft_end_to_end.ipynb](../notebooks/synccraft_end_to_end.ipynb): Hub-first prebuilt shard loader (`TassP0lar/synccraft-v3-blockdsl-dataset`), `StreamMixtureSampler` (100% `main` without replacement, `<= 3` aux repeats), `apply_chat_template` + `<think>` stripping, VRAM-safe pre-FP32 response-only loss (`max_seq_len=8192`), periodic 20-min full-state checkpointing & `resume_from_checkpoint`, dynamic tokenizer-measured `max_gen_tokens`, honest `LFM Open License v1.0` (`< $10M ARR`) vs `Apache-2.0` Model Card, and hard refusal to upload smoke/CPU artifacts (CPU smoke-tested locally; live GPU run pending) |
| Exporters, CLI, VS Code Extension, and Gate G2 helpers | Blockbench 5 `.bbmodel`, Bedrock `1.12.0` `.geo.json` + `1.8.0` `.animation.json` (exact left-handed X-reflection), GeckoLib, Java modern/legacy `.json`, LabPBR 1.3, and glTF 2.0 with post-serialization 8-corner world vertex re-audit (`runtime_import_verified=False`); [editors/vscode-blockdsl](../editors/vscode-blockdsl) TextMate grammar, Markdown fence injection, expanded-symbol auto-completion, hover cards, instant lint + `.venv` SE(3)/SAT bridge, and `.vsix` packager ([scripts/package_vscode_extension.py](../scripts/package_vscode_extension.py)); monotonic `CostLedger`; artifact-derived Gate G2 (`configs/gate_g2_evidence.json` remains `template_no_experiment_run` until GPU holdout evaluation is executed) |

Validation summaries are in [artifacts/validation/summary.json](../artifacts/validation/summary.json)
and [artifacts/validation/cpu-tests.txt](../artifacts/validation/cpu-tests.txt). Comprehensive architectural review is in
[REPO_COMPREHENSIVE_REVIEW.md](./REPO_COMPREHENSIVE_REVIEW.md).

## Not yet implemented or pending external/GPU execution

| Capability | Blocking task |
| --- | --- |
| Polygon contour extrusion (`extrude`) and legacy `@shade` / `@motion` keywords | `P1-01` (`UnsupportedFeature` raised explicitly) |
| Grip/display solving and hand constraints (`@ergonomics`) | `P1-04` (`UnsupportedFeature` raised explicitly) |
| Live tokenizer XGrammar masks and KV-cache statement rollback | `P1-07` (`NotImplementedError` raised explicitly) |
| Live Blockbench 5 GUI roundtrip & pinned Minecraft 1.21.11 / Iris LabPBR 1.3 / Bedrock / GeckoLib client runtime verification | `P0-02` / `P3-02` external acceptance (`runtime_import_verified=False`) |
| Live GPU candidate smoke checks, 8K 1-epoch bake-off (`LFM3-VL-2B` vs `Qwen2.5-Coder-7B-Instruct`), and Colab G4 RTX PRO 6000 BF16 LoRA training run | `P0-03` / `P0-04` / `P1-09` |
| Full unseen dense-assembly GPU evaluation and empirical Gate G2 (`90%`) decision | `P1-10` |
| Conditional learned residual refiner training & cached ablations | `P2-01` / `P2-02`, strictly conditional on measured valid Gate G2 pass rate `< 0.90` |
| `HandCraft-Bench-50` external benchmark and final release package | `P4-01` / `P4-02` |

## External validation not performed here

No Blockbench GUI import/save/reopen, Minecraft Java or Bedrock client import, GeckoLib runtime execution, live Colab G4 GPU training run,
live tokenizer XGrammar masking run, or external human-modeler study was performed in this local CPU environment.
