# Starter status and handoff boundary

Release: **0.1.0**, prepared 2026-11-13. Intent baseline: supplied SyncCraft v3.0
report. This is a local development starter, not a trained generator or completed
research system.

## Implemented and exercised in this environment

| Capability | Evidence and limits |
| --- | --- |
| Static box DSL | Typed parsing, validation, canonical formatting; 500 seeded AST roundtrips |
| Local/world hierarchy | Root, attach, place, Euler composition, reflected subtrees |
| CPU OBB collision | Degenerate-axis-safe SAT, 1,000 actual flush fixtures, direct-vs-smooth-loss counterexample |
| Static geometry diagnostics | Non-exempt collision check, exact world-AABB broadphase, P1-02 physical contact/co-penetration/gap/connectivity policy (`enforce_physical_contact=True`), authorized `detached=true` bone tracking |
| Procedural static dataset (`P1-08`) | 14 combinatorial typed-AST family generators (`8` seen + `6` unseen, `15–23` lowered boxes, `3–6` bones), multi-core JSONL/Parquet shard builder (`smoke`, `pilot_200`, `bakeoff_8k`, `full_60k` = `60,000 + 1,200` SHA-256-deduplicated rows) |
| Stateful surface & 10-family material processes (`ADR 0003`, `P1-03a`, `P1-03b`) | Separated `MaterialProgram`, `PersistentSurfaceState`, `OpticalFields`, `StyleConfig`; optional `@materials` & `@style` DSL blocks; rest-space `SurfaceDomain` with 3 distinct graphs (`manifold_seams`, `posed_contacts`, `symmetry_pairs`); conservative manifold Laplacian & stratified Gray–Scott RD (`supported_pattern_mechanism`); 10 material families with explicit `reduced_order_model` labels |
| Multi-target Pixel-PBR compiler & 3D world renderer (`P1-03c`) | Exact interval-overlap `downsample_optical_fields`; linear-space Cook–Torrance GGX (`alpha = alpha_p^2`, distinct `metalness` vs `dielectric_f0`, Airy thin-film) + OKLCH palette ramp quantization; separated Vanilla, LabPBR 1.3 (`_n`/`_s` channel-audited), and glTF 2.0 PBR (`ORM` + `KHR_materials_iridescence`) exporters; Z-buffered orthographic 3D ray-OBB renderer `render_world_observation` (serialized & channel-audited on CPU; live Blockbench GUI / Minecraft Iris LabPBR shader runtime pending `P0-02`) |
| Shared state evolution & 7 causal interventions (`P1-03d`) | Pure transition operator `step_world(S_t, a_t, E_t, dt)` over typed pose, trajectory, 2-bone analytic IK (`solve_two_bone_ik`), rubbing, material/geometry/density/lighting/UV-repack actions with sub-stepped swept SAT checks; 7/7 causal intervention suite verified (`0` 3D pixel diff on UV repack, `0.0` rest-albedo swim on repose, Archard hinge wear -> selective bare-metal oxidation, `0.46%` Turing spot drift across solver/pixel resolutions) |
| Inverse material inference & world trajectories (`P1-03e`) | Permissive-license gated (`CC0-1.0`/`CC-BY-4.0`/`MIT`/`Apache-2.0`/`BSD-3-Clause`) decompiler for `.bbmodel`/Java `.json`/Bedrock `.geo.json` outputting top-`K` `MaterialProgram` hypotheses + `StampOverride`s + identifiability scores; `pilot_200` (`210`-row) `infer`/`predict`/`plan_repair` trajectory dataset with zero compositional triple leakage and matched-budget policy evaluation |
| Colab E2E BF16 LoRA + World-Model notebook | `notebooks/synccraft_end_to_end.ipynb`: standalone Colab repo bootstrap, dataset-first HF Dataset Hub upload + `world_trajectories.jsonl`, native BF16 PEFT LoRA (`r=32, alpha=64, quantization=none`) with selectable `SUPERVISION_CURRICULUM` (`program_only_sft` vs `world_model_joint`), dynamic Gate G2 prerequisite inspection (`inspect_runtime_gate_prerequisites()`), and matched-budget world-model evaluation (CPU smoke-tested locally; live RTX PRO 6000 GPU run pending user execution) |
| Blockbench 5 export | Stable UUIDs, groups/outliner, transforms, atlas; structural/numeric tests only |
| Java export | Modern XYZ and conservative legacy; composed rotation and bounds checks; not a packaged resource pack |
| Texture/preview | Integer padded per-face atlas (`allocate_uv_layout`) + both diagnostic flat colors (`build_atlas`) and multi-target Pixel-PBR shading (`shade_asset`); CPU diagnostic four views + 3D ray-OBB world renderer |
| CLI and file pipeline | Actual export files, deterministic repeated output, controlled collision failures |
| Cost/gate helpers | Positive monotonic operation counts; explicit G2 threshold; empty evidence stays blocked |

Exact executed results and runtime versions are in `artifacts/validation/summary.json`
and `artifacts/validation/cpu-tests.txt`. Example outputs are in
`artifacts/examples/`. `scripts/check.py` reruns the current suite locally.

## Not yet implemented

| Capability | Blocking task |
| --- | --- |
| Extrusion, `@shade` DSL grammar, nested mirrors | P1-01 |
| Live Blockbench GUI roundtrip & pinned Minecraft Iris LabPBR 1.3 shader runtime verification | P0-02 / P1-03 external acceptance |
| Grip/display solving and hand constraints (`@ergonomics`) | P1-04 |
| Differentiable Torch geometry, gradients and PGD | P1-05 / P1-06 |
| XGrammar, live symbol masks and real KV rollback | P1-07 |
| Live GPU model bake-off, RTX PRO 6000 BF16 LoRA training run, GPU memory/restart proof | P0-03 / P0-04 / P1-09 |
| Full baseline evaluation and G2 evidence | P1-10 |
| Learned refiner and training | P2-01 / P2-02, conditional only |
| `@motion` text grammar recovery, stance-foot gait fitting, Bedrock and GeckoLib export | P3-01 / P3-02 / P3-03 |
| Independent benchmark and final paper evidence | P4-01 / P4-02 |

## External validation not performed here

No Blockbench GUI import/save/reopen, Minecraft game import, GPU execution, model
download, tokenizer masking run, training job, Colab billing measurement, or external
modeler benchmark was performed. Linux CPU tests do not prove Windows/macOS setup
scripts or GPU dependency compatibility. Local import and platform checks are the
first handoff tasks.

## Known limitations that affect interpretation

`audit_scene(..., enforce_physical_contact=True)` certifies finite valid OBB evaluation,
absence of non-exempt collisions, genuine physical surface contact or bounded shallow inset
(`<= 0.5` model units) on declared `attach`/`place` pairs (rejecting separated floating gaps
and deep co-penetrations such as `NUMERICAL_NOTES.md` §4 `D=0` and `D=50`), and physical
connectivity up to explicitly authorized `detached=true` bones. It **does not** establish
usable hand grip (`P1-04`), production shading/UV seam quality (`P1-03`), `extrude` polygons
(`P1-01`), or live GUI/game engine import (`P0-02`).

The plan's full static example includes features deliberately rejected by the
starter subset. The bundled examples use supported syntax and original geometry.
Flat palette colors are diagnostic; the configured ramps/hue shifts/emission are
preserved in the AST but explicitly listed as unused by diagnostic builds.

## First work queue

1. Reproduce S0 and record the local environment.
2. Import the real generated fixtures in Blockbench/Minecraft and record versions.
3. Complete P1-02's contact contracts and P1-01's extrusion/static language.
4. Work through the remaining dependencies in `tasks/BACKLOG.md`.

Do not start the learned branch based on starter pass rates. The supplied
`configs/gate_g2_evidence.json` intentionally yields a blocked decision and exit 2.
