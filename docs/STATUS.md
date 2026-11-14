# Starter status and handoff boundary

Release: **0.1.0**, updated 2026-11-14 (`ADR 0006` / `P1-08-r4`). Intent baseline: supplied SyncCraft v3.0
report + `ADR 0003`–`0006`. This is a scientifically grounded local CPU world-model & compiler foundation,
not yet a GPU-trained model checkpoint or live-GUI-verified release.

## Implemented and exercised in this environment

| Capability | Evidence and limits |
| --- | --- |
| Block-DSL v3.1 (`ADR 0004`, `ADR 0006`) | Typed parsing, validation, canonical formatting (`500` seeded static AST roundtrips + full `v3.1` roundtrip); semantic 3D `locator` sockets (`6` roles), per-axis `joint_limits`, arbitrary-depth nested `mirror_x` with $O(3)$ parity toggling (`MAX_EXPANDED_MIRROR_NODES = 8192`), extended `@materials` (`weld_group`), `@style`, `@muscles`, `@kinematics`, and `@behavior` |
| Local/world SE(3) & O(3) hierarchy | Root, `attach`, `place`, Euler composition, arbitrary-depth reflected subtrees (`det(R) = +1`), posed `joint_limits` clamping, `box_inflate_deltas`, `root_translation_delta`, and world-space `WorldLocator` compilation |
| Reduced-order musculoskeletal & closed-loop IK (`ADR 0006`) | Hill-type muscle-tendon equilibrium (`evaluate_muscles_at_activations`) with agonist–antagonist co-contraction stiffness, biarticular `TendonCoupling` pantographs, exact 3D SE(3) fiber-shortening `belly_box` `bulge_inflate` (integer Box-UV immutable), multi-bone `solve_look_chain`, world-anchored $C^1$ `foot_cycle` stance IK (`< 0.02` units horizontal stance slip), and multi-phase swept SAT `audit_behavior_clip` |
| CPU OBB collision & physical contact policy | Degenerate-axis-safe 15-axis SAT, 1,000 actual flush fixtures, direct-vs-smooth-loss counterexample, world-AABB broadphase, P1-02 physical contact/co-penetration/gap/connectivity policy (`enforce_physical_contact=True`), authorized `detached=true` bone tracking |
| Categorized 55-family + 10-Apex Complex dataset (`ADR 0005` + `ADR 0006`) | `55` core procedural families (`40` seen + `15` held-out unseen `>=16` boxes) + `10` Apex Complex entity/asset families (`8` seen + `2` held-out unseen `>=40` boxes, `32–64` boxes/asset: Monster Hunter-class nested-mirror wyverns/dragons/leviathans/switch-axe + clean-room Cobblemon-class creature & interactive station), batch SPC QC, `~5%` controlled defect-repair supervision, and reserved `human_gathered_real_corpus` track |
| Closed-loop artist assistant & non-diffusion progressive builder (`ADR 0006`) | Multi-format project intake (`Block-DSL`, `.bbmodel`, Java `.json`, Bedrock/Cobblemon `.geo.json` + zero-`eval` Molang `.animation.json` decompiler), `CC-BY-NC` (`CC-BY-NC-4.0`/`CC-BY-NC-SA-4.0`) `research_nc` quarantine gate, 5-pillar quantitative artist critique (`analyze_artist_project`), iterative rule-verified upgrade loop (`run_looped_artist_enhancement`), and 5-stage progressive from-scratch builder (`build_progressive_from_scratch_trajectory`) |
| Stateful surface & 10-family material processes (`ADR 0003`, `P1-03a`, `P1-03b`) | Separated `MaterialProgram`, `PersistentSurfaceState`, `OpticalFields`, `StyleConfig`; rest-space `SurfaceDomain` with 3 distinct graphs (`manifold_seams` honoring `weld_group`, `posed_contacts`, nested-mirror `symmetry_pairs`); conservative manifold Laplacian & stratified Gray–Scott RD; 10 material families with explicit `reduced_order_model` labels |
| Multi-target Pixel-PBR compiler & 3D world renderer (`P1-03c`) | Exact interval-overlap `downsample_optical_fields`; linear-space Cook–Torrance GGX (`alpha = alpha_p^2`, Airy thin-film) + OKLCH palette ramp quantization; Vanilla, LabPBR 1.3 (`_n`/`_s` channel-audited), and glTF 2.0 PBR (`ORM` + `KHR_materials_iridescence`) exporters; Z-buffered orthographic 3D ray-OBB renderer `render_world_observation` |
| Shared state evolution & causal interventions (`P1-03d`, `ADR 0006`) | Pure transition operator `step_world(S_t, a_t, E_t, dt)` over pose, trajectory, 2-bone analytic IK, `ExecuteBehaviorClipAction`, `ActivateBehaviorStateAction`, rubbing, material/geometry/density/lighting/UV-repack actions with sub-stepped swept SAT checks; 7/7 core causal intervention suite + musculoskeletal clip/state transitions verified |
| Inverse material inference & world trajectories (`P1-03e`) | License-gated (`clean_room_permissive` vs quarantined `research_nc`) decompiler outputting top-`K` `MaterialProgram` hypotheses + `StampOverride`s + identifiability scores; 9-geometry-family `infer`/`predict`/`plan_repair` trajectory dataset with zero compositional triple leakage |
| Colab E2E BF16 LoRA + Assistant World-Model notebook | [notebooks/synccraft_end_to_end.ipynb](file:///Users/tianrui6641/SyncCraft/notebooks/synccraft_end_to_end.ipynb): standalone Colab bootstrap targeting `LiquidAI/LFM3-VL-2B` BF16 PEFT LoRA (`r=32, alpha=64`, zero quantization) over categorized assistant + world-trajectory turns (CPU smoke-tested locally; live GPU run pending) |
| Exporters, CLI, and cost/gate helpers | Blockbench 5 `.bbmodel` and Java modern/legacy `.json` exporters; deterministic CLI pipeline; monotonic cost counters; explicit Gate G2 threshold (`configs/gate_g2_evidence.json` remains honestly blocked until unseen dense-assembly empirical evidence is recorded) |

Exact executed results and runtime versions are in [artifacts/validation/summary.json](file:///Users/tianrui6641/SyncCraft/artifacts/validation/summary.json)
and [artifacts/validation/cpu-tests.txt](file:///Users/tianrui6641/SyncCraft/artifacts/validation/cpu-tests.txt). Comprehensive architectural review is in
[docs/REPO_COMPREHENSIVE_REVIEW.md](file:///Users/tianrui6641/SyncCraft/docs/REPO_COMPREHENSIVE_REVIEW.md).

## Not yet implemented

| Capability | Blocking task |
| --- | --- |
| Polygon extrusion (`extrude`) and legacy `@shade` / `@motion` text keywords | P1-01 / P3-01 (`UnsupportedFeature` raised explicitly) |
| Live Blockbench GUI roundtrip & pinned Minecraft Iris LabPBR 1.3 shader runtime verification | P0-02 / P1-03 external acceptance |
| Grip/display solving and hand constraints (`@ergonomics`) | P1-04 |
| Differentiable Torch geometry, gradients and PGD | P1-05 / P1-06 |
| XGrammar, live symbol masks and real KV rollback | P1-07 |
| Live GPU model bake-off, RTX PRO 6000 BF16 LoRA training run, GPU memory/restart proof | P0-03 / P0-04 / P1-09 |
| Full baseline evaluation and Gate G2 empirical unlock | P1-10 |
| Learned refiner and training | P2-01 / P2-02, conditional only (`< 90%` G2 trigger) |
| Outbound Bedrock `.geo.json` and GeckoLib exporters | P3-03 (`ExportError` raised explicitly; inbound Bedrock/Cobblemon decompiler is implemented) |
| Independent benchmark and final paper evidence | P4-01 / P4-02 |

## External validation not performed here

No Blockbench GUI import/save/reopen, Minecraft game import, GeckoLib runtime execution, live GPU training run,
live tokenizer XGrammar masking run, or external human-modeler study was performed in this CPU session.

## Known limitations that affect interpretation

`audit_scene(..., enforce_physical_contact=True)` and `audit_behavior_clip(...)` certify finite valid OBB evaluation,
absence of non-exempt rest/swept collisions, genuine physical surface contact or bounded shallow inset (`<= 0.5` model units)
on declared `attach`/`place` pairs, physical connectivity up to authorized `detached=true` bones, and `< 0.02` model-unit
horizontal stance-foot slip on `foot_cycle` IK clips. Musculoskeletal mechanics (`@muscles`) use a quasi-static reduced-order
Hill-type equilibrium solver (`reduced_order_model`), not full rigid-body forward Euler–Lagrange multi-body contact dynamics.
