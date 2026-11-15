# SyncCraft v3.2 — Comprehensive Technical, Mathematical & Architectural Audit Report

**Prepared for**: Independent Cross-AI & Principal Engineering Peer Review  
**Repository**: [SyncCraft](https://github.com/BaiTian6641/SyncCraft)  
**Dataset Hub**: `TassP0lar/synccraft-v3-blockdsl-dataset` (`https://huggingface.co/datasets/TassP0lar/synccraft-v3-blockdsl-dataset`)  
**Architecture Baseline**: SyncCraft v3.0 Research Plan (`source/synccraft-v3-plan.pdf`) + `ADR 0001`–`0006`  
**Candidate Backbones (`configs/model_candidates.yaml`)**: `LiquidAI/LFM3-VL-2B` (`LFM Open License v1.0` — commercial cap `< $10M ARR` covering derivatives) vs `Qwen/Qwen2.5-Coder-7B-Instruct` (`Apache-2.0` unrestricted commercial), BF16 PEFT LoRA `r=32, alpha=64`, `max_sequence_length=8192`, dynamic tokenizer-measured `max_gen_tokens`, zero quantization  
**Codebase Scale**: Synchronized across `src/synccraft/`, `notebooks/`, and `tests/` (verified via `scripts/check.py` and `scripts/demo.py` on CPU)  
**Local Verification Environment**: `Python 3.12.16`, `macOS 27.2.1 (arm64 Darwin 27.2.0)`, `numpy 2.3.5`, `Pillow 13.0.0`, `pyarrow 26.0.1`, `nbformat 5.11.1`, `torch 2.15.0` (CPU)

---

## Table of Contents

1. [Executive Summary & Anti-Diffusion World-Model Thesis](#1-executive-summary--anti-diffusion-world-model-thesis)
2. [End-to-End Architecture & Complete Module Inventory](#2-end-to-end-architecture--complete-module-inventory)
3. [Exact Mathematical, Geometric, Optical & Biomechanical Formulations](#3-exact-mathematical-geometric-optical--biomechanical-formulations)
   - 3.1 Block-DSL v3.1 Typed AST & Zero-`eval` Security Contract
   - 3.2 Shared $SE(3)$ Chain & Arbitrary-Depth $O(3)$ Reflection Parity
   - 3.3 Exact 15-Axis OBB SAT Audit & P1-02 Physical Contact Policy
   - 3.4 Rest-Space Manifold Surface Domain, Gray–Scott Reaction–Diffusion & Tribology
   - 3.5 Multi-Target Pixel-PBR Compiler (Vanilla, LabPBR 1.3, glTF 2.0) & 3D Ray-OBB Renderer
   - 3.6 Reduced-Order Hill Musculoskeletal Mechanics, Tendon Pantographs & Isochoric Bulge
   - 3.7 World-Anchored $C^1$ Stance-Foot IK & Swept-Pose Collision Auditing
4. [Complex Entities, Quarantine Governance & Closed-Loop Artist-Assistant Capabilities (`ADR 0006`)](#4-complex-entities-quarantine-governance--closed-loop-artist-assistant-capabilities-adr-0006)
   - 4.1 Clean-Room Biomechanical Entities & Morphing Mechanisms (32–68 Boxes)
   - 4.2 Stylized Companion/Station Rigs, Semantic Locators & `CC-BY-NC` / `NoAI` Quarantine
   - 4.3 Zero-`eval` Bedrock `.geo.json` (Left-Handed X-Reflection) & Molang `.animation.json` Decompilers
   - 4.4 Closed-Loop 5-Pillar Artist Critique, Iterative Upgrade & 5-Stage Progressive Builder
5. [Dataset Taxonomy, Curriculum, SPC Quality Control & Provenance Governance](#5-dataset-taxonomy-curriculum-spc-quality-control--provenance-governance)
6. [Executed Verification Matrix & Quantitative Telemetry](#6-executed-verification-matrix--quantitative-telemetry)
7. [Self-Critical Audit: Approximations, Unimplemented Milestones & Priority Attack Vectors for External AI Review](#7-self-critical-audit-approximations-unimplemented-milestones--priority-attack-vectors-for-external-ai-review)

---

## 1. Executive Summary & Anti-Diffusion World-Model Thesis

Most contemporary "3D/texture AI" pipelines treat asset creation as **unconstrained statistical denoising** over raw pixels, occupancy grids, or unstructured triangle soups. While visually impressive on isolated renders, black-box diffusion fails structural game-engine requirements:
1. **Topological & Kinematic Amnesia**: Denoised meshes lack articulated bone hierarchies, pivot semantics, joint limits, rider/VFX/hitbox locators, and integer Box-UV texel preservation.
2. **Physical Incoherence**: Unconstrained pose generation produces self-intersecting limbs, floating detached geometry, and horizontal foot-sliding ("moonwalking").
3. **Texture-Geometry Dissociation**: Diffusion texture baking bakes static lighting directly into albedo, causing rest-albedo "swimming" during skeletal deformation, UV seam discontinuities across adjacent boxes, and inability to evolve persistent physical states (moisture, Archard hinge wear, selective bare-metal oxidation, biofilm growth).
4. **Opaque Artist Iteration**: An artist cannot hand an existing `.bbmodel` or `.geo.json` project to a pixel denoiser and receive deterministic, rule-verified structural critiques or incremental stage-by-stage refinements.

**SyncCraft v3.2** takes the opposite scientific stance: **assets are executable, typed, causally coupled physical programs** ($\mathcal{W} = (\mathcal{G}, \mathcal{M}, \mathcal{S}, \mathcal{K}, \mathcal{B})$):
- **Geometry ($\mathcal{G}$)**: Constructive box/plane-and-socket CAD assembly compiled through a single shared $SE(3) / O(3)$ forward kinematics engine ([se3_chain.py](../src/synccraft/kinematics/se3_chain.py)) with Box-UV dimensions kept strictly invariant under continuous surface inflation (`inflate`).
- **Surface Physics ($\mathcal{M}, \mathcal{S}$)**: Rest-space 3D box-manifold PDEs ([domain.py](../src/synccraft/surface/domain.py), [processes.py](../src/synccraft/surface/processes.py)) separating intrinsic `MaterialProgram` parameters from time-evolving `PersistentSurfaceState` fields (moisture, wear, oxidation, temperature, biofilm, Gray–Scott $(u, v)$ morphogens), compiled through linear-space Cook–Torrance GGX / Airy thin-film optics ([compiler.py](../src/synccraft/shading/compiler.py)).
- **Biomechanics & Locomotion ($\mathcal{K}, \mathcal{B}$)**: Reduced-order Zajac/Hill muscle-tendon actuators (`@muscles` with active force-length + passive parallel-elastic curves), biarticular passive tendon pantographs, agonist–antagonist co-contraction stiffness, Box-UV-safe `belly_box` `bulge_inflate`, and world-anchored $C^1$ `foot_cycle` 2-bone IK (`< 0.02` model-unit stance slip) ([musculoskeletal.py](../src/synccraft/kinematics/musculoskeletal.py)).
- **Closed-Loop Artist Collaboration**: Deterministic project intake, 5-pillar quantitative critique, iterative rule-verified enhancement, and 5-stage progressive from-scratch construction ([artist_loop.py](../src/synccraft/world/artist_loop.py)) — every intermediate state audited by exact 15-axis Separating Axis Theorem (SAT) collision and `P1-02` / `authored` physical contact checks ([sync_audit.py](../src/synccraft/audit/sync_audit.py)).

---

## 2. End-to-End Architecture & Complete Module Inventory

```
                     ┌──────────────────────────────────────────────────────────┐
                     │          External Project / Prompt / Artist Loop         │
                     │  Block-DSL v3.1 | .bbmodel | Java .json | .geo/.anim.json│
                     └────────────────────────────┬─────────────────────────────┘
                                                  │ (Zero-eval Parser / License-Gated Decompiler)
                                                  ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Typed Program AST (synccraft.dsl)                                    │
│  @meta | @palette | @rig (Bone, Box, Locator, mirror_x, joint_limits) | @materials | @style       │
│  @muscles (Hill + TendonCoupling + belly_box) | @kinematics (IK + Look) | @behavior (Clip/State) │
└───────┬───────────────────────────────┬───────────────────────────────────┬───────────────────────┘
        │                               │                                   │
        ▼                               ▼                                   ▼
┌───────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────────┐
│  Kinematics & Biomech │   │  Rest-Space Surface Manifold  │   │     Audit & Verification Gate     │
│ (synccraft.kinematics)│   │      (synccraft.surface)      │   │         (synccraft.audit)         │
│ - Shared SE(3)/O(3)   │   │ - 3 Graphs: manifold_seams    │   │ - Exact 15-axis OBB SAT           │
│ - Nested mirror_x     │   │   (weld_group), posed_contacts│   │ - P1-02 Contact/Gap/Inset Policy  │
│ - Hill Muscle Solver  │   │   nested symmetry_pairs       │   │ - Multi-phase Swept Pose SAT      │
│ - 2-Bone LM/Analytic  │   │ - Conservative Laplacian      │   │ - Stance-Foot Slip (<0.02u) Audit │
│   IK & Stance Pinning │   │ - Gray-Scott RD + Archard Wear│   │ - Z-Buffered 3D Ray-OBB Renderer  │
└───────────┬───────────┘   └───────────────┬───────────────┘   └─────────────────┬─────────────────┘
            │                               │                                     │
            └───────────────────────────────┼─────────────────────────────────────┘
                                            ▼
                 ┌──────────────────────────────────────────────────────┐
                 │    Causal World Transition & Artist Loop Engine      │
                 │                  (synccraft.world)                   │
                 │  step_world(S_t, a_t, dt) | analyze_artist_project   │
                 │  run_looped_artist_enhancement | progressive_stages  │
                 └──────────────────────────┬───────────────────────────┘
                                            │
                                            ▼
                 ┌──────────────────────────────────────────────────────┐
                 │     Shading Compiler & Target Backend Exporters      │
                 │      (synccraft.shading + synccraft.backends)        │
                 │  Cook-Torrance GGX / Airy Thin-Film / OKLCH Ramps    │
                 │  Blockbench 5 (.bbmodel + anim) | Bedrock/GeckoLib   │
                 │  Java (.json) | glTF 2.0 | LabPBR 1.3 | Vanilla      │
                 └──────────────────────────────────────────────────────┘
```

### Complete Package & File Inventory (`src/synccraft/`)

| Subsystem | Module | Responsibility & Status |
|---|---|---|
| **DSL (`dsl/`)** | [ast_nodes.py](../src/synccraft/dsl/ast_nodes.py) | Typed immutable AST dataclasses (`Program`, `Bone`, `Box`, `Locator`, `MaterialDecl`, `StyleDecl`, `MuscleDecl`, `TendonCoupling`, `KinematicsDecl`, `BehaviorDecl`), v3.2 `size_step`/`plane`/`rotate`/`@textures`, semantic validator, and `MAX_EXPANDED_MIRROR_NODES = 8192` pre-expansion guard. **Complete (v3.2).** |
| | [parser.py](../src/synccraft/dsl/parser.py) | Bounded recursive-descent lexer/parser (`2 MB` / `200k` token / depth `64` caps) + compact canonical `format_program` serializer omitting default noise. Zero `eval`. Explicit `UnsupportedFeature` on `extrude`, `@shade`, `@motion`, `@ergonomics`. **Complete (v3.2).** |
| | [grammar.lark](../src/synccraft/dsl/grammar.lark) | Declarative EBNF reference grammar with verified parity against `parser.py` v3.2 for downstream XGrammar integration. **Verified spec.** |
| | [stateful_decoder.py](../src/synccraft/dsl/stateful_decoder.py) | Explicit stub raising `NotImplementedError` pending `P1-07` live tokenizer XGrammar + KV rollback. **Deferred (`P1-07`).** |
| **Kinematics (`kinematics/`)** | [se3_chain.py](../src/synccraft/kinematics/se3_chain.py) | Single source of truth for forward $SE(3)$ & arbitrary-depth $O(3)$ `mirror_x` expansion, `WorldBone`, `WorldBox`, `WorldLocator`, `joint_limits` clamping, `box_inflate_deltas`, and `root_translation_delta` (cross-checked with PyTorch). **Complete.** |
| | [musculoskeletal.py](../src/synccraft/kinematics/musculoskeletal.py) | Reduced-order Zajac/Hill active force-length + passive parallel-elastic equilibrium (`fidelity_label == "reduced_order_qualitative"`), agonist–antagonist stiffness, biarticular `TendonCoupling` pantographs, `belly_box` `bulge_inflate`, `solve_look_chain`, $C^1$ world-anchored `foot_cycle` IK, and `audit_behavior_clip`. **Complete (`ADR 0006`).** |
| | [sat_potential.py](../src/synccraft/kinematics/sat_potential.py) | Exact 15-axis Separating Axis Theorem OBB overlap solver with degenerate cross-axis masking (`1e-8`) + smooth LogSumExp surrogate diagnostic (cross-checked with PyTorch). **Complete.** |
| **Audit (`audit/`)** | [sync_audit.py](../src/synccraft/audit/sync_audit.py) | World-AABB broadphase + exact 15-axis SAT + `P1-02` physical contact-patch (`>= 1e-4` area) & `authored` profile auditor. **Complete.** |
| | [offscreen_renderer.py](../src/synccraft/audit/offscreen_renderer.py) | Deterministic CPU 4-view orthographic painter (`render_four_views`) + Z-buffered ray-OBB 3D textured/lit world renderer (`render_world_observation`). **Complete.** |
| **Surface (`surface/`)** | [contracts.py](../src/synccraft/surface/contracts.py) | `MaterialProgram`, `MaterialSpec`, `PersistentSurfaceState`, `OpticalFields`, `StyleConfig`, `WorldState`, 3-tier `fidelity_label` tags, and cited physical constants. **Complete.** |
| | [domain.py](../src/synccraft/surface/domain.py) | Discretized 3D box-surface manifold cells + 3 distinct topology graphs (`manifold_seams` with `weld_group` gating, `posed_contacts`, nested-mirror `symmetry_pairs`) + conservative graph Laplacian. **Complete.** |
| | [processes.py](../src/synccraft/surface/processes.py) | 10 material generators, CFL-substepped Gray–Scott Pearson regimes, moisture/thermal diffusion, Archard frictional wear, selective bare-metal oxidation, and biofilm growth. **Complete.** |
| **Shading (`shading/`)** | [uv_packer.py](../src/synccraft/shading/uv_packer.py) | Deterministic integer padded per-face atlas allocator (`allocate_uv_layout`) with zero face overlap. **Complete.** |
| | [compiler.py](../src/synccraft/shading/compiler.py) | Exact area-weighted `downsample_optical_fields`, linear-space Cook–Torrance GGX (white-furnace `<= 1.0`, reciprocity), Airy thin-film analytic extrema, and OKLab/OKLCH ramp quantizer (`shade_asset`). **Complete.** |
| | [cavity_shader.py](../src/synccraft/shading/cavity_shader.py) | Diagnostic flat-color OKLCH->sRGB atlas builder (`build_atlas`) for unshaded geometry checks. **Complete.** |
| **World (`world/`)** | [transitions.py](../src/synccraft/world/transitions.py) | Pure causal transition operator `step_world(S_t, a_t, E_t, dt)` over 12 typed action classes including `ExecuteBehaviorClipAction` and `ActivateBehaviorStateAction`. **Complete.** |
| | [inverse_inference.py](../src/synccraft/world/inverse_inference.py) | Dual-track license gate (`clean_room_permissive` vs `research_nc` `CC-BY-NC-3.0`/`4.0`, rejecting `CC-BY-ND` and `NoAI`/`do-not-train` notices, `NOASSERTION` default for unlabeled inputs), animated `.bbmodel`, Java `.json`, and left-handed X-reflected Bedrock `.geo.json` + zero-`eval` Molang/Catmull-Rom `.animation.json` decompilers, and top-`K` inverse `MaterialProgram` estimator. **Complete.** |
| | [artist_loop.py](../src/synccraft/world/artist_loop.py) | 5-pillar quantitative artist project critique, zero-vertex-drift flat-rig articulation, iterative rule-verified enhancement loop, 5-stage progressive from-scratch trajectory builder, and roundtrip re-audited artist export (`export_artist_project_bundle`). **Complete (`ADR 0006`).** |
| | [interventions.py](../src/synccraft/world/interventions.py) | 7-scenario causal physical invariance suite (UV-repack pixel invariance, rest-albedo repose invariance, Archard wear -> selective rust, Turing scale stability). **Complete.** |
| | [trajectory_dataset.py](../src/synccraft/world/trajectory_dataset.py) | 7-Level World Capability Ladder (`read_structure` through `human_transfer`) with rendered multi-view observations, parametric consistency probes, genuine `evaluate_world_capability_benchmark(records, model_completions)` evaluator, and explicit `simulator_oracle_upper_bound`, `constant_prior_baseline`, and `nearest_neighbor_train_retrieval`. **Complete.** |
| | [exemplars.py](../src/synccraft/world/exemplars.py) | Canonical stateful world-model exemplars across geometry/material archetypes. **Complete.** |
| **Data Gen (`data_gen/`)** | [static_families.py](../src/synccraft/data_gen/static_families.py) + [multi_domain_taxonomy.py](../src/synccraft/data_gen/multi_domain_taxonomy.py) + [dragon_and_fauna_expansion.py](../src/synccraft/data_gen/dragon_and_fauna_expansion.py) + [industrial_and_vehicle_expansion.py](../src/synccraft/data_gen/industrial_and_vehicle_expansion.py) | `55` core procedural families (`40` seen + `15` held-out unseen `>=16` boxes, `>=35` crossbow tree shapes) across 7 macro-domains, zero split-name prompt leakage, and 3 holdout splits (`test_unseen_params`, `test_unseen`, `test_compositional`). **Complete (`ADR 0005`).** |
| | [complex_entity_and_asset_pack.py](../src/synccraft/data_gen/complex_entity_and_asset_pack.py) | `10` neutral clean-room Apex families (`8` seen + `2` held-out unseen `>=40` boxes; `elemental_companion_critter`, `modular_botanical_healing_station`, etc.) with nested `mirror_x`, `@muscles`, `@kinematics`, and `@behavior`. **Complete (`ADR 0006`).** |
| | [ik_motion_templates.py](../src/synccraft/data_gen/ik_motion_templates.py) | Analytic hinge + associative-$SO(3)$ Levenberg–Marquardt 2-bone IK & CCD chain solvers with `joint_limits` clamping. **Complete.** |
| | [curriculum_and_assistant.py](../src/synccraft/data_gen/curriculum_and_assistant.py) + [diversity_engine.py](../src/synccraft/data_gen/diversity_engine.py) + [batch_pipeline.py](../src/synccraft/data_gen/batch_pipeline.py) + [taxonomy_catalog.py](../src/synccraft/data_gen/taxonomy_catalog.py) | 5-tier complexity curriculum, comment-free `@behavior` AST completions, pinned specification prompts, `distinct_tree_shapes_by_family`, and batch SPC QC. **Complete.** |
| **Backends & CLI** | [bbmodel5_exporter.py](../src/synccraft/backends/bbmodel5_exporter.py), [java_exporter.py](../src/synccraft/backends/java_exporter.py), [labpbr_exporter.py](../src/synccraft/backends/labpbr_exporter.py), [gltf_exporter.py](../src/synccraft/backends/gltf_exporter.py), [common.py](../src/synccraft/backends/common.py), [cli.py](../src/synccraft/cli.py) | Blockbench 5 `.bbmodel`, Java modern/legacy `.json`, LabPBR 1.3 `_n`/`_s` PNGs, glTF 2.0 PBR (`ORM` + `KHR_materials_iridescence`), and deterministic CLI. **Complete (CPU verified; live GUI pending `P0-02`).** |
| | [bedrock_gecko_exporter.py](../src/synccraft/backends/bedrock_gecko_exporter.py) | Outbound Bedrock `1.12.0` `.geo.json` + `1.8.0` `.animation.json` (`export_bedrock`) and GeckoLib (`export_geckolib`) with exact left-handed X-reflection and post-serialization 8-corner world vertex re-audit (`< 1e-4`, `runtime_import_verified=False`). **Complete (CPU verified; live client pending `P3-02`).** |
| **Evaluation & Modeling** | [budget.py](../src/synccraft/evaluation/budget.py), [gate_g2.py](../src/synccraft/evaluation/gate_g2.py), `modeling/*.py`, [notebooks/synccraft_end_to_end.ipynb](../notebooks/synccraft_end_to_end.ipynb) | Monotonic `CostLedger`, artifact-derived Gate G2 + `verify_prompt_adherence`, `run_bounded_repair_policy` (`<= 3` re-audits), and standalone Colab G4 BF16 LoRA (`max_seq_len=8192`) training + evaluation notebook. **CPU verified; live GPU run & Gate G2 empirical unlock pending (`P1-09`/`P1-10`).** |

---

## 3. Exact Mathematical, Geometric, Optical & Biomechanical Formulations

### 3.1 Block-DSL v3.2 Typed AST & Zero-`eval` Security Contract
Every textual Block-DSL program is tokenized and parsed by a hand-written bounded recursive-descent parser ([parser.py](../src/synccraft/dsl/parser.py)) with verified Lark grammar parity ([grammar.lark](../src/synccraft/dsl/grammar.lark)), enforcing hard resource ceilings before AST allocation:
- UTF-8 byte ceiling: `2,097,152` bytes (`2 MB`)
- Lexer token ceiling: `200,000` tokens
- AST declaration ceiling: `4,096` declarations; maximum bone nesting depth: `64`
- Pre-expansion nested mirror ceiling: `MAX_EXPANDED_MIRROR_NODES = 8,192` (rejects $2^k$ mirror bombs such as 15-level nested mirrors = `32,768` nodes in $O(N)$ AST walk prior to cloning).
- Zero calls to `eval()`, `exec()`, `compile()`, or dynamic `importlib` exist anywhere in `src/synccraft/`.

### 3.2 Shared $SE(3)$ Chain & Arbitrary-Depth $O(3)$ Reflection Parity
Every bone $b$ has world pivot $\mathbf{p}_b \in \mathbb{R}^3$ and proper rotation $\mathbf{R}_b \in SO(3)$ ($\det \mathbf{R}_b = +1$). Intrinsic XYZ Euler angles $(\phi_x, \phi_y, \phi_z)$ in degrees map at the compiler boundary to:
$$\mathbf{R}_{\text{local}}(\phi_x, \phi_y, \phi_z) = \mathbf{R}_z(\phi_z)\,\mathbf{R}_y(\phi_y)\,\mathbf{R}_x(\phi_x), \qquad \mathbf{R}_b = \mathbf{R}_{\text{parent}(b)}\,\mathbf{R}_{\text{local}}.$$
For an attached child bone (`attach(parent_box=P, anchor=a, align=g, offset=o)`):
$$\mathbf{p}_b = \mathbf{c}_P + \mathbf{R}_{\text{parent}(b)}\left(\frac{1}{2}\mathbf{s}_P \odot \mathbf{a} + \mathbf{o}\right),$$
and the first direct box $B_0$ in bone $b$ centers at $\mathbf{c}_{B_0} = \mathbf{p}_b + \mathbf{R}_b\left(-\frac{1}{2}\mathbf{s}_{B_0} \odot \mathbf{g}\right)$.

**Arbitrary-Depth Nested `mirror_x` Reflection (`ADR 0006`)**:
Let $\mathbf{M}_x = \operatorname{diag}(-1, 1, 1) \in O(3) \setminus SO(3)$ ($\det \mathbf{M}_x = -1$). Reflecting a bone frame across the world $X=0$ plane conjugates its rotation so the bone frame itself remains right-handed ($\det \mathbf{R}' = (-1)^2 (+1) = +1$):
$$\mathbf{p}'_b = \mathbf{M}_x \mathbf{p}_b, \qquad (\phi'_x, \phi'_y, \phi'_z) = (\phi_x, -\phi_y, -\phi_z), \qquad (a'_x, g'_x, o'_x) = (-a_x, -g_x, -o_x).$$
When a `mirror_x` bone $b_2$ is nested inside an outer `mirror_x` subtree $b_1$ (e.g., symmetric inner/outer wing phalanges `wing_digit_inner_1` -> `wing_digit_outer_1` inside `wing_arm_l` -> `wing_arm_r`), reflection parity composes in $\mathbb{Z}_2$:
$$\text{mirrored}(n) = \text{mirrored}(\text{parent}) \oplus \mathbb{I}[\text{is\_mirror\_clone}(n)].$$
At depth 1 (`arm_r__phalanx`), `mirrored = True` and `east <-> west` cullfaces swap; at depth 2 (`arm_r__digit_out__phalanx`), two reflections compose to a pure proper rotation ($\mathbf{M}_x^2 = \mathbf{I}_3$), restoring `mirrored = False` and the original `east` cullface! Semantic `Locator` offsets reflect identically ($\ell'_x = -\ell_x$).

### 3.3 Exact 15-Axis OBB SAT Audit & P1-02 Physical Contact Policy
For any non-exempt box pair $(A, B)$ with centers $\mathbf{c}_A, \mathbf{c}_B$, axes $\mathbf{u}_{A,i}, \mathbf{u}_{B,j}$, integer sizes $\mathbf{s}_A, \mathbf{s}_B \in \mathbb{Z}_{>0}^3$, and continuous inflations $\delta_A, \delta_B \ge 0$, effective half-extents are:
$$\mathbf{h}_A = \frac{1}{2}\mathbf{s}_A + \delta_A\mathbf{1}_3, \qquad \mathbf{h}_B = \frac{1}{2}\mathbf{s}_B + \delta_B\mathbf{1}_3.$$
Note that continuous $\delta_A$ (`inflate` + muscle `bulge_inflate`) modifies physical OBB clearance while leaving $\mathbf{s}_A \in \mathbb{Z}_{>0}^3$ strictly untouched, guaranteeing zero Box-UV texture layout invalidation.
Across all 15 candidate separating axes $\mathcal{L} = \{\mathbf{u}_{A,i}\}_{i=1}^3 \cup \{\mathbf{u}_{B,j}\}_{j=1}^3 \cup \{\mathbf{u}_{A,i} \times \mathbf{u}_{B,j}\}_{i,j=1}^3$ (masking near-parallel cross products $\|\mathbf{u}_{A,i} \times \mathbf{u}_{B,j}\|_2 \le 10^{-8}$):
$$\omega(\hat{\mathbf{n}}) = \sum_{i=1}^3 h_{A,i}|\mathbf{u}_{A,i}\cdot\hat{\mathbf{n}}| + \sum_{j=1}^3 h_{B,j}|\mathbf{u}_{B,j}\cdot\hat{\mathbf{n}}| - |(\mathbf{c}_B - \mathbf{c}_A)\cdot\hat{\mathbf{n}}|.$$
- **Non-exempt collision**: $\min_{\hat{\mathbf{n}} \in \mathcal{L}} \omega(\hat{\mathbf{n}}) > \epsilon_{\text{tol}}$ (`collision_tolerance = 1e-7`).
- **P1-02 Structural Contact Contract (`enforce_physical_contact=True`)**: Blanket parent/sibling collision immunity is forbidden. Only explicitly declared `attach(detached=false)` and `place(base_box=...)` pairs receive contact exemptions, and each such pair MUST satisfy $-10^{-4} \le d_{\text{SAT}}(A, B) \le \max(0.50, \delta_{\text{hinge}}(\theta)) + \delta_A + \delta_B + \epsilon_{\text{tol}}$ (where $\delta_{\text{hinge}}(\theta) = \frac{\sqrt{3}}{2} t_{\min} \sin\theta$ for acute hinge angles $0 < \theta \le 60^\circ$, with neither box center buried $> 0.50\text{u}$ inside its partner) — rejecting both floating detached gaps ($d_{\text{SAT}} < -10^{-4}$) and deep co-penetrations. Furthermore, the physical contact graph must not exceed the number of independent box-bearing root/authorized-detached bone subtrees (walking across any intermediate locator-only socket hubs so floating boxes cannot hide behind zero-box bones).

### 3.4 Rest-Space Manifold Surface Domain, Gray–Scott Reaction–Diffusion & Tribology
`build_surface_domain` discretizes all 6 faces of every box at resolution $r$ cells/unit in **rest pose**, constructing three distinct sparse graphs:
1. `manifold_seams`: Connects boundary cells of boxes sharing the same bone OR sharing an explicit `weld_group` tag across child joints when rest-space 3D cell centers lie within tolerance.
2. `posed_contacts`: Rebuilt dynamically at each world pose $q_t$ to capture transient rubbing/moisture/heat transfer across distinct colliding/touching parts.
3. `symmetry_pairs`: Maps each cell in a `mirror_x` source subtree (including nested depths) to its reflected counterpart.

For conservative scalar diffusion over variable-area cells $A_i$, symmetric conductance $G_{ij} = G_{ji}$ preserves total mass $\sum_i A_i \phi_i$:
$$(\Delta_{\mathcal{M}}\phi)_i = \frac{1}{A_i}\sum_{j \in \mathcal{N}(i)} G_{ij}(\phi_j - \phi_i).$$
Biological skin/coat patterns (`patterned_skin`) evolve two-species Gray–Scott reaction–diffusion under explicit CFL substepping ($\Delta\tau \le \frac{0.20}{\max(D_u, D_v)\lambda_{\max}}$), while `lichen_biofilm` models moisture- and upward-exposure-gated surface colonization:
$$\frac{\partial u}{\partial t} = D_u \Delta_{\mathcal{M}} u - u v^2 + F(1 - u), \qquad \frac{\partial v}{\partial t} = D_v \Delta_{\mathcal{M}} v + u v^2 - (F + k)v.$$
During skeletal motion or `RubContactAction`, Archard frictional wear accumulates $\Delta w_i = k_w P_i \|\Delta\mathbf{x}_i\|$, stripping protective coating $c_i$ and exposing bare metal $m_i(1 - c_i)$ to moisture-accelerated oxidation $\frac{do_i}{dt} = k_{\text{ox}}\,m_i(1 - c_i)\,\text{moisture}_i\,(1 - o_i)$.

### 3.5 Multi-Target Pixel-PBR Compiler & 3D Ray-OBB Renderer
High-resolution manifold `OpticalFields` (`base_color`, `normal_ts`, `roughness`, `metalness`, `f0_dielectric`, `porosity`, `subsurface`, `emission`, `ao`, `film_thickness_nm`) are area-integrated onto target UV texels via exact interval overlap (`downsample_optical_fields`), then compiled in linear light (`alpha = roughness^2` GGX microfacet + Airy thin-film phase shift $\delta = \frac{4\pi n_{\text{film}} d \cos\theta_t}{\lambda}$) into three separated `@style.target_encoding` targets:
- `vanilla`: Controlled directional + cavity baked sRGB PNG atlas with optional OKLCH palette ramp quantization.
- `labpbr_1_3`: Unbaked albedo PNG + `_n.png` (DirectX-convention tangent normal XY, AO in B, height in A) + `_s.png` (perceptual smoothness $1 - \sqrt{\alpha}$ in R, $F_0$ / metal code `230..255` in G, porosity `0..64` / subsurface `65..255` in B, emissive `0..254` with `255` never-emissive sentinel in A).
- `gltf_2_0`: glTF 2.0 `.gltf` scene + baseColor, tangent normal, packed `ORM` (Occlusion/Roughness/Metallic), emissive, and `KHR_materials_iridescence` (`iridescenceThicknessMinimum/Maximum` in nm).

### 3.6 Reduced-Order Hill Musculoskeletal Mechanics, Tendon Pantographs & Isochoric Bulge (`ADR 0006`)
To bridge `MASS` (Lee et al., SIGGRAPH 2019) and `MyoSuite` (Caggiano et al., 2022) with lightweight block-rig authoring without requiring artists to tune stiff forward multi-body integrators, `evaluate_muscles_at_activations` solves a closed-form **quasi-static Hill-type agonist–antagonist equilibrium** independently per joint axis $(b, \alpha) \in \text{Bone} \times \{x, y, z\}$:
1. **Per-Axis Agonist–Antagonist Equilibrium, Co-Contraction Stiffness & Joint Stops**:
   Given muscle activations $a_m \in [0, 1]$, signed moment-arm excursion capacities $\theta_m^{\max}$ (`moment_arm_deg`), and activation-modulated passive joint-capsule stiffness $k_{\text{passive}} = \max\!\big(0,\; 1 - \max_{m \in \mathcal{M}_{b,\alpha}} a_m\big)$:
   $$\Delta\theta_{b,\alpha} = \operatorname{clip}_{[\theta_{b,\alpha}^{\min}, \theta_{b,\alpha}^{\max}]}\!\left(\Delta\theta_{b,\alpha}^{\text{base}} + \frac{\sum_{m \in \mathcal{M}_{b,\alpha}} a_m \theta_m^{\max}}{k_{\text{passive}} + \sum_{m \in \mathcal{M}_{b,\alpha}} a_m}\right), \qquad K_{b,\alpha} = k_{\text{passive}} + \sum_{m \in \mathcal{M}_{b,\alpha}} a_m.$$
   - *Orthogonal axis decoupling*: Grouping strictly by $(b, \alpha)$ ensures yaw muscles on $y$ never dilute pitch excursion on $x$.
   - *Single agonist ($a_1 = 0.85, a_2 = 0$)*: $K_{b,\alpha} = 0.15 + 0.85 = 1.0$, yielding $\Delta\theta = 0.85\,\theta_1^{\max}$ (clamped to `joint_limits`).
   - *Symmetric isometric co-contraction ($a_1 = a_2 = 0.85, \theta_2^{\max} = -\theta_1^{\max}$)*: Net torque cancels ($\Delta\theta = 0$), joint impedance surges to $K_{b,\alpha} = 0.15 + 1.70 = 1.85$, and **both** agonist and antagonist bellies bulge simultaneously!
2. **Biarticular Passive Tendon Pantograph (`TendonCoupling`)**:
   Distal joints (avian wing carpals/digits, digitigrade tarsals) couple mechanically to the **clamped** primary joint excursion $\Delta\theta_{b, \alpha}$ (and are themselves clamped to the target bone's `joint_limits`, preventing passive tendon overflow past anatomical stops):
   $$\Delta\theta_{b_{\text{coupled}}, \beta} \leftarrow \operatorname{clip}_{[\theta_{\min}, \theta_{\max}]}\!\big(\Delta\theta_{b_{\text{coupled}}, \beta} + \rho_{\text{tendon}} \cdot w_m \cdot \Delta\theta_{b, \alpha}\big).$$
3. **3D SE(3) Fiber Strain & Box-UV-Safe Isochoric Belly Bulge**:
   Comparing world-space origin-to-insertion anchor distance $L(q)$ against rest length $L_0 = L(0)$ yields normalized length $\tilde{L} = L(q)/L_0$ and shortening gain $g(\tilde{L}) = \operatorname{clip}(0.68 + 0.32\,\operatorname{clip}((1 - \tilde{L})/0.25, -0.5, 1.0), 0.35, 1.0)$. The associated `belly_box` expands radially by:
   $$\delta_{\text{bulge}}(m) = \min\big(0.35,\; a_m \cdot \delta_{\max}(m) \cdot g(\tilde{L})\big),$$
   routed exclusively through `box_inflate_deltas` (and re-evaluated on the post-IK pose during `@behavior` clips with `ik_drives`) so integer UV dimensions remain untouched.

### 3.7 World-Anchored $C^1$ Stance-Foot IK & Swept-Pose Collision Auditing
Standard sinusoidal forward-kinematics walk cycles slide feet horizontally across the ground. In `evaluate_behavior_clip`, every `ik_drive` (`foot_cycle(stride=S, lift=H, phase_deg=phi0)`) evaluates raw phase $\tilde{\psi} = \phi + \frac{\phi_0}{360}$ ($k = \lfloor\tilde{\psi}\rfloor$, $u = \tilde{\psi} - k \in [0, 1)$, duty factor $\eta = 0.60$) around the rest-pose world effector $\mathbf{e}_0 = (x_0, y_0, z_0)$ while translating the body root forward by $\Delta\mathbf{r}(\phi) = \big(0,\; -0.10\,H_{\max}(0.5 - 0.5\cos 4\pi\phi),\; \phi\,\bar{S}\big)$:
- **Stance ($u \in [0, 0.60)$)**: The foot is pinned to a fixed world ground coordinate $\mathbf{p}_{\text{stance}}(k) = \mathbf{e}_0 + \big(k - \frac{\phi_0}{360} + 0.30\big)S\,\hat{\mathbf{z}}$ while the root advances over it — yielding **zero target horizontal slip** across frames sharing step index $k$!
- **Swing ($u \in [0.60, 1.0)$)**: Smooth $C^1$ raised-cosine progression $\sigma(s) = \frac{1}{2} - \frac{1}{2}\cos(\pi s)$ ($s = \frac{u - 0.60}{0.40}$) advances the foot to the next foothold with vertical clearance $y(s) = y_0 + H \sin^2(\pi s)$.
- **`solve_two_bone_ik` Acceleration**: Inner Levenberg–Marquardt iterations precompute exact associative $SO(3)$ parent/rest rotation matrices (`r_up_0`, `r_lo_local_rest`) and clamp steps to `joint_limits`, achieving a **~200x speedup** on 64-box rigs while running a final full `compile_posed_geometry` oracle verification.
- **`audit_behavior_clip`**: Audits $N$ uniformly sampled frames across the clip for non-exempt 15-axis OBB collisions (`collision_tolerance = 1e-7`, `enforce_physical_contact=False` on dynamic posed frames while `enforce_physical_contact=True` is enforced at rest) and enforces $\max_{\text{stance}} \|\mathbf{e}_{\text{horizontal}}(t_{n+1}) - \mathbf{e}_{\text{horizontal}}(t_n)\|_2 \le 0.02$ model units.

---

## 4. Complex Entities, Quarantine Governance & Closed-Loop Artist-Assistant Capabilities (`ADR 0006`)

### 4.1 Clean-Room Biomechanical Entities & Morphing Mechanisms (`32–68` Boxes)
[complex_entity_and_asset_pack.py](../src/synccraft/data_gen/complex_entity_and_asset_pack.py) implements **10 neutral clean-room Apex Complex families** (`8` Seen + `2` Held-Out Unseen $\ge 40$ boxes), each verified across multiple random seeds for rest SAT, P1-02 physical contact, multi-phase swept clip SAT, and `< 0.02u` stance-foot slip:

| Family ID | Split | Boxes | Bones | Locators | Key Biomechanical / Structural Features |
|---|---|---:|---:|---:|---|
| `apex_flying_wyvern` | Seen | 56 | 25 | 6 | 2-level nested `mirror_x` wings (carpal hub + 4 symmetric phalanges/membranes per wing), digitigrade IK talons, pectoralis muscles + wing-fold pantograph, segmented flame-sac tail |
| `brute_theropod_wyvern` | Seen | 34 | 16 | 6 | Massive jaw/cranial battering horn, muscular gastrocnemius/quadriceps hindlegs with `belly_box` bulge, counterbalanced clubbed tail, bipedal stomp gait |
| `pelagic_leviathan_wyvern` | Seen | 39 | 22 | 8 | Travelling-wave undulating spine/neck/fluke (`phase_deg` cascade), flaring dorsal conductive sails, webbed pectoral fins, hydro-vortex locators |
| `fanged_thunder_wyvern` | Seen | 36 | 19 | 7 | Quadruped muscular forelimbs, articulating fulgur-capacitor dorsal plates (`fulgur_overcharge_state` emissive transition), Diagonal trot IK |
| `armored_arthropod_colossus` | Seen | 38 | 19 | 9 | Multi-leg alternating tripod gait, raptorial crushing chelicerae with flexor muscles, segmented carapace |
| `morphing_switch_blade_axe` | Seen | 32 | 9 | 2 | Mechanical phial cam-latch weapon (`axe_to_sword_morph_cycle`) with synchronized latch/blade transformation + overheated phial state |
| `elemental_companion_critter` | Seen | 35 | 18 | 10 | Stylized companion creature rig (`rider_seat`, `camera_focus`, `hitbox`, `socket`, `vfx_emitter` locators, ear-perk muscles, Turing coat) |
| `modular_botanical_healing_station` | Seen | 33 | 8 | 4 | Interactive modular healing/scanner block (`block_entity`) with articulated cradle trays, scanner arm, and emissive status visor |
| `elder_calamity_dragon` | **Unseen ($\ge 40$)** | 64 | 29 | 8 | True hexapod (4 terrestrial IK legs + 2 nested-mirror `mirror_x` wings = 6 limbs), cranial crown horns, elder aether state |
| `relic_automaton_guardian` | **Unseen ($\ge 40$)** | 49 | 17 | 5 | Ancient mechanical construct with twin shoulder turbines, articulated core shutters, piston-coupled greaves, heavy guardian march IK |

### 4.2 Dual-Track `CC-BY-NC` & `NoAI` Quarantine Governance
External community creature/block repositories often release assets under **`CC-BY-NC-3.0` / `CC-BY-NC-4.0`**. SyncCraft enforces a hard programmatic quarantine in [inverse_inference.py](../src/synccraft/world/inverse_inference.py) (`verify_asset_provenance`, `build_quarantined_research_nc_dataset`) and [configs/data_policy.yaml](../configs/data_policy.yaml):
1. **`clean_room_permissive` (Default Track — `Apache-2.0` Compatible)**:
   - Accepts **only** `{"CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-3-Clause"}` with zero `NoAI` / `do not train` creator notices.
   - Sets `commercial_weights_eligible = True`. Unlabeled local workspace files default to `license_spdx = "NOASSERTION"` (`commercial_weights_eligible = False`), never `CC-BY-4.0`.
2. **`research_nc` (Explicit Non-Commercial Research Quarantine Track & Dataset Builder)**:
   - Additionally accepts `{"CC-BY-NC-3.0", "CC-BY-NC-SA-3.0", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}` **only** when `attribution_notice` is non-empty and no `NoAI` / `do not train` opt-out is present (while rejecting `CC-BY-ND-*` NoDerivatives licenses).
   - Permanently stamps `license_track = "research_nc"` and `commercial_weights_eligible = False` under `research_nc/...` splits so non-commercial community assets never contaminate `Apache-2.0` weights.
3. **Unconditional Mojang / Proprietary Ban**:
   - Any provenance containing Mojang/vanilla extraction or proprietary markers raises `LicenseGateError` under **all** policies including `research_nc`.

### 4.3 Multi-Format SE(3) Geometry (Exact Bedrock Left-Handed X-Reflection) & Zero-`eval` Molang Decompilers
- **Multi-Format `.bbmodel`, Java `.json`, & Bedrock `.geo.json` Decompilers (`_decompile_bbmodel`, `_decompile_java_json`, `_decompile_bedrock_geojson`)**: Apply genuine Bedrock left-handed X-reflection (`-(origin_x + size_x)`, negated pivot/locator X, negated X/Y rest & keyframe rotations, negated X position), preserve fractional `size_step`, `0`-thickness `plane` quads, per-cube `rotate`, and 3D `locators` (`< 1e-4` 8-corner fidelity).
- **Zero-`eval` Balanced-Paren Molang, Catmull-Rom & Harmonic-Phase `.animation.json` Decompiler (`decompile_bedrock_animation_to_behavior`)**: Parses nested-parenthesis Molang arithmetic, affine-on-trig wrappers, `q.life_time`, ternaries, `math.abs`, `math.clamp`, Catmull-Rom `post` keyframes, GeckoLib `{"vector": [...]}`, and timestamped Molang dicts without calling `eval`.

### 4.4 Closed-Loop Artist-Assistant Workflow & Roundtrip Export (`artist_loop.py`, `bedrock_gecko_exporter.py`)
1. **`analyze_artist_project`**: Ingests `Block-DSL`, `.bbmodel`, Java `.json`, or Bedrock `.geo.json` (+ `.animation.json`) and evaluates **5 weighted quantitative pillars** (`0.32` `geometry_contact`, `0.18` `rig_hierarchy`, `0.18` `surface_material`, `0.16` `musculoskeletal`, `0.16` `behavior_motion`):
   - *Geometry & Contact Integrity*: Non-exempt SAT overlaps, floating gaps, deep co-penetrations (`>0.5u`), detached physical components.
   - *Rig & Socket Ergonomics*: Monolithic multi-box entity bones needing articulation splits (`articulate_flat_rig`), missing `joint_limits`, and missing semantic `Locator` sockets (`rider_seat`/`camera_focus`/`vfx_emitter`).
   - *Surface & Material Realism*: Palettes lacking `@materials` process bindings, unwelded organic skin/scale materials lacking `weld_group`, and missing `@style` contracts.
   - *Musculoskeletal & IK Readiness*: Articulated rigs lacking `@muscles` Hill actuators or `@kinematics` `ik_chain`/`look_chain` descriptors.
   - *Behavior & Stance Grounding*: Animated target profiles lacking `@behavior` clips, or clips failing swept-SAT / `<0.02u` stance-slip audits.
2. **`run_looped_artist_enhancement`**: Iterates through ranked actionable suggestions in priority order without stalling on saturated ops — executing rest-contact repair, **zero-vertex-drift flat-rig articulation** (`_articulate_flat_hierarchy`, placing anatomical joint pivots directly on contacting OBB faces while preserving all 8 box corners to `< 10^{-12}`), `joint_limits` + `Locator` injection, scientific `@materials` + `weld_group` synthesis, and SAT-verified `@muscles` + `@kinematics` + `@behavior` synthesis.
3. **`build_progressive_from_scratch_trajectory`**: Constructs any complex entity from scratch through 5 rule-verified artist stages (`Stage 0: Rough Primary Mass Blockout` -> `Stage 1: Articulated Skeletal Hierarchy & Nested Mirrors` -> `Stage 2: Secondary Anatomy, Muscle Bellies & Semantic Locators` -> `Stage 3: Scientific Welded Surface Materials & Turing Patterns` -> `Stage 4: Musculoskeletal Tendon Couplings, Closed-Loop Stance IK & @behavior`), certifying 100% SAT + P1-02 compliance at every intermediate stage.
4. **`export_artist_project_bundle` (`export_bedrock`, `export_geckolib`, animated `export_bbmodel`)**: Closes the artist roundtrip loop by exporting enhanced `Block-DSL v3.1` programs back into Bedrock `1.12.0` `.geo.json` + `1.8.0` `.animation.json`, GeckoLib, or animated Blockbench 5.0 `.bbmodel` (baking Hill `@muscles`, tendon pantographs, and stance-IK poses into loop-closed keyframes or `"molang"` channels) and immediately re-ingesting the serialized JSON to verify `< 10^{-8}` 8-corner world vertex agreement (`runtime_import_verified = False`).

---

## 5. Dataset Taxonomy, Unbiased Multi-Stream Sampling & Provenance Governance

SyncCraft's public Hugging Face dataset (`TassP0lar/synccraft-v3-blockdsl-dataset`, `Apache-2.0`) and [notebooks/synccraft_end_to_end.ipynb](../notebooks/synccraft_end_to_end.ipynb) (`max_sequence_length = 8192`, Hub-first SHA-256 shard verification on Colab GPU, `StreamMixtureSampler` iterating 100% of `main` rows without replacement and capping `apex`/`world` auxiliary repeats at `<= 3`) comprise three cryptographically hashed (`SHA-256`) clean-room streams:

1. **Core Categorized 55-Family Assistant Corpus (`bakeoff_8k` & `full_75k`)**:
   - `40` Seen families + `15` Held-Out Unseen families (`>= 16` boxes, `>= 35` distinct crossbow tree shapes), zero split-name prompt leakage, comment-free `@behavior` AST completions, `distinct_tree_shapes_by_family`, and 3 evaluation holdout splits (`test_unseen_params`, `test_unseen`, `test_compositional`).
2. **7-Level World Capability Ladder (`world_capability_ladder.jsonl`)**:
   - Covers `level_1_read_structure`, `level_2_predict_pose_collision`, `level_3_predict_motion_gait`, `level_4_predict_surface`, `level_5_invert_observation` (with rendered 4-view images for VLM inputs), `level_6_edit_and_repair`, and `level_7_human_transfer` (`audit_profile="authored"`), evaluated strictly from `model_completions` alongside `simulator_oracle_upper_bound`, `constant_prior_baseline`, and `nearest_neighbor_train_retrieval`.
3. **Clean-Room Apex Musculoskeletal & Artist-Loop Expansion (`apex_complex_v3_2`)**:
   - Covers the `10` neutral clean-room Apex families (`8` Seen + `2` Held-Out Unseen `>= 40` boxes: `elder_calamity_dragon` & `relic_automaton_guardian`) across full masterwork, progressive stage upgrade, and closed-loop artist enhancement turns.

---

## 6. Executed Verification Matrix & Quantitative Telemetry

Every test suite below executes locally on CPU (`Python 3.12`, `arm64 macOS`) via `python scripts/check.py` with zero mocks or skipped tests:

| Test Suite | Key Quantitative Invariants Proven |
|---|---|
| [test_musculoskeletal_complex_and_artist_loop.py](../tests/test_musculoskeletal_complex_and_artist_loop.py) | `Block-DSL v3.2` `AST <-> DSL` roundtrip; nested `mirror_x` $\det(\mathbf{R}) = +1$; `8,192`-node mirror-bomb guard; Zajac/Hill active force-length + passive parallel-elastic equilibrium (`fidelity_label == "reduced_order_qualitative"`), agonist–antagonist stiffness, `belly_box` bulge; `CC-BY-NC-3.0`/`4.0` + `NoAI` quarantine; all `10` clean-room Apex families pass rest SAT + P1-02 + swept SAT + `<0.02u` stance slip |
| [test_world_interventions_and_inference.py](../tests/test_world_interventions_and_inference.py) | `7/7` causal physical interventions; 7-Level World Capability Ladder + genuine `evaluate_world_capability_benchmark` scoring real `model_completions` against `simulator_oracle_upper_bound`, `constant_prior_baseline`, and `nearest_neighbor_train_retrieval` |
| [test_surface_and_pbr.py](../tests/test_surface_and_pbr.py) | Conservative Laplacian mass conservation (`<1e-10` drift), GGX white-furnace (`<= 1.0`) + reciprocity, Airy thin-film extrema, Gray–Scott Pearson regimes, LabPBR 1.3 `_n`/`_s` channel encoding, glTF 2.0 `ORM` + `KHR_materials_iridescence` export |
| [test_data_gen_and_contact.py](../tests/test_data_gen_and_contact.py) | `P1-02` contact-patch policy, zero split-name leakage across all prompts, zero `// @animation_plan` comment claims, `>= 35` distinct `mechanical_crossbow` tree shapes, 3 holdout splits (`test_unseen_params`, `test_unseen`, `test_compositional`) |
| [test_adversarial_challenger.py](../tests/test_adversarial_challenger.py) & [test_adversarial_stress.py](../tests/test_adversarial_stress.py) | Adversarial parser bounds, Unicode/NaN/Inf injection, degenerate OBB cross-axis SAT stability, smooth-vs-direct SAT counterexamples, monotonic `CostLedger` & artifact-derived Gate G2 integrity |
| [test_notebook_e2e.py](../tests/test_notebook_e2e.py) | Colab v3.2 notebook integrity, `StreamMixtureSampler` (`100%` `main` without replacement, `<= 3` aux repeats), VRAM-safe pre-FP32 response loss, periodic full-state checkpoint save + `resume_from_checkpoint` parity, `apply_chat_template` + `<think>` stripping, dynamic `max_gen_tokens`, smoke-upload refusal, honest `LFM Open License v1.0` vs `Apache-2.0` Model Card, `run_bounded_repair_policy` (`<= 3` re-audits) + `verify_prompt_adherence` |
| Core, PyTorch & exporter suites (`tests/test_dsl*.py`, `tests/test_kinematics*.py`, `tests/test_bedrock*.py`, `tests/test_exporters*.py`, `tests/test_budget_and_gate.py`) | Seeded AST & Lark parity roundtrips, `1,000` flush OBB SAT fixtures, NumPy vs PyTorch SE(3)/SAT cross-checks, asymmetric Bedrock left-handed X-reflection & 7-bone/9-cube authored creature roundtrip (`< 1e-4`), CLI determinism, Gate G2 blocker enforcement |

---

## 7. Self-Critical Audit: Approximations, Unimplemented Milestones & Priority Attack Vectors for External AI Review

In accordance with `AGENTS.md`, this section separates **implemented & CPU-verified behavior** from **reduced-order physical approximations** and **unexecuted external milestones**, giving an external AI reviewer an unvarnished map of where to probe deepest.

### 7.1 Reduced-Order Physical Approximations (`reduced_order_qualitative`)
1. **Quasi-Static Zajac/Hill Muscle Equilibrium vs. Full Forward Multi-Body Dynamics**:
   - *Current state*: `evaluate_muscles_at_activations` evaluates normalized Zajac/Hill active force-length plus passive parallel-elastic curves (`fidelity_label == "reduced_order_qualitative"`) and solves algebraic moment-balance per joint axis with kinematic tendon pantographs (`TendonCoupling`). It does **not** integrate second-order rigid-body Euler–Lagrange equations $\mathbf{M}(q)\ddot{q} + \mathbf{C}(q, \dot{q}) = \boldsymbol{\tau}_{\text{muscle}} + \mathbf{J}^T\boldsymbol{\lambda}_{\text{contact}}$.
2. **Isochoric Bulge Approximation on Axis-Aligned Box Primitives**:
   - *Current state*: Muscle contraction increases `box.inflate` uniformly along all 3 axes (`h_i = s_i/2 + inflate`) because Minecraft/Blockbench `inflate` is a single scalar per box (capped at `0.35u`, audited by SAT so it never causes hidden collisions).
3. **2-Bone + CCD IK Degrees of Freedom**:
   - *Current state*: `solve_two_bone_ik` optimizes 3 rotational DOFs at the proximal ball joint (`upper`) + 1 hinge DOF (`hinge_axis`) at the distal joint (`lower`), complemented by multi-bone CCD spine/tail chains. Full uneven-terrain foot-normal adaptation remains a future extension.
4. **Surface PDE Discretization (`reduced_order_qualitative`) vs. Optics (`exact_physical`)**:
   - *Current state*: Surface diffusion and Gray–Scott reaction–diffusion operate on a cell-centered rest-pose box graph (`subdivisions_per_unit = 1..4`, labeled `reduced_order_qualitative`), whereas Cook–Torrance GGX microfacet reflectance and Airy thin-film interference satisfy analytical white-furnace (`<= 1.0`), reciprocity, and extrema tests (`exact_physical`).

### 7.2 Explicitly Unimplemented / Deferred Plan Milestones
1. **Polygon Extrusion (`extrude`), `@shade`, `@ergonomics`, Legacy `@motion` (`P1-01`, `P1-04`)**:
   - Explicitly raise `UnsupportedFeature` in `parser.py` rather than silently approximating shapes.
2. **Live Bedrock / GeckoLib / Blockbench 5 / Minecraft Java Runtime Import Verification (`P0-02`, `P3-02`)**:
   - Outbound `export_bedrock`, `export_geckolib`, and inbound `_decompile_bedrock_geojson` now apply exact left-handed X-axis reflection (`-(origin_x + size_x)`, negated pivot/locator X, negated X/Y rotations, negated X position) and preserve fractional `size_step`, `0`-thickness `plane` quads, and per-cube `rotate` (`< 1e-4` 8-corner world vertex parity), while `runtime_import_verified` remains explicitly stamped `False` until tested inside pinned live GUI/game clients.
3. **Live Tokenizer XGrammar Masking & KV-Cache Statement Rollback (`P1-07`)**:
   - `grammar.lark` maintains verified parity with `parser.py` v3.2, while `stateful_decoder.py` remains an explicit `NotImplementedError` stub until wired to a live GPU tokenizer and KV-cache checkpointing runtime.
4. **Live Colab G4 RTX PRO 6000 GPU Bake-Off, Training & Empirical Gate G2 Unlock (`P0-03`, `P0-04`, `P1-09`, `P1-10`)**:
   - Shared PyTorch `SE(3)`/SAT, Baseline-PGD, `run_bounded_repair_policy` (`<= 3` re-audits), `verify_prompt_adherence`, and the 7-Level World Capability Benchmark are implemented and verified on CPU, while live GPU execution (`LiquidAI/LFM3-VL-2B` under `LFM Open License v1.0` `< $10M ARR` vs `Qwen/Qwen2.5-Coder-7B-Instruct` under `Apache-2.0`) and empirical Gate G2 measurement remain pending (`configs/gate_g2_evidence.json` status: `template_no_experiment_run`).
5. **External Human Benchmark (`HandCraft-Bench-50`, `P4-01`)**:
   - `external_gold_inventory_present: false` in `configs/data_policy.yaml` until reviewed permissive external fixtures are collected.

### 7.3 Priority Questions & Next Verification Steps
1. **Live Client Compatibility Gate (`P0-02` / `P3-02`)**: Execute pinned Blockbench 5.0, Minecraft Java 1.21.11, Iris LabPBR 1.3, Bedrock, and GeckoLib client import checks on asymmetric rotated/planar/animated fixtures and archive logs under `artifacts/validation/`.
2. **Comparable 8K Candidate Bake-Off (`P0-03` / `P0-04`)**: Execute the 1-epoch matched bake-off between `LiquidAI/LFM3-VL-2B` (multimodal Level-5 VLM capability, `LFM Open License v1.0` `< $10M ARR` cap) and `Qwen/Qwen2.5-Coder-7B-Instruct` (`Apache-2.0` unrestricted commercial) on Colab G4 NVIDIA RTX PRO 6000 (`~8.8 CU/hr`).
3. **Empirical Gate G2 Evaluation (`P1-10`)**: Measure `run_bounded_repair_policy` (`<= 3` re-audits) + `verify_prompt_adherence` across `test_unseen_params`, `test_unseen`, `test_compositional`, and `apex_unseen` (`elder_calamity_dragon`, `relic_automaton_guardian`) to decide whether conditional Phase 2 learned refinement (`< 90%` threshold) is warranted.
