# SyncCraft v3.1 — Comprehensive Technical, Mathematical & Architectural Audit Report

**Prepared for**: Independent Cross-AI & Principal Engineering Peer Review  
**Repository**: `/Users/tianrui6641/SyncCraft` (`https://github.com/BaiTian6641/SyncCraft`)  
**Dataset Hub**: `TassP0lar/synccraft-v3-blockdsl-dataset` (`https://huggingface.co/datasets/TassP0lar/synccraft-v3-blockdsl-dataset`)  
**Architecture Baseline**: SyncCraft v3.0 Research Plan (`source/synccraft-v3-plan.pdf`) + `ADR 0001`–`0006`  
**Default Target Backbone**: `LiquidAI/LFM3-VL-2B` (BF16 PEFT LoRA `r=32, alpha=64`, zero quantization)  
**Codebase Scale**: `25,284` LOC across `54` Python files (`44` library modules + `10` package entrypoints in `src/synccraft/`, plus `45`-line `grammar.lark`) + `5,171` LOC across `14` test suites (`tests/`) = **`30,455` Python LOC**  
**Local Verification Environment**: `Python 3.12.16`, `macOS 27.2.1 (arm64 Darwin 27.2.0)`, `numpy 2.3.5`, `Pillow 13.0.0`, `pyarrow 26.0.1`, `torch 2.15.0` (CPU)

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
4. [Complex Entity, Cobblemon Ecosystem & Closed-Loop Artist-Assistant Capabilities (`ADR 0006`)](#4-complex-entity-cobblemon-ecosystem--closed-loop-artist-assistant-capabilities-adr-0006)
   - 4.1 Monster Hunter-Class Complex Entities & Morphing Weapons (32–64 Boxes)
   - 4.2 Cobblemon-Class Creature/Block Rigs, Semantic Locators & `CC-BY-NC` Quarantine
   - 4.3 Zero-`eval` Bedrock `.geo.json` & Molang `.animation.json` Decompilers
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

**SyncCraft v3.1** takes the opposite scientific stance: **assets are executable, typed, causally coupled physical programs** ($\mathcal{W} = (\mathcal{G}, \mathcal{M}, \mathcal{S}, \mathcal{K}, \mathcal{B})$):
- **Geometry ($\mathcal{G}$)**: Constructive box-and-socket CAD assembly compiled through a single shared $SE(3) / O(3)$ forward kinematics engine ([se3_chain.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/kinematics/se3_chain.py)) with integer Box-UV dimensions kept strictly invariant under continuous surface inflation (`inflate`).
- **Surface Physics ($\mathcal{M}, \mathcal{S}$)**: Rest-space 3D box-manifold PDEs ([domain.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/surface/domain.py), [processes.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/surface/processes.py)) separating intrinsic `MaterialProgram` parameters from time-evolving `PersistentSurfaceState` fields (moisture, wear, oxidation, temperature, biofilm, Gray–Scott $(u, v)$ morphogens), compiled through linear-space Cook–Torrance GGX / Airy thin-film optics ([compiler.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/shading/compiler.py)).
- **Biomechanics & Locomotion ($\mathcal{K}, \mathcal{B}$)**: Reduced-order Hill-type muscle-tendon actuators (`@muscles`), biarticular passive tendon pantographs, agonist–antagonist co-contraction stiffness, isochoric `belly_box` `bulge_inflate`, and world-anchored $C^1$ `foot_cycle` 2-bone IK (`< 0.02` model-unit stance slip) ([musculoskeletal.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/kinematics/musculoskeletal.py)).
- **Closed-Loop Artist Collaboration**: Deterministic project intake, 5-pillar quantitative critique, iterative rule-verified enhancement, and 5-stage progressive from-scratch construction ([artist_loop.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/artist_loop.py)) — every intermediate state audited by exact 15-axis Separating Axis Theorem (SAT) collision and P1-02 physical contact checks ([sync_audit.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/audit/sync_audit.py)).

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
                 │  Blockbench 5 (.bbmodel) | Java (.json) | glTF 2.0   │
                 │  LabPBR 1.3 (_n/_s) | Vanilla Shaded Atlas           │
                 └──────────────────────────────────────────────────────┘
```

### Complete Package & File Inventory (`src/synccraft/` — `25,284` Python LOC + `45`-line EBNF)

| Subsystem | Module | LOC | Responsibility & Status |
|---|---|---:|---|
| **DSL (`dsl/`)** | [ast_nodes.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/ast_nodes.py) | 766 | Typed immutable AST dataclasses (`Program`, `Bone`, `Box`, `Locator`, `MaterialDecl`, `StyleDecl`, `MuscleDecl`, `TendonCoupling`, `KinematicsDecl`, `BehaviorDecl`), semantic validator, and `MAX_EXPANDED_MIRROR_NODES = 8192` pre-expansion guard. **Complete (v3.1).** |
| | [parser.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/parser.py) | 926 | Bounded recursive-descent lexer/parser (`2 MB` / `200k` token / depth `64` caps) + canonical `format_program` serializer. Zero `eval`. Explicit `UnsupportedFeature` on `extrude`, `@shade`, `@motion`, `@ergonomics`. **Complete (v3.1).** |
| | [grammar.lark](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/grammar.lark) | 45 | Declarative EBNF reference grammar mirroring `parser.py` v3.1 for downstream XGrammar integration. **Reference spec.** |
| | [stateful_decoder.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/stateful_decoder.py) | 8 | Explicit stub raising `NotImplementedError` pending `P1-07` live tokenizer XGrammar + KV rollback. **Deferred (`P1-07`).** |
| **Kinematics (`kinematics/`)** | [se3_chain.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/kinematics/se3_chain.py) | 431 | Single source of truth for forward $SE(3)$ & arbitrary-depth $O(3)$ `mirror_x` expansion, `WorldBone`, `WorldBox`, `WorldLocator`, `joint_limits` clamping, `box_inflate_deltas`, and `root_translation_delta`. **Complete.** |
| | [musculoskeletal.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/kinematics/musculoskeletal.py) | 751 | Per-axis quasi-static Hill-type muscle equilibrium with `joint_limits` stop enforcement, agonist–antagonist stiffness, biarticular `TendonCoupling` pantographs, exact 3D SE(3) post-IK muscle strain & `belly_box` `bulge_inflate`, `solve_look_chain` (`±x/±y/±z`), $C^1$ world-anchored `foot_cycle` IK, and `audit_behavior_clip`. **Complete (`ADR 0006`).** |
| | [sat_potential.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/kinematics/sat_potential.py) | 95 | Exact 15-axis Separating Axis Theorem OBB overlap solver with degenerate cross-axis masking (`1e-8`) + smooth LogSumExp surrogate energy diagnostic. **Complete (NumPy).** |
| **Audit (`audit/`)** | [sync_audit.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/audit/sync_audit.py) | 294 | World-AABB broadphase + exact 15-axis SAT + P1-02 physical contact/inset (`<= 0.5u`)/gap/ancestor-aware connectivity auditor (`enforce_physical_contact=True`). **Complete.** |
| | [offscreen_renderer.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/audit/offscreen_renderer.py) | 241 | Deterministic CPU 4-view orthographic painter (`render_four_views`) + Z-buffered ray-OBB 3D textured/lit world renderer (`render_world_observation`). **Complete.** |
| **Surface (`surface/`)** | [contracts.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/surface/contracts.py) | 631 | `MaterialProgram`, `MaterialSpec`, `PersistentSurfaceState`, `OpticalFields`, `StyleConfig`, `WorldState`, and 10-family physical default tables. **Complete.** |
| | [domain.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/surface/domain.py) | 508 | Discretized 3D box-surface manifold cells + 3 distinct topology graphs (`manifold_seams` with `weld_group` gating, `posed_contacts`, nested-mirror `symmetry_pairs`) + conservative graph Laplacian. **Complete.** |
| | [processes.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/surface/processes.py) | 815 | 10 reduced-order material generators, CFL-substepped Gray–Scott reaction–diffusion PDE, moisture/thermal diffusion, Archard frictional wear, selective bare-metal oxidation, and biofilm growth. **Complete.** |
| **Shading (`shading/`)** | [uv_packer.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/shading/uv_packer.py) | 143 | Deterministic integer padded per-face atlas allocator (`allocate_uv_layout`) with zero face overlap. **Complete.** |
| | [compiler.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/shading/compiler.py) | 402 | Exact area-weighted `downsample_optical_fields`, linear-space Cook–Torrance GGX (`alpha = alpha_p^2`), Airy thin-film interference, and OKLCH ramp quantizer (`shade_asset`). **Complete.** |
| | [cavity_shader.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/shading/cavity_shader.py) | 54 | Diagnostic flat-color OKLCH->sRGB atlas builder (`build_atlas`) for unshaded geometry checks. **Complete.** |
| **World (`world/`)** | [transitions.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/transitions.py) | 933 | Pure causal transition operator `step_world(S_t, a_t, E_t, dt)` over 12 typed action classes including `ExecuteBehaviorClipAction` and `ActivateBehaviorStateAction`. **Complete.** |
| | [inverse_inference.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/inverse_inference.py) | 1564 | Dual-track license gate (`clean_room_permissive` vs `research_nc` `CC-BY-NC`), multi-root/per-cube-rotated `.bbmodel`, Java `.json`, and Bedrock `.geo.json` (`<1e-9` 8-corner vertex error + locators) decompilers, zero-`eval` Molang `.animation.json` decompiler (`q.anim_time` speed + `math.clamp`), and top-`K` inverse `MaterialProgram` estimator. **Complete.** |
| | [artist_loop.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/artist_loop.py) | 1259 | 5-pillar quantitative artist project critique, zero-vertex-drift flat-rig articulation (`_articulate_flat_hierarchy`), non-blocking iterative rule-verified enhancement loop, and 5-stage progressive from-scratch trajectory builder. **Complete (`ADR 0006`).** |
| | [interventions.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/interventions.py) | 372 | 7-scenario causal physical invariance suite (UV-repack pixel invariance, rest-albedo repose invariance, Archard wear -> selective rust, Turing scale stability). **Complete.** |
| | [trajectory_dataset.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/trajectory_dataset.py) | 698 | Multi-step `infer`/`predict`/`plan_repair` trajectory generator with held-out triple quarantine and matched-cost policy evaluator. **Complete.** |
| | [exemplars.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/exemplars.py) | 296 | Canonical stateful world-model exemplars across geometry/material archetypes. **Complete.** |
| **Data Gen (`data_gen/`)** | [static_families.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/static_families.py) + [multi_domain_taxonomy.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/multi_domain_taxonomy.py) + [dragon_and_fauna_expansion.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/dragon_and_fauna_expansion.py) + [industrial_and_vehicle_expansion.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/industrial_and_vehicle_expansion.py) | 7394 | `55` core procedural families (`40` seen + `15` held-out unseen `>=16` boxes) across 7 macro-domains and 21 categories. **Complete (`ADR 0005`).** |
| | [complex_entity_and_asset_pack.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/complex_entity_and_asset_pack.py) | 2241 | `10` Apex Complex families (`8` seen + `2` held-out unseen `>=40` boxes; `32–64` boxes/asset) with nested `mirror_x`, `@muscles`, `@kinematics`, `@behavior`, semantic locators, and 23-column Parquet/JSONL `generate_apex_complex_dataset`. **Complete (`ADR 0006`).** |
| | [ik_motion_templates.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/ik_motion_templates.py) | 312 | Analytic hinge + associative-$SO(3)$ Levenberg–Marquardt 2-bone IK solver (`solve_two_bone_ik`) with `joint_limits` clamping (~200x accelerated). **Complete.** |
| | [curriculum_and_assistant.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/curriculum_and_assistant.py) + [diversity_engine.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/diversity_engine.py) + [batch_pipeline.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/batch_pipeline.py) + [taxonomy_catalog.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/taxonomy_catalog.py) | 2565 | 5-tier complexity curriculum, 4 assistant modes + 5% controlled defect-repair pairs, 6 naming registers, batch SPC QC, and 3D+2D contact-sheet renderer. **Complete.** |
| **Backends & CLI** | [bbmodel5_exporter.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/backends/bbmodel5_exporter.py), [java_exporter.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/backends/java_exporter.py), [labpbr_exporter.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/backends/labpbr_exporter.py), [gltf_exporter.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/backends/gltf_exporter.py), [common.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/backends/common.py), [cli.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/cli.py), `__init__.py`s | 1403 | Blockbench 5 `.bbmodel`, Java modern/legacy `.json`, LabPBR 1.3 `_n`/`_s` PNGs, glTF 2.0 PBR (`ORM` + `KHR_materials_iridescence`), and deterministic CLI. **Complete (CPU structural/channel verified).** |
| | [bedrock_gecko_exporter.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/backends/bedrock_gecko_exporter.py) | 11 | Explicit `ExportError` guard on outbound Bedrock/GeckoLib export (`P3-03`); inbound Bedrock/Cobblemon decompiler is in `world/inverse_inference.py`. **Outbound deferred (`P3-03`).** |
| **Evaluation & Modeling** | [budget.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/evaluation/budget.py), [gate_g2.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/evaluation/gate_g2.py), `modeling/*.py`, [notebooks/synccraft_end_to_end.ipynb](file:///Users/tianrui6641/SyncCraft/notebooks/synccraft_end_to_end.ipynb) | 151 + NB | Monotonic cost ledger, Gate G2 `<90%` decision evaluator, explicit stubs for Torch PGD/learned refiner (`P1-06`/`P2-01`), and standalone Colab `LiquidAI/LFM3-VL-2B` BF16 LoRA training pipeline. **G2 & GPU run pending.** |

---

## 3. Exact Mathematical, Geometric, Optical & Biomechanical Formulations

### 3.1 Block-DSL v3.1 Typed AST & Zero-`eval` Security Contract
Every textual Block-DSL program is tokenized and parsed by a hand-written bounded recursive-descent parser ([parser.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/parser.py)) enforcing hard resource ceilings before AST allocation:
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

## 4. Complex Entity, Cobblemon Ecosystem & Closed-Loop Artist-Assistant Capabilities (`ADR 0006`)

### 4.1 Monster Hunter-Class Complex Entities & Morphing Weapons (`32–64` Boxes)
[complex_entity_and_asset_pack.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/data_gen/complex_entity_and_asset_pack.py) implements **10 clean-room Apex Complex families** (`8` Seen + `2` Held-Out Unseen $\ge 40$ boxes), each verified across multiple random seeds for rest SAT, P1-02 physical contact, multi-phase swept clip SAT, and `< 0.02u` stance-foot slip:

| Family ID | Split | Boxes | Bones | Locators | Key Biomechanical / Structural Features |
|---|---|---:|---:|---:|---|
| `apex_flying_wyvern` | Seen | 56 | 25 | 6 | 2-level nested `mirror_x` wings (carpal hub + 4 symmetric phalanges/membranes per wing), digitigrade IK talons, pectoralis muscles + wing-fold pantograph, segmented flame-sac tail |
| `brute_theropod_wyvern` | Seen | 34 | 16 | 6 | Massive jaw/cranial battering horn, muscular gastrocnemius/quadriceps hindlegs with `belly_box` bulge, counterbalanced clubbed tail, bipedal stomp gait |
| `pelagic_leviathan_wyvern` | Seen | 39 | 22 | 8 | Travelling-wave undulating spine/neck/fluke (`phase_deg` cascade), flaring dorsal conductive sails, webbed pectoral fins, hydro-vortex locators |
| `fanged_thunder_wyvern` | Seen | 36 | 19 | 7 | Quadruped muscular forelimbs, articulating fulgur-capacitor dorsal plates (`fulgur_overcharge_state` emissive transition), Diagonal trot IK |
| `armored_arthropod_colossus` | Seen | 38 | 19 | 9 | Multi-leg alternating tripod gait, raptorial crushing chelicerae with flexor muscles, segmented carapace |
| `morphing_switch_blade_axe` | Seen | 32 | 9 | 2 | Mechanical phial cam-latch weapon (`axe_to_sword_morph_cycle`) with synchronized latch/blade transformation + overheated phial state |
| `elemental_pocket_companion` | Seen | 35 | 18 | 10 | Cobblemon-class expressive creature rig (`rider_seat`, `camera_focus`, `hitbox`, `socket`, `vfx_emitter` locators, ear-perk muscles, Turing coat) |
| `modular_apricorn_healing_station` | Seen | 33 | 8 | 4 | Cobblemon-class interactive mechanical healing/crafting block (`block_entity`) with articulated cradle trays, scanner arm, and emissive status visor |
| `elder_calamity_dragon` | **Unseen ($\ge 40$)** | 64 | 29 | 8 | True hexapod (4 terrestrial IK legs + 2 nested-mirror `mirror_x` wings = 6 limbs), cranial crown horns, elder aether state |
| `relic_automaton_guardian` | **Unseen ($\ge 40$)** | 49 | 17 | 5 | Ancient mechanical construct with twin shoulder turbines, articulated core shutters, piston-coupled greaves, heavy guardian march IK |

### 4.2 Cobblemon-Class Assets & Dual-Track `CC-BY-NC` Quarantine Governance
Community creature/block repositories such as Cobblemon release assets under **`CC-BY-NC-4.0`** (Creative Commons Attribution-NonCommercial 4.0). SyncCraft enforces a hard programmatic quarantine in [inverse_inference.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/world/inverse_inference.py) (`verify_asset_provenance`) and [configs/data_policy.yaml](file:///Users/tianrui6641/SyncCraft/configs/data_policy.yaml):
1. **`clean_room_permissive` (Default Track — `Apache-2.0` Compatible)**:
   - Accepts **only** `{"CC0-1.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-3-Clause"}`.
   - Sets `commercial_weights_eligible = True`. Calling `ingest_external_asset(..., license_policy="clean_room_permissive")` on a `CC-BY-NC-4.0` asset raises `LicenseGateError` immediately.
2. **`research_nc` (Explicit Non-Commercial Research Quarantine Track)**:
   - Additionally accepts `{"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0"}` **only** when `attribution_notice` is non-empty.
   - Marks `license_track = "research_nc"` and `commercial_weights_eligible = False` on the returned `IngestedAssetRecord`, routing output exclusively to `research_nc/` splits so non-commercial community assets can train/evaluate academic world-model benchmarks without contaminating commercial `Apache-2.0` weights.
3. **Unconditional Mojang / Proprietary Ban**:
   - Any provenance containing Mojang/vanilla extraction or proprietary markers (`mojang`, `minecraft_jar`, `official_vanilla_jar`, `all_rights_reserved`, `proprietary`, `unlicensed`) raises `LicenseGateError` under **all** policies including `research_nc`.

### 4.3 Multi-Format SE(3) Geometry & Zero-`eval` Molang `.animation.json` Decompilers
- **Multi-Format `.bbmodel`, Java `.json`, & Bedrock / Cobblemon `.geo.json` Decompilers (`_decompile_bbmodel`, `_decompile_java_json`, `_decompile_bedrock_geojson`)**: Reconstruct arbitrary multi-root outliner hierarchies, parent-rotated bone chains, per-cube/first-element local pivots and rotations (via child sub-bones), `inflate`, and Bedrock `"locators"` dictionaries (`[x, y, z]` and `{"offset": [...], "rotation": [...]}` forms) into typed Block-DSL `Bone`/`Box`/`Locator` ASTs with inferred semantic roles (`rider_seat`, `camera_focus`, `vfx_emitter`, `hitbox`, `ik_effector`, `socket`), verified against an independent **8-corner world vertex oracle** (`< 10^{-12}` error on integer-sized cubes).
- **Zero-`eval` Molang `.animation.json` Decompiler (`decompile_bedrock_animation_to_behavior`)**: Parses harmonic Molang channel expressions (`math.sin`/`math.cos`, `q.anim_time` / `query.anim_time` angular velocities in deg/s, `math.clamp(expr, lo, hi)` extrema bounds, and standalone static biases isolated from secondary harmonics), deduplicates clip names against the global AST identifier namespace, and converts discrete keyframe timelines into typed `@behavior` `ClipDecl` (`wave` / `flap` / `hinge` `TrackDecl`s). Unrecognized or adversarial code strings (e.g., `__import__('os').system(...)`) are ignored and **never** passed to `eval`.

### 4.4 Closed-Loop Artist-Assistant Workflow (`artist_loop.py`)
1. **`analyze_artist_project`**: Ingests `Block-DSL`, `.bbmodel`, Java `.json`, or Bedrock `.geo.json` (+ `.animation.json`) and evaluates **5 weighted quantitative pillars** (`0.32` `geometry_contact`, `0.18` `rig_hierarchy`, `0.18` `surface_material`, `0.16` `musculoskeletal`, `0.16` `behavior_motion`):
   - *Geometry & Contact Integrity*: Non-exempt SAT overlaps, floating gaps, deep co-penetrations (`>0.5u`), detached physical components.
   - *Rig & Socket Ergonomics*: Monolithic multi-box entity bones needing articulation splits (`articulate_flat_rig`), missing `joint_limits`, and missing semantic `Locator` sockets (`rider_seat`/`camera_focus`/`vfx_emitter`).
   - *Surface & Material Realism*: Palettes lacking `@materials` process bindings, unwelded organic skin/scale materials lacking `weld_group`, and missing `@style` contracts.
   - *Musculoskeletal & IK Readiness*: Articulated rigs lacking `@muscles` Hill actuators or `@kinematics` `ik_chain`/`look_chain` descriptors.
   - *Behavior & Stance Grounding*: Animated target profiles lacking `@behavior` clips, or clips failing swept-SAT / `<0.02u` stance-slip audits.
2. **`run_looped_artist_enhancement`**: Iterates through ranked actionable suggestions in priority order without stalling on saturated ops — executing rest-contact repair, **zero-vertex-drift flat-rig articulation** (`_articulate_flat_hierarchy`, placing anatomical joint pivots directly on contacting OBB faces while preserving all 8 box corners to `< 10^{-12}`), `joint_limits` + `Locator` injection, scientific `@materials` + `weld_group` synthesis, and SAT-verified `@muscles` + `@kinematics` + `@behavior` synthesis.
3. **`build_progressive_from_scratch_trajectory`**: Constructs any complex entity from scratch through 5 rule-verified artist stages (`Stage 0: Rough Primary Mass Blockout` -> `Stage 1: Articulated Skeletal Hierarchy & Nested Mirrors` -> `Stage 2: Secondary Anatomy, Muscle Bellies & Semantic Locators` -> `Stage 3: Scientific Welded Surface Materials & Turing Patterns` -> `Stage 4: Musculoskeletal Tendon Couplings, Closed-Loop Stance IK & @behavior`), certifying 100% SAT + P1-02 compliance at every intermediate stage.

---

## 5. Dataset Taxonomy, Curriculum, SPC Quality Control & Provenance Governance

SyncCraft's public Hugging Face dataset (`TassP0lar/synccraft-v3-blockdsl-dataset`, `Apache-2.0`) now comprises three complementary, cryptographically hashed (`SHA-256`) clean-room pillars:

1. **Core Categorized 55-Family Assistant Corpus (`bakeoff_8k` = `8,600` rows; scalable to `full_75k` = `77,400` rows)**:
   - `40` Seen families (`train`: `7,200`, `val`: `800`) + `15` Held-Out Unseen families (`test_unseen`: `600`, strictly $\ge 16$ boxes).
   - Spans `7` macro-domains (`artificial`, `industrial`, `dragon`, `building`, `natural`, `animal`, `humanoid`) and `21` categories across `4` complexity tiers (`4–35` boxes) and `4` assistant modes (`create_full_asset`, `coarse_to_fine_upgrade`, `texture_and_animate_rig`, plus `~5%` `diagnose_and_repair_defect`).
   - `10/10` SPC batches accepted (`1.0` clean SAT pass rate, `1.0` swept animation SAT pass rate, `7.587` bits QD Shannon entropy across `366` behavior cells).
2. **Action-Conditioned World Trajectories (`world_trajectories.jsonl` = `1,200` trajectories)**:
   - Multi-step `infer` (inverse material inference from observations), `predict` (forward counterfactual rollouts under wear/moisture/pose/UV-repack actions), and `plan_repair` trajectories with zero compositional triple leakage.
3. **Apex Complex Entity, Musculoskeletal & Artist-Loop Expansion (`apex_complex_v3_1` shard)**:
   - Covers the `10` Apex Complex families (`8` Seen + `2` Held-Out Unseen $\ge 40$ boxes, `32–64` boxes/asset) across **full masterwork generation**, **5-stage progressive from-scratch trajectories**, and **closed-loop artist critique & upgrade** supervision pairs, all formatted in canonical `Block-DSL v3.1`.

---

## 6. Executed Verification Matrix & Quantitative Telemetry

Every test below was executed locally on CPU (`Python 3.12.16`, `arm64 macOS`) with zero mocks or skipped tests:

| Test Suite | Tests | Key Quantitative Invariants Proven |
|---|---:|---|
| [test_musculoskeletal_complex_and_artist_loop.py](file:///Users/tianrui6641/SyncCraft/tests/test_musculoskeletal_complex_and_artist_loop.py) | 16 | Full `Block-DSL v3.1` `AST <-> DSL` roundtrip; 2-level & 3-level nested `mirror_x` $\det(\mathbf{R}) = +1$ and cullface parity toggling; `8,192`-node mirror-bomb rejection; per-axis Hill agonist–antagonist co-contraction (`0.0` net angle, `1.85x` stiffness surge, orthogonal-axis decoupling, `joint_limits` tendon-stop clamping, dual `belly_box` bulge with immutable integer sizes); `±y` & `+z` `solve_look_chain` convergence; `CC-BY-NC` quarantine + Mojang ban; multi-root `.bbmodel` / rotated-cube Bedrock / rotated-`elem[0]` Java 8-corner vertex reconstruction (`<1e-9`) + `math.clamp`/`q.anim_time` zero-`eval` Molang decompilation; P1-02 locator-hub connectivity trap; flat 10-box zero-drift articulation -> animated upgrade (`readiness_score = 1.0`); all `10` Apex families (`32–64` boxes) pass rest SAT + P1-02 + swept SAT + `<0.02u` stance slip |
| [test_world_interventions_and_inference.py](file:///Users/tianrui6641/SyncCraft/tests/test_world_interventions_and_inference.py) | 7 | `7/7` causal physical interventions (`0` 3D pixel diff under UV repack, `0.0` rest-albedo swim under repose, Archard hinge wear -> selective bare-metal rust, `0.46%` Gray–Scott spot drift across resolutions); `.bbmodel`/Java/Bedrock inverse inference; zero triple leakage |
| [test_surface_and_pbr.py](file:///Users/tianrui6641/SyncCraft/tests/test_surface_and_pbr.py) | 9 | Conservative Laplacian mass conservation (`<1e-10` drift), LabPBR 1.3 `_n`/`_s` channel encoding (`emissive=255` sentinel never emitted accidentally), glTF 2.0 `ORM` + `KHR_materials_iridescence` export |
| [test_data_gen_and_contact.py](file:///Users/tianrui6641/SyncCraft/tests/test_data_gen_and_contact.py) | 15 | P1-02 contact policy rejects floating gaps & deep co-penetrations (`D=0`, `D=50`), 55-family curriculum & SPC QC, 5% defect-repair verification |
| [test_adversarial_challenger.py](file:///Users/tianrui6641/SyncCraft/tests/test_adversarial_challenger.py) & [test_adversarial_stress.py](file:///Users/tianrui6641/SyncCraft/tests/test_adversarial_stress.py) | 67 | Adversarial parser bounds, Unicode/NaN/Inf injection, degenerate OBB cross-axis SAT stability, smooth-vs-direct SAT counterexamples |
| [test_notebook_e2e.py](file:///Users/tianrui6641/SyncCraft/tests/test_notebook_e2e.py) | 31 | Colab notebook integrity, `LiquidAI/LFM3-VL-2B` BF16 LoRA config, dataset manifest/row-schema contract, unseen family isolation |
| Core starter suites (`test_dsl`, `test_kinematics`, `test_sat`, `test_exporters`, `test_pipeline`, `test_atlas_numeric`, `test_budget_and_gate`) | 27 | `500` seeded AST roundtrips, `1,000` flush OBB SAT fixtures, Blockbench 5 / Java export bounds, CLI determinism, Gate G2 blocker enforcement |
| **Total Executed CPU Suite (`scripts/check.py`)** | **172** | **100% Pass (`0` failures, `0` skipped)** |

---

## 7. Self-Critical Audit: Approximations, Unimplemented Milestones & Priority Attack Vectors for External AI Review

In accordance with `AGENTS.md`, this section separates **implemented & CPU-verified behavior** from **reduced-order physical approximations** and **unexecuted external milestones**, giving an external AI reviewer an unvarnished map of where to probe deepest.

### 7.1 Reduced-Order Physical Approximations (`reduced_order_model`)
1. **Quasi-Static Hill Muscle Equilibrium vs. Full Forward Multi-Body Dynamics**:
   - *Current state*: `evaluate_muscles_at_activations` solves algebraic moment-balance per joint axis with activation-modulated passive stiffness $k_{\text{passive}} = \max(0, 1 - a_{\max})$ and kinematic tendon pantographs (`TendonCoupling`). It does **not** integrate second-order rigid-body Euler–Lagrange equations $\mathbf{M}(q)\ddot{q} + \mathbf{C}(q, \dot{q}) = \boldsymbol{\tau}_{\text{muscle}} + \mathbf{J}^T\boldsymbol{\lambda}_{\text{contact}}$. Consequently, inertial whip, ballistic flight phases, angular momentum conservation during aerial leaps, and passive ground-reaction torque feedback are not modeled dynamically.
2. **Isochoric Bulge Approximation on Axis-Aligned Box Primitives**:
   - *Current state*: Muscle contraction increases `box.inflate` uniformly along all 3 axes (`h_i = s_i/2 + inflate`) because Minecraft/Blockbench `inflate` is a single scalar per box. A true constant-volume cylinder ($V = \pi r^2 L = \text{const} \implies r \propto L^{-1/2}$) expands radially while shortening axially; uniform scalar `inflate` expands axial extent by $+\delta$ as well, slightly over-estimating axial OBB length at peak bulge (capped at `0.35u`, audited by SAT so it never causes hidden collisions).
3. **2-Bone IK Degrees of Freedom**:
   - *Current state*: `solve_two_bone_ik` optimizes 3 rotational DOFs at the proximal ball joint (`upper`) + 1 hinge DOF (`hinge_axis`) at the distal joint (`lower`). Unguligrade or 3-segment insect/wyvern legs (`coxa -> femur -> tibia -> tarsus`) must currently designate 2 primary IK segments and drive remaining segments via `TendonCoupling` or `@behavior` tracks rather than a full $N$-link Jacobian transpose / FABRIK chain solver with ground-normal adaptation on uneven terrain.
4. **Surface PDE Discretization & Bedrock Handedness Convention**:
   - *Current state*: Surface diffusion and Gray–Scott reaction–diffusion operate on a cell-centered rest-pose box graph (`subdivisions_per_unit = 1..4`); anisotropic curvature across sharp $90^\circ$ box corners is approximated via Euclidean face-edge distance conductances. Additionally, `_decompile_bedrock_geojson` currently evaluates `.geo.json` coordinates and `rotation` triples directly in SyncCraft's right-handed XYZ frame; native Bedrock runtime left-handed X/Z sign reflection (`P3-03` bidirectional parity) should be locked together with outbound `export_bedrock` / `export_geckolib`.

### 7.2 Explicitly Unimplemented / Deferred Plan Milestones
1. **Polygon Extrusion (`extrude`), `@shade`, `@ergonomics`, Legacy `@motion` (`P1-01`, `P1-04`, `P3-01`)**:
   - Explicitly raise `UnsupportedFeature` in `parser.py` rather than silently approximating shapes. Note that `@materials`+`@style` (`ADR 0004`) and `@muscles`+`@kinematics`+`@behavior` (`ADR 0006`) provide clean typed replacements for shading and motion without guessing the PDF's truncated `spring_step` syntax.
2. **Outbound Bedrock `.geo.json` & GeckoLib Exporters (`P3-03`)**:
   - Inbound decompilation (`decompile_bedrock_geojson`, `decompile_bedrock_animation_to_behavior`) works end-to-end, and Blockbench 5 `.bbmodel`, Java `.json`, LabPBR 1.3, and glTF 2.0 exporters work end-to-end. However, `export_bedrock` and `export_geckolib` in [bedrock_gecko_exporter.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/backends/bedrock_gecko_exporter.py) still explicitly raise `ExportError` until schema-pinned target-runtime roundtrips are completed.
3. **Differentiable PyTorch Kinematics/SAT & PGD Continuous Refinement (`P1-05`, `P1-06`)**:
   - `pgd_projector.py` and `learned_refiner.py` remain explicit `NotImplementedError` stubs. Current repair (`run_looped_artist_enhancement` and `data_gen` repair supervision) uses deterministic geometric displacement along minimum-overlap axes on CPU rather than autograd PGD.
4. **Live Tokenizer XGrammar Masking & KV-Cache Statement Rollback (`P1-07`)**:
   - `grammar.lark` documents the v3.1 EBNF, but `stateful_decoder.py` is not yet wired to `LiquidAI/LFM3-VL-2B`'s live tokenizer symbol table and KV-cache checkpoints.
5. **External GUI / Game Shader Runtime & Live GPU Training (`P0-02`, `P0-03`, `P1-09`, `P1-10`)**:
   - No live Blockbench 5 GUI open/save roundtrip, live Minecraft Iris LabPBR 1.3 shader render, or live RTX PRO 6000 GPU training run was executed in this local macOS CPU environment. `configs/gate_g2_evidence.json` rightly remains in `blocked` status until real unseen dense-assembly evaluation runs are logged.

### 7.3 Priority Questions & Attack Vectors for the Reviewing AI
When reviewing this repository with another AI, ask it to probe these five highest-leverage technical questions:
1. **Outbound Cobblemon / GeckoLib Export Loop (`P3-03`)**: Since inbound `.geo.json` + Molang `.animation.json` decompilation is now implemented, what exact schema/pivot/locator roundtrip invariants are needed to complete outbound `export_bedrock` (`.geo.json` `1.12.0` + `.animation.json` `1.8.0` baked/analytic keyframes) so an artist can import a Cobblemon rig, enhance it in SyncCraft, and export it directly back into a Cobblemon mod pack?
2. **Multi-Link ($N \ge 3$) Digitigrade & Uneven-Terrain IK**: Should `@kinematics` generalize `ik_chain` from 2-bone analytic/LM (`upper`, `lower`) to an arbitrary $K$-bone damped-least-squares (DLS) chain with `TendonCoupling` null-space posture priors so 3-segment theropod/wyvern legs (`femur -> tibia -> tarsometatarsus`) solve stance pinning natively without splitting into 2-bone IK + passive tendon?
3. **Anisotropic `inflate` Exporter Lowering**: Given that Minecraft/Bedrock/Blockbench only support scalar `inflate` per cube, is scalar `belly_box` `bulge_inflate` the optimal Box-UV-preserving muscle deformation representation, or should `glTF 2.0` export optional morph targets / bone scale channels while clamping `.bbmodel`/`.geo.json` to scalar `inflate`?
4. **Multi-Harmonic & Catmull–Rom Molang Expressiveness**: `decompile_bedrock_animation_to_behavior` extracts primary `sin`/`cos` harmonics, `q.anim_time` angular speeds, standalone biases, `math.clamp(...)` bounds, and min/max keyframe laws, while discarding higher-order overtones (`0.5*sin(2ωt)`) and Catmull–Rom (`"catmullrom"`) keyframe tangents. Extending `TrackDecl` to a multi-harmonic Fourier sum (`harmonics=[(amp, freq, phase), ...]`) would make `.animation.json <-> @behavior` nearly lossless.
5. **PyTorch Vectorized Batched SE(3) + SAT (`P1-05`/`P1-06`)**: How should `se3_chain.py` and `sat_potential.py` be mirrored in batched `torch.float64`/`float32` tensors (with verified `<1e-6` parity against NumPy across nested `mirror_x` trees) to unlock PGD continuous refinement and measure Gate G2 (`90%` threshold) on the held-out `test_unseen` + `apex_unseen` (`elder_calamity_dragon`, `relic_automaton_guardian`) splits?
