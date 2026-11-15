# ADR 0006: Block-DSL v3.1/v3.2 Musculoskeletal Biomechanics, Clean-Room Complex Entity World-Modeling, Closed-Loop Artist Refinement & CC-BY-NC Research Quarantine

- **Status**: Accepted (updated for v3.2 clean-room neutral naming, Zajac/Hill curve, Bedrock X-reflection & `NoAI` provenance gate)
- **Date**: 2026-11-14
- **Tasks**: `P1-01-ext`, `P1-03f`, `P1-08-r4`, `P3-01-biomech`
- **Extends**: `ADR 0003` (Stateful Surface-Material World Model), `ADR 0004` (Multi-Domain Anti-Overfitting Taxonomy), `ADR 0005` (Categorized Assistant World-Model Dataset & Batch QC)

## Context

SyncCraft's core thesis is a **rule-grounded, symbolic + physical 3D asset world model** — capable of constructing geometry, rigging hierarchies, synthesizing multi-target textures, and generating collision-free animations from scientific and mechanical invariants rather than black-box pixel/voxel/mesh diffusion denoising.

Scaling this world model toward **large biomechanical multi-limb entities and morphing mechanisms** (`32–68` cuboids, `8–30` bones, multi-phalanx wings, counterbalancing tails, reactive emissive organ states, transforming cam-latch tools) and **stylized companion creatures and modular interactive stations** exposes four structural requirements:

1. **Block-DSL Expressiveness for Deep Rigs, Authored Primitives & Attachments**:
   - Starter `Block-DSL` rejected nested `mirror_x` subtrees (needed for mirrored wing/chela phalanges and bilateral sub-structures), lacked joint Range-of-Motion (`joint_limits`) bounds, lacked named 3D attachment/effector `locator` sockets (standard in Bedrock `.geo.json` for `head`, `seat_1`, `target`, `item`, VFX emitters, and foot-contact anchors), lacked fractional `size_step` / zero-thickness `plane` / per-box `rotate` primitives for human-authored rigs, and represented animation plans as untyped comments rather than validated `@behavior` AST declarations.
   - Starter `@materials` exposed only six scalar fields in text syntax, omitting cross-box `weld_group` manifold seam continuity (required for seamless Gray–Scott Turing skin patterns across multi-segment torsos/tails), reaction-diffusion `(rd_feed, rd_kill)` regime selection, `anisotropy`, `subsurface`, `porosity`, `emissive`, and thin-film thickness.
2. **Unified Musculoskeletal & Tendon-Coupled Movement (`MASS` / `MyoSuite` / Zajac 1989 Inspiration)**:
   - Character biomechanics research — notably **Zajac** (1989), **MASS: Muscle-Actuated Skeletal System** (Lee et al., ACM SIGGRAPH 2019), and **MyoSuite** (Caggiano et al., NeurIPS 2022) — demonstrates that independent joint-angle oscillation produces robotic postures, whereas **antagonist flexor–extensor muscle-tendon units (MTUs)** with active force-length and passive parallel-elastic curves, **biarticular tendon linkages**, and **secondary muscle-belly bulging** unify anatomical form with movement control.
   - In integer-UV block models (`AGENTS.md`: *"Integer sizes are discrete and fixed during refinement; `inflate` and physical size are separate"*), secondary muscle-belly cuboids express contraction cleanly via bounded sub-texel `inflate` modulation ($\Delta\text{inflate}_m$), audited under swept 15-axis OBB SAT.
3. **Closed-Loop Artist-in-the-Loop Iterative Refinement (`Text2CAD` / `BlenderAlchemy` Inspiration)**:
   - Consistent with iterative symbolic program synthesis (**Text2CAD**, Khan et al., NeurIPS 2024; **BlenderAlchemy**, Huang et al., ECCV 2024), an effective modeling & animation assistant operates in a **rule-preserving closed loop**:
     - **Project Intake & Critique**: Ingest an artist's existing `Block-DSL`, `.bbmodel`, Java `.json`, or Bedrock `.geo.json` + `.animation.json` (applying exact left-handed X-axis reflection conversion), audit its physics/hierarchy/anatomy/materials/swept-motion, and emit ranked, rule-grounded enhancement suggestions.
     - **Progressive From-Scratch Construction**: Build from `Stage 0` silhouette blockout -> `Stage 1` articulated skeleton & locators -> `Stage 2` secondary musculature & armor -> `Stage 3` scientific `@materials`/`@style` & welded seams -> `Stage 4` musculoskeletal `@kinematics` + `@behavior` clips & reactive states, with every stage SAT + contact audited.
4. **Clean-Room Naming & Strict `CC-BY-NC` / `NoAI` Quarantine Governance**:
   - Clean-room procedural generators released under `Apache-2.0` must use **100% neutral biomechanical and functional identifiers and prompt language** (`elemental_companion_critter`, `modular_botanical_healing_station`, `apex_flying_wyvern`, etc.), never third-party franchise or mod names.
   - External community Bedrock/Blockbench creature assets released under **`CC-BY-NC-3.0` / `CC-BY-NC-SA-3.0` / `CC-BY-NC-4.0` / `CC-BY-NC-SA-4.0`** may be analyzed strictly inside the quarantined `research_nc/` non-commercial split (`commercial_weights_eligible=False`), while any `CC-BY-ND` (NoDerivatives) license, any `NoAI` / `do not train` creator opt-out notice, or unlabeled local provenance (`NOASSERTION` default) is blocked from commercial training.

## Decision

### 1. Backward-Compatible `Block-DSL v3.1 / v3.2` Extensions (`src/synccraft/dsl/`)

While keeping unimplemented starter placeholders (`extrude`, `@shade`, `@motion`, `@ergonomics`) raising `UnsupportedFeature` per `ADR 0001`, `Block-DSL` adds:

- **Arbitrary-Depth Nested `mirror_x` & Sugar (`let`, `def`/`use`, `chain`, `mirror`/`radial`)**: Deterministic prefix chaining (`outer_r__inner_r__leaf`) and reflection parity toggling ($\text{mirrored}_{\text{clone}} = \neg\text{mirrored}_{\text{src}}$) with $\det(R) = +1$, bounded by `MAX_EXPANDED_MIRROR_NODES = 8192`.
- **Authored Geometry Primitives (`size_step`, `plane`, per-box `rotate`, `@textures`)**: Preserves fractional cube grids, `0`-thickness flat ear/fin quads, and rotated cubes in human-authored `.bbmodel` and Bedrock `.geo.json` imports (`< 1e-4` corner error).
- **`joint_limits`, `spring` & `locator` Nodes in `@rig`**: Child `Bone`s accept per-axis `joint_limits`, secondary `spring` parameters, and zero-geometry 3D `locator` sockets compiled under the shared `SE(3)` chain.
- **Typed `@muscles`, `@kinematics`, `@controller`, and `@behavior` Sections**:
  - `@muscles`: Reduced-order Zajac/Hill active force-length + passive parallel-elastic actuators (`fidelity_label == "reduced_order_qualitative"`), optional `tendon_coupling`, and Box-UV-preserving `belly_box` + `bulge_inflate`.
  - `@kinematics` & `@behavior`: Validated `ik_chain`, `look_chain`, harmonic/keyframe `track`s, `muscle_drive`s, and closed-loop `foot_cycle` stance IK (`< 0.02u` stance slip) serialized directly into AST syntax (zero `// @animation_plan` trailing comments).

### 2. Clean-Room 10-Family Apex Complex Suite (`src/synccraft/data_gen/complex_entity_and_asset_pack.py`)

- Implement 10 neutral clean-room Apex families (`8` Seen: `apex_flying_wyvern`, `brute_theropod_wyvern`, `pelagic_leviathan_wyvern`, `fanged_thunder_wyvern`, `armored_arthropod_colossus`, `morphing_switch_blade_axe`, `elemental_companion_critter`, `modular_botanical_healing_station`; `2` Held-Out Unseen $\ge 40$ boxes: `elder_calamity_dragon`, `relic_automaton_guardian`) with rich multi-variant topological branching and zero split-name leakage in prompts.

### 3. Dual-Track `CC-BY-NC` Quarantine, `NoAI` Enforcement & Exact Bedrock X-Reflection

- **`clean_room_permissive` (default)**: Permits only `CC0-1.0`, `CC-BY-4.0`, `MIT`, `Apache-2.0`, `BSD-3-Clause` with zero `NoAI` notices (`commercial_weights_eligible=True`). Unlabeled local workspace files default to `license_spdx="NOASSERTION"` (`commercial_weights_eligible=False`), never `CC-BY-4.0`.
- **`research_nc`**: Additionally permits `CC-BY-NC-3.0`, `CC-BY-NC-SA-3.0`, `CC-BY-NC-4.0`, and `CC-BY-NC-SA-4.0` strictly inside `research_nc/` (`commercial_weights_eligible=False`), while rejecting `CC-BY-ND-*` and any description containing `NoAI` or `do not train`.
- **Bedrock Handedness**: Both outbound `export_bedrock` and inbound `_decompile_bedrock_geojson` apply the exact left-handed X-axis reflection (`-(origin_x + size_x)`, negated pivot/locator X, negated X/Y rotations, negated X position).

## Consequences

- **Positive**: Unifies multi-bone biomechanical modeling, Zajac/Hill muscle-tendon coupling, closed-loop artist project critique/refinement, exact Bedrock roundtrip conversion, neutral clean-room `Apache-2.0` procedural synthesis, and `NoAI`-respecting `CC-BY-NC` quarantine under one deterministic world model.
- **Boundaries Preserved**: Explicit `fidelity_label` tags (`physical_law_validated`, `reduced_order_qualitative`, `stylistic`) prevent overstating reduced-order solvers; live GUI/engine runtime imports remain tracked under `P0-02` / `P3-02` (`runtime_import_verified=False`); and GPU training/Gate G2 remain tracked under `P1-09` / `P1-10`.
