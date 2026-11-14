# ADR 0006: Block-DSL v3.1 Musculoskeletal Biomechanics, Complex Entity World-Modeling, Closed-Loop Artist Refinement & CC-BY-NC Research Quarantine

- **Status**: Accepted
- **Date**: 2026-11-14
- **Tasks**: `P1-01-ext`, `P1-03f`, `P1-08-r4`, `P3-01-biomech`
- **Extends**: `ADR 0003` (Stateful Surface-Material World Model), `ADR 0004` (Multi-Domain Anti-Overfitting Taxonomy), `ADR 0005` (Categorized Assistant World-Model Dataset & Batch QC)

## Context

SyncCraft's core thesis is a **rule-grounded, symbolic + physical 3D asset world model** — capable of constructing geometry, rigging hierarchies, synthesizing multi-target textures, and generating collision-free animations from scientific and mechanical invariants rather than black-box pixel/voxel/mesh diffusion denoising.

Scaling this world model toward **Monster Hunter-class complex biomechanical entities** (40–85+ cuboids, 16–30+ bones, multi-phalanx wings, counterbalancing tails, rage/charged organ states, morphing weapons) and **Cobblemon-style stylized creatures and interactive game blocks** exposes four structural requirements:

1. **Block-DSL Expressiveness for Deep Rigs & Attachments**:
   - Starter `Block-DSL` rejected nested `mirror_x` subtrees (needed for mirrored wing/chela talons and bilateral sub-structures), lacked joint Range-of-Motion (`joint_limits`) bounds, lacked named 3D attachment/effector `locator` sockets (standard in Cobblemon Bedrock `.geo.json` for `head`, `seat_1`, `target`, `item`, VFX emitters, and foot-contact anchors, forcing earlier decompilers to insert colliding `1x1x1` dummy cubes), and represented animation plans only as untyped trailing JSON comments rather than validated AST declarations.
   - Starter `@materials` exposed only six scalar fields in text syntax, omitting cross-box `weld_group` manifold seam continuity (required for seamless Gray–Scott Turing skin patterns across multi-segment torsos/tails), reaction-diffusion `(rd_feed, rd_kill)` regime selection, `anisotropy`, `subsurface`, `porosity`, `emissive`, and thin-film thickness.
2. **Unified Musculoskeletal & Tendon-Coupled Movement (`MASS` / `MyoSuite` Inspiration)**:
   - Frontier character biomechanics research — notably **MASS: Muscle-Actuated Skeletal System** (Lee et al., ACM SIGGRAPH 2019) and **MyoSuite** (Caggiano et al., NeurIPS 2022) — demonstrates that independent joint-angle oscillation produces robotic or physiologically impossible postures, whereas **antagonist flexor–extensor muscle-tendon units (MTUs)**, **biarticular tendon linkages** (e.g., avian/wyvern propatagial four-bar wing folding and digitigrade achilles pantographs), and **secondary muscle-belly bulging** unify anatomical form with movement control.
   - In integer-UV block models (`AGENTS.md`: *"Integer sizes are discrete and fixed during refinement; `inflate` and physical size are separate"*), muscle contraction cannot mutate integer `Box.size` without tearing Box-UV islands. Instead, secondary muscle-belly cuboids express isochoric contraction cleanly via bounded sub-texel `inflate` modulation ($\Delta\text{inflate}_m = \beta_m \cdot a_m(t)$), audited under swept 15-axis OBB SAT.
3. **Closed-Loop Artist-in-the-Loop Iterative Refinement (`Text2CAD` / `BlenderAlchemy` Inspiration)**:
   - Real 3D creature and game-asset artists do not emit a 65-box rig in one unstructured shot, nor do they rely on destructive mesh denoising. Consistent with iterative visual/symbolic program synthesis (**Text2CAD**, Khan et al., NeurIPS 2024; **BlenderAlchemy**, Huang et al., ECCV 2024), an effective modeling & animation assistant must operate in a **rule-preserving closed loop**:
     - **Project Intake & Critique**: Ingest an artist's existing `Block-DSL`, `.bbmodel`, Java `.json`, or Bedrock/Cobblemon `.geo.json` + `.animation.json`, audit its physics/hierarchy/anatomy/materials/swept-motion, and emit ranked, rule-grounded enhancement suggestions plus executable AST upgrades.
     - **Progressive From-Scratch Construction**: Build from a `Stage 0` rough conceptual silhouette blockout (`4–8` boxes) -> `Stage 1` articulated skeleton & locators -> `Stage 2` secondary musculature & osteoderm armor -> `Stage 3` scientific `@materials`/`@style` & welded Turing seams -> `Stage 4` musculoskeletal `@kinematics` + `@behavior` clips & reactive states, with every stage 100% SAT + P1-02 valid.
4. **Cobblemon (`CC-BY-NC`) Ingestion & Clean-Room Weight Isolation**:
   - Cobblemon's open repository distributes high-craft Bedrock `.geo.json` models, Molang `.animation.json` clips, and textures under **Creative Commons Attribution-NonCommercial (`CC-BY-NC-SA-4.0` / `CC-BY-NC-4.0`)** (alongside third-party creature IP considerations).
   - While `CC-BY-NC` is acceptable for non-commercial academic/research dataset analysis and structural world-model study when explicitly authorized by the user, mixing `CC-BY-NC` records into SyncCraft's default `Apache-2.0` clean-room commercial track would create license taint. An explicit, machine-enforced dual-track license quarantine is therefore mandatory.

## Decision

### 1. Backward-Compatible `Block-DSL v3.1` Extensions (`src/synccraft/dsl/`)

While keeping unimplemented starter placeholders (`@shade`, `@motion`, `@ergonomics`, `extrude`) raising `UnsupportedFeature` per `ADR 0001`, `Block-DSL v3.1` adds:

- **Arbitrary-Depth Nested `mirror_x`**: Deterministic prefix chaining (`outer_r__inner_r__leaf`) and reflection parity toggling ($\text{mirrored}_{\text{clone}} = \neg\text{mirrored}_{\text{src}}$, leveraging $S^2 = I$ for $S = \text{diag}(-1, 1, 1)$) so nested reflected subtrees satisfy exact global $O(3)$ reflection with $\det(R) = +1$ and reciprocal `SurfaceDomain` `symmetry_pairs`.
- **`joint_limits` & `locator` Nodes in `@rig`**:
  - Child `Bone`s accept optional `joint_limits: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]` (degrees around `rest_euler`, requiring $x_{\min} \le 0 \le x_{\max}$).
  - `Bone` items accept `locator <name> { offset: [x, y, z], role: <socket|ik_effector|vfx_emitter|camera_focus|hitbox|rider_seat> }`, compiled into `Scene.locators` under the exact shared `SE(3)` chain without adding phantom OBB collision geometry. Structural bones may attach children directly to a parent `Locator` (`anchor=[0,0,0]`).
- **Extended `@materials` Physical & Reaction–Diffusion Parameters**:
  - Optional default-preserving fields: `anisotropy`, `subsurface`, `porosity`, `emissive`, `rd_feed`, `rd_kill`, `film_nm`, `wear`, `oxidation`, `biofilm`, and `weld_group`. Canonical `format_program` emits extended kwargs only when non-default so all existing `v3.0` canonical hashes remain byte-identical.
- **Typed `@muscles`, `@kinematics`, and `@behavior` Sections**:
  - `@muscles`: Declares reduced-order Hill-type muscle-tendon units (`origin_bone`, `insertion_bone`, `action_axis`, `moment_arm_deg`, optional biarticular `tendon_coupling: [bone, axis, ratio]`, and optional isochoric `belly_box` + `bulge_inflate` in `[0.0, 0.35]`). Explicitly labeled `reduced_order_model` in telemetry payloads.
  - `@kinematics`: Declares validated `ik_chain` (`upper`, `lower`, `effector`, `hinge_axis`) and multi-bone `look_chain` (`bones`, `weight_decay`) descriptors.
  - `@behavior`: Declares named `clip` definitions (`track` analytic laws `wave`/`flap`/`hinge`/`impulse`, `muscle_drive` excitations, and closed-loop stance-locked `ik_drive` `foot_cycle` targets achieving $< 0.02$ model-unit horizontal stance slip) and coupled `state` definitions (`pose_bias`, `muscle_tone`, `material_mod` modulating `emissive`, `moisture`, and `wear_delta` in `WorldState`). Allowed only on articulated/editor profiles (`bbmodel_5`, `bedrock_entity`, `geckolib`).

### 2. Musculoskeletal Simulator & World Transition Coupling (`src/synccraft/kinematics/musculoskeletal.py`, `src/synccraft/world/transitions.py`)

- Evaluate `@muscles`, `@kinematics`, and `@behavior` strictly through `compile_posed_geometry` and `solve_two_bone_ik` on the shared right-handed `SE(3)` chain.
- Audit clips across dense cycle phases via 15-axis OBB SAT (including dynamic muscle-belly `bulge_inflate`) and measure world-space stance-foot horizontal slip.
- Expose `ExecuteBehaviorClipAction` and `ActivateBehaviorStateAction` in `step_world(S_t, a_t, E_t, dt)` so movement cycles causally accumulate Archard sliding wear at rubbing joints and trigger state-coupled optical/material transitions.

### 3. Closed-Loop Artist Assistant, Roundtrip Export & Apex Complex Entity Suite (`src/synccraft/world/artist_loop.py`, `src/synccraft/backends/bedrock_gecko_exporter.py`, `src/synccraft/data_gen/complex_entity_and_asset_pack.py`)

- Implement a deterministic 5-pillar project analyzer (`analyze_artist_project`), single/multi-step rule-based enhancer (`enhance_artist_project_step`, `run_looped_artist_enhancement` with zero-vertex-drift flat-rig articulation), 5-stage progressive from-scratch trajectory builder (`build_progressive_from_scratch_trajectory`: `Stage 0` gesture blockout -> `Stage 4` musculoskeletal animated asset), and roundtrip re-audited artist export (`export_artist_project_bundle`) across Bedrock `1.12.0`/`1.8.0` (`export_bedrock`), GeckoLib (`export_geckolib`), and animated Blockbench 5.0 (`export_bbmodel(..., program=...)`), explicitly stamping `runtime_import_verified=False` until live GUI/client verification is performed.
- Add a 10-family **Apex Complex Entity & Game-Asset Pack** (`8` Seen + `2` Held-Out Unseen, `32–64` boxes, `8–29` bones) covering Monster Hunter-inspired biomechanical archetypes (Flying Wyvern, Brute Wyvern, Pelagic Leviathan, Fanged Thunder Wyvern, Armored Arthropod Colossus, Elder Calamity Dragon, Morphing Switch-Blade Axe) and Cobblemon-inspired companion rigs & functional healing/scanner blocks, integrated into `notebooks/synccraft_end_to_end.ipynb` (`max_sequence_length=8192`) while keeping legacy `SEEN_STATIC_FAMILIES` (`40`) and `UNSEEN_STATIC_FAMILIES` (`15`) untouched for strict Gate G2 regression continuity.

### 4. Dual-Track `CC-BY-NC` Research Quarantine & Cobblemon Decompiler (`src/synccraft/world/inverse_inference.py`)

- Partition external ingestion into two explicit policies plus a dedicated quarantine dataset builder (`build_quarantined_research_nc_dataset`):
  1. `license_policy="clean_room_permissive"` (default): Permits only `CC0-1.0`, `CC-BY-4.0`, `MIT`, `Apache-2.0`, `BSD-3-Clause`; `commercial_weights_eligible=True`.
  2. `license_policy="research_nc"`: Additionally permits `CC-BY-NC-4.0` and `CC-BY-NC-SA-4.0` (e.g., Cobblemon assets) while permanently tagging `license_track="research_nc"`, routing splits under `research_nc/`, and setting `commercial_weights_eligible=False`, while continuing to block Mojang vanilla JARs and proprietary/unlicensed sources.
- Upgrade `.bbmodel` (including Blockbench 5 `"animations"`), Java `.json`, and Bedrock/Cobblemon `.geo.json` decompilation to preserve `locators`, planar `0`-thickness fin/wing/leaf quads (exact midplane + unshifted Box-UV strip coordinates + zero-area edge face skipping), multi-root/per-cube/parent-rotated SE(3) transforms (`< 1e-9` 8-corner vertex error), and zero-`eval` `.animation.json` decompilation supporting nested-parenthesis Molang arithmetic (`math.sin((q.anim_time - 0.25) * 90 * 4) * 10`), Catmull-Rom `{"pre": ..., "post": [[...]], "lerp_mode": "catmullrom"}` keyframes, GeckoLib `{"vector": [...]}`, timestamped Molang keyframe dicts, `math.clamp`, and global AST identifier deduplication.

## Consequences

- **Positive**: Unifies complex multi-bone creature modeling, physiological tendon/muscle synergy, closed-loop artist project critique/progressive blockout refinement, stateful PBR/Turing textures, and legally quarantined `CC-BY-NC` ingestion under one deterministic world model.
- **Boundaries Preserved**: No neural diffusion denoiser is introduced; `reduced_order_model` labels distinguish Hill-tendon/bulge approximations from continuum FEM; live Blockbench/Minecraft/Bedrock/GeckoLib runtime imports remain tracked under `P0-02` / `P3-02`; and `CC-BY-NC` research records can never silently enter Apache-2.0 clean-room training splits.
