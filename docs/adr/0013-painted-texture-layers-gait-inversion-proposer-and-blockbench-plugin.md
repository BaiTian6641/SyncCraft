# ADR 0013: Preserve-and-Layer Painted Textures, Inverse Gait & Deterministic Search, Additive Motion Layers, Proposer Harness, and Blockbench 5 Plugin

- **Status**: Accepted (additive Block-DSL v4.1 motion layers/effects, non-destructive texture pipeline, gait inversion/search/retargeting, Gate D evaluation harness, and local Blockbench 5 plugin service; 100% backward-compatible with ADR 0001–0012)
- **Date**: 2026-11-17
- **Related Tasks**: `P1-01`, `P1-03`, `P1-06`, `P1-10`, `P3-01`, `P3-02`
- **Builds On**: `ADR 0008`, `ADR 0009`, `ADR 0011`, `ADR 0012`

## Context

With Gates A, B, and C completed in `ADR 0011` and the quantitative 6-pillar Sculpted Style Anchor + V9 masterworks added in `ADR 0012`, four concrete non-GPU layers from the Nov 16 Blockbench Pipeline Roadmap remain to bridge the compiler/world-model engine to real Blockbench artist workflows and Gate D training:

1. **Phase 3 Texture Exit Gate (Preserve-and-Layer, Variants, & Selective Face Rebake)**:
   - When an imported model already carries a hand-painted texture, baking over it destroys the artist's pixel clusters and outlines. The engine must preserve every original pixel untouched, extract perceptual OKLCH palette profiles, offer OKLCH recolor variants, fill only blank (`A == 0`) faces when requested, emit separate transparent RGBA `wear` and `glow` layers, and support selective single-part face rebaking without regenerating unaffected UV faces.
2. **Additive Secondary Motion Layers & Stance-Touchdown Effect Keyframes**:
   - Artists layer breathing, ear flicks, tail sway, and wing settle over locomotion clips without mutating the base gait (`layer breathe: weight=0.8`), and game runtimes (GeckoLib, Bedrock `1.8.0`, Blockbench `.bbmodel`) trigger footstep audio and dust particles at stance touchdown (`effect touchdown: { sound: "...", particle: "..." }`).
3. **Inverse Gait Fitting, Bounded Deterministic Search, & Cross-Creature Retargeting**:
   - Real permissive animated `.bbmodel` / Bedrock clips animate legs via raw keyframes (often in-place on a treadmill with zero root translation). Recovering dimensionless `(pattern, froude, duty, phase, lift_ratio, spine_wave_deg, style)` from world/body-frame foot trajectories provides direct supervision for the 5-task proposer (`Label`, `Propose`, `Repair`, `Texture plan`, `Invert`), while bounded deterministic parameter search (`search_or_repair_gait`) repairs reach/clipping/balance failures before re-querying a language model.
4. **Blockbench 5 Plugin & Localhost Python Bridge**:
   - Artists work inside Blockbench, not a standalone CLI. A single-file Blockbench plugin (`Plugin.register`) talking to a `127.0.0.1`-bound Python service exposes `Analyze`, `Check` (zero-ML motion audit on artist clips), `Generate Locomotion`, `Retarget`, `Add Secondary Motion`, and `Texture Pass`, writing ordinary editable keyframes and non-destructive texture layers inside a single `Undo.initEdit` / `Undo.finishEdit` step and refusing to write unverified outputs silently.

---

## Decision

### 1. Block-DSL v4.1 Additive Clip `layer` & `effect` Statements (`ast_nodes.py`, `parser.py`, `grammar.lark`, Exporters)
- **`ClipLayerDecl`** (`layer <source_clip>: weight=<float>`):
  - Evaluated additively onto bone Euler/position poses; when layered onto a `gait` clip, secondary axial motion (`breathe`, `tail_sway`, `ear_flick`) is applied **before** closed-loop world limb IK so planted stance feet retain `< 0.02u` slip even as the torso heaves or pitches.
  - Cyclic `layer` references across clips are rejected deterministically by `validate_program`.
- **`EffectTriggerDecl`** (`effect <time_s | touchdown>: { sound: "...", particle: "...", locator: "...", instruction: "..." }`):
  - Explicit timestamps (`0 <= t <= duration_s`) emit effect triggers directly; `effect touchdown` expands deterministically across every `@morphology` limb's stance-start instants in a `gait` clip.
  - Exported natively to Bedrock `1.8.0` (`sound_effects` / `particle_effects`), GeckoLib (`sound_instructions` / `particle_instructions` / `custom_instructions`), and Blockbench `.bbmodel` (`animators["effects"]`).

### 2. Preserve-and-Layer Painted Texture Pipeline (`src/synccraft/shading/preserve_and_layer.py`)
- **`preserve_and_layer_painted_texture`**: Guarantees `0` modified pixels on `original_png`, emits separate transparent RGBA layers (`blank_fill_png`, `wear_layer_png`, `glow_layer_png`) plus non-destructive `variant_png` (`generate_oklch_texture_variant` in Ottosson OKLCH preserving achromatic outlines and binary alpha cutouts).
- **`rebake_selected_faces`**: Re-evaluates and writes pixels strictly inside the UV rectangles of targeted boxes/faces, preserving bit-identical pixels across all other faces.

### 3. Inverse Gait Fitting, Deterministic Search, & Role Retargeting (`src/synccraft/kinematics/gait_inversion.py`)
- **`invert_clip_to_gait`**: Samples any animated clip (`keyframes`, `tracks`, `ik_drives`, or `gait`), tracks body-relative and world-space limb contact trajectories (handling both root-motion clips and in-place treadmill animations), and recovers a validated `GaitDecl` (`pattern`, `froude`, `duty`, `phase`, `lift_ratio`, `heave_ratio`, `pitch_deg`, `spine_wave_deg`, `tail_lag`, `style`).
- **`search_or_repair_gait`**: Runs bounded, failure-directed coordinate search over `GaitDecl` parameters using 8-rule `gait_audit` diagnostics while charging every compiler and audit invocation to `CostLedger`.
- **`retarget_clip_across_models`**: Retargets locomotion and secondary axial motion across disparate morphologies (quadruped, biped, sprawled, hexapod) via `@morphology` roles and verifies the resulting clip.

### 4. 5-Task Proposer Supervision & Gate D Baseline Harness (`src/synccraft/world/proposer_tasks.py`)
- Synthesizes and validates structured text-in / typed-parameter-out records across `label_morphology_roles`, `propose_gait_from_style`, `repair_failed_gait`, `plan_texture_and_stamps`, and `invert_clip_to_gait`.
- Enforces strict **`body_plan_group` + `source_author`** split isolation (`verify_body_plan_and_author_split_isolation`) and benchmarks any proposer against **Fixed Presets** and **Bounded Deterministic Search** with full `CostLedger` accounting.

### 5. Blockbench 5 Plugin & Localhost Bridge (`editors/blockbench-synccraft/`, `src/synccraft/backends/blockbench_plugin_bridge.py`, `cli.py`)
- `synccraft serve-blockbench` binds strictly to `127.0.0.1` by default and serves deterministic JSON endpoints (`/health`, `/analyze`, `/check`, `/generate_locomotion`, `/retarget`, `/secondary_motion`, `/texture_pass`).
- `editors/blockbench-synccraft/synccraft_assistant.js` registers all 6 actions via `Plugin.register`, applies verified edits as ordinary `BoneAnimator` keyframes and non-destructive textures inside one `Undo` transaction, and keeps `runtime_import_verified = False` until exercised inside a live Blockbench GUI client.
