# Implemented Block-DSL v3.1 language subset (`ADR 0004` + `ADR 0006`)

This package implements the typed **box + semantic locator + stateful surface + musculoskeletal/kinematic/behavior** subset of SyncCraft v3 (`§2.1–2.4`, PDF pp. 3–6, extended by `ADR 0004` and `ADR 0006`). It operates cleanly on CPU without executing generated code, requiring a GPU, or calling black-box diffusion models.

## Available now

`parse_program(text)` returns a typed `Program`; `format_program(program)` serializes it canonically (`AST -> DSL -> AST` exact roundtrip). Both operate without a model, tokenizer, Lark runtime, GPU, or network. The source is in [src/synccraft/dsl/](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/). [grammar.lark](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/grammar.lark) documents the same grammar for later XGrammar/CFG integration; the runtime is the bounded recursive-descent parser in [parser.py](file:///Users/tianrui6641/SyncCraft/src/synccraft/dsl/parser.py).

Supported declarations:

| Declaration | Runtime behavior |
|---|---|
| `@meta` | Validates archetype/profile vocabulary and positive integer texel density. Rejects entities targeting vanilla Java item/block profiles. |
| `@palette` / `ramp` | Records OKLCH base color (`L in [0,1]`, `C >= 0`, `H` in degrees), steps, hue shift, and emissive flag. |
| `root bone` / `pivot_local` | Defines the single root bone and its world translation. |
| `bone` / `attach` / `rest_euler` / `joint_limits` | Creates child bones attached to a preceding direct-parent `box` or `locator` (`anchor=[0,0,0]` when attaching to a dimensionless locator). Rotation is intrinsic XYZ (`Rz Ry Rx` in parent frame); optional per-axis `joint_limits: [[rx_lo, rx_hi], [ry_lo, ry_hi], [rz_lo, rz_hi]]` clamp posed deltas during IK, muscle equilibrium, gaze tracking, and world interventions. |
| `box` / `place` / `cullface` | Strictly positive integer Box-UV dimensions; finite nonnegative inflation (`inflate` scales physical surface clearance without mutating integer UV sizes); explicit `place(...)` for non-first boxes in a bone. |
| `locator` | Semantic zero-geometry 3D anchor (`offset`, `role` in `{socket, vfx_emitter, rider_seat, hitbox, camera_focus, ik_effector}`) compiled through the exact SE(3) chain and reflected across `mirror_x`. |
| `mirror_x` (arbitrary depth) | Recursively expands mirrored subtrees across X (`<mirror_name>__<descendant>`), toggling $O(3)$ parity (`mirrored = not mirrored`) at each reflection depth so doubly-mirrored subtrees (e.g., symmetric wing digits inside mirrored wings) maintain $\det(R) = +1$ and exact east/west cullface parity. Bounded by `MAX_EXPANDED_MIRROR_NODES = 8192`. |
| `@materials` | Maps palette identifiers to reduced-order stateful material processes (`10` families, `roughness`, `metalness`, `coating`, `ior`, `scale`, `seed`, `anisotropy`, `subsurface`, `porosity`, `emissive`, Gray–Scott `rd_feed`/`rd_kill`, `film_nm`, `wear`, `oxidation`, `biofilm`, and cross-box `weld_group` seam coupling). |
| `@style` | Configures `target_encoding` (`vanilla`, `labpbr_1_3`, `gltf_2_0`), `quantize_ramp`, `bevel_highlight`, and `view_deg`. |
| `@muscles` | Declares Hill-type reduced-order muscle-tendon actuators (`origin_bone`, `insertion_bone`, `action_axis`, `moment_arm_deg`, optional biarticular `tendon_coupling: [bone, axis, ratio]`, and optional Box-UV-preserving `belly_box` + `bulge_inflate`). |
| `@kinematics` | Named `ik_chain` (`upper`, `lower`, `effector`, `hinge_axis`) and multi-bone `look_chain` (`bones`, `weight_decay`) descriptors. |
| `@behavior` | Named `clip` blocks (`duration_s`, `loop`, harmonic/damped `track`s, `muscle_drive` activations, and world-anchored $C^1$ `foot_cycle` `ik_drive`s) plus persistent `state` blocks (`pose_bias`, `muscle_tone`, `material_mod`). |

Palette names, bones, boxes, locators, mirror sibling names, generated mirror descendant names, muscles, IK/look chains, behavior clips, and behavior states share a single unique identifier namespace. Referenced materials, bones, boxes, muscles, and IK chains must be declared. Anchor and align components are finite and within `[-1, 1]`. A connected (`detached=false`) attachment permits offset components only in `[-0.5, 0.5]`. Lexical/semantic acceptance does not replace physical geometry or swept-pose SAT audits (`audit_scene`, `audit_behavior_clip`).

## Deliberately explicit limits

`extrude`, `@shade`, `@ergonomics`, and the PDF's truncated legacy `@motion` keyword raise `UnsupportedFeature` (`issubclass(UnsupportedFeature, DSLValidationError)`). They are never silently ignored, approximated with bounding boxes, or stripped from user assets.

1. Only the first direct box in a bone may omit `place`. Later direct boxes require explicit placement, preventing accidental stacking at the joint.
2. A `place` without `base_box` must have `anchor=[0,0,0]`. Its reference point is the bone pivot, and its center is `-0.5 * size * align + offset`.
3. Nested `mirror_x` expansion is pre-checked against `MAX_EXPANDED_MIRROR_NODES = 8192` before materializing clones to prevent exponential memory blowups on adversarial inputs.
4. Every bone must contain at least one direct `box` or `locator`, and `@rig` as a whole must contain at least one `box`. Negative rest inflation and non-box polygon extrusions remain deferred.
5. Input is capped at 2 MB UTF-8, 200,000 lexer tokens, 4,096 declarations, and bone depth 64. No `eval`, dynamic imports, or generated-code execution is used anywhere in the DSL or Molang animation decompiler.

## Remaining language milestones

Resolve extrusion occupancy/layer-overlap rules and material mapping (`extrude`); implement supported legacy `@shade` passes if needed beyond `@materials`/`@style`; add ergonomic hand/grip fit constraints (`@ergonomics`); and wire live tokenizer XGrammar masks + KV-cache statement checkpoints (`P1-07`).
