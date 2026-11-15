# Implemented Block-DSL v3.2 language subset (`ADR 0004` + `ADR 0006`)

This package implements the typed **constructive geometry + macro/symmetry sugar + stateful surface + Zajac/Hill musculoskeletal/kinematic/behavior** language subset of SyncCraft v3 (`§2.1–2.4`, PDF pp. 3–6, extended by `ADR 0004` and `ADR 0006`). It operates deterministically on CPU without executing generated code, requiring a GPU, or calling black-box diffusion models.

## Available now

`parse_program(text)` returns a typed `Program`; `format_program(program)` serializes it into compact canonical form (`AST -> DSL -> AST` exact roundtrip, omitting default noise such as `offset=[0,0,0]`, `detached=false`, and `inflate: 0.0`). The source is in [src/synccraft/dsl/](../src/synccraft/dsl/). [grammar.lark](../src/synccraft/dsl/grammar.lark) maintains verified EBNF parity with the bounded recursive-descent parser in [parser.py](../src/synccraft/dsl/parser.py).

Supported declarations:

| Declaration | Runtime behavior |
|---|---|
| `@meta` | Validates archetype/profile vocabulary, positive integer texel `density`, and optional fractional `size_step` (`1`, `0.5`, `0.25`, `0.125`) for authored assets. Rejects entities targeting vanilla Java item/block profiles. |
| `@palette` / `@textures` | Records OKLCH/OKLab base colors (`ramp`, steps, hue shift, emissive flag) and optional named `@textures` bindings for multi-texture authored rigs. |
| `let` / `def` / `use` | Compile-time numeric/vector bindings (`let`) and parameterized reusable sub-assembly templates (`def` / `use`) expanded deterministically without `eval`. |
| `root bone` / `pivot_local` | Defines the root bone and its world translation (`[0, 0, 0]` canonical default). |
| `bone` / `attach` / `rest_euler` / `joint_limits` / `spring` | Creates child bones attached to a parent `box` or `locator` (`Rz Ry Rx` intrinsic XYZ rotation), optional `joint_limits`, and optional secondary `spring` follow-through parameters. |
| `box` / `plane` / `place` / `rotate` / `cullface` | Box primitives (integer or `size_step`-aligned dimensions, nonnegative `inflate` separate from UV size), zero-thickness `plane` quads for flat fins/ears, optional per-element local `rotate`, and `place(...)` relative to a base box or bone pivot. |
| `chain` / `mirror` / `radial` / `mirror_x` | Structured serial bone chains (`chain`), bilateral/planar reflections (`mirror`, arbitrary-depth `mirror_x` with $\mathbb{Z}_2$ parity toggling bounded by `MAX_EXPANDED_MIRROR_NODES = 8192`), and $N$-fold rotational arrays (`radial`). |
| `locator` | Semantic zero-geometry 3D socket (`offset`, `role` in `{socket, vfx_emitter, rider_seat, hitbox, camera_focus, ik_effector}`) compiled through the shared `SE(3)` chain. |
| `@materials` & `@style` | Maps palette entries to 10 stateful surface material families (`roughness`, `metalness`, `coating`, `ior`, Gray–Scott `rd_feed`/`rd_kill`, `film_nm`, `wear`, `oxidation`, `biofilm`, cross-box `weld_group`) and target encoding (`vanilla`, `labpbr_1_3`, `gltf_2_0`). |
| `@muscles` | Declares reduced-order Zajac/Hill muscle-tendon units with active force-length + passive parallel-elastic curves (`fidelity_label == "reduced_order_qualitative"`), `moment_arm_deg`, optional biarticular `tendon_coupling`, and Box-UV-preserving `belly_box` + `bulge_inflate`. |
| `@kinematics` & `@controller` | Named 2-bone `ik_chain`, multi-bone CCD/gaze `look_chain`, and state-machine `@controller` transitions. |
| `@behavior` | Named `clip` blocks (`duration_s`, `loop`, harmonic/damped `track`s, explicit `key` keyframes, `muscle_drive` activations, and world-anchored $C^1$ `foot_cycle` `ik_drive`s with `< 0.02u` stance slip) plus persistent `state` blocks. |

## Deliberately explicit limits

`extrude`, `@shade`, `@ergonomics`, and the PDF's truncated legacy `@motion` keyword raise `UnsupportedFeature` (`issubclass(UnsupportedFeature, DSLValidationError)`). They are never silently ignored or approximated with bounding boxes.

1. Only the first direct geometry element in a bone may omit `place`; subsequent elements require explicit placement.
2. `place` without `base_box` requires `anchor=[0, 0, 0]` relative to the bone pivot.
3. Nested `mirror_x` expansion is pre-checked against `MAX_EXPANDED_MIRROR_NODES = 8192` before cloning.
4. Input is capped at 2 MB UTF-8, 200,000 lexer tokens, 4,096 declarations, and bone depth 64. Zero `eval()`, `exec()`, or dynamic code execution exists anywhere in `src/synccraft/`.

## Remaining language milestones

Implement 2D contour marching-squares polygon extrusion (`extrude`, `P1-01`), ergonomic grip/display constraints (`@ergonomics`, `P1-04`), and live tokenizer XGrammar token masking with KV-cache statement rollback (`P1-07`).
