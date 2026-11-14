# Implemented Block-DSL v3 starter subset

This package implements the static **box** subset of the supplied v3 plan (§2.1–2.4, PDF pp. 3–6). It is an initial compiler foundation. It does not implement the complete published language or model-guided decoding.

## Available now

`parse_program(text)` returns a typed `Program`; `format_program(program)` serializes it canonically. Both operate without a model, tokenizer, Lark, GPU, or network. The source is in `src/synccraft/dsl/`. `grammar.lark` documents the same syntactic subset for later parser-engine integration; the runtime is the bounded recursive-descent implementation in `parser.py`.

Supported declarations:

| Declaration | Runtime behavior |
|---|---|
| `@meta` | Validates archetype/profile vocabulary and positive integer texel density. Rejects entities targeting vanilla Java item/block profiles. This does not certify an exporter for that profile. |
| `@palette` / `ramp` | Records OKLCH base color, steps, hue shift, and emissive flag. L is [0,1], C is nonnegative, hue is degrees. No claim that all shading/export flags are rendered. |
| `root bone` / `pivot_local` | Defines the one root and its world translation. |
| `bone` / `attach` / `rest_euler` | Creates child bones attached to a preceding direct-parent box. Rotation is X, then Y, then Z, represented as `Rz Ry Rx`; public DSL angles are degrees. |
| `box` | Strictly positive integer dimensions; finite nonnegative inflation in this starter. |
| `place` | Places another box relative to an earlier same-bone box, or directly relative to the bone pivot. |
| `mirror_x` | Reserves a generated sibling and descendant IDs for deterministic geometry expansion. |
| `cullface` | Preserves valid face labels. Exporters remain responsible for capability checks. |

Palette names, bones, boxes, mirror sibling names, and generated mirror descendant names all share a unique identifier namespace. Referenced materials must be declared. A child's `parent_box` must already exist in its immediate parent bone. A `place.base_box` must precede the placed box in that same bone. An ancestor or future declaration is not a legal substitute.

Anchor and align components are finite and within [-1,1]. A connected (`detached=false`) attachment permits offset components only in [-0.5,0.5]. Geometry still has to be compiled and audited: lexical/semantic acceptance does not prove valid contacts or legal target world bounds.

## Deliberately explicit limits

`extrude`, `@shade`, `@ergonomics`, and `@motion` raise `UnsupportedFeature`. They are not ignored, approximated with a bounding box, or reported as supported. Do not remove these sections from a user asset automatically. A separate simplified fixture is acceptable when clearly named and explained.

The supplied PDF visibly truncates the motion grammar at `spring_step` on p. 4; this package does not reconstruct that missing specification as though it were authoritative. The learned-refiner code is also truncated. Resolve those interfaces before their implementation milestones.

The parser applies these conservative starter rules where the plan leaves behavior ambiguous:

1. Only the first direct box in a bone may omit `place`. Later direct boxes require explicit placement, preventing accidental stacking at the joint. This is stricter than the PDF's optional-clause grammar.
2. A `place` without `base_box` must have `anchor=[0,0,0]`. Its reference point is the bone pivot, and its center is `-0.5 * size * align + offset`. A dimensionless nonzero anchor has no defined parent extent in the report.
3. Mirror clones use the specified sibling name. Each descendant ID becomes `<mirror_name>__<original_descendant_name>`. These names are reserved before geometry expansion. A nested `mirror_x` inside a mirrored subtree is explicitly deferred.
4. Every bone must contain at least one direct box. Negative inflation and non-box primitives are deferred.
5. Input is capped at 2 MB UTF-8, 200,000 lexer tokens, 4,096 declarations, and bone depth 64. These are parser bounds, not promises that quadratic collision audits scale to the maximum.

`DSLParseError` includes line/column information for lexical and syntactic errors. `DSLValidationError` identifies a complete-declaration semantic failure. `UnsupportedFeature` is a subclass of `DSLValidationError`. No `eval`, dynamic imports, or generated-code execution is used.

Direct Python AST construction follows the annotations: vectors and face lists are tuples, while palette and bone-item collections are lists. Booleans are not numeric coordinates or dimensions. Integers outside the finite float64 geometry range fail validation. Canonical serialization normalizes floating fields to the same representation whether the direct AST used integer or floating literals.

## Examples and handoff

`examples/simple_blade.blockdsl` demonstrates a root, two same-bone boxes, and nested attached bones. `examples/mirrored_wings.blockdsl` adds a 22.5-degree joint and a mirrored subtree. `examples/stacked_block.blockdsl` demonstrates a three-box connected pedestal. These fixtures are original procedural shapes and use no imported game art.

The current tests include 500 seeded AST→DSL→AST cases; this validates this subset's parser/serializer. It is **not** the plan's 500-program multi-exporter schema/real-application acceptance gate, which requires extrusion and target-version integration work.

Next language tasks: resolve the extrusion occupancy/layer-overlap rules and material mapping; implement `extrude` with deterministic voxelization and merging; freeze a complete motion grammar; implement supported shade passes before accepting `@shade`; add ergonomic fit constraints before accepting `@ergonomics`; then integrate the selected tokenizer's XGrammar and statement checkpoints. The grammar file alone does not provide any of those decoding guarantees.
