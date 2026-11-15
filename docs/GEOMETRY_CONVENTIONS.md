# Shared geometry contract (`Block-DSL v3.2`)

This document defines the shared geometry, reflection, Backend handedness, and collision/contact contract across SyncCraft's NumPy reference engine, PyTorch differentiable kinematics/SAT modules, exporters, decompilers, and `audit_scene`. Compiler, preview, exporters, and audit all evaluate the same shared `Scene`.

## Coordinates, order, and units

- **Internal Frame**: Right-handed coordinates (`+X` right/east, `+Y` up, `+Z` south/front), column vectors.
- **Rotations**: DSL Euler angles are degrees; `euler_xyz_matrix` composes intrinsic XYZ (`R = Rz @ Ry @ Rx`), converting degrees to radians strictly at named trigonometric boundaries.
- **Hierarchy**: A root `pivot_local` is a world translation (canonicalized to `[0, 0, 0]` in clean-room procedural generators). A child pivot is measured in its direct parent bone's coordinates (`parent @ translation @ rotation`).
- **Dimensions, `size_step`, `plane`, and per-box `rotate`**:
  - Clean-room procedural static/Apex assets use positive integer box sizes (`@meta.size_step = 1`); at `density = 16`, one modeling unit equals one UV texel.
  - Authored/imported assets (`Block-DSL v3.2`) may declare `@meta { size_step: 0.5 | 0.25 | 0.125 }` for sub-unit fractional dimensions, zero-thickness `plane` elements for flat fins/ears/foliage quads, and optional per-box local `rotate` / `pivot` offsets without mutating parent bone frames.
  - Continuous `inflate` (plus dynamic muscle `bulge_inflate`) expands OBB half-extents $\mathbf{h} = \frac{1}{2}\mathbf{s} + \delta\mathbf{1}_3$ without mutating nominal UV layout dimensions $\mathbf{s}$.
- **Placement**:
  - The root's first box without `place` centers at its bone pivot. A child's first box without `place` has local center `-0.5 * size * attach.align`.
  - Same-bone `place(base_box=...)` has local center `base.center + 0.5 * base.size * anchor - 0.5 * size * align + offset`.
  - Without `base_box`, `anchor` must be `[0, 0, 0]` relative to the bone pivot.
  - Child `attach(parent_box=...)` places the joint at `base.center + 0.5 * base.size * anchor + offset` (or at the parent `locator` offset when attaching to a dimensionless 3D socket with `anchor=[0, 0, 0]`).

## Mirror & radial expansion (`mirror_x`, `mirror`, `radial`)

- `mirror_x(right)` duplicates a child bone and its entire descendant subtree across X, supporting **arbitrary nesting depth** up to `MAX_EXPANDED_MIRROR_NODES = 8192` (`outer_r__inner_r__leaf`).
- Reflection conjugates rotations by $\mathbf{S} = \operatorname{diag}(-1, 1, 1)$ (`(rx, ry, rz) -> (rx, -ry, -rz)`, X components of anchors/offsets/locators negated) so every compiled OBB frame remains a proper right-handed rotation with $\det(\mathbf{R}) = +1$.
- Reflection parity composes in $\mathbb{Z}_2$ (`mirrored = parent.mirrored ^ is_mirror_clone`): odd reflection depths swap `east <-> west` cullfaces, while doubly-mirrored subtrees ($\mathbf{S}^2 = \mathbf{I}_3$) restore original handedness.
- v3.2 sugar constructs (`chain`, `mirror`, `radial`, `let`, `def`/`use`) lower deterministically into explicit `SE(3)` bone/box nodes before audit and export.

## Bedrock left-handed X-axis conversion (`C5` / `M16`)

Minecraft Bedrock `1.12.0` `.geo.json` and `1.8.0` `.animation.json` use a left-handed X convention relative to SyncCraft's right-handed space:
- **Outbound (`export_bedrock` / `export_geckolib`)** and **Inbound (`_decompile_bedrock_geojson` / `decompile_bedrock_animation_to_behavior`)** apply the exact involution:
  - Cube origin X: `origin_x_bedrock = -(origin_x_rh + size_x)`
  - Bone pivot X and locator X: negated (`-x`)
  - Rest and keyframe Euler rotations: X and Y negated (`(-rx, -ry, rz)` in Bedrock convention so asymmetric rotated rigs reconstruct all 8 world corners to `< 1e-4`)
  - Animation bone position X: negated (`-tx`)

## SAT: exact audit versus smooth surrogate

- `sat_obb` (NumPy) and `torch_sat` (PyTorch) test all 6 face normals plus 9 pairwise edge cross products, masking near-parallel cross products ($\|\mathbf{u}_i \times \mathbf{v}_j\|_2 \le 10^{-8}$) rather than normalizing degenerate axes to zero.
- Final acceptance always uses **direct raw minimum axis overlap** (`collision_tolerance = 1e-7`). Returned `penetration_depth` is a linear projection distance, never an intersection volume.
- Smooth `softmin` / LogSumExp potentials are optimization surrogates only: **zero smooth surrogate loss is never interpreted as zero overlap**. Every PGD/repair candidate and quantized/serialized export undergoes a fresh direct raw-SAT re-audit.

## Physical contact profiles (`P1-02` vs `authored`)

Blanket parent/sibling collision immunity is prohibited:
1. Only explicitly declared non-detached `attach` pairs and `place(base_box=...)` pairs enter `Scene.adjacency` (`declared_contact_pairs`), and every exempted pair still logs its raw SAT depth.
2. **`audit_profile="P1-02"` (default for clean-room generation & Gate G2)**: Enforces positive coplanar contact-patch area (`>= 1e-4` sq units) or bounded hinge inset (`<= 0.5` units without burying box centers), zero floating gaps (`< -1e-4`), and full physical connectivity across non-detached subtrees.
3. **`audit_profile="authored"` (for imported human-authored rigs & Level-7 Human Transfer)**: Audits non-exempt collisions, finite valid transforms, and tree integrity while accommodating intentional artistic hinge insets and floating decorative sub-elements common in community `.bbmodel`/`.geo.json` rigs.

## Current status vs remaining milestones

- **Implemented on CPU**: NumPy & PyTorch `SE(3)` + 15-axis SAT parity checks, `P1-02` & `authored` contact audits, Baseline-PGD projector (`pgd_projector.py`), and `run_bounded_repair_policy` (`<= 3` re-audits with monotonic `CostLedger` accounting).
- **Remaining before release claims**: Hand/grip ergonomics (`@ergonomics`, `P1-04`), live GUI/engine runtime import verification (`P0-02`, `P3-02`), and empirical GPU Gate G2 evaluation on the frozen unseen dense-assembly holdout (`P1-10`).
