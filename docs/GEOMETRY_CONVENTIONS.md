# Shared geometry contract

This document defines the executable **starter subset**, not a completed
differentiable engine. Its NumPy implementation is the reference for future
PyTorch parity tests. Compiler, preview, and audit all use the same `Scene`.

## Coordinates, order, and units

- Right-handed coordinates, Y up, column vectors.
- DSL Euler angles are degrees; `euler_xyz_matrix` applies X, then Y, then Z:
  `R = Rz @ Ry @ Rx`. Trigonometry converts to radians internally.
- A root pivot is a world translation. A child pivot is measured in its direct
  parent bone's coordinates. World transforms compose `parent @ translation @ rotation`.
- Box sizes remain positive integer modeling units. At density 16, one modeling
  unit corresponds to one texture pixel. `inflate` changes half extents only,
  without changing attachment/placement dimensions.
- The root's first box has local center zero. A child's first box without
  `place` has local center `-0.5 * size * attach.align`.
- A same-bone `place(base_box=...)` has center
  `base.center + base.size/2 * anchor - size/2 * align + offset`.
- Without `base_box`, the reference is the bone pivot. The starter requires
  `anchor=[0,0,0]`, avoiding an undefined anchor scale.
- An attach uses `base.center + base.size/2 * anchor + offset` for the child
  joint. Attachment alignment positions its primary box in the child's local
  coordinates before applying the child rotation.

## Mirror expansion

`mirror_x(right)` duplicates a child bone and its subtree. The duplicate root is
named `right`; descendants use `right__original_name`. Nested mirror declarations
are currently rejected. Material identifiers retain their names.

The mirror plane is the attached parent box's local X center plane. The root
attachment keeps the parent box center and reflects X anchor/offset components.
Every subtree local center, attachment vector, and placement vector reflects X;
Euler angles map `(x,y,z)` to `(x,-y,-z)`. This uses `S R S`, `S=diag(-1,1,1)`,
so each exported OBB orientation remains a proper rotation with determinant +1.
It reflects geometry; texture handedness needs separate exporter/UV acceptance.
This parent-box-centered convention is an explicit resolution of the report's
implicit mirror frame, and must be preserved in the PyTorch implementation.

## SAT: exact audit versus optimization surrogate

`sat_obb` tests the two boxes' six face normals and their nine pairwise edge
cross products. It drops numerically degenerate cross products rather than
normalizing them to zero. Including a zero candidate would incorrectly make
identical parallel boxes appear collision-free.

The deterministic audit uses the **raw minimum axis overlap** with a `1e-7`
distance tolerance. Returned `penetration_depth` is a distance, never an exact
intersection volume. Touching and separated boxes return zero depth/energy.

The planned `softmin(beta=25)` surrogate is also reported, with the report's
`energy_tolerance=0.01`. Its unnormalised log-sum-exp is a lower bound on the
minimum; it can be negative when all exact axis overlaps are slightly positive.
Consequently, **zero surrogate loss is not evidence of zero collision**. Every
projection result, including rounded output, requires a fresh raw-SAT audit.
The test suite contains a corner-overlap counterexample proving this distinction.

## Declared-contact exclusions

Only these pairs enter the starter `Scene.adjacency` set:

1. A non-detached attach's parent box and the child bone's first box.
2. A placed box and its explicitly named same-bone base box.

All other pairs are audited, including ordinary siblings and mirrored siblings.
The report mentions an additional symmetric-root exemption; it is deliberately
not inferred here because no unambiguous AST marker identifies those intended
contacts. Add an explicit validated contact relation before broadening masks.

Every excluded pair still has its raw SAT depth reported under
`declared_contact_pairs`. Declared contact is not a physical guarantee: a gap,
deep intentional overlap, or unsupported construction needs a separate contact
region audit. `connected_components` is the declared-edge graph only.

## Scope and required follow-up

The starter audit exposes `scope=starter_static_geometry` and
`complete_for_release=false`. It has no hand/ergonomics, engine/schema, exact
intersection volume, texture seam, motion, or foot-skating acceptance.

The CPU regression tests include 1,000 rotated same-orientation flush contacts,
positive 0.2 overlaps, parallel degeneracy, inflation, hierarchy composition,
and reflected geometry. They do not satisfy the plan's differentiable gradient
gate or 20-step PGD claim. Before marking that gate complete:

1. Port this transform contract to PyTorch without detaching tensors.
2. Check forward parity and finite-difference gradients for nondegenerate cases.
3. Verify zero gradients at flush contact across relevant precision settings.
4. Optimize only valid pose variables, preserve integer sizes/contact constraints,
   and test world-bounds penalties for each backend.
5. Recompile and rerun exact SAT after two-decimal output quantization.
6. Report the failed optimization cases rather than weakening acceptance criteria.
