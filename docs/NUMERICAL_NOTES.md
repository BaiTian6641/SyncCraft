# Numerical review and acceptance corrections

**Scope:** engineering review of the supplied SyncCraft v3.0 report, especially
sections 3.2 and 6.3, against the executable starter. The numerical results below
were reproduced with this repository's NumPy implementation. They do not report
trained-model, PyTorch-gradient, PGD, or engine-import results.

## 1. Raw SAT and soft minimum answer different questions

For each valid unit candidate axis `l`, the report defines projected overlap

```text
d(l) = sum_i eA_i * abs(dot(A_i,l))
     + sum_j eB_j * abs(dot(B_j,l))
     - abs(dot(centerB-centerA,l)).
```

For nondegenerate boxes, positive overlap on every SAT candidate axis means
intersection. The CPU audit uses `m = min(d(l))`, with a numerical tolerance.
Its reported penetration is an axis distance, **not intersection volume**.

For `K` valid axes, the report's unnormalised smooth minimum satisfies

```text
m - log(K)/beta <= softmin_beta(d) <= m.
```

It is a lower bound. A true small positive intersection can therefore have a
negative smooth minimum and zero squared-hinge energy. Decreasing the hinge's
`0.01` tolerance alone does not remove this blind region.

### Reproduced corner-overlap counterexample

Use two axis-aligned boxes, both of size `[2,2,2]`, with centers `[0,0,0]` and
`[1.99,1.99,1.99]`. The implementation has 12 valid candidate axes after omitting
three zero crosses; repeated directions are retained. Every axis overlap is
`0.01`, to floating-point rounding.

| Quantity | Result |
| --- | ---: |
| Raw minimum overlap | `0.010000000000000009` |
| Smooth minimum, beta 25 | `-0.08939626599152001` |
| Report's squared-hinge surrogate, tolerance 0.01 | `0.0` |
| Raw SAT collision decision | `true` |

This case is covered by
`tests/test_sat.py::test_softmin_can_miss_positive_overlap_audit_does_not`.
The starter correctly fails the non-exempt collision audit.

**Acceptance requirement:** preserve an independent raw-SAT final audit. Compare
candidate optimization losses on adversarial shallow overlaps as well as ordinary
assemblies. Do not mark success from energy convergence alone. Adding `log(K)/beta`
to normalize softmin produces a different bound and can lose zero-contact
behavior; it is not an automatic repair. Changing the loss is an engineering
decision requiring a recorded ablation and explicit contact tests.

## 2. Degenerate cross axes must be excluded

SAT tests six face normals plus nine edge cross products. For parallel edges,
the cross product is zero and provides no separating direction. Dividing that
zero vector by a clamped norm still gives zero; its projected overlap is then
zero and can incorrectly suppress every real intersection in a minimum-based
test. Identical parallel boxes are a direct regression case.

The starter omits crosses whose norm is at most `1e-10`. Face normals remain.
It validates orthonormal, determinant-`+1` rotations. A future batched GPU version
must mask invalid axes out of minima/log-sum-exp; replacing them with zero is
incorrect. Slot masks and pair masks are separate from axis validity masks.

The number of retained axes changes near parallel configurations; the surrogate's
log-sum-exp bias can change with that count. This matters for optimization and
autograd continuity, even when the raw collision decision remains stable. Record
the chosen axis/masking policy and test perturbations around degeneracy.

## 3. Tolerances and zero-gradient claims

| Quantity | Starter default | Meaning |
| --- | ---: | --- |
| Audit collision tolerance | `1e-7` modeling units | An overlap at or below this threshold is not counted as a collision. |
| Cross-axis norm threshold | `1e-10` | Omit an unreliable direction before normalization. |
| Surrogate hinge tolerance | `0.01` modeling units | Planned optimization dead zone, separate from final audit. |
| Smooth-min sharpness | `25.0` | Approximation parameter, not a correctness tolerance. |

The current CPU implementation is float64. Its tests include 1,000 deterministic
same-orientation rotated flush contacts. Those tests check returned energies and
decisions; they do **not** check derivatives. A universal claim of exact real
arithmetic or zero penetration volume would exceed their evidence.

The complete potential is not automatically differentiable everywhere: absolute
values, coincident projections, axis validity changes, and clipping introduce
nonsmooth cases even if softmin and the squared hinge are smooth in their own
scalar arguments. PyTorch can provide useful selected subgradients, but they must
be tested. Require:

1. NumPy/PyTorch forward parity in float64 and float32.
2. Finite-difference or `gradcheck` coverage away from nonsmooth points.
3. Explicit finite-gradient and intended zero-gradient behavior at flush contact,
   repeated axes, coincident centers, mirrored pairs, and masked empty slots.
4. Tolerances justified at representative coordinate magnitudes and precisions.
5. Recompilation and a new raw-SAT audit **after** final two-decimal quantization.

## 4. Declared adjacency can hide both penetration and gaps

**This is a blocker for production audit scores and Gate G2 evaluation.** The
starter follows the report's declared attach/place exclusions and reports every
excluded pair's depth, but it does not yet validate a physical contact region.
Its `passed` field concerns the stated starter scope only, and
`complete_for_release` remains false.

Two executed examples show why this distinction is necessary. Start with two
`[20,20,20]` boxes and place the second using:

```text
place(base_box=base, anchor=[0,0,0], align=[0,0,0], offset=[D,0,0])
```

| D | Physical geometry | Current audit | Declared graph |
| ---: | --- | --- | --- |
| `0` | Coincident boxes; SAT depth `20.0` | `passed=true`; excluded collision remains in the report | Connected |
| `50` | Boxes separated by a 30-unit gap | `passed=true`; excluded raw overlap `-30.0` | Connected |

Both are valid current DSL. A declared parent/base relationship does not establish
that a surface joint exists, nor authorize unlimited intersection. Restricting
connected attach offsets to `[-0.5,0.5]` is insufficient: zero offsets and interior
anchors already allow the first failure. A graph of declaration edges cannot prove
that an object is physically connected or supported.

### Required contact-policy milestone

Before reporting an audit pass rate, implement a typed, explicit contact policy:

- Define surface contacts, permitted embedded joints, detached attachments, and
  permitted clearance with bounded geometric tolerances. Preserve deliberately
  embedded details through an explicit mode or contact-region representation.
- Validate transformed anchor residuals, face/region relationships, allowed depth,
  and permitted contact extent. Anchor coincidence alone is insufficient for two
  coincident boxes. Evaluate intended contacts in the actual rotated geometry.
- Exempt only the permitted contact region/overlap, or use a conservative policy
  that rejects cases the audit cannot certify. Do not waive an entire pair solely
  because it has an attach/place declaration.
- Form physical connectivity from validated contacts; keep the declaration graph
  as separate diagnostic data. Detached structures need explicit intent.
- Reject the coincident and far-gap examples above, while accepting intended flush
  joints, bounded embedded details, mirrors, and authorized detached elements.
- Add cases where a child joint is valid but another region of the same two boxes
  interpenetrates. Do not add blanket sibling or mirror-root exemptions to make
  demonstrations pass.

The local agent should register this as a high-priority Phase 1 audit task. A
benchmark that uses the current starter's `passed` flag as full asset validity
would be systematically optimistic.

## 5. Twenty PGD steps are a fixture target, not a universal guarantee

The report requests eliminating a 0.2-unit overlap within 20 steps without
breaking contacts. Success depends on the permitted degrees of freedom, integer
dimensions, offset bounds, Euler constraints, target bounds, and surrounding
parts. Some assemblies have no feasible correction within those restrictions;
others are nonconvex and may trap a local solver. Neither adding iterations nor
changing the success tolerance proves feasibility.

Make the acceptance case precise: commit a fixed feasible fixture, its editable
variables, initial overlap, legal solution, bounds, optimizer configuration, and
seed. Measure its raw collision/contact/bounds audit before and after projection.
Separately include infeasible and difficult cases; they must return a documented
failure or an exhausted budget, without silently changing dimensions, unlocking
frozen legacy angles, breaking contacts, or weakening the audit.

Failure under the benchmark budget does not by itself prove a local-minimum
bottleneck. Rule out implementation errors, false contact exclusions, impossible
constraints, and inconsistent budget accounting before interpreting Gate G2 as
evidence in favor of a learned refiner. Keep the full failure denominator, count
re-audits and discarded generation, and freeze the held-out fixture families.

## Evidence boundary

- Source requirement: supplied report sections 3.2 and 6.3.
- Implemented conventions: [GEOMETRY_CONVENTIONS.md](GEOMETRY_CONVENTIONS.md).
- Executable evidence: `src/synccraft/kinematics/sat_potential.py`,
  `src/synccraft/audit/sync_audit.py`, and `tests/test_sat.py`.
- Contact-region validation, differentiable parity, and constrained PGD remain
  follow-up work; this note does not mark those gates complete.
