# Implementation backlog

Task status is also indexed in `TASKS.json`. IDs and dependencies are stable.
The source's week numbers are planning targets, not elapsed-time guarantees. Work
in dependency order and update evidence links when a task passes acceptance.

## Recommended execution order

First reproduce S0, pin the local environment (P0-01), and establish actual target
roundtrips (P0-02). Then close the static language and physical-audit gaps (P1-01,
P1-02). UV, grip, Torch geometry, and model readiness can proceed in parallel after
their relevant contracts are fixed. Complete a trustworthy pilot dataset before
the 8K candidate bake-off. Scale training only after correct labels and real
decoder behavior exist. G2 uses the completed pipeline, not the starter audit.

## S0 — Reproduce the delivered starter

Status: delivered; local reproduction required. Dependencies: none.

Run the packaged CPU tests and demos. Compare logs with `artifacts/validation/` and
record any environment-dependent difference. Inspect the delivered preview and
`.bbmodel` examples. Preserve the source PDF and its SHA-256 provenance.

Acceptance: CPU suite and three demos succeed locally; unsupported sections produce
clear errors; `complete_for_release` remains false. Do not change pending phases to
done based on this result.

## P0-01 — Establish the actual development environment

Status: pending. Dependencies: S0.

Record Python, OS, architecture, GPU, driver/runtime, RAM/VRAM, Blockbench version,
Java target version, and any Bedrock/GeckoLib versions to be supported later. Choose
separate CPU and GPU environments. Resolve and save real dependency locks rather
than treating `requirements-core.txt` as a complete ML lock. Set up a CI job for the
CPU suite using verified current actions/runners and a Python 3.11/3.12 matrix.

Acceptance: fresh environment installation succeeds; CPU tests pass; GPU availability
is accurately labeled. A missing GPU does not block compiler work. Write
`artifacts/local_environment.json` with versions, not credentials.

## P0-02 — Validate real target imports and freeze compatibility fixtures

Status: pending. Dependencies: P0-01.

Pin Blockbench source/version to the installed editor. Import, save, and reopen
three originals plus a nested multi-axis fixture, inflation fixture, and asymmetric
face-color fixture. Preserve pivots/hierarchy and world geometry. Create an isolated
minimal Java resource pack with the correct model, texture and dispatch layout for
the selected game version; compare logs and rendered result. Keep resource pack
fixtures separate from the compiler's individual-model exports.

Acceptance: target logs and before/after assets show no missing textures or broken
rotations. Numeric reconstruction is independent of exporter helper functions.
Mirror direction and face orientation are inspected with asymmetric markings.
Fix serializers before admitting generated dataset labels. Record unsupported
versions explicitly. No game/runtime claim from JSON parsing alone.

## P0-03 — Resolve candidate models and run the first five smoke checks

Status: pending. Dependencies: P0-01.

Resolve the report's three candidate labels to exact repositories/revisions. Read
license text, model configuration and tokenizer behavior. Do not guess the Spark
repository or download remote code unreviewed. Record six-check readiness and
actually run the first five checks where hardware is available: license/revision,
LoRA gradients, 2048-token checkpointed forward/backward, declaration-close hidden
state extraction, and tokenizer-specific grammar masking. Module targets must be
discovered in the actual model; check trainable parameter counts and nonzero grads.

Acceptance: immutable model manifests and actual smoke logs exist; failures and
remote-code requirements are documented. No global Transformers pin based on only
one candidate. Keep the proposed 4.57.1 Spark pin only if tested for that revision.
If hardware is unavailable, finish metadata/config work and leave GPU checks pending.

## P0-04 — Perform the comparable 8K candidate bake-off

Status: pending. Dependencies: P0-03, P1-07, P1-08.

Use identical audited 8,000 static examples, one epoch, three candidate-specific but
documented memory strategies. Freeze prompt/template, response masking, validation
split, generation limits and seeds. Measure first full compile rate in <=512-token
and 513–2048-token declaration/reference buckets. Define the report's Spark >3
percentage-point drop rule and its comparator before running it; avoid unequal
problem difficulty between buckets. Also record latency and peak memory.

Acceptance: choose one base using recorded results and licensing evidence, or
record why none passed. Produce `configs/selected_model.json` with exact revisions,
adapter shapes, actual parameter counts and measured memory. Do not insert invented
numbers from the report's estimate table.

## P1-01 — Complete static DSL and deterministic extrusion

Status: pending; box foundation implemented. Dependencies: P0-01.

Resolve region precedence, overlap, pixel-center polygon fill, polygon validation,
layer thickness/offset, and `place` against an extrusion aggregate in an ADR. Add
typed rectangle/polygon layers; deterministically rasterize and greedily merge
occupied cells into integer cuboids without changing the occupied volume. Preserve
semantic primitive IDs through one-to-many lowering and symbol references.

Finish necessary lexer/grammar productions and migrate to Lark only with parser
parity tests. Keep span mappings for statement checkpoints. Define nested mirror
behavior before enabling it. Recover or retype the full static example from PDF p6.

Acceptance: full static syntax roundtrips; 500 seeded static cases include boxes,
extrusions, placement and mirroring; occupied-voxel equivalence is checked against
an independent reference; bad polygons/ambiguous references fail. Export and import
the full static example after shading/grip tasks support its other sections.

## P1-02 — Replace broad contact exemptions with physical contact checks

Status: pending; high priority. Dependencies: P0-01.

Turn `attach` and `place` into explicit typed contact contracts: intended anchor
surface/region, permitted depth/gap, inflation allowance, and detachment semantics.
Measure transformed contact locations and normals. Audit excluded pairs for deep
penetration and separation. Build physical connectivity/support separately from
the declaration graph. Do not simply exclude every parent/child or sibling pair.

Required adversarial cases: fully coincident large boxes; a linked primitive moved
far away; rotated flush contact; intentional shallow gem insertion; inflated armor;
mirrored wings sharing a root; non-contact siblings intersecting. Preserve reported
raw depths and explain every permitted contact.

Acceptance: the two known false-pass fixtures in `NUMERICAL_NOTES.md` fail; permitted
contacts pass under documented limits; labels distinguish declared edges from
geometric contact. This task blocks full asset-validity metrics and G2.

## P1-03 — Implement production integer UV and analytic pixel shading

Status: pending; diagnostic atlas implemented. Dependencies: P1-01, P0-02.

Implement target-appropriate Box UV/per-face mapping, deterministic packing,
integer texel density, palette ramps, short-radius cavity AO, edge highlights,
seam stripes and pixel stamps. Establish correct target lighting conventions and
emissive output behavior. Define all shader parameter bounds. Retain base colors
without directional light baked twice. Implement mirrored directional U/V and
face mappings with marked-face fixtures; a flat atlas cannot verify them.

Acceptance: exact integer UVs and no overlapping islands unless intentionally
shared; six-face labels/arrow patterns survive target import; cross-seam strokes
agree; deterministic texture hashes; engine lighting inspected. No silent loss of
palette `steps`, `hue_shift`, `emissive`, or requested shade passes.

## P1-04 — Add grip ergonomics, display and target constraints

Status: pending. Dependencies: P0-02, P1-02.

Parse `@ergonomics`. Build original Steve/Alex-sized analytic hand proxies with
documented frames, without copying training assets. Solve selected grip and
first/third-person/GUI display transformations with explicit parameter bounds.
Separate target engine restrictions from conservative project policies. Do not
infer valid hand fit solely from a primitive named `grip`.

Acceptance: grip contact/collision, GUI-fill and display fixtures have numeric and
visual checks; transforms roundtrip into both hands and target views; unavailable
target poses return an explicit capability error. Re-audit final serialized values.

## P1-05 — Build differentiable shared kinematics and SAT

Status: pending; NumPy reference implemented. Dependencies: P1-01, P1-02.

Implement Torch equivalents from the same graph, unit conventions and mirror
decisions. Cross-check dense/batched/masked forward results against NumPy. Avoid
host conversions that detach gradients. Mask degenerate axes and invalid slots
before reductions. Define hinge/min/absolute-value subgradients and precision.

Acceptance: 1,000 constructed flush fixtures satisfy the specified value/gradient
tolerances; generic non-boundary derivatives match finite differences; all-masked
batches have finite zero contribution; mirrors and rotated ancestors receive
correct gradients. Exact audit still detects shallow overlaps missed by softmin.

## P1-06 — Implement constrained Baseline-PGD

Status: pending. Dependencies: P1-05, P1-04.

Optimize only allowed offsets/Euler variables; keep sizes, topology and materials
fixed. Project connected offsets into their bounds; freeze legacy rotations.
Implement bounds, hand and regularization terms with documented scales. Compare
a simple deterministic optimizer before tuning more complicated schedules. Track
best feasible iterates, report nonconvergence, and charge every gradient step and
re-audit. Quantize to the specified two decimals and recompile/re-audit.

Acceptance: published feasible 0.2-unit overlap fixtures converge within 20 steps
without violating contacts or bounds; infeasible examples fail explicitly. Record
16/32-step distributions on validation families. Refiner outputs are actually
applied to the AST and exported; no dropped pose-update result.

## P1-07 — Implement real constrained decoding and transactional rollback

Status: pending. Dependencies: P1-01, P0-03.

Compile the correct tokenizer's XGrammar and dynamic symbol masks. Identifiers may
cross tokens; test trie completion and newly declared/removed symbols. Detect
statement boundaries from parsed token spans. Snapshot and restore KV, tokens,
grammar/AST, symbols, geometry and RNG consistently. Inject repair feedback with
correct prefix/cache invalidation. Enforce retry limits and monotonic cost accounting.

Acceptance: real model decoding respects syntax and declared references; injected
out-of-bounds and flattened-legacy-rotation violations roll back; no stale cache or
symbol survives; over-budget runs terminate and remain in the denominator. Record
accepted and discarded tokens separately. A scripted fake model is useful for unit
tests but does not satisfy real-model acceptance.

## P1-08 — Build audited original data generators and frozen splits

Status: pending. Dependencies: P1-01, P1-02, P1-03, P1-04, P0-02.

Begin with 200 examples and their provenance before scaling to 8K/60K. Implement
seen-family templates with combinatorial structure and parameter ranges. Freeze
heldout family/body-plan IDs before generation; prevent template, prompt and asset
leakage. Keep validation separate from G2 and the final external benchmark. Record
compiler/auditor/generator versions in each row; review outliers and failure tails.

Acceptance: reproducible pilot shard; schema/provenance checks; no duplicate hashes
across splits; no heldout topology source in training; every training label has the
completed audit. Produce large shards only after the pilot is correct. The 800-icon
CC0 collection mentioned in the report must be sourced/reviewed, not fabricated.

## P1-09 — Implement and validate resumable LoRA training

Status: pending. Dependencies: P0-04, P1-08.

Instantiate the selected backbone and actual LoRA modules. Use response-only
supervision and a version-verified chunked fused linear cross entropy; compare its
loss and gradients with an ordinary small-batch reference. Start seq=2048,
microbatch=2, accumulation=64, r=32, three seeds. Tune learning rate/optimizer on a
pilot validation set, then freeze the run recipe. Measure memory instead of promising
the report's 14.5–18GB figures.

Implement complete atomic checkpoints on the configured 25-minute interval:
weights, optimizer, scheduler, RNGs, data cursor and manifest. Upload destinations
must be actually configured/authorized; complete-marker logic must reject partial
files. Stage shards on local NVMe, not many small files on mounted storage.

Acceptance: 20-step interrupt/resume comparison passes the declared numerical
policy; shard/sample IDs resume exactly; real allocated/reserved memory is reported;
one pilot run learns against heldout validation; three-seed runs are preserved.

## P1-10 — Evaluate Repair/PGD and make the G2 decision

Status: blocked. Dependencies: P1-06, P1-07, P1-09, P1-08, P0-02.

Freeze policy, budgets, timeout handling, metrics and dataset hashes before seeing
the G2 outputs. Evaluate dense unseen assemblies with at least 16 lowered parts
under an explicitly defined counting rule and no more than three re-audits. Record
Pass@1, Repair@K, PGD improvements, failure types, raw cost vectors, fixed-hardware
latency, and uncertainty. Set all gate evidence fields only from linked artifacts.

Acceptance: every generated item, including failures and timeouts, appears in the
ledger; matched-budget comparisons can be reproduced; the G2 calculator receives
real manifests and produces the correct branch. At >=90%, skip P2. At <90%, diagnose
implementation/constraint defects before attributing failures to a learned-solver need.

## P2-01 — Implement learned residual refinement only if eligible

Status: conditional. Dependencies: P1-10 decision below 90% with valid evidence.

Recover the truncated reference implementation or implement the declared layer
contract with explicit gradient/stop-gradient choices. Match the selected model's
actual hidden dimension, declaration-close features, 24D FK descriptor, 6D pose
deltas and 6D log variances. Cache FK keys/values outside the loop. Mask every
attention, potential, state update and loss; define all-masked history safely.

Acceptance: count actual trainable parameters; five real slots padded to 8 and 16
match forward/backward within 1e-6; invalid slots have no gradient influence; outputs
are applied and audited; mixed-precision and finite-value tests pass.

## P2-02 — Run cached ablations and conditional joint tuning

Status: conditional. Dependencies: P2-01.

Compare FK attention, exact-gradient input, masking and learned residual choices
at equal budgets with three seeds. Cache features with exact model/tokenizer/revision
keys; invalidate caches after backbone changes. Joint tuning follows a reproducible
cached-feature benefit, not merely a successful training loss decrease.

Acceptance: actual heldout net benefit over Repair/PGD with latency/memory and
confidence intervals; negative results retained. Roll back to the classical system
if improvements disappear under matched costs or introduce numerical failures.

## P3-01 — Recover motion grammar and implement analytic IK templates

Status: pending. Dependencies: P1-10; this task does not require the P2 branch.

Specify complete `oscillator`, `spring_step`, channel and timeline semantics before
implementation. Build stance-foot world locks and analytic two-link IK with
reachable/unreachable target handling and smooth swing trajectories. Separate
skeletal motion from static geometry projection. Fit harmonic/damped parameters
and retain approximation residuals, rather than promising exact finite-harmonic IK.

Acceptance: biped/quadruped/hexapod stance slip <0.02 units on specified fixtures,
including root motion and world coordinate conversion; fit angle RMSE <1.5 degrees;
unreachable goals are flagged; topology holdouts do not contaminate training.

## P3-02 — Implement Bedrock and GeckoLib with independent capabilities

Status: pending. Dependencies: P3-01, P0-02.

Pin supported versions and inspect primary format/runtime documentation. Implement
geometry, attachment, expression, clip and state-container translations separately.
Test missing/unsupported functions and clip timing. Add custom-block-specific
geometry/culling and item attachable support only under explicit versioned profiles.

Acceptance: actual target imports/playback show geometry and timing parity; expression
sampling matches reference curves; runtime logs are clean; unsupported archetype
and profile combinations fail explicitly. Do not treat names like Molang as proof
of identical semantics between runtimes.

## P3-03 — Build dynamic dataset and cross-topology evaluation

Status: pending. Dependencies: P3-01, P3-02, P1-09.

Generate the report's 15K dynamic target only after a reviewed pilot. Keep unseen
hexapod, heavy-tail tyrannosaur and segmented sea-dragon body plans held out. Record
geometry validity, motion fit, contacts and runtime acceptance separately. Stage and
train with the same checkpoint discipline as the static phase.

Acceptance: reproducible dynamic shards with provenance; no topology leakage;
cross-topology pass/slip/fit distributions and representative failure videos.

## P4-01 — Establish HandCraft-Bench-50 and external baselines

Status: pending. Dependencies: P1-10.

Prepare the 15 weapon/15 complex block/10 extruded item/10 animated creature brief,
rights records, blinded review protocol and target-camera specification. Obtain
actual independent original assets and rights before claiming a 50-asset benchmark.
Any communication/commissioning requires the user's authorization. Implement the
Blockbench MCP and mesh-derived baseline adapters with the same audit feedback
and explicit cost/reporting boundaries; do not count manual repair as free work.

Acceptance: real asset inventory, author/license records, fixed evaluation protocol,
and reproducible baseline scripts. The starter's three simple examples are not the
external benchmark and cannot be relabeled as such.

## P4-02 — Freeze reproducible results and release package

Status: pending. Dependencies: P4-01, P3-03, and P2-02 only if that branch was used.

Run final frozen evaluation with recorded software/model/data hashes and fixed
camera/render setup. Produce artifact-quality comparisons, blinded preference
statistics, per-family distributions, raw costs, latency/memory and failure analysis.
Check code/data/model licensing per actual provenance. Reproduce from a clean
checkout and publish only with authorization.

Acceptance: another machine/agent can reproduce the core results from manifests;
paper claims link to actual evidence; missing results are labeled; default runtime
works without optional learned weights when the classical branch was selected.
