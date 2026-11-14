# Acceptance gates and evidence required

The tests already present are **starter regression tests**, not completed copies of
all report gates. Never add skipped or empty tests to make an acceptance file look
covered. Each gate needs commands, exact versions, fixtures, raw outputs, and a
reviewable result.

| Gate | Required acceptance evidence | Current boundary |
| --- | --- | --- |
| S0 starter | CPU suite; three example builds; deterministic output; explicit errors | Delivered evidence in `artifacts/validation/` |
| G0 environment/model readiness | Pinned target/tool/model revisions; six source smoke checks; real candidate bake-off | Pending |
| G1a language/compiler | 500 valid full-DSL static cases with boxes/extrusions/place/mirror; negative cases; actual target imports | Starter covers box subset and AST roundtrip only |
| G1b geometry/projector | 1,000 truly flush rotated fixtures; Torch potential and gradients <1e-7 where intended; fixed feasible 0.2-unit penetration tests solved within 20 steps with anchors preserved | NumPy values and collision fixtures only; no Torch/PGD |
| G1c asset validity | Physical contact policy, connectedness, grip/display, integer UV/seams, serialized output and engine checks | Not complete; declared-contact exceptions can hide invalid attachments |
| G1d decoding/training | Real token masks and full-state rollback; all work counted; reliable dataset generation; resumable LoRA run | Pending |
| G2 branch | Frozen dense unseen set (>=16 parts); specified Repair/PGD policy; matched budgets; <=3 re-audits; every attempt counted | Blocked |
| G3 learned branch | Only if G2 <90%; 5 real slots padded to 8/16; forward/backward differences <1e-6; all-masked history; ablation/equal-cost evidence | Conditional, not started |
| G4 motion | Biped/quadruped/hexapod stance horizontal slip <0.02 units; exported animation angle RMSE <1.5 degrees; runtime playback | Pending; grammar recovery needed |
| G5 training recovery | seq=2048, microbatch=2; interrupt after 20 steps; exact resume policy, optimizer/scheduler/RNG/cursor restoration; measured allocated/reserved peaks | Pending GPU execution |
| G6 final evaluation | HandCraft-Bench-50 provenance, fixed cameras, blinded evaluation, baselines and statistical uncertainty | Pending |

## Necessary details behind the report's thresholds

- The reported <18.0 GB allocated-memory target for 4B BF16 LoRA is a target, not
  a demonstrated measurement here. Record decimal GB and binary GiB explicitly,
  along with peak allocated and reserved GPU memory, dtype, optimizer, adapter
  shapes, kernels, vocabulary, and checkpointing. Do not substitute NF4 while
  labeling the test BF16.
- Resume reproducibility must specify tolerances and deterministic kernels. Compare
  data IDs, random draws, optimizer/scheduler state, losses, and parameters for a
  resumed run versus continuous control; a visually similar curve is insufficient.
- The G1b 20-step criterion applies to published feasible fixtures. Keep failed
  arbitrary examples and report their rate; universal nonconvex convergence is not
  a justified acceptance claim.
- SAT depth is not overlap volume. Implement an independently verified intersection
  measure if reporting volume, or explicitly adopt depth/contact metrics with an ADR.
- “No schema error” must be replaced by useful target-specific checks where no
  official complete schema exists. Capture the actual parser/editor/game result.

## Engine roundtrip checklist

Record app/game versions and fixture SHA-256. Import the generated asset in the
target, inspect hierarchy, origins, face ordering, mirror parity, texture scale and
rotation. Save/reopen in Blockbench. Compare world-space geometry and UV mapping
after the roundtrip. For Java, install the model/texture in a minimal versioned
resource-pack fixture and inspect the game log and visible result. A standalone
model file is not an installable pack. Bedrock/GeckoLib use their own runtime paths.

Include no original game assets in training simply because they were used as
compatibility references. Runtime screenshots are test evidence, not training rows.

## G2 policy

Freeze the combined Repair/PGD schedule on validation data. State whether each
sample runs repair then PGD, alternates, or uses another fixed sequence. Do not
cherry-pick the better output using hidden gold labels. Use heldout families and a
fixed part-count rule. Count syntax failures, timeouts, budget exhaustion, invalid
exports, and solver failures in the full denominator. Preserve per-sample costs.

At least 90%: skip the learned module. Below 90%: diagnose defects, then allow a
learned-refiner experiment if the evidence remains valid. Report confidence
intervals and per-family results; threshold arithmetic alone does not establish
scientific significance or prove a particular failure mechanism.

## What to preserve

Store test logs, configuration hashes, dataset manifests, model revisions, hardware
details, actual profiler output, engine logs, and representative success/failure
assets. Link them from the task ledger. Do not overwrite the frozen G2 set during
debugging; derive separate development fixtures from disclosed failure categories.
