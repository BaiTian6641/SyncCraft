# SyncCraft local-agent engineering instructions

## Goal and authority

Develop the SyncCraft v3.0 system described by `source/synccraft-v3-plan.pdf`,
starting from this repository. Preserve its typed-program approach, shared
kinematics, classical baselines, held-out evaluation, and conditional learned
refiner. This is a starter, not evidence that Phase 0 or Phase 1 is finished.

User instructions override this file. The PDF specifies research intent;
`docs/PLAN_RECONCILIATION.md` and ADRs document concrete corrections or currently
unresolved semantics. Do not silently rewrite the plan to fit a shortcut.

## Work autonomously within the assigned task

Inspect before editing. Preserve existing user changes. Implement and verify
complete local increments; do not stop after a plan, empty class, or illustrative
mock. Do not seek permission for routine local code, tests, or documentation.
If an external credential, hardware run, or ambiguous missing source truly blocks
the next action, finish independent local work and report the precise blocker.
Use a session report from `agent/SESSION_REPORT_TEMPLATE.md`.

Do not launch paid compute, publish assets, contact people, or upload private
datasets merely because a configuration mentions those operations. The current
request authorizes local development and preparation. Apply any additional user
authorization normally when supplied. Never commit secrets or log token values.

## Required first actions

1. Read the status, reconciliation, architecture, backlog, and acceptance files.
2. Run `python scripts/check.py` and `python scripts/demo.py`; record actual output.
3. Record local Python, GPU/driver, Blockbench, target Minecraft, and OS versions.
4. Work on the earliest unblocked task. Complete evidence before changing status.
5. Keep implemented, externally validated, untested, and deferred work distinct.

## Architecture and numerical rules

- Parse DSL into typed data; never evaluate generated Python or use `eval`.
- Every compiler and optimizer must share the SE(3) definition. Exporters transform
  representation, not create a second geometry engine. Cross-check Torch vs NumPy.
- Integer sizes are discrete and fixed during refinement. Degree/radian conversion
  happens at named boundaries. `inflate` and physical size are separate.
- Derive placement from the exact parent/local frames. Specify mirror expansion,
  identifier remapping, references, rotations, cull directions, and texture parity.
- Use direct SAT for final collision audit. Mask degenerate cross axes; do not
  interpret zero smooth loss as no overlap or call SAT depth an intersection volume.
- Treat contact exclusions as explicit contracts, never blanket parent/sibling
  immunity. Report every exemption; implement a contact-region policy before G2.
- Re-audit serialized/rounded outputs. Do not assume optimization success survives
  quantization or exporter conversion.
- Fail explicitly for unsupported features. No silently ignored `@shade`, motion,
  extrusions, grip rules, or decoder-mask requirements.

## Evidence and phase gates

Run relevant tests after a change and the CPU suite before handoff. Use independent
oracles or actual target imports for compatibility. Do not replace engine import
tests with a validator that simply repeats the serializer's assumptions.

The full gates in `docs/ACCEPTANCE.md` are not met by the starter's test count.
Skipped tests, mock GPU runs, fabricated datasets, and invented performance tables
are not evidence. Preserve failure examples. Log tool versions and exact commands.
Do not edit a threshold to get a passing run without an explicit ADR and new
preregistration before evaluating the affected holdout.

G2 requires a frozen unseen dense-assembly set, a defined combined Repair/PGD
policy, matched raw costs, all failure cases in the denominator, and at most three
re-audits per sample. Enable the learned branch only for a measured rate below
90%. At or above 90%, skip it. Failure alone does not prove local minima.

## Model/data/reproducibility rules

Resolve model revisions and inspect licenses before use. The report's claims are
not a license record or a measured VRAM result. Keep model selection open until
the six smoke checks and distance-bucket evaluation are complete.

Do not add Mojang assets, copyrighted community derivatives, or incompatible
motion corpora to training. Keep file provenance, generator versions, asset IDs,
and family-level splits. Separate training/validation/G2/final external sets.
Freeze policy choices on validation data, not G2 or the final benchmark.

Count generated, discarded, and repair tokens, projection steps, compiler calls,
initial audits, and re-audits. A rollback cannot refund cost. Never claim a fixed
token-to-gradient-step exchange is physically hardware independent. Report raw
counters plus calibrated latency/memory measurements and the calibration method.

## Project hygiene

Use stable task IDs in commits and reports. Keep generated large datasets,
checkpoints, credentials, virtual environments, and downloaded third-party assets
out of source control. Record actual dependency locks per tested backend/hardware.
The supplied CPU pins do not constitute a GPU lock. New modules should have real
behavior or explicit unsupported entry points; remove placeholders when completed.
