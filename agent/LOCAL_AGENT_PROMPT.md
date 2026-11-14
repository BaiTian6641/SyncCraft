# Paste this into the local agent

You are implementing SyncCraft v3.0 from this repository and its bundled source
report. Read `AGENTS.md`, `docs/STATUS.md`, `docs/PLAN_RECONCILIATION.md`,
`docs/ARCHITECTURE.md`, `docs/tasks/BACKLOG.md`, and `docs/ACCEPTANCE.md` first.
Inspect actual code before planning edits. Treat the starter as a tested static-box
foundation, not as completed Phase 0/1 and not as an already trained model.

My intended outcome is a working, editable block-asset generation system following
the report. Continue implementing it in dependency order. Make the first session
concrete: reproduce the CPU tests/demos, record environment versions, run actual
Blockbench import/save/reopen checks if the application is available, and begin
the earliest unblocked Phase 1 task. Resolve ambiguous semantics in a short ADR
before implementing them. Preserve the current tested behavior.

Prioritize the complete static vertical slice: typed DSL including `place` and
extrusion, shared geometry, correct exporters, integer UV/shading, contact/grip
audits, real constrained decoding with complete state rollback, and classical
Repair/PGD baselines. Model smoke checks can run in parallel once correct compiler
labels and an audited pilot dataset exist. Do not build the learned refiner before
G2 provides the required evidence. Do not start motion at the expense of finishing
the static path.

Use the source report's candidate models, LoRA/batch/sequence intentions, data
splits, and evaluation goals as starting configurations, verifying exact versions
and measurements. If a dependency/API differs, document the evidence and make a
compatible implementation. Never manufacture model IDs, release claims, timing,
memory, license approvals, benchmark assets, or test results.

Work through local implementation and verification without stopping to ask for
routine confirmation. Use real functions and integration tests rather than mock
success. If credentials, paid compute, missing hardware, or truncated source
material blocks a task, finish independent local work and report the precise
remaining input. Do not download huge models or launch paid training just to
complete a scaffold; first record resource requirements and use authorization
already provided in this session.

At each handoff, update `docs/STATUS.md` and the task ledger, and provide changed
files, commands actually run, observed results, remaining risks, and the next
executable action. A task is done only after its acceptance evidence exists.
Use `agent/SESSION_REPORT_TEMPLATE.md` so I can resume with another agent.
