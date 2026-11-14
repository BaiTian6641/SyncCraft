# Starter status and handoff boundary

Release: **0.1.0**, prepared 2026-11-13. Intent baseline: supplied SyncCraft v3.0
report. This is a local development starter, not a trained generator or completed
research system.

## Implemented and exercised in this environment

| Capability | Evidence and limits |
| --- | --- |
| Static box DSL | Typed parsing, validation, canonical formatting; 500 seeded AST roundtrips |
| Local/world hierarchy | Root, attach, place, Euler composition, reflected subtrees |
| CPU OBB collision | Degenerate-axis-safe SAT, 1,000 actual flush fixtures, direct-vs-smooth-loss counterexample |
| Static geometry diagnostics | Non-exempt collision check, reported excluded contacts and declaration components |
| Blockbench 5 export | Stable UUIDs, groups/outliner, transforms, atlas; structural/numeric tests only |
| Java export | Modern XYZ and conservative legacy; composed rotation and bounds checks; not a packaged resource pack |
| Texture/preview | Integer padded per-face atlas with flat colors; CPU diagnostic four views |
| CLI and file pipeline | Actual export files, deterministic repeated output, controlled collision failures |
| Cost/gate helpers | Positive monotonic operation counts; explicit G2 threshold; empty evidence stays blocked |

Exact executed results and runtime versions are in `artifacts/validation/summary.json`
and `artifacts/validation/cpu-tests.txt`. Example outputs are in
`artifacts/examples/`. `scripts/check.py` reruns the current suite locally.

## Not yet implemented

| Capability | Blocking task |
| --- | --- |
| Extrusion, full static DSL coverage, nested mirrors | P1-01 |
| True permitted-contact regions and physical connectivity | P1-02 |
| Box UV production layout, shading, palette ramps, directional mirror textures | P1-03 |
| Grip/display solving and hand constraints | P1-04 |
| Differentiable Torch geometry, gradients and PGD | P1-05 / P1-06 |
| XGrammar, live symbol masks and real KV rollback | P1-07 |
| Audited dataset generators and large shards | P1-08 |
| Model bake-off, LoRA training, GPU memory/restart proof | P0-03 / P0-04 / P1-09 |
| Full baseline evaluation and G2 evidence | P1-10 |
| Learned refiner and training | P2-01 / P2-02, conditional only |
| IK, motion, Bedrock and GeckoLib export | P3-01 / P3-02 / P3-03 |
| Independent benchmark and final paper evidence | P4-01 / P4-02 |

## External validation not performed here

No Blockbench GUI import/save/reopen, Minecraft game import, GPU execution, model
download, tokenizer masking run, training job, Colab billing measurement, or external
modeler benchmark was performed. Linux CPU tests do not prove Windows/macOS setup
scripts or GPU dependency compatibility. Local import and platform checks are the
first handoff tasks.

## Known limitations that affect interpretation

`audit_scene(...).passed` certifies only finite valid OBB evaluation and absence of
non-exempt collisions. It **does not** establish validity of declared contact pairs,
connected physical geometry, usable grip, shading, UV seams or engine compatibility.
Deep overlap and separation on exempt declared pairs are visible in the report but
are not currently rejection conditions. See `NUMERICAL_NOTES.md`; fix P1-02 before
using the audit as a training-label or G2 acceptance oracle.

The plan's full static example includes features deliberately rejected by the
starter subset. The bundled examples use supported syntax and original geometry.
Flat palette colors are diagnostic; the configured ramps/hue shifts/emission are
preserved in the AST but explicitly listed as unused by diagnostic builds.

## First work queue

1. Reproduce S0 and record the local environment.
2. Import the real generated fixtures in Blockbench/Minecraft and record versions.
3. Complete P1-02's contact contracts and P1-01's extrusion/static language.
4. Work through the remaining dependencies in `tasks/BACKLOG.md`.

Do not start the learned branch based on starter pass rates. The supplied
`configs/gate_g2_evidence.json` intentionally yields a blocked decision and exit 2.
