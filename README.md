# SyncCraft v3.0 starter pack

**A runnable, CPU-tested starting point and a concrete handoff for a local coding agent.**

The supplied v3.0 report defines the destination: editable block assets from a typed
DSL, deterministic compilers, shared kinematics, constrained decoding, and measured
Repair/PGD baselines. This package implements a deliberately bounded static-box
slice and provides the engineering queue to finish that destination.

Start with [START_HERE.md](START_HERE.md). Give your agent
[agent/LOCAL_AGENT_PROMPT.md](agent/LOCAL_AGENT_PROMPT.md), and have it follow
[AGENTS.md](AGENTS.md). The full source PDF is included under `source/`.

## Run the starter

Python 3.11 or newer is the declared target. The delivered checks were run on
Python 3.12.14 with NumPy 2.3.5 and Pillow 13.0.0. CPU only; no model download,
GPU, account, or API key is required for these commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m pip install --no-deps -e .
python scripts/check.py
python scripts/demo.py
```

On Windows, run `powershell -File scripts/bootstrap.ps1`, then
`.venv\Scripts\python.exe scripts/demo.py`. On macOS/Linux,
`bash scripts/bootstrap.sh` performs the setup and checks. Dependency installation
requires ordinary package-registry access. With NumPy/Pillow already installed,
`python scripts/check.py` and `python scripts/demo.py` work directly without pip.

```bash
synccraft doctor
synccraft validate examples/simple_blade.blockdsl
synccraft compile examples/simple_blade.blockdsl --target java_modern --out runs/blade-java
synccraft compile examples/mirrored_wings.blockdsl --out runs/wings
```

Each compile writes a `.bbmodel`, canonical DSL, typed AST JSON, world-space scene,
audit report, integer UV atlas, and four-view diagnostic preview. Java targets also
write `asset.java.json`. Reuse an output directory with explicit `--overwrite`.
The Java JSON is one model, not an installable resource pack. Its default texture
resource is `synccraft:block/atlas`; installation and item/block dispatch are a
later task. A failed static audit prevents export by default.

## What works now

| Area | Delivered scope |
| --- | --- |
| DSL | Typed static box AST; parser; canonical formatter; strict names and references |
| Assembly | `attach`, `place`, degree-based Euler rotations, named mirrored subtrees |
| Geometry | Shared NumPy SE(3), oriented boxes, integer dimensions and inflation |
| Collision | Direct SAT using up to 15 valid axes; smooth potential kept separate |
| Export | Blockbench 5 source-based serializer; modern Java; conservative Java legacy |
| Texture | Deterministic padded per-face atlas, flat OKLCH base colors |
| Inspection | CPU four-view previews and JSON audit with explicit coverage gaps |
| Experiment setup | Raw-cost ledger, blocked-by-default G2 calculator, candidate/config templates |
| Handoff | Task dependencies, acceptance gates, ADRs, local-agent prompt, source report |

## What still needs implementation or external validation

Extrusions, full palette ramps and `@shade`, grip/display solving, real contact
validation, differentiable PyTorch kernels, PGD, XGrammar/KV rollback, model bake-off,
training, learned refinement, IK, animation, and Bedrock/GeckoLib export remain
tracked work. Unsupported DSL sections fail explicitly. Future modules that expose
entry points raise `NotImplementedError`; they never return fake results.

Export files have structural/numerical checks, **not a completed Blockbench or
Minecraft import test**. `build_info.json` records this. The diagnostic previews are
software projections, not screenshots from an engine. `audit.json` always reports
`complete_for_release: false` in this release.

See [docs/STATUS.md](docs/STATUS.md), [docs/tasks/BACKLOG.md](docs/tasks/BACKLOG.md),
and [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) before marking a phase complete.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/synccraft/` | Implemented core and explicit future-module boundaries |
| `examples/` | Three original small assets; no imported game textures |
| `tests/` | Executable CPU regression and integration checks |
| `configs/` | Candidate models, training intentions, budgets, gate and dataset policies |
| `docs/` | Architecture, decisions, compatibility, validation, and task queue |
| `agent/` | Ready-to-paste task prompt and session reporting template |
| `artifacts/` | Delivered example outputs and actual validation evidence |
| `source/` | Unmodified supplied plan plus extracted text and provenance |

## Licensing

New starter source code and original example definitions are supplied under
Apache-2.0; see `LICENSE` and `NOTICE`. The user's source PDF retains its original
rights and is not relicensed by this repository. No model weights, community game
assets, or training dataset are bundled. Model/data rights must be established per
revision before training or redistribution; a candidate's name is not verification.
