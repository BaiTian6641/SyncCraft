# Source-plan reconciliation and open decisions

Source: `source/synccraft-v3-plan.pdf`, supplied v3.0 freeze dated 2026-11-13, 18
pages. The report is the intent baseline. This file distinguishes that intent from
evidence actually established in this starter.

## Preserved decisions

- Typed block-program generation, deterministic multiple backends, shared SE(3).
- `place(...)` for multiple independent primitives on a bone; integer sizes.
- Classical Repair and PGD baselines before a conditional learned refiner.
- No quaternion/phased-oscillator residual solver reinstated by default.
- Static assets first; IK-based motion later; held-out families and external gold set.
- Local CPU/data/render work separated from GPU training and evaluation.
- Rollback work is charged; GPU checkpoint recovery must be measurable.

## Concrete corrections and gaps

| Source section | Finding | Starter decision / required follow-up |
| --- | --- | --- |
| §2.1, PDF p4 | The animation grammar cuts off partway through `spring_step`; some lexical productions are absent | Do not call a guessed grammar complete. Recover it or adopt a reviewed ADR before P3-01 |
| §3.3, PDF p10 | The refiner `forward` code is visibly clipped at the loop | No copied half-forward or fabricated working model. Reconstruct from mathematical contracts only after G2 |
| §2.1/§2.3 | Multiple boxes without placement and origin-relative nonzero anchors are ambiguous | Require `place` on later boxes; without `base_box`, require zero anchor. See ADR 0001 |
| §2.3 | Mirror naming/descendant reference details are not fully specified | Deterministic named clone convention; nested mirrors deferred. Asymmetric texture reflection remains open |
| §3.2 | Normalizing a zero cross product with a denominator floor leaves a false zero candidate | Remove degenerate cross axes from SAT minima; retain a regression |
| §3.2 | Unnormalised softmin is a lower bound; it can miss shallow real penetration | Raw valid-axis SAT minimum owns final audit; smooth potential is only an optimization surrogate |
| §3.2 | Broad adjacency exemptions can hide deep overlap or physical separation | Starter reports those exclusions explicitly. P1-02 must implement contact-region checks and physical connectivity before valid-asset metrics |
| §6.3 | “Any rotated flush pair” is not a valid unrestricted fixture generator | Construct geometry that is actually flush in world space, then test it; zero-gradient verification still needs Torch |
| §6.3 | Twenty-step PGD success is not a universal guarantee for a nonconvex constrained problem | Apply it to published feasible fixtures and report heldout success distributions and residual failures |
| §6.3 | “No schema error” implies an official complete `.bbmodel` schema | Official Blockbench docs say no complete format specification exists; use documented fields, pinned upstream behavior, golden imports, save/reopen and numerical reconstruction [1] |
| §1.3 | Java modern multi-axis rotation is supported | Confirmed by Mojang's 1.21.11 notes; new x/y/z values use X, then Y, then Z. Do not accidentally emit both old and new rotation forms [2] |
| §1.3 | Bounds could mean element coordinates or final rotated envelope | Starter enforces both conservatively; record the envelope rule as project policy, not an independently established engine restriction |
| §1.3 | Bedrock/GeckoLib equivalence is overstated as a universal adapter assumption | Verify both versioned runtimes independently; adapters currently fail explicitly |
| §4.1 | Candidate licenses/IDs/configurations are asserted in the report | Retain candidate names, but record exact source revision, license text, config and tokenizer evidence locally before model selection |
| §5 | VRAM, wall-clock time, Colab CU rates and subscription quantities are estimates/input claims | Preserve as planning assumptions. Measure on assigned hardware; do not market them as verified package performance |
| §5.3 | Token count + fixed SAT-step weighting is called hardware-independent compute | Retain a raw operation vector; a scalar proxy needs preregistered calibration and is not equivalent physical compute across models |
| §6.2 | A failed baseline is described as proof of local minima | It can justify an engineering experiment but does not isolate a cause; diagnose optimizer, constraints, generation and audit defects first |
| §6.2 | Three model 8K bake-offs precede a complete compiler/dataset | Bootstrap deterministic labels and a pilot generator first; parallelize model readiness with compiler implementation, then run comparable bake-offs |

## Evidence levels

1. **Source requirement:** present in the user's report, not independently measured.
2. **Primary-document checked:** official format behavior verified at the URLs below.
3. **Executed CPU evidence:** logs and output files in `artifacts/validation/`.
4. **Pending runtime evidence:** actual installed Blockbench/Minecraft imports and GPU runs.

Do not move an item between these levels because a file or test name exists.

## Source references

[1] Blockbench official wiki, `.bbmodel` format, retrieved 2026-11-13:
https://blockbench.net/wiki/docs/bbmodel/

[2] Mojang, Minecraft Java Edition 1.21.11, “Block Models”, retrieved 2026-11-13:
https://www.minecraft.net/en-us/article/minecraft-java-edition-1-21-11

[3] Blockbench upstream serializer, moving `master` reference (pin a commit locally):
https://github.com/JannisX11/blockbench/blob/master/js/formats/bbmodel.js

[4] Bjorn Ottosson, Oklab numerical definitions used for diagnostic base colors:
https://bottosson.github.io/posts/oklab/

The report's research bibliography and model license claims have not received a
full independent re-review in this starter task. Keep them labeled as source
claims until the corresponding research/model-selection task verifies them.
