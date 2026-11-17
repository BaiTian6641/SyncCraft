# SyncCraft Assistant — Blockbench 5 Plugin (`v0.4.1`)

Connects Blockbench (`>= 4.8 / 5.0`) to SyncCraft's local Python engine (`127.0.0.1:18765`) so artists stay inside Blockbench while running physical geometry/motion audits, Froude-scaled locomotion synthesis, cross-species retargeting, secondary motion layering, and non-destructive painted texture passes.

## Quick Start

1. Start the local SyncCraft bridge in a terminal:
   ```bash
   synccraft serve-blockbench --host 127.0.0.1 --port 18765
   ```
2. In Blockbench, open **File -> Plugins -> Load Plugin from File** and select `synccraft_assistant.js`.
3. Open **Tools** to access the six SyncCraft actions:
   - **SyncCraft: Analyze Model**: Derives `@morphology` limb roles, CoM, ground support polygon, and 6-pillar sculpted style score.
   - **SyncCraft: Check Rig & Motion**: Runs deterministic 15-axis OBB SAT collision, foot-slip (`<= 0.02u`), and joint-step audits on your existing `.bbmodel` animations with zero ML.
   - **SyncCraft: Generate Locomotion**: Generates a verified gait (`walk`, `trot`, `stalk`, `gallop`) and writes ordinary editable `BoneAnimator` keyframes inside a single `Undo` step.
   - **SyncCraft: Retarget Animation**: Transfers locomotion across quadrupeds, bipeds, sprawled reptiles, and hexapods via `@morphology` roles.
   - **SyncCraft: Add Secondary Motion**: Synthesizes `breathe`, `tail_sway`, or `wing_settle` clips or layers them onto an existing gait clip.
   - **SyncCraft: Non-Destructive Texture Pass**: Preserves `100%` of hand-painted pixels while adding separate transparent RGBA `wear`, `glow`, `blank_face_fill`, or perceptual OKLCH `variant` texture layers.

## Trustworthiness & Scope Notes

- All modifications execute inside a single `Undo.initEdit` / `Undo.finishEdit` step and never silently write unverified animations (`verified: false`).
- Automated Node.js + live localhost HTTP roundtrip tests verify the plugin controller and `.bbmodel` serialization contracts; interactive manual testing inside a live Blockbench GUI client remains tracked separately (`runtime_import_verified: false`).
