# SyncCraft Block-DSL for Visual Studio Code (`synccraft-blockdsl`)

Official language support, semantic syntax highlighting, context-aware auto-completion, hover physics/kinematics reference cards, snippets, go-to-definition, document outline, canonical formatting, and live SE(3) + 15-axis SAT collision auditing for **SyncCraft Block-DSL v3.2** (`.blockdsl`, `.bdsl`, and Markdown ```` ```blockdsl ```` / ```` ```dsl ```` fenced code blocks).

## Features

1. **Full Block-DSL v3.2 Grammar & Markdown Highlighting**
   - Highlights all 10 top-level sections (`@meta`, `@textures`, `@palette`, `@rig`, `@materials`, `@style`, `@muscles`, `@kinematics`, `@behavior`, `@controller`), compile-time metaprogramming (`let`, `def`/`use`, `chain`), spatial attachments (`attach`, `joint`, `place`), symmetry modifiers (`mirror_x`, `mirror`, `radial`), `blend_with` sculpted seams, `spin` mechanical tracks, and all 10 physical material families.
   - Immediately highlights unimplemented reserved keywords (`@shade`, `@motion`, `@ergonomics`, `extrude`) and capitalized Python booleans (`True`/`False`) as illegal syntax.
   - Embedded Markdown injection highlights ```` ```blockdsl ```` and ```` ```dsl ```` blocks inside `.md` specs and dataset cards.

2. **Deterministic Expanded-Symbol Auto-Completion**
   - Automatically harvests and expands symbols across `mirror_x(twin)` (`twin__<child>`), `mirror(axis=z, as=twin)`, `radial(count=K)` (`<bone>_r1__<child>`), `use macro(...) as pref` (`pref_<child>`), and `chain ch ... segments=K` (`ch_1_<box>`).
   - Context-aware completions for `material:`, `parent_box=`, `base_box=`, `blend_with: [...]`, `belly_box:`, `origin_bone:`/`insertion_bone:`, `ik_chain` `bones: [...]`, `track` (`wave`, `flap`, `hinge`, `impulse`, `spin`), `muscle_drive`, `ik_drive` (`foot_cycle`), `@materials` (`process(family=...)`), `@controller` state transitions, and flush-stacking presets (`align = -anchor`).

3. **Hover Cards, Go-to-Definition & Outline**
   - Hover any section, spatial operator, physics family, parameter, or expanded bone/box (`thigh_r__shin_l`) to inspect its formula, valid parameter bounds, and declaration line.
   - `F12` (*Go to Definition*) jumps directly to declared bones, boxes, palettes, macros, muscles, IK chains, clips, and states (including from mirrored/radial expanded names).

4. **Two-Tier Diagnostics & Canonical Formatting**
   - **Instant Lint (< 3 ms on keystroke)**: Catches duplicate global/expanded identifiers, unknown `@palette` references, invalid `joint_limits`, negative `inflate`, and `entity` + `java_*` profile mismatches without requiring Python.
   - **Deep SyncCraft Compiler & 15-Axis SAT Audit (on save / command)**: Auto-detects `<workspace>/.venv/bin/python` to run `parse_program`, `compile_geometry`, `audit_scene(enforce_physical_contact=True)` (`strict` & `sculpted`), and 12-frame swept `audit_behavior_clip`, plus `Shift+Alt+F` canonical formatting via `format_program`.

## Installation

### Option A — Install Packaged `.vsix` (Recommended)
1. Build the `.vsix` bundle from the SyncCraft repository root:
   ```bash
   .venv/bin/python scripts/package_vscode_extension.py
   ```
   This produces `dist/synccraft-blockdsl-0.3.2.vsix`.
2. In VS Code / Antigravity / Cursor: open the Command Palette (`Cmd+Shift+P` / `Ctrl+Shift+P`) -> **`Extensions: Install from VSIX...`** -> select `dist/synccraft-blockdsl-0.3.2.vsix`.

### Option B — Symlink for Live Extension Development
```bash
ln -s "$(pwd)/editors/vscode-blockdsl" ~/.vscode/extensions/synccraft.synccraft-blockdsl-0.3.2
```
