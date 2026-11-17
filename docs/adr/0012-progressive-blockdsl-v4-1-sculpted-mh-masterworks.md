# ADR 0012: Quantitative Sculpted Blockbench Style Anchor & Progressive Block-DSL v4.1 Monster Hunter Masterworks (V9)

- **Status**: Accepted (additive audit/shading refinements & V9 progressive sculpted dataset pipeline; 100% backward-compatible with ADR 0009–0011)
- **Date**: 2026-11-17
- **Related Tasks**: `P1-01`, `P1-02`, `P1-03`, `P1-06`, `P3-01`, `P3-02`
- **Builds On**: `ADR 0004`, `ADR 0008`, `ADR 0010`, `ADR 0011`

## Context

Visual critique of the programmatic `mh_dragons_v8` asset pack against authentic human-artist Sculpted Blockbench benchmarks (`.agents/skills/blockbench-sculpted-style/SKILL.md`, `@zyhfisme`'s Zinogre & Kirin turnarounds, `Illusia/Irushia` Young Plague Wyrm, *Ice and Fire* dragon anatomy, and official Monster Hunter physiological renders) revealed four root causes that made programmatically stamped models look uniform and un-Blockbench-like:

1. **Shared Coordinate Stencils Across Unrelated Lineages**: `mh_dragons_v8` emitted `.blockdsl` strings from shared Python rig templates, giving serpentine dragons (`fatalis`), blind cloak-elders (`gore_magala`), jet-propelled elders (`crimson_glow_valstrax`), and aristocratic frost dragons (`velkhana`) nearly identical horse-dog torso/snout coordinates.
2. **Doghouse Muzzles, Ladder-Plank Scutes, & Uni-Sheet Wing Rigidness**: Wide axis-aligned muzzle boxes lacked yaw-converging maxilla slopes; thick horizontal belly plates protruded with dark gaps; and single-sheet wing cards cannot fold or bend cleanly under Minecraft/Blockbench rigid-body kinematics.
3. **Desaturated Surface Fills & Two Engine Edge Cases**:
   - `shade_patch_banded_linear` in `src/synccraft/shading/compiler.py` ignored `patch.height`, suppressing `coat:` pattern relief under `shade: banded(...)`.
   - `compute_plane_edge_alpha_mask` evaluated `local_u = (i % 2) / 1` (`0.0` or `1.0`) when `width = 2`, collapsing `tri = 1.0 - |2 local_u - 1|` identically to `0.0` instead of alternating sawtooth notches and tips.
   - `compile_geometry` in `src/synccraft/kinematics/se3_chain.py` applied `plane` `layer` normal offsets to `local_center` after computing `unrot_local_center`, causing a `0.25 * layer` vertex offset in `export_bedrock`'s 8-corner serialization re-audit on rotated layered planes.
4. **Missing Quantitative Art-Style Anchor**: While `audit_scene` verified `SE(3)` collisions, burial (`<= 0.65`), and `Y = 0` grounding, no evaluator measured whether a compiled asset actually embodies Sculpted Blockbench aesthetics (wedged snout taper, `70/25/5` volume hierarchy, rotated cuboid flow, hugged scutes, non-uni-sheet wing partitioning, and hue-shifted cel vibrancy).

---

## Decision

### 1. Core Engine & Shading Fixes
1. **Exact Rotated `plane` `layer` Serialization Parity (`src/synccraft/kinematics/se3_chain.py`)**:
   - When `item.plane_normal is not None and item_layer != 0`, `unrot_local_center` is updated consistently (`unrot_local_center + layer_vec * cumulative_scale` when `item.pivot is not None`, or `local_center` when rotating about box center), achieving `0.0` (`< 1e-15`) 8-corner error across `Scene`, `.bbmodel`, and Bedrock `.geo.json`.
2. **Sawtooth `width=2` Pixel Cutout Mask & Banded Relief (`src/synccraft/shading/compiler.py`)**:
   - `compute_plane_edge_alpha_mask` alternates `tri in {0.0, 1.0}` when `w_tooth <= 2` so 2-pixel sawtooth fur/spike combs produce crisp binary `{0, 255}` teeth.
   - `shade_patch_banded_linear` incorporates centered `patch.height` relief into cel-band quantization so `coat:` patterns (`scale_rows`, `banded_limb`, `striped`, `spotted`) step cleanly across discrete cel bands.

### 2. Quantitative 6-Pillar Sculpted Style Anchor (`src/synccraft/audit/sculpted_style_evaluator.py`)
`evaluate_sculpted_style_anchor(program, scene, atlas_rgba=..., require_wings_partitioned=...)` scores compiled Block-DSL programs on `[0, 1]` across six artistic pillars derived from `SKILL.md`:
1. `organic_rotated_flow` (`0.16` weight): fraction of 3D cuboids carrying deliberate non-90-degree rotations.
2. `snout_and_head_sculpture` (`0.20` weight): anterior-to-cranial width taper (`snout_taper_ratio <= 0.88`, flagging `snout_doghouse_warning` otherwise) and cranial sculpture density.
3. `volume_hierarchy` (`0.14` weight): proximity of sorted cuboid volumes to the `~70 / 25 / 5` primary / secondary / tertiary mass distribution.
4. `connected_seating_and_scutes` (`0.20` weight): `audit_scene` pass (`0` collisions, `0` swallowed boxes), `Y = 0` ground contact, and absence of tall `ventral_plank_protrusion` plates.
5. `non_unisheet_cutout_silhouette` (`0.16` weight): binary `{0, 255}` alpha-cutout `plane` silhouettes (`edge:`) and, for winged creatures (`require_wings_partitioned=True`), multi-bone wing partitioning (`>= 6` wing/cloak/thruster/vane bones, flagging `unisheet_wing_warning` otherwise).
6. `vibrant_cel_palette` (`0.14` weight): hue-shifted OKLCH ramps, `shade: banded` cel steps, and pixel chroma verification (`flat_grey_desaturation_warning` guard).

### 3. Progressive Hand-Authored Block-DSL v4.1 Masterworks (`src/synccraft/data_gen/mh_sculpted_v9/`)
1. **Individual Dragon-by-Dragon Progressive Authoring**:
   - Every specimen in the 22-monster roster owns a dedicated `dsl/<id>/` directory containing a visual `trait_sheet.md`, `stage1_blockout.blockdsl` (primary masses & gesture), `stage2_secondary.blockdsl` (wedged snout, secondary armor, multi-bone wings/tail), and `<id>.blockdsl` (tertiary detail, binary-cutout planes, `@materials`, `@style`, `@stamps`, `@controller`, and 12 vigorous Monster Hunter `@behavior` clips).
2. **6-Mode Progressive Supervision & Strict Quarantine**:
   - `scripts/build_mh_dragon_dataset_v9.py` packages 6 progressive World-Model supervision modes (`stage1_blockout_synthesis`, `stage1_to_stage2_refinement`, `stage2_to_masterwork_refinement`, `vigorous_behavior_and_controller_authoring`, `create_full_masterwork_asset`, `sculpted_style_evaluation_and_critique`) with zero family-group split leakage (`research_nc/train`, `research_nc/val`, `research_nc/test_unseen`), `commercial_weights_eligible = False`, and `runtime_import_verified = False`.
