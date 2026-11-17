# SyncCraft V9 Monster Hunter Sculptor Guide (Progressive Block-DSL v4.1)

Every dragon MUST be sculpted **individually from its own visual references** — NEVER copy/paste another monster's skeleton stencil or write a Python geometry generator!

## 1. Mandatory Visual Reasoning & Artist Iteration Loop (Per Monster)
1. **Study Visual References First (`read_file` on images!)**:
   - Read `.agents/skills/blockbench-sculpted-style/SKILL.md`
   - View `artifacts/mh_reference_study/style_anchors/zyhfisme_zinogre_01.png` (primary sculpted Blockbench style & snout anchor)
   - View `artifacts/mh_reference_study/monsters/<monster_id>_reference.png` (and any lineage style anchor in `artifacts/mh_reference_study/style_anchors/`).
2. **Write `dsl/<id>/trait_sheet.md`**:
   - Measure `1 HL` (head length) and write a region-by-region Keep / Simplify / Exaggerate / Paint table for Whole Gesture, Head & Snout, Neck & Torso, Wings / Appendages, Limbs, Tail, and Surface Palette.
3. **Stage 1 Primary Masses (`dsl/<id>/stage1_blockout.blockdsl`, ~24–42 elements)**:
   - Sculpt only primary posture, S-curve spine, limb zig-zag grounded on `Y = 0`, wedged snout core, and primary wing/tail silhouette.
   - Run: `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/inspect_mh_sculpt.py <id> --stage 1`
   - **View `stage1_visual_check.png` via `read_file`!** Fix any wrong horn/tail/wing/leg sign BEFORE Stage 2.
4. **Stage 2 Secondary Anatomy (`dsl/<id>/stage2_secondary.blockdsl`, ~95–150 elements)**:
   - Add 3D wedged snout (`maxilla_wedge_l/r`, `nasal_bridge`, `brow_ridge_l/r`, `cheek_jowl_l/r`, recessed `eye`s, `jaw`), multi-segment horns/crests, hugged ventral scutes, limb armor/toes, tail blades/fans, and articulated multi-bone wing spars.
   - Run: `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/inspect_mh_sculpt.py <id> --stage 2`
   - **View `stage2_visual_check.png` via `read_file`!**
5. **Stage 3 Vibrant Masterwork (`dsl/<id>/<id>.blockdsl`, >= 245 elements, >= 28 bones, 12 vigorous `@behavior` clips)**:
   - Add tertiary shingles, fangs/claws, dorsal studs/crystals/radiators, layered binary-cutout `plane`s (`edge: ...`, `layer: 1|2`), `@materials`, `@style`, `@stamps`, `spring`s, `@controller`, and **12 vigorous, wild Monster Hunter `@behavior` clips** (must include `idle`, `walk`, `run`, `roar`, plus 8 species-signature attacks including the exact `signature_gif_clip` from `catalog.py`!).
   - Fast audit check (`~1.5s`): `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/inspect_mh_sculpt.py <id> --stage 3 --no-render`
   - Full studio render (`~25s`): `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 .venv/bin/python scripts/inspect_mh_sculpt.py <id> --stage 3 --gif`
   - **View `<id>_studio_review.png`, `<id>_stages_and_silhouette.png`, and `head_hero.png` via `read_file`**, critique honestly against `<id>_reference.png`, and perform at least one visual polish iteration!
   - Target certification: `static_sculpted_audit_passed == true`, `serialized_bedrock_vertex_max_error == 0.0`, `style_anchor_score >= 0.88`, `style_diagnostics == []`.

---

## 2. Exact Right-Handed `SE(3)` Coordinate & Sign Cheat-Sheet
- Frame: **`+X` = Monster's Left**, **`+Y` = Up**, **`+Z` = Forward (toward Nose)**; `R = Rz(z) @ Ry(y) @ Rx(x)`.
- **Forward-extending bones (`neck`, `head`, `snout`, forward `horn` with `align=[0, 0, -1]`)**:
  - `-Rx` pitches **UPWARD (`+Y`)**; `+Rx` pitches **DOWNWARD (`-Y`)** (e.g. `jaw` opening downward = `+Rx`!).
  - `+Ry` yaws toward `+X` (left); `-Ry` yaws toward `-X` (right).
- **Backward-extending bones (`tail`, swept-back `ear`/`horn` with `align=[0, 0, 1]`)**:
  - **`+Rx` lifts tail/ear UPWARD (`+Y`)**; `-Rx` droops it DOWNWARD (`-Y`).
- **Downward-extending `+X` Left Limb (`align=[0, 1, 0]`)**:
  - **`+Rz` splays `+X` leg WIDE OUTWARD (`+X`)**; `-Rz` tucks inward (`-X`).
  - `-Rx` swings leg FORWARD (`+Z`); `+Rx` angles leg BACKWARD (`-Z`).
- **Outward-extending `+X` Left Wing (`align=[-1, 0, 0]`)**:
  - **`+Rz` raises wing UPWARD (`+Y`)**; `-Rz` flaps wing DOWNWARD (`-Y`).
  - `+Ry` sweeps wingtip BACKWARD (`-Z`); `-Ry` sweeps wingtip FORWARD (`+Z`).

---

## 3. Anti-Burial, Connectivity, & Non-Uni-Sheet Wing Rules
1. **Opposite-Sign `align` Rule (Prevents `swallowed_redundant_box` & `excessive_sculpted_overlap`)**:
   - Because `center = ref - 0.5 * size * align`, when placing a surface plate on `anchor=[ax, ay, az]`, **set `align` to the OPPOSITE sign (`align ≈ -0.55 * anchor` to `-0.80 * anchor`)** so ~20–35% seats inside the parent box and ~65–80% protrudes cleanly!
   - Match or harmonize `rotate` on tiny child studs/sparks placed onto an already-rotated parent blade so corners don't miss or over-penetrate.
   - Add `blend_with: [neighbor_box_1, ...]` whenever a sculpted plate intentionally bridges across into a neighboring bone/box.
2. **Non-Uni-Sheet Multi-Bone Wings (Mandatory on ALL `is_winged=True` Dragons)**:
   - NEVER build a wing membrane as one big flat rectangle!
   - Build an articulated wing rig (`wing_arm_l` -> `wing_forearm_l` -> `wing_hand_l` -> 3–4 articulated finger/cloak/vane/thruster bones `wing_finger_1_l`, `wing_finger_2_l`, `wing_finger_3_l`, `wing_finger_4_l` via `mirror_x`), with **separate `plane` membrane bays on each finger/arm bone** (`>= 6` wing bones total across L+R) and `layer: 1` / `layer: 2` so overlapping bays never Z-fight when folding!
3. **`plane` `edge:` Cut-Out Side Mapping (`texel_density: 32` = `2 px/unit`)**:
   - `normal: x` (`east` face): `side=top` = `+Y` (upward dorsal spikes); `side=bottom` = `-Y` (hanging beard/icicles); **`side=right` = `-Z` (backward-trailing fur/fin edge!)**; `side=left` = `+Z`.
   - `normal: z` (`south` face): `side=top` = `+Y` (upward collar/crown spikes); `side=bottom` = `-Y` (hanging cloak/ruff).
   - `normal: y` (`up` face, horizontal wings): **`side=top` = `-Z` (trailing wing membrane bays!)**; `side=bottom` = `+Z`; `side=right` = `+X`.
   - Use bold cut depths (`depth=4..8` on `scallop`/`ragged`, `min_len=3, max_len=8` on `spikes`/`drip`) so cuts read crisply at `2 px/unit`.
   - Never place two `plane`s on the exact same plane with the same `layer:` (`coplanar_z_fighting`); alternate `layer: 1` and `layer: 2` or angle them in a slight `±12°` V!

---

## 4. Exact Block-DSL v4.1 Grammar & Mirrored Animation Rules
- Valid `@materials` `family=` values ONLY:
  `keratin_scales`, `patterned_skin`, `painted_metal`, `brushed_metal`, `polymer_dielectric`, `woven_fabric`, `solid_wood`, `weathered_stone`, `lichen_biofilm`, `thin_film_iridescence`
- Valid `coat=` values ONLY:
  `plain`, `mottled`, `banded_limb`, `striped`, `scale_rows`, `spotted`
- Clip header syntax:
  `clip roar { duration_s: 2.4, loop: false ... }` (`loop` MUST follow `duration_s` with a comma on the same header, never on a separate line without a comma!).
- **`mirror_x` Bone Naming in `@behavior` Keyframes**:
  - When `bone fore_shoulder_l ... mirror_x(fore_shoulder_r)` contains child `bone fore_arm_l`, the mirrored root bone is `fore_shoulder_r`, and its mirrored child bone is **`fore_shoulder_r__fore_arm_l`** (NOT `fore_arm_r`!).
- Keep single-bone keyframe angles within `[-165, 165]` degrees (distribute big flips across `pelvis` + `waist` + `chest` + `tail_1..5`) so `SO(3)` Euler extraction stays smooth without `±180°` wrap jumps, and omit restrictive `joint_limits` on acrobatic spine/wing/limb bones so vigorous Monster Hunter attacks express full motion!
