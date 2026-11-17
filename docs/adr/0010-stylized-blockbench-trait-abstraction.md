# ADR 0010: Per-Species Sculpted Blockbench Construction, Calm x32 Cel Art, and Stage-by-Stage Visual Refinement

Date: 2026-11-17

Status: Accepted for the `research_nc_fan_reference` Monster Hunter Block-DSL v4.0 study corpus (`mh_dragons_v8`).
Supersedes: Legacy V8 cone/tube generator (`v8_sculptor.py` / `clade_*.py`) AND the shared-stencil `16px/block` prototype (`sculpted_kit.py` / `sculpted_*.py`).

## Context

Inspection of both early cone-snout rigs and the subsequent shared-stencil `sculpted_kit.py` prototype against:
- `.agents/skills/blockbench-sculpted-style/SKILL.md`
- The 8-page reconstructed `Lagiacrus` Blockbench benchmark (`Lagiacrus_Art_Notebook.pdf`: `974` elements, `1024x512` calm `x32` pixel atlas, Stage 1 -> Stage 2 -> Stage 3 regional close-up review)
- Community sculpted Blockbench dragon exemplars (`zyhfisme` Zinogre, Red-and-Cream Western Dragon, Tan Club-Tailed Wyvern, Blue Ice Dragon)
- Harvested multi-source official renders, screenshots, concepts, and 3D models (`references/mh_wide_boards/`)

identified five architectural flaws in the shared-stencil `16px/block` pass:

1. **Shared anatomical stencils erased species silhouette diversity**:
   Routing distinct morphologies (e.g., long U-snouted aquatic *Lagiacrus*, broad bulldog *Ruiner Nergigante*, narrow fox-snouted *Mizutsune*, anvil-headed *Jin Dahaad*) through one shared head/muzzle generator gave unrelated dragons uniform snout shapes and posture.
2. **Per-segment `tip_color` and `horn_keratin` gradient resets caused candy-cane striping**:
   Material-level `tip_color` plus zero-to-four axial resets on multi-segment `tapered_arch` chains painted alternating rings across horns and fin rays.
3. **Noisy periodic wing/hide textures and bloom washout**:
   Periodic sine stripes (`wing_rathalos`, `wing_legiana`), `0.24` Bayer checkerboards, `+`-shaped seam crosses, and heavy soft bloom washed out crisp form reading.
4. **Occluded throat scutes & single-sided/dark-backed webbing planes**:
   Sunken throat plates Z-clipped behind opaque neck/chest cores, and dark-navy `web_back` materials hid webbed cheek frills (`hood_web`) and paddle fins when viewed from anterior 3/4 angles.
5. **Uncompensated `roar` spine pitch**:
   Additive backward chest and neck pitches without counter-pitch on `head` tipped roaring skulls past vertical and cracked chest-neck ventral seams.

## Decision

1. **Template-Free Geometry Primitives + Per-Species Sculpt Modules (`dsl_primitives.py` + `species/*.py`)**:
   - Remove `sculpted_kit.py` and `sculpted_*.py` completely. Every creature (`22 / 22`) owns its own bespoke module under `src/synccraft/data_gen/mh_dragons_v8/species/` built solely from low-level SE(3) primitives (`RigCanvas`: `bone`, `box`, `span_box`, `tapered_arch`, `shingle_arc`, `cutout_panel`, `winged_sail_rig`).
   - Support true 3-stage authoring (`stage_cap = 1, 2, 3`) for verifiable Stage 1 primary-mass blockout -> Stage 2 regional anatomy -> Stage 3 refined detail progression.
   - Express genuine species- and variant-level morphology (e.g., `Chaotic Gore Magala` true `-X` golden Shagaru molt vs `+X` unbroken Gore side; `White Fatalis` branching antlers + flowing white mane; `Rey Dau` articulating `rail_horn_l/r` calipers; `Arkveld` twin 3-link `chain_1..3` braided chainblades; `Jin Dahaad` anvil T-crest & antique bronze radiator towers).
2. **True `32 px/block` (`2 texels/unit`, `1024x512` POT) Calm Cel Painter (`v8_painter.py`)**:
   - Paint `88%` calm midtone hides with sparse `~7–8%` directional scale lips, `1 px` overlap shadows on matte belly scutes, lengthwise horn chitin grain, distal-only `tip_color` transitions (`dsl_primitives.tapered_arch` terminal segment only), organic wing gradients without periodic stripes, hairline diagonal energy fissures, binary `{0, 255}` cutout silhouettes (`hood_lobe_cutout`, `ear_frill_cutout`, `tail_paddle_fin`, `sakura_fin`, `fur_tuft`), and tight focal emissive masks (`<= 4%` surface glow).
3. **Regional Stage Inspection & Animation Calibration (`v8_renderer.py` + `v8_animator.py`)**:
   - Render `render_stage_and_regional_sheet` (Stage 1 clay + Rear 3/4 + Head H/S/F + Limbs + Tail + Silhouette + Atlas Swatches) alongside the 8-panel `<id>_studio_review.png`.
   - Counter-pitch `head` forward (`+16°..+18°`) in `roar` and `leviathan_thunder_discharge` with coordinated forelimb bracing so jaws aim toward the hunter.
4. **Verification & Governance (`tests/test_mh_dragon_dataset.py`)**:
   - Enforce `260 <= elements <= 1200`, `cuboid_ratio >= 0.80` (`>= 0.85` overall), `bones >= 22`, `wing_banks >= 6` on winged wyverns/elders, `0.0` (`< 1e-4`) SE(3) & Bedrock vertex drift, zero family leakage across the `110` supervision rows (`75 / 20 / 15`), strict `research_nc_fan_reference` (`commercial_weights_eligible = false`), and `runtime_import_verified = false`.

## Consequences

All 22 Monster Hunter creatures compile natively via Block-DSL v4.0 (`6,277` 3D cuboids + `382` cutout cards = `6,659` elements, `94.3%` cuboids, `22/22` `1024x512` `32px/block` atlases, `264` FK clips, `0.0` compiler/Bedrock parity error) with unmistakably distinct anatomy across every clade and variant.
