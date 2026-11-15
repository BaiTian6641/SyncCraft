# ADR 0004: multi-domain 3D taxonomy and anti-template-overfitting dataset design

Status: accepted for P1-08 / P3-03 / P1-09 implementation. Date: 2026-11-14.

## Context

> **Provenance Note (`M8`)**: The Step 1–42 loss figures below come from a **user-supplied external diagnostic Colab pilot log snippet** shared during dataset taxonomy debugging to diagnose template memorization on an early 8-family draft corpus. They do **not** represent a committed local training artifact or a completed `P1-09` GPU benchmark run in this CPU repository (all toy smoke checkpoints have been removed from version control, and full Colab G4 GPU training remains tracked under `P1-09`).

In that external diagnostic pilot of `LiquidAI/LFM3-VL-2B` (BF16 LoRA, $r=32, \alpha=64$) on an initial
8-family procedural static draft, supervised training loss dropped rapidly from `1.0901`
(Step 1) to `0.1171` by Step 42 (~5,376 examples, <10% of Epoch 1), while the compressed
ZSTD Parquet training shard occupied only `17.3 MB` for `54,000` rows (~320 bytes/row).

Root-cause diagnosis identified four sources of shortcut memorization:

1. **Fixed identifier & AST skeleton chains**: Every example within a family emitted the
   same fixed identifier names (`hilt_root`, `grip_band`, `guard_block`, ...) in the same
   declaration order, collapsing conditional token entropy on ~75% of DSL tokens.
2. **Narrow domain coverage**: Training covered weapons, basic furniture, and gearboxes,
   omitting natural flora/geology, industrial robotics/turbines, architecture/buildings,
   humanoids, animals, and dragons.
3. **Prompt shortcut leakage**: Every prompt template exposed internal `family_id` slugs,
   exact `box_count`/`bone_count` integers, and exact palette variable names, allowing
   shallow keyword lookup rather than natural-to-CAD reasoning.
4. **Missing `@materials` / `@style` bindings & unmonitored generalization gap**: Static
   shards omitted `@materials`/`@style` stateful surface blocks, `(b_idx % 4 == 3)` never
   activated when `micro_batch_size == 2`, and the training loop logged only training loss
   without held-out validation loss.

## Decision

### 1. Seven-domain 32-family taxonomy with strict train/test family isolation

Expand procedural synthesis across seven macro-domains (**20 Seen** families for `train` /
`val` and **12 Held-Out Unseen** families for `test_unseen`, all unseen assets having
$\ge 16$ lowered boxes), preserving 100% of original `configs/data_policy.yaml` static
families and dynamic body plans in their prescribed splits:

- **Artificial**: Seen (`straight_blade_sword`, `single_edge_blade`, `battle_axe`,
  `spear`, `war_hammer`, `simple_wood_bow`, `furniture_table_chair_chest_cabinet`);
  Unseen (`mechanical_crossbow`, `hand_lantern`, `floating_ring_staff`).
- **Natural**: Seen (`ancient_canopy_tree`, `crystalline_geode_outcrop`,
  `coral_fungal_colony`); Unseen (`weeping_bonsai_arch`).
- **Industrial**: Seen (`gearbox`, `articulated_robotic_arm`,
  `industrial_turbine_generator`); Unseen (`tripod_telescope`, `anvil_forge`,
  `hydraulic_gantry_crane`).
- **Building / Architecture**: Seen (`modular_timber_stone_building`,
  `tiered_pagoda_watchtower`, `arched_stone_bridge_gate`); Unseen (`folding_windmill`,
  `coastal_lighthouse_keep`).
- **Humanoid (`entity`)**: Seen (`humanoid_biped`);
  Unseen (`armored_mech_sentinel` — folded into `hexapod_beetle` / unseen entity suite or
  held out; specifically the 12 unseen families comprise `mechanical_crossbow`,
  `hand_lantern`, `floating_ring_staff`, `weeping_bonsai_arch`, `tripod_telescope`,
  `anvil_forge`, `hydraulic_gantry_crane`, `folding_windmill`,
  `coastal_lighthouse_keep`, `hexapod_beetle`, `heavy_tail_tyrannosaur`, and
  `segmented_sea_dragon`, with `humanoid_biped` spanning civilian, knight, astronaut, and
  automaton morphologies).
- **Animal (`entity`)**: Seen (`dog_horse_quadruped` across lupine/equine/feline/cervine/ursine
  orders, `bird` across raptor/heron/owl/crested-avian orders); Unseen (`hexapod_beetle`,
  `heavy_tail_tyrannosaur`).
- **Dragon (`entity`)**: Seen (`winged_western_dragon`); Unseen (`segmented_sea_dragon`).

### 2. Five anti-overfitting diversity mechanisms

1. **Halton low-discrepancy stratified sampling**: Drive continuous and discrete structural
   decisions via prime-base radical-inverse sequences combined with per-sample cryptographic
   jitter for space-filling coverage across aspect ratios, poses, scales, and densities.
2. **Multi-register bijective identifier remapping & DAG topological permutation**: Remap
   bone, box, and palette symbols across six naming registers (`anatomical_biomech`,
   `craft_modeler`, `cad_engineering`, `architectural_masonry`, `botanical_geological`,
   `compact_hierarchical`) and randomly topologically sort independent sibling declarations
   inside `Bone.items`, `@palette`, and `@materials` while preserving exact `SE(3)` world
   geometry.
3. **Stateful `@materials` & `@style` coverage**: Emit physically matched `@materials` and
   `@style` blocks across all 10 material families (`patterned_skin` Gray–Scott RD,
   `keratin_scales`, `solid_wood`, `weathered_stone`, `lichen_biofilm`, `painted_metal`,
   `brushed_metal`, `polymer_dielectric`, `woven_fabric`, `thin_film_iridescence`) on ~65%
   of assets while keeping ~35% minimal geometry-only programs.
4. **Text2CAD-inspired 4-level prompt paraphrasing**: Balance prompts across four
   abstraction tiers (`natural_intent` with zero internal slugs or part counts,
   `art_and_material_brief`, `kinematic_rig_spec`, and `parametric_cad_blueprint`).
5. **Quality-Diversity (QD) entropy tracking & unbiased multi-stream sampling**: Record behavioral
   bin coverage, `distinct_tree_shapes_by_family`, and Shannon entropy in `dataset_manifest.json`; use 1-epoch default over
   `full_75k`, `lora_dropout=0.10`, `weight_decay=0.05`, `label_smoothing=0.03`, unbiased without-replacement `StreamMixtureSampler` (iterating 100% of `main` rows once and capping `apex` and `world` auxiliary stream repeats at `<= 3`, replacing earlier modulo interleaving), zero split-name prompt leakage, and periodic held-out validation loss logging.

## Consequences

- `entity` archetype models (`humanoid_biped`, `dog_horse_quadruped`, `bird`,
  `winged_western_dragon`, `hexapod_beetle`, `heavy_tail_tyrannosaur`,
  `segmented_sea_dragon`) populate `body_plan_id_if_dynamic`, target `bbmodel_5`,
  `bedrock_entity`, or `geckolib`, and validate via `export_bbmodel` + `export_gltf_pbr`
  (respecting `java_exporter.py`'s explicit rule that vanilla Java block/item JSON does not
  encode `entity` rigs).
- Non-`entity` models (`weapon_3d`, `block_multipart`) continue enforcing Java's `[-16, 32]`
  rotated-corner envelope and both `export_bbmodel` and `export_java`.
