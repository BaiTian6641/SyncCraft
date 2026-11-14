# ADR 0005: Categorized 55-Family Assistant World-Model Dataset, Simple-to-Complex Curriculum, Concise Texture/Animation Synthesis & Batch Statistical QC

- **Status**: Accepted
- **Date**: 2026-11-14
- **Tasks**: `P1-08-r3`, `P1-03e`, `P3-01-prep`
- **Supersedes / Extends**: `ADR 0003` (Stateful Surface-Material World Model), `ADR 0004` (7-Domain Anti-Overfitting Dataset)

## Context

SyncCraft's objective is an **assistant-based 3D modeling, texturing, and animation world model** grounded in explicit kinematics, physically/biologically interpretable surface parameters, and collision-free motion laws — **not** an opaque pixel/voxel denoising diffusion black box.

Following `ADR 0004`, five high-leverage dataset engineering extensions are required before scaling batch synthesis and Hugging Face Hub publication:
1. **Broad Categorized Coverage**:
   - **Dragon / Fancy Imaginary Creatures (`dragon`)**: Expand beyond Western Dragon + Sea Dragon to cover distinct mythic/prehistoric morphologies: **Dragon** (4 legs + 2 wings), **Drake** (4 legs, wingless armored), **Lung / Eastern Dragon** (serpentine 4-clawed antlered, wingless), **Wyvern** (2 hind legs + 2 forelimb wings + stinger tail), **Lindworm** (2 forelegs + serpentine body), **Zilant** (crowned bipedal winged serpent), **Raptor** (feathered dromaeosaurid sickle-claw biped), **Amphithere** (legless hooded feathered-winged serpent), and **Hydra** (polycephalic 3-necked dragon).
   - **Industrial, Mechanical Engineering & Vehicles (`industrial`)**: Expand beyond gearboxes/turbines/arms to reciprocating **piston-crank engines**, **CNC milling presses**, **centrifugal pump & valve skids**, and **land/construction/rail/air/marine vehicles** (**haul dump trucks**, **tracked hydraulic excavators**, **steam rail locomotives**, **twin-rotor helicopters/drones**, **cargo harbor tugboats**, plus held-out **mast forklifts** and **deep-sea ROV submersibles**).
   - **Craft, Flora, Architecture & Fauna**: Add **pendulum clocks**, **alchemical distillation benches**, **tiered pine conifers**, **rotating-paddle waterwheel millhouses**, and **aquatic manta/shark swimmers**, while keeping **Humanoid** concise (`humanoid_biped`) with an explicit `human_gathered_extension` schema slot reserved for the upcoming real-human model corpus.
2. **Hierarchical Categorization**: Every record must carry structured `taxonomy` metadata (`macro_domain`, `category`, `sub_category`, `complexity_tier`, `motion_archetype`) so batches and evaluations slice cleanly along any semantic or mechanical axis.
3. **Simple-to-Complex Progressive Curriculum (`tier_1_simple` -> `tier_4_masterwork`)**:
   - A modeling assistant must know both how to lay down a clean 4–8 box silhouette blockout (`tier_1_simple`) and how to refine that blockout step-by-step into a 16–36 box articulated rig (`tier_3_complex` / `tier_4_masterwork`).
   - All `test_unseen` Gate G2 records strictly preserve `boxes >= 16` per `docs/ACCEPTANCE.md`.
4. **Concise Unified Model + Texture + Animation Path**:
   - Rather than treating texture as raw pixel noise or animation as unverified frame dumps, each asset couples:
     1. Typed `Block-DSL v3.0` rig (`@meta`, `@palette`, `@rig`),
     2. Concise parametric `@materials` + `@style` substrate/coating/Turing Gray–Scott & LabPBR/vanilla recipe, and
     3. Concise analytic `@animation` bone-channel specification (`harmonic_hinge`, `wing_flap_phase`, `serpentine_traveling_wave`, `wheel_rotor_spin`, `alternating_gait`) whose amplitudes are **swept-SAT verified across cycle keyframes**.
5. **Batch-by-Batch Statistical QC + Controlled `5%` Bad-Data Diagnostic Repair**:
   - Inspecting 50k–100k models individually by eye does not scale. Instead, generation runs **batch by batch** with automated **Statistical Process Control (SPC)** gates (`compute_batch_statistical_qc`) plus **stratified random-sample 3D PBR + Posed Keyframe + UV Atlas contact sheets (`.png`)** for targeted visual inspection.
   - Each assistant batch intentionally mixes **95% `verified_clean` examples** (`create_full_asset`, `coarse_to_fine_upgrade`, `texture_and_animate_rig`) with **5% `controlled_defect_with_repair` examples** (`diagnose_and_repair_defect`), pairing realistic geometric/contact/swept-motion defects + exact auditor diagnostics with root-cause explanations and 100% SAT-verified repairs.

## Decision

1. **55-Family Categorized Taxonomy (`40` Seen + `15` Unseen)**:
   - Formalized in `src/synccraft/data_gen/taxonomy_catalog.py` and `configs/data_policy.yaml` across `7` macro-domains and `21` engineering/biological categories (`40` Seen families for `train`/`val`, `15` Held-Out Unseen families for `test_unseen`).
2. **4-Tier Progressive Complexity & Assistant Modes (`src/synccraft/data_gen/curriculum_and_assistant.py`)**:
   - `coarsen_program_to_tier` derives physically connected, SAT-verified `tier_1_simple` (`4–8` boxes) and `tier_2_medium` (`9–15` boxes) blockouts directly from full rigs, enabling exact paired `(P_coarse -> P_detailed)` coarse-to-fine supervision.
   - `synthesize_verified_animation_spec` binds domain-appropriate analytic bone motion laws and clamps amplitudes under multi-keyframe swept 15-axis SAT so clean animations never self-intersect mid-swing.
   - `build_controlled_defect_repair_pair` generates the explicit `5%` bad-data diagnostic-repair curriculum (`sibling_obb_collision`, `floating_detached_gap`, `coincident_copenetration`, `swept_animation_collision`) verified negative before repair and 100% positive after repair.
3. **Batch Generation, Statistical QC & Contact-Sheet Visual Inspection (`src/synccraft/data_gen/batch_pipeline.py`)**:
   - Emits self-contained batch shards (`batch_XXXX.jsonl` + ZSTD `batch_XXXX.parquet`), `batch_XXXX_qc.json` statistical reports, and stratified random-sample `.png` contact sheets combining 3D Z-buffered PBR rest renders, posed animation keyframe renders, and 2D UV albedo atlases.

## Consequences

- **Positive**: Teaches the model *how* to construct, refine, texture, animate, and debug 3D block models across simple-to-complex tiers, 10 draconic morphologies, and 16 mechanical/vehicle families without memorization or black-box pixel diffusion.
- **Guardrails**: Gate G2 (`test_unseen`) remains strictly `100%` clean and `>= 16` boxes; controlled bad-data examples are explicitly tagged (`quality_label = "controlled_defect_with_repair"`) so broken geometry is never trained as a positive target.
