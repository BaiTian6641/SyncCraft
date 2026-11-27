---
name: "blockbench-sculpted-style"
description: "Use when designing, critiquing, training or generating Blockbench/Minecraft-style 3D models (creatures, real animals, humanoids, bosses, props, armor, vehicles, mechs, scenes) in the formal Vigorous Carved Blockbench aesthetic (`16x` or `32x` high-detail cuboids + `2x` focal eye/lens `uv_scale`, 4-part periorbital eye carving, anti-toy sculpted tails/plates/accessories, and DeepSeek-inspired Visual Primitives native-resolution edge/skeleton tracing — engine-agnostic across .bbmodel, Bedrock .geo.json, GeckoLib, and SyncCraft Block-DSL). Pass it to every subagent."
---

# Sculpted Blockbench style (v45.0 — Declarative `.texdsl` Texture Language + Part-Aware Semantic Entity Detailing & Orc/Character PBR Kit + Animated Hold/Ping-Pong `.mcmeta` & `_e` Emissive Flipbooks (`TexGraph v8`: `125` Nodes, `22` PBR Subgraphs, `31` Presets, Kahn DAG Validation, Incremental Cache, Multi-Domain QA) · Flexible Interlocking Semantic UV v3 `93–97%` Padded · Full 10-Perspective + Bilateral Eye Check · `>=8.5/10` Trait Gate)

This skill defines a formal, standalone **3D artistic medium and universal spatial-reasoning discipline** — **not** a rigid skeleton template and **not** specific to any single game franchise. It fixes *how* forms are conceived in 3D space across **real animals, imaginary creatures, stylized humanoids, vehicles/airships, battle mechs, armor/props, and architectural dioramas**: starting from multi-perspective **1:1 native-resolution references & ad-free video timelapse + transcript studies** analyzed via **DeepSeek-inspired Visual Primitives (`scripts/visual_primitives_prep.py` — zoom FIRST, then perform fine `0.10` coordinate grid & structural outline/edge trace ON every zoomed region via `zoom --autocrop --with-grid --with-edges`, plus dashed skeleton centerlines `- - -`, joint bend angles `θ°`, concavity polarity `CONCAVE_UP/DOWN/S_INFLECTION`, and proportion calipers)** and **explicit temperament polarity (`Gentle & Approachable` vs. `Furious & Unapproachable`)**, packed from honest cuboids/low-poly meshes across a **human-artist multi-iteration pipeline (`>= 3` White-Model passes -> `2–3` Texture passes -> `>= 2` Animation passes)**, carved with **post-zoom species-calibrated snouts (`snout_ahead_of_eye`: `40–48%` lupine/brute, `52–56%` sleek wyverns/vampire dragons, `56–60%` regal long-jawed elder dragons — never over-stretched OR over-shortened!) + 100%-closed flush rest jaws (ZERO exposed teeth at rest!) opening into sculpted V-ribbed oral cavities & cohesive `32x` tapered fangs** (for organic creatures) and **recessed physical cavities, glazed 3D-interior cockpits, alternating painted armor vs. dark actuators, UV Seam Tool + checkerboard pixel-snap hygiene** (for mechanical/hard-surface models; `references/hard_surface_and_mechanical_craft.md`), carved at **`16x` or `32 px/block` with `2x` focal `uv_scale` (`32x–64x`) on eyes/optical lenses** plus **4-part 3D periorbital sockets and anti-toy faceted/shingled tails, plates, and collared accessories**, and animated through an **industry-grade universal `SO(3)` / `SE(3)` / `Sim(3)` biomechanical + mechanical solver** (`auto_rig`, multi-joint fold-preserving N-bone IK, closed-loop Standing Limb IK Pinning `ground_pin_limbs`, mandatory **animated feet & multi-phalanx toe verification**, serpentine neck Spline-IK, 3D COM tail counterbalance, Reverse-Foot Rocker + digit articulation/grasp, aerodynamic flight, whole-body `Sim(3)` muscle bulge, acute-fold `160°` SAT, and rigid vehicle/mecha axles & hinges) separated cleanly from entity movement (`in_place: true`).

**The formal aesthetic ("unmistakable Blockbench silhouette, master-sculpted craft on second look"):** proud, honest cuboids (`90°`, `45°`, `30°`, `22.5°`, `15°` planar box facets) and clean low-poly hard-surface chamfers **packed `40–85%` deep into compound muscular, architectural, vehicular, and armored volumes**; species-compact heads with recessed 4-part periorbital eye/lens sockets (`2x` focal pixel depth) and deep V-ribbed oral cavities; tapered multi-plane tails, staggered overlapping shingle plates, recessed wheel arches/cockpits, and bracketed/collared accessories; low-slung coiled weight or clear mechanical ground clearance; wide splayed articulated multi-phalanx extremities; 3D vaulted full-chord or feathered alpha wings; and crisp **`16x`/`32x` cross-cube pixel-grid relief carving** (transverse bevels, imbricated U-scales, overlap shadow valleys, directional striations, clean hard-surface panel seams/decals/glass reflections, and `2 px` organized pixel clusters — never blurry airbrushed noise!). **Blockbench is Blockbench — NEVER treat Blockbench as a substitute for Blender** (`G11`), and **NEVER collapse into toy-like Lego minimalism** (zero bare 1-box stick limbs, zero blunt rectangular tail/plate planks, zero stretched crocodile snouts on wolves/dragons, zero frozen ski-paddle feet in animations, zero 2-pixel sticker eyes, zero floating un-collared accessories; `G8`, `G12`, `G13`, `G14`).

Models that fail this style rarely fail at *naming* the subject's traits. They fail at **medium-honest 3D spatial translation**: either under-building into stiff, non-overlapping toy sticks on tall stilts (often induced by naive collision rules or `place()` surface stickers), OR over-slicing shapes like Blender to fake smooth $C^2$ curvature, OR stretching snouts (`>60%` ahead of eye) and leaving feet/toes unrigged, OR homogenizing distinct species/machines onto one generic box skeleton. Every rule, spatial curriculum stage, and `G1–G14` gate below instills genuine human master Blockbench craft instead.

## 0. How to use this skill

- **Think like a master sculptor carving in honest Minecraft cuboids and crisp low-poly geometry, never a Blender curve-slicer or a coordinate sticker-placer.** Decompose references through the **6-Stage Vision & Spatial Reasoning Curriculum (`5.1d`)**: establish world scale and coiled center-of-gravity (or vehicle ground clearance) first, pack `3–7` intersecting cuboids deeply into muscular/carapace/hull envelopes (`3.1c`), stop adding cubes the moment `16 px`/`32 px` pixel-relief carving carries the detail better (`10.2b`), and never let experimental harness quirks flatten your silhouette (`11.0`).
- **Read it in full** before the first reference review, the first construction step and the first critique of every pass. Consult:
  - `references/hard_surface_and_mechanical_craft.md` (**NEW `v35`:** Universal craft guide for vehicles, mechs, walkers, airships, turrets, robots & hard-surface props — Generic Mesh vs. Cuboid-only topology, `180°/N` flat-tread wheel math, recessed wheel arches/beds/glazed 3D cockpits, alternating armor vs. dark mechanical actuators, UV Seam Tool + checkerboard pixel-snap hygiene, and rigid mechanical rigging);
  - `references/cranial_and_jaw_structural_reasoning.md` (**UPDATED `v37` §8–§9.3:** Anti-Stretched-Snout Cranial Compactness Law `snout_ahead_of_eye / head_L`, **V-Shaped Sloped Head/Mandible Faceting, Pure Reference Snout Material Boundaries (`0` unreferenced `snout_ridge` blocks!), Open-Basin Lower Jaw Trough, Half-Phase Interlocking Sawtooth Occlusal Bite Wave, Converging Alveolar Gum Rails & 2-Tone Spaced `0.5u` V-Arch 3D Dentition, Upturned Rhinarium Bump, Curved `CONCAVE_UP` Chin Prongs & Taut Striated Buccal Webs**, plus `v34` Curvature Concavity Verification);
  - `references/unified_biomechanical_animation_solver.md` (**UPDATED `v35` §7B #12:** Mandatory Foot & Multi-Phalanx Toe Animation Verification Law — `SO(3)` flat soles, full `bio_joint` digits, `auto_rig: true`, `foot_roll`, `digit_grasp`, `ground_pin_limbs`, `moving_digits >= 16`, and close-up animated foot/toe zoom gate);
  - `references/high_res_focal_carving_and_visual_primitives.md` (**UPDATED `v37` §1, §1D–§1H:** Proactive Web, 3D-Viewer & **Stylized Spin-Off Abstraction (e.g. *Monster Hunter Stories*)** Reference Search Law, DeepSeek Visual Primitives native-resolution edge/skeleton/concavity pipeline via `scripts/visual_primitives_prep.py`, **Multi-Track Body-Spike & Spherical Upper-Limb Ring Mapping + Zero-Dummy-Brick Direct Bone-Pivot Placement (`ZERO` leftover base/keel cubes!) + Parent-Chain Pitch Compensation, Multi-Host Protrusion Audits (incl. Horizontal Side-Spreading Tail Spikes!), Anti-Square-Block Torso Chest Bump & Exposed Under-Neck Segmented Throat / Stepped Fur Carving, Track-Following Stylized `32x` Glow Patterns (`<= 4%`), Living Artist Trait-Placement Ledger & Pre-Signoff Region-by-Region Review Discipline**, `32x` + `2x` focal `uv_scale` eye/lens carving, and anti-toy tail/plate/accessory craft);
  - `references/universal_anatomy_reasoning_and_self_check_principles.md` (**NEW `v38`:** Franchise-agnostic, first-principles Blockbench spatial reasoning & double-check framework — exact rotation sign derivation, mandatory 8-Ratio Deep Facial/Focal Proportional Remapping, Scale-Harmonious Eye Resolution & Simplification Law, automated 4-invariant `G14` eye audit, Zero-Blink `0.5u` Dual-Sided Membrane Slabs & deep structural burial, anti-wedding-cake organic massing, isolated per-digit collision hygiene, full-body land/air/underwater `swim {}` secondary-chain motion, and the repeatable 7-stage expert multi-pass self-check loop);
  - `references/texgraph_master_texturing_study.md` (**NEW `v45.0`:** Master Blockbench & Pixel Art Texturing Study distilled from 16 expert YouTube timelapses — 4-Stage Studio Blockbench Progression, 1 px Cast-Shadow Material Seam Rule, Orc/Goblin/Humanoid Skin + Forged Steel + Stitched Leather + Woven Cloth + Keratin/Ivory recipes, Anti-Blanket Part-Aware Landmark Rules, `.mcmeta` Hold/Ping-Pong/Emissive Animations, and `.texdsl` declarative programming);
  - `references/aesthetic_mood_spectrum_and_intent_router.md` (`Domain D1–D5 × Mood M1–M4` Intent Router), `references/vigorous_carved_aesthetic_and_spatial_reasoning.md`, `references/youtube_master_distillation.md` (`50+` master YouTube tutorials, transcripts & `1080p` ad-free video studies), and `references/ecology_and_whole_body_balance_animation.md`.
- **Subagents & Anti-Drift Scope Discipline (`v39.3` — Mandatory Before EVERY Model!):**
  1. **Strict `1` Monster (or `1` Base + Shared-Skeleton Color Variants) MAX per Fresh Subagent (`Model="inherit"`):** NEVER pack `2+` distinct bespoke skeletons into one subagent session — multi-monster batching causes context compression, rushed basic blocks, skipped references, and flat plastic-toy gradients! Always run on the full top model (`Model="inherit"`), never a weaker/simplified model for 3D art.
  2. **Mandatory Multi-Source Internet + YouTube Ecology/Gameplay Research (`Stage 0 + A0`):** Before placing Cube `#1` on ANY creature, gather official multi-angle renders/concept art AND study YouTube ecology/gameplay/turnaround footage so silhouette, feather/fur/scale layering, eye integration, and movement physics are 100% grounded in real visual references — NEVER build generic basic blocks from imagination!
  3. **Mandatory Fresh `read_file` of `SKILL.md` Before EVERY Single Stage (`W1`, `T1`, `A1`):** Never script all passes in one blind batch! Re-read the relevant sections of this latest `SKILL.md` before sculpting, before painting, and before animating, inspecting full rendered outputs visually at every pass!
  4. **Mandatory `v39.3` Anti-Flat-Gradient / Anti-Toy Pixel-Relief Law (`jump>25 ∈ [4.5%, 14.0%]`, `lum_step ∈ [5.0, 8.5]`):** NEVER paint simple smooth airbrush gradients (`jump>25 < 3.5%` = **FAIL "Plastic Toy Flat Gradient"**)! Every `32x` creature MUST display crisp, deliberate, hand-painted Blockbench pixel-art relief: organized `2 px` stepped pixel clusters, `1–2 px` dark plate/feather/scale/fur-tuft overlap valleys, `1 px` sunlit bevel speculars, and `45–50°` hue-shifted cool-shadow-to-warm-light color ramps (zero random TV-static noise `> 18%`, AND zero lifeless airbrushed gradients `< 3.5%`!).
  5. **Mandatory `0–10` Reference-vs-Model Trait Rating Gate (`>= 8.5/10` Required at `W3` AND `T2`):** Open official references and rendered turnarounds/close-ups side-by-side via `read_file` and log an explicit `0–10` scorecard in `PROJECT_GOAL.md` across all 6 core traits: (1) Overall Silhouette & Sculpted Non-Toy Proportions, (2) Head Shape & Snout/Beak Ratio, (3) **Eye Surface Integration & Expression (`0` extruded eye bricks!)**, (4) **Flush Closed Sawtooth Jaw + Anchored Rear Hinge + Perimeter Inner Teeth (`0` exposed internal teeth at rest!)**, (5) Signature Anatomical Traits (layered wings/feathers/thrusters, horns/crests, shells/fins, limbs/claws, tail), and (6) **True `32x` Hand-Painted Blockbench Pixel-Relief Craft (`jump>25 ∈ [4.5%, 14.0%]`, `lum_step ∈ [5.0, 8.5]`) & Palette Fidelity**. **ANY trait rated `< 8.5/10` BLOCKS export** until fixed and re-rated `>= 8.5/10`!
  6. **Use `scripts/studio_render_kit.py` (DO NOT REINVENT RENDERERS OR METRIC SCRIPTS!):** Import `/Users/tianrui6641/BlockBench-Skill/skills/blockbench-sculpted-style/scripts/studio_render_kit.py` directly (`fetch_reference_images`, `render_8panel_turnaround`, `render_head_closeup_sheet`, `render_t1_unlit_squint`, `render_variants_showcase`, `render_a2_filmstrip`, `compute_atlas_metrics`, `run_full_verification`, `verify_bb5_wing_continuity`, `stage_swarm_screenshot`) so 95% of your effort goes into **sculpting, hand-painted `32x` pixel-relief art, and biomechanical animation** rather than boilerplate camera code! Every winged creature MUST additionally pass `verify_bb5_wing_continuity(bb_path)` (`0.000u` air gap across rest + all animated clips, zero detached `wing_l_f4` or front wing claws)!
  7. **Mandatory `v40.0` Semantic Anatomical UV Partitioning & Parametric Pixel Composer (`synccraft.texgraph`):** NEVER ship sparse, jumbled UV atlases (`13–38%` fill) or toy-like flat gradients! Pack EVERY entity atlas with **`repack_bbmodel_uvs(bb, mirror_symmetric=True)`** (or `allocate_uv_layout(..., packing_mode="semantic_maxrects")`) to partition islands cleanly by body-part zone (`head`, `neck`, `torso`, `wings`, `forelimbs`, `hindlimbs`, `tail`, `emissive`, `accents`) at **`>= 60%` padded utilization** with **Adaptive Hybrid L/R Mirroring** (hero focal zones unique, mirrored wings/limbs/accents shared) and verify the exported `render_uv_zone_map(...)` layout visually! Author expert studio pixel art — for **entities, seamless tileable blocks, and items (static or `.mcmeta` animated)** — using **`synccraft.texgraph`** (3D `MeshBakeContext` world-pos/normal/AO/edge-bevel/joint-cavity baking, tileable noises, Gray-Scott Turing, fixed geometric/scale/feather/fur/vein patterns, pure algorithmic non-ML denoisers Kuwahara/Bilateral/TV/Despeckle, OKLCH hue-shifted ramps, Selout, emissive extraction, declarative `.texgraph.json` recipes, and interactive HTML Node-Graph Inspector; see `§10.1b`).
  8. Give every subagent this skill, `references/hard_surface_and_mechanical_craft.md`, `references/high_res_focal_carving_and_visual_primitives.md`, `references/universal_anatomy_reasoning_and_self_check_principles.md`, `references/aesthetic_mood_spectrum_and_intent_router.md`, `references/vigorous_carved_aesthetic_and_spatial_reasoning.md`, `references/youtube_master_distillation.md`, `references/ecology_and_whole_body_balance_animation.md`, `references/cranial_and_jaw_structural_reasoning.md`, `references/unified_biomechanical_animation_solver.md`, `scripts/studio_render_kit.py`, `scripts/visual_primitives_prep.py`, `scripts/construct_check.py`, `scripts/verify_bbmodel_ground.py`, and the brief in Section 12.
- **Start clean.** Do not reuse models, scripts, textures or notes from earlier projects as templates. Reuse only this skill, your target subject's zoomed references, and SyncCraft.
- **Don't start from the last skeleton.** The exemplars, test flights and earlier models in the same project are evidence of quality, not starting points. Their skeletons, macros and poses are never a default. Every model starts from its own concept (5.0).
- **Numbers are bars, not blueprints.** The non-negotiables in Section 1 always hold. Every other number in this skill is a measured default or a check: head ratios, the wing "M" arm, finger angles, segment counts. Experienced Blockbench artists flex conventions whenever silhouette readability and world-fit clearly win — depart from a default when the concept needs it, write the reason in the log, and make sure the gate that number protects still passes.
- **Keep a project log** (`PROJECT_GOAL.md`) holding:
  - the non-negotiables;
  - the roster;
  - each model's current pass;
  - the owner's latest corrections, word for word.

  Re-read it at the start of every pass so instructions survive context compression.

## 1. Non-negotiables

1. **Proactive Web/3D-Viewer Reference Search + Regional Ultra-Zoom -> Unfiltered Full-Color Cutout + Post-Zoom Grid & Crisp 1 px NMS Contour Trace -> Category Analysis & Trait Reconstruction (`v37` Zero-Guessing Law).** **Whenever provided or cached references feel insufficient, obscured, or leave ANY angle ambiguous** (such as top-down dorsal/caudal spike tracks, neck-to-shoulder spread, upper-limb spike rings, exact snout material boundaries, or interior jaw/tooth geometry), **continue searching the internet (`search_web` + `browser` subagent on official 3D model turnarounds, interactive Sketchfab / 3D viewers, and artbook orthographics) until you have crystal-clear multi-angle references — NEVER guess unseen details!** Every box is a named anatomical or mechanical form (`maxilla_l`, `snout_bridge_slope`, `snout_lip_l`, `jaw_rail_l`, `arm_ring_1..6`, `sh_neck_start`, `pelvis_spike_1`, `tail_blade_1..6`). **ALWAYS zoom and cut out each region FIRST**, feeding BOTH the **100% unfiltered full-color native cutout (`<region>_native.png`)** AND the **3-panel post-zoom `0.10` grid + crisp 1 px bilateral color-tensor NMS contour plate (`scripts/visual_primitives_prep.py zoom --autocrop --with-grid --with-edges`)** into multimodal vision across **all 6 regions** (`Head/Canopy`, `Anterior Torso/Chest/Prow`, `Dorsal Back/Shoulders/Superstructure`, `Flanks/Ribcage/Waist/Side Hull`, **`Muscular Limbs/Ring-Spikes/Gauntlets/Paws/Toes/Wheels`**, `Tail/Accessories`). Never pin or redistribute external artists' reference images inside the skill folder — distill structural rules instead.
2. **Compound cuboid mass-packing & coiled predatory weight — never bare toy sticks, tall stilts, or Lego-stud stickers.** Primary anatomical cores anchor the body (`top-10` non-wing boxes hold `>= 50%` on standard creatures, `>= 36%` on hero monsters, and `>= 26%` on `240–550` element epic bosses; `G1`), while on hero and epic boss tiers **no major limb shaft, wing-arm, or neck segment may remain a single unadorned rectangular stick** (`G8`, `3.1b`) and heavy grounded quadrupeds/brutes/elders must hunker with **coiled low-slung center-of-gravity** (`belly_clearance <= 0.42x shoulder_height`, `G12`, `3.1c`). Sculpt muscles, stepped chamfers, and jaw/belly osteoderms by **sinking secondary cuboids `40–85%` deep into their host mass** (`3.1c`), never gluing flat bricks onto outer box walls.
3. **Blockbench is Blockbench — ZERO fake-Blender body wafers, ZERO unreferenced `snout_ridge` color blocks, & Scoped `0.5u` V-Sloped Cranial/Dental Craft (`v37`, `G11`).** Never slice torso, limb, or tail shafts into `>= 6` thin `< 1.0u` wafer steps to imitate smooth Blender $C^2$ curvature (`G11`), never distort `16 px` square texels with irregular decimals (`G7`), **never invent contrast-colored dorsal `snout_ridge` blocks where the official snout is pure primary hide/scale**, and never stack boxy `yaw = 0` rectangular snout steps (`G2`). **Scoped `v37` Cranial/Jaw/Dental Allowance:** On high-focus heads, snouts, serrated lips, converging `jaw_rail_l/r` mandibles, hooked chin prongs, and **individual `0.5u` 3D V-arch teeth** (`16–22` discrete teeth tucked 100% inside the closed upper lip at rest; `references/cranial_and_jaw_structural_reasoning.md §9`), purposeful `0.5u–1.0u` facets are **explicitly permitted (`construct_check.py` `G11`)** to sculpt razor-sharp **V-shaped sloped heads and clean volumetric V-arch dentition** without over-voxelizing the body!
4. **Real block horns, spikes and articulated multi-jointed digits with negative space.** A horn is `2–6` compound-rotated tapering box segments on child bones (`7.1`); a spike is a 2–3 step tapered rotated box chain (`7.2`); hero and boss hands/feet (`8.3b`, `G9`) use **4–5 splayed multi-segment digits** (`proximal -> distal phalanx -> hooked talon`, middle digit longest, plus opposable thumb or rear velociraptor dewclaw) — never a flat brick slab with 3 identical parallel peg toes, and never crowd the skull roof with parallel french-fry horn bundles (`G3`).
5. **Every rotation has a 3D reason:** the pose, the form's 3D wedge/chamfer (`pitch`, `yaw` and `roll` working together), a row's shared lean, or—strictly for clustered/stacked props and foliage (3.3, 8.7)—deliberate contact-pivot organic tilt (`5-15°`). Never leave paired face, shoulder or thigh wedges at `yaw = 0, roll = 0` (axis-aligned crate bias), and never add random jitter on creature anatomy or structural architecture. Mirrored parts get mirrored angles.
6. **Deeply Rooted Zero-Floating Attachments (`30–55%` burial), Multi-Track Body-Spike & Limb-Ring Mapping, and Horn/Spike Collision Policy (`v36–v37`).** Every limb spike, circumferential upper-limb spike ring (`arm_ring_1..6` wrapping around `(X_local, Z_local)` perpendicular to limb shaft `Y_local`), shoulder/back/tail spike, fur bolster, or mechanical bracket must bury **`30–55%` deep** inside its host limb, hull, or parallel fused mounting keel (`ulnar_crest_base`; OBB `sep <= -0.5u`, **0 floating parts**!). Map **distinct continuous body-spike tracks** on orthographic top-down + lateral grids (`references/high_res_focal_carving_and_visual_primitives.md §1E` — e.g., **Track A** originating below the neck, arching convexly over the shoulder, and running unbroken along `thorax -> waist -> pelvis -> tail_1..N` to the tail tip vs. **Track B** secondary inner dorsal row immediately behind the cranial horns — NEVER lump them into two isolated sideways shoulder bull-horns!). Apply the **Horn & Body-Spike Chain Collision Policy**: assign shared track `sculpt_group` tags (`hb_sc`, `trk_a_sc`, `trk_b_sc`, `arm_ring_sc`, `fsp_sc`) + explicit 1-hop cross-bone `blend_with` pairs along each continuous track while keeping full cross-limb & ground SAT enforcement active!
7. **Build symmetric.** Build only the left side (`-X`) and centreline (`X = 0`); the right comes from `mirror_x`. The rest (bind) pose is symmetric (Section 4). Attitude lives in animation or display clips, never in the rest pose.
8. **The head passes first.** Build and approve the head alone before the body (6.6). Heads are deep, narrow 3D wedges: the cranium is at least as tall as it is wide, and from the front the head without horns/frills is taller than wide. The jaw is a separate bone on anything player-sized or larger, closed and tucked under the upper jaw at rest (6.5).
9. **Deep Facial Trait Proportional Remapping + Tier-Harmonious Eye Resolution & Unblocked Recess (`v38.4`, `G14`).** Before placing Head Cube `#1`, complete the **8-Ratio Deep Facial Trait Analysis Table** (`references/cranial_and_jaw_structural_reasoning.md §10`) on zoomed `0°` Front, `35°` 3/4, and `85°` Side `0.10`-grid crops (`eye_h/head_h`, `eye_l/eye_h`, orbital elevation/depth, snout taper, inter-eye gap, brow overhang, ear/cheek anchors) so facial traits remap true species proportions rather than weird cartoon ratios. **Match eye resolution and socket complexity to creature scale (NEVER `uv_scale >= 3` `96x–128x` mixels!):**
   - **`hero` / `epic_boss` (Physically large subjects inferred from references, `head_h >= 7u` — UNCHANGED!):** Full 4-part recessed periorbital bezel (`6.3`) + `uv_scale: 2` (`64x` max) 7-layer focal eyes on compact species-accurate cuboids (`eye_h/head_h = 0.12–0.21`) + rich sculpted cuboid massing.
   - **`small` Entities, Wildlife, Critters & Props (Physically small subjects inferred from references, `head_h < 6u`):** **Balance honest Blockbench cuboid craft between two traps — NEVER over-carve 16-box noisy spheres, AND NEVER leave a living creature's rounded/fluffy head, puffball body, or pelagic bell as a raw flat-roofed `90°` cardboard shipping crate (`top10 > 95%`)!**
    - *Small props, nectar sacs & rolled orbs* (honey cubes, mud/magma balls, carried payloads): `1` bold proud cube (or `2–3` clean stacked cuboids) with painterly corner-darkening & specular highlights reads unmistakably Minecraft-authentic.
    - *Rounded/fluffy creature heads, puffballs & domes (`top10 = 78–92%` sweet spot):* Anchor on `1` bold core (`60–75%` volume), then wrap with **`2–4` purposeful bevel/taper planes** — an inset **domed crown cap** (breaking flat `90°` cube roofs!), softly tapered **cheek/snout/rump pads**, and (for fluffy/shaggy/feathered creatures) crisp **angled `0.5u–1.0u` fur-tuft / bang / ruff overhang planes** breaking raw square corners; for **radial jellyfish bells & round domes**, pair a `0°` core with a **`45°` Y-rotated octagonal sibling + inset crown dome** (`3–4` bold boxes total)!
   - *Small-entity & smooth-headed creature eyes (`v39.2` Anti-Extruded-Eye Law — CRITICAL!):* **NEVER extrude a thick separate eye cuboid proud of the head surface on small animals, birds, fish, mammals, lizards, or smooth-headed small creatures!** Extruded eye bricks sticking out `+0.15u..+0.80u` off the cheek or face look weird, boxy, and detached from `0°` Front and `3/4` perspective. Instead:
     - **Preferred #1 (`0.00u` Extrusion — Painted Directly ON the Head Surface!):** Paint the eye directly onto the `head_core` / `cranium` / `cheek` surface texture (`0` separate extruded eye boxes; crisp dark pupil + iris/sclera accent + `1 px` specular catchlight; optionally apply `uv_scale: 2` to the cranium/head box itself if higher eye detail is needed).
     - **Preferred #2 (Flush `0.0u` Hard-Alpha Eye Plane):** If a separate eye element is needed for blinking or `uv_scale: 2`, use a zero-thickness `0.0u` hard-alpha plane seated flush (`<= +0.02u`) on the head surface with transparent outer margin — NEVER a chunky protruding 3D brick!
     - **Only Biological Exception:** Taxa whose verified official reference genuinely has external bulging domes, conical turrets, eyestalks, or hemisphere compound eyes (frogs/toads, chameleons, crabs/shrimp, wasps/dragonflies/spiders) — sculpted organically to match the reference, never as flat glued cheek dominoes!
   - **Master Orbital Sweet Spot (Automated in `G14`):** Across ALL non-bulging tiers, **ZERO extruded solid eye cuboids (`eye_vs_cranium <= +0.02u`, local cheek/face extrusion `<= +0.04u`, `eye_above_crown <= +0.10u`)** — either paint directly on the head surface (`small` / smooth-headed `standard`), use a flush `0.0u` plane, OR carve a true deeply recessed 4-part brow/cheek orbital socket (`hero` / `epic_boss`) with **ZERO viewing-face occlusion** and **`v39.1` Single-Primary-Viewing-Plane painting (`0` double-pupil corner wrap)**!
10. **Epic 3D volume, anti-toy tails/plates/accessories & gap-free vaulted wings (`G8–G10`, `G13`).** Creatures player-sized or larger get deliberate sculpted volume (`240–550` elements for epic bosses; `9.1`). Break the Lego/toy look on **tails, dorsal plates, armor trim, and attached accessories** at *both* modeling and texture scales: facet cross-sections, use staggered overlapping shingle plates (`>= 2` pitched plates per armored segment) + 2-to-3 step tapered blade/fin tips, seat accessories inside contrasting mounting collars/brackets, and paint crisp bevels + inter-plate shadow grooves (`8.9`, `references/high_res_focal_carving_and_visual_primitives.md`). Flyers get vaulted gap-free wings (`>= 1.08x` chord coverage, `G10`).
11. **Adaptive high-craft resolution (`16x` or `32x` + `2x` focal `uv_scale`), `45–50°` hue-shifted palettes & cross-cube pixel-relief carving.**
    - Choose `16 px/block` (+ `uv_scale: 2` on eyes) for vanilla-lite mobs/props OR **`32 px/block` (+ `uv_scale: 2` = `64x` on eyes/lenses)** for studio-master creatures, heroes, animals, mechs, and detailed vehicles (`9.1`, `references/high_res_focal_carving_and_visual_primitives.md` §2), always packed with `>= 2 px` UV padding (`10.1`).
    - Block macro value zones in unlit view, pass the `25%` zoom squint gate, and keep `32x` crisp via **organized `2 px` stepped clusters + `1 px` razor bevels/crevices** — NEVER airbrushed blur or dusty confetti!
    - **Cross-cube pixel-relief carving (`10.2b`):** carve clean cuboids optically with imbricated U-scallop/hexagonal scales, continuous cross-seam central grooves (`Kundzio`), offset plate drop-shadows + `1 px` leading bevels (`Wackyblocks`), corrugated striations, and 5-layer wing shading (`10.3b`).
    - Banded, hue-shifted light (`45–50°` hue swing: cool saturated violet/indigo shadows, warm lights; never grey/black) + localized focal glow (`<= 4%` area).
    - **Native-Resolution Post-Zoom Visual Primitives Pipeline (`v35`):** NEVER crush reference images into compressed thumbnails; always run `scripts/visual_primitives_prep.py` — macro `grid --dual-edges` to locate bounds, **THEN zoom into every region FIRST and run fine `0.10` grid lines + outline/edge trace AFTER zooming (`zoom --autocrop --with-grid --with-edges`)**, plus `trace` (dashed skeleton `- - -`, joint angles, concavity polarity, and proportion calipers) across multiple perspectives AND ad-free timelapse video frames!
12. **True iterative passes WITH FULL TEXTURE ON — never batch-scripted fake passes and NEVER grayscale/clay-only inspection (`v28` Law).** Minimum passes: 3 white-model (`W1-W3`), 2 detail (`D1-D2`), 2 texture (`T1-T2`), plus 1 rig-and-motion check (`A1`); for formal epic boss builds, run **distinct iterative passes** where **EVERY pass is rendered and visually inspected WITH FULL TEXTURES & HARD ALPHA ENABLED (`clay=False`)**! Clay/grayscale/pastel renders hide collapsed wing-membrane alpha cutouts, inverted `mirror_x` UV cutouts (`box.center[0] < 0` vs `name.endswith("_l")`), eye extrusion/occlusion, and washed-out face contrast. **Never generate multiple passes in a single automated batch loop:** Pass `N+1` may only be authored after opening Pass `N`'s **textured** renders alongside zoomed reference crops and logging `>= 3` concrete corrections.
13. **Punch back & delete/rebuild courage.** A model or region that fails a gate in `5.4` or triggers an automatic rubric cap in `13.2` returns to `W1`/`W2` with failing placeholder limbs, snouts, or wing strips **deleted and rebuilt fresh** from explicit centres (`6.5c`). Never bolt patch boxes onto broken coordinates.
14. **Tier-honest complexity & engine independence.** Budgets are ceilings, not filler targets — yet under-building an epic boss at `< 240` elements with bare stick-limbs, stilt posture, 3-peg toes, and narrow strip wings is equally a failure (`G8–G12`, `9.1`). Never allow experimental SyncCraft collision caps (`0.25` parallel / `0.65` burial) to degrade compound cuboid carving into surface stickers (`11.0`).
15. **Audit engine-agnostically, measure 3D ratios, then judge adversarially.** Run `python3 scripts/construct_check.py [--tier epic_boss] <model.{bbmodel|geo.json|blockdsl}>` (`G1–G12`) every pass (plus `audit_scene` when targeting SyncCraft), verify the **12-Point Proportional Ratio Matrix (`5.1c`)**, inspect full turnarounds + `no-wing` 4-views + zoomed region crops, and apply the **Adversarial Self-Judgment Score Caps (`13.2` — forbid `28/28` rubber-stamping)**.

### 1.5 Small-AI Artist Intent Router, Universal Domains (`D1–D5`) & Temperament Polarity (`Gentle/Approachable` vs. `Furious/Unapproachable`, `v28`)
Before placing Cube `#1`, every AI model or artist MUST classify the target on **Domain (`D1–D5`) × Mood (`M1–M4`)** AND explicitly lock **Approachability Polarity** (full matrices in `references/aesthetic_mood_spectrum_and_intent_router.md` §1–§5):
- **Asset Domain (`D1–D5`):** `D1` Blocks/Props/Items/Weapons (`12–65` el.) · `D2` Small Fauna/Pets/Chibis (`15–50` el.) · `D3` Standard Monsters/Wildlife/Mounts (`60–220` el.) · `D4` Legendary/Imaginary Bosses & Mythic Titans (`240–550` el.) · **`D5` Large Vehicles, Airships, Battle Mechs & Architectural Dioramas (`80–400` el., `references/youtube_master_distillation.md` Part 2B)**.
- **Legendary Creature Temperament Polarity (`Pole A: Gentle & Approachable` vs. `Pole B: Furious & Unapproachable`):**
  - **Pole A — Gentle, Benevolent & Approachable** (*Wackyblocks Ocean Cargo Fish*, sanctuary guardians, friendly mounts, `M1/M3`): smooth rounded-chamfer convex silhouettes (`0` hostile body spikes), **open/arched brow (`Rx <= 0°`, `-12°..0°`)**, warm visible almond/round eyes, peaceful `0°–2°` closed lip, soft plump plumes, and harmonious `3:1` warm palette transitions.
  - **Pole B — Furious, Wrathful & "Unapproachable"** *(mandatory for Elder Dragons & Monster Hunter/Cataclysm bosses: White Fatalis, Fatalis, Deviljho, Gore Magala, Zelno, `M4`)*:
    1. **Eyebrow Euler Sign Law (`#1` Anti-Polite Fix):** **`Rx > 0°` (`+18°..+24°`)** plunges the anterior `+Z` glabella end **SHARPLY DOWNWARD** (`-z·sin(+22°) = -0.37`) while sweeping `-Z` up into the horns, paired with **`Rz = -26°..-32°` (Left) / `+26°..+32°` (Right)** to carve a continuous **`\ /` razor scowl** meeting across the forehead! *(NEVER use `Rx < 0°` on a wrathful boss — it lifts the inner brow into sad puppy eyes!)*
    2. **Dead-Front (`0°`) Perspective & Balanced Recessed Orbital Shelf Law (`v28`):** Always sum cumulative `pelvis -> waist -> chest -> neck_1..N -> head` X-pitches so **net world head pitch is `+6°..+10°` downward predator glare** (NEVER `-10°..-15°` accidental chin-up foreshortening, which stacks the chin & nose across the eyes like horizontal Lego bricks!). Taper `cranium -> snout_rear -> snout_front -> prow` smoothly (`~0.72x` per step) + diagonal `maxilla_l/r` & `jaw_ramus_l/r` V-wedges, separate crescent horn tips cleanly (`>= 12–16u` open crown), and **seat `eye_l/r` on the balanced orbital shelf** (`x_front_outer < cranium_xmin` so `cranium_core` never swallows the eye in `Side`/`3-4`, YET `x_rear_outer >= brow_xmin + 0.35u` so the eye NEVER extrudes past the overhanging brow or cheekbone!).
    3. **100%-Closed Rest Mouth (ZERO Exposed Teeth at Rest) + Hazard Spikes + `10:1` Chiaroscuro:** Close the rest/bind pose jaw completely flush (`jaw rest_euler[0] ~ -2°..0°`) with the narrower lower mandible tucked cleanly inside the overhanging upper lip rails and all recessed `32x` tooth planes concealed inside the oral enclosure (ZERO exposed teeth or red palate peeking out when closed — only external anatomical sabers/boar tusks may protrude!), opening wide `38–44°` via `mandible_depressor` during roars, howls, and bites; arm shoulders/elbows/wrists/snout with outward-jutting spikes, and push storm-charcoal/obsidian crevices (`RGB 20–38`) against stark alabaster ridges (`RGB 245–254`) and arterial crimson fissures!

## 2. The look

### 2.1 What it is
- **Form**
  - Large rotated cuboids and converging 3D wedges whose silhouettes flow as one continuous muscular animal or crafted object from front, side, three-quarter and top.
  - Hexagonal or keeled torso/neck cross-sections rather than square 90-degree crates.
  - They overlap at joints (with gusseted wide-to-narrow transitions) so the body reads as one continuous form.
  - Silhouette first: each part reads as a dark shape from front, side, three-quarter and top.
- **Hierarchy, about 70 / 25 / 5 by non-wing volume:**
  - primary masses (`chest`, `waist`, `hips`, `cranium`, `thighs`);
  - secondary forms that change the outline (`maxilla rails`, `brow`, `cheek`, `jaw`, `trapezius gussets`, `thigh meat`, `shoulder`, `knee`, `horn`);
  - tertiary accents (`claws`, `teeth`, `fangs`, `row spikes`).
- **Pushed proportions:** heads, horns, claws and the signature feature are bigger than life, and everything tapers thick to thin at knees, wrists and ankles.
- **Thin things are planes with cut-out alpha:** membranes, tattered streamers, fins, frills, fur tufts, manes, leaves, cloth, glass.
- **Paint**
  - Unlit-verified, hand-painted pixel art at the vanilla pixel size, `16 px` per block (9.1).
  - 3-5 banded values per colour, with hue-shifted ramps (warm lights, cool shadows) and cross-seam 3D continuity.
  - A 1 px light top/leading edge and a 1 px dark contact seam.
  - A limited vibrant palette: dominant, supporting and one accent.
  - Glow is a small, focal material (`<= 4%` area).

### 2.2 What it is not
- **Shard armour:** dozens of small slabs at random angles covering the body. This is the most common failure (see 2.5).
- **Mecha-locker / golem posture:** a rectangular vertical chest box under a narrow "Pez-dispenser" chimney neck, dangling straight arms, pasted vertical Gundam hip-skirt slabs, or flat snowshoe sole planks.
- **Wedding-cake snouts & french-fry crowns:** `yaw = 0` stepped rectangular snout boxes or 6-12 parallel horns packed across the skull roof without negative space.
- **2D cardboard-plank wings:** coplanar wing spars reading as a straight 1D line in top view, or a solid rectangular membrane wall filling the armpit.
- **High voxelization / voxel volumes:** staircase spheres, domes, curves and slopes made from many tiny stepped cubes instead of a few large rotated cuboids.
- **Horns as stick chains or sheets;** spikes as cube staircases.
- **A long flat face:** the head as one long box with features painted on, or eyes dropped onto flat `yaw = 0` side walls.
- **Toy monsters:** few boxes, straight limbs, flat colours, everything axis-aligned on a creature that should be epic.
- **Dirty paint:** noise, dusty per-pixel value jitter, many high-contrast specks, black outlines, grey shadows, or patterns that reset at every segment seam.
- **Plastic / procedural paint:** uniform concentric pixel-step rings hugging every face border symmetrically, rigid horizontal stripe gradients, or brightness-only ramps with no hue shift.
- **Repulsive rendering of creepy creatures:** slime, veins, gore, eye clusters (see 9.5).

### 2.3 Distilled Aesthetic Vocabulary (Self-Contained Style Grammar)
Our formal **"Minecraft at first glance, vigorous carved monster/world on second look"** aesthetic is fully codified in prose, quantitative gates (`G1–G12`), and craft patterns below (no external anchor images are bundled in the skill pack — every project instead gathers and inspects its own subject-specific references via the **2-Pass Grid-First -> Regional Ultra-Zoom** protocol in `§5.1`, and `references/vigorous_carved_aesthetic_and_spatial_reasoning.md` documents the structural craft rules in full):

1. **Organic Wyvern Silhouette & Radial Scalloped Wings (Craft Archetype 01):** organic curved multi-joint wing arms and fingers with deeply scalloped radial-shaded membranes; layered overlapping scale cuboids hugging primary torso/neck/tail masses; slim wedge head with overhanging brow, focused eye, and tiny peeking fangs; chunky splayed toes with hooked pale claws; dramatic top-down lighting where pale highlights kiss only dorsal ridges, horn tips, and claws against deep cool shadows.
2. **Sculptural Head Mass & Crystalline Accents (Craft Archetype 02):** commanding sculptural head with heavy brow over a focused glowing eye, interlocking teeth along a closed jaw line, jagged alpha-cutout cheek/neck frills, clustered crystal prisms rooted in dark organic/rock sockets, and muscular curved limbs painted with crisp `2–3 px` stone/scale chips and vivid hue-shifted shading.
3. **Classic Blockbench Cuboid Clarity (Craft Archetype 03):** clean, unmistakable Blockbench cuboid clarity on a compact world footprint; continuous stepped ventral plate column from throat to tail tip; zig-zag segmented cuboid horns changing direction at each joint; thick wing arms and spars holding wide membranes with diagonal painterly brushstroke hatching, ragged torn trailing edges, and irregular perforations.
4. **Cozy Stylized Props & Interior Scenes (Craft Archetype 04):** vanilla-compatible `16 px/block` handcrafted props and interior composition — slatted wooden crates (corner posts + gapped slats) filled with chunky organically tilted/bundled produce, hanging clusters, labeled jars, warm bright emissive lantern cores contrasting cool stone ambient shadows and crisp directional light.
5. **Upright Boss Silhouette & Focal Accent Restraint (Craft Archetype 05):** dynamic upright digitigrade biped boss posture; thick armored leading-edge wing spars with free protruding finger tips extending past tattered gradient membranes; glowing crystal horn crown and shoulder shards as the sole saturated focal accent against deep obsidian/slate plates with crisp edge highlights.
6. **Runic Carving & 3-Tier Feathered Wings (Craft Archetype 06):** athletic grounded quadruped combining `4–5` stepped dorsal snout scutes, chunky `4`-segment stone horns **pixel-carved with Greek-key runic spirals**, rhythmic ventral chest plates, **3 echeloned rows of pixel-cutout alpha feather planes** (`coverts -> secondaries -> primaries`) + flank root-transition tufts, and chevron-carved legs.
7. **Extreme Compound Muscle-Packing (Craft Archetype 07):** cantilevered brute mass proving how extreme muscularity is sculpted from **pure, honest `16 px` cuboids packed `50–75%` deep into each other** (bull neck, trapezius, oblique flank, drumstick thigh), `20+` rotated cuboid osteoderm spikes sunk `65%` into a swollen chin block, continuous sawtooth dorsal osteoderms, and **painted cross-seam muscle crevices + `1–2 px` recessed energy fissures**.
8. **Radical Low-Slung Sumo Mass & Native `16x` Pixel Clusters (Craft Archetype 08):** front-loaded wide silhouette (`belly_clearance < 0.15x height`) built from `4–5` wide stepped belly slabs studded with protruding `1×2` cuboid dermal warts, wide flat skull sunk directly into massive deltoids, wide-planted `4`-finger palms, and unmistakable **vanilla-native `2–4 px` mossy cobblestone / lichen pixel clusters**.
9. **Coiled Carapace & Corrugated Wing-Arm Mantles (Craft Archetype 09):** low-slung coiled predatory stance (`belly_clearance ~ 0.30x shoulder`, wide `4`-digit clawed paws), dense echeloned dark carapace rows blushed with saturated tip bevels, and **colossal arched wing-arm mantles** crowned with shoulder/elbow horns and carved optically with **bold transverse corrugated ladder-ribs** across clean rectangular beams.

- **Distilled master artist video studies:** `references/youtube_master_distillation.md` logs **39 YouTube Blockbench master tutorials and epic boss timelapses** — including **6 full `.mp4` builds analyzed second-by-second**.
- **Cobblemon discipline:** borrow construction cleanliness (clean large forms, strict symmetry, separate jaw/eye elements ready for animation) while scaling detail density to the chosen tier (`D1–D5`).

### 2.4 What the owner's set shows
1. **Rotated, stacked cuboids that read as flowing curves.** You can count the boxes, but the outline reads as one flowing form.
2. **Silhouette edges drawn in pixels:** torn membranes, fringes, drips and tatters on cut-out planes. Hard box edges alone look unfinished.
3. **Banded cel shading** in 3-5 steps per colour, filled with short directional strokes of neighbouring values (10.2); shadows toward violet or blue, lights toward cyan, yellow or cream.
4. **Two body colours plus one hard-part colour plus at most one glow:** red and cream, blue and navy, tan and sand, with ivory horns.
5. **Segmented undersides are geometry:** wide overlapping plates in the underside colour.
6. **Round machine parts are octagons** (wheels, hubs, pots). Round *organic* parts are single boxes with painted light.
7. **Saturation is controlled:** the strongest colour and value sit at the focal point.

Per image:
- **Red and cream dragon:** thick spar chains, a dithered cream-to-tan membrane per bay with a ragged trailing edge, pale antler-like horns that change direction at each segment, cream toes.
- **Black and blue elder dragon:** dense detail that is still *organised*. A row of rounded scale boxes armours the wing's leading edge; cyan crystal horns are the only saturated accent; a pale chest panel.
- **Blue ice dragon:** a head about a third of the body mass, a heavy block jaw with a row of yellow teeth, one red eye set deep under the brow, crystal prisms on rock bases.
- **Pink chibi dragon:** about 25 boxes, a head nearly half the body. Proof that the style scales down without losing banded paint or a clear silhouette.
- **Tan club-tailed wyvern:** thick wing arms, sand membranes, a row of curved dorsal spikes, a studded mace tail, grey segmented ram horns.
- **Plated beast and quill beast:** built from overlapping slabs in *coherent rows* that all lean the same way. That is density, not scatter.
- **Scenes:** warm emissive lights against cool ambient; props that lean slightly; chamfered hard surfaces; weathering that follows gravity.

Round-3 references (owner; faces, fins and spikes):
- **Grey dragon with a red eye:**
  - The mouth is closed; the jaw sits flush under the snout as a pale lip band.
  - A stack of long brow plates sweeps back over the skull.
  - One small glowing red eye sits under the brow, and a glowing ember sits at the chest.
  - The hide is painted as stone-like 2-4 px chips.
- **Ice dragon:**
  - A narrow, deep head, with big teeth interlocking along the closed mouth line.
  - A red eye with a dark pupil under a heavy brow plate.
  - Spiky cut-out fins with light tips on the cheeks and neck, and crystal spikes as clustered prisms.
- **Forest dragon:**
  - Wood-grain hide with glowing rune lines.
  - Ear fins built as a thick frame around a glowing panel.
  - Its pixels match the vanilla leaves behind it.
- **Fruit crates and a small octopus-like critter:** simple boxes with soft painterly light. The critter's eyes read instantly as lit irises inside a dark band.

Round-4 reference (owner; small creatures): **a small spider-like critter.**
- About fifteen big boxes, with pixels as coarse as the world's.
- The eyes are two lit irises inside a dark band that wraps the corner of the face.
- The legs are chunky blocks, not sticks, and the abdomen carries stripes along its length.
- Light is baked into every face: brighter on top and front, darker below.
- It reads as a spider at a glance and feels friendly, not creepy.
- **Together: Minecraft at a glance, real on a second look.**
  - Pixels at the vanilla size, and textures that echo vanilla materials, make a model read as Minecraft.
  - Anatomy, layered forms, glow and believable light make it feel real.

Round-6 references (owner):
- **Feet (anchors 4 and 1 again).**
  - On the ice dragon, the foot is a chunky block that continues the leg, and the toes are thick blocks.
  - On the red and cream dragon, the toes are cream blocks side by side across the front of the foot, as tall as the foot's front edge.
  - Both are blocks. The owner prefers block toes, fingers and claws to sheet claws.
- **Props and scenes** (a desk globe, rifles, a blaster, a relic on a plinth, a workshop, a wheel loader, a skid steer, a pantry, a witch's room, a snow base, an aquarium mech, an overgrown train).
  - Big flat colour fields.
  - A 1 px lit top edge and a dark contact edge.
  - Panel lines and vents as rows of short dark rectangles; rivet dots; chamfer strips that catch the light.
  - Wood in the vanilla-plank language.
  - Glass and leaves as planes.
  - Small, very bright light sources against cool ambient light.
  - Weathering that runs downhill: rust under seams, snow on tops, ivy hanging.

Round-7 references (owner; wings, elder dragons and amethyst dragons):
- **Plum flyer (Blockbench):**
  - Three fingers from one wrist hub, each a straight thick bar with one knuckle bend, and cream hook claws on the tips.
  - Concave scalloped bays between the tips.
  - The membrane is a dark field with long, light 3 px streaks radiating from the hub.
  - A cream dorsal spike row.
- **Black-blue elder (Blockbench), the template for upright bosses:**
  - An upright digitigrade biped.
  - The wing's forearm is a chain of rounded knob blocks about twice the spar thickness, with the spar tips running free past the membrane.
  - Two layered bays, painted saturated at the arm and fading to pale lilac, with long torn streamers and blob holes.
  - A pale chest panel, crystal horns and crystal shoulder clusters.
- **Teal dragon (Blockbench):** the wings are folded and raised, the fingers lie parallel to a vertical forearm, the slack membrane bows outward and its edge is saw-toothed. The hide uses large camo strokes and glowing rune dashes.
- **Amethyst dragons (2022 card art, RJ Palmer):**
  - Long prisms fan back behind the skull.
  - Geode clusters erupt at the shoulders.
  - Faceted plate scales; a cream, ridged throat and belly; crystal plates on the tail.
- **2024 Monster Manual red dragon:** a crown of blades around the face, hooked horns and a dark leading edge.

Round-8 references (owner; heads and hands):
- **Blue crystal dragon (r8a):**
  - A long, narrow head with a heavy brow over a red eye.
  - Snout: a straight box with a squared nose bump on the tip.
  - Teeth: big cream teeth along the lip edge interlock from the nose back to under the eye.
  - Jaw: deep at the hinge and tapering to the chin, framed by cheek spikes behind it.
  - Hands: wide palms with jointed fingers and long hooked claws.
- **Plum flyer head (r8b):** a slim skull, and a straight snout narrower than the skull with a squared end and a side nostril. The thin jaw is part of the wedge silhouette.
  - Two long straight horns sweep back with pale tips.

### 2.5 Exemplars
- **Good: "Saffron Squallwing" test flight,** which the owner judged amazing.
  - 151 elements, 50 bones, about 2.5 boxes per bone.
  - The 10 largest boxes hold 55% of the volume; the median element is about 6 x 3.5 x 6 units on a 170-unit span.
  - Partitioned membranes, 4-segment block horns, a banded texture with chevrons on the dorsal parts only and calm flanks.
  - **Its one rule break:** the head was turned 8 degrees and the tail curved sideways in the rest pose, which broke mirror symmetry. That attitude belongs in a display clip.
- **Bad: SyncCraft v9 Zinogre** (repo commit `ba969f3`).
  - 262 elements, 6.4 boxes per bone; the 10 largest boxes hold only 48% of the volume.
  - The median element is about 2 x 3 x 5 units on a 94-unit body.
  - The white render is shard armour: the head is a pile of plates with no readable skull, snout, jaw or eye; the horns are thin bars; the tail is a stack of tilted slabs; mirror parity is broken.
- **Bad: Unchamfered Upright Golem Dragon.**
  - 33 boxes dumped directly on `head` with 12 parallel `~52°` crystal sticks erasing crown negative space, a `yaw = 0` wedding-cake stepped snout, a square locker chest under a chimney neck, pasted vertical Gundam hip-skirts, flat snowshoe sole planks, and `0`-camber cardboard plank wings. Caught automatically by `scripts/construct_check.py` (`G1-G6`).

### 2.6 Anchor measurements (four owner images, viewed directly)
- **Wings dominate the picture.**
  - Span: about 2.3x nose-to-tail.
  - Upper arm: rises steeply from a shoulder mass to an elbow at or above head height.
  - Forearm: runs out almost level from the elbow and forms the top edge.
  - Fingers: three thick fingers fan down and out from the wrist, 25-35 degrees apart. The longest is about 1.2x the forearm.
  - Spars: about as thick as the lower leg.
  - Bays: deep, with the inner bay reaching the flank between shoulder and hip.
  - Membrane: underside colour as diagonal strokes that lighten toward a combed trailing edge, with a few blob holes near the tips. Elder wings are double-layered and heavily torn.
- **Heads are deep.**
  - The cranium is at least as tall as it is wide, and length to height is about 1.2-1.6.
  - The muzzle is a separate, narrower box that steps forward and down.
  - The jaw is heavy, at least 40% of head height, with big readable teeth.
  - One eye sits deep under a heavy brow.
- **Horns have a type:** antlers that change direction at every segment and carry a tine; crystal prisms; short back horns.
- **Underside:** one continuous column of stacked plates from throat to tail tip, alternating slightly in width.
- **Tails:** about one body length or more, many segments, a curl at the tip, the underside colour to the end, a small dorsal fringe.
- **Armour:** rows of long, narrow plates that all lean back, plus a row of rounded boxes armouring the wing's leading edge. Claws are separate boxes with coloured tips.
- **Cheeks:** the cheeks give the face its edge without making it wide. Anchor 1 has a cream frill of three stacked prongs fanning back from behind the jaw hinge; anchor 4 frames the cheeks with swept horns and crystal shards over a heavy hinge block.
- **Paint (anchors 1 and 4, zoomed):**
  - Short 1 px dashes of neighbouring values run along each form: lengthwise on legs, spars and horns, horizontal on the torso.
  - Soft steps: on the anchors' interior patches, the mean luminance step between neighbouring pixels is 5-9 (on 0-255), and only 4-7% of neighbours jump more than 25.
  - Legs carry darker ring bands. Belly plates have a seam row under each plate.
- **Good exemplar from these anchors: the "Jadecrown Wyvern" v3 test flight.**
  - 137 elements; the 10 largest boxes hold 63% of the volume without the wings.
  - Span 2.37x nose-to-tail, at 16 px per block.
  - Cranium 10 wide x 11 tall; from the front the head is 0.82x as wide as it is tall.
  - The jaw is about 40% of head height, closed at rest with a 2-unit overbite.
  - Cheeks as in 6.2b: the cheekbones stand out 1.25 units, and a framed fin sweeps back behind them.
  - Antlers: 3 segments plus a tine.
  - Wings: 7 panels per side on an "M" arm.
  - Walk and run stay in place.

## 3. Construction grammar

### 3.0 Master spatial and visual reasoning habits (before placing Cube #1)
Top Blockbench artists do not guess coordinates box by box; they think in world scale, anatomical masses, pivots and clean silhouettes before touching detail (see `references/youtube_master_distillation.md`):
- **Explicit world-scale block ruler:** Before placing the first mass, anchor the model against a 1-block (`16 px`) world ruler or floor checkerboard grid (e.g., an `8x8` block ruler for an 8-block dragon boss expanding to ~12 blocks across with wings and tail, or a 1-block crate/player height reference for props and critters). Monumental boss presence comes from silhouette scale and framing, never cube bloat.
- **2D box-overlay decomposition + real skeletal anatomy:** Before placing the first cube, mentally (or literally) overlay rectangular bounding volumes onto a side-view and front-view reference (cranium, muzzle, neck chain, ribcage, waist, pelvis, tail segments) and trace the **real underlying skeleton** (e.g., digitigrade hind leg = pelvis -> forward-pitched femur -> backward-pitched tibia -> forward-pitched hock/metatarsus -> grounded foot; foreleg = scapula -> humerus -> forearm -> wrist/paw). Bone pivots land on true biological hinges from the start.
- **Pivot before you rotate or pose:** Mastering pivot placement separates good models from great ones. Every independently rotated or animated part sits in its own named bone whose pivot is placed at the true anatomical hinge (jaw condyle, shoulder socket, elbow, wing wrist, horn root, foliage stem base, prop ground contact) *before* rotating or posing children. Never rotate a part around a stale offset origin.
- **Uniform graybox or flat material-color placeholder blocking:** Do not model hundreds of boxes in noisy default rainbow UV checkers. Work in a clean uniform neutral graybox during W1 (so silhouette and mass flow are unmistakable) and shift to flat placeholder material colours in W2–D1 (dominant hide, underside, hard parts, focal accent) to verify 60/30/10 colour balance and face orientation before pixel painting.
- **Tier-calibrated sculptural density (anti-voxelization AND anti-stick-simplicity):** On small critters (`15-30` elems) or sleek streamlined leviathan hulls (`60-170` elems), dissolve redundant segments when 2 bold cuboids read cleaner than 3. Conversely, on **Complex / Epic Boss Tier (`240-550` elems, `80-160` bones — e.g., winged Elder Dragons, multi-phase bosses)**, *never* leave limbs, necks, or wing-arms as single bare rectangular sticks (`3.1b`, `G8`). Consolidate hidden internal clutter in `W5` (like Vijay's *Sphinx* pruning `1,471 -> 1,214`), while keeping visible silhouette musculature, stepped chamfers, and armor scute rows richly sculpted.

### 3.1 Build order inside any part
1. Host core mass (chamfered or pitched to the flow line, anchored to its true anatomical pivot).
2. Stepped cuboid chamfers ("Minecraft Circle Math", `3.1b`), transition gussets, and muscular taper wedges (`maxilla rails`, `trapezius`, `thigh meat`, `bicep/calf swell`, `brow`, `jaw`).
3. Coherent directional rows of shingled armor plates, host-ribs, vambraces/greaves, or graded spikes following anatomical flow lines.
4. Articulated extremities & micro-traits: multi-jointed splayed phalanges, hooked talons, dewclaws/thumbs, joint spurs, staggered fangs, compound-curved horn tips (`8.3b`, `8.9`).
5. Gap-free nested-fan or shingled planes for thin edges (wing membranes `>= 1.08x` fingertip gap, fringes, sails, collars; `8.4d`).
6. Unlit macro value blocking + hard-alpha bay cutouts, then directional pixel paint, 5-layer wing shading, and optical value carving (`10.1–10.3b`).

Each step must improve the silhouette, cross-sectional volume, or focal area. If it doesn't, undo it.

### 3.1b Overcoming stick-limb simplicity: Stepped Cuboid Chamfering & 3-Layer Boss Limb Grammar (`G8`)
A recurring failure on complex dragons and bosses is **stick-limb syndrome**: building a `20–34u` upper arm, forearm, shin, or wing-arm out of 1 bare rectangular stick (or 1 stick + 1 tacked-on spike). Top Blockbench boss artists (ArtsByKev `WC33YHWU9NU` & `0WTvd2KNRqs`, Vijay `YDYIFgTZKRI` & `cQOaep_2y6o`, Nyux xi `NI4IhlIIVbw`, Wackyblocks `rycGulcQ-Lo`) build every hero/boss limb, neck, and torso region in **three organised layers**:
1. **Stepped Cuboid Chamfering ("Minecraft Circle Math", ArtsByKev):** Instead of a raw brick, pair the host core with a stepped contour slab inset `1–2 u` on top/front/sides to round harsh `90°` box edges while respecting **gravity** (grounded undersides stay flat) and **flow** (curved backs, deltoids, brows, and thighs step smoothly).
2. **Anatomical Muscle-Swell Wedge:** Every major limb shaft carries a pitched/rolled proximal muscle mass (`deltoid/bicep`, `triceps`, `quad/hamstring meat`, `gastrocnemius calf`, `pectoral/latissimus bridge`) creating a bold `~2.0–2.4x` thick-to-thin taper toward the wrist or ankle.
3. **Coherent Species Surface Architecture:** Layer species-defining external armor over the muscle — overlapping `30–50%` shingled vambrace/greave plates, `45°` diamond scale rows (`NI4IhlIIVbw`), dorsal wing-arm leading-edge shields, patellar knee caps, olecranon elbow spurs, and flaring wrist/ankle cuffs. All plates in a row share one flow direction (organised sculptural density, *never* random shard scatter).

### 3.1c Vigorous Carved Mass-Packing (`40–85%` Sunken Relief), Coiled Weight (`G12`) & Anti-Blender Law (`G11`)
Watching master `.mp4` builds second-by-second (`Darto 3d Rathalos` `Z58DMwfZtx4` `01:08`, `MHCraft Deviljho` `IzCwIiYj-Xk` `03:26`, `TrongOBD Zelno` `02:15`, `Kundzio Hammerhead` `1qB6m5Uct0k` `01:28`) alongside Anchors `06–09` reveals why procedural AI models look "toy-like" while Blender-imitators look broken:
1. **Deep Sub-Surface Cuboid Embedding (`40–85%` Sunken Mass-Packing, NOT Surface Stickers):**
   - Never glue thin flat boxes onto the outside of a rectangular limb or torso (`place(anchor=±1)` Lego-stud habit). Instead, sink secondary muscle swells, thigh scute crests (`4 x 3 x 9` rotated `[-60°, 0°, -10°]` inside Rathalos's `7 x 11 x 24` thigh), stepped sumo-belly tiers (`Anchor 08`), and chin osteoderm teeth (`20+` rotated `1x2`/`2x3` blocks inside Deviljho's jaw, `Anchor 07`) **`40–85%` deep inside the host mass** so only crisp `1–2.5u` stepped ridges and muscular contours emerge!
   - When packing dense faceted carapace helmets, chest cores, or gauntlets (`Zelno`, `Anchor 09`), snap `30°`, `35.26°`, and `45°` bevel cuboids **flush edge-to-edge** using distinct unlit pastel part colours during white-model passes so zero raw `90°` inner corners poke through.
2. **Coiled Predatory Weight & Grounded Stance (`G12` — Cure Tall Stilt-Horse Posture):**
   - Heavy quadrupeds, wing-arm Elder Dragons (`Anchor 09`), drakes (`Hammerhead`, `Slung Dragon`), and amphibians (`Anchor 08`) do **not** stand on tall vertical stilt posts under a high floating box belly.
   - Build the arched spine/belly chain *first* low to the ground (`belly_clearance = Y_belly_min / Y_shoulder_top <= 0.35–0.45` on grounded prowlers/quadrupeds, `< 0.18` on sumo sprawl beasts; `G12`), and tuck thick, flexed 3-segment limbs *beside* the ribcage planting wide `4–5` digit splayed paws (`1.8–2.4x` wrist width).
3. **The Anti-Blender Law (`G11` — Blockbench Is Blockbench):**
   - **Zero tolerance for fake-Blender curve slicing:** Never slice snouts, horns, tails, domes, or muscles into `>= 6` thin `< 1.0u` wafer steps or `> 8`-band concentric cylinders/spheres to approximate smooth $C^2$ NURBS surfaces (`G11`).
   - Every element must remain a proud, readable `0.5u`/integer cuboid (`min_edge >= 1.5u` on hero/boss bodies except terminal tips and `0`-thickness planes). Stop adding cubes the instant **cross-cube `16 px` transverse pixel-relief carving (`10.2b`)** expresses the surface richer and cleaner!

### 3.1d Half-Angle Sloped Joint-Bridge & Instep Gusset Continuity Law (`v30` — Zero `90°` Limb-to-Foot L-Corners, Hock Shelves, or Torso V-Notches)
Owner-directed signature aesthetic principle (*White Fatalis `v30`*): **We do NOT build spheres or multi-slice curves at joints (`G11`), yet we NEVER leave raw `90°` architectural L-corners, exposed flat joint shelves, or open triangular V-gaps where two angled segments meet!** Whenever parent cuboid $\mathcal{B}_1$ and child cuboid $\mathcal{B}_2$ meet across a hinge bend (`limb ↔ paw/foot`, `shin ↔ ankle` hock, `humerus ↔ forearm` elbow, or `tail ↔ pelvis ↔ waist ↔ chest` ventral belly plates), bridge the transition with **1–3 compact `0.5u`-grid Sloped Bridge / Gusset Cuboids** (`min_edge >= 2.0–2.5u`) oriented along the **3D half-angle / vector bisector**:
1. **`90°` Limb-to-Foot / Wrist-to-Paw Instep Slope Triad (`instep_slope` + `heel_bridge` + `cuff_bevel`, `8.3c`):**
   - Where a vertical/angled lower leg (`ankle_core` or `arm_fore_core`) meets a flat `y = 0` grounded `foot_pad` / `paw_pad`, fill the stark `90°` front L-corner with a **`+35°..+40°` Sloped Instep Block** (`0.75–0.85x` pad width, sloping smoothly from the front of the lower leg down onto `foot_knuckle`/`paw_knuckle`), pair the rear with a **`-26°..-34°` Heel / Palmar Wrist Gusset** (`min_y >= +0.6u` so `0.00u` flat sole contact is `100%` preserved!), and step lateral width via a **`+14°..+16°` Tarsal/Carpal Cuff Bevel** (`5.5u shaft -> 6.0–6.5u cuff -> 6.0–7.0u pad`).
2. **3D Vector-Aligned Hock & Elbow Joint-Pocket Caps (`hock_cap_slope` + `hock_front_slope`):**
   - At zig-zag digitigrade hocks (`shin_core` sloping down-and-back/outward into vertical `ankle_core`), seat `ankle_core` deep (`seat = 0.25–0.30`) so its flat square top never pokes out as a horizontal shelf, and compute the **true 3D vector** $\Delta\mathbf{p} = \mathbf{c}_{\text{shin}} - \mathbf{c}_{\text{ankle}}$ in the child bone's frame to pitch *and* roll `hock_cap_slope` (`e.g. rotate: [+26°, 0°, -22°]`) flush from `ankle_core`/`ankle_tendon` into `shin_core`/`shin_calf` — eliminating both the rear shelf and dark crease from every rear-3/4 view!
3. **Half-Angle Ventral & Dorsal Torso Hinge Bridges (`waist_groin_bridge`, `waist_sternum_bridge`, `tail_groin_bridge`):**
   - Wherever consecutive spine/tail segments pitch (`pelvis [0°] -> waist [-19°] -> chest [-39°]` and `pelvis [0°] -> tail_1 [-24°]`), ventral scute plates otherwise separate into V-shaped sky notches. Place a **half-angle scute bridge** ($\theta_{\text{bridge}} = \frac{1}{2}(\theta_i + \theta_{i+1})$, e.g. `+10°` groin bridge & `-12°` sternum bridge on `waist`, `+12°` groin bridge on `tail_1`) to form one **100% unbroken, continuous armored belly flow** from sternum to tail tip!

### 3.2 Remap table: real shapes into box language
| Real shape | Build it as | Never |
|---|---|---|
| Hinged joint transition (`limb ↔ foot/paw`, `shin ↔ ankle` hock, `pelvis ↔ waist` belly) | **Half-angle sloped bridge cuboids (`3.1d`, `8.3c`):** `35–40°` instep slope + `-26.. -34°` heel gusset + width-stepping cuff bevel on paws/feet; 3D vector-aligned `hock_cap_slope`; half-angle ventral scute bridges | raw `90°` vertical-shaft-on-flat-brick L-corners, exposed flat hock shelves, open ventral V-gaps, OR fake-Blender multi-wafer spheres (`G11`) |
|---|---|---|
| Tapered reptile / dragon snout & jaw | **Multi-block progressive wedge (`6.5d`):** 4-step tapering snout core + rhythmic dorsal nasal scutes + **2-step partial upper-jaw lip bosses** (`maxilla_l -> maxilla_ant_l`, `10-16°` yaw) + **solid 4-step watertight lower mandible** (`jaw_rear -> jaw_mid -> jaw_chin -> jaw_chin_tip`) | two oversized single V-shape diagonal planks (`jaw_ramus` V-bars or single `9u` brow/maxilla planks) or untapered `yaw = 0` wedding-cake slabs |
| Sphere, egg, round belly, dome, skull cap | 1 core box, optionally plus 1-2 angled chamfer/keel boxes on edges that show in silhouette; painted light does the roundness | stacked rings, concentric rotated boxes, `prism(n)` with n > 8, slices |
| Wide-chest to narrow-neck join | mirrored **trapezius / clavicle gusset wedges** (`roll 25-40°`, `yaw 12-20°`) + shingled collar/sternum plates (8.1-8.2) | a narrow vertical neck post planted straight into a flat chest top ("Pez chimney") |
| Muscular thigh or upper arm | pitched shaft box + rear/dorsal **proximal `meat` muscle-taper wedge** pitched `-20 to -35°` and rolled `8-14°` inward so hip/shoulder is `~2x` knee/elbow width (8.3) | a vertical post with a flat rectangular slab glued outside the hip ("Gundam hip-skirt") |
| Long slope (snout ridge, back line, wing root) | 1 box rotated to the slope; 2 overlapping boxes if the angle changes | a staircase of offset boxes |
| Bevelled edge (snout top, jaw underside, chamfered plate) | a core box, a narrower deck (or keel) on it, and 1 thin plate (1 unit thick) per edge, rolled to the bevel angle (45-50 degrees). The plate's outer face lies on the bevel plane, and the core's and deck's corners lie on that plane, so no corner pokes through. Put the parts in their own bone if the form is pitched, so the plates need only a roll | leaving a bulky block when the owner asks for bevels; a plate laid over an unshortened corner |
| Smooth curve (neck, tail, horn arc) | 3-6 seated segments (`seat = 0.28`) with a steady angle step (`10-25°`) and steady taper; each segment at least 1.5x as long as thick | 8+ thin slices; segments shorter than their thickness |
| Spike, thorn, fang | 1 slim box (`height >= 2.5x width`, centreline spikes rolled `45°`), graded dramatically along the spine (`1.8-2.2x` larger at withers/shoulders) with progressive lean; 2 boxes only if longer than 4x its base and bends | 3+ shrinking cubes; flat planes; identical fence-post spikes at one size/angle |
| Claw, talon, nail | 1 thick block, 0.7-0.8x as wide as its toe or finger and longer than it is wide, pitched down 15-20 degrees; painted dark where it leaves the digit and pale at the point | a hooked sheet; a small pale cube on the end of a thin toe ("marshmallow toes") |
| Foot, paw, hand | 1 heavy arched block (`height >= 0.55-0.65x length`, `>= 0.6x` lower-leg thickness) continuing the limb angle into splayed digits (8.3) | a flat horizontal snowshoe plank under a vertical post |
| Toe, finger, thumb | 1 thick block per digit, about a third of the foot's width each, small gaps between them, splayed `5-15°`, protruding at least half a digit length | thin sticks; a sheet |
| Horn or crystal crown | `1` outward-splayed primary pair (`2-4` thick segments on child bones) + `<= 1` divergent secondary pair (`>= 20°` angle separation) preserving open sky (7.1) | a plane; thin sticks; a cube staircase; `6-12` parallel "french-fry" prisms crowding the skull roof |
| Many thin spikes (quills, fur spikes, crystal fringe) | cut-out planes (`edge: spikes`), 2-3 staggered layers | dozens of thin splinter boxes |
| Wing sail / membrane | **3D vaulted spar chain** (`humerus -> forearm -> 2-bone curved fingers`, camber `>= 5%`) + per-bay cut-out alpha sub-panels + concave catenary inner armpit bay (8.4) | flat coplanar 2D plank wings or a rectangular wall blocking the armpit |
| Membrane, fin, frill, ear fin | plane(s) with alpha cut-out, one panel per bone, pivot at attachment root/spar, hard-edged 100% alpha carving | thick slabs; one plane spanning moving bones; blurry semi-transparent alpha borders |
| Fur, mane | 2-3 layers of cut-out planes or tilted slabs in rows | a fuzzy block; scattered cubes |
| Armour plates, scales | rows of slabs with the same lean, overlapping 30-50%; pair each plate's 1 px lit edge with a 1 px cast shadow on the surface beneath; small scales are painted | random shards |
| Sub-grid curvature, pincer/claw separation, subtle surface creases | keep the cuboid clean and **carve the contour through value** (dark perimeter cutaways, darkened outer limb corners, dark-plus-highlight dividing lines, 10.2) | subdividing into tiny bevel cubes ("high voxelization") |
| Recessed cavity (open mouth, slatted crate, shop/pantry shelf, window interior) | real recessed box depth framed by outer walls/slats/lips with interior props, tongue or dark backing inside for true 3D parallax | painting a flat window, mouth or crate contents onto a solid block |
| Tentacle, trunk, whip tail | 4-7 seated segments, each 0.8-0.85 of the previous, gentle steady curl | many short slices; uniform pipes |
| Wheel, hub, cylinder, barrel, pot (hard surface) | a true octagon: four bands, each w x 0.414w, at 0, 45, 90 and 135 degrees, sharing one `prism_group` | SyncCraft's `octagon`, which is two equal squares, one turned 45 degrees, and reads as an 8-pointed star from above; `prism(12)` on organic parts |
| Glass, window pane | a clear cut-out plane with one short glint on its outside face; the frame carries the rim | an opaque pane; glints on both faces, which cross into an X |
| Wire, bail, rail, thin handle (props) | a 1 px block: at 16 px per block a wire really is 1 px thick | a plane |
| Grip, sleeve or collar on a shaft | the segments stacked end to end (pommel, grip, shaft, band, socket) | a stick running through sleeves; the audit rejects the deep overlap |
| Eye | painted stamp on a **recessed angled eye-socket facet** tucked `0.5-1.0u` under the overhanging brow (`18-25°` forward yaw); shallow box only for bulging amphibian/insect eyes (6.3) | a cube stuck on an outer wall or painted flat on a vertical `yaw = 0` box side |

### 3.3 Rotation discipline
- **Allowed sources:**
  - (a) the pose, through bone rest angles;
  - (b) **3D constructive wedge/chamfer geometry** (`pitch`, `yaw` and `roll` working together to taper snouts, cheeks, trapezius gussets and thighs — never `pitch`-only modelling that leaves top and front views rectangular);
  - (c) a row's shared lean: plates or spikes lean back along the curve (`15-35°`), stepping smoothly `2-5°` per element in a steady progression;
  - (d) **clustered props, produce bundles and foliage only** (Anchor 04 Pantry, 8.7): deliberate small organic tilts (`5-15°`) and asymmetric splay/staggering **pivoted at the ground-contact or stem-attachment point** so leaning bottles, stacked crates, fanned carrots/leeks, hanging chili clusters and cascading vines feel handcrafted and lived-in rather than warehouse-grid copies.
- **Not allowed:**
  - per-box random angles on creature anatomy or structural architecture;
  - `pitch`-only modelling on paired face/torso/limb wedges;
  - "hand-made wobble" on structural parts;
  - rotation added to raise a style score.
- Comment every non-zero `rotate` with its reason (`// wedge yaw`, `// gusset roll`, `// slope`, `// row lean`, `// pose`, `// prop settle`).
- Prefer clean artist angles: 5, 7.5, 10, 12, 15, 22.5, 30, 45 (in Blockbench, default snap `15°`, `Shift` `45°`, `Ctrl` `1°`).

### 3.4 Size floors and texel-clean sizing
- **Whole-unit / clean-grid box dimensions (never messy decimals):** Arbitrary fractional cube sizes (e.g. `4.13 x 6.27`) distort texel density and cause uneven, squished pixels across neighbouring faces. Keep box dimensions on clean whole-unit or `size_step` (`0.5`) increments so pixels stay square at 16 px per block.
- **Fix coplanar Z-fighting without breaking UVs:** Never resize a cube to an ugly decimal to stop flicker. Separate coplanar overlaps with `layer: k` offsets on planes, clean `0.5`-unit stepping, or micro-inflate/offset (`±0.05` / `-0.001` in Blockbench) so UVs stay crisp and aligned.
- **On creatures larger than the player,** no box is thinner than 1.5 units in two dimensions. Exceptions: tips (claw, tooth, fang, horn tip, spike tip) and eyes. Finer detail goes into texture or a cut-out plane.
- **Minimum add-on size:** a plate or add-on is at least one eighth the size of the part it sits on. Anything smaller is painted.
- **Thin or block** (owner, round 6): decide by the part's real thickness at 16 px per block.
  - Genuinely thin things are planes: membranes, glass, leaves, cloth, paper.
  - Everything with thickness is a block: digits, claws, horns, feet, handles. Wires and rails on props can be 1 px blocks.
  - The failure is the in-between: a plank foot, a stick toe, a sliver claw, or a sheet standing in for a digit.

### 3.5 Connection, corner-flush junctions and joint-overlap skin
- Every element touches or overlaps a neighbour; nothing floats.
- **Corner-flush junction mindset:** When two angled masses meet (sloped chest into belly, neck into skull, 45-degree limb bends, angled pauldron over shoulder), align their meeting outer edges/vertices flush corner-to-corner (Blockbench Vertex Snap habit) and seat the inner face cleanly inside the neighbour, instead of eyeballing loose offsets that leave stair-step gaps.
- Seat chain segments (neck, spine, tail, horn) 25-30% into the parent: `attach(..., seat=0.28)`.
- **Joint-overlap skin for extreme boss bends (Longridr):** Extend the hidden rear of multi-segment neck and tail cores slightly inside their parent sockets (without creating coplanar Z-fighting) and stress-test extreme upward roar or curl bends early so gaps never open under deep articulation.
- **Overlap where forms flow together,** but each box must still show a real part of itself. SyncCraft rejects a box that is more than about 65% buried.
- **Wireframe / internal-overlap check:** Inspect seated torso, neck, shoulder and hip junctions in wireframe/X-ray (or cross-section renders) so hidden internal intersections are intentional, clean and never burst through the opposite wall.
- **Cross-bone seams use `blend_with`.** For a seam into a mirrored subtree, list both names, because `mirror_x` does not rename `blend_with` targets (`leg_r__thigh`).
- **Joint test:** rotate each joint 30 degrees both ways. Overlap must cover the gap.

### 3.6 Engine-agnostic construction, proportion & style-grammar check (`scripts/construct_check.py`, run every pass)
Run `python3 scripts/construct_check.py [--tier auto|small|standard|hero|epic_boss] [<SyncCraft_repo>] <model.{bbmodel|geo.json|blockdsl}>` on every pass (**always pass `--tier epic_boss` when building Elder Dragons or complex boss monsters!**). It natively parses **Blockbench `.bbmodel`, Minecraft Bedrock `.geo.json`, AND SyncCraft `.blockdsl`** with zero third-party dependencies, prints **3D Proportional, Coiled-Stance & Wing-Chord Telemetry** (`span`, `wingspan/length`, `head L/H & W/H`, `chest:waist V-taper`, `belly_clearance_ratio`, `max_adjacent_tip_gap` vs `max_plane_chord`), and enforces **twelve deterministic gates (`G1-G12`)**:
- **G1 Tier-Aware Hierarchy, Density & Mirror Parity:** top-10 non-wing boxes hold `>= 55%` (`small`), `>= 48%` (`standard`), `>= 36%` (`hero`), or `>= 26%` (`epic_boss`) of non-wing box volume (liberating `240-550` element sculpted bosses while blocking `600`-shard soup); non-head average `<= 3.5-4.0` boxes/bone; `<= 8-10` boxes on any body bone and `<= 14` on `head`; left/right mirror parity `OK`.
- **G2 V-Rail Snout Gate:** flags multiple centred `yaw = 0` snout boxes stacked without yawed maxilla side rails (`12-22.5°` yaw) that create "wedding-cake" rectangular top views.
- **G3 Crown Negative-Space Gate:** flags `> 8` horn/crest/crystal boxes crowded on `head` within a narrow pitch cone ("french-fry / organ-pipe" congestion).
- **G4 3D Wing Camber & Knuckle Gate:** verifies single-wing SVD out-of-plane camber ratio `>= 0.05` (`>= 0.08` target for spread sails) and sibling finger root spacing `>= 1.5u` along a palm block (while cleanly supporting sequential shared-wrist nested fan chains!).
- **G5 Spine Chamfer & Gusset Gate:** flags `4+` near-cubic `rot = 0` torso/neck cores lacking trapezius/clavicle gussets or lateral rib chamfers ("crate-train / chimney spine").
- **G6 Limb Taper & Paw Gate:** flags nearly upright outer hip-skirt slabs (`roll < 10°, pitch < 8°`) and thin snowshoe foot planks (`height / length < 0.38`).
- **G7 Texel-Clean Sizing & Bone Naming Hygiene:** flags non-`0.5u` off-grid box sizes that distort `16 px` texels and uppercase/spaced bone names that break Bedrock/GeckoLib/Block-DSL hygiene.
- **G8 Epic Complexity Floor & Anti-Stick-Limb Shaft Gate (`hero` & `epic_boss`):** fails `epic_boss` models under `< 240` sculpted elements and flags any major limb or wing-arm shaft (`humerus`, `forearm`, `thigh`, `shin`, `ankle`, `wing_upper`, `wing_fore`) built from a bare single stick (or a `>= 18u` boss shaft with `< 3` sculpted core/chamfer/armor boxes; `3.1b`).
- **G9 Articulated Extremity, Digit Splay & Multi-Phalanx Gate (`hero` & `epic_boss`):** fails `epic_boss` paws/hands/feet with `< 4` root digits (must include 3 splayed forward digits + opposable thumb or rear dewclaw/hallux), `< 20°` total digit fan splay, or unjointed 1-bone stub toes lacking `proximal -> distal/claw` phalanx chains and middle-digit length hierarchy (`8.3b`).
- **G10 Continuous Wing Membrane Chord Coverage Gate:** computes exact 3D fingertip distances between adjacent splayed wing fingers and fails whenever `max_plane_chord < 0.88x max_adjacent_tip_gap` — deterministically catching narrow strip membranes that open ugly V-shaped sky gaps between fingers (`8.4d`).
- **G11 Anti-Blender Micro-Slice & Fake-Sphere Abuse Gate (`3.1c`):** fails or warns when any bone stacks `>= 6` ultra-thin non-plane wafer slices (`min_edge < 1.0u`) or `> 8` concentric prism bands trying to fake smooth Blender curvature/spheres instead of using proud cuboids + pixel-relief carving.
- **G12 Coiled Predatory Weight & Anti-Stilt Toy Posture Gate (`3.1c`):** measures grounded multi-legged `belly_clearance_ratio = Y_belly_min / Y_torso_top` and warns when a quadruped/elder stands on tall skinny toy-horse stilts (`clearance_ratio > 0.58`) instead of hunkering with coiled mass and thick tucked limbs.

Passing `G1-G12` is mandatory before presenting any pass; failing any `[FAIL]` gate triggers immediate geometric reconstruction before moving forward.

## 4. Symmetry and the rest pose
- **Build only the left side (-X) and the centreline.** Generate the right side with `mirror_x(name)` or `mirror(axis=x)` on bones and macros, and `mirror: true` on stamps. Never hand-place a right-side part.
- **Centreline parts:** centre at x = 0 exactly, symmetric sizes, no yaw or roll.
- **The rest (bind) pose is symmetric about the X = 0 plane.**
  - Spine, neck, head and tail use pitch only.
  - Left and right limbs mirror each other.
  - Gesture comes from pitch: an S-curved neck, a lifted tail, a lowered head.
- **Attitude goes in a display pose or animation clip,** never the rest pose. That means a turned head, a tail swing, a raised paw or one wing higher.
- **Real asymmetric anatomy** (a broken horn, a one-sided molt) is declared in the trait sheet. Build it as a clearly named overlay *after* symmetry is verified.
- **Verify three ways:**
  1. the construction check (mirror parity);
  2. `analyze_morphology(program).symmetry_pairs`;
  3. front and top renders beside a mirrored copy.
- **Motion must stay symmetric too.**
  - On mirrored bones, roll (z) and yaw (y) wave tracks need `phase_deg` +180, and the opposite `bias_deg`. Without that, one wing rises while the other falls.
  - Pitch (x) tracks are symmetric as written.
  - Check a paired tip (left and right wing tip) across the clip's frames: the mirrored positions must match.
- **Outliner, naming and shared-UV hygiene (production culture):**
  - Every independently animated or angled sub-assembly gets a clear, unique **lowercase `snake_case`** bone name (no spaces, no capitals, no duplicates — e.g. `neck_1`, `jaw_lower`, `wing_l_forearm`) so exports parse cleanly across Bedrock, Java and Block-DSL.
  - Share mirrored UV occupancy on symmetric limbs, digits and repeated legs where asymmetrical wear isn't needed; paint once, stay consistent everywhere, and never leave dead orphan cubes or unused UV islands in the final asset.

## 5. Pipeline and pass gates

### 5.0 Concept first: a fresh body plan (before the reference board is final)
The style is fixed; the creature is not. Repeating one skeleton is a failure even when every gate passes.
- **Write the concept in one paragraph** in `PROJECT_GOAL.md`: what the creature *is* as an image or metaphor ("the dragon is the volcano"), where it lives, how it moves and fights, and its one signature.
- **Decide each design axis on purpose,** and write the choice and its reason:
  - stance: upright, sprawled, crouched, serpentine, many-legged, floating;
  - mass: front-heavy, back-heavy, low and wide, tall and narrow;
  - neck and head carriage: none, short and thick, long; head high, level or low;
  - limbs: count, type and how they meet the ground;
  - wings: role and rest pose (raised, folded, draped, wing-walk, vestigial, none);
  - tail: counterweight, weapon, rudder or signature;
  - signature placement: head, back, tail, wings or the whole body;
  - material language: one or two vanilla materials it echoes (10.2).
- **"What's new" check before W1.** List at least three choices that differ from the last two models in the project and from the exemplars. Proportion, pose, wing type, signature, material and detail all count; the skeleton itself may stay classic. If the side silhouette of the W1 blockout could be swapped for an earlier model's, redo W1.
- **The signature reads from the player's view.** Game creatures, and bosses above all, face the player. Check the signature at rest from the front and three-quarter front, not only from the side.
- **Boss-scale presence and monumentality (Longridr, Faentum, Kundzio):** Monumental bosses do not feel huge because you packed 500 tiny cubes onto them — they feel huge because of **world-scale proportion framing** (measured on an explicit block floor ruler, e.g. 8-block body / 12-block envelope), bold silhouette counterweights (deep keeled chest, dramatic wings, long serpentine tail or radiating caudal fin fan), clear negative space, and calm macro armor surfaces with strictly concentrated emissive focal zones (`<= 4%` area).
- **Use the body-plan vocabulary freely** (9.2): mix plans or invent one. A class never decides the skeleton by itself.
- **Classic skeletons are welcome (owner, round 7).** When the references show a normal dragon (four legs and wings, a wyvern, an upright elder), build that skeleton. Freshness means making each model distinct, not making its anatomy strange. Never invent odd anatomy (a crosswise hammer tail, "smith" arms) just to pass the what's-new check. Put the novelty in proportion, pose, wing type, signature placement, material and paint.

### 5.1 Reference board (before any box)
- **At least 10 images per model, filed by category:**
  - front, side, back, top;
  - three-quarter front and back;
  - head close-ups (side and front);
  - limb or foot close-up;
  - wing or tail close-up;
  - official render;
  - gameplay screenshot;
  - approved community Blockbench work.
- **Label missing views.** If images cannot be viewed in your environment, say so and ask for them. Never invent what a picture shows.
- **2D box-overlay and anatomical skeleton trace (before writing any coordinates).** On the cleanest side and front references, decompose the creature into stacked primary bounding cuboids (cranium, muzzle, neck chain, chest, waist, pelvis, tail segments) and map the real skeletal hinge chain (shoulder, elbow, wrist, hip, knee, hock/ankle, jaw condyle) so every bone's pivot and proportion ratio in head-lengths is solved before Cube #1.
- **Part map and 4-bucket trait triage (traits resolve and remap).** For each region (head, neck and torso, forelimbs or wings, hind limbs, tail, surface), write 2-5 lines in head-lengths: shape, proportion, angles, counts of horns, spikes and toes, and colours. Route every observed reference trait explicitly into one of four outcomes (plus Drop):
  1. **Silhouette cuboid mass** (keep or merge): forms that alter the 2D outline from side, front, top or 3/4 view (cranium depth, brow overhang, cheek flare, jaw hinge, shoulder/thigh bulk, antler segments, crystal prisms, slatted crate frames).
  2. **Exaggerated species marker** (`1.2-1.5x`): iconic focal traits that shrink or vanish at 16 px block scale (head mass, glowing eye socket, interlocking fangs, wing span, signature crest/horns, chunky produce).
  3. **Alpha-cutout plane**: genuinely zero-thickness edges (wing membranes, tattered trailing edges, cheek/neck fur spikes, ear frills, hanging cloth, cascading vines, glass panes).
  4. **Paint / value-carved surface**: sub-grid scales, ribs, rounded limb curvature, fine carapace seams, wood planks, rivets, weathering and minor secondary eyes — rendered optically through value contrast and manual anti-aliasing (10.2) instead of geometry.
  - **Drop:** micro-greebles, hyper-realistic muscle noise or tiny steps that cause high voxelization or fail the 25% squint test.
- **Mandatory Formal Visual Primitives & Native-Resolution Reference Pipeline (`v33`, Inspired by DeepSeek *"Thinking with Visual Primitives"* — `scripts/visual_primitives_prep.py` & `references/high_res_focal_carving_and_visual_primitives.md` §1):**
  - **Why (Eliminate BOTH the Perception Gap & the Reference Gap):** Crushing references into compressed `400–520 px` contact-sheet thumbnails destroys eyelid rims, overlapping tail shingles, joint bends, and accessory collars before the vision encoder ever sees them — while natural language alone (*"tapers smoothly"*, *"Z-fold leg"*) drifts spatially during reasoning.
  - **NEVER downsample reference crops below 1:1 native pixels, NEVER guess crop bounds blindly, and NEVER hardcode one species' skeleton as a universal default!** Always inspect **multiple perspectives** (`front`, `side`, `3/4`, `rear`, focal close-ups) AND **ad-free timelapse/motion video frames** (strictly filtering out commercials, intros, and sponsor segments) via this formal 3-step programmatic pipeline:
    1. **Pass 1 — Native 1:1 Coordinate Grid + Structural Edge-Map (`visual_primitives_prep.py grid <ref> <out> --autocrop --dual-edges`):** Inspect the full-resolution image paired side-by-side with its Scharr + chromatic + DoG structural ink edge-map so true silhouette breaks, eyelid rims, plate seams, joint inflections, and accessory brackets stand out unmistakably without lighting/texture clutter. Read exact `(x, y)` coordinates directly off the `0.10` grid.
    2. **Pass 2 — Visual Primitives Structural Trace & Curvature Concavity Proof (`visual_primitives_prep.py trace <ref> <spec.json> <out> --autocrop --export-crops-dir <crops>`):** Render explicit **dashed skeleton centerlines (`- - -`)**, numbered **joint waypoints (`<point>`)** with automatic **interior bend angles (`θ°`)** AND **automatic signed curvature concavity classification (`CONCAVE_UP (U)` vs. `CONCAVE_DOWN (arch)` vs. `S_INFLECTION (~)` via `"label_concavity": true`)** on EVERY curved structure (horns, antlers, tusks, beaks, tails, necks, spines, claws, blades, wings, canopies, arches), **dotted proportion calipers (`· · ·`)** with automatic ratios (`hind/fore_chain`, `hip_h/shldr_h`, `head_L/head_H`, `eye_W/head_L`), and **focal `<box>` callout rectangles** (`head_eye`, `limb_joints`, `tail_plates`, `accessories`). Never infer curvature concavity from endpoint direction alone (`references/high_res_focal_carving_and_visual_primitives.md §1B`)!
    3. **Pass 3 — Native 1:1 Focal Zoom + Edge-Map Pairs & Render Comparison:** Open EVERY auto-exported lossless `*_native_and_edges.png` close-up directly via vision to map exact eye-socket carving, tail shingle layering, stretched-out T-pose segment ratios, and accessory collars — then run `edges` / `trace` on our own white-model and textured renders to verify geometric parity!
- **Zoomed crops** of the head, eyes, horns, limbs and tail tip are what every regional check compares against.
- **Proactive Internet & 3D-Viewer Reference Expansion (`v37`):** Whenever existing references feel insufficient, shadowed, or leave any structural arrangement unclear (such as top-down dorsal/tail spike tracks, circumferential upper-limb spike rings, exact snout material boundaries, or interior jaw/tooth geometry), **do NOT guess** — continue searching on the internet (`search_web` + `browser` subagent on official 3D model turnarounds, interactive Sketchfab / 3D viewers, and artbook orthographics) until every angle is clearly verified at 1:1 native resolution!
- **Recent, high-quality art only (owner, round 7).** Use current game and book art, such as the 2024 Monster Manual dragons, the Fizban's gem dragons and current game concept art, plus top Blockbench work. Old book illustrations mislead. Images found through search but not opened or uploaded cannot be measured: say so, use their text descriptions only for traits, and ask the owner to upload the pieces that matter.
- **Mapping table (visual reasoning).** Before W1, write one row per reference in `<project>/references/MAPPING.md` with three columns:
  - what it shows (measured on zoomed crops);
  - the box-language construction (and triage bucket) it maps to;
  - which gate or clip it informs.

  Translate painterly detail (translucency, pebbled scale, glow haze) into paint, never into extra boxes. Translate every structure (knuckled spars, knob rows, crystal clusters, plate columns) into named forms. Mark traits you are not using and say why.

### 5.1b De-Homogenization, Bespoke Cranial Archetypes & Variant Divergence Law
A critical LLM/procedural failure mode is **template homogenization**: building a crocodilian leviathan, an aristocratic vampire Elder Dragon, and a knightly paladin variant out of the *exact same* generic box-snout head (`cranium + nasal_bridge + squared nose_boss + maxilla_rail`), uniform 3-box neck, and flat brick `foot_base`, or generating a subspecies/variant by running string `replace()` on box dimensions.
1. **Strict Ban on `replace()` Variant Reskins & Skeleton Cloning:** Never generate a structural variant (e.g., Base Malzeno vs. Primordial Malzeno, Rathalos vs. Dreadking, Plains vs. Ice Dragon `rxLyJWHKoYU`) by regex-scaling box numbers in the base `.blockdsl`! Every structurally divergent region (crown visor, chest/sternum architecture, wing-shield blades vs. tattered bat-cape, halberd lance vs. candelabrum tail) must be authored with **bespoke bone/box topology** reflecting its own silhouette mechanics.
2. **Select the Exact Cranial Architecture Archetype Before `W1` (Never Default to Generic Box-Snout!):**
   - *Archetype A — Aquiline / Hooked Avian-Draconic Visor* (Malzeno, Velkhana, Kushala, Rathalos `Z58DMwfZtx4`): steeply tapered narrow beak-rostrum ending in a **downward-hooked sharp beak tip** (**NO square `nose_boss` brick!**), swept supraorbital brow plates that flow **continuously without a break** into `4–6` compound-rotated S-curved crown horns (`brow -> horn_1 -> horn_2 -> horn_3 -> tip`).
   - *Archetype B — Broad Dorsoventrally-Compressed Crocodilian / Leviathan* (Lagiacrus, Royal Ludroth): low, wide armored skull table, bulging rear pterygoid jowl cushions behind the jaw hinge, elongated narrow V-snout with raised terminal nostril bosses.
   - *Archetype C — Blunt Crushing Brute / Theropod Block-Jaw* (Deviljho `Anchor 07`, Tigrex, Anjanath): tall vaulted maxilla, massive swollen mandible (`>= 50–55%` head height) bristling with **`16–24` rotated cuboid chin osteoderms sunk `60–75%` into the jaw mass**, plus compound-packed neck/trapezius musculature.
   - *Archetype D — Compact Lupine / Feline Fanged Wyvern* (Zinogre, Odogaron, Nargacuga): short carnivoran muzzle (`0.5–0.7x` cranium), pronounced zygomatic cheek fur/ear sweeps, carnassial fang notch.
   - *Archetype E — Plated Knightly Casque / Carapace Helm* (Primordial Malzeno, Diablos, Glavenus, Brachydios, Zelno `iJHhXy-YigI`, Mecha `rycGulcQ-Lo`): faceted `30–45°` edge-locked visor shields framing deep slit eyes, cheek-guard plates, armored chin gorget.
   - *Archetype F — Botanical / Arthropod / Multi-Headed Eldritch* (Plant Monster `NI4IhlIIVbw`, Scorpion `4w_N8r4j2vc`, Typhon `cQOaep_2y6o`): radial petal/hood brims, recessed oral cavity with articulated chelicerae/pincers, or multi-plane sculpted masks.
   - *Archetype G — Sumo-Amphibian / Collar-Sunk Flat Skull* (Tetranadon `Anchor 08`, Zamtrios, Dodogama): wide dorsoventrally-compressed frog/platypus skull sunk directly into hulking deltoids (zero neck stalk!), protruding amphibian eye blocks, and `4–5` wide stepped horizontal gular/belly slabs studded with `1x2` dermal warts (`belly_clearance < 0.18x height`).
   - *Archetype H — Coiled Wing-Arm Mantle & Echeloned Carapace Elder* (Gore/Chaotic Magala `Anchor 09`, Gaismagorm, Nergigante): low-hunkered predatory head/chest (`belly_clearance ~ 0.30x shoulder`), dense echeloned carapace plates, and colossal corrugated wing-arms arching from spiked shoulder/elbow crowns to plant heavy wing-fists beside `4`-digit beast paws.
3. **Mandatory Signature DNA & Anti-Template Diff Table (`MAPPING.md`):** Before writing Cube #1, document `5–8` irreducible species-specific traits (exact attachment point, cross-section shape, flow vector) AND explicitly state how this creature's **head topology, chest/collar architecture, extremity hands/feet, wing structure, and tail weapon** differ structurally from prior models.

### 5.1c Comprehensive 3D Spatial & 12-Point Proportional Ratio Matrix
Never judge proportions from a single bounding-box `L/H` number. Before `W1` — and audited numerically + visually at `W2`, `W3`, and `W5` — measure and log this **12-Point Spatial Ratio Matrix** in `Cranium-Units (1 CU = skull height without horns)` or `Head-Lengths (1 HL)` across all three orthogonal planes (`YZ` side, `XY` front, `XZ` top):
- **`R1` Rostral Taper & Hook:** `W_snout_tip / W_cheek` (e.g., `0.22–0.35` sharp aquiline beak vs `0.55` brute) & `H_snout_tip / H_cranium`.
- **`R2` Horn-to-Skull Sweep & Splay:** `L_horn_arc / L_head` (`1.2–2.4x` on Elder Dragons), `tip_span / W_cheek`, and side-view continuity angle with the brow/neck line.
- **`R3` Cervical S-Arc & Cross-Section Taper:** `L_neck / L_torso`, `W_neck_top / W_neck_base` (`0.65–0.80`), and `D_neck_keel / W_neck` (inverted teardrop or hexagonal keel, never a round/square pipe).
- **`R4` Thoracic-to-Waist V-Taper:** `W_chest / W_waist` (`>= 1.55–1.95` on athletic predators/dragons) & `D_chest_keel / D_pelvis`.
- **`R5` Forelimb Cascade & Muscular Taper:** `L_humerus : L_forearm : L_hand` & `thickness_shoulder : elbow : wrist` (`>= 1.9 : 1.3 : 1.0`).
- **`R6` Hindlimb Digitigrade Z-Cascade:** `L_thigh : L_shin : L_metatarsus : L_mid_toe`, knee-to-hock Z-offset, and `D_thigh / D_ankle >= 2.2`.
- **`R7` Wing Frame & Chord Coverage:** `L_humerus : L_forearm : L_f1 : L_f2 : L_f3 : L_f4`, `open_wingspan / body_length` (`2.0–2.8x`), and `plane_chord / fingertip_gap >= 1.08x` (`G10`).
- **`R8` Caudal Taper & Terminal Weapon Contrast:** `L_tail / L_body` (`1.0–1.5x`), `W_tail_base -> W_tail_stalk` (`~3.5:1` taper), expanding sharply into `W_tail_weapon` (`2.2–3.5x` stalk width for tridents, halberds, clubs, or fans).
- **`R9` Extremity Footprint & Digit Hierarchy:** `L_hand_total / L_forearm` (`0.45–0.65`), `L_mid_digit / L_outer_digit` (`1.20–1.35`), total digit splay (`24–45°`).
- **`R10` Negative-Space Triangles:** verify open sky windows under the arched belly (`Y_belly / Y_withers`), inside the S-neck throat curve, under the concave catenary wing armpit, between splayed crown horns, and between splayed talons.
- **`R11` Cross-Sectional Silhouette Variety:** verify non-rectangular cross-sections per region (keeled diamond chest, triangular visor snout, chamfered oval thigh, bladed wing leading edge).
- **`R12` Player-View Frontal Authority:** in `front` and `3/4 front` camera (`el = 8–15°`), verify that the head, eyes, chest core, and wing/horn frame command clear focal hierarchy without occluding each other.

### 5.1d Six-Stage Vision & 3D Spatial Reasoning Curriculum (`S1–S6`, for Agents & Small-Model Training)
To translate 2D concept references into **Vigorous Carved Blockbench** without drifting into toy sticks or fake-Blender curves (full mathematical specification in `references/vigorous_carved_aesthetic_and_spatial_reasoning.md`), execute these six spatial reasoning steps explicitly before and during construction:
1. **`S1` World-Scale Envelope, CoG Triangle & Stance Archetype:** Anchor on a `16 px = 1 block` ruler; classify stance (`coiled_prowler`, `cantilevered_brute`, `sumo_sprawl`, `regal_quadruped`, `upright_elder`, `serpentine_leviathan`); lock `belly_clearance_ratio <= 0.42` (`< 0.18` sumo) and wide paw support polygon (`>= 1.35x` chest width, `G12`).
2. **`S2` Honest Compound Cuboid Mass-Packing Graph (Anti-Blender Decomposition):** Decompose every muscular/armored region into `1` Host Core Cuboid + `2–6` Sunken Muscular/Bevel Cuboids (`burial_ratio = 0.40–0.82`, clean `0.5u` sizes, `min_edge >= 1.5u`) — **zero wafer-sliced fake curves (`G11`)**.
3. **`S3` Directional Echelon & Silhouette-Rhythm Vectors:** Parameterize dorsal osteoderms (`Anchor 07`), carapace rows (`Anchor 09`), chin studs, and `3`-tier feather planes (`Anchor 06`) as coherent progression curves (`scale_envelope`, `shared_lean_deg`, `30–50%` overlap).
4. **`S4` Ground-Grip Extremity & Hinge Topology:** Place pivots at biological hinges *before* rotating; branch hands/feet into `4–5` splayed multi-phalanx digits (`25–45°` spread, middle digit `1.25x` outer, + thumb/dewclaw; `G9`).
5. **`S5` 3-Way Relief Split Boundary (`Cube` vs. `Alpha Plane` vs. `Transverse Pixel Carving`):** Stop adding cubes the instant a feature changes silhouette `< 1.5u`: use **Cuboids** for thick 3D mass (`>= 1.5u`), **`100%` Hard-Alpha Planes** for `< 1.0u` membranes/feathers/tufts, and **`16 px` Transverse Pixel Carving (`10.2b`)** for scale rungs, ladder-ribs, runes, and muscle clefts!
6. **`S6` Multi-View Adversarial Render-to-Reference Loop:** Audit `front`, `side`, `top`, `3/4`, `no-wing 4-view`, and zoomed crops against `G1–G12` after every single pass.

### 5.2b Formal Multi-Pass Per-Stage Human-Artist Rhythm (`>= 3 W` -> `2–3+ T` -> `>= 2 A`) & Ban on Batch Fake-Passes
Just like a real human Blockbench artist, **NEVER rush a build in 1 white pass + 1 texture pass + 1 animation pass, and NEVER generate `W1..W5`, `T1..T5`, or `A1..A5` in one automated script loop!** Even on streamlined test flights, execute **at least `3` genuine White-Model iterations (`W1 -> W2 -> W3[-> W4]`), `2–3` genuine Texture iterations (`T1 -> T2 -> T3`), and `>= 2` genuine Animation iterations (`A1_draft -> A2_final_export`)** — stopping after every single iteration to visually inspect rendered turnarounds/macro zooms/filmstrips against reference crops and applying at least **3 logged flaw-to-fix interventions** before authoring the next iteration:

- **White Model Stage (`W1 -> W5`):**
  - **`W1` Skeletal Rig & Ratio Blockout (`~40–70` elems):** World-scale ruler, true joint pivots, primary spine/limb/wing-arm cores; verify `R1–R10` ratios (`<= 12%` error vs. ortho refs) and `no-wing` 4-view posture.
  - **`W2` Bespoke Cranial Sculpt & Secondary Musculature (`~110–180` elems):** **Delete & replace** `W1` placeholder head/limb sticks; build the chosen Cranial Archetype (`5.1b`), multi-segment compound-rotated horns, stepped cuboid chamfers, and proximal muscle swells (`3.1b`). Pass `Head Gate` (`6.6`) at rest + open roar.
  - **`W3` Articulated Extremities & Gap-Free Wing Frame (`~190–280` elems):** Sculpt 4–5 digit multi-phalanx hands & raptor feet + dewclaws (`8.3b`, `G9`), 4-finger 3D vaulted wing spars, and full-chord overlapping fan membrane planes (`>= 1.08x` fingertip gap, `G10`). Verify `0` static SAT collisions.
  - **`W4` Species Armor Architecture & Silhouette Micro-Traits (`~250–500` elems):** Layer coherent shingled scute/rib/gorget rows, vambraces/greaves, individual staggered fangs, elbow/knee/calcaneal spurs, multi-bone collars/capelets, and articulated tail weapons (`8.9`).
  - **`W5` Consolidation, 6-Pose `SE(3)` Extreme Stress Test & Adversarial Crop Diff:** Prune redundant/buried internal boxes (timelapse consolidation habit, `_h6xjye56Sk`), validate 6 extreme combat/flight poses via `SE(3)` for zero joint cracks, pass `G1–G10` (`--tier epic_boss`), and pass side-by-side Reference-Crop Overlay critique.
- **Texture Stage (`T1 -> T5`, `16 px/block`, `>= 1–2 px` UV padding):**
  - **`T1` Unlit Macro Value Architecture & Exact Bay-Polygon Hard-Alpha Cutouts:** Lock 60/30/10 colour/value hierarchy, pass `25%` squint gate, **AND carve exact spar-to-spar triangular bay polygons + concave trailing scallops on wing/frill planes (`10.3b`)** — visually verify ZERO inter-finger V-gaps or comb teeth on the 3D wing before `T2`!
  - **`T2` Volumetric Form Shading, Limb Rounding & Bone-Hugging Wing Gradients:** Bake top-down directional light, cylindrical limb corner darkening, deep joint AO, and **both-spar bone-hugging dark AO -> translucent mid-bay belly gradients** on wing membranes.
  - **`T3` Cross-Seam Material Craft, Roof-Tile Popping & Tension Venation:** Paint 3D cross-seam scales/plates/fur, `1 px` bright rim over `1 px` cast shadow on every overlapping scute/shield, horn fluting, and radial wrist-to-margin wing tension folds + forked veins.
  - **`T4` Focal Polish, Anti-Plastic Breakup & Hidden-Face Articulation Audit:** Hand-crisp pixel-grid eyes (erase muddy overlays off eyes/teeth per ArtsByKev `6MshY3JGrTw`), optical claw-hook carving (`87jJmKAgFqE`), manual `1 px` stair-corner AA, break symmetric concentric face borders, and rotate jaws/wings/limbs to paint all concealed inner faces.
  - **`T5` Restrained Emissive Life (`<= 4%`), Uplight & Raw Atlas / Reference Audit:** Add localized focal glow + warm/cool uplight onto neighbouring armor, verify interior patch smoothness (`lum_step 4–9`, `jumps > 25 <= 7%`), inspect the raw UV atlas at `100%` for bleed/orphan islands, and compare side-by-side against this model's zoomed reference crops and `G1–G12`.
- **Animation Stage (`A1 -> A5`, SyncCraft `SE(3)` + Adaptive `288`-Frame Swept 15-Axis OBB `SAT`):**
  - **`A1` Weight, Balance & Grounded Locomotion (`idle`, `walk`, `run`):** Breathing phase waves, heavy grounded footfalls, spine S-undulation, rotary/diagonal limb timing, and treadmill `root.x/z = 0` (`11.5b`). Visually check limb arcs & weight!
  - **`A2` Aerodynamic Flight & Wing Folding (`takeoff`, `fly_glide`, `hover_flap`, `dash/land`):** Fast forceful downstroke vs. slower recovery upstroke, `25–45°` root-to-tip finger lag, marine spine counter-pitch (`11.5c`), and clean folded-to-spread fan overlap.
  - **`A3` Signature Combat Mechanics (`anticipation -> 2-4f snap strike -> apex hold -> recovery`):** Choreograph species-signature weapon-wing, jaw, claw, and tail strikes engaging the whole spine/chest/counterweight tail rather than moving 1 limb on a frozen body.
  - **`A4` Ultimate Boss Phases, Roars & Secondary Follow-Through:** Layer secondary overlap on multi-bone collars/cloaks, ear-fins, grasping hand phalanges, jaw/tongue, and splaying tail tridents/fans.
  - **`A5` Full `288`-Frame Swept OBB `SAT` Clearance, Filmstrip Critique & Export:** Resolve every mid-swing collision physically via realistic arc clearance (never flattening poses!), verify `step_bound_satisfied` & `0` stance slip, visually inspect 4-to-6 frame filmstrips of every clip, and validate `.bbmodel` / Bedrock exports.

### 5.2 Passes
| Pass | Goal | Must produce |
|---|---|---|
| W1 Blockout | 12-25 primary & V-rail masses on anatomical pivots in uniform graybox against a world-scale ruler, spine S-curve, proportions in head-lengths; head gate (6.6) | unlit/graybox renders: front, side, top, three-quarter front and back, **`no-wing` body 4-view**, side and front black silhouettes |
| W2 Secondary | gussets, brows, cheeks, V-mandible, proximal `meat` limb tapers, shoulder and haunch, knees and elbows, horn chains, spaced wing spars, chunky arched paws; optional flat material placeholder colours to verify 60/30/10 mass balance | the same, plus zoomed region sheet: head 4-view (closed + open roar); torso-neck front/side; limb side; wing top/front; tail top |
| W3 Refine | fix every regional critique; recheck proportions with numbers; dissolve or merge anything that does not change the silhouette | the same + `scripts/construct_check.py` (`G1-G7`) clean pass |
| D1 Detail | graded dorsal rows, continuous ventral scute column, block claws/fangs, vaulted wing sub-panels, catenary inner armpit bay, fur/fin planes, on focal areas first | the same + wing fold/clearance check |
| D2 Detail | progressive row leans, 28% seating + joint-overlap skin, corner-flush junctions, 30-degree joint-gap test, wireframe check, `blend_with` seam audit and symmetry; remove anything noisy | the same + zero audit/gate warnings |
| T1 Macro Texture | unlit flat base colour blocking per material (whole-element first), underside column, membrane gradient, hot eyes; **25% zoom macro value-block gate** before micro-strokes; local silhouette check | unlit + shaded renders plus `1 px`-padded atlas |
| T2 Painterly Texture | hue-shifted 5-step ramps, cross-seam 3D directional strokes, calm `>= 60%` base, break plastic banding, rim-over-shadow depth & manual AA, sharpen focal contrast, `<= 4%` glow + uplight, **animation-safe hidden-face check** (10.1), interior patch luminance check, squint test | final turnarounds + atlas + patch luminance check |
| A1 Rig and motion | pivots at joints, relative `joint_limits`, 30-degree test at every joint, symmetric rest pose, idle + roar + **in-place** walk/run/flight (`11.5b`, `11.5c`), `288`-frame swept-motion audit | fixed-frame treadmill filmstrips + zero foot-skate verification |

The white model is a draft. It must survive W2 and W3 before any detail.

**Mandatory `no-wing` body render (W1-D2):** large wings hide broken shoulders, Pez-dispenser chimney necks, locker chests and Gundam hip-skirts. Every `W1-D2` pass on winged creatures must render a **`no-wing` 4-view (`side`, `front`, `3/4 front`, `3/4 back`)** and verify clean body proportions with the wings hidden.

### 5.3 Every pass
1. Re-read this skill and open this model's reference board + zoomed regional crops (`§5.1`).
2. Build or edit, **cycling views continuously** (Side -> Front -> Top -> 3/4 -> Wireframe/internal-overlap) after every major form group instead of waiting until the end of the pass.
3. Run SyncCraft's `audit_scene` AND `python3 scripts/construct_check.py` (`G1-G7`, 3.6).
4. Render the full view set (including `no-wing` 4-view on winged assets) and the zoomed region sheet.
5. For each region (`head`, `neck-chest`, `arms/wings`, `hips-legs-paws`, `tail`), write: `reference shows ... -> model shows ... -> concrete coordinate/angle edit ...`.
6. **Controlled geometry <-> texture feedback (T1):** Flat colour blocking often exposes subtle local proportion or clearance issues (a snout reading slightly thick, a plate crowding an eye, a limb flattening) that were invisible in white renders. Adjust local box scale or seating immediately when colour reveals a read failure—and re-run the audits. Never use this to patch a structural failure: any 5.4 trigger still punches back to W1.
7. Log in `PROJECT_GOAL.md`: what changed, the check results and the open issues.

### 5.4 Punch back to W1 when any of these is true
- The silhouette does not read as the subject from the side and front.
- The head fails its gate (`6.6`) or its proportions (`6.1`).
- The `no-wing` front/side body reads as a square locker chest, Pez chimney neck or Gundam hip-skirt golem.
- `W1 top` wings read as a flat 1D cardboard plank (`G4` camber `< 0.05`).
- Mirror parity is broken without a declared reason.
- The construction guards (`G1` or critical `construct_check.py` gates, 3.6) fail.
- A forbidden construction from 3.2 is present.
- The owner or a reviewer says it does not look like the reference or the style.

Detail and texture passes may not be used to fix base problems. Rebuild the failing region fresh from explicit centres; never bolt patch boxes onto broken coordinates.

## 6. Heads (most failures happen here)

### 6.1 Proportions
- **Head length to head height** (back of skull to nose tip, against top of cranium to bottom of the closed jaw):

  | Head type | Length : height |
  |---|---|
  | Wolf-like and fanged heads | 1.2-1.5 |
  | Flying wyverns and dragons | 1.2-1.7 |
  | Brutes | 1.1-1.5 |
  | Long-snouted leviathans and crocodilians | up to 2.2 |
  | Small and chibi creatures | 1.0-1.3 |

  These ranges come from the owner's anchors (2.6). Above the range the face is too long: **shorten the snout first.** Below about 1.1 on a dragon, the head reads as a cube.
- **Depth:** cranium height at least equal to cranium width. A cranium wider than it is tall reads flat; the Mesa Glider's 10 x 8 cranium did, and the owner called it flattened.
- **Jaw depth:**
  - predators: at least 40% of head height;
  - small prey animals: 20-30%.
- **Muzzle:**
  - length 0.6-1.0x the cranium length (up to 1.2 for leviathans; 0.4-0.6 for rodents and other short-faced small animals);
  - width 0.6-0.8x the cranium width;
  - its top 10-25% lower than the brow.
- **Widest at the cheeks or the back of the skull, never at the snout.** SyncCraft flags a snout wider than 0.88 of the skull as a "doghouse" head.
- **Width:** from the front, without horns and fins, the head is taller than wide: its width is 0.8-0.95x its height.
  - For dragons and wyverns the cranium is about 10% taller than wide.
  - The top neck segment is no wider than the cranium.
- **Size:** a large monster's head is 15-25% of its body length without the tail. Never shrink the head to make room for detail.

### 6.2 Depth & V-rail wedge construction (no flat faces or wedding-cake snouts)
A face is flat when front, top and three-quarter views show one rectangular plane. Never stack centred `yaw = 0` snout boxes like a wedding cake (verified across `Arial Galewing`, `Honey Wyvern` and `ArtsByKev Dragon` timelapses):
1. **Cranium & occipital slope:** compact tall cranium core (`height >= width`) plus a rear occipital slope plate pitched `-30 to -42°` so the skull flows cleanly into `neck_top` with zero 90-degree back wall.
2. **Multi-Block Partial Upper-Jaw Lip Bosses (`6.5d`):** instead of two oversized single straight planks, step `2` progressive compound-rotated upper-jaw lip bosses per side (`maxilla_l -> maxilla_ant_l`, `yaw = ±10 to ±16°`) framing a 4-step tapering snout core (`snout_rear -> snout_mid -> snout_front -> snout_beak_hook`). In `head top` and `0°` front views, the muzzle tapers organically with zero flat wedding-cake walls and zero stiff single-plank diagonals (`G2`).
3. **Nasal bridge, rhythmic dorsal scutes & rostral boss:** sloped central nasal deck (`pitch = -10 to -18°`) crowned with `2` rhythmic stepped dorsal nasal scutes (`snout_scute_1 -> snout_ridge`) and a narrow hooked or squared rostral nose boss (`width <= 0.45x` cheek width).
4. **3-block progressive supraorbital scowl brow (`6.5d`):** per side, `3` progressive compound-rotated blocks (`brow_inner -> brow_ridge -> brow_outer`) overhanging the balanced orbital eye shelf by `0.5-1.0u` (`0%` skull embedding, `0.00u` outer extrusion). Its geometric shadow gives the fierce expression without looking like a straight V-bar.
5. **2-stepped jugal cheekbones** per side (`zygoma_l + zygoma_post_l`) behind and under the eye, widening the head into a shield (built as in `6.2b`).

Then the **solid 4-step watertight lower mandible (`jaw` bone, `6.5d`):** a separate bone hinged inside the rear cheek mass, built from **4 solid stepped axial cuboids** (`jaw_rear -> jaw_mid -> jaw_chin -> jaw_chin_tip` — NEVER two V-shaped `jaw_ramus` planks that pinch/distort the chin!), narrower and tucked cleanly under the partial upper-jaw lip bosses (`6.5`, `6.5d`).

**Front-view layering:** working from the top down and from the back forward, a dragon head shows:
1. horn roots;
2. brow;
3. recessed eye socket;
4. cheek;
5. converging maxilla / muzzle;
6. nose boss;
7. tucked V-jaw.

Each layer steps in depth from the one before. Fewer than four visible layers reads as a mask.

**Tests:**
- *Top view:* smooth converging V-wedge with no parallel `yaw = 0` snout walls.
- *Front view:* the head outline is a wedge or shield (wide at the cheeks and top, narrower at the muzzle and jaw), never a rectangle.
- *Side view:* the profile shows the brow step, the sloped nasal bridge, the nose boss, a clear interlocking mouth line and the deep jaw hinge.

### 6.2b Cheeks
**The cheek is three forms. Build them in this order:**
1. **Cheekbone ridge:** one box per side.
   - It runs from under the eye back to the jaw hinge; its top edge sits just under the eye, so the eye stays above it.
   - It stands out 1-1.5 units from the skull's side.
   - Turn it 3-6 degrees (yaw) so it widens slightly toward the back, and roll it 8-12 degrees so its top catches light.
   - Across the cheekbones the face is 1.2-1.4x the cranium width. At 1.7x (round 2) the owner found the head too wide; the anchors' heads are narrow and deep.
2. **Jaw-hinge mass:** one box per side at the back-lower corner of the cranium, over the jaw joint.
   - It stands out at most 1.5 units and is narrower than the cheekbone, so the front view tapers to the jaw.
   - It sits on the head bone and touches the mandible without going inside it.
3. **One signature cheek feature,** in the hard-part colour:
   - a framed fin (8.8): 2-3 rays with a membrane between, sweeping back 10-20 degrees (yaw) and a little up;
   - or a single cheek spike;
   - or a cut-out frill plane.

   Seen from the front, it sits behind the head outline and adds at most about half the head's width.

**Clearance:**
- Attach the frill bone to the cranium, which is a centred box, so `mirror_x` works.
- Align its root so it starts at the cranium's side.
- Keep the top neck segment no wider than the cranium. A wider neck swallows the frill root.
- Then tilt the head 20 degrees each way (pitch and yaw) and the frill 10 degrees each way. The hinge mass, the frill and the jaw are the first parts to hit the neck; set the head's joint limits from this test.

**Paint:**
- The cheekbone's top face is one step lighter.
- A darker crease row runs under the cheekbone where it meets the jaw.

**Tests:**
- *Front view:* the head without horns and fins is taller than wide (6.1). Its outline steps from the brow to the cheekbones (widest), then the hinge, then the jaw (narrowest).
- *Front view eyes:* the eyes show at the cranium's front corners. Wrap the iris 1 px onto the front face under the brow.
- *Nostrils:* 1 px slits at the top front of the nose, one step darker than the hide. Two black dots on the nose read as a second pair of eyes from the front.

### 6.3 Eyes & Optical Sensors (Flush On-Surface for Small Animals vs. Recessed 4-Part Orbital Cavity for Heroes/Bosses, `v39.2`)
- **Rule 0 — Scale-First Eye Geometry Selection (`v39.2` Anti-Extruded-Eye Law):**
  - **Small Animals, Critters, Birds, Fish, Mammals & Smooth-Headed Small Monsters (`small` / smooth `standard`):** **NEVER extrude a separate protruding solid eye cube off the head surface!** Paint the eye **directly ON the head/cranium/cheek surface texture** (`0.00u` extrusion; optionally set `uv_scale: 2` on the cranium/head box itself if higher crispness is needed) OR use a flush zero-thickness `0.0u` hard-alpha eye plane seated `<= +0.02u` on the surface — NEVER a thick rectangular eye brick sticking off the skull wall! Only verified bulging/stalked taxa (frogs/toads dorsal domes, chameleons, crabs/shrimp eyestalks, hemisphere compound-eye insects) may bulge organically outward to match official references.
  - **Heroes, Dragons, Armored Bosses & Mechs (`hero` / `epic_boss`, `head_h >= 7u`):** Use the **4-Part Recessed 3D Periorbital Carving (`G14`)** below so the solid `uv_scale: 2` eye cuboid sits strictly **RECESSED INWARD (`eye_outer <= cranium_wall`, `0.00u` outward extrusion!)** — NEVER protruding beyond the cheek, snout, or brow envelope!
- **4-Part 3D Periorbital Carving (Large Monsters / Dragons / Bosses — never paste or extrude proud onto a flat `yaw = 0` side wall; `G14`):**
  1. **Recessed Orbital Cavity Step (`cranium_orbital` / recessed housing):** Step the skull/helmet inward by `0.75–1.5u` at the eye bay so the eye sits inside real 3D physical depth (`0.00u` extrusion in `0°` Dead-Front!), angled `16–24°` forward (`yaw`) on forward-facing predators/humanoids (more lateral on prey animals).
  2. **Sculpted Upper Eyelid Hood (`eyelid_upper`) under Overhanging Brow (`brow_ridge`):** Tuck an angled upper lid bar hugging the top `20–25%` of the eye to cast real physical shadow and form a sharp almond/hooded contour (`ArtDanPol 3D` craft).
  3. **Sculpted Lower Eyelid Cradle (`eyelid_lower`) & Lacrimal Corner:** Seat a beveled lower lid rim directly above the cheekbone (`zygoma_l/r`) to frame the lower curve and anterior tear-duct corner.
  4. **Post-Orbital Corner Rim (`orbital_post_bar`):** Close the rear lateral corner down to the temple/zygoma to complete a 4-sided sculpted orbital bezel (`Wackyblocks` diamond-bezel craft).
- **Double-Resolution (`uv_scale: 2`) 7-Layer Realistic Eye Painting (`references/high_res_focal_carving_and_visual_primitives.md` §3):**
  - Grant eye/lens boxes **`uv_scale: 2`** (`32 px/block` on `16x` rigs or **`64 px/block` = `10×6..14×8 px`** on `32x` rigs) so the eye reads as a living focal centerpiece rather than a 2-pixel dot!
  - Paint all 7 layers strictly on the exposed eye face(s) (keep hidden interior faces dark obsidian socket AO so zero color leaks onto the cheek):
    1. **Outer `1 px` dark orbital socket rim**;
    2. **Upper-eyelid cast shadow** darkening the top `25–30%` of *both* sclera and iris by `40–55%`;
    3. **Shaded moist sclera & warm canthus corners**;
    4. **Crisp `1 px` dark limbal ring** framing the iris ellipse;
    5. **Multi-zone radial iris fibers + bright lower-crescent internal refraction glow** below the pupil;
    6. **Anatomically authentic pupil** (vertical slit for dragons/felines/reptiles, horizontal bar for ungulates/amphibians, crisp round for mammals/humanoids/birds, concentric aperture reticle for mechs/sensors);
    7. **Offset `1×2`/`2×1` pure white wet specular catchlight** on the upper-front iris/limbus + subtle lower-rear bounce pinprick. Keep emissive strictly selective to mid-iris/refraction + catchlight so dark pupil, limbus, and lid shadow stay razor sharp!
- **Wrong:** pasted flat on a cheek, centered mid-height, left/right mismatched, glowing flat yellow square with no lid shadow/limbus/sclera, or two equal dark dots on the snout tip reading as duplicate eyes.

### 6.4 Horns, crests and ears on the head
- Roots sit at the back-top of the cranium or on the brow, overlapping the skull. Directions follow the reference; construction follows 7.1.
- Crest plates fan from the back of the skull over the first neck segment. This is what joins head to neck.

### 6.5 Mouth and solid 4-step lower mandible (`jaw` bone)
- **Solid 4-step mandible structure (`6.5d` — Anti-V-Plank Law):** hinge deep inside the rear masseter blocks (`>= 35-45%` of head height at the rear jaw), built from **4 solid stepped watertight axial cuboids** (`jaw_rear -> jaw_mid -> jaw_chin -> jaw_chin_tip` with upward `-10°..-12°` chin bevels — NEVER two long tilted V-rail `jaw_ramus` planks that distort the chin in front/3-4 views!) so the closed snarl, underside, and open roar (`20-34°` gape with zero hinge gap) read as a muscular, cohesive dragon mandible.
- **Teeth & rich interior 3D maw (`Anchors 06 & 09`, Blockbench Official Guide, `v23`):** **NEVER use clunky 3D volumetric tooth cubes** on standard/boss maws (as noted in the Blockbench Official Minecraft Style Guide, volumetric teeth cause Z-fighting, UV bloat, collision clutter, and "buck-tooth / chiclet" ugliness). Instead, construct a **complete 6-plane zero-thickness (`size = 0` on normal axis) single-pixel-square tooth U-arcade** (`upper_teeth_row` L/R, `upper_front_teeth`, `lower_teeth_row` L/R, `lower_front_teeth`) recessed cleanly inside the upper/lower jaw rims using **explicit integer-pixel `1×1 px` ivory squares + `1×2 px` anterior canines separated by `1 px` transparent gaps**, staggered `1 px` between upper and lower jaws for natural interlocking bite (NEVER use floating-point alpha thresholds like `tip_t <= 0.96` on `h=2 px` planes, which erase the tip row!). Pair with a **4-layer cavernous interior**: (1) dark mulberry-plum upper palate painted with `1 px` transverse fleshy `rugae`, (2) dark medial cheek/jaw walls + recessed abyssal rear pharynx backing (`mouth_throat_back`), (3) zero-thickness concave hard-alpha stretched inner cheek-tendon webs (`plane jaw_cheek_web` L/R), and (4) an **articulated 2-segment sculpted curved tongue (`bone tongue` → `tongue_base` + tapered `tongue_tip`)** in species-harmonized deep smoked plum-crimson (zero bubblegum-pink!). Reserve 3D protruding tooth/tusk boxes strictly for massive external signature tusks (e.g., Anchor 02 crystal fangs or Anchor 07 Deviljho chin spikes). Full blueprint in `references/cranial_and_jaw_structural_reasoning.md` §1.
  - **Optical lip slimming:** use dark lip-crease cutaways along the closed mouth line to sharpen the wedge silhouette without adding extra jaw slabs.
- **Rest pose: 100% closed flush interlocking sawtooth bite (`0` exposed internal teeth at rest — `v39.2`):** Keep jaw pitch closed at rest (`0°`) with overhanging upper maxillary lip curtains + staggered half-phase triangular/sawtooth-wave bite rims (paired with a crisp `1 px` dark occlusal crease) so **ZERO internal teeth, fangs, or gums peek out** from Dead-Front (`0°`), Hero `3/4`, OR Pure Side (`-90°`)! Only authentic external tusks/sabers (e.g., boar/mammoth tusks) may project outside the closed lips.
  - **Physically Anchored Rear Jaw Hinge (Zero Detached Open-Jaw Gap!):** Extend the rear mandibular root & ascending ramus/cheek-hinge plates `+1.8–2.5u` directly inside the skull's rear cheek socket (with the `jaw` bone pivot centered inside that overlapping cheek hinge + dark inner buccal cheek webs) so the lower jaw NEVER detaches or floats in mid-air when roaring open (`+32°..+42°`)!
  - **Perimeter-Aligned Inner Tooth Arch:** Place left and right upper/lower inner tooth rows cleanly along the **outer-inner perimeter edge of the jaw** (`~0.35–0.45u` inside the outer lip wall, opening a clean central tongue channel down the middle — NEVER bunched together in the center of the mouth floor!), while staying 100% concealed when closed!
  - **Anti-Flat-Gradient / True Blockbench Pixel-Relief Discipline (`v39.3`, `jump>25 ∈ [4.5%, 14.0%]`, `lum_step ∈ [5.0, 8.5]`):** NEVER substitute lazy smooth airbrush gradients (`jump>25 < 3.5%` — plastic toy look!) OR chaotic random pixel static (`jump>25 > 18%`). Hand-craft organized `2 px` stepped pixel-art scale/feather/fur/armor clusters, crisp `1–2 px` dark overlap shadow valleys, `1 px` sunlit bevel speculars, and rich `45–50°` hue-shifted cool-shadow-to-warm-light color ramps across every surface!
- **Tucked jaw:**
  - 0.5-1 unit narrower on each side than the muzzle;
  - its front 1.5-2 units behind the nose tip (an overbite);
  - its top 0.5 unit up inside the muzzle;
  - no chin lump past its front or below its line.

  A jaw as long or as wide as the upper jaw reads as "split out".
- **Colour:** the jaw is in the hide colour, with an underside-colour throat plate beneath that starts the underside column. A cream jaw the colour of the teeth hides the teeth and reads as a separate block.
- **The jaw stays.** Only creatures smaller than the player that never open their mouth may merge it into the head.

### 6.5b Snout and mandible recipe (round 8, refs r8a blue crystal dragon, r8b plum flyer)
- **Avoid a bulky head:** a 16 x 18 skull with a wide nose block and lip ledges read bulky. Slim it:
  - skull about 13 x 15 x 16 at boss scale;
  - the top neck segment no wider than the skull;
  - head length to height (without horns) about 1.5-1.6.
- **Muzzle bone** (child of the head, rest pitch about 6 so the snout slopes down from the brow). Its parts, all axis-aligned in that bone:
  - core: about 0.73x the skull's width, 5 tall, as long as the skull;
  - deck: a narrower box on top;
  - two bevel plates on the top edges, starting where the snout leaves the skull;
  - a nasal ridge on the deck;
  - a squared nose pad narrower than the snout, never a wider block.
- **Mandible:**
  - hinge at the back-bottom of the skull, so the roar gape opens wide;
  - a ramus deep enough that the jaw is at least 40% of head height;
  - a slim body 1.25 narrower per side than the snout, its top tucked 0.5 inside the snout;
  - a keel and two bevel plates underneath, giving a V-shaped underside;
  - the chin about 2 behind the nose;
  - rest pitch matching the snout slope.
- **Check:** render the head alone at rest and at full gape (side and three-quarter front) before moving on.

### 6.5c Build a head from explicit centres (round 10, owner-directed W1 rebuild)
When a head fails twice, punch back to W1 and rebuild it. **Don't keep the old coordinates and bolt parts onto them;** that is how stray strips and pegs appear. Design every box in one frame (skull centre at the origin): give each one an explicit centre and rotation, and compute each `place` offset from the centre of the box it touches. (Rotation is Rz.Ry.Rx about the box centre.) The working recipe:
- **Skull:**
  - a rear block plus a narrower frontal block;
  - an occipital slope plate (pitched about -40) so the back of the skull falls into the neck, with no sheer back wall;
  - the crystals or crest cover the rear.
- **Eyes:** an eye facet per side, a thin plate bridging the rear block's side and the frontal block's side, turned about 22 degrees forward. The eye is painted on it, so it reads in front and 3/4 views. Above it, a thick brow pitched down 10, turned in 10 and rolled in 15 (the scowl).
- **Cheeks:** a solid cheek wedge per side (about 5 x 8 x 12), yawed 13 with the back end out and rolled 8. It is the widest point and bridges the eye down to the jaw hinge. Never a thin bar.
- **Snout:** stepped and tapering in width and height: rear snout about 0.65x the cheek width, mid snout 0.5x, raised nose tip 0.4x. A nasal bridge plate slopes from the forehead to the snout, so the top line is continuous.
- **Jaw:** hinge inside the cheek masses; about 2x deeper at the hinge than at the chin (ramus, rear body, front body, chin angled up); a countershaded keel underneath. 2 fangs per side, tucked 0.5 inside the lip line and set back from the nose. No throat block hanging under the hinge.
- **Crystals or horns:** graded tiers from the brow backward, smallest in front, leaning back 45-55 and splayed a little. Jawline blades lie tight along the lower cheek.
- **Gate questions (W1, white + Texture-On):**
  - Is the top view a wedge with no parallel snout walls?
  - Is the front view a shield with both eyes visible under a segmented scowl brow?
  - Are there no floating strips, stray blocks or bare 90-degree back walls?
  - Does the open jaw (20-25 degrees) show no hinge gap?

### 6.5d Canonical `v28` Multi-Block Dragon Head, Partial Upper-Jaw & Solid Stepped Lower-Mandible Standard (Anti-Single-Plank Law)
Owner-verified on *White Fatalis `v28` (`P3_sculpted_final`)*: **NEVER rely on two oversized single V-shape planks** (`[4.5, 2.0, 9.0]` straight `brow_ridge`, `[2.5, 2.5, 7.5]` straight `maxilla`, or tilted `jaw_ramus` V-bars) to form brows, snouts, or jaws — single long diagonal bars look overly straightforward/geometric in `0°` Dead-Front view and distort lower jaws! Always construct legendary/Monster-Hunter dragon heads from **five interlocking multi-block stepped subassemblies** (full blueprint in `references/cranial_and_jaw_structural_reasoning.md` §6):
1. **Cumulative `+6°..+10°` World Head Glare & Stepped Forehead Bridge (`head`):** Sum all parent `pelvis -> waist -> chest -> neck_1..N -> head` X-pitches so net world pitch is `+6°..+10°` downward glare (never `-10°..-15°` chin-up foreshortening!), and step `cranium_core (9.5×6.5×9.0) -> crown_roof (7.5×2.0×8.0) -> forehead_step (5.5×2.0×4.0)`.
2. **3-Block Progressive Supraorbital Brow Arch (`bone brow_sculpt_l/r`):** Sculpt each brow from **3 progressive compound-rotated blocks** (`brow_inner -> brow_ridge -> brow_outer`) arching organically from `forehead_step` over the orbital socket into the horn base.
3. **1-Block Recessed Orbital Skull Step & 2-Block Stepped Cheekbones (`cranium_orbital` + `eye_l/r` + `zygoma_l` + `zygoma_post_l`):** Split the cranium into rear `cranium_core (9.5×6.5×6.0)` + **1-block narrower `cranium_orbital (6.5×6.0×3.0)`** (`~1.5u` narrower per side under the brow socket!). Step `eye_l/r` inward onto `cranium_orbital` (`center_x = ±3.52u`, `outer_x = ±4.70u`) so the eye sits **proud of `cranium_orbital` (`±3.25u` -> `0%` embedded in `3/4` & `-90°` Side)** WHILE **`outer_x` (`±4.70u`) stays strictly inside `cranium_core` (`±4.75u`) & `brow_ridge` (`±6.20u`)** — eliminating all lateral eye extrusion in `0°` Dead-Front view! Separate crescent/antler horn tips cleanly (`~12–14u` crown gap).
4. **4-Step Snout Core + Rhythmic Nasal Scutes + 2-Step Partial Upper-Jaw Lip Bosses (`bone snout_assembly`):** Step `snout_rear (7.0) -> snout_mid (6.0) -> snout_front (4.5) -> snout_beak_hook (3.5)` crowned by `snout_scute_1 -> snout_ridge` + snout horns, with **2-step maxilla lip bosses** (`maxilla_l -> maxilla_ant_l`) framing external interlocking upper canines (`tusk_upper_l/r`) and 3 zero-thickness upper tooth planes.
5. **Solid 4-Step Watertight Lower Mandible (`bone jaw` — ZERO Crossed V-Rails!):** Step `jaw_rear (8.5) -> jaw_mid (6.5) -> jaw_chin (5.0) -> jaw_chin_tip (3.5)` as **solid axial cuboids** (never crossed V-rail bars!), holding a tense `14–16°` snarl gape with compact lower canines, 3 lower tooth planes, dark mulberry 2-segment tongue, and backward-swept (`Rx +28°..+34°`) 3D chin/throat beard cores!

### 6.6 Head gate (end of W1 and every later pass)
Render the head alone at 9 px per unit or more: `head side`, `head front`, `head q_front` and `head top` (both closed rest AND `25°` open roar). Compare with the head close-ups.

**It passes only when:**
- the proportions are in range (6.1) and from the front (without horns and fins) the head is taller than wide;
- `head top` reads as a clean converging V-wedge with zero `yaw = 0` wedding-cake snout steps (`G2`);
- at least 4 front-view depth layers (`horn roots -> V-brow -> recessed eye pocket -> cheekbones -> converging maxilla -> nose boss -> tucked jaw`) read cleanly;
- eyes sit inside recessed angled sockets per 6.3;
- horns read as real blocks with open negative space (`G3`, `<= 14` boxes on `head` itself — multi-segment horns live on child bones);
- the jaw is present, closed and tucked at rest, and opens (`25°` roar) as a tapered triangular V-mandible with zero hinge gap;
- a viewer could name the species from the head alone.

Otherwise stop and rebuild the head fresh from explicit centres (`6.5c`) before anything else.

## 7. Horns, negative space, spikes and plates

### 7.1 Horn types & Crown Negative-Space Law
**Choose the horn type first, because it sets every angle:**

| Type | Root | Segment angles | Extras |
|---|---|---|---|
| Antler | back-top of the cranium | the direction changes at every segment: up and out, then forward and in, then back and out, 20-30 degrees each | one tine (a single box on its own bone) from the second joint, pointing forward and out |
| Swept / Elder | back-top of the cranium | sweeps back 40-65 degrees along the skull and **splays outward 18-32 degrees** with a steady 10-20 degree curve | check it clears the neck when the head tilts back |
| **Forward Oni / Bull-Lyre** (*Zinogre*, *Diablos*, *Nergigante*) | supraorbital brow / front-crown (`crown_roof`, separated across midline furrow!) | **ALWAYS verify concavity first (`§1B`)!** For **CONCAVE-DOWN (`∩`)** forward-arching oni spears (*Zinogre*): surge steeply upward-and-forward (`hb_1` `[-65°, ∓25°, ±8°]`), then arch progressively forward (`+12° -> +9° -> +6°`) so tips still spear `~20°` above horizontal; for **CONCAVE-UP (`∪`)** bull hooks (*Diablos*): launch shallower (`-15°..-24°`), then curve increasingly upward (`-15° -> -18° -> -14°`). Bow wide into a `( )` lyre (`mid_sep >= tip_sep >= 8u`)! | pair with swept-back `occiput_crown_l/r` transition plates + `1–2` stepped dorsal corrugation ridges (`hb_ridge_1..2`); verify BOTH sweep direction AND concavity (`CONCAVE_DOWN ∩` vs `CONCAVE_UP ∪`) via `visual_primitives_prep.py trace` on **native 1:1 orthographic side profiles** FIRST, and assert compiled world YZ segment pitches (`cranial_and_jaw_structural_reasoning.md §7.1`, `high_res_focal_carving_and_visual_primitives.md §1B`)! |
| Brow or rhino | on the brow | rises forward and up, curving forward 10-15 degrees | blend the root with the brow |
| Ram | side-back of the cranium | a spiral in one plane, 4-5 segments, 30-45 degrees per step | the tip points forward |
| Crystal crown | back-top / brow | 2-3 asymmetric faceted prisms of mixed height, splayed outward | seat in dark rock/scale root sockets (Anchor 02); glow and bleed at the base |

**Crown Negative-Space Law (anti-"french-fry / organ-pipe" rule, `G3`):**
- Never crowd the skull roof with `6-12` parallel horns or crystal sticks at a uniform pitch (`pitch std < 10°`).
- Use **1 dominant mirrored primary pair** (root `>= 1/5` cranium width, splayed outward `>= 10-32°` so open sky widens between the tips in `front` and `top` views — never flat `0°` parallel bars) plus **at most 1 subordinate cheek/brow pair** (`<= 0.6x` main length and diverging by `>= 20°` in pitch or yaw), plus at most 1 low nasal/occipital ridge (`<= 0.4x` main height).
- Parent multi-segment horns onto **dedicated child bones** (`horn_l_1 -> horn_l_2`) rather than bloating `head` past 14 boxes.

**Process:**
1. Root.
2. Segments.
3. Taper.
4. Tines.
5. Joint test: tilt the head ±20 degrees and check that the horns clear the neck, collar and frills. Antlers hit the neck first.

- **Segments:** 2-4. Use 5 only for a signature horn longer than the head.
- **Root thickness:** at least one fifth of the cranium width, and at least 2 x 2 units on large creatures.
- **Taper:** each segment 70-85% of the previous one, and at least twice as long as it is thick.
- **Curve & Mandatory Concavity Verification (`v34`):** BEFORE setting horn/antler/tusk/beak/tail/blade/claw/arch Euler angles, run `visual_primitives_prep.py trace` (`label_concavity: true`) on a native 1:1 orthographic side profile to verify mathematically whether the structure bends **`CONCAVE_UP (∪)`**, **`CONCAVE_DOWN (∩)`**, or **`S_INFLECTION (~)`** (`references/high_res_focal_carving_and_visual_primitives.md §1B`), and confirm compiled 3D world-space segment pitch angles match both the concavity sign and terminal tip angle (never rely on endpoint direction or scalar Euler sums alone!). Bend 10-25 degrees per segment smoothly and seat each segment 20-25% into the previous one.
- **Pairs and secondaries:** pairs are mirrored with outward splay (`±10°` minimum). Secondary horns are shorter, divergent and a different length from the main pair.
- **Tip & Species Horn Surface Patterns (`16 px/block`):** the last segment may be a small chiselled box (`hb_tip`). Paint the brightest honed ivory/gold highlight **strictly on the terminal tip segment** (never striping intermediate segment ends), anchor `hb_1` with a **`1–2 px` dark keratin root cuticle collar**, keep left/right horn roots separated across a scale/hide crown furrow (`0%` fused monobrow block), and carve species-defining horn relief across faces — e.g., **transverse V-chevron corrugation grooves** on `up`/`south` + **staggered dark porous Thunderbug/keratin pits** or Greek-key flutes (`Anchor 06`) on `west`/`east` (`cranial_and_jaw_structural_reasoning.md §7.2`).

### 7.2 Graded diamond spikes
- **One box each** (3.2).
- **Shape:**
  - slim: at least 2.5x as tall as wide (wider boxes read as posts);
  - a square section turned 45 degrees about the long axis, so a ridge faces forward and one side catches light. A square section stays mirror-symmetric, so this is allowed on the centreline; paired dorsal/lateral rows splay outward `±10°` rather than standing vertical at `0°` (Longridr).
- **Paint:** the root third one step darker, the tip third one step lighter. Crystal spikes (prisms) get a light top and one bright facet.
- **Dramatically graded rows (5-12):** never repeat identical fence-post spikes at one height and angle. Grade largest over the withers/shoulders (`1.8-2.2x` neck/tail size), stepping down smoothly along neck and tail with progressive lean (`20 -> 42°`).
- **Clusters of thin spikes** are cut-out planes.

### 7.3 Plates and armour
- **Rows:** plates follow the body's flow lines, and each covers 30-50% of the next. A row shares one lean.
- **Size:** largest over the shoulders, smaller toward head and tail.
- **Underside scutes:** wide (nearly the body's width), sunk so about 1 unit protrudes, parallel to the body.
- **Roof-tile overlap lighting (Anchors 01, 02, 05):** every overlapping scale, pauldron or belly plate pairs a 1 px brighter top/leading rim with a **1 px dark cast-shadow row directly on the plate beneath/behind it**. That highlight-over-shadow pairing is what makes stacked boxes read like sculpted overlapping armour.
- **Only plates that change the silhouette are geometry.** The rest are paint.

## 8. Body parts

### 8.0 Choose the process by body type
Two proven master starting orders exist—pick the one that fits the subject, while always passing the head gate (6.6) before W2:
- **Focal head-first** (best for species-driven monsters and dragons): lock the head, jaw and horns at origin first to fix the species identity and `1 head-length` ruler, then build the torso, limbs, wings and tail outward.
- **Skeleton-hierarchy graybox first** (best for complex articulated flyers, quadrupeds and upright bosses): pre-create the nested anatomical bone tree (`body -> chest -> neck -> head -> jaw`, `leg_l -> shin -> ankle -> toes`, `wing_l -> forearm -> fingers`) in uniform neutral graybox so global proportions and kinematic reach lock before sculpting individual regions.
| Type | Wing or front limb | Legs | Build order |
|---|---|---|---|
| Six-limbed dragon | wings rooted on their own shoulder mass above the forelegs | four legs: fore pair straighter, hind pair zig-zag | torso masses, legs, then wings so they clear the legs |
| Wyvern (wing-walker) | the wing arm is the foreleg: thick humerus and forearm, a wrist claw that touches the ground, fingers folded back along the forearm | two heavy hind legs with a large thigh, crouched | hind legs and tail first for balance, then wing arms |
| Upright elder or biped | wings on the back; long arms with clawed hands | long digitigrade legs, three long toes | spine curve, legs, arms, wings |
| Feathered or bird-like | arm hidden under three rows of feather planes | thin tarsus, big thigh fluff, splayed toes | body, then feather rows from inner to outer |
| Insect or many-legged | thin plane wings with a vein texture, or wing cases | femur up and out, tibia down, small foot | body segments, then legs in mirrored pairs |
| Machine | panels, rotors or none | pistons and linkages (8.7) | chassis, joints, then limbs |

**Starting proportions in head-lengths (head : neck : torso : tail):**

| Body plan | Ratio | Notes |
|---|---|---|
| Western dragon | 1 : 2 : 3 : 4 | chest deep enough for two shoulder girdles |
| Wyvern | 1 : 1.5 : 2.5 : 3.5 | big chest, small hips |
| Upright elder / biped boss | 1.2 : 1.4 : 2.6 : 3.8 | S-curved predatory spine, broad V-taper chest, active clawed arms |
| Drake or fanged beast | 1 : 0.8 : 3 : 2.5 | heavy forequarters |
| Brute | 1.2 : 0.6 : 2.5 : 2.5 | thick counterweight tail |
| Eastern dragon | 1 : body 10-14 in one long S | the body is the neck |
| Serpent or leviathan | 1 : body 8-12 | the read comes from the curve and the fin rhythm |
| Elder or boss | add 20% to the head and the signature feature | |

### 8.1 Neck & trapezius bridge (no Pez-dispenser chimneys OR Elder-Dragon giraffe-neck drift!)
- **Match neck segment count & reach strictly to the species body plan (`cranial_and_jaw_structural_reasoning.md §7.3`):**
  - *6-limbed Elder Dragons / Flying Wyverns:* `4–5` supple S-curved segments (`1 : 1.5–2.0` head-to-neck ratio).
  - *Fanged Wyverns & Heavy-Forequarter Quadrupeds (*Zinogre*, *Odogaron*, *Magnamalo*, *Tigrex*, wolves/lions/bears):* **NEVER clone a `4–5` segment `34u` Elder-Dragon neck!** Use **`3` ultra-thick, deeply seated (`seat = 0.34–0.38`) bull-wolf segments** (`net reach ~ 0.65–0.85x` head length, `~11–14u`) surging compactly out of the `>= 1.50x` barrel chest so the head sits powerful and low-slung right ahead of the shoulders.
- Each segment tapers smoothly from a deep base (`~0.85x` chest front depth) to `<= 0.95x` cranium width at the head.
- **Hexagonal / keeled cross-section (Arial Galewing `0:45`, `G5`):** bevel neck segments with angled lateral muscle slopes or a `45°`-rolled ventral throat keel plus overlapping throat scutes so front/3/4 views never look like stacked square pipes.
- **Trapezius / clavicle gussets (`G5`):** bridge the wide shoulder girdle diagonally up into `neck_1` with mirrored gusset wedges (`roll ±28-38°`, `yaw ±12-18°`) and shingled collar plates so `no-wing front` shows a muscular sloping shoulder-to-neck triangle, never a vertical pipe on a horizontal shelf.
- **Joint-overlap skin (Longridr):** extend hidden rear neck cores slightly inside their parent sockets and test extreme upward roar bends so no gaps open.
- Add throat plates and a dorsal row that continues from the back.

### 8.2 Torso & Anti-Golem upright boss rules
- **Three pitched, chamfered masses:** deep keeled `chest`, narrower cinched `waist` (`W_chest : W_waist >= 1.5-1.7 : 1`), and muscular `hips`, pitched relative to each other so the back line arches, with lateral rib/latissimus slopes so neither top nor front view is a rectangular crate (`G5`).
- **Shoulder and haunch masses** on the flanks (including a muscular "meaty" wing-shoulder bridge on six-limbed dragons); limbs attach to these.
- A smaller box on the chest for withers or a hump, so the top edge isn't a straight rectangle.
- **Connection check:** in `no-wing` side and front views the outline from skull to tail tip is one flowing muscular line, never a row of crates.
- **Upright bosses (round 10, owner + Opus v17):** kill the stiff "man-in-a-suit / golem" reading (a column torso, arms pinned to the ribs, pillar legs):
  - **Predatory forward lean:** rear the waist (`-45 to -60°` from horizontal, never a `90°` vertical pole), then pitch the chest **forward** `+12 to +20°` relative to the waist into an athletic S-curve.
  - **Torso:** a broad V-taper chest (`~2.1x` skull width with shoulders flaring past it), cinched waist, solid hips, and a **proud forward-protruding keeled sternum** of shingled underside plates (never a flat sticker).
  - **Active predatory arms (`8.6`):** arm length `>= 0.80x` torso length; roll upper arms outward `18-28°`, flex elbows forward `45-65°` with elbow cap + bony spur, and hold spread half-curled clawed hands clearly in front of and outside the thighs in every clip.
  - **Dynamic digitigrade legs:** splay knees outward (`thigh roll -8 to -14°`), sweep shins sharply back (`-50 to -65°`), pitch elongated hocks/metatarsals forward (`+40 to +55°`), and level feet with counter-roll.
  - **Tail:** the base (`~0.8x` hip width/depth) blended smoothly into the haunches.
  - **Head:** large enough to command the silhouette next to the wings. Size it against the chest's width as well as against body length.

### 8.3 Limbs, proximal `meat` taper wedges, Regional Zoom Topology Verification & Stretched T-Pose Ratio Discipline
- **Trace THIS creature's real skeleton from Pass-2 Regional Ultra-Zoom crops (`5.1`) — NEVER apply a one-size-fits-all default leg:**
  - Do NOT force every imaginary creature into one hardcoded limb template! Before placing limb boxes, run the **2-Pass Grid-First -> Regional Ultra-Zoom** on the reference's forelimbs AND hindlimbs separately so vision compression never hides the real segment count, fold structure, or relative length.
  - **Verify BOTH Folded Silhouette AND Stretched-Out (T-Pose) Segment-Length Ratios:**
    - A frequent hidden error on crouched or pitched creatures (discovered via T-pose inspection on *White Fatalis* and *Zinogre*) is calibrating limbs solely to reach `y = 0` from a lower pelvis (`pelvis_y < chest_y`) using short, barely-bent segments — making a folded leg read like a straight vertical peg AND making the stretched-out rear legs accidentally shorter than the front legs when the reference actually shows longer, deeply folded legs!
    - Always check **two complementary truths** against your zoomed limb crops:
      1. **True Fold Structure & Frontal Plane (Anti-Frog-Splay Law):** Match BOTH the side-profile fold topology AND the front/rear-profile limb plane of the zoomed reference! E.g., when the zoomed reference shows a **parasagittal 3-segment digitigrade Z-fold** (dragons, wyverns, wolves, felines, raptors), align thigh -> shin -> ankle in a **tight vertical-sagittal plane** (`thigh` roll `-6°..-12°`, knee & hock near `0°` lateral roll, `plane: sagittal`, pelvis lifted cleanly off the floor rather than squatting flat!) instead of `-28°..-36°` frog-leg hip splay or `plane: sprawled` poles; conversely, when the zoomed reference genuinely shows **pillar** (elephants/golems), **plantigrade** (bears/apes/humanoids), **sprawled** (lizards/crocs/frogs), or **arthropod** limbs, build THAT exact architecture!
      2. **True Stretched-Out (T-Pose) Segment Ratio (`hind_tot : fore_tot`):** Sum pivot-to-pivot segment lengths (`upper + lower [+ metatarsus]`) and confirm the stretched-out hind-to-fore ratio matches what the creature's zoomed anatomy implies (e.g., `~1.20–1.45×` longer rear legs on coiled/semi-erect hunters like *White Fatalis* or *Zinogre*, vs. longer forelimbs on knuckle-walking apes or pterosaurs), and give folded joints wide extension `joint_limits` so limbs unfold cleanly in T-pose, leaping, or flight!
- **Proximal `meat` muscle-taper wedges (`Arial Galewing` `meat3`, `G6`):** never glue flat vertical outer slabs (`roll ~ 0°`) onto hips or shoulders ("Gundam hip-skirts"). Pitch the main femur/humerus along the bone line and tuck a rear/dorsal **hamstring/quad or triceps `meat` wedge** pitched `-20 to -35°` against the shaft and rolled `8-14°` inward so the proximal limb is `~2.0-2.3x` knee/elbow width and tapers smoothly downward.
- **Choose the stance from the reference:** pillar (heavy), cursorial zig-zag (most dragons' hind legs), crouched biped (wyverns) or sprawled (lizards, insects).
- **Segments:** thigh or shoulder mass, then upper segment, lower segment and foot.
  - Thickness steps down about 30% per segment.
  - The foot widens again, to about 1.3x the lower leg.
  - On smaller or cleaner Minecraft-faithful creatures, proactively **dissolve** over-segmented 3-part legs into 2 bolder, better-angled segments if the silhouette reads stronger (3.0).
- **Knee or elbow box, calf bulge and heel block.** These three boxes per leg turn a post into a leg. Fake cylindrical thigh and shin curvature in paint by subtly darkening the outer corners along the limb's length (10.2).
- **Arched heavy foot / paw (`G6`):** one heavy arched block that continues the limb (`height >= 0.55-0.65x length` and `>= 0.6x` lower-leg thickness), blended into it. Never a flat horizontal snowshoe plank (round 6: a 6-unit plank under a 13-unit leg read as a board).
- **Toes:** 3-4 thick blocks, each about a third of the foot's width, with small gaps between them. Splay them `8-15°` outward and pitch slightly downward (`-20 to -30°` when gripping), protruding at least half a toe length.
- **Claws:** one block per toe, 0.7-0.8x as wide as the toe and longer than it is wide, pitched down 15-20 degrees so the point meets the ground.
  - Paint it dark where it leaves the toe and pale at the point (the elder anchor's coloured tips). The middle claw is the longest.
  - The owner prefers block claws to sheet claws.
  - A small pale cube on a thin toe reads as a marshmallow. The cure is a thicker toe and a darker root, not a sheet.
  - Alternative (Anchor 03 red & cream dragon): toes in the underside colour as contiguous blocks across the front of the foot, with no separate claws.
- **Level the feet:** sprawled or splayed legs tilt the foot. Set each foot's rest angle from its world tilt, then seat the root so every foot lands.
- **Grounding:** toes rest on the ground (y = 0). Splay legs at the knee, with the opposite roll on the foot, not at the hip.

### 8.3b Articulated boss hands, raptor feet, splayed phalanges & dewclaws (`G9`)
Never build hero/boss extremities as a flat square brick (`foot_base`) with 3 identical parallel `2x2x4` peg boxes (`G9`):
1. **Tapered Carpal / Tarsal Bridge + Dorsal Knuckle Shield:** Slope the hand/foot bridge (`+14 to +24°`) narrower at the wrist/ankle and widening gently across the knuckles, topped by an overlapping dorsal metacarpal/metatarsal scute and anchored behind by a **calcaneal heel block or ankle spur**.
2. **4-to-5 Digit Splay & Middle-Digit Hierarchy (ArtsByKev Vampire Boss `0WTvd2KNRqs`, Longridr `yyQTcdpzLS8`, Darto 3d Rathalos `Z58DMwfZtx4`):**
   - **Fore-hands / grasping talons:** 3–4 splayed forward digits (`inner`, `middle`, `outer`[, `pinky`]) fanned across `24–42°` total yaw/roll spread + **1 opposable inner thumb / pollex (`thumb_1 -> thumb_2`)** angled inward/downward `25–40°`.
   - **Digitigrade hind feet:** 3 splayed forward raptor toes (`±18–28°` outer splay) + **1 elevated or ground-gripping rear hallux / velociraptor dewclaw** behind the ankle.
   - **Length grading:** grade the **middle forward digit `1.20–1.35x` longer** than the inner/outer digits — never clone 3 identical boxes!
3. **Multi-Jointed Phalanx -> Hooked Talon Chains (`epic_boss` mandatory):** Chain each forward digit across `2–3` child bones (`proximal_phalanx -> distal_phalanx -> hooked_talon`), pitching distal segments progressively downward (`-18° -> -35°`) so the hand/foot grips the ground with predatory tension.
4. **Optical Claw-Hook & Crevice Sculpting (Longridr `87jJmKAgFqE`, Mickey Joe `4w_N8r4j2vc`):** Keep block claws thick enough to avoid sheet-thin slivers, then carve a razor-sharp hooked underside optically using dark underside cutaway pixels / local-symmetry alpha notches and a dark-root -> ivory/gold-tip ramp.
5. **Analytical `SO(3)` 100%-Flat Grounded Dragon Paws & Compact Arched Monster-Hunter Claws (`v29`, Capcom `wf_01/wf_02`):**
   - **Never carve over-long horizontal ski-toes (`8–9u`) or single-axis guessed foot pitches:** On pitched/rolled dragon forelimbs (`R_fore = R_chest · R_hum · R_fore`), solve exact 3D `SO(3)` world-flat sole orientation `R_paw_local = R_fore^T · R_y_world(splay_deg)` + sub-`0.5u` wrist `offset_y` so all 4 bottom corners of `paw_pad` (`[6.0, 3.0, 5.0]`) and `foot_pad` (`[7.0, 3.0, 5.5]`) sit at **`y = 0.0000u`** (`0.00u` tilt, `0.00u` hover!).
   - **Compact Predator Knuckle Arch (`<= 4.5–5.0u` forward toe reach):** Pitch proximal scaly phalanges (`[1.5–2.0, 2.0, 2.5]`) slightly **upward** (`Rx = -12°`) from the dorsal knuckle hood (`paw_knuckle` / `foot_knuckle`), then hook narrower razor-tapered obsidian talons (`[1.0, 1.5, 2.5–3.0]`) sharply **downward** (`Rx = +30°`) so claw tips kiss just above `y = 0` without dragging like skis during gait swing!

### 8.3c Canonical `v30` Sloped Joint-Continuity Blueprint (Wrist/Paw, Hock/Ankle, Foot Instep & Groin/Sternum Bridges)
To eliminate `90°` joint stair-steps while preserving pure Blockbench cuboid facets (`min(size) >= 2.5u`, `0.5u` grid, `0` SAT collisions across all animations):
- **Forelimb Shoulder-to-Elbow-to-Wrist Continuity Suite (`arm_l/r_upper -> arm_l/r_fore -> paw_l/r`):**
  1. **Elbow Socket Tuck (`seat = 0.28`, `anchor_z = -0.16`):** Seat `arm_fore_core` deep inside `arm_hum_core` so the flat `5.5×6.0` top pipe end of the forward-bent (`-56°`) forearm never juts out into open air in front of the bicep!
  2. **Half-Angle Elbow & Armpit Caps:** Bridge the shoulder armpit with `arm_axilla_gusset` (`[5.5, 6.0, 6.5]`, `rotate: [-12, 8, 28]`), bridge the anterior elbow crease with `arm_bicep_bridge` (`[5.5, 5.5, 4.5]`, `rotate: [-28, 4, 10]` on `arm_l_upper`), and cap the posterior/outer elbow with `arm_olecranon_cap` (`[5.5, 5.5, 4.5]`, `rotate: [+28, 0, -10]` on `arm_l_fore`).
  3. **Forepaw Carpal Bridge Quartet (`min_y >= +0.78u`) + Distal Tone Gradient:** Combine `arm_wrist_back_slope` (`[5.0, 5.0, 4.0]`, `rotate: [-22, 0, 0]`, `min_y >= +0.85u` on `arm_l_fore`) with `paw_instep_slope` (`+36°`), `paw_wrist_bridge` (`-30°`), and `paw_cuff_bevel` (`+16°`) on `paw_l`, and blend the distal forearm stocking ramp smoothly into the paw scale value so `3/4` and inner profiles show zero abrupt color seam!
- **Hindfoot Tarsal Bridge Triad (parented to `foot_l/r`, `min_y >= +0.65u`):**
  1. `foot_instep_slope` (`[5.5, 3.5, 4.5]`, `rotate: [+38, 0, 0]`) — replaces the `90°` vertical-ankle-to-horizontal-foot L-corner with a `38°` dorsal slope;
  2. `foot_heel_slope` (`[5.0, 3.0, 3.5]`, `rotate: [-34, 0, 0]`) — bridges the calcaneal heel smoothly into `ankle_tendon`;
  3. `foot_tarsal_collar` (`[6.5, 2.5, 4.5]`, `rotate: [+16, 0, 0]`) — steps width cleanly `5.5u (ankle) -> 6.5u (collar) -> 7.0u (foot_pad)`.
- **3D Vector-Aligned Digitigrade Hock Bridge (parented to `leg_l/r_ankle`, `seat = 0.28–0.30`):**
  - Seat `ankle_core` deep inside `shin_core` so its flat top corner never pokes out behind the hock, lengthen `ankle_tendon` up into `shin_calf`, and align `hock_cap_slope` (`[5.5, 5.5, 5.5]`, `rotate: [+26, 0, -22]` on Left / `[+26, 0, +22]` on Right) + `hock_front_slope` (`[4.5, 4.0, 3.5]`, `rotate: [+30, 0, -14]`) along $\mathbf{c}_{\text{shin}} - \mathbf{c}_{\text{ankle}}$.
- **Dynamic Animation SAT Rule:** Tag each joint-bridge cuboid with the shared joint-flexion `sculpt_group` (`wrist_flex`, `ankle_flex`, `knee_flex`, `hip_flex`, `fore_thorax`) and explicit `blend_with` to its parent shaft and sibling knuckle so reverse-foot rocker, aerial tuck, and torso breathing remain `100%` collision-free across `288`-frame swept-OBB audits!

### 8.4 3D Vaulted Wings (no 2D planks or armpit walls)
- **Size first, and bigger is allowed.**
  - Span at least 1.8x nose-to-tail, measured with the wings open (a display or roar pose). A rest pose may raise, fold or drape the wings; then measure the span in the open clip and say so in the log. The anchors measure 2.2-2.5x, and larger is fine.
  - Each wing reaches at least 1.5x nose-to-hip.
  - Small, level, or single-finger wings read as constrained.
  - **To grow a wing,** lengthen the forearm and fingers and thicken the spars with them; the leading finger is about one ninth as thick as it is long. Root on a meaty shoulder mass over the dorsal chest.
  - **The only upper limits are clearance:**
    - the wing tips stay above the ground at rest and in every clip;
    - the inner bays clear the spine at the top of the flap;
    - the flap and run clips pass the swept audit.
- **3D Aerodynamic Vault (`G4`, never a flat 1D plank in top view):**
  - **Upper arm (humerus):** rises steeply `60-75°` from the shoulder mass **up, out and slightly aft (`12-22°` back sweep)** so the elbow sits at or above head height.
  - **Forearm:** sweeps **out and forward (`18-30°` forward inflection)** almost level (`1.4-1.7x` humerus) to form the high wrist apex.
  - **Curved 2-segment fingers:** fan `25-35°` apart down and out from the wrist (`f1` longest at `1.2-1.5x` forearm), with **both in-plane knuckle bend (`10-18°`) and out-of-plane 3D cup (`10-20°` pitch/roll across forearm -> proximal -> distal phalanx)** so `W1 top` reads as a **3D parabolic vault (`SVD camber ratio >= 0.05`, target `>= 0.08`)**, never a flat cardboard board.
  - Upper arm about 0.75x the lower leg's thickness, forearm about 0.65x. Fingers are 3-5 units at hero tier; at boss tier, scale them with the arm (about 0.65-0.75x the forearm).
- **Palm & spaced knuckles (`G4`):** make the wrist a block about twice as deep as it is wide. Space the knuckles along it so neighbouring finger roots sit **`>= 1.5-2.5u` apart** (at least the sum of their half-thicknesses). Co-located roots overlap too deeply for SAT audits.
- **Fingers:**
  - 3 at hero tier, 4 at elder tier, 25-35 degrees apart.
  - The longest is 1.2-1.5x the forearm.
  - The two longest fingers get a knuckle (two bones).
  - The trailing finger is the thinnest and points toward the hip.
- **Open the fan downward:** twist the forearm about its own axis. Never roll the hand. Below the forearm, use yaw for bay alignment plus controlled distal phalanx cup for 3D camber.
- **Leading edge:** 2-4 armour boxes in a row along the forearm top, plus a thumb claw at the wrist. **Elder variant:** the forearm itself is a chain of 5-7 rounded knob blocks, 1.6-1.8x the spar thickness, alternating about 0.8x in size. Paint them with light pooled mid-face and dark rims, so they read as pebbles rather than cubes.
- **Membrane partition & catenary armpit clearance:** one cut-out panel per bone/phalanx, parented to the spar on its leading side.
  - Cut each panel in alpha to its bay polygon. Once the knuckles are spaced, use root-to-root quads: finger root, finger tip, next finger's tip, next finger's root.
  - **Kill the armpit cardboard wall:** anchor the inner bay (`plagiopatagium`) on the **mid-flank** (never near the dorsal spine) and carve its lower trailing edge as a **deep concave catenary arch** sweeping from the trailing finger tip up toward the armpit/latissimus so front and 3/4 views preserve open negative space around the waist and chest.
  - Scallop and tear only the trailing edge; at most two holes in the outer panels.
- **Panel heights** (in each spar's frame):
  - inside the panel's own spar's thickness, outside every spar its rectangle crosses. Recheck every height whenever spars get thicker.
  - consecutive panels differ by at least 0.125 (use `layer: k`).
- **Folding & offset fold pivot (Longridr):** fold with yaw at the elbow, wrist and fingers; flap with roll at the shoulder. When a dragon must fold wings flush against the flank AND open wide cleanly, add a small offset fold-pivot bone at the wing root ("secret sauce") and test the folded pose from above before texturing.
- **Joint limits that come from big wings:**
  - the arm can rise only until the inner bays near each other over the spine;
  - the tail can swing only until it reaches the inner bay.

  Set both from the 30-degree joint test.
- **Poses:**
  - *raised display* for turnarounds: upper arm up and back, forearm up and forward, fingers fanned down and out;
  - *folded at rest*: fingers closed and pointing to the tail, the wing held against the flank;
  - *wing-walk*: wrist on the ground, panels stacked along the forearm;
  - *frontal plume*: the arm pitched about its own axis so the whole fan faces forward. It frames the head from the front and turns edge-on from the side, so a signature on the back stays readable (Ashmantle, 14).

  The folded, wing-walk and plume poses are mirrored on both sides. In a raised pose the two inner (forearm) panels' rectangles can meet over the back: clip the forearm panel to its spar and set the closing limits from the joint test.
- **Feathered wings:** arm boxes under 3 rows of feather planes, with 3-5 separate primaries at the tip.

#### 8.4b Wing process: sketch, then build (versatile wings)
Build every wing from a 2D sketch in its own plane (SyncCraft tool `wings.py`, proven on four wing types and a boss).
1. **Sketch the front view of the display pose.** u points outward from the shoulder and v toward the leading edge. Mark S (shoulder), E (elbow) and W (wrist), plus one polyline per finger starting at the wrist. Each knuckle is a bend in the polyline.
2. **One straight run becomes one bone with one spar box,** rotated by yaw only. The rest yaw is the run's angle minus its parent's angle; positive yaw moves the outer end toward v. The skeleton stays in one plane, so every membrane can be a flat cut-out plane.
3. **Orient the plane with one plane bone on a scapula block:** rest pitch is the tilt (-90 stands the sketch up; past -90 the tops lean back) and rest yaw is the sweep. Put the sweep on this bone, never on a bone whose parent is pitched: a yaw under a reared chest turns about a tilted axis and drives the wing tops inboard.
4. **Bays are polygons in the sketch:** hub, finger, trailing curve, next finger. Positive sag makes a concave scallop; negative sag makes a slack bulge for folded wings or an overlapping layer.
   - Finger bays belong to the hand bone while the fingers do not move independently.
   - Split the inner bay into hand, forearm and root panels. The root panel gets its own bone at the shoulder, framed along u, and clips move it at half the humerus angle, so neither the body edge nor the arm edge opens a gap.
   - Neighbouring panels overlap by one curve sample, so no seam line shows.
5. **Panel rectangles are audited, not their alpha.**
   - Each rectangle must contain its bone pivot and never reach inboard of the shoulder.
   - Stagger panel heights so no two panel rectangles in a wing are coplanar.
   - Add `blend_with` for every spar a panel crosses. Where the inner bay meets the flank, blend it with the hips, thighs and shoulder, naming mirrored boxes in full (`leg_r__thigh`).
6. **Leave free tips:** end a bay at 70-90% of a finger (`tip_frac`) when the reference shows spars running past the membrane.

**Wing types from one module** (the lab sheet shows all four):
- *classic*: 3 knuckled fingers, scalloped bays, radial paint, block hook claws on the tips;
- *elder*: knob-chain forearm, 2-3 long fingers with free tips, layered torn bays with a gradient;
- *folded*: a vertical forearm, the fingers parallel to it, slack (negative-sag) bays and a saw edge;
- *glider*: 4 fingers spread low and wide, with a combed edge.

The owner's M still holds for display poses: the elbow is the high point, the forearm runs nearly level and the fingers fan down and out.

**Edges and membrane paint:**
- *scallop*: a concave arc plus a 1 px light rim (Anchor 01 plum wyvern).
- *torn*: lanes along the trailing edge 2-2.5 units wide, about half of them cut 0.5-1.0x the tear depth (12-17 units at boss scale), plus 4-6 blob holes inside the outer band (Anchors 03 & 05).
- *saw*: teeth 2.5 units wide and tall.
- *comb*: fine fringe.
- Paint *radial* (dark field with long light streaks radiating from the hub, Anchor 01) or *gradient* (saturated at the arm, pale at the edge, bands curving around the hub, Anchor 05) or *hatched parchment* (diagonal painterly brushstroke bands with ragged outer tears, Anchor 03).
- Use soft-layered directional strokes and straight radial rib lines (`Shift`-line discipline in Blockbench) to build smooth organic membrane tension, then carve all trailing scallops and tears with a **hard 100%-opacity alpha cut** (never fuzzy semi-transparent borders).
- Keep membrane dashes calm: no more than 12-20% of lanes, 2 px lanes and 6-12 px dashes (about 5 mean step and 6% jumps).

#### 8.4c Wing depth, armour and damage (round 10)
- **Depth without breaking panels:** keep each region planar and bend only at the elbow and the wrist.
  - Roll the forearm forward about 8 and the hand back about 15, so the fingers fan back and the 3/4 view has depth.
  - These roll axes run close to the inner bay's split lines (W to the curve, E to the curve), so the panels barely gap.
  - Re-measure the span afterwards, because a backward fan shortens it.
- **Leading edge:** 4-6 broad, thin, shingled plates (thin through the wing, tilted about 14, tapering toward the wrist). Never a chain of identical cubes (a "caterpillar track").
- **Fingers:**
  - spars taper about 20% per run;
  - a knuckle bulge at each joint (same `prism_group` as the spar);
  - a pointed tip cap instead of a chopped end;
  - a thumb claw on the wrist.
- **Damage:**
  - deep concave scallops between the tips (sag 0.28-0.3);
  - tears that run along the fingers' direction: lanes measured around the hub, 5 units wide, about 24% of them cut. Lanes measured along the edge make vertical "comb teeth".
  - at most 2 holes, on the outer bays only.
- **Value hierarchy:** membranes stay dark (indigo at the spars, lighter only near the torn tips). The brightest, most saturated pixels belong to the head crystals, the eyes and the chest core, under 5% of the area.

#### 8.4d Gap-free wing membrane geometry: Shared-Wrist Nested Fans & Full-Chord Coverage (`G10`)
A fatal wing error is attaching narrow rectangular strip planes (`chord < fingertip_gap`) independently along each finger spar: as the fingers fan `20–30°` apart, rectangular strips split open into disconnected paddles with **giant empty V-shaped sky gaps** between fingers (`coverage = 0.40x`). Eliminate inter-finger V-gaps permanently using one of two master Blockbench architectures (`Vijay Dragon Master` `5o_Ijoq_dGQ`, `super fuzzygoat` `77uIydqJVrw`, `Vijay Sphinx` `_h6xjye56Sk`):
1. **Method A — Full-Chord Nested Fan Overlap (Bat / Wyvern / Elder Dragon Wings, `G10`):**
   - Either nest sequential finger bones from the shared wrist hub (`wing_f1 -> wing_f2 -> wing_f3 -> wing_f4`, `5o_Ijoq_dGQ`) OR space roots tightly along the palm while sizing **every bay plane's chord width to `>= 1.08–1.18x` the 3D world distance between adjacent fingertips (`tip_i` to `tip_{i+1}`)**!
   - Because each plane reaches completely across to overlap `1–2 u` past the trailing neighbour spar (`f_{i+1}`), the raw untextured planes form a **100% solid, overlapping folding-fan canopy** that never opens V-gaps at rest or mid-flap.
   - Stagger adjacent bay planes with distinct `layer: 1, 2, 3, 4` (or micro-`inflate` / `0.01`-thickness double-sided sheets per `77uIydqJVrw` `13:28`) and list crossed neighbour spars in `blend_with` so `SAT` and depth-sorting stay `100%` flicker-free.
   - Carve the exact `[root_i -> tip_i -> concave_scallop_arc(tip_i, tip_{i+1}) -> root_{i+1}]` bay wedge in `100%` hard alpha during `T1` (`10.3b`), erasing the overhanging rectangular corner past `f_{i+1}` so the finished membrane lands **flush along both bounding finger spars**!
2. **Method B — Multi-Tier Shingled Feather-Blade Canopy (Avian / Angelic / Paladin Wings, `_h6xjye56Sk`):**
   - Layer `3` overlapping rows under the wing arm and hand: short dorsal **lesser/median coverts** (chamfered plates or small planes), mid-length **secondary blades**, and long radiating **primary flight-feather blades** (`7–12` rotated overlapping planes per wing) fanning continuously with zero unintentional gaps.
3. **Unified Coplanar + Shared-Crease Inner Wing Membrane & `wing_l_f4` Rigid Attachment Law (`v25` Anti-Split & Anti-Double-Layer Law):**
   - **NEVER stack 3 detached overlapping inner-wing sheets at arbitrary angles** (e.g., `wing_inner_sail` at `-26°` on `wing_humerus` + `wing_fore_patagium` at `-36°` on `wing_radius` + `wing_bay_4` at `-42°` on `wing_l_f4`), and **NEVER** parent `wing_l_f4` onto a freely yawing `wing_l_wrist` separately from the wing arm! Zooming `5×` onto both sides of the inner wing near the torso exposes a hideous floating double-layer staircase/gap (`4–24u` perpendicular offset) and splits `f4` out during animation!
   - **Exact 3D Coplanar + Shared-Crease Fix (Zero Detached Second Layer, `0.000u` Seam):**
     1. **Attach `wing_l_f4` / `wing_r_f4` directly onto `wing_l_fore` (`parent_box = wing_radius`)** at the distal carpal hub and keep `wing_l_f4` rigidly locked (`[0, 0, 0]` relative to `wing_l_fore` across animations) so `f4_spar` + `wing_radius` + `wing_bay_4` move as ONE unsplittable unit!
     2. **DELETE redundant `wing_fore_patagium` completely!** Place a single unified **`plane wing_bay_4` directly on `wing_l_fore` (`parent_box = wing_radius`, `layer: 0`)** at the exact analytical coplanar angle `rx_bay4 = atan2(f4_tip_local.y, -f4_tip_local.z)` so it passes through **BOTH `wing_radius` AND `f4_spar` to `0.000u` perpendicular error**!
     3. **Edge-lock `plane wing_inner_sail` on `wing_l_upper` (`parent_box = wing_humerus`, `layer: 0`)** at the exact angle `rx_inner` containing the shared elbow trailing point `P_el_trail`, yielding a gentle natural `~10°` dihedral hinge along the **exact 3D plane-plane intersection crease `L_crease = p0 + t · (n_bay4 × n_inner)` (`err = 0.000u`)**!
     4. **Strict Non-Overlapping Hard-Alpha Crease Partition + Continuous 3D World-Space Shading (`T12`):** Cut `wing_bay_4` strictly on the outer side of `L_crease` (`s_crease >= -0.45u`) and `wing_inner_sail` strictly on the inner side (`s_crease_in >= -0.45u`, tucking its inner ribcage edge `P_sh -> P_flank` cleanly inside `chest_core`), and shade **BOTH planes with ONE continuous 3D world-space shader** (3D distance to leading bones `wing_humerus`/`wing_radius`/`f4_spar` + flank + continuous 3D world-space pleats + velvety trailing hem) so color, value, and pleats flow `100%` unbroken across the elbow crease!
     5. **Animation Hinge Discipline:** Keep `wing_l_fore` roll bends moderate (`<= 14–16°` during cloak-folded `idle_noble`/`walk_patrol`, shifting downward cloak carriage cleanly onto `wing_l_upper`) and keep `wing_l_f3` gentle (`[0, -2, 0]`) so `wing_bay_3` NEVER overshoots behind `f4_spar` and the elbow crease stays sealed across every moving pose! Verify with mandatory `5×` zoomed renders on **BOTH sides** (Front-3/4, Rear-3/4, Underside/Ventral, Dorsal Top-Down, Edge-On Cross-Section, and 6-pose moving zoom)!

### 8.5 Tail
- 5-7 segments tapering to 20-30% of the base thickness. The base is nearly as thick as the hips.
- Continue the gesture with pitch, and lift or curl the tip.
- Carry the dorsal row down the tail, shrinking.
- End with **one** signature feature, built as:
  - a spade: one flat box rotated 45 degrees;
  - a fluke: one horizontal membrane polygon;
  - a tuft: two crossed cut-out planes;
  - a club: 1-2 boxes with single-box studs;
  - a radiating aquatic caudal fan (Faentum Leviathan): 3-5 graded flat blade boxes or framed alpha rays fanning symmetrically from the tail tip, optionally carrying the boss's focal glow accent.
- **Length:** about one body length or more (nose to hip x 1.0-1.3), in 6-8 segments, with hidden joint-overlap skin inside parent sockets so deep curls never gap.
- **Underside:** the underside column of plates continues to the tip.
- **Dorsal row:** graded single-box spikes on the first three segments.
- **Rest pose:** lift and curl with pitch only. A sideways sweep belongs in a display clip.

### 8.5b Body detail (from the anchors)
- **Underside column:** throat, chest, belly and tail plates form one continuous column in the underside colour. Each plate is wide, sunk about 1 unit, and alternates slightly in width.
- **Dorsal row:** single boxes, graded largest at the withers, all leaning back about 25 degrees (row lean). Use spikes, not shards.
- **Legs:** knee, calf and heel boxes; toes in the underside colour; claws as separate boxes. Paint at most one darker band row in five.
- **Cheeks:** cheekbone ridge, jaw-hinge mass and one signature feature (6.2b).

### 8.6 Arms and hands
- **Chain:** shoulder, upper arm, forearm, palm, then three fingers and a thumb. Each finger is one thick block plus a block claw (8.3). A thumb is a short thick digit, not a stick.
- **Rest pose:** elbows bent and fingers half curled. A straight arm with flat fingers looks like a toy.
- **Hand recipe** (round 8, owner approved "hands and finger looks good"):
  - a palm narrower than the wrist above it, longer than it is thick;
  - 3 fingers, each 2 bones (knuckle plus distal): 3.5 then 3 thick, 8-9 then 6-7 long;
  - one block claw per finger, 0.8x the digit, about 6 long, hooked;
  - splay the fingers by roll (about 9 degrees) and curl them about 28 then 20 degrees toward the palm;
  - an opposed thumb with its own claw.
  - The previous brick palm with stub fingers and pale cube claws read as marshmallow toes.
- **Check the curl direction numerically,** not by sign rules: print the world centres of the palm, the finger segments and the claw. The claws must end lower than the knuckles. In round 8, a negative pitch curled the fingers up toward the back of the hand.
- **Elbow** (owner: "consider adding elbow part"):
  - an elbow cap (about 11 x 9 x 9) on the outside of the bend, where the upper arm and forearm meet, blended into the upper arm;
  - one bony spur pointing along the outside of the bend (down and back for a forward-reaching forearm);
  - the spur is painted like a claw, dark at the root and pale at the point.

  Find the outside of the bend from the two segment directions; never guess from local axes.

### 8.7 Machines, vehicles, buildings, props and scenes (Anchor 04 Pantry)
- **Scale first:** build at real Minecraft scale (16 px per block) and measure before detailing. Keep tiny handheld props, clutter and surrounding architecture on one shared texel density so everything belongs to the same world.
  - Guides: a handheld tool is 1-1.4 blocks; a lantern is 0.6-1; a cart is about 1.5 long; a crate is 1; a jam jar or bottle is 0.25-0.45.
  - The first test pick was 2 blocks tall, as tall as the player.
- **Volumes:** start from two or three big volumes in uniform graybox, then chamfer the corners that show in silhouette with rotated boxes.
- **Open-frame cavities and recessed interior depth (Anchor 04 Pantry):**
  - Build crates and baskets with real corner posts and gapped horizontal slats revealing chunky produce or bottles packed inside, rather than solid painted cubes.
  - Build shelves, cupboards and shop windows as genuine recessed cavities with foreground, midground and interior prop layering for real parallax.
- **Clustered props, produce and lived-in organic variation (Anchor 04 Pantry, 3.3):**
  - Place each prop's pivot at its **ground-contact base or hanging stem** before rotating.
  - Apply deliberate small organic tilts (`5-15°`), splay and staggered height variation on bundled produce (carrots, leeks, bread sticks), hanging chili clusters, leaning bottles and stacked crates so repeated items feel handcrafted and inhabited rather than sterile grid copies.
  - Keep structural tables, shelves, beams and walls square and level; only the loose contents and hanging accents lean.
- **Build in segments, end to end.** A handle is pommel, grip, shaft, band and socket stacked; never a stick through sleeves.
- **Round parts** are true octagons from four bands (3.2).
- **Glass** is a clear plane, framed by posts.
- **Light sources** are small, very bright warm emissive glow blocks (lantern cores, candles, embers) framed in dark posts against cool ambient stone/wood shadows (Anchor 04). A flame is two boxes turned 45 degrees (a body and a narrower tip).
- **One accent per prop,** placed where the eye should go: an ember, a crystal, a corked coloured jar label, a striped table runner or a flame.
- **Greebles:** every one must look like it does something: hinge, vent, lamp, rail, pipe.
- **Joints and wheels:**
  - joints are octagon hubs with a painted bolt, larger than the limb they join;
  - wheels are an octagon tyre with tread pixels, a rim and a hub;
  - a piston is a rod sliding in a thicker sleeve;
  - a hose is 4-7 seated segments, not dozens.
- **Decals tell the story:** numbers, hazard stripes, patched panels.
- **Weathering follows gravity and use:** streaks under seams, wear on edges, a top-surface layer (snow, moss, dust).
  - Keep streaks short (2-4 px) and muted. Long saturated streaks read as painted stripes.
- **Paint hard surfaces crisply** (round-6 references):
  - Flat base fields, with a 1 px lit top edge and a 1 px dark bottom edge.
  - Panel lines and vent slots as rows of short dark rectangles.
  - A rivet is a lit pixel over a dark pixel.
  - Brushed metal gets a few long dashes, no speckle.
  - Contrast lives on edges, seams and rivets, so hard surfaces measure 6-14% large jumps, against 4-7% on hides. Keep the fields themselves flat.
- **Paint wood in the vanilla-plank language:**
  - grain dashes along the length;
  - board seams every 4 px;
  - darker corner posts;
  - rings on cut ends, in mid wood tones.
- **Other prop materials:**
  - leather wraps are horizontal turns with a dark seam;
  - stone and ore are 2-4 px chips with 2 px ore clusters.
- **Watch for faces.** Two light squares side by side read as eyes: log ends, nostrils, vents, glints on front and back panes. Darken or break one of them.
- **Motion:**
  - wheels go on their own bones, with `bio_joint: wheel_axle` (`±360°` pitch spin, `0°` camber/toe wobble) inside optional `bio_joint: steered_wheel_knuckle` (`±35°` Ackermann steering yaw);
  - turrets, cranes, cannons, rotors, doors, and hydraulic pistons use `turret_yaw`, `cannon_elevation`, `rotor_propeller`, `hatch_door`, and `hydraulic_hinge` (`v28` mechanical presets — automatically exempt from `Sim(3)` biological muscle bulge!);
  - hinged handles get a sway;
  - a held tool needs no clip.

#### 8.7b Universal Large Vehicles, Bio-Mechanical Vessels, Battle Mechs & Dioramas (`v28`, Wackyblocks & KayaTa Distillation)
When building trucks (`WbyCbA1c8BM`), tanks (`cnwSSjQDA1A`), cargo airships/vessels (`2pc-iXaSjYM`), battle robots (`rycGulcQ-Lo`), or architectural dioramas (`NB_uHkB1dz8`), follow the **7-Stage Hard-Surface & Hybrid Construction Order** (`references/youtube_master_distillation.md` Part 2B):
1. **Chassis / Hull Frame & Wheelbase Ruler First:** Establish axle height (`wheel_radius` above `y = 0.00u`) and wheelbase-to-track ratio (`~1.6–2.2 : 1` trucks, `~1.2–1.5 : 1` tanks) or stepped hull bulkheads before adding trim.
2. **Recessed Wheel Wells & Stepped `45°` Fender Arches:** Carve real negative space around wheels/tracks with trapezoidal fender flares + dark inner wheel-well liners so tires never stab through flat body slabs.
3. **Octagonal Wheels & Servo Drums with Inset Hubs:** Build tires and mecha knee/shoulder joints as chunky octagonal/stepped drums with `0.5–1.0u` recessed hubcaps/bolts (`steer_fl -> wheel_fl`).
4. **Sloped Cab Pillars & Recessed `normal: z/x` Glass Planes:** Tilt windshields/A-pillars (`22.5–35°`) and recess tinted glass planes `0.5u` inside structural timber/steel pillar frames.
5. **Offset External Mecha Armor & Functional Hinges:** Seat battle-mech armor pauldrons/greaves `0.5–1.0u` proud of dark inner frames (`rycGulcQ-Lo`), and pivot tailgates, ramps, turrets, and cannons on true mechanical hinge axes (`hatch_door`, `turret_yaw`, `cannon_elevation`).
6. **Purposeful Asymmetric Greebles:** Exhaust stacks, side mirrors, winches, cranes, portholes, hazard-striped bumpers, and restrained `<= 4%` headlights/lanterns.
7. **Crisp Industrial Panel & Gravity-Weathering Paint:** Flat clean cab/hull color fields, `1 px` bright top bevels, `1 px` dark panel seams + rivet pairs, chipped gunmetal corners, downward rust/salt streaks under portholes, and upward-fading lower-chassis mud/dust splash!

### 8.8 Cut-out planes
- Paint the edge shape in alpha with a crisp **hard-edged (100% opacity, 0% softness)** stepped pixel outline; no straight plane borders and no blurry semi-transparent alpha fringes should show.
- **Pivot at the attachment root or stem:** place the plane's pivot where it joins the spar, cheek, neck, vine branch or clothesline so fanning or swaying rotates cleanly around the contact edge without detaching or clipping.
- Planes are always double-sided (or single-face alpha-carved panes when intentional), and never coplanar with a box face or another plane (use `layer`).
- **Fringes:** run along the back of the neck, the jaw line, the elbows and the tail, with spikes 1-3 px wide pointing away from the direction of travel.
- **Hanging things** (moss, cloth, roots, icicles): drips of uneven length.
- **Framed fins** (head frills, ear fins, hero dorsal sails):
  - 2-3 ray boxes in the hard-part colour, rooted apart on a small root block, plus one cut-out membrane plane between them on the same bone.
  - Cut the membrane in paint to the polygon through the ray roots and tips, and scallop it between the tips.
  - Paint it in the wing-membrane colour, lighter toward the edge.
  - Keep the membrane's rectangle clear of other bones, because the audit tests the rectangle.
- **Single-plane fins** (sails along a long body): paint rays every 4 px leaning back, a 1 px lit outer edge, and darker root rows.

### 8.9 Silhouette-defining micro-traits catalogue (complex & boss tiers)
Small anatomical and armour details separate handcrafted showcase bosses from generic blockouts when placed purposefully at silhouette inflection points:
- **Compound-curved multi-segment horns (`3–6` segments, `cQOaep_2y6o`, `YDYIFgTZKRI`):** combine progressive `pitch + yaw + roll` across seated child bones emerging directly out of supraorbital brow bosses or zygomatic cheek shields (`brow_boss -> horn_1 -> horn_2 -> horn_3 -> sharp_tip`), never straight cylinders poked vertically into a flat skull roof.
- **Staggered heterodont dentition:** sculpt prominent curved canine/vampire fangs (`2.5–4 u` tall) paired with smaller premolar/carnassial blocks tucked flush inside the maxilla rim, interlocking against painted/sculpted lower teeth (`6.5`).
- **Joint spurs & patellar/olecranon shields:** add backward-swept elbow spurs, winged ankle plumes/calcaneal spurs, angled knee-cap shields, and 2-to-3 segment scythe wing-thumb claws (`metacarpal -> proximal -> curved claw`).
- **Articulated collars, host-ribs & multi-bone mantles (`0WTvd2KNRqs`):** build high flared collar lapels, sternum rib-cages, pectoral gorgets, or 3-bone articulated shoulder capelets on dedicated child bones so they breathe and flare during roars.
- **Anti-Toy Tail, Laminated Plate & Collared Accessory Carving (`v33`, `G13`, `references/high_res_focal_carving_and_visual_primitives.md` §4):**
  - **Multi-Plane Tapered Tails (zero stacked plain bricks!):** Sculpt tapered tails/appendages from chamfered cores + recessed flank troughs (fur/flesh/cable channels) + **staggered overlapping dorsal & ventral shingle armor scutes (`>= 2` progressively pitched plates per segment)** instead of single long flat planks.
  - **2-to-3 Step Tapered Blades, Fins & Spurs:** End every tail weapon, dorsal shocker, ear pinna, horn, forearm spur, or armor fin in `2–3` progressively narrowing & thinning beveled segments (`base -> mid -> sharp point`), NEVER blunt square sticks.
  - **Physical Mounting Collars & Brackets on Attached Accessories:** Seat dorsal plates, shoulder horns, pauldrons, shields, straps, cables, pouches, and weapon mounts inside contrasting recessed collar bezels/cuffs (`0.5–1.0u` proud dark-slate/metal/leather seat) painted with root occlusion and `1 px` top bevels.
- **Articulated terminal tail weapons:** 3-pronged splaying candelabrum tridents, broad diamond halberd lances, layered overlapping pine-cone dragon-blade clubs, bulbous barbed telsons, or 5-ray caudal fans where individual blades taper sharply and hinge cleanly during attacks.

## 9. Creature classes, scale and categories

### 9.1 Scale tiers
| Subject | Treatment | Element guide (floor – ceiling) | Typical bone count |
|---|---|---|---|
| Complex / Epic Elder Dragons & centerpiece bosses | `epic_boss`: 3-layer sculpted limbs, multi-jointed 4–5 digit talons, bespoke visor/horns, gap-free nested-fan wings (`G8–G10`) | **240 – 550** *(up to ~800+ on multi-headed/feathered titans)* | **85 – 160+** |
| Hero large monsters (wyverns, leviathans, brutes) | `hero`: muscular tapers, articulated jaws/digits/fins | **150 – 320** | **55 – 110** |
| Player-sized monsters and mounts | `standard` to `hero` | **80 – 200** | **30 – 65** |
| Smaller than the player (fish, birds, insects, critters) | `small` creature treatment (9.3) | **15 – 35** | **8 – 22** |
| Gathering points, props and diorama scenes | `prop` / scene | **8 – 60** per prop (`90 – 250` full scene) | **1 – 25** |

**Texel density (`v33` Adaptive High-Craft Matrix — `references/high_res_focal_carving_and_visual_primitives.md` §2):**
- **Vanilla-lite mobs, critters & simple world props:** `16 px/block` (`1 px/u`) globally, with **`uv_scale: 2` (`32 px/block`) on focal eyes/dials** so faces stay crisp beside vanilla blocks.
- **Studio-master creatures, animals, bosses, humanoids, armored mechs & detailed vehicles (recommended `v33`):** **`32 px/block` (`2 px/u`) globally + `uv_scale: 2` (`64 px/block` = `4 px/u`) on focal eyes/optical lenses/emblems**, packing cleanly into `1024×512`–`2048×2048` Per-Face UV atlases! Paint `32x` with organized `2 px` stepped pixel clusters, crisp `1–2 px` bevel highlights, deep `2 px` plate-overlap crevices, and directional striations — NEVER blurry airbrushed noise!
- At 16 px per block, plan faces on the pixel grid: whole-unit face sizes keep pixels whole, and feature ranges fall on whole units.
- On multi-limbed creatures (spiders, centipedes, repeated digits), share UV occupancy across identical mirrored limbs so shading stays consistent and atlas resolution isn't wasted.
- Small faces get their readability from design, not pixels: big lit irises in a dark band.
- Use 32 only for a showcase piece the owner asks for. Keep one density per asset.

**Community scale anchors (ceilings and low-cube boss proof points):** a popular animated community fire dragon is ~415 cubes, 8 blocks tall, at 16 px per block — a practical upper ceiling for hero animation, never a target. Meanwhile, Longridr's 8-block dragon boss reads monumental at **132 cubes**, and Faentum's colossal armored Deep-Sea Leviathan achieves epic boss silhouette in **56 cubes** by letting bold hull segmentation, alpha fin silhouettes and restrained focal glow do the work.

### 9.2 Monster Hunter classes as body plans
| Class | Body plan and construction notes | Examples |
|---|---|---|
| Flying wyvern | 2 legs, wing-arms; deep chest, small hips; wings 1.3-1.5x nose-to-tail span; tail tip is the signature | Rathalos, Rathian |
| Elder dragon | often six limbs (4 legs + wings) or a unique plan; horns that sweep into the dorsal line; one strong signature | Fatalis, Kushala Daora, Velkhana, Valstrax, Alatreon, Kirin |
| Fanged wyvern | wolf-like quadruped; heavy forequarters; short deep muzzle (6.1); fur as cut-out rows; one glow accent | Zinogre |
| Leviathan | long body in an S made with pitch, low limbs, fins and frills as partitioned planes | Lagiacrus, Mizutsune |
| Brute wyvern | biped, oversized head and jaw, thick counterweight tail | Anjanath, Deviljho |
| Bird wyvern | light biped, beak or crest; often player-sized | Yian Kut-Ku |
| Fanged beast | mammal-like, fur rows, expressive face | Rajang |
| Temnoceran | spider-like; see 9.5 | Nerscylla |
| Cephalopod | tentacled; see 9.5 | Nu Udra, Xu Wu |

Check each game's current roster on the web, and file each monster under its class from an official source.

These plans are a vocabulary for analysing references, not a menu for original monsters. An original may mix plans (a sprawled elder with plume wings) or invent one (5.0).

### 9.3 Environment creatures
- **Smaller than the player** (fish, birds, small herbivores, insects, critters): the small-creature treatment. Simplify hard without losing the animal's feel.
  - **Keep 2-3 identity traits and push them 1.3-1.5x.** Drop everything else to paint.
    - Jumping mouse: ears, hind legs and long feet, and a tufted tail.
    - Spider: eight splayed legs, a big striped abdomen behind a waist, and the eye band.
  - **15-30 elements, all chunky.**
    - No box is under 2 units in two dimensions. Thin things (ears, tufts, fins) are cut-out planes.
    - The head is one box, plus at most a muzzle and ears. No chamfers or cheek boxes.
    - The jaw is painted on, unless a clip opens it.
  - **Limbs:**
    - at least a third of the body's width;
    - each foot is one box in the hide colour, with pale toes or soles painted on. Separate white sole slabs read as shoes;
    - thin legs read as toys.
  - **Proportions:** the head is 40-60% of the body; a big head reads as young or cute. Multi-mass bodies (spiders, ants) get a narrow waist box, so the masses read separately.
  - **The eye is the main read.**
    - Mammals: a 3 x 3 px dark eye high on the side, with a 1 px highlight, wrapping 1 px onto the face.
    - Arthropods and many-eyed creatures: a dark band across the face, two lit main eyes of 2 x 2 px with a hot core, and 1 px side eyes. The band wraps the corner.
  - **Ears:** oval cut-out planes turned 30-45 degrees to face forward and out. Edge-on ears read as spikes from the front.
  - **Pose:** low and grounded.
    - Spider legs are arches: the femur rises 25-30 degrees, and the tibia drops a little outside the knee.
    - Fan the legs by yaw: about +34, +12, -12 and -30 degrees from front to back.
    - Make the femur thin only front-to-back, so neighbouring roots sit at least their own thickness apart.
  - **Paint:** the same 16 px per block.
    - Fewer, bigger pattern shapes: stripes or chevrons 1-2 px wide, socks, countershading.
    - No scattered dots; they read as dirt.
    - 2-3 materials plus one glow.
  - **Motion:** snappy and in place.
    - A scuttle swings each femur by yaw, 5 degrees or less. Adjacent legs move in opposite phase.
    - On mirrored yaw bones, the same phase makes a left-right pair alternate.
    - Raise the rest pose a fraction of a pixel if a clip dips the feet below the ground.
  - **Scale:** keep it readable before accurate. Small creatures may be built larger than life.
  - **Exemplars:**
    - Dunehop v2: 22 elements, 1.1 boxes per bone.
    - Pebble Weaver: 21 elements, 1.2 boxes per bone, with idle and scuttle clips.
- **Small objects and props** (crates, baskets, fruit, tools):
  - Build at block scale. Planks are boxes at least 2 units thick, with corner posts.
  - Contents are one box each, with a soft painted highlight.
  - Recognisable before accurate.
- **Herbivores larger than the player** follow the standard tier.

### 9.4 Gathering points
- **Rocks and ore outcrops:** 3-7 large rotated boxes, never voxel boulders.
- **Crystals:** 3-7 single-box prisms of mixed height.
- **Plants and mushrooms:** cut-out planes or chunky caps (one box each).
- **Bones:** a few large box bones.
- **Marker:** a small focal glow or contrasting accent that shows the point is interactable.

### 9.4b Sea creatures (leviathans, sea serpents)
- **Body:** a pitch-only S-wave in the rest pose. The head rises and the belly touches the ground at its lowest segment. The neck, body and tail chains each follow the remap limits.
- **Dorsal sails:** one fin plane per segment, graded.
  - Alternate a centred fin with a mirrored ±0.25 pair, so neighbours never share a plane.
  - Centred-only fins z-fight at the joints. Alternating ±0.25 offsets break mirror parity.
- **Frill fins:** on their own bones. On the left side, positive yaw sweeps backward and outward.
- **Flippers:** an arm box plus a flat paddle box. Set yaw limits from the joint test, because paddles swing into the body.
- **Swim clip:** a pitch wave travelling down the chain, adding about 60 degrees of phase per segment. Flipper strokes use mirrored-phase roll (4).

### 9.5 Unsettling creatures (spiders, cephalopods, parasites): faithful, not repulsive
- **Keep the identity:** body plan, limb count, signature shapes.
- **Clean the presentation:**
  - Limbs and tentacles are thicker, tapered 3-4 segment chains with a steady rhythm. No tangles, no hair-thin legs.
  - A slightly larger body and head relative to the legs.
  - Two clear main eyes with highlights; any extra eyes small and painted.
  - Mouthparts simplified into 2-3 shapes; no gore, no detailed mouth interior.
  - A graphic, saturated palette instead of flesh tones; no slime, veins or wet noise.
- **The aim:** not cute, not disgusting. Imposing and readable.

## 10. Texture: unlit macro blocking, 3D seam flow, anti-plastic craft & hue shifting

### 10.1 Workflow
1. **Unlit viewport + `1 px` padded UVs:** disable 3D viewport shading while painting so baked ramps are honest and unmasked by preview shadows, and pack UV islands with `1 px` bleed padding to prevent mip/screenshot edge seams.
2. **Flat base colour blocking & 25% Zoom Macro Value-Block Gate (T1):** flood-fill whole elements per material from the palette (in Blockbench: Paint Bucket in Cube/Element mode first, then Face mode for undersides; turn mirror painting OFF for narrow center noses/snouts or organic belly variation), cut in the underside scute column, underjaw countershade, horn distal ramp, wing membrane radial/edge gradient and hot eye irises, then **zoom out to 25%**. Do not paint micro-strokes until macro colour-blocking reads unmistakably from afar. If flat blocking exposes a local proportion or clearance flaw, fix the box now (5.3).
3. **Banded directional & contact lighting (Minecraft lights models only lightly, so shading lives in the texture):**
   - top-left / top-down lighting: top and front faces `0.5-1` step brighter, bottom and back `0.5-1` step darker;
   - selective `1 px` light top/leading ridge and `1 px` dark contact edge;
   - a contact shadow half a step darker within about `1.5u` of every joint root (under chin/neck, armpits, inner thighs, under tail and beneath overhanging armour plates), plus warm uplight gradients on faces looking toward glowing cores/crystals.
4. **Cross-seam 3D directional strokes & value sculpting (10.2):** with Mirror Painting enabled, paint longitudinal `1 px x 3-5 px` half-step strokes continuously across adjacent `neck`, `torso` and `tail` segments in 3D view so stripes, countershading and scale rows flow uninterrupted across joints instead of resetting per box. Choose the right painting mode per surface (crisp 1 px palette-ramp vs. soft-layered directional blending for membranes), carve optical volume, add roof-tile overlap cast shadows, and soften curved pixel transitions with manual anti-aliasing.
5. **Features & boss-scale focal glow restraint:** eyes, interlocking mouth/teeth, markings, and strictly localized focal glow (`<= 4%` surface area, `1-3 px` bleed). On colossal bosses (Faentum, Kundzio), keep large armor/body fields calm and concentrate emissive contrast on 1-2 unmistakable gameplay-distance read zones (eyes + crown/core or tail fan/wingtips).
6. **T2 calm & anti-plastic pass:** prune visual clutter until `>= 60%` of each face sits at its base tone, break any mechanical concentric outlines, and audit interior patches (`mean step 4-9`, `<= 7%` jumps `> 25`).
7. **Animation-safe hidden-face check (mandatory before finishing T2):** hide overlapping limbs/wings/plates AND rotate every animated joint (jaw open, legs swung, wings lifted, tail curled) to inspect concealed inner faces. Paint all inner thighs, under-chins, armpits and mouth interiors with deep occlusion shadows so flat, bright or untextured pixels never flash mid-clip.
8. **Final squint test at 25% size.**

### 10.1b Flexible Semantic UV Partitioning v3, Declarative `.texdsl` Texture Language & Part-Aware `synccraft.texgraph` v8 (`v45.0`)
Never ship scattered low-utilization (`15–35%`) square atlases, uniform fur/scale texture blankets that paint over eyes/noses/beaks/teeth/claws, rock-hard rigid mechanical scale grids, dirty/muddy albedo shading, noisy static, `0.0u` flickering/floating single-plane wing sheets, or missing PBR material maps. SyncCraft `v45.0` provides an **industry-level, open-source, pure-algorithmic Substance-Designer + Pixel-Composer-parity procedural texture engine** (`125` typed nodes, `22` reusable PBR MaterialBus subgraphs, `31` multi-domain studio presets, declarative **`.texdsl`** texture programming language, zero ML/transformer dependencies, future-UI-ready `.texgraph.json` v3.1 / `engine_version: "8.0"` schema; see `references/texgraph_master_texturing_study.md`):

1. **Declarative `.texdsl` Texture Programming Language (`synccraft/texgraph/texdsl.py` — ZERO 300-Line Python Boilerplate!):**
   - Program complete multi-pass, multi-material, animated PBR textures directly in 30–70 line `.texdsl` files (`@texture`, `@palette` with `@name` & `@bank.<palette>`, `@pass`, `select(pattern=..., zone=..., faces=...)`, `use <subgraph>(...)`, boolean/arithmetic mask expressions `a | b`, `a & !c`, `@animate { drive node.param = sine|wiggler|keyframes(...) }`, `@expose`, `@output material(...)`).
   - Full lossless round-trip compiler & decompiler (`parse_texdsl`, `load_texdsl`, `export_texdsl`, `TexGraph.from_texdsl`, `TexGraph.to_texdsl`, `TexGraph.load`), supported natively across EVERY CLI command (`synccraft texgraph render/bake/validate/qa my_entity.texdsl --bbmodel my_entity.bbmodel`).

2. **Part-Aware Semantic Entity Detailing & Character/Orc Multi-Material Kit (`generators_character_materials.py` — Anti-Blanket Law!):**
   - **NEVER cover eyes, orbital sockets, noses, beaks, mouths, tongues, teeth/tusks, claws/hooves/horns, inner ears, fins, or glowing runes with a uniform body fur/scale shader!**
   - Always finish multi-material 3D entities with **`paint_entity_eyes_and_parts(...)`** (protected LAST with automatic front/lateral/both eye-face selection `eye_facing="auto"`, dark recessed orbital sockets on non-facing sides, crisp studio eyes OR non-standard gradient eyes, smooth **animated `eye_blink`** & **`eye_glow`/`rune_glow`**, glossy wet noses, dark desaturated plum `#2A1422` mouth interiors, coral tongues, dark-rooted warm ivory teeth/tusks, keratin beaks/claws/hooves/horns, and ray-striated backlit fins!).
   - Dedicated Character & Gear PBR MaterialBus nodes (with distinct roughness & hue-shift separation): **`orc_flesh_skin`** (olive/emerald hue-shifted skin + warm subcutaneous brow/knuckle blush + pale scars + optional tribal war paint), **`stitched_leather`** (warm umber/tan with concentric stepped rim bevels & perimeter pixel stitch dots, roughness `~0.64`), **`forged_battle_metal`** (cool charcoal-indigo `#1B1726` shadows, crisp 1 px metallic bevels, scratches, corner rivets & cavity rust, metallic `~0.88`, roughness `~0.25`), **`woven_battle_cloth`** (soft vertical drape folds + embroidered hem trim or twisted rope, roughness `~0.86`), and **`keratin_horn_bone`** (dark root collar stepping into striated warm ivory distal tips).

3. **Industry-Grade DAG Execution Core, Validation, Caching, Animated `.mcmeta` Hold/Ping-Pong & Batch Production (`core.py`, `pixel_art_ops.py`, `quality.py`, `production.py`):**
   - **True Topological Sort & Static DAG Validator (`topological_order()`, `validate_graph()`):** Kahn's algorithm executes recipes in any node order with explicit `GraphCycleError` cycle detection, missing-node / unknown-port checking, and unreachable-node warnings (`synccraft texgraph validate`).
   - **Content-Addressed Incremental Memoization (`_content_cache`):** Hashes `(op, effective_params, input_hashes, resolution, tileable, mesh_id, phase)` per node so tweaking downstream grading/color parameters or rendering multi-frame flipbooks automatically gets `0.0 ms` cache hits (`4×–8×` speedup!).
   - **Animated `.mcmeta` Custom Frame Schedules & Synchronized `_e` Emissive Strips:** `export_flipbook` and `render_pbr_flipbook` natively support hold-frame schedules (`[{"index": 0, "time": 40}, 1, 2, 3, 2, 1]` for natural eye blinks / ore glints), ping-pong loops (`[0, 1, 2, 3, 2, 1]`), per-frame GIF timing, and synchronized `_e_flipbook.png` + `_e_flipbook.png.mcmeta` emissive-only strips!
   - **Multi-Domain Automated QA & Batch Production (`evaluate_texture_quality`, `run_batch`, `pack_sprite_atlas`, `PALETTE_BANK`):** Scores Blocks, Items, and Entities with `18` curated cool-shadow-to-warm-highlight hue-shifted palettes.

4. **Flexible Non-Square Interlocking Semantic UV Packer v3 + `0.5u` Membrane Rim Culling (`synccraft.shading.uv_packer`):**
   - Classifies every element into anatomical zones (`head`, `neck`, `torso`, `wings`, `forelimbs`, `hindlimbs`, `tail`, `emissive`, `accents`) and packs via **Zone-Anchor Gravity MaxRects** onto **flexible `16 px`-aligned non-square canvases (`canvas_mode="tight"`)**, achieving **`93%–97%+` padded utilization** with Adaptive Hybrid L/R mirroring.
   - Automatically culls the 4 thin `<= 0.55u` side rim faces (`north/south/east/west`) on dual-sided solid wing membrane slabs (`bay|sail|membrane|patag|lobe|fin|web`) to `[0, 0, 0, 0]` while packing full-resolution distinct `up` (Dorsal Sheet) and `down` (Ventral Sheet) UV islands.

5. **Full Substance + Pixel Composer Node Suite (`125` Nodes) Across Blocks, Items & Entities (`Generate -> Iterate -> Simulate -> Animate`):**
   - **Substance & Pixel Composer Core Operators (`engine_ops.py`):** Analytical 2D **`sdf_shape`** (`20` primitives & item silhouettes) + **`sdf_boolean`**, seamless **`tile_sampler`**, toroidal **`flood_fill`**, **`slope_blur`**, **`histogram_scan` / `histogram_select` / `auto_levels`**, **`curvature` & `edge_wear`**, **`displace_uv`**, **`normal_light_2d`**, and Brandon James Greer **`directional_selout`**.
   - **World Blocks, Items, Animals, Characters & Fantasy Creatures (`generators_bio_materials.py`, `generators_character_materials.py`, `generators_painterly.py`, `presets_library.py`):** `cobblestone_masonry`, `bark_and_timber`, `crystalline_ore_vein`, `caustic_water`, `organic_fur_pelt`, `layered_plumage`, `chitin_carapace`, `aquatic_scales_skin`, `elemental_golem_surface`, `orc_flesh_skin`, `stitched_leather`, `forged_battle_metal`, `woven_battle_cloth`, `keratin_horn_bone`, `paint_entity_eyes_and_parts`, `anatomical_muscle_scales`, and Dual-Layer `colossal_dragon_membrane`.
   - **Iterate, Simulate & Animate:** `iterate_inline`, `feedback_loop`, `particle_sim`, `fluid_smoke_sim`, `anim_driver` + `g.bind_driver(...)` (`sine`, `wiggler`, `keyframes`), explicit sequential passes (`with g.pass_(...)`), and unified world-space atmospheric `material_grade`.

6. **Zero-Blink `0.5u` Dual-Layer Piece-by-Piece Attached Wing Slabs (Malzeno & Primordial Malzeno Standard), Bilateral Eyes & Baked PBR Format:**
   - **Piece-by-Piece `0.5u` Dual-Sided Solid Wing Slabs + Orthogonal Bay-Plane Projection (`mesh_baker.py`):** Build every wing bay as a **solid `0.5u`-thick dual-sided `box` (`size: [W, 0.5, D]`)** buried `>= 2.0u` deep inside its bounding spars and staggered `±0.60u` along local `Y` (`0` Z-fighting blink, `99%–100%` dorsal/ventral silhouette parity).
   - **Non-Standard Luminous Gradient Eyes & Baked Studio PBR + Blockbench 5 Plugin:** Export `_baked.png`, `_baked.bbmodel`, `.synccraft.pbr.json`, LabPBR 1.3 `_n`/`_s`, Bedrock `_mer`, and `.mcmeta` flipbooks via `g.export_baked_pbr(...)` / `synccraft texgraph bake`, and preview/rebuild interactively inside Blockbench 5 via `editors/blockbench-pbr-rebuilder/synccraft_pbr_rebuilder.js`.

### 10.2 Clean, not dirty — and organic, not plastic
- **Base value:** at least 60% of each face sits at its base value. Let solid colour islands breathe; strokes and bands make the rest.
- **Two complementary painting modes (use the right one per surface):**
  - *Crisp 1 px palette-ramp mode* (default for 16 px Minecraft hides, scales, armor bevels, crystals, wood and props — Anchors 02, 03, 04, 05): hard 1 px strokes from a curated 4-5 step hue-shifted ramp for crisp, readable world pixels.
  - *Soft-layered directional blending mode* (for broad wing membranes, sweeping atmospheric fades and subtle large-mass gradients — Anchors 01, 05): low-opacity directional strokes (`~25-35%`) plus straight radial rib lines (`Shift`-line habit) and frequent tone sampling (`Alt`), followed by crisp 100%-opacity alpha carving on trailing edges.
- **Break "plastic" procedural contours (critical anti-AI/amateur rule):**
  - Uniformly stepping concentric dark -> mid -> light rings symmetrically around every face edge, or painting rigid mechanical horizontal gradient stripes, makes a model look robotic and plastic.
  - Paint transitions in **large organic directional strokes** and **jagged/zigzag clump borders** (pulling light pixels up and dark pixels down across fur or scale boundaries), and **deliberately break/offset edge highlights** selectively rather than boxing every face border identically.
- **Manual anti-aliasing (smooth curves without blur):**
  - On curved scales, jaws, markings or diagonal borders at 16 px resolution, tuck **1 px intermediate palette steps** into pixel stair corners so curves read smooth while staying 100% opaque and crisp. Never use blurry semi-transparent alpha edges.
- **Sculpt volume and sub-silhouettes through value:**
  - *Fake roundness on cuboids:* subtly darken the outer/top/bottom corners of cylindrical limbs, necks and tails along their long axis so 4-sided boxes read rounded in 3D.
  - *Carve bulky forms optically:* slim chunky claws, toes, pincers or armour rims by painting dark recessed cutaway pixels and a crisp dark+highlight dividing seam instead of adding tiny extra cubes.
  - *Rim-over-shadow popping (Anchors 01, 02, 05):* on raised scales, carapace plates and ventral scutes, pair a 1 px bright top/leading edge with a **1 px dark cast-shadow row immediately beneath/behind the rim** so every overlap pops in 3D.
- **Strokes, not speckle** (Anchors 01, 02, 03, 05):
  - 1 px dashes, 3-5 px long, half a ramp step lighter or darker than the face's base value, covering about a third of the face.
  - They run along the form's long axis: lengthwise on limbs, spars, horns, neck and tail, and horizontal along the body on the torso.
  - Set the torso direction explicitly. A chest box taller than it is long otherwise gets vertical strokes that look like rain.
- **Measure the paint** on interior patches (no box edges) of a flank, a limb and a membrane, rendered at 3-5 screen pixels per texel:
  - the mean luminance step between neighbouring pixels is 4-9 (on 0-255);
  - at most 7% of neighbours jump more than 25.

  Higher reads as dirty; much lower reads as plastic.
- **Ramps** (5 steps per material):
  - Build each ramp from the midtone outward (or bootstrap natural material relationships by sampling blurred patches of real photo references, then boosting saturation and contrast to fit Minecraft vibrancy).
  - Keep value steps even: neighbouring steps within about 10 of each other on 0-255, roughly 30-40 apart.
  - **Shadow saturation and hue shift:** never shade toward neutral grey or pure black (it washes out the asset). As shadow deepens, **slightly increase saturation** and shift hue toward cool ambient shadow (indigo, violet or deep blue — Anchors 01 & 02); shift lights toward warm yellow, cyan or cream.
  - **Anti-muddy-midtone check:** low-opacity blending between warm highlights and cool/grey tones can produce dead greenish or chalky midtones; sample and nudge warmth/saturation back up whenever mud appears.
  - Body ramps swing 20-50 degrees in hue from darkest to lightest.
  - Cream and bone ramps swing less and keep their shadows muted. A generator pushed toward blue can turn cream shadows orange; check it.
  - A ramp whose top step jumps twice as far as the others reads as plastic highlights.
- **UV axis alignment and clean atlas hygiene:**
  - Orient UV islands cleanly along each form's primary axis so lengthwise or horizontal pixel strokes never shear diagonally across skewed UVs.
  - Use power-of-two packed atlases with clean UV padding; share mirrored limb/digit UVs where symmetry holds; never waste atlas area on permanently buried internal faces or leave stray unpainted pixels.
  - If organic meshes are ever used outside strict cuboid assets, control seams explicitly along edge loops (`Divide`/`Join`), tune Edge/Island angle thresholds, and stitch adjacent faces (`Connect UV Faces`) to prevent jagged fragmented islands.
- **Material language** (round-3 references): borrow vanilla materials where they fit.
  - Stone or deepslate mottle, as 2-4 px chips, for rocky and armoured hides.
  - Plank grain for wood, leaf speckle for foliage, prisms for crystals.
  - Glow is the life accent: eyes, a chest core, runes or a crystal, with a 1 px bleed.
  - Columnar stone (vanilla basalt): lengthwise strokes, a cross-joint every 6-8 px, a lit edge on one facet.
- **Light from the glow:** faces that look toward a glowing part get a warm gradient near it (uplight), and cracks near it carry the glow colour. The glow then reads even from views that hide its source.
- **Glow reads as eyes.** Two lit dots on the front of a snout read as a second pair of eyes. Keep glow off the face except the eyes.
- **Ring bands** on limbs and tail: 2 px one step darker, every 6 px, across the long axis.
- **Belly plates:** a seam row one step darker under each plate (every 5 px), with horizontal strokes between.
- **Horns, claws and spikes:** lengthwise strokes; the root third one step darker, the tip third one step lighter.
- **Markings:** large graphic shapes that follow the form, such as dorsal chevrons or stripes along the long axis.
- **Colour budget:** at most 5 colours per material; 3 + 1 per creature (body, underside, hard parts, one accent).
- **Never:** per-pixel noise, black outlines, grey shadows, or stripes reset on every segment (candy-cane horns).
- **Outlines in any colour count.** A bright edge running round every face reads as UI. Light only the top or leading edge.
- **Fur is not scales.** Use short strokes along the hair direction at half-step contrast, with countershading and a clean belly line. Scattered 1 px flecks read as dirt (round 4), and brick scale rows read as stitches.
- **Glow:** at most 4% of the surface, near the focal point.
- **Paint follows geometry:** eyes go where 6.3 says, teeth on the tooth line, highlights on top faces.
- **Second texture pass:** it removes as much as it adds.

### 10.2b Vigorous Cross-Cube Pixel-Grid Relief Carving (`16 px/block` Master Secret)
Half of the "vigorous carved monster" feeling in Anchors `06–09` and Kundzio's/TrongOBD's timelapses (`hammerhead_1qB6m5Uct0k` `02:45–06:12`, `G08H2Aj-xy4` `02:20–06:48`, `zelno` `08:30–15:42`) comes from **carving deep 3D optical relief across clean, honest cuboids** instead of subdividing geometry:
1. **`45–50°` Hue-Shifted Gradient Palette Bootstrap (`1qB6m5Uct0k` `02:02–02:45`):** Lock `Per-Face UV` at `16 px/block` and generate `5–8` step ramps with a **`45°–50°` hue shift** so highlights pull warm (cream/gold/magenta/sage) while deep crevices pull cool and saturated (violet, indigo, plum-black, swamp-teal).
2. **Cross-Cube Transverse Scale-Rung & Chevron Wrapping (`1qB6m5Uct0k` `02:45–06:12`, `Anchors 06 & 07`):** Down clean dorsal spines, tails, and limb fronts (`shins/forearms`), paint rhythmic **`2–3 px` transverse chevron / U-shaped scale rungs** (`1 px` lit leading bevel + `1–2 px` mid-plate + `1 px` deep crevice shadow), and **wrap every rung `3–5 px` down onto the adjacent left/right vertical side faces** across cube seams! Finish lower flanks with a soft dark ambient-occlusion belly fade.
3. **Corrugated Exoskeleton Ladder-Ribs (`Anchor 09` Wing-Arms):** On monumental armored wing-arm beams or greaves, keep the beam a bold rectangular cuboid and carve **transverse `2 px` lit violet/metal bars alternating with `2 px` deep recessed troughs** framed between raised `1–2 px` longitudinal rail borders.
4. **Cross-Seam Anatomical Muscle Crevices & Elemental Fissures (`Anchor 07` Deviljho):** Paint deep dark tendon/muscle-belly grooves continuously across tilted compound cuboids, tucking **`1–2 px` glowing crimson/amber energy cracks strictly inside the darkest crevices** (never scribbled over flat midtones).
5. **Runic / Geometric Horn Carving & Vanilla-Kin Material Clusters (`Anchors 06 & 08`):** Carve pale multi-segment horns with `1–2 px` dark-stone Greek-key / runic spirals wrapping two faces (`Anchor 06`), and texture rocky/amphibian hides in crisp **`2–4 px` vanilla-native mossy stone / lichen / tuff pixel clusters** (`Anchor 08`).

### 10.2c Micro-Depiction Aesthetic Refinements & Mandatory 4X Ultra-Zoom Audit (`v21`)
See `references/ecology_and_whole_body_balance_animation.md` (Part 3) for full macro evidence against Anchors `06–09`:
1. **Zero-Leak Single-Face Recessed Orbital Eyes (`Trap #12`):** Never let an eye cuboid bulge past the cranium/brow wall or paint glowing iris pixels on `south` + `west` + `east` simultaneously (which causes eyes to leak out past cheeks/front). Recess the eye flush inside the lateral orbital pocket (`~0.5–0.7u` lateral relief, tucked behind the front face of `cranium_core`), paint ALL 5 non-lateral faces dark obsidian orbital wall, and paint the **4-layer dragon eye** (`1 px` dark socket rim -> fiery iris gradient -> crisp `1 px` vertical dark slit pupil -> `1 px` warm-white catchlight speck) **strictly on the single outward-facing lateral face** (`west` on `-X`, `east` on `+X`).
2. **Sleek Tapered Snout & Upper-Jaw Discipline (`Trap #13`):** Never exaggerate a dragon/visor muzzle into a drooping `35–50°` parrot/vulture hook. Trace close-up profiles directly (`9–16°` continuous aristocratic sallet taper, `2x1 px` dark lateral slit nostril, flush maxilla V-rails with `1 px` dark gum-groove + staggered ivory teeth, and clear chin-tip separation with zero nose Z-fighting).
3. **Disciplined Weathered Saturation & Organic Shingle Scales (`Trap #14`):** Keep `85%` of metallic/organic surfaces at disciplined `20–42%` saturation matching Anchors `06–09` (warm pearl-platinum over slate-ash, burnished champagne-gold over umber-bronze, smoked war-bronze — NEVER baby-blue silver, cartoon banana-yellow gold, or blown-out paper-white wing centers). Replace uniform `yy == 0` Lego-tile border lines on pauldrons/cuirass with **organic `U`-scallop dragon-scale shingles**, carve diagonal rib plates across exposed `chest_core` side walls, and taper pauldrons flush to the deltoid contour.
4. **Mandatory 4X Ultra-Zoom 6-Angle Part Audit:** Before finalizing `W5` or `T5`, render 6-angle 4x zoomed sheets (`FRONT`, `3/4`, `PROFILE`, `TOP`, `UNDER`, `BACK`) for Head/Eyes/Snout, Torso/Pauldrons/Hands, and Hindlimbs/Wings/Tail to catch eye leaks, awkward hooks, Z-fighting, or plain gray side walls invisible at turnaround distance.

### 10.2d Tiny Vigorous Craft Details: Elevating Blockbench From Clean to Alive & Gorgeous (`v23`)
To make a model feel **vigorous, grounded, and genuinely gorgeous** without adding high-poly clutter or losing the `16 px/block` Minecraft soul, apply disciplined micro-storytelling ONLY at high-value focal transitions (full table & pitfalls in `references/cranial_and_jaw_structural_reasoning.md` §5):
1. **Zero-Thickness Single-Pixel-Square Teeth & Rich Interior Mouth (`§6.5`):** Build a continuous 6-plane (`0`-thickness) single-pixel-square ivory U-arcade (`1×1 px` squares + `1×2 px` front canines, staggered `1 px` upper-vs-lower), `1 px` upper palate rugae, dark medial jaw walls + recessed pharynx box, concave hard-alpha cheek-tendon webs (`plane jaw_cheek_web`), and a species-harmonized articulated 2-segment curved tongue (`bone tongue`). Verify in a `6x` macro grid at **Closed (`0°`)**, **Half-Snarl (`20°`, Anchor `06` pose)**, and **Full Roar (`38°`, Anchor `09` / Capcom pose)**!
2. **Subtle Recessed Nostril Slits (Anti-Googly-Eye Law):** Place a single subtle dark-slate (`base[0]`) `1×1 px` recessed nostril tucked forward near the anterior corner of the snout bridge. **NEVER** stack a pitch-black dot directly beneath a bright white highlight dot in the middle of the snout side — it reads immediately as a cartoon googly eye!
3. **Orbital Lower-Eyelid Rim & Tear-Duct Framing:** Pair every recessed glowing slit eye with a **`1 px` pale catchlight** along the top ridge of the cheekbone (`zygoma_shield`) directly beneath the eye + a **`1 px` dark orbital shadow crease** right under it, framing the gaze like forged knight-visor steel or layered reptile scales.
4. **Selective `1 px` Honed Bevel Specular Micro-Glints:** Place crisp, sparse `1 px` warm specular catchlights (`ramp[6]`) strictly on the 4–6 sharpest apex bevels (forehead crest diamond, beak prow apex, sternum V-keel peak, leading pauldron corner, halberd blade tip) — never outlining entire box edges symmetrically.
5. **Keratinized Root Collars, Joint Flexor Creases & Scale-Overlap Cast Shadows:** Anchor every horn root and talon base with a `1–2 px` darkened keratin cuticle collar, carve `1 px` deep flexor tension creases inside elbow/knee/wrist bends, and paint a `1 px` cool-indigo cast shadow on the host box directly underneath every overlapping armor plate or dorsal scute (compensating for Blockbench's flat directional viewport shading).

### 10.3 Paint by region
- **Head:** darker around the mouth and eye sockets; lighter on the brow ridge and snout top. The underjaw is in the underside colour, with a lip line one shade darker.
- **Torso:** countershaded (dark back, mid flank, light belly), with one soft band at each transition. Patterns follow the long axis and thin toward the belly.
- **Limbs:** darker toward the feet, a lighter kneecap highlight. Bands are at most one dark row in five.
- **Claws, horns, teeth:** a light top edge and a dark base. The lighter tip goes on the last segment only.

### 10.3b Five-Layer Painterly Wing Membrane Cutout & Shading Grammar
Never paint wing membranes as flat uniform rectangles, horizontal stripe bands, or jagged comb strips (`8.4d`, `G10`). Process every wing membrane in **five deliberate layers** (`Vijay` `5o_Ijoq_dGQ` & `YDYIFgTZKRI`, `Longridr` `87jJmKAgFqE`, `Darto 3d Rathalos` `Z58DMwfZtx4`, `Anchors 01, 03, 05`):
1. **Exact Bay-Polygon `100%` Hard-Alpha Mask + `>= 1–2 px` UV Padding (`T1`):**
   - Map each full-chord (`>= 1.08x tip_gap`) wing plane into its 3D/2D bay sector `[root_i -> tip_i -> concave_scallop_arc(tip_i, tip_{i+1}) -> root_{i+1}]` (when scripting in SyncCraft, project each texel's `p_world` directly onto the adjacent `WorldBox` finger-spar vectors so the alpha edge lands **to the exact pixel along Finger `i+1`** with `1 px` tuck under the spar!).
   - Erase everything outside that polygon with `alpha = 0` (`100%` hard binary `{0, 255}` alpha, `0%` feather blur), carve a taut concave catenary/scallop (`sag = 0.18–0.30`) between fingertips leaving `10–20%` of the distal claw/tip protruding free, and mirror double-sided UVs on the **correct axis** (`normal: y` flips **V** `[x, y+h, x+w, y]`; `normal: x/z` flips **U** `[x+w, y, x, y+h]`) with `>= 1–2 px` padded UV borders so neighbouring islands never bleed phantom outlines (`87jJmKAgFqE` `0:24`).
2. **Both-Spar Bone-Hugging AO & Saturation Frame (`T2`):**
   - Shade `2–4 px` along **both** bounding spars (`f_i` leading bone AND `f_{i+1}` trailing bone + wrist hub) `1–1.5` ramp steps darker and more saturated (deep indigo, plum-wine, or obsidian) so the membrane visibly anchors into and pulls away from both bones (`Z58DMwfZtx4`, `87jJmKAgFqE` `19:07`).
3. **Translucent Subsurface Mid-Bay Cushion (`T2`):**
   - Swell a warmer, lighter, luminous value (`steps 2–4`) softly across the stretched center belly of each bay cell (`~25–35%` soft directional brush or quantized barycentric distance from the spars) to convey thin stretched patagium translucency.
4. **Radial Tension Folds & Organic Forked Venation (`T3`):**
   - Pull `1–2 px` straight half-step tension striations (`Shift`-line discipline, `Anchor 01`) radially outward from the wrist knuckle toward the scalloped edge, plus `1–2` asymmetric forked blood/mana veins branching from the proximal spar — *never* uniform parallel sine-wave stripes!
5. **Trailing Rim, Battle Tatters & Cross-Bay Species Motifs (`T4–T5`):**
   - Finish the scalloped hem according to species: crisp `1 px` rim catchlight (`Anchor 01`), pixel-stepped ragged notches + `1–2` outer-bay oval perforations (`Z58DMwfZtx4`, `Anchors 03 & 05` — strictly in the outer `25%`, never carving away `> 30%` of bay depth!), layered luminous feather-vane tips (`Primordial / Sphinx`), or bold calligraphic/whorl motifs spanning continuously across all bays (`YDYIFgTZKRI`).

### 10.4 Presentation
- **Turnarounds:** a dark neutral backdrop.
- **Hero shots:** a simple scene, a key light with a coloured rim, a little bloom on emissive parts, and a cast shadow.

## 11. Engine-Agnostic Blockbench Craft vs. SyncCraft Block-DSL v4.1

### 11.0 Decoupling Formal Blockbench Mastery from Experimental SyncCraft Limits
This skill defines a **standalone artistic and spatial grammar** across native Blockbench `.bbmodel`, Minecraft Bedrock `.geo.json`, GeckoLib, human-artist co-piloting, small-model vision-spatial training, AND SyncCraft Block-DSL (`references/vigorous_carved_aesthetic_and_spatial_reasoning.md` Section 5). **Never allow experimental SyncCraft collision constants to flatten a creature into a simplified toy:**
- **Known Experimental SyncCraft Over-Restrictions (`sync_audit.py`):**
  1. `MAX_SCULPTED_PARALLEL_OVERLAP_RATIO = 0.25` (at `< 5°`) and `MAX_SCULPTED_BURIED_VOLUME_RATIO = 0.65` reject `50–85%`-sunken parallel stepped chamfers (`Anchor 08`), sunken Rathalos thigh scutes (`Z58DMwfZtx4` `01:08`), and embedded Deviljho chin osteoderms (`Anchor 07`).
  2. `place(anchor=±1)` / `attach()` forbids burying a child box's center inside its host OBB (`max_inset_depth = 0.5u`), inducing outer-wall **"Lego-stud sticker"** construction if used naively.
  3. Sibling bones under a shared parent (`deltoid_l` vs `trapezius_l`) lack automatic sculpt immunity unless listed in `blend_with`.
- **Mandatory Authoring Rule:**
  - When authoring **native `.bbmodel` or Bedrock `.geo.json`**, pack compound muscles, `30–45°` bevel clusters, and `60–80%`-sunken osteoderms freely just like human masters, audited directly by `python3 scripts/construct_check.py --tier <tier> model.bbmodel` (`G1–G12`)!
  - When authoring inside **SyncCraft Block-DSL**, avoid `place(anchor=±1)` for compound musculature; instead place explicit bone-local `box ... from=[...] to=[...]` coordinates and assign shared **`prism_group: "<anatomical_cluster>"`** + cross-bone `blend_with` to intra-limb / intra-jaw / intra-carapace bevel-clusters (legitimately unlocking `40–85%` sunken mass-packing while keeping **inter-limb / wing-to-body** `288`-frame swept 15-axis OBB SAT, `SE(3)` kinematics, grounding, and coplanar Z-fighting checks 100% active!).

### 11.1 Specifications
When compiling through SyncCraft, read `docs/BLOCK_DSL_V3_2_SPECIFICATION.md` and `docs/BLOCK_DSL_V4_SPECIFICATION.md`. Use `SE(3)` forward kinematics, coplanar Z-fighting detection, grounding, and inter-limb swept OBB `SAT` as physical error correction alongside `scripts/construct_check.py` (`G1–G12`).

### 11.2 Meta and layout
- `@meta { dsl_version: "4.1", audit_profile: sculpted, size_step: 0.5, units_per_block: 16, texel_density: 16 }`.
- `@layout { budget max_boxes = ..., envelope grounded = true }`. Budgets are ceilings.

### 11.3 Constructs
- `attach(..., seat=0.28)` for chains.
- `mirror_x(name)` / `mirror(axis=x)` for symmetry; `mirror: true` on stamps.
- `chain <name> attach(...) segments=N taper=... seat=... bend=[...]` for necks, tails and horns, with N within the remap limits.
- `plane` with `layer: k` (0.25 units per step) and `edge: ragged|spikes|drip|scallop|triangle_bay`.
- `blend_with: [...]` for cross-bone seams.
- `octagon` for hard-surface round parts. `prism(n)` never on organic parts.
- **`joint_limits` are relative to the rest pose:** min <= 0 <= max. Writing absolute angles fails validation or allows IK flips.
- **`mirror_x` on a bone attached to a non-centred box** mirrors only the anchor inside that box, so the right copy lands inside the body. Attach mirrored limbs to centred boxes; put shoulder masses inside the limb's own root bone.
- **Deep overlap is rejected even with `blend_with`.** Radiating spines, fingers and fan rays need separate roots, spaced apart.
- **Pitch sign depends on chain direction:**
  - positive pitch lowers a chain that points forward (neck, head, jaw);
  - positive pitch lifts a chain that points backward (tail).

  Write the intended direction in the comment, then check it in a render.
- **Plane `size` order:**
  - `normal: x` is `[height, length]`;
  - `normal: y` is `[width, length]`;
  - `normal: z` is `[width, height]`.

  A wrong order still passes the audit.

### 11.4 Surface
- `@style { shade: banded(steps=4, direction=down, cluster=diagonal), glow_bleed: 2 }`.
- Coats: `scale_rows`, `banded_limb`, `plain`, `striped`, `spotted`.
- `@stamps` for eyes and teeth.
- **Keep Block-DSL clean like a master Blockbench artist:** whole-unit or `size_step` (`0.5`) box dimensions keep UV texels square; resolve coplanar flicker with `layer: k` or clean offsets rather than irregular decimals; name every bone in lowercase `snake_case` anchored to its anatomical pivot; paint by hand or by script on the exported UV layout whenever procedural coats look plastic, uniform or fail to deliver rim-over-shadow overlap and manual anti-aliasing (10.2), and record that you did.

### 11.5 Checks
- `audit_scene(scene, audit_profile="sculpted")`: collisions, coplanar faces, contact and burial, budget, grounding.
- `python3 scripts/construct_check.py [<SyncCraft_repo>] <model.blockdsl>`: automated `G1-G7` construction and style-grammar audit (3.6).
- `analyze_morphology(program)`: limbs, posture, `symmetry_pairs`.
- `audit_behavior_clip(...)`: joint limits and swept motion.
  - Gait patterns: `lateral_sequence`, `diagonal_pairs`, `rotary_gallop`, `transverse_gallop`, `biped_alternate`, `sprawl_diagonal`, `hexapod_tripod`. There is no hop; build hops from in-phase tracks.
  - A walk that clips usually has too long a stride, about 1x hip height is right, or legs set too far forward under the chest.
  - Track clips are not ground-checked. Measure the lowest point across the frames yourself.
  - **Fast clips (runs)** can exceed the 0.5-unit step bound at the default 144 frames, because the audit measures world motion including the root's advance. Pass `max_frames=288`: denser sampling makes the swept check stricter, not looser.

### 11.5b Locomotion & flight clips stay in place (Minecraft Entity-Space Contract, `v26` native)
The game's own entity controller moves and steers the creature through the world (`+Z` travel, heading turns, and cruising flight elevation). Therefore every walk, run, trot, hover, cruise, glide, takeoff, and swim clip MUST operate **in-place** in entity space (`in_place: true`, now native across `gait`, `flight`, and `clip` in SyncCraft `v26`):
- **Native SyncCraft `v26` behavior (`in_place: true`):**
  - `gait` solves closed-loop `SE(3)` world IK in virtual advancing space (`slip < 0.02u`), then strips ONLY macro `adv_z` from exported root position keyframes while preserving local vertical heave/crouch AND writing exact matched speed into `synccraft_root_motion` (`speed_blocks_per_s`, `velocity_vec`, `"in_place": true`).
  - `flight` strips fake macro flight altitude (`+30..+80u`) and forward Z-drift, keeping ONLY authentic anti-phase center-of-mass lift recoil (`±0.5..4.5u`) on `root.y` while emitting matched cruising velocity in `synccraft_root_motion`.
  - If editing legacy non-`in_place` `.bbmodel` exports manually, zero `x` and `z` on the root's `position` keyframes while preserving local `y` heave/recoil and `synccraft_root_motion`.
- **Check:**
  - every root `x`/`z` locomotion & flight position key is `0.00u`, and aerial clips never bake fake `+30u` altitude (`max|root_y| <= 4.5u`);
  - with the root's macro advance removed, each planted foot moves backward at the exact exported speed (`0` skate when Minecraft advances the entity at `speed_blocks_per_s`);
  - track/expressive clips put zero macro forward translation on the root.
- **Show it:** render multi-frame filmstrips of the clip on one fixed camera with ground ticks. The creature stays centered in entity space while feet cycle cleanly (`< 0.02u` stance slip) or wings beat asymmetrically with local COM lift recoil.
- **Run, compared with the walk:**
  - the same pattern, with about twice the Froude number (`Fr`);
  - lower duty factor (`< 0.55` for fast charge/gallop) and higher `lift_ratio`;
  - wings or arms dynamically stabilized via `flight`, `free_limbs` auto-swing, or mirrored-phase tracks.
- `evaluate_sculpted_style_anchor(program, scene, atlas_rgba=...)`.

### 11.5c Aerodynamic flight waves (ArtsByKev flight mechanics)
When rigging and animating flying dragons, wyverns or winged bosses:
- **Asymmetrical wing stroke force:** drive shoulder roll with a **faster, forceful downstroke** and a **slower, pausing recovery upstroke** so the creature carries real weight in air.
- **Linear timing with manual braking:** prefer controlled linear interpolation with explicit incremental/decremental braking keyframes immediately before stroke peaks rather than floaty uncontrolled curves.
- **Root-to-tip phase lag:** lag `wing_forearm` and `wing_f*_tip` behind shoulder roll by `25-45°` (kept tight enough that wingtips trace a crisp figure-8 arc rather than flapping flimsily).
- **Marine spine counter-motion ("birds are shrimp/fish"):** counter-pitch `chest`, `back` and `tail` opposite the downstroke so the torso lifts on power strokes and trails cleanly through the tail.

### 11.5d YouTube Ecology Research & Whole-Body Kinetic Balance Laws (`v21`)
Static promotional renders almost always show maximum-display combat poses and mislead `idle`/`walk` choreography. Before authoring `A1`, search YouTube for HUD-free wildlife **ecology research footage** (`"<monster> ecology"`, `"<モンスター名> 生態 観察"`) and extract timestamped motion study sheets (`references/ecology_and_whole_body_balance_animation.md`):
- **Zero Frozen Appendages (Six-Subsystem Coupling):** Every animation clip must visibly balance ALL six subsystems simultaneously: `pelvis_root` (position + pitch/yaw/roll center-of-gravity shift), `waist/chest` (opposite-phase spinal torque), `neck_1..3/head` (hawk-like gaze stabilization + jaw breathing), `wings` (natural resting cloak drape vs. aerodynamic counter-spread), `limbs` (stepping, crouching, bracing, or clutching into every attack — NEVER frozen at `[0,0,0]`), and `weighted tail` (`tail_1..6` counter-momentum).
- **Resting Wing-Cloak Carriage vs. Combat Sail:** When calm (`idle`, `walk`, `royal_sit`), fold and drape large wings sleekly aft over the flanks like a knight's cloak (SAT-probed at `coll == 0` first) paired with a **low horizontal S-gliding tail rudder**, rather than leaving wings frozen straight up in the air.
- **Conservation of Angular Momentum:** During asymmetric wing/claw sweeps, drop `pelvis_root`, step the leading forelimb into the swing, twist `pelvis_root + waist + chest` together, stabilize the head on target, and **hurl `tail_1..6` in the opposite horizontal arc** (`±20–26°`). During forward bows or overhead tail strikes, pitch `pelvis_root` forward and plant both forelimbs wide to counter-weight the arched tail. During rear-up roars/supernovas, pitch `pelvis_root` steeply back (`-18°..-22°`) with foreclaws clutched to the chest before exploding outward.

### 11.5e Unified Universal Biomechanical Animation Pipeline (`SO(3)` / `SE(3)` / `Sim(3)`, `v26`)
Instead of hand-typing brittle raw bone keyframes or hardcoding species-specific animation scripts, animate **any real animal OR brand-new imaginary creature** (0-leg serpents/fish, 2-leg humanoids with free arms + fingers, 4-leg quadrupeds/dragons, 6/8/10+ leg spiders/aliens with arbitrary non-English bone names, 2/4/6-wing flyers, multi-headed hydras, multi-tailed kitsunes, tentacled krakens) directly from its **skeleton and muscles** (full reference in `references/unified_biomechanical_animation_solver.md`):
1. **Universal Capability Introspection:** Call `analyze_motion_capabilities(program)` to inspect the rig's geometry-first locomotion class, grounded limbs, free upper limbs, digits, multi-pair wings/fins, multi-head neck chains, multi-tail chains, tentacles, and muscles — zero English bone-name assumptions.
2. **`SO(3)` Biological Joint Harness (`bio_joint` & `auto_bio_limits: true`):** Tag anatomical joints with SyncCraft's 26 `BIO_JOINT_PRESETS` (e.g., `finger_flex` blocking backward finger hyperextension, `hinge_elbow`/`hinge_knee`, `jaw_hinge`, **`wing_anchor_locked`** locking `wing_l_f4` strictly at `[0, 0, 0]` to keep inner wing membranes sealed) and enable `auto_bio_limits: true` for smooth $C^1$ `tanh` elastic ligament clamping + automatic geometry-first constraint inference (`infer_universal_bio_constraints`). Remember: all `joint_limits` are **deltas relative to `rest_euler`** (`min <= 0 <= max`).
3. **Closed-Loop `SE(3)` Grounded Locomotion (`gait`):** Supports `lateral_sequence`, `diagonal_pairs`, `rotary_gallop`, `transverse_gallop`, `biped_alternate` (with automatic anti-phase `free_limbs` arm counter-swing!), `sprawl_diagonal`, `hexapod_tripod`, **`metachronal_wave`** (8+ legs), **`serpentine_wave`**, and **`swim_undulate`**. End `ik_chain`s **above** flat ground-contacting paw/foot pads (`2`-bone forelimb / `3`-bone digitigrade hindlimb) with anatomically outward `pole` vectors so feet stay flat (`< 0.02u` slip) with zero paw branch-flips and zero chest clipping.
4. **Universal Aerodynamic & Hydrodynamic Airfoil Solver (`flight <name> { ... }`):** Supports `hover`, `cruise`, `glide`, `soar`, `takeoff_burst`, **`mantle`** (ground-locked `SE(3)` threat wing-flare at `0.000u` foot slip!), and `swim_fins`. Automatically runs 3D angular Jacobian axis calibration, exact mirrored bilateral symmetry (`asym = 0.00000`), bind-pose dorsal elevation anti-clap clamping, asymmetric fast power downstroke (~44%) vs folded/feathered recovery upstroke (~56%), Figure-8 wingtip sweep, distal phase lag, multi-pair wing phase staggering (`pair_phase_deg`), Jacobian-abducted aerial `limb_tuck`, anti-phase `in_place` COM lift recoil, and `wing_anchor_locked` preservation.
5. **Unified `Sim(3)` Muscle Bulge, Standing Limb IK Pinning (`ground_pin_limbs`, `v28`), Gaze Lock & Universal Mechanical Hardware (`v28`):**
   - **Closed-Loop Standing Limb IK Pinning (`ground_pin_limbs: true | [hind_l, hind_r]`):** On any non-`gait` standing clip (`idle`, `bite`, `cloak_sweep`, `tail_whip`, or hind-pinned bipedal `roar`), declare `ground_pin_limbs` so `solve_standing_limb_ik_pinning` locks wrist/ankle world pivots (`< 0.02u` slip) AND applies exact `SO(3)` world-flat sole orientation lock before and after secondary springs — **completely eliminating the `-4..-6.5u` raw Blockbench `.bbmodel` FK floor dip**!
   - **Vehicles, Airships & Mechs (`BIO_JOINT_PRESETS` + Rigid `Sim(3)` Exemption):** Tag vehicle/mecha joints with `wheel_axle`, `steered_wheel_knuckle`, `rotor_propeller`, `turret_yaw`, `cannon_elevation`, `hydraulic_hinge`, `hatch_door`, or `gimbal_thruster`; `auto_assemble_biomechanical_rig` automatically detects `wheeled_vehicle` / `MechanicalSubsystem` and exempts all rigid mechanical bones from `Sim(3)` muscle bulge!
   - Attach `@muscles` with Zajac/Hill mechanics, `tendon_coupling`, and volume-preserving `scale_bulge` on biological limbs/thorax; add `SO(3)` vestibulo-ocular `head_stabilize`; and layer implicit-Euler damped `spring` follow-through.

### 11.6 Stage files and renders
`scripts/inspect_mh_sculpt.py <id> --stage 1|2|3 --dsl-root <your dsl dir>` renders stage files laid out as `dsl/<id>/stage1_blockout.blockdsl` (W passes), `stage2_secondary.blockdsl` (D passes) and `<id>.blockdsl` (final). Keep every pass as `passes/<id>_<pass>.blockdsl`.

### 11.7 Scores are diagnostics, not targets
The style evaluator rewards a high share of rotated boxes, and the dataset tests demand at least 260 elements. Chasing either produces shard armour. Treat both as diagnostics, never as targets.

### 11.8 Gotchas
- **Re-ground** after every pose change, so the lowest toe sits at y = 0.
- **The audit tests a plane's whole rectangle,** not its alpha shape.
- **Every size** is a multiple of `size_step`.
- **Box `rotate` is applied as Rz·Ry·Rx about the box centre,** and `place` alignment is applied before the rotation. To keep a rotated prism's base on its anchor, use offset = R·(0, h/2 - bury, 0) - (0, h/2, 0) (the `seat()` helper). Keep the burial under the overlap limit, about 1 unit for 3.5-wide prisms.
- **Anchors must sit on a face** (one component at ±1). An anchor inside the box buries the part and fails the overlap check.
- **`anchor` and `align` components must lie in [-1, 1].** To start a box beyond its pivot, move the pivot instead.
- **On a reared torso, the belly is the chest's local -y face and the back is +y.** Put ventral panels and plates on anchor y = -1, and point crystals using world-up expressed in the bone frame.
- **Cross-subtree `blend_with` needs the mirrored names in full** (`leg_r__thigh`, `arm_r__shoulder`). Names inside a mirrored subtree are prefixed automatically.
- **Gait IK can drive a bent knee deeper than its joint limits.** If the knee seam collides in walk, start the shank at the thigh's bottom face, blend the knee seam and lower `lift_ratio`.

## 12. Subagent brief (paste verbatim into every subagent call)

```
You are working on <model> in the formal Vigorous Carved Blockbench style (v40.0, tier=<small|standard|hero|epic_boss|vehicle_mech>).
Before any visual reasoning or construction, and again at the start of every pass:
  1. read SKILL.md, references/high_res_focal_carving_and_visual_primitives.md, references/aesthetic_mood_spectrum_and_intent_router.md, references/vigorous_carved_aesthetic_and_spatial_reasoning.md, references/youtube_master_distillation.md, references/ecology_and_whole_body_balance_animation.md, references/cranial_and_jaw_structural_reasoning.md, and references/unified_biomechanical_animation_solver.md in full;
  2. run scripts/visual_primitives_prep.py (`grid --autocrop --dual-edges` + `trace --export-crops-dir`) on multi-perspective native-resolution references & ad-free video frames (NEVER crush references into compressed thumbnails!) to inspect structural edge-maps, dashed skeleton traces (`- - -`), joint bend angles, proportion calipers, and 1:1 focal zooms (`head_eye`, `limb_joints`, `tail_plates`, `accessories`);
  3. read PROJECT_GOAL.md (Domain D1-D5 x Temperament Polarity Gentle/Approachable vs Furious/Unapproachable, 6-Stage Spatial Curriculum S1-S6, 12-Point Ratio Matrix, Signature DNA table + verbatim owner corrections).
Your pass: <W1..W4+ | T1..T3+ | A1..A4+>. Follow the human-artist multi-iteration rhythm (>=3 White-Model passes -> 2-3 Texture passes -> >=2 Animation passes); NEVER batch-generate multiple passes in one script loop!
Hard rules - Section 1 of the skill, all 15, apply without exception:
  - 32x + 2x Focal Eye Carving & Anti-Toy Tail/Plate/Accessory Craft (v33, 6.3, 8.9, G13, G14): Use texel_density: 32 (or 16 for vanilla-lite) + uv_scale: 2 on eyes/optical lenses (10x6..14x8 px!), carve 4-part recessed periorbital bezels (socket step + upper lid hood + lower lid cradle + post-orbital corner) with all 7 realistic eye paint layers (dark rim, upper-lid shadow, shaded sclera, dark limbal ring, radial iris + lower refraction crescent, authentic pupil, wet specular catchlight), and break the Lego/toy look on tails, plates & accessories via faceted cross-sections, staggered overlapping shingles (>=2/segment), 2-3 step tapered blade/spur tips, contrasting mounting collars/brackets, and crisp 32x bevel/crevice/striation painting!
  - Temperament Polarity, Dead-Front Perspective & Balanced Recessed Eye Shelf (1.5, 6.3): Lock Gentle/Approachable (Rx<=0 open brow, 0-2 deg lip, smooth convex silhouettes) vs Furious/Unapproachable (+Rx +18..+24 plunging \ / scowl brow, +6..+10 deg true world downward head glare, 12-16 deg snarl + external interlocking tusks, outward spikes, 10:1 chiaroscuro). Use box.center[0] < 0 (never name.endswith("_l")) on mirrored plane cutouts!
  - Blockbench Is Blockbench (Anti-Blender & Anti-Toy Law, 3.1c, G11, G12): ZERO wafer-sliced fake curves (<1.0u slices) or >8-band fake spheres, AND ZERO bare 1-box toy sticks, tall stilt-horse legs (keep grounded belly_clearance <= 0.42x shoulder), or outer-wall Lego stickers.
  - Unified Biomechanical + Standing-IK + Mechanical Solver (11.5b, 11.5d, 11.5e): Enable ground_pin_limbs: true (or [hind_l, hind_r] on bipedal roars) on standing non-gait clips (<0.02u slip); use wheel_axle/steered_wheel_knuckle/turret_yaw/cannon_elevation/hydraulic_hinge/hatch_door on vehicles & mechs (0% Sim(3) bulge on rigid metal!); keep ALL clips in_place=true, lock wing_l_f4 with wing_anchor_locked, pass 288-frame swept OBB SAT with 0 collisions, and verify textured filmstrips!
  - Before W1: execute S1-S6 Spatial Reasoning (5.1d), select from the 8 bespoke Cranial/Body Archetypes (5.1b — ban generic box-snouts & ban replace() variant reskins!), log 5-8 Signature DNA traits + 12-Point Ratio Matrix (5.1c).
  - Complexity, 3-Layer Limbs & Articulated Extremities (3.1b, 8.3b, G8, G9): on epic_boss (240-550 elems), sculpt core + sunken muscle/chamfer + coherent scute/vambrace rows on every shaft, plus 4-5 splayed multi-bone digits (proximal -> distal -> hooked talon, middle digit 1.2-1.35x longest, + opposable thumb or rear dewclaw).
  - Gap-Free Nested-Fan / 3-Tier Feathered Wings (8.4d, 10.3b, G10): size wing bay plane chords >= 1.08x adjacent fingertip gap (or 3 echeloned alpha feather rows), stagger layer/inflate, carve exact 100% hard-alpha polygons (>=2 px UV padding), and paint 5-layer wing shading.
  - Vigorous Cross-Cube Pixel-Relief Carving, Semantic UV v2 & TexGraph (10.1-10.2b, 11.5): Pack UVs with Semantic Anatomical UV Packer v2 (`repack_bbmodel_uvs(..., mirror_symmetric=True)`, `>= 60%` padded utilization, verified `uv_zones.png`); author studio pixel art via `synccraft.texgraph` (3D MeshBaker AO/bevel/cavity, noise/Turing/pattern nodes, pure algorithmic non-ML Kuwahara/Bilateral/TV/Despeckle polish, 45-50 deg OKLCH hue-shifted ramps, organized 2 px clusters + 1 px bevels/crevices, <= 4% focal glow, optional `.mcmeta` flipbook), hidden-face articulation check, 288-frame swept OBB SAT inter-limb clearance, and treadmill root.x/z = 0.
Run python3 scripts/construct_check.py --tier <tier> <model.{bbmodel|geo.json|blockdsl}> (G1-G14) every pass (plus audit_scene when in SyncCraft);
render full turnarounds + mandatory no-wing 4-view + zoomed region crops; apply Section 13.2 Adversarial Score Caps (never rubber-stamp 28/28!).
```

## 13. Review rubric & Adversarial Self-Judgment Law

### 13.1 14-Item Review Rubric (`0`, `1`, or `2` per item; `28` max)
1. Silhouette (`front`, `side`, `3/4`, `top`, plus mandatory `no-wing` body 4-view)
2. Bespoke Cranial Archetype (`5.1b`), V-rail/beak/visor depth (`>= 4` front layers) & tucked V-mandible (`6.6`)
3. Recessed angled eye sockets & crisp pixel-grid iris readability (`6.3`)
4. Compound-curved horns integrated into brow/cheek ridges, crown negative space & graded spikes (`7.1-7.2`, `8.9`)
5. Tier-honest sculptural complexity & purity: zero stick-limbs (`G8`), zero brick-paws/peg-toes (`G9`), zero shard soup (`G1-G10` clean)
6. Articulated extremities (4-5 splayed multi-bone phalanges + thumb/dewclaw, `8.3b`) & seamless `28%` seated / gusseted joints (`3.5`)
7. Rest-pose & motion mirror symmetry + clean lowercase `snake_case` bone hierarchy (`4`)
8. Quantitative fidelity to the **12-Point Proportional Ratio Matrix** (`R1–R12`, `<= 12%` drift vs. reference crops; `5.1c`)
9. Tier-aware volume hierarchy (`G1`) + 3-layer core/chamfer/scute limb & torso taper (`3.1b`)
10. **Gap-free nested-fan 3D vaulted wings** (`SVD >= 0.05`, `chord >= 1.08x fingertip_gap`, zero inter-finger V-gaps; `8.4d`, `G10`) — or prop parallax & contact-pivot tilts (`8.7`)
11. Rich, organised, anti-plastic paint + **5-layer wing membrane shading** (both-spar bone AO, translucent center, tension folds, `lum_step 4-9`, `<= 7%` jumps `> 25`; `10.2`, `10.3b`)
12. Vibrancy, focal contrast & bespoke trait fidelity (hue-shifted shadows, `<= 4%` glow, eyes/teeth clear of muddy overlays)
13. Expressive multi-phase animation choreography (weight, anticipation/snap/hold, phase-lagged flight, `288`-frame swept OBB `SAT` `0` collisions, treadmill `x/z = 0`, painted hidden faces)
14. Anti-homogenization & variant divergence (`5.0`, `5.1b`): unmistakable species/variant architecture with zero template snout or `replace()` reskinning

**Pass bar:** zero `0` scores and **`>= 23 / 28`**. Any `0` on items `1, 2, 5, 6, 7, 10` or `14` mandates immediate deletion & reconstruction of the failing subtrees.

### 13.2 Adversarial Self-Judgment Law & Mandatory Automatic Score Caps (`<= 14 / 28`)
Self-congratulatory `28 / 28` rubber-stamping is strictly forbidden. Every pass review must operate as an **adversarial art director** comparing rendered crops side-by-side against zoomed reference crops and listing the model's **top 3 remaining visual weaknesses** even when passing. Furthermore, **any single defect below AUTOMATICALLY CAPS the total score at `<= 14 / 28` (AUTOMATIC FAIL):**
1. **Stick-Limb / Under-Complex Boss Cap (`G8`):** An `epic_boss` asset has `< 240` elements OR leaves major limb/wing shafts (`humerus`, `forearm`, `thigh`, `shin`, `wing_upper`, `wing_fore`) as bare 1-to-2 box rectangular sticks -> `Item 5 = 0`, total capped `<= 14/28`.
2. **Inter-Finger Wing V-Gap / Rectangular Flag Cap (`G10`):** Splayed wing fingers show unintended empty triangular V-gaps between bays (`chord_coverage < 0.88x`) or over-erased thin comb strips -> `Item 10 = 0`, total capped `<= 14/28`.
3. **Brick-Paw / Clone-Peg Digits Cap (`G9`):** Hero/boss paws, hands, or feet use flat brick bases with 3 parallel 1-bone stub pegs lacking multi-phalanx chains, middle-digit length grading, and a thumb/dewclaw -> `Item 6 = 0`, total capped `<= 14/28`.
4. **Template Snout / `replace()` Variant Homogenization Cap (`5.1b`):** Reusing a generic square-nosed wolf/croc muzzle on a beaked/visored creature, OR generating a structural variant by string-replacing box sizes in the base `.blockdsl` -> `Item 2 = 0`, `Item 14 = 0`, total capped `<= 14/28`.
5. **Batch-Loop Fake-Pass Cap (`5.2b`):** Generating `W1..W5`, `T1..T5`, or `A1..A5` in one automated script run without stopping after each pass to inspect renders and execute `>= 3` concrete geometric/painterly fixes -> **Immediate Pass Invalidation**.
6. **Fake-Blender Micro-Slice / Sphere Abuse Cap (`3.1c`, `G11`):** Slicing organic curves, domes, or horns into `>= 6` thin `< 1.0u` wafer steps or `> 8`-band fake spheres/cylinders instead of using proud `16 px` cuboids + pixel-relief carving -> `Item 5 = 0`, total capped `<= 14/28`.
7. **Toy Stilt-Leg / Outer-Sticker Massing Cap (`3.1c`, `G12`):** Holding a heavy grounded quadruped/brute/elder torso high on skinny vertical stilt legs (`belly_clearance > 0.58x` torso top) OR pasting un-sunken Lego-brick stickers onto flat walls instead of `40–85%` sunken compound mass-packing -> `Item 1 = 0`, `Item 9 = 0`, total capped `<= 14/28`.

## 14. Distilled failure modes, master artist habits & test-flight lessons

### 14.1 Twenty-five recurring LLM / Block-DSL traps (pre-submission checklist)
Check every pass against these twenty-five structural, visual-reasoning, mood, wing-bounding, biomechanical animation, and painterly traps before presenting renders:
1. **Wedding-cake `yaw = 0` snout:** stacking narrower axis-aligned boxes along `-Z`. *Fix:* `±14-22.5°` yawed maxilla & mandible V-rails + sloped nasal bridge (`G2`).
2. **French-fry / organ-pipe crown:** packing `6-12` parallel horns or crystals across the skull roof. *Fix:* `1` outward-splayed (`18-32°`) primary pair on child bones + `<= 1` divergent (`>= 20°`) secondary pair preserving open sky (`G3`).
3. **Mecha-locker chest & Pez chimney neck:** vertical rectangular torso + narrow vertical neck post + dangling arms on upright bosses. *Fix:* S-curve predatory forward chest lean, `>= 1.6:1` chest-to-waist V-taper, keeled sternum, trapezius shoulder-to-neck gussets, and abducted/flexed clawed forearms (`G5`).
4. **Gundam hip-skirts & snowshoe boots:** flat vertical flank slabs (`roll ~ 0`) pasted outside thighs and thin sole boards under vertical shins. *Fix:* pitched/rolled proximal `meat` hamstring/quad wedges tapering to a tight knee, swept digitigrade hocks, and tall arched paws (`G6`).
5. **2D cardboard-plank wings, armpit walls & V-gapped strip membranes:** coplanar wing bones reading as a line in `W1 top`, rectangular armpit walls, or narrow rectangular finger strips (`chord < fingertip_gap`) leaving empty V-shaped sky gaps between splayed fingers. *Fix:* 3D vaulted humerus-forearm-phalanx camber (`>= 5-10%`), full-chord (`>= 1.08x tip_gap`) nested-fan sheets (`G10`), exact bay-polygon hard-alpha mask (`10.3b`), and a deep concave catenary armpit arch (`G4`).
6. **Stick-limb shafts & clone-peg toes on bosses:** bare 1-to-2 box rectangular arm/leg/wing shafts and flat brick paws with 3 parallel 1-bone stub toes. *Fix:* 3-layer core + stepped chamfer/muscle swell + coherent vambrace/scute rows (`3.1b`, `G8`), and 4–5 splayed multi-bone phalanx chains + thumb/velociraptor dewclaw (`8.3b`, `G9`).
7. **Template snout & `replace()` variant homogenization:** pasting one square-nosed wolf/croc muzzle across beaked/visored species or string-scaling box numbers for variants. *Fix:* select from the 6 bespoke Cranial Archetypes and build divergent variant anatomy from scratch (`5.1b`).
8. **Additive coordinate patching:** bolting strips and pegs onto old coordinates after critique. *Fix:* delete and rebuild the failing region fresh from explicit centres (`6.5c`).
9. **Grey/black shading, plastic concentric rings & per-box UV resets:** scaling RGB brightness linearly, outlining every face border symmetrically, or letting stripes/scales reset at every bone seam. *Fix:* unlit `25%` macro block-in first, hue-shifted 5-step ramps with saturated shadows, broken organic directional strokes crossing 3D seams, and manual anti-aliasing (`10.1-10.2`).
10. **Root-drifting locomotion & flashing hidden faces:** baked forward root translation in `.bbmodel` clips or unpainted inner thighs/armpits/mouths. *Fix:* zero `root.x/z` position keys while retaining `root.y` heave (`11.5b`), and run the T4 joint-rotation hidden-face audit (`10.1`).
11. **Isolated-limb animation & frozen billboard wings (zero ecological balance):** animating only legs or one slashing wing while leaving `pelvis_root.rotation`, opposite limbs, neck, and tail frozen at `[0,0,0]` or keeping wings stuck straight up during calm idle/walk. *Fix:* study YouTube ecology research footage first, fold wings into a SAT-clear Knight-Cloak drape during calm idle/walk/sit, and couple all 6 subsystems with opposite-side angular-momentum tail counterbalance (`11.5d`).
12. **Bulging / leaking multi-face eye cuboids:** centering an eye box on the front-outer skull corner and painting red iris pixels on `south` + `west` + `east` so eyes visually leak out past the brow and cheeks. *Fix:* recess the socket flush inside the lateral wall, paint all 5 non-lateral faces dark obsidian socket, and paint a 4-layer dragon eye (`vertical dark slit pupil + catchlight`) strictly on the single outward lateral face (`10.2c`).
13. **Extreme snout distortion (dangling parrot-hook, Pinocchio needle, stubby pig-box, exterior barcode teeth, or clunky square tooth cubes):** pitching beak tips `35–50°` straight down, stretching a skinny snout `> 2.1×` past a stubby lower jaw, over-shortening into a blunt square pig-nose (`< 1.55×`), painting `1 px` black/white zebra teeth across the flat exterior cheek wall, or poking chunky square tooth cubes through the cheek/lips while scattering speckled pixel noise across the jaw floor. *Fix:* apply the **6-Plane Cranial, Alpha-Perimeter Teeth & Trait-Priority Essence-Distillation Framework** (`references/cranial_and_jaw_structural_reasoning.md` §1 & §1B): match `head_L/H` to official side/front/roar renders (`~1.70–1.85` for aquiline paladin dragons), fuse the anterior prow cleanly with `88–92%` length-matched U-trough lower jaw, recess **100% binary hard-alpha serrated perimeter tooth-row planes** (`upper_teeth_row`, `upper_front_teeth`, `lower_teeth_row`) inside the inner jaw rims around a **smooth recessed 3D muscular tongue** + rear cheek webs + clean shadowed dark-plum oral cavity (directly matching User Anchors `06` & `09`), and follow the 3-tier **Trait-Priority Law** (keep iconic soul, translate medium details into native Blockbench idiom, deliberately drop high-poly micro-noise).
14. **Cartoon "primary-color" metals & blown-out white wing cores:** painting silver baby-blue, gold banana-yellow, and wing centers pure white (`> 235`), or boxing every pauldron with uniform `yy == 0` Lego stripes. *Fix:* hold `85%` of armor at `20–42%` weathered pearl-platinum / antique champagne-gold / smoked war-bronze saturation, cap wing nebula cores at moody periwinkle, paint organic `U`-scallop dragon-scale shingles, and pass the mandatory 4x ultra-zoom 6-angle per-region audit (`10.2c`).
15. **Baking environmental VFX / ambient creatures onto creature albedo (the Qurio trap):** misreading floating key-art particles, ambient creatures (e.g., *Sunbreak* Qurio dots floating in front of a chest), or lighting flares as permanent gems/markings on armor. *Fix:* verify every focal accent across `>= 2` neutral references (concept art + calm ecology video) and strip all non-anatomical VFX noise (`references/cranial_and_jaw_structural_reasoning.md` §2).
16. **Mid-air floating forelimbs & un-audited Blockbench ground contact:** relying only on global scene `y_min = 0` (which passes even when front paws float `+6u` in mid-air while hind feet touch `0.0`!), OR flipping elbow/knee flexion sign during bows/sits so paws hover `+17u` or punch `-10u` underground in Blockbench. *Fix:* enforce the **4-Foot Ground-Locking Gate** (`|y_min(FL, FR, HL, HR)| <= 0.30u` in static rest pose; continuous per-limb ground audit across all 16 clips AND directly on exported `.bbmodel` FK; `references/cranial_and_jaw_structural_reasoning.md` §3).
17. **Glitchy proximal inner wing membranes & pelvis-to-tail shard leakage:** painting bright circular bullseyes/white blobs or noisy bright-yellow jagged borders onto inner torso-adjacent wing sails (`wing_inner_sail`, `wing_fore_patagium`), or mounting flat `pelvic_skirt_fin` planes that stab across the pelvis midline and poke through `tail_1`. *Fix:* use 3-sided organic bone+flank AO, curved catenary edges, and velvety dark plum-indigo trailing hems on proximal wings; delete midline-crossing pelvic fin planes and use outward-hugging solid hip tassets (`references/cranial_and_jaw_structural_reasoning.md` §4).
18. **Batch-loop fake passes & `28/28` self-congratulation:** generating `W1..W5` or `T1..T5` in one script loop and rubber-stamping `28/28`. *Fix:* stop after every pass for visual reference-crop critique (`>= 3` fixes per pass, `5.2b`) and enforce the automatic `<= 14/28` score caps (`13.2`).
19. **Accidental "Cute & Dumb" Boss Syndrome (Mood Drift from `M4` → `M1/M2`):** building a ferocious legendary Elder Dragon with an upright giraffe neck (`-40°/-22°`), straight stiff stilt legs (`belly_clearance > 0.44`), vertical `85°` bunny-ear horns, a flat bright-white square marshmallow nose, hidden eyes under a flat brow, or chalky low-contrast face values — OR over-correcting an eye into a giant tall vertical red brick. *Fix:* run the Turn-1 **Small-AI Intent Router (`§1.5`)** & **10-Knob Mood Control Matrix** (`references/aesthetic_mood_spectrum_and_intent_router.md`): coiled predator crouch (`belly_clearance 0.26–0.36`), forward-surging S-neck + downward Kubrick-stare chin tuck, plunging golden `V`-brow (`+12..+22°` pitch, `-22..-28°` roll), compact **sleek horizontal slit hawk-eye (`2.0u` tall)** angled `20°` binocularly inside a pitch-dark orbital mask, swept-back scimitar horns (`40–55°`), razor-narrow arrowhead prow, and deep Gothic Chiaroscuro jaw/cheek shading!
20. **Blind Fractional Reference Cropping (Zooming Into Sky/Wings Instead of Head & Eyes):** guessing hardcoded image crop rectangles (`0.30..0.72, 0.02..0.44`) without first locating the anatomy on the full image, causing the model to zoom into sky, wings, or neck and mis-reason eye shape and placement. *Fix:* strictly enforce the **2-Pass "Locate-First, Then Zoom" Protocol (`§5.1`)**: Pass 1 overlays a `0.0–1.0` coordinate grid across the whole official image/video frame to read exact `(x, y)` head & eye locations visually; Pass 2 crops Head Macro + Eye Ultra-Zoom around those verified coordinates and inspects both via vision before sculpting!
21. **Splitting Innermost Wing Finger (`wing_l_f4` / `wing_r_f4`) & Unbounded Wing Membranes During Motion:** parenting `wing_l_f4` onto a high-yaw `wing_l_wrist` separately from the wing arm, OR letting `wing_l_f3` / `wing_l_f4` diverge during cloak folds so `f4` splits out or `wing_bay_3` overshoots behind `f4_spar`. *Fix:* attach `wing_l_f4` / `wing_r_f4` directly onto `wing_l_fore` (`wing_radius`), lock `wing_l_f4` rigidly coplanar with unified `wing_bay_4` (`[0, 0, 0]` relative to `wing_l_fore`), fold outer fingers gently (`f1 -> f2 -> f3` without overshooting `f4_spar`), and distribute downward wing-cloak carriage cleanly onto `wing_l_upper` (`§8.4d` #3).
22. **Detached Double-Layer Inner Wing Membranes Close to Torso:** stacking three separate overlapping inner wing sheets (`wing_inner_sail` + `wing_fore_patagium` + `wing_bay_4`) at arbitrary angles (`-26° / -36° / -42°`), creating a visible `4–24u` floating double-layer step, color jump, and Venetian-blind edge-on cross-section beside the torso. *Fix:* delete redundant `wing_fore_patagium`, unify `wing_bay_4` directly on `wing_l_fore` (`wing_radius`, `layer: 0`) at exact coplanar `rx` through BOTH `wing_radius` AND `f4_spar` (`0.000u` error), edge-lock `wing_inner_sail` (`layer: 0`) along the exact 3D plane-plane intersection crease `L_crease = p0 + t·(n_bay4 × n_inner)`, partition texels strictly across `L_crease` (`s >= -0.45u`) with ONE continuous 3D world-space shader, and inspect `5×` zoomed on BOTH sides including edge-on cross-section (`§8.4d` #3).
23. **Baked Macro Entity Altitude / Forward Z-Drift & Unconstrained Keyframe Spam (Anti-Biomechanical Rigging):** baking `+34u` fake hover elevation or forward `Z` root displacement into Blockbench animations (which breaks Minecraft entity physics!), hand-typing stiff symmetric wing hinges without aerodynamic downstroke/upstroke asymmetry or dorsal anti-clap clamping, putting ground paws inside IK chains (causing `70°` paw flips and chest clipping), or omitting `bio_joint` constraints so fingers/knees hyperextend backward. *Fix:* wire the creature's skeleton + biological `SO(3)` `bio_joint` harness (`wing_anchor_locked`, `finger_flex`, `jaw_hinge`, `auto_bio_limits: true`) + `@muscles` (`Sim(3)` `scale_bulge` + `tendon_coupling`) + declarative `gait` / `flight` solvers with **`in_place: true`** (`max|root_z| == 0.00u`, local COM lift recoil `<= 4.5u`, matched `synccraft_root_motion`), `2`-bone/`3`-bone `ik_chain`s ending above flat paw/foot pads with outward poles (`< 0.02u` slip), vestibulo-ocular `head_stabilize`, and implicit-Euler `spring` follow-through (`§11.5b`, `§11.5e`, `references/unified_biomechanical_animation_solver.md`).
24. **Elder-Dragon Long-Neck Drift on Muscular Quadrupeds & Foreshortened Reversed-Horn Sign Trap (*Zinogre* RSI Catch):** cloning a `4–5` segment `34u` Elder-Dragon neck onto a heavy-forequartered Fanged Wyvern/quadruped, OR reading a foreshortened `3/4` combat screenshot and sweeping forward-curving bull/oni horns backward (`-Z`) over the mane instead of forward (`+Z`) over the brow/snout. *Fix:* enforce the **Fanged-Wyvern Compact Bull-Neck Law** (`3` deeply seated `seat=0.36` thick segments, `0.65–0.85x` head length) and the **Orthographic-First Horn Curvature, Size, Pattern & Trait Protocol** (lock `+Z` forward vs. `-Z` backward sweep + `( )` oni-lyre front bow from orthographic side/front turnarounds & isolated profile studies FIRST, assert `hb3_z > cran_z + 9.0u` mathematically, add rear `occiput_crown_l/r` transition plates + stepped dorsal corrugation ridges, and paint transverse V-grooves + lateral porous pits + dark root collar + terminal-only tip highlight; `references/cranial_and_jaw_structural_reasoning.md §7`).
25. **Whole-Image Vision Compression & Crouched Limb-Ratio Collapse (Skipping Part-by-Part Regional Ultra-Zoom!):** reading limbs, horns, or necks only from full-canvas images causes vision encoders to miss folded structures and relative segment lengths — e.g., sizing crouched dragon/wyvern rear legs to `pelvis_y` using short straight-peg segments (`hind/fore = 0.92×` when stretched into T-pose, missing the reference's real 3-segment Z-fold). *Fix:* **NEVER hardcode one default leg/neck/horn template across creatures!** Instead, always execute the **2-Pass Grid-First -> Regional Ultra-Zoom Protocol (`§5.1`)** independently on **every smaller body region** (`head/eyes`, `horns`, `neck-chest`, `forelimbs`, `hindlimbs`, `tail/wings`), read THIS creature's true fold structure & stretched-out (T-pose) segment ratios directly from the zoomed crops, and encode species-specific ratio/fold checks in `_verify_species_geometry` (`§8.3`, `references/cranial_and_jaw_structural_reasoning.md §7.4`)!

### 14.2 Master Blockbench distillation (33 YouTube tutorials & epic boss timelapses + 5 Basis Aesthetic Anchors)
Full video-by-video log in `references/youtube_master_distillation.md`. Core behavioral habits that separate top Blockbench artists from rigid procedural builds:
1. **World-scale ruler, skeleton & 12-Point Ratio Matrix before Cube #1** (Longridr, Everbloom #2, Raboy13): anchor scale on a block floor grid, trace real joint hinges, log `R1–R12` ratios (`5.1c`), and model in uniform neutral graybox -> flat material placeholder colours (Raboy13, KayaTa, TrongOBD).
2. **Pivot-first, corner-flush & joint-overlap skin** (ArtsByKev, Everbloom #2, Longridr, NeXuS): place pivots at biological hinges or prop contact/stem bases *before* rotating; snap angled torso/limb junctions corner-to-corner; extend hidden joint-overlap skin inside neck/tail sockets and stress-test extreme roar bends; check overlaps in wireframe.
3. **Tier-calibrated sculptural richness + stepped cuboid chamfering** (ArtsByKev `WC33YHWU9NU` & `0WTvd2KNRqs`, Nyux xi `321`, Wackyblocks `340`, Vijay `237–558`): keep clean `0.5u`/whole box dimensions (`G7`), step-chamfer corners ("Minecraft Circle Math"), layer 3-part boss limbs (`G8`), build 4–5 digit multi-phalanx talons + dewclaws (`G9`), and consolidate buried cubes before UV packing (`_h6xjye56Sk`).
4. **Lived-in organic tilt for clustered props vs. strict creature symmetry** (Anchor 04 Pantry, NeXuS, KayaTa): keep frames and creature anatomy square/mirrored, but tilt loose bottles, produce bundles, hanging chilies and vines `5-15°` around contact/stem pivots and build genuine recessed crate/shelf/mouth cavities.
5. **Break "plastic" procedural contours & use manual anti-aliasing** (Mickey Joe, BB-Academy): replace uniform concentric edge rings and rigid striped gradients with large organic directional strokes, zigzag fur/scale borders, selectively broken rim highlights and `1 px` intermediate stair-corner pixels.
6. **Value-carved volume & roof-tile rim-over-shadow** (Mickey Joe, Everbloom #3, Anchors 01/02/05): darken outer corners of cylindrical cuboids to fake roundness, carve bulky sub-silhouettes optically, and pair every overlapping scale/plate's `1 px` lit rim with a `1 px` dark cast shadow directly beneath it.
7. **Shadow saturation, anti-muddy midtones & boss focal glow restraint** (Everbloom #3, Linky Playz, Faentum, Kundzio): bump saturation slightly into cool shadows, repair greenish low-opacity mud, and concentrate boss emissives strictly on 1-2 gameplay-distance read zones (`<= 4%` area).
8. **Animation-safe hidden-face check, phase-lagged flight & controlled T1 feedback** (Everbloom #3, ArtsByKev `fcvKOpI3o88`, BB-Academy, Vijay): hide limbs and rotate joints to paint every concealed inner face; animate wings with forceful downstrokes, linear manual-braking peaks, `25-45°` tip phase lag and marine spine counter-pitch; adjust minor local geometry flaws immediately when T1 colour blocking reveals them (still punching back to W1 on any structural failure).

### 14.3 Empirical lessons from SyncCraft test flights
- **Measure proportions before detailing.** On a standard dragon the head is at least a third of the torso length, and the upper leg at least half the torso's depth.
- **Check every joint close up.** A jaw that floats 1.5 units below the snout can still pass the audit.
- **Colour the lower jaw like the hide,** with an underside throat plate beneath. In round 3, a cream jaw the colour of the teeth hid the teeth and read as a block sticking out.
- **Seat segments about 28% into their parents.** Shingle the joints and fan crest plates from the skull over the neck; the row of boxes becomes one form.
- **Partitioned wings fold cleanly.** Size each panel to its bay polygon and regenerate it whenever a bone changes.
- **The Squallwing run.**
  - What moved it most: a wedge snout (down 5 degrees, nasal ridge down 12), jaw open 14 degrees, horns swept back 52 degrees and splayed 34, dorsal fins graded largest at the withers, a forked tail with flat streamers.
  - Texture: low-contrast brick scale rows, dorsal-only chevrons, an eye in a dark socket under the brow.
  - Its mistake: attitude in the rest pose.
- **Mesa Glider run (owner feedback):** the head was "too flatten" (cranium 10 wide by 8 tall) and the wings "too constrained" (span 1.56x on one pterosaur finger). Those led to the depth rule in 6.1 and the wing size and "M"-arm rules in 8.4.
- **Test flights (wyvern, leviathan, jumping mouse):** all three passed every gate with these lessons:
  - **Wyvern wings:** big wings need a palm block for the knuckles and a forearm twist, not a hand roll. The inner bay is anchored on the flank, and roll waves get mirrored phase.
  - **Leviathan sails:** alternate centred and paired sails.
  - **Mouse (v1):** 32 px per block, a big high eye, forward-turned ears, fur paint, and a tail bias that respects the chain-direction pitch rule. Rebuilt in round 4 as a small creature (9.3).
- **Owner round 2 (Jadecrown v2):** the owner asked for better cheeks, freedom for larger wings, in-place locomotion and paint closer to anchors 1 and 4.
  - **Cheeks:** a cheekbone ridge, a jaw-hinge mass and a three-plate frill turned a box face into a shield. The frill root first sat inside the neck; aligning it to the cranium's side fixed that.
  - **Wings:** span went from 1.77x to 2.37x. Thicker spars forced every panel height to be restaggered.
  - **Paint:** stroke paint at half-step contrast. Full ramp steps looked like grass.
  - **Locomotion:** the delivered walk had baked a 38-unit root advance; it is now in place, and the run needed `max_frames=288`.
- **Owner round 3:** first the lower jaw looked "split out", then the head looked too wide.
  - **The jaw** had been open 10 degrees, cream, almost as long as the snout, with a chin lump under it.
    - It is now closed at rest with a 2-unit overbite, hide-coloured, with a throat plate under it and teeth hanging over a dark lip row.
    - A roar clip opens it.
  - **The head** was 1.7x the cranium width at the cheeks.
    - Fixes: cranium 10 x 11; cheekbones standing out 1.25 units; fin sweep cut from 32 to 14 degrees; the top neck segment narrowed to the cranium width.
    - The front view went from 1.15x to 0.82x as wide as tall.
  - **16 px per block** made the pixels read as Minecraft. The eye and teeth had to be redrawn on whole pixels.
  - **Spikes** became slim diamonds instead of square posts, and the cheek frill became a framed fin with a membrane.
- **Owner round 4 (small creatures and texture):**
  - **Dunehop v1 read as a toy:**
    - legs 1.5-2 units thick, and white sole slabs that looked like shoes;
    - five small boxes on the head;
    - speckled fur that looked like dirt.
  - **v2 fixes:** 16 px per block, every box at least 2 units, one-box head, painted jaw and nose, a 3 x 3 px eye, hide-coloured feet with pale toes, and strokes instead of flecks.
  - **Pebble Weaver** (an original small spider): the first build had a box-shaped body and colliding leg roots.
    - A waist box and an abdomen wider than the head made it read as a spider.
    - Thin-front-to-back femurs spaced their own thickness apart fixed the collisions.
    - The scuttle passed at 5 degrees per leg. At 6 degrees, adjacent femurs collided.
  - **Texture pass:**
    - Baked front-to-back light and contact shadows at joints separated forms on every creature.
    - Rebalancing the sea dragon's ramp removed a top step twice the size of the others.
- **Owner round 5 (epic boss, "Ashmantle Sovereign"):** the owner stopped the build mid-way: "the skill is for you develop a standard way, not frozen your art style and modeling completely… I'm seeing you always start from similar skeletons. try to allow creative too." That led to 5.0.
  - **v0** started from the wyvern skeleton: upright legs, long neck, "M" wings. It passed the gates and still read as another Jadecrown.
  - **v1** started from a concept, "the dragon is the volcano":
    - a sprawled four-legged stance, a short thick neck and a huge low head;
    - a crater of leaning basalt columns on the back with a molten pool, and a tail ending in a column cluster;
    - wings raised as a frontal plume (8.4);
    - a hide of blackstone-like chips, cool basalt columns and an ochre underside, with molten amber as the only accent.
  - **Result:** 149 elements, the 10 largest boxes hold 56% of volume (wings excluded), span 1.71x at rest and 1.80x in the roar, and idle, roar and an in-place walk all pass a 48-frame swept audit.
  - **Sprawled-limb pitfalls:**
    - feet inherit a tilt from the sprawl angles: set each foot's rest angle from its world tilt, then seat the root so both pairs of feet land;
    - the stride swings the upper limb about the vertical: a 20-degree shoulder-yaw limit made one forefoot slip 0.3 units, and 25 degrees fixed it.
  - **Paint:** ember nostrils read as a second pair of eyes from the front, and an uplight on the column faces made the crater read from the front.
- **Owner round 6 (feet, then props).**
  - The boss feet read as boards with marshmallows: a 6-unit plank foot, thin toe sticks and pale cube claws.
    - A sheet-claw version (hooked cut-out planes) was built first, and the owner asked for block style instead.
    - Delivered: a heavy 10-unit foot block blended into the leg, three thick toe blocks with gaps, and block claws painted from a dark root to an ivory point.
    - The heavier feet needed 8 degrees of foot yaw and a shorter stride (Froude number 0.038) to keep the walk slip-free.
  - **Props test flight:** Emberpick (a handheld pick), Stormglass lantern, Rustback ore cart and Cinder cauldron. All pass the static audit, and the lantern, cart and cauldron also pass their motion audits.
    - 12-47 elements each, at 16 px per block.
    - Lessons:
      - SyncCraft's `octagon` is a star, so use four-band octagons.
      - Sleeves over a shaft fail the overlap audit, so stack segments instead.
      - Glints on both glass panes form an X, so put a glint only on the outside face.
      - Log ends and glints read as eyes.
- **Geodeheart, the Amethyst Forge-Wyrm** (round 7, boss test flight; references r7a-r7i).
  - The first attempt used an invented skeleton (smith arms, a crosswise hammer tail), and the owner asked for a normal dragon skeleton. It was rebuilt as a classic upright elder with amethyst gem-dragon traits.
  - 166 elements, audit PASS, and idle, roar and walk PASS (in place, slip about 5e-6).
  - Span 2.0x nose-to-tail at rest; 12.3 blocks tall with the wings; glow area 0.1%.
  - What worked:
    - the wing module and lab (four wing types from one sketch format);
    - the knob-chain forearm;
    - the torn gradient bays with free tips;
    - a calcite chest panel with a glowing forge-heart core;
    - a 5-prism crown behind the skull;
    - 3-prism shoulder clusters;
    - one dorsal crystal row leaning one way;
    - a crystal tail tip.
  - Paint roles added:
    - *crystal*: a lit top, two facets per face on a slanted split, one bright leading edge, one glint and a dark root;
    - *knob*.
  - Fixes along the way:
    - ventral plates moved to local -y;
    - crystal seating made rotation-aware;
    - the sweep moved to the plane bone;
    - the root membrane given a half-angle bone;
    - the bronze bracers toned down so they don't compete with the crystal accent.
- **Geodeheart, round 8 (head, hands, elbow).**
  - Hands rebuilt as jointed, curled fingers with hooked block claws; the owner approved.
  - Head: two passes from "bulky" to slim.
    - The skull was reduced.
    - The muzzle became its own bevelled bone.
    - The teeth were moved under the snout after the owner saw them "squeezed out" of its sides.
    - The jaw got a back hinge and a V-shaped keel.
  - An elbow cap and spur were added.
  - Result: 190 elements; audit and the idle, roar and walk clips PASS.
  - Lesson: when the owner says a head is bulky, cut width and height first (skull, nose, ledges), then add bevels; adding more parts makes it worse.
- **Habits that large models expose (owner analysis, round 10).** Check for each one before showing a pass:
  1. Additive patching: fixing a critique by bolting boxes onto old coordinates. Rebuild the region at W1 instead.
  2. Axis-aligned bias: pitch is the only rotation used. Paired cheek, brow and snout-wall masses need yaw and roll too, or the top view stays a rectangle.
  3. "Roof plus wall" faces: a skull box with a brow slab on top and a snout box in front traps the eye on a vertical wall. Use an eye facet and a tapering wedge.
  4. Brightness-only ramps: shading by scaling one colour's brightness. Use hue-shifted ramps and a warm countershade on the jaw and throat.
  5. Edge noise per UV column: it makes vertical comb tears. Measure tears radially from the hub.
  6. Golem grids: arms and legs straight down at plus or minus the body width. Splay them and use an S-curve spine.
  7. Loop-tiled armour: N identical cubes. Use a few broad, varied, shingled plates.
- **Repo v9 diagnosis.** Visual reasoning was fine, but construction shattered every form. Most of the owner's complaints trace to that one cause: dirty, ugly, a flat long face, eyes in odd places, sheet horns, broken symmetry.
- **Owner round 11 (Malzeno & Primordial Malzeno epic boss autopsy -> v19 upgrade):** six structural & process failures exposed on high-complexity winged Elder Dragons, now permanently guarded by `5.1b–5.2b`, `8.3b`, `8.4d`, `10.3b`, `13.2`, and `G8–G10`:
  1. *Under-complex stick-limbs:* holding `top-10 >= 50%` on a 6-limbed Elder Dragon kept `172` total elements and left upper/lower arms and wing-arms as bare rectangular sticks. **v19 fix:** tier-aware `G1` (`>= 26%` on `epic_boss`), `240–550` element epic floor, and 3-layer stepped chamfer + muscle taper + vambrace/scute grammar (`3.1b`, `G8`).
  2. *Template & variant homogenization:* reusing Lagiacrus's squared-`nose_boss` box-snout on Malzeno (instead of an aquiline hooked beak + continuous brow-to-horn visor) and building Primordial Malzeno via `replace()` box scaling. **v19 fix:** 6 bespoke Cranial Archetypes and strict ban on `replace()` variant cloning (`5.1b`).
  3. *Slab feet & clone-peg toes:* `7.5x4x7.5` slab paws with 3 parallel 1-bone `2x2x4` stubs. **v19 fix:** 4–5 splayed multi-bone digits (`proximal -> distal -> hooked talon`, middle digit `1.2–1.35x`, plus thumb/velociraptor dewclaw & ankle spur; `8.3b`, `G9`).
  4. *Shallow bounding-box ratio checks:* logging a single `L/H` box metric hid internal thickness, taper, and negative-space errors. **v19 fix:** 12-Point 3D Spatial Ratio Matrix (`R1–R12`, `5.1c`) + automated telemetry in `construct_check.py`.
  5. *Gapped rectangular-strip wing bays (`0.40x` chord coverage):* `17u` rectangular planes attached along splayed fingers opened `42u` empty V-gaps at the wingtips, and early cutouts carved comb strips. **v19 fix:** full-chord (`>= 1.08x tip_gap`) nested-fan wing sheets (`8.4d`, `G10`) + exact spar-to-spar bay-polygon hard-alpha mask & 5-layer both-spar bone-AO / translucent-belly shading (`10.3b`).
  6. *Batch-loop fake passes & `28/28` self-congratulation:* running `T1–T5` in one script loop and self-scoring `28/28`. **v19 fix:** mandatory per-pass visual crop inspection (`>= 3` fixes per pass, `5.2b`) and automatic `<= 14/28` score caps (`13.2`).
- **Owner round 12 (Anchors `06–09` Vigorous Carved Aesthetic, 6-Video Native Multimodal Ingestion & SyncCraft Decoupling -> v20 upgrade):** cured both the **"toy-like procedural Blockbench"** failure mode and the **"fake-Blender curve-slicing"** failure mode (`references/vigorous_carved_aesthetic_and_spatial_reasoning.md`):
  1. *Blockbench Is Blockbench (Anti-Blender & Anti-Toy Law):* codified proud `16 px/block` planar cuboids (`0` wafer-sliced fake curves `G11`) combined with **`40–85%` sunken compound cuboid mass-packing** (`Darto 3d Rathalos` `01:08` sunken thigh scutes, `MHCraft Deviljho` `03:26` / `Anchor 07` embedded jaw osteoderms, `TrongOBD Zelno` `02:15` `30–45°` edge-locked bevel clusters).
  2. *Coiled Predatory Weight (`G12`):* anchored low-slung spine-first massing (`belly_clearance <= 0.42x shoulder`, `Anchor 09` Gore/Elder, `Anchor 08` Tetranadon `< 0.18x`, `Kundzio Hammerhead` `01:28`) to eliminate tall stilt-horse legs.
  3. *Vigorous Cross-Cube Pixel-Grid Relief Carving (`10.2b`):* distilled Kundzio's `45–50°` hue-shifted `Generate Gradient Palette` + rhythmic `2–3 px` transverse chevron/scale-rung wraps (`3–5 px` down side faces), `Anchor 09` corrugated wing-arm ladder-ribs, `Anchor 06` Greek-key horn runes & 3-tier alpha feather planes, and `Anchor 08` vanilla mossy-stone clustering.
  4. *6-Stage Vision & 3D Spatial Reasoning Curriculum (`S1–S6`, `5.1d`):* formalized an explicit intermediate spatial representation pipeline transferable to small-model vision-spatial training and human-artist co-pilots.
  5. *SyncCraft Experimental Environment Decoupling (`11.0`) & Cross-Format `construct_check.py` (`G1–G12`):* diagnosed `sync_audit.py`'s over-harsh `0.25` parallel / `0.65` burial / `place(anchor=±1)` sticker biases alongside its loose stick-limb blind spots; upgraded `construct_check.py` to audit native **`.bbmodel`, Bedrock `.geo.json`, and `.blockdsl`** directly.
- **Owner round 13 (Primordial Malzeno Blockbench Live Inspection -> `v22` Cranial/3D Maw, Ground-Locking, VFX Isolation & Seamless Inner-Wing Upgrade):** live inspection in Blockbench exposed 5 subtle master-level flaws invisible to single-number audits, now codified permanently in `references/cranial_and_jaw_structural_reasoning.md` and Traps `#13, #15–17`:
  1. *VFX-vs-Anatomy Isolation (Qurio trap):* removed mistaken red dots on `pec_shield_upper` (floating environmental Qurio creatures in key art, NOT armor gems).
  2. *4-Foot Blockbench Ground-Locking:* checking only global `y_min = 0` hid `+6.21u` mid-air front paws in static rest and `+17u` hover / `-10u` sink in animations; fixed via simultaneous `FL/FR/HL/HR <= 0.30u` static grounding + corrected elbow/knee flexion sign audit across all 16 clips AND direct `.bbmodel` FK verification.
  3. *6-Plane Cranial, Master Blockbench Alpha-Perimeter Maw & Trait-Priority Distillation (`W11+T8`):* replaced needle/stubby/barcode snouts and clunky square tooth cubes with Capcom official-matched (`pm_04`, `pm_06`, `pm_09`, `pm_11`, `pm_15`) `head_L/H = 1.73` armet-visor proportions, `90%` length-matched U-trough lower jaw, real 3D hollow oral chamber housing a smooth recessed 3D `jaw_tongue` + rear `jaw_cheek_web` tendons + **recessed `100%` binary hard-alpha serrated perimeter tooth-row planes** matching User Anchors `06` & `09` (`0` exterior tooth clutter when closed, dramatic roaring maw at `38°`), plus the generalizable 3-tier **Trait-Priority & Essence-Distillation Law** (§1B) teaching AI modelers which traits to preserve, translate, or deliberately drop.
  4. *Proximal Inner-Wing Continuity:* replaced boxy armpit patagium slabs, stray white circular bullseyes, and noisy yellow cutout outlines with 3-sided organic bone+flank AO, smooth catenary edges, and velvety dark twilight-plum trailing hems.
  5. *Zero Pelvis-to-Tail Midline Leakage:* deleted `pelvic_skirt_fin` planes that stabbed through `tail_1` and replaced them with outward-hugging armored hip tassets.
- **Owner round 14 (Blockbench Official Zero-Thickness Teeth & Tiny Vigorous Micro-Traits -> `v23` Upgrade):** live review of Anchor `06` & the Blockbench Official Style Guide caught a classic low-res UV alpha bug (`tip_t <= 0.96` silently erasing `h = 2 px` tooth-tip rows!), a `3.5u` toothless cheek-to-front gap, bubblegum-pink tongue saturation, and a stark black+white nostril reading like a cartoon googly eye — permanently fixed in `W12 + T9` and codified into `SKILL.md` §6.5 & §10.2d + `references/cranial_and_jaw_structural_reasoning.md` §1 & §5:
  1. *Zero-Thickness 6-Plane Single-Pixel-Square Tooth U-Arcade:* adopted the Blockbench Official Style Guide ban on chunky volumetric tooth cubes; built continuous `0`-thickness flat `plane` tooth rows (`upper_teeth_row` L/R `10×2 px`, `upper_front_teeth` `4×2 px`, `lower_teeth_row` L/R `8×2 px`, `lower_front_teeth` `4×2 px`) painted with **explicit integer-pixel `1×1 px` ivory squares + `1×2 px` prominent canines** staggered `1 px` for interlocking bite.
  2. *4-Layer Cavernous Mouth & Articulated 2-Segment Tongue:* added recessed abyssal pharynx backing (`mouth_throat_back`), `1 px` transverse upper palate rugae, dark medial `maxilla_wall`/`mandible_bar` faces, `0`-thickness concave hard-alpha `plane jaw_cheek_web` rear sinews, and an articulated `bone tongue` (`tongue_base` + arched `tongue_tip`) in royal smoked plum-crimson (animated during roars!).
  3. *Tiny Vigorous Craft Details (`§10.2d`):* codified subtle anterior dark-slate nostrils (banning cartoon black-below-white googly-eye nostrils), orbital lower-eyelid rim catchlights on `zygoma_shield`, sparse `1 px` honed bevel glints, keratinized root collars, and scale-overlap cast shadows.
- **Owner round 15 (Aesthetic Mood Spectrum `M1–M4`, Small-AI Intent Router & 2-Pass Locate-First Reference Zoom -> `v24` Upgrade):** live evaluation identified two fundamental AI-modeling breakthroughs: (a) a technically clean boss dragon still read "a bit cute and a bit dumb" due to upright giraffe-neck posture, vertical bunny-ear horns, flat bright-white nose cube, chalky face lighting, and hidden eyes; and (b) blindly guessing fractional reference crop boxes zoomed into sky/wings instead of the head & eyes, causing an over-tall vertical red eye block on the first ferocity attempt. Permanently solved in `W14 + T11 + A8` and codified into `references/aesthetic_mood_spectrum_and_intent_router.md`, `SKILL.md` §1.5 / §5.1 / Traps `#19–20`:
  1. *4-Tier Aesthetic Mood Spectrum & Small-AI Intent Router (`D1–D4 × M1–M4`):* lets a small AI model classify any artist task (from cozy `D1` blocks/items/weapons & `M1` cute critters to `M3` realistic beasts & `M4` ferocious legendary bosses) on Turn 1 and apply 10 quantitative geometric/chiaroscuro control knobs.
  2. *Mandatory 2-Pass "Locate-First, Then Zoom" Visual-Spatial Protocol:* Pass 1 overlays a `0.0–1.0` coordinate grid over full official renders & in-game screenshots (`pm_01`, `pm_04`, `pm_06`, `pm_10`, `pm_11`, `pm_14`, `pm_15`) to read exact `(x, y)` head & eye locations visually; Pass 2 generates verified Head Macro + Eye Ultra-Zoom sheets before sculpting.
  3. *Verified Hawk-Dragon Horizontal Almond Eye & Coiled Apex Predator Transformation (`W14 + T11`):* replaced the `3.0u`-tall vertical red eye block with a sleek `2.0u`-tall horizontal blood-ruby almond eye (`5×2 px`, dark hawk-brow upper eyelid shadow + crisp jet-black pupil + intense gem-like ruby iris) nestled seamlessly beneath a continuous golden `V`-brow (`+14°/-22°`) and swept silver cheek blade, paired with `-40°` battle-charred swept scimitar horns, razor-narrow `2.0u` falcon prow, Gothic Chiaroscuro jaw/cheek shading, and a coiled 4-foot-grounded (`FL=0.02, HL=0.00`) panther-dragon prowl (`belly_clearance = 0.36`).
- **Owner round 16 (Unified Coplanar + Shared-Crease Inner Wing Membrane & Rigid `wing_l_f4`/`wing_r_f4` Attachment -> `v25` Upgrade):** live Blockbench inspection zoomed onto both wings in motion and right beside the torso exposed two subtle membrane rigging failures: (a) `wing_l_f4` / `wing_r_f4` splitting out from the wing arm while moving; and (b) two detached overlapping inner-wing membrane layers (`wing_fore_patagium` + `wing_bay_4` + `wing_inner_sail` at `-26°/-36°/-42°`, creating a `4–24u` floating step right beside the torso). Permanently solved in `W16 + T12 + A9` and codified into `§8.4d #3` and Traps `#21–22`:
  1. *Rigid Wing-Arm Attachment of `wing_l_f4`/`wing_r_f4` + Exact Coplanar Unified `wing_bay_4`:* attached `wing_l_f4`/`wing_r_f4` directly onto `wing_l_fore` (`wing_radius`), deleted redundant `wing_fore_patagium`, and mounted a single unified `wing_bay_4` (`layer: 0, rotate: [-20.58, 0, 0]`) directly on `wing_radius` passing through BOTH `wing_radius` AND `f4_spar` to `0.000u` perpendicular error so `f4` and its membrane can NEVER split out in any clip.
  2. *Exact `0.000u` Plane-Plane Intersection Elbow Crease + Unified 3D World-Space Shader (`T12`):* edge-locked `wing_inner_sail` (`layer: 0, rotate: [-13.27, 0, 0]`) to meet `wing_bay_4` at `0.000u` along their exact 3D intersection line (`10.39°` natural elbow dihedral), partitioned hard-alpha cutouts strictly across the crease (`s >= -0.45u`, ribcage edge seated flush inside `chest_core`), and shaded both planes with ONE continuous 3D world-space shader (`delta_RGB = 0` across the crease).
  3. *Motion-Sealed Wing Choreography & `5×` Multi-Angle Verification:* balanced cloak-drape roll onto `wing_l_upper` (halving elbow kink in `idle_noble`/`walk_patrol`/`ecology_royal_sit` and stopping `wing_bay_3` overshoot behind `f4_spar`), verified via `5×` zoomed renders on BOTH sides (front-3/4, rear-3/4, ventral underside, dorsal top-down, edge-on cross-section, and 6 moving poses).
- **Owner round 17 (Unified Universal `SO(3)` / `SE(3)` / `Sim(3)` Biomechanical Animation Solver & Minecraft Entity-Space Separation -> `v26` Upgrade):** live animation review exposed that baking macro root translations (`+34u` fake hover height, forward `Z` drift) conflicts with Minecraft's entity movement system, while manually keyframing limbs/wings feels mechanical and breaks across arbitrary creatures. Permanently solved at the **engine level inside `synccraft_v20`** and codified into `§11.5b`, `§11.5e`, Trap `#23`, and `references/unified_biomechanical_animation_solver.md`:
  1. *Universal Geometry-First Skeleton & Muscle Solver (`bio_constraints.py`, `morphology.py`, `flight_solver.py`, `gait_solver.py`):* works on ANY real animal or brand-new imaginary creature regardless of limb/wing/head/tail count or bone names — proven across a 5-archetype test suite (`34/34` unit tests OK: 2-leg Humanoid with `finger_flex` + automatic arm counter-swing; 4-leg Quadruped with `Sim(3)` `scale_bulge` exported to `.bbmodel` & Bedrock; 8-legged Imaginary Alien with zero English bone names running `metachronal_wave` + `auto_bio_limits`; 0-leg Serpentine Sea-Wyrm running `serpentine_wave`; and 4-winged 2-headed Imaginary Seraph-Hydra running multi-pair `flight hover` at `0.0000` asymmetry and `0` collisions).
  2. *True Aerodynamic Airfoil Physics + `in_place: true` Minecraft Contract:* replaced rigid manual wing flaps and baked `+34u` hover offsets with asymmetric Figure-8 aerodynamic `flight` (`hover`, `cruise`, `glide`, `soar`, `takeoff_burst`, ground-locked `mantle`, `swim_fins`), 3D angular Jacobian axis calibration, bind-pose dorsal anti-clap clamping, `wing_anchor_locked` membrane sealing, and native `in_place: true` entity-space export (`max|root_z| == 0.00u`, local COM lift recoil `<= 4.5u`, exact matched `synccraft_root_motion`).
  3. *Live Base Malzeno `v26` Proof (`12/12` Clips `PASS`, `0` SAT Collisions, `23%` Leaner DSL):* upgraded Base Malzeno (`malzeno_v26_biomech.blockdsl`, `62.7 KB` vs `81.7 KB` legacy) to declarative `@muscles` (`Sim(3)` ribcage `scale_bulge: 0.032` + jaw-to-tongue `tendon_coupling`) + 2-bone/3-bone outward-poled `ik_chain`s (`walk_prowl` slip `0.000002u`, `run_blood_charge` slip `0.0004u`, `wing_mantle_threat` slip `0.00000u`) + aerodynamic `flight` + `head_stabilize` + damped `spring`s — verified by `construct_check.py --tier epic_boss` (`rc=0`), direct `.bbmodel` audit, and 4 multi-frame visual filmstrips including `5×` inner-wing aero seam zoom.
- **Owner round 18 (*White Fatalis `v28–v30`* Cranial/Claw/Sloped-Joint Continuity + *Zinogre `v30–v31`* Manual RSI Test Flight):** two consecutive boss builds distilled 5 enduring craft and biomechanical laws into `§3.1d`, `§5.2b`, `§7.1`, `§8.1`, `§8.3b–8.3c`, Trap `#24`, `references/cranial_and_jaw_structural_reasoning.md §6–7`, and `references/unified_biomechanical_animation_solver.md §7B #9–10`:
  1. *Multi-Block Cranial Sculpt, 1-Block Recessed Eye Shelf & Sloped Joint-Continuity Bridges (`v28–v30`):* replaced single V-planks with 3-block supraorbital arches + 4-step snout/solid U-mandible, recessed the orbital skull `1` block inward (`0.00u` front eye extrusion, `0%` side embedding), carved compact arched dragon claws (`<= 4.5–5.0u`), and bridged `90°` wrist/paw, ankle/foot, hock, elbow, and ventral spine angles with half-angle sloped gusset cuboids (`min_edge >= 2.0u`, `0` SAT collisions).
  2. *Fanged-Wyvern Compact Bull-Neck vs. Elder-Dragon S-Neck (`Zinogre P1 -> W1`):* caught and cured Elder-Dragon long-neck drift by compacting *Zinogre*'s neck from `4` long segments (`34u`) to **`3` deeply seated (`seat=0.36`) ultra-thick bull-wolf segments (`~12u` net reach)** emerging from a `25u` barrel chest (`chest:waist = 1.52`, `belly_clearance = 0.30`) under a swept-back 3D white shockfur mane.
  3. *Orthographic-First Forward-Curving `( )` Oni-Lyre Horn Protocol (`Zinogre W2 -> W4` + `T2–T3`):* caught foreshortened combat-screenshot horn reversal and codified verifying `+Z` forward vs. `-Z` backward horn curve on orthographic side/front turnarounds first (`assert hb3_z > cran_z + 9.0u`), pairing forward-curving corrugated golden horns (`hb_1..hb_tip` + stepped `hb_ridge_1..2`, `horn_tip_sep = 11.22u`) with rear `occiput_crown_l/r` transition plates, transverse V-chevron grooves, lateral porous Thunderbug pits, dark root collar, and terminal-only honed tip.
  4. *Human-Artist Multi-Iteration Stage Discipline (`4 W` + `3 T` + `2 A` passes):* enforced `>= 3` White-Model iterations, `2–3` Texture iterations, and `>= 2` Animation iterations with visual inspection and `>= 3` concrete fixes between every iteration.
  5. *3-of-4 Selective Standing Limb Pinning, Multi-Joint Fold-Preserving N-Bone IK & Anti-Frog-Splay Sagittal Purity (`Zinogre A1 -> A6` & `White Fatalis v32`):* proved `ground_pin_limbs: [fore_r, hind_l, hind_r]` / `[fore_l, hind_l, hind_r]` for true outward-overhead single-forepaw slams (`+18u` raise + flush landing), haunch-brace `pelvis_root.y < 0` on skyward howls, erecting dorsal `shocker_*` plates, and axial-only `@muscles` (`0.0000u` stance slip across all clips). Caught and fixed a fundamental multi-joint IK frog-leg bug where averaging opposing Z-fold joints (`0.5*(knee + hock)`) plus `posture="sprawled"` / `±20°` knee roll twisted thighs/shins `-24°..-34°` sideways even at idle — upgrading `solve_n_bone_ik` to transport every interior joint independently via minimal `SO(3)` chord swing (`0.000°` idle drift, `< 0.3°` knee/hock off-axis twist in walk/gallop/roar), tightening anatomical hinge off-axis limits to `±8°`, preserving `SE(3)` wrist/ankle splits on explicit `ik_chain`s, and aligning both models' rear legs into clean vertical-sagittal Z-folds with elevated pelvises (`~23.4u`)!
  6. *Universal Part-by-Part Grid-Then-Ultra-Zoom Spatial Reasoning & T-Pose Limb-Ratio Audit (`White Fatalis v31–v32` & `Zinogre v31–v32` Catch):* user T-pose and animated front/rear inspection exposed both short straight-peg hindlimbs (`hind_tot / fore_tot = 0.92×` on White Fatalis, `0.98×` on Zinogre) AND excessive outward frog-splay during Z-fold animation. Codified the **Universal 2-Pass Grid-First -> Regional Ultra-Zoom Protocol across ALL body parts** (`§5.1`, `§8.3`, Trap `#25`) — explicitly teaching models to derive each species' own side-profile fold topology, front/rear limb plane, and stretched T-pose ratios from zoomed regional crops rather than hardcoding one default leg/neck/horn template!
- **Owner round 19 (6-Archetype Special-Skeleton Stress-Test Campaign -> `v38` Universal Multi-Skeleton & Underwater Musculoskeletal Edition):** moved completely outside classic quadrupeds/Elder Dragons by building and animating 6 distinct *Monster Hunter* skeletons end-to-end (**Lagiacrus** sprawled amphibious Leviathan, **Legiana** true 4-limbed flying wyvern with zero forelegs, **Wintermoon Nettle** 0-leg radial pelagic jellyfish, **Puppet Spider** 8-legged articulated arachnid, **Wirebug** 6-legged luminous winged insect, and **Sandstar** agile bipedal hopping desert jerboa) with live owner reviews driving 9 permanent engine, validator, and craft upgrades:
  1. *Native SyncCraft Underwater Buoyancy & Musculoskeletal Swim Positioning (`swim { ... }` in `swim_solver.py`):* added declarative underwater swimming (`leviathan_cruise`, `leviathan_hover`, `fish_subcarangiform`, `eel_anguilliform`, `manta_mobuliform`, `pelagic_pulse`) with neutral buoyancy (`buoyancy_y`), hydrodynamic limb positioning (`hydro_tuck`, `paddle_row`, `pectoral_glide`, `trail_drift`), traveling spine/tail carangiform waves with `head_stabilize`, fin phase-lag, and **14+ lateral `Sim(3)` myomere scale contractions**, fixing bilateral world-Jacobian symmetry (`FL == FR`, `HL == HR`), `base_pose_deltas` muscle composition, mirrored `angle_sign` clamping, and units/sec root-motion export!
  2. *Zero-Blink `0.5u` Dual-Sided Membrane Slabs & Continuous Butterfly-Diamond Tail Bonding (Owner Legiana Review):* painting opposite dorsal/ventral textures on zero-thickness `0.0u` planes causes severe OpenGL Z-fighting blinking in Blockbench 5 / Minecraft, and edge-tacked petals detach in flight. Solved permanently via **solid `0.5u` dual-sided `box` slabs** (`up`/`down` full-res `32x` UV rects, transparent `alpha=0` corner `[0,0,1,1]` on the 4 thin `0.5u` side rims, alternating `±0.55..0.70u` normal stagger between overlapping bays/slabs), deep `1.8–2.8u` burial inside parent spars/vertebrae, `6–8.5u` consecutive tail-slab overlaps carved in continuous world-tail Z space, and true official Capcom radial moth-wing & star-vein dorsal/ventral pattern fidelity.
  3. *Periorbital Sightline Window Contract (Owner Lagiacrus Eye Review):* counting nearby periorbital boxes (`G14`) does NOT guarantee eye visibility! Codified unobstructed line-of-sight windows across `0°` Dead-Front, `35°` Front-3/4, and `85°` Side views (`maxilla_top_y <= eye_min_y + 0.5*eye_h`, `cheek_top_y <= eye_min_y + 0.15u`, `brow_bottom_y >= eye_max_y - 0.28*eye_h`, proud orbital shelf `|eye_xmin| > |maxilla_xmin|`).
  4. *Right-Handed `+Z` Toe-Splay Sign Law, Isolated Per-Toe `sculpt_group`s & Non-Fandom 4K Extremity References (Owner Legiana Claw Review):* on a `-X` left foot pointing `+Z`, `Ry(+θ)` fans **medial (`+X`)** and `Ry(-θ)` fans **lateral (`-X`)** (`toe_in_l: +24°`, `toe_out_l: -34°`). NEVER share one `leg_sc` across sibling toes (which silently hides crossed-toe intersections from SAT!); isolate per-toe `sculpt_group`s (`toe_in_sc`, `toe_mid_sc`, `toe_out_sc`, `toe_hal_sc`), assert programmatic tip divergence (`out_x < mid_x < in_x`), and download high-res non-Fandom references (ZBrush/artbook/Capcom Figure Builder) directly for occluded extremities.
  5. *Right-Handed Wing-Sweep / Tail-Pitch Sign Laws & Coplanar Leading-Edge Counter-Offset:* on `-X` wings with trailing `-Z` sails, `Rx < 0` tilts sails UP away from pelvis/thigh, `Ry > 0` sweeps FORWARD away from tail, and `Rz < 0` raises UP; on posterior `-Z` tails, `+Rx` lifts UP (`+Y`); leading-edge vertical counter-offset is `+0.5 * align_z * chord * sin(rx)`.
  6. *Sprawled Leviathan & Digitigrade Raptor Kinematics Fixes (`bio_constraints.py` + `whole_body_solver.py` + `artisan_painter.py`):* added `elbow_sprawled` (auto-selected on `posture="sprawled"`), rest-pose-aware `hinge_knee`/`hinge_elbow`/`hinge_hock` delta bounds (fixing pre-bent digitigrade raptor & jerboa knees), `elytron_hinge`, `arthropod_chelicera`, `arthropod_pedipalp`, `pelagic_bell`; excluded digit tokens from `whole_body_solver.py` `foot_hand_bone` hub selection; fixed `artisan_painter.py` `uv_scale` scaling and `normal: y` V-axis back-face mirroring; fixed `backends/common.py` `data_url` auto-encoding.
  7. *Non-Quadruped Endemic Archetype Craft Laws (`Wintermoon Nettle`, `Puppet Spider`, `Wirebug`, `Sandstar`):* use **4-band flat-sided octagons** (`0°, 45°, 90°, 135°`) for pelagic bells/caps and `leaf = name.split("__")[-1]` shader dispatch; use coplanar vertical radial IK planes on 8-legged arachnids (`0.000°` idle drift) with `hexapod_tripod` / `metachronal_wave`; separate rigid V-open `elytron_hinge` wing cases from fast `12 Hz` flapping hindwings and use topological tree-depth spine ordering on C-curved insect abdomens; auto-suppress `±32°` arm counter-swings on `< 0.35×` vestigial paws (`biped_with_free_arms`) and give small micro-landmarks (`uv_scale: 2` nostrils/eyes) crisp resolution.
  8. *Upgraded Universal Multi-Skeleton Validators (`construct_check.py` + `verify_bbmodel_ground.py`):* upgraded `construct_check.py` for `0.5u` dual-sided membrane slabs (`G10`/`G11`/`G13`), `hindwing`/`elytron` wings, arthropod/insect torsos, true rotated OBB `span` (`rot_zyx`), sagittal midline `layer: 1` plane tolerance, mirror-prefix-safe `_leaf()` across `G3`/`G14`, `.geo.json` bone-level yaw (`G2`), grounded-forelimb `G12` guard, and `radial_pelagic_telemetry`; upgraded `verify_bbmodel_ground.py` with token-boundary N-leg auto-discovery (`FLOATING`, `HOVER`, `BIPED_WITH_FREE_ARMS`, `ARTHROPOD_NLEG`, `BIPED_2FOOT`, `QUADRUPED_4FOOT`), `WING_L/R` tracking, and `ALL_min` floor-plunge verification!
  9. *Live Parser-Synced Animation Docs (`unified_biomechanical_animation_solver.md` §6 & §6B):* synchronized `@muscles`, `@kinematics`, `flight {}`, and new `swim {}` syntax 1:1 with live `synccraft_v20/src/synccraft/dsl/parser.py`.
  10. *Universal General-Blockbench Reasoning, Scale-Harmonious Eye Detail, Small-Flyer Proportions & Multi-Angle Video Reference Escalation (`references/universal_anatomy_reasoning_and_self_check_principles.md`, `high_res_focal_carving_and_visual_primitives.md §1`, `cranial_and_jaw_structural_reasoning.md §10`, `unified_biomechanical_animation_solver.md §7B #15–16`):* distilled all multi-archetype findings into franchise-agnostic first-principles knowledge reusable for ANY real animal, bird/small flying creature, character, vehicle, mech, or prop — enforcing the **Human-Artist Multi-Perspective Reference Discovery Ladder** (`>= 4–5` verified angles, exact species/colorway identity gate, mandatory YouTube/video ecology & turnaround escalation whenever static images are sparse), **Small Flying Animal & Avian Wing-to-Torso / Head-Nesting Law** (`75–100%` folded dorsal span, `2.2–3.8×` unfolded wingspan with full chord, tilted avian posture vs. single-mass puffballs), **Physical Prop-Interaction & Coupled-Posture Law** (zero detached air gaps!), Two-Tier Scale-Driven Abstraction + Anti-Over-Carved-Sphere Law (`uv_scale: 1` native eyes on `small`, `<= 2` `64x` preserved on `hero/epic_boss`, zero `uv_scale >= 3` mixels!), 4-invariant `construct_check.py G14`, 100% dynamic tail/appendage responsiveness, AND **Stage A0 Video/Moveset Motion Research + Blind-Label Filmstrip Silhouette Gate**!

## 15. References, originality and licensing
- **Monster Hunter monsters are Capcom's IP.** Fan reconstructions are for non-commercial research only. Tag them like the repo's `research_nc`, keep them out of commercial weights, and never copy textures from screenshots.
- **For commercial work,** design originals:
  1. one concept line (habitat + temperament + role);
  2. 2-3 animal anchors for anatomy;
  3. one shape language (blades: fast and aggressive; slabs: heavy and ancient; curves: aquatic or young);
  4. one signature feature at 1.5x, plus at most two supporting features;
  5. a 3 + 1 palette;
  6. a plausibility pass: muscles, centre of mass, how the wings fold, how it eats.

  Change the body plan, shape language and signature feature, not just the colours.

  **Theme motifs:**

  | Theme | Forms | Surface | Glow |
  |---|---|---|---|
  | Fire | cracked plates, swept horns, vents | charcoal and deep red, bone horns | orange in cracks and throat |
  | Ice | crystal clusters, blades, fur ruffs | pale blue and white, navy shadow | cyan in crystals and eyes |
  | Storm | fin crests, forked horns, whip tail | slate blue with gold | electric blue along the dorsal row |
  | Poison | sacs, frills, barbels | olive, teal, yellow belly | acid green |
  | Earth | slab armour, club tail, thick horns | ochre and grey | amber in seams, or none |
  | Water | fins, whiskers, smooth body | deep blue to pale belly | bioluminescent rows |
  | Shadow | torn membranes, thin limbs | near-black purple | magenta or violet |
- **Credit** community artists, Cobblemon (CC-BY-NC), and Blockbench educators/timelapse creators (`references/youtube_master_distillation.md`) where their techniques informed a model.

## 16. Deliverables per asset
1. Concept paragraph, reference board (`>= 10` views/crops), **6-Stage Vision & Spatial Reasoning Plan (`S1–S6`, `5.1d`)**, chosen Cranial/Body Archetype (`A–H`, `5.1b`) + Signature DNA anti-template diff, **12-Point 3D Spatial Ratio Matrix (`5.1c`)**, world-scale ruler & skeleton plan, 4-bucket trait triage, and `MAPPING.md` table.
2. True iterative pass files (`W1-W3, D1-D2, T1-T2, A1` minimum — or `W1-W5, T1-T5, A1-A5` on formal epic boss builds, **never batch-generated in one loop**), each with turnaround renders, mandatory **`no-wing` 4-view** body renders (on winged assets), zoomed region crop comparisons, `scripts/construct_check.py --tier <tier>` (`G1-G12` across `.bbmodel`, `.geo.json`, or `.blockdsl`, plus `audit_scene` when in SyncCraft) logs, and `>= 3` concrete flaw-to-fix interventions per pass.
3. The final `.bbmodel` / Bedrock `.geo.json` (and `.blockdsl` when using SyncCraft), the cleaned `>= 1-2 px`-padded power-of-two `16 px/block` texture atlas (with cross-cube transverse pixel-relief carving, 5-layer wing shading, hidden faces painted, and zero dead islands), clean lowercase `snake_case` animation hierarchy + verified in-place treadmill locomotion + `288`-frame swept 15-axis OBB `SAT` inter-limb clearance, and the adversarial 14-item rubric scorecard (`>= 23/28`, zero `0`s, subject to `13.2` automatic caps and listing top 3 remaining trade-offs).
4. A short note on what was simplified, consolidated, or exaggerated, and why.
