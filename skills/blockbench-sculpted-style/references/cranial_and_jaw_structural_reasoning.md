# Cranial & Jaw Structural Reasoning, V-Sloped Head & `0.5u` V-Arch Dentition Craft, Species-Calibrated Snouts & Horn Concavity (`v37`)

> **Core Mission:** Give AI modelers a formal, transferable visual-and-spatial reasoning framework for constructing **species-authentic, non-template skeletons and coherent, thick, V-tapering heads, branched horns/antlers, paired/unique snout ornaments, full solid watertight lower jaws, external interlocking boss tusks, 3D backward-swept beards/ruffs, and 100%-span co-parented wings** across *both* imaginary creatures (gentle approachable guardians vs. furious unapproachable Elder Dragons) and real-world animals from official + in-game + YouTube master references — paired with **mandatory Texture-On every-pass inspection** and **closed-loop Standing Limb IK Pinning (`ground_pin_limbs`)**.

---

## 1. The 6-Plane Cranial & Jaw Structural Reasoning Framework

Why do AI-generated Blockbench creature heads most commonly fail? Because builders jump straight to stacking centered boxes from a single thumbnail, producing one of four classic failure modes:
1. **The Dangling Parrot Hook:** A high-angle (`35–50°`) tip block glued beneath the front of the nose that clashes with the chin.
2. **The Pinocchio / Swordfish Needle:** A thin, overly long (`head_L/H > 2.1`) telescope of narrow boxes jutting far past a stubby lower jaw.
3. **The Stubby Pig-Box:** Over-correcting by making the snout too short (`head_L/H < 1.55`) and square-faced, losing predatory/regal identity.
4. **The Piano-Key Barcode & Empty Hinge:** Painting alternating `1 px` black-and-white dots across a flat outer cheek wall, while leaving the roaring mouth as an empty floating door-hinge gap.

### Step 1 — Mandatory 2-Pass "Locate-First, Then Zoom" Multi-Angle Reference Extraction (`v24`)
**NEVER blindly crop an unseen reference with guessed fractional bounds** (e.g., `crop(0.30, 0.02, 0.72, 0.44)`) — blind cropping regularly lands on empty sky, wingtips, or shoulders, poisoning eye/snout spatial reasoning! Always execute two strict passes before placing any head box:
- **Pass 1 (Whole-Image `0.0–1.0` Coordinate Grid Localization):** Overlay a `0.10`-spaced coordinate grid across **at least 4–6 full official renders, in-game screenshots, and ecology frames**, open the gridded sheet via image vision, and **explicitly read & log the exact `(x_min, y_min, x_max, y_max)` of the head AND the exact `(x, y)` pixel center of the eyes** in each image.
- **Pass 2 (Verified Target Head Macro + Eye Ultra-Zoom):** Crop BOTH a **Head Macro** AND a dedicated **Eye-Socket Ultra-Zoom** centered strictly on the Pass-1 verified coordinates, and inspect BOTH via vision across 4 complementary views:
  1. **Strict Side Profile (Closed, e.g. `pm_14` / `pm_04`):** Measure `head_L : head_H`, upper-to-lower jaw ratio (`88–94%`), exact eye aspect (compact sleek horizontal hawk-almond vs. round — NEVER accidentally build a tall vertical neon-red wall!), brow-to-horn sweep angle (`40–55°` swept back for `M4` predators!), and rear cheek-blade flare.
  2. **Dead Front (`0°`, e.g. `pm_06`):** Verify the cross-sectional visor prow width, continuous golden/armored `V`-brow plunge angle, binocular eye visibility (`20°` yaw), and flared temple/ear plume silhouette.
  3. **Front-3/4 (`30–40°`, e.g. `pm_01` / `pm_09` / `pm_15`):** Study how brow ridges, recessed orbital pockets, cheekbones, subtle anterior nostrils, and deep chiaroscuro jaw shadows wrap around the skull corner.
  4. **Open-Mouth Roar (`30–42°`, e.g. `pm_11`):** Study interior mouth architecture — 6-plane zero-thickness tooth arcade alignment, upper palate rugae, stretched inner cheek webs, and 2-segment curved tongue color & arch.

### Step 2 — Classify the Creature's Jaw Archetype

| Jaw Archetype | Reference Examples | Target `head_L/H` | Upper Rostrum & Prow | Lower Mandible (`%` of Upper) |
|---|---|---:|---|---|
| **Aquiline / Paladin-Visor Dragon** | Primordial Malzeno, Valstrax, Griffins | `1.70–1.85` | Deep tapered bridge (`5.0 → 4.0u` H) with sharp median visor keel & subtle downward shield-bevel front prow (`18–22°`, overlapping chin by `1.0–1.5u`) | `88–92%` length; sturdy U-trough (`3.0–3.5u` H) tucked flush inside cheek skirts at rest |
| **Wedge Crocodilian / Classic Wyvern** | Lagiacrus, Anchor `06` Amethyst Dragon, Rathalos | `1.75–2.10` | Broad sloped dorsal muzzle tapering smoothly via `12–18°` V-rail maxilla cheeks to a rounded-square prow | `92–96%` length; full-length stout jaw with interlocking lateral canine wave |
| **Brutal Heavy-Mandible Predator** | Anchor `07` Deviljho, Anchor `09` Gore Magala, T-Rex | `1.35–1.65` | Thick deep squared-off skull, heavy armored supraorbital ridges, high muscular cheek mass | `95–102%` length; massive deep chin (`4.5–6.0u` H), prominent projecting jaw teeth |
| **Wide Amphibian / Beaked Gape** | Anchor `08` Tetranadon, Turtles, Avians | `1.10–1.45` | Broad flattened dorsal dome or keratinized upper beak curving cleanly into wide lateral lips | `85–95%` length; wide deep gular pouch / throat sac seamlessly merging into chest |

### Step 3 — Construct the Skull Across 6 Structural Planes

1. **Plane 1 — Cranium Vault & Forehead Crest:** Anchor the braincase (`cranium_core`) cleanly to `neck_3`; transition into the snout via an angled brow/forehead diamond (`forehead_diamond`) so the forehead never drops at a harsh `90°` stair-step.
2. **Plane 2 — Orbital Cradle (Zero-Leak Recessed Eyes):** Recess `eye_socket_facet` **100% flush inside the cranium's front & lateral envelope**, framed closely above by the angled brow visor and below by the flared cheekbone (`zygoma_shield`). Paint the 4-layer slit-pupil eye strictly on the **single outer lateral face** (`west` left / `east` right); all other 5 faces are dark obsidian socket (`0%` eye leak).
3. **Plane 3 — Deep Tapered Upper Rostrum + Integrated Anterior Prow:** Use `inset_carve(...)` (NEVER detached floating `place(...)` plugs!) to fuse the nasal bridge (`nasal_visor`), proud median keel (`nasal_ridge_crest`), tapered fore-snout (`beak_upper_bridge`), and subtle anterior visor prow (`beak_hook_tip`) into **one continuous flowing dorsal silhouette** (`6–9° → 12–14° → 18–22°`). Keep substantial vertical depth (`4.0–5.0u`) so the muzzle never looks needle-thin.
4. **Plane 4 — Converging Maxilla Cheek Skirts (`12–18°` Yaw) & Clean Occlusal Seam:** Attach left/right `maxilla_wall` skirts beneath the cheekbones angled inward (`12–15°` yaw) to form a true **triangular kite/diamond top-down wedge** (passing G2 cleanly). Along the closed outer lip edge, paint a **clean beveled dark occlusal crease** — **NEVER** alternating `1 px` black-and-white zebra/barcode dots across the exterior cheek! Place 1–2 discrete carved conical ivory canines (`maxilla_fang_plate`, `maxilla_rear_tooth`) cleanly tucked along the ventral rim.
5. **Plane 5 — Proportional Length-Matched U-Trough Mandible (`88–92%` Upper Length):** Hinge `jaw` far back beneath the rear cheekbone (`anchor.z ≈ 0.24`), build a sturdy `jaw_ramus` + `jaw_symphysis` reaching forward right behind the anterior prow tip, wrap it with raised converging left/right `mandible_rail_l/r` walls, and blend the underside cleanly into the neck via a swept `jaw_gular_keel` (zero detached floating chin lumps!).
6. **Plane 6 — Real 3D Hollow Oral Chamber, Zero-Thickness Single-Pixel-Square Teeth & Articulated Tongue (Roar Mastery!):**
   - Raise the upper palate roof (`nasal_visor` `anchor.y ≈ +0.16`) while letting the outer `maxilla_wall` skirts drop down to seal the closed lip line. This creates a **true physical 3D hollow oral chamber** between the upper palate and lower jaw floor with **0% SAT collision / 0% `swallowed_redundant_box`**.
   - **Adopt the Blockbench Official Style Guide & Master Mod Standard (ZERO Volumetric Tooth Cubes!):**
     - Volumetric 3D block teeth cause Z-fighting, UV bloat, collision issues, and chunky "chiclet / buck-tooth" clutter at `16 px/block` scale (explicitly warned against in the Blockbench Official Minecraft Style Guide, and avoided by *Anchor 06*, *Anchor 09*, *Ice and Fire*, *Alex's Mobs*, and *Cataclysm*).
     - **Build a Complete 6-Plane Zero-Thickness (`size = 0` on normal axis) Single-Pixel-Square Tooth U-Arcade:**
       - `upper_teeth_row` (L/R, `normal: x`, spanning continuously from the rear cheek corner to the front beak visor with ZERO toothless gap!) + `upper_front_teeth` (`normal: z`, tucked cleanly inside the front visor lip).
       - `lower_teeth_row` (L/R, `normal: x`, spanning continuously along the top inner edge of `mandible_bar`) + `lower_front_teeth` (`normal: z`, seated cleanly along the front chin rim to complete the gleaming `Anchor 06` U-ring!).
     - **Explicit Integer-Pixel Single-Pixel-Square Painting (NEVER Floating-Point Thresholds on `h = 2 px` Planes!):**
       - On a `2 px`-tall plane (`h = 2`), continuous formulas like `tip_t <= 0.96` evaluate `1.0 <= 0.96 -> False` and silently erase the ENTIRE tooth-tip row to `alpha = 0`!
       - Always assign pixels explicitly by integer row (`p_norm[..., 1]` distinguishing root vs. tip) and sorted front-to-back index (`front_idx = 0..w-1` from local `+Z`): paint crisp **`1×1 px` bright ivory squares** for cheek teeth/incisors and **`1×2 px` prominent fangs** for anterior canines, separated by **`1 px` transparent gaps (`alpha = 0`)**, and **stagger lower teeth by `1 px`** relative to upper teeth for natural carnivorous interlock!
   - **Complete 4-Layer Cavernous Interior Mouth Anatomy:**
     1. **Upper Palate Rugae:** Paint EVERY underside face of the upper snout (`nasal_visor`, `beak_upper_bridge`, `beak_hook_tip` `down`) deep dark mulberry-plum with rhythmic `1 px` transverse fleshy palate ridges (`rugae`) — ZERO bright silver/exterior hide leaking onto the mouth roof!
     2. **Medial Jaw Walls & Abyssal Rear Pharynx (`mouth_throat_back`):** Paint every inward-facing face of `maxilla_wall`, `mandible_bar`, `jaw_ramus`, and `jaw_symphysis` deep cavernous plum-black, and place a recessed dark throat backing box inside the rear jaw hinge so `38°` roars NEVER expose empty sky or neck armor.
     3. **Zero-Thickness Concave Cheek-Tendon Web (`plane jaw_cheek_web` L/R):** Span the rear jaw corner (`mandible_bar → maxilla_wall`) with a `0`-thickness `100%` binary hard-alpha concave sinew plane painted with subtle vertical bruised-mulberry tendon striations (`0` bulky cubes when closed, seamless stretched flesh in `38°` profile!).
     4. **Articulated 2-Segment Curved Tongue (`bone tongue` child of `jaw`):** Build `tongue_base` rising from the deep throat + narrower tapered `tongue_tip` arching down-forward (`+10–12°`), painted in **species-harmonized smoked royal plum-crimson / dark mulberry** (NEVER neon bubblegum-pink or flat traffic-cone red!) with a dark dorsal median sulcus groove and moist catchlight; animate a subtle `-10..-14°` upward curl during roar clips!
   - **Verification Gate:** Render a **6-angle `6x` Head Macro** (Closed 3/4, **Half-Snarl `20°` 3/4 & Front-Low**, **Full Roar `38°` 3/4, Profile & Front-Low**) side-by-side against official references AND Anchors `06 & 09`!

---

## 1B. The Trait-Priority & Essence-Distillation Law ("Keep the Soul, Translate to Blockbench, Drop Micro-Noise")

> **Core Aesthetic Philosophy:** Minecraft Blockbench is **not** a 200,000-polygon photorealistic sculpting engine. Trying to force every microscopic bump, wrinkle, and minor secondary detail from high-poly official artwork into tiny cubes ruins readability and breaks the Minecraft-Blockbench aesthetic. Master Blockbench artists preserve the **overall feeling and unmistakable identity** of a reference by sorting every observed trait into **3 disciplined priority tiers**:

| Priority Tier | Rule | Concrete Examples (Primordial Malzeno & General Creatures) |
|---|---|---|
| **Tier 1: Iconic Identity Anchors** | **100% KEEP & EXAGGERATE (`1.2–1.4×`)** — the `4–6` silhouette features that make the species recognizable in `0.5 s` even as a pure black silhouette. | Twin golden Paladin crown horns + Valkyrie temple fans; sleek knight-visor skull & recessed crimson slit eyes; towering serrated bronze-gold wing blades + sapphire sails; layered pearl-silver breastplate; diamond-blade halberd tail. |
| **Tier 2: Medium-Translated Structures** | **ADAPT INTO NATIVE BLOCKBENCH GRAMMAR** — preserve the *visual feeling* using Blockbench's cleanest native craft rather than literal high-poly geometry. | Serrated dragon dentition → **recessed `100%` hard-alpha perimeter tooth-row planes** (`Anchor 06/09`); complex overlapping body scales/ribs → **`16 px` painterly `U`-scallop shingles & transverse rung relief**; curved organic wing sails → **vaulted full-chord nested-fan alpha sheets** with 3-sided organic AO. |
| **Tier 3: Micro-Noise & Non-Anatomical Clutter** | **DELIBERATELY DROP** — high-poly clutter that fights for `1 px` space at `16 px/block` OR non-anatomical environmental VFX. | Sub-millimeter skin wrinkles, individual molar/rear-tooth micro-cubes, redundant overlapping cheek shards, stray microscopic spikes, and environmental/VFX artifacts (**Qurio dots**, ambient dust, lens flares, hit-sparks). |

**Decision Heuristic for AI Modelers:** Whenever a micro-trait from a high-poly reference makes a `16 px/block` region look cluttered, noisy, or clunky (e.g., chunky square tooth blocks on the cheek or flat blue pelvic fins clipping the tail), **drop the clunky micro-geometry and ask: "How would `Anchor 06` or `Anchor 09` express this *feeling* cleanly in Blockbench?"**

---

## 1C. Anti-Template Species-First Morphometry, In-Game Close-Up Verification & Mathematical Rotation-Sign Laws (`v27`)

> **Origin Trap (White Fatalis `W1–W3` Template Cloning & Sign Inversion):** Even with high-res official renders on disk, an AI model fell into **4 compounding generalization failures** that made White Fatalis look like a recolored clone of Malzeno:
> 1. **Skeleton Template Cloning:** Reused a horizontal `0°` tabletop quadruped spine (`waist +2°`, `chest +2°`) and short 4-segment neck instead of measuring White Fatalis's **~46° semi-erect pear-chest posture (`chest_y >> pelvis_y`)** and **long 5-segment swan/cobra S-neck**.
> 2. **Horn Template Cloning & Missed Branching:** Reused 4 simple unbranched rods instead of verifying **actual in-game front/3-4 close-ups**, which revealed **TRUE `Y`-branched stag/celestial antler horns** (curved upward crescent main beam + outward lateral fork tine + secondary lower temple horn) AND **TWIN paired lateral snout horns (`nose_horn_l/r`)** rather than a single center unicorn horn!
> 3. **Reverse-V Snout & Inward-Leg Rotation Sign Bug:** Applied `-14°` Y-yaw to the left (`x < 0`) `+Z` maxilla/jaw rails, flaring the nose tip OUTWARD into a **Reverse-V duckbill** instead of a fierce tapering dragon V-wedge — and applied `+22°` Z-roll to the left (`x < 0`) `-Y` thigh, swinging the hind knees INWARD under the groin instead of wide outward crouching haunches!
> 4. **Thin Pencil-Snout & Hollow Lower-Jaw Floor:** Over-narrowed the cranium/snout into a thin pencil and left a narrow `3.0u` `jaw_floor` with visible see-through gaps beside `jaw_rail_l/r` during `38°` roars instead of building a **full, solid, stepped watertight lower mandible (`jaw_floor_rear` + `jaw_floor` + thick `jaw_rail_l/r` + `jaw_chin_prow`)**.

### Mandatory 4-Step Anti-Template Procedure (Before & During `W1–W4`)
Every AI model (especially small RSI-trained multimodal models) MUST run these 4 checks on EVERY new creature — **never inherit a skeleton or horn shape from a previous model!**

1. **Step A — Pass-2B Multi-Angle In-Game Close-Up Verification (Beyond Promotional Key Art!):**
   - Official key art, statues, and dark cinematic thumbnails routinely hide horn branching, paired vs. single snout spikes, head thickness, and lower-jaw volume behind foreshortening or shadows.
   - ALWAYS collect and inspect **≥ 3–4 crisp in-game screenshots or 3D model-viewer captures** covering **dead-front (`0°`), front-3/4 (`35°`), and side-profile (`90°`)** of the head, horns, and torso, and explicitly log answers to 5 yes/no morphological questions BEFORE writing `@rig`:
     - *Torso pitch:* Horizontal (`0°`) vs. Semi-erect (`-35..-50°`) vs. Upright biped (`-65..-85°`)?
     - *Horn topology:* Unbranched rods vs. **`Y`-branched antlers** vs. spiral/ram horns vs. cranial crest/frill? Which pair is longest/widest?
     - *Snout ornaments:* None vs. single median horn vs. **paired lateral nose horns (`L/R` V-notch)**?
     - *Cranial volume & jaw:* Slim razor-wedge vs. **broad chunky armored skull**, and is the lower mandible **100% solid & watertight** from front, side, and underneath at `38°` roar?
     - *Limb & tail asymmetry:* Equal legs vs. **wide low crouching haunches + slender tucked forearms**, and elevated whip tail vs. **heavy low-sweeping ground tail**?

2. **Step B — Species-Specific Mathematical Morphometric Assertions (`_verify_species_geometry`):**
   - Never rely on "eyeballing" alone when coordinate sign flips can invert a V-snout or collapse a posture! Add **5 automated 3D bounding-box assertions** to the build script that FAIL LOUDLY on template drift:
     - **True V-Snout & V-Jaw Taper Check:** Measure X-width of front half (`z >= z_mid`) vs. rear half (`z <= z_mid`) of `maxilla_l/r` and `jaw_rail_l/r` -> assert `front_w < rear_w - 0.8u` (mathematically prevents Reverse-V snouts forever!).
     - **Species Posture Check:** Assert `chest_y - pelvis_y` and `head_y - chest_y` match the reference's measured torso pitch & neck height (prevents collapsing a semi-erect dragon onto a flat dog back!).
     - **Ornament Topology Check:** Assert the exact species-specific horn/crest boxes exist AND satisfy measured lateral spread (`antler_branch_x > antler_crescent_x`, twin `nose_horn_l` + `nose_horn_r` present with open-V separation).
     - **Full Solid Lower-Jaw Check:** Assert stepped solid `jaw_floor_rear` + `jaw_floor` seal the entire lateral span between `jaw_rail_l` and `jaw_rail_r` with zero hollow cracks when roaring at `38°`.

3. **Step C — Right-Handed SyncCraft / Blockbench Rotation-Sign Cheat Sheet (ZERO Sign Guessing!):**
   | Local Part Axis | Side | Desired Visual Direction | Exact Right-Handed Euler Sign (`euler_xyz`) |
   |---|---|---|---|
   | **Forward `+Z` Snout / Maxilla / Jaw Rail** | Left (`x < 0`) | Taper **INWARD** (`+X`) into fierce **V-shape** | **`ry > 0`** (e.g. `+13°..+15°`; Right uses `-13°..-15°`) |
   | **Backward `-Z` Crown / Temple Horn** | Left (`x < 0`) | Splay **OUTWARD** (`-X`) away from neck | **`ry > 0`** (e.g. `+25°..+52°`) |
   | **Backward `-Z` Crown / Antler Horn** | Either | Curve **UPWARD** (`+Y`) | **`rx > 0`** (e.g. `+20°..+38°`) |
   | **Upward `+Y` Paired Nose Horns** | Left (`x < 0`) | Flare **OUTWARD** (`-X`) into open **V-notch** | **`rz > 0`** (e.g. `+15°..+22°`; `-rz` tilts inward!) |
   | **Downward `-Y` Thigh / Upper Arm** | Left (`x < 0`) | Splay knee/elbow **OUTWARD** (`-X`) wide | **`rz < 0`** (e.g. `-10°..-24°`; `+rz` tucks inward!) |
   | **Forward `+Z` Spine / Waist / Chest** | Center | Pitch **UPWARD** (`+Y`) semi-erect torso | **`rx < 0`** (e.g. `-20°..-26°`) |
   | **Outward `-X` Left Wing Spar** | Left (`x < 0`) | Lift wingtip **UPWARD** (`+Y`) into M-arch | **`rz < 0`** (e.g. `-22°..-46°`) |
   | **Outward `-X` Left Wing Finger (`f1..f3`)** | Left (`x < 0`) | Fan finger **BACKWARD** (`-Z`) across membrane | **`ry < 0`** (e.g. `-2°, -26°, -52°`; **`+ry` swings forward `+Z` away from `-Z` membrane!**) |
   | **Wing Canopy Plane (`normal: y`, `-Z` trail)** | World | Face **FORWARD (`+Z`)** & drape scallops **DOWN (`-Y`)** | **`world_rx < 0`** (`-55°..-65°`, e.g. `wing_l_upper rx=-22°` inside `-39°` chest = `-61°` cathedral sail!) |
   | **Supraorbital Brow (`brow_ridge_l/r`)** | L (`x<0`) / R (`x>0`) | Plunge front glabella **DOWN** into furious **`\ /` scowl** | **`rx > 0` (`+18°..+24°`) + `rz < 0` on Left (`-26°..-32°`, `+26°..+32°` on Right)** *(Negative `rx < 0` lifts front brow into sad puppy eyes!)* |
   | **Hanging `-Y` Chin Beard (`beard_*`)** | Center / L / R | Sweep beard **BACKWARD (`-Z`)** along throat | **`rx > 0` (`+26°..+34°`)** *(Negative `rx < 0` juts forward `+Z` into a snowplow goatee!)* |
   | **Outward `-X` Shoulder / Cheek Spike** | Left (`x < 0`) | Point `+Y` spike **OUTWARD (`-X`)** clear of body | **`rz > 0` (`+38°..+48°`)** (`Rz(+θ) @ (0,1,0) = (-sin θ, cos θ, 0)`) |

4. **Step D — Leading-Edge-Pinned Inner Wing Sail Math (`0.000u` Seam Without Tilting Off Humerus):**
   - Whenever rotating a `normal: y` inner wing sail (`size: [W, L]`, leading edge at `+L/2`) around X by `rx` degrees to meet `wing_bay_4` cleanly at the elbow, rotating around the plane center pulls the leading edge off `wing_hum_core` by `0.5 * L * sin(rx)`!
   - Pin the leading edge at **`0.000u`** by adding `offset: [0, +0.5 * L * sin(rx), +0.5 * L * (1 - cos(rx))]` — guaranteeing `PHYS_COMP == 1`, `0.000u` humerus attachment, AND seamless overlap with `wing_bay_4` across the elbow!

5. **Step E — Multi-Block Segmented Supraorbital Arch, Solid Stepped Mandible, Balanced Orbital Shelf & 3D Backward-Swept Fur (`v28`):**
   - **NEVER rely on two giant single V-shape planks (`brow_ridge_l/r` `[4.5, 2.0, 9.0]` or `jaw_ramus_l/r` `[2.0, 2.0, 6.5]`) to form brows, snouts, or jaws:** two oversized straight diagonal bars look stiff and overly geometric/straightforward in Dead-Front (`0°`) view! Instead, carve organic curves from **multi-block progressive stepped chains**:
     - **3-Block Segmented Supraorbital Arch (`brow_inner -> brow_arch -> brow_outer` on child `brow_sculpt_l/r`)** paired with a **stepped central forehead-to-nasal scute column** (`forehead_step -> snout_scute_1 -> snout_ridge`);
     - **4-Step Snout Core (`snout_rear -> snout_mid -> snout_front -> snout_beak_hook`)** + **2-step maxilla lip bosses** + **2-step jugal cheekbones (`zygoma_l` + `zygoma_post_l`)**;
     - **Solid Stepped Watertight Lower Mandible (`jaw_rear -> jaw_mid -> jaw_chin -> jaw_chin_tip`, `8.5 -> 6.5 -> 5.0 -> 3.5u`)** — NEVER replace a solid stepped U-mandible with crossed V-rail bars!
   - **Balanced Orbital Shelf Law (Neither Embedded Nor Extruded!):**
     - Seat `eye_l/r` so its front-outer lateral edge clears `cranium_core`'s side wall (`x_front_outer < cranium_xmin`, preventing `cranium_core` from burying the eye in `Side`/`3-4`), WHILE its rear-outer corner stays strictly tucked inside the overhanging 3-block brow and 2-block cheekbone (`x_rear_outer >= brow_xmin + 0.35u`, **0.00u eye extrusion outside the head silhouette**!).
   - **Furious Boss Snarl & External Interlocking Tusks:** Hold a tense `14–16°` resting snarl gape (`jaw rest_euler`) and mount **3D protruding upper canines (`tusk_upper_l/r`) laterally at the outer maxilla lip rim (`|x| > 0.5 * jaw_chin.width`)**, interlocking outside lower underbite tusks (`tusk_lower_l/r`) with **`0.00u` SAT overlap**!
   - **3D + Serrated Plane Fur Law & `mirror_x` UV Symmetry Trap:** Build beards and cheek ruffs from **3D sculpted core cuboids (`beard_chin_core`, `cheek_fur_mass`)** paired with swept jagged alpha planes. When painting asymmetric lateral cutouts (`cheek_ruff_front`) in Python/shader scripts, **ALWAYS test world side via `float(box.center[0]) < 0.0` — NEVER `box.name.endswith("_l")`**, because `mirror_x` names the right copy `cheek_ruff_r__cheek_ruff_front_l` (still ending in `"_l"`!), which otherwise puts the straight root edge on the outside of the right cheek!

---

## 2. Reference Signal vs. Environmental / VFX Noise Isolation & Mandatory Texture-On Pass Law

> **Origin Traps:**
> 1. *(Primordial Malzeno `T5/T6`):* Capcom key art showed tiny glowing red dots floating in front of Primordial Malzeno's chest — actually ambient **Qurio** particles, NOT chest gems!
> 2. *(White Fatalis `v27 -> v28`):* Inspecting white/clay/grayscale geometry passes alone hid **97% erased outer wing-bay alpha cutouts**, **inverted right-cheek fur UV cutouts**, **occluded dead-front irises**, and **low-contrast chalky snout blocks**!

**Mandatory `v28` Texture-On Every-Pass Verification Rule:**
- **NEVER evaluate or sign off ANY modeling or animation pass in clay, grayscale, or flat pastel mode alone!** Every single pass (`P1..P4` / `W1..T5` / `A1..A5`) MUST be rendered and visually inspected **WITH FULL TEXTURES & HARD ALPHA ENABLED (`clay=False`)** across `6-View Turnaround`, `No-Wing 4-View`, `6× Head/Eye/Jaw/Fur Macro`, `5× Full-Span & Inner-Seam Wing Zoom`, and `25% Squint Gate`.

**Mandatory Pre-Texture Isolation Rule:**
- Before painting ANY glowing dot, floating spark, color cast, highlight rim, or surface blemish seen in promotional key art or gameplay screenshots, **cross-verify across ≥ 2 neutral references** (concept art turnaround + calm wildlife ecology video).
- **Reject & Strip** any element that moves independently, floats off the surface, changes position between frames, or comes from environment lighting/particles (Qurio, fireflies, dust motes, lava sparks, breath particles, lens flares, hit-sparks). Only paint **permanent anatomical structures**.

---

## 3. Blockbench 4-Foot Ground-Locking Gate (Static + Dynamic Animations)

> **Origin Trap (Primordial Malzeno `W6 + A6`):** Checking only global scene `y_min = 0.003` hid the fact that **hind feet sat at `y = 0.00` while BOTH front paws floated `+6.21u` in mid-air** — and in animations (`idle_noble`, `walk_patrol`, `ecology_royal_sit`), upward torso pitch hoisted front paws `+12..+21u` above Blockbench's ground grid while hind legs sank `-8.4u` underground!

### Static Rest-Pose 4-Foot Grounding Contract
Never rely on global `min(y)` alone! Compute per-limb world `y_min` independently for **all weight-bearing extremities** (`FL`, `FR`, `HL`, `HR`):
- **All 4 feet MUST touch the exact `y = 0.00` plane simultaneously:**
  $$\max(|y_{\min}(\text{FL})|,\ |y_{\min}(\text{FR})|,\ |y_{\min}(\text{HL})|,\ |y_{\min}(\text{HR})|) \le 0.30\text{ u}$$
- Flatten distal paw/foot rest angles (`paw_l`, `foot_l`, and toe tips) so the **entire paw pad AND all primary claws** sit level on the ground (`y ∈ [0.00, 0.90]u`), rather than balancing on a single drooping dewclaw.

### Dynamic Animation Ground-Locking Contract
Audit **every frame** of **every clip** (and verify directly on the exported `.bbmodel` FK!) against these strict physics bounds:
1. **Standing Idle / Guard / Sit Clips (`idle_noble`, `perched_greatshield_guard`, `ecology_royal_sit`):**
   - ALL planted feet (`FL`, `FR`, `HL`, `HR`) stay continuously pinned on the floor (**`y_min ∈ [-0.6, +1.2]u`**).
   - **Elbow/Knee Flexion Sign Law:** Always verify the joint pitch sign! When chest/pelvis bows **down/forward**, flex elbows/knees to *shorten* the vertical reach so paws stay on `y = 0` instead of punching underground (`-10u`). When pelvis drops & pitches **up/back** (royal sit or upright bloom), extend forelimbs cleanly downward while folding hind thighs/shins tightly forward so **both haunches/hind feet AND propped front paws sit flat on `y = 0`**!
2. **Locomotion Clips (`walk_patrol`, `run_charge`):**
   - Every single leg (`FL`, `FR`, `HL`, `HR`) MUST plant cleanly during stance (**`min(y_min) ∈ [-1.0, +0.6]u`**) and lift cleanly during swing (**`max(y_min) ∈ [+3.5, +8.5]u`**) with zero underground sinking.
3. **Grounded Combat & Transition Clips (`bow_stretch`, `shield_bash`, `lance_thrust`, `tail_pierce`, `takeoff_vault`, `supernova`):**
   - Start & end frames grounded (`FL/FR/HL/HR ≈ 0.0`); weight-bearing support feet never dip deeper than `-1.5u` and tails never clip underground (`TAIL_min > +1.0u`).

---

## 4. Seamless Proximal Wing Membranes & Zero Pelvis-to-Tail Leakage

1. **Proximal Inner Wing Membrane Texture & Geometry (`wing_inner_sail`, `wing_fore_patagium`, `wing_bay_4`):**
   - **Geometry:** Keep proximal inner panels progressively tighter in chord (`wing_inner_sail` tucked cleanly behind `wing_humerus`, `wing_fore_patagium` tilted smoothly to match `wing_bay_4`) so the trailing edge forms **one graceful concave arch** into the ribcage instead of double-notched zig-zag overlaps.
   - **3-Sided Organic AO (NO Flat Armpit Slabs or Bullseyes!):** Inner wing panels closest to the torso MUST darken along **ALL 3 structural borders** (leading wing bone + inner torso/shoulder flank + outer elbow seam) with curved catenary alpha cutouts on all free edges. Never paint circular lunar rings, glowing dots, or bright marbling blobs on the proximal armpit membrane — use deep twilight-indigo leather with subtle radiating shoulder/elbow tension pleats.
   - **Velvety Dark Leathery Trailing Hem:** Never outline serrated binary-alpha wing cutouts with bright yellow/gold `1 px` pixels (which accentuate pixel stair-stepping). Rim the trailing scallop with a **dark velvety plum-indigo leathery hem** for painterly cohesion matching Anchors `06–09`.
2. **Zero Pelvis-to-Tail Midline Leakage:**
   - NEVER mount flat wing-membrane planes (`pelvic_skirt_fin`) across the pelvis-to-tail root junction, where they stab straight through `tail_1` and poke out as floating shards on both sides of the tail base.
   - Sculpt hip flanks exclusively with solid, outward-hugging beveled armor tassets (`hip_rib_slope` + `pelvic_skirt_blade`) that stay laterally clear of `tail_1` across all pitch/yaw bends.

---

## 5. Developing Tiny Vigorous Craft Details ("From Clean to Alive & Gorgeous")

Once macro proportions (`W1–W5`), clean `16 px/block` pixel-relief shading (`T1–T5`), and the 3-Tier Trait-Priority Law (`§1B`) are locked, what separates a merely "accurate" Blockbench model from a **vigorous, breathtaking, master-artist centerpiece**? **Disciplined micro-storytelling pixels** — tiny intentional details placed ONLY at focal anatomical transitions without adding volumetric geometry clutter:

| Micro-Trait | Correct Master Blockbench Craft (`16 px/block`) | Classic Pitfall to Avoid |
|---|---|---|
| **Teeth & Oral Interior** | **Zero-thickness 2D flat `plane` U-arcade** with crisp explicit `1×1 px` ivory pixel squares + `1×2 px` front canines, staggered `1 px`, paired with `1 px` upper palate rugae, dark medial walls, concave cheek-tendon plane, and smoked mulberry-crimson 2-segment tongue. | Chunky 3D volumetric tooth cubes (Z-fighting / buck-teeth); floating-point alpha thresholds erasing `h=2 px` tips; neon bubblegum-pink tongue; bright silver interior mouth walls. |
| **Nostril Micro-Slits** | A single recessed **`1×1 px` dark-slate (`base[0]`) pixel** tucked cleanly near the anterior corner of the snout bridge (`beak_upper_bridge` lateral face) — subtle, readable, predatory. | Putting a pitch-black square right underneath a bright pure-white square in the middle of the snout wall (**reads immediately as a cartoon googly eye!**). |
| **Orbital Lower-Eyelid Rim** | Pair the recessed glowing slit eye with a **`1 px` crisp pale catchlight** along the top ridge of `zygoma_shield` + a **`1 px` dark orbital shadow crease** right beneath it, framing the eye like forged helmet armor or reptile scales. | Leaving the cheekbone below the eye a uniform flat grey slab, or letting emissive iris pixels bleed across front/cheek corners. |
| **Selective Honed Bevel Glints** | Place sparse, deliberate **`1 px` bright warm specular glints** (`step 6`) strictly on the 4–6 sharpest apex bevels (forehead crest diamond, beak prow ridge, sternum V-keel peak, leading pauldron rim, halberd tip) so metal/bone catches crisp directional light. | Outlining every cuboid border symmetrically (`yy == 0` UI wireframe look) or scattering random white dust specks. |
| **Keratinized Root Collars & Joint Flexor Creases** | Paint a **`1–2 px` dark darkened keratin cuticle collar** where every horn emerges from the skull and every talon emerges from a toe, plus **`1 px` deep shadowed flexor creases** inside elbow, knee, and wing-wrist folds so limbs feel organically rooted rather than plugged together. | Uniformly colored claws/horns with no root grounding or flat pastel joint interiors that flash brightly during animation. |
| **Scale-Overlap Cast Shadows** | Beneath every raised armor plate, dorsal scute, or ventral belly rung, tuck a **`1 px` cool-violet/indigo cast shadow** directly on the underlying host box — turning flat overlapping cuboids into lush, vigorous 3D relief. | Relying solely on Blockbench's flat directional viewport shading (which casts zero local self-shadows between overlapping cuboids!). |

---

## 6. Canonical `v28` Multi-Block Dragon Head, Partial Upper-Jaw & Solid Stepped Lower-Mandible Blueprint (Proven on *White Fatalis v28*)

> **Owner-Verified Gold Standard (`P3_sculpted_final`):** During live `0°` Dead-Front, `3/4`, and `-90°` Side iteration on *White Fatalis*, we proved **why single large V-shape planks fail** and **how human-master multi-block stepped sculpting succeeds**:
> - **Anti-Pattern A (Large Single V-Planks):** Using one long `9.0u` diagonal `brow_ridge` plank per side, one `7.5u` `maxilla` plank, or tilted `jaw_ramus` V-bars looks overly straightforward and mechanically geometric in `0°` Dead-Front view — and on the lower jaw, V-rail planks pinch and distort the chin!
> - **Anti-Pattern B (Chin-Up Foreshortening & Unbalanced Eyes):** Failing to sum cumulative `pelvis -> waist -> chest -> neck_1..N -> head` X-pitches left `head` at `-13°` world pitch (chin tilted UP!), stacking the chin & snout across the eyes; over-pushing eyes outward (`-6.49u` past `-6.36u` brow) extruded them outside the skull, while over-pulling them inward (`-4.54u` inside `-4.75u` cranium) buried them inside the head wall!

### The 5-Subassembly Multi-Block Cranial & Mandible Recipe (`SAT = PASS, coll = 0`, `G1–G12` Clean)

1. **Subassembly 1 — Broad Predator Braincase, Stepped Forehead Bridge & `+6°..+10°` True World Glare (`bone head`):**
   - Sum all parent spine + neck X-pitches so `head` sits at **`+6°..+10°` net downward world pitch** (dropping the snout tip cleanly *below* the eye line in `0°` Dead-Front view while preserving a coiled S-neck in `-90°` Profile!).
   - Build `cranium_core` (`9.5w × 6.5h × 9.0L`) + `crown_roof` (`7.5 × 2.0 × 8.0`, `-8°` pitch) + `forehead_step` (`5.5 × 2.0 × 4.0`, `+10°` downward bridge into the snout).
2. **Subassembly 2 — 3-Block Progressive Supraorbital Brow Arch (`bone brow_sculpt_l/r`, `mirror_x`):**
   - Replace single straight V-planks with **3 progressive compound-rotated blocks per side**:
     - `brow_inner` (`2.5 × 1.5 × 3.5`, `rotate: [12°, 10°, -14°]`): steps out from `forehead_step`;
     - `brow_ridge_l` (`3.0 × 2.0 × 4.0`, `rotate: [16°, 18°, -18°]`): hoods the orbital socket cleanly;
     - `brow_outer` (`3.0 × 2.0 × 4.0`, `rotate: [20°, 26°, -14°]`): sweeps wide and upward into the crimson-veined horn root (`ha_1`)!
3. **Subassembly 3 — Balanced Orbital Eye Shelf + 2-Block Stepped Jugal Cheekbones (`eye_l/r` + `zygoma_l` + `zygoma_post_l`):**
   - Seat `eye_l/r` (`1.5 × 2.0 × 3.0`, `rotate: [6°, ±16°, ∓8°]`) at the **exact orbital balance point**:
     - `x_front_outer < cranium_xmin` (`~0.25–0.50u` proud of `cranium_core`'s flat side wall so `cranium_core` NEVER embeds/swallows the iris in `Side` or `3/4`), AND
     - `x_rear_outer >= brow_xmin + 0.35u` (strictly recessed `0.5–1.0u` inside `brow_ridge_l` above and `2`-stepped `zygoma_l [3.0, 2.0, 4.5]` + `zygoma_post_l [3.5, 2.5, 5.0]` below — **0.00u extrusion outside the head contour**!).
   - Separate crescent/branched crown horn tips cleanly (`~12–14u` open crown span) so they frame the central crest without pinching into a closed `/\` cone.
4. **Subassembly 4 — 4-Step Upper Snout Core + Rhythmic Dorsal Nasal Scutes + 2-Step Partial Upper-Jaw Lip Bosses (`bone snout_assembly`):**
   - Step the upper snout smoothly across **4 cores**: `snout_rear [7.0, 3.5, 3.5]` -> `snout_mid [6.0, 3.5, 3.0]` -> `snout_front [4.5, 3.0, 3.0]` -> hooked `snout_beak_hook [3.5, 2.5, 2.0]` (`+18°` pitch).
   - Crown the bridge with **rhythmic stepped dorsal nasal scutes** (`snout_scute_1 [3.5, 1.5, 4.0]` -> `snout_ridge [2.5, 1.5, 4.5]`) + species snout horns (`nose_horn_l/r`), and sculpt the partial upper-jaw lip margin from **2 progressive lip bosses per side** (`maxilla_l [2.0, 2.5, 4.0]` at `[-4°, ±10°, ∓8°]` -> `maxilla_ant_l [2.0, 2.0, 4.0]` at `[-4°, ±14°, ∓10°]`) framing external interlocking upper canines (`tusk_upper_l/r`, `tusk_rear_l/r`) + dark crimson `upper_palate` & 3 zero-thickness upper tooth planes.
5. **Subassembly 5 — Solid 4-Step Watertight Lower Mandible (`bone jaw` — ZERO V-Shape `jaw_ramus` Planks!):**
   - Construct the lower mandible from **4 solid stepped watertight cuboids**: `jaw_rear [8.5, 3.0, 4.5]` -> `jaw_mid [6.5, 2.5, 4.0]` -> `jaw_chin [5.0, 2.0, 3.5]` (`-10°` upward chin bevel) -> `jaw_chin_tip [3.5, 2.0, 2.0]` (`-12°` prow), holding a tense `14–16°` resting snarl gape with compact lower canines (`tusk_lower_l/r`), 3 zero-thickness lower tooth planes, smoked mulberry 2-segment `tongue`, and backward-swept (`Rx +28°..+34°`) 3D chin/throat beard cores!

---

## 7. Orthographic-First Horn Curvature, Size, Pattern & Trait Discipline + Fanged-Wyvern Compact Neck Law (`v31` *Zinogre* Manual RSI Distillation)

> **Origin Trap (*Zinogre* `P1–W3` Neck & Horn RSI Catch):** During the *Zinogre* (*Monster Hunter* Thunder Wolf Wyvern) test flight, two classic cross-species habits surfaced and were caught through human-artist visual review:
> 1. **Elder-Dragon Long-Neck Drift (`P1`):** Starting from a `4`-segment `34u` Elder-Dragon neck made a muscular quadruped Fanged Wyvern look giraffe/plesiosaur-necked instead of compact and bull-shouldered.
> 2. **Foreshortened Gameplay Horn-Direction Reversal (`W2 -> W3`):** Reading a steep top-down mid-attack gameplay screenshot misled `W2` into sweeping Zinogre's golden horns *backward* (`-Z`) over the mane, whereas orthographic side/front concept turnarounds (`MHP3rd` & `MHWI`/`MHRS`) prove Zinogre's twin corrugated horns **sweep boldly FORWARD (`+Z`) over the brow and snout** in an outward-bowing `( )` oni-lyre crown!

### 7.1 Mandatory Orthographic-First Horn Sweep & Curvature Concavity Polarity (`CONCAVE_UP ∪` vs. `CONCAVE_DOWN ∩` vs. `S_INFLECTION ~`, `v34`)
- **Never determine horn sweep OR bending concavity from foreshortened `3/4` combat screenshots alone:** Camera pitch, head tilt during roars, or articulated enrage plates can invert BOTH apparent direction (`+Z` forward vs. `-Z` backward) AND apparent curvature concavity (`CONCAVE_UP ∪` vs. `CONCAVE_DOWN ∩`)!
- **Strict 3-View Native-Resolution Reference Hierarchy for Curved Structures (`visual_primitives_prep.py trace --label-concavity`):**
  1. **Orthographic Side Profile + Isolated Head Side Study (Primary Direction & Concavity Authority):**
     - First confirm **primary axis direction**: Forward (`+Z`) over brow/snout vs. Upward (`+Y`) crown diadem vs. Backward (`-Z`) swept spear/antler.
     - Second, **explicitly trace `3–4` centerline waypoints (`root -> base -> mid -> tip`) with `visual_primitives_prep.py trace`** (`label_concavity: true`) to prove mathematically whether the horn bends **`CONCAVE_DOWN (∩)`** (steep launch arching progressively downward/forward — e.g., forward-arching oni/ram spears like *Zinogre*, raptor beaks, curved claws), **`CONCAVE_UP (∪)`** (swept launch curving progressively upward/forward — e.g., gothic crown diadems like *Malzeno*, upturned crescent tines on *White Fatalis*, scooped bull/buffalo hooks), or **`S_INFLECTION (~)`** (inflecting mid-span — e.g., long regal *White Fatalis* main antler beams or recurve blades)! See `references/high_res_focal_carving_and_visual_primitives.md §1B` for the full signed Euler delta table.
  2. **Orthographic Dead-Front (`0°`) View (Crown Silhouette Authority):** Measure whether the horns flare straight outward (`\ /`), bow wide at mid-beam and converge gently inward (`( )` lyre crown, `mid_sep >= tip_sep >= 8.0u`), or fork into clearly separated `Y`-branched antlers (`ht_tip` framing the central crest cleanly without crossing at midline).
  3. **3/4 Macro Zoom (Surface Pattern & Root Socket Authority):** Verify whether left and right horn roots stay separated by a visible scale/hide furrow (`crown_roof`, **never fused into a solid yellow monobrow block!**) and whether rear occipital transition plates (`occiput_crown_l/r`) sweep backward behind the horn base to bridge smoothly into the neck mane.
- **Right-Handed Forward-Arching `CONCAVE_DOWN (∩)` `( )` Oni-Lyre Horn Euler Formula (Verified on *Zinogre v34*, `z04_head_profile`: `-27.7°` base, `-14.9°` mid!):**
  - On a **forward (`+Z`)** horn pitched upward (`Rx < 0`), **`Rz > 0` (`+6°..+10°` on Left `-X`)** rolls local `+Y` outward (`-X`), while **positive child pitch deltas (`rx_2, rx_3 > 0`)** curve the arch **CONCAVE-DOWN (`∩`)**! (Note: because non-zero yaw/roll couple into world pitch, always assert world YZ sagittal pitches `p0 > p1 > p2 > p3 >= 12°` directly on compiled boxes!)
  - Canonical `CONCAVE-DOWN (∩)` 3-segment + chiselled-tip chain (*Zinogre v34*):
    - `hb_1` (`[4.0, 3.5, 6.5]`, `rest_euler: [-65°, -25°, +8°]`): surges steeply upward-and-forward (`~57°` world pitch) and bows wide outward off the crown;
    - `hb_2` (`[3.5, 2.5, 6.0]`, `rest_euler: [+12°, +18°, -4°]`): arches smoothly **CONCAVE-DOWN** (`~42°` world pitch) and curves inward;
    - `hb_3` (`[2.5, 2.0, 5.0]`, `rest_euler: [+9°, +16°, -4°]`) + `hb_tip` (`[1.5, 1.5, 3.5]`, `rotate: [+6°, +8°, 0°]`): continues the **CONCAVE-DOWN** spear arch while terminating at `~20°` upward-and-forward (`horn_tip_sep = 9.06u`, `p0 > p1 > p2 > p3 >= 12°`)!
  - By contrast, for a **`CONCAVE-UP (∪)` upward-sweeping gothic crown diadem** (e.g. *Malzeno v34*, `135.8° -> 107.3° -> 82.2° -> 71.7°`): launch back-and-up (`[-52°, 14°, 22°]` on a `+Y` horn) then apply **positive child pitch deltas** (`[+26°, -4°, -10°]` -> `[+24°, -4°, -10°]`) so the blade sweeps cleanly forward-and-up into vertical/forward tips!

### 7.2 Horn Size, Physical Ridges & `16 px/block` Surface-Trait Protocol
Every hero/boss horn must combine **disciplined silhouette geometry** with **species-authentic `16 px` pixel relief**:
1. **Substantial Root Mass & Clean Taper:** Size brute/fanged-wyvern bull horns at `>= 0.40×` cranium width at the root (`4.0 × 3.5u` on a `9.0u` skull), tapering `~75–85%` per segment (`4.0 -> 3.5 -> 2.5 -> 1.5u`) so they read muscular rather than twig-thin.
2. **Physical Stepped Dorsal Corrugation Ridges (`hb_ridge_1..2`):** Instead of slicing a horn into `8+` micro-wafers (`G11` violation) or leaving it a smooth featureless cylinder, mount `1–2` compact stepped dorsal ridge cuboids (`[3.5, 1.5, 4.0]` -> `[3.0, 1.5, 3.5]`, `rotate: [+8°..+10°, 0, 0]`) along the upper/outer horn curve so side and `3/4` silhouettes show unmistakable sawtooth horn ridges.
3. **4-Part `16 px` Horn Surface Paint Grammar:**
   - **Dorsal / Anterior Transverse V-Chevron Grooves (`up`, `south`):** Paint rhythmic `2–3 px` dark-umber -> bright-gold transverse V-chevrons across segment seams (`0%` candy-cane per-box reset!).
   - **Lateral Species Texture (`west`, `east`):** Carve staggered `1×1` / `2×1 px` dark porous fulgurite/keratin pits (e.g., *Zinogre*'s Thunderbug horn pores) or lengthwise dark-stone flutes (`Anchor 06`).
   - **Darkened Root Cuticle Collar:** Shade the basal `2–3 px` of `hb_1` `1–1.5` ramp steps darker where the horn erupts from the brow/skull.
   - **Terminal-Segment-Only Honed Tip:** Restrict the brightest ivory-gold specular ramp step strictly to `hb_tip` (never striping every intermediate segment tip).

### 7.3 Fanged-Wyvern & Heavy-Forequarter Quadruped Compact Neck Law
- **Never clone an Elder-Dragon `4–5` segment S-neck onto a Fanged Wyvern or muscular quadruped predator** (*Zinogre*, *Odogaron*, *Magnamalo*, *Tigrex*, lions, wolves, bears):
  - **Target Ratio (`head : neck : torso : tail`):** **`1 : 0.70–0.85 : 3.0 : 2.8`** (net neck forward projection **`~11–14u`** on a `16u` skull — less than half an Elder Dragon's `28–34u` neck!).
  - **Construction (`3` Deeply Seated Thick Segments, `seat = 0.34–0.38`):** Build `neck_1` (`15.5w × 13.5h × 7.5L`) -> `neck_2` (`13.0 × 11.5 × 6.5`) -> `neck_3` (`10.5 × 9.5 × 6.0`), deeply seated `36%` into each parent, flanked by thick `±28°`-rolled `neck_1_trapezius_l/r` wedges and crowned by a voluminous swept-back 3D mane (`chest_mane_core` + `neck_1..3_mane_core` + serrated alpha fur locks) so the skull sits coiled and powerful right in front of the colossal `1.52×` V-tapered barrel shoulders!

### 7.4 Universal Part-by-Part Grid-Then-Ultra-Zoom Visual-Spatial Reasoning & T-Pose Ratio Verification (`v31` Generalization Law)
- **Do NOT Hardcode One Default Leg, Neck, or Horn Template Across All Creatures!**
  - The real root cause behind *White Fatalis* and *Zinogre* initially having short, nearly-straight rear legs (`0.92–0.98×` of forelimb length when stretched into T-pose) — as well as earlier horn-direction and neck-length errors — was **whole-image vision-encoder compression**: inspecting only full-canvas renders compresses folded limb segments, small horn patterns, and joint angles so the model falls back to shallow guesses instead of seeing the reference's actual structure!
- **Universal 2-Pass Part-by-Part Spatial Reasoning Method (Apply to EVERY Major Body Region of ANY Real or Imaginary Creature):**
  1. **Pass 1 — Whole-Reference `0.10` Coordinate Grid:** Lay a numbered `0.0–1.0` coordinate grid across the cleanest orthographic turnarounds, official renders, and in-game screenshots, and log exact `(x_min, y_min, x_max, y_max)` crop boxes for **every smaller anatomical region independently** (`Head / Eyes / Jaw`, `Horns / Crests`, `Neck-to-Chest`, `Forelimbs & Paws/Hands`, **`Hindlimbs & Feet`**, `Wings / Tail`).
  2. **Pass 2 — High-Resolution Regional Ultra-Zoom Analysis:** Open and inspect **each smaller part zoomed in close-up** so its exact structure is crisp and un-compressed, answering three species-specific questions directly from the pixels:
     - **Fold Topology:** How many anatomical segments make up THIS creature's forelimbs and hindlimbs, and how do they fold? (e.g., a 3-segment digitigrade Z-fold on dragons/wolves/raptors vs. columnar pillars on elephants/golems vs. plantigrade limbs on bears/primates vs. sprawled lizard limbs vs. multi-jointed arthropod legs!)
     - **Stretched-Out (T-Pose) Segment-Length Ratio:** If every folded joint in the zoomed reference is unfolded straight, what is the true ratio of `hind_total : fore_total`, `neck : head : torso`, and `proximal : mid : distal` segment lengths? Never shrink a deeply folded limb into short straight pegs just to reach `y = 0` under a low pelvis!
     - **Species-Specific Sanity Checks:** Encode the ratios and fold structure measured *for this specific subject* into its build verification (e.g., on *White Fatalis `v31`* and *Zinogre `v31`*, Pass-2 limb ultra-zooms revealed a long 3-segment digitigrade Z-fold with stretched `hind/fore ≈ 1.35–1.45×`, verified via `knee_z > hip_z + 5.5u`, `hock_z < knee_z - 7.0u`, `foot_z > hock_z + 4.0u` and wide extension `joint_limits` so T-pose, strides, leaps, and hovers unfold authentically!).


---

## 8. Anti-Stretched-Snout Cranial Compactness Law & Sculpted Oral Cavity / Cohesive Dentition Craft (`v35`)

> **Origin Real-Artist Critique (`v34 -> v35`):** Real human Blockbench artists reviewing our `v34` sculpted dragon/wolf heads immediately flagged **"the face is stretched too long"** — and direct head-local measurements proved ALL THREE `v34` rigs suffered the exact same systemic bias: stacking `4` telescoping snout boxes pushed `snout_ahead_of_eye / head_L` to **`60.6–62.7%`** (turning lupine and aristocratic dragons into crocodile muzzles!), while leaving bulky outward-poking `1×2` tooth cubes glued to cheek lips and flat paper-thin `0.5u` tongues!

### 8.1 Mandatory Species-Calibrated `snout_ahead_of_eye / head_L` Proportionality Gate (Measure on Post-Zoom Gridded Side Profile — NEVER Over-Shorten Long-Jawed Dragons!)
Before approving ANY creature, animal, or boss head, zoom into the subject's orthographic side profile FIRST and run `visual_primitives_prep.py zoom --autocrop --with-grid --with-edges` to measure its **true species-specific snout ratio** in head-local coordinates (`head` bone basis):
$$\text{snout\_ahead\_of\_eye} = \frac{z_{\max}(\text{snout}) - z(\text{eye\_center})}{z_{\max}(\text{head}) - z_{\min}(\text{cranium})}$$
**CRITICAL BALANCE RULE (`v35`):** Do **NOT** apply one blanket "short snout" percentage across every species! While wolves/fanged wyverns look stretched above `52%`, regal long-necked Elder Dragons (*White Fatalis*) and aristocratic vampire dragons (*Malzeno*) look **pug-nosed / stubby** if over-shortened below `53%`! Always calibrate directly to the subject's own post-zoom gridded side profile within its anatomical archetype band:

| Species / Archetype | Target `snout_ahead_of_eye / head_L` | Target Closed-Jaw `head_L / head_H` | Structural Snout Stage Budget |
|---|---:|---:|---|
| **Lupine / Feline / Brute Predators** (wolves, lions, bears, fanged wyverns like *Zinogre*, T-Rex) | **`40–48%`** | **`1.15–1.45`** | **`2.5` deep stages** (`snout_rear` -> `snout_front` -> compact overhanging `snout_tip`), `snout_H >= 65–75%` of `cranium_H` |
| **Aristocratic / Sleek Wyverns & Vampire Dragons** (*Malzeno*, sleek drakes, raptors) | **`52–56%`** | **`1.60–1.90`** | **`2–3` sleek tapered stages** ending in a downward-hooked anterior beak/prow closing cleanly over the lower chin |
| **Regal Long-Jawed Elder Dragons / Serpentine Dragons** (*White Fatalis*, Eastern/classical dragons) | **`56–60%`** | **`1.85–2.30`** (excl. beard/horns) | **`3` refined tapering stages** (`snout_rear` -> `snout_front` -> `snout_beak_hook`) paired with a matching elongated lower mandible |
| **Long-Snouted Crocodilians / Gharials / Sea Serpents ONLY** | **`60–70%`** | **`2.30–3.20`** | Only when orthographic reference explicitly confirms an elongated aquatic rostrum |

- **Automated Build Assertion:** Add an explicit build assertion verifying `snout_ahead_of_eye / head_L` and `head_L / head_H` directly against your subject's Pass-2 post-zoom gridded orthographic side-profile measurement so snouts neither stretch into alligators NOR collapse into pug faces!

### 8.2 Sculpted Multi-Plane Oral Cavity & Cohesive `32x` Tapered Dentition Grammar (Original Aesthetic — Zero Copy-Paste)
When a creature opens its mouth to snarl, bite, or roar, the inside of the mouth must read as a **deep, sculpted biological chamber**, never a hollow box with floating square teeth:
1. **Vaulted V-Ribbed Upper Palate (`upper_palate` & snout underside):** Paint **bold V-shaped transverse fleshy palatal rugae ribs** converging along a dark central median raphe, shading smoothly from warm moist anterior rose/mulberry behind the front fangs into deep cavernous shadow toward the rear throat.
2. **Sealed Dark Pharynx (`mouth_throat_back`):** Close the rear throat opening with a deeply shaded abyssal pharyngeal box (`t <= 0.14`) so wide `35–48°` roars never reveal empty background sky or neck hinges.
3. **Volumetric Grooved Sculpted Tongue (`tongue_1 -> tongue_2`, `>= 1.0u` thick):** Build a substantial, muscular 2-segment tongue filling the lower mandible trough cleanly (NOT a paper-thin `0.5u` flat strip!), painted with a **deep central longitudinal sulcus groove**, twin fleshy lateral crowns, subtle moist papillae texture, and shadowed outer edges.
4. **Volumetric 3D Sculpted Fangs OR Cohesive `32x` Recessed Arcades (BAN Bulky Exterior Cheek Tooth Pegs & Bright Lipstick Strips!):**
   - **Never** glue bulky tooth cubes sticking sideways off exterior maxilla cheek walls.
   - **Preferred Studio Master Option — Recessed 3D Volumetric Cuboid Fangs (`ufang_canine_l/r`, `ufang_premolar_l/r`, `ufang_molar_l/r`, `ufang_incisor_row`, matched on lower jaw):** For hero/boss predators where roaring close-ups matter, carve real 3D volumetric cuboid fangs (`[1.0, 1.5, 1.0]` anterior canines, `[1.0, 1.0, 1.5]` carnassials/molars, `[1.5–2.0, 1.0, 1.0]` incisors) recessed **cleanly INSIDE the closed lip walls** (`anchor_x ~ ±0.54..±0.74`, `anchor_y ~ ±0.40..±0.45`) with full cross-jaw `blend_with` pairs — **100% hidden when the mouth is closed at rest**, yet gleaming with genuine 3D thickness, enamel root-to-tip shading, and depth from every angle when roaring!
   - **Lightweight Option — Cohesive Tapered Serrated `32x` Alpha-Plane Arcades:** When element budget is tight, seat crisp tapered triangular serrated alpha-plane fang rows cleanly inside the upper/lower lip margins with **transparent interdental gaps (`alpha = 0`)** all the way to the recessed shadowed lip line — **NEVER paint a bright pink/magenta solid horizontal gum bar across the front of a tooth plane**, which reads like bright lipstick in closed `3/4` face views!
5. **100% Fully-Closed Rest Mouth Law (ZERO Exposed Teeth or Red Gap at Rest!):**
   - **Default Universal Rule:** Unless a species specifically grows external sabers/tusks outside the lips (e.g., wild boars, walruses, sabretooths), **the rest/bind pose mouth MUST close completely flush (`jaw rest_euler[0] ~ -2°..0°`) with ZERO exposed teeth and ZERO red interior palate peeking through**!
   - **Exact 3D Enclosure Construction:**
     1. Seat the narrower lower mandible (`jaw_mid`, `jaw_chin`) snugly **UP INSIDE** the upper maxilla lip rails (`maxilla_l/r`, `snout_front`, `snout_tip`/`beak_hook`) so the upper lip cleanly roofs over the occlusal seam.
     2. Raise `upper_palate` (`anchor_y ~ -0.64..-0.76`) and tuck all 6 tooth planes (`upper_teeth_l/r/front`, `lower_teeth_l/r/front`) **inward (`anchor_x ~ ±0.60..±0.76`, `anchor_y ~ ±0.45..±0.58`)** completely inside the closed outer lip walls, declaring cross-jaw `blend_with` pairs so the closed bite passes SAT with `0` collisions.
     3. Keep `idle` breathing clips at `mandible_depressor` `peak=0.0, baseline=0.0` (closed restful bite!), and give `mandible_depressor` `moment_arm_deg: 38–44°` so roar, howl, bite, and threat clips swing the jaws **wide open** revealing the full sculpted V-ribbed palate, deep dark pharynx, volumetric grooved tongue, and razor `32x` fangs!


---

## 9. V-Shaped Sloped Head & Mandible Faceting, Pure Reference Material Boundaries & Individual `0.5u` 3D V-Arch Dentition (`v37`)

> **Origin Critique & Studio Breakthrough (`v36 -> v37`):** Inspecting exported `.bbmodel` creature heads against official renders and master sculpted 3D Blockbench dragon/wolf heads exposed three subtle failure modes caused by over-strict cube-size habits:
> 1. **Unreferenced `snout_ridge` Material Artifacts:** Adding a contrasting golden/accent dorsal `snout_ridge` block on top of a snout that is **100% pure primary scale/hide color** in official artwork creates an immediate "weird block on the snout" artifact.
> 2. **Boxy Stair-Stepped Snouts vs. V-Shaped Sloped Heads:** Relying only on thick `>= 1.5–2.0u` rectangular snout steps makes sleek lupine, draconic, and wyvern muzzles look chunky and blunt rather than forming a **sharp, converging V-shaped sloped wedge** with a serrated upper-lip overhang.
> 3. **Coarse `1.0–1.5u` Block Teeth vs. Individual `0.5u` V-Arch Dentition:** At head scale (`6–9u` wide), `1.0u`-wide cuboid teeth look like thick square chiclets (`4–6` blocks total), whereas **purposeful `0.5u` sub-unit 3D teeth** (`16–22` individual teeth arranged along the converging V-arch of the upper palate and lower gum bed) read razor-sharp, clean, and anatomically fierce!

### 9.1 Strict Reference Material-Zone Fidelity on Snouts & Brows (BAN Unverified Accent Blocks!)
- **Never invent accent-colored dorsal snout ridges, nose plates, or golden brow trims** just to "add detail"! Before assigning any secondary/accent material (`gold`, `horn`, `bone`, `crystal`) to a snout bridge, maxilla cheek, or brow, zoom `100%` into the official front and `3/4` head renders and verify **exact material boundaries**:
  - If the official render shows a **pure monochromatic scale/hide snout** (e.g., *Zinogre*'s pure blue-teal nasal bridge, V-cheeks, and brow with dark nose leather, where golden armor starts *only* at the main horns, rear cheek spurs, and lower mandible armor), keep every upper-snout and brow facet in that pure primary material (`teal_scale`) — **DELETE** any unreferenced `snout_ridge` accent block!

### 9.2 Scoped `0.5u` Cranial, Snout, Lip & Dental Allowance (`G11` Exemption Without Over-Voxelizing!)
- **Why Scoped Small Blocks Win on Heads:** While torso, limb, and tail shafts strictly forbid `>= 6` thin `< 1.0u` wafer slices (`G11` Anti-Fake-Blender rule), **heads, snouts, lips, jaws, chin prongs, and dentition are high-focus sculptural zones** where purposeful `0.5u–1.0u` facets (`construct_check.py` `G11` cranial/dental exemption) build crisp **V-shaped sloped surfaces** without turning the whole model into high-voxel noise!
- **4-Part V-Shaped Sloped Snout & Mandible Architecture:**
  1. **Converging V-Sloped Cheek Wedges + Flush Nasal Bridge Slope:** Pair a compact internal snout core (`snout_rear` -> `snout_front`) with **converging V-angled cheek wedges** (`maxilla_l/r`, yawed **`Ry = +18°..+22°` Left / `-18°..-22°` Right** and rolled `Rz = ∓14°..∓18°`), bridged on top by a flush `1.0u` sloped dorsal plate (`snout_bridge_slope`, `Rx = +8°..+12°`) and an upturned/hooked nose tip (`snout_tip`) — tapering smoothly from wide orbital skull (`8–9u`) to a sharp `2.5u` lupine/draconic point while obeying `snout_ahead_of_eye / head_L` (`§8.1`)!
  2. **Serrated Overhanging Upper-Lip Rails (`snout_lip_l/r` + `lip_serr_l/r`):** Run slim `1.0 × 1.0u` converging upper-lip rails (`Ry = ±19°..±21°`) + purposeful `0.5 × 0.5 × 1.5u` carnivorous lip serration steps along the lower margin of `maxilla_l/r`. Paint a `1 px` dark sub-lip shadow edge along their ventral border so the closed jaw line reads crisp and predatory.
  3. **Converging V-Angled Lower Mandible Rails & Multi-Step Chin Prongs:** Mirror the upper V-wedge on the lower jaw using **V-converging lateral mandible armor rails** (`jaw_rail_l/r`, `[1.5, 2.0, 4.5]`, `Ry = ∓18°..∓20°`) meeting at `jaw_chin`, plus `2–3` progressive stepped chin spikes/prongs (`1.5u -> 1.0u -> 0.5u` chiselled tip) where supported by the reference.
  4. **Individual `0.5u` 3D Volumetric V-Arch Dentition (`16–22` Discrete Teeth + Crimson Alveolar Gum Rim):**
     - Group upper and lower teeth onto dedicated clean dental child bones (`upper_dental` / `upper_teeth_l,r` and `lower_dental` / `lower_teeth_l,r`, keeping `<= 10` boxes per bone for `G1` hygiene!).
     - Place **individual `0.5u`-thick 3D ivory cuboids** along the **converging V-curve** of `upper_palate` and `jaw_gum_bed`:
       - *Anterior Incisors:* `[0.5, 1.0, 0.5]` near the front apex (`x ~ ±0.35u`);
       - *Prominent Fang Canines:* `[0.5, 1.5, 0.5]` at the antero-lateral corner (`x ~ ±0.90u`, pitched/rolled slightly outward-forward along the V-arch);
       - *Graded Premolars & Carnassial Molars:* `3` pairs per jaw (`[0.5, 1.0, 0.5]` -> `[0.5, 1.0, 1.0]`) marching back along `Ry = ±16°..±18°` with real `0.35–0.55u` negative-space gaps between every tooth!
     - **Oral Interior Contrast:** Frame the `0.5u` ivory V-arch teeth with a **moist crimson-coral outer alveolar gum margin** on `upper_palate` and `jaw_gum_bed`, contrasting against dark V-ribbed posterior palatal rugae and a **deep violet-plum / mulberry grooved 2-segment tongue (`tongue_1 -> tongue_2`)**.
     - **100% Closed-Lip Concealment Verification:** Verify programmatically in head-local coordinates that **all `20–28` `0.5u` teeth sit cleanly inside `maxilla_beam_l/r`, `lip_serr_*`, `snout_tip`, and `jaw_beam_l/r`** when the jaw is closed (`0` exposed teeth at rest!), while gleaming cleanly along both V-arches at `30–42°` roar!

### 9.3 Open-Basin Lower Jaw, Half-Phase Interlocking Sawtooth Occlusal Wave, Converging Alveolar Gum Rails & 2-Tone Mouth Interior Formula (`v37` Transferable Master Recipe)
Whenever an artist or AI model builds a realistic predatory, draconic, lupine, dinosaur, or boss head from references (or adapts an existing base model), **do NOT guess individual tooth coordinates or fill the lower jaw flat to the brim like a solid block!** Follow this exact 5-part geometric + texturing spatial relation:

1. **Open Armored U/V-Basin Lower Jaw Trough (`jaw_rear -> jaw_mid -> jaw_chin` + Raised Converging `jaw_beam_l/r`):**
   - Lower the structural jaw floor (`jaw_rear -> jaw_mid`, `height ~ 2.0u`) by **`0.8–1.0u` BELOW** the top crest of the raised inward-converging lateral armor rails (`jaw_beam_l/r`, `height ~ 3.0u`, Left `Ry = +12°..+15°` -> **ALWAYS verify true inward convergence `front_w < rear_w`**, never accidental Reverse-V `-Ry`!), capped at the front by a solid `jaw_chin` symphysis wall (`height ~ 2.5u`).
   - Seat `jaw_gum_bed` and the flat, center-grooved 2-segment tongue (`tongue_1 -> tongue_2`) inside that recessed center trough — and assert mathematically **`jaw_gum_bed_zmax <= jaw_chin_zmax - 0.8u`** and **`tongue_2_zmax <= jaw_chin_zmax - 0.8u`** so oral tissue **never pokes through the front of `jaw_chin`**!
2. **Converging Alveolar Gum Rails (`upper_gum_rail_l`, `lower_gum_rail_l`) + `+0.55–0.65u` Upper-Over-Lower V-Wall Width Offset:**
   - Inside mirrored child bones `upper_teeth_l` (`mirror_x(upper_teeth_r)`) and `lower_teeth_l` (`mirror_x(lower_teeth_r)`), create Box `#1` as a slim converging **alveolar gum rail** (`size: [0.5, 0.5, L]`, `rest_euler: [0, +14..+16°, 0]` upper / `[-2°, +11..+13°, 0]` lower), placing the upper rail **`+0.55–0.65u` wider laterally (`|x|`)** than the lower rail so upper teeth close cleanly *outside* lower teeth while staying strictly *inside* `maxilla_beam_l/r` and `lip_serr_1..4_l/r`!
   - Place `6–7` discrete `0.5u` volumetric teeth directly along `base_box=upper/lower_gum_rail_l` at evenly-spaced normalized Z-stations (`+0.92, +0.66, +0.38, +0.10, -0.18, -0.46, -0.74` -> **`~0.50u` crisp negative-space gap** between every tooth!): tilt the anterior corner fangs (`ant_l`) slightly inward-and-upward leaving a clean **`1.6–2.0u` open midline gap** directly in front of the tongue tip, follow with a taller anterior canine (`2.0u` tall), and march `5` evenly-spaced cheek teeth back along the V-wall.
3. **2-Tone Tooth Shading, Crimson-to-Violet Oral Contrast & Taut Striated Buccal Cheek-Tendon Webs:**
   - **2-Tone Volumetric Teeth:** Paint the basal **`35%` tooth root** warm ochre-tan dentin (`[146, 122, 84]` -> `[188, 166, 124]`) with a dark gingival collar, stepping crisply into **gleaming ivory-white enamel (`[240, 236, 224]` -> `[255, 254, 250]`)** across the upper `65%` crown.
   - **Oral Basin & Tongue Contrast:** Paint `upper_gum_rail`, `lower_gum_rail`, and the outer/posterior rim of `jaw_gum_bed` vivid moist **crimson-scarlet (`maw_crimson`)**, transitioning via an **organic ragged pixel border** on `tongue_1` into a **flat, center-sulcus-grooved deep violet-plum / mulberry tongue (`tongue_1 -> tongue_2`)**.
   - **Taut Buccal Cheek-Tendon Membranes (`plane jaw_cheek_web_l/r`):** Span the rear jaw corner (`anchor_z ~ +0.65..+0.75` on `jaw_rear`, `size: [4.5, 5.5]`, `normal: x`) with a **dedicated buccal-web plane shader** (NEVER accidentally routed into fur/teeth shaders!) painting vertical stretched maroon-crimson muscle fibers with a concave hard-alpha front scoop — sealing the rear jaw corner at `36–44°` roar with `0` empty floating gaps!
4. **Half-Phase Interlocking Sawtooth Occlusal Bite Wave (`\/ \/ \/ \/` Upper + `/\ /\ /\ /\` Lower):**
   - When references feature jagged/serrated interlocking jaws (wolves, fanged wyverns, dragons, crocodilians), sculpt bold angled sawtooth facets along **BOTH** the lower rim of the upper maxilla (`lip_serr_1..4_l/r`, `size: [1.0, 1.5, 1.5]`, `Rx = +30°..+34°`) **AND** the upper rim of the lower jaw (`jaw_serr_1..4_l/r`, `size: [1.0, 1.5, 1.5]`, `Rx = -30°..-34°`), **staggered by half a wave-period along Z** so upper teeth-rim points nest snugly into lower jaw valleys when closed! Enhance the bite interlock optically in `32x` texture with a `1 px` dark occlusal shadow along the meeting edge and a `1 px` sunlit bevel glint on each angled triangle crest.
5. **Sculpted Upturned Rhinarium Nose Bump, Shingled Ventral Chin & Gracefully Curved Chin Prongs:**
   - Model the anterior lupine/draconic rhinarium with a subtle proud upturned **`snout_nose_bump`** (`rotate: [-18°..-20°, 0, 0]`) above `snout_tip` flanked by flared `nostril_ala_l/r` — keeping the entire dorsal snout pure primary scale/hide while carving an **inverted-triangle dark nose-leather pad + central philtrum cleft** on `snout_tip`'s front face and dark recessed nostril pits on `nostril_ala_l/r`.
   - Break flat box-bottom mandibles with **`4–5` rhythmic stepped ventral armor shingles (`jaw_vent_1..5`, `Rx = -18°..-22°`)** + **fanned posterior jaw-hinge spurs (`jaw_beard_top/mid/bot`)**, and curve protruding twin chin prongs gracefully (**`CONCAVE_UP` dip-then-scoop `+18° -> -14° -> -18°` + subtle `( )` curve**, never stiff straight horizontal rods!).

---

## 10. Deep Facial Trait Proportional Remapping & Scale-Harmonious Eye Recess Law (`v38`)

> **Core Reasoning Principle:** Across real wildlife, stylized characters, and fantasy creatures alike, AI modelers repeatedly fall into **four linked cranial errors** whenever they sculpt heads by intuition instead of measuring proportional relationships on references first:
> 1. **Oversized Cartoon Eye Ratio (`eye_h / head_h = 30–60%`):** Inflating 3D `eye_l/r` cuboids so iris paint looks "bigger" — breaking graceful animal/predator/character likenesses.
> 2. **Outward Eye Extrusion ("Bug-Eye / Earmuff Protrusion"):** Mounting `eye_l/r` laterally proud of `cranium_core` in `0°` Dead-Front or `85°` Top-Down views (`eye_vs_cranium > +0.08u`).
> 3. **Buried / Blocked Eye Occlusion:** Placing `head_core`, `snout_mask`, `throat_mask`, or `maxilla` overlapping `>= 40%` of the eye's outward viewing face.
> 4. **Mixel Over-Resolution on Small Critters (`uv_scale >= 3`):** Painting `96x–128x` hyper-dense micro-irises on tiny `small`-tier heads, clashing unnaturally with their `16x`/`32x` bodies.

### 10.1 Mandatory 8-Ratio Deep Facial Trait Analysis Table (Complete BEFORE Head Cube `#1`!)
Before authoring `W1`, run `visual_primitives_prep.py zoom --autocrop --with-grid --with-edges` on the subject's **Dead-Front (`0°`)**, **Front-3/4 (`35°`)**, and **Side (`85°`)** face crops, and explicitly measure & log all **8 facial ratios** into `PROJECT_GOAL.md` so every cuboid remaps the true species proportions instead of guessing weird ratios:

| # | Facial Trait Ratio / Landmark | How to Measure on `0.10` Zoomed Face Grid | Target Calibration & Anti-Distortion Guard |
|---|---|---|---|
| **F1** | `eye_h / head_h` *(Vertical Eye Ratio)* | Eye aperture height ÷ cranium core height (excluding tall ears/horns/crests) | **NEVER oversize 3D eye cuboids to get more pixels — use `uv_scale: 2` (`64x`) on compact species-accurate cubes!** See taxon bands in §10.2. |
| **F2** | `eye_l / eye_h` *(Aperture Aspect)* | Horizontal eye opening length ÷ vertical eye height | Narrow fox/predator almond `1.7–2.4:1`; alert bird/primate `1.0–1.3:1`; vertical slit reptile `1.4–2.0:1`. |
| **F3** | `eye_y_ctr / head_h` *(Orbital Elevation)* | Eye center height from jaw-bottom to skull crown | Lower-mid (`40–52%`) cute critters/birds; upper-mid (`58–74%`) predators, crocodilians, foxes, dragons. |
| **F4** | `eye_z_ctr / head_L` *(Orbital Depth)* | Eye center along snout-to-occiput length (`0%` occiput → `100%` nose tip) | Directly governs `snout_ahead_of_eye`: `42–48%` foxes/wolves, `52–60%` dragons/crocodilians, `25–38%` short-billed songbirds/primates. |
| **F5** | `eye_outer_x - bezel_outer_x` *(Orbital Recess)* | Compare outermost `|X|` of `eye_l` vs. surrounding `brow`/`eyelid`/`cheek`/`cranium` in `0°` Dead-Front & `85°` Top-Down | **MUST satisfy §10.3 Master Orbital Sweet Spot (`<= -0.15u` flush/recessed inside bezel)** on all non-stalked/non-bulging taxa! |
| **F6** | `snout_w / cranium_w` & `snout_h / head_h` | Width & height of anterior muzzle step vs. cranium width/height | Taper smoothly in 2–3 sloped steps (`0.72×` per step) — ban abrupt rectangular "brick-nose" steps on foxes, birds, and dragons! |
| **F7** | `inter_eye_gap / head_w` & `brow_overhang_pct` | Front-view bridge width between inner eye corners + `%` of top eye covered by upper eyelid/brow | Prevents cross-eyed squeezing OR wide-set fish-eyes; overhang `15–25%` on furious bosses/predators, `0–10%` on gentle wildlife. |
| **F8** | `ear_or_crest_h / head_h` & `cheek_ruff_w / head_w` | Crown appendage height & cheek-fur/frill flare relative to cranium | Locks silhouette proportions (`fox` tall pointed ears + lateral cheek tufts, `macaque` broad horizontal ears, `songbird` sleek cheek contour). |

### 10.2 Tier-Harmonious Eye Resolution, Simplification & Taxon Ratio Bands (`v38` — Ratio First, NO Weird Mixel Over-Resolution on Small Critters!)
> **Universal Scale-Harmony Rule:** **Match eye proportion accurately to the reference FIRST — AND keep eye pixel resolution & micro-carving harmonious with the model's scale!**
> - **NEVER use `uv_scale >= 3` (`96x–128x` hyper-dense mixels)!** Putting ultra-dense `96x/128x` iris art on a tiny `1.0u` eye clashes harshly with a `16x`/`32x` body and feels unnatural in Blockbench.
> - **Tier-Harmonious Eye Detail & Cuboid Abstraction Policy (Protect Boss Craft, Simplify Small Critters Deliberately!):**
>   - **`small` Wildlife, Pets & Critters (`head_h < 6u` — small songbirds, penguins, reef-eels, moles, snails, frogs, insects, lizards, jerboas, small rodents):**
>     1. Use **abstract, iconic Blockbench eyes at native body texel density (`uv_scale: 1` at `16x` or `32x` — clean `2×2` to `3×3 px`, 2–3 bold colors: dark pupil + vivid iris accent + optional `1 px` specular pixel, NEVER photorealistic 7-layer micro-gradients!)**.
>     2. **Drop cluttered 4-part micro-eyelid slivers** (one clean flush-seated eye cuboid or flush head-face paint, zero overlap with `head_core`/`snout_mask`/`throat_mask`).
>     3. **Use honest on-level cuboid abstraction on simple round sacs, honey droplets, and plump bellies** (1 bold proud cube or 2–3 crisp stacked cuboids instead of over-carving 10–16-box fake spheres!).
>   - **`standard` Creatures & Animals (`head_h = 5–7u` — foxes, primates, medium fish, mollusks, mounts):** Match native body density (`uv_scale: 1` at `32x`, clean graphic `3×3..5×5 px` eyes) or at most `uv_scale: 1.5–2` only when the eye is very slim, framed by 1–2 clean brow/cheek contours.
>   - **`hero` & `epic_boss` Monsters / Large Figures (`head_h >= 7u` — dragons, wyverns, crocodilians, giant beasts, mechs — UNCHANGED!):** Full `32x` cross-cube sculpted relief + **`uv_scale: 2` (`64x` max)** 7-layer focal eyes + recessed 4-part periorbital socket (`eyelid_upper`, `eyelid_lower`, `brow`, `orbital_post_bar`).

General taxon vertical eye ratio bands (`eye_h / head_h`, with `eye_above_crown <= 0.00u` and zero blocking by `head_core`/`snout_mask`/`maxilla`):
- **Canines, Foxes, Felines & Agile Mammals:** `eye_h / head_h = 0.14 – 0.21`, sleek horizontal almond (`eye_l / eye_h = 1.7 – 2.2`).
- **Large Reptiles, Dragons, Wyverns & Aquatic Predators:** `eye_h / head_h = 0.12 – 0.20`, fierce angled predator aperture (`eye_l / eye_h = 1.5 – 2.2`).
- **Songbirds, Waterfowl, Penguins & Small Lizards:** `eye_h / head_h = 0.15 – 0.24`, flush-set clean `uv_scale: 1` eye seated cleanly clear of beak/throat masks and inside the cheek contour.
- **Expressive Primates, Reef-Eels & Nocturnal Mammals/Owls:** `eye_h / head_h = 0.20 – 0.30` — framed cleanly with `0%` overlap by `head_core`/`maxilla_side` and **NEVER exceeding `cranium_ymax`**!
- **Explicitly Dorsal-Turret / Stalked / Compound-Head Taxa Only (toads, frogs, insects, stalked mollusks, crustaceans):** Match exact reference dome/stalk ratio directly on the zoomed grid (`uv_scale: 1` on `small` tier!) — never invent oversized eyes or artificial `96x–128x` mixels!

### 10.3 The Master Orbital Recess Sweet Spot (Simultaneous Anti-Extrusion & Anti-Occlusion Law)
For every non-bulging vertebrate (`-X` left side, mirrored symmetrically to `+X`):
$$\text{maxilla\_outer\_}|X| \;<\; \text{eye\_outer\_}|X| \;\le\; \text{periorbital\_bezel\_outer\_}|X| - 0.15\text{u}$$
1. **ZERO Outward Extrusion (Anti-Bug-Eye):** In `0°` Dead-Front AND `85°` Top-Down views, the lateral face of `eye_l/r` MUST sit **flush with or recessed `0.15–0.50u` INSIDE** its surrounding 4-part periorbital frame (`eyelid_upper`, `eyelid_lower`, `brow_ridge`, `cheek_zygoma` / `face_mask`). Never paste an eyeball box onto the outside of a skull wall!
2. **ZERO Anterior Occlusion (Periorbital Sightline Window):** Simultaneously, keep `maxilla_wall` / `snout_side` strictly narrower (`|X_maxilla| < |X_eye|`) and lower (`maxilla_top_y <= eye_min_y + 0.50 * eye_h`), and keep `cheek_top_y <= eye_min_y + 0.15u`, ensuring `>= 45%` clean binocular sightline in `0°` Dead-Front AND `>= 65%` in `35°` 3/4 and `85°` Side!
3. **Mandatory Dead-Front (`0°`) + 3/4 (`35°`) + Side (`85°`) Macro Audit:** Before advancing past `W2` (and again at `T2`), inspect all three close-up angles together and verify: (a) NO eyeball protrudes laterally past the cheek/brow wall, (b) `eye_h / head_h` matches the measured reference table within `±0.04`, and (c) the eye is crisp and unobstructed from all 3 angles!
