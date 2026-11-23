# Universal Blockbench Spatial Reasoning, Multi-Archetype Craft & Expert Double-Check Principles

> **Purpose:** A franchise-agnostic, repeatable engineering and artistic reasoning framework for general Blockbench 3D modeling across **any subject** — real wildlife (mammals, birds, reptiles, amphibians, fish, mollusks, arthropods, pelagic life), stylized characters, fantasy/sci-fi creatures, vehicles, mechs, props, and environments. Every rule below is a generalizable geometric or visual invariant paired with a concrete self-check method so an artist or AI model reasons from first principles rather than memorizing one narrow skeleton template.

---

## 1. First-Principles Spatial & Rotation Reasoning (Never Guess Signs!)

Before assembling any articulated hierarchy in a right-handed Y-up coordinate system (`+Z` forward, `-Z` backward, `+Y` up, `-X` left; Euler `Rz * Ry * Rx`):

1. **Derive Every Axis Sign Explicitly Before Keying or Placing:**
   - **Anterior (`+Z`) chains** (necks, heads, hinged jaws, cannons): `-Rx` pitches UP, `+Rx` pitches DOWN (lower jaw opens with `+Rx`).
   - **Posterior (`-Z`) chains** (tails, rear stabilizers, trailing capes): **`+Rx` lifts UP (`+Y`)**, `-Rx` drops DOWN (`-Y`).
   - **Left-side (`-X`) vertical limbs (`-Y`):** `-Rz` splays OUTWARD (`-X`); `+Rx` sweeps BACKWARD (`-Z`), `-Rx` sweeps FORWARD (`+Z`).
   - **Left-side (`-X`) forward digits / rails (`+Z`):** **`+Ry` fans MEDIAL (`+X`)**, **`-Ry` fans LATERAL (`-X`)**.
   - **Left-side (`-X`) lateral wings / fins / arms (`-X`):** `-Ry` sweeps tip BACKWARD (`-Z`), `+Ry` sweeps FORWARD (`+Z`); `-Rz` raises tip UP (`+Y`), `+Rz` lowers DOWN (`-Y`); `-Rx` pitches a trailing (`-Z`) sail UP away from the flank/thigh.
2. **Programmatic Endpoint Verification (Self-Check Rule):**
   - For any multi-branch extremity (toes, fingers, splayed horns, wing spars, radially arranged legs/tentacles), **never assume visual separation from angles alone** — programmatically assert world-space tip ordering (e.g., `x_outer < x_middle < x_inner` on a `-X` foot) and minimum spread before advancing out of White-Model `W2`!
3. **Plane Back-Face UV Mirroring Invariant:**
   - On vertical planes (`normal: x` or `normal: z`), aligning front/back alpha cutouts flips **U** (`[x+w, y, x, y+h]`).
   - On horizontal planes (`normal: y`, `up`/`down`), both faces share `-X → +X` along U and mirror along Z, so back-face alignment **MUST flip V** (`[x, y+h, x+w, y]`).

---

## 2. Deep Facial & Focal-Feature Proportional Remapping + Scale-Harmonious Detail

Weird proportions, bug-eyed faces, and jarring mixel resolution happen when an artist jumps straight into placing cubes by intuition instead of measuring proportional relationships on references first.

### 2.1 Mandatory 8-Ratio Pre-Sculpt Facial & Focal Trait Analysis
Before placing Cube `#1` on any head, cockpit, mask, or focal landmark, overlay a `0.10` coordinate grid on zoomed **Dead-Front (`0°`)**, **Front-3/4 (`35°`)**, and **Pure Side (`90°`)** references and record all 8 ratios:
1. **Primary Focal Height Ratio** (`eye_h / head_h`, or `visor_h / helmet_h`): Measure directly on the grid — never inflate 3D eye cuboids just to paint more pixels!
2. **Focal Aperture Aspect** (`eye_l / eye_h`): Sleek horizontal predators/canines `1.7–2.4:1`, alert birds/primates `1.0–1.3:1`.
3. **Vertical Elevation** (`eye_y / head_h`): Lower-mid (`40–52%`) small cute fauna; upper-mid (`58–74%`) predators, aquatic reptiles, tall heads.
4. **Anterior-Posterior Depth** (`eye_z / head_L`): Directly locks muzzle/snout proportion (`snout_ahead_of_eye`).
5. **Lateral Recess vs. Cranial Envelope** (`max_eye_x - cranium_half_w`): Must sit flush or recessed inside the core skull (`<= +0.08u`) on all non-stalked/non-turret taxa.
6. **Anterior Taper Ratio** (`snout_w / cranium_w`, `snout_h / head_h`): Taper smoothly across 2–3 sloped steps (`~0.72×` per step) — avoid abrupt rectangular box-noses.
7. **Inter-Feature Spacing & Brow Overhang:** Prevent squeezed cross-eyes or overly wide-set eyes; keep brow overhang `<= 25%` on fierce subjects and `<= 10%` on gentle subjects.
8. **Crown & Cheek Anchors** (`ear/crest/horn_h / head_h`, `cheek_w / head_w`): Anchor ears, crests, horns, and cheek ruffs directly to grid proportions.

### 2.2 The Simultaneous 4-Invariant Eye Placement Check (Automated in `construct_check.py G14`)
Every non-stalked focal eye MUST satisfy **all four invariants simultaneously**:
1. **Honest Proportion:** `eye_h / head_h` matches the reference band (`0.12–0.21` sleek mammals/predators/dragons, `0.15–0.24` birds/reptiles, `0.20–0.30` big-eyed primates/nocturnal critters — never `> 0.32`).
2. **Zero Outward Extrusion:** `max_eye_abs_x <= cranium_core_half_w + 0.08u` (no bug-eyes or earmuff rings bulging past the skull wall in `0°` Dead-Front).
3. **Zero Crown Overflow:** `eye_top_y <= cranium_top_y + 0.10u` (eyes never tower above the top of the head except on genuinely dorsal-turret amphibians or stalked eyes).
4. **Zero Viewing-Face Occlusion:** No solid `head_core`, `cranium_core`, `snout_mask`, `throat_mask`, or `maxilla` cuboid may clip or cover `>= 40%` of the eye's outward viewing face!

### 2.3 Scale-Driven Abstraction, Anti-Over-Carved-Sphere Law & Balanced Small-Entity Eyes (`v38` Two-Tier Balance)
**Always infer physical world scale directly from the provided references FIRST, and match geometric abstraction and eye complexity strictly to that scale — never breaking our sculpted Hero/Epic Boss style, and never over-carving small entities into uncanny high-poly spheres!**

- **Tier A — Large Heroes & Epic Bosses (`tier = hero | epic_boss`, physically large creatures/mechs, `head_h >= 7u`):**
  - **PRESERVE FULL SCULPTED CRAFT UNCHANGED:** Rich compound cuboid muscles/armor, recessed 4-part periorbital sockets, and **`uv_scale: 2` (`64x` max) 7-layer focal eyes** with crisp cross-cube `32x` pixel-relief carving.
- **Tier B — Small Entities, Critters, Wildlife & Compact Props (`tier = small`, inferred as physically small from references, `head_h < 6u`):**
  - **Anti-Over-Carved-Sphere & Anti-Fine-Curve Law (Curing the "Weird Feeling"):** The uncanny, un-Minecraft feeling on small models comes directly from burning 8–16 rotated or octagonal cubes trying to carve a smooth sphere or fine curve (e.g., a honey/nectar droplet, a plump songbird belly, a round amphibian torso, a fruit/berry, or an orb). **Whether a round volume bounces OR holds a fixed shape, represent it cleanly in honest Blockbench cuboid abstraction — 1 bold proud primary cube (or 2–3 crisp stepped cuboids) and let painterly pixel-art corner-darkening & specular highlights sell the roundness!** Reserve multi-band octagons strictly for large structural domes, conical shells, or mechanical wheels where a flat box clearly breaks silhouette likeness.
  - **Balanced Expressive Small-Entity Eyes (Clean Graphic Readability — NO Over-Busy `64x–128x` Mixels!):** Keep small-entity eyes **compact and flush-recessed** (`eye_h/head_h = 0.15–0.26`, zero outward bulge, zero head/mask blocking, zero crown overflow) with a **clean, expressive Blockbench graphic motif at native resolution (`uv_scale: 1` at `16x`/`32x` — crisp dark pupil + 1 clear sclera/iris accent + 1 crisp white specular highlight pixel)**, framed simply by the head/cheek contour (drop fussy 4-part micro-eyelid slivers and photorealistic 7-layer radial gradients!).
- **Hard Rule Across ALL Tiers:** **NEVER use `uv_scale >= 3` (`96x–128x` mixels)** — extreme resolution mismatch on a tiny eye clashes with the body's pixel grid.

---

## 3. Universal Thin-Surface, Dual-Sided Membrane & Zero-Detachment Craft

Whenever any model includes thin surfaces — wings, fins, ears, capes, sails, banners, tail fans, leaf blades, or cloth:

1. **Zero-Blink `0.5u` Dual-Sided Solid Slab Law (Blockbench / OpenGL Z-Fighting Prevention):**
   - Putting **two different textures** on opposing faces of a `0.0u` zero-thickness plane — or stacking adjacent `0.0u` planes at the same normal depth — causes flickering Z-fighting in Blockbench and Minecraft!
   - Whenever front/back (or dorsal/ventral) sides carry distinct patterns OR overlap consecutively, build them as **solid `0.5u`-thick `box` slabs**:
     - Pack opposing primary faces (`up`/`down` or `east`/`west`) into separate full-res UV rects.
     - Map all **4 thin `0.5u` side rim faces** to a single transparent `alpha = 0` texel `[0, 0, 1, 1]` for razor-thin organic silhouettes with zero blocky side walls.
     - Alternate normal stagger (`±0.55–0.70u`) across consecutive overlapping bays/segments so NO two surfaces share identical depth!
2. **Deep Structural Burial & Continuous World-Space Carving (Zero Animation Detachment):**
   - Edge-tacked membranes split open during flapping, flexing, or whipping animations.
   - Bury leading/root edges **`1.5–2.8u` deep** inside their parent structural spar or spine vertebra, overlap consecutive segmented membrane panels by `4–8u` along the chain axis, and evaluate organic alpha cutouts & surface gradients in **continuous world-space coordinates** across the entire wing/fan/cape!
3. **Pragmatic Scope Discipline:**
   - Lock clean silhouette fidelity, zero Z-fighting blink, zero motion detachment, and faithful graphic surface patterns — stop before burning endless passes trying to mimic smooth Blender-curved surfaces beyond Blockbench's cuboid medium.

---

## 4. Universal Multi-Archetype Massing, Limbs & Collision Hygiene

Never force every creature or machine onto one default quadruped/humanoid template:

1. **Anti-Wedding-Cake & Anti-Shipping-Crate Organic Curvature (`v39.0` Sweet Spot):**
   - Never build curved domes, shells, bells, turrets, or rounded hulls out of blunt stepped vertical cylinders (`wedding-cake stacks`), 16-box noisy staircases, OR raw single flat-roofed `90°` cardboard shipping crates (`top10 > 95%`)!
   - **Rounded/Fluffy Creature Heads & Puffballs (`top10 = 78–92%`):** Anchor on `1` bold core (`60–75%` volume) plus **`2–4` purposeful bevel/taper planes** — an inset domed crown cap, softly tapered cheek/snout/rump pads, and (for furry/feathered silhouettes) crisp angled `0.5u–1.0u` fur-tuft, bang, or feather-ruff overhang planes that break raw `90°` box corners cleanly from every angle! On articulated flush-closing mouths/petals, stop hood bevels `0.25u` behind the hinge plane and continue matching bevels onto the hinged petals so rest stays smoothly rounded while full gape remains `0`-collision!
   - **Radial Bells, Shells & Domes:** Use **true convex regular octagons (`4` `prism_group` rectangular prisms at `0°/45°/90°/135°` with `side = D * tan(22.5°)` & progressive `inflate: 0.00..0.06` + inset crown dome — zero square pizza-box corners AND zero shuriken star notches!)**, **smoothly tilted overlapping conical whorls**, and **broad flowing continuous fin/cape planes** instead of narrow disconnected sticks!
2. **Universal Limb Posture & Joint-Limit Awareness:**
   - **Analytical `SO(3)` Contact Leveling:** Solve exact wrist/ankle/sole rest Euler rotations so all grounded contact pads sit dead-flat at `y = 0.00u` (`verify_bbmodel_ground.py`).
   - **Posture-Specific Kinematics:** Erect limbs (`hinge_elbow`), sprawled reptilian limbs (`elbow_sprawled`), pre-bent digitigrade legs (rest-pose-relative `hinge_knee` bounds), radially splayed 6-/8-leg arthropod legs (coplanar vertical radial IK planes), rigid hinged wing-covers/shields (`elytron_hinge`), and free/vestigial upper arms (suppress blind full-arm gait swings on tiny `< 0.35×` paws).
   - **Isolated Per-Digit Collision Groups:** NEVER put sibling toes, fingers, or claws into one shared `sculpt_group` (e.g., `leg_sc`), which silently hides self-intersections from SAT collision audits! Isolate each digit (`digit_1_sc`, `digit_2_sc`, ...) and blend only its proximal base into the palm/foot pad.
3. **High-Res Unblocked Extremity References:**
   - Whenever hands, feet, claws, beaks, hinges, or mechanical joints are occluded or blurry in full-body art, fetch clean high-res close-ups from non-hotlink-blocked sources (official renders, artbooks, sculpts, figure photos — avoid broken/cropped wiki hotlinks) before `W2`!
4. **Physical Prop-Interaction & Coupled-Posture Law (Zero Detached Sidekick Props!):**
   - Whenever an entity **pushes, rolls, carries, clutches, perches on, or inhabits an accompanying prop or payload** (e.g., rolling beetles pushing spheres/cubes, insects clutching nectar sacs, hermit crabs inside shells, workers carrying loads, riders on mounts, or characters bracing shields/weapons):
     - **NEVER float or park the prop detached behind/beside the actor with an empty air gap in rest pose!** That breaks silhouette cohesion immediately (e.g., showing two separate floating shapes in Side/Top-Down silhouette checks).
     - **Derive the Actor's Rest Posture Directly Around the Contact Interface FIRST:** Match the exact reference posture — such as a steep nose-down handstand push (`+35°..+45°` body tilt, forelegs planted on `y = 0`, mid + hind legs reaching back/up to press flush against the prop's sides & upper face with `<= 0.1u` contact clearance), or firm ventral leg-clasp / shell insertion — so the actor and prop read as **one unified interactive silhouette** in Front, Side, Top-Down, and Silhouette views!
     - **Couple Prop Physics to Limb Kinematics:** Animate the prop synchronously with the pushing/carrying limbs (e.g., rolling cube rotating around its center while pushing legs stroke against its surface, or carried payload bobbing with inertia) while keeping `0` illegal interpenetrations at `288` frames (`blend_with` or exact contact clearance between gripping tips and prop face).
5. **Small Flying Animals & Avian Proportion, Wing-to-Torso & Head-Nesting Law (Birds, Owls, Songbirds, Bats & Small Flying Creatures):**
   - Small flying animals are a major Blockbench category where two predictable proportion errors recur if artists sculpt from sparse 1–2 image sets: **undersized stubby wings relative to the torso**, and **stiff stacked boxy heads on vertical pillar bodies**.
   - **Wing-Size-to-Torso Proportion Rule:** Measure real wing length and feather/membrane chord directly from spread-wing photos or video frames BEFORE massing:
     - **Folded at rest:** Folded wings should span **`~75%–100%`** of the dorsal torso-to-rump length hugging the flanks cleanly (never tiny side-paddles dangling halfway down!).
     - **Unfolded / Flight / Courtship Display:** Unfolded wingspan should reach **`2.2×–3.8×` torso length** (wider still for soaring or display-fan birds!) with real feathered/membranous primary & secondary chord built as zero-blink `0.5u` dual-sided slabs whose inner roots blend seamlessly into the shoulder mantle (zero empty shoulder-root air gap in Top-Down/Front views!).
   - **Avian Head & Torso Integration (Zero Vertical Pillar or Stacked-Snowman Silos!):**
     - **Streamlined Perchers, Songbirds, Owls, Raptors & Hummingbirds:** Pitch the primary torso gently (`~18°–35°` breast-to-rump avian slant) and taper/nest the head & neck ruff smoothly into the chest/mantle contour — never stack a flat square head block rigidly atop a straight vertical column!
     - **Plump Round Puffball Birds & Small Mammals (chicks, crakes, round songbirds, dandelion fuzzballs):** Keep the body and head as **ONE cohesive sculpted puffball mass** (NEVER stack a giant second square head cube on top like a 2-block snowman, AND NEVER leave it as a raw flat-roofed `90°` shipping box!) — anchor on `1` bold rounded core wrapped with a gentle **inset domed crown cap**, **tapered cheek/chest/rump bevel pads**, and delicate **angled `0.5u–1.0u` feather/fur-tuft & wing-ruff planes** (`top10 = 78–92%`), seating the cute native-resolution eyes and tiny beak cleanly on the sculpted upper-front face!
   - **Multi-Angle Reference Discovery Before Sculpting Any Small Flyer:** Never build a bird or small flying creature from `1–2` static renders — verify exact species/colorway identity (`0` cousin-species placeholders!) and escalate to **YouTube / video ecology & flight observation footage** whenever static photos omit side profile, rear plumage, spread-wing span, or courtship/flight motion (`references/high_res_focal_carving_and_visual_primitives.md §1`).

---

## 5. Universal Full-Body Motion & Secondary-Chain Responsiveness

Whether an asset walks, hops, flies, swims, floats, or operates mechanically:

1. **Match Locomotion Physics to the Medium:**
   - **Terrestrial:** In-place `gait {}` (`in_place: true`, `max|root_z| == 0.00u`, matched `synccraft_root_motion`), foot-roll + multi-phalanx digit articulation, `ground_pin_limbs` on stationary/combat poses.
   - **Aerial:** Aerodynamic Figure-8 `flight {}` (`hover`, `cruise`, `glide`, `soar`, `takeoff_burst`) with `pelvis_root` roll `= 0°` on declarative symmetric clips (author banked rolls on keyframed aerial clips).
   - **Aquatic / Pelagic:** Native underwater buoyancy & musculoskeletal positioning (`swim {}`: neutral `buoyancy_y`, traveling axial waves with `head_stabilize`, bilaterally symmetric `hydro_tuck` / `paddle_row` / `pectoral_glide`, and lateral `Sim(3)` myomere scale contractions!).
2. **100% Full-Skeleton Kinetic Chain & Anti-Static-Appendage Rule:**
   - Never leave distal segments of multi-segment tails, capes, antennae, ears, eyestalks, or fins un-driven or acting as a fixed rigid curve!
   - Drive **every vertebra from root to tip** with progressive **`25–45°` phase lag per segment** and monotonically increasing distal amplitude — lagging downward on upward launch, flowing through an S-wave at apex, and whipping upward on touchdown — verified quantitatively via per-bone angular excursion (`bone_angular_excursion_deg`) AND visually via a 4-phase sequential motion strip!
3. **Mandatory Stage A0 Video & Moveset Motion Reference Research (NEVER Guess Known Animations!):**
   - Whenever animating a **known boss, character, real animal, vehicle, or documented fictional creature**, **NEVER invent combat, guard, sleep, or signature behavioral poses from generic text assumptions!**
   - Before authoring Clip `#1`, actively search **video platforms (especially YouTube — e.g. `"<subject> all attacks"`, `"<subject> moveset"`, `"<subject> real footage"`) and authoritative moveset sources** via web search and browser video inspection to document the real **4-Beat Choreography** (`Anticipation -> Active Peak Pose -> Follow-Through -> Recovery`) and **exact world-space appendage targets** (e.g., armored wings folding *forward in front of the chest/face* as a shield wall rather than flaring out to the sides; a tail spear driving *forward past the flank/overhead* toward the target rather than curling behind; a sleep clip genuinely dropping belly & folded wings flat to `y = 0` rather than standing upright!).
   - **Blind-Label Filmstrip Silhouette Test:** Inspect every rendered `A2` filmstrip with the clip title mentally covered — if a viewer cannot immediately recognize the exact signature action from the 3D silhouette alone, **reject and re-choreograph**, even if swept-SAT reports `0` collisions!

---

## 6. Repeatable Expert Multi-Pass Workflow & Double-Check Checklist

Never batch-script multiple stages in one loop without inspecting visual renders side-by-side against references:

| Pass | Focus | Mandatory Visual & Quantitative Double-Check Before Advancing |
|---|---|---|
| **Stage 0** | Multi-Angle References & Spatial Plan | Gather **`>= 4–5` verified multi-angle references** (exact species/colorway identity — ZERO wrong-variant placeholders! Escalate to YouTube/video ecology & turnaround footage whenever static images `< 4` angles); run `visual_primitives_prep.py` zoom + `0.10` grid + edges; log **8-Ratio Facial/Focal Table** & wing-to-torso/silhouette ratios. |
| **W1** | Primary Massing | Check macro proportions, stance, center of gravity, and silhouette against references; log `>= 3` fixes. |
| **W2** | Head/Focal Gate & Limbs | Render **Dead-Front (`0°`), 3/4 (`35°`), Pure Side (`90°`), Top-Down (`85°`)** close-ups; verify all **4 eye invariants** (`G14`), isolated digit spread, and joint tapers; log `>= 3` fixes. |
| **W3** | Refined Detail & Audit | Eliminate staircase cylinders, floating parts, and `0.0u` dual-sided Z-fighting; pass `audit_scene` (`0` coll, `0` cont) & `construct_check.py` (`rc=0`, `0` WARN). |
| **T1** | Unlit Macro Value Gate | Block major value zones unlit; pass `25%` squint silhouette readability check. |
| **T2** | Scale-Harmonious Surface Relief | Paint hue-shifted `16x`/`32x` cross-cube pixel relief + scale-appropriate eyes (`uv_scale: 1` on `small`, `<= 2` on `hero/epic_boss`); compare `T2_8panel` + `T2_closeup` side-by-side against references. |
| **A0** | Video & Moveset Motion Study | On any known boss/character/animal, search YouTube / video platforms + moveset references FIRST; log `ANIMATION_REFS.md` 4-beat phases and explicit `+Z/-Z/±X/+Y` limb/wing/tail targets so zero signature attacks are guessed. |
| **A1–A2** | Full-Body Motion & Export | Verify `100%` active spine/limb/tail/appendage chains, `288`-frame swept-SAT (`0` collisions), `verify_bbmodel_ground.py` (`PASS=True`), `construct_check.py` across `.blockdsl`, `.bbmodel`, `.geo.json` (`rc=0`, `0` WARN), AND pass the **Blind-Label Filmstrip Silhouette Test** against `A0` references. |
