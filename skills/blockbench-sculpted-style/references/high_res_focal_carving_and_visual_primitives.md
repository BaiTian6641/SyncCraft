# High-Resolution Focal Carving (`32x` + `2x` Eye UV), Proactive Multi-Angle Reference Expansion, Multi-Track Spike Mapping & Horn/Spike Collision Policy (`v37`)

> **Universal Craft Mission:** Elevate AI-authored Blockbench models from solid `16x` silhouettes to **human-master studio polish** across *every* 3D subject — real animals, imaginary creatures, stylized humanoids, vehicles, mechs, armor, and props — by combining six formal, transferable pillars:
> 1. **DeepSeek-Inspired Visual Primitives, Native-Resolution Edge Inspection & Proactive Web/3D-Viewer Reference Search** (whenever references feel insufficient, search the internet for orthographic/top-down/3D-viewer close-ups — NEVER guess!).
> 2. **Universal Curvature Concavity Verification (`CONCAVE_UP ∪` vs. `CONCAVE_DOWN ∩` vs. `S_INFLECTION ~`)** on every curved appendage, horn, antler, beak, tusk, tail, claw, blade, wing spar, or structural arch before writing `@rig`.
> 3. **Multi-Track Body-Spike & Circumferential Limb-Ring Topology Mapping + Horn/Body-Spike Collision Policy (`v37`)** (mapping distinct continuous dorsal/lateral spike tracks and curved limb rings from top-down + side grids without false SAT breaks).
> 4. **Higher Texture Resolution (`32 px/block`) + Per-Element `2x` Focal UV Density (`uv_scale: 2`)** without losing unmistakable Blockbench clarity.
> 5. **4-Part 3D Periorbital Carving & Realistic Multi-Layer Eye / Optical-Sensor Painting**.
> 6. **Anti-Toy Carving on Tapered Appendages (Tails/Limbs), Laminated Plates, and Mounted Accessories** at *both* the modeling scale and the `32x` texture scale.

---

## 1. Formal Visual Primitives & Native-Resolution Reference Pipeline

Inspired by **"Thinking with Visual Primitives"** (DeepSeek / PKU / Tsinghua) and native visual-grounding workflows, AI spatial-reasoning failures stem from two separate bottlenecks:
- **The Perception Gap:** Crushing high-res references into tiny compressed JPEG contact-sheet thumbnails (`~400–520 px`) destroys eyelid rims, beveled plate seams, tail scale overlaps, joint inflection points, and accessory brackets before the vision encoder ever sees them.
- **The Reference Gap:** Natural language alone (*"the tail tapers smoothly"*, *"the eye is recessed"*, *"rear leg folds like a Z"*) is spatially ambiguous during Chain-of-Thought, causing proportion drift and simplified box geometry.

### Mandatory 4-Stage Visual Primitives Protocol (`scripts/visual_primitives_prep.py`)

Execute these four stages using the bundled zero-extra-dependency script `scripts/visual_primitives_prep.py` (Pillow + NumPy) BEFORE writing `@rig` and AGAIN when auditing white-model (`W1–W3`) and textured (`T1–T3`) passes:

1. **Stage A — Collect Multi-Perspective Native References, Proactively Expand Insufficient Angles on the Internet, & Study Ad-Free Video Frames (`v37` Zero-Guessing Reference Law):**
   - Gather **multiple perspectives** like a human artist: `0°` front, `90°` side profile, `35°` hero 3/4, **orthographic top-down dorsal**, rear/caudal, and close-ups of focal zones (eyes/head, open/closed oral interior, neck-to-shoulder tracks, upper-limb circumference, limb joints, tail/plates, accessories).
   - **Mandatory Proactive Internet & 3D-Viewer Search Rule (`v37`):** **Whenever the initially provided or cached references feel insufficient, foreshortened, shadowed, or leave ANY structural detail ambiguous** (for example: how dorsal/shoulder spike tracks spread from neck to tail in top view, whether an upper limb carries a circumferential spike ring vs. vertical studs, exact snout material boundaries, or interior jaw/tooth arrangement), **STOP and proactively search the internet (`search_web` + `browser` subagent)** for official 3D model turnarounds, interactive Sketchfab / 3D viewers, artbook orthographics, and high-res multi-angle captures! **NEVER guess unseen anatomy or mechanical routing when additional detailed references exist online!**
   - **Video Timelapses & Motion Studies:** Always inspect real modeling/texturing timelapses and in-motion video frames, NOT only static renders. **Strictly filter out ads, pre-roll commercials, intros, and mid-roll sponsor segments** — jump directly to verified modeling, UV, and painting timestamps (`yt-dlp` + `pyav` native frame seek).
   - **ZERO Downsampling & ZERO Bundled External Images Law:** Save working reference crops in your conversation's scratch/artifact directory as **lossless PNGs at 1:1 native pixel resolution** (never crush a `1024–1920 px` reference into a `400 px` thumbnail tile, and never pin or redistribute external artists' reference images inside the skill folder itself — distill structural knowledge into reusable rules!).

2. **Stage B — Native Macro Coordinate Grid + Structural Edge/Contour Extraction (`grid --dual-edges`):**
   ```bash
   python3 scripts/visual_primitives_prep.py grid <ref.png> <ref_grid_dual.png> --autocrop --dual-edges
   ```
   - Overlays a `0.10`-spaced `0.0–1.0` coordinate grid at **100% native pixel resolution** paired side-by-side with a **multi-scale Scharr + chromatic + DoG structural ink edge-map** for macro body landmarks and regional crop selection.

3. **Stage C — ZOOM FIRST, Cut Out the Clean Full-Color Native Region for Multimodal Vision, THEN Perform Crisp 1 px NMS Color-Context Contours + Post-Zoom Grid (`zoom --autocrop --with-grid --with-edges` + `trace`):**
   ```bash
   # ALWAYS zoom & cut out each regional crop FIRST -> outputs BOTH:
   #   1) <region>_native.png (100% UN-FILTERED, UN-GRIDDED full-color cutout for direct multimodal vision!)
   #   2) <region>_zoom_grid_edges.png (3-Panel Strip: [Clean Native Color Cutout | Post-Zoom 0.10 Grid | Crisp 1 px NMS Color-Context Contours])
   python3 scripts/visual_primitives_prep.py zoom <ref.png> <out_dir> --boxes '{"region":[x0,y0,x1,y1]}' --autocrop --with-grid --with-edges
   # Then run explicit Visual Primitives dashed skeleton / joint angles / concavity / calipers:
   python3 scripts/visual_primitives_prep.py trace <ref.png> <spec.json> <ref_trace.png> --autocrop --edge-blend 0.25 --export-crops-dir <crops_dir>
   ```
   - **Mandatory Post-Zoom Anti-Blur & Multimodal Color-Cutout Rule (`v35`):**
     1. **Never rely on blurry grayscale edge filters alone!** Standard thick Gaussian/Sobel black-and-white edge filters blur armor seams, swallow muscle shading, and sever color-to-structure reasoning. `visual_primitives_prep.py` uses **edge-preserving bilateral smoothing + Di Zenzo multi-channel color-tensor gradients + 8-neighbor Canny Non-Maximum Suppression (NMS) + hysteresis contour linking** (`color_ink` mode) to extract **razor-crisp `1 px` structural contours** overlaid on a clean `34%` full-color context wash!
     2. **ALWAYS feed the clean, un-filtered full-color native cutout (`<region>_native.png`) alongside the 3-panel gridded+contour plate directly into multimodal vision!** Seeing the **un-filtered full-color cutout** side-by-side with the **post-zoom `0.10` coordinate grid** and **razor-sharp `1 px` NMS structural contours** lets the artist/AI connect **true muscle mass, material value contrast (deep shadow crevices vs. sunlit bevels), 3D raised studs/spikes, layered overlapping limb gauntlets, and wide outward-splayed toe angles** directly to exact 3D coordinates!
     3. **Zoom into EVERY major region FIRST (`Head/Jaw`, `Anterior Chest/Torso`, `Dorsal Back/Shoulders`, `Lateral Flanks`, `Forelimbs/Gauntlets/Paws/Toes`, `Hindlimbs/Feet`, `Tail/Accessories`)** — never skip limbs, paws, toes, or shoulder spikes!
   - Read exact `(x, y)` coordinates from Stage B/C and explicitly draw **Visual Primitives**:
     - **Dashed Skeleton Centerlines (`- - -`)** tracing `spine -> neck -> head`, `tail`, `forelimb`, `hindlimb`, `wings`, or mechanical `chassis -> boom -> turret` with numbered **joint waypoints (`<point>`)**.
     - **Automatic Joint Bend Angles (`θ°`)** printed directly on every interior node (e.g., `knee (77°)`, `hock (68°)`, `ankle (145°)`), preventing straight-post legs, pigeon-toed inward-folded toes, or wrong horn sweeps.
     - **Dotted Proportion Calipers (`· · ·`) & Automated Ratios** (e.g., `hind/fore_chain = 1.34x`, `hip_h/shldr_h = 0.96x`, `head_L/head_H`, `eye_W/head_L`) pinned into the header so proportions are verified quantitatively, not eyeballed.
     - **Focal Bounding Boxes (`<box>`)** around `head_eye`, `torso_chest`, `torso_flanks`, `forelimbs_paws`, `hindlimbs_feet`, `tail_plates`, and `accessories` — `--export-crops-dir` automatically exports **lossless 1:1 native color cutouts + 3-panel post-zoom grid & crisp 1 px NMS edge maps** for every single focal box!

4. **Stage D — Inspect Unfiltered Native Color Cutouts + 3-Panel Gridded/Crisp-Edge Plates & Repeat on Model Renders:**
   - Open every exported clean `<region>_native.png` AND 3-panel `<region>_zoom_grid_edges.png` directly via vision (`read_file`) to map exact closed-jaw occlusal alignment, eye-socket carving, bilateral torso & shoulder horns, **muscular limb volume + layered gauntlet shingles + 3D osteoderm boss studs + outward-splayed heavyweight toes**, rich chiaroscuro color/value contrast (deep dark shadow crevices vs. sunlit bevels — never flat pastel boards!), plate shingle direction, tail cross-section, curvature concavity, and accessory mounts.
   - During White-Model (`W2/W3`) and Textured (`T2/T3`) passes, run `zoom --autocrop --with-grid --with-edges` / `trace` on our own orthographic renders and compare **dashed skeleton ratios, curvature concavity signs, muscular silhouette weight, outward toe splay, and crisp 1 px contours** directly against the reference cutouts!

---

## 1B. Universal Curvature Concavity Verification Law (`CONCAVE_UP ∪` vs. `CONCAVE_DOWN ∩` vs. `S_INFLECTION ~`, `v34`)

> **Why Endpoint Direction Alone Fails:** A subtle, high-impact visual reasoning error occurs whenever a builder checks only **where a curved structure points** (*"horns point forward and up"*, *"tail lifts up"*, *"blade hooks forward"*) without verifying **which way the curve bends (`concavity` / 2nd derivative)**!
> Two horns can BOTH start at the same crown root and terminate at the exact same forward-and-up tip `(z, y)` — yet one curls **CONCAVE-UP (`∪`)** like a banana/bull hook (launching shallow, then curving steeply skyward) while the other arches **CONCAVE-DOWN (`∩`)** like an oni/ram spear (launching steeply up-and-forward off the skull, then arching smoothly forward over the brow while still pointing ~20° above horizontal)! That single sign flip instantly breaks silhouette fidelity across **horns, antlers, tusks, beaks, tails, necks, dorsal spines, claws, talons, scythes, wings, vehicle fenders, and gothic arches**.

### Mandatory 3-Step Curvature Concavity Check (Before Writing `@rig`!)

Apply this check to **every multi-segment curved structure (`>= 3` segments)** on ANY subject (creatures, real animals, humanoids, weapons, vehicles, mechs):

1. **Step 1 — Trace the Structure's True Orthogonal Profile at 1:1 Native Resolution (`visual_primitives_prep.py trace`):**
   - Pick the cleanest **pure side profile** (for sagittal pitch curvature: horns, beaks, necks, tails, spines, claws, canopies) or **dead-front / top view** (for lateral bow/pincer curvature: lyre horns, tusks, insect mandibles, trident pincers, swept wings).
   - Trace `>= 3–4` centerline waypoints (`root -> base -> mid -> tip`) with `"label_concavity": true` in the `trace` spec JSON.
   - `visual_primitives_prep.py trace` automatically computes the **signed 2D turning deflection (`signed_deflections_deg`)** along the canonical horizontal flow direction (`+X_flow` left-to-right or right-to-left, normalized so `+` ALWAYS curves upward toward `-Y_img` and `-` ALWAYS curves downward toward `+Y_img`) and classifies the curve deterministically:
     - **`CONCAVE_UP (U)`** (`signed_deflections > +5°`): bends **upward** along its progression (cup/bowl/crescent opening toward the sky — e.g., sweeping vampire crown diadems, upturned scorpion/club tails, upward-hooking bull tusks, curved scimitar blades, upturned secondary antler tines).
     - **`CONCAVE_DOWN (arch)`** (`signed_deflections < -5°`): bends **downward** along its progression (arch/dome opening toward the ground — e.g., forward-arching oni/ram spears, raptor beaks, downward-hooking claws/talons, vaulted bridge/canopy arches, sweeping dragon necks cresting over the head).
     - **`S_INFLECTION (~)`** (opposite signs along proximal vs. distal joints): reverses concavity mid-span (e.g., swan/dragon S-necks, long regal stag/celestial main antler beams, recurve bows, aerodynamic airframe spines).
     - **`STRAIGHT (--)`** (`|signed_deflections| <= 5°`): straight tapering lance/rod/strut.

2. **Step 2 — Map Measured Concavity Directly to Right-Handed Euler Deltas (`euler_xyz`, ZERO Sign Guessing!):**
   | Structure Direction in Local Space | Desired Sagittal Concavity | Required Progression of Child Pitch Deltas (`rest_euler[0]`) | World YZ Elevation Angle (`atan2(dy, |dz|)`) |
   |---|---|---|---|
   | **Forward-projecting (`+Z`)** (forward horns, beaks, tusks, forward brow crests) | **`CONCAVE_DOWN (∩)` arch** | Steep negative launch (`rx_1 = -55°..-68°`) followed by **POSITIVE** child deltas (`rx_2, rx_3 > 0`, e.g. `+12° -> +9° -> +6°`) | **Monotonically DECREASING** (`58° -> 42° -> 29° -> 20°`, still pointing up-and-forward!) |
   | **Forward-projecting (`+Z`)** (upturned bull hooks, warthog tusks) | **`CONCAVE_UP (∪)` scoop** | Shallow launch (`rx_1 = -10°..-25°`) followed by **NEGATIVE** child deltas (`rx_2, rx_3 < 0`, e.g. `-16° -> -18° -> -14°`) | **Monotonically INCREASING** (`15° -> 31° -> 49° -> 63°`) |
   | **Upward-projecting (`+Y`)** (crown horns, vertical antlers, dorsal crests) | **`CONCAVE_UP / FORWARD (∪)` diadem** | Swept-back launch (`rx_1 = -45°..-58°`) followed by **POSITIVE** child deltas (`rx_2, rx_3 > 0`, e.g. `+26° -> +24°`) | Sweeps smoothly from backward (`136°`) -> vertical (`107°`) -> forward-up (`82°`)! |
   | **Backward-projecting (`-Z`)** (swept horns, tails, dorsal fins) | **`CONCAVE_UP (∪)` sweep** | Low/level launch followed by **POSITIVE** child deltas (`rx_2, rx_3 > 0`, since `Rx(+θ) @ (0,0,-1)` lifts `-Z` upward!) | **Monotonically INCREASING** upward lift along `-Z` |
   | **Backward-projecting (`-Z`)** (drooping/arching spears, tail hooks) | **`CONCAVE_DOWN (∩)` arch** | High launch (`rx_1 = +28°..+45°`) followed by **NEGATIVE** child deltas (`rx_2, rx_3 < 0`) | **Monotonically DECREASING** along `-Z` |
   | **Lateral Pincer / Lyre (`±X` bow)** (lyre horns, trident outer prongs, mandibles) | **`CONCAVE_INWARD ( | )`** | Base flares **OUTWARD**, distal segments yaw **INWARD** toward midline (`mid_sep > tip_sep`) | Outer arc bows widest mid-span and pincers cleanly inward at the tips |

3. **Step 3 — Guard Against 3D Euler Coupling via World-Frame Pitch Assertion:**
   - Whenever a curved chain ALSO has non-zero yaw (`ry`) and roll (`rz`) (such as wide-bowing `( )` lyre horns or flared antlers), scalar addition of `rest_euler[0]` is **NOT exact** — 3D rotation coupling can accelerate pitch drop by `5–10°` per segment and accidentally droop an upward-forward horn tip flat!
   - Always verify the compiled 3D segment axes directly in world space (`d_i = box_i.rotation @ local_axis`, `pitch_i = atan2(d_i[1], d_i[2])`) inside your geometry verification check and confirm that:
     1. The **monotonic sign of `pitch_{i+1} - pitch_i`** matches the reference's measured `CONCAVE_UP` / `CONCAVE_DOWN` / `S_INFLECTION` classification, AND
     2. The **terminal tip angle** (`pitch_tip`) matches the reference's native profile angle (e.g., `>= +12°..+22°` upward-forward rather than drooping flat to `0°`).

---

## 1C. Universal Whole-Subject Regional Ultra-Zoom, Category Analysis & Trait Reconstruction Method (`v35` — Anti-Neglected-Torso / Expert Blockbench Law)

> **Critical Generalization Lesson:** Even when an artist or AI model nails a creature's head, eyes, and extremities by ultra-zooming on the face, it frequently **forgets to apply the exact same regional ultra-zoom and trait-reconstruction rigor to the TORSO / CORE HULL** (chest, sternum/prow, dorsal spine/back armor, ribcage flanks, waist/belly — or on vehicles/mechs/props: fuselage/chassis panels, engine bays, vents, hull plating, and structural ribbing)! Leaving the core torso as plain flat cuboids with a generic patch immediately drops the whole model back toward "boxy beginner craft."

Apply this **4-Step Expert Blockbench Reasoning Loop** systematically across **EVERY major region of ANY 3D subject** (creatures, real animals, humanoids, armor sets, vehicles, mechs, architecture, and props):

### Step 1 — Partition the Subject Into `6` Mandatory Native-Resolution Regions & Run Post-Zoom Grid + Outline Trace (NEVER Skip the Torso/Core Hull!)
Run `visual_primitives_prep.py grid --dual-edges` across multiple perspectives (`front`, `side`, `3/4`, `dorsal/rear`) to locate regional crop bounds, **THEN ZOOM FIRST into every region and perform the fine `0.10` coordinate grid AND structural outline/edge trace AFTER zooming** (`visual_primitives_prep.py zoom <ref.png> <x0,y0,x1,y1> <out.png> --autocrop --with-grid --with-edges`) across **all 6 regions independently**:
1. **Region 1 — Head / Face / Canopy / Cockpit:** Cranium compactness (`snout_ahead_of_eye`), 4-part recessed eyes/lenses, brows, horns/crests, jaw & sculpted oral cavity (or glazed canopy + 3D interior).
2. **Region 2 — Anterior Torso / Chest / Sternum / Front Hull Prow:** Chest silhouette (e.g., pointed V-brisket prow, pleated mantle/jabot, segmented sternal plastron, armored cuirass, sloped radiator grille/bumper), neck-to-chest collar integration, and pectoral massing.
3. **Region 3 — Dorsal Torso / Back / Shoulders / Roof Superstructure:** Shoulder cape/mane drape, overlapping dorsal armor shingles/spines/shockers, scapula ridges, wing/arm socket frames, or roof cab/turret/vent mounts.
4. **Region 4 — Lateral Flanks / Ribcage / Waist / Belly / Side Chassis:** Intercostal rib/muscle furrows, lateral osteoderm studs, waist tuck & ventral segmented belly scutes, or recessed wheel wells, fender chamfers, side vents & underbody clearance.
5. **Region 5 — Limbs / Wings / Wheels / Mechanical Actuators:** True joint fold topology & T-pose ratios, forearm gauntlets/spurs, articulated paws/feet/digits (or recessed rims, flat-tread `180°/N` wheels, and alternating armor vs. dark gunmetal joints).
6. **Region 6 — Tail / Rear Assembly / Mounted Accessories:** Tapered faceted/shingled tail geometry, collared/bracketed fins/weapons/mirrors, and clean rear hull closure.

### Step 2 — Category Analysis (Decompose Every Observed Trait Before Placing Cubes!)
Inside EACH zoomed region, explicitly sort every visual trait visible in the native pixels & edge-map into **4 structural categories** so you never confuse geometry with texture or paste flat boxes onto curved volumes:

| Trait Category | What to Identify in the 1:1 Regional Zoom | How to Translate Into Medium-Honest Blockbench Craft |
|---|---|---|
| **Cat 1: Macro Structural Mass & Silhouette Flow** | Overall hull/torso cross-section (hexagonal, V-prow, barrel-to-wasp taper, sloped hood, chamfered cab) | Interlocking primary + secondary `40–80%` sunken cuboids (`chest_core`, `pectoral`, `scapula`, `rib_slope`, `sternum_keel`, chamfered hull deck) |
| **Cat 2: Layered 3D Anatomical / Mechanical Relief** | Overlapping chest/lumbar plates, pointed V-brisket fur prows, shoulder armor collars, osteoderm boss spikes, recessed vents/wheel arches, hydraulic joints | Dedicated compound-angled `inset_carve` / stepped cuboids + `dark` mounting collars (`>= 2` overlapping shingle layers where armor plates stack!) |
| **Cat 3: Thin Directional Fringes, Fur & Membranes** | Sweeping mane capes, collar tufts, feathery jabots, wing sails, windshields, glass canopies | Zero-thickness `100%` hard-alpha planes (`normal: x/y/z`, bilateral `center[0] < 0` symmetry!) or layered translucent glass over recessed dark 3D interiors |
| **Cat 4: Optical Surface Relief & Material Grammar** | Diagonal flank striping / intercostal muscle crevices, imbricated scale clusters, transverse ventral plastron bands, panel seams, rivets, stenciled decals, weathering | High-craft `16x`/`32x` cross-cube pixel-relief carving (`45–50°` hue-shifted shadow valleys + `1 px` leading bevels) mapped across cube boundaries |

### Step 3 — Spatial Trait Remapping (Anchor, Angle, Concavity & Rig Ownership)
For every Cat 1–3 trait discovered in the regional zoom, log its exact 3D mapping BEFORE editing geometry:
- **Exact Landmark Anchor:** Which host box does it emerge from, and how deeply (`40–75%`) is its root seated so zero floating gaps appear during animation?
- **3D Flow & Concavity:** Which way do overlapping shingles, mane locks, flank stripes, or armor bevels sweep (`+Z` forward vs. `-Z` swept-back, `CONCAVE_UP` vs. `CONCAVE_DOWN`)?
- **Transition Continuity:** How does the torso trait flow seamlessly into adjacent regions (`neck mane -> shoulder cape -> dorsal lumbar shingles -> tail`; `front chest prow -> ribbed ventral sternum -> waist belly -> groin`)? Eliminate abrupt unblended box patches (e.g., an isolated square white block stuck on a blue chest)!

### Step 4 — Reconstruct & Verify Region-by-Region Side-by-Side
After rebuilding each region's mapped traits in Blockbench/Block-DSL and textures, render **matched regional close-ups** (`Head`, **`Anterior Chest/Torso`**, **`Lateral Flank/Back`**, `Limbs/Feet`, `Tail`) WITH FULL TEXTURES ON (`clay=False`) and inspect each region directly beside its Stage-1 native reference zoom. Only advance once **every region — especially the torso/core hull — carries the same reference-faithful density and sculptural cohesion as the head!**

---

## 1D. Zero-Floating Structural Attachment, Chain-of-Structures Continuity & Limb Mass Calibration Law (`v36`)

> **Critical Generalization Lesson:** Do **NOT** focus carving discipline solely on horns or isolated showpieces while letting limb spikes, shoulder armor chains, osteoderm studs, fur tufts, or mechanical brackets **float in open air** or attach via thin awkward offsets! A single detached forearm spike or hovering shoulder plate immediately breaks 3D believability across *every* genre (creatures, bosses, humanoids, armor sets, mechs, and vehicles).

### 1. The Zero-Floating Deep-Socket Rooting Law (Every Small Part MUST Physically Embed!)
1. **Never offset surface attachments into open air (`offset` trap):** Never push a shoulder spike, flank wing, forearm spur, osteoderm stud, mechanical thruster, or armor guard outward into thin air with an outward `offset` + outer-edge `align`. Every attached part must bury **30–55% of its root volume directly inside its host limb, hull, or mounting carapace** (verified programmatically via OBB signed distance `sep <= -0.5u` to its host — `sep > 0` is an automatic bug!).
2. **Never chain angled limb spurs off a single tilted spur (`tilted-chain drift` trap):** When mounting a row of `3+` spikes, blades, fins, or pistons down a forearm, shin, tail, or mech arm, **NEVER** attach Spike 2 to Spike 1 and Spike 3 to Spike 2 inside a steeply pitched child bone — stepping down a tilted local `-Y` axis walks the lower spikes straight out into thin air!
   - **Correct Standard Architecture:** First fuse a **continuous parallel mounting ridge / keel** (`ulnar_crest_base`, `shin_ridge`, `armor_rail`) aligned at `rest_euler: [0, 0, 0]` (`50%` embedded along the full length of the limb shaft from elbow/knee to wrist/ankle). Then mount **every individual spike/blade independently onto that solid mounting keel** (`35–45%` buried root + `55–65%` projecting cleanly along the reference's true world sweep angle!).
3. **Zero Flat "Paper-Sticker" Fur / Detail Slabs:** Never paste thick rectangular white fur blocks or flat square bricks flat onto the outer wall of a thigh, elbow, or shoulder. Taper and embed fur bolsters cleanly along posterior joint ridges so layered feathered alpha tufts flow seamlessly over underlying armor roots, and embed osteoderm studs/rivets `>= 50%` deep as angled faceted bosses rather than glued-on rectangular bars.

### 2. Chain-of-Structures Continuity (Interlocked Shoulder, Flank & Hull Chains — Never Isolated Pegs!)
When references show a **chain of spikes, overlapping shoulder plates, lateral flank ridges, or linked mechanical armor collars** wrapping along both sides of a torso or chassis:
- **Trace the full multi-link flow in the zoomed references BEFORE modeling** (e.g., `anterior clavicle collar -> scapular mounting carapace + parallel shoulder spikes -> mid-flank mounting plate + rear flank spikes -> lumbar ridge`).
- **Physically overlap adjacent chain links (`sep <= -0.25u`):** Every mounting carapace plate must interlock solidly into BOTH the underlying torso/hull core AND its neighboring chain plate (`clavicle <-> scapula <-> flank`) with **zero floating gaps and zero protruding bare peg collars**.

### 3. Limb & Extremity Mass Calibration (Match Reference Powerhouse Proportions — Never Skinny Stick Forelimbs!)
- Measure limb width and distal taper directly on the **post-zoom gridded front & side reference plates** (`forelimb_width / chest_width`, `wrist_width / elbow_width`, `paw_width / forearm_width`):
  - Heavyweight quadrupedal brawlers, apes, bears, and front-heavy wyverns often **WIDEN distally toward the wrist/gauntlet** (`upper_gauntlet < mid_gauntlet < lower_gauntlet`, flaring directly into a wide, heavy, outward-splayed paw!) — NEVER default all limbs to narrow downward-tapering cylinders!
  - Wrap heavy armored forearms on **all 4 faces** (stepped front shingles, outer vambrace, medial inner guard, and solid rear ulnar crest) so dead-front, outer-side, rear-3/4, and medial close-up audits all read as one unified, rock-solid powerhouse limb.
- **Mandatory 6-Angle Ultra-Close-Up Seam Audit:** Before finalizing any model, render dedicated close-up focus views (`Forelimb Outer`, `Forelimb Rear-3/4`, `Forelimb Front-3/4`, `Shoulder/Flank Chain Outer`, `Shoulder/Flank Front-3/4`, `Shoulder/Flank Top-3/4`, plus `Hindlimb Outer/Rear`) and visually confirm **ZERO floating spikes, ZERO detached collars, ZERO boxy sticker slabs, and 100% solid mechanical/anatomical continuity**.

---

## 1E. Multi-Track Body-Spike & Circumferential Limb-Ring Topology Mapping + Horn/Body-Spike Collision Policy (`v37`)

> **Critical Generalization Lesson (`v36 -> v37`):** When a creature, dinosaur, dragon, armored boss, or mech carries dense rows of spikes, osteoderms, or armor fins across its neck, shoulders, back, limbs, and tail, inspecting only `3/4` front renders causes a classic guessing error: builders lump shoulder spikes into "two oversized horns sticking sideways off the shoulders," drop `2–3` flat vertical bricks down an upper arm, and leave a gap across the mid-back/pelvis! Orthographic **top-down (`85–90°` dorsal)** + **pure lateral (`90°`)** + **interactive 3D turnaround** inspection reveals that master creature designs organize body spikes into **distinct continuous anatomical tracks and circumferential limb rings**.

### 1. Mandatory Top-Down + Lateral Multi-Track Mapping (Zero Guessing!)
Before placing a single dorsal, shoulder, flank, or limb spike, map **every continuous spike/plate system** on a post-zoom **top-down dorsal coordinate grid** paired with a **pure lateral grid**:
1. **Separate Distinct Parallel / Branching Body Tracks (Never Collapse Two Tracks Into One!):**
   - **Outer Front-Chest -> `155°` Horseshoe Shoulder-Surrounding Arch -> Dorso-Lateral Back -> Tail Track (`Track A`):** Check whether an outer armor/spike line originates **low in front of the chest right beside the chest bump** (`chest_vblade_l/r`, aiming forward-outward-upward), **rotates smoothly in a `~155°` horseshoe arch continuously around the shoulder socket perimeter** (`sh_arch_1 -> sh_arch_2 -> sh_arch_3 -> sh_arch_4`, fanning from anterior-pectoral over the lateral deltoid crown to posterior-scapular — NOT two giant isolated sideways horns!), and then flows **unbroken along both dorso-lateral sides of the back (`thorax -> waist -> pelvis -> tail_1 -> ... -> tail_tip`)** all the way to the tail! Choose `+Z` vs. `-Z` local spike axis per arch link so Euler rotations stay roll-free (`|rz| < 90°`), and keep every vertebral segment (`chest`, `waist`, `pelvis`, `tail_1..N`) carrying its mapped link so both top-down AND side silhouettes read as one flowing S-curve from front chest to tail tip!
   - **Inner Post-Horn / Post-Cranial Secondary Dorsal Row (`Track B`):** Check whether a **secondary inner spike/plate row** originates **immediately behind the main cranial horns or crest** (`occiput_crown_l/r`) and marches backward in parallel dorsal rows down the neck mane and anterior spine (`neck_3 -> neck_2 -> neck_1 -> withers`), clearly medial (`|x|` smaller) to the outer shoulder-to-tail arch!
2. **Circumferential Upper-Limb Osteoderm / Spike Rings (Never Linear Flat Bricks!):**
   - Zoom `100%` into the upper forelimb / thigh in `3/4` and lateral views. When small spikes wrap around a deltoid, bicep, or thigh cuff, **compute the limb shaft's true local long axis (`Y_local`)** and distribute `5–6` small pointed spikes (`~1.5 × 1.5 × 2.5–3.0u`, `45–50%` buried root) along a **smooth `140–170°` circumferential arc in `(X_local, Z_local)`** wrapping from anterior bicep across outer deltoid to posterior tricep — NEVER stack 3 thick rectangular bricks vertically along one flat side!
3. **Sleek 3-Step Tapered Limb Spurs (`base -> mid -> chiselled point`):**
   - On forearm ulnar crests, shin greaves, or wing elbows, avoid thick blunt rectangular club-bars (`3.5 × 3.0u` uniform cross-section). Sculpt sleek, razor-tapered `3`-step spurs (`2.5×2.0u` base -> `1.5×1.5u` mid -> `1.0×1.0u` horn tip) anchored independently into a parallel fused mounting keel (`ulnar_crest_base`) along solved outward-upward world trajectories.

### 2. Formal Horn & Body-Spike Chain Collision Policy (`sculpt_group` Track Tags + Cross-Segment `blend_with`)
Curved horns, convex shoulder-to-tail spike tracks, post-horn secondary rows, and circumferential limb rings cross **articulated spine, neck, shoulder, limb, and tail joints** that pitch, yaw, coil, and slam dynamically (`howl`, `bipedal_roar`, `paw_slam`, `tail_whip`, `gallop`). Applying naive rigid-box SAT between adjacent spikes in the same continuous track either triggers false collision failures OR forces ugly unnatural gaps between spikes!
- **Rule 1 — Shared Track `sculpt_group` Identifiers:** Assign a clean, track-specific `sculpt_group` token to every element belonging to the same continuous anatomical ornament chain:
  - `hb_sc` for multi-segment cranial horn beams + dorsal horn corrugation ridges + occipital post-horn transition plates;
  - `trk_a_sc` across the entire outer sub-neck -> shoulder arch -> flank -> waist -> pelvis -> `tail_1..N` lateral blade track;
  - `trk_b_sc` across the inner post-horn occipital -> cervical mane -> anterior dorsal shocker track;
  - `arm_ring_sc` across circumferential upper-limb spike rings and their host deltoid/bicep/tricep muscles;
  - `fsp_sc` across fused ulnar/tibial mounting crests and their 3-step tapered limb spurs.
- **Rule 2 — Explicit 1-Hop Cross-Bone Neighbor `blend_with`:** Declare explicit `blend_with` pairs between adjacent track links across neighboring articulated bones (`neck_1 <-> chest`, `chest <-> waist`, `waist <-> pelvis`, `pelvis <-> tail_1 <-> tail_2 <-> ... <-> tail_N`, and `arm_ring <-> shoulder_arch / anterior_shocker` for high overhead limb raises).
- **Rule 3 — Preserve Real Cross-Limb & Ground Collision Integrity:** Never put unrelated limbs (e.g., `forearm_spurs` vs. `hind_shin` or `paw` vs. `tail`) in the same `sculpt_group`! Keep full swept-SAT active between independent limbs, torso cores, head, and ground so real clipping during walk/gallop/slam animations is caught and fixed via true `SO(3)` trajectory math.

---

## 1F. Expert Blockbench Diagnostic Laws: Parent-Chain Pitch Compensation, Multi-Host Protrusion Audits, Zero-Drift Keels & Texture-Geometry Synergy (`v37`)

Real professional Blockbench artists don't rely on visual luck — when complex multi-track spikes, limb rings, or faceted heads look "slightly off," they diagnose the exact 3D cause systematically across modeling and texturing:

1. **Parent-Chain Cumulative World-Pitch Compensation (Anti-Buried Dorsal/Shoulder Spikes):**
   - Whenever a torso or neck chain pitches steeply upward (`waist -15° + chest -19° = -34°` world pitch), attaching a `-Z` dorsal or shoulder spike at a shallow local angle (`local_rx = +12°..+35°`) results in a **negative or near-zero world angle**, diving the spike straight DOWN into the chest or burying it completely inside thick dorsal manes!
   - **Exact Rule:** Sum parent spine X-pitches (`world_rx_parent`) before setting spike angles: set **`local_rx = desired_world_spear_angle - world_rx_parent`** (e.g., `+44°` world spear on `-34°` chest requires **`local_rx ≈ +78°`** or mounting through a spine-rooted dark collar pedestal rising through the mane!).
2. **Dual-Bound Multi-Host Rooting & Protrusion Audit (Zero Floating AND Zero Swallowed Parts):**
   - Auditing a limb-ring spike or shoulder spike against only `arm_hum_core` or `chest_core` can report "deeply rooted + protruding" while the spike is actually **100% swallowed inside `arm_deltoid`, `arm_bicep`, `arm_tricep`, or `chest_mane_core`**!
   - Always enforce **BOTH bounds against the full union of neighboring muscle & mane boxes (`H_all`)**:
     - **Zero-Floating Root Bound:** $\min_{h \in H_{\text{all}}} \text{obb\_sep}(\text{spike\_base},\ h) \le -0.25\text{ u}$ (`30–55%` solidly embedded);
     - **Proud-Clearance Tip Bound:** $\max_{p \in \text{corners}(\text{spike\_tip})} \min_{h \in H_{\text{all}}} \text{sd}(p,\ h) \ge +1.50\text{ u}$ (tip protrudes crisply outside EVERY muscle and mane volume!).
3. **Zero-Dummy-Brick Direct Bone-Pivot Placement (`ZERO Leftover Base Bricks!`) & Centerline `mirror_x` Attachment:**
   - **Never chain Spike 2 off Spike 1 inside a tilted bone** — stepping along a tilted local `-Z` axis drifts Spike 2 out into mid-air (`sep > 0`).
   - **AND never leave fake dark structural "base/keel" cubes (`sh_arch_base`, `waist_trk_base`, `pelvis_trk_base`, `tail_k_trk_keel`, `arm_ring_anchor`, `occiput_collar`, `shocker_ch1_collar`) inside the `.bbmodel` just to position multi-spike tracks!** A human artist opening the model in Blockbench immediately spots unnecessary mounting bricks protruding behind shoulders or cluttering the Outliner!
   - **Exact Master Solution:** Attach the `rest_euler: [0, 0, 0]` track bone to a **centerline (`X = 0`) `parent_box`** (e.g., `chest_core`, `waist_core`, `pelvis_core`, `tail_k_core` — mandatory for `G1` `mirror_parity = OK`!) with `align=[0, 0, 0]`, and place every spike independently directly relative to the bone pivot via **`place(anchor=[0, 0, 0], align=[...], offset=[dx, dy, dz])` (omitting `base_box`)**! Every spike roots straight into real host anatomy (`sep <= -0.25u` against `chest_scapula`/`waist_core`/`tail_k_core`) with **ZERO dummy cubes** in the viewport or Blockbench Outliner!
4. **Absolute Bone-Frame `rotate` Law, Multi-Segment Tip Co-Linearity Audit (`tip_bend <= 1.0°`) & Exact `SO(3)` Radial Arm-Spike Lift (`+32°–35°` Straight Outward!):**
   - **CRITICAL BLOCKDSL TRAP:** In BlockDSL (`se3_chain.py`), `place(base_box=A)` anchors a child box's *position* onto `A`'s rotated face, **NOT its orientation** — `B.rotate` is ALWAYS evaluated directly in the **parent bone's frame**, never summed onto `A.rotate`! Writing small delta angles like `rotate: [4, 0, 0]` on a `*_tip` or `*_point` box placed on an angled spike (`[74, 38, -20]` or `[-34, -66, 0]`) snaps the tip straight back to the bone's neutral axis, creating a **`45°–150°` folded/bent spike**!
   - **Exact Rule:** For every straight multi-box spike, spur, blade, or tail barb inside one bone, **`tip.rotate` and `point.rotate` MUST REPEAT `base.rotate` verbatim** (enforced via an automated `tip_bend <= 1.0°` co-linearity assertion!).
   - **Exact `SO(3)` Radial Lift & Zero-`align` Outward-Offset Law for Circumferential Limb Rings & Heavily Rotated Forearm Spurs:**
     - Never guess raw Euler angles for radial spikes inside a pitched/rolled limb bone (`Rx(-24°)` after `Ry(-84°)` gives only `2.6°` of lift, laying the spike flat!). Solve the exact bone-local `SO(3)` rotation (`bone.rotation.T @ desired_world_dir`, choosing `+Z` vs. `-Z` local spike axis so `|rz| < 90°` with zero `180°` roll flips) so every upper-arm ring spike AND large forearm spur lifts cleanly **`+32°–35°` straight outward** (`0.0°` bend!).
     - **CRITICAL PRE-ROTATION `align` TRAP:** In BlockDSL (`se3_chain.py`), `align` shifts `unrot_local_center = ref - 0.5 * size * align` **BEFORE** applying `rotate`! On heavily rotated forearm/shin spurs (`Ry ≈ -64°..-76°`), a non-zero `align` shifts the spike base **inward into the forearm core** (`sep = -5.0u`, leaving only `1–2u` stubby nubs poking out!). Always place heavily rotated limb spurs with **`align=[0, 0, 0]` + an explicit world-solved outward `offset = R_bone.T @ (root_w + 0.36 * L * dir_w - piv_w)`**, and assert **multi-host `tip_clear >= +3.0u` outside ALL forearm boxes**!
   - **Parallel Twin Dorsal Row Separation Gate (`Track B` `inner_gap >= 2.5u`):** Never give left-side (`-X`) dorsal spikes negative inward `Ry` yaw (which causes rear spikes to converge and overlap on the centerline!). Keep outward/parallel `Ry >= +2°..+6°` and assert `inner_gap = cr[:,0].min() - cl[:,0].max() >= 2.5u` on every twin dorsal pair (`occiput..withers`)!
5. **Texture-Geometry Structural Re-Balancing & `+Z`/`-Z` Spike Shader Parity (Expert Blockbench Craft Synergy):**
   - Great Blockbench models unify modeling and texturing rather than treating paint as an afterthought:
     - **Darkened Keratin Root Collars, Longitudinal Bone-Flutes & `+Z` vs. `-Z` Axis Parity (`grows_pos_z`):** Shade the basal `30–35%` of every horn, dorsal spear, shoulder blade, and limb-ring spike `1.5–2` steps darker with longitudinal fibrous bone flutes — and keep any recessed basal emissive/bioluminescent crevice glint strictly tucked into the rooted basal `35%` of non-tip base cubes! **CRITICAL:** Any spike oriented along local **`+Z`** (e.g., forward `arm_ring_1..3`, `forearm_spur_*`, `chest_vblade*`, `sh_arch_1*`) MUST be routed into `grows_pos_z` inside the painter so dark roots and basal glow never flip onto exposed ivory-gold distal tips!
     - **Razor Occlusal Shadows & Specular Ridge Glints:** Pair geometric sawtooth jaw waves (`lip_serr` / `jaw_serr`) with a `1 px` dark contact line and `1 px` leading bevel highlight (plus a subtle **`+6°–8°` rest jaw pitch** on fierce predators so the interlocking triangle bite seam reads unmistakably even at rest!).
     - **Dedicated Plane-Shader Routing:** Always dispatch zero-thickness hard-alpha planes explicitly by anatomical role (`cheek_web` -> striated maroon sinew; `shockfur` -> serrated snow-white tufts with `box.center[0] < 0` bilateral symmetry; `membrane` -> 3-sided organic AO leather) so buccal webs or sails never inherit fur textures!

---

## 1G. Expert Human-Artist Carefulness Discipline: Stylized Abstraction References, Living Trait-Placement Ledger & Pre-Signoff Region-by-Region Audit (`v37`)

> **Why Experienced Blockbench Artists Outperform Fast "First-Draft" Builds:** Real expert Blockbench artists do not finish a pass, glance at one hero thumbnail, and call it done. They maintain **explicit working notes ("where must each trait sit? what surface curvature does it ride? what emotional expression does this eye convey? did every segment of the tail, limbs, back, and jaw actually get checked?")**, cross-check stylized official abstractions alongside realistic renders, and inspect **every single anatomical region** before signing off!

### 1. Proactively Consult Official Stylized Game Abstractions Alongside Realistic Renders
- When modeling a complex high-poly creature or boss (e.g., mainline *Monster Hunter World/Rise/Wilds* renders covered in dense fur/scales), **proactively search for official stylized / cel-shaded / RPG spin-off 3D renders** (such as *Monster Hunter Stories / Stories 2*, official figure turnarounds, or clean stylized artbook orthographics)!
- **Why this accelerates Blockbench spatial reasoning:** Stylized official models have *already* performed professional artistic abstraction — simplifying noisy high-poly bumps into unmistakable color zones, clean eye-slant temperament geometry, clear spherical/cylindrical spike placement (e.g., showing whether an upper-arm spike ring rides high across the **spherical upper deltoid dome** vs. near the elbow!), and readable dorsal/lateral tail-spike crowns!

### 2. Mandatory Living "Trait-Placement & Temperament Ledger" (Maintain Inside `PROJECT_GOAL.md`!)
Before placing Cube `#1` — and updated after EVERY single white-model (`W1–W3`), texture (`T1–T2`), and animation (`A1–A2`) pass — keep an explicit human-artist checklist ledger in `PROJECT_GOAL.md` tracking **every focal zone, its exact 3D surface placement, its emotional/temperament target, and its verified audit proof**:

| Body Region | Artist Verification Questions (Must Answer & Check Visually on EVERY Pass!) | Quantitative / Visual Gate |
|---|---|---|
| **1. Head & Expression** | Does the eye/brow angle match the species' **exact temperament** (*Gentle/Approachable* open round eye vs. *Wild/Furious `M4`* steeply-slanted `\ /` brow `Rx=+24..+30°, Rz=∓26..∓34°`, narrow angular slit, dark upper-lid scowl shadow)? Are material boundaries pure (ZERO unreferenced snout-ridge accent blocks)? Is the anterior rhinarium bump present? | Close-up Side (`-90°`), Dead-Front (`0°`), & `3/4` Head Macro inspected |
| **2. Jaw, Bite & Mouth Interior** | Do upper & lower jaw rims carry matching **interlocking half-phase sawtooth waves** if referenced? Are chin prongs curved naturally (`CONCAVE_UP` scoop, not stiff straight rods)? Are all `20–28` V-arch teeth 100% hidden at `0°` rest, spaced on converging alveolar rails, 2-tone shaded, framed by crimson gums, grooved tongue, and taut buccal webs at `+36°` open? | `exposed_teeth == 0`; `basin_zmax <= chin_zmax - 0.8u`; `+36°` Side + Top-Front + Low-3/4 inspected |
| **3. Neck, Shoulders & Back Tracks** | Are distinct parallel tracks separated cleanly (e.g., **Track A** sub-neck -> proud convex shoulder arch -> flank -> waist -> pelvis vs. **Track B** inner post-horn row)? Did we compensate cumulative spine pitch (`world_rx_parent`) so ZERO spikes dive into the chest or bury inside manes? Are there **ZERO dummy base/keel cubes** (`sh_arch_base`, `waist_trk_base`, `tail_trk_keel`) sticking out or cluttering the Blockbench Outliner? | Top-Down Ortho (`86°`) + Upper-3/4 Shoulder (`-54°, 34°`) inspected; `0` dummy bricks; every base `sep <= -0.25u`, every tip `clear >= +1.5u` |
| **3b. Torso Chest Bump, Under-Neck Throat & Track Glow** | Does the torso break away from a flat square box via a **protruding forward-downward brisket chest bump** (`chest_prow_bump` + pointed V-shield `chest_fur_prow -> tip -> apex`) sweeping smoothly up into a tucked athletic belly? Are the **overlapping segmented under-neck throat plates** (`neck_3 -> neck_2 -> neck_1_throat`) clearly exposed between the chin and chest bump, framed laterally by stepped side-neck fur capes? Do stylized `32x` glowing energy veins sweep organically along the side shoulder spikes without `+` grid crosshairs or noisy high-res clutter? | Dedicated Front-Low (`0°, -4°`), Front-3/4 Low (`-36°, -2°`), & Pure Side Torso (`-90°, 2°`) inspected; `glow_area_pct ∈ [1.5%, 3.5%]` (`<= 4.0%`) |
| **4. Limbs, Rings, Spurs & Digits** | Are upper-limb spike rings riding the **true anatomical surface curvature** (e.g., high across the **spherical deltoid dome**, NOT drooping down on the elbow)? Are forearm/shin spurs sleek `3`-step tapers on parallel zero-drift keels? Do distal gauntlets/paws match reference mass and splay wide flat on `y = 0`? | Outer + Rear-3/4 Limb Zoom inspected; `arm_ring` multi-host `sep <= -0.25u` & `clear >= +1.5u`; `FL/FR/HL/HR = 0.0` |
| **5. Tail & Rear Assembly** | **NEVER forget to audit the tail spikes AND their surface orientation!** Check top-down + rear-3/4 references to confirm whether the tail's dorsal surface is a wide fur plume flanked by **horizontally-spreading lateral sawtooth blades (`Rx ≈ +6..+10°` along the tail plane, `Ry = ±36..±48°`, `|dx_local| >= 1.8 * |dy_local|` — NEVER vertical stegosaurus sticks!)** + golden ventral shingles, vs. a vertical dorsal fin crest! | Dedicated Rear-3/4 + Side + Top-Down Tail Zoom inspected; ALL `tail_1..N` blades verified `clear >= +2.2u` & orientation-matched |
| **6. Texture & Silhouette Balance** | Does `32x`/`64x` cross-cube carving + darkened root collars + `1 px` dark occlusal lines re-balance geometric transitions without noise (`glow <= 4%`, `25%` squint gate instant read, `mirror_parity = OK`, `G1–G14` `0` warnings)? | `T1` unlit + `T2` lit + `25%` squint + `construct_check` `rc=0` + `verify_bbmodel_ground` `PASS=True` |

### 3. The "Slow Down & Walk Around Every Region" Pre-Signoff Law
Before presenting ANY milestone (`.bbmodel`, turnaround, or animation) to the user or shipping it:
1. **Walk systematically through all 7 rows of your Trait-Placement Ledger** while looking directly at high-res cropped renders of **each** region (`Head/Eye`, `Open/Closed Jaw`, `Front-Low Chest Bump & Exposed Bronze Throat`, `Top-Down & Upper-3/4 Shoulder Tracks`, `Upper-Arm Spherical Dome & Forearm`, `Hindlimbs & Feet`, **`Tail Spikes & Tip`**).
2. **Never omit a tail, underside chest/throat, hindlimb, or rear-angle from your programmatic protrusion/rooting audit** — any region left out of the automated audit AND close-up visual walkaround is the exact region where flat tail spikes, boxy square chests, buried throat plates, leftover dummy base cubes, drifting arm rings, or calm eyes slip through!

---

## 1H. Anti-Square-Block Torso Sculpting, Exposed Under-Neck Throat Carving & Track-Following `32x` Glow Patterns (`v37`)

Even well-proportioned quadrupeds and bosses instantly betray a "boxy beginner" look if the thorax is left as one flat rectangular box and the neck underside is either flat or buried in undifferentiated fur. Expert Blockbench artists sculpt the front torso and neck as three interacting organic surfaces:

1. **Break the Square-Block Torso with a True Armored `\  /` V-Shaped Chest Bump (`neck -> sternum_keel -> waist_sternum_bridge`) + Verified Normal-Vector Signs & Sleek Lower Fur Keel:**
   - **Material Truth First (Never Bury an Armored Chest Under a Giant Fur Pillow!):** Verify directly from BOTH official renders and 3D Blockbench references which surfaces form the **armored chest V-plate** vs. where **chest fur** actually sits! On armored lupine/draconic quadrupeds (e.g., Zinogre), the broad upper-to-mid chest descending from `neck_1_throat` is an **armored V-shaped chest shield** (two large angled side-chest plates flanking a continuous central ribbed bronze sternum plate flowing unbroken into `sternum_keel -> waist_sternum_bridge`), while the **white chest fur** sits cleanly as a **sleek, narrow V-pointed lower keel/plume** along the lower brisket apex & ventral `sternum_keel -> waist_sternum_bridge` ridge!
   - **CRITICAL RIGHT-HANDED `V`-DIRECTION & CONVEX-PROW SIGN LAW (Zero Upside-Down `/ \` Skirts & Zero Backward Slopes!):** Never guess `Rx`/`Ry`/`Rz` signs inside a pitched chest bone (`world_rx = -34°`)! Solve `R_local = R_parent.T @ R_world_target` directly in `SO(3)` and verify EVERY world-axis sign on the **`-X` (left) plate**:
     1. **True `\  /` Front V-Shape (Wide under shoulders -> converging inward toward `sternum_keel`):** Requires **`ay.x < 0`** (`world_rz > 0`, e.g. `+20°..+24°` on `-X`), verified via **`top_X_mean < bot_X_mean - 1.5u`**! (A negative `Rz` on `-X` flips the chest into an upside-down `/ \` A-tent!)
     2. **True `/\` Top-Down Convex Prow (Proud at centerline sternum -> sweeping back toward shoulders):** Requires front normal **`az.x < -0.20` AND `az.z > 0.70`** (`world_ry < 0`, e.g. `-24°..-26°` on `-X`)! (A positive `Ry` on `-X` turns the chest into a concave `\_/` bowl!)
     3. **True Continuous Neck -> Deep Brisket -> Tucked Waist Side Curve (`sternum_keel` + `waist_sternum_bridge`):** Build the central armored sternum as a smooth convex arc:
        - `chest_sternum_upper` (`\` sloping forward-down from `neck_1_throat`, `world_rx ≈ -22°` so `ay.z < 0`, projecting `>= +1.5u` ahead of `neck_1_throat` & `>= +5.5u` ahead of `chest_core`);
        - `chest_sternum_lower` (`/` curving down-and-back into `sternum_keel`, `world_rx ≈ +30°..+32°` so `ay.z > 0`, narrowing in a cascading V-taper `10.5u -> 7.5u -> 5.5u`);
        - Deep `sternum_keel` catching the lower sternum flush (`Y_min < waist_belly Y_min - 0.5u` — deeper ribcage brisket than the tucked athletic waist!);
        - Upward-sweeping `waist_sternum_bridge` (`world_rx > 0`, e.g. `+10°..+12°` on a Z-long plate so `az.y < 0` -> deeper at front `sternum_keel`, climbing smoothly UP-and-BACK into `waist_belly`!);
        - Sleek narrow V-pointed chest fur (`chest_fur_prow` `4.5u` -> `chest_fur_keel` `3.0u` + serrated `chest_fur_fringe_l/r`) hugging ONLY the lower brisket apex and trailing cleanly back along `sternum_keel -> waist_sternum_bridge`.
   - **Mandatory Programmatic V-Chest Audit:** Assert ALL four invariants in your build script: `cb_zmax >= neck_throat_zmax + 1.5u`, `v_top_x < v_bot_x - 1.5u` (true `\ /` V-shape), `v_az[0] < -0.20` (true convex `/\` prow), AND `sternum_keel_ymin < waist_belly_ymin - 0.5u` (deep brisket tuck)!
2. **Expose Segmented Under-Neck Throat Plates Framed by Stepped Side-Neck Fur:**
   - Check front-low (`0°, -4°`) and front-3/4 (`-36°, -2°`) references: many iconic creatures (Zinogre, elder dragons, drakes) display a **curved column of overlapping horizontal armored throat plates (`neck_3_throat -> neck_2_throat -> neck_1_throat`)** descending cleanly from under the jaw to the top of the chest bump!
   - Never hide those throat plates beneath a giant chin-beard cube — keep the **throat center exposed**, sculpt crisp dark horizontal chevron overlap seams between plates, and carve the snow-white neck mane as **stepped upper-lateral fur capes (`neck_2_cape_l/r`, `neck_1_cape_l/r`)** flanking the neck sides cleanly down toward the shoulder track!
3. **Track-Following Stylized `32x` Emissive / Bioluminescent Patterns (Zero `+` Grid Crosshairs & Zero High-Res Noise):**
   - When translating high-res creature markings (such as Sketchfab/capcom bioluminescent lightning veins or glowing energy cracks) into `32x` Blockbench pixel art, **never spam noisy fractals or perpendicular `+` grid lines**!
   - Instead, route **swept diagonal jagged `2 px` lightning/energy ribbons (`~1.5–3.5%` total emissive area, strictly `<= 4.0%`) directly parallel to the anatomical armor track** — hugging the shoulder-spike skirt (`chest_scapula`, upper `arm_deltoid`), curving forward onto `neck_1_trapezius`/`neck_1/2_flank`, trailing down `chest_rib_slope`/`waist_flank`, and tucking crisp emissive glints into recessed spike root crevices & lateral throat-plate notches!

---

## 1I. Winged Elder-Dragon Volumetric Dental Arcades, Material-Truth Audits & Anti-Plaid `32x` Carving (`v37`)

Validated directly across full fresh `W1 -> W2 -> W3 -> T1 -> T2` builds of **Malzeno `v37`** (`347` el., `0` collisions) and **White Fatalis `v37`** (`398` el., `0` collisions):

1. **Zero-Dummy-Brick Dental Arcade Child Bones (`upper_dental_l`, `lower_dental_l`) & `busiest bone <= 8`:**
   - Never stuff `upper_gum_rail` + `6` volumetric teeth + `maxilla_rail` + `3` lip serrations onto a single `maxilla_l` bone (`11` boxes -> triggers `G1` Bone Bloat!). Instead, place `upper_gum_rail` as the primary real-anatomy box of child bone `upper_dental_l` (`attach(parent_box=maxilla_rail)`) carrying its `6` spaced `0.5u` 2-tone volumetric fangs (`7` boxes total, `0` dummy bricks!), and mirror the exact pattern for `lower_dental_l` inside `mandible_side_l`.
   - Always pair the volumetric dental arcade with **taut striated buccal cheek-tendon webs (`jaw_cheek_web`, `normal: x`)** on `cheek_l/r` so the `+36°..+40°` roar jaw corner seals organically rather than exposing an empty air gap.
   - Verify BOTH closed rest-snarl (`+3°..+4°` jaw pitch -> `0` teeth poking outside closed lateral lip) AND wide open roar (`+36°..+40°` -> all `24` 2-tone ochre-root->ivory fangs, crimson gums, grooved tongue & buccal webs visible).
2. **Stage A–D Native-Cutout Material-Truth Guard (Before Painting a Single Pixel!):**
   - Never assume legacy palette shortcuts match official renders — verify primary 60/30/10 material splits directly from native cutouts (`visual_primitives_prep.py zoom`):
     - *Malzeno:* 60% **luminous moonlit pearl-silver** across skull, neck, V-chest, torso, limbs, wing spars & tail (NEVER charcoal-black body!), 30% velvet crimson strictly on the tapered feathered V-jabot, bat-ear webs, serrated forearm/ankle cuff plumes (NEVER thick red brick slabs!), and veined bat wings, 10% antique bone-gold crown/spurs/talons/trident.
     - *White Fatalis:* 60% **luminous pearl-ivory & warm alabaster-champagne** overlapping chest/belly armor (NEVER a loud pink-red paint stripe smeared down the white chest!), 30% silky snow-white mane/crest/cheek-wings/backward-sweeping beard + weathered silver-parchment scorched-hem wings, 10% dark charcoal-bronze lower stockings, obsidian talons & dark keratin antlers with **surgical crimson fissures restricted to recessed orbital & antler-root crevices** (`glow <= 3.5%`).
3. **Right-Handed `-Z` Rear Spur Yaw Sign & Stride Clearance (`elbow_spur_l`):**
   - For a rearward-pointing (`-Z`) elbow spur mounted on the left (`-X`) arm, `Ry(+theta) @ [0, 0, -1] = [-sin(theta), 0, -cos(theta)]` flares cleanly **OUTWARD (`-X`)** away from the ribcage and hind knee! Using `Ry(-theta)` rotates `-Z` **INWARD (`+X`)** directly into `chest_core` and `waist_core`!
4. **Anti-Plaid / Anti-Checkerboard `32x` Feather & Keratin Shader Discipline:**
   - Never multiply or add perpendicular high-frequency `sin(x)` + `(y % 1)` bands on front-facing plumage (`jabot`) or curved horns/antlers (`ha_1..3`), which produces digital plaid/checkerboard artifacts under `32x` rendering!
   - Carve plumage with **continuous V-chevron tier overlaps (`(v_down * 2.6 - |x| * 0.72) % 1.0`)**, and carve horns/antlers with **low-frequency longitudinal growth ridges (`sin(prog * 7*pi) * 0.05`) + darkened keratin cuticle root collars + smooth apex weathering**, keeping `interior_mean_lum_step` cleanly inside **`5.0–7.5`**.

---

## 2. Texture Resolution Strategy (`32 px/block` + `2x` Focal `uv_scale`)

Traditional `16 px/block` (`1 px/unit`) leaves tiny focal structures like eyes with only `2×2` or `3×2` pixels — physically impossible for realistic iris, sclera, limbal ring, eyelid shadow, and specular catchlight craft — and limits carved plates/tails to coarse single-pixel stripes. Top human Blockbench artists routinely combine **`32 px/block` (`2 px/unit`) base atlases** with **multi-density `2x` focal UVs** (demonstrated directly in `ArtDanPol 3D`, `TrongOBD`, `Wackyblocks`, `Yaju Senpai`, and `TheAdventurousMiner` timelapses).

### Resolution Tier Matrix (General Across All Genres)

| Asset Profile | Global `texel_density` | Focal Eye / Lens `uv_scale` | Effective Focal Density | Typical Atlas Size | Best For |
|---|---:|---:|---:|---|---|
| **Vanilla-Lite Mob / Small Prop** | `16` (`1 px/u`) | `2.0` on eyes/dials | `32 px/block` (`2 px/u`) | `256–512` | Vanilla-matching mobs, compact critters, handheld tools |
| **Studio Master Creature / Hero / Mech (Recommended `v33`)** | `32` (`2 px/u`) | `2.0` on eyes/lenses/emblems | **`64 px/block` (`4 px/u`)** | `1024–2048` | High-craft creatures, bosses, animals, heroes, armored mechs, detailed vehicles |
| **Showcase Portrait / Ultra-Detail Bust** | `32` (`2 px/u`) | `2.0–4.0` on eyes/face | `64–128 px/block` | `2048` | Close-up hero faces, expressive NPCs, flagship showcase pieces |

### SyncCraft Block-DSL & Native Blockbench Implementation
- **Block-DSL (`v33`):** Set `texel_density: 32` in `@meta` AND add `uv_scale: 2` on focal eye, optic-lens, visor, or emblem boxes:
  ```blockdsl
  @meta { archetype: entity, target_profile: bedrock_entity, texel_density: 32, size_step: 0.5, audit_profile: sculpted }
  ...
  box eye_l place(base_box=cranium_orbital, anchor=[-0.98, 0.24, -0.06], align=[0.45, 0, -0.25]) {
    size: [1.5, 2.0, 3.0], material: eye_mat, rotate: [6, 16, -8], sculpt_group: "head_sc", uv_scale: 2
  }
  ```
  Because Blockbench 5, Bedrock, GeckoLib, and glTF all consume **Per-Face UV pixel rectangles** (`[u1, v1, u2, v2]`), a `1.5 × 2.0 × 3.0` eye cube seamlessly receives a crisp **`12 × 8 px`** outer eye island (`6 × 8 px` front bevel) inside the exact same single PNG atlas — zero engine hacks!
- **Anti-Mud Law at `32x`:** Doubling resolution MUST NEVER turn into blurry airbrushed gradients or salt-and-pepper static noise! Preserve crisp Blockbench pixel-art discipline:
  - Paint in **organized `2 px` stepped pixel clusters** for organic hide/scales/fur, reserving crisp `1 px` strokes for razor edge-bevels, plate-overlap crevices, rivets, pupil slits, and wet specular catchlights.
  - Use **hue-shifted `5–7` step palette ramps** (`30–50°` cool shadow shift -> warm lit bevels) with `>= 2 px` UV island padding and RGB edge dilation.

---

## 3. Sculpted 4-Part Periorbital Carving & Realistic Focal Eyes / Optics

Flat 2-pixel dots or unframed glowing cubes immediately betray an AI model. Real human Blockbench masters (`ArtDanPol 3D`, `Wackyblocks`, `Yaju Senpai`, `TheAdventurousMiner`) build **every eye or optical sensor as a 3D recessed architectural assembly**:

### A. Modeling Scale — The 4-Part Periorbital Cradle (Zero Flat Stickers!)
1. **Recessed Orbital Cavity Step (`cranium_orbital` / recessed housing):** Step the skull/helmet inward by `0.75–1.5u` at the eye bay so the eye sits inside real 3D physical depth (`0.00u` outward extrusion in `0°` Dead-Front!).
2. **Sculpted Upper Eyelid Hood (`eyelid_upper`) under Overhanging Brow (`brow_ridge`):** Place a dedicated angled upper eyelid bar (`size ~ [1.5..2.0, 1.0, 3.0..3.5]`) hugging the top `20–25%` of the eye beneath the brow arch. This creates a natural almond/hooded contour and casts real physical shadow across the upper eye!
3. **Sculpted Lower Eyelid Cradle (`eyelid_lower`) & Lacrimal Corner:** Seat a slim beveled lower lid rim directly between the bottom of the eye and the cheekbone/zygoma (`zygoma_l/r`), framing the lower ocular arc and anchoring the inner tear-duct corner (`inner canthus`).
4. **Post-Orbital Rim / Rear Corner Bezel (`orbital_post_bar`):** Close the rear lateral eye corner with a subtle angled rim bar bridging the outer brow down into the rear cheekbone/temple — completing a 4-sided sculpted bezel around the recessed globe (matching `Wackyblocks`'s 4-sided diamond orbital bezel and `ArtDanPol 3D`'s wrapped lid rig!).

### B. Texture Scale — The 7-Layer High-Res Eye Painting Recipe (`10×6` to `16×10 px` via `uv_scale: 2`)
Paint the outer focal face strictly with all 7 depth layers (keeping all hidden interior faces pitch-dark socket AO so zero color leaks onto the cheek!):
1. **Outer Socket Rim & Lash Crease (`1 px` dark obsidian border):** Frames the almond/round aperture in deep recessed shadow.
2. **Upper-Eyelid Cast Shadow (Top `25–30%` darkened `40–55%`):** Darkens the top rows of *both* the sclera AND the iris — the single strongest realism cue separating a real recessed eye from a glowing sticker!
3. **Shaded Curved Sclera & Warm Canthus Corners:** Off-white / muted parchment / tinted sclera darkened toward the inner/outer corners (with subtle warm terracotta/rose shadow at the anterior lacrimal corner on biological subjects).
4. **Dark Contrasting Limbal Ring (`1 px` ring around the iris):** Crisp dark ring separating the iris cleanly from the sclera so the eye reads sharply even at gameplay distance.
5. **Multi-Zone Radial Iris + Lower-Crescent Internal Refraction Glow (`ArtDanPol 3D` Law):**
   - Outer iris margin: deeper saturated tone with subtle radial fiber variation.
   - Mid-iris: signature primary hue.
   - **Lower iris crescent (below the pupil):** brighter luminous bounce/refraction arc — mimics light passing through a real curved cornea and landing on the lower iris bowl!
6. **Anatomically Authentic Pupil:** Match the reference species/subject (never default everything to one shape!):
   - *Dragons / felines / nocturnal reptiles:* crisp vertical elliptical slit (`1–2 px` wide × `50–65%` height).
   - *Herbivores / ungulates / amphibians:* horizontal rectangular bar.
   - *Humans / canines / birds / gentle creatures:* round dark pupil with rich iris collarette ring.
   - *Mechs / robots / vehicles:* recessed multi-ring concentric dark-glass lens housing with colored inner reticle ring & angled glass sheen.
7. **Offset Wet Cornea Specular Catchlight + Subtle Rim Bounce:** Place a sharp `1×2` or `2×1` pure white-cream highlight offset on the upper-front iris/limbus edge, balanced by a tiny dim bounce glint on the opposite lower-rear iris. Keep emissive glow (if any) strictly selective to the mid-iris/lower refraction + catchlight so the dark pupil, limbus, and lid shadow stay razor sharp!

---

## 4. Anti-Toy Carving on Tails, Laminated Plates & Mounted Accessories

Why do tails, dorsal fins/plates, armor trim, and mounted accessories frequently look like **"Lego toys"** even when a model's torso and limbs look great?
- **Toy Failure #1 (Extruded Uniform Bricks):** Stacking plain rectangular boxes (`[w, h, l]`) end-to-end along a tail or tentacle with 90° flat edges and identical cross-sections.
- **Toy Failure #2 (Single Flat Plank Plates):** Gluing one long flat rectangular strip on top of each segment instead of staggered overlapping armor shingles.
- **Toy Failure #3 (Blunt Chopped-Off Tips):** Ending tail blades, fins, horns, feathers, armor pauldrons, or exhaust spurs in wide square blocks instead of 2–3 step tapered beveled points.
- **Toy Failure #4 (Floating Un-Anchored Accessories):** Sticking shields, spikes, cables, pouches, straps, turrets, or dorsal plates directly onto a flat wall with zero mounting socket, collar, bracket, or crease.

### Modeling-Scale Anti-Toy Rules (Universal Across Creatures, Animals, Humanoids & Mechs)
1. **Multi-Plane Faceted Cross-Sections (Zero Plain Rectangles):** Build every tapered appendage (`tail`, `tentacle`, `cape`, `mech boom`, `fuselage`) from a **stepped core + chamfered lateral flank bevels + distinct dorsal & ventral trim ridges** — or carve a **recessed flank trough** (e.g., fur/flesh/wiring channel framed between proud upper and lower armor rails!).
2. **Staggered Overlapping Shingle Rows (`>= 2` Progressive Pitched Plates per Segment):** Break long single-board dorsal/ventral plates into **2+ overlapping sawtooth shingles per segment** (`pitch +14° -> +20°`) so the silhouette reads as articulated natural scales, feathers, or riveted lamellar armor plates, NEVER flat plastic rulers!
3. **2-to-3 Step Tapered Distal Blades & Tips:** Every protruding blade, tail fan, dorsal crest, ear pinna, fin, wing spur, armor vane, or claw MUST taper progressively in **both width AND thickness** across `2–3` stepped boxes (`base -> mid bevel -> narrow sharp tip`), fanning radially with subtle pitch/yaw progression!
4. **Physical Mounting Collars, Brackets & Straps on Every Accessory (`Wackyblocks` Law):** Whenever attaching an accessory (dorsal electro-plates, shoulder horns, armor pauldrons, saddlebags, scabbards, cables, lanterns, thruster pods, or weapons), seat it inside a **contrasting recessed collar ring, cuff, strap, or mounting bracket** (`0.5–1.0u` proud dark-slate/leather/iron bezel) so it reads as physically forged or strapped onto the body!

### Texture-Scale Anti-Toy Rules (`32 px/block` Relief & Micro-Bevel Craft)
1. **Cross-Seam Continuous Dorsal/Flank Groove (`Kundzio` Technique):** Paint a dark `2–3 px` central recessed shadow trough running continuously across segment boundaries, framed by crisp `1 px` bright outer corner highlights so multi-bone tails read as one sculpted continuous organism.
2. **Plate-Overlap Shadow Valleys & `1–2 px` Leading Bevels (`Wackyblocks` Technique):** On every shingled plate, scale, or armor panel, paint:
   - A **deep `2 px` dark occlusion groove** along the seated root/underlap edge.
   - A **crisp `1–2 px` sunlit pale bevel highlight** along the exposed upper/leading rim.
   - **Dark underside drop-shadow** along the lower edge.
3. **Directional Surface Grain & Imbrication (Never Flat Solid Color Fills!):**
   - *Organic scales/hide:* staggered U-scallop or hexagonal rows (`4–6 px` per scale at `32x`) with `1 px` dark lower crescent + lit upper crown.
   - *Horn / bone / chitin plates:* stepped transverse chevron/corrugation rungs + subtle longitudinal growth striations + sparse weathered mineral pits.
   - *Armor / metal / wood / leather accessories:* dark `1 px` recessed panel seams, `1 px` rivet/stitch dots with lower pale glint, subtle worn bevel chips, and gravity-aligned grime/rust/root AO.

---

## 4B. `v38` Universal Multi-Skeleton Craft Laws (6-Archetype Empirical Distillation)

1. **Non-Fandom Direct-Download High-Res Extremity Reference Law:** Never settle for Fandom lightbox crops or hotlink-compressed thumbnails when claws, paws, teeth, wing patterns, or wrist spurs are obscured! Download full-resolution non-Fandom sources directly to disk (**4K ZBrush sculpts, ArtStation turnarounds, Capcom Figure Builder photos, official artbook orthographics, `monsterhunterwiki.org`**) and run `visual_primitives_prep.py zoom --autocrop --with-grid --with-edges` on the downloaded images before Pass `W2`.
2. **Periorbital Sightline Window Law (Anti-Buried-Eye Guard — *Lagiacrus*):** When carving a 4-part recessed eye socket (especially on narrow/low-browed crocodilian or predatory skulls), ALWAYS verify an unobstructed sightline window across **3 mandatory views (`Dead-Front 0°`, `3/4 Front 35°`, and `Side 85°`)**:
   - `brow_lower_edge_y >= eye_center_y + 0.25 * eye_h` (overhangs ONLY the top ~20–25% for a fierce `\ /` scowl — never drops across the iris/pupil!).
   - `cheek_upper_edge_y <= eye_min_y + 0.15 * eye_h` (cradles the bottom edge cleanly without rising over the eye!).
   - Angle `eye_l/r` onto the forward-tapered orbital shelf so BOTH glowing irises read unmistakably even at Dead-Front (`0°`).
3. **Zero-Blink `0.5u` Dual-Sided Membrane Slab & Continuous Tail-Fan Law (Blockbench/Minecraft OpenGL — *Legiana*):**
   - **Never** paint two different textures onto the `up` and `down` faces of a `0.0u`-thick `plane` (or stack two `0.0u` planes at the same Y height) — in Blockbench 5 and Minecraft OpenGL they share identical depth and **blink/Z-fight continuously**!
   - For **dual-sided differently-colored membranes** (dorsal cobalt star wings vs. ventral cream wings, bifurcated crests, broad continuous butterfly-diamond tail fans), use **razor-thin `0.5u` solid slabs** (`up` = dorsal pattern, `down` = ventral pattern, thin `0.5u` rims alpha `0`), staggered by `±0.5u` in normal offset wherever adjacent bays overlap (`construct_check.py` `G10`/`G11` natively support `0.5u` dual-sided membrane slabs!).
   - Bury every wing & tail membrane slab **`>= 1.5–2.0u` deep inside its host bone spar** and overlap consecutive articulated tail-fan slabs by `>= 2.5–3.0u` so membranes stay **100% fused with zero detachment** across flight, dive, and tail-whip animations!
4. **Right-Handed `+Z` Toe-Splay Sign Law & Per-Toe Isolated `sculpt_group`s:** On a `-X` left foot pointing `+Z`, `toe_in_l` (`anchor.x > 0`) MUST use `Ry(+θ)` (fanning medial `+X`) and `toe_out_l` (`anchor.x < 0`) MUST use `Ry(-θ)` (fanning lateral `-X`). Assign each toe its own `sculpt_group` so `audit_scene` strictly rejects crossed toes, assert outward tip X-divergence programmatically, and inspect an unobstructed **Top-Down Foot Splay** panel!
5. **True 4-Band Octagons, `leaf` Shader Dispatch & Two-Tier Bioluminescent Organ Glow (*Wintermoon Nettle, Wirebug, Puppet Spider, Sandstar*):**
   - Build clean organic domes/bells from **4 explicit thin flat-sided bands** (`[D, round_half(D*tan(22.5°)), depth]` at `0°/45°/90°/135°`, `prism_group`, nonnegative staggered `+0.02*i` end-cap inflate) rather than two spiky `0°/45°` squares!
   - Always dispatch procedural shaders on **`leaf = box.name.lower().split("__")[-1]`** so `mirror_x` ancestor prefixes (`bell_lobe_r__tentacle_l_rib3`) never shadow child shaders!
   - **Two-Tier Glow Policy:** Keep focal HDR emissive cores `<= 4.0%` on standard bosses/dragons; on naturally bioluminescent endemic life (jellyfish, fireflies, Wirebugs, Sandstars), pair luminous high-value photophore albedo with `<= 4–6%` focal HDR emissive core (or `<= 30%` soft low-intensity subsurface organ wash) contrasting sharply against deep cool non-emissive carapace!
   - On `small`-tier critters, `uv_scale: 2` (`64 px/block`) elevates not just eyes (including 6-eye spider clusters & `14×14 px` compound insect ommatidia!) but also micro-landmarks like twin-nostril pink nosepads.

---

## 5. Quick Pre-Export Verification Checklist (`v38`)
- [ ] **Non-Fandom High-Res References + Visual Primitives Checked:** Downloaded native non-Fandom references (incl. close-up extremity/claw/pattern artbooks or 4K sculpts) and ran `visual_primitives_prep.py zoom --autocrop --with-grid --with-edges` + concavity trace before `W2`.
- [ ] **Periorbital Sightline Window & Toe Splay Verified:** Eyes unobstructed in Dead-Front (`0°`), 3/4, and Side views (`uv_scale: 2`, all 7 eye layers); sibling toes in isolated `sculpt_group`s fanning cleanly outward in an unobstructed Top-Down Foot Splay check.
- [ ] **Zero-Blink Dual-Sided Membranes & Continuous Bonding:** Dual-sided differently-colored membranes use `0.5u` staggered slabs (0 Blockbench Z-fighting blink!), rooted `>= 1.5–2.0u` inside spars with zero detachment in flight/whip/swim clips.
- [ ] **Universal Multi-Skeleton Gates Pass:** `audit_scene` (`0` SAT collisions, `0` Z-fighting), upgraded `construct_check.py` (`G1–G14` across `.blockdsl`/`.bbmodel`/`.geo.json`), upgraded `verify_bbmodel_ground.py` (quadruped/biped/arthropod/hover/floating), and full `audit_behavior_clip(..., max_frames=288)` (`0` swept collisions, native `swim {}` for underwater creatures).
