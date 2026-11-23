# Ecology Research, Whole-Body Animation Balance & Micro-Depiction Aesthetic Refinements (`v21`)

Distilled from:
- **YouTube *Monster Hunter Rise: Sunbreak* Wildlife Ecology Footage:** *Primordial Malzeno Ecology* (`U5Gi1UZxmp0`, 682 s), *Qurio State Posture Changes* (`-XvsEr7QjYU`, 406 s), *Roar & Bloodlust Supernova* (`32S_44AhuVs`, 55 s), and *Paladin Wing-Shield & Halberd Tail Combos* (`k5EcPQ_H1Zk`, 46 s).
- **4X Ultra-Zoom Micro-Depiction Audits** comparing our `v20` build directly against the 4 Owner Master Style Anchors (`anchor_06` Amethyst Runic Dragon, `anchor_07` Deviljho, `anchor_08` Tetranadon, `anchor_09` Gore Magala) and in-game 4K ecology close-ups (`study_04` `283–287s` head/snout/eye & `473–477s` halberd tail).

---

## Part 1: Why Static Key Art Lies — The Mandatory YouTube Ecology Research Step

Official promotional renders and turnaround sheets almost always depict winged creatures in **maximum-display combat poses** (wings spread vertically straight up like an open billboard, tail curled high). Copying that static render into `idle` and `walk` makes animations feel **stiff, robotic, and weightless**.

Before authoring Animation Pass `A1` on any hero or boss creature, **search YouTube for HUD-free wildlife / ecology observation videos** (e.g., `"<monster> ecology"`, `"<モンスター名> 生態 観察"`, `"<creature> idle walking sleeping grooming animations"`) and extract timestamped motion study sheets for:
1. **Resting / Patrolling Wing & Tail Carriage:** How does the creature fold or drape heavy wings when calm? (*Primordial Malzeno* drapes her wings sleekly backward over her flanks like a **heavy Knight's Cloak** with carpal hubs high and long blades trailing past her hips, while her Halberd tail glides **low and level** behind her as a horizontal rudder — completely different from her vertical combat sail!)
2. **Signature Non-Combat Behaviors:** Royal Sentinel Sit (`"お座り"` — sitting upright on haunches with wings draped around her torso like a cathedral cloak), Bow-Stretch Wing Unfurl (chest dips low, forelimbs bend, tail arches high behind as a counterweight while wings bloom outward), Grooming, and Horizon Gaze Scans.
3. **Combat Weight Transfer & Momentum Recovery:** How does the center of mass (`pelvis_root`) drop, pitch, yaw, and roll *before* the blade lands, and which limbs step/brace to absorb recoil?

---

## Part 2: The 5 Universal Whole-Body Balance & Kinetic-Coupling Laws (`Section 11.5d`)

### Law 1 — Zero Frozen Appendages (Six-Subsystem Kinetic Coupling)
Never animate a wing slash, roar, or walk cycle by moving only 1–2 appendages while leaving `pelvis_root.rotation`, `arm`, `leg`, `neck`, or `tail` frozen at `[0, 0, 0]`. Every frame of every clip must visibly coordinate **all six subsystems**:
1. **`pelvis_root` (Position + Pitch / Yaw / Roll):** Center-of-mass heave, forward attack lunge pitch, rear-up supernova pitch, and lateral stride hip roll.
2. **`waist` + `chest` (Spinal Counter-Torque):** Counter-yaws opposite `pelvis_root` during quadruped strides; coils *with* `pelvis_root` during anticipation and snaps explosively through combat strikes.
3. **`neck_1..3` + `head` (Hawk Gaze Stabilization + Breathing):** Counter-rotates against torso bob/yaw so the creature's eyes stay fiercely locked on the horizon or target; subtle `jaw` breathing (`2–4°` idle, `22–38°` roars/impacts).
4. **`wings` (`upper -> fore -> wrist -> f1..f4`):** Draped **Knight-Cloak** fold in calm idle/walk/sit with subtle alternating shoulder inertia lag opposite each forelimb step; rapid asymmetric downstroke + `30°` root-to-tip whip lag in flight; counter-elevated clear of the hips during tail sweeps.
5. **`limbs` (`arm_l/r` + `leg_l/r`):** Never static pillars during combat! Plant and splay forelimbs forward when the tail arcs overhead (`tail_halberd_pierce`); step the leading forelimb into asymmetric wing thrusts (`twin_wing_lance_thrust`, `wing_crescent_sweep`); clutch claws to chest during rear-up roars and fling them wide on detonation (`bloodlust_sacred_supernova`).
6. **Weighted Tail (`tail_1..6` + terminal weapon):** Low S-gliding horizontal rudder during locomotion; **equal-and-opposite horizontal whip** conserving angular momentum during every asymmetric wing/torso slash; high rear counterweight arch during forward bows and crouches.

### Law 2 — Conservation of Angular Momentum (Opposite-Side Counterbalance Table)
| Action / Impulse | Torso (`pelvis + chest`) | Limbs (`arms / legs`) | Wings | Weighted Tail (`tail_1..6`) |
|---|---|---|---|---|
| **Calm Idle & Patrol Walk** | Subtle hip roll/yaw (`±3°`), chest counter-yaw (`∓5°`), gaze-stabilized head | Diagonal quadruped gait + soft paw weight shift | **Draped Knight-Cloak** trailing sleekly aft, swaying subtly opposite shoulders | **Low level horizontal S-wave** opposite pelvis yaw (`-12°` root pitch) |
| **Forward Bow / Stretch / Overhead Tail Thrust** | `pelvis_root` pitches **forward (`+12..16°`)**, chest dips low | Forelimbs **plant wide & flex** to bear front weight; hindlegs brace aft | Spread wide horizontally for roll balance | Arches **high overhead** as rear/forward counterweight |
| **Asymmetric Wing Cleave / Spear Thrust** | Drops low, **whips yaw/roll INTO strike**; head counter-yaws to lock eyes on target | **Leading forelimb steps into strike**, opposite hindleg anchors | Off-hand wing lifts high & clear as counterbalance | **Hurls horizontally to OPPOSITE side (`±20–26°`)** with root-to-tip whip lag |
| **Rear-Up Roar / Aerial Supernova** | `pelvis_root` pitches **steeply back (`-18..-22°`)** + airborne arch | Forelimbs **clutch chest on windup -> fling wide on burst**; hindlegs trail | Wrap tight around chest in cocoon -> **explode into wide cruciform spread** | Curls under for thrust -> **snaps aft & splays terminal weapon** |

### Law 3 — Mandatory SAT Swept Clearance on Draped Cloaks & Extreme Poses
Before writing multi-keyframe clips that use tight folded wings (`Knight-Cloak`, `Royal Sit`, `Greatshield Guard`), **probe the key rest/apex pose through 15-axis OBB `SAT` first** (tuning `wing_upper` forward yaw vs. `wing_fore`/`wrist` fold roll until `coll == 0` with `>= 2–3u` flank margin), then verify the entire clip with `288`-frame adaptive discrete-step swept SAT (`0` collisions, `0.000` root XZ drift on locomotion).

---

## Part 3: Micro-Depiction Aesthetic Refinements (`v21` — From 4X Ultra-Zoom Audits vs. Anchors `06–09`)

### 1. Zero-Leak Single-Face Recessed Dragon Eyes (`Trap #12`)
- **Failure Mode Caught at 4X Zoom:** Placing an eye box centered near the front-outer corner of the cranium (`anchor=[-0.78, 0.28, 0.58]`, `align=[0,0,0]`) AND painting glowing red iris pixels on `west`, `east`, AND `south` causes **both eyes to physically bulge and visually leak out** past the brow, nasal bridge, and cheek edges like bug-eyes.
- **Master Rule (Anchors `06`, `08`, `09` + Ecology Study `04` `283s`):**
  1. **Recess the socket flush inside the lateral skull wall** (`behind` the front face of `cranium_core` with only `~0.5–0.7u` lateral relief, cleanly crowned by a swept overhanging brow visor and framed tightly below by the cheekbone with zero gaps).
  2. **Single-Outer-Face Iris Rule:** Paint the glowing eye **STRICTLY on the single outward-facing lateral face** (`"west"` on left `-X` eye, `"east"` on right `+X` eye). Paint ALL other 5 faces (`south`, `north`, `up`, `down`, inner side) solid **dark obsidian orbital socket** so eye color can *never* leak forward or sideways.
  3. **4-Layer Dragon Eye Pixel Craft:** `1 px` deep charcoal-obsidian orbital socket frame -> glowing fiery iris gradient -> **crisp `1 px` vertical dark reptile slit pupil** (or tiered horizontal lid on amphibians like Anchor `08`) -> `1 px` warm-white catchlight speck offset `1 px` forward-up from the pupil.

### 2. Aristocratic Snout / Upper-Jaw Taper — Ban Ugly Dangling Parrot Hooks (`Trap #13`)
- **Failure Mode Caught at 4X Zoom:** Stacking steeply pitched `28°` + `48°` blocks on the front of a dragon snout creates a clunky, bulbous vulture/parrot hook hanging awkwardly below the jaw line (and risks Z-fighting against lower chin spikes).
- **Master Rule (Ecology Close-Up `study_04` `283s/477s` + Anchor `06`):**
  - Keep noble dragon / knightly visor snouts **sleek, elongated, and unified along ONE continuous gentle downward taper (`9° -> 13° -> 18°` max)** where the upper prow tip stays flush with the jawline (`head_L/H ~ 1.75–1.95`).
  - Paint a **crisp dark slit nostril** (`2x1 px`) on the lateral nasal bridge and **staggered ivory dragon teeth over a `1 px` dark gum-groove** along the straight lower edge of `maxilla_wall` (Anchor `06`), keeping bulky extruded fang cubes subtle and integrated inside the lip contour.

### 3. Disciplined Saturation, Weathered Metals & Organic Shingle Scales (`Trap #14`)
- **Failure Mode Caught vs. Anchors `06–09`:**
  1. Over-saturated "toy" primary metals (baby-blue silver, cartoon banana-yellow gold) and blown-out paper-white wing nebula cores (`RGB > 235`) make epic Elder Dragons look like bright plastic toys compared to the grounded, moody palette discipline of Anchors `06–09`.
  2. Painting a uniform bright `yy == 0` line across every armor cuboid makes layered pauldrons look like glowing stacked Lego tiles, and leaving side ribcage faces (`chest_core` `east/west`) plain gray breaks sculptural continuity.
- **Master Rule (`v21` Palette & Painterly Craft):**
  - **Hold 85% of the creature at disciplined `20–42%` weathered saturation:** shift silver to **warm pearl-platinum over cool slate-ash shadows**, gold to **burnished antique champagne-gold over umber-bronze crevices**, and war-bronze to **smoked aged bronze** (`study_04` `473s`). Cap translucent wing nebula highlights around moody luminous periwinkle (`RGB ~ 185–205`), never blown-out white.
  - **Organic Shingle Scales & Carved Flank Ribs:** Break `40%+` of box-top bevel pixels organically, paint **staggered overlapping `U`-scallop dragon-scale shingles** across pauldrons, cuirass plates, and tassets, and carve diagonal silver rib-plates across exposed lateral torso walls (`chest_core`).
  - **Sleek Contour-Hugging Pauldrons & Overlapped Cuirass:** Taper shoulder tassets tightly along the deltoid flow (`thickness <= 4.5–5.0u`) rather than sticking square blocks outward; verify chest cuirass plates overlap continuously from throat gorget to keel with zero dark rectangular gaps.
  - **Mandatory 4X Ultra-Zoom 6-Angle Part Audit:** Before finalizing `W5` or `T5`, render 4x zoomed 6-angle sheets (`FRONT`, `3/4`, `PROFILE`, `TOP`, `UNDER`, `BACK`) for **Head/Eyes/Snout**, **Torso/Cuirass/Pauldrons/Hands**, and **Hindlimbs/Wings/Tail**, and visually inspect every joint for leaks, hooks, Z-fighting, or flat gray walls.
