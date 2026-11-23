# Aesthetic Mood Spectrum (`M1–M4`), Approachability Polarity & Universal Domain Router (`v28`)

> **Why This Document Exists:** A formal Blockbench aesthetic must cover the **entire creative spectrum** — from cozy blocks, dioramas, vehicles, and mechs to **legendary imaginary creatures of opposite temperaments**: some legendary creatures should feel **gentle, benevolent, ancient, and approachable** (guardian spirits, celestial mounts, *Wackyblocks Ocean Cargo Fish*), while others MUST feel **furious, wrathful, perilous, and "unapproachable"** (*White Fatalis*, *Fatalis*, *Deviljho*, *Gore Magala*, *TrongOBD Zelno*). Without explicit geometric & value polarity laws, AI models chronically drift toward a "polite neutral toy middle" — making wrathful Elder Dragons look calm/tame! This guide gives **both human artists and small AI models** a deterministic **`Domain D1–D5` × `Temperament M1–M4`** router plus exact mathematical sign & value rules to nail the intended emotional aura on Turn 1.

---

## 1. Turn-1 Deterministic Intent Router (For Small AI Models & Artist Co-Pilots)

Before creating Cube `#1`, classify the artist's goal on **two orthogonal axes** and declare `Target = (Domain, Mood)` in the build plan:

### Axis A — Asset Domain (`D1–D4`)

| Domain | Asset Scope | Typical Scale & Element Budget | Core Pipeline Priority |
|---|---|---|---|
| **`D1` Blocks, Props, Items & Weapons** | Crates, lanterns, furniture, tools, swords, bows, gathering outcrops, decorative scenes (`Anchor 04`). | `12–65` elements, `1–12` bones (`16 px/block`, `0` meshes). | Crisp iconic silhouette at hotbar/world distance, contact-pivot organic tilt (`5–15°`) on loose clusters, zero redundant buried cubes. |
| **`D2` Small Fauna, Critters & Chibi Companions** | Pets, small woodland/ambient mobs, baby dragons, stylized companions (`Dunehop`, `Pebble Weaver`, cute pets). | `15–50` elements, `6–20` bones (`>= 2u` chunky boxes, cut-out ears/fins). | Push `2–3` signature traits `1.3–1.5×`; expressive readable eyes; drop micro-plates to clean painterly strokes. |
| **`D3` Standard Creatures, Wildlife & Hero Mobs** | Realistic animals, mounts, knights, standard fantasy monsters, Cobblemon-scale creatures (`Anchor 01–03, 06, 08`). | `60–220` elements, `25–75` bones. | Biological joint honesty, grounded `4`-foot or biped balance, clean cross-cube pixel-relief carving, full `W1–W3 / T1–T2 / A1` passes. |
| **`D4` Legendary / Imaginary Bosses & Apex Monsters** | Elder Dragons, Monster Hunter bosses, mythic titans, raid bosses (*White Fatalis*, *Primordial Malzeno*, *Gore Magala*, *Deviljho*, *Zelno*; `Anchors 05, 07, 09`) AND benevolent mythic guardians (*Ocean Cargo Fish*). | `240–550` elements, `80–160` bones (`G1–G12` epic boss gates). | Overwhelming silhouette presence, explicit **Gentle/Approachable vs. Furious/Unapproachable polarity lock**, compound `40–85%`-sunken cuboid packing, 100%-span co-parented wings, Standing-IK ground pinning (`ground_pin_limbs`). |
| **`D5` Large Vehicles, Airships, Mechs & Architectural Dioramas** | Pickup trucks (`WbyCbA1c8BM`), tanks (`cnwSSjQDA1A`), cargo airships (`2pc-iXaSjYM`), battle robots (`rycGulcQ-Lo`), mini-homes (`NB_uHkB1dz8`). | `80–400` elements, `12–90` bones (`wheel_axle`, `steered_wheel_knuckle`, `turret_yaw`, `cannon_elevation`, `hydraulic_hinge`, `hatch_door`). | Chassis-first wheelbase/track framing, recessed wheel wells + octagonal tires, sloped glass-plane cabs, rigid `SO(3)/SE(3)` motion (`0%` `Sim(3)` squishy bulge on metal!), industrial panel-line & mud/rust weathering. |

### Axis B — Aesthetic Mood Tier (`M1–M4`)

| Mood Tier | Artist Intent Cues & Reference Archetypes | Primal Visual Psychology |
|---|---|---|
| **`M1` Cute / Adorable / Cozy** | *"cute", "chibi", "baby", "pet", "cozy", "friendly", "plush", "adorable", "wholesome"*; cozy pantry props (`Anchor 04`), baby mobs, companions. | **Neoteny (*Kindchenschema*):** rounded/soft-square silhouettes, oversized head, big low-set glossy eyes, relaxed/upright cozy posture, warm soft high-key palette. |
| **`M2` Normal / Vanilla-Plus / Grounded** | *"vanilla style", "normal", "standard mob", "block", "item", "weapon", "realistic wildlife"*; clean functional Minecraft-native entities & tools (`Anchor 03`, Cobblemon baseline). | **Clarity & World-Fit:** balanced biological/functional ratios, crisp neutral expressions, honest rectangular/45°-chamfered cuboids, clean top-lit vanilla-kin shading. |
| **`M3` Noble / Heroic / Realistic-Vigorous** | *"majestic", "noble", "regal", "heroic", "realistic beast", "graceful", "paladin", "refined"*; proud mounts, valiant knights, dignified mythical beasts (`Anchor 01, 06`). | **Poise & Athletic Grace:** tapered wedges, sculpted S-curves, focused inset eyes, poised alert brow, weathered `20–42%` saturation metallics & natural hides. |
| **`M4` Ferocious / Apex / Legendary Boss** | *"ferocious", "terrifying", "apex", "legendary boss", "Elder Dragon", "predator", "menacing", "Monster Hunter boss", "Cataclysm/Mowzie's boss"*; (`Anchors 05, 07, 09`, *Primordial Malzeno*, *Fatalis*, *Gore Magala*). | **Lethal Predation & Dread:** sharp acute triangles & diagonals, coiled low-slung predator crouch, forward-surging S-neck + downward head glare, plunging golden/armored `V`-brow, narrow binocular glowing slit eyes out of an obsidian war-mask, swept-back aerodynamic scimitar horns, razor arrowhead snout, dramatic chiaroscuro contrast. |

> **Small-AI Routing Rule:** When the artist specifies an iconic boss/monster IP (e.g., *Monster Hunter Elder Dragon*, *Dark Fantasy Dragon Boss*, *L_Ender's Cataclysm*-style raid boss) without requesting "cute/chibi", **DEFAULT AUTOMATICALLY TO `(D4, M4 Ferocious / Apex Legendary)`** — NEVER default to polite upright `(M2/M3)` posture! When the artist requests a weapon, block, tool, or prop, route to **`(D1, M2)`** (or **`(D1, M4)`** for dark legendary boss weapons with jagged silhouettes and focal emissive cores).

---

## 2. The 10-Knob Quantitative Mood Control Matrix (`M1 → M4`)

A small AI model must NEVER rely on vague words like "make it scarier." Set these **10 concrete structural and painterly knobs** directly according to the selected Mood Tier:

| # | Control Knob | `M1` Cute / Adorable | `M2` Normal / Vanilla-Plus | `M3` Noble / Realistic-Vigorous | `M4` Ferocious / Apex / Legendary |
|---|---|---|---|---|---|
| **1** | **Primary Shape Grammar** | Soft squares, rounded-chamfer pills, bottom-heavy pear | Honest rectangles, clean `45°` bevels | Tapered sculpted wedges, athletic S-curves | **Razor acute triangles, aggressive diagonals, top-heavy inverted-`V` shoulders** |
| **2** | **Head : Body Mass Ratio** | `1:1` to `1:2` (oversized chibi head) | Biological `~1:3` | Athletic `~1:3.5` | **Compact lethal skull on massive surging chest/wings (`1:4–1:5`) OR huge bone-crushing brute jaw (`Anchor 07`)** |
| **3** | **Eye Size, Angle & Pupil** | Large `3×3–5×5 px`, low on face, wide round pupil + big white specular dot | `2×2–3×2 px`, neutral side-front (`0–10°` yaw) | Inset `2×2` or `3×2 px`, calm focused iris (`10–15°` yaw) | **Compact sleek horizontal almond slit (`5×2 px`, `2.0u` tall — NEVER a tall vertical red brick!), binocular-angled (`20°` yaw), dark hawk-eyelid top shadow + intense blood-ruby iris + crisp `1 px` jet-black pupil inside an obsidian orbital mask (verified via Pass-1→Pass-2 Eye Ultra-Zoom!)** |
| **4** | **Brow Angle (Emotional Switch!)** | Absent or curved UP (`pitch -10°..0°`) | Neutral flat (`pitch 0°..+5°`) | Subtle noble ridge (`pitch +6°..+10°`, `roll -14°`) | **Plunging `V`-brow scowl (`pitch +12°..+22°`, `roll -22°..-28°`) slashing sharply inward toward the nasal bridge!** |
| **5** | **Snout / Anterior Prow** | Short stubby button nose (`L/H 0.9–1.2`), flat friendly front | Proportional muzzle (`L/H 1.3–1.6`) | Clean tapered visor/muzzle (`L/H 1.65–1.80`) | **Razor-narrow arrowhead/falcon prow (`width 2.0–2.5u`, angled `22–28°` down — NEVER a blunt bright-white square marshmallow bumper!) + flared swept cheek blades!** |
| **6** | **Horns / Ears / Crests** | Short stubby nubs or upright round ears (`70–90°` vertical) | Moderate natural antlers/ears (`30–60°`) | Regal swept crown (`25–45°` back) | **Aerodynamically swept-back scimitar horns & flared war-crest plumes (`45–60°` swept BACK — NEVER `80–90°` upright bunny ears!) + battle-charred dark tips!** |
| **7** | **Neck Curve & Head Pitch** | Short/vertical neck, head tilted `0°` or slightly UP (`-5°..0°`, eager) | Relaxed natural neck, level head (`0°`) | Proud arched S-neck, level regal gaze (`0°..+4°`) | **Forward-surging low predatory S-neck (`neck_1` forward `-18°..-24°`, NOT vertical `-40°`!) + downward Kubrick-stare chin tuck (net head pitch `+10°..+16°`, glaring UP under the V-brow!)** |
| **8** | **Posture & `belly_clearance`** | Cozy upright/sitting or stubby legs (`0.35–0.50`) | Balanced neutral stand (`0.42–0.52`) | Poised athletic stance (`0.38–0.44`) | **Low-slung coiled predator crouch (`belly_clearance = 0.26–0.36`), hunched surging chest, abducted bent panther elbows (`arm_l_fore -45°..-58°`), coiled digitigrade haunches — all 4 feet locked at `y = 0.00`!** |
| **9** | **Mouth & Dentition** | Hidden/small friendly smile or painted cute mouth | Simple clean jaw line | Clean closed visor + 6-plane interlocking pixel-square teeth on roar | **Menacing dark occluded jaw seam at rest + full 6-plane zero-thickness interlocking pixel-square fangs (`1×2 px` anterior canines), cavernous palate rugae, concave cheek sinew & curled 2-segment tongue!** |
| **10** | **Value Contrast (Chiaroscuro)** | High-key soft pastels (`steps 3–6`, gentle shadows) | Clean vanilla-balanced top-bright / bottom-dark (`steps 1–5`) | Weathered `20–42%` refined metallics & midtones | **Dramatic Gothic Chiaroscuro:** push face lower cheeks, jaw, orbital masks & under-plates `1.5–2` steps darker (`steps 0–2` gunmetal/obsidian/smoked bronze) so ONLY the razor nasal keel, V-brow, and blazing slit eyes slash bright! |

---

## 3. Empirical Case Study: Killing the "Cute & Dumb" Dragon Syndrome (`Primordial Malzeno v23 → v24`)

Even a technically clean `281`-element, `0`-collision Elder Dragon can read **"cute and dumb"** if its Mood Knobs accidentally drift toward `M1/M2`! Live owner evaluation of `v23` exposed **7 exact cute/dumb drift bugs**, all cured in one disciplined `M4` pass (`W13 + T10`):

1. **Vertical Giraffe Neck (`neck_1 = -40°, neck_2 = -22°`) + Stiff Upright Pony Stand (`belly_clearance = 0.45`)** → made the boss look like a friendly pet standing politely.
   - *`M4` Cure:* Lowered torso into a coiled crouch (`pelvis_root.y: 33.14 → 31.73`, `belly_clearance → 0.36`), bent panther elbows (`arm_l_fore: -20° → -51.5°`), surged `neck_1` forward (`-22°`), and tucked `head` into a `+12°` downward predator glare while keeping exact 4-foot grounding (`FL = 0.02, HL = 0.00`)!
2. **Upright `85°` Vertical Bunny-Ear Horns (`horn_l_1 = -16°` on `+46°` head)** → read as playful deer antlers / bunny ears!
   - *`M4` Cure:* Swept `horn_l_1` sharply backward `-46°` (`temple_fan_l` `-48°`) with dark battle-charred tips (`pm_04` / `pm_06`), creating supersonic aerodynamic war-scimitars!
3. **Hidden Eyes + Flat Upward Brow (`brow_visor = -10°`, `eye = 10°` yaw) — AND Over-Correcting Into a Tall Vertical Red Brick (`W13`):**
   - *Lesson:* First `v23` hid the eyes from dead-front, then blindly guessing fractional crop coordinates on official renders zoomed into the sky instead of the eyes, leading `W13` to build a `3.0u`-tall vertical red block that looked odd!
   - *`M4` Cure (`W14 + T11`, Verified via 2-Pass Locate-First Eye Ultra-Zoom on `pm_14`, `pm_06`, `pm_11`, `pm_15`):* Locked a **compact `2.0u`-tall (`5×2 px`) sleek horizontal hawk-dragon blood-ruby almond eye** at `20°` binocular yaw (dark hawk-brow upper eyelid row + crisp `3×1 px` ruby-pupil-ruby horizontal slit) nestled tightly beneath a seamless plunging golden `V`-brow (`+14°` pitch, `-22°` roll) and swept silver cheek blade — menacing from **dead front, 3/4, AND profile** with `0%` eye leak!
4. **Wide Flat White Marshmallow Nose (`beak_hook_tip` `3.5×3.5×3.0`)** → read like a stuffed puppy snout!
   - *`M4` Cure:* Narrowed `beak_hook_tip` to a razor-keeled `2.0×3.5×3.0` (`26°`) falcon-knight arrowhead prow!
5. **Chalky Uniform Pale-White Face Values (`steps 4–5` everywhere)** → washed out every predatory plane!
   - *`M4` Cure:* Added `T10` **Gothic Chiaroscuro** — pitch-dark obsidian orbital war-masks, `-0.22` tempered gunmetal-slate shading on jaws and lower maxilla cheeks, and smoked war-bronze horn roots & charred tips!

---

## 4. Mandatory Mood Anti-Drift Gate (Run Before Every Final Export!)

Every small AI model or artist pass MUST run this **10-second Mood Polarity Audit** on the `DEAD FRONT`, `3/4 HEAD`, and `SIDE PROFILE` renders:

- **If `Mood == M4` (Ferocious / Apex / Legendary Boss):**
  - [ ] Are **BOTH eyes** unmistakably readable and menacing from `DEAD FRONT` and `3/4` beneath a downward-plunging `V`-brow?
  - [ ] Do the primary horns/crests sweep **backward (`>= 40°`)** rather than sticking straight up like bunny ears?
  - [ ] Is the snout prow a **sharp angular wedge/keel** (NOT a flat pale square marshmallow)?
  - [ ] Does the `SIDE PROFILE` show a **coiled predatory crouch + forward-surging S-neck + downward glare** (NOT a polite vertical giraffe neck on straight vertical stilts)?
  - [ ] Does the face have **true chiaroscuro value contrast** (deep dark orbital masks & jaw shadows against razor highlights)?
  - **If ANY check fails → DO NOT EXPORT.** Flip the failing knob(s) from Section 2 immediately!
- **If `Mood == M1` (Cute / Adorable):**
  - [ ] Are proportions neotenic (`1:1–1:2` head, chunky short limbs), eyes large & low with bright specular catchlights, brows soft/absent, and values warm & inviting (zero grimdark obsidian horror masks)?
- **If `Domain == D1` or `D5` (Blocks / Items / Weapons / Vehicles / Mechs / Dioramas):**
  - [ ] Does the silhouette read instantaneously at world/hotbar scale, respect `16 px/block` texel grid (`0` mixels), use recessed wheel wells / octagonal hubs / rigid `SO(3)` mechanical hinges (`0%` squishy `Sim(3)` bulge on metal), and match the requested mood (`M1` cozy home vs. `M2` utility pickup truck vs. `M4` weaponized battle robot)?

---

## 5. `v28` Legendary Creature Temperament Polarity Law: "Gentle & Approachable" vs. "Furious & Unapproachable"

> **Owner Core Critique (`White Fatalis v28`):** *"I didn't feel like it's furious enough because I do feel like there's some basic aesthetic style problem in our skill... Some legendary imaginary creatures should be gentle and approachable, and some of them should be a little bit more furious and being 'unapproachable'."*

Even among monumental `D4` legendary imaginary creatures (`240–550` elements), **scale alone never creates emotion**. A `270`-element dragon with level brows (`Rx <= 0°`), a politely closed `0°` mouth, hidden eyes, smooth rounded flanks, and uniform ivory shading (`RGB 210–245`) reads as **calm, tame, and approachable** — catastrophic when building *White Fatalis*, yet ideal when building a *Celestial Guardian Whale* or *Ocean Cargo Fish*! Before placing Cube `#1` on ANY legendary creature, explicitly lock **Pole A (`Gentle & Approachable`)** or **Pole B (`Furious & Unapproachable`)** across all **8 Polarity Axes**:

| # | Polarity Axis | Pole A: **Gentle, Benevolent & Approachable** (*Ocean Cargo Fish*, Sanctuary Guardians, Friendly Mounts) | Pole B: **Furious, Wrathful & "Unapproachable"** (*White Fatalis*, *Fatalis*, *Deviljho*, *Gore Magala*, *Zelno*) |
|---|---|---|---|
| **1** | **Silhouette Edge Hazard ("Touchability")** | **Smooth, convex, welcoming curves:** rounded-stepped chamfers, blunt/curved horn tips, soft plumed crests, zero outward-stabbing body spikes (invites the player closer). | **Hostile thorny silhouette ("Stay Back!"):** razor-hooked beak prow, twin nose daggers, jagged swept cheek blades, outward-jutting shoulder/elbow/carpal spikes, and tattered soot-scorched wing tips. |
| **2** | **Eyebrow Euler Sign Law (`Rx` & `Rz` — #1 Trap!)** | **Open / Arched / Calm Brow:** **`Rx <= 0°` (`-12°..0°`)** lifts the anterior `+Z` brow end UPWARD (`-z·sin(Rx) > 0`), paired with gentle `Rz` (`0°..∓10°`), creating an open, kind, curious expression. | **Plunging `\ /` Razor Scowl:** **`Rx > 0°` (`+18°..+24°`)** plunges the anterior `+Z` glabella end **SHARPLY DOWNWARD** (`-z·sin(+22°) = -0.37`) while sweeping the rear `-Z` end high into the horns, paired with **`Rz = -26°..-32°` (Left) / `+26°..+32°` (Right)**! *(NEVER use negative `Rx` on an angry boss — it creates sad puppy eyebrows!)* |
| **3** | **Orbital Socket & Gaze Attitude** | **Warm Open Gaze:** large visible almond/round eye (`3×3` or `4×3 px`), soft warm sclera/catchlights, level or slightly upward head pitch (`-6°..+4°`). | **Abyssal Cavern + Binocular Kubrick Glare:** pitch-black recessed orbital pit (`RGB 10,4,8`) under overhanging scowl brow + searing crimson/amber pinpoint slash (`28°` binocular yaw, unoccluded in `0°` dead-front!) + coiled **`+16°..+22°` downward head glare**. |
| **4** | **Resting Jaw State & External Dentition** | **Closed Peaceful Lip (`0°–2°`):** smooth tucked mandible, teeth hidden at rest (only shown softly when yawning/eating). | **Permanent Tense Snarl Gape (`12°–16°` rest_euler!) + 3D Interlocking Tusks:** snarling diagonal `maxilla_l/r` cheek-blades (`±17°` yaw) baring a dark crimson oral cleft, 6-plane serrated teeth, AND **3D external interlocking upper & lower razor canines** (`tusk_upper_l/r` set laterally outside `jaw_chin` for `0.00u` SAT clearance!). |
| **5** | **Posture & Forelimb Tension** | **Relaxed / Buoyant Poise:** upright or buoyant level spine, comfortable stride stance, relaxed wing fold or gentle fin scull. | **Coiled Spring / Cataclysmic Towering Loom:** low-crouched hind haunches (`belly_clearance ~ 0.28–0.33`), surging barrel chest, wide bowed `~60°`-flexed muscular forelimbs gripping the ground, and **`-60°` world-pitched towering frontal cathedral wings** framing the skull! |
| **6** | **Fur, Mane & Ruff Flow** | **Soft Plump Plumes:** fluffy rounded 3D tufts and softly scalloped feather fans with gentle pastel root-to-tip transitions. | **Windswept Razor Lightning Locks:** 3D backward-swept (`Rx +28..+34°`) chin beard + **4-spike razor-pointed cheek fur blades** (using `box.center[0] < 0` so mirrored right ruffs cut on the outer edge!) with storm-indigo root shadows. |
| **7** | **Value Contrast Ratio (Chiaroscuro)** | **Harmonious `3:1` High/Mid-Key (`RGB 95 -> 245`):** warm sunlit midtones, soft indigo-sky shadows, gentle transitions. | **Violent `10:1` Elder-God Chiaroscuro (`RGB 20 -> 254`):** storm-charcoal/obsidian scale trenches, dark battle-scarred nasal bridge & orbital war-mask, stark alabaster ridge glints, and pulsing arterial-crimson throat/chest lightning fissures! |
| **8** | **Idle & Combat Motion Language** | Slow sinusoidal breathing (`0.20–0.28 Hz`), curious head tilts, soft ear/fin flutter. | Heavy coiled diaphragm heave, piercing vestibulo-ocular `head_stabilize` lock, hind-IK-pinned bipedal thunder roars (`ground_pin_limbs: [hind_l, hind_r]`), and high-snap predatory strikes! |
