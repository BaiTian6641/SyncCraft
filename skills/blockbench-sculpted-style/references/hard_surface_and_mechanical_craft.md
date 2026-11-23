# Hard-Surface, Mechanical, Vehicle & Sci-Fi Blockbench Craft Guide (`v35`)

> **Purpose:** Extend the `blockbench-sculpted-style` aesthetic beyond organic creatures and characters into **all Blockbench genres** — vehicles, mechs, walkers, airships, turrets, robots, weapons, architectural props, and hard-surface equipment — without losing our signature crisp low-poly sculpting, clean pixel-art shading, and exact mechanical animation discipline.

---

## 1. Format & Topology Discipline: Generic Mesh vs. Cuboid-Only Engines

Before placing the first primitive, match your modeling geometry to the target runtime:

| Target Format | Primary Primitives | How to Carve Recesses, Chamfers & Cylinders |
|---|---|---|
| **Blockbench Generic Model / glTF / Sketchfab / Unity / Godot** | **Meshes** (cuboids, cylinders, prisms with Face/Edge/Vertex editing, Loop Cut, Extrude `Shift+E`, Inset, Create Face `Shift+F`) | Subdivide a base cuboid with evenly spaced **loop cuts**, delete faces to open wheel wells / cargo beds / cockpits, extrude interior walls inward, and close inner faces cleanly. Use `8`- to `12`-sided cylinders for wheels, turrets, thrusters, and pistons. |
| **Minecraft Bedrock / GeckoLib / SyncCraft (`.geo.json` / `.blockdsl`)** | **Rotated Cuboids + Hard-Alpha Planes** (`0.5u` size step, clean `sculpt_group` overlaps) | Build recessed cavities (wheel wells, cockpit bays, intake vents) out of **perimeter frame cuboids surrounding a recessed dark backing box** rather than carving a single mesh. Build octagonal/faceted tires, engine cowls, and cannons from `2–4` symmetrically rotated overlapping cuboids or layered chamfer plates. |

Both pipelines share the **exact same visual, silhouette, rigging, UV, and pixel-art shading laws** below.

---

## 2. Structural Carving Laws for Vehicles, Mechs & Machinery (Anti-Box-Car Law)

A beginner mechanical Blockbench model looks like a painted shoebox with stickers on the side. A studio-grade mechanical model carves real physical negative space and industrial bevels:

### 2.1 Ground Clearance & Polygonal Wheel Flat-Tread Math
1. **Establish ground clearance first:** Lift the chassis above `y = 0` so tires, treads, or mech feet carry the machine cleanly above the ground plane (`y = 0.00u`) with visible underbody clearance.
2. **Flat-tread alignment for low-poly wheels:** On an `N`-sided polygonal tire cylinder (e.g., `8`-sided octagon), rotate the wheel geometry by **`180° / N`** (`22.5°` for an `8`-sided wheel) so a **flat tread facet** sits flush on `y = 0` rather than balancing awkwardly on a sharp bottom vertex! Verify flush alignment in flat orthographic side view.
3. **Recessed rims & hubs (never flat hockey pucks):** Inset the outer wheel face (`Alt` loop-select -> `Inset`) and extrude inward by `1 px` (or nest a slightly narrower recessed hub cuboid inside the tire rim in Bedrock) so tires read with real physical sidewall thickness and axle depth.

### 2.2 Mandatory Recessed Cavities (Wheel Arches, Cargo Beds, Intake Vents & Glazed Cockpits)
Never paint wheels, truck beds, vents, or cockpits onto a flat solid hull wall:
- **Wheel Arches:** Frame every wheel inside a recessed physical arch cavity (top fender deck + angled chamfered front/rear wheel-well walls + dark interior backing). Give at least `0.5–1.0u` clearance around the rotating tire so turning or suspension bounce never clips the fender!
- **Cargo Beds / Turret Wells / Engine Bays:** Leave an evenly spaced structural rim border around the opening and extrude the floor downward into the chassis (stopping cleanly above axles/wheels underneath).
- **Glazed Multi-Layer Cockpit Canopies (Mechs, Starfighters, Vehicles):** Frame a **chamfered translucent glass shell** (tinted cool slate-blue/cyan with diagonal pixel-art glass reflection streaks) **OVER a genuine dark recessed 3D cockpit interior** (control console, pilot seat, internal bulkhead ribs visible behind the glass).

### 2.3 Compound Silhouette De-Boxification (Slants, Hood Curvature & Approach Angles)
Break raw 90° box corners on every major hull volume:
- **Windshields & Canopies:** Slant the upper windshield edge backward and taper the cabin slightly narrower than the lower beltline.
- **Aerodynamic Hoods & Armor Cowls:** Pull outer front corners backward and step the upper front edge downward in `2` subtle stages to build a clean curved nose/hood crown.
- **Bumpers & Underbody Skids:** Angle lower front/rear bumpers inward (approach & departure clearance) so the chassis feels engineered to traverse real terrain.
- **Secondary Hard-Surface Fixtures:** Mount side mirrors, antennas, grab-handles, exhaust stacks, and headlight housings on explicit **offset stalks / recessed bezels** (`1 px` clear of the main hull), angled purposefully (e.g., side mirrors tilted toward the driver).

### 2.4 Alternating Painted Armor Casings vs. Dark Mechanical Skeleton (Mechs & Walkers)
For articulated mechs, walkers, cranes, and robotic limbs:
- **Never fuse an entire robot out of one solid color.** Alternate **chamfered painted armor casings** (hull cowls, shin guards, shoulder pauldrons, armored leg segments) with **exposed dark gunmetal/charcoal mechanical joints** (clevis hinges, hydraulic cylinders, universal knuckle blocks, shock struts, flexible conduits).
- Every articulated bend (shoulder, elbow, hip, knee, ankle, claw pincer) should visibly expose a recessed dark mechanical axle/knuckle block flanked by beveled armor cuffs — immediately communicating how the machine physically moves!

---

## 3. Hard-Surface UV Hygiene, Seam Control & Mirror Painting

Official Blockbench UV discipline is critical on hard-surface pixel art because straight panel lines, grilles, headlights, and decals expose any half-pixel warp instantly:

1. **Uniform Texel Density & Zero Stretching:** Every exterior hull panel must share the exact same pixel-per-unit ratio (`16 px/block` or `32 px/block`, with optional `2×` focal budget ONLY on dense instrument panels/emblems). Never stretch a single UV rectangle across a long fuselage!
2. **Always Enable UV Padding (`>= 1–2 px`)** when generating texture templates so UV islands have safe adjustment headroom and zero mipmap/bilinear color bleed across panel edges.
3. **Intentional Seam Placement (`Seam Tool: Auto / Divide / Join`):**
   - **`Divide`** edges separating distinct materials or sharp structural breaks (e.g., glass windshield vs. painted hood, armor cowl vs. recessed dark intake) so colors never smear across seams.
   - **`Join`** edges across smooth chamfers/bevels that should receive continuous wrapped shading or decals.
4. **UV Preview Checkerboard Gate (Mandatory Pre-Paint Check):** Switch to **UV Preview (checkerboard)** BEFORE painting a single stroke. Inspect front grilles, bumpers, canopies, and symmetrical hull panels; drag misaligned UV vertices with **`Shift` pixel-grid snapping** until every panel border is crisp and 100% symmetrical left-to-right.
5. **X-Centered Mirror Painting & Element Base-Blocking:**
   - Center the entire chassis strictly on **`X = 0`** so **Mirror Painting** paints bilateral armor, decals, headlights, and rivets symmetrically in half the time.
   - Block macro zones first (`Fill > Element`), then paint directional pixel-art lighting:
     - **60% Primary Painted Hull:** Bright sunlit top deck (hood/roof/pauldrons), clean mid-tone flanks, cool hue-shifted lower hull shadows, subtle `1 px` bright beveled top edges + `1 px` dark panel-seam crevices underneath.
     - **30% Matte Dark Gunmetal / Rubber / Underbody:** Tires, recessed wheel wells, hydraulic joints, chassis frame, louvered grille slots & cooling vents (painted darker inside recesses for strong structural depth).
     - **10% Crisp Functional Accents & Decals:** Diagonal `1–2 px` pale reflection streaks across dark glass windshields/canopies, amber/white recessed headlight & taillight lenses inside dark bezels, razor-sharp white/warning-yellow hull striping/stencil decals, and tiny recessed cyan/amber glowing LED servo status nodes on mechanical joints (kept `<= 3%` emissive area!).

---

## 4. Rigid Mechanical Rigging & Animation Laws

Mechanical models obey **rigid-body kinematics**, NOT soft biological tissue rules:

1. **Root Group + Functional Sub-Group Rigging (`groups = the rig`):**
   - Enclose the entire vehicle/mech inside a root chassis group (`root` / `chassis_root`) for suspension heave, acceleration squat, braking nose-dive, and banking roll.
   - Isolate every moving mechanical sub-assembly into its own bone group (`wheel_left_front`, `wheel_right_front`, `wheel_left_rear`, `wheel_right_rear`, `turret_yaw`, `turret_pitch`, `cockpit_hatch`, `leg_fl_coxa`, `leg_fl_femur`, `leg_fl_tibia`).
   - **Exact Axle Pivot Centering (`P` / Center Pivot) BEFORE Mirroring:** Place each wheel/hinge pivot at the exact geometric center of its axle/cylinder BEFORE duplicating (`Ctrl+D`) and mirroring (`Transform > Flip X`, which automatically renames `left -> right`).
2. **Rigid Kinematic Constraints:**
   - **Zero `Sim(3)` Muscle Bulge on Rigid Steel:** Never apply soft-body scale bulging (`scale_bulge = 0`) to metal armor plates, wheels, or steel pistons! (Reserve scale animation solely for telescoping hydraulic inner rods if explicitly rigged, or thruster exhaust cones.)
   - **True Mechanical DOFs:** Wheels spin strictly around local `X` (`Δθ_wheel = Δdistance / R_wheel` with zero ground slip!), front steering hubs yaw around `Y` (`±30°`), hinged armor hatches and knee clevises rotate cleanly around a single pin axis (`1-DOF`), and bipedal/multi-legged mech feet obey the **exact same `SO(3)` flat-sole grounding & `SE(3)` zero-slip stance lock** (`0.00u` floor penetration) as organic walkers!
3. **Presentation & Silhouette Audit:** Inspect final hard-surface renders with ground contact shadow / subtle AO on a clean backdrop to confirm crisp panel-edge highlights, clear recessed cavities, and zero floating wheels or feet.
