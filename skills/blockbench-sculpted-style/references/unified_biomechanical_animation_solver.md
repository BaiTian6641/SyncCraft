# Universal Unified Biomechanical & Mechanical Animation Solver (`SO(3)` / `SE(3)` / `Sim(3)`), Multi-Joint Fold-Preserving IK, Standing Limb Pinning & Minecraft Entity-Space Contract (`v32`)

> **Audience:** Human Blockbench artists, Minecraft modders (GeckoLib / Bedrock / AzureLib / Fabric / Forge), and multimodal or small AI co-pilot models assisting an end-to-end sketch-to-animated-asset pipeline.
> **Scope:** Truly universal across **every** real animal, humanoid, arthropod, aquatic/serpentine creature, winged flyer, **brand-new imaginary creature**, AND **large vehicles, airships, battle robots/mechs, and articulated props** — covering **full body from torso/spine, COM balance, serpentine neck Spline-IK, 100%-span co-parented wings, limbs -> wrists/ankles -> Reverse-Foot Rocker -> Standing Limb IK Pinning (`ground_pin_limbs`) -> multi-phalanx digits & opposable thumbs, whole-body `Sim(3)` antagonist muscle bulge, AND rigid mechanical `SO(3)/SE(3)` axles, steering knuckles, rotors, turrets, cannons, and hydraulic hinges**, regardless of limb count or bone naming language.

---

## 1. Core Philosophy: Skeleton + Muscles First, Never Brittle Keyframe Spam

Traditional Blockbench animation forces artists (and AI models) to manually keyframe dozens of bone angles every few frames across every clip. That fails in four predictable ways:
1. **Baked Minecraft Entity-Motion Bugs:** Macro forward translation (`root.z`) or fake hovering elevation (`root.y = +35u`) gets baked into animation files, fighting Minecraft's entity movement/physics controller and snapping back on loop boundaries.
2. **Biologically Impossible Joint Motion:** Unconstrained keyframes bend fingers, elbows, knees, or necks backward past physical limits, breaking believability immediately.
3. **Skating Feet & Uncoupled Appendages:** Manually keyed legs skate across the ground (`> 0.5u` slip), wings flap like rigid barn doors with zero downstroke/upstroke aerodynamics, and heads bob wildly instead of stabilizing gaze.
4. **Species Lock-In:** Hardcoded "dragon walk" or "humanoid run" scripts collapse the moment an artist sketches an original 8-legged crystal crab, a 4-winged two-headed seraph-hydra, or a legless sea-wyrm.

**The SyncCraft Unified `SO(3)` / `SE(3)` / `Sim(3)` Paradigm** mirrors modern production character rigs (Maya / Unreal Control Rig / Unity Rigging) directly inside Blockbench/Block-DSL:
1. **Define the creature's Skeleton & Biological Joint Harness (`SO(3)`):** Tag anatomical joints with `bio_joint` presets (or enable `auto_bio_limits: true` for geometry-first inference) so impossible hyperextension is structurally prevented by smooth $C^1$ elastic barriers.
2. **Define the creature's Muscles & Tendons (`@muscles` + `Sim(3)`):** Attach Zajac/Hill muscle actuators, biarticular `tendon_coupling`, and volume-preserving `scale_bulge` directly onto anatomical chains (ribcage diaphragm breathing, jaw-to-tongue coupling, pectoral/tail flares).
3. **Declare Intent, Let the Solver Resolve Physics (`SE(3)`):** Declare `gait`, `flight`, `muscle_drive`, `look_chain` (`head_stabilize`), and `spring` secondary dynamics. The unified solver computes collision-free, ground-locked, aerodynamically asymmetric, entity-ready animations automatically — cutting animation boilerplate by `~25–60%` while letting artists layer expressive keyframes cleanly on top!

---

## 2. Minecraft & Blockbench `in_place: true` Entity-Space Contract

In Minecraft mods, **the game engine's entity controller owns macro world displacement and heading rotation** (walking forward, turning, climbing altitude, flying at `Y = 85`). An animation clip operates strictly in **local entity space**:

| Motion Component | Owned By Minecraft Entity Controller? | Owned By `.bbmodel` / `.animation.json` Clip? | Audit Rule (`v26`) |
|---|---|---|---|
| Macro forward/lateral travel (`+Z`, `±X`) | **YES** (driven by `synccraft_root_motion.speed_blocks_per_s`) | **NO** (`in_place: true` strips macro `adv_z`) | `max\|root_z\| == 0.00u` on all locomotion & flight clips |
| Macro flight cruising altitude (`+30..+80u`) | **YES** (Minecraft flying AI / `Mob.setNoGravity`) | **NO** (never bake `+35u` fake hover elevation!) | `root.y` carries **only** authentic local COM recoil/heave (`±0.5..4.5u`) |
| Heading turns (`±180°` yaw navigation) | **YES** (Entity `yRot` / `yBodyRot`) | **NO** (only subtle anticipation/banking lean `±5–18°`) | Seamless periodic loop closure (`phase 0 == phase 1`) |
| Local COM heave, crouch, recoil, breathing, lift-bob | NO | **YES** (computed from gait weight / wingbeat thrust) | Anti-phase to wing downstroke; grounded stance slip `< 0.02u` |

**How SyncCraft Guarantees Zero Foot Skate:**
- Inside `evaluate_gait_clip(..., in_place=True)`, the solver computes stance lock in **exact virtual world space** (`virtual_root_delta` advancing at Alexander/Jayes Froude speed), solves closed-loop `SE(3)` world IK (`slip < 0.02u`), strips ONLY macro `adv_z` from exported root position keyframes, and writes matched entity speed into `synccraft_root_motion`:
  ```json
  "synccraft_root_motion": {
    "in_place": true,
    "speed_units_per_s": 15.04,
    "speed_blocks_per_s": 0.94,
    "velocity_vec": [0.0, 0.0, 15.04]
  }
  ```
- When the Minecraft mod moves the entity at `0.94 blocks/s`, every planted talon locks to the voxel ground with **zero skating**.

---

## 3. Universal Geometry-First Morphology & Capability Resolution

Before authoring clips on any new or imaginary creature, call:
```python
from synccraft.kinematics import analyze_morphology, analyze_motion_capabilities
caps = analyze_motion_capabilities(program)
```
`analyze_morphology` and `analyze_motion_capabilities` inspect the **compiled 3D rest geometry and kinematic tree** — never relying on English bone names:

| Creature Archetype (Real or Imaginary) | Discovered Topology | Automatic Solver Modes & Behaviors |
|---|---|---|
| **0-Leg Serpent / Eel / Fish / Leviathan** | `limbs = ()`, long `spine` + `tail` | `serpentine_wave` or `swim_undulate` traveling sinusoidal wave + `head_stabilize` + `in_place` velocity |
| **2-Leg Humanoid / Theropod / Mech** | `2` grounded hind limbs + `free_limbs` (arms) + `digits` | `biped_alternate` + **automatic anti-phase arm counter-swing & elbow flexion** + `finger_flex`/`thumb_saddle` protection (digits stopped cleanly before IK!) |
| **4-Leg Quadruped / Wyvern / Elder Dragon** | `4` grounded limbs (`fore_l/r`, `hind_l/r`) | `lateral_sequence` / `diagonal_pairs` / `rotary_gallop` / `transverse_gallop` + 4-foot `SE(3)` world IK (`< 0.02u` slip) |
| **6 / 8 / 10+ Leg Spider / Insect / Alien** *(even with zero English names: `core_0`, `stalk_a..d`)* | `>= 6` limbs ordered geometrically by anterior-to-posterior `Z` & bilateral `X` | `hexapod_tripod` or **`metachronal_wave`** + `auto_bio_limits: true` (`arthropod_coxa` + `arthropod_hinge`) |
| **2 / 4 / 6-Winged Flyer / Finned Swimmer** | Multi-pair `wings` sorted by `(pair_index, side)` with `finger_bones` & `locked_spar_bones` | `flight` (`hover`, `cruise`, `glide`, `soar`, `takeoff_burst`, `mantle`, `swim_fins`) + `pair_phase_deg` fore/aft wing phasing |
| **Multi-Headed Hydra / Multi-Tailed Kitsune / Kraken** | `neck_chains` (N heads), `tail_chains` (N tails), `tentacles` (N tendrils) | Independent `SO(3)` vestibulo-ocular gaze lock per head, phase-staggered waves on all tails, dual-axis sinusoidal waves on all tentacles |

---

## 4. `SO(3)` Biological Joint Constraints (`bio_joint` & `auto_bio_limits`)

### 4.1 Critical Relative-Delta Law
All `joint_limits` and `bio_joint` ranges in SyncCraft are **deltas relative to the bone's rest pose (`rest_euler`)**, requiring `min <= 0 <= max`.
- Example: If a digitigrade knee (`leg_l_shin`) already sits bent at `rotate: [44, 0, 0]` in bind pose, setting `rx: [-34, 48]` allows it to extend cleanly to `+10°` during stride push-off and flex deeply to `+92°` during swing clearance — while still making backward hyperextension (`< +10°`) mathematically impossible!
- When `auto_bio_limits: true` is active on a clip, `soft_clamp_euler_to_limits` applies a smooth $C^1$ `tanh` elastic barrier near anatomical limits (like real ligaments tightening smoothly instead of hard robotic snapping).

### 4.2 Complete `BIO_JOINT_PRESETS` Reference (26 Anatomical Presets)

| Family | `bio_joint` Preset | Anatomical Function & Protection |
|---|---|---|
| **Hands & Digits** | `finger_flex` | Pure 1-axis curl (`0°` backward hyperextension, zero lateral splay!) |
| | `finger_proximal` | Knuckle MCP joint: wide flexion, tiny abduction (`±12°`), near-zero hyperextension |
| | `thumb_saddle` | Opposable CMC thumb joint: 2-axis opposition + flexion |
| **Vertebrate Limbs** | `ball_shoulder` | 3-DOF glenohumeral socket with anatomical elevation/retraction cone |
| | `ball_hip` | 3-DOF acetabular hip socket (tighter abduction than shoulder) |
| | `hinge_elbow` | Single-axis hinge blocking backward hyperextension (`±6°` residual twist) |
| | `hinge_knee` | Single-axis knee hinge blocking backward hyperextension |
| | `hinge_hock` | Digitigrade ankle / tarsometatarsal spring hinge |
| | `wrist_ankle` | 2-DOF flexion/extension + slight roll/deviation |
| **Arthropods** | `arthropod_coxa` | Wide horizontal sweep (`±42°` yaw) + moderate lift (`-22..+34°` pitch) |
| | `arthropod_hinge` | Exoskeleton femur-tibia single-axis hinge |
| **Axial & Cranial** | `spine_vertebra` | Moderate coupled flex/lateral-bend/twist (`-18..+22°` pitch, `±18°` yaw) |
| | `serpent_vertebra` | Wide lateral undulation (`±34°` yaw) for snakes/eels/wyrms |
| | `neck_vertebra` | Supple multi-axis neck joint (`-32..+32°` pitch, `±28°` yaw) |
| | `head_atlas` | Fine pitch/yaw/roll gaze tracking (`±28°` pitch, `±35°` yaw) |
| | `jaw_hinge` | Pure opening hinge (`-2..+55°` pitch, `±2°` yaw/roll — never opens upward through maxilla!) |
| | `tail_vertebra` | Supple traveling-wave caudal vertebra (`±28°` pitch, `±34°` yaw) |
| | `tentacle_segment` | Omnidirectional muscular hydrostat (`±42°` pitch/yaw) |
| | `ear_pinna` | Expressive ear / collar / frill swivel |
| **Wings & Fins** | `wing_shoulder` | High-range pectoral flight socket (`-45..+45°` pitch, `-68..+75°` roll) |
| | `wing_elbow` | Wing folding hinge (`-85..+18°` flexion, pronation twist) |
| | `wing_wrist` | Carpal spar splay & recovery fold |
| | `wing_finger` | Distal phalanx aerodynamic camber flex |
| | **`wing_anchor_locked`** | **Rigid `[0, 0, 0]` lock (`±0°`)** — guarantees innermost wing finger (`wing_l_f4`) stays coplanar with `wing_l_fore` & `wing_bay_4` (`v25` Anti-Split Wing Lock!) |
| | `fin_root` / `fin_ray` | Pectoral/dorsal aquatic fin sculling & soft trailing ray lag |
| **Vehicles & Mechs (`v28`)** | `wheel_axle` | Continuous `±360°` rolling pitch spin (`0°` camber/toe wobble) for truck/tank/landing-gear wheels |
| | `steered_wheel_knuckle` | Ackermann front steering yaw (`±35°` Y) with locked X/Z |
| | `rotor_propeller` | Continuous `±360°` axial spin for helicopter rotors, airship propellers & turbines |
| | `turret_yaw` | Full `±180°` horizontal turret/crane rotation (`0°` pitch/roll tilt) |
| | `cannon_elevation` | Pure vertical barrel elevation (`-12°..+45°` X, `0°` yaw/roll) |
| | `hydraulic_hinge` | Heavy 1-DOF industrial loader/mecha actuator (`-65°..+65°` X) |
| | `hatch_door` | Clean 1-DOF cargo ramp / tailgate / cockpit canopy / gullwing door hinge (`0°..+110°`) |
| | `gimbal_thruster` | 2-DOF vectored thruster nozzle (`±28°` X/Z) |

If a bone has NO explicit `joint_limits` and NO `bio_joint`, setting `auto_bio_limits: true` runs `infer_universal_bio_constraints(program)` which inspects child geometry, rest bend direction, and bone hierarchy to infer biologically valid `SO(3)` limits automatically — even on arbitrary non-English rigs!

---

## 5. Closed-Loop `SE(3)` Grounded Locomotion (`gait`) & IK Rules

### 5.1 Supported Universal Gait Patterns
`lateral_sequence` (quadruped walk/prowl) · `diagonal_pairs` (trot) · `rotary_gallop` · `transverse_gallop` · `biped_alternate` (humanoids/theropods + auto free-arm counter-swing) · `sprawl_diagonal` (lizards/crocodilians) · `hexapod_tripod` (6-legged insects) · **`metachronal_wave`** (8+ legged arachnids/centipedes/aliens) · **`serpentine_wave`** (legless serpents) · **`swim_undulate`** (aquatic fish/sea-wyrms).

### 5.2 Mastering 2-Bone & 3-Bone Limb IK Chains (Zero Slip, Zero Flip, Zero Chest Clip)
When defining `@kinematics` `ik_chain` blocks for complex sculpted rigs (especially broad-chested quadrupeds, bosses, or humanoids):
1. **End the IK Chain ABOVE the Ground-Contacting Paw/Foot Pad:**
   - Forelimb (2-bone): `bones: [arm_l_upper, arm_l_fore]`, `effector: paw_sole_l` (locator on `paw_l` at `y = 0`).
   - Digitigrade Hindlimb (3-bone): `bones: [leg_l_thigh, leg_l_shin, leg_l_ankle]`, `effector: foot_sole_l` (locator on `foot_l` at `y = 0`).
   - *Why:* Including a flat ground paw/foot bone inside a damped least-squares IK chain invites branch-flipping (`70°` paw jumps) and elbow-into-ribcage clipping. Leaving `paw_l`/`foot_l` outside the IK chain keeps paws flat, clean, and `0.000002u` slip!
2. **Set Anatomically Outward Pole Vectors:**
   - Place the `pole` laterally outward on the limb's side (`-X` for left, `+X` for right) AND along the natural hinge bend direction (posterior-outward for mammal/dragon elbows, anterior-outward for knees).
   - Example on a broad dragon chest (`chest_width = 16u`): `fore_l` pole `[-30, 20, 12]` (keeps elbows sweeping cleanly outside the ribcage); `hind_l` pole `[-32, 22, 24]`.
3. **Automatic Reach-Safe Lift & `Sim(3)` Scale Compensation:**
   - Erect straight-legged rigs (`rest_ratio >= 0.90`) automatically crouch slightly (`crouch_y < 0`) so stride endpoints stay inside spherical reach.
   - Coiled broad-chested quadrupeds (`rest_ratio <= 0.89`, non-erect) automatically apply a gentle reach-capped athletic lift (`<= 0.91 × reach` headroom) so the chest clears moving forearms cleanly while guaranteeing `max_reach_ratio < 0.96`.
   - Whenever `@muscles` apply `Sim(3)` `scale_bulge` to a trunk bone (`chest` breathing), `pose_scale_factors` is forwarded directly into `solve_n_bone_ik` so ground-locked limbs compensate automatically for ribcage expansion!

---

## 6. Universal Aerodynamic & Hydrodynamic Airfoil Solver (`flight`)

Hand-keyframing convincing dragon, bird, insect, or manta-ray flight is notoriously error-prone: straight-arm wing flaps look like toy hinges, high-mounted bind-pose wings clap together dorsally above the spine, and baked `+35u` height breaks Minecraft flight controllers.

Declare `flight <name> { ... }` directly inside any `clip` (synced verbatim to `synccraft_v20/src/synccraft/dsl/parser.py`):

```blockdsl
clip hover_vampire_lord {
  duration_s: 2.4, loop: true, in_place: true, auto_rig: true, digit_grasp: 0.72, antagonist_bulge: 0.76
  flight aero_hover {
    mode: hover,
    flap_hz: 0.8333,          // integer cycles in duration_s (2 cycles / 2.4s) for exact loop!
    stroke_deg: 24,           // angular wing stroke amplitude (auto-clamped against dorsal wing-clap!)
    sweep_deg: 11,            // fore-aft Figure-8 wingtip stroke sweep
    pronation_deg: 12,        // aerodynamic pronation (flat downstroke) / supination (edge-on upstroke)
    elbow_fold_deg: 16,       // authentic recovery upstroke wing-fold to shed drag
    wrist_lag_deg: 24,        // distal phase lag (carpus trails humerus)
    finger_fan_deg: 6,        // trailing-edge camber whip on phalanges (wing_anchor_locked stays [0,0,0]!)
    lift_recoil_u: 3.0,       // anti-phase in-place COM lift recoil (±3.0u, NEVER fake +35u altitude!)
    body_pitch_deg: 14,       // poised hover torso tilt + counter-pitching spine wave
    head_stabilize: 0.85,
    tail_counter_deg: 14,
    limb_tuck: 0.45,          // Jacobian-abducted aerial leg tuck (stays clear of torso & wings!)
    pair_phase_deg: 0,        // multi-pair wing phase offset (e.g. 90° for 4-winged dragonfly/seraph!)
    pectoral_bulge: 0.85,
    ground_clearance: true,
    in_place: true
  }
  muscle_drive thorax_diaphragm: activation(peak=0.72, freq_hz=0.8333, phase_deg=15, baseline=0.14)
}
```

### Flight Modes & Physics Guarantees
| Mode | Use Case | Aerodynamic Behavior |
|---|---|---|
| `hover` | Stationary aerial boss hover / hummingbird / seraph | Steep poised body pitch, deep Figure-8 sweep, strong anti-phase lift recoil (`±2–4.5u`), Jacobian-tucked limbs |
| `cruise` | Forward level flight (`in_place: true` + matched `synccraft_root_motion` speed!) | Streamlined horizontal torso, forceful propulsive downstroke + sleek folded recovery, zero Z-drift |
| `glide` / `soar` | Thermal riding / stooped dive / banking | Broadly cambered low-frequency dihedral wings, subtle thermal micro-corrections, high cruising speed metadata |
| `takeoff_burst` | Explosive vault from perch into air | Anticipation squat -> explosive power beat -> rapid climb recoil -> tucked limbs |
| **`mantle`** | **Grounded threat wing-flare (eagles, dragons, vampires)** | **Keeps all grounded feet locked to `y = 0` via closed-loop `SE(3)` world IK (`0.000u` slip!)** while violently unfurling and beating wings overhead! |
| `swim_fins` | Underwater wing-flying (manta rays, sea turtles, plesiosaurs) | Smooth sinusoidal hydrodynamic sculling + gentle pitch undulation |
| **`jet_hover`** | **VTOL jet-propelled hover (jet dragons, VTOL mecha, Harrier-style flyers)** | **ZERO wing flapping / paddling!** Holds explicit downward-vectored pylon/nacelle poses while synthesizing buoyant VTOL vertical cushion + `3×` exhaust tremor, independent phased multi-nozzle `gimbal_deg` trim across all non-locked thruster blades (`bio_joint: gimbal_thruster`), `0.0000u` `in_place` Z-drift, and zero flapping pectoral recruitment |
| **`jet_cruise`** | **Supersonic / fighter-jet forward flight (`in_place: true` + matched supersonic `root_motion`)** | **ZERO wing flapping!** Swept-back clamped thruster blades with subtle `gimbal_deg` shock-tremor, smooth aerodynamic fighter-jet banking roll (`bank_deg` on root `rz` + tail counter-rudder), retracted landing-gear limb tuck, and body-local bilateral symmetry audit |
| **`jet_boost`** | **Impulsive VTOL mid-air rocket-hop / afterburner dash burst** | **ZERO wing flapping!** Sharp vertical/vectored rocket-impulse recoil + `3×` thruster shudder across all independent engine nozzles while honoring explicit vectoring keyframes |

### Built-In Safety & Symmetry Invariants
1. **3D Angular Jacobian Axis Calibration:** Automatically discovers which local bone axis moves each wing vertically (flap), fore-aft (sweep), and twists its chord (feather) — works on arbitrary bone orientations and arbitrary non-English rigs!
2. **Exact Body-Local Bilateral Symmetry (`asym = 0.00000`):** Mirrors left/right wing strokes and multi-nozzle jet gimbals cleanly around the sagittal plane and audits tip symmetry in the fuselage's body-local frame (so even banked `jet_cruise` turns with `bank_deg != 0` verify exact intrinsic wing symmetry!).
3. **Zero `in_place` Horizontal Drift (`surge_z == 0`):** Whenever `in_place: true` (or in any `jet_*` mode), `flight_solver.py` guarantees `max|root_z| == 0.0000u`.
4. **Bind-Pose Dorsal Elevation Anti-Clap:** Measures each wing's rest elevation (`rest_elev_deg`) and automatically limits upward lift (`max_up_lift = max(4°, 56° - rest_elev)`) so high-mounted dihedral dragon wings never collide above the spine.
5. **Rigid Inner-Wing Membrane Lock (`wing_anchor_locked`) vs. Independent Thrusters (`gimbal_thruster`):** Automatically locks `wing_l_f4` / `wing_r_f4` (or any bone tagged `bio_joint: wing_anchor_locked`) at `[0, 0, 0]` to seal bat/dragon wing membranes (`v25` Anti-Split Law), while leaving non-membrane multi-nozzle jet engines (`bio_joint: gimbal_thruster`) 100% free to gimbal independently!
6. **Zero-Offset `mirror_x` Cladding Auto-Exclusion (`morphology.py`):** Bilateral sculpted cladding helper bones mounted at `X = 0` with `mirror_x` are automatically excluded from axial `spine`/`neck`/`tail` chains so procedural waves and `head_stabilize` never twist bilateral armor cheeks or dental arches.

---

## 6B. Universal Underwater Buoyancy & Musculoskeletal Swimming Positioning System (`swim`, NEW `v38`)

Aquatic and amphibious creatures (Leviathans like *Lagiacrus*, crocodiles, sharks, eels, sea serpents, marine iguanas, cephalopods, and pelagic jellyfish like *Wintermoon Nettle*) MUST NEVER reuse ground-walking `gait` stances underwater (which leave limbs dangling straight down as if walking on dry land!). Use SyncCraft's first-class declarative **`swim <name> { ... }`** solver (`synccraft/kinematics/swim_solver.py`):

```blockdsl
clip swim_abyssal_cruise {
  duration_s: 2.0, loop: true, in_place: true, auto_rig: true, auto_bio_limits: true
  swim sea_lord_cruise {
    mode: leviathan_cruise,     // leviathan_cruise | leviathan_hover | carangiform | subcarangiform | anguilliform | medusa_pulse
    freq_hz: 0.5,               // integer cycles in duration_s for exact loop wrap
    wave_amp_deg: 16.0,         // progressive traveling S-wave amplitude (head -> torso -> paddle-tail)
    wavelength_ratio: 0.95,     // body wavelengths from skull to tail tip (true traveling wave, NOT standing wag!)
    buoyancy_y: 12.0,           // mid-water neutral buoyancy elevation (+u above seabed)
    heave_amp_u: 1.8,           // gentle sinusoidal buoyancy bob
    body_pitch_deg: -4.0,       // hydrodynamic trim (-pitch ascending cruise, +pitch ambush dive)
    roll_bank_deg: 3.5,         // subtle stroke-coupled hydrodynamic roll banking
    head_stabilize: 0.80,       // inertial vestibulo-ocular lock keeping predator eyes locked dead-ahead
    limb_mode: hydro_tuck,      // hydro_tuck (streamlined backward hull sweep!) | paddle_row (breaststroke) | none
    limb_tuck: 0.85,            // Jacobian-calibrated aft sweep + flank/tail clearance
    myomere_bulge: 0.85,        // active Sim(3) lateral myomere muscle bulge synced to stroke curvature!
    fin_lag_deg: 14.0,          // phase-lagged trailing hydro-feathering on webbed paws, digits & fins
    digit_articulation: 0.85,
    ground_clearance: true,     // closed-loop seabed clearance guard (min_solid_y >= 2.0u)
    in_place: true
  }
  muscle_drive thorax_diaphragm: activation(peak=0.65, freq_hz=0.5, baseline=0.12)
}
```

### Underwater Swimming Modes & Musculoskeletal Guarantees
- **`leviathan_cruise` + `limb_mode: hydro_tuck`:** Neutral-buoyant horizontal cruise where 3D angular Jacobians sweep forelimbs & hindlimbs cleanly **backward along the hull** (with lateral hind-thigh abduction clearing the sweeping `tail_1` root!) and feather webbed digits straight aft, driven by a progressive `SO(3)` traveling S-wave projected onto every bone's world-Y yaw axis + active `Sim(3)` lateral myomere muscle contraction (`14+` axial muscle bones bulging rhythmically with stroke thrust!).
- **`leviathan_hover` + `limb_mode: paddle_row`:** Mid-water ambush treading with gentle anti-phase breaststroke paddling, deeper buoyancy bob, and elevated cobra-hood alert trim.
- **`carangiform` / `subcarangiform` / `anguilliform`:** Predatory fish, sharks, and sea serpents (`limb_mode: none`), scaling the amplitude envelope $A(s)$ cleanly from rigid anterior hull (`carangiform`) to full-body eel wave (`anguilliform`).
- **`medusa_pulse`:** Radial pelagic sky/sea jellyfish — rhythmic `Sim(3)` bell contraction/recovery + Jacobian-projected radial tentacle & oral-arm propulsion!
- **Quick-Reference BlockDSL Animation Grammar Guard (`v38`):** `root` bone (`pelvis_root` / `thorax_root` / `bell_root`) uses `key <root>.position` / `key <root>.rotation` (never `track`); child bones use `key` or `track` (`law ∈ {wave, flap, hinge, impulse, spin}`); `audit_behavior_clip` returns `total_collision_events`, `swept_collision_frame_count`, `colliding_frames`, plus `swim_audit` / `flight_audit` / `gait_audit`; `export_bedrock` returns a `dict` (`{"geometry", "animations", ...}`).

---

## 7. Unified `Sim(3)` Muscle Bulge, Gaze Lock & Secondary Springs

All four biomechanical layers compose automatically inside a single clip evaluation (`gait`, `flight`, or expressive keyframe clips):

```blockdsl
@muscles {
    muscle thorax_diaphragm {
        bones: [waist, chest],
        axis: x,
        action: [0.35, 1.0],
        max_torque_deg: 5.5,
        optimal_length: 1.0,
        passive_stiffness: 0.45,
        scale_bulge: 0.032       // Sim(3) volume-preserving ribcage swell exported as .bbmodel scale channel!
    }
    muscle mandible_depressor {
        bones: [jaw],
        axis: x,
        action: [1.0],
        max_torque_deg: 36.0,
        tendon_coupling: [tongue_base, x, 0.38]  // opening jaw automatically drives tongue!
    }
}

@kinematics {
    look_chain predator_gaze { bones: [neck_1, neck_2, neck_3, head], weights: [0.16, 0.24, 0.28, 0.32] }
    spring tail_4 { stiffness: 28, damping: 7.5, mass: 1.0, axis: y, max_deg: 18 }
    spring ear_collar_l { stiffness: 36, damping: 8.5, mass: 0.5, axis: z, max_deg: 12 }
}
```

1. **Zajac/Hill Muscles + Biarticular Tendons:** `muscle_drive` excites muscles smoothly; passive elastic stiffness returns joints naturally to rest; `tendon_coupling` coordinates coupled anatomy (jaw ↔ tongue, forelimb ↔ digit curl, chest ↔ shoulder splay) without extra keyframes.
2. **`Sim(3)` Volume-Preserving `scale_bulge`:** When a muscle activates at `a ∈ [0, 1]`, target bones expand radially by `1 + scale_bulge · a` and contract slightly axially by `(1 + scale_bulge · a)^(-1/2)` (exact volume conservation `det(S) ≈ 1`), exported directly into `.bbmodel` `scale` keyframes AND Bedrock/GeckoLib `.animation.json` `scale` channels!
3. **Exact `SO(3)` Vestibulo-Ocular Gaze Lock (`head_stabilize: 0.0..1.0`):** Computes the accumulated world rotation of the neck base at every frame and distributes the exact inverse `SO(3)` counter-rotation across `look_chain` (or all discovered `neck_chains` for multi-headed hydras!) so eyes stay hawk-steady during prowling, charging, hovering, and cruising.
4. **Implicit-Euler Secondary `spring` Follow-Through:** Integrates damped harmonic lag across ears, collars, tail tips, manes, and streamers with exact periodic wrap-around closure on looping clips.

---

## 7B. `v27` Industry-Grade Full-Body Biomechanical Expansion (Unreal Control Rig / Maya / Houdini KineFX Parity)

Version `v27` expands SyncCraft's solver (`synccraft/kinematics/whole_body_solver.py`) from basic limb/wing motion to **100% full-body biomechanical coverage** — so artists and small AI models can animate ANY real or imaginary creature from torso to fingertips without manual keyframe spam:

1. **Universal Skeleton-to-Rig Auto-Assembly (`auto_rig: true` -> `auto_assemble_biomechanical_rig`):**
   - Inspects compiled 3D rest geometry & bone topology (zero dependency on English bone names — verified on arbitrary German/synthetic names like `zentral_0`, `stelze_a..c`, `fluegel_l`, `hals_1..2`, `schweif_1..2`!) to automatically assemble:
     - `spine_chain`, `neck_assemblies` (head, jaw, tongue, throat), `tail_chains`
     - `wing_assemblies` (pectoral chest host, shoulder, forearm, wrist, fingers, and rigid `locked_spar_bones`)
     - `limb_assemblies` (upper, lower, ankle/wrist, `foot_hand_bone`, multi-phalanx `digit_chains`, and `thumb_chains`)
     - `auto_muscle_groups` & complete `SO(3)` `bio_constraints`!
2. **Serpentine N-Segment Neck & Spine Spline-IK (`neck_aim: [x, y, z]`, `spine_coil: deg`):**
   - `solve_spine_neck_spline_ik(...)` distributes smooth 3D world-target aiming and S-curve predator coiling across arbitrary 1-to-8+ segment necks and spines while respecting per-vertebra `SO(3)` biological limits.
3. **3D Volume-Weighted COM Balance & Angular-Momentum Counterweight (`com_balance: 0.0..1.0`):**
   - `solve_com_balance_and_counterweight(...)` evaluates the creature's instantaneous 3D volume-weighted Center of Mass across all posed boxes, normalizes lateral/fore-aft offset strictly against **core stance / support-polygon width** (NEVER giant outstretched wingspan!), and automatically drives **angular-momentum tail counterbalance** + subtle grounded `pelvis_root` X stabilization.
4. **Reverse-Foot Rocker (`foot_roll: 0.0..1.0`) & Universal Multi-Phalanx Digit Articulation (`digit_articulation`, `digit_grasp`):**
   - `solve_foot_roll_and_digit_articulation(...)` drives realistic **Reverse-Foot Rocker** phases across `heel-strike -> flat plant -> toe-off -> swing`, paired with:
     - **Stance digit splay** when grounded, **aerial/swing talon curl** when airborne, and **coordinated flexor fist/talon grasp + opposable thumb opposition** (`digit_grasp: 0..1`) during roaring/grappling/perching — strictly clamped by `finger_proximal`, `finger_flex`, and `thumb_saddle` so **digits can NEVER hyperextend backward!**
5. **Full Pectoral-to-Phalanx Wing Anatomy (`wing_fold`, `pectoral_bulge`, `ground_clearance`):**
   - `solve_wing_full_anatomy(...)` couples continuous grounded/perched `wing_fold ∈ [0, 1]`, automatic **wingtip ground-clearance guard** (`y_tip >= 0.5u`), **pectoral + wing-brachial `Sim(3)` downstroke muscle bulge**, and strict `wing_anchor_locked` (`[0, 0, 0]`) preservation on `wing_l_f4` / `wing_r_f4`.
6. **Whole-Body `Sim(3)` Antagonist Muscle Bulge (`antagonist_bulge: 0.0..1.0`):**
   - `solve_full_body_antagonist_muscle_bulge(...)` computes angular-flexion & muscle-drive radial volume swell (`det(S) == 1.0` exact volume conservation!) across **chest/pectorals, neck/throat sac, wing arms, upper/lower limbs, and tail base** — with **`preserve_grounded_limb_length=True`** keeping axial scale `== 1.0` on grounded IK bones so stance slip stays `< 0.001u` while visibly flexing biceps, thighs, pectorals, and throats in `.bbmodel` & Bedrock `.animation.json` `scale` channels!
7. **Closed-Loop Standing Limb IK Pinning (`ground_pin_limbs: true | false | [hind_l, hind_r]`, `v28`):**
   - **The Raw FK Floor-Dip Bug It Solves:** On any non-`gait` standing clip (`idle`, `bite`, `wing_cloak_sweep`, `tail_whip`, or bipedal `roar`), keyframing or breathing on `pelvis_root`, `waist`, or `chest` moves the shoulder/hip sockets in world space; without closed-loop limb IK baked into the exported keyframes, raw Blockbench `.bbmodel` FK plunges the front or hind paws **`-4..-6.5u` underground** (failing `verify_bbmodel_ground.py`)!
   - **Exact `SE(3) + SO(3)` Solution (`solve_standing_limb_ik_pinning`):**
     - Declare `ground_pin_limbs: true` (pins all grounded limbs) OR selective `ground_pin_limbs: [hind_l, hind_r]` (pins hind feet rock-solid at `y = 0.00u` while freeing the forelimbs to rear high into the sky during a bipedal boss roar!).
     - Splits `ik_bones = lb.chain[:-1]` above `paw_l`/`foot_l` whenever `lb.chain[-1] == la.foot_hand_bone` (preventing 3-link heel tilt!), solves damped-least-squares `SE(3)` world IK locking the wrist/ankle pivot at its exact rest world coordinate (`< 0.02u` slip), AND applies the exact closed-form `SO(3)` world-flat sole lock:
       $$R_{\text{delta}} = R_{\text{foot,rest,local}}^{\top}\,\bigl(R_{\text{parent,posed}}^{\top}\,R_{\text{foot,rest,world}}\bigr)$$
       both before AND after secondary spring integration — baking `100%` floor-locked FK rotations directly into `.bbmodel` & `.animation.json`!
8. **Universal Rigid Mechanical Subsystem Discovery & `Sim(3)` Exemption (`v28`):**
   - `auto_assemble_biomechanical_rig` automatically populates `MechanicalSubsystem` (`wheels`, `steering_knuckles`, `rotors`, `turrets`, `cannons`, `hydraulics`, `hatches`, `thrusters`), classifies `wheeled_vehicle` capabilities (excluding rigid wheel chains from biological leg counts even across `mirror_x` prefixes like `steer_fr__wheel_fl`), and exempts every `is_rigid_mechanical_bone` from `Sim(3)` muscle bulge so steel wheels, cannons, and mecha pistons never squish!
9. **`v29` Strict `SO(3)` / `SE(3)` / `Sim(3)` Lie-Group Animation & Exporter Invariants (Zero Foot Twist, Zero Floor Plunge, Supple S-Neck & Full-Span Wing Projection):**
   - **`SE(3)` Wrist/Ankle Split + World-Sagittal `SO(3)` Reverse-Foot Rocker (`gait_solver.py`):** Never pass `paw_l`/`foot_l` (`la.foot_hand_bone`) as a 3rd link inside CCD/DLS limb IK (which twists paws `±25°` in roll/yaw and plunges toes `-7u` underground!). Split `ik_bones = lb.chain[:-1]` at the wrist/ankle pivot (`0.000u` stance slip + hard `y >= ground_y` swing barrier) and solve `paw_l`/`foot_l` strictly in world-sagittal `SO(3)`:
     $$R_{\text{foot,world}}(t) = R_{x,\text{world}}\bigl(\theta_{\text{rocker}}(t)\bigr)\,R_{\text{foot,rest,world}}$$
     giving `0.00°` roll/yaw foot wobble across `heel-strike -> flat plant -> toe-off -> swing`!
   - **Per-Bone `so(3) \to SO(3)` Exponential-Map Aerodynamic Wing Projection (`flight_solver.py`):** Steeply pitched cathedral dragon wings (`-61°` world pitch on `chest + wing_l_upper`) suffer severe roll/yaw cross-talk under naive single-axis Euler indexing. Construct the desired world angular velocity vector $\boldsymbol{\omega}_{\text{world}} \in \mathfrak{so}(3)$ (vertical stroke + fore-aft sweep + chord pronation/supination), map to the bone's local frame $\boldsymbol{\omega}_{\text{local}} = R_{\text{bone,rest}}^{\top}\,\boldsymbol{\omega}_{\text{world}}$, and integrate via Rodrigues' exact `SO(3)` exponential map $\exp([\boldsymbol{\omega}_{\text{local}}]_\times)$ across humerus, radius/ulna (`elbow_fold_deg`), carpus (`wrist_lag_deg`), and outer phalanges (`wing_l_f4` locked at `[0,0,0]`).
   - **Jacobian-Verified Upward `SO(3)` Aerial Limb Tuck + `SE(3)` Airborne Ground Barrier:** Verify $\bigl(\boldsymbol{\omega}_x \times \mathbf{r}_{\text{limb}}\bigr)_y \ge 0$ before tucking limbs in flight so forelimbs fold **upward** against the sternum (never downward `-9.8u` through the floor!), backed by a closed-loop `SE(3)` IK clearance guard (`min_y >= ground_y + 1.5u`).
   - **Supple Phase-Delayed `SO(3)` Serpentine Neck Wave vs. Distal-Atlas-Only Gaze Stabilization:** On `4+`-segment dragon/leviathan necks (`neck_1..neck_5 -> head`), propagate a traveling `SO(3)` cervical wave ($\Delta\phi \approx 42^\circ$/segment) across `neck_1..neck_N` and apply `head_stabilize` (`<= 0.35`) **only** to the distal `2` atlas/skull joints — keeping the neck vibrant and fluid instead of locked like a rod!
   - **Baked Solver Trajectory Precedence in Exporters (`bbmodel5_exporter.py` & `bedrock_gecko_exporter.py`):** Whenever a clip contains procedural `SO(3)/SE(3)/Sim(3)` solvers (`gait`, `flight`, `ground_pin_limbs`, `muscle_drive`, `spring`), **always export the evaluated multi-frame `SO(3)/SE(3)/Sim(3)` trajectory** even if a bone also has an additive `key` track (which `evaluate_behavior_clip` has already composed and IK/SAT-corrected!), ensuring `verify_bbmodel_ground.py` (using Blockbench 5's exact additive `R_xyz(rest_r + drot)` + `Sim(3)` `scale`) matches internal solver physics to `< 0.01u`!
   - **Solid-Cuboid `ground_y` Datum Law (`morphology.py`):** Always compute `morph.ground_y` exclusively across **solid 3D cuboids (`plane_normal is None`)** — never include zero-thickness alpha-cutout planes (`wing_bay_*`, capes, fins), whose transparent bounding-box corners on steeply pitched cathedral wings can dip `-7.2u` below `y = 0` and drag every `gait`, `mantle`, and `flight` ground barrier underground!
10. **`v31` 3-of-4 Selective Standing Limb Pinning, Explicit 3D Sprawled `ik_chain`s & Symmetric Muscle Law (*Zinogre* RSI Distillation):**
   - **3-Foot-Pinned Single-Forelimb Overhead Slams (`ground_pin_limbs: [fore_r, hind_l, hind_r]` / `[fore_l, hind_l, hind_r]`):** For asymmetric quadruped paw/claw slams, pin the **3 weight-bearing stance limbs** (`< 0.001u` slip!) while leaving the striking forelimb free to wind up high & wide overhead (`delta_rx = -64°..-72°` on a `+60°` rest humerus) and crash flat onto `y = 0.00u` on impact!
   - **Declare Clean `plane: sagittal` `ik_chain`s (NEVER `plane: sprawled` or Sideways Frog-Poles on Parasagittal Limbs!):** For upright/parasagittal quadrupeds, wolves, dinosaurs, birds, and dragons, declare `@kinematics { ik_chain hind_l { bones: [...], effector: [...], plane: sagittal } }` (reserving `plane: sprawled` strictly for genuinely lateral-femur spiders, lizards, crabs, and crocodilians). `solve_n_bone_ik` in `"sagittal"` mode solves full 3D `(X, Y, Z)` target tracking via the 3-DOF hip/shoulder ball-socket (`0.0000u` slip) while keeping knees/hocks strictly vertical-sagittal (`< 1°` off-axis twist).
   - **Haunch-Brace Crouch (`pelvis_root.y < 0`) on Skyward Howls:** When pitching `pelvis + waist + chest` upward (`-20°..-26°`) during a 4-foot-pinned skyward howl, never simultaneously lift `pelvis_root.position.y > 0` (which hoists the front shoulder sockets beyond `100%` spherical forelimb extension!). Crouch `pelvis_root.position.y` slightly (`-1.0..-1.6u`) like a bracing wolf so all 4 paws stay comfortably inside reach (`0.0000u` slip).
   - **Never Attach Single-Sided Limb `@muscles` (`insertion_bone: arm_l_upper`) Alongside `auto_rig: true`:** An asymmetric `+X` moment-arm muscle on `arm_l_upper` adds `+35°..+45°` backward pitch only to the left humerus, driving the left forearm backward into the waist/knee! Reserve `@muscles` for axial/cranial muscles (`thorax_diaphragm`, `neck_sac`, `mandible_depressor`, `tail_lash`) and let `auto_rig: true` + `antagonist_bulge` generate symmetric `Sim(3)` limb bulge on all 4 limbs automatically.
11. **`v31–v32` Universal Acute-Fold `SAT`, Multi-Joint Bind-Pose Fold-Preserving N-Bone IK & Anatomical 1-DOF Hinge Purity (Creatures, Pillar Quadrupeds & Folding Mechanical Linkages):**
    - **Continuous Acute-Hinge SAT Ceiling (`MAX_SCULPTED_HINGE_ANGLE_DEG = 160.0` in `sync_audit.py`):** Prevents abrupt `6.0u -> 0.0u` hinge-tolerance dropoffs whenever deeply folded digitigrade/unguligrade Z-fold knees (`125°–145°`), folding bird/bat wings, arthropod legs, or `140°` excavator/crane hydraulic booms flex past `135°`.
    - **Static vs. Dynamic Coupled-Joint Bridge Separation (`is_dynamic_coupled_joint` in `_evaluate_sculpted_overlap`):** Static rest-pose audits (`enforce_physical_contact=True`) continue to strictly reject buried/swallowed redundant cubes (`buried_frac > cap`), while dynamic animation frames (`enforce_physical_contact=False`) allow explicitly coupled joint bridges (`sculpt_group`, `blend_with`, or parent-child joint pairs on the same side/axial chain) to tuck transiently into the joint pocket at peak flexion — keeping `coplanar_z_fighting` and **100% of non-joint inter-limb, limb-to-torso, and left-to-right collisions strictly rejected**.
    - **Rest-Pose-Aware `SO(3)` Hinge Limit Inference & 1-DOF Sagittal Purity (`bio_constraints.py`):** `_infer_hinge_limit_from_rest_euler` computes straightening headroom all the way back to straight (`0°` world angle), caps cumulative flexion at `158°` (applied to both mid-limb knees/elbows and distal deeply-folded hock/metatarsus joints with `|rest_rx| > 45°`), AND restricts off-axis knee/elbow/hock yaw & roll to **`±8°`** so hinge joints never bow sideways like frog legs, while `hydraulic_hinge` supports `±145°` rigid mechanical folds (`0%` `Sim(3)` squish).
    - **Multi-Joint Bind-Pose Fold-Preserving N-Bone IK (`solve_n_bone_ik` in `ik_motion_templates.py` & `morphology.py`):** On any `N >= 3`-bone authored fold (Z-fold legs, S-fold arms, zigzag arthropod limbs), averaging opposing interior joints (`0.5 * (knee + hock)`) cancels out the fold and injects a massive phantom pole cost that violently twists thighs/shins outward (`-34°` frog-leg kick even at idle!). `solve_n_bone_ik` now transports **every interior joint independently** from its bind-pose offset via minimal `SO(3)` chord swing + slack ratio scaling (`p_i_ideal == p_i_0` identically at rest -> `0.000°` idle drift!), enforces strong sagittal off-axis regularization on interior hinges, removes the bogus `+0.14 * total_len` lateral flare in `"sagittal"` mode, and `morphology.py` classifies `"sprawled"` only when the proximal segment genuinely projects laterally sideways (`prox_lat > 0.65 * prox_sag`).
    - **Uniform `SE(3)` Wrist/Ankle Split Across Explicit `@kinematics ik_chain`s (`gait_solver.py`):** `foot_split_info` now applies equally to auto-discovered limbs AND explicit `@kinematics ik_chain`s (whether terminating at the foot bone or at the forearm/ankle bone directly above `foot_hand_bone`), preventing explicit `ik_chain` wrist/ankle pivots from ever being driven down to `y = 0` (`-5u` sole sink) while preserving full `SO(3)` reverse-foot rocker and hard `y >= ground_y` swing clearance.
12. **`v35` Mandatory Foot, Multi-Phalanx Toe Animation & Standing-IK Verification Law (Zero Wooden Ski-Paddle Paws!):**
    - **Origin Bug (*Malzeno v34*):** Inheriting legacy manual-keyframe rigs with **zero `bio_joint` declarations on wrists/ankles/digits** and omitting `auto_rig: true`, `foot_roll`, `digit_articulation`, `digit_grasp`, and `ground_pin_limbs` produced **`0/36` moving foot/toe bones** across every clip — leaving rigid flat paws hovering `+1.4u` on claw tips at rest, tilting `45°` like wooden ski paddles during aerial hover, plunging `-8.0u` underground in tail attacks, and floating `+6.7u` in perched poses!
    - **5-Point Mandatory Foot & Toe Rigging Contract (Every Articulated Creature/Boss):**
      1. **Analytical `SO(3)` Flat-Sole Rest Leveling:** Solve exact `paw_l`/`foot_l` `rest_euler = recover_euler(R_parent.T @ R_y(yaw))` so BOTH `paw_pad`/`foot_pad` rest `100%` level at `y = 0.00u` (`ptp <= 0.05u`), and arch proximal toes (`-8°..-12°`) into hooked distal talons (`+20°..+30°`) so sole pads and claws grip `y = 0.00u` together.
      2. **Complete `bio_joint` Coverage on ALL Distal Foot/Hand & Phalanx Bones:** Tag every wrist/ankle (`bio_joint: wrist_ankle`), thumb/dewclaw (`bio_joint: thumb_saddle`), proximal finger/toe (`bio_joint: finger_proximal`), and distal claw tip (`bio_joint: finger_flex`).
      3. **Enable Full Solver Control Per Clip, Guard Unpinned `digit_grasp` on Grounded Clips, & Strip Stale Legacy `pole:` Hacks:** Turn on `auto_rig: true` across ALL clips; enable `foot_roll: 0.82–0.90` + `digit_articulation: 0.90–0.94` on every `gait` clip; enable `digit_grasp: 0.68–0.82` on **airborne `flight`/`hover`** clips, while keeping **`digit_grasp: 0.08–0.15`** on grounded `roar`/`slam` clips whose free paws start/end on `y = 0.0u` (preventing high constant `digit_grasp` from plunging hooked talons `-4.2u` below the floor at `t = 0.0s`!); enable `ground_pin_limbs: true` (or `[hind_l, hind_r]` on bipedal rear-ups, `[fore_r/l, hind_l, hind_r]` on single-paw slams) on every idle, perch, and grounded combat clip! **Never leave stale manual `pole:` vectors pointing opposite the natural elbow/knee bend inside `@kinematics`** — let `auto_rig: true` discover limbs cleanly or use pure `plane: sagittal` chains.
      4. **Quantitative Moving-Digit & FK Floor-Contact Gate:** Assert **`moving_digits >= 16` (`24–32` on 4-legged 4-digit bosses)** across locomotion, flight, and combat clips, and inspect EVERY clip's FK min/max `FL/FR/HL/HR` bounds in `verify_bbmodel_ground.py` to catch any `-4..-8u` unpinned claw floor plunge or `+5u` perch float!
      5. **Mirror-Safe Same-Side `sculpt_group` & True Cross-Limb Stride Law (`v37`):** When resolving 1-hop cranial/neck/shoulder anatomical seam overlaps on `mirror_x` rigs, use **same-side `sculpt_group` tokens** (not left-only `blend_with: [ear_base]`, which leaves `ear_pinna_r__ear_base` unmatched on the mirrored side!). Conversely, NEVER mask genuine cross-limb collisions (`forelimb <-> hindlimb/knee_spur` during high-speed rotary gallops) — calibrate `froude` / `duty` / `lift_ratio` / abduction arcs so limbs clear cleanly via real trajectory math!
      6. **Close-Up Animated Foot & Toe Filmstrip Gate:** Always render a dedicated close-up zoom sheet inspecting animated paws, feet, and curling toes across `walk` (stance -> toe-off roll -> swing curl), `hover/flight` (curled aerial talons), and `perch/combat` (pinned flat soles) before sign-off!
13. **`v37` Smooth Serpentine Multi-Vertebra Pitching, Winged-Dragon Gait Clearance & Blockbench 5 Root-Position Parity (*Malzeno & White Fatalis v37* Distillation):**
    - **Smooth Multi-Vertebra S-Neck Pitching (`<= 20°` Per Joint, Zero Single-Vertebra Kink!):** Never dump an abrupt `>35°` pitch reversal onto a single distal cervical vertebra (e.g. `neck_4 = -8° -> neck_5 = +46°`), which cracks open an ugly dorsal gap between `neck_4_core` and `neck_5_core` and breaks dorsal mane flow. Distribute serpentine S-curve pitch smoothly across `neck_1..neck_5` (`<= 20°` per upper vertebra with `seat = 0.28–0.30` and overlapping dorsal crests/mane planes), and distribute forward lunge pitch in combat (`neck_serpentine_bite`) evenly across `neck_1..neck_3` (`[6°, 10°, 12°]` rather than slamming `neck_1` `+16°` into `chest_scute_front`).
    - **Right-Handed Wing-Sail Clearance Above Forelimb Elbow Spurs During Walk/Run (`Rx+ / Rz-` on `-X` Wings):** On winged elder dragons carrying outward-and-backward forelimb elbow spurs (`arm_elbow_spur`), negative `Rx` (`-8°..-14°`) on `wing_l_upper` pitches the rear trailing edge (`-Z`) of `wing_inner_sail` **downward** right into the rising elbow spur during rear-stance push-off! Use positive `Rx` (`+10°..+14°`, tilting the trailing cape edge upward) paired with upward `Rz` (`-12°..-16°` on `-X`, `+12°..+16°` on `+X`) during `walk`/`run` so folded cape wings soar cleanly above striding forelimbs (`0` swept-SAT collisions).
    - **Root-Position Total-Sum Parity in Exporters (`root_translation_delta + pose_position_deltas[root]`):** Whenever a procedurally solved clip (`auto_rig: true` / `ground_pin_limbs: true` / `com_balance > 0`) ALSO carries authored manual `key pelvis_root.position` crouch or rear-up tracks (such as `ecology_cape_drape_perch` `-4.6u` crouch or `roar_silver_duke` `+1.4u` rear-up), `se3_chain.py` sums `root_translation_delta + pose_position_deltas[root_bone_name]`. Both `bbmodel5_exporter.py` and `bedrock_gecko_exporter.py` MUST export that exact sum — otherwise Blockbench 5 drops the crouch/lift while keeping the bent IK leg angles, making crouched perches float `+4.6u` in mid-air! Always verify `verify_bbmodel_ground.py`'s animated FK table (`FL/FR/HL/HR`) on every ground-pinned clip (`0.0u ± 0.3u`).
14. **`v38` Universal Multi-Skeleton Biomechanics Playbook (*Lagiacrus, Legiana, Wintermoon Nettle, Puppet Spider, Wirebug & Sandstar* 6-Archetype Distillation):**
    - **Underwater Buoyancy & Hydrodynamic Limb Positioning (`swim { ... }`):** Never animate swimming Leviathans, crocodiles, or sea monsters with grounded `gait` clips (leaving legs dangling straight down). Use SyncCraft's native `swim_solver.py` (`leviathan_cruise` + `limb_mode: hydro_tuck` sweeping fore/hind limbs backward & laterally outside `tail_1`, or `leviathan_hover` + `paddle_row`, with `Sim(3)` lateral myomere muscle contraction and `buoyancy_y` + `ground_clearance: true`).
    - **Right-Handed `+Z` Toe-Splay Sign Law & Isolated Per-Toe `sculpt_group`s:** On a `-X` left foot pointing `+Z` forward, `Ry(+θ)` fans medial (`+X`, for `toe_in_l`) and `Ry(-θ)` fans lateral (`-X`, for `toe_out_l`). **NEVER share one `sculpt_group` (e.g. `leg_sc`) across sibling toes/fingers** — it silently masks crossed-toe intersections from `audit_scene`! Give every toe its own `sculpt_group` (`toe_in_sc`, `toe_mid_sc`, `toe_out_sc`, `toe_hal_sc`), assert outward tip divergence programmatically (`tip_x(out) < tip_x(mid) < tip_x(in)`), and visually inspect an unobstructed Top-Down Foot Splay close-up!
    - **Pure-Radial Arthropod IK & Coplanar Radial Poles (`Puppet Spider`):** On radially fanned 6- and 8-legged arthropods (`coxa = pure Ry`, `femur = pure Rz`, `tibia = pure Rz`), place each leg's IK pole strictly inside its radial plane (`pole_w = hip_w + normalize_xz(eff_w - hip_w) * L + [0, +2*L, 0]`) for `0.000°` idle drift and `0.00000u` slip across `hexapod_tripod` & `metachronal_wave`. Tag downward fangs with `bio_joint: arthropod_chelicera` / `arthropod_pedipalp` (not vertebrate `jaw_hinge`).
    - **Coleoptera Dual-Wing Semantics (`Wirebug`):** Keep hardened V-raised `elytra` (`bio_joint: elytron_hinge`) rigid with `±2–5°` harmonic buzz (`0°` mid-shell bend) while membranous hindwings execute high-frequency Figure-8 strokes (`±25–35°`). Order C-curved abdomens topologically by tree depth (not world Z).
    - **Saltatorial Bipeds & Rest-Pose-Aware Digitigrade Knees (`Sandstar` & `Legiana`):** `resolve_bone_joint_limits` now offsets `hinge_knee`/`hinge_elbow`/`hinge_hock` headroom by `bone.rest_euler[0]` so deeply folded knees (`+68°..+72°`) extend cleanly in mid-jump/takeoff; `gait_solver` automatically suppresses blind `±32°` humanoid arm counter-swings on tiny `< 0.35×` vestigial forepaws!
    - **Radial Pelagic Analytical Jacobian Projection (`Wintermoon Nettle`):** Project radial flare, CCW helical swirl, and wind waves directly from `WorldBone.rotation` Jacobians (`v_rx = cross(R[:,0], -R[:,2])`, `v_ry = cross(R[:,1], -R[:,2])`) and anchor `N`-fold rings via `joint(at=[r*cos(phi), r*sin(phi), z])` (never `attach` onto non-square octagon bars!). Always verify floating/hovering models with upgraded `verify_bbmodel_ground.py` (`ALL_min >= 0`).
15. **`v38.1` Mandatory Full-Body Kinetic Chain & Dynamic Tail / Secondary Appendage Responsiveness Law (*Owner Sandstar Animation Review*):**
    - **Failure Mode Caught:** Authoring keyframes only on proximal `tail_1..tail_4` (omitting distal `tail_5..tail_6` / `plume`) or balancing `tail_1..tail_4` pitch signs so their cumulative angle cancels (`~8°` net change) leaves long tails (jerboas, lizards, foxes, macaques, dragons) moving like a **fixed rigid S-curve statue** rather than physically reacting to body hops, turns, pitch, and momentum!
    - **4-Point Mandatory Full-Chain Responsiveness Contract (Every Articulated Creature & Endemic Life):**
      1. **100% Segment Coverage (`tail_1` -> `tail_N` + Tip Plume/Fin + Ears/Eyestalks/Antennae):** NEVER leave the outer 1–3 distal tail bones or secondary appendages un-driven! Drive EVERY vertebra down to the terminal plume/fin (`tail_1..tail_N`, `ear_*`, `eyestalk_*`, `antenna_*`, `mantle_*`) plus declare `@kinematics { spring tail_k { stiffness, damping } }` on distal tips.
      2. **Inertial Heave & Pitch Follow-Through (Anti-Fixed-Curve Rule):** On hops, leaps, crouches, sprints, and landings, couple tail pitch (`Rx`) with **progressive `25–45°` phase lag per vertebra** and **monotonically increasing distal amplitude** (`tail_1` `±10–14°` -> `tail_N` `±20–36°`) so the tail genuinely drags downward on explosive upward launch, streams through an S-wave at ballistic apex, and whips upward in elastic overshoot on landing — paired with counter-phase lateral `Ry` sway during strides and turns (`com_balance: 0.25..0.55`).
      3. **Quantitative Excursion Proof:** Print `bone_angular_excursion_deg` across `tail_1..tail_N` in every A1/A2 audit report and reject any active locomotion/jump/combat clip where distal tail bones stay frozen (`< 10°` excursion on a multi-segment tail!).
      4. **Multi-Phase Motion Sequence Visual Proof:** On tailed jumpers, runners, swimmers, and bosses, verify dynamic curvature evolution across at least 4 sequential phases (`Crouch -> Launch -> Apex -> Landing` or `Stride 0% -> 25% -> 50% -> 75%`) — confirming the tail shape genuinely changes curvature rather than translating rigidly as one fixed curve!
16. **`v38.6` Mandatory Stage A0 Video & Moveset Motion Reference Study + Blind-Label Choreography Gate (Universal Anti-Assumption Animation Law):**
    - **Failure Mode Caught:** Animating known bosses, characters, wildlife, or iconic fictional creatures from text assumptions alone produces **generic "fake-label" clips** that miss signature real-world/in-game choreography — for example, spreading wings sideways in a generic V-pose during a clip labeled `"wing_shield_bash"` instead of folding the armored wing faces **forward in front of the chest/face (`+Z`) like a tower shield**, curling a tail lazily behind `-Z` during a `"tail_spear_thrust"` instead of driving the spear-tip straight forward (`+Z`) at the target, leaving a creature standing on all four legs during `"sleep"`, or crouching on all fours during a `"bipedal_breath"`!
    - **4-Step Mandatory Stage A0 Pre-Animation Research Protocol (Applies to ALL Known Entities & Real Animals — Franchise-Agnostic!):**
      1. **Video Platform & Web Motion Research FIRST (Before Authoring Clip #1):** Whenever the target is a known game boss, film/anime character, real animal, or documented fictional creature (or when the user supplies references), **explicitly search video platforms (especially YouTube — e.g., `"<subject> all attacks"`, `"<subject> moveset breakdown"`, `"<subject> wildlife locomotion / hunting slow motion"`) AND authoritative moveset guides** via web search and browser video inspection to observe the ACTUAL choreography before writing a single combat or behavioral keyframe! *(If a video platform is blocked/unavailable, log the limitation explicitly and cross-verify against authoritative multi-source moveset breakdowns — NEVER invent fake generic swipes from thin air!)*
      2. **Author an Explicit `ANIMATION_REFS.md` 4-Beat Choreography & World-Target Table:** For every signature attack, guard, locomotion, and ecology clip, record:
         - **4 Beats:** `Anticipation / Wind-Up` -> `Active Peak Impact / Guard Pose` -> `Follow-Through` -> `Recovery` (exact cadence & frame hold).
         - **Explicit World-Space Appendage Targets (`+Z` front / `-Z` aft / `±X` lateral / `+Y` up / `y = 0` ground):** E.g., *"Forward Wing-Shield Guard: both wing humeri swing forward (`+Z`), elbows bend medially, and flat dorsal shield surfaces wall off the head/chest (`wing_fore.z > chest.z + 6u`) before lunging forward"*; *"Tail-Lance Stab: body pivots/rears or front-flips while driving tail tip forward past flank/overhead toward `+Z`"*; *"Sleep Rest: `pelvis_root.y` drops belly/haunches flush onto `y = 0`, limbs tuck horizontally, wings fold smoothly down over back, head rests low near paws"*.
      3. **Quantitative Spatial Target Assertion (Not Just Zero Collisions!):** Passing `0` SAT collisions proves only that parts did not intersect — it does NOT prove the action actually happened! Verify the defining spatial invariant of every signature clip (e.g., shield wings actually cross in front of chest `+Z`; lance wingtip actually points straight forward `+Z`; sleep belly actually drops onto `y ~ 0`; bipedal attacks actually lift forelimbs high off the floor!).
      4. **The Blind-Label Filmstrip Silhouette Test (Mandatory Visual Gate `A5b`):** After rendering every `A2` filmstrip, inspect each row and ask adversarially: **"If I erased the title text above this row, could a viewer IMMEDIATELY recognize the exact signature move (`shield_guard`, `lance_thrust`, `tail_stab`, `backflip_slam`, `sleep`, `sky_summon`) from the 3D silhouettes alone?"** If the answer is NO, **REJECT AND RE-CHOREOGRAPH** before sign-off!
17. **`v38.8` Cross-Archetype Kinematic, Floor-Clearance & Origami-Wing Precision Laws:**
    - **Same-Sign Forearm Sphinx-Fold & Sleep Breathing Pitch Moderation (Zero Underground Sleep/Crouch Dips):** On rest-grounded quadrupeds (`rest_upper ~ +32..+60°`, `rest_fore ~ -44..-52°`), opposite-sign pitch deltas on `arm_upper` and `arm_fore` cancel out and plunge forearms `-14..-21u` below `y = 0` on unpinned floor-lying sleep or crouch-to-leap frames! Apply **same-sign negative deltas** (`arm_upper < 0` + `arm_fore < 0`) so forelimbs fold horizontally forward (sphinx sleep) or tightly against the chest (crouch/barge/flip), and moderate `muscle_drive thorax_diaphragm` breathing peaks (`~0.25`) on floor-lying sleep clips so chest expansion never drives resting heads or paws below `y = 0` (`verify_bbmodel_ground.py` `ALL >= -0.8u`).
    - **Symmetric Bell-Curve Trajectory for True `±360°` Aerial Somersaults:** Pair `solve_vestibulo_ocular_stabilization`'s smooth $C^1$ acrobatic roll-off (`55°–95°` torso tilt -> `gain = 0.0`, eliminating `180°` antipodal snaps!) with a smooth symmetric bell-curve `pelvis_root.y` parabola across the `-90° -> -185° -> -285° -> -360°` arc so peak vertical velocity stays well inside adaptive `288`-frame swept-SAT convergence (`0` collisions, `TAIL_min > 0`).
    - **Deterministic Morphology Bone Naming & Arthropod/Digitigrade IK Headroom:**
      - Avoid embedding `"ear"` inside axial bone names (use `thorax_aft` / `body_aft`, NEVER `thorax_rear`!) and avoid `"hind"` on folded insect wings (use `wing_l_fan` / `wing_r_fan` instead of `hindwing_l`, preventing `-dz` forward-axis bias).
      - Name elevated non-grounded prop-pushing limbs `palp_*` or `claw_*` so `morphology.py` and `verify_bbmodel_ground.py` cleanly isolate grounded walking legs.
      - On radial hexapods/arachnids/caterpillars, emit explicit 3-bone `[coxa, shin, foot]` chains (or coplanar `Ry(yaw) @ Rx(pitch)` planes for 8-leg `metachronal_wave`), and give digitigrade bipeds coiled rest headroom (`-28°/+56°/-28°`, `reach < 0.89`) for `< 0.0001u` stance slip!
    - **Rigid Origami Wing Unfold & Zero-Collision Membrane Precision:**
      - Apply exact counter-yaw (`Ry = +22..+24°`) to innermost `0.5u` membrane slabs on swept-back wings/capes so medial edges stay strictly parallel to `X = 0` (`0` dorsal midline collisions); stagger `3–4` overlapping feather/fin bays in alternating `Low -> Mid -> Low` normal offsets (`0.55–0.60u`, `0` Z-fighting).
      - On armored beetles & hoppers, conceal multi-fold rigid origami hindwings inside gullwing dorsal elytra (`+45°/+34°`) and unfold via **yaw-first linear sweep** (`Ry` before roll under ZYX Euler) to `2.4–3.6×` torso span without illegal scale hacks; on bats, place carpal finger hubs at `pivot: [0,0,0]` for clean concave scallop fanning; and on legless ground cushions, pair vertical `Sim(3)` breathing swells with pivot-height compensation (`ALL_min >= -0.10u`).
18. **`v38.9` DSL Parser Parity, `mirror_x` Bilateral Keyframe Law & Specialized Aquatic/Folded-Wing/Curl Craft:**
    - **Exact SyncCraft DSL Syntax Guard (Zero Parser Retries):** `spring` accepts `stiffness` and `damping`; `muscle` requires `origin_bone`, `insertion_bone`, `action_axis`, `moment_arm_deg`; `gait` uses `lift_ratio`; and keyframe tracks accept `linear` or `catmullrom` (never `smooth`) — always use **`linear`** on flush-closing articulated jaws/petals so Catmull-Rom never overshoots negative past `0°`!
    - **`mirror_x` Bilateral Keyframe Sign & Locator Laws:** Because `se3_chain.py` applies `delta_euler` directly to the already-mirrored `_r` bone's `local_rotation`, bilateral mirror tracks require **same-sign `Rx`** and **inverted `-Ry` / `-Rz`** on `_r` bones (while circumferential radial swirls around a vertical ring use same-sign `Rz` across all radials). Always pass bilateral `-X` offsets directly in `joint(parent=..., at=[-x, y, z])` (never rely on off-center locator offsets) and classify mirrored dual-sided exterior/interior faces via world normals (`box.center[0] * n_vec[0]`).
    - **Folded-Rest Wing-vs-Foreleg Clearance & Pure `Rz` Unfold:** Keep low-draping folded wings at `solid_min_y >= 2.35u` above ground (above `morphology.py`'s `2.20u` quadruped-foreleg threshold) and fold `-X` wing arms strictly around pure `Rz` with distinct 3-layer Y offsets (`-0.35u / +0.25u / -0.70u`) so declarative unfold keys compose natively with `flight { mode: cruise | hover }` (`asym = 0.0000`, `3.4×` gap-free span!).
    - **Teleost/Cephalopod `swim {}` Topology & Paired-Fin Yaw Sign Law:** Root fish at `body_mid` or `body_aft` (`-> body_front -> head` + `tail_1..3` + `caudal_fin`), stagger foldable multi-bone dorsal sails (`inflate: 0.00/0.03/0.06`), sweep `-Z`-trailing `-X` pectoral/pelvic fins outward with **`+Ry`** (never `-Ry`, which sweeps into the belly!), name cephalopod crown arms `crown_arm_*` (avoiding quadruped `arm_*` matching), and on vertical (`+Y`-facing) medusae orient 8 radial tentacles with local `+Z` outward (`Ry = 0°..180°`) paired with horizontal `(X, Z)` `bell_root.scale` pulsing.
    - **Flush 4-Petal Radial Ring-Maws & Reclined Ball-Curl Clips:** Front a hollow recessed chamber with 4 mitered flush petals (`lip_top`, `lip_bot`, `lip_l`, `lip_r`) that stay 100% sealed at rest and roll back `±56°..68°` on lunge; on stepped heads (`cranium_core` wider behind narrower `orbital_core`/`snout_brow`), lateral `32x` eyes stay flush-unobstructed with `eye_vs_cranium <= 0.00u`; and on deeply reclined defensive ball-curl clips (`pelvis_root.rx < -50°`), disable `auto_rig` `head_stabilize` (retaining `auto_bio_limits` & `Sim(3)`) so gaze stabilization doesn't pitch the head back into tucked limbs!

---

## 8. End-to-End Artist + AI Co-Pilot Workflow & Verification Gates

1. **Build the Species-Authentic Sculpted Skeleton First (`W1–W5+`, `T1–T3+`):** Follow the `v27` Anti-Template Species-First Morphometric Measurement & In-Game Close-Up Verification Protocol (`cranial_and_jaw_structural_reasoning.md §1C`), carve cross-cube `16/32 px` pixel relief, lock static grounding (`<= 0.30u`), and seal inner wing membranes (`0.000u` seam).
2. **Stage A0 — Mandatory Video & Moveset Motion Reference Study (`ANIMATION_REFS.md`):** Before writing combat, guard, sleep, or signature behavioral clips on any known boss, character, or real animal, research real video footage (especially YouTube moveset/locomotion breakdowns) + authoritative guides and lock the **4-Beat Choreography & World-Space Appendage Target Table** (`§7B #16`) so every clip matches authentic motion rather than guesswork!
3. **Annotate Biological Harness (`@rig` + `@muscles` + `@kinematics`, or simply enable `auto_rig: true`):**
   - Add `bio_joint` tags on key constraint joints (especially `wing_anchor_locked` on `wing_l_f4`, `finger_proximal`/`finger_flex`/`thumb_saddle` on digits, `jaw_hinge` on jaw).
   - Add `2–6` functional `@muscles` (diaphragm breathing, throat/lightning sac with `scale_bulge`, jaw/tongue tendon coupling, tail whip).
   - Enable `auto_rig: true`, `foot_roll`, `digit_articulation`, `com_balance`, `antagonist_bulge`, `pectoral_bulge`, `neck_aim`, and `spine_coil` cleanly per clip.
4. **Run Capability Introspection (`auto_assemble_biomechanical_rig` / `analyze_motion_capabilities`):** Verify the solver automatically discovered the creature's exact locomotion class, limb count, multi-phalanx digits, wing pairs, necks, tails, and recommended gait/flight modes.
5. **Author Declarative & Reference-Grounded Clips (`in_place: true`, `auto_rig: true`, `head_stabilize`, `auto_bio_limits: true`):** Combine `gait` + `flight` + `swim` + `muscle_drive` + YouTube/reference-verified signature combat/ecology keyframes.
6. **Run All 6 Mandatory `v38.6` Animation Audit Gates:**
   - **Gate A1 (Swept OBB SAT):** `audit_behavior_clip(..., max_frames=288)` -> `passed=True`, **`0` collision events** across 100% of clips.
   - **Gate A2 (Closed-Loop Foot Lock & Foot-Roll):** `max_stance_slip_units < 0.02u` on all grounded `gait` and `flight(mode=mantle)` clips; all 8 gait checks (`foot_slip`, `reach`, `joint_limits`, `clipping`, `balance`, `ground`, `loop`, `speed_match`) `True`; multi-phalanx digits flex cleanly with `0°` backward hyperextension.
   - **Gate A3 (Minecraft In-Place Entity Contract & Multi-Bone `Sim(3)` Scale Proof):** Direct `.bbmodel` & `.animation.json` inspection verifies `in_place == True`, `max|root_z| == 0.00u`, zero fake `+30u` hover altitude, matched `synccraft_root_motion.speed_blocks_per_s`, and live multi-bone `Sim(3)` `scale` channel keyframes across chest, throat/neck, wing arms, limbs, and tail.
   - **Gate A4 (Bilateral Airfoil Symmetry & Wing-Seal):** `flight` clips pass `asym <= 0.001` and `wing_l_f4` delta `== [0, 0, 0]` across every frame.
   - **Gate A5 (Visual Multi-Frame Filmstrip Proof + Blind-Label Choreography Gate):** Render multi-frame filmstrips covering every locomotion, flight/swim, combat, guard, and ecology clip, and verify BOTH (a) clean kinematics/seams/feet AND (b) the **Blind-Label Silhouette Test (`§7B #16`)** confirming every signature attack/guard/sleep pose is unmistakably faithful to real video/moveset references!
