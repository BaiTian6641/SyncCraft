# Block-DSL v4.0 / v4.1 — Canonical Language & Pipeline Specification

> [!IMPORTANT]
> **Canonical v4 Specification (`ADR 0008`–`ADR 0011`)**: This document is the authoritative specification for **Block-DSL v4.0 / v4.1**, superseding the v3.2 title of [BLOCK_DSL_V3_2_SPECIFICATION.md](BLOCK_DSL_V3_2_SPECIFICATION.md) (which remains available for v3.0–v3.2 construct details and backward-compatibility notes). Every syntax production in this specification is verified against live `lark.Lark` parsing in [grammar.lark](../src/synccraft/dsl/grammar.lark) and the zero-`eval` compiler in [parser.py](../src/synccraft/dsl/parser.py).

---

## 1. Version Gate & `@meta` Contract

Any program using v4 modular assemblies (`@layout`, `@component`, `@assembly`), mechanical `prismatic_joint` / `slide` channels, or non-default `units_per_block` **must** declare `dsl_version: "4.0"` or `"4.1"` in `@meta` (omitting `dsl_version` defaults to `"3.2"` and raises `DSLValidationError` when v4 constructs are present):

```blockdsl
@meta {
  archetype: entity,
  target_profile: bedrock_entity,
  texel_density: 16,
  units_per_block: 16,
  dsl_version: "4.1",
  size_step: 0.5,
  audit_profile: sculpted,
  gravity: 156.96
}
```

- `gravity`: Optional gravitational acceleration in model units $/ \text{s}^2$. Defaults to $9.81 \times \text{units\_per\_block}$ (`156.96` $\text{u/s}^2$ at `16` units/block) for dimensionless Froude locomotion scaling.

---

## 2. Geometry Authoring Rules, Plane Orientation & Sculpted Audit (`ADR 0007` / `ADR 0011`)

### 2.1 Zero-Thickness `plane` 2D `size` Ordering (Important Authoring Note)

A `plane` has zero thickness along its `normal` axis and takes the **remaining two axes in ascending `(X, Y, Z)` order**:

| `normal` | 3D Box Size Expansion | `size: [a, b]` Meaning | Typical Use |
| :--- | :--- | :--- | :--- |
| `normal: x` | `[0, a, b]` | **`[Y_height, Z_length]`** (height first, fore-aft length second!) | Dorsal crests, manes, tail fins on the sagittal plane |
| `normal: y` | `[a, 0, b]` | **`[X_width, Z_length]`** | Horizontal wing membranes, leaves, ground decals |
| `normal: z` | `[a, b, 0]` | **`[X_width, Y_height]`** | Transverse frills, collars, banners |

### 2.2 Seating (`seat`), Plane Layering (`layer`), Octagons (`octagon` / `prism`) & Multi-Bend Chains

- **Segment Seating (`attach(..., seat=s)`)**: Scales `anchor` inward by $(1 - s)$ for $s \in [0, 0.9]$ (e.g., `seat=0.28` seats a neck/spine/tail child `28%` into its parent without violating the `0.5u` connected-attach offset rule).
- **Plane Layering (`layer: k`)**: Shifts a `plane` by $0.25 \times k$ units ($k \in [-8, 8]$) along its normal (and across mirror sign on `normal: x`) so overlapping wing panels and centerline fins never coplanar Z-fight.
- **Octagons & Prisms**:
  ```blockdsl
  octagon wheel_hub place(base_box=axle, anchor=[1, 0, 0], align=[-1, 0, 0]) {
    size: [6, 6, 2], axis: x, material: forged_brass
  }
  prism(12) boiler_drum { size: [10, 10, 14], axis: z, material: riveted_hull }
  ```
  Expands deterministically into concentric rotated facet boxes sharing `prism_group`, accepted as `sculpted_prism_hub` under `audit_profile: sculpted`.
- **Multi-Bend Chains**: `chain horn attach(...) segments=5 taper=0.82 seat=0.25 bend=[-20, -10, 12, 22, 15] { box seg { ... } }` assigns per-segment `rest_euler` angles for S-curve necks and direction-changing horns.

### 2.3 Tightened `audit_profile: sculpted` & Integrated `@layout` Contracts

1. **Cross-Bone `blend_with`**: `blend_with: [target_box]` is honored across **any two bones** in the expanded rig, enabling joint-bridging crest plates and armour shingles.
2. **Anti-Swallowing Buried-Volume Bound**: Any non-plane, non-`prism_group` sculpted pair where more than **`65%` (`~2/3`)** of either OBB's 3D volume lies inside its partner is rejected as `swallowed_redundant_box`.
3. **Integrated `@layout` Contracts in `audit_scene`**: `audit_scene(scene)` directly enforces `@layout` `budget` (`max_boxes`, `max_bones`) and `envelope` (`max_span`, `min_ground_y`, and `grounded = true` requiring `|y_min - ground_y| <= ground_tol`).

---

## 3. Modular Parametric Assemblies (`@layout`, `@component`, `@assembly`)

```blockdsl
@layout {
  param body_length = 26 [18, 48]
  let pelvis_len = max(6, round(body_length * 0.35))
  budget max_boxes = 48
  budget max_bones = 28
  envelope max_span = [56, 38, 60]
  envelope grounded = true
}
```

- `@component <Name> { param ..., material_slots: [...], uv_policy: unique|shared, box/plane/octagon/bone/frame/control ... }` encapsulates reusable submodels with local coordinates and oriented 6-DOF `frame` interfaces.
- `@assembly <Name> { root_pivot: [...], instance ..., subassembly ..., connect a.f1 to b.f2 { type: fixed|revolute|prismatic, axis: x|y|z, limits: [lo, hi] | continuous } }` mates instances in exact SE(3) and emits a bidirectional `SourceMap`.

---

## 4. Joint Limits, Continuous Rotors & Unified Swept Motion Audit

- **Hard Compile-Time `joint_limits` Enforcement**: Any `track`, `slide`, `key` (including Catmull-Rom spline interior points), or `pose_bias` exceeding finite `joint_limits` / `prismatic_joint` limits raises `DSLValidationError` at compile time — never silently clamped during audit while exporting unclamped keys.
- **Unbounded Continuous Rotors (`continuous`)**: Wheels, gears, propellers, and beacons declare `joint_limits: continuous`, `joint_limits: [continuous, [-5, 5], [-5, 5]]`, or `connect ... { type: revolute, axis: x, limits: continuous }`. Both Bedrock/GeckoLib and Blockbench `.bbmodel` exporters preserve unwrapped multi-revolution `spin` angles (`360°`, `-720°`).
- **Unified Adaptive Swept Audit**: `audit_behavior_clip` runs adaptive subdivision (`max_step_deg = 10.0°`, `max_step_units = 0.5u`, `max_frames = 144`) across **all** DSL versions and clip types; if `measured_max_step_deg > 10.0°` or `measured_max_step_units > 0.5u` at `max_frames`, `step_bound_satisfied` and `passed` are `False`.

---

## 5. Derived `@morphology` & Dimensionless Froude `gait` (`ADR 0011`)

### 5.1 Derived `@morphology`

`analyze_morphology(program)` deterministically extracts ground-contacting limb chains, canonical roles (`fore_l`, `fore_r`, `mid_l`, `mid_r`, `hind_l`, `hind_r`), posture (`erect` | `crouched` | `sprawled`), hip height $h$, limb length, 3D reach bounds, sole contact point, `spine` / `neck` / `tail` / `wing` chains, box-volume CoM, and 2D ground support polygon.

### 5.2 Dimensionless `gait` Statement & Planar/Sprawled IK

```blockdsl
@kinematics {
  ik_chain leg_fl { bones: [foreleg_l, foreleg_l.shin], effector: [0, -6, 0.5], plane: sagittal }
}
@behavior {
  clip walk {
    duration_s: 1.6
    gait walk {
      pattern: lateral_sequence,
      froude: 0.20,
      duty: 0.68,
      lift_ratio: 0.12,
      heave_ratio: 0.03,
      pitch_deg: 1.5,
      spine_wave_deg: 3.0,
      tail_lag: 0.2,
      style: { weight: 0.5, snap: 0.3 }
    }
  }
}
```

- Dynamic similarity speed $v = \sqrt{\mathrm{Fr}\, g\, h_{\text{hip}}}$ scales stride length and step frequency automatically across morphologies (quadrupeds, bipeds/wyverns, sprawled reptiles, and 6-legged hexapods) from a **single unchanged `gait` block**.
- Closed-loop world-space IK solves limb poses **after** torso heave, pitch, and `spine_wave_deg` so ancestor spine motion produces `< 0.02u` stance slip, and exports `synccraft_root_motion` (`speed_units_per_s`, `speed_blocks_per_s`, `velocity_vec`, `displacement_per_cycle`).

---

## 6. Sculpted Pixel-Art Surface Layer (`edge`, `shade: banded`, `coat`, `@stamps`)

```blockdsl
@materials {
  teal_hide: process(family=patterned_skin, coat=scale_rows)
  limb_hide: process(family=patterned_skin, coat=banded_limb)
  belly_cream: process(family=patterned_skin, coat=plain)
}
@style {
  target_encoding: vanilla,
  shade: banded(steps=4, direction=down, cluster=diagonal),
  glow_bleed: 2
}
@stamps {
  stamp eye on skull.side { at: [0.35, 0.42], size: [2, 1], color: blossom_glow, emissive: 0.95, mirror: true }
  stamp teeth on skull.side { at: [0.65, 0.85], size: [1, 1], color: ivory, rows: 4, spacing: 1.5, mirror: true }
}
```

- **Plane Alpha Cut-Outs (`edge`)**: `ragged`, `spikes`, `drip`, `scallop`, and `triangle_bay` bake binary `{0, 255}` cut-out alpha onto `plane` UV islands; `render_world_observation` performs order-independent alpha-tested (`A >= 128`) ray-OBB traversal so cut-out silhouettes and membrane holes reveal geometry behind them.
- **Banded Cel Shading (`shade: banded(...)`)**: Centers discrete OKLCH value bands directly on `palette.base_lch` (preventing unwanted darkening), adds stepped diagonal pixel clusters, 1-px top highlight and bottom contact-shadow rows, and optional `glow_bleed: 1..6` px around emissive features while keeping physical GGX (`shade: physical`, default) available.

---

## 7. Additive Secondary Motion `layer` & Stance-Touchdown `effect` Triggers (`ADR 0013`)

```blockdsl
@behavior {
  clip breathe {
    duration_s: 2.4
    track lumbar: wave(axis=x, amp_deg=2.0, freq_hz=0.4167)
  }
  clip walk {
    duration_s: 1.6
    gait prowl {
      pattern: lateral_sequence,
      froude: 0.18,
      duty: 0.68
    }
    layer breathe: weight=0.8
    effect touchdown: { sound: "mob.wyvern.step", particle: "synccraft:dust_puff" }
    effect 0.40: { instruction: "mantle_shimmer" }
  }
}
```

- **Additive Motion Layers (`layer <clip>: weight=W`)**: Evaluates secondary clips (`breathe`, `tail_sway`, `wing_settle`) additively; when layered inside a `gait` clip, axial spine/neck deltas are applied **before** closed-loop world limb IK so stance feet maintain `< 0.02u` slip. Cyclic layer graphs are rejected by `validate_program`.
- **Audio, Particle, & Timeline Triggers (`effect <time_s | touchdown>: { ... }`)**: Explicit timestamps (`0 <= t <= duration_s`) emit at `t`; `effect touchdown` expands deterministically across every `@morphology` limb's stance touchdown instant and exports natively to Bedrock `1.8.0` (`sound_effects`, `particle_effects`, `timeline`), GeckoLib, and Blockbench 5 `.bbmodel` (`animators["effects"]`).
