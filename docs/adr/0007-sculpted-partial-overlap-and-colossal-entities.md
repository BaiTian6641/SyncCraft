# ADR 0007: Explicit Sculpted Partial-Overlap Contract (`audit_profile: sculpted` & `blend_with`), Continuous Mechanical `spin` Kinematics & Colossal Entity/Vehicle Scaling

- **Status**: Accepted
- **Date**: 2026-11-15
- **Tasks**: `P1-01-sculpt`, `P1-02-sculpt`, `P1-03-colossal`, `P3-01-vehicle`
- **Extends**: `ADR 0001` (Starter Scope), `ADR 0002` (Audit & Cost Evidence), `ADR 0006` (Musculoskeletal Complex Entities & Artist Loop)

## Context

Artists and world-model assistants constructing **large, complex multi-bone entities and colossal moving structures** — such as multi-winged biomechanical apex dragons (`60–150+` cuboids with tapered multi-slope snouts, swept `> 60°` horns, curved armored carapaces, and imbricated dorsal scales) or **castle-like moving fortress vehicles** (`70–200+` cuboids with chamfered armored prows, mitered bastion domes, rotating turrets, continuous drive wheels/gears/turbines, and multi-leg hydraulic stance-IK locomotion) — require intentional, rule-governed **partial cuboid overlap** to form sloped surfaces and piecewise-linear semi-curves out of rigid OBBs.

Prior to this ADR, Block-DSL v3.2 provided only two `audit_profile` modes:

1. **`strict` (default, `P1-02` / `M4`)**: Enforces `<= 0.5u` inset, `delta_hinge(theta) <= 2.5u` capped abruptly at `theta <= 60°`, `center_buried == False`, and `1e-7u` zero-tolerance SAT on any pair lacking a direct `place`/`attach`/`joint` edge. While essential for zero-slop mechanical fits and frozen Gate `G2` benchmarks, `strict` rejects:
   - Mitered multi-box slopes/curves sharing a `base_box` (`slope_a` and `slope_b` both placed on `snout_core` or `dome_ring`),
   - Wedge/chamfer boxes tucked `> 0.5u` into a thicker structural core, and
   - Steep `> 60°` horns, mandibles, arched roofs, and imbricated scale plates.
2. **`authored`**: Designed strictly for lossless decompilation of third-party human `.bbmodel` / `.geo.json` assets (e.g., `1,967 / 1,967` Cobblemon rigs). It allows negative `inflate`, skips `disconnected_physical_components`, and permits broad legacy parent/sibling overlaps that are too permissive for clean-room synthetic or external-AI dataset generation.

Additionally, colossal moving castle vehicles exposed two concise mechanical/surface gaps:
- `@behavior` `track` laws (`wave`, `flap`, `hinge`, `impulse`) were purely oscillatory and lacked a continuous `360°` **`spin`** law for wheels, sprockets, gears, steam flywheels, windmills, and observatory domes.
- `allocate_uv_layout` defaulted to a hard `max_width=512` row wrap even when colossal fortress walls or `100+`-box entity atlases exceeded `512 px` width or `512x512` area.

## Decision

### 1. Preserve `strict` and `authored` Invariants Untouched

- **`audit_profile: strict`** thresholds (`max_inset_depth = 0.5`, `MAX_HINGE_ANGLE_DEG = 60.0`, `MAX_HINGE_ALLOWANCE_UNITS = 2.5`, zero sibling/same-bone overlap immunity) remain **100% unchanged**. All frozen `G2` dense-assembly holdouts and `55` pre-flight canary families continue to run under `strict`.
- **`audit_profile: authored`** remains reserved exclusively for decompiled external human assets (`negative inflate` allowed only in `authored`).

### 2. Explicit `audit_profile: sculpted` + Per-Box `blend_with: [...]` Contract

We introduce `audit_profile: sculpted` in `@meta` together with an optional explicit `blend_with: [box_a, ...]` field on `box` / `plane` declarations (compiled into `Scene.blend_pairs` across unmirrored, `mirror`, and `radial` expansions):

- **Authorized Sculpted Overlap Scope (Never Blanket Sibling Immunity)**:
  1. `declared_place`, `declared_attach`, `declared_joint` edges,
  2. Explicit `declared_blend` pairs (`blend_with: [...]`, supporting cross-bone mitered prows, mirrored centerline plates, cheek-to-jaw bevels, and cyclic `radial` dome rings),
  3. `sculpted_same_bone_miter` (tilted/stepped cubes within the same bone forming a continuous slope, arch, or scale row), and
  4. `sculpted_parent_child_joint` (adjacent parent-child bone segments flexing across rest pose and `@behavior` animation).
  Separate sibling bones (`thigh_l` vs `thigh_r`, `turret_1` vs `turret_2`, `wing_l` vs `arm_l`) receive **zero implicit overlap immunity** unless explicitly linked via `blend_with: [...]`.
- **Mathematical Bounds Enforced on Every Sculpted Pair**:
  - **Exterior Protrusion Invariant (`swallowed_redundant_box` Ban)**: For overlapping OBBs $(A, B)$, neither box may be swallowed inside the partner OBB. Specifically, by convexity of OBBs, each box's 8 corners must achieve exterior signed protrusion $\ge \min(0.25,\, 0.15 \cdot t_{\text{thin}}(A, B))$ (or $> \epsilon$ for zero-thickness `plane` quads). Every sloped/curved box must contribute visible exterior silhouette.
  - **Bounded Overlap Depth (`MAX_SCULPTED_OVERLAP_RATIO = 0.85`)**:
    $$\text{raw\_min\_overlap} \le \max\!\big(\text{max\_inset\_depth},\; \delta_{\text{hinge}}(\theta; 135^\circ, 6.0),\; 0.85 \cdot t_{\text{med}}(A, B)\big) + \text{inflate}_A + \text{inflate}_B + \epsilon$$
    where $t_{\text{med}}(A, B) = \min(\text{median}(s_A^+), \text{median}(s_B^+))$ accounts for tilted separating-axis cross-projection when thin sloped plates tuck into thicker structural cores.
  - **Physical Connectivity**: Declared edges must physically touch (`raw_min_overlap >= -1e-4u`), and static `disconnected_physical_components` are rejected (`inflate >= 0.0` required).
  - **Zero-Tolerance Defect Bans**: Exact coplanar Z-fighting (`0°` face flicker) is always rejected (`z_fighting_pairs`), and **all non-adjacent bone pairs** (e.g., wing vs. torso, foreleg vs. hindleg, walking-castle leg vs. hull tower, turret barrel vs. chimney) enforce `1e-7u` zero-tolerance 15-axis SAT in both static `audit_scene` and 12-frame swept `audit_behavior_clip`.
  - **Explicit Audit Telemetry**: Every exempted overlap is logged in `audit_scene(...)["sculpted_overlap_pairs"]` with `pair`, `exemption_reason`, `raw_min_overlap`, `max_allowed_overlap`, `relative_angle_deg`, and `min_protrusion_units`.

### 3. Continuous `spin` Track Law & Gimbal-Safe Axle Export

- `@behavior` `track` adds `spin(axis=x|y|z, amp_deg=..., freq_hz=..., phase_deg=..., bias_deg=...)` evaluating continuous angular progression $\theta(t) = \text{bias\_deg} + \text{amp\_deg} \cdot (t \cdot f + \phi / 360^\circ)$ (e.g., `amp_deg=360` for forward wheel/rotor spin, `-360` for counter-rotating meshed gears).
- Keyframe extraction in `bedrock_gecko_exporter` preserves exact signed continuous angles on single-axis axles/turrets (`R_xyz(e_rest + delta) == R_local`) so `360°` Y-axis turrets and X-axis wheels never suffer `±90°` Euler gimbal flips, and emits linear `query.anim_time` Molang expressions in `molang` mode.

### 4. Colossal Power-of-Two UV Packing Preservation

- When `allocate_uv_layout` uses its default `max_width=512`, any asset whose islands fit within `512 px` width and `<= 512x512` padded area retains **bit-identical** UV coordinates. Only when a colossal face exceeds `512 px` or total padded texel area exceeds `512x512` (`262,144 px²`) does the packer automatically promote sheet width to the next balanced power of two (`1024`, `2048`, `4096`, up to `8192`), while explicit caller `max_width` bounds continue to fail strictly when exceeded.

## Consequences

- **Positive**: Unlocks Cobblemon-style tapered snouts/shells, Monster Hunter-scale sculpted dragons, and colossal walking/tracked castle vehicles in clean-room and external-AI Block-DSL generation without weakening `strict` (`G2`) or permitting Z-fighting, floating parts, swallowed boxes, or limb-clipping animations.
- **Boundaries Preserved**: `strict` and `authored` semantics remain bit-identical; live game-client runtime import checks remain explicitly tracked (`runtime_import_verified: False`).
