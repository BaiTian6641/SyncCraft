# ADR 0011: Trustworthy Motion/Sculpted Audits, Derived `@morphology`, Dimensionless Froude `gait`, and Sculpted Pixel-Art Primitives

- **Status**: Accepted (additive Block-DSL v4.0/v4.1 hardening & product layer; 100% backward-compatible with valid v3.0–v4.0 programs and ADR 0009/0010 corpora)
- **Date**: 2026-11-16
- **Related Tasks**: `P1-01`, `P1-02`, `P1-03`, `P1-06`, `P3-01`, `P3-02`
- **Builds On**: `ADR 0004`, `ADR 0006`, `ADR 0007`, `ADR 0008`, `ADR 0009`, `ADR 0010`

## Context

Empirical verification of commit `1f41187` plus three sculpted Blockbench dragon test flights (`blockbench-sculpted-style`) identified five critical audit/compiler defects that could let invalid geometry or mismatched motion pass, alongside three product-layer capabilities needed for an interactive simulator-grounded Blockbench assistant:

1. **Five Correctness Defects (Gate A)**:
   - **Silent `joint_limits` clamping & keyframe audit/export divergence**: `spin` tracks exceeding `[-180, 180]` were silently clamped at `180°` (freezing flywheels/pinions mid-clip), while `key` tracks were audited after clamping but exported unclamped.
   - **Conditional & non-failing swept motion stepping**: Adaptive subdivision ran only for `v4.0`, `spin`, or `slide` clips, allowing a `v3.2` keyed `360°` turn to pass on 12 frames despite 55 colliding frames at `1,440` samples; hitting the frame ceiling also did not fail the audit.
   - **Over-permissive `sculpted` burial & missing cross-bone `blend_with` ergonomics**: Same-bone pairs overlapped without `blend_with` and bounded depth by `t_med` rather than thinner-axis/volume share, allowing two `8×8×8` cubes overlapping `6.5u` (`81%` buried) and an `8×1×8` plate `96%` buried to pass; conversely, concentric `octagon` hubs needed a first-class primitive.
   - **Advisory `@layout` budgets/envelopes & ungrounded creatures**: `audit_scene` ignored `@layout` `budget`/`envelope` violations (`max_boxes = 1` passed on a 2-box scene), and nothing verified that grounded entities touch the floor (`Y = 0`).
   - **Unversioned v4 blocks**: `@layout`, `@component`, and `@assembly` parsed even when `dsl_version` defaulted to `"3.0"`.

2. **Interactive Speed (`< 2 s` Target)**:
   - Pairwise Python collision checks and per-patch posed AO raycasts took ~28 s on 120-box dragons; vectorized broadphase AABB pruning and vectorized ray-OBB culling are required for interactive feedback.

3. **Product & Sculpted Art-Style Layer (Stages 1–3)**:
   - **Stage 1 (`@morphology`)**: Deterministic geometric extraction of limbs (`fore_l`/`fore_r`/`hind_l`/`hind_r`/`mid_l`/`mid_r`), hip height, leg length, posture (`erect` | `crouched` | `sprawled`), contact points, axial chains (`spine`, `neck`, `tail`), wings, mass/CoM, symmetry pairs, and reach envelope from any rig.
   - **Stage 2 (`gait` + Planar/Sprawled/Spine-Compensated IK + Root Motion)**: Dimensionless Froude-scaled `gait` statements (`Alexander & Jayes 1983`, $v = \sqrt{\mathrm{Fr}\, g\, h_{\text{hip}}}$) binding by `@morphology` roles across quadrupeds, bipeds, sprawlers, and hexapods with zero bone names, `plane: sagittal` / `sprawled` IK, ancestor spine-motion compensation, and exported `root_motion`.
   - **Stage 3 & Sculpted Style (`edge` alpha cut-outs, `shade: banded`, `@stamps`, `coat`, `octagon`, `seat`, `layer`)**: First-class transparent pixel-cutout `edge` profiles on `plane`s rendered with alpha-tested ray-OBB traversal, discrete 3–5 step `shade: banded(...)` cel-shading with normalized mid-tone lightness and `glow_bleed`, parametric `stamp`s (`mirror` support), named `coat:` patterns, `octagon`/`prism` faceted hubs, `attach(..., seat=...)`, and `plane` `layer: <int>`.

---

## Decision

### 1. Gate A Correctness Invariants

1. **Strict `joint_limits` Enforcement & `continuous` Axes**:
   - `joint_limits: continuous` (and per-axis `continuous`, e.g. `[continuous, [-5, 5], [-5, 5]]`, plus `@assembly` `connect { ..., limits: continuous }`) explicitly marks unbounded rotational axes (`(-inf, +inf)`).
   - Any `track`, `slide`, `key` (`rotation`/`position`), or `pose_bias` exceeding finite `joint_limits` / `PrismaticJointDecl.limits` beyond `1e-4` raises `DSLValidationError` at validation time. Exporters (`export_bedrock`, `export_geckolib`, `export_bbmodel`) and `audit_behavior_clip` evaluate the exact same poses.
2. **Unified Swept Audit Across All Versions (`max_frames = 144`)**:
   - `audit_behavior_clip` applies adaptive subdivision (`max_step_deg = 10.0°`, `max_step_units = 0.5u`, `max_frames = 144`) to **every** clip regardless of `dsl_version` or channel type. If `measured_max_step_deg > max_step_deg` or `measured_max_step_units > max_step_units` after reaching `max_frames`, `step_bound_satisfied` is `False` and `passed` is `False`.
3. **Tightened `sculpted` Burial & Explicit `blend_with` / `prism` Rules**:
   - Undeclared same-bone box pairs (`not is_declared and not is_blend_pair`) no longer receive automatic `sculpted_same_bone_miter` immunity; same-bone overlaps require either direct `place(base_box=...)` adjacency or explicit `blend_with` (which also works across **any two bones** for joint-bridging shingles/crests).
   - Deterministic $6 \times 6 \times 6$ OBB-in-OBB interior lattice integration rejects any non-plane `sculpted` pair whose buried volume fraction exceeds **`0.65`** (`> 2/3` buried), unless both boxes belong to the same generated `prism` / `octagon` hub (`prism_group`). Parallel (`theta ~ 0`) non-hinge overlap is bounded by `0.85 * t_thin`.
4. **Enforced `@layout` Budgets, Envelopes & `grounded` Floor Check in `audit_scene`**:
   - `compile_geometry` attaches `layout_contracts` to `Scene`, and `audit_scene(scene)` folds `max_boxes`, `max_bones`, `max_span`, `min_ground_y`, and `grounded` (`|min_y - ground_y| <= ground_tol`, default `ground_y = 0.0`, `ground_tol = 0.15`) directly into `audit_scene["passed"]` and `contract_violations`.
5. **Mandatory Version Gate**:
   - Using `@layout`, `@component`, `@assembly`, `slide`, `prismatic_joint`, or non-default `units_per_block` without `dsl_version` in `{"4.0", "4.1"}` raises `DSLValidationError`.

### 2. Derived `@morphology` & Dimensionless Froude `gait`

- `analyze_morphology(program, scene=None)` derives limbs, roles (`fore_l`, `fore_r`, `mid_l`, `mid_r`, `hind_l`, `hind_r`), posture (`erect`, `crouched`, `sprawled`), hip height $h_{\text{hip}}$, segment sum $L_{\text{leg}}$, reach envelope, ground contacts, `spine`/`neck`/`tail`/`wing` chains, volume-weighted mass & CoM, support polygon, and X-symmetry pairs.
- `gait <name> { pattern, froude, duty, phase, lift_ratio, heave_ratio, pitch_deg, spine_wave_deg, tail_lag, style }` inside `@behavior` `clip` (or `expand_gait_to_clip`) scales speed $v = \sqrt{\mathrm{Fr}\, g\, h_{\text{hip}}}$ (`g = meta.gravity` or `9.81 * meta.units_per_block`), stride, cycle period, world-planted foot trajectories (compensating ancestor spine heave/pitch/yaw!), sagittal/sprawled IK planes, and `root_motion` velocity.

### 3. Sculpted Pixel-Art Language & Alpha-Tested Renderer

- **`plane` `edge:` Cut-Outs & `layer:`**: `edge: ragged(...) | spikes(...) | drip(...) | scallop(...) | triangle_bay(...)` writes crisp binary pixel-art alpha (`A in {0, 255}`) onto plane UV islands; `layer: k` offsets planes by `0.25 * k` units along their normal. `render_world_observation` performs multi-hit RGBA alpha testing (`A >= 128`).
- **`@style` `shade: banded(...)` & `glow_bleed`**: Discrete 3–6 step hue-shifted cel bands (`direction=down`, `cluster=diagonal`, 1 px top highlight & bottom contact shadow, mid-tone lightness preservation) plus `glow_bleed: 0..4` px halo bleeding around emissive pixels.
- **`@stamps`, `@materials` `coat:`, `attach(..., seat=...)`, and `octagon` / `prism`**: Parametric face-unit feature stamps (`mirror` flag), named `coat:` presets (`plain`, `mottled`, `spotted`, `striped`, `banded_limb`, `scale_rows`), segment seating `seat` in `[0, 0.65]`, and multi-box `octagon` / `prism` expansion.
