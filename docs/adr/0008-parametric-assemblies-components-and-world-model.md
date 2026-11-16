# ADR 0008: Parametric Submodels, Oriented Attachment Frames, Top-Down Layouts, and Modular World-Model Supervision (Block-DSL v4.0)

- **Status**: Accepted (2026-11-15)
- **Extends**: `ADR 0001` (Typed Block-DSL & Zero-`eval` Compilation), `ADR 0003` (Stateful Surface Physics), `ADR 0006` (Musculoskeletal Control & Clean-Room Provenance), `ADR 0007` (Sculpted Partial-Overlap & Colossal Entities)
- **Preserves**: 100% backward compatibility with all Block-DSL `v3.0`–`v3.2` `@rig`-direct programs, `strict` Gate `G2` (`<= 0.5u` inset, `0` non-exempt overlap), and right-handed `SE(3)` forward kinematics (`det(R) = +1`).

---

## 1. Context and Problem Statement

Block-DSL `v3.2` provides hierarchical `@rig` bones, `let` expressions, lightweight `def`/`use` macros, `mirror_x`/`radial`/`chain` operators, `@materials`, `@muscles`, `@kinematics`, `@behavior`, and `audit_profile: sculpted`. However, authoring and training a generative 3D world model (`LiquidAI/LFM3-VL-2B`) on large multi-part creatures, vehicles, and mechanisms exposed six structural limitations:

1. **Design Organization Coupled to Runtime Bones**: Grouping parts organizationally (`head_assembly`, `engine_module`) previously required emitting runtime animation bones containing geometry or dummy locators.
2. **Fragile Box-Implementation Attachment Coupling**: External child bones attached directly to internal parent box names (`attach(parent_box=ribcage_box, ...)`), so subdividing a chest box into sculpted plates broke downstream limb connections.
3. **Expanded Prefix Leakage (`pod_r__pod_l_shell`)**: `def`/`use` and `mirror_x` exposed flat string-prefixed identifiers rather than encapsulated component namespaces with abstract material slots, parameters, and independent instance controls.
4. **Point-Only Locators Without Orientation**: `locator` lacked orientation (`rot`), preventing full 6-DOF `SE(3)` frame mating and dropping rotation metadata supported by modern Bedrock `1.26.40` locators.
5. **Narrow `joint` Semantics & Fixed-Sample Motion Checks**: `joint(...)` was placement shorthand rather than explicit mechanical DOF (`fixed`, `revolute`, `prismatic`), and fixed 12-sample clip checks could step by `30°–60°` on fast-spinning gears.
6. **Monolithic World-Model Context**: Synthesizing or editing a 70-box asset as one monolithic `@rig` forced the LLM/VLM to regenerate unrelated subtrees instead of planning a master `@layout` and synthesizing/swapping scoped `@component` modules with guaranteed edit locality.

---

## 2. Decision

We introduce **Block-DSL v4.0 (`dsl_version: "4.0"`)** as an **additive Typed Assembly IR** above our shared `Program` / `Scene` representation (`src/synccraft/dsl/assembly_ir.py`), without deprecating or altering any `v3.0`–`v3.2` `@rig` file.

### 2.1 Four Distinct Internal Representations

| Representation | Question Answered | Concrete Type |
| :--- | :--- | :--- |
| **Top-Down Layout (`@layout`)** | What master parameters, derived proportions, budgets, and envelopes govern the asset? | `LayoutDecl` (acyclic topological DAG + non-mutating `BudgetSpec` / `EnvelopeSpec` assertions) |
| **Component Definitions (`@component`)** | What reusable parametric part is this, and what public frames, material slots, and controls does it expose? | `ComponentDef` (encapsulated local namespace, typed `ParamDecl` bounds, `AttachmentFrameDecl`, `material_slots`, internal sub-rig) |
| **Assembly Ownership (`@assembly`)** | Which instances and organizational `subassembly` groups exist, and how are their frames mated? | `AssemblyDecl`, `SubassemblyDecl`, `ComponentInstanceDecl`, `ConnectionDecl` |
| **Resolved Scene & `SourceMap`** | What moves with what at runtime, and which component instance produced each exported bone/cube/UV? | Shared `Scene` (`WorldBone`, `WorldBox`, `WorldLocator`) + bidirectional `SourceMap` |

### 2.2 Rigid SE(3) Frame Mating Law (`connect`)

Every `@component` publishes named 6-DOF attachment frames ${}^{C}T_F = (\mathbf{R}_F, \mathbf{t}_F) \in SE(3)$ declared either in component coordinates (`pos`, `rot`) or anchored to a local box surface (`on_box`, `anchor`, `offset`, `rot`). Connecting child instance $C$ (mount frame $B$) to parent instance $P$ (socket frame $A$) evaluates:

$$
{}^{W}T_C(q) = {}^{W}T_P \, {}^{P}T_A \, M(q) \, \left({}^{C}T_B\right)^{-1}
$$

- **Mate Orientation Mode**:
  - `mode: align` *(default)*: $\mathbf{R}_{\text{mode}} = \mathbf{I}_3$ (coinciding right-handed frames `+X` East/Right, `+Y` Up, `+Z` Forward at $q = 0$).
  - `mode: oppose`: $\mathbf{R}_{\text{mode}} = \mathbf{R}_y(180^\circ) = \operatorname{diag}(-1, 1, -1)$ (opposing $+Z$ normals while keeping $+Y$ Up aligned).
- **Explicit Connection Types (`M(q)`) & Smart Runtime Bone Lowering**:
  1. **`fixed`**: $M = {}^{A}T_{B,0}$ constant. By default (`preserve_bone: false`), welds $C$'s root rigid elements directly onto $P$'s runtime bone (preventing bone bloat in Bedrock/GeckoLib while recording exact instance identity in `SourceMap`), unless $C$ is explicitly bound via `bind` or targeted by visibility/animation channels.
  2. **`revolute`**: 1-DOF rotation $q \in [\theta_{\min}, \theta_{\max}]$ about `axis` $\in \{x, y, z\}$. Allocates a child runtime bone at frame $A$'s pivot with rest orientation $\mathbf{R}_A \mathbf{R}_{\text{mode}} \mathbf{R}_{\text{trim}} \mathbf{R}_B^\top$.
  3. **`prismatic`**: 1-DOF linear translation $q \in [d_{\min}, d_{\max}]$ units along `axis` $\in \{x, y, z\}$. Allocates a child runtime bone at frame $A$'s pivot and lowers `@behavior` `slide` drives into `pose_position_deltas` exported natively to Bedrock `1.8.0` and Blockbench 5 `position` channels.
- **Mirror Completeness**: `mirror_x(twin)` on an `instance` reflects component geometry, conjugates all public `frame` rotations ($\det(\mathbf{R}) = +1$), flips `revolute` $Y/Z$ and `prismatic` $X$ signs/limits, and resolves mirrored parent sockets (`socket_l` <-> `socket_r` or reflected centerline frame).
- **Contact vs. Ownership Separation**: Frame mating never grants blanket cross-instance SAT immunity. Only the specific boundary boxes carrying `on_box` on frames $A$ and $B$ receive a declared `adjacency` contact edge (`reason="connect"`), audited under `strict` (`P1-02`) or `sculpted` (`ADR 0007`, with optional `sculpted_blend: true`).

### 2.3 Material Slots, `uv_policy: unique | shared`, Surface Basis & Units Separation

1. **Decoupled Units**: `@meta` defines `units_per_block: 16` independently of `texel_density` (`16`, `32`, `64`). Changing `texel_density` alters only UV texel resolution (`texel_density / units_per_block`), never geometry or tolerances.
2. **Material Slots**: Components declare `material_slots: [slot_a, slot_b]` bound per `instance` (`materials: { slot_a: palette_id }`).
3. **UV Ownership**:
   - `uv_policy: unique` *(default)*: Each instance receives distinct packed UV islands, supporting asymmetric world-space wear, moisture, and biofilm (`PersistentSurfaceState`).
   - `uv_policy: shared` *(opt-in)*: Unmirrored instances of the same component with identical evaluated dimensions and bound materials share a single packed UV chart (`WorldBox.uv_share_key`), reducing atlas footprint on repeated gears, struts, and battlements.
4. **Hybrid Procedural Coordinate Basis**: Intrinsic 3D solid material fields (`solid_wood`, `brushed_metal`, `keratin_scales`, `weathered_stone`) evaluate in **component-local rest coordinates** (`WorldBox.component_local_pos` / `component_local_rot`) so rotating an instance never swims or shears its grain, whereas posed cavity AO, lighting, and environmental weathering evaluate in **world coordinates**.

### 2.4 Adaptive Swept Motion Validation & Bedrock Attachables

1. **Adaptive Angular/Linear Step Subdivision**: `audit_behavior_clip` subdivides time intervals whenever inter-frame rotation exceeds `max_step_deg` (`10.0°` default) or translation exceeds `max_step_units` (`0.5u` default), up to `max_frames=144`, reporting `motion_audit_method: "adaptive_discrete_step"`, `evaluated_frame_count`, `measured_max_step_deg`, and `measured_max_step_units`.
2. **`SourceMap` & Bedrock `minecraft:attachable`**: Compilation emits a structured `SourceMap` (`stable_id`, `readable_path`, `component_name`, `runtime_bone_name`, `exported_boxes`, `assembly_provenance`) plus `export_bedrock_attachable` (`1.10.0` attachable JSON + `1.12.0` geometry + animations) for held/worn submodels.

### 2.5 Modular World-Model Supervision & Provenance Hygiene

1. **Four Compositional World-Model Tasks (`assembly_world_tasks.py`)**:
   - `plan_layout_and_interfaces`: Synthesize `@layout` + component `frame` contracts from asset specs.
   - `synthesize_component_from_contract`: Synthesize one encapsulated `@component` (`~400–900` tokens) against its `param`, `frame`, `material_slots`, and `envelope` contract.
   - `surgical_component_swap_and_repair`: Swap or repair a single `@component` / `connect` mount with **verified `1.0` edit locality** (`0` vertex drift on non-target instances).
   - `propagate_layout_and_instance_controls`: Predict world frame shifts, budget/envelope compliance, and independent instance poses under `@layout` parameter edits.
2. **Honest Decompiled Provenance**: Any assembly structure heuristically recovered from flat Bedrock/Blockbench imports is strictly labeled `assembly_provenance = "inferred_subtree"` (never `"authored"`).
