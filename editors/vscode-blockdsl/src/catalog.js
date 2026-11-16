// @ts-check
"use strict";

/**
 * Canonical SyncCraft Block-DSL v3.2 language catalog.
 * Kept in 1:1 parity with `src/synccraft/dsl/ast_nodes.py` and `src/synccraft/dsl/grammar.lark`
 * (enforced by `tests/test_vscode_blockdsl_extension.py`).
 */

const TOP_SECTIONS = [
  {
    name: "@meta",
    detail: "Required header: archetype, target_profile, texel_density, size_step, audit_profile",
    insertText: "@meta { archetype: ${1|entity,block_multipart,weapon_3d|}, target_profile: ${2|bedrock_entity,geckolib,bbmodel_5,java_modern,java_legacy|}, texel_density: ${3|16,32,64|} }",
    doc: "Declares target format, discrete voxel grid (`size_step`), and collision contract (`strict` vs `sculpted`). Must be the first `@` section."
  },
  {
    name: "@textures",
    detail: "Optional external texture sheet bindings",
    insertText: "@textures {\n  ${1:sheet_main}: image(\"${2:textures/entity/model.png}\", ${3:64}, ${4:64})\n}",
    doc: "Optional external image sheet declarations (`<id>: image(\"<path>\", width, height)`)."
  },
  {
    name: "@palette",
    detail: "Required perceptual OKLCh color ramps (>= 1)",
    insertText: "@palette {\n  ${1:primary_mat}: ramp(base_lch=[${2:0.52}, ${3:0.08}, ${4:235}], steps=${5:8}, hue_shift=[${6:-4}, ${7:4}])\n}",
    doc: "Defines perceptual OKLCh color ramps referenced by every `box` and `plane` `material:` field."
  },
  {
    name: "@rig",
    detail: "Required SE(3) skeletal hierarchy (`root bone ...`)",
    insertText: "@rig {\n  root bone ${1:root_body} {\n    pivot_local: [0, ${2:8}, 0]\n    box ${3:core_box} { size: [${4:8, 8, 8}], material: ${5:primary_mat} }\n  }\n}",
    doc: "Right-handed Y-up SE(3) bone tree (`+X` East/Right, `+Y` Up, `+Z` South/Front)."
  },
  {
    name: "@materials",
    detail: "Optional 3D manifold physical surface processes (10 families)",
    insertText: "@materials {\n  ${1:primary_mat}: process(family=${2|keratin_scales,brushed_metal,painted_metal,weathered_stone,solid_wood,woven_fabric,polymer_dielectric,patterned_skin,lichen_biofilm,thin_film_iridescence|}, roughness=${3:0.35}, wear=${4:0.25})\n}",
    doc: "Compiles 3D Worley/fBm solid fields, manifold Gray-Scott reaction-diffusion patterns, Airy thin-film interference, and GGX reflectance."
  },
  {
    name: "@style",
    detail: "Optional optical encoding (`vanilla`, `labpbr_1_3`, `gltf_2_0`)",
    insertText: "@style { target_encoding: ${1|vanilla,labpbr_1_3,gltf_2_0|}, quantize_ramp: ${2|true,false|}, bevel_highlight: ${3:0.08} }",
    doc: "Selects Vanilla baked atlas, LabPBR 1.3 (`_n` + `_s`), or glTF 2.0 metallic-roughness output."
  },
  {
    name: "@muscles",
    detail: "Optional reduced-order Zajac/Hill muscle-tendon actuators",
    insertText: "@muscles {\n  muscle ${1:flexor_l} {\n    origin_bone: ${2:torso}, insertion_bone: ${3:limb_l}, action_axis: ${4|x,y,z|}, moment_arm_deg: ${5:-35}\n  }\n}",
    doc: "Models active force-length + passive parallel-elastic torque equilibrium, biarticular `tendon_coupling`, and Box-UV-safe `belly_box` `bulge_inflate`."
  },
  {
    name: "@kinematics",
    detail: "Optional N-bone `ik_chain`, `look_chain`, and damped `spring` dynamics",
    insertText: "@kinematics {\n  ik_chain ${1:leg_ik_l} { bones: [${2:thigh_l}, ${3:shin_l}], effector: [0, -6, 0.5], hinge_axis: x }\n}",
    doc: "Declares analytic 2-bone / CCD N-bone IK chains, distributed gaze `look_chain`s, and inertial lag `spring`s."
  },
  {
    name: "@behavior",
    detail: "Optional multi-modal animation `clip`s and reactive `state`s",
    insertText: "@behavior {\n  clip ${1:idle_loop} {\n    duration_s: ${2:2.0}\n    track ${3:child_bone}: ${4|wave,flap,hinge,impulse,spin|}(axis=${5|x,y,z|}, amp_deg=${6:15}, freq_hz=${7:0.5})\n  }\n}",
    doc: "Combines harmonic/`spin` tracks, Catmull-Rom/linear `key` splines, Hill `muscle_drive`s, `< 0.02u`-slip `foot_cycle` IK, and `state` modulations."
  },
  {
    name: "@controller",
    detail: "Optional reactive state-machine transitions",
    insertText: "@controller {\n  initial: ${1:idle_loop}\n  ${1:idle_loop} -> ${2:active_state} when ${3:speed} > ${4:0.5}, blend ${5:0.25}\n}",
    doc: "Declarative state transitions (`<from> -> <to> when <var> <op> <threshold>, blend <seconds>`)."
  }
];

const ARCHETYPES = ["entity", "weapon_3d", "block_multipart", "item_extruded"];
const PROFILES = ["bedrock_entity", "geckolib", "java_modern", "java_legacy", "bbmodel_5"];
const ANIMATED_PROFILES = ["bedrock_entity", "geckolib", "bbmodel_5"];
const AUDIT_PROFILES = ["strict", "sculpted", "authored"];
const DSL_VERSIONS = ["\"3.2\"", "\"3.1\"", "\"3.0\""];
const ALLOWED_SIZE_STEPS = ["1.0", "0.5", "0.25", "0.125", "0.0625", "0.03125", "0.1", "0.05", "0.01", "0.001"];
const FACES = ["north", "south", "east", "west", "up", "down"];
const AXES_3D = ["x", "y", "z"];
const MIRROR_AXES = ["x", "z"];
const LOCATOR_ROLES = ["socket", "ik_effector", "vfx_emitter", "camera_focus", "hitbox", "rider_seat"];
const TRACK_LAWS = ["wave", "flap", "hinge", "impulse", "spin"];
const KEYFRAME_CHANNELS = ["rotation", "position", "scale"];
const KEYFRAME_INTERPOLATIONS = ["linear", "catmullrom"];
const TARGET_ENCODINGS = ["vanilla", "labpbr_1_3", "gltf_2_0"];
const UNSUPPORTED_KEYWORDS = ["@shade", "@motion", "@ergonomics", "extrude"];

const MATERIAL_FAMILIES = [
  { name: "painted_metal", detail: "Dielectric enamel clearcoat over metal + Archard edge chipping", params: "roughness=0.34, metalness=0.25, coating=0.6, wear=0.45" },
  { name: "brushed_metal", detail: "Conductive GGX microfacet + anisotropic brush tangent grooves", params: "roughness=0.24, metalness=0.92, anisotropy=2.0, wear=0.3" },
  { name: "polymer_dielectric", detail: "Fresnel dielectric (F0 from ior) + subtle mold/scuff roughness", params: "roughness=0.45, ior=1.52" },
  { name: "woven_fabric", detail: "Orthogonal warp/weft thread relief + asperity sheen + porosity", params: "roughness=0.80, subsurface=0.22, porosity=0.4" },
  { name: "solid_wood", detail: "3D coaxial growth rings + axial fiber anisotropy", params: "roughness=0.66, anisotropy=2.2, scale=1.0" },
  { name: "weathered_stone", detail: "Multi-scale 3D Worley/fBm mineral clasts + porous cavity AO", params: "roughness=0.86, porosity=0.48, biofilm=0.2" },
  { name: "keratin_scales", detail: "Oriented imbricated 3D scales + Gray-Scott Turing pigmentation", params: "roughness=0.34, coating=0.35, rd_feed=0.038, rd_kill=0.061" },
  { name: "patterned_skin", detail: "Manifold-continuous Gray-Scott Turing spots/stripes + subsurface", params: "roughness=0.58, subsurface=0.35, rd_feed=0.036, rd_kill=0.061" },
  { name: "lichen_biofilm", detail: "Colony growth favoring upward/occluded porous surfaces", params: "roughness=0.88, porosity=0.6, biofilm=0.75" },
  { name: "thin_film_iridescence", detail: "Exact Airy dielectric thin-film interference (`film_nm`, `ior`)", params: "roughness=0.14, ior=1.62, film_nm=450, emissive=0.35" }
];

const MATERIAL_PROCESS_ARGS = [
  { name: "family", range: "10 physics families", defaultVal: "required" },
  { name: "roughness", range: "[0.02, 1.0]", defaultVal: "0.35" },
  { name: "metalness", range: "[0.0, 1.0]", defaultVal: "0.0" },
  { name: "coating", range: ">= 0.0 (0..1)", defaultVal: "0.0" },
  { name: "ior", range: ">= 1.0 (1.3..2.4)", defaultVal: "1.5" },
  { name: "scale", range: "> 0.0", defaultVal: "1.0" },
  { name: "seed", range: "int", defaultVal: "101" },
  { name: "anisotropy", range: "> 0.0 (0.5..4.0)", defaultVal: "1.0" },
  { name: "subsurface", range: "[0.0, 1.0]", defaultVal: "0.0" },
  { name: "porosity", range: "[0.0, 1.0]", defaultVal: "0.08" },
  { name: "emissive", range: "[0.0, 2.0]", defaultVal: "0.0" },
  { name: "rd_feed", range: "[0.002, 0.2]", defaultVal: "0.036" },
  { name: "rd_kill", range: "[0.002, 0.2]", defaultVal: "0.062" },
  { name: "film_nm", range: ">= 0.0 (200..750 nm)", defaultVal: "380.0" },
  { name: "wear", range: ">= 0.0", defaultVal: "0.35" },
  { name: "oxidation", range: ">= 0.0", defaultVal: "0.25" },
  { name: "biofilm", range: ">= 0.0", defaultVal: "0.20" },
  { name: "weld_group", range: "IDENT", defaultVal: "none" }
];

const FLUSH_CONTACT_PRESETS = [
  { label: "place-top (+Y flush)", detail: "Stack flush ON TOP of base_box (anchor=[0, 1, 0], align=[0, -1, 0])", insertText: "place(base_box=${1:base_box}, anchor=[0, 1, 0], align=[0, -1, 0])" },
  { label: "place-bottom (-Y flush)", detail: "Hang flush BELOW base_box (anchor=[0, -1, 0], align=[0, 1, 0])", insertText: "place(base_box=${1:base_box}, anchor=[0, -1, 0], align=[0, 1, 0])" },
  { label: "place-front (+Z flush)", detail: "Extend FORWARD (+Z snout/prow: anchor=[0, 0, 1], align=[0, 0, -1])", insertText: "place(base_box=${1:base_box}, anchor=[0, 0, 1], align=[0, 0, -1])" },
  { label: "place-back (-Z flush)", detail: "Extend BACKWARD (-Z tail/stern: anchor=[0, 0, -1], align=[0, 0, 1])", insertText: "place(base_box=${1:base_box}, anchor=[0, 0, -1], align=[0, 0, 1])" },
  { label: "attach-left (-X shoulder)", detail: "Attach child bone flush to -X side (anchor=[-1, 0, 0], align=[1, 0, 0])", insertText: "attach(parent_box=${1:parent_box}, anchor=[-1, 0, 0], align=[1, 0, 0])" },
  { label: "attach-right (+X side)", detail: "Attach child bone flush to +X side (anchor=[1, 0, 0], align=[-1, 0, 0])", insertText: "attach(parent_box=${1:parent_box}, anchor=[1, 0, 0], align=[-1, 0, 0])" }
];

const HOVER_DOCUMENTATION = {
  "@meta": "**`@meta`** — Required file header.\n\nFields:\n- `archetype`: `entity` | `block_multipart` | `weapon_3d`\n- `target_profile`: `bedrock_entity` | `geckolib` | `bbmodel_5` | `java_modern` | `java_legacy`\n- `texel_density`: positive integer (`16`, `32`, `64`)\n- `size_step` *(optional)*: voxel grid quantum (`1.0`, `0.5`, `0.25`, `0.125`, ...)\n- `audit_profile` *(optional)*: `strict` (default `<= 0.5u` inset) | `sculpted` (`ADR 0007` sloped miters + `blend_with`) | `authored` (decompiled human imports only)",
  "@palette": "**`@palette`** — Perceptual OKLCh color ramps.\n\nSyntax: `<id>: ramp(base_lch=[L, C, H], steps=2..256, hue_shift=[lo_deg, hi_deg], emissive=false)`\n- `L` in `[0.0, 1.0]`, `C >= 0.0`, `H` in degrees.",
  "@rig": "**`@rig`** — Shared right-handed Y-up SE(3) skeletal hierarchy (`det(R) = +1`).\n\nAxes: `+X` East/Right, `+Y` Up (ground at `Y = 0`), `+Z` South/Front.",
  "@materials": "**`@materials`** — Stateful 3D manifold surface physics (`ADR 0003`).\n\nEvaluates 3D solid Worley/fBm fields, manifold Gray-Scott PDEs across `weld_group` seams, Airy thin-film interference, and GGX microfacet reflectance.",
  "@style": "**`@style`** — Output texture encoding.\n\n- `target_encoding`: `vanilla` | `labpbr_1_3` | `gltf_2_0`\n- `quantize_ramp`: `true` | `false`\n- `bevel_highlight`: `>= 0.0` (default `0.08`)\n- `view_deg`: `[0.0, 89.0]` (default `20.0`)",
  "@muscles": "**`@muscles`** — Reduced-order Zajac (1989) / Hill muscle-tendon units (`ADR 0006`).\n\nSolves active Gaussian force-length + passive parallel-elastic torque equilibrium with optional biarticular `tendon_coupling: [bone, axis, ratio]` and Box-UV-safe `belly_box` `bulge_inflate` (`0..0.35`).",
  "@kinematics": "**`@kinematics`** — Inverse kinematics & secondary motion.\n\nSupports 2-bone / CCD N-bone `ik_chain` (`2..16` direct parent->child bones), distributed `look_chain` (`1..12` bones), and second-order damped `spring`s.",
  "@behavior": "**`@behavior`** — Multi-modal motion `clip`s and reactive `state`s.\n\nAudited across 12 swept frames via `audit_behavior_clip` (`0` non-exempt OBB collisions, `< 0.02u` `foot_cycle` stance slip).",
  "@controller": "**`@controller`** — Reactive state transition graph.\n\nSyntax: `<from_clip_or_state> -> <to_clip_or_state> when <sensor> <op> <value>, blend <0..10>`.",
  "audit_profile": "**`audit_profile`** (`strict` | `sculpted` | `authored`)\n\n- **`strict`** *(default)*: Flush `P1-02` contacts (`gap <= 1e-4u`, `inset <= 0.5u`), zero non-exempt SAT overlap (`<= 1e-7u`). Required for Gate `G2`.\n- **`sculpted`** (*ADR 0007*): Enables sloped organic snouts, `> 60°` horns, imbricated dorsal armor, and chamfered/domed vehicles (`<= 0.85 * t_med` overlap on declared/same-bone/parent-child/`blend_with` seams, mandatory exterior protrusion `> min(0.25u, 0.15 * t_thin)`, `0` Z-fighting, `0` sibling-bone immunity).",
  "blend_with": "**`blend_with: [<box_id>, ...]`** (*ADR 0007*)\n\nDeclares an explicit surface-blending miter contract with another expanded `box` or `plane` under `audit_profile: sculpted` (supports `mirror_x` centerline pairs such as `prow_r__plate_l` and cyclic `radial` domes such as `dome_rib_r1__arch`). Still enforces exterior protrusion (`0` swallowed boxes) and `0` coplanar Z-fighting.",
  "attach": "**`attach(parent_box=B, anchor=[ax, ay, az], align=[gx, gy, gz], offset=[ox, oy, oz], detached=false)`**\n\nPlaces child bone pivot at `c_B + R_B * (0.5 * s_B * anchor) + offset` and shifts the child's primary box by `-0.5 * s_child * align`. For flush contact, set `align = -anchor` and `offset = [0, 0, 0]` (`offset` bounded to `[-0.5, 0.5]` when `detached=false`).",
  "joint": "**`joint(parent=B, at=[dx, dy, dz], align=[gx, gy, gz], detached=false)`**\n\nPlaces child bone pivot at explicit offset `at` from `B`'s center without the `[-0.5, 0.5]` `offset` cap (ideal for `radial` hubs and mechanical axles).",
  "place": "**`place(base_box=B, anchor=[ax, ay, az], align=[gx, gy, gz], offset=[ox, oy, oz])`**\n\nRequired on every 2nd+ `box`/`plane` inside a bone. For flush face-to-face contact (`0` gap, `0` overlap), set **`align = -anchor`** along the contact normal!",
  "mirror_x": "**`mirror_x(<twin_bone_name>)`**\n\nReflects the entire bone subtree across `X = 0` (`S = diag(-1, 1, 1)`), conjugating Euler angles `(rx, ry, rz) -> (rx, -ry, -rz)` so `det(R) = +1`.\n\n**Expanded child naming**: Descendants are prefixed with **double underscore** `<twin>__<child_name>`!",
  "radial": "**`radial(count=K, axis=x|y|z)`**\n\nExpands `K - 1` rotational clones `<bone>_r1 .. <bone>_r{K-1}` (descendants `<bone>_r{k}__<child_name>`) spaced by `360° / K` around `axis`.",
  "chain": "**`chain <name> attach(...) segments=K taper=T { ... }`**\n\nExpands a `1`-indexed nested serial bone chain `<name>_1 -> ... -> <name>_K` with boxes `<name>_{s}_<box_name>` (single underscore `_`), scaling sizes by `taper^(s-1)`.",
  "spin": "**`spin(axis=x|y|z, amp_deg=<-1440..1440>, freq_hz=..., phase_deg=0, bias_deg=0)`** (*ADR 0007*)\n\nContinuous linear mechanical rotation `bias_deg + amp_deg * (t * freq_hz + phase_deg / 360)` for `360°` wheels, propellers, windmill rotors, turrets, and meshed multi-ratio gears (e.g. `+360°` wheel + `-720°` `2:-1` pinion). Exports gimbal-safe unwrapped keyframes & native Molang `query.anim_time`.",
  "foot_cycle": "**`foot_cycle(stride=<0..24>, lift=<0..16>, phase_deg=0)`**\n\nWorld-anchored $C^1$ locomotion stance/swing trajectory solved via `ik_chain`, guaranteeing `< 0.02` units horizontal stance foot slip.",
  "inflate": "**`inflate: <delta>`**\n\nExpands physical OBB half-extents `h = 0.5 * size + inflate` without changing discrete UV island texel dimensions (`>= 0.0` in `strict` and `sculpted` modes).",
  "joint_limits": "**`joint_limits: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]`**\n\nPer-axis angular bounds in degrees. Every axis interval must contain rest pose `0` (`-180 <= min <= 0 <= max <= 180`)."
};

module.exports = {
  TOP_SECTIONS,
  ARCHETYPES,
  PROFILES,
  ANIMATED_PROFILES,
  AUDIT_PROFILES,
  DSL_VERSIONS,
  ALLOWED_SIZE_STEPS,
  FACES,
  AXES_3D,
  MIRROR_AXES,
  LOCATOR_ROLES,
  TRACK_LAWS,
  KEYFRAME_CHANNELS,
  KEYFRAME_INTERPOLATIONS,
  TARGET_ENCODINGS,
  UNSUPPORTED_KEYWORDS,
  MATERIAL_FAMILIES,
  MATERIAL_PROCESS_ARGS,
  FLUSH_CONTACT_PRESETS,
  HOVER_DOCUMENTATION
};
