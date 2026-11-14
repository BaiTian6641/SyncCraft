# Export compatibility and local verification gate

Status: **multi-backend geometry, PBR, and animation exporters with CPU post-serialization numerical & roundtrip reconstruction tests (`runtime_import_verified=False`); live GUI and game/mod-loader imports are not yet verified**. JSON parsing alone is not an engine acceptance test. No complete official `.bbmodel` JSON Schema exists, according to the Blockbench wiki [1]. Do not report these fixtures as passing official schema validation.

## Implemented surfaces

| Surface | Implemented now | Required before a release claim |
| --- | --- | --- |
| Blockbench 5.0 `.bbmodel` | Generic Model (`model_format: free`); separate `groups` list and UUID `outliner`; cuboids, hierarchy, pivots, local rotations, inflation; embedded PNG and per-face UVs; stable UUIDs; optional baked `@behavior` bone-group keyframe `animations` when `program` is passed (`program=None` preserves byte-identical 7-key static output) | Open all fixtures in the selected Blockbench build, check dimensions/axes/UVs/keyframe signs, save, reload, compare world corners and hierarchy |
| `java_modern` | Static element JSON using `rotation.origin` and `x/y/z`; complete world transform flattening; per-face atlas UVs normalized to 0..16 | Pin Minecraft 1.21.11 or a specifically verified successor, assemble a temporary resource pack, load/reload and inspect game logs and images |
| `java_legacy` | Conservative single-axis final rotations in `{-45,-22.5,0,22.5,45}`; final composed matrices are checked | Pin one exact legacy version and test it; this profile intentionally does not model every intermediate version's expanded capabilities |
| Bedrock / Cobblemon (`bedrock_entity`) | `1.12.0` `minecraft:geometry` (cumulative unrotated bone pivots, rest rotations, cubes, `inflate`, per-face or Box-UVs, 3D semantic `locators`, visible bounds) + `1.8.0` `animations` (baked from `evaluate_behavior_clip` across Hill `@muscles`, tendon pantographs, and stance-IK, or `"molang"` mode) + post-serialization 8-corner world vertex re-audit (`<= 1e-6`, `runtime_import_verified=False`) | Pin target Bedrock / Cobblemon mod version, verify handedness/axis sign conventions in live client & Blockbench Bedrock Entity format, and record fixture logs |
| GeckoLib (`geckolib`) | `1.12.0` `minecraft:geometry` + `1.8.0` `animations` (`"keyframes"` or `"molang"`) with strict `target_profile == "geckolib"` check and post-serialization 8-corner re-audit (`runtime_import_verified=False`) | Separate selected-mod-loader/GeckoLib version fixtures, expression and animation runtime tests; do not assume full runtime interchangeability with Bedrock |
| LabPBR 1.3 (`labpbr_1_3`) & glTF 2.0 (`gltf_2_0`) | Unbaked albedo + `_n.png` (DirectX tangent normal, AO, height) + `_s.png` (smoothness, $F_0$/metal code `230..255`, porosity/SSS, emissive `0..254` with `255` sentinel) and `.gltf` (`ORM` + `KHR_materials_iridescence`) | Pin Minecraft + Iris + LabPBR 1.3 shader pack and glTF Validator / Blender / Khronos viewer for visual & runtime acceptance |
| Resource-pack wrapper | Not generated | Version-specific `pack.mcmeta`, asset namespaces, atlas locations and modern item dispatch models |
| Grip/display transforms | Not generated | Implement `@ergonomics`, hand geometry audit and display-slot transforms |

The parser rejects unimplemented `extrude`, `@shade`, `@ergonomics` and legacy `@motion` (`Block-DSL v3.1` uses typed `@materials`/`@style` and `@muscles`/`@kinematics`/`@behavior`). Exporters never turn an incompatible target profile into a silent fallback.

## Geometry decisions

The shared scene stores proper world rotation matrices in `Rz @ Ry @ Rx` order. The Java exporter recovers the corresponding Euler triplet after the whole hierarchy has been composed. It never copies a child's local Euler angles into a flattened model. This follows the X, then Y, then Z application order in Mojang's 1.21.11 release notes [2]. Modern output uses only the `x/y/z` notation because Mojang says the older `axis/angle` notation takes precedence if both are present [2].

Each Java element is re-centered on its world-space OBB center and rotated around that center. Inflation expands the half extents. From/to bounds and the final rotated corners are both checked against the plan's `[-16,32]` envelope. The rotated-corner check is an intentionally conservative SyncCraft policy, not a claim that Mojang imposes that additional check in all releases. A model rejected for that check may still be representable by the game; changing this policy requires an ADR and tests.

The Blockbench and Bedrock/GeckoLib hierarchies use cumulative **unrotated** local translations for stored group `origin` / bone `pivot` values, local bone rotations, and cuboid coordinates relative to those stored pivots. Substituting compiled world pivots into group origins would rotate descendant translation twice. Both the unit tests and `bedrock_gecko_exporter._reaudit_serialized_bedrock_geometry` independently reconstruct the hierarchy from serialized JSON and compare all 8 corners of every cuboid against the shared scene (`<= 1e-6`). Real Blockbench and target client imports are still required to confirm runtime coordinate/handedness contracts (`runtime_import_verified=False`).

When `program=None` (default), `export_bbmodel` emits no animation keyframes. When `program` with `@behavior` clips is passed, `export_bbmodel` bakes posed bone delta Euler keyframes (`channel: "rotation"`) per group UUID; live Blockbench 5.0 GUI keyframe sign verification [1] remains required before claiming `runtime_verified`.

`cullface` names remain unchanged under rotation, following Mojang's rule that face directions retain their original names after element rotation [2]. Reflection is different: source east/west cullfaces exchange on `mirror_x` derivatives. The Scene keeps the source cullface selection plus its `mirrored` flag; exporters perform this exchange once. The starter does not prove a requested cullface is visually safe for a multipart asset; the local import gate must check those assets. Asymmetric texture artwork has no mirrored-UV guarantee yet: the bundled atlas uses uniform base colors, so future seam/stripe/stamp work must add directional face/U/V fixtures.

Public exporters validate finite positive cuboid extents, proper rotations, unique/topological names, and agreement between local and world transforms. This prevents directly constructed or mutated Scenes from auditing one geometry and exporting another.

The optional `atlas` argument is a dictionary:

```python
{
    "width": 64,
    "height": 32,
    "data_url": "data:image/png;base64,...",  # mandatory for bbmodel embedding
    "faces": {"box_id": {"north": [u0, v0, u1, v1], ...}},
}
```

All six faces of all expanded boxes, including mirrors, must be present. `.bbmodel` gets pixel UV coordinates and texture index 0. Java gets UV coordinates scaled independently by `16/width` and `16/height`. When no atlas is provided, the `.bbmodel` exporter creates an original white 1px embedded texture to keep faces visible. The Java exporter references an external texture resource location; the caller must actually package that texture.

## Run the local import gate

1. Record OS, Blockbench exact version, Minecraft exact version, and relevant plugins. Do not substitute “latest”.
2. Compile the bundled examples and inspect their audit JSON. Also create a nested multi-axis test and a gimbal-lock test.
3. Open the `.bbmodel` in Blockbench. Verify the root pivot, child pivots, mirrored branch naming, cuboid count, measured sizes, texture resolution and all face mappings.
4. Save and reopen the project. Parse the saved file independently and compare world corners with the source scene. Keep a small versioned fixture and sanitized log, not a large binary screenshot corpus.
5. For Java, create a temporary resource pack appropriate to the pinned version. Place block textures in the correct atlas, and add the required modern item-model dispatch wrapper if displaying an item. A model JSON alone is not a complete resource pack.
6. Load/reload the pack and inspect logs, four standard views, culling, interpolation and display behavior. Record actual observations; a numerical reconstruction test cannot prove runtime compatibility.
7. Commit the engine/build IDs, fixture hashes, logs and result to `evidence/`. Only then update the capability table to `runtime_verified` for that exact target.

## Primary sources checked for this starter

[1] Blockbench official wiki, “[The .bbmodel format](https://blockbench.net/wiki/docs/bbmodel/)”. Confirmed 5.0 groups/outliner split, per-texture UV dimensions for Generic Models, keyframe sign changes, and absence of a complete specification. Accessed 2026-11-13.

[2] Mojang, “[Minecraft Java Edition 1.21.11](https://www.minecraft.net/en-us/article/minecraft-java-edition-1-21-11)”, sections “Block Models” and “Resource Pack Versions 70.0 through 75.0”. Confirmed multi-axis float rotation fields, order/precedence and atlas changes. Accessed 2026-11-13.

[3] Blockbench upstream source, [`js/formats/bbmodel.js`](https://github.com/JannisX11/blockbench/blob/master/js/formats/bbmodel.js). Its project serializer/loader confirms separate `elements`, `groups`, `outliner`, embedded texture source and per-face numeric texture indices. This link is a moving branch; the local agent must pin the matching upstream commit for its compatibility fixture. Accessed 2026-11-13. No upstream source files or game assets are redistributed in this starter.

[4] Blockbench official reference docs, [Group](https://web.blockbench.net/docs/classes/custom_group.Group.html) and [Cube](https://web.blockbench.net/docs/classes/custom_cube.Cube.html), for the editor object boundaries. Accessed 2026-11-13.
