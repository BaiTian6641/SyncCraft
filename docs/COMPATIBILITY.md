# Export compatibility, axis conversion, and local verification gate

Status: **multi-backend geometry, PBR, and animation exporters/decompilers with CPU post-serialization 8-corner reconstruction & license-quarantine tests (`runtime_import_verified=False`); live GUI and game/mod-loader client imports are not yet verified**. JSON parsing alone is not an engine acceptance test. No complete official `.bbmodel` JSON Schema exists, according to the Blockbench wiki [1]. Do not report these fixtures as passing official schema validation.

## Implemented surfaces

| Surface | Implemented now (CPU-verified) | Required before a `runtime_verified` claim |
| --- | --- | --- |
| Blockbench 5.0 `.bbmodel` | Generic Model (`model_format: free`); separate `groups` list and UUID `outliner`; cuboids, fractional `size_step`, zero-thickness `plane`s, per-cube `rotate`, hierarchy, pivots, inflation; embedded PNG and per-face UVs; optional baked `@behavior` keyframe `animations` | Open all fixtures in pinned Blockbench 5.0 GUI, check dimensions/axes/UVs/keyframe signs, save, reload, compare world corners and hierarchy |
| `java_modern` | Static element JSON using `rotation.origin` and `x/y/z`; complete world transform flattening; per-face atlas UVs normalized to `0..16` | Pin Minecraft Java 1.21.11+, assemble a temporary resource pack, load/reload and inspect game logs and screenshots |
| `java_legacy` | Conservative single-axis final rotations in `{-45,-22.5,0,22.5,45}`; final composed matrices verified | Pin one exact legacy version and test live |
| Bedrock Entity (`bedrock_entity`) | `1.12.0` `minecraft:geometry` + `1.8.0` `animations` with **exact left-handed X-axis conversion** (`-(origin_x + size_x)`, negated pivot/locator X, negated X/Y rest & keyframe rotations, negated X position), zero-thickness `plane`s, per-cube `rotate`, 3D `locators`, and 8-corner world vertex re-audit (`< 1e-4`, `runtime_import_verified=False`) | Pin target Bedrock / mod-loader build, verify handedness and Catmull-Rom/Molang playback in live client & Blockbench Bedrock Entity mode |
| GeckoLib (`geckolib`) | `1.12.0` `minecraft:geometry` + `1.8.0` `animations` (`"keyframes"` or `"molang"`) with left-handed X-axis conversion, `target_profile == "geckolib"` check, and 8-corner post-serialization re-audit (`runtime_import_verified=False`) | Separate pinned mod-loader/GeckoLib runtime fixtures and animation playback verification |
| LabPBR 1.3 (`labpbr_1_3`) & glTF 2.0 (`gltf_2_0`) | Unbaked albedo + `_n.png` (DirectX tangent normal, AO, height) + `_s.png` (smoothness, $F_0$/metal code `230..255`, porosity/SSS, emissive `0..254` with `255` sentinel) and `.gltf` (`ORM` + `KHR_materials_iridescence`) | Pin Minecraft + Iris + LabPBR 1.3 shader pack and Khronos glTF Validator / Blender for visual & runtime acceptance |
| External Asset Intake & License Gate | Zero-`eval` `.bbmodel`, Java `.json`, and Bedrock `.geo.json` + `.animation.json` decompilers with dual-track provenance gate (`clean_room_permissive` vs `research_nc` allowing `CC-BY-NC-3.0`/`4.0`, rejecting `CC-BY-ND` and `NoAI`/`do-not-train` notices, and defaulting unlabeled local files to `NOASSERTION`) | Audit external corpus provenance manifests before any benchmark ingestion |
| Resource-pack wrapper & `@ergonomics` | Not generated (`@ergonomics` raises `UnsupportedFeature`) | Implement `P1-04` hand/grip & display-slot solver and `pack.mcmeta` dispatch wrapper |

## Geometry & handedness decisions

1. **Java Flattening**: The shared scene stores proper right-handed world rotation matrices (`Rz @ Ry @ Rx`). `export_java` recovers exact world Euler triplets after full hierarchy composition [2] and checks `[-16, 32]` bounds.
2. **Bedrock / GeckoLib Left-Handed X Reflection (`C5` / `M16`)**: Unlike internal right-handed `Block-DSL`, Bedrock `1.12.0` geometry and `1.8.0` animations reflect the X axis. Both outbound `export_bedrock` / `export_geckolib` and inbound `_decompile_bedrock_geojson` / `decompile_bedrock_animation_to_behavior` apply the genuine X-reflection (`origin_x = -(x_rh + w)`, negated bone pivot X & locator X, negated X/Y Euler rotations, and negated X translation keyframes) rather than omitting conversion on both sides. Asymmetric multi-bone rotated rigs are tested end-to-end (`tests/test_bedrock_and_geckolib.py`).
3. **Fractional Sizes, `0`-Thickness Planes & Per-Cube Rotation (`H6`)**: Authored community rigs frequently combine fractional `.5`/`.25` cube sizes, `0`-thickness flat ear/fin planes, per-cube pivot/rotation, and Catmull-Rom / Molang animations. `Block-DSL v3.2` preserves these natively via `@meta.size_step`, `plane`, per-box `rotate`, and `audit_profile="authored"`.
4. **Cullface Reflection Parity**: `cullface` names stay invariant under rotation [2] and swap `east <-> west` at odd `mirror_x` reflection depths.

## Run the local import gate (`P0-02` / `P3-02`)

1. Record OS, exact Blockbench version, exact Minecraft / mod-loader version, and shader/plugin versions. Never record `"latest"`.
2. Compile bundled examples plus asymmetric multi-axis rotated and nested-mirror fixtures.
3. Open `.bbmodel` and `.geo.json` in Blockbench; verify pivots, left/right asymmetry, `0`-thickness planes, UVs, and animation keyframe directions; save, reopen, and compare reconstructed 8-corner coordinates.
4. Load Java / LabPBR / Bedrock / GeckoLib packs in the pinned game client and record logs and observations under `artifacts/validation/` before setting `runtime_import_verified = True`.

## Primary sources

[1] Blockbench official wiki, “[The .bbmodel format](https://blockbench.net/wiki/docs/bbmodel/)”. Accessed 2026-11-13.

[2] Mojang, “[Minecraft Java Edition 1.21.11](https://www.minecraft.net/en-us/article/minecraft-java-edition-1-21-11)”, sections “Block Models” and “Resource Pack Versions 70.0 through 75.0”. Accessed 2026-11-13.

[3] Blockbench upstream source, [`js/formats/bbmodel.js`](https://github.com/JannisX11/blockbench/blob/master/js/formats/bbmodel.js). Accessed 2026-11-13.

[4] Blockbench official reference docs, [Group](https://web.blockbench.net/docs/classes/custom_group.Group.html) and [Cube](https://web.blockbench.net/docs/classes/custom_cube.Cube.html). Accessed 2026-11-13.
