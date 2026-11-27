# SyncCraft TexGraph & 3D Modeling Suite — Commercial Industry Roadmap (`v10 → v13`)

> **Living Architectural Master Plan** — Persistent source of truth surviving context compression across sessions.
> Benchmarks SyncCraft against **Pixel Composer 1.22+**, **Adobe Substance 3D Designer & Painter**, **Marmoset Toolbag 5**, **Unreal Engine 5.7 (Substrate & Triplanar)**, **Unity URP/HDRP Shader Graph**, and **Autodesk Maya LookdevX / MaterialX + OpenPBR**.

---

## 1. Baseline Audit of `TexGraph v9` (`SyncCraft v46.0` @ `e4c0018`)

| Dimension | `TexGraph v9` Status | Quantitative Audit Finding | Commercial Gap to Close |
|---|---|---|---|
| **DAG Execution Core** | Strong | `140` nodes, `19` categories, `22` subgraphs, `21` presets, Kahn topological sort, `GraphCycleError`, SHA-256 content-addressed memoization (`0 ms` cache hits), `.texdsl` round-trip compiler/decompiler | Add GUI node metadata (`pos`, `label`, `collapsed`, `bypassed`, `preview_pinned`), automatic Sugiyama layered `auto_layout()`, and partial dirty-subgraph evaluation for 60 fps interactive GUI dragging. |
| **GUI Parameter Schema** | **Critical Gap Found** | Only **`14 / 140` (10%)** nodes had valid `ParamSchema`; **`15` core nodes** accidentally bound `ParamSchema` tuples into `outputs=` instead of `params=`; **`0`** port-compatibility rules or socket colors | Upgrade to **Schema v4.0**: `100%` of nodes (`144/144`) expose typed `ParamSchema` (`widget`, `default`, `min`, `max`, `step`, `choices`, `group`, `animatable`, `visible_if`, `tooltip`), `0` misplaced schemas, `PORT_COLORS` & `PORT_COMPATIBILITY` matrix. |
| **UV-Geometric Edge Solving & Cross-Seam Continuity** | Partial | 3D `world_pos`/`world_normal` + `noise_3d` + `organic_domain_warp_3d`; no universal triplanar node, no 3D UV-edge adjacency graph, no cross-seam healing, no connected block-edge solver | Build **3D UV-Edge Adjacency Solver** + orthonormal `world_tangent`/`world_bitangent` in `mesh_baker.py`, plus `triplanar_project` (UE 5.7 World/Local/Pre-Skinned + dithered option), `uv_seam_heal`, and `block_edge_transition`. |
| **Precomputed Shadow & Special PBR Sharing** | Partial | `9` `MATERIAL` bus ports (`albedo`, `height`, `normal`, `roughness`, `metallic`, `ao`, `emissive`, `sss`, `porosity`); LabPBR 1.3 + Bedrock MER + basic baked albedo | Add true **3D Ray-vs-OBB Precomputed `baked_shadow`**, **Hemisphere `bent_normal`**, **Volume `thickness` (SSS)**, and **`curvature_3d`** to `mesh_baker.py`; extend `MATERIAL_BUS_PORTS` to **12 channels** (`+ shadow`, `+ bent_normal`, `+ thickness`); add `precompute_shadow_pbr` and multi-engine export (`SyncCraft Special PBR v2.0`, `LabPBR 1.3`, `Bedrock MER`, `Unreal/glTF ORM`, `Unity MaskMap`, `Vanilla Pre-Baked Albedo`). |
| **Interactive GUI & AI Co-Pilot** | Offline HTML Inspector only | Static single-file `export_html_inspector` with basic azimuth relighting | Build interactive 5-pane docked GUI studio (Pixel Composer + Substance + LookdevX layout), live Three.js 3D viewport, and natural-language ↔ `.blockdsl`/`.texdsl` AI co-pilot loop. |

---

## 2. SyncCraft Special PBR & Precomputed Shadow Architecture

Designed so a single authored graph shares EVERY channel seamlessly across engines — from vanilla Minecraft/Blockbench (zero shaders) to LabPBR 1.3, Bedrock RTX, Unreal Engine 5, Unity, and Maya/Arnold:

### A. Unified 12-Channel `MATERIAL` Bus (`MATERIAL_BUS_PORTS`)
| Channel | Dtype | Space / Range | Role |
|---|---|---|---|
| `albedo` | `RGBA` | sRGB `[0..1]^4` | Base color + hard/soft alpha |
| `height` | `HEIGHT` | Linear `[0..1]` | Relief displacement / POM depth |
| `normal` | `NORMAL` | Tangent RGB `[0..1]^3` | OpenGL green-up micro + macro surface normal |
| `roughness` | `SCALAR` | Linear `[0..1]` | Microfacet GGX roughness (`0.08` glass → `0.94` matte cloth) |
| `metallic` | `SCALAR` | Linear `[0..1]` | Dielectric (`0.0`) vs Conductor (`1.0`) |
| `ao` | `SCALAR` | Linear `[0..1]` | Multi-box contact + heightfield ambient occlusion |
| `shadow` | `SCALAR` | Linear `[0..1]` | **NEW v10:** Precomputed 3D ray-traced directional self-shadow map |
| `bent_normal` | `NORMAL` | World RGB `[0..1]^3` | **NEW v10:** 3D unoccluded sky-hemisphere bent normal (UE5/Substance standard) |
| `thickness` | `SCALAR` | Linear `[0..1]` | **NEW v10:** Ray-traced internal volume depth (`0`=thin wing/ear/fin → `1`=thick core) |
| `emissive` | `RGBA` | sRGB `[0..1]^4` | Selective glow RGB + emissive alpha mask |
| `sss` | `SCALAR` | Linear `[0..1]` | Subsurface scattering intensity (modulated by `thickness`) |
| `porosity` | `SCALAR` | Linear `[0..1]` | Wetness / LabPBR porosity |

### B. Multi-Engine Shared Channel Packing Targets (`synccraft.pbr.bundle/2.0`)
1. **SyncCraft Special PBR Bundle (`.synccraft.pbr.json` v2.0):** Preserves all 12 full-precision channels + precomputed shadow & bent-normal maps + bake lighting recipe so lighting can be re-baked non-destructively at any time.
2. **Vanilla / Stylized Pre-Baked Albedo (`_baked.png`, `_baked.bbmodel`):** Physically composites `albedo` + `normal` + `baked_shadow` + `bent_normal` hemisphere IBL + `thickness`-weighted SSS + `ao` + `emissive` into a single crisp sRGB texture for vanilla Minecraft / Blockbench without shaders.
3. **Minecraft Java LabPBR 1.3:** `_n.png` (`Nx, Ny, AO, POM Height`) + `_s.png` (`Smoothness, F0/MetalID, Porosity/SSS, Emissive`) + `_e.png` emissive strip.
4. **Minecraft Bedrock RTX:** `_mer.png` (`Metallic, Emissive, Roughness`) + `.texture_set.json`.
5. **Unreal Engine / glTF 2.0 / Three.js:** `_orm.png` (`R=AO, G=Roughness, B=Metallic`) + `_normal.png` + `_emissive.png` + `_shadow.png` + `_bent_normal.png`.
6. **Unity URP / HDRP:** `_maskmap.png` (`R=Metallic, G=AO, B=Detail/Height, A=Smoothness`) + `_normal.png`.

---

## 3. Milestone Execution Plan (`v10 → v13`)

### Milestone v10 (`SyncCraft v47.0`) — GUI-Ready Schema v4.0, UV-Edge Geometric Solver & Precomputed Shadow Special PBR *(COMPLETED)*
- [x] **10.1** Deep Industry Research (Pixel Composer 1.21.7–1.22.4, Substance Painter 8.1/Stylization, Substance Designer Expose Params, Marmoset Toolbag 5, UE 5.7 Triplanar, Maya LookdevX/MaterialX) + Quantitative Audit of `TexGraph v9`.
- [x] **10.2** Author persistent `ROADMAP.md` (`synccraft_v20/ROADMAP.md` & skill reference mirror).
- [x] **10.3** **GUI-Ready Schema v4.0 (`registry.py` + `core.py`):**
  - Fix all `15` nodes with misplaced `ParamSchema` in `outputs`.
  - Guarantee **100% (`144/144`)** typed `ParamSchema` coverage (including `group`, `animatable`, `visible_if`, `tooltip`), `PORT_COLORS`, and `PORT_COMPATIBILITY` matrix.
  - Add GUI layout & state fields to `NodeSpec` (`pos`, `label`, `collapsed`, `bypassed`, `preview_pinned`), bypass execution support, and topological layered `TexGraph.auto_layout()`.
- [x] **10.4** **3D Ray-Traced Mesh Baker & UV-Edge Adjacency Solver (`mesh_baker.py`):**
  - Analytical ray-vs-OBB slab tracer baking **`baked_shadow`** (multi-box directional soft penumbra shadow), **`bent_normal`** (unoccluded sky hemisphere normal), **`thickness`** (ray-traced internal volume depth for SSS), **`curvature_3d`**, and orthonormal **`world_tangent` / `world_bitangent`**.
  - **3D Geometric UV-Edge Adjacency Solver:** Matches UV island border texels across shared 3D edges (`north↔up`, `front↔side`, contacting cuboids) into vectorized `uv_seam_pairs` + `uv_seam_mask`.
- [x] **10.5** **4 Commercial Seamless-Edge & Special PBR Nodes (`seam_and_pbr_ops.py` -> `144` Total Nodes):**
  - `triplanar_project` — Seamless 3D World/Local/Pre-Skinned normal-blended (and UE 5.7 dithered) triplanar projection for scalar, RGBA, and full `MATERIAL` buses + tangent-frame normal reprojection.
  - `uv_seam_heal` — Geometric 3D UV-edge seam stitcher healing `albedo`, `height`, `roughness`, and world-reoriented `normal` across connected UV borders (2-pass symmetric convergence + same-island feather).
  - `block_edge_transition` — Connected block-edge & corner transition solver (`bevel_trim`, `moss_overhang`, `worn_stone`, `metal_frame`, `glow_circuit`, 4-bit/8-bit neighbor bitmask) for seamless multi-block structures.
  - `precompute_shadow_pbr` — SyncCraft Special PBR Shadow, Bent-Normal IBL & Thickness-SSS Baker writing both separated 12-channel PBR maps and optional pre-baked stylized albedo.
- [x] **10.6** **12-Channel Shared PBR Bus & Multi-Engine Export (`pipeline.py`, `pbr.py`):**
  - Upgrade `MATERIAL_BUS_PORTS` to 12 channels (`+ shadow`, `+ bent_normal`, `+ thickness`).
  - Upgrade `pack_labpbr_1_3` & `export_baked_pbr_package` (`synccraft.pbr.bundle/2.0`) to export Unreal/glTF `_orm.png`, Unity `_maskmap.png`, `_shadow.png`, `_bent_normal.png`, and `_thickness.png`, and feed `baked_shadow`/`bent_normal`/`thickness` physically into `bake_pbr_to_albedo`.
- [x] **10.7** **v10 Visual Verification, Automated Tests & GitHub Release:**
  - Render `v10_seamless_uv_and_shadow_pbr_sheet.png` proving: (a) 2D UV Seam Break vs `triplanar_project` + `uv_seam_heal` seamless 90° corner wrap (`2,440` solved 3D seam pairs), (b) `block_edge_transition` connected 3×3 multi-block walls (moss overhang, forged metal frame, arcane glow circuit, carved timber bevel), (c) True 3D Ray-Traced `baked_shadow`, `bent_normal`, `thickness` & `precompute_shadow_pbr` on Ironjaw Orc Warlord & Temple Monolith!
  - Pass `100%` of the expanded pytest suite (`76 passed, 16 subtests passed, 0 failed`) and push `SyncCraft` + `MH-blockbench-assets`.

---

### Milestone v11 (`SyncCraft v48.0`) — Interactive Local Node-Graph GUI Studio (`synccraft studio`) *(COMPLETED)*
- [x] **11.1** Zero-Dependency Local Python Workstation Server (`studio_server.py`, `ThreadingHTTPServer`, `127.0.0.1:18790`) executing incremental dirty-subgraph updates via SHA-256 content-addressed cache (`0.0 ms` cached hits, orphan-safe live evaluation, undo/redo stack up to 64 snapshots, `.bbmodel` -> WebGL mesh extractor, and multi-engine `synccraft.pbr.bundle/2.0` export).
- [x] **11.2** **Commercial 5-Pane Docked Workstation UI (`studio_ui/` — `index.html`, `studio.css`, `studio_gl3d.js`, `studio_graph.js`, `studio_app.js`):**
  - **Top-Left Pane (`2D / 3D PBR Stage`):** Zero-CDN WebGL 3D PBR Viewport (real GGX Cook-Torrance + tangent normals + ray-traced shadow + AO + roughness + metallic + emissive + studio floor grid + live `.bbmodel` & Cube/Wall/Pillar/Sphere lookdev primitives) + 2D Nearest-Neighbor Pixel Canvas (3×3 seamless tile wrap, zoom/fit, hover pixel eyedropper HUD) + 18-channel PBR switcher + interactive Key Light Azimuth/Elevation & 4 lighting rigs!
  - **Center Pane (`Infinite Node Graph Canvas`):** Auto-driven by Schema v4.0 (`144` nodes) — color + shape typed sockets (Circle/Diamond/Square for color-blind accessibility), cubic Bézier wires with click-to-sever & live `PORT_COMPATIBILITY` drag highlighting, drop-on-empty context-filtered Fuzzy Command Palette (`Tab`/`⌘K`), live pixel thumbnails, `ms` / `cached` badges, translucent `@pass` group frames, vertically centered Sugiyama `auto_layout()`, bypass (`B`), collapse, output pin (`★`), and interactive click-drag Minimap!
  - **Right Pane (`Schema v4.0 Inspector` · `Live TexDSL` · `QA & Profile`):** Zero-handwritten-form inspector auto-rendered across all `144` nodes (`slider`, `int`, `select`, `checkbox`, `color`, `gradient_stops` ramp editor, `visible_if` rules, quick `+` Expose & `~` Animate buttons), plus Automated Multi-Domain QA scorecard & per-node timing profiler!
  - **Bottom Pane (`Timeline` · `Exposed Params` · `Diagnostics & Log`):** Visual `@animate` driver waveforms, frame/phase scrubber & playback (`▶ Play`, `⏮`, `⏭`), Substance-style top-level exposed sliders, and live engine validation log.
  - **Bottom-Left Pane (`Node Library`):** Instant fuzzy filter across all `144` nodes with one-click canvas spawn + 4 resizable workspace layout presets (`5-Pane Studio`, `3D Lookdev`, `Graph Focus`, `TexDSL Split`).
- [x] **11.3** Live bidirectional sync between visual node graph canvas, `.texgraph.json`, and live editable `.texdsl` compiler pane + full real-browser E2E verification (`0` console errors, `0` HTTP 4xx/5xx errors, `78` passing tests).

---

### Milestone v12 (`SyncCraft v49.0`) — AI-Assisted Human-in-the-Loop 3D & Texture Pipeline
- [ ] **12.1** **Unified `.blockdsl` + `.texdsl` Asset Project Format (`.syncproj`):** Single version-controlled manifest linking 3D sculpted Blockbench geometry, semantic UV layout, multi-pass procedural texture graph, PBR bake preset, and animation clips.
- [ ] **12.2** **AI Graph Authoring & Surgical Patching Protocol:** Structured JSON/DSL tool interface allowing an AI assistant to inspect rendered views, diagnose flaws (e.g., seam break, muddy value range, weak bounce light, covered eyes), and emit minimal diffs to `.texdsl` / `.blockdsl` without overwriting human artist edits or exposed slider overrides!
- [ ] **12.3** **Reference-Guided Style Extraction (`extract_style_dna`):** Pure-algorithmic reference analyzer extracting curated OKLCH hue-shifted 6-color ramps, cluster scale statistics, and target luminance contrast curves from artist concept art to bootstrap a `.texdsl` palette block automatically.
- [ ] **12.4** **Batch Variant & Biome Morph Factory:** Drive exposed parameters (`@expose`) from a variant table (e.g., `normal`, `frost`, `volcanic`, `gilded`) to generate dozens of consistent PBR `.bbmodel` variants in seconds with automated QA gating (`score >= 85`).

---

### Milestone v13 (`SyncCraft v50.0`) — Cross-Ecosystem Commercial Interchange & Ecosystem Plugins
- [ ] **13.1** **MaterialX (`.mtlx`) & OpenPBR Graph Compiler:** Compile compatible `TexGraph` PBR material graphs directly into standard `.mtlx` XML for Autodesk Maya LookdevX, Houdini Solaris/Copernicus, Arnold, and Unreal Engine Substrate.
- [ ] **13.2** **Connected Block Tileset (CTM 47-Tile / Wang 16-Tile) Batch Generator:** Combine `block_edge_transition` with automated Minecraft OptiFine/Continuity CTM & Hytale autotile sprite-sheet export.
- [ ] **13.3** **Packaged Smart Material Archive (`.syncmat`):** Self-contained distributable subgraph package (like Substance `.sbsar`) bundling `.texdsl`, exposed parameters, thumbnail, and default presets for team libraries & marketplaces.
- [ ] **13.4** **Blockbench 5 Live WebSocket Bridge Plugin:** Extend `synccraft_pbr_rebuilder.js` so tweaking any slider in `synccraft studio` updates the open `.bbmodel` texture & 3D PBR preview inside Blockbench 5 in real time!

---

## 4. Progress Ledger (Anti-Context-Loss Tracker)
- **v43.0 (`TexGraph v5`):** Pixel-Composer `Generate -> Iterate -> Simulate -> Animate`, Baked PBR format, Blockbench 5 PBR Rebuilder plugin. *(DONE)*
- **v44.0 (`TexGraph v7`):** Substance/Pixel Composer core ops (`sdf_shape`, `tile_sampler`, `flood_fill`, `slope_blur`, `edge_wear`, `displace_uv`, `directional_selout`), Kahn topo sort, content-addressed cache, multi-domain QA. *(DONE)*
- **v45.0 (`TexGraph v8`):** Declarative `.texdsl` language, Character/Orc PBR kit, `paint_entity_eyes_and_parts`, animated `.mcmeta` hold-frame & `_e` emissive flipbooks. *(DONE @ `3eebad9`)*
- **v46.0 (`TexGraph v9`):** 15 Pixel Composer atomic nodes (`pixel_dither`, `palette_posterize`, `de_corner`, `de_stray`, `pixel_clusters`, `pixel_bevel_directional`, `heightmap_shadow`, `pixel_outline`, `bake_emissive_bounce`, etc.), 6 composable multi-node `.texdsl` showcases, `75` passing tests. *(DONE @ `e4c0018` / `de2cd8e`)*
- **v47.0 (`TexGraph v10`):** GUI Schema v4.0 (`100%` typed params across `144` nodes, `0` misplaced outputs), `NodeSpec` GUI layout + `auto_layout()` + bypass, True 3D Ray-Traced `baked_shadow`, `bent_normal`, `thickness`, `curvature_3d`, UV-Edge Adjacency Solver, `triplanar_project`, `uv_seam_heal`, `block_edge_transition`, `precompute_shadow_pbr`, 12-channel PBR bus + Unreal/glTF ORM & Unity MaskMap export (`synccraft.pbr.bundle/2.0`), `76` passing tests. *(DONE @ `539e32f` / `d31bca3`)*
- **v48.0 (`SyncCraft Studio` / Milestone `v11`):** Commercial 5-pane interactive local Node-Graph & 3D WebGL PBR Workstation (`synccraft studio`, zero external dependencies, live `.bbmodel` 3D GGX PBR viewport, Schema v4.0 auto-inspector across all `144` nodes, cubic Bézier typed wires + `@pass` frames + minimap + fuzzy palette, `@animate` timeline waveforms, bidirectional `.texdsl` live compiler, multi-engine PBR v2.0 export, E2E verified in real browser with `0` console/HTTP errors & `78` passing tests). *(DONE — Next up: Milestone `v12` AI-Assisted Human-in-the-Loop `.syncproj` Pipeline)*
