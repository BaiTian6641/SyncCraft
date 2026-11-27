# SyncCraft Studio `v51.1` (`TexGraph v15.1`) — Pipeline Hardening, Anti-Glitch & Optimization Audit Report

> **Date:** 2026-11-27  
> **Scope:** Full-stack forensic audit of `mesh_baker.py`, `core.py`, `studio_server.py`, `studio_gl3d.js`, `studio_app.js`, and `studio_ui/index.html` covering loading latency, 1-to-1 project fidelity, cross-project/async texture glitch elimination, and persistent file logging.

---

## 1. Root Causes Identified & Fixed (Texture Glitch & Non-1-to-1 Project Loading)

| # | Defect / Symptom | Exact Root Cause Found | Permanent Architectural Fix |
|---|---|---|---|
| **1** | **Old project's animation kept glitching onto new project's 3D model** | `pSel.onchange` in `studio_app.js` never called `stopAnimLoop()` or `invalidateAnimRing()`, leaving `gl3d.animRing` / `activeAnimIdx` active across project switches. | Added centralized `resetWorkspaceVisuals()` + `applyServerState()` epoch-change hook that stops playback, deletes GPU ring textures, cancels pending `evalTimer`, and resets all WebGL slots before loading. |
| **2** | **Slower in-flight `/api/evaluate` overwrote newer edits/projects** | Neither `/api/state` nor `/api/evaluate` carried a workspace epoch or monotonic sequence number, allowing an earlier slow evaluation (e.g. 900 ms Kaelith eval) to arrive after a faster project/parameter update and overwrite `state.evalData` & WebGL textures. | Added server-side monotonic `workspace_epoch` (bumped on `_set_workspace`, `load_mesh`, `enter_group`, `exit_group`) and `eval_seq` + client-side `evalReqSeq` / `evalAppliedSeq` guards discarding any superseded or prior-epoch response. |
| **3** | **Out-of-order `Image.onload` async decode overwrote newer WebGL & 2D textures** | `_loadTexFromUrl()` in `studio_gl3d.js` and 2D image decode in `studio_app.js` had no generation token, so a larger earlier PNG finishing decode after a smaller newer PNG overwrote the live GPU texture. | Added per-slot monotonic `_texSlotGen[slot]` in `studio_gl3d.js`, `_animUploadGen` inside `uploadAnimFrames()` (auto-disposing aborted GPU textures), and `state.img2dGen` in `studio_app.js`. |
| **4** | **Stale Normal / ORM / Shadow / Emissive / Thickness leaked across projects** | `gl3d.updateChannels()` used `if (channels.normal) ...` without resetting absent channels, leaving Kaelith's 12-channel PBR maps bound on non-PBR/2D presets forever and flashing Project A's texture onto Project B's mesh during load. | Added `gl3d.resetTextures()` (called immediately on project/mesh switch) and `_resetSlotToDefault(slot)` whenever any channel is absent (`normal -> [128,128,255,255]` + `hasNormalTex=false`, `orm`, `shadow`, `emissive`, `thickness`, `hasTransparency=false`). |
| **5** | **Cross-mesh Python cache key collision (`id()` reuse)** | `TexGraph._node_content_hash()` hashed Python memory address `id(self.mesh_context)` (which CPython reuses after GC) and `TexGraph.attach_mesh()` only cleared `_static_cache` without clearing `_content_cache`. | Added cryptographic 32-hex SHA-256 `MeshBakeContext.content_fingerprint` (hashing canonical geometry, rotations, UV rects & bake kwargs), hashed `content_fingerprint` in `_node_content_hash()`, and cleared `_content_cache` in `attach_mesh()` whenever fingerprint changes. |
| **6** | **Non-1-to-1 project loading & silent sample fallback** | Loading standalone `.texdsl` / `.texgraph.json` silently retained the previous project's `.bbmodel` mesh (resizing 2D blocks to 3D entity UV dims); missing `kaelith`/`orc` files silently substituted `monolith_v10_3d`; no unified Open File loader existed for `.syncproj` / `.bbmodel` / `.texdsl` / `.texgraph.json`. | Made `load_project()` strictly 1-to-1 (`keep_mesh=False` default for `.texdsl`/`.texgraph.json`), removed silent fallback (`FileNotFoundError`), added `bbmodel_project`, `syncproj`, and `disk_file` (with sibling `.bbmodel` auto-pairing) loaders, and added a topbar **Open** button (`#btnOpenProject`). |
| **7** | **`Selected Node Output` stuck on previous node** | Clicking another node while `#channelSelect === "selected"` updated the Inspector without refreshing `selected_preview`. | Clicking any node while in `selected` channel now immediately fetches updated `selected_preview` (`~44 ms` via channel PNG cache hit); switching projects resets channel cleanly to `pbr_lit`. |

---

## 2. Loading & Evaluation Pipeline Optimizations (Measured Before vs. After)

Profiling `StudioSession` with `cProfile` revealed that loading Kaelith (`174` sculpted boxes, `208×240` UV) spent **`8.65–11.11 seconds`** across **`10,913,252` Python calls** in `mesh_baker.py` (`2.52 s` pure-Python UV-edge adjacency loop, `2.07 s` across `99,393` scalar `_ray_obb_hits` calls, `1.94 s` across `188,964` scalar `_obb_signed_distance` calls), with zero mesh-bake caching across project switches or restarts.

### Engineering Optimizations Implemented
1. **Fully Vectorized Batched NumPy Mesh Baker (`mesh_baker.py`):**
   - Replaced per-neighbor Python loops with batched float32 `_obb_signed_distance_batch()` (`np.einsum("nki,kij->nkj", ...)`).
   - Replaced per-candidate Python ray loops with conservative penumbra-cone cylinder broad-phase culling + batched `_ray_obb_hits_batch()`.
   - Replaced the `14,900 × 27` pure-Python `_solve_uv_edge_adjacency` loop with vectorized per-cell 3D spatial-batch broadcasting (`pa[:, None, :] - pb[None, :, :]`).
2. **Two-Tier Content-Addressed SHA-256 Mesh Bake Cache (`mesh_baker.py`):**
   - **L1 In-Memory LRU (`max 16`):** Instant `< 2 ms` project switching with bit-for-bit identical `baked_ao`, `baked_shadow`, and `uv_seam_src_yx`.
   - **L2 Persistent Disk `.npz` Cache (`~/.synccraft/cache/mesh_bakes/<fp>.npz`):** Survives server restarts (`~12.9 ms` load vs `8,651 ms` cold!).
3. **Static Mesh Diagnostic & Output Channel PNG Caching (`studio_server.py`):**
   - Cached static `uv_seams`, `uv_regions`, and `inner_area` PNGs per `mesh_ctx.content_fingerprint` (`_mesh_diag_cache`).
   - Cached encoded output PBR channel sets per `(workspace_epoch, output_content_hash, emissive_hash, lighting_preset, light_az, light_el)` (`_channel_png_cache`), eliminating redundant 16-channel PNG re-encoding on channel switches and node inspection.
4. **Progressive Non-Blocking Studio Boot (`studio_app.js`):**
   - Fetches `/api/schema` + `/api/state` first (`~10 ms`) and begins evaluating/rendering immediately while `/api/schema/previews` (`228` swatches), `/api/schema/preview_anim_sheet`, and `/api/syncmat/list` hydrate asynchronously in parallel (`~450 ms` faster Time-to-First-Render).

### Verified Benchmark Comparison (Apple Silicon, Kaelith 174-Box Flagship & Full Suite)

| Benchmark Metric | Before (`v15.0`) | After (`v15.1`) | Speedup |
|---|---:|---:|---:|
| **Kaelith 174-Box `bake_bbmodel_maps` (Cold, uncached)** | `11,110.3 ms` *(profiled)* / `8,651.3 ms` | **`1,444.8 ms`** | **`6.0× – 7.7×` faster** |
| **Kaelith `bake_bbmodel_maps` (Disk `.npz` cache hit across restart)** | `8,710.2 ms` | **`12.9 ms`** | **`675×` faster** |
| **Kaelith `bake_bbmodel_maps` (In-Memory LRU hit on switch)** | `8,710.2 ms` | **`1.44 ms`** | **`6,036×` faster** |
| **Full `load_project(kaelith_dragonborn_3d)` (Cold)** | `8,651.3 ms` | **`1,524.1 ms`** | **`5.7×` faster** |
| **Full `load_project(kaelith_dragonborn_3d)` (Switch Reload)** | `8,710.2 ms` | **`105.1 ms`** | **`82.9×` faster** |
| **Load Temple Monolith 3D** | `86.8 ms` | **`20.7 ms`** | **`4.2×` faster** |
| **Node Inspection (`Selected Node Output`, channel cache hit)** | `~180–290 ms` | **`44.5 ms`** | **`4× – 6×` faster** |
| **Full 58-Test TexGraph/Studio/PBR Regression Suite** | `56.88 s` *(57 tests)* | **`18.31 s`** *(58 tests)* | **`3.1×` faster** |

---

## 3. Persistent Rotating File Logging & Diagnostics

- **File Location:** `~/.synccraft/logs/synccraft_studio.log` (`RotatingFileHandler`, `2 MB × 3` backups, UTF-8 NDJSON lines with ISO-8601 UTC timestamps).
- **Logged Events:** Workspace loads (`epoch`, `graph`, `nodes`, `dims`, `mesh_fp`, `mesh_cache_hit`, `load_ms`), every evaluation (`epoch`, `eval_seq`, `eval_ms`, `total_ms`, `cache_hits/misses`, `ch_cache_hit`, `qa`), mutations, exports, and HTTP `4xx`/`5xx` errors.
- **API & UI Controls:**
  - `GET /api/logs/file?tail=200` (JSON tail) and `GET /api/logs/file?tail=2000&download=1` (downloadable `.log` file)
  - `POST /api/logs/clear`
  - `GET /api/health` (`engine_version: "15.1"`, `workspace_epoch`, `log_file`, `mesh_cache`)
  - Bottom **Log** tab toolbar: **File Tail**, **Download .log**, and **Clear**.

---

## 4. Verification Summary
- **Pytest:** `58 passed, 0 failed` in `18.31s` (including `test_v15_1_loading_integrity_epoch_and_cache_isolation`).
- **Live `1440×900` Browser Stress Test:** `0` console errors, `0` uncaught page errors, `0` HTTP `>=400` errors across mid-playback project switches (`brass_crystal_workflow_3d` -> `kaelith_dragonborn_3d`), 3D -> 2D preset -> 3D cached reloads, 1-to-1 `Selected Node Output` node switching, topbar `Open` button zero-clipping check, and persistent `File Tail` verification.
