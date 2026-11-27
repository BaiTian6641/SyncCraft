# Gemini Deep Research Report — SyncCraft Studio `v53.0` (`TexGraph v17`) Architecture Audit

**Top-Line Summary:** This 3-track primary-source research audit establishes exact engineering specifications for SyncCraft Studio `v53.0` (`TexGraph v17`) across Blockbench 5 skeletal animation playback, 3D camera-projected PBR decals with front-most OBB occlusion, mip-safe Euclidean/JFA UV island dilation, multi-engine PBR export presets, and WCAG AA neutral dark workstation UX. The audit also uncovers four real specification discrepancies in our existing codebase (`studio.css` muted contrast `3.02:1`, Unity HDRP `MaskMap.B` height packing, DirectX vs OpenGL normal Y-axis parity for UE5/LabPBR, and MaterialX `1.39` vs `1.38` for `open_pbr_surface`) that `v53.0` resolves directly.

---

## Track 1: Blockbench 5 Skeletal Rigging & 3D Camera-Projected Decals

### 1.1 Blockbench 5.0 (`.bbmodel`) Skeletal Hierarchy & Animation Math
- **Schema Split (`format_version: "5.0"`):** Top-level `groups[]` holds bone metadata (`uuid`, `name`, `origin`, `rotation`), while `outliner[]` stores the hierarchy tree of `{uuid, children}` and element UUID leaf strings ([Blockbench `.bbmodel` Wiki](https://blockbench.net/wiki/docs/bbmodel/), [`bbmodel.js`](https://github.com/JannisX11/blockbench/blob/master/js/formats/bbmodel.js)).
- **Coordinate System & Euler Order:** Native `.bbmodel` WebGL/Three.js scenes are right-handed Y-up with **no X-reflection** (unlike Bedrock `.geo.json`). Rotations evaluate in degrees in **X-then-Y-then-Z extrinsic order** = **intrinsic `ZYX`**: $\mathbf{R}_{\text{ZYX}} = \mathbf{R}_z(r_z)\,\mathbf{R}_y(r_y)\,\mathbf{R}_x(r_x)$.
- **Hierarchical Pivot Transform:** Because every bone `origin` $\mathbf{O}_B$ and element `origin` $\mathbf{O}_E$ is stored in cumulative unrotated world space, child translations subtract parent pivots:
  $$\mathbf{M}_{\text{local}}(B, t) = \mathbf{T}\big((\mathbf{O}_B - \mathbf{O}_P) + \Delta\mathbf{p}(t)\big) \cdot \mathbf{R}_{\text{ZYX}}\big(\mathbf{e}_{\text{rest}}(B) + \mathbf{r}(t)\big) \cdot \mathbf{S}\big(\mathbf{s}(t)\big)$$
  $$\mathbf{M}_{\text{elem}}(E) = \mathbf{T}(\mathbf{O}_E - \mathbf{O}_B) \cdot \mathbf{R}_{\text{ZYX}}(\mathbf{e}_E)$$
- **Keyframe Interpolation & Safe Molang:** Supports `linear`, `step`, and uniform `catmullrom` cubic splines ($\tau = 0.5$) plus discontinuous `pre`/`post` (`data_points.length == 2`) and pure-JS recursive-descent Molang expressions (`query.anim_time` / `q.anim_time` in seconds; `math.sin`/`math.cos` in **degrees** — zero JS `eval()`).

### 1.2 3D Camera-Projected PBR Decal / Stencil Projection (`decal_stamp`)
- **Seamless UV-Independent Math:** Applying roll $\theta$ to orthonormal projector frame $(\mathbf{r}, \mathbf{u}, \mathbf{f})$ projects every UV texel's baked `world_pos` continuously into `(u, v, z_p)` across all touching UV islands with zero seam discontinuities.
- **Backface Culling & Normal-Angle Falloff:** Smoothstep attenuation of $\cos\alpha = \operatorname{dot}(\mathbf{n}_w, -\mathbf{f})$ between `max_angle_deg` and `max_angle_deg - angle_fade_deg` (Substance 3D Painter planar projection parity).
- **Exact Front-Most OBB Occlusion:** Normal culling alone cannot stop decals bleeding through to a rear limb or backplate. Vectorized Kay-Kajiya ray-vs-OBB slab intersection computes $t_{\text{first}}$ across all scene cuboids and masks out occluded texels where $s_i > t_{\text{first}, i} + \varepsilon$.

---

## Track 2: Mip-Safe UV Dilation, Asset Atlas Packing & 6-Engine Export Audit

### 2.1 Mip-Safe Pixel-Art Dilation (`mip_safe_dilate`)
- **Mipmap Bleed Physics:** At mip level $k$, GPU hardware averages $2^k \times 2^k$ texel blocks; SyncCraft's legacy `1px` rectangular clamp leaks `(0,0,0,0)` into edges at mip `1..4` and fails to pad non-rectangular alpha cutouts (wings/foliage).
- **Exact Voronoi / Jump-Flooding (`JFA+1`) Rules:**
  - **Straight-Alpha Bleed:** Dilate RGB outward from `alpha > cutoff` while keeping `alpha == 0` intact to eliminate dark cutout halos.
  - **Normal Maps:** Dilate tangent RGB via nearest-seed Voronoi (never blur), renormalize $\|\mathbf{N}\| = 1$, and use flat `(128, 128, 255)` background.
  - **Data & Sentinel Maps:** Dilate in linear space while preserving LabPBR `_n.A in [1..255]` and `_s.A = 255` (no emission) outside UV islands.

### 2.2 Multi-Engine PBR Specification Discrepancies Resolved in `v53.0`

| Target Preset | Filenames & Sidecars | Channel Layout & Critical Fix Applied | Normal Y-Axis |
| :--- | :--- | :--- | :--- |
| **Minecraft Java LabPBR 1.3** | `<stem>.png`, `_n.png`, `_s.png` | **Fixed:** Emit DirectX Y-down in `_n.G` per shaderLABS spec (previously OpenGL Y-up). | **DirectX (Y-down)** |
| **Minecraft Bedrock RTX / Vibrant Visuals** | `<stem>.png`, `_mer.png`, `_mers.png`, `_normal.png`, `.texture_set.json` | **Upgraded:** Support `"format_version": "1.21.30"` (`_mers` with Subsurface in `A`) alongside `"1.16.100"`. | **OpenGL / DirectX toggle** |
| **Unreal Engine 5 (Substrate / Native)** | `T_<stem>_D.png`, `T_<stem>_N.png`, `T_<stem>_ORM.png`, `T_<stem>_E.png` | **Fixed:** Native UE5 naming + DirectX Y-down normal flip (`1 - ny`). | **DirectX (Y-down)** |
| **Unity HDRP / URP** | `<stem>_Albedo.png`, `_Normal.png`, `_MaskMap.png`, `_HeightMap.png` | **Fixed Bug:** `MaskMap.B` is **Detail Mask** (`255` neutral), NOT Height (`pbr.py:200`); Height moves to `_HeightMap.png`. | **OpenGL (Y-up)** |
| **Maya LookdevX / Karma / OpenUSD** | `<stem>.mtlx` + OpenGL PBR PNGs | **Fixed Spec Version:** Emit MaterialX **`1.39`** for `<open_pbr_surface>` and **`1.38`** for `<standard_surface>` (`syncmat_and_mtlx.py:66`). | **OpenGL (Y-up)** |
| **Blockbench / Hytale Pre-Baked** | `<stem>_baked.png` + `<stem>_baked.bbmodel` | **Upgraded:** Run mip-safe straight-alpha Voronoi dilation (`bleed_px >= 4`) before embedding into `.bbmodel`. | N/A (Baked) |

---

## Track 3: Commercial DCC Workstation UI/UX Audit (`gemini-frontend-design`)

- **WCAG AA Contrast & Neutral Calibration:** Current `--text-muted: #6c7689` scores `3.02:1` on `--bg-header` (fails AA `4.5:1`), and blue-tinted `#232833` viewport chrome introduces simultaneous-contrast warmth bias. `v53.0` adopts achromatic neutral dark stops (`#121316`–`#27292e`) and `--text-muted: #9499a3` (`4.83:1–6.16:1` AA across all surfaces).
- **Typography & Micro-Interaction Hygiene:** Restrict `--font-mono` strictly to code/logs, use `font-variant-numeric: tabular-nums` for numeric UI readouts, switch `ALL CAPS` headers to **Sentence case**, remove pulsing decorative badges, replace `transition: all`, and honor `@media (prefers-reduced-motion: reduce)`.
- **Reclaiming `+100px` 3D Canvas Height at `1440×900`:** Collapse the 2–3 row 3D button wall + permanent 6-slider light grid into a single `32px` Substance-style contextual toolbar (`[Orbit | Pick | Paint | Decal | Light]` + `Lighting ▾` & `View ▾` popovers), and mount Paint, Decal, and Skeletal Animation Transport (`Idle / Walk / Cast / Inspect`, scrubber, speed, bone gizmos) as zero-reflow translucent glass overlays inside `.viewport-stage`.
- **Substance-Style Export Modal:** Replace the 6-button footer wall with an **Output Template** selector, mip-dilation control, live RGBA channel-packing preview chips, and one primary `Export` action.

---

## Unresolved / Deferred Items
1. Blockbench 5 `"interpolation": "bezier"` per-axis custom Bézier handle arrays (`bezier_left_time/value`, `bezier_right_time/value`) and `type: "mesh"` arbitrary non-cuboid mesh vertices remain out-of-scope for `v53.0` (standard `linear`, `catmullrom`, `step` + Molang cuboid rigs cover 100% of SyncCraft `.bbmodel` creatures).
2. Grazing-angle anisotropic prefiltering for high-frequency projected decals uses bilinear UV sampling + smoothstep angle fade in `v53.0`; full footprint EWA prefiltering is deferred to `v54.0`.
