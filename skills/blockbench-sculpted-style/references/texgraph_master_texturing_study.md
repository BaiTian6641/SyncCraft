# SyncCraft TexGraph v8 — Master Blockbench & Pixel Art Texturing Study

> Distilled from a full multi-timestamp study of **16 expert Blockbench & Pixel Composer YouTube timelapses and tutorials** (Voxel Spawns, TugkanDeMan, BearSpirit08, Mickey Joe, TheAdventurousMiner, Everbloom Games, ArtsByKev, Longridr, Captain Atlas, tafaar, MakhamDev, Etomic, MrXuiryusGaming, Juuliet, Side Quest), paired with SyncCraft's declarative **`.texdsl` Texture Language** and **Part-Aware Semantic Entity Pipeline**.

---

## 1. The Universal 4-Stage Studio Blockbench Texturing Progression

Every top-tier Blockbench artist follows the exact same four-stage discipline — whether painting an Orc Warlord, a Pirate Pillager, a Possessed Knight, a Husky, or a Dragon:

### Stage 1 — Clean Semantic Zone Blocking (Base Midtones + UV Discipline)
- **Texel Density Discipline:** Maintain `16 px/block` (standard Minecraft) or `32 px/block` (boss/detailed entity) consistently across connected body parts. Never cram heads or faces into tiny UV islands while wasting atlas pixels on hidden joints.
- **UV Padding:** Keep `1–2 px` gutter padding between semantic UV islands and pad outward (`material_pad_islands`) so linear filtering never bleeds dark background color onto edges.
- **Part Partitioning First:** Before adding a single pixel of noise or fur, separate EVERY distinct semantic material into its own clean mask/layer:
  - *Primary body skin / hide / fur / plumage / scales*
  - *Facial landmarks* (`eye`, `eye_socket`, `brow`, `nose`/`snout_tip`, `beak`, `mouth`/`palate`/`tongue`, `teeth`/`fang`/`tusk`)
  - *Keratin & hard appendages* (`horn`, `claw`, `talon`, `hoof`, `spike`, `spine`)
  - *Worn gear & accessories* (`metal`/`armor`/`pauldron`/`blade`, `leather`/`strap`/`belt`, `cloth`/`cape`/`loincloth`, `wood`/`haft`, `rope`, `fur_trim`)
  - *Emissive focal points* (`eye_glow`, `rune`, `core`, `vein`, `lantern`)
- **Rule #1 (Anti-Blanket Law):** **NEVER paint a uniform fur, scale, or stone shader across an entire entity!** Covering eyes, nostrils, teeth, beaks, claws, hooves, straps, or armor with a uniform body texture immediately destroys character readability and makes a model look procedurally stamped instead of studio-crafted.

### Stage 2 — Volumetric Light Planes & 1 px Contact Cast Shadows
- **Top-Down Form Planes (3-Value Readability First):** Block light-catching top/front planes, neutral side planes, and cool underside planes using baked 3D world normals (`body_flow_gradient("top_light_wrap")`), per-island height gradients (`per_island_gradient`), and multi-box contact ambient occlusion (`mesh_channel("ao")`).
- **The 1 px Cast-Shadow Material Seam Rule (Voxel Spawns / BearSpirit08 / ArtsByKev):**
  - Directly **underneath** every overlapping belt, pauldron rim, leather strap, brow ridge, ear fold, or armor plate, paint a **1 px dark cool cast-shadow band** on the underlying skin/cloth.
  - Directly **on the top outer edge** of the overlapping strap/plate, paint a **1 px lighter catchlight bevel**.
  - This single 2-pixel pairing (`1 px dark shadow underneath + 1 px top bevel catchlight`) gives flat Blockbench cubes unmistakable 3D layered depth at Minecraft viewing distances.

### Stage 3 — Material-Specific Micro-Craft (Hue-Shifted Ramps & Specular Separation)
Different materials MUST have visibly different **contrast ranges**, **hue-shift trajectories**, **edge crispness**, and **PBR roughness/metallic values**:

| Material | Hue-Shift Ramp Rule | Contrast & Edge Craft | LabPBR Targets |
|---|---|---|---|
| **Orc / Goblin / Troll Skin** (`orc_flesh_skin`) | **Warm yellow-green** highlights (forehead, brow ridge, nose bridge, cheekbones, knuckles) → **rich emerald / olive (`#33984b` / `#4E7A38`)** midtones → **cool deep teal-green / blue-violet** crevices. Add subtle warm reddish-brown subcutaneous blush on nose bridge, cheeks, knuckles, and fresh scars! | Smooth painterly muscle planes with soft cluster shading (`3–5` tonal steps), crisp dark brow crease, optional tribal war paint & healed pale scars. | Roughness `0.56–0.70`, Metallic `0.0`, SSS `0.30–0.45` |
| **Forged Iron / Steel Armor** (`forged_battle_metal`) | **Cool charcoal-indigo** shadows (`#1E1824`) → **slate steel** body (`#48474D`–`#6E7382`) → **bright ice-silver** 1 px edge bevels (`#CFD9E6`). Never pure grey! | **Wide value jump** (`3×` sharper specular transitions than leather/cloth), crisp 1 px outer diamond bevels, diagonal battle scratches, dark rivet pits, optional warm ochre rust in cavities. | Roughness `0.20–0.36`, Metallic `0.85–0.94` |
| **Stitched Leather & Straps** (`stitched_leather`) | **Deep cool umber** creases (`#241614`) → **warm saddle/chestnut brown** body (`#6B4228`) → **warm tan** stitched bevels (`#A8744B`). | **Much lower specular contrast than steel.** Concentric stepped rim shading (darker recessed center panel, warmer outer rim), regular 1–2 px stitch dots along borders, soft crease wear. | Roughness `0.58–0.74`, Metallic `0.0` |
| **Woven Battle Cloth & Rope** (`woven_battle_cloth`) | **Muted dark wine / plum / charcoal-olive** folds → **desaturated crimson / linen / moss** midtones → **warm dusty highlight** ridges. | Soft 2–3 px vertical drape streaks with gentle value steps (never crunchy), subtle cross-weave micro-grain, tattered/frayed lower hems; diagonal alternating tan/umber twists for rope wraps. | Roughness `0.82–0.94`, Metallic `0.0` |
| **Keratin: Tusks, Horns, Teeth, Claws, Hooves, Beaks** (`keratin_horn_bone`) | **Dark desaturated root/gumline collar** (`#3B302C` or `#231C1E`) → **warm ochre-taupe** mid shaft → **pale warm ivory/cream** distal tip (`#E8DFC8` — NEVER stark `#FFFFFF`!). | Longitudinal growth striations + subtle cross-rings on horns/tusks; **jagged 1–2 px stepped teeth** where hooves/claws transition into fur/hide so junctions feel organic instead of cut with scissors. | Roughness `0.38–0.52`, Metallic `0.0` |
| **Organic Fur & Plumage** (`organic_fur_pelt` / `layered_plumage`) | Cool lavender/slate undercoat shadows → rich midtone directional clumps → warm sunlit tips; lighter cream belly/chest counter-shading (`ArtsByKev`, `Everbloom Games`, `Captain Atlas`). | Directional overlapping V-tufts / chevron feathers — **NEVER uniform TV-static noise!** Feather tips and wing edges carry crisp scalloped alpha cutouts and darker rachis/edge trim. | Roughness `0.72–0.88`, Metallic `0.0` |
| **Eyes, Noses & Mouths** (`paint_entity_eyes_and_parts`) | **Eyes:** 1–2 px recessed dark socket + overhanging brow shadow → dark pupil + saturated multi-tone or glowing gradient iris + **1 px crisp specular catchlight** (avoid bright `#FFFFFF` sclera on fierce monsters/orcs!). **Mouth interior:** **Very dark desaturated plum/purple** (`#261422` — NEVER pure `#000000`!) contrasting with ivory teeth and muted coral-pink tongue. **Nose/Ear inner:** Dark glossy wet nose tip (`roughness ~0.22`) vs matte muzzle; muted coral-pink inner ear anchored by dark maroon rim. | Always painted **LAST** (or composited over the base body bus with protected semantic masks) so fur/scales/noise NEVER bleed onto eyes, teeth, noses, or claws! | Eye Roughness `0.08–0.14`, Wet Nose `0.22`, Emissive iris optional |

### Stage 4 — Global Atmospheric Grading & Silhouette Cohesion
- Apply one unified world-space light wrap & color-harmony grade (`material_grade` with subtle sky zenith lift, cool under-bounce, contact AO darkening, and light OKLCH vibrance) so skin, steel, leather, cloth, bone, and fur feel photographed inside the **same lighting atmosphere** while preserving emissive glow purity.

---

## 2. Animated Minecraft & Pixel Composer Texture Craft

Distilled from **tafaar**, **MakhamDev (Pixel Composer)**, **Etomic**, **MrXuiryusGaming**, and **Juuliet**:

### A. Minecraft Flipbook Strip & `.png.mcmeta` Specification
- **Layout:** Vertical stack of `N` frames at `W × (H * N)` (e.g., `16×256` for 16 frames of a `16×16` block, or `256×2048` for 8 frames of a `256×256` entity atlas).
- **Basic `.png.mcmeta`:**
  ```json
  {
    "animation": {
      "frametime": 2,
      "interpolate": false
    }
  }
  ```
  (`frametime` is measured in Minecraft game ticks: `20 ticks = 1 second`; `frametime: 2` = `10 fps`).
- **Hold-Frame Pattern (Eye Blinks, Ore Glints, Shimmer Flashes):** Keep frame `0` (open eyes / idle metal) held for 2–3 seconds (`time: 40–60` ticks), then snap rapidly through blink/glint frames `1, 2, 3` (`time: 2` ticks each):
  ```json
  {
    "animation": {
      "frametime": 2,
      "interpolate": false,
      "frames": [{"index": 0, "time": 44}, 1, 2, 3, 2, 1]
    }
  }
  ```
- **Ping-Pong Loop Pattern (Breathing Ember Eyes, Pulsing Runes, Magma Glow):** Avoid jarring wrap jumps by reversing through the middle frames: `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1]`.
- **Synchronized Emissive Animation (`_e.png` + `_e.png.mcmeta`):** Export a matching emissive strip containing **ONLY glowing pixels** (eyes, runes, molten fissures) over transparent alpha (`rgba = (0,0,0,0)` everywhere else) with the exact same `frametime` & `frames` schedule!

### B. Pixel Composer Procedural Animation Patterns (Pure Algorithmic in TexGraph)
1. **Seamless Scrolling Tileable Noise (`perlin`/`fbm`/`voronoi` + `scroll`):** Scroll vertically/diagonally with `tileable=True`, mask by a vertical gradient, sharpen via `levels`, and map through a hot hue-shifted palette (`fire`, `magma`, `water_caustics`).
2. **Masked UV Displacement (`displace_uv`):** Drive a scrolling noise displacement map through a root-to-tip strength mask (`mid_value=0.5`) so flame/cape/banner bases stay pinned while tips dance organically.
3. **Keyframed / Oscillator Parameter Drivers (`@animate { drive node.param = ... }`):** Drive `eye_blink`, `emissive_pulse`, `phase_offset`, or `lightness_offset` cleanly via `sine`, `triangle`, `wiggler`, or `keyframes(...)` across frames with automatic static-subgraph memoization!
4. **Particle & Fluid Simulations (`particle_sim`, `fluid_smoke_sim`):** Spawn rising embers, fire wisps, or swirling arcane smoke directly into RGBA or emissive masks with lifespan scale & color gradients.

---

## 3. Declarative `.texdsl` Texture Language (Program Textures Directly, Not 300 Lines of Python!)

Instead of writing verbose Python scripts for every new asset, author textures directly in clean, version-controllable **`.texdsl`** files and run them from CLI or Python:

```bash
synccraft texgraph render ironjaw_orc_warlord.texdsl --bbmodel ironjaw_orc_warlord.bbmodel --out out/
synccraft texgraph validate ironjaw_orc_warlord.texdsl
synccraft texgraph qa ironjaw_orc_warlord.texdsl --bbmodel ironjaw_orc_warlord.bbmodel
```

### Canonical `.texdsl` Structure (4-Stage Part-Aware Entity Template)
```texdsl
@texture {
  name: "ironjaw_orc_warlord"
  size: [256, 256]
  domain: "entity"
  seed: 808
  frames: 8
  frametime: 3
  interpolate: false
  frames_schedule: [{index: 0, time: 18}, 1, 2, 3, 4, 5, 6, 7]
}

@palette {
  orc_skin: ["#182924", "#274735", "#3D6E3E", "#659449", "#98B85B", "#C8D67E"]
  iron_steel: ["#1B1724", "#2F313E", "#4D5263", "#798194", "#B0BCCB", "#E2ECF5"]
  crimson_cloth: ["#220B14", "#4A1420", "#7D1F2A", "#AD3138", "#D65A54"]
}

@pass bake "Pass 1: 3D Mesh Bake & Semantic Part Masks" {
  ao          = mesh_channel("ao")
  m_metal     = select(pattern="pauldron|chest_plate|bracer|greave|helm|axe_head|buckle|ring")
  m_leather   = select(pattern="belt|strap|boot|wrap|grip|glove|harness")
  m_cloth     = select(pattern="loincloth|tabard|cape|sash|cloth")
  m_fur       = select(pattern="pelt|fur|mantle|collar")
}

@pass materials "Pass 2: Distinct Studio PBR Materials" {
  skin_mat    = orc_flesh_skin(stops=@orc_skin, scar_amount=0.35, warpaint_color="#A82424")
  metal_mat   = forged_battle_metal(stops=@iron_steel, scratches=0.55, rust=0.22)
  leather_mat = stitched_leather(stitch_density=1.0, wear=0.35)
  cloth_mat   = woven_battle_cloth(stops=@crimson_cloth, fold_contrast=0.45)
  fur_mat     = organic_fur_pelt(palette="wolf_charcoal")
}

@pass composite "Pass 3: Layered Height-Blend + Cast Shadows + Protected Facial Landmarks" {
  m1          = material_height_blend(skin_mat, cloth_mat, mask=m_cloth, contrast=7.0, seam_ao_strength=0.42)
  m2          = material_height_blend(m1, leather_mat, mask=m_leather, contrast=8.0, seam_ao_strength=0.45)
  m3          = material_height_blend(m2, fur_mat, mask=m_fur, contrast=5.5, seam_ao_strength=0.35)
  m4          = material_height_blend(m3, metal_mat, mask=m_metal, contrast=9.0, seam_ao_strength=0.48)
  # Protected anatomical landmark pass — NEVER lets skin/fur/metal overwrite eyes, tusks, teeth, nails, runes!
  detailed    = paint_entity_eyes_and_parts(
    m4,
    eye_iris="#FF6B1A",
    eye_glow=0.88,
    eye_blink=0.0,
    rune_color="#FF7A21",
    rune_glow=0.92,
    tusk_tip="#EBE0C6"
  )
}

@pass grade "Pass 4: Global Atmosphere & UV Padding" {
  graded      = material_grade(detailed, ao_bake=ao, ao_darken=0.38, atmosphere_strength=0.14, vibrance=0.16, contrast=1.06)
  final_mat   = material_pad_islands(graded, pad=2)
}

@animate {
  drive detailed.eye_glow  = sine(min_val=0.55, max_val=1.0, freq=1.0)
  drive detailed.rune_glow = sine(min_val=0.40, max_val=1.0, freq=1.0, phase_offset=0.15)
  drive detailed.eye_blink = keyframes([[0.0, 0.0], [0.72, 0.0], [0.82, 0.95], [0.92, 0.0], [1.0, 0.0]])
}

@output material(final_mat, normal_strength=2.5)
```

---

## 4. Self-Check QA Checklist for Every 3D Entity Texture
Before shipping ANY 3D entity texture:
1. **4× Face/Eye Macro Check:** Are both eyes clearly framed by a dark socket & overhanging brow, with distinct iris/pupil/catchlight (or crisp glowing gradient), and **ZERO** fur/scale/noise bleed across the eye or sclera?
2. **Anatomical Separation Check:** Are nostrils/wet nose tips, beaks, mouths (dark plum interior), ivory teeth/tusks, dark-rooted claws/hooves/talons, and fins cleanly distinguished from body hide?
3. **Material Specular Contrast Check:** Do steel plates read sharply reflective (`roughness ~0.25`, `metallic ~0.9`), leather warm and satin (`roughness ~0.64`), cloth soft and matte (`roughness ~0.86`), and skin/fur organically shaded?
4. **1 px Cast Shadow Check:** Does every belt, strap, pauldron, and collar cast a subtle dark contact edge onto the underlying layer?
5. **UV Padding & Animation Check:** Zero dark UV seam bleeding at islands; if animated, verify frame-to-frame loop continuity and clean `_e.png` emissive isolation.
