# Shrieking Legiana (*Rime-Crowned Gale Wyvern*) — Sculptor Trait Sheet

**Specimen ID**: `shrieking_legiana`
**Lineage / Split**: Legiana Lineage (`research_nc/val`, Hoarfrost Reach Master-Rank Variant)
**Visual References Studied**:
- `artifacts/mh_reference_study/monsters/shrieking_legiana_reference.png` (Hoarfrost Reach blizzard crouch art, rime-darkened wing silhouette, and pale silver-blue diamond-marked frost wing dorsal study)
- `artifacts/mh_reference_study/style_anchors/velkhana_sculpted_benchmark.png` & `zyhfisme_zinogre_01.png` (sculpted Blockbench crystal-blade & cel-band style anchors)

## 1. Head-Length (`1 HL`) Proportional System & Distinct Variant Architecture
- **`1 HL` = `13.5` Block-DSL units** (measured from rear rime-cranium to hooked frost-viper beak tip).
- **Distinct Structural & Postural Changes vs. Standard `legiana` (Zero Simple Recolor!)**:
  1. **Steeper Predatory Ambush Crouch**: Lower, coiled forward-leaning Hoarfrost stance (`pelvis` at `Y=17.45`, steeper crouched thigh/shin zig-zag, forward-lunging 3-segment S-neck) built to brace against sub-zero blizzards on `Y = 0.0`.
  2. **3-Segment Rime-Crowned Swept Head-Vanes**: Elongated `head_vane_l` -> `head_vane_mid_l` -> `head_vane_tip_l` (`~1.45x` longer than standard Legiana) encrusted with jagged white/cyan rime-frost crystal blades and spiky ice-crystal planes.
  3. **Frosted Chest Freezer-Sac Carapace**: Reinforced alabaster-white & ice-cyan cryo-sac breastplate with crystalline frost fissures and hanging icicle ruffs (`edge: drip` / `spikes`).
  4. **Serrated Ice-Blade Wing Leading Edges & Wing-Tips**: 14-wing-bone split kite-wings armed along the humerus, radius leading edge, wrist, and all 4 primary spar tips with angled white/cyan ice-blade spikes and jagged `ragged`/`spikes` trailing membranes in pale glacial silver-blue & midnight charcoal-indigo.
  5. **Enlarged Jagged 5-Ribbed Cryo Swallowtail Fan**: Expanded 5-bone stabilizer fan (`tail_fan_c` + `tail_fan_inner_l/r` + `tail_fan_outer_l/r`) bristling with terminal ice lances.

## 2. Region-by-Region Sculpting Plan (`Keep` / `Exaggerate` / `Simplify` / `Paint`)

| Region | Anatomical & Sculptural Plan | Decision |
|---|---|---|
| **Whole Gesture** | Low coiled Hoarfrost blizzard crouch on 2 splayed ice-talon raptor legs (`Y = 0.0`), forward-thrust S-neck, swept rime-crown, high-swept serrated frost wings, and wide 5-ribbed cryo tail fan. | **Exaggerate (steeper crouch + 1.35x rime silhouettes)** |
| **Head & Rime Crown** | Midnight-indigo & frosted silver-blue viper-wedge skull (`snout_taper_ratio < 0.35`), jagged brow ice-spurs, icicle cheek-barbs, fierce glowing ice-gold/cyan eyes, and 3-segment swept `head_vane_l/r` studded with white rime-crystal prisms and spiky frost planes. | **Exaggerate (1.45x vane length & crystal encrustation)** |
| **Neck & Freezer-Sac Torso** | Forward-coiled 3-bone S-neck (`neck_1..3`) with dorsal rime-ice spines; deep `chest` armored in thick frosted white freezer-sac plates, glowing cyan cryo-core fissures, and hanging chest icicle planes; tucked greyhound `waist` and `pelvis` with frosted hip winglets. | **Keep & Exaggerate (frosted freezer-sac armor)** |
| **Serrated Frost Kite-Wings** | Non-uni-sheet 7-bone-per-side rig (`wing_arm_l` -> `wing_forearm_l` -> `wing_digit_1..4_l` + `wing_patagium_l`): serrated row of angled white/cyan ice-blade spikes along the entire wing leading edge & extended scythe-tips; pale silver-cyan pleated membranes (`layer: 1|2`) with dark midnight-slate outer star/border trims. | **Exaggerate (serrated leading-edge ice blades)** |
| **Biped Ice-Grip Hindlegs** | Deeper crouched cursorial zig-zag (`hind_thigh_l` -> `hind_shin_l` -> `hind_foot_l`) with frosted thigh/shin rime-spurs and 3 splayed ice-grip raptor talons + rear hallux planted cleanly on `Y = 0.0`. | **Keep** |
| **Tail & 5-Ribbed Cryo Fan** | 5-segment whip tail (`tail_1..5`) lined with dorsal rime shards, fanning into 5 articulated stabilizer ribs (`tail_fan_c`, `tail_fan_inner_l/r`, `tail_fan_outer_l/r`) with jagged spiky ice-crystal membranes. | **Exaggerate (5-ribbed jagged cryo swallowtail)** |
| **Surface Palette** | Midnight slate-indigo (`midnight_indigo`, `obsidian_rime`), pale glacial silver-blue (`glacial_silver`, `frost_wing_pale`), crisp alabaster rime-white (`rime_white`), fierce amber-gold eyes (`amber_eye`), and radiant emissive ice-cyan crystals (`rime_cyan`). | **Paint (6-step OKLCH hue-shifted cel ramps)** |
