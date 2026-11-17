# Legiana (*Apex Glider of Coral Highlands*) — Sculptor Trait Sheet

**Specimen ID**: `legiana`
**Lineage / Split**: Legiana Lineage (`research_nc/val`, Canonical Flagship)
**Visual References Studied**:
- `artifacts/mh_reference_study/monsters/legiana_reference.png` (official aerial front-3/4, lateral wing-spread turnaround, swallowtail dorsal glide view, and head/vane color anchor)
- `artifacts/mh_reference_study/style_anchors/zyhfisme_zinogre_01.png` & `velkhana_sculpted_benchmark.png` (sculpted Blockbench cuboid-and-plane style anchors)

## 1. Head-Length (`1 HL`) Proportional System
- **`1 HL` = `13.0` Block-DSL units** (measured from rear cranium crown to sharp hooked viper rostrum tip).
- **Overall Gesture & Dimensions**:
  - Biped high-altitude glider Flying Wyvern — **2 digitigrade bird-of-prey hindlegs planted on `Y = 0.0`, ZERO walking forelegs** (forelimbs are colossal multi-lobed split kite-wings).
  - Nose-to-tail-fan arc length: `~6.8 HL` (`~88u`); Wingspan (open high-aspect glider display): `~8.2 HL` (`~106u`); Standing crown height: `~3.2 HL` (`~42u`), wingtips reaching `~4.5 HL` (`~58u`).
  - **Three Instant Silhouette Identifiers**:
    1. Ultra-narrow viper-wedge skull (`snout_taper_ratio < 0.38`) crowned by twin swept pleated cream-gold antennae head-vanes (`head_vane_l/r`).
    2. Colossal leaf-lobed split kite-wings (4 radiating primary finger lobes + broad trailing secondary forearm patagium sail).
    3. Long sinuous whip tail terminating in a 3-lobed retractable swallowtail stabilizer fan (`tail_fan_l`, `tail_fan_c`, `tail_fan_r`).

## 2. Region-by-Region Sculpting Plan (`Keep` / `Exaggerate` / `Simplify` / `Paint`)

| Region | Anatomical & Sculptural Plan | Decision |
|---|---|---|
| **Whole Gesture** | Poised raptorial S-curve: digitigrade talons grounded at `Y = 0`, narrow aerodynamic pelvis & greyhound waist, deep keeled glider chest, tall 3-segment swan S-neck, and raised-and-fanned Coral Highlands glider wings. | **Keep & Exaggerate (1.2x wing span)** |
| **Head & Viper Snout** | Wide armored royal-cobalt rear cranium & cheek jowls (`~10.0u` span) tapering aggressively through angled `maxilla_wedge_l/r` (`±19°` yaw) into a razor-narrow viper rostrum & hooked beak (`~2.2u` tip width -> `snout_taper_ratio ≈ 0.25 < 0.38`). Recessed fierce golden-amber slit eyes under sharp cobalt brows; cream-beige articulated lower jaw. Twin articulated `head_vane_l/r` (`vane_base` -> `vane_tip`) with cobalt leading ribs and wide pleated cream-gold `plane` sails. | **Exaggerate (1.3x head-vane flare & viper taper)** |
| **Neck & Torso** | 3-bone swan S-neck (`neck_1` rising `-28°`, `neck_2` arching `+8°`, `neck_3` pitching `+22°` into head) with hugged cream-beige throat scutes, lateral cryo-gill pleats, and swept cobalt dorsal spine-fins. Deep `chest` with ventral cryo-sac keel & shimmering ice-blue chest vents; narrow tucked `waist` and `pelvis` bridged by overlapping cobalt carapace shingles. | **Keep (connected 26–28% segment seating)** |
| **Split Kite-Wings** | Non-uni-sheet 7-bone-per-side rig (`14` wing bones across L+R): `wing_arm_l` -> `wing_forearm_l` (with wrist alula claw) -> `wing_digit_1_l`, `wing_digit_2_l`, `wing_digit_3_l`, `wing_digit_4_l` + trailing `wing_patagium_l`. Multi-segment tapering cobalt/cream spars ending in dark navy spine points; alternating `layer: 1` & `layer: 2` scalloped/pleated cream-ochre `plane` leaf-lobes (`edge: scallop` / `edge: ragged`) so folding never Z-fights. | **Exaggerate (1.35x leaf-lobe chord & scallop cuts)** |
| **Biped Hindlegs** | Agile bird-of-prey cursorial zig-zag (`hind_thigh_l` -> `hind_shin_l` -> `hind_foot_l`): feathered/plated cobalt thigh, slender banded raptor tibia-tarsus with backward hock spur, grounded 3-toed raptorial foot pad + curved navy/ivory talons on `Y = 0.0` + rear hallux claw. | **Keep** |
| **Tail & Swallowtail Fan** | 5-segment tapering cobalt whip tail (`tail_1..tail_5`) sweeping down then curling gracefully aloft, crowned at `tail_5` by 3 articulated stabilizer fan bones (`tail_fan_l`, `tail_fan_c`, `tail_fan_r`) with cobalt ribs and wide scalloped slate-indigo/cyan cryo-pleated `plane` blades. | **Exaggerate (1.3x swallowtail fan spread)** |
| **Surface Palette** | Dominant royal navy-cobalt (`cobalt_hide`, `navy_carapace`), supporting warm parchment cream-gold (`kite_cream`, `ochre_pleat`, `cream_belly`), dark slate-indigo wingtip/fan accents (`slate_fan`), glowing golden-amber eyes (`amber_eye`), and focal emissive glacial cyan cryo-vents (`cryo_cyan`). Fine scale rows & membrane rib striations baked at `32 px/block`. | **Paint (6-step OKLCH hue-shifted cel ramps)** |
