"""Reusable Pro Studio TexGraph + Full PBR Recipes for Entities (3D-Mesh-Aware), Tileable Blocks & Items.

Every recipe outputs:
- Crisp, vibrant, clean pixel-art Albedo (OKLCH hue-shifted ramps, zero muddy/dirty shadows)
- Full LabPBR 1.3 + Bedrock MER channels: Height, Scharr Tangent-Space Normal, Roughness, Metallic,
  Baked AO, Subsurface Scattering / Porosity, and Emissive (with synchronized animated flipbooks!)
"""
from __future__ import annotations

from typing import Any

from .core import TexGraph
from .mesh_baker import MeshBakeContext


# ---------------------------------------------------------------------------
# 1. Seamless Tileable Block & Item Showcase Recipes (Full PBR + Vibrant)
# ---------------------------------------------------------------------------


def build_neon_hex_block(size: int = 32, *, seed: int = 404) -> TexGraph:
    """Seamless animated tileable 'Antigravity x Gemini 4 Neon' Hex Conduit block (Full PBR, 16-frame loop)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="neon_hex_conduit_block")

    hex_h = g.hex_grid(cols=4, mortar=0.14, bevel=0.20, output="height", id="hex_tiles")
    hex_edge = g.hex_grid(cols=4, mortar=0.18, output="edge", id="hex_channels")
    metal_grain = g.gabor(scale=12.0, angle_deg=30.0, anisotropy=4.0, octaves=2, id="brushed_grain")
    micro_h = g.blend(hex_h, metal_grain, mode="soft_light", opacity=0.28, id="tile_relief")
    clean_h = g.kuwahara(micro_h, radius=1, sharpness=10.0, id="kuwahara_tiles")
    shaded = g.shade(clean_h, use_mesh=False, bump_strength=1.85, spec_strength=0.32, id="tile_shade")

    # Vibrant polished cosmic titanium-indigo hex armor (clean cerulean bevels, rich violet crevices!)
    obsidian_rgba = g.hue_ramp(
        shaded,
        base_lch=(0.35, 0.078, 266.0),
        steps=6,
        l_dark_delta=-0.18,
        l_light_delta=0.32,
        hue_shift=(24.0, -34.0),
        dither="clustered",
        dither_strength=0.10,
        id="obsidian_hex_rgba",
    )

    # Animated traveling cyan -> magenta plasma pulse through the hex grout channels
    wave = g.fbm(scale=2.0, octaves=2, mode="fbm", loop_speed=1.0, seed=seed + 17, id="plasma_wave")
    scrolled = g.scroll(wave, speed_x=1.0, speed_y=1.0, id="flowing_wave")
    vein_mask = g.math(hex_edge, g.levels(scrolled, in_min=0.12, in_max=0.92, gamma=0.80, id="wave_boost"), op="mul", id="conduit_mask")
    pulsed_mask = g.pulse(vein_mask, min_val=0.60, max_val=1.0, freq=1.0, id="pulsed_conduit")

    neon_col = g.hue_ramp(
        scrolled,
        colors=["#3B0A75", "#B52CE8", "#F04AE8", "#06B6D4", "#67E8F9", "#FFFFFF"],
        quantize=False,
        id="neon_palette",
    )

    alb_out, em_out = g.emissive(obsidian_rgba, pulsed_mask, color=neon_col, glow_bleed_px=2, glow_strength=0.46, id="final_hex")
    vib_out = g.vibrance(alb_out, vibrance=0.22, contrast=1.06, lift=0.015, id="hex_vibrance")
    clean_out = g.despeckle(vib_out, threshold=0.22, id="final_clean")

    # Full PBR material channels: metallic hex caps (M=0.88, R=0.24), matte glowing energy channels
    pbr_metal = g.levels(hex_edge, in_min=0.2, in_max=0.65, invert=True, out_min=0.0, out_max=0.88, id="pbr_hex_metallic")
    pbr_rough = g.blend(g.constant(0.24, id="cap_rough"), g.constant(0.14, id="glow_rough"), mask=hex_edge, id="pbr_hex_roughness")
    pbr_ao = g.levels(hex_h, in_min=0.0, in_max=0.75, out_min=0.38, out_max=1.0, id="pbr_hex_ao")
    pbr_nrm = g.height_to_normal(clean_h, strength=2.8, id="pbr_hex_normal")

    g.set_output(
        clean_out,
        emissive_node=em_out,
        height_node=clean_h,
        normal_node=pbr_nrm,
        roughness_node=pbr_rough,
        metallic_node=pbr_metal,
        ao_node=pbr_ao,
        normal_strength=2.8,
    )
    return g


def build_mossy_carved_brick(size: int = 32, *, seed: int = 108) -> TexGraph:
    """Seamless tileable Mossy Carved Ruin Brick (Vibrant Ashlar Stone + Emerald Turing Moss + Full PBR)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="mossy_carved_brick_block")

    bricks = g.brick(cols=2, rows=4, style="running", mortar=0.11, bevel=0.20, chip_strength=0.12, id="brick_base")
    mortar_m = g.brick(cols=2, rows=4, style="running", mortar=0.14, output="mortar", id="mortar_mask")
    stone_n = g.fbm(scale=6.0, octaves=3, mode="fbm", id="stone_grain")
    carved = g.blend(bricks, stone_n, mode="soft_light", opacity=0.24, id="weathered_stone_h")
    eroded = g.iterate(carved, operation="erode_weather", steps=3, strength=0.22, id="eroded_stone_h")
    kuw_h = g.kuwahara(eroded, radius=1, sharpness=10.0, id="kuwahara_stone")
    stone_sh = g.shade(kuw_h, use_mesh=False, bump_strength=1.9, spec_strength=0.15, id="stone_shade")

    # Crisp warm sunlit limestone-slate bricks (clean cool-indigo mortar shadows -> warm ivory stone bevels!)
    stone_rgba = g.hue_ramp(
        stone_sh,
        base_lch=(0.56, 0.044, 92.0),
        steps=6,
        l_dark_delta=-0.24,
        l_light_delta=0.24,
        hue_shift=(-34.0, 14.0),
        dither="clustered",
        dither_strength=0.10,
        id="stone_rgba",
    )

    # Vibrant emerald-jade Turing Gray-Scott moss hugging crevices cleanly
    turing_moss = g.gray_scott(preset="coral", steps=240, sim_scale=1, seed=seed + 31, id="turing_moss")
    drip = g.weathering_drip(bricks, cavity=mortar_m, streak_length=5, strength=0.80, decay=0.80, id="grime_drip")
    moss_raw = g.math(turing_moss, g.levels(drip, in_min=0.18, in_max=0.72, id="drip_mask"), op="mul", id="moss_spread")
    moss_clean = g.iterate(moss_raw, operation="moss_spread", steps=3, strength=0.42, id="moss_iterated")
    moss_mask = g.levels(moss_clean, in_min=0.24, in_max=0.52, id="moss_alpha")
    moss_h = g.blend(moss_clean, stone_n, mode="soft_light", opacity=0.35, id="moss_h")
    moss_sh = g.shade(moss_h, use_mesh=False, bump_strength=1.55, id="moss_shade")
    moss_rgba = g.hue_ramp(
        moss_sh,
        base_lch=(0.58, 0.145, 138.0),
        steps=5,
        l_dark_delta=-0.24,
        l_light_delta=0.22,
        hue_shift=(28.0, -26.0),  # cool forest-teal shadows -> sunlit chartreuse moss tips!
        id="moss_rgba",
    )

    comp = g.blend(stone_rgba, moss_rgba, mode="normal", opacity=0.96, mask=moss_mask, id="moss_over_stone")
    sel = g.selout(comp, darken=0.30, cool_tint=(0.04, 0.06, 0.16), edge_threshold=0.24, silhouette=False, interior=True, id="brick_selout")
    final = g.vibrance(sel, vibrance=0.20, contrast=1.05, lift=0.01, id="final_brick")

    # Full PBR channels: carved POM height, high stone roughness (0.72), velvety moss (0.90) + porosity/SSS!
    total_h = g.blend(kuw_h, g.levels(moss_h, out_min=0.58, out_max=0.95, id="moss_h_lift"), mask=moss_mask, id="pbr_brick_height")
    pbr_rough = g.blend(g.constant(0.68, id="stone_r"), g.constant(0.90, id="moss_r"), mask=moss_mask, id="pbr_brick_rough")
    pbr_por = g.blend(g.constant(0.45, id="stone_por"), g.constant(0.85, id="moss_por"), mask=moss_mask, id="pbr_brick_porosity")
    pbr_sss = g.math(moss_mask, g.constant(0.32, id="moss_sss_c"), op="mul", id="pbr_brick_sss")
    pbr_ao = g.levels(bricks, in_min=0.0, in_max=0.65, out_min=0.35, out_max=1.0, id="pbr_brick_ao")
    pbr_nrm = g.height_to_normal(total_h, strength=2.9, id="pbr_brick_normal")

    g.set_output(
        final,
        height_node=total_h,
        normal_node=pbr_nrm,
        roughness_node=pbr_rough,
        metallic_node=g.constant(0.0, id="pbr_brick_metal"),
        ao_node=pbr_ao,
        porosity_node=pbr_por,
        sss_node=pbr_sss,
        normal_strength=2.9,
    )
    return g


def build_molten_scales_block(size: int = 32, *, seed: int = 777) -> TexGraph:
    """Seamless animated tileable Molten Elder-Dragon Scales block (Vibrant Obsidian + Blazing Magma + Full PBR)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="molten_dragon_scales_block")

    sc_h = g.scales(cols=4, rows=4, shape="dragon", keel=0.48, id="dragon_scales")
    sc_edge = g.scales(cols=4, rows=4, shape="dragon", output="edge", id="scale_crevices")
    micro = g.voronoi(scale=8.0, output="f1", id="scale_micro")
    h_mix = g.blend(sc_h, micro, mode="soft_light", opacity=0.20, id="scale_height")
    h_kuw = g.kuwahara(h_mix, radius=1, sharpness=10.0, id="scale_kuwahara")
    sh = g.shade(h_kuw, use_mesh=False, bump_strength=2.1, spec_strength=0.34, id="scale_shade")

    # Vibrant slate-obsidian dragon scales (deep indigo-plum shadows -> luminous steel-copper ridge bevels!)
    charcoal_rgba = g.hue_ramp(
        sh,
        base_lch=(0.30, 0.055, 290.0),
        steps=6,
        l_dark_delta=-0.16,
        l_light_delta=0.32,
        hue_shift=(-18.0, 48.0),
        dither="clustered",
        dither_strength=0.08,
        id="charcoal_scales",
    )

    # Blazing animated orange-gold-white magma veins surging through scale seams!
    magma_flow = g.energy_veins(scale=3.0, thickness=0.30, warp_strength=0.10, style="nebula", flow_speed=1.0, seed=seed + 9, id="magma_flow")
    crack_mask = g.math(
        g.levels(sc_edge, in_min=0.40, in_max=0.88, id="deep_cracks"),
        g.levels(magma_flow, in_min=0.02, in_max=0.70, out_min=0.45, out_max=1.0, id="magma_heat"),
        op="mul",
        id="molten_mask",
    )
    magma_rgba = g.hue_ramp(
        magma_flow,
        colors=["#7F1D1D", "#DC2626", "#F97316", "#FACC15", "#FEF9C3", "#FFFFFF"],
        quantize=False,
        id="magma_palette",
    )

    alb, em = g.emissive(charcoal_rgba, crack_mask, color=magma_rgba, glow_bleed_px=2, glow_strength=0.58, id="final_molten")
    vib = g.vibrance(alb, vibrance=0.24, contrast=1.08, lift=0.01, id="molten_vibrance")

    # Full PBR: glossy volcanic obsidian scales (R=0.28, M=0.55 on keel ridges!), bright emissive cracks
    pbr_rough = g.blend(g.levels(sc_h, in_min=0.2, in_max=0.95, invert=True, out_min=0.24, out_max=0.52, id="sc_r"), g.constant(0.18, id="mag_r"), mask=crack_mask, id="pbr_molten_rough")
    pbr_metal = g.levels(sc_h, in_min=0.55, in_max=0.95, out_min=0.0, out_max=0.65, id="pbr_molten_metal")
    pbr_ao = g.levels(sc_h, in_min=0.0, in_max=0.7, out_min=0.35, out_max=1.0, id="pbr_molten_ao")
    pbr_nrm = g.height_to_normal(h_kuw, strength=3.0, id="pbr_molten_normal")

    g.set_output(
        vib,
        emissive_node=em,
        height_node=h_kuw,
        normal_node=pbr_nrm,
        roughness_node=pbr_rough,
        metallic_node=pbr_metal,
        ao_node=pbr_ao,
        normal_strength=3.0,
    )
    return g


def build_prism_blade_item(size: int = 32, *, seed: int = 901) -> TexGraph:
    """Crisp animated pixel-art item sprite: Aethelion's Neon Prism Greatsword (Full PBR, 16-frame loop)."""
    g = TexGraph(size, size, tileable=False, seed=seed, name="aethelion_prism_blade_item")

    sil = g.sprite_mask("crystal_blade", id="blade_silhouette")
    guard_m = g.sprite_mask("crystal_blade_guard", id="guard_mask")
    core_m = g.sprite_mask("crystal_blade_core", id="energy_core_mask")

    diag_grad = g.blend(g.gradient("horizontal", id="gx"), g.levels(g.gradient("vertical", id="gy"), invert=True, id="gy_inv"), mode="normal", opacity=0.5, id="diag_grad")
    metal_grain = g.gabor(scale=12.0, angle_deg=45.0, anisotropy=4.0, octaves=2, id="blade_grain")
    blade_h = g.blend(diag_grad, metal_grain, mode="soft_light", opacity=0.26, id="blade_relief")
    bevel = g.bevel_emboss(sil, strength=0.52, light_angle_deg=135.0, id="blade_bevel")
    blade_sh = g.blend(blade_h, bevel, mode="overlay", opacity=0.55, id="blade_shaded")

    # Vibrant holographic diamond-sapphire blade steel + radiant metallic gold crossguard!
    steel_rgba = g.hue_ramp(
        blade_sh,
        base_lch=(0.50, 0.110, 252.0),
        steps=6,
        l_dark_delta=-0.24,
        l_light_delta=0.38,
        hue_shift=(24.0, -38.0),
        alpha_mask=sil,
        id="obsidian_steel",
    )
    gold_rgba = g.hue_ramp(
        blade_sh,
        base_lch=(0.76, 0.145, 82.0),
        steps=5,
        l_dark_delta=-0.26,
        l_light_delta=0.20,
        hue_shift=(-26.0, 14.0),
        alpha_mask=sil,
        id="astrolabe_gold",
    )
    base_sword = g.blend(steel_rgba, gold_rgba, mode="normal", opacity=1.0, mask=guard_m, id="sword_base")

    # Flowing cyan-to-magenta starlight pulse traveling along the blade fuller
    energy_wave = g.scroll(g.fbm(scale=3.0, octaves=2, tileable=True, id="blade_energy_f"), speed_x=1.0, speed_y=-1.0, id="blade_energy_scroll")
    neon_rgb = g.hue_ramp(
        energy_wave,
        colors=["#9333EA", "#F43F5E", "#22D3EE", "#A5F3FC", "#FFFFFF"],
        quantize=False,
        id="blade_neon_col",
    )
    core_pulsed = g.pulse(core_m, min_val=0.72, max_val=1.0, freq=1.0, id="core_pulsed")
    alb, em = g.emissive(base_sword, core_pulsed, color=neon_rgb, glow_bleed_px=1, glow_strength=0.48, id="sword_emissive")
    sel = g.selout(alb, darken=0.52, cool_tint=(0.05, 0.02, 0.22), edge_threshold=0.26, silhouette=True, interior=False, id="blade_selout")
    final = g.vibrance(sel, vibrance=0.22, contrast=1.05, lift=0.01, id="final_blade")

    # Full PBR: razor-sharp crystal blade bevel height, metallic steel (0.90) + metallic gold guard (0.95), glossy R=0.16!
    sword_h = g.math(sil, g.blend(bevel, blade_h, opacity=0.45, id="sw_h_mix"), op="mul", id="pbr_blade_height")
    pbr_metal = g.math(sil, g.blend(g.constant(0.88, id="b_met"), g.constant(0.96, id="g_met"), mask=guard_m, id="sw_met_m"), op="mul", id="pbr_blade_metal")
    pbr_rough = g.blend(g.constant(0.16, id="b_r"), g.constant(0.22, id="g_r"), mask=guard_m, id="pbr_blade_rough")
    pbr_nrm = g.height_to_normal(sword_h, strength=3.2, id="pbr_blade_normal")

    g.set_output(
        final,
        emissive_node=em,
        height_node=sword_h,
        normal_node=pbr_nrm,
        roughness_node=pbr_rough,
        metallic_node=pbr_metal,
        normal_strength=3.2,
    )
    return g


# ---------------------------------------------------------------------------
# 2. 3D-Mesh-Aware Expert Entity Texture Recipes (Clean, Vibrant + Full PBR!)
# ---------------------------------------------------------------------------


def build_aethelion_texgraph(mesh_ctx: MeshBakeContext, *, seed: int = 4204) -> TexGraph:
    """Expert 3D-Mesh-Aware Full-PBR TexGraph for Aethelion v3 (Antigravity x Gemini 4 Neon).

    Upgrades in v3:
    - 100% Symmetric 7-Layer Sculpted Dragon Eyes (eye_l AND head_face_r__eye_l) strictly on
      outward lateral faces (vertical obsidian slit pupil, electric-cyan->gold starburst iris,
      dark limbal ring, brow shadow, crisp white catchlight; zero leakage on other 5 faces!)
    - Sweeping Gothic-Cathedral Scalloped Wing Membrane Alpha Cutouts (replaces rectangular slabs!)
      with glowing cyan-magenta trailing-edge filigree + luminous cosmic violet nebula veins + SSS
    - Vibrant Royal Cosmic Sapphire-Obsidian dragon scales (rich readability in every profile!)
    - Luminous Metallic Champagne-Gold astrolabe armor (LabPBR Metal ID 231, brushed normal map!)
    - Full LabPBR 1.3 + Bedrock MER channels (Height, Normal, Roughness, Metallic, AO, SSS, Emissive)
    """
    g = TexGraph(
        mesh_ctx.width,
        mesh_ctx.height,
        tileable=False,
        seed=seed,
        name="aethelion_v3_pbr_neon_gravity_weaver",
        mesh_context=mesh_ctx,
    )

    # 1. Seamless 3D rest-space surface microstructure + imbricated dragon scales
    n3_macro = g.noise_3d(scale=0.12, octaves=3, mode="fbm", seed=seed + 1, id="macro_3d_mottle")
    n3_ridge = g.noise_3d(scale=0.30, octaves=3, mode="ridged", seed=seed + 13, id="obsidian_3d_facets")
    scales_2d = g.scales(cols=max(18, mesh_ctx.width // 16), rows=max(16, mesh_ctx.height // 16), shape="dragon", keel=0.44, seed=seed + 29, id="dragon_scales")
    turing_cosmic = g.gray_scott(preset="labyrinth", steps=220, sim_scale=4, seed=seed + 53, id="cosmic_turing")

    armor_h_raw = g.blend(scales_2d, n3_ridge, mode="soft_light", opacity=0.42, id="armor_h_raw")
    armor_h_mottle = g.blend(armor_h_raw, n3_macro, mode="soft_light", opacity=0.22, id="armor_h_mottle")
    armor_h_clean = g.kuwahara(armor_h_mottle, radius=1, sharpness=10.0, id="armor_kuwahara")

    # Clean, non-crushed 3D studio shading (full AO stays in LabPBR _n.png; albedo keeps crisp jewel contrast!)
    armor_sh = g.shade(
        armor_h_clean,
        use_mesh=True,
        bump_strength=1.65,
        ao_strength=0.30,
        bevel_strength=0.38,
        cavity_strength=0.26,
        rim_strength=0.32,
        spec_strength=0.30,
        id="armor_3d_shade",
    )

    # Hero Material 1: Vibrant Royal Cosmic Sapphire-Obsidian Armor (readable in EVERY profile!)
    obsidian_rgba = g.hue_ramp(
        armor_sh,
        base_lch=(0.40, 0.095, 264.0),
        steps=7,
        l_dark_delta=-0.20,
        l_light_delta=0.34,
        hue_shift=(28.0, -38.0),  # rich deep royal-amethyst shadows -> luminous astral electric-cerulean bevels!
        dither="clustered",
        dither_strength=0.10,
        id="obsidian_armor_rgba",
    )

    # Iridescent ultraviolet cosmic nebula shimmer across sunlit dorsal ridges
    irid_mask = g.levels(g.math(turing_cosmic, g.mesh_channel("normal_y", id="nrm_y"), op="mul", id="irid_raw"), in_min=0.30, in_max=0.76, out_max=0.46, id="irid_mask")
    irid_rgba = g.hue_ramp(
        armor_sh,
        base_lch=(0.54, 0.165, 306.0),
        steps=5,
        l_dark_delta=-0.20,
        l_light_delta=0.24,
        hue_shift=(-22.0, 34.0),
        id="cosmic_iridescence_rgba",
    )
    obsidian_irid = g.blend(obsidian_rgba, irid_rgba, mode="screen", opacity=0.48, mask=irid_mask, id="obsidian_iridescent")

    # Hero Material 2: Radiant Brushed Astrolabe Champagne-Gold (Metallic PBR!)
    gold_mask = g.mesh_channel("element", pattern=r"(gold|trim|collar|diadem|gemini_star|astrolabe|brow|carpal_hub|elbow_fin|claw_base|vambrace|pauldron|tricep_plate|thigh_armor|patella|knuckle_hood|horn_3|tip_a|tip_b|ring_top|ring_low|nasal_bridge)", id="gold_elem_mask")
    gold_grain = g.gabor(scale=28.0, angle_deg=18.0, anisotropy=4.0, octaves=2, id="gold_brushed_grain")
    gold_h = g.blend(gold_grain, n3_macro, mode="soft_light", opacity=0.28, id="gold_h")
    gold_sh = g.shade(gold_h, use_mesh=True, bump_strength=1.25, ao_strength=0.32, bevel_strength=0.38, cavity_strength=0.28, spec_strength=0.42, spec_power=16.0, id="gold_3d_shade")
    gold_rgba = g.hue_ramp(
        gold_sh,
        base_lch=(0.75, 0.135, 82.0),
        steps=6,
        l_dark_delta=-0.28,
        l_light_delta=0.20,
        hue_shift=(-28.0, 14.0),  # rich warm bronze-amber crevices -> luminous sunlit champagne-gold edges!
        dither="clustered",
        dither_strength=0.08,
        id="astrolabe_gold_rgba",
    )
    comp1 = g.blend(obsidian_irid, gold_rgba, mode="normal", opacity=1.0, mask=gold_mask, id="comp_gold")

    # Hero Material 3: Luminous Opalescent Pearl Ventral Scutes & Throat
    scute_mask = g.mesh_channel("element", pattern=r"(ventral|sternum|throat|groin_bridge|pearl|horn_tip|tail_ventral|jaw_ventral)", id="scute_elem_mask")
    scute_ribs = g.brick(cols=16, rows=48, style="stacked", mortar=0.11, bevel=0.22, chip_strength=0.04, id="scute_ribs")
    scute_kuw = g.kuwahara(scute_ribs, radius=1, sharpness=9.0, id="scute_kuw")
    scute_sh = g.shade(scute_kuw, use_mesh=True, bump_strength=1.35, ao_strength=0.32, bevel_strength=0.24, cavity_strength=0.28, id="scute_3d_shade")
    scute_rgba = g.hue_ramp(
        scute_sh,
        base_lch=(0.82, 0.042, 232.0),
        steps=6,
        l_dark_delta=-0.26,
        l_light_delta=0.15,
        hue_shift=(24.0, -30.0),
        dither="clustered",
        dither_strength=0.08,
        id="pearl_scute_rgba",
    )
    comp2 = g.blend(comp1, scute_rgba, mode="normal", opacity=1.0, mask=scute_mask, id="comp_scute")

    # Hero Material 4: Swept Scalloped Gothic-Cathedral Wing Membranes (Amethyst Starlight + Curved Neon Rim!)
    sail_mask = g.mesh_channel("element", pattern=r"(wing_lobe|wing_hum_sail|wing_sail|membrane|patagium)", id="wing_sail_mask")
    lu = g.mesh_channel("local_u", id="mesh_lu")
    lv = g.mesh_channel("local_v", id="mesh_lv")
    side_s = g.mesh_channel("side_sign", id="mesh_side_sign")
    up_down_f = g.mesh_channel("face", faces="up,down", id="wing_updown_face")
    sail_updown_m = g.math(sail_mask, up_down_f, op="mul", id="sail_updown_mask")
    # Also cull the thin edge faces of wing_lobe so scalloped cutouts stay razor-clean with zero box frame!
    sail_side_m = g.math(sail_mask, g.levels(up_down_f, invert=True, id="not_updown"), op="mul", id="sail_edge_faces")

    scallop_a, scallop_rim = g.wing_scallop(lu, lv, arches=2, depth=0.32, rim_width=0.09, side_taper=0.06, id="wing_scallop")
    wing_barbs = g.feathers(cols=max(14, mesh_ctx.width // 28), barb_freq=38.0, barb_angle_deg=24.0, rachis_width=0.08, seed=seed + 71, id="wing_silk_grain")
    wing_kuw = g.kuwahara(wing_barbs, radius=1, sharpness=8.0, id="wing_kuw")
    wing_sh = g.shade(wing_kuw, use_mesh=True, bump_strength=1.30, ao_strength=0.30, bevel_strength=0.22, cavity_strength=0.24, rim_strength=0.34, id="wing_3d_shade")
    wing_rgba = g.hue_ramp(
        wing_sh,
        base_lch=(0.38, 0.155, 290.0),
        steps=6,
        l_dark_delta=-0.18,
        l_light_delta=0.28,
        hue_shift=(-18.0, 32.0),  # deep cosmic indigo -> vibrant radiant royal-amethyst silk!
        dither="clustered",
        dither_strength=0.12,
        id="wing_sail_rgba",
    )
    comp3 = g.blend(comp2, wing_rgba, mode="normal", opacity=1.0, mask=sail_mask, id="comp_wings")

    # Hero Material 5: Dark Obsidian Talons, Oral Cavity, Tongue & Ivory Fangs
    oral_mask = g.mesh_channel("element", pattern=r"(palate|pharynx|throat_back|oral_floor|cheek_web)", id="oral_mask")
    tongue_mask = g.mesh_channel("element", pattern=r"tongue", id="tongue_mask")
    fang_mask = g.mesh_channel("element", pattern=r"(teeth|fang)", id="fang_mask")
    talon_mask = g.mesh_channel("element", pattern=r"(talon|claw_tip)", id="talon_mask")
    comp4 = g.blend(comp3, g.hue_ramp(armor_sh, base_lch=(0.20, 0.09, 312.0), steps=4, id="oral_rgba"), mask=oral_mask, id="comp_oral")
    comp5 = g.blend(comp4, g.hue_ramp(armor_sh, base_lch=(0.48, 0.14, 345.0), steps=4, id="tongue_rgba"), mask=tongue_mask, id="comp_tongue")
    comp6 = g.blend(comp5, g.hue_ramp(armor_sh, base_lch=(0.91, 0.03, 92.0), steps=4, id="fang_rgba"), mask=fang_mask, id="comp_fangs")
    comp7 = g.blend(comp6, g.hue_ramp(armor_sh, base_lch=(0.22, 0.04, 268.0), steps=5, l_light_delta=0.30, id="talon_rgba"), mask=talon_mask, id="comp_talons")

    # Hero Material 6: 100% SYMMETRIC 7-LAYER SCULPTED DRAGON EYES (eye_l AND head_face_r__eye_l!)
    eye_all_mask = g.mesh_channel("element", pattern=r"(^|/|__)eye_[lr]", id="eye_all_mask")
    eye_outward_mask = g.mesh_channel("outward_eye", pattern=r"(^|/|__)eye_[lr]", id="eye_outward_mask")
    # Paint all 5 non-outward eye faces deep midnight orbital socket so ZERO glow leaks past brow/cheeks!
    socket_rgba = g.constant("#0A0C1B", id="eye_socket_dark")
    comp_eye_sock = g.blend(comp7, socket_rgba, mode="normal", opacity=1.0, mask=eye_all_mask, id="comp_eye_socket")
    # Paint razor-sharp 7-layer dragon eye strictly on BOTH outward lateral faces (west on L, east on R)!
    eye_rgba, eye_iris_em = g.eye_paint(
        lu,
        lv,
        side_sign=side_s,
        iris_inner="#FEF08A",
        iris_mid="#22D3EE",
        iris_outer="#0284C7",
        limbal="#1E1B4B",
        sclera="#0D132B",
        pupil="#050410",
        catchlight="#FFFFFF",
        pupil_shape="slit",
        id="dragon_7layer_eye",
    )
    comp_eyes = g.blend(comp_eye_sock, eye_rgba, mode="normal", opacity=1.0, mask=eye_outward_mask, id="comp_eyes")
    eye_em_masked = g.math(eye_outward_mask, eye_iris_em, op="mul", id="eye_emissive_mask")

    # Hero Material 7: Animated Gemini 4 Neon Energy Veins (Wing Nebula + Curved Scallop Filigree + Slits!)
    em_elem_mask = g.mesh_channel("element", pattern=r"(slit|gravity_cyan|gravity_mag|gem|halo_energy|thruster_core|neon_conduit)", id="neon_elem_mask")

    sail_veins = g.energy_veins(scale=max(9.0, mesh_ctx.width / 44.0), thickness=0.14, warp_strength=0.06, style="nebula", flow_speed=1.0, seed=seed + 137, id="wing_nebula_veins")
    sail_veins_masked = g.math(sail_updown_m, g.levels(sail_veins, in_min=0.42, in_max=0.88, out_max=0.92, id="sail_vein_cut"), op="mul", id="sail_neon_mask")

    # Curved neon filigree tracing the exact Gothic scalloped arch trailing edge!
    sail_scallop_rim_m = g.math(sail_updown_m, scallop_rim, op="mul", id="sail_scallop_rim_mask")

    # Subtle glowing energy channels strictly inside dorsal torso/wing seams
    seam_glow_n = g.noise_3d(scale=0.18, octaves=2, mode="ridged", flow_vec=(0.0, 2.0, 2.0), seed=seed + 173, id="seam_energy_3d")
    energy_zones = g.math(g.mesh_channel("zone", zone="torso,wings", id="energy_zone_m"), g.mesh_channel("joint_cavity", id="cav_ch"), op="mul", id="zone_cav_m")
    core_vein_mask = g.math(
        energy_zones,
        g.levels(seam_glow_n, in_min=0.76, in_max=0.96, out_max=0.50, id="seam_spark_cut"),
        op="mul",
        id="crevice_neon_mask",
    )

    total_neon_mask = g.math(
        g.math(g.math(em_elem_mask, sail_veins_masked, op="max", id="neon_m1"), core_vein_mask, op="max", id="neon_m2"),
        sail_scallop_rim_m,
        op="max",
        id="neon_m3",
    )
    neon_pulsed = g.pulse(total_neon_mask, min_val=0.75, max_val=1.0, freq=1.0, id="neon_pulsed_mask")

    # Radiant dual-tone Antigravity Cyan <-> Gemini 4 Neon Magenta gradient
    pos_grad = g.blend(g.mesh_channel("side_sym", id="wing_span_grad"), g.mesh_channel("front_grad", id="body_z_grad"), mode="normal", opacity=0.45, id="neon_spatial_grad")
    neon_wave = g.scroll(g.fbm(scale=4.0, octaves=2, tileable=True, id="neon_shimmer_n"), speed_x=1.0, speed_y=1.0, id="neon_shimmer_scroll")
    neon_hue_driver = g.blend(pos_grad, neon_wave, mode="soft_light", opacity=0.45, id="neon_hue_driver")
    neon_color = g.hue_ramp(
        neon_hue_driver,
        colors=["#0284C7", "#06B6D4", "#67E8F9", "#C084FC", "#EC4899", "#F472B6"],
        quantize=False,
        id="gemini4_neon_ramp",
    )

    all_em_mask = g.math(neon_pulsed, eye_em_masked, op="max", id="all_emissive_mask")
    all_em_col = g.blend(neon_color, eye_rgba, mode="normal", opacity=1.0, mask=eye_outward_mask, id="all_emissive_col")
    alb_em, em_map = g.emissive(comp_eyes, all_em_mask, color=all_em_col, glow_bleed_px=2, glow_strength=0.44, id="aethelion_glow")

    # Apply vibrant studio polish + carve crisp Gothic scalloped alpha cutouts into wing sails!
    vib_alb = g.vibrance(alb_em, vibrance=0.22, contrast=1.06, lift=0.015, shadow_cool=0.06, id="aethelion_vibrance")
    clean_alb = g.despeckle(vib_alb, threshold=0.22, id="aethelion_despeckled")
    cut_alb1 = g.set_alpha(clean_alb, scallop_a, mask=sail_updown_m, id="aethelion_scalloped_wings")
    cut_alb = g.set_alpha(cut_alb1, g.constant(0.0, id="zero_a"), mask=sail_side_m, id="aethelion_clean_sail_edges")
    cut_em = g.set_alpha(em_map, scallop_a, mask=sail_updown_m, id="aethelion_scalloped_em")

    padded_alb = g.pad_islands(cut_alb, id="final_aethelion_atlas")
    padded_em = g.pad_islands(cut_em, id="final_aethelion_emissive")

    # Full LabPBR 1.3 + Bedrock MER Material Maps for Aethelion!
    h1 = g.blend(armor_h_clean, gold_h, mask=gold_mask, id="pbr_h_1")
    h2 = g.blend(h1, scute_kuw, mask=scute_mask, id="pbr_h_2")
    pbr_height = g.blend(h2, wing_kuw, mask=sail_mask, id="pbr_aethelion_height")
    pbr_normal = g.height_to_normal(pbr_height, strength=2.6, id="pbr_aethelion_normal")

    # Metallic: brushed gold trim (0.94 -> LabPBR Gold ID 231!) + subtle metallic sheen on dragon scales
    pbr_metallic = g.blend(g.constant(0.14, id="obs_met"), g.constant(0.94, id="gold_met"), mask=gold_mask, id="pbr_aethelion_metallic")

    # Roughness: glossy cornea eyes (0.08), polished gold (0.18), enameled obsidian scales (0.30), pearl scutes (0.36), velvet wing silk (0.48)
    r1 = g.blend(g.constant(0.32, id="obs_r"), g.constant(0.18, id="gold_r"), mask=gold_mask, id="pbr_r_1")
    r2 = g.blend(r1, g.constant(0.38, id="scute_r"), mask=scute_mask, id="pbr_r_2")
    r3 = g.blend(r2, g.constant(0.48, id="sail_r"), mask=sail_mask, id="pbr_r_3")
    pbr_roughness = g.blend(r3, g.constant(0.08, id="eye_r"), mask=eye_outward_mask, id="pbr_aethelion_roughness")

    # Subsurface Scattering: luminous translucent wing membranes (0.72) + pearl scutes (0.24)
    sss1 = g.math(sail_mask, g.constant(0.72, id="sail_sss_c"), op="mul", id="pbr_sss_1")
    pbr_sss = g.math(sss1, g.math(scute_mask, g.constant(0.24, id="scute_sss_c"), op="mul", id="pbr_sss_2"), op="max", id="pbr_aethelion_sss")
    pbr_ao = g.mesh_channel("baked_ao", id="pbr_aethelion_ao")

    g.set_output(
        padded_alb,
        emissive_node=padded_em,
        height_node=pbr_height,
        normal_node=pbr_normal,
        roughness_node=pbr_roughness,
        metallic_node=pbr_metallic,
        ao_node=pbr_ao,
        sss_node=pbr_sss,
        normal_strength=2.6,
    )
    return g


def build_rhenoplos_texgraph(mesh_ctx: MeshBakeContext, *, seed: int = 3940) -> TexGraph:
    """Clean, Vibrant, Sun-Baked Expert 3D-Mesh-Aware Full-PBR TexGraph for MH Rhenoplos (Zero Dirt/Mud!).

    Separates PBR Ambient Occlusion cleanly into LabPBR `_n.png` while keeping Albedo crisp,
    vibrant, high-chroma, and readable:
    - Sunlit olive-sage desert reptile hide (cool jade-teal shadows -> warm golden-chartreuse highlights!)
    - Pale warm desert-cream ventral underbelly countershading
    - Crisp sun-baked terracotta-amber hexagonal armored carapace scutes with `1 px` sunlit sandstone bevels
    - Bright clean fluted ivory-sandstone bulldozer crest & hooves (ZERO muddy smudges!)
    - Polished bronze-obsidian nasal horns & tail mace spikes + symmetric 7-layer amber reptile eyes!
    """
    g = TexGraph(
        mesh_ctx.width,
        mesh_ctx.height,
        tileable=False,
        seed=seed,
        name="rhenoplos_v2_vibrant_pbr",
        mesh_context=mesh_ctx,
    )

    # 1. Clean sunlit olive-sage desert hide (crisp subtle scales, gentle albedo AO so ZERO muddiness!)
    hide_n3 = g.noise_3d(scale=0.22, octaves=2, mode="fbm", seed=seed + 1, id="hide_3d_mottle")
    hide_scales = g.scales(cols=max(20, mesh_ctx.width // 14), rows=max(12, mesh_ctx.height // 14), shape="reptile", keel=0.18, seed=seed + 7, id="hide_scales")
    hide_h = g.kuwahara(g.blend(hide_scales, hide_n3, mode="soft_light", opacity=0.22, id="hide_h_raw"), radius=1, sharpness=10.0, id="hide_kuw")
    hide_sh = g.shade(
        hide_h,
        use_mesh=True,
        bump_strength=1.25,
        ao_strength=0.22,      # Keep albedo CLEAN & vibrant; full AO lives in LabPBR _n.png!
        bevel_strength=0.24,
        cavity_strength=0.16,
        rim_strength=0.18,
        id="hide_3d_shade",
    )
    hide_rgba = g.hue_ramp(
        hide_sh,
        base_lch=(0.65, 0.094, 114.0),
        steps=6,
        l_dark_delta=-0.20,
        l_light_delta=0.20,
        hue_shift=(28.0, -18.0),  # cool emerald-teal shadows (NEVER muddy brown!) -> sunlit golden-sage highlights!
        dither="clustered",
        dither_strength=0.08,
        id="desert_hide_rgba",
    )

    # Subtle warm sand-cream ventral belly countershading on downward-facing hide
    belly_m = g.levels(g.mesh_channel("normal_y", id="rh_nrm_y"), in_min=0.05, in_max=0.38, invert=True, out_max=0.72, id="belly_countershade_m")
    belly_rgba = g.hue_ramp(
        hide_sh,
        base_lch=(0.78, 0.055, 96.0),
        steps=5,
        l_dark_delta=-0.16,
        l_light_delta=0.14,
        hue_shift=(-14.0, 10.0),
        id="belly_cream_rgba",
    )
    hide_cs = g.blend(hide_rgba, belly_rgba, mode="normal", opacity=0.85, mask=belly_m, id="hide_with_belly")

    # 2. Vibrant Sun-Baked Terracotta-Amber Tortoise Scute Carapace (clean carved scutes + warm sandstone bevels!)
    carapace_mask = g.mesh_channel("element", pattern=r"(carapace|scute|shell|shield|plate|boss|hump|dorsal|back_shell|mace|club_body|club_side_boss|tail_[1-6]_core|tail_.*_cap|tail_.*_boss|osteoderm|flank_rib|pelvis_cap|waist_cap)", id="carapace_mask")
    scute_mosaic = g.mosaic(scale=max(10.0, mesh_ctx.width / 28.0), jitter=0.65, grout=0.09, bevel=0.18, seed=seed + 23, id="tortoise_scutes")
    scute_kuw = g.kuwahara(scute_mosaic, radius=1, sharpness=10.0, id="scute_kuw")
    scute_sh = g.shade(
        scute_kuw,
        use_mesh=True,
        bump_strength=1.55,
        ao_strength=0.24,
        bevel_strength=0.24,
        cavity_strength=0.20,
        spec_strength=0.18,
        id="carapace_3d_shade",
    )
    carapace_rgba = g.hue_ramp(
        scute_sh,
        base_lch=(0.50, 0.082, 40.0),
        steps=6,
        l_dark_delta=-0.18,
        l_light_delta=0.22,
        hue_shift=(-24.0, 22.0),  # rich warm plum-mahogany seams -> vibrant sun-baked terracotta -> honey-sandstone rim bevels!
        dither="clustered",
        dither_strength=0.06,
        id="carapace_rgba",
    )
    comp1 = g.blend(hide_cs, carapace_rgba, mode="normal", opacity=1.0, mask=carapace_mask, id="comp_carapace")

    # 3. Clean Bright Fluted Ivory-Sandstone Bulldozer Crown Crest & Hooves (ZERO dark grime blotches!)
    crest_mask = g.mesh_channel("element", pattern=r"(crest_|crown|frill|bulldozer|ear_rim|hoof|hooves)", id="crest_mask")
    flutes = g.gabor(scale=22.0, angle_deg=90.0, anisotropy=5.0, octaves=2, seed=seed + 41, id="crest_flutes")
    crest_h = g.kuwahara(g.blend(flutes, hide_n3, mode="soft_light", opacity=0.16, id="crest_h_raw"), radius=1, sharpness=10.0, id="crest_kuw")
    crest_sh = g.shade(
        crest_h,
        use_mesh=True,
        bump_strength=1.35,
        ao_strength=0.20,
        bevel_strength=0.26,
        cavity_strength=0.14,
        spec_strength=0.18,
        id="crest_3d_shade",
    )
    crest_rgba = g.hue_ramp(
        crest_sh,
        base_lch=(0.84, 0.056, 88.0),
        steps=6,
        l_dark_delta=-0.22,
        l_light_delta=0.14,
        hue_shift=(-24.0, 10.0),  # clean warm ochre-sienna flute lines -> luminous ivory-sandstone crown!
        dither="clustered",
        dither_strength=0.06,
        id="crest_rgba",
    )
    comp2 = g.blend(comp1, crest_rgba, mode="normal", opacity=1.0, mask=crest_mask, id="comp_crest")

    # 4. Crisp Polished Obsidian-Bronze Nasal Horns & Tail Mace Spikes
    horn_mask = g.mesh_channel("element", pattern=r"(horn|spike|stud|tusk|nose_tip|beak)", id="horn_mask")
    horn_sh = g.shade(crest_h, use_mesh=True, bump_strength=1.45, ao_strength=0.24, bevel_strength=0.34, spec_strength=0.36, id="horn_3d_shade")
    horn_rgba = g.hue_ramp(
        horn_sh,
        base_lch=(0.34, 0.044, 52.0),
        steps=5,
        l_dark_delta=-0.16,
        l_light_delta=0.30,
        hue_shift=(-22.0, 24.0),
        id="horn_rgba",
    )
    comp3 = g.blend(comp2, horn_rgba, mode="normal", opacity=1.0, mask=horn_mask, id="comp_horns")

    # 5. Crisp Symmetric 7-Layer Amber-Gold Reptile Eyes (strictly on eye_l / eye_r plane, dark socket on eye_socket!)
    lu = g.mesh_channel("local_u", id="rh_lu")
    lv = g.mesh_channel("local_v", id="rh_lv")
    side_s = g.mesh_channel("side_sign", id="rh_side_sign")
    eye_sock_m = g.mesh_channel("element", pattern=r"eye_socket", id="rh_eye_socket")
    eye_plane_m = g.mesh_channel("element", pattern=r"(^|/|__)eye_[lr]$", id="rh_eye_plane")
    comp_esock = g.blend(comp3, g.constant("#1D1912", id="rh_eye_sock"), mask=eye_sock_m, id="comp_rh_eye_sock")
    eye_rgba, eye_em = g.eye_paint(
        lu,
        lv,
        side_sign=side_s,
        iris_inner="#FFFFFF",
        iris_mid="#FEF08A",
        iris_outer="#F59E0B",
        limbal="#B45309",
        sclera="#F59E0B",
        pupil="#0C0806",
        catchlight="#FFFFFF",
        pupil_shape="round",
        id="rh_amber_eye",
    )
    comp_eyes = g.blend(comp_esock, eye_rgba, mask=eye_plane_m, id="comp_rh_eyes")
    rh_eye_em_m = g.math(eye_plane_m, eye_em, op="mul", id="rh_eye_em_mask")
    rh_alb_em, rh_em_map = g.emissive(comp_eyes, rh_eye_em_m, color=eye_rgba, glow_bleed_px=1, glow_strength=0.22, id="rh_eye_glow")

    # Subtle clean vibrance grade + despeckle (ZERO muddy brown dirt!)
    vib = g.vibrance(rh_alb_em, vibrance=0.22, contrast=1.05, lift=0.015, shadow_cool=0.04, id="rhenoplos_vibrance")
    clean = g.despeckle(vib, threshold=0.22, id="rhenoplos_despeckled")
    final = g.pad_islands(clean, id="final_rhenoplos_atlas")
    final_em = g.pad_islands(rh_em_map, id="final_rhenoplos_emissive")

    # Full LabPBR 1.3 + Bedrock MER Material Maps for Rhenoplos!
    rh_h1 = g.blend(hide_h, scute_kuw, mask=carapace_mask, id="pbr_rh_h1")
    pbr_height = g.blend(rh_h1, crest_h, mask=crest_mask, id="pbr_rhenoplos_height")
    pbr_normal = g.height_to_normal(pbr_height, strength=2.6, id="pbr_rhenoplos_normal")

    # Roughness: glossy eye (0.10), polished horn (0.28), semi-gloss mineral carapace (0.40), bone crest (0.52), matte reptile hide (0.72)
    rr1 = g.blend(g.constant(0.72, id="rh_hide_r"), g.constant(0.40, id="rh_cara_r"), mask=carapace_mask, id="pbr_rh_r1")
    rr2 = g.blend(rr1, g.constant(0.52, id="rh_crest_r"), mask=crest_mask, id="pbr_rh_r2")
    rr3 = g.blend(rr2, g.constant(0.28, id="rh_horn_r"), mask=horn_mask, id="pbr_rh_r3")
    pbr_roughness = g.blend(rr3, g.constant(0.10, id="rh_eye_r"), mask=eye_plane_m, id="pbr_rhenoplos_roughness")

    pbr_metallic = g.math(horn_mask, g.constant(0.22, id="rh_horn_m"), op="mul", id="pbr_rhenoplos_metallic")
    pbr_porosity = g.blend(g.constant(0.32, id="rh_hide_por"), g.constant(0.55, id="rh_crest_por"), mask=crest_mask, id="pbr_rhenoplos_porosity")
    pbr_ao = g.mesh_channel("baked_ao", id="pbr_rhenoplos_ao")

    g.set_output(
        final,
        emissive_node=final_em,
        height_node=pbr_height,
        normal_node=pbr_normal,
        roughness_node=pbr_roughness,
        metallic_node=pbr_metallic,
        ao_node=pbr_ao,
        porosity_node=pbr_porosity,
        normal_strength=2.6,
    )
    return g


PRESET_REGISTRY: dict[str, Any] = {
    "neon_hex_conduit": build_neon_hex_block,
    "neon_hex_block": build_neon_hex_block,
    "mossy_carved_brick": build_mossy_carved_brick,
    "molten_dragon_scales": build_molten_scales_block,
    "aethelion_prism_blade": build_prism_blade_item,
    "prism_blade_item": build_prism_blade_item,
}
