"""Reusable Pro Studio TexGraph v3 + Full PBR Recipes (Smooth Gradient Path + Painterly Blockbench Craft).

Distilled directly from master Blockbench studio art, YouTube tutorials (ArtsByKev, Everbloom,
Brandon James Greer, Vijay Dragon Master), and user visual references:
- Smooth-to-pixel hybrid OKLab multi-stop gradients (calm, silky, vibrant, zero static noise!)
- Backlit translucent smooth-gradient dragon wing membranes with bone-proximity AO & organic scalloped arches
- Non-standard luminous gradient dragon eyes (100% L/R mirror-symmetric)
- Full LabPBR 1.3 (_n, _s) + Bedrock RTX (_mer) + synchronized multi-channel animated PBR flipbooks!
"""
from __future__ import annotations

from typing import Any

from .core import TexGraph
from .mesh_baker import MeshBakeContext


# ---------------------------------------------------------------------------
# 1. Seamless Tileable Studio Blocks, Membranes & Item Showcase Recipes
# ---------------------------------------------------------------------------


def build_neon_hex_block(size: int = 32, *, seed: int = 404) -> TexGraph:
    """Seamless animated tileable 'Antigravity x Gemini 4 Neon' Hex Conduit block (Smooth PBR, 16-frame loop)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="neon_hex_conduit_block")

    hex_h = g.hex_grid(cols=4, mortar=0.13, bevel=0.22, output="height", id="hex_tiles")
    hex_edge = g.hex_grid(cols=4, mortar=0.17, output="edge", id="hex_channels")
    metal_grain = g.gabor(scale=12.0, angle_deg=30.0, anisotropy=4.0, octaves=2, id="brushed_grain")
    micro_h = g.blend(hex_h, metal_grain, mode="soft_light", opacity=0.22, id="tile_relief")
    clean_h = g.kuwahara(micro_h, radius=1, sharpness=10.0, id="kuwahara_tiles")
    shaded = g.shade(clean_h, use_mesh=False, bump_strength=1.85, spec_strength=0.34, id="tile_shade")

    # Smooth-to-pixel hybrid OKLab gradient on polished cosmic titanium-indigo hex plates
    obsidian_rgba = g.multi_stop_gradient(
        shaded,
        stops=[
            (0.00, "#12102B"),
            (0.32, "#232559"),
            (0.62, "#394885"),
            (0.85, "#6082BD"),
            (1.00, "#B8E1FF"),
        ],
        smooth_blend=0.62,
        quantize_steps=7,
        dither="clustered",
        dither_strength=0.08,
        id="obsidian_hex_rgba",
    )

    wave = g.fbm(scale=2.0, octaves=2, mode="fbm", loop_speed=1.0, seed=seed + 17, id="plasma_wave")
    scrolled = g.scroll(wave, speed_x=1.0, speed_y=1.0, id="flowing_wave")
    vein_mask = g.math(hex_edge, g.levels(scrolled, in_min=0.10, in_max=0.90, gamma=0.82, id="wave_boost"), op="mul", id="conduit_mask")
    pulsed_mask = g.pulse(vein_mask, min_val=0.62, max_val=1.0, freq=1.0, id="pulsed_conduit")

    neon_col = g.multi_stop_gradient(
        scrolled,
        stops=[(0.0, "#4C1D95"), (0.30, "#C026D3"), (0.60, "#06B6D4"), (0.85, "#67E8F9"), (1.0, "#FFFFFF")],
        smooth_blend=1.0,
        id="neon_palette",
    )

    alb_out, em_out = g.emissive(obsidian_rgba, pulsed_mask, color=neon_col, glow_bleed_px=2, glow_strength=0.48, id="final_hex")
    clean_out = g.despeckle(g.vibrance(alb_out, vibrance=0.18, contrast=1.05, id="hex_vib"), threshold=0.22, id="final_clean")

    pbr_metal = g.levels(hex_edge, in_min=0.2, in_max=0.65, invert=True, out_min=0.0, out_max=0.88, id="pbr_hex_metallic")
    pbr_rough = g.blend(g.constant(0.22, id="cap_rough"), g.constant(0.14, id="glow_rough"), mask=hex_edge, id="pbr_hex_roughness")
    pbr_ao = g.ao_from_height(clean_h, radius=3, strength=0.85, id="pbr_hex_ao")
    pbr_nrm = g.height_to_normal(clean_h, strength=2.8, id="pbr_hex_normal")

    g.set_output(clean_out, emissive_node=em_out, height_node=clean_h, normal_node=pbr_nrm, roughness_node=pbr_rough, metallic_node=pbr_metal, ao_node=pbr_ao, normal_strength=2.8)
    g.expose_param("glow_intensity", "final_hex", "glow_strength", min_val=0.0, max_val=1.0, step=0.05)
    return g


def build_temple_carved_stone(size: int = 32, *, seed: int = 108) -> TexGraph:
    """Sunlit Mayan/Ruin Carved Sandstone Masonry + Emerald Vine Moss (Inspired by Reference Image 6, Full PBR)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="temple_carved_stone_block")

    st_h = g.temple_stone(cols=2, rows=4, carve_motif=True, output="height", seed=seed, id="temple_height")
    st_seam = g.temple_stone(cols=2, rows=4, carve_motif=True, output="seam", seed=seed, id="temple_seam")
    sun_grad = g.gradient_field("linear", angle_deg=-90.0, ease="smoothstep", id="sun_top_grad")
    shaded = g.shade(st_h, use_mesh=False, bump_strength=1.75, spec_strength=0.12, id="stone_shade")
    lit_mix = g.blend(shaded, sun_grad, mode="soft_light", opacity=0.28, id="stone_sun_mix")

    # Warm sunlit sandstone-limestone with silky cool-slate recessed seams (clean like Image 6!)
    stone_rgba = g.multi_stop_gradient(
        lit_mix,
        stops=[
            (0.00, "#3D4754"),
            (0.26, "#697173"),
            (0.54, "#A29E8B"),
            (0.80, "#CDC4A4"),
            (1.00, "#ECE5C8"),
        ],
        smooth_blend=0.65,
        quantize_steps=7,
        id="temple_stone_rgba",
    )

    # Lush emerald jungle moss hugging mortar & hanging gently from top crevices
    turing_m = g.gray_scott(preset="coral", steps=200, sim_scale=1, seed=seed + 29, id="vine_turing")
    drip = g.weathering_drip(st_h, cavity=st_seam, streak_length=5, strength=0.82, decay=0.80, id="vine_drip")
    moss_raw = g.math(turing_m, g.levels(drip, in_min=0.22, in_max=0.74, id="drip_cut"), op="mul", id="moss_spread")
    moss_mask = g.levels(g.iterate(moss_raw, operation="moss_spread", steps=2, strength=0.40, id="moss_it"), in_min=0.26, in_max=0.52, id="moss_alpha")
    moss_rgba = g.multi_stop_gradient(
        g.shade(moss_raw, use_mesh=False, bump_strength=1.4, id="moss_sh"),
        stops=[(0.0, "#14532D"), (0.45, "#22C55E"), (0.82, "#84CC16"), (1.0, "#D9F99D")],
        smooth_blend=0.60,
        quantize_steps=5,
        id="moss_rgba",
    )

    comp = g.blend(stone_rgba, moss_rgba, mode="normal", opacity=0.95, mask=moss_mask, id="moss_over_stone")
    final = g.vibrance(comp, vibrance=0.18, contrast=1.04, lift=0.01, id="final_temple_stone")

    total_h = g.blend(st_h, g.levels(moss_raw, out_min=0.62, out_max=0.95, id="moss_h"), mask=moss_mask, id="pbr_temple_height")
    pbr_nrm = g.height_to_normal(total_h, strength=2.8, id="pbr_temple_normal")
    pbr_rough = g.blend(g.constant(0.68, id="st_r"), g.constant(0.88, id="ms_r"), mask=moss_mask, id="pbr_temple_rough")
    pbr_ao = g.ao_from_height(total_h, radius=3, strength=0.90, id="pbr_temple_ao")
    pbr_por = g.blend(g.constant(0.45, id="st_p"), g.constant(0.85, id="ms_p"), mask=moss_mask, id="pbr_temple_porosity")

    g.set_output(final, height_node=total_h, normal_node=pbr_nrm, roughness_node=pbr_rough, metallic_node=g.constant(0.0, id="st_m"), ao_node=pbr_ao, porosity_node=pbr_por, normal_strength=2.8)
    return g


def build_mossy_carved_brick(size: int = 32, *, seed: int = 108) -> TexGraph:
    """Backwards-compatible alias to the upgraded Mayan/Ruin Carved Sandstone + Moss block."""
    return build_temple_carved_stone(size=size, seed=seed)


def build_molten_scales_block(size: int = 32, *, seed: int = 777) -> TexGraph:
    """Chunky Calm Blockbench Dragon Plates + Smooth Animated Ember Seams (Inspired by Images 2 & 3, Full PBR)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="molten_dragon_scales_block")

    plates_h = g.layered_plates(cols=4, rows=4, stagger=0.5, gap=0.11, top_rim=0.24, variance=0.22, seed=seed, id="chunky_plates")
    plates_seam = g.layered_plates(cols=4, rows=4, stagger=0.5, gap=0.13, output="seam", seed=seed, id="plate_seams")
    brush = g.brushstrokes(angle_deg=-15.0, scale_along=3.0, scale_across=12.0, contrast=0.22, seed=seed + 5, id="plate_brush")
    h_mix = g.blend(plates_h, brush, mode="soft_light", opacity=0.22, id="plate_height")
    sh = g.shade(h_mix, use_mesh=False, bump_strength=1.9, spec_strength=0.28, id="plate_shade")

    # Calm chunky dark obsidian-indigo dragon scales with cool slate highlights (like Image 3!)
    charcoal_rgba = g.multi_stop_gradient(
        sh,
        stops=[
            (0.00, "#181326"),
            (0.32, "#2D2A45"),
            (0.64, "#4C4E6E"),
            (0.86, "#7A82A6"),
            (1.00, "#C4D1E8"),
        ],
        smooth_blend=0.58,
        quantize_steps=6,
        id="charcoal_plates",
    )

    magma_flow = g.scroll(g.fbm(scale=2.5, octaves=2, mode="fbm", tileable=True, seed=seed + 9, id="magma_n"), speed_x=1.0, speed_y=1.0, id="magma_scroll")
    crack_mask = g.math(plates_seam, g.levels(magma_flow, in_min=0.15, in_max=0.85, out_min=0.45, out_max=1.0, id="magma_heat"), op="mul", id="molten_mask")
    pulsed = g.pulse(crack_mask, min_val=0.68, max_val=1.0, freq=1.0, id="molten_pulse")
    magma_rgba = g.multi_stop_gradient(
        magma_flow,
        stops=[(0.0, "#7F1D1D"), (0.35, "#EA580C"), (0.72, "#FACC15"), (1.0, "#FEFCE8")],
        smooth_blend=1.0,
        id="magma_palette",
    )

    alb, em = g.emissive(charcoal_rgba, pulsed, color=magma_rgba, glow_bleed_px=2, glow_strength=0.55, id="final_molten")
    vib = g.vibrance(alb, vibrance=0.20, contrast=1.06, id="molten_vibrance")

    pbr_rough = g.blend(g.constant(0.34, id="pl_r"), g.constant(0.18, id="mg_r"), mask=plates_seam, id="pbr_molten_rough")
    pbr_metal = g.levels(plates_h, in_min=0.60, in_max=0.95, out_min=0.0, out_max=0.45, id="pbr_molten_metal")
    pbr_ao = g.ao_from_height(h_mix, radius=3, strength=0.90, id="pbr_molten_ao")
    pbr_nrm = g.height_to_normal(h_mix, strength=2.9, id="pbr_molten_normal")

    g.set_output(vib, emissive_node=em, height_node=h_mix, normal_node=pbr_nrm, roughness_node=pbr_rough, metallic_node=pbr_metal, ao_node=pbr_ao, normal_strength=2.9)
    return g


def build_stylized_wood_and_foliage(size: int = 32, *, seed: int = 207) -> TexGraph:
    """Warm Painterly Studio Timber Planks + Lush 3-Tone Canopy Trim (Inspired by Reference Image 7, Full PBR)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="stylized_wood_planks_block")

    wood_h = g.wood_planks(planks=4, output="height", seed=seed, id="wood_height")
    wood_seam = g.wood_planks(planks=4, output="seam", seed=seed, id="wood_seam")
    fol_h = g.foliage_clusters(cols=4, rows=4, output="height", seed=seed + 19, id="foliage_h")
    top_strip = g.levels(g.gradient_field("linear", angle_deg=90.0, id="top_y"), in_min=0.0, in_max=0.38, invert=True, id="top_mask_raw")
    leaf_mask = g.levels(g.math(top_strip, fol_h, op="mul", id="leaf_edge"), in_min=0.18, in_max=0.42, id="leaf_alpha")

    wood_rgba = g.multi_stop_gradient(
        g.shade(wood_h, use_mesh=False, bump_strength=1.6, spec_strength=0.14, id="wood_shade"),
        stops=[(0.0, "#4B281B"), (0.30, "#8B5030"), (0.62, "#C2824F"), (0.86, "#E3B27E"), (1.0, "#F6D9AE")],
        smooth_blend=0.72,
        quantize_steps=7,
        id="wood_rgba",
    )
    leaf_rgba = g.multi_stop_gradient(
        g.shade(fol_h, use_mesh=False, bump_strength=1.5, id="leaf_shade"),
        stops=[(0.0, "#1C5926"), (0.42, "#4EA82A"), (0.76, "#8CD431"), (1.0, "#DDF56B")],
        smooth_blend=0.52,
        quantize_steps=5,
        id="leaf_rgba",
    )

    comp = g.blend(wood_rgba, leaf_rgba, mode="normal", opacity=1.0, mask=leaf_mask, id="wood_with_foliage")
    final = g.vibrance(comp, vibrance=0.18, contrast=1.04, id="final_wood_foliage")

    total_h = g.blend(wood_h, g.levels(fol_h, out_min=0.65, out_max=1.0, id="lf_h_lift"), mask=leaf_mask, id="pbr_wood_height")
    pbr_nrm = g.height_to_normal(total_h, strength=2.6, id="pbr_wood_normal")
    pbr_rough = g.blend(g.constant(0.64, id="wd_r"), g.constant(0.80, id="lf_r"), mask=leaf_mask, id="pbr_wood_rough")
    pbr_sss = g.math(leaf_mask, g.constant(0.45, id="lf_sss"), op="mul", id="pbr_wood_sss")
    pbr_ao = g.ao_from_height(total_h, radius=3, strength=0.85, id="pbr_wood_ao")

    g.set_output(final, height_node=total_h, normal_node=pbr_nrm, roughness_node=pbr_rough, metallic_node=g.constant(0.0, id="wd_m"), ao_node=pbr_ao, sss_node=pbr_sss, normal_strength=2.6)
    return g


def build_dragon_wing_membrane_tile(size: int = 64, *, seed: int = 515) -> TexGraph:
    """Dedicated Backlit Translucent Smooth-Gradient Dragon Wing Membrane Showcase (Inspired by Images 1 & 4!)."""
    g = TexGraph(size, size, tileable=False, seed=seed, name="dragon_wing_membrane_studio")
    u = g.gradient("horizontal", id="wu")
    v = g.levels(g.gradient("vertical", id="wv"), invert=True, id="wv_up")

    w = g.smooth_wing(
        u,
        v,
        palette_stops=[
            (0.00, "#3D181E"),
            (0.24, "#7C3028"),
            (0.52, "#C46E41"),
            (0.80, "#EAB27D"),
            (1.00, "#FFF2D6"),
        ],
        rim_color="#59221E",
        arch_depth=0.22,
        arches=1,
        tatter_strength=0.05,
        fold_count=5.0,
        fold_strength=0.14,
        glow_center_strength=0.92,
        bone_shadow_width=0.24,
        rim_width=0.04,
        smooth_blend=0.78,
        quantize_steps=9,
        seed=seed,
        id="amber_dragon_wing",
    )
    pbr_nrm = g.height_to_normal(w["height"], strength=2.2, id="wing_normal")
    pbr_sss = g.math(w["alpha"], g.constant(0.82, id="w_sss"), op="mul", id="wing_sss")
    pbr_rough = g.constant(0.52, id="wing_rough")
    g.set_output(w["rgba"], height_node=w["height"], normal_node=pbr_nrm, roughness_node=pbr_rough, metallic_node=g.constant(0.0, id="w_met"), sss_node=pbr_sss, normal_strength=2.2)
    return g


def build_prism_blade_item(size: int = 32, *, seed: int = 901) -> TexGraph:
    """Crisp animated pixel-art item sprite: Aethelion's Smooth-Gradient Prism Greatsword (Full PBR, 16-frame loop)."""
    g = TexGraph(size, size, tileable=False, seed=seed, name="aethelion_prism_blade_item")

    sil = g.sprite_mask("crystal_blade", id="blade_silhouette")
    guard_m = g.sprite_mask("crystal_blade_guard", id="guard_mask")
    core_m = g.sprite_mask("crystal_blade_core", id="energy_core_mask")

    diag_grad = g.gradient_field("linear", angle_deg=-45.0, ease="smoothstep", id="diag_grad")
    metal_grain = g.gabor(scale=12.0, angle_deg=45.0, anisotropy=4.0, octaves=2, id="blade_grain")
    blade_h = g.blend(diag_grad, metal_grain, mode="soft_light", opacity=0.22, id="blade_relief")
    bevel = g.bevel_emboss(sil, strength=0.52, light_angle_deg=135.0, id="blade_bevel")
    blade_sh = g.blend(blade_h, bevel, mode="overlay", opacity=0.52, id="blade_shaded")

    steel_rgba = g.multi_stop_gradient(
        blade_sh,
        stops=[(0.0, "#1E2959"), (0.35, "#2563EB"), (0.68, "#38BDF8"), (0.90, "#BAE6FD"), (1.0, "#FFFFFF")],
        smooth_blend=0.65,
        quantize_steps=7,
        alpha_mask=sil,
        id="obsidian_steel",
    )
    gold_rgba = g.multi_stop_gradient(
        blade_sh,
        stops=[(0.0, "#6E3B10"), (0.38, "#C9851C"), (0.72, "#F6C84C"), (1.0, "#FFF3B8")],
        smooth_blend=0.65,
        quantize_steps=6,
        alpha_mask=sil,
        id="astrolabe_gold",
    )
    base_sword = g.blend(steel_rgba, gold_rgba, mode="normal", opacity=1.0, mask=guard_m, id="sword_base")

    energy_wave = g.scroll(g.fbm(scale=3.0, octaves=2, tileable=True, id="blade_energy_f"), speed_x=1.0, speed_y=-1.0, id="blade_energy_scroll")
    neon_rgb = g.multi_stop_gradient(
        energy_wave,
        stops=[(0.0, "#7E22CE"), (0.35, "#EC4899"), (0.68, "#22D3EE"), (1.0, "#FFFFFF")],
        smooth_blend=1.0,
        id="blade_neon_col",
    )
    core_pulsed = g.pulse(core_m, min_val=0.72, max_val=1.0, freq=1.0, id="core_pulsed")
    alb, em = g.emissive(base_sword, core_pulsed, color=neon_rgb, glow_bleed_px=1, glow_strength=0.48, id="sword_emissive")
    sel = g.selout(alb, darken=0.50, cool_tint=(0.05, 0.02, 0.22), edge_threshold=0.26, silhouette=True, interior=False, id="blade_selout")
    final = g.vibrance(sel, vibrance=0.20, contrast=1.05, id="final_blade")

    sword_h = g.math(sil, g.blend(bevel, blade_h, opacity=0.45, id="sw_h_mix"), op="mul", id="pbr_blade_height")
    pbr_metal = g.math(sil, g.blend(g.constant(0.88, id="b_met"), g.constant(0.96, id="g_met"), mask=guard_m, id="sw_met_m"), op="mul", id="pbr_blade_metal")
    pbr_rough = g.blend(g.constant(0.15, id="b_r"), g.constant(0.20, id="g_r"), mask=guard_m, id="pbr_blade_rough")
    pbr_nrm = g.height_to_normal(sword_h, strength=3.2, id="pbr_blade_normal")

    g.set_output(final, emissive_node=em, height_node=sword_h, normal_node=pbr_nrm, roughness_node=pbr_rough, metallic_node=pbr_metal, normal_strength=3.2)
    return g


# ---------------------------------------------------------------------------
# 2. 3D-Mesh-Aware Expert Entity Recipes (Smooth Gradient Path + Studio Craft!)
# ---------------------------------------------------------------------------


def build_aethelion_texgraph(mesh_ctx: MeshBakeContext, *, seed: int = 4204) -> TexGraph:
    """Expert Studio Smooth-Gradient + Full-PBR TexGraph for Aethelion v4 (Antigravity x Gemini 4 Neon).

    Distilled directly from the 7 Master Blockbench Studio References & User Feedback:
    - **Non-Standard Luminous Gradient Eyes** (`gradient_eye`, 100% mirror-symmetric on BOTH eye_l AND head_face_r__eye_l!):
      Fierce radiant cyan -> starlight-white -> pale-gold diagonal energy gradient with subtle brow shadow
      (ZERO cartoon black slit pupil, strictly on outward lateral faces, dark socket on other 5 faces!)
    - **Master Studio Backlit Smooth-Gradient Wings** (`smooth_wing`):
      Deep velvety cosmic-indigo bone-proximity AO gradient hugging every wing spar -> silky smooth luminous
      translucent amethyst-to-cyan backlit center -> subtle radial tension folds & painterly brushwork ->
      single graceful concave tattered bat-wing arch per bay (80% calm, ZERO noisy marble confetti!)
    - **Smooth-to-Pixel Hybrid Body Armor**:
      Continuous 3D `body_flow_gradient` + `per_island_gradient` + `directional_brushstrokes` + chunky calm
      `layered_plates` (like Images 2 & 3!) + brushed metallic champagne-gold astrolabe trim (`LabPBR Gold 231`) +
      cohesive atmospheric `color_balance` + full LabPBR 1.3 & Bedrock MER + synchronized 8-frame animated PBR!
    """
    g = TexGraph(
        mesh_ctx.width,
        mesh_ctx.height,
        tileable=False,
        seed=seed,
        name="aethelion_v4_smooth_studio_pbr",
        mesh_context=mesh_ctx,
    )

    # 1. Silky 3D Macro Light Wrap + Per-Face Smooth Gradient + Calm Chunky Dragon Plates + Painterly Brushstrokes
    flow_lit = g.body_flow_gradient("top_light_wrap", gamma=0.95, id="body_light_wrap")
    flow_dv = g.body_flow_gradient("dorsal_ventral", id="body_dorsal_ventral")
    isl_grad = g.per_island_gradient("top_to_bottom", curve=0.88, id="island_top_grad")
    plates = g.layered_plates(
        cols=max(20, mesh_ctx.width // 18),
        rows=max(24, mesh_ctx.height // 18),
        stagger=0.5,
        gap=0.07,
        top_rim=0.20,
        variance=0.16,
        seed=seed + 11,
        id="chunky_dragon_plates",
    )
    brush = g.brushstrokes(angle_deg=-18.0, scale_along=4.0, scale_across=18.0, contrast=0.22, seed=seed + 29, id="painterly_strokes")
    n3_macro = g.noise_3d(scale=0.09, octaves=2, mode="fbm", seed=seed + 1, id="macro_3d_wash")

    # Calm structural heightfield: 65% clean chunky plates + 25% smooth brushwork + 10% 3D wash
    armor_h_raw = g.blend(plates, brush, mode="soft_light", opacity=0.32, id="armor_h_brush")
    armor_h = g.blend(armor_h_raw, n3_macro, mode="soft_light", opacity=0.20, id="armor_h_clean")

    # Silky smooth-to-pixel directional shading: 3D macro wrap + per-face gradient + subtle crisp 1px edge bevels!
    armor_sh_3d = g.shade(
        armor_h,
        use_mesh=True,
        bump_strength=1.25,
        ao_strength=0.26,
        bevel_strength=0.32,
        cavity_strength=0.22,
        rim_strength=0.30,
        spec_strength=0.24,
        id="armor_3d_shade",
    )
    # Blend in smooth macro 3D wrap & per-island gradient so cuboids read silky and painterly like Images 2, 3, 4!
    armor_smooth_sh = g.blend(
        g.blend(armor_sh_3d, flow_lit, mode="normal", opacity=0.34, id="armor_flow_mix"),
        isl_grad,
        mode="soft_light",
        opacity=0.26,
        id="armor_final_shade",
    )

    # Hero Material 1: Silky Cosmic Midnight-Sapphire Armor (deep velvety plum-indigo -> royal sapphire -> cerulean rim!)
    obsidian_rgba = g.multi_stop_gradient(
        armor_smooth_sh,
        stops=[
            (0.00, "#0D0B24"),
            (0.24, "#1C1B4B"),
            (0.50, "#2B3A7A"),
            (0.76, "#4670B8"),
            (0.92, "#7CB8EB"),
            (1.00, "#D1F0FF"),
        ],
        smooth_blend=0.66,
        quantize_steps=8,
        dither="clustered",
        dither_strength=0.06,
        id="obsidian_armor_rgba",
    )

    # Subtle smooth dorsal cosmic-violet sheen sweeping along the crest & spine (inspired by Image 4 atmospheric wash!)
    dorsal_sheen_m = g.levels(g.math(flow_dv, n3_macro, op="mul", id="dorsal_m_raw"), in_min=0.34, in_max=0.78, out_max=0.40, id="dorsal_sheen_mask")
    dorsal_violet = g.multi_stop_gradient(
        armor_smooth_sh,
        stops=[(0.0, "#240E4A"), (0.45, "#6D28D9"), (0.82, "#C084FC"), (1.0, "#F5D0FE")],
        smooth_blend=0.78,
        quantize_steps=7,
        id="dorsal_violet_rgba",
    )
    obsidian_irid = g.blend(obsidian_rgba, dorsal_violet, mode="screen", opacity=0.42, mask=dorsal_sheen_m, id="obsidian_iridescent")

    # Hero Material 2: Silky Brushed Champagne-Gold Astrolabe Armor (LabPBR Gold Metal ID 231!)
    gold_mask = g.mesh_channel("element", pattern=r"(gold|trim|collar|diadem|gemini_star|astrolabe|brow|carpal_hub|elbow_fin|claw_base|vambrace|pauldron|tricep_plate|thigh_armor|patella|knuckle_hood|horn_3|tip_a|tip_b|ring_top|ring_low|nasal_bridge)", id="gold_elem_mask")
    gold_grain = g.gabor(scale=24.0, angle_deg=18.0, anisotropy=4.0, octaves=2, id="gold_brushed_grain")
    gold_h = g.blend(gold_grain, isl_grad, mode="soft_light", opacity=0.30, id="gold_h")
    gold_sh = g.blend(
        g.shade(gold_h, use_mesh=True, bump_strength=1.0, ao_strength=0.28, bevel_strength=0.36, cavity_strength=0.22, spec_strength=0.42, spec_power=16.0, id="gold_3d_shade"),
        flow_lit,
        mode="normal",
        opacity=0.28,
        id="gold_smooth_shade",
    )
    gold_rgba = g.multi_stop_gradient(
        gold_sh,
        stops=[
            (0.00, "#5C2C14"),
            (0.28, "#A8611B"),
            (0.58, "#E3A735"),
            (0.84, "#FAD96B"),
            (1.00, "#FFF9D6"),
        ],
        smooth_blend=0.72,
        quantize_steps=7,
        id="astrolabe_gold_rgba",
    )
    comp1 = g.blend(obsidian_irid, gold_rgba, mode="normal", opacity=1.0, mask=gold_mask, id="comp_gold")

    # Hero Material 3: Silky Opalescent Pearl Ventral Scutes & Ribbed Throat
    scute_mask = g.mesh_channel("element", pattern=r"(ventral|sternum|throat|groin_bridge|pearl|horn_tip|tail_ventral|jaw_ventral)", id="scute_elem_mask")
    scute_ribs = g.layered_plates(cols=12, rows=44, stagger=0.0, gap=0.09, top_rim=0.22, variance=0.08, seed=seed + 41, id="scute_ribs")
    scute_sh = g.blend(
        g.shade(scute_ribs, use_mesh=True, bump_strength=1.15, ao_strength=0.28, bevel_strength=0.22, cavity_strength=0.24, id="scute_3d_shade"),
        isl_grad,
        mode="soft_light",
        opacity=0.25,
        id="scute_smooth_sh",
    )
    scute_rgba = g.multi_stop_gradient(
        scute_sh,
        stops=[(0.0, "#4B5578"), (0.35, "#8E9EC2"), (0.72, "#CEDBF0"), (1.0, "#F5F9FF")],
        smooth_blend=0.75,
        quantize_steps=7,
        id="pearl_scute_rgba",
    )
    comp2 = g.blend(comp1, scute_rgba, mode="normal", opacity=1.0, mask=scute_mask, id="comp_scute")

    # Hero Material 4: MASTER STUDIO BACKLIT SMOOTH-GRADIENT WINGS (Inspired directly by Images 1, 3, 4!)
    sail_mask = g.mesh_channel("element", pattern=r"(wing_lobe|wing_hum_sail|wing_sail|membrane|patagium)", id="wing_sail_mask")
    lu = g.mesh_channel("local_u", id="mesh_lu")
    lv = g.mesh_channel("local_v", id="mesh_lv")
    side_s = g.mesh_channel("side_sign", id="mesh_side_sign")
    up_down_f = g.mesh_channel("face", faces="up,down", id="wing_updown_face")
    sail_updown_m = g.math(sail_mask, up_down_f, op="mul", id="sail_updown_mask")
    sail_side_m = g.math(sail_mask, g.levels(up_down_f, invert=True, id="not_updown"), op="mul", id="sail_edge_faces")

    wing = g.smooth_wing(
        lu,
        lv,
        palette_stops=[
            (0.00, "#120B2B"),  # deep velvety indigo bone-contact AO
            (0.24, "#2B185C"),  # rich royal twilight plum
            (0.50, "#5B2EA8"),  # silky cosmic amethyst mid-membrane
            (0.76, "#A855F7"),  # luminous backlit orchid-violet glow
            (0.91, "#67E8F9"),  # radiant Gemini 4 Neon cyan inner translucency!
            (1.00, "#E0F7FF"),  # soft starlight core highlight
        ],
        rim_color="#38BDF8",
        arch_depth=0.24,
        arches=1,              # single sweeping, majestic Gothic bat-wing concave arch per bay (like Images 1 & 4!)
        tatter_strength=0.055, # subtle organic tattered edge notches (like Images 1 & 3!)
        fold_count=5.0,
        fold_strength=0.14,
        glow_center_strength=0.88,
        bone_shadow_width=0.25,
        rim_width=0.048,
        smooth_blend=0.78,     # silky painterly smooth-gradient wings!
        quantize_steps=9,
        seed=seed + 71,
        id="aethelion_studio_wing",
    )
    comp3 = g.blend(comp2, wing["rgba"], mode="normal", opacity=1.0, mask=sail_mask, id="comp_wings")

    # Hero Material 5: Dark Obsidian Talons, Oral Cavity, Tongue & Ivory Fangs
    oral_mask = g.mesh_channel("element", pattern=r"(palate|pharynx|throat_back|oral_floor|cheek_web)", id="oral_mask")
    tongue_mask = g.mesh_channel("element", pattern=r"tongue", id="tongue_mask")
    fang_mask = g.mesh_channel("element", pattern=r"(teeth|fang)", id="fang_mask")
    talon_mask = g.mesh_channel("element", pattern=r"(talon|claw_tip)", id="talon_mask")
    comp4 = g.blend(comp3, g.multi_stop_gradient(armor_smooth_sh, stops=[(0.0, "#140924"), (1.0, "#3D184D")], smooth_blend=0.8, id="oral_rgba"), mask=oral_mask, id="comp_oral")
    comp5 = g.blend(comp4, g.multi_stop_gradient(armor_smooth_sh, stops=[(0.0, "#521636"), (1.0, "#B84A76")], smooth_blend=0.8, id="tongue_rgba"), mask=tongue_mask, id="comp_tongue")
    comp6 = g.blend(comp5, g.multi_stop_gradient(armor_smooth_sh, stops=[(0.0, "#B8B09E"), (1.0, "#FFFDF5")], smooth_blend=0.8, id="fang_rgba"), mask=fang_mask, id="comp_fangs")
    comp7 = g.blend(comp6, g.multi_stop_gradient(armor_smooth_sh, stops=[(0.0, "#0D0D1A"), (0.7, "#2E3552"), (1.0, "#8DA4CC")], smooth_blend=0.75, id="talon_rgba"), mask=talon_mask, id="comp_talons")

    # Hero Material 6: NON-STANDARD LUMINOUS GRADIENT EYES (User's Favorite Style, 100% Symmetric L & R!)
    eye_all_mask = g.mesh_channel("element", pattern=r"(^|/|__)eye_[lr]", id="eye_all_mask")
    eye_outward_mask = g.mesh_channel("outward_eye", pattern=r"(^|/|__)eye_[lr]", id="eye_outward_mask")
    comp_eye_sock = g.blend(comp7, g.constant("#090B1C", id="eye_socket_dark"), mask=eye_all_mask, id="comp_eye_socket")
    eye_rgba, eye_grad_em = g.gradient_eye(
        lu,
        lv,
        side_sign=side_s,
        stops=[
            (0.00, "#0B3B7A"),
            (0.24, "#06B6D4"),
            (0.54, "#67E8F9"),
            (0.80, "#FEF08A"),
            (1.00, "#FFFFFF"),
        ],
        brow_shadow=0.38,
        angle_deg=-22.0,
        style="gradient_glow",
        rim_color="#090D22",
        id="aethelion_gradient_eye",
    )
    comp_eyes = g.blend(comp_eye_sock, eye_rgba, mode="normal", opacity=1.0, mask=eye_outward_mask, id="comp_eyes")
    eye_em_masked = g.math(eye_outward_mask, eye_grad_em, op="mul", id="eye_emissive_mask")

    # Hero Material 7: Focused Smooth Animated Gemini 4 Neon Energy Accents (Slits + Gentle Backlit Wing Pulse!)
    em_elem_mask = g.mesh_channel("element", pattern=r"(slit|gravity_cyan|gravity_mag|gem|halo_energy|thruster_core|neon_conduit)", id="neon_elem_mask")
    # Gentle smooth luminous breathing wave traveling across the backlit wing core & trailing edge!
    wing_wave = g.scroll(g.fbm(scale=3.0, octaves=2, tileable=True, seed=seed + 131, id="wing_breath_n"), speed_x=1.0, speed_y=1.0, id="wing_breath_scroll")
    wing_soft_em = g.math(
        g.math(sail_updown_m, wing["glow"], op="mul", id="wing_glow_masked"),
        g.levels(wing_wave, in_min=0.05, in_max=0.95, out_min=0.45, out_max=0.95, id="wing_breath_mod"),
        op="mul",
        id="wing_soft_em_mask",
    )
    total_neon_mask = g.math(em_elem_mask, wing_soft_em, op="max", id="neon_m_comb")
    neon_pulsed = g.pulse(total_neon_mask, min_val=0.76, max_val=1.0, freq=1.0, id="neon_pulsed_mask")

    pos_grad = g.blend(g.mesh_channel("side_sym", id="wing_span_grad"), g.mesh_channel("front_grad", id="body_z_grad"), mode="normal", opacity=0.45, id="neon_spatial_grad")
    neon_hue_driver = g.blend(pos_grad, wing_wave, mode="soft_light", opacity=0.40, id="neon_hue_driver")
    neon_color = g.multi_stop_gradient(
        neon_hue_driver,
        stops=[(0.0, "#0EA5E9"), (0.32, "#22D3EE"), (0.58, "#A5F3FC"), (0.80, "#C084FC"), (1.0, "#F472B6")],
        smooth_blend=1.0,
        id="gemini4_neon_ramp",
    )

    all_em_mask = g.math(neon_pulsed, eye_em_masked, op="max", id="all_emissive_mask")
    all_em_col = g.blend(neon_color, eye_rgba, mode="normal", opacity=1.0, mask=eye_outward_mask, id="all_emissive_col")
    alb_em, em_map = g.emissive(comp_eyes, all_em_mask, color=all_em_col, glow_bleed_px=2, glow_strength=0.40, id="aethelion_glow")

    # Cohesive atmospheric color grading + clean vibrance + graceful bat-wing alpha cutouts!
    graded = g.color_balance(alb_em, shadows=(-0.01, -0.005, 0.025), midtones=(0.0, 0.008, 0.015), highlights=(0.02, 0.018, 0.01), id="aethelion_grade")
    vib_alb = g.vibrance(graded, vibrance=0.16, contrast=1.04, lift=0.01, shadow_cool=0.04, id="aethelion_vibrance")
    clean_alb = g.despeckle(vib_alb, threshold=0.22, id="aethelion_despeckled")
    cut_alb1 = g.set_alpha(clean_alb, wing["alpha"], mask=sail_updown_m, id="aethelion_scalloped_wings")
    cut_alb = g.set_alpha(cut_alb1, g.constant(0.0, id="zero_a"), mask=sail_side_m, id="aethelion_clean_sail_edges")
    cut_em = g.set_alpha(em_map, wing["alpha"], mask=sail_updown_m, id="aethelion_scalloped_em")

    padded_alb = g.pad_islands(cut_alb, id="final_aethelion_atlas")
    padded_em = g.pad_islands(cut_em, id="final_aethelion_emissive")

    # Full LabPBR 1.3 + Bedrock MER Material Maps!
    h1 = g.blend(armor_h, gold_h, mask=gold_mask, id="pbr_h_1")
    h2 = g.blend(h1, scute_ribs, mask=scute_mask, id="pbr_h_2")
    pbr_height = g.blend(h2, wing["height"], mask=sail_mask, id="pbr_aethelion_height")
    pbr_normal = g.height_to_normal(pbr_height, strength=2.4, id="pbr_aethelion_normal")

    pbr_metallic = g.blend(g.constant(0.14, id="obs_met"), g.constant(0.94, id="gold_met"), mask=gold_mask, id="pbr_aethelion_metallic")
    r1 = g.blend(g.constant(0.30, id="obs_r"), g.constant(0.17, id="gold_r"), mask=gold_mask, id="pbr_r_1")
    r2 = g.blend(r1, g.constant(0.36, id="scute_r"), mask=scute_mask, id="pbr_r_2")
    r3 = g.blend(r2, g.constant(0.46, id="sail_r"), mask=sail_mask, id="pbr_r_3")
    pbr_roughness = g.blend(r3, g.constant(0.08, id="eye_r"), mask=eye_outward_mask, id="pbr_aethelion_roughness")

    sss1 = g.math(sail_mask, g.constant(0.78, id="sail_sss_c"), op="mul", id="pbr_sss_1")
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
        normal_strength=2.4,
    )
    g.expose_param("wing_smoothness", "aethelion_studio_wing", "smooth_blend", min_val=0.0, max_val=1.0, step=0.05)
    g.expose_param("eye_brow_shadow", "aethelion_gradient_eye", "brow_shadow", min_val=0.0, max_val=0.8, step=0.05)
    g.add_group("Smooth Body Armor", ["body_light_wrap", "chunky_dragon_plates", "painterly_strokes", "obsidian_armor_rgba"], color="#1E293B")
    g.add_group("Backlit Studio Wings", ["aethelion_studio_wing", "comp_wings", "aethelion_scalloped_wings"], color="#2E1065")
    g.add_group("Non-Standard Gradient Eyes", ["aethelion_gradient_eye", "comp_eyes"], color="#083344")
    return g


def build_rhenoplos_texgraph(mesh_ctx: MeshBakeContext, *, seed: int = 3940) -> TexGraph:
    """Clean, Silky Smooth-to-Pixel Hybrid Vibrant Desert PBR TexGraph for MH Rhenoplos v3."""
    g = TexGraph(
        mesh_ctx.width,
        mesh_ctx.height,
        tileable=False,
        seed=seed,
        name="rhenoplos_v3_smooth_studio_pbr",
        mesh_context=mesh_ctx,
    )

    flow_lit = g.body_flow_gradient("top_light_wrap", gamma=0.95, id="rh_light_wrap")
    isl_grad = g.per_island_gradient("top_to_bottom", curve=0.90, id="rh_island_grad")
    brush = g.brushstrokes(angle_deg=12.0, scale_along=4.0, scale_across=16.0, contrast=0.18, seed=seed + 5, id="rh_brush")
    hide_n3 = g.noise_3d(scale=0.18, octaves=2, mode="fbm", seed=seed + 1, id="hide_3d_wash")

    # 1. Silky sunlit olive-sage reptile hide (clean, smooth painterly brushwork, zero mud!)
    hide_h = g.blend(brush, hide_n3, mode="soft_light", opacity=0.35, id="hide_h_clean")
    hide_sh = g.blend(
        g.shade(hide_h, use_mesh=True, bump_strength=1.05, ao_strength=0.20, bevel_strength=0.22, cavity_strength=0.14, rim_strength=0.18, id="hide_3d_shade"),
        g.blend(flow_lit, isl_grad, opacity=0.35, id="rh_macro_grad"),
        mode="normal",
        opacity=0.34,
        id="hide_smooth_sh",
    )
    hide_rgba = g.multi_stop_gradient(
        hide_sh,
        stops=[
            (0.00, "#354F32"),
            (0.30, "#5B7D3B"),
            (0.62, "#89A94E"),
            (0.86, "#B6CF6E"),
            (1.00, "#DFF0A1"),
        ],
        smooth_blend=0.70,
        quantize_steps=7,
        id="desert_hide_rgba",
    )

    belly_m = g.levels(g.mesh_channel("normal_y", id="rh_nrm_y"), in_min=0.05, in_max=0.38, invert=True, out_max=0.72, id="belly_countershade_m")
    belly_rgba = g.multi_stop_gradient(hide_sh, stops=[(0.0, "#8B8A66"), (0.55, "#C7C299"), (1.0, "#EBE5C5")], smooth_blend=0.75, quantize_steps=6, id="belly_cream_rgba")
    hide_cs = g.blend(hide_rgba, belly_rgba, mode="normal", opacity=0.85, mask=belly_m, id="hide_with_belly")

    # 2. Calm, Solid Sun-Baked Terracotta Armored Scute Plates (chunky Blockbench plates, zero noisy mosaic!)
    carapace_mask = g.mesh_channel("element", pattern=r"(carapace|scute|shell|shield|plate|boss|hump|dorsal|back_shell|mace|club_body|club_side_boss|tail_[1-6]_core|tail_.*_cap|tail_.*_boss|osteoderm|flank_rib|pelvis_cap|waist_cap)", id="carapace_mask")
    scute_plates = g.layered_plates(cols=max(10, mesh_ctx.width // 24), rows=max(10, mesh_ctx.height // 24), stagger=0.5, gap=0.07, top_rim=0.20, variance=0.14, seed=seed + 23, id="carapace_plates")
    scute_h = g.blend(scute_plates, brush, mode="soft_light", opacity=0.25, id="scute_h_clean")
    scute_sh = g.blend(
        g.shade(scute_h, use_mesh=True, bump_strength=1.35, ao_strength=0.22, bevel_strength=0.28, cavity_strength=0.18, spec_strength=0.18, id="carapace_3d_shade"),
        isl_grad,
        mode="soft_light",
        opacity=0.26,
        id="carapace_smooth_sh",
    )
    carapace_rgba = g.multi_stop_gradient(
        scute_sh,
        stops=[
            (0.00, "#4A2226"),
            (0.28, "#7D3E35"),
            (0.58, "#AC5F45"),
            (0.84, "#D48F68"),
            (1.00, "#F0C5A1"),
        ],
        smooth_blend=0.68,
        quantize_steps=7,
        id="carapace_rgba",
    )
    comp1 = g.blend(hide_cs, carapace_rgba, mode="normal", opacity=1.0, mask=carapace_mask, id="comp_carapace")

    # 3. Silky Clean Fluted Ivory-Sandstone Bulldozer Crown Crest & Hooves
    crest_mask = g.mesh_channel("element", pattern=r"(crest_|crown|frill|bulldozer|ear_rim|hoof|hooves)", id="crest_mask")
    flutes = g.gabor(scale=20.0, angle_deg=90.0, anisotropy=5.0, octaves=2, seed=seed + 41, id="crest_flutes")
    crest_h = g.blend(flutes, isl_grad, mode="soft_light", opacity=0.28, id="crest_h_clean")
    crest_sh = g.blend(
        g.shade(crest_h, use_mesh=True, bump_strength=1.15, ao_strength=0.18, bevel_strength=0.24, cavity_strength=0.12, spec_strength=0.16, id="crest_3d_shade"),
        flow_lit,
        mode="normal",
        opacity=0.25,
        id="crest_smooth_sh",
    )
    crest_rgba = g.multi_stop_gradient(
        crest_sh,
        stops=[(0.0, "#8B7558"), (0.35, "#C7B593"), (0.72, "#E8DFC5"), (1.0, "#FCF8EB")],
        smooth_blend=0.76,
        quantize_steps=7,
        id="crest_rgba",
    )
    comp2 = g.blend(comp1, crest_rgba, mode="normal", opacity=1.0, mask=crest_mask, id="comp_crest")

    # 4. Polished Bronze-Obsidian Horns & Spikes
    horn_mask = g.mesh_channel("element", pattern=r"(horn|spike|stud|tusk|nose_tip|beak)", id="horn_mask")
    horn_sh = g.shade(crest_h, use_mesh=True, bump_strength=1.3, ao_strength=0.22, bevel_strength=0.32, spec_strength=0.34, id="horn_3d_shade")
    horn_rgba = g.multi_stop_gradient(horn_sh, stops=[(0.0, "#1F1B1C"), (0.45, "#4E433E"), (0.82, "#877568"), (1.0, "#C9B9A7")], smooth_blend=0.70, quantize_steps=6, id="horn_rgba")
    comp3 = g.blend(comp2, horn_rgba, mode="normal", opacity=1.0, mask=horn_mask, id="comp_horns")

    # 5. Luminous Smooth Golden-Amber Gradient Eyes!
    lu = g.mesh_channel("local_u", id="rh_lu")
    lv = g.mesh_channel("local_v", id="rh_lv")
    side_s = g.mesh_channel("side_sign", id="rh_side_sign")
    eye_sock_m = g.mesh_channel("element", pattern=r"eye_socket", id="rh_eye_socket")
    eye_plane_m = g.mesh_channel("element", pattern=r"(^|/|__)eye_[lr]$", id="rh_eye_plane")
    comp_esock = g.blend(comp3, g.constant("#1D1912", id="rh_eye_sock"), mask=eye_sock_m, id="comp_rh_eye_sock")
    eye_rgba, eye_em = g.gradient_eye(
        lu,
        lv,
        side_sign=side_s,
        stops=[(0.0, "#9A3412"), (0.35, "#F59E0B"), (0.72, "#FEF08A"), (1.0, "#FFFFFF")],
        brow_shadow=0.22,
        angle_deg=-15.0,
        style="gradient_glow",
        rim_color="#29180B",
        id="rh_amber_eye",
    )
    comp_eyes = g.blend(comp_esock, eye_rgba, mask=eye_plane_m, id="comp_rh_eyes")
    rh_eye_em_m = g.math(eye_plane_m, eye_em, op="mul", id="rh_eye_em_mask")
    rh_alb_em, rh_em_map = g.emissive(comp_eyes, rh_eye_em_m, color=eye_rgba, glow_bleed_px=1, glow_strength=0.25, id="rh_eye_glow")

    vib = g.vibrance(rh_alb_em, vibrance=0.16, contrast=1.04, lift=0.01, shadow_cool=0.03, id="rhenoplos_vibrance")
    clean = g.despeckle(vib, threshold=0.22, id="rhenoplos_despeckled")
    final = g.pad_islands(clean, id="final_rhenoplos_atlas")
    final_em = g.pad_islands(rh_em_map, id="final_rhenoplos_emissive")

    rh_h1 = g.blend(hide_h, scute_h, mask=carapace_mask, id="pbr_rh_h1")
    pbr_height = g.blend(rh_h1, crest_h, mask=crest_mask, id="pbr_rhenoplos_height")
    pbr_normal = g.height_to_normal(pbr_height, strength=2.4, id="pbr_rhenoplos_normal")

    rr1 = g.blend(g.constant(0.70, id="rh_hide_r"), g.constant(0.42, id="rh_cara_r"), mask=carapace_mask, id="pbr_rh_r1")
    rr2 = g.blend(rr1, g.constant(0.50, id="rh_crest_r"), mask=crest_mask, id="pbr_rh_r2")
    rr3 = g.blend(rr2, g.constant(0.28, id="rh_horn_r"), mask=horn_mask, id="pbr_rh_r3")
    pbr_roughness = g.blend(rr3, g.constant(0.10, id="rh_eye_r"), mask=eye_plane_m, id="pbr_rhenoplos_roughness")

    pbr_metallic = g.math(horn_mask, g.constant(0.22, id="rh_horn_m"), op="mul", id="pbr_rhenoplos_metallic")
    pbr_porosity = g.blend(g.constant(0.30, id="rh_hide_por"), g.constant(0.52, id="rh_crest_por"), mask=crest_mask, id="pbr_rhenoplos_porosity")
    pbr_ao = g.mesh_channel("baked_ao", id="pbr_rhenoplos_ao")

    g.set_output(final, emissive_node=final_em, height_node=pbr_height, normal_node=pbr_normal, roughness_node=pbr_roughness, metallic_node=pbr_metallic, ao_node=pbr_ao, porosity_node=pbr_porosity, normal_strength=2.4)
    return g


PRESET_REGISTRY: dict[str, Any] = {
    "neon_hex_conduit": build_neon_hex_block,
    "neon_hex_block": build_neon_hex_block,
    "temple_carved_stone": build_temple_carved_stone,
    "mossy_carved_brick": build_mossy_carved_brick,
    "molten_dragon_scales": build_molten_scales_block,
    "stylized_wood_and_foliage": build_stylized_wood_and_foliage,
    "dragon_wing_membrane": build_dragon_wing_membrane_tile,
    "aethelion_prism_blade": build_prism_blade_item,
    "prism_blade_item": build_prism_blade_item,
}
