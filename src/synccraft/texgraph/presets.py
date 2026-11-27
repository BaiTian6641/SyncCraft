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
    """5-Pass Industry-Standard Material-Bus Showcase: Carved Ruin Sandstone + Height-Blended Emerald Moss (Full PBR)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="temple_carved_stone_block")

    # Pass 1: Structural Geometry & Weathering Drip Mask Bake
    with g.pass_("pass_1_bake", title="Geometry & Seam Bake", description="Carved sandstone mortar seams & gravity drip paths.", color="#1E293B") as p1:
        st_h = g.temple_stone(cols=2, rows=4, carve_motif=True, output="height", seed=seed, id="temple_height")
        st_seam = g.temple_stone(cols=2, rows=4, carve_motif=True, output="seam", seed=seed, id="temple_seam")
        drip = g.weathering_drip(st_h, cavity=st_seam, streak_length=5, strength=0.82, decay=0.80, id="vine_drip")
        drip_cut = g.levels(drip, in_min=0.18, in_max=0.72, id="drip_mask")
        p1.preview_ref = st_h

    # Pass 2: Macro Material Block-In (Reusable Stone Subgraph)
    with g.pass_("pass_2_macro_blockin", title="Sandstone Block-In", description="Sunlit carved ashlar sandstone MaterialBus subgraph.", color="#422006") as p2:
        stone_mat = g.use_subgraph("carved_ruin_stone_pbr", prefix="ruin_stone", seed=seed, cols=2, rows=4, carve_motif=True)
        p2.preview_ref = f"{stone_mat}.albedo"

    # Pass 3: Surface Sculpt & Substance Height-Based Material Blending (Moss creeps physically through mortar cracks!)
    with g.pass_("pass_3_height_blend", title="Height-Blend Moss", description="Turing coral moss Height-Blended cleanly into stone seams.", color="#14532D") as p3:
        moss_mat = g.use_subgraph("emerald_moss_pbr", prefix="jungle_moss", seed=seed + 29, steps=200)
        moss_grow = g.iterate(g.math(f"{moss_mat}.height", drip_cut, op="mul", id="moss_seed"), operation="moss_spread", steps=2, strength=0.40, id="moss_spread")
        moss_mask = g.levels(moss_grow, in_min=0.22, in_max=0.58, id="moss_blend_mask")
        blended_mat = g.material_height_blend(
            stone_mat,
            moss_mat,
            mask=moss_mask,
            height_offset=0.12,
            contrast=5.5,
            seam_ao_strength=0.30,
            id="stone_moss_height_blend",
        )
        p3.preview_ref = f"{blended_mat}.albedo"

    # Pass 4: Atmospheric Lighting, Contact AO & Color Grading
    with g.pass_("pass_4_grade", title="Lighting & Grade", description="Sunlit warm highlight balance & perceptual vibrance.", color="#1E3A8A") as p4:
        final_mat = g.material_grade(
            blended_mat,
            shadows=(-0.01, 0.0, 0.02),
            highlights=(0.02, 0.015, -0.005),
            vibrance=0.18,
            contrast=1.04,
            lift=0.01,
            id="final_temple_mat",
        )
        p4.preview_ref = f"{final_mat}.albedo"

    g.set_output_material(final_mat, normal_strength=2.8)
    g.expose_param("moss_height_offset", blended_mat, "height_offset", min_val=-0.4, max_val=0.4, step=0.05)
    g.expose_param("moss_blend_contrast", blended_mat, "contrast", min_val=1.0, max_val=12.0, step=0.5)
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
    """Monumental Normal Dragon Wing Membrane Studio Showcase (Arched Tendon Ribs + Diagonal Pleats + Battle Tatters + Backlit SSS)."""
    g = TexGraph(size, size, tileable=False, seed=seed, name="dragon_wing_membrane_studio")
    with g.pass_("pass_1_coords", title="Wing Coords", description="Aligned span_u & trail_v wing coordinates.") as p1:
        u = g.gradient("horizontal", id="wu")
        v = g.gradient("vertical", id="wv")
        p1.preview_ref = v
    with g.pass_("pass_2_membrane", title="Colossal Membrane", description="Bone-hugging AO, curved cross-bay tendon ribs, diagonal pleats & tattered concave arch.") as p2:
        wing_mat = g.colossal_dragon_membrane(
            u,
            v,
            palette_stops=[
                (0.00, "#260F16"),
                (0.24, "#5C2122"),
                (0.52, "#9F4732"),
                (0.78, "#DD8E58"),
                (0.94, "#F8CD94"),
                (1.00, "#FFF5DC"),
            ],
            rim_color="#3E1719",
            tendon_color="#401A1B",
            arch_depth=0.24,
            arches=1,
            tatter_strength=0.085,
            tendon_ribs=3,
            pleat_count=9.0,
            pleat_strength=0.18,
            backlit_sss=0.88,
            smooth_blend=0.76,
            quantize_steps=9,
            seed=seed,
            id="amber_dragon_wing",
        )
        p2.preview_ref = f"{wing_mat}.albedo"
    g.set_output_material(wing_mat, normal_strength=2.5)
    return g


def build_arcane_ember_vfx_block(size: int = 32, *, seed: int = 808) -> TexGraph:
    """Pixel-Composer-Parity 4-Stage Pipeline Showcase: Generate -> Iterate -> Simulate -> Animate (16-Frame Looping PBR)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="arcane_ember_vfx_block")

    # Stage 1: GENERATE — Dark Basalt Frame & Glowing Forge Conduit Mask
    with g.pass_("pass_1_generate", title="Generate Base", description="Basalt masonry tiles & forge vent emitter mask.", color="#1E293B") as p1:
        plates = g.layered_plates(cols=3, rows=3, stagger=0.5, gap=0.14, top_rim=0.22, variance=0.18, seed=seed, id="basalt_plates")
        seams = g.layered_plates(cols=3, rows=3, stagger=0.5, gap=0.16, output="seam", seed=seed, id="basalt_seams")
        sh = g.shade(plates, use_mesh=False, bump_strength=1.9, spec_strength=0.22, id="basalt_sh")
        basalt_alb = g.multi_stop_gradient(
            sh,
            stops=[(0.0, "#120D1D"), (0.35, "#231F38"), (0.70, "#3D3C5C"), (1.0, "#757BA3")],
            smooth_blend=0.62,
            quantize_steps=6,
            id="basalt_rgba",
        )
        basalt_mat = g.material_make(basalt_alb, height=plates, roughness=0.52, metallic=0.12, ao=g.ao_from_height(plates, radius=3, id="basalt_ao"), normal_strength=2.8, id="basalt_mat")
        p1.preview_ref = f"{basalt_mat}.albedo"

    # Stage 2: ITERATE — Cascading Inline Curl-Warp Erosion on Forge Seam Energy
    with g.pass_("pass_2_iterate", title="Iterate Inline", description="Multi-step recursive curl-warp cascade along forge fissures.", color="#431407") as p2:
        warped_seams = g.iterate_inline(seams, mode="warp_cascade", steps=4, strength=0.28, decay=0.78, seed=seed + 13, id="iterated_fissures")
        p2.preview_ref = warped_seams

    # Stage 3: SIMULATE — 2D Fluid Smoke/Flame Advection + Looping Ember Particles + Feedback Trail!
    with g.pass_("pass_3_simulate", title="Simulate Fluid & Sparks", description="Semi-Lagrangian 2D flame advection + looping pixel particles + temporal feedback trails.", color="#7C2D12") as p3:
        fluid = g.fluid_smoke_sim(
            source_mask=warped_seams,
            steps=14,
            buoyancy=1.75,
            curl_strength=1.95,
            dissipation=0.92,
            seed=seed + 29,
            color_stops=[(0.0, "#1C0818"), (0.30, "#9A1B29"), (0.60, "#EA580C"), (0.85, "#FACC15"), (1.0, "#FFFBEB")],
            id="forge_fluid_sim",
        )
        sparks = g.particle_sim(
            count=28,
            emitter_mask=seams,
            vel_y=-0.65,
            spread=0.22,
            gravity_y=-0.12,
            turbulence=0.25,
            size_px=1.5,
            trail_steps=3,
            seed=seed + 41,
            color_stops=[(0.0, "#EA580C"), (0.55, "#FACC15"), (1.0, "#FFFFFF")],
            id="ember_particles",
        )
        # Stateful Feedback Inline buffer leaving crisp glowing smoke/spark trails across frames!
        trails = g.feedback_loop(f"{sparks}.density", decay=0.76, drift_x_px=0.0, drift_y_px=-1.0, blur_radius=1, pre_roll=5, loop_frames=16, id="spark_feedback")
        vfx_mask = g.math(g.math(f"{fluid}.density", warped_seams, op="max", id="fl_seam_m"), trails, op="max", id="vfx_em_mask")
        p3.preview_ref = f"{fluid}.rgba"

    # Stage 4: ANIMATE — Timeline Parameter Drivers (Wavetable + Wiggler) & Final PBR Output!
    with g.pass_("pass_4_animate", title="Animate Drivers & PBR", description="Seamless wiggler & sine drivers pulsing glow intensity and PBR emissive output.", color="#312E81") as p4:
        em_comp = g.emissive_composite(
            f"{basalt_mat}.albedo",
            vfx_mask,
            color=f"{fluid}.rgba",
            glow_bleed_px=2,
            glow_strength=0.58,
            id="animated_forge_glow",
        )
        final_mat = g.material_make(
            f"{em_comp}.albedo",
            height=f"{basalt_mat}.height",
            normal=f"{basalt_mat}.normal",
            roughness=g.blend(f"{basalt_mat}.roughness", g.constant(0.18, id="lava_r"), mask=vfx_mask, id="final_r"),
            metallic=f"{basalt_mat}.metallic",
            ao=f"{basalt_mat}.ao",
            emissive=f"{em_comp}.emissive",
            normal_strength=2.8,
            id="final_arcane_ember_mat",
        )
        p4.preview_ref = f"{final_mat}.albedo"

    # Bind real Pixel-Composer-style procedural animation drivers!
    g.bind_driver(em_comp, "glow_strength", kind="wiggler", min_val=0.45, max_val=0.78, freq=1.0, octaves=3, seed=seed + 7)
    g.set_output_material(final_mat, normal_strength=2.8)
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


def build_aethelion_texgraph(mesh_ctx: MeshBakeContext, *, seed: int = 4205) -> TexGraph:
    """Industry-Standard 5-Pass Material-Bus TexGraph for Aethelion v5 (Antigravity x Gemini 4 Neon).

    Structured into 5 Explicit Sequential Pipeline Passes (Substance Designer / MaterialX / Pixel Composer parity):
    - **Pass 1 (`pass_1_bake`) — 3D Geometry, Wing Coords & Semantic Masks**:
      Bakes 3D light wrap, dorsal-ventral flow, per-island gradients, baked AO, top/bottom-aligned
      `wing_span_u` & `wing_trail_v` (ZERO inverted dorsal/ventral cutouts!), and anatomical masks.
    - **Pass 2 (`pass_2_macro_blockin`) — Macro Material Block-In (25% Squint Readability)**:
      Instantiates reusable PBR Material Subgraphs (`dragon_armor_plates_pbr`, `brushed_metal_pbr`,
      `pearl_scutes_pbr`) plus oral/talon/fang material buses and composites them into a cohesive base bus.
    - **Pass 3 (`pass_3_surface_sculpt`) — Sculpted Aerodynamic Pinion Wings & Height-Blended Relief**:
      Builds Aethelion's signature **Sculpted Metallic Obsidian Pinion-Feather Louvers** (`sculpted_pinion_wing_pbr`
      subgraphs — razor swept lance blades, brushed champagne-gold astrolabe anchor arc, subtle barb micro-relief,
      and **Twin-Spectrum Horizon Cyan outer (`f1/f1d/f2/f2d`) + Cosmic Violet-Magenta inner (`f3/f4/fore`)**
      recessed energy quill filaments!) — 100% aligned top & bottom, zero cupcake blobs, zero colliding slabs!
    - **Pass 4 (`pass_4_lighting_grade`) — 3D Contact AO, Atmospheric Color Balance & Despeckle**:
      Unified `material_grade` pass applying subtle cool contact shadow depth, atmospheric grading & vibrance.
    - **Pass 5 (`pass_5_emissive_pbr`) — Bilateral Gradient Eyes, Neon Conduits, Wing Cutouts & LabPBR Final**:
      Paints bilateral symmetric **non-standard luminous gradient eyes** (`gradient_eye_multi`), animates
      focused Gemini 4 Neon conduits & pinion quill pulses, applies crisp 100% top/bottom aligned hard-alpha
      wing cutouts, clamp-bleeds ALL PBR channels into UV padding, and binds via `set_output_material(...)`!
    """
    g = TexGraph(
        mesh_ctx.width,
        mesh_ctx.height,
        tileable=False,
        seed=seed,
        name="aethelion_v6_celestial_pbr",
        mesh_context=mesh_ctx,
    )

    # =========================================================================
    # PASS 1: 3D Geometry Bake, Exact Bay-Sector Wing Coords & Material Masks
    # =========================================================================
    with g.pass_(
        "pass_1_bake",
        title="3D Bake & Masks",
        description="3D muscle cavities, exact 3D finger-to-finger bay coords & anatomical masks.",
        color="#1E293B",
    ) as p1:
        baked_ao = g.mesh_channel("baked_ao", id="mesh_baked_ao")
        flow_lit = g.body_flow_gradient("top_light_wrap", gamma=0.95, id="body_light_wrap")
        flow_dv = g.body_flow_gradient("dorsal_ventral", id="body_dorsal_ventral")
        isl_grad = g.per_island_gradient("top_to_bottom", curve=0.88, id="island_top_grad")
        lu = g.mesh_channel("local_u", id="mesh_lu")
        lv = g.mesh_channel("local_v", id="mesh_lv")
        side_s = g.mesh_channel("side_sign", id="mesh_side_sign")
        bay_u = g.mesh_channel("wing_bay_u", id="mesh_wing_bay_u")
        bay_v = g.mesh_channel("wing_bay_v", id="mesh_wing_bay_v")
        bay_sector_m = g.mesh_channel("wing_bay_mask", id="mesh_wing_bay_mask")
        face_dn = g.mesh_channel("face_is_down", id="mesh_face_is_down")
        up_down_f = g.mesh_channel("face", faces="up,down", id="wing_updown_face")
        not_updown = g.levels(up_down_f, invert=True, id="not_updown")

        gold_mask = g.mesh_channel(
            "element",
            pattern=r"(gold|trim|collar|diadem|gemini_star|astrolabe|brow|carpal_hub|elbow_fin|claw_base|vambrace|pauldron|tricep_plate|thigh_armor|patella|knuckle_hood|horn_3|tip_a|tip_b|ring_top|ring_low|nasal_bridge)",
            id="gold_elem_mask",
        )
        scute_mask = g.mesh_channel(
            "element",
            pattern=r"(ventral|sternum|throat|groin_bridge|pearl|horn_tip|tail_ventral|jaw_ventral)",
            id="scute_elem_mask",
        )
        oral_mask = g.mesh_channel("element", pattern=r"(palate|pharynx|throat_back|oral_floor|cheek_web)", id="oral_mask")
        tongue_mask = g.mesh_channel("element", pattern=r"tongue", id="tongue_mask")
        fang_mask = g.mesh_channel("element", pattern=r"(teeth|fang)", id="fang_mask")
        talon_mask = g.mesh_channel("element", pattern=r"(talon|claw_tip)", id="talon_mask")
        eye_all_mask = g.mesh_channel("element", pattern=r"(^|/|__)eye_[lr]", id="eye_all_mask")
        eye_outward_mask = g.mesh_channel("outward_eye", pattern=r"(^|/|__)eye_[lr]", id="eye_outward_mask")

        # Separate masks for Twin-Spectrum Attached Wing Membrane Bays (Cyan outer 1-2 vs Horizon-Magenta inner 3-4+sail!)
        wing_all_mask = g.mesh_channel(
            "element",
            pattern=r"(wing_bay_|wing_inner_sail|inner_sail|wing_lobe|wing_hum_sail|wing_sail|membrane|patagium)",
            id="wing_lobe_all_mask",
        )
        wing_cyan_m = g.mesh_channel("element", pattern=r"(wing_bay_(1|2)|wing_lobe_(1|1d|2|2d))$", id="wing_bay_cyan_mask")
        wing_mag_m = g.mesh_channel("element", pattern=r"(wing_bay_(3|4)|wing_inner_sail|wing_lobe_(3|4|fore))$", id="wing_bay_mag_mask")
        wing_updown_m = g.math(wing_all_mask, up_down_f, op="mul", id="lobe_updown_mask")
        wing_side_m = g.math(wing_all_mask, not_updown, op="mul", id="lobe_side_mask")

        p1.preview_ref = baked_ao

    # =========================================================================
    # PASS 2: 3D Muscle-Fitted Obsidian Armor, Brushed Gold & Pearl Scutes
    # =========================================================================
    with g.pass_(
        "pass_2_macro_blockin",
        title="Muscle Armor & Gold",
        description="3D-fitted midnight-obsidian muscle scales, brushed metallic gold & opalescent scutes.",
        color="#172554",
    ) as p2:
        obsidian_mat = g.use_subgraph(
            "anatomical_muscle_scales_pbr",
            prefix="obsidian_muscle",
            seed=seed + 11,
            plate_cols=max(56, mesh_ctx.width // 8),
            plate_rows=max(56, mesh_ctx.height // 7),
            scale_stops=[
                (0.00, "#09081B"),
                (0.22, "#161B47"),
                (0.48, "#263B7E"),
                (0.74, "#406AB8"),
                (0.91, "#74ADE8"),
                (1.00, "#C7EAFF"),
            ],
            belly_stops=[
                (0.00, "#475569"),
                (0.45, "#94A3B8"),
                (0.80, "#CBD5E1"),
                (1.00, "#F1F5F9"),
            ],
            muscle_groove_depth=0.42,
            scale_relief=0.56,
            organic_distortion=0.64,
            softness=0.50,
            hue_variation=0.38,
            roughness_base=0.36,
            metallic_crown=0.20,
            smooth_blend=0.52,
            quantize_steps=7,
        )

        # Subtle smooth dorsal cosmic-violet iridescence sweeping along crest & spine
        n3_macro = g.noise_3d(scale=0.09, octaves=2, mode="fbm", seed=seed + 1, id="macro_3d_wash")
        dorsal_sheen_m = g.levels(g.math(flow_dv, n3_macro, op="mul", id="dorsal_m_raw"), in_min=0.36, in_max=0.80, out_max=0.34, id="dorsal_sheen_mask")
        dorsal_violet = g.multi_stop_gradient(
            f"{obsidian_mat}.height",
            stops=[(0.0, "#240E4A"), (0.45, "#6D28D9"), (0.82, "#C084FC"), (1.0, "#F5D0FE")],
            smooth_blend=0.78,
            quantize_steps=7,
            id="dorsal_violet_rgba",
        )
        dorsal_violet_mat = g.material_make(dorsal_violet, height=f"{obsidian_mat}.height", roughness=0.26, metallic=0.22, ao=baked_ao, id="dorsal_violet_mat")
        obs_irid_mat = g.material_blend(obsidian_mat, dorsal_violet_mat, mask=dorsal_sheen_m, opacity=0.36, albedo_mode="screen", id="obsidian_irid_mat")

        # Brushed Champagne-Gold Astrolabe Armor Subgraph (LabPBR Gold Metal ID 231!)
        gold_mat = g.use_subgraph(
            "brushed_metal_pbr",
            prefix="gold_astrolabe",
            inputs={"flow_lit": flow_lit, "isl_grad": isl_grad, "ao": baked_ao},
            seed=seed + 23,
            roughness=0.16,
            metallic=0.94,
        )

        # Opalescent Pearl Ventral Scutes Subgraph
        scute_mat = g.use_subgraph(
            "pearl_scutes_pbr",
            prefix="pearl_scutes",
            inputs={"isl_grad": isl_grad, "ao": baked_ao},
            seed=seed + 41,
            roughness=0.36,
            sss=0.24,
        )

        # Oral, Tongue, Ivory Fang & Obsidian Talon Material Buses
        oral_mat = g.material_make(
            g.multi_stop_gradient(flow_lit, stops=[(0.0, "#140924"), (1.0, "#3D184D")], smooth_blend=0.8, id="oral_rgba"),
            height=f"{obsidian_mat}.height", roughness=0.42, metallic=0.0, ao=baked_ao, sss=0.20, id="oral_mat",
        )
        tongue_mat = g.material_make(
            g.multi_stop_gradient(flow_lit, stops=[(0.0, "#521636"), (1.0, "#B84A76")], smooth_blend=0.8, id="tongue_rgba"),
            height=f"{obsidian_mat}.height", roughness=0.32, metallic=0.0, ao=baked_ao, sss=0.38, id="tongue_mat",
        )
        fang_mat = g.material_make(
            g.multi_stop_gradient(flow_lit, stops=[(0.0, "#B8B09E"), (1.0, "#FFFDF5")], smooth_blend=0.8, id="fang_rgba"),
            height=f"{obsidian_mat}.height", roughness=0.22, metallic=0.0, ao=baked_ao, id="fang_mat",
        )
        talon_mat = g.material_make(
            g.multi_stop_gradient(flow_lit, stops=[(0.0, "#0D0D1A"), (0.7, "#2E3552"), (1.0, "#8DA4CC")], smooth_blend=0.75, id="talon_rgba"),
            height=f"{obsidian_mat}.height", roughness=0.20, metallic=0.32, ao=baked_ao, id="talon_mat",
        )

        # Composite full body MaterialBus stack (with Height-Blend on gold trim over obsidian plates!)
        body_m1 = g.material_height_blend(obs_irid_mat, gold_mat, mask=gold_mask, height_offset=0.08, contrast=8.0, seam_ao_strength=0.18, id="body_gold_blend")
        body_m2 = g.material_blend(body_m1, scute_mat, mask=scute_mask, id="body_scute_blend")
        body_m3 = g.material_blend(body_m2, oral_mat, mask=oral_mask, id="body_oral_blend")
        body_m4 = g.material_blend(body_m3, tongue_mat, mask=tongue_mask, id="body_tongue_blend")
        body_m5 = g.material_blend(body_m4, fang_mat, mask=fang_mask, id="body_fang_blend")
        body_mat = g.material_blend(body_m5, talon_mat, mask=talon_mask, id="body_macro_mat")
        p2.preview_ref = f"{body_mat}.albedo"

    # =========================================================================
    # PASS 3: 100%-Attached Twin-Spectrum Celestial Astrolabe Wing Membranes!
    # =========================================================================
    with g.pass_(
        "pass_3_celestial_wings",
        title="Attached Celestial Wings",
        description="100% finger-anchored Twin-Spectrum Cyan/Magenta translucent membranes with gold astrolabe tendon ribs.",
        color="#3B0764",
    ) as p3:
        quill_wave = g.scroll(g.fbm(scale=3.0, octaves=2, tileable=True, seed=seed + 131, id="quill_wave_n"), speed_x=1.0, speed_y=1.0, id="quill_wave_scroll")
        wing_in = {"span_u": bay_u, "trail_v": bay_v, "face_is_down": face_dn}

        # Outer Bays (wing_bay_1, wing_bay_2): TRUE Dual-Layer 0.5u Wing Sheet!
        # Dorsal (`up`): Velvet astral-sapphire armored membrane + champagne-gold astrolabe ribs!
        # Ventral (`down`): Luminous backlit Starlight-Cyan translucent sail (`SSS=0.90`)!
        mem_cyan = g.use_subgraph(
            "colossal_dragon_membrane_pbr",
            prefix="aeth_mem_cyan",
            inputs=wing_in,
            seed=seed + 71,
            dorsal_palette_stops=[
                (0.00, "#0A0C22"),
                (0.24, "#152354"),
                (0.52, "#1E4280"),
                (0.78, "#2E75B2"),
                (0.93, "#62BCE0"),
                (1.00, "#C7F2FF"),
            ],
            palette_stops=[
                (0.00, "#0C0D26"),
                (0.24, "#152B63"),
                (0.50, "#1D5CA1"),
                (0.76, "#2EA7DE"),
                (0.92, "#79EEFA"),
                (1.00, "#E8FCFF"),
            ],
            rim_color="#1D365C",
            tendon_color="#236596",
            dorsal_tendon_color="#D4A438",
            arch_depth=0.22,
            arches=1,
            tatter_strength=0.045,
            tendon_ribs=3,
            pleat_count=9.0,
            pleat_strength=0.15,
            organic_distortion=0.58,
            backlit_sss=0.90,
            smooth_blend=0.80,
            quantize_steps=8,
        )

        # Inner Bays (wing_bay_3, wing_bay_4, wing_inner_sail): TRUE Dual-Layer 0.5u Wing Sheet!
        # Dorsal (`up`): Royal obsidian-amethyst protective outer mantle + champagne-gold astrolabe ribs!
        # Ventral (`down`): Radiant backlit Horizon-Magenta translucent underwing sail (`SSS=0.90`)!
        mem_mag = g.use_subgraph(
            "colossal_dragon_membrane_pbr",
            prefix="aeth_mem_mag",
            inputs=wing_in,
            seed=seed + 83,
            dorsal_palette_stops=[
                (0.00, "#0F0A22"),
                (0.24, "#27144D"),
                (0.52, "#4B2280"),
                (0.78, "#823AA8"),
                (0.93, "#C47CE0"),
                (1.00, "#F0D4FF"),
            ],
            palette_stops=[
                (0.00, "#140B29"),
                (0.24, "#35165E"),
                (0.50, "#6E23A3"),
                (0.76, "#BC3CD9"),
                (0.92, "#F08CFA"),
                (1.00, "#FCEBFF"),
            ],
            rim_color="#38155B",
            tendon_color="#6E2285",
            dorsal_tendon_color="#D4A438",
            arch_depth=0.22,
            arches=1,
            tatter_strength=0.045,
            tendon_ribs=3,
            pleat_count=9.0,
            pleat_strength=0.15,
            organic_distortion=0.58,
            backlit_sss=0.90,
            smooth_blend=0.80,
            quantize_steps=8,
        )

        # Composite BOTH Twin-Spectrum attached membranes cleanly into the MaterialBus!
        w_m1 = g.material_blend(body_mat, mem_cyan, mask=wing_cyan_m, id="comp_wing_cyan")
        with_wings_mat = g.material_blend(w_m1, mem_mag, mask=wing_mag_m, id="body_and_wings_mat")

        # Exact 100%-attached bay-sector hard-alpha mask (solid all the way to every bounding spar!)
        a1 = g.blend(g.constant(1.0, id="solid_one"), f"{mem_cyan}.alpha", mask=wing_cyan_m, id="wing_alpha_1")
        a2 = g.blend(a1, f"{mem_mag}.alpha", mask=wing_mag_m, id="wing_alpha_2")
        wing_alpha_all = g.math(a2, bay_sector_m, op="mul", id="wing_alpha_all")

        # Subtle luminous energy tracing strictly along curved astrolabe tendon ribs (focused accent!)
        t_c = g.levels(f"{mem_cyan}.tendon_mask", in_min=0.74, in_max=0.98, out_max=0.45, id="tendon_glow_c")
        t_m = g.levels(f"{mem_mag}.tendon_mask", in_min=0.74, in_max=0.98, out_max=0.45, id="tendon_glow_m")
        q1 = g.blend(g.constant(0.0, id="zero_q"), t_c, mask=wing_cyan_m, id="quill_g1")
        quill_glow_all = g.math(g.blend(q1, t_m, mask=wing_mag_m, id="quill_g2"), wing_updown_m, op="mul", id="quill_glow_masked")

        p3.preview_ref = f"{with_wings_mat}.albedo"

    # =========================================================================
    # PASS 4: Contact AO, Cohesive Atmospheric Color Grading & Clean Despeckle
    # =========================================================================
    with g.pass_(
        "pass_4_lighting_grade",
        title="Lighting & Grading",
        description="Cool shadow contact AO, atmospheric color balance, vibrance & despeckle.",
        color="#0F172A",
    ) as p4:
        graded_mat = g.material_grade(
            with_wings_mat,
            ao_bake=baked_ao,
            ao_darken=0.14,
            shadows=(-0.008, -0.004, 0.026),
            midtones=(0.004, 0.010, 0.016),
            highlights=(0.024, 0.020, 0.012),
            vibrance=0.20,
            contrast=1.03,
            lift=0.014,
            shadow_cool=0.05,
            atmosphere_strength=0.22,
            sky_tint=(0.24, 0.42, 0.72),
            bounce_tint=(0.26, 0.14, 0.44),
            organic_wash_strength=0.10,
            despeckle_threshold=0.38,
            id="aethelion_graded_mat",
        )
        p4.preview_ref = f"{graded_mat}.albedo"

    # =========================================================================
    # PASS 5: Bilateral Non-Standard Gradient Eyes, Focused Emissive, Cutouts & PBR Pad
    # =========================================================================
    with g.pass_(
        "pass_5_emissive_pbr",
        title="Eyes, Neon & PBR Final",
        description="Bilateral non-standard gradient eyes, Twin-Spectrum emissive slits, 100% attached wing alpha & island padding.",
        color="#083344",
    ) as p5:
        # Bilateral Non-Standard Luminous Gradient Eyes (User's Favorite — 100% Mirror-Symmetric L & R!)
        eye_socket_mat = g.material_make(g.constant("#090B1C", id="eye_socket_dark"), roughness=0.45, metallic=0.0, ao=baked_ao, id="eye_socket_mat")
        with_sock_mat = g.material_blend(graded_mat, eye_socket_mat, mask=eye_all_mask, id="comp_eye_socket_mat")

        eye_multi = g.gradient_eye_multi(
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
        eye_em_masked = g.math(eye_outward_mask, f"{eye_multi}.emissive_mask", op="mul", id="eye_em_masked")
        eye_mat = g.material_make(
            f"{eye_multi}.albedo",
            height=g.constant(0.65, id="eye_h_const"),
            roughness=0.08,
            metallic=0.0,
            ao=g.constant(1.0, id="eye_ao_const"),
            id="eye_pbr_mat",
        )
        with_eyes_mat = g.material_blend(with_sock_mat, eye_mat, mask=eye_outward_mask, id="comp_eyes_mat")

        # Focused Twin-Spectrum Neon Energy Accents (Cyan slits/gems + Magenta f3/f4 slits + Curved Ribs + Eyes!)
        cyan_slit_m = g.mesh_channel("element", pattern=r"(f1_.*slit|f2_.*slit|gravity_cyan|carpal_gem|gemini_star|chest_core_gem|halo_energy|thruster_core|neon_conduit|tail_prism_cyan|tail_vane_slit)", id="cyan_slit_mask")
        mag_slit_m = g.mesh_channel("element", pattern=r"(f3_.*slit|f4_.*slit|gravity_mag|tail_prism_magenta)", id="mag_slit_mask")
        cyan_pulsed = g.pulse(cyan_slit_m, min_val=0.80, max_val=1.0, freq=1.0, id="cyan_pulsed")
        mag_pulsed = g.pulse(mag_slit_m, min_val=0.80, max_val=1.0, freq=1.0, phase_offset=0.5, id="mag_pulsed")

        cyan_col = g.multi_stop_gradient(quill_wave, stops=[(0.0, "#0284C7"), (0.45, "#22D3EE"), (0.82, "#A5F3FC"), (1.0, "#FFFFFF")], smooth_blend=1.0, id="cyan_neon_col")
        mag_col = g.multi_stop_gradient(quill_wave, stops=[(0.0, "#7E22CE"), (0.45, "#D946EF"), (0.82, "#F5D0FE"), (1.0, "#FFFFFF")], smooth_blend=1.0, id="mag_neon_col")

        slit_em_mask = g.math(cyan_pulsed, mag_pulsed, op="max", id="slit_em_mask")
        neon_and_quill_mask = g.math(slit_em_mask, quill_glow_all, op="max", id="neon_quill_mask")
        all_em_mask = g.math(neon_and_quill_mask, eye_em_masked, op="max", id="all_emissive_mask")

        em_col_1 = g.blend(cyan_col, mag_col, mask=g.math(mag_slit_m, wing_mag_m, op="max", id="mag_color_region"), id="twin_spec_col")
        all_em_col = g.blend(em_col_1, f"{eye_multi}.albedo", mask=eye_outward_mask, id="all_emissive_col")

        em_comp = g.emissive_composite(
            f"{with_eyes_mat}.albedo",
            all_em_mask,
            color=all_em_col,
            glow_bleed_px=2,
            glow_strength=0.45,
            id="aethelion_glow",
        )

        lit_mat = g.material_make(
            f"{em_comp}.albedo",
            height=f"{with_eyes_mat}.height",
            normal=f"{with_eyes_mat}.normal",
            roughness=f"{with_eyes_mat}.roughness",
            metallic=f"{with_eyes_mat}.metallic",
            ao=f"{with_eyes_mat}.ao",
            emissive=f"{em_comp}.emissive",
            sss=f"{with_eyes_mat}.sss",
            porosity=f"{with_eyes_mat}.porosity",
            id="aethelion_lit_mat",
        )

        # Apply crisp 100% attached top/bottom hard-alpha wing membrane cutouts & clear edge faces!
        cut_mat_1 = g.material_set_alpha(lit_mat, wing_alpha_all, mask=wing_updown_m, id="aethelion_cut_pinions")
        cut_mat_2 = g.material_set_alpha(cut_mat_1, g.constant(0.0, id="zero_a"), mask=wing_side_m, id="aethelion_clean_lobe_edges")
        final_mat = g.material_pad_islands(cut_mat_2, id="final_aethelion_material")
        p5.preview_ref = f"{final_mat}.albedo"

    g.bind_driver(em_comp, "glow_strength", kind="wiggler", min_val=0.38, max_val=0.56, freq=1.0, octaves=3, seed=seed + 7)
    g.set_output_material(final_mat, normal_strength=2.5)
    g.expose_param("eye_brow_shadow", eye_multi, "brow_shadow", min_val=0.0, max_val=0.8, step=0.05)
    g.expose_param("glow_strength", em_comp, "glow_strength", min_val=0.0, max_val=1.0, step=0.05)
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


def build_colossal_dragon_texgraph(mesh_ctx: MeshBakeContext, *, seed: int = 5309) -> TexGraph:
    """5-Pass Industry-Standard Material-Bus TexGraph for Monumental Normal-Wing Dragons (Distilled from Reference Images 2, 3 & 4).

    Combines:
    - **Pass 1 (`pass_1_bake`)**: 3D geometry bake + 100% aligned top/bottom wing coords (`wing_span_u`, `wing_trail_v`, `face_is_down`).
    - **Pass 2 (`pass_2_muscle_scales`)**: 3D-Anatomical Scale-to-Muscle Fitting (`anatomical_muscle_scales_pbr`) wrapping chunky
      slate-obsidian & storm-teal armor plates along limbs, neck rings, ribcage & thighs with recessed muscle crevices & pale belly scutes!
    - **Pass 3 (`pass_3_colossal_wings`)**: Colossal Full-Chord Bat-Dragon Membrane (`colossal_dragon_membrane_pbr`) with velvety
      bone-hugging shadow, arched cross-bay tendon ribs, diagonal leathery tension pleats, backlit volcanic-amber SSS (`0.88`), & ragged battle tatters!
    - **Pass 4 (`pass_4_grade`)**: Cool-shadow contact AO + cohesive atmospheric color grading + clean despeckle!
    - **Pass 5 (`pass_5_ember_eyes_pbr`)**: Fluid-simulated (`fluid_smoke_sim`) + sparked (`particle_sim`) + wiggler-driven molten
      chest core (`<= 3%` focal area like Reference Image 4!), bilateral radiant volcanic-amber gradient eyes, 100% aligned top/bottom
      wing cutouts, and full UV island bleed padding -> `set_output_material(...)`!
    """
    g = TexGraph(
        mesh_ctx.width,
        mesh_ctx.height,
        tileable=False,
        seed=seed,
        name="colossal_wing_dragon_v5_pbr",
        mesh_context=mesh_ctx,
    )

    with g.pass_("pass_1_bake", title="3D Bake & Masks", description="3D muscle cavities, aligned wing coords & anatomical masks.", color="#1E293B") as p1:
        baked_ao = g.mesh_channel("baked_ao", id="mesh_baked_ao")
        flow_lit = g.body_flow_gradient("top_light_wrap", gamma=0.95, id="body_light_wrap")
        lu = g.mesh_channel("local_u", id="mesh_lu")
        lv = g.mesh_channel("local_v", id="mesh_lv")
        side_s = g.mesh_channel("side_sign", id="mesh_side_sign")
        span_u = g.mesh_channel("wing_bay_u", id="mesh_wing_bay_u")
        trail_v = g.mesh_channel("wing_bay_v", id="mesh_wing_bay_v")
        bay_sector_m = g.mesh_channel("wing_bay_mask", id="mesh_wing_bay_mask")
        face_dn = g.mesh_channel("face_is_down", id="mesh_face_is_down")
        up_down_f = g.mesh_channel("face", faces="up,down", id="wing_updown_face")
        not_updown = g.levels(up_down_f, invert=True, id="not_updown")

        wing_mem_m = g.mesh_channel("element", pattern=r"(wing_mem|wing_lobe|wing_bay|membrane|patagium|wing_sail|wing_inner_sail|inner_sail)", id="wing_mem_mask")
        horn_talon_m = g.mesh_channel("element", pattern=r"(horn|claw|talon|spike|spur|spine|tip_a|tip_b)", id="horn_talon_mask")
        chest_core_m = g.mesh_channel("element", pattern=r"(ember_core|chest_core_gem|gemini_star|sternum_core|slit|gravity_)", id="ember_core_mask")
        scute_m = g.mesh_channel("element", pattern=r"(ventral|sternum|throat|groin_bridge|belly|scute)", id="ventral_scute_mask")
        oral_m = g.mesh_channel("element", pattern=r"(palate|pharynx|oral_floor|cheek_web|tongue)", id="oral_mask")
        fang_m = g.mesh_channel("element", pattern=r"(teeth|fang)", id="fang_mask")
        eye_all_m = g.mesh_channel("element", pattern=r"(^|/|__)eye_[lr]", id="eye_all_mask")
        eye_out_m = g.mesh_channel("outward_eye", pattern=r"(^|/|__)eye_[lr]", id="eye_outward_mask")
        mem_updown_m = g.math(wing_mem_m, up_down_f, op="mul", id="mem_updown_mask")
        mem_side_m = g.math(wing_mem_m, not_updown, op="mul", id="mem_side_mask")
        p1.preview_ref = baked_ao

    with g.pass_("pass_2_muscle_scales", title="Muscle-Fit Scales", description="3D-fitted obsidian-slate & storm-teal osteoderm scales hugging muscle volumes.", color="#172554") as p2:
        # Organic 3D-Painterly Dragon Hide & Muscle-Fitted Scale Clusters (Distilled from Reference Images 1-4!)
        body_mat = g.use_subgraph(
            "anatomical_muscle_scales_pbr",
            prefix="dragon_muscle_scales",
            seed=seed + 11,
            plate_cols=max(56, mesh_ctx.width // 8),
            plate_rows=max(56, mesh_ctx.height // 7),
            scale_stops=[
                (0.00, "#0C0D1D"),  # velvety obsidian-plum crevice shadow
                (0.22, "#1A2542"),  # rich dark volcanic basalt hide
                (0.48, "#2C4B6E"),  # lush ocean storm-teal scale body
                (0.74, "#4A7A9C"),  # sunlit weathered steel-teal dome
                (0.91, "#7BB4CF"),  # crisp honed dorsal scale catchlight
                (1.00, "#CBEAF8"),  # delicate rim highlight
            ],
            belly_stops=[
                (0.00, "#383032"),
                (0.42, "#7C6E63"),
                (0.78, "#BEB099"),
                (1.00, "#ECE2CD"),
            ],
            muscle_groove_depth=0.42,
            scale_relief=0.56,
            organic_distortion=0.66,
            softness=0.50,
            hue_variation=0.40,
            roughness_base=0.40,
            metallic_crown=0.16,
            smooth_blend=0.52,
            quantize_steps=7,
        )
        # Warm dark obsidian-horn & ivory-tipped claws/spines
        horn_alb = g.multi_stop_gradient(flow_lit, stops=[(0.0, "#141118"), (0.45, "#3A3138"), (0.80, "#7A6A64"), (1.0, "#D8C8B4")], smooth_blend=0.72, quantize_steps=6, id="horn_alb")
        horn_mat = g.material_make(horn_alb, height=f"{body_mat}.height", roughness=0.28, metallic=0.20, ao=baked_ao, id="horn_mat")
        scute_mat = g.use_subgraph(
            "pearl_scutes_pbr",
            prefix="ventral_scutes",
            inputs={"ao": baked_ao},
            seed=seed + 23,
            stops=[(0.0, "#3D3537"), (0.38, "#7D7066"), (0.76, "#C4B69D"), (1.0, "#EFE6D2")],
            roughness=0.48,
            sss=0.18,
        )
        oral_mat = g.material_make(g.multi_stop_gradient(flow_lit, stops=[(0.0, "#2B0C18"), (1.0, "#7C223B")], smooth_blend=0.8, id="oral_alb"), roughness=0.38, ao=baked_ao, sss=0.30, id="oral_mat")
        fang_mat = g.material_make(g.multi_stop_gradient(flow_lit, stops=[(0.0, "#AEA390"), (1.0, "#FFFDF5")], smooth_blend=0.8, id="fang_alb"), roughness=0.22, ao=baked_ao, id="fang_mat")

        b1 = g.material_height_blend(body_mat, horn_mat, mask=horn_talon_m, height_offset=0.10, contrast=7.0, seam_ao_strength=0.24, id="comp_horns")
        b2 = g.material_blend(b1, scute_mat, mask=scute_m, id="comp_scutes")
        b3 = g.material_blend(b2, oral_mat, mask=oral_m, id="comp_oral")
        body_full = g.material_blend(b3, fang_mat, mask=fang_m, id="body_full_mat")
        p2.preview_ref = f"{body_full}.albedo"

    with g.pass_("pass_3_colossal_wings", title="Colossal Membranes", description="Special arched-tendon, diagonal-pleated, backlit translucent dragon wing membranes.", color="#451A03") as p3:
        # True Dual-Layer 0.5u Bat-Dragon Wing Membrane (Dorsal Smoked Mahogany-Leather vs. Ventral Glowing Amber SSS!)
        wing_mat = g.use_subgraph(
            "colossal_dragon_membrane_pbr",
            prefix="colossal_mem",
            inputs={"span_u": span_u, "trail_v": trail_v, "face_is_down": face_dn},
            seed=seed + 71,
            dorsal_palette_stops=[
                (0.00, "#16101B"),  # deep smoked obsidian-plum spar root
                (0.24, "#311E28"),  # dark gothic mahogany-umber leather
                (0.50, "#593232"),  # weathered bronze-crimson dorsal mantle
                (0.76, "#8F5340"),  # sunlit copper-sienna membrane sheen
                (0.92, "#C4885E"),  # warm golden-ochre outer feathering
                (1.00, "#EBD0A4"),  # subtle rim light
            ],
            palette_stops=[
                (0.00, "#1C121F"),  # velvety obsidian-plum bone-contact shadow
                (0.24, "#42242E"),  # rich dark burgundy-umber
                (0.50, "#7C3F38"),  # warm leathery amber-crimson membrane
                (0.76, "#C46C46"),  # glowing backlit volcanic-amber cushion
                (0.92, "#F0B074"),  # sunlit thin-skin subsurface warmth
                (1.00, "#FFF0C8"),  # translucent core radiance
            ],
            rim_color="#1E1420",
            tendon_color="#251624",
            dorsal_tendon_color="#1B131E",
            arch_depth=0.25,
            arches=1,
            tatter_strength=0.095,
            tendon_ribs=3,
            pleat_count=9.0,
            pleat_strength=0.16,
            organic_distortion=0.60,
            backlit_sss=0.90,
            smooth_blend=0.78,
            quantize_steps=8,
        )
        with_wings = g.material_blend(body_full, wing_mat, mask=wing_mem_m, id="dragon_with_wings_mat")
        p3.preview_ref = f"{with_wings}.albedo"

    with g.pass_("pass_4_grade", title="Lighting & Grade", description="Cool shadow depth, atmospheric balance & clean despeckle.", color="#0F172A") as p4:
        graded = g.material_grade(
            with_wings,
            ao_bake=baked_ao,
            ao_darken=0.16,
            shadows=(-0.008, -0.004, 0.024),
            midtones=(0.006, 0.004, 0.006),
            highlights=(0.026, 0.018, 0.008),
            vibrance=0.18,
            contrast=1.03,
            lift=0.014,
            shadow_cool=0.05,
            atmosphere_strength=0.20,
            sky_tint=(0.24, 0.38, 0.56),
            bounce_tint=(0.36, 0.16, 0.12),
            organic_wash_strength=0.12,
            despeckle_threshold=0.38,
            id="dragon_graded_mat",
        )
        p4.preview_ref = f"{graded}.albedo"

    with g.pass_("pass_5_ember_eyes_pbr", title="Ember Core & PBR Final", description="Fluid-sim molten chest core, volcanic gradient eyes, aligned wing cutouts & island pad.", color="#7C2D12") as p5:
        eye_sock = g.material_make(g.constant("#120A14", id="eye_sock_dark"), roughness=0.45, ao=baked_ao, id="eye_sock_mat")
        with_sock = g.material_blend(graded, eye_sock, mask=eye_all_m, id="comp_eye_sock")
        eye_m = g.gradient_eye_multi(
            lu, lv, side_sign=side_s,
            stops=[(0.0, "#7F1D1D"), (0.28, "#EA580C"), (0.62, "#FACC15"), (0.88, "#FEF9C3"), (1.0, "#FFFFFF")],
            brow_shadow=0.38, angle_deg=-20.0, style="ember_slit", rim_color="#120912",
            id="volcanic_eye",
        )
        eye_em_m = g.math(eye_out_m, f"{eye_m}.emissive_mask", op="mul", id="eye_em_masked")
        eye_pbr = g.material_make(f"{eye_m}.albedo", height=g.constant(0.65, id="eye_hc"), roughness=0.08, ao=g.constant(1.0, id="eye_aoc"), id="eye_pbr_mat")
        with_eyes = g.material_blend(with_sock, eye_pbr, mask=eye_out_m, id="comp_eyes")

        # Animated Fluid-Sim Molten Chest Core (inspired directly by Reference Image 4!)
        ember_fluid = g.fluid_smoke_sim(
            steps=10, buoyancy=1.4, curl_strength=1.6, dissipation=0.92, seed=seed + 91,
            color_stops=[(0.0, "#991B1B"), (0.30, "#F97316"), (0.68, "#FDE047"), (1.0, "#FFFFFF")],
            id="chest_ember_fluid",
        )
        ember_core_pbr = g.material_make(
            f"{ember_fluid}.rgba",
            height=g.constant(0.85, id="ember_hc"),
            roughness=0.12,
            ao=g.constant(1.0, id="ember_aoc"),
            id="ember_core_pbr_mat",
        )
        with_ember_core = g.material_blend(with_eyes, ember_core_pbr, mask=chest_core_m, id="comp_ember_core")
        core_pulsed = g.pulse(chest_core_m, min_val=0.88, max_val=1.0, freq=1.0, id="core_pulsed")
        all_em_m = g.math(core_pulsed, eye_em_m, op="max", id="all_em_mask")
        all_em_c = g.blend(f"{ember_fluid}.rgba", f"{eye_m}.albedo", mask=eye_out_m, id="all_em_col")
        em_comp = g.emissive_composite(f"{with_ember_core}.albedo", all_em_m, color=all_em_c, glow_bleed_px=3, glow_strength=0.78, id="dragon_ember_glow")

        lit_mat = g.material_make(
            f"{em_comp}.albedo",
            height=f"{with_ember_core}.height", normal=f"{with_ember_core}.normal",
            roughness=f"{with_ember_core}.roughness", metallic=f"{with_ember_core}.metallic",
            ao=f"{with_ember_core}.ao", emissive=f"{em_comp}.emissive",
            sss=f"{with_ember_core}.sss", porosity=f"{with_ember_core}.porosity",
            id="dragon_lit_mat",
        )
        bay_alpha = g.math(f"{wing_mat}.alpha", bay_sector_m, op="mul", id="dragon_bay_sector_alpha")
        cut_1 = g.material_set_alpha(lit_mat, bay_alpha, mask=mem_updown_m, id="dragon_cut_wings")
        cut_2 = g.material_set_alpha(cut_1, g.constant(0.0, id="zero_a"), mask=mem_side_m, id="dragon_clean_wing_edges")
        final_mat = g.material_pad_islands(cut_2, id="final_colossal_dragon_mat")
        p5.preview_ref = f"{final_mat}.albedo"

    g.bind_driver(em_comp, "glow_strength", kind="wiggler", min_val=0.48, max_val=0.72, freq=1.0, octaves=3, seed=seed + 5)
    g.set_output_material(final_mat, normal_strength=2.6)
    return g


from .presets_library import V7_PRESET_REGISTRY

PRESET_REGISTRY: dict[str, Any] = {
    "neon_hex_conduit": build_neon_hex_block,
    "neon_hex_block": build_neon_hex_block,
    "temple_carved_stone": build_temple_carved_stone,
    "mossy_carved_brick": build_mossy_carved_brick,
    "molten_dragon_scales": build_molten_scales_block,
    "stylized_wood_and_foliage": build_stylized_wood_and_foliage,
    "dragon_wing_membrane": build_dragon_wing_membrane_tile,
    "arcane_ember_vfx": build_arcane_ember_vfx_block,
    "aethelion_prism_blade": build_prism_blade_item,
    "prism_blade_item": build_prism_blade_item,
    **V7_PRESET_REGISTRY,
}
