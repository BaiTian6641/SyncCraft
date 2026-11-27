"""Full-Production Multi-Domain TexGraph v7 Studio Library (Blocks, Items, Real-Life Animals & Imaginary Creatures).

Every preset is a validated, multi-pass / MaterialBus-connected procedural graph exporting full
Albedo + Baked Studio Albedo + LabPBR 1.3 (_n/_s) + Bedrock RTX (_mer) and passing the
domain-specific automated ``evaluate_texture_quality(...)`` QA gate!
"""
from __future__ import annotations

from typing import Any

from .core import TexGraph
from .mesh_baker import MeshBakeContext
from .production import PALETTE_BANK


# ---------------------------------------------------------------------------
# 1. Seamless World Block Presets (16x16 Vanilla & 32x32+ HD, 100% Tileable!)
# ---------------------------------------------------------------------------


def build_cobblestone_block(size: int = 32, *, seed: int = 101, moss_amount: float = 0.0) -> TexGraph:
    """Seamless Minecraft Cobblestone Masonry (Flood Fill + Slope Blur + Curvature Edge Wear, ArtsByKev benchmark)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="cobblestone_masonry_block")
    cols = 3 if size <= 20 else 4
    rows = 3 if size <= 20 else 4
    with g.pass_("pass_1_masonry", title="Flood-Fill Cobbles", color="#334155") as p1:
        cob = g.cobblestone_masonry(
            cols=cols, rows=rows,
            stone_stops=list(PALETTE_BANK["cobble_warm_slate"]),
            mortar_color="#2D2A33",
            moss_amount=moss_amount,
            wear_strength=0.42,
            smooth_blend=0.38 if size <= 20 else 0.45,
            quantize_steps=5 if size <= 20 else 6,
            seed=seed,
            id="cobble_mat",
        )
        p1.preview_ref = f"{cob}.albedo"
    with g.pass_("pass_2_grade", title="Lighting & Grade", color="#1E3A8A") as p2:
        final = g.material_grade(
            cob,
            shadows=(-0.01, -0.005, 0.02),
            highlights=(0.025, 0.018, 0.005),
            vibrance=0.14,
            contrast=1.04,
            id="cobble_final",
        )
        p2.preview_ref = f"{final}.albedo"
    g.set_output_material(final, normal_strength=2.6)
    g.expose_param("moss_amount", cob, "moss_amount", min_val=0.0, max_val=0.9, step=0.05)
    g.expose_param("wear_strength", cob, "wear_strength", min_val=0.0, max_val=1.0, step=0.05)
    return g


def build_mossy_cobblestone_block(size: int = 32, *, seed: int = 102) -> TexGraph:
    """Seamless Mossy Cobblestone Masonry with lush crevice moss tucked between rounded cobbles."""
    g = build_cobblestone_block(size=size, seed=seed, moss_amount=0.48)
    g.name = "mossy_cobblestone_block"
    return g


def build_ancient_oak_bark_block(size: int = 32, *, seed: int = 109) -> TexGraph:
    """Seamless Furrowed Ancient Oak Bark with Slope-Blur ridges and crevice lichen/moss."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="ancient_oak_bark_block")
    with g.pass_("pass_1_bark", title="Furrowed Bark", color="#451A03") as p1:
        bark = g.bark_and_timber(
            bark_stops=list(PALETTE_BANK["oak_bark"]),
            furrow_count=4.5 if size <= 20 else 6.0,
            moss_amount=0.22,
            seed=seed,
            id="bark_mat",
        )
        p1.preview_ref = f"{bark}.albedo"
    with g.pass_("pass_2_grade", title="Atmospheric Grade", color="#14532D") as p2:
        final = g.material_grade(bark, shadows=(-0.01, -0.005, 0.015), highlights=(0.02, 0.014, 0.004), vibrance=0.18, contrast=1.05, id="bark_final")
        p2.preview_ref = f"{final}.albedo"
    g.set_output_material(final, normal_strength=2.8)
    return g


def build_deepslate_prism_ore_block(size: int = 32, *, seed: int = 211) -> TexGraph:
    """Seamless Deepslate Embedded with Sparkling Faceted Starlight-Prism Ore Crystals (Animated Emissive Pulse!)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="deepslate_prism_ore_block")
    with g.pass_("pass_1_ore", title="Host Slate & Crystal Ore", color="#083344") as p1:
        ore = g.crystalline_ore_vein(
            host_stops=list(PALETTE_BANK["deepslate_obsidian"]),
            ore_stops=list(PALETTE_BANK["starlight_sapphire"]),
            ore_density=0.62,
            emissive_strength=0.72,
            seed=seed,
            id="ore_mat",
        )
        p1.preview_ref = f"{ore}.albedo"
    with g.pass_("pass_2_grade", title="Grade & Pulse", color="#172554") as p2:
        final = g.material_grade(ore, shadows=(-0.01, 0.0, 0.025), highlights=(0.01, 0.02, 0.025), vibrance=0.20, contrast=1.05, id="ore_final")
        p2.preview_ref = f"{final}.albedo"
    g.bind_driver(ore, "emissive_strength", kind="sine", min_val=0.45, max_val=0.90, freq=1.0)
    g.set_output_material(final, normal_strength=2.9)
    return g


def build_caustic_water_block(size: int = 32, *, seed: int = 313) -> TexGraph:
    """Seamless Looping Animated Caustic Water Block (16-frame ready LabPBR specular surface!)."""
    g = TexGraph(size, size, tileable=True, seed=seed, name="caustic_water_block")
    with g.pass_("pass_1_water", title="Dual Voronoi Caustics", color="#0C4A6E") as p1:
        w = g.caustic_water(scale=3.5 if size <= 20 else 4.5, seed=seed, id="water_mat")
        p1.preview_ref = f"{w}.albedo"
    g.set_output_material(w, normal_strength=1.8)
    return g


# ---------------------------------------------------------------------------
# 2. Parametric SDF + Directional Selout Minecraft Item Forge (16x16 & 32x32!)
# ---------------------------------------------------------------------------


def _build_sdf_item_core(
    *,
    name: str,
    shape: str,
    size: int,
    seed: int,
    primary_stops: list[tuple[float, str]],
    accent_stops: list[tuple[float, str]] | None = None,
    accent_region: str | None = None,  # "bottom_left" | "top_strip" | "center_core" | "bottom_fill" | "rim"
    accent_cut: float = 0.52,
    emissive_region: str | None = None,  # "rune_star" | "blade_fuller" | "liquid_core" | "gem_glint" | None
    emissive_color: str = "#67E8F9",
    metallic_base: float = 0.78,
    roughness_base: float = 0.22,
    light_angle_deg: float = -45.0,
    facet_crystal: bool = False,
    specular_glint: bool = True,
) -> TexGraph:
    """Reusable parametric SDF + Directional-Selout studio item forge (clean at BOTH 16x16 and 32x32!)."""
    g = TexGraph(size, size, tileable=False, seed=seed, name=name)
    with g.pass_("pass_1_sdf", title="SDF Silhouette & Bevel", color="#1E293B") as p1:
        sh = g.sdf_shape(shape=shape, bevel_width=0.18, outline_width=0.045, id="item_sdf")
        p1.preview_ref = f"{sh}.bevel"

    with g.pass_("pass_2_shade", title="Specular Ridge & Shading", color="#312E81") as p2:
        # Directional light from top-left (-135 deg in screen UV where +y is down -> top-left bright, bottom-right deep!)
        diag = g.gradient_field("linear", angle_deg=-135.0, ease="smoothstep", id="item_light")
        if facet_crystal:
            # Crisp angular multi-facet crystal planes for gemstones!
            con = g.gradient_field("conical", angle_deg=-30.0, ease="linear", id="gem_conical")
            fac = g.posterize_soft(con, steps=6, blend=0.85, id="gem_facets")
            h_mix = g.blend(g.blend(f"{sh}.bevel", fac, mode="overlay", opacity=0.65, id="gem_f_mix"), diag, mode="soft_light", opacity=0.55, id="item_h_raw")
        else:
            # Sculpted 3D-feel item bevel combining SDF ridge dome + top-left directional specular glide!
            h_mix = g.blend(f"{sh}.bevel", diag, mode="overlay", opacity=0.58, id="item_h_raw")
        h_lit = g.levels(h_mix, in_min=0.04, in_max=0.90, gamma=0.84, id="item_h_lit")
        base_rgba = g.multi_stop_gradient(
            h_lit,
            stops=primary_stops,
            smooth_blend=0.45 if size <= 20 else 0.56,
            quantize_steps=5 if size <= 20 else 6,
            alpha_mask=f"{sh}.mask",
            id="item_primary_rgba",
        )
        cur_rgba = base_rgba
        acc_mask = None
        if accent_stops and accent_region:
            if accent_region == "bottom_left":
                # Hilt/guard/haft on bottom-left of diagonal swords & pickaxes (angle +135 deg: bottom-left = 1.0, top-right = 0.0!)
                bl = g.gradient_field("linear", angle_deg=135.0, ease="linear", id="acc_diag")
                acc_mask = g.histogram_scan(bl, position=float(accent_cut), contrast=0.92, id="acc_mask")
            elif accent_region == "top_strip":
                # Leaf / cork / crown region strictly at top of item
                vy = g.gradient_field("linear", angle_deg=-90.0, ease="linear", id="acc_top_y")
                acc_mask = g.histogram_scan(vy, position=float(accent_cut), contrast=0.92, id="acc_mask")
            elif accent_region == "bottom_fill":
                # Deep glowing potion liquid inside lower bulb + warm cork at very top!
                vy_bot = g.gradient_field("linear", angle_deg=90.0, ease="linear", id="acc_bot_y")
                in_b = g.histogram_scan(f"{sh}.bevel", position=0.72, contrast=0.88, id="acc_inner")
                acc_mask = g.math(g.histogram_scan(vy_bot, position=0.56, contrast=0.90, id="acc_lo"), in_b, op="mul", id="acc_mask")
            elif accent_region == "rim":
                # Crisp metal trim hugging shield / book border (only outer bevel band!)
                acc_mask = g.histogram_scan(f"{sh}.bevel", position=0.66, contrast=0.90, invert=True, id="acc_mask")
            elif accent_region == "center_core":
                core_s = g.sdf_shape(shape="star", center=(0.5, 0.5), size=(0.14, 0.14), sides=4, inner_ratio=0.42, id="acc_star")
                acc_mask = f"{core_s}.mask"

            if acc_mask is not None:
                acc_rgba = g.multi_stop_gradient(
                    h_lit,
                    stops=accent_stops,
                    smooth_blend=0.45 if size <= 20 else 0.56,
                    quantize_steps=5,
                    alpha_mask=f"{sh}.mask",
                    id="item_accent_rgba",
                )
                cur_rgba = g.blend(base_rgba, acc_rgba, mode="normal", opacity=1.0, mask=acc_mask, id="item_comp_rgba")

        # Optional curved top-left specular glass/metal catchlight glint (iconic Minecraft item polish!)
        if specular_glint:
            gl_s = g.sdf_shape(shape="circle", center=(0.40, 0.36), size=(0.08, 0.08), bevel_width=0.08, id="spec_glint_sdf")
            gl_m = g.math(g.math(f"{gl_s}.bevel", f"{sh}.mask", op="mul", id="spec_gl_in"), g.constant(0.68, id="spec_gl_w"), op="mul", id="spec_gl_m")
            cur_rgba = g.blend(cur_rgba, g.constant("#FFFFFF", id="spec_white"), mode="screen", opacity=0.72, mask=gl_m, id="item_spec_comp")
        p2.preview_ref = cur_rgba

    with g.pass_("pass_3_selout_pbr", title="Directional Selout & PBR", color="#083344") as p3:
        # True Brandon James Greer directional selective outline (warm lit top-left -> cool deep shadow bottom-right!)
        sel = g.directional_selout(cur_rgba, light_angle_deg=-135.0, lit_darken=0.70, shadow_darken=0.30, id="item_selout")
        vib = g.vibrance(sel, vibrance=0.18, contrast=1.06, id="item_vib")

        em_node = None
        final_alb = vib
        if emissive_region:
            if emissive_region == "rune_star":
                st = g.sdf_shape(shape="star", center=(0.5, 0.48), size=(0.12, 0.12), sides=4, inner_ratio=0.38, id="em_star")
                em_m = f"{st}.mask"
            elif emissive_region == "blade_fuller":
                fl = g.sdf_shape(shape="capsule", center=(0.59, 0.41), size=(0.045, 0.26), angle_deg=-45.0, id="em_fuller")
                em_m = f"{fl}.mask"
            elif emissive_region == "gem_glint":
                st = g.sdf_shape(shape="star", center=(0.46, 0.44), size=(0.10, 0.10), sides=4, inner_ratio=0.36, id="em_gem_s")
                em_m = f"{st}.mask"
            elif emissive_region == "liquid_core" and acc_mask is not None:
                c_in = g.sdf_shape(shape="circle", center=(0.48, 0.62), size=(0.12, 0.10), bevel_width=0.10, id="em_liq_c")
                em_m = g.math(f"{c_in}.bevel", acc_mask, op="mul", id="em_liq_m")
            else:
                em_m = g.constant(0.0, id="em_zero")

            em_masked = g.math(g.pulse(em_m, min_val=0.74, max_val=1.0, freq=1.0, id="item_em_pulse"), f"{sh}.mask", op="mul", id="item_em_mask")
            g_str = 0.26 if emissive_region == "rune_star" else 0.42
            em_comp = g.emissive_composite(vib, em_masked, color=g.constant(emissive_color, id="item_em_col"), glow_bleed_px=1, glow_strength=g_str, id="item_em_comp")
            final_alb = g.set_alpha(f"{em_comp}.albedo", f"{sh}.mask", id="item_hard_alpha")
            em_node = f"{em_comp}.emissive"

        item_h = g.math(f"{sh}.bevel", f"{sh}.mask", op="mul", id="item_pbr_h")
        item_nrm = g.height_to_normal(item_h, strength=3.0, id="item_pbr_n")
        item_met = g.math(f"{sh}.mask", g.constant(metallic_base, id="item_met_c"), op="mul", id="item_pbr_m")
        item_rgh = g.constant(roughness_base, id="item_pbr_r")
        mat = g.material_make(
            final_alb,
            height=item_h,
            normal=item_nrm,
            roughness=item_rgh,
            metallic=item_met,
            emissive=em_node,
            id="item_final_mat",
        )
        p3.preview_ref = f"{mat}.albedo"

    g.set_output_material(mat, normal_strength=3.0)
    return g


def build_runic_greatsword_item(size: int = 32, *, seed: int = 401) -> TexGraph:
    """Forged Steel & Champagne-Gold Runic Greatsword with focused starlight fuller & directional selout."""
    return _build_sdf_item_core(
        name="runic_greatsword_item", shape="sword", size=size, seed=seed,
        primary_stops=list(PALETTE_BANK["forged_steel"]),
        accent_stops=list(PALETTE_BANK["champagne_gold"]),
        accent_region="bottom_left",
        accent_cut=0.44,
        emissive_region="blade_fuller",
        emissive_color="#67E8F9",
        metallic_base=0.90, roughness_base=0.18,
    )


def build_ruby_potion_item(size: int = 32, *, seed: int = 402) -> TexGraph:
    """Alchemist Ruby Health Potion Flask with luminous liquid belly, crisp glass sheen & directional selout."""
    return _build_sdf_item_core(
        name="ruby_potion_item", shape="potion", size=size, seed=seed,
        primary_stops=[(0.0, "#33415C"), (0.38, "#7D9AB8"), (0.74, "#C5DDF0"), (1.0, "#FFFFFF")],
        accent_stops=list(PALETTE_BANK["ruby_crimson"]),
        accent_region="bottom_fill",
        emissive_region="liquid_core",
        emissive_color="#FB7185",
        metallic_base=0.05, roughness_base=0.10,
    )


def build_dragonscale_shield_item(size: int = 32, *, seed: int = 403) -> TexGraph:
    """Emerald & Champagne-Gold Kite Shield with brushed gold rim, star boss & crisp directional selout."""
    return _build_sdf_item_core(
        name="dragonscale_shield_item", shape="shield", size=size, seed=seed,
        primary_stops=list(PALETTE_BANK["emerald_jade"]),
        accent_stops=list(PALETTE_BANK["champagne_gold"]),
        accent_region="rim",
        emissive_region="rune_star",
        emissive_color="#FEF08A",
        metallic_base=0.72, roughness_base=0.24,
    )


def build_starlight_gem_item(size: int = 32, *, seed: int = 404) -> TexGraph:
    """Faceted Starlight-Sapphire Octagonal Gemstone with polygonal facets & focused starlight glint."""
    return _build_sdf_item_core(
        name="starlight_gem_item", shape="gem", size=size, seed=seed,
        primary_stops=list(PALETTE_BANK["starlight_sapphire"]),
        accent_stops=list(PALETTE_BANK["amethyst_void"]),
        accent_region="rim",
        emissive_region="gem_glint",
        emissive_color="#E0F7FF",
        metallic_base=0.25, roughness_base=0.08,
        facet_crystal=True,
    )


def build_dwarven_pickaxe_item(size: int = 32, *, seed: int = 405) -> TexGraph:
    """Forged Dwarven Pickaxe with bright forged steel crescent head, warm oak haft & directional selout."""
    return _build_sdf_item_core(
        name="dwarven_pickaxe_item", shape="pickaxe", size=size, seed=seed,
        primary_stops=[(0.0, "#242942"), (0.30, "#4F5E85"), (0.62, "#8FA6CE"), (0.86, "#D2E3F7"), (1.0, "#FFFFFF")],
        accent_stops=list(PALETTE_BANK["oak_bark"]),
        accent_region="bottom_left",
        accent_cut=0.48,
        metallic_base=0.84, roughness_base=0.22,
    )


def build_arcane_spellbook_item(size: int = 32, *, seed: int = 406) -> TexGraph:
    """Crimson Leather Arcane Grimoire with brushed gold border/clasp & focused glowing amethyst star rune."""
    return _build_sdf_item_core(
        name="arcane_spellbook_item", shape="book", size=size, seed=seed,
        primary_stops=list(PALETTE_BANK["ruby_crimson"]),
        accent_stops=list(PALETTE_BANK["champagne_gold"]),
        accent_region="rim",
        emissive_region="rune_star",
        emissive_color="#F0ABFC",
        metallic_base=0.32, roughness_base=0.38,
    )


def build_golden_apple_item(size: int = 32, *, seed: int = 407) -> TexGraph:
    """Luminous Golden Hearth Apple with warm spherical luster, emerald leaf & directional selout."""
    return _build_sdf_item_core(
        name="golden_apple_item", shape="apple", size=size, seed=seed,
        primary_stops=list(PALETTE_BANK["champagne_gold"]),
        accent_stops=list(PALETTE_BANK["lush_canopy"]),
        accent_region="top_strip",
        accent_cut=0.28,
        emissive_region=None,
        metallic_base=0.86, roughness_base=0.16,
    )


# ---------------------------------------------------------------------------
# 3. Real-Life Animal & Imaginary Creature Entity Material Presets (3D & 2D!)
# ---------------------------------------------------------------------------


def _finalize_entity_preset(g: TexGraph, mat_id: str, *, vibrance: float = 0.16, atmosphere: float = 0.18, sky_tint: tuple[float, float, float] = (0.24, 0.36, 0.56), bounce_tint: tuple[float, float, float] = (0.28, 0.20, 0.16)) -> TexGraph:
    with g.pass_("pass_2_grade", title="Atmospheric Unity & Pad", color="#0F172A") as p2:
        ao_n = g.mesh_channel("baked_ao", id="ent_ao") if g.mesh_context else None
        gr = g.material_grade(
            mat_id,
            ao_bake=ao_n,
            ao_darken=0.14 if ao_n else 0.0,
            vibrance=vibrance,
            contrast=1.04,
            atmosphere_strength=atmosphere,
            sky_tint=sky_tint,
            bounce_tint=bounce_tint,
            organic_wash_strength=0.08,
            id="ent_graded",
        )
        final = g.material_pad_islands(gr, id="ent_final") if g.mesh_context else gr
        p2.preview_ref = f"{final}.albedo"
    g.set_output_material(final, normal_strength=2.2)
    return g


def build_husky_wolf_pelt(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 501) -> TexGraph:
    """Real-Life Animal: Arctic Husky / Timber Wolf Fur Pelt (Everbloom Husky benchmark — espresso saddle + jagged cream belly!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="husky_wolf_pelt", mesh_context=mesh_ctx)
    with g.pass_("pass_1_fur", title="Layered Tufted Fur", color="#451A03") as p1:
        fur = g.organic_fur_pelt(
            coat_stops=list(PALETTE_BANK["husky_espresso"]),
            belly_stops=[(0.0, "#756761"), (0.38, "#AFA096"), (0.75, "#DBD0C7"), (1.0, "#F7F2EC")],
            marking_stops=[(0.0, "#191214"), (0.50, "#352623"), (1.0, "#5C453E")],
            pattern="saddle",
            tuft_scale=10.0,
            countershade_height=0.46,
            seed=seed,
            id="husky_fur",
        )
        p1.preview_ref = f"{fur}.albedo"
    return _finalize_entity_preset(g, fur, vibrance=0.14)


def build_red_fox_pelt(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 502) -> TexGraph:
    """Real-Life Animal: Woodland Red Fox Fur Pelt (vibrant russet-orange tufts + warm ivory belly/chest!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="red_fox_pelt", mesh_context=mesh_ctx)
    with g.pass_("pass_1_fur", title="Russet Fox Fur", color="#7C2D12") as p1:
        fur = g.organic_fur_pelt(
            coat_stops=list(PALETTE_BANK["red_fox_russet"]),
            belly_stops=[(0.0, "#8B756A"), (0.42, "#CBB8AC"), (0.80, "#EFE5DC"), (1.0, "#FFFBF5")],
            marking_stops=[(0.0, "#241618"), (0.5, "#5A281A"), (1.0, "#9E4624")],
            pattern="saddle",
            tuft_scale=11.0,
            countershade_height=0.40,
            seed=seed,
            id="fox_fur",
        )
        p1.preview_ref = f"{fur}.albedo"
    return _finalize_entity_preset(g, fur, vibrance=0.18, bounce_tint=(0.35, 0.18, 0.10))


def build_snow_leopard_pelt(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 503) -> TexGraph:
    """Real-Life Animal: Snow Leopard Spotted Fur Pelt (pale alpine cream coat + charcoal-slate rosettes!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="snow_leopard_pelt", mesh_context=mesh_ctx)
    with g.pass_("pass_1_fur", title="Alpine Spotted Fur", color="#334155") as p1:
        fur = g.organic_fur_pelt(
            coat_stops=[(0.0, "#464957"), (0.30, "#7E808A"), (0.62, "#B8B5B2"), (0.86, "#DDD8D0"), (1.0, "#F7F5F0")],
            belly_stops=[(0.0, "#7C7E87"), (0.45, "#C7C5C2"), (0.82, "#ECEAE6"), (1.0, "#FFFFFF")],
            marking_stops=[(0.0, "#171720"), (0.50, "#353642"), (1.0, "#5E606E")],
            pattern="spots",
            tuft_scale=11.0,
            countershade_height=0.34,
            seed=seed,
            id="leopard_fur",
        )
        p1.preview_ref = f"{fur}.albedo"
    return _finalize_entity_preset(g, fur, vibrance=0.12)


def build_barn_owl_plumage(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 504) -> TexGraph:
    """Real-Life Animal: Barn Owl Layered Barred Plumage (scalloped overlapping feathers + central quills!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="barn_owl_plumage", mesh_context=mesh_ctx)
    with g.pass_("pass_1_plumage", title="Barred Owl Feathers", color="#44403C") as p1:
        pl = g.layered_plumage(
            feather_stops=list(PALETTE_BANK["barn_owl_plumage"]),
            accent_stops=[(0.0, "#241D26"), (0.50, "#5C453E"), (1.0, "#A87C54")],
            cols=6 if w <= 80 else 10,
            rows=7 if h <= 80 else 12,
            marking="barred",
            iridescence=0.08,
            seed=seed,
            id="owl_plumage",
        )
        p1.preview_ref = f"{pl}.albedo"
    return _finalize_entity_preset(g, pl, vibrance=0.14)


def build_emperor_scorpion_chitin(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 505) -> TexGraph:
    """Real-Life Arthropod / Monster: Emperor Scorpion Segmented Chitin (Mickey Joe Scorpion benchmark!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="emperor_scorpion_chitin", mesh_context=mesh_ctx)
    with g.pass_("pass_1_chitin", title="Segmented Chitin Plates", color="#3B0764") as p1:
        ch = g.chitin_carapace(
            chitin_stops=list(PALETTE_BANK["scorpion_chitin"]),
            segments=5 if h <= 80 else 8,
            iridescence=0.34,
            glossiness=0.76,
            seed=seed,
            id="scorpion_chitin",
        )
        p1.preview_ref = f"{ch}.albedo"
    return _finalize_entity_preset(g, ch, vibrance=0.18, sky_tint=(0.24, 0.28, 0.48), bounce_tint=(0.30, 0.14, 0.24))


def build_coral_reef_fish_scales(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 506) -> TexGraph:
    """Real-Life Aquatic Animal: Iridescent Coral Reef Fish Scales (wet glossy cycloid scales + golden stripe!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="coral_reef_fish_scales", mesh_context=mesh_ctx)
    with g.pass_("pass_1_scales", title="Iridescent Cycloid Scales", color="#0E7490") as p1:
        aq = g.aquatic_scales_skin(
            base_stops=list(PALETTE_BANK["reef_turquoise"]),
            spot_stops=list(PALETTE_BANK["champagne_gold"]),
            mode="fish_scales",
            scale_cols=7 if w <= 80 else 12,
            scale_rows=8 if h <= 80 else 14,
            wetness=0.88,
            seed=seed,
            id="reef_fish",
        )
        p1.preview_ref = f"{aq}.albedo"
    return _finalize_entity_preset(g, aq, vibrance=0.22, sky_tint=(0.18, 0.48, 0.68))


def build_poison_dart_frog_skin(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 507) -> TexGraph:
    """Real-Life Amphibian: Cobalt Poison Dart Frog Moist Skin (wet specular gloss + aposematic gold/obsidian spots!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="poison_dart_frog_skin", mesh_context=mesh_ctx)
    with g.pass_("pass_1_skin", title="Moist Aposematic Skin", color="#1E3A8A") as p1:
        aq = g.aquatic_scales_skin(
            base_stops=list(PALETTE_BANK["dart_frog_cobalt"]),
            spot_stops=[(0.0, "#10101C"), (0.45, "#221E32"), (0.85, "#FACC15"), (1.0, "#FEF9C3")],
            mode="amphibian_moist",
            wetness=0.92,
            seed=seed,
            id="dart_frog",
        )
        p1.preview_ref = f"{aq}.albedo"
    return _finalize_entity_preset(g, aq, vibrance=0.22, sky_tint=(0.20, 0.42, 0.72))


def build_obsidian_magma_golem(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 508) -> TexGraph:
    """Imaginary Construct: Obsidian Magma Golem (Flood-Fill + Slope-Blur basalt slabs + pulsing molten crevices!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="obsidian_magma_golem", mesh_context=mesh_ctx)
    with g.pass_("pass_1_golem", title="Fractured Magma Slabs", color="#7C2D12") as p1:
        gol = g.elemental_golem_surface(
            rock_stops=list(PALETTE_BANK["deepslate_obsidian"]),
            fissure_stops=[(0.0, "#7F1D1D"), (0.38, "#EA580C"), (0.74, "#FACC15"), (1.0, "#FFFBEB")],
            slab_scale=4.0 if w <= 80 else 6.0,
            glow_strength=0.88,
            seed=seed,
            id="magma_golem",
        )
        p1.preview_ref = f"{gol}.albedo"
    g.bind_driver(gol, "glow_strength", kind="wiggler", min_val=0.65, max_val=1.0, freq=1.0, octaves=3, seed=seed + 5)
    return _finalize_entity_preset(g, gol, vibrance=0.20, bounce_tint=(0.45, 0.16, 0.08))


def build_celestial_phoenix_plumage(size: int = 64, *, mesh_ctx: MeshBakeContext | None = None, seed: int = 509) -> TexGraph:
    """Imaginary Creature: Celestial Phoenix Iridescent Plumage (layered flame-gold & crimson feathers!)."""
    w = mesh_ctx.width if mesh_ctx else size
    h = mesh_ctx.height if mesh_ctx else size
    g = TexGraph(w, h, tileable=False, seed=seed, name="celestial_phoenix_plumage", mesh_context=mesh_ctx)
    with g.pass_("pass_1_phoenix", title="Iridescent Flame Feathers", color="#9A3412") as p1:
        phx = g.layered_plumage(
            feather_stops=[(0.0, "#380D24"), (0.28, "#8F1D30"), (0.58, "#E85D24"), (0.84, "#FACC15"), (1.0, "#FFFBEB")],
            accent_stops=[(0.0, "#3B0764"), (0.55, "#A855F7"), (1.0, "#F5D0FE")],
            cols=6 if w <= 80 else 10,
            rows=7 if h <= 80 else 12,
            marking="iridescent",
            iridescence=0.42,
            seed=seed,
            id="phoenix_feathers",
        )
        p1.preview_ref = f"{phx}.albedo"
    return _finalize_entity_preset(g, phx, vibrance=0.22, bounce_tint=(0.42, 0.16, 0.18))


V7_PRESET_REGISTRY: dict[str, Any] = {
    # World Blocks
    "cobblestone_block": build_cobblestone_block,
    "mossy_cobblestone_block": build_mossy_cobblestone_block,
    "ancient_oak_bark_block": build_ancient_oak_bark_block,
    "deepslate_prism_ore_block": build_deepslate_prism_ore_block,
    "caustic_water_block": build_caustic_water_block,
    # Items
    "runic_greatsword_item": build_runic_greatsword_item,
    "ruby_potion_item": build_ruby_potion_item,
    "dragonscale_shield_item": build_dragonscale_shield_item,
    "starlight_gem_item": build_starlight_gem_item,
    "dwarven_pickaxe_item": build_dwarven_pickaxe_item,
    "arcane_spellbook_item": build_arcane_spellbook_item,
    "golden_apple_item": build_golden_apple_item,
    # Real-Life Animals & Imaginary Creatures
    "husky_wolf_pelt": build_husky_wolf_pelt,
    "red_fox_pelt": build_red_fox_pelt,
    "snow_leopard_pelt": build_snow_leopard_pelt,
    "barn_owl_plumage": build_barn_owl_plumage,
    "emperor_scorpion_chitin": build_emperor_scorpion_chitin,
    "coral_reef_fish_scales": build_coral_reef_fish_scales,
    "poison_dart_frog_skin": build_poison_dart_frog_skin,
    "obsidian_magma_golem": build_obsidian_magma_golem,
    "celestial_phoenix_plumage": build_celestial_phoenix_plumage,
}
