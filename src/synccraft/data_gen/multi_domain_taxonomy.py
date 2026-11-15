"""Multi-Domain Procedural 3D Asset Generators for SyncCraft v3.0 (ADR 0004).

Expands the procedural grammar from narrow weapon/furniture props to a 7-domain
32-family taxonomy (20 Seen + 12 Held-Out Unseen):
  1. Artificial (weapons, tools, furniture, lanterns, crossbows, arcane staves)
  2. Natural (ancient canopy trees, crystal geodes, coral/fungal colonies, weeping bonsai)
  3. Industrial (gearboxes, 6-axis robotic arms, turbines, telescopes, forges, gantry cranes)
  4. Building / Architecture (timber-stone halls, pagodas, arched bridges, windmills, lighthouses)
  5. Humanoid (articulated bipedal knights, artisans, astronauts, automatons)
  6. Animal (quadrupeds, avian birds, hexapod beetles, bipedal theropod dinosaurs)
  7. Dragon (4-legged winged western dragons, undulating multi-somite sea dragons)

Every generator produces typed Block-DSL v3 Program ASTs that compile through the
shared right-handed SE(3) chain and satisfy 15-axis SAT + P1-02 physical contact audits.
"""

from __future__ import annotations

import random
from typing import Any

from synccraft.dsl.ast_nodes import Attach, Bone, Box, Meta, Palette, Place, Program

# Complete 32-family -> 7-macro-domain mapping
FAMILY_DOMAINS: dict[str, str] = {
    # 1. Artificial (7 seen + 3 unseen)
    "straight_blade_sword": "artificial",
    "single_edge_blade": "artificial",
    "battle_axe": "artificial",
    "spear": "artificial",
    "war_hammer": "artificial",
    "simple_wood_bow": "artificial",
    "furniture_table_chair_chest_cabinet": "artificial",
    "mechanical_crossbow": "artificial",
    "hand_lantern": "artificial",
    "floating_ring_staff": "artificial",
    # 2. Natural (3 seen + 1 unseen)
    "ancient_canopy_tree": "natural",
    "crystalline_geode_outcrop": "natural",
    "coral_fungal_colony": "natural",
    "weeping_bonsai_arch": "natural",
    # 3. Industrial (3 seen + 3 unseen)
    "gearbox": "industrial",
    "articulated_robotic_arm": "industrial",
    "industrial_turbine_generator": "industrial",
    "tripod_telescope": "industrial",
    "anvil_forge": "industrial",
    "hydraulic_gantry_crane": "industrial",
    # 4. Building / Architecture (3 seen + 2 unseen)
    "modular_timber_stone_building": "building",
    "tiered_pagoda_watchtower": "building",
    "arched_stone_bridge_gate": "building",
    "folding_windmill": "building",
    "coastal_lighthouse_keep": "building",
    # 5. Humanoid (1 seen)
    "humanoid_biped": "humanoid",
    # 6. Animal (2 seen + 2 unseen)
    "dog_horse_quadruped": "animal",
    "bird": "animal",
    "hexapod_beetle": "animal",
    "heavy_tail_tyrannosaur": "animal",
    # 7. Dragon (1 seen + 1 unseen)
    "winged_western_dragon": "dragon",
    "segmented_sea_dragon": "dragon",
}

# Dynamic entity body plans (target_profile in bedrock_entity / geckolib / bbmodel_5)
ENTITY_BODY_PLANS: dict[str, str] = {
    "humanoid_biped": "humanoid_biped",
    "dog_horse_quadruped": "dog_horse_quadruped",
    "bird": "bird",
    "winged_western_dragon": "winged_western_dragon",
    "hexapod_beetle": "hexapod_beetle",
    "heavy_tail_tyrannosaur": "heavy_tail_tyrannosaur",
    "segmented_sea_dragon": "segmented_sea_dragon",
}

# Domain-specific perceptual OKLCH palette banks
_FOLIAGE_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("canopy_leaf", (0.54, 0.145, 142.0)),
    ("pine_needle", (0.40, 0.110, 152.0)),
    ("autumn_foliage", (0.62, 0.175, 58.0)),
    ("blossom_canopy", (0.76, 0.135, 348.0)),
    ("moss_lichen", (0.58, 0.130, 132.0)),
)

_BARK_EARTH_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("gnarled_bark", (0.36, 0.068, 52.0)),
    ("redwood_bole", (0.42, 0.098, 36.0)),
    ("silver_birch_bark", (0.78, 0.024, 78.0)),
    ("peat_root_soil", (0.28, 0.048, 48.0)),
)

_MINERAL_CORAL_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("amethyst_crystal", (0.64, 0.195, 305.0)),
    ("aquamarine_prism", (0.76, 0.155, 205.0)),
    ("citrine_quartz", (0.82, 0.175, 82.0)),
    ("reef_coral_pink", (0.70, 0.175, 24.0)),
    ("biolum_spore_teal", (0.80, 0.185, 178.0)),
)

_STONE_MASONRY_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("limestone_ashlar", (0.74, 0.032, 84.0)),
    ("granite_masonry", (0.52, 0.018, 250.0)),
    ("weathered_cobble", (0.44, 0.022, 235.0)),
    ("terracotta_roof", (0.54, 0.140, 38.0)),
    ("glazed_jade_tile", (0.48, 0.115, 168.0)),
    ("slate_shingle", (0.34, 0.026, 255.0)),
)

_INDUSTRIAL_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("safety_yellow_steel", (0.78, 0.168, 86.0)),
    ("servo_orange_alloy", (0.66, 0.182, 46.0)),
    ("cnc_gunmetal", (0.38, 0.022, 252.0)),
    ("brushed_titanium", (0.72, 0.018, 240.0)),
    ("hydraulic_cobalt", (0.50, 0.135, 248.0)),
    ("polymer_cable_dark", (0.24, 0.015, 260.0)),
)

_CREATURE_HIDES_SCALES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("timber_pelt_hide", (0.46, 0.065, 58.0)),
    ("chestnut_coat_skin", (0.48, 0.115, 42.0)),
    ("arctic_silver_fur", (0.82, 0.018, 235.0)),
    ("golden_plumage_scale", (0.70, 0.145, 76.0)),
    ("crimson_wyrm_scale", (0.48, 0.175, 26.0)),
    ("emerald_drake_scale", (0.46, 0.145, 152.0)),
    ("sapphire_abyss_scale", (0.44, 0.140, 238.0)),
    ("obsidian_carapace", (0.26, 0.032, 275.0)),
    ("iridescent_elytra", (0.52, 0.165, 188.0)),
)

_HORN_BONE_ACCENTS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("ivory_horn_keratin", (0.86, 0.048, 86.0)),
    ("charcoal_claw_horn", (0.28, 0.020, 255.0)),
    ("amber_beak_scale", (0.72, 0.155, 74.0)),
    ("pale_underbelly_skin", (0.78, 0.055, 78.0)),
)

_HUMANOID_GEAR_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("templar_steel_plate", (0.74, 0.024, 245.0)),
    ("cobalt_tunic_cloth", (0.46, 0.130, 252.0)),
    ("crimson_tabard_cloth", (0.48, 0.160, 24.0)),
    ("astronaut_exo_alloy", (0.86, 0.016, 240.0)),
    ("brass_clockwork_plate", (0.70, 0.145, 78.0)),
    ("warm_traveller_skin", (0.68, 0.085, 54.0)),
)


def _pal(
    rng: random.Random,
    spec: tuple[str, tuple[float, float, float]],
    *,
    emissive: bool = False,
    suffix: str = "",
) -> Palette:
    raw_name, (l0, c0, h0) = spec
    l_val = round(min(0.95, max(0.15, l0 + rng.uniform(-0.045, 0.045))), 4)
    c_val = round(max(0.006, c0 + rng.uniform(-0.015, 0.022)), 4)
    h_val = round((h0 + rng.uniform(-18.0, 18.0)) % 360.0, 2)
    steps = rng.choice((4, 5, 6, 7))
    hue_shift = (round(rng.uniform(-6.0, 0.0), 2), round(rng.uniform(0.0, 8.0), 2))
    return Palette(
        name=f"{raw_name}{suffix}",
        base_lch=(l_val, c_val, h_val),
        steps=steps,
        hue_shift=hue_shift,
        emissive=emissive,
    )


def _jpiv(rng: random.Random, base_xyz: tuple[float, float, float], span: float = 0.6) -> tuple[float, float, float]:
    _ = (rng, base_xyz, span)
    return (0.0, 0.0, 0.0)


def _static_profile_and_density(v_idx: int) -> tuple[str, int]:
    profile = "java_modern" if (v_idx // 3) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 6 == 0) else 16
    return profile, density


def _entity_profile_and_density(v_idx: int) -> tuple[str, int]:
    profiles = ("bbmodel_5", "bedrock_entity", "geckolib")
    profile = profiles[(v_idx // 2) % len(profiles)]
    density = 32 if (v_idx % 5 == 0) else 16
    return profile, density


# ==============================================================================
# NATURAL DOMAIN (3 Seen + 1 Unseen)
# ==============================================================================


def _gen_ancient_canopy_tree(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("buttress_oak_tree", "moss_elder_banyan", "autumn_maple_canopy", "blossom_spirit_tree")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_bark = _pal(rng, rng.choice(_BARK_EARTH_PALETTES))
    p_leaf = _pal(rng, _FOLIAGE_PALETTES[v_idx % len(_FOLIAGE_PALETTES)])
    p_moss = _pal(rng, rng.choice(_FOLIAGE_PALETTES), suffix="_moss")
    has_spore = (v_idx % 3 == 0)
    p_accent = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=has_spore, suffix="_accent")

    mound_w = rng.choice((8, 10))
    trunk_low_h = rng.choice((5, 6, 7))
    trunk_up_h = rng.choice((5, 6))
    bough_len = rng.choice((4, 5))
    lean_deg = rng.choice((-7.5, 0.0, 7.5))
    bough_deg = rng.choice((7.5, 11.25, 15.0))
    bark_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    bough_bone = Bone(
        "bough_limb_l",
        rest_euler=(0.0, 0.0, bough_deg),
        attach=Attach("bole_upper", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="bough_limb_r",
        items=[
            Box("bough_arm", (bough_len, 2, 2), p_bark.name),
            Box("sub_canopy_pad", (4, 2, 4), p_leaf.name, Place("bough_arm", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0))),
            Box("hanging_moss", (2, 2, 2), p_moss.name, Place("sub_canopy_pad", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    upper_items: list[Box | Bone] = [
        Box("bole_upper", (3, trunk_up_h, 3), p_bark.name),
        Box("shelf_conk_n", (2, 1, 1), p_accent.name, Place("bole_upper", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("crown_stem", (2, 2, 2), p_bark.name, Place("bole_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("main_canopy_dome", (6, 3, 6), p_leaf.name, Place("crown_stem", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("apical_foliage_cap", (4, 2, 4), p_moss.name, Place("main_canopy_dome", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        bough_bone,
    ]
    if has_spore:
        upper_items.append(
            Bone(
                "floating_wisp_orb",
                attach=Attach("apical_foliage_cap", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.25, 0.0), detached=True),
                items=[Box("spirit_spore_core", (2, 2, 2), p_accent.name)],
            )
        )

    root_items: list[Box | Bone] = [
        Box("earth_root_plate", (mound_w, 2, mound_w), p_moss.name),
        Box("buttress_root_n", (2, 2, 2), p_bark.name, Place("earth_root_plate", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("buttress_root_s", (2, 2, 2), p_bark.name, Place("earth_root_plate", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("buttress_root_e", (2, 2, 2), p_bark.name, Place("earth_root_plate", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        Box("buttress_root_w", (2, 2, 2), p_bark.name, Place("earth_root_plate", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        Box("bole_lower", (4, trunk_low_h, 4), p_bark.name, Place("earth_root_plate", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=bark_inf),
        Bone(
            "upper_bole_rig",
            rest_euler=(lean_deg, 0.0, 0.0),
            attach=Attach("bole_lower", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            items=upper_items,
        ),
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_bark, p_leaf, p_moss, p_accent],
        root=Bone("taproot_origin", pivot_local=_jpiv(rng, (8.0, 2.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 2 + trunk_low_h + trunk_up_h + 7,
        "secondary_span": 3 + 2 * (bough_len + 4),
        "joint_angle_deg": bough_deg,
        "has_mirror_x": True,
        "has_detached_bone": has_spore,
        "inflate_detail": bark_inf,
        "material_theme": f"{p_bark.name}+{p_leaf.name}+{p_moss.name}",
        "notes": f"buttress-rooted trunk ({trunk_low_h}+{trunk_up_h} tall) leaning {lean_deg:g} deg with mirrored {bough_len}-unit boughs, hanging moss, and layered canopy dome",
        "extra": {"mound_w": mound_w, "lean_deg": lean_deg, "has_spore": has_spore},
    }


def _gen_crystalline_geode_outcrop(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("amethyst_geode_cavity", "aquamarine_prism_vein", "citrine_spire_boulder", "biolum_cavern_geode")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_rock = _pal(rng, rng.choice(_STONE_MASONRY_PALETTES))
    p_crust = _pal(rng, rng.choice(_STONE_MASONRY_PALETTES), suffix="_crust")
    p_cryst = _pal(rng, _MINERAL_CORAL_PALETTES[v_idx % len(_MINERAL_CORAL_PALETTES)], emissive=True)
    p_moss = _pal(rng, rng.choice(_FOLIAGE_PALETTES))

    base_w = rng.choice((10, 12))
    main_spire_h = rng.choice((6, 7, 8))
    side_spire_len = rng.choice((4, 5))
    splay_deg = rng.choice((15.0, 22.5, 30.0))
    tilt_x = rng.choice((0.0, 5.0, 7.5, 11.25))
    crust_inf = round(rng.choice((0.0, 0.03, 0.06)), 3)

    main_spire = Bone(
        "central_crystal_spire",
        rest_euler=(tilt_x, 0.0, 0.0),
        attach=Attach("geode_basin", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("prism_column", (2, main_spire_h, 2), p_cryst.name),
            Box("prism_termination", (1, 3, 1), p_cryst.name, Place("prism_column", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    lateral_cluster = Bone(
        "lateral_prism_l",
        rest_euler=(0.0, 0.0, splay_deg),
        attach=Attach("geode_basin", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="lateral_prism_r",
        items=[
            Box("splay_crystal_shaft", (side_spire_len, 2, 2), p_cryst.name),
            Box("splay_crystal_point", (2, 1, 1), p_cryst.name, Place("splay_crystal_shaft", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    fore_crystal = Bone(
        "fore_needle_cluster",
        rest_euler=(15.0, 0.0, 0.0),
        attach=Attach("strata_shelf", (0.0, 1.0, 0.8), (0.0, -1.0, 0.0)),
        items=[
            Box("fore_prism", (2, 4, 2), p_cryst.name),
            Box("fore_apex", (1, 2, 1), p_cryst.name, Place("fore_prism", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    has_floating_shard = (v_idx % 2 == 0)
    root_items: list[Box | Bone] = [
        Box("bedrock_plinth", (base_w, 2, base_w), p_rock.name),
        Box("moss_skirt_e", (2, 1, 5), p_moss.name, Place("bedrock_plinth", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.3))),
        Box("moss_skirt_w", (2, 1, 5), p_moss.name, Place("bedrock_plinth", (-1.0, 0.5, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -0.3))),
        Box("strata_shelf", (base_w - 2, 2, base_w - 2), p_crust.name, Place("bedrock_plinth", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=crust_inf),
        Box("geode_basin", (6, 2, 5), p_crust.name, Place("strata_shelf", (0.0, 1.0, -0.2), (0.0, -1.0, 0.0))),
        Box("rear_hood_wall", (6, 4, 1), p_rock.name, Place("geode_basin", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
        Box("hood_cap_stone", (4, 1, 2), p_moss.name, Place("rear_hood_wall", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        main_spire,
        lateral_cluster,
        fore_crystal,
    ]
    if has_floating_shard:
        root_items.append(
            Bone(
                "resonant_crystal_mote",
                rest_euler=(0.0, 18.4, 0.0),
                attach=Attach("hood_cap_stone", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 2.2, 0.0), detached=True),
                items=[Box("levitating_shard", (1, 2, 1), p_cryst.name)],
            )
        )

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_rock, p_crust, p_cryst, p_moss],
        root=Bone("outcrop_root", pivot_local=(0.0, 0.0, 0.0), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 6 + main_spire_h + 3,
        "secondary_span": base_w + 4,
        "joint_angle_deg": splay_deg,
        "has_mirror_x": True,
        "has_detached_bone": has_floating_shard,
        "inflate_detail": crust_inf,
        "material_theme": f"{p_rock.name}+{p_cryst.name}+{p_moss.name}",
        "notes": f"stepped {base_w}-wide bedrock geode basin with rear hood wall, emissive {main_spire_h}-tall central prism, mirrored {splay_deg:g} deg splay crystals, and moss skirts",
        "extra": {"base_w": base_w, "main_spire_h": main_spire_h},
    }


def _gen_coral_fungal_colony(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("staghorn_reef_colony", "biolum_shelf_mycelium", "tiered_umbrella_fungi", "brain_coral_polyp_mound")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_substrate = _pal(rng, rng.choice(_STONE_MASONRY_PALETTES))
    p_stalk = _pal(rng, rng.choice(_BARK_EARTH_PALETTES))
    p_cap = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=(v_idx % 2 == 1))
    p_polyp = _pal(rng, rng.choice(_FOLIAGE_PALETTES), suffix="_polyp")

    stalk_h = rng.choice((5, 6, 7))
    cap_w = rng.choice((6, 8))
    branch_len = rng.choice((3, 4))
    branch_deg = rng.choice((15.0, 22.5))
    floating_spores = (v_idx % 2 == 0)
    gill_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    antler_branch = Bone(
        "colony_branch_l",
        rest_euler=(0.0, 0.0, branch_deg),
        attach=Attach("reef_mound", (1.0, 0.4, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="colony_branch_r",
        items=[
            Box("branch_stipe", (branch_len, 2, 2), p_stalk.name),
            Box("polyp_cup", (3, 2, 3), p_cap.name, Place("branch_stipe", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0))),
            Box("polyp_crown_tip", (1, 2, 1), p_polyp.name, Place("polyp_cup", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    cap_items: list[Box | Bone] = [
        Box("central_stipe", (2, stalk_h, 2), p_stalk.name),
        Box("gill_annulus", (4, 1, 4), p_polyp.name, Place("central_stipe", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=gill_inf),
        Box("umbrella_pileus", (cap_w, 2, cap_w), p_cap.name, Place("gill_annulus", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("pileus_umbo", (cap_w - 3, 1, cap_w - 3), p_cap.name, Place("umbrella_pileus", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
    ]
    if floating_spores:
        cap_items.append(
            Bone(
                "drifting_spore_halo",
                attach=Attach("pileus_umbo", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.3, 0.0), detached=True),
                items=[
                    Box("spore_nucleus", (2, 1, 2), p_polyp.name),
                    Box("spore_satellite", (1, 1, 1), p_cap.name, Place("spore_nucleus", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                ],
            )
        )

    root_items: list[Box | Bone] = [
        Box("substrate_slab", (10, 2, 10), p_substrate.name),
        Box("reef_mound", (6, 3, 6), p_polyp.name, Place("substrate_slab", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("front_polyp_nodule", (3, 2, 2), p_cap.name, Place("reef_mound", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("rear_polyp_nodule", (3, 2, 2), p_cap.name, Place("reef_mound", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        antler_branch,
        Bone(
            "central_fruiting_stem",
            rest_euler=(rng.choice((-7.5, 0.0, 7.5)), 0.0, 0.0),
            attach=Attach("reef_mound", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            items=cap_items,
        ),
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_substrate, p_stalk, p_cap, p_polyp],
        root=Bone("colony_base", pivot_local=_jpiv(rng, (8.0, 2.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 5 + stalk_h + 4,
        "secondary_span": 6 + 2 * (branch_len + 3),
        "joint_angle_deg": branch_deg,
        "has_mirror_x": True,
        "has_detached_bone": floating_spores,
        "inflate_detail": gill_inf,
        "material_theme": f"{p_substrate.name}+{p_cap.name}+{p_polyp.name}",
        "notes": f"substrate reef/mycelial mound with {cap_w}-wide umbrella cap on {stalk_h}-tall stipe, mirrored {branch_len}-unit polyp branches at {branch_deg:g} deg",
        "extra": {"cap_w": cap_w, "floating_spores": floating_spores},
    }


def _gen_weeping_bonsai_arch(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Natural family (>= 16 boxes): root-over-rock cascading bonsai in ceramic tray."""
    substyles = ("root_over_rock_pine", "weeping_juniper_cascade", "scholar_stone_maple_bonsai")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_pot = _pal(rng, rng.choice(_STONE_MASONRY_PALETTES))
    p_bark = _pal(rng, rng.choice(_BARK_EARTH_PALETTES))
    p_leaf = _pal(rng, rng.choice(_FOLIAGE_PALETTES))
    p_rock = _pal(rng, rng.choice(_STONE_MASONRY_PALETTES), suffix="_scholar")

    tray_w = rng.choice((12, 14))
    trunk_h = rng.choice((5, 6))
    arch_len = rng.choice((5, 6))
    arch_deg = rng.choice((-15.0, -11.25))
    wing_deg = rng.choice((11.25, 15.0))
    pot_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    cloud_branch = Bone(
        "cloud_bough_l",
        rest_euler=(0.0, 0.0, wing_deg),
        attach=Attach("ascending_trunk", (0.0, 0.4, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("fore_twig", (2, 2, 3), p_bark.name),
            Box("fore_cloud_pad", (4, 1, 4), p_leaf.name, Place("fore_twig", (0.0, 1.0, 1.0), (0.0, -1.0, 0.0))),
            Box("fore_weeping_frond", (1, 3, 1), p_leaf.name, Place("fore_cloud_pad", (0.0, -1.0, 0.6), (0.0, 1.0, 0.0))),
        ],
    )

    aft_cloud_branch = Bone(
        "aft_cloud_bough",
        rest_euler=(-11.25, 0.0, 0.0),
        attach=Attach("ascending_trunk", (0.0, 0.5, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("aft_twig", (2, 2, 3), p_bark.name),
            Box("aft_cloud_pad", (4, 1, 4), p_leaf.name, Place("aft_twig", (0.0, 1.0, -1.0), (0.0, -1.0, 0.0))),
            Box("aft_weeping_frond", (1, 3, 1), p_leaf.name, Place("aft_cloud_pad", (0.0, -1.0, -0.6), (0.0, 1.0, 0.0))),
        ],
    )

    cascade_bone = Bone(
        "cascade_arch_limb",
        rest_euler=(0.0, 0.0, arch_deg),
        attach=Attach("trunk_elbow", (-1.0, 0.5, 0.0), (1.0, 0.0, 0.0)),
        items=[
            Box("arch_beam", (arch_len, 2, 2), p_bark.name),
            Box("upper_canopy_cloud", (5, 2, 5), p_leaf.name, Place("arch_beam", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("apex_needle_tuft", (3, 1, 3), p_leaf.name, Place("upper_canopy_cloud", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("cascade_tip_pad", (4, 1, 4), p_leaf.name, Place("arch_beam", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            Box("weeping_tassel_n", (1, 3, 1), p_leaf.name, Place("cascade_tip_pad", (-0.5, -1.0, -0.6), (0.0, 1.0, 0.0))),
            Box("weeping_tassel_s", (1, 3, 1), p_leaf.name, Place("cascade_tip_pad", (-0.5, -1.0, 0.6), (0.0, 1.0, 0.0))),
        ],
    )

    trunk_bone = Bone(
        "bonsai_trunk_rig",
        attach=Attach("moss_soil_bed", (0.45, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("nebari_root_clasp", (3, 2, 3), p_bark.name),
            Box("ascending_trunk", (2, trunk_h, 2), p_bark.name, Place("nebari_root_clasp", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("trunk_elbow", (3, 2, 2), p_bark.name, Place("ascending_trunk", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            cloud_branch,
            aft_cloud_branch,
            cascade_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("ceramic_bonsai_tray", (tray_w, 2, 8), p_pot.name, inflate=pot_inf),
        Box("tray_foot_east", (2, 1, 6), p_pot.name, Place("ceramic_bonsai_tray", (0.75, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("tray_foot_west", (2, 1, 6), p_pot.name, Place("ceramic_bonsai_tray", (-0.75, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("moss_soil_bed", (tray_w - 2, 1, 6), p_leaf.name, Place("ceramic_bonsai_tray", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("scholar_viewing_stone", (3, 4, 3), p_rock.name, Place("moss_soil_bed", (-0.55, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("stone_peak_spire", (2, 2, 2), p_rock.name, Place("scholar_viewing_stone", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        trunk_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_pot, p_bark, p_leaf, p_rock],
        root=Bone("bonsai_tray_root", pivot_local=_jpiv(rng, (8.0, 3.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 5 + trunk_h + 6,
        "secondary_span": tray_w,
        "joint_angle_deg": arch_deg,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": pot_inf,
        "material_theme": f"{p_pot.name}+{p_bark.name}+{p_leaf.name}",
        "notes": f"footed {tray_w}-wide ceramic bonsai tray with scholar viewing stone, nebari root clasp, {trunk_h}-tall trunk, cascading {arch_len}-unit arch limb, cloud foliage pads, and weeping fronds",
        "extra": {"tray_w": tray_w, "arch_len": arch_len},
    }


# ==============================================================================
# INDUSTRIAL DOMAIN (2 New Seen + 1 New Unseen; plus gearbox/telescope/forge)
# ==============================================================================


def _gen_articulated_robotic_arm(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("six_axis_assembly_robot", "servo_pick_place_arm", "heavy_welding_manipulator", "cleanroom_wafer_robot")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_paint = _pal(rng, _INDUSTRIAL_PALETTES[v_idx % 2])
    p_alloy = _pal(rng, rng.choice(_INDUSTRIAL_PALETTES[2:4]))
    p_dark = _pal(rng, _INDUSTRIAL_PALETTES[5])
    p_beacon = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=True)

    waist_yaw = rng.choice((-22.5, -15.0, 0.0, 15.0, 22.5))
    shoulder_pitch = rng.choice((-11.25, -7.5, 0.0, 7.5))
    elbow_pitch = rng.choice((-22.5, -15.0, -7.5, 0.0))
    boom_h = rng.choice((5, 6, 7))
    stick_len = rng.choice((4, 5, 6))
    collar_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    gripper_jaw = Bone(
        "gripper_finger_l",
        rest_euler=(0.0, rng.choice((-7.5, 0.0, 7.5)), 0.0),
        attach=Attach("wrist_flange", (0.75, 0.0, 1.0), (0.0, 0.0, -1.0)),
        mirror_name="gripper_finger_r",
        items=[
            Box("jaw_knuckle", (1, 1, 2), p_alloy.name),
            Box("jaw_pad", (1, 1, 2), p_dark.name, Place("jaw_knuckle", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    forearm_bone = Bone(
        "forearm_stick_stage",
        rest_euler=(elbow_pitch, 0.0, 0.0),
        attach=Attach("elbow_servo_drum", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("forearm_tube", (2, 2, stick_len), p_paint.name),
            Box("cable_conduit_top", (1, 1, stick_len - 1), p_dark.name, Place("forearm_tube", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("wrist_flange", (3, 2, 1), p_alloy.name, Place("forearm_tube", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), inflate=collar_inf),
            Box("optic_camera_pod", (1, 1, 2), p_beacon.name, Place("wrist_flange", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            gripper_jaw,
        ],
    )

    shoulder_bone = Bone(
        "shoulder_boom_stage",
        rest_euler=(shoulder_pitch, 0.0, 0.0),
        attach=Attach("turret_yoke", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("upper_boom_spar", (2, boom_h, 2), p_paint.name),
            Box("hydraulic_cylinder_back", (1, max(2, boom_h - 2), 1), p_alloy.name, Place("upper_boom_spar", (0.0, 0.25, -1.0), (0.0, 0.0, 1.0))),
            Box("elbow_servo_drum", (4, 2, 2), p_dark.name, Place("upper_boom_spar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            forearm_bone,
        ],
    )

    waist_bone = Bone(
        "rotary_waist_turret",
        rest_euler=(0.0, waist_yaw, 0.0),
        attach=Attach("slew_ring_collar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("turret_yoke", (4, 3, 4), p_paint.name),
            Box("counterweight_pack", (4, 2, 2), p_dark.name, Place("turret_yoke", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            shoulder_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("anchor_baseplate", (8, 2, 8), p_dark.name),
        Box("plc_control_cabinet", (2, 4, 4), p_alloy.name, Place("anchor_baseplate", (-1.0, 0.5, 0.0), (1.0, 0.0, 0.0))),
        Box("status_stack_light", (1, 2, 1), p_beacon.name, Place("plc_control_cabinet", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("slew_ring_collar", (5, 2, 5), p_alloy.name, Place("anchor_baseplate", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        waist_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_paint, p_alloy, p_dark, p_beacon],
        root=Bone("pedestal_datum", pivot_local=_jpiv(rng, (8.0, 2.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 6 + boom_h + stick_len,
        "secondary_span": 8,
        "joint_angle_deg": waist_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": collar_inf,
        "material_theme": f"{p_paint.name}+{p_alloy.name}+{p_dark.name}",
        "notes": f"anchored pedestal with PLC cabinet, {waist_yaw:g} deg rotary waist turret, {boom_h}-tall shoulder boom, {elbow_pitch:g} deg forearm stick, optic camera pod, and mirrored parallel gripper jaws",
        "extra": {"waist_yaw": waist_yaw, "shoulder_pitch": shoulder_pitch, "elbow_pitch": elbow_pitch},
    }


def _gen_industrial_turbine_generator(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("gas_turbine_dynamo", "steam_impeller_skid", "hydro_pelton_generator", "cryo_compressor_turbine")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_casing = _pal(rng, rng.choice(_INDUSTRIAL_PALETTES[:5]))
    p_rotor = _pal(rng, _INDUSTRIAL_PALETTES[3], suffix="_rotor")
    p_skid = _pal(rng, _INDUSTRIAL_PALETTES[2], suffix="_skid")
    p_gauge = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=True)

    barrel_len = rng.choice((7, 8, 9))
    blade_len = rng.choice((3, 4))
    rotor_roll = rng.choice((0.0, 15.0, 22.5, 30.0, 45.0))
    duct_deg = rng.choice((11.25, 15.0, 22.5))
    flange_inf = round(rng.choice((0.0, 0.04, 0.07)), 3)

    rotor_bone = Bone(
        "impeller_rotor_shaft",
        rest_euler=(0.0, 0.0, rotor_roll),
        attach=Attach("intake_bellmouth", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("spinner_nose_cone", (2, 2, 2), p_rotor.name),
            Box("fan_blade_north_up", (1, blade_len, 1), p_rotor.name, Place("spinner_nose_cone", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("fan_blade_south_dn", (1, blade_len, 1), p_rotor.name, Place("spinner_nose_cone", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("fan_blade_east", (blade_len, 1, 1), p_rotor.name, Place("spinner_nose_cone", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("fan_blade_west", (blade_len, 1, 1), p_rotor.name, Place("spinner_nose_cone", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        ],
    )

    exhaust_wing = Bone(
        "exhaust_manifold_l",
        rest_euler=(0.0, 0.0, duct_deg),
        attach=Attach("stator_barrel", (1.0, 0.3, 0.3), (-1.0, 0.0, 0.0)),
        mirror_name="exhaust_manifold_r",
        items=[
            Box("manifold_elbow", (3, 2, 2), p_casing.name),
            Box("exhaust_stack_cap", (1, 3, 2), p_rotor.name, Place("manifold_elbow", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("heavy_skid_deck", (10, 2, barrel_len + 3), p_skid.name),
        Box("cradle_saddle_fore", (6, 2, 2), p_skid.name, Place("heavy_skid_deck", (0.0, 1.0, -0.5), (0.0, -1.0, 0.0))),
        Box("cradle_saddle_aft", (6, 2, 2), p_skid.name, Place("heavy_skid_deck", (0.0, 1.0, 0.5), (0.0, -1.0, 0.0))),
        Box("stator_barrel", (6, 6, barrel_len), p_casing.name, Place("cradle_saddle_fore", (0.0, 1.0, 1.0), (0.0, -1.0, 0.25)), inflate=flange_inf),
        # Wait: keep stator_barrel centered cleanly above heavy_skid_deck via central pedestal!
    ]
    # Rebuild root_items cleanly so stator_barrel sits flush on a central_cradle with zero sibling ambiguity:
    root_items = [
        Box("heavy_skid_deck", (10, 2, barrel_len + 4), p_skid.name),
        Box("central_stator_cradle", (6, 2, barrel_len - 2), p_skid.name, Place("heavy_skid_deck", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("stator_barrel", (6, 6, barrel_len), p_casing.name, Place("central_stator_cradle", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=flange_inf),
        Box("top_heat_exchanger", (4, 2, barrel_len - 2), p_rotor.name, Place("stator_barrel", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("telemetry_gauge_pod", (2, 2, 2), p_gauge.name, Place("top_heat_exchanger", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("aft_dynamo_tailcone", (4, 4, 2), p_rotor.name, Place("stator_barrel", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("intake_bellmouth", (4, 4, 1), p_rotor.name, Place("stator_barrel", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        rotor_bone,
        exhaust_wing,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_casing, p_rotor, p_skid, p_gauge],
        root=Bone("turbine_skid_root", pivot_local=_jpiv(rng, (8.0, 3.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": barrel_len + 6,
        "secondary_span": 6 + 2 * 4,
        "joint_angle_deg": rotor_roll,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": flange_inf,
        "material_theme": f"{p_casing.name}+{p_rotor.name}+{p_skid.name}",
        "notes": f"skid-mounted {barrel_len}-long stator barrel with top heat exchanger, 4-bladed impeller rolled to {rotor_roll:g} deg, aft dynamo cone, and mirrored exhaust manifolds",
        "extra": {"barrel_len": barrel_len, "blade_len": blade_len},
    }


def _gen_hydraulic_gantry_crane(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Industrial family (>= 18 boxes): twin-portal gantry crane with trolley & spreader hook."""
    substyles = ("shipyard_portal_gantry", "foundry_beam_crane", "container_hoist_gantry")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_girder = _pal(rng, _INDUSTRIAL_PALETTES[0])
    p_strut = _pal(rng, _INDUSTRIAL_PALETTES[2])
    p_hoist = _pal(rng, _INDUSTRIAL_PALETTES[3])
    p_beacon = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=True)

    span_x = rng.choice((16, 18))
    leg_h = rng.choice((10, 12))
    hoist_drop = rng.choice((3, 4))
    jib_pitch = rng.choice((-11.25, 0.0, 11.25))
    beam_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    portal_leg = Bone(
        "portal_column_l",
        attach=Attach("bridge_box_girder", (0.85, -1.0, 0.0), (0.0, 1.0, 0.0)),
        mirror_name="portal_column_r",
        items=[
            Box("column_capital", (2, 2, 3), p_girder.name),
            Box("vertical_stanchion", (2, leg_h, 2), p_strut.name, Place("column_capital", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("bogie_sill_beam", (3, 2, 8), p_girder.name, Place("vertical_stanchion", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("rail_wheel_fore", (2, 2, 2), p_hoist.name, Place("bogie_sill_beam", (0.0, -1.0, -0.75), (0.0, 1.0, 0.0))),
            Box("rail_wheel_aft", (2, 2, 2), p_hoist.name, Place("bogie_sill_beam", (0.0, -1.0, 0.75), (0.0, 1.0, 0.0))),
            Box("hydraulic_brace_pod", (1, 4, 2), p_hoist.name, Place("vertical_stanchion", (1.0, 0.2, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    spreader_bone = Bone(
        "spreader_swivel_rig",
        rest_euler=(jib_pitch, 0.0, 0.0),
        attach=Attach("winch_cable_mast", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("spreader_crossbar", (6, 1, 2), p_girder.name),
            Box("twistlock_claw_e", (1, 2, 2), p_hoist.name, Place("spreader_crossbar", (0.8, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("twistlock_claw_w", (1, 2, 2), p_hoist.name, Place("spreader_crossbar", (-0.8, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    trolley_bone = Bone(
        "hoist_trolley_carriage",
        attach=Attach("bridge_box_girder", (rng.choice((-0.2, 0.0, 0.2)), -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("trolley_chassis", (4, 2, 4), p_strut.name),
            Box("winch_cable_mast", (2, hoist_drop, 2), p_hoist.name, Place("trolley_chassis", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            spreader_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box(
            "bridge_box_girder",
            (span_x, 3, 4),
            p_girder.name,
            Place(None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 14.0, 0.0)),
            inflate=beam_inf,
        ),
        Box("catwalk_deck_top", (span_x - 2, 1, 2), p_hoist.name, Place("bridge_box_girder", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("operator_cab_pod", (3, 2, 2), p_strut.name, Place("bridge_box_girder", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("warning_strobe_top", (2, 2, 2), p_beacon.name, Place("catwalk_deck_top", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        portal_leg,
        trolley_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_girder, p_strut, p_hoist, p_beacon],
        root=Bone("gantry_datum", pivot_local=_jpiv(rng, (8.0, 20.0, 8.0), span=0.4), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": span_x,
        "secondary_span": leg_h + 9,
        "joint_angle_deg": jib_pitch,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": beam_inf,
        "material_theme": f"{p_girder.name}+{p_strut.name}+{p_hoist.name}",
        "notes": f"{span_x}-wide overhead bridge girder with top catwalk, warning strobe, mirrored {leg_h}-tall wheeled bogie portal columns, trolley carriage, and articulated spreader claws",
        "extra": {"span_x": span_x, "leg_h": leg_h},
    }


# ==============================================================================
# BUILDING / ARCHITECTURE DOMAIN (3 Seen + 1 New Unseen; plus folding_windmill)
# ==============================================================================


def _gen_modular_timber_stone_building(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("half_timber_cottage", "guildhall_workshop", "gabled_stone_tavern", "alpine_chalet_house")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_stone = _pal(rng, _STONE_MASONRY_PALETTES[0])
    p_timber = _pal(rng, rng.choice(_BARK_EARTH_PALETTES))
    p_roof = _pal(rng, _STONE_MASONRY_PALETTES[3 + (v_idx % 3)])
    p_window = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=True)

    hall_w = rng.choice((8, 10))
    hall_d = rng.choice((8, 10))
    wall_h = rng.choice((6, 7))
    roof_pitch = rng.choice((-22.5, -30.0))
    beam_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    bay_wing = Bone(
        "side_bay_window_l",
        attach=Attach("timber_frame_hall", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="side_bay_window_r",
        items=[
            Box("oriel_corbel_box", (2, 3, 4), p_timber.name),
            Box("leaded_casement_pane", (1, 2, 2), p_window.name, Place("oriel_corbel_box", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), offset=(0.0, 0.0, 0.0))
            if False
            else Box("leaded_casement_pane", (1, 2, 2), p_window.name, Place("oriel_corbel_box", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    roof_slope = Bone(
        "gable_roof_slope_l",
        rest_euler=(0.0, 0.0, roof_pitch),
        attach=Attach("ridge_kingpost_beam", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="gable_roof_slope_r",
        items=[
            Box("rafter_shingle_slab", (6, 1, hall_d + 2), p_roof.name),
            Box("eave_fascia_trim", (1, 1, hall_d + 2), p_timber.name, Place("rafter_shingle_slab", (1.0, -0.5, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    attic_bone = Bone(
        "gable_roof_truss",
        attach=Attach("timber_frame_hall", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("attic_tie_deck", (hall_w, 1, hall_d), p_timber.name, inflate=beam_inf),
            Box("gable_core_wall", (4, 3, hall_d), p_stone.name, Place("attic_tie_deck", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("ridge_kingpost_beam", (2, 2, hall_d + 2), p_timber.name, Place("gable_core_wall", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("hearth_chimney_flue", (2, 3, 2), p_stone.name, Place("ridge_kingpost_beam", (0.0, 1.0, -0.55), (0.0, -1.0, 0.0))),
            Box("chimney_crown_pot", (1, 2, 1), p_roof.name, Place("hearth_chimney_flue", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            roof_slope,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("ashlar_plinth", (hall_w + 2, 2, hall_d + 2), p_stone.name),
        Box("entry_porch_step", (4, 1, 2), p_stone.name, Place("ashlar_plinth", (0.0, -0.5, 1.0), (0.0, 0.0, -1.0))),
        Box("timber_frame_hall", (hall_w, wall_h, hall_d), p_stone.name, Place("ashlar_plinth", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("front_door_canopy", (4, 1, 2), p_timber.name, Place("timber_frame_hall", (0.0, -0.15, 1.0), (0.0, 0.0, -1.0))),
        Box("front_lantern_window", (2, 2, 1), p_window.name, Place("timber_frame_hall", (0.0, 0.6, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.25))),
        bay_wing,
        attic_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_stone, p_timber, p_roof, p_window],
        root=Bone("foundation_datum", pivot_local=_jpiv(rng, (8.0, 2.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 2 + wall_h + 7,
        "secondary_span": hall_w + 6,
        "joint_angle_deg": roof_pitch,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": beam_inf,
        "material_theme": f"{p_stone.name}+{p_timber.name}+{p_roof.name}",
        "notes": f"ashlar stone plinth with porch step, {hall_w}x{wall_h}x{hall_d} timber-framed hall, mirrored oriel bay windows, rear chimney flue, and {roof_pitch:g} deg pitched gable roof",
        "extra": {"hall_w": hall_w, "hall_d": hall_d, "wall_h": wall_h},
    }


def _gen_tiered_pagoda_watchtower(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("upturned_eave_pagoda", "frontier_timber_watchtower", "jade_tiled_bell_pagoda", "shrine_lantern_spire")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_base = _pal(rng, _STONE_MASONRY_PALETTES[1])
    p_wood = _pal(rng, _BARK_EARTH_PALETTES[1])
    p_tile = _pal(rng, _STONE_MASONRY_PALETTES[4 if v_idx % 2 == 0 else 5])
    p_gold = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=(v_idx % 3 == 0))

    low_h = rng.choice((5, 6))
    up_h = rng.choice((4, 5))
    eave_tilt = rng.choice((11.25, 15.0, 22.5))
    trim_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    lower_eave_wing = Bone(
        "lower_flying_eave_l",
        rest_euler=(0.0, 0.0, eave_tilt),
        attach=Attach("lower_eave_cornice", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="lower_flying_eave_r",
        items=[
            Box("lower_eave_tip", (2, 1, 10), p_tile.name),
            Box("wind_bell_pendant", (1, 2, 1), p_gold.name, Place("lower_eave_tip", (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0))),
        ],
    )

    upper_eave_wing = Bone(
        "upper_flying_eave_l",
        rest_euler=(0.0, 0.0, eave_tilt),
        attach=Attach("upper_eave_cornice", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="upper_flying_eave_r",
        items=[Box("upper_eave_tip", (2, 1, 8), p_tile.name)],
    )

    spire_bone = Bone(
        "sorin_finial_spire",
        attach=Attach("upper_hip_crown", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("dew_basin_roban", (2, 1, 2), p_gold.name),
            Box("sorin_ring_column", (1, 4, 1), p_gold.name, Place("dew_basin_roban", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("sacred_flame_jewel", (2, 2, 2), p_gold.name, Place("sorin_ring_column", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    upper_story_bone = Bone(
        "second_tier_pavilion",
        attach=Attach("lower_roof_step", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("upper_sanctuary_core", (6, up_h, 6), p_wood.name),
            Box("upper_eave_cornice", (8, 1, 8), p_tile.name, Place("upper_sanctuary_core", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("upper_hip_crown", (4, 2, 4), p_tile.name, Place("upper_eave_cornice", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            upper_eave_wing,
            spire_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("battered_stone_podium", (12, 3, 12), p_base.name),
        Box("entry_stair_flight", (4, 2, 2), p_base.name, Place("battered_stone_podium", (0.0, -0.33, 1.0), (0.0, 0.0, -1.0))),
        Box("lower_hall_chamber", (8, low_h, 8), p_wood.name, Place("battered_stone_podium", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=trim_inf),
        Box("lower_eave_cornice", (10, 1, 10), p_tile.name, Place("lower_hall_chamber", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("lower_roof_step", (7, 1, 7), p_tile.name, Place("lower_eave_cornice", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        lower_eave_wing,
        upper_story_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_base, p_wood, p_tile, p_gold],
        root=Bone("pagoda_podium_root", pivot_local=_jpiv(rng, (8.0, 2.0, 8.0), span=0.4), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 5 + low_h + up_h + 10,
        "secondary_span": 14,
        "joint_angle_deg": eave_tilt,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": trim_inf,
        "material_theme": f"{p_base.name}+{p_wood.name}+{p_tile.name}",
        "notes": f"2-tier pagoda watchtower on battered stone podium with mirrored {eave_tilt:g} deg upturned eaves, pendant bells, upper sanctuary, and sorin finial spire",
        "extra": {"low_h": low_h, "up_h": up_h},
    }


def _gen_arched_stone_bridge_gate(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("crenellated_canal_bridge", "twin_pier_viaduct_gate", "buttressed_fortress_span", "lantern_arch_footbridge")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_ashlar = _pal(rng, _STONE_MASONRY_PALETTES[0])
    p_pier = _pal(rng, _STONE_MASONRY_PALETTES[2])
    p_timber = _pal(rng, rng.choice(_BARK_EARTH_PALETTES))
    p_brazier = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=True)

    span_w = rng.choice((8, 10))
    pier_h = rng.choice((6, 7, 8))
    ramp_deg = rng.choice((-11.25, 0.0, 11.25))
    coping_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    abutment_pier = Bone(
        "abutment_tower_l",
        attach=Attach("keystone_road_deck", (1.0, 0.0, 0.0), (-1.0, 0.25, 0.0)),
        mirror_name="abutment_tower_r",
        items=[
            Box("pier_shaft_block", (4, pier_h, 8), p_pier.name),
            Box("cutwater_footing", (5, 2, 10), p_pier.name, Place("pier_shaft_block", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("approach_ramp_slab", (3, 1, 6), p_timber.name, Place("pier_shaft_block", (1.0, 0.25, 0.0), (-1.0, 0.0, 0.0))),
            Box("watch_parapet_turret", (4, 3, 4), p_ashlar.name, Place("pier_shaft_block", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("turret_brazier_flame", (2, 2, 2), p_brazier.name, Place("watch_parapet_turret", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    gate_portal_bone = Bone(
        "overhead_gate_arch",
        rest_euler=(ramp_deg, 0.0, 0.0),
        attach=Attach("parapet_wall_north", (0.0, 1.0, 0.0), (0.0, -1.0, -1.0)),
        items=[
            Box("north_gate_pilaster", (2, 4, 1), p_ashlar.name),
            Box("gate_cross_lintel", (4, 1, 8), p_timber.name, Place("north_gate_pilaster", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
            Box("crest_shield_boss", (2, 2, 1), p_brazier.name, Place("gate_cross_lintel", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("keystone_road_deck", (span_w, 2, 8), p_ashlar.name, inflate=coping_inf),
        Box("voussoir_arch_rib", (span_w - 2, 1, 6), p_pier.name, Place("keystone_road_deck", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("parapet_wall_north", (span_w, 2, 1), p_ashlar.name, Place("keystone_road_deck", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
        Box("parapet_wall_south", (span_w, 2, 1), p_ashlar.name, Place("keystone_road_deck", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        abutment_pier,
        gate_portal_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_ashlar, p_pier, p_timber, p_brazier],
        root=Bone("bridge_crown_datum", pivot_local=_jpiv(rng, (8.0, 10.0, 8.0), span=0.4), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": span_w + 14,
        "secondary_span": pier_h + 7,
        "joint_angle_deg": ramp_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": coping_inf,
        "material_theme": f"{p_ashlar.name}+{p_pier.name}+{p_timber.name}",
        "notes": f"{span_w}-unit voussoir arch road deck with north/south parapets, overhead gate portal, and mirrored {pier_h}-tall cutwater abutment towers crowned with braziers",
        "extra": {"span_w": span_w, "pier_h": pier_h},
    }


def _gen_coastal_lighthouse_keep(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Building family (>= 18 boxes): headland lighthouse tower, keeper annex & Fresnel cupola."""
    substyles = ("headland_beacon_keep", "striped_harbor_lighthouse", "stormwall_pharos_tower")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)

    p_rock = _pal(rng, _STONE_MASONRY_PALETTES[2])
    p_tower = _pal(rng, _STONE_MASONRY_PALETTES[0])
    p_roof = _pal(rng, _STONE_MASONRY_PALETTES[3])
    p_lens = _pal(rng, _MINERAL_CORAL_PALETTES[2], emissive=True)

    s1_h = rng.choice((5, 6))
    s2_h = rng.choice((5, 6))
    s3_h = rng.choice((4, 5))
    vane_yaw = rng.choice((-15.0, 0.0, 15.0))
    cornice_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    balcony_rail = Bone(
        "gallery_rail_wing_l",
        attach=Attach("corbel_gallery_deck", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0)),
        mirror_name="gallery_rail_wing_r",
        items=[
            Box("side_parapet_rail", (1, 2, 6), p_roof.name),
            Box("storm_horn_siren", (1, 1, 2), p_tower.name, Place("side_parapet_rail", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    lantern_bone = Bone(
        "fresnel_lantern_cupola",
        rest_euler=(0.0, vane_yaw, 0.0),
        attach=Attach("corbel_gallery_deck", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("lantern_mullion_base", (4, 1, 4), p_roof.name),
            Box("fresnel_optic_drum", (3, 3, 3), p_lens.name, Place("lantern_mullion_base", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("copper_cupola_dome", (4, 2, 4), p_roof.name, Place("fresnel_optic_drum", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("lightning_finial_rod", (1, 3, 1), p_tower.name, Place("copper_cupola_dome", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("wind_vane_pennant", (2, 1, 1), p_roof.name, Place("lightning_finial_rod", (1.0, 0.5, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    gallery_bone = Bone(
        "watch_gallery_stage",
        attach=Attach("tower_shaft_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("corbel_gallery_deck", (8, 1, 8), p_tower.name, inflate=cornice_inf),
            balcony_rail,
            lantern_bone,
        ],
    )

    annex_bone = Bone(
        "keeper_cottage_annex",
        attach=Attach("sea_crag_promontory", (0.0, 1.0, 0.65), (0.0, -1.0, 0.0)),
        items=[
            Box("annex_stone_room", (5, 3, 4), p_tower.name),
            Box("annex_pitched_roof", (6, 2, 5), p_roof.name, Place("annex_stone_room", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("annex_porch_lantern", (1, 2, 1), p_lens.name, Place("annex_stone_room", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("sea_crag_promontory", (12, 2, 16), p_rock.name),
        Box("breakwater_seawall_n", (10, 2, 2), p_rock.name, Place("sea_crag_promontory", (0.0, 1.0, -0.85), (0.0, -1.0, 0.0))),
        Box("tower_shaft_lower", (7, s1_h, 7), p_tower.name, Place("sea_crag_promontory", (0.0, 1.0, -0.25), (0.0, -1.0, 0.0))),
        Box("tower_shaft_mid", (6, s2_h, 6), p_roof.name, Place("tower_shaft_lower", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("lance_window_slit", (2, 2, 1), p_lens.name, Place("tower_shaft_mid", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.25))),
        Box("tower_shaft_upper", (5, s3_h, 5), p_tower.name, Place("tower_shaft_mid", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        annex_bone,
        gallery_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_rock, p_tower, p_roof, p_lens],
        root=Bone("headland_datum", pivot_local=_jpiv(rng, (8.0, 1.5, 8.0), span=0.35), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 2 + s1_h + s2_h + s3_h + 10,
        "secondary_span": 14,
        "joint_angle_deg": vane_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": cornice_inf,
        "material_theme": f"{p_rock.name}+{p_tower.name}+{p_roof.name}",
        "notes": f"sea-crag promontory with breakwater, keeper cottage annex, 3-stage tapered lighthouse shaft ({s1_h}+{s2_h}+{s3_h}), cantilevered gallery, emissive Fresnel drum, and lightning cupola",
        "extra": {"s1_h": s1_h, "s2_h": s2_h, "s3_h": s3_h},
    }


# ==============================================================================
# HUMANOID, ANIMAL & DRAGON DOMAINS (4 Seen + 3 Unseen; archetype="entity")
# ==============================================================================


def _gen_humanoid_biped(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("armored_paladin_knight", "village_craft_explorer", "deep_space_astronaut", "clockwork_brass_automaton")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)

    p_primary = _pal(rng, _HUMANOID_GEAR_PALETTES[v_idx % 5])
    p_cloth = _pal(rng, rng.choice(_HUMANOID_GEAR_PALETTES[1:4]), suffix="_trim")
    p_skin = _pal(rng, _HUMANOID_GEAR_PALETTES[5] if v_idx % 4 == 1 else _HUMANOID_GEAR_PALETTES[0], suffix="_visage")
    p_visor = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=(v_idx % 2 == 0))

    torso_h = rng.choice((5, 6))
    arm_len = rng.choice((3, 4))
    leg_len = rng.choice((4, 5))
    stride_deg = rng.choice((-15.0, -7.5, 0.0, 11.25, 15.0))
    head_yaw = rng.choice((-15.0, 0.0, 15.0))
    armor_inf = round(rng.choice((0.0, 0.04, 0.08)), 3)

    head_bone = Bone(
        "cervical_head",
        rest_euler=(0.0, head_yaw, 0.0),
        attach=Attach("thorax_cuirass", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("neck_gorget", (2, 1, 2), p_cloth.name),
            Box("cranium_helm", (4, 4, 4), p_skin.name, Place("neck_gorget", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=armor_inf),
            Box("visor_faceplate", (4, 2, 1), p_visor.name, Place("cranium_helm", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.25))),
            Box("helm_plume_crest", (1, 2, 3), p_cloth.name, Place("cranium_helm", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    arm_bone = Bone(
        "shoulder_arm_l",
        rest_euler=(-stride_deg, 0.0, 11.25),
        attach=Attach("thorax_cuirass", (1.0, 0.65, 0.0), (-1.0, 0.5, 0.0)),
        mirror_name="shoulder_arm_r",
        items=[
            Box("pauldron_cap", (2, 2, 3), p_primary.name, inflate=armor_inf),
            Box("upper_brachium", (2, arm_len, 2), p_cloth.name, Place("pauldron_cap", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Bone(
                "elbow_forearm_l",
                rest_euler=(-15.0, 0.0, 0.0),
                attach=Attach("upper_brachium", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                items=[
                    Box("vambrace_forearm", (2, arm_len, 2), p_primary.name),
                    Box("gauntlet_fist", (2, 2, 2), p_skin.name, Place("vambrace_forearm", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    leg_bone = Bone(
        "hip_thigh_l",
        rest_euler=(stride_deg, 0.0, 5.0),
        attach=Attach("pelvic_belt", (0.65, -1.0, 0.0), (0.0, 1.0, 0.0)),
        mirror_name="hip_thigh_r",
        items=[
            Box("femur_cuisse", (2, leg_len, 2), p_cloth.name),
            Bone(
                "knee_shin_l",
                rest_euler=(11.25, 0.0, 0.0),
                attach=Attach("femur_cuisse", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                items=[
                    Box("greave_shin", (2, leg_len, 2), p_primary.name),
                    Box("sabaton_boot", (2, 2, 3), p_primary.name, Place("greave_shin", (0.0, -1.0, -0.25), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("pelvic_belt", (5, 2, 3), p_cloth.name),
        Box("thorax_cuirass", (6, torso_h, 3), p_primary.name, Place("pelvic_belt", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=armor_inf),
        Box("dorsal_pack_unit", (4, 4, 2), p_cloth.name, Place("thorax_cuirass", (0.0, 0.1, 1.0), (0.0, 0.0, -1.0))),
        head_bone,
        arm_bone,
        leg_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_primary, p_cloth, p_skin, p_visor],
        root=Bone("biped_pelvis_root", pivot_local=_jpiv(rng, (8.0, 14.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 2 + torso_h + 2 * leg_len + 7,
        "secondary_span": 12,
        "joint_angle_deg": stride_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": armor_inf,
        "material_theme": f"{p_primary.name}+{p_cloth.name}+{p_skin.name}",
        "notes": f"articulated bipedal humanoid ({substyle.replace('_', ' ')}) with pelvis, {torso_h}-tall cuirass + dorsal pack, yawed visored head, mirrored 2-joint arms, and mirrored 2-joint striding legs",
        "extra": {"torso_h": torso_h, "leg_len": leg_len, "stride_deg": stride_deg},
    }


def _gen_dog_horse_quadruped(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("dire_timber_wolf", "barded_courser_horse", "prowling_saber_tiger", "crowned_forest_stag", "mountain_grizzly_bear")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)

    p_coat = _pal(rng, _CREATURE_HIDES_SCALES[v_idx % 4])
    p_belly = _pal(rng, _HORN_BONE_ACCENTS[3])
    p_hoof = _pal(rng, _HORN_BONE_ACCENTS[0 if v_idx % 5 == 3 else 1])
    p_eye = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=(v_idx % 5 == 0))

    body_len = rng.choice((7, 8, 9))
    leg_len = rng.choice((4, 5))
    neck_pitch = rng.choice((-15.0, -11.25))
    gait_deg = rng.choice((0.0, 7.5, 11.25))
    coat_inf = round(rng.choice((0.0, 0.04, 0.07)), 3)

    ear_tine_h = 4 if substyle == "crowned_forest_stag" else 2
    ear_bone = Bone(
        "pinna_or_antler_l",
        rest_euler=(0.0, 0.0, -15.0),
        attach=Attach("cranial_skull", (0.75, 1.0, 0.25), (0.0, -1.0, 0.0)),
        mirror_name="pinna_or_antler_r",
        items=[Box("ear_or_tine", (1, ear_tine_h, 1), p_hoof.name)],
    )

    head_neck_bone = Bone(
        "cervical_neck_rig",
        rest_euler=(neck_pitch, 0.0, 0.0),
        attach=Attach("fore_ribcage", (0.0, 1.0, -0.55), (0.0, -1.0, 0.0)),
        items=[
            Box("muscular_neck", (3, 4, 3), p_coat.name),
            Box("cranial_skull", (4, 3, 4), p_coat.name, Place("muscular_neck", (0.0, 1.0, -0.35), (0.0, -1.0, 0.0))),
            Box("snout_muzzle", (2, 2, 3), p_belly.name, Place("cranial_skull", (0.0, -0.33, -1.0), (0.0, 0.0, 1.0))),
            Box("brow_eye_ridge", (4, 1, 1), p_eye.name, Place("cranial_skull", (0.0, 1.0, -0.75), (0.0, -1.0, 0.0))),
            ear_bone,
        ],
    )

    foreleg_bone = Bone(
        "fore_limb_l",
        rest_euler=(gait_deg, 0.0, 5.0),
        attach=Attach("fore_ribcage", (0.75, -1.0, -0.55), (0.0, 1.0, 0.0)),
        mirror_name="fore_limb_r",
        items=[
            Box("scapula_brachium", (2, leg_len, 2), p_coat.name),
            Bone(
                "fore_cannon_l",
                rest_euler=(-7.5, 0.0, 0.0),
                attach=Attach("scapula_brachium", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                items=[
                    Box("metacarpal_shank", (2, leg_len - 1, 2), p_belly.name),
                    Box("fore_paw_hoof", (2, 1, 3), p_hoof.name, Place("metacarpal_shank", (0.0, -1.0, -0.2), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    hindleg_bone = Bone(
        "hind_limb_l",
        rest_euler=(-gait_deg, 0.0, 5.0),
        attach=Attach("lumbar_croup", (0.75, -1.0, 0.55), (0.0, 1.0, 0.0)),
        mirror_name="hind_limb_r",
        items=[
            Box("femur_haunch", (2, leg_len, 3), p_coat.name),
            Bone(
                "hind_hock_l",
                rest_euler=(11.25, 0.0, 0.0),
                attach=Attach("femur_haunch", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                items=[
                    Box("tibia_cannon", (2, leg_len - 1, 2), p_belly.name),
                    Box("hind_paw_hoof", (2, 1, 3), p_hoof.name, Place("tibia_cannon", (0.0, -1.0, -0.2), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    tail_bone = Bone(
        "caudal_tail",
        rest_euler=(-22.5, 0.0, 0.0),
        attach=Attach("lumbar_croup", (0.0, 0.4, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("tail_dock", (2, 2, 4), p_coat.name),
            Box("tail_brush_plume", (2, 2, 3), p_belly.name, Place("tail_dock", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("fore_ribcage", (6, 5, 6), p_coat.name, inflate=coat_inf),
        Box("lumbar_croup", (5, 4, body_len - 1), p_coat.name, Place("fore_ribcage", (0.0, 0.1, 1.0), (0.0, 0.0, -1.0))),
        head_neck_bone,
        foreleg_bone,
        hindleg_bone,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_coat, p_belly, p_hoof, p_eye],
        root=Bone("quadruped_thorax_root", pivot_local=_jpiv(rng, (8.0, 13.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": body_len + 10,
        "secondary_span": 5 + 2 * leg_len,
        "joint_angle_deg": neck_pitch,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": coat_inf,
        "material_theme": f"{p_coat.name}+{p_belly.name}+{p_hoof.name}",
        "notes": f"quadrupedal mammal ({substyle.replace('_', ' ')}) with deep ribcage + croup, {neck_pitch:g} deg neck/skull/muzzle + mirrored ears/antlers, 2-joint fore & hind limbs, and plumed tail",
        "extra": {"body_len": body_len, "leg_len": leg_len},
    }


def _gen_bird(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("soaring_golden_eagle", "great_horned_owl", "marsh_crowned_heron", "radiant_crested_phoenix")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)

    p_plumage = _pal(rng, _CREATURE_HIDES_SCALES[(v_idx + 2) % len(_CREATURE_HIDES_SCALES)])
    p_breast = _pal(rng, _HORN_BONE_ACCENTS[3])
    p_beak = _pal(rng, _HORN_BONE_ACCENTS[2])
    is_phoenix = (substyle == "radiant_crested_phoenix")
    p_crest = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=is_phoenix)

    wing_in_w = rng.choice((5, 6, 7))
    wing_out_w = rng.choice((5, 6))
    dihedral_deg = rng.choice((11.25, 15.0, 22.5, -7.5))
    sweep_tip_deg = rng.choice((-15.0, -11.25, 11.25))
    feather_inf = round(rng.choice((0.0, 0.03, 0.06)), 3)

    head_bone = Bone(
        "avian_cervical_head",
        rest_euler=(-11.25, 0.0, 0.0),
        attach=Attach("keeled_sternum", (0.0, 1.0, -0.65), (0.0, -1.0, 0.0)),
        items=[
            Box("avian_neck", (2, 2, 2), p_breast.name),
            Box("feathered_cranium", (3, 3, 3), p_plumage.name, Place("avian_neck", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("hooked_rhamphotheca_beak", (1, 1, 3), p_beak.name, Place("feathered_cranium", (0.0, -0.1, -1.0), (0.0, 0.0, 1.0))),
            Box("occipital_crest_plume", (1, 2, 3), p_crest.name, Place("feathered_cranium", (0.0, 1.0, 0.3), (0.0, -1.0, 0.0))),
        ],
    )

    wing_bone = Bone(
        "humeral_wing_l",
        rest_euler=(0.0, 0.0, dihedral_deg),
        attach=Attach("keeled_sternum", (1.0, 0.55, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="humeral_wing_r",
        items=[
            Box("inner_wing_covert", (wing_in_w, 1, 5), p_plumage.name),
            Box("secondary_feather_trailing", (wing_in_w - 1, 1, 2), p_crest.name, Place("inner_wing_covert", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "carpal_pinion_l",
                rest_euler=(0.0, sweep_tip_deg, -11.25),
                attach=Attach("inner_wing_covert", (1.0, 0.0, -0.2), (-1.0, 0.0, 0.0)),
                items=[
                    Box("primary_remiges_panel", (wing_out_w, 1, 4), p_plumage.name),
                    Box("alula_wingtip_blade", (3, 1, 2), p_crest.name, Place("primary_remiges_panel", (1.0, 0.0, -0.25), (-1.0, 0.0, 0.0))),
                ],
            ),
        ],
    )

    talon_leg = Bone(
        "avian_tarsus_l",
        rest_euler=(11.25, 0.0, 5.0),
        attach=Attach("keeled_sternum", (0.6, -1.0, 0.1), (0.0, 1.0, 0.0)),
        mirror_name="avian_tarsus_r",
        items=[
            Box("feathered_tibiotarsus", (2, 3, 2), p_plumage.name),
            Box("raptor_talon_foot", (2, 1, 3), p_beak.name, Place("feathered_tibiotarsus", (0.0, -1.0, -0.25), (0.0, 1.0, 0.0))),
        ],
    )

    tail_fan = Bone(
        "rectrices_tail_fan",
        rest_euler=(-15.0, 0.0, 0.0),
        attach=Attach("uropygium_rump", (0.0, 0.2, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("central_rectrix_fan", (4, 1, 5), p_plumage.name),
            Box("outer_tail_streamer", (6, 1, 2), p_crest.name, Place("central_rectrix_fan", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("keeled_sternum", (5, 4, 6), p_breast.name, inflate=feather_inf),
        Box("dorsal_mantle_saddle", (4, 1, 3), p_plumage.name, Place("keeled_sternum", (0.0, 1.0, 0.45), (0.0, -1.0, 0.0))),
        Box("uropygium_rump", (3, 3, 3), p_plumage.name, Place("keeled_sternum", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        head_bone,
        wing_bone,
        talon_leg,
        tail_fan,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_plumage, p_breast, p_beak, p_crest],
        root=Bone("avian_keel_root", pivot_local=_jpiv(rng, (8.0, 12.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 5 + 2 * (wing_in_w + wing_out_w + 3),
        "secondary_span": 16,
        "joint_angle_deg": dihedral_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": feather_inf,
        "material_theme": f"{p_plumage.name}+{p_breast.name}+{p_beak.name}",
        "notes": f"articulated avian rig ({substyle.replace('_', ' ')}) with keeled sternum, hooked beak & occipital crest, 2-stage mirrored wings at {dihedral_deg:g} deg dihedral, taloned legs, and spread rectrices tail fan",
        "extra": {"wing_in_w": wing_in_w, "wing_out_w": wing_out_w},
    }


def _gen_winged_western_dragon(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("crimson_fire_wyrm", "glacial_frost_drake", "emerald_forest_dragon", "obsidian_storm_wyrm")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)

    p_scale = _pal(rng, _CREATURE_HIDES_SCALES[4 + (v_idx % 4)])
    p_membrane = _pal(rng, _HORN_BONE_ACCENTS[3], suffix="_patagium")
    p_horn = _pal(rng, _HORN_BONE_ACCENTS[0])
    p_breath = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=True)

    wing_span_in = rng.choice((6, 7))
    wing_span_out = rng.choice((6, 7))
    wing_dihedral = rng.choice((22.5, 30.0))
    jaw_open_deg = rng.choice((11.25, 15.0, 22.5))
    scale_inf = round(rng.choice((0.0, 0.04, 0.08)), 3)

    horn_bone = Bone(
        "swept_dragon_horn_l",
        rest_euler=(22.5, 0.0, -15.0),
        attach=Attach("draconic_cranium", (0.75, 1.0, 0.3), (0.0, -1.0, 0.0)),
        mirror_name="swept_dragon_horn_r",
        items=[
            Box("horn_base_core", (1, 3, 1), p_horn.name),
            Box("horn_distal_tine", (1, 3, 1), p_horn.name, Place("horn_base_core", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    skull_bone = Bone(
        "draconic_skull_rig",
        rest_euler=(15.0, 0.0, 0.0),
        attach=Attach("cervical_arch", (0.0, 1.0, -0.35), (0.0, -1.0, 0.0)),
        items=[
            Box("draconic_cranium", (4, 3, 4), p_scale.name),
            Box("maxilla_snout", (3, 1, 4), p_scale.name, Place("draconic_cranium", (0.0, 0.55, -1.0), (0.0, 0.0, 1.0))),
            Box("gullet_flame_core", (1, 1, 2), p_breath.name, Place("maxilla_snout", (0.0, -1.0, 0.35), (0.0, 1.0, 0.0))),
            Bone(
                "hinged_mandible",
                rest_euler=(-jaw_open_deg, 0.0, 0.0),
                attach=Attach("draconic_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                items=[Box("lower_jaw_ramus", (3, 1, 4), p_membrane.name)],
            ),
            horn_bone,
        ],
    )

    neck_bone = Bone(
        "draconic_neck",
        rest_euler=(-15.0, 0.0, 0.0),
        attach=Attach("armored_thorax", (0.0, 1.0, -0.65), (0.0, -1.0, 0.0)),
        items=[
            Box("cervical_arch", (3, 5, 3), p_scale.name),
            Box("cervical_osteoderm", (1, 2, 1), p_horn.name, Place("cervical_arch", (0.0, 0.45, 1.0), (0.0, 0.0, -1.0))),
            skull_bone,
        ],
    )

    wing_bone = Bone(
        "dragon_wing_arm_l",
        rest_euler=(0.0, 0.0, wing_dihedral),
        attach=Attach("armored_thorax", (1.0, 0.75, -0.1), (-1.0, 0.0, 0.0)),
        mirror_name="dragon_wing_arm_r",
        items=[
            Box("chiropteran_spar_in", (wing_span_in, 1, 2), p_scale.name),
            Box("patagium_membrane_in", (wing_span_in - 1, 1, 5), p_membrane.name, Place("chiropteran_spar_in", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("thumb_alula_claw", (1, 1, 2), p_horn.name, Place("chiropteran_spar_in", (1.0, 0.0, -1.0), (-1.0, 0.0, 1.0))),
            Bone(
                "dragon_wing_pinion_l",
                rest_euler=(0.0, 0.0, -15.0),
                attach=Attach("chiropteran_spar_in", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                items=[
                    Box("chiropteran_spar_out", (wing_span_out, 1, 2), p_scale.name),
                    Box("patagium_membrane_out", (wing_span_out - 1, 1, 4), p_membrane.name, Place("chiropteran_spar_out", (0.15, 0.0, 1.0), (0.0, 0.0, -1.0))),
                ],
            ),
        ],
    )

    fore_limb = Bone(
        "drake_forelimb_l",
        rest_euler=(-11.25, 0.0, 7.5),
        attach=Attach("armored_thorax", (0.8, -1.0, -0.45), (0.0, 1.0, 0.0)),
        mirror_name="drake_forelimb_r",
        items=[
            Box("fore_humerus", (2, 4, 2), p_scale.name),
            Box("fore_talon_manus", (2, 1, 3), p_horn.name, Place("fore_humerus", (0.0, -1.0, -0.25), (0.0, 1.0, 0.0))),
        ],
    )

    hind_limb = Bone(
        "drake_hindlimb_l",
        rest_euler=(11.25, 0.0, 7.5),
        attach=Attach("pelvic_haunch_hull", (0.8, -1.0, 0.3), (0.0, 1.0, 0.0)),
        mirror_name="drake_hindlimb_r",
        items=[
            Box("hind_crus_thigh", (2, 5, 3), p_scale.name),
            Box("hind_talon_pes", (2, 1, 4), p_horn.name, Place("hind_crus_thigh", (0.0, -1.0, -0.25), (0.0, 1.0, 0.0))),
        ],
    )

    tail_bone = Bone(
        "caudal_whip_prox",
        rest_euler=(-11.25, 11.25, 0.0),
        attach=Attach("pelvic_haunch_hull", (0.0, 0.2, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("caudal_seg_prox", (3, 3, 5), p_scale.name),
            Bone(
                "caudal_whip_dist",
                rest_euler=(7.5, -15.0, 0.0),
                attach=Attach("caudal_seg_prox", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("caudal_seg_dist", (2, 2, 5), p_scale.name),
                    Box("caudal_spade_blade", (4, 1, 3), p_horn.name, Place("caudal_seg_dist", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("armored_thorax", (6, 5, 7), p_scale.name, inflate=scale_inf),
        Box("thoracic_dorsal_spine", (1, 2, 4), p_horn.name, Place("armored_thorax", (0.0, 1.0, 0.35), (0.0, -1.0, 0.0))),
        Box("pelvic_haunch_hull", (5, 4, 6), p_scale.name, Place("armored_thorax", (0.0, -0.1, 1.0), (0.0, 0.0, -1.0))),
        neck_bone,
        wing_bone,
        fore_limb,
        hind_limb,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_scale, p_membrane, p_horn, p_breath],
        root=Bone("dragon_sternum_root", pivot_local=_jpiv(rng, (8.0, 12.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 6 + 2 * (wing_span_in + wing_span_out),
        "secondary_span": 27,
        "joint_angle_deg": wing_dihedral,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": scale_inf,
        "material_theme": f"{p_scale.name}+{p_membrane.name}+{p_horn.name}",
        "notes": f"4-legged winged western dragon ({substyle.replace('_', ' ')}) with dorsal osteoderms, S-curved neck, jawed skull ({jaw_open_deg:g} deg) + swept horns, 2-stage bat-membrane wings, clawed limbs, and spade-tipped tail",
        "extra": {"wing_span_in": wing_span_in, "jaw_open_deg": jaw_open_deg},
    }


def _gen_hexapod_beetle(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Animal family (>= 20 boxes): armored 6-legged beetle with elytra & mandibles."""
    substyles = ("hercules_dynastes_beetle", "jewel_scarab_carapace", "stag_mandible_lucanid")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)

    p_elytra = _pal(rng, _CREATURE_HIDES_SCALES[8])
    p_chitin = _pal(rng, _CREATURE_HIDES_SCALES[7])
    p_horn = _pal(rng, _HORN_BONE_ACCENTS[1])
    p_eye = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=True)

    abdomen_len = rng.choice((7, 8))
    elytra_flare = rng.choice((11.25, 15.0, 22.5))
    horn_pitch = rng.choice((-22.5, -15.0))
    chitin_inf = round(rng.choice((0.0, 0.04, 0.07)), 3)

    mandible_bone = Bone(
        "pincer_mandible_l",
        rest_euler=(0.0, -15.0, 0.0),
        attach=Attach("cephalon_capsule", (0.75, -0.2, -1.0), (0.0, 0.0, 1.0)),
        mirror_name="pincer_mandible_r",
        items=[
            Box("mandible_ramus", (1, 1, 3), p_horn.name),
            Box("mandible_tooth", (1, 1, 2), p_horn.name, Place("mandible_ramus", (-0.5, 0.0, -1.0), (0.0, 0.0, 1.0))),
        ],
    )

    head_bone = Bone(
        "insect_cephalon_rig",
        rest_euler=(horn_pitch, 0.0, 0.0),
        attach=Attach("pronotum_shield", (0.0, -0.1, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("cephalon_capsule", (4, 3, 3), p_chitin.name),
            Box("compound_eye_bar", (5, 1, 1), p_eye.name, Place("cephalon_capsule", (0.0, 0.2, -0.3), (0.0, 0.0, 0.0)))
            if False
            else Box("clypeal_upcurved_horn", (1, 2, 4), p_horn.name, Place("cephalon_capsule", (0.0, 0.6, -1.0), (0.0, 0.0, 1.0))),
            Box("ocular_tubercle", (4, 1, 1), p_eye.name, Place("cephalon_capsule", (0.0, 1.0, -0.5), (0.0, -1.0, 0.0))),
            mandible_bone,
        ],
    )

    elytra_bone = Bone(
        "elytron_shell_l",
        rest_euler=(-7.5, elytra_flare, 0.0),
        attach=Attach("abdomen_sternite", (0.6, 1.0, -0.65), (0.0, -1.0, -1.0)),
        mirror_name="elytron_shell_r",
        items=[
            Box("elytron_carapace", (3, 2, abdomen_len - 1), p_elytra.name, inflate=chitin_inf),
            Box("elytron_apex_lip", (2, 1, 2), p_elytra.name, Place("elytron_carapace", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    def _make_hexapod_leg(name_l: str, name_r: str, parent_bx: str, z_anc: float, yaw_deg: float) -> Bone:
        return Bone(
            name_l,
            rest_euler=(0.0, yaw_deg, -22.5),
            attach=Attach(parent_bx, (1.0, -0.6, z_anc), (-1.0, 0.0, 0.0)),
            mirror_name=name_r,
            items=[
                Box(f"{name_l}_femur", (3, 1, 1), p_chitin.name),
                Box(f"{name_l}_tibia_claw", (1, 3, 1), p_horn.name, Place(f"{name_l}_femur", (1.0, -0.5, 0.0), (-1.0, 1.0, 0.0))),
            ],
        )

    root_items: list[Box | Bone] = [
        Box("pronotum_shield", (6, 4, 5), p_elytra.name, inflate=chitin_inf),
        Box("thoracic_dynastes_horn", (2, 2, 5), p_horn.name, Place("pronotum_shield", (0.0, 1.0, -0.7), (0.0, -1.0, 0.5))),
        Box("abdomen_sternite", (7, 4, abdomen_len), p_chitin.name, Place("pronotum_shield", (0.0, -0.1, 1.0), (0.0, 0.0, -1.0))),
        head_bone,
        elytra_bone,
        _make_hexapod_leg("prothoracic_leg_l", "prothoracic_leg_r", "pronotum_shield", -0.35, -22.5),
        _make_hexapod_leg("mesothoracic_leg_l", "mesothoracic_leg_r", "abdomen_sternite", -0.55, 0.0),
        _make_hexapod_leg("metathoracic_leg_l", "metathoracic_leg_r", "abdomen_sternite", 0.45, 22.5),
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_elytra, p_chitin, p_horn, p_eye],
        root=Bone("hexapod_thorax_root", pivot_local=_jpiv(rng, (8.0, 9.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 5 + abdomen_len + 6,
        "secondary_span": 15,
        "joint_angle_deg": elytra_flare,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": chitin_inf,
        "material_theme": f"{p_elytra.name}+{p_chitin.name}+{p_horn.name}",
        "notes": f"6-legged armored beetle ({substyle.replace('_', ' ')}) with thoracic horn, clypeal horn + mirrored mandibles, flared iridescent elytra, and 3 distinct mirrored leg pairs",
        "extra": {"abdomen_len": abdomen_len, "elytra_flare": elytra_flare},
    }


def _gen_heavy_tail_tyrannosaur(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Animal family (>= 20 boxes): bipedal theropod dinosaur with jawed skull & heavy tail."""
    substyles = ("apex_tyrannosaur_rex", "crested_allosaur_theropod", "armored_carnotaur_saurian")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)

    p_hide = _pal(rng, _CREATURE_HIDES_SCALES[(v_idx + 4) % 7])
    p_under = _pal(rng, _HORN_BONE_ACCENTS[3])
    p_claw = _pal(rng, _HORN_BONE_ACCENTS[0])
    p_eye = _pal(rng, rng.choice(_MINERAL_CORAL_PALETTES), emissive=False)

    tail_len = rng.choice((6, 7))
    jaw_deg = rng.choice((15.0, 22.5))
    stride_deg = rng.choice((-11.25, 0.0, 11.25))
    scute_inf = round(rng.choice((0.0, 0.04, 0.07)), 3)

    skull_bone = Bone(
        "theropod_skull_rig",
        rest_euler=(22.5, 0.0, 0.0),
        attach=Attach("s_curve_neck", (0.0, 1.0, -0.35), (0.0, -1.0, 0.0)),
        items=[
            Box("temporal_cranium", (4, 4, 4), p_hide.name),
            Box("deep_maxilla_rostrum", (3, 2, 5), p_hide.name, Place("temporal_cranium", (0.0, 0.45, -1.0), (0.0, 0.0, 1.0))),
            Box("lacrimal_brow_crests", (4, 1, 3), p_claw.name, Place("temporal_cranium", (0.0, 1.0, -0.4), (0.0, -1.0, 0.0))),
            Box("upper_tooth_row", (2, 1, 4), p_claw.name, Place("deep_maxilla_rostrum", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Bone(
                "theropod_dentary_jaw",
                rest_euler=(-jaw_deg, 0.0, 0.0),
                attach=Attach("temporal_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                items=[Box("lower_dentary_bar", (3, 1, 5), p_under.name)],
            ),
        ],
    )

    neck_bone = Bone(
        "theropod_cervical",
        rest_euler=(-22.5, 0.0, 0.0),
        attach=Attach("thoracic_rib_barrel", (0.0, 0.55, -1.0), (0.0, -1.0, 0.4)),
        items=[
            Box("s_curve_neck", (3, 4, 4), p_hide.name),
            skull_bone,
        ],
    )

    forearm_bone = Bone(
        "didactyl_forearm_l",
        rest_euler=(22.5, 0.0, 11.25),
        attach=Attach("thoracic_rib_barrel", (1.0, -0.55, -0.55), (-1.0, 1.0, 0.0)),
        mirror_name="didactyl_forearm_r",
        items=[
            Box("short_brachium", (1, 3, 1), p_hide.name),
            Box("didactyl_claws", (1, 1, 2), p_claw.name, Place("short_brachium", (0.0, -1.0, -0.5), (0.0, 1.0, 0.0))),
        ],
    )

    hindleg_bone = Bone(
        "theropod_thigh_l",
        rest_euler=(stride_deg, 0.0, 7.5),
        attach=Attach("sacral_pelvis", (1.0, -0.35, 0.0), (-1.0, 0.6, 0.0)),
        mirror_name="theropod_thigh_r",
        items=[
            Box("muscular_femur", (3, 5, 4), p_hide.name),
            Bone(
                "digitigrade_shank_l",
                rest_euler=(15.0, 0.0, 0.0),
                attach=Attach("muscular_femur", (0.0, -1.0, 0.2), (0.0, 1.0, 0.0)),
                items=[
                    Box("tibia_metatarsus", (2, 5, 2), p_under.name),
                    Box("tridactyl_pes_foot", (3, 1, 4), p_claw.name, Place("tibia_metatarsus", (0.0, -1.0, -0.4), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    tail_bone = Bone(
        "heavy_tail_prox",
        rest_euler=(0.0, 7.5, 0.0),
        attach=Attach("sacral_pelvis", (0.0, 0.25, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("caudal_muscle_base", (4, 4, tail_len), p_hide.name),
            Box("caudal_osteoderm_prox", (1, 1, tail_len - 1), p_eye.name, Place("caudal_muscle_base", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Bone(
                "heavy_tail_dist",
                rest_euler=(0.0, -11.25, 0.0),
                attach=Attach("caudal_muscle_base", (0.0, 0.1, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("caudal_whip_rod", (2, 2, tail_len), p_hide.name),
                    Box("caudal_tip_fin", (1, 2, 3), p_claw.name, Place("caudal_whip_rod", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("sacral_pelvis", (6, 5, 6), p_hide.name, inflate=scute_inf),
        Box("dorsal_pelvic_ridge", (2, 1, 5), p_eye.name, Place("sacral_pelvis", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("thoracic_rib_barrel", (6, 5, 6), p_hide.name, Place("sacral_pelvis", (0.0, 0.05, -1.0), (0.0, 0.0, 1.0))),
        Box("gastralia_belly_plate", (4, 1, 5), p_under.name, Place("thoracic_rib_barrel", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        neck_bone,
        forearm_bone,
        hindleg_bone,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_hide, p_under, p_claw, p_eye],
        root=Bone("theropod_pelvis_root", pivot_local=_jpiv(rng, (8.0, 14.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 12 + 2 * tail_len + 9,
        "secondary_span": 17,
        "joint_angle_deg": jaw_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": scute_inf,
        "material_theme": f"{p_hide.name}+{p_under.name}+{p_claw.name}",
        "notes": f"bipedal heavy-tailed theropod ({substyle.replace('_', ' ')}) with horizontal pelvis/ribcage, S-neck, deep toothy maxilla + {jaw_deg:g} deg mandible, compact didactyl arms, digitigrade hindlimbs, and 2-stage counterbalanced tail",
        "extra": {"tail_len": tail_len, "jaw_deg": jaw_deg},
    }


def _gen_segmented_sea_dragon(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Dragon family (>= 20 boxes): undulating multi-somite sea dragon leviathan."""
    substyles = ("abyssal_biolum_leviathan", "coral_frilled_sea_wyrm", "pelagic_serpent_drake")
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)

    p_scale = _pal(rng, _CREATURE_HIDES_SCALES[6])
    p_fin = _pal(rng, _MINERAL_CORAL_PALETTES[1], emissive=True)
    p_belly = _pal(rng, _HORN_BONE_ACCENTS[3])
    p_horn = _pal(rng, _HORN_BONE_ACCENTS[0])

    seg_len = rng.choice((5, 6))
    wave_deg = rng.choice((15.0, 22.5))
    flipper_span = rng.choice((5, 6))
    fin_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    gill_frill = Bone(
        "branchial_frill_l",
        rest_euler=(0.0, -22.5, 11.25),
        attach=Attach("leviathan_cranium", (1.0, 0.2, 0.2), (-1.0, 0.0, 0.0)),
        mirror_name="branchial_frill_r",
        items=[
            Box("frill_ray_spar", (3, 2, 1), p_fin.name),
            Box("frill_web_tip", (2, 3, 1), p_fin.name, Place("frill_ray_spar", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    head_bone = Bone(
        "leviathan_head_rig",
        rest_euler=(7.5, -wave_deg * 0.5, 0.0),
        attach=Attach("pectoral_somite", (0.0, 0.1, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("leviathan_cranium", (5, 4, 5), p_scale.name),
            Box("serrated_rostrum", (3, 2, 5), p_scale.name, Place("leviathan_cranium", (0.0, -0.2, -1.0), (0.0, 0.0, 1.0))),
            Box("angler_esca_lure", (1, 3, 3), p_fin.name, Place("leviathan_cranium", (0.0, 1.0, -0.6), (0.0, -1.0, 0.0))),
            gill_frill,
        ],
    )

    pectoral_flipper = Bone(
        "pectoral_paddle_l",
        rest_euler=(0.0, -15.0, -15.0),
        attach=Attach("pectoral_somite", (1.0, -0.35, -0.1), (-1.0, 0.0, 0.0)),
        mirror_name="pectoral_paddle_r",
        items=[
            Box("paddle_humeral_root", (3, 1, 3), p_scale.name),
            Box("paddle_hydrofoil_blade", (flipper_span, 1, 4), p_fin.name, Place("paddle_humeral_root", (1.0, 0.0, 0.2), (-1.0, 0.0, 0.0))),
        ],
    )

    pelvic_fin = Bone(
        "pelvic_fin_l",
        rest_euler=(0.0, -15.0, -15.0),
        attach=Attach("pelvic_somite", (1.0, -0.35, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="pelvic_fin_r",
        items=[Box("pelvic_fin_blade", (4, 1, 3), p_fin.name)],
    )

    caudal_bone = Bone(
        "caudal_fluke_stage",
        rest_euler=(0.0, wave_deg, 0.0),
        attach=Attach("pelvic_somite", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("caudal_peduncle", (2, 2, seg_len), p_scale.name),
            Box("lunate_fluke_upper", (1, 4, 4), p_fin.name, Place("caudal_peduncle", (0.0, 1.0, 0.8), (0.0, -1.0, -0.5))),
            Box("lunate_fluke_lower", (1, 3, 3), p_fin.name, Place("caudal_peduncle", (0.0, -1.0, 0.8), (0.0, 1.0, -0.5))),
        ],
    )

    pelvic_bone = Bone(
        "pelvic_somite_stage",
        rest_euler=(0.0, -wave_deg, 0.0),
        attach=Attach("mid_trunk_somite", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("pelvic_somite", (4, 3, seg_len), p_scale.name),
            Box("pelvic_dorsal_sail", (1, 3, seg_len - 1), p_fin.name, Place("pelvic_somite", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            pelvic_fin,
            caudal_bone,
        ],
    )

    mid_bone = Bone(
        "mid_somite_stage",
        rest_euler=(0.0, wave_deg, 0.0),
        attach=Attach("pectoral_somite", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("mid_trunk_somite", (5, 4, seg_len), p_scale.name),
            Box("mid_dorsal_sail", (1, 3, seg_len - 1), p_fin.name, Place("mid_trunk_somite", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("ventral_biolum_keel", (2, 1, seg_len - 2), p_belly.name, Place("mid_trunk_somite", (0.0, -1.0, 0.25), (0.0, 1.0, 0.0))),
            pelvic_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("pectoral_somite", (6, 5, seg_len), p_scale.name, inflate=fin_inf),
        Box("anterior_dorsal_sail", (1, 4, seg_len - 1), p_fin.name, Place("pectoral_somite", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("throat_plastron_plate", (4, 1, seg_len - 1), p_horn.name, Place("pectoral_somite", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        head_bone,
        pectoral_flipper,
        mid_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_scale, p_fin, p_belly, p_horn],
        root=Bone("sea_dragon_anterior_root", pivot_local=_jpiv(rng, (8.0, 10.0, 6.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 4 * seg_len + 14,
        "secondary_span": 6 + 2 * (flipper_span + 3),
        "joint_angle_deg": wave_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": fin_inf,
        "material_theme": f"{p_scale.name}+{p_fin.name}+{p_belly.name}",
        "notes": f"4-somite S-undulating sea dragon ({substyle.replace('_', ' ')}, +-{wave_deg:g} deg yaw wave) with bioluminescent angler lure, mirrored gill frills, hydrofoil pectoral paddles, pelvic fins, dorsal sails, and lunate caudal fluke",
        "extra": {"seg_len": seg_len, "wave_deg": wave_deg, "flipper_span": flipper_span},
    }


MULTI_DOMAIN_DISPATCH: dict[str, Any] = {
    # Seen (12 new)
    "ancient_canopy_tree": _gen_ancient_canopy_tree,
    "crystalline_geode_outcrop": _gen_crystalline_geode_outcrop,
    "coral_fungal_colony": _gen_coral_fungal_colony,
    "articulated_robotic_arm": _gen_articulated_robotic_arm,
    "industrial_turbine_generator": _gen_industrial_turbine_generator,
    "modular_timber_stone_building": _gen_modular_timber_stone_building,
    "tiered_pagoda_watchtower": _gen_tiered_pagoda_watchtower,
    "arched_stone_bridge_gate": _gen_arched_stone_bridge_gate,
    "humanoid_biped": _gen_humanoid_biped,
    "dog_horse_quadruped": _gen_dog_horse_quadruped,
    "bird": _gen_bird,
    "winged_western_dragon": _gen_winged_western_dragon,
    # Unseen (6 new)
    "weeping_bonsai_arch": _gen_weeping_bonsai_arch,
    "hydraulic_gantry_crane": _gen_hydraulic_gantry_crane,
    "coastal_lighthouse_keep": _gen_coastal_lighthouse_keep,
    "hexapod_beetle": _gen_hexapod_beetle,
    "heavy_tail_tyrannosaur": _gen_heavy_tail_tyrannosaur,
    "segmented_sea_dragon": _gen_segmented_sea_dragon,
}
