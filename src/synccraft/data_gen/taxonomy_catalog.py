"""Hierarchical Categorization & 55-Family Taxonomy Catalog for SyncCraft v3.0 (ADR 0005).

Organizes 40 Seen families (train/val) and 15 Held-Out Unseen families (test_unseen)
across 7 macro-domains and structured mechanical/biological categories, linking each
family to its recommended physical/Turing material substrates and motion archetype.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TaxonomyEntry:
    family_id: str
    split_group: str  # "seen" | "unseen"
    macro_domain: str
    category: str
    sub_category: str
    motion_archetype: str
    recommended_materials: tuple[str, ...]
    entity_body_plan: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


COMPLEXITY_TIERS: dict[str, tuple[int, int]] = {
    "tier_1_simple": (4, 8),
    "tier_2_medium": (9, 15),
    "tier_3_complex": (16, 24),
    "tier_4_masterwork": (25, 42),
}

# Explicit schema reservation for the real-human dataset currently being gathered
RESERVED_EXTENSION_TRACKS: dict[str, dict[str, Any]] = {
    "human_gathered_real_corpus": {
        "status": "gathering_external_permissive_assets_deferred",
        "macro_domain": "humanoid",
        "category": "curated_human_anatomy_and_apparel",
        "notes": "Reserved for second-stage human-based dataset after real human model collection completes.",
    }
}

TAXONOMY_ENTRIES: tuple[TaxonomyEntry, ...] = (
    # ==========================================================================
    # SEEN FAMILIES (40 total across 7 macro-domains)
    # ==========================================================================
    # 1. Artificial — Weapons, Craft, Horology, Artisan & Furnishing (9 Seen)
    TaxonomyEntry(
        "straight_blade_sword", "seen", "artificial", "weaponry_and_props", "double_edge_sword",
        "hinge_articulation", ("brushed_metal", "solid_wood", "woven_fabric"),
    ),
    TaxonomyEntry(
        "single_edge_blade", "seen", "artificial", "weaponry_and_props", "curved_saber_katana",
        "hinge_articulation", ("brushed_metal", "woven_fabric", "painted_metal"),
    ),
    TaxonomyEntry(
        "battle_axe", "seen", "artificial", "weaponry_and_props", "poll_and_bearded_axe",
        "hinge_articulation", ("brushed_metal", "solid_wood", "weathered_stone"),
    ),
    TaxonomyEntry(
        "spear", "seen", "artificial", "weaponry_and_props", "polearm_halberd",
        "hinge_articulation", ("brushed_metal", "solid_wood", "woven_fabric"),
    ),
    TaxonomyEntry(
        "war_hammer", "seen", "artificial", "weaponry_and_props", "maul_warhammer",
        "hinge_articulation", ("brushed_metal", "painted_metal", "solid_wood"),
    ),
    TaxonomyEntry(
        "simple_wood_bow", "seen", "artificial", "weaponry_and_props", "recurve_longbow",
        "hinge_articulation", ("solid_wood", "woven_fabric", "keratin_scales"),
    ),
    TaxonomyEntry(
        "furniture_table_chair_chest_cabinet", "seen", "artificial", "furnishing", "articulated_woodwork",
        "hinge_articulation", ("solid_wood", "woven_fabric", "brushed_metal"),
    ),
    TaxonomyEntry(
        "clockwork_pendulum_clock", "seen", "artificial", "horology_and_artisan", "grandfather_regulator_clock",
        "pendulum_oscillation", ("solid_wood", "brushed_metal", "thin_film_iridescence"),
    ),
    TaxonomyEntry(
        "alchemical_workbench_apparatus", "seen", "artificial", "horology_and_artisan", "distillation_assay_bench",
        "pendulum_oscillation", ("solid_wood", "polymer_dielectric", "brushed_metal"),
    ),
    # 2. Industrial — Mechanical Engineering, Automation & Vehicles (11 Seen)
    TaxonomyEntry(
        "gearbox", "seen", "industrial", "mechanical_power", "spur_bevel_transmission",
        "rotary_spin", ("brushed_metal", "painted_metal", "polymer_dielectric"),
    ),
    TaxonomyEntry(
        "articulated_robotic_arm", "seen", "industrial", "robotics_automation", "six_axis_manipulator",
        "hydraulic_boom_chain", ("painted_metal", "brushed_metal", "polymer_dielectric"),
    ),
    TaxonomyEntry(
        "industrial_turbine_generator", "seen", "industrial", "mechanical_power", "axial_power_turbine",
        "rotary_spin", ("brushed_metal", "painted_metal", "thin_film_iridescence"),
    ),
    TaxonomyEntry(
        "piston_crank_engine", "seen", "industrial", "mechanical_power", "reciprocating_ic_engine",
        "reciprocating_mechanism", ("brushed_metal", "painted_metal", "polymer_dielectric"),
    ),
    TaxonomyEntry(
        "cnc_milling_press", "seen", "industrial", "manufacturing_machinery", "gantry_spindle_machine_tool",
        "reciprocating_mechanism", ("painted_metal", "brushed_metal", "polymer_dielectric"),
    ),
    TaxonomyEntry(
        "centrifugal_pump_valve_skid", "seen", "industrial", "fluid_process_engineering", "volute_pump_manifold",
        "rotary_spin", ("painted_metal", "brushed_metal", "polymer_dielectric"),
    ),
    TaxonomyEntry(
        "wheeled_haul_truck", "seen", "industrial", "vehicle_land", "articulated_dump_truck",
        "vehicle_drive_and_articulate", ("painted_metal", "polymer_dielectric", "brushed_metal"),
    ),
    TaxonomyEntry(
        "tracked_excavator", "seen", "industrial", "vehicle_construction", "hydraulic_crawler_excavator",
        "hydraulic_boom_chain", ("painted_metal", "brushed_metal", "polymer_dielectric"),
    ),
    TaxonomyEntry(
        "steam_rail_locomotive", "seen", "industrial", "vehicle_rail", "steam_drive_locomotive",
        "vehicle_drive_and_articulate", ("painted_metal", "brushed_metal", "solid_wood"),
    ),
    TaxonomyEntry(
        "twin_rotor_helicopter_drone", "seen", "industrial", "vehicle_aerospace", "rotary_wing_aircraft",
        "rotary_spin", ("painted_metal", "polymer_dielectric", "brushed_metal"),
    ),
    TaxonomyEntry(
        "cargo_tugboat_vessel", "seen", "industrial", "vehicle_marine", "harbor_pusher_tugboat",
        "vehicle_drive_and_articulate", ("painted_metal", "solid_wood", "brushed_metal"),
    ),
    # 3. Natural — Botany, Geology, Mycology (4 Seen)
    TaxonomyEntry(
        "ancient_canopy_tree", "seen", "natural", "flora_arboreal", "broadleaf_canopy_tree",
        "wind_canopy_sway", ("solid_wood", "lichen_biofilm", "patterned_skin"),
    ),
    TaxonomyEntry(
        "crystalline_geode_outcrop", "seen", "natural", "geology_and_reef", "quartz_prism_geode",
        "pendulum_oscillation", ("weathered_stone", "thin_film_iridescence", "lichen_biofilm"),
    ),
    TaxonomyEntry(
        "coral_fungal_colony", "seen", "natural", "geology_and_reef", "reef_mycelial_terraces",
        "wind_canopy_sway", ("lichen_biofilm", "patterned_skin", "weathered_stone"),
    ),
    TaxonomyEntry(
        "pine_conifer_evergreen", "seen", "natural", "flora_arboreal", "whorled_alpine_conifer",
        "wind_canopy_sway", ("solid_wood", "lichen_biofilm", "weathered_stone"),
    ),
    # 4. Building / Architecture (4 Seen)
    TaxonomyEntry(
        "modular_timber_stone_building", "seen", "building", "vernacular_architecture", "half_timber_gable_hall",
        "hinge_articulation", ("weathered_stone", "solid_wood", "painted_metal"),
    ),
    TaxonomyEntry(
        "tiered_pagoda_watchtower", "seen", "building", "vernacular_architecture", "upturned_eave_pagoda",
        "hinge_articulation", ("solid_wood", "weathered_stone", "painted_metal"),
    ),
    TaxonomyEntry(
        "arched_stone_bridge_gate", "seen", "building", "civil_infrastructure", "voussoir_gate_bridge",
        "hinge_articulation", ("weathered_stone", "lichen_biofilm", "brushed_metal"),
    ),
    TaxonomyEntry(
        "waterwheel_millhouse", "seen", "building", "vernacular_architecture", "paddle_watermill_hall",
        "rotary_spin", ("solid_wood", "weathered_stone", "lichen_biofilm"),
    ),
    # 5. Humanoid (1 Seen — foundation rig; human_gathered_real_corpus reserved separately)
    TaxonomyEntry(
        "humanoid_biped", "seen", "humanoid", "bipedal_humanoid", "articulated_stylized_biped",
        "biped_stride", ("woven_fabric", "painted_metal", "polymer_dielectric"),
        entity_body_plan="humanoid_biped",
    ),
    # 6. Animal — Terrestrial, Avian & Aquatic Fauna (3 Seen)
    TaxonomyEntry(
        "dog_horse_quadruped", "seen", "animal", "terrestrial_and_avian_fauna", "cursorial_mammal_quadruped",
        "quadruped_stride", ("patterned_skin", "keratin_scales", "woven_fabric"),
        entity_body_plan="dog_horse_quadruped",
    ),
    TaxonomyEntry(
        "bird", "seen", "animal", "terrestrial_and_avian_fauna", "raptorial_passerine_avian",
        "wing_and_gait", ("keratin_scales", "patterned_skin", "thin_film_iridescence"),
        entity_body_plan="bird",
    ),
    TaxonomyEntry(
        "articulated_manta_shark", "seen", "animal", "aquatic_and_arthropod_fauna", "pelagic_ray_shark",
        "aquatic_undulation", ("patterned_skin", "keratin_scales", "thin_film_iridescence"),
        entity_body_plan="articulated_manta_shark",
    ),
    # 7. Dragon — Fancy Mythic & Prehistoric Draconic Morphologies (8 Seen)
    TaxonomyEntry(
        "winged_western_dragon", "seen", "dragon", "mythic_draconic", "hexapodal_winged_dragon",
        "wing_and_gait", ("keratin_scales", "patterned_skin", "thin_film_iridescence"),
        entity_body_plan="winged_western_dragon",
    ),
    TaxonomyEntry(
        "quadruped_armored_drake", "seen", "dragon", "mythic_draconic", "wingless_osteoderm_drake",
        "quadruped_stride", ("keratin_scales", "weathered_stone", "patterned_skin"),
        entity_body_plan="quadruped_armored_drake",
    ),
    TaxonomyEntry(
        "eastern_lung_dragon", "seen", "dragon", "mythic_draconic", "imperial_serpentine_lung",
        "serpentine_wave", ("keratin_scales", "thin_film_iridescence", "patterned_skin"),
        entity_body_plan="eastern_lung_dragon",
    ),
    TaxonomyEntry(
        "bipedal_wyvern", "seen", "dragon", "mythic_draconic", "forewing_stinger_wyvern",
        "wing_and_gait", ("keratin_scales", "patterned_skin", "thin_film_iridescence"),
        entity_body_plan="bipedal_wyvern",
    ),
    TaxonomyEntry(
        "forelimb_lindworm", "seen", "dragon", "mythic_draconic", "two_legged_serpent_lindworm",
        "serpentine_wave", ("keratin_scales", "patterned_skin", "lichen_biofilm"),
        entity_body_plan="forelimb_lindworm",
    ),
    TaxonomyEntry(
        "feathered_raptor", "seen", "dragon", "mythic_draconic", "sickle_claw_dromaeosaur_raptor",
        "biped_stride", ("patterned_skin", "keratin_scales", "thin_film_iridescence"),
        entity_body_plan="feathered_raptor",
    ),
    TaxonomyEntry(
        "winged_amphithere", "seen", "dragon", "mythic_draconic", "legless_feathered_amphithere",
        "serpentine_wave", ("thin_film_iridescence", "keratin_scales", "patterned_skin"),
        entity_body_plan="winged_amphithere",
    ),
    TaxonomyEntry(
        "multi_headed_hydra", "seen", "dragon", "mythic_draconic", "polycephalic_three_headed_hydra",
        "polycephalic_sway", ("keratin_scales", "patterned_skin", "lichen_biofilm"),
        entity_body_plan="multi_headed_hydra",
    ),

    # ==========================================================================
    # HELD-OUT UNSEEN FAMILIES (15 total, all >= 16 boxes for Gate G2)
    # ==========================================================================
    # Artificial (3 Unseen)
    TaxonomyEntry(
        "mechanical_crossbow", "unseen", "artificial", "weaponry_and_props", "windlass_arbalest",
        "hinge_articulation", ("solid_wood", "brushed_metal", "woven_fabric"),
    ),
    TaxonomyEntry(
        "hand_lantern", "unseen", "artificial", "weaponry_and_props", "gimbaled_beacon_lantern",
        "pendulum_oscillation", ("brushed_metal", "painted_metal", "thin_film_iridescence"),
    ),
    TaxonomyEntry(
        "floating_ring_staff", "unseen", "artificial", "weaponry_and_props", "orbital_halo_scepter",
        "pendulum_oscillation", ("solid_wood", "thin_film_iridescence", "brushed_metal"),
    ),
    # Natural (1 Unseen)
    TaxonomyEntry(
        "weeping_bonsai_arch", "unseen", "natural", "flora_arboreal", "cascade_bonsai_tray",
        "wind_canopy_sway", ("solid_wood", "lichen_biofilm", "weathered_stone"),
    ),
    # Industrial & Vehicles (5 Unseen)
    TaxonomyEntry(
        "tripod_telescope", "unseen", "industrial", "scientific_instrument", "equatorial_refractor",
        "hinge_articulation", ("brushed_metal", "solid_wood", "thin_film_iridescence"),
    ),
    TaxonomyEntry(
        "anvil_forge", "unseen", "industrial", "manufacturing_machinery", "blacksmith_hearth_anvil",
        "hinge_articulation", ("weathered_stone", "brushed_metal", "solid_wood"),
    ),
    TaxonomyEntry(
        "hydraulic_gantry_crane", "unseen", "industrial", "manufacturing_machinery", "portal_trolley_crane",
        "hydraulic_boom_chain", ("painted_metal", "brushed_metal", "polymer_dielectric"),
    ),
    TaxonomyEntry(
        "articulated_forklift_loader", "unseen", "industrial", "vehicle_land", "counterbalance_mast_forklift",
        "vehicle_drive_and_articulate", ("painted_metal", "polymer_dielectric", "brushed_metal"),
    ),
    TaxonomyEntry(
        "submersible_rov_craft", "unseen", "industrial", "vehicle_marine", "dual_manipulator_rov",
        "vehicle_drive_and_articulate", ("painted_metal", "polymer_dielectric", "thin_film_iridescence"),
    ),
    # Building (2 Unseen)
    TaxonomyEntry(
        "folding_windmill", "unseen", "building", "vernacular_architecture", "lattice_sail_smock_mill",
        "rotary_spin", ("solid_wood", "weathered_stone", "woven_fabric"),
    ),
    TaxonomyEntry(
        "coastal_lighthouse_keep", "unseen", "building", "civil_infrastructure", "fresnel_beacon_lighthouse",
        "rotary_spin", ("weathered_stone", "painted_metal", "thin_film_iridescence"),
    ),
    # Animal (2 Unseen)
    TaxonomyEntry(
        "hexapod_beetle", "unseen", "animal", "aquatic_and_arthropod_fauna", "horned_coleoptera_beetle",
        "quadruped_stride", ("thin_film_iridescence", "keratin_scales", "patterned_skin"),
        entity_body_plan="hexapod_beetle",
    ),
    TaxonomyEntry(
        "heavy_tail_tyrannosaur", "unseen", "animal", "terrestrial_and_avian_fauna", "apex_theropod_tyrannosaur",
        "biped_stride", ("keratin_scales", "patterned_skin", "weathered_stone"),
        entity_body_plan="heavy_tail_tyrannosaur",
    ),
    # Dragon (2 Unseen — Sea Dragon + Crowned Zilant)
    TaxonomyEntry(
        "segmented_sea_dragon", "unseen", "dragon", "mythic_draconic", "undulating_pelagic_leviathan",
        "serpentine_wave", ("thin_film_iridescence", "patterned_skin", "keratin_scales"),
        entity_body_plan="segmented_sea_dragon",
    ),
    TaxonomyEntry(
        "crowned_zilant", "unseen", "dragon", "mythic_draconic", "crowned_bipedal_serpent_zilant",
        "wing_and_gait", ("keratin_scales", "brushed_metal", "thin_film_iridescence"),
        entity_body_plan="crowned_zilant",
    ),
)

TAXONOMY_BY_FAMILY: dict[str, TaxonomyEntry] = {entry.family_id: entry for entry in TAXONOMY_ENTRIES}

CATALOG_SEEN_FAMILIES: tuple[str, ...] = tuple(
    e.family_id for e in TAXONOMY_ENTRIES if e.split_group == "seen"
)
CATALOG_UNSEEN_FAMILIES: tuple[str, ...] = tuple(
    e.family_id for e in TAXONOMY_ENTRIES if e.split_group == "unseen"
)
CATALOG_FAMILY_DOMAINS: dict[str, str] = {e.family_id: e.macro_domain for e in TAXONOMY_ENTRIES}
CATALOG_ENTITY_BODY_PLANS: dict[str, str] = {
    e.family_id: e.entity_body_plan
    for e in TAXONOMY_ENTRIES
    if e.entity_body_plan is not None
}
ALL_CATEGORIES: tuple[str, ...] = tuple(sorted({e.category for e in TAXONOMY_ENTRIES}))


def classify_complexity_tier(box_count: int) -> str:
    """Map lowered box count to progressive simple-to-complex curriculum tier."""
    if box_count <= 8:
        return "tier_1_simple"
    if box_count <= 15:
        return "tier_2_medium"
    if box_count <= 24:
        return "tier_3_complex"
    return "tier_4_masterwork"


def get_taxonomy_metadata(family_id: str, *, box_count: int) -> dict[str, Any]:
    """Return structured category + complexity tier metadata for a generated asset."""
    entry = TAXONOMY_BY_FAMILY.get(family_id)
    tier = classify_complexity_tier(box_count)
    if entry is None:
        return {
            "macro_domain": "artificial",
            "category": "general_prop",
            "sub_category": family_id,
            "motion_archetype": "hinge_articulation",
            "complexity_tier": tier,
        }
    return {
        "macro_domain": entry.macro_domain,
        "category": entry.category,
        "sub_category": entry.sub_category,
        "motion_archetype": entry.motion_archetype,
        "complexity_tier": tier,
    }
