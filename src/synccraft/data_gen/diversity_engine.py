"""Anti-overfitting diversity engine for SyncCraft v3.0 procedural 3D datasets (ADR 0004).

Implements five research-grounded mitigations against procedural template memorization:
1. Prime-base Halton low-discrepancy space-filling sampling across continuous/discrete axes.
2. Six-register bijective semantic identifier remapping + valid DAG topological AST permutation
   (preserving 100% identical right-handed SE(3) world OBB geometry).
3. Physically grounded @materials & @style stateful PBR / Gray-Scott RD / thin-film binding
   injection across all 10 SyncCraft surface families.
4. Text2CAD-inspired 4-level multi-granularity & multi-persona prompt paraphrasing
   (from zero-slug natural user requests up to full parametric CAD blueprints).
5. MAP-Elites-style Quality-Diversity (QD) behavioral descriptors and Shannon entropy metrics.
"""

from __future__ import annotations

from collections import Counter
import math
import random
import re
from typing import Any

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import (
    Attach,
    Bone,
    Box,
    Locator,
    MaterialDecl,
    Place,
    Program,
    StyleDecl,
    validate_program,
)
from synccraft.kinematics.se3_chain import Scene, compile_geometry

HALTON_PRIMES: tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

MACRO_DOMAINS: tuple[str, ...] = (
    "artificial",
    "natural",
    "industrial",
    "building",
    "humanoid",
    "animal",
    "dragon",
)

NAMING_REGISTERS: tuple[str, ...] = (
    "preserve_canonical",
    "anatomical_biomech",
    "craft_modeler",
    "cad_engineering",
    "architectural_botanical",
    "compact_symbolic",
)

# Domain -> human-readable noun phrases for natural-language prompts (zero internal slug leakage)
HUMAN_FAMILY_NOUNS: dict[str, tuple[str, ...]] = {
    "straight_blade_sword": ("knightly double-edged longsword", "cruciform arming broadsword", "fullered steel greatsword", "chivalric bastard sword"),
    "single_edge_blade": ("curved single-edged saber", "forged uchigatana katana", "heavy cavalry backsword", "single-bevel hunting messer"),
    "battle_axe": ("twin-bitted dwarven battle axe", "bearded war cleaver", "crescent halberd-axe", "spiked pollaxe"),
    "spear": ("winged boar spear", "socketed infantry pike", "leaf-bladed partisan polearm", "bannered ceremonial lance"),
    "war_hammer": ("flanged war hammer", "beaked lucerne maul", "heavy blacksmith battle-maul", "spiked horseman's pick-hammer"),
    "simple_wood_bow": ("laminated recurve hunting bow", "carved yew longbow", "reflex composite stave bow", "ranger's horn-tipped bow"),
    "furniture_table_chair_chest_cabinet": ("carved timber tavern furniture piece", "iron-bound storage chest or cabinet", "joiner's trestle table or armchair", "hinged oak reliquary coffer"),
    "ancient_canopy_tree": ("gnarled ancient canopy tree", "buttress-rooted woodland oak", "forking moss-draped forest tree", "twisted elder banyan tree"),
    "crystalline_geode_outcrop": ("fractured quartz crystal geode outcrop", "amethyst spire rock cluster", "prismatic mineral vein formation", "bioluminescent cavern crystal boulder"),
    "coral_fungal_colony": ("branching reef coral colony", "tiered bioluminescent shelf-mushroom cluster", "antler-coral & polyp formation", "deep-forest mycelial fruiting cluster"),
    "gearbox": ("exposed industrial spur-gear reducer", "clockwork bevel transmission unit", "multi-shaft mechanical gearbox", "meshed toothed gear train housing"),
    "articulated_robotic_arm": ("six-axis industrial robotic manipulator arm", "pedestal factory assembly robot", "articulated servo gripper arm", "heavy CNC pick-and-place manipulator"),
    "industrial_turbine_generator": ("bladed industrial gas-turbine generator", "ducted rotary dynamo unit", "heavy power-plant impeller housing", "flanged steam-turbine assembly"),
    "modular_timber_stone_building": ("half-timbered stone cottage building", "medieval guildhall workshop structure", "gabled masonry-and-beam house", "pitched-roof village dwelling"),
    "tiered_pagoda_watchtower": ("multi-tiered cantilevered wooden pagoda", "battered-stone frontier watchtower", "upturned-eave temple spire tower", "fortified timber-and-ashlar lookout tower"),
    "arched_stone_bridge_gate": ("voussoir arched stone bridge", "crenellated twin-pier gatehouse", "buttressed masonry canal span", "fortified stone archway passage"),
    "humanoid_biped": ("articulated humanoid character figure", "bipedal armored explorer or knight", "proportioned two-legged humanoid rig", "suited bipedal adventurer or automaton"),
    "dog_horse_quadruped": ("four-legged mammalian creature", "articulated wolf, hound, stag, or steed", "quadrupedal wild beast", "striding four-limbed animal"),
    "bird": ("feathered avian bird with articulated wings", "gliding raptor eagle or perched owl", "long-winged heron or songbird", "taloned bird of prey"),
    "winged_western_dragon": ("four-legged winged dragon", "horned fire drake with bat-like wing membranes", "scaled western wyrm with spined tail", "articulated mythical dragon"),
    "quadruped_armored_drake": ("wingless four-legged armored terrestrial drake", "osteoderm-plated canyon drake with clubbed tail", "heavy quadrupedal earth drake", "thorn-backed terrestrial war drake"),
    "eastern_lung_dragon": ("imperial wingless eastern lung dragon", "antlered serpentine shenlong cloud dragon", "whiskered celestial river dragon with dorsal mane", "four-clawed undulating eastern dragon"),
    "bipedal_wyvern": ("two-legged winged wyvern with barbed venom stinger", "raptorial cliff wyvern with bat-membrane forewings", "crested bipedal wyvern with scorpion-stinger tail", "taloned storm wyvern"),
    "forelimb_lindworm": ("two-legged serpentine lindworm wyrm", "wingless Nordic forelimb lindworm", "armored two-clawed serpent dragon", "coiling fore-limbed cavern lindworm"),
    "feathered_raptor": ("feathered dromaeosaurid sickle-claw raptor", "plumaged swift-striding theropod raptor", "wing-fanned predatory deinonychus raptor", "crested feathered hunting raptor"),
    "winged_amphithere": ("legless feathered-wing amphithere serpent", "winged plumed sky-serpent amphithere", "hooded avian-winged amphithere dragon", "soaring limbless feathered dragonet"),
    "multi_headed_hydra": ("three-headed polycephalic swamp hydra", "multi-necked regenerating serpent hydra", "triple-jawed venomous hydra", "armored three-headed mythic hydra"),
    "articulated_manta_shark": ("pelagic winged manta ray or shark", "undulating cartilaginous ocean swimmer", "pectoral-finned reef manta-shark", "deep-sea hydrofoil ray-shark"),
    "pine_conifer_evergreen": ("tiered alpine pine conifer evergreen", "whorled mountain spruce or cedar tree", "snow-shedding boreal conifer tree", "resinous taproot evergreen pine"),
    "waterwheel_millhouse": ("half-timbered stone waterwheel gristmill", "paddle-wheel riverside millhouse", "gabled masonry watermill with timber wheel", "brookside miller's hall and sluice wheel"),
    "clockwork_pendulum_clock": ("tall grandfather pendulum regulator clock", "waisted brass-bob horological floor clock", "carved oak clockwork longcase clock", "chiming escapement pendulum clock"),
    "alchemical_workbench_apparatus": ("alchemical distillation workbench and balance", "retort-and-condenser apothecary assay table", "scholar's furnace-and-flask laboratory bench", "artisan glassblowing and alchemy table"),
    "piston_crank_engine": ("multi-cylinder reciprocating piston-crank engine", "inline diesel block with flywheel and manifold", "exposed mechanical crankshaft-and-piston engine", "cast-iron industrial combustion engine"),
    "cnc_milling_press": ("portal gantry CNC vertical milling press", "precision machine-tool spindle and T-slot bed", "industrial 3-axis machining press", "automated workshop milling center"),
    "centrifugal_pump_valve_skid": ("flanged centrifugal volute pump and valve skid", "industrial fluid-process manifold and motor skid", "piped gate-valve and impeller pump unit", "chemical transfer pump assembly"),
    "wheeled_haul_truck": ("heavy articulated off-highway dump haul truck", "six-wheel mining tipper truck with hydraulic bed", "industrial quarry haulage truck", "cab-over articulated dump vehicle"),
    "tracked_excavator": ("hydraulic crawler excavator with boom-stick-bucket", "slewing-cab tracked earthmoving digger", "caterpillar-tread trenching excavator", "heavy construction backhoe shovel"),
    "steam_rail_locomotive": ("boiler-and-smokestack steam rail locomotive", "cowcatcher-fitted iron railway engine", "coupled-wheel vintage steam train engine", "cab-and-tender rail locomotive"),
    "twin_rotor_helicopter_drone": ("articulated rotorcraft helicopter or heavy drone", "skid-landed rotary-wing utility helicopter", "tail-boom or twin-rotor aerial vehicle", "search-and-rescue turbine helicopter"),
    "cargo_tugboat_vessel": ("pilothouse harbor cargo tugboat vessel", "fendered marine pusher tug with twin funnels", "rudder-and-propeller workboat hull", "coastal utility towboat"),
    # Additional multi-domain & complex evaluation families
    "mechanical_crossbow": ("windlass siege arbalest crossbow", "ratcheted steel-prod crossbow", "lever-cocked composite arbalest"),
    "hand_lantern": ("caged brass storm lantern", "glazed wrought-iron beacon lantern", "bail-handled oil lamp"),
    "floating_ring_staff": ("arcane scepter with levitating gyroscopic rings", "halo-crowned archmage pole-staff", "floating-orb ritual scepter"),
    "weeping_bonsai_arch": ("cascading root-over-rock weeping bonsai tree", "scholar-stone miniature arch pine", "tray-planted windswept bonsai"),
    "tripod_telescope": ("brass-tubed equatorial tripod telescope", "surveyor's alt-azimuth observatory refractor", "geared astronomical spyglass on tripod"),
    "anvil_forge": ("blacksmith horn anvil and hooded coal hearth", "smithy hearth with chimney and anvil block", "ironworker's hearth and London-pattern anvil"),
    "hydraulic_gantry_crane": ("twin-leg industrial gantry crane with hoist trolley", "hydraulic shipyard portal crane", "overhead beam girder crane"),
    "articulated_forklift_loader": ("counterbalanced mast forklift warehouse loader", "twin-tine hydraulic pallet lift vehicle", "compact industrial yard lift truck"),
    "submersible_rov_craft": ("tethered deep-sea submersible craft with twin manipulators", "syntactic-foam underwater research vehicle", "vectored-thruster marine inspection craft"),
    "folding_windmill": ("lattice-sailed timber smock windmill", "four-vane grain mill tower", "cantilevered wind-pump millhouse"),
    "coastal_lighthouse_keep": ("tapered masonry coastal lighthouse tower", "beacon headland keep with lantern cupola", "sea-wall lighthouse and keeper annex"),
    "hexapod_beetle": ("six-legged armored rhinoceros beetle", "iridescent-carapace six-limbed insect", "mandibled stag-or-hercules beetle"),
    "heavy_tail_tyrannosaur": ("bipedal heavy-tailed theropod dinosaur", "jawed theropod saurian with counterbalanced tail", "digitigrade predatory saurian"),
    "segmented_sea_dragon": ("undulating multi-segment pelagic dragon leviathan", "frilled serpentine ocean wyrm", "bioluminescent finned marine serpent"),
    "crowned_zilant": ("crowned bipedal winged serpent wyvern", "mythic heraldic winged wyrm with golden three-prong crest", "taloned winged serpent-dragon"),
    # Clean-room Apex musculoskeletal & interactive families
    "apex_flying_wyvern": ("large biomechanical flying wyvern with nested wing digits", "crested aerial wyvern with articulated patagium wings", "raptorial soaring wyvern with segmented tail"),
    "brute_theropod_wyvern": ("heavy bipedal theropod wyvern with dorsal ridge plates", "crushing-jawed terrestrial brute wyvern", "counterbalanced heavy-skulled saurian wyvern"),
    "pelagic_leviathan_wyvern": ("undulating pelagic sea leviathan with hydrofoil fins", "long-bodied abyssal marine wyvern", "frilled oceanic leviathan swimmer"),
    "fanged_thunder_wyvern": ("quadrupedal storm wyvern with dorsal capacitor spines", "muscular four-legged fulgur wyvern", "insulated-mane clawed thunder drake"),
    "armored_arthropod_colossus": ("multi-legged carapace arthropod colossus with pincers", "chitin-plated siege arachnid-crustacean", "heavy chelicerate carapace colossus"),
    "morphing_switch_blade_axe": ("transforming mechanical switch-blade pollaxe", "phial-charged folding cleaver-axe weapon", "hinged dual-mode mechanical war axe"),
    "elemental_companion_critter": ("stylized elemental companion creature with expressive ears", "small agile quadruped companion critter with tail plume", "bright-eyed stylized companion beast"),
    "modular_botanical_healing_station": ("modular botanical rejuvenation and canister station", "dual-cradle bio-botanical recovery console", "articulated botanical infusion station"),
    "elder_calamity_dragon": ("six-limbed calamity elder dragon with great wings", "four-legged dual-winged ancient obsidian dragon", "horned six-limbed cataclysm dragon"),
    "relic_automaton_guardian": ("ancient overgrown basalt clockwork automaton guardian", "cyclopean stone-and-bronze relic golem", "steam-flywheel monolith guardian construct"),
}

# Semantic stem synonym banks per register so remapped symbols stay meaningful yet diverse
_STEM_SYNONYMS: dict[str, dict[str, tuple[str, ...]]] = {
    "anatomical_biomech": {
        "root": ("soma_origin", "axial_pelvis", "core_centrum", "basal_frame"),
        "head": ("cranium", "cephalon", "skull_caps", "rostrum"),
        "jaw": ("mandible", "maxilla", "gnathos", "chin_plate"),
        "neck": ("cervical", "atlas_link", "nuchal_seg", "throat_col"),
        "torso": ("thorax", "sternum", "rib_hull", "soma_trunk"),
        "pelvis": ("pelvic_girdle", "sacrum", "ilium_base", "hip_yoke"),
        "arm": ("brachium", "humerus", "forelimb_up", "pectoral_limb"),
        "forearm": ("antebrachium", "ulna_seg", "radius_link", "distal_arm"),
        "leg": ("femur", "crural_up", "thigh_seg", "hind_prox"),
        "shin": ("tibia", "metatarsus", "shank_seg", "hind_dist"),
        "wing": ("patagium", "alar_span", "chiroptera_limb", "remiges_fan"),
        "tail": ("caudal", "urostyle", "coccyx_seg", "fluke_link"),
        "spine": ("vertebra", "neural_crest", "osteoderm", "dorsal_ridge"),
        "horn": ("cornu", "antler_tine", "brow_prong", "crest_horn"),
    },
    "craft_modeler": {
        "root": ("rig_anchor", "main_pivot", "center_group", "base_hub"),
        "head": ("head_block", "visage_box", "face_shell", "crown_piece"),
        "jaw": ("lower_chin", "mouth_hinge", "snout_jaw", "beak_low"),
        "neck": ("neck_post", "collar_joint", "throat_piece", "neck_bridge"),
        "torso": ("chest_block", "body_core", "trunk_hull", "mid_body"),
        "pelvis": ("waist_block", "hip_core", "lower_trunk", "belt_base"),
        "arm": ("upper_arm", "shoulder_limb", "bicep_block", "sleeve_up"),
        "forearm": ("lower_arm", "wrist_piece", "gauntlet_seg", "hand_limb"),
        "leg": ("upper_leg", "thigh_block", "haunch_piece", "hip_limb"),
        "shin": ("lower_leg", "calf_block", "boot_shank", "paw_strut"),
        "wing": ("wing_panel", "feather_blade", "pinion_arm", "sail_wing"),
        "tail": ("tail_piece", "plume_seg", "sweep_tail", "tip_frond"),
        "blade": ("edge_plate", "cutter_bar", "fuller_slab", "point_seg"),
        "grip": ("handle_core", "haft_wrap", "pommel_shaft", "tang_grip"),
    },
    "cad_engineering": {
        "root": ("datum_origin", "assy_base", "chassis_zero", "mount_frame"),
        "torso": ("primary_housing", "stator_shell", "main_enclosure", "frame_body"),
        "head": ("sensor_turret", "optic_pod", "head_unit", "top_module"),
        "arm": ("boom_proximal", "actuator_a", "cantilever_link", "jib_upper"),
        "forearm": ("boom_distal", "effector_arm", "wrist_stage", "gripper_link"),
        "leg": ("strut_column", "pilon_leg", "support_post", "bogie_strut"),
        "gear": ("pinion_wheel", "spur_disk", "rotor_hub", "drive_cog"),
        "shaft": ("spindle_rod", "axle_bar", "arbor_core", "mandrel_pin"),
        "wall": ("panel_bulkhead", "curtain_slab", "casing_plate", "web_girder"),
        "roof": ("canopy_deck", "truss_cap", "shroud_cover", "gable_plate"),
    },
    "architectural_botanical": {
        "root": ("plinth_footing", "taproot_crown", "bedrock_pad", "podium_base"),
        "torso": ("nave_core", "bole_trunk", "tower_shaft", "keep_chamber"),
        "wall": ("ashlar_course", "timber_bays", "masonry_pier", "facade_block"),
        "roof": ("cantilever_eave", "gable_slope", "crown_foliage", "cupola_cap"),
        "arm": ("bough_limb", "flying_buttress", "arch_rib", "frond_branch"),
        "head": ("lantern_belfry", "apical_crown", "spire_finial", "polyp_cap"),
        "crystal": ("quartz_prism", "geode_point", "mineral_lance", "facet_tower"),
    },
}


def halton_scalar(index: int, base: int) -> float:
    """Deterministic radical-inverse Halton sequence value in [0, 1)."""
    assert base >= 2
    result = 0.0
    f = 1.0 / base
    i = max(1, int(index))
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def halton_vector(index: int, dim: int = 8, jitter_rng: random.Random | None = None) -> tuple[float, ...]:
    """Space-filling Cranley-Patterson-rotated Halton point in [0, 1)^dim."""
    out: list[float] = []
    for d in range(dim):
        h = halton_scalar(index + 1, HALTON_PRIMES[d % len(HALTON_PRIMES)])
        if jitter_rng is not None:
            h = (h + jitter_rng.random() * 0.125) % 1.0
        out.append(h)
    return tuple(out)


def _infer_material_family_for_palette(
    pal_name: str,
    base_lch: tuple[float, float, float],
    emissive: bool,
    domain: str,
    h_u: float,
) -> tuple[str, float, float, float, float]:
    """Return (family, roughness, metalness, coating, ior) grounded in domain + palette optics."""
    low = pal_name.lower()
    lightness, chroma, hue = base_lch

    if emissive or any(k in low for k in ("crystal", "ember", "flame", "rune", "glow", "prism", "beacon", "spore")):
        if h_u < 0.55:
            return ("thin_film_iridescence", round(0.08 + 0.14 * h_u, 3), 0.0, 0.35, round(1.55 + 0.35 * h_u, 3))
        return ("polymer_dielectric", round(0.12 + 0.18 * h_u, 3), 0.0, 0.25, 1.52)

    if domain in ("animal", "dragon"):
        if any(k in low for k in ("scale", "horn", "claw", "shell", "carapace", "beak", "elytra")):
            if "beetle" in low or "elytra" in low or h_u > 0.82:
                return ("thin_film_iridescence", round(0.14 + 0.16 * h_u, 3), 0.05, 0.45, 1.62)
            return ("keratin_scales", round(0.24 + 0.26 * h_u, 3), 0.0, 0.20, 1.54)
        return ("patterned_skin", round(0.42 + 0.30 * h_u, 3), 0.0, 0.0, 1.44)

    if domain == "humanoid":
        if any(k in low for k in ("steel", "iron", "brass", "bronze", "alloy", "visor", "plate", "mech")):
            fam = "painted_metal" if h_u < 0.5 else "brushed_metal"
            return (fam, round(0.18 + 0.24 * h_u, 3), round(0.75 + 0.25 * h_u, 3), 0.45 if fam == "painted_metal" else 0.05, 1.50)
        if any(k in low for k in ("skin", "hide", "flesh")):
            return ("polymer_dielectric", round(0.48 + 0.20 * h_u, 3), 0.0, 0.0, 1.42)
        return ("woven_fabric", round(0.62 + 0.26 * h_u, 3), 0.0, 0.0, 1.46)

    if domain == "natural":
        if any(k in low for k in ("moss", "leaf", "canopy", "lichen", "coral", "fung", "polyp", "frond", "needle")):
            return ("lichen_biofilm", round(0.55 + 0.30 * h_u, 3), 0.0, 0.0, 1.45)
        if any(k in low for k in ("wood", "oak", "walnut", "ash", "yew", "bark", "bole", "root")):
            return ("solid_wood", round(0.46 + 0.28 * h_u, 3), 0.0, 0.10, 1.48)
        return ("weathered_stone", round(0.58 + 0.32 * h_u, 3), 0.0, 0.0, 1.52)

    if domain == "building":
        if any(k in low for k in ("wood", "oak", "walnut", "timber", "rafter", "eave", "beam")):
            return ("solid_wood", round(0.44 + 0.28 * h_u, 3), 0.0, 0.15, 1.49)
        if any(k in low for k in ("moss", "tile_green", "patina")):
            return ("lichen_biofilm", round(0.52 + 0.28 * h_u, 3), 0.0, 0.05, 1.47)
        if any(k in low for k in ("brass", "iron", "copper", "steel", "bell")):
            return ("painted_metal", round(0.26 + 0.22 * h_u, 3), 0.85, 0.35, 1.50)
        return ("weathered_stone", round(0.60 + 0.30 * h_u, 3), 0.0, 0.0, 1.53)

    if domain == "industrial":
        if any(k in low for k in ("rubber", "polymer", "cable", "hose", "gasket")):
            return ("polymer_dielectric", round(0.38 + 0.30 * h_u, 3), 0.0, 0.10, 1.50)
        fam = "painted_metal" if (chroma > 0.08 or h_u < 0.55) else "brushed_metal"
        return (fam, round(0.16 + 0.28 * h_u, 3), round(0.80 + 0.20 * h_u, 3), 0.55 if fam == "painted_metal" else 0.05, 1.50)

    # Artificial (weapons, bows, furniture)
    if any(k in low for k in ("steel", "iron", "silver", "brass", "bronze", "copper", "electrum", "pewter")):
        fam = "brushed_metal" if h_u < 0.65 else "painted_metal"
        return (fam, round(0.15 + 0.25 * h_u, 3), round(0.82 + 0.18 * h_u, 3), 0.30 if fam == "painted_metal" else 0.0, 1.50)
    if any(k in low for k in ("wood", "oak", "walnut", "ash", "yew", "ebony")):
        return ("solid_wood", round(0.40 + 0.28 * h_u, 3), 0.0, 0.18, 1.49)
    if any(k in low for k in ("velvet", "canvas", "cordage", "leather", "cloth")):
        return ("woven_fabric", round(0.56 + 0.28 * h_u, 3), 0.0, 0.05, 1.46)
    if lightness > 0.65 and chroma < 0.04:
        return ("brushed_metal", 0.24, 0.9, 0.0, 1.5)
    return ("weathered_stone", round(0.55 + 0.28 * h_u, 3), 0.0, 0.0, 1.52)


def _remap_single_identifier(
    orig: str,
    register: str,
    rng: random.Random,
    seq_idx: int,
    kind: str,
) -> str:
    if register == "preserve_canonical":
        return orig

    low = orig.lower()
    side_suffix = ""
    if low.endswith("_l"):
        side_suffix = rng.choice(("_l", "_left", "_port", "_sin"))
        base = low[:-2]
    elif low.endswith("_r"):
        side_suffix = rng.choice(("_r", "_right", "_stbd", "_dex"))
        base = low[:-2]
    else:
        base = low

    if register == "compact_symbolic":
        prefix_map = {"palette": ("mat", "pal", "srf"), "bone": ("bn", "jnt", "grp"), "box": ("bx", "prt", "el")}
        pfx = rng.choice(prefix_map.get(kind, ("id",)))
        short_tag = re.sub(r"[^a-z0-9]", "", base)[:4] or "seg"
        return f"{pfx}_{short_tag}_{seq_idx:02d}{side_suffix}"

    bank = _STEM_SYNONYMS.get(register, {})
    replaced = base
    for key, choices in bank.items():
        if key in base:
            chosen = choices[(seq_idx + rng.randrange(len(choices))) % len(choices)]
            replaced = chosen
            # Preserve numeric suffix if original had one (e.g., seg_2)
            num_match = re.search(r"(\d+)$", base)
            if num_match:
                replaced = f"{replaced}_{num_match.group(1)}"
            break
    else:
        tag = rng.choice(("unit", "part", "elem", "seg", "blk", "node"))
        replaced = f"{base}_{tag}" if len(base) <= 12 else f"{base[:10]}_{seq_idx}"

    clean = re.sub(r"[^a-z0-9_]", "_", replaced).strip("_") or f"{kind}_{seq_idx}"
    if clean[0].isdigit():
        clean = f"p_{clean}"
    return f"{clean}{side_suffix}"


def _topological_permute_bone_items(
    items: list[Box | Locator | Bone], rng: random.Random
) -> list[Box | Locator | Bone]:
    """Randomize sibling Box/Locator/Bone order inside a Bone while respecting all base_box/parent_box edges."""
    if len(items) <= 2:
        return list(items)

    # First primary Box in a bone has place is None (or place.base_box is None) and MUST precede other Boxes!
    n = len(items)
    anchor_index_by_name: dict[str, int] = {}
    first_primary_box_idx: int | None = None
    for idx, it in enumerate(items):
        if isinstance(it, Box):
            anchor_index_by_name[it.name] = idx
            if (it.place is None or it.place.base_box is None) and first_primary_box_idx is None:
                first_primary_box_idx = idx
        elif isinstance(it, Locator):
            anchor_index_by_name[it.name] = idx

    deps: dict[int, set[int]] = {i: set() for i in range(n)}
    for idx, it in enumerate(items):
        if isinstance(it, Box):
            if first_primary_box_idx is not None and idx != first_primary_box_idx:
                deps[idx].add(first_primary_box_idx)
            if it.place is not None and it.place.base_box is not None and it.place.base_box in anchor_index_by_name:
                deps[idx].add(anchor_index_by_name[it.place.base_box])
        elif isinstance(it, Bone):
            if first_primary_box_idx is not None:
                deps[idx].add(first_primary_box_idx)
            if it.attach is not None and it.attach.parent_box in anchor_index_by_name:
                deps[idx].add(anchor_index_by_name[it.attach.parent_box])

    ready = [i for i in range(n) if not deps[i]]
    ordered_indices: list[int] = []
    placed: set[int] = set()

    while ready:
        if first_primary_box_idx is not None and first_primary_box_idx in ready:
            pick = first_primary_box_idx
            ready.remove(pick)
        else:
            pick = ready.pop(rng.randrange(len(ready)))
        ordered_indices.append(pick)
        placed.add(pick)
        for j in range(n):
            if j not in placed and j not in ready and deps[j].issubset(placed):
                ready.append(j)

    if len(ordered_indices) != n:
        return list(items)
    return [items[i] for i in ordered_indices]


def diversify_program_ast(
    program: Program,
    *,
    rng: random.Random,
    domain: str,
    variant_index: int,
) -> tuple[Program, dict[str, Any]]:
    """Apply register-varied bijective symbol remapping, valid DAG permutation, and @materials/@style binding."""
    hvec = halton_vector(variant_index, dim=8, jitter_rng=rng)

    # 1. Select naming register via stratified Halton coordinate (keep ~18% canonical, ~82% diversified across 5 registers)
    reg_idx = min(len(NAMING_REGISTERS) - 1, int(hvec[0] * len(NAMING_REGISTERS)))
    register = NAMING_REGISTERS[reg_idx]

    # Collect all original symbols and build collision-free bijective map (including mirror_name__descendant!)
    symbol_map: dict[str, str] = {}
    used_new: set[str] = set()
    seq_counter = 0

    def alloc_symbol(orig: str, kind: str, *, mirror_twin_of: str | None = None) -> str:
        nonlocal seq_counter
        if orig in symbol_map:
            return symbol_map[orig]
        seq_counter += 1
        if mirror_twin_of is not None and mirror_twin_of in symbol_map:
            twin_new = symbol_map[mirror_twin_of]
            for l_tok, r_tok in (("_left", "_right"), ("_port", "_stbd"), ("_sin", "_dex"), ("_l", "_r")):
                if twin_new.endswith(l_tok):
                    cand = twin_new[: -len(l_tok)] + r_tok
                    if cand not in used_new:
                        symbol_map[orig] = cand
                        used_new.add(cand)
                        return cand
            cand = f"{twin_new}_twin"
            if cand in used_new:
                cand = f"{twin_new}_r{seq_counter}"
            symbol_map[orig] = cand
            used_new.add(cand)
            return cand

        cand = _remap_single_identifier(orig, register, rng, seq_counter, kind)
        attempt = 0
        while cand in used_new:
            attempt += 1
            cand = f"{cand}_{attempt}"
        symbol_map[orig] = cand
        used_new.add(cand)
        return cand

    for pal in program.palettes:
        alloc_symbol(pal.name, "palette")

    def collect_bone(b: Bone) -> list[str]:
        b_new = alloc_symbol(b.name, "bone")
        m_new = alloc_symbol(b.mirror_name, "bone", mirror_twin_of=b.name) if b.mirror_name else None
        desc_new_names: list[str] = []
        for it in b.items:
            if isinstance(it, Box):
                bx_new = alloc_symbol(it.name, "box")
                desc_new_names.append(bx_new)
            elif isinstance(it, Locator):
                loc_new = alloc_symbol(it.name, "box")
                desc_new_names.append(loc_new)
            elif isinstance(it, Bone):
                sub_descs = collect_bone(it)
                desc_new_names.append(symbol_map[it.name])
                desc_new_names.extend(sub_descs)
        if m_new:
            for d_name in desc_new_names:
                used_new.add(f"{m_new}__{d_name}")
        _ = b_new
        return desc_new_names

    collect_bone(program.root)

    # 2. Rewrite Bone tree with remapped symbols + canonical root origin + randomized valid DAG item ordering
    permute_declarations = hvec[1] >= 0.20  # 80% of variants permute sibling/palette order

    def rewrite_bone(b: Bone, *, is_root: bool = False) -> Bone:
        rewritten_children: list[Box | Locator | Bone] = []
        for it in b.items:
            if isinstance(it, Box):
                new_place = None
                if it.place is not None:
                    new_base = symbol_map[it.place.base_box] if it.place.base_box is not None else None
                    new_place = Place(new_base, it.place.anchor, it.place.align, it.place.offset)
                rewritten_children.append(
                    Box(
                        name=symbol_map[it.name],
                        size=it.size,
                        material=symbol_map[it.material],
                        place=new_place,
                        inflate=it.inflate,
                        cullface=it.cullface,
                    )
                )
            elif isinstance(it, Locator):
                rewritten_children.append(
                    Locator(
                        name=symbol_map[it.name],
                        offset=it.offset,
                        role=it.role,
                    )
                )
            else:
                rewritten_children.append(rewrite_bone(it, is_root=False))

        if permute_declarations:
            rewritten_children = _topological_permute_bone_items(rewritten_children, rng)

        new_attach = None
        if b.attach is not None:
            new_attach = Attach(
                parent_box=symbol_map[b.attach.parent_box],
                anchor=b.attach.anchor,
                align=b.attach.align,
                offset=b.attach.offset,
                detached=b.attach.detached,
            )
        return Bone(
            name=symbol_map[b.name],
            pivot_local=(0.0, 0.0, 0.0) if is_root else b.pivot_local,
            rest_euler=b.rest_euler,
            attach=new_attach,
            mirror_name=symbol_map[b.mirror_name] if b.mirror_name else None,
            items=rewritten_children,
            joint_limits=b.joint_limits,
        )

    new_palettes = [
        type(p)(
            name=symbol_map[p.name],
            base_lch=p.base_lch,
            steps=p.steps,
            hue_shift=p.hue_shift,
            emissive=p.emissive,
        )
        for p in program.palettes
    ]
    if permute_declarations and len(new_palettes) > 1:
        rng.shuffle(new_palettes)

    # 3. Physically grounded @materials & @style injection with small canonical deterministic seeds (1..N)
    emit_surface_materials = (hvec[2] < 0.65) or bool(program.materials)
    new_materials: list[MaterialDecl] = []
    new_style: StyleDecl | None = program.style
    material_families_used: list[str] = []

    if emit_surface_materials:
        orig_by_new = {symbol_map[p.name]: p.name for p in program.palettes}
        for idx, pal in enumerate(new_palettes):
            orig_pname = orig_by_new.get(pal.name, pal.name)
            fam, rough, metal, coat, ior = _infer_material_family_for_palette(
                orig_pname,
                pal.base_lch,
                pal.emissive,
                domain,
                (hvec[(3 + idx) % len(hvec)] + 0.17 * idx) % 1.0,
            )
            scale = round(0.65 + 1.55 * ((hvec[(4 + idx) % len(hvec)]) % 1.0), 3)
            m_seed = int(idx + 1)
            new_materials.append(
                MaterialDecl(
                    name=pal.name,
                    family=fam,
                    roughness=rough,
                    metalness=metal,
                    coating=coat,
                    ior=ior,
                    scale=scale,
                    seed=m_seed,
                )
            )
            material_families_used.append(fam)
        if permute_declarations and len(new_materials) > 1:
            rng.shuffle(new_materials)

        if hvec[5] < 0.75:
            enc_choices = ("vanilla", "labpbr_1_3", "gltf_2_0")
            enc = enc_choices[int(hvec[6] * len(enc_choices)) % len(enc_choices)]
            new_style = StyleDecl(
                target_encoding=enc,
                quantize_ramp=(hvec[7] >= 0.18),
                bevel_highlight=round(0.05 + 0.22 * hvec[4], 3),
                view_deg=round(10.0 + 35.0 * hvec[5], 1),
            )

    diversified = Program(
        meta=program.meta,
        palettes=new_palettes,
        root=rewrite_bone(program.root, is_root=True),
        shade=program.shade,
        ergonomics=program.ergonomics,
        materials=tuple(new_materials),
        style=new_style,
    )

    # 4. Continuous rest-rotation broadening beyond the 24 discrete steps (SAT & envelope verified)
    if program.meta.target_profile != "java_legacy" and hvec[0] < 0.80:
        diversified = _broaden_continuous_rest_angles(diversified, rng=rng)

    info = {
        "naming_register": register,
        "declarations_permuted": permute_declarations,
        "has_materials_block": bool(new_materials),
        "material_families": sorted(set(material_families_used)),
        "style_encoding": new_style.target_encoding if new_style is not None else "implicit_vanilla",
    }
    return diversified, info


def _nudge_non_discrete_angle(val: float, rng: random.Random, max_mag: float) -> float:
    """Perturb an angle to a 0.1-deg continuous value not lying on 7.5-deg or 11.25-deg discrete steps."""
    sign = -1.0 if rng.random() < 0.5 else 1.0
    delta = sign * rng.uniform(0.3, max_mag)
    cand = round(val + delta, 1)
    if abs(cand) < 0.1 or abs(cand % 7.5) < 1e-6 or abs(cand % 11.25) < 1e-6:
        cand = round(cand + (0.3 * sign), 1)
    return float(max(-85.0, min(85.0, cand)))


def _check_diversified_bounds(program: Program, scene: Scene) -> bool:
    lo, hi = (-48.0, 64.0) if program.meta.archetype == "entity" else (-16.0, 32.0)
    for box in scene.boxes:
        pts = box.corners()
        if float(pts.min()) < lo - 1e-6 or float(pts.max()) > hi + 1e-6:
            return False
    return True


def _broaden_continuous_rest_angles(program: Program, *, rng: random.Random) -> Program:
    """Broaden child bone rest_euler angles onto continuous 0.1-deg values when SAT & P1-02 clean."""
    for max_mag in (2.8, 1.1):
        changed = False

        def perturb_bone(b: Bone, *, is_root: bool = False) -> Bone:
            nonlocal changed
            new_items: list[Box | Locator | Bone] = [
                perturb_bone(it, is_root=False) if isinstance(it, Bone) else it
                for it in b.items
            ]
            rx, ry, rz = b.rest_euler
            if not is_root:
                if abs(rx) > 1e-6 or abs(ry) > 1e-6 or abs(rz) > 1e-6:
                    nrx = _nudge_non_discrete_angle(rx, rng, max_mag) if abs(rx) > 1e-6 else 0.0
                    nry = _nudge_non_discrete_angle(ry, rng, max_mag) if abs(ry) > 1e-6 else 0.0
                    nrz = _nudge_non_discrete_angle(rz, rng, max_mag) if abs(rz) > 1e-6 else 0.0
                    changed = True
                    return Bone(
                        name=b.name,
                        pivot_local=b.pivot_local,
                        rest_euler=(nrx, nry, nrz),
                        attach=b.attach,
                        mirror_name=b.mirror_name,
                        items=new_items,
                        joint_limits=b.joint_limits,
                    )
                if b.attach is not None and b.attach.detached:
                    nry = _nudge_non_discrete_angle(0.0, rng, max_mag)
                    changed = True
                    return Bone(
                        name=b.name,
                        pivot_local=b.pivot_local,
                        rest_euler=(0.0, nry, 0.0),
                        attach=b.attach,
                        mirror_name=b.mirror_name,
                        items=new_items,
                        joint_limits=b.joint_limits,
                    )
            return Bone(
                name=b.name,
                pivot_local=b.pivot_local,
                rest_euler=b.rest_euler,
                attach=b.attach,
                mirror_name=b.mirror_name,
                items=new_items,
                joint_limits=b.joint_limits,
            )

        cand = Program(
            meta=program.meta,
            palettes=program.palettes,
            root=perturb_bone(program.root, is_root=True),
            shade=program.shade,
            ergonomics=program.ergonomics,
            materials=program.materials,
            style=program.style,
            muscles=program.muscles,
            kinematics=program.kinematics,
            behavior=program.behavior,
        )
        if not changed:
            return program
        try:
            validate_program(cand)
            sc = compile_geometry(cand)
            if not _check_diversified_bounds(cand, sc):
                continue
            aud = audit_scene(
                sc,
                collision_tolerance=1e-7,
                enforce_physical_contact=True,
                max_contact_gap=1e-4,
                max_inset_depth=0.5,
            )
            if aud["passed"]:
                return cand
        except Exception:
            continue
    return program


def _color_phrase_from_lch(lch: tuple[float, float, float], emissive: bool) -> str:
    lightness, chroma, hue = lch
    tone = "dark" if lightness < 0.36 else ("pale" if lightness > 0.78 else "mid-tone")
    if chroma < 0.032:
        hue_word = "slate-gray" if lightness < 0.55 else "silvery-ash"
    elif hue < 32 or hue >= 335:
        hue_word = "crimson-ochre"
    elif hue < 68:
        hue_word = "warm-amber"
    elif hue < 105:
        hue_word = "gilded-brass"
    elif hue < 168:
        hue_word = "moss-emerald"
    elif hue < 225:
        hue_word = "cerulean-teal"
    elif hue < 285:
        hue_word = "cobalt-steel"
    else:
        hue_word = "violet-plum"
    return f"glowing {hue_word}" if emissive else f"{tone} {hue_word}"


_FORBIDDEN_PROMPT_REGEX = re.compile(
    r"(?<![A-Za-z])(?:held[-\s_]*out|holdout|test_unseen_params|test_compositional|test_unseen|"
    r"unseen_complex_holdout|unseen|seen_train|validation|apricorn|cobblemon|monster[\s_-]*hunter|pok[eé]mon)(?![A-Za-z])",
    re.IGNORECASE,
)


def sanitize_prompt_no_leakage(prompt: str, family_id: str = "") -> str:
    """Enforce zero split-name, borrowed-IP, or raw snake_case family_id leakage in user prompts."""
    cleaned = _FORBIDDEN_PROMPT_REGEX.sub("", prompt)
    if family_id and "_" in family_id:
        nouns = HUMAN_FAMILY_NOUNS.get(family_id)
        replacement = nouns[0] if nouns else family_id.replace("_", " ")
        cleaned = re.sub(re.escape(family_id), replacement, cleaned, flags=re.IGNORECASE)
    for fam_key, fam_nouns in HUMAN_FAMILY_NOUNS.items():
        if "_" in fam_key and fam_key.lower() in cleaned.lower():
            cleaned = re.sub(re.escape(fam_key), fam_nouns[0], cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def extract_pinned_spec_summary(
    program: Program | None,
    *,
    box_count: int,
    bone_count: int,
    joint_angle_deg: float,
) -> dict[str, str]:
    """Extract exact pinned geometric dimensions and continuous joint angles from an AST."""
    if program is None:
        ang_str = f"[{joint_angle_deg:g}, 0.0, 0.0] deg" if abs(joint_angle_deg) > 1e-5 else "orthogonal 0.0 deg"
        return {
            "primary_dims": "proportional integer blocks",
            "key_angles": ang_str,
            "offset_clause": "",
        }

    all_boxes: list[Box] = []
    nonzero_angles: list[str] = []
    nonzero_offsets = 0
    detached_count = 0

    def _walk(b: Bone, is_root: bool = False) -> None:
        nonlocal nonzero_offsets, detached_count
        if not is_root:
            rx, ry, rz = b.rest_euler
            if abs(rx) > 1e-5 or abs(ry) > 1e-5 or abs(rz) > 1e-5:
                ang_txt = f"[{rx:g}, {ry:g}, {rz:g}]"
                if ang_txt not in nonzero_angles:
                    nonzero_angles.append(ang_txt)
            if b.attach is not None:
                ox, oy, oz = b.attach.offset
                if abs(ox) > 1e-5 or abs(oy) > 1e-5 or abs(oz) > 1e-5:
                    nonzero_offsets += 1
                if b.attach.detached:
                    detached_count += 1
        for it in b.items:
            if isinstance(it, Box):
                all_boxes.append(it)
                if it.place is not None:
                    px, py, pz = it.place.offset
                    if abs(px) > 1e-5 or abs(py) > 1e-5 or abs(pz) > 1e-5:
                        nonzero_offsets += 1
            elif isinstance(it, Bone):
                _walk(it, False)

    _walk(program.root, True)

    root_boxes = [it for it in program.root.items if isinstance(it, Box)]
    primary_sz = list(root_boxes[0].size) if root_boxes else (list(all_boxes[0].size) if all_boxes else [4, 4, 4])
    ranked_boxes = sorted(all_boxes[1:], key=lambda bx: -(bx.size[0] * bx.size[1] * bx.size[2]))
    sec_dims: list[str] = []
    for bx in ranked_boxes:
        s_txt = f"[{bx.size[0]}, {bx.size[1]}, {bx.size[2]}]"
        if s_txt not in sec_dims:
            sec_dims.append(s_txt)
        if len(sec_dims) >= 3:
            break

    dims_txt = f"primary mass [{primary_sz[0]}, {primary_sz[1]}, {primary_sz[2]}]"
    if sec_dims:
        dims_txt += f", secondary masses {', '.join(sec_dims)}"

    if nonzero_angles:
        angles_txt = ", ".join(nonzero_angles[:4]) + " deg"
    elif abs(joint_angle_deg) > 1e-5:
        angles_txt = f"{joint_angle_deg:g} deg"
    else:
        angles_txt = "0.0 deg orthogonal"

    extras: list[str] = []
    if nonzero_offsets > 0:
        extras.append(f"{nonzero_offsets} non-zero offset placement(s)")
    if detached_count > 0:
        extras.append(f"{detached_count} detached floating accessory sub-bone(s)")

    return {
        "primary_dims": dims_txt,
        "key_angles": angles_txt,
        "offset_clause": ("; " + ", ".join(extras)) if extras else "",
    }


def _humanize_token(token: str) -> str:
    return token.replace("_", " ").strip()


def compose_multilevel_prompt(
    rng: random.Random,
    *,
    variant_index: int,
    domain: str,
    family_id: str,
    substyle: str,
    archetype: str,
    target_profile: str,
    texel_density: int,
    box_count: int,
    bone_count: int,
    has_mirror_x: bool,
    has_detached: bool,
    joint_angle_deg: float,
    palettes: list[Any],
    material_families: list[str],
    style_encoding: str,
    structural_notes: str,
    program: Program | None = None,
) -> tuple[str, str]:
    """Synthesize zero-leakage 4-level multi-granularity & pinned-specification prompts."""
    h_level = halton_scalar(variant_index + 3, 5)
    level_idx = min(3, int(h_level * 4))
    nouns = HUMAN_FAMILY_NOUNS.get(family_id, (_humanize_token(substyle),))
    human_subject = nouns[(variant_index + rng.randrange(len(nouns))) % len(nouns)]
    sub_human = _humanize_token(substyle)
    clean_notes = sanitize_prompt_no_leakage(structural_notes.replace("_", " "), family_id)
    profile_human = _humanize_token(target_profile)
    archetype_human = _humanize_token(archetype)
    encoding_human = _humanize_token(style_encoding)

    color_phrases = [_color_phrase_from_lch(p.base_lch, p.emissive) for p in palettes[:4]]
    color_summary = ", ".join(color_phrases)
    mat_phys_summary = (
        ", ".join(_humanize_token(f) for f in material_families)
        if material_families
        else "clean pixel-ramped block surfaces"
    )
    pinned = extract_pinned_spec_summary(
        program,
        box_count=box_count,
        bone_count=bone_count,
        joint_angle_deg=joint_angle_deg,
    )

    if level_idx == 0:
        # Level 1: Natural human user intent (zero slugs, zero split labels)
        openers = (
            f"Create a 3D block model of a {human_subject}",
            f"Design an articulated {human_subject} ({sub_human} style)",
            f"Build a detailed multi-part {human_subject}",
            f"Model a collision-free {human_subject} in a {domain} theme",
        )
        pose_hint = (
            "with angled joints for a dynamic silhouette"
            if abs(joint_angle_deg) > 1e-5
            else "in a clean balanced stance"
        )
        sym_hint = "and mirrored left-right parts" if has_mirror_x else "with distinct directional features"
        det_hint = " plus a hovering detached accent element" if has_detached else ""
        prompt = (
            f"{rng.choice(openers)} {pose_hint} {sym_hint}{det_hint}. "
            f"Color it in {color_summary} tones featuring {clean_notes}."
        )
        return sanitize_prompt_no_leakage(prompt, family_id), "level_1_natural_intent"

    if level_idx == 1:
        # Level 2: Art direction & physical material-science brief
        prompt = (
            f"[{domain.upper()} ART & MATERIAL BRIEF] Craft a {sub_human} ({human_subject}) "
            f"using {mat_phys_summary} in a palette of {color_summary}. "
            f"Silhouette & assembly notes: {clean_notes}. "
            f"Target shader and export style: {encoding_human} at {texel_density}px/block texel density "
            f"for the {profile_human} runtime."
        )
        return sanitize_prompt_no_leakage(prompt, family_id), "level_2_art_and_material_brief"

    if level_idx == 2:
        # Level 3: Kinematic rig & runtime profile specification with pinned counts & dimensions
        sym_clause = "bilateral mirrored limb or wing expansion" if has_mirror_x else "asymmetric or cardinal sub-assemblies"
        det_clause = " with an authorized detached floating sub-bone" if has_detached else ""
        prompt = (
            f"Specify a kinematically valid Block-DSL v3.0 {archetype_human} rig ({domain} category: {human_subject}, "
            f"variant '{sub_human}') targeting the {profile_human} profile ({texel_density} texels/unit). "
            f"Target {box_count} boxes across {bone_count} bones ({pinned['primary_dims']}) using {sym_clause}{det_clause}, "
            f"key joint Euler angles {pinned['key_angles']}, and surface materials ({mat_phys_summary}). "
            f"Key sub-assemblies: {clean_notes}."
        )
        return sanitize_prompt_no_leakage(prompt, family_id), "level_3_kinematic_rig_spec"

    # Level 4: Pinned parametric CAD engineering specification (explicit box count, dimensions, and joint angles)
    sym_str = "bilateral mirrored symmetry enabled" if has_mirror_x else "unmirrored directional layout"
    det_str = ", includes detached floating sub-assembly" if has_detached else ""
    prompt = (
        f"[PINNED CAD SPECIFICATION] Domain: {domain} | Subject: {human_subject} ({sub_human})\n"
        f"Runtime target: {archetype_human} on {profile_human} ({texel_density}px/unit, {encoding_human} shading)\n"
        f"Pinned topology: {box_count} boxes across {bone_count} bones at canonical [0, 0, 0] root origin ({sym_str}{det_str})\n"
        f"Pinned dimensions & angles: {pinned['primary_dims']}; key joint angles {pinned['key_angles']}{pinned['offset_clause']}\n"
        f"Surface palette & physics: {len(palettes)} palettes ({color_summary}) with {mat_phys_summary}\n"
        f"Assembly details: {clean_notes}"
    )
    return sanitize_prompt_no_leakage(prompt, family_id), "level_4_parametric_cad_blueprint"


def compute_program_tree_shape(program: Program, scene: Scene | None = None) -> str:
    """Compute canonical unlabeled bone-and-box tree topology signature from lowered Scene.

    Each lowered bone is represented as ``(num_boxes_in_bone:[sorted_child_signatures])``
    so two programs share a signature iff their lowered bone-and-box trees are isomorphic.
    """
    sc = scene if scene is not None else compile_geometry(program)
    box_counts: dict[str, int] = Counter(b.bone_name for b in sc.boxes)
    children_by_parent: dict[str | None, list[str]] = {}
    root_name: str | None = None
    for bone in sc.bones:
        if bone.parent_name is None:
            root_name = bone.name
        else:
            children_by_parent.setdefault(bone.parent_name, []).append(bone.name)

    def _sig(bname: str) -> str:
        ch_sigs = sorted(_sig(ch) for ch in children_by_parent.get(bname, ()))
        nb = int(box_counts.get(bname, 0))
        if not ch_sigs:
            return f"({nb})"
        return f"({nb}:[{','.join(ch_sigs)}])"

    return _sig(root_name) if root_name is not None else "(0)"


def _max_bone_depth(bone: Bone, depth: int = 1) -> int:
    sub = [
        _max_bone_depth(it, depth + 1)
        for it in bone.items
        if isinstance(it, Bone)
    ]
    return max([depth, *sub])


def compute_qd_signature(
    program: Program,
    scene: Scene,
    *,
    domain: str,
    prompt_style: str,
) -> dict[str, Any]:
    """Compute MAP-Elites-style behavioral descriptor for Quality-Diversity auditing."""
    all_pts = np.concatenate([np.asarray(b.corners(), dtype=np.float64) for b in scene.boxes], axis=0)
    spans = np.maximum(1e-6, all_pts.max(axis=0) - all_pts.min(axis=0))
    dx, dy, dz = float(spans[0]), float(spans[1]), float(spans[2])
    max_horiz = max(dx, dz)

    if dy / max_horiz >= 1.55:
        aspect = "tall_vertical"
    elif max_horiz / dy >= 1.65:
        aspect = "wide_or_long_sprawling"
    elif abs(dx - dz) / max_horiz >= 0.42:
        aspect = "elongated_directional"
    else:
        aspect = "balanced_volumetric"

    depth = _max_bone_depth(program.root)
    box_bin = "12_16" if len(scene.boxes) <= 16 else ("17_22" if len(scene.boxes) <= 22 else "23_plus")
    qd_cell = f"{domain}|{aspect}|d{min(depth, 5)}|{box_bin}|m{int(bool(program.materials))}|{prompt_style[:7]}"
    return {
        "domain": domain,
        "aspect_category": aspect,
        "max_bone_depth": int(depth),
        "bbox_spans_xyz": [round(dx, 2), round(dy, 2), round(dz, 2)],
        "qd_cell": qd_cell,
    }


def summarize_corpus_diversity(
    *,
    domains: list[str],
    prompt_styles: list[str],
    naming_registers: list[str],
    qd_cells: list[str],
    unique_identifiers: set[str],
    materials_block_count: int,
) -> dict[str, Any]:
    """Compute corpus-level Shannon entropy and coverage statistics for dataset_manifest.json."""
    n = max(1, len(qd_cells))
    cell_counts = Counter(qd_cells)
    probs = [c / n for c in cell_counts.values()]
    shannon_bits = -sum(p * math.log2(p) for p in probs if p > 0)
    max_possible_bits = math.log2(max(2, len(cell_counts)))

    return {
        "domain_counts": dict(sorted(Counter(domains).items())),
        "prompt_granularity_counts": dict(sorted(Counter(prompt_styles).items())),
        "naming_register_counts": dict(sorted(Counter(naming_registers).items())),
        "occupied_qd_behavior_cells": int(len(cell_counts)),
        "qd_shannon_entropy_bits": round(shannon_bits, 4),
        "qd_normalized_evenness": round(shannon_bits / max(1e-9, max_possible_bits), 4),
        "unique_ast_identifier_vocab_size": int(len(unique_identifiers)),
        "stateful_materials_coverage_ratio": round(materials_block_count / n, 4),
    }
