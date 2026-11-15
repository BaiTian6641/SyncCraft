"""Dragon, Mythic Fauna, Aquatic, Conifer & Watermill Expansion for SyncCraft v3.0 (ADR 0004).

Adds 11 anatomically grounded, multi-bone procedural Block-DSL v3 AST generators:
  A. Dragon / Mythic Creature Domain (7 Seen + 1 Held-Out Unseen; archetype="entity"):
     1. quadruped_armored_drake   (Seen: wingless 4-legged osteoderm drake + tail mace)
     2. eastern_lung_dragon       (Seen: wingless 4-somite serpentine lung + antlers + whiskers)
     3. bipedal_wyvern            (Seen: 2-legged bat-wing wyvern + venomous barbed stinger)
     4. forelimb_lindworm         (Seen: 2-forelimb wingless/hindlimbless 4-coil serpent wyrm)
     5. feathered_raptor          (Seen: dromaeosaurid theropod + wing-fans + raised sickle claw)
     6. winged_amphithere         (Seen: legless hooded feathered serpent + 2-tier avian wings)
     7. multi_headed_hydra        (Seen: 3-headed polycephalic reptile with flared lateral necks)
     8. crowned_zilant            (Unseen: bipedal crowned wyvern-serpent with 3-prong golden crown)
  B. Animal, Natural & Building Expansion (3 Seen):
     9. articulated_manta_shark   (Seen, Animal: pelagic manta / hammerhead / chimaera swimmer)
    10. pine_conifer_evergreen    (Seen, Natural: rooted alpine conifer with 3 drooping whorls)
    11. waterwheel_millhouse      (Seen, Building: half-timbered stone gristmill + side waterwheel)

Every generator produces typed Block-DSL v3 Program ASTs that compile through the shared
right-handed SE(3) chain and satisfy 15-axis OBB SAT + P1-02 physical contact audits.
"""

from __future__ import annotations

import random
from typing import Any

from synccraft.dsl.ast_nodes import Attach, Bone, Box, Meta, Palette, Place, Program
from .diversity_engine import halton_vector

EXPANSION_FAMILY_DOMAINS: dict[str, str] = {
    "quadruped_armored_drake": "dragon",
    "eastern_lung_dragon": "dragon",
    "bipedal_wyvern": "dragon",
    "forelimb_lindworm": "dragon",
    "feathered_raptor": "dragon",
    "winged_amphithere": "dragon",
    "multi_headed_hydra": "dragon",
    "crowned_zilant": "dragon",
    "articulated_manta_shark": "animal",
    "pine_conifer_evergreen": "natural",
    "waterwheel_millhouse": "building",
}

EXPANSION_SEEN_FAMILIES: tuple[str, ...] = (
    "quadruped_armored_drake",
    "eastern_lung_dragon",
    "bipedal_wyvern",
    "forelimb_lindworm",
    "feathered_raptor",
    "winged_amphithere",
    "multi_headed_hydra",
    "articulated_manta_shark",
    "pine_conifer_evergreen",
    "waterwheel_millhouse",
)

EXPANSION_UNSEEN_FAMILIES: tuple[str, ...] = (
    "crowned_zilant",
)

EXPANSION_ENTITY_BODY_PLANS: dict[str, str] = {
    "quadruped_armored_drake": "quadruped_armored_drake",
    "eastern_lung_dragon": "eastern_lung_dragon",
    "bipedal_wyvern": "bipedal_wyvern",
    "forelimb_lindworm": "forelimb_lindworm",
    "feathered_raptor": "feathered_raptor",
    "winged_amphithere": "winged_amphithere",
    "multi_headed_hydra": "multi_headed_hydra",
    "crowned_zilant": "crowned_zilant",
    "articulated_manta_shark": "articulated_manta_shark",
}

EXPANSION_HUMAN_FAMILY_NOUNS: dict[str, tuple[str, ...]] = {
    "quadruped_armored_drake": (
        "wingless four-legged armored terrestrial drake",
        "osteoderm-plated canyon drake with clubbed tail",
        "heavy quadrupedal earth drake with snapping jaw",
        "thorn-backed terrestrial war drake",
    ),
    "eastern_lung_dragon": (
        "imperial wingless eastern lung dragon",
        "antlered serpentine shenlong cloud dragon",
        "whiskered celestial river dragon with dorsal mane",
        "four-clawed undulating eastern dragon",
    ),
    "bipedal_wyvern": (
        "two-legged winged wyvern with barbed venom stinger",
        "raptorial cliff wyvern with bat-membrane forewings",
        "crested bipedal wyvern with scorpion-stinger tail",
        "taloned storm wyvern",
    ),
    "forelimb_lindworm": (
        "two-forelimbed serpentine nordic lindworm",
        "wingless clawed coil wyrm with dorsal finlets",
        "horned pectoral serpent lindorm",
        "undulating two-armed fen lindworm",
    ),
    "feathered_raptor": (
        "feathered dromaeosaurid sickle-claw raptor",
        "winged-arm deinonychus theropod predator",
        "crested velociraptor with stiff plumed tail fan",
        "digitigrade feathered raptor dinosaur",
    ),
    "winged_amphithere": (
        "legless hooded feathered amphithere serpent",
        "winged coatl sky-serpent with multi-tiered pinions",
        "cobra-hooded avian-winged serpent dragon",
        "plumed legless amphithere glider",
    ),
    "multi_headed_hydra": (
        "three-headed polycephalic swamp hydra",
        "tri-jawed reptilian hydra beast with twin-fluked tail",
        "massive four-legged three-necked hydra",
        "lernaean multi-headed serpent monster",
    ),
    "crowned_zilant": (
        "mythic golden-crowned bipedal zilant wyvern",
        "heraldic three-prong-crowned winged serpent dragon",
        "taloned crowned zilant with curved serpent tail",
        "royal-crowned breastplated wyvern",
    ),
    "articulated_manta_shark": (
        "pelagic manta ray or hammerhead shark swimmer",
        "hydrofoil reef manta glider with pectoral wing-fins",
        "cephalofoil pelagic shark or abyssal chimaera",
        "lunate-tailed oceanic cartilaginous swimmer",
    ),
    "pine_conifer_evergreen": (
        "rooted alpine spruce or pine conifer tree",
        "tiered drooping-bough evergreen conifer",
        "buttress-rooted boreal cedar spire tree",
        "resinous mountain fir tree on mossy rock mound",
    ),
    "waterwheel_millhouse": (
        "half-timbered stone watermill with side paddle wheel",
        "oak-and-ashlar gristmill house beside millrace sluice",
        "gabled village waterwheel millhouse",
        "medieval stone sluice mill with rotary paddle wheel",
    ),
}

# Perceptual OKLCH palette banks (L in [0.15, 0.95], C >= 0, H in [0, 360))
_DRAKE_SCALES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("basalt_osteoderm_scale", (0.34, 0.038, 255.0)),
    ("ochre_canyon_carapace", (0.52, 0.125, 58.0)),
    ("verdigris_wyrm_scale", (0.46, 0.135, 162.0)),
    ("crimson_drake_hide", (0.45, 0.168, 28.0)),
    ("imperial_jade_scale", (0.54, 0.148, 154.0)),
    ("celestial_azure_scale", (0.50, 0.150, 236.0)),
    ("obsidian_wyvern_hide", (0.27, 0.032, 272.0)),
    ("quetzal_plume_emerald", (0.56, 0.165, 148.0)),
)

_PLUMAGE_AND_MEMBRANES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("patagium_wing_membrane", (0.64, 0.072, 46.0)),
    ("dusk_leathery_web", (0.38, 0.058, 18.0)),
    ("amber_remiges_plume", (0.68, 0.158, 68.0)),
    ("cobalt_pinion_feather", (0.48, 0.155, 244.0)),
    ("scarlet_crest_plume", (0.58, 0.185, 26.0)),
    ("pale_ventral_scute", (0.80, 0.052, 82.0)),
)

_KERATIN_AND_METALS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("ivory_keratin_horn", (0.86, 0.048, 86.0)),
    ("charcoal_talon_claw", (0.28, 0.022, 252.0)),
    ("imperial_crown_gold", (0.78, 0.162, 82.0)),
    ("burnished_electrum_gold", (0.82, 0.145, 88.0)),
    ("forged_iron_hub", (0.42, 0.020, 248.0)),
)

_PELAGIC_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("pelagic_indigo_dorsum", (0.40, 0.115, 242.0)),
    ("reef_slate_hydrofoil", (0.48, 0.048, 225.0)),
    ("abyssal_violet_skin", (0.34, 0.095, 282.0)),
    ("countershade_pearl_belly", (0.86, 0.022, 215.0)),
    ("biolum_cyan_fin", (0.78, 0.165, 196.0)),
)

_CONIFER_AND_EARTH: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("spruce_needle_bough", (0.40, 0.112, 154.0)),
    ("alpine_pine_crown", (0.46, 0.125, 146.0)),
    ("blue_cedar_foliage", (0.48, 0.088, 178.0)),
    ("resinous_pine_bark", (0.36, 0.078, 46.0)),
    ("redwood_heart_timber", (0.44, 0.102, 38.0)),
    ("mossy_granite_root", (0.50, 0.085, 132.0)),
)

_MILLHOUSE_PALETTES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("river_cobble_masonry", (0.48, 0.024, 240.0)),
    ("limewashed_plaster_wall", (0.82, 0.032, 86.0)),
    ("seasoned_oak_beam", (0.42, 0.092, 54.0)),
    ("waterlogged_paddle_wood", (0.34, 0.068, 48.0)),
    ("terracotta_gable_shingle", (0.52, 0.138, 38.0)),
    ("slate_mill_roof", (0.35, 0.026, 252.0)),
)

_EMISSIVE_ACCENTS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("venom_acid_glow", (0.80, 0.195, 136.0)),
    ("dragon_ember_core", (0.74, 0.205, 44.0)),
    ("aether_pearl_glow", (0.84, 0.165, 198.0)),
    ("warm_hearth_lantern", (0.86, 0.175, 78.0)),
)


def _jitter_lch(
    rng: random.Random,
    base_lch: tuple[float, float, float],
    *,
    l_span: float = 0.045,
    c_span: float = 0.018,
    h_span: float = 16.0,
) -> tuple[float, float, float]:
    l0, c0, h0 = base_lch
    l_val = round(min(0.95, max(0.15, l0 + rng.uniform(-l_span, l_span))), 4)
    c_val = round(max(0.006, c0 + rng.uniform(-c_span, c_span * 1.2)), 4)
    h_val = round((h0 + rng.uniform(-h_span, h_span)) % 360.0, 2)
    return (l_val, c_val, h_val)


def _pal(
    rng: random.Random,
    spec: tuple[str, tuple[float, float, float]],
    *,
    emissive: bool = False,
    suffix: str = "",
) -> Palette:
    raw_name, base_lch = spec
    steps = rng.choice((4, 5, 6, 7))
    hue_shift = (round(rng.uniform(-6.0, 0.0), 2), round(rng.uniform(0.0, 8.0), 2))
    return Palette(
        name=f"{raw_name}{suffix}",
        base_lch=_jitter_lch(rng, base_lch),
        steps=steps,
        hue_shift=hue_shift,
        emissive=emissive,
    )


def _jpiv(
    rng: random.Random,
    base_xyz: tuple[float, float, float],
    span: float = 0.55,
) -> tuple[float, float, float]:
    _ = (rng, base_xyz, span)
    return (0.0, 0.0, 0.0)


def _hpick(h_u: float, choices: tuple[Any, ...]) -> Any:
    idx = min(len(choices) - 1, int(h_u * len(choices)))
    return choices[idx]


def _entity_profile_and_density(v_idx: int) -> tuple[str, int]:
    profiles = ("bbmodel_5", "bedrock_entity", "geckolib")
    profile = profiles[(v_idx // 2) % len(profiles)]
    density = 32 if (v_idx % 5 == 0) else 16
    return profile, density


def _static_profile_and_density(v_idx: int) -> tuple[str, int]:
    profile = "java_modern" if (v_idx % 2 == 0) else "bbmodel_5"
    density = 32 if (v_idx % 6 == 0) else 16
    return profile, density


# ==============================================================================
# A. DRAGON & FANCY MYTHIC CREATURE DOMAIN (8 Generators: 7 Seen + 1 Unseen)
# ==============================================================================


def _gen_quadruped_armored_drake(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Wingless 4-legged terrestrial armored drake with dorsal osteoderms & tail mace (>= 20 boxes)."""
    substyles = (
        "ankylosaur_earth_drake",
        "basalt_osteoderm_drake",
        "ironhide_canyon_drake",
        "thornback_terrestrial_drake",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 3 + 1, dim=6, jitter_rng=rng)

    p_carapace = _pal(rng, _DRAKE_SCALES[v_idx % 4])
    p_belly = _pal(rng, _PLUMAGE_AND_MEMBRANES[5], suffix="_plastron")
    p_horn = _pal(rng, _KERATIN_AND_METALS[v_idx % 2], suffix="_osteoderm")
    p_maw = _pal(rng, _EMISSIVE_ACCENTS[1], emissive=(v_idx % 3 == 0), suffix="_maw")

    thorax_w = _hpick(hv[0], (7, 8))
    thorax_l = _hpick(hv[1], (7, 8))
    pelvis_l = _hpick(hv[2], (6, 7))
    jaw_deg = _hpick(hv[3], (11.25, 15.0, 22.5))
    tail_yaw = _hpick(hv[4], (-15.0, -11.25, 11.25, 15.0))
    stride_deg = _hpick(hv[5], (-11.25, 0.0, 11.25))
    armor_inf = round(rng.choice((0.0, 0.04, 0.07)), 3)

    brow_horn = Bone(
        "squamosal_horn_l",
        rest_euler=(22.5, 0.0, -18.0),
        attach=Attach("drake_cranium", (0.8, 1.0, 0.3), (0.0, -1.0, 0.0)),
        mirror_name="squamosal_horn_r",
        items=[
            Box("squamosal_horn_spike", (1, 3, 1), p_horn.name),
        ],
    )

    skull_bone = Bone(
        "snapping_skull_rig",
        rest_euler=(11.25, 0.0, 0.0),
        attach=Attach("armored_neck_seg", (0.0, 0.15, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("drake_cranium", (5, 3, 4), p_carapace.name),
            Box("hooked_rostrum", (3, 2, 4), p_carapace.name, Place("drake_cranium", (0.0, 0.45, -1.0), (0.0, 0.0, 1.0))),
            Box("nasal_boss_horn", (1, 2, 1), p_horn.name, Place("hooked_rostrum", (0.0, 1.0, -0.4), (0.0, -1.0, 0.0))),
            brow_horn,
            Bone(
                "articulated_lower_jaw",
                rest_euler=(-jaw_deg, 0.0, 0.0),
                attach=Attach("drake_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                items=[
                    Box("crushing_mandible", (3, 1, 4), p_belly.name),
                    Box("chin_osteoderm_spike", (2, 1, 2), p_maw.name, Place("crushing_mandible", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    neck_bone = Bone(
        "cervical_collar",
        rest_euler=(-11.25, 0.0, 0.0),
        attach=Attach("thoracic_carapace", (0.0, 0.25, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("armored_neck_seg", (4, 3, 3), p_carapace.name),
            Box("nuchal_osteoderm_plate", (2, 1, 2), p_horn.name, Place("armored_neck_seg", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            skull_bone,
        ],
    )

    foreleg_bone = Bone(
        "foreleg_upper_l",
        rest_euler=(stride_deg, 0.0, 15.0),
        attach=Attach("thoracic_carapace", (1.0, -0.45, -0.45), (-1.0, 0.6, 0.0)),
        mirror_name="foreleg_upper_r",
        items=[
            Box("fore_brachium", (3, 4, 3), p_carapace.name),
            Bone(
                "foreleg_lower_l",
                rest_euler=(-11.25, 0.0, -7.5),
                attach=Attach("fore_brachium", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                items=[
                    Box("fore_antebrachium", (2, 3, 2), p_belly.name),
                    Box("fore_clawed_manus", (3, 1, 4), p_horn.name, Place("fore_antebrachium", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    hindleg_bone = Bone(
        "hindleg_upper_l",
        rest_euler=(-stride_deg, 0.0, 15.0),
        attach=Attach("pelvic_carapace", (1.0, -0.35, 0.2), (-1.0, 0.6, 0.0)),
        mirror_name="hindleg_upper_r",
        items=[
            Box("hind_femur", (3, 4, 3), p_carapace.name),
            Bone(
                "hindleg_lower_l",
                rest_euler=(11.25, 0.0, -7.5),
                attach=Attach("hind_femur", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                items=[
                    Box("hind_crus_shank", (2, 3, 2), p_belly.name),
                    Box("hind_clawed_pes", (3, 1, 4), p_horn.name, Place("hind_crus_shank", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    tail_bone = Bone(
        "caudal_tail_prox",
        rest_euler=(-7.5, tail_yaw, 0.0),
        attach=Attach("pelvic_carapace", (0.0, 0.1, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("tail_prox_seg", (4, 3, 5), p_carapace.name),
            Box("tail_prox_osteoderm", (2, 1, 3), p_horn.name, Place("tail_prox_seg", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Bone(
                "caudal_tail_club",
                rest_euler=(7.5, -tail_yaw, 0.0),
                attach=Attach("tail_prox_seg", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("tail_dist_shaft", (2, 2, 5), p_carapace.name),
                    Box("ankylosaur_tail_mace", (5, 3, 4), p_horn.name, Place("tail_dist_shaft", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                    Box("mace_spike_l", (2, 1, 2), p_horn.name, Place("ankylosaur_tail_mace", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                    Box("mace_spike_r", (2, 1, 2), p_horn.name, Place("ankylosaur_tail_mace", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("thoracic_carapace", (thorax_w, 5, thorax_l), p_carapace.name, inflate=armor_inf),
        Box("pectoral_plastron", (thorax_w - 2, 1, thorax_l - 1), p_belly.name, Place("thoracic_carapace", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("thoracic_osteoderm_plate", (thorax_w - 2, 1, thorax_l - 2), p_horn.name, Place("thoracic_carapace", (0.0, 1.0, 0.1), (0.0, -1.0, 0.0))),
        Box("dorsal_scute_keel", (2, 1, thorax_l - 3), p_carapace.name, Place("thoracic_osteoderm_plate", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("pelvic_carapace", (thorax_w - 1, 4, pelvis_l), p_carapace.name, Place("thoracic_carapace", (0.0, 0.05, 1.0), (0.0, 0.0, -1.0))),
        Box("pelvic_osteoderm_shield", (thorax_w - 3, 1, pelvis_l - 1), p_horn.name, Place("pelvic_carapace", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        neck_bone,
        foreleg_bone,
        hindleg_bone,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_carapace, p_belly, p_horn, p_maw],
        root=Bone("drake_thorax_root", pivot_local=_jpiv(rng, (8.0, 11.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": thorax_l + pelvis_l + 25,
        "secondary_span": thorax_w + 8,
        "joint_angle_deg": jaw_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": armor_inf,
        "material_theme": f"{p_carapace.name}+{p_horn.name}+{p_belly.name}",
        "notes": f"wingless 4-legged terrestrial armored drake ({substyle.replace('_', ' ')}) with thoracic/pelvic osteoderm plates, horned snapping skull ({jaw_deg:g} deg jaw), 4 mirrored 2-joint clawed legs, and spiked tail mace",
        "extra": {"thorax_w": thorax_w, "thorax_l": thorax_l, "jaw_deg": jaw_deg},
    }


def _gen_eastern_lung_dragon(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Wingless Imperial Eastern Lung dragon with 4-somite serpentine body, antlers & whiskers (>= 22 boxes)."""
    substyles = (
        "imperial_celestial_lung",
        "jade_river_shenlong",
        "azure_tempest_lung",
        "gilded_cloud_dragon",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 5 + 2, dim=5, jitter_rng=rng)

    p_scale = _pal(rng, _DRAKE_SCALES[4 + (v_idx % 2)])
    p_mane = _pal(rng, _PLUMAGE_AND_MEMBRANES[2 + (v_idx % 3)], suffix="_mane")
    p_antler = _pal(rng, _KERATIN_AND_METALS[2 if v_idx % 2 == 0 else 0], suffix="_antler")
    p_pearl = _pal(rng, _EMISSIVE_ACCENTS[2], emissive=True, suffix="_pearl")

    seg_len = _hpick(hv[0], (6, 7))
    wave_yaw = _hpick(hv[1], (15.0, 22.5))
    wave_pitch = _hpick(hv[2], (-7.5, 7.5))
    jaw_deg = _hpick(hv[3], (11.25, 15.0, 22.5))
    scale_inf = round(_hpick(hv[4], (0.0, 0.04, 0.06)), 3)

    antler_bone = Bone(
        "stag_antler_l",
        rest_euler=(22.5, 11.25, -18.0),
        attach=Attach("lung_cranium", (0.85, 1.0, 0.2), (0.0, -1.0, 0.0)),
        mirror_name="stag_antler_r",
        items=[
            Box("antler_main_beam", (1, 4, 1), p_antler.name),
            Box("antler_brow_tine", (1, 1, 2), p_antler.name, Place("antler_main_beam", (0.0, 0.35, -1.0), (0.0, 0.0, 1.0))),
            Box("antler_crown_fork", (2, 2, 1), p_antler.name, Place("antler_main_beam", (1.0, 0.75, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    whisker_bone = Bone(
        "snout_whisker_l",
        rest_euler=(5.0, -25.0, -11.25),
        attach=Attach("imperial_snout_bridge", (1.0, 0.1, -0.55), (-1.0, 0.0, 0.0)),
        mirror_name="snout_whisker_r",
        items=[
            Box("whisker_tendril_prox", (3, 1, 1), p_mane.name),
            Box("whisker_tendril_tip", (2, 1, 1), p_pearl.name, Place("whisker_tendril_prox", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    head_bone = Bone(
        "noble_dragon_head",
        rest_euler=(7.5, -wave_yaw * 0.5, 0.0),
        attach=Attach("anterior_trunk_seg", (0.0, 0.25, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("lung_cranium", (5, 4, 5), p_scale.name),
            Box("imperial_snout_bridge", (3, 2, 4), p_scale.name, Place("lung_cranium", (0.0, 0.15, -1.0), (0.0, 0.0, 1.0))),
            Box("crown_mane_crest", (2, 2, 3), p_mane.name, Place("lung_cranium", (0.0, 1.0, 0.2), (0.0, -1.0, 0.0))),
            antler_bone,
            whisker_bone,
            Bone(
                "bearded_lower_jaw",
                rest_euler=(-jaw_deg, 0.0, 0.0),
                attach=Attach("lung_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                items=[
                    Box("lung_mandible", (3, 1, 4), p_scale.name),
                    Box("chin_beard_tuft", (2, 2, 1), p_mane.name, Place("lung_mandible", (0.0, -1.0, -0.6), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    fore_claws = Bone(
        "fore_claw_limb_l",
        rest_euler=(-15.0, 0.0, 15.0),
        attach=Attach("anterior_trunk_seg", (1.0, -0.5, -0.1), (-1.0, 0.5, 0.0)),
        mirror_name="fore_claw_limb_r",
        items=[
            Box("fore_limb_arm", (2, 3, 2), p_scale.name),
            Box("fore_cloud_talons", (2, 1, 3), p_antler.name, Place("fore_limb_arm", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
        ],
    )

    hind_claws = Bone(
        "hind_claw_limb_l",
        rest_euler=(11.25, 0.0, 15.0),
        attach=Attach("lumbar_trunk_seg", (1.0, -0.5, 0.0), (-1.0, 0.5, 0.0)),
        mirror_name="hind_claw_limb_r",
        items=[
            Box("hind_limb_leg", (2, 3, 2), p_scale.name),
            Box("hind_cloud_talons", (2, 1, 3), p_antler.name, Place("hind_limb_leg", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
        ],
    )

    tail_tuft_bone = Bone(
        "tail_tuft",
        rest_euler=(0.0, -wave_yaw, 0.0),
        attach=Attach("caudal_trunk_seg", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("tail_brush_core", (2, 2, 4), p_mane.name),
            Box("tail_pearl_plume", (1, 3, 4), p_pearl.name, Place("tail_brush_core", (0.0, 0.2, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    caudal_somite = Bone(
        "caudal_somite",
        rest_euler=(wave_pitch, wave_yaw * 1.25, 0.0),
        attach=Attach("lumbar_trunk_seg", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("caudal_trunk_seg", (3, 2, seg_len - 1), p_scale.name),
            Box("caudal_dorsal_mane", (1, 2, seg_len - 3), p_mane.name, Place("caudal_trunk_seg", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            tail_tuft_bone,
        ],
    )

    lumbar_somite = Bone(
        "lumbar_somite",
        rest_euler=(-wave_pitch, -wave_yaw * 1.35, 0.0),
        attach=Attach("mid_trunk_seg", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("lumbar_trunk_seg", (4, 3, seg_len), p_scale.name),
            Box("lumbar_dorsal_mane", (1, 2, seg_len - 2), p_mane.name, Place("lumbar_trunk_seg", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            hind_claws,
            caudal_somite,
        ],
    )

    mid_somite = Bone(
        "mid_somite",
        rest_euler=(wave_pitch, wave_yaw, 0.0),
        attach=Attach("anterior_trunk_seg", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("mid_trunk_seg", (4, 4, seg_len), p_scale.name),
            Box("mid_dorsal_mane", (1, 2, seg_len - 2), p_mane.name, Place("mid_trunk_seg", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            lumbar_somite,
        ],
    )

    has_celestial_pearl = (v_idx % 2 == 0)
    root_items: list[Box | Bone] = [
        Box("anterior_trunk_seg", (5, 4, seg_len), p_scale.name, inflate=scale_inf),
        Box("anterior_belly_scutes", (3, 1, seg_len - 2), p_antler.name, Place("anterior_trunk_seg", (0.0, -1.0, -0.2), (0.0, 1.0, 0.0), (0.0, 0.0, 0.25))),
        Box("anterior_dorsal_mane", (1, 2, seg_len - 2), p_mane.name, Place("anterior_trunk_seg", (0.0, 1.0, -0.15), (0.0, -1.0, 0.0), (0.0, 0.0, -0.2))),
        head_bone,
        fore_claws,
        mid_somite,
    ]
    if has_celestial_pearl:
        root_items.append(
            Bone(
                "celestial_orbit_pearl",
                rest_euler=(0.0, 14.6, 0.0),
                attach=Attach("anterior_dorsal_mane", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.8, -1.5), detached=True),
                items=[Box("luminous_sky_pearl", (2, 2, 2), p_pearl.name)],
            )
        )

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_scale, p_mane, p_antler, p_pearl],
        root=Bone("anterior_somite", pivot_local=(0.0, 0.0, 0.0), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 4 * seg_len + 16,
        "secondary_span": 15,
        "joint_angle_deg": wave_yaw,
        "has_mirror_x": True,
        "has_detached_bone": has_celestial_pearl,
        "inflate_detail": scale_inf,
        "material_theme": f"{p_scale.name}+{p_mane.name}+{p_antler.name}",
        "notes": f"wingless 4-somite Imperial Eastern Lung dragon ({substyle.replace('_', ' ')}) with dorsal mane ridge, 4 short clawed limbs, branching stag antlers, flowing snout whiskers, bearded jaw, and tail tuft",
        "extra": {"seg_len": seg_len, "wave_yaw": wave_yaw},
    }


def _gen_bipedal_wyvern(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Bipedal Wyvern with NO separate forelegs, 2 bat-wings, raptorial legs & venom stinger (>= 22 boxes)."""
    substyles = (
        "venom_barb_wyvern",
        "cliff_crag_wyvern",
        "cinder_wing_wyvern",
        "storm_talon_wyvern",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 7 + 3, dim=6, jitter_rng=rng)

    p_hide = _pal(rng, _DRAKE_SCALES[3 + (v_idx % 4)])
    p_web = _pal(rng, _PLUMAGE_AND_MEMBRANES[v_idx % 2], suffix="_patagium")
    p_claw = _pal(rng, _KERATIN_AND_METALS[1], suffix="_talon")
    p_venom = _pal(rng, _EMISSIVE_ACCENTS[0], emissive=True, suffix="_stinger")

    w_in = _hpick(hv[0], (6, 7))
    w_out = _hpick(hv[1], (7, 8))
    wing_dihedral = _hpick(hv[2], (18.0, 22.5, 30.0))
    neck_h = _hpick(hv[3], (5, 6))
    jaw_deg = _hpick(hv[4], (15.0, 22.5))
    tail_yaw = _hpick(hv[5], (-15.0, 0.0, 15.0))
    scale_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    neck_bone = Bone(
        "wyvern_neck",
        rest_euler=(-22.5, 0.0, 0.0),
        attach=Attach("keeled_thorax", (0.0, 0.85, -0.65), (0.0, -1.0, 0.0)),
        items=[
            Box("elongated_cervical", (3, neck_h, 3), p_hide.name),
            Box("cervical_fin_crest", (1, 3, 1), p_claw.name, Place("elongated_cervical", (0.0, 0.25, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "wyvern_skull",
                rest_euler=(22.5, 0.0, 0.0),
                attach=Attach("elongated_cervical", (0.0, 1.0, -0.3), (0.0, -1.0, 0.0)),
                items=[
                    Box("wyvern_cranium", (4, 3, 4), p_hide.name),
                    Box("hooked_beak_snout", (2, 2, 4), p_hide.name, Place("wyvern_cranium", (0.0, 0.2, -1.0), (0.0, 0.0, 1.0))),
                    Box("occipital_sweep_crest", (2, 2, 4), p_claw.name, Place("wyvern_cranium", (0.0, 1.0, 0.4), (0.0, -1.0, -0.4))),
                    Bone(
                        "wyvern_jaw",
                        rest_euler=(-jaw_deg, 0.0, 0.0),
                        attach=Attach("wyvern_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                        items=[Box("serrated_mandible", (2, 1, 4), p_web.name)],
                    ),
                ],
            ),
        ],
    )

    wing_bone = Bone(
        "wing_humerus_l",
        rest_euler=(0.0, 0.0, wing_dihedral),
        attach=Attach("keeled_thorax", (1.0, 0.65, -0.25), (-1.0, 0.0, 0.0)),
        mirror_name="wing_humerus_r",
        items=[
            Box("humerus_wing_spar", (w_in, 2, 2), p_hide.name),
            Box("propatagium_panel", (w_in - 2, 1, 5), p_web.name, Place("humerus_wing_spar", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "wing_forearm_claw_l",
                rest_euler=(0.0, -11.25, -15.0),
                attach=Attach("humerus_wing_spar", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                items=[
                    Box("forearm_metacarpal_spar", (w_out, 1, 2), p_hide.name),
                    Box("alular_walking_claw", (1, 1, 3), p_claw.name, Place("forearm_metacarpal_spar", (-0.8, 0.0, -1.0), (0.0, 0.0, 1.0))),
                    Box("dactylopatagium_panel", (w_out - 2, 1, 5), p_web.name, Place("forearm_metacarpal_spar", (0.2, 0.0, 1.0), (0.0, 0.0, -1.0))),
                ],
            ),
        ],
    )

    hindleg_bone = Bone(
        "thigh_l",
        rest_euler=(-15.0, 0.0, 11.25),
        attach=Attach("wyvern_pelvis", (1.0, -0.35, 0.0), (-1.0, 0.6, 0.0)),
        mirror_name="thigh_r",
        items=[
            Box("raptorial_femur", (2, 5, 3), p_hide.name),
            Bone(
                "shin_talon_l",
                rest_euler=(22.5, 0.0, -7.5),
                attach=Attach("raptorial_femur", (0.0, -1.0, 0.1), (0.0, 1.0, 0.0)),
                items=[
                    Box("tarsometatarsus_shank", (2, 5, 2), p_web.name),
                    Box("grasping_raptor_talons", (3, 1, 4), p_claw.name, Place("tarsometatarsus_shank", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    tail_bone = Bone(
        "wyvern_tail_prox",
        rest_euler=(-11.25, tail_yaw, 0.0),
        attach=Attach("wyvern_pelvis", (0.0, 0.2, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("tail_seg_prox", (3, 3, 5), p_hide.name),
            Bone(
                "wyvern_tail_mid",
                rest_euler=(15.0, -tail_yaw, 0.0),
                attach=Attach("tail_seg_prox", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("tail_seg_mid", (2, 2, 5), p_hide.name),
                    Bone(
                        "wyvern_stinger_tip",
                        rest_euler=(-18.0, 0.0, 0.0),
                        attach=Attach("tail_seg_mid", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                        items=[
                            Box("venom_bulb_telson", (3, 2, 3), p_claw.name),
                            Box("barbed_stinger_needle", (1, 1, 4), p_venom.name, Place("venom_bulb_telson", (0.0, 0.25, 1.0), (0.0, 0.0, -1.0))),
                        ],
                    ),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("keeled_thorax", (6, 5, 7), p_hide.name, inflate=scale_inf),
        Box("avian_sternal_keel", (2, 2, 5), p_web.name, Place("keeled_thorax", (0.0, -1.0, -0.15), (0.0, 1.0, 0.0))),
        Box("thoracic_ridge_spines", (1, 2, 3), p_claw.name, Place("keeled_thorax", (0.0, 1.0, 0.45), (0.0, -1.0, 0.0))),
        Box("wyvern_pelvis", (4, 4, 5), p_hide.name, Place("keeled_thorax", (0.0, -0.1, 1.0), (0.0, 0.0, -1.0))),
        neck_bone,
        wing_bone,
        hindleg_bone,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_hide, p_web, p_claw, p_venom],
        root=Bone("wyvern_sternum_root", pivot_local=_jpiv(rng, (8.0, 15.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 6 + 2 * (w_in + w_out),
        "secondary_span": 30,
        "joint_angle_deg": wing_dihedral,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": scale_inf,
        "material_theme": f"{p_hide.name}+{p_web.name}+{p_venom.name}",
        "notes": f"bipedal wyvern ({substyle.replace('_', ' ')}) with keeled avian sternum, zero separate forelegs, 2 expansive bat-membrane forewings ({wing_dihedral:g} deg dihedral), raptorial hindlegs, crested skull, and venomous barbed tail stinger",
        "extra": {"w_in": w_in, "w_out": w_out, "neck_h": neck_h},
    }


def _gen_forelimb_lindworm(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Serpentine Lindworm with 2 clawed forelimbs, ZERO hindlimbs/wings & 4-coil tail (>= 18 boxes)."""
    substyles = (
        "nordic_fen_lindworm",
        "crag_coil_lindorm",
        "venom_hood_lindworm",
        "runic_scale_lindworm",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 3 + 4, dim=5, jitter_rng=rng)

    p_scale = _pal(rng, _DRAKE_SCALES[(v_idx + 2) % len(_DRAKE_SCALES)])
    p_fin = _pal(rng, _PLUMAGE_AND_MEMBRANES[2 + (v_idx % 3)], suffix="_finlet")
    p_belly = _pal(rng, _PLUMAGE_AND_MEMBRANES[5], suffix="_scute")
    p_horn = _pal(rng, _KERATIN_AND_METALS[0], suffix="_fang")

    seg_l = _hpick(hv[0], (5, 6))
    wave_deg = _hpick(hv[1], (15.0, 22.5))
    jaw_deg = _hpick(hv[2], (15.0, 22.5))
    fore_pitch = _hpick(hv[3], (-15.0, -7.5, 7.5))
    scale_inf = round(_hpick(hv[4], (0.0, 0.04, 0.07)), 3)

    neck_bone = Bone(
        "lindworm_neck",
        rest_euler=(-18.0, 0.0, 0.0),
        attach=Attach("pectoral_thorax_hull", (0.0, 0.75, -0.6), (0.0, -1.0, 0.0)),
        items=[
            Box("serpent_cervical", (4, 4, 3), p_scale.name),
            Bone(
                "lindworm_skull",
                rest_euler=(18.0, 0.0, 0.0),
                attach=Attach("serpent_cervical", (0.0, 1.0, -0.3), (0.0, -1.0, 0.0)),
                items=[
                    Box("viperine_cranium", (5, 3, 4), p_scale.name),
                    Box("fang_snout", (3, 2, 4), p_scale.name, Place("viperine_cranium", (0.0, 0.15, -1.0), (0.0, 0.0, 1.0))),
                    Box("brow_horn_l", (1, 2, 3), p_horn.name, Place("viperine_cranium", (0.8, 1.0, 0.35), (0.0, -1.0, -0.35))),
                    Box("brow_horn_r", (1, 2, 3), p_horn.name, Place("viperine_cranium", (-0.8, 1.0, 0.35), (0.0, -1.0, -0.35))),
                    Bone(
                        "lindworm_jaw",
                        rest_euler=(-jaw_deg, 0.0, 0.0),
                        attach=Attach("viperine_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                        items=[Box("split_chin_mandible", (3, 1, 4), p_belly.name)],
                    ),
                ],
            ),
        ],
    )

    forelimb_bone = Bone(
        "lindworm_forelimb_l",
        rest_euler=(fore_pitch, 0.0, 18.0),
        attach=Attach("pectoral_thorax_hull", (1.0, -0.35, -0.2), (-1.0, 0.5, 0.0)),
        mirror_name="lindworm_forelimb_r",
        items=[
            Box("muscular_brachium", (3, 4, 3), p_scale.name),
            Bone(
                "lindworm_foreclaw_l",
                rest_euler=(-15.0, 0.0, -7.5),
                attach=Attach("muscular_brachium", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
                items=[
                    Box("forearm_gauntlet_scale", (2, 3, 2), p_belly.name),
                    Box("digging_talon_manus", (3, 1, 4), p_horn.name, Place("forearm_gauntlet_scale", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    coil_seg_4 = Bone(
        "coil_seg_4",
        rest_euler=(0.0, -wave_deg, 0.0),
        attach=Attach("serpent_body_s3", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("serpent_tail_s4", (2, 2, seg_l - 1), p_scale.name),
            Box("caudal_paddle_finlet", (1, 3, 4), p_fin.name, Place("serpent_tail_s4", (0.0, 0.2, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    coil_seg_3 = Bone(
        "coil_seg_3",
        rest_euler=(0.0, wave_deg * 1.35, 0.0),
        attach=Attach("serpent_body_s2", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("serpent_body_s3", (3, 3, seg_l), p_scale.name),
            Box("dorsal_finlet_s3", (1, 2, seg_l - 2), p_fin.name, Place("serpent_body_s3", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            coil_seg_4,
        ],
    )

    coil_seg_2 = Bone(
        "coil_seg_2",
        rest_euler=(-5.0, -wave_deg * 1.35, 0.0),
        attach=Attach("serpent_body_s1", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("serpent_body_s2", (4, 3, seg_l), p_scale.name),
            Box("dorsal_finlet_s2", (1, 2, seg_l - 2), p_fin.name, Place("serpent_body_s2", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            coil_seg_3,
        ],
    )

    coil_seg_1 = Bone(
        "coil_seg_1",
        rest_euler=(5.0, wave_deg, 0.0),
        attach=Attach("pectoral_thorax_hull", (0.0, -0.1, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("serpent_body_s1", (5, 4, seg_l), p_scale.name),
            Box("dorsal_finlet_s1", (1, 2, seg_l - 2), p_fin.name, Place("serpent_body_s1", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            coil_seg_2,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("pectoral_thorax_hull", (6, 5, 6), p_scale.name, inflate=scale_inf),
        Box("keeled_chest_plastron", (4, 1, 5), p_belly.name, Place("pectoral_thorax_hull", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("pectoral_dorsal_finlet", (1, 2, 2), p_fin.name, Place("pectoral_thorax_hull", (0.0, 1.0, 0.55), (0.0, -1.0, 0.0))),
        neck_bone,
        forelimb_bone,
        coil_seg_1,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_scale, p_fin, p_belly, p_horn],
        root=Bone("lindworm_pectoral_root", pivot_local=_jpiv(rng, (8.0, 10.0, 5.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 6 + 4 * seg_l + 10,
        "secondary_span": 14,
        "joint_angle_deg": wave_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": scale_inf,
        "material_theme": f"{p_scale.name}+{p_fin.name}+{p_horn.name}",
        "notes": f"wingless hindlimbless lindworm ({substyle.replace('_', ' ')}) with 2 muscular clawed forelimbs on armored pectoral hull, horned viperine skull + jaw, and 4-segment undulating serpentine coil with dorsal finlets",
        "extra": {"seg_l": seg_l, "wave_deg": wave_deg},
    }


def _gen_feathered_raptor(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Dromaeosaurid Feathered Raptor with wing-fans, raised inner sickle claw & tail fan (>= 20 boxes)."""
    substyles = (
        "velociraptor_steppe_hunter",
        "deinonychus_crest_stalker",
        "microraptor_iridescent_glider",
        "pyroraptor_ember_plume",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 5 + 6, dim=5, jitter_rng=rng)

    p_body = _pal(rng, _DRAKE_SCALES[(v_idx + 1) % len(_DRAKE_SCALES)])
    p_plume = _pal(rng, _PLUMAGE_AND_MEMBRANES[2 + (v_idx % 3)], suffix="_plumage")
    p_under = _pal(rng, _PLUMAGE_AND_MEMBRANES[5], suffix="_belly")
    p_claw = _pal(rng, _KERATIN_AND_METALS[1], suffix="_sickle")

    tail_len = _hpick(hv[0], (6, 7))
    jaw_deg = _hpick(hv[1], (11.25, 15.0, 22.5))
    stride_deg = _hpick(hv[2], (-11.25, 0.0, 11.25))
    tail_yaw = _hpick(hv[3], (-11.25, 0.0, 11.25))
    feather_inf = round(_hpick(hv[4], (0.0, 0.03, 0.06)), 3)

    neck_bone = Bone(
        "raptor_neck",
        rest_euler=(-25.0, 0.0, 0.0),
        attach=Attach("horizontal_thorax", (0.0, 0.55, -1.0), (0.0, -1.0, 0.5)),
        items=[
            Box("s_curve_cervical", (2, 4, 3), p_body.name),
            Bone(
                "raptor_head",
                rest_euler=(25.0, 0.0, 0.0),
                attach=Attach("s_curve_cervical", (0.0, 1.0, -0.35), (0.0, -1.0, 0.0)),
                items=[
                    Box("raptor_cranium", (3, 3, 3), p_body.name),
                    Box("narrow_snout_maxilla", (2, 2, 4), p_body.name, Place("raptor_cranium", (0.0, 0.1, -1.0), (0.0, 0.0, 1.0))),
                    Box("crown_feather_crest", (1, 2, 4), p_plume.name, Place("raptor_cranium", (0.0, 1.0, 0.35), (0.0, -1.0, -0.35))),
                    Bone(
                        "raptor_toothed_jaw",
                        rest_euler=(-jaw_deg, 0.0, 0.0),
                        attach=Attach("raptor_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                        items=[Box("toothed_dentary", (2, 1, 4), p_under.name)],
                    ),
                ],
            ),
        ],
    )

    wing_fan_bone = Bone(
        "arm",
        rest_euler=(18.0, 0.0, 22.5),
        attach=Attach("horizontal_thorax", (1.0, -0.2, -0.45), (-1.0, 0.5, 0.0)),
        mirror_name="arm_r",
        items=[
            Box("raptor_brachium", (2, 3, 2), p_body.name),
            Bone(
                "feather_fan",
                rest_euler=(-22.5, 0.0, 15.0),
                attach=Attach("raptor_brachium", (0.5, -1.0, 0.0), (-0.5, 1.0, 0.0)),
                items=[
                    Box("manus_claw_digits", (1, 2, 3), p_claw.name),
                    Box("primary_remiges_fan", (1, 3, 5), p_plume.name, Place("manus_claw_digits", (1.0, -0.45, 0.6), (-1.0, 0.45, -0.4))),
                ],
            ),
        ],
    )

    hindleg_bone = Bone(
        "raptor_thigh_l",
        rest_euler=(stride_deg, 0.0, 11.25),
        attach=Attach("agile_pelvis", (1.0, -0.25, 0.0), (-1.0, 0.5, 0.0)),
        mirror_name="raptor_thigh_r",
        items=[
            Box("feathered_femur", (2, 5, 3), p_body.name),
            Bone(
                "digitigrade_pes_l",
                rest_euler=(15.0, 0.0, -7.5),
                attach=Attach("feathered_femur", (0.0, -1.0, 0.15), (0.0, 1.0, 0.0)),
                items=[
                    Box("metatarsus_shank", (2, 5, 2), p_under.name),
                    Box("outer_running_toes", (2, 1, 4), p_claw.name, Place("metatarsus_shank", (0.2, -1.0, -0.45), (0.0, 1.0, 0.0))),
                    Box("raised_sickle_claw", (1, 2, 2), p_claw.name, Place("metatarsus_shank", (-1.0, -0.55, -0.6), (1.0, 0.0, 0.5))),
                ],
            ),
        ],
    )

    tail_bone = Bone(
        "stiff_tail_rod",
        rest_euler=(-5.0, tail_yaw, 0.0),
        attach=Attach("agile_pelvis", (0.0, 0.35, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("caudal_rod_prox", (2, 2, tail_len), p_body.name),
            Bone(
                "caudal_plumage_tip",
                rest_euler=(5.0, -tail_yaw, 0.0),
                attach=Attach("caudal_rod_prox", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("caudal_rod_dist", (1, 1, tail_len - 1), p_body.name),
                    Box("rectrices_tail_fan", (5, 1, 5), p_plume.name, Place("caudal_rod_dist", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("agile_pelvis", (4, 4, 5), p_body.name, inflate=feather_inf),
        Box("horizontal_thorax", (5, 4, 6), p_body.name, Place("agile_pelvis", (0.0, 0.1, -1.0), (0.0, 0.0, 1.0))),
        Box("dorsal_plumage_mantle", (3, 1, 5), p_plume.name, Place("horizontal_thorax", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        neck_bone,
        wing_fan_bone,
        hindleg_bone,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_body, p_plume, p_under, p_claw],
        root=Bone("raptor_pelvis_root", pivot_local=_jpiv(rng, (8.0, 13.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 11 + 2 * tail_len + 10,
        "secondary_span": 13,
        "joint_angle_deg": jaw_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": feather_inf,
        "material_theme": f"{p_body.name}+{p_plume.name}+{p_claw.name}",
        "notes": f"dromaeosaurid feathered raptor ({substyle.replace('_', ' ')}) with S-curved neck, crested toothed snout, mirrored forelimb feather-fans, digitigrade hindlegs with raised inner sickle claws, and stiffened plumed tail fan",
        "extra": {"tail_len": tail_len, "jaw_deg": jaw_deg},
    }


def _gen_winged_amphithere(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Legless Feathered Amphithere with hooded cobra neck, 2-tier avian wings & 3-segment tail (>= 20 boxes)."""
    substyles = (
        "quetzal_emerald_amphithere",
        "sun_plume_coatl",
        "sapphire_sky_amphithere",
        "opal_cloud_amphithere",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 7 + 5, dim=6, jitter_rng=rng)

    p_serpent = _pal(rng, _DRAKE_SCALES[7 if v_idx % 2 == 0 else 5])
    p_pinion = _pal(rng, _PLUMAGE_AND_MEMBRANES[2 + (v_idx % 3)], suffix="_pinions")
    p_scute = _pal(rng, _PLUMAGE_AND_MEMBRANES[5], suffix="_ventral")
    p_beak = _pal(rng, _KERATIN_AND_METALS[2], suffix="_crest")

    w_in = _hpick(hv[0], (6, 7))
    w_out = _hpick(hv[1], (6, 7))
    wing_dihedral = _hpick(hv[2], (18.0, 22.5, 30.0))
    seg_l = _hpick(hv[3], (5, 6))
    wave_deg = _hpick(hv[4], (15.0, 22.5))
    jaw_deg = _hpick(hv[5], (15.0, 22.5))
    plume_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    hood_neck = Bone(
        "hooded_cobra_neck",
        rest_euler=(-22.5, 0.0, 0.0),
        attach=Attach("anterior_serpent_chest", (0.0, 1.0, -0.65), (0.0, -1.0, 0.0)),
        items=[
            Box("cobra_cervical_column", (3, 6, 3), p_serpent.name),
            Box("feathered_hood_flare_l", (2, 3, 1), p_pinion.name, Place("cobra_cervical_column", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("feathered_hood_flare_r", (2, 3, 1), p_pinion.name, Place("cobra_cervical_column", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            Bone(
                "amphithere_head",
                rest_euler=(22.5, 0.0, 0.0),
                attach=Attach("cobra_cervical_column", (0.0, 1.0, -0.25), (0.0, -1.0, 0.0)),
                items=[
                    Box("avian_serpent_cranium", (4, 3, 4), p_serpent.name),
                    Box("fanged_beak_rostrum", (2, 2, 4), p_beak.name, Place("avian_serpent_cranium", (0.0, 0.1, -1.0), (0.0, 0.0, 1.0))),
                    Box("radiant_plume_crest", (2, 3, 3), p_pinion.name, Place("avian_serpent_cranium", (0.0, 1.0, 0.25), (0.0, -1.0, 0.0))),
                    Bone(
                        "amphithere_jaw",
                        rest_euler=(-jaw_deg, 0.0, 0.0),
                        attach=Attach("avian_serpent_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                        items=[Box("fanged_lower_beak", (2, 1, 3), p_scute.name)],
                    ),
                ],
            ),
        ],
    )

    wing_bone = Bone(
        "inner_wing",
        rest_euler=(0.0, 0.0, wing_dihedral),
        attach=Attach("anterior_serpent_chest", (1.0, 0.45, -0.1), (-1.0, 0.0, 0.0)),
        mirror_name="inner_wing_r",
        items=[
            Box("inner_covert_spar", (w_in, 1, 2), p_serpent.name),
            Box("secondary_feather_tier", (w_in - 2, 1, 3), p_pinion.name, Place("inner_covert_spar", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "outer_primary_pinions",
                rest_euler=(0.0, -11.25, -11.25),
                attach=Attach("inner_covert_spar", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                items=[
                    Box("outer_pinion_spar", (w_out, 1, 2), p_beak.name),
                    Box("primary_pinion_tier", (w_out - 2, 1, 5), p_pinion.name, Place("outer_pinion_spar", (0.2, 0.0, 1.0), (0.0, 0.0, -1.0))),
                ],
            ),
        ],
    )

    tail_s3 = Bone(
        "serpent_tail_s3",
        rest_euler=(7.5, wave_deg, 0.0),
        attach=Attach("amphithere_coil_s2", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("amphithere_coil_s3", (2, 2, seg_l - 1), p_serpent.name),
            Box("wide_feathered_tail_fan", (6, 1, 5), p_pinion.name, Place("amphithere_coil_s3", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("tail_streamer_ribbon", (2, 1, 4), p_beak.name, Place("wide_feathered_tail_fan", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    tail_s2 = Bone(
        "serpent_tail_s2",
        rest_euler=(-11.25, -wave_deg * 1.35, 0.0),
        attach=Attach("amphithere_coil_s1", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("amphithere_coil_s2", (3, 2, seg_l), p_serpent.name),
            tail_s3,
        ],
    )

    tail_s1 = Bone(
        "serpent_tail_s1",
        rest_euler=(7.5, wave_deg, 0.0),
        attach=Attach("anterior_serpent_chest", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("amphithere_coil_s1", (4, 3, seg_l), p_serpent.name),
            tail_s2,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("anterior_serpent_chest", (5, 4, 6), p_serpent.name, inflate=plume_inf),
        Box("ventral_iridescent_scutes", (3, 1, 5), p_scute.name, Place("anterior_serpent_chest", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("dorsal_feather_mantle", (2, 1, 3), p_pinion.name, Place("anterior_serpent_chest", (0.0, 1.0, 0.45), (0.0, -1.0, 0.0))),
        hood_neck,
        wing_bone,
        tail_s1,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_serpent, p_pinion, p_scute, p_beak],
        root=Bone("amphithere_hood_root", pivot_local=_jpiv(rng, (8.0, 14.0, 6.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 5 + 2 * (w_in + w_out),
        "secondary_span": 6 + 3 * seg_l + 9,
        "joint_angle_deg": wing_dihedral,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": plume_inf,
        "material_theme": f"{p_serpent.name}+{p_pinion.name}+{p_beak.name}",
        "notes": f"legless feathered amphithere ({substyle.replace('_', ' ')}) with cobra-hooded anterior neck, crested fanged beak, 2 multi-tiered avian wings, and 3-segment undulating serpent tail with wide plumage fan",
        "extra": {"w_in": w_in, "w_out": w_out, "seg_l": seg_l},
    }


def _gen_multi_headed_hydra(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Polycephalic 3-Headed Hydra with massive torso, 4 pillar legs & 3 non-colliding necks/jaws (>= 24 boxes)."""
    substyles = (
        "lernaean_swamp_hydra",
        "tri_fang_abyssal_hydra",
        "pyroclast_three_headed_hydra",
        "venom_marsh_polycephalic",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 11 + 2, dim=6, jitter_rng=rng)

    p_hide = _pal(rng, _DRAKE_SCALES[v_idx % len(_DRAKE_SCALES)])
    p_belly = _pal(rng, _PLUMAGE_AND_MEMBRANES[5], suffix="_plastron")
    p_spine = _pal(rng, _KERATIN_AND_METALS[v_idx % 2], suffix="_crest")
    p_venom = _pal(rng, _EMISSIVE_ACCENTS[v_idx % 2], emissive=True, suffix="_fang_glow")

    neck_h = _hpick(hv[0], (5, 6))
    flare_rz = _hpick(hv[1], (-28.0, -32.0, -35.0))
    flare_ry = _hpick(hv[2], (-20.0, -25.0))
    jaw_deg = _hpick(hv[3], (15.0, 22.5))
    tail_yaw = _hpick(hv[4], (-11.25, 0.0, 11.25))
    hide_inf = round(_hpick(hv[5], (0.0, 0.04, 0.07)), 3)

    # Center neck on (0, y, -1) pitched upward & forward along X = 0 midline
    center_neck = Bone(
        "center_neck",
        rest_euler=(-22.5, 0.0, 0.0),
        attach=Attach("massive_hydra_thorax", (0.0, 0.65, -1.0), (0.0, -1.0, 0.6)),
        items=[
            Box("center_cervical_low", (3, neck_h, 3), p_hide.name),
            Box("center_cervical_spine", (1, 2, 1), p_spine.name, Place("center_cervical_low", (0.0, 0.45, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "center_neck_upper",
                rest_euler=(-11.25, 0.0, 0.0),
                attach=Attach("center_cervical_low", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
                items=[
                    Box("center_cervical_up", (3, 4, 3), p_hide.name),
                    Bone(
                        "center_head",
                        rest_euler=(32.0, 0.0, 0.0),
                        attach=Attach("center_cervical_up", (0.0, 1.0, -0.25), (0.0, -1.0, 0.0)),
                        items=[
                            Box("center_cranium", (4, 3, 4), p_hide.name),
                            Box("center_snout", (3, 2, 4), p_hide.name, Place("center_cranium", (0.0, 0.15, -1.0), (0.0, 0.0, 1.0))),
                            Box("center_crown_fin", (1, 2, 3), p_venom.name, Place("center_cranium", (0.0, 1.0, 0.2), (0.0, -1.0, 0.0))),
                            Bone(
                                "center_jaw",
                                rest_euler=(-jaw_deg, 0.0, 0.0),
                                attach=Attach("center_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                                items=[Box("center_mandible", (3, 1, 4), p_belly.name)],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # Left & Right necks via mirror_x flared outward along +-X by Ry/Rz for generous lateral clearance
    lateral_neck = Bone(
        "left_neck",
        rest_euler=(-18.0, flare_ry, flare_rz),
        attach=Attach("massive_hydra_thorax", (0.85, 0.7, -0.7), (0.0, -1.0, 0.0)),
        mirror_name="right_neck",
        items=[
            Box("lateral_cervical", (3, neck_h + 1, 3), p_hide.name),
            Box("lateral_neck_spine", (1, 3, 1), p_spine.name, Place("lateral_cervical", (0.0, 0.15, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "lateral_head",
                rest_euler=(22.5, 0.0, 11.25),
                attach=Attach("lateral_cervical", (0.0, 1.0, -0.25), (0.0, -1.0, 0.0)),
                items=[
                    Box("lateral_cranium", (4, 3, 4), p_hide.name),
                    Box("lateral_snout", (3, 2, 4), p_hide.name, Place("lateral_cranium", (0.0, 0.15, -1.0), (0.0, 0.0, 1.0))),
                    Box("lateral_crest_horn", (1, 2, 3), p_venom.name, Place("lateral_cranium", (0.0, 1.0, 0.25), (0.0, -1.0, 0.0))),
                    Bone(
                        "lateral_jaw",
                        rest_euler=(-jaw_deg, 0.0, 0.0),
                        attach=Attach("lateral_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                        items=[Box("lateral_mandible", (3, 1, 4), p_belly.name)],
                    ),
                ],
            ),
        ],
    )

    fore_pillar = Bone(
        "fore_pillar_leg_l",
        rest_euler=(-7.5, 0.0, 11.25),
        attach=Attach("massive_hydra_thorax", (1.0, -0.5, -0.35), (-1.0, 0.5, 0.0)),
        mirror_name="fore_pillar_leg_r",
        items=[
            Box("fore_pillar_column", (3, 5, 3), p_hide.name),
            Box("fore_pillar_claws", (3, 2, 4), p_spine.name, Place("fore_pillar_column", (0.0, -1.0, -0.25), (0.0, 1.0, 0.0))),
        ],
    )

    hind_pillar = Bone(
        "hind_pillar_leg_l",
        rest_euler=(7.5, 0.0, 11.25),
        attach=Attach("hydra_pelvis", (1.0, -0.45, 0.2), (-1.0, 0.5, 0.0)),
        mirror_name="hind_pillar_leg_r",
        items=[
            Box("hind_pillar_column", (3, 5, 3), p_hide.name),
            Box("hind_pillar_claws", (3, 2, 4), p_spine.name, Place("hind_pillar_column", (0.0, -1.0, -0.25), (0.0, 1.0, 0.0))),
        ],
    )

    tail_bone = Bone(
        "hydra_tail_prox",
        rest_euler=(-7.5, tail_yaw, 0.0),
        attach=Attach("hydra_pelvis", (0.0, 0.1, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("tail_trunk_prox", (5, 4, 6), p_hide.name),
            Bone(
                "hydra_tail_dist",
                rest_euler=(7.5, -tail_yaw, 0.0),
                attach=Attach("tail_trunk_prox", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("tail_trunk_dist", (3, 3, 5), p_hide.name),
                    Box("twin_fluke_blade_l", (3, 1, 4), p_venom.name, Place("tail_trunk_dist", (1.0, 0.0, 0.75), (-1.0, 0.0, -0.5))),
                    Box("twin_fluke_blade_r", (3, 1, 4), p_venom.name, Place("tail_trunk_dist", (-1.0, 0.0, 0.75), (1.0, 0.0, -0.5))),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("massive_hydra_thorax", (10, 6, 8), p_hide.name, inflate=hide_inf),
        Box("hydra_ventral_plastron", (8, 1, 7), p_belly.name, Place("massive_hydra_thorax", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("hydra_dorsal_ridge", (4, 1, 6), p_spine.name, Place("massive_hydra_thorax", (0.0, 1.0, 0.2), (0.0, -1.0, 0.0))),
        Box("hydra_pelvis", (8, 5, 6), p_hide.name, Place("massive_hydra_thorax", (0.0, -0.1, 1.0), (0.0, 0.0, -1.0))),
        center_neck,
        lateral_neck,
        fore_pillar,
        hind_pillar,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_hide, p_belly, p_spine, p_venom],
        root=Bone("hydra_torso_root", pivot_local=_jpiv(rng, (8.0, 12.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 32,
        "secondary_span": 24,
        "joint_angle_deg": abs(flare_rz),
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": hide_inf,
        "material_theme": f"{p_hide.name}+{p_spine.name}+{p_venom.name}",
        "notes": f"polycephalic 3-headed hydra ({substyle.replace('_', ' ')}) on 4 pillar legs with twin-fluked tail, central upward neck, and mirrored lateral necks flared by ({flare_ry:g}, {flare_rz:g}) deg Ry/Rz with snapping jaws",
        "extra": {"neck_h": neck_h, "flare_rz": flare_rz, "flare_ry": flare_ry},
    }


def _gen_crowned_zilant(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Unseen Held-Out Mythic Crowned Zilant: bipedal crowned serpent-wyvern with 3-prong crown (>= 20 boxes)."""
    substyles = (
        "kazan_golden_crown_zilant",
        "obsidian_imperial_zilant",
        "emerald_heraldic_zilant",
        "ruby_wyrm_zilant",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 13 + 7, dim=6, jitter_rng=rng)

    p_scale = _pal(rng, _DRAKE_SCALES[(v_idx + 4) % len(_DRAKE_SCALES)])
    p_wing = _pal(rng, _PLUMAGE_AND_MEMBRANES[v_idx % 4], suffix="_wing")
    p_crown = _pal(rng, _KERATIN_AND_METALS[2 + (v_idx % 2)], suffix="_gold_crown")
    p_horn = _pal(rng, _KERATIN_AND_METALS[0], suffix="_horn")

    w_in = _hpick(hv[0], (5, 6))
    w_out = _hpick(hv[1], (6, 7))
    wing_dihedral = _hpick(hv[2], (18.0, 22.5, 30.0))
    jaw_deg = _hpick(hv[3], (15.0, 22.5))
    tail_yaw = _hpick(hv[4], (-15.0, 15.0))
    scale_inf = round(_hpick(hv[5], (0.0, 0.04, 0.06)), 3)

    crown_bone = Bone(
        "golden_three_prong_crown",
        rest_euler=(-5.0, 0.0, 0.0),
        attach=Attach("zilant_cranium", (0.0, 1.0, -0.15), (0.0, -1.0, 0.0)),
        items=[
            Box("crown_circlet_band", (3, 1, 3), p_crown.name),
            Box("crown_prong_center", (1, 2, 1), p_crown.name, Place("crown_circlet_band", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
            Box("crown_prong_left", (1, 2, 1), p_crown.name, Place("crown_circlet_band", (1.0, 1.0, 0.35), (1.0, -1.0, 0.0))),
            Box("crown_prong_right", (1, 2, 1), p_crown.name, Place("crown_circlet_band", (-1.0, 1.0, 0.35), (-1.0, -1.0, 0.0))),
        ],
    )

    neck_bone = Bone(
        "zilant_neck",
        rest_euler=(-18.0, 0.0, 0.0),
        attach=Attach("zilant_avian_thorax", (0.0, 1.0, -0.55), (0.0, -1.0, 0.0)),
        items=[
            Box("zilant_cervical", (3, 5, 3), p_scale.name),
            Bone(
                "zilant_head",
                rest_euler=(18.0, 0.0, 0.0),
                attach=Attach("zilant_cervical", (0.0, 1.0, -0.25), (0.0, -1.0, 0.0)),
                items=[
                    Box("zilant_cranium", (4, 3, 4), p_scale.name),
                    Box("zilant_draconic_snout", (2, 2, 4), p_scale.name, Place("zilant_cranium", (0.0, 0.15, -1.0), (0.0, 0.0, 1.0))),
                    Box("zilant_horn_l", (1, 1, 3), p_horn.name, Place("zilant_cranium", (1.0, 0.55, 0.45), (-1.0, 0.0, -0.5))),
                    Box("zilant_horn_r", (1, 1, 3), p_horn.name, Place("zilant_cranium", (-1.0, 0.55, 0.45), (1.0, 0.0, -0.5))),
                    crown_bone,
                    Bone(
                        "zilant_jaw",
                        rest_euler=(-jaw_deg, 0.0, 0.0),
                        attach=Attach("zilant_cranium", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0)),
                        items=[Box("zilant_mandible", (2, 1, 4), p_horn.name)],
                    ),
                ],
            ),
        ],
    )

    wing_bone = Bone(
        "zilant_wing_in_l",
        rest_euler=(0.0, 0.0, wing_dihedral),
        attach=Attach("zilant_avian_thorax", (1.0, 0.65, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="zilant_wing_in_r",
        items=[
            Box("zilant_wing_arm", (w_in, 1, 2), p_scale.name),
            Box("zilant_covert_membrane", (w_in - 2, 1, 4), p_wing.name, Place("zilant_wing_arm", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "zilant_wing_out_l",
                rest_euler=(0.0, -11.25, -15.0),
                attach=Attach("zilant_wing_arm", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                items=[
                    Box("zilant_pinion_spar", (w_out, 1, 2), p_scale.name),
                    Box("zilant_leathery_pinion", (w_out - 2, 1, 4), p_wing.name, Place("zilant_pinion_spar", (0.2, 0.0, 1.0), (0.0, 0.0, -1.0))),
                ],
            ),
        ],
    )

    avian_leg = Bone(
        "zilant_leg_l",
        rest_euler=(-11.25, 0.0, 11.25),
        attach=Attach("zilant_pelvic_arch", (1.0, -0.35, -0.1), (-1.0, 0.5, 0.0)),
        mirror_name="zilant_leg_r",
        items=[
            Box("scaled_avian_thigh", (2, 4, 3), p_scale.name),
            Bone(
                "zilant_talon_shank_l",
                rest_euler=(18.0, 0.0, -7.5),
                attach=Attach("scaled_avian_thigh", (0.0, -1.0, 0.1), (0.0, 1.0, 0.0)),
                items=[
                    Box("avian_tarsus_scale", (2, 4, 2), p_crown.name),
                    Box("heraldic_avian_talons", (3, 1, 4), p_horn.name, Place("avian_tarsus_scale", (0.0, -1.0, -0.35), (0.0, 1.0, 0.0))),
                ],
            ),
        ],
    )

    tail_bone = Bone(
        "zilant_tail_s1",
        rest_euler=(-15.0, tail_yaw, 0.0),
        attach=Attach("zilant_pelvic_arch", (0.0, 0.2, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("zilant_coil_s1", (3, 3, 5), p_scale.name),
            Bone(
                "zilant_tail_s2",
                rest_euler=(22.5, -tail_yaw * 1.25, 0.0),
                attach=Attach("zilant_coil_s1", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("zilant_coil_s2", (2, 2, 5), p_scale.name),
                    Bone(
                        "zilant_tail_s3",
                        rest_euler=(-15.0, tail_yaw, 0.0),
                        attach=Attach("zilant_coil_s2", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                        items=[
                            Box("zilant_coil_s3", (2, 2, 4), p_scale.name),
                            Box("arrow_spade_barb", (3, 1, 3), p_crown.name, Place("zilant_coil_s3", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                        ],
                    ),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("zilant_avian_thorax", (6, 5, 6), p_scale.name, inflate=scale_inf),
        Box("heraldic_breastplate_scales", (4, 4, 1), p_crown.name, Place("zilant_avian_thorax", (0.0, -0.1, -1.0), (0.0, 0.0, 1.0))),
        Box("zilant_pelvic_arch", (4, 4, 5), p_scale.name, Place("zilant_avian_thorax", (0.0, -0.15, 1.0), (0.0, 0.0, -1.0))),
        neck_bone,
        wing_bone,
        avian_leg,
        tail_bone,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_scale, p_wing, p_crown, p_horn],
        root=Bone("zilant_thorax_root", pivot_local=_jpiv(rng, (8.0, 14.0, 7.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 6 + 2 * (w_in + w_out),
        "secondary_span": 28,
        "joint_angle_deg": wing_dihedral,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": scale_inf,
        "material_theme": f"{p_scale.name}+{p_crown.name}+{p_wing.name}",
        "notes": f"mythic crowned winged serpent ({substyle.replace('_', ' ')}) with breastplate scales, 2 scaled avian legs, 2 feathered or leathery wings, horned dragon head topped by golden 3-prong crown, and 3-segment curved serpent tail",
        "extra": {"w_in": w_in, "w_out": w_out, "jaw_deg": jaw_deg},
    }


# ==============================================================================
# B. ANIMAL, NATURAL & BUILDING EXPANSION (3 Seen Generators)
# ==============================================================================


def _gen_articulated_manta_shark(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Pelagic aquatic swimmer: manta glider / hammerhead shark / abyssal chimaera (>= 18 boxes)."""
    substyles = (
        "reef_manta_glider",
        "hammerhead_pelagic_shark",
        "abyssal_chimaera",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _entity_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 5 + 9, dim=6, jitter_rng=rng)

    p_dorsum = _pal(rng, _PELAGIC_PALETTES[v_idx % 3])
    p_belly = _pal(rng, _PELAGIC_PALETTES[3], suffix="_ventral")
    p_fin = _pal(rng, _PELAGIC_PALETTES[4], emissive=(substyle == "abyssal_chimaera"), suffix="_foil")
    p_cartilage = _pal(rng, _KERATIN_AND_METALS[0], suffix="_rostrum")

    body_w = 7 if substyle == "reef_manta_glider" else _hpick(hv[0], (5, 6))
    body_l = _hpick(hv[1], (7, 8))
    pec_in = 6 if substyle == "reef_manta_glider" else _hpick(hv[2], (4, 5))
    pec_out = 5 if substyle == "reef_manta_glider" else _hpick(hv[3], (4, 5))
    fin_dihedral = _hpick(hv[4], (-15.0, -11.25, 11.25, 15.0))
    tail_yaw = _hpick(hv[5], (11.25, 15.0, 22.5))
    jaw_deg = rng.choice((11.25, 15.0))
    skin_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    ceph_w = 3 if substyle == "hammerhead_pelagic_shark" else 2
    ceph_tip_l = 3 if substyle == "reef_manta_glider" else 2
    ceph_yaw = -15.0 if substyle == "reef_manta_glider" else 0.0

    cephalic_lobe = Bone(
        "cephalic_lobe_l",
        rest_euler=(0.0, ceph_yaw, -7.5),
        attach=Attach("rostral_hydro_snout", (1.0, 0.0, -0.35), (-1.0, 0.0, 0.0)),
        mirror_name="cephalic_lobe_r",
        items=[
            Box("cephalic_stalk_foil", (ceph_w, 1, 2), p_dorsum.name),
            Box("cephalic_ampullae_tip", (1, 1, ceph_tip_l), p_fin.name, Place("cephalic_stalk_foil", (1.0, 0.0, -0.35), (-1.0, 0.0, 0.35))),
        ],
    )

    ventral_jaw = Bone(
        "ventral_pelagic_jaw",
        rest_euler=(-jaw_deg, 0.0, 0.0),
        attach=Attach("fusiform_hydrofoil_disc", (0.0, -0.85, -1.0), (0.0, 1.0, 1.0)),
        items=[
            Box("cartilage_mandible", (3, 1, 3), p_cartilage.name),
        ],
    )

    dorsal_fin = Bone(
        "tall_dorsal_fin",
        rest_euler=(15.0, 0.0, 0.0),
        attach=Attach("fusiform_hydrofoil_disc", (0.0, 1.0, 0.05), (0.0, -1.0, 0.0)),
        items=[
            Box("dorsal_fin_base", (1, 4, 4), p_dorsum.name),
            Box("dorsal_fin_apex", (1, 2, 2), p_fin.name, Place("dorsal_fin_base", (0.0, 1.0, 0.35), (0.0, -1.0, 0.0))),
        ],
    )

    pectoral_wing = Bone(
        "inner_pectoral",
        rest_euler=(0.0, 0.0, fin_dihedral),
        attach=Attach("fusiform_hydrofoil_disc", (1.0, -0.1, 0.05), (-1.0, 0.0, 0.0)),
        mirror_name="inner_pectoral_r",
        items=[
            Box("inner_pectoral_foil", (pec_in, 1, 5), p_dorsum.name),
            Box("pectoral_trailing_lobe", (pec_in - 2, 1, 2), p_belly.name, Place("inner_pectoral_foil", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Bone(
                "outer_fin_tip",
                rest_euler=(0.0, -11.25, -fin_dihedral * 0.8),
                attach=Attach("inner_pectoral_foil", (1.0, 0.0, -0.2), (-1.0, 0.0, 0.0)),
                items=[
                    Box("outer_wing_fin", (pec_out, 1, 3), p_fin.name),
                    Box("swept_fin_apex", (2, 1, 2), p_fin.name, Place("outer_wing_fin", (1.0, 0.0, 0.35), (-1.0, 0.0, 0.0))),
                ],
            ),
        ],
    )

    caudal_tail = Bone(
        "caudal_peduncle_prox",
        rest_euler=(0.0, tail_yaw, 0.0),
        attach=Attach("fusiform_hydrofoil_disc", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("peduncle_seg_prox", (3, 3, 5), p_dorsum.name),
            Box("second_dorsal_finlet", (1, 2, 2), p_fin.name, Place("peduncle_seg_prox", (0.0, 1.0, 0.2), (0.0, -1.0, 0.0))),
            Bone(
                "caudal_fluke_dist",
                rest_euler=(0.0, -tail_yaw * 1.35, 0.0),
                attach=Attach("peduncle_seg_prox", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
                items=[
                    Box("peduncle_seg_dist", (2, 2, 5), p_dorsum.name),
                    Box("lunate_fluke_dorsal", (1, 4, 3), p_fin.name, Place("peduncle_seg_dist", (0.0, 1.0, 0.75), (0.0, -1.0, -0.5))),
                    Box("lunate_fluke_ventral", (1, 3, 3), p_fin.name, Place("peduncle_seg_dist", (0.0, -1.0, 0.75), (0.0, 1.0, -0.5))),
                ],
            ),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("fusiform_hydrofoil_disc", (body_w, 4, body_l), p_dorsum.name, inflate=skin_inf),
        Box("rostral_hydro_snout", (4, 2, 3), p_dorsum.name, Place("fusiform_hydrofoil_disc", (0.0, 0.1, -1.0), (0.0, 0.0, 1.0))),
        Box("ventral_gill_plate", (body_w - 2, 1, body_l - 2), p_belly.name, Place("fusiform_hydrofoil_disc", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        cephalic_lobe,
        ventral_jaw,
        dorsal_fin,
        pectoral_wing,
        caudal_tail,
    ]

    prog = Program(
        meta=Meta("entity", profile, density),
        palettes=[p_dorsum, p_belly, p_fin, p_cartilage],
        root=Bone("pelagic_fusiform_root", pivot_local=_jpiv(rng, (8.0, 11.0, 7.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": body_w + 2 * (pec_in + pec_out + 2),
        "secondary_span": body_l + 16,
        "joint_angle_deg": tail_yaw,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": skin_inf,
        "material_theme": f"{p_dorsum.name}+{p_belly.name}+{p_fin.name}",
        "notes": f"pelagic aquatic swimmer ({substyle.replace('_', ' ')}) with fusiform hydrofoil body, mirrored cephalic lobes, ventral jaw, tall dorsal fin, 2-stage pectoral wing-fins, and 2-segment yawing lunate caudal tail",
        "extra": {"body_w": body_w, "pec_in": pec_in, "pec_out": pec_out},
    }


def _gen_pine_conifer_evergreen(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Rooted alpine pine / spruce / cedar centered near (8, 1, 8) inside [-16, 32] (18-22 boxes)."""
    substyles = (
        "alpine_spruce_evergreen",
        "boreal_pine_spire",
        "subalpine_cedar_conifer",
        "snowcap_timberline_fir",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 7 + 8, dim=6, jitter_rng=rng)

    p_bark = _pal(rng, _CONIFER_AND_EARTH[3 + (v_idx % 2)])
    p_needle_low = _pal(rng, _CONIFER_AND_EARTH[v_idx % 3], suffix="_low")
    p_needle_up = _pal(rng, _CONIFER_AND_EARTH[(v_idx + 1) % 3], suffix="_crown")
    p_mound = _pal(rng, _CONIFER_AND_EARTH[5], suffix="_mound")

    mound_w = _hpick(hv[0], (8, 10))
    low_h = _hpick(hv[1], (4, 5))
    mid_h = _hpick(hv[2], (4, 5))
    up_h = _hpick(hv[3], (3, 4))
    droop_deg = _hpick(hv[4], (7.5, 11.25, 15.0))
    sway_deg = _hpick(hv[5], (-5.0, 0.0, 5.0))
    bark_inf = round(rng.choice((0.0, 0.03, 0.05)), 3)

    # Tier 1: Lowest, widest conifer whorl on lower_resinous_bole (+-X mirrored + N/S bough bones)
    whorl_tier1_lr = Bone(
        "whorl_tier1_bough_l",
        rest_euler=(0.0, 0.0, -droop_deg),
        attach=Attach("lower_resinous_bole", (1.0, 0.7, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="whorl_tier1_bough_r",
        items=[
            Box("tier1_needle_fan", (4, 2, 7), p_needle_low.name),
            Box("tier1_fringe_tip", (2, 1, 5), p_needle_up.name, Place("tier1_needle_fan", (1.0, -0.25, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    whorl_tier1_n = Bone(
        "whorl_tier1_bough_n",
        rest_euler=(-droop_deg, 0.0, 0.0),
        attach=Attach("lower_resinous_bole", (0.0, 0.7, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("tier1_needle_north", (2, 2, 4), p_needle_low.name),
        ],
    )

    whorl_tier1_s = Bone(
        "whorl_tier1_bough_s",
        rest_euler=(droop_deg, 0.0, 0.0),
        attach=Attach("lower_resinous_bole", (0.0, 0.7, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("tier1_needle_south", (2, 2, 4), p_needle_low.name),
        ],
    )

    # Tier 2: Mid conifer whorl on mid_resinous_trunk (+-X mirrored drooping boughs)
    whorl_tier2_lr = Bone(
        "whorl_tier2_bough_l",
        rest_euler=(0.0, 0.0, -droop_deg),
        attach=Attach("mid_resinous_trunk", (1.0, 0.65, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="whorl_tier2_bough_r",
        items=[
            Box("tier2_needle_fan", (3, 2, 5), p_needle_up.name),
            Box("tier2_pinecone_cluster", (1, 1, 2), p_bark.name, Place("tier2_needle_fan", (0.7, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    # Tier 3: Upper crown whorl + apical spire atop upper_trunk_leader
    crown_items: list[Box | Bone] = [
        Box("tier3_crown_whorl", (6, 2, 6), p_needle_low.name),
        Box("tier3_upper_cone", (4, 2, 4), p_needle_up.name, Place("tier3_crown_whorl", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("apical_needle_spire", (2, 3, 2), p_needle_up.name, Place("tier3_upper_cone", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
    ]
    if v_idx % 2 == 0:
        crown_items.append(
            Box("apical_leader_shoot", (1, 2, 1), p_needle_low.name, Place("apical_needle_spire", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)))
        )

    upper_crown_bone = Bone(
        "upper_crown_spire_rig",
        rest_euler=(sway_deg, 0.0, 0.0),
        attach=Attach("upper_trunk_leader", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=crown_items,
    )

    root_items: list[Box | Bone] = [
        Box("stone_moss_mound", (mound_w, 2, mound_w), p_mound.name),
        Box("root_buttress_n", (2, 2, 2), p_bark.name, Place("stone_moss_mound", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        Box("root_buttress_s", (2, 2, 2), p_bark.name, Place("stone_moss_mound", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("root_buttress_e", (2, 2, 2), p_bark.name, Place("stone_moss_mound", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        Box("root_buttress_w", (2, 2, 2), p_bark.name, Place("stone_moss_mound", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
        Box("lower_resinous_bole", (4, low_h, 4), p_bark.name, Place("stone_moss_mound", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), inflate=bark_inf),
        Box("mid_resinous_trunk", (3, mid_h, 3), p_bark.name, Place("lower_resinous_bole", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("upper_trunk_leader", (2, up_h, 2), p_bark.name, Place("mid_resinous_trunk", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        whorl_tier1_lr,
        whorl_tier1_n,
        whorl_tier1_s,
        whorl_tier2_lr,
        upper_crown_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_bark, p_needle_low, p_needle_up, p_mound],
        root=Bone("conifer_root_origin", pivot_local=_jpiv(rng, (8.0, 1.0, 8.0), span=0.35), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 2 + low_h + mid_h + up_h + 8,
        "secondary_span": 16,
        "joint_angle_deg": droop_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": bark_inf,
        "material_theme": f"{p_bark.name}+{p_needle_low.name}+{p_mound.name}",
        "notes": f"rooted alpine conifer ({substyle.replace('_', ' ')}) on stone/moss mound with 4 root buttresses, 3-stage tapered resinous trunk, 3 graduated drooping needle whorls ({droop_deg:g} deg droop), and apical spire",
        "extra": {"mound_w": mound_w, "droop_deg": droop_deg, "low_h": low_h, "mid_h": mid_h},
    }


def _gen_waterwheel_millhouse(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    """Half-timbered stone & oak watermill centered near (8, 1, 8) inside [-16, 32] (18-22 boxes)."""
    substyles = (
        "undershot_brook_gristmill",
        "half_timber_river_millhouse",
        "stone_sluice_watermill",
        "oak_wheel_miller_cottage",
    )
    substyle = substyles[v_idx % len(substyles)]
    profile, density = _static_profile_and_density(v_idx)
    hv = halton_vector(v_idx * 11 + 4, dim=5, jitter_rng=rng)

    p_stone = _pal(rng, _MILLHOUSE_PALETTES[0])
    p_timber = _pal(rng, _MILLHOUSE_PALETTES[2], suffix="_oak")
    p_wheel = _pal(rng, _MILLHOUSE_PALETTES[3], suffix="_wheel")
    p_roof = _pal(rng, _MILLHOUSE_PALETTES[4 + (v_idx % 2)], suffix="_roof")
    p_lamp = _pal(rng, _EMISSIVE_ACCENTS[3], emissive=True, suffix="_window")

    side_sign = 1.0 if (v_idx % 2 == 0) else -1.0
    wall_h = _hpick(hv[0], (6, 7))
    wheel_roll_deg = _hpick(hv[1], (-45.0, -30.0, -22.5, -15.0, 0.0, 15.0, 22.5, 30.0, 45.0))
    roof_pitch = _hpick(hv[2], (22.5, 30.0))
    spoke_len = _hpick(hv[3], (2, 3))
    beam_inf = round(_hpick(hv[4], (0.0, 0.03, 0.05)), 3)

    # Articulated side-mounted paddle waterwheel protruding cleanly from +-X wall:
    # axle_journal_boss spans [x_wall, x_wall + 2], so axle_hub + 4 spokes/paddles lie strictly in
    # [x_wall + 2, x_wall + 4] with y_min >= 1.78 > millrace_bed_slab top (0.5) at ALL roll angles!
    waterwheel_bone = Bone(
        "paddle_waterwheel",
        rest_euler=(wheel_roll_deg, 0.0, 0.0),
        attach=Attach("axle_journal_boss", (side_sign * 1.0, 0.0, 0.0), (-side_sign * 1.0, 0.0, 0.0)),
        items=[
            Box("axle_hub", (2, 2, 2), p_timber.name),
            Box("wheel_spoke_up", (1, spoke_len, 1), p_wheel.name, Place("axle_hub", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("paddle_float_up", (2, 1, 3), p_wheel.name, Place("wheel_spoke_up", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("wheel_spoke_down", (1, spoke_len, 1), p_wheel.name, Place("axle_hub", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("paddle_float_down", (2, 1, 3), p_wheel.name, Place("wheel_spoke_down", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("wheel_spoke_north", (1, 1, spoke_len), p_wheel.name, Place("axle_hub", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("paddle_float_north", (2, 3, 1), p_wheel.name, Place("wheel_spoke_north", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("wheel_spoke_south", (1, 1, spoke_len), p_wheel.name, Place("axle_hub", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("paddle_float_south", (2, 3, 1), p_wheel.name, Place("wheel_spoke_south", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        ],
    )

    roof_slope_n = Bone(
        "gable_roof_slope_n",
        rest_euler=(-roof_pitch, 0.0, 0.0),
        attach=Attach("attic_gable_core", (0.0, 0.85, -1.0), (0.0, 0.0, 1.0)),
        items=[
            Box("north_shingle_rafter", (8, 1, 4), p_roof.name),
        ],
    )

    roof_slope_s = Bone(
        "gable_roof_slope_s",
        rest_euler=(roof_pitch, 0.0, 0.0),
        attach=Attach("attic_gable_core", (0.0, 0.85, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("south_shingle_rafter", (8, 1, 4), p_roof.name),
        ],
    )

    roof_truss_bone = Bone(
        "gable_roof_truss",
        attach=Attach("timber_frame_hall", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("attic_gable_core", (8, 3, 4), p_timber.name, inflate=beam_inf),
            Box("hearth_stone_chimney", (2, 4, 2), p_stone.name, Place("attic_gable_core", (-side_sign * 0.65, 1.0, 0.0), (0.0, -1.0, 0.0))),
            roof_slope_n,
            roof_slope_s,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("stone_mill_plinth", (8, 3, 10), p_stone.name),
        Box("millrace_bed_slab", (6, 1, 10), p_stone.name, Place("stone_mill_plinth", (side_sign * 1.0, -1.0, 0.0), (-side_sign * 1.0, -1.0, 0.0))),
        Box("millrace_outer_wall", (1, 3, 10), p_stone.name, Place("millrace_bed_slab", (side_sign * 1.0, 1.0, 0.0), (side_sign * 1.0, -1.0, 0.0))),
        Box("sluice_weir_post", (1, 2, 2), p_timber.name, Place("millrace_outer_wall", (0.0, 1.0, -0.75), (0.0, -1.0, 0.0))),
        Box("entry_porch_step", (3, 1, 2), p_stone.name, Place("stone_mill_plinth", (0.0, -0.5, 1.0), (0.0, 0.0, -1.0))),
        Box("timber_frame_hall", (8, wall_h, 10), p_timber.name, Place("stone_mill_plinth", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("front_door_canopy", (4, 1, 2), p_roof.name, Place("timber_frame_hall", (0.0, -0.25, 1.0), (0.0, 0.0, -1.0))),
        Box("gable_lantern_window", (2, 2, 1), p_lamp.name, Place("timber_frame_hall", (0.0, 0.45, 1.0), (0.0, 0.0, -1.0))),
        Box("axle_journal_boss", (2, 2, 2), p_stone.name, Place("timber_frame_hall", (side_sign * 1.0, 0.15, 0.0), (-side_sign * 1.0, 0.0, 0.0))),
        waterwheel_bone,
        roof_truss_bone,
    ]

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_stone, p_timber, p_wheel, p_roof, p_lamp],
        root=Bone("millhouse_foundation_root", pivot_local=_jpiv(rng, (8.0, 1.0, 8.0), span=0.35), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 3 + wall_h + 7,
        "secondary_span": 14,
        "joint_angle_deg": wheel_roll_deg,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": beam_inf,
        "material_theme": f"{p_stone.name}+{p_timber.name}+{p_wheel.name}",
        "notes": f"half-timbered stone & oak watermill ({substyle.replace('_', ' ')}) with stone foundation, millrace sluice channel wall, pitched gable roof + chimney, and side-mounted 4-paddle rotary waterwheel ({wheel_roll_deg:g} deg roll on {'+X' if side_sign > 0 else '-X'} wall)",
        "extra": {"side_sign": side_sign, "wheel_roll_deg": wheel_roll_deg, "spoke_len": spoke_len},
    }


DRAGON_AND_FAUNA_DISPATCH: dict[str, Any] = {
    # A. Dragon & Mythic Creature Domain (7 Seen + 1 Unseen)
    "quadruped_armored_drake": _gen_quadruped_armored_drake,
    "eastern_lung_dragon": _gen_eastern_lung_dragon,
    "bipedal_wyvern": _gen_bipedal_wyvern,
    "forelimb_lindworm": _gen_forelimb_lindworm,
    "feathered_raptor": _gen_feathered_raptor,
    "winged_amphithere": _gen_winged_amphithere,
    "multi_headed_hydra": _gen_multi_headed_hydra,
    "crowned_zilant": _gen_crowned_zilant,
    # B. Animal, Natural & Building Expansion (3 Seen)
    "articulated_manta_shark": _gen_articulated_manta_shark,
    "pine_conifer_evergreen": _gen_pine_conifer_evergreen,
    "waterwheel_millhouse": _gen_waterwheel_millhouse,
}
