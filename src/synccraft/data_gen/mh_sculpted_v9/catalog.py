"""Canonical Monster Hunter 22-Specimen Roster, Lineages, Splits & Artisan Motifs (V9).

Every specimen is authored individually stage-by-stage in native Block-DSL v4.1
(`dsl/<id>/stage1_blockout.blockdsl` -> `stage2_secondary.blockdsl` -> `<id>.blockdsl`)
with zero shared coordinate stencils and strict `research_nc_fan_reference` quarantine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MonsterSpec:
    id: str
    title: str
    title_epithet: str
    species_class: str
    variant_kind: str
    parent_species_id: str | None
    wiki_url: str
    split: str
    is_winged: bool
    signature_gif_clip: str
    artisan_motif: str
    accent_rgb: tuple[int, int, int]
    anatomy_summary: str


CANONICAL_22_SPECS: tuple[MonsterSpec, ...] = (
    # --- 1. Rathalos Lineage (Train) ---
    MonsterSpec(
        id="rathalos",
        title="Rathalos",
        title_epithet="King of the Skies",
        species_class="Flying Wyvern",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Rathalos",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="aerial_venom_talon_swoop",
        artisan_motif="rathalos_crimson",
        accent_rgb=(255, 168, 42),
        anatomy_summary=(
            "Raptorial biped flying wyvern: hooked eagle-beak wedged snout, swept obsidian crown & ear-frill horns, "
            "mandibular chin spike, broad 4-finger non-uni-sheet flame-scroll wing sails, digitigrade venom talons, "
            "and flared spiked mace tail."
        ),
    ),
    MonsterSpec(
        id="azure_rathalos",
        title="Azure Rathalos",
        title_epithet="Cerulean Sky Sovereign",
        species_class="Flying Wyvern",
        variant_kind="subspecies",
        parent_species_id="rathalos",
        wiki_url="https://monsterhunter.fandom.com/wiki/Azure_Rathalos",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="aerial_venom_talon_swoop",
        artisan_motif="rathalos_azure",
        accent_rgb=(96, 232, 168),
        anatomy_summary=(
            "High-altitude cerulean subspecies: cobalt-sapphire carapace, elongated swept temporal crest prongs, "
            "wider emerald-tinted scroll wing membranes, streamlined aerodynamic chest keel, and razor tail club."
        ),
    ),
    MonsterSpec(
        id="silver_rathalos",
        title="Silver Rathalos",
        title_epithet="Incandescent Silver Sun",
        species_class="Flying Wyvern",
        variant_kind="rare_species",
        parent_species_id="rathalos",
        wiki_url="https://monsterhunter.fandom.com/wiki/Silver_Rathalos",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="incandescent_hellfire_dive",
        artisan_motif="rathalos_silver",
        accent_rgb=(92, 186, 255),
        anatomy_summary=(
            "Rare metallic wyvern: gleaming platinum-silver overlapping armored shell, incandescent blue-white "
            "hellfire throat/chest fissures, reinforced brow crown, charcoal-indigo wings, and heavy double-spiked mace tail."
        ),
    ),
    # --- 2. Legiana Lineage (Validation) ---
    MonsterSpec(
        id="legiana",
        title="Legiana",
        title_epithet="Apex Glider of Coral Highlands",
        species_class="Flying Wyvern",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Legiana_Ecology",
        split="research_nc/val",
        is_winged=True,
        signature_gif_clip="glacial_corkscrew_spiral",
        artisan_motif="legiana_cryo",
        accent_rgb=(110, 224, 255),
        anatomy_summary=(
            "Ultra-sleek high-altitude glider wyvern: narrow viper-wedge skull, twin pleated cream-gold antennae "
            "head-vanes, swan S-neck, expansive leaf-lobed split kite-wings + secondary forearm patagia, and "
            "retractable 3-lobed swallowtail stabilizer fan."
        ),
    ),
    MonsterSpec(
        id="shrieking_legiana",
        title="Shrieking Legiana",
        title_epithet="Rime-Crowned Gale Wyvern",
        species_class="Flying Wyvern",
        variant_kind="variant",
        parent_species_id="legiana",
        wiki_url="https://monsterhunter.fandom.com/wiki/Shrieking_Legiana",
        split="research_nc/val",
        is_winged=True,
        signature_gif_clip="glacial_corkscrew_spiral",
        artisan_motif="legiana_rime",
        accent_rgb=(156, 242, 255),
        anatomy_summary=(
            "Hoarfrost master-rank variant: white rime-frost crystal blades encrusting longer swept head-vanes, "
            "serrated ice-spiked wing leading edges, frosted chest freezer-sac plates, and enlarged jagged cryo tail fan."
        ),
    ),
    # --- 3. Rey Dau (Train) ---
    MonsterSpec(
        id="rey_dau",
        title="Rey Dau",
        title_epithet="Fulminous Railgun Apex",
        species_class="Flying Wyvern",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Rey_Dau",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="railgun_horn_convergence_blast",
        artisan_motif="rey_dau_fulgurite",
        accent_rgb=(78, 198, 255),
        anatomy_summary=(
            "Windward Plains electromagnetic wyvern: articulating twin caliper railgun horns that lock forward into "
            "a lightning barrel, glowing blue neck capacitor gills, thorny burnished desert-gold plates, sweeping "
            "crescent scimitar wings with hanging cyan fulgurite glass scythes, and forked capacitor tail."
        ),
    ),
    # --- 4. Arkveld (Test Unseen) ---
    MonsterSpec(
        id="arkveld",
        title="Arkveld",
        title_epithet="The White Wraith",
        species_class="Flying Wyvern",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Arkveld",
        split="research_nc/test_unseen",
        is_winged=True,
        signature_gif_clip="chainblade_whip_cataclysm",
        artisan_motif="arkveld_wraith",
        accent_rgb=(218, 84, 255),
        anatomy_summary=(
            "Extinct-lineage White Wraith: heavy ram-horned pale slate-ivory skull, billowing fibrous silver-white "
            "shoulder/wing pelage, armored wing-arms bearing twin multi-link articulated chainblade flails surging "
            "with crimson-violet dragon energy, and muscular brawling stance."
        ),
    ),
    # --- 5. Zinogre Lineage (Train) ---
    MonsterSpec(
        id="zinogre",
        title="Zinogre",
        title_epithet="Lord of Usurpers",
        species_class="Fanged Wyvern",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Zinogre",
        split="research_nc/train",
        is_winged=False,
        signature_gif_clip="fulgur_backflip_tail_slam",
        artisan_motif="zinogre_thunder",
        accent_rgb=(88, 240, 255),
        anatomy_summary=(
            "Front-heavy lupine fanged wyvern (`zyhfisme` style anchor): wedged storm-teal wolf snout, twin swept "
            "golden-ochreridged horns, jagged jaw tusks, cascading directional golden-slate carapace shingles "
            "interleaved with alpha-cut white shock-fur tufts, massive planted forelimbs, glowing cyan fulgurbug "
            "charge seams, and heavy upturned fulgur-tail."
        ),
    ),
    MonsterSpec(
        id="stygian_zinogre",
        title="Stygian Zinogre",
        title_epithet="Hellwolf Fanged Wyvern",
        species_class="Fanged Wyvern",
        variant_kind="subspecies",
        parent_species_id="zinogre",
        wiki_url="https://monsterhunter.fandom.com/wiki/Stygian_Zinogre",
        split="research_nc/train",
        is_winged=False,
        signature_gif_clip="dracophage_backflip_slam",
        artisan_motif="zinogre_stygian",
        accent_rgb=(255, 48, 68),
        anatomy_summary=(
            "Draconic Hellwolf subspecies: volcanic charcoal-obsidian directional armor plates cracked with glowing "
            "crimson Dracophage lightning veins, stark alabaster-silver mane & forearm shock-fur ruffs, sharper "
            "hooked dark-silver horns, and crimson-taloned crushing forepaws."
        ),
    ),
    # --- 6. Gore Magala Lineage (Train) ---
    MonsterSpec(
        id="gore_magala",
        title="Gore Magala",
        title_epithet="Black Eclipse Wyvern",
        species_class="Elder Dragon",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Gore_Magala",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="frenzy_eclipse_cloak_slam",
        artisan_motif="gore_eclipse",
        accent_rgb=(196, 88, 255),
        anatomy_summary=(
            "Blind six-limbed Demi-Elder: eyeless faceted obsidian-indigo visor skull, erectile luminescent violet "
            "Frenzy feelers, massive dorsal prehensile clawed wing-arms, and multi-column articulated tattered "
            "Reaper wing-cloak draping over shoulders and trailing like a dark mantle."
        ),
    ),
    MonsterSpec(
        id="chaotic_gore_magala",
        title="Chaotic Gore Magala",
        title_epithet="Half-Molted Fallen Seraph",
        species_class="Elder Dragon",
        variant_kind="variant",
        parent_species_id="gore_magala",
        wiki_url="https://monsterhunter.fandom.com/wiki/Chaotic_Gore_Magala",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="chaotic_asymmetric_frenzy_slam",
        artisan_motif="gore_chaotic",
        accent_rgb=(255, 196, 68),
        anatomy_summary=(
            "Asymmetrical interrupted-molt variant: left side trapped in dark obsidian Gore visor & tattered violet "
            "cloak; right side molted into radiant golden Shagaru Magala seraphic wing-plates, unfolded golden horn, "
            "and opened glowing eye, surging with unstable dual violet-amber energy."
        ),
    ),
    # --- 7. Ruiner Nergigante (Train) ---
    MonsterSpec(
        id="ruiner_nergigante",
        title="Ruiner Nergigante",
        title_epithet="Eater of Elders",
        species_class="Elder Dragon",
        variant_kind="variant",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Ruiner_Nergigante",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="extinction_metallic_divebomb",
        artisan_motif="nergigante_iron",
        accent_rgb=(255, 112, 64),
        anatomy_summary=(
            "Battle-scarred juggernaut six-limbed Elder Dragon: massive swept bull-demon horns (one battle-scarred "
            "asymmetrical metallic regrown horn), bulldog predatory snout, hunchbacked shoulders bristling with "
            "gunmetal-iron & bone-white regenerative spike clusters, gorilla-like brawling forelimbs, and spiked wings."
        ),
    ),
    # --- 8. Crimson Glow Valstrax (Test Unseen) ---
    MonsterSpec(
        id="crimson_glow_valstrax",
        title="Crimson Glow Valstrax",
        title_epithet="Argent Sky Comet",
        species_class="Elder Dragon",
        variant_kind="variant",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Crimson_Glow_Valstrax",
        split="research_nc/test_unseen",
        is_winged=True,
        signature_gif_clip="supersonic_crimson_comet_ambush",
        artisan_motif="valstrax_jet",
        accent_rgb=(255, 46, 62),
        anatomy_summary=(
            "Supersonic jet-propelled Elder Dragon: sleek metallic gunmetal-silver falcon-beak skull & swept crest, "
            "glowing crimson ventral chest air-intake turbine vents, articulated mechanical gimbal wing-pods "
            "splitting into 3-lance trident jet thrusters blasting blazing crimson dragon-plasma exhaust plumes."
        ),
    ),
    # --- 9. Fatalis Lineage (Train) ---
    MonsterSpec(
        id="fatalis",
        title="Fatalis",
        title_epithet="The Black Dragon of Schrade",
        species_class="Elder Dragon",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Fatalis",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="schrade_demise_hellfire_sweep",
        artisan_motif="fatalis_obsidian",
        accent_rgb=(255, 108, 38),
        anatomy_summary=(
            "Classic Forbidden Black Dragon: towering sinuous 5-bone serpentine S-neck, fierce wedged draconic skull "
            "with 4 swept-back obsidian horns, incandescent molten-armor sternum fissure, athletic waist, 4 clawed "
            "dragon legs, colossal 4-finger tattered bat wings, and extra-long dragging serpentine whip tail."
        ),
    ),
    MonsterSpec(
        id="white_fatalis",
        title="White Fatalis",
        title_epithet="Ancestral God of Eclipse Lightning",
        species_class="Elder Dragon",
        variant_kind="rare_species",
        parent_species_id="fatalis",
        wiki_url="https://monsterhunter.fandom.com/wiki/White_Fatalis",
        split="research_nc/train",
        is_winged=True,
        signature_gif_clip="ancestral_eclipse_thunder_ascent",
        artisan_motif="fatalis_white",
        accent_rgb=(255, 44, 68),
        anatomy_summary=(
            "Ancient Old Fatalis: luminous pearl-silver scales, towering multi-tine branching stag-antler crown horns, "
            "billowing celestial white mane & jaw-beard cascading down the long S-neck and chest, scarlet dragon-lightning "
            "chest cracks, and vast eclipse-violet bat wings."
        ),
    ),
    # --- 10. Velkhana (Validation) ---
    MonsterSpec(
        id="velkhana",
        title="Velkhana",
        title_epithet="The Iceborne Empress",
        species_class="Elder Dragon",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Velkhana",
        split="research_nc/val",
        is_winged=True,
        signature_gif_clip="scorpion_tail_rapier_flurry",
        artisan_motif="velkhana_frost",
        accent_rgb=(112, 238, 255),
        anatomy_summary=(
            "Aristocratic Elder Dragon of Frost: slender tapered noble snout, radiant crystalline ice tiara & forward "
            "brow lances, hanging cheek icicle ruffs, graceful swan S-neck, greyhound waist, multi-layered frost-feather "
            "crystal blade wings, and long articulated scorpion-rapier tail lance."
        ),
    ),
    # --- 11. Zoh Shia (Test Unseen) ---
    MonsterSpec(
        id="zoh_shia",
        title="Zoh Shia",
        title_epithet="Seraphic Wylk Construct",
        species_class="Construct",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Zoh_Shia",
        split="research_nc/test_unseen",
        is_winged=True,
        signature_gif_clip="black_dragon_wylk_metamorphosis",
        artisan_motif="zoh_shia_wylk",
        accent_rgb=(196, 138, 255),
        anatomy_summary=(
            "Ruins of Wyveria Guardian Construct: pure porcelain-white crystallized Wylk seraphic wing-hands cradling "
            "the shoulders and crowned halo crest, fracturing asymmetrically to expose grotesque primordial Black-Dragon "
            "obsidian horns, crimson-violet energy cores, and mutated dark wing-tendrils."
        ),
    ),
    # --- 12. Lagiacrus Lineage (Train) ---
    MonsterSpec(
        id="lagiacrus",
        title="Lagiacrus",
        title_epithet="Lord of the Seas",
        species_class="Leviathan",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Lagiacrus",
        split="research_nc/train",
        is_winged=False,
        signature_gif_clip="maelstrom_coil_thunder_discharge",
        artisan_motif="lagiacrus_ocean",
        accent_rgb=(76, 228, 255),
        anatomy_summary=(
            "Apex Marine Leviathan: long low-slung crocodilian-plesiosaur S-curve, elongated tapering U-snout lined "
            "with conical teeth, tall swept plated crown horns, notched multi-lobe cobra gill-hood, ranked burnt-orange "
            "dorsal bio-electric shock-shell spires, four sprawled webbed paddles, and laterally flattened ribbon-fin tail."
        ),
    ),
    MonsterSpec(
        id="ivory_lagiacrus",
        title="Ivory Lagiacrus",
        title_epithet="White Conqueror of Land & Surf",
        species_class="Leviathan",
        variant_kind="subspecies",
        parent_species_id="lagiacrus",
        wiki_url="https://monsterhunter.fandom.com/wiki/Ivory_Lagiacrus",
        split="research_nc/train",
        is_winged=False,
        signature_gif_clip="maelstrom_coil_thunder_discharge",
        artisan_motif="lagiacrus_ivory",
        accent_rgb=(128, 214, 255),
        anatomy_summary=(
            "Terrestrial-adapted pearl-ivory subspecies: alabaster-white & slate-amethyst carapace, broader reinforced "
            "land-striding limbs, enlarged continuous-discharge amber-violet dorsal shock-spires, and swept horned hood."
        ),
    ),
    MonsterSpec(
        id="abyssal_lagiacrus",
        title="Abyssal Lagiacrus",
        title_epithet="Hadal Thunder Dynamo",
        species_class="Leviathan",
        variant_kind="rare_species",
        parent_species_id="lagiacrus",
        wiki_url="https://monsterhunter.fandom.com/wiki/Abyssal_Lagiacrus",
        split="research_nc/train",
        is_winged=False,
        signature_gif_clip="hadal_vortex_dynamo_surge",
        artisan_motif="lagiacrus_abyssal",
        accent_rgb=(48, 232, 255),
        anatomy_summary=(
            "Ancient deep-trench rare species: pitch-black midnight carapace pulsing with bioluminescent electric-cyan "
            "dermal seams, crimson abyssal eyes, permanently supercharged towering cyan Thunder-Dynamo dorsal spires, "
            "and giant pelagic vortex paddle tail."
        ),
    ),
    # --- 13. Mizutsune (Train) ---
    MonsterSpec(
        id="mizutsune",
        title="Mizutsune",
        title_epithet="Bewitching Bubblefox Leviathan",
        species_class="Leviathan",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Mizutsune",
        split="research_nc/train",
        is_winged=False,
        signature_gif_clip="bewitching_bubble_drift_pirouette",
        artisan_motif="mizutsune_sakura",
        accent_rgb=(255, 118, 186),
        anatomy_summary=(
            "Graceful vulpine Leviathan: slender tapered fox snout, extravagant fanned sakura-pink & gold iris-petal "
            "head-crests, silky dark-violet throat & chest fur ruff, undulating petal dorsal fin row, agile low-slung "
            "limbs, and massive voluminous curled dark-violet & sakura fox-brush tail."
        ),
    ),
    # --- 14. Jin Dahaad (Validation) ---
    MonsterSpec(
        id="jin_dahaad",
        title="Jin Dahaad",
        title_epithet="Iceshard Thermodynamic Behemoth",
        species_class="Leviathan",
        variant_kind="canonical_flagship",
        parent_species_id=None,
        wiki_url="https://monsterhunter.fandom.com/wiki/Jin_Dahaad",
        split="research_nc/val",
        is_winged=False,
        signature_gif_clip="supernova_radiator_frost_cataclysm",
        artisan_motif="jin_dahaad_copper",
        accent_rgb=(124, 240, 255),
        anatomy_summary=(
            "Colossal Iceshard Cliffs Leviathan: armored T-hooded metallic skull with split mandible, arched cathedral "
            "arrays of parallel oxidized copper-bronze heat-sink radiator fins along skull, neck, spine, limbs & "
            "two-pronged rake tail, frosted ice-grip sprawled pillar limbs, and glowing thermal vent core."
        ),
    ),
)

CANONICAL_22_BY_ID: dict[str, MonsterSpec] = {s.id: s for s in CANONICAL_22_SPECS}
CANONICAL_22_ORDER: tuple[str, ...] = tuple(s.id for s in CANONICAL_22_SPECS)


def get_monster_spec(monster_id: str) -> MonsterSpec:
    if monster_id not in CANONICAL_22_BY_ID:
        raise KeyError(f"Unknown Monster Hunter specimen {monster_id!r}; valid IDs: {list(CANONICAL_22_ORDER)}")
    return CANONICAL_22_BY_ID[monster_id]
