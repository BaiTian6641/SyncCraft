"""Combinatorial multi-domain Block-DSL v3 procedural asset generators & shard builder (P1-08 / ADR 0004).

Builds 100% original, typed-AST, SE(3)-compiled, 15-axis SAT + P1-02 physical-contact
verified 3D articulated block models across 7 macro-domains and 32 disjoint families:
- 20 Seen families (Artificial, Natural, Industrial, Building, Humanoid, Animal, Dragon;
  12-28 lowered boxes, 3-13 bones per asset)
- 12 Unseen held-out families (>= 16 lowered boxes per asset for Gate G2)

Applies Halton low-discrepancy sampling, 6-register bijective identifier remapping,
valid DAG topological declaration permutation, stateful @materials/@style PBR/RD binding,
and Text2CAD-style 4-level multi-granularity prompts to prevent template overfitting.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import (
    Attach,
    Bone,
    Box,
    DSLValidationError,
    Meta,
    Palette,
    Place,
    Program,
    validate_program,
)
from synccraft.dsl.parser import DSLParseError, format_program, parse_program
from synccraft.kinematics.se3_chain import Scene, compile_geometry
from .diversity_engine import (
    MACRO_DOMAINS,
    NAMING_REGISTERS,
    compose_multilevel_prompt,
    compute_qd_signature,
    diversify_program_ast,
    summarize_corpus_diversity,
)
from .multi_domain_taxonomy import (
    ENTITY_BODY_PLANS,
    FAMILY_DOMAINS,
    MULTI_DOMAIN_DISPATCH,
)

GENERATOR_REVISION = "synccraft-v3.0-multidomain-p1-08-r2"

SEEN_STATIC_FAMILIES: tuple[str, ...] = (
    # Artificial (7)
    "straight_blade_sword",
    "single_edge_blade",
    "battle_axe",
    "spear",
    "war_hammer",
    "simple_wood_bow",
    "furniture_table_chair_chest_cabinet",
    # Industrial (3)
    "gearbox",
    "articulated_robotic_arm",
    "industrial_turbine_generator",
    # Natural (3)
    "ancient_canopy_tree",
    "crystalline_geode_outcrop",
    "coral_fungal_colony",
    # Building / Architecture (3)
    "modular_timber_stone_building",
    "tiered_pagoda_watchtower",
    "arched_stone_bridge_gate",
    # Humanoid (1)
    "humanoid_biped",
    # Animal (2)
    "dog_horse_quadruped",
    "bird",
    # Dragon (1)
    "winged_western_dragon",
)

UNSEEN_STATIC_FAMILIES: tuple[str, ...] = (
    # Artificial (3)
    "mechanical_crossbow",
    "hand_lantern",
    "floating_ring_staff",
    # Natural (1)
    "weeping_bonsai_arch",
    # Industrial (3)
    "tripod_telescope",
    "anvil_forge",
    "hydraulic_gantry_crane",
    # Building / Architecture (2)
    "folding_windmill",
    "coastal_lighthouse_keep",
    # Animal (2)
    "hexapod_beetle",
    "heavy_tail_tyrannosaur",
    # Dragon (1)
    "segmented_sea_dragon",
)

SEEN_FAMILIES: tuple[str, ...] = SEEN_STATIC_FAMILIES
UNSEEN_FAMILIES: tuple[str, ...] = UNSEEN_STATIC_FAMILIES

MANDATORY_ROW_FIELDS: tuple[str, ...] = (
    "asset_id",
    "family_id",
    "body_plan_id_if_dynamic",
    "generator_revision",
    "generator_parameters",
    "canonical_dsl",
    "canonical_dsl_sha256",
    "prompt_and_provenance",
    "source_license_and_review_record",
    "split",
    "audit_version_and_results",
    "target_profile_and_version",
    "seed",
)

SYNCCRAFT_ATTRIBUTION: dict[str, Any] = {
    "project": "SyncCraft v3.0 — Typed Kinematic Block-DSL & Direct 15-Axis SAT Procedural Asset Framework",
    "authors": "SyncCraft Contributors (https://github.com/BaiTian6641/SyncCraft)",
    "maintainer": "BaiTian6641",
    "repository_url": "https://github.com/BaiTian6641/SyncCraft",
    "license": "Apache-2.0",
    "upstream_attributions": [
        "SyncCraft v3.0 Typed Block-DSL Compiler, Shared SE(3) Kinematics, P1-02 Contact Policy & Direct 15-Axis SAT Collision Audit — Copyright 2026 SyncCraft Contributors (Apache-2.0).",
        "Oklab / OKLCH Perceptual Color Space — Bjorn Ottosson (2020), 'A perceptual color space for image processing' (https://bottosson.github.io/posts/oklab/).",
        "Blockbench 5.0 (bbmodel_5) and Minecraft: Java Edition Modern (1.21.11+) are referenced strictly as nominal export compatibility targets; zero Mojang game assets or third-party proprietary models/textures are included.",
    ],
}


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    train_per_seen_family: int
    val_per_seen_family: int
    test_per_unseen_family: int
    hf_size_category: str

    @property
    def seen_per_family(self) -> int:
        return self.train_per_seen_family + self.val_per_seen_family

    @property
    def total_seen(self) -> int:
        return self.seen_per_family * len(SEEN_STATIC_FAMILIES)

    @property
    def total_unseen(self) -> int:
        return self.test_per_unseen_family * len(UNSEEN_STATIC_FAMILIES)

    @property
    def total_rows(self) -> int:
        return self.total_seen + self.total_unseen


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "smoke": DatasetPreset(
        name="smoke",
        train_per_seen_family=1,
        val_per_seen_family=1,
        test_per_unseen_family=1,
        hf_size_category="n<1K",
    ),
    "pilot_200": DatasetPreset(
        name="pilot_200",
        train_per_seen_family=9,
        val_per_seen_family=1,
        test_per_unseen_family=5,
        hf_size_category="n<1K",
    ),
    "bakeoff_8k": DatasetPreset(
        name="bakeoff_8k",
        train_per_seen_family=360,
        val_per_seen_family=40,
        test_per_unseen_family=50,
        hf_size_category="1K<n<10K",
    ),
    "full_60k": DatasetPreset(
        name="full_60k",
        train_per_seen_family=2700,
        val_per_seen_family=300,
        test_per_unseen_family=100,
        hf_size_category="10K<n<100K",
    ),
    "full_75k": DatasetPreset(
        name="full_75k",
        train_per_seen_family=3375,
        val_per_seen_family=375,
        test_per_unseen_family=200,
        hf_size_category="10K<n<100K",
    ),
}


# Curated perceptual OKLCH material banks (L in [0.14, 0.96], C >= 0, H in [0, 360))
_METALS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forged_steel", (0.76, 0.025, 245.0)),
    ("tempered_iron", (0.58, 0.020, 252.0)),
    ("silver_alloy", (0.86, 0.018, 235.0)),
    ("dark_iron", (0.31, 0.022, 260.0)),
    ("blued_steel", (0.46, 0.055, 255.0)),
    ("damascus_steel", (0.64, 0.032, 240.0)),
)

_ACCENT_METALS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("aged_brass", (0.72, 0.145, 78.0)),
    ("gilded_bronze", (0.68, 0.155, 64.0)),
    ("burnished_copper", (0.64, 0.165, 44.0)),
    ("electrum_trim", (0.83, 0.135, 88.0)),
    ("pewter_trim", (0.62, 0.018, 230.0)),
)

_WOODS_AND_GRIPS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("walnut_wood", (0.38, 0.085, 48.0)),
    ("seasoned_oak", (0.52, 0.105, 58.0)),
    ("charred_ash", (0.28, 0.035, 52.0)),
    ("yew_heartwood", (0.56, 0.135, 66.0)),
    ("oxblood_leather", (0.36, 0.115, 26.0)),
    ("lacquered_ebony", (0.22, 0.025, 285.0)),
)

_STONES_AND_FABRICS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("quarried_slate", (0.44, 0.024, 245.0)),
    ("basalt_masonry", (0.29, 0.016, 258.0)),
    ("bleached_canvas", (0.87, 0.038, 84.0)),
    ("crimson_velvet", (0.42, 0.165, 22.0)),
    ("hemp_cordage", (0.74, 0.075, 76.0)),
)

_GLOW_MATERIALS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("aether_crystal", (0.86, 0.195, 205.0)),
    ("forge_ember", (0.72, 0.225, 42.0)),
    ("lantern_flame", (0.92, 0.185, 86.0)),
    ("sunstone_core", (0.84, 0.190, 72.0)),
    ("verdant_rune", (0.80, 0.205, 148.0)),
)


def _derive_rng(family_id: str, split: str, variant_index: int, base_seed: int) -> tuple[random.Random, int]:
    digest = hashlib.sha256(
        f"{GENERATOR_REVISION}:{family_id}:{split}:{variant_index}:{base_seed}".encode("utf-8")
    ).digest()
    numeric_seed = int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF
    return random.Random(numeric_seed), numeric_seed


def _make_palette(
    rng: random.Random,
    base_spec: tuple[str, tuple[float, float, float]],
    *,
    emissive: bool = False,
    suffix: str = "",
) -> Palette:
    raw_name, (l0, c0, h0) = base_spec
    name = f"{raw_name}{suffix}"
    l_val = round(min(0.96, max(0.14, l0 + rng.uniform(-0.045, 0.045))), 4)
    c_val = round(max(0.005, c0 + rng.uniform(-0.015, 0.020)), 4)
    h_val = round((h0 + rng.uniform(-16.0, 16.0)) % 360.0, 2)
    steps = rng.choice((3, 4, 5, 6, 7))
    hue_shift = (round(rng.uniform(-6.0, 0.0), 2), round(rng.uniform(0.0, 8.0), 2))
    return Palette(name=name, base_lch=(l_val, c_val, h_val), steps=steps, hue_shift=hue_shift, emissive=emissive)


def _jitter_pivot(rng: random.Random, base_xyz: tuple[float, float, float], span: float = 0.75) -> tuple[float, float, float]:
    return (
        round(base_xyz[0] + rng.uniform(-span, span), 3),
        round(base_xyz[1] + rng.uniform(-0.35, 0.35), 3),
        round(base_xyz[2] + rng.uniform(-span, span), 3),
    )


def _check_java_envelope(scene: Scene) -> bool:
    for box in scene.boxes:
        half = np.asarray(box.size, dtype=np.float64) * 0.5 + float(box.inflate)
        from_ = np.asarray(box.center, dtype=np.float64) - half
        to = np.asarray(box.center, dtype=np.float64) + half
        coords = np.concatenate((from_, to, np.asarray(box.corners(), dtype=np.float64).reshape(-1)))
        if not np.isfinite(coords).all() or (coords < -16.0 - 1e-8).any() or (coords > 32.0 + 1e-8).any():
            return False
    return True


def _check_entity_envelope(scene: Scene) -> bool:
    """Verify entity scenes remain finite and well-bounded in [-48, 64] Blockbench/GeckoLib/glTF space."""
    for box in scene.boxes:
        coords = np.asarray(box.corners(), dtype=np.float64).reshape(-1)
        if not np.isfinite(coords).all() or (coords < -48.0 - 1e-8).any() or (coords > 64.0 + 1e-8).any():
            return False
    return True


def _compose_prompt(
    rng: random.Random,
    *,
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
    material_names: list[str],
    structural_notes: str,
) -> tuple[str, str]:
    sym_clause = "with mirror_x bilateral symmetry" if has_mirror_x else "with explicit asymmetric/cardinal sub-assemblies"
    det_clause = ", including an authorized detached floating sub-rig" if has_detached else ""
    angle_clause = (
        f"articulated at {joint_angle_deg:g} deg rest_euler joints"
        if abs(joint_angle_deg) > 1e-6
        else "with orthogonal axis-aligned joint frames"
    )
    mats = ", ".join(material_names)
    style_idx = rng.randrange(5)
    if style_idx == 0:
        prompt = (
            f"Generate a SyncCraft v3.0 Block-DSL {archetype} model for family '{family_id}' "
            f"(sub-style: {substyle}, target_profile: {target_profile}, texel_density: {texel_density}). "
            f"Construct {box_count} lowered boxes across {bone_count} bones {sym_clause}{det_clause}, "
            f"{angle_clause}. Materials: {mats}. Structure: {structural_notes}."
        )
        style_name = "technical_specification"
    elif style_idx == 1:
        prompt = (
            f"Create a collision-free multi-part {substyle.replace('_', ' ')} ({family_id}) in Block-DSL "
            f"targeting {target_profile} ({texel_density}px density). Use {bone_count} rigged bones and "
            f"{box_count} boxes {sym_clause}{det_clause}, {angle_clause}, using OKLCH palettes [{mats}]. "
            f"Key geometry: {structural_notes}."
        )
        style_name = "modeler_brief"
    elif style_idx == 2:
        prompt = (
            f"// Family: {family_id} | Variant: {substyle} | Profile: {target_profile}@{texel_density}\n"
            f"// Rig target: {bone_count} bones, {box_count} boxes, {angle_clause}, {sym_clause}{det_clause}\n"
            f"// Palettes: {mats} | Notes: {structural_notes}\n"
            f"Write the complete SAT-verified Block-DSL program."
        )
        style_name = "structured_header_spec"
    elif style_idx == 3:
        prompt = (
            f"Design an articulated 3D block {substyle.replace('_', ' ')} belonging to the '{family_id}' family "
            f"for {target_profile} ({archetype}, {texel_density} texel density). Include {structural_notes}, "
            f"totaling {box_count} primitives in {bone_count} bones {sym_clause}{det_clause} ({angle_clause}), "
            f"shaded with {mats}."
        )
        style_name = "art_direction"
    else:
        prompt = (
            f"Synthesize typed Block-DSL geometry for '{family_id}' ({substyle}): "
            f"archetype={archetype}, profile={target_profile}, density={texel_density}, "
            f"parts={box_count} boxes / {bone_count} bones, {sym_clause}{det_clause}, {angle_clause}, "
            f"materials=[{mats}], assembly={structural_notes}."
        )
        style_name = "compact_parametric"
    return prompt, style_name


# ==============================================================================
# 8 Seen Static Family Combinatorial AST Generators (12-28 boxes, 3-9 bones)
# ==============================================================================


def _gen_straight_blade_sword(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("arming_longsword", "fullered_greatsword", "templar_broadsword", "runed_bastard_sword")
    substyle = substyles[v_idx % len(substyles)]
    profile = "java_modern" if (v_idx // 4) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 7 == 0) else 16

    p_blade = _make_palette(rng, rng.choice(_METALS))
    p_guard = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_grip = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    use_gem = (v_idx % 2 == 0)
    p_gem = _make_palette(rng, rng.choice(_GLOW_MATERIALS), emissive=True) if use_gem else _make_palette(rng, rng.choice(_STONES_AND_FABRICS))

    grip_h = rng.choice((4, 5, 6))
    guard_w = rng.choice((4, 6))
    quillon_len = rng.choice((3, 4, 5))
    quillon_deg = rng.choice((-22.5, -15.0, -11.25, 0.0, 11.25, 15.0, 22.5))
    blade_w = rng.choice((2, 3))
    b1_h = rng.choice((4, 5, 6))
    b2_h = rng.choice((4, 5, 6))
    b3_h = rng.choice((3, 4))
    wrap_inflate = round(rng.choice((0.0, 0.04, 0.06, 0.08)), 3)
    gem_inset = round(rng.choice((0.0, -0.15, -0.25)), 3)

    hilt_items: list[Box | Bone] = [
        Box("grip_core", (2, grip_h, 2), p_grip.name),
        Box("grip_wrap", (2, 2, 2), p_guard.name, Place("grip_core", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), inflate=wrap_inflate if grip_h >= 5 else 0.0),
    ]
    # Wait: grip_wrap centered at [0,0,0] on grip_core would co-penetrate grip_core!
    # Instead, stack grip_lower -> grip_band -> grip_upper cleanly along Y!
    g_low = rng.choice((2, 3))
    g_up = rng.choice((2, 3))
    hilt_items = [
        Box("grip_band", (2, 2, 2), p_guard.name, inflate=wrap_inflate),
        Box("grip_lower", (2, g_low, 2), p_grip.name, Place("grip_band", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("grip_upper", (2, g_up, 2), p_grip.name, Place("grip_band", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("pommel_neck", (2, 1, 2), p_guard.name, Place("grip_lower", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("pommel_wheel", (3, 2, 3), p_guard.name, Place("pommel_neck", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("pommel_peen", (1, 1, 1), p_blade.name, Place("pommel_wheel", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)), cullface=("down",)),
    ]

    quillon_bone = Bone(
        "quillon_l",
        rest_euler=(0.0, 0.0, quillon_deg),
        attach=Attach("guard_block", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="quillon_r",
        items=[
            Box("quillon_arm", (quillon_len, 1, 2), p_guard.name),
            Box("quillon_terminal", (1, 2, 2), p_guard.name, Place("quillon_arm", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    blade_items: list[Box | Bone] = [
        Box("ricasso_base", (blade_w + 1, 2, 2), p_blade.name),
        Box("blade_lower", (blade_w, b1_h, 1), p_blade.name, Place("ricasso_base", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("fuller_front", (1, max(2, b1_h - 1), 1), p_guard.name, Place("blade_lower", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.25))),
        Box("blade_mid", (blade_w, b2_h, 1), p_blade.name, Place("blade_lower", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("blade_upper", (max(1, blade_w - 1), b3_h, 1), p_blade.name, Place("blade_mid", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("blade_point", (1, 2, 1), p_blade.name, Place("blade_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), cullface=("up",)),
    ]

    guard_bone = Bone(
        "crossguard",
        attach=Attach("grip_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("guard_block", (guard_w, 2, 2), p_guard.name),
            Box("ecusson_gem", (2, 1, 1), p_gem.name, Place("guard_block", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, gem_inset))),
            Box("rear_counter_boss", (2, 1, 1), p_gem.name, Place("guard_block", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, -gem_inset))),
            quillon_bone,
            Bone("blade_assembly", attach=Attach("guard_block", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), items=blade_items),
        ],
    )
    hilt_items.append(guard_bone)

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_blade, p_guard, p_grip, p_gem],
        root=Bone("hilt_root", pivot_local=_jitter_pivot(rng, (8.0, 6.0, 8.0)), items=hilt_items),
    )
    meta_info = {
        "substyle": substyle,
        "primary_span": b1_h + b2_h + b3_h + 4,
        "secondary_span": guard_w + 2 * quillon_len,
        "joint_angle_deg": quillon_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": wrap_inflate,
        "material_theme": f"{p_blade.name}+{p_guard.name}+{p_grip.name}",
        "notes": f"3-part wrapped hilt with wheel pommel, {guard_w}-wide guard with mirrored {quillon_len}-unit quillons at {quillon_deg:g} deg, inset ecusson gem, ricasso, fuller, and 3-stage tapered blade",
        "extra": {"blade_w": blade_w, "gem_inset": gem_inset},
    }
    return prog, meta_info


def _gen_single_edge_blade(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("uchigatana_saber", "heavy_falchion", "cavalry_backsword", "hunting_messer")
    substyle = substyles[v_idx % len(substyles)]
    profile = "java_modern" if (v_idx // 4) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 8 == 1) else 16

    p_edge = _make_palette(rng, rng.choice(_METALS))
    p_spine = _make_palette(rng, rng.choice(_METALS), suffix="_spine")
    p_mount = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_wrap = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))

    tsuka_low = rng.choice((2, 3))
    tsuka_up = rng.choice((2, 3))
    tsuba_w = rng.choice((3, 4, 5))
    # Authentic katana sori curves toward the mune/spine (-Z), i.e. negative X Euler pitch
    curve_1 = rng.choice((0.0, -5.0, -7.5, -11.25))
    curve_2 = rng.choice((-7.5, -11.25, -15.0))
    seg1_h = rng.choice((5, 6, 7))
    seg2_h = rng.choice((4, 5, 6))
    bevel_d = rng.choice((1, 2))
    fuchi_inflate = round(rng.choice((0.0, 0.03, 0.06)), 3)

    rim_wing = Bone(
        "tsuba_wing_l",
        rest_euler=(0.0, 0.0, rng.choice((0.0, 11.25, -11.25))),
        attach=Attach("tsuba_disc", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="tsuba_wing_r",
        items=[Box("rim_lobe", (1, 1, 3), p_mount.name)],
    )

    kissaki_bone = Bone(
        "kissaki_section",
        rest_euler=(curve_2, 0.0, 0.0),
        attach=Attach("mune_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("kissaki_spine", (1, 3, 1), p_spine.name),
            Box("kissaki_ha", (1, 2, 1), p_edge.name, Place("kissaki_spine", (0.0, 1.0, 1.0), (0.0, 1.0, -1.0))),
            Box("kissaki_point", (1, 2, 1), p_edge.name, Place("kissaki_spine", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    nagasa_bone = Bone(
        "nagasa_lower",
        rest_euler=(curve_1, 0.0, 0.0),
        attach=Attach("habaki_collar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("mune_lower", (1, seg1_h, 1), p_spine.name),
            Box("ha_bevel_lower", (1, seg1_h - 1, bevel_d), p_edge.name, Place("mune_lower", (0.0, 1.0, 1.0), (0.0, 1.0, -1.0))),
            Box("mune_upper", (1, seg2_h, 1), p_spine.name, Place("mune_lower", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("ha_bevel_upper", (1, seg2_h - 1, 1), p_edge.name, Place("mune_upper", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            kissaki_bone,
        ],
    )

    tsuba_bone = Bone(
        "tsuba_assembly",
        attach=Attach("tsuka_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("tsuba_disc", (tsuba_w, 1, 4), p_mount.name),
            Box("seppa_washer", (2, 1, 2), p_mount.name, Place("tsuba_disc", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("habaki_collar", (2, 2, 2), p_mount.name, Place("seppa_washer", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            rim_wing,
            nagasa_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("menuki_ring", (2, 1, 2), p_mount.name, inflate=fuchi_inflate),
        Box("tsuka_lower", (2, tsuka_low, 2), p_wrap.name, Place("menuki_ring", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("tsuka_upper", (2, tsuka_up, 2), p_wrap.name, Place("menuki_ring", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("kashira_pommel", (2, 1, 3), p_mount.name, Place("tsuka_lower", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("cord_tassel", (1, 2, 1), p_wrap.name, Place("kashira_pommel", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        tsuba_bone,
    ]

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_edge, p_spine, p_mount, p_wrap],
        root=Bone("tsuka_root", pivot_local=_jitter_pivot(rng, (8.0, 6.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": seg1_h + seg2_h + 5,
        "secondary_span": tsuba_w + 2,
        "joint_angle_deg": round(curve_1 + curve_2, 2),
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": fuchi_inflate,
        "material_theme": f"{p_edge.name}+{p_mount.name}+{p_wrap.name}",
        "notes": f"single-edged curved blade with separate spine (mune) and cutting bevel (ha), {curve_1:g}+{curve_2:g} deg pitch curve, habaki collar, mirrored tsuba lobes, and wrapped tsuka hilt",
        "extra": {"curve_1": curve_1, "curve_2": curve_2, "bevel_d": bevel_d},
    }


def _gen_battle_axe(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("double_bitted_greataxe", "bearded_norse_axe", "crescent_halberd_axe", "dwarven_cleaver")
    substyle = substyles[v_idx % len(substyles)]
    double_bitted = (v_idx % 2 == 0)
    profile = "java_modern" if (v_idx // 4) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 6 == 2) else 16

    p_head = _make_palette(rng, rng.choice(_METALS))
    p_edge = _make_palette(rng, rng.choice(_METALS), suffix="_edge")
    p_haft = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_trim = _make_palette(rng, rng.choice(_ACCENT_METALS))

    haft_low = rng.choice((4, 5, 6))
    haft_up = rng.choice((5, 6, 7))
    cheek_w = rng.choice((3, 4))
    bit_h = rng.choice((5, 6, 7))
    beard_h = rng.choice((2, 3))
    cant_deg = rng.choice((-11.25, 0.0, 11.25))
    grip_inf = round(rng.choice((0.0, 0.04, 0.07)), 3)

    cheek_items: list[Box | Bone] = [
        Box("cheek_web", (cheek_w, 3, 2), p_head.name),
        Box("cutting_bit", (2, bit_h, 1), p_edge.name, Place("cheek_web", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        Box("upper_horn_tip", (1, 2, 1), p_edge.name, Place("cutting_bit", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("lower_beard_lug", (1, beard_h, 1), p_edge.name, Place("cutting_bit", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
    ]

    head_children: list[Box | Bone] = [
        Box("socket_eye", (4, 4, 3), p_head.name),
        Box("top_ferrule", (2, 2, 2), p_trim.name, Place("socket_eye", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("thrust_spike", (1, rng.choice((3, 4)), 1), p_edge.name, Place("top_ferrule", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("socket_rivet_n", (2, 2, 1), p_trim.name, Place("socket_eye", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.2))),
        Box("socket_rivet_s", (2, 2, 1), p_trim.name, Place("socket_eye", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.2))),
    ]

    if double_bitted:
        head_children.append(
            Bone(
                "axe_wing_l",
                rest_euler=(0.0, 0.0, cant_deg),
                attach=Attach("socket_eye", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                mirror_name="axe_wing_r",
                items=cheek_items,
            )
        )
    else:
        poll_deg = rng.choice((15.0, 22.5))
        head_children.append(
            Bone(
                "front_blade_wing",
                rest_euler=(0.0, 0.0, cant_deg),
                attach=Attach("socket_eye", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                items=cheek_items,
            )
        )
        head_children.append(
            Bone(
                "rear_poll_spike",
                rest_euler=(0.0, 0.0, poll_deg),
                attach=Attach("socket_eye", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                items=[
                    Box("poll_shank", (3, 2, 2), p_head.name),
                    Box("poll_beak", (2, 1, 1), p_edge.name, Place("poll_shank", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
                ],
            )
        )
        head_children.append(
            Bone(
                "langet_strip_l",
                attach=Attach("socket_eye", (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0)),
                mirror_name="langet_strip_r",
                items=[Box("langet_bar", (1, 3, 1), p_trim.name)],
            )
        )

    root_items: list[Box | Bone] = [
        Box("grip_sleeve", (2, 2, 2), p_trim.name, inflate=grip_inf),
        Box("haft_lower", (2, haft_low, 2), p_haft.name, Place("grip_sleeve", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("pommel_cap", (3, 1, 3), p_trim.name, Place("haft_lower", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("butt_spike", (1, 2, 1), p_head.name, Place("pommel_cap", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("haft_upper", (2, haft_up, 2), p_haft.name, Place("grip_sleeve", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Bone("axe_head_core", attach=Attach("haft_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), items=head_children),
    ]

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_head, p_edge, p_haft, p_trim],
        root=Bone("haft_root", pivot_local=_jitter_pivot(rng, (8.0, 7.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": haft_low + haft_up + 9,
        "secondary_span": 4 + 2 * (cheek_w + 2),
        "joint_angle_deg": cant_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": grip_inf,
        "material_theme": f"{p_head.name}+{p_haft.name}+{p_trim.name}",
        "notes": f"{'double-bitted mirrored' if double_bitted else 'single-bit bearded + rear poll spike'} battle axe with {bit_h}-tall cutting bit, {beard_h}-unit beard lug, top thrust spike, and iron-shod haft",
        "extra": {"double_bitted": double_bitted, "bit_h": bit_h, "beard_h": beard_h},
    }


def _gen_spear(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("winged_boar_spear", "imperial_partisan", "tri_lug_pike", "banneret_lance")
    substyle = substyles[v_idx % len(substyles)]
    profile = "java_modern" if (v_idx // 4) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 7 == 3) else 16

    p_steel = _make_palette(rng, rng.choice(_METALS))
    p_collar = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_pole = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_pennant = _make_palette(rng, rng.choice(_STONES_AND_FABRICS))

    pole_low = rng.choice((6, 7, 8))
    pole_up = rng.choice((6, 7, 8))
    wing_deg = rng.choice((15.0, 22.5, 30.0, -15.0, -22.5))
    wing_len = rng.choice((3, 4))
    blade_low_h = rng.choice((4, 5))
    blade_up_h = rng.choice((3, 4))
    sleeve_inf = round(rng.choice((0.0, 0.04, 0.06)), 3)

    lug_wing = Bone(
        "boar_lug_l",
        rest_euler=(0.0, 0.0, wing_deg),
        attach=Attach("socket_sleeve", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="boar_lug_r",
        items=[
            Box("lug_spar", (wing_len, 1, 1), p_collar.name),
            Box("lug_flare", (2, 2, 1), p_steel.name, Place("lug_spar", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
        ],
    )

    spearhead_bone = Bone(
        "spearhead_blade",
        attach=Attach("collar_knot", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("blade_shoulder", (3, 2, 1), p_steel.name),
            Box("blade_leaf", (2, blade_low_h, 1), p_steel.name, Place("blade_shoulder", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("midrib_front", (1, blade_low_h - 1, 1), p_collar.name, Place("blade_leaf", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.25))),
            Box("midrib_back", (1, blade_low_h - 1, 1), p_collar.name, Place("blade_leaf", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.25))),
            Box("blade_taper", (1, blade_up_h, 1), p_steel.name, Place("blade_leaf", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("bodkin_tip", (1, 2, 1), p_steel.name, Place("blade_taper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    socket_bone = Bone(
        "socket_assembly",
        attach=Attach("shaft_fore", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("socket_sleeve", (2, 3, 2), p_collar.name),
            Box("pennant_tie", (2, 1, 3), p_pennant.name, Place("socket_sleeve", (0.0, -1.0, 1.0), (0.0, -1.0, -1.0))),
            Box("collar_knot", (3, 1, 2), p_collar.name, Place("socket_sleeve", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            lug_wing,
            spearhead_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("grip_band", (2, 3, 2), p_pennant.name, inflate=sleeve_inf),
        Box("shaft_rear", (2, pole_low, 2), p_pole.name, Place("grip_band", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("butt_ferrule", (2, 2, 2), p_collar.name, Place("shaft_rear", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("ground_spike", (1, 2, 1), p_steel.name, Place("butt_ferrule", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("shaft_fore", (2, pole_up, 2), p_pole.name, Place("grip_band", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        socket_bone,
    ]

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_steel, p_collar, p_pole, p_pennant],
        root=Bone("pole_root", pivot_local=_jitter_pivot(rng, (8.0, 4.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": pole_low + pole_up + blade_low_h + blade_up_h + 10,
        "secondary_span": 2 + 2 * (wing_len + 2),
        "joint_angle_deg": wing_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": sleeve_inf,
        "material_theme": f"{p_steel.name}+{p_collar.name}+{p_pole.name}",
        "notes": f"long ash/oak shaft with butt spike, socketed collar, mirrored {wing_len}-unit boar lugs angled at {wing_deg:g} deg, raised midribs, and tapered leaf spearhead",
        "extra": {"wing_len": wing_len, "blade_low_h": blade_low_h},
    }


def _gen_war_hammer(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("lucerne_bec_de_corbin", "flanged_siege_maul", "knights_polehammer", "runic_forge_maul")
    substyle = substyles[v_idx % len(substyles)]
    symmetric_maul = (v_idx % 2 == 1)
    profile = "java_modern" if (v_idx // 4) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 5 == 0) else 16

    p_iron = _make_palette(rng, rng.choice(_METALS))
    p_brass = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_haft = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    use_rune = (substyle == "runic_forge_maul")
    p_inlay = _make_palette(rng, rng.choice(_GLOW_MATERIALS), emissive=use_rune)

    haft_low = rng.choice((5, 6, 7))
    haft_up = rng.choice((5, 6, 7))
    neck_len = rng.choice((2, 3))
    face_h = rng.choice((4, 5))
    beak_deg = rng.choice((15.0, 22.5, 30.0)) if not symmetric_maul else 0.0
    ring_inf = round(rng.choice((0.0, 0.05, 0.08)), 3)

    bolster_items: list[Box | Bone] = [
        Box("central_block", (4, 4, 4), p_iron.name),
        Box("rune_boss_n", (2, 2, 1), p_inlay.name, Place("central_block", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.2))),
        Box("rune_boss_s", (2, 2, 1), p_inlay.name, Place("central_block", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.2))),
        Box("crown_collar", (2, 1, 2), p_brass.name, Place("central_block", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("top_dague_spike", (1, rng.choice((3, 4, 5)), 1), p_iron.name, Place("crown_collar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
    ]

    if symmetric_maul:
        bolster_items.append(
            Bone(
                "maul_head_l",
                attach=Attach("central_block", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                mirror_name="maul_head_r",
                items=[
                    Box("maul_neck", (neck_len, 2, 2), p_iron.name),
                    Box("striking_block", (2, face_h, 4), p_iron.name, Place("maul_neck", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                    Box("impact_crown", (1, face_h - 1, 3), p_brass.name, Place("striking_block", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                ],
            )
        )
    else:
        bolster_items.append(
            Bone(
                "hammer_face_arm",
                attach=Attach("central_block", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                items=[
                    Box("hammer_neck", (neck_len, 2, 2), p_iron.name),
                    Box("hammer_face", (2, face_h, 4), p_iron.name, Place("hammer_neck", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                    Box("prong_upper", (1, 1, 3), p_brass.name, Place("hammer_face", (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))),
                    Box("prong_lower", (1, 1, 3), p_brass.name, Place("hammer_face", (1.0, -1.0, 0.0), (-1.0, -1.0, 0.0))),
                ],
            )
        )
        bolster_items.append(
            Bone(
                "corbin_beak_arm",
                rest_euler=(0.0, 0.0, beak_deg),
                attach=Attach("central_block", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                items=[
                    Box("beak_root", (3, 2, 2), p_iron.name),
                    Box("beak_mid", (2, 1, 1), p_iron.name, Place("beak_root", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
                    Box("beak_point", (2, 1, 1), p_brass.name, Place("beak_mid", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
                ],
            )
        )

    root_items: list[Box | Bone] = [
        Box("hand_stop_ring", (3, 1, 3), p_brass.name, inflate=ring_inf),
        Box("haft_grip_low", (2, haft_low, 2), p_haft.name, Place("hand_stop_ring", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("pommel_bulb", (3, 2, 3), p_brass.name, Place("haft_grip_low", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("pommel_stud", (1, 1, 1), p_iron.name, Place("pommel_bulb", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("haft_shaft_up", (2, haft_up, 2), p_haft.name, Place("hand_stop_ring", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("langet_collar", (3, 2, 3), p_brass.name, Place("haft_shaft_up", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Bone("bolster_head", attach=Attach("langet_collar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), items=bolster_items),
    ]

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_iron, p_brass, p_haft, p_inlay],
        root=Bone("maul_root", pivot_local=_jitter_pivot(rng, (8.0, 7.5, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": haft_low + haft_up + 12,
        "secondary_span": 4 + 2 * (neck_len + 3),
        "joint_angle_deg": beak_deg,
        "has_mirror_x": symmetric_maul,
        "has_detached_bone": False,
        "inflate_detail": ring_inf,
        "material_theme": f"{p_iron.name}+{p_brass.name}+{p_haft.name}",
        "notes": f"{'mirrored twin-face siege maul' if symmetric_maul else f'pronged polehammer with {beak_deg:g} deg rear bec-de-corbin beak'}, top dague thrusting spike, rune bosses, and ringed haft",
        "extra": {"symmetric_maul": symmetric_maul, "face_h": face_h},
    }


def _gen_simple_wood_bow(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyles = ("recurve_longbow", "composite_horn_bow", "ranger_stave_bow", "laminated_warbow")
    substyle = substyles[v_idx % len(substyles)]
    profile = "java_modern" if (v_idx // 4) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 6 == 4) else 16

    p_stave = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_horn = _make_palette(rng, rng.choice(_METALS))
    p_trim = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_cord = _make_palette(rng, rng.choice(_STONES_AND_FABRICS))

    riser_h = rng.choice((4, 5))
    limb_in_h = rng.choice((4, 5, 6))
    limb_out_h = rng.choice((4, 5))
    sweep_in = rng.choice((11.25, 15.0, 22.5))
    recurve_out = rng.choice((-15.0, -11.25, 7.5, 11.25))
    grip_inf = round(rng.choice((0.0, 0.04, 0.07)), 3)

    upper_siyah = Bone(
        "upper_siyah",
        rest_euler=(recurve_out, 0.0, 0.0),
        attach=Attach("u_limb_core", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("u_siyah_stave", (2, limb_out_h, 1), p_stave.name),
            Box("u_horn_nock", (1, 2, 2), p_horn.name, Place("u_siyah_stave", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("u_string_loop", (1, 3, 1), p_cord.name, Place("u_horn_nock", (0.0, -1.0, -1.0), (0.0, 1.0, 1.0))),
        ],
    )
    lower_siyah = Bone(
        "lower_siyah",
        rest_euler=(-recurve_out, 0.0, 0.0),
        attach=Attach("l_limb_core", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("l_siyah_stave", (2, limb_out_h, 1), p_stave.name),
            Box("l_horn_nock", (1, 2, 2), p_horn.name, Place("l_siyah_stave", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("l_string_loop", (1, 3, 1), p_cord.name, Place("l_horn_nock", (0.0, 1.0, -1.0), (0.0, -1.0, 1.0))),
        ],
    )

    upper_limb = Bone(
        "upper_limb",
        rest_euler=(sweep_in, 0.0, 0.0),
        attach=Attach("upper_fade", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("u_limb_core", (2, limb_in_h, 2), p_stave.name),
            Box("u_sinew_backing", (2, limb_in_h - 1, 1), p_horn.name, Place("u_limb_core", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.2))),
            upper_siyah,
        ],
    )
    lower_limb = Bone(
        "lower_limb",
        rest_euler=(-sweep_in, 0.0, 0.0),
        attach=Attach("lower_fade", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0)),
        items=[
            Box("l_limb_core", (2, limb_in_h, 2), p_stave.name),
            Box("l_sinew_backing", (2, limb_in_h - 1, 1), p_horn.name, Place("l_limb_core", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.2))),
            lower_siyah,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("riser_grip", (2, riser_h, 3), p_cord.name, inflate=grip_inf),
        Box("arrow_rest_shelf", (1, 1, 2), p_trim.name, Place("riser_grip", (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))),
        Box("sight_bracket", (1, 2, 1), p_trim.name, Place("riser_grip", (-1.0, 1.0, 1.0), (1.0, 1.0, -1.0))),
        Box("upper_fade", (2, 2, 2), p_stave.name, Place("riser_grip", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("lower_fade", (2, 2, 2), p_stave.name, Place("riser_grip", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        upper_limb,
        lower_limb,
    ]

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_stave, p_horn, p_trim, p_cord],
        root=Bone("bow_riser_root", pivot_local=_jitter_pivot(rng, (8.0, 8.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": riser_h + 4 + 2 * (limb_in_h + limb_out_h + 2),
        "secondary_span": 6,
        "joint_angle_deg": sweep_in,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": grip_inf,
        "material_theme": f"{p_stave.name}+{p_horn.name}+{p_cord.name}",
        "notes": f"multi-joint {substyle.replace('_', ' ')} with wrapped riser, arrow shelf, sinew-backed inner limbs ({sweep_in:g} deg), and recurve horn siyahs ({recurve_out:g} deg)",
        "extra": {"sweep_in": sweep_in, "recurve_out": recurve_out},
    }


def _gen_furniture(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    kind_idx = v_idx % 4
    profile = "java_modern" if (v_idx // 4) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 8 == 5) else 16

    p_wood = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_trim = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_iron = _make_palette(rng, rng.choice(_METALS))
    p_upholstery = _make_palette(rng, rng.choice(_STONES_AND_FABRICS))

    if kind_idx == 0:
        substyle = "trestle_banquet_table"
        top_w = rng.choice((12, 14, 16))
        top_d = rng.choice((8, 10, 12))
        leg_h = rng.choice((7, 8, 9))
        splay_deg = rng.choice((0.0, 7.5, -7.5))

        trestle_l = Bone(
            "trestle_leg_l",
            rest_euler=(0.0, 0.0, splay_deg),
            attach=Attach("tabletop_slab", (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0 if False else -0.5, 0.0, 0.0)),
            mirror_name="trestle_leg_r",
            items=[
                Box("top_cleat", (2, 1, top_d - 2), p_wood.name),
                Box("front_pillar", (2, leg_h, 2), p_wood.name, Place("top_cleat", (0.0, -1.0, 1.0), (0.0, 1.0, 1.0))),
                Box("rear_pillar", (2, leg_h, 2), p_wood.name, Place("top_cleat", (0.0, -1.0, -1.0), (0.0, 1.0, -1.0))),
                Box("floor_skid", (2, 1, top_d), p_trim.name, Place("front_pillar", (0.0, -1.0, 1.0), (0.0, 1.0, 1.0)), cullface=("down",)),
                Box("side_brace", (1, 1, max(2, top_d - 6)), p_iron.name, Place("front_pillar", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            ],
        )
        root_items: list[Box | Bone] = [
            Box("tabletop_slab", (top_w, 2, top_d), p_wood.name),
            Box("table_runner", (max(4, top_w - 4), 1, top_d), p_upholstery.name, Place("tabletop_slab", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, -0.25, 0.0))),
            Box("apron_north", (max(4, top_w - 6), 2, 1), p_wood.name, Place("tabletop_slab", (0.0, -1.0, -1.0), (0.0, 1.0, -1.0))),
            Box("apron_south", (max(4, top_w - 6), 2, 1), p_wood.name, Place("tabletop_slab", (0.0, -1.0, 1.0), (0.0, 1.0, 1.0))),
            Box("drawer_pull_front", (2, 1, 1), p_trim.name, Place("apron_south", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("center_finial_dish", (2, 1, 2), p_trim.name, Place("table_runner", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            trestle_l,
        ]
        joint_deg, mirrored = splay_deg, True
        span_a, span_b = top_w, leg_h + 4
        notes = f"{top_w}x{top_d} banquet trestle table with runner, twin aprons, drawer pull, and mirrored雙-pillar trestle legs ({splay_deg:g} deg)".replace("雙", "twin")

    elif kind_idx == 1:
        substyle = "carved_highback_armchair"
        seat_w = rng.choice((8, 10))
        seat_d = rng.choice((8, 10))
        leg_h = rng.choice((5, 6))
        back_h = rng.choice((8, 9, 10))
        recline_deg = rng.choice((-11.25, -7.5, 0.0))

        arm_l = Bone(
            "armrest_l",
            attach=Attach("seat_frame", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0)),
            mirror_name="armrest_r",
            items=[
                Box("arm_front_stump", (1, 3, 1), p_wood.name, Place(None, (0.0, 0.0, 0.0), (1.0, -1.0, -1.0), (0.0, 0.0, 2.0))),
                Box("arm_rail", (2, 1, seat_d - 2), p_wood.name, Place("arm_front_stump", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0), (0.0, 0.0, -1.0))),
            ],
        )
        # Keep primary box without nonzero origin anchor: make arm_rail primary!
        arm_l = Bone(
            "armrest_l",
            attach=Attach("seat_frame", (1.0, 1.0, 1.0), (1.0, -1.0, 1.0)),
            mirror_name="armrest_r",
            items=[
                Box("arm_front_post", (1, 3, 1), p_wood.name),
                Box("arm_top_pad", (2, 1, seat_d - 2), p_upholstery.name, Place("arm_front_post", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
            ],
        )
        backrest = Bone(
            "backrest_frame",
            rest_euler=(recline_deg, 0.0, 0.0),
            attach=Attach("seat_frame", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0)),
            items=[
                Box("lower_back_rail", (seat_w, 2, 1), p_wood.name),
                Box("stile_left", (1, back_h, 1), p_wood.name, Place("lower_back_rail", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
                Box("stile_right", (1, back_h, 1), p_wood.name, Place("lower_back_rail", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
                Box("center_splat", (max(2, seat_w - 4), back_h - 1, 1), p_upholstery.name, Place("lower_back_rail", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("crest_rail", (seat_w, 2, 2), p_trim.name, Place("stile_left", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
            ],
        )
        leg_pair_l = Bone(
            "chair_legs_l",
            attach=Attach("seat_frame", (1.0, -1.0, 0.0), (1.0, 1.0, 0.0)),
            mirror_name="chair_legs_r",
            items=[
                Box("side_apron", (1, 1, seat_d), p_wood.name),
                Box("front_leg", (2, leg_h, 2), p_wood.name, Place("side_apron", (1.0, -1.0, 1.0), (1.0, 1.0, 1.0))),
                Box("back_leg", (2, leg_h, 2), p_wood.name, Place("side_apron", (1.0, -1.0, -1.0), (1.0, 1.0, -1.0))),
                Box("side_stretcher", (1, 1, max(2, seat_d - 4)), p_trim.name, Place("front_leg", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            ],
        )
        root_items = [
            Box("seat_frame", (seat_w, 2, seat_d), p_wood.name),
            Box("seat_cushion", (seat_w - 2, 1, seat_d - 2), p_upholstery.name, Place("seat_frame", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            leg_pair_l,
            armrest_l := arm_l,
            backrest,
        ]
        joint_deg, mirrored = recline_deg, True
        span_a, span_b = seat_w, leg_h + back_h + 4
        notes = f"high-back armchair with upholstered cushion, {recline_deg:g} deg reclined splat backrest, crest rail, mirrored armrests, and stretchered legs"

    elif kind_idx == 2:
        substyle = "ironbound_hinged_chest"
        cw = rng.choice((12, 14))
        cd = rng.choice((8, 10))
        wall_h = rng.choice((5, 6))
        open_deg = rng.choice((0.0, -11.25, -15.0, -22.5))

        lid_bone = Bone(
            "hinged_lid",
            rest_euler=(open_deg, 0.0, 0.0),
            attach=Attach("wall_back", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0)),
            items=[
                Box("lid_rim", (cw, 2, cd), p_wood.name),
                Box("lid_dome", (cw - 2, 2, cd - 2), p_wood.name, Place("lid_rim", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("iron_strap_l", (1, 1, cd - 2), p_iron.name, Place("lid_dome", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
                Box("iron_strap_r", (1, 1, cd - 2), p_iron.name, Place("lid_dome", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
                Box("front_hasp", (2, 2, 1), p_trim.name, Place("lid_rim", (0.0, -1.0, 1.0), (0.0, 1.0, -1.0))),
            ],
        )
        bail_l = Bone(
            "carry_bail_l",
            attach=Attach("chest_floor", (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)),
            mirror_name="carry_bail_r",
            items=[
                Box("handle_plate", (1, 2, 4), p_iron.name),
                Box("handle_grip", (1, 1, 2), p_trim.name, Place("handle_plate", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            ],
        )
        root_items = [
            Box("chest_floor", (cw, 2, cd), p_wood.name),
            Box("foot_nw", (2, 1, 2), p_iron.name, Place("chest_floor", (-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0)), cullface=("down",)),
            Box("foot_ne", (2, 1, 2), p_iron.name, Place("chest_floor", (1.0, -1.0, -1.0), (1.0, 1.0, -1.0)), cullface=("down",)),
            Box("foot_sw", (2, 1, 2), p_iron.name, Place("chest_floor", (-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)), cullface=("down",)),
            Box("foot_se", (2, 1, 2), p_iron.name, Place("chest_floor", (1.0, -1.0, 1.0), (1.0, 1.0, 1.0)), cullface=("down",)),
            Box("wall_back", (cw, wall_h, 2), p_wood.name, Place("chest_floor", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
            Box("wall_front", (cw, wall_h, 2), p_wood.name, Place("chest_floor", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
            Box("wall_west", (2, wall_h, cd - 4), p_wood.name, Place("chest_floor", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
            Box("wall_east", (2, wall_h, cd - 4), p_wood.name, Place("chest_floor", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
            Box("lock_plate", (2, 2, 1), p_trim.name, Place("wall_front", (0.0, -1.0, 1.0), (0.0, -1.0, -1.0))),
            bail_l,
            lid_bone,
        ]
        joint_deg, mirrored = open_deg, True
        span_a, span_b = cw, wall_h + 7
        notes = f"hollow {cw}x{cd} ironbound storage chest with 4 corner feet, recessed walls, lock plate, mirrored carry handles, and {abs(open_deg):g} deg hinged vaulted lid"

    else:
        substyle = "double_door_apothecary_cabinet"
        cab_w = rng.choice((12, 14))
        cab_d = rng.choice((8, 10))
        cab_h = rng.choice((10, 12))
        door_swing = rng.choice((0.0, 15.0, 22.5, 30.0))
        leaf_w = (cab_w - 2) // 2

        door_l = Bone(
            "cabinet_door_rside",
            rest_euler=(0.0, door_swing, 0.0),
            attach=Attach("carcass_floor", (1.0, 1.0, 1.0), (1.0, -1.0, -1.0)),
            mirror_name="cabinet_door_lside",
            items=[
                Box("door_leaf", (leaf_w, cab_h - 1, 1), p_wood.name),
                Box("door_molding", (max(2, leaf_w - 2), cab_h - 3, 1), p_upholstery.name, Place("door_leaf", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.25))),
                Box("brass_knob", (1, 2, 1), p_trim.name, Place("door_molding", (-1.0, 0.0, 1.0), (-1.0, 0.0, -1.0))),
            ],
        )
        root_items = [
            Box("plinth_skirt", (cab_w, 2, cab_d), p_trim.name, cullface=("down",)),
            Box("carcass_floor", (cab_w, 1, cab_d), p_wood.name, Place("plinth_skirt", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("gable_west", (1, cab_h, cab_d - 1), p_wood.name, Place("carcass_floor", (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0))),
            Box("gable_east", (1, cab_h, cab_d - 1), p_wood.name, Place("carcass_floor", (1.0, 1.0, -1.0), (1.0, -1.0, -1.0))),
            Box("back_board", (cab_w - 2, cab_h, 1), p_wood.name, Place("carcass_floor", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
            Box("mid_shelf", (cab_w - 2, 1, cab_d - 2), p_wood.name, Place("back_board", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("cornice_deck", (cab_w, 1, cab_d), p_wood.name, Place("back_board", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
            Box("cornice_crown", (cab_w + 2, 2, cab_d + 2), p_trim.name, Place("cornice_deck", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("shelf_jar_w", (2, 3, 2), p_iron.name, Place("mid_shelf", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (1.0, 0.0, 0.0))),
            Box("shelf_jar_e", (2, 3, 2), p_trim.name, Place("mid_shelf", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 0.0, 0.0))),
            door_l,
        ]
        joint_deg, mirrored = door_swing, True
        span_a, span_b = cab_w + 2, cab_h + 6
        notes = f"double-door cabinet with plinth, side gables, interior shelf with storage jars, crown cornice, and mirrored hinged doors swung {door_swing:g} deg outward"

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_wood, p_trim, p_iron, p_upholstery],
        root=Bone("furniture_root", pivot_local=_jitter_pivot(rng, (8.0, 6.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": span_a,
        "secondary_span": span_b,
        "joint_angle_deg": joint_deg,
        "has_mirror_x": mirrored,
        "has_detached_bone": False,
        "inflate_detail": 0.0,
        "material_theme": f"{p_wood.name}+{p_trim.name}+{p_upholstery.name}",
        "notes": notes,
        "extra": {"furniture_kind": kind_idx},
    }


def _gen_gearbox(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    kind_idx = v_idx % 3
    profile = "java_modern" if (v_idx // 3) % 3 == 0 else "bbmodel_5"
    density = 32 if (v_idx % 7 == 6) else 16

    p_case = _make_palette(rng, rng.choice(_METALS))
    p_gear = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_shaft = _make_palette(rng, rng.choice(_METALS), suffix="_shaft")
    p_gauge = _make_palette(rng, rng.choice(_GLOW_MATERIALS), emissive=(v_idx % 2 == 0))

    phase_deg = rng.choice((-45.0, -30.0, -22.5, -15.0, 0.0, 15.0, 22.5, 30.0, 45.0))

    if kind_idx == 0:
        substyle = "twin_shaft_spur_reducer"
        cw = rng.choice((10, 12))
        ch = rng.choice((8, 10))
        cd = rng.choice((8, 10))
        tooth_len = rng.choice((2, 3))

        rotor_front = Bone(
            "front_spur_rotor",
            rest_euler=(0.0, 0.0, phase_deg),
            attach=Attach("casing_body", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            items=[
                Box("front_journal", (4, 4, 1), p_case.name),
                Box("input_shaft", (2, 2, 2), p_shaft.name, Place("front_journal", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                Box("gear_hub", (4, 4, 2), p_gear.name, Place("input_shaft", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                Box("tooth_up", (2, tooth_len, 2), p_gear.name, Place("gear_hub", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("tooth_down", (2, tooth_len, 2), p_gear.name, Place("gear_hub", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
                Box("tooth_east", (tooth_len, 2, 2), p_gear.name, Place("gear_hub", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                Box("tooth_west", (tooth_len, 2, 2), p_gear.name, Place("gear_hub", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            ],
        )
        bearing_l = Bone(
            "side_out_bearing_l",
            attach=Attach("casing_body", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
            mirror_name="side_out_bearing_r",
            items=[
                Box("bearing_flange", (1, 4, 4), p_case.name),
                Box("stub_axle", (3, 2, 2), p_shaft.name, Place("bearing_flange", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            ],
        )
        root_items: list[Box | Bone] = [
            Box("mounting_bed", (cw + 2, 2, cd + 2), p_case.name, cullface=("down",)),
            Box("casing_body", (cw, ch, cd), p_case.name, Place("mounting_bed", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("inspection_cover", (cw - 2, 1, cd - 2), p_gear.name, Place("casing_body", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("oil_breather", (2, 2, 2), p_shaft.name, Place("inspection_cover", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("sight_gauge", (2, 4, 1), p_gauge.name, Place("casing_body", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.2))),
            rotor_front,
            bearing_l,
        ]
        mirrored = True
        span_a, span_b = cw + 8, ch + 5
        notes = f"industrial spur reducer casing ({cw}x{ch}x{cd}) with oil breather, sight gauge, mirrored side axle bearings, and 4-tooth rotor at {phase_deg:g} deg"

    elif kind_idx == 1:
        substyle = "bevel_handwheel_actuator"
        body_s = rng.choice((8, 10))
        yaw_deg = rng.choice((-30.0, -15.0, 0.0, 15.0, 30.0))
        wheel_r = rng.choice((3, 4))

        turret = Bone(
            "top_bevel_turret",
            rest_euler=(0.0, yaw_deg, 0.0),
            attach=Attach("bonnet_cap", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
            items=[
                Box("turret_collar", (4, 2, 4), p_gear.name),
                Box("vertical_spindle", (2, 4, 2), p_shaft.name, Place("turret_collar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("coupling_plate", (6, 1, 6), p_gear.name, Place("vertical_spindle", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("drive_dog_n", (2, 1, 1), p_shaft.name, Place("coupling_plate", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
                Box("drive_dog_s", (2, 1, 1), p_shaft.name, Place("coupling_plate", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
            ],
        )
        handwheel = Bone(
            "front_handwheel",
            rest_euler=(0.0, 0.0, phase_deg),
            attach=Attach("gear_cube", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            items=[
                Box("worm_stem", (2, 2, 3), p_shaft.name),
                Box("wheel_boss", (3, 3, 1), p_gear.name, Place("worm_stem", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                Box("spoke_n", (1, wheel_r, 1), p_gear.name, Place("wheel_boss", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("spoke_s", (1, wheel_r, 1), p_gear.name, Place("wheel_boss", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
                Box("spoke_e", (wheel_r, 1, 1), p_gear.name, Place("wheel_boss", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                Box("spoke_w", (wheel_r, 1, 1), p_gear.name, Place("wheel_boss", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
                Box("crank_knob", (1, 1, 3), p_gauge.name, Place("spoke_n", (0.0, 1.0, 1.0), (0.0, 1.0, -1.0))),
            ],
        )
        root_items = [
            Box("sole_plate", (body_s + 2, 2, body_s + 2), p_case.name, cullface=("down",)),
            Box("gear_cube", (body_s, body_s, body_s), p_case.name, Place("sole_plate", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("bonnet_cap", (body_s - 2, 2, body_s - 2), p_case.name, Place("gear_cube", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("rear_pipe_port", (4, 4, 2), p_gear.name, Place("gear_cube", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("status_lamp", (2, 2, 1), p_gauge.name, Place("bonnet_cap", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            turret,
            handwheel,
        ]
        mirrored = False
        span_a, span_b = body_s + 4, body_s + 11
        notes = f"bevel gearbox actuator with vertical output coupling turret ({yaw_deg:g} deg yaw) and 4-spoke crank handwheel ({phase_deg:g} deg roll)"

    else:
        substyle = "dual_stage_clockwork_frame"
        fw = rng.choice((12, 14))
        fh = rng.choice((8, 10))
        roll_b = rng.choice((-22.5, 15.0, 30.0))

        pinion_front = Bone(
            "primary_pinion",
            rest_euler=(0.0, 0.0, phase_deg),
            attach=Attach("front_stanchion", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
            items=[
                Box("p1_arbor", (2, 2, 2), p_shaft.name),
                Box("p1_disc", (4, 4, 1), p_gear.name, Place("p1_arbor", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
                Box("p1_lug_u", (2, 2, 1), p_gear.name, Place("p1_disc", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("p1_lug_d", (2, 2, 1), p_gear.name, Place("p1_disc", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
                Box("p1_lug_e", (2, 2, 1), p_gear.name, Place("p1_disc", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                Box("p1_lug_w", (2, 2, 1), p_gear.name, Place("p1_disc", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            ],
        )
        flywheel_rear = Bone(
            "secondary_flywheel",
            rest_euler=(0.0, 0.0, roll_b),
            attach=Attach("rear_stanchion", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)),
            items=[
                Box("p2_arbor", (2, 2, 2), p_shaft.name),
                Box("p2_hub", (3, 3, 2), p_case.name, Place("p2_arbor", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
                Box("p2_vane_u", (1, 3, 2), p_gear.name, Place("p2_hub", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
                Box("p2_vane_d", (1, 3, 2), p_gear.name, Place("p2_hub", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
                Box("p2_vane_e", (3, 1, 2), p_gear.name, Place("p2_hub", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
                Box("p2_vane_w", (3, 1, 2), p_gear.name, Place("p2_hub", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            ],
        )
        root_items = [
            Box("clockwork_bed", (fw, 2, 10), p_case.name, cullface=("down",)),
            Box("front_stanchion", (fw - 2, fh, 2), p_case.name, Place("clockwork_bed", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
            Box("rear_stanchion", (fw - 2, fh, 2), p_case.name, Place("clockwork_bed", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
            Box("spacer_beam", (4, 2, 6), p_gear.name, Place("clockwork_bed", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("governor_core", (3, 3, 3), p_gauge.name, Place("spacer_beam", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("top_tie_bridge", (fw - 2, 1, 10), p_case.name, Place("front_stanchion", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
            pinion_front,
            flywheel_rear,
        ]
        mirrored = False
        span_a, span_b = fw, fh + 4
        notes = f"open-frame dual-stage clockwork drive with front pinion ({phase_deg:g} deg), rear flywheel ({roll_b:g} deg), tie bridge, and central governor core"

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_case, p_gear, p_shaft, p_gauge],
        root=Bone("gearbox_root", pivot_local=_jitter_pivot(rng, (8.0, 4.5, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": span_a,
        "secondary_span": span_b,
        "joint_angle_deg": phase_deg,
        "has_mirror_x": mirrored,
        "has_detached_bone": False,
        "inflate_detail": 0.0,
        "material_theme": f"{p_case.name}+{p_gear.name}+{p_shaft.name}",
        "notes": notes,
        "extra": {"gearbox_kind": kind_idx},
    }


# ==============================================================================
# 6 Held-Out Unseen Static Family Generators (Strictly >= 16 lowered boxes!)
# ==============================================================================


def _gen_mechanical_crossbow(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyle = ("arbalest_windlass_crossbow", "compound_cam_crossbow", "siege_latch_crossbow")[v_idx % 3]
    profile = "java_modern" if (v_idx % 4 == 0) else "bbmodel_5"
    density = 32 if (v_idx % 5 == 1) else 16

    p_stock = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_prod = _make_palette(rng, rng.choice(_METALS))
    p_brass = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_cord = _make_palette(rng, rng.choice(_STONES_AND_FABRICS))

    tiller_fore = rng.choice((5, 6, 7))
    tiller_mid = rng.choice((5, 6))
    prod_sweep = rng.choice((0.0, -11.25, -15.0, -22.5))
    limb_seg = rng.choice((3, 4))

    prod_l = Bone(
        "prod_limb_l",
        rest_euler=(0.0, 0.0, prod_sweep),
        attach=Attach("prod_socket", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        mirror_name="prod_limb_r",
        items=[
            Box("limb_inner", (limb_seg, 1, 2), p_prod.name),
            Box("limb_outer", (limb_seg, 1, 2), p_prod.name, Place("limb_inner", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("pulley_bracket", (2, 1, 2), p_brass.name, Place("limb_outer", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("eccentric_cam", (2, 1, 3), p_prod.name, Place("pulley_bracket", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("cable_guide", (1, 2, 1), p_cord.name, Place("limb_inner", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("butt_stock", (2, 4, 3), p_stock.name),
        Box("cheek_piece", (2, 3, 1), p_brass.name, Place("butt_stock", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("pistol_grip", (2, 2, 4), p_stock.name, Place("butt_stock", (0.0, 1.0, -1.0), (0.0, -1.0, 1.0))),
        Box("trigger_guard", (1, 3, 1), p_brass.name, Place("pistol_grip", (0.0, 1.0, 1.0), (0.0, -1.0, 1.0))),
        Box("tiller_mid", (2, tiller_mid, 2), p_stock.name, Place("butt_stock", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("nut_housing", (3, 2, 2), p_brass.name, Place("tiller_mid", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("tiller_fore", (2, tiller_fore, 2), p_stock.name, Place("nut_housing", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("flight_rail", (2, tiller_fore, 1), p_prod.name, Place("tiller_fore", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
        Box("prod_socket", (4, 2, 2), p_brass.name, Place("tiller_fore", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("stirrup_yoke", (4, 1, 2), p_prod.name, Place("prod_socket", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("stirrup_arm_l", (1, 3, 1), p_prod.name, Place("stirrup_yoke", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
        Box("stirrup_arm_r", (1, 3, 1), p_prod.name, Place("stirrup_yoke", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
        Box("stirrup_toe_bar", (4, 1, 1), p_prod.name, Place("stirrup_arm_l", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
        prod_l,
    ]
    # 13 root boxes + 2 * 5 mirrored prod boxes = 23 boxes (>= 16!), 3 bones!

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_stock, p_prod, p_brass, p_cord],
        root=Bone("crossbow_tiller_root", pivot_local=_jitter_pivot(rng, (8.0, 3.5, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": tiller_mid + tiller_fore + 13,
        "secondary_span": 4 + 2 * (2 * limb_seg + 2),
        "joint_angle_deg": prod_sweep,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": 0.0,
        "material_theme": f"{p_stock.name}+{p_prod.name}+{p_brass.name}",
        "notes": f"held-out {substyle.replace('_', ' ')} with cheek piece, trigger guard, bolt flight rail, foot stirrup, and mirrored cam-equipped steel prods ({prod_sweep:g} deg)",
        "extra": {"limb_seg": limb_seg},
    }


def _gen_hand_lantern(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyle = ("storm_cage_lantern", "watchman_brass_lantern", "miners_beacon_lantern")[v_idx % 3]
    profile = "java_modern" if (v_idx % 4 == 1) else "bbmodel_5"
    density = 32 if (v_idx % 6 == 0) else 16

    p_frame = _make_palette(rng, rng.choice(_METALS))
    p_brass = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_flame = _make_palette(rng, rng.choice(_GLOW_MATERIALS), emissive=True)

    base_s = rng.choice((8, 10))
    cage_h = rng.choice((7, 8, 9))
    bail_tilt = rng.choice((-22.5, -15.0, 0.0, 15.0, 22.5))

    bail_bone = Bone(
        "swing_bail",
        rest_euler=(bail_tilt, 0.0, 0.0),
        attach=Attach("chimney_crown", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("bail_yoke", (6, 1, 1), p_brass.name),
            Box("bail_post_w", (1, 3, 1), p_brass.name, Place("bail_yoke", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
            Box("bail_post_e", (1, 3, 1), p_brass.name, Place("bail_yoke", (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))),
            Box("bail_grip_bar", (6, 1, 1), p_frame.name, Place("bail_post_w", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
        ],
    )

    canopy_bone = Bone(
        "lantern_canopy",
        attach=Attach("mullion_nw", (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0)),
        items=[
            Box("hood_deck", (base_s, 1, base_s), p_frame.name),
            Box("vent_drum", (base_s - 4, 2, base_s - 4), p_brass.name, Place("hood_deck", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("chimney_crown", (base_s - 2, 1, base_s - 2), p_frame.name, Place("vent_drum", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            bail_bone,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("fount_base", (base_s, 2, base_s), p_frame.name),
        Box("foot_nw", (2, 1, 2), p_brass.name, Place("fount_base", (-1.0, -1.0, -1.0), (-1.0, 1.0, -1.0)), cullface=("down",)),
        Box("foot_ne", (2, 1, 2), p_brass.name, Place("fount_base", (1.0, -1.0, -1.0), (1.0, 1.0, -1.0)), cullface=("down",)),
        Box("foot_sw", (2, 1, 2), p_brass.name, Place("fount_base", (-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0)), cullface=("down",)),
        Box("foot_se", (2, 1, 2), p_brass.name, Place("fount_base", (1.0, -1.0, 1.0), (1.0, 1.0, 1.0)), cullface=("down",)),
        Box("oil_pot", (4, 2, 4), p_brass.name, Place("fount_base", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("burner_cone", (2, 1, 2), p_brass.name, Place("oil_pot", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("mantle_glow", (2, max(3, cage_h - 4), 2), p_flame.name, Place("burner_cone", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("mullion_nw", (1, cage_h, 1), p_frame.name, Place("fount_base", (-1.0, 1.0, -1.0), (-1.0, -1.0, -1.0))),
        Box("mullion_ne", (1, cage_h, 1), p_frame.name, Place("fount_base", (1.0, 1.0, -1.0), (1.0, -1.0, -1.0))),
        Box("mullion_sw", (1, cage_h, 1), p_frame.name, Place("fount_base", (-1.0, 1.0, 1.0), (-1.0, -1.0, 1.0))),
        Box("mullion_se", (1, cage_h, 1), p_frame.name, Place("fount_base", (1.0, 1.0, 1.0), (1.0, -1.0, 1.0))),
        canopy_bone,
    ]
    # 12 + 3 + 4 = 19 boxes (>= 16!), 3 bones!

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_frame, p_brass, p_flame],
        root=Bone("lantern_frame_root", pivot_local=_jitter_pivot(rng, (8.0, 5.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": cage_h + 10,
        "secondary_span": base_s,
        "joint_angle_deg": bail_tilt,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": 0.0,
        "material_theme": f"{p_frame.name}+{p_brass.name}+{p_flame.name}",
        "notes": f"held-out {substyle.replace('_', ' ')} with 4 feet, oil pot, emissive mantle, 4 corner cage mullions, vented chimney hood, and {bail_tilt:g} deg swinging bail handle",
        "extra": {"cage_h": cage_h},
    }


def _gen_tripod_telescope(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyle = ("brass_refractor_telescope", "observatory_survey_scope", "navigator_zenith_scope")[v_idx % 3]
    profile = "java_modern" if (v_idx % 4 == 2) else "bbmodel_5"
    density = 32 if (v_idx % 5 == 2) else 16

    p_brass = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_wood = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_iron = _make_palette(rng, rng.choice(_METALS))
    p_lens = _make_palette(rng, rng.choice(_GLOW_MATERIALS), emissive=True)

    leg_seg = rng.choice((4, 5))
    ota_len = rng.choice((6, 7, 8))
    elev_deg = rng.choice((-22.5, -15.0, -11.25, 0.0, 11.25, 15.0))
    splay_deg = rng.choice((11.25, 15.0, 22.5))

    leg_front = Bone(
        "tripod_leg_north",
        rest_euler=(splay_deg, 0.0, 0.0),
        attach=Attach("azimuth_hub", (0.0, -1.0, -1.0), (0.0, 1.0, -1.0)),
        items=[
            Box("leg_n_upper", (1, leg_seg, 1), p_wood.name),
            Box("leg_n_lower", (1, leg_seg, 1), p_wood.name, Place("leg_n_upper", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("leg_n_ferrule", (2, 1, 2), p_brass.name, Place("leg_n_lower", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )
    leg_rear_l = Bone(
        "tripod_leg_se",
        rest_euler=(-splay_deg, 0.0, splay_deg),
        attach=Attach("azimuth_hub", (1.0, -1.0, 1.0), (1.0, 1.0, 1.0)),
        mirror_name="tripod_leg_sw",
        items=[
            Box("leg_s_upper", (1, leg_seg, 1), p_wood.name),
            Box("leg_s_lower", (1, leg_seg, 1), p_wood.name, Place("leg_s_upper", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("leg_s_ferrule", (2, 1, 2), p_brass.name, Place("leg_s_lower", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        ],
    )
    ota_bone = Bone(
        "optical_tube_assembly",
        rest_euler=(elev_deg, 0.0, 0.0),
        attach=Attach("altitude_pillar", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("cradle_saddle", (4, 1, 4), p_iron.name),
            Box("main_barrel", (3, 3, ota_len), p_brass.name, Place("cradle_saddle", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("dew_hood", (4, 4, 4), p_brass.name, Place("main_barrel", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("objective_glass", (3, 3, 1), p_lens.name, Place("dew_hood", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -0.25))),
            Box("focuser_housing", (2, 2, 3), p_iron.name, Place("main_barrel", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("star_diagonal", (1, 2, 2), p_brass.name, Place("focuser_housing", (0.0, 0.0, -1.0), (0.0, -1.0, 1.0))),
            Box("eyepiece_barrel", (1, 2, 1), p_iron.name, Place("star_diagonal", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("finder_stalk", (1, 1, 1), p_iron.name, Place("main_barrel", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("finder_scope", (1, 1, 4), p_brass.name, Place("finder_stalk", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("azimuth_hub", (5, 2, 5), p_iron.name),
        Box("altitude_pillar", (3, 3, 3), p_brass.name, Place("azimuth_hub", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        leg_front,
        leg_rear_l,
        ota_bone,
    ]
    # 2 + 3 + 2*3 + 9 = 20 boxes (>= 16!), 5 bones!

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_brass, p_wood, p_iron, p_lens],
        root=Bone("tripod_stand_root", pivot_local=_jitter_pivot(rng, (8.0, 11.0, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": 2 * leg_seg + 10,
        "secondary_span": ota_len + 8,
        "joint_angle_deg": elev_deg,
        "has_mirror_x": True,
        "has_detached_bone": False,
        "inflate_detail": 0.0,
        "material_theme": f"{p_brass.name}+{p_wood.name}+{p_lens.name}",
        "notes": f"held-out {substyle.replace('_', ' ')} with 3 splayed wooden legs ({splay_deg:g} deg), azimuth/altitude mount, dew hood, emissive objective lens, star diagonal eyepiece, and finderscope ({elev_deg:g} deg elevation)",
        "extra": {"ota_len": ota_len, "splay_deg": splay_deg},
    }


def _gen_folding_windmill(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyle = ("smock_sail_windmill", "stone_tower_gristmill", "timber_fantail_windmill")[v_idx % 3]
    profile = "java_modern" if (v_idx % 4 == 3) else "bbmodel_5"
    density = 32 if (v_idx % 6 == 3) else 16

    p_tower = _make_palette(rng, rng.choice(_STONES_AND_FABRICS))
    p_timber = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_sail = _make_palette(rng, rng.choice(_STONES_AND_FABRICS), suffix="_sail")
    p_iron = _make_palette(rng, rng.choice(_METALS))

    low_h = rng.choice((5, 6))
    mid_h = rng.choice((5, 6))
    spar_len = rng.choice((6, 7, 8))
    sail_len = spar_len - 2
    roll_deg = rng.choice((-30.0, -22.5, -15.0, 0.0, 15.0, 22.5, 30.0, 45.0))

    rotor = Bone(
        "windshaft_rotor",
        rest_euler=(0.0, 0.0, roll_deg),
        attach=Attach("windshaft_neck", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)),
        items=[
            Box("poll_end_hub", (2, 2, 2), p_iron.name),
            Box("whip_up", (1, spar_len, 1), p_timber.name, Place("poll_end_hub", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("sail_sheet_up", (2, sail_len, 1), p_sail.name, Place("whip_up", (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))),
            Box("whip_down", (1, spar_len, 1), p_timber.name, Place("poll_end_hub", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
            Box("sail_sheet_down", (2, sail_len, 1), p_sail.name, Place("whip_down", (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0))),
            Box("whip_east", (spar_len, 1, 1), p_timber.name, Place("poll_end_hub", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("sail_sheet_east", (sail_len, 2, 1), p_sail.name, Place("whip_east", (1.0, -1.0, 0.0), (1.0, 1.0, 0.0))),
            Box("whip_west", (spar_len, 1, 1), p_timber.name, Place("poll_end_hub", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            Box("sail_sheet_west", (sail_len, 2, 1), p_sail.name, Place("whip_west", (-1.0, 1.0, 0.0), (-1.0, -1.0, 0.0))),
        ],
    )

    cap_yaw = rng.choice((-15.0, 0.0, 15.0))
    cap_bone = Bone(
        "revolving_cap",
        rest_euler=(0.0, cap_yaw, 0.0),
        attach=Attach("upper_smock", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)),
        items=[
            Box("cap_curb", (6, 3, 8), p_timber.name),
            Box("cap_ridge", (4, 2, 8), p_timber.name, Place("cap_curb", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("fantail_outrigger", (1, 1, 5), p_timber.name, Place("cap_curb", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("fantail_wheel", (1, 4, 3), p_sail.name, Place("fantail_outrigger", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("windshaft_neck", (3, 3, 2), p_iron.name, Place("cap_curb", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            rotor,
        ],
    )

    root_items: list[Box | Bone] = [
        Box("plinth_foundation", (10, 2, 10), p_tower.name, cullface=("down",)),
        Box("door_step", (4, 1, 2), p_tower.name, Place("plinth_foundation", (0.0, -1.0, 1.0), (0.0, -1.0, -1.0))),
        Box("lower_octagon", (8, low_h, 8), p_tower.name, Place("plinth_foundation", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("reefing_stage", (10, 1, 10), p_timber.name, Place("lower_octagon", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("upper_smock", (6, mid_h, 6), p_tower.name, Place("reefing_stage", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        cap_bone,
    ]
    # 5 + 5 + 9 = 19 boxes (>= 16!), 3 bones!

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_tower, p_timber, p_sail, p_iron],
        root=Bone("windmill_tower_root", pivot_local=_jitter_pivot(rng, (8.0, 1.5, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": low_h + mid_h + 10,
        "secondary_span": 2 * spar_len + 2,
        "joint_angle_deg": roll_deg,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": 0.0,
        "material_theme": f"{p_tower.name}+{p_timber.name}+{p_sail.name}",
        "notes": f"held-out {substyle.replace('_', ' ')} with stepped masonry tower, reefing stage, {cap_yaw:g} deg revolving cap with rear fantail, and 4-sail rotor at {roll_deg:g} deg",
        "extra": {"spar_len": spar_len, "cap_yaw": cap_yaw},
    }


def _gen_anvil_forge(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyle = ("blacksmith_hearth_and_anvil", "armorer_chimney_forge", "farrier_coal_forge")[v_idx % 3]
    profile = "java_modern" if (v_idx % 4 == 0) else "bbmodel_5"
    density = 32 if (v_idx % 5 == 4) else 16

    p_stone = _make_palette(rng, rng.choice(_STONES_AND_FABRICS))
    p_iron = _make_palette(rng, rng.choice(_METALS))
    p_wood = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_ember = _make_palette(rng, rng.choice(_GLOW_MATERIALS), emissive=True)

    hearth_w = rng.choice((10, 12))
    flue_h = rng.choice((5, 6, 7))
    anvil_yaw = rng.choice((-15.0, 0.0, 15.0))

    anvil_bone = Bone(
        "anvil_workstation",
        rest_euler=(0.0, anvil_yaw, 0.0),
        attach=Attach("hearth_plinth", (0.0, -1.0, 1.0), (0.0, -1.0, -1.0)),
        items=[
            Box("stump_pedestal", (6, 4, 6), p_wood.name, cullface=("down",)),
            Box("anvil_foot", (6, 1, 4), p_iron.name, Place("stump_pedestal", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("anvil_waist", (3, 2, 3), p_iron.name, Place("anvil_foot", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("anvil_body", (6, 3, 4), p_iron.name, Place("anvil_waist", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("bick_horn", (3, 2, 2), p_iron.name, Place("anvil_body", (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0))),
            Box("horn_point", (2, 1, 1), p_iron.name, Place("bick_horn", (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0))),
            Box("hardy_heel", (3, 2, 3), p_iron.name, Place("anvil_body", (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))),
            Box("hardened_face", (6, 1, 4), p_iron.name, Place("anvil_body", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
            Box("quench_slack_tub", (3, 4, 3), p_wood.name, Place("stump_pedestal", (1.0, -1.0, 1.0), (1.0, -1.0, -1.0))),
        ],
    )

    hood_bone = Bone(
        "exhaust_hood_assembly",
        attach=Attach("fireback_wall", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0)),
        items=[
            Box("canopy_hood", (hearth_w, 2, 7), p_iron.name),
            Box("chimney_flue", (4, flue_h, 4), p_stone.name, Place("canopy_hood", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    root_items: list[Box | Bone] = [
        Box("hearth_plinth", (hearth_w, 3, 10), p_stone.name, cullface=("down",)),
        Box("fireback_wall", (hearth_w, 6, 2), p_stone.name, Place("hearth_plinth", (0.0, 1.0, -1.0), (0.0, -1.0, -1.0))),
        Box("cheek_wall_w", (2, 5, 6), p_stone.name, Place("fireback_wall", (-1.0, -1.0, 1.0), (-1.0, -1.0, -1.0))),
        Box("cheek_wall_e", (2, 5, 6), p_stone.name, Place("fireback_wall", (1.0, -1.0, 1.0), (1.0, -1.0, -1.0))),
        Box("cinder_bed", (hearth_w - 4, 2, 6), p_iron.name, Place("hearth_plinth", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("live_coals", (4, 1, 4), p_ember.name, Place("cinder_bed", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("tuyere_pipe", (2, 2, 3), p_iron.name, Place("fireback_wall", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
        hood_bone,
        anvil_bone,
    ]
    # 7 + 2 + 9 = 18 boxes (>= 16!), 3 bones!

    prog = Program(
        meta=Meta("block_multipart", profile, density),
        palettes=[p_stone, p_iron, p_wood, p_ember],
        root=Bone("forge_hearth_root", pivot_local=_jitter_pivot(rng, (8.0, 3.0, 5.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": flue_h + 11,
        "secondary_span": hearth_w + 3,
        "joint_angle_deg": anvil_yaw,
        "has_mirror_x": False,
        "has_detached_bone": False,
        "inflate_detail": 0.0,
        "material_theme": f"{p_stone.name}+{p_iron.name}+{p_ember.name}",
        "notes": f"held-out {substyle.replace('_', ' ')} with masonry hearth, tuyere pipe, emissive coal bed, smoke hood & chimney flue, plus oak-stump London anvil ({anvil_yaw:g} deg yaw) and slack tub",
        "extra": {"hearth_w": hearth_w, "flue_h": flue_h},
    }


def _gen_floating_ring_staff(rng: random.Random, v_idx: int) -> tuple[Program, dict[str, Any]]:
    substyle = ("celestial_orrery_staff", "aether_halo_scepter", "void_astrolabe_staff")[v_idx % 3]
    profile = "java_modern" if (v_idx % 4 == 1) else "bbmodel_5"
    density = 32 if (v_idx % 5 == 0) else 16

    p_shaft = _make_palette(rng, rng.choice(_WOODS_AND_GRIPS))
    p_metal = _make_palette(rng, rng.choice(_ACCENT_METALS))
    p_glow = _make_palette(rng, rng.choice(_GLOW_MATERIALS), emissive=True)

    low_len = rng.choice((4, 5))
    mid_len = rng.choice((4, 5))
    use_detached = (v_idx % 2 == 0)
    float_gap = round(rng.choice((0.8, 1.1, 1.4)), 2) if use_detached else 0.0
    halo_yaw = rng.choice((-30.0, -15.0, 0.0, 15.0, 30.0, 45.0))
    halo_tilt = rng.choice((-15.0, 0.0, 15.0))
    wrap_inf = round(rng.choice((0.0, 0.05, 0.08)), 3)

    crown_bone = Bone(
        "floating_halo_crown",
        rest_euler=(halo_tilt, halo_yaw, 0.0),
        attach=Attach("apex_spire", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, float_gap, 0.0), detached=use_detached),
        items=[
            Box("orrery_core", (2, 2, 2), p_glow.name),
            Box("spoke_north", (1, 1, 2), p_metal.name, Place("orrery_core", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("spoke_south", (1, 1, 2), p_metal.name, Place("orrery_core", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("spoke_west", (2, 1, 1), p_metal.name, Place("orrery_core", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            Box("spoke_east", (2, 1, 1), p_metal.name, Place("orrery_core", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("ring_arc_north", (6, 1, 1), p_metal.name, Place("spoke_north", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))),
            Box("ring_arc_south", (6, 1, 1), p_metal.name, Place("spoke_south", (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            Box("ring_arc_west", (1, 1, 6), p_metal.name, Place("spoke_west", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))),
            Box("ring_arc_east", (1, 1, 6), p_metal.name, Place("spoke_east", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0))),
            Box("zenith_finial", (1, 2, 1), p_glow.name, Place("orrery_core", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        ],
    )

    prong_l = Bone(
        "calyx_prong_l",
        rest_euler=(0.0, 0.0, rng.choice((-15.0, -22.5, -30.0))),
        attach=Attach("astragal_disc", (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)),
        mirror_name="calyx_prong_r",
        items=[Box("calyx_tine", (1, 4, 1), p_metal.name)],
    )

    root_items: list[Box | Bone] = [
        Box("spiral_grip", (2, 3, 2), p_metal.name, inflate=wrap_inf),
        Box("shaft_lower", (2, low_len, 2), p_shaft.name, Place("spiral_grip", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("butt_collar", (3, 1, 3), p_metal.name, Place("shaft_lower", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("end_crystal", (1, 2, 1), p_glow.name, Place("butt_collar", (0.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        Box("shaft_upper", (2, mid_len, 2), p_shaft.name, Place("spiral_grip", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("astragal_disc", (4, 1, 4), p_metal.name, Place("shaft_upper", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("prism_pedestal", (2, 1, 2), p_metal.name, Place("astragal_disc", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("focus_prism", (2, 3, 2), p_glow.name, Place("prism_pedestal", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        Box("apex_spire", (1, 2, 1), p_metal.name, Place("focus_prism", (0.0, 1.0, 0.0), (0.0, -1.0, 0.0))),
        prong_l,
        crown_bone,
    ]
    # 9 + 2 + 10 = 21 boxes (>= 16!), 4 bones!

    prog = Program(
        meta=Meta("weapon_3d", profile, density),
        palettes=[p_shaft, p_metal, p_glow],
        root=Bone("staff_shaft_root", pivot_local=_jitter_pivot(rng, (8.0, 7.5, 8.0)), items=root_items),
    )
    return prog, {
        "substyle": substyle,
        "primary_span": low_len + mid_len + 17,
        "secondary_span": 8,
        "joint_angle_deg": halo_yaw,
        "has_mirror_x": True,
        "has_detached_bone": use_detached,
        "inflate_detail": wrap_inf,
        "material_theme": f"{p_shaft.name}+{p_metal.name}+{p_glow.name}",
        "notes": f"held-out {substyle.replace('_', ' ')} with butt crystal, astragal disc, mirrored calyx prongs, focus prism, and {'detached hovering' if use_detached else 'mounted'} 8-segment halo ring ({halo_yaw:g} deg yaw)",
        "extra": {"float_gap": float_gap, "halo_tilt": halo_tilt},
    }


_FAMILY_DISPATCH = {
    "straight_blade_sword": _gen_straight_blade_sword,
    "single_edge_blade": _gen_single_edge_blade,
    "battle_axe": _gen_battle_axe,
    "spear": _gen_spear,
    "war_hammer": _gen_war_hammer,
    "simple_wood_bow": _gen_simple_wood_bow,
    "furniture_table_chair_chest_cabinet": _gen_furniture,
    "gearbox": _gen_gearbox,
    "mechanical_crossbow": _gen_mechanical_crossbow,
    "hand_lantern": _gen_hand_lantern,
    "tripod_telescope": _gen_tripod_telescope,
    "folding_windmill": _gen_folding_windmill,
    "anvil_forge": _gen_anvil_forge,
    "floating_ring_staff": _gen_floating_ring_staff,
    **MULTI_DOMAIN_DISPATCH,
}

_IDENT_SCAN_RE = re.compile(r"\b(?:box|bone)\s+([A-Za-z_][A-Za-z0-9_]*)")


def generate_candidate_for_family(
    family_id: str,
    *,
    split: str,
    variant_index: int,
    base_seed: int = 2026,
) -> tuple[Program, dict[str, Any], int]:
    """Generate a single deterministic typed Program AST and metadata for a family."""
    if family_id not in _FAMILY_DISPATCH:
        raise ValueError(f"Unknown family_id: {family_id!r}")
    rng, numeric_seed = _derive_rng(family_id, split, variant_index, base_seed)
    program, meta_info = _FAMILY_DISPATCH[family_id](rng, variant_index)
    return program, meta_info, numeric_seed


def verify_and_package_candidate(
    family_id: str,
    *,
    split: str,
    variant_index: int,
    base_seed: int = 2026,
    max_attempts: int = 16,
    require_min_boxes: int | None = None,
    run_exporter_check: bool = False,
) -> dict[str, Any]:
    """Generate, diversify, compile, 15-axis SAT + P1-02 contact-audit, and package a verified row."""
    min_boxes = require_min_boxes if require_min_boxes is not None else (16 if split == "test_unseen" else 12)
    domain = FAMILY_DOMAINS.get(family_id, "artificial")
    body_plan_id = ENTITY_BODY_PLANS.get(family_id)
    last_error = "uninitialized"

    for attempt in range(max_attempts):
        effective_seed = base_seed + attempt * 104_729
        try:
            raw_program, meta_info, numeric_seed = generate_candidate_for_family(
                family_id, split=split, variant_index=variant_index, base_seed=effective_seed
            )
            div_rng = random.Random(numeric_seed ^ 0x3C3C_7171)
            program, div_info = diversify_program_ast(
                raw_program,
                rng=div_rng,
                domain=domain,
                variant_index=variant_index,
            )
            validate_program(program)
            canonical_dsl = format_program(program)
            # Round-trip parse guarantee on the canonical serialized program
            reparsed = parse_program(canonical_dsl)
            scene = compile_geometry(reparsed)
            if len(scene.boxes) < min_boxes:
                last_error = f"box_count {len(scene.boxes)} < {min_boxes}"
                continue

            is_entity = reparsed.meta.archetype == "entity"
            if is_entity:
                if not _check_entity_envelope(scene):
                    last_error = "exceeded_entity_[-48,64]_envelope"
                    continue
            else:
                if not _check_java_envelope(scene):
                    last_error = "exceeded_java_[-16,32]_envelope"
                    continue

            audit_res = audit_scene(
                scene,
                collision_tolerance=1e-7,
                enforce_physical_contact=True,
                max_contact_gap=1e-4,
                max_inset_depth=0.5,
            )
            if not audit_res["passed"]:
                last_error = (
                    f"audit_failed(collisions={len(audit_res['collision_pairs'])}, "
                    f"contact_violations={len(audit_res['contact_violations'])})"
                )
                continue

            if run_exporter_check:
                from synccraft.backends.bbmodel5_exporter import export_bbmodel
                from synccraft.backends.java_exporter import export_java
                from synccraft.shading.uv_packer import build_atlas

                atlas, _ = build_atlas(reparsed, scene)
                export_bbmodel(scene, name=f"{family_id}_{variant_index}", atlas=atlas)
                if not is_entity:
                    export_java(scene, profile="java_modern", atlas=atlas)

            prompt_rng = random.Random(numeric_seed ^ 0x5A5A_1234)
            prompt_text, prompt_style = compose_multilevel_prompt(
                prompt_rng,
                variant_index=variant_index,
                domain=domain,
                family_id=family_id,
                substyle=meta_info["substyle"],
                archetype=reparsed.meta.archetype,
                target_profile=reparsed.meta.target_profile,
                texel_density=reparsed.meta.texel_density,
                box_count=len(scene.boxes),
                bone_count=len(scene.bones),
                has_mirror_x=bool(meta_info["has_mirror_x"]),
                has_detached=bool(meta_info["has_detached_bone"]),
                joint_angle_deg=float(meta_info["joint_angle_deg"]),
                palettes=reparsed.palettes,
                material_families=div_info["material_families"],
                style_encoding=div_info["style_encoding"],
                structural_notes=meta_info["notes"],
            )
            qd_sig = compute_qd_signature(
                reparsed,
                scene,
                domain=domain,
                prompt_style=prompt_style,
            )

            dsl_sha256 = hashlib.sha256(canonical_dsl.encode("utf-8")).hexdigest()
            asset_id = f"{family_id}_{split}_{variant_index:05d}_s{numeric_seed % 100000:05d}"
            profile_version_tag = (
                f"{reparsed.meta.target_profile}:entity_rig_v3.0"
                if is_entity
                else f"{reparsed.meta.target_profile}:java_modern_1.21.11+"
            )

            return {
                "asset_id": asset_id,
                "family_id": family_id,
                "body_plan_id_if_dynamic": body_plan_id,
                "generator_revision": GENERATOR_REVISION,
                "generator_parameters": {
                    "domain": domain,
                    "preset_substyle": str(meta_info["substyle"]),
                    "archetype": str(reparsed.meta.archetype),
                    "target_profile": str(reparsed.meta.target_profile),
                    "texel_density": int(reparsed.meta.texel_density),
                    "variant_index": int(variant_index),
                    "primary_span": int(meta_info["primary_span"]),
                    "secondary_span": int(meta_info["secondary_span"]),
                    "joint_angle_deg": float(meta_info["joint_angle_deg"]),
                    "has_mirror_x": bool(meta_info["has_mirror_x"]),
                    "has_detached_bone": bool(meta_info["has_detached_bone"]),
                    "material_theme": str(meta_info["material_theme"]),
                    "naming_register": str(div_info["naming_register"]),
                    "declarations_permuted": bool(div_info["declarations_permuted"]),
                    "has_materials_block": bool(div_info["has_materials_block"]),
                    "style_encoding": str(div_info["style_encoding"]),
                    "qd_cell": str(qd_sig["qd_cell"]),
                    "inflate_detail": float(meta_info["inflate_detail"]),
                    "structural_spec_json": json.dumps(meta_info["extra"], sort_keys=True),
                },
                "canonical_dsl": canonical_dsl,
                "canonical_dsl_sha256": dsl_sha256,
                "prompt_and_provenance": {
                    "prompt": prompt_text,
                    "prompt_style": prompt_style,
                    "domain": domain,
                    "origin": "synccraft_procedural_ast_grammar_generator",
                    "repository": "https://github.com/BaiTian6641/SyncCraft",
                    "authors": "SyncCraft Contributors (BaiTian6641)",
                },
                "source_license_and_review_record": {
                    "license": "Apache-2.0",
                    "copyright": "Copyright 2026 SyncCraft Contributors",
                    "status": "reviewed_genuine_original",
                    "ip_derivative_risk": "none",
                    "color_space_attribution": "Oklab/OKLCH by Bjorn Ottosson (https://bottosson.github.io/posts/oklab/)",
                },
                "split": split,
                "audit_version_and_results": {
                    "sat_audit_version": "1.1",
                    "contact_policy_version": "p1-02/1.0",
                    "passed": True,
                    "declared_contacts": int(len(audit_res["declared_contact_pairs"])),
                    "contact_violations": int(len(audit_res["contact_violations"])),
                    "collision_pairs": int(len(audit_res["collision_pairs"])),
                    "physical_components": int(len(audit_res["physical_connected_components"])),
                    "boxes": int(len(scene.boxes)),
                    "bones": int(len(scene.bones)),
                    "java_bounds_verified": bool(not is_entity),
                },
                "target_profile_and_version": profile_version_tag,
                "seed": int(numeric_seed),
            }
        except (DSLParseError, DSLValidationError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    raise RuntimeError(
        f"Failed to synthesize verified candidate for {family_id} ({split} #{variant_index}) "
        f"after {max_attempts} attempts; last_error={last_error}"
    )


def _worker_generate_chunk(task: tuple[str, str, int, int, int, bool]) -> list[dict[str, Any]]:
    family_id, split, start_idx, count, base_seed, exporter_on_first = task
    rows: list[dict[str, Any]] = []
    for offset in range(count):
        v_idx = start_idx + offset
        row = verify_and_package_candidate(
            family_id,
            split=split,
            variant_index=v_idx,
            base_seed=base_seed,
            run_exporter_check=(exporter_on_first and offset == 0),
        )
        rows.append(row)
    return rows


def _write_parquet_splits_if_available(datasets_dir: Path, split_paths: dict[str, Path]) -> dict[str, str]:
    """Write compressed Parquet shards alongside JSONL splits when pyarrow is installed."""
    if importlib.util.find_spec("pyarrow") is None:
        return {}
    import pyarrow as pa
    import pyarrow.parquet as pq

    def _batch_to_uniform_table(batch_rows: list[dict[str, Any]]) -> Any:
        tbl = pa.Table.from_pylist(batch_rows)
        bp_idx = tbl.schema.get_field_index("body_plan_id_if_dynamic")
        if bp_idx >= 0 and tbl.schema.field(bp_idx).type != pa.string():
            tbl = tbl.set_column(
                bp_idx,
                "body_plan_id_if_dynamic",
                pa.array([r.get("body_plan_id_if_dynamic") for r in batch_rows], type=pa.string()),
            )
        return tbl

    parquet_dir = datasets_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    batch_size = 4096

    for split_name, jsonl_path in split_paths.items():
        pq_rel = f"parquet/{split_name}.parquet"
        pq_path = datasets_dir / pq_rel
        writer: pq.ParquetWriter | None = None
        batch: list[dict[str, Any]] = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    batch.append(json.loads(line))
                    if len(batch) >= batch_size:
                        table = _batch_to_uniform_table(batch)
                        if writer is None:
                            writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
                        writer.write_table(table)
                        batch.clear()
                if batch:
                    table = _batch_to_uniform_table(batch)
                    if writer is None:
                        writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
                    writer.write_table(table)
                    batch.clear()
        finally:
            if writer is not None:
                writer.close()
        if pq_path.is_file():
            written[split_name] = pq_rel
    return written


def build_verified_dataset(
    datasets_dir: Path | str,
    *,
    preset_name: str = "full_75k",
    base_seed: int = 2026,
    max_workers: int | None = None,
    include_full_rows_threshold: int = 2000,
    write_parquet: bool = True,
) -> dict[str, Any]:
    """Synthesize, SAT+P1-02 audit, SHA-256 deduplicate, and serialize a SyncCraft dataset."""
    if preset_name not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset preset {preset_name!r}; expected one of {sorted(DATASET_PRESETS)}")
    preset = DATASET_PRESETS[preset_name]
    out_dir = Path(datasets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, str, int, int, int, bool]] = []
    chunk_limit = 500

    for fam in SEEN_STATIC_FAMILIES:
        # Train split chunks (variant indices 0 .. train_per_seen_family - 1)
        t_total = preset.train_per_seen_family
        for start in range(0, t_total, chunk_limit):
            cnt = min(chunk_limit, t_total - start)
            tasks.append((fam, "train", start, cnt, base_seed, start == 0))
        # Val split chunks (non-overlapping variant indices starting at train_per_seen_family)
        v_total = preset.val_per_seen_family
        v_base = preset.train_per_seen_family
        for start in range(0, v_total, chunk_limit):
            cnt = min(chunk_limit, v_total - start)
            tasks.append((fam, "val", v_base + start, cnt, base_seed + 17, False))

    for fam in UNSEEN_STATIC_FAMILIES:
        u_total = preset.test_per_unseen_family
        for start in range(0, u_total, chunk_limit):
            cnt = min(chunk_limit, u_total - start)
            tasks.append((fam, "test_unseen", start, cnt, base_seed + 31, start == 0))

    cpu_workers = max_workers if max_workers is not None else min(os.cpu_count() or 1, 12)
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    spawn_safe = not (multiprocessing.get_start_method(allow_none=True) != "fork" and main_file == "<stdin>")
    use_multiprocess = preset.total_rows >= 200 and cpu_workers > 1 and spawn_safe

    split_paths = {
        "train": out_dir / "train.jsonl",
        "val": out_dir / "val.jsonl",
        "test_unseen": out_dir / "test_unseen.jsonl",
    }
    split_handles = {sp: open(path, "w", encoding="utf-8") for sp, path in split_paths.items()}
    split_hashers = {sp: hashlib.sha256() for sp in split_paths}
    seen_dsl_hashes: set[str] = set()

    embed_full_rows = preset.total_rows <= include_full_rows_threshold
    manifest_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    sampled_per_family: dict[str, int] = {}

    split_counts = {"train": 0, "val": 0, "test_unseen": 0}
    family_counts: dict[str, int] = {}
    box_counts: list[int] = []
    bone_counts: list[int] = []
    domains_seen: list[str] = []
    prompt_styles_seen: list[str] = []
    registers_seen: list[str] = []
    qd_cells_seen: list[str] = []
    unique_ast_identifiers: set[str] = set()
    materials_block_count = 0
    collision_resamples = 0

    def _consume_chunk(chunk_rows: list[dict[str, Any]]) -> None:
        nonlocal collision_resamples, materials_block_count
        for row in chunk_rows:
            fam = row["family_id"]
            sp = row["split"]
            v_idx = row["generator_parameters"]["variant_index"]
            salt = 1
            while row["canonical_dsl_sha256"] in seen_dsl_hashes:
                collision_resamples += 1
                row = verify_and_package_candidate(
                    fam,
                    split=sp,
                    variant_index=v_idx + 1_000_000 * salt,
                    base_seed=base_seed + 7919 * salt,
                    run_exporter_check=False,
                )
                salt += 1

            seen_dsl_hashes.add(row["canonical_dsl_sha256"])
            canonical_line = json.dumps(row, sort_keys=True, ensure_ascii=False)
            split_handles[sp].write(canonical_line + "\n")
            split_hashers[sp].update((canonical_line + "\n").encode("utf-8"))

            split_counts[sp] += 1
            family_counts[fam] = family_counts.get(fam, 0) + 1
            box_counts.append(row["audit_version_and_results"]["boxes"])
            bone_counts.append(row["audit_version_and_results"]["bones"])

            gp = row["generator_parameters"]
            domains_seen.append(gp["domain"])
            prompt_styles_seen.append(row["prompt_and_provenance"]["prompt_style"])
            registers_seen.append(gp["naming_register"])
            qd_cells_seen.append(gp["qd_cell"])
            if gp["has_materials_block"]:
                materials_block_count += 1
            unique_ast_identifiers.update(_IDENT_SCAN_RE.findall(row["canonical_dsl"]))

            if embed_full_rows:
                manifest_rows.append(row)
            if sampled_per_family.get(fam, 0) < 2:
                sample_rows.append(row)
                sampled_per_family[fam] = sampled_per_family.get(fam, 0) + 1

    consumed_tasks = 0
    try:
        if use_multiprocess:
            src_root = str(Path(__file__).resolve().parents[2])
            existing_pp = os.environ.get("PYTHONPATH", "")
            pp_entries = [p for p in existing_pp.split(os.pathsep) if p]
            if src_root not in pp_entries:
                os.environ["PYTHONPATH"] = os.pathsep.join([src_root, *pp_entries])
            try:
                with ProcessPoolExecutor(max_workers=cpu_workers) as pool:
                    for chunk_result in pool.map(_worker_generate_chunk, tasks):
                        _consume_chunk(chunk_result)
                        consumed_tasks += 1
            except (BrokenProcessPool, PermissionError, OSError, RuntimeError):
                for task in tasks[consumed_tasks:]:
                    _consume_chunk(_worker_generate_chunk(task))
                    consumed_tasks += 1
        else:
            for task in tasks:
                _consume_chunk(_worker_generate_chunk(task))
                consumed_tasks += 1
    finally:
        for handle in split_handles.values():
            handle.close()

    parquet_files = _write_parquet_splits_if_available(out_dir, split_paths) if write_parquet else {}

    split_sha256 = {sp: hasher.hexdigest() for sp, hasher in split_hashers.items()}
    if embed_full_rows:
        manifest_sha256 = hashlib.sha256(
            json.dumps(manifest_rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
    else:
        manifest_sha256 = hashlib.sha256(
            json.dumps(split_sha256, sort_keys=True).encode("utf-8")
        ).hexdigest()

    total_records = sum(split_counts.values())
    complexity_stats = {
        "min_boxes": int(min(box_counts)) if box_counts else 0,
        "mean_boxes": round(float(sum(box_counts)) / max(1, len(box_counts)), 2),
        "max_boxes": int(max(box_counts)) if box_counts else 0,
        "min_bones": int(min(bone_counts)) if bone_counts else 0,
        "mean_bones": round(float(sum(bone_counts)) / max(1, len(bone_counts)), 2),
        "max_bones": int(max(bone_counts)) if bone_counts else 0,
        "unique_canonical_dsl_hashes": len(seen_dsl_hashes),
        "hash_collision_resamples": collision_resamples,
    }
    diversity_stats = summarize_corpus_diversity(
        domains=domains_seen,
        prompt_styles=prompt_styles_seen,
        naming_registers=registers_seen,
        qd_cells=qd_cells_seen,
        unique_identifiers=unique_ast_identifiers,
        materials_block_count=materials_block_count,
    )

    manifest_payload: dict[str, Any] = {
        "schema": "synccraft.dataset-manifest/1.2",
        "preset": preset.name,
        "hf_size_category": preset.hf_size_category,
        "generator_revision": GENERATOR_REVISION,
        "dataset_manifest_sha256": manifest_sha256,
        "split_sha256": split_sha256,
        "total_records": total_records,
        "split_counts": split_counts,
        "family_counts": family_counts,
        "macro_domains": list(MACRO_DOMAINS),
        "naming_registers": list(NAMING_REGISTERS),
        "complexity_stats": complexity_stats,
        "diversity_stats": diversity_stats,
        "shard_file": "train.jsonl",
        "shard_bytes": split_paths["train"].stat().st_size,
        "parquet_shards": parquet_files,
        "seen_families": sorted(SEEN_STATIC_FAMILIES),
        "unseen_families": sorted(UNSEEN_STATIC_FAMILIES),
        "provenance": "procedural_ast_grammar",
        "license": "Apache-2.0",
        "attribution": SYNCCRAFT_ATTRIBUTION,
    }
    if embed_full_rows:
        manifest_payload["rows"] = manifest_rows
    else:
        manifest_payload["sample_rows"] = sample_rows

    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return manifest_payload
