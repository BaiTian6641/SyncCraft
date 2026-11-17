"""src/synccraft/data_gen/mh_dragons_v8/species/rathalos.py

Monumental Sculpted Blockbench Reconstruction of the RATHALOS Lineage (Biped Flying Wyverns):
  - `rathalos`: King of the Skies (crimson & dark obsidian dorsal armor, hooked eagle-beak raptor skull,
    swept obsidian crown & temporal cheek horns, central jaw chin-spike, titanic 3-bank flame-veined
    `wing_rathalos` sails, heavy venom-talon legs, and signature multi-spike tail club).
  - `azure_rathalos`: Azure King of the Skies (cobalt-sapphire carapace, emerald-speckled wings, sharper
    elongated temporal crest prongs & sleek aerial build).
  - `silver_rathalos`: Incandescent Silver Wyvern (gleaming silver-platinum overlapped shell plating, extra
    layered hellfire brow & pectoral shield plates, indigo-charcoal wings, crisp cobalt-flame ember fissures,
    and heavy double-crowned mace tail).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def rathalos_materials(variant: str = "canonical") -> dict[str, dict[str, Any]]:
    if variant == "canonical":
        return {
            "scales": {"color": [158, 42, 36], "kind": "scales", "coat": "calm_flank"},
            "scales_deep": {"color": [108, 30, 28], "kind": "scales", "coat": "calm_flank"},
            "armor": {"color": [48, 36, 40], "kind": "armor", "coat": "armor_shingle"},
            "shell_hi": {"color": [178, 56, 42], "kind": "armor", "coat": "armor_shingle"},
            "belly": {"color": [212, 186, 138], "kind": "belly", "coat": "belly_scute"},
            "horn": {"color": [42, 34, 38], "kind": "horn", "coat": "horn_keratin"},
            "wing_mem": {"color": [172, 118, 74], "kind": "membrane", "coat": "wing_rathalos"},
            "wing_back": {"color": [126, 78, 48], "kind": "membrane", "coat": "wing_rathalos"},
            "claw": {"color": [54, 46, 44], "kind": "horn", "coat": "horn_keratin"},
            "mouth": {"color": [138, 44, 52], "kind": "mouth"},
            "tooth": {"color": [232, 224, 200], "kind": "bone"},
            "eye": {"color": [250, 178, 42], "kind": "eye", "emissive": 0.86},
        }
    if variant == "azure":
        return {
            "scales": {"color": [48, 108, 162], "kind": "scales", "coat": "calm_flank"},
            "scales_deep": {"color": [34, 72, 116], "kind": "scales", "coat": "calm_flank"},
            "armor": {"color": [32, 48, 72], "kind": "armor", "coat": "armor_shingle"},
            "shell_hi": {"color": [68, 138, 188], "kind": "armor", "coat": "armor_shingle"},
            "belly": {"color": [206, 198, 168], "kind": "belly", "coat": "belly_scute"},
            "horn": {"color": [36, 48, 66], "kind": "horn", "coat": "horn_keratin"},
            "wing_mem": {"color": [92, 152, 98], "kind": "membrane", "coat": "wing_rathalos"},
            "wing_back": {"color": [64, 112, 74], "kind": "membrane", "coat": "wing_rathalos"},
            "claw": {"color": [46, 48, 56], "kind": "horn", "coat": "horn_keratin"},
            "mouth": {"color": [134, 46, 62], "kind": "mouth"},
            "tooth": {"color": [232, 226, 206], "kind": "bone"},
            "eye": {"color": [255, 196, 54], "kind": "eye", "emissive": 0.86},
        }
    return {  # silver
        "scales": {"color": [174, 180, 190], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [126, 132, 146], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [92, 96, 114], "kind": "armor", "coat": "armor_shingle"},
        "shell_hi": {"color": [208, 212, 222], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [212, 214, 222], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [64, 68, 84], "kind": "horn", "coat": "horn_keratin"},
        "wing_mem": {"color": [74, 78, 98], "kind": "membrane", "coat": "wing_rathalos"},
        "wing_back": {"color": [50, 54, 72], "kind": "membrane", "coat": "wing_rathalos"},
        "claw": {"color": [54, 56, 68], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [118, 48, 92], "kind": "mouth"},
        "tooth": {"color": [238, 236, 232], "kind": "bone"},
        "eye": {"color": [120, 198, 255], "kind": "eye", "emissive": 0.88},
    }


def _build_rathalos_specimen(variant: str = "canonical", *, stage_cap: int = 3) -> dict[str, Any]:
    if variant == "canonical":
        mid, title, ep, vkind, par = "rathalos", "Rathalos", "King of the Skies", "canonical_flagship", None
        wiki = "https://monsterhunter.fandom.com/wiki/Rathalos"
    elif variant == "azure":
        mid, title, ep, vkind, par = "azure_rathalos", "Azure Rathalos", "Azure King of the Skies", "subspecies", "rathalos"
        wiki = "https://monsterhunter.fandom.com/wiki/Azure_Rathalos"
    else:
        mid, title, ep, vkind, par = "silver_rathalos", "Silver Rathalos", "Incandescent Silver Wyvern", "rare_species", "rathalos"
        wiki = "https://monsterhunter.fandom.com/wiki/Silver_Rathalos"

    is_azure = variant == "azure"
    is_silver = variant == "silver"
    hell_seam = [110, 178, 255] if is_silver else None

    m = RigCanvas(
        mid,
        title,
        ep,
        "Flying Wyvern",
        variant_kind=vkind,
        parent_species_id=par,
        wiki_url=wiki,
        materials=rathalos_materials(variant),
        notes=[f"Monumental biped flying wyvern sculpted from `{mid}_wide_board.jpg`."],
        texels_per_unit=2,
    )

    # Classic Biped Flying Wyvern Stance (`hinged pelvis -> broad arched torso -> S-neck -> raptor skull`)
    m.bone("pelvis", "root", (0.0, 22.0, -8.0))
    m.bone("chest", "pelvis", (0.0, 26.8, 7.0))
    m.bone("neck_1", "chest", (0.0, 31.8, 17.5))
    m.bone("neck_2", "neck_1", (0.0, 37.5, 24.5))
    m.bone("head", "neck_2", (0.0, 41.2, 31.0))
    m.bone("jaw", "head", (0.0, 38.2, 31.8))

    # =========================================================================
    # STAGE 1: BIPAL FLYING WYVERN MASS & TITANIC WING SPAN BLOCKOUT
    # =========================================================================
    torso_stations = [
        ((0.0, 21.0, -14.5), 11.8, 10.8, "pelvis"),
        ((0.0, 23.0, -4.5), 14.5, 13.0, "pelvis"),
        ((0.0, 26.5, 6.0), 17.8, 15.8, "chest"),
        ((0.0, 29.8, 14.5), 16.0, 14.8, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.8, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 30.8, 16.2), (0.0, 36.2, 23.0), 11.2, 10.6),
        ("neck_2", (0.0, 36.2, 23.0), (0.0, 41.0, 30.2), 9.6, 9.2),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    # Predatory Eagle-Beak Skull + Compact Deep Jaw (`short-to-medium hooked predator snout!`)
    m.box("skull_cranium", "head", (0.0, 42.0, 34.2), (9.2, 6.4, 8.2), "shell_hi", face_materials={"up": "armor"}, rotation=(8.0, 0.0, 0.0), component="head", stage1=True)
    m.box("raptor_beak_snout", "head", (0.0, 40.8, 40.4), (6.0, 4.2, 6.2), "scales", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 38.2, 37.8), (5.8, 2.5, 9.8), "belly", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    # Long Articulated 7-Bone Armored Tail + Flared Club Mace
    t_pts = [
        ((0.0, 21.2, -13.5), 10.5, 9.5),
        ((0.8, 20.5, -22.5), 8.8, 8.2),
        ((2.2, 20.4, -31.5), 7.5, 7.2),
        ((3.5, 21.0, -40.5), 6.2, 6.2),
        ((3.8, 22.4, -49.5), 5.2, 5.4),
        ((2.8, 24.4, -58.2), 4.2, 4.6),
        ((1.2, 26.4, -66.5), 5.6, 5.4),  # Heavy mace flare at tip!
        ((0.2, 27.6, -73.0), 4.2, 4.2),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(7)]
    for i in range(7):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.6, pattern="calm_flank", stage1=(i % 2 == 0))

    # Powerful Biped Talon Legs (`hind_upper`, `hind_lower`, `hind_paw`)
    for s, q in ((1.0, "l"), (-1.0, "r")):
        p0 = (s * 8.2, 21.0, -6.5)
        p1 = (s * 12.2, 12.0, -1.0)
        p2 = (s * 12.0, 3.8, -8.5)
        bu = m.bone(f"hind_upper_{q}", "pelvis", p0)
        bl = m.bone(f"hind_lower_{q}", bu, p1)
        bp = m.bone(f"hind_paw_{q}", bl, p2)
        m.span_box(f"hind_thigh_{q}", bu, p0, p1, 8.8, 8.4, "scales", len_pad=2.4, pattern="calm_flank", stage1=True)
        m.span_box(f"hind_shin_{q}", bl, p1, p2, 6.2, 6.0, "scales", len_pad=2.4, pattern="leg_bands", stage1=True)
        m.box(f"hind_ankle_{q}", bp, (p2[0], 2.8, p2[2] + 2.2), (7.6, 3.6, 7.6), "scales_deep", stage1=True)

    # Titanic 3-Bank Bat/Raptor Wing Sails (`14+` total membrane cards!)
    wing_span = 1.06 if is_azure else 1.0
    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.winged_sail_rig(
            s,
            "chest",
            (s * 8.0, 29.8, 8.5),
            (s * 22.0, 42.5, 4.5),
            (s * 34.0, 52.0, 14.5),
            (
                (s * 66.0 * wing_span, 56.0, 2.5),
                (s * 59.0 * wing_span, 41.0, -10.0),
                (s * 45.0 * wing_span, 29.0, -19.0),
                (s * 28.0 * wing_span, 20.0, -23.0),
            ),
            (s * 7.5, 22.0, -8.0),
            spar_mat="shell_hi",
            hub_mat="armor",
            claw_mat="claw",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_rathalos",
            spar_w=3.4,
            arm_w=5.0,
            leading_scales=6,
            tattered_outer=True,
        )

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: RAPTOR SKULL, HOOKED BEAK, CHIN BARB, TALONS & TAIL CLUB
    # =========================================================================
    m.box("nasal_bridge_ridge", "head", (0.0, 43.2, 39.2), (4.6, 2.2, 6.8), "armor", seam_color=hell_seam, rotation=(14.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.span_box("eagle_hook_beak", "head", (0.0, 41.0, 42.8), (0.0, 38.6, 45.6), 3.6, 2.6, "horn", pattern="horn_keratin", component="snout")
    m.span_box("chin_raptor_barb", "jaw", (0.0, 37.6, 39.2), (0.0, 33.4, 36.2), 2.2, 2.6, "horn", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_armor_{q}", "head", (s * 4.8, 41.2, 34.5), (2.4, 3.8, 6.2), "armor" if is_silver else "shell_hi", seam_color=hell_seam, rotation=(8.0, s * 12.0, -s * 10.0), pattern="armor_shingle", component="head")
        m.box(f"brow_horn_ridge_{q}", "head", (s * 4.2, 44.2, 36.5), (2.6, 2.0, 5.5), "armor", seam_color=hell_seam, rotation=(12.0, s * 12.0, -s * 14.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.6, 42.6, 36.8), (1.1, 1.5, 2.6), "mouth", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_raptor_slit_{q}", "head", (s * 4.9, 42.6, 37.0), (0.7, 1.1, 2.0), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        # Main swept obsidian/silver crown horns + flared Azure/Silver ear cheek-fins!
        c_reach = 1.24 if (is_azure or is_silver) else 1.0
        m.tapered_arch(f"crown_horn_{q}", "head", [(s * 3.0, 44.8, 33.2), (s * 4.8 * c_reach, 48.2, 28.0), (s * 5.8 * c_reach, 51.2, 22.8)], [(2.8, 2.6), (2.0, 1.9), (1.0, 1.0)], "horn", component="horns", stage1=True)
        m.tapered_arch(f"temporal_ear_horn_{q}", "head", [(s * 4.8, 42.4, 32.8), (s * 8.2 * c_reach, 44.5, 27.2), (s * 10.4 * c_reach, 46.2, 22.0)], [(2.2, 2.6), (1.6, 1.9), (0.9, 1.1)], "armor", seam_color=hell_seam, component="horns")

        for ti, tz in enumerate((35.5, 37.8, 40.0, 42.0)):
            ux = 2.8 - 0.30 * ti
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * ux, 39.2, tz), (0.8, 1.4 if ti == 2 else 1.0, 0.8), "tooth", component="teeth")
            if ti in (1, 3):
                m.box(f"lower_tooth_{q}_{ti}", "jaw", (s * (ux - 0.3), 39.4, tz - 0.4), (0.7, 1.1, 0.8), "tooth", component="teeth")

        # Forehead / thigh heavy armor + 3 forward talons + sharp rear dewclaw!
        bl, bp = f"hind_lower_{q}", f"hind_paw_{q}"
        m.box(f"thigh_armor_plate_{q}", f"hind_upper_{q}", (s * 9.5, 22.5, -6.2), (5.8, 6.2, 7.6), "shell_hi", seam_color=hell_seam, rotation=(10.0, s * 10.0, -s * 14.0), pattern="armor_shingle")
        m.box(f"knee_spur_shield_{q}", bl, (s * 12.6, 13.0, 0.2), (4.8, 4.5, 5.2), "armor", rotation=(22.0, s * 12.0, 0.0), pattern="armor_shingle")
        for bi in range(4):
            pt_b = lerp3((s * 12.2, 12.0, -1.0), (s * 12.0, 3.8, -8.5), 0.16 + 0.22 * bi) + np.array([s * 2.0, 0.8, 1.5])
            m.box(f"shin_scale_{q}_{bi}", bl, pt_b, (2.4, 3.2, 4.0), "armor" if bi % 2 == 0 else "shell_hi", rotation=(16.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")

        for di, ang_d in ((0, -22.0), (1, 0.0), (2, 22.0)):
            yaw = math.radians(s * ang_d)
            dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            ta = np.array([s * (12.0 + (-2.0 + 2.0 * di) * s), 2.4, -3.5])
            tb = ta + dv * 4.4
            tc = tb + dv * 3.5 - np.array([0.0, 1.0, 0.0])
            m.span_box(f"talon_toe_{q}_{di}", bp, ta, tb, 2.2, 2.2, "scales_deep", len_pad=0.5, component="claws")
            m.span_box(f"talon_claw_{q}_{di}", bp, tb, tc, 1.6, 1.8, "claw", len_pad=0.4, component="claws")
        # Rear venom dewclaw
        m.span_box(f"rear_dewclaw_{q}", bp, (s * 12.0, 2.5, -10.0), (s * 12.0, 1.2, -14.8), 1.8, 1.8, "claw", component="claws")

    # Stepped Cream Throat & Breast Sternum Scutes
    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 35.8, 25.0), 7.4, 5.6, 22.0),
        ("neck_1", (0.0, 31.2, 20.5), 9.0, 6.2, 18.0),
        ("chest", (0.0, 26.8, 16.8), 10.6, 6.8, 22.0),
        ("chest", (0.0, 22.5, 13.5), 11.8, 7.2, 12.0),
        ("chest", (0.0, 19.5, 6.2), 13.2, 7.5, 4.0),
        ("chest", (0.0, 18.8, -1.5), 12.2, 7.2, 0.0),
        ("pelvis", (0.0, 17.8, -9.0), 11.2, 6.8, -2.0),
    )):
        m.box(f"belly_scute_{ki}", kb, kpos, (kw, 2.4, kd), "belly", seam_color=hell_seam if (is_silver and ki in (2, 3)) else None, rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    # Signature Heavy Spiked Rathalos Poison Tail Club (`tail_6..7`)
    m.box("tail_mace_housing", tbones[5], (2.0, 25.4, -62.5), (8.2, 6.4, 8.6), "armor", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
    m.box("tail_mace_core", tbones[6], (0.6, 27.0, -70.0), (10.5, 7.2, 9.2), "armor", seam_color=hell_seam, rotation=(10.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for si, (dx, dy, dz, len_sp) in enumerate((
            (5.5, 1.0, -64.0, 5.2),
            (6.2, 1.5, -69.5, 6.5),
            (4.8, -1.5, -72.0, 4.8),
            (2.8, 3.2, -73.0, 5.0),
        )):
            base_s = np.array([0.6 + s * dx * 0.6, 27.0 + dy, dz])
            tip_s = base_s + np.array([s * len_sp * 0.85, dy * 0.5, -len_sp * 0.55])
            m.span_box(f"mace_spike_{q}_{si}", tbones[6], base_s, tip_s, 1.8, 2.0, "claw", component="spines")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: DORSAL CARAPACE ARMOR, WING SHOULDER PLATES & TAIL RINGING
    # =========================================================================
    # Neck & Torso Dorsal Shell Spines + Layered Flank Shingles
    for ni, (nb, npos, nw, nh) in enumerate((
        ("neck_2", (0.0, 41.2, 26.5), 5.8, 3.6),
        ("neck_1", (0.0, 36.2, 20.5), 6.8, 4.2),
        ("chest", (0.0, 34.8, 12.5), 8.2, 4.6),
        ("chest", (0.0, 33.8, 4.5), 8.8, 4.8),
        ("pelvis", (0.0, 30.2, -4.5), 7.6, 4.4),
        ("pelvis", (0.0, 27.2, -12.5), 6.4, 3.8),
    )):
        m.box(f"dorsal_shell_plate_{ni}", nb, npos, (nw, 2.4, 6.2), "armor", seam_color=hell_seam if (is_silver and ni % 2 == 0) else None, rotation=(20.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        m.span_box(f"dorsal_spine_{ni}", nb, npos, npos + np.array([0.0, nh, -3.8]), 2.0, 2.6, "horn", component="spines")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        # Neck lateral overlapping crimson armor pleats (`eliminates flat green/red cylinder neck!`)
        for nb, p_a, p_b in (
            ("neck_1", (s * 4.8, 32.0, 17.0), (s * 4.2, 36.2, 22.8)),
            ("neck_2", (s * 4.0, 37.0, 24.0), (s * 3.6, 40.5, 29.8)),
        ):
            m.shingle_arc(f"neck_side_scales_{nb}_{q}", nb, p_a, p_b, (s * 0.35, 0.25, -0.88), (s * 0.28, 0.28, -0.90), count=5, length=5.2, width=2.8, thickness=1.6, material="shell_hi", alt_material="armor", two_step=False, pattern="armor_shingle")
        # Scapular shoulder pauldron + wing root overlap platelets
        m.box(f"wing_scapula_shield_{q}", "chest", (s * 8.8, 30.8, 9.5), (5.5, 3.8, 8.2), "armor", seam_color=hell_seam, rotation=(14.0, s * 16.0, -s * 22.0), pattern="armor_shingle", component="armor")
        for wi in range(4):
            w_pt = lerp3((s * 8.0, 29.8, 8.5), (s * 22.0, 42.5, 4.5), 0.15 + 0.24 * wi)
            m.box(f"wing_humerus_scale_{q}_{wi}", f"wing_{q}", w_pt + np.array([0.0, 2.2, 1.2]), (3.6, 2.4, 4.8), "armor" if wi % 2 == 0 else "shell_hi", rotation=(12.0, s * 18.0, 0.0), pattern="armor_shingle", component="armor")
        # Thigh muscle armor ridges & chest/belly rib pleats
        for thi in range(4):
            th_p = lerp3((s * 8.2, 21.0, -6.5), (s * 12.2, 12.0, -1.0), 0.18 + 0.22 * thi) + np.array([s * 2.8, 0.8, 0.6])
            m.box(f"thigh_muscle_plate_{q}_{thi}", f"hind_upper_{q}", th_p, (3.2, 4.0, 4.8), "shell_hi" if thi % 2 == 0 else "armor", rotation=(14.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
        m.shingle_arc(f"chest_flank_armor_{q}", "chest", (s * 9.5, 27.5, 12.5), (s * 8.2, 24.5, -2.5), (s * 0.25, 0.14, -0.96), (s * 0.20, 0.12, -0.97), count=6, length=6.8, width=3.4, thickness=2.2, material="shell_hi", alt_material="armor", seam_color=hell_seam, two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"chest_ventral_ribs_{q}", "chest", (s * 6.8, 21.0, 11.5), (s * 6.4, 19.5, -1.5), (s * 0.45, -0.25, -0.86), (s * 0.40, -0.22, -0.88), count=5, length=5.2, width=2.8, thickness=1.6, material="belly", alt_material="scales", two_step=False, pattern="belly_scute")
        m.shingle_arc(f"pelvis_flank_armor_{q}", "pelvis", (s * 7.6, 22.8, -4.5), (s * 5.8, 21.2, -15.5), (s * 0.22, 0.12, -0.97), (s * 0.18, 0.10, -0.98), count=5, length=6.0, width=2.8, thickness=1.8, material="armor", alt_material="shell_hi", two_step=False, pattern="armor_shingle")
        if is_silver or is_azure:
            # Extra temporal & collar spurs unique to Azure / Silver Rathalos
            m.span_box(f"variant_collar_spur_{q}", "chest", (s * 7.8, 32.2, 13.0), (s * 12.2, 36.5, 5.5), 2.2, 2.8, "armor", component="spines")

    for ti in range(6):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.box(f"tail_dorsal_ring_{ti}", tbones[ti], mid_t + np.array([0.0, 3.2 - 0.15 * ti, 0.0]), (max(2.6, 5.8 - 0.45 * ti), 2.0, 6.8), "armor" if ti % 2 == 0 else "shell_hi", rotation=(12.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.span_box(f"tail_flank_barb_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.5, 3.8 - 0.3 * ti), 0.2, 0.0]), mid_t + np.array([s * max(2.8, 6.2 - 0.35 * ti), 1.8, -3.5]), 1.6, 2.0, "horn", component="spines")

    return m.doc


def build_rathalos(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_rathalos_specimen("canonical", stage_cap=stage_cap)


def build_azure_rathalos(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_rathalos_specimen("azure", stage_cap=stage_cap)


def build_silver_rathalos(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_rathalos_specimen("silver", stage_cap=stage_cap)
