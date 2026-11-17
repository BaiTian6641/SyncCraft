"""src/synccraft/data_gen/mh_dragons_v8/species/fatalis.py

Monumental Sculpted Blockbench Reconstruction of FATALIS & WHITE FATALIS (Forbidden Black/White Elder Dragons):
  - `fatalis`: Black Dragon (unmistakably long 4-bone serpentine S-neck, classic obsidian western-dragon
    horned skull with twin towering swept horns + brow horns, fiery-fissured sternum plates, 3-bank
    obsidian bat wings, and extra-long 8-bone serpentine tail).
  - `white_fatalis`: White God of Destruction (pearl-platinum hide, billowing cascading white mane/beard
    fur plumes down `neck_1..4`, multi-tine branching antler crown horns, glowing crimson chest lightning
    scars, midnight-violet dark inner wings, and fierce crimson eyes).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def fatalis_materials(variant: str = "canonical") -> dict[str, dict[str, Any]]:
    if variant == "canonical":
        return {
            "scales": {"color": [34, 34, 42], "kind": "scales", "coat": "calm_flank"},
            "scales_deep": {"color": [22, 22, 28], "kind": "scales", "coat": "calm_flank"},
            "armor": {"color": [48, 46, 56], "kind": "armor", "coat": "armor_shingle"},
            "belly": {"color": [126, 98, 78], "kind": "belly", "coat": "belly_scute"},
            "horn": {"color": [38, 36, 46], "kind": "horn", "coat": "horn_keratin"},
            "wing_mem": {"color": [52, 42, 46], "kind": "membrane", "coat": "wing_gore"},
            "wing_back": {"color": [34, 28, 32], "kind": "membrane", "coat": "wing_gore"},
            "claw": {"color": [46, 42, 48], "kind": "horn", "coat": "horn_keratin"},
            "mouth": {"color": [134, 42, 48], "kind": "mouth"},
            "tooth": {"color": [230, 224, 206], "kind": "bone"},
            "eye": {"color": [255, 86, 42], "kind": "eye", "emissive": 0.90},
        }
    return {  # white_fatalis
        "scales": {"color": [214, 212, 218], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [168, 166, 178], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [188, 186, 198], "kind": "armor", "coat": "armor_shingle"},
        "fur": {"color": [234, 234, 240], "kind": "fur", "coat": "fur_mane"},
        "belly": {"color": [196, 186, 188], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [86, 84, 96], "kind": "horn", "coat": "horn_keratin"},
        "wing_mem": {"color": [74, 70, 88], "kind": "membrane", "coat": "wing_gore"},
        "wing_back": {"color": [46, 42, 58], "kind": "membrane", "coat": "wing_gore"},
        "claw": {"color": [54, 52, 64], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [146, 46, 62], "kind": "mouth"},
        "tooth": {"color": [238, 236, 230], "kind": "bone"},
        "eye": {"color": [255, 48, 56], "kind": "eye", "emissive": 0.92},
    }


def _build_fatalis_specimen(variant: str = "canonical", *, stage_cap: int = 3) -> dict[str, Any]:
    is_white = variant == "white"
    mid = "white_fatalis" if is_white else "fatalis"
    title = "White Fatalis" if is_white else "Fatalis"
    ep = "Old Ancestor of Light & Ruin" if is_white else "Legend of Schrade"
    vkind = "rare_species" if is_white else "canonical_flagship"
    par = "fatalis" if is_white else None
    wiki = f"https://monsterhunter.fandom.com/wiki/{'White_Fatalis' if is_white else 'Fatalis'}"
    chest_seam = [248, 62, 68] if is_white else [244, 108, 42]

    m = RigCanvas(
        mid,
        title,
        ep,
        "Elder Dragon",
        variant_kind=vkind,
        parent_species_id=par,
        wiki_url=wiki,
        materials=fatalis_materials(variant),
        notes=[f"Classic long-necked Forbidden Dragon built from `{mid}_wide_board.jpg` with 4-segment S-neck & 8-bone serpentine whip tail."],
        texels_per_unit=2,
    )

    # Extra-Long Serpentine 4-Segment S-Neck (`signature Fatalis silhouette!`)
    m.bone("pelvis", "root", (0.0, 21.0, -14.0))
    m.bone("chest", "pelvis", (0.0, 25.5, 5.0))
    m.bone("neck_1", "chest", (0.0, 29.5, 16.0))
    m.bone("neck_2", "neck_1", (0.0, 35.8, 23.0))
    m.bone("neck_3", "neck_2", (0.0, 42.5, 29.0))
    m.bone("neck_4", "neck_3", (0.0, 48.2, 34.5))
    m.bone("head", "neck_4", (0.0, 51.0, 40.5))
    m.bone("jaw", "head", (0.0, 48.0, 41.2))

    # =========================================================================
    # STAGE 1: LONG-NECKED ELDER MASS & WIDE DRAGON WINGS
    # =========================================================================
    torso_stations = [
        ((0.0, 20.5, -20.0), 12.2, 11.2, "pelvis"),
        ((0.0, 22.5, -9.5), 15.2, 13.4, "pelvis"),
        ((0.0, 25.5, 1.5), 18.2, 15.8, "chest"),
        ((0.0, 28.0, 11.5), 17.0, 15.2, "chest"),
        ((0.0, 29.5, 18.0), 13.5, 12.8, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.8, pattern="calm_flank", stage1=True)

    neck_stations = [
        ((0.0, 29.2, 15.8), 11.8, 11.2, "neck_1"),
        ((0.0, 35.2, 22.4), 10.6, 10.0, "neck_2"),
        ((0.0, 41.8, 28.4), 9.6, 9.0, "neck_3"),
        ((0.0, 47.6, 34.0), 8.6, 8.2, "neck_4"),
        ((0.0, 50.8, 40.2), 8.2, 7.8, "head"),
    ]
    for i in range(len(neck_stations) - 1):
        p0, w0, h0, bn = neck_stations[i]
        p1, w1, h1, _ = neck_stations[i + 1]
        m.span_box(f"neck_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    # Classic Western High-Dragon Cranium & Square Snout
    m.box("skull_cranium", "head", (0.0, 51.8, 43.5), (9.4, 6.4, 8.5), "armor", rotation=(6.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("dragon_muzzle", "head", (0.0, 50.5, 50.0), (6.4, 4.2, 6.4), "scales", rotation=(10.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 47.8, 47.6), (6.0, 2.5, 11.0), "scales_deep", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    # Extra-Long 8-Bone Serpentine Dragon Tail (`~54u` trailing whip!)
    t_pts = [
        ((0.0, 20.5, -19.5), 11.0, 10.0),
        ((1.2, 19.4, -28.5), 9.5, 8.8),
        ((3.0, 18.6, -37.5), 8.2, 7.6),
        ((4.8, 18.5, -46.5), 7.0, 6.5),
        ((5.5, 19.2, -55.5), 5.8, 5.5),
        ((4.4, 20.5, -64.2), 4.8, 4.6),
        ((2.2, 22.0, -72.5), 3.8, 3.8),
        ((-0.2, 23.4, -80.2), 3.0, 3.0),
        ((-2.4, 24.4, -87.0), 2.2, 2.2),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(8)]
    for i in range(8):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.6, pattern="calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (9.5, 22.8, 9.5), (13.2, 12.5, 6.0), (12.8, 3.5, 11.5), (6.8, 6.4), (5.2, 5.0)),
        ("hind", "pelvis", (8.8, 20.2, -13.5), (12.4, 11.2, -7.0), (12.0, 3.2, -14.8), (8.2, 7.8), (5.8, 5.5)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0, p1, p2 = (s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.2, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.2, pattern="leg_bands", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.5, p2[2] + 2.5), (7.0, 3.5, 7.2), "scales_deep", stage1=True)

    # Classic Broad 3-Bank Bat Dragon Wings
    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.winged_sail_rig(
            s,
            "chest",
            (s * 8.2, 30.2, 8.5),
            (s * 22.0, 42.5, 3.5),
            (s * 34.0, 52.0, 14.0),
            ((s * 68.0, 56.0, 2.0), (s * 60.0, 41.0, -10.5), (s * 46.0, 28.0, -19.5), (s * 29.0, 19.0, -23.0)),
            (s * 7.8, 22.0, -8.5),
            spar_mat="armor",
            hub_mat="scales_deep",
            claw_mat="claw",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_gore",
            spar_w=3.4,
            arm_w=5.0,
            leading_scales=6,
            tattered_outer=True,
        )

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: HIGH-DRAGON HEAD HORNS/ANTLERS, WHISKERS, FANGS & STERNUM
    # =========================================================================
    m.box("nasal_bridge", "head", (0.0, 53.0, 48.2), (4.8, 2.2, 6.8), "armor", rotation=(12.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.span_box("snout_horn_nub", "head", (0.0, 52.2, 51.4), (0.0, 54.4, 52.6), 1.8, 1.8, "horn", component="snout")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_armor_{q}", "head", (s * 4.8, 51.0, 44.0), (2.4, 3.8, 6.5), "armor", rotation=(8.0, s * 12.0, -s * 10.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.6, 52.6, 46.2), (1.1, 1.5, 2.6), "mouth", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_fatal_{q}", "head", (s * 4.9, 52.6, 46.4), (0.8, 1.1, 2.0), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        if is_white:
            # White Fatalis BRANCHING ANTLER HORNS + flowing cheek beard fur (`white_fatalis_wide_board.jpg`!)
            m.tapered_arch(f"antler_beam_{q}", "head", [(s * 3.0, 54.6, 42.8), (s * 5.2, 60.2, 38.2), (s * 6.8, 66.2, 34.0), (s * 7.5, 71.5, 31.0)], [(2.6, 2.6), (2.2, 2.2), (1.6, 1.6), (0.9, 0.9)], "horn", component="horns", stage1=True)
            m.span_box(f"antler_tine_fwd_{q}", "head", (s * 5.2, 60.2, 38.2), (s * 6.6, 65.2, 41.8), 1.6, 1.6, "horn", component="horns")
            m.span_box(f"antler_tine_mid_{q}", "head", (s * 6.8, 66.2, 34.0), (s * 9.8, 68.8, 30.2), 1.4, 1.4, "horn", component="horns")
            m.tapered_arch(f"cheek_beard_plume_{q}", "head", [(s * 4.8, 49.5, 42.2), (s * 8.2, 47.8, 37.5), (s * 10.8, 44.8, 32.5)], [(2.4, 3.0), (2.0, 2.4), (1.2, 1.6)], "fur", pattern="fur_mane", component="surface")
        else:
            # Classic Black Fatalis Heavy Curved Twin Horns + Lower Brow Pair!
            m.tapered_arch(f"fatalis_crown_horn_{q}", "head", [(s * 3.2, 54.6, 42.8), (s * 5.2, 59.5, 36.8), (s * 6.2, 64.5, 30.8), (s * 6.5, 68.5, 25.2)], [(3.0, 2.8), (2.4, 2.2), (1.8, 1.7), (1.0, 1.0)], "horn", component="horns", stage1=True)
            m.tapered_arch(f"fatalis_brow_horn_{q}", "head", [(s * 4.8, 52.8, 42.2), (s * 8.0, 54.5, 36.8), (s * 10.2, 56.2, 31.8)], [(2.2, 2.2), (1.6, 1.6), (0.9, 0.9)], "horn", component="horns")

        for ti, tz in enumerate((45.0, 47.4, 49.6, 51.6)):
            ux = 3.0 - 0.25 * ti
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * ux, 48.8, tz), (0.8, 1.4 if ti == 2 else 1.0, 0.8), "tooth", component="teeth")
            if ti % 2 == 1:
                m.box(f"lower_tooth_{q}_{ti}", "jaw", (s * (ux - 0.3), 49.0, tz - 0.4), (0.7, 1.1, 0.8), "tooth", component="teeth")

        for kind, p0_r, p1_r, p2_r in (
            ("fore", (9.5, 22.8, 9.5), (13.2, 12.5, 6.0), (12.8, 3.5, 11.5)),
            ("hind", (8.8, 20.2, -13.5), (12.4, 11.2, -7.0), (12.0, 3.2, -14.8)),
        ):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0, p1, p2 = np.array([s * p0_r[0], p0_r[1], p0_r[2]]), np.array([s * p1_r[0], p1_r[1], p1_r[2]]), np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_shoulder_plate_{q}", bu, p0 + np.array([s * 1.0, 1.6, 0.5]), (6.6, 6.6, 7.6), "fur" if (is_white and kind == "fore") else "armor", rotation=(8.0, s * 10.0, -s * 14.0), pattern="fur_mane" if (is_white and kind == "fore") else "armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.16 + 0.22 * bi) + np.array([s * 1.8, 0.8, 1.2])
                m.box(f"{kind}_scale_{q}_{bi}", bl, pt_b, (2.2, 3.4, 4.2), "armor", rotation=(18.0, s * 14.0, -s * 20.0), pattern="armor_shingle", component="armor")
            for di, ang_d in ((0, -18.0), (1, 0.0), (2, 18.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-1.8 + 1.8 * di), 2.2, p2[2] + 4.8])
                tb = ta + dv * 3.6
                tc = tb + dv * 2.8 - np.array([0.0, 0.8, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 1.8, 1.8, "scales_deep", len_pad=0.4, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.4, 1.5, "claw", len_pad=0.3, component="claws")

    # Proud Stepped Sternum & Throat Belly Plates (flame/crimson chest fissures ONLY on middle sternum!)
    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_4", (0.0, 45.4, 35.8), 6.4, 5.2, 32.0),
        ("neck_3", (0.0, 40.0, 30.2), 7.2, 5.6, 36.0),
        ("neck_2", (0.0, 33.8, 24.4), 8.2, 6.0, 34.0),
        ("neck_1", (0.0, 27.6, 18.2), 9.4, 6.4, 26.0),
        ("chest", (0.0, 20.6, 12.2), 11.8, 7.2, 14.0),
        ("chest", (0.0, 18.2, 4.2), 12.6, 7.4, 4.0),
        ("pelvis", (0.0, 17.2, -4.5), 11.6, 7.0, 0.0),
        ("pelvis", (0.0, 16.5, -13.5), 10.2, 6.6, -2.0),
    )):
        m.box(f"belly_scute_{ki}", kb, kpos, (kw, 2.4, kd), "belly", seam_color=chest_seam if ki in (4, 5) else None, rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: SERPENTINE NECK SHINGLES, WHITE FATALIS FLOWING MANE & DORSAL SPINES
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for nb, p_a, p_b in (
            ("neck_1", (s * 4.8, 29.8, 16.2), (s * 4.4, 34.6, 21.8)),
            ("neck_2", (s * 4.4, 35.8, 22.8), (s * 4.0, 41.2, 27.8)),
            ("neck_3", (s * 4.0, 42.4, 28.8), (s * 3.6, 47.0, 33.4)),
            ("neck_4", (s * 3.6, 48.0, 34.4), (s * 3.4, 50.2, 39.2)),
        ):
            m.shingle_arc(f"neck_side_{nb}_{q}", nb, p_a, p_b, (s * 0.35, 0.22, -0.88), (s * 0.28, 0.25, -0.90), count=4, length=5.2, width=2.8, thickness=1.6, material="fur" if is_white else "armor", alt_material="armor", two_step=False, pattern="fur_mane" if is_white else "armor_shingle")
        m.shingle_arc(f"chest_flank_shingles_{q}", "chest", (s * 9.8, 26.8, 11.5), (s * 8.4, 23.8, -2.5), (s * 0.25, 0.14, -0.96), (s * 0.20, 0.12, -0.97), count=6, length=6.8, width=3.4, thickness=2.1, material="armor", alt_material="scales_deep", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"pelvis_flank_shingles_{q}", "pelvis", (s * 8.0, 22.8, -4.5), (s * 6.2, 21.0, -18.5), (s * 0.22, 0.12, -0.97), (s * 0.18, 0.10, -0.98), count=5, length=6.0, width=2.8, thickness=1.8, material="armor", alt_material="scales_deep", two_step=False, pattern="armor_shingle")

        if is_white:
            # Billowing Flowing White Fatalis Dorsal Mane along Neck Segments (`4` tiers!)
            for mi, (nb, mpos) in enumerate((
                ("neck_4", (0.0, 50.5, 35.8)),
                ("neck_3", (0.0, 44.5, 29.8)),
                ("neck_2", (0.0, 38.2, 23.8)),
                ("neck_1", (0.0, 31.8, 17.5)),
            )):
                m.span_box(f"white_mane_flow_{q}_{mi}", nb, (s * 2.2, mpos[1], mpos[2]), (s * 5.8, mpos[1] - 2.0, mpos[2] - 8.5), 3.4, 2.8, "fur", pattern="fur_mane", component="surface")

    # Crisp Dorsal Ridge Spines down Neck, Back & All 8 Tail Segments
    for db, dp in (
        ("neck_4", (0.0, 51.0, 36.2)),
        ("neck_3", (0.0, 45.2, 30.2)),
        ("neck_2", (0.0, 39.2, 24.5)),
        ("neck_1", (0.0, 32.8, 18.2)),
        ("chest", (0.0, 32.8, 10.5)),
        ("chest", (0.0, 31.8, 2.5)),
        ("pelvis", (0.0, 28.5, -5.5)),
        ("pelvis", (0.0, 26.0, -13.5)),
    ):
        m.span_box(f"dorsal_crest_{db}_{int(dp[2])}", db, dp, dp + np.array([0.0, 4.5, -4.0]), 1.8, 2.4, "horn" if not is_white else "fur", component="spines")

    for ti in range(8):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_spine_{ti}", tbones[ti], mid_t + np.array([0.0, 2.5, 0.0]), mid_t + np.array([0.0, 5.8 - 0.25 * ti, -3.8]), 1.6, 2.0, "horn", component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_scute_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.0, 3.8 - 0.35 * ti), 0.2, 0.0]), (1.6, max(1.3, 2.2 - 0.12 * ti), 6.0), "armor", rotation=(8.0, -s * 14.0, -s * 16.0), pattern="armor_shingle", component="armor")

    return m.doc


def build_fatalis(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_fatalis_specimen("canonical", stage_cap=stage_cap)


def build_white_fatalis(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_fatalis_specimen("white", stage_cap=stage_cap)
