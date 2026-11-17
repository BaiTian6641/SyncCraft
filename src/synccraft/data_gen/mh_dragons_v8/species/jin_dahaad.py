"""src/synccraft/data_gen/mh_dragons_v8/species/jin_dahaad.py

Monumental Sculpted Blockbench Reconstruction of JIN DAHAAD (`Rimechain Apex Leviathan`, MH Wilds).

Built strictly from `jin_dahaad_wide_board.jpg` (`colossal frost-metal battleship Leviathan`):
  - Gigantic broad anvil-shaped T-CREST SKULL (`broad flat steel-copper crown dome + overhanging
    anvil brow plates + heavy crocodile jaw`) — completely distinct from Lagiacrus/Mizutsune!
  - Multi-tier towering weathered BRONZE / FROSTED COPPER HEAT-SINK RADIATOR SHINGLE TOWERS lining
    neck, towering shoulder humps, spine, and pelvis (`seam_color=[106, 216, 252]` icy radiator slits!).
  - Massive broad armored pillar limbs gripping rock + gigantic segmented BRONZE-COPPER PADDLE TAIL
    topped with towering twin-row cooling radiator fins (`~360+` sculpted elements).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def jin_dahaad_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [64, 72, 86], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [42, 48, 62], "kind": "scales", "coat": "calm_flank"},
        "bronze_armor": {"color": [118, 92, 70], "kind": "armor", "coat": "armor_shingle"},
        "copper_radiator": {"color": [142, 108, 78], "kind": "armor", "coat": "armor_shingle"},
        "frost_steel": {"color": [174, 198, 218], "kind": "crystal", "coat": "crystal_facet"},
        "belly": {"color": [138, 152, 168], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [116, 98, 82], "kind": "horn", "coat": "horn_keratin"},
        "claw": {"color": [54, 58, 72], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [104, 48, 78], "kind": "mouth"},
        "tooth": {"color": [224, 232, 238], "kind": "bone"},
        "eye": {"color": [102, 230, 255], "kind": "eye", "emissive": 0.90},
    }


def build_jin_dahaad(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "jin_dahaad",
        "Jin Dahaad",
        "Rimechain Glacial Apex",
        "Leviathan",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Jin_Dahaad",
        materials=jin_dahaad_materials(),
        notes=[
            "Built from jin_dahaad_wide_board.jpg: colossal anvil T-crest bronze-armored head, heavy pillar limbs, multi-tier frosted copper heat-sink radiator towers along back & tail.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 16.5, -16.0))
    m.bone("chest", "pelvis", (0.0, 20.5, 7.0))
    m.bone("neck_1", "chest", (0.0, 24.5, 19.5))
    m.bone("neck_2", "neck_1", (0.0, 29.5, 27.5))
    m.bone("neck_3", "neck_2", (0.0, 34.5, 34.0))
    m.bone("head", "neck_3", (0.0, 36.2, 39.5))
    m.bone("jaw", "head", (0.0, 32.8, 40.5))

    # =========================================================================
    # STAGE 1: COLOSSAL GLACIAL DREADNOUGHT LEVIATHAN MASSES
    # =========================================================================
    torso_stations = [
        ((0.0, 15.8, -24.5), 16.5, 13.5, "pelvis"),
        ((0.0, 17.5, -13.5), 21.5, 16.5, "pelvis"),
        ((0.0, 19.8, -1.5), 25.5, 19.2, "chest"),
        ((0.0, 22.0, 10.5), 26.8, 20.5, "chest"),
        ((0.0, 24.0, 20.0), 19.5, 16.0, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=4.4, pattern="calm_flank", stage1=True)

    neck_stations = [
        ((0.0, 24.0, 19.0), 16.0, 14.5, "neck_1"),
        ((0.0, 28.8, 26.5), 14.2, 13.0, "neck_2"),
        ((0.0, 33.8, 33.0), 12.8, 11.8, "neck_3"),
        ((0.0, 36.2, 39.2), 12.2, 10.8, "head"),
    ]
    for i in range(len(neck_stations) - 1):
        p0, w0, h0, bn = neck_stations[i]
        p1, w1, h1, _ = neck_stations[i + 1]
        m.span_box(f"neck_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.4, pattern="calm_flank", stage1=True)

    # Signature Broad Square ANVIL T-CREST Skull & Armored Muzzle (`jin_dahaad_wide_board.jpg`!)
    m.box("skull_anvil_cranium", "head", (0.0, 37.8, 42.8), (15.5, 7.6, 10.8), "bronze_armor", rotation=(6.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("broad_anvil_muzzle", "head", (0.0, 36.0, 50.8), (11.2, 5.2, 7.5), "scales_deep", rotation=(8.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 32.2, 46.8), (10.2, 3.4, 15.2), "belly", face_materials={"up": "mouth"}, rotation=(3.0, 0.0, 0.0), component="jaw", stage1=True)

    # Massive 8-Bone Armored Dreadnought Tail
    t_pts = [
        ((0.0, 15.6, -23.5), 15.2, 12.4),
        ((1.8, 14.8, -32.8), 13.6, 11.4),
        ((4.5, 14.2, -42.0), 12.0, 10.4),
        ((7.2, 14.0, -51.2), 10.4, 9.6),
        ((8.2, 14.5, -60.2), 8.8, 9.0),
        ((6.8, 15.6, -68.8), 7.4, 8.6),
        ((3.6, 16.8, -76.5), 6.2, 8.0),
        ((0.2, 17.8, -83.2), 5.2, 7.2),
        ((-2.6, 18.4, -89.0), 3.6, 5.4),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(8)]
    for i in range(8):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.0, pattern="calm_flank", stage1=(i % 2 == 0))

    # Colossal Sprawled Armored Pillar Limbs
    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (13.0, 18.2, 10.0), (19.5, 11.2, 6.2), (19.5, 3.2, 14.5), (10.5, 9.6), (8.2, 7.8)),
        ("hind", "pelvis", (11.5, 16.0, -14.5), (18.0, 9.8, -9.8), (18.0, 3.0, -17.0), (10.5, 9.6), (8.2, 7.8)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0, p1, p2 = (s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.8, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "bronze_armor", len_pad=2.6, pattern="armor_shingle", stage1=True)
            m.box(f"{kind}_foot_pad_{q}", bp, (p2[0], 2.5, p2[2] + 2.8), (10.4, 3.6, 10.2), "scales_deep", face_materials={"up": "bronze_armor"}, stage1=True)

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: ANVIL T-CREST SKULL, COPPER HEAT-SINK TOWERS & RADIATOR FINS
    # =========================================================================
    # Huge Flat Hammerhead/Anvil T-Crest Brow Shield Overhanging Eyes (`jin_dahaad_wide_board.jpg`)
    m.box("anvil_crown_canopy", "head", (0.0, 41.2, 46.2), (18.8, 3.2, 11.2), "copper_radiator", seam_color=[106, 216, 252], rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="head")
    m.box("nasal_prow_armor", "head", (0.0, 38.8, 52.5), (8.2, 2.8, 6.2), "bronze_armor", rotation=(12.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("chin_icebreaker", "jaw", (0.0, 31.2, 53.0), (8.4, 2.8, 5.2), "copper_radiator", pattern="armor_shingle", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_anvil_wing_{q}", "head", (s * 9.8, 38.8, 41.5), (5.5, 4.2, 9.2), "bronze_armor", rotation=(8.0, s * 18.0, -s * 12.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 6.8, 37.5, 46.2), (1.3, 1.8, 3.0), "mouth", rotation=(6.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_glacial_{q}", "head", (s * 7.2, 37.5, 46.4), (0.9, 1.2, 2.2), "eye", rotation=(6.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")
        # Thick Swept Bronze Lateral Crown & Temple Radiator Pylons
        m.tapered_arch(f"t_crest_horn_{q}", "head", [(s * 7.8, 41.5, 40.5), (s * 13.5, 43.2, 35.5), (s * 17.5, 44.5, 30.2)], [(3.8, 3.2), (3.0, 2.6), (1.8, 1.6)], "copper_radiator", seam_color=[106, 216, 252], component="horns", stage1=True)
        for ti, tz in enumerate((42.0, 44.8, 47.6, 50.2, 52.4)):
            ux = 5.0 - 0.3 * ti
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * ux, 33.8, tz), (1.0, 1.6 if ti == 1 else 1.2, 1.0), "tooth", component="teeth")

        # Pillared Copper Bracer Limbs + Broad Glaciated Clawed Digits
        for kind, p0_r, p1_r, p2_r in (
            ("fore", (13.0, 18.2, 10.0), (19.5, 11.2, 6.2), (19.5, 3.2, 14.5)),
            ("hind", (11.5, 16.0, -14.5), (18.0, 9.8, -9.8), (18.0, 3.0, -17.0)),
        ):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0, p1, p2 = np.array([s * p0_r[0], p0_r[1], p0_r[2]]), np.array([s * p1_r[0], p1_r[1], p1_r[2]]), np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_radiator_shoulder_{q}", bu, p0 + np.array([s * 1.5, 2.5, 0.5]), (9.8, 9.4, 10.8), "bronze_armor", seam_color=[106, 216, 252], rotation=(10.0, s * 10.0, -s * 14.0), pattern="armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.14 + 0.22 * bi) + np.array([s * 2.8, 1.0, 1.2])
                m.box(f"{kind}_radiator_plate_{q}_{bi}", bl, pt_b, (2.8, 4.4, 5.6), "copper_radiator" if bi % 2 == 0 else "bronze_armor", seam_color=[106, 216, 252], rotation=(18.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
            for di, ang_d in ((0, -24.0), (1, -6.0), (2, 12.0), (3, 28.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-3.2 + 2.1 * di), 2.4, p2[2] + 6.4])
                tb = ta + dv * 4.5
                tc = tb + dv * 3.4 - np.array([0.0, 0.9, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 2.4, 2.4, "scales_deep", len_pad=0.5, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.8, 2.0, "claw", len_pad=0.4, component="claws")

    # Broad Armored Belly Plates (`belly`)
    for bi, (bn, pos, pw, kd, rx) in enumerate((
        ("neck_3", (0.0, 29.8, 34.5), 10.2, 6.2, 34.0),
        ("neck_2", (0.0, 24.0, 28.0), 11.8, 6.6, 38.0),
        ("neck_1", (0.0, 18.2, 21.5), 13.6, 7.0, 28.0),
        ("chest", (0.0, 13.5, 13.0), 17.5, 7.8, 14.0),
        ("chest", (0.0, 11.2, 4.5), 19.2, 8.2, 4.0),
        ("chest", (0.0, 10.5, -4.5), 18.0, 7.8, 0.0),
        ("pelvis", (0.0, 10.2, -13.5), 16.0, 7.4, -2.0),
        ("pelvis", (0.0, 10.4, -22.0), 13.6, 6.8, -4.0),
    )):
        m.box(f"belly_scute_{bi}", bn, pos, (pw, 2.8, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    # Signature Towering HEAT-SINK RADIATOR TOWERS along Spine (`chest` & `pelvis`)!
    for di, (db, dz, by, sep_x, h_tower) in enumerate((
        ("neck_1", 19.5, 31.2, 4.5, 9.5),
        ("chest", 11.5, 31.5, 6.4, 13.5),
        ("chest", 3.5, 30.8, 6.8, 15.0),
        ("chest", -4.5, 28.5, 6.0, 13.0),
        ("pelvis", -12.5, 25.5, 5.2, 11.0),
        ("pelvis", -20.5, 22.8, 4.2, 9.2),
    )):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            b_rad = np.array([s * sep_x, by - 0.5, dz])
            t_rad = b_rad + np.array([s * 1.8, h_tower, -4.8])
            # Stepped square bronze cooling fin stack (`wide beveled heat-sink monoliths!`)
            m.span_box(f"radiator_tower_{di}_{q}_a", db, b_rad, lerp3(b_rad, t_rad, 0.58), 4.6, 5.8, "bronze_armor", seam_color=[106, 216, 252], len_pad=0.6, pattern="armor_shingle", component="spines")
            m.span_box(f"radiator_tower_{di}_{q}_b", db, lerp3(b_rad, t_rad, 0.52), t_rad, 3.4, 4.6, "copper_radiator", tip_color=[156, 206, 236], tip_start=0.58, len_pad=0.4, pattern="armor_shingle", component="spines")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: MULTI-TIER BRONZE FLANK RADIATOR SHINGLES & BROAD TAIL PADDLE
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for nb, p_a, p_b in (
            ("neck_1", (s * 7.5, 24.5, 19.2), (s * 6.8, 28.8, 25.8)),
            ("neck_2", (s * 6.8, 29.2, 26.8), (s * 6.2, 33.4, 32.5)),
            ("neck_3", (s * 6.2, 34.0, 33.2), (s * 5.8, 36.0, 38.0)),
        ):
            m.shingle_arc(f"neck_radiator_shingles_{nb}_{q}", nb, p_a, p_b, (s * 0.35, 0.22, -0.89), (s * 0.28, 0.25, -0.90), count=5, length=6.2, width=3.2, thickness=2.0, material="bronze_armor", alt_material="copper_radiator", seam_color=[106, 216, 252], two_step=False, pattern="armor_shingle")
        # Multi-tier dorsal carapace shelves + heavy lower bronze rib cages + limb upper armor!
        m.shingle_arc(f"chest_radiator_flank_{q}", "chest", (s * 13.4, 22.8, 12.0), (s * 12.4, 19.8, -4.5), (s * 0.28, 0.16, -0.94), (s * 0.22, 0.14, -0.96), count=6, length=7.8, width=3.8, thickness=2.4, material="copper_radiator", alt_material="bronze_armor", seam_color=[106, 216, 252], two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"chest_lower_rib_{q}", "chest", (s * 12.2, 16.5, 10.5), (s * 11.4, 14.8, -3.5), (s * 0.35, -0.18, -0.92), (s * 0.30, -0.16, -0.94), count=5, length=6.8, width=3.2, thickness=2.0, material="scales_deep", alt_material="bronze_armor", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"pelvis_radiator_flank_{q}", "pelvis", (s * 11.4, 18.8, -5.5), (s * 8.6, 16.6, -22.0), (s * 0.25, 0.14, -0.95), (s * 0.20, 0.12, -0.97), count=6, length=7.2, width=3.4, thickness=2.2, material="bronze_armor", alt_material="frost_steel", two_step=False, pattern="armor_shingle")
        for kind, p0_r, p1_r in (("fore", (13.0, 18.2, 10.0), (19.5, 11.2, 6.2)), ("hind", (11.5, 16.0, -14.5), (18.0, 9.8, -9.8))):
            for thi in range(4):
                tp = lerp3((s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), 0.16 + 0.22 * thi) + np.array([s * 3.0, 1.0, 0.5])
                m.box(f"{kind}_upper_bronze_{q}_{thi}", f"{kind}_upper_{q}", tp, (3.2, 4.2, 5.0), "copper_radiator" if thi % 2 == 0 else "bronze_armor", seam_color=[106, 216, 252], rotation=(14.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
        for shi, (sh_b, sh_z, sh_y, sh_w) in enumerate((("chest", 11.5, 29.2, 5.2), ("chest", 3.5, 28.6, 5.6), ("chest", -4.5, 27.2, 5.0), ("pelvis", -12.5, 24.5, 4.6), ("pelvis", -20.5, 22.0, 4.0))):
            m.box(f"dorsal_deck_shelf_{shi}_{q}", sh_b, (s * 9.8, sh_y, sh_z), (sh_w, 2.4, 7.2), "bronze_armor", rotation=(6.0, s * 10.0, -s * 26.0), pattern="armor_shingle", component="armor")

    # Central Vertebral Heat-Sink Ridge + Gigantic Bronze Radiator Paddle Tail (`8` segments + twin side flanges!)
    for bi, (db, dp) in enumerate((
        ("neck_3", (0.0, 39.2, 34.8)),
        ("neck_2", (0.0, 34.5, 28.0)),
        ("neck_1", (0.0, 29.8, 21.2)),
        ("chest", (0.0, 30.8, 8.0)),
        ("chest", (0.0, 29.2, -1.0)),
        ("pelvis", (0.0, 26.2, -9.5)),
    )):
        m.box(f"neural_cooling_ridge_{bi}", db, dp, (3.6, 3.8, 6.2), "frost_steel", rotation=(24.0, 0.0, 0.0), pattern="crystal_facet", component="spines")

    for ti in range(8):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_radiator_blade_{ti}", tbones[ti], mid_t + np.array([0.0, 3.2, 0.0]), mid_t + np.array([0.0, 9.2 - 0.25 * ti, -4.5]), 2.6, 3.8, "copper_radiator" if ti % 2 == 0 else "bronze_armor", tip_color=[156, 206, 236], component="spines")
        m.box(f"tail_ventral_armor_{ti}", tbones[ti], mid_t + np.array([0.0, -4.5 + 0.25 * ti, 0.0]), (max(3.2, 9.6 - 0.65 * ti), 2.4, 8.2), "belly", rotation=(-6.0, 0.0, 0.0), pattern="belly_scute", component="surface")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_radiator_skirt_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.8, 5.6 - 0.45 * ti), 0.2, 0.0]), (2.4, max(1.8, 3.2 - 0.15 * ti), 7.2), "bronze_armor", seam_color=[106, 216, 252] if ti >= 3 else None, rotation=(8.0, -s * 14.0, -s * 20.0), pattern="armor_shingle", component="armor")

    return m.doc
