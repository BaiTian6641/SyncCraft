"""src/synccraft/data_gen/mh_dragons_v8/species/velkhana.py

Monumental Sculpted Blockbench Reconstruction of VELKHANA (`Lord of the Hoarfrost`, Elder Dragon).

Built directly from `velkhana_wide_board.jpg` (`[1],[2],[9],[11],[13],[14]`):
  - Slender aristocratic frost-sapphire/platinum skull crowned by a towering central + twin-side
    CRYSTAL TIARA OF ICE SPIRES (`crystal_facet` panes!) and razor sapphire snout.
  - Royal flared Elizabethan neck-shoulder ICE PLUMAGE MANTLE (`layered translucent cyan-white blades`)
    cascading behind the neck & shoulders.
  - Shimmering metallic-silver/sapphire carapace, broad frosted feather-crystal 3-bank dragon wings
    studded along leading spars with long pointed ice feathers, and a sleek whipping tail ending in
    an elongated spear-like FACETED ICE RAPIER LANCE (`tail_rapier`!).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def velkhana_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [124, 152, 186], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [76, 102, 142], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [158, 184, 214], "kind": "armor", "coat": "armor_shingle"},
        "ice_crystal": {"color": [162, 214, 246], "kind": "crystal", "coat": "crystal_facet"},
        "ice_deep": {"color": [86, 138, 198], "kind": "crystal", "coat": "crystal_facet"},
        "belly": {"color": [206, 218, 232], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [132, 178, 224], "kind": "crystal", "coat": "crystal_facet"},
        "wing_mem": {"color": [152, 184, 218], "kind": "membrane", "coat": "wing_legiana"},
        "wing_back": {"color": [96, 128, 172], "kind": "membrane", "coat": "wing_legiana"},
        "claw": {"color": [84, 114, 158], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [108, 62, 104], "kind": "mouth"},
        "tooth": {"color": [234, 240, 248], "kind": "bone"},
        "eye": {"color": [96, 218, 255], "kind": "eye", "emissive": 0.88},
    }


def build_velkhana(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "velkhana",
        "Velkhana",
        "Empress of Frost",
        "Elder Dragon",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Velkhana",
        materials=velkhana_materials(),
        notes=[
            "Built from velkhana_wide_board.jpg: ice-crystal tiara crown, narrow aristocratic muzzle, flared ice-plumage royal neck ruff, crystalline wing feathers, and scorpion-whip rapier tail spear.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 21.5, -12.0))
    m.bone("chest", "pelvis", (0.0, 26.5, 5.5))
    m.bone("neck_1", "chest", (0.0, 32.0, 16.5))
    m.bone("neck_2", "neck_1", (0.0, 38.0, 23.0))
    m.bone("head", "neck_2", (0.0, 42.2, 29.8))
    m.bone("jaw", "head", (0.0, 39.8, 30.6))

    # =========================================================================
    # STAGE 1: LITHE ARISTOCRATIC ICE ELDER BLOCKOUT
    # =========================================================================
    torso_stations = [
        ((0.0, 21.0, -18.5), 10.5, 9.6, "pelvis"),
        ((0.0, 22.8, -8.5), 13.0, 11.8, "pelvis"),
        ((0.0, 26.2, 2.0), 15.8, 14.0, "chest"),
        ((0.0, 29.2, 12.0), 14.8, 13.2, "chest"),
        ((0.0, 31.2, 18.0), 11.8, 11.0, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.6, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 31.5, 16.0), (0.0, 37.2, 22.2), 9.4, 9.0),
        ("neck_2", (0.0, 37.2, 22.2), (0.0, 42.0, 29.5), 8.0, 7.6),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=2.8, pattern="calm_flank", stage1=True)

    # Slender Tapered Crystalline Skull (`narrow snout, regal silhouette`)
    m.box("skull_cranium", "head", (0.0, 42.8, 32.8), (7.8, 5.2, 7.6), "armor", rotation=(6.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("ice_muzzle", "head", (0.0, 41.8, 38.8), (4.8, 3.4, 6.2), "scales", rotation=(10.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 39.6, 36.4), (4.6, 2.0, 9.8), "belly", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    # Whip Tail Ending in Giant Faceted Ice-Rapier Lance (`tail_1..7`)
    t_pts = [
        ((0.0, 21.2, -18.0), 9.2, 8.4),
        ((0.8, 20.4, -27.0), 7.6, 7.0),
        ((2.2, 20.0, -36.0), 6.2, 5.8),
        ((3.2, 20.8, -45.0), 5.0, 4.8),
        ((2.8, 22.4, -54.0), 4.0, 4.0),
        ((1.6, 24.4, -62.5), 3.2, 3.2),
        ((0.4, 26.2, -70.5), 2.5, 2.5),
        ((0.0, 27.2, -79.5), 1.4, 1.4),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(7)]
    for i in range(7):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "ice_crystal" if i == 6 else "scales", len_pad=2.4, pattern="crystal_facet" if i == 6 else "calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (7.8, 23.0, 9.2), (11.2, 12.5, 6.0), (10.8, 3.2, 11.2), (5.8, 5.6), (4.6, 4.4)),
        ("hind", "pelvis", (7.4, 20.5, -12.0), (10.8, 11.2, -6.5), (10.4, 3.0, -13.8), (6.8, 6.6), (5.0, 4.8)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0, p1, p2 = (s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.0, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.0, pattern="leg_bands", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.4, p2[2] + 2.2), (6.2, 3.0, 6.4), "scales_deep", stage1=True)

    # Majestic Ice-Feathered 3-Bank Elder Wings (`winged_sail_rig`)
    for s, q in ((1.0, "l"), (-1.0, "r")):
        el_pt, wr_pt = (s * 20.5, 43.0, 3.8), (s * 33.0, 52.5, 14.0)
        m.winged_sail_rig(
            s,
            "chest",
            (s * 7.0, 30.8, 7.8),
            el_pt,
            wr_pt,
            ((s * 66.0, 59.0, 6.0), (s * 60.0, 44.0, -7.5), (s * 47.0, 31.0, -17.5), (s * 30.0, 21.0, -22.0)),
            (s * 6.8, 22.5, -7.5),
            spar_mat="armor",
            hub_mat="ice_crystal",
            claw_mat="ice_deep",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_legiana",
            spar_w=3.0,
            arm_w=4.4,
            leading_scales=7,
            tattered_outer=False,
        )
        if stage_cap >= 2:
            b_wr = f"wing_wrist_{q}"
            for fi in range(4):
                fp = lerp3(el_pt, wr_pt, 0.25 + 0.22 * fi)
                m.span_box(f"wing_ice_plume_{q}_{fi}", f"wing_fore_{q}", fp, fp + np.array([s * 2.5, 5.5, -4.5]), 1.6, 2.2, "ice_crystal", pattern="crystal_facet", component="spines")
            m.span_box(f"wing_wrist_crystal_{q}", b_wr, wr_pt, np.array(wr_pt) + np.array([s * 5.0, 7.5, -4.0]), 2.0, 2.6, "ice_crystal", pattern="crystal_facet", component="spines")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: ICE TIARA CROWN, ROYAL PLUMAGE MANTLE & RAPIER TAIL LANCE
    # =========================================================================
    m.box("nasal_crystal_bridge", "head", (0.0, 43.8, 37.8), (3.8, 1.8, 6.4), "ice_deep", rotation=(12.0, 0.0, 0.0), pattern="crystal_facet", component="snout")
    m.span_box("snout_crystal_point", "head", (0.0, 42.2, 41.2), (0.0, 41.2, 43.8), 2.4, 1.8, "ice_crystal", pattern="crystal_facet", component="snout")

    # Iconic Central Velkhana Crown Tiara Lance + Side Crest Crown (`3` soaring ice spires!)
    m.tapered_arch("crown_tiara_center", "head", [(0.0, 44.8, 32.5), (0.0, 50.8, 27.5), (0.0, 56.8, 22.8)], [(2.4, 2.8), (1.8, 2.2), (0.9, 1.2)], "ice_crystal", tip_color=[220, 244, 255], component="horns", stage1=True)

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_ice_plate_{q}", "head", (s * 4.0, 42.2, 33.0), (1.9, 3.2, 5.8), "armor", rotation=(8.0, s * 14.0, -s * 12.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 3.8, 43.4, 35.2), (0.9, 1.3, 2.3), "mouth", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_velk_{q}", "head", (s * 4.1, 43.4, 35.4), (0.7, 0.9, 1.7), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")
        m.tapered_arch(f"crown_tiara_wing_{q}", "head", [(s * 2.8, 44.6, 32.2), (s * 5.8, 50.2, 26.8), (s * 8.2, 55.4, 21.8)], [(2.2, 2.6), (1.7, 2.0), (0.9, 1.2)], "ice_crystal", tip_color=[220, 244, 255], component="horns", stage1=True)
        m.tapered_arch(f"cheek_tiara_fin_{q}", "head", [(s * 4.2, 42.2, 30.8), (s * 7.8, 43.2, 25.5), (s * 10.6, 44.0, 20.5)], [(1.6, 2.2), (1.3, 1.8), (0.8, 1.0)], "ice_deep", pattern="crystal_facet", component="horns")

        for ti, tz in enumerate((34.2, 36.4, 38.6, 40.4)):
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * (2.2 - 0.25 * ti), 40.4, tz), (0.7, 1.1, 0.7), "tooth", component="teeth")

        for kind, p0_r, p1_r, p2_r in (
            ("fore", (7.8, 23.0, 9.2), (11.2, 12.5, 6.0), (10.8, 3.2, 11.2)),
            ("hind", (7.4, 20.5, -12.0), (10.8, 11.2, -6.5), (10.4, 3.0, -13.8)),
        ):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0, p1, p2 = np.array([s * p0_r[0], p0_r[1], p0_r[2]]), np.array([s * p1_r[0], p1_r[1], p1_r[2]]), np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_shoulder_shield_{q}", bu, p0 + np.array([s * 0.8, 1.4, 0.5]), (5.6, 5.4, 6.6), "ice_crystal", rotation=(8.0, s * 10.0, -s * 14.0), pattern="crystal_facet")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.16 + 0.22 * bi) + np.array([s * 1.5, 0.6, 1.0])
                m.box(f"{kind}_greave_{q}_{bi}", bl, pt_b, (2.0, 3.0, 3.8), "armor", rotation=(18.0, s * 14.0, -s * 20.0), pattern="armor_shingle", component="armor")
            for di, ang_d in ((0, -18.0), (1, 0.0), (2, 18.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-1.6 + 1.6 * di), 2.0, p2[2] + 4.4])
                tb = ta + dv * 3.4
                tc = tb + dv * 2.6 - np.array([0.0, 0.7, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 1.6, 1.6, "scales_deep", len_pad=0.4, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.2, 1.3, "claw", len_pad=0.3, component="claws")

    # Signature Radiating Elizabethan Royal Ice-Plumage Neck Mantle (`velkhana_wide_board.jpg`!)
    for ci, (bn, cp, clen, cw, count) in enumerate((
        ("neck_2", (0.0, 42.5, 25.0), 11.2, 3.4, 7),
        ("neck_1", (0.0, 37.8, 18.5), 14.2, 4.0, 9),
        ("chest", (0.0, 34.2, 11.5), 16.0, 4.4, 9),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            ang = frac * 70.0
            ra = math.radians(ang)
            b_c = np.array([math.sin(ra) * (3.6 + 1.6 * ci), cp[1] - abs(frac) * 3.8, cp[2]])
            dv = unit([math.sin(ra) * 0.56, math.cos(ra) * 0.34, -0.80])
            t_c = b_c + dv * clen
            m.span_box(f"ice_plumage_{ci}_{k}", bn, b_c, t_c, cw, 1.8, "ice_crystal" if k % 2 == 0 else "armor", tip_color=[220, 244, 255] if k % 2 == 0 else None, len_pad=0.7, pattern="crystal_facet", component="spines")

    # Frosted Belly Scutes + Iconic GIANT ICE-RAPIER LANCE TAIL TIP (`tail_7`)
    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 36.2, 24.5), 6.2, 5.2, 22.0),
        ("neck_1", (0.0, 30.2, 19.5), 7.6, 5.8, 18.0),
        ("chest", (0.0, 21.8, 11.5), 10.2, 6.8, 10.0),
        ("chest", (0.0, 19.5, 3.5), 11.2, 7.0, 2.0),
        ("pelvis", (0.0, 18.5, -5.0), 10.2, 6.8, 0.0),
        ("pelvis", (0.0, 17.8, -13.5), 8.8, 6.2, -2.0),
    )):
        m.box(f"belly_scute_{ki}", kb, kpos, (kw, 2.0, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    r_base = v3(t_pts[6][0])
    r_tip = r_base + np.array([-0.5, 1.8, -19.5])
    m.tapered_arch("tail_rapier_spear", tbones[6], [r_base, lerp3(r_base, r_tip, 0.48), r_tip], [(2.8, 2.8), (3.4, 2.4), (0.8, 0.8)], "ice_crystal", tip_color=[224, 248, 255], pattern="crystal_facet", component="spines", stage1=True)

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: CRYSTAL CARAPACE SHINGLING, NECK COLLAR PLEATS & TAIL PRISMS
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for nb, p_a, p_b in (
            ("neck_1", (s * 4.0, 32.0, 16.5), (s * 3.6, 36.8, 22.0)),
            ("neck_2", (s * 3.4, 37.8, 23.0), (s * 3.0, 41.5, 29.0)),
        ):
            m.shingle_arc(f"neck_frost_plumes_{nb}_{q}", nb, p_a, p_b, (s * 0.36, 0.22, -0.89), (s * 0.28, 0.25, -0.90), count=5, length=5.0, width=2.6, thickness=1.5, material="armor", alt_material="ice_crystal", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"chest_frost_shingles_{q}", "chest", (s * 8.4, 27.2, 11.5), (s * 7.2, 24.0, -3.0), (s * 0.25, 0.14, -0.96), (s * 0.20, 0.12, -0.97), count=6, length=6.6, width=3.2, thickness=2.0, material="armor", alt_material="ice_crystal", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"pelvis_frost_shingles_{q}", "pelvis", (s * 6.8, 22.8, -4.5), (s * 5.2, 21.2, -16.5), (s * 0.22, 0.12, -0.97), (s * 0.18, 0.10, -0.98), count=5, length=5.8, width=2.8, thickness=1.8, material="armor", alt_material="scales_deep", two_step=False, pattern="armor_shingle")

    for db, dp in (
        ("chest", (0.0, 32.8, 9.5)),
        ("chest", (0.0, 31.8, 2.0)),
        ("pelvis", (0.0, 28.5, -5.5)),
        ("pelvis", (0.0, 26.2, -13.0)),
    ):
        m.span_box(f"dorsal_ice_crest_{int(dp[2])}", db, dp, dp + np.array([0.0, 5.5, -4.5]), 1.6, 2.2, "ice_crystal", pattern="crystal_facet", component="spines")

    for ti in range(7):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_ice_crest_{ti}", tbones[ti], mid_t + np.array([0.0, 2.2, 0.0]), mid_t + np.array([0.0, 5.0 - 0.2 * ti, -3.8]), 1.4, 1.9, "ice_crystal" if ti % 2 == 0 else "armor", pattern="crystal_facet", component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_frost_plate_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.0, 3.2 - 0.28 * ti), 0.1, 0.0]), (1.4, 1.9, 5.6), "armor", rotation=(8.0, -s * 14.0, -s * 16.0), pattern="armor_shingle", component="armor")

    return m.doc
