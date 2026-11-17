"""src/synccraft/data_gen/mh_dragons_v8/species/rey_dau.py

Monumental Sculpted Blockbench Reconstruction of REY DAU (`Fulminous Railgun Apex`, Flying Wyvern).

Built strictly from `rey_dau_wide_board.jpg` (`[1],[6],[9],[10],[11],[12]`):
  - Burnt-ochre / golden desert-sand plate armor streaked with charred obsidian/fulgurite veins &
    pale ivory-sand sternum scutes.
  - Iconic HINGED RAILGUN CALIPER CRANIAL HORNS (`rail_horn_l/r` on `head`): massive broad amber/obsidian
    segmented caliper prong shields clamping forward ahead of the narrow hawk muzzle during discharge.
  - Titanic curved wingtip FULGURITE SCYTHE BLADES wrapping outer wingtips + vivid sapphire-blue fulgurite
    lightning crystals protruding along leading spars and shoulders.
  - Armored wyvern torso with heavy layered golden dorsal scales & unique TWIN-PRONG SPLIT FULGURITE TAIL!
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def rey_dau_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [156, 120, 56], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [108, 80, 40], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [46, 42, 46], "kind": "armor", "coat": "armor_shingle"},
        "ochre_plate": {"color": [188, 146, 68], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [216, 198, 154], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [196, 148, 62], "kind": "horn", "coat": "horn_keratin"},
        "fulgurite": {"color": [36, 48, 108], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.12},
        "fulgurite_bright": {"color": [54, 106, 186], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.22},
        "wing_mem": {"color": [166, 128, 64], "kind": "membrane", "coat": "wing_rathalos"},
        "wing_back": {"color": [112, 84, 44], "kind": "membrane", "coat": "wing_rathalos"},
        "claw": {"color": [38, 36, 44], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [126, 46, 62], "kind": "mouth"},
        "tooth": {"color": [232, 224, 198], "kind": "bone"},
        "eye": {"color": [88, 224, 255], "kind": "eye", "emissive": 0.88},
    }


def build_rey_dau(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "rey_dau",
        "Rey Dau",
        "Fulminous Railgun Wyvern",
        "Flying Wyvern",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Rey_Dau",
        materials=rey_dau_materials(),
        notes=[
            "Built strictly from rey_dau_wide_board.jpg (`MH Wilds`): hinged `rail_horn_l/r` caliper prongs, curved fulgurite wingtip scythes, and sapphire-crystal twin-prong fork tail.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 22.0, -8.0))
    m.bone("chest", "pelvis", (0.0, 27.0, 6.5))
    m.bone("neck_1", "chest", (0.0, 32.0, 17.0))
    m.bone("neck_2", "neck_1", (0.0, 37.5, 24.5))
    m.bone("head", "neck_2", (0.0, 40.5, 31.0))
    m.bone("jaw", "head", (0.0, 37.8, 31.8))

    # =========================================================================
    # STAGE 1: AGILE RAILGUN WYVERN PRIMARY MASSES
    # =========================================================================
    torso_stations = [
        ((0.0, 21.0, -14.5), 11.0, 10.2, "pelvis"),
        ((0.0, 23.0, -4.5), 14.0, 12.6, "pelvis"),
        ((0.0, 26.8, 6.0), 17.5, 15.2, "chest"),
        ((0.0, 30.2, 14.5), 15.5, 14.0, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.8, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 31.2, 16.0), (0.0, 36.2, 23.2), 10.5, 10.0),
        ("neck_2", (0.0, 36.2, 23.2), (0.0, 40.5, 30.6), 9.0, 8.8),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    m.box("skull_cranium", "head", (0.0, 41.2, 34.0), (9.0, 5.8, 7.8), "ochre_plate", face_materials={"up": "armor"}, rotation=(8.0, 0.0, 0.0), component="head", stage1=True)
    m.box("hawk_beak_snout", "head", (0.0, 40.0, 39.8), (5.2, 3.6, 5.8), "scales", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 37.8, 37.5), (5.0, 2.2, 9.2), "belly", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 21.2, -13.5), 10.0, 9.2),
        ((0.8, 20.4, -22.5), 8.6, 8.0),
        ((2.0, 20.0, -31.5), 7.2, 6.8),
        ((3.2, 20.6, -40.5), 6.0, 5.8),
        ((3.4, 22.0, -49.5), 5.0, 5.0),
        ((2.2, 24.0, -58.2), 4.0, 4.2),
        ((0.6, 26.0, -66.5), 3.4, 3.6),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(6)]
    for i in range(6):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.6, pattern="calm_flank", stage1=(i % 2 == 0))

    for s, q in ((1.0, "l"), (-1.0, "r")):
        p0, p1, p2 = (s * 8.0, 21.0, -6.5), (s * 12.0, 12.0, -1.0), (s * 11.8, 3.6, -8.5)
        bu = m.bone(f"hind_upper_{q}", "pelvis", p0)
        bl = m.bone(f"hind_lower_{q}", bu, p1)
        bp = m.bone(f"hind_paw_{q}", bl, p2)
        m.span_box(f"hind_thigh_{q}", bu, p0, p1, 8.4, 8.0, "scales", len_pad=2.2, pattern="calm_flank", stage1=True)
        m.span_box(f"hind_shin_{q}", bl, p1, p2, 5.8, 5.6, "scales", len_pad=2.2, pattern="leg_bands", stage1=True)
        m.box(f"hind_ankle_{q}", bp, (p2[0], 2.6, p2[2] + 2.2), (7.4, 3.4, 7.2), "scales_deep", stage1=True)

    # Swept Scythe-Wing 3-Bank Architecture (`巨大 sweeping fulgurite blade wings!`)
    for s, q in ((1.0, "l"), (-1.0, "r")):
        wr_pt = (s * 34.0, 48.0, 16.0)
        tip0 = (s * 70.0, 62.0, 4.0)
        wb = m.winged_sail_rig(
            s,
            "chest",
            (s * 7.8, 29.8, 8.0),
            (s * 22.0, 40.5, 4.5),
            wr_pt,
            (tip0, (s * 60.0, 42.0, -10.5), (s * 45.0, 29.0, -19.5), (s * 28.0, 20.0, -23.0)),
            (s * 7.2, 22.0, -8.0),
            spar_mat="ochre_plate",
            hub_mat="armor",
            claw_mat="fulgurite",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_rathalos",
            spar_w=3.4,
            arm_w=4.8,
            leading_scales=7,
            tattered_outer=False,
        )
        if stage_cap >= 2:
            # Iconic Giant Fulgurite Wingtip Sickle-Scythes (`rey_dau_wide_board.jpg` `[1],[6],[12]`!)
            b_dist = f"wing_distal_{q}"
            scythe_a = lerp3(wr_pt, tip0, 0.55)
            scythe_b = v3(tip0) + np.array([s * 6.5, 5.5, -2.5])
            scythe_c = v3(tip0) + np.array([s * 12.0, 1.5, -10.5])
            m.tapered_arch(f"fulgurite_wing_scythe_{q}", b_dist, [scythe_a, scythe_b, scythe_c], [(3.8, 2.2), (3.2, 1.8), (1.4, 1.2)], "armor", tip_color=[64, 108, 186], tip_start=0.55, component="spines", stage1=True)
            # Fulgurite sapphire crystals protruding along wing forearm leading spar
            b_fo = f"wing_fore_{q}"
            for ci in range(3):
                c_pt = lerp3((s * 22.0, 40.5, 4.5), wr_pt, 0.25 + 0.28 * ci)
                m.span_box(f"wing_fulgurite_spur_{q}_{ci}", b_fo, c_pt, c_pt + np.array([s * 1.5, 4.8, 3.8]), 1.8, 2.2, "fulgurite_bright", pattern="crystal_facet", component="spines")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: ARTICULATING RAILGUN CALIPER HORNS, HAWK SKULL & TWIN-PRONG TAIL
    # =========================================================================
    m.box("hawk_beak_ridge", "head", (0.0, 42.2, 38.8), (4.2, 2.0, 6.2), "armor", rotation=(14.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.span_box("hawk_beak_hook", "head", (0.0, 40.2, 42.2), (0.0, 38.0, 44.6), 3.0, 2.2, "armor", component="snout")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_crest_{q}", "head", (s * 4.5, 40.4, 34.0), (2.2, 3.6, 6.0), "ochre_plate", rotation=(8.0, s * 14.0, -s * 10.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.2, 41.8, 36.2), (1.0, 1.4, 2.4), "armor", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_cyan_{q}", "head", (s * 4.5, 41.8, 36.4), (0.7, 1.0, 1.8), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        # Iconic ARTICULATING RAILGUN CALIPER HORNS (`rail_horn_l` / `rail_horn_r`!)
        rh_bone = m.bone(f"rail_horn_{q}", "head", (s * 3.8, 43.2, 34.5))
        m.tapered_arch(
            f"railgun_caliper_arm_{q}",
            rh_bone,
            [
                (s * 3.8, 43.4, 34.8),
                (s * 8.2, 45.2, 41.2),
                (s * 9.8, 44.6, 48.5),
                (s * 7.5, 42.8, 55.2),
            ],
            [(3.0, 2.4), (3.4, 2.2), (2.8, 1.9), (1.4, 1.4)],
            "ochre_plate",
            tip_color=[56, 88, 162],
            tip_start=0.58,
            component="horns",
            stage1=True,
        )
        m.box(f"railgun_inner_core_{q}", rh_bone, (s * 7.8, 43.8, 46.8), (1.4, 1.6, 6.8), "fulgurite_bright", rotation=(-8.0, -s * 16.0, 0.0), pattern="crystal_facet", component="horns")

        # Secondary swept rear brow horns (`frame skull backwards like rey_dau_wide_board.jpg [12]!`)
        m.tapered_arch(f"rear_crown_blade_{q}", "head", [(s * 4.2, 42.5, 32.0), (s * 8.8, 45.2, 26.2), (s * 12.2, 47.0, 20.5)], [(2.4, 2.8), (1.8, 2.2), (1.0, 1.2)], "armor", component="horns")

        # Talon legs + ankle spurs
        bp = f"hind_paw_{q}"
        for bi in range(4):
            pt_b = lerp3((s * 12.0, 12.0, -1.0), (s * 11.8, 3.6, -8.5), 0.16 + 0.22 * bi) + np.array([s * 1.8, 0.8, 1.5])
            m.box(f"shin_plate_{q}_{bi}", f"hind_lower_{q}", pt_b, (2.2, 3.0, 3.8), "ochre_plate" if bi % 2 == 0 else "armor", rotation=(16.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
        for di, ang_d in ((0, -22.0), (1, 0.0), (2, 22.0)):
            yaw = math.radians(s * ang_d)
            dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            ta = np.array([s * (11.8 + (-2.0 + 2.0 * di) * s), 2.4, -3.8])
            tb = ta + dv * 4.2
            tc = tb + dv * 3.4 - np.array([0.0, 1.0, 0.0])
            m.span_box(f"talon_toe_{q}_{di}", bp, ta, tb, 2.0, 2.0, "scales_deep", len_pad=0.4, component="claws")
            m.span_box(f"talon_claw_{q}_{di}", bp, tb, tc, 1.5, 1.6, "claw", len_pad=0.3, component="claws")

    # Cream Belly Scutes + Iconic TWIN-PRONG FULGURITE RAIL TAIL (`tail_6`)
    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 35.5, 25.0), 6.8, 5.4, 22.0),
        ("neck_1", (0.0, 30.8, 20.2), 8.4, 6.0, 18.0),
        ("chest", (0.0, 26.8, 16.5), 10.2, 6.8, 22.0),
        ("chest", (0.0, 22.8, 13.0), 11.2, 7.0, 12.0),
        ("chest", (0.0, 20.0, 5.8), 12.6, 7.2, 4.0),
        ("chest", (0.0, 19.2, -1.8), 11.6, 7.0, 0.0),
        ("pelvis", (0.0, 18.0, -9.2), 10.6, 6.6, -2.0),
    )):
        m.box(f"belly_scute_{ki}", kb, kpos, (kw, 2.2, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    tail_hub = np.array([0.6, 26.0, -66.5])
    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.tapered_arch(
            f"tail_fulgurite_prong_{q}",
            tbones[5],
            [tail_hub + np.array([s * 1.4, 0.4, -1.5]), tail_hub + np.array([s * 3.6, 1.6, -9.2]), tail_hub + np.array([s * 2.2, 3.0, -17.5])],
            [(2.4, 2.6), (2.2, 2.2), (1.1, 1.2)],
            "fulgurite",
            tip_color=[64, 118, 204],
            tip_start=0.45,
            component="spines",
            stage1=True,
        )

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: DESERT FULGURITE CARAPACE SHINGLING & DORSAL TAIL RAILS
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for nb, p_a, p_b in (
            ("neck_1", (s * 4.6, 32.0, 16.5), (s * 4.0, 36.2, 22.8)),
            ("neck_2", (s * 3.8, 37.0, 24.0), (s * 3.4, 40.0, 29.8)),
        ):
            m.shingle_arc(f"neck_ochre_plates_{nb}_{q}", nb, p_a, p_b, (s * 0.35, 0.25, -0.88), (s * 0.28, 0.28, -0.90), count=5, length=5.2, width=2.8, thickness=1.6, material="ochre_plate", alt_material="armor", two_step=False, pattern="armor_shingle")
        m.box(f"wing_pauldron_{q}", "chest", (s * 8.6, 31.0, 9.0), (5.4, 3.8, 8.0), "ochre_plate", rotation=(14.0, s * 16.0, -s * 22.0), pattern="armor_shingle", component="armor")
        for wi in range(4):
            w_pt = lerp3((s * 7.8, 29.8, 8.0), (s * 22.0, 40.5, 4.5), 0.15 + 0.24 * wi)
            m.box(f"wing_humerus_scale_{q}_{wi}", f"wing_{q}", w_pt + np.array([0.0, 2.2, 1.2]), (3.6, 2.4, 4.8), "ochre_plate" if wi % 2 == 0 else "armor", rotation=(12.0, s * 18.0, 0.0), pattern="armor_shingle", component="armor")
        for thi in range(4):
            th_p = lerp3((s * 8.0, 21.0, -6.5), (s * 12.0, 12.0, -1.0), 0.18 + 0.22 * thi) + np.array([s * 2.6, 0.8, 0.6])
            m.box(f"thigh_fulgur_plate_{q}_{thi}", f"hind_upper_{q}", th_p, (3.2, 3.8, 4.6), "ochre_plate" if thi % 2 == 0 else "armor", rotation=(14.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
        m.shingle_arc(f"chest_ochre_shingles_{q}", "chest", (s * 9.2, 27.8, 12.2), (s * 8.0, 24.5, -2.5), (s * 0.25, 0.14, -0.96), (s * 0.20, 0.12, -0.97), count=6, length=6.8, width=3.4, thickness=2.1, material="ochre_plate", alt_material="armor", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"chest_ventral_ribs_{q}", "chest", (s * 6.4, 21.0, 11.2), (s * 6.0, 19.5, -1.5), (s * 0.45, -0.24, -0.86), (s * 0.40, -0.22, -0.88), count=5, length=5.0, width=2.6, thickness=1.5, material="belly", alt_material="scales", two_step=False, pattern="belly_scute")
        m.shingle_arc(f"pelvis_ochre_shingles_{q}", "pelvis", (s * 7.4, 22.8, -4.5), (s * 5.6, 21.2, -15.5), (s * 0.22, 0.12, -0.97), (s * 0.18, 0.10, -0.98), count=5, length=6.0, width=2.8, thickness=1.8, material="armor", alt_material="ochre_plate", two_step=False, pattern="armor_shingle")
        # Shoulder fulgurite crystal spires
        m.span_box(f"shoulder_fulgurite_{q}", "chest", (s * 6.5, 31.8, 11.2), (s * 9.8, 37.8, 4.8), 2.4, 2.8, "fulgurite_bright", pattern="crystal_facet", component="spines")

    for ni, (nb, npos, nw) in enumerate((
        ("neck_2", (0.0, 40.2, 26.2), 5.6),
        ("neck_1", (0.0, 35.8, 20.0), 6.4),
        ("chest", (0.0, 34.5, 12.2), 7.8),
        ("chest", (0.0, 33.4, 4.2), 8.2),
        ("pelvis", (0.0, 30.0, -4.5), 7.2),
    )):
        m.box(f"dorsal_armor_plate_{ni}", nb, npos, (nw, 2.2, 6.2), "armor" if ni % 2 == 0 else "ochre_plate", rotation=(20.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        m.span_box(f"dorsal_fin_{ni}", nb, npos, npos + np.array([0.0, 4.2, -4.0]), 1.8, 2.4, "ochre_plate", component="spines")

    for ti in range(6):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.box(f"tail_fulgur_armor_{ti}", tbones[ti], mid_t + np.array([0.0, 3.0 - 0.15 * ti, 0.0]), (max(2.4, 5.4 - 0.45 * ti), 2.0, 6.6), "armor" if ti % 2 == 0 else "ochre_plate", rotation=(12.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.span_box(f"tail_fulgurite_spike_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.4, 3.6 - 0.3 * ti), 0.2, 0.0]), mid_t + np.array([s * max(2.6, 5.8 - 0.35 * ti), 1.6, -3.8]), 1.6, 1.9, "fulgurite" if ti >= 3 else "ochre_plate", component="spines")

    return m.doc
