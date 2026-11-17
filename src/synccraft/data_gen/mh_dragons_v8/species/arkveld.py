"""src/synccraft/data_gen/mh_dragons_v8/species/arkveld.py

Monumental Sculpted Blockbench Reconstruction of ARKVELD (`The White Wraith`, Extinct-Lineage Flying Wyvern).

Built strictly from `arkveld_wide_board.jpg`:
  - Ghostly pale pearl-platinum & bone-grey armor carapace draped in cascading layers of flowing
    silver-white fur/feather mantle locks along neck, shoulders, torso flanks & tail base.
  - Compact pale wraith skull with fierce crimson eyes, flat pale muzzle, and heavy swept-down/back
    ribbed SILVER RAM HORNS flanking cheeks!
  - Dual Giant Articulating 3-LINK WING-CHAINBLADE WHIPS (`chain_1_l/r -> chain_2_l/r -> chain_3_l/r`!)
    extending boldly forward/downward from the wing wrists with stacked jagged pale-steel chain
    blades & subtle crimson dragon-absorption fissures (`<= 3%` emissive), backed by true 6-bank
    tattered wraith feather-wing cape panels (`wing_l/fore/wrist/mid/distal`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def arkveld_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [146, 148, 158], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [96, 98, 112], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [182, 184, 194], "kind": "armor", "coat": "armor_shingle"},
        "chain_steel": {"color": [198, 200, 210], "kind": "armor", "coat": "armor_shingle"},
        "fur": {"color": [224, 224, 232], "kind": "fur", "coat": "fur_mane"},
        "belly": {"color": [158, 156, 168], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [128, 130, 144], "kind": "horn", "coat": "horn_keratin"},
        "wing_mem": {"color": [194, 194, 206], "kind": "membrane", "coat": "wing_legiana"},
        "wing_back": {"color": [132, 134, 148], "kind": "membrane", "coat": "wing_legiana"},
        "claw": {"color": [68, 68, 82], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [116, 44, 68], "kind": "mouth"},
        "tooth": {"color": [236, 234, 228], "kind": "bone"},
        "eye": {"color": [248, 64, 72], "kind": "eye", "emissive": 0.88},
    }


def build_arkveld(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "arkveld",
        "Arkveld",
        "The White Wraith",
        "Flying Wyvern",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Arkveld",
        materials=arkveld_materials(),
        notes=[
            "Built from arkveld_wide_board.jpg: cascading pale fur cape, swept silver ram horns, 6-bank tattered wraith wings, and dual 3-link articulating whip chainblades (`chain_1..3_l/r`).",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 22.0, -8.0))
    m.bone("chest", "pelvis", (0.0, 27.0, 6.5))
    m.bone("neck_1", "chest", (0.0, 32.5, 17.5))
    m.bone("neck_2", "neck_1", (0.0, 38.0, 25.0))
    m.bone("head", "neck_2", (0.0, 41.2, 31.2))
    m.bone("jaw", "head", (0.0, 38.4, 32.0))

    # =========================================================================
    # STAGE 1: HEAVY FUR-CLOAKED CHAINBLADE WYVERN BLOCKOUT
    # =========================================================================
    torso_stations = [
        ((0.0, 21.0, -14.5), 11.8, 10.6, "pelvis"),
        ((0.0, 23.0, -4.5), 14.8, 13.0, "pelvis"),
        ((0.0, 26.8, 6.0), 18.2, 15.8, "chest"),
        ((0.0, 30.2, 14.8), 16.5, 14.5, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.8, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 31.5, 16.2), (0.0, 36.8, 23.5), 11.4, 11.0),
        ("neck_2", (0.0, 36.8, 23.5), (0.0, 41.0, 30.8), 9.8, 9.4),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    m.box("skull_cranium", "head", (0.0, 42.0, 34.2), (9.4, 6.2, 8.2), "armor", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("wraith_snout", "head", (0.0, 40.8, 40.0), (5.8, 3.8, 5.8), "scales", rotation=(10.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 38.4, 37.8), (5.5, 2.4, 9.4), "belly", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 21.2, -13.5), 10.2, 9.2),
        ((0.8, 20.5, -22.5), 8.8, 8.0),
        ((2.2, 20.2, -31.5), 7.6, 7.0),
        ((3.4, 20.8, -40.5), 6.2, 6.0),
        ((3.6, 22.2, -49.5), 5.2, 5.0),
        ((2.4, 24.0, -58.2), 4.2, 4.2),
        ((1.0, 25.8, -66.5), 3.4, 3.4),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(6)]
    for i in range(6):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.6, pattern="calm_flank", stage1=(i % 2 == 0))

    for s, q in ((1.0, "l"), (-1.0, "r")):
        p0, p1, p2 = (s * 8.2, 21.0, -6.5), (s * 12.2, 12.0, -1.0), (s * 12.0, 3.6, -8.5)
        bu = m.bone(f"hind_upper_{q}", "pelvis", p0)
        bl = m.bone(f"hind_lower_{q}", bu, p1)
        bp = m.bone(f"hind_paw_{q}", bl, p2)
        m.span_box(f"hind_thigh_{q}", bu, p0, p1, 8.6, 8.2, "fur", len_pad=2.2, pattern="fur_mane", stage1=True)
        m.span_box(f"hind_shin_{q}", bl, p1, p2, 6.0, 5.8, "scales", len_pad=2.2, pattern="leg_bands", stage1=True)
        m.box(f"hind_ankle_{q}", bp, (p2[0], 2.6, p2[2] + 2.2), (7.6, 3.6, 7.4), "scales_deep", stage1=True)

    # 3-Bank Wraith Wing Sails + MASSIVE 3-Link Articulating Chainblade Whips (`chain_1..3_l/r`)!
    for s, q in ((1.0, "l"), (-1.0, "r")):
        wr_pt = (s * 32.0, 44.0, 14.5)
        m.winged_sail_rig(
            s,
            "chest",
            (s * 8.0, 30.2, 8.5),
            (s * 21.0, 40.0, 4.5),
            wr_pt,
            ((s * 60.0, 48.0, 4.0), (s * 54.0, 35.0, -8.5), (s * 42.0, 24.0, -17.5), (s * 26.0, 17.5, -22.0)),
            (s * 7.5, 22.0, -8.0),
            spar_mat="chain_steel",
            hub_mat="armor",
            claw_mat="claw",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_legiana",
            spar_w=3.4,
            arm_w=5.0,
            leading_scales=6,
            tattered_outer=True,
        )
        # Signature Arkveld Dual Twin-Blade Chain Whips (`3` articulated chain bones dangling forward/out from wrist!)
        b_wr = f"wing_wrist_{q}"
        c_pts = [
            np.array(wr_pt) + np.array([0.0, -1.0, 1.5]),
            np.array([s * 28.0, 29.0, 22.0]),
            np.array([s * 24.0, 16.5, 28.5]),
            np.array([s * 18.0, 6.0, 32.5]),
        ]
        cb_names = [
            m.bone(f"chain_1_{q}", b_wr, c_pts[0]),
            m.bone(f"chain_2_{q}", f"chain_1_{q}", c_pts[1]),
            m.bone(f"chain_3_{q}", f"chain_2_{q}", c_pts[2]),
        ]
        for ci in range(3):
            pa, pb = c_pts[ci], c_pts[ci + 1]
            # Heavy wide braided silver-steel & fur chainblade whip planks (`no skinny red-pegged poles!`)
            m.span_box(f"chain_blade_upper_{q}_{ci}", cb_names[ci], pa, pb, 7.2 - 0.7 * ci, 3.2, "chain_steel", seam_color=[204, 72, 92], len_pad=1.0, pattern="armor_shingle", component="wing_arm", stage1=True)
            m.span_box(f"chain_blade_rim_{q}_{ci}", cb_names[ci], pa + np.array([s * 1.4, -0.6, 0.5]), pb + np.array([s * 1.2, -0.5, 0.5]), 4.4 - 0.4 * ci, 2.4, "armor", len_pad=0.8, pattern="armor_shingle", component="wing_arm")
            if stage_cap >= 2:
                mid_c = lerp3(pa, pb, 0.5)
                m.box(f"chain_cog_guard_{q}_{ci}", cb_names[ci], mid_c + np.array([0.0, 1.4, 0.0]), (8.2 - 0.8 * ci, 3.2, 4.8), "fur" if ci == 0 else "armor", rotation=(24.0, s * 14.0, 0.0), pattern="fur_mane" if ci == 0 else "armor_shingle", component="armor")
                # Subtle serrated edge teeth in pale chain steel (matching `arkveld_wide_board.jpg` `[1]`!)
                m.span_box(f"chain_barb_{q}_{ci}", cb_names[ci], mid_c, mid_c + np.array([s * 4.2, -2.0, 1.8]), 2.6, 2.0, "chain_steel", component="spines")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: RAM-HORNED WRAITH SKULL, FUR CAPE CASCADE & ARMOR
    # =========================================================================
    m.box("nasal_crest", "head", (0.0, 42.8, 38.6), (4.6, 2.0, 6.2), "armor", rotation=(12.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("chin_guard", "jaw", (0.0, 37.6, 40.2), (4.8, 2.2, 4.2), "armor", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_plate_{q}", "head", (s * 4.6, 41.2, 34.4), (2.4, 3.6, 6.2), "armor", rotation=(8.0, s * 12.0, -s * 10.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.4, 42.4, 36.6), (1.1, 1.4, 2.5), "mouth", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_wraith_{q}", "head", (s * 4.7, 42.4, 36.8), (0.8, 1.0, 1.9), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        # Iconic Swept Down-and-Back Thick CURVED SILVER RAM HORNS (`arkveld_wide_board.jpg`!)
        m.tapered_arch(
            f"ram_horn_{q}",
            "head",
            [(s * 3.8, 43.8, 32.8), (s * 7.6, 44.5, 27.5), (s * 9.8, 41.2, 23.0), (s * 9.2, 36.5, 22.8)],
            [(3.0, 2.8), (2.6, 2.4), (2.0, 1.9), (1.1, 1.1)],
            "horn",
            tip_color=[216, 218, 228],
            tip_start=0.45,
            component="horns",
            stage1=True,
        )
        for ti, tz in enumerate((35.5, 37.8, 40.0, 41.8)):
            ux = 2.7 - 0.25 * ti
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * ux, 39.2, tz), (0.8, 1.3 if ti == 2 else 1.0, 0.8), "tooth", component="teeth")

        bp = f"hind_paw_{q}"
        for bi in range(4):
            pt_b = lerp3((s * 12.2, 12.0, -1.0), (s * 12.0, 3.6, -8.5), 0.16 + 0.22 * bi) + np.array([s * 1.8, 0.8, 1.5])
            m.box(f"shin_shingle_{q}_{bi}", f"hind_lower_{q}", pt_b, (2.2, 3.2, 4.0), "chain_steel", rotation=(16.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
        for di, ang_d in ((0, -22.0), (1, 0.0), (2, 22.0)):
            yaw = math.radians(s * ang_d)
            dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            ta = np.array([s * (12.0 + (-2.0 + 2.0 * di) * s), 2.4, -3.8])
            tb = ta + dv * 4.2
            tc = tb + dv * 3.4 - np.array([0.0, 1.0, 0.0])
            m.span_box(f"talon_toe_{q}_{di}", bp, ta, tb, 2.0, 2.0, "scales_deep", len_pad=0.4, component="claws")
            m.span_box(f"talon_claw_{q}_{di}", bp, tb, tc, 1.5, 1.6, "claw", len_pad=0.3, component="claws")

    # Signature Billowing Wraith Fur-Feather Cape along Neck & Upper Back (`arkveld_wide_board.jpg`)
    for ci, (bn, cp, clen, cw, count) in enumerate((
        ("neck_2", (0.0, 41.5, 25.5), 11.5, 3.8, 7),
        ("neck_1", (0.0, 37.5, 19.0), 14.8, 4.4, 9),
        ("chest", (0.0, 34.5, 11.5), 16.5, 4.8, 9),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            ang = frac * 72.0
            ra = math.radians(ang)
            b_c = np.array([math.sin(ra) * (4.0 + 1.8 * ci), cp[1] - abs(frac) * 4.2, cp[2]])
            dv = unit([math.sin(ra) * 0.56, math.cos(ra) * 0.28, -0.82])
            t_c = b_c + dv * clen
            m.span_box(f"fur_cape_{ci}_{k}", bn, b_c, t_c, cw, 2.4, "fur" if k % 2 == 0 else "armor", len_pad=0.8, pattern="fur_mane" if k % 2 == 0 else "armor_shingle", component="surface")

    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 35.8, 25.2), 7.2, 5.6, 22.0),
        ("neck_1", (0.0, 31.2, 20.5), 9.0, 6.2, 18.0),
        ("chest", (0.0, 27.0, 16.8), 10.6, 6.8, 22.0),
        ("chest", (0.0, 23.0, 13.5), 11.8, 7.0, 12.0),
        ("chest", (0.0, 20.0, 6.2), 13.0, 7.2, 4.0),
        ("chest", (0.0, 19.2, -1.5), 12.2, 7.0, 0.0),
        ("pelvis", (0.0, 18.2, -9.0), 11.0, 6.6, -2.0),
    )):
        m.box(f"belly_scute_{ki}", kb, kpos, (kw, 2.2, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: WRAITH FLANK FUR CUTOUTS, CARAPACE SHINGLES & PLUMED TAIL
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for nb, p_a, p_b in (
            ("neck_1", (s * 4.8, 32.5, 17.0), (s * 4.2, 36.8, 23.0)),
            ("neck_2", (s * 4.0, 37.4, 24.2), (s * 3.6, 40.5, 30.0)),
        ):
            m.shingle_arc(f"neck_fur_collar_{nb}_{q}", nb, p_a, p_b, (s * 0.36, 0.24, -0.89), (s * 0.28, 0.28, -0.90), count=5, length=5.2, width=2.8, thickness=1.6, material="fur", alt_material="armor", two_step=False, pattern="fur_mane")
        m.box(f"wing_pauldron_cape_{q}", "chest", (s * 8.8, 31.2, 9.2), (5.5, 3.8, 8.2), "armor", rotation=(14.0, s * 16.0, -s * 22.0), pattern="armor_shingle", component="armor")
        for wi in range(4):
            w_pt = lerp3((s * 8.0, 30.2, 8.5), (s * 21.0, 40.0, 4.5), 0.15 + 0.24 * wi)
            m.box(f"wing_humerus_fur_{q}_{wi}", f"wing_{q}", w_pt + np.array([0.0, 2.2, 1.2]), (3.6, 2.4, 4.8), "fur" if wi % 2 == 0 else "chain_steel", rotation=(12.0, s * 18.0, 0.0), pattern="fur_mane", component="armor")
        for thi in range(4):
            th_p = lerp3((s * 8.2, 21.0, -6.5), (s * 12.2, 12.0, -1.0), 0.18 + 0.22 * thi) + np.array([s * 2.8, 0.8, 0.6])
            m.box(f"thigh_fur_armor_{q}_{thi}", f"hind_upper_{q}", th_p, (3.2, 4.0, 4.8), "fur" if thi % 2 == 0 else "armor", rotation=(14.0, s * 14.0, -s * 18.0), pattern="fur_mane", component="surface")
        m.shingle_arc(f"chest_fur_cloak_{q}", "chest", (s * 9.5, 28.0, 12.5), (s * 8.2, 24.8, -2.8), (s * 0.25, 0.14, -0.96), (s * 0.20, 0.12, -0.97), count=6, length=7.0, width=3.4, thickness=2.2, material="fur", alt_material="armor", two_step=False, pattern="fur_mane")
        m.shingle_arc(f"chest_ventral_ribs_{q}", "chest", (s * 6.8, 21.2, 11.5), (s * 6.4, 19.6, -1.5), (s * 0.45, -0.24, -0.86), (s * 0.40, -0.22, -0.88), count=5, length=5.2, width=2.8, thickness=1.6, material="belly", alt_material="scales", two_step=False, pattern="belly_scute")
        m.shingle_arc(f"pelvis_fur_skirt_{q}", "pelvis", (s * 7.6, 22.8, -4.5), (s * 6.0, 21.2, -15.5), (s * 0.22, 0.12, -0.97), (s * 0.18, 0.10, -0.98), count=5, length=6.4, width=3.0, thickness=1.9, material="fur", alt_material="chain_steel", two_step=False, pattern="fur_mane")
        m.cutout_panel(f"flank_fur_fringes_{q}", "pelvis", [v3((s * 8.2, 22.5, -2.0)), v3((s * 6.8, 21.0, -16.5)), v3((s * 6.4, 15.0, -15.5)), v3((s * 7.8, 16.0, -2.0))], "fur", pattern="fur_tuft", component="surface")

    for ti in range(6):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.box(f"tail_fur_ridge_{ti}", tbones[ti], mid_t + np.array([0.0, 3.2 - 0.15 * ti, 0.0]), (max(2.6, 5.8 - 0.45 * ti), 2.2, 6.8), "fur" if ti < 3 else "chain_steel", rotation=(12.0, 0.0, 0.0), pattern="fur_mane" if ti < 3 else "armor_shingle", component="armor")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.span_box(f"tail_feather_barb_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.4, 3.6 - 0.3 * ti), 0.2, 0.0]), mid_t + np.array([s * max(2.6, 6.0 - 0.35 * ti), 1.8, -3.5]), 1.7, 2.1, "chain_steel", component="spines")

    return m.doc
