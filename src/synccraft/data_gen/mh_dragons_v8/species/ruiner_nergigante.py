"""src/synccraft/data_gen/mh_dragons_v8/species/ruiner_nergigante.py

Monumental Sculpted Blockbench Reconstruction of RUINER NERGIGANTE (`Devourer of Elders`, Elder Dragon).

Built directly from `ruiner_nergigante_wide_board.jpg` (`[1],[2],[3],[13]`):
  - Wide bulldog predator cranium & ultra-short flat snarling snout (zero tapered lizard muzzle!)
    plus asymmetric regrown iron bull horns curling wide & forward (`left` jagged regrown metallic spire).
  - Heavy hunched gorilla-like muscular shoulders & oversized forelegs armed with dense clusters
    of iron-and-ivory regrown porcupine spike cones (`crown`, `shoulders`, `forearms`, `back`, `tail`).
  - Towering 3-bank dark tattered dragon wings studded along the entire upper leading wing-arm
    with multi-angle iron spike bursts (`>= 6` wing banks).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def ruiner_nergigante_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [36, 34, 42], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [24, 22, 28], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [54, 48, 56], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [124, 68, 62], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [68, 66, 76], "kind": "horn", "coat": "horn_keratin"},
        "iron_quill": {"color": [152, 154, 166], "kind": "horn", "coat": "horn_keratin"},
        "bone_quill": {"color": [196, 186, 168], "kind": "horn", "coat": "horn_keratin"},
        "wing_mem": {"color": [68, 42, 46], "kind": "membrane", "coat": "wing_gore"},
        "wing_back": {"color": [44, 26, 32], "kind": "membrane", "coat": "wing_gore"},
        "claw": {"color": [42, 40, 48], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [148, 42, 54], "kind": "mouth"},
        "tooth": {"color": [236, 230, 214], "kind": "bone"},
        "eye": {"color": [255, 68, 48], "kind": "eye", "emissive": 0.90},
    }


def build_ruiner_nergigante(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "ruiner_nergigante",
        "Ruiner Nergigante",
        "Immortal Destroyer of Elders",
        "Elder Dragon",
        variant_kind="variant",
        wiki_url="https://monsterhunter.fandom.com/wiki/Ruiner_Nergigante",
        materials=ruiner_nergigante_materials(),
        notes=[
            "Built from ruiner_nergigante_wide_board.jpg: short bulldog muzzle, asymmetric curved bull horns, iron-quill cascades over shoulders/back/wing-arms, and brutal spiked forelimbs.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 21.0, -14.0))
    m.bone("chest", "pelvis", (0.0, 25.5, 5.5))
    m.bone("neck_1", "chest", (0.0, 29.0, 17.0))
    m.bone("neck_2", "neck_1", (0.0, 32.0, 23.8))
    m.bone("head", "neck_2", (0.0, 33.8, 29.2))
    m.bone("jaw", "head", (0.0, 30.2, 29.8))

    # =========================================================================
    # STAGE 1: HEAVY BRAWLER ELDER MASSES (`gorilla chest + broad bulldog skull`)
    # =========================================================================
    torso_stations = [
        ((0.0, 20.5, -20.5), 14.0, 12.2, "pelvis"),
        ((0.0, 22.0, -10.5), 17.5, 15.0, "pelvis"),
        ((0.0, 25.0, 0.0), 22.5, 19.0, "chest"),
        ((0.0, 27.8, 10.8), 25.5, 21.5, "chest"),
        ((0.0, 29.6, 18.8), 19.5, 17.5, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=4.0, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 28.8, 16.5), (0.0, 31.8, 23.0), 16.0, 15.0),
        ("neck_2", (0.0, 31.8, 23.0), (0.0, 33.8, 29.0), 13.8, 13.0),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    # Broad Short Bulldog Cranium & Snarling Jaw (`short stubby muzzle!`)
    m.box("skull_cranium", "head", (0.0, 35.0, 32.2), (12.4, 7.8, 8.6), "armor", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("bulldog_muzzle", "head", (0.0, 33.6, 37.6), (8.2, 4.8, 4.6), "scales_deep", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 30.4, 35.4), (8.0, 3.0, 8.8), "scales_deep", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 20.5, -20.0), 12.2, 11.2),
        ((0.8, 19.6, -29.0), 10.6, 9.8),
        ((2.2, 19.4, -38.0), 9.0, 8.4),
        ((3.2, 20.0, -47.0), 7.6, 7.2),
        ((2.6, 21.4, -55.8), 6.2, 6.2),
        ((1.2, 23.2, -64.0), 5.0, 5.2),
        ((-0.2, 24.8, -71.2), 4.0, 4.2),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(6)]
    for i in range(6):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.8, pattern="calm_flank", stage1=(i % 2 == 0))

    # Huge Brawler Forelegs + Heavy Hindlegs
    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (12.2, 23.5, 10.2), (17.2, 13.5, 5.2), (16.5, 3.8, 12.5), (10.8, 10.2), (9.2, 8.8)),
        ("hind", "pelvis", (9.6, 20.5, -13.5), (13.4, 11.5, -7.0), (13.0, 3.5, -15.0), (9.0, 8.6), (6.6, 6.4)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0, p1, p2 = (s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.4, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.4, pattern="calm_flank", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.8, p2[2] + 3.0), (9.8 if kind == "fore" else 8.2, 4.2, 8.4), "scales_deep", stage1=True)

    # Massive 3-Bank Spiked Elder Dragon Wings (`iron quill bristles lining wing leading arms!`)
    for s, q in ((1.0, "l"), (-1.0, "r")):
        sh_pt, el_pt, wr_pt = (s * 10.5, 34.5, 8.5), (s * 24.0, 48.0, 2.5), (s * 35.0, 56.0, 14.0)
        m.winged_sail_rig(
            s,
            "chest",
            sh_pt,
            el_pt,
            wr_pt,
            ((s * 66.0, 62.0, 4.0), (s * 58.0, 44.0, -9.5), (s * 44.0, 30.0, -19.0), (s * 28.0, 20.0, -23.0)),
            (s * 9.5, 24.0, -10.0),
            spar_mat="armor",
            hub_mat="iron_quill",
            claw_mat="claw",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_gore",
            spar_w=3.8,
            arm_w=5.4,
            leading_scales=7,
            tattered_outer=True,
        )
        if stage_cap >= 2:
            # Dense Metallic Iron Quills bristling off Humerus & Wing Forearm!
            b_wing = f"wing_fore_{q}"
            for qi in range(5):
                qp = lerp3(el_pt, wr_pt, 0.15 + 0.20 * qi)
                q_tip = qp + np.array([s * 2.5, 8.5, -5.5])
                m.span_box(f"wing_iron_quill_{q}_{qi}", b_wing, qp, q_tip, 2.2, 2.2, "iron_quill", tip_color=[208, 210, 222], component="spines")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: ASYMMETRIC BULL HORNS, SNARLING BULLDOG JAW & IRON QUILL BANKS
    # =========================================================================
    m.box("snout_boss_ridge", "head", (0.0, 35.8, 36.2), (6.5, 2.5, 4.8), "armor", rotation=(16.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("chin_spike_guard", "jaw", (0.0, 29.0, 36.8), (6.2, 2.4, 4.2), "armor", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"jowl_armor_{q}", "head", (s * 5.6, 33.8, 32.5), (2.8, 4.4, 6.2), "armor", rotation=(8.0, s * 14.0, -s * 8.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 5.2, 35.5, 34.8), (1.2, 1.6, 2.6), "mouth", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_feral_{q}", "head", (s * 5.6, 35.5, 35.0), (0.8, 1.1, 2.0), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        # Iconic Ruiner Massive Curving Forward Bull Horns (LEFT slightly chipped/asymmetric regrown!)
        h_tip_y = 43.2 if q == "r" else 41.2
        h_tip_z = 42.5 if q == "r" else 40.0
        m.tapered_arch(
            f"bull_horn_{q}",
            "head",
            [
                (s * 4.5, 37.8, 32.2),
                (s * 10.5, 40.5, 30.5),
                (s * 13.8, 42.2, 35.5),
                (s * 11.2, h_tip_y, h_tip_z),
            ],
            [(4.0, 3.8), (3.8, 3.6), (2.8, 2.8), (1.5, 1.5)],
            "horn",
            tip_color=[176, 178, 190] if q == "r" else [206, 208, 220],
            tip_start=0.45,
            component="horns",
            stage1=True,
        )
        # Crown hedgehog spike fan behind skull
        for hk in (-1, 0, 1):
            m.span_box(f"head_quill_fan_{q}_{hk}", "head", (s * (2.5 + 1.4 * abs(hk)), 37.5, 30.0), (s * (4.5 + 2.5 * hk), 44.5, 22.5), 2.2, 2.2, "iron_quill", component="spines")

        for ti, tz in enumerate((33.5, 35.5, 37.5, 39.2)):
            ux = 3.8 - 0.35 * ti
            m.box(f"upper_fang_{q}_{ti}", "head", (s * ux, 31.4, tz), (0.9, 1.8 if ti == 2 else 1.2, 0.9), "tooth", component="teeth")
            if ti % 2 == 1:
                m.box(f"lower_fang_{q}_{ti}", "jaw", (s * (ux - 0.3), 31.8, tz - 0.4), (0.8, 1.3, 0.9), "tooth", component="teeth")

        # Massive Quill-Studded Foreleg & Hindleg Armor + Heavy Claws
        for kind, p0_r, p1_r, p2_r in (
            ("fore", (12.2, 23.5, 10.2), (17.2, 13.5, 5.2), (16.5, 3.8, 12.5)),
            ("hind", (9.6, 20.5, -13.5), (13.4, 11.5, -7.0), (13.0, 3.5, -15.0)),
        ):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0, p1, p2 = np.array([s * p0_r[0], p0_r[1], p0_r[2]]), np.array([s * p1_r[0], p1_r[1], p1_r[2]]), np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_shoulder_hump_{q}", bu, p0 + np.array([s * 1.2, 2.0, 0.5]), (8.4, 8.2, 9.2), "armor", rotation=(8.0, s * 10.0, -s * 14.0), pattern="armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.14 + 0.22 * bi) + np.array([s * 2.5, 1.0, 1.5])
                m.span_box(f"{kind}_arm_quill_{q}_{bi}", bl, pt_b, pt_b + np.array([s * 4.2, 4.5, -4.5]), 2.2, 2.4, "iron_quill" if bi % 2 == 0 else "bone_quill", tip_color=[208, 210, 222], component="spines")
            for di, ang_d in ((0, -20.0), (1, -5.0), (2, 10.0), (3, 24.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-2.8 + 1.8 * di), 2.6, p2[2] + 5.2])
                tb = ta + dv * 4.2
                tc = tb + dv * 3.4 - np.array([0.0, 1.0, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 2.4, 2.6, "scales_deep", len_pad=0.5, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.8, 2.0, "claw", len_pad=0.4, component="claws")

    # Crimson-Brown Belly Plates
    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 27.2, 26.2), 8.2, 5.8, 22.0),
        ("neck_1", (0.0, 23.2, 22.0), 10.0, 6.4, 18.0),
        ("chest", (0.0, 18.8, 13.8), 12.6, 7.6, 12.0),
        ("chest", (0.0, 15.8, 5.5), 14.0, 8.0, 4.0),
        ("chest", (0.0, 14.8, -2.5), 13.2, 7.6, 0.0),
        ("pelvis", (0.0, 14.2, -10.5), 12.0, 7.2, -2.0),
    )):
        m.box(f"belly_scute_{ki}", kb, kpos, (kw, 2.4, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: MULTI-TIER IRON & BONE PORCUPINE QUILL CASCADES (`neck/shoulders/back/tail`)
    # =========================================================================
    # Massive Shoulder & Dorsal Spike Bursts (`the unmistakable Ruiner Nergigante silhouette!`)
    for row_i, (rbone, rz, ry, sep_x, qlen, count) in enumerate((
        ("neck_2", 26.5, 36.2, 3.8, 10.5, 5),
        ("neck_1", 20.5, 35.0, 5.2, 12.0, 7),
        ("chest", 12.8, 37.2, 7.5, 14.0, 7),
        ("chest", 5.2, 36.5, 7.2, 13.5, 7),
        ("chest", -2.5, 34.2, 6.2, 12.0, 5),
        ("pelvis", -10.5, 30.8, 5.2, 11.0, 5),
        ("pelvis", -18.0, 28.0, 4.2, 9.8, 5),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            bx = frac * sep_x
            base_q = np.array([bx, ry - abs(frac) * 1.8, rz])
            dir_q = unit([frac * 0.45, 0.58, -0.68])
            mid_q = base_q + dir_q * (qlen * 0.55)
            tip_q = base_q + dir_q * qlen
            qmat = "iron_quill" if (k + row_i) % 2 == 0 else "bone_quill"
            m.span_box(f"quill_bank_{row_i}_{k}_a", rbone, base_q, mid_q, 2.4, 2.4, "horn", len_pad=0.6, component="spines")
            m.span_box(f"quill_bank_{row_i}_{k}_b", rbone, mid_q, tip_q, 1.5, 1.5, qmat, tip_color=[210, 212, 224], len_pad=0.4, component="spines")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.shingle_arc(f"chest_quill_flank_{q}", "chest", (s * 12.0, 27.0, 12.5), (s * 10.2, 24.5, -3.5), (s * 0.35, 0.25, -0.90), (s * 0.28, 0.20, -0.93), count=6, length=7.4, width=3.2, thickness=2.2, material="iron_quill", alt_material="armor", two_step=False, pattern="armor_shingle")

    for ti in range(6):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_quill_top_{ti}", tbones[ti], mid_t + np.array([0.0, 2.8, 0.0]), mid_t + np.array([0.0, 7.8 - 0.3 * ti, -5.2]), 2.0, 2.2, "iron_quill", tip_color=[210, 212, 224], component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.span_box(f"tail_quill_side_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.6, 4.4 - 0.4 * ti), 0.2, 0.0]), mid_t + np.array([s * max(3.6, 8.2 - 0.4 * ti), 2.8, -4.8]), 1.8, 2.0, "bone_quill" if ti % 2 == 0 else "iron_quill", component="spines")

    return m.doc
