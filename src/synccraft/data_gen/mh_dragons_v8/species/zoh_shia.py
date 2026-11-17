"""src/synccraft/data_gen/mh_dragons_v8/species/zoh_shia.py

Monumental Sculpted Blockbench Reconstruction of ZOH SHIA (`Forbidden Wylk Construct`, Elder Dragon).

Built directly from `zoh_shia_wide_board.jpg` (`[1],[2],[6],[12]` — MH Wilds Final Boss Construct):
  - Towering heavy-shouldered draconic construct encased in stark pearl-white CHALKY WYLK CRYSTAL
    CATHEDRAL ARMOR plates fracturing over dark Fatalis/Black Dragon obsidian corruption hide beneath!
  - Regal veiled Wylk Seraphic Mask Skull (`haloed sculpted white crown mask + dark obsidian corrupted
    curved dragon horns bursting out + eerie glowing azure-violet Wylk eye-slits`).
  - Signature SERAPHIC MULTI-BANK CRYSTAL-FEATHER WINGS (`layered giant ivory-white angel wing
    feather-blades fanning off dark obsidian/Wylk inner corruption membranes`).
  - Jagged cathedral Wylk-crystal dorsal pylons & cracked chest reactor fissures (`<= 3%` emissive),
    massive armored brawler fore-arms ending in blackened dragon claws, and Wylk-crusted tail!
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def zoh_shia_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [34, 34, 42], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [22, 22, 28], "kind": "scales", "coat": "calm_flank"},
        "wylk_white": {"color": [226, 226, 232], "kind": "armor", "coat": "armor_shingle"},
        "wylk_crystal": {"color": [204, 212, 228], "kind": "crystal", "coat": "crystal_facet"},
        "armor": {"color": [54, 54, 68], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [182, 184, 196], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [32, 30, 42], "kind": "horn", "coat": "horn_keratin"},
        "wylk_feather": {"color": [232, 234, 240], "kind": "crystal", "coat": "crystal_facet"},
        "wing_mem": {"color": [56, 48, 66], "kind": "membrane", "coat": "wing_gore"},
        "wing_back": {"color": [36, 30, 44], "kind": "membrane", "coat": "wing_gore"},
        "claw": {"color": [36, 34, 46], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [126, 42, 74], "kind": "mouth"},
        "tooth": {"color": [238, 238, 242], "kind": "bone"},
        "eye": {"color": [116, 226, 255], "kind": "eye", "emissive": 0.90},
    }


def build_zoh_shia(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "zoh_shia",
        "Zoh Shia",
        "Wylk Seraphic Construct",
        "Elder Dragon",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Zoh_Shia",
        materials=zoh_shia_materials(),
        notes=[
            "Built from zoh_shia_wide_board.jpg: chalk-white Wylk cathedral armor encasing pitch-obsidian Fatalis corruption beneath, angel-halo mask skull, 3-bank Wylk crystal-feather shroud wings, and monumental armored gauntlets.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 22.0, -14.0))
    m.bone("chest", "pelvis", (0.0, 27.5, 5.5))
    m.bone("neck_1", "chest", (0.0, 32.5, 17.5))
    m.bone("neck_2", "neck_1", (0.0, 38.0, 25.0))
    m.bone("head", "neck_2", (0.0, 41.5, 32.0))
    m.bone("jaw", "head", (0.0, 38.5, 32.8))

    # =========================================================================
    # STAGE 1: COLOSSAL SERAPHIC CONSTRUCT BLOCKOUT
    # =========================================================================
    torso_stations = [
        ((0.0, 21.5, -20.5), 13.5, 11.8, "pelvis"),
        ((0.0, 23.5, -10.5), 16.8, 14.5, "pelvis"),
        ((0.0, 27.0, 0.5), 21.5, 18.2, "chest"),
        ((0.0, 29.8, 11.2), 23.5, 20.0, "chest"),
        ((0.0, 31.8, 19.0), 17.8, 16.0, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=4.0, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 31.8, 16.8), (0.0, 37.0, 24.0), 13.8, 13.0),
        ("neck_2", (0.0, 37.0, 24.0), (0.0, 41.4, 31.6), 11.8, 11.2),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    m.box("skull_wylk_mask", "head", (0.0, 42.5, 35.2), (10.8, 6.8, 8.8), "wylk_white", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("construct_muzzle", "head", (0.0, 41.2, 41.5), (6.8, 4.4, 6.2), "wylk_crystal", rotation=(12.0, 0.0, 0.0), pattern="crystal_facet", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 38.4, 39.0), (6.5, 2.8, 10.6), "scales_deep", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 21.5, -20.0), 11.8, 10.8),
        ((1.0, 20.6, -29.0), 10.2, 9.5),
        ((2.4, 20.2, -38.0), 8.6, 8.0),
        ((3.6, 20.8, -47.0), 7.2, 6.8),
        ((3.2, 22.2, -56.0), 5.8, 5.6),
        ((1.8, 24.0, -64.5), 4.6, 4.6),
        ((0.2, 25.8, -72.2), 3.4, 3.6),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(6)]
    for i in range(6):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.8, pattern="calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (11.5, 24.8, 10.0), (16.5, 14.0, 5.5), (15.8, 3.8, 12.2), (9.8, 9.4), (8.2, 8.0)),
        ("hind", "pelvis", (9.2, 21.2, -13.5), (13.0, 11.8, -7.0), (12.6, 3.4, -15.0), (8.8, 8.4), (6.4, 6.2)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0, p1, p2 = (s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.4, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "wylk_white" if kind == "fore" else "scales", len_pad=2.2, pattern="armor_shingle", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.6, p2[2] + 2.8), (8.8 if kind == "fore" else 7.8, 3.8, 8.0), "scales_deep", stage1=True)

    # Signature 3-Bank Seraphic Wylk Angel-Wing Architecture + Massive Wylk Crystal Feather Blades!
    for s, q in ((1.0, "l"), (-1.0, "r")):
        sh_pt, el_pt, wr_pt = (s * 9.8, 34.5, 9.5), (s * 24.5, 48.0, 4.5), (s * 36.5, 56.0, 15.0)
        m.winged_sail_rig(
            s,
            "chest",
            sh_pt,
            el_pt,
            wr_pt,
            ((s * 68.0, 62.0, 4.0), (s * 60.0, 44.0, -9.5), (s * 46.0, 30.0, -19.0), (s * 30.0, 20.0, -23.0)),
            (s * 9.0, 23.0, -9.5),
            spar_mat="wylk_white",
            hub_mat="wylk_crystal",
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
            # Giant White Wylk Cathedral Angel Feather-Blades (`zoh_shia_wide_board.jpg` `[1],[2]`!)
            b_wr = f"wing_wrist_{q}"
            for fi, (ang_y, len_f) in enumerate(((10.0, 18.0), (-2.0, 21.0), (-14.0, 19.0), (-26.0, 16.0))):
                fb = lerp3(el_pt, wr_pt, 0.25 + 0.24 * fi)
                ft = fb + np.array([s * 12.0, ang_y * 0.45, -len_f * 0.65])
                m.span_box(f"wylk_angel_blade_{q}_{fi}", b_wr if fi >= 2 else f"wing_fore_{q}", fb, ft, 3.2, 2.0, "wylk_feather", tip_color=[240, 244, 250], pattern="crystal_facet", component="spines")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: SERAPHIC MASK CROWN, CORRUPTION HORNS & WYLK CATHEDRAL ARMOR
    # =========================================================================
    m.box("mask_forehead_veil", "head", (0.0, 45.2, 38.2), (8.2, 2.4, 8.8), "wylk_white", rotation=(14.0, 0.0, 0.0), pattern="armor_shingle", component="head")
    m.box("mask_beak_ridge", "head", (0.0, 43.0, 43.2), (4.4, 2.2, 5.2), "wylk_crystal", rotation=(20.0, 0.0, 0.0), pattern="crystal_facet", component="snout")
    m.box("wylk_chin_crest", "jaw", (0.0, 37.2, 41.8), (4.6, 2.4, 4.2), "wylk_white", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"mask_cheek_{q}", "head", (s * 4.8, 42.2, 35.8), (2.4, 3.8, 6.5), "wylk_white", rotation=(8.0, s * 14.0, -s * 10.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.6, 43.4, 38.0), (1.1, 1.5, 2.6), "armor", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_wylk_{q}", "head", (s * 4.9, 43.4, 38.2), (0.8, 1.1, 1.9), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        # Corrupted Black Fatalis Devil Horns emerging through Pure White Wylk Halo Crest!
        m.tapered_arch(f"corruption_crown_horn_{q}", "head", [(s * 3.4, 45.8, 34.0), (s * 6.2, 51.2, 28.5), (s * 7.8, 56.8, 22.8)], [(2.8, 2.6), (2.2, 2.0), (1.1, 1.1)], "horn", tip_color=[64, 62, 82], component="horns", stage1=True)
        m.tapered_arch(f"wylk_halo_ear_{q}", "head", [(s * 4.8, 43.8, 33.2), (s * 9.5, 47.2, 27.5), (s * 13.0, 50.2, 21.8)], [(2.8, 3.2), (2.2, 2.6), (1.2, 1.5)], "wylk_white", pattern="armor_shingle", component="horns")

        for ti, tz in enumerate((36.5, 38.8, 41.0, 42.8)):
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * (3.0 - 0.3 * ti), 39.2, tz), (0.8, 1.3, 0.8), "tooth", component="teeth")

        for kind, p0_r, p1_r, p2_r in (
            ("fore", (11.5, 24.8, 10.0), (16.5, 14.0, 5.5), (15.8, 3.8, 12.2)),
            ("hind", (9.2, 21.2, -13.5), (13.0, 11.8, -7.0), (12.6, 3.4, -15.0)),
        ):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0, p1, p2 = np.array([s * p0_r[0], p0_r[1], p0_r[2]]), np.array([s * p1_r[0], p1_r[1], p1_r[2]]), np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_wylk_pauldron_{q}", bu, p0 + np.array([s * 1.2, 2.0, 0.5]), (8.4, 8.0, 9.0), "wylk_white", seam_color=[92, 206, 248], rotation=(8.0, s * 10.0, -s * 14.0), pattern="armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.14 + 0.22 * bi) + np.array([s * 2.4, 1.0, 1.5])
                m.box(f"{kind}_wylk_vambrace_{q}_{bi}", bl, pt_b, (2.6, 3.8, 4.6), "wylk_crystal" if bi % 2 == 0 else "wylk_white", rotation=(18.0, s * 14.0, -s * 20.0), pattern="crystal_facet", component="armor")
            for di, ang_d in ((0, -20.0), (1, -5.0), (2, 10.0), (3, 24.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-2.6 + 1.8 * di), 2.5, p2[2] + 5.0])
                tb = ta + dv * 4.0
                tc = tb + dv * 3.2 - np.array([0.0, 0.9, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 2.2, 2.2, "scales_deep", len_pad=0.4, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.6, 1.8, "claw", len_pad=0.3, component="claws")

    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 35.8, 25.5), 8.0, 5.8, 22.0),
        ("neck_1", (0.0, 30.0, 21.0), 9.6, 6.4, 18.0),
        ("chest", (0.0, 21.0, 13.5), 12.2, 7.6, 12.0),
        ("chest", (0.0, 18.0, 5.2), 13.6, 8.0, 4.0),
        ("pelvis", (0.0, 17.0, -3.5), 12.6, 7.4, 0.0),
        ("pelvis", (0.0, 16.4, -12.0), 11.4, 7.0, -2.0),
    )):
        m.box(f"belly_wylk_scute_{ki}", kb, kpos, (kw, 2.4, kd), "belly", seam_color=[92, 206, 248] if ki in (2, 3) else None, rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: CATHEDRAL CRYSTAL DORSAL SPIRES, FRACTURED WYLK FLANKS & TAIL
    # =========================================================================
    for row_i, (rbone, rz, ry, sep_x, qlen, count) in enumerate((
        ("neck_2", 26.5, 41.2, 3.6, 9.5, 5),
        ("neck_1", 20.5, 39.2, 4.8, 11.2, 5),
        ("chest", 12.8, 39.5, 6.5, 13.5, 5),
        ("chest", 5.2, 38.5, 6.2, 13.0, 5),
        ("pelvis", -3.0, 34.5, 5.5, 11.5, 5),
        ("pelvis", -11.5, 31.0, 4.5, 10.0, 5),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            base_q = np.array([frac * sep_x, ry - abs(frac) * 1.6, rz])
            dir_q = unit([frac * 0.40, 0.62, -0.68])
            tip_q = base_q + dir_q * qlen
            m.span_box(f"wylk_cathedral_spire_{row_i}_{k}", rbone, base_q, tip_q, 2.6, 2.8, "wylk_crystal" if k % 2 == 0 else "wylk_white", tip_color=[240, 246, 255], pattern="crystal_facet", component="spines")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.shingle_arc(f"chest_wylk_armor_{q}", "chest", (s * 11.2, 28.5, 12.5), (s * 9.5, 25.2, -3.5), (s * 0.28, 0.20, -0.94), (s * 0.22, 0.16, -0.96), count=6, length=7.2, width=3.4, thickness=2.2, material="wylk_white", alt_material="armor", seam_color=[92, 206, 248], two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"pelvis_wylk_armor_{q}", "pelvis", (s * 8.4, 23.2, -5.0), (s * 6.5, 21.5, -18.5), (s * 0.24, 0.14, -0.96), (s * 0.18, 0.12, -0.97), count=5, length=6.4, width=3.0, thickness=1.9, material="wylk_crystal", alt_material="armor", two_step=False, pattern="crystal_facet")

    for ti in range(6):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_wylk_spire_{ti}", tbones[ti], mid_t + np.array([0.0, 2.6, 0.0]), mid_t + np.array([0.0, 7.2 - 0.3 * ti, -4.8]), 2.0, 2.2, "wylk_crystal", component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_wylk_plate_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.2, 4.2 - 0.38 * ti), 0.2, 0.0]), (1.8, max(1.5, 2.6 - 0.15 * ti), 6.2), "wylk_white" if ti % 2 == 0 else "armor", rotation=(8.0, -s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")

    return m.doc
