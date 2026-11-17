"""src/synccraft/data_gen/mh_dragons_v8/species/zinogre.py

Monumental Sculpted Blockbench Reconstruction of ZINOGRE (`Lord of Usurpers`, Fanged Wyvern).

Refined directly from `zyhfisme` Zinogre (`e58d992a.jpg`) + Stage 2 visual review:
  - Heavy wolf-drake muzzle with black leather nose bridge, crisp upper/lower fangs, thick storm-teal
    brow ridge, glowing cyan slit eye, and swept cyan cheek blades.
  - Swept-back streamlined cascading sandstone/ivory neck mane slabs (`-Z` streamlined wrap behind the
    skull & across the broad teal back saddle, never standing like a vertical brush) + jagged cream keel.
  - Calm storm-teal torso saddle studded with 2x6 crisp ochre dorsal charge pips.
  - Massive forearm bracer wrap (`wide overlapping diagonal cream/sandstone armor plates + furry tricep
    tufts + tall black claw housings with gleaming cyan lightning talon tips`), rich flank fur cowls,
    and heavy chevron-plated layered club tail (`~440+` elements).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def zinogre_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [34, 96, 108], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [24, 68, 80], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [46, 118, 128], "kind": "armor", "coat": "armor_shingle"},
        "sand_plate": {"color": [194, 172, 128], "kind": "armor", "coat": "armor_shingle"},
        "sand_light": {"color": [220, 206, 172], "kind": "armor", "coat": "armor_shingle"},
        "ochre_stud": {"color": [182, 138, 76], "kind": "horn", "coat": "horn_keratin"},
        "belly": {"color": [214, 198, 162], "kind": "belly", "coat": "belly_scute"},
        "fur": {"color": [228, 222, 204], "kind": "fur", "coat": "fur_mane"},
        "horn": {"color": [204, 184, 136], "kind": "horn", "coat": "horn_keratin"},
        "nose": {"color": [24, 30, 38], "kind": "armor", "coat": "armor_shingle"},
        "claw_base": {"color": [24, 28, 36], "kind": "horn", "coat": "horn_keratin"},
        "claw_tip": {"color": [88, 206, 228], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.16},
        "mouth": {"color": [146, 48, 56], "kind": "mouth"},
        "tooth": {"color": [238, 234, 218], "kind": "bone"},
        "eye": {"color": [110, 240, 255], "kind": "eye", "emissive": 0.88},
    }


def build_zinogre(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "zinogre",
        "Zinogre",
        "Lord of Usurpers",
        "Fanged Wyvern",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Zinogre",
        materials=zinogre_materials(),
        notes=[
            "Sculpted from `zyhfisme` Zinogre (`e58d992a.jpg`) with Stage 2->3 fixes: streamlined swept mane cowl, full-wrap forearm sandstone bracer plates, black-to-cyan tall claw housings, and thick chevron tail.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 20.0, -14.0))
    m.bone("chest", "pelvis", (0.0, 24.5, 5.5))
    m.bone("neck_1", "chest", (0.0, 28.0, 17.0))
    m.bone("neck_2", "neck_1", (0.0, 31.8, 24.0))
    m.bone("head", "neck_2", (0.0, 34.0, 29.8))
    m.bone("jaw", "head", (0.0, 30.8, 30.5))
    cap_l = m.bone("capacitor_l", "chest", (6.5, 29.5, 6.0))
    cap_r = m.bone("capacitor_r", "chest", (-6.5, 29.5, 6.0))

    # =========================================================================
    # STAGE 1: MASSIVE WOLF-DRAKE SILHOUETTE BLOCKOUT
    # =========================================================================
    torso_stations = [
        ((0.0, 19.5, -21.0), 13.5, 11.5, "pelvis"),
        ((0.0, 20.5, -11.0), 15.5, 13.5, "pelvis"),
        ((0.0, 23.0, -0.5), 21.0, 17.5, "chest"),
        ((0.0, 26.2, 10.5), 24.0, 20.0, "chest"),
        ((0.0, 28.5, 18.5), 18.5, 16.5, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=4.0, pattern="calm_flank", stage1=True)

    neck_stations = [
        ((0.0, 27.8, 16.8), 15.0, 14.5, "neck_1"),
        ((0.0, 31.2, 23.2), 12.8, 12.6, "neck_2"),
        ((0.0, 34.0, 29.5), 11.0, 10.5, "head"),
    ]
    for i in range(len(neck_stations) - 1):
        p0, w0, h0, bn = neck_stations[i]
        p1, w1, h1, _ = neck_stations[i + 1]
        m.span_box(f"neck_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    m.box("skull_cranium", "head", (0.0, 34.8, 32.5), (10.5, 6.8, 8.0), "scales", face_materials={"up": "armor"}, rotation=(8.0, 0.0, 0.0), component="head", stage1=True)
    m.box("wolf_muzzle", "head", (0.0, 33.4, 38.2), (6.5, 4.2, 6.0), "scales", rotation=(10.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 30.8, 36.2), (6.2, 2.8, 9.8), "scales_deep", face_materials={"up": "mouth", "down": "fur"}, rotation=(6.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 20.0, -20.0), 12.2, 11.0),
        ((0.5, 20.6, -28.5), 11.2, 10.2),
        ((1.2, 21.8, -37.0), 10.4, 9.8),
        ((1.6, 23.8, -45.5), 9.6, 9.2),
        ((1.2, 26.2, -53.8), 8.8, 8.6),
        ((0.4, 28.4, -61.5), 7.8, 7.6),
        ((-0.5, 29.8, -68.8), 6.2, 6.2),
        ((-1.2, 30.2, -75.0), 4.2, 4.4),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(7)]
    for i in range(7):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.8, pattern="calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (11.5, 23.0, 10.2), (16.2, 13.5, 4.8), (15.5, 3.8, 12.2), (10.5, 10.0), (9.8, 9.4)),
        ("hind", "pelvis", (9.2, 19.8, -13.5), (12.8, 11.2, -7.2), (12.5, 3.5, -15.5), (8.8, 8.4), (6.5, 6.2)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0 = (s * p0_r[0], p0_r[1], p0_r[2])
            p1 = (s * p1_r[0], p1_r[1], p1_r[2])
            p2 = (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.4, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.4, pattern="calm_flank", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.8, p2[2] + 3.2), (10.8 if kind == "fore" else 8.4, 4.5, 9.0), "scales_deep", stage1=True)

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: WOLF-DRAKE HEAD, STREAMLINED MANE FAN, FULL-WRAP BRACER FOREARMS
    # =========================================================================
    m.box("nasal_bridge_ridge", "head", (0.0, 35.8, 37.5), (5.2, 2.4, 6.4), "armor", rotation=(14.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("snout_side_lip_slope", "head", (0.0, 34.4, 37.8), (7.2, 2.8, 5.8), "scales", rotation=(10.0, 0.0, 0.0), component="snout")
    m.box("wolf_nose_tip", "head", (0.0, 34.4, 41.2), (3.6, 2.5, 2.5), "nose", rotation=(16.0, 0.0, 0.0), component="snout")
    m.box("chin_beard_tuft", "jaw", (0.0, 28.8, 35.2), (5.0, 3.4, 7.2), "fur", rotation=(-24.0, 0.0, 0.0), pattern="fur_mane", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_jowl_{q}", "head", (s * 4.8, 33.4, 32.8), (3.2, 4.4, 6.2), "armor", rotation=(6.0, s * 14.0, -s * 8.0), pattern="armor_shingle", component="head")
        m.box(f"brow_shelf_{q}", "head", (s * 4.2, 36.4, 34.5), (2.4, 1.8, 5.2), "armor", rotation=(12.0, s * 10.0, -s * 14.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.6, 35.2, 34.8), (1.2, 1.6, 2.8), "nose", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_cyan_slit_{q}", "head", (s * 5.0, 35.2, 35.0), (0.8, 1.2, 2.1), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        m.tapered_arch(f"cheek_teal_fin_{q}", "head", [(s * 5.2, 34.5, 32.5), (s * 8.2, 36.4, 27.2), (s * 10.2, 38.4, 22.0)], [(1.9, 2.8), (1.4, 2.2), (0.9, 1.4)], "armor", pattern="armor_shingle", component="horns")
        # Dominant Swept Forward-Curving Sandstone Crown Horns (`e58d992a.jpg`)
        m.tapered_arch(f"crown_horn_outer_{q}", "head", [(s * 3.8, 37.8, 31.2), (s * 5.8, 41.4, 35.4), (s * 6.4, 43.2, 41.8)], [(3.2, 2.8), (2.6, 2.2), (1.5, 1.4)], "sand_plate", tip_color=[228, 216, 184], component="horns", stage1=True)
        m.tapered_arch(f"crown_horn_inner_{q}", "head", [(s * 1.8, 38.0, 30.8), (s * 2.5, 42.2, 34.6), (s * 2.6, 44.6, 39.8)], [(2.8, 2.5), (2.2, 1.9), (1.2, 1.2)], "sand_light", tip_color=[234, 224, 198], component="horns")
        m.tapered_arch(f"cheek_mandible_horn_{q}", "jaw", [(s * 3.6, 31.2, 33.8), (s * 5.8, 29.5, 30.2), (s * 7.2, 28.0, 26.8)], [(1.8, 1.8), (1.4, 1.4), (0.9, 0.9)], "sand_plate", component="horns")

        for ti, tz in enumerate((33.2, 35.4, 37.6, 39.4)):
            ux = 3.2 - 0.35 * ti
            m.box(f"upper_fang_{q}_{ti}", "head", (s * ux, 31.6, tz), (0.9, 1.8 if ti == 2 else 1.2, 0.9), "tooth", component="teeth")
            if ti in (1, 3):
                m.box(f"lower_fang_{q}_{ti}", "jaw", (s * (ux - 0.3), 32.2, tz - 0.4), (0.8, 1.3, 0.9), "tooth", component="teeth")
        m.span_box(f"chin_pike_{q}", "jaw", (s * 2.8, 29.5, 34.8), (s * 3.2, 24.5, 31.8), 1.6, 2.0, "sand_light", component="horns")

    # Streamlined Back-Swept Cascading Sandstone & Fur Mane Mantle (`-Z` sleek overlap matching `e58d992a.jpg`!)
    for ri, (rbone, rpos, rlen, rw, rt, count) in enumerate((
        ("head", (0.0, 37.8, 29.2), 13.5, 3.0, 4.4, 7),
        ("neck_2", (0.0, 36.2, 24.2), 16.5, 3.4, 5.0, 9),
        ("neck_1", (0.0, 33.6, 18.8), 18.5, 3.8, 5.4, 9),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            ang_fan = frac * 72.0
            rad_f = math.radians(ang_fan)
            base_pt = np.array([math.sin(rad_f) * (3.8 + 1.8 * ri), rpos[1] - abs(frac) * 4.5, rpos[2]])
            # Strong backwards sweep (-0.82 Z, +0.40 Y) instead of sticking straight up like hedgehog bristles!
            dir_v = unit([math.sin(rad_f) * 0.54, math.cos(rad_f) * 0.36, -0.82])
            mid_pt = base_pt + dir_v * (rlen * 0.56)
            tip_pt = base_pt + dir_v * rlen
            mat_m = "sand_light" if (k % 2 == 0) else "sand_plate"
            m.span_box(f"mane_slab_{ri}_{k}_a", rbone, base_pt, mid_pt, rw, rt, mat_m, len_pad=0.8, pattern="armor_shingle", component="horns")
            m.span_box(f"mane_slab_{ri}_{k}_b", rbone, mid_pt, tip_pt, rw * 0.72, rt * 0.78, "horn", len_pad=0.5, pattern="armor_shingle", tip_color=[234, 224, 198], component="horns")
            if k in (0, count - 1):
                # Lower cheek/neck fluffy ivory ruff under armor slabs (`cutout_panel`!)
                m.span_box(f"neck_fur_lock_{ri}_{k}", rbone, base_pt + np.array([0.0, -3.5, 0.5]), base_pt + np.array([math.copysign(3.5, frac), -7.0, -4.5]), 2.6, 3.0, "fur", pattern="fur_mane", component="surface")

    # Deep Angular Cream/Sandstone Chest Sternum Keel
    for ki, (kb, kpos, kw, kh, kd, rx, kmat) in enumerate((
        ("neck_2", (0.0, 27.0, 26.8), 8.2, 4.2, 6.2, 22.0, "fur"),
        ("neck_1", (0.0, 22.8, 23.2), 10.0, 5.0, 7.2, 18.0, "sand_light"),
        ("chest", (0.0, 18.5, 19.2), 11.8, 5.6, 8.4, 12.0, "sand_light"),
        ("chest", (0.0, 14.5, 13.8), 12.4, 5.0, 8.8, 8.0, "sand_plate"),
        ("chest", (0.0, 12.8, 6.2), 13.0, 3.6, 8.4, 2.0, "belly"),
        ("pelvis", (0.0, 12.8, -5.5), 11.8, 3.2, 9.0, 0.0, "belly"),
        ("pelvis", (0.0, 13.2, -15.0), 10.0, 3.0, 8.5, -4.0, "belly"),
    )):
        m.box(f"chest_keel_plate_{ki}", kb, kpos, (kw, kh, kd), kmat, rotation=(rx, 0.0, 0.0), pattern="fur_mane" if kmat == "fur" else "belly_scute", component="armor")

    # Sculpted Shoulder Pylons (`capacitor_l/r`) + Calm Teal Saddle + 2x6 Ochre Studs
    for s, q in ((1.0, "l"), (-1.0, "r")):
        cb = cap_l if q == "l" else cap_r
        for pi, (pz, py, ph) in enumerate(((12.5, 34.5, 11.5), (6.5, 35.0, 12.5), (0.0, 33.8, 10.2))):
            b_p = np.array([s * 8.0, py - 2.0, pz])
            m_p = b_p + np.array([s * 2.2, ph * 0.55, -2.5])
            t_p = b_p + np.array([s * 4.0, ph, -5.2])
            m.span_box(f"shoulder_pylon_{q}_{pi}_a", cb, b_p, m_p, 3.6, 4.2, "sand_plate" if pi != 1 else "sand_light", seam_color=[94, 224, 248], len_pad=0.7, pattern="armor_shingle", component="spines")
            m.span_box(f"shoulder_pylon_{q}_{pi}_b", cb, m_p, t_p, 2.5, 2.8, "horn", tip_color=[234, 224, 198], len_pad=0.5, pattern="armor_shingle", component="spines")

        for di, dz in enumerate((11.5, 8.0, 4.5, 1.0, -2.8, -6.5)):
            yb = 34.8 - 0.55 * di
            m.box(f"dorsal_pip_{q}_{di}", "chest" if di < 4 else "pelvis", (s * (5.2 - 0.25 * di), yb, dz), (2.6, 3.0, 2.8), "ochre_stud", rotation=(12.0, s * 10.0, -s * 18.0), pattern="horn_keratin", component="spines")

    # Massive FULL-WRAP Sandstone Foreleg Bracer Armor + Tall Black/Cyan Wolf Claws (`e58d992a.jpg`!)
    for kind, p0_r, p1_r, p2_r in (
        ("fore", (11.5, 23.0, 10.2), (16.2, 13.5, 4.8), (15.5, 3.8, 12.2)),
        ("hind", (9.2, 19.8, -13.5), (12.8, 11.2, -7.2), (12.5, 3.5, -15.5)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0 = np.array([s * p0_r[0], p0_r[1], p0_r[2]])
            p1 = np.array([s * p1_r[0], p1_r[1], p1_r[2]])
            p2 = np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_muscular_shoulder_{q}", bu, p0 + np.array([s * 1.0, 2.0, 0.5]), (9.4 if kind == "fore" else 7.4, 8.4, 10.0), "scales", face_materials={"up": "armor"}, rotation=(8.0, s * 10.0, -s * 14.0), pattern="calm_flank")
            # Broad outer + front wrap-around sandstone diagonal bracers completely sheeting forearms!
            n_br = 5 if kind == "fore" else 4
            for bi in range(n_br):
                f_pt = lerp3(p1, p2, 0.10 + 0.20 * bi) + np.array([s * (2.6 if kind == "fore" else 1.8), 1.4, 2.4])
                bw_w = 6.2 if kind == "fore" else 4.2
                m.box(f"{kind}_front_bracer_{q}_{bi}", bl, f_pt, (bw_w, 4.4 if kind == "fore" else 3.2, 5.8 if kind == "fore" else 4.2), "sand_plate" if bi % 2 == 0 else "sand_light", seam_color=[94, 224, 248], rotation=(28.0, s * 22.0, -s * 18.0), pattern="armor_shingle", component="armor")
                m.box(f"{kind}_outer_bracer_{q}_{bi}", bl, f_pt + np.array([s * 2.2, -0.5, -2.0]), (2.6, 4.2 if kind == "fore" else 3.0, 5.4), "sand_light" if bi % 2 == 0 else "sand_plate", rotation=(16.0, s * 12.0, -s * 28.0), pattern="armor_shingle", component="armor")

            if kind == "fore":
                # Rear elbow twin sandstone blade spurs & tricep fur lock
                m.span_box(f"elbow_blade_main_{q}", bl, p1 + np.array([s * 2.5, 1.0, -1.8]), p1 + np.array([s * 5.6, 8.5, -7.2]), 2.8, 3.8, "sand_light", tip_color=[234, 224, 198], pattern="armor_shingle", component="spines")
                m.span_box(f"elbow_blade_sub_{q}", bl, p1 + np.array([s * 3.0, -0.5, -1.2]), p1 + np.array([s * 5.8, 4.2, -5.8]), 2.2, 2.8, "sand_plate", component="spines")
                m.box(f"tricep_fur_cowl_{q}", bu, lerp3(p0, p1, 0.65) + np.array([s * 1.8, -0.5, -3.5]), (5.2, 5.5, 4.8), "fur", rotation=(-24.0, 0.0, -s * 14.0), pattern="fur_mane", component="surface")
            else:
                m.span_box(f"hock_spur_{q}", bl, p1 + np.array([s * 1.5, 0.5, 1.5]), p1 + np.array([s * 2.2, 3.8, 5.0]), 2.2, 2.8, "sand_plate", component="spines")

            # Sharp Black-to-Cyan Claw Pillars (dark housing over glowing cyan chisel talons — matching `e58d992a.jpg`!)
            digits = ((0, -20.0, 5.4), (1, -6.0, 6.4), (2, 8.0, 6.2), (3, 22.0, 5.0)) if kind == "fore" else ((0, -16.0, 4.6), (1, 0.0, 5.4), (2, 16.0, 4.6))
            for di, ang_d, l_d in digits:
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                kn_x = s * (-3.2 + 2.1 * di) if kind == "fore" else s * (-2.0 + 2.0 * di)
                ka = np.array([p2[0] + kn_x, 3.0, p2[2] + 5.4])
                kb = ka + dv * (l_d * 0.56)
                kb[1] = 2.6
                kc = kb + dv * (l_d * 0.44)
                kc[1] = 1.6
                tw = 2.8 if kind == "fore" else 2.2
                th = 3.8 if kind == "fore" else 2.8
                m.span_box(f"{kind}_knuckle_hood_{q}_{di}", bp, ka, kb, tw, th, "claw_base", len_pad=0.6, pattern="armor_shingle", component="claws")
                m.span_box(f"{kind}_cyan_claw_{q}_{di}", bp, kb, kc, tw * 0.72, th * 0.85, "claw_tip", len_pad=0.4, tip_color=[138, 246, 255], tip_start=0.35, pattern="crystal_facet", component="claws")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: SCULPTED FLANK CARAPACE BANKS, FUR CUTOUTS & BROAD CLUB TAIL
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        # Upper chest/rib sandstone shoulder cowl banks breaking flat teal wall!
        m.shingle_arc(f"upper_saddle_ridge_{q}", "chest", (s * 11.4, 29.5, 13.0), (s * 9.5, 25.0, -4.0), (s * 0.28, 0.42, -0.86), (s * 0.24, 0.38, -0.88), count=7, length=7.6, width=3.8, thickness=2.4, material="sand_plate", alt_material="sand_light", two_step=False, pattern="armor_shingle")
        # Flank teal armor plates breaking flat vertical chest/ribs (`x=11.4..12.4`)
        m.shingle_arc(f"chest_flank_armor_{q}", "chest", (s * 11.8, 22.8, 12.5), (s * 10.8, 20.5, -2.5), (s * 0.26, -0.10, -0.96), (s * 0.20, -0.10, -0.97), count=5, length=6.8, width=3.6, thickness=2.2, material="armor", alt_material="scales_deep", two_step=False, pattern="armor_shingle")
        # Abdominal white fur tufts (`3D` + cutout silhouette)
        m.shingle_arc(f"flank_fur_locks_{q}", "pelvis", (s * 8.6, 17.5, -4.5), (s * 6.8, 16.5, -18.5), (s * 0.35, -0.25, -0.90), (s * 0.28, -0.22, -0.92), count=5, length=6.2, width=3.0, thickness=2.0, material="fur", two_step=False, pattern="fur_mane")
        m.cutout_panel(f"flank_fur_fringes_{q}", "pelvis", [v3((s * 8.5, 19.5, -3.0)), v3((s * 7.2, 18.0, -19.0)), v3((s * 6.8, 12.5, -18.0)), v3((s * 8.2, 13.5, -3.0))], "fur", pattern="fur_tuft", component="surface")

    # Heavy Broad Layered Tail Shingles & Serpentine Fur (`zero little pyramids!`)
    for ti in range(7):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.box(f"tail_top_shield_{ti}", tbones[ti], mid_t + np.array([0.0, 5.2 - 0.32 * ti, 0.0]), (max(3.6, 7.6 - 0.45 * ti), 3.2 - 0.16 * ti, 8.6), "sand_light" if ti % 2 == 0 else "sand_plate", rotation=(-16.0, 0.0, 0.0), pattern="armor_shingle", component="spines")
        m.box(f"tail_fur_under_{ti}", tbones[ti], mid_t + np.array([0.0, -4.6 + 0.28 * ti, 0.0]), (max(2.8, 8.2 - 0.55 * ti), 3.0, 8.4), "fur", rotation=(-8.0, 0.0, 0.0), pattern="fur_mane", component="surface")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_chevron_upper_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(2.2, 6.0 - 0.45 * ti), 1.8 - 0.1 * ti, 0.0]), (3.0, 3.8 - 0.18 * ti, 8.0), "sand_plate", rotation=(14.0, -s * 16.0, -s * 24.0), seam_color=[94, 224, 248] if ti >= 3 else None, pattern="armor_shingle", component="armor")
            m.box(f"tail_chevron_lower_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(2.0, 5.4 - 0.42 * ti), -1.8 + 0.1 * ti, 0.0]), (2.8, 3.2 - 0.16 * ti, 7.6), "sand_light", rotation=(-14.0, -s * 14.0, s * 22.0), pattern="armor_shingle", component="armor")
            if ti in (4, 6):
                m.span_box(f"tail_tip_prong_{ti}_{q}", tbones[ti], mid_t + np.array([s * 2.2, -0.4, -2.0]), mid_t + np.array([s * 4.4, 3.2, -9.2]), 2.4, 3.0, "sand_light", tip_color=[234, 224, 198], component="spines")

    return m.doc
