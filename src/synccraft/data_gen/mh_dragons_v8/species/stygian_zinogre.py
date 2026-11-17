"""src/synccraft/data_gen/mh_dragons_v8/species/stygian_zinogre.py

Monumental Sculpted Blockbench Reconstruction of STYGIAN ZINOGRE (`Emperor of Hell`, Fanged Wyvern).

Built distinctly from `stygian_zinogre_wide_board.jpg` (`[1],[2],[4],[5],[8],[9],[10]`):
  - Backward-swept razor obsidian crown horns (`vs` Zinogre's forward crests) & red eyes.
  - Billowing snow-white neck/shoulder/flank fur mane locks contrasting obsidian carapace plates.
  - Full-wrap obsidian foreleg bracer shields veined subtly in crimson dracophage fissures, tall black
    knuckle housings with blood-crimson claw blades, and heavy broad obsidian+white-fur club tail (`~440+`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def stygian_zinogre_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [34, 34, 42], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [22, 22, 28], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [48, 48, 58], "kind": "armor", "coat": "armor_shingle"},
        "obsidian": {"color": [28, 28, 36], "kind": "armor", "coat": "armor_shingle"},
        "fur": {"color": [232, 232, 236], "kind": "fur", "coat": "fur_mane"},
        "belly": {"color": [196, 192, 194], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [42, 42, 52], "kind": "horn", "coat": "horn_keratin"},
        "crimson_stud": {"color": [206, 44, 52], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.22},
        "claw_base": {"color": [24, 24, 32], "kind": "horn", "coat": "horn_keratin"},
        "claw_tip": {"color": [222, 42, 48], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.24},
        "mouth": {"color": [152, 44, 58], "kind": "mouth"},
        "tooth": {"color": [238, 234, 226], "kind": "bone"},
        "eye": {"color": [255, 52, 56], "kind": "eye", "emissive": 0.90},
    }


def build_stygian_zinogre(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "stygian_zinogre",
        "Stygian Zinogre",
        "Emperor of Hell",
        "Fanged Wyvern",
        variant_kind="subspecies",
        parent_species_id="zinogre",
        wiki_url="https://monsterhunter.fandom.com/wiki/Stygian_Zinogre",
        materials=stygian_zinogre_materials(),
        notes=[
            "Built from stygian_zinogre_wide_board.jpg: backward-swept obsidian horns, billowing snow-white shoulder mane fur, full-wrap obsidian bracer shields, and crimson dragon-lightning talon tips.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 20.5, -14.0))
    m.bone("chest", "pelvis", (0.0, 25.0, 5.5))
    m.bone("neck_1", "chest", (0.0, 28.5, 17.5))
    m.bone("neck_2", "neck_1", (0.0, 32.5, 24.5))
    m.bone("head", "neck_2", (0.0, 34.8, 30.2))
    m.bone("jaw", "head", (0.0, 31.6, 30.8))
    cap_l = m.bone("capacitor_l", "chest", (6.5, 30.0, 6.0))
    cap_r = m.bone("capacitor_r", "chest", (-6.5, 30.0, 6.0))

    # =========================================================================
    # STAGE 1: HEAVY PROWLING HELLHOUND WYVERN MASSES
    # =========================================================================
    torso_stations = [
        ((0.0, 20.0, -21.0), 13.5, 11.5, "pelvis"),
        ((0.0, 21.0, -11.0), 15.8, 13.8, "pelvis"),
        ((0.0, 23.5, -0.5), 21.5, 17.8, "chest"),
        ((0.0, 26.8, 10.5), 24.5, 20.5, "chest"),
        ((0.0, 29.0, 18.8), 18.5, 16.8, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=4.0, pattern="calm_flank", stage1=True)

    neck_stations = [
        ((0.0, 28.2, 17.0), 15.2, 14.5, "neck_1"),
        ((0.0, 31.8, 23.8), 13.0, 12.8, "neck_2"),
        ((0.0, 34.8, 29.8), 11.2, 10.8, "head"),
    ]
    for i in range(len(neck_stations) - 1):
        p0, w0, h0, bn = neck_stations[i]
        p1, w1, h1, _ = neck_stations[i + 1]
        m.span_box(f"neck_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    m.box("skull_cranium", "head", (0.0, 35.6, 33.0), (10.5, 6.8, 8.2), "obsidian", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("wolf_muzzle", "head", (0.0, 34.2, 38.6), (6.4, 4.2, 6.2), "scales_deep", rotation=(10.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 31.6, 36.6), (6.0, 2.8, 9.8), "scales_deep", face_materials={"up": "mouth", "down": "fur"}, rotation=(6.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 20.5, -20.0), 12.5, 11.5),
        ((0.5, 21.2, -28.8), 11.8, 10.8),
        ((1.2, 22.8, -37.6), 11.2, 10.4),
        ((1.6, 25.0, -46.2), 10.5, 10.0),
        ((1.2, 27.6, -54.6), 9.6, 9.4),
        ((0.4, 29.8, -62.5), 8.5, 8.4),
        ((-0.5, 31.2, -69.8), 6.8, 6.8),
        ((-1.2, 31.8, -76.2), 4.6, 4.8),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(7)]
    for i in range(7):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.8, pattern="calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (11.8, 23.5, 10.5), (16.6, 13.8, 4.8), (15.8, 3.8, 12.2), (10.8, 10.2), (9.8, 9.4)),
        ("hind", "pelvis", (9.2, 20.2, -13.5), (13.0, 11.5, -7.2), (12.6, 3.5, -15.5), (8.8, 8.4), (6.5, 6.2)),
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
    # STAGE 2: STYGIAN OBSIDIAN BACK-SWEPT HORNS, SNOW-WHITE MANE & CRIMSON CLAWS
    # =========================================================================
    m.box("nasal_bridge_ridge", "head", (0.0, 36.6, 37.8), (5.0, 2.4, 6.4), "obsidian", rotation=(14.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("wolf_nose_tip", "head", (0.0, 35.2, 41.6), (3.4, 2.5, 2.5), "scales_deep", rotation=(16.0, 0.0, 0.0), component="snout")
    m.box("chin_beard_fur", "jaw", (0.0, 29.4, 35.6), (5.2, 3.6, 7.2), "fur", rotation=(-24.0, 0.0, 0.0), pattern="fur_mane", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_jowl_{q}", "head", (s * 4.8, 34.2, 33.0), (3.2, 4.4, 6.2), "armor", rotation=(6.0, s * 14.0, -s * 8.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.6, 36.0, 35.2), (1.2, 1.6, 2.8), "scales_deep", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_crimson_slit_{q}", "head", (s * 5.0, 36.0, 35.4), (0.8, 1.2, 2.1), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")
        m.tapered_arch(f"cheek_razor_fin_{q}", "head", [(s * 5.2, 35.2, 32.8), (s * 8.6, 37.6, 26.8), (s * 10.8, 40.0, 21.2)], [(1.9, 2.8), (1.4, 2.2), (0.9, 1.4)], "obsidian", tip_color=[216, 44, 52], component="horns")
        # Signature BACKWARD-SWEPT towering obsidian demon crown horns!
        m.tapered_arch(f"stygian_crown_horn_{q}", "head", [(s * 3.2, 38.6, 31.8), (s * 5.4, 43.8, 26.5), (s * 6.6, 49.0, 21.0), (s * 7.0, 53.8, 16.0)], [(3.2, 2.8), (2.4, 2.1), (1.5, 1.4), (0.8, 0.8)], "horn", tip_color=[176, 36, 44], tip_start=0.68, component="horns", stage1=True)
        m.tapered_arch(f"brow_horn_flare_{q}", "head", [(s * 4.5, 37.4, 35.0), (s * 6.4, 40.4, 31.0), (s * 7.8, 42.6, 27.2)], [(2.1, 2.0), (1.4, 1.4), (0.8, 0.8)], "obsidian", tip_color=[176, 36, 44], tip_start=0.72, component="horns")
        for ti, tz in enumerate((33.6, 35.8, 38.0, 39.8)):
            ux = 3.2 - 0.35 * ti
            m.box(f"upper_fang_{q}_{ti}", "head", (s * ux, 32.4, tz), (0.9, 1.8 if ti == 2 else 1.2, 0.9), "tooth", component="teeth")
            if ti in (1, 3):
                m.box(f"lower_fang_{q}_{ti}", "jaw", (s * (ux - 0.3), 33.0, tz - 0.4), (0.8, 1.3, 0.9), "tooth", component="teeth")

    # Streamlined Back-Swept Snow-White Fur Mantle + Outer Obsidian Blades (`stygian_zinogre_wide_board.jpg`)
    for ri, (rbone, rpos, rlen, rw, rt, count) in enumerate((
        ("head", (0.0, 38.0, 29.0), 13.8, 3.2, 4.4, 7),
        ("neck_2", (0.0, 36.4, 24.0), 16.8, 3.6, 5.2, 9),
        ("neck_1", (0.0, 33.8, 18.6), 18.8, 4.0, 5.6, 9),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            ang_fan = frac * 72.0
            rad_f = math.radians(ang_fan)
            base_pt = np.array([math.sin(rad_f) * (3.8 + 1.8 * ri), rpos[1] - abs(frac) * 4.5, rpos[2]])
            dir_v = unit([math.sin(rad_f) * 0.56, math.cos(rad_f) * 0.36, -0.82])
            mid_pt = base_pt + dir_v * (rlen * 0.56)
            tip_pt = base_pt + dir_v * rlen
            is_blade = (abs(frac) > 0.65)
            mat_f = "obsidian" if is_blade else "fur"
            m.span_box(f"stygian_mane_{ri}_{k}_a", rbone, base_pt, mid_pt, rw, rt, mat_f, pattern="armor_shingle" if is_blade else "fur_mane", component="horns" if is_blade else "surface")
            m.span_box(f"stygian_mane_{ri}_{k}_b", rbone, mid_pt, tip_pt, rw * 0.72, rt * 0.78, mat_f, tip_color=[210, 42, 50] if is_blade else None, pattern="armor_shingle" if is_blade else "fur_mane", component="horns" if is_blade else "surface")

    for ki, (kb, kpos, kw, kh, kd, rx, kmat) in enumerate((
        ("neck_2", (0.0, 27.6, 27.0), 8.6, 4.4, 6.5, 22.0, "fur"),
        ("neck_1", (0.0, 23.2, 23.6), 10.6, 5.2, 7.2, 18.0, "fur"),
        ("chest", (0.0, 18.8, 19.4), 11.8, 5.6, 8.4, 12.0, "obsidian"),
        ("chest", (0.0, 14.8, 13.8), 12.4, 5.0, 8.8, 8.0, "obsidian"),
        ("chest", (0.0, 13.2, 6.2), 13.0, 3.6, 8.4, 2.0, "belly"),
        ("pelvis", (0.0, 13.2, -5.5), 11.8, 3.2, 9.0, 0.0, "belly"),
        ("pelvis", (0.0, 13.6, -15.0), 10.0, 3.0, 8.5, -4.0, "belly"),
    )):
        m.box(f"chest_keel_plate_{ki}", kb, kpos, (kw, kh, kd), kmat, rotation=(rx, 0.0, 0.0), pattern="fur_mane" if kmat == "fur" else "belly_scute", component="armor")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        cb = cap_l if q == "l" else cap_r
        for pi, (pz, py, ph) in enumerate(((12.5, 35.0, 11.8), (6.5, 35.5, 13.0), (0.0, 34.2, 10.4))):
            b_p = np.array([s * 8.2, py - 2.0, pz])
            m_p = b_p + np.array([s * 2.2, ph * 0.55, -2.5])
            t_p = b_p + np.array([s * 4.2, ph, -5.4])
            m.span_box(f"shoulder_pylon_{q}_{pi}_a", cb, b_p, m_p, 3.6, 4.2, "obsidian", seam_color=[216, 42, 48], len_pad=0.7, pattern="armor_shingle", component="spines")
            m.span_box(f"shoulder_pylon_{q}_{pi}_b", cb, m_p, t_p, 1.8, 2.0, "horn", tip_color=[186, 38, 46], tip_start=0.68, len_pad=0.5, pattern="dorsal_blade", component="spines")

        for di, dz in enumerate((11.5, 8.0, 4.5, 1.0, -2.8, -6.5)):
            yb = 35.4 - 0.55 * di
            m.box(f"dragon_pip_{q}_{di}", "chest" if di < 4 else "pelvis", (s * (5.2 - 0.25 * di), yb, dz), (2.2, 2.6, 2.4), "obsidian", seam_color=[218, 44, 52], rotation=(12.0, s * 10.0, -s * 18.0), pattern="armor_shingle", component="spines")

    # Full-Wrap Obsidian Foreleg Bracer Armor + Tall Black/Crimson Claws
    for kind, p0_r, p1_r, p2_r in (
        ("fore", (11.8, 23.5, 10.5), (16.6, 13.8, 4.8), (15.8, 3.8, 12.2)),
        ("hind", (9.2, 20.2, -13.5), (13.0, 11.5, -7.2), (12.6, 3.5, -15.5)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0 = np.array([s * p0_r[0], p0_r[1], p0_r[2]])
            p1 = np.array([s * p1_r[0], p1_r[1], p1_r[2]])
            p2 = np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_muscular_shoulder_{q}", bu, p0 + np.array([s * 1.0, 2.2, 0.5]), (9.4 if kind == "fore" else 7.4, 8.6, 10.0), "fur" if kind == "fore" else "scales", face_materials={"up": "obsidian"}, rotation=(8.0, s * 10.0, -s * 14.0), pattern="fur_mane" if kind == "fore" else "calm_flank")
            n_br = 5 if kind == "fore" else 4
            for bi in range(n_br):
                f_pt = lerp3(p1, p2, 0.10 + 0.20 * bi) + np.array([s * (2.6 if kind == "fore" else 1.8), 1.4, 2.4])
                bw_w = 6.2 if kind == "fore" else 4.2
                m.box(f"{kind}_front_bracer_{q}_{bi}", bl, f_pt, (bw_w, 4.4 if kind == "fore" else 3.2, 5.8 if kind == "fore" else 4.2), "obsidian", seam_color=[236, 48, 56], rotation=(28.0, s * 22.0, -s * 18.0), pattern="armor_shingle", component="armor")
                m.box(f"{kind}_outer_bracer_{q}_{bi}", bl, f_pt + np.array([s * 2.2, -0.5, -2.0]), (2.6, 4.2 if kind == "fore" else 3.0, 5.4), "armor" if bi % 2 == 0 else "obsidian", rotation=(16.0, s * 12.0, -s * 28.0), pattern="armor_shingle", component="armor")
            if kind == "fore":
                m.span_box(f"elbow_crimson_blade_{q}", bl, p1 + np.array([s * 2.5, 1.0, -1.8]), p1 + np.array([s * 6.0, 9.4, -8.2]), 2.8, 3.8, "obsidian", tip_color=[224, 44, 52], pattern="armor_shingle", component="spines")
                m.box(f"tricep_fur_cowl_{q}", bu, lerp3(p0, p1, 0.65) + np.array([s * 1.8, -0.5, -3.5]), (5.4, 5.6, 5.0), "fur", rotation=(-24.0, 0.0, -s * 14.0), pattern="fur_mane", component="surface")
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
                m.span_box(f"{kind}_crimson_claw_{q}_{di}", bp, kb, kc, tw * 0.72, th * 0.85, "claw_tip", len_pad=0.4, tip_color=[244, 56, 62], tip_start=0.35, pattern="crystal_facet", component="claws")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: SNOW-WHITE FLANK FUR LOCKS, OBSIDIAN SADDLE & BROAD CLUB TAIL
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.shingle_arc(f"obsidian_saddle_ridge_{q}", "chest", (s * 11.4, 29.8, 13.0), (s * 9.5, 25.4, -4.0), (s * 0.28, 0.42, -0.86), (s * 0.24, 0.38, -0.88), count=7, length=7.6, width=3.8, thickness=2.4, material="obsidian", alt_material="armor", seam_color=[236, 48, 56], two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"chest_flank_armor_{q}", "chest", (s * 12.0, 23.0, 12.5), (s * 10.8, 20.8, -2.5), (s * 0.26, -0.10, -0.96), (s * 0.20, -0.10, -0.97), count=5, length=6.8, width=3.6, thickness=2.2, material="obsidian", alt_material="armor", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"white_pelvis_fur_{q}", "pelvis", (s * 8.6, 18.0, -4.5), (s * 6.8, 17.0, -18.5), (s * 0.35, -0.25, -0.90), (s * 0.28, -0.22, -0.92), count=5, length=6.4, width=3.2, thickness=2.0, material="fur", two_step=False, pattern="fur_mane")
        m.cutout_panel(f"flank_fur_fringes_{q}", "pelvis", [v3((s * 8.5, 20.0, -3.0)), v3((s * 7.2, 18.5, -19.0)), v3((s * 6.8, 13.0, -18.0)), v3((s * 8.2, 14.0, -3.0))], "fur", pattern="fur_tuft", component="surface")

    for ti in range(7):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.box(f"tail_top_obsidian_{ti}", tbones[ti], mid_t + np.array([0.0, 5.5 - 0.32 * ti, 0.0]), (max(3.6, 7.8 - 0.45 * ti), 3.4 - 0.16 * ti, 8.6), "obsidian", tip_color=[176, 36, 44] if ti >= 4 else None, tip_start=0.72, rotation=(-16.0, 0.0, 0.0), pattern="armor_shingle", component="spines")
        m.box(f"tail_white_fur_{ti}", tbones[ti], mid_t + np.array([0.0, -4.8 + 0.28 * ti, 0.0]), (max(2.8, 8.4 - 0.55 * ti), 3.4, 8.4), "fur", rotation=(-8.0, 0.0, 0.0), pattern="fur_mane", component="surface")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_chevron_upper_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(2.2, 6.2 - 0.45 * ti), 1.8 - 0.1 * ti, 0.0]), (3.0, 4.0 - 0.18 * ti, 8.0), "obsidian", rotation=(14.0, -s * 16.0, -s * 24.0), seam_color=[236, 48, 56] if ti >= 3 else None, pattern="armor_shingle", component="armor")
            m.box(f"tail_chevron_lower_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(2.0, 5.6 - 0.42 * ti), -1.8 + 0.1 * ti, 0.0]), (2.8, 3.4 - 0.16 * ti, 7.6), "fur" if ti < 3 else "armor", rotation=(-14.0, -s * 14.0, s * 22.0), pattern="fur_mane" if ti < 3 else "armor_shingle", component="armor")
            if ti in (4, 6):
                m.span_box(f"tail_tip_prong_{ti}_{q}", tbones[ti], mid_t + np.array([s * 2.2, -0.4, -2.0]), mid_t + np.array([s * 4.6, 3.4, -9.4]), 2.4, 3.0, "obsidian", tip_color=[224, 44, 52], component="spines")

    return m.doc
