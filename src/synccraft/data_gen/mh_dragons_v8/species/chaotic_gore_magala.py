"""src/synccraft/data_gen/mh_dragons_v8/species/chaotic_gore_magala.py

Monumental Asymmetric Sculpted Blockbench Reconstruction of CHAOTIC GORE MAGALA (`Tortured Eclipse`, Elder Dragon).

Built strictly from `chaotic_gore_magala_wide_board.jpg` (`[1],[2],[7],[8]` — bilateral half-molt!):
  - LEFT HALF (`+X` / `_l`): unbroken pitch-obsidian Gore Magala visor skull (`eyeless`), swept dark
    frenzy feeler, charcoal collar cloak, dark tattered violet wing-arm (`wing_gore`).
  - RIGHT HALF (`-X` / `_r`): fractured violently into blazing GOLDEN SHAGARU MAGALA armor!
      * Exposed glowing amber-crimson eye beneath split cranial shell (`cheek & brow GOLD`)
      * Towering jagged thicker gilded Shagaru crown horn rising steeply skyward
      * Enlarged celestial gold/ivory wing-arm (`+18%` longer swept fingers, golden `wing_shagaru`
        amber-veined membrane, cracked luminous amber-gold shoulder gauntlets & claws!)
      * Right chest, right flank, right thigh & right tail plates erupt in molten gold plating vs
        unbroken obsidian charcoal on the left (`true anatomical bilaterality asymmetry`).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def chaotic_gore_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [30, 30, 38], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [20, 20, 28], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [46, 44, 58], "kind": "armor", "coat": "armor_shingle"},
        "obsidian": {"color": [26, 24, 34], "kind": "armor", "coat": "armor_shingle"},
        "gold_plate": {"color": [174, 130, 52], "kind": "armor", "coat": "armor_shingle"},
        "gold_bright": {"color": [206, 166, 74], "kind": "armor", "coat": "armor_shingle"},
        "ivory_gold": {"color": [216, 196, 146], "kind": "horn", "coat": "horn_keratin"},
        "belly": {"color": [58, 52, 72], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [36, 32, 46], "kind": "horn", "coat": "horn_keratin"},
        "feeler_glow": {"color": [174, 72, 224], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.42},
        "wing_gore": {"color": [48, 32, 64], "kind": "membrane", "coat": "wing_gore"},
        "wing_gore_b": {"color": [30, 22, 44], "kind": "membrane", "coat": "wing_gore"},
        "wing_gold": {"color": [182, 134, 52], "kind": "membrane", "coat": "wing_shagaru"},
        "wing_gold_b": {"color": [132, 92, 38], "kind": "membrane", "coat": "wing_shagaru"},
        "claw": {"color": [68, 62, 76], "kind": "horn", "coat": "horn_keratin"},
        "claw_gold": {"color": [216, 172, 78], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [128, 42, 76], "kind": "mouth"},
        "tooth": {"color": [228, 222, 214], "kind": "bone"},
        "eye": {"color": [255, 110, 44], "kind": "eye", "emissive": 0.90},
    }


def build_chaotic_gore_magala(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "chaotic_gore_magala",
        "Chaotic Gore Magala",
        "Tortured Eclipse",
        "Elder Dragon",
        variant_kind="variant",
        parent_species_id="gore_magala",
        wiki_url="https://monsterhunter.fandom.com/wiki/Chaotic_Gore_Magala",
        materials=chaotic_gore_materials(),
        notes=[
            "Asymmetric half-molted sculpt from chaotic_gore_magala_wide_board.jpg: left side unbroken Gore Magala obsidian/violet (`eyeless`), right side erupted blazing Shagaru Magala gold armor, taller gilded crown horn, exposed orange eye, and enlarged celestial golden wing.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 21.0, -14.0))
    m.bone("chest", "pelvis", (0.0, 25.0, 5.5))
    m.bone("neck_1", "chest", (0.0, 28.5, 17.5))
    m.bone("neck_2", "neck_1", (0.0, 32.0, 25.5))
    m.bone("head", "neck_2", (0.0, 34.5, 32.5))
    m.bone("jaw", "head", (0.0, 31.6, 33.2))

    # =========================================================================
    # STAGE 1: PREDATORY MASSES (SPLIT CORE PLATING RIGHT-GOLD vs LEFT-DARK!)
    # =========================================================================
    torso_stations = [
        ((0.0, 20.5, -21.0), 12.2, 10.5, "pelvis"),
        ((0.0, 21.5, -11.0), 15.0, 12.8, "pelvis"),
        ((0.0, 24.0, 0.0), 18.2, 15.6, "chest"),
        ((0.0, 26.8, 10.5), 19.8, 17.0, "chest"),
        ((0.0, 28.8, 18.8), 14.8, 13.6, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.8, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 28.2, 17.2), (0.0, 31.6, 24.5), 11.6, 11.0),
        ("neck_2", (0.0, 31.6, 24.5), (0.0, 34.5, 32.2), 9.8, 9.5),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    m.box("skull_visor_dome", "head", (0.0, 35.4, 35.8), (8.8, 6.2, 9.0), "obsidian", rotation=(6.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("snout_viper_beak", "head", (0.0, 34.0, 42.2), (5.8, 3.8, 6.5), "scales_deep", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 31.4, 39.2), (5.5, 2.5, 11.2), "scales_deep", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 20.5, -20.0), 10.5, 9.5),
        ((1.2, 19.6, -29.0), 9.0, 8.2),
        ((2.8, 19.2, -38.0), 7.6, 7.0),
        ((3.8, 20.0, -47.0), 6.2, 5.8),
        ((3.2, 21.8, -56.0), 5.0, 4.8),
        ((1.2, 24.0, -64.5), 3.8, 3.8),
        ((-1.2, 26.2, -72.0), 2.8, 3.0),
        ((-3.5, 27.8, -78.5), 2.0, 2.2),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(7)]
    for i in range(7):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.6, pattern="calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (9.5, 22.5, 9.5), (13.0, 12.5, 6.2), (12.5, 3.5, 11.8), (6.8, 6.5), (5.5, 5.4)),
        ("hind", "pelvis", (8.5, 20.0, -13.5), (12.2, 11.2, -7.2), (11.8, 3.2, -14.8), (8.2, 7.8), (5.8, 5.6)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0 = (s * p0_r[0], p0_r[1], p0_r[2])
            p1 = (s * p1_r[0], p1_r[1], p1_r[2])
            p2 = (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "gold_plate" if q == "r" else "scales", len_pad=2.2, pattern="armor_shingle" if q == "r" else "calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.2, pattern="calm_flank", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.5, p2[2] + 2.5), (7.2, 3.6, 7.2), "scales_deep", stage1=True)

    # ASYMMETRIC WINGS: Left (`s=+1`) compact dark Gore cloak | Right (`s=-1`) enlarged blazing Shagaru gold wing!
    for s, q in ((1.0, "l"), (-1.0, "r")):
        is_gold = (q == "r")
        reach = 1.18 if is_gold else 0.96
        sh_pt = (s * 8.2, 30.8, 10.2)
        el_pt = (s * (26.5 if is_gold else 23.0), 47.0 if is_gold else 42.5, 4.5 if is_gold else 3.0)
        wr_pt = (s * (38.0 if is_gold else 32.5), 45.5 if is_gold else 40.5, 16.0 if is_gold else 14.0)
        tips = tuple((s * px * reach, py + (4.0 if is_gold else 0.0), pz) for px, py, pz in (
            (58.0, 44.0, 4.0),
            (52.0, 33.0, -9.0),
            (41.0, 23.0, -18.0),
            (28.0, 18.0, -24.0),
        ))
        wing_bones = m.winged_sail_rig(
            s,
            "chest",
            sh_pt,
            el_pt,
            wr_pt,
            tips,
            (s * 7.8, 22.0, -10.5),
            spar_mat="gold_plate" if is_gold else "obsidian",
            hub_mat="gold_bright" if is_gold else "armor",
            claw_mat="claw_gold" if is_gold else "claw",
            mem_mat="wing_gold" if is_gold else "wing_gore",
            mem_back_mat="wing_gold_b" if is_gold else "wing_gore_b",
            mem_pattern="wing_shagaru" if is_gold else "wing_gore",
            spar_w=3.6 if is_gold else 3.0,
            arm_w=5.4 if is_gold else 4.8,
            leading_scales=6,
            tattered_outer=(not is_gold),
        )
        if stage_cap >= 2:
            b_wr = f"wing_wrist_{q}"
            m.box(f"wing_palm_gauntlet_{q}", b_wr, (wr_pt[0], wr_pt[1] - 1.5, wr_pt[2] + 2.5), (5.8 if is_gold else 5.2, 4.4, 6.4), "gold_bright" if is_gold else "armor", rotation=(12.0, s * 15.0, 0.0), pattern="armor_shingle", component="wing_arm")
            for ci, (cx_off, cz_off, clen) in enumerate(((-1.8, 3.5, 9.5), (0.0, 4.5, 11.5), (1.8, 3.2, 9.2))):
                ca = np.array([wr_pt[0] + s * cx_off, wr_pt[1] - 2.0, wr_pt[2] + cz_off])
                cb = ca + np.array([s * 1.2, -5.5, clen * 0.45])
                cc = cb + np.array([-s * 1.0, -clen * 0.45, clen * 0.12])
                m.tapered_arch(f"wing_reaper_claw_{q}_{ci}", b_wr, [ca, cb, cc], [(2.3, 2.3), (1.7, 1.7), (0.8, 0.8)], "claw_gold" if is_gold else "claw", component="claws")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: HALF-MOLTED SKULL (BLIND LEFT vs GOLDEN EYED SHAGARU RIGHT!)
    # =========================================================================
    # Split cranial visor (left stays dark visor, right breaks open with Shagaru gold ridge & glowing eye!)
    m.box("left_blind_visor_hood", "head", (2.2, 38.0, 37.5), (5.2, 2.5, 11.2), "obsidian", rotation=(14.0, 0.0, -8.0), pattern="armor_shingle", component="head")
    m.box("right_gold_brow_plate", "head", (-2.6, 38.6, 37.8), (5.2, 2.6, 11.0), "gold_plate", seam_color=[255, 130, 50], rotation=(16.0, -10.0, 12.0), pattern="armor_shingle", component="head")
    m.box("nasal_crest_hook", "head", (0.0, 35.8, 44.5), (4.6, 2.2, 4.8), "gold_plate", rotation=(24.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("lower_chin_prow", "jaw", (0.0, 30.8, 43.2), (4.2, 2.2, 3.8), "armor", rotation=(-10.0, 0.0, 0.0), pattern="armor_shingle", component="jaw")

    # Right Eye Exposed Only (`-X` side!) + Golden Cheek Plate
    m.box("eye_recess_r", "head", (-4.8, 36.4, 39.2), (1.4, 1.8, 3.0), "mouth", rotation=(8.0, -10.0, 0.0), component="eyes")
    m.box("eye_chaos_slit_r", "head", (-5.2, 36.4, 39.4), (1.0, 1.3, 2.4), "eye", rotation=(8.0, -10.0, 0.0), pattern="eye_pixel", component="eyes")
    m.box("blind_cheek_shield_l", "head", (4.8, 34.8, 36.2), (2.2, 3.8, 8.2), "obsidian", rotation=(10.0, 10.0, -16.0), pattern="armor_shingle", component="head")
    m.box("gold_cheek_shield_r", "head", (-5.0, 34.6, 36.0), (2.6, 4.0, 8.4), "gold_plate", seam_color=[255, 130, 50], rotation=(10.0, -14.0, 14.0), pattern="armor_shingle", component="head")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        is_gold = (q == "r")
        m.box(f"jaw_blade_flank_{q}", "jaw", (s * 3.0, 31.6, 39.2), (1.5, 2.4, 9.8), "gold_plate" if is_gold else "armor", pattern="armor_shingle", component="jaw")
        for ti, tz in enumerate((36.5, 39.0, 41.4, 43.4)):
            ux = 2.6 - 0.25 * ti
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * ux, 32.6, tz), (0.8, 1.4 if ti == 2 else 1.0, 0.8), "tooth", component="teeth")
            if ti % 2 == 1:
                m.box(f"lower_tooth_{q}_{ti}", "jaw", (s * (ux - 0.3), 32.8, tz - 0.4), (0.7, 1.0, 0.8), "tooth", component="teeth")

        fb = m.bone(f"feeler_{q}", "head", (s * 3.8, 37.5, 35.5))
        if is_gold:
            # Towering Gilded Asymmetric Shagaru Crown Horn (`r`) curving skyward and much thicker!
            m.tapered_arch("shagaru_gold_crown_r", fb, [(-3.8, 38.0, 35.8), (-6.8, 44.5, 31.2), (-9.2, 51.0, 25.5), (-10.0, 56.8, 19.5)], [(3.4, 3.0), (2.8, 2.5), (2.0, 1.8), (1.1, 1.1)], "gold_bright", tip_color=[255, 232, 164], tip_start=0.45, component="horns", stage1=True)
            m.tapered_arch("shagaru_halo_blade_r", "head", [(-4.8, 36.8, 33.2), (-9.5, 40.5, 27.0), (-13.2, 43.5, 21.0)], [(2.8, 3.4), (2.2, 2.8), (1.2, 1.6)], "gold_plate", tip_color=[255, 232, 164], component="horns")
        else:
            m.tapered_arch("frenzy_feeler_horn_l", fb, [(3.8, 37.8, 35.8), (6.5, 42.2, 30.5), (8.2, 46.8, 24.5), (8.8, 50.8, 19.0)], [(2.4, 2.4), (2.0, 2.0), (1.5, 1.5), (0.9, 0.9)], "horn", tip_color=[186, 84, 238], tip_start=0.25, component="horns", stage1=True)
            m.box("feeler_glow_nodule_l", fb, (7.8, 46.0, 25.5), (1.4, 1.6, 3.8), "feeler_glow", rotation=(-32.0, 18.0, 0.0), pattern="crystal_facet", component="horns")
            m.tapered_arch("cranial_mantle_horn_l", "head", [(4.5, 36.2, 33.2), (7.8, 38.2, 27.5), (10.2, 39.5, 22.0)], [(2.4, 3.0), (1.8, 2.4), (1.0, 1.4)], "obsidian", component="horns")

    # Split Mantle Collar (left half obsidian, right half Shagaru gold petals!)
    for ci, (bn, cp, clen, cw, count) in enumerate((
        ("neck_2", (0.0, 35.5, 26.0), 10.8, 4.2, 6),
        ("neck_1", (0.0, 32.8, 19.5), 14.0, 4.8, 8),
        ("chest", (0.0, 30.8, 12.8), 16.0, 5.2, 8),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            ang = frac * 72.0
            ra = math.radians(ang)
            b_c = np.array([math.sin(ra) * (4.2 + 1.8 * ci), cp[1] - abs(frac) * 4.0, cp[2]])
            dv = unit([math.sin(ra) * 0.58, math.cos(ra) * 0.30, -0.82])
            t_c = b_c + dv * clen
            mat_p = ("gold_bright" if k % 2 == 0 else "gold_plate") if frac < -0.05 else ("obsidian" if k % 2 == 0 else "armor")
            m.span_box(f"collar_petal_{ci}_{k}", bn, b_c, t_c, cw, 2.0, mat_p, len_pad=0.7, pattern="armor_shingle", component="armor")

    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 29.0, 27.5), 7.4, 5.8, 22.0),
        ("neck_1", (0.0, 25.0, 22.5), 9.0, 6.4, 18.0),
        ("chest", (0.0, 18.2, 13.5), 11.8, 7.2, 10.0),
        ("chest", (0.0, 16.5, 5.5), 12.8, 7.5, 2.0),
        ("chest", (0.0, 16.2, -2.5), 12.2, 7.2, 0.0),
        ("pelvis", (0.0, 15.8, -10.5), 11.2, 7.0, 0.0),
        ("pelvis", (0.0, 16.2, -18.5), 9.5, 6.2, -4.0),
    )):
        m.box(f"ventral_plate_{ki}", kb, kpos, (kw, 2.4, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    for kind, p0_r, p1_r, p2_r in (
        ("fore", (9.5, 22.5, 9.5), (13.0, 12.5, 6.2), (12.5, 3.5, 11.8)),
        ("hind", (8.5, 20.0, -13.5), (12.2, 11.2, -7.2), (11.8, 3.2, -14.8)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            is_gold = (q == "r")
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0 = np.array([s * p0_r[0], p0_r[1], p0_r[2]])
            p1 = np.array([s * p1_r[0], p1_r[1], p1_r[2]])
            p2 = np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_shoulder_cap_{q}", bu, p0 + np.array([s * 0.8, 1.5, 0.5]), (6.6, 6.6, 7.6), "gold_bright" if is_gold else "armor", seam_color=[255, 130, 50] if is_gold else None, rotation=(8.0, s * 10.0, -s * 14.0), pattern="armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.15 + 0.22 * bi) + np.array([s * 1.8, 0.8, 1.2])
                m.box(f"{kind}_vambrace_{q}_{bi}", bl, pt_b, (2.5, 3.8, 4.5), "gold_plate" if (is_gold and bi % 2 == 0) else "obsidian", rotation=(20.0, s * 14.0, -s * 20.0), pattern="armor_shingle", component="armor")
            for di, ang_d in ((0, -18.0), (1, 0.0), (2, 18.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-1.8 + 1.8 * di), 2.2, p2[2] + 4.8])
                tb = ta + dv * 3.5
                tc = tb + dv * 2.8 - np.array([0.0, 0.8, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 1.8, 1.8, "gold_plate" if is_gold else "obsidian", len_pad=0.4, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.4, 1.4, "claw_gold" if is_gold else "claw", len_pad=0.3, component="claws")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: SPLIT MOLT FLANKS, CRACKED CARAPACE SEAMS & TAIL SHINGLES
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        is_gold = (q == "r")
        mat_a = "gold_plate" if is_gold else "armor"
        mat_b = "gold_bright" if is_gold else "obsidian"
        sc_col = [255, 130, 50] if is_gold else None
        m.shingle_arc(f"chest_flank_shingles_{q}", "chest", (s * 9.8, 25.5, 11.5), (s * 8.4, 23.0, -3.5), (s * 0.26, 0.14, -0.95), (s * 0.20, 0.12, -0.97), count=6, length=6.8, width=3.4, thickness=2.1, material=mat_a, alt_material=mat_b, seam_color=sc_col, two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"pelvis_flank_shingles_{q}", "pelvis", (s * 7.8, 21.8, -5.0), (s * 6.0, 20.2, -19.5), (s * 0.24, 0.12, -0.96), (s * 0.18, 0.10, -0.98), count=5, length=6.2, width=3.0, thickness=1.9, material=mat_b, alt_material=mat_a, two_step=False, pattern="armor_shingle")

    for di, (db, dy, dz, dh) in enumerate((
        ("chest", 34.5, 8.5, 6.4),
        ("chest", 33.2, 2.5, 7.0),
        ("chest", 31.8, -3.5, 6.5),
        ("pelvis", 28.8, -10.0, 5.8),
        ("pelvis", 26.8, -16.5, 5.4),
    )):
        m.span_box(f"dorsal_razor_spine_{di}", db, (0.0, dy - 1.5, dz), (0.0, dy + dh, dz - 5.2), 1.9, 2.8, "gold_plate" if di % 2 == 1 else "obsidian", tip_color=[255, 224, 148] if di % 2 == 1 else [152, 72, 198], tip_start=0.50, pattern="dorsal_blade", component="spines")

    for ti in range(7):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_spine_blade_{ti}", tbones[ti], mid_t + np.array([0.0, 2.5, 0.5]), mid_t + np.array([0.0, 6.4 - 0.3 * ti, -4.4]), 1.6, 2.4, "gold_plate" if ti % 2 == 1 else "obsidian", tip_color=[240, 196, 98] if ti >= 3 else None, pattern="dorsal_blade", component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            is_gold = (q == "r")
            m.box(f"tail_side_scute_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.0, 3.8 - 0.38 * ti), 0.2, 0.0]), (1.7, max(1.4, 2.4 - 0.15 * ti), 6.2), "gold_plate" if is_gold else "armor", rotation=(8.0, -s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")

    return m.doc
