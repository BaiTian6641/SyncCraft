"""src/synccraft/data_gen/mh_dragons_v8/species/gore_magala.py

Monumental Sculpted Blockbench Reconstruction of GORE MAGALA (`Black Eclipse Wyvern`, ??? / Demi-Elder).

Built directly from `gore_magala_wide_board.jpg`:
  - Eyeless blind obsidian helmet-visor skull with tapered hooked upper snout armor & stepped jaw
    (ZERO glowing eyes in calm form; sensory perception via articulating `feeler_l/r` antennae!).
  - Articulating multi-jointed glowing violet/magenta frenzy feelers arching off temporal brow.
  - Overlapping jagged obsidian collar plates enveloping hunched shoulders & neck.
  - Signature 6-limbed Wing-Cloak architecture (`winged_sail_rig` 3-hinge-bank tattered `wing_gore`
    cloak sails + oversized 3-clawed muscular Reaper wing-hands hanging forward from wrists).
  - Sleek armored draconic torso, layered dorsal spine ridges, strong clawed quadruped stance,
    and sinuous whip tail with violet sub-wing membrane veins (`~520+` elements).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def gore_magala_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [30, 30, 38], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [20, 20, 28], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [46, 44, 58], "kind": "armor", "coat": "armor_shingle"},
        "obsidian": {"color": [26, 24, 34], "kind": "armor", "coat": "armor_shingle"},
        "collar_dark": {"color": [38, 34, 50], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [54, 48, 68], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [36, 32, 46], "kind": "horn", "coat": "horn_keratin"},
        "feeler_glow": {"color": [174, 72, 224], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.42},
        "wing_mem": {"color": [48, 32, 64], "kind": "membrane", "coat": "wing_gore"},
        "wing_back": {"color": [30, 22, 44], "kind": "membrane", "coat": "wing_gore"},
        "claw": {"color": [68, 62, 76], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [124, 42, 78], "kind": "mouth"},
        "tooth": {"color": [224, 218, 216], "kind": "bone"},
    }


def build_gore_magala(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "gore_magala",
        "Gore Magala",
        "Black Eclipse Wyvern",
        "Elder Dragon",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Gore_Magala",
        materials=gore_magala_materials(),
        notes=[
            "Built from gore_magala_wide_board.jpg: eyeless blind-visor skull, articulating violet frenzy feelers, layered collar mantle, 3-bank tattered cloak wings + 3-clawed wing-arms.",
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
    # STAGE 1: PREDATORY 6-LIMBED WYVERN BLOCKOUT
    # =========================================================================
    torso_stations = [
        ((0.0, 20.5, -21.0), 12.0, 10.5, "pelvis"),
        ((0.0, 21.5, -11.0), 14.8, 12.8, "pelvis"),
        ((0.0, 24.0, 0.0), 18.0, 15.5, "chest"),
        ((0.0, 26.8, 10.5), 19.5, 16.8, "chest"),
        ((0.0, 28.8, 18.8), 14.5, 13.5, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.8, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 28.2, 17.2), (0.0, 31.6, 24.5), 11.5, 11.0),
        ("neck_2", (0.0, 31.6, 24.5), (0.0, 34.5, 32.2), 9.8, 9.4),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    # Eyeless Predatory Cranium + Hooked Viper Snout
    m.box("skull_visor_dome", "head", (0.0, 35.4, 35.8), (8.8, 6.2, 9.0), "obsidian", rotation=(6.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("snout_viper_beak", "head", (0.0, 34.0, 42.2), (5.8, 3.8, 6.5), "scales_deep", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 31.4, 39.2), (5.5, 2.5, 11.2), "scales_deep", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    # Agile Serpent-Dragon 7-Bone Tail
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

    # Quadruped Forelegs + Hindlegs (`plus massive wing-arms as 3rd limb pair!`)
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
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.2, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.2, pattern="calm_flank", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.5, p2[2] + 2.5), (7.2, 3.6, 7.2), "scales_deep", stage1=True)

    # Signature Enormous Overarching 3-Bank Reaper Cloak Wings (`winged_sail_rig` -> 12 panels + 3-claw wing-hands!)
    for s, q in ((1.0, "l"), (-1.0, "r")):
        sh_pt = (s * 8.2, 30.8, 10.2)
        el_pt = (s * 22.0, 42.5, 5.5)
        wr_pt = (s * 32.0, 36.5, 16.5)
        tips = (
            (s * 58.0, 39.0, 4.0),
            (s * 53.0, 28.0, -9.0),
            (s * 42.0, 20.0, -18.0),
            (s * 28.0, 15.0, -24.0),
        )
        wing_bones = m.winged_sail_rig(
            s,
            "chest",
            sh_pt,
            el_pt,
            wr_pt,
            tips,
            (s * 7.8, 22.0, -10.5),
            spar_mat="obsidian",
            hub_mat="armor",
            claw_mat="claw",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_gore",
            spar_w=3.2,
            arm_w=5.0,
            leading_scales=6,
            tattered_outer=True,
        )
        if stage_cap >= 2:
            # Gore Magala's Iconic 3-Clawed Grabbing Reaper Hand on `wing_wrist`!
            b_wr = f"wing_wrist_{q}"
            m.box(f"wing_palm_gauntlet_{q}", b_wr, (wr_pt[0], wr_pt[1] - 1.5, wr_pt[2] + 2.5), (5.5, 4.2, 6.2), "armor", rotation=(12.0, s * 15.0, 0.0), pattern="armor_shingle", component="wing_arm")
            for ci, (cx_off, cz_off, clen) in enumerate(((-1.8, 3.5, 9.5), (0.0, 4.5, 11.2), (1.8, 3.2, 9.2))):
                ca = np.array([wr_pt[0] + s * cx_off, wr_pt[1] - 2.0, wr_pt[2] + cz_off])
                cb = ca + np.array([s * 1.2, -5.5, clen * 0.45])
                cc = cb + np.array([-s * 1.0, -clen * 0.45, clen * 0.12])
                m.tapered_arch(f"wing_reaper_claw_{q}_{ci}", b_wr, [ca, cb, cc], [(2.2, 2.2), (1.7, 1.7), (0.8, 0.8)], "claw", component="claws")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: EYELESS BLIND VISOR HELM, ARTICULATING FEELERS & COLLAR MANTLE
    # =========================================================================
    # Armored Blind-Visor Helm plates completely shading eye sockets (`NO glowing eyes!`)
    m.box("blind_visor_hood_top", "head", (0.0, 38.0, 37.5), (9.6, 2.5, 11.2), "obsidian", rotation=(14.0, 0.0, 0.0), pattern="armor_shingle", component="head")
    m.box("nasal_crest_hook", "head", (0.0, 35.8, 44.5), (4.6, 2.2, 4.8), "armor", rotation=(24.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("lower_chin_prow", "jaw", (0.0, 30.8, 43.2), (4.2, 2.2, 3.8), "armor", rotation=(-10.0, 0.0, 0.0), pattern="armor_shingle", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"blind_cheek_shield_{q}", "head", (s * 4.8, 34.8, 36.2), (2.2, 3.8, 8.2), "collar_dark", rotation=(10.0, s * 10.0, -s * 16.0), pattern="armor_shingle", component="head")
        m.box(f"jaw_blade_flank_{q}", "jaw", (s * 3.0, 31.6, 39.2), (1.4, 2.4, 9.8), "armor", pattern="armor_shingle", component="jaw")
        for ti, tz in enumerate((36.5, 39.0, 41.4, 43.4)):
            ux = 2.6 - 0.25 * ti
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * ux, 32.6, tz), (0.8, 1.4 if ti == 2 else 1.0, 0.8), "tooth", component="teeth")
            if ti % 2 == 1:
                m.box(f"lower_tooth_{q}_{ti}", "jaw", (s * (ux - 0.3), 32.8, tz - 0.4), (0.7, 1.0, 0.8), "tooth", component="teeth")

        # Articulating Violet-Glow Frenzy Feelers (`feeler_l`, `feeler_r`) swept sleekly along skull sides
        fb = m.bone(f"feeler_{q}", "head", (s * 3.8, 37.5, 35.5))
        m.tapered_arch(
            f"frenzy_feeler_horn_{q}",
            fb,
            [(s * 3.8, 37.8, 35.8), (s * 6.5, 42.2, 30.5), (s * 8.2, 46.8, 24.5), (s * 8.8, 50.8, 19.0)],
            [(2.4, 2.4), (2.0, 2.0), (1.5, 1.5), (0.9, 0.9)],
            "horn",
            tip_color=[186, 84, 238],
            tip_start=0.25,
            component="horns",
            stage1=True,
        )
        m.box(f"feeler_glow_nodule_{q}", fb, (s * 7.8, 46.0, 25.5), (1.4, 1.6, 3.8), "feeler_glow", rotation=(-32.0, s * 18.0, 0.0), pattern="crystal_facet", component="horns")

        # Temporal skull crown wing-crests framing behind feelers
        m.tapered_arch(
            f"cranial_mantle_horn_{q}",
            "head",
            [(s * 4.5, 36.2, 33.2), (s * 7.8, 38.2, 27.5), (s * 10.2, 39.5, 22.0)],
            [(2.4, 3.0), (1.8, 2.4), (1.0, 1.4)],
            "obsidian",
            component="horns",
        )

    # Signature Layered Obsidian Neck Collar Shingle Mantle (`Reaper shoulder cloak collar!`)
    for ci, (bn, cp, clen, cw, count) in enumerate((
        ("neck_2", (0.0, 35.5, 26.0), 10.5, 4.2, 5),
        ("neck_1", (0.0, 32.8, 19.5), 13.5, 4.8, 7),
        ("chest", (0.0, 30.8, 12.8), 15.5, 5.2, 7),
    )):
        for k in range(count):
            frac = (k / max(1, count - 1)) * 2.0 - 1.0
            ang = frac * 70.0
            ra = math.radians(ang)
            b_c = np.array([math.sin(ra) * (4.2 + 1.8 * ci), cp[1] - abs(frac) * 4.0, cp[2]])
            dv = unit([math.sin(ra) * 0.58, math.cos(ra) * 0.30, -0.82])
            t_c = b_c + dv * clen
            m.span_box(f"collar_petal_{ci}_{k}", bn, b_c, t_c, cw, 2.0, "collar_dark" if k % 2 == 0 else "obsidian", len_pad=0.7, pattern="armor_shingle", component="armor")

    # Dark Ventral Scutes & Fore/Hind Limb Gauntlets + Claws
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
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0 = np.array([s * p0_r[0], p0_r[1], p0_r[2]])
            p1 = np.array([s * p1_r[0], p1_r[1], p1_r[2]])
            p2 = np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_shoulder_cap_{q}", bu, p0 + np.array([s * 0.8, 1.5, 0.5]), (6.4, 6.4, 7.4), "armor", rotation=(8.0, s * 10.0, -s * 14.0), pattern="armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.15 + 0.22 * bi) + np.array([s * 1.8, 0.8, 1.2])
                m.box(f"{kind}_vambrace_{q}_{bi}", bl, pt_b, (2.4, 3.6, 4.4), "collar_dark", rotation=(20.0, s * 14.0, -s * 20.0), pattern="armor_shingle", component="armor")
            for di, ang_d in ((0, -18.0), (1, 0.0), (2, 18.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-1.8 + 1.8 * di), 2.2, p2[2] + 4.8])
                tb = ta + dv * 3.5
                tc = tb + dv * 2.8 - np.array([0.0, 0.8, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 1.8, 1.8, "obsidian", len_pad=0.4, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.4, 1.4, "claw", len_pad=0.3, component="claws")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: DORSAL VERTEBRAL KNIFE BLADES, FLANK SHINGLES & WHIP TAIL SPINES
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.shingle_arc(f"chest_flank_shingles_{q}", "chest", (s * 9.8, 25.5, 11.5), (s * 8.4, 23.0, -3.5), (s * 0.26, 0.14, -0.95), (s * 0.20, 0.12, -0.97), count=6, length=6.8, width=3.2, thickness=2.0, material="armor", alt_material="obsidian", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"pelvis_flank_shingles_{q}", "pelvis", (s * 7.8, 21.8, -5.0), (s * 6.0, 20.2, -19.5), (s * 0.24, 0.12, -0.96), (s * 0.18, 0.10, -0.98), count=5, length=6.2, width=2.8, thickness=1.8, material="obsidian", alt_material="collar_dark", two_step=False, pattern="armor_shingle")

    # Swept Dorsal Spine Blades down chest, pelvis & all 7 tail segments
    for di, (db, dy, dz, dh) in enumerate((
        ("chest", 34.5, 8.5, 6.2),
        ("chest", 33.2, 2.5, 6.8),
        ("chest", 31.8, -3.5, 6.4),
        ("pelvis", 28.8, -10.0, 5.8),
        ("pelvis", 26.8, -16.5, 5.4),
    )):
        m.span_box(f"dorsal_razor_spine_{di}", db, (0.0, dy - 1.5, dz), (0.0, dy + dh, dz - 5.2), 1.6, 2.4, "obsidian", pattern="dorsal_blade", component="spines")

    for ti in range(7):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_spine_blade_{ti}", tbones[ti], mid_t + np.array([0.0, 2.5, 0.5]), mid_t + np.array([0.0, 6.2 - 0.3 * ti, -4.2]), 1.4, 2.2, "obsidian", pattern="dorsal_blade", component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_side_scute_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.0, 3.8 - 0.38 * ti), 0.2, 0.0]), (1.6, max(1.4, 2.4 - 0.15 * ti), 6.2), "armor", rotation=(8.0, -s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")

    return m.doc
