"""src/synccraft/data_gen/mh_dragons_v8/species/crimson_glow_valstrax.py

Monumental Sculpted Blockbench Reconstruction of CRIMSON GLOW VALSTRAX (`Amber Comet`, Elder Dragon).

Built directly from `crimson_glow_valstrax_wide_board.jpg` (`[1],[5],[13],[14],[15],[16]`):
  - Sleek supersonic gunmetal-silver raptor skull with narrow aerodynamic swept-back blade crest,
    crimson dragon-energy chest intake turbine glowing (`<= 3%` emissive), and segmented chrome torso.
  - Enormous Articulating 3-LANCET JET-THRUSTER WINGS (`wing_l -> wing_fore_l -> wing_wrist_l -> wing_mid -> distal`):
      * Oversized heavy articulated forearm stabilizer yoke
      * THREE stacked tapered supersonic intake/exhaust rocket thruster nacelles (`thruster_0/1/2`)
        with dark recessed exhaust throats & glowing crimson afterburner jets + swept delta blade fins!
  - Agile quadrupedal metallic stance & slim swept supersonic tail stabilizing fins (`~320+` elements).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def valstrax_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [126, 132, 146], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [74, 80, 96], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [168, 174, 188], "kind": "armor", "coat": "armor_shingle"},
        "chrome_hi": {"color": [196, 202, 214], "kind": "armor", "coat": "armor_shingle"},
        "jet_dark": {"color": [34, 34, 42], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [92, 96, 112], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [88, 94, 110], "kind": "horn", "coat": "horn_keratin"},
        "crimson_glow": {"color": [244, 48, 54], "kind": "energy", "coat": "energy", "emissive": 0.84},
        "wing_mem": {"color": [144, 44, 54], "kind": "membrane", "coat": "wing_valstrax"},
        "wing_back": {"color": [96, 28, 38], "kind": "membrane", "coat": "wing_valstrax"},
        "claw": {"color": [48, 50, 62], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [136, 44, 58], "kind": "mouth"},
        "tooth": {"color": [232, 232, 238], "kind": "bone"},
        "eye": {"color": [255, 64, 68], "kind": "eye", "emissive": 0.90},
    }


def build_crimson_glow_valstrax(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "crimson_glow_valstrax",
        "Crimson Glow Valstrax",
        "Crimson Comet Elder",
        "Elder Dragon",
        variant_kind="variant",
        wiki_url="https://monsterhunter.fandom.com/wiki/Crimson_Glow_Valstrax",
        materials=valstrax_materials(),
        notes=[
            "Built strictly from crimson_glow_valstrax_wide_board.jpg `[14]..[16]` blueprints: peregrine jet skull, ventral crimson chest turbine intake, and articulating 3-barrel rocket-thruster wing nacelles.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 21.0, -14.0))
    m.bone("chest", "pelvis", (0.0, 25.5, 5.5))
    m.bone("neck_1", "chest", (0.0, 30.0, 17.0))
    m.bone("neck_2", "neck_1", (0.0, 34.5, 24.5))
    m.bone("head", "neck_2", (0.0, 37.5, 31.2))
    m.bone("jaw", "head", (0.0, 35.0, 31.8))

    # =========================================================================
    # STAGE 1: SLEEK JET-DRAGON AERODYNAMIC BLOCKOUT
    # =========================================================================
    torso_stations = [
        ((0.0, 20.5, -20.5), 11.0, 10.0, "pelvis"),
        ((0.0, 22.2, -10.0), 13.8, 12.2, "pelvis"),
        ((0.0, 25.2, 1.0), 17.0, 15.2, "chest"),
        ((0.0, 27.8, 11.2), 18.8, 16.5, "chest"),
        ((0.0, 29.5, 19.0), 14.0, 13.0, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.8, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 29.5, 16.5), (0.0, 33.8, 23.5), 10.2, 9.8),
        ("neck_2", (0.0, 33.8, 23.5), (0.0, 37.5, 30.8), 8.5, 8.2),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    # Sharp Peregrine Fighter-Jet Skull & Tapered Nose Cone
    m.box("skull_cranium", "head", (0.0, 38.2, 34.0), (8.4, 5.4, 8.0), "armor", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("jet_snout_cone", "head", (0.0, 37.2, 39.8), (5.0, 3.4, 5.8), "chrome_hi", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 34.8, 37.4), (4.8, 2.2, 9.4), "scales_deep", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    t_pts = [
        ((0.0, 20.5, -20.0), 9.5, 8.8),
        ((0.8, 19.8, -29.0), 8.0, 7.4),
        ((2.0, 19.5, -38.0), 6.8, 6.4),
        ((2.8, 20.2, -47.0), 5.6, 5.4),
        ((2.4, 21.6, -56.0), 4.4, 4.4),
        ((1.2, 23.2, -64.5), 3.4, 3.6),
        ((0.0, 24.8, -72.0), 2.4, 2.6),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(6)]
    for i in range(6):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.6, pattern="calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (9.2, 22.8, 10.0), (12.8, 12.5, 6.2), (12.4, 3.5, 11.8), (6.6, 6.4), (5.2, 5.0)),
        ("hind", "pelvis", (8.2, 20.2, -13.5), (12.0, 11.2, -7.0), (11.6, 3.2, -14.8), (7.8, 7.6), (5.5, 5.4)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0, p1, p2 = (s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.2, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.2, pattern="leg_bands", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.5, p2[2] + 2.5), (6.8, 3.4, 7.0), "scales_deep", stage1=True)

    # Signature Valstrax 3-Lancet Rocket-Jet Wings (`armored silver yoke + 3 thruster nacelles + plasma fins!`)
    for s, q in ((1.0, "l"), (-1.0, "r")):
        sh_pt, el_pt, wr_pt = (s * 8.2, 31.2, 9.5), (s * 22.0, 44.5, 4.0), (s * 34.0, 46.5, 13.5)
        m.winged_sail_rig(
            s,
            "chest",
            sh_pt,
            el_pt,
            wr_pt,
            ((s * 66.0, 50.0, -8.0), (s * 58.0, 38.0, -18.0), (s * 45.0, 27.0, -25.0), (s * 30.0, 20.0, -26.0)),
            (s * 7.6, 22.0, -9.0),
            spar_mat="chrome_hi",
            hub_mat="jet_dark",
            claw_mat="claw",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_valstrax",
            spar_w=3.8,
            arm_w=5.2,
            leading_scales=5,
            tattered_outer=False,
        )
        # Signature Trio of Heavy Supersonic Rocket Thruster Nacelles mounted on `wing_wrist`/`wing_mid`!
        b_wr = f"wing_wrist_{q}"
        b_mid = f"wing_mid_{q}"
        for li, (dx_l, dy_l, dz_l, th_len) in enumerate(((10.5, 4.5, -14.0, 24.0), (14.5, 0.5, -16.0, 26.0), (11.5, -3.5, -15.0, 22.0))):
            intake = np.array(wr_pt) + np.array([s * 2.0, dy_l * 0.4, 2.0])
            nozzle = intake + np.array([s * dx_l, dy_l * 0.6, dz_l - th_len * 0.35])
            m.span_box(f"thruster_nacelle_{q}_{li}_a", b_wr, intake, lerp3(intake, nozzle, 0.65), 4.6, 4.2, "armor", len_pad=0.8, pattern="armor_shingle", component="wing_spars", stage1=True)
            m.span_box(f"thruster_nacelle_{q}_{li}_b", b_mid, lerp3(intake, nozzle, 0.60), nozzle, 4.2, 3.8, "jet_dark", len_pad=0.6, tip_color=[242, 56, 62], tip_start=0.65, pattern="armor_shingle", component="wing_spars")
            # Controlled fiery exhaust nozzle cap inside nacelle rear mouth!
            m.box(f"thruster_nozzle_glow_{q}_{li}", b_mid, nozzle, (2.6, 2.4, 2.2), "crimson_glow", pattern="energy", component="wing_spars")

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: SUPERSONIC FALCON HELM, VENTRAL TURBINE & CHROME PLATES
    # =========================================================================
    m.box("snout_aeroblade", "head", (0.0, 39.2, 38.6), (3.8, 1.8, 6.8), "chrome_hi", rotation=(14.0, 0.0, 0.0), component="snout")
    m.span_box("raptor_beak_tip", "head", (0.0, 37.4, 42.2), (0.0, 35.6, 44.5), 2.8, 2.0, "jet_dark", component="snout")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_spoiler_{q}", "head", (s * 4.2, 37.8, 34.0), (2.0, 3.2, 6.2), "chrome_hi", rotation=(8.0, s * 14.0, -s * 12.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 4.0, 38.6, 36.2), (1.0, 1.4, 2.4), "jet_dark", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_crimson_{q}", "head", (s * 4.3, 38.6, 36.4), (0.7, 1.0, 1.8), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")
        # Towering swept silver supersonic delta-fin cranial crests (`[14],[15]`)
        m.tapered_arch(f"head_delta_crest_{q}", "head", [(s * 2.2, 40.2, 33.0), (s * 4.2, 44.5, 25.5), (s * 5.2, 47.8, 17.5)], [(2.2, 3.0), (1.8, 2.5), (0.9, 1.4)], "chrome_hi", tip_color=[218, 54, 62], tip_start=0.65, component="horns", stage1=True)
        m.tapered_arch(f"temple_intake_fin_{q}", "head", [(s * 4.4, 38.8, 31.8), (s * 8.0, 40.2, 26.2), (s * 10.8, 40.8, 20.5)], [(1.8, 2.4), (1.4, 1.8), (0.8, 1.0)], "armor", component="horns")

        for ti, tz in enumerate((35.5, 37.6, 39.6)):
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * (2.5 - 0.3 * ti), 36.0, tz), (0.7, 1.2, 0.8), "tooth", component="teeth")

        for kind, p0_r, p1_r, p2_r in (
            ("fore", (9.2, 22.8, 10.0), (12.8, 12.5, 6.2), (12.4, 3.5, 11.8)),
            ("hind", (8.2, 20.2, -13.5), (12.0, 11.2, -7.0), (11.6, 3.2, -14.8)),
        ):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0, p1, p2 = np.array([s * p0_r[0], p0_r[1], p0_r[2]]), np.array([s * p1_r[0], p1_r[1], p1_r[2]]), np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_shoulder_cowling_{q}", bu, p0 + np.array([s * 1.0, 1.5, 0.5]), (6.4, 6.2, 7.4), "chrome_hi", rotation=(8.0, s * 10.0, -s * 14.0), pattern="armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.16 + 0.22 * bi) + np.array([s * 1.8, 0.8, 1.2])
                m.box(f"{kind}_aero_greave_{q}_{bi}", bl, pt_b, (2.2, 3.4, 4.2), "armor", rotation=(18.0, s * 14.0, -s * 20.0), pattern="armor_shingle", component="armor")
            for di, ang_d in ((0, -18.0), (1, 0.0), (2, 18.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-1.8 + 1.8 * di), 2.2, p2[2] + 4.8])
                tb = ta + dv * 3.5
                tc = tb + dv * 2.8 - np.array([0.0, 0.8, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 1.8, 1.8, "jet_dark", len_pad=0.4, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.4, 1.4, "claw", len_pad=0.3, component="claws")

    # Chest Ventral Intake Turbine Core (`recessed crimson intake louvers under sternum!`)
    m.box("sternum_intake_grille", "chest", (0.0, 21.8, 18.5), (9.2, 4.8, 4.2), "jet_dark", rotation=(26.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
    m.box("sternum_reactor_core", "chest", (0.0, 21.6, 19.6), (6.4, 2.8, 2.4), "crimson_glow", rotation=(26.0, 0.0, 0.0), pattern="energy", component="armor")
    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 32.0, 25.5), 6.8, 5.4, 22.0),
        ("neck_1", (0.0, 27.0, 20.8), 8.2, 6.0, 18.0),
        ("chest", (0.0, 18.0, 11.2), 11.2, 7.0, 8.0),
        ("chest", (0.0, 17.0, 3.0), 11.8, 7.2, 0.0),
        ("pelvis", (0.0, 16.2, -5.5), 10.8, 6.8, 0.0),
        ("pelvis", (0.0, 16.5, -13.5), 9.2, 6.2, -4.0),
    )):
        m.box(f"ventral_keel_{ki}", kb, kpos, (kw, 2.2, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: SUPERSONIC CHROME PANELING, DELTA TAIL FINS & NECK LOUVERS
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for nb, p_a, p_b in (
            ("neck_1", (s * 4.4, 29.8, 17.0), (s * 3.8, 33.4, 22.8)),
            ("neck_2", (s * 3.6, 34.2, 24.0), (s * 3.2, 37.0, 29.5)),
        ):
            m.shingle_arc(f"neck_vent_plates_{nb}_{q}", nb, p_a, p_b, (s * 0.36, 0.24, -0.89), (s * 0.28, 0.28, -0.90), count=5, length=5.0, width=2.6, thickness=1.5, material="chrome_hi", alt_material="jet_dark", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"chest_jet_cowling_{q}", "chest", (s * 9.6, 26.8, 12.2), (s * 8.2, 24.0, -3.0), (s * 0.25, 0.14, -0.96), (s * 0.20, 0.12, -0.97), count=6, length=6.8, width=3.4, thickness=2.1, material="chrome_hi", alt_material="armor", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"pelvis_jet_plates_{q}", "pelvis", (s * 7.4, 22.0, -5.0), (s * 5.8, 20.5, -18.5), (s * 0.22, 0.12, -0.97), (s * 0.18, 0.10, -0.98), count=5, length=6.0, width=2.8, thickness=1.8, material="armor", alt_material="jet_dark", two_step=False, pattern="armor_shingle")

    for ni, (nb, npos, nw) in enumerate((
        ("neck_2", (0.0, 36.8, 26.0), 5.6),
        ("neck_1", (0.0, 32.5, 20.2), 6.4),
        ("chest", (0.0, 32.8, 10.5), 7.8),
        ("chest", (0.0, 31.8, 2.5), 8.0),
        ("pelvis", (0.0, 28.0, -5.5), 7.0),
        ("pelvis", (0.0, 25.5, -13.5), 6.0),
    )):
        m.box(f"dorsal_fuselage_plate_{ni}", nb, npos, (nw, 2.2, 6.2), "chrome_hi" if ni % 2 == 0 else "armor", rotation=(16.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        m.span_box(f"dorsal_rudder_fin_{ni}", nb, npos, npos + np.array([0.0, 4.2, -4.5]), 1.6, 2.4, "jet_dark", component="spines")

    for ti in range(6):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_rudder_fin_{ti}", tbones[ti], mid_t + np.array([0.0, 2.4, 0.0]), mid_t + np.array([0.0, 6.0, -5.0]), 1.5, 2.2, "chrome_hi", component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            # Horizontal delta fighter tail stabilizers near tail tip!
            span_w = 5.2 if ti >= 4 else 2.2
            m.span_box(f"tail_stabilizer_{ti}_{q}", tbones[ti], mid_t + np.array([s * 1.4, 0.0, 0.0]), mid_t + np.array([s * (2.2 + span_w), -0.4, -3.5]), 1.4, 1.4, "chrome_hi" if ti >= 4 else "armor", component="spines")

    return m.doc
