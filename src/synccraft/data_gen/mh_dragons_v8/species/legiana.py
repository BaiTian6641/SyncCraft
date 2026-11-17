"""src/synccraft/data_gen/mh_dragons_v8/species/legiana.py

Monumental Sculpted Blockbench Reconstruction of LEGIANA & SHRIEKING LEGIANA (Flying Wyverns):
  - `legiana`: Elegant Aerodynamic Apex (sleek dark-indigo mantis/falcon head with twin long swept V-ear
    antennae crests, smooth cream breast keel, HUGE broad rounded butterfly/moth wing sails with charcoal
    chequered veins & vivid azure leading wingtip fans, and broad 5-blade flared aerodynamic tail fan!).
  - `shrieking_legiana`: Frost-Crusted Variant (storm-charcoal hide, taller frosted skull antennae, cracked
    pale-ice frost mantle crystals (`crystal_facet`) erupting across neck, shoulders, chest, and leading wing
    edges, plus pale ice-blue crystal tail fan blades!).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def legiana_materials(variant: str = "canonical") -> dict[str, dict[str, Any]]:
    if variant == "canonical":
        return {
            "scales": {"color": [42, 48, 72], "kind": "scales", "coat": "calm_flank"},
            "scales_deep": {"color": [28, 32, 52], "kind": "scales", "coat": "calm_flank"},
            "armor": {"color": [54, 62, 92], "kind": "armor", "coat": "armor_shingle"},
            "belly": {"color": [224, 214, 184], "kind": "belly", "coat": "belly_scute"},
            "horn": {"color": [36, 44, 72], "kind": "horn", "coat": "horn_keratin"},
            "crest_blue": {"color": [72, 138, 218], "kind": "armor", "coat": "armor_shingle"},
            "wing_mem": {"color": [206, 184, 138], "kind": "membrane", "coat": "wing_legiana"},
            "wing_back": {"color": [156, 136, 102], "kind": "membrane", "coat": "wing_legiana"},
            "tail_fan": {"color": [64, 108, 178], "kind": "membrane", "coat": "wing_legiana"},
            "claw": {"color": [44, 48, 66], "kind": "horn", "coat": "horn_keratin"},
            "mouth": {"color": [132, 52, 74], "kind": "mouth"},
            "tooth": {"color": [234, 230, 218], "kind": "bone"},
            "eye": {"color": [248, 206, 62], "kind": "eye", "emissive": 0.85},
        }
    return {  # shrieking
        "scales": {"color": [62, 66, 82], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [42, 44, 58], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [88, 94, 112], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [220, 224, 232], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [54, 58, 76], "kind": "horn", "coat": "horn_keratin"},
        "crest_blue": {"color": [146, 196, 238], "kind": "crystal", "coat": "crystal_facet"},
        "frost": {"color": [188, 218, 242], "kind": "crystal", "coat": "crystal_facet"},
        "wing_mem": {"color": [174, 184, 202], "kind": "membrane", "coat": "wing_legiana"},
        "wing_back": {"color": [128, 138, 158], "kind": "membrane", "coat": "wing_legiana"},
        "tail_fan": {"color": [148, 198, 236], "kind": "membrane", "coat": "wing_legiana"},
        "claw": {"color": [48, 52, 66], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [116, 56, 88], "kind": "mouth"},
        "tooth": {"color": [238, 240, 246], "kind": "bone"},
        "eye": {"color": [124, 224, 255], "kind": "eye", "emissive": 0.88},
    }


def _build_legiana_specimen(variant: str = "canonical", *, stage_cap: int = 3) -> dict[str, Any]:
    is_shr = variant == "shrieking"
    mid = "shrieking_legiana" if is_shr else "legiana"
    title = "Shrieking Legiana" if is_shr else "Legiana"
    ep = "Frostbound Shrieking Empress" if is_shr else "Elegant Apex of the Highlands"
    vkind = "variant" if is_shr else "canonical_flagship"
    par = "legiana" if is_shr else None
    wiki = f"https://monsterhunter.fandom.com/wiki/{'Shrieking_Legiana' if is_shr else 'Legiana'}"

    m = RigCanvas(
        mid,
        title,
        ep,
        "Flying Wyvern",
        variant_kind=vkind,
        parent_species_id=par,
        wiki_url=wiki,
        materials=legiana_materials(variant),
        notes=[f"Slender avian-mantis flying wyvern built from `{mid}_wide_board.jpg` with broad rounded wing sails & 5-blade tail rudder fan."],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 22.0, -7.0))
    m.bone("chest", "pelvis", (0.0, 27.5, 6.5))
    m.bone("neck_1", "chest", (0.0, 33.5, 16.5))
    m.bone("neck_2", "neck_1", (0.0, 40.5, 22.8))
    m.bone("head", "neck_2", (0.0, 45.2, 28.5))
    m.bone("jaw", "head", (0.0, 42.8, 29.2))

    # =========================================================================
    # STAGE 1: LITHE AERIAL SILHOUETTE (LONG GRACEFUL NECK & BROAD FAN WINGS)
    # =========================================================================
    torso_stations = [
        ((0.0, 21.0, -13.5), 9.6, 8.8, "pelvis"),
        ((0.0, 23.5, -3.5), 11.8, 10.6, "pelvis"),
        ((0.0, 27.2, 6.2), 14.5, 12.8, "chest"),
        ((0.0, 31.0, 14.2), 13.0, 11.8, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.6, pattern="calm_flank", stage1=True)

    for i, (bn, a, b, w, h) in enumerate((
        ("neck_1", (0.0, 32.8, 15.8), (0.0, 39.2, 21.8), 8.8, 8.4),
        ("neck_2", (0.0, 39.2, 21.8), (0.0, 45.0, 28.2), 7.6, 7.2),
    )):
        m.span_box(f"neck_core_{i}", bn, a, b, w, h, "scales", len_pad=3.0, pattern="calm_flank", stage1=True)

    # Narrow Sleek Mantis/Peregrine Skull
    m.box("skull_cranium", "head", (0.0, 46.0, 31.2), (7.6, 5.2, 7.4), "armor", rotation=(6.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("falcon_muzzle", "head", (0.0, 45.0, 36.8), (4.6, 3.4, 5.6), "scales", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 42.8, 34.4), (4.5, 2.0, 8.8), "belly", face_materials={"up": "mouth"}, rotation=(4.0, 0.0, 0.0), component="jaw", stage1=True)

    # Agile 7-Bone Rudder Tail
    t_pts = [
        ((0.0, 21.2, -13.0), 8.2, 7.6),
        ((0.6, 20.5, -22.0), 6.8, 6.4),
        ((1.8, 20.4, -31.0), 5.6, 5.4),
        ((2.8, 21.2, -40.0), 4.8, 4.6),
        ((2.4, 22.8, -49.0), 4.0, 3.8),
        ((1.2, 24.8, -57.8), 3.4, 3.2),
        ((0.0, 26.8, -66.0), 3.0, 2.8),
        ((-0.8, 28.2, -73.0), 2.2, 2.2),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(7)]
    for i in range(7):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.4, pattern="calm_flank", stage1=(i % 2 == 0))

    # Slender Raptor Hindlegs
    for s, q in ((1.0, "l"), (-1.0, "r")):
        p0, p1, p2 = (s * 6.8, 21.0, -5.5), (s * 10.2, 11.8, -0.5), (s * 10.0, 3.5, -7.8)
        bu = m.bone(f"hind_upper_{q}", "pelvis", p0)
        bl = m.bone(f"hind_lower_{q}", bu, p1)
        bp = m.bone(f"hind_paw_{q}", bl, p2)
        m.span_box(f"hind_thigh_{q}", bu, p0, p1, 6.8, 6.6, "scales", len_pad=2.0, pattern="calm_flank", stage1=True)
        m.span_box(f"hind_shin_{q}", bl, p1, p2, 4.6, 4.5, "scales", len_pad=2.0, pattern="leg_bands", stage1=True)
        m.box(f"hind_ankle_{q}", bp, (p2[0], 2.4, p2[2] + 2.0), (6.0, 3.0, 6.5), "scales_deep", stage1=True)

    # Enormous Broad Rounded Moth/Butterfly Wing Sails (`5` fingers -> `12` wing panels per side!)
    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.winged_sail_rig(
            s,
            "chest",
            (s * 6.5, 30.2, 7.5),
            (s * 20.0, 42.0, 4.5),
            (s * 32.0, 52.0, 14.5),
            (
                (s * 64.0, 58.0, 8.0),
                (s * 60.0, 46.0, -4.5),
                (s * 50.0, 34.0, -14.5),
                (s * 37.0, 24.0, -21.0),
                (s * 22.0, 19.0, -23.0),
            ),
            (s * 6.0, 22.0, -7.5),
            spar_mat="frost" if is_shr else "crest_blue",
            hub_mat="armor",
            claw_mat="claw",
            mem_mat="wing_mem",
            mem_back_mat="wing_back",
            mem_pattern="wing_legiana",
            spar_w=2.8,
            arm_w=4.2,
            leading_scales=7,
            tattered_outer=False,
        )

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: V-CREST HEAD ANTENNAE, MANTIS FACE, TAIL RUDDER FAN & FROST SPIRES
    # =========================================================================
    m.box("beak_ridge", "head", (0.0, 46.8, 36.2), (3.8, 1.8, 5.8), "armor", rotation=(14.0, 0.0, 0.0), component="snout")
    m.span_box("raptor_beak_hook", "head", (0.0, 45.2, 39.2), (0.0, 43.2, 41.2), 2.8, 2.0, "horn", component="snout")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_hood_{q}", "head", (s * 4.2, 44.8, 31.8), (2.2, 3.4, 5.8), "armor", rotation=(8.0, s * 14.0, -s * 12.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 3.8, 46.2, 33.8), (1.0, 1.4, 2.4), "mouth", rotation=(8.0, s * 10.0, 0.0), component="eyes")
        m.box(f"eye_slit_{q}", "head", (s * 4.1, 46.2, 34.0), (0.7, 1.0, 1.8), "eye", rotation=(8.0, s * 10.0, 0.0), pattern="eye_pixel", component="eyes")

        # Tall Iconic Swept-Back V-Crest Ear-Antennae Blades (`legiana_wide_board.jpg` `[1],[2],[12]`)
        v_scale = 1.18 if is_shr else 1.0
        m.tapered_arch(f"v_antennae_crest_{q}", "head", [(s * 2.4, 47.6, 30.8), (s * 5.2 * v_scale, 53.5 * v_scale, 25.0), (s * 7.6 * v_scale, 59.5 * v_scale, 18.8)], [(2.2, 3.0), (1.8, 2.4), (1.0, 1.4)], "frost" if is_shr else "horn", tip_color=[168, 218, 250] if is_shr else [92, 162, 236], tip_start=0.35, component="horns", stage1=True)
        # Flared cobalt/frost ear-flank winglet below antennae
        m.tapered_arch(f"cheek_fan_blade_{q}", "head", [(s * 4.4, 44.8, 29.8), (s * 8.4, 45.8, 25.2), (s * 11.4, 46.2, 21.2)], [(1.8, 2.6), (1.4, 2.0), (0.8, 1.2)], "crest_blue", component="horns")

        # Slender Raptor Legs + 3 Fore-Talons
        bp = f"hind_paw_{q}"
        for bi in range(4):
            pt_b = lerp3((s * 10.2, 11.8, -0.5), (s * 10.0, 3.5, -7.8), 0.16 + 0.22 * bi) + np.array([s * 1.4, 0.6, 1.2])
            m.box(f"shin_vane_{q}_{bi}", f"hind_lower_{q}", pt_b, (1.8, 2.6, 3.4), "crest_blue", rotation=(16.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
        for di, ang_d in ((0, -20.0), (1, 0.0), (2, 20.0)):
            yaw = math.radians(s * ang_d)
            dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
            ta = np.array([s * (10.0 + (-1.6 + 1.6 * di) * s), 2.0, -4.0])
            tb = ta + dv * 4.0
            tc = tb + dv * 3.0 - np.array([0.0, 0.9, 0.0])
            m.span_box(f"talon_toe_{q}_{di}", bp, ta, tb, 1.8, 1.8, "scales_deep", len_pad=0.4, component="claws")
            m.span_box(f"talon_claw_{q}_{di}", bp, tb, tc, 1.4, 1.5, "claw", len_pad=0.3, component="claws")

    # Cream/Ice Throat & Sternum Keel
    for ki, (kb, kpos, kw, kd, rx) in enumerate((
        ("neck_2", (0.0, 39.0, 22.8), 5.8, 5.2, 22.0),
        ("neck_1", (0.0, 33.2, 18.2), 7.2, 5.8, 18.0),
        ("chest", (0.0, 28.8, 15.2), 8.6, 6.4, 22.0),
        ("chest", (0.0, 24.5, 12.0), 9.6, 6.8, 14.0),
        ("chest", (0.0, 21.8, 5.8), 11.2, 7.0, 6.0),
        ("chest", (0.0, 21.2, -1.0), 10.4, 6.8, 0.0),
        ("pelvis", (0.0, 19.6, -8.5), 9.2, 6.4, -2.0),
    )):
        m.box(f"belly_scute_{ki}", kb, kpos, (kw, 2.2, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    # Iconic Broad 5-Petal Aerodynamic Tail Rudder Fan (`tail_5..7` — zero Rathalos club!)
    fan_bone = tbones[6]
    fan_hub = np.array([-0.5, 27.6, -70.5])
    for fi, ang_f in enumerate((-52.0, -26.0, 0.0, 26.0, 52.0)):
        rad = math.radians(ang_f)
        tip_f = fan_hub + np.array([math.sin(rad) * 14.0, abs(math.cos(rad)) * 2.5, -math.cos(rad) * 14.0])
        m.span_box(f"tail_fan_rib_{fi}", fan_bone, fan_hub, tip_f, 1.5, 1.5, "frost" if is_shr else "horn", component="spines")
        if fi < 4:
            rad_n = math.radians(-52.0 + 26.0 * (fi + 1))
            tip_n = fan_hub + np.array([math.sin(rad_n) * 14.0, abs(math.cos(rad_n)) * 2.5, -math.cos(rad_n) * 14.0])
            m.cutout_panel(f"tail_fan_web_{fi}", fan_bone, [fan_hub, tip_f, tip_n, lerp3(fan_hub, tip_n, 0.15)], "tail_fan", back_material="tail_fan", pattern="sakura_fin", component="membrane")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: FEATHERED NECK MANTLE, DORSAL FINS & SHRIEKING FROST CRYSTALS
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        # Streamlined neck gill/feather collar plates along graceful avian neck (`+20` shingles!)
        for nb, p_a, p_b in (
            ("neck_1", (s * 3.8, 33.5, 16.5), (s * 3.4, 38.8, 21.6)),
            ("neck_2", (s * 3.2, 39.8, 22.8), (s * 2.8, 44.2, 27.8)),
        ):
            m.shingle_arc(f"neck_gill_plumes_{nb}_{q}", nb, p_a, p_b, (s * 0.36, 0.24, -0.89), (s * 0.28, 0.28, -0.90), count=5, length=5.0, width=2.6, thickness=1.5, material="crest_blue", alt_material="armor", two_step=False, pattern="armor_shingle")
        # Wing forearm airfoil vanes + shoulder blade plumes
        m.box(f"wing_shoulder_plume_{q}", "chest", (s * 7.2, 31.2, 8.8), (4.8, 3.4, 7.6), "crest_blue" if not is_shr else "frost", rotation=(14.0, s * 16.0, -s * 22.0), pattern="armor_shingle", component="armor")
        for wi in range(5):
            w_pt = lerp3((s * 6.5, 30.2, 7.5), (s * 20.0, 42.0, 4.5), 0.12 + 0.20 * wi)
            m.box(f"wing_arm_vane_{q}_{wi}", f"wing_{q}", w_pt + np.array([0.0, 1.8, 1.0]), (3.2, 2.2, 4.4), "armor" if wi % 2 == 0 else "crest_blue", rotation=(12.0, s * 18.0, 0.0), pattern="armor_shingle", component="armor")
        # Thigh aerodynamic vanes & chest ventral rib plates
        for thi in range(4):
            th_p = lerp3((s * 6.8, 21.0, -5.5), (s * 10.2, 11.8, -0.5), 0.18 + 0.22 * thi) + np.array([s * 2.2, 0.6, 0.5])
            m.box(f"thigh_aero_plate_{q}_{thi}", f"hind_upper_{q}", th_p, (2.6, 3.4, 4.2), "crest_blue" if thi % 2 == 0 else "armor", rotation=(14.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="armor")
        # Sleek shoulder/wing-arm mantle plumes framing the chest cleanly
        m.shingle_arc(f"shoulder_mantle_{q}", "chest", (s * 7.8, 29.8, 12.2), (s * 6.5, 26.2, -3.5), (s * 0.28, 0.22, -0.93), (s * 0.22, 0.18, -0.95), count=6, length=6.6, width=3.2, thickness=1.8, material="frost" if is_shr else "armor", alt_material="belly" if is_shr else "scales", two_step=False, pattern="crystal_facet" if is_shr else "armor_shingle")
        m.shingle_arc(f"chest_rib_plumes_{q}", "chest", (s * 5.8, 22.8, 11.0), (s * 5.4, 21.0, -1.5), (s * 0.44, -0.22, -0.87), (s * 0.38, -0.20, -0.89), count=5, length=4.8, width=2.4, thickness=1.5, material="belly", alt_material="armor", two_step=False, pattern="belly_scute")
        m.shingle_arc(f"pelvis_flank_plumes_{q}", "pelvis", (s * 6.4, 23.0, -4.5), (s * 5.0, 21.4, -14.0), (s * 0.24, 0.14, -0.96), (s * 0.18, 0.12, -0.97), count=5, length=5.6, width=2.6, thickness=1.6, material="armor", alt_material="scales", two_step=False, pattern="armor_shingle")

        if is_shr:
            # Shrieking Legiana Heavy Frost-Ice Crystal Mantle across Neck, Shoulder & Outer Wing Arm (`40+` extra crystal planes/spires!)
            for cr_i, (cbn, cpos, csize) in enumerate((
                ("neck_2", (s * 3.6, 43.2, 23.8), (2.2, 5.2, 2.8)),
                ("neck_1", (s * 4.4, 37.5, 17.8), (2.6, 6.2, 3.2)),
                ("chest", (s * 8.2, 34.2, 10.5), (3.4, 7.6, 4.2)),
                (f"wing_fore_{q}", (s * 23.5, 45.2, 5.5), (3.2, 6.8, 3.6)),
                (f"wing_wrist_{q}", (s * 35.5, 55.2, 15.0), (3.0, 7.2, 3.8)),
            )):
                m.box(f"shrieking_frost_cluster_{q}_{cr_i}", cbn, cpos, csize, "frost", rotation=(-36.0, s * 14.0, -s * 24.0), pattern="crystal_facet", component="spines")

    # Dorsal neck/back ridge plumes + Swept Dorsal Tail Rudder Vanes
    for bi, (db, dp, dw) in enumerate((
        ("neck_2", (0.0, 44.5, 25.0), 4.4),
        ("neck_1", (0.0, 38.2, 19.2), 5.0),
        ("chest", (0.0, 34.2, 11.5), 6.2),
        ("chest", (0.0, 31.8, 3.5), 6.4),
        ("pelvis", (0.0, 27.8, -4.5), 5.6),
    )):
        m.box(f"dorsal_plume_plate_{bi}", db, dp, (dw, 1.8, 5.4), "crest_blue" if not is_shr else "frost", rotation=(20.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        m.span_box(f"dorsal_crest_vane_{bi}", db, dp, dp + np.array([0.0, 4.0, -3.5]), 1.6, 2.2, "frost" if is_shr else "crest_blue", component="spines")

    for ti in range(6):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_keel_fin_{ti}", tbones[ti], mid_t + np.array([0.0, 2.0, 0.0]), mid_t + np.array([0.0, 4.6, -3.5]), 1.4, 2.0, "frost" if is_shr else "crest_blue", component="spines")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(f"tail_side_plume_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.0, 3.2 - 0.25 * ti), 0.0, 0.0]), (1.4, 2.0, 5.4), "armor", rotation=(8.0, -s * 14.0, -s * 16.0), pattern="armor_shingle", component="armor")

    return m.doc


def build_legiana(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_legiana_specimen("canonical", stage_cap=stage_cap)


def build_shrieking_legiana(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_legiana_specimen("shrieking", stage_cap=stage_cap)
