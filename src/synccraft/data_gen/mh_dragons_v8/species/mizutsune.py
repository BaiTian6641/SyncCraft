"""src/synccraft/data_gen/mh_dragons_v8/species/mizutsune.py

Monumental Sculpted Blockbench Reconstruction of MIZUTSUNE (`Bewitching Dancer`, Leviathan).

Built strictly from `mizutsune_wide_board.jpg` (`ZERO boxy armor slabs!`):
  - Graceful slender ivory-pearl & plum-violet fox-leviathan proportions with supple arched S-neck
    and extra-long lithe undulating whip tail (`~102u` nose to brush tip).
  - Narrow slender tapered vulpine skull, sharp snout whiskers, and TWO SWEEPING SCALLOPED
    CRIMSON-ROSE FAN EAR-FRILLS (`hood_l/r`) bordered in golden blossom spines.
  - Lush cascading violet-purple fluffy fur locks along chest, elbows, flanks, hips & thick plumed
    fox-brush tail tip (`fur_mane` + `fur_tuft` cutouts!).
  - Undulating translucent rose-coral blossom dorsal petal-fins (`sakura_fin` cutouts) arching along
    the entire spine cleanly down onto the tail.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def mizutsune_materials() -> dict[str, dict[str, Any]]:
    return {
        "scales": {"color": [224, 212, 210], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [178, 164, 174], "kind": "scales", "coat": "calm_flank"},
        "plum_dorsal": {"color": [146, 78, 126], "kind": "armor", "coat": "armor_shingle"},
        "fur": {"color": [126, 72, 132], "kind": "fur", "coat": "fur_mane"},
        "fur_light": {"color": [184, 124, 168], "kind": "fur", "coat": "fur_mane"},
        "belly": {"color": [232, 224, 216], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [214, 176, 108], "kind": "horn", "coat": "horn_keratin"},
        "sakura": {"color": [214, 86, 118], "kind": "membrane", "coat": "sakura_fin"},
        "sakura_back": {"color": [172, 60, 92], "kind": "membrane", "coat": "sakura_fin"},
        "claw": {"color": [112, 74, 86], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [154, 54, 82], "kind": "mouth"},
        "tooth": {"color": [238, 234, 226], "kind": "bone"},
        "eye": {"color": [250, 198, 56], "kind": "eye", "emissive": 0.86},
    }


def build_mizutsune(*, stage_cap: int = 3) -> dict[str, Any]:
    m = RigCanvas(
        "mizutsune",
        "Mizutsune",
        "Bewitching Bubble-Fox Leviathan",
        "Leviathan",
        variant_kind="canonical_flagship",
        wiki_url="https://monsterhunter.fandom.com/wiki/Mizutsune",
        materials=mizutsune_materials(),
        notes=[
            "Built from mizutsune_wide_board.jpg: narrow fox-leviathan snout, giant sweeping crimson-gold ear frill fans (`hood_l/r`), violet fur locks, rose dorsal petal fins, and plumed brush tail.",
        ],
        texels_per_unit=2,
    )

    m.bone("pelvis", "root", (0.0, 14.5, -14.0))
    m.bone("chest", "pelvis", (0.0, 17.5, 7.0))
    m.bone("neck_1", "chest", (0.0, 23.0, 18.0))
    m.bone("neck_2", "neck_1", (0.0, 31.0, 25.5))
    m.bone("neck_3", "neck_2", (0.0, 38.5, 32.5))
    m.bone("head", "neck_3", (0.0, 41.2, 37.5))
    m.bone("jaw", "head", (0.0, 39.0, 38.5))

    # =========================================================================
    # STAGE 1: LITHE S-CURVED FOX-SERPENT MASSES
    # =========================================================================
    torso_stations = [
        ((0.0, 14.0, -22.0), 10.2, 9.0, "pelvis"),
        ((0.0, 15.2, -12.0), 12.8, 11.2, "pelvis"),
        ((0.0, 17.0, -1.5), 14.8, 12.6, "chest"),
        ((0.0, 19.2, 9.0), 15.4, 13.0, "chest"),
        ((0.0, 22.0, 18.0), 11.8, 10.8, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.6, pattern="calm_flank", stage1=True)

    neck_stations = [
        ((0.0, 22.5, 17.5), 10.2, 9.6, "neck_1"),
        ((0.0, 29.8, 24.5), 8.6, 8.2, "neck_2"),
        ((0.0, 37.2, 31.5), 7.4, 7.0, "neck_3"),
        ((0.0, 41.2, 37.2), 6.8, 6.4, "head"),
    ]
    for i in range(len(neck_stations) - 1):
        p0, w0, h0, bn = neck_stations[i]
        p1, w1, h1, _ = neck_stations[i + 1]
        m.span_box(f"neck_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.8, pattern="calm_flank", stage1=True)

    # Narrow Aristocratic Vulpine Skull & Tapered Slender Snout (`mizutsune_wide_board.jpg`!)
    m.box("skull_cranium", "head", (0.0, 41.8, 40.5), (7.6, 4.8, 7.6), "plum_dorsal", rotation=(5.0, 0.0, 0.0), pattern="armor_shingle", component="head", stage1=True)
    m.box("vulpine_muzzle_mid", "head", (0.0, 40.8, 46.8), (5.0, 3.2, 6.2), "scales", rotation=(8.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("vulpine_nose_tip", "head", (0.0, 40.2, 51.2), (3.4, 2.2, 3.8), "scales", rotation=(10.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, 38.6, 44.5), (4.5, 2.0, 12.0), "belly", face_materials={"up": "mouth"}, rotation=(3.0, 0.0, 0.0), component="jaw", stage1=True)

    # Extra-Long Lithe 8-Bone Serpent-Fox Tail (`~56u` sweeping up into furred brush!)
    t_pts = [
        ((0.0, 14.0, -21.0), 9.4, 8.4),
        ((2.2, 13.2, -30.0), 8.0, 7.2),
        ((5.5, 13.0, -39.0), 6.8, 6.2),
        ((8.2, 14.0, -48.0), 5.6, 5.4),
        ((8.8, 16.0, -56.8), 4.8, 4.8),
        ((6.8, 19.0, -64.8), 5.6, 5.6),
        ((3.2, 22.0, -72.0), 6.2, 6.2),  # Fluffy brush flare!
        ((-0.8, 24.5, -78.8), 4.6, 4.6),
        ((-3.8, 26.2, -84.5), 2.8, 2.8),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(8)]
    for i in range(8):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "fur" if i >= 5 else "scales", len_pad=2.6, pattern="fur_mane" if i >= 5 else "calm_flank", stage1=(i % 2 == 0))

    for kind, pbone, p0_r, p1_r, p2_r, up_wh, lo_wh in (
        ("fore", "chest", (7.2, 15.8, 9.2), (12.0, 9.6, 6.5), (12.0, 2.8, 12.8), (5.8, 5.4), (4.6, 4.4)),
        ("hind", "pelvis", (6.8, 14.2, -13.5), (11.5, 8.8, -9.5), (11.5, 2.6, -15.8), (6.4, 6.0), (4.8, 4.5)),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0, p1, p2 = (s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), (s * p2_r[0], p2_r[1], p2_r[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.0, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.0, pattern="calm_flank", stage1=True)
            m.box(f"{kind}_paw_core_{q}", bp, (p2[0], 2.0, p2[2] + 2.2), (6.4, 2.6, 7.0), "scales_deep", stage1=True)

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: SWEEPING FOX EAR-FRILLS, FLOWER DORSAL PETAL FINS & FUR LOCKS
    # =========================================================================
    m.box("nasal_crest_line", "head", (0.0, 42.6, 44.5), (3.8, 1.6, 6.8), "plum_dorsal", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("chin_silk_tuft", "jaw", (0.0, 37.5, 45.2), (3.6, 1.8, 4.8), "fur_light", pattern="fur_mane", component="jaw")

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_scale_{q}", "head", (s * 4.0, 41.2, 39.8), (1.8, 2.8, 5.4), "plum_dorsal", rotation=(6.0, s * 12.0, -s * 10.0), pattern="armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 3.8, 42.2, 42.2), (0.9, 1.3, 2.3), "mouth", rotation=(6.0, s * 8.0, 0.0), component="eyes")
        m.box(f"eye_fox_{q}", "head", (s * 4.1, 42.2, 42.4), (0.6, 0.9, 1.7), "eye", rotation=(6.0, s * 8.0, 0.0), pattern="eye_pixel", component="eyes")
        # Iconic Swept Golden Fox Whiskers / Antler Horns (`mizutsune_wide_board.jpg`)
        m.tapered_arch(f"fox_whisker_horn_{q}", "head", [(s * 2.4, 44.0, 38.5), (s * 4.2, 48.2, 33.2), (s * 5.2, 52.0, 27.5)], [(1.8, 1.8), (1.4, 1.4), (0.8, 0.8)], "horn", tip_color=[228, 126, 146], component="horns", stage1=True)
        m.span_box(f"cheek_whisker_{q}", "head", (s * 3.4, 40.0, 46.2), (s * 8.8, 39.2, 40.5), 1.0, 1.0, "fur_light", component="horns")

        # Iconic COLOSSAL SWEEPING FAN EAR-FRILLS (`hood_l/r` articulated wide rose-gold ear fans!)
        hb = m.bone(f"hood_{q}", "neck_3", (s * 3.8, 39.5, 34.5))
        fan_rays = (
            ((s * 3.8, 42.2, 36.2), (s * 13.5, 50.8, 27.5)),
            ((s * 3.8, 40.4, 35.0), (s * 16.5, 45.2, 24.5)),
            ((s * 3.8, 38.2, 33.6), (s * 15.8, 38.2, 22.8)),
            ((s * 3.4, 36.2, 32.0), (s * 12.6, 31.8, 21.8)),
        )
        for li, (r_pt, t_pt) in enumerate(fan_rays):
            m.span_box(f"ear_fan_rib_{q}_{li}", hb, r_pt, t_pt, 1.6, 1.4, "horn", tip_color=[228, 118, 142], tip_start=0.55, component="horns")
            if li < 3:
                nxt_r, nxt_t = fan_rays[li + 1]
                m.cutout_panel(f"ear_fan_petal_{q}_{li}", hb, [v3(r_pt), v3(t_pt), v3(nxt_t), v3(nxt_r)], "sakura", back_material="sakura_back", pattern="ear_frill_cutout", component="membrane")

        for ti, tz in enumerate((42.5, 45.0, 47.4, 49.6)):
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * (2.2 - 0.25 * ti), 39.4, tz), (0.7, 1.0, 0.7), "tooth", component="teeth")

        # Sleek Fur-Cuffed Claws + Flowing Elbow / Hip Fur Ribbons!
        for kind, p0_r, p1_r, p2_r in (
            ("fore", (7.2, 15.8, 9.2), (12.0, 9.6, 6.5), (12.0, 2.8, 12.8)),
            ("hind", (6.8, 14.2, -13.5), (11.5, 8.8, -9.5), (11.5, 2.6, -15.8)),
        ):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0, p1, p2 = np.array([s * p0_r[0], p0_r[1], p0_r[2]]), np.array([s * p1_r[0], p1_r[1], p1_r[2]]), np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_fur_cuff_{q}", bu, p0 + np.array([s * 1.0, 1.2, 0.5]), (6.2, 5.8, 6.8), "fur", rotation=(10.0, s * 10.0, -s * 14.0), pattern="fur_mane")
            m.span_box(f"{kind}_silk_ribbon_{q}", bl, p1, p1 + np.array([s * 3.8, -1.5, -6.5]), 2.6, 1.8, "fur_light", pattern="fur_mane", component="surface")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.16 + 0.22 * bi) + np.array([s * 1.5, 0.6, 0.8])
                m.box(f"{kind}_plum_scale_{q}_{bi}", bl, pt_b, (1.8, 2.6, 3.6), "plum_dorsal" if bi % 2 == 0 else "fur", rotation=(16.0, s * 14.0, -s * 18.0), pattern="fur_mane" if bi % 2 == 1 else "armor_shingle", component="armor")
            for di, ang_d in ((0, -22.0), (1, -6.0), (2, 10.0), (3, 24.0)):
                yaw = math.radians(s * ang_d)
                dv = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                ta = np.array([p2[0] + s * (-2.2 + 1.5 * di), 1.8, p2[2] + 4.4])
                tb = ta + dv * 3.4
                tc = tb + dv * 2.4 - np.array([0.0, 0.6, 0.0])
                m.span_box(f"{kind}_toe_{q}_{di}", bp, ta, tb, 1.5, 1.4, "scales_deep", len_pad=0.4, component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, tb, tc, 1.2, 1.2, "claw", len_pad=0.3, component="claws")

    # Proud Pale Cream Throat & Belly Ribbons (`zero boxy corners!`)
    for bi, (bn, pos, pw, kd, rx) in enumerate((
        ("neck_3", (0.0, 35.8, 32.0), 5.6, 5.2, 34.0),
        ("neck_2", (0.0, 29.0, 26.2), 6.6, 5.6, 40.0),
        ("neck_1", (0.0, 21.8, 19.5), 7.8, 6.0, 32.0),
        ("chest", (0.0, 14.5, 11.8), 10.2, 6.5, 14.0),
        ("chest", (0.0, 11.8, 4.0), 11.4, 6.8, 4.0),
        ("chest", (0.0, 11.2, -3.5), 10.6, 6.4, 0.0),
        ("pelvis", (0.0, 10.8, -11.5), 9.6, 6.2, 0.0),
        ("pelvis", (0.0, 11.0, -19.0), 8.2, 5.8, -4.0),
    )):
        m.box(f"belly_scute_{bi}", bn, pos, (pw, 2.0, kd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    # Signature Sweeping Translucent Rose-Blossom Dorsal Petal Fins (`mizutsune_wide_board.jpg`)
    for pi, (pb, p_a, p_b, h_f) in enumerate((
        ("neck_2", (0.0, 32.5, 24.5), (0.0, 38.5, 31.0), 7.5),
        ("neck_1", (0.0, 25.0, 17.5), (0.0, 31.8, 24.0), 9.2),
        ("chest", (0.0, 23.0, 11.0), (0.0, 24.5, 1.0), 11.5),
        ("chest", (0.0, 22.8, 0.0), (0.0, 21.2, -9.5), 11.0),
        ("pelvis", (0.0, 20.5, -10.5), (0.0, 19.0, -21.0), 9.5),
    )):
        m.span_box(f"dorsal_petal_rib_{pi}", pb, p_a, lerp3(p_a, p_b, 0.6) + np.array([0.0, h_f, -2.5]), 1.6, 2.0, "horn", tip_color=[228, 112, 140], component="spines")
        m.cutout_panel(f"dorsal_sakura_fin_{pi}", pb, [v3(p_a), v3(p_b), v3(p_b) + np.array([0.0, h_f * 0.85, -1.5]), v3(p_a) + np.array([0.0, h_f, -1.0])], "sakura", back_material="sakura_back", pattern="sakura_fin", component="membrane")

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: LUSH VIOLET FUR MANE COLLARS, FLANK SILK & PLUMED BRUSH TAIL
    # =========================================================================
    for s, q in ((1.0, "l"), (-1.0, "r")):
        for nb, p_a, p_b in (
            ("neck_1", (s * 4.5, 23.0, 17.8), (s * 3.8, 29.0, 23.8)),
            ("neck_2", (s * 3.6, 30.2, 24.8), (s * 3.0, 36.2, 30.8)),
            ("neck_3", (s * 2.8, 37.0, 31.8), (s * 2.4, 40.4, 36.2)),
        ):
            m.shingle_arc(f"neck_fur_ruff_{nb}_{q}", nb, p_a, p_b, (s * 0.36, 0.24, -0.89), (s * 0.28, 0.28, -0.90), count=5, length=5.2, width=2.8, thickness=1.8, material="fur", alt_material="fur_light", two_step=False, pattern="fur_mane")
        # Layered shoulder silk cape, dorsal rib ribbons & hip fur cascades (`+52` elements!)
        m.shingle_arc(f"chest_fur_cascade_{q}", "chest", (s * 7.8, 20.5, 11.5), (s * 6.8, 18.2, -3.5), (s * 0.28, 0.16, -0.94), (s * 0.22, 0.14, -0.96), count=6, length=6.8, width=3.2, thickness=2.0, material="fur", alt_material="plum_dorsal", two_step=False, pattern="fur_mane")
        m.shingle_arc(f"chest_upper_silk_{q}", "chest", (s * 6.4, 22.8, 10.5), (s * 5.8, 20.5, -2.5), (s * 0.28, 0.28, -0.91), (s * 0.22, 0.22, -0.94), count=5, length=6.0, width=2.8, thickness=1.8, material="plum_dorsal", alt_material="fur_light", two_step=False, pattern="armor_shingle")
        m.shingle_arc(f"chest_ventral_silk_{q}", "chest", (s * 6.0, 15.6, 10.5), (s * 5.5, 13.8, -2.5), (s * 0.42, -0.22, -0.88), (s * 0.38, -0.20, -0.90), count=5, length=5.2, width=2.4, thickness=1.6, material="belly", alt_material="fur_light", two_step=False, pattern="belly_scute")
        m.shingle_arc(f"pelvis_fur_skirt_{q}", "pelvis", (s * 6.6, 17.5, -5.0), (s * 5.2, 15.8, -19.5), (s * 0.25, 0.14, -0.95), (s * 0.20, 0.12, -0.97), count=6, length=6.4, width=3.0, thickness=1.9, material="fur_light", alt_material="fur", two_step=False, pattern="fur_mane")
        for kind, p0_r, p1_r in (("fore", (7.2, 15.8, 9.2), (12.0, 9.6, 6.5)), ("hind", (6.8, 14.2, -13.5), (11.5, 8.8, -9.5))):
            for fi in range(3):
                fp = lerp3((s * p0_r[0], p0_r[1], p0_r[2]), (s * p1_r[0], p1_r[1], p1_r[2]), 0.20 + 0.28 * fi) + np.array([s * 2.2, 0.6, 0.4])
                m.box(f"{kind}_upper_silk_tuft_{q}_{fi}", f"{kind}_upper_{q}", fp, (2.2, 3.0, 4.2), "fur_light" if fi % 2 == 0 else "plum_dorsal", rotation=(14.0, s * 14.0, -s * 18.0), pattern="fur_mane", component="surface")
        m.cutout_panel(f"flank_fur_fringes_{q}", "pelvis", [v3((s * 6.8, 17.8, -3.0)), v3((s * 5.4, 16.0, -20.0)), v3((s * 5.0, 11.0, -19.0)), v3((s * 6.4, 12.0, -3.0))], "fur", pattern="fur_tuft", component="surface")

    # Plumed Blossom Tail Fins + Fluffy Fox-Brush Tail Tip + Dorsal Petal Spines (`tail_1..8`)
    for ti in range(8):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.span_box(f"tail_petal_spine_{ti}", tbones[ti], mid_t + np.array([0.0, 2.2, 0.0]), mid_t + np.array([0.0, 6.8 - 0.25 * ti, -3.6]), 1.6, 2.0, "horn" if ti < 5 else "fur", tip_color=[228, 118, 142] if ti < 5 else None, component="spines")
        m.cutout_panel(f"tail_sakura_fin_{ti}", tbones[ti], [pa + np.array([0.0, 2.0, 0.0]), pb + np.array([0.0, 2.0, 0.0]), pb + np.array([0.0, 7.2, -1.2]), pa + np.array([0.0, 6.5, -0.8])], "sakura", back_material="sakura_back", pattern="sakura_fin", component="membrane")
        m.box(f"tail_ventral_silk_{ti}", tbones[ti], mid_t + np.array([0.0, -2.8 + 0.15 * ti, 0.0]), (max(2.2, 6.2 - 0.35 * ti), 1.8, 6.8), "fur_light" if ti >= 4 else "belly", rotation=(-6.0, 0.0, 0.0), pattern="fur_mane" if ti >= 4 else "belly_scute", component="surface")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            fmat = "fur" if ti >= 4 else "plum_dorsal"
            m.box(f"tail_plume_lock_{ti}_{q}", tbones[ti], mid_t + np.array([s * max(1.4, 3.8 - 0.2 * ti if ti < 5 else 3.4), 0.2, 0.0]), (2.2 if ti >= 5 else 1.6, 2.8 if ti >= 5 else 2.0, 6.6), fmat, rotation=(8.0, -s * 16.0, -s * 18.0), pattern="fur_mane" if ti >= 4 else "armor_shingle", component="surface")

    return m.doc
