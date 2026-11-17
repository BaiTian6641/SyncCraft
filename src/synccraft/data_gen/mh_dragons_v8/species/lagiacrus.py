"""src/synccraft/data_gen/mh_dragons_v8/species/lagiacrus.py

Monumental Sculpted Blockbench Reconstruction of the LAGIACRUS Trio (Leviathan):
  - `lagiacrus`: Lord of the Seas (ocean teal hide, broad swept rust shock-shell pylons, amber-webbed
    4-lobe cheek hood, continuous cream throat/sternum scutes, swept deep-blue-to-ochre horns, swimming paddle).
  - `ivory_lagiacrus`: White Conqueror of Land (pearl-ivory armor, heavier elevated terrestrial shoulders,
    wider crocodilian brow, tall sapphire shock crystal clusters, storm-slate underbelly).
  - `abyssal_lagiacrus`: Master of the Deep (abyssal obsidian-indigo carapace, elongated devil horns,
    expanded 5-lobe electric-cyan rimmed cheek hood & carpal fins, high abyssal paddle sail, crimson eyes).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl_primitives import RigCanvas, lerp3, unit, v3


def lagiacrus_materials(variant: str = "canonical") -> dict[str, dict[str, Any]]:
    if variant == "canonical":
        return {
            "scales": {"color": [34, 114, 140], "kind": "scales", "coat": "calm_flank"},
            "scales_deep": {"color": [24, 82, 108], "kind": "scales", "coat": "calm_flank"},
            "armor": {"color": [26, 68, 98], "kind": "armor", "coat": "armor_shingle"},
            "rust_organ": {"color": [166, 80, 44], "kind": "horn", "coat": "dorsal_blade"},
            "rust_ridge": {"color": [196, 116, 62], "kind": "horn", "coat": "dorsal_blade"},
            "rust_socket": {"color": [68, 52, 48], "kind": "armor", "coat": "armor_shingle"},
            "belly": {"color": [216, 200, 164], "kind": "belly", "coat": "belly_scute"},
            "horn": {"color": [38, 90, 116], "kind": "horn", "coat": "horn_keratin"},
            "hood_fin": {"color": [42, 120, 144], "kind": "armor", "coat": "armor_shingle"},
            "claw": {"color": [94, 76, 62], "kind": "horn", "coat": "horn_keratin"},
            "mouth": {"color": [132, 50, 58], "kind": "mouth"},
            "tooth": {"color": [230, 222, 196], "kind": "bone"},
            "web_fin": {"color": [182, 122, 74], "kind": "membrane"},
            "web_back": {"color": [136, 86, 54], "kind": "membrane"},
            "eye": {"color": [248, 172, 46], "kind": "eye", "emissive": 0.85},
        }
    if variant == "ivory":
        return {
            "scales": {"color": [206, 200, 190], "kind": "scales", "coat": "calm_flank"},
            "scales_deep": {"color": [150, 150, 158], "kind": "scales", "coat": "calm_flank"},
            "armor": {"color": [132, 130, 146], "kind": "armor", "coat": "armor_shingle"},
            "rust_organ": {"color": [66, 126, 212], "kind": "crystal", "coat": "crystal_facet"},
            "rust_ridge": {"color": [112, 178, 240], "kind": "crystal", "coat": "crystal_facet"},
            "rust_socket": {"color": [74, 78, 96], "kind": "armor", "coat": "armor_shingle"},
            "belly": {"color": [88, 92, 114], "kind": "belly", "coat": "belly_scute"},
            "horn": {"color": [174, 168, 154], "kind": "horn", "coat": "horn_keratin"},
            "hood_fin": {"color": [176, 174, 184], "kind": "armor", "coat": "armor_shingle"},
            "claw": {"color": [68, 70, 86], "kind": "horn", "coat": "horn_keratin"},
            "mouth": {"color": [108, 62, 112], "kind": "mouth"},
            "tooth": {"color": [236, 232, 222], "kind": "bone"},
            "web_fin": {"color": [88, 138, 212], "kind": "membrane"},
            "web_back": {"color": [64, 98, 166], "kind": "membrane"},
            "eye": {"color": [72, 182, 255], "kind": "eye", "emissive": 0.86},
        }
    return {  # abyssal
        "scales": {"color": [28, 42, 70], "kind": "scales", "coat": "calm_flank"},
        "scales_deep": {"color": [18, 26, 48], "kind": "scales", "coat": "calm_flank"},
        "armor": {"color": [22, 30, 54], "kind": "armor", "coat": "armor_shingle"},
        "rust_organ": {"color": [44, 162, 216], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.22},
        "rust_ridge": {"color": [84, 218, 248], "kind": "crystal", "coat": "crystal_facet", "emissive": 0.36},
        "rust_socket": {"color": [36, 44, 68], "kind": "armor", "coat": "armor_shingle"},
        "belly": {"color": [62, 84, 120], "kind": "belly", "coat": "belly_scute"},
        "horn": {"color": [30, 48, 82], "kind": "horn", "coat": "horn_keratin"},
        "hood_fin": {"color": [36, 68, 110], "kind": "armor", "coat": "armor_shingle"},
        "claw": {"color": [58, 148, 198], "kind": "horn", "coat": "horn_keratin"},
        "mouth": {"color": [52, 138, 190], "kind": "mouth"},
        "tooth": {"color": [220, 236, 246], "kind": "bone"},
        "web_fin": {"color": [54, 174, 224], "kind": "membrane"},
        "web_back": {"color": [34, 114, 172], "kind": "membrane"},
        "eye": {"color": [255, 58, 64], "kind": "eye", "emissive": 0.90},
    }


def _build_lagiacrus_specimen(variant: str = "canonical", *, stage_cap: int = 3) -> dict[str, Any]:
    if variant == "canonical":
        mid, title, ep, vkind, par = "lagiacrus", "Lagiacrus", "Lord of the Seas", "canonical_flagship", None
        wiki = "https://monsterhunter.fandom.com/wiki/Lagiacrus"
        notes = [
            "Monumental x32 Lagiacrus reconstructed from wide ref board & PDF blueprint.",
            "Low U-shaped crocodile snout, continuous proud parchment throat/belly scutes, warm amber-webbed 4-lobe cheek hood, and broad faceted rust shock pylons.",
        ]
    elif variant == "ivory":
        mid, title, ep, vkind, par = "ivory_lagiacrus", "Ivory Lagiacrus", "White Conqueror of Land", "subspecies", "lagiacrus"
        wiki = "https://monsterhunter.fandom.com/wiki/Ivory_Lagiacrus"
        notes = [
            "Terrestrial heavy subspecies with taller armored shoulders (+14% chest height/width), broad ivory brow shields, sapphire shock-crystal clusters, and slate underbelly.",
        ]
    else:
        mid, title, ep, vkind, par = "abyssal_lagiacrus", "Abyssal Lagiacrus", "Master of the Deep", "rare_species", "lagiacrus"
        wiki = "https://monsterhunter.fandom.com/wiki/Abyssal_Lagiacrus"
        notes = [
            "Abyssal rare species with longer swept devil horns, expanded 5-lobe electric-cyan cheek & forearm fins, bioluminescent razor dorsal crystals, and broad deep-sea paddle.",
        ]

    m = RigCanvas(
        mid,
        title,
        ep,
        "Leviathan",
        variant_kind=vkind,
        parent_species_id=par,
        wiki_url=wiki,
        materials=lagiacrus_materials(variant),
        notes=notes,
        texels_per_unit=2,
    )

    is_ivory = variant == "ivory"
    is_abyssal = variant == "abyssal"
    shoulder_lift = 1.8 if is_ivory else 0.0
    bulk_x = 1.12 if is_ivory else (1.04 if is_abyssal else 1.0)
    neck_reach = 1.08 if is_abyssal else 1.0

    # Skeleton (`30` bones: `pelvis`, `chest`, 3 necks, `head`, `jaw`, `hood_l/r`, 12 limb bones, 8 tail segments!)
    m.bone("pelvis", "root", (0.0, 14.5 + shoulder_lift * 0.4, -16.0))
    m.bone("chest", "pelvis", (0.0, 17.5 + shoulder_lift, 7.0))
    m.bone("neck_1", "chest", (0.0, 21.5 + shoulder_lift, 18.0 * neck_reach))
    m.bone("neck_2", "neck_1", (0.0, 28.0 + shoulder_lift, 25.5 * neck_reach))
    m.bone("neck_3", "neck_2", (0.0, 34.4 + shoulder_lift, 32.0 * neck_reach))
    m.bone("head", "neck_3", (0.0, 36.6 + shoulder_lift, 36.5 * neck_reach))
    m.bone("jaw", "head", (0.0, 34.2 + shoulder_lift, 37.8 * neck_reach))

    # =========================================================================
    # STAGE 1: 18+ PRIMARY SCULPTURAL MASSES
    # =========================================================================
    torso_stations = [
        ((0.0, 13.8, -24.0), 13.5 * bulk_x, 11.2, "pelvis"),
        ((0.0, 15.0 + shoulder_lift * 0.3, -13.5), 17.0 * bulk_x, 13.2, "pelvis"),
        ((0.0, 16.5 + shoulder_lift * 0.6, -2.5), 19.5 * bulk_x, 15.0 + shoulder_lift, "chest"),
        ((0.0, 18.4 + shoulder_lift, 8.8), 20.5 * bulk_x, 16.0 + shoulder_lift, "chest"),
        ((0.0, 20.8 + shoulder_lift, 18.5 * neck_reach), 15.0 * bulk_x, 13.0, "chest"),
    ]
    for i in range(len(torso_stations) - 1):
        p0, w0, h0, bn = torso_stations[i]
        p1, w1, h1, _ = torso_stations[i + 1]
        m.span_box(f"torso_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=4.0, pattern="calm_flank", stage1=True)

    neck_stations = [
        ((0.0, 20.4 + shoulder_lift, 17.2 * neck_reach), 13.0, 11.5, "neck_1"),
        ((0.0, 26.2 + shoulder_lift, 23.6 * neck_reach), 11.0, 10.0, "neck_2"),
        ((0.0, 32.6 + shoulder_lift, 30.2 * neck_reach), 9.6, 8.8, "neck_3"),
        ((0.0, 36.6 + shoulder_lift, 36.5 * neck_reach), 9.0, 8.2, "head"),
    ]
    for i in range(len(neck_stations) - 1):
        p0, w0, h0, bn = neck_stations[i]
        p1, w1, h1, _ = neck_stations[i + 1]
        m.span_box(f"neck_core_{i}", bn, p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=3.2, pattern="calm_flank", stage1=True)

    hz = 36.5 * neck_reach
    hy = 36.6 + shoulder_lift
    # Low U-shaped crocodilian skull + two-tier tapered muzzle
    m.box("skull_cranium", "head", (0.0, hy + 0.6, hz + 3.2), (11.5, 6.2, 9.5), "scales", face_materials={"up": "armor"}, rotation=(4.0, 0.0, 0.0), component="head", stage1=True)
    m.box("muzzle_mid", "head", (0.0, hy - 0.3, hz + 10.0), (8.2, 4.4, 7.5), "scales", rotation=(6.0, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("muzzle_tip", "head", (0.0, hy - 1.1, hz + 15.2), (6.4, 3.4, 5.0), "scales", rotation=(7.5, 0.0, 0.0), pattern="snout_nostril", component="snout", stage1=True)
    m.box("jaw_core", "jaw", (0.0, hy - 2.6, hz + 8.0), (7.6, 2.8, 16.5), "belly", face_materials={"up": "mouth"}, rotation=(3.0, 0.0, 0.0), pattern="belly_scute", component="jaw", stage1=True)

    # Long Sinuous 8-Bone Swimming Tail
    t_pts = [
        ((0.0, 13.5, -22.5), 12.5, 10.2),
        ((1.8, 12.6, -31.5), 11.0, 9.4),
        ((4.6, 11.8, -40.5), 9.6, 8.6),
        ((7.4, 11.6, -49.5), 7.8, 8.0),
        ((8.6, 12.2, -58.2), 6.0, 8.6),
        ((7.2, 13.6, -66.5), 4.4, 9.8),
        ((4.0, 15.2, -74.0), 3.2, 9.5),
        ((0.5, 16.5, -80.5), 2.2, 7.4),
        ((-2.5, 17.5, -86.2), 1.6, 5.2),
    ]
    tbones = [m.bone(f"tail_{i + 1}", "pelvis" if i == 0 else f"tail_{i}", t_pts[i][0]) for i in range(8)]
    for i in range(8):
        p0, w0, h0 = t_pts[i]
        p1, w1, h1 = t_pts[i + 1]
        m.span_box(f"tail_core_{i}", tbones[i], p0, p1, 0.5 * (w0 + w1), 0.5 * (h0 + h1), "scales", len_pad=2.8, pattern="calm_flank", stage1=(i % 2 == 0))

    # Sprawled Reptile Forelimbs & Hindlimbs (Ivory stands taller & wider!)
    limb_specs = (
        ("fore", "chest", (9.8 * bulk_x, 15.2 + shoulder_lift, 9.5), (15.8 * bulk_x, 9.6 + shoulder_lift * 0.4, 6.5), (15.8 * bulk_x, 3.0, 14.0), (8.2, 7.0), (6.0, 5.5), 4),
        ("hind", "pelvis", (8.8 * bulk_x, 14.0, -14.5), (14.8 * bulk_x, 8.5, -10.0), (14.8 * bulk_x, 2.8, -16.5), (8.6, 7.5), (6.2, 5.5), 3),
    )
    for kind, pbone, p0_raw, p1_raw, p2_raw, up_wh, lo_wh, n_dig in limb_specs:
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p0 = (s * p0_raw[0], p0_raw[1], p0_raw[2])
            p1 = (s * p1_raw[0], p1_raw[1], p1_raw[2])
            p2 = (s * p2_raw[0], p2_raw[1], p2_raw[2])
            bu = m.bone(f"{kind}_upper_{q}", pbone, p0)
            bl = m.bone(f"{kind}_lower_{q}", bu, p1)
            bp = m.bone(f"{kind}_paw_{q}", bl, p2)
            m.span_box(f"{kind}_thigh_{q}", bu, p0, p1, up_wh[0], up_wh[1], "scales", len_pad=2.4, pattern="calm_flank", stage1=True)
            m.span_box(f"{kind}_shin_{q}", bl, p1, p2, lo_wh[0], lo_wh[1], "scales", len_pad=2.2, pattern="leg_bands", stage1=True)
            m.box(f"{kind}_foot_pad_{q}", bp, (p2[0], 2.2, p2[2] + 2.5), (8.4, 3.0, 8.8), "scales", face_materials={"up": "armor"}, stage1=True)

    if stage_cap <= 1:
        return m.doc

    # =========================================================================
    # STAGE 2: BESPOKE ANATOMY & SILHOUETTE FIXES
    # =========================================================================
    # 2A. Refined Muzzle Slope, Enamel Teeth, Sleek Crown Horns & Scalloped Cheek Hood Webbing
    m.box("nasal_bridge_slope_a", "head", (0.0, hy + 2.7, hz + 6.8), (7.5, 2.4, 6.5), "armor", rotation=(11.0, 0.0, 0.0), pattern="armor_shingle", component="head")
    m.box("nasal_bridge_slope_b", "head", (0.0, hy + 1.3, hz + 12.6), (5.5, 2.0, 6.0), "armor", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="snout")
    m.box("snout_rounded_prow", "head", (0.0, hy - 0.5, hz + 17.6), (5.0, 2.4, 2.5), "scales", rotation=(12.0, 0.0, 0.0), pattern="snout_nostril", component="snout")
    m.box("lower_chin_prow", "jaw", (0.0, hy - 3.6, hz + 15.8), (5.4, 2.4, 3.8), "belly", pattern="belly_scute", component="jaw")
    m.box("occipital_skull_plate", "head", (0.0, hy + 3.4, hz + 1.0), (8.2, 2.2, 6.5), "armor", rotation=(8.0, 0.0, 0.0), pattern="armor_shingle", component="head")

    crown_tip_col = [198, 168, 114] if variant == "canonical" else ([96, 178, 245] if is_ivory else [92, 228, 252])
    hood_tip_col = [178, 126, 78] if variant == "canonical" else ([88, 138, 212] if is_ivory else [78, 216, 248])

    for s, q in ((1.0, "l"), (-1.0, "r")):
        m.box(f"cheek_flare_{q}", "head", (s * 5.8, hy - 0.5, hz + 2.5), (3.0, 4.4, 7.5), "scales", rotation=(4.0, s * 12.0, -s * 8.0), pattern="armor_shingle", component="head")
        brow_w = 3.2 if is_ivory else 2.5
        m.box(f"brow_ridge_{q}", "head", (s * 5.0, hy + 3.0, hz + 5.8), (brow_w, 2.2, 5.8), "armor", rotation=(8.0, s * 10.0, -s * 14.0), pattern="armor_shingle", component="head")
        m.box(f"supraorbital_crest_{q}", "head", (s * 4.5, hy + 3.8, hz + 3.2), (2.2, 1.8, 4.2), "rust_organ" if is_ivory else "armor", rotation=(16.0, s * 12.0, -s * 10.0), pattern="crystal_facet" if is_ivory else "armor_shingle", component="head")
        m.box(f"eye_recess_{q}", "head", (s * 5.2, hy + 1.4, hz + 6.0), (1.2, 1.8, 3.0), "mouth", rotation=(5.0, s * 8.0, 0.0), component="eyes")
        m.box(f"eye_amber_slit_{q}", "head", (s * 5.6, hy + 1.4, hz + 6.2), (0.8, 1.2, 2.2), "eye", rotation=(5.0, s * 8.0, 0.0), pattern="eye_pixel", component="eyes")
        m.box(f"mandible_flank_scute_{q}", "jaw", (s * 4.0, hy - 3.0, hz + 8.8), (1.9, 2.5, 14.2), "belly", pattern="belly_scute", component="jaw")

        for ti, tz in enumerate((hz + 4.8, hz + 7.4, hz + 9.8, hz + 12.2, hz + 14.6, hz + 16.4)):
            ux = 3.9 - 0.28 * ti
            th = 1.6 if ti in (1, 4) else 1.2
            m.box(f"upper_tooth_{q}_{ti}", "head", (s * ux, hy - 2.3, tz), (0.9, th, 1.0), "tooth", rotation=(0.0, s * 8.0, -s * 8.0), component="teeth")
            if ti % 2 == 0:
                m.box(f"lower_tooth_{q}_{ti}", "jaw", (s * (ux - 0.5), hy - 1.7, tz - 0.5), (0.8, 1.1, 0.9), "tooth", component="teeth")

        # Sleek Dominant Crown Horns (NO repeated striping! Only final segment fades to ochre/cyan tip!)
        h_scale = 1.36 if is_abyssal else (1.16 if is_ivory else 1.0)
        m.tapered_arch(
            f"crown_horn_{q}",
            "head",
            [
                (s * 3.4, hy + 3.4, hz + 1.8),
                (s * (5.2 * h_scale), hy + 8.2 * h_scale, hz - 3.6 * h_scale),
                (s * (6.4 * h_scale), hy + 13.4 * h_scale, hz - 9.4 * h_scale),
                (s * (6.8 * h_scale), hy + 18.2 * h_scale, hz - 14.2 * h_scale),
            ],
            [(3.6, 3.0), (2.8, 2.4), (2.0, 1.8), (1.1, 1.1)],
            "horn",
            tip_color=crown_tip_col,
            tip_start=0.35,
            component="horns",
            stage1=True,
        )
        m.tapered_arch(
            f"temple_horn_{q}",
            "head",
            [(s * 5.0, hy + 1.8, hz + 0.5), (s * (8.0 * h_scale), hy + 4.2, hz - 4.2), (s * (10.2 * h_scale), hy + 6.2, hz - 8.2)],
            [(2.4, 2.0), (1.8, 1.5), (1.0, 1.0)],
            "horn",
            tip_color=crown_tip_col,
            component="horns",
        )

        # Articulating Multi-Lobe Scalloped Cheek Hood (`4` rays canonical/ivory, `5` flared abyssal angler rays!)
        hb = m.bone(f"hood_{q}", "neck_3", (s * 4.8, hy - 2.6, hz - 4.0))
        hood_lobes = [
            ((s * 4.8, hy - 0.4, hz - 1.4), (s * (13.4 if is_abyssal else 12.4), hy + 3.0, hz - 5.6), 2.8, 1.6),
            ((s * 4.8, hy - 2.8, hz - 2.4), (s * (15.8 if is_abyssal else 14.2), hy - 1.5, hz - 7.6), 3.1, 1.6),
            ((s * 4.6, hy - 5.2, hz - 4.0), (s * (15.2 if is_abyssal else 13.6), hy - 6.0, hz - 9.4), 2.7, 1.6),
            ((s * 4.2, hy - 7.4, hz - 5.8), (s * (12.8 if is_abyssal else 11.2), hy - 9.8, hz - 11.2), 2.3, 1.5),
        ]
        if is_abyssal:
            hood_lobes.append(((s * 3.8, hy - 9.4, hz - 7.6), (s * 10.6, hy - 12.8, hz - 12.8), 2.0, 1.4))

        for li, (r_pt, t_pt, lw, lh) in enumerate(hood_lobes):
            mid_l = lerp3(r_pt, t_pt, 0.56)
            m.span_box(f"hood_lobe_{q}_{li}_a", hb, r_pt, mid_l, lw, lh, "hood_fin", roll_deg=-s * (10.0 + 4.0 * li), len_pad=0.9, pattern="armor_shingle", component="horns")
            m.span_box(f"hood_lobe_{q}_{li}_b", hb, mid_l, t_pt, lw * 0.78, lh * 0.85, "hood_fin", roll_deg=-s * (12.0 + 4.0 * li), len_pad=0.5, pattern="armor_shingle", tip_color=hood_tip_col, tip_start=0.40, component="horns")
            if li < len(hood_lobes) - 1:
                nxt_r, nxt_t = hood_lobes[li + 1][0], hood_lobes[li + 1][1]
                # Both faces warm webbed membrane (`web_fin` front, `web_back` warm deep shade!)
                m.cutout_panel(
                    f"hood_web_{q}_{li}",
                    hb,
                    [v3(r_pt), v3(t_pt), v3(nxt_t), v3(nxt_r)],
                    "web_fin",
                    back_material="web_back",
                    pattern="hood_lobe_cutout",
                    component="membrane",
                )

    # Occipital nape & neural dorsal spine ridges linking head -> neck -> dorsal shock rows
    for ci, (bn, y_c, z_c, h_sp) in enumerate((
        ("neck_3", hy + 2.6, hz - 4.0, 4.6),
        ("neck_2", hy - 2.2 + 0.5, 26.5 * neck_reach, 4.4),
        ("neck_1", hy - 8.2 + 0.5, 20.5 * neck_reach, 4.2),
        ("chest", 25.0 + shoulder_lift, 14.5, 3.8),
        ("chest", 24.0 + shoulder_lift, 2.5, 3.6),
        ("pelvis", 21.5 + shoulder_lift * 0.4, -10.5, 3.4),
    )):
        m.box(f"neural_crest_{ci}", bn, (0.0, y_c, z_c), (2.6, h_sp, 5.2), "armor", rotation=(26.0, 0.0, 0.0), pattern="armor_shingle", component="spines")

    # 2B. Proud Continuous Stepped Parchment Throat & Sternum Scutes (`y` pushed proudly outside `neck_core` / `torso_core`!)
    belly_plates = (
        ("jaw", (0.0, hy - 4.3, hz + 0.4), 7.8, 2.4, 4.6, 16.0),
        ("neck_3", (0.0, hy - 6.6, hz - 4.4), 9.2, 3.0, 5.2, 34.0),
        ("neck_2", (0.0, 27.2 + shoulder_lift, 29.2 * neck_reach), 10.2, 3.2, 5.5, 42.0),
        ("neck_2", (0.0, 22.8 + shoulder_lift, 25.4 * neck_reach), 11.4, 3.2, 5.6, 42.0),
        ("neck_1", (0.0, 18.6 + shoulder_lift, 21.4 * neck_reach), 12.6, 3.2, 5.8, 36.0),
        ("neck_1", (0.0, 15.2 + shoulder_lift, 17.6 * neck_reach), 14.0, 3.2, 6.0, 26.0),
        ("chest", (0.0, 12.2 + shoulder_lift * 0.7, 13.0), 15.6 * bulk_x, 3.2, 6.2, 16.0),
        ("chest", (0.0, 9.8 + shoulder_lift * 0.5, 7.6), 16.6 * bulk_x, 3.2, 6.2, 8.0),
        ("chest", (0.0, 8.6 + shoulder_lift * 0.3, 1.8), 16.0 * bulk_x, 3.0, 6.0, 2.0),
        ("chest", (0.0, 8.2, -4.0), 14.8 * bulk_x, 3.0, 5.8, -2.0),
        ("pelvis", (0.0, 8.0, -9.8), 13.8 * bulk_x, 3.0, 5.8, -2.0),
        ("pelvis", (0.0, 8.2, -15.4), 12.2 * bulk_x, 2.8, 5.6, -3.0),
        ("pelvis", (0.0, 8.6, -20.8), 10.2 * bulk_x, 2.6, 5.2, -5.0),
    )
    for bi, (bn, pos, pw, ph, pd, rx) in enumerate(belly_plates):
        m.box(f"belly_scute_plate_{bi}", bn, pos, (pw, ph, pd), "belly", rotation=(rx, 0.0, 0.0), pattern="belly_scute", component="surface")

    # 2C. Broad Heavy Faceted Shock Pylons (`NO skinny sticks!`)
    dorsal_specs = (
        ("chest", 11.5, 4.8 * bulk_x, 8.6 if variant == "canonical" else 9.8),
        ("chest", 5.2, 5.8 * bulk_x, 10.8 if variant == "canonical" else 12.8),
        ("chest", -1.2, 5.6 * bulk_x, 10.2 if variant == "canonical" else 12.0),
        ("pelvis", -7.8, 5.0 * bulk_x, 8.6 if variant == "canonical" else 10.0),
        ("pelvis", -14.2, 4.2 * bulk_x, 6.8 if variant == "canonical" else 7.8),
    )
    pylon_pat = "crystal_facet" if (is_ivory or is_abyssal) else "dorsal_blade"
    pylon_tip = [118, 206, 255] if is_ivory else ([110, 240, 255] if is_abyssal else [212, 146, 82])
    for di, (dbone, dz, sep_x, h_max) in enumerate(dorsal_specs):
        by = (24.5 + shoulder_lift) if dbone == "chest" else (20.6 + shoulder_lift * 0.4)
        for s, q in ((1.0, "l"), (-1.0, "r")):
            # Broad crater socket rim wrapping base of the dorsal shock organ
            m.box(f"organ_socket_{di}_{q}", dbone, (s * sep_x, by - 0.5, dz), (6.0, 3.2, 6.8), "rust_socket", rotation=(-12.0, s * 10.0, -s * 16.0), pattern="armor_shingle", component="spines")
            m.box(f"organ_buttress_out_{di}_{q}", dbone, (s * (sep_x + 2.0), by - 1.2, dz - 0.4), (3.4, 3.6, 5.2), "armor", rotation=(-18.0, s * 14.0, -s * 24.0), pattern="armor_shingle", component="spines")

            # Main broad swept carapace shell pylon (heavy chisel wedge!)
            b_main = np.array([s * sep_x, by + 0.6, dz])
            m_main = b_main + np.array([s * 1.4, h_max * 0.54, -2.6])
            t_main = b_main + np.array([s * 2.4, h_max, -5.8])
            m.span_box(f"shock_pylon_main_{di}_{q}_a", dbone, b_main, m_main, 3.5, 3.6, "rust_organ", roll_deg=-s * 14.0, len_pad=0.9, pattern=pylon_pat, component="spines")
            m.span_box(f"shock_pylon_main_{di}_{q}_b", dbone, m_main - np.array([0.0, 0.6, 0.0]), t_main, 2.3, 2.4, "rust_ridge", roll_deg=-s * 14.0, len_pad=0.5, pattern=pylon_pat, tip_color=pylon_tip, tip_start=0.45, component="spines")

            # Flanking fore & aft stepped crystal/shell ridges merging seamlessly into main pylon
            for sub_name, dz_sub, hs in (("fore", 1.8, 0.58), ("aft", -1.6, 0.68)):
                b_sub = b_main + np.array([s * 1.1, -0.4, dz_sub])
                t_sub = b_sub + np.array([s * 1.6, h_max * hs, -3.2])
                m.span_box(f"shock_pylon_{sub_name}_{di}_{q}", dbone, b_sub, t_sub, 2.4, 2.6, "rust_organ", roll_deg=-s * 18.0, len_pad=0.6, pattern=pylon_pat, component="spines")

    # 2D. Detailed Limbs + Carpal Forearm Fins (`Lagiacrus` ref `[11],[12]`!) + 4 Splayed Clawed Digits
    for kind, pbone, p0_r, p1_r, p2_r, sc_w in (
        ("fore", "chest", (9.8 * bulk_x, 15.2 + shoulder_lift, 9.5), (15.8 * bulk_x, 9.6 + shoulder_lift * 0.4, 6.5), (15.8 * bulk_x, 3.0, 14.0), 8.2),
        ("hind", "pelvis", (8.8 * bulk_x, 14.0, -14.5), (14.8 * bulk_x, 8.5, -10.0), (14.8 * bulk_x, 2.8, -16.5), 8.8),
    ):
        for s, q in ((1.0, "l"), (-1.0, "r")):
            bu, bl, bp = f"{kind}_upper_{q}", f"{kind}_lower_{q}", f"{kind}_paw_{q}"
            p0 = np.array([s * p0_r[0], p0_r[1], p0_r[2]])
            p1 = np.array([s * p1_r[0], p1_r[1], p1_r[2]])
            p2 = np.array([s * p2_r[0], p2_r[1], p2_r[2]])
            m.box(f"{kind}_flank_shoulder_{q}", bu, p0 + np.array([-s * 1.0, 1.8, 0.0]), (sc_w, sc_w * 1.15, sc_w * 1.25), "scales", face_materials={"up": "armor"}, rotation=(10.0, s * 10.0, -s * 14.0), pattern="calm_flank")
            m.box(f"{kind}_scapula_cap_{q}", bu, p0 + np.array([s * 1.4, 3.2, 0.5]), (5.2, 3.4, 6.8), "armor", rotation=(8.0, s * 14.0, -s * 24.0), pattern="armor_shingle", component="armor")
            m.box(f"{kind}_joint_plate_{q}", bl, p1 + np.array([s * 1.2, 0.8, 1.2 if kind == "hind" else -1.2]), (5.2, 4.6, 5.2), "armor", rotation=(20.0, s * 12.0, 0.0), pattern="armor_shingle")
            for bi in range(4):
                pt_b = lerp3(p1, p2, 0.16 + 0.23 * bi) + np.array([s * 2.4, 0.7, 0.7])
                m.box(f"{kind}_bracer_scute_{q}_{bi}", bl, pt_b, (2.0, 3.6, 4.4), "armor", rotation=(16.0, s * 14.0, -s * 18.0), pattern="armor_shingle", component="spines")
            if kind == "fore":
                # Signature swept carpal forearm fin (`fin_edge` + web cutout)
                f_base = p1 + np.array([s * 3.0, 1.0, -0.5])
                f_tip = f_base + np.array([s * (6.2 if is_abyssal else 4.2), -2.5, -5.5])
                m.span_box(f"fore_carpal_fin_{q}", bl, f_base, f_tip, 1.4, 2.2, "hood_fin", roll_deg=-s * 14.0, tip_color=hood_tip_col, component="horns")

            for di, ang_deg, len_d in ((0, -24.0, 4.8), (1, -6.0, 5.8), (2, 12.0, 5.5), (3, 30.0, 4.2)):
                yaw = math.radians(s * ang_deg)
                dir_d = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
                toe_a = np.array([p2[0] + s * (-2.8 + 1.8 * di), 2.0, p2[2] + 5.8])
                toe_b = toe_a + dir_d * (len_d * 0.62)
                toe_b[1] = 1.6
                claw_c = toe_b + dir_d * (len_d * 0.45)
                claw_c[1] = 0.8
                m.span_box(f"{kind}_toe_{q}_{di}", bp, toe_a, toe_b, 2.0, 1.8, "scales_deep", len_pad=0.6, pattern="armor_shingle", component="claws")
                m.span_box(f"{kind}_claw_{q}_{di}", bp, toe_b, claw_c, 1.5, 1.5, "claw", len_pad=0.4, pattern="horn_keratin", component="claws")

    # 2E. Tall Sinuous Paddle Tail + Broad Caudal Fin Webbing (`web_fin` visible on BOTH sides!)
    for ti in range(8):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.box(f"tail_ventral_scute_{ti}", tbones[ti], mid_t + np.array([0.0, -3.8 + 0.22 * ti, 0.0]), (max(2.6, 8.0 - 0.70 * ti), 2.2, 8.2), "belly", rotation=(-6.0, 0.0, 0.0), pattern="belly_scute")
        if ti >= 1:
            fin_h = 4.2 + (1.35 * ti if ti >= 4 else 0.4 * ti)
            tip_t = mid_t + np.array([0.0, fin_h, -3.8])
            fin_mat = "rust_organ" if ti in (2, 4) else "hood_fin"
            m.span_box(f"tail_dorsal_blade_{ti}", tbones[ti], mid_t + np.array([0.0, 2.8, 0.6]), tip_t, 2.1 if ti < 5 else 1.5, 3.0, fin_mat, len_pad=0.6, pattern="dorsal_blade", tip_color=hood_tip_col if ti >= 3 else None, component="spines")
        if ti >= 3:
            # Double-sided warm amber/sapphire/cyan swimming sail connecting dorsal fin rays (`+X` side outward facing first!)
            h_top = 8.2 + 0.6 * (ti - 3)
            m.cutout_panel(
                f"tail_paddle_web_{ti}",
                tbones[ti],
                [
                    pb + np.array([0.2, 2.0, 0.0]),
                    pa + np.array([0.2, 2.0, 0.0]),
                    pa + np.array([0.2, h_top * 0.88, -1.2]),
                    pb + np.array([0.2, h_top, -1.6]),
                ],
                "web_fin",
                back_material="web_back",
                pattern="tail_paddle_fin",
                component="membrane",
                serrated_hem=True,
            )

    if stage_cap <= 2:
        return m.doc

    # =========================================================================
    # STAGE 3: RICH SURFACE SHINGLING, CARAPACE BEVELING & PROUD FLANK ARMOR
    # =========================================================================
    neck_rigs = (
        ("neck_1", (0.0, 21.0 + shoulder_lift, 17.5 * neck_reach), (0.0, 25.5 + shoulder_lift, 22.8 * neck_reach), 6.8),
        ("neck_2", (0.0, 27.0 + shoulder_lift, 24.5 * neck_reach), (0.0, 31.6 + shoulder_lift, 29.2 * neck_reach), 5.8),
        ("neck_3", (0.0, 32.5 + shoulder_lift, 30.2 * neck_reach), (0.0, 35.8 + shoulder_lift, 34.5 * neck_reach), 5.0),
    )
    for ni, (nb, n_a, n_b, lat_x) in enumerate(neck_rigs):
        # Upper dorsal neck collar plates breaking flat `neck_core` box look
        mid_n = lerp3(n_a, n_b, 0.5)
        m.box(f"neck_dorsal_shell_{ni}", nb, mid_n + np.array([0.0, 4.5 - 0.5 * ni, -0.5]), (7.6 - 0.9 * ni, 2.6, 6.2), "armor", rotation=(38.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            p_sh = v3(n_a) + np.array([s * lat_x, 0.8, 0.0])
            p_end = v3(n_b) + np.array([s * (lat_x - 0.6), 1.0, 0.0])
            m.shingle_arc(
                f"neck_side_osteoderms_{ni}_{q}",
                nb,
                p_sh,
                p_end,
                (s * 0.35, 0.25, -0.88),
                (s * 0.25, 0.30, -0.90),
                count=4,
                length=5.6,
                width=2.8,
                thickness=1.8,
                material="armor",
                alt_material="scales_deep",
                two_step=False,
                pattern="armor_shingle",
            )

    # Proud layered flank carapace banks & lateral rib pleats sitting cleanly OUTSIDE torso core!
    for s, q in ((1.0, "l"), (-1.0, "r")):
        cx = 10.6 * bulk_x
        m.shingle_arc(
            f"flank_armor_bank_chest_{q}",
            "chest",
            (s * cx, 19.8 + shoulder_lift, 11.5),
            (s * (cx + 0.4), 17.5 + shoulder_lift, -3.5),
            (s * 0.28, 0.16, -0.94),
            (s * 0.24, 0.12, -0.96),
            count=6,
            length=6.8,
            width=3.6,
            thickness=2.2,
            material="armor",
            alt_material="scales_deep",
            two_step=False,
            pattern="armor_shingle",
        )
        m.shingle_arc(
            f"flank_lower_rib_bank_{q}",
            "chest",
            (s * (cx - 0.4), 15.2 + shoulder_lift * 0.7, 9.5),
            (s * (cx - 0.2), 13.8 + shoulder_lift * 0.4, -3.0),
            (s * 0.25, -0.10, -0.96),
            (s * 0.20, -0.12, -0.97),
            count=5,
            length=6.2,
            width=3.2,
            thickness=1.9,
            material="scales_deep",
            alt_material="armor",
            two_step=False,
            pattern="armor_shingle",
        )
        m.shingle_arc(
            f"flank_armor_bank_pelvis_{q}",
            "pelvis",
            (s * (9.8 * bulk_x), 16.8 + shoulder_lift * 0.4, -5.0),
            (s * (7.6 * bulk_x), 14.8, -21.0),
            (s * 0.22, 0.14, -0.96),
            (s * 0.18, 0.12, -0.97),
            count=5,
            length=6.5,
            width=3.2,
            thickness=2.0,
            material="armor",
            alt_material="scales_deep",
            two_step=False,
            pattern="armor_shingle",
        )
        # Beveled dorsal carapace shelf between flank and dorsal shock organs (`eliminates block edges!`)
        for shi, (sh_b, sh_z, sh_y, sh_w) in enumerate((
            ("chest", 11.0, 23.2 + shoulder_lift, 4.4),
            ("chest", 3.5, 22.8 + shoulder_lift, 4.8),
            ("chest", -3.5, 21.8 + shoulder_lift * 0.6, 4.4),
            ("pelvis", -10.5, 20.0 + shoulder_lift * 0.3, 4.0),
            ("pelvis", -17.5, 18.4, 3.4),
        )):
            m.box(f"back_carapace_shelf_{shi}_{q}", sh_b, (s * 7.8 * bulk_x, sh_y, sh_z), (sh_w, 2.2, 6.2), "armor", rotation=(6.0, s * 10.0, -s * 28.0), pattern="armor_shingle", component="armor")

    # Dorsal & lateral tail osteoderm armor rows (`tail_1..7`)
    for ti in range(7):
        pa, pb = v3(t_pts[ti][0]), v3(t_pts[ti + 1][0])
        mid_t = 0.5 * (pa + pb)
        m.box(f"tail_dorsal_scute_{ti}", tbones[ti], mid_t + np.array([0.0, 4.2 - 0.12 * ti, 0.0]), (max(2.0, 6.4 - 0.62 * ti), 1.8, 7.6), "armor", rotation=(6.0, 0.0, 0.0), pattern="armor_shingle", component="armor")
        for s, q in ((1.0, "l"), (-1.0, "r")):
            m.box(
                f"tail_lateral_scute_{ti}_{q}",
                tbones[ti],
                mid_t + np.array([s * max(1.2, 4.8 - 0.48 * ti), 0.2, 0.0]),
                (1.9, max(1.5, 2.6 - 0.16 * ti), 6.2),
                "armor",
                rotation=(6.0, -s * 14.0, -s * 18.0),
                pattern="armor_shingle",
                component="spines",
            )

    return m.doc


def build_lagiacrus(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_lagiacrus_specimen("canonical", stage_cap=stage_cap)


def build_ivory_lagiacrus(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_lagiacrus_specimen("ivory", stage_cap=stage_cap)


def build_abyssal_lagiacrus(*, stage_cap: int = 3) -> dict[str, Any]:
    return _build_lagiacrus_specimen("abyssal", stage_cap=stage_cap)
