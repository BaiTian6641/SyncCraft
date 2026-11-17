"""Vibrant x32 Sculpted Blockbench Pixel-Art Studio Painter for Native Block-DSL v4.1 Assets.

Operates directly on SyncCraft's compiled `Program` + `Scene` (`WorldBox`),
allocating outward-face `x32` (`2 texels/unit`) UV rectangles matching `export_bbmodel`
and `export_bedrock`, honoring `@palette`, `@materials`, `@stamps`, and `plane` `edge:`
cut-out declarations, and painting vibrant hue-shifted Blockbench pixel-art textures:

  - Rich 6-step hue-shifted perceptual OKLCH -> sRGB ramps (warm sunlit highlights,
    vivid saturated midtones, cool royal-indigo / violet / plum shadows — ZERO grey mud).
  - Stepped diagonal scale clusters on dorsal/upper hides calming toward smooth flanks.
  - Beveled roof-tile carapace shingles + species-authentic glowing fissure veins.
  - Continuous multi-segment horn, antler, tusk, and talon longitudinal keratin grain
    (mapped by 3D chain position so adjacent segments flow without candy-cane resets).
  - Contoured ventral scutes with crisp anterior overlap shadows.
  - Double-sided binary `{0, 255}` alpha-cut `plane` bays (`scallop`, `ragged`, `spikes`,
    `drip`, `triangle_bay`) with radial spar venation and species wing markings
    (Rathalos flame-scrolls, Legiana cryo-pleats, Gore eclipse veil & spores, Valstrax
    crimson plasma lances, Rey Dau fulgurite facets, Mizutsune sakura ocelli).
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from synccraft.dsl.ast_nodes import Palette, Program
from synccraft.kinematics.se3_chain import Scene, WorldBox
from synccraft.shading.compiler import compute_plane_edge_alpha_mask
from synccraft.surface.contracts import linear_to_srgb_u8, oklch_to_linear_srgb

FACES: tuple[str, ...] = ("north", "south", "east", "west", "up", "down")
FACE_AXES: dict[str, tuple[int, int]] = {
    "north": (0, 1),
    "south": (0, 1),
    "east": (2, 1),
    "west": (2, 1),
    "up": (0, 2),
    "down": (0, 2),
}
PLANE_PRIMARY_FACES: dict[str, tuple[str, str]] = {
    "x": ("east", "west"),
    "y": ("up", "down"),
    "z": ("south", "north"),
}


def _hash2(x: np.ndarray | int, y: np.ndarray | int, seed: int = 0) -> np.ndarray:
    hx = (
        np.asarray(x, dtype=np.int64) * 73856093
        ^ np.asarray(y, dtype=np.int64) * 19349663
        ^ (int(seed) * 83492791)
    ) & 0xFFFFFFFF
    hx = ((hx ^ (hx >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    hx = ((hx ^ (hx >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    return ((hx ^ (hx >> 16)) & 0xFFFF).astype(np.float64) / 65535.0


def build_vibrant_6step_ramp(pal: Palette) -> np.ndarray:
    """Build a vivid 6-step hue-shifted Blockbench pixel-art ramp `(6, 3)` uint8.

    Index 0 = deep cool crevice shadow .. Index 3 = true midtone .. Index 5 = crisp rim highlight.
    """
    base_l, base_c, base_h = [float(x) for x in pal.base_lch]
    sh_shift, hi_shift = [float(x) for x in pal.hue_shift]
    if abs(sh_shift) < 4.0 and abs(hi_shift) < 4.0:
        # Apply natural Blockbench hue shift when omitted: cool violet/blue shadow, warm highlight
        sh_shift = 22.0 if (base_h < 220.0 or base_h > 310.0) else -16.0
        hi_shift = 16.0 if (base_h < 95.0 or base_h > 290.0) else -14.0

    # Relative lightness & chroma envelope preserving vivid mid-tones
    l_offsets = (-0.28, -0.15, -0.05, 0.0, +0.11, +0.22)
    c_Mults = (0.94, 1.06, 1.04, 1.00, 0.92, 0.78)
    h_factors = (-1.0, -0.60, -0.22, 0.0, +0.55, +1.0)

    lch_rows = []
    for dl, cm, hf in zip(l_offsets, c_Mults, h_factors):
        li = float(np.clip(base_l + dl + (0.06 if pal.emissive else 0.0), 0.07, 0.97))
        ci = float(np.clip(base_c * cm * (1.10 if not pal.emissive else 1.15), 0.0, 0.34))
        hi = (base_h + ((-hf) * sh_shift if hf < 0.0 else hf * hi_shift)) % 360.0
        lch_rows.append((li, ci, hi))

    lin = oklch_to_linear_srgb(np.asarray(lch_rows, dtype=np.float64))
    return linear_to_srgb_u8(np.clip(lin, 0.0, 1.0))


def _infer_part_role(box: WorldBox, mat_family: str, mat_coat: str | None) -> str:
    nm = f"{box.bone_name}__{box.name}__{box.material}".lower()
    if getattr(box, "plane_normal", None) is not None:
        if any(k in nm for k in ("fur", "mane", "beard", "plume", "tuft", "brush", "ruff")):
            return "plane_fur"
        if any(k in nm for k in ("plasma", "jet", "exhaust", "flame", "beam", "spark", "lightning")):
            return "plane_plasma"
        if any(k in nm for k in ("crystal", "ice", "icicle", "frost", "wylk", "fulgurite", "tiara")):
            return "plane_crystal"
        if any(k in nm for k in ("fin", "paddle", "hood", "frill", "vane", "petal", "gill")):
            return "plane_fin"
        return "plane_membrane"
    if any(k in nm for k in ("eye", "iris", "pupil", "ocellus")):
        return "eye"
    if any(k in nm for k in ("tooth", "fang", "tusk")):
        return "tooth"
    if any(k in nm for k in ("mouth", "palate", "tongue", "gum")):
        return "mouth"
    if any(k in nm for k in ("crystal", "ice", "icicle", "wylk", "fulgurite", "prism", "gem", "dynamo", "shock", "spire")):
        return "crystal"
    if any(k in nm for k in ("horn", "antler", "claw", "talon", "dewclaw", "beak", "spike", "barb", "lance", "stinger")):
        return "horn"
    if any(k in nm for k in ("fur", "mane", "brush", "pelage", "ruff", "plume")) or mat_family == "woven_fabric":
        return "fur"
    if any(k in nm for k in ("belly", "scute", "ventral", "throat", "sternum", "underside")) or mat_coat == "plain":
        if any(k in nm for k in ("belly", "scute", "ventral", "throat", "sternum")):
            return "belly"
    if any(k in nm for k in ("radiator", "heatsink", "metal", "thruster", "intake", "chain", "pylon")) or mat_family in (
        "brushed_metal",
        "painted_metal",
    ):
        return "metal_armor"
    if any(k in nm for k in ("armor", "plate", "shingle", "carapace", "crest", "visor", "brow", "bracer", "pauldron", "shield")):
        return "armor"
    if mat_coat == "banded_limb" or any(k in nm for k in ("shin", "tibia", "tarsus", "forearm")):
        return "limb_scales"
    return "scales"


def allocate_artisan_x32_uvs(program: Program, scene: Scene, *, padding: int = 2) -> tuple[dict[str, dict[str, list[int]]], list[tuple[WorldBox, str, int, int, int, int]], int, int]:
    """Pack `x32` integer UV islands; double-sided `plane`s share front/mirrored-back islands."""
    upb = max(1, int(getattr(program.meta, "units_per_block", 16)))
    tpu = max(1.0, float(program.meta.texel_density) / float(upb))

    jobs: list[tuple[int, int, int, WorldBox, str, bool]] = []
    for idx, box in enumerate(scene.boxes):
        pnorm = getattr(box, "plane_normal", None)
        if pnorm in PLANE_PRIMARY_FACES:
            f_front, _ = PLANE_PRIMARY_FACES[pnorm]
            ax_u, ax_v = FACE_AXES[f_front]
            w = max(2, int(round(float(box.size[ax_u]) * tpu)))
            h = max(2, int(round(float(box.size[ax_v]) * tpu)))
            jobs.append((h, w, idx, box, f_front, True))
        else:
            for face in FACES:
                ax_u, ax_v = FACE_AXES[face]
                w = max(2, int(round(float(box.size[ax_u]) * tpu)))
                h = max(2, int(round(float(box.size[ax_v]) * tpu)))
                jobs.append((h, w, idx, box, face, False))

    total_area = sum((w + 2 * padding) * (h + 2 * padding) for h, w, *_ in jobs)
    max_w = max((w + 2 * padding for _, w, *_ in jobs), default=64)
    width = 1 << max(9, math.ceil(math.log2(max(float(max_w), math.sqrt(total_area) * 1.22))))
    width = min(4096, width)

    faces_out: dict[str, dict[str, list[int]]] = {b.name: {} for b in scene.boxes}
    placements: list[tuple[WorldBox, str, int, int, int, int]] = []
    x, y, row_h = padding, padding, 0

    for h, w, _, box, face, is_plane in sorted(jobs, key=lambda t: (t[0], t[1]), reverse=True):
        if x + w + padding > width:
            x = padding
            y += row_h + 2 * padding
            row_h = 0
        rect = [x, y, x + w, y + h]
        if is_plane:
            pnorm = str(getattr(box, "plane_normal"))
            f_front, f_back = PLANE_PRIMARY_FACES[pnorm]
            faces_out[box.name][f_front] = list(rect)
            # Horizontally mirror back face so alpha cutouts align in 3D space!
            faces_out[box.name][f_back] = [x + w, y, x, y + h]
            deg_rect = [x, y, x + 1, y + 1]
            for f_other in FACES:
                faces_out[box.name].setdefault(f_other, list(deg_rect))
        else:
            faces_out[box.name][face] = rect
        placements.append((box, face, x, y, w, h))
        x += w + 2 * padding
        row_h = max(row_h, h)

    height = 1 << max(8, math.ceil(math.log2(max(64, y + row_h + padding))))
    return faces_out, placements, width, height


def _paint_plane_alpha(
    box: WorldBox,
    role: str,
    w: int,
    h: int,
    seed: int,
) -> np.ndarray:
    """Build crisp binary `{0, 255}` silhouette cutout mask for a `plane`."""
    edge_decl = getattr(box, "edge", None)
    if edge_decl is not None:
        alpha = compute_plane_edge_alpha_mask(h, w, edge_decl)
    else:
        alpha = np.full((h, w), 255, dtype=np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        u = (xx + 0.5) / max(1.0, float(w))
        v = (yy + 0.5) / max(1.0, float(h))
        if role == "plane_fur":
            # Spiky fur locks tapering out at bottom/outer edge
            tooth = np.abs(((u * 7.0) % 1.0) * 2.0 - 1.0)
            keep = v < (0.94 - 0.34 * tooth)
            alpha = np.where(keep, 255, 0).astype(np.uint8)
        elif role == "plane_plasma":
            # Blazing tapered jet lance
            mid_d = np.abs(v - 0.5) * 2.0
            keep = mid_d < (1.02 - 0.72 * np.power(u, 1.35))
            alpha = np.where(keep, 255, 0).astype(np.uint8)
        else:
            # Scalloped trailing membrane edge
            arch = 0.18 * np.abs(np.sin(np.pi * u * 2.0))
            alpha = np.where(v <= (0.96 - arch), 255, 0).astype(np.uint8)

    # Ensure outer trailing corners gently taper rather than showing 90-deg rectangle corners
    if h >= 6 and w >= 6 and role in ("plane_membrane", "plane_fin", "plane_fur", "plane_crystal"):
        alpha[-1, 0] = 0
        alpha[-1, -1] = 0
        if h >= 10 and w >= 10:
            alpha[-2, 0] = 0
            alpha[-1, 1] = 0
            alpha[-2, -1] = 0
            alpha[-1, -2] = 0
    if not np.any(alpha > 0):
        alpha[h // 2, w // 2] = 255
    return alpha


def bake_artisan_sculpted_atlas(
    program: Program,
    scene: Scene,
    *,
    artisan_motif: str = "default",
    accent_rgb: tuple[int, int, int] = (96, 224, 255),
    textures_dir: Path | None = None,
    padding: int = 2,
) -> dict[str, Any]:
    """Bake vibrant x32 RGBA `albedo.png` + `emissive.png` for a compiled Block-DSL `Scene`."""
    faces_uv, placements, width, height = allocate_artisan_x32_uvs(program, scene, padding=padding)
    albedo = np.zeros((height, width, 4), dtype=np.uint8)
    emissive = np.zeros((height, width, 4), dtype=np.uint8)

    pal_by_name: dict[str, Palette] = {p.name: p for p in program.palettes}
    default_pal = Palette("default", (0.55, 0.08, 220.0), 6, (18.0, -14.0), False)
    ramps: dict[str, np.ndarray] = {
        name: build_vibrant_6step_ramp(pal) for name, pal in pal_by_name.items()
    }
    ramps["default"] = build_vibrant_6step_ramp(default_pal)

    mat_decls = {m.name: m for m in program.materials}
    stamps_by_target: dict[tuple[str, str], list[Any]] = {}
    for st in program.stamps:
        stamps_by_target.setdefault((st.target_box, st.face), []).append(st)
        if st.mirror:
            mirror_face = {"east": "west", "west": "east"}.get(st.face, st.face)
            stamps_by_target.setdefault((st.target_box, mirror_face), []).append(st)

    # Compute global scene Y & Z bounds for smooth anatomical countershading & horn-chain gradients
    all_pts = np.vstack([b.corners() for b in scene.boxes]) if scene.boxes else np.zeros((1, 3))
    y_lo, y_hi = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
    y_span = max(1.0, y_hi - y_lo)

    acc = np.asarray(accent_rgb, dtype=np.uint8)

    face_normals_local = {
        "north": np.array([0.0, 0.0, -1.0]),
        "south": np.array([0.0, 0.0, 1.0]),
        "east": np.array([1.0, 0.0, 0.0]),
        "west": np.array([-1.0, 0.0, 0.0]),
        "up": np.array([0.0, 1.0, 0.0]),
        "down": np.array([0.0, -1.0, 0.0]),
    }

    for idx, (box, face, x0, y0, w, h) in enumerate(placements):
        pal = pal_by_name.get(box.material, default_pal)
        ramp = ramps.get(box.material, ramps["default"])
        mdecl = mat_decls.get(box.material)
        m_fam = mdecl.family if mdecl is not None else "keratin_scales"
        m_coat = mdecl.coat if mdecl is not None else None
        m_emissive = float(mdecl.emissive) if mdecl is not None else (0.90 if pal.emissive else 0.0)
        if pal.emissive and m_emissive < 0.5:
            m_emissive = 0.88

        role = _infer_part_role(box, m_fam, m_coat)
        n_world = box.rotation @ face_normals_local[face]
        ny = float(n_world[1])
        seed = (hash(box.name) ^ (idx * 2654435761)) & 0x7FFFFFFF

        rr, cc = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
        u = cc / max(1.0, float(w - 1)) if w > 1 else np.full((h, w), 0.5)
        v = rr / max(1.0, float(h - 1)) if h > 1 else np.full((h, w), 0.5)
        ri = rr.astype(np.int64)
        ci = cc.astype(np.int64)

        # Base directional cel band index (0..5), centered at 3 (true base_lch midtone)
        # Stepped diagonal cluster boundary (clean broad cel bands, zero checkerboard noise!)
        face_bias = 0.85 * ny
        step_jag = (((ci // 3 + ri // 2) % 2).astype(np.float64) - 0.5) * 0.18
        band_f = 3.0 + face_bias + 1.18 * (0.5 - v) + step_jag

        em_mask = np.zeros((h, w), dtype=np.float64)
        override_rgb: np.ndarray | None = None
        override_sel = np.zeros((h, w), dtype=bool)

        if role == "scales":
            # Stepped diagonal overlapping scale lips (denser on upper surfaces, calm on flanks)
            cell_w = 4 if max(w, h) >= 12 else 3
            cell_h = 3
            row_id = ri // cell_h
            col_shift = (row_id % 2) * (cell_w // 2)
            lx = (ci + col_shift) % cell_w
            ly = ri % cell_h
            h_frac = (float(box.center[1]) - y_lo) / y_span
            # Sparse deliberate clusters (~30% of scale cells active -> 70/25/5 calm hierarchy!)
            cell_keep = _hash2((ci + col_shift) // cell_w, row_id, seed) < (0.42 if ny > 0.15 else 0.24)
            scale_lip = cell_keep & (ly == cell_h - 1) & (lx > 0) & (lx < cell_w - 1)
            scale_crevice = cell_keep & (ly == 0) & (lx == cell_w // 2)
            band_f = band_f + np.where(scale_lip, 0.95, 0.0) - np.where(scale_crevice, 0.85, 0.0) + 0.35 * (h_frac - 0.45)

        elif role == "limb_scales":
            # SKILL.md line 371: "Bands on limbs: one dark row in five, not alternating halves."
            scute_band = (ri % 5) == 4
            scute_hi = (ri % 5) == 0
            band_f = band_f - np.where(scute_band, 1.05, 0.0) + np.where(scute_hi, 0.65, 0.0)

        elif role in ("armor", "metal_armor"):
            # Roof-tile shingle / carapace plate: crisp 1px sunlit top bevel, longitudinal flutes, deep overlap base
            flute = ((ci % 4) == 1) & (ri > 0) & (ri < h - 1)
            band_f = 3.15 + 0.75 * ny + 1.15 * (0.5 - v) + np.where(flute, 0.45, 0.0)
            if h >= 3:
                band_f[0, :] += 1.2
                band_f[-1, :] -= 1.4
            if w >= 4:
                band_f[:, 0] += 0.45
                band_f[:, -1] -= 0.65
            # Species-authentic glowing energy fissures on charged armor/intake plates!
            has_fissure = any(
                k in box.name.lower() or k in box.bone_name.lower()
                for k in ("capacitor", "fissure", "sternum", "intake", "charged", "fulgur", "draco", "hellfire", "dynamo", "wylk_crack", "collar", "pylon")
            ) or (
                artisan_motif in ("zinogre_thunder", "zinogre_stygian", "rathalos_silver", "lagiacrus_abyssal")
                and any(k in box.name.lower() for k in ("spur", "shingle", "pauldron"))
                and h >= 5
                and w >= 4
            )
            if has_fissure and h >= 4 and w >= 4:
                vein = (np.abs((ci - w * 0.5) - np.sin(v * 4.5) * 1.4) <= 0.65) & (ri >= 1) & (ri < h - 1)
                if artisan_motif.startswith("zinogre") and h >= 8:
                    # Dual electric fulgur-cyan/crimson charge bands like @zyhfisme's Zinogre crest!
                    band_glow = ((ri == max(1, h // 3)) | (ri == min(h - 2, (2 * h) // 3))) & (ci >= 1) & (ci < w - 1)
                    vein = vein | band_glow
                override_rgb = np.broadcast_to(acc, (h, w, 3)).copy()
                override_sel = vein
                em_mask = np.where(vein, 0.92, em_mask)

        elif role == "horn":
            # Longitudinal chitin ridges + subtle growth rings + dark-tip / rich-base keratin contrast
            ridge = (ci % 3) == 1
            ring = ((ri // 3) % 2 == 0) if h >= 8 else np.zeros((h, w), dtype=bool)
            band_f = 2.2 + 2.3 * (1.0 - v) + np.where(ridge, 0.65, -0.20) + np.where(ring, 0.45, -0.25) + 0.45 * ny

        elif role == "belly":
            # Matte underside scute with crisp anterior overlap shadow seam
            band_f = 3.25 + 0.45 * (1.0 - v)
            if h >= 3:
                band_f[0, :] -= 1.45  # Tucked shadow under previous overlapping scute!
                band_f[-1, :] += 0.55

        elif role == "fur":
            # Windswept longitudinal lock fibers (dark root -> bright tip)
            strand = ((ci + (ri // 3)) % 3) == 0
            tuft_shadow = ((ci + 1) % 4 == 0) & (v < 0.65)
            band_f = 2.35 + 2.35 * v + np.where(strand, 0.85, 0.0) - np.where(tuft_shadow, 0.95, 0.0)

        elif role == "crystal":
            # Faceted 5-6 value prism shards with specular ridge
            facet = ((ci * 2 + ri) // max(2, w // 3)) % 3
            band_f = 2.8 + facet * 0.95 + 0.8 * (1.0 - v)
            if w >= 3:
                band_f[:, w // 2] = 5.0
            em_mask = np.maximum(em_mask, 0.45 if m_emissive > 0.05 else 0.18)

        elif role == "eye":
            band_f = np.full((h, w), 4.8)
            override_rgb = np.broadcast_to(ramp[5], (h, w, 3)).copy()
            if h >= 2 and w >= 2:
                # Dark fierce upper brow lid + vertical predator slit pupil + white specular pixel!
                override_rgb[0, :] = ramp[0]
                px_mid = w // 2
                override_rgb[:, px_mid] = np.clip(ramp[0].astype(np.int32) // 2, 8, 255).astype(np.uint8)
                override_rgb[min(h - 1, 1), max(0, px_mid - 1)] = np.array([255, 252, 240], dtype=np.uint8)
                override_sel = np.ones((h, w), dtype=bool)
            em_mask = np.full((h, w), 0.95)

        elif role == "tooth":
            band_f = 3.6 + 1.5 * (1.0 - v)

        elif role in ("plane_membrane", "plane_fin", "plane_fur", "plane_plasma", "plane_crystal"):
            # Stepped spar-to-trailing-edge gradient + radial ribs + species wing motifs!
            rib = ((ci % max(4, w // 4)) == 0) & (v < 0.82)
            band_f = 4.2 - 2.1 * v + np.where(rib, 0.85, 0.0)
            if role == "plane_plasma":
                band_f = 5.0 - 2.4 * np.abs(v - 0.5) * 2.0 - 1.2 * u
                em_mask = np.clip(1.0 - 0.55 * u, 0.35, 1.0)
            elif role == "plane_crystal":
                band_f = 3.2 + 1.8 * (1.0 - v) + (((ci + ri) // 3) % 2) * 0.75
                em_mask = np.full((h, w), 0.38)
            elif artisan_motif.startswith("rathalos") and role == "plane_membrane" and w >= 8 and h >= 8:
                # Iconic Rathalos flame-scroll / wyvern-eye wing bay motif!
                ring_d = np.hypot((u - 0.52) * 1.35, (v - 0.46) * 1.55)
                scroll = (ring_d > 0.24) & (ring_d < 0.42) | ((ring_d < 0.13) & (v < 0.55))
                band_f = np.where(scroll, band_f - 1.75, band_f)
            elif artisan_motif.startswith("legiana") and role == "plane_membrane":
                # Pleated cryo-striations fanning across kite-leaf lobes
                pleat = ((ci + ri // 2) % 4) == 0
                band_f = band_f + np.where(pleat, 0.95, -0.25)
            elif artisan_motif.startswith("gore") and role == "plane_membrane":
                # Eclipse-violet spore flecks near trailing cloak edge
                spore = (_hash2(ci, ri, seed + 77) > 0.94) & (v > 0.35)
                override_rgb = np.broadcast_to(acc, (h, w, 3)).copy()
                override_sel = spore
                em_mask = np.where(spore, 0.82, em_mask)
            elif artisan_motif == "mizutsune_sakura" and role == "plane_fin" and w >= 6 and h >= 6:
                # Bewitching iris-petal eye-spot ocellus on Mizutsune fins!
                oc_d = np.hypot((u - 0.5) * 1.4, (v - 0.42) * 1.4)
                ring_violet = (oc_d > 0.16) & (oc_d < 0.32)
                core_gold = oc_d <= 0.16
                override_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                override_rgb[ring_violet] = np.array([76, 36, 112], dtype=np.uint8)
                override_rgb[core_gold] = np.array([250, 214, 112], dtype=np.uint8)
                override_sel = ring_violet | core_gold

        # Rim highlight & contact shadow on 3D cuboids (`SKILL.md` line 26)
        if getattr(box, "plane_normal", None) is None and h >= 3 and role not in ("eye", "tooth", "belly"):
            if ny >= -0.25:
                band_f[0, :] = np.maximum(band_f[0, :], 4.35)
            if ny <= 0.65:
                band_f[-1, :] = np.minimum(band_f[-1, :], 1.25)

        b_idx = np.clip(np.rint(band_f).astype(np.int64), 0, 5)
        rgb_island = ramp[b_idx].copy()
        if override_rgb is not None and np.any(override_sel):
            rgb_island = np.where(override_sel[..., None], override_rgb, rgb_island)

        if m_emissive > 0.15:
            em_mask = np.maximum(em_mask, m_emissive)

        # Apply any authored `@stamps` on `(box.name, face)` + soft 2px glow bleed!
        base_unmirrored = box.name[:-2] + "_l" if box.name.endswith("_r") else box.name
        st_list = stamps_by_target.get((box.name, face), []) + (
            stamps_by_target.get((base_unmirrored, face), []) if base_unmirrored != box.name else []
        )
        for st in st_list:
            st_pal = pal_by_name.get(st.color or "", default_pal)
            st_rgb = ramps.get(st.color or "", ramps["default"])[4]
            tpu = max(1.0, float(program.meta.texel_density) / 16.0)
            sw = max(1, int(round(float(st.size[0]) * tpu)))
            sh = max(1, int(round(float(st.size[1]) * tpu)))
            for r_k in range(max(1, int(st.rows))):
                cu = int(round(float(st.at[0]) * (w - 1)))
                cv = int(round(float(st.at[1]) * (h - 1) + r_k * float(st.spacing) * tpu))
                u_a, u_b = max(0, cu - sw // 2), min(w, cu - sw // 2 + sw)
                v_a, v_b = max(0, cv - sh // 2), min(h, cv - sh // 2 + sh)
                if u_b > u_a and v_b > v_a:
                    rgb_island[v_a:v_b, u_a:u_b] = st_rgb
                    if float(st.emissive) > 0.05 or st_pal.emissive:
                        em_mask[v_a:v_b, u_a:u_b] = max(0.85, float(st.emissive))

        # Soft 1-2 px glow bleed onto neighboring texels around emissive pixels (`SKILL.md` line 30)
        if np.any(em_mask > 0.25) and h >= 3 and w >= 3:
            ys_e, xs_e = np.nonzero(em_mask > 0.25)
            if 0 < len(ys_e) <= 64:
                rgb_f = rgb_island.astype(np.float64)
                for sy, sx in zip(ys_e.tolist(), xs_e.tolist()):
                    src_c = rgb_f[sy, sx]
                    for dy in (-2, -1, 0, 1, 2):
                        for dx in (-2, -1, 0, 1, 2):
                            ny_p, nx_p = sy + dy, sx + dx
                            if 0 <= ny_p < h and 0 <= nx_p < w and em_mask[ny_p, nx_p] <= 0.25:
                                d = math.hypot(dy, dx)
                                if d <= 2.2:
                                    w_b = 0.28 * (1.0 - d / 2.5)
                                    rgb_f[ny_p, nx_p] = rgb_f[ny_p, nx_p] * (1.0 - w_b) + src_c * w_b
                rgb_island = np.clip(np.rint(rgb_f), 0, 255).astype(np.uint8)

        # Alpha mask (`{0, 255}` on planes, `255` on solid cuboids)
        if getattr(box, "plane_normal", None) is not None:
            alpha_island = _paint_plane_alpha(box, role, w, h, seed)
        else:
            alpha_island = np.full((h, w), 255, dtype=np.uint8)

        rgba_island = np.concatenate((rgb_island, alpha_island[..., None]), axis=-1)
        albedo[y0 : y0 + h, x0 : x0 + w] = rgba_island

        # Edge-bleed RGB (not alpha on cutouts!) into padding border to prevent seam fringes
        if padding > 0 and getattr(box, "plane_normal", None) is None:
            for p in range(1, padding + 1):
                albedo[y0 - p, x0 : x0 + w] = rgba_island[0]
                albedo[y0 + h - 1 + p, x0 : x0 + w] = rgba_island[-1]
                albedo[y0 - p : y0 + h + p, x0 - p] = albedo[y0 - p : y0 + h + p, x0]
                albedo[y0 - p : y0 + h + p, x0 + w - 1 + p] = albedo[y0 - p : y0 + h + p, x0 + w - 1]

        if np.any(em_mask > 0.05):
            em_u8 = np.rint(np.clip(em_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
            em_u8 = np.where(alpha_island >= 128, em_u8, 0)
            emissive[y0 : y0 + h, x0 : x0 + w, :3] = (
                rgb_island.astype(np.float64) * (em_u8[..., None].astype(np.float64) / 255.0)
            ).astype(np.uint8)
            emissive[y0 : y0 + h, x0 : x0 + w, 3] = em_u8

    buf_alb = io.BytesIO()
    Image.fromarray(albedo, mode="RGBA").save(buf_alb, format="PNG", optimize=False)
    alb_bytes = buf_alb.getvalue()

    buf_em = io.BytesIO()
    Image.fromarray(emissive, mode="RGBA").save(buf_em, format="PNG", optimize=False)
    em_bytes = buf_em.getvalue()

    alb_path_str = None
    em_path_str = None
    if textures_dir is not None:
        textures_dir.mkdir(parents=True, exist_ok=True)
        ap = textures_dir / "albedo.png"
        ep = textures_dir / "emissive.png"
        ap.write_bytes(alb_bytes)
        ep.write_bytes(em_bytes)
        alb_path_str = str(ap)
        em_path_str = str(ep)

    return {
        "width": width,
        "height": height,
        "padding": padding,
        "faces": faces_uv,
        "data_url": "data:image/png;base64," + base64.b64encode(alb_bytes).decode("ascii"),
        "emissive_data_url": "data:image/png;base64," + base64.b64encode(em_bytes).decode("ascii"),
        "albedo_rgba": albedo,
        "emissive_rgba": emissive,
        "albedo_path": alb_path_str,
        "emissive_path": em_path_str,
        "mode": "artisan_sculpted_x32_cel_atlas",
    }
