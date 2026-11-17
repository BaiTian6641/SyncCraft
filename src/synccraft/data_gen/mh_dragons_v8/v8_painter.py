"""src/synccraft/data_gen/mh_dragons_v8/v8_painter.py

Calm Painterly x32 Pixel-Art Atlas Packer & Surface Compiler (`blockbench-sculpted-style`).

Aligned with `Lagiacrus, reconstructed` (Pages 1, 6), `zyhfisme` Zinogre/Elder Dragon, Red & Cream
Western Dragon, Tan Club-Tailed Wyvern, and Blue Ice Dragon:
  - True `x32` texel density (`2 texels per Blockbench unit` -> `1024x512` / `1024x1024` POT atlases).
  - Clean 5-step hue-shifted cel ramps (`cool violet/indigo shadow -> saturated midtone -> warm highlight`).
  - Deliberate, calm painterly surfaces (`70 / 25 / 5` visual hierarchy, ZERO noisy checkerboards):
    * `calm_flank` / `scales`: quiet rich base tone with sparse (~8%) direction-aware `2-3 px` arched
      highlight+shadow scale lips and `1 px` top/bottom bevels.
    * `armor_shingle`: clean overlapping plate face (`1 px` warm top bevel, calm saturated body, `1 px`
      cool-dark contact shadow line — zero zebra stripes).
    * `belly_scute`: matte parchment/underside plate with crisp `1 px` dark overlap seam line.
    * `horn_keratin`: muted natural matte gradient (dark root -> keratin mid -> warm worn tip).
    * `fur_mane`: directional longitudinal lock fibers with shaded root & lighter windswept tip.
    * `crystal_facet`: crisp faceted 5-value cel prism steps without blown-out white neon.
    * `membrane`: Bayer-stepped gradient from spar to trailing edge + species markings + binary alpha cut.
"""

from __future__ import annotations

import base64
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from synccraft.kinematics.se3_chain import euler_xyz_matrix

FACES: tuple[str, ...] = ("north", "south", "east", "west", "up", "down")
FACE_AXES: dict[str, tuple[int, int]] = {
    "north": (0, 1),
    "south": (0, 1),
    "east": (2, 1),
    "west": (2, 1),
    "up": (0, 2),
    "down": (0, 2),
}
NORMAL_FACES: dict[int, tuple[str, str]] = {
    0: ("east", "west"),
    1: ("up", "down"),
    2: ("south", "north"),
}


def face_vertices_local(lo: np.ndarray, hi: np.ndarray) -> dict[str, list[list[float]]]:
    x0, y0, z0 = float(lo[0]), float(lo[1]), float(lo[2])
    x1, y1, z1 = float(hi[0]), float(hi[1]), float(hi[2])
    return {
        "north": [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
        "south": [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        "east": [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
        "west": [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
        "up": [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
        "down": [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
    }


def native_uv_faces(elem: dict[str, Any]) -> dict[str, list[int]]:
    """Convert painter-space `[u0, v0, u1, v1]` into Blockbench outward-face UV coordinates."""
    out: dict[str, list[int]] = {}
    for face, rect in elem["uv"].items():
        u0, v0, u1, v1 = [int(x) for x in rect[:4]]
        if face in ("north", "east"):
            out[face] = [u1, v0, u0, v1]
        elif face == "up":
            out[face] = [u0, v1, u1, v0]
        else:
            out[face] = [u0, v0, u1, v1]
    return out


def prepare_elements_and_pack_uv(doc: dict[str, Any], texels_per_unit: int | None = None) -> tuple[list[dict[str, Any]], int, int]:
    """Pack cuboid & cutout-plane faces at true `x32` (`2 texels/unit`, `1024x512`–`1024x1024` POT)."""
    tpu = int(texels_per_unit if texels_per_unit is not None else doc.get("texels_per_unit", 2))
    boxes: list[dict[str, Any]] = []
    for e in doc["elements"]:
        c = np.asarray(e["center"], dtype=np.float64)
        s = np.asarray(e["size"], dtype=np.float64)
        rot = euler_xyz_matrix(e["rotation"])
        zero_axes = np.flatnonzero(s == 0.0)
        hidden: list[str] = []
        if len(zero_axes):
            hidden = [f for f in FACES if f not in NORMAL_FACES[int(zero_axes[0])]]
        fverts = {
            f: np.round((np.asarray(verts, dtype=np.float64) - c) @ rot.T + c, 6).tolist()
            for f, verts in face_vertices_local(c - 0.5 * s, c + 0.5 * s).items()
        }
        boxes.append(
            {
                **e,
                "min": (c - 0.5 * s).tolist(),
                "max": (c + 0.5 * s).tolist(),
                "uv": {},
                "hidden_faces": hidden,
                "face_vertices": fverts,
            }
        )

    jobs: list[tuple[int, int, str, str]] = []
    for b in boxes:
        zeros = [i for i, v in enumerate(b["size"]) if v == 0.0]
        if zeros:
            active_faces = (
                NORMAL_FACES[zeros[0]]
                if b.get("back_material") and b.get("back_material") != b["material"]
                else NORMAL_FACES[zeros[0]][:1]
            )
        else:
            active_faces = FACES
        for f in active_faces:
            ax_u, ax_v = FACE_AXES[f]
            w = max(2, int(round(float(b["size"][ax_u]) * tpu)))
            h = max(2, int(round(float(b["size"][ax_v]) * tpu)))
            jobs.append((h, w, b["name"], f))

    total_area = sum((w + 4) * (h + 4) for h, w, _, _ in jobs)
    max_w = max((w + 4 for _, w, _, _ in jobs), default=64)
    width = 2 ** max(9, math.ceil(math.log2(max(float(max_w), math.sqrt(total_area) * 1.18))))
    width = min(2048, width)

    by_name = {b["name"]: b for b in boxes}
    x, y, row_h = 2, 2, 0
    for h, w, bname, fname in sorted(jobs, reverse=True):
        if x + w + 2 > width:
            x = 2
            y += row_h + 4
            row_h = 0
        by_name[bname]["uv"][fname] = [x, y, x + w, y + h]
        x += w + 4
        row_h = max(row_h, h)

    height = 2 ** max(8, math.ceil(math.log2(max(64, y + row_h + 4))))
    for b in boxes:
        if len(b["uv"]) == 1:
            r0 = list(next(iter(b["uv"].values())))
            b["uv"] = {f: list(r0) for f in FACES}
        elif any(s == 0.0 for s in b["size"]):
            r0 = list(next(iter(b["uv"].values())))
            for f in FACES:
                b["uv"].setdefault(f, list(r0))
    return boxes, width, height


def build_hue_shifted_ramp(rgb: list[int] | tuple[int, int, int], *, emissive: float = 0.0) -> np.ndarray:
    """Build a crisp 5-step hue-shifted painterly Blockbench ramp (`0=deep cool shadow .. 4=warm highlight`).

    Never adds grey/black/white only:
      - Cool shadows shift toward deep indigo/violet/slate (`+B`, `-G`, slight `R/B` cool shift).
      - Highlights drift warm toward cream/ochre on warm & earth hides, or bright sky-aqua on blue/teal hides.
      - Matte non-emissive hides cap highlight headroom around ~240 so beveled highlights remain distinct.
    """
    c = np.asarray(rgb, dtype=np.float64)
    luma = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    if emissive <= 0.15:
        if luma > 214.0:
            c = c * (214.0 / luma)
        # Keep rich colors crisp while softening raw synthetic neon extremes
        chroma = float(np.max(c) - np.min(c))
        if chroma > 140.0:
            desat = min(0.18, (chroma - 140.0) / 420.0)
            c = c * (1.0 - desat) + luma * desat

    is_cool = float(c[2]) > float(c[0]) + 10.0
    sh_shift = np.array((-4.0, -10.0, 22.0)) if is_cool else np.array((4.0, -12.0, 18.0))
    hi_shift = np.array((-2.0, 20.0, 18.0)) if is_cool else np.array((20.0, 14.0, -4.0))

    steps = [
        c * 0.46 + sh_shift * 1.10,   # 0: deep shadow / overlap crevice
        c * 0.70 + sh_shift * 0.65,   # 1: cool lower shadow plane
        c * 1.00,                     # 2: calm canonical base midtone
        c * 1.13 + hi_shift * 0.65,   # 3: warm sunlit upper plane / scale lip
        c * 1.24 + hi_shift * 1.15,   # 4: crisp 1px top/leading edge highlight
    ]
    if emissive > 0.25:
        steps = [s + c * (0.22 * emissive) + 14.0 * emissive for s in steps]
    max_cap = 255.0 if emissive > 0.20 else 244.0
    return np.clip(np.rint(steps), 8, max_cap).astype(np.uint8)


def rasterize_polygon_mask(
    poly_uv: list[list[float]],
    w: int,
    h: int,
    *,
    bleed: float = 0.45,
    serrated: bool = False,
    tattered_holes: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Rasterize binary `{0, 255}` pixel-art wing/fin cutout alpha masks (zero smooth alpha blur)."""
    poly = np.asarray(poly_uv, dtype=np.float64)
    if len(poly) < 3:
        return np.full((h, w), 255, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    px_u = (xx + 0.5) / max(1, w)
    py_v = 1.0 - (yy + 0.5) / max(1, h)

    inside = np.zeros((h, w), dtype=bool)
    min_dist = np.full((h, w), 1e9, dtype=np.float64)
    scale = np.array([float(w), float(h)], dtype=np.float64)

    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > py_v) != (y2 > py_v)
        x_int = (x2 - x1) * (py_v - y1) / np.where(np.abs(y2 - y1) < 1e-12, 1e-12, y2 - y1) + x1
        inside ^= cond & (px_u < x_int)

        a = poly[i] * scale
        b_pt = poly[(i + 1) % n] * scale
        ab = b_pt - a
        ab_len2 = max(1e-12, float(np.dot(ab, ab)))
        p_grid = np.dstack((xx + 0.5, (h - 1 - yy) + 0.5))
        t = np.clip(((p_grid - a) @ ab) / ab_len2, 0.0, 1.0)
        proj = a + t[..., None] * ab
        dist = np.linalg.norm(p_grid - proj, axis=-1)
        min_dist = np.minimum(min_dist, dist)

    keep = inside | (min_dist <= bleed)

    if serrated and w >= 8 and h >= 8:
        phase = px_u * math.pi * 3.0 + (seed % 7) * 0.4
        scallop = 1.8 + 2.2 * np.abs(np.sin(phase))
        step_jag = ((xx // 2 + yy) % 3 == 0).astype(np.float64) * 1.4
        # Cut ONLY outer trailing margin (`py_v < 0.28`), keeping inner spar seams tight!
        erode = (min_dist < (scallop + step_jag)) & (py_v < 0.28)
        keep = keep & (~erode)

    if tattered_holes and w >= 14 and h >= 14:
        h1 = ((px_u - 0.68) ** 2 / 0.0085 + (py_v - 0.28) ** 2 / 0.0110) < 1.0
        h2 = ((px_u - 0.42) ** 2 / 0.0055 + (py_v - 0.22) ** 2 / 0.0075) < 1.0
        keep = keep & (~(h1 | h2))

    return np.where(keep, 255, 0).astype(np.uint8)


def _rasterize_special_cutout_mask(pattern: str, w: int, h: int, xx: np.ndarray, yy: np.ndarray) -> np.ndarray | None:
    """Hand-placed binary pixel-cutout silhouettes for teeth, irregular hood lobes, dorsal fins, fur & lightning."""
    pat = pattern.lower()
    tu = (xx + 0.5) / max(1, w)
    tv = 1.0 - (yy + 0.5) / max(1, h)

    if pat in ("tooth_skirt_upper", "tooth_skirt_lower"):
        # Clean pointy enamel teeth separated by crisp dark mouth gaps
        period = max(3, w // 6)
        local_x = (xx % period) / max(1.0, float(period - 1))
        tooth_shape = 1.0 - 2.0 * np.abs(local_x - 0.45)
        arch = 0.72 + 0.28 * np.sin(np.clip(tu, 0.02, 0.98) * math.pi)
        reach = np.clip(0.18 + 0.76 * tooth_shape * arch, 0.14, 0.96)
        dist = (1.0 - tv) if "upper" in pat else tv
        return np.where(dist <= reach, 255, 0).astype(np.uint8)

    if pat in ("spine_cutout", "dorsal_fin_cutout", "tail_paddle_fin"):
        # Asymmetric swept fin blades / irregular scalloped dorsal lobes (Lagiacrus Page 3 & Red/Tan Dragons)
        n_sp = 2 if pat == "tail_paddle_fin" else max(2, min(4, max(2, w // 10)))
        su = (tu * n_sp) % 1.0
        prof = np.where(su < 0.72, (su / 0.72) ** 0.70, np.maximum(0.0, 1.0 - (su - 0.72) / 0.25))
        height_mod = 0.78 + 0.22 * np.sin(tu * math.pi * 1.15)
        keep = tv <= (0.26 + 0.70 * prof * height_mod)
        return np.where(keep, 255, 0).astype(np.uint8)

    if pat in ("hood_lobe_cutout", "ear_frill_cutout"):
        # Organic scalloped creature hood/ear silhouette spanning 80-96% between rays with rooted base
        wave = 0.80 + 0.14 * np.sin(tv * math.pi * 2.2) + 0.06 * np.cos(tv * 8.5)
        keep = (tu <= np.clip(wave, 0.25, 0.98)) | (tu <= 0.30)
        return np.where(keep, 255, 0).astype(np.uint8)

    if pat in ("fur_tuft", "mane_tuft"):
        n_sp = max(3, min(6, max(3, w // 6)))
        phase = tu * n_sp * math.pi
        arch = np.sin(np.clip(tu, 0.03, 0.97) * math.pi) ** 0.42
        tip_h = (0.48 + 0.46 * np.abs(np.sin(phase))) * arch
        return np.where(tv <= tip_h, 255, 0).astype(np.uint8)

    if pat == "sakura_fin":
        arch = np.sin(np.clip(tu, 0.03, 0.97) * math.pi) ** 0.55
        scallop = 0.84 + 0.14 * np.cos(tu * math.pi * 6.0)
        return np.where(tv <= arch * scallop, 255, 0).astype(np.uint8)

    if pat == "lightning_cutout":
        zig = 0.50 + 0.24 * np.sin(tv * 9.0) + 0.10 * np.cos(tv * 19.0)
        branch = (np.abs(tu - (zig + 0.18 * (tv - 0.4))) < 0.09) & (tv > 0.32) & (tv < 0.84)
        keep = ((np.abs(tu - zig) <= 0.14) | branch) & (tv >= 0.04) & (tv <= 0.96)
        return np.where(keep, 255, 0).astype(np.uint8)

    return None


def _sample_face_coordinates(b: dict[str, Any], face: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:h, 0:w]
    tu = (xx + 0.5) / max(1, w)
    tv = 1.0 - (yy + 0.5) / max(1, h)
    if b.get("surface_vertices"):
        rot = euler_xyz_matrix(b["rotation"])
        lx = (tu - 0.5) * float(b["size"][0])
        ly = (tv - 0.5) * float(b["size"][1])
        if b.get("paint_uvs") and b.get("surface_uv"):
            A = np.c_[np.asarray(b["surface_uv"], dtype=np.float64), np.ones(len(b["surface_uv"]))]
            fit = np.linalg.lstsq(A, np.asarray(b["paint_uvs"], dtype=np.float64), rcond=None)[0]
            pu = np.clip(tu * fit[0, 0] + tv * fit[1, 0] + fit[2, 0], 0.0, 1.0)
            pv = np.clip(tu * fit[0, 1] + tv * fit[1, 1] + fit[2, 1], 0.0, 1.0)
            return tu, tv, pu, pv
        return tu, tv, tu, tv
    return tu, tv, tu, tv


def _stable_hash(name: str) -> int:
    h_val = 2166136261
    for ch in name:
        h_val = ((h_val ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return int(h_val)


def paint_face_patch(doc: dict[str, Any], b: dict[str, Any], face: str, w: int, h: int) -> tuple[Image.Image, Image.Image | None]:
    """Paint one crisp `x32` integer-pixel Blockbench face island (`w x h`) with calm painterly coats."""
    yy, xx = np.mgrid[0:h, 0:w]
    py = (h - 1) - yy  # 0 at bottom, h-1 at top
    px = xx

    mat_key = b.get("face_materials", {}).get(face) or b["material"]
    zeros = [i for i, s in enumerate(b["size"]) if s == 0.0]
    if zeros and face == ("west", "down", "north")[zeros[0]] and b.get("back_material"):
        mat_key = b["back_material"]

    mat = doc["materials"][mat_key]
    kind = str(mat.get("kind", "scales")).lower()
    coat = str(b.get("pattern") or mat.get("coat") or kind).lower()
    emissive_strength = float(mat.get("emissive", 0.0))
    ramp = build_hue_shifted_ramp(mat["color"], emissive=emissive_strength)
    tu, tv, pu, pv = _sample_face_coordinates(b, face, w, h)
    b_hash = _stable_hash(f"{b['name']}:{face}")

    # Soft 5-step directional form baseline (`calm`: zero confetti dither across midtones!)
    if face == "up":
        # Crown top light with clean 1-px highlight bevel around top edges
        idx = np.full((h, w), 3, dtype=np.int32)
        crown_strip = (np.abs(tu - 0.5) < 0.28) & (w >= 6)
        idx[crown_strip] = 4
    elif face == "down":
        idx = np.full((h, w), 0, dtype=np.int32)
        idx[(tv > 0.35) & (h >= 4)] = 1
    else:
        # Clean vertical 3-band form lighting: lower shadow -> wide calm midtone -> subtle top bevel!
        idx = np.full((h, w), 2, dtype=np.int32)
        idx[tv <= 0.22] = 1
        idx[tv >= 0.80] = 3

    emissive_mask = np.zeros((h, w), dtype=np.float32)

    # 1. Focal Eye (`2x1` or `2x2` slit with dark lid + 1px warm specular glint)
    if coat in ("eye_pixel", "eye") or kind == "eye":
        idx[:, :] = 2 if emissive_strength < 0.4 else 3
        if h >= 3 and w >= 3:
            idx[0, :] = 0  # dark upper brow contact lid line
            idx[min(1, h - 1), min(1, w - 1)] = 4
        emissive_mask[:, :] = max(0.75, emissive_strength)

    # 2. Pure Emissive Energy Cores / Flame Vents (`<= 3-5%` of monster!)
    elif coat in ("energy", "crystal_emissive") or emissive_strength >= 0.65:
        idx[:, :] = 2
        idx[tv >= 0.55] = 3
        center_glow = (np.abs(tu - 0.5) < 0.28) & (np.abs(tv - 0.5) < 0.32)
        idx[center_glow] = 4
        emissive_mask[:, :] = max(0.70, emissive_strength)
        emissive_mask[center_glow] = min(1.0, max(0.85, emissive_strength + 0.15))

    # 3. Snout & Muzzle (`darker lip shadow line + clean 1-2px nostril recess + gentle nasal ridge`)
    elif coat in ("snout_nostril", "muzzle"):
        if not zeros and face in ("north", "south", "east", "west") and h >= 4:
            idx[0, :] = 3
            idx[h - 1, :] = 1
        if face == "south" and w >= 5 and h >= 4:
            idx[h // 2 - 1, 1] = 0
            idx[h // 2 - 1, w - 2] = 0
        elif face in ("east", "west") and w >= 6 and h >= 4:
            nx = (w - 3) if face == "west" else 2
            ny = max(1, h // 3)
            idx[ny : ny + 1, max(0, nx - 1) : min(w, nx + 1)] = 0

    # 4. Teeth (`calm enamel white with subtle base gum shadow`)
    elif coat in ("tooth_skirt_upper", "tooth_skirt_lower") or kind == "bone":
        idx[:, :] = 3
        if "upper" in coat:
            idx[tv > 0.72] = 1
            idx[tv < 0.30] = 4
        elif "lower" in coat:
            idx[tv < 0.28] = 1
            idx[tv > 0.70] = 4
        else:
            idx[yy == h - 1] = 1

    # 5. Matte Belly Scutes & Throat Plates (`Lagiacrus Page 6`: calm cream + 1 px dark overlap shadow!)
    elif coat in ("belly_plates", "belly_scute", "belly", "ventral_scutes") or kind == "belly":
        idx[:, :] = 2
        if face == "down" or face == "south":
            dome = (np.abs(tu - 0.5) < 0.30)
            idx[dome] = 3
        if h >= 3:
            idx[0, :] = 3
            idx[h - 1, :] = 0  # dark overlap seam explaining plate stacking!
            # Subtle lateral edge tuck shadow on wide belly scutes
            if w >= 6:
                idx[:, 0] = np.minimum(idx[:, 0], 1)
                idx[:, w - 1] = np.minimum(idx[:, w - 1], 1)

    # 6. Overlapping Armor Shingles / Dorsal Shells (`1 px top highlight + clean saturated body + 1 px bottom overlap`, ZERO stripes!)
    elif coat in ("armor_shingle", "armor", "light_shingle", "carapace") or kind == "armor":
        idx[:, :] = 2
        if h >= 4:
            idx[0, :] = 3
            if face == "up" and w >= 4:
                idx[:, 0] = 3
                idx[h - 1, :] = 1
            else:
                idx[h - 1, :] = 0
        # Occasional single 2x1 weathered bevel chip (never repetitive stripes!)
        if w >= 6 and h >= 5 and (b_hash % 3 == 0):
            cx_n = (b_hash % max(2, w - 3)) + 1
            cy_n = 1
            idx[cy_n, cx_n : min(w - 1, cx_n + 2)] = 4

    # 7. Matte Keratin Horns, Claws & Dorsal Blades (smooth lengthwise grain + gentle root shade, zero cross-striping!)
    elif coat in ("horn_keratin", "horn_grain", "horn", "dorsal_blade", "claw") or kind == "horn":
        if face == "south":
            idx[:, :] = 3
        elif face == "north":
            idx[:, :] = 1
        elif face in ("east", "west", "up", "down"):
            grad_t = tu if face in ("east", "west") else tv
            perp = px if face in ("up", "down") else py
            grain = ((perp * 3 + (b_hash & 7)) % 5 == 0)
            idx[:, :] = 2
            idx[grad_t < 0.22] = 1
            idx[grad_t > 0.78] = 3
            idx[grain & (grad_t >= 0.22) & (grad_t <= 0.78)] = 1
        else:
            idx[:, :] = 2
        if not zeros and h >= 4:
            idx[0, :] = np.minimum(4, idx[0, :] + 1)
            idx[h - 1, :] = np.maximum(0, idx[h - 1, :] - 1)

    # 8. Fur & Mane Locks (`zyhfisme` Zinogre directional tapered strands along lock axis!)
    elif coat in ("fur", "fur_mane", "fur_tuft", "mane_tuft") or kind == "fur":
        grad_l = tv if zeros else (tu if face in ("east", "west") else tv)
        strand_id = (px if (zeros or face not in ("east", "west")) else py) // 2
        strand_jit = ((strand_id * 5 + (b_hash & 7)) % 3) - 1  # {-1, 0, +1} subtle strand tone shift
        idx = np.clip(np.rint(1.2 + 2.0 * grad_l + 0.35 * strand_jit).astype(int), 0, 4)

    # 9. Faceted Crystals (`Blue Ice Dragon` clean cel mineral planes, ZERO glow blowout!)
    elif coat in ("crystal_facet", "crystal") or kind == "crystal":
        diag_band = 0.55 * tv + 0.35 * tu + 0.10 * (((px + py) & 1) - 0.5)
        idx = np.clip(np.rint(0.5 + 3.4 * diag_band).astype(int), 0, 4)
        if not zeros and h >= 4:
            idx[0, :] = 4
            idx[h - 1, :] = 0
        if emissive_strength > 0.05:
            emissive_mask[:, :] = float(emissive_strength) * (0.4 + 0.6 * tv)

    # 10. Limb Segments (`1-in-6` subtle sparse band + darker ankle fade — `SKILL.md` line 262)
    elif coat == "leg_bands":
        idx[:, :] = 2
        idx[tv >= 0.76] = 3
        idx[tv <= 0.24] = 1
        band = (np.mod(py + (b_hash % 5), 6) == 2) & (h >= 6) & (tv > 0.18) & (tv < 0.85)
        idx[band] = 1

    # 11. Wing Membranes & Swimming Fins (calm painterly tonal gradient, organic subtle veins — ZERO zebra bars!)
    elif coat in ("membrane", "wing_rathalos", "wing_legiana", "wing_gore", "wing_shagaru", "wing_valstrax", "sakura_fin"):
        bayer2 = (((px & 1) * 2 + (py & 1)) - 1.5) * 0.10
        spar_grad = np.clip(0.88 - 0.56 * (1.0 - pv) - 0.14 * pu, 0.0, 1.0)
        idx = np.clip(np.rint(0.9 + 2.5 * spar_grad + bayer2).astype(int), 0, 4)
        if coat == "wing_rathalos":
            # Organic subtle flame-edge shading along trailing rim & single soft inner vein arch
            rim_dark = (pv < 0.20) | ((np.abs(pu - 0.5) < 0.06) & (pv < 0.72))
            idx[rim_dark] = np.maximum(0, idx[rim_dark] - 1)
        elif coat == "wing_legiana":
            # Crisp pale leading edge + delicate inner wingtip feather band (no diagonal zebra stripes!)
            lead_hi = pv > 0.76
            idx[lead_hi] = np.minimum(4, idx[lead_hi] + 1)
            rim_lo = (pv < 0.16) | ((np.abs(pu - 0.5) < 0.05) & (pv < 0.65))
            idx[rim_lo] = np.maximum(0, idx[rim_lo] - 1)
        elif coat == "wing_valstrax":
            streak = (np.abs(pu - 0.5) < 0.10) | (pv < 0.18)
            idx[streak] = 0
        elif coat in ("wing_gore", "wing_shagaru"):
            vein = (np.abs(pu - 0.5) < 0.06) | ((pv < 0.22) & (np.abs(pu - 0.3) < 0.08))
            idx[vein] = np.minimum(4, idx[vein] + 1)

    # 12. CALM PAINTERLY FLANK / HIDE (`Lagiacrus, reconstructed` Page 6: 88% clean base + SPARSE directional scale lips!)
    else:
        if face == "up" and w >= 5 and h >= 5:
            # Subtle dorsal crown highlight + sparse cluster along spine ridge
            idx[:, :] = 2
            ridge = np.abs(tu - 0.5) < 0.32
            idx[ridge] = 3
            idx[0, :] = np.maximum(idx[0, :], 3)
            # Sparse `2x1` head/back scale cluster accents (~7% pixels)
            gx, gy = px // 4, py // 4
            sel = (np.mod(gx * 7 + gy * 11 + (b_hash & 15), 13) == 0) & (np.mod(py, 4) == 1) & (np.mod(px, 4) <= 1)
            idx[sel] = 3
        elif face == "down":
            idx[:, :] = 1
        else:
            # Calm side flank: `88%` smooth clean step-2 base with soft top light & lower shadow,
            # plus ONLY sparse direction-aware 2-pixel scale lip pairs (1 highlight above 1 dark lip)!
            idx[:, :] = 2
            idx[tv >= 0.84] = 3
            idx[tv <= 0.16] = 1
            if w >= 6 and h >= 5:
                gx, gy = px // 5, py // 4
                seed_cell = np.mod(gx * 17 + gy * 31 + ((gy & 1) * 7) + (b_hash & 31), 17)
                lip_hit = (seed_cell == 0) & (tv > 0.22) & (tv < 0.78)
                lip_hi = lip_hit & (np.mod(py, 4) == 2) & (np.mod(px, 5) < 2)
                lip_lo = lip_hit & (np.mod(py, 4) == 1) & (np.mod(px, 5) < 2)
                idx[lip_hi] = 3
                idx[lip_lo] = 1

    # Sharp 1px top-light / bottom-contact bevel on non-plane cuboids (`SKILL.md` line 26!)
    if not zeros and h >= 3 and coat not in ("energy", "crystal_emissive"):
        if face in ("north", "south", "east", "west"):
            idx[0, :] = np.minimum(4, np.maximum(idx[0, :], idx[0, :] + 1))
            idx[h - 1, :] = np.maximum(0, np.minimum(idx[h - 1, :], idx[h - 1, :] - 1))

    rgb_arr = ramp[np.clip(idx, 0, 4)].copy()

    # Crisp 1px warm specular dot inside eye socket
    if coat in ("eye_pixel", "eye") or kind == "eye":
        if h >= 3 and w >= 3:
            rgb_arr[1, min(w - 2, 1)] = np.array([255, 250, 232], dtype=np.uint8)

    # Optional clean stepped root-to-tip transition (muted bone/ochre or accent tips)
    tip_rgb = b.get("tip_color") or mat.get("tip_color")
    if tip_rgb is not None:
        tip_em = float(mat.get("tip_emissive", b.get("tip_emissive", 0.0)))
        tip_ramp = build_hue_shifted_ramp(tip_rgb, emissive=tip_em)
        tip_arr = tip_ramp[np.clip(idx, 0, 4)]
        if zeros:
            grad_t = pv
        elif face == "south":
            grad_t = np.ones((h, w), dtype=np.float64)
        elif face == "north":
            grad_t = np.zeros((h, w), dtype=np.float64)
        elif face in ("east", "west"):
            grad_t = tu
        else:
            grad_t = tv
        step_cut = float(b.get("tip_start", 0.62))
        bayer_t = (((px & 1) * 2 + (py & 1)) - 1.5) * 0.08
        tip_sel = (grad_t + bayer_t) >= step_cut
        rgb_arr[tip_sel] = tip_arr[tip_sel]
        if tip_em > 0.15:
            emissive_mask[tip_sel] = max(0.55, tip_em)

    # Selective thin energy crevice hairline (ONLY where `seam_color` explicitly set, <= 3% surface, NO `+` crosses!)
    seam_rgb = b.get("seam_color") or mat.get("seam_color")
    if seam_rgb is not None and w >= 7 and h >= 6 and not zeros:
        diag_band = (px - py + (b_hash % 5) - 2)
        crack = (np.abs(diag_band) == 0) & (px > 1) & (px < w - 2) & (py > 1) & (py < h - 2)
        # Keep only one compact hairline fissure segment (`<= 45%` of diagonal)
        crack &= (px >= w // 4) & (px <= (3 * w) // 4)
        if np.any(crack):
            s_col = np.asarray(seam_rgb, dtype=np.float32)
            rgb_arr[crack] = np.clip(np.rint(rgb_arr[crack].astype(np.float32) * 0.25 + s_col * 0.75), 0, 255).astype(np.uint8)
            emissive_mask[crack] = 0.72

    # Binary `{0, 255}` cutout alpha mask
    special_a = _rasterize_special_cutout_mask(coat, w, h, xx, yy) if zeros else None
    if special_a is not None:
        a_mask = special_a
    elif "alpha" in b and zeros:
        alpha_cfg = b["alpha"]
        a_mask = rasterize_polygon_mask(
            alpha_cfg["polygon"],
            w,
            h,
            bleed=float(alpha_cfg.get("edge_bleed", 0.48)),
            serrated=bool(alpha_cfg.get("serrated_hem", False)) and (coat != "sakura_fin"),
            tattered_holes=bool(alpha_cfg.get("tattered_holes", False)) and (coat != "sakura_fin"),
            seed=b_hash,
        )
    else:
        a_mask = np.full((h, w), 255, dtype=np.uint8)

    rgba = np.dstack((rgb_arr, a_mask))
    em_img = None
    if np.any(emissive_mask > 0.15):
        mask_f = emissive_mask * (a_mask.astype(np.float32) / 255.0)
        em_rgb = np.clip(np.rint(rgb_arr.astype(np.float32) * mask_f[..., None]), 0, 255).astype(np.uint8)
        em_alpha = np.clip(np.rint(mask_f * 255.0), 0, 255).astype(np.uint8)
        em_img = Image.fromarray(np.dstack((em_rgb, em_alpha)), mode="RGBA")
    return Image.fromarray(rgba, mode="RGBA"), em_img


def bake_x32_atlas(
    doc: dict[str, Any],
    boxes: list[dict[str, Any]],
    width: int,
    height: int,
    out_folder: Path,
) -> dict[str, Any]:
    """Paint all packed islands into `albedo.png` + `emissive.png` with 2px gutter bleed on opaque solids."""
    out_folder.mkdir(parents=True, exist_ok=True)
    atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    emissive_atlas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    has_emissive = False
    painted_rects: set[tuple[int, int, int, int]] = set()

    for b in boxes:
        is_plane_cutout = any(s == 0.0 for s in b["size"])
        for f, rect in b["uv"].items():
            key = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
            if key in painted_rects:
                continue
            painted_rects.add(key)
            x0, y0, x1, y1 = key
            patch_im, em_im = paint_face_patch(doc, b, f, x1 - x0, y1 - y0)
            atlas.paste(patch_im, (x0, y0))

            if not is_plane_cutout:
                atlas.paste(patch_im.crop((0, 0, 1, patch_im.height)).resize((2, patch_im.height), Image.Resampling.NEAREST), (x0 - 2, y0))
                atlas.paste(patch_im.crop((patch_im.width - 1, 0, patch_im.width, patch_im.height)).resize((2, patch_im.height), Image.Resampling.NEAREST), (x1, y0))
                atlas.paste(patch_im.crop((0, 0, patch_im.width, 1)).resize((patch_im.width, 2), Image.Resampling.NEAREST), (x0, y0 - 2))
                atlas.paste(patch_im.crop((0, patch_im.height - 1, patch_im.width, patch_im.height)).resize((patch_im.width, 2), Image.Resampling.NEAREST), (x0, y1))

            if em_im is not None:
                has_emissive = True
                emissive_atlas.paste(em_im, (x0, y0))

    albedo_path = out_folder / "albedo.png"
    atlas.save(albedo_path, optimize=True)
    albedo_bytes = albedo_path.read_bytes()
    result: dict[str, Any] = {
        "width": width,
        "height": height,
        "albedo_path": str(albedo_path),
        "data_url": "data:image/png;base64," + base64.b64encode(albedo_bytes).decode("ascii"),
        "faces": {b["name"]: native_uv_faces(b) for b in boxes},
    }
    if has_emissive:
        em_path = out_folder / "emissive.png"
        emissive_atlas.save(em_path, optimize=True)
        result["emissive_path"] = str(em_path)
        result["emissive_data_url"] = "data:image/png;base64," + base64.b64encode(em_path.read_bytes()).decode("ascii")
    return result
