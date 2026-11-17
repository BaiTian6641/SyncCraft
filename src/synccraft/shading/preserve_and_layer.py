"""Phase 3 Non-Destructive Painted Texture Preserve-and-Layer, OKLCH Variants, & Selective Face Rebake (ADR 0013).

Implements the three artist texture workflows from the Blockbench Pipeline Roadmap:
1. **Hand-painted model (`preserve_and_layer_painted_texture`)**:
   - Keeps every original pixel untouched on the base layer (`original_pixels_changed == 0`).
   - Reads perceptual OKLCH palette statistics per part (`extract_painted_palette_profile`).
   - Fills only faces the artist left blank (`A == 0`) on a separate `blank_fill` layer.
   - Emits separate transparent RGBA `wear_layer` and `glow_layer` overlays for Blockbench `TextureLayer`s.
   - Generates perceptual OKLCH recolour variants (`generate_oklch_texture_variant`) preserving pixel clusters,
     achromatic outlines, and binary `{0, 255}` cut-out alpha masks.
2. **Selective edit (`rebake_selected_faces`)**:
   - Re-bakes only the UV rectangles of targeted boxes or `(box, face)` pairs, leaving every non-targeted pixel
     bit-for-bit identical.
"""

from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from synccraft.dsl.ast_nodes import FACES, Program
from synccraft.kinematics.se3_chain import Scene
from synccraft.surface.contracts import (
    WorldState,
    linear_srgb_to_oklch,
    linear_to_srgb_u8,
    oklch_to_linear_srgb,
    srgb_u8_to_linear,
)
from .compiler import compile_vanilla_atlas


def decode_rgba_png(png_or_rgba: bytes | np.ndarray) -> np.ndarray:
    """Decode PNG bytes or copy an `(H, W, 4)` `uint8` array into a fresh `uint8` RGBA buffer."""
    if isinstance(png_or_rgba, (bytes, bytearray)):
        with Image.open(io.BytesIO(png_or_rgba)) as im:
            return np.asarray(im.convert("RGBA"), dtype=np.uint8).copy()
    arr = np.asarray(png_or_rgba, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected (H, W, 3|4) image array, got shape {arr.shape}")
    if arr.shape[2] == 3:
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        return np.concatenate((arr, alpha), axis=2)
    return arr.copy()


def encode_rgba_png(rgba: np.ndarray) -> bytes:
    """Encode `(H, W, 4)` `uint8` RGBA array to deterministic PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgba, dtype=np.uint8), mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def rgba_to_data_url(rgba_or_png: np.ndarray | bytes) -> str:
    """Return a `data:image/png;base64,...` URL suitable for `.bbmodel` `textures` / `layers`."""
    raw = rgba_or_png if isinstance(rgba_or_png, (bytes, bytearray)) else encode_rgba_png(rgba_or_png)
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def alpha_over_rgba(bottom_rgba: np.ndarray, top_rgba: np.ndarray) -> np.ndarray:
    """Standard Porter-Duff `source-over` composite in linear sRGB space."""
    bot = np.asarray(bottom_rgba, dtype=np.uint8)
    top = np.asarray(top_rgba, dtype=np.uint8)
    if bot.shape != top.shape:
        raise ValueError(f"Layer shape mismatch: {bot.shape} vs {top.shape}")
    a_t = top[..., 3:4].astype(np.float64) / 255.0
    a_b = bot[..., 3:4].astype(np.float64) / 255.0
    a_out = a_t + a_b * (1.0 - a_t)

    lin_t = srgb_u8_to_linear(top[..., :3])
    lin_b = srgb_u8_to_linear(bot[..., :3])
    safe_a = np.maximum(1e-12, a_out)
    lin_out = np.where(a_out > 1e-12, (lin_t * a_t + lin_b * a_b * (1.0 - a_t)) / safe_a, 0.0)
    rgb_u8 = linear_to_srgb_u8(lin_out)
    alpha_u8 = np.rint(np.clip(a_out * 255.0, 0.0, 255.0)).astype(np.uint8)
    return np.concatenate((rgb_u8, alpha_u8), axis=-1)


def _iter_face_uv_rects(scene: Scene, atlas: dict[str, Any], width: int, height: int) -> list[tuple[str, str, tuple[int, int, int, int]]]:
    """Yield `(box_name, face_name, (x0, y0, x1, y1))` pixel bounds across `scene`."""
    faces_map = atlas.get("faces") or {}
    out: list[tuple[str, str, tuple[int, int, int, int]]] = []
    for box in scene.boxes:
        b_faces = faces_map.get(box.name) or {}
        uv_f_dict = dict(getattr(box, "uv_faces", None) or ())
        for fn in FACES:
            raw_uv = None
            if fn in b_faces:
                entry = b_faces[fn]
                raw_uv = entry.get("uv") if isinstance(entry, dict) else entry
            elif fn in uv_f_dict:
                raw_uv = uv_f_dict[fn]
            if raw_uv is None or len(raw_uv) < 4:
                continue
            x0 = int(max(0, min(width, math.floor(min(float(raw_uv[0]), float(raw_uv[2]))))))
            x1 = int(max(0, min(width, math.ceil(max(float(raw_uv[0]), float(raw_uv[2]))))))
            y0 = int(max(0, min(height, math.floor(min(float(raw_uv[1]), float(raw_uv[3]))))))
            y1 = int(max(0, min(height, math.ceil(max(float(raw_uv[1]), float(raw_uv[3]))))))
            if x1 > x0 and y1 > y0:
                out.append((box.name, fn, (x0, y0, x1, y1)))
    return out


def extract_painted_palette_profile(
    program: Program,
    scene: Scene,
    atlas: dict[str, Any],
    base_png_or_rgba: bytes | np.ndarray,
) -> dict[str, Any]:
    """Read perceptual OKLCH palette statistics and classify painted vs blank faces without mutating pixels."""
    rgba = decode_rgba_png(base_png_or_rgba)
    h_px, w_px = rgba.shape[:2]
    box_by_name = scene.box_by_name

    painted_faces: list[tuple[str, str]] = []
    blank_faces: list[tuple[str, str]] = []
    per_material_lch: dict[str, list[np.ndarray]] = {}
    per_box_summary: dict[str, dict[str, Any]] = {}

    for bname, fname, (x0, y0, x1, y1) in _iter_face_uv_rects(scene, atlas, w_px, h_px):
        patch = rgba[y0:y1, x0:x1]
        opaque = patch[..., 3] > 0
        if not np.any(opaque):
            blank_faces.append((bname, fname))
            continue
        painted_faces.append((bname, fname))
        rgb_u8 = patch[..., :3][opaque]
        lch = linear_srgb_to_oklch(srgb_u8_to_linear(rgb_u8))
        mat_name = box_by_name[bname].material if bname in box_by_name else "default"
        per_material_lch.setdefault(mat_name, []).append(lch)
        if bname not in per_box_summary:
            hue_r = np.deg2rad(lch[:, 2])
            mean_h = float(np.rad2deg(np.arctan2(np.mean(np.sin(hue_r)), np.mean(np.cos(hue_r)))) % 360.0)
            unique_colors = int(len(np.unique(rgb_u8.reshape(-1, 3), axis=0)))
            per_box_summary[bname] = {
                "material": mat_name,
                "mean_lch": [round(float(np.mean(lch[:, 0])), 4), round(float(np.mean(lch[:, 1])), 4), round(mean_h, 2)],
                "distinct_swatches": unique_colors,
            }

    inferred_ramps: dict[str, dict[str, Any]] = {}
    for mat_name, chunks in sorted(per_material_lch.items()):
        all_lch = np.concatenate(chunks, axis=0)
        hue_r = np.deg2rad(all_lch[:, 2])
        mean_h = float(np.rad2deg(np.arctan2(np.mean(np.sin(hue_r)), np.mean(np.cos(hue_r)))) % 360.0)
        quant_l = np.unique(np.round(all_lch[:, 0], 2))
        inferred_ramps[mat_name] = {
            "base_lch": [
                round(float(np.median(all_lch[:, 0])), 4),
                round(float(np.median(all_lch[:, 1])), 4),
                round(mean_h, 2),
            ],
            "lightness_span": [round(float(np.min(all_lch[:, 0])), 4), round(float(np.max(all_lch[:, 0])), 4)],
            "estimated_steps": int(max(1, min(12, len(quant_l)))),
            "pixel_count": int(all_lch.shape[0]),
        }

    return {
        "width": w_px,
        "height": h_px,
        "painted_face_count": len(painted_faces),
        "blank_face_count": len(blank_faces),
        "painted_faces": [[b, f] for b, f in painted_faces],
        "blank_faces": [[b, f] for b, f in blank_faces],
        "inferred_ramps": inferred_ramps,
        "per_box_summary": per_box_summary,
    }


def generate_oklch_texture_variant(
    base_png_or_rgba: bytes | np.ndarray,
    *,
    hue_shift_deg: float = 0.0,
    chroma_scale: float = 1.0,
    lightness_shift: float = 0.0,
    preserve_achromatic_threshold: float = 0.028,
    target_rects: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """Shift painted colors in perceptual Ottosson OKLCH while preserving pixel clusters, outlines, and alpha."""
    rgba = decode_rgba_png(base_png_or_rgba)
    out = rgba.copy()
    h_px, w_px = out.shape[:2]

    if target_rects is None:
        mask = out[..., 3] > 0
    else:
        region = np.zeros((h_px, w_px), dtype=bool)
        for x0, y0, x1, y1 in target_rects:
            region[max(0, y0) : min(h_px, y1), max(0, x0) : min(w_px, x1)] = True
        mask = region & (out[..., 3] > 0)

    if not np.any(mask):
        return out

    lin = srgb_u8_to_linear(out[..., :3][mask])
    lch = linear_srgb_to_oklch(lin)
    achromatic = lch[:, 1] < float(preserve_achromatic_threshold)

    new_l = np.clip(lch[:, 0] + float(lightness_shift), 0.0, 1.0)
    new_c = np.where(achromatic, lch[:, 1], np.clip(lch[:, 1] * float(chroma_scale), 0.0, 0.37))
    new_h = np.where(achromatic, lch[:, 2], (lch[:, 2] + float(hue_shift_deg)) % 360.0)

    recolored_lin = oklch_to_linear_srgb(np.stack((new_l, new_c, new_h), axis=-1))
    out_rgb = out[..., :3].copy()
    out_rgb[mask] = linear_to_srgb_u8(recolored_lin)
    out[..., :3] = out_rgb
    # Alpha channel is preserved bit-for-bit
    return out


def preserve_and_layer_painted_texture(
    program: Program,
    scene: Scene,
    atlas: dict[str, Any],
    base_png_or_rgba: bytes | np.ndarray,
    *,
    world_state: WorldState | None = None,
    wear_strength: float = 0.25,
    glow_strength: float = 0.0,
    glow_bleed: int = 2,
    fill_blank_faces: bool = True,
    variant_hue_shift_deg: float | None = None,
    variant_chroma_scale: float = 1.0,
    variant_lightness_shift: float = 0.0,
    seed: int = 101,
) -> dict[str, Any]:
    """Keep every original painted pixel untouched and emit non-destructive RGBA layers + optional OKLCH variant."""
    orig_rgba = decode_rgba_png(base_png_or_rgba)
    orig_snapshot = orig_rgba.copy()
    h_px, w_px = orig_rgba.shape[:2]

    profile = extract_painted_palette_profile(program, scene, atlas, orig_rgba)
    face_rects = _iter_face_uv_rects(scene, atlas, w_px, h_px)

    # 1. Blank-face fill layer (non-zero ONLY on faces whose entire UV rect had A == 0 in `orig_rgba`)
    blank_fill_rgba = np.zeros((h_px, w_px, 4), dtype=np.uint8)
    if fill_blank_faces and profile["blank_face_count"] > 0:
        if world_state is not None:
            baked_atlas, baked_png = compile_vanilla_atlas(world_state)
            baked_rgba = decode_rgba_png(baked_png)
        else:
            baked_atlas, baked_rgba = atlas, None
        blank_set = {(b, f) for b, f in profile["blank_faces"]}
        for bname, fname, (x0, y0, x1, y1) in face_rects:
            if (bname, fname) not in blank_set:
                continue
            if baked_rgba is not None and baked_atlas.get("width") == w_px and baked_atlas.get("height") == h_px:
                blank_fill_rgba[y0:y1, x0:x1] = baked_rgba[y0:y1, x0:x1]
            else:
                mat_name = scene.box_by_name[bname].material if bname in scene.box_by_name else ""
                pal = next((p for p in program.palettes if p.name == mat_name), program.palettes[0])
                fill_rgb = linear_to_srgb_u8(oklch_to_linear_srgb(pal.base_lch))
                blank_fill_rgba[y0:y1, x0:x1, :3] = fill_rgb
                blank_fill_rgba[y0:y1, x0:x1, 3] = 255

    # 2. Separate transparent RGBA wear layer (perimeter/bottom-edge dust & scuff; base layer untouched!)
    wear_layer_rgba = np.zeros((h_px, w_px, 4), dtype=np.uint8)
    w_str = float(np.clip(wear_strength, 0.0, 1.0))
    if w_str > 1e-6:
        rng = np.random.default_rng(int(seed))
        for bname, fname, (x0, y0, x1, y1) in face_rects:
            base_patch = orig_rgba[y0:y1, x0:x1]
            fill_patch = blank_fill_rgba[y0:y1, x0:x1]
            active_alpha = (base_patch[..., 3] > 0) | (fill_patch[..., 3] > 0)
            if not np.any(active_alpha):
                continue
            ph, pw = y1 - y0, x1 - x0
            rr, cc = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
            dist_edge = np.minimum(np.minimum(rr, ph - 1 - rr), np.minimum(cc, pw - 1 - cc))
            bottom_bias = (rr + 1) / float(max(1, ph))
            noise = rng.random((ph, pw))
            wear_mask = active_alpha & ((dist_edge == 0) | ((bottom_bias > 0.65) & (noise < 0.45 * w_str))) & (
                noise < (0.35 + 0.65 * w_str)
            )
            if not np.any(wear_mask):
                continue
            # Warm weathered edge highlight / dust speckling on a separate semi-transparent RGBA layer
            wear_rgb = np.array([198, 186, 164], dtype=np.uint8)
            wear_alpha = int(round(min(220.0, 165.0 * w_str + 25.0)))
            sub = wear_layer_rgba[y0:y1, x0:x1]
            sub[wear_mask, :3] = wear_rgb
            sub[wear_mask, 3] = wear_alpha

    # 3. Separate transparent RGBA glow/emissive layer (from emissive palettes, `@stamps`, or `glow_strength`)
    glow_layer_rgba = np.zeros((h_px, w_px, 4), dtype=np.uint8)
    emissive_mats = {p.name: p for p in program.palettes if p.emissive}
    for md in getattr(program, "materials", ()):
        if float(md.emissive) > 0.05:
            pal_m = next((p for p in program.palettes if p.name == md.name), None)
            if pal_m is not None:
                emissive_mats[md.name] = pal_m
    emissive_stamps = [st for st in getattr(program, "stamps", ()) if float(st.emissive) > 0.05 or st.color in emissive_mats]
    g_str = float(np.clip(glow_strength, 0.0, 1.0))
    if emissive_mats or emissive_stamps or g_str > 1e-6:
        core_mask = np.zeros((h_px, w_px), dtype=bool)
        glow_color_buf = np.zeros((h_px, w_px, 3), dtype=np.uint8)
        rect_lookup = {(bname, fname): (x0, y0, x1, y1) for bname, fname, (x0, y0, x1, y1) in face_rects}
        for bname, fname, (x0, y0, x1, y1) in face_rects:
            if bname not in scene.box_by_name:
                continue
            mat_n = scene.box_by_name[bname].material
            if mat_n in emissive_mats:
                pal_e = emissive_mats[mat_n]
                e_rgb = linear_to_srgb_u8(oklch_to_linear_srgb((min(0.95, pal_e.base_lch[0] + 0.12), pal_e.base_lch[1], pal_e.base_lch[2])))
                patch_a = (orig_rgba[y0:y1, x0:x1, 3] > 0) | (blank_fill_rgba[y0:y1, x0:x1, 3] > 0)
                core_mask[y0:y1, x0:x1] |= patch_a
                glow_color_buf[y0:y1, x0:x1][patch_a] = e_rgb
        for st in emissive_stamps:
            pal_st = next((p for p in program.palettes if p.name == st.color), program.palettes[0])
            st_rgb = linear_to_srgb_u8(oklch_to_linear_srgb((min(0.95, pal_st.base_lch[0] + 0.10), pal_st.base_lch[1], pal_st.base_lch[2])))
            cand_faces = (st.face,) if st.face != "side" else ("east", "west")
            for fn in cand_faces:
                rc = rect_lookup.get((st.target_box, fn))
                if rc is None:
                    continue
                rx0, ry0, rx1, ry1 = rc
                rw, rh = max(1, rx1 - rx0), max(1, ry1 - ry0)
                cx = int(round(rx0 + float(st.at[0]) * rw))
                cy = int(round(ry0 + float(st.at[1]) * rh))
                sx0 = max(rx0, cx - max(1, int(round(0.5 * float(st.size[0])))))
                sx1 = min(rx1, max(sx0 + 1, cx + max(1, int(round(0.5 * float(st.size[0]))))))
                sy0 = max(ry0, cy - max(1, int(round(0.5 * float(st.size[1])))))
                sy1 = min(ry1, max(sy0 + 1, cy + max(1, int(round(0.5 * float(st.size[1]))))))
                core_mask[sy0:sy1, sx0:sx1] = True
                glow_color_buf[sy0:sy1, sx0:sx1] = st_rgb
        if np.any(core_mask):
            glow_layer_rgba[core_mask, :3] = glow_color_buf[core_mask]
            glow_layer_rgba[core_mask, 3] = int(round(190 + 60 * max(g_str, 0.5)))
            bleed_r = max(0, min(6, int(glow_bleed)))
            if bleed_r > 0:
                ys, xs = np.nonzero(core_mask)
                for dy in range(-bleed_r, bleed_r + 1):
                    for dx in range(-bleed_r, bleed_r + 1):
                        dist = math.hypot(dx, dy)
                        if dist == 0.0 or dist > bleed_r + 0.25:
                            continue
                        ny = np.clip(ys + dy, 0, h_px - 1)
                        nx = np.clip(xs + dx, 0, w_px - 1)
                        falloff_a = int(round(110.0 * (1.0 - dist / (bleed_r + 0.75))))
                        upd = (~core_mask[ny, nx]) & (orig_rgba[ny, nx, 3] > 0) & (glow_layer_rgba[ny, nx, 3] < falloff_a)
                        glow_layer_rgba[ny[upd], nx[upd], :3] = glow_color_buf[ys[upd], xs[upd]]
                        glow_layer_rgba[ny[upd], nx[upd], 3] = falloff_a

    # 4. Optional perceptual OKLCH recolor variant (also preserves exact alpha & outlines)
    has_variant = (
        variant_hue_shift_deg is not None
        or abs(float(variant_chroma_scale) - 1.0) > 1e-6
        or abs(float(variant_lightness_shift)) > 1e-6
    )
    variant_rgba = (
        generate_oklch_texture_variant(
            orig_rgba,
            hue_shift_deg=float(variant_hue_shift_deg or 0.0),
            chroma_scale=float(variant_chroma_scale),
            lightness_shift=float(variant_lightness_shift),
        )
        if has_variant
        else None
    )

    # Verify strict non-destructive contract on `orig_rgba`
    changed_orig_px = int(np.sum(np.any(orig_rgba != orig_snapshot, axis=-1)))
    if changed_orig_px != 0:
        raise AssertionError("preserve_and_layer_painted_texture mutated original pixels")

    active_base = variant_rgba if variant_rgba is not None else orig_rgba
    comp = alpha_over_rgba(active_base, blank_fill_rgba)
    comp = alpha_over_rgba(comp, wear_layer_rgba)
    comp = alpha_over_rgba(comp, glow_layer_rgba)

    return {
        "original_pixels_changed": changed_orig_px,
        "palette_profile": profile,
        "original_rgba": orig_rgba,
        "original_png": encode_rgba_png(orig_rgba),
        "blank_fill_rgba": blank_fill_rgba,
        "blank_fill_png": encode_rgba_png(blank_fill_rgba),
        "wear_layer_rgba": wear_layer_rgba,
        "wear_layer_png": encode_rgba_png(wear_layer_rgba),
        "glow_layer_rgba": glow_layer_rgba,
        "glow_layer_png": encode_rgba_png(glow_layer_rgba),
        "variant_rgba": variant_rgba,
        "variant_png": encode_rgba_png(variant_rgba) if variant_rgba is not None else None,
        "composited_rgba": comp,
        "composited_png": encode_rgba_png(comp),
    }


def rebake_selected_faces(
    world_state: WorldState | Program,
    base_atlas: dict[str, Any],
    base_png_or_rgba: bytes | np.ndarray,
    selected_targets: set[str] | set[tuple[str, str]] | list[str | tuple[str, str]],
) -> dict[str, Any]:
    """Re-bake ONLY the UV rectangles of `selected_targets`, leaving all other pixels bit-for-bit identical."""
    if isinstance(world_state, Program):
        from synccraft.surface.contracts import MaterialProgram, StyleConfig
        from synccraft.surface.domain import build_surface_domain, initialize_surface_state

        mat_prog = MaterialProgram.from_program(world_state)
        style_cfg = StyleConfig.from_program(world_state)
        domain = build_surface_domain(world_state, mat_prog)
        ws = WorldState(
            program=world_state,
            material_program=mat_prog,
            surface_state=initialize_surface_state(domain),
            style_config=style_cfg,
        )
    else:
        ws = world_state

    orig_rgba = decode_rgba_png(base_png_or_rgba)
    h_px, w_px = orig_rgba.shape[:2]

    baked_atlas, baked_png = compile_vanilla_atlas(ws)
    baked_rgba = decode_rgba_png(baked_png)
    bh_px, bw_px = baked_rgba.shape[:2]

    target_boxes: set[str] = set()
    target_box_faces: set[tuple[str, str]] = set()
    for item in selected_targets:
        if isinstance(item, tuple) and len(item) == 2:
            target_box_faces.add((str(item[0]), str(item[1])))
        else:
            target_boxes.add(str(item))

    from synccraft.kinematics.se3_chain import compile_geometry

    scene = compile_geometry(ws.program)
    baked_rects = {
        (bname, fname): rect
        for bname, fname, rect in _iter_face_uv_rects(scene, baked_atlas, bw_px, bh_px)
    }

    out_rgba = orig_rgba.copy()
    targeted_mask = np.zeros((h_px, w_px), dtype=bool)
    updated_face_count = 0

    for bname, fname, (x0, y0, x1, y1) in _iter_face_uv_rects(scene, base_atlas, w_px, h_px):
        if bname in target_boxes or (bname, fname) in target_box_faces:
            src_r = baked_rects.get((bname, fname))
            if src_r is None:
                continue
            bx0, by0, bx1, by1 = src_r
            patch = baked_rgba[by0:by1, bx0:bx1]
            dst_h, dst_w = y1 - y0, x1 - x0
            if patch.shape[:2] != (dst_h, dst_w):
                patch = np.asarray(
                    Image.fromarray(patch, mode="RGBA").resize((dst_w, dst_h), resample=Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            out_rgba[y0:y1, x0:x1] = patch
            targeted_mask[y0:y1, x0:x1] = True
            updated_face_count += 1

    untouched_diff = int(np.sum(np.any(out_rgba[~targeted_mask] != orig_rgba[~targeted_mask], axis=-1)))
    if untouched_diff != 0:
        raise AssertionError("rebake_selected_faces modified pixels outside targeted UV rectangles")

    return {
        "updated_face_count": updated_face_count,
        "non_target_pixels_changed": untouched_diff,
        "rgba": out_rgba,
        "png": encode_rgba_png(out_rgba),
        "data_url": rgba_to_data_url(out_rgba),
    }
