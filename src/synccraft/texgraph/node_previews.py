"""Real Procedural Visual Thumbnail & Live Animated Multi-Frame Preview Generator (`TexGraph v13`).

Evaluates EVERY registered TexGraph node (`210+` nodes) at crisp pixel-art preview resolution (`40x40`)
using deterministic reference inputs (scalar ridge field, circular sprite RGBA, binary mask, normal map,
material bus, and a miniature 3D baked cuboid mesh) and caches:
  1. Static swatch catalog (`generate_node_preview_catalog(size=40)`)
  2. Live multi-frame animated preview loops (`generate_node_anim_preview(op, size=48, frames=8)`)
     so every dynamic/animated node visibly plays motion in the Add-Node Browser & Inspector!
"""
from __future__ import annotations

import base64
import io
import threading
from typing import Any

import numpy as np
from PIL import Image

from .core import NodeSpec, TexGraph
from .generators_gradients import multi_stop_gradient
from .generators_noise import fbm_noise, uv_grid
from .mesh_baker import MeshBakeContext, bake_bbmodel_maps
from .pbr import height_to_normal_map
from .pipeline import material_make
from .pixel_art_ops import rgba_float_to_u8
from .registry import NODE_CATALOG

_CACHE_LOCK = threading.Lock()
_PREVIEW_CACHE: dict[int, dict[str, str]] = {}
_ANIM_CACHE: dict[tuple[str, int, int], list[str]] = {}
_ANIM_SHEET_CACHE: dict[tuple[int, int], dict[str, list[str]]] = {}
_REF_CACHE: dict[int, tuple[dict[str, Any], MeshBakeContext]] = {}


def _to_data_url(arr: Any, size: int) -> str:
    if isinstance(arr, dict):
        for k in ("albedo", "rgba", "height", "normal", "mask", "emissive"):
            if k in arr and isinstance(arr[k], np.ndarray):
                arr = arr[k]
                break
        else:
            for k, v in arr.items():
                if not str(k).startswith("__") and isinstance(v, np.ndarray):
                    arr = v
                    break
    a = np.asarray(arr)
    if a.dtype == np.uint8:
        if a.ndim == 2:
            u8 = np.stack((a, a, a, np.full_like(a, 255)), axis=-1)
        elif a.ndim == 3 and a.shape[2] == 3:
            u8 = np.concatenate((a, np.full(a.shape[:2] + (1,), 255, dtype=np.uint8)), axis=-1)
        else:
            u8 = a
    else:
        u8 = rgba_float_to_u8(a)
    im = Image.fromarray(u8, mode="RGBA")
    if im.size != (size, size):
        im = im.resize((size, size), resample=Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=3)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _build_mini_mesh_ctx(size: int) -> MeshBakeContext:
    """Fast 2-box sculpted cube fixture baked once at `size x size` for 3D-aware node previews."""
    w = size
    h = size
    half = size // 2
    bb = {
        "meta": {"format_version": "4.10", "model_format": "free"},
        "name": "preview_cube",
        "resolution": {"width": w, "height": h},
        "elements": [
            {
                "name": "body_core",
                "uuid": "e1",
                "from": [-4, 0, -4],
                "to": [4, 8, 4],
                "origin": [0, 4, 0],
                "rotation": [0, 0, 0],
                "faces": {
                    "north": {"uv": [2, 2, half - 1, half - 1]},
                    "south": {"uv": [half + 1, 2, w - 2, half - 1]},
                    "up": {"uv": [2, half + 1, half - 1, h - 2]},
                    "east": {"uv": [half + 1, half + 1, w - 2, h - 2]},
                },
            },
            {
                "name": "cap_trim",
                "uuid": "e2",
                "from": [-3, 8, -3],
                "to": [3, 11, 3],
                "origin": [0, 9.5, 0],
                "rotation": [0, 0, 0],
                "faces": {
                    "up": {"uv": [4, 4, half - 3, half - 3]},
                },
            },
        ],
        "outliner": ["e1", "e2"],
    }
    return bake_bbmodel_maps(bb, ao_samples=2, bake_shadows=True, solve_uv_seams=False)


def _build_reference_inputs(size: int) -> dict[str, Any]:
    """Deterministic stylized reference height, mask, RGBA sprite, normal & material bus for filter previews."""
    u, v = uv_grid(size, size)
    noise_base = fbm_noise(size, size, scale=4.0, octaves=3, seed=19, tileable=True)
    dist = np.sqrt(((u - 0.5) * 2.15) ** 2 + ((v - 0.5) * 2.15) ** 2)
    mask = (dist <= 0.78).astype(np.float32)
    mask_b = (np.abs(u - 0.5) <= 0.32).astype(np.float32)
    dome = np.clip(1.0 - dist / 0.85, 0.0, 1.0).astype(np.float32)
    height = np.clip(0.55 * dome + 0.45 * noise_base, 0.0, 1.0).astype(np.float32)

    tile_rgba = multi_stop_gradient(
        height,
        [(0.0, "#1C1838"), (0.35, "#3E5094"), (0.68, "#62B8D9"), (1.0, "#FAF2D6")],
        space="oklab",
        smooth_blend=0.75,
        quantize_steps=6,
    )
    sprite_rgba = tile_rgba.copy()
    sprite_rgba[..., 3] = mask

    alt_h = np.clip(1.0 - np.abs(u - v), 0.0, 1.0).astype(np.float32)
    alt_rgba = multi_stop_gradient(
        alt_h,
        [(0.0, "#3D1828"), (0.5, "#C86832"), (1.0, "#FCE298")],
        space="oklab",
        smooth_blend=0.75,
        quantize_steps=5,
    )
    nrm = height_to_normal_map(height, strength=2.6, tileable=True)
    mat_base = material_make(tile_rgba, height=height, normal=nrm, roughness=0.52, metallic=0.12)
    mat_top = material_make(alt_rgba, height=alt_h, roughness=0.26, metallic=0.82)
    return {
        "u": u,
        "v": v,
        "scalar": height,
        "height": height,
        "mask": mask,
        "mask_b": mask_b,
        "rgba": tile_rgba,
        "sprite_rgba": sprite_rgba,
        "alt_scalar": alt_h,
        "alt_rgba": alt_rgba,
        "normal": nrm,
        "material": mat_base,
        "material_top": mat_top,
        "side_sign": np.ones((size, size), dtype=np.float32),
    }


def _get_refs_and_mesh(sz: int) -> tuple[dict[str, Any], MeshBakeContext]:
    with _CACHE_LOCK:
        hit = _REF_CACHE.get(sz)
        if hit is not None:
            return hit
    pair = (_build_reference_inputs(sz), _build_mini_mesh_ctx(sz))
    with _CACHE_LOCK:
        _REF_CACHE[sz] = pair
    return pair


# Fast preview parameter overrides for compute-heavy simulation / multi-step nodes
_FAST_PREVIEW_PARAMS: dict[str, dict[str, Any]] = {
    "gray_scott": {"preset": "labyrinth", "steps": 75},
    "cellular_automata": {"preset": "cave", "steps": 3},
    "iterate": {"operation": "erode_weather", "steps": 3},
    "iterate_inline": {"mode": "warp_cascade", "steps": 3},
    "feedback_loop": {"loop_frames": 4, "pre_roll": 2},
    "particle_sim": {"count": 26, "trail_steps": 2},
    "fluid_smoke_sim": {"steps": 10},
    "nlm_denoise": {"patch_radius": 1, "search_radius": 2},
    "kuwahara": {"radius": 1},
    "bilateral": {"radius": 1},
    "tv_denoise": {"iterations": 5},
    "color_replace": {"src_palette": ["#3E5094", "#62B8D9"], "dst_palette": ["#B83258", "#FCE298"]},
    "region_base_fill": {"stops": ["#242240", "#6B689E", "#CBD0EC", "#FAFBFF"], "trim_edge": "bottom", "trim_color": "#ECC560", "trim_width_px": 2.0},
    "region_restrained_noise": {"pattern": "clump_scales", "amplitude": 0.22, "edge_restraint": 0.85},
    "region_bevel_and_ao": {"highlight_strength": 0.24, "shadow_strength": 0.26, "ao_strength": 0.30},
    "region_composite": {"seam_ao_strength": 0.35},
    "paint_entity_eyes_and_parts": {"eye_glow": 0.95},
    "bake_emissive_bounce": {"max_emitters": 16, "intensity": 0.6},
    "constant": {"value": "#4F8EF7"},
    "mask_uv_box": {"mode": "uv_rect", "u_min": 0.18, "u_max": 0.82, "v_min": 0.18, "v_max": 0.82},
    "mask_combine": {"mode": "xor"},
    "material_region_apply": {"opacity": 1.0, "seam_ao_strength": 0.30},
    "math_scalar": {"math_op": "sin", "value_b": 12.0},
    "map_range": {"from_min": 0.25, "from_max": 0.78, "interpolation": "smoothstep"},
    "clamp_value": {"min_val": 0.22, "max_val": 0.78},
    "mix_blend": {"blend_mode": "overlay", "fac_value": 0.65},
    "vector_math": {"vec_op": "normalize"},
    "fresnel_weight": {"ior": 1.65},
    "tex_coord": {"coord_type": "generated_uv"},
    "checker_basic": {"scale_x": 6, "scale_y": 6},
    "paint_mask": {"base_value": 0.08},
    "float_constant": {"value": 0.72},
    "color_constant": {"color": "#38BDF8"},
    "script_expr": {"mode": "rgb", "expr": "0.5 + 0.5 * sin(r_dist * 20.0 + ang * 2.0 - t * tau)", "expr_g": "0.5 + 0.5 * sin(r_dist * 20.0 - t * tau + 1.8)", "expr_b": "0.5 + 0.5 * cos(r_dist * 16.0 - t * tau)"},
}

# For animated multi-frame previews, ensure time-capable nodes visibly animate even if default speed is 0
_ANIM_PREVIEW_OVERRIDES: dict[str, dict[str, Any]] = {
    "perlin": {"loop_speed": 1.0},
    "fbm": {"loop_speed": 1.0},
    "voronoi": {"loop_speed": 1.0},
    "simplex_noise": {"loop_speed": 1.0},
    "bubble_noise": {"loop_speed": 1.0},
    "domain_warp": {"flow_speed": 1.0, "strength": 0.14},
    "noise_3d": {"flow_vec": (0.25, 0.15, 0.0)},
    "scroll": {"speed_x": 1.0, "speed_y": 1.0},
    "caustic_noise": {"loop_speed": 1.0},
    "curl_flow_noise": {"loop_speed": 1.0},
    "starfield_sparkle": {"twinkle_speed": 1.0},
    "plasma_wave": {"speed": 1.0},
    "clockwork_gears": {"rotate_speed": 1.0},
    "anim_shimmer": {"speed": 1.0, "intensity": 0.85},
    "anim_flicker_pulse": {"freq": 2.0, "min_val": 0.35, "max_val": 1.35},
    "anim_flow_warp": {"speed": 1.0, "strength": 0.10},
}

_SPRITE_INPUT_OPS: frozenset[str] = frozenset({
    "pixel_outline",
    "directional_selout",
    "selout",
    "bloom_glow",
    "pixel_mirror",
    "de_stray",
    "de_corner",
})


def _fallback_swatch(cat: str, size: int) -> str:
    u, v = uv_grid(size, size)
    t = np.clip(0.55 * (1.0 - v) + 0.45 * u, 0.0, 1.0).astype(np.float32)
    pal = [(0.0, "#1E2538"), (0.5, "#3B629B"), (1.0, "#9CC5F8")]
    if "Material" in cat or "Character" in cat:
        pal = [(0.0, "#2A1F18"), (0.5, "#9A6C38"), (1.0, "#F6D98C")]
    elif "Region" in cat or "Mask" in cat:
        pal = [(0.0, "#1C1A38"), (0.5, "#7A74B8"), (1.0, "#F0F2FF")]
    elif "Sim" in cat or "Anim" in cat:
        pal = [(0.0, "#24102C"), (0.5, "#C23B72"), (1.0, "#FFD280")]
    rgba = multi_stop_gradient(t, pal, space="oklab", smooth_blend=0.8, quantize_steps=5)
    return _to_data_url(rgba, size)


def _eval_single_node_preview(
    op: str,
    sz: int,
    refs: dict[str, Any],
    mesh_ctx: MeshBakeContext,
    *,
    phase: float = 0.0,
    for_anim: bool = False,
) -> str:
    meta = NODE_CATALOG[op]
    try:
        g = TexGraph(sz, sz, tileable=True, seed=42, name=f"prev_{op}", mesh_context=mesh_ctx)
        if op == "feedback_loop":
            s_id = g.add("perlin", scale=5.0, loop_speed=1.0, seed=42)
            fb_id = g.add("feedback_loop", inputs={"src": s_id}, loop_frames=4, pre_roll=2, decay=0.82, drift_y_px=-1.0)
            g.set_output(fb_id)
            return _to_data_url(g.evaluate(phase=float(phase)).output, sz)
        if op == "paint_layer_stack":
            u, v = refs["u"], refs["v"]
            bg = np.zeros((sz, sz, 4), dtype=np.float32)
            bg[..., 0] = 0.16 + 0.08 * (1.0 - v)
            bg[..., 1] = 0.19 + 0.09 * (1.0 - v)
            bg[..., 2] = 0.27 + 0.10 * (1.0 - v)
            bg[..., 3] = 1.0
            # Diagonal metallic gold stroke + bright cyan arcane rune stroke
            d_gold = np.abs((v - 0.36) - 0.18 * np.sin(u * 5.0))
            m_gold = np.clip(1.0 - d_gold / 0.14, 0.0, 1.0) * ((u > 0.10) & (u < 0.90))
            d_cyan = np.abs((v - 0.68) + 0.15 * np.sin(u * 5.6))
            m_cyan = np.clip(1.0 - d_cyan / 0.12, 0.0, 1.0) * ((u > 0.14) & (u < 0.88))
            gold_c = np.array([0.94, 0.78, 0.34], dtype=np.float32)
            cyan_c = np.array([0.22, 0.94, 1.00], dtype=np.float32)
            bg[..., :3] = bg[..., :3] * (1.0 - m_gold[..., None]) + gold_c[None, None, :] * m_gold[..., None]
            bg[..., :3] = bg[..., :3] * (1.0 - m_cyan[..., None]) + cyan_c[None, None, :] * m_cyan[..., None]
            return _to_data_url(bg, sz)
        if op == "decal_stamp":
            from .v17_decal_atlas_ops import render_decal_stencil
            st = render_decal_stencil("chrono_sigil", size=sz)
            bg = np.zeros((sz, sz, 4), dtype=np.float32)
            bg[..., :3] = np.array([0.15, 0.16, 0.20], dtype=np.float32)
            bg[..., 3] = 1.0
            a = st["albedo"][..., 3:4]
            bg[..., :3] = bg[..., :3] * (1.0 - a) + st["albedo"][..., :3] * a
            return _to_data_url(bg, sz)
        if op == "mip_safe_dilate":
            from .v17_decal_atlas_ops import mip_safe_dilate_rgba
            dil = mip_safe_dilate_rgba(refs["sprite_rgba"], bleed_px=max(4, sz // 5), preserve_alpha=False)
            return _to_data_url(dil, sz)
        if op == "paint_mask":
            u, v = refs["u"], refs["v"]
            d_curve = np.abs((v - 0.5) - 0.24 * np.sin((u - 0.15) * 5.2))
            stroke = np.clip(1.0 - d_curve / 0.15, 0.0, 1.0) * ((u > 0.12) & (u < 0.88))
            m = np.clip(0.10 + 0.90 * (stroke ** 0.7), 0.0, 1.0).astype(np.float32)
            return _to_data_url(m, sz)
        if op == "float_constant":
            u, v = refs["u"], refs["v"]
            ramp = np.clip(0.18 + 0.72 * u, 0.0, 1.0).astype(np.float32)
            bar = ((np.abs(v - 0.5) < 0.14) & (u < 0.72)).astype(np.float32)
            out_fc = np.clip(ramp * 0.35 + bar * 0.65, 0.0, 1.0).astype(np.float32)
            return _to_data_url(out_fc, sz)

        _WIRE_OPT_OPS = frozenset({
            "math_scalar", "map_range", "clamp_value", "invert_field",
            "mix_blend", "rgb_combine", "rgb_separate", "vector_math",
            "uv_mapping", "fresnel_weight",
        })
        vals: dict[str, Any] = {}
        inputs: dict[str, str] = {}
        for idx, port in enumerate(meta.inputs):
            pname = port.name
            if port.optional and op in _WIRE_OPT_OPS:
                vid = f"_in_{idx}_{pname}"
                if pname in ("a", "src"):
                    vals[vid] = refs["rgba"] if op in ("mix_blend", "rgb_separate", "uv_mapping") else refs["height"]
                elif pname == "b":
                    vals[vid] = refs["alt_rgba"] if op == "mix_blend" else (refs["v"] if op == "rgb_combine" else refs["alt_scalar"])
                elif pname == "r":
                    vals[vid] = refs["u"]
                elif pname == "g":
                    vals[vid] = refs["v"]
                elif pname == "fac":
                    vals[vid] = refs["height"]
                elif pname in ("vec_a", "normal"):
                    vals[vid] = refs["normal"]
                elif pname == "vec_b":
                    vals[vid] = refs["alt_rgba"]
                else:
                    continue
                inputs[pname] = vid
                continue
            if port.optional and pname not in ("mask", "mask_b", "base", "top", "layer"):
                continue
            if port.optional and op in ("region_composite", "material_blend", "material_height_blend", "material_region_apply", "blend") and pname == "mask":
                vals["_in_mask"] = refs["mask"]
                inputs[pname] = "_in_mask"
                continue
            if port.optional and op == "mask_combine" and pname == "mask_b":
                vals["_in_mask_b"] = refs["mask_b"]
                inputs[pname] = "_in_mask_b"
                continue
            if port.optional:
                continue

            vid = f"_in_{idx}_{pname}"
            dt = port.dtype
            if pname in ("local_u", "span_u"):
                vals[vid] = refs["u"]
            elif pname in ("local_v", "trail_v"):
                vals[vid] = refs["v"]
            elif pname == "side_sign":
                vals[vid] = refs["side_sign"]
            elif dt == "MATERIAL":
                vals[vid] = refs["material_top"] if pname in ("top", "layer") else refs["material"]
            elif dt == "NORMAL":
                vals[vid] = refs["normal"]
            elif dt == "MASK":
                vals[vid] = refs["mask_b"] if pname == "mask_b" else refs["mask"]
            elif dt == "RGBA":
                if op in _SPRITE_INPUT_OPS:
                    vals[vid] = refs["sprite_rgba"]
                elif pname in ("top", "b", "layer", "color"):
                    vals[vid] = refs["alt_rgba"]
                else:
                    vals[vid] = refs["rgba"]
            elif dt == "ANY":
                if op in _SPRITE_INPUT_OPS:
                    vals[vid] = refs["sprite_rgba"]
                elif pname in ("top", "b", "layer"):
                    vals[vid] = refs["alt_rgba"]
                else:
                    vals[vid] = refs["rgba"]
            else:
                vals[vid] = refs["alt_scalar"] if pname in ("b", "slope_map") else refs["height"]
            inputs[pname] = vid

        params = {p.name: p.default for p in meta.params}
        if op in _FAST_PREVIEW_PARAMS:
            params.update(_FAST_PREVIEW_PARAMS[op])
        if for_anim and op in _ANIM_PREVIEW_OVERRIDES:
            params.update(_ANIM_PREVIEW_OVERRIDES[op])

        sp = NodeSpec(id="prev", op=op, inputs=inputs, params=params)
        res = g._eval_node(sp, vals, phase=float(phase))
        return _to_data_url(res, sz)
    except Exception:
        return _fallback_swatch(meta.category, sz)


def generate_node_preview_catalog(size: int = 40) -> dict[str, str]:
    """Generate (and memoize) real procedural PNG preview data URLs for all registered TexGraph nodes."""
    sz = max(24, min(64, int(size)))
    with _CACHE_LOCK:
        if sz in _PREVIEW_CACHE:
            return dict(_PREVIEW_CACHE[sz])

    refs, mesh_ctx = _get_refs_and_mesh(sz)
    out: dict[str, str] = {}
    for op in NODE_CATALOG:
        out[op] = _eval_single_node_preview(op, sz, refs, mesh_ctx, phase=0.20, for_anim=False)

    with _CACHE_LOCK:
        _PREVIEW_CACHE[sz] = dict(out)
    return out


def generate_node_anim_preview(op: str, *, size: int = 48, frames: int = 8) -> list[str]:
    """Generate (and memoize) an N-frame live looping animation preview for any dynamic TexGraph node."""
    if op not in NODE_CATALOG:
        raise ValueError(f"Unknown node op: {op!r}")
    sz = max(24, min(64, int(size)))
    nf = max(2, min(16, int(frames)))
    key = (op, sz, nf)
    with _CACHE_LOCK:
        hit = _ANIM_CACHE.get(key)
        if hit is not None:
            return list(hit)

    refs, mesh_ctx = _get_refs_and_mesh(sz)
    out_frames: list[str] = []
    for i in range(nf):
        ph = float(i) / float(nf)
        out_frames.append(_eval_single_node_preview(op, sz, refs, mesh_ctx, phase=ph, for_anim=True))

    with _CACHE_LOCK:
        _ANIM_CACHE[key] = list(out_frames)
    return out_frames


def generate_anim_preview_sheet(*, size: int = 36, frames: int = 6) -> dict[str, list[str]]:
    """Generate (and memoize) multi-frame animated swatches for EVERY time-aware node in one batch."""
    sz = max(24, min(48, int(size)))
    nf = max(2, min(12, int(frames)))
    key = (sz, nf)
    with _CACHE_LOCK:
        hit = _ANIM_SHEET_CACHE.get(key)
        if hit is not None:
            return dict(hit)

    out: dict[str, list[str]] = {}
    for op, meta in NODE_CATALOG.items():
        if meta.time_aware:
            out[op] = generate_node_anim_preview(op, size=sz, frames=nf)

    with _CACHE_LOCK:
        _ANIM_SHEET_CACHE[key] = dict(out)
    return out
