"""Real Procedural Visual Thumbnail Generator for SyncCraft Studio Node Library & Add-Node Browser.

Evaluates every registered TexGraph node (`149` nodes) at crisp pixel-art preview resolution (`40x40`)
using deterministic reference inputs (scalar ridge field, circular/diamond sprite RGBA, binary mask,
normal map, material bus, and a miniature 3D baked cuboid mesh) and caches base64 PNG data URLs
in memory so artists can pick noises, patterns, materials, and FX visually in Grid or List mode!
"""
from __future__ import annotations

import base64
import io
import threading
from typing import Any

import numpy as np
from PIL import Image

from .core import TexGraph
from .generators_gradients import multi_stop_gradient
from .generators_noise import fbm_noise, uv_grid
from .mesh_baker import MeshBakeContext, bake_bbmodel_maps
from .pbr import height_to_normal_map
from .pipeline import material_make
from .pixel_art_ops import rgba_float_to_u8
from .registry import NODE_CATALOG

_CACHE_LOCK = threading.Lock()
_PREVIEW_CACHE: dict[int, dict[str, str]] = {}


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
    # Crisp beveled diamond/orb heightfield
    dist = np.sqrt(((u - 0.5) * 2.15) ** 2 + ((v - 0.5) * 2.15) ** 2)
    mask = (dist <= 0.78).astype(np.float32)
    dome = np.clip(1.0 - dist / 0.85, 0.0, 1.0).astype(np.float32)
    height = np.clip(0.55 * dome + 0.45 * noise_base, 0.0, 1.0).astype(np.float32)

    # Rich warm-to-cool stylized RGBA tile + masked sprite variant
    tile_rgba = multi_stop_gradient(
        height,
        [(0.0, "#1C1838"), (0.35, "#3E5094"), (0.68, "#62B8D9"), (1.0, "#FAF2D6")],
        space="oklab",
        smooth_blend=0.75,
        quantize_steps=6,
    )
    sprite_rgba = tile_rgba.copy()
    sprite_rgba[..., 3] = mask

    # Secondary RGBA & height for blend/composite previews
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
        "rgba": tile_rgba,
        "sprite_rgba": sprite_rgba,
        "alt_scalar": alt_h,
        "alt_rgba": alt_rgba,
        "normal": nrm,
        "material": mat_base,
        "material_top": mat_top,
        "side_sign": np.ones((size, size), dtype=np.float32),
    }


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
}

# Ops that read best with a hard-alpha masked sprite input (outline, selout, silhouette)
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
    """Clean crisp geometric fallback if an exotic node requires custom graph state."""
    u, v = uv_grid(size, size)
    t = np.clip(0.55 * (1.0 - v) + 0.45 * u, 0.0, 1.0).astype(np.float32)
    pal = [(0.0, "#1E2538"), (0.5, "#3B629B"), (1.0, "#9CC5F8")]
    if "Material" in cat or "Character" in cat:
        pal = [(0.0, "#2A1F18"), (0.5, "#9A6C38"), (1.0, "#F6D98C")]
    elif "Region" in cat:
        pal = [(0.0, "#1C1A38"), (0.5, "#7A74B8"), (1.0, "#F0F2FF")]
    elif "Sim" in cat or "Anim" in cat:
        pal = [(0.0, "#24102C"), (0.5, "#C23B72"), (1.0, "#FFD280")]
    rgba = multi_stop_gradient(t, pal, space="oklab", smooth_blend=0.8, quantize_steps=5)
    return _to_data_url(rgba, size)


def generate_node_preview_catalog(size: int = 40) -> dict[str, str]:
    """Generate (and memoize) real procedural PNG preview data URLs for all registered TexGraph nodes."""
    sz = max(24, min(64, int(size)))
    with _CACHE_LOCK:
        if sz in _PREVIEW_CACHE:
            return dict(_PREVIEW_CACHE[sz])

    refs = _build_reference_inputs(sz)
    mesh_ctx = _build_mini_mesh_ctx(sz)
    out: dict[str, str] = {}

    for op, meta in NODE_CATALOG.items():
        try:
            g = TexGraph(sz, sz, tileable=True, seed=42, name=f"prev_{op}", mesh_context=mesh_ctx)
            # Pre-seed synthetic input values directly into evaluation dict
            vals: dict[str, Any] = {}
            inputs: dict[str, str] = {}
            for idx, port in enumerate(meta.inputs):
                pname = port.name
                if port.optional and pname not in ("mask", "base", "top", "layer"):
                    continue
                if port.optional and op in ("region_composite", "material_blend", "material_height_blend", "blend") and pname == "mask":
                    vals["_in_mask"] = refs["mask"]
                    inputs[pname] = "_in_mask"
                    continue
                if port.optional and pname == "base" and op == "block_edge_transition":
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
                    vals[vid] = refs["mask"]
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
                    elif pname in ("top", "b"):
                        vals[vid] = refs["alt_rgba"]
                    else:
                        vals[vid] = refs["rgba"]
                else:
                    # SCALAR / HEIGHT
                    vals[vid] = refs["alt_scalar"] if pname in ("b", "disp_y_map") else refs["height"]
                inputs[pname] = vid

            params = {p.name: p.default for p in meta.params}
            if op in _FAST_PREVIEW_PARAMS:
                params.update(_FAST_PREVIEW_PARAMS[op])

            nid = g.add(op, id="target", inputs=inputs, validate_refs=False, **params)
            res = g._eval_node(g.nodes[nid], vals, 0.25)
            out[op] = _to_data_url(res, sz)
        except Exception:
            out[op] = _fallback_swatch(meta.category, sz)

    with _CACHE_LOCK:
        _PREVIEW_CACHE[sz] = dict(out)
    return out


__all__ = ["generate_node_preview_catalog"]
