"""SyncCraft TexGraph v16 (`v52.0`) Commercial Workstation Pro Engines:

1. **Interactive 3D/2D Multi-Material Painting Layer Stack (`paint_layer_stack_op`)**:
   Substance-Painter-style layered PBR painting directly onto Albedo, Roughness, Metallic,
   Height (auto-converting micro-relief to tangent Normal!), and Emissive channels with
   non-destructive layer visibility, opacity, blend modes (`normal`, `multiply`, `screen`,
   `overlay`, `add`), and curated PBR brush presets (`gold`, `steel`, `arcane`, `rust`, `amethyst`, `eraser`).
2. **AST-Sandboxed Vectorized Script Node (`script_expr_op`)**:
   Zero-`eval`/zero-`exec` pure Python `ast.parse(mode="eval")` expression engine compiled directly
   onto vectorized NumPy arrays (`u, v, r_dist, ang, t, phase, a, b, sin, cos, mix, smoothstep,
   clamp, hash2, ...`) with strict AST allow-listing, depth/length bounds, and `ScriptSecurityError`.
3. **Distributable `.syncpack` Node Macro Marketplace (`synccraft.syncpack/1.0`)**:
   Curated built-in commercial macro packages + persistent `~/.synccraft/syncpacks/` library +
   1-click import, save, and collision-free macro spawning into any active `TexGraph`.
4. **Multi-Asset Batch Bakery (`scan_batch_folder`, `run_batch_bakery`)**:
   Batch-bakes folders of `.bbmodel` / `.texdsl` / `.syncproj` creatures & blocks with sibling
   auto-pairing, vectorized mesh-bake caching, automated QA scoring, `synccraft.pbr.bundle/2.0`
   + ASWF MaterialX `.mtlx` export, live progress callbacks, and JSON summary reports.
"""
from __future__ import annotations

import ast
import base64
import copy
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from .generators_noise import uv_grid
from .pbr import height_to_normal_map
from .pipeline import material_make
from .pixel_art_ops import blend_layers, hex_to_srgb, rgba_float_to_u8


# ===========================================================================
# 1. Interactive 3D/2D Multi-Material PBR Painting Layer Stack
# ===========================================================================

PBR_BRUSH_PRESETS: dict[str, dict[str, Any]] = {
    "gold_trim": {
        "title": "Celestial Gold",
        "color": "#F0C85A",
        "roughness": 0.18,
        "metallic": 0.95,
        "height": 0.78,
        "emissive_color": "#000000",
        "emissive_strength": 0.0,
        "paint_albedo": True,
        "paint_roughness": True,
        "paint_metallic": True,
        "paint_height": True,
        "paint_emissive": False,
    },
    "chipped_steel": {
        "title": "Forged Steel",
        "color": "#94A3B8",
        "roughness": 0.28,
        "metallic": 0.90,
        "height": 0.65,
        "emissive_color": "#000000",
        "emissive_strength": 0.0,
        "paint_albedo": True,
        "paint_roughness": True,
        "paint_metallic": True,
        "paint_height": True,
        "paint_emissive": False,
    },
    "arcane_glow": {
        "title": "Arcane Cyan Glow",
        "color": "#38F2FF",
        "roughness": 0.16,
        "metallic": 0.10,
        "height": 0.72,
        "emissive_color": "#38F2FF",
        "emissive_strength": 1.0,
        "paint_albedo": True,
        "paint_roughness": True,
        "paint_metallic": True,
        "paint_height": True,
        "paint_emissive": True,
    },
    "amethyst_rune": {
        "title": "Amethyst Rune",
        "color": "#B564FF",
        "roughness": 0.14,
        "metallic": 0.35,
        "height": 0.76,
        "emissive_color": "#C084FC",
        "emissive_strength": 0.85,
        "paint_albedo": True,
        "paint_roughness": True,
        "paint_metallic": True,
        "paint_height": True,
        "paint_emissive": True,
    },
    "worn_rust": {
        "title": "Weathered Rust",
        "color": "#8C3B22",
        "roughness": 0.86,
        "metallic": 0.08,
        "height": 0.36,
        "emissive_color": "#000000",
        "emissive_strength": 0.0,
        "paint_albedo": True,
        "paint_roughness": True,
        "paint_metallic": True,
        "paint_height": True,
        "paint_emissive": False,
    },
    "matte_ink": {
        "title": "Shadow Ink",
        "color": "#161824",
        "roughness": 0.90,
        "metallic": 0.0,
        "height": 0.25,
        "emissive_color": "#000000",
        "emissive_strength": 0.0,
        "paint_albedo": True,
        "paint_roughness": True,
        "paint_metallic": True,
        "paint_height": True,
        "paint_emissive": False,
    },
    "eraser": {
        "title": "Layer Eraser",
        "color": "#000000",
        "roughness": 0.5,
        "metallic": 0.0,
        "height": 0.5,
        "emissive_color": "#000000",
        "emissive_strength": 0.0,
        "erase": True,
    },
}


def make_empty_paint_layer(layer_id: str = "layer_1", name: str = "PBR Paint Layer 1") -> dict[str, Any]:
    return {
        "id": str(layer_id),
        "name": str(name),
        "visible": True,
        "opacity": 1.0,
        "blend_mode": "normal",
        "albedo_url": "",
        "pbr_url": "",  # Packed RGBA PNG: R=roughness, G=metallic, B=height, A=pbr_paint_weight
        "emissive_url": "",  # RGBA PNG: RGB=emissive color, A=emissive strength
    }


def parse_layers_json(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        layers = [dict(x) for x in raw if isinstance(x, dict)]
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            layers = [dict(x) for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []
        except Exception:
            layers = []
    else:
        layers = []
    if not layers:
        layers = [make_empty_paint_layer("layer_1", "PBR Paint Layer 1")]
    return layers


def _decode_rgba_url(url: str, W: int, H: int) -> np.ndarray | None:
    if not isinstance(url, str) or "," not in url:
        return None
    try:
        raw = base64.b64decode(url.split(",", 1)[1])
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        if im.size != (W, H):
            im = im.resize((W, H), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0
    except Exception:
        return None


def _encode_rgba_url(arr: np.ndarray) -> str:
    u8 = np.clip(np.round(np.asarray(arr, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="RGBA").save(buf, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def apply_pbr_brush_strokes_to_layer(
    layer: dict[str, Any],
    width: int,
    height: int,
    strokes: list[dict[str, Any]],
    *,
    brush: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Paint multi-channel PBR brush strokes onto a single `paint_layer_stack` layer."""
    W, H = max(1, int(width)), max(1, int(height))
    b = dict(brush or {})
    preset_id = str(b.get("preset") or "")
    if preset_id in PBR_BRUSH_PRESETS:
        merged = dict(PBR_BRUSH_PRESETS[preset_id])
        merged.update({k: v for k, v in b.items() if v is not None})
        b = merged

    is_erase = bool(b.get("erase", False))
    col_rgb = np.array(hex_to_srgb(str(b.get("color") or "#F0C85A")), dtype=np.float32)
    em_rgb = np.array(hex_to_srgb(str(b.get("emissive_color") or b.get("color") or "#38F2FF")), dtype=np.float32)
    rough_v = float(np.clip(b.get("roughness", 0.25), 0.04, 0.98))
    metal_v = float(np.clip(b.get("metallic", 0.90), 0.0, 1.0))
    height_v = float(np.clip(b.get("height", 0.72), 0.0, 1.0))
    em_str = float(np.clip(b.get("emissive_strength", 0.0), 0.0, 1.0))

    paint_alb = bool(b.get("paint_albedo", True))
    paint_pbr = bool(b.get("paint_roughness", True) or b.get("paint_metallic", True) or b.get("paint_height", True))
    paint_em = bool(b.get("paint_emissive", em_str > 0.01))

    alb = _decode_rgba_url(str(layer.get("albedo_url") or ""), W, H)
    if alb is None:
        alb = np.zeros((H, W, 4), dtype=np.float32)
    pbr = _decode_rgba_url(str(layer.get("pbr_url") or ""), W, H)
    if pbr is None:
        pbr = np.zeros((H, W, 4), dtype=np.float32)
        pbr[..., 0] = 0.55
        pbr[..., 1] = 0.0
        pbr[..., 2] = 0.50
        pbr[..., 3] = 0.0
    em = _decode_rgba_url(str(layer.get("emissive_url") or ""), W, H)
    if em is None:
        em = np.zeros((H, W, 4), dtype=np.float32)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for st in strokes:
        u = float(st.get("u", 0.5))
        v = float(st.get("v", 0.5))
        rad = max(1.0, float(st.get("radius_px", b.get("radius_px", 5.0))))
        soft = float(np.clip(st.get("softness", b.get("softness", 0.35)), 0.0, 0.95))
        opac = float(np.clip(st.get("opacity", b.get("opacity", 1.0)), 0.05, 1.0))
        cx = u * max(1, W - 1)
        cy = v * max(1, H - 1)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        inner = rad * (1.0 - soft)
        w_m = np.clip(1.0 - (d - inner) / max(0.5, rad - inner), 0.0, 1.0)
        w_m = (w_m * w_m * (3.0 - 2.0 * w_m)) * opac

        if is_erase:
            keep = 1.0 - w_m
            alb[..., 3] *= keep
            pbr[..., 3] *= keep
            em[..., 3] *= keep
        else:
            if paint_alb:
                alb[..., :3] = alb[..., :3] * (1.0 - w_m[..., None]) + col_rgb[None, None, :] * w_m[..., None]
                alb[..., 3] = np.clip(alb[..., 3] + w_m * (1.0 - alb[..., 3]), 0.0, 1.0)
            if paint_pbr:
                if b.get("paint_roughness", True):
                    pbr[..., 0] = pbr[..., 0] * (1.0 - w_m) + rough_v * w_m
                if b.get("paint_metallic", True):
                    pbr[..., 1] = pbr[..., 1] * (1.0 - w_m) + metal_v * w_m
                if b.get("paint_height", True):
                    pbr[..., 2] = pbr[..., 2] * (1.0 - w_m) + height_v * w_m
                pbr[..., 3] = np.clip(pbr[..., 3] + w_m * (1.0 - pbr[..., 3]), 0.0, 1.0)
            if paint_em and em_str > 0.005:
                em[..., :3] = em[..., :3] * (1.0 - w_m[..., None]) + em_rgb[None, None, :] * w_m[..., None]
                em[..., 3] = np.clip(em[..., 3] * (1.0 - w_m) + em_str * w_m, 0.0, 1.0)

    out_layer = dict(layer)
    out_layer["albedo_url"] = _encode_rgba_url(alb) if float(alb[..., 3].max()) > 1e-4 else ""
    out_layer["pbr_url"] = _encode_rgba_url(pbr) if float(pbr[..., 3].max()) > 1e-4 else ""
    out_layer["emissive_url"] = _encode_rgba_url(em) if float(em[..., 3].max()) > 1e-4 else ""
    return out_layer


def paint_layer_stack_op(
    height: int,
    width: int,
    *,
    base: dict[str, np.ndarray] | np.ndarray | None = None,
    layers_json: Any = "",
    height_normal_strength: float = 2.6,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Composite a Substance-Painter-style stack of multi-channel PBR paint layers onto a MATERIAL bus."""
    H, W = int(height), int(width)
    if isinstance(base, dict) and "albedo" in base:
        alb = np.asarray(base["albedo"], dtype=np.float32).copy()
        h_ch = np.asarray(base.get("height", np.full((H, W), 0.5, dtype=np.float32)), dtype=np.float32).copy()
        r_ch = np.asarray(base.get("roughness", np.full((H, W), 0.55, dtype=np.float32)), dtype=np.float32).copy()
        m_ch = np.asarray(base.get("metallic", np.zeros((H, W), dtype=np.float32)), dtype=np.float32).copy()
        ao_ch = np.asarray(base.get("ao", np.ones((H, W), dtype=np.float32)), dtype=np.float32).copy()
        em_ch = (
            np.asarray(base["emissive"], dtype=np.float32).copy()
            if "emissive" in base and base["emissive"] is not None
            else np.zeros((H, W, 4), dtype=np.float32)
        )
        sh_ch = base.get("shadow")
        bn_ch = base.get("bent_normal")
        th_ch = base.get("thickness")
        sss_ch = base.get("sss")
        por_ch = base.get("porosity")
        nrm_in = base.get("normal")
    elif isinstance(base, np.ndarray):
        arr = np.asarray(base, dtype=np.float32)
        if arr.ndim == 2:
            alb = np.stack((arr, arr, arr, np.ones_like(arr)), axis=-1)
        elif arr.shape[2] == 3:
            alb = np.concatenate((arr, np.ones((H, W, 1), dtype=np.float32)), axis=-1)
        else:
            alb = arr.copy()
        h_ch = np.full((H, W), 0.5, dtype=np.float32)
        r_ch = np.full((H, W), 0.55, dtype=np.float32)
        m_ch = np.zeros((H, W), dtype=np.float32)
        ao_ch = np.ones((H, W), dtype=np.float32)
        em_ch = np.zeros((H, W, 4), dtype=np.float32)
        sh_ch = bn_ch = th_ch = sss_ch = por_ch = nrm_in = None
    else:
        alb = np.zeros((H, W, 4), dtype=np.float32)
        alb[..., :3] = np.array([0.22, 0.25, 0.32], dtype=np.float32)
        alb[..., 3] = 1.0
        h_ch = np.full((H, W), 0.5, dtype=np.float32)
        r_ch = np.full((H, W), 0.58, dtype=np.float32)
        m_ch = np.zeros((H, W), dtype=np.float32)
        ao_ch = np.ones((H, W), dtype=np.float32)
        em_ch = np.zeros((H, W, 4), dtype=np.float32)
        sh_ch = bn_ch = th_ch = sss_ch = por_ch = nrm_in = None

    layers = parse_layers_json(layers_json)
    any_height_painted = False

    for lyr in layers:
        if not lyr.get("visible", True):
            continue
        opac = float(np.clip(lyr.get("opacity", 1.0), 0.0, 1.0))
        if opac <= 1e-4:
            continue
        bmode = str(lyr.get("blend_mode") or "normal").lower()

        alb_l = _decode_rgba_url(str(lyr.get("albedo_url") or ""), W, H)
        if alb_l is not None:
            w = np.clip(alb_l[..., 3] * opac, 0.0, 1.0)
            if float(w.max()) > 1e-4:
                if bmode == "normal":
                    alb[..., :3] = alb[..., :3] * (1.0 - w[..., None]) + alb_l[..., :3] * w[..., None]
                    alb[..., 3] = np.clip(np.maximum(alb[..., 3], w), 0.0, 1.0)
                else:
                    blended = blend_layers(alb, alb_l, mode=bmode, opacity=opac)
                    alb[..., :3] = alb[..., :3] * (1.0 - w[..., None]) + blended[..., :3] * w[..., None]
                    alb[..., 3] = np.clip(np.maximum(alb[..., 3], w), 0.0, 1.0)

        pbr_l = _decode_rgba_url(str(lyr.get("pbr_url") or ""), W, H)
        if pbr_l is not None:
            wp = np.clip(pbr_l[..., 3] * opac, 0.0, 1.0)
            if float(wp.max()) > 1e-4:
                r_ch = r_ch * (1.0 - wp) + pbr_l[..., 0] * wp
                m_ch = m_ch * (1.0 - wp) + pbr_l[..., 1] * wp
                h_ch = h_ch * (1.0 - wp) + pbr_l[..., 2] * wp
                any_height_painted = True

        em_l = _decode_rgba_url(str(lyr.get("emissive_url") or ""), W, H)
        if em_l is not None:
            we = np.clip(em_l[..., 3] * opac, 0.0, 1.0)
            if float(we.max()) > 1e-4:
                em_ch[..., :3] = np.clip(em_ch[..., :3] * (1.0 - we[..., None]) + em_l[..., :3] * we[..., None], 0.0, 1.0)
                em_ch[..., 3] = np.clip(np.maximum(em_ch[..., 3], we), 0.0, 1.0)

    if any_height_painted or nrm_in is None:
        nrm_from_h = height_to_normal_map(h_ch, strength=float(height_normal_strength), tileable=tileable)
        if nrm_in is not None and any_height_painted:
            # Reoriented blend of painted relief normal over incoming normal
            nrm_out = np.clip(0.55 * nrm_from_h + 0.45 * np.asarray(nrm_in, dtype=np.float32), 0.0, 1.0)
        else:
            nrm_out = nrm_from_h
    else:
        nrm_out = np.asarray(nrm_in, dtype=np.float32)

    mat = material_make(
        np.clip(alb, 0.0, 1.0),
        height=np.clip(h_ch, 0.0, 1.0),
        normal=nrm_out,
        roughness=np.clip(r_ch, 0.04, 0.98),
        metallic=np.clip(m_ch, 0.0, 1.0),
        ao=np.clip(ao_ch, 0.0, 1.0),
        emissive=em_ch if float(em_ch[..., 3].max()) > 1e-4 else None,
        shadow=sh_ch,
        bent_normal=bn_ch,
        thickness=th_ch,
        sss=sss_ch,
        porosity=por_ch,
    )
    return mat


# ===========================================================================
# 2. AST-Sandboxed Vectorized NumPy Script Expression Node (`script_expr`)
# ===========================================================================


class ScriptSecurityError(ValueError):
    """Raised when a `script_expr` contains any disallowed AST construct or exceeds safety bounds."""


_MAX_EXPR_LEN = 2048
_MAX_AST_NODES = 220
_MAX_AST_DEPTH = 28


def _safe_clamp(x: Any, lo: Any = 0.0, hi: Any = 1.0) -> np.ndarray:
    return np.clip(x, lo, hi)


def _safe_mix(a: Any, b: Any, t: Any) -> np.ndarray:
    return a * (1.0 - t) + b * t


def _safe_smoothstep(e0: Any, e1: Any, x: Any) -> np.ndarray:
    denom = np.where(np.abs(e1 - e0) < 1e-6, 1e-6, e1 - e0)
    t = np.clip((x - e0) / denom, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _safe_step(edge: Any, x: Any) -> np.ndarray:
    return (x >= edge).astype(np.float32)


def _safe_fract(x: Any) -> np.ndarray:
    return x - np.floor(x)


def _safe_hash2(x: Any, y: Any, seed: Any = 42.0) -> np.ndarray:
    h = np.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453123
    return (h - np.floor(h)).astype(np.float32)


def _safe_div(a: Any, b: Any) -> np.ndarray:
    return a / np.where(np.abs(b) < 1e-6, np.where(b >= 0, 1e-6, -1e-6), b)


def _safe_pow(a: Any, b: Any) -> np.ndarray:
    return np.power(np.maximum(np.abs(a), 1e-6), np.clip(b, -8.0, 8.0))


_SAFE_FUNCS: dict[str, Callable[..., Any]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": lambda x: np.clip(np.tan(x), -10.0, 10.0),
    "abs": np.abs,
    "sqrt": lambda x: np.sqrt(np.maximum(x, 0.0)),
    "exp": lambda x: np.exp(np.clip(x, -12.0, 12.0)),
    "log": lambda x: np.log(np.maximum(x, 1e-6)),
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "fract": _safe_fract,
    "sign": np.sign,
    "min": np.minimum,
    "max": np.maximum,
    "clamp": _safe_clamp,
    "mix": _safe_mix,
    "lerp": _safe_mix,
    "smoothstep": _safe_smoothstep,
    "step": _safe_step,
    "pow": _safe_pow,
    "atan2": np.arctan2,
    "hypot": np.hypot,
    "mod": lambda a, b: np.mod(a, np.where(np.abs(b) < 1e-6, 1.0, b)),
    "hash2": _safe_hash2,
    "wave": lambda x: 0.5 + 0.5 * np.sin(x * (2.0 * math.pi)),
}


def validate_safe_script_ast(expr: str) -> ast.Expression:
    """Parse and strictly allow-list validate an expression AST (zero `eval`/`exec`, zero attributes)."""
    s = str(expr or "").strip()
    if not s:
        raise ScriptSecurityError("Expression cannot be empty")
    if len(s) > _MAX_EXPR_LEN:
        raise ScriptSecurityError(f"Expression exceeds {_MAX_EXPR_LEN} character limit")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as exc:
        raise ScriptSecurityError(f"Expression syntax error: {exc.msg}") from exc

    node_count = 0

    def _walk(node: ast.AST, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_AST_NODES:
            raise ScriptSecurityError(f"Expression exceeds {_MAX_AST_NODES} AST nodes")
        if depth > _MAX_AST_DEPTH:
            raise ScriptSecurityError(f"Expression nesting exceeds {_MAX_AST_DEPTH} levels")

        if isinstance(node, ast.Expression):
            _walk(node.body, depth + 1)
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ScriptSecurityError(f"Only numeric constants are allowed, got {type(node.value).__name__}")
        elif isinstance(node, ast.Name):
            if node.id.startswith("_"):
                raise ScriptSecurityError(f"Forbidden identifier: {node.id!r}")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.UAdd, ast.USub)):
                raise ScriptSecurityError(f"Forbidden unary operator: {type(node.op).__name__}")
            _walk(node.operand, depth + 1)
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                raise ScriptSecurityError(f"Forbidden binary operator: {type(node.op).__name__}")
            _walk(node.left, depth + 1)
            _walk(node.right, depth + 1)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                raise ScriptSecurityError("Only single scalar/array comparisons (<, <=, >, >=, ==, !=) are allowed")
            _walk(node.left, depth + 1)
            for comp in node.comparators:
                _walk(comp, depth + 1)
        elif isinstance(node, ast.IfExp):
            _walk(node.test, depth + 1)
            _walk(node.body, depth + 1)
            _walk(node.orelse, depth + 1)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                fn = node.func.id if isinstance(node.func, ast.Name) else type(node.func).__name__
                raise ScriptSecurityError(f"Function {fn!r} is not in the safe shader math allowlist")
            if node.keywords:
                raise ScriptSecurityError("Keyword arguments are not permitted in script expressions")
            for arg in node.args:
                _walk(arg, depth + 1)
        else:
            raise ScriptSecurityError(f"Forbidden AST construct: {type(node).__name__}")

    _walk(tree, 0)
    return tree


def _eval_ast_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body, env)
    if isinstance(node, ast.Constant):
        return np.float32(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ScriptSecurityError(f"Unknown variable {node.id!r}; available: {', '.join(sorted(env))}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        v = _eval_ast_node(node.operand, env)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        l = _eval_ast_node(node.left, env)
        r = _eval_ast_node(node.right, env)
        if isinstance(node.op, ast.Add):
            return l + r
        if isinstance(node.op, ast.Sub):
            return l - r
        if isinstance(node.op, ast.Mult):
            return l * r
        if isinstance(node.op, ast.Div):
            return _safe_div(l, r)
        if isinstance(node.op, ast.FloorDiv):
            return np.floor(_safe_div(l, r))
        if isinstance(node.op, ast.Mod):
            return _SAFE_FUNCS["mod"](l, r)
        if isinstance(node.op, ast.Pow):
            return _safe_pow(l, r)
    if isinstance(node, ast.Compare):
        l = _eval_ast_node(node.left, env)
        r = _eval_ast_node(node.comparators[0], env)
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return (l < r).astype(np.float32)
        if isinstance(op, ast.LtE):
            return (l <= r).astype(np.float32)
        if isinstance(op, ast.Gt):
            return (l > r).astype(np.float32)
        if isinstance(op, ast.GtE):
            return (l >= r).astype(np.float32)
        if isinstance(op, ast.Eq):
            return (np.abs(l - r) < 1e-5).astype(np.float32)
        if isinstance(op, ast.NotEq):
            return (np.abs(l - r) >= 1e-5).astype(np.float32)
    if isinstance(node, ast.IfExp):
        cond = _eval_ast_node(node.test, env) > 0.5
        tb = _eval_ast_node(node.body, env)
        fb = _eval_ast_node(node.orelse, env)
        return np.where(cond, tb, fb)
    if isinstance(node, ast.Call):
        fn = _SAFE_FUNCS[node.func.id]  # type: ignore[union-attr]
        args = [_eval_ast_node(a, env) for a in node.args]
        return fn(*args)
    raise ScriptSecurityError(f"Unsupported AST node: {type(node).__name__}")


def _to_luma_and_rgb(arr: Any, H: int, W: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if arr is None:
        z = np.zeros((H, W), dtype=np.float32)
        return z, z, z, z
    if isinstance(arr, dict):
        arr = arr.get("albedo", arr.get("height"))
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        return a, a, a, a
    if a.ndim == 3 and a.shape[2] >= 3:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
        return lum, r, g, b
    z = np.zeros((H, W), dtype=np.float32)
    return z, z, z, z


def script_expr_op(
    height: int,
    width: int,
    *,
    a: Any = None,
    b: Any = None,
    expr: str = "0.5 + 0.5 * sin(r_dist * 18.0 - t * tau)",
    expr_g: str = "",
    expr_b: str = "",
    mode: str = "scalar",  # "scalar" | "rgb"
    param_k: float = 1.0,
    phase: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Evaluate a safe AST-sandboxed vectorized NumPy shader expression over `(H, W)` UV space."""
    H, W = int(height), int(width)
    u, v = uv_grid(H, W)
    cx = (u - 0.5).astype(np.float32)
    cy = (v - 0.5).astype(np.float32)
    r_dist = np.hypot(cx, cy).astype(np.float32)
    ang = np.arctan2(cy, cx).astype(np.float32)
    a_lum, a_r, a_g, a_b = _to_luma_and_rgb(a, H, W)
    b_lum, b_r, b_g, b_b = _to_luma_and_rgb(b, H, W)

    env: dict[str, Any] = {
        "u": u,
        "v": v,
        "x": u,
        "y": v,
        "cx": cx,
        "cy": cy,
        "r_dist": r_dist,
        "ang": ang,
        "t": np.float32(phase),
        "phase": np.float32(phase),
        "k": np.float32(param_k),
        "seed": np.float32(seed),
        "pi": np.float32(math.pi),
        "tau": np.float32(2.0 * math.pi),
        "a": a_lum,
        "b": b_lum,
        "a_r": a_r,
        "a_g": a_g,
        "a_b": a_b,
        "b_r": b_r,
        "b_g": b_g,
        "b_b": b_b,
    }

    def _run(ex: str) -> np.ndarray:
        tree = validate_safe_script_ast(ex)
        res = _eval_ast_node(tree, env)
        arr = np.broadcast_to(np.asarray(res, dtype=np.float32), (H, W)).copy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(arr, 0.0, 1.0)

    r_out = _run(expr)
    if str(mode).lower() == "rgb" or (expr_g.strip() or expr_b.strip()):
        g_out = _run(expr_g.strip() or expr)
        b_out = _run(expr_b.strip() or expr)
        a_out = np.ones((H, W), dtype=np.float32)
        return np.stack((r_out, g_out, b_out, a_out), axis=-1).astype(np.float32)
    return r_out.astype(np.float32)


# ===========================================================================
# 3. `.syncpack` Node Macro Package Marketplace (`synccraft.syncpack/1.0`)
# ===========================================================================


def _syncpack_dir() -> Path:
    d = Path.home() / ".synccraft" / "syncpacks"
    d.mkdir(parents=True, exist_ok=True)
    return d


BUILTIN_SYNCPACKS: list[dict[str, Any]] = [
    {
        "schema": "synccraft.syncpack/1.0",
        "id": "pack_rpg_forge",
        "name": "Stylized RPG Forge & Armor Pack",
        "author": "SyncCraft Core",
        "version": "1.0.0",
        "builtin": True,
        "color": "#B45309",
        "tags": ["metal", "armor", "forge", "rpg"],
        "description": "Production multi-node macros for battle-worn forged armor, guilded brass filigree, and riveted shield plates.",
        "macros": [
            {
                "id": "macro_battle_steel",
                "name": "Battle-Worn Forged Steel",
                "description": "Brushed dark steel with chipped edge wear and oil patina.",
                "frame_color": "#475569",
                "nodes": [
                    {"id": "steel", "op": "mat_brushed_metal", "inputs": {}, "params": {"base_color": "#64748B", "roughness": 0.28, "streak_scale": 14.0, "seed": 31}, "pos": [0, 0]},
                    {"id": "wear", "op": "material_weathering", "inputs": {"mat": "steel"}, "params": {"preset": "edge_wear", "amount": 0.36, "color": "#CBD5E1", "roughness_target": 0.16, "metallic_target": 0.96, "seed": 31}, "pos": [280, 0]},
                ],
                "output_node": "wear",
            },
            {
                "id": "macro_gilded_trim",
                "name": "Gilded Astrolabe Trim",
                "description": "High-metallic celestial brass with subtle recessed verdigris patina.",
                "frame_color": "#B45309",
                "nodes": [
                    {"id": "brass", "op": "mat_brass", "inputs": {}, "params": {"base_color": "#E6B84C", "patina_color": "#3E695C", "roughness": 0.20, "patina_amount": 0.14, "seed": 77}, "pos": [0, 0]},
                    {"id": "shimmer", "op": "anim_shimmer", "inputs": {"src": "brass"}, "params": {"angle_deg": -32.0, "speed": 1.0, "width": 0.12, "intensity": 0.42, "color": "#FFF6CC"}, "pos": [280, 0]},
                ],
                "output_node": "shimmer",
            },
        ],
    },
    {
        "schema": "synccraft.syncpack/1.0",
        "id": "pack_chrono_arcane",
        "name": "Chrono-Arcane VFX & Shaders Pack",
        "author": "SyncCraft Core",
        "version": "1.0.0",
        "builtin": True,
        "color": "#7E22CE",
        "tags": ["vfx", "animated", "crystal", "script"],
        "description": "Animated radial portal ripples, glowing rune circuits, and translucent amethyst crystal shader macros.",
        "macros": [
            {
                "id": "macro_portal_ripple",
                "name": "Chronal Portal Ripple (Script)",
                "description": "AST-sandboxed animated spiral-ring wave mapped through a luminous cyan-amethyst gradient.",
                "frame_color": "#7E22CE",
                "nodes": [
                    {
                        "id": "portal_wave",
                        "op": "script_expr",
                        "inputs": {},
                        "params": {
                            "expr": "0.5 + 0.5 * sin(r_dist * 24.0 + ang * 3.0 - t * tau)",
                            "mode": "scalar",
                            "param_k": 1.0,
                            "seed": 42,
                        },
                        "pos": [0, 0],
                    },
                    {
                        "id": "portal_grad",
                        "op": "multi_stop_gradient",
                        "inputs": {"src": "portal_wave"},
                        "params": {"stops": ["#120B2E", "#5B21B6", "#38BDF8", "#F0F9FF"], "quantize_steps": 6},
                        "pos": [280, 0],
                    },
                ],
                "output_node": "portal_grad",
            },
            {
                "id": "macro_crystal_core",
                "name": "Faceted Amethyst Crystal",
                "description": "Translucent violet crystal with animated internal cyan emissive sparkle.",
                "frame_color": "#6D28D9",
                "nodes": [
                    {"id": "gem", "op": "mat_crystal_gem", "inputs": {}, "params": {"crystal_color": "#9333EA", "core_glow_color": "#38F2FF", "opacity": 0.78, "facet_scale": 6.5, "glow_strength": 0.65, "roughness": 0.10, "seed": 88}, "pos": [0, 0]},
                ],
                "output_node": "gem",
            },
        ],
    },
    {
        "schema": "synccraft.syncpack/1.0",
        "id": "pack_shader_math",
        "name": "Procedural Math & Shader Lab Pack",
        "author": "SyncCraft Core",
        "version": "1.0.0",
        "builtin": True,
        "color": "#0369A1",
        "tags": ["math", "blender", "utility", "script"],
        "description": "Blender-inspired procedural math chains: masked FBM remapping, RGB iridescent Fresnel shader, and tiled UV warp.",
        "macros": [
            {
                "id": "macro_iridescent_rgb",
                "name": "Iridescent Prism Shader (RGB Script)",
                "description": "3-channel animated rainbow interference expression ready for holographic armor & wings.",
                "frame_color": "#0284C7",
                "nodes": [
                    {
                        "id": "prism_rgb",
                        "op": "script_expr",
                        "inputs": {},
                        "params": {
                            "mode": "rgb",
                            "expr": "0.5 + 0.5 * sin((u + v) * 8.0 - t * tau)",
                            "expr_g": "0.5 + 0.5 * sin((u + v) * 8.0 - t * tau + 2.094)",
                            "expr_b": "0.5 + 0.5 * sin((u + v) * 8.0 - t * tau + 4.188)",
                            "param_k": 1.0,
                            "seed": 19,
                        },
                        "pos": [0, 0],
                    },
                ],
                "output_node": "prism_rgb",
            },
            {
                "id": "macro_contrast_fbm",
                "name": "Smoothstep Remapped FBM Mask",
                "description": "Tileable FBM routed through Blender Map Range + Invert for crisp wear masks.",
                "frame_color": "#0F766E",
                "nodes": [
                    {"id": "fbm_src", "op": "fbm", "inputs": {}, "params": {"scale": 6.0, "octaves": 4, "seed": 42}, "pos": [0, 0]},
                    {"id": "remap", "op": "map_range", "inputs": {"value": "fbm_src"}, "params": {"from_min": 0.30, "from_max": 0.72, "to_min": 0.0, "to_max": 1.0, "interpolation": "smoothstep"}, "pos": [270, 0]},
                ],
                "output_node": "remap",
            },
        ],
    },
    {
        "schema": "synccraft.syncpack/1.0",
        "id": "pack_painter_layers",
        "name": "3D PBR Layer Painting Kit",
        "author": "SyncCraft Core",
        "version": "1.0.0",
        "builtin": True,
        "color": "#059669",
        "tags": ["paint", "3d", "pbr", "layers"],
        "description": "Ready-to-wire multi-channel 3D PBR Paint Layer Stack + 3D Curvature Wear Mask starter.",
        "macros": [
            {
                "id": "macro_pbr_paint_stack",
                "name": "Multi-Channel 3D PBR Paint Stack",
                "description": "Drop into any graph and paint Albedo/Roughness/Metallic/Height/Emissive directly in 3D!",
                "frame_color": "#059669",
                "nodes": [
                    {"id": "pbr_paint", "op": "paint_layer_stack", "inputs": {}, "params": {"layers_json": json.dumps([make_empty_paint_layer("layer_1", "Hero Detail Paint")]), "active_layer_id": "layer_1", "height_normal_strength": 2.6}, "pos": [0, 0]},
                ],
                "output_node": "pbr_paint",
            },
        ],
    },
]


def list_all_syncpacks() -> list[dict[str, Any]]:
    packs: list[dict[str, Any]] = [copy.deepcopy(p) for p in BUILTIN_SYNCPACKS]
    pdir = _syncpack_dir()
    for fp in sorted(pdir.glob("*.syncpack.json")):
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and isinstance(doc.get("macros"), list):
                doc["builtin"] = False
                doc["disk_path"] = str(fp)
                packs.append(doc)
        except Exception:
            continue
    return packs


def save_user_syncpack(pack: dict[str, Any]) -> Path:
    pid = "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(pack.get("id") or pack.get("name") or "custom_pack").lower()).strip("_") or "custom_pack"
    out = dict(pack)
    out["schema"] = "synccraft.syncpack/1.0"
    out["id"] = pid
    out["builtin"] = False
    path = _syncpack_dir() / f"{pid}.syncpack.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def spawn_syncpack_macro_into_graph(
    graph: Any,
    pack_id: str,
    macro_id: str,
    *,
    pos: tuple[float, float] = (240.0, 140.0),
    wrap_frame: bool = True,
) -> tuple[list[str], str, dict[str, Any] | None]:
    all_packs = {p["id"]: p for p in list_all_syncpacks()}
    if pack_id not in all_packs:
        raise ValueError(f"Unknown .syncpack package: {pack_id!r}")
    pack = all_packs[pack_id]
    macro = next((m for m in (pack.get("macros") or []) if m.get("id") == macro_id), None)
    if macro is None:
        raise ValueError(f"Unknown macro {macro_id!r} in pack {pack_id!r}")
    preset_like = {
        "name": macro.get("name") or macro_id,
        "frame_color": macro.get("frame_color") or pack.get("color") or "#2563EB",
        "nodes": copy.deepcopy(macro.get("nodes") or []),
        "output_node": macro.get("output_node"),
    }
    from .syncmat_and_mtlx import insert_syncmat_into_graph

    return insert_syncmat_into_graph(graph, preset_like, pos=pos, wrap_frame=wrap_frame)


# ===========================================================================
# 4. Multi-Asset Batch Bakery (`synccraft batch-bake` + Studio Queue)
# ===========================================================================


def scan_batch_folder(input_dir: str | Path, *, max_assets: int = 32) -> list[dict[str, Any]]:
    """Discover `.bbmodel`, `.texdsl`, and `.syncproj` assets in a folder with sibling auto-pairing."""
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Batch input directory does not exist: {root}")
    found: list[dict[str, Any]] = []
    seen_stems: set[str] = set()

    # Prioritize .bbmodel files (pairing with sibling .texdsl if present!)
    bb_files = sorted(root.rglob("*.bbmodel")) if root.is_dir() else ([root] if root.suffix.lower() == ".bbmodel" else [])
    for bb_p in bb_files:
        if "_baked" in bb_p.stem.lower():
            continue
        key = str(bb_p.with_suffix(""))
        seen_stems.add(key)
        sib_dsl = bb_p.with_suffix(".texdsl")
        found.append(
            {
                "id": f"asset_{len(found) + 1}",
                "stem": bb_p.stem,
                "kind": "bbmodel+texdsl" if sib_dsl.is_file() else "bbmodel",
                "bbmodel_path": str(bb_p),
                "texdsl_path": str(sib_dsl) if sib_dsl.is_file() else None,
            }
        )
        if len(found) >= max_assets:
            return found

    # Include standalone .texdsl files that had no sibling .bbmodel
    if root.is_dir():
        for dsl_p in sorted(root.rglob("*.texdsl")):
            key = str(dsl_p.with_suffix(""))
            if key in seen_stems:
                continue
            found.append(
                {
                    "id": f"asset_{len(found) + 1}",
                    "stem": dsl_p.stem,
                    "kind": "texdsl",
                    "bbmodel_path": None,
                    "texdsl_path": str(dsl_p),
                }
            )
            if len(found) >= max_assets:
                break
    return found


def run_batch_bakery(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    preset: str = "studio_hero",
    template_texdsl: str | Path | None = None,
    max_assets: int = 24,
    embed_manifest: bool = True,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Batch-bake an entire directory of `.bbmodel` / `.texdsl` assets into PBR v2.0 + MaterialX bundles."""
    from .core import TexGraph
    from .mesh_baker import bake_bbmodel_maps
    from .quality import evaluate_texture_quality
    from .syncmat_and_mtlx import export_materialx_xml
    from .texdsl import export_texdsl, parse_texdsl

    t_all = time.perf_counter()
    assets = scan_batch_folder(input_dir, max_assets=max_assets)
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    template_src: str | None = None
    if template_texdsl:
        tp = Path(template_texdsl).expanduser().resolve()
        if not tp.is_file():
            raise FileNotFoundError(f"Template .texdsl not found: {tp}")
        template_src = tp.read_text(encoding="utf-8")

    items: list[dict[str, Any]] = [
        {
            "id": a["id"],
            "stem": a["stem"],
            "kind": a["kind"],
            "status": "queued",
            "qa_score": None,
            "qa_passed": None,
            "ms": 0.0,
            "mesh_cache_hit": False,
            "files_count": 0,
            "error": None,
        }
        for a in assets
    ]

    def _emit(running: bool = True) -> None:
        if on_progress:
            done_n = sum(1 for x in items if x["status"] in ("done", "error"))
            ok_n = sum(1 for x in items if x["status"] == "done")
            err_n = sum(1 for x in items if x["status"] == "error")
            on_progress(
                {
                    "running": running,
                    "total": len(items),
                    "completed": done_n,
                    "succeeded": ok_n,
                    "failed": err_n,
                    "out_dir": str(out_root),
                    "items": copy.deepcopy(items),
                }
            )

    _emit(True)

    for idx, asset in enumerate(assets):
        items[idx]["status"] = "baking"
        _emit(True)
        t0 = time.perf_counter()
        try:
            bb_doc = None
            mc = None
            if asset["bbmodel_path"]:
                bb_doc = json.loads(Path(asset["bbmodel_path"]).read_text(encoding="utf-8"))
                mc = bake_bbmodel_maps(bb_doc)
                items[idx]["mesh_cache_hit"] = bool(mc.bake_cache_hit)

            if asset["texdsl_path"]:
                g = parse_texdsl(Path(asset["texdsl_path"]).read_text(encoding="utf-8"), mesh_context=mc)
            elif template_src is not None:
                g = parse_texdsl(template_src, mesh_context=mc)
                g.name = asset["stem"]
            elif mc is not None:
                from .studio_server import StudioSession
                g = StudioSession._build_bbmodel_starter_graph(bb_doc or {}, mc)
                g.name = asset["stem"]
            else:
                raise ValueError("Asset has neither .texdsl nor .bbmodel")

            if mc is not None:
                g.attach_mesh(mc)
                g.domain = "entity"

            ev = g.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
            qa_rep = evaluate_texture_quality(
                ev.output,
                domain="entity" if mc is not None else "block",
                tileable=g.tileable,
                mesh_ctx=mc,
                pbr_channels=ev.pbr_channels,
                pbr_maps=ev.pbr_maps,
            )
            asset_out = out_root / asset["stem"]
            pkg = g.export_baked_pbr(
                asset_out,
                stem=asset["stem"],
                preset=preset,
                bbmodel_doc=bb_doc,
                embed_maps_in_manifest=embed_manifest,
            )
            (asset_out / f"{asset['stem']}.texdsl").write_text(export_texdsl(g), encoding="utf-8")
            (asset_out / f"{asset['stem']}.mtlx").write_text(
                export_materialx_xml(asset["stem"], shader_model="open_pbr_surface"),
                encoding="utf-8",
            )
            items[idx]["status"] = "done"
            items[idx]["qa_score"] = round(float(qa_rep.score), 1)
            items[idx]["qa_passed"] = bool(qa_rep.passed)
            items[idx]["files_count"] = len(pkg["files"]) + 2
            items[idx]["out_dir"] = str(asset_out)
        except Exception as exc:
            items[idx]["status"] = "error"
            items[idx]["error"] = f"{type(exc).__name__}: {exc}"
        items[idx]["ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        _emit(True)

    total_ms = round((time.perf_counter() - t_all) * 1000.0, 2)
    succeeded = sum(1 for x in items if x["status"] == "done")
    failed = sum(1 for x in items if x["status"] == "error")
    qa_scores = [x["qa_score"] for x in items if isinstance(x["qa_score"], (int, float))]
    report = {
        "schema": "synccraft.batch_bakery/1.0",
        "ok": failed == 0,
        "input_dir": str(Path(input_dir).expanduser().resolve()),
        "out_dir": str(out_root),
        "preset": preset,
        "total": len(items),
        "succeeded": succeeded,
        "failed": failed,
        "mean_qa": round(sum(qa_scores) / len(qa_scores), 2) if qa_scores else None,
        "total_ms": total_ms,
        "items": items,
    }
    report_path = out_root / "batch_bakery_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    _emit(False)
    return report
