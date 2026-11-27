"""Blender-Style Basic Math, Vector, Mapping, Color Utility & Direct Brush Paint-Mask Ops (`TexGraph v15`).

Provides fundamental shader-graph building blocks familiar from Blender Shader Nodes,
Unreal Material Editor, Unity Shader Graph, and Substance Designer:
- `math_scalar`: 21 mathematical & comparison ops on scalar/mask/image fields
- `map_range`: Linear / Stepped / Smoothstep / Smootherstep value remapping
- `clamp_value`: Min-Max / Range scalar & color clamping
- `mix_blend`: Unified Blend/Mix node (Mix, Add, Multiply, Screen, Overlay, Darken, Lighten, Difference, Subtract, Divide)
- `rgb_combine`: Pack 3-4 scalar channels into RGBA (`RGB`, `HSV`, `HSL`)
- `rgb_separate`: Split RGBA into `r/g/b/a` (or `h/s/v`) selectable scalar output
- `vector_math`: 3-channel normal/vector math (`add`, `subtract`, `multiply`, `normalize`, `dot`, `cross`, `length`, `scale`, `reflect`, `invert`, `abs`)
- `uv_mapping`: Blender Mapping transform (`location`, `rotation_deg`, `scale`, `tile`)
- `tex_coord`: Blender Texture Coordinate generator (`uv_x`, `uv_y`, `centered`, `radial_dist`, `radial_angle`, `world_pos`, `world_normal`)
- `float_constant`: Clean constant scalar value generator
- `color_constant`: Clean constant RGBA swatch generator
- `invert_field`: Blender Invert Color / Mask node
- `fresnel_weight`: Blender Layer Weight / Fresnel rim falloff (3D mesh-aware or height-normal)
- `checker_basic`: Classic Blender Checker procedural texture
- `paint_mask`: Interactive 3D/2D Direct Brush Painted Grayscale Mask stored in the DAG!
"""
from __future__ import annotations

import base64
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from .generators_noise import sample_field_bilinear, uv_grid
from .mesh_baker import MeshBakeContext
from .pixel_art_ops import hex_to_srgb


def _to_f32_2d(arr: np.ndarray | float | None, h: int, w: int, default: float = 0.0) -> np.ndarray:
    if arr is None:
        return np.full((h, w), float(default), dtype=np.float32)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 0:
        return np.full((h, w), float(a), dtype=np.float32)
    if a.ndim == 3:
        if a.shape[2] >= 3:
            a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        else:
            a = a[..., 0]
    if a.shape != (h, w):
        img = Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), mode="L").resize((w, h), Image.Resampling.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
    return a.astype(np.float32)


def _to_f32_rgba(arr: np.ndarray | None, h: int, w: int, fallback_hex: str = "#808080") -> np.ndarray:
    if arr is None:
        r, g, b = hex_to_srgb(fallback_hex)
        out = np.empty((h, w, 4), dtype=np.float32)
        out[..., 0] = r
        out[..., 1] = g
        out[..., 2] = b
        out[..., 3] = 1.0
        return out
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack((a, a, a, np.ones_like(a)), axis=-1)
    elif a.ndim == 3 and a.shape[2] == 3:
        a = np.concatenate((a, np.ones((a.shape[0], a.shape[1], 1), dtype=np.float32)), axis=-1)
    if a.shape[:2] != (h, w):
        img = Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), mode="RGBA").resize((w, h), Image.Resampling.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
    return a.astype(np.float32)


def _to_f32_vec3(arr: np.ndarray | None, h: int, w: int, default: tuple[float, float, float] = (0.5, 0.5, 1.0)) -> np.ndarray:
    if arr is None:
        out = np.empty((h, w, 3), dtype=np.float32)
        out[..., 0], out[..., 1], out[..., 2] = default
        return out
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        a = np.stack((a, a, a), axis=-1)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[..., :3]
    if a.shape[:2] != (h, w):
        img = Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), mode="RGB").resize((w, h), Image.Resampling.BILINEAR)
        a = np.asarray(img, dtype=np.float32) / 255.0
    return a.astype(np.float32)


# ===========================================================================
# 1. Blender Math & Range Nodes
# ===========================================================================


def math_scalar(
    height: int,
    width: int,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    *,
    math_op: str = "multiply",
    value_a: float = 0.5,
    value_b: float = 0.5,
    clamp_out: bool = True,
) -> np.ndarray:
    """Blender-style Scalar Math node supporting 21 arithmetic, trig & comparison operations."""
    va = _to_f32_2d(a, height, width, default=value_a)
    vb = _to_f32_2d(b, height, width, default=value_b)
    op = str(math_op or "multiply").strip().lower()

    if op == "add":
        out = va + vb
    elif op == "subtract":
        out = va - vb
    elif op == "multiply":
        out = va * vb
    elif op == "divide":
        out = va / np.where(np.abs(vb) > 1e-6, vb, 1e-6)
    elif op == "power":
        out = np.power(np.maximum(va, 0.0), vb)
    elif op == "sqrt":
        out = np.sqrt(np.maximum(va, 0.0))
    elif op == "abs":
        out = np.abs(va)
    elif op == "min":
        out = np.minimum(va, vb)
    elif op == "max":
        out = np.maximum(va, vb)
    elif op == "smooth_min":
        k = np.maximum(np.abs(vb), 1e-3)
        h = np.maximum(k - np.abs(va - vb), 0.0) / k
        out = np.minimum(va, vb) - h * h * k * 0.25
    elif op == "modulo":
        denom = np.where(np.abs(vb) > 1e-5, vb, 1.0)
        out = np.mod(va, denom)
    elif op == "pingpong":
        scale = np.maximum(np.abs(vb), 1e-4)
        out = scale - np.abs(np.mod(va, 2.0 * scale) - scale)
    elif op == "sin":
        out = 0.5 + 0.5 * np.sin(va * (2.0 * math.pi * np.maximum(vb, 0.01)))
    elif op == "cos":
        out = 0.5 + 0.5 * np.cos(va * (2.0 * math.pi * np.maximum(vb, 0.01)))
    elif op == "greater_than":
        out = (va > vb).astype(np.float32)
    elif op == "less_than":
        out = (va < vb).astype(np.float32)
    elif op == "round":
        steps = np.maximum(np.round(vb * 16.0), 1.0)
        out = np.round(va * steps) / steps
    elif op == "floor":
        out = np.floor(va * np.maximum(vb * 8.0, 1.0)) / np.maximum(vb * 8.0, 1.0)
    elif op == "ceil":
        out = np.ceil(va * np.maximum(vb * 8.0, 1.0)) / np.maximum(vb * 8.0, 1.0)
    elif op == "fract":
        out = va - np.floor(va)
    elif op == "sign":
        out = np.sign(va - vb) * 0.5 + 0.5
    else:
        out = va * vb

    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if clamp_out:
        out = np.clip(out, 0.0, 1.0)
    return out


def map_range(
    height: int,
    width: int,
    value: np.ndarray | None = None,
    *,
    from_min: float = 0.0,
    from_max: float = 1.0,
    to_min: float = 0.0,
    to_max: float = 1.0,
    interpolation: str = "linear",
    steps: int = 4,
    clamp_out: bool = True,
) -> np.ndarray:
    """Blender Map Range node (`linear`, `stepped`, `smoothstep`, `smootherstep`)."""
    v = _to_f32_2d(value, height, width, default=0.5)
    denom = float(from_max) - float(from_min)
    if abs(denom) < 1e-6:
        denom = 1e-6
    t = (v - float(from_min)) / denom
    if clamp_out:
        t = np.clip(t, 0.0, 1.0)

    interp = str(interpolation or "linear").strip().lower()
    if interp == "stepped":
        st = max(1, int(steps))
        t = np.floor(t * (st + 1)) / float(st)
        t = np.clip(t, 0.0, 1.0)
    elif interp == "smoothstep":
        tc = np.clip(t, 0.0, 1.0)
        t = tc * tc * (3.0 - 2.0 * tc)
    elif interp == "smootherstep":
        tc = np.clip(t, 0.0, 1.0)
        t = tc * tc * tc * (tc * (tc * 6.0 - 15.0) + 10.0)

    out = float(to_min) + t * (float(to_max) - float(to_min))
    if clamp_out:
        lo, hi = min(float(to_min), float(to_max)), max(float(to_min), float(to_max))
        out = np.clip(out, lo, hi)
    return out.astype(np.float32)


def clamp_value(
    height: int,
    width: int,
    value: np.ndarray | None = None,
    *,
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """Blender Clamp node constraining any scalar or RGBA field between `[min_val, max_val]`."""
    lo, hi = min(float(min_val), float(max_val)), max(float(min_val), float(max_val))
    if value is None:
        return np.full((height, width), np.clip(0.5, lo, hi), dtype=np.float32)
    a = np.asarray(value, dtype=np.float32)
    if a.ndim == 3:
        out = a.copy()
        out[..., :3] = np.clip(out[..., :3], lo, hi)
        return out
    return np.clip(_to_f32_2d(a, height, width), lo, hi).astype(np.float32)


def mix_blend(
    height: int,
    width: int,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    factor: np.ndarray | None = None,
    *,
    blend_mode: str = "mix",
    fac_value: float = 0.5,
    color_a: str = "#3B82F6",
    color_b: str = "#F59E0B",
    clamp_out: bool = True,
) -> np.ndarray:
    """Blender Mix Color / Blend node with optional Factor mask and 10 blend modes."""
    ca = _to_f32_rgba(a, height, width, fallback_hex=color_a)
    cb = _to_f32_rgba(b, height, width, fallback_hex=color_b)
    f = np.clip(_to_f32_2d(factor, height, width, default=fac_value), 0.0, 1.0)[..., None]

    ra, rb = ca[..., :3], cb[..., :3]
    md = str(blend_mode or "mix").strip().lower()
    if md == "add":
        blended = ra + rb
    elif md == "subtract":
        blended = ra - rb
    elif md == "multiply":
        blended = ra * rb
    elif md == "screen":
        blended = 1.0 - (1.0 - ra) * (1.0 - rb)
    elif md == "overlay":
        blended = np.where(ra < 0.5, 2.0 * ra * rb, 1.0 - 2.0 * (1.0 - ra) * (1.0 - rb))
    elif md == "darken":
        blended = np.minimum(ra, rb)
    elif md == "lighten":
        blended = np.maximum(ra, rb)
    elif md == "difference":
        blended = np.abs(ra - rb)
    elif md == "divide":
        blended = ra / np.maximum(rb, 1e-4)
    else:
        blended = rb

    rgb = ra * (1.0 - f) + blended * f
    if clamp_out:
        rgb = np.clip(rgb, 0.0, 1.0)
    alpha = np.clip(ca[..., 3:4] * (1.0 - f) + cb[..., 3:4] * f, 0.0, 1.0)
    return np.concatenate((rgb, alpha), axis=-1).astype(np.float32)


def rgb_combine(
    height: int,
    width: int,
    r: np.ndarray | None = None,
    g: np.ndarray | None = None,
    b: np.ndarray | None = None,
    a: np.ndarray | None = None,
    *,
    mode: str = "rgb",
    r_val: float = 0.5,
    g_val: float = 0.5,
    b_val: float = 0.5,
    a_val: float = 1.0,
) -> np.ndarray:
    """Blender Combine Color node (`rgb` or `hsv`) packing scalar inputs into RGBA."""
    c0 = np.clip(_to_f32_2d(r, height, width, default=r_val), 0.0, 1.0)
    c1 = np.clip(_to_f32_2d(g, height, width, default=g_val), 0.0, 1.0)
    c2 = np.clip(_to_f32_2d(b, height, width, default=b_val), 0.0, 1.0)
    ca = np.clip(_to_f32_2d(a, height, width, default=a_val), 0.0, 1.0)

    if str(mode or "rgb").strip().lower() == "hsv":
        h6 = (c0 % 1.0) * 6.0
        s = c1
        v = c2
        i = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        rr = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v], default=v)
        gg = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p], default=v)
        bb = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q], default=v)
        return np.stack((rr, gg, bb, ca), axis=-1).astype(np.float32)

    return np.stack((c0, c1, c2, ca), axis=-1).astype(np.float32)


def rgb_separate(
    height: int,
    width: int,
    color: np.ndarray | None = None,
    *,
    channel: str = "r",
) -> np.ndarray:
    """Blender Separate Color node extracting `r`, `g`, `b`, `a`, `luma`, `hue`, `sat`, or `val`."""
    rgba = _to_f32_rgba(color, height, width)
    ch = str(channel or "r").strip().lower()
    r, g, b, a = rgba[..., 0], rgba[..., 1], rgba[..., 2], rgba[..., 3]
    if ch == "g":
        return g.astype(np.float32)
    if ch == "b":
        return b.astype(np.float32)
    if ch == "a":
        return a.astype(np.float32)
    if ch == "luma":
        return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    if ch in ("hue", "sat", "val"):
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        df = mx - mn
        if ch == "val":
            return mx.astype(np.float32)
        if ch == "sat":
            return np.where(mx > 1e-6, df / np.maximum(mx, 1e-6), 0.0).astype(np.float32)
        hue = np.zeros_like(mx)
        nz = df > 1e-6
        m_r = nz & (mx == r)
        m_g = nz & (mx == g) & ~m_r
        m_b = nz & ~m_r & ~m_g
        hue[m_r] = ((g[m_r] - b[m_r]) / df[m_r]) % 6.0
        hue[m_g] = ((b[m_g] - r[m_g]) / df[m_g]) + 2.0
        hue[m_b] = ((r[m_b] - g[m_b]) / df[m_b]) + 4.0
        return (hue / 6.0).astype(np.float32)
    return r.astype(np.float32)


def vector_math(
    height: int,
    width: int,
    vec_a: np.ndarray | None = None,
    vec_b: np.ndarray | None = None,
    *,
    vec_op: str = "normalize",
    scale: float = 1.0,
) -> np.ndarray:
    """Blender Vector Math node for 3-channel normal/vector/RGB fields."""
    va = _to_f32_vec3(vec_a, height, width, default=(0.5, 0.5, 1.0)) * 2.0 - 1.0
    vb = _to_f32_vec3(vec_b, height, width, default=(0.5, 0.5, 1.0)) * 2.0 - 1.0
    op = str(vec_op or "normalize").strip().lower()

    if op == "add":
        res = va + vb
    elif op == "subtract":
        res = va - vb
    elif op == "multiply":
        res = va * vb
    elif op == "scale":
        res = va * float(scale)
    elif op == "invert":
        res = -va
    elif op == "abs":
        res = np.abs(va)
    elif op == "cross":
        res = np.cross(va, vb)
    elif op == "reflect":
        nb = vb / np.maximum(np.linalg.norm(vb, axis=-1, keepdims=True), 1e-5)
        dot = np.sum(va * nb, axis=-1, keepdims=True)
        res = va - 2.0 * dot * nb
    elif op == "dot":
        na = va / np.maximum(np.linalg.norm(va, axis=-1, keepdims=True), 1e-5)
        nb = vb / np.maximum(np.linalg.norm(vb, axis=-1, keepdims=True), 1e-5)
        d = np.clip(np.sum(na * nb, axis=-1) * 0.5 + 0.5, 0.0, 1.0)
        return np.stack((d, d, d), axis=-1).astype(np.float32)
    elif op == "length":
        ln = np.clip(np.linalg.norm(va, axis=-1) / math.sqrt(3.0), 0.0, 1.0)
        return np.stack((ln, ln, ln), axis=-1).astype(np.float32)
    else:  # normalize
        res = va / np.maximum(np.linalg.norm(va, axis=-1, keepdims=True), 1e-5)

    return np.clip(res * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)


# ===========================================================================
# 2. Blender Mapping, Texture Coordinates, Constants, Fresnel & Checker
# ===========================================================================


def uv_mapping(
    height: int,
    width: int,
    src: np.ndarray | None = None,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    rotation_deg: float = 0.0,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    tileable: bool = True,
) -> np.ndarray:
    """Blender Mapping Node: translate, rotate, and tile-scale any scalar or RGBA texture."""
    u, v = uv_grid(height, width)
    du = (u - 0.5) * float(scale_x)
    dv = (v - 0.5) * float(scale_y)
    rad = math.radians(float(rotation_deg))
    c, s = math.cos(rad), math.sin(rad)
    su = du * c - dv * s + 0.5 + float(offset_x)
    sv = du * s + dv * c + 0.5 + float(offset_y)
    if tileable:
        su = su % 1.0
        sv = sv % 1.0
    else:
        su = np.clip(su, 0.0, 1.0)
        sv = np.clip(sv, 0.0, 1.0)

    if src is None:
        return np.stack((su, sv, np.zeros_like(su), np.ones_like(su)), axis=-1).astype(np.float32)

    a = np.asarray(src, dtype=np.float32)
    if a.ndim == 2:
        return sample_field_bilinear(a, su, sv, tileable=tileable).astype(np.float32)
    out = np.empty((height, width, a.shape[2]), dtype=np.float32)
    for k in range(a.shape[2]):
        out[..., k] = sample_field_bilinear(a[..., k], su, sv, tileable=tileable)
    return out


def tex_coord(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    coord_type: str = "generated_uv",
) -> np.ndarray:
    """Blender Texture Coordinate node (`generated_uv`, `uv_x`, `uv_y`, `radial_dist`, `radial_angle`, `world_pos`, `world_normal`)."""
    u, v = uv_grid(height, width)
    ct = str(coord_type or "generated_uv").strip().lower()
    if ct == "uv_x":
        return u.astype(np.float32)
    if ct == "uv_y":
        return v.astype(np.float32)
    if ct == "radial_dist":
        return np.clip(np.hypot(u - 0.5, v - 0.5) * 2.0, 0.0, 1.0).astype(np.float32)
    if ct == "radial_angle":
        return ((np.arctan2(v - 0.5, u - 0.5) / (2.0 * math.pi)) % 1.0).astype(np.float32)
    if ct == "world_pos" and mesh_ctx is not None and mesh_ctx.world_pos.shape[:2] == (height, width):
        wp = mesh_ctx.world_pos
        lo = np.min(wp.reshape(-1, 3), axis=0)
        hi = np.max(wp.reshape(-1, 3), axis=0)
        norm = np.clip((wp - lo) / np.maximum(hi - lo, 1e-4), 0.0, 1.0)
        return np.concatenate((norm, np.ones((height, width, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    if ct == "world_normal" and mesh_ctx is not None and mesh_ctx.world_normal.shape[:2] == (height, width):
        wn = np.clip(mesh_ctx.world_normal * 0.5 + 0.5, 0.0, 1.0)
        return np.concatenate((wn, np.ones((height, width, 1), dtype=np.float32)), axis=-1).astype(np.float32)
    return np.stack((u, v, np.zeros_like(u), np.ones_like(u)), axis=-1).astype(np.float32)


def float_constant(height: int, width: int, *, value: float = 0.5) -> np.ndarray:
    """Blender Value Constant node outputting a uniform scalar field."""
    return np.full((height, width), float(value), dtype=np.float32)


def color_constant(height: int, width: int, *, color: str = "#3B82F6", alpha: float = 1.0) -> np.ndarray:
    """Blender RGB Color Constant node outputting a uniform RGBA swatch."""
    r, g, b = hex_to_srgb(color)
    out = np.empty((height, width, 4), dtype=np.float32)
    out[..., 0] = r
    out[..., 1] = g
    out[..., 2] = b
    out[..., 3] = float(np.clip(alpha, 0.0, 1.0))
    return out


def invert_field(height: int, width: int, src: np.ndarray | None = None, *, fac: float = 1.0) -> np.ndarray:
    """Blender Invert Color / Mask node (`out = lerp(src, 1 - src, fac)`)."""
    f = float(np.clip(fac, 0.0, 1.0))
    if src is None:
        return np.full((height, width), 1.0 - f * 0.5, dtype=np.float32)
    a = np.asarray(src, dtype=np.float32)
    if a.ndim == 3:
        out = a.copy()
        out[..., :3] = np.clip(a[..., :3] * (1.0 - f) + (1.0 - a[..., :3]) * f, 0.0, 1.0)
        return out
    return np.clip(a * (1.0 - f) + (1.0 - a) * f, 0.0, 1.0).astype(np.float32)


def fresnel_weight(
    height: int,
    width: int,
    normal: np.ndarray | None = None,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    ior: float = 1.45,
    power: float = 3.0,
) -> np.ndarray:
    """Blender Layer Weight / Fresnel rim falloff computed from tangent/world normal or radial dome."""
    if normal is not None:
        n = _to_f32_vec3(normal, height, width) * 2.0 - 1.0
        n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-5)
        ndotv = np.clip(np.abs(n[..., 2]), 0.0, 1.0)
    elif mesh_ctx is not None and mesh_ctx.world_normal.shape[:2] == (height, width):
        wn = mesh_ctx.world_normal
        view = np.array([0.0, 0.25, 0.968], dtype=np.float32)
        ndotv = np.clip(np.abs(np.sum(wn * view[None, None, :], axis=-1)), 0.0, 1.0)
    else:
        u, v = uv_grid(height, width)
        r = np.clip(np.hypot(u - 0.5, v - 0.5) * 2.0, 0.0, 1.0)
        ndotv = np.sqrt(np.maximum(1.0 - r * r, 0.0))

    f0 = ((float(ior) - 1.0) / max(float(ior) + 1.0, 1.001)) ** 2
    fr = f0 + (1.0 - f0) * np.power(1.0 - ndotv, max(0.5, float(power)))
    return np.clip(fr, 0.0, 1.0).astype(np.float32)


def checker_basic(
    height: int,
    width: int,
    *,
    scale_x: int = 8,
    scale_y: int = 8,
    color_a: str = "#E2E8F0",
    color_b: str = "#1E293B",
) -> np.ndarray:
    """Classic Blender Checker Texture generator."""
    u, v = uv_grid(height, width)
    cx = np.floor(u * max(1, int(scale_x))).astype(np.int32)
    cy = np.floor(v * max(1, int(scale_y))).astype(np.int32)
    m = ((cx + cy) & 1).astype(np.float32)[..., None]
    ra, ga, ba = hex_to_srgb(color_a)
    rb, gb, bb = hex_to_srgb(color_b)
    ca = np.array([ra, ga, ba, 1.0], dtype=np.float32)[None, None, :]
    cb = np.array([rb, gb, bb, 1.0], dtype=np.float32)[None, None, :]
    return (ca * (1.0 - m) + cb * m).astype(np.float32)


# ===========================================================================
# 3. Direct 3D / 2D Brush Mask Painting Node (`paint_mask`)
# ===========================================================================


def paint_mask(
    height: int,
    width: int,
    *,
    mask_data_url: str = "",
    base_value: float = 0.0,
    blur_px: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """Direct 3D/2D Brush-Painted Grayscale Mask stored as lossless PNG data URL right inside the node graph."""
    if mask_data_url and isinstance(mask_data_url, str) and "," in mask_data_url:
        try:
            b64 = mask_data_url.split(",", 1)[1]
            raw = base64.b64decode(b64)
            im = Image.open(io.BytesIO(raw)).convert("L")
            if im.size != (width, height):
                im = im.resize((width, height), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        except Exception:
            arr = np.full((height, width), float(np.clip(base_value, 0.0, 1.0)), dtype=np.float32)
    else:
        arr = np.full((height, width), float(np.clip(base_value, 0.0, 1.0)), dtype=np.float32)

    if int(blur_px) > 0:
        from .mesh_mask_ops import _box_blur_2d

        arr = _box_blur_2d(arr, int(blur_px), tileable=False)
    if invert:
        arr = 1.0 - arr
    return np.clip(arr, 0.0, 1.0).astype(np.float32)
