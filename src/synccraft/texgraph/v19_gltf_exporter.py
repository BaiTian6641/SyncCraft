"""SyncCraft TexGraph v19 (`SyncCraft Studio v55.0`):
Pure-Python Zero-Dependency Khronos glTF 2.0 Binary (`.glb`) Skinned Mesh,
Skeletal Animation & Embedded PBR Exporter + Strict Khronos Validator.

Features:
- 100% pure stdlib (`struct`, `json`, `math`, `io`) + NumPy + Pillow — zero third-party glTF libraries.
- Exact Blockbench 5 `.bbmodel` -> glTF 2.0 right-handed Y-up meter-scale (`1/16 = 0.0625 m/unit`) conversion.
- Indexed quad meshes (33% smaller vertex buffers), exact outward CCW winding, unit `NORMAL`,
  and mirrored-UV-safe `TANGENT` (`VEC4` with exact `+/-1` handedness for OpenGL `+Y` normal maps).
- Full hierarchical skeletal skinning (`Armature` common root, `skin.skeleton`, `JOINTS_0`, `WEIGHTS_0`,
  column-major `inverseBindMatrices` satisfying `M_global_bind @ IBM == I_4` identically, root-level
  identity skinned mesh node per glTF 2.0 §3.7.3.1).
- Zero-`eval` recursive-descent Molang evaluator & `linear`/`catmullrom`/`step` keyframe baker converting
  ALL Blockbench 5 skeletal clips into absolute glTF 2.0 TRS animations with shortest-arc quaternion
  hemisphere continuity (`dot < 0` flip) and strictly-increasing float32 timestamps (`>= 2` keyframes).
- Embedded PNG PBR textures inside GLB `BIN` chunk (`baseColorTexture`, shared linear ORM
  `R=AO, G=Roughness, B=Metallic` for both `occlusionTexture` & `metallicRoughnessTexture`, OpenGL `+Y`
  `normalTexture`, optional `emissiveTexture` + `KHR_materials_emissive_strength`, optional
  `KHR_materials_clearcoat`, pixel-art `NEAREST` / `NEAREST_MIPMAP_LINEAR` + `CLAMP_TO_EDGE` sampler,
  mip-safe Voronoi edge dilation, and multi-texture `.bbmodel` slot primitive splitting).
- Multi-tier `LOD0 / LOD1 / LOD2` `.glb` bundle export (`export_lod_glb_bundle`).
- Strict built-in `validate_glb_bytes()` checking all Khronos `glTF-Validator` binary, alignment,
  float32 min/max, quaternion, IBM, PNG, and multi-engine (UE5/Unity/Godot 4/Blender/Three.js) invariants.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .mesh_baker import _FACE_BASIS, _euler_xyz_mat
from .pbr import bake_pbr_to_albedo, height_to_normal_map, pack_labpbr_1_3, rgba_float_to_u8
from .v17_decal_atlas_ops import mip_safe_dilate_normal, mip_safe_dilate_rgba
from .v18_lod_baker_ops import (
    extract_bbmodel_texture_slots,
    generate_bbmodel_lod_chain,
    resolve_face_texture_slot,
)

# ---------------------------------------------------------------------------
# Khronos glTF 2.0 Spec Binary & Enum Constants
# ---------------------------------------------------------------------------

GLB_MAGIC = 0x46546C67  # b"glTF"
GLB_VERSION = 2
GLB_CHUNK_JSON = 0x4E4F534A  # b"JSON"
GLB_CHUNK_BIN = 0x004E4942  # b"BIN\x00"

COMP_BYTE = 5120
COMP_UBYTE = 5121
COMP_SHORT = 5122
COMP_USHORT = 5123
COMP_UINT = 5125
COMP_FLOAT = 5126

COMP_BYTE_SIZES = {
    COMP_BYTE: 1,
    COMP_UBYTE: 1,
    COMP_SHORT: 2,
    COMP_USHORT: 2,
    COMP_UINT: 4,
    COMP_FLOAT: 4,
}
TYPE_NUM_COMPS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963

FILTER_NEAREST = 9728
FILTER_LINEAR = 9729
FILTER_NEAREST_MIPMAP_NEAREST = 9984
FILTER_NEAREST_MIPMAP_LINEAR = 9986
WRAP_CLAMP_TO_EDGE = 33071
WRAP_REPEAT = 10497


# ---------------------------------------------------------------------------
# Zero-Eval Molang Evaluator & Blockbench 5 Keyframe Sampler (Python Parity)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def eval_molang_expr(val: Any, anim_time: float) -> float:
    """Safe zero-`eval` Molang math evaluator (degree-mode trig, exact parity with `studio_gl3d.js`)."""
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else 0.0
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    if _NUM_RE.match(s):
        try:
            f = float(s)
            return f if math.isfinite(f) else 0.0
        except Exception:
            return 0.0

    src = s.lower()
    n = len(src)
    pos = [0]

    def _skip_ws() -> None:
        while pos[0] < n and src[pos[0]] <= " ":
            pos[0] += 1

    def _parse_expr() -> float:
        return _parse_ternary()

    def _parse_ternary() -> float:
        c = _parse_compare()
        _skip_ws()
        if pos[0] < n and src[pos[0]] == "?":
            pos[0] += 1
            a = _parse_expr()
            _skip_ws()
            if pos[0] < n and src[pos[0]] == ":":
                pos[0] += 1
            b = _parse_expr()
            return a if c != 0.0 else b
        return c

    def _parse_compare() -> float:
        left = _parse_add_sub()
        while True:
            _skip_ws()
            two = src[pos[0] : pos[0] + 2]
            if two in ("<=", ">=", "==", "!="):
                pos[0] += 2
                right = _parse_add_sub()
                if two == "<=":
                    left = 1.0 if left <= right else 0.0
                elif two == ">=":
                    left = 1.0 if left >= right else 0.0
                elif two == "==":
                    left = 1.0 if abs(left - right) < 1e-6 else 0.0
                else:
                    left = 1.0 if abs(left - right) >= 1e-6 else 0.0
            elif pos[0] < n and src[pos[0]] in ("<", ">"):
                op = src[pos[0]]
                pos[0] += 1
                right = _parse_add_sub()
                left = 1.0 if ((left < right) if op == "<" else (left > right)) else 0.0
            else:
                break
        return left

    def _parse_add_sub() -> float:
        left = _parse_mul_div()
        while True:
            _skip_ws()
            if pos[0] < n and src[pos[0]] in ("+", "-"):
                op = src[pos[0]]
                pos[0] += 1
                right = _parse_mul_div()
                left = (left + right) if op == "+" else (left - right)
            else:
                break
        return left

    def _parse_mul_div() -> float:
        left = _parse_unary()
        while True:
            _skip_ws()
            if pos[0] < n and src[pos[0]] in ("*", "/"):
                op = src[pos[0]]
                pos[0] += 1
                right = _parse_unary()
                if op == "*":
                    left = left * right
                else:
                    left = (left / right) if abs(right) > 1e-8 else 0.0
            else:
                break
        return left

    def _parse_unary() -> float:
        _skip_ws()
        if pos[0] < n:
            ch = src[pos[0]]
            if ch == "+":
                pos[0] += 1
                return _parse_unary()
            if ch == "-":
                pos[0] += 1
                return -_parse_unary()
            if ch == "!":
                pos[0] += 1
                return 0.0 if _parse_unary() != 0.0 else 1.0
        return _parse_primary()

    def _parse_primary() -> float:
        _skip_ws()
        if pos[0] >= n:
            return 0.0
        ch = src[pos[0]]
        if ch == "(":
            pos[0] += 1
            v = _parse_expr()
            _skip_ws()
            if pos[0] < n and src[pos[0]] == ")":
                pos[0] += 1
            return v
        start = pos[0]
        if ("0" <= ch <= "9") or ch == ".":
            while pos[0] < n and (("0" <= src[pos[0]] <= "9") or src[pos[0]] == "."):
                pos[0] += 1
            if pos[0] < n and src[pos[0]] in ("e", "E"):
                pos[0] += 1
                if pos[0] < n and src[pos[0]] in ("+", "-"):
                    pos[0] += 1
                while pos[0] < n and "0" <= src[pos[0]] <= "9":
                    pos[0] += 1
            try:
                return float(src[start : pos[0]])
            except Exception:
                return 0.0
        while pos[0] < n and (src[pos[0]].isalnum() or src[pos[0]] in ("_", ".")):
            pos[0] += 1
        ident = src[start : pos[0]]
        _skip_ws()
        if pos[0] < n and src[pos[0]] == "(":
            pos[0] += 1
            args: list[float] = []
            _skip_ws()
            if pos[0] < n and src[pos[0]] != ")":
                while pos[0] < n:
                    args.append(_parse_expr())
                    _skip_ws()
                    if pos[0] < n and src[pos[0]] == ",":
                        pos[0] += 1
                        continue
                    break
            if pos[0] < n and src[pos[0]] == ")":
                pos[0] += 1
            a0 = args[0] if len(args) > 0 else 0.0
            a1 = args[1] if len(args) > 1 else 0.0
            a2 = args[2] if len(args) > 2 else 0.0
            if ident in ("math.sin", "sin"):
                return math.sin(math.radians(a0))
            if ident in ("math.cos", "cos"):
                return math.cos(math.radians(a0))
            if ident in ("math.tan", "tan"):
                return math.tan(math.radians(a0))
            if ident == "math.asin":
                return math.degrees(math.asin(max(-1.0, min(1.0, a0))))
            if ident == "math.acos":
                return math.degrees(math.acos(max(-1.0, min(1.0, a0))))
            if ident == "math.atan":
                return math.degrees(math.atan(a0))
            if ident == "math.atan2":
                return math.degrees(math.atan2(a0, a1))
            if ident in ("math.abs", "abs"):
                return abs(a0)
            if ident == "math.floor":
                return float(math.floor(a0))
            if ident == "math.ceil":
                return float(math.ceil(a0))
            if ident == "math.round":
                return float(round(a0))
            if ident == "math.sqrt":
                return math.sqrt(max(0.0, a0))
            if ident == "math.pow":
                try:
                    return math.pow(a0, a1)
                except Exception:
                    return 0.0
            if ident == "math.mod":
                return math.fmod(a0, a1) if abs(a1) > 1e-8 else 0.0
            if ident == "math.min":
                return min(a0, a1)
            if ident == "math.max":
                return max(a0, a1)
            if ident == "math.clamp":
                return max(a1, min(a2, a0))
            if ident == "math.lerp":
                return a0 + (a1 - a0) * a2
            if ident == "math.hermite_blend":
                u = max(0.0, min(1.0, a0))
                return u * u * (3.0 - 2.0 * u)
            return 0.0
        if ident in ("query.anim_time", "q.anim_time", "query.life_time", "q.life_time", "time"):
            return float(anim_time)
        if ident in ("math.pi", "pi"):
            return math.pi
        return 0.0

    try:
        res = _parse_expr()
        return res if math.isfinite(res) else 0.0
    except Exception:
        return 0.0


def _eval_kf_vec(kf: dict[str, Any], anim_time: float, use_post: bool) -> tuple[float, float, float]:
    pts = kf["p"]
    raw = pts[1] if (use_post and len(pts) > 1) else pts[0]
    return (
        eval_molang_expr(raw[0], anim_time),
        eval_molang_expr(raw[1], anim_time),
        eval_molang_expr(raw[2], anim_time),
    )


def eval_bone_channel_py(
    kfs: list[dict[str, Any]],
    t: float,
    loop_mode: str,
    is_scale: bool = False,
) -> tuple[float, float, float]:
    """Evaluate a Blockbench 5 bone track (`linear`, `catmullrom`, `step` + Molang) at time `t`."""
    default = (1.0, 1.0, 1.0) if is_scale else (0.0, 0.0, 0.0)
    if not kfs:
        return default
    n = len(kfs)
    if n == 1:
        return _eval_kf_vec(kfs[0], t, False)
    if t <= kfs[0]["t"]:
        return _eval_kf_vec(kfs[0], t, False)
    if t >= kfs[-1]["t"]:
        return _eval_kf_vec(kfs[-1], t, len(kfs[-1]["p"]) > 1)
    i1 = 1
    while i1 < n - 1 and kfs[i1]["t"] < t:
        i1 += 1
    i0 = i1 - 1
    k0, k1 = kfs[i0], kfs[i1]
    dt = max(1e-6, k1["t"] - k0["t"])
    u = max(0.0, min(1.0, (t - k0["t"]) / dt))
    v0 = _eval_kf_vec(k0, t, len(k0["p"]) > 1)
    if k0["i"] == "step":
        return v0
    v1 = _eval_kf_vec(k1, t, False)
    if k0["i"] == "catmullrom" or k1["i"] == "catmullrom":
        im = i0 - 1 if i0 > 0 else (n - 1 if loop_mode == "loop" else 0)
        ip = i1 + 1 if i1 < n - 1 else (0 if loop_mode == "loop" else n - 1)
        vm = _eval_kf_vec(kfs[im], t, False)
        vp = _eval_kf_vec(kfs[ip], t, False)
        u2 = u * u
        u3 = u2 * u
        out = []
        for c in range(3):
            out.append(
                0.5
                * (
                    (2.0 * v0[c])
                    + (-vm[c] + v1[c]) * u
                    + (2.0 * vm[c] - 5.0 * v0[c] + 4.0 * v1[c] - vp[c]) * u2
                    + (-vm[c] + 3.0 * v0[c] - 3.0 * v1[c] + vp[c]) * u3
                )
            )
        return (out[0], out[1], out[2])
    return (
        v0[0] + (v1[0] - v0[0]) * u,
        v0[1] + (v1[1] - v0[1]) * u,
        v0[2] + (v1[2] - v0[2]) * u,
    )


# ---------------------------------------------------------------------------
# Rotation & Matrix Helpers (Exact Blockbench ZYX Intrinsic == Rz @ Ry @ Rx)
# ---------------------------------------------------------------------------


def euler_zyx_deg_to_quat(rx_deg: float, ry_deg: float, rz_deg: float) -> tuple[float, float, float, float]:
    """Convert Blockbench 5 Euler degrees `(rx, ry, rz)` (`R = Rz @ Ry @ Rx`) to unit glTF quaternion `(x, y, z, w)`."""
    hx = math.radians(float(rx_deg)) * 0.5
    hy = math.radians(float(ry_deg)) * 0.5
    hz = math.radians(float(rz_deg)) * 0.5
    cx, sx = math.cos(hx), math.sin(hx)
    cy, sy = math.cos(hy), math.sin(hy)
    cz, sz = math.cos(hz), math.sin(hz)
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    qw = cx * cy * cz + sx * sy * sz
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    inv = 1.0 / norm
    return (qx * inv, qy * inv, qz * inv, qw * inv)


def quat_to_mat3(q: tuple[float, float, float, float] | list[float]) -> np.ndarray:
    """Convert unit quaternion `(x, y, z, w)` to 3x3 rotation matrix."""
    x, y, z, w = (float(c) for c in q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _trs_mat4(
    t: tuple[float, float, float],
    q: tuple[float, float, float, float],
    s: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    r = quat_to_mat3(q)
    m[:3, 0] = r[:, 0] * float(s[0])
    m[:3, 1] = r[:, 1] * float(s[1])
    m[:3, 2] = r[:, 2] * float(s[2])
    m[:3, 3] = [float(t[0]), float(t[1]), float(t[2])]
    return m


def _f32_clean(x: float) -> float:
    """Round-trip Python float through exact IEEE-754 float32 (eliminating -0.0 and float64/f32 min/max drift)."""
    v = struct.unpack("<f", struct.pack("<f", float(x)))[0]
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v == 0.0 else v


def _sanitize_name(raw: str, fallback: str, used: set[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "").strip()).strip("_") or fallback
    cand = cleaned
    idx = 2
    while cand in used:
        cand = f"{cleaned}_{idx}"
        idx += 1
    used.add(cand)
    return cand


# ---------------------------------------------------------------------------
# Binary Buffer Builder (4-Byte Aligned, Exact Float32 Min/Max)
# ---------------------------------------------------------------------------


class _GltfBinBuilder:
    """Builds tightly 4-byte-aligned glTF 2.0 `BIN` bufferViews and accessors."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def _pad4(self) -> None:
        rem = len(self.buf) % 4
        if rem:
            self.buf.extend(b"\x00" * (4 - rem))

    def append_buffer_view(self, raw_bytes: bytes, *, target: int | None = None, name: str | None = None) -> int:
        self._pad4()
        offset = len(self.buf)
        self.buf.extend(raw_bytes)
        bv: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(raw_bytes),
        }
        if target is not None:
            bv["target"] = int(target)
        if name:
            bv["name"] = name
        self._pad4()
        idx = len(self.buffer_views)
        self.buffer_views.append(bv)
        return idx

    def append_accessor(
        self,
        arr: np.ndarray,
        *,
        component_type: int,
        acc_type: str,
        target: int | None = None,
        include_bounds: bool = False,
        name: str | None = None,
    ) -> int:
        dtype_map = {
            COMP_BYTE: "<i1",
            COMP_UBYTE: "<u1",
            COMP_SHORT: "<i2",
            COMP_USHORT: "<u2",
            COMP_UINT: "<u4",
            COMP_FLOAT: "<f4",
        }
        a = np.ascontiguousarray(arr, dtype=dtype_map[component_type])
        if component_type == COMP_FLOAT:
            # Replace non-finite values & canonicalize -0.0 -> +0.0 BEFORE packing and min/max
            a = np.where(np.isfinite(a), a, np.float32(0.0)).astype("<f4", copy=False)
            a = np.where(a == np.float32(0.0), np.float32(0.0), a).astype("<f4", copy=False)
        n_comps = TYPE_NUM_COMPS[acc_type]
        if a.ndim == 1:
            count = int(a.shape[0] // n_comps)
            a_2d = a.reshape((count, n_comps))
        else:
            count = int(a.shape[0])
            a_2d = a.reshape((count, n_comps))

        bv_idx = self.append_buffer_view(a_2d.tobytes(), target=target)
        acc: dict[str, Any] = {
            "bufferView": bv_idx,
            "componentType": int(component_type),
            "count": count,
            "type": acc_type,
        }
        if include_bounds and count > 0:
            if component_type == COMP_FLOAT:
                # Exact float32 min/max cast to Python float via unpack("<f", ...) -> zero float64 mismatch!
                mins = a_2d.min(axis=0)
                maxs = a_2d.max(axis=0)
                acc["min"] = [_f32_clean(float(v)) for v in mins]
                acc["max"] = [_f32_clean(float(v)) for v in maxs]
            else:
                mins = a_2d.min(axis=0)
                maxs = a_2d.max(axis=0)
                acc["min"] = [int(v) for v in mins]
                acc["max"] = [int(v) for v in maxs]
        if name:
            acc["name"] = name
        idx = len(self.accessors)
        self.accessors.append(acc)
        return idx


# ---------------------------------------------------------------------------
# Texture & Material Packing Helpers (Embedded PNGs + Shared Linear ORM)
# ---------------------------------------------------------------------------


def _encode_png_bytes(img_u8: np.ndarray, mode: str = "RGBA") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img_u8, mode=mode).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _decode_data_url_rgba(source: str) -> np.ndarray | None:
    if not source or not isinstance(source, str) or "," not in source:
        return None
    try:
        raw = base64.b64decode(source.split(",", 1)[1])
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def _build_material_maps_from_graph(
    graph: Any,
    *,
    bleed_px: int = 4,
    use_baked_albedo: bool = False,
    lighting_preset: str = "studio_hero",
    coat_weight: float = 0.0,
) -> dict[str, Any]:
    """Evaluate a `TexGraph` into glTF 2.0 embedded PNG buffers (BaseColor, Shared ORM, OpenGL Normal, Emissive)."""
    ev = graph.evaluate(phase=0.0, capture_intermediates=False, build_pbr=True, allow_orphan_errors=True)
    pm = ev.pbr_maps or pack_labpbr_1_3(ev.output)
    ch = ev.pbr_channels or {}
    H, W = int(ev.output.shape[0]), int(ev.output.shape[1])
    mc = getattr(graph, "mesh_context", None)

    albedo_f = np.asarray(ev.output, dtype=np.float32)
    h_f = np.asarray(ch["height"], dtype=np.float32) if "height" in ch and ch["height"] is not None else np.full((H, W), 0.5, dtype=np.float32)
    r_f = np.asarray(ch["roughness"], dtype=np.float32) if "roughness" in ch and ch["roughness"] is not None else np.full((H, W), 0.55, dtype=np.float32)
    m_f = np.asarray(ch["metallic"], dtype=np.float32) if "metallic" in ch and ch["metallic"] is not None else np.zeros((H, W), dtype=np.float32)
    ao_f = np.asarray(ch["ao"], dtype=np.float32) if "ao" in ch and ch["ao"] is not None else np.ones((H, W), dtype=np.float32)

    if use_baked_albedo:
        baked = bake_pbr_to_albedo(
            albedo_f,
            h_f,
            r_f,
            m_f,
            ao_f,
            emissive=ev.emissive,
            baked_shadow=ch.get("shadow"),
            bent_normal=ch.get("bent_normal"),
            thickness=ch.get("thickness"),
            sss=ch.get("sss"),
            preset=lighting_preset,
        )
        albedo_u8 = baked if baked.dtype == np.uint8 else rgba_float_to_u8(baked)
    else:
        albedo_u8 = np.asarray(pm.get("albedo", rgba_float_to_u8(albedo_f)), dtype=np.uint8).copy()

    # Determine alphaMode BEFORE dilation (checking real UV island texels!)
    island_mask = getattr(mc, "island_mask", None)
    valid_seed = (
        np.asarray(island_mask, dtype=bool)
        if (island_mask is not None and island_mask.shape == (H, W) and np.any(island_mask))
        else (albedo_u8[..., 3] > 5)
    )
    alpha_vals = albedo_u8[valid_seed, 3] if np.any(valid_seed) else albedo_u8[..., 3].ravel()
    has_cutout = bool(np.any(alpha_vals <= 12))
    has_semitrans = bool(np.any((alpha_vals > 12) & (alpha_vals < 245)))
    if has_semitrans:
        alpha_mode = "BLEND"
    elif has_cutout:
        alpha_mode = "MASK"
    else:
        alpha_mode = "OPAQUE"

    # Extract OpenGL (+Y up) tangent-space normal RGB uint8
    if "normal_rgb" in pm:
        nrm_u8 = np.asarray(pm["normal_rgb"], dtype=np.uint8)[..., :3].copy()
    else:
        nrm_f = height_to_normal_map(h_f, strength=float(graph.pbr_config.get("normal_strength", 2.4)))
        nrm_u8 = np.clip(nrm_f * 255.0, 0, 255).round().astype(np.uint8)

    # Build linear ORM RGBA (R=AO, G=Roughness, B=Metallic, A=255)
    if "orm_rgb" in pm:
        orm_rgb = np.asarray(pm["orm_rgb"], dtype=np.uint8)[..., :3]
        orm_u8 = np.concatenate((orm_rgb, np.full((H, W, 1), 255, dtype=np.uint8)), axis=-1)
    else:
        orm_u8 = np.empty((H, W, 4), dtype=np.uint8)
        orm_u8[..., 0] = np.clip(ao_f * 255.0, 0, 255).round().astype(np.uint8)
        orm_u8[..., 1] = np.clip(r_f * 255.0, 0, 255).round().astype(np.uint8)
        orm_u8[..., 2] = np.clip(m_f * 255.0, 0, 255).round().astype(np.uint8)
        orm_u8[..., 3] = 255

    # Optional Emissive map (only emit if meaningful glow exists)
    em_png: bytes | None = None
    has_emissive = False
    emissive_strength = 1.0
    if ev.emissive is not None:
        em_f = np.asarray(ev.emissive, dtype=np.float32)
        if em_f.ndim == 3 and em_f.shape[2] >= 3:
            em_a = em_f[..., 3:4] if em_f.shape[2] == 4 else np.ones((H, W, 1), dtype=np.float32)
            em_rgb = np.clip(em_f[..., :3] * em_a, 0.0, 1.0)
            if float(em_rgb.max()) > 0.03:
                has_emissive = True
                em_u8 = np.concatenate(
                    (
                        np.clip(em_rgb * 255.0, 0, 255).round().astype(np.uint8),
                        np.full((H, W, 1), 255, dtype=np.uint8),
                    ),
                    axis=-1,
                )
                if bleed_px > 0 and np.any(valid_seed) and not np.all(valid_seed):
                    em_u8 = mip_safe_dilate_rgba(em_u8, valid_mask=valid_seed, bleed_px=bleed_px, preserve_alpha=False)
                em_png = _encode_png_bytes(em_u8[..., :3], mode="RGB")
                emissive_strength = 1.45

    # Mip-safe Voronoi edge bleed (protecting straight-alpha cutouts & unit tangent normals!)
    if bleed_px > 0 and np.any(valid_seed) and not np.all(valid_seed):
        albedo_u8 = mip_safe_dilate_rgba(albedo_u8, valid_mask=valid_seed, bleed_px=bleed_px, preserve_alpha=(alpha_mode != "OPAQUE"))
        orm_u8 = mip_safe_dilate_rgba(orm_u8, valid_mask=valid_seed, bleed_px=bleed_px, preserve_alpha=False)
        nrm_u8 = mip_safe_dilate_normal(nrm_u8, valid_seed, bleed_px=bleed_px)

    return {
        "width": W,
        "height": H,
        "alpha_mode": alpha_mode,
        "double_sided": alpha_mode != "OPAQUE",
        "base_color_png": _encode_png_bytes(albedo_u8, mode="RGBA"),
        "orm_png": _encode_png_bytes(orm_u8[..., :3], mode="RGB"),
        "normal_png": _encode_png_bytes(nrm_u8[..., :3], mode="RGB"),
        "emissive_png": em_png if has_emissive else None,
        "emissive_strength": emissive_strength if has_emissive else 1.0,
        "coat_weight": float(np.clip(coat_weight, 0.0, 1.0)),
    }


def _build_material_maps_from_slot(slot_info: dict[str, Any], *, bleed_px: int = 4) -> dict[str, Any]:
    """Build embedded PNG material maps directly from a `.bbmodel` `textures[slot]` entry."""
    rgba = _decode_data_url_rgba(str(slot_info.get("source") or ""))
    if rgba is None:
        w = max(16, min(256, int(slot_info.get("width") or 64)))
        h = max(16, min(256, int(slot_info.get("height") or 64)))
        rgba = np.full((h, w, 4), (168, 174, 188, 255), dtype=np.uint8)
    H, W = int(rgba.shape[0]), int(rgba.shape[1])
    a = rgba[..., 3]
    has_cutout = bool(np.any(a <= 12))
    has_semitrans = bool(np.any((a > 12) & (a < 245)))
    alpha_mode = "BLEND" if has_semitrans else ("MASK" if has_cutout else "OPAQUE")
    valid_m = a > 12
    if bleed_px > 0 and has_cutout and np.any(valid_m) and not np.all(valid_m):
        rgba = mip_safe_dilate_rgba(rgba, valid_mask=valid_m, bleed_px=bleed_px, preserve_alpha=True)
    return {
        "width": W,
        "height": H,
        "alpha_mode": alpha_mode,
        "double_sided": alpha_mode != "OPAQUE",
        "base_color_png": _encode_png_bytes(rgba, mode="RGBA"),
        "orm_png": None,
        "normal_png": None,
        "emissive_png": None,
        "emissive_strength": 1.0,
        "coat_weight": 0.0,
    }


# ---------------------------------------------------------------------------
# Skeleton, Indexed Mesh Primitives & Tangent VEC4 Builder
# ---------------------------------------------------------------------------


def _extract_bbmodel_rig_and_clips(
    bb_doc: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    dict[str, list[tuple[np.ndarray, np.ndarray]]],
    list[dict[str, Any]],
]:
    """Extract hierarchical Blockbench 5 bones (parent-before-child) & animation clips."""
    groups_by_id: dict[str, dict[str, Any]] = {}
    for g in bb_doc.get("groups") or []:
        if isinstance(g, dict) and g.get("uuid"):
            groups_by_id[str(g["uuid"])] = g

    bones: list[dict[str, Any]] = []
    bone_idx_by_uuid: dict[str, int] = {}
    bone_idx_by_name: dict[str, int] = {}
    elem_bone_idx: dict[str, int] = {}
    elem_chains: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    def _walk(nodes: list[Any], parent_idx: int, chain: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for item in nodes or []:
            if isinstance(item, str):
                elem_bone_idx[item] = parent_idx
                elem_chains[item] = list(reversed(chain))
            elif isinstance(item, dict):
                uid = str(item.get("uuid") or f"bone_{len(bones)}")
                meta = groups_by_id.get(uid, item)
                bname = str(meta.get("name") or item.get("name") or f"bone_{len(bones)}")
                orig_raw = meta.get("origin", item.get("origin", [0.0, 0.0, 0.0])) or [0.0, 0.0, 0.0]
                rot_raw = meta.get("rotation", item.get("rotation", [0.0, 0.0, 0.0])) or [0.0, 0.0, 0.0]
                origin = np.array([float(orig_raw[0]), float(orig_raw[1]), float(orig_raw[2])], dtype=np.float64)
                rest_rot = [float(rot_raw[0]), float(rot_raw[1]), float(rot_raw[2])]
                b_idx = len(bones)
                bones.append(
                    {
                        "id": b_idx,
                        "uuid": uid,
                        "name": bname,
                        "parent": parent_idx,
                        "origin": origin,
                        "rest_rot": rest_rot,
                    }
                )
                bone_idx_by_uuid[uid] = b_idx
                bone_idx_by_name[bname] = b_idx
                R = _euler_xyz_mat(rest_rot)
                _walk(item.get("children") or [], b_idx, chain + [(origin, R)])

    _walk(bb_doc.get("outliner") or [], -1, [])

    clips: list[dict[str, Any]] = []
    for anim in bb_doc.get("animations") or []:
        if not isinstance(anim, dict):
            continue
        aname = str(anim.get("name") or f"clip_{len(clips) + 1}")
        loop_val = anim.get("loop", "loop")
        loop_mode = ("loop" if loop_val else "once") if isinstance(loop_val, bool) else str(loop_val or "loop")
        length_s = max(1.0 / 30.0, float(anim.get("length") or 1.0))
        tracks: dict[int, dict[str, list[dict[str, Any]]]] = {}
        animators = anim.get("animators") or {}
        if isinstance(animators, dict):
            for a_uid, a_obj in animators.items():
                if not isinstance(a_obj, dict) or a_obj.get("type") not in (None, "bone"):
                    continue
                b_idx = bone_idx_by_uuid.get(str(a_uid))
                if b_idx is None and a_obj.get("name"):
                    b_idx = bone_idx_by_name.get(str(a_obj["name"]))
                if b_idx is None:
                    continue
                ch_map: dict[str, list[dict[str, Any]]] = {}
                for kf in a_obj.get("keyframes") or []:
                    if not isinstance(kf, dict):
                        continue
                    ch = str(kf.get("channel") or "")
                    if ch not in ("rotation", "position", "scale"):
                        continue
                    pts = [
                        [dp.get("x", 0), dp.get("y", 0), dp.get("z", 0)]
                        for dp in (kf.get("data_points") or [])
                        if isinstance(dp, dict)
                    ]
                    if not pts:
                        continue
                    ch_map.setdefault(ch, []).append(
                        {
                            "t": max(0.0, float(kf.get("time", 0.0))),
                            "i": str(kf.get("interpolation") or "linear"),
                            "p": pts,
                        }
                    )
                if ch_map:
                    for ch_k in ch_map:
                        ch_map[ch_k].sort(key=lambda r: r["t"])
                    tracks[b_idx] = ch_map
        if tracks:
            clips.append({"name": aname, "loop": loop_mode, "length": length_s, "tracks": tracks})

    return bones, elem_bone_idx, elem_chains, clips


def _compute_quad_tangent_vec4(
    p_tl: np.ndarray,
    p_tr: np.ndarray,
    p_br: np.ndarray,
    uv_tl: tuple[float, float],
    uv_tr: tuple[float, float],
    uv_br: tuple[float, float],
    wn: np.ndarray,
    fallback_t: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute orthonormal glTF 2.0 `TANGENT` (`VEC4`) with exact `+/-1` handedness for OpenGL `+Y` normal maps."""
    e1 = p_tr - p_tl
    e2 = p_br - p_tl
    du1 = uv_tr[0] - uv_tl[0]
    du2 = uv_br[0] - uv_tl[0]
    # glTF UV +V points DOWN; flip sign so bitangent B aligns with OpenGL +Y normal-map green channel!
    dv1 = -(uv_tr[1] - uv_tl[1])
    dv2 = -(uv_br[1] - uv_tl[1])
    det = du1 * dv2 - du2 * dv1
    if abs(det) > 1e-12:
        inv_det = 1.0 / det
        t_raw = (dv2 * e1 - dv1 * e2) * inv_det
        b_raw = (-du2 * e1 + du1 * e2) * inv_det
    else:
        t_raw = fallback_t
        b_raw = np.cross(wn, t_raw)

    # Gram-Schmidt orthogonalize T against unit normal N
    t_ortho = t_raw - wn * float(np.dot(wn, t_raw))
    tn = float(np.linalg.norm(t_ortho))
    if tn < 1e-8:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(wn[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        t_ortho = np.cross(wn, ref)
        tn = float(np.linalg.norm(t_ortho))
    t_unit = t_ortho / max(1e-12, tn)
    w_sign = 1.0 if float(np.dot(np.cross(wn, t_unit), b_raw)) >= 0.0 else -1.0
    return (float(t_unit[0]), float(t_unit[1]), float(t_unit[2]), w_sign)


# ---------------------------------------------------------------------------
# Main Pure-Python Khronos glTF 2.0 Binary (`.glb`) Exporter
# ---------------------------------------------------------------------------


def export_bbmodel_to_glb(
    bb_doc: dict[str, Any],
    *,
    graph: Any = None,
    out_path: str | Path | None = None,
    meter_scale: float = 1.0 / 16.0,
    bleed_px: int = 4,
    include_animations: bool = True,
    fps: int = 24,
    use_baked_albedo: bool = False,
    lighting_preset: str = "studio_hero",
    coat_weight: float = 0.0,
    pixel_filter: str = "nearest_mipmap",
) -> tuple[bytes, dict[str, Any]]:
    """Export any Blockbench 5 `.bbmodel` (+ optional live `TexGraph`) to a Khronos glTF 2.0 `.glb` binary.

    Returns `(glb_bytes, report_dict)` with complete structural stats and validation summary.
    """
    t0 = time.perf_counter()
    scale = float(meter_scale) if meter_scale > 0 else (1.0 / 16.0)
    res = bb_doc.get("resolution") or {}
    tw = max(1.0, float(res.get("width", 64)))
    th = max(1.0, float(res.get("height", 64)))
    raw_textures = [t for t in (bb_doc.get("textures") or []) if isinstance(t, dict)]
    texture_slots = extract_bbmodel_texture_slots(bb_doc)
    bones, elem_bone_idx, elem_chains, clips = _extract_bbmodel_rig_and_clips(bb_doc)

    has_skin = len(bones) > 0
    used_node_names: set[str] = set()
    used_anim_names: set[str] = set()
    used_mat_names: set[str] = set()

    # 1. Build Joint Hierarchy & Global Bind Matrices (`node 0` = Armature root when skinned!)
    #    Joint index mapping: joint 0 = Armature (identity, catches unparented elems), joint (bi+1) = bones[bi]
    gltf_nodes: list[dict[str, Any]] = []
    joint_node_indices: list[int] = []
    global_bind_mats: list[np.ndarray] = []

    if has_skin:
        arm_name = _sanitize_name("Armature", "Armature", used_node_names)
        arm_children: list[int] = []
        gltf_nodes.append({"name": arm_name})  # node 0
        joint_node_indices.append(0)
        global_bind_mats.append(np.eye(4, dtype=np.float64))

        bone_children: dict[int, list[int]] = {bi: [] for bi in range(len(bones))}
        for bi, bn in enumerate(bones):
            node_idx = bi + 1
            p_idx = bn["parent"]
            p_orig = bones[p_idx]["origin"] if 0 <= p_idx < len(bones) else np.zeros(3, dtype=np.float64)
            t_loc = tuple(float((bn["origin"][k] - p_orig[k]) * scale) for k in range(3))
            q_loc = euler_zyx_deg_to_quat(bn["rest_rot"][0], bn["rest_rot"][1], bn["rest_rot"][2])
            m_loc = _trs_mat4(t_loc, q_loc)
            p_global = global_bind_mats[p_idx + 1] if 0 <= p_idx < len(bones) else global_bind_mats[0]
            global_bind_mats.append(p_global @ m_loc)

            nd: dict[str, Any] = {"name": _sanitize_name(bn["name"], f"bone_{bi}", used_node_names)}
            if any(abs(c) > 1e-7 for c in t_loc):
                nd["translation"] = [_f32_clean(c) for c in t_loc]
            if abs(q_loc[0]) > 1e-7 or abs(q_loc[1]) > 1e-7 or abs(q_loc[2]) > 1e-7 or abs(q_loc[3] - 1.0) > 1e-7:
                nd["rotation"] = [_f32_clean(c) for c in q_loc]
            gltf_nodes.append(nd)
            joint_node_indices.append(node_idx)
            if 0 <= p_idx < len(bones):
                bone_children[p_idx].append(node_idx)
            else:
                arm_children.append(node_idx)

        if arm_children:
            gltf_nodes[0]["children"] = arm_children
        for bi, ch_list in bone_children.items():
            if ch_list:
                gltf_nodes[bi + 1]["children"] = ch_list

    # 2. Extract Per-Texture-Slot Indexed Quad Geometry (strictly matching Blockbench bind positions!)
    slot_prims: dict[int, dict[str, list[Any]]] = {}

    def _get_slot_bucket(s_idx: int) -> dict[str, list[Any]]:
        if s_idx not in slot_prims:
            slot_prims[s_idx] = {
                "pos": [],
                "nrm": [],
                "tan": [],
                "uv": [],
                "joints": [],
                "weights": [],
                "indices": [],
            }
        return slot_prims[s_idx]

    box_count = 0
    for elem in bb_doc.get("elements") or []:
        if not isinstance(elem, dict) or elem.get("visibility") is False:
            continue
        f_pt = elem.get("from")
        t_pt = elem.get("to")
        if not f_pt or not t_pt:
            continue
        box_count += 1
        inf = float(elem.get("inflate", 0.0) or 0.0)
        x0, y0, z0 = float(f_pt[0]) - inf, float(f_pt[1]) - inf, float(f_pt[2]) - inf
        x1, y1, z1 = float(t_pt[0]) + inf, float(t_pt[1]) + inf, float(t_pt[2]) + inf
        cx, cy, cz = 0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)
        hx, hy, hz = 0.5 * (x1 - x0), 0.5 * (y1 - y0), 0.5 * (z1 - z0)

        e_orig = np.array(elem.get("origin", [cx, cy, cz]), dtype=np.float64)
        e_rot = _euler_xyz_mat(elem.get("rotation", [0.0, 0.0, 0.0]))
        e_uid = str(elem.get("uuid") or "")
        chain = elem_chains.get(e_uid, [])
        b_idx = elem_bone_idx.get(e_uid, -1)
        joint_slot = (b_idx + 1) if (has_skin and 0 <= b_idx < len(bones)) else 0

        def _xform_pt(p: np.ndarray) -> np.ndarray:
            q = e_orig + e_rot @ (p - e_orig)
            for b_orig, b_rot in chain:
                q = b_orig + b_rot @ (q - b_orig)
            return q * scale

        def _xform_dir(d: np.ndarray) -> np.ndarray:
            q = e_rot @ d
            for _, b_rot in chain:
                q = b_rot @ q
            ln = float(np.linalg.norm(q))
            return q / ln if ln > 1e-8 else q

        faces = elem.get("faces") or {}
        center_l = np.array([cx, cy, cz], dtype=np.float64)
        half_l = np.array([hx, hy, hz], dtype=np.float64)

        for fname, (tu_l, tv_l, fn_l) in _FACE_BASIS.items():
            finfo = faces.get(fname)
            if not isinstance(finfo, dict):
                continue
            if "texture" in finfo and finfo["texture"] is None:
                continue
            uv_rect = finfo.get("uv")
            if not uv_rect or len(uv_rect) < 4:
                continue
            slot_idx = resolve_face_texture_slot(finfo.get("texture", 0), raw_textures)
            if slot_idx is None:
                continue
            f_tw, f_th = tw, th
            if 0 <= slot_idx < len(texture_slots):
                f_tw = max(1.0, float(texture_slots[slot_idx].get("uv_width") or tw))
                f_th = max(1.0, float(texture_slots[slot_idx].get("uv_height") or th))
            u0 = float(uv_rect[0]) / f_tw
            v0 = float(uv_rect[1]) / f_th
            u1 = float(uv_rect[2]) / f_tw
            v1 = float(uv_rect[3]) / f_th
            if abs(u1 - u0) < 1e-6 or abs(v1 - v0) < 1e-6:
                continue

            tu_v = np.array(tu_l, dtype=np.float64)
            tv_v = np.array(tv_l, dtype=np.float64)
            fn_v = np.array(fn_l, dtype=np.float64)

            du = tu_v * float(np.abs(np.dot(tu_v, half_l)))
            dv = tv_v * float(np.abs(np.dot(tv_v, half_l)))
            dn = fn_v * float(np.abs(np.dot(fn_v, half_l)))
            fc = center_l + dn

            p_tl = _xform_pt(fc - du - dv)
            p_tr = _xform_pt(fc + du - dv)
            p_br = _xform_pt(fc + du + dv)
            p_bl = _xform_pt(fc - du + dv)

            # Skip zero-area degenerate faces (prevents non-unit normal / NaN tangent validator failures!)
            cross_diag = np.cross(p_tr - p_tl, p_br - p_tl)
            if float(np.linalg.norm(cross_diag)) < 1e-10:
                continue

            wn = _xform_dir(fn_v)
            wt_fb = _xform_dir(tu_v)
            tan4 = _compute_quad_tangent_vec4(p_tl, p_tr, p_br, (u0, v0), (u1, v0), (u1, v1), wn, wt_fb)

            bucket = _get_slot_bucket(slot_idx)
            v_base = len(bucket["pos"]) // 3
            quad_corners = (
                (p_tl, (u0, v0)),
                (p_tr, (u1, v0)),
                (p_br, (u1, v1)),
                (p_bl, (u0, v1)),
            )
            for pt, uv_pt in quad_corners:
                bucket["pos"].extend((pt[0], pt[1], pt[2]))
                bucket["nrm"].extend((wn[0], wn[1], wn[2]))
                bucket["tan"].extend(tan4)
                bucket["uv"].extend((uv_pt[0], uv_pt[1]))
                if has_skin:
                    bucket["joints"].extend((joint_slot, 0, 0, 0))
                    bucket["weights"].extend((1.0, 0.0, 0.0, 0.0))

            # Guarantee right-handed outward CCW triangle winding!
            if float(np.dot(cross_diag, wn)) >= 0.0:
                bucket["indices"].extend((v_base, v_base + 1, v_base + 2, v_base, v_base + 2, v_base + 3))
            else:
                bucket["indices"].extend((v_base, v_base + 2, v_base + 1, v_base, v_base + 3, v_base + 2))

    if not slot_prims:
        raise ValueError("Cannot export .glb: .bbmodel contains zero visible textured faces")

    # 3. Assemble Binary Buffer, Materials, Embedded PNGs, Mesh Primitives, Skin & Animations
    bb = _GltfBinBuilder()
    extensions_used: list[str] = []

    def _use_ext(ext_name: str) -> None:
        if ext_name not in extensions_used:
            extensions_used.append(ext_name)

    min_f = FILTER_NEAREST_MIPMAP_LINEAR if (pixel_filter == "nearest_mipmap" and bleed_px > 0) else FILTER_NEAREST
    samplers = [
        {
            "magFilter": FILTER_NEAREST,
            "minFilter": min_f,
            "wrapS": WRAP_CLAMP_TO_EDGE,
            "wrapT": WRAP_CLAMP_TO_EDGE,
        }
    ]
    images: list[dict[str, Any]] = []
    textures: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    slot_to_mat_idx: dict[int, int] = {}

    def _embed_png_texture(png_bytes: bytes, tex_name: str) -> int:
        bv_idx = bb.append_buffer_view(png_bytes, target=None, name=f"img_{tex_name}")
        img_idx = len(images)
        images.append({"bufferView": bv_idx, "mimeType": "image/png", "name": tex_name})
        tex_idx = len(textures)
        textures.append({"sampler": 0, "source": img_idx, "name": tex_name})
        return tex_idx

    # Pre-evaluate graph PBR maps once (used for slot 0 or single-slot models!)
    graph_maps = (
        _build_material_maps_from_graph(
            graph,
            bleed_px=bleed_px,
            use_baked_albedo=use_baked_albedo,
            lighting_preset=lighting_preset,
            coat_weight=coat_weight,
        )
        if graph is not None
        else None
    )

    embedded_image_count = 0
    for s_idx in sorted(slot_prims.keys()):
        s_meta = texture_slots[s_idx] if 0 <= s_idx < len(texture_slots) else {"name": f"slot_{s_idx}"}
        m_name = _sanitize_name(str(s_meta.get("name") or f"mat_{s_idx}").rsplit(".", 1)[0], f"mat_{s_idx}", used_mat_names)
        if graph_maps is not None and (s_idx == 0 or len(slot_prims) == 1):
            m_maps = graph_maps
        else:
            m_maps = _build_material_maps_from_slot(s_meta, bleed_px=bleed_px)

        bc_tex = _embed_png_texture(m_maps["base_color_png"], f"{m_name}_baseColor")
        embedded_image_count += 1
        pbr_mr: dict[str, Any] = {
            "baseColorTexture": {"index": bc_tex},
            "metallicFactor": 1.0 if m_maps["orm_png"] is not None else 0.0,
            "roughnessFactor": 1.0 if m_maps["orm_png"] is not None else 0.72,
        }
        mat_obj: dict[str, Any] = {
            "name": m_name,
            "pbrMetallicRoughness": pbr_mr,
        }
        if m_maps["orm_png"] is not None:
            orm_tex = _embed_png_texture(m_maps["orm_png"], f"{m_name}_orm")
            embedded_image_count += 1
            pbr_mr["metallicRoughnessTexture"] = {"index": orm_tex}
            mat_obj["occlusionTexture"] = {"index": orm_tex, "strength": 1.0}
        if m_maps["normal_png"] is not None:
            nrm_tex = _embed_png_texture(m_maps["normal_png"], f"{m_name}_normal")
            embedded_image_count += 1
            mat_obj["normalTexture"] = {"index": nrm_tex, "scale": 1.0}
        if m_maps["emissive_png"] is not None:
            em_tex = _embed_png_texture(m_maps["emissive_png"], f"{m_name}_emissive")
            embedded_image_count += 1
            mat_obj["emissiveTexture"] = {"index": em_tex}
            mat_obj["emissiveFactor"] = [1.0, 1.0, 1.0]
            if m_maps["emissive_strength"] > 1.001:
                _use_ext("KHR_materials_emissive_strength")
                mat_obj.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": round(float(m_maps["emissive_strength"]), 3)
                }
        if m_maps["coat_weight"] > 0.01:
            _use_ext("KHR_materials_clearcoat")
            mat_obj.setdefault("extensions", {})["KHR_materials_clearcoat"] = {
                "clearcoatFactor": round(float(m_maps["coat_weight"]), 3),
                "clearcoatRoughnessFactor": 0.12,
            }
        if m_maps["alpha_mode"] == "MASK":
            mat_obj["alphaMode"] = "MASK"
            mat_obj["alphaCutoff"] = 0.5
        elif m_maps["alpha_mode"] == "BLEND":
            mat_obj["alphaMode"] = "BLEND"
        if m_maps["double_sided"]:
            mat_obj["doubleSided"] = True

        slot_to_mat_idx[s_idx] = len(materials)
        materials.append(mat_obj)

    # 4. Build Mesh Primitives
    primitives: list[dict[str, Any]] = []
    total_verts = 0
    total_tris = 0
    for s_idx in sorted(slot_prims.keys()):
        bkt = slot_prims[s_idx]
        v_count = len(bkt["pos"]) // 3
        t_count = len(bkt["indices"]) // 3
        total_verts += v_count
        total_tris += t_count

        pos_acc = bb.append_accessor(
            np.asarray(bkt["pos"], dtype=np.float32),
            component_type=COMP_FLOAT,
            acc_type="VEC3",
            target=TARGET_ARRAY_BUFFER,
            include_bounds=True,
        )
        nrm_acc = bb.append_accessor(
            np.asarray(bkt["nrm"], dtype=np.float32),
            component_type=COMP_FLOAT,
            acc_type="VEC3",
            target=TARGET_ARRAY_BUFFER,
        )
        tan_acc = bb.append_accessor(
            np.asarray(bkt["tan"], dtype=np.float32),
            component_type=COMP_FLOAT,
            acc_type="VEC4",
            target=TARGET_ARRAY_BUFFER,
        )
        uv_acc = bb.append_accessor(
            np.asarray(bkt["uv"], dtype=np.float32),
            component_type=COMP_FLOAT,
            acc_type="VEC2",
            target=TARGET_ARRAY_BUFFER,
        )
        # Switch to UNSIGNED_INT when v_count >= 65535 (index 65535 is forbidden primitive restart sentinel!)
        idx_comp = COMP_UINT if v_count >= 65535 else COMP_USHORT
        idx_acc = bb.append_accessor(
            np.asarray(bkt["indices"], dtype=(np.uint32 if idx_comp == COMP_UINT else np.uint16)),
            component_type=idx_comp,
            acc_type="SCALAR",
            target=TARGET_ELEMENT_ARRAY_BUFFER,
        )

        attrs: dict[str, int] = {
            "POSITION": pos_acc,
            "NORMAL": nrm_acc,
            "TANGENT": tan_acc,
            "TEXCOORD_0": uv_acc,
        }
        if has_skin:
            j_comp = COMP_UBYTE if len(joint_node_indices) <= 255 else COMP_USHORT
            attrs["JOINTS_0"] = bb.append_accessor(
                np.asarray(bkt["joints"], dtype=(np.uint8 if j_comp == COMP_UBYTE else np.uint16)),
                component_type=j_comp,
                acc_type="VEC4",
                target=TARGET_ARRAY_BUFFER,
            )
            attrs["WEIGHTS_0"] = bb.append_accessor(
                np.asarray(bkt["weights"], dtype=np.float32),
                component_type=COMP_FLOAT,
                acc_type="VEC4",
                target=TARGET_ARRAY_BUFFER,
            )

        primitives.append(
            {
                "attributes": attrs,
                "indices": idx_acc,
                "material": slot_to_mat_idx[s_idx],
                "mode": 4,
            }
        )

    mesh_name = _sanitize_name(str(bb_doc.get("name") or "synccraft_mesh"), "synccraft_mesh", used_node_names)
    meshes = [{"name": mesh_name, "primitives": primitives}]

    # 5. Build Skin + Column-Major Inverse Bind Matrices (`IBM = inv(M_global_bind)`)
    skins: list[dict[str, Any]] = []
    mesh_node_idx = len(gltf_nodes)
    mesh_node: dict[str, Any] = {"name": f"{mesh_name}_Mesh", "mesh": 0}
    scene_root_nodes: list[int] = []

    if has_skin:
        ibm_list: list[np.ndarray] = []
        for gmat in global_bind_mats:
            inv_m = np.linalg.inv(gmat)
            # Enforce exact bottom row [0, 0, 0, 1] and pack column-major (transposed row-major)
            inv_m[3, :] = [0.0, 0.0, 0.0, 1.0]
            ibm_list.append(inv_m.T.reshape(16).astype(np.float32))
        ibm_acc = bb.append_accessor(
            np.stack(ibm_list, axis=0),
            component_type=COMP_FLOAT,
            acc_type="MAT4",
            target=None,
            name="inverseBindMatrices",
        )
        skins.append(
            {
                "name": f"{mesh_name}_Skin",
                "inverseBindMatrices": ibm_acc,
                "skeleton": 0,
                "joints": joint_node_indices,
            }
        )
        mesh_node["skin"] = 0
        # glTF 2.0 §3.7.3.1: place Armature & identity Skinned Mesh node as direct siblings of Scene root!
        scene_root_nodes = [0, mesh_node_idx]
    else:
        scene_root_nodes = [mesh_node_idx]

    gltf_nodes.append(mesh_node)

    # 6. Bake Blockbench 5 Skeletal Animations -> glTF 2.0 Absolute TRS Channels (Hemisphere-Continuous!)
    gltf_animations: list[dict[str, Any]] = []
    if has_skin and include_animations and clips:
        sample_fps = max(10, min(60, int(fps)))
        for clip in clips:
            dur = max(1.0 / 30.0, float(clip["length"]))
            n_frames = max(2, int(round(dur * sample_fps)) + 1)
            raw_times = np.linspace(0.0, dur, n_frames, dtype=np.float64)
            # Strictly increasing float32 timestamps
            times_f32: list[float] = []
            for tv in raw_times:
                tf = _f32_clean(float(tv))
                if not times_f32 or tf > times_f32[-1]:
                    times_f32.append(tf)
            if len(times_f32) < 2:
                times_f32 = [0.0, _f32_clean(max(1.0 / 30.0, dur))]
            times_arr = np.asarray(times_f32, dtype=np.float32)

            channels: list[dict[str, Any]] = []
            samplers_anim: list[dict[str, Any]] = []
            time_acc_idx: int | None = None

            def _ensure_time_acc() -> int:
                nonlocal time_acc_idx
                if time_acc_idx is None:
                    time_acc_idx = bb.append_accessor(
                        times_arr,
                        component_type=COMP_FLOAT,
                        acc_type="SCALAR",
                        target=None,
                        include_bounds=True,
                    )
                return time_acc_idx

            for bi, ch_map in sorted(clip["tracks"].items()):
                if not (0 <= bi < len(bones)):
                    continue
                bn = bones[bi]
                node_idx = bi + 1
                p_idx = bn["parent"]
                p_orig = bones[p_idx]["origin"] if 0 <= p_idx < len(bones) else np.zeros(3, dtype=np.float64)
                base_t = (
                    (bn["origin"][0] - p_orig[0]) * scale,
                    (bn["origin"][1] - p_orig[1]) * scale,
                    (bn["origin"][2] - p_orig[2]) * scale,
                )
                rest_r = bn["rest_rot"]
                rest_q = euler_zyx_deg_to_quat(rest_r[0], rest_r[1], rest_r[2])

                if ch_map.get("rotation"):
                    q_samples: list[tuple[float, float, float, float]] = []
                    max_dev = 0.0
                    for tv in times_f32:
                        dr = eval_bone_channel_py(ch_map["rotation"], tv, clip["loop"], is_scale=False)
                        q = euler_zyx_deg_to_quat(rest_r[0] + dr[0], rest_r[1] + dr[1], rest_r[2] + dr[2])
                        if q_samples:
                            prev = q_samples[-1]
                            if (prev[0] * q[0] + prev[1] * q[1] + prev[2] * q[2] + prev[3] * q[3]) < 0.0:
                                q = (-q[0], -q[1], -q[2], -q[3])
                        dot_rest = abs(rest_q[0] * q[0] + rest_q[1] * q[1] + rest_q[2] * q[2] + rest_q[3] * q[3])
                        max_dev = max(max_dev, 1.0 - min(1.0, dot_rest))
                        q_samples.append(q)
                    if max_dev > 1e-6:
                        q_arr = np.asarray(q_samples, dtype=np.float32)
                        # Re-normalize in float32 to pass strict Khronos quaternion unit norm check
                        q_norms = np.linalg.norm(q_arr, axis=1, keepdims=True)
                        q_arr = np.where(q_norms > 1e-6, q_arr / np.maximum(q_norms, 1e-6), np.array([0, 0, 0, 1], dtype=np.float32)).astype(np.float32)
                        out_acc = bb.append_accessor(q_arr, component_type=COMP_FLOAT, acc_type="VEC4", target=None)
                        s_idx_a = len(samplers_anim)
                        samplers_anim.append({"input": _ensure_time_acc(), "interpolation": "LINEAR", "output": out_acc})
                        channels.append({"sampler": s_idx_a, "target": {"node": node_idx, "path": "rotation"}})

                if ch_map.get("position"):
                    t_samples: list[tuple[float, float, float]] = []
                    max_dev = 0.0
                    for tv in times_f32:
                        dp = eval_bone_channel_py(ch_map["position"], tv, clip["loop"], is_scale=False)
                        tx = base_t[0] + dp[0] * scale
                        ty = base_t[1] + dp[1] * scale
                        tz = base_t[2] + dp[2] * scale
                        max_dev = max(max_dev, abs(dp[0]), abs(dp[1]), abs(dp[2]))
                        t_samples.append((tx, ty, tz))
                    if max_dev > 1e-5:
                        out_acc = bb.append_accessor(
                            np.asarray(t_samples, dtype=np.float32),
                            component_type=COMP_FLOAT,
                            acc_type="VEC3",
                            target=None,
                        )
                        s_idx_a = len(samplers_anim)
                        samplers_anim.append({"input": _ensure_time_acc(), "interpolation": "LINEAR", "output": out_acc})
                        channels.append({"sampler": s_idx_a, "target": {"node": node_idx, "path": "translation"}})

                if ch_map.get("scale"):
                    s_samples: list[tuple[float, float, float]] = []
                    max_dev = 0.0
                    for tv in times_f32:
                        ds = eval_bone_channel_py(ch_map["scale"], tv, clip["loop"], is_scale=True)
                        # Clamp away from zero/negative scale (prevents UE5/glTFast determinant flip!)
                        sx = max(0.01, float(ds[0]))
                        sy = max(0.01, float(ds[1]))
                        sz = max(0.01, float(ds[2]))
                        max_dev = max(max_dev, abs(sx - 1.0), abs(sy - 1.0), abs(sz - 1.0))
                        s_samples.append((sx, sy, sz))
                    if max_dev > 1e-4:
                        out_acc = bb.append_accessor(
                            np.asarray(s_samples, dtype=np.float32),
                            component_type=COMP_FLOAT,
                            acc_type="VEC3",
                            target=None,
                        )
                        s_idx_a = len(samplers_anim)
                        samplers_anim.append({"input": _ensure_time_acc(), "interpolation": "LINEAR", "output": out_acc})
                        channels.append({"sampler": s_idx_a, "target": {"node": node_idx, "path": "scale"}})

            if channels:
                gltf_animations.append(
                    {
                        "name": _sanitize_name(clip["name"], f"anim_{len(gltf_animations) + 1}", used_anim_names),
                        "channels": channels,
                        "samplers": samplers_anim,
                    }
                )

    # 7. Finalize glTF 2.0 JSON (strictly omitting empty top-level arrays!)
    bb._pad4()
    bin_bytes = bytes(bb.buf)
    gltf_doc: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "SyncCraft Studio v55.0 (TexGraph v19 Khronos glTF 2.0 Exporter)",
        },
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": scene_root_nodes}],
        "nodes": gltf_nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": samplers,
        "accessors": bb.accessors,
        "bufferViews": bb.buffer_views,
        "buffers": [{"byteLength": len(bin_bytes)}],
    }
    if skins:
        gltf_doc["skins"] = skins
    if gltf_animations:
        gltf_doc["animations"] = gltf_animations
    if extensions_used:
        gltf_doc["extensionsUsed"] = extensions_used

    json_raw = json.dumps(gltf_doc, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")
    json_pad = (4 - (len(json_raw) % 4)) % 4
    json_chunk = json_raw + (b" " * json_pad)

    total_len = 12 + (8 + len(json_chunk)) + (8 + len(bin_bytes))
    glb = bytearray()
    glb.extend(struct.pack("<III", GLB_MAGIC, GLB_VERSION, total_len))
    glb.extend(struct.pack("<II", len(json_chunk), GLB_CHUNK_JSON))
    glb.extend(json_chunk)
    glb.extend(struct.pack("<II", len(bin_bytes), GLB_CHUNK_BIN))
    glb.extend(bin_bytes)
    glb_bytes = bytes(glb)

    if out_path is not None:
        out_fp = Path(out_path)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_bytes(glb_bytes)

    val_res = validate_glb_bytes(glb_bytes)
    report = {
        "ok": val_res["valid"],
        "schema": "synccraft.glb_export/1.0",
        "name": mesh_name,
        "byte_length": len(glb_bytes),
        "kb": round(len(glb_bytes) / 1024.0, 1),
        "box_count": box_count,
        "vertex_count": total_verts,
        "triangle_count": total_tris,
        "primitive_count": len(primitives),
        "material_count": len(materials),
        "image_count": embedded_image_count,
        "joint_count": len(joint_node_indices),
        "animation_count": len(gltf_animations),
        "animation_names": [a["name"] for a in gltf_animations],
        "extensions_used": extensions_used,
        "meter_scale": round(scale, 6),
        "bleed_px": int(bleed_px),
        "validation": val_res,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    if out_path is not None:
        report["out_path"] = str(out_path)
    return glb_bytes, report


def export_lod_glb_bundle(
    bb_doc: dict[str, Any],
    out_dir: str | Path,
    *,
    stem: str | None = None,
    graph: Any = None,
    lod1_ratio: float = 0.58,
    lod2_ratio: float = 0.35,
    meter_scale: float = 1.0 / 16.0,
    bleed_px: int = 4,
    include_animations: bool = True,
    fps: int = 24,
    use_baked_albedo: bool = False,
    lighting_preset: str = "studio_hero",
    coat_weight: float = 0.0,
) -> dict[str, Any]:
    """Export silhouette- & skeleton-preserving `LOD0 / LOD1 / LOD2` rigged `.glb` files + `*_glb_lod_manifest.json`."""
    t0 = time.perf_counter()
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    st = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stem or bb_doc.get("name") or "synccraft_asset")).strip("_") or "synccraft_asset"

    chain = generate_bbmodel_lod_chain(bb_doc, lod1_ratio=lod1_ratio, lod2_ratio=lod2_ratio)
    levels_out: list[dict[str, Any]] = []
    written_files: dict[str, str] = {}

    for lvl in ("lod0", "lod1", "lod2"):
        info = chain[lvl]
        glb_fp = out_root / f"{st}_{lvl}.glb"
        _, rep = export_bbmodel_to_glb(
            info["bbmodel"],
            graph=graph,
            out_path=glb_fp,
            meter_scale=meter_scale,
            bleed_px=bleed_px,
            include_animations=include_animations,
            fps=fps,
            use_baked_albedo=use_baked_albedo,
            lighting_preset=lighting_preset,
            coat_weight=coat_weight,
        )
        written_files[f"{lvl}_glb"] = str(glb_fp)
        levels_out.append(
            {
                "level": lvl,
                "box_count": info["box_count"],
                "tri_count": rep["triangle_count"],
                "vertex_count": rep["vertex_count"],
                "joint_count": rep["joint_count"],
                "animation_count": rep["animation_count"],
                "reduction_pct": info["reduction_pct"],
                "kb": rep["kb"],
                "valid_gltf": rep["validation"]["valid"],
                "glb_path": str(glb_fp),
            }
        )

    manifest = {
        "schema": "synccraft.glb_lod_bundle/1.0",
        "ok": all(r["valid_gltf"] for r in levels_out),
        "stem": st,
        "out_dir": str(out_root),
        "meter_scale": round(float(meter_scale), 6),
        "bleed_px": int(bleed_px),
        "levels": levels_out,
        "files": written_files,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    mf_fp = out_root / f"{st}_glb_lod_manifest.json"
    mf_fp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(mf_fp)
    return manifest


# ---------------------------------------------------------------------------
# Strict Built-In Khronos glTF 2.0 Binary & Math Validator
# ---------------------------------------------------------------------------


def _read_accessor_np(doc: dict[str, Any], bin_data: bytes, acc_idx: int) -> np.ndarray:
    acc = doc["accessors"][acc_idx]
    bv = doc["bufferViews"][acc["bufferView"]]
    off = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    ct = int(acc["componentType"])
    nc = TYPE_NUM_COMPS[acc["type"]]
    cnt = int(acc["count"])
    dtype_map = {
        COMP_BYTE: "<i1",
        COMP_UBYTE: "<u1",
        COMP_SHORT: "<i2",
        COMP_USHORT: "<u2",
        COMP_UINT: "<u4",
        COMP_FLOAT: "<f4",
    }
    byte_len = cnt * nc * COMP_BYTE_SIZES[ct]
    return np.frombuffer(bin_data, dtype=dtype_map[ct], count=cnt * nc, offset=off).reshape((cnt, nc))


def validate_glb_bytes(glb_bytes: bytes) -> dict[str, Any]:
    """Strictly audit a `.glb` binary against Khronos glTF 2.0 spec & multi-engine invariants."""
    errors: list[str] = []
    warnings: list[str] = []

    if len(glb_bytes) < 20:
        return {"valid": False, "errors": ["GLB_LENGTH_TOO_SMALL"], "warnings": []}

    magic, version, total_len = struct.unpack_from("<III", glb_bytes, 0)
    if magic != GLB_MAGIC:
        errors.append(f"GLB_INVALID_MAGIC: 0x{magic:08x}")
    if version != GLB_VERSION:
        errors.append(f"GLB_INVALID_VERSION: {version}")
    if total_len != len(glb_bytes):
        errors.append(f"GLB_LENGTH_MISMATCH: header={total_len} actual={len(glb_bytes)}")
    if len(glb_bytes) % 4 != 0:
        errors.append("GLB_UNALIGNED_TOTAL_LENGTH")

    j_len, j_type = struct.unpack_from("<II", glb_bytes, 12)
    if j_type != GLB_CHUNK_JSON:
        errors.append(f"GLB_UNEXPECTED_FIRST_CHUNK: 0x{j_type:08x}")
    if j_len % 4 != 0:
        errors.append("GLB_JSON_CHUNK_UNALIGNED")
    json_slice = glb_bytes[20 : 20 + j_len]
    if b"\x00" in json_slice:
        errors.append("GLB_JSON_CONTAINS_NUL_PADDING")
    try:
        doc = json.loads(json_slice.decode("utf-8"))
    except Exception as exc:
        return {"valid": False, "errors": errors + [f"INVALID_JSON: {exc}"], "warnings": warnings}

    bin_off = 20 + j_len
    bin_data = b""
    if bin_off < len(glb_bytes):
        b_len, b_type = struct.unpack_from("<II", glb_bytes, bin_off)
        if b_type != GLB_CHUNK_BIN:
            errors.append(f"GLB_UNEXPECTED_BIN_CHUNK: 0x{b_type:08x}")
        if b_len % 4 != 0:
            errors.append("GLB_BIN_CHUNK_UNALIGNED")
        bin_data = glb_bytes[bin_off + 8 : bin_off + 8 + b_len]

    # Top-level empty array check
    for k in ("scenes", "nodes", "meshes", "materials", "textures", "images", "samplers", "skins", "animations", "extensionsUsed", "extensionsRequired"):
        if k in doc and isinstance(doc[k], list) and len(doc[k]) == 0:
            errors.append(f"EMPTY_ENTITY: {k}")

    if doc.get("asset", {}).get("version") != "2.0":
        errors.append("ASSET_VERSION_NOT_2_0")

    # Buffers check
    bufs = doc.get("buffers") or []
    if not bufs:
        errors.append("NO_BUFFERS")
    else:
        b0 = bufs[0]
        if "uri" in b0:
            errors.append("GLB_BUFFER0_HAS_URI")
        bl0 = int(b0.get("byteLength", -1))
        if not (len(bin_data) - 3 <= bl0 <= len(bin_data)):
            errors.append(f"BUFFER_GLB_CHUNK_SIZE_MISMATCH: buffer={bl0} bin_chunk={len(bin_data)}")

    # BufferViews bounds
    bvs = doc.get("bufferViews") or []
    for i, bv in enumerate(bvs):
        off = int(bv.get("byteOffset", 0))
        ln = int(bv.get("byteLength", 0))
        if off < 0 or ln <= 0 or off + ln > len(bin_data):
            errors.append(f"BUFFER_VIEW_OOB: bv[{i}]")

    # Accessors alignment & finite check
    accs = doc.get("accessors") or []
    for i, acc in enumerate(accs):
        ct = int(acc["componentType"])
        csz = COMP_BYTE_SIZES.get(ct, 0)
        bv = bvs[acc["bufferView"]]
        a_off = int(acc.get("byteOffset", 0))
        tot_off = int(bv.get("byteOffset", 0)) + a_off
        if csz == 0 or (a_off % csz != 0) or (tot_off % csz != 0):
            errors.append(f"ACCESSOR_OFFSET_ALIGNMENT: acc[{i}]")
        arr = _read_accessor_np(doc, bin_data, i)
        if ct == COMP_FLOAT and not np.isfinite(arr).all():
            errors.append(f"ACCESSOR_INVALID_FLOAT: acc[{i}]")
        if "min" in acc and "max" in acc and arr.shape[0] > 0:
            amin = arr.min(axis=0)
            amax = arr.max(axis=0)
            for c_i in range(arr.shape[1]):
                if abs(float(amin[c_i]) - float(acc["min"][c_i])) > 1e-6:
                    errors.append(f"ACCESSOR_MIN_MISMATCH: acc[{i}][{c_i}]")
                if abs(float(amax[c_i]) - float(acc["max"][c_i])) > 1e-6:
                    errors.append(f"ACCESSOR_MAX_MISMATCH: acc[{i}][{c_i}]")

    # Embedded PNGs check
    for i, im in enumerate(doc.get("images") or []):
        bv = bvs[im["bufferView"]]
        if "target" in bv or "byteStride" in bv:
            errors.append(f"IMAGE_BUFFER_VIEW_INVALID_FIELDS: img[{i}]")
        off = int(bv.get("byteOffset", 0))
        raw = bin_data[off : off + int(bv["byteLength"])]
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            errors.append(f"IMAGE_INVALID_PNG_SIGNATURE: img[{i}]")
        else:
            try:
                with Image.open(io.BytesIO(raw)) as pil_im:
                    pil_im.verify()
            except Exception as exc:
                errors.append(f"IMAGE_CORRUPT_PNG: img[{i}] {exc}")

    # Material rules
    ext_declared = set(doc.get("extensionsUsed") or [])
    ext_actually_used: set[str] = set()
    for i, mat in enumerate(doc.get("materials") or []):
        am = mat.get("alphaMode", "OPAQUE")
        if am != "MASK" and "alphaCutoff" in mat:
            errors.append(f"MATERIAL_ALPHA_CUTOFF_INVALID_MODE: mat[{i}]")
        if "emissiveTexture" in mat:
            ef = mat.get("emissiveFactor", [0, 0, 0])
            if max(ef) <= 0:
                errors.append(f"MATERIAL_EMISSIVE_FACTOR_ZERO: mat[{i}]")
        for ex in (mat.get("extensions") or {}).keys():
            ext_actually_used.add(ex)
    if ext_actually_used - ext_declared:
        errors.append(f"UNDECLARED_EXTENSION: {sorted(ext_actually_used - ext_declared)}")
    if ext_declared - ext_actually_used:
        warnings.append(f"UNUSED_EXTENSION: {sorted(ext_declared - ext_actually_used)}")

    # Mesh primitives & attributes check
    skins = doc.get("skins") or []
    joint_limit = len(skins[0]["joints"]) if skins else 0
    for mi, mesh in enumerate(doc.get("meshes") or []):
        for pi, prim in enumerate(mesh.get("primitives") or []):
            attrs = prim["attributes"]
            pos_acc = accs[attrs["POSITION"]]
            if "min" not in pos_acc or "max" not in pos_acc:
                errors.append(f"POSITION_MISSING_MIN_MAX: mesh[{mi}].prim[{pi}]")
            vcount = int(pos_acc["count"])
            for aname, aidx in attrs.items():
                if int(accs[aidx]["count"]) != vcount:
                    errors.append(f"ATTRIBUTE_COUNT_MISMATCH: {aname}")
            if "NORMAL" in attrs:
                nrms = _read_accessor_np(doc, bin_data, attrs["NORMAL"])
                n_lens = np.linalg.norm(nrms, axis=1)
                if float(np.max(np.abs(n_lens - 1.0))) > 2e-3:
                    errors.append(f"ACCESSOR_VECTOR3_NON_UNIT: NORMAL mesh[{mi}].prim[{pi}]")
            if "TANGENT" in attrs:
                tans = _read_accessor_np(doc, bin_data, attrs["TANGENT"])
                t_lens = np.linalg.norm(tans[:, :3], axis=1)
                if float(np.max(np.abs(t_lens - 1.0))) > 2e-3:
                    errors.append(f"ACCESSOR_VECTOR3_NON_UNIT: TANGENT mesh[{mi}].prim[{pi}]")
                if not np.all(np.isin(tans[:, 3], [-1.0, 1.0])):
                    errors.append(f"ACCESSOR_INVALID_SIGN: TANGENT.w mesh[{mi}].prim[{pi}]")
            if "indices" in prim:
                i_acc = accs[prim["indices"]]
                idxs = _read_accessor_np(doc, bin_data, prim["indices"]).ravel()
                if len(idxs) % 3 != 0:
                    errors.append(f"INDICES_NOT_MULTIPLE_OF_3: mesh[{mi}].prim[{pi}]")
                if len(idxs) > 0 and int(idxs.max()) >= vcount:
                    errors.append(f"ACCESSOR_INDEX_OOB: mesh[{mi}].prim[{pi}]")
                sentinel = {COMP_UBYTE: 255, COMP_USHORT: 65535, COMP_UINT: 4294967295}.get(int(i_acc["componentType"]), -1)
                if len(idxs) > 0 and int(idxs.max()) == sentinel:
                    errors.append(f"ACCESSOR_INDEX_PRIMITIVE_RESTART: mesh[{mi}].prim[{pi}]")
            if "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                j_arr = _read_accessor_np(doc, bin_data, attrs["JOINTS_0"])
                w_arr = _read_accessor_np(doc, bin_data, attrs["WEIGHTS_0"])
                if int(accs[attrs["JOINTS_0"]]["componentType"]) not in (COMP_UBYTE, COMP_USHORT):
                    errors.append("ACCESSOR_JOINTS_INVALID_COMPONENT_TYPE")
                if joint_limit > 0 and int(j_arr.max()) >= joint_limit:
                    errors.append("ACCESSOR_JOINTS_INDEX_OOB")
                if np.any(j_arr[w_arr == 0.0] != 0):
                    errors.append("ACCESSOR_JOINTS_USED_ZERO_WEIGHT")
                w_sums = w_arr.sum(axis=1)
                if float(np.max(np.abs(w_sums - 1.0))) > 1e-5:
                    errors.append("ACCESSOR_WEIGHTS_NON_NORMALIZED")

    # Skinning & Node Hierarchy Check (§3.7.3.1 + IBM @ M_global_bind == I)
    nodes = doc.get("nodes") or []
    scene_roots = set((doc.get("scenes") or [{}])[0].get("nodes") or [])
    for ni, nd in enumerate(nodes):
        if "skin" in nd:
            if ni not in scene_roots:
                errors.append(f"NODE_SKINNED_MESH_NON_ROOT: node[{ni}]")
            if any(k in nd for k in ("translation", "rotation", "scale", "matrix")):
                errors.append(f"NODE_SKINNED_MESH_LOCAL_TRANSFORMS: node[{ni}]")

    if skins:
        sk = skins[0]
        ibm_bv = bvs[accs[sk["inverseBindMatrices"]]["bufferView"]]
        if "target" in ibm_bv or "byteStride" in ibm_bv:
            errors.append("SKIN_IBM_BUFFER_VIEW_INVALID")
        ibms = _read_accessor_np(doc, bin_data, sk["inverseBindMatrices"]).reshape((-1, 4, 4))
        # Build parent map & global bind matrices from JSON TRS to prove M_global @ IBM == I_4!
        parent_of: dict[int, int] = {}
        for ni, nd in enumerate(nodes):
            for ch in nd.get("children") or []:
                parent_of[int(ch)] = ni
        glob_cache: dict[int, np.ndarray] = {}

        def _global_m(ni: int) -> np.ndarray:
            if ni in glob_cache:
                return glob_cache[ni]
            nd = nodes[ni]
            t = tuple(nd.get("translation") or (0.0, 0.0, 0.0))
            q = tuple(nd.get("rotation") or (0.0, 0.0, 0.0, 1.0))
            s = tuple(nd.get("scale") or (1.0, 1.0, 1.0))
            m_loc = _trs_mat4(t, q, s)
            m = (_global_m(parent_of[ni]) @ m_loc) if ni in parent_of else m_loc
            glob_cache[ni] = m
            return m

        for ji, j_node in enumerate(sk["joints"]):
            # Unpack column-major IBM back to standard row-major 4x4 matrix
            ibm_mat = ibms[ji].T.astype(np.float64)
            if float(np.max(np.abs(ibm_mat[3, :] - np.array([0.0, 0.0, 0.0, 1.0])))) > 1e-5:
                errors.append(f"ACCESSOR_INVALID_IBM_BOTTOM_ROW: joint[{ji}]")
            prod = _global_m(int(j_node)) @ ibm_mat
            if float(np.max(np.abs(prod - np.eye(4)))) > 1e-3:
                errors.append(f"IBM_BIND_IDENTITY_MISMATCH: joint[{ji}]")

    # Animations Check
    for ai, anim in enumerate(doc.get("animations") or []):
        if not anim.get("channels") or not anim.get("samplers"):
            errors.append(f"EMPTY_ANIMATION: anim[{ai}]")
        for ci, ch in enumerate(anim.get("channels") or []):
            smp = anim["samplers"][ch["sampler"]]
            in_acc = accs[smp["input"]]
            if "min" not in in_acc or "max" not in in_acc:
                errors.append(f"ANIM_INPUT_MISSING_MIN_MAX: anim[{ai}].ch[{ci}]")
            in_bv = bvs[in_acc["bufferView"]]
            out_bv = bvs[accs[smp["output"]]["bufferView"]]
            if "target" in in_bv or "byteStride" in in_bv or "target" in out_bv or "byteStride" in out_bv:
                errors.append(f"ANIM_SAMPLER_BUFFER_VIEW_INVALID: anim[{ai}].ch[{ci}]")
            ts = _read_accessor_np(doc, bin_data, smp["input"]).ravel()
            if len(ts) < 2:
                errors.append(f"ANIM_SINGLE_KEYFRAME: anim[{ai}].ch[{ci}]")
            if len(ts) > 0 and float(ts[0]) < 0.0:
                errors.append(f"ACCESSOR_ANIMATION_INPUT_NEGATIVE: anim[{ai}].ch[{ci}]")
            if len(ts) > 1 and np.any(np.diff(ts) <= 0.0):
                errors.append(f"ACCESSOR_ANIMATION_INPUT_NON_INCREASING: anim[{ai}].ch[{ci}]")
            if ch["target"]["path"] == "rotation":
                qs = _read_accessor_np(doc, bin_data, smp["output"])
                q_lens = np.linalg.norm(qs, axis=1)
                if float(np.max(np.abs(q_lens - 1.0))) > 2e-3:
                    errors.append(f"ACCESSOR_ANIMATION_NON_UNIT_QUAT: anim[{ai}].ch[{ci}]")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks_passed": len(errors) == 0,
    }
