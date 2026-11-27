"""Industry-Standard Multi-Pass Pipeline, First-Class `MATERIAL` Bus & Reusable Subgraphs for `synccraft.texgraph`.

Brings Substance Designer / MaterialX / Blender Shader Nodes / Pixel Composer graph architecture
to pure deterministic NumPy + Pillow (zero ML dependencies, UI-ready JSON schema):
- **First-Class `MATERIAL` Bus**: Carries all PBR channels (`albedo`, `height`, `normal`, `roughness`,
  `metallic`, `ao`, `emissive`, `sss`, `porosity`) as a single typed multi-port dict (`"node.albedo"`, etc.)
- **Height-Based Material Blending (`material_height_blend`)**: Substance Designer parity — higher relief
  physically pokes through cracks/seams cleanly across ALL PBR channels simultaneously!
- **Full-Stack Material Operations**: `material_make`, `material_blend`, `material_height_blend`,
  `material_grade`, `material_set_alpha`, `material_pad_islands`
- **Staged Multi-Pass Pipeline (`PassSpec` / `MultiPassResult` / `render_multipass_sheet`)**:
  Explicit sequential passes (`bake -> macro_blockin -> surface_sculpt -> lighting_grade -> emissive_pbr_export`)
  with per-pass checkpoint snapshots and studio progression contact sheet rendering
- **Reusable Parameterized Subgraphs (`SUBGRAPH_REGISTRY`)**: Substance `.sbsar` / Group parity
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw

from .ops_fx_color import ao_from_height, color_balance, normal_blend_rnm, oklch_adjust
from .pbr import height_to_normal_map
from .pixel_art_ops import blend_layers, rgba_float_to_u8, vibrance_grade
from .simulation_filters import morphological_despeckle

MATERIAL_BUS_PORTS: tuple[str, ...] = (
    "albedo",
    "height",
    "normal",
    "roughness",
    "metallic",
    "ao",
    "emissive",
    "sss",
    "porosity",
)


def is_material_bus(val: Any) -> bool:
    """Return True if ``val`` is a typed MATERIAL bus dict."""
    return isinstance(val, dict) and ("albedo" in val) and isinstance(val["albedo"], np.ndarray)


def _to_scalar_2d(val: np.ndarray | float | int | None, H: int, W: int, default: float) -> np.ndarray:
    if val is None:
        return np.full((H, W), float(default), dtype=np.float32)
    if isinstance(val, (int, float)):
        return np.full((H, W), float(val), dtype=np.float32)
    arr = np.asarray(val, dtype=np.float32)
    if arr.ndim == 3:
        return np.clip(0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2], 0.0, 1.0).astype(np.float32)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _to_rgba_4d(val: np.ndarray | None, H: int, W: int, *, default_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)) -> np.ndarray:
    if val is None:
        out = np.zeros((H, W, 4), dtype=np.float32)
        out[..., 0], out[..., 1], out[..., 2], out[..., 3] = default_rgba
        return out
    arr = np.asarray(val, dtype=np.float32)
    if arr.ndim == 2:
        out = np.zeros((H, W, 4), dtype=np.float32)
        out[..., 0] = arr
        out[..., 1] = arr
        out[..., 2] = arr
        out[..., 3] = 1.0
        return np.clip(out, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.concatenate((np.clip(arr, 0.0, 1.0), np.ones((H, W, 1), dtype=np.float32)), axis=-1)
    return np.clip(arr[..., :4], 0.0, 1.0).astype(np.float32)


def _to_normal_3d(val: np.ndarray | None, height: np.ndarray, *, normal_strength: float = 2.4, tileable: bool = False) -> np.ndarray:
    if val is None:
        return height_to_normal_map(height, strength=float(normal_strength), tileable=tileable)
    arr = np.asarray(val, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.clip(arr[..., :3], 0.0, 1.0).astype(np.float32)
    return height_to_normal_map(height, strength=float(normal_strength), tileable=tileable)


def material_make(
    albedo: np.ndarray,
    *,
    height: np.ndarray | float | None = None,
    normal: np.ndarray | None = None,
    roughness: np.ndarray | float | None = 0.55,
    metallic: np.ndarray | float | None = 0.0,
    ao: np.ndarray | float | None = 1.0,
    emissive: np.ndarray | None = None,
    sss: np.ndarray | float | None = 0.0,
    porosity: np.ndarray | float | None = 0.0,
    normal_strength: float = 2.4,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Pack Albedo + PBR channels into a unified Substance/MaterialX-style `MATERIAL` bus dict."""
    alb = np.asarray(albedo, dtype=np.float32)
    H, W = alb.shape[:2]
    alb4 = _to_rgba_4d(alb, H, W)
    h_map = _to_scalar_2d(height, H, W, 0.5)
    n_map = _to_normal_3d(normal, h_map, normal_strength=normal_strength, tileable=tileable)
    r_map = _to_scalar_2d(roughness, H, W, 0.55)
    m_map = _to_scalar_2d(metallic, H, W, 0.0)
    ao_map = _to_scalar_2d(ao, H, W, 1.0)
    em_map = _to_rgba_4d(emissive, H, W, default_rgba=(0.0, 0.0, 0.0, 0.0))
    sss_map = _to_scalar_2d(sss, H, W, 0.0)
    por_map = _to_scalar_2d(porosity, H, W, 0.0)
    return {
        "__bus__": np.array([1.0], dtype=np.float32),  # marker stripped during export/inspect
        "albedo": alb4,
        "height": h_map,
        "normal": n_map,
        "roughness": r_map,
        "metallic": m_map,
        "ao": ao_map,
        "emissive": em_map,
        "sss": sss_map,
        "porosity": por_map,
    }


def _ensure_bus(val: dict[str, np.ndarray] | np.ndarray, *, tileable: bool = False) -> dict[str, np.ndarray]:
    if is_material_bus(val):
        alb = val["albedo"]
        H, W = alb.shape[:2]
        return material_make(
            alb,
            height=val.get("height"),
            normal=val.get("normal"),
            roughness=val.get("roughness", 0.55),
            metallic=val.get("metallic", 0.0),
            ao=val.get("ao", 1.0),
            emissive=val.get("emissive"),
            sss=val.get("sss", 0.0),
            porosity=val.get("porosity", 0.0),
            tileable=tileable,
        )
    return material_make(np.asarray(val, dtype=np.float32), tileable=tileable)


def material_blend(
    base: dict[str, np.ndarray],
    top: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    opacity: float = 1.0,
    albedo_mode: str = "normal",
    normal_mode: str = "lerp",
    preserve_base_alpha: bool = False,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Composite two full PBR `MATERIAL` buses simultaneously across every channel."""
    b = _ensure_bus(base, tileable=tileable)
    t = _ensure_bus(top, tileable=tileable)
    H, W = b["albedo"].shape[:2]
    m = np.ones((H, W), dtype=np.float32) if mask is None else _to_scalar_2d(mask, H, W, 1.0)
    w = np.clip(m * float(opacity), 0.0, 1.0).astype(np.float32)
    w3 = w[..., None]
    w4 = w[..., None]

    if albedo_mode == "normal":
        alb_rgb = b["albedo"][..., :3] * (1.0 - w3) + t["albedo"][..., :3] * w3
        alb_a = b["albedo"][..., 3:4] if preserve_base_alpha else (b["albedo"][..., 3:4] * (1.0 - w4) + t["albedo"][..., 3:4] * w4)
        alb = np.clip(np.concatenate((alb_rgb, alb_a), axis=-1), 0.0, 1.0).astype(np.float32)
    else:
        alb = blend_layers(b["albedo"], t["albedo"], mode=albedo_mode, opacity=float(opacity), mask=m)
        if preserve_base_alpha:
            alb[..., 3] = b["albedo"][..., 3]

    height = np.clip(b["height"] * (1.0 - w) + t["height"] * w, 0.0, 1.0).astype(np.float32)
    if normal_mode == "rnm":
        rnm = normal_blend_rnm(b["normal"], t["normal"], strength=1.0)
        normal = np.clip(b["normal"] * (1.0 - w3) + rnm * w3, 0.0, 1.0).astype(np.float32)
    else:
        # Linear tangent-space normal blend + renormalize
        nb = b["normal"] * 2.0 - 1.0
        nt = t["normal"] * 2.0 - 1.0
        nm = nb * (1.0 - w3) + nt * w3
        nlen = np.maximum(1e-6, np.linalg.norm(nm, axis=-1, keepdims=True))
        normal = np.clip((nm / nlen) * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)

    roughness = np.clip(b["roughness"] * (1.0 - w) + t["roughness"] * w, 0.0, 1.0).astype(np.float32)
    metallic = np.clip(b["metallic"] * (1.0 - w) + t["metallic"] * w, 0.0, 1.0).astype(np.float32)
    ao = np.clip(b["ao"] * (1.0 - w) + t["ao"] * w, 0.0, 1.0).astype(np.float32)
    sss = np.clip(b["sss"] * (1.0 - w) + t["sss"] * w, 0.0, 1.0).astype(np.float32)
    porosity = np.clip(b["porosity"] * (1.0 - w) + t["porosity"] * w, 0.0, 1.0).astype(np.float32)
    em_rgb = b["emissive"][..., :3] * (1.0 - w3) + t["emissive"][..., :3] * w3
    em_a = b["emissive"][..., 3:4] * (1.0 - w4) + t["emissive"][..., 3:4] * w4
    emissive = np.clip(np.concatenate((em_rgb, em_a), axis=-1), 0.0, 1.0).astype(np.float32)

    out = material_make(
        alb,
        height=height,
        normal=normal,
        roughness=roughness,
        metallic=metallic,
        ao=ao,
        emissive=emissive,
        sss=sss,
        porosity=porosity,
        tileable=tileable,
    )
    out["blend_mask"] = w
    return out


def material_height_blend(
    base: dict[str, np.ndarray],
    top: dict[str, np.ndarray],
    *,
    mask: np.ndarray | None = None,
    height_offset: float = 0.0,
    contrast: float = 6.0,
    seam_ao_strength: float = 0.25,
    normal_mode: str = "lerp",
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Substance Designer Height-Blend Material Composite!

    Blends ``top`` over ``base`` using physical height competition modulated by ``mask``:
    - Higher relief in ``top`` emerges first through cracks & mortar seams of ``base``
    - ``contrast`` controls transition crispness (1.5 = soft painterly blend, 10.0 = razor crisp step)
    - Automatically injects subtle hue-shifted contact AO along the physical material seam boundary
    - Emits all standard `MATERIAL` bus channels PLUS `"blend_mask"` and `"seam_mask"` ports!
    """
    b = _ensure_bus(base, tileable=tileable)
    t = _ensure_bus(top, tileable=tileable)
    H, W = b["albedo"].shape[:2]
    m = np.full((H, W), 0.5, dtype=np.float32) if mask is None else _to_scalar_2d(mask, H, W, 0.5)

    # Classic Substance height-lerp formulation:
    # top_score = top_height + height_offset + (m - 0.5) * 1.4
    # base_score = base_height - (m - 0.5) * 0.6
    # Guarantee mask=0 -> pure base and mask=1 -> pure top at extremes!
    h_top = np.clip(t["height"] + float(height_offset), -0.5, 1.5)
    h_base = b["height"]
    diff = (h_top - h_base) + (m * 2.0 - 1.0) * 0.95
    c = max(0.5, float(contrast))
    raw_w = np.clip(0.5 + diff * (0.5 * c), 0.0, 1.0)
    # Smoothstep for silky anti-aliased transitions without ringing
    w = (raw_w * raw_w * (3.0 - 2.0 * raw_w)).astype(np.float32)
    # Respect exact hard 0.0 / 1.0 boundaries on binary element masks
    w = np.where(m <= 0.002, 0.0, np.where(m >= 0.998, 1.0, w)).astype(np.float32)

    # Physical contact seam band right where materials meet (w in (0.05..0.95))
    seam = np.clip(4.0 * w * (1.0 - w), 0.0, 1.0).astype(np.float32)

    out = material_blend(b, t, mask=w, opacity=1.0, normal_mode=normal_mode, tileable=tileable)
    # Composite height takes maximum relief along seam so raised moss/filigree sits physically proud!
    out["height"] = np.clip(np.maximum(b["height"] * (1.0 - w * 0.35), t["height"] * (0.65 + 0.35 * w)) * w + b["height"] * (1.0 - w), 0.0, 1.0).astype(np.float32)

    s_ao = float(seam_ao_strength)
    if s_ao > 1e-4:
        ao_drop = 1.0 - seam * np.clip(s_ao, 0.0, 0.75)
        out["ao"] = np.clip(out["ao"] * ao_drop, 0.05, 1.0).astype(np.float32)
        # Cool subtle shadow darkening in albedo right along the contact seam
        out["albedo"][..., 0] = np.clip(out["albedo"][..., 0] * (1.0 - seam * s_ao * 0.55), 0.0, 1.0)
        out["albedo"][..., 1] = np.clip(out["albedo"][..., 1] * (1.0 - seam * s_ao * 0.50), 0.0, 1.0)
        out["albedo"][..., 2] = np.clip(out["albedo"][..., 2] * (1.0 - seam * s_ao * 0.32), 0.0, 1.0)

    out["blend_mask"] = w
    out["seam_mask"] = seam
    return out


def material_grade(
    mat: dict[str, np.ndarray],
    *,
    ao_bake: np.ndarray | None = None,
    ao_darken: float = 0.0,
    shadows: tuple[float, float, float] = (0.0, 0.0, 0.0),
    midtones: tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlights: tuple[float, float, float] = (0.0, 0.0, 0.0),
    hue_shift_deg: float = 0.0,
    chroma_scale: float = 1.0,
    lightness_offset: float = 0.0,
    vibrance: float = 0.0,
    contrast: float = 1.0,
    lift: float = 0.0,
    shadow_cool: float = 0.0,
    despeckle_threshold: float = 0.0,
    detail_normal: np.ndarray | None = None,
    detail_normal_strength: float = 0.5,
    world_normal: np.ndarray | None = None,
    world_pos_norm: np.ndarray | None = None,
    atmosphere_strength: float = 0.0,
    sky_tint: tuple[float, float, float] = (0.14, 0.24, 0.42),
    bounce_tint: tuple[float, float, float] = (0.12, 0.08, 0.24),
    organic_wash_strength: float = 0.0,
    seed: int = 42,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Pass-4 Unified World-Space Atmosphere, Contact AO, Color Balance, Vibrance & RNM Detail Normal Grading."""
    m = _ensure_bus(mat, tileable=tileable)
    alb = m["albedo"].copy()
    ao = m["ao"].copy()
    nrm = m["normal"].copy()

    # Unified World-Space Atmospheric Light Wrap & Organic Pigment Unity (harmonizes hide, gold, claws & wings!)
    atm = float(np.clip(atmosphere_strength, 0.0, 0.6))
    if atm > 1e-4 and world_normal is not None:
        wn = np.asarray(world_normal, dtype=np.float32)
        sky_w = np.clip(wn[..., 1] * 0.5 + 0.5, 0.0, 1.0)
        bnc_w = np.clip(-wn[..., 1] * 0.5 + 0.5, 0.0, 1.0)
        if world_pos_norm is not None:
            wpn = np.asarray(world_pos_norm, dtype=np.float32)
            elev = np.clip(wpn[..., 1], 0.0, 1.0)
            sky_w = np.clip(sky_w * (0.65 + 0.35 * elev), 0.0, 1.0)
            bnc_w = np.clip(bnc_w * (1.0 - 0.35 * elev), 0.0, 1.0)
        st = np.array(sky_tint, dtype=np.float32)[None, None, :]
        bt = np.array(bounce_tint, dtype=np.float32)[None, None, :]
        # Soft atmospheric additive/screen sky zenith lift + velvety under-bounce tint
        alb[..., :3] = np.clip(
            alb[..., :3] * (1.0 - 0.16 * atm * bnc_w[..., None] + 0.14 * atm * sky_w[..., None])
            + atm * 0.38 * sky_w[..., None] * st * (1.0 - alb[..., :3] * 0.55)
            + atm * 0.32 * bnc_w[..., None] * bt * (1.0 - alb[..., :3] * 0.65),
            0.0,
            1.0,
        )

    ow = float(np.clip(organic_wash_strength, 0.0, 0.35))
    if ow > 1e-4 and world_pos_norm is not None:
        from .generators_noise import noise_3d_field
        wash_3d = noise_3d_field(np.asarray(world_pos_norm, dtype=np.float32), scale=2.8, octaves=2, mode="fbm", seed=int(seed) + 19)
        alb[..., :3] = np.clip(alb[..., :3] * (1.0 + (wash_3d - 0.5)[..., None] * ow * 1.4), 0.0, 1.0)

    if ao_bake is not None:
        ab = _to_scalar_2d(ao_bake, alb.shape[0], alb.shape[1], 1.0)
        ao = np.clip(ao * ab, 0.08, 1.0).astype(np.float32)
        if float(ao_darken) > 1e-4:
            inv_ao = (1.0 - ab) * float(ao_darken)
            # Hue-shifted cool contact shadow (shadows lose warmth, preserve rich blue/indigo)
            alb[..., 0] = np.clip(alb[..., 0] * (1.0 - inv_ao * 1.05), 0.0, 1.0)
            alb[..., 1] = np.clip(alb[..., 1] * (1.0 - inv_ao * 0.96), 0.0, 1.0)
            alb[..., 2] = np.clip(alb[..., 2] * (1.0 - inv_ao * 0.72), 0.0, 1.0)

    if any(abs(c) > 1e-5 for c in (*shadows, *midtones, *highlights)):
        alb = color_balance(alb, shadows=shadows, midtones=midtones, highlights=highlights)

    if abs(float(hue_shift_deg)) > 1e-4 or abs(float(chroma_scale) - 1.0) > 1e-4 or abs(float(lightness_offset)) > 1e-4:
        alb = oklch_adjust(alb, hue_shift_deg=hue_shift_deg, chroma_scale=chroma_scale, lightness_offset=lightness_offset)

    if abs(float(vibrance)) > 1e-4 or abs(float(contrast) - 1.0) > 1e-4 or abs(float(lift)) > 1e-4 or abs(float(shadow_cool)) > 1e-4:
        alb = vibrance_grade(alb, vibrance=vibrance, contrast=contrast, lift=lift, shadow_cool=shadow_cool)

    if float(despeckle_threshold) > 1e-4:
        alb = morphological_despeckle(alb, threshold=float(despeckle_threshold), tileable=tileable)

    if detail_normal is not None:
        nrm = normal_blend_rnm(nrm, detail_normal, strength=float(detail_normal_strength))

    return material_make(
        alb,
        height=m["height"],
        normal=nrm,
        roughness=m["roughness"],
        metallic=m["metallic"],
        ao=ao,
        emissive=m["emissive"],
        sss=m["sss"],
        porosity=m["porosity"],
        tileable=tileable,
    )


def material_set_alpha(
    mat: dict[str, np.ndarray],
    alpha: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    tileable: bool = False,
) -> dict[str, np.ndarray]:
    """Apply hard or soft alpha cutout cleanly to BOTH albedo and emissive inside a `MATERIAL` bus."""
    m = _ensure_bus(mat, tileable=tileable)
    H, W = m["albedo"].shape[:2]
    a = _to_scalar_2d(alpha, H, W, 1.0)
    alb = m["albedo"].copy()
    em = m["emissive"].copy()
    if mask is not None:
        mm = _to_scalar_2d(mask, H, W, 1.0)
        alb[..., 3] = alb[..., 3] * (1.0 - mm) + a * mm
        em[..., 3] = em[..., 3] * (1.0 - mm) + np.minimum(em[..., 3], a) * mm
    else:
        alb[..., 3] = a
        em[..., 3] = np.minimum(em[..., 3], a)
    return material_make(
        alb,
        height=m["height"],
        normal=m["normal"],
        roughness=m["roughness"],
        metallic=m["metallic"],
        ao=m["ao"],
        emissive=em,
        sss=m["sss"],
        porosity=m["porosity"],
        tileable=tileable,
    )


def material_pad_islands(mat: dict[str, np.ndarray], mesh_context: Any, *, pad: int | None = None, tileable: bool = False) -> dict[str, np.ndarray]:
    """Clamp-bleed EVERY channel of a `MATERIAL` bus cleanly into UV island padding."""
    m = _ensure_bus(mat, tileable=tileable)
    if mesh_context is None:
        return m
    out = dict(m)
    for k in MATERIAL_BUS_PORTS:
        if k in out and isinstance(out[k], np.ndarray):
            out[k] = mesh_context.pad_rgba(out[k], pad=pad)
    return out


# ---------------------------------------------------------------------------
# Staged Multi-Pass Pipeline Structures & Progression Contact Sheet Renderer
# ---------------------------------------------------------------------------


@dataclass
class PassSpec:
    """One named sequential stage in a Multi-Pass TexGraph pipeline."""

    name: str
    title: str
    description: str = ""
    node_ids: list[str] = field(default_factory=list)
    preview_ref: str | None = None
    color: str = "#1E293B"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "nodes": list(self.node_ids),
            "preview": self.preview_ref,
            "color": self.color,
        }


@dataclass
class PassSnapshot:
    name: str
    title: str
    description: str
    preview_ref: str | None
    preview_rgba: np.ndarray | None
    node_count: int
    time_ms: float


@dataclass
class MultiPassResult:
    eval_result: Any  # EvalResult
    passes: list[PassSnapshot]
    total_ms: float


def render_multipass_sheet(
    mp_result: MultiPassResult,
    *,
    title: str = "SyncCraft TexGraph | Multi-Pass Pipeline Progression",
    tile_size: int = 256,
) -> Image.Image:
    """Render a crisp studio Multi-Pass Progression Contact Sheet (Pass 1 -> Pass N -> Final GGX PBR Lit)."""
    panels: list[tuple[str, str, np.ndarray]] = []
    for idx, ps in enumerate(mp_result.passes, start=1):
        if ps.preview_rgba is not None:
            panels.append((
                f"Pass {idx}: {ps.title}",
                f"{ps.node_count} nodes | {ps.time_ms:.1f} ms",
                rgba_float_to_u8(ps.preview_rgba),
            ))
    ev = mp_result.eval_result
    if ev.pbr_preview is not None:
        panels.append((
            "Final: GGX PBR Lit",
            f"Total: {mp_result.total_ms:.1f} ms",
            rgba_float_to_u8(ev.pbr_preview),
        ))
    if not panels:
        panels.append(("Output", "", rgba_float_to_u8(ev.output)))

    n = len(panels)
    cols = min(n, 3 if n > 4 else n)
    rows = (n + cols - 1) // cols
    pad = 14
    hdr_h = 46
    lbl_h = 34
    cell_w = tile_size
    cell_h = tile_size + lbl_h
    W = cols * cell_w + (cols + 1) * pad
    H = hdr_h + rows * cell_h + (rows + 1) * pad

    canvas = Image.new("RGBA", (W, H), (11, 15, 23, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, W, hdr_h], fill=(16, 21, 33, 255))
    draw.text((pad, 14), title, fill=(232, 239, 250, 255))
    draw.text((W - 220, 15), f"{n} Stages | Pure Algorithmic", fill=(34, 211, 238, 255))

    # Subtle dark checkerboard background for alpha cutout panels
    chk = Image.new("RGBA", (tile_size, tile_size), (22, 27, 38, 255))
    cd = ImageDraw.Draw(chk)
    sq = 16
    for yy in range(0, tile_size, sq):
        for xx in range(0, tile_size, sq):
            if ((xx // sq) + (yy // sq)) % 2 == 0:
                cd.rectangle([xx, yy, xx + sq - 1, yy + sq - 1], fill=(31, 36, 51, 255))

    for i, (p_title, p_sub, u8) in enumerate(panels):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = hdr_h + pad + r * (cell_h + pad)
        draw.rounded_rectangle([x0 - 2, y0 - 2, x0 + cell_w + 2, y0 + cell_h + 2], radius=8, fill=(20, 26, 38, 255), outline=(38, 49, 71, 255))

        im = Image.fromarray(u8, mode="RGBA")
        # Fit aspect-preserving inside tile_size x tile_size with NEAREST crisp pixels
        iw, ih = im.size
        sc = min(tile_size / float(max(1, iw)), tile_size / float(max(1, ih)))
        nw, nh = max(1, int(round(iw * sc))), max(1, int(round(ih * sc)))
        im_r = im.resize((nw, nh), resample=Image.Resampling.NEAREST)
        tile_bg = chk.copy()
        tile_bg.alpha_composite(im_r, dest=((tile_size - nw) // 2, (tile_size - nh) // 2))
        canvas.paste(tile_bg, (x0, y0))

        draw.text((x0 + 6, y0 + tile_size + 4), p_title[:34], fill=(232, 239, 250, 255))
        draw.text((x0 + 6, y0 + tile_size + 18), p_sub[:34], fill=(141, 160, 189, 255))

    return canvas


# ---------------------------------------------------------------------------
# Reusable Parameterized Material Subgraphs (Substance .sbsar / Group Parity)
# ---------------------------------------------------------------------------

SubgraphBuilder = Callable[..., str]
SUBGRAPH_REGISTRY: dict[str, dict[str, Any]] = {}


def register_subgraph(
    name: str,
    *,
    category: str = "Studio PBR Materials",
    description: str = "",
    defaults: dict[str, Any] | None = None,
) -> Callable[[SubgraphBuilder], SubgraphBuilder]:
    """Register a reusable parameterized TexGraph Material Subgraph."""
    def _dec(fn: SubgraphBuilder) -> SubgraphBuilder:
        SUBGRAPH_REGISTRY[name] = {
            "name": name,
            "category": category,
            "description": description or (fn.__doc__ or "").strip().splitlines()[0],
            "defaults": dict(defaults or {}),
            "builder": fn,
        }
        return fn
    return _dec


@register_subgraph(
    "dragon_armor_plates_pbr",
    description="Calm chunky overlapping dragon armor plates + studio brushwork + OKLab gradient + LabPBR MaterialBus.",
    defaults={
        "cols": 24,
        "rows": 28,
        "stops": [
            (0.00, "#0D0B24"),
            (0.24, "#1C1B4B"),
            (0.50, "#2B3A7A"),
            (0.76, "#4670B8"),
            (0.92, "#7CB8EB"),
            (1.00, "#D1F0FF"),
        ],
        "roughness": 0.30,
        "metallic": 0.16,
        "smooth_blend": 0.66,
        "quantize_steps": 8,
        "normal_strength": 2.4,
    },
)
def _sg_dragon_armor_plates(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 42,
    cols: int = 24,
    rows: int = 28,
    stops: list[tuple[float, str]] | None = None,
    roughness: float = 0.30,
    metallic: float = 0.16,
    smooth_blend: float = 0.66,
    quantize_steps: int = 8,
    normal_strength: float = 2.4,
) -> str:
    pal = stops or SUBGRAPH_REGISTRY["dragon_armor_plates_pbr"]["defaults"]["stops"]
    flow_lit = inputs.get("flow_lit") or g.body_flow_gradient("top_light_wrap", gamma=0.95, id=f"{prefix}_flow")
    isl_grad = inputs.get("isl_grad") or g.per_island_gradient("top_to_bottom", curve=0.88, id=f"{prefix}_isl")
    plates = g.layered_plates(cols=cols, rows=rows, stagger=0.5, gap=0.07, top_rim=0.20, variance=0.16, seed=seed, id=f"{prefix}_plates")
    brush = g.brushstrokes(angle_deg=-18.0, scale_along=4.0, scale_across=18.0, contrast=0.20, seed=seed + 17, id=f"{prefix}_brush")
    n3 = g.noise_3d(scale=0.09, octaves=2, mode="fbm", seed=seed + 3, id=f"{prefix}_n3")
    h1 = g.blend(plates, brush, mode="soft_light", opacity=0.30, id=f"{prefix}_h1")
    h_out = g.blend(h1, n3, mode="soft_light", opacity=0.18, id=f"{prefix}_height")
    sh3d = g.shade(h_out, use_mesh=(g.mesh_context is not None), bump_strength=1.25, ao_strength=0.26, bevel_strength=0.32, cavity_strength=0.22, rim_strength=0.28, spec_strength=0.24, id=f"{prefix}_sh3d")
    sh_sm = g.blend(g.blend(sh3d, flow_lit, opacity=0.34, id=f"{prefix}_sh_fl"), isl_grad, mode="soft_light", opacity=0.26, id=f"{prefix}_shade")
    alb = g.multi_stop_gradient(sh_sm, stops=pal, smooth_blend=smooth_blend, quantize_steps=quantize_steps, dither="clustered", dither_strength=0.06, id=f"{prefix}_albedo")
    ao_in = inputs.get("ao") or (g.mesh_channel("baked_ao", id=f"{prefix}_ao") if g.mesh_context else g.ao_from_height(h_out, radius=3, strength=0.85, id=f"{prefix}_ao"))
    return g.material_make(alb, height=h_out, roughness=roughness, metallic=metallic, ao=ao_in, normal_strength=normal_strength, id=f"{prefix}_mat")


@register_subgraph(
    "brushed_metal_pbr",
    description="Anisotropic brushed luxury champagne-gold / chrome metal PBR MaterialBus (LabPBR Metal ID ready).",
    defaults={
        "stops": [
            (0.00, "#5C2C14"),
            (0.28, "#A8611B"),
            (0.58, "#E3A735"),
            (0.84, "#FAD96B"),
            (1.00, "#FFF9D6"),
        ],
        "angle_deg": 18.0,
        "roughness": 0.17,
        "metallic": 0.94,
        "normal_strength": 2.2,
    },
)
def _sg_brushed_metal(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 42,
    stops: list[tuple[float, str]] | None = None,
    angle_deg: float = 18.0,
    roughness: float = 0.17,
    metallic: float = 0.94,
    normal_strength: float = 2.2,
) -> str:
    pal = stops or SUBGRAPH_REGISTRY["brushed_metal_pbr"]["defaults"]["stops"]
    flow_lit = inputs.get("flow_lit") or g.body_flow_gradient("top_light_wrap", gamma=0.95, id=f"{prefix}_flow")
    isl_grad = inputs.get("isl_grad") or g.per_island_gradient("top_to_bottom", curve=0.90, id=f"{prefix}_isl")
    grain = g.gabor(scale=24.0, angle_deg=angle_deg, anisotropy=4.2, octaves=2, seed=seed, id=f"{prefix}_grain")
    h_out = g.blend(grain, isl_grad, mode="soft_light", opacity=0.30, id=f"{prefix}_height")
    sh3d = g.shade(h_out, use_mesh=(g.mesh_context is not None), bump_strength=1.0, ao_strength=0.26, bevel_strength=0.38, cavity_strength=0.20, spec_strength=0.44, spec_power=16.0, id=f"{prefix}_sh3d")
    sh_sm = g.blend(sh3d, flow_lit, opacity=0.28, id=f"{prefix}_shade")
    alb = g.multi_stop_gradient(sh_sm, stops=pal, smooth_blend=0.72, quantize_steps=7, id=f"{prefix}_albedo")
    ao_in = inputs.get("ao") or (g.mesh_channel("baked_ao", id=f"{prefix}_ao") if g.mesh_context else g.ao_from_height(h_out, radius=3, strength=0.80, id=f"{prefix}_ao"))
    return g.material_make(alb, height=h_out, roughness=roughness, metallic=metallic, ao=ao_in, normal_strength=normal_strength, id=f"{prefix}_mat")


@register_subgraph(
    "pearl_scutes_pbr",
    description="Silky opalescent ribbed ventral scutes with soft SSS PBR MaterialBus.",
    defaults={
        "stops": [(0.0, "#4B5578"), (0.35, "#8E9EC2"), (0.72, "#CEDBF0"), (1.0, "#F5F9FF")],
        "roughness": 0.36,
        "sss": 0.24,
    },
)
def _sg_pearl_scutes(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 42,
    stops: list[tuple[float, str]] | None = None,
    roughness: float = 0.36,
    sss: float = 0.24,
) -> str:
    pal = stops or SUBGRAPH_REGISTRY["pearl_scutes_pbr"]["defaults"]["stops"]
    isl_grad = inputs.get("isl_grad") or g.per_island_gradient("top_to_bottom", curve=0.90, id=f"{prefix}_isl")
    ribs = g.layered_plates(cols=12, rows=44, stagger=0.0, gap=0.09, top_rim=0.22, variance=0.08, seed=seed, id=f"{prefix}_ribs")
    sh3d = g.shade(ribs, use_mesh=(g.mesh_context is not None), bump_strength=1.15, ao_strength=0.28, bevel_strength=0.22, cavity_strength=0.24, id=f"{prefix}_sh3d")
    sh = g.blend(sh3d, isl_grad, mode="soft_light", opacity=0.25, id=f"{prefix}_shade")
    alb = g.multi_stop_gradient(sh, stops=pal, smooth_blend=0.75, quantize_steps=7, id=f"{prefix}_albedo")
    ao_in = inputs.get("ao") or (g.mesh_channel("baked_ao", id=f"{prefix}_ao") if g.mesh_context else g.ao_from_height(ribs, radius=3, strength=0.80, id=f"{prefix}_ao"))
    return g.material_make(alb, height=ribs, roughness=roughness, metallic=0.04, ao=ao_in, sss=sss, normal_strength=2.2, id=f"{prefix}_mat")


@register_subgraph(
    "sculpted_pinion_wing_pbr",
    description="Aerodynamic metallic obsidian pinion-blade wing feathers + gold astrolabe arc + Twin-Spectrum neon quills (Full MaterialBus).",
    defaults={
        "vanes": 5,
        "sweep": 0.84,
        "blade_stops": [
            (0.00, "#0A091E"),
            (0.24, "#171A45"),
            (0.52, "#283D7E"),
            (0.78, "#4D7EC7"),
            (0.93, "#99CEFA"),
            (1.00, "#E3F6FF"),
        ],
        "gold_stops": [(0.0, "#753E14"), (0.45, "#D99527"), (0.82, "#F9DA70"), (1.0, "#FFF9D9")],
        "neon_stops": [(0.0, "#0284C7"), (0.45, "#22D3EE"), (0.82, "#A5F3FC"), (1.0, "#FFFFFF")],
        "smooth_blend": 0.70,
        "quantize_steps": 8,
    },
)
def _sg_sculpted_pinion_wing(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 42,
    vanes: int = 5,
    sweep: float = 0.84,
    blade_stops: list[tuple[float, str]] | None = None,
    gold_stops: list[tuple[float, str]] | None = None,
    neon_stops: list[tuple[float, str]] | None = None,
    smooth_blend: float = 0.70,
    quantize_steps: int = 8,
    envelope_style: str = "primary",
    gold_arc: bool = True,
    quill_glow_strength: float = 0.92,
) -> str:
    dfl = SUBGRAPH_REGISTRY["sculpted_pinion_wing_pbr"]["defaults"]
    span_u = inputs.get("span_u") or (g.mesh_channel("wing_span_u", id=f"{prefix}_su") if g.mesh_context else g.gradient("horizontal", id=f"{prefix}_su"))
    trail_v = inputs.get("trail_v") or (g.mesh_channel("wing_trail_v", id=f"{prefix}_tv") if g.mesh_context else g.gradient("vertical", id=f"{prefix}_tv"))
    is_down = inputs.get("face_is_down") or (g.mesh_channel("face_is_down", id=f"{prefix}_dn") if g.mesh_context else g.constant(0.0, id=f"{prefix}_dn"))
    ins = {"span_u": span_u, "trail_v": trail_v, "face_is_down": is_down}
    if "wave" in inputs and inputs["wave"]:
        ins["wave"] = inputs["wave"]
    return g.add(
        "sculpted_pinion_wing",
        id=f"{prefix}_mat",
        inputs=ins,
        vanes=vanes,
        sweep=sweep,
        blade_stops=blade_stops or dfl["blade_stops"],
        gold_stops=gold_stops or dfl["gold_stops"],
        neon_stops=neon_stops or dfl["neon_stops"],
        smooth_blend=smooth_blend,
        quantize_steps=quantize_steps,
        envelope_style=envelope_style,
        gold_arc=gold_arc,
        quill_glow_strength=quill_glow_strength,
        seed=seed,
    )


@register_subgraph(
    "carved_ruin_stone_pbr",
    description="Sunlit stepped Mayan/ruin carved ashlar sandstone masonry MaterialBus.",
    defaults={"cols": 2, "rows": 4, "carve_motif": True},
)
def _sg_carved_ruin_stone(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 108,
    cols: int = 2,
    rows: int = 4,
    carve_motif: bool = True,
) -> str:
    st_h = g.temple_stone(cols=cols, rows=rows, carve_motif=carve_motif, output="height", seed=seed, id=f"{prefix}_h")
    sun = g.gradient_field("linear", angle_deg=-90.0, ease="smoothstep", id=f"{prefix}_sun")
    sh = g.blend(g.shade(st_h, use_mesh=False, bump_strength=1.75, spec_strength=0.12, id=f"{prefix}_sh"), sun, mode="soft_light", opacity=0.28, id=f"{prefix}_lit")
    alb = g.multi_stop_gradient(
        sh,
        stops=[(0.00, "#3D4754"), (0.26, "#697173"), (0.54, "#A29E8B"), (0.80, "#CDC4A4"), (1.00, "#ECE5C8")],
        smooth_blend=0.65,
        quantize_steps=7,
        id=f"{prefix}_alb",
    )
    ao = g.ao_from_height(st_h, radius=3, strength=0.90, id=f"{prefix}_ao")
    return g.material_make(alb, height=st_h, roughness=0.68, metallic=0.0, ao=ao, porosity=0.45, normal_strength=2.8, id=f"{prefix}_mat")


@register_subgraph(
    "emerald_moss_pbr",
    description="Lush Turing reaction-diffusion emerald jungle moss MaterialBus with high roughness & porosity.",
    defaults={"steps": 200},
)
def _sg_emerald_moss(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 137,
    steps: int = 200,
) -> str:
    turing = g.gray_scott(preset="coral", steps=steps, sim_scale=1, seed=seed, id=f"{prefix}_turing")
    sh = g.shade(turing, use_mesh=False, bump_strength=1.5, spec_strength=0.08, id=f"{prefix}_sh")
    alb = g.multi_stop_gradient(
        sh,
        stops=[(0.0, "#14532D"), (0.45, "#22C55E"), (0.82, "#84CC16"), (1.0, "#D9F99D")],
        smooth_blend=0.60,
        quantize_steps=5,
        id=f"{prefix}_alb",
    )
    h_lift = g.levels(turing, out_min=0.58, out_max=0.98, id=f"{prefix}_h")
    ao = g.ao_from_height(h_lift, radius=3, strength=0.85, id=f"{prefix}_ao")
    return g.material_make(alb, height=h_lift, roughness=0.88, metallic=0.0, ao=ao, sss=0.35, porosity=0.85, normal_strength=2.6, id=f"{prefix}_mat")


@register_subgraph(
    "anatomical_muscle_scales_pbr",
    description="3D-anatomical scale-to-muscle texture fitting MaterialBus (directional muscle flow + osteoderm crowns + belly countershading).",
    defaults={
        "plate_cols": 24,
        "plate_rows": 28,
        "muscle_groove_depth": 0.42,
        "scale_relief": 0.36,
        "roughness_base": 0.40,
        "metallic_crown": 0.14,
        "smooth_blend": 0.62,
        "quantize_steps": 7,
    },
)
def _sg_anatomical_muscle_scales(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 42,
    **kw: Any,
) -> str:
    return g.add("anatomical_muscle_scales", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph(
    "colossal_dragon_membrane_pbr",
    description="Monumental full-chord dragon wing membrane MaterialBus (curved tendon ribs + diagonal pleats + tattered arch + backlit SSS).",
    defaults={
        "arch_depth": 0.26,
        "arches": 1,
        "tatter_strength": 0.09,
        "tendon_ribs": 3,
        "pleat_count": 9.0,
        "pleat_strength": 0.18,
        "backlit_sss": 0.85,
        "smooth_blend": 0.74,
        "quantize_steps": 8,
    },
)
def _sg_colossal_dragon_membrane(
    g: Any,
    *,
    prefix: str,
    inputs: dict[str, str],
    seed: int = 42,
    **kw: Any,
) -> str:
    span_u = inputs.get("span_u") or (g.mesh_channel("wing_span_u", id=f"{prefix}_su") if g.mesh_context else g.gradient("horizontal", id=f"{prefix}_su"))
    trail_v = inputs.get("trail_v") or (g.mesh_channel("wing_trail_v", id=f"{prefix}_tv") if g.mesh_context else g.gradient("vertical", id=f"{prefix}_tv"))
    is_dn = inputs.get("face_is_down") or (g.mesh_channel("face_is_down", id=f"{prefix}_dn") if g.mesh_context else g.constant(0.0, id=f"{prefix}_dn"))
    return g.add(
        "colossal_dragon_membrane",
        id=f"{prefix}_mat",
        inputs={"span_u": span_u, "trail_v": trail_v, "face_is_down": is_dn},
        seed=seed,
        **kw,
    )


@register_subgraph("cobblestone_masonry_pbr", category="World Blocks", description="Seamless Flood-Fill + Slope-Blur cobblestone masonry MaterialBus.", defaults={"cols": 4, "rows": 4, "moss_amount": 0.0})
def _sg_cobblestone_masonry(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 42, **kw: Any) -> str:
    return g.add("cobblestone_masonry", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("bark_timber_pbr", category="World Blocks", description="Seamless furrowed organic tree bark + crevice moss MaterialBus.", defaults={"furrow_count": 6.0, "moss_amount": 0.2})
def _sg_bark_timber(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 109, **kw: Any) -> str:
    return g.add("bark_and_timber", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("crystalline_ore_pbr", category="World Blocks", description="Seamless deepslate/basalt embedded with sparkling faceted crystal ore MaterialBus.", defaults={"ore_density": 0.55, "emissive_strength": 0.65})
def _sg_crystalline_ore(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 211, **kw: Any) -> str:
    return g.add("crystalline_ore_vein", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("caustic_water_pbr", category="World Blocks", description="Seamless looping animated dual-layer Voronoi caustic water MaterialBus.", defaults={"scale": 4.0})
def _sg_caustic_water(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 313, **kw: Any) -> str:
    return g.add("caustic_water", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("mammal_fur_pelt_pbr", category="Creature Bio", description="Directional tufted mammal fur with jagged countershading & coat markings MaterialBus.", defaults={"pattern": "saddle", "tuft_scale": 12.0})
def _sg_mammal_fur_pelt(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 42, **kw: Any) -> str:
    return g.add("organic_fur_pelt", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("avian_plumage_pbr", category="Creature Bio", description="Layered scalloped bird/griffin/phoenix plumage with central rachis quill MaterialBus.", defaults={"marking": "barred", "iridescence": 0.18})
def _sg_avian_plumage(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 42, **kw: Any) -> str:
    return g.add("layered_plumage", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("insect_chitin_pbr", category="Creature Bio", description="Segmented scorpion/beetle chitin carapace with cool plum shadows & iridescence MaterialBus.", defaults={"segments": 6, "iridescence": 0.26})
def _sg_insect_chitin(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 42, **kw: Any) -> str:
    return g.add("chitin_carapace", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("aquatic_scales_pbr", category="Creature Bio", description="Wet glossy fish scales or moist amphibian aposematic skin MaterialBus.", defaults={"mode": "fish_scales", "wetness": 0.82})
def _sg_aquatic_scales(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 42, **kw: Any) -> str:
    return g.add("aquatic_scales_skin", id=f"{prefix}_mat", seed=seed, **kw)


@register_subgraph("elemental_golem_pbr", category="Creature Bio", description="Flood-Fill + Slope-Blur fractured stone/obsidian golem slabs with glowing fissures MaterialBus.", defaults={"slab_scale": 4.5, "glow_strength": 0.85})
def _sg_elemental_golem(g: Any, *, prefix: str, inputs: dict[str, str], seed: int = 42, **kw: Any) -> str:
    return g.add("elemental_golem_surface", id=f"{prefix}_mat", seed=seed, **kw)
