"""3D Model-Aware Spatial, Direction, UV-Box & Substance-Style Smart Mask Operators (`TexGraph v13`).

Lets artists mask any texture or PBR material chain directly from the loaded 3D `.bbmodel`
(or cleanly fall back to 2D coordinates on flat blocks/spheres):
- ``mask_uv_part``          : Mask by 3D model part/bone regex, anatomical zone, or cuboid face
- ``mask_world_position``   : Mask by 3D world-space X / Y / Z height & position bounds
- ``mask_world_direction``  : Mask by 3D surface normal direction (Top/Bottom/Front/Side/Custom Vector)
- ``mask_uv_box``           : Mask by UV atlas box rectangle, UV gradient, or per-island border/interior
- ``mask_3d_distance``      : Mask by 3D sphere, cylinder, or bilateral symmetry distance
- ``mask_smart_edge_wear``  : Substance Painter Smart Edge Wear (3D curvature + bevel + noise chips)
- ``mask_smart_crevice_dirt``: Substance Painter Smart Crevice Dirt (3D AO + joint cavity + ground grime)
- ``mask_combine``          : Boolean mask combiner (Union, Intersect, Subtract, XOR, Multiply) + refine
"""
from __future__ import annotations

import math
import numpy as np

from .generators_noise import fbm_noise, uv_grid
from .generators_noise_ext import grunge_blotch
from .mesh_baker import MeshBakeContext
from .region_ops import _mesh_island_fields, resolve_part_mask


def _box_blur_2d(arr: np.ndarray, radius: int, *, tileable: bool = False) -> np.ndarray:
    r = max(0, int(radius))
    if r == 0:
        return arr.astype(np.float32)
    out = arr.astype(np.float32)
    mode = "wrap" if tileable else "edge"
    k = 2 * r + 1
    # Fast 2-pass separable box blur
    for _ in range(2):
        p = np.pad(out, ((0, 0), (r, r)), mode=mode)
        cs = np.cumsum(np.pad(p, ((0, 0), (1, 0)), mode="constant"), axis=1)
        out = (cs[:, k:] - cs[:, :-k]) / float(k)
        p = np.pad(out, ((r, r), (0, 0)), mode=mode)
        cs = np.cumsum(np.pad(p, ((1, 0), (0, 0)), mode="constant"), axis=0)
        out = (cs[k:, :] - cs[:-k, :]) / float(k)
    return out.astype(np.float32)


def _morph_scalar(arr: np.ndarray, radius: int, *, dilate: bool = True) -> np.ndarray:
    r = max(0, int(radius))
    if r == 0:
        return arr.astype(np.float32)
    out = arr.astype(np.float32)
    for _ in range(r):
        p = np.pad(out, 1, mode="edge")
        nb = (p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:], p[1:-1, 1:-1])
        out = np.maximum.reduce(nb) if dilate else np.minimum.reduce(nb)
    return out.astype(np.float32)


def _apply_contrast_invert(m: np.ndarray, contrast: float = 1.0, invert: bool = False) -> np.ndarray:
    c = max(0.05, float(contrast))
    out = np.clip((m - 0.5) * c + 0.5, 0.0, 1.0)
    if invert:
        out = 1.0 - out
    return out.astype(np.float32)


def mask_uv_part(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    part_pattern: str = "",
    zone: str = "all",
    face_filter: str = "all",
    feather_px: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """Select an exact 3D `.bbmodel` part/bone regex, anatomical zone, or cuboid face as a clean MASK."""
    m = resolve_part_mask(
        height, width, mesh_ctx,
        part_pattern=part_pattern,
        face_filter=face_filter,
        min_island_px=0.0,
    )
    z = str(zone or "all").strip().lower()
    if z and z != "all" and mesh_ctx is not None:
        m = m * mesh_ctx.zone_mask(z)
    if int(feather_px) > 0:
        m = _box_blur_2d(m, int(feather_px), tileable=False)
    if invert:
        if mesh_ctx is not None and mesh_ctx.island_mask.shape == (height, width):
            m = np.where(mesh_ctx.island_mask, 1.0 - m, 0.0)
        else:
            m = 1.0 - m
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def mask_world_position(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    axis: str = "y_height",
    range_min: float = 0.0,
    range_max: float = 1.0,
    contrast: float = 1.0,
    noise_breakup: float = 0.0,
    noise_scale: float = 6.0,
    seed: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """3D Model Spatial Position Mask (normalized X / Y height / Z front-back coordinate)."""
    f = _mesh_island_fields(height, width, mesh_ctx)
    wp = f["world_pos_norm"]
    ax = str(axis or "y_height").strip().lower()
    if ax in ("x", "x_left_right"):
        coord = wp[..., 0]
    elif ax in ("z", "z_front_back"):
        coord = wp[..., 2]
    elif ax == "x_symmetry":
        coord = np.abs(wp[..., 0] * 2.0 - 1.0)
    else:
        coord = wp[..., 1]

    lo = float(range_min)
    hi = float(range_max)
    if abs(hi - lo) < 1e-4:
        hi = lo + 1e-4
    val = np.clip((coord - lo) / (hi - lo), 0.0, 1.0)

    nb = float(noise_breakup)
    if nb > 1e-4:
        n = fbm_noise(height, width, scale=noise_scale, octaves=3, seed=seed + 19, tileable=False) - 0.5
        val = np.clip(val + n * nb, 0.0, 1.0)

    out = _apply_contrast_invert(val, contrast=contrast, invert=invert)
    return (out * f["island_mask"]).astype(np.float32)


def mask_world_direction(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    direction: str = "top_up",
    azimuth_deg: float = 0.0,
    elevation_deg: float = 90.0,
    threshold: float = 0.35,
    softness: float = 0.25,
    noise_breakup: float = 0.0,
    seed: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """3D Surface / World Normal Direction Mask (Top, Bottom, Front, Back, Left, Right, or Custom Vector)."""
    f = _mesh_island_fields(height, width, mesh_ctx)
    wn = f["world_normal"]  # (H, W, 3)

    d = str(direction or "top_up").strip().lower()
    dirs = {
        "top_up": (0.0, 1.0, 0.0),
        "bottom_down": (0.0, -1.0, 0.0),
        "front_north": (0.0, 0.0, -1.0),
        "back_south": (0.0, 0.0, 1.0),
        "right_east": (1.0, 0.0, 0.0),
        "left_west": (-1.0, 0.0, 0.0),
    }
    if d in dirs:
        vx, vy, vz = dirs[d]
    else:
        az = math.radians(float(azimuth_deg))
        el = math.radians(float(elevation_deg))
        vx = math.cos(el) * math.sin(az)
        vy = math.sin(el)
        vz = math.cos(el) * math.cos(az)

    dot = wn[..., 0] * vx + wn[..., 1] * vy + wn[..., 2] * vz
    # Map dot [-1..1] -> [0..1]
    score = dot * 0.5 + 0.5
    if float(noise_breakup) > 1e-4:
        n = fbm_noise(height, width, scale=7.0, octaves=3, seed=seed + 43, tileable=False) - 0.5
        score = np.clip(score + n * float(noise_breakup), 0.0, 1.0)

    thr = float(np.clip(threshold, 0.0, 0.99))
    soft = max(0.01, float(softness))
    m = np.clip((score - thr) / soft, 0.0, 1.0)
    m = m * m * (3.0 - 2.0 * m)
    if invert:
        m = 1.0 - m
    return (m * f["island_mask"]).astype(np.float32)


def mask_uv_box(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    mode: str = "uv_rect",
    u_min: float = 0.0,
    u_max: float = 1.0,
    v_min: float = 0.0,
    v_max: float = 1.0,
    border_px: float = 1.5,
    feather_px: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """UV Atlas Box Rectangle, UV Position Gradient, or Per-Island Border/Interior Mask."""
    u, v = uv_grid(height, width)
    f = _mesh_island_fields(height, width, mesh_ctx)
    md = str(mode or "uv_rect").strip().lower()

    if md == "atlas_u":
        m = u
    elif md == "atlas_v":
        m = v
    elif md == "island_u":
        m = f["local_u"] * f["island_mask"]
    elif md == "island_v":
        m = f["local_v"] * f["island_mask"]
    elif md == "island_border":
        bp = max(0.5, float(border_px))
        m = ((f["edge_px"] < bp) & (f["island_mask"] > 0.5)).astype(np.float32)
    elif md == "island_interior":
        bp = max(0.5, float(border_px))
        m = ((f["edge_px"] >= bp) & (f["island_mask"] > 0.5)).astype(np.float32)
    else:
        # Exact UV atlas rectangle [u_min..u_max] x [v_min..v_max]
        m = ((u >= float(u_min)) & (u <= float(u_max)) & (v >= float(v_min)) & (v <= float(v_max))).astype(np.float32)

    if int(feather_px) > 0:
        m = _box_blur_2d(m, int(feather_px), tileable=False)
    if invert:
        m = 1.0 - m
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def mask_3d_distance(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    shape: str = "sphere",
    cx: float = 0.5,
    cy: float = 0.5,
    cz: float = 0.5,
    radius: float = 0.35,
    falloff: float = 0.25,
    invert: bool = False,
) -> np.ndarray:
    """3D Spherical, Cylindrical, or Bilateral Symmetry Distance Mask in normalized model space."""
    f = _mesh_island_fields(height, width, mesh_ctx)
    wp = f["world_pos_norm"]
    dx = wp[..., 0] - float(cx)
    dy = wp[..., 1] - float(cy)
    dz = wp[..., 2] - float(cz)

    sh = str(shape or "sphere").strip().lower()
    if sh == "cylinder_y":
        dist = np.sqrt(dx * dx + dz * dz + 1e-12)
    elif sh == "symmetry_x":
        dist = np.abs(dx)
    else:
        dist = np.sqrt(dx * dx + dy * dy + dz * dz + 1e-12)

    rad = max(0.01, float(radius))
    fo = max(0.01, float(falloff))
    m = np.clip(1.0 - (dist - rad * (1.0 - fo)) / max(1e-4, rad * fo), 0.0, 1.0)
    m = m * m * (3.0 - 2.0 * m)
    if invert:
        m = 1.0 - m
    return (m * f["island_mask"]).astype(np.float32)


def mask_smart_edge_wear(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    wear_amount: float = 0.55,
    contrast: float = 1.8,
    chip_scale: float = 8.0,
    chip_amount: float = 0.45,
    seed: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """Substance Painter Smart Edge Wear Mask driven by 3D curvature, UV edge bevel & grunge chipping."""
    f = _mesh_island_fields(height, width, mesh_ctx)
    if mesh_ctx is not None and mesh_ctx.curvature_3d.shape == (height, width):
        curv = np.clip((mesh_ctx.curvature_3d - 0.45) * 2.2, 0.0, 1.0)
        edge = np.maximum(curv, mesh_ctx.edge_bevel * 0.85)
    else:
        edge = np.clip(1.0 - f["edge_px"] / 3.0, 0.0, 1.0)

    chips = grunge_blotch(height, width, scale=chip_scale, contrast=1.5, speckle=0.35, seed=seed + 77, tileable=False)
    ca = float(np.clip(chip_amount, 0.0, 1.0))
    wear = edge * (1.0 - ca + ca * chips * 1.45) + (chips * edge) * 0.25
    wa = float(np.clip(wear_amount, 0.0, 1.0))
    val = np.clip(wear * (0.4 + 1.4 * wa) - (1.0 - wa) * 0.35, 0.0, 1.0)
    out = _apply_contrast_invert(val, contrast=contrast, invert=invert)
    return (out * f["island_mask"]).astype(np.float32)


def mask_smart_crevice_dirt(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    dirt_amount: float = 0.55,
    ao_weight: float = 0.75,
    cavity_weight: float = 0.65,
    ground_bias: float = 0.25,
    grunge_scale: float = 6.0,
    seed: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """Substance Painter Smart Crevice Dirt / Patina Mask (3D AO + joint cavity + ground grime + grunge)."""
    f = _mesh_island_fields(height, width, mesh_ctx)
    inv_ao = 1.0 - np.clip(f["baked_ao"], 0.0, 1.0)
    cav = np.clip(f["joint_cavity"], 0.0, 1.0)
    low_y = 1.0 - np.clip(f["world_pos_norm"][..., 1], 0.0, 1.0)
    grunge = grunge_blotch(height, width, scale=grunge_scale, contrast=1.35, speckle=0.3, seed=seed + 131, tileable=False)

    base = (
        inv_ao * float(ao_weight)
        + cav * float(cavity_weight)
        + (low_y ** 2.2) * float(ground_bias)
    )
    da = float(np.clip(dirt_amount, 0.0, 1.0))
    val = np.clip((base * (0.55 + 0.75 * grunge) + grunge * 0.20 * da) * (0.5 + 1.1 * da), 0.0, 1.0)
    if invert:
        val = 1.0 - val
    return (val * f["island_mask"]).astype(np.float32)


def mask_combine(
    mask_a: np.ndarray,
    mask_b: np.ndarray | None = None,
    *,
    mode: str = "union",
    op: str | None = None,
    contrast: float = 1.0,
    expand_px: int = 0,
    blur_px: int = 0,
    invert: bool = False,
) -> np.ndarray:
    """Boolean & Morphological Mask Combiner (`union`, `intersect`, `subtract`, `xor`, `multiply`, `add`, `screen`)."""
    a = np.asarray(mask_a, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    a = np.clip(a, 0.0, 1.0)
    if mask_b is not None:
        b = np.asarray(mask_b, dtype=np.float32)
        if b.ndim == 3:
            b = b[..., 0]
        b = np.clip(b, 0.0, 1.0)
        o = str(op or mode or "union").strip().lower()
        if o in ("intersect", "min"):
            out = np.minimum(a, b)
        elif o == "subtract":
            out = np.clip(a - b, 0.0, 1.0)
        elif o == "xor":
            out = np.abs(a - b)
        elif o == "multiply":
            out = a * b
        elif o == "add":
            out = np.clip(a + b, 0.0, 1.0)
        elif o == "screen":
            out = 1.0 - (1.0 - a) * (1.0 - b)
        else:
            out = np.maximum(a, b)
    else:
        out = a.copy()

    ep = int(expand_px)
    if ep > 0:
        out = _morph_scalar(out, ep, dilate=True)
    elif ep < 0:
        out = _morph_scalar(out, -ep, dilate=False)
    if int(blur_px) > 0:
        out = _box_blur_2d(out, int(blur_px), tileable=False)
    return _apply_contrast_invert(out, contrast=contrast, invert=invert)
