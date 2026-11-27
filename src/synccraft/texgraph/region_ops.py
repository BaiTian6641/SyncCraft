"""Per-UV-Island & Per-Body-Part Region-Constrained Pixel Composer Operators (`TexGraph v11`).

Distilled directly from official Pixel Composer (`docs.pixel-composer.com`) workflows and expert
Blockbench character texturing timelapses (Everbloom Games, ArtsByKev, MakhamDev):

1. ``uv_island_coords``        : Per-island normalized UV, pixel border distance, eroded ``Inner Area``
                                 mask (`PB FX Bevel`), island pixel scale & face-direction masks
                                 (`Node_UV_Area` parity).
2. ``region_base_fill``        : Clean Everbloom/ArtsByKev per-island directional value blocking
                                 (`up` sunlit / `side` top-to-bottom gradient / `down` shadow) plus
                                 pixel-locked border-exact hem/cuff/collar trim (`Node_Region_Fill`).
3. ``region_restrained_noise`` : Area-restrained per-island pixel noise with two-stage `Level In/Out`
                                 amplitude clamping, `Inner Area` 1 px rim protection, `min_island_px`
                                 micro-face gate, and texel-locked 2 px pixel clumps (zero static
                                 confetti, zero cross-island bleed!).
4. ``region_bevel_and_ao``     : Per-island broken sunlit `1 px` directional bevel (`PB FX Bevel`)
                                 and hue-shifted 3D joint/contact AO (never neutral gray).
5. ``region_composite``        : Alpha-preserving (`Preserve Alpha = true`) per-region `MATERIAL` bus
                                 compositor with `1 px` boundary seam AO and optional per-region
                                 OKLab `Node_Posterize` palette ramp cohesion.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .generators_gradients import multi_stop_gradient
from .generators_noise import uv_grid
from .mesh_baker import MeshBakeContext
from .pbr import height_to_normal_map
from .pipeline import MATERIAL_BUS_PORTS, _ensure_bus, material_make
from .pixel_art_ops import hex_to_srgb
from .pixel_composer_ops import palette_posterize


def _normalize_stops(stops: Sequence[Any]) -> tuple[tuple[float, str], ...]:
    seq = list(stops)
    if not seq:
        return ((0.0, "#1C1A30"), (0.5, "#7E7AA8"), (1.0, "#F2F4FF"))
    if isinstance(seq[0], str):
        n = len(seq)
        if n == 1:
            return ((0.0, str(seq[0])), (1.0, str(seq[0])))
        return tuple((float(i) / float(n - 1), str(c)) for i, c in enumerate(seq))
    return tuple((float(p[0]), str(p[1])) for p in seq)


def _stops_to_hex_list(stops: Sequence[Any]) -> list[str]:
    return [c for _, c in _normalize_stops(stops)]


def _hash2d(ix: np.ndarray, iy: np.ndarray, seed_map: np.ndarray | int) -> np.ndarray:
    """Fast deterministic vectorized 2D integer hash returning float32 in [0, 1]."""
    n = (
        ix.astype(np.int64) * 374761393
        + iy.astype(np.int64) * 668265263
        + np.asarray(seed_map, dtype=np.int64) * 1442695041
    )
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return ((n & 0x7FFFFFFF).astype(np.float64) / float(0x7FFFFFFF)).astype(np.float32)


def _value_noise_coords(x: np.ndarray, y: np.ndarray, seed_map: np.ndarray | int) -> np.ndarray:
    """Smooth bilinear value noise sampled directly in per-island local coordinates."""
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = (x - ix).astype(np.float32)
    fy = (y - iy).astype(np.float32)
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    v00 = _hash2d(ix, iy, seed_map)
    v10 = _hash2d(ix + 1, iy, seed_map)
    v01 = _hash2d(ix, iy + 1, seed_map)
    v11 = _hash2d(ix + 1, iy + 1, seed_map)
    a = v00 * (1.0 - sx) + v10 * sx
    b = v01 * (1.0 - sx) + v11 * sx
    return (a * (1.0 - sy) + b * sy).astype(np.float32)


def _mesh_island_fields(
    H: int,
    W: int,
    mesh_ctx: MeshBakeContext | None,
) -> dict[str, np.ndarray]:
    """Return aligned per-UV-island geometry maps with seamless 2D fallback when `mesh_ctx` is None."""
    if mesh_ctx is not None and mesh_ctx.local_uv.shape[:2] == (H, W):
        # Support older MeshBakeContext fixtures gracefully if any new attribute has default (1, 1) shape
        if mesh_ctx.island_edge_px.shape == (H, W):
            edge_px = mesh_ctx.island_edge_px
            size_min = mesh_ctx.island_size_min
            size_wh = mesh_ctx.island_size_wh
            inner_area = mesh_ctx.island_inner_area
            face_up = mesh_ctx.face_is_up
            face_side = mesh_ctx.face_is_side
        else:
            edge_px = np.clip((1.0 - mesh_ctx.edge_bevel) * 2.0, 0.0, 8.0).astype(np.float32)
            size_min = np.full((H, W), 8.0, dtype=np.float32)
            size_wh = np.full((H, W, 2), 8.0, dtype=np.float32)
            inner_area = (edge_px >= 1.0).astype(np.float32)
            face_up = (mesh_ctx.world_normal[..., 1] > 0.65).astype(np.float32)
            face_side = (np.abs(mesh_ctx.world_normal[..., 1]) <= 0.65).astype(np.float32)
        return {
            "island_mask": mesh_ctx.island_mask.astype(np.float32),
            "local_u": mesh_ctx.local_uv[..., 0].astype(np.float32),
            "local_v": mesh_ctx.local_uv[..., 1].astype(np.float32),
            "edge_px": edge_px.astype(np.float32),
            "size_min": size_min.astype(np.float32),
            "size_w": np.maximum(1.0, size_wh[..., 0]).astype(np.float32),
            "size_h": np.maximum(1.0, size_wh[..., 1]).astype(np.float32),
            "inner_area": inner_area.astype(np.float32),
            "face_up": face_up.astype(np.float32),
            "face_side": face_side.astype(np.float32),
            "face_down": mesh_ctx.face_is_down.astype(np.float32),
            "world_pos_norm": mesh_ctx.world_pos_norm.astype(np.float32),
            "world_normal": mesh_ctx.world_normal.astype(np.float32),
            "baked_ao": mesh_ctx.baked_ao.astype(np.float32),
            "joint_cavity": mesh_ctx.joint_cavity.astype(np.float32),
            "elem_id": mesh_ctx.elem_id_map.astype(np.int32),
            "face_id": mesh_ctx.face_id_map.astype(np.int32),
        }

    u, v = uv_grid(H, W)
    dx_px = np.minimum(np.arange(W, dtype=np.float32), (W - 1) - np.arange(W, dtype=np.float32))
    dy_px = np.minimum(np.arange(H, dtype=np.float32), (H - 1) - np.arange(H, dtype=np.float32))
    edge_px = np.minimum(dx_px[None, :], dy_px[:, None]).astype(np.float32)
    min_wh = float(min(H, W))
    inner = np.clip(edge_px, 0.0, 1.0).astype(np.float32) if min_wh >= 4.0 else np.zeros((H, W), dtype=np.float32)
    wpn = np.stack((u, 1.0 - v, np.full_like(u, 0.5)), axis=-1).astype(np.float32)
    wn = np.stack((np.zeros_like(u), np.full_like(u, 0.45), np.full_like(u, 0.89)), axis=-1).astype(np.float32)
    return {
        "island_mask": np.ones((H, W), dtype=np.float32),
        "local_u": u.astype(np.float32),
        "local_v": v.astype(np.float32),
        "edge_px": edge_px,
        "size_min": np.full((H, W), min_wh, dtype=np.float32),
        "size_w": np.full((H, W), float(W), dtype=np.float32),
        "size_h": np.full((H, W), float(H), dtype=np.float32),
        "inner_area": inner,
        "face_up": np.zeros((H, W), dtype=np.float32),
        "face_side": np.ones((H, W), dtype=np.float32),
        "face_down": np.zeros((H, W), dtype=np.float32),
        "world_pos_norm": wpn,
        "world_normal": wn,
        "baked_ao": np.ones((H, W), dtype=np.float32),
        "joint_cavity": np.zeros((H, W), dtype=np.float32),
        "elem_id": np.zeros((H, W), dtype=np.int32),
        "face_id": np.zeros((H, W), dtype=np.int32),
    }


def resolve_part_mask(
    H: int,
    W: int,
    mesh_ctx: MeshBakeContext | None,
    *,
    part_pattern: str = "",
    face_filter: str = "all",
    min_island_px: float = 0.0,
) -> np.ndarray:
    """Resolve a strict float32 (H, W) binary region mask for a body part regex, face filter & min island size.

    Supports optional leading ``!`` negation (e.g. ``"!eye|crystal"`` selects all valid islands except matches).
    """
    f = _mesh_island_fields(H, W, mesh_ctx)
    m = f["island_mask"].copy()

    pat = str(part_pattern or "").strip()
    if pat and mesh_ctx is not None:
        if pat.startswith("!"):
            inner_pat = pat[1:].strip()
            if inner_pat:
                m = m * (1.0 - mesh_ctx.match_elements_mask(inner_pat))
        else:
            m = m * mesh_ctx.match_elements_mask(pat)

    ff = str(face_filter or "all").strip().lower()
    if ff and ff != "all":
        if ff == "up":
            m = m * f["face_up"]
        elif ff == "down":
            m = m * f["face_down"]
        elif ff == "side":
            m = m * f["face_side"]
        elif mesh_ctx is not None:
            m = m * mesh_ctx.face_filter_mask(ff)

    if float(min_island_px) > 0.0:
        m = m * (f["size_min"] >= float(min_island_px)).astype(np.float32)

    return np.clip(m, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 1. `uv_island_coords` — Pixel Composer `Node_UV_Area` & `Inner Area` Parity
# ---------------------------------------------------------------------------


def uv_island_coords(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    channel: str = "inner_area",
    part_pattern: str = "",
    face_filter: str = "all",
    min_island_px: float = 0.0,
    inner_margin_px: float = 1.0,
) -> np.ndarray:
    """Emit per-UV-island local coordinates, pixel edge distance, eroded `Inner Area`, or part masks."""
    f = _mesh_island_fields(height, width, mesh_ctx)
    reg = resolve_part_mask(
        height, width, mesh_ctx,
        part_pattern=part_pattern,
        face_filter=face_filter,
        min_island_px=min_island_px,
    )
    ch = str(channel or "inner_area").strip().lower()
    if ch == "local_u":
        val = f["local_u"]
    elif ch == "local_v":
        val = f["local_v"]
    elif ch == "edge_px":
        val = np.clip(f["edge_px"] / 8.0, 0.0, 1.0)
    elif ch == "inner_area":
        margin = max(0.0, float(inner_margin_px))
        min_sz = max(float(min_island_px), 2.0 * margin + 2.0)
        val = ((f["edge_px"] >= margin) & (f["size_min"] >= min_sz)).astype(np.float32)
    elif ch == "island_scale":
        val = np.clip(f["size_min"] / 32.0, 0.0, 1.0)
    elif ch == "face_up":
        val = f["face_up"]
    elif ch == "face_side":
        val = f["face_side"]
    elif ch == "face_down":
        val = f["face_down"]
    elif ch == "part_mask":
        val = np.ones((height, width), dtype=np.float32)
    else:
        raise ValueError(f"Unknown uv_island_coords channel: {channel!r}")
    return (val * reg).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. `region_base_fill` — Per-Part Clean Directional Blocking & Border-Exact Trim
# ---------------------------------------------------------------------------


def region_base_fill(
    height: int,
    width: int,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    part_pattern: str = "",
    stops: Sequence[Any] = (
        (0.00, "#242240"),
        (0.25, "#4A4573"),
        (0.50, "#817EB0"),
        (0.75, "#B7B9DC"),
        (0.92, "#DFE3F6"),
        (1.00, "#F7F8FF"),
    ),
    top_boost: float = 0.16,
    bottom_darken: float = 0.18,
    side_grad: float = 0.20,
    body_grad: float = 0.12,
    smooth_blend: float = 0.72,
    quantize_steps: int = 6,
    trim_edge: str = "none",
    trim_color: str = "#ECC560",
    trim_width_px: float = 2.0,
    trim_faces: str = "side",
    trim_metallic: float = 0.88,
    trim_roughness: float = 0.22,
    roughness_base: float = 0.64,
    metallic_base: float = 0.0,
    height_base: float = 0.50,
    normal_strength: float = 2.2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate clean per-UV-island directional value blocking + optional border-exact pixel-locked trim.

    Follows the Everbloom / ArtsByKev / Pixel Composer `Node_Region_Fill` blocking principle:
      - Sunlit `up` planes get a crisp upward value step (`top_boost`)
      - Grounded `down` planes drop into cool shadow (`bottom_darken`)
      - Vertical `side` planes carry a calm top-to-bottom per-island value sweep (`side_grad`)
      - Subtle 3D dorsal-ventral continuity (`body_grad`) keeps multi-box limbs unified
      - Optional border-exact `trim_edge` (`bottom`, `top`, `both`, `border`) paints crisp pixel-locked
        coat hems, cuffs, collars & armor rims strictly inside each island's real border pixels!
    """
    _ = seed
    f = _mesh_island_fields(height, width, mesh_ctx)
    reg_mask = resolve_part_mask(height, width, mesh_ctx, part_pattern=part_pattern)

    # Clean directional value field per UV island
    wn_y = f["world_normal"][..., 1]
    wn_z = f["world_normal"][..., 2]
    wpn_y = f["world_pos_norm"][..., 1]

    val = (
        0.50
        + f["face_up"] * float(top_boost)
        - f["face_down"] * float(bottom_darken)
        + f["face_side"] * (0.45 - f["local_v"]) * float(side_grad)
        + (wpn_y - 0.5) * float(body_grad)
        + 0.06 * wn_y
        + 0.03 * wn_z
    )
    # Soft 1px convex arch on vertical faces so cuboids feel subtly rounded
    arch = (1.0 - np.abs(f["local_u"] - 0.5) * 2.0) * 0.04 * f["face_side"]
    val = np.clip(val + arch, 0.04, 0.96).astype(np.float32)

    norm_stops = _normalize_stops(stops)
    albedo = multi_stop_gradient(
        val,
        norm_stops,
        space="oklab",
        smooth_blend=float(smooth_blend),
        quantize_steps=int(quantize_steps),
    )

    h_map = np.clip(float(height_base) + (val - 0.5) * 0.22, 0.05, 0.95).astype(np.float32)
    rough = np.full((height, width), float(np.clip(roughness_base, 0.04, 1.0)), dtype=np.float32)
    metal = np.full((height, width), float(np.clip(metallic_base, 0.0, 1.0)), dtype=np.float32)
    trim_mask = np.zeros((height, width), dtype=np.float32)

    # Border-exact pixel-locked hem / cuff / collar trim (never slices unrelated faces horizontally!)
    te = str(trim_edge or "none").strip().lower()
    tw = max(0.5, float(trim_width_px))
    if te != "none":
        tf = str(trim_faces or "side").strip().lower()
        if tf == "side":
            face_ok = f["face_side"] > 0.5
        elif tf == "up":
            face_ok = f["face_up"] > 0.5
        elif tf == "down":
            face_ok = f["face_down"] > 0.5
        else:
            face_ok = f["island_mask"] > 0.5

        d_bot_px = (1.0 - f["local_v"]) * f["size_h"]
        d_top_px = f["local_v"] * f["size_h"]
        min_h_ok = f["size_h"] >= max(4.0, tw * 2.2)

        trim_bool = np.zeros((height, width), dtype=bool)
        seam_bool = np.zeros((height, width), dtype=bool)
        if te in ("bottom", "both"):
            trim_bool |= face_ok & min_h_ok & (d_bot_px <= tw)
            seam_bool |= face_ok & min_h_ok & (d_bot_px > tw) & (d_bot_px <= tw + 1.05)
        if te in ("top", "both"):
            trim_bool |= face_ok & min_h_ok & (d_top_px <= tw)
            seam_bool |= face_ok & min_h_ok & (d_top_px > tw) & (d_top_px <= tw + 1.05)
        if te == "border":
            min_s_ok = f["size_min"] >= max(4.0, tw * 2.5)
            trim_bool |= face_ok & min_s_ok & (f["edge_px"] < tw)
            seam_bool |= face_ok & min_s_ok & (f["edge_px"] >= tw) & (f["edge_px"] < tw + 1.05)

        trim_bool &= reg_mask > 0.5
        seam_bool &= (reg_mask > 0.5) & (~trim_bool)

        if np.any(trim_bool):
            tr, tg, tb = hex_to_srgb(trim_color)
            trim_rgb = np.array([tr, tg, tb], dtype=np.float32)
            # Subtle warm metallic/embroidered sheen along the trim strip
            sheen = (0.80 + 0.24 * np.sin(f["local_u"] * math.pi) + 0.08 * np.clip(wn_y, -0.5, 1.0)).astype(np.float32)
            trim_shaded = np.clip(trim_rgb[None, None, :] * sheen[..., None], 0.0, 1.0)
            albedo[..., :3] = np.where(trim_bool[..., None], trim_shaded, albedo[..., :3])
            # Crisp 1px tailored dark seam immediately bordering the trim
            albedo[..., :3] = np.where(seam_bool[..., None], albedo[..., :3] * 0.72, albedo[..., :3])
            h_map = np.where(trim_bool, np.clip(h_map + 0.18, 0.0, 1.0), h_map)
            h_map = np.where(seam_bool, np.clip(h_map - 0.12, 0.0, 1.0), h_map)
            rough = np.where(trim_bool, float(np.clip(trim_roughness, 0.04, 1.0)), rough)
            metal = np.where(trim_bool, float(np.clip(trim_metallic, 0.0, 1.0)), metal)
            trim_mask = trim_bool.astype(np.float32)

    # Preserve strict per-part alpha when `part_pattern` is specified
    if str(part_pattern or "").strip():
        albedo[..., 3] = reg_mask
    else:
        albedo[..., 3] = np.where(f["island_mask"] > 0.5, 1.0, 0.0).astype(np.float32)

    bus = material_make(
        albedo,
        height=h_map,
        roughness=rough,
        metallic=metal,
        ao=np.ones((height, width), dtype=np.float32),
        normal_strength=float(normal_strength),
    )
    bus["region_mask"] = reg_mask
    bus["trim_mask"] = trim_mask
    return bus


# ---------------------------------------------------------------------------
# 3. `region_restrained_noise` — Two-Stage Level Clamp + `Inner Area` Restraint
# ---------------------------------------------------------------------------


def region_restrained_noise(
    mat: dict[str, np.ndarray] | np.ndarray,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    part_pattern: str = "",
    pattern: str = "organic_clump",
    scale: float = 1.0,
    amplitude: float = 0.14,
    level_in_lo: float = 0.28,
    level_in_hi: float = 0.72,
    clump_px: int = 2,
    edge_restraint: float = 0.85,
    min_island_px: float = 4.0,
    height_bump: float = 0.16,
    roughness_var: float = 0.10,
    protect_trim: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Apply Pixel Composer area-restrained, texel-locked noise strictly inside each UV island.

    Prevents the two classic procedural texturing failures:
      1. **Static Confetti**: Two-stage `Level In/Out` clamping + quantized `clump_px` pixel clusters.
      2. **Bleed & Muddy Rims**: Per-island isolated coordinates + `Inner Area` (`edge_restraint`)
         rim preservation + `min_island_px` micro-face gate + optional `trim_mask` protection!
    """
    bus = _ensure_bus(mat)
    albedo = bus["albedo"].copy()
    H, W = albedo.shape[:2]
    f = _mesh_island_fields(H, W, mesh_ctx)

    if str(part_pattern or "").strip():
        reg_m = resolve_part_mask(H, W, mesh_ctx, part_pattern=part_pattern)
    elif "region_mask" in bus and isinstance(bus["region_mask"], np.ndarray) and bus["region_mask"].shape == (H, W):
        reg_m = bus["region_mask"].astype(np.float32)
    else:
        reg_m = (albedo[..., 3] > 0.05).astype(np.float32) * f["island_mask"]

    # Pixel Composer `Inner Area` edge protection + `min_island_px` micro-island protection
    er = float(np.clip(edge_restraint, 0.0, 1.0))
    restraint_m = (1.0 - er) + er * f["inner_area"]
    size_ok = (f["size_min"] >= float(min_island_px)).astype(np.float32)
    eff_mask = reg_m * restraint_m * size_ok
    if protect_trim and "trim_mask" in bus and isinstance(bus["trim_mask"], np.ndarray) and bus["trim_mask"].shape == (H, W):
        eff_mask = eff_mask * (1.0 - 0.85 * bus["trim_mask"].astype(np.float32))

    # Per-island isolated pixel coordinates (strictly zero cross-island bleeding!)
    clump = max(1, int(clump_px))
    px_x = f["local_u"] * f["size_w"]
    px_y = f["local_v"] * f["size_h"]
    qx = np.floor(px_x / float(clump)) * float(clump)
    qy = np.floor(px_y / float(clump)) * float(clump)
    # Deterministic per-(element, face) seed so adjacent islands have natural independent grain
    island_seed = (f["elem_id"].astype(np.int64) * 37 + f["face_id"].astype(np.int64) * 13 + int(seed))
    sc = max(0.15, float(scale))

    pat_key = str(pattern or "organic_clump").strip().lower()
    if pat_key == "clump_scales":
        # Staggered imbricated draconic/reptile scale shingles strictly in per-island texel space!
        cell_w = max(2.0, 4.0 / sc)
        cell_h = max(2.0, 3.5 / sc)
        row_i = np.floor(px_y / cell_h)
        stagger = (row_i.astype(np.int64) & 1).astype(np.float32) * 0.5
        su = (px_x / cell_w + stagger) % 1.0
        sv = (px_y / cell_h) % 1.0
        # Rounded overlapping scale dome: crisp bright upper-center belly, dark bottom overlap lip
        dx_s = (su - 0.5) * 2.0
        arch = np.clip(1.0 - 0.65 * (dx_s * dx_s) - 0.45 * np.maximum(0.0, sv - 0.55), 0.0, 1.0)
        lip_shadow = np.clip((sv - 0.76) / 0.24, 0.0, 1.0) * 0.55
        tile_var = _hash2d(np.floor(px_x / cell_w + stagger), row_i, island_seed) * 0.24 - 0.12
        raw = np.clip(0.56 * arch - lip_shadow + 0.32 + tile_var, 0.0, 1.0).astype(np.float32)
    elif pat_key == "weave_fabric":
        # Fine woven twill cloth grain + soft vertical drape fold shadows per island
        twill = (((np.floor(px_x * sc).astype(np.int64) + np.floor(px_y * sc).astype(np.int64)) % 4) < 2).astype(np.float32)
        fold_phase = _hash2d(f["elem_id"], f["face_id"], int(seed) + 17) * 6.28318
        drape = 0.5 + 0.5 * np.sin(f["local_u"] * math.pi * (2.0 + 1.5 * sc) + fold_phase)
        clump_n = _value_noise_coords(qx * 0.22 * sc, qy * 0.22 * sc, island_seed)
        raw = np.clip(0.32 * twill + 0.44 * drape + 0.24 * clump_n, 0.0, 1.0).astype(np.float32)
    elif pat_key == "brushed_metal":
        # Anisotropic fine brushed metal streaks + subtle bevelled glint variations
        streak = _value_noise_coords(qx * 0.08 * sc, qy * 0.65 * sc, island_seed)
        micro = _value_noise_coords(qx * 0.35 * sc, qy * 0.35 * sc, island_seed + 91)
        sheen = 0.5 + 0.5 * np.sin((f["local_u"] * 0.7 + f["local_v"] * 0.5) * math.pi * 2.0)
        raw = np.clip(0.55 * streak + 0.25 * micro + 0.20 * sheen, 0.0, 1.0).astype(np.float32)
    elif pat_key == "leather_grain":
        # Tumbled pebble grain + subtle warm wear creases
        peb = _value_noise_coords(qx * 0.48 * sc, qy * 0.48 * sc, island_seed)
        crease = np.abs(_value_noise_coords(qx * 0.14 * sc, qy * 0.28 * sc, island_seed + 41) - 0.5) * 2.0
        raw = np.clip(0.62 * peb + 0.38 * (1.0 - crease), 0.0, 1.0).astype(np.float32)
    elif pat_key == "crystal_facet":
        # Sharp faceted angular crystal planes (steep diagonal Voronoi-like bevel planes!)
        n1 = _value_noise_coords((qx + qy * 0.6) * 0.26 * sc, (qy - qx * 0.6) * 0.26 * sc, island_seed)
        n2 = _value_noise_coords((qx - qy * 0.5) * 0.45 * sc, (qy + qx * 0.5) * 0.45 * sc, island_seed + 53)
        facet = np.floor(np.clip(0.65 * n1 + 0.35 * n2, 0.0, 0.999) * 4.0) / 3.0
        raw = facet.astype(np.float32)
    else:
        # "organic_clump": Calm Everbloom 2-octave painterly pixel clusters (zero static!)
        n1 = _value_noise_coords(qx * 0.26 * sc, qy * 0.26 * sc, island_seed)
        n2 = _value_noise_coords(qx * 0.52 * sc, qy * 0.52 * sc, island_seed + 79)
        raw = np.clip(0.68 * n1 + 0.32 * n2, 0.0, 1.0).astype(np.float32)

    # Stage 1 Pixel Composer `Level In/Out` clamp -> restrained [-amplitude .. +amplitude]
    lo = float(np.clip(level_in_lo, 0.0, 0.95))
    hi = float(np.clip(level_in_hi, lo + 0.05, 1.0))
    clamped = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
    delta = ((clamped - 0.5) * 2.0 * float(amplitude) * eff_mask).astype(np.float32)

    # Apply hue-shifted luminance breakup (warm highlights, cool shadows — never muddy gray!)
    pos_d = np.maximum(0.0, delta)[..., None]
    neg_d = np.maximum(0.0, -delta)[..., None]
    warm_w = np.array([1.04, 1.01, 0.95], dtype=np.float32)[None, None, :]
    cool_w = np.array([1.06, 1.02, 0.92], dtype=np.float32)[None, None, :]
    rgb = albedo[..., :3]
    rgb = rgb + pos_d * warm_w * (0.35 + 0.65 * (1.0 - rgb))
    rgb = rgb - neg_d * cool_w * (0.25 + 0.75 * rgb)
    albedo[..., :3] = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    # Preserve Alpha strictly (`Preserve Alpha = true`)!
    albedo[..., 3] = bus["albedo"][..., 3]

    h_new = np.clip(bus["height"] + delta * float(height_bump), 0.0, 1.0).astype(np.float32)
    r_new = np.clip(bus["roughness"] - delta * float(roughness_var), 0.04, 1.0).astype(np.float32)
    n_new = height_to_normal_map(h_new, strength=2.2, tileable=False)

    out = dict(bus)
    out["albedo"] = albedo
    out["rgba"] = albedo
    out["height"] = h_new
    out["normal"] = n_new
    out["roughness"] = r_new
    out["noise_mask"] = eff_mask
    return out


# ---------------------------------------------------------------------------
# 4. `region_bevel_and_ao` — Pixel Composer `PB FX Bevel` + Hue-Shifted AO
# ---------------------------------------------------------------------------


def region_bevel_and_ao(
    mat: dict[str, np.ndarray] | np.ndarray,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    part_pattern: str = "",
    highlight_strength: float = 0.16,
    shadow_strength: float = 0.20,
    shadow_tint: str = "#1C1A3A",
    highlight_tint: str = "#FFF8EE",
    bevel_breakup: float = 0.45,
    ao_strength: float = 0.28,
    cavity_strength: float = 0.34,
    min_island_px: float = 3.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Apply per-UV-island broken sunlit `1 px` directional bevels (`PB FX Bevel`) & hue-shifted joint AO."""
    bus = _ensure_bus(mat)
    albedo = bus["albedo"].copy()
    H, W = albedo.shape[:2]
    f = _mesh_island_fields(H, W, mesh_ctx)

    if str(part_pattern or "").strip():
        reg_m = resolve_part_mask(H, W, mesh_ctx, part_pattern=part_pattern)
    elif "region_mask" in bus and isinstance(bus["region_mask"], np.ndarray) and bus["region_mask"].shape == (H, W):
        reg_m = bus["region_mask"].astype(np.float32)
    else:
        reg_m = (albedo[..., 3] > 0.05).astype(np.float32) * f["island_mask"]

    px_x = f["local_u"] * f["size_w"]
    px_y = f["local_v"] * f["size_h"]
    size_ok = (f["size_min"] >= float(min_island_px)).astype(np.float32)

    # 1px top/left sunlit leading edge vs bottom/right shadow trailing edge
    top_edge = (px_y < 1.05).astype(np.float32)
    left_edge = (px_x < 1.05).astype(np.float32)
    bot_edge = ((f["size_h"] - px_y) < 1.05).astype(np.float32)
    right_edge = ((f["size_w"] - px_x) < 1.05).astype(np.float32)

    # Deterministic painterly breakup so bevels read as hand-painted broken highlights, never neon grids!
    island_seed = f["elem_id"].astype(np.int64) * 53 + f["face_id"].astype(np.int64) * 19 + int(seed)
    brk = _value_noise_coords(px_x * 0.45, px_y * 0.45, island_seed)
    keep_hi = np.clip((brk - float(bevel_breakup) * 0.55) / max(0.15, 1.0 - float(bevel_breakup) * 0.55), 0.0, 1.0)

    exposed = np.clip(1.0 - f["joint_cavity"] * 1.2, 0.0, 1.0) * f["baked_ao"]
    hi_m = np.clip(np.maximum(top_edge, left_edge * 0.65) * keep_hi * exposed * size_ok * reg_m, 0.0, 1.0)
    sh_m = np.clip(np.maximum(bot_edge, right_edge * 0.65) * size_ok * reg_m, 0.0, 1.0)

    hi_rgb = np.array(hex_to_srgb(highlight_tint), dtype=np.float32)[None, None, :]
    sh_rgb = np.array(hex_to_srgb(shadow_tint), dtype=np.float32)[None, None, :]

    rgb = albedo[..., :3]
    # Apply sunlit broken bevel highlight
    hi_w = (hi_m * float(highlight_strength))[..., None]
    rgb = rgb * (1.0 - hi_w) + np.clip(rgb * 0.55 + hi_rgb * 0.55, 0.0, 1.0) * hi_w
    # Apply trailing edge shadow tinted toward `shadow_tint`
    sh_w = (sh_m * float(shadow_strength))[..., None]
    rgb = rgb * (1.0 - sh_w) + (rgb * sh_rgb * 1.45) * sh_w

    # Apply hue-shifted 3D baked AO & joint cavity (never neutral gray!)
    occ = np.clip((1.0 - f["baked_ao"]) * float(ao_strength) + f["joint_cavity"] * float(cavity_strength), 0.0, 0.78) * reg_m
    occ_w = occ[..., None]
    rgb = rgb * (1.0 - occ_w) + (rgb * (0.38 + 0.62 * sh_rgb)) * occ_w

    albedo[..., :3] = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    albedo[..., 3] = bus["albedo"][..., 3]

    h_new = np.clip(bus["height"] + hi_m * 0.08 - sh_m * 0.08 - f["joint_cavity"] * 0.10 * reg_m, 0.0, 1.0).astype(np.float32)
    ao_new = np.clip(bus["ao"] * (1.0 - occ * 0.65), 0.12, 1.0).astype(np.float32)
    n_new = height_to_normal_map(h_new, strength=2.2, tileable=False)

    out = dict(bus)
    out["albedo"] = albedo
    out["rgba"] = albedo
    out["height"] = h_new
    out["normal"] = n_new
    out["ao"] = ao_new
    out["bevel_mask"] = hi_m
    return out


# ---------------------------------------------------------------------------
# 5. `region_composite` — Preserve-Alpha Region Composite + Seam AO + Palette Snap
# ---------------------------------------------------------------------------


def _dilate1(mask: np.ndarray) -> np.ndarray:
    """Fast 4-connected 1 px morphological dilation (non-wrapping)."""
    m = np.asarray(mask, dtype=np.float32)
    out = m.copy()
    out[1:, :] = np.maximum(out[1:, :], m[:-1, :])
    out[:-1, :] = np.maximum(out[:-1, :], m[1:, :])
    out[:, 1:] = np.maximum(out[:, 1:], m[:, :-1])
    out[:, :-1] = np.maximum(out[:, :-1], m[:, 1:])
    return out


def region_composite(
    base: dict[str, np.ndarray] | np.ndarray,
    layer: dict[str, np.ndarray] | np.ndarray,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    part_pattern: str = "",
    mask: np.ndarray | None = None,
    opacity: float = 1.0,
    seam_ao_strength: float = 0.30,
    seam_tint: str = "#1A1834",
    palette_snap_stops: Sequence[Any] | None = None,
    palette_snap_mix: float = 0.0,
    preserve_alpha: bool = True,
) -> dict[str, np.ndarray]:
    """Composite a per-body-part `MATERIAL` layer cleanly onto `base` with `Preserve Alpha`, seam AO & optional OKLab palette snap."""
    b = _ensure_bus(base)
    l = _ensure_bus(layer)
    H, W = b["albedo"].shape[:2]
    f = _mesh_island_fields(H, W, mesh_ctx)

    # Determine region composite weight
    if mask is not None:
        m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
        if m.ndim == 3:
            m = m[..., 0]
        if str(part_pattern or "").strip():
            m = m * resolve_part_mask(H, W, mesh_ctx, part_pattern=part_pattern)
    elif str(part_pattern or "").strip():
        m = resolve_part_mask(H, W, mesh_ctx, part_pattern=part_pattern)
    elif "region_mask" in l and isinstance(l["region_mask"], np.ndarray) and l["region_mask"].shape == (H, W):
        m = np.clip(l["region_mask"].astype(np.float32), 0.0, 1.0)
    else:
        m = np.clip(l["albedo"][..., 3].astype(np.float32), 0.0, 1.0)

    m = np.clip(m * float(np.clip(opacity, 0.0, 1.0)), 0.0, 1.0).astype(np.float32)

    layer_alb = l["albedo"].copy()
    if palette_snap_stops and float(palette_snap_mix) > 1e-4:
        hex_pal = _stops_to_hex_list(palette_snap_stops)
        if len(hex_pal) >= 2:
            snapped = palette_posterize(
                layer_alb,
                hex_pal,
                space="oklab",
                posterize_alpha=False,
                blend=float(np.clip(palette_snap_mix, 0.0, 1.0)),
            )
            layer_alb[..., :3] = snapped[..., :3]

    # 1px exterior contact seam AO where `layer` meets `base` inside the same island/atlas
    seam_band = np.clip(_dilate1(m) - m, 0.0, 1.0) * f["island_mask"]
    st_rgb = np.array(hex_to_srgb(seam_tint), dtype=np.float32)[None, None, :]
    base_rgb = b["albedo"][..., :3].copy()
    if float(seam_ao_strength) > 1e-4 and np.any(seam_band > 0.01):
        sw = (seam_band * float(np.clip(seam_ao_strength, 0.0, 0.85)))[..., None]
        base_rgb = base_rgb * (1.0 - sw) + (base_rgb * (0.35 + 0.65 * st_rgb)) * sw

    m3 = m[..., None]
    out_alb = np.empty_like(b["albedo"])
    out_alb[..., :3] = np.clip(base_rgb * (1.0 - m3) + layer_alb[..., :3] * m3, 0.0, 1.0)
    if preserve_alpha:
        out_alb[..., 3] = b["albedo"][..., 3]
    else:
        out_alb[..., 3] = np.clip(b["albedo"][..., 3] * (1.0 - m) + layer_alb[..., 3] * m, 0.0, 1.0)

    out_h = (b["height"] * (1.0 - m) + l["height"] * m - seam_band * 0.08 * float(seam_ao_strength)).astype(np.float32)
    out_h = np.clip(out_h, 0.0, 1.0)
    out_r = np.clip(b["roughness"] * (1.0 - m) + l["roughness"] * m, 0.04, 1.0).astype(np.float32)
    out_met = np.clip(b["metallic"] * (1.0 - m) + l["metallic"] * m, 0.0, 1.0).astype(np.float32)
    out_ao = np.clip(b["ao"] * (1.0 - m) + l["ao"] * m - seam_band * 0.25 * float(seam_ao_strength), 0.10, 1.0).astype(np.float32)
    out_sss = np.clip(b["sss"] * (1.0 - m) + l["sss"] * m, 0.0, 1.0).astype(np.float32)
    out_por = np.clip(b["porosity"] * (1.0 - m) + l["porosity"] * m, 0.0, 1.0).astype(np.float32)

    out_n = b["normal"] * (1.0 - m3) + l["normal"] * m3
    # Re-normalize blended tangent-space normal
    nv = out_n * 2.0 - 1.0
    nlen = np.maximum(1e-6, np.linalg.norm(nv, axis=-1, keepdims=True))
    out_n = np.clip((nv / nlen) * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)

    out_em = b["emissive"] * (1.0 - m[..., None]) + l["emissive"] * m[..., None]
    out_em = np.clip(out_em, 0.0, 1.0).astype(np.float32)

    res: dict[str, np.ndarray] = {
        "__bus__": np.array([1.0], dtype=np.float32),
        "albedo": out_alb,
        "rgba": out_alb,
        "height": out_h,
        "normal": out_n,
        "roughness": out_r,
        "metallic": out_met,
        "ao": out_ao,
        "emissive": out_em,
        "sss": out_sss,
        "porosity": out_por,
        "blend_mask": m,
        "seam_mask": seam_band.astype(np.float32),
    }
    return res


def build_uv_region_overlay(mesh_ctx: MeshBakeContext) -> np.ndarray:
    """Render a crisp color-coded (H, W, 4) uint8 UV-Island & Anatomical Region Overlay for SyncCraft Studio."""
    H, W = mesh_ctx.height, mesh_ctx.width
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[..., :3] = (14, 18, 28)
    out[..., 3] = 255
    zone_colors: dict[str, tuple[int, int, int]] = {
        "head": (129, 140, 248),
        "neck": (96, 165, 250),
        "torso": (52, 211, 153),
        "arms": (251, 191, 36),
        "legs": (244, 114, 182),
        "tail": (192, 132, 252),
        "wings": (56, 189, 248),
        "horns": (245, 158, 11),
        "accessories": (45, 212, 191),
        "body": (148, 163, 184),
    }
    for isl in mesh_ctx.islands:
        r, g, b = zone_colors.get(isl.zone, (148, 163, 184))
        # Subtle deterministic brightness step per element index so adjacent islands separate crisply
        tint = 0.78 + 0.22 * ((isl.elem_idx * 29 + len(isl.face) * 7) % 5) / 4.0
        col = (int(min(255, r * tint)), int(min(255, g * tint)), int(min(255, b * tint)))
        y0, y1 = isl.y0, min(H, isl.y0 + isl.h)
        x0, x1 = isl.x0, min(W, isl.x0 + isl.w)
        out[y0:y1, x0:x1, 0] = col[0]
        out[y0:y1, x0:x1, 1] = col[1]
        out[y0:y1, x0:x1, 2] = col[2]
        # Bright 1px island border highlight so UV boundaries & `Inner Area` read instantly
        if y1 > y0 and x1 > x0:
            edge = (min(255, col[0] + 55), min(255, col[1] + 55), min(255, col[2] + 55))
            out[y0, x0:x1, :3] = edge
            out[y1 - 1, x0:x1, :3] = edge
            out[y0:y1, x0, :3] = edge
            out[y0:y1, x1 - 1, :3] = edge
    return out


__all__ = [
    "build_uv_region_overlay",
    "region_base_fill",
    "region_bevel_and_ao",
    "region_composite",
    "region_restrained_noise",
    "resolve_part_mask",
    "uv_island_coords",
]
