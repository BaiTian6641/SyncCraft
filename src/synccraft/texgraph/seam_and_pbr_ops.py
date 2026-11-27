"""Seamless UV-Geometric Edge Solving, Triplanar Projection & Precomputed Shadow Special PBR (`TexGraph v10`).

Commercial industry-standard operators inspired by Unreal Engine 5.7 Triplanar, Substance Painter
Baked Lighting & Seam Healing, Marmoset Toolbag 5, and Connected Block-Edge Systems:
1. ``triplanar_project``     : Seamless 3D World/Local normal-blended (or UE 5.7 Bayer-dithered
                               single-axis) triplanar projection across scalar, RGBA, or full
                               12-channel `MATERIAL` buses — ZERO UV seams around 90-degree cube
                               edges, with world-to-tangent normal re-projection!
2. ``uv_seam_heal``          : Geometric 3D UV-Edge Adjacency seam stitcher — smoothly blends
                               albedo, height, roughness, and world-reoriented tangent normals
                               across matched 3D shared edges (`north<->up`, `south<->east`,
                               contacting parent/child cuboids).
3. ``block_edge_transition`` : Connected multi-block edge & corner transition solver supporting
                               4-bit NESW neighbor masks (`bevel_trim`, `moss_overhang`,
                               `worn_stone`, `metal_frame`, `glow_circuit`) for seamless modular
                               blocks, walls, pillars & floors.
4. ``precompute_shadow_pbr`` : SyncCraft Precomputed Shadow & Special PBR Pipeline — synthesizes
                               3D ray-traced `baked_shadow`, hemisphere `bent_normal` IBL, and
                               ray-traced `thickness` SSS translucency into BOTH separated
                               12-channel PBR maps AND optional stylized pre-baked albedo!
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .generators_noise import bayer_matrix, fbm_noise, uv_grid
from .pbr import height_to_normal_map
from .pipeline import MATERIAL_BUS_PORTS, _ensure_bus, is_material_bus, material_make
from .pixel_art_ops import hex_to_srgb


def _sample_tile_nearest_or_bilinear(arr: np.ndarray, u: np.ndarray, v: np.ndarray, *, pixel_snap: bool = False) -> np.ndarray:
    """Fast vectorized toroidal 2D sampling on scalar (H,W) or multi-channel (H,W,C) arrays."""
    H, W = arr.shape[:2]
    fu = (u % 1.0) * W
    fv = (v % 1.0) * H
    if pixel_snap:
        ix = np.floor(fu).astype(np.int32) % W
        iy = np.floor(fv).astype(np.int32) % H
        return arr[iy, ix].astype(np.float32)

    x0 = np.floor(fu - 0.5).astype(np.int32)
    y0 = np.floor(fv - 0.5).astype(np.int32)
    fx = (fu - 0.5) - x0.astype(np.float32)
    fy = (fv - 0.5) - y0.astype(np.float32)
    x0m = x0 % W
    x1m = (x0 + 1) % W
    y0m = y0 % H
    y1m = (y0 + 1) % H

    if arr.ndim == 2:
        c00 = arr[y0m, x0m]
        c10 = arr[y0m, x1m]
        c01 = arr[y1m, x0m]
        c11 = arr[y1m, x1m]
        top = c00 * (1.0 - fx) + c10 * fx
        bot = c01 * (1.0 - fx) + c11 * fx
        return (top * (1.0 - fy) + bot * fy).astype(np.float32)

    fx_e = fx[..., None]
    fy_e = fy[..., None]
    c00 = arr[y0m, x0m]
    c10 = arr[y0m, x1m]
    c01 = arr[y1m, x0m]
    c11 = arr[y1m, x1m]
    top = c00 * (1.0 - fx_e) + c10 * fx_e
    bot = c01 * (1.0 - fx_e) + c11 * fx_e
    return (top * (1.0 - fy_e) + bot * fy_e).astype(np.float32)


def triplanar_project(
    src: dict[str, np.ndarray] | np.ndarray,
    mesh_context: Any | None,
    *,
    scale: float = 0.08,
    sharpness: float = 6.0,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    space: str = "world",  # "world" | "local_norm"
    dither_blend: bool = False,  # UE 5.7 crisp single-axis stochastic/Bayer triplanar selection!
    pixel_snap: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray] | np.ndarray:
    """Seamless 3D Triplanar Projection for scalar, RGBA, or full 12-channel `MATERIAL` buses!

    Projects a 2D seamless tile around all three orthogonal 3D axes (`XY`, `ZY`, `XZ`) weighted
    by `|world_normal|^sharpness` (or UE 5.7 Bayer-dithered single-axis selection to preserve
    razor-sharp pixel clusters on curved/angled cuboids!). When projecting a `MATERIAL` bus,
    also re-orients projected tangent-space normals cleanly into each face's 3D tangent frame!
    """
    if mesh_context is None:
        return src

    mc = mesh_context
    H, W = mc.height, mc.width
    pos = mc.world_pos_norm if space == "local_norm" else mc.world_pos
    sc = float(scale)
    ox, oy, oz = (float(offset[0]), float(offset[1]), float(offset[2]))
    px = pos[..., 0] * sc + ox
    py = pos[..., 1] * sc + oy
    pz = pos[..., 2] * sc + oz

    wn = np.asarray(mc.world_normal, dtype=np.float32)
    abs_n = np.abs(wn)
    shp = max(0.5, float(sharpness))
    w_raw = np.power(np.maximum(abs_n, 1e-4), shp)
    w_sum = np.maximum(1e-6, np.sum(w_raw, axis=-1, keepdims=True))
    w_tri = (w_raw / w_sum).astype(np.float32)  # (H, W, 3): [wx (ZY plane), wy (XZ plane), wz (XY plane)]

    if dither_blend:
        # Unreal Engine 5.7 Dithered Triplanar: pick ONE crisp axis per pixel via ordered Bayer 4x4 threshold!
        bm = bayer_matrix(H, W, size=4)
        wx, wy = w_tri[..., 0], w_tri[..., 1]
        pick_x = (bm < wx).astype(np.float32)
        pick_y = ((bm >= wx) & (bm < (wx + wy))).astype(np.float32)
        pick_z = (1.0 - pick_x - pick_y).astype(np.float32)
        w_tri = np.stack((pick_x, pick_y, pick_z), axis=-1).astype(np.float32)

    # Consistent right-handed planar UVs (V flips -Y so patterns stay right-side-up on vertical walls!)
    u_x, v_x = pz, -py  # ZY plane (East/West faces, normal along ±X)
    u_y, v_y = px, pz   # XZ plane (Up/Down faces, normal along ±Y)
    u_z, v_z = px, -py  # XY plane (North/South faces, normal along ±Z)

    def _proj_array(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        sx = _sample_tile_nearest_or_bilinear(a, u_x, v_x, pixel_snap=pixel_snap)
        sy = _sample_tile_nearest_or_bilinear(a, u_y, v_y, pixel_snap=pixel_snap)
        sz = _sample_tile_nearest_or_bilinear(a, u_z, v_z, pixel_snap=pixel_snap)
        if a.ndim == 2:
            res = sx * w_tri[..., 0] + sy * w_tri[..., 1] + sz * w_tri[..., 2]
        else:
            res = sx * w_tri[..., 0:1] + sy * w_tri[..., 1:2] + sz * w_tri[..., 2:3]
        return np.clip(res, 0.0, 1.0).astype(np.float32)

    if not is_material_bus(src):
        out_arr = _proj_array(np.asarray(src, dtype=np.float32))
        return out_arr

    # Full 12-channel MATERIAL Bus triplanar projection + world-to-tangent normal reprojection!
    bus = _ensure_bus(src, tileable=True)
    out_bus: dict[str, np.ndarray] = {"__bus__": np.array([1.0], dtype=np.float32)}
    for k in MATERIAL_BUS_PORTS:
        if k not in bus:
            continue
        if k in ("shadow", "bent_normal", "thickness"):
            # Preserve true 3D mesh-space baked shadow / bent normal / thickness without re-tiling!
            out_bus[k] = bus[k].copy()
        elif k == "normal":
            # Project tangent normals on each axis into World Space (RNM-style swizzle), blend, then transform to face TBN!
            nx_m = _sample_tile_nearest_or_bilinear(bus["normal"], u_x, v_x, pixel_snap=pixel_snap) * 2.0 - 1.0
            ny_m = _sample_tile_nearest_or_bilinear(bus["normal"], u_y, v_y, pixel_snap=pixel_snap) * 2.0 - 1.0
            nz_m = _sample_tile_nearest_or_bilinear(bus["normal"], u_z, v_z, pixel_snap=pixel_snap) * 2.0 - 1.0

            sgn_x = np.where(wn[..., 0:1] >= 0.0, 1.0, -1.0)
            sgn_y = np.where(wn[..., 1:2] >= 0.0, 1.0, -1.0)
            sgn_z = np.where(wn[..., 2:3] >= 0.0, 1.0, -1.0)

            # World-space perturbed normals per projection axis
            wn_x = np.concatenate((nx_m[..., 2:3] * sgn_x, nx_m[..., 1:2], nx_m[..., 0:1]), axis=-1)
            wn_y = np.concatenate((ny_m[..., 0:1], ny_m[..., 2:3] * sgn_y, -ny_m[..., 1:2]), axis=-1)
            wn_z = np.concatenate((nz_m[..., 0:1], nz_m[..., 1:2], nz_m[..., 2:3] * sgn_z), axis=-1)

            w_blend = wn_x * w_tri[..., 0:1] + wn_y * w_tri[..., 1:2] + wn_z * w_tri[..., 2:3]
            w_blend = w_blend / np.maximum(1e-6, np.linalg.norm(w_blend, axis=-1, keepdims=True))

            # Project world normal cleanly onto face orthonormal tangent frame (T, -B_down, N)
            wt = getattr(mc, "world_tangent", None)
            wb = getattr(mc, "world_bitangent", None)
            if wt is not None and wb is not None and wt.shape == wn.shape:
                tx = np.sum(w_blend * wt, axis=-1)
                # OpenGL green-up (+Y up = -world_bitangent_down)
                ty = np.sum(w_blend * (-wb), axis=-1)
                tz = np.maximum(0.05, np.sum(w_blend * wn, axis=-1))
                tn = np.stack((tx, ty, tz), axis=-1)
                tn = tn / np.maximum(1e-6, np.linalg.norm(tn, axis=-1, keepdims=True))
                out_bus["normal"] = np.clip(tn * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
            else:
                out_bus["normal"] = _proj_array(bus["normal"])
        else:
            out_bus[k] = _proj_array(bus[k])
    return out_bus


def uv_seam_heal(
    src: dict[str, np.ndarray] | np.ndarray,
    mesh_context: Any | None,
    *,
    strength: float = 0.72,
    radius_px: int = 2,
    heal_normals: bool = True,
) -> dict[str, np.ndarray] | np.ndarray:
    """Geometric 3D UV-Edge Adjacency Seam Healer (Substance Painter / Marmoset Seam-Stitching Parity!).

    Uses `MeshBakeContext`'s solved 3D edge pairs (`uv_seam_src_yx`, `uv_seam_dst_yx`, `uv_seam_weight`)
    to smoothly match values across connected 3D cube edges (`north<->up`, `south<->east`, contacting
    cuboids), then feathers smoothly `radius_px` inward inside each UV island without blurring interiors!
    """
    if mesh_context is None:
        return src
    mc = mesh_context
    src_yx = getattr(mc, "uv_seam_src_yx", None)
    dst_yx = getattr(mc, "uv_seam_dst_yx", None)
    seam_w = getattr(mc, "uv_seam_weight", None)
    if src_yx is None or dst_yx is None or len(src_yx) == 0:
        return src

    s_val = float(np.clip(strength, 0.0, 1.0))
    if s_val <= 1e-4:
        return src

    sy, sx = src_yx[:, 0], src_yx[:, 1]
    dy, dx = dst_yx[:, 0], dst_yx[:, 1]
    w_pair = np.clip(seam_w * s_val * 0.5, 0.0, 0.5).astype(np.float32)
    rad = max(0, min(4, int(radius_px)))
    isl_mask = mc.island_mask
    eid = mc.elem_id_map
    fid = mc.face_id_map

    def _heal_field(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32).copy()
        orig = a.copy()
        # Step 1: Symmetric 3D edge stitch right across matched UV island borders
        # (2 symmetric passes when strength >= 0.65 gives crisp sub-pixel seam convergence
        # even for distance-attenuated corner pairs, while staying idempotent at w=0.5!)
        n_iters = 2 if s_val >= 0.65 else 1
        for _ in range(n_iters):
            prev = a.copy()
            if a.ndim == 2:
                v_src = prev[sy, sx]
                v_dst = prev[dy, dx]
                a[sy, sx] = v_src * (1.0 - w_pair) + v_dst * w_pair
            else:
                wp = w_pair[:, None]
                v_src = prev[sy, sx]
                v_dst = prev[dy, dx]
                a[sy, sx] = v_src * (1.0 - wp) + v_dst * wp

        # Step 2: Feather the stitched border delta gently `rad` px inward within the SAME face island!
        if rad >= 1:
            delta = a - orig
            cur = delta.copy()
            for step in range(rad):
                decay = 0.42 / float(step + 1)
                acc = np.zeros_like(cur)
                cnt = np.zeros(a.shape[:2], dtype=np.float32)
                for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    sh_c = np.roll(np.roll(cur, oy, axis=0), ox, axis=1)
                    sh_e = np.roll(np.roll(eid, oy, axis=0), ox, axis=1)
                    sh_f = np.roll(np.roll(fid, oy, axis=0), ox, axis=1)
                    same_face = isl_mask & (sh_e == eid) & (sh_f == fid)
                    m_f = same_face.astype(np.float32)
                    acc += sh_c * (m_f if a.ndim == 2 else m_f[..., None])
                    cnt += m_f
                avg = acc / np.maximum(1.0, cnt if a.ndim == 2 else cnt[..., None])
                cur = np.where((np.abs(cur) > 1e-5) if a.ndim == 2 else (np.max(np.abs(cur), axis=-1, keepdims=True) > 1e-5), cur, avg * decay)
            # Only apply interior feather to non-border island texels so border matches remain crisp!
            border_m = np.zeros(a.shape[:2], dtype=bool)
            border_m[sy, sx] = True
            inner = isl_mask & (~border_m)
            if a.ndim == 2:
                a[inner] = np.clip(orig[inner] + cur[inner], 0.0, 1.0)
            else:
                a[inner] = np.clip(orig[inner] + cur[inner], 0.0, 1.0)
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    if not is_material_bus(src):
        return _heal_field(np.asarray(src, dtype=np.float32))

    bus = _ensure_bus(src, tileable=False)
    out = dict(bus)
    for k in ("albedo", "height", "roughness", "metallic", "ao", "sss"):
        if k in out:
            out[k] = _heal_field(out[k])
    if heal_normals and "normal" in out:
        wt = getattr(mc, "world_tangent", None)
        wb = getattr(mc, "world_bitangent", None)
        wn = mc.world_normal
        if wt is not None and wb is not None and wt.shape == wn.shape:
            # Convert tangent normal -> world space, heal across 3D seam, convert back to each island's tangent frame!
            tn = out["normal"] * 2.0 - 1.0
            w_nrm = (
                tn[..., 0:1] * wt
                + tn[..., 1:2] * (-wb)
                + tn[..., 2:3] * wn
            )
            w_nrm = w_nrm / np.maximum(1e-6, np.linalg.norm(w_nrm, axis=-1, keepdims=True))
            w_healed = _heal_field(w_nrm * 0.5 + 0.5) * 2.0 - 1.0
            w_healed = w_healed / np.maximum(1e-6, np.linalg.norm(w_healed, axis=-1, keepdims=True))
            tx = np.sum(w_healed * wt, axis=-1)
            ty = np.sum(w_healed * (-wb), axis=-1)
            tz = np.maximum(0.05, np.sum(w_healed * wn, axis=-1))
            tn_out = np.stack((tx, ty, tz), axis=-1)
            tn_out = tn_out / np.maximum(1e-6, np.linalg.norm(tn_out, axis=-1, keepdims=True))
            out["normal"] = np.clip(tn_out * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
        else:
            out["normal"] = _heal_field(out["normal"])
    return out


def _parse_neighbor_mask(neighbors: str) -> tuple[bool, bool, bool, bool]:
    """Return ``(conn_N, conn_E, conn_S, conn_W)`` booleans for connected block-edge solving."""
    s = str(neighbors).strip().lower()
    presets = {
        "exposed_all": (False, False, False, False),
        "connected_all": (True, True, True, True),
        "top_exposed": (False, True, True, True),
        "bottom_exposed": (True, True, False, True),
        "pillar_mid": (True, False, True, False),
        "beam_mid": (False, True, False, True),
        "corner_tl": (False, True, True, False),
        "corner_tr": (False, False, True, True),
        "corner_bl": (True, True, False, False),
        "corner_br": (True, False, False, True),
    }
    if s in presets:
        return presets[s]
    if len(s) == 4 and all(c in "01" for c in s):
        return (s[0] == "1", s[1] == "1", s[2] == "1", s[3] == "1")
    return (False, False, False, False)


def block_edge_transition(
    height: int,
    width: int,
    *,
    base: dict[str, np.ndarray] | np.ndarray | None = None,
    neighbors: str = "exposed_all",  # "exposed_all"|"top_exposed"|"pillar_mid"|"beam_mid"|"corner_tl"|...|"NESW" 4-bit
    style: str = "bevel_trim",  # "bevel_trim"|"moss_overhang"|"worn_stone"|"metal_frame"|"glow_circuit"
    border_width: float = 0.14,
    corner_strength: float = 0.65,
    transition_color: str = "#3E4B5E",
    highlight_color: str = "#B6C8D6",
    moss_color: str = "#4E823E",
    glow_color: str = "#38D9F5",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Connected Block-Edge & Corner Transition Solver (CTM / Autotile / Multi-Block Material Continuity!).

    Given a 4-neighbor connection state (`NESW`), automatically places crisp stepped bevel trims,
    overhanging organic moss drapes, chipped worn stone chamfers, riveted metal frames, or glowing
    arcane conduits STRICTLY along exposed block borders & corners while keeping connected edges
    100% seamless with adjacent blocks!

    Returns a complete `MATERIAL` bus when `base` is provided, PLUS auxiliary ports
    `{"edge_mask", "corner_mask", "transition_h"}`!
    """
    H, W = int(height), int(width)
    u, v = uv_grid(H, W)
    conn_n, conn_e, conn_s, conn_w = _parse_neighbor_mask(neighbors)
    bw = max(0.03, float(np.clip(border_width, 0.03, 0.40)))
    n_org = fbm_noise(H, W, scale=7.0, octaves=3, seed=seed, tileable=True)
    # Organic scalloped edge modulation (zero modulation at connected edges so tile boundaries stay seamless!)
    org_w = (n_org - 0.5) * bw * (0.85 if style in ("moss_overhang", "worn_stone") else 0.18)

    d_n = np.ones((H, W), dtype=np.float32) if conn_n else np.clip((v - org_w) / bw, 0.0, 1.0)
    d_s = np.ones((H, W), dtype=np.float32) if conn_s else np.clip(((1.0 - v) - org_w) / bw, 0.0, 1.0)
    d_w = np.ones((H, W), dtype=np.float32) if conn_w else np.clip((u - org_w) / bw, 0.0, 1.0)
    d_e = np.ones((H, W), dtype=np.float32) if conn_e else np.clip(((1.0 - u) - org_w) / bw, 0.0, 1.0)

    e_n = (1.0 - d_n)
    e_s = (1.0 - d_s)
    e_w = (1.0 - d_w)
    e_e = (1.0 - d_e)

    # Exposed border mask (1.0 along any unconnected side, 0.0 on connected sides & interior!)
    edge_raw = np.maximum(np.maximum(e_n, e_s), np.maximum(e_w, e_e)).astype(np.float32)
    # Corner mask highlights intersections of two exposed edges
    c_tl = np.minimum(e_n, e_w)
    c_tr = np.minimum(e_n, e_e)
    c_bl = np.minimum(e_s, e_w)
    c_br = np.minimum(e_s, e_e)
    corner_m = np.clip(np.maximum(np.maximum(c_tl, c_tr), np.maximum(c_bl, c_br)) * float(corner_strength) * 1.4, 0.0, 1.0).astype(np.float32)

    # Directional top-left lit rim vs bottom-right shadow rim on exposed edges
    lit_rim = np.maximum(e_n, e_w)
    shd_rim = np.maximum(e_s, e_e)

    st = style.lower()
    if st == "moss_overhang":
        # Moss drapes primarily down from exposed top edges with soft organic tufts!
        moss_e = np.clip((e_n * 1.35 + 0.35 * np.maximum(e_w, e_e)) * (0.65 + 0.70 * n_org), 0.0, 1.0)
        edge_m = (moss_e > 0.32).astype(np.float32)
        trans_h = np.clip(0.45 + 0.42 * moss_e, 0.0, 1.0).astype(np.float32)
    elif st == "glow_circuit":
        # Crisp recessed glowing conduit inset slightly from exposed edges
        ring_m = ((edge_raw > 0.25) & (edge_raw < 0.72)).astype(np.float32)
        edge_m = (edge_raw > 0.12).astype(np.float32)
        trans_h = np.clip(0.50 + 0.30 * edge_m - 0.35 * ring_m, 0.0, 1.0).astype(np.float32)
    else:
        edge_m = (edge_raw > 0.18).astype(np.float32)
        trans_h = np.clip(0.50 + 0.32 * edge_m + 0.18 * corner_m - 0.14 * shd_rim, 0.0, 1.0).astype(np.float32)

    c_trim = np.array(hex_to_srgb(transition_color), dtype=np.float32)[None, None, :]
    c_hi = np.array(hex_to_srgb(highlight_color), dtype=np.float32)[None, None, :]
    c_moss = np.array(hex_to_srgb(moss_color), dtype=np.float32)[None, None, :]
    c_glow = np.array(hex_to_srgb(glow_color), dtype=np.float32)[None, None, :]

    if base is None:
        base_bus = material_make(
            np.concatenate((np.broadcast_to(c_trim, (H, W, 3)).copy(), np.ones((H, W, 1), dtype=np.float32)), axis=-1),
            height=trans_h,
            roughness=0.62,
            metallic=0.08,
            ao=1.0,
        )
    else:
        base_bus = _ensure_bus(base, tileable=False)

    rgb = base_bus["albedo"][..., :3].copy()
    h_out = base_bus["height"].copy()
    r_out = base_bus["roughness"].copy()
    m_out = base_bus["metallic"].copy()
    ao_out = base_bus["ao"].copy()
    em_out = base_bus["emissive"].copy()

    # Cast 1 px cool contact shadow just inside the exposed trim border!
    inner_sh = np.clip(((edge_raw > 0.02) & (edge_m < 0.5)).astype(np.float32) * 0.55, 0.0, 1.0)
    rgb *= (1.0 - inner_sh[..., None] * 0.35)
    ao_out = np.clip(ao_out * (1.0 - inner_sh * 0.38), 0.10, 1.0)

    if st == "moss_overhang":
        shade_m = (0.72 + 0.45 * n_org - 0.25 * shd_rim)[..., None]
        moss_col = np.clip(c_moss * shade_m, 0.0, 1.0)
        rgb = rgb * (1.0 - edge_m[..., None]) + moss_col * edge_m[..., None]
        h_out = np.clip(h_out * (1.0 - edge_m) + trans_h * edge_m, 0.0, 1.0)
        r_out = np.clip(r_out * (1.0 - edge_m) + 0.88 * edge_m, 0.08, 0.98)
        m_out = m_out * (1.0 - edge_m)
    elif st == "metal_frame":
        metal_col = np.clip(c_trim * (0.78 + 0.32 * lit_rim - 0.24 * shd_rim)[..., None] + c_hi * (lit_rim * 0.35 + corner_m * 0.35)[..., None], 0.0, 1.0)
        rgb = rgb * (1.0 - edge_m[..., None]) + metal_col * edge_m[..., None]
        h_out = np.clip(h_out * (1.0 - edge_m) + trans_h * edge_m, 0.0, 1.0)
        r_out = np.clip(r_out * (1.0 - edge_m) + 0.24 * edge_m, 0.08, 0.96)
        m_out = np.clip(m_out * (1.0 - edge_m) + 0.88 * edge_m, 0.0, 1.0)
    elif st == "glow_circuit":
        ring_m = ((edge_raw > 0.26) & (edge_raw < 0.68)).astype(np.float32)
        frame_col = np.clip(c_trim * (0.75 + 0.28 * lit_rim)[..., None], 0.0, 1.0)
        rgb = rgb * (1.0 - edge_m[..., None]) + frame_col * edge_m[..., None]
        rgb = rgb * (1.0 - ring_m[..., None]) + c_glow * ring_m[..., None]
        em_out[..., :3] = np.clip(em_out[..., :3] * (1.0 - ring_m[..., None]) + c_glow * ring_m[..., None], 0.0, 1.0)
        em_out[..., 3] = np.maximum(em_out[..., 3], ring_m)
        h_out = np.clip(h_out * (1.0 - edge_m) + trans_h * edge_m, 0.0, 1.0)
        r_out = np.clip(r_out * (1.0 - ring_m) + 0.16 * ring_m, 0.08, 0.96)
    elif st == "worn_stone":
        chip_col = np.clip(c_hi * 0.45 + rgb * 0.55, 0.0, 1.0)
        w_wear = edge_m * np.clip(0.35 + 0.65 * lit_rim - 0.30 * shd_rim, 0.0, 1.0)
        rgb = rgb * (1.0 - w_wear[..., None] * 0.72) + chip_col * (w_wear[..., None] * 0.72)
        rgb = rgb * (1.0 - (edge_m * shd_rim)[..., None] * 0.35)
        h_out = np.clip(h_out - edge_m * (0.12 + 0.15 * n_org), 0.02, 1.0)
    else:  # "bevel_trim"
        trim_col = np.clip(c_trim * (0.82 + 0.26 * lit_rim - 0.26 * shd_rim)[..., None] + c_hi * (lit_rim * 0.28 + corner_m * 0.22)[..., None], 0.0, 1.0)
        rgb = rgb * (1.0 - edge_m[..., None] * 0.85) + trim_col * (edge_m[..., None] * 0.85)
        h_out = np.clip(h_out * (1.0 - edge_m) + trans_h * edge_m, 0.0, 1.0)

    albedo = np.concatenate((np.clip(rgb, 0.0, 1.0), base_bus["albedo"][..., 3:4]), axis=-1).astype(np.float32)
    out = material_make(
        albedo,
        height=h_out,
        normal=height_to_normal_map(h_out, strength=2.5, tileable=False),
        roughness=r_out,
        metallic=m_out,
        ao=ao_out,
        shadow=base_bus.get("shadow", 1.0),
        bent_normal=base_bus.get("bent_normal"),
        thickness=base_bus.get("thickness", 0.5),
        emissive=em_out,
        sss=base_bus.get("sss", 0.0),
        porosity=base_bus.get("porosity", 0.0),
    )
    out["edge_mask"] = edge_m.astype(np.float32)
    out["corner_mask"] = corner_m.astype(np.float32)
    out["transition_h"] = trans_h.astype(np.float32)
    return out


def precompute_shadow_pbr(
    mat: dict[str, np.ndarray] | np.ndarray,
    mesh_context: Any | None,
    *,
    preset: str = "studio_hero",  # "studio_hero" | "dramatic_boss" | "golden_rim" | "overcast_soft"
    shadow_strength: float = 0.62,
    bent_normal_strength: float = 0.48,
    thickness_sss_strength: float = 0.42,
    bake_into_albedo: float = 0.45,
    sss_color: str = "#EA723D",
    shadow_tint: str = "#1F2442",
    sky_tint: str = "#B6D2F5",
    shadow_posterize_steps: int = 0,
) -> dict[str, np.ndarray]:
    """SyncCraft Precomputed Shadow & Special PBR Pipeline (`MATERIAL -> 12-Channel MATERIAL` Bus!).

    Integrates true 3D ray-traced `baked_shadow`, 3D hemisphere `bent_normal` sky/ground IBL,
    and ray-traced `thickness`-weighted SSS translucency directly into the unified 12-channel
    PBR bus (`shadow`, `bent_normal`, `thickness`, `ao`, `sss`, `albedo`).
    - When `bake_into_albedo = 0.0`: keeps `albedo` pure PBR and stores all precomputed lighting
      strictly in the separated `shadow`, `bent_normal`, `thickness`, and `ao` maps!
    - When `bake_into_albedo > 0.0` (e.g. `0.40–0.75`): gracefully bakes hue-shifted cool cast
      shadows, bent-normal sky lift, and thin-edge SSS warmth into `albedo` for vanilla Minecraft
      and stylized Blockbench rendering while STILL preserving all 12 separated PBR channels!
    """
    bus = _ensure_bus(mat, tileable=False)
    H, W = bus["albedo"].shape[:2]

    # Lighting preset multipliers
    preset_cfg = {
        "studio_hero": (1.0, 1.0, 1.0),
        "dramatic_boss": (1.30, 0.85, 1.15),
        "golden_rim": (1.10, 1.15, 1.35),
        "overcast_soft": (0.65, 1.25, 0.85),
    }
    p_sh, p_bn, p_sss = preset_cfg.get(str(preset).lower(), (1.0, 1.0, 1.0))

    if mesh_context is not None and getattr(mesh_context, "baked_shadow", None) is not None:
        sh_map = np.clip(np.asarray(mesh_context.baked_shadow, dtype=np.float32), 0.0, 1.0)
        bn_vec = np.asarray(mesh_context.bent_normal, dtype=np.float32)
        thick_map = np.clip(np.asarray(mesh_context.thickness, dtype=np.float32), 0.0, 1.0)
        ao_mesh = np.clip(np.asarray(mesh_context.baked_ao, dtype=np.float32), 0.0, 1.0)
    else:
        # 2D Heightfield fallback (for blocks & 2D items without a 3D .bbmodel!)
        from .pixel_composer_ops import heightmap_shadow
        sh_map = heightmap_shadow(bus["height"], light_angle_deg=-135.0, elevation_deg=35.0, max_steps=6, strength=0.75, tileable=False)
        n0 = bus["normal"] * 2.0 - 1.0
        bn_vec = n0 / np.maximum(1e-6, np.linalg.norm(n0, axis=-1, keepdims=True))
        thick_map = np.clip(bus["height"], 0.15, 1.0).astype(np.float32)
        ao_mesh = bus["ao"]

    if int(shadow_posterize_steps) >= 2:
        st = float(int(shadow_posterize_steps))
        sh_map = np.clip(np.rint(sh_map * st) / st, 0.0, 1.0).astype(np.float32)

    bn_rgb = np.clip(bn_vec * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    ao_combined = np.clip(bus["ao"] * (0.45 + 0.55 * ao_mesh), 0.10, 1.0).astype(np.float32)
    # Physically grounded SSS boost on thin parts (wings, ears, fins, snouts -> low thickness!)
    thinness = np.clip(1.0 - thick_map, 0.0, 1.0)
    sss_boost = np.clip(bus["sss"] + thinness * float(thickness_sss_strength) * p_sss * 0.45 * (1.0 - bus["metallic"]), 0.0, 1.0).astype(np.float32)

    alb = bus["albedo"].copy()
    bake_w = float(np.clip(bake_into_albedo, 0.0, 1.0))
    if bake_w > 1e-4:
        rgb = alb[..., :3]
        em_protect = np.clip(1.0 - bus["emissive"][..., 3] * 0.92, 0.08, 1.0)[..., None]

        # 1. Hue-shifted cool directional cast shadow (NEVER neutral muddy grey!)
        c_sh = np.array(hex_to_srgb(shadow_tint), dtype=np.float32)[None, None, :]
        sh_amt = np.clip((1.0 - sh_map) * float(shadow_strength) * p_sh * bake_w, 0.0, 0.82)[..., None] * em_protect
        rgb = rgb * (1.0 - sh_amt * 0.62) + (rgb * c_sh * 1.45) * (sh_amt * 0.62)

        # 2. Bent-normal unoccluded sky zenith lift vs cool ground bounce
        sky_up = np.clip(bn_vec[..., 1] * 0.5 + 0.5, 0.0, 1.0)[..., None]
        c_sky = np.array(hex_to_srgb(sky_tint), dtype=np.float32)[None, None, :]
        bn_w = float(bent_normal_strength) * p_bn * bake_w * 0.28
        rgb = rgb * (1.0 + (sky_up - 0.45) * bn_w * 0.9) + c_sky * (sky_up * bn_w * 0.42) * (1.0 - rgb * 0.55) * em_protect

        # 3. Thin-volume warm backlit SSS glow (ears, fins, dragon wing edges!)
        c_sss = np.array(hex_to_srgb(sss_color), dtype=np.float32)[None, None, :]
        sss_w = (thinness * np.clip(0.35 + 0.65 * sh_map, 0.0, 1.0) * float(thickness_sss_strength) * p_sss * bake_w * 0.34 * (1.0 - bus["metallic"]))[..., None] * em_protect
        rgb = rgb * (1.0 - sss_w * 0.45) + np.clip(rgb * 0.55 + c_sss * 0.85, 0.0, 1.0) * (sss_w * 0.65)

        alb[..., :3] = np.clip(rgb, 0.0, 1.0)

    out = material_make(
        alb,
        height=bus["height"],
        normal=bus["normal"],
        roughness=bus["roughness"],
        metallic=bus["metallic"],
        ao=ao_combined,
        shadow=sh_map,
        bent_normal=bn_rgb,
        thickness=thick_map,
        emissive=bus["emissive"],
        sss=sss_boost,
        porosity=bus["porosity"],
    )
    return out
