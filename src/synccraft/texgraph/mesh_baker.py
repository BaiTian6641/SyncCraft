"""Fast vectorized 3D Mesh-to-UV Attribute Baker for Blockbench 5 .bbmodel & SyncCraft Scenes.

Bakes aligned (H, W) UV-space 3D channels directly from geometry so procedural pixel-art
node graphs are 100% 3D-aware (zero seams, real multi-box Ambient Occlusion, exposed-edge
bevels, joint-crease cavities, directional lighting normals, and anatomical zone masks):
  - ``island_mask``   (H, W)    bool     active UV texels
  - ``world_pos``     (H, W, 3) float32  physical 3D rest position (for seamless 3D noise!)
  - ``world_pos_norm``(H, W, 3) float32  normalized [0, 1] bounding-box position
  - ``world_normal``  (H, W, 3) float32  unit 3D outward surface normal
  - ``baked_ao``      (H, W)    float32  multi-box hemisphere/SDF ambient occlusion [0..1]
  - ``edge_bevel``    (H, W)    float32  exposed perimeter bevel mask [0..1]
  - ``joint_cavity``  (H, W)    float32  neighboring-box contact/crease cavity mask [0..1]
  - ``local_uv``      (H, W, 2) float32  per-face local (u, v) in [0, 1]^2
  - ``zone_masks``    dict[str, (H, W) float32]  anatomical partition masks
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any

import numpy as np

from synccraft.shading.uv_packer import (
    ANATOMICAL_ZONES,
    FACES,
    _extract_bbmodel_bone_paths,
    classify_anatomical_zone,
)

# Face tangent basis in box-local space: (tangent_right (+u), tangent_down (+v), outward_normal)
# Matches Blockbench 5 / Minecraft standard cuboid UV mapping
_FACE_BASIS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = {
    "north": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "south": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "east": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "west": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "down": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}


def _euler_xyz_mat(deg: tuple[float, float, float] | list[float]) -> np.ndarray:
    """Blockbench 5 / SyncCraft ZYX extrinsic (= XYZ intrinsic: Rz @ Ry @ Rx) rotation matrix."""
    rx, ry, rz = (math.radians(float(c)) for c in deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    my = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    mz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return mz @ my @ mx


@dataclass
class BakedIsland:
    elem_idx: int
    elem_name: str
    bone_name: str
    zone: str
    face: str
    x0: int
    y0: int
    w: int
    h: int


@dataclass
class MeshBakeContext:
    """Aligned UV-space 3D geometry maps baked from a .bbmodel or SyncCraft Scene."""

    width: int
    height: int
    padding: int
    island_mask: np.ndarray
    world_pos: np.ndarray
    world_pos_norm: np.ndarray
    world_normal: np.ndarray
    baked_ao: np.ndarray
    edge_bevel: np.ndarray
    joint_cavity: np.ndarray
    local_uv: np.ndarray
    elem_id_map: np.ndarray
    zone_masks: dict[str, np.ndarray]
    face_id_map: np.ndarray = field(default_factory=lambda: np.full((1, 1), -1, dtype=np.int8))
    elem_center_x_map: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    wing_span_u: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    wing_trail_v: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    face_is_down: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    wing_bay_u: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    wing_bay_v: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    wing_bay_mask: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=np.float32))
    islands: list[BakedIsland] = field(default_factory=list)
    elem_names: list[str] = field(default_factory=list)
    elem_bones: list[str] = field(default_factory=list)
    elem_zones: list[str] = field(default_factory=list)
    elem_centers_x: list[float] = field(default_factory=list)
    # TexGraph v10: Precomputed 3D Ray-Traced Shadow, Bent Normal, Thickness, Curvature & UV-Edge Solver!
    baked_shadow: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=np.float32))
    bent_normal: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float32))
    thickness: np.ndarray = field(default_factory=lambda: np.full((1, 1), 0.5, dtype=np.float32))
    curvature_3d: np.ndarray = field(default_factory=lambda: np.full((1, 1), 0.5, dtype=np.float32))
    world_tangent: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float32))
    world_bitangent: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), dtype=np.float32))
    uv_seam_mask: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    uv_seam_src_yx: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    uv_seam_dst_yx: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    uv_seam_weight: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    # TexGraph v11 (Pixel Composer Per-UV / Per-Body-Part Restraint Channels):
    island_edge_px: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    island_size_min: np.ndarray = field(default_factory=lambda: np.full((1, 1), 16.0, dtype=np.float32))
    island_size_wh: np.ndarray = field(default_factory=lambda: np.full((1, 1, 2), 16.0, dtype=np.float32))
    island_inner_area: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=np.float32))
    face_is_up: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.float32))
    face_is_side: np.ndarray = field(default_factory=lambda: np.ones((1, 1), dtype=np.float32))
    # TexGraph v15.1: Content-addressed SHA-256 geometry fingerprint & bake telemetry!
    content_fingerprint: str = ""
    bake_cache_hit: bool = False
    bake_ms: float = 0.0

    def match_elements_mask(self, pattern: str) -> np.ndarray:
        """Return a float32 (H, W) mask matching elements whose bone/name OR mirrored short-name matches ``pattern``."""
        rx = re.compile(pattern, re.IGNORECASE)
        hits = []
        for i, (en, bn) in enumerate(zip(self.elem_names, self.elem_bones)):
            short_en = en.split("__")[-1]
            if rx.search(f"{bn}/{en}") or rx.search(f"{bn}/{short_en}") or rx.search(short_en):
                hits.append(i)
        if not hits:
            return np.zeros((self.height, self.width), dtype=np.float32)
        return np.isin(self.elem_id_map, hits).astype(np.float32)

    def outward_lateral_face_mask(self, pattern: str) -> np.ndarray:
        """Return 1.0 ONLY on the outward-facing lateral face ('west' when x<0, 'east' when x>0) of matching boxes.

        Guarantees eyes (both `eye_l` and mirrored `head_face_r__eye_l`) paint strictly on their
        visible outward lateral face while keeping the other 5 faces clean dark orbital sockets!
        """
        em = self.match_elements_mask(pattern) > 0.5
        if not np.any(em):
            return np.zeros((self.height, self.width), dtype=np.float32)
        east_idx = FACES.index("east")
        west_idx = FACES.index("west")
        is_left = self.elem_center_x_map < 0.0
        outward = np.where(is_left, self.face_id_map == west_idx, self.face_id_map == east_idx)
        return (em & outward).astype(np.float32)

    def face_filter_mask(self, faces: str) -> np.ndarray:
        """Return 1.0 on UV texels belonging to the comma-separated cuboid face(s), e.g. 'up,down'."""
        wanted = [FACES.index(f.strip().lower()) for f in faces.split(",") if f.strip().lower() in FACES]
        if not wanted:
            return np.zeros((self.height, self.width), dtype=np.float32)
        return np.isin(self.face_id_map, wanted).astype(np.float32)

    def zone_mask(self, zone: str) -> np.ndarray:
        """Return float32 (H, W) mask for an anatomical zone (or comma-separated list of zones)."""
        parts = [z.strip().lower() for z in str(zone).split(",") if z.strip()]
        out = np.zeros((self.height, self.width), dtype=np.float32)
        for p in parts:
            if p in self.zone_masks:
                out = np.maximum(out, self.zone_masks[p])
        return out

    def pad_rgba(self, rgba: np.ndarray, *, pad: int | None = None) -> np.ndarray:
        """Clamp-bleed every baked UV island (scalar or multi-channel) into its padding border."""
        p_px = self.padding if pad is None else int(pad)
        out = np.asarray(rgba, dtype=np.float32).copy()
        if p_px <= 0:
            return out
        H, W = out.shape[:2]
        for isl in self.islands:
            x, y, w, h = isl.x0, isl.y0, isl.w, isl.h
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W or y + h > H:
                continue
            patch = out[y : y + h, x : x + w]
            for d in range(1, p_px + 1):
                if y - d >= 0:
                    out[y - d, x : x + w] = patch[0]
                if y + h - 1 + d < H:
                    out[y + h - 1 + d, x : x + w] = patch[-1]
                yl = max(0, y - d)
                yh = min(H, y + h + d)
                if x - d >= 0:
                    out[yl:yh, x - d] = out[yl:yh, x]
                if x + w - 1 + d < W:
                    out[yl:yh, x + w - 1 + d] = out[yl:yh, x + w - 1]
        return out


def _extract_bbmodel_obbs(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute world-space OBB (center, rotation, half_extents) for every .bbmodel element."""
    groups_by_uuid = {g["uuid"]: g for g in doc.get("groups", []) if isinstance(g, dict)}
    elem_group_chain: dict[str, list[dict[str, Any]]] = {}

    def _walk(nodes: list[Any], chain: list[dict[str, Any]]) -> None:
        for node in nodes:
            if isinstance(node, str):
                elem_group_chain[node] = list(chain)
            elif isinstance(node, dict):
                uid = node.get("uuid", "")
                g = groups_by_uuid.get(uid) or node
                _walk(node.get("children") or [], chain + [g])

    _walk(doc.get("outliner") or [], [])
    elem_paths = _extract_bbmodel_bone_paths(doc)

    obbs: list[dict[str, Any]] = []
    for idx, el in enumerate(doc.get("elements", [])):
        uid = el.get("uuid", f"el_{idx}")
        ename = el.get("name", f"box_{idx}")
        bname, ancs = elem_paths.get(uid, ("root", ()))
        f_pt = np.asarray(el.get("from", [0.0, 0.0, 0.0]), dtype=np.float64)
        t_pt = np.asarray(el.get("to", [0.0, 0.0, 0.0]), dtype=np.float64)
        inflate = float(el.get("inflate", 0.0))
        half = np.maximum(0.0, 0.5 * np.abs(t_pt - f_pt) + inflate)
        center = 0.5 * (f_pt + t_pt)

        # Apply element local rotation around element origin
        el_rot = el.get("rotation") or [0.0, 0.0, 0.0]
        el_orig = np.asarray(el.get("origin", center), dtype=np.float64)
        R = _euler_xyz_mat(el_rot)
        center = el_orig + R @ (center - el_orig)

        # Apply ancestor group rotations from innermost child bone outward to root
        for g in reversed(elem_group_chain.get(uid, [])):
            g_rot = g.get("rotation") or [0.0, 0.0, 0.0]
            if any(abs(float(c)) > 1e-8 for c in g_rot):
                g_orig = np.asarray(g.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
                Rg = _euler_xyz_mat(g_rot)
                center = g_orig + Rg @ (center - g_orig)
                R = Rg @ R

        # Compute conservative world-space AABB for fast broad-phase AO culling
        abs_r = np.abs(R)
        aabb_half = abs_r @ half
        obbs.append(
            {
                "idx": idx,
                "name": ename,
                "bone": bname,
                "ancs": ancs,
                "center": center,
                "R": R,
                "RT": R.T,
                "half": half,
                "aabb_min": center - aabb_half,
                "aabb_max": center + aabb_half,
                "el": el,
            }
        )
    return obbs


def _obb_signed_distance(pts: np.ndarray, obb: dict[str, Any]) -> np.ndarray:
    """Vectorized exact Euclidean signed distance from world points (N, 3) to an OBB."""
    local = (pts - obb["center"][None, :]) @ obb["R"]  # equivalent to R.T @ (p - c)
    q = np.abs(local) - obb["half"][None, :]
    out_d = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    in_d = np.minimum(np.maximum(np.maximum(q[:, 0], q[:, 1]), q[:, 2]), 0.0)
    return out_d + in_d


def _obb_signed_distance_batch(
    pts: np.ndarray,
    centers: np.ndarray,
    Rs: np.ndarray,
    halfs: np.ndarray,
) -> np.ndarray:
    """Vectorized signed distance from (N, 3) world points to K OBBs simultaneously -> (N, K) float32."""
    diff = pts[:, None, :] - centers[None, :, :]  # (N, K, 3)
    local = np.einsum("nki,kij->nkj", diff, Rs)
    q = np.abs(local) - halfs[None, :, :]
    pos = np.maximum(q, 0.0)
    out_d = np.sqrt(np.sum(pos * pos, axis=-1))
    in_d = np.minimum(np.max(q, axis=-1), 0.0)
    return out_d + in_d


def _ray_obb_hits(
    origins: np.ndarray,
    ray_dir: np.ndarray,
    obb: dict[str, Any],
    *,
    t_min: float = 0.08,
    t_max: float = 1e4,
) -> np.ndarray:
    """Vectorized Kay-Kajiya slab Ray-vs-OBB intersection test.

    Returns boolean mask (N,) where ray `origins[i] + t * ray_dir` hits ``obb`` for `t in [t_min, t_max]`.
    """
    loc_o = (origins - obb["center"][None, :]) @ obb["R"]  # (N, 3)
    loc_d = obb["RT"] @ ray_dir  # (3,)
    half = obb["half"]  # (3,)
    t_enter = np.full(origins.shape[0], float(t_min), dtype=np.float64)
    t_exit = np.full(origins.shape[0], float(t_max), dtype=np.float64)
    valid = np.ones(origins.shape[0], dtype=bool)
    for ax in range(3):
        da = float(loc_d[ax])
        ha = float(half[ax])
        oa = loc_o[:, ax]
        if abs(da) < 1e-8:
            valid &= (oa >= -ha) & (oa <= ha)
        else:
            inv_d = 1.0 / da
            t0 = (-ha - oa) * inv_d
            t1 = (ha - oa) * inv_d
            lo = np.minimum(t0, t1)
            hi = np.maximum(t0, t1)
            t_enter = np.maximum(t_enter, lo)
            t_exit = np.minimum(t_exit, hi)
    return valid & (t_exit >= t_enter) & (t_exit >= t_min)


def _ray_obb_hits_batch(
    origins: np.ndarray,
    ray_dir: np.ndarray,
    centers: np.ndarray,
    Rs: np.ndarray,
    RTs: np.ndarray,
    halfs: np.ndarray,
    *,
    t_min: float = 0.12,
    t_max: float = 180.0,
) -> np.ndarray:
    """Vectorized Kay-Kajiya slab Ray-vs-K-OBBs test for (N, 3) rays -> (N,) bool any-hit mask!"""
    n_pts = origins.shape[0]
    k_box = centers.shape[0]
    if n_pts == 0 or k_box == 0:
        return np.zeros(n_pts, dtype=bool)
    diff = origins[:, None, :] - centers[None, :, :]  # (N, K, 3)
    loc_o = np.einsum("nki,kij->nkj", diff, Rs)  # (N, K, 3)
    loc_d = np.einsum("kij,j->ki", RTs, ray_dir)  # (K, 3)

    t_enter = np.full((n_pts, k_box), float(t_min), dtype=np.float32)
    t_exit = np.full((n_pts, k_box), float(t_max), dtype=np.float32)
    valid = np.ones((n_pts, k_box), dtype=bool)
    for ax in range(3):
        da = loc_d[:, ax]  # (K,)
        ha = halfs[:, ax]  # (K,)
        oa = loc_o[:, :, ax]  # (N, K)
        par = np.abs(da) < 1e-7
        if np.any(par):
            p_in = (oa >= -ha[None, :]) & (oa <= ha[None, :])
            valid &= np.where(par[None, :], p_in, True)
        safe_d = np.where(par, 1.0, da)
        inv_d = (1.0 / safe_d).astype(np.float32)
        t0 = (-ha[None, :] - oa) * inv_d[None, :]
        t1 = (ha[None, :] - oa) * inv_d[None, :]
        lo = np.minimum(t0, t1)
        hi = np.maximum(t0, t1)
        if np.any(par):
            t_enter = np.where(par[None, :], t_enter, np.maximum(t_enter, lo))
            t_exit = np.where(par[None, :], t_exit, np.minimum(t_exit, hi))
        else:
            t_enter = np.maximum(t_enter, lo)
            t_exit = np.minimum(t_exit, hi)
    hits_nk = valid & (t_exit >= t_enter) & (t_exit >= float(t_min))
    return np.any(hits_nk, axis=1)


def _solve_uv_edge_adjacency(
    islands: list[BakedIsland],
    world_pos: np.ndarray,
    world_normal: np.ndarray,
    H: int,
    W: int,
    *,
    seam_tol: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve 3D Geometric UV-Edge Adjacency across distinct UV islands via vectorized spatial cell batching!

    Pairs every border texel `(y, x)` on island A with its nearest 3D-coincident border texel
    on a DIFFERENT island B (`north<->up`, `south<->east`, contacting parent/child boxes!).
    Returns `(src_yx, dst_yx, weight, seam_mask)`.
    """
    seam_mask = np.zeros((H, W), dtype=np.float32)
    if len(islands) < 2:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            seam_mask,
        )

    by_list: list[np.ndarray] = []
    bx_list: list[np.ndarray] = []
    bi_list: list[np.ndarray] = []
    for isl_id, isl in enumerate(islands):
        x0, y0, w, h = isl.x0, isl.y0, isl.w, isl.h
        if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
            continue
        if w <= 2 or h <= 2:
            yy, xx = np.mgrid[y0 : y0 + h, x0 : x0 + w]
            ys, xs = yy.ravel().astype(np.int32), xx.ravel().astype(np.int32)
        else:
            top_x = np.arange(x0, x0 + w, dtype=np.int32)
            top_y = np.full(w, y0, dtype=np.int32)
            bot_x = top_x
            bot_y = np.full(w, y0 + h - 1, dtype=np.int32)
            mid_y = np.arange(y0 + 1, y0 + h - 1, dtype=np.int32)
            lft_x = np.full(mid_y.size, x0, dtype=np.int32)
            rgt_x = np.full(mid_y.size, x0 + w - 1, dtype=np.int32)
            ys = np.concatenate((top_y, bot_y, mid_y, mid_y))
            xs = np.concatenate((top_x, bot_x, lft_x, rgt_x))
        by_list.append(ys)
        bx_list.append(xs)
        bi_list.append(np.full(ys.size, isl_id, dtype=np.int32))

    if not by_list:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            seam_mask,
        )

    by = np.concatenate(by_list)
    bx = np.concatenate(bx_list)
    bi = np.concatenate(bi_list)
    bpos = world_pos[by, bx].astype(np.float32)  # (M, 3)

    cell = max(0.35, float(seam_tol))
    tol = max(1e-4, float(seam_tol))
    tol2 = tol * tol
    cells = np.floor(bpos / cell).astype(np.int32)
    grid_lists: dict[tuple[int, int, int], list[int]] = {}
    cx_col = cells[:, 0].tolist()
    cy_col = cells[:, 1].tolist()
    cz_col = cells[:, 2].tolist()
    for idx_b in range(by.size):
        key = (cx_col[idx_b], cy_col[idx_b], cz_col[idx_b])
        lst = grid_lists.get(key)
        if lst is None:
            grid_lists[key] = [idx_b]
        else:
            lst.append(idx_b)
    grid_arr: dict[tuple[int, int, int], np.ndarray] = {k: np.asarray(v, dtype=np.int32) for k, v in grid_lists.items()}

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    w_chunks: list[np.ndarray] = []
    offsets = [(ox, oy, oz) for ox in (-1, 0, 1) for oy in (-1, 0, 1) for oz in (-1, 0, 1)]

    for (cx, cy, cz), idx_a in grid_arr.items():
        nb_parts = []
        for ox, oy, oz in offsets:
            b = grid_arr.get((cx + ox, cy + oy, cz + oz))
            if b is not None:
                nb_parts.append(b)
        if not nb_parts:
            continue
        idx_b = nb_parts[0] if len(nb_parts) == 1 else np.concatenate(nb_parts)
        ia = bi[idx_a]
        ib = bi[idx_b]
        if ia.size == 0 or ib.size == 0:
            continue
        # Fast exit if all points in this 3x3x3 neighborhood belong to a single island
        if int(ia.min()) == int(ia.max()) and int(ib.min()) == int(ib.max()) and int(ia[0]) == int(ib[0]):
            continue
        pa = bpos[idx_a]  # (Na, 3)
        pb = bpos[idx_b]  # (Nb, 3)
        diff = pa[:, None, :] - pb[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)  # (Na, Nb)
        d2[ia[:, None] == ib[None, :]] = 1e9
        best_col = np.argmin(d2, axis=1)
        best_d2 = d2[np.arange(idx_a.size), best_col]
        hit = best_d2 < tol2
        if not np.any(hit):
            continue
        hit_a = idx_a[hit]
        hit_b = idx_b[best_col[hit]]
        w_vals = np.maximum(0.15, 1.0 - np.sqrt(best_d2[hit]) / tol).astype(np.float32)
        src_yx = np.stack((by[hit_a], bx[hit_a]), axis=-1)
        dst_yx = np.stack((by[hit_b], bx[hit_b]), axis=-1)
        src_chunks.append(src_yx)
        dst_chunks.append(dst_yx)
        w_chunks.append(w_vals)
        np.maximum.at(seam_mask, (src_yx[:, 0], src_yx[:, 1]), w_vals)

    if not src_chunks:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            seam_mask,
        )
    return (
        np.concatenate(src_chunks, axis=0).astype(np.int32),
        np.concatenate(dst_chunks, axis=0).astype(np.int32),
        np.concatenate(w_chunks, axis=0).astype(np.float32),
        seam_mask,
    )


# ---------------------------------------------------------------------------
# Content-Addressed SHA-256 Mesh Bake Cache (In-Memory LRU + Disk `.npz`)
# ---------------------------------------------------------------------------
_MESH_BAKE_CACHE_VERSION = "mesh_bake_v15_1"
_MESH_BAKE_LOCK = threading.RLock()
_MESH_BAKE_LRU: "OrderedDict[str, MeshBakeContext]" = OrderedDict()
_MESH_BAKE_LRU_MAX = 16


def _mesh_cache_dir() -> Path:
    return Path.home() / ".synccraft" / "cache" / "mesh_bakes"


def clear_mesh_bake_cache(*, clear_disk: bool = False) -> None:
    """Clear in-memory (and optionally disk) content-addressed mesh bake caches."""
    with _MESH_BAKE_LOCK:
        _MESH_BAKE_LRU.clear()
        if clear_disk:
            cdir = _mesh_cache_dir()
            if cdir.is_dir():
                for fp in cdir.glob("*.npz"):
                    fp.unlink(missing_ok=True)
                for fp in cdir.glob("*.meta.json"):
                    fp.unlink(missing_ok=True)


def mesh_bake_cache_stats() -> dict[str, Any]:
    with _MESH_BAKE_LOCK:
        cdir = _mesh_cache_dir()
        disk_n = len(list(cdir.glob("*.npz"))) if cdir.is_dir() else 0
        return {"memory_entries": len(_MESH_BAKE_LRU), "disk_entries": disk_n, "cache_dir": str(cdir)}


def compute_bbmodel_fingerprint(
    bbmodel_doc: dict[str, Any],
    *,
    padding: int,
    ao_radius: float,
    ao_samples: int,
    zone_overrides: dict[str, str] | None,
    shadow_light_dir: tuple[float, float, float],
    bake_shadows: bool,
    solve_uv_seams: bool,
) -> str:
    """Cryptographic SHA-256 geometry+UV+bake-parameter fingerprint for 1-to-1 cache integrity."""
    res = bbmodel_doc.get("resolution") or {}
    elems_sig = []
    for el in bbmodel_doc.get("elements") or []:
        faces = el.get("faces") or {}
        f_uvs = {k: faces[k].get("uv") for k in FACES if isinstance(faces.get(k), dict) and faces[k].get("uv")}
        elems_sig.append(
            [
                el.get("uuid"),
                el.get("name"),
                el.get("from"),
                el.get("to"),
                el.get("origin"),
                el.get("rotation"),
                el.get("inflate", 0.0),
                f_uvs,
            ]
        )
    groups_sig = [
        [g.get("uuid"), g.get("name"), g.get("origin"), g.get("rotation")]
        for g in (bbmodel_doc.get("groups") or [])
        if isinstance(g, dict)
    ]
    payload = {
        "v": _MESH_BAKE_CACHE_VERSION,
        "w": int(res.get("width", 1)),
        "h": int(res.get("height", 1)),
        "elements": elems_sig,
        "groups": groups_sig,
        "outliner": bbmodel_doc.get("outliner") or [],
        "pad": int(padding),
        "ao_r": round(float(ao_radius), 4),
        "ao_s": int(ao_samples),
        "zov": sorted((zone_overrides or {}).items()),
        "sdir": [round(float(c), 4) for c in shadow_light_dir],
        "bsh": bool(bake_shadows),
        "suv": bool(solve_uv_seams),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def _save_mesh_bake_disk(fp: str, ctx: MeshBakeContext) -> None:
    try:
        cdir = _mesh_cache_dir()
        cdir.mkdir(parents=True, exist_ok=True)
        npz_path = cdir / f"{fp}.npz"
        meta_path = cdir / f"{fp}.meta.json"
        arrays: dict[str, np.ndarray] = {
            "island_mask": ctx.island_mask,
            "world_pos": ctx.world_pos,
            "world_pos_norm": ctx.world_pos_norm,
            "world_normal": ctx.world_normal,
            "baked_ao": ctx.baked_ao,
            "edge_bevel": ctx.edge_bevel,
            "joint_cavity": ctx.joint_cavity,
            "local_uv": ctx.local_uv,
            "elem_id_map": ctx.elem_id_map,
            "face_id_map": ctx.face_id_map,
            "elem_center_x_map": ctx.elem_center_x_map,
            "wing_span_u": ctx.wing_span_u,
            "wing_trail_v": ctx.wing_trail_v,
            "face_is_down": ctx.face_is_down,
            "wing_bay_u": ctx.wing_bay_u,
            "wing_bay_v": ctx.wing_bay_v,
            "wing_bay_mask": ctx.wing_bay_mask,
            "baked_shadow": ctx.baked_shadow,
            "bent_normal": ctx.bent_normal,
            "thickness": ctx.thickness,
            "curvature_3d": ctx.curvature_3d,
            "world_tangent": ctx.world_tangent,
            "world_bitangent": ctx.world_bitangent,
            "uv_seam_mask": ctx.uv_seam_mask,
            "uv_seam_src_yx": ctx.uv_seam_src_yx,
            "uv_seam_dst_yx": ctx.uv_seam_dst_yx,
            "uv_seam_weight": ctx.uv_seam_weight,
            "island_edge_px": ctx.island_edge_px,
            "island_size_min": ctx.island_size_min,
            "island_size_wh": ctx.island_size_wh,
            "island_inner_area": ctx.island_inner_area,
            "face_is_up": ctx.face_is_up,
            "face_is_side": ctx.face_is_side,
        }
        for z, arr in ctx.zone_masks.items():
            arrays[f"zone__{z}"] = arr
        tmp_npz = cdir / f".{fp}.{threading.get_ident()}.tmp.npz"
        np.savez(tmp_npz, **arrays)
        tmp_npz.replace(npz_path)
        meta = {
            "v": _MESH_BAKE_CACHE_VERSION,
            "fp": fp,
            "width": ctx.width,
            "height": ctx.height,
            "padding": ctx.padding,
            "islands": [
                [isl.elem_idx, isl.elem_name, isl.bone_name, isl.zone, isl.face, isl.x0, isl.y0, isl.w, isl.h]
                for isl in ctx.islands
            ],
            "elem_names": ctx.elem_names,
            "elem_bones": ctx.elem_bones,
            "elem_zones": ctx.elem_zones,
            "elem_centers_x": ctx.elem_centers_x,
        }
        tmp_meta = cdir / f".{fp}.{threading.get_ident()}.meta.tmp"
        tmp_meta.write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")
        tmp_meta.replace(meta_path)
    except Exception:
        pass


def _load_mesh_bake_disk(fp: str) -> MeshBakeContext | None:
    try:
        cdir = _mesh_cache_dir()
        npz_path = cdir / f"{fp}.npz"
        meta_path = cdir / f"{fp}.meta.json"
        if not npz_path.is_file() or not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("v") != _MESH_BAKE_CACHE_VERSION or meta.get("fp") != fp:
            return None
        t0 = time.perf_counter()
        with np.load(npz_path) as data:
            zone_masks = {z: data[f"zone__{z}"].copy() for z in ANATOMICAL_ZONES if f"zone__{z}" in data}
            islands = [
                BakedIsland(int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), int(r[5]), int(r[6]), int(r[7]), int(r[8]))
                for r in (meta.get("islands") or [])
            ]
            ctx = MeshBakeContext(
                width=int(meta["width"]),
                height=int(meta["height"]),
                padding=int(meta["padding"]),
                island_mask=data["island_mask"].copy(),
                world_pos=data["world_pos"].copy(),
                world_pos_norm=data["world_pos_norm"].copy(),
                world_normal=data["world_normal"].copy(),
                baked_ao=data["baked_ao"].copy(),
                edge_bevel=data["edge_bevel"].copy(),
                joint_cavity=data["joint_cavity"].copy(),
                local_uv=data["local_uv"].copy(),
                elem_id_map=data["elem_id_map"].copy(),
                zone_masks=zone_masks,
                face_id_map=data["face_id_map"].copy(),
                elem_center_x_map=data["elem_center_x_map"].copy(),
                wing_span_u=data["wing_span_u"].copy(),
                wing_trail_v=data["wing_trail_v"].copy(),
                face_is_down=data["face_is_down"].copy(),
                wing_bay_u=data["wing_bay_u"].copy(),
                wing_bay_v=data["wing_bay_v"].copy(),
                wing_bay_mask=data["wing_bay_mask"].copy(),
                islands=islands,
                elem_names=list(meta.get("elem_names") or []),
                elem_bones=list(meta.get("elem_bones") or []),
                elem_zones=list(meta.get("elem_zones") or []),
                elem_centers_x=[float(x) for x in (meta.get("elem_centers_x") or [])],
                baked_shadow=data["baked_shadow"].copy(),
                bent_normal=data["bent_normal"].copy(),
                thickness=data["thickness"].copy(),
                curvature_3d=data["curvature_3d"].copy(),
                world_tangent=data["world_tangent"].copy(),
                world_bitangent=data["world_bitangent"].copy(),
                uv_seam_mask=data["uv_seam_mask"].copy(),
                uv_seam_src_yx=data["uv_seam_src_yx"].copy(),
                uv_seam_dst_yx=data["uv_seam_dst_yx"].copy(),
                uv_seam_weight=data["uv_seam_weight"].copy(),
                island_edge_px=data["island_edge_px"].copy(),
                island_size_min=data["island_size_min"].copy(),
                island_size_wh=data["island_size_wh"].copy(),
                island_inner_area=data["island_inner_area"].copy(),
                face_is_up=data["face_is_up"].copy(),
                face_is_side=data["face_is_side"].copy(),
                content_fingerprint=fp,
                bake_cache_hit=True,
                bake_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            )
            return ctx
    except Exception:
        return None


def bake_bbmodel_maps(
    bbmodel_doc: dict[str, Any],
    *,
    padding: int = 1,
    ao_radius: float = 4.5,
    ao_samples: int = 3,
    zone_overrides: dict[str, str] | None = None,
    shadow_light_dir: tuple[float, float, float] = (-0.42, 0.80, 0.42),
    bake_shadows: bool = True,
    solve_uv_seams: bool = True,
    use_cache: bool = True,
) -> MeshBakeContext:
    """Bake 3D world position, normal, tangent frame, multi-box AO, ray-traced shadow, bent normal, thickness, curvature & UV-edge adjacency."""
    t_bake_start = time.perf_counter()
    fp = compute_bbmodel_fingerprint(
        bbmodel_doc,
        padding=padding,
        ao_radius=ao_radius,
        ao_samples=ao_samples,
        zone_overrides=zone_overrides,
        shadow_light_dir=shadow_light_dir,
        bake_shadows=bake_shadows,
        solve_uv_seams=solve_uv_seams,
    )
    if use_cache:
        with _MESH_BAKE_LOCK:
            hit = _MESH_BAKE_LRU.get(fp)
            if hit is not None:
                _MESH_BAKE_LRU.move_to_end(fp)
                hit.bake_cache_hit = True
                return hit
        disk_hit = _load_mesh_bake_disk(fp)
        if disk_hit is not None:
            with _MESH_BAKE_LOCK:
                _MESH_BAKE_LRU[fp] = disk_hit
                while len(_MESH_BAKE_LRU) > _MESH_BAKE_LRU_MAX:
                    _MESH_BAKE_LRU.popitem(last=False)
            return disk_hit

    res = bbmodel_doc.get("resolution") or {}
    W = max(1, int(res.get("width", 1)))
    H = max(1, int(res.get("height", 1)))
    obbs = _extract_bbmodel_obbs(bbmodel_doc)

    island_mask = np.zeros((H, W), dtype=bool)
    world_pos = np.zeros((H, W, 3), dtype=np.float32)
    world_normal = np.zeros((H, W, 3), dtype=np.float32)
    world_normal[..., 1] = 1.0
    world_tangent = np.zeros((H, W, 3), dtype=np.float32)
    world_tangent[..., 0] = 1.0
    world_bitangent = np.zeros((H, W, 3), dtype=np.float32)
    world_bitangent[..., 2] = 1.0
    baked_ao = np.ones((H, W), dtype=np.float32)
    baked_shadow = np.ones((H, W), dtype=np.float32)
    bent_normal = np.zeros((H, W, 3), dtype=np.float32)
    bent_normal[..., 1] = 1.0
    thickness = np.full((H, W), 0.5, dtype=np.float32)
    curvature_3d = np.full((H, W), 0.5, dtype=np.float32)
    edge_bevel = np.zeros((H, W), dtype=np.float32)
    joint_cavity = np.zeros((H, W), dtype=np.float32)
    local_uv = np.zeros((H, W, 2), dtype=np.float32)
    elem_id_map = np.full((H, W), -1, dtype=np.int32)
    face_id_map = np.full((H, W), -1, dtype=np.int8)
    elem_center_x_map = np.zeros((H, W), dtype=np.float32)
    wing_span_u = np.zeros((H, W), dtype=np.float32)
    wing_trail_v = np.zeros((H, W), dtype=np.float32)
    face_is_down = np.zeros((H, W), dtype=np.float32)
    face_is_up = np.zeros((H, W), dtype=np.float32)
    face_is_side = np.zeros((H, W), dtype=np.float32)
    island_edge_px = np.zeros((H, W), dtype=np.float32)
    island_size_min = np.zeros((H, W), dtype=np.float32)
    island_size_wh = np.zeros((H, W, 2), dtype=np.float32)
    island_inner_area = np.zeros((H, W), dtype=np.float32)
    zone_masks: dict[str, np.ndarray] = {z: np.zeros((H, W), dtype=np.float32) for z in ANATOMICAL_ZONES}

    # Normalize directional key light & build 3-ray soft penumbra cone for 3D ray-traced shadows!
    l_main = np.asarray(shadow_light_dir, dtype=np.float64)
    l_main = l_main / max(1e-8, float(np.linalg.norm(l_main)))
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(l_main[1]) > 0.92 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    l_u = np.cross(up_ref, l_main)
    l_u /= max(1e-8, float(np.linalg.norm(l_u)))
    l_v = np.cross(l_main, l_u)
    shadow_rays = (
        l_main,
        (l_main + 0.09 * l_u + 0.06 * l_v) / np.linalg.norm(l_main + 0.09 * l_u + 0.06 * l_v),
        (l_main - 0.09 * l_u - 0.06 * l_v) / np.linalg.norm(l_main - 0.09 * l_u - 0.06 * l_v),
    )

    islands: list[BakedIsland] = []
    elem_names: list[str] = []
    elem_bones: list[str] = []
    elem_zones: list[str] = []
    elem_centers_x: list[float] = []

    # Pre-stack OBB arrays in float32 once for fully vectorized AO, cavity, shadow & thickness!
    if obbs:
        all_aabb_min = np.stack([o["aabb_min"] for o in obbs], axis=0).astype(np.float32)
        all_aabb_max = np.stack([o["aabb_max"] for o in obbs], axis=0).astype(np.float32)
        all_centers = np.stack([o["center"] for o in obbs], axis=0).astype(np.float32)
        all_Rs = np.stack([o["R"] for o in obbs], axis=0).astype(np.float32)
        all_RTs = np.stack([o["RT"] for o in obbs], axis=0).astype(np.float32)
        all_halfs = np.stack([o["half"] for o in obbs], axis=0).astype(np.float32)
        all_radii = (np.linalg.norm(all_aabb_max - all_aabb_min, axis=1) * 0.5).astype(np.float32)
        all_half_spans = (np.linalg.norm(all_halfs, axis=1) * 1.4).astype(np.float32)
        scene_min = all_aabb_min.min(axis=0).astype(np.float64)
        scene_max = all_aabb_max.max(axis=0).astype(np.float64)
        scene_span = np.maximum(1e-4, scene_max - scene_min)
        diag = float(np.linalg.norm(scene_span))
        eff_ao_r = float(np.clip(ao_radius * (diag / 65.0), 1.4, 7.5))
    else:
        all_aabb_min = np.zeros((0, 3), dtype=np.float32)
        all_aabb_max = np.zeros((0, 3), dtype=np.float32)
        all_centers = np.zeros((0, 3), dtype=np.float32)
        all_Rs = np.zeros((0, 3, 3), dtype=np.float32)
        all_RTs = np.zeros((0, 3, 3), dtype=np.float32)
        all_halfs = np.zeros((0, 3), dtype=np.float32)
        all_radii = np.zeros((0,), dtype=np.float32)
        all_half_spans = np.zeros((0,), dtype=np.float32)
        scene_min = np.zeros(3)
        scene_span = np.ones(3)
        eff_ao_r = float(ao_radius)

    l_main_f32 = l_main.astype(np.float32)
    shadow_rays_f32 = tuple(r.astype(np.float32) for r in shadow_rays)
    all_idx_arr = np.arange(len(obbs), dtype=np.int32)

    # Probe distances along outward normal & grazing directions for soft multi-scale AO
    probe_dists = np.linspace(0.28 * eff_ao_r, eff_ao_r, max(2, int(ao_samples)), dtype=np.float32)
    cav_r = max(0.6, 0.35 * eff_ao_r)

    for obb in obbs:
        idx = obb["idx"]
        el = obb["el"]
        zone = classify_anatomical_zone(
            obb["bone"], obb["name"], ancestor_bones=obb["ancs"], zone_overrides=zone_overrides
        )
        elem_names.append(obb["name"])
        elem_bones.append(obb["bone"])
        elem_zones.append(zone)
        elem_centers_x.append(float(obb["center"][0]))

        # Broad-phase neighbor OBB indices within eff_ao_r of this element
        pad_min = (obb["aabb_min"] - eff_ao_r - 0.5).astype(np.float32)
        pad_max = (obb["aabb_max"] + eff_ao_r + 0.5).astype(np.float32)
        nb_mask = np.all((all_aabb_max >= pad_min) & (all_aabb_min <= pad_max), axis=1)
        nb_mask[idx] = False
        nb_indices = np.nonzero(nb_mask)[0]
        has_nb = nb_indices.size > 0
        if has_nb:
            nb_centers = all_centers[nb_indices]
            nb_Rs = all_Rs[nb_indices]
            nb_halfs = all_halfs[nb_indices]
            nb_aabb_min = all_aabb_min[nb_indices]
            nb_aabb_max = all_aabb_max[nb_indices]
            nb_spans = all_half_spans[nb_indices]

        half = obb["half"]
        R = obb["R"]
        center = obb["center"]
        faces_dict = el.get("faces") or {}

        for face in FACES:
            fdata = faces_dict.get(face)
            if not isinstance(fdata, dict) or not fdata.get("uv"):
                continue
            u0, v0, u1, v1 = (float(c) for c in fdata["uv"])
            # Skip zero-area culled faces AND mirrored slave faces (u1 < u0: master already bakes the island!)
            if u1 <= u0 + 1e-6 or v1 <= v0 + 1e-6:
                continue
            x0 = int(max(0, min(W, math.floor(u0))))
            y0 = int(max(0, min(H, math.floor(v0))))
            x1 = int(max(0, min(W, math.ceil(u1))))
            y1 = int(max(0, min(H, math.ceil(v1))))
            w_px, h_px = x1 - x0, y1 - y0
            if w_px <= 0 or h_px <= 0:
                continue

            islands.append(BakedIsland(idx, obb["name"], obb["bone"], zone, face, x0, y0, w_px, h_px))
            tr_b, td_b, n_b = (np.asarray(v, dtype=np.float64) for v in _FACE_BASIS[face])

            # Local face extents along tr_b, td_b, n_b
            half_u = float(np.dot(np.abs(tr_b), half))
            half_v = float(np.dot(np.abs(td_b), half))
            half_n = float(np.dot(np.abs(n_b), half))

            uu = (np.arange(w_px, dtype=np.float64) + 0.5) / float(w_px)
            vv = (np.arange(h_px, dtype=np.float64) + 0.5) / float(h_px)
            ug, vg = np.meshgrid(uu, vv)  # (h_px, w_px)

            loc_pts = (
                ((ug - 0.5) * (2.0 * half_u))[..., None] * tr_b
                + ((vg - 0.5) * (2.0 * half_v))[..., None] * td_b
                + half_n * n_b
            )
            w_pts = center + np.einsum("ij,...j->...i", R, loc_pts)
            w_nrm = R @ n_b
            w_nrm = w_nrm / max(1e-12, float(np.linalg.norm(w_nrm)))

            # Edge bevel proximity in texels (1.0 at outermost 1px border, smoothly dropping inward over 2.2px)
            dx_px = np.minimum(np.arange(w_px, dtype=np.float32), (w_px - 1) - np.arange(w_px, dtype=np.float32))
            dy_px = np.minimum(np.arange(h_px, dtype=np.float32), (h_px - 1) - np.arange(h_px, dtype=np.float32))
            d_border = np.minimum(dx_px[None, :], dy_px[:, None])
            bevel_raw = np.clip(1.0 - d_border / 1.8, 0.0, 1.0)

            # Vectorized multi-box Ambient Occlusion, Bent Normal, Ray-Traced Shadow, Thickness & Cavity!
            pts_flat = w_pts.reshape(-1, 3).astype(np.float32)
            n_pts = pts_flat.shape[0]
            ao_flat = np.ones(n_pts, dtype=np.float32)
            cav_flat = np.zeros(n_pts, dtype=np.float32)
            w_nrm_f32 = w_nrm.astype(np.float32)
            bn_flat = np.broadcast_to(w_nrm_f32[None, :], (n_pts, 3)).copy()
            w_tr = R @ tr_b
            w_tr = w_tr / max(1e-12, float(np.linalg.norm(w_tr)))
            w_td = R @ td_b
            w_td = w_td / max(1e-12, float(np.linalg.norm(w_td)))
            w_tr_f32 = w_tr.astype(np.float32)
            w_td_f32 = w_td.astype(np.float32)

            face_min = pts_flat.min(axis=0)
            face_max = pts_flat.max(axis=0)

            if has_nb:
                # 1. Surface proximity cavity (batched across all close neighbors in 1 NumPy op!)
                c_mask = np.all((nb_aabb_max >= face_min - cav_r) & (nb_aabb_min <= face_max + cav_r), axis=1)
                if np.any(c_mask):
                    d_mat = _obb_signed_distance_batch(pts_flat, nb_centers[c_mask], nb_Rs[c_mask], nb_halfs[c_mask])
                    min_sdf = d_mat.min(axis=1)
                    cav_flat = np.clip(1.0 - min_sdf / float(cav_r), 0.0, 1.0)

                # 2. Multi-step hemisphere AO + true 3D Bent Normal (batched per-probe AABB-culled!)
                dirs_f32 = (
                    w_nrm_f32,
                    ((w_nrm + 0.55 * w_tr) / np.linalg.norm(w_nrm + 0.55 * w_tr)).astype(np.float32),
                    ((w_nrm - 0.55 * w_tr) / np.linalg.norm(w_nrm - 0.55 * w_tr)).astype(np.float32),
                    ((w_nrm + 0.55 * w_td) / np.linalg.norm(w_nrm + 0.55 * w_td)).astype(np.float32),
                    ((w_nrm - 0.55 * w_td) / np.linalg.norm(w_nrm - 0.55 * w_td)).astype(np.float32),
                )
                occ_sum = np.zeros(n_pts, dtype=np.float32)
                bn_acc = np.zeros((n_pts, 3), dtype=np.float32)
                tot_w = 0.0
                for d_idx, dist_f in enumerate(probe_dists):
                    dist = float(dist_f)
                    step_w = 1.0 / float(d_idx + 1)
                    inv_dist = 1.0 / max(1e-4, dist)
                    for d_vec in dirs_f32:
                        shift = d_vec * dist
                        p_min = face_min + shift - dist
                        p_max = face_max + shift + dist
                        sub_m = np.all((nb_aabb_max >= p_min) & (nb_aabb_min <= p_max), axis=1)
                        if np.any(sub_m):
                            probe = pts_flat + shift[None, :]
                            d_mat = _obb_signed_distance_batch(probe, nb_centers[sub_m], nb_Rs[sub_m], nb_halfs[sub_m])
                            min_d = np.minimum(dist, d_mat.min(axis=1))
                            occ_i = np.clip(1.0 - min_d * inv_dist, 0.0, 1.0)
                            occ_sum += step_w * occ_i
                            bn_acc += (step_w * (1.0 - occ_i))[:, None] * d_vec[None, :]
                        else:
                            bn_acc += step_w * d_vec[None, :]
                        tot_w += step_w
                ao_flat = np.clip(1.0 - 0.88 * (occ_sum / max(1e-6, tot_w)), 0.14, 1.0)
                bn_acc += 0.20 * w_nrm_f32[None, :]
                bn_len = np.maximum(1e-6, np.linalg.norm(bn_acc, axis=-1, keepdims=True))
                bn_flat = bn_acc / bn_len

            # 3. True 3D Ray-vs-OBB Directional Self-Shadowing (vectorized broad-phase + batched slab test!)
            ndotl_face = float(np.dot(w_nrm, l_main))
            self_shade = float(np.clip(ndotl_face * 0.55 + 0.45, 0.12, 1.0))
            sh_flat = np.full(n_pts, self_shade, dtype=np.float32)
            if bake_shadows and ndotl_face > -0.15 and len(obbs) > 1:
                face_c = 0.5 * (face_min + face_max)
                face_r = float(np.linalg.norm(face_max - face_min) * 0.5)
                rel_c = all_centers - face_c[None, :]
                along = rel_c @ l_main_f32
                perp_vec = rel_c - along[:, None] * l_main_f32[None, :]
                perp_d = np.linalg.norm(perp_vec, axis=1)
                # Conservative penumbra-cone cylinder cull (zero false negatives!)
                cone_r = face_r + all_radii + 0.14 * np.maximum(0.0, along) + 0.5
                cand_mask = (all_idx_arr != idx) & (along > -(all_radii + face_r)) & (perp_d <= cone_r)
                cand_idx = np.nonzero(cand_mask)[0]
                if cand_idx.size > 0:
                    c_centers = all_centers[cand_idx]
                    c_Rs = all_Rs[cand_idx]
                    c_RTs = all_RTs[cand_idx]
                    c_halfs = all_halfs[cand_idx]
                    ray_orig = pts_flat + w_nrm_f32 * 0.14
                    lit_acc = np.zeros(n_pts, dtype=np.float32)
                    for r_dir in shadow_rays_f32:
                        hit_any = _ray_obb_hits_batch(ray_orig, r_dir, c_centers, c_Rs, c_RTs, c_halfs, t_min=0.12, t_max=180.0)
                        lit_acc += (~hit_any).astype(np.float32)
                    cast_vis = 0.22 + 0.78 * (lit_acc / float(len(shadow_rays_f32)))
                    sh_flat = np.clip(self_shade * cast_vis, 0.08, 1.0)

            # 4. Ray-traced internal volume thickness along -w_nrm (vectorized!)
            own_thick = 2.0 * float(half_n)
            thick_flat = np.full(n_pts, own_thick, dtype=np.float32)
            if has_nb:
                in_shift = -w_nrm_f32 * max(0.5, own_thick)
                in_min = face_min + in_shift
                in_max = face_max + in_shift
                t_mask = np.all((nb_aabb_max >= in_min) & (nb_aabb_min <= in_max), axis=1)
                if np.any(t_mask):
                    in_probe = pts_flat + in_shift[None, :]
                    d_in = _obb_signed_distance_batch(in_probe, nb_centers[t_mask], nb_Rs[t_mask], nb_halfs[t_mask])
                    inside_mat = d_in < 0.0
                    if np.any(inside_mat):
                        spans_sel = nb_spans[t_mask]
                        contrib = np.where(inside_mat, own_thick + spans_sel[None, :] * 0.5, own_thick)
                        thick_flat = np.maximum(thick_flat, contrib.max(axis=1))
            thick_ref = max(2.0, float(np.median(scene_span)) * 0.35) if obbs else 8.0
            thick_norm = np.clip(thick_flat / thick_ref, 0.02, 1.0)

            ao_grid = ao_flat.reshape(h_px, w_px).astype(np.float32)
            cav_grid = cav_flat.reshape(h_px, w_px).astype(np.float32)
            sh_grid = sh_flat.reshape(h_px, w_px).astype(np.float32)
            bn_grid = bn_flat.reshape(h_px, w_px, 3).astype(np.float32)
            th_grid = thick_norm.reshape(h_px, w_px).astype(np.float32)

            # Exposed edge bevel is suppressed where buried in a joint crease!
            exposed_bevel = (bevel_raw * np.clip(1.0 - cav_grid * 1.15, 0.0, 1.0) * ao_grid).astype(np.float32)
            curv_grid = np.clip(0.5 + 0.5 * exposed_bevel - 0.5 * cav_grid, 0.0, 1.0).astype(np.float32)

            island_mask[y0:y1, x0:x1] = True
            world_pos[y0:y1, x0:x1] = w_pts.astype(np.float32)
            world_normal[y0:y1, x0:x1] = w_nrm.astype(np.float32)
            world_tangent[y0:y1, x0:x1] = w_tr.astype(np.float32)
            world_bitangent[y0:y1, x0:x1] = w_td.astype(np.float32)
            baked_ao[y0:y1, x0:x1] = ao_grid
            baked_shadow[y0:y1, x0:x1] = sh_grid
            bent_normal[y0:y1, x0:x1] = bn_grid
            thickness[y0:y1, x0:x1] = th_grid
            curvature_3d[y0:y1, x0:x1] = curv_grid
            edge_bevel[y0:y1, x0:x1] = exposed_bevel
            joint_cavity[y0:y1, x0:x1] = cav_grid
            local_uv[y0:y1, x0:x1, 0] = ug.astype(np.float32)
            local_uv[y0:y1, x0:x1, 1] = vg.astype(np.float32)
            # Wing-aligned top/bottom consistent coordinates:
            #   wing_span_u : 0.0 at inner root -> 1.0 at outer wingtip (symmetric L/R!)
            #   wing_trail_v: 0.0 at leading spar (+Z) -> 1.0 at trailing tip (-Z) (IDENTICAL on up & down faces!)
            u_tip = (1.0 - ug) if float(center[0]) < 0.0 else ug
            v_trail = (1.0 - vg) if face == "up" else (vg if face == "down" else vg)
            wing_span_u[y0:y1, x0:x1] = u_tip.astype(np.float32)
            wing_trail_v[y0:y1, x0:x1] = v_trail.astype(np.float32)
            face_is_down[y0:y1, x0:x1] = 1.0 if face == "down" else 0.0
            face_is_up[y0:y1, x0:x1] = 1.0 if face == "up" else 0.0
            face_is_side[y0:y1, x0:x1] = 1.0 if face in ("north", "south", "east", "west") else 0.0
            min_wh = float(min(w_px, h_px))
            island_edge_px[y0:y1, x0:x1] = d_border.astype(np.float32)
            island_size_min[y0:y1, x0:x1] = min_wh
            island_size_wh[y0:y1, x0:x1, 0] = float(w_px)
            island_size_wh[y0:y1, x0:x1, 1] = float(h_px)
            # Pixel Composer `PB FX Bevel -> Inner Area`: 1.0 strictly interior (>=1px from border on islands >=4px)
            if min_wh >= 4.0:
                island_inner_area[y0:y1, x0:x1] = np.clip(d_border, 0.0, 1.0).astype(np.float32)
            else:
                island_inner_area[y0:y1, x0:x1] = 0.0
            elem_id_map[y0:y1, x0:x1] = idx
            face_id_map[y0:y1, x0:x1] = FACES.index(face)
            elem_center_x_map[y0:y1, x0:x1] = float(center[0])
            if zone in zone_masks:
                zone_masks[zone][y0:y1, x0:x1] = 1.0

    world_pos_norm = np.clip((world_pos - scene_min.astype(np.float32)) / scene_span.astype(np.float32), 0.0, 1.0)

    # 3D Wing Bay Barycentric Sector Projection (exact finger-to-finger clipping for giant bat-dragon membranes!)
    wing_bay_u = wing_span_u.copy()
    wing_bay_v = wing_trail_v.copy()
    wing_bay_mask = np.ones((H, W), dtype=np.float32)

    def _obb_endpoints(o: dict[str, Any], near_ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (proximal_end, distal_end) along an OBB's longest axis (nearest to ``near_ref`` when given)."""
        ax = int(np.argmax(o["half"]))
        axis_w = o["R"][:, ax] * float(o["half"][ax])
        p_a = o["center"] - axis_w
        p_b = o["center"] + axis_w
        if near_ref is not None:
            if float(np.linalg.norm(p_a - near_ref)) <= float(np.linalg.norm(p_b - near_ref)):
                return p_a, p_b
            return p_b, p_a
        # Proximal end is closer to body/centerline (smaller |x|, or larger z if same |x|)
        if abs(p_a[0]) < abs(p_b[0]) - 0.5 or (abs(abs(p_a[0]) - abs(p_b[0])) <= 0.5 and p_a[2] > p_b[2]):
            return p_a, p_b
        return p_b, p_a

    # Index left-wing spars (`center[0] < 0`) by short element name
    left_obbs: dict[str, dict[str, Any]] = {}
    for o in obbs:
        if float(o["center"][0]) < -0.5:
            sn = o["name"].split("__")[-1].lower()
            left_obbs[sn] = o

    def _find_obb(*cands: str) -> dict[str, Any] | None:
        for c in cands:
            if c in left_obbs:
                return left_obbs[c]
        return None

    f1_o = _find_obb("f1_spar", "f1_prox_spar")
    f1_d = _find_obb("f1_dist_spar", "f1_dist")
    f2_o = _find_obb("f2_spar", "f2_prox_spar")
    f2_d = _find_obb("f2_dist_spar", "f2_dist")
    f3_o = _find_obb("f3_spar", "f3_prox_spar")
    f3_d = _find_obb("f3_dist_spar", "f3_dist")
    f4_o = _find_obb("f4_spar")
    rad_o = _find_obb("wing_rad_core", "wing_radius")
    hum_o = _find_obb("wing_hum_core", "wing_humerus")
    carpal_o = _find_obb("wing_carpal_hub", "wing_wrist_hub")

    wrist_ref: np.ndarray | None = None
    if carpal_o is not None:
        wrist_ref = carpal_o["center"]
    elif rad_o is not None:
        _, wrist_ref = _obb_endpoints(rad_o)

    def _finger_span(prox: dict[str, Any], dist: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
        r_p, t_p = _obb_endpoints(prox, near_ref=wrist_ref)
        if dist is not None:
            _, t_d = _obb_endpoints(dist, near_ref=r_p)
            return r_p, t_d
        return r_p, t_p

    # True 4-corner finger-bay quads: (r_lead, t_lead, r_trail, t_trail) so BOTH bounding spars hug at 0.00u!
    finger_quads: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = []
    if f1_o and f2_o:
        r1, t1 = _finger_span(f1_o, f1_d)
        r2, t2 = _finger_span(f2_o, f2_d)
        finger_quads.append(("wing_bay_1", (r1, t1, r2, t2)))
    if f2_o and f3_o:
        r2, t2 = _finger_span(f2_o, f2_d)
        r3, t3 = _finger_span(f3_o, f3_d)
        finger_quads.append(("wing_bay_2", (r2, t2, r3, t3)))
    if f3_o and f4_o:
        r3, t3 = _finger_span(f3_o, f3_d)
        r4, t4 = _obb_endpoints(f4_o, near_ref=wrist_ref)
        finger_quads.append(("wing_bay_3", (r3, t3, r4, t4)))

    bay_defs: list[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = []
    if rad_o and f4_o:
        r_rad, t_rad = _obb_endpoints(rad_o)  # r_rad=elbow, t_rad=wrist
        r4, t4 = _obb_endpoints(f4_o, near_ref=wrist_ref)
        # Bay 4 spans cleanly from forearm/f4 tip back toward elbow trailing point (shared with inner sail!)
        el_trail = np.array([r_rad[0] * 0.72, max(scene_min[1] + 12.0, 0.5 * (r_rad[1] + t4[1]) - 6.0), t4[2] + 8.0], dtype=np.float64)
        bay_defs.append(("wing_bay_4", (t_rad, t4, r_rad)))
        bay_defs.append(("wing_bay_4_quad", (r_rad, t4, el_trail)))
        if hum_o:
            r_hum, t_hum = _obb_endpoints(hum_o)  # shoulder -> elbow
            flank_back = np.array([r_hum[0] * 0.55, max(scene_min[1] + 12.0, r_hum[1] - 14.0), r_hum[2] - 26.0], dtype=np.float64)
            bay_defs.append(("wing_inner_sail", (r_hum, t_hum, flank_back)))
            bay_defs.append(("wing_inner_sail_quad", (t_hum, el_trail, flank_back)))
    elif hum_o and rad_o:
        r_hum, t_hum = _obb_endpoints(hum_o)
        flank_back = np.array([r_hum[0] * 0.55, max(scene_min[1] + 12.0, r_hum[1] - 14.0), r_hum[2] - 26.0], dtype=np.float64)
        bay_defs.append(("wing_inner_sail", (r_hum, t_hum, flank_back)))

    def _tri_bary(pts_3d: np.ndarray, o_p: np.ndarray, a_p: np.ndarray, b_p: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        e1 = a_p - o_p
        e2 = b_p - o_p
        a11 = float(np.dot(e1, e1)) + 1e-8
        a12 = float(np.dot(e1, e2))
        a22 = float(np.dot(e2, e2)) + 1e-8
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-6:
            return None
        rel = pts_3d - o_p[None, :]
        d1 = rel @ e1
        d2 = rel @ e2
        s = (a22 * d1 - a12 * d2) / det
        t = (-a12 * d1 + a11 * d2) / det
        return s, t

    def _point_to_seg_dist(pts_3d: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        seg = p1 - p0
        seg_l2 = float(np.dot(seg, seg)) + 1e-8
        proj = np.clip(((pts_3d - p0[None, :]) @ seg) / seg_l2, 0.0, 1.0)
        closest = p0[None, :] + proj[:, None] * seg[None, :]
        return np.linalg.norm(pts_3d - closest, axis=-1)

    # Project finger bays onto exact 4-corner quads (r_a -> t_a -> t_b -> r_b) so BOTH spars hug at <= 0.15u!
    for bay_name, (r_a, t_a, r_b, t_b) in finger_quads:
        bay_o = left_obbs.get(bay_name)
        if bay_o is None:
            continue
        pix_m = (elem_id_map == bay_o["idx"]) & island_mask
        if not np.any(pix_m):
            continue
        pts = world_pos[pix_m].astype(np.float64)
        b1 = _tri_bary(pts, r_a, t_a, t_b)
        if b1 is None:
            continue
        s1, t1 = b1
        in_sec = (s1 >= -0.055) & (t1 >= -0.055) & ((s1 + t1) >= -0.03) & ((s1 + t1) <= 1.055)
        if float(np.linalg.norm(r_a - r_b)) > 0.35:
            b2 = _tri_bary(pts, r_b, t_b, r_a)
            if b2 is not None:
                s2, t2 = b2
                in_tri2 = (s2 >= -0.065) & (t2 >= -0.065) & ((s2 + t2) >= -0.03) & ((s2 + t2) <= 1.065)
                in_sec = in_sec | in_tri2
        # Project points onto the bay plane before segment-distance evaluation so 0.5u staggered dual-sided slabs
        # achieve 100.00% mathematically identical (u, v) and silhouette cutouts on `up` (Dorsal) and `down` (Ventral)!
        bay_n = np.cross(t_a - r_a, t_b - r_a)
        bn_len = float(np.linalg.norm(bay_n))
        if bn_len > 1e-6:
            bay_n = bay_n / bn_len
            pts_proj = pts - ((pts - r_a[None, :]) @ bay_n)[:, None] * bay_n[None, :]
            r_b_p = r_b - float(np.dot(r_b - r_a, bay_n)) * bay_n
        else:
            pts_proj = pts
            r_b_p = r_b
        # C-infinity continuous (u, v): u=0 dead on leading spar (r_a->t_a), u=1 dead on trailing spar (r_b->t_b)!
        d_lead = _point_to_seg_dist(pts_proj, r_a, t_a)
        d_trail = _point_to_seg_dist(pts_proj, r_b_p, t_b)
        u_val = np.clip(d_lead / np.maximum(1e-5, d_lead + d_trail), 0.0, 1.0)
        root_u = (1.0 - u_val)[:, None] * r_a[None, :] + u_val[:, None] * r_b_p[None, :]
        tip_u = (1.0 - u_val)[:, None] * t_a[None, :] + u_val[:, None] * t_b[None, :]
        span_u_vec = tip_u - root_u
        span_l2 = np.sum(span_u_vec * span_u_vec, axis=-1) + 1e-8
        v_val = np.clip(np.sum((pts_proj - root_u) * span_u_vec, axis=-1) / span_l2, 0.0, 1.0)
        wing_bay_u[pix_m] = u_val.astype(np.float32)
        wing_bay_v[pix_m] = v_val.astype(np.float32)
        wing_bay_mask[pix_m] = in_sec.astype(np.float32)

    # Project forearm bay 4 & inner sail onto exact 3D quad sectors (with generous -0.07 tuck into humerus/radius/flank!)
    bay_map = {k: v for k, v in bay_defs}
    for bay_name, (orig_p, tip_a, tip_b) in bay_map.items():
        if bay_name.endswith("_quad"):
            continue
        bay_o = left_obbs.get(bay_name)
        if bay_o is None:
            continue
        b_idx = bay_o["idx"]
        pix_m = (elem_id_map == b_idx) & island_mask
        if not np.any(pix_m):
            continue
        pts = world_pos[pix_m].astype(np.float64)
        b1 = _tri_bary(pts, orig_p, tip_a, tip_b)
        if b1 is None:
            continue
        s, t = b1

        quad_key = f"{bay_name}_quad"
        if quad_key in bay_map:
            o2, ta2, tb2 = bay_map[quad_key]
            b2 = _tri_bary(pts, o2, ta2, tb2)
            if b2 is not None:
                s2, t2 = b2
                in_tri1 = (s >= -0.07) & (t >= -0.07) & ((s + t) <= 1.08)
                in_tri2 = (s2 >= -0.07) & (t2 >= -0.07) & ((s2 + t2) <= 1.08)
                in_sec = in_tri1 | in_tri2
                if bay_name == "wing_inner_sail":
                    r_val = np.where(in_tri2, np.clip(s2 + t2, 0.0, 1.05), np.clip(t + 0.35 * s, 0.0, 1.05))
                    u_val = np.where(in_tri2, np.clip(t2 / np.maximum(0.05, s2 + t2), 0.0, 1.0), np.clip(1.0 - s, 0.0, 1.0))
                else:
                    r_val = np.where(in_tri1, np.clip(s + 0.35 * t, 0.0, 1.05), np.clip(s2 + t2, 0.0, 1.05))
                    u_val = np.where(in_tri1, np.clip(t / np.maximum(0.05, s + t), 0.0, 1.0), np.clip(0.50 + 0.50 * (t2 / np.maximum(0.05, s2 + t2)), 0.0, 1.0))
                wing_bay_u[pix_m] = np.clip(u_val, 0.0, 1.0).astype(np.float32)
                wing_bay_v[pix_m] = np.clip(r_val, 0.0, 1.0).astype(np.float32)
                wing_bay_mask[pix_m] = in_sec.astype(np.float32)
                continue

        r_sum = s + t
        in_sec = (s >= -0.055) & (t >= -0.055) & (r_sum >= -0.02) & (r_sum <= 1.055)
        u_val = np.clip(t / np.maximum(0.02, r_sum), 0.0, 1.0)
        v_val = np.clip(r_sum, 0.0, 1.0)
        wing_bay_u[pix_m] = u_val.astype(np.float32)
        wing_bay_v[pix_m] = v_val.astype(np.float32)
        wing_bay_mask[pix_m] = in_sec.astype(np.float32)

    # 3D Geometric UV-Edge Adjacency Solver (maps shared 3D edges across distinct UV islands!)
    if solve_uv_seams and len(islands) >= 2:
        seam_tol = float(np.clip(float(np.linalg.norm(scene_span)) * 0.022, 0.55, 1.45)) if obbs else 0.85
        uv_seam_src_yx, uv_seam_dst_yx, uv_seam_weight, uv_seam_mask = _solve_uv_edge_adjacency(
            islands, world_pos, world_normal, H, W, seam_tol=seam_tol
        )
    else:
        uv_seam_src_yx = np.zeros((0, 2), dtype=np.int32)
        uv_seam_dst_yx = np.zeros((0, 2), dtype=np.int32)
        uv_seam_weight = np.zeros((0,), dtype=np.float32)
        uv_seam_mask = np.zeros((H, W), dtype=np.float32)

    ctx = MeshBakeContext(
        width=W,
        height=H,
        padding=padding,
        island_mask=island_mask,
        world_pos=world_pos,
        world_pos_norm=world_pos_norm,
        world_normal=world_normal,
        baked_ao=baked_ao,
        edge_bevel=edge_bevel,
        joint_cavity=joint_cavity,
        local_uv=local_uv,
        elem_id_map=elem_id_map,
        zone_masks=zone_masks,
        face_id_map=face_id_map,
        elem_center_x_map=elem_center_x_map,
        wing_span_u=wing_span_u,
        wing_trail_v=wing_trail_v,
        face_is_down=face_is_down,
        wing_bay_u=wing_bay_u,
        wing_bay_v=wing_bay_v,
        wing_bay_mask=wing_bay_mask,
        islands=islands,
        elem_names=elem_names,
        elem_bones=elem_bones,
        elem_zones=elem_zones,
        elem_centers_x=elem_centers_x,
        baked_shadow=baked_shadow,
        bent_normal=bent_normal,
        thickness=thickness,
        curvature_3d=curvature_3d,
        world_tangent=world_tangent,
        world_bitangent=world_bitangent,
        uv_seam_mask=uv_seam_mask,
        uv_seam_src_yx=uv_seam_src_yx,
        uv_seam_dst_yx=uv_seam_dst_yx,
        uv_seam_weight=uv_seam_weight,
        island_edge_px=island_edge_px,
        island_size_min=island_size_min,
        island_size_wh=island_size_wh,
        island_inner_area=island_inner_area,
        face_is_up=face_is_up,
        face_is_side=face_is_side,
        content_fingerprint=fp,
        bake_cache_hit=False,
        bake_ms=round((time.perf_counter() - t_bake_start) * 1000.0, 2),
    )
    if use_cache:
        with _MESH_BAKE_LOCK:
            _MESH_BAKE_LRU[fp] = ctx
            while len(_MESH_BAKE_LRU) > _MESH_BAKE_LRU_MAX:
                _MESH_BAKE_LRU.popitem(last=False)
        _save_mesh_bake_disk(fp, ctx)
    return ctx
