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

from dataclasses import dataclass, field
import math
import re
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
    islands: list[BakedIsland] = field(default_factory=list)
    elem_names: list[str] = field(default_factory=list)
    elem_bones: list[str] = field(default_factory=list)
    elem_zones: list[str] = field(default_factory=list)
    elem_centers_x: list[float] = field(default_factory=list)

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


def bake_bbmodel_maps(
    bbmodel_doc: dict[str, Any],
    *,
    padding: int = 1,
    ao_radius: float = 4.5,
    ao_samples: int = 3,
    zone_overrides: dict[str, str] | None = None,
) -> MeshBakeContext:
    """Bake 3D world position, normal, multi-box Ambient Occlusion, edge bevel & zone masks from a .bbmodel."""
    res = bbmodel_doc.get("resolution") or {}
    W = max(1, int(res.get("width", 1)))
    H = max(1, int(res.get("height", 1)))
    obbs = _extract_bbmodel_obbs(bbmodel_doc)

    island_mask = np.zeros((H, W), dtype=bool)
    world_pos = np.zeros((H, W, 3), dtype=np.float32)
    world_normal = np.zeros((H, W, 3), dtype=np.float32)
    world_normal[..., 1] = 1.0
    baked_ao = np.ones((H, W), dtype=np.float32)
    edge_bevel = np.zeros((H, W), dtype=np.float32)
    joint_cavity = np.zeros((H, W), dtype=np.float32)
    local_uv = np.zeros((H, W, 2), dtype=np.float32)
    elem_id_map = np.full((H, W), -1, dtype=np.int32)
    face_id_map = np.full((H, W), -1, dtype=np.int8)
    elem_center_x_map = np.zeros((H, W), dtype=np.float32)
    wing_span_u = np.zeros((H, W), dtype=np.float32)
    wing_trail_v = np.zeros((H, W), dtype=np.float32)
    face_is_down = np.zeros((H, W), dtype=np.float32)
    zone_masks: dict[str, np.ndarray] = {z: np.zeros((H, W), dtype=np.float32) for z in ANATOMICAL_ZONES}

    islands: list[BakedIsland] = []
    elem_names: list[str] = []
    elem_bones: list[str] = []
    elem_zones: list[str] = []
    elem_centers_x: list[float] = []

    # Pre-stack OBB AABBs for rapid neighbor lookup per box
    if obbs:
        all_aabb_min = np.stack([o["aabb_min"] for o in obbs], axis=0)
        all_aabb_max = np.stack([o["aabb_max"] for o in obbs], axis=0)
        scene_min = all_aabb_min.min(axis=0)
        scene_max = all_aabb_max.max(axis=0)
        scene_span = np.maximum(1e-4, scene_max - scene_min)
        # Scale AO probe radius adaptively to model size so tiny endemic life & giant dragons both look grounded!
        diag = float(np.linalg.norm(scene_span))
        eff_ao_r = float(np.clip(ao_radius * (diag / 65.0), 1.4, 7.5))
    else:
        all_aabb_min = np.zeros((0, 3))
        all_aabb_max = np.zeros((0, 3))
        scene_min = np.zeros(3)
        scene_span = np.ones(3)
        eff_ao_r = float(ao_radius)

    # Probe distances along outward normal & grazing directions for soft multi-scale AO
    probe_dists = np.linspace(0.28 * eff_ao_r, eff_ao_r, max(2, int(ao_samples)), dtype=np.float64)
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
        pad_min = obb["aabb_min"] - eff_ao_r - 0.5
        pad_max = obb["aabb_max"] + eff_ao_r + 0.5
        nb_mask = np.all((all_aabb_max >= pad_min) & (all_aabb_min <= pad_max), axis=1)
        nb_mask[idx] = False
        neighbors = [obbs[j] for j in np.nonzero(nb_mask)[0].tolist()]

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

            # Vectorized multi-box Ambient Occlusion + Joint Crease Cavity!
            pts_flat = w_pts.reshape(-1, 3)
            n_pts = pts_flat.shape[0]
            ao_flat = np.ones(n_pts, dtype=np.float64)
            cav_flat = np.zeros(n_pts, dtype=np.float64)

            if neighbors:
                # 1. Surface proximity cavity (detects buried overlap & tight joint creases!)
                min_sdf = np.full(n_pts, 1e6, dtype=np.float64)
                for nb in neighbors:
                    d_s = _obb_signed_distance(pts_flat, nb)
                    min_sdf = np.minimum(min_sdf, d_s)
                cav_flat = np.clip(1.0 - min_sdf / cav_r, 0.0, 1.0)

                # 2. Multi-step hemisphere AO occlusion along normal + 4 tilted directions
                w_tr = R @ tr_b
                w_td = R @ td_b
                dirs = (
                    w_nrm,
                    (w_nrm + 0.55 * w_tr) / np.linalg.norm(w_nrm + 0.55 * w_tr),
                    (w_nrm - 0.55 * w_tr) / np.linalg.norm(w_nrm - 0.55 * w_tr),
                    (w_nrm + 0.55 * w_td) / np.linalg.norm(w_nrm + 0.55 * w_td),
                    (w_nrm - 0.55 * w_td) / np.linalg.norm(w_nrm - 0.55 * w_td),
                )
                occ_sum = np.zeros(n_pts, dtype=np.float64)
                tot_w = 0.0
                for d_idx, dist in enumerate(probe_dists):
                    step_w = 1.0 / float(d_idx + 1)
                    for d_vec in dirs:
                        probe = pts_flat + d_vec[None, :] * dist
                        min_d = np.full(n_pts, dist, dtype=np.float64)
                        for nb in neighbors:
                            # Fast AABB skip per probe cloud
                            if np.any(nb["aabb_max"] < probe.min(axis=0) - dist) or np.any(nb["aabb_min"] > probe.max(axis=0) + dist):
                                continue
                            min_d = np.minimum(min_d, _obb_signed_distance(probe, nb))
                        # Soft occlusion ratio: 1.0 when probe hits inside neighbor, decaying smoothly to 0 at dist
                        occ_sum += step_w * np.clip(1.0 - min_d / max(1e-4, dist), 0.0, 1.0)
                        tot_w += step_w
                ao_flat = np.clip(1.0 - 0.88 * (occ_sum / max(1e-6, tot_w)), 0.14, 1.0)

            ao_grid = ao_flat.reshape(h_px, w_px).astype(np.float32)
            cav_grid = cav_flat.reshape(h_px, w_px).astype(np.float32)

            # Exposed edge bevel is suppressed where buried in a joint crease!
            exposed_bevel = (bevel_raw * np.clip(1.0 - cav_grid * 1.15, 0.0, 1.0) * ao_grid).astype(np.float32)

            island_mask[y0:y1, x0:x1] = True
            world_pos[y0:y1, x0:x1] = w_pts.astype(np.float32)
            world_normal[y0:y1, x0:x1] = w_nrm.astype(np.float32)
            baked_ao[y0:y1, x0:x1] = ao_grid
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
            elem_id_map[y0:y1, x0:x1] = idx
            face_id_map[y0:y1, x0:x1] = FACES.index(face)
            elem_center_x_map[y0:y1, x0:x1] = float(center[0])
            if zone in zone_masks:
                zone_masks[zone][y0:y1, x0:x1] = 1.0

    world_pos_norm = np.clip((world_pos - scene_min.astype(np.float32)) / scene_span.astype(np.float32), 0.0, 1.0)
    return MeshBakeContext(
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
        islands=islands,
        elem_names=elem_names,
        elem_bones=elem_bones,
        elem_zones=elem_zones,
        elem_centers_x=elem_centers_x,
    )
