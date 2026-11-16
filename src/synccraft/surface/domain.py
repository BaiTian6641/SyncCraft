"""Canonical rest-space SurfaceDomain and posed contact/visibility evaluation (P1-03a).

Keeps three relationships strictly distinct:
1. ``manifold_seams``: Surface continuity across intra-box 12 cuboid edges and
   explicitly welded same-material cross-box flush edges.
2. ``posed_contacts``: Temporary physical proximity and cavity visibility in
   posed world coordinates.
3. ``symmetry_pairs``: Correspondence between ``mirror_x`` twin parts (never
   connected in ``manifold_seams``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

FACES = ("north", "south", "east", "west", "up", "down")
from synccraft.dsl.ast_nodes import Program, Vec3
from synccraft.kinematics.se3_chain import Scene, WorldBox, compile_geometry, compile_posed_geometry
from .contracts import FaceKey, FacePersistentPatch, MaterialProgram, PersistentSurfaceState

# Right-handed orthonormal face frames matching Minecraft Java / Blockbench [u0, v0, u1, v1]:
# +T_right is +u (left-to-right in UV), +T_down is +v (top-to-bottom in UV),
# +B_up = -T_down is OpenGL/LabPBR +Y tangent, and T_right x B_up == N_outward on all 6 faces.
FACE_AXES_BOX: dict[str, tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = {
    "north": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
    "south": ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "east": ((0.0, 0.0, -1.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0)),
    "west": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)),
    "up": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    "down": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
}


def _as_unit_extent(val: float) -> int | float:
    f = float(val)
    r = round(f)
    return int(r) if abs(f - r) <= 1e-9 else f


def face_uv_extents_units(
    size_xyz: np.ndarray | tuple[int | float, int | float, int | float],
    face: str,
) -> tuple[int | float, int | float]:
    """Return (width_u, height_v) in model units for ``face`` (preserving fractional v3.2 steps)."""
    sx, sy, sz = _as_unit_extent(size_xyz[0]), _as_unit_extent(size_xyz[1]), _as_unit_extent(size_xyz[2])
    if face in ("north", "south"):
        return sx, sy
    if face in ("east", "west"):
        return sz, sy
    if face in ("up", "down"):
        return sx, sz
    raise ValueError(f"Unknown face {face!r}")


@dataclass(frozen=True)
class FaceSampleGrid:
    """Rest-space physical sample grid for one (box_name, face)."""

    box_name: str
    bone_name: str
    material_name: str
    face: str
    mirrored: bool
    width_units: int | float
    height_units: int | float
    subdivisions_per_unit: int
    dx: float
    # Sample coordinates in [0, width_units] x [0, height_units], shape (H, W)
    u_units: np.ndarray
    v_units: np.ndarray
    # 3D coordinates in box-local, bone-local, and rest-world frames, shape (H, W, 3)
    pos_box: np.ndarray
    pos_bone: np.ndarray
    pos_rest_world: np.ndarray
    # Orthonormal basis in box-local frame (3,)
    tangent_right_box: np.ndarray
    tangent_down_box: np.ndarray
    bitangent_up_box: np.ndarray
    normal_box: np.ndarray
    # Geometric features in model units / normalized [0,1], shape (H, W)
    edge_distance: np.ndarray
    dorsal_coord: np.ndarray
    bone_axial_coord: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.u_units.shape[0]), int(self.u_units.shape[1])


@dataclass(frozen=True)
class BoundaryNeighborMap:
    """Vectorized boundary neighbor lookup for edge-side {'top','bottom','left','right'}."""

    # Maps (face_key, side) -> (neighbor_face_key, neighbor_rows, neighbor_cols)
    # where side='top' corresponds to row 0, 'bottom' to row H-1, 'left' to col 0, 'right' to col W-1.
    links: dict[tuple[FaceKey, str], tuple[FaceKey, np.ndarray, np.ndarray]]
    seam_pair_count: int


@dataclass(frozen=True)
class SymmetryLink:
    """Symmetry correspondence between unmirrored and mirror_x faces (NOT a PDE seam)."""

    source_key: FaceKey
    mirrored_key: FaceKey
    flip_u: bool = True


@dataclass(frozen=True)
class PosedFaceKinematics:
    """Posed world-space sample coordinates, normals, cavity AO, and OBB contact distance."""

    world_pos: np.ndarray  # (H, W, 3)
    world_normal: np.ndarray  # (H, W, 3)
    cavity_ao: np.ndarray  # (H, W) in [0.15, 1.0]
    min_obb_distance: np.ndarray  # (H, W) signed distance to nearest distinct OBB
    in_contact: np.ndarray  # (H, W) bool where min_obb_distance <= contact_threshold


@dataclass(frozen=True)
class SurfaceDomain:
    """Shared surface domain built from the deterministic geometry compiler."""

    subdivisions_per_unit: int
    dx: float
    rest_scene: Scene
    faces: dict[FaceKey, FaceSampleGrid]
    manifold_seams: BoundaryNeighborMap
    symmetry_pairs: tuple[SymmetryLink, ...]
    connected_components: tuple[tuple[FaceKey, ...], ...]


def build_surface_domain(
    program: Program,
    material_program: MaterialProgram | None = None,
    *,
    subdivisions_per_unit: int = 4,
) -> SurfaceDomain:
    """Construct canonical rest-space SurfaceDomain and its manifold/symmetry graphs."""
    if type(subdivisions_per_unit) is not int or subdivisions_per_unit < 1 or subdivisions_per_unit > 32:
        raise ValueError("subdivisions_per_unit must be an integer in [1, 32]")

    rest_scene = compile_geometry(program)
    dx = 1.0 / float(subdivisions_per_unit)
    bones_by_name = rest_scene.bone_by_name

    # Compute overall rest Y bounds for anatomical dorsal-ventral normalization
    all_corners = np.concatenate([b.corners(include_inflate=False) for b in rest_scene.boxes], axis=0)
    y_min = float(all_corners[:, 1].min())
    y_span = max(1e-6, float(all_corners[:, 1].max() - y_min))

    faces: dict[FaceKey, FaceSampleGrid] = {}
    # Hash table for intra-box 12 cuboid edges: (box_name, quantized_box_edge_pt) -> list of (key, side, idx)
    intrabox_edge_buckets: dict[tuple[str, tuple[int, int, int]], list[tuple[FaceKey, str, int, int, int]]] = {}
    # Hash table for opt-in cross-box weld_group edges in rest world space
    weld_edge_buckets: dict[tuple[str, str, tuple[int, int, int]], list[tuple[FaceKey, str, int, int, int]]] = {}

    scale_q = int(round(4 * subdivisions_per_unit))

    for box in rest_scene.boxes:
        bone = bones_by_name[box.bone_name]
        half_size = np.asarray(box.size, dtype=np.float64) / 2.0

        # Compute axial extent in component-local rest space (when part of a @component instance)
        # or bone-local rest space for direct @rig boxes (ADR 0008).
        use_comp_basis = (
            getattr(box, "instance_name", None) is not None
            and getattr(box, "component_rest_pos", None) is not None
            and getattr(box, "component_rest_rot", None) is not None
        )
        if use_comp_basis:
            comp_boxes = [
                b for b in rest_scene.boxes
                if getattr(b, "instance_name", None) == box.instance_name
                and getattr(b, "component_rest_pos", None) is not None
            ]
            bone_pts = np.concatenate(
                [
                    (np.asarray(((-1, -1, -1), (1, 1, 1)), dtype=np.float64) * (np.asarray(bb.size) / 2.0))
                    + np.asarray(bb.component_rest_pos)
                    for bb in comp_boxes
                ],
                axis=0,
            )
        else:
            bone_boxes = [b for b in rest_scene.boxes if b.bone_name == box.bone_name]
            bone_pts = np.concatenate(
                [
                    (np.asarray(((-1, -1, -1), (1, 1, 1)), dtype=np.float64) * (np.asarray(bb.size) / 2.0))
                    + np.asarray(bb.local_center)
                    for bb in bone_boxes
                ],
                axis=0,
            )
        spans = bone_pts.max(axis=0) - bone_pts.min(axis=0)
        major_axis = int(np.argmax(spans))
        ax_min = float(bone_pts[:, major_axis].min())
        ax_span = max(1e-6, float(spans[major_axis]))

        weld_group: str | None = None
        if material_program is not None and box.material in material_program.materials:
            weld_group = material_program.materials[box.material].weld_group

        r_box_in_bone = np.asarray(bone.rotation, dtype=np.float64).T @ np.asarray(box.rotation, dtype=np.float64)

        for face in FACES:
            wu, hv = face_uv_extents_units(box.size, face)
            cols = max(1, int(round(float(wu) * subdivisions_per_unit)))
            rows = max(1, int(round(float(hv) * subdivisions_per_unit)))
            eff_wu = float(wu) if float(wu) > 1e-9 else dx
            eff_hv = float(hv) if float(hv) > 1e-9 else dx

            u_1d = (np.arange(cols, dtype=np.float64) + 0.5) * (eff_wu / float(cols))
            v_1d = (np.arange(rows, dtype=np.float64) + 0.5) * (eff_hv / float(rows))
            u_grid, v_grid = np.meshgrid(u_1d, v_1d)

            t_r, t_d, n_out = (np.asarray(vec, dtype=np.float64) for vec in FACE_AXES_BOX[face])
            b_up = -t_d

            u_centered = u_grid - 0.5 * eff_wu
            v_centered = v_grid - 0.5 * eff_hv
            normal_offset = float(np.abs(n_out) @ half_size)

            pos_box = (
                u_centered[..., None] * t_r
                + v_centered[..., None] * t_d
                + normal_offset * n_out
            )
            if use_comp_basis:
                pos_bone = (
                    pos_box @ np.asarray(box.component_rest_rot, dtype=np.float64).T
                    + np.asarray(box.component_rest_pos, dtype=np.float64)
                )
            else:
                pos_bone = pos_box @ r_box_in_bone.T + np.asarray(box.local_center, dtype=np.float64)
            pos_rest_world = pos_box @ np.asarray(box.rotation, dtype=np.float64).T + np.asarray(box.center, dtype=np.float64)

            d_edge = np.minimum(
                np.minimum(u_grid, eff_wu - u_grid),
                np.minimum(v_grid, eff_hv - v_grid),
            )
            dorsal = np.clip((pos_rest_world[..., 1] - y_min) / y_span, 0.0, 1.0)
            axial = np.clip((pos_bone[..., major_axis] - ax_min) / ax_span, 0.0, 1.0)

            fkey: FaceKey = (box.name, face)
            faces[fkey] = FaceSampleGrid(
                box_name=box.name,
                bone_name=box.bone_name,
                material_name=box.material,
                face=face,
                mirrored=box.mirrored,
                width_units=wu,
                height_units=hv,
                subdivisions_per_unit=subdivisions_per_unit,
                dx=dx,
                u_units=u_grid,
                v_units=v_grid,
                pos_box=pos_box,
                pos_bone=pos_bone,
                pos_rest_world=pos_rest_world,
                tangent_right_box=t_r,
                tangent_down_box=t_d,
                bitangent_up_box=b_up,
                normal_box=n_out,
                edge_distance=d_edge,
                dorsal_coord=dorsal,
                bone_axial_coord=axial,
            )

            # Register boundary samples on the 4 edges of this face
            boundary_specs = (
                ("top", 0, slice(None), -t_d, cols),
                ("bottom", rows - 1, slice(None), t_d, cols),
                ("left", slice(None), 0, -t_r, rows),
                ("right", slice(None), cols - 1, t_r, rows),
            )
            for side, r_idx, c_idx, out_dir, length in boundary_specs:
                edge_pts_box = pos_box[r_idx, c_idx] + 0.5 * dx * out_dir
                q_box = np.rint(edge_pts_box * scale_q).astype(int)
                for k in range(length):
                    rr = int(r_idx if isinstance(r_idx, int) else k)
                    cc = int(c_idx if isinstance(c_idx, int) else k)
                    q_tup = (int(q_box[k, 0]), int(q_box[k, 1]), int(q_box[k, 2]))
                    intrabox_edge_buckets.setdefault((box.name, q_tup), []).append((fkey, side, k, rr, cc))

                if weld_group is not None:
                    rot_m = np.asarray(box.rotation, dtype=np.float64)
                    out_world = rot_m @ out_dir
                    n_world = rot_m @ n_out
                    edge_pts_world = pos_rest_world[r_idx, c_idx] + 0.5 * dx * out_world
                    q_world = np.rint(edge_pts_world * scale_q).astype(int)
                    for k in range(length):
                        rr = int(r_idx if isinstance(r_idx, int) else k)
                        cc = int(c_idx if isinstance(c_idx, int) else k)
                        qw_tup = (int(q_world[k, 0]), int(q_world[k, 1]), int(q_world[k, 2]))
                        weld_edge_buckets.setdefault((box.material, weld_group, qw_tup), []).append(
                            (fkey, side, k, rr, cc, n_world, out_world)
                        )

    # Assemble strictly reciprocal slot_partner involution: (fk, side, k) <-> (fk2, side2, k2)
    slot_coords: dict[tuple[FaceKey, str, int], tuple[int, int]] = {}
    slot_partner: dict[tuple[FaceKey, str, int], tuple[FaceKey, str, int]] = {}

    def _connect_slots(
        s_a: tuple[FaceKey, str, int],
        rc_a: tuple[int, int],
        s_b: tuple[FaceKey, str, int],
        rc_b: tuple[int, int],
    ) -> None:
        if s_a == s_b:
            return
        if s_a in slot_partner:
            old_a = slot_partner.pop(s_a)
            slot_partner.pop(old_a, None)
        if s_b in slot_partner:
            old_b = slot_partner.pop(s_b)
            slot_partner.pop(old_b, None)
        slot_coords[s_a] = rc_a
        slot_coords[s_b] = rc_b
        slot_partner[s_a] = s_b
        slot_partner[s_b] = s_a

    for entries in intrabox_edge_buckets.values():
        if len(entries) == 2:
            (fk_a, side_a, ka, ra, ca), (fk_b, side_b, kb, rb, cb) = entries
            if fk_a != fk_b:
                _connect_slots((fk_a, side_a, ka), (ra, ca), (fk_b, side_b, kb), (rb, cb))

    # Opt-in cross-box same-material weld_group seams on flush coplanar exterior faces
    for entries_w in weld_edge_buckets.values():
        if len(entries_w) >= 2:
            used_idx: set[int] = set()
            for idx_a in range(len(entries_w)):
                if idx_a in used_idx:
                    continue
                fk_a, side_a, ka, ra, ca, nw_a, ow_a = entries_w[idx_a]
                for idx_b in range(idx_a + 1, len(entries_w)):
                    if idx_b in used_idx:
                        continue
                    fk_b, side_b, kb, rb, cb, nw_b, ow_b = entries_w[idx_b]
                    if fk_a[0] != fk_b[0] and float(np.dot(nw_a, nw_b)) > 0.99 and float(np.dot(ow_a, ow_b)) < -0.99:
                        _connect_slots((fk_a, side_a, ka), (ra, ca), (fk_b, side_b, kb), (rb, cb))
                        used_idx.add(idx_a)
                        used_idx.add(idx_b)
                        break

    side_accum: dict[tuple[FaceKey, str], dict[int, tuple[FaceKey, str, int, int]]] = {}
    for (fk_a, side_a, ka), (fk_b, side_b, kb) in slot_partner.items():
        rb, cb = slot_coords[(fk_b, side_b, kb)]
        side_accum.setdefault((fk_a, side_a), {})[ka] = (fk_b, side_b, rb, cb)

    # Filter to complete whole-side links and enforce strict reciprocity
    tentative_links: dict[tuple[FaceKey, str], tuple[FaceKey, str, np.ndarray, np.ndarray]] = {}
    for (fk, side), mapping in side_accum.items():
        grid = faces[fk]
        rows, cols = grid.shape
        expected_len = cols if side in ("top", "bottom") else rows
        if len(mapping) == expected_len:
            first_nb, first_nb_side, _, _ = mapping[0]
            if all(mapping[m][0] == first_nb and mapping[m][1] == first_nb_side for m in range(expected_len)):
                nb_rows = np.array([mapping[m][2] for m in range(expected_len)], dtype=np.intp)
                nb_cols = np.array([mapping[m][3] for m in range(expected_len)], dtype=np.intp)
                tentative_links[(fk, side)] = (first_nb, first_nb_side, nb_rows, nb_cols)

    links: dict[tuple[FaceKey, str], tuple[FaceKey, np.ndarray, np.ndarray]] = {}
    face_adj: dict[FaceKey, set[FaceKey]] = {fk: set() for fk in faces}
    seam_pairs = 0
    for (fk, side), (nb_fk, nb_side, nb_rows, nb_cols) in tentative_links.items():
        if (nb_fk, nb_side) in tentative_links and tentative_links[(nb_fk, nb_side)][0] == fk and tentative_links[(nb_fk, nb_side)][1] == side:
            links[(fk, side)] = (nb_fk, nb_rows, nb_cols)
            face_adj[fk].add(nb_fk)
            face_adj[nb_fk].add(fk)
            seam_pairs += int(len(nb_rows))
    seam_pairs //= 2

    # Connected components of manifold_seams (mirrored limbs stay disconnected!)
    visited: set[FaceKey] = set()
    components: list[tuple[FaceKey, ...]] = []
    for fk in sorted(faces):
        if fk in visited:
            continue
        comp: list[FaceKey] = []
        stack = [fk]
        visited.add(fk)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in sorted(face_adj[cur]):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(tuple(sorted(comp)))

    # Build Graph 3: symmetry_pairs for mirror_x boxes (including arbitrary-depth nested mirror_x)
    symmetry_links: list[SymmetryLink] = []
    box_map = rest_scene.box_by_name
    bone_map = rest_scene.bone_by_name
    mirror_face_swap = {"east": "west", "west": "east"}
    for box in rest_scene.boxes:
        if "__" in box.name:
            outer_prefix, orig_box = box.name.split("__", 1)
            if outer_prefix in bone_map and orig_box in box_map and box_map[orig_box].mirrored != box.mirrored:
                for face in FACES:
                    src_face = mirror_face_swap.get(face, face)
                    symmetry_links.append(
                        SymmetryLink(source_key=(orig_box, src_face), mirrored_key=(box.name, face), flip_u=True)
                    )

    return SurfaceDomain(
        subdivisions_per_unit=subdivisions_per_unit,
        dx=dx,
        rest_scene=rest_scene,
        faces=faces,
        manifold_seams=BoundaryNeighborMap(links=links, seam_pair_count=seam_pairs),
        symmetry_pairs=tuple(symmetry_links),
        connected_components=tuple(components),
    )


def initialize_surface_state(domain: SurfaceDomain) -> PersistentSurfaceState:
    """Create a zeroed PersistentSurfaceState matching ``domain``."""
    patches = {fk: FacePersistentPatch.zeros(grid.shape) for fk, grid in domain.faces.items()}
    return PersistentSurfaceState(subdivisions_per_unit=domain.subdivisions_per_unit, patches=patches)


def remap_surface_state(
    old_state: PersistentSurfaceState,
    new_domain: SurfaceDomain,
    *,
    policy: str = "preserve_matching_parts",
) -> PersistentSurfaceState:
    """Explicitly remap or reset persistent surface fields after a geometry edit."""
    if policy == "reset":
        return initialize_surface_state(new_domain)
    if policy != "preserve_matching_parts":
        raise ValueError(f"Unknown remap policy {policy!r}")

    new_patches: dict[FaceKey, FacePersistentPatch] = {}
    for fk, grid in new_domain.faces.items():
        if fk in old_state.patches and old_state.patches[fk].moisture.shape == grid.shape:
            new_patches[fk] = old_state.patches[fk].copy()
        elif fk in old_state.patches:
            # Nearest-neighbor physical-coordinate resample on resized cuboid face
            old_p = old_state.patches[fk]
            oh, ow = old_p.moisture.shape
            nh, nw = grid.shape
            r_idx = np.clip((np.arange(nh) * oh) // max(1, nh), 0, oh - 1)
            c_idx = np.clip((np.arange(nw) * ow) // max(1, nw), 0, ow - 1)
            rr, cc = np.meshgrid(r_idx, c_idx, indexing="ij")
            new_patches[fk] = FacePersistentPatch(
                moisture=old_p.moisture[rr, cc].copy(),
                oxidation=old_p.oxidation[rr, cc].copy(),
                coating_damage=old_p.coating_damage[rr, cc].copy(),
                wear=old_p.wear[rr, cc].copy(),
                biofilm=old_p.biofilm[rr, cc].copy(),
                rd_u=old_p.rd_u[rr, cc].copy(),
                rd_v=old_p.rd_v[rr, cc].copy(),
                rd_initialized=old_p.rd_initialized,
            )
        else:
            new_patches[fk] = FacePersistentPatch.zeros(grid.shape)
    return PersistentSurfaceState(subdivisions_per_unit=new_domain.subdivisions_per_unit, patches=new_patches)


def evaluate_posed_surface_kinematics(
    domain: SurfaceDomain,
    program: Program,
    pose_euler_deltas: dict[str, Vec3] | None = None,
    *,
    ao_radius: float = 3.0,
    contact_threshold: float = 0.35,
) -> tuple[Scene, dict[FaceKey, PosedFaceKinematics]]:
    """Evaluate Graph 2 (posed world positions, cavity AO visibility, and OBB contact distance)."""
    posed_scene = compile_posed_geometry(program, pose_euler_deltas)
    posed_boxes = posed_scene.box_by_name

    # Precompute world AABBs for fast OBB neighbor culling
    box_list = list(posed_scene.boxes)
    aabbs_lo = np.zeros((len(box_list), 3), dtype=np.float64)
    aabbs_hi = np.zeros((len(box_list), 3), dtype=np.float64)
    for idx, b in enumerate(box_list):
        corners = b.corners(include_inflate=True)
        aabbs_lo[idx] = corners.min(axis=0)
        aabbs_hi[idx] = corners.max(axis=0)

    results: dict[FaceKey, PosedFaceKinematics] = {}
    for fk, grid in domain.faces.items():
        pbox: WorldBox = posed_boxes[grid.box_name]
        rot = np.asarray(pbox.rotation, dtype=np.float64)
        ctr = np.asarray(pbox.center, dtype=np.float64)

        w_pos = grid.pos_box @ rot.T + ctr
        w_nrm = rot @ grid.normal_box

        rows, cols = grid.shape
        ao = np.ones((rows, cols), dtype=np.float64)
        min_dist = np.full((rows, cols), np.inf, dtype=np.float64)

        p_lo = w_pos.reshape(-1, 3).min(axis=0) - ao_radius
        p_hi = w_pos.reshape(-1, 3).max(axis=0) + ao_radius

        for idx, other in enumerate(box_list):
            if other.name == pbox.name:
                continue
            if np.any(aabbs_hi[idx] < p_lo) or np.any(aabbs_lo[idx] > p_hi):
                continue

            o_rot = np.asarray(other.rotation, dtype=np.float64)
            o_ctr = np.asarray(other.center, dtype=np.float64)
            o_half = other.half_extents

            # Transform face points into other OBB's local frame
            q = (w_pos - o_ctr) @ o_rot
            q_abs = np.abs(q) - o_half
            outside = np.maximum(q_abs, 0.0)
            inside = np.minimum(np.max(q_abs, axis=-1), 0.0)
            signed_d = np.linalg.norm(outside, axis=-1) + inside
            min_dist = np.minimum(min_dist, signed_d)

            # Closest point on other OBB in world space for hemisphere visibility check
            q_clamped = np.clip(q, -o_half, o_half)
            closest_w = q_clamped @ o_rot.T + o_ctr
            vec_to_occ = closest_w - w_pos
            dist_pos = np.maximum(1e-6, np.linalg.norm(vec_to_occ, axis=-1))
            cos_hemi = np.einsum("ijk,k->ij", vec_to_occ, w_nrm) / dist_pos

            # Occluders strictly in front of or tangent to face contribute cavity AO
            in_front = (cos_hemi > -0.05) & (signed_d < ao_radius)
            if np.any(in_front):
                falloff = np.clip(1.0 - np.maximum(0.0, signed_d) / ao_radius, 0.0, 1.0)
                hemi_weight = np.clip(cos_hemi + 0.15, 0.0, 1.0)
                occ = 0.52 * (falloff * falloff) * hemi_weight
                ao = np.where(in_front, ao * (1.0 - occ), ao)

        ao = np.clip(ao, 0.15, 1.0)
        in_contact = min_dist <= contact_threshold
        results[fk] = PosedFaceKinematics(
            world_pos=w_pos,
            world_normal=np.broadcast_to(w_nrm, (rows, cols, 3)).copy(),
            cavity_ao=ao,
            min_obb_distance=min_dist,
            in_contact=in_contact,
        )

    return posed_scene, results
