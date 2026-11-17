"""Composable physical & biological material-process library (P1-03b).

Implements 10 editable material families over the canonical rest-space SurfaceDomain:
1. ``painted_metal``: metallic substrate + dielectric paint coat + reduced wear/oxidation
2. ``brushed_metal``: anisotropic machined/forged conductor + reduced oxidation
3. ``polymer_dielectric``: dielectric IOR/F0 + smooth microfacet + subsurface
4. ``woven_fabric``: orthotropic warp/weft yarn weave + porosity + wear fray
5. ``solid_wood``: 3D rest-space orthotropic cylindrical growth rings & rays
6. ``weathered_stone``: 3D rest-space strata & Voronoi clasts + cleavage + patina
7. ``keratin_scales``: staggered imbricated cellular osteoderm/scale plates
8. ``patterned_skin``: distance-aware anisotropic Gray-Scott RD + countershading
9. ``lichen_biofilm``: moisture/exposure-gated crustose thallus colonization
10. ``thin_film_iridescence``: single-layer Airy optical path interference

All color blending and optical fields operate strictly in linear sRGB space.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from synccraft.dsl.ast_nodes import Palette, Program
from .contracts import (
    ARCHARD_DIRECT_RUBBING_COUPLING,
    ARCHARD_SWEPT_CONTACT_COUPLING,
    ARCHARD_WEAR_NORMALIZATION,
    DIELECTRIC_IOR_REFERENCE_TABLE,
    FaceKey,
    FaceOpticalPatch,
    MaterialProgram,
    MaterialSpec,
    OpticalFields,
    PersistentSurfaceState,
    StampOverride,
    ior_to_dielectric_f0,
    oklch_to_linear_srgb,
)
from .domain import FaceSampleGrid, PosedFaceKinematics, SurfaceDomain


# ---------------------------------------------------------------------------
# Deterministic 3D Rest-Space Noise & Voronoi Primitives (Resolution-Invariant)
# ---------------------------------------------------------------------------


def _hash3_to_unit(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic integer lattice hash in [0, 1] using 64-bit integer mixing."""
    h = (
        ix.astype(np.int64) * 73856093
        ^ iy.astype(np.int64) * 19349663
        ^ iz.astype(np.int64) * 83492791
        ^ (int(seed) * 2654435761)
    ) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return h.astype(np.float64) / 4294967295.0


def value_noise_3d(pos: np.ndarray, seed: int) -> np.ndarray:
    """Smooth trilinear 3D value noise in [0, 1] evaluated at physical 3D coords."""
    fl = np.floor(pos)
    ix = fl[..., 0].astype(np.int64)
    iy = fl[..., 1].astype(np.int64)
    iz = fl[..., 2].astype(np.int64)
    fx = pos[..., 0] - fl[..., 0]
    fy = pos[..., 1] - fl[..., 1]
    fz = pos[..., 2] - fl[..., 2]
    # Quintic C2 fade
    ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0)

    c000 = _hash3_to_unit(ix, iy, iz, seed)
    c100 = _hash3_to_unit(ix + 1, iy, iz, seed)
    c010 = _hash3_to_unit(ix, iy + 1, iz, seed)
    c110 = _hash3_to_unit(ix + 1, iy + 1, iz, seed)
    c001 = _hash3_to_unit(ix, iy, iz + 1, seed)
    c101 = _hash3_to_unit(ix + 1, iy, iz + 1, seed)
    c011 = _hash3_to_unit(ix, iy + 1, iz + 1, seed)
    c111 = _hash3_to_unit(ix + 1, iy + 1, iz + 1, seed)

    x00 = c000 + ux * (c100 - c000)
    x10 = c010 + ux * (c110 - c010)
    x01 = c001 + ux * (c101 - c001)
    x11 = c011 + ux * (c111 - c011)
    y0 = x00 + uy * (x10 - x00)
    y1 = x01 + uy * (x11 - x01)
    return y0 + uz * (y1 - y0)


def fbm_3d(pos: np.ndarray, seed: int, octaves: int = 3) -> np.ndarray:
    """Band-limited 3D fractional Brownian motion in [0, 1]."""
    total = np.zeros(pos.shape[:-1], dtype=np.float64)
    amp = 0.5
    norm = 0.0
    freq = 1.0
    for oc in range(octaves):
        total += amp * value_noise_3d(pos * freq, seed + oc * 97)
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / max(1e-8, norm)


def worley_3d(pos: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (F1_distance, F2_minus_F1_edge) 3D Worley cellular fields."""
    fl = np.floor(pos)
    bx = fl[..., 0].astype(np.int64)
    by = fl[..., 1].astype(np.int64)
    bz = fl[..., 2].astype(np.int64)

    f1 = np.full(pos.shape[:-1], 1e9, dtype=np.float64)
    f2 = np.full(pos.shape[:-1], 1e9, dtype=np.float64)

    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx_i in (-1, 0, 1):
                cx = bx + dx_i
                cy = by + dy
                cz = bz + dz
                jx = _hash3_to_unit(cx, cy, cz, seed + 11)
                jy = _hash3_to_unit(cx, cy, cz, seed + 29)
                jz = _hash3_to_unit(cx, cy, cz, seed + 53)
                px = cx.astype(np.float64) + jx
                py = cy.astype(np.float64) + jy
                pz = cz.astype(np.float64) + jz
                dist2 = (pos[..., 0] - px) ** 2 + (pos[..., 1] - py) ** 2 + (pos[..., 2] - pz) ** 2
                less1 = dist2 < f1
                f2 = np.where(less1, f1, np.minimum(f2, dist2))
                f1 = np.where(less1, dist2, f1)

    d1 = np.sqrt(np.maximum(0.0, f1))
    d2 = np.sqrt(np.maximum(0.0, f2))
    return d1, np.clip(d2 - d1, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Conservative Distance-Aware Manifold Diffusion & Gray-Scott PDE Solver
# ---------------------------------------------------------------------------


def manifold_divergence_operator(
    domain: SurfaceDomain,
    fields: dict[FaceKey, np.ndarray],
    *,
    active_faces: set[FaceKey] | None = None,
    grain_axis_bone: np.ndarray | None = None,
    anisotropy: float = 1.0,
) -> dict[FaceKey, np.ndarray]:
    """Compute conservative div_M(D * grad_M(U)) across seam-connected faces.

    Guarantees:
    1. Constant field U == c produces 0 to machine precision (~1e-15).
    2. Closed / zero-flux boundary conservation: sum_{f,i,j} L(U) * dx^2 == 0 to ~1e-14,
       even with 3D bone-axis anisotropy across 90-degree folded cuboid edges.
    """
    target_faces = active_faces if active_faces is not None else set(fields.keys())
    inv_dx2 = 1.0 / (domain.dx * domain.dx)
    links = domain.manifold_seams.links

    axis = None
    if grain_axis_bone is not None and abs(anisotropy - 1.0) > 1e-8:
        a_arr = np.asarray(grain_axis_bone, dtype=np.float64)
        nrm = float(np.linalg.norm(a_arr))
        if nrm > 1e-12:
            axis = a_arr / nrm

    def _dir_conductance(grid: FaceSampleGrid, step_dir_box: np.ndarray) -> float:
        if axis is None:
            return 1.0
        cos_a = float(np.dot(step_dir_box, axis))
        return 1.0 + (anisotropy - 1.0) * (cos_a * cos_a)

    # Precompute symmetric boundary conductances across seam links
    opp_dir = {
        "top": lambda g: -g.tangent_down_box,
        "bottom": lambda g: g.tangent_down_box,
        "left": lambda g: -g.tangent_right_box,
        "right": lambda g: g.tangent_right_box,
    }

    out: dict[FaceKey, np.ndarray] = {}
    for fk in target_faces:
        u = fields[fk]
        grid = domain.faces[fk]
        gu = _dir_conductance(grid, grid.tangent_right_box)
        gv = _dir_conductance(grid, grid.tangent_down_box)

        lap = np.zeros_like(u, dtype=np.float64)

        # Interior horizontal fluxes (+u / -u)
        if u.shape[1] > 1:
            flux_u = gu * (u[:, 1:] - u[:, :-1])
            lap[:, :-1] += flux_u
            lap[:, 1:] -= flux_u

        # Interior vertical fluxes (+v / -v)
        if u.shape[0] > 1:
            flux_v = gv * (u[1:, :] - u[:-1, :])
            lap[:-1, :] += flux_v
            lap[1:, :] -= flux_v

        # Cross-seam boundary fluxes (symmetric conductance guarantees exact conservation)
        for side, r_sl, c_sl, g_local in (
            ("top", 0, slice(None), gv),
            ("bottom", u.shape[0] - 1, slice(None), gv),
            ("left", slice(None), 0, gu),
            ("right", slice(None), u.shape[1] - 1, gu),
        ):
            seam = links.get((fk, side))
            if seam is None:
                continue  # Zero-flux Neumann on open boundary
            nb_fk, nb_r, nb_c = seam
            if nb_fk not in target_faces:
                continue  # Zero-flux across distinct materials!
            nb_grid = domain.faces[nb_fk]
            # Symmetric mean of normal conductances on both sides of the hinge
            g_nb_u = _dir_conductance(nb_grid, nb_grid.tangent_right_box)
            g_nb_v = _dir_conductance(nb_grid, nb_grid.tangent_down_box)
            # Identify which side nb_fk uses to point back to fk
            g_nb = g_nb_v
            for cand_side in ("top", "bottom", "left", "right"):
                rev = links.get((nb_fk, cand_side))
                if rev is not None and rev[0] == fk:
                    g_nb = g_nb_v if cand_side in ("top", "bottom") else g_nb_u
                    break
            g_sym = 0.5 * (g_local + g_nb)
            u_nb = fields[nb_fk][nb_r, nb_c]
            lap[r_sl, c_sl] += g_sym * (u_nb - u[r_sl, c_sl])

        out[fk] = lap * inv_dx2
    return out


def step_pure_diffusion(
    domain: SurfaceDomain,
    fields: dict[FaceKey, np.ndarray],
    *,
    diffusivity: float,
    total_time: float,
    active_faces: set[FaceKey] | None = None,
    grain_axis_bone: np.ndarray | None = None,
    anisotropy: float = 1.0,
) -> dict[FaceKey, np.ndarray]:
    """Advance pure diffusion dU/dt = div_M(D grad_M U) with automatic CFL sub-stepping."""
    if total_time <= 0.0 or diffusivity <= 0.0:
        return {k: v.copy() for k, v in fields.items()}
    target_faces = active_faces if active_faces is not None else set(fields.keys())
    max_scale = max(1.0, abs(anisotropy))
    dt_cfl = 0.18 * (domain.dx * domain.dx) / max(1e-8, diffusivity * max_scale)
    n_steps = max(1, int(math.ceil(total_time / dt_cfl)))
    dt = total_time / float(n_steps)

    cur = {k: v.copy() for k, v in fields.items()}
    for _ in range(n_steps):
        laps = manifold_divergence_operator(
            domain, cur, active_faces=target_faces, grain_axis_bone=grain_axis_bone, anisotropy=anisotropy
        )
        for fk in target_faces:
            cur[fk] = cur[fk] + dt * diffusivity * laps[fk]
    return cur


def _seed_gray_scott_component(
    domain: SurfaceDomain,
    comp_faces: Iterable[FaceKey],
    spec: MaterialSpec,
    component_seed: int,
) -> tuple[dict[FaceKey, np.ndarray], dict[FaceKey, np.ndarray]]:
    """Seed resolution-invariant physical 3D Gaussian nucleation spots on ``comp_faces``."""
    comp_list = sorted(comp_faces)
    rng = np.random.default_rng(component_seed)

    # Gather rest bounding box of this component in bone-local coordinates
    all_pts = np.concatenate([domain.faces[fk].pos_bone.reshape(-1, 3) for fk in comp_list], axis=0)
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    span = np.maximum(0.5, hi - lo)
    approx_area = sum(max(0.0, float(domain.faces[fk].width_units) * float(domain.faces[fk].height_units)) for fk in comp_list)
    n_spots = max(3, min(36, int(round(approx_area / max(4.0, 9.0 / spec.feature_scale)))))
    spot_radius = 0.75 / max(0.35, spec.feature_scale)

    # Pick deterministic 3D surface-proximate centers by sampling points on the analytical surface
    # parameterized in continuous normalized (face_idx, u_frac, v_frac) — independent of dx!
    face_areas = np.array(
        [max(0.0, float(domain.faces[fk].width_units) * float(domain.faces[fk].height_units)) for fk in comp_list],
        dtype=np.float64,
    )
    area_sum = float(face_areas.sum())
    if area_sum <= 1e-12:
        # Degenerate 0-area plane edge faces form isolated 1-face components on v3.2 planes
        probs = np.full(len(comp_list), 1.0 / float(max(1, len(comp_list))), dtype=np.float64)
    else:
        probs = face_areas / area_sum
    # Stratified face assignment so every major face receives deterministic nucleation centers
    base_faces = [idx for idx, area in enumerate(face_areas) if area >= 12.0]
    if base_faces and len(base_faces) <= n_spots:
        extra = list(rng.choice(len(comp_list), size=n_spots - len(base_faces), replace=True, p=probs))
        chosen_faces = np.array(base_faces + extra, dtype=np.intp)
    else:
        chosen_faces = rng.choice(len(comp_list), size=n_spots, replace=True, p=probs)
    u_fracs = rng.uniform(0.16, 0.84, size=n_spots)
    v_fracs = rng.uniform(0.16, 0.84, size=n_spots)

    centers_bone = np.zeros((n_spots, 3), dtype=np.float64)
    rest_boxes = domain.rest_scene.box_by_name
    for s_idx in range(n_spots):
        fk = comp_list[int(chosen_faces[s_idx])]
        g = domain.faces[fk]
        b = rest_boxes[g.box_name]
        eff_wu = max(float(g.dx), float(g.width_units))
        eff_hv = max(float(g.dx), float(g.height_units))
        uc = (u_fracs[s_idx] - 0.5) * eff_wu
        vc = (v_fracs[s_idx] - 0.5) * eff_hv
        noff = float(np.abs(g.normal_box) @ (np.asarray(b.size, dtype=np.float64) / 2.0))
        p_box = uc * g.tangent_right_box + vc * g.tangent_down_box + noff * g.normal_box
        if g.mirrored and spec.symmetry_mode == "mirrored":
            p_box = p_box * np.array((-1.0, 1.0, 1.0))
        centers_bone[s_idx] = p_box + np.asarray(b.local_center, dtype=np.float64) * (
            np.array((-1.0, 1.0, 1.0)) if (g.mirrored and spec.symmetry_mode == "mirrored") else 1.0
        )

    u_init: dict[FaceKey, np.ndarray] = {}
    v_init: dict[FaceKey, np.ndarray] = {}
    inv_2sig2 = 1.0 / (2.0 * spot_radius * spot_radius)

    for fk in comp_list:
        g = domain.faces[fk]
        pts = g.pos_bone
        if g.mirrored and spec.symmetry_mode == "mirrored":
            pts = pts * np.array((-1.0, 1.0, 1.0))
        v_peak = np.zeros(g.shape, dtype=np.float64)
        for s_idx in range(n_spots):
            d2 = np.sum((pts - centers_bone[s_idx]) ** 2, axis=-1)
            v_peak = np.maximum(v_peak, np.exp(-d2 * inv_2sig2))
        v_arr = 0.42 * v_peak
        u_arr = np.clip(1.0 - 1.8 * v_arr, 0.25, 1.0)
        u_init[fk] = u_arr
        v_init[fk] = v_arr

    return u_init, v_init


def integrate_gray_scott_on_state(
    domain: SurfaceDomain,
    state: PersistentSurfaceState,
    spec: MaterialSpec,
    active_faces: set[FaceKey],
    *,
    duration: float,
    dorsal_modulation: float = 0.04,
) -> int:
    """Evolve Gray-Scott reaction-diffusion PDE in-place on ``state`` across seam-connected components.

    Governing equations (Pearson, J. E., "Complex Patterns in a Simple System",
    Science 261(5118):189-192, 1993):
        ∂U/∂t = D_u ∇_M^2 U - U V^2 + F (1 - U)
        ∂V/∂t = D_v ∇_M^2 V + U V^2 - (F + k) V
    with diffusivity ratio D_u : D_v = 2 : 1, model length unit 1 u = 0.0625 m,
    and automatic CFL sub-stepping Δt <= 0.18 Δx^2 / max(D_u, D_v).
    """
    if not active_faces:
        return 0

    total_steps = 0
    # Group active faces by connected manifold components (mirrored limbs stay separate!)
    for comp_idx, comp in enumerate(domain.connected_components):
        sub = [fk for fk in comp if fk in active_faces]
        if not sub:
            continue
        sub_set = set(sub)
        needs_init = any(not state.patches[fk].rd_initialized for fk in sub)
        run_time = duration
        if needs_init:
            any_mirrored = any(domain.faces[fk].mirrored for fk in sub)
            if any_mirrored and spec.symmetry_mode == "independent_seed":
                comp_seed = spec.seed + 7919 + comp_idx * 1013
            else:
                # For symmetry_mode='mirrored', match unmirrored component seed
                comp_seed = spec.seed + 101
            u0, v0 = _seed_gray_scott_component(domain, sub, spec, comp_seed)
            for fk in sub:
                state.patches[fk].rd_u = u0[fk]
                state.patches[fk].rd_v = v0[fk]
                state.patches[fk].rd_initialized = True
            run_time = max(duration, spec.rd_initial_time)

        if run_time <= 0.0:
            continue

        max_d = max(spec.rd_du, spec.rd_dv, 1e-6) * max(1.0, abs(spec.anisotropy))
        dt_cfl = min(0.85, 0.18 * (domain.dx * domain.dx) / max_d)
        n_steps = max(1, int(math.ceil(run_time / dt_cfl)))
        dt = run_time / float(n_steps)

        u_map = {fk: state.patches[fk].rd_u for fk in sub}
        v_map = {fk: state.patches[fk].rd_v for fk in sub}
        grain_ax = np.asarray(spec.grain_axis, dtype=np.float64)

        for _ in range(n_steps):
            lap_u = manifold_divergence_operator(
                domain, u_map, active_faces=sub_set, grain_axis_bone=grain_ax, anisotropy=spec.anisotropy
            )
            lap_v = manifold_divergence_operator(
                domain, v_map, active_faces=sub_set, grain_axis_bone=grain_ax, anisotropy=spec.anisotropy
            )
            for fk in sub:
                g = domain.faces[fk]
                u_c = u_map[fk]
                v_c = v_map[fk]
                # Optional mild anatomical modulation along dorsal coordinate (0.0 for pure Pearson 1993)
                f_loc = spec.rd_feed * (1.0 + dorsal_modulation * (g.dorsal_coord - 0.5))
                k_loc = spec.rd_kill
                uvv = u_c * v_c * v_c
                u_next = np.clip(u_c + dt * (spec.rd_du * lap_u[fk] - uvv + f_loc * (1.0 - u_c)), 0.0, 1.0)
                v_next = np.clip(v_c + dt * (spec.rd_dv * lap_v[fk] + uvv - (f_loc + k_loc) * v_c), 0.0, 1.0)
                u_map[fk] = u_next
                v_map[fk] = v_next

        total_steps += n_steps
        for fk in sub:
            state.patches[fk].rd_u = u_map[fk]
            state.patches[fk].rd_v = v_map[fk]
    return total_steps


def classify_gray_scott_regime(
    v_field: np.ndarray,
    *,
    threshold: float = 0.16,
) -> dict[str, float | int | str]:
    """Classify a 2D Gray-Scott morphogen V field into Pearson (1993) morphological regimes.

    Distinguishes:
    - ``"homogeneous"``: trivial extinction (V -> 0) or uniform high-V state (std < 1e-3);
    - ``"spots"``: discrete disconnected high-V mitosis/soliton islands (n_components >= 2,
      active_fraction < 0.42, max_component_fraction < 0.20);
    - ``"stripes_labyrinth"``: extended connected coral/labyrinthine bands (active_fraction >= 0.35
      or max_component_fraction >= 0.20 with non-trivial spatial variance).
    """
    v = np.asarray(v_field, dtype=np.float64)
    v_std = float(np.std(v))
    v_max = float(np.max(v))
    if v_std < 1e-3 or v_max < threshold:
        return {
            "regime": "homogeneous",
            "v_max": v_max,
            "v_std": v_std,
            "active_fraction": 0.0 if v_max < threshold else float(np.mean(v >= threshold)),
            "n_components": 0,
            "max_component_fraction": 0.0,
        }

    mask = v >= threshold
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comp_sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                sz = 0
                while stack:
                    cr, cc = stack.pop()
                    sz += 1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                comp_sizes.append(sz)

    total_px = float(max(1, h * w))
    active_frac = float(np.mean(mask))
    n_comp = len(comp_sizes)
    max_comp_frac = float(max(comp_sizes) / total_px) if comp_sizes else 0.0
    if n_comp >= 2 and max_comp_frac < 0.20 and active_frac < 0.42:
        regime = "spots"
    else:
        regime = "stripes_labyrinth"
    return {
        "regime": regime,
        "v_max": v_max,
        "v_std": v_std,
        "active_fraction": active_frac,
        "n_components": n_comp,
        "max_component_fraction": max_comp_frac,
    }


# ---------------------------------------------------------------------------
# Seam-Consistent Tangent Normal Derivation (Preserving Hard 90-deg Cuboid Edges)
# ---------------------------------------------------------------------------


def derive_seam_consistent_tangent_normals(
    domain: SurfaceDomain,
    heights: dict[FaceKey, np.ndarray],
    material_by_face: dict[FaceKey, str],
) -> dict[FaceKey, np.ndarray]:
    """Derive unit tangent normals (T_right, B_up, N_out) from physical height h(u,v).

    Uses physical sample spacing dx and cross-seam micro-height neighbors within the
    same material so surface relief is seam-continuous across UV cuts while hard
    90-degree cuboid geometric normals remain completely un-smoothed.
    """
    links = domain.manifold_seams.links
    dx = domain.dx
    normals: dict[FaceKey, np.ndarray] = {}

    for fk, h in heights.items():
        rows, cols = h.shape
        mat = material_by_face[fk]

        # Padded height array (rows+2, cols+2) with Neumann fallback, overridden by same-material seam links
        h_pad = np.pad(h, 1, mode="edge")

        seam_top = links.get((fk, "top"))
        if seam_top is not None and material_by_face.get(seam_top[0]) == mat:
            h_pad[0, 1:-1] = heights[seam_top[0]][seam_top[1], seam_top[2]]

        seam_bot = links.get((fk, "bottom"))
        if seam_bot is not None and material_by_face.get(seam_bot[0]) == mat:
            h_pad[-1, 1:-1] = heights[seam_bot[0]][seam_bot[1], seam_bot[2]]

        seam_left = links.get((fk, "left"))
        if seam_left is not None and material_by_face.get(seam_left[0]) == mat:
            h_pad[1:-1, 0] = heights[seam_left[0]][seam_left[1], seam_left[2]]

        seam_right = links.get((fk, "right"))
        if seam_right is not None and material_by_face.get(seam_right[0]) == mat:
            h_pad[1:-1, -1] = heights[seam_right[0]][seam_right[1], seam_right[2]]

        dh_du = (h_pad[1:-1, 2:] - h_pad[1:-1, :-2]) / (2.0 * dx)
        # Note: +v is downward row index; +B_up is -v, so dh/dB_up = (h_top - h_bottom) / (2 dx)
        dh_db_up = (h_pad[:-2, 1:-1] - h_pad[2:, 1:-1]) / (2.0 * dx)

        nx = -dh_du
        ny = -dh_db_up
        nz = np.ones_like(h, dtype=np.float64)
        inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
        normals[fk] = np.stack((nx * inv_len, ny * inv_len, nz * inv_len), axis=-1)

    return normals


# ---------------------------------------------------------------------------
# Airy Single-Layer Thin-Film Interference Helper
# ---------------------------------------------------------------------------


def evaluate_airy_thin_film_linear_rgb(
    thickness_nm: np.ndarray,
    film_ior: np.ndarray | float,
    cos_theta_i: np.ndarray | float,
    substrate_rgb_lin: np.ndarray,
) -> np.ndarray:
    """Evaluate single-layer dielectric thin-film Airy interference reflectance in linear sRGB.

    Implements Born & Wolf, *Principles of Optics* (7th ed., 1999, §7.6.1, Eq. 14):
        R(λ) = (r_01^2 + r_12^2 + 2 r_01 r_12 cos δ) / (1 + r_01^2 r_12^2 + 2 r_01 r_12 cos δ)
    with Snell's law sin(θ_t) = sin(θ_i) / n_film, effective optical path difference
    Δ = 2 n_film d cos(θ_t) + λ/2 (incorporating the π reflection phase shift at the
    rarer-to-denser air->film interface), phase δ = 2π Δ / λ, and representative CIE
    RGB wavelengths λ = (650 nm, 532 nm, 460 nm).
    """
    n1 = np.maximum(1.01, np.asarray(film_ior, dtype=np.float64))
    cos_i = np.clip(np.asarray(cos_theta_i, dtype=np.float64), 0.05, 1.0)
    sin2_i = np.maximum(0.0, 1.0 - cos_i * cos_i)
    cos_t = np.sqrt(np.maximum(0.02, 1.0 - sin2_i / (n1 * n1)))

    # Fresnel amplitude reflection at air->film interface
    r01 = (1.0 - n1) / (1.0 + n1)
    R01 = r01 * r01 + (1.0 - r01 * r01) * np.power(1.0 - cos_i, 5.0)
    r1 = np.sqrt(np.clip(R01, 0.0, 0.99))

    # Effective substrate reflection amplitude per RGB channel
    r2_rgb = np.sqrt(np.clip(substrate_rgb_lin, 0.01, 0.99))

    wavelengths_nm = np.array((650.0, 532.0, 460.0), dtype=np.float64)
    opd_nm = (2.0 * n1 * np.asarray(thickness_nm, dtype=np.float64) * cos_t)[..., None]
    # Include half-wave phase shift at air->film interface
    delta = (2.0 * math.pi * opd_nm / wavelengths_nm) + math.pi

    r1_exp = np.asarray(r1)[..., None]
    cos_d = np.cos(delta)
    num = r1_exp * r1_exp + r2_rgb * r2_rgb + 2.0 * r1_exp * r2_rgb * cos_d
    den = np.maximum(1e-6, 1.0 + (r1_exp * r2_rgb) ** 2 + 2.0 * r1_exp * r2_rgb * cos_d)
    return np.clip(num / den, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Full 10-Family Material Process Evaluator
# ---------------------------------------------------------------------------


def _material_coords_3d(grid: FaceSampleGrid, spec: MaterialSpec) -> np.ndarray:
    """Rest-space 3D coordinates respecting symmetry_mode ('independent_seed' vs 'mirrored')."""
    pts = grid.pos_bone.copy()
    if grid.mirrored:
        if spec.symmetry_mode == "mirrored":
            pts[..., 0] = -pts[..., 0]
        else:
            # Offset independent mirrored component deterministically in noise space
            pts = pts + np.array((37.125, -19.375, 53.625), dtype=np.float64)
    return pts * spec.feature_scale


def evaluate_optical_fields(
    program: Program,
    domain: SurfaceDomain,
    material_program: MaterialProgram,
    surface_state: PersistentSurfaceState,
    posed_kinematics: dict[FaceKey, PosedFaceKinematics],
) -> OpticalFields:
    """Evaluate all 10 material-process families + persistent states -> Intrinsic OpticalFields."""
    palette_map: dict[str, Palette] = {p.name: p for p in program.palettes}

    # First ensure any patterned_skin faces have initialized/evolved Gray-Scott state
    rd_faces_by_mat: dict[str, set[FaceKey]] = {}
    for fk, grid in domain.faces.items():
        spec = material_program.spec_for(grid.material_name, palette_map.get(grid.material_name))
        if spec.family == "patterned_skin":
            rd_faces_by_mat.setdefault(grid.material_name, set()).add(fk)

    for mat_name, fk_set in rd_faces_by_mat.items():
        spec = material_program.spec_for(mat_name, palette_map.get(mat_name))
        if any(not surface_state.patches[fk].rd_initialized for fk in fk_set):
            integrate_gray_scott_on_state(domain, surface_state, spec, fk_set, duration=0.0)

    raw_patches: dict[FaceKey, dict[str, np.ndarray | int]] = {}
    heights_map: dict[FaceKey, np.ndarray] = {}
    mat_by_face: dict[FaceKey, str] = {}

    for fk, grid in domain.faces.items():
        pal = palette_map.get(grid.material_name)
        spec = material_program.spec_for(grid.material_name, pal)
        st = surface_state.patches[fk]
        posed = posed_kinematics[fk]
        rows, cols = grid.shape

        p3 = _material_coords_3d(grid, spec)
        eff_wu_grid = max(float(grid.dx), float(grid.width_units))
        u_m = (eff_wu_grid - grid.u_units) if (grid.mirrored and spec.symmetry_mode == "mirrored") else grid.u_units
        v_m = grid.v_units

        c_sub = oklch_to_linear_srgb(spec.substrate_lch)
        c_sec = oklch_to_linear_srgb(spec.secondary_lch)
        c_coat = oklch_to_linear_srgb(spec.coating_lch)
        c_ox = oklch_to_linear_srgb(spec.oxide_lch)
        f0_diel = ior_to_dielectric_f0(spec.dielectric_ior)

        albedo = np.broadcast_to(c_sub, (rows, cols, 3)).copy()
        height = np.zeros((rows, cols), dtype=np.float64)
        rough = np.full((rows, cols), spec.perceptual_roughness, dtype=np.float64)
        metal = np.full((rows, cols), spec.metalness, dtype=np.float64)
        f0_arr = np.full((rows, cols), f0_diel, dtype=np.float64)
        poros = np.full((rows, cols), spec.porosity, dtype=np.float64)
        subsurf = np.full((rows, cols), spec.subsurface, dtype=np.float64)
        emissive = np.full((rows, cols), spec.emissive_strength, dtype=np.float64)
        irid_nm = np.zeros((rows, cols), dtype=np.float64)
        irid_ior = np.full((rows, cols), spec.film_ior, dtype=np.float64)

        # Cumulative dimensionless Archard (1953) wear depth Δh / h_coat,0
        # evolved in world/transitions.py via ARCHARD_SWEPT_CONTACT_COUPLING (0.22)
        # and ARCHARD_DIRECT_RUBBING_COUPLING (0.45); see ARCHARD_WEAR_NORMALIZATION.
        total_wear = st.wear + st.coating_damage
        fam = spec.family

        if fam == "painted_metal":
            # Brushed metallic substrate under a dielectric paint coat abraded by persistent wear
            brush = 0.03 * np.sin((u_m * 6.0 + v_m * 0.8) * spec.feature_scale)
            bevel = np.clip((grid.edge_distance - 0.15) / 0.5, 0.0, 1.0)
            coat_rem = np.clip(spec.coating_thickness - total_wear, 0.0, max(1e-6, spec.coating_thickness))
            coat_cov = np.clip(coat_rem / max(0.08, 0.35 * max(0.2, spec.coating_thickness)), 0.0, 1.0)

            exposed_metal = 1.0 - coat_cov
            ox = np.clip(st.oxidation * exposed_metal, 0.0, 1.0)

            sub_col = c_sub * (0.92 + 0.12 * (0.5 + 0.5 * np.sin(u_m * 5.0 * spec.feature_scale)))[..., None]
            metal_or_ox = sub_col * (1.0 - ox[..., None]) + c_ox * ox[..., None]
            albedo = metal_or_ox * (1.0 - coat_cov[..., None]) + c_coat * coat_cov[..., None]

            height = -0.04 * (1.0 - bevel) + 0.035 * coat_cov + brush * exposed_metal - 0.02 * ox
            rough_sub = spec.substrate_roughness * (1.0 - ox) + 0.88 * ox
            rough = rough_sub * exposed_metal + spec.perceptual_roughness * coat_cov
            metal = spec.substrate_metalness * exposed_metal * (1.0 - 0.85 * ox)
            poros = 0.02 * coat_cov + 0.45 * ox

        elif fam == "brushed_metal":
            groove = np.sin((u_m * 8.0 + 0.5 * fbm_3d(p3 * 0.7, spec.seed)) * spec.feature_scale)
            bevel = np.clip(grid.edge_distance / 0.45, 0.0, 1.0)
            ox = np.clip(st.oxidation, 0.0, 1.0)
            wear_polish = np.clip(total_wear, 0.0, 1.0)

            base_m = c_sub * (0.94 + 0.06 * groove[..., None])
            albedo = base_m * (1.0 - ox[..., None]) + c_ox * ox[..., None]
            height = 0.025 * groove - 0.035 * (1.0 - bevel) - 0.015 * ox
            rough = np.clip(spec.substrate_roughness * (1.0 - 0.35 * wear_polish) + 0.65 * ox, 0.04, 1.0)
            metal = np.clip(spec.substrate_metalness * (1.0 - 0.85 * ox), 0.0, 1.0)
            poros = 0.02 + 0.40 * ox

        elif fam == "polymer_dielectric":
            micro = fbm_3d(p3 * 1.2, spec.seed) - 0.5
            bevel = np.clip(grid.edge_distance / 0.4, 0.0, 1.0)
            scuff = np.clip(total_wear, 0.0, 1.0)
            albedo = c_sub * (1.0 - 0.15 * scuff[..., None]) + c_sec * (0.15 * scuff[..., None])
            height = 0.015 * micro - 0.025 * (1.0 - bevel) - 0.02 * scuff
            rough = np.clip(spec.perceptual_roughness + 0.42 * scuff, 0.03, 1.0)
            metal = np.zeros((rows, cols), dtype=np.float64)

        elif fam == "woven_fabric":
            freq = 3.2 * spec.feature_scale
            wu_s = np.sin(u_m * math.pi * freq)
            wv_s = np.sin(v_m * math.pi * freq)
            warp_over = (np.sin(u_m * math.pi * freq * 0.5) * np.sin(v_m * math.pi * freq * 0.5)) >= 0.0
            yarn_h = np.where(warp_over, np.abs(wu_s), np.abs(wv_s))
            fray = np.clip(total_wear, 0.0, 1.0)
            wet = np.clip(st.moisture, 0.0, 1.0)

            yarn_col = np.where(warp_over[..., None], c_sub, c_sec)
            albedo = yarn_col * (1.0 - 0.32 * wet[..., None]) * (0.88 + 0.12 * yarn_h[..., None])
            height = 0.04 * yarn_h - 0.02 * fray
            rough = np.clip(max(0.65, spec.perceptual_roughness) + 0.18 * fray - 0.20 * wet, 0.15, 1.0)
            poros = np.clip(max(0.55, spec.porosity), 0.0, 1.0)
            metal = np.zeros((rows, cols), dtype=np.float64)

        elif fam == "solid_wood":
            # 3D orthotropic cylindrical growth rings around grain_axis in rest coordinates
            g_ax = np.asarray(spec.grain_axis, dtype=np.float64)
            g_ax = g_ax / max(1e-8, float(np.linalg.norm(g_ax)))
            along = np.einsum("ijk,k->ij", p3, g_ax)
            radial_vec = p3 - along[..., None] * g_ax
            radius = np.linalg.norm(radial_vec, axis=-1)
            turb = 0.42 * fbm_3d(p3 * np.array((0.65, 0.22, 0.65)), spec.seed)
            rings = 0.5 + 0.5 * np.sin((radius * 3.4 + turb * 3.0) * math.pi)
            latewood = np.power(rings, 2.2)
            rays = 0.12 * (value_noise_3d(p3 * np.array((4.5, 0.35, 4.5)), spec.seed + 31) - 0.5)

            wet = np.clip(st.moisture, 0.0, 1.0)
            polish = np.clip(total_wear, 0.0, 1.0)
            wood_col = c_sub * (1.0 - latewood[..., None]) + c_sec * latewood[..., None]
            wood_col = np.clip(wood_col + rays[..., None], 0.0, 1.0)
            # Wet wood darkens optically due to internal pore index matching
            albedo = wood_col * (1.0 - 0.30 * wet[..., None])
            height = 0.022 * (1.0 - latewood) + 0.01 * rays
            rough = np.clip(spec.perceptual_roughness + 0.12 * latewood - 0.22 * polish - 0.18 * wet, 0.08, 1.0)
            poros = np.clip(max(0.35, spec.porosity), 0.0, 1.0)
            metal = np.zeros((rows, cols), dtype=np.float64)

        elif fam == "weathered_stone":
            d1, edge_w = worley_3d(p3 * 0.55, spec.seed)
            strata = 0.5 + 0.5 * np.sin(p3[..., 1] * 1.8 + 1.6 * fbm_3d(p3 * 0.45, spec.seed + 19))
            chip = np.clip(total_wear + 0.25 * np.clip(1.0 - grid.edge_distance / 0.35, 0.0, 1.0), 0.0, 1.0)
            moss = np.clip(st.biofilm, 0.0, 1.0)
            wet = np.clip(st.moisture, 0.0, 1.0)

            clast_mix = np.clip(0.55 * strata + 0.45 * (1.0 - d1), 0.0, 1.0)
            rock_col = c_sub * clast_mix[..., None] + c_sec * (1.0 - clast_mix[..., None])
            rock_col = rock_col * (1.0 - 0.22 * chip[..., None]) + np.minimum(1.0, c_sub * 1.25) * (0.22 * chip[..., None])
            moss_col = oklch_to_linear_srgb((0.46, 0.11, 132.0))
            albedo = (rock_col * (1.0 - 0.25 * wet[..., None])) * (1.0 - moss[..., None]) + moss_col * moss[..., None]

            height = 0.045 * np.clip(edge_w, 0.0, 0.6) - 0.03 * chip + 0.035 * moss
            rough = np.clip(max(0.68, spec.perceptual_roughness) - 0.25 * wet + 0.20 * moss, 0.15, 1.0)
            poros = np.clip(max(0.45, spec.porosity), 0.0, 1.0)
            metal = np.zeros((rows, cols), dtype=np.float64)

        elif fam == "keratin_scales":
            sc = 1.6 * spec.feature_scale
            row_f = v_m * sc
            row_id = np.floor(row_f)
            col_f = u_m * sc + 0.5 * (row_id % 2.0)
            fu = (col_f - np.floor(col_f)) - 0.5
            fv = (row_f - row_id)
            dist_rim = np.sqrt(fu * fu + (fv * 0.85) ** 2)
            plate = np.clip((0.58 - dist_rim) / 0.16, 0.0, 1.0)
            # Imbricated overlap: higher toward trailing edge fv -> 1
            imbrication = plate * (0.35 + 0.65 * fv)
            dorsal_shade = 1.0 - spec.countershade_strength * (1.0 - grid.dorsal_coord)

            scale_col = c_sub * plate[..., None] + c_sec * (1.0 - plate[..., None])
            albedo = np.clip(scale_col * dorsal_shade[..., None], 0.0, 1.0)
            height = 0.055 * imbrication
            rough = np.clip(spec.perceptual_roughness + 0.25 * (1.0 - plate) + 0.20 * np.clip(total_wear, 0.0, 1.0), 0.08, 1.0)
            # Reptilian/avian α-keratin refractive index n = 1.55 -> F0 = 0.0465
            # (Leertouwer, Wilts & Stavenga, J. R. Soc. Interface 8:1687-1694, 2011)
            keratin_ior = float(DIELECTRIC_IOR_REFERENCE_TABLE["alpha_keratin_scale"]["ior"])
            f0_arr = np.full((rows, cols), ior_to_dielectric_f0(keratin_ior), dtype=np.float64)
            subsurf = np.full((rows, cols), max(0.12, spec.subsurface), dtype=np.float64)
            metal = np.zeros((rows, cols), dtype=np.float64)

        elif fam == "patterned_skin":
            coat_mode = getattr(spec, "coat", None)
            if coat_mode == "plain":
                pigment = np.zeros((rows, cols), dtype=np.float64)
            elif coat_mode == "banded_limb":
                # Anatomical horizontal bands along limb/bone axial coordinate (`SKILL.md` §Limbs: 1 darker row in 5)
                band_idx = np.floor(grid.bone_axial_coord * max(4.0, 5.0 * spec.feature_scale)).astype(np.int64)
                pigment = np.where((band_idx % 5) == 2, 0.85, 0.0)
            elif coat_mode == "scale_rows":
                # Anatomical staggered scale rows following the body axis, thinning toward the belly
                sc_r = 2.0 * spec.feature_scale
                r_f = grid.dorsal_coord * 6.0 * sc_r
                r_id = np.floor(r_f)
                c_f = grid.bone_axial_coord * 8.0 * sc_r + 0.5 * (r_id % 2.0)
                du_s = (c_f - np.floor(c_f)) - 0.5
                dv_s = (r_f - r_id) - 0.5
                rim = np.sqrt(du_s * du_s + dv_s * dv_s)
                dorsal_weight = np.clip((grid.dorsal_coord - 0.25) / 0.55, 0.0, 1.0)
                pigment = np.where(rim > 0.38, 0.72 * dorsal_weight, 0.0)
            else:
                v_conc = st.rd_v
                pigment = np.clip((v_conc - 0.10) / 0.14, 0.0, 1.0)
            # Anatomical dorsal-ventral countershading & distal extremity darkening
            counter = 1.0 + spec.countershade_strength * (0.5 - grid.dorsal_coord) * 1.4
            extrem = 1.0 - spec.extremity_darkening * grid.bone_axial_coord

            skin_base = np.clip(c_sub * counter[..., None] * extrem[..., None], 0.0, 1.0)
            albedo = skin_base * (1.0 - pigment[..., None]) + c_sec * pigment[..., None]
            pore = (
                np.zeros((rows, cols), dtype=np.float64)
                if coat_mode == "plain"
                else 0.012 * (value_noise_3d(p3 * 3.5, spec.seed) - 0.5)
            )
            height = pore + 0.014 * pigment
            rough = np.clip(spec.perceptual_roughness - 0.08 * pigment - 0.20 * st.moisture, 0.12, 1.0)
            subsurf = np.full((rows, cols), max(0.28, spec.subsurface), dtype=np.float64)
            poros = np.full((rows, cols), max(0.18, spec.porosity), dtype=np.float64)
            metal = np.zeros((rows, cols), dtype=np.float64)

        elif fam == "lichen_biofilm":
            d1, _ = worley_3d(p3 * 0.9, spec.seed)
            rosette = np.clip(1.0 - d1 * 1.35, 0.0, 1.0)
            upward = np.clip(posed.world_normal[..., 1] * 0.5 + 0.5, 0.0, 1.0)
            colonized = np.clip(st.biofilm + spec.biofilm_affinity * rosette * (0.4 + 0.6 * upward), 0.0, 1.0)
            thallus = np.clip((colonized - 0.25) / 0.35, 0.0, 1.0)

            albedo = c_sub * (1.0 - thallus[..., None]) + c_sec * thallus[..., None]
            height = 0.05 * thallus * (0.6 + 0.4 * rosette)
            rough = np.clip(spec.perceptual_roughness * (1.0 - thallus) + 0.90 * thallus, 0.15, 1.0)
            poros = np.clip(spec.porosity * (1.0 - thallus) + 0.65 * thallus, 0.0, 1.0)
            metal = np.zeros((rows, cols), dtype=np.float64)

        elif fam == "thin_film_iridescence":
            micro_var = 38.0 * (fbm_3d(p3 * 0.85, spec.seed) - 0.5)
            coat_intact = np.clip(1.0 - total_wear, 0.0, 1.0)
            d_nm = np.maximum(0.0, (spec.film_thickness_nm + micro_var) * coat_intact)
            irid_nm = d_nm
            albedo = np.broadcast_to(c_sub, (rows, cols, 3)).copy()
            height = 0.018 * np.sin((u_m + v_m) * 3.0 * spec.feature_scale) * coat_intact
            rough = np.clip(spec.perceptual_roughness + 0.35 * (1.0 - coat_intact), 0.03, 1.0)
            metal = np.full((rows, cols), spec.metalness, dtype=np.float64)

        # Apply resolution-independent artist stamps/overrides in physical (u, v) coordinates
        for stamp in material_program.stamps:
            if stamp.box_name != grid.box_name or stamp.face != grid.face:
                continue
            cu = 0.5 * (stamp.u_min + stamp.u_max)
            cv = 0.5 * (stamp.v_min + stamp.v_max)
            ru = max(1e-6, 0.5 * (stamp.u_max - stamp.u_min))
            rv = max(1e-6, 0.5 * (stamp.v_max - stamp.v_min))

            in_box = (grid.u_units >= stamp.u_min) & (grid.u_units <= stamp.u_max) & (grid.v_units >= stamp.v_min) & (grid.v_units <= stamp.v_max)
            if stamp.shape == "rect" or stamp.shape in ("stripe_u", "stripe_v"):
                mask = in_box
            elif stamp.shape == "disc":
                mask = (((grid.u_units - cu) / ru) ** 2 + ((grid.v_units - cv) / rv) ** 2) <= 1.0
            elif stamp.shape == "cross":
                arm_u = np.abs(grid.u_units - cu) <= (0.34 * ru)
                arm_v = np.abs(grid.v_units - cv) <= (0.34 * rv)
                mask = in_box & (arm_u | arm_v)
            else:
                mask = in_box

            if np.any(mask):
                stamp_lin = oklch_to_linear_srgb(stamp.lch)
                albedo[mask] = stamp_lin
                height[mask] += stamp.height_delta
                if stamp.perceptual_roughness is not None:
                    rough[mask] = stamp.perceptual_roughness
                if stamp.metalness is not None:
                    metal[mask] = stamp.metalness
                if stamp.emissive is not None:
                    emissive[mask] = stamp.emissive

        heights_map[fk] = np.broadcast_to(height, (rows, cols)).copy()
        mat_by_face[fk] = grid.material_name
        raw_patches[fk] = {
            "albedo": np.clip(np.broadcast_to(albedo, (rows, cols, 3)), 0.0, 1.0).copy(),
            "rough": np.clip(np.broadcast_to(rough, (rows, cols)), 0.02, 1.0).copy(),
            "metal": np.clip(np.broadcast_to(metal, (rows, cols)), 0.0, 1.0).copy(),
            "f0": np.clip(np.broadcast_to(f0_arr, (rows, cols)), 0.0, 0.25).copy(),
            "poros": np.clip(np.broadcast_to(poros, (rows, cols)), 0.0, 1.0).copy(),
            "subsurf": np.clip(np.broadcast_to(subsurf, (rows, cols)), 0.0, 1.0).copy(),
            "emissive": np.clip(np.broadcast_to(emissive, (rows, cols)), 0.0, 1.0).copy(),
            "ao": posed.cavity_ao,
            "irid_nm": np.broadcast_to(irid_nm, (rows, cols)).copy(),
            "irid_ior": np.broadcast_to(irid_ior, (rows, cols)).copy(),
            "metal_code": int(spec.labpbr_metal_code),
        }

    tangent_normals = derive_seam_consistent_tangent_normals(domain, heights_map, mat_by_face)

    optical_patches: dict[FaceKey, FaceOpticalPatch] = {}
    for fk, raw in raw_patches.items():
        h_arr = heights_map[fk]
        optical_patches[fk] = FaceOpticalPatch(
            base_color_linear=raw["albedo"],  # type: ignore[arg-type]
            height=h_arr,
            tangent_normal=tangent_normals[fk],
            perceptual_roughness=raw["rough"],  # type: ignore[arg-type]
            metalness=raw["metal"],  # type: ignore[arg-type]
            dielectric_f0=raw["f0"],  # type: ignore[arg-type]
            porosity=raw["poros"],  # type: ignore[arg-type]
            subsurface=raw["subsurf"],  # type: ignore[arg-type]
            opacity=np.ones_like(h_arr, dtype=np.float64),
            emissive=raw["emissive"],  # type: ignore[arg-type]
            cavity_ao=raw["ao"],  # type: ignore[arg-type]
            iridescence_thickness_nm=raw["irid_nm"],  # type: ignore[arg-type]
            iridescence_ior=raw["irid_ior"],  # type: ignore[arg-type]
            labpbr_metal_code=int(raw["metal_code"]),
        )

    return OpticalFields(
        subdivisions_per_unit=float(domain.subdivisions_per_unit),
        patches=optical_patches,
    )
