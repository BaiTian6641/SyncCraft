"""Simulation, iterative processes, Turing patterns & pure algorithmic (non-ML) denoisers.

Includes:
- Gray-Scott Reaction-Diffusion (Turing spots, stripes, labyrinth, coral, mitosis, scales)
- Cellular Automata (cave cracks, moss growth, organic clustering)
- Generic Iterate feedback loop (weathering, erosion, patina, sharpen-cluster, warp feedback)
- Weathering & Gravity Drip simulation (rain streaks, cavity grime, top-edge wear)
- Pure Algorithmic Denoisers & Pixel-Art Cluster Filters:
  * Anisotropic Kuwahara (painterly pixel clusters with razor-sharp edges!)
  * Bilateral Filter (edge-preserving spatial + tonal smoothing)
  * Chambolle Total-Variation TV-L2 Denoiser
  * Non-Local Means (patch-similarity self-denoising)
  * Morphological Despeckle (orphan 1-px pixel-art cleaner)
  * Pixel Cluster Mode (majority-vote palette cluster consolidator)
"""
from __future__ import annotations

import math
import numpy as np

from .generators_noise import domain_warp, fbm_noise, uv_grid

GRAY_SCOTT_PRESETS: dict[str, tuple[float, float]] = {
    "spots": (0.030, 0.062),
    "stripes": (0.035, 0.060),
    "labyrinth": (0.029, 0.057),
    "coral": (0.0545, 0.062),
    "mitosis": (0.0367, 0.0649),
    "holes": (0.039, 0.058),
    "scales": (0.042, 0.061),
    "fingerprint": (0.037, 0.060),
}


def _laplacian_2d(arr: np.ndarray, *, tileable: bool = True, aniso_x: float = 1.0, aniso_y: float = 1.0) -> np.ndarray:
    """9-point isotropic/anisotropic 2D discrete Laplacian."""
    if tileable:
        up = np.roll(arr, 1, axis=0)
        dn = np.roll(arr, -1, axis=0)
        lt = np.roll(arr, 1, axis=1)
        rt = np.roll(arr, -1, axis=1)
        ul = np.roll(up, 1, axis=1)
        ur = np.roll(up, -1, axis=1)
        dl = np.roll(dn, 1, axis=1)
        dr = np.roll(dn, -1, axis=1)
    else:
        p = np.pad(arr, 1, mode="edge")
        up, dn = p[:-2, 1:-1], p[2:, 1:-1]
        lt, rt = p[1:-1, :-2], p[1:-1, 2:]
        ul, ur = p[:-2, :-2], p[:-2, 2:]
        dl, dr = p[2:, :-2], p[2:, 2:]
    ax, ay = float(aniso_x), float(aniso_y)
    diag = 0.05 * 0.5 * (ax + ay)
    card_x = 0.20 * ax
    card_y = 0.20 * ay
    center = -(2.0 * card_x + 2.0 * card_y + 4.0 * diag)
    return card_x * (lt + rt) + card_y * (up + dn) + diag * (ul + ur + dl + dr) + center * arr


def gray_scott_turing(
    height: int,
    width: int,
    *,
    preset: str = "labyrinth",
    feed: float | None = None,
    kill: float | None = None,
    du: float = 0.16,
    dv: float = 0.08,
    steps: int = 350,
    sim_scale: int = 1,
    aniso_x: float = 1.0,
    aniso_y: float = 1.0,
    seed: int = 42,
    tileable: bool = True,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized Gray-Scott Turing Reaction-Diffusion pattern in [0, 1].

    ``sim_scale > 1`` runs the PDE on a coarser grid (e.g. 2x/4x coarser) and smoothly
    upsamples for wider organic spots/stripes at instant speed!
    """
    f_def, k_def = GRAY_SCOTT_PRESETS.get(preset, GRAY_SCOTT_PRESETS["labyrinth"])
    F = float(feed if feed is not None else f_def)
    K = float(kill if kill is not None else k_def)
    sc = max(1, int(sim_scale))
    sh = max(8, height // sc)
    sw = max(8, width // sc)

    rng = np.random.default_rng(int(seed))
    U = np.ones((sh, sw), dtype=np.float32)
    V = np.zeros((sh, sw), dtype=np.float32)

    if seed_mask is not None:
        sm = np.asarray(seed_mask, dtype=np.float32)
        if sm.shape != (sh, sw):
            ys = np.linspace(0, sm.shape[0] - 1, sh).astype(np.int64)
            xs = np.linspace(0, sm.shape[1] - 1, sw).astype(np.int64)
            sm = sm[np.ix_(ys, xs)]
        V = np.clip(sm, 0.0, 1.0) * 0.45
        U = 1.0 - V
    else:
        # Stratified random nucleation spots so Turing waves fill the entire canvas evenly
        n_seeds = max(6, (sh * sw) // 90)
        for _ in range(n_seeds):
            cy = int(rng.integers(0, sh))
            cx = int(rng.integers(0, sw))
            r = int(rng.integers(1, max(2, min(4, min(sh, sw) // 8))))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    yy = (cy + dy) % sh if tileable else min(sh - 1, max(0, cy + dy))
                    xx = (cx + dx) % sw if tileable else min(sw - 1, max(0, cx + dx))
                    V[yy, xx] = 0.25 + float(rng.uniform(0.0, 0.15))
                    U[yy, xx] = 0.50

    # Keep CFL stable even with anisotropic conductance
    dt = np.float32(min(1.0, 0.92 / max(1.0, max(aniso_x, aniso_y))))
    Du = np.float32(du)
    Dv = np.float32(dv)
    Ff = np.float32(F)
    Fk = np.float32(F + K)

    for _ in range(max(1, int(steps))):
        lu = _laplacian_2d(U, tileable=tileable, aniso_x=aniso_x, aniso_y=aniso_y)
        lv = _laplacian_2d(V, tileable=tileable, aniso_x=aniso_x, aniso_y=aniso_y)
        uvv = U * V * V
        U = np.clip(U + dt * (Du * lu - uvv + Ff * (1.0 - U)), 0.0, 1.0)
        V = np.clip(V + dt * (Dv * lv + uvv - Fk * V), 0.0, 1.0)

    v_lo, v_hi = float(V.min()), float(V.max())
    out = (V - v_lo) / max(1e-5, v_hi - v_lo)
    if (sh, sw) != (height, width):
        from .generators_noise import sample_field_bilinear
        u_g, v_g = uv_grid(height, width)
        out = sample_field_bilinear(out, u_g, v_g, tileable=tileable)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def cellular_automata(
    height: int,
    width: int,
    *,
    preset: str = "cave",
    born: tuple[int, ...] | None = None,
    survive: tuple[int, ...] | None = None,
    density: float = 0.45,
    steps: int = 8,
    seed: int = 42,
    tileable: bool = True,
    initial_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized Moore-neighborhood Cellular Automata (organic caves, moss clusters, coral)."""
    presets = {
        "cave": ((5, 6, 7, 8), (4, 5, 6, 7, 8)),
        "moss": ((3, 4, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8)),
        "coral": ((3,), (4, 5, 6, 7, 8)),
        "maze": ((3,), (1, 2, 3, 4, 5)),
        "cracks": ((4, 5, 6, 7, 8), (2, 3, 4, 5)),
    }
    b_def, s_def = presets.get(preset, presets["cave"])
    b_set = set(born if born is not None else b_def)
    s_set = set(survive if survive is not None else s_def)

    if initial_mask is not None:
        grid = (np.asarray(initial_mask, dtype=np.float32) > 0.5).astype(np.uint8)
    else:
        rng = np.random.default_rng(int(seed))
        grid = (rng.random((height, width)) < float(density)).astype(np.uint8)

    for _ in range(max(1, int(steps))):
        if tileable:
            nb = (
                np.roll(grid, 1, 0) + np.roll(grid, -1, 0)
                + np.roll(grid, 1, 1) + np.roll(grid, -1, 1)
                + np.roll(np.roll(grid, 1, 0), 1, 1)
                + np.roll(np.roll(grid, 1, 0), -1, 1)
                + np.roll(np.roll(grid, -1, 0), 1, 1)
                + np.roll(np.roll(grid, -1, 0), -1, 1)
            )
        else:
            p = np.pad(grid, 1, mode="edge")
            nb = (
                p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
                + p[:-2, :-2] + p[:-2, 2:] + p[2:, :-2] + p[2:, 2:]
            )
        b_mask = np.isin(nb, list(b_set))
        s_mask = np.isin(nb, list(s_set))
        grid = np.where(grid == 1, s_mask, b_mask).astype(np.uint8)

    return grid.astype(np.float32)


def weathering_drip(
    height_field: np.ndarray,
    *,
    cavity_mask: np.ndarray | None = None,
    streak_length: int = 8,
    strength: float = 0.65,
    decay: float = 0.84,
    seed: int = 11,
    tileable: bool = True,
) -> np.ndarray:
    """Iterative downward gravity streak / grime / moss-drip simulator."""
    h, w = height_field.shape[:2]
    src = np.clip(1.0 - np.asarray(height_field, dtype=np.float32), 0.0, 1.0)
    if cavity_mask is not None:
        src = np.maximum(src, np.clip(np.asarray(cavity_mask, dtype=np.float32), 0.0, 1.0))

    # Break streaks horizontally with fine vertical noise
    streak_mod = fbm_noise(h, w, scale=max(4, w // 4), scale_y=max(2, h // 12), octaves=2, seed=seed, tileable=tileable)
    cur = src * (0.55 + 0.45 * streak_mod)
    acc = cur.copy()
    d_fac = float(np.clip(decay, 0.2, 0.98))
    for _ in range(max(1, int(streak_length))):
        shifted = np.roll(cur, 1, axis=0) if tileable else np.vstack([cur[:1], cur[:-1]])
        # Slight horizontal meander
        cur = shifted * d_fac * (0.85 + 0.15 * streak_mod)
        acc = np.maximum(acc, cur)
    return np.clip(acc * float(strength), 0.0, 1.0).astype(np.float32)


def iterate_process(
    field: np.ndarray,
    *,
    operation: str = "erode_weather",
    steps: int = 6,
    strength: float = 0.35,
    seed: int = 0,
    tileable: bool = True,
) -> np.ndarray:
    """Generic iterative feedback-loop node (mimics Pixel Composer Iterate groups).

    Operations: ``erode_weather``, ``moss_spread``, ``diffuse_smooth``, ``sharpen_cluster``, ``warp_feedback``.
    """
    cur = np.asarray(field, dtype=np.float32).copy()
    h, w = cur.shape[:2]
    st = float(strength)
    for k in range(max(1, int(steps))):
        if operation == "erode_weather":
            # Thermal + hydraulic erosion: smooth steep downward slopes & carve micro-rills
            lap = _laplacian_2d(cur, tileable=tileable)
            down = np.roll(cur, 1, axis=0) if tileable else np.vstack([cur[:1], cur[:-1]])
            slope = np.maximum(0.0, down - cur)
            cur = np.clip(cur + st * 0.25 * lap - st * 0.20 * slope, 0.0, 1.0)
        elif operation == "moss_spread":
            # Organic growth expanding outward from seeded regions modulated by noise
            noise = fbm_noise(h, w, scale=max(4, w // 4), octaves=2, seed=seed + k * 23, tileable=tileable)
            p = np.pad(cur, 1, mode="wrap" if tileable else "edge")
            nb_max = np.maximum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
            grow = np.where((nb_max > 0.45) & (noise > (0.62 - st * 0.35)), nb_max * 0.96, cur)
            cur = np.clip(np.maximum(cur, grow), 0.0, 1.0)
        elif operation == "diffuse_smooth":
            cur = np.clip(cur + st * 0.35 * _laplacian_2d(cur, tileable=tileable), 0.0, 1.0)
        elif operation == "sharpen_cluster":
            lap = _laplacian_2d(cur, tileable=tileable)
            cur = np.clip(cur - st * 0.28 * lap, 0.0, 1.0)
        elif operation == "warp_feedback":
            cur = domain_warp(cur, strength=st * 0.04, scale=4.0, octaves=2, mode="curl", seed=seed + k * 37, tileable=tileable)
    return cur.astype(np.float32)


# ---------------------------------------------------------------------------
# Pure Algorithmic (Non-Transformer, Non-ML) Denoisers & Pixel-Art Filters
# ---------------------------------------------------------------------------


def _ensure_3d(arr: np.ndarray) -> tuple[np.ndarray, bool]:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        return a[..., None], True
    return a, False


def _box_mean_sum(img: np.ndarray, r_y0: int, r_y1: int, r_x0: int, r_x1: int, *, tileable: bool) -> np.ndarray:
    """Fast vectorized mean over an offset rectangular quadrant window [r_y0..r_y1, r_x0..r_x1]."""
    pad_t, pad_b = max(0, -r_y0), max(0, r_y1)
    pad_l, pad_r = max(0, -r_x0), max(0, r_x1)
    h, w = img.shape[:2]
    mode = "wrap" if tileable else "edge"
    padded = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode=mode)
    # Integral image for O(1)-per-pixel box sum regardless of radius!
    sat = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1, img.shape[2]), dtype=np.float64)
    sat[1:, 1:] = np.cumsum(np.cumsum(padded.astype(np.float64), axis=0), axis=1)
    y0 = pad_t + r_y0
    y1 = pad_t + r_y1 + 1
    x0 = pad_l + r_x0
    x1 = pad_l + r_x1 + 1
    total = (
        sat[y1 : y1 + h, x1 : x1 + w]
        - sat[y0 : y0 + h, x1 : x1 + w]
        - sat[y1 : y1 + h, x0 : x0 + w]
        + sat[y0 : y0 + h, x0 : x0 + w]
    )
    area = float((r_y1 - r_y0 + 1) * (r_x1 - r_x0 + 1))
    return (total / area).astype(np.float32)


def anisotropic_kuwahara(
    img: np.ndarray,
    *,
    radius: int = 2,
    sharpness: float = 8.0,
    tileable: bool = False,
) -> np.ndarray:
    """Generalized Kuwahara edge-preserving painterly pixel-cluster filter (pure algorithmic!).

    Eliminates noisy micro-grit while snapping surfaces into cohesive, hand-painted pixel clusters
    with razor-sharp boundaries between scales, plates, and lighting steps.
    """
    a, was_2d = _ensure_3d(img)
    r = max(1, int(radius))
    # 4 overlapping axis quadrants + 4 diagonal sub-windows for crisp directional preservation
    quads = (
        (-r, 0, -r, 0),
        (-r, 0, 0, r),
        (0, r, -r, 0),
        (0, r, 0, r),
    )
    a2 = a * a
    means = []
    vars_l = []
    for y0, y1, x0, x1 in quads:
        m = _box_mean_sum(a, y0, y1, x0, x1, tileable=tileable)
        m2 = _box_mean_sum(a2, y0, y1, x0, x1, tileable=tileable)
        v = np.maximum(0.0, m2 - m * m)
        # Scalar luminance/RGB variance across channels (ignore alpha channel if RGBA)
        c_end = min(3, a.shape[2])
        v_sc = np.mean(v[..., :c_end], axis=-1)
        means.append(m)
        vars_l.append(v_sc)

    mean_stack = np.stack(means, axis=0)  # (4, H, W, C)
    var_stack = np.stack(vars_l, axis=0)  # (4, H, W)
    # Soft-min polynomial weighting controlled by `sharpness` (smoothly generalizes hard Kuwahara!)
    q = float(max(1.0, sharpness))
    w = np.power(np.maximum(1e-6, var_stack + 1e-4), -0.5 * q)
    w_sum = np.maximum(1e-12, np.sum(w, axis=0, keepdims=True))
    out = np.sum(mean_stack * (w / w_sum)[..., None], axis=0).astype(np.float32)

    # Preserve original alpha channel exactly if RGBA
    if a.shape[2] == 4:
        out[..., 3] = a[..., 3]
    return out[..., 0] if was_2d else out


def bilateral_denoise(
    img: np.ndarray,
    *,
    radius: int = 2,
    sigma_s: float = 1.5,
    sigma_r: float = 0.14,
    tileable: bool = False,
) -> np.ndarray:
    """Pure-NumPy vectorized edge-preserving bilateral denoiser."""
    a, was_2d = _ensure_3d(img)
    h, w, c = a.shape
    r = max(1, min(5, int(radius)))
    mode = "wrap" if tileable else "edge"
    padded = np.pad(a, ((r, r), (r, r), (0, 0)), mode=mode)
    c_cmp = min(3, c)
    inv_2s2 = 1.0 / (2.0 * max(1e-4, float(sigma_s)) ** 2)
    inv_2r2 = 1.0 / (2.0 * max(1e-4, float(sigma_r)) ** 2)

    acc = np.zeros_like(a, dtype=np.float64)
    w_tot = np.zeros((h, w, 1), dtype=np.float64)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ws = math.exp(-(dx * dx + dy * dy) * inv_2s2)
            nb = padded[r + dy : r + dy + h, r + dx : r + dx + w]
            diff = nb[..., :c_cmp] - a[..., :c_cmp]
            dist2 = np.sum(diff * diff, axis=-1, keepdims=True)
            wr = np.exp(-dist2 * inv_2r2)
            wt = ws * wr
            acc += nb * wt
            w_tot += wt

    out = (acc / np.maximum(1e-12, w_tot)).astype(np.float32)
    if c == 4:
        out[..., 3] = a[..., 3]
    return out[..., 0] if was_2d else out


def total_variation_denoise(
    img: np.ndarray,
    *,
    weight: float = 0.08,
    iterations: int = 12,
) -> np.ndarray:
    """Chambolle's projection algorithm for edge-preserving Total Variation (TV-L2) denoising.

    Flattens noisy micro-speckle into clean stepped pixel-art plateaus while preserving sharp edges!
    """
    a, was_2d = _ensure_3d(img)
    c_end = min(3, a.shape[2])
    out = a.copy()
    tau = 0.24
    lam = max(1e-4, float(weight))

    for ch in range(c_end):
        f = a[..., ch].astype(np.float64)
        px = np.zeros_like(f)
        py = np.zeros_like(f)
        for _ in range(max(1, int(iterations))):
            # div(p)
            div_p = np.zeros_like(f)
            div_p[:, 0] = px[:, 0]
            div_p[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
            div_p[:, -1] = -px[:, -2]
            div_p[0, :] += py[0, :]
            div_p[1:-1, :] += py[1:-1, :] - py[:-2, :]
            div_p[-1, :] += -py[-2, :]

            u = div_p - f / lam
            gx = np.zeros_like(f)
            gy = np.zeros_like(f)
            gx[:, :-1] = u[:, 1:] - u[:, :-1]
            gy[:-1, :] = u[1:, :] - u[:-1, :]
            norm = np.sqrt(gx * gx + gy * gy)
            denom = 1.0 + tau * norm
            px = (px + tau * gx) / denom
            py = (py + tau * gy) / denom

        div_p = np.zeros_like(f)
        div_p[:, 0] = px[:, 0]
        div_p[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
        div_p[:, -1] = -px[:, -2]
        div_p[0, :] += py[0, :]
        div_p[1:-1, :] += py[1:-1, :] - py[:-2, :]
        div_p[-1, :] += -py[-2, :]
        out[..., ch] = np.clip(f - lam * div_p, 0.0, 1.0).astype(np.float32)

    return out[..., 0] if was_2d else out


def non_local_means_denoise(
    img: np.ndarray,
    *,
    patch_radius: int = 1,
    search_radius: int = 3,
    h_param: float = 0.08,
    tileable: bool = False,
) -> np.ndarray:
    """Fast integral-image Non-Local Means (patch-similarity self-denoising, pure NumPy!)."""
    a, was_2d = _ensure_3d(img)
    H, W, C = a.shape
    c_cmp = min(3, C)
    pr = max(1, min(2, int(patch_radius)))
    sr = max(1, min(4, int(search_radius)))
    inv_h2 = 1.0 / max(1e-6, float(h_param) ** 2 * c_cmp)
    mode = "wrap" if tileable else "edge"
    pad_total = sr
    padded = np.pad(a, ((pad_total, pad_total), (pad_total, pad_total), (0, 0)), mode=mode)

    acc = np.zeros_like(a, dtype=np.float64)
    w_tot = np.zeros((H, W, 1), dtype=np.float64)
    base_rgb = a[..., :c_cmp]

    for dy in range(-sr, sr + 1):
        for dx in range(-sr, sr + 1):
            shifted = padded[pad_total + dy : pad_total + dy + H, pad_total + dx : pad_total + dx + W]
            sq_diff = np.sum((shifted[..., :c_cmp] - base_rgb) ** 2, axis=-1, keepdims=True)
            patch_dist = _box_mean_sum(sq_diff, -pr, pr, -pr, pr, tileable=tileable)
            wt = np.exp(-patch_dist * inv_h2)
            acc += shifted * wt
            w_tot += wt

    out = (acc / np.maximum(1e-12, w_tot)).astype(np.float32)
    if C == 4:
        out[..., 3] = a[..., 3]
    return out[..., 0] if was_2d else out


def morphological_despeckle(
    img: np.ndarray,
    *,
    threshold: float = 0.14,
    min_dissent: int = 7,
    tileable: bool = False,
) -> np.ndarray:
    """Pixel-art orphan pixel cleaner: replaces isolated 1-px noise specks with neighbor median."""
    a, was_2d = _ensure_3d(img)
    h, w, c = a.shape
    c_cmp = min(3, c)
    mode = "wrap" if tileable else "edge"
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode=mode)
    nbs = [
        p[0:h, 0:w], p[0:h, 1:w + 1], p[0:h, 2:w + 2],
        p[1:h + 1, 0:w], p[1:h + 1, 2:w + 2],
        p[2:h + 2, 0:w], p[2:h + 2, 1:w + 1], p[2:h + 2, 2:w + 2],
    ]
    nb_stack = np.stack(nbs, axis=0)  # (8, H, W, C)
    diff = np.linalg.norm(nb_stack[..., :c_cmp] - a[None, ..., :c_cmp], axis=-1)
    dissent_count = np.sum(diff > float(threshold), axis=0)
    orphan = dissent_count >= int(min_dissent)
    med = np.median(nb_stack, axis=0).astype(np.float32)
    out = np.where(orphan[..., None], med, a)
    if c == 4:
        out[..., 3] = a[..., 3]
    return out[..., 0] if was_2d else out


def pixel_cluster_mode(img: np.ndarray, *, tileable: bool = False) -> np.ndarray:
    """Majority-vote 3x3 pixel-cluster consolidator (cleans jagged 1px noise after quantization)."""
    a, was_2d = _ensure_3d(img)
    h, w, c = a.shape
    mode = "wrap" if tileable else "edge"
    p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode=mode)
    # 5-pixel cross + center weighted median preserves sharp horizontal/vertical pixel-art lines!
    cross = np.stack(
        [a, a, p[0:h, 1:w + 1], p[2:h + 2, 1:w + 1], p[1:h + 1, 0:w], p[1:h + 1, 2:w + 2]],
        axis=0,
    )
    out = np.median(cross, axis=0).astype(np.float32)
    if c == 4:
        out[..., 3] = a[..., 3]
    return out[..., 0] if was_2d else out
