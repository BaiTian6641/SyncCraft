"""Pixel-Composer-Inspired `Iterate -> Simulate -> Animate` Engine for `synccraft.texgraph` v5.

Pure deterministic vectorized NumPy + Pillow (zero ML dependencies):
- **Animate Drivers (`eval_anim_driver`)**: `wavetable` (`sine`, `triangle`, `sawtooth`, `square`, `pulse`),
  seamless-looping `wiggler` (circular 1D gradient noise), and eased `keyframes` curves (`linear`,
  `ease_in`, `ease_out`, `ease_in_out`, `smoothstep`, `bounce`) driving any node parameter over `phase in [0, 1)`.
- **Iterate Inline (`iterate_inline`)**: Multi-step recursive operator accumulation with per-step
  transform/warp/decay/blend progression (`scale_echo`, `warp_cascade`, `trail_stack`, `kuwahara_stack`,
  `erode_weather`, `ridge_sharpen`, `moss_spread`).
- **Feedback Inline (`feedback_step_sequence`)**: True stateful frame-to-frame feedback buffer with
  configurable warm-up pre-roll so looping `.mcmeta` flipbooks maintain seamless temporal trails,
  advection drift, diffusion blur, and decay from frame 0 to frame N-1.
- **Particle Simulation (`particle_sim_field`)**: Deterministic pixel-art particle emitter/simulator
  with mask/point/line/ring spawn, gravity, curl turbulence, radial vortex attractor, drag, life ramp,
  and seamless loop wrapping.
- **Fluid / Smoke Advection (`fluid_smoke_sim_field`)**: Fast 2D semi-Lagrangian buoyancy + curl-velocity
  smoke, flame, lava & arcane plasma simulation with seamless tiling and periodic loop blending.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .generators_gradients import multi_stop_gradient
from .generators_noise import domain_warp, fbm_noise, uv_grid
from .ops_fx_color import gaussian_blur, tile_transform
from .pixel_art_ops import blend_layers
from .simulation_filters import anisotropic_kuwahara, iterate_process


# ---------------------------------------------------------------------------
# 1. ANIMATE: Procedural Timeline Drivers (Wavetable / Wiggler / Keyframes)
# ---------------------------------------------------------------------------


def _ease_val(t: float, mode: str) -> float:
    t = max(0.0, min(1.0, float(t)))
    if mode == "step":
        return 0.0 if t < 1.0 else 1.0
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if mode in ("ease_in_out", "smoothstep", "smooth"):
        return t * t * (3.0 - 2.0 * t)
    if mode == "smootherstep":
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    if mode == "bounce":
        if t < 1.0 / 2.75:
            return 7.5625 * t * t
        if t < 2.0 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        if t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375
    return t


def _eval_monotone_cubic_spline(pts: list[tuple[float, float]], ph: float) -> float:
    """Fritsch-Carlson monotone cubic Hermite spline (overshoot-free smooth animation curve!)."""
    n = len(pts)
    if n == 1:
        return pts[0][1]
    if ph <= pts[0][0]:
        return pts[0][1]
    if ph >= pts[-1][0]:
        return pts[-1][1]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ds = [(ys[i + 1] - ys[i]) / max(1e-6, xs[i + 1] - xs[i]) for i in range(n - 1)]
    ms = [0.0] * n
    ms[0] = ds[0]
    ms[-1] = ds[-1]
    for i in range(1, n - 1):
        if ds[i - 1] * ds[i] <= 0.0:
            ms[i] = 0.0
        else:
            ms[i] = 0.5 * (ds[i - 1] + ds[i])
    for i in range(n - 1):
        if abs(ds[i]) < 1e-8:
            ms[i] = 0.0
            ms[i + 1] = 0.0
        else:
            a = ms[i] / ds[i]
            b = ms[i + 1] / ds[i]
            s = math.hypot(a, b)
            if s > 3.0:
                ms[i] = (3.0 / s) * a * ds[i]
                ms[i + 1] = (3.0 / s) * b * ds[i]
    for i in range(n - 1):
        if xs[i] <= ph <= xs[i + 1]:
            h = max(1e-6, xs[i + 1] - xs[i])
            u = (ph - xs[i]) / h
            u2 = u * u
            u3 = u2 * u
            h00 = 2.0 * u3 - 3.0 * u2 + 1.0
            h10 = u3 - 2.0 * u2 + u
            h01 = -2.0 * u3 + 3.0 * u2
            h11 = u3 - u2
            return h00 * ys[i] + h10 * h * ms[i] + h01 * ys[i + 1] + h11 * h * ms[i + 1]
    return ys[-1]


def eval_anim_driver(
    phase: float,
    *,
    kind: str = "sine",
    min_val: float = 0.0,
    max_val: float = 1.0,
    freq: float = 1.0,
    phase_offset: float = 0.0,
    duty: float = 0.5,
    octaves: int = 3,
    seed: int = 42,
    keyframes: Sequence[Sequence[Any]] | None = None,
) -> float:
    """Evaluate a Pixel-Composer-style procedural animation driver at normalized ``phase in [0, 1]``."""
    raw_ph = float(phase) * float(freq) + float(phase_offset)
    if kind in ("keyframes", "bezier_curve", "spline") and keyframes:
        if 0.0 <= raw_ph <= 1.0:
            ph = raw_ph
        else:
            ph = raw_ph % 1.0
    else:
        ph = raw_ph % 1.0
    lo, hi = float(min_val), float(max_val)

    if kind in ("keyframes", "bezier_curve", "spline") and keyframes:
        pts = sorted(keyframes, key=lambda k: float(k[0]))
        if ph <= float(pts[0][0]):
            return float(pts[0][1])
        if ph >= float(pts[-1][0]):
            return float(pts[-1][1])
        first_mode = str(pts[0][2]).lower() if len(pts[0]) >= 3 else ("spline" if kind in ("bezier_curve", "spline") else "smoothstep")
        if kind in ("bezier_curve", "spline") or first_mode in ("spline", "bezier"):
            return float(_eval_monotone_cubic_spline([(float(k[0]), float(k[1])) for k in pts], ph))
        for i in range(len(pts) - 1):
            t0, v0 = float(pts[i][0]), float(pts[i][1])
            t1, v1 = float(pts[i + 1][0]), float(pts[i + 1][1])
            if t0 <= ph <= t1:
                ease_m = str(pts[i][2]).lower() if len(pts[i]) >= 3 else "smoothstep"
                if ease_m in ("spline", "bezier"):
                    return float(_eval_monotone_cubic_spline([(float(k[0]), float(k[1])) for k in pts], ph))
                u = (ph - t0) / max(1e-6, t1 - t0)
                return v0 + (v1 - v0) * _ease_val(u, ease_m)
        return float(pts[-1][1])

    if kind == "triangle":
        n = 1.0 - abs(2.0 * ph - 1.0)
    elif kind == "sawtooth":
        n = ph
    elif kind == "square":
        n = 1.0 if ph < float(duty) else 0.0
    elif kind == "wiggler":
        # Seamless circular 1D harmonic noise (100% periodic at phase 0 == phase 1!)
        rng = np.random.default_rng(int(seed))
        val = 0.0
        amp = 1.0
        amp_sum = 0.0
        for oct_i in range(1, max(1, int(octaves)) + 1):
            ang = 2.0 * math.pi * ph * oct_i
            p_off = float(rng.uniform(0.0, 2.0 * math.pi))
            val += amp * math.sin(ang + p_off)
            amp_sum += amp
            amp *= 0.55
        n = 0.5 + 0.5 * (val / max(1e-6, amp_sum))
    else:  # "sine"
        n = 0.5 + 0.5 * math.sin(2.0 * math.pi * ph)

    return lo + (hi - lo) * max(0.0, min(1.0, float(n)))


# ---------------------------------------------------------------------------
# 2. ITERATE: Inline Multi-Step Recursive Operator Loop
# ---------------------------------------------------------------------------


def iterate_inline(
    src: np.ndarray,
    *,
    mode: str = "warp_cascade",
    steps: int = 5,
    strength: float = 0.35,
    decay: float = 0.82,
    step_dx: float = 0.0,
    step_dy: float = 0.0,
    step_scale: float = 1.0,
    step_rotate_deg: float = 0.0,
    blend_mode: str = "screen",
    seed: int = 42,
    tileable: bool = False,
    phase: float = 0.0,
) -> np.ndarray:
    """Pixel-Composer-style `Iterate Inline` loop: applies a progressive operator chain across ``steps``."""
    arr = np.asarray(src, dtype=np.float32).copy()
    n = max(1, int(steps))

    if mode in ("erode_weather", "ridge_sharpen", "moss_spread"):
        return iterate_process(arr, operation=mode, steps=n, strength=strength, seed=seed, tileable=tileable)

    if mode == "kuwahara_stack":
        cur = arr
        for _ in range(min(n, 4)):
            cur = anisotropic_kuwahara(cur, radius=max(1, int(round(1 + strength * 2.0))), sharpness=9.0, tileable=tileable)
        return cur.astype(np.float32)

    if mode == "warp_cascade":
        cur = arr
        w_str = float(strength) * 0.25
        for k in range(n):
            cur = domain_warp(
                cur,
                strength=w_str * (float(decay) ** k),
                scale=3.0 + 1.5 * k,
                octaves=2,
                mode="curl" if (k % 2 == 1) else "fbm",
                seed=int(seed) + k * 37,
                tileable=tileable,
                phase=phase,
                flow_speed=1.0 if abs(phase) > 1e-6 else 0.0,
            )
        return np.clip(cur, 0.0, 1.0).astype(np.float32)

    # "scale_echo" / "trail_stack": accumulate transformed echoes with progressive decay
    acc = arr.copy()
    cur = arr.copy()
    weight = float(strength)
    for k in range(1, n + 1):
        prog = float(k) / float(n)
        cur = tile_transform(
            arr,
            scale_x=float(step_scale) ** k,
            scale_y=float(step_scale) ** k,
            offset_x=float(step_dx) * k,
            offset_y=float(step_dy) * k,
            rotate_deg=float(step_rotate_deg) * k,
            tileable=tileable,
        )
        op = max(0.02, min(1.0, weight * (float(decay) ** (k - 1)) * (1.0 - 0.15 * prog)))
        acc = blend_layers(acc, cur, mode=blend_mode, opacity=op)
    return np.clip(acc, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. SIMULATE: Stateful Frame-to-Frame Feedback Loop (Seamless Pre-Roll!)
# ---------------------------------------------------------------------------


def feedback_step_sequence(
    frame_emitter_fn: Any,
    *,
    num_frames: int = 16,
    pre_roll: int = 8,
    decay: float = 0.80,
    drift_x_px: float = 0.0,
    drift_y_px: float = -1.0,
    blur_radius: int = 1,
    blend_mode: str = "screen",
    tileable: bool = True,
) -> list[np.ndarray]:
    """Pixel-Composer-style `Feedback Inline` temporal state simulator with warm-up pre-roll.

    Runs ``pre_roll + num_frames`` steps, feeding the previous frame back through drift,
    optional diffusion blur, and decay before compositing the current emitted frame.
    Returns the final ``num_frames`` seamlessly looping frames!
    """
    n = max(1, int(num_frames))
    pre = max(0, int(pre_roll))
    state: np.ndarray | None = None
    out: list[np.ndarray] = []

    for step in range(pre + n):
        ph = float(step % n) / float(n)
        cur = np.asarray(frame_emitter_fn(ph), dtype=np.float32)
        if state is None:
            state = cur.copy()
        else:
            prev = state
            sx = int(round(float(drift_x_px)))
            sy = int(round(float(drift_y_px)))
            if sx != 0 or sy != 0:
                prev = np.roll(np.roll(prev, sy, axis=0), sx, axis=1)
                if not tileable:
                    if sy > 0:
                        prev[:sy] = 0.0
                    elif sy < 0:
                        prev[sy:] = 0.0
                    if sx > 0:
                        prev[:, :sx] = 0.0
                    elif sx < 0:
                        prev[:, sx:] = 0.0
            if int(blur_radius) > 0:
                prev = gaussian_blur(prev, radius=int(blur_radius), tileable=tileable)
            if prev.ndim == 3 and prev.shape[2] == 4:
                prev = prev.copy()
                prev[..., :3] *= float(decay)
                prev[..., 3] *= float(decay)
            else:
                prev = prev * float(decay)
            state = blend_layers(prev, cur, mode=blend_mode, opacity=1.0)
        if step >= pre:
            out.append(np.clip(state, 0.0, 1.0).astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# 4. SIMULATE: Deterministic Vectorized Pixel Particle System
# ---------------------------------------------------------------------------


def particle_sim_field(
    height: int,
    width: int,
    *,
    count: int = 48,
    emitter: str = "bottom",
    emitter_mask: np.ndarray | None = None,
    vel_x: float = 0.0,
    vel_y: float = -0.55,
    spread: float = 0.25,
    gravity_y: float = -0.18,
    turbulence: float = 0.22,
    vortex: float = 0.0,
    size_px: float = 1.6,
    trail_steps: int = 3,
    phase: float = 0.0,
    seed: int = 42,
    tileable: bool = True,
    color_stops: Sequence[tuple[float, str]] | None = None,
) -> dict[str, np.ndarray]:
    """Deterministic Pixel-Composer-style Particle System (100% seamless loop over ``phase in [0, 1)``!).

    Returns ``{"rgba", "density", "heat"}`` so particles can feed either RGBA glow layers OR scalar masks!
    """
    H, W = max(1, int(height)), max(1, int(width))
    n_p = max(1, int(count))
    rng = np.random.default_rng(int(seed))

    # Spawn positions (normalized [0, 1]^2)
    if emitter_mask is not None:
        m = np.asarray(emitter_mask, dtype=np.float32)
        ys, xs = np.nonzero(m > 0.25)
        if len(xs) > 0:
            pick = rng.integers(0, len(xs), size=n_p)
            x0 = (xs[pick].astype(np.float32) + rng.uniform(0.0, 1.0, n_p).astype(np.float32)) / float(W)
            y0 = (ys[pick].astype(np.float32) + rng.uniform(0.0, 1.0, n_p).astype(np.float32)) / float(H)
        else:
            x0 = rng.uniform(0.0, 1.0, n_p).astype(np.float32)
            y0 = rng.uniform(0.0, 1.0, n_p).astype(np.float32)
    elif emitter == "center":
        ang = rng.uniform(0.0, 2.0 * math.pi, n_p).astype(np.float32)
        rad = rng.uniform(0.0, 0.08, n_p).astype(np.float32)
        x0 = 0.5 + rad * np.cos(ang)
        y0 = 0.5 + rad * np.sin(ang)
    elif emitter == "ring":
        ang = rng.uniform(0.0, 2.0 * math.pi, n_p).astype(np.float32)
        rad = rng.uniform(0.26, 0.34, n_p).astype(np.float32)
        x0 = 0.5 + rad * np.cos(ang)
        y0 = 0.5 + rad * np.sin(ang)
    elif emitter == "full":
        x0 = rng.uniform(0.0, 1.0, n_p).astype(np.float32)
        y0 = rng.uniform(0.0, 1.0, n_p).astype(np.float32)
    else:  # "bottom"
        x0 = rng.uniform(0.0, 1.0, n_p).astype(np.float32)
        y0 = rng.uniform(0.82, 0.98, n_p).astype(np.float32)

    birth = rng.uniform(0.0, 1.0, n_p).astype(np.float32)
    vx = float(vel_x) + rng.normal(0.0, float(spread), n_p).astype(np.float32)
    vy = float(vel_y) + rng.normal(0.0, float(spread) * 0.65, n_p).astype(np.float32)
    wob_f = rng.uniform(1.0, 3.0, n_p).astype(np.float32)
    wob_p = rng.uniform(0.0, 2.0 * math.pi, n_p).astype(np.float32)
    rad_scale = rng.uniform(0.7, 1.35, n_p).astype(np.float32)

    density = np.zeros((H, W), dtype=np.float32)
    heat = np.zeros((H, W), dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    n_tr = max(1, int(trail_steps))
    ph = float(phase) % 1.0

    for tr in range(n_tr):
        dt_tr = tr * 0.032
        age = (ph - birth - dt_tr) % 1.0  # seamless periodic age in [0, 1)!
        # Smooth fade-in and fade-out at birth/death so looping has ZERO pop!
        life_env = np.sin(np.pi * age) ** 0.85
        tr_w = (0.68 ** tr) * life_env

        px = x0 + vx * age + float(turbulence) * 0.14 * np.sin(2.0 * math.pi * age * wob_f + wob_p)
        py = y0 + vy * age + 0.5 * float(gravity_y) * (age ** 2) + float(turbulence) * 0.06 * np.cos(2.0 * math.pi * age * wob_f + wob_p)

        if abs(float(vortex)) > 1e-4:
            dx_c = px - 0.5
            dy_c = py - 0.5
            ang_v = float(vortex) * age * 2.0
            cv, sv = np.cos(ang_v), np.sin(ang_v)
            px = 0.5 + dx_c * cv - dy_c * sv
            py = 0.5 + dx_c * sv + dy_c * cv

        if tileable:
            px = px % 1.0
            py = py % 1.0

        # Splat particles crisply on the pixel grid
        for i in range(n_p):
            w_i = float(tr_w[i])
            if w_i < 0.03:
                continue
            cx = float(px[i]) * W
            cy = float(py[i]) * H
            r = max(0.7, float(size_px) * float(rad_scale[i]) * (1.0 - 0.35 * float(age[i])) * (0.85 ** tr))
            if not tileable and (cx < -r or cx > W + r or cy < -r or cy > H + r):
                continue
            dx = xx - cx
            dy = yy - cy
            if tileable:
                dx = ((dx + 0.5 * W) % W) - 0.5 * W
                dy = ((dy + 0.5 * H) % H) - 0.5 * H
            d2 = dx * dx + dy * dy
            splat = np.clip(1.0 - np.sqrt(d2) / max(0.5, r), 0.0, 1.0) * w_i
            density = np.maximum(density, splat)
            # Hotter particles when younger (1 - age)
            heat = np.maximum(heat, splat * (1.0 - 0.65 * float(age[i])))

    density = np.clip(density, 0.0, 1.0).astype(np.float32)
    heat = np.clip(heat, 0.0, 1.0).astype(np.float32)
    stops = color_stops or ((0.0, "#4C1D95"), (0.35, "#EA580C"), (0.72, "#FACC15"), (1.0, "#FFFFFF"))
    rgba = multi_stop_gradient(heat, stops, space="oklab", smooth_blend=0.85, alpha_mask=(density > 0.08).astype(np.float32))
    rgba[..., 3] = np.clip(density * 1.35, 0.0, 1.0)
    return {"rgba": rgba.astype(np.float32), "density": density, "heat": heat}


# ---------------------------------------------------------------------------
# 5. SIMULATE: Fast 2D Semi-Lagrangian Smoke / Flame / Plasma Fluid Solver
# ---------------------------------------------------------------------------


def _advect_2d(field: np.ndarray, vx: np.ndarray, vy: np.ndarray, dt: float, *, tileable: bool) -> np.ndarray:
    H, W = field.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx = xx - vx * float(dt)
    sy = yy - vy * float(dt)
    if tileable:
        sx = sx % W
        sy = sy % H
    else:
        sx = np.clip(sx, 0.0, W - 1.001)
        sy = np.clip(sy, 0.0, H - 1.001)
    x0 = np.floor(sx).astype(np.int64)
    y0 = np.floor(sy).astype(np.int64)
    fx = sx - x0
    fy = sy - y0
    if tileable:
        x1 = (x0 + 1) % W
        y1 = (y0 + 1) % H
    else:
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
    f00 = field[y0, x0]
    f10 = field[y0, x1]
    f01 = field[y1, x0]
    f11 = field[y1, x1]
    return (
        f00 * (1.0 - fx) * (1.0 - fy)
        + f10 * fx * (1.0 - fy)
        + f01 * (1.0 - fx) * fy
        + f11 * fx * fy
    ).astype(np.float32)


def fluid_smoke_sim_field(
    height: int,
    width: int,
    *,
    source_mask: np.ndarray | None = None,
    steps: int = 18,
    buoyancy: float = 1.65,
    curl_strength: float = 1.85,
    dissipation: float = 0.93,
    viscosity: float = 0.15,
    phase: float = 0.0,
    seed: int = 42,
    tileable: bool = True,
    color_stops: Sequence[tuple[float, str]] | None = None,
) -> dict[str, np.ndarray]:
    """Deterministic 2D Semi-Lagrangian Smoke / Fire / Arcane Plasma Fluid Simulation (Pure NumPy!).

    Uses divergence-free curl-noise velocity + thermal buoyancy + semi-Lagrangian dye advection
    with periodic loop cross-fading so animated flipbooks wrap seamlessly!
    Returns ``{"rgba", "density", "temperature"}``.
    """
    H, W = max(1, int(height)), max(1, int(width))
    u, v = uv_grid(H, W)
    if source_mask is not None:
        src = np.clip(np.asarray(source_mask, dtype=np.float32), 0.0, 1.0)
    else:
        # Classic centered glowing plume source at bottom
        src = np.clip(1.0 - ((u - 0.5) / 0.24) ** 2 - ((v - 0.86) / 0.14) ** 2, 0.0, 1.0).astype(np.float32)

    n_steps = max(4, int(steps))
    ph = float(phase) % 1.0

    # Divergence-free velocity field derived from spatial curl of looping potential psi(x, y, phase)!
    psi = fbm_noise(H, W, scale=3.2, octaves=3, mode="fbm", seed=int(seed), tileable=tileable, phase=ph, loop_speed=1.0)
    if tileable:
        dpsi_dy = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * 0.5
        dpsi_dx = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * 0.5
    else:
        dpsi_dy = np.gradient(psi, axis=0)
        dpsi_dx = np.gradient(psi, axis=1)

    vx = (dpsi_dy * float(curl_strength) * (W / 4.0)).astype(np.float32)
    vy = (-dpsi_dx * float(curl_strength) * (H / 4.0) - float(buoyancy) * (H / 16.0)).astype(np.float32)

    density = src.copy()
    temp = src.copy()
    dt = 0.55
    diss = float(np.clip(dissipation, 0.55, 0.995))

    for s in range(n_steps):
        # Slight phase swirl per substep so vortices curl richly upward
        ang = 2.0 * math.pi * (ph + s / float(n_steps))
        vx_s = vx + 0.35 * math.sin(ang) * float(curl_strength)
        vy_s = vy - 0.45 * temp * float(buoyancy)
        density = _advect_2d(density, vx_s, vy_s, dt, tileable=tileable) * diss
        temp = _advect_2d(temp, vx_s * 1.1, vy_s * 1.15, dt, tileable=tileable) * (diss * 0.96)
        if float(viscosity) > 1e-3 and (s % 4 == 3):
            density = density * (1.0 - float(viscosity)) + gaussian_blur(density, radius=1, tileable=tileable) * float(viscosity)
        # Re-inject glowing emitter core
        density = np.clip(np.maximum(density, src), 0.0, 1.0)
        temp = np.clip(np.maximum(temp, src), 0.0, 1.0)

    density = np.clip(density, 0.0, 1.0).astype(np.float32)
    temp = np.clip(temp, 0.0, 1.0).astype(np.float32)
    stops = color_stops or ((0.0, "#12071F"), (0.28, "#7F1D1D"), (0.56, "#EA580C"), (0.82, "#FACC15"), (1.0, "#FFFBEB"))
    rgba = multi_stop_gradient(
        np.clip(0.55 * density + 0.45 * temp, 0.0, 1.0),
        stops,
        space="oklab",
        smooth_blend=0.78,
        quantize_steps=7,
        alpha_mask=(density > 0.12).astype(np.float32),
    )
    rgba[..., 3] = np.clip((density - 0.08) / 0.35, 0.0, 1.0)
    return {"rgba": rgba.astype(np.float32), "density": density, "temperature": temp}
