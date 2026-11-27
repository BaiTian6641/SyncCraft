"""Automated Multi-Domain Studio Texture Quality Evaluation Harness (`TexGraph v7`).

Quantitatively audits generated textures across ALL THREE production domains so our engine
can run true closed-loop recursive self-improvement AND enforce hard CI/production gates:
- ``domain="block"``  : Seamless toroidal border wrap error, `SKILL.md §10.2` painterly smoothness/relief,
                        value range (no crushed blacks / blown highlights), hue-shift richness, PBR validity.
- ``domain="item"``   : 100% hard-alpha silhouette purity, framing coverage, Brandon James Greer directional
                        `selout` modulation (not dead-black uniform outline!), low orphan speckle, highlight pop.
- ``domain="entity"`` : `SKILL.md §10.2` interior-patch `mean_lum_step`, `jump_gt25_pct`, `calm_interior_pct`,
                        3D cross-seam balance, PBR physical map sanity (unit-length normals, finite roughness/metallic).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .pixel_art_ops import measure_painterly_metrics, rgba_float_to_u8


@dataclass
class QualityReport:
    domain: str
    score: float  # 0.0 .. 100.0
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_pbr_validity(pbr_channels: dict[str, np.ndarray] | None, pbr_maps: dict[str, np.ndarray] | None) -> tuple[float, list[str]]:
    issues: list[str] = []
    penalty = 0.0
    if pbr_channels:
        for k, arr in pbr_channels.items():
            a = np.asarray(arr, dtype=np.float32)
            if not np.isfinite(a).all():
                issues.append(f"Non-finite values in PBR channel {k!r}")
                penalty += 25.0
            if k in ("roughness", "metallic", "ao", "height"):
                if float(a.min()) < -0.02 or float(a.max()) > 1.02:
                    issues.append(f"Out-of-range PBR channel {k!r}: [{float(a.min()):.2f}, {float(a.max()):.2f}]")
                    penalty += 8.0
    if pbr_maps and "normal_rgb" in pbr_maps:
        nrm = np.asarray(pbr_maps["normal_rgb"], dtype=np.float32)
        if nrm.max() > 1.5:
            nrm = nrm / 255.0
        v = nrm[..., :3] * 2.0 - 1.0
        lens = np.linalg.norm(v, axis=-1)
        bad_len = float(np.mean(np.abs(lens - 1.0) > 0.18))
        if bad_len > 0.05:
            issues.append(f"Normal map has {bad_len * 100:.1f}% non-unit vectors")
            penalty += 10.0
    return penalty, issues


def _hue_shift_span_deg(rgb_f: np.ndarray, mask: np.ndarray) -> float:
    """Measures perceptual hue spread (degrees) between dark shadows & bright highlights."""
    if not np.any(mask):
        return 0.0
    pix = rgb_f[mask]
    lum = 0.299 * pix[:, 0] + 0.587 * pix[:, 1] + 0.114 * pix[:, 2]
    mx = pix.max(axis=-1)
    mn = pix.min(axis=-1)
    sat = (mx - mn) / np.maximum(1e-5, mx)
    chroma = sat > 0.08
    if int(chroma.sum()) < 8:
        return 0.0
    p = pix[chroma]
    l_c = lum[chroma]
    # Compute HSV hue in degrees [0, 360)
    mx_c = p.max(axis=-1)
    mn_c = p.min(axis=-1)
    d = np.maximum(1e-6, mx_c - mn_c)
    h = np.zeros(len(p), dtype=np.float32)
    r_m = mx_c == p[:, 0]
    g_m = (~r_m) & (mx_c == p[:, 1])
    b_m = (~r_m) & (~g_m)
    h[r_m] = (60.0 * ((p[r_m, 1] - p[r_m, 2]) / d[r_m])) % 360.0
    h[g_m] = 60.0 * ((p[g_m, 2] - p[g_m, 0]) / d[g_m] + 2.0)
    h[b_m] = 60.0 * ((p[b_m, 0] - p[b_m, 1]) / d[b_m] + 4.0)

    q25 = float(np.percentile(l_c, 25))
    q75 = float(np.percentile(l_c, 75))
    sh_h = h[l_c <= q25]
    hi_h = h[l_c >= q75]
    if sh_h.size == 0 or hi_h.size == 0:
        return 0.0
    # Circular mean difference
    def _cmean(deg: np.ndarray) -> float:
        r = np.deg2rad(deg)
        return float(np.rad2deg(np.arctan2(np.mean(np.sin(r)), np.mean(np.cos(r))))) % 360.0

    diff = abs(_cmean(hi_h) - _cmean(sh_h))
    return float(min(diff, 360.0 - diff))


def evaluate_texture_quality(
    albedo: np.ndarray,
    *,
    domain: str = "block",
    tileable: bool = True,
    mesh_ctx: Any | None = None,
    pbr_channels: dict[str, np.ndarray] | None = None,
    pbr_maps: dict[str, np.ndarray] | None = None,
) -> QualityReport:
    """Comprehensive industry-grade texture quality audit for blocks, items & entities."""
    arr = np.asarray(albedo, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.stack((arr, arr, arr, np.ones_like(arr)), axis=-1)
    elif arr.shape[2] == 3:
        arr = np.concatenate((arr, np.ones((*arr.shape[:2], 1), dtype=np.float32)), axis=-1)

    H, W = arr.shape[:2]
    rgb = np.clip(arr[..., :3], 0.0, 1.0)
    alpha = np.clip(arr[..., 3], 0.0, 1.0)
    u8 = rgba_float_to_u8(arr)

    score = 100.0
    issues: list[str] = []
    recs: list[str] = []
    metrics: dict[str, float] = {}

    if not np.isfinite(arr).all():
        return QualityReport(domain=domain, score=0.0, passed=False, issues=["Output contains NaN/Inf"], recommendations=["Fix non-finite node math"])

    pbr_pen, pbr_iss = _check_pbr_validity(pbr_channels, pbr_maps)
    score -= pbr_pen
    issues.extend(pbr_iss)

    solid = alpha >= 0.5
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) * 255.0
    if np.any(solid):
        p5 = float(np.percentile(lum[solid], 5))
        p95 = float(np.percentile(lum[solid], 95))
        val_span = p95 - p5
    else:
        p5 = p95 = val_span = 0.0
    metrics["lum_p5"] = round(p5, 2)
    metrics["lum_p95"] = round(p95, 2)
    metrics["value_span"] = round(val_span, 2)

    hue_span = _hue_shift_span_deg(rgb, solid)
    metrics["hue_shift_deg"] = round(hue_span, 2)

    dom = domain.lower()
    if dom == "block":
        pm = measure_painterly_metrics(u8, mesh_ctx=None)
        metrics.update(pm)
        # 1. Seamless toroidal tiling continuity check (wrap seam vs interior baseline!)
        if tileable:
            # Compare boundary step (col W-1 -> col 0 and row H-1 -> row 0) to interior average step
            dx_int = np.abs(lum[:, 1:] - lum[:, :-1])
            dy_int = np.abs(lum[1:, :] - lum[:-1, :])
            int_step = float(0.5 * (dx_int.mean() + dy_int.mean()))
            seam_x = float(np.abs(lum[:, 0] - lum[:, -1]).mean())
            seam_y = float(np.abs(lum[0, :] - lum[-1, :]).mean())
            seam_delta = max(seam_x, seam_y)
            seam_ratio = seam_delta / max(2.0, int_step)
            metrics["seamless_edge_delta"] = round(seam_delta, 2)
            metrics["seamless_ratio"] = round(seam_ratio, 2)
            if seam_ratio > 2.35 and seam_delta > 18.0:
                score -= min(28.0, (seam_ratio - 2.0) * 12.0)
                issues.append(f"Visible tiling seam (edge delta={seam_delta:.1f}, ratio={seam_ratio:.2f}x interior)")
                recs.append("Enable tileable=True on all noise/blur/sampler nodes and check Flood Fill wrap")

        # 2. Painterly smoothness vs crisp readability (block pixels allow slightly livelier contrast at 16x16)
        max_step = 18.0 if min(H, W) <= 20 else 13.5
        max_jump = 24.0 if min(H, W) <= 20 else 18.0
        if pm["mean_lum_step"] < 3.0:
            score -= 14.0
            issues.append(f"Block reads flat/plastic (mean_lum_step={pm['mean_lum_step']:.2f})")
            recs.append("Increase Flood Fill random variation, Slope Blur relief, or curvature edge highlights")
        elif pm["mean_lum_step"] > max_step:
            score -= min(20.0, (pm["mean_lum_step"] - max_step) * 2.0)
            issues.append(f"Block reads noisy/busy (mean_lum_step={pm['mean_lum_step']:.2f})")
            recs.append("Consolidate into larger 3-5px pixel clusters and lower high-frequency noise")
        if pm["jump_gt25_pct"] > max_jump:
            score -= min(18.0, (pm["jump_gt25_pct"] - max_jump) * 1.2)
            issues.append(f"Excessive harsh pixel contrast jumps ({pm['jump_gt25_pct']:.1f}%)")
            recs.append("Soften seam contrast or raise smooth_blend")

        if val_span < 32.0:
            score -= 12.0
            issues.append(f"Low tonal contrast (value_span={val_span:.1f})")
            recs.append("Widen OKLab ramp value range between shadow crevices and lit top-left bevels")

    elif dom == "item":
        # 1. Hard binary alpha silhouette purity (zero muddy translucent anti-alias fringe!)
        semi = float(((alpha > 0.04) & (alpha < 0.96)).mean() * 100.0)
        fill_pct = float(solid.mean() * 100.0)
        metrics["semi_transparent_pct"] = round(semi, 2)
        metrics["silhouette_fill_pct"] = round(fill_pct, 2)
        if semi > 2.5:
            score -= min(22.0, semi * 2.5)
            issues.append(f"Muddy semi-transparent fringe ({semi:.1f}% pixels)")
            recs.append("Use hard alpha threshold + directional_selout for crisp Minecraft sprite edges")
        if fill_pct < 12.0 or fill_pct > 78.0:
            score -= 15.0
            issues.append(f"Poor item framing fill ({fill_pct:.1f}%, target 16%-72%)")
            recs.append("Scale SDF silhouette so sprite sits comfortably inside canvas with 1-2px margin")

        # 2. Directional Selout check: boundary pixels should modulate lighter (top-left) vs darker (bottom-right)
        p = np.pad(solid, 1, mode="constant", constant_values=False)
        border = solid & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
        if np.any(border):
            b_lum = lum[border]
            b_std = float(b_lum.std())
            b_mean = float(b_lum.mean())
            metrics["outline_lum_mean"] = round(b_mean, 2)
            metrics["outline_lum_std"] = round(b_std, 2)
            if b_std < 4.5 and b_mean < 18.0:
                score -= 12.0
                issues.append("Uniform dead-black outline instead of directional selective outline (selout)")
                recs.append("Apply directional_selout so lit edges shift warmer/lighter than shadow undersides")
        # 3. Orphan 1px speckle check (isolated single-pixel background specks!)
        neigh = p[:-2, 1:-1].astype(np.int32) + p[2:, 1:-1].astype(np.int32) + p[1:-1, :-2].astype(np.int32) + p[1:-1, 2:].astype(np.int32)
        orphans = int((solid & (neigh == 0)).sum())
        metrics["orphan_pixels"] = float(orphans)
        if orphans > 2:
            score -= min(14.0, orphans * 3.0)
            issues.append(f"{orphans} stray isolated 1px orphan pixels outside main silhouette")
            recs.append("Clean silhouette mask with morph_op or despeckle")

        if val_span < 55.0:
            score -= 12.0
            issues.append(f"Item lacks specular highlight pop (value_span={val_span:.1f})")
            recs.append("Add diagonal specular ridge & crisp top-left bevel catchlights")

    else:  # "entity" (real-life animals & fantasy creatures)
        pm = measure_painterly_metrics(u8, mesh_ctx=mesh_ctx)
        metrics.update(pm)
        if pm["mean_lum_step"] < 3.0:
            score -= 14.0
            issues.append(f"Entity surface too smooth/airbrushed (mean_lum_step={pm['mean_lum_step']:.2f}, target 3.5-8.5)")
            recs.append("Increase organic scale/fur/feather cluster relief & 2-3px painterly brushwork")
        elif pm["mean_lum_step"] > 9.5:
            score -= min(22.0, (pm["mean_lum_step"] - 9.5) * 3.0)
            issues.append(f"Entity surface reads busy/dirty (mean_lum_step={pm['mean_lum_step']:.2f}, target 3.5-8.5)")
            recs.append("Keep >=60% calm base fields and reduce micro-noise")
        if pm["jump_gt25_pct"] > 15.0:
            score -= min(20.0, (pm["jump_gt25_pct"] - 15.0) * 1.5)
            issues.append(f"Too many harsh jumps >25 ({pm['jump_gt25_pct']:.1f}%, target <=14%)")
            recs.append("Replace hard wireframe seams with soft domain-warped overlap shadows")
        if pm["calm_interior_pct"] < 55.0:
            score -= 12.0
            issues.append(f"Insufficient calm breathing fields ({pm['calm_interior_pct']:.1f}%, target >=60%)")
            recs.append("Let solid hue-shifted color clusters breathe across flanks and midtones")
        if val_span < 42.0:
            score -= 10.0
            issues.append(f"Narrow value contrast (value_span={val_span:.1f})")
            recs.append("Deepen cool shadow AO and brighten top-facing dorsal highlights")

    score = round(float(np.clip(score, 0.0, 100.0)), 1)
    passed = score >= 78.0 and not any("NaN" in x for x in issues)
    return QualityReport(
        domain=dom,
        score=score,
        passed=passed,
        metrics=metrics,
        issues=issues,
        recommendations=recs,
    )
