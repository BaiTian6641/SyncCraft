"""Pure-Algorithmic AI-Assisted Human-in-the-Loop Co-Pilot (`SyncCraft v49.0 / Milestone v12`).

Provides three commercial studio capabilities with ZERO ML/diffusion dependencies:
  1. `diagnose_asset(graph, mesh_ctx)` — Evaluates texture quality, value span, hue-shift richness,
     3D UV seam discontinuity, and PBR contrast to emit structured findings with safe surgical patches.
  2. `apply_surgical_patch(graph, patch_ops, artist_overrides=...)` — Applies minimal node/param/palette
     patches to a TexGraph while STRICTLY preserving human artist overrides (`artist_overrides`) and
     verifying DAG acyclicity + before/after QA scores.
  3. `extract_style_dna(image_source)` — Pure-algorithmic reference image analyzer extracting a curated
     6-stop hue-shifted color ramp, top-chroma accent stops, contrast telemetry, and a ready-to-paste
     `.texdsl` `@palette` block.
"""
from __future__ import annotations

import base64
import copy
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import TexGraph, _parse_ref
from .mesh_baker import MeshBakeContext
from .pixel_art_ops import hex_to_srgb, rgba_float_to_u8
from .quality import evaluate_texture_quality


def _rgb_to_hex(rgb: tuple[float, float, float] | np.ndarray) -> str:
    r, g, b = (int(np.clip(round(float(c) * 255.0), 0, 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgb_to_hsv_deg(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    s = np.where(mx > 1e-6, df / np.maximum(mx, 1e-6), 0.0)
    h = np.zeros_like(mx)
    m_r = (df > 1e-6) & (mx == r)
    m_g = (df > 1e-6) & (mx == g) & (~m_r)
    m_b = (df > 1e-6) & (~m_r) & (~m_g)
    h[m_r] = (60.0 * ((g[m_r] - b[m_r]) / df[m_r])) % 360.0
    h[m_g] = (60.0 * ((b[m_g] - r[m_g]) / df[m_g]) + 120.0) % 360.0
    h[m_b] = (60.0 * ((r[m_b] - g[m_b]) / df[m_b]) + 240.0) % 360.0
    return h, s, mx


# ---------------------------------------------------------------------------
# 1. Pure-Algorithmic Reference Style-DNA Extractor (`extract_style_dna`)
# ---------------------------------------------------------------------------

BUILT_IN_STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "chronos_amethyst": {
        "name": "chronos_amethyst",
        "title": "Chrono-Amethyst (D&D Time × Amethyst × Gemini 4)",
        "stops": ["#18152C", "#2D254F", "#4F3B82", "#7C5CB8", "#B28DE3", "#E2CEFF"],
        "accent_stops": ["#0E4B68", "#18A8C8", "#4FF8FF", "#E879F9"],
        "metal_stops": ["#462614", "#8B5320", "#C88E34", "#EAC25E", "#FFF0AE"],
        "mood": "Celestial Chronal Gem & Brass",
    },
    "solar_aeon": {
        "name": "solar_aeon",
        "title": "Solar Aeon (Radiant Amber-Gold & Obsidian)",
        "stops": ["#1C141F", "#3B222A", "#703A2E", "#B26234", "#E69B45", "#FCE08B"],
        "accent_stops": ["#7C2D12", "#EA580C", "#FACC15", "#FFFBEB"],
        "metal_stops": ["#3F2712", "#7D4E1E", "#C48A30", "#F0C860", "#FFF4C2"],
        "mood": "Sun-Forged Ancient Timekeeper",
    },
    "void_eclipse": {
        "name": "void_eclipse",
        "title": "Void Eclipse (Abyssal Indigo & Crimson-Magenta)",
        "stops": ["#0C0B18", "#1A1730", "#322752", "#563B7E", "#8A54B2", "#C989EB"],
        "accent_stops": ["#5B0E36", "#B81D5A", "#F43F8E", "#FFD6EC"],
        "metal_stops": ["#1C1D29", "#34374D", "#585E7A", "#8C95B5", "#CFD7EB"],
        "mood": "Planar Rift Singularity",
    },
    "frost_singularity": {
        "name": "frost_singularity",
        "title": "Frost Singularity (Glacial Sapphire & Silver)",
        "stops": ["#111A2E", "#1E3254", "#325482", "#5482B5", "#8BBBE0", "#D4EEFF"],
        "accent_stops": ["#0B3E52", "#1490B8", "#38F2FF", "#E0FCFF"],
        "metal_stops": ["#242A38", "#48546B", "#7B8BA6", "#B8C7DE", "#EEF5FF"],
        "mood": "Frozen Timeline Crystalline",
    },
}


def extract_style_dna(
    image_source: str | Path | np.ndarray | Image.Image,
    *,
    num_stops: int = 6,
    name: str = "reference_dna",
) -> dict[str, Any]:
    """Extract a curated OKLCH-inspired hue-shifted color ramp & texture style telemetry from reference art.

    Accepts a built-in preset key (`"preset:chronos_amethyst"`), local file path, base64 data URL,
    PIL Image, or NumPy RGB/RGBA array. Pure NumPy + Pillow (zero ML models).
    """
    if isinstance(image_source, str) and image_source.startswith("preset:"):
        key = image_source.split(":", 1)[1]
        if key not in BUILT_IN_STYLE_PRESETS:
            raise ValueError(f"Unknown style preset: {key!r}")
        p = BUILT_IN_STYLE_PRESETS[key]
        dsl = (
            f"@palette {{\n"
            f"  {p['name']}_primary: {p['stops']}\n"
            f"  {p['name']}_accent:  {p['accent_stops']}\n"
            f"  {p['name']}_metal:   {p['metal_stops']}\n"
            f"}}\n"
        ).replace("'", '"')
        return {
            "name": p["name"],
            "title": p["title"],
            "stops": list(p["stops"]),
            "accent_stops": list(p["accent_stops"]),
            "metal_stops": list(p["metal_stops"]),
            "texdsl_palette_block": dsl,
            "stats": {
                "value_span": 195.0,
                "hue_shift_deg": 44.0,
                "mean_saturation": 0.48,
                "recommended_smooth_blend": 0.56,
            },
        }

    # Decode image source -> float32 (H, W, 4) in [0, 1]
    if isinstance(image_source, np.ndarray):
        arr = rgba_float_to_u8(image_source) if image_source.dtype != np.uint8 else image_source
        im = Image.fromarray(arr).convert("RGBA")
    elif isinstance(image_source, Image.Image):
        im = image_source.convert("RGBA")
    elif isinstance(image_source, str) and image_source.startswith("data:image/"):
        b64 = image_source.split(",", 1)[1]
        im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    else:
        im = Image.open(Path(image_source)).convert("RGBA")

    # Downsample large photos cleanly for fast statistical extraction
    if max(im.size) > 256:
        im.thumbnail((256, 256), Image.Resampling.BILINEAR)

    rgba = np.asarray(im, dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    fg = alpha > 0.25
    if not np.any(fg):
        fg = np.ones(alpha.shape, dtype=bool)

    pix = rgb[fg]
    lum = 0.2126 * pix[:, 0] + 0.7152 * pix[:, 1] + 0.0722 * pix[:, 2]
    h_deg, sat, val = _rgb_to_hsv_deg(pix)

    n = max(3, min(8, int(num_stops)))
    quantiles = np.linspace(0.06, 0.94, n)
    edges = np.quantile(lum, np.linspace(0.02, 0.98, n + 1))

    stops: list[str] = []
    stop_hues: list[float] = []
    for i in range(n):
        lo, hi = float(edges[i]), float(edges[i + 1]) + 1e-5
        band = (lum >= lo) & (lum <= hi)
        if not np.any(band):
            q_val = float(np.quantile(lum, quantiles[i]))
            idx = int(np.argmin(np.abs(lum - q_val)))
            col = pix[idx].copy()
        else:
            # Weight slightly toward richer-chroma pixels so extracted ramps stay vibrant (never muddy gray!)
            w = 0.35 + sat[band] * 1.4
            col = np.sum(pix[band] * w[:, None], axis=0) / np.maximum(1e-6, np.sum(w))
        # Subtle shadow cool-shift / highlight warm-shift safeguard if reference is overly flat
        t = i / max(1.0, float(n - 1))
        col[2] = float(np.clip(col[2] + (0.03 * (1.0 - t)), 0.0, 1.0))
        col[0] = float(np.clip(col[0] + (0.02 * t), 0.0, 1.0))
        stops.append(_rgb_to_hex(col))
        sh, _, _ = _rgb_to_hsv_deg(col[None, :])
        stop_hues.append(float(sh[0]))

    # Extract 4 high-chroma accent / emissive stops
    hi_sat_thresh = float(np.quantile(sat, 0.80))
    acc_mask = (sat >= max(0.28, hi_sat_thresh)) & (lum >= 0.18)
    if np.sum(acc_mask) >= 12:
        acc_pix = pix[acc_mask]
        acc_lum = lum[acc_mask]
        a_edges = np.quantile(acc_lum, np.linspace(0.08, 0.96, 5))
        accent_stops = []
        for i in range(4):
            bm = (acc_lum >= a_edges[i]) & (acc_lum <= a_edges[i + 1] + 1e-5)
            c = np.mean(acc_pix[bm], axis=0) if np.any(bm) else acc_pix[0]
            accent_stops.append(_rgb_to_hex(c))
    else:
        accent_stops = stops[-4:]

    # Local pixel step telemetry for cluster guidance
    lum2d = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) * 255.0
    dx = np.abs(lum2d[:, 1:] - lum2d[:, :-1])
    dy = np.abs(lum2d[1:, :] - lum2d[:-1, :])
    mean_step = float(0.5 * (dx.mean() + dy.mean()))
    value_span = float((np.quantile(lum, 0.95) - np.quantile(lum, 0.05)) * 255.0)
    dh = abs(stop_hues[-1] - stop_hues[0])
    hue_shift = float(min(dh, 360.0 - dh))
    smooth_rec = float(np.clip(0.68 - (mean_step / 45.0) * 0.28, 0.35, 0.72))

    safe_name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip().lower()) or "reference_dna"
    dsl = (
        f"@palette {{\n"
        f"  {safe_name}_primary: {stops}\n"
        f"  {safe_name}_accent:  {accent_stops}\n"
        f"}}\n"
    ).replace("'", '"')

    return {
        "name": safe_name,
        "title": f"Extracted Style DNA ({safe_name})",
        "stops": stops,
        "accent_stops": accent_stops,
        "texdsl_palette_block": dsl,
        "stats": {
            "value_span": round(value_span, 2),
            "hue_shift_deg": round(hue_shift, 1),
            "mean_saturation": round(float(sat.mean()), 3),
            "mean_lum_step": round(mean_step, 2),
            "recommended_smooth_blend": round(smooth_rec, 2),
        },
    }


# ---------------------------------------------------------------------------
# 2. Structured AI Asset Critic & Surgical Non-Destructive Patcher
# ---------------------------------------------------------------------------


def diagnose_asset(
    graph: TexGraph,
    mesh_ctx: MeshBakeContext | None = None,
    *,
    phase: float = 0.0,
) -> dict[str, Any]:
    """Inspect a live TexGraph + optional 3D mesh and emit actionable findings with surgical patch ops."""
    mc = mesh_ctx or graph.mesh_context
    val = graph.validate_graph()
    ev = graph.evaluate(phase=phase, capture_intermediates=True, build_pbr=True, allow_orphan_errors=True)
    dom = "entity" if mc is not None else (getattr(graph, "domain", None) or "block")
    qa = evaluate_texture_quality(
        ev.output,
        domain=dom,
        tileable=graph.tileable,
        mesh_ctx=mc,
        pbr_channels=ev.pbr_channels,
        pbr_maps=ev.pbr_maps,
    )

    findings: list[dict[str, Any]] = []
    out_id = (graph.output_material_node or graph.output_node or (graph.node_order[-1] if graph.node_order else "")).split(".")[0]
    grade_id = next((nid for nid in reversed(graph.node_order) if graph.nodes[nid].op == "material_grade"), None)
    shadow_id = next((nid for nid in reversed(graph.node_order) if graph.nodes[nid].op == "precompute_shadow_pbr"), None)
    seam_id = next((nid for nid in reversed(graph.node_order) if graph.nodes[nid].op == "uv_seam_heal"), None)

    metrics = qa.metrics
    v_span = float(metrics.get("value_span", 110.0))
    m_step = float(metrics.get("mean_lum_step", 7.0))
    hue_deg = float(metrics.get("hue_shift_deg", 45.0))
    calm_pct = float(metrics.get("calm_interior_pct", 75.0))

    # Finding 1: Low value contrast / flat look
    if v_span < 72.0:
        target = grade_id or out_id
        cur_c = float(graph.nodes[target].params.get("contrast", 1.0)) if target in graph.nodes else 1.0
        findings.append(
            {
                "id": "low_value_span",
                "severity": "warn",
                "title": f"Low Luminance Contrast (span={v_span:.1f}, target >= 80)",
                "detail": "Surface values look slightly compressed; boosting contrast and shadow cool-shift deepens 3D readability.",
                "patch_ops": [
                    {"op": "set_param", "node_id": target, "param": "contrast", "value": round(min(1.24, cur_c + 0.08), 2)},
                    {"op": "set_param", "node_id": target, "param": "vibrance", "value": 0.20},
                ]
                if grade_id
                else [
                    {
                        "op": "append_node",
                        "node_op": "material_grade",
                        "id": "ai_contrast_grade",
                        "params": {"contrast": 1.12, "vibrance": 0.18, "shadow_cool": 0.06},
                    }
                ],
            }
        )

    # Finding 2: Noisy / overly busy pixels
    if m_step > 12.5 or calm_pct < 58.0:
        cob_id = next((nid for nid in graph.node_order if "smooth_blend" in graph.nodes[nid].params), None)
        patch = (
            [{"op": "set_param", "node_id": cob_id, "param": "smooth_blend", "value": 0.68}]
            if cob_id
            else [
                {
                    "op": "append_node",
                    "node_op": "material_grade",
                    "id": "ai_calm_grade",
                    "params": {"vibrance": 0.14, "contrast": 1.02},
                }
            ]
        )
        findings.append(
            {
                "id": "busy_pixel_noise",
                "severity": "warn",
                "title": f"High Pixel Contrast Noise (step={m_step:.1f}, calm={calm_pct:.0f}%)",
                "detail": "Pixel clusters are fragmented; raising smooth_blend or consolidating pixel clusters restores studio clarity.",
                "patch_ops": patch,
            }
        )

    # Finding 3: 3D Mesh attached without Precomputed Ray-Traced Shadow Special PBR
    if mc is not None and shadow_id is None and graph.output_material_node is not None:
        findings.append(
            {
                "id": "missing_shadow_pbr",
                "severity": "info",
                "title": "3D Mesh Attached Without Ray-Traced Shadow Special PBR",
                "detail": "Appending `precompute_shadow_pbr` injects real 3D overhang shadows, sky-hemisphere bent normals & thickness SSS.",
                "patch_ops": [
                    {
                        "op": "append_node",
                        "node_op": "precompute_shadow_pbr",
                        "id": "ai_shadow_pbr",
                        "params": {
                            "shadow_strength": 0.56,
                            "bent_normal_strength": 0.44,
                            "thickness_sss_strength": 0.38,
                            "bake_into_albedo": 0.38,
                        },
                    }
                ],
            }
        )

    # Finding 4: 3D Mesh with cross-island seams without UV seam stitcher
    if mc is not None and len(mc.uv_seam_src_yx) > 200 and seam_id is None and graph.output_material_node is not None:
        findings.append(
            {
                "id": "missing_uv_seam_heal",
                "severity": "info",
                "title": f"{len(mc.uv_seam_src_yx)} Solved 3D UV Seam Pairs Available for Stitching",
                "detail": "Adding `uv_seam_heal` smoothly blends albedo, height & tangent normals across connected 3D cube corners.",
                "patch_ops": [
                    {
                        "op": "append_node",
                        "node_op": "uv_seam_heal",
                        "id": "ai_seam_heal",
                        "params": {"strength": 0.78, "radius_px": 2, "heal_normals": True},
                    }
                ],
            }
        )

    # Finding 5: Flat hue shift (< 22 deg) -> suggest cool-shadow / warm-light studio grade
    if hue_deg < 22.0 and grade_id is not None:
        findings.append(
            {
                "id": "narrow_hue_shift",
                "severity": "info",
                "title": f"Narrow Hue Swing ({hue_deg:.1f}°, studio target 30°–50°)",
                "detail": "Nudging shadow cool-shift and vibrance enriches cool-shadow-to-warm-highlight depth.",
                "patch_ops": [
                    {"op": "set_param", "node_id": grade_id, "param": "shadow_cool", "value": 0.08},
                    {"op": "set_param", "node_id": grade_id, "param": "vibrance", "value": 0.22},
                ],
            }
        )

    # If everything already passes cleanly, offer a constructive polish micro-tune
    if not findings:
        findings.append(
            {
                "id": "studio_certified",
                "severity": "ok",
                "title": f"Studio Certified ({round(qa.score)}/100 QA · {len(graph.nodes)} Nodes)",
                "detail": "Value span, hue shift, calm pixel clusters, and PBR channels all meet commercial quality standards.",
                "patch_ops": (
                    [{"op": "set_param", "node_id": grade_id, "param": "vibrance", "value": round(min(0.32, float(graph.nodes[grade_id].params.get("vibrance", 0.16)) + 0.04), 2)}]
                    if grade_id
                    else []
                ),
            }
        )

    return {
        "ok": True,
        "domain": dom,
        "qa": qa.to_dict(),
        "validation": val.to_dict(),
        "finding_count": len(findings),
        "findings": findings,
    }


def apply_surgical_patch(
    graph: TexGraph,
    patch_ops: list[dict[str, Any]],
    *,
    artist_overrides: dict[str, Any] | None = None,
    preserve_artist_overrides: bool = True,
) -> dict[str, Any]:
    """Apply minimal surgical patches to `graph` while strictly honoring human `artist_overrides`.

    `artist_overrides` keys follow `"node_id.param_key"` (or exposed param public names!).
    Returns before/after QA scores, applied operations, and any skipped operations protected by artist locks.
    """
    mc = graph.mesh_context
    dom = "entity" if mc is not None else (getattr(graph, "domain", None) or "block")

    ev_before = graph.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
    qa_before = evaluate_texture_quality(
        ev_before.output, domain=dom, tileable=graph.tileable, mesh_ctx=mc, pbr_channels=ev_before.pbr_channels, pbr_maps=ev_before.pbr_maps
    )

    # Build canonical set of artist-protected `(node_id, param_key)` pairs
    protected: set[tuple[str, str]] = set()
    overrides_map = dict(artist_overrides or {})
    if preserve_artist_overrides:
        for k in overrides_map:
            if "." in k:
                nid_k, p_k = k.split(".", 1)
                protected.add((nid_k, p_k))
            elif k in graph.exposed_params:
                info = graph.exposed_params[k]
                protected.add((str(info["node_id"]), str(info["param_key"])))

    applied: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []

    for op_spec in patch_ops:
        kind = str(op_spec.get("op", ""))
        if kind == "set_param":
            nid = str(op_spec["node_id"])
            pkey = str(op_spec["param"])
            if nid not in graph.nodes:
                continue
            if (nid, pkey) in protected:
                preserved.append({"node_id": nid, "param": pkey, "reason": "Protected by human artist override"})
                continue
            graph.nodes[nid].params[pkey] = copy.deepcopy(op_spec["value"])
            applied.append(op_spec)

        elif kind == "replace_palette":
            stops = list(op_spec.get("stops") or [])
            target_nodes = op_spec.get("node_ids") or list(graph.node_order)
            for nid in target_nodes:
                if nid not in graph.nodes:
                    continue
                sp = graph.nodes[nid]
                for pk in list(sp.params.keys()):
                    if pk.endswith("stops") or pk in ("stops", "dorsal_stops", "stone_stops", "coat_stops", "host_stops", "bark_stops"):
                        if (nid, pk) in protected:
                            preserved.append({"node_id": nid, "param": pk, "reason": "Protected by human artist override"})
                            continue
                        if isinstance(sp.params[pk], list) and sp.params[pk] and isinstance(sp.params[pk][0], str) and sp.params[pk][0].startswith("#"):
                            sp.params[pk] = list(stops)
                            applied.append({"op": "replace_palette", "node_id": nid, "param": pk})
                            break

        elif kind == "append_node":
            node_op = str(op_spec["node_op"])
            base_id = str(op_spec.get("id") or f"ai_{node_op}")
            nid = base_id
            idx = 1
            while nid in graph.nodes:
                nid = f"{base_id}_{idx}"
                idx += 1
            prev_out = graph.output_material_node or (graph.output_node.split(".")[0] if graph.output_node else graph.node_order[-1])
            in_key = "src" if node_op in ("triplanar_project", "uv_seam_heal") else ("base" if node_op == "block_edge_transition" else "mat")
            params = dict(op_spec.get("params") or {})
            graph.add(node_op, id=nid, inputs={in_key: prev_out}, **params)
            if graph.output_material_node is not None:
                graph.set_output_material(nid, normal_strength=float(graph.pbr_config.get("normal_strength", 2.4)))
            else:
                graph.set_output(nid)
            applied.append({"op": "append_node", "id": nid, "node_op": node_op, "after": prev_out})

        elif kind == "toggle_bypass":
            nid = str(op_spec["node_id"])
            if nid in graph.nodes:
                graph.nodes[nid].bypassed = bool(op_spec.get("bypassed", not graph.nodes[nid].bypassed))
                applied.append(op_spec)

    graph._static_cache.clear()
    graph.auto_layout(col_spacing=290.0, row_spacing=195.0)
    val = graph.validate_graph()
    ev_after = graph.evaluate(phase=0.0, build_pbr=True, allow_orphan_errors=True)
    qa_after = evaluate_texture_quality(
        ev_after.output, domain=dom, tileable=graph.tileable, mesh_ctx=mc, pbr_channels=ev_after.pbr_channels, pbr_maps=ev_after.pbr_maps
    )

    return {
        "ok": val.ok,
        "applied_count": len(applied),
        "applied": applied,
        "preserved_overrides": preserved,
        "qa_before": round(float(qa_before.score), 1),
        "qa_after": round(float(qa_after.score), 1),
        "qa_delta": round(float(qa_after.score - qa_before.score), 1),
        "validation": val.to_dict(),
    }
