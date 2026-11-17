"""Quantitative Sculpted Blockbench Style Anchor Evaluator (`SKILL.md` / ADR 0011).

Evaluates a compiled Block-DSL `Program` + `Scene` (+ optional RGBA pixel atlas)
against the six core artistic pillars of `.agents/skills/blockbench-sculpted-style/SKILL.md`:

1. `organic_rotated_flow`: Cuboids are deliberately rotated into slopes, tapers,
   and S-curves rather than sitting as rigid axis-aligned box robots.
2. `snout_and_head_sculpture`: Head, brow, jaw, and wedged maxilla/nasal bridge
   taper from skull width toward nose tip (detecting rectangular "doghouse" muzzles).
3. `volume_hierarchy`: Clear Big / Medium / Small (~70 / 25 / 5) volume distribution
   plus readable head-to-torso and limb-to-torso anatomical proportions.
4. `connected_seating_and_scutes`: Grounded rest contact (`Y = 0`), clean `audit_scene`
   sculpted connectivity (`0` swallowed boxes), and hugged ventral scutes (detecting
   protruding ladder-plank underside plates).
5. `non_unisheet_cutout_silhouette`: Thin `plane` silhouette cards with binary `{0, 255}`
   alpha cut-out edges (`edge:`) and, on winged creatures, multi-bone partitioned wing
   panels (`>= 6` articulated wing/cloak/thruster/vane banks across both sides) rather
   than rigid single-sheet wings.
6. `vibrant_cel_palette`: Hue-shifted perceptual OKLCH ramps (warm highlights, saturated
   midtones, cool indigo/violet shadows — never muddy grey), discrete cel steps, and
   controlled focal emissive accents (`0.3% - 10%` of opaque texels).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.dsl.ast_nodes import Program
from synccraft.kinematics.se3_chain import Scene


@dataclass(frozen=True)
class SculptedStyleAnchorReport:
    """Structured quantitative evaluation of Sculpted Blockbench art style."""

    style_anchor_score: float
    passed: bool
    static_audit_passed: bool
    pillar_scores: dict[str, float]
    metrics: dict[str, Any]
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "style_anchor_score": round(float(self.style_anchor_score), 4),
            "passed": bool(self.passed),
            "static_audit_passed": bool(self.static_audit_passed),
            "pillar_scores": {k: round(float(v), 4) for k, v in sorted(self.pillar_scores.items())},
            "metrics": dict(self.metrics),
            "diagnostics": list(self.diagnostics),
        }


def _is_non_axis_aligned(rot: np.ndarray, tol: float = 0.035) -> bool:
    """True when rotation matrix `rot` departs from 90-degree axis permutations."""
    max_entry_per_col = np.max(np.abs(rot), axis=0)
    return bool(np.any(max_entry_per_col < (1.0 - tol)))


def evaluate_sculpted_style_anchor(
    program: Program,
    scene: Scene,
    *,
    atlas_rgba: np.ndarray | None = None,
    require_wings_partitioned: bool | None = None,
) -> SculptedStyleAnchorReport:
    """Score `program` and `scene` against the Sculpted Blockbench Style specification."""
    diagnostics: list[str] = []
    aud = audit_scene(scene)
    static_ok = bool(aud["passed"])
    if not static_ok:
        diagnostics.append(
            f"static_sculpted_audit_failed: {len(aud['collision_pairs'])} collisions, "
            f"{len(aud['contact_violations'])} contact/burial violations"
        )

    cuboids = [b for b in scene.boxes if getattr(b, "plane_normal", None) is None and float(np.min(b.size)) > 0.0]
    planes = [b for b in scene.boxes if getattr(b, "plane_normal", None) is not None or float(np.min(b.size)) <= 0.0]
    n_total = len(scene.boxes)
    n_cuboids = len(cuboids)
    n_planes = len(planes)

    # ------------------------------------------------------------------
    # 1. Organic Rotated Flow (`organic_rotated_flow`)
    # ------------------------------------------------------------------
    rotated_cuboids = sum(1 for b in cuboids if _is_non_axis_aligned(b.rotation))
    rotated_ratio = rotated_cuboids / float(max(1, n_cuboids))
    # Sculpted creatures should have >= 40% rotated cuboids (65%+ ideal)
    flow_score = float(np.clip(rotated_ratio / 0.55, 0.0, 1.0))
    if rotated_ratio < 0.30:
        diagnostics.append(
            f"low_rotated_cuboid_ratio ({rotated_ratio:.1%}): model risks reading as a rigid axis-aligned box robot"
        )

    # ------------------------------------------------------------------
    # 2. Snout & Head Wedge Sculpture (`snout_and_head_sculpture`)
    # ------------------------------------------------------------------
    head_keywords = ("head", "skull", "cranium", "snout", "muzzle", "jaw", "mandible", "beak", "visor", "nose", "brow", "cheek")
    head_boxes = [
        b
        for b in cuboids
        if any(k in b.name.lower() or k in b.bone_name.lower() for k in head_keywords)
        and not any(x in b.name.lower() for x in ("horn", "antler", "vane", "frill", "mane", "ear", "feeler"))
    ]
    snout_taper_ratio = 1.0
    if len(head_boxes) >= 3:
        z_centers = np.array([float(b.center[2]) for b in head_boxes], dtype=np.float64)
        z_min, z_max = float(np.min(z_centers)), float(np.max(z_centers))
        z_span = z_max - z_min
        if z_span > 2.5:
            rear_cut = z_min + 0.42 * z_span
            tip_cut = z_min + 0.68 * z_span
            rear_x = [float(np.ptp(b.corners()[:, 0])) + 2.0 * abs(float(b.center[0])) * 0.5 for b in head_boxes if float(b.center[2]) <= rear_cut]
            tip_boxes = [b for b in head_boxes if float(b.center[2]) >= tip_cut]
            if rear_x and tip_boxes:
                rear_pts = np.vstack([b.corners() for b in head_boxes if float(b.center[2]) <= rear_cut])
                tip_pts = np.vstack([b.corners() for b in tip_boxes])
                rear_width = float(np.ptp(rear_pts[:, 0]))
                tip_width = float(np.ptp(tip_pts[:, 0]))
                snout_taper_ratio = tip_width / max(1e-6, rear_width)
    if len(head_boxes) < 3:
        head_score = 0.55
        diagnostics.append("sparse_head_sculpture: head should spend primary detail budget (skull, wedged snout, brow, jaw)")
    elif snout_taper_ratio > 0.88:
        head_score = 0.45
        diagnostics.append(
            f"snout_doghouse_warning: anterior snout width is {snout_taper_ratio:.2f}x rear skull width; add angled maxilla/nasal taper"
        )
    else:
        taper_quality = float(np.clip((0.88 - snout_taper_ratio) / 0.30, 0.25, 1.0))
        count_quality = float(np.clip(len(head_boxes) / 8.0, 0.3, 1.0))
        head_score = 0.55 * taper_quality + 0.45 * count_quality

    # ------------------------------------------------------------------
    # 3. Volume Hierarchy ~ 70 / 25 / 5 (`volume_hierarchy`)
    # ------------------------------------------------------------------
    vols = np.array([float(np.prod(b.size)) for b in cuboids], dtype=np.float64) if cuboids else np.ones(1)
    vols_sorted = np.sort(vols)[::-1]
    tot_vol = max(1e-6, float(np.sum(vols_sorted)))
    k_big = max(1, int(round(0.16 * len(vols_sorted))))
    k_med = max(k_big + 1, int(round(0.52 * len(vols_sorted))))
    big_share = float(np.sum(vols_sorted[:k_big])) / tot_vol
    med_share = float(np.sum(vols_sorted[k_big:k_med])) / tot_vol
    small_share = float(np.sum(vols_sorted[k_med:])) / tot_vol
    # Ideal roughly 0.55-0.78 big, 0.16-0.36 medium, 0.03-0.16 small
    hier_err = abs(big_share - 0.66) + 0.8 * abs(med_share - 0.25) + 0.8 * abs(small_share - 0.09)
    hierarchy_score = float(np.clip(1.0 - hier_err * 1.15, 0.25, 1.0))
    if big_share < 0.38:
        diagnostics.append(
            f"fragmented_volume_hierarchy (big={big_share:.0%}, med={med_share:.0%}, small={small_share:.0%}): lacks dominant primary masses"
        )

    # ------------------------------------------------------------------
    # 4. Connected Seating, Ground Contact & Contoured Scutes
    # ------------------------------------------------------------------
    all_corners = np.vstack([b.corners() for b in scene.boxes]) if scene.boxes else np.zeros((1, 3))
    min_y = float(np.min(all_corners[:, 1]))
    grounded_err = abs(min_y)
    ground_ok = grounded_err <= 0.35
    if not ground_ok:
        diagnostics.append(f"ungrounded_rest_pose: lowest vertex min_y={min_y:.2f}u (expected ~0.0u)")

    scute_boxes = [b for b in cuboids if any(k in b.name.lower() for k in ("scute", "belly", "ventral", "sternum", "throat_plate"))]
    plank_warnings = 0
    for sb in scute_boxes:
        # A plank that is taller than wide/long on the belly reads like a loose board
        sx, sy, sz = [float(x) for x in sb.size]
        if sy > max(sx, sz) * 1.15:
            plank_warnings += 1
    if plank_warnings > 2:
        diagnostics.append(f"ventral_plank_protrusion: {plank_warnings} underside plates are taller than wide")

    seating_score = (0.55 if static_ok else 0.15) + (0.30 if ground_ok else max(0.0, 0.30 - 0.12 * grounded_err)) + (
        0.15 if plank_warnings <= 2 else 0.04
    )
    seating_score = float(np.clip(seating_score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 5. Cut-Out Plane Silhouette & Non-Uni-Sheet Wing Partitioning
    # ------------------------------------------------------------------
    plane_ratio = n_planes / float(max(1, n_total))
    edges_declared = sum(1 for p in planes if getattr(p, "edge", None) is not None)
    wing_bone_names = {
        b.name
        for b in scene.bones
        if any(k in b.name.lower() for k in ("wing", "patag", "cloak", "thruster", "sail", "vane", "feather_bank", "chainblade"))
    }
    wing_plane_bones = {
        p.bone_name
        for p in planes
        if any(k in p.bone_name.lower() or k in p.name.lower() for k in ("wing", "patag", "cloak", "thruster", "membrane", "bay", "vane", "feather"))
    }
    has_wings = require_wings_partitioned if require_wings_partitioned is not None else (len(wing_bone_names) >= 2)

    if n_planes == 0:
        cutout_score = 0.30
        diagnostics.append("missing_cutout_planes: sculpted style uses alpha-cutout planes for membranes, fur tufts, fins, or fringes")
    else:
        plane_share_q = float(np.clip(plane_ratio / 0.06, 0.35, 1.0)) if plane_ratio <= 0.32 else 0.65
        edge_q = float(np.clip(edges_declared / float(max(1, n_planes)), 0.45, 1.0))
        if has_wings:
            if len(wing_plane_bones) < 4 and len(wing_bone_names) < 6:
                wing_q = 0.30
                diagnostics.append(
                    f"unisheet_wing_warning: winged creature has only {len(wing_plane_bones)} membrane-carrying wing bones; partition across finger/cloak bones"
                )
            else:
                wing_q = float(np.clip(max(len(wing_plane_bones), len(wing_bone_names) // 2) / 6.0, 0.5, 1.0))
            cutout_score = 0.35 * plane_share_q + 0.25 * edge_q + 0.40 * wing_q
        else:
            cutout_score = 0.55 * plane_share_q + 0.45 * edge_q

    # ------------------------------------------------------------------
    # 6. Vibrant Hue-Shifted Cel Palette & Focal Glow
    # ------------------------------------------------------------------
    pals = list(program.palettes)
    hue_shifted_count = sum(1 for p in pals if (abs(float(p.hue_shift[0])) + abs(float(p.hue_shift[1]))) >= 8.0)
    vibrant_mid_count = sum(1 for p in pals if float(p.base_lch[1]) >= 0.065)
    emissive_pals = sum(1 for p in pals if bool(p.emissive))

    hue_q = float(np.clip(hue_shifted_count / float(max(1, len(pals))), 0.25, 1.0))
    vib_q = 1.0 if vibrant_mid_count >= 1 else 0.45
    style_banded_q = 1.0 if (program.style is not None and program.style.shade_mode == "banded") else 0.72

    alpha_cutout_verified = False
    if atlas_rgba is not None and atlas_rgba.ndim == 3 and atlas_rgba.shape[2] == 4:
        alpha_ch = atlas_rgba[..., 3]
        alpha_cutout_verified = bool(np.any(alpha_ch == 0) and np.any(alpha_ch == 255))
        opaque_rgb = atlas_rgba[alpha_ch >= 128, :3].astype(np.float64)
        if len(opaque_rgb) > 0:
            chroma_px = np.max(opaque_rgb, axis=-1) - np.min(opaque_rgb, axis=-1)
            p90_chroma = float(np.percentile(chroma_px, 90))
            if p90_chroma < 28.0:
                vib_q *= 0.65
                diagnostics.append(f"flat_grey_desaturation_warning: 90th-percentile pixel chroma is {p90_chroma:.1f}/255")

    palette_score = float(np.clip(0.40 * hue_q + 0.35 * vib_q + 0.25 * style_banded_q, 0.0, 1.0))

    pillar_scores = {
        "organic_rotated_flow": flow_score,
        "snout_and_head_sculpture": head_score,
        "volume_hierarchy": hierarchy_score,
        "connected_seating_and_scutes": seating_score,
        "non_unisheet_cutout_silhouette": cutout_score,
        "vibrant_cel_palette": palette_score,
    }
    weights = {
        "organic_rotated_flow": 0.16,
        "snout_and_head_sculpture": 0.20,
        "volume_hierarchy": 0.14,
        "connected_seating_and_scutes": 0.20,
        "non_unisheet_cutout_silhouette": 0.16,
        "vibrant_cel_palette": 0.14,
    }
    overall = sum(pillar_scores[k] * weights[k] for k in weights)
    passed = bool(static_ok and ground_ok and overall >= 0.78)

    metrics = {
        "elements_total": n_total,
        "cuboid_boxes": n_cuboids,
        "cutout_planes": n_planes,
        "cuboid_ratio": round(n_cuboids / float(max(1, n_total)), 4),
        "rotated_cuboid_ratio": round(rotated_ratio, 4),
        "head_box_count": len(head_boxes),
        "snout_taper_ratio": round(snout_taper_ratio, 4),
        "volume_shares_big_med_small": [round(big_share, 3), round(med_share, 3), round(small_share, 3)],
        "min_ground_y": round(min_y, 4),
        "wing_bone_count": len(wing_bone_names),
        "wing_plane_bone_count": len(wing_plane_bones),
        "hue_shifted_palette_count": hue_shifted_count,
        "emissive_palette_count": emissive_pals,
        "alpha_cutout_verified": alpha_cutout_verified,
        "sculpted_overlap_pairs": len(aud.get("sculpted_overlap_pairs", [])),
    }

    return SculptedStyleAnchorReport(
        style_anchor_score=overall,
        passed=passed,
        static_audit_passed=static_ok,
        pillar_scores=pillar_scores,
        metrics=metrics,
        diagnostics=tuple(diagnostics),
    )
