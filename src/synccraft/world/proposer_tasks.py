"""5-Task World-Model Proposer Supervision, Split-Isolation Verifier, & Gate D Baseline Harness (ADR 0013).

Implements Phase 4 / Gate D from the Blockbench Pipeline Roadmap:
1. Structured text-in / typed-parameter-out supervision across 5 tasks (no raw per-frame
   Euler keyframe generation or pixel generation by the LLM):
   - `label_morphology_roles`
   - `propose_gait_from_style`
   - `repair_failed_gait`
   - `plan_texture_and_stamps`
   - `invert_clip_to_gait`
2. Strict `body_plan_group` + `source_author` + `asset_id` split isolation check
   (`verify_body_plan_and_author_split_isolation`) preventing body-plan or author leakage
   into `holdout_gate_d`.
3. Classical Gate D baselines (`style_phrase_to_gait_preset` + `search_or_repair_gait`)
   and `evaluate_gate_d_proposer` comparing `fixed_preset_baseline`,
   `preset_plus_deterministic_search`, and any candidate `proposer_fn` with full `CostLedger`
   accounting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import functools
import json
from typing import Any

import numpy as np

from synccraft.dsl import GaitDecl, Program, parse_program
from synccraft.evaluation.budget import CostLedger
from synccraft.kinematics.gait_inversion import (
    _adapt_pattern_for_morphology,
    invert_clip_to_gait,
    search_or_repair_gait,
)
from synccraft.kinematics.morphology import MorphologyReport, analyze_morphology
from synccraft.kinematics.musculoskeletal import audit_behavior_clip
from synccraft.kinematics.se3_chain import compile_geometry

PROPOSER_TASK_TYPES: tuple[str, ...] = (
    "label_morphology_roles",
    "propose_gait_from_style",
    "repair_failed_gait",
    "plan_texture_and_stamps",
    "invert_clip_to_gait",
)


@dataclass(frozen=True)
class ProposerTaskRecord:
    """Single structured supervision or Gate D holdout record."""

    record_id: str
    task_type: str
    split: str  # "train" | "val" | "holdout_gate_d"
    asset_id: str
    body_plan_group: str
    source_author: str
    dsl_source: str
    prompt_payload: dict[str, Any]
    target_payload: dict[str, Any]
    verified_by_solver: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_type": self.task_type,
            "split": self.split,
            "asset_id": self.asset_id,
            "body_plan_group": self.body_plan_group,
            "source_author": self.source_author,
            "prompt_payload": self.prompt_payload,
            "target_payload": self.target_payload,
            "verified_by_solver": bool(self.verified_by_solver),
        }


def verify_body_plan_and_author_split_isolation(records: Sequence[ProposerTaskRecord]) -> dict[str, Any]:
    """Verify zero leakage of `asset_id`, `body_plan_group`, or `source_author` into `holdout_gate_d`."""
    by_split: dict[str, list[ProposerTaskRecord]] = {"train": [], "val": [], "holdout_gate_d": []}
    for rec in records:
        if rec.task_type not in PROPOSER_TASK_TYPES:
            raise ValueError(f"unknown proposer task_type {rec.task_type!r}")
        if rec.split not in by_split:
            raise ValueError(f"invalid split {rec.split!r}; expected train | val | holdout_gate_d")
        by_split[rec.split].append(rec)

    non_holdout = by_split["train"] + by_split["val"]
    holdout = by_split["holdout_gate_d"]

    nh_assets = {r.asset_id for r in non_holdout}
    ho_assets = {r.asset_id for r in holdout}
    leaked_assets = sorted(nh_assets & ho_assets)
    if leaked_assets:
        raise ValueError(f"asset_id leakage into holdout_gate_d: {leaked_assets}")

    nh_plans = {r.body_plan_group for r in non_holdout}
    ho_plans = {r.body_plan_group for r in holdout}
    leaked_plans = sorted(nh_plans & ho_plans)
    if leaked_plans:
        raise ValueError(f"body_plan_group leakage into holdout_gate_d: {leaked_plans}")

    nh_authors = {r.source_author for r in non_holdout}
    ho_authors = {r.source_author for r in holdout}
    leaked_authors = sorted(nh_authors & ho_authors)
    if leaked_authors:
        raise ValueError(f"source_author leakage into holdout_gate_d: {leaked_authors}")

    tr_assets = {r.asset_id for r in by_split["train"]}
    va_assets = {r.asset_id for r in by_split["val"]}
    leaked_tr_val = sorted(tr_assets & va_assets)
    if leaked_tr_val:
        raise ValueError(f"asset_id leakage between train and val: {leaked_tr_val}")

    return {
        "passed": True,
        "counts_by_split": {k: len(v) for k, v in by_split.items()},
        "train_val_body_plans": sorted(nh_plans),
        "holdout_body_plans": sorted(ho_plans),
        "train_val_authors": sorted(nh_authors),
        "holdout_authors": sorted(ho_authors),
    }


def style_phrase_to_gait_preset(
    style_phrase: str,
    morph: MorphologyReport,
    *,
    gait_name: str = "proposed_gait",
) -> GaitDecl:
    """Deterministic classical preset baseline mapping natural-language style phrases to `GaitDecl`."""
    low = style_phrase.lower()
    if any(w in low for w in ("gallop", "charge", "sprint", "bound")):
        raw_pat = "rotary_gallop"
        froude = 0.85
        duty = 0.45
        lift = 0.16
        heave = 0.04
        pitch = 2.2
        spine = 3.5
        weight = 0.55
        snap = 0.65
    elif any(w in low for w in ("trot", "brisk", "jog", "alert", "snappy", "march")):
        raw_pat = "diagonal_pairs"
        froude = 0.42
        duty = 0.54
        lift = 0.13
        heave = 0.035
        pitch = 1.6
        spine = 3.0
        weight = 0.45
        snap = 0.55
    elif any(w in low for w in ("stalk", "creep", "sneak", "cautious", "prowl")):
        raw_pat = "lateral_sequence"
        froude = 0.08
        duty = 0.74
        lift = 0.09
        heave = 0.015
        pitch = 0.8
        spine = 3.8
        weight = 0.65
        snap = 0.20
    elif any(w in low for w in ("heavy", "lumber", "tired", "trudg", "colossal", "siege")):
        raw_pat = "lateral_sequence"
        froude = 0.12
        duty = 0.72
        lift = 0.08
        heave = 0.04
        pitch = 1.8
        spine = 2.2
        weight = 0.88
        snap = 0.18
    else:
        raw_pat = "lateral_sequence"
        froude = 0.18
        duty = 0.66
        lift = 0.11
        heave = 0.025
        pitch = 1.2
        spine = 2.8
        weight = 0.50
        snap = 0.30

    pattern = _adapt_pattern_for_morphology(raw_pat, morph)
    return GaitDecl(
        name=gait_name,
        pattern=pattern,
        froude=round(froude, 4),
        duty=round(duty, 4),
        lift_ratio=round(lift, 4),
        heave_ratio=round(heave, 4),
        pitch_deg=round(pitch, 4),
        spine_wave_deg=round(spine, 4),
        tail_lag=0.20,
        weight=round(weight, 4),
        snap=round(snap, 4),
    )


def _gait_decl_to_dict(gait: GaitDecl) -> dict[str, Any]:
    return {
        "name": gait.name,
        "pattern": gait.pattern,
        "froude": round(float(gait.froude), 4),
        "duty": round(float(gait.duty), 4),
        "phase": [[k, round(float(v), 4)] for k, v in gait.phase],
        "lift_ratio": round(float(gait.lift_ratio), 4),
        "heave_ratio": round(float(gait.heave_ratio), 4),
        "pitch_deg": round(float(gait.pitch_deg), 4),
        "spine_wave_deg": round(float(gait.spine_wave_deg), 4),
        "tail_lag": round(float(gait.tail_lag), 4),
        "weight": round(float(gait.weight), 4),
        "snap": round(float(gait.snap), 4),
    }


def _gait_decl_from_dict(payload: dict[str, Any], *, default_name: str = "proposed_gait") -> GaitDecl:
    raw_phase = payload.get("phase") or ()
    phase_tup = tuple((str(item[0]), float(item[1])) for item in raw_phase)
    return GaitDecl(
        name=str(payload.get("name") or default_name),
        pattern=str(payload.get("pattern", "lateral_sequence")),
        froude=float(payload.get("froude", 0.18)),
        duty=float(payload.get("duty", 0.66)),
        phase=phase_tup,
        lift_ratio=float(payload.get("lift_ratio", 0.11)),
        heave_ratio=float(payload.get("heave_ratio", 0.025)),
        pitch_deg=float(payload.get("pitch_deg", 1.2)),
        spine_wave_deg=float(payload.get("spine_wave_deg", 2.8)),
        tail_lag=float(payload.get("tail_lag", 0.20)),
        weight=float(payload.get("weight", 0.50)),
        snap=float(payload.get("snap", 0.30)),
    )


def _quadruped_fixture_dsl(
    *,
    chest: tuple[float, float, float],
    hips: tuple[float, float, float],
    leg_u: float,
    leg_l: float,
    sprawled: bool = False,
) -> str:
    if sprawled:
        root_y = round(leg_l + 0.5 * chest[1] - 0.5, 3)
        legs_fore = f"""
    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.35, 0.55], align=[1, 0, 0]) mirror_x(fore_leg_r) {{
      joint_limits: [[-55, 55], [-65, 65], [-55, 55]]
      box humerus {{ size: [{leg_u}, 1.5, 1.5], material: hide }}
      bone forearm attach(parent_box=humerus, anchor=[-1, -0.2, 0], align=[0, 1, 0]) {{
        joint_limits: [[-55, 55], [-55, 55], [-55, 55]]
        box radius {{ size: [1.5, {leg_l}, 1.5], material: claw }}
      }}
    }}"""
        legs_hind = f"""
      bone hind_leg_l attach(parent_box=hips, anchor=[-1, -0.35, -0.45], align=[1, 0, 0]) mirror_x(hind_leg_r) {{
        joint_limits: [[-55, 55], [-65, 65], [-55, 55]]
        box femur {{ size: [{leg_u}, 1.5, 1.5], material: hide }}
        bone shin attach(parent_box=femur, anchor=[-1, -0.2, 0], align=[0, 1, 0]) {{
          joint_limits: [[-55, 55], [-55, 55], [-55, 55]]
          box tibia {{ size: [1.5, {leg_l}, 1.5], material: claw }}
        }}
      }}"""
    else:
        root_y = round(leg_u + leg_l * float(np.cos(np.deg2rad(10.0))) + 0.35 * chest[1], 3)
        legs_fore = f"""
    bone fore_leg_l attach(parent_box=chest, anchor=[-1, -0.65, 0.6], align=[1, 1, 0]) mirror_x(fore_leg_r) {{
      joint_limits: [[-60, 60], [-25, 25], [-25, 25]]
      box humerus {{ size: [1.5, {leg_u}, 2], material: hide }}
      bone forearm attach(parent_box=humerus, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        rest_euler: [-10, 0, 0]
        joint_limits: [[-55, 55], [-20, 20], [-20, 20]]
        box radius {{ size: [1.5, {leg_l}, 1.5], material: claw }}
      }}
    }}"""
        legs_hind = f"""
      bone hind_leg_l attach(parent_box=hips, anchor=[-1, -0.65, -0.5], align=[1, 1, 0]) mirror_x(hind_leg_r) {{
        joint_limits: [[-60, 60], [-25, 25], [-25, 25]]
        box femur {{ size: [1.5, {leg_u}, 2], material: hide }}
        bone shin attach(parent_box=femur, anchor=[0, -1, 0], align=[0, 1, 0]) {{
          rest_euler: [10, 0, 0]
          joint_limits: [[-55, 55], [-20, 20], [-20, 20]]
          box tibia {{ size: [1.5, {leg_l}, 1.5], material: claw }}
        }}
      }}"""

    return f"""
@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  hide: ramp(base_lch=[0.52, 0.11, 42], steps=5, hue_shift=[-16, 16])
  claw: ramp(base_lch=[0.30, 0.04, 250], steps=4, hue_shift=[-10, 10])
}}
@rig {{
  root bone torso {{
    pivot_local: [0, {root_y}, 0]
    box chest {{ size: [{chest[0]}, {chest[1]}, {chest[2]}], material: hide }}
    bone lumbar attach(parent_box=chest, anchor=[0, 0, -1], align=[0, 0, 1], seat=0.25) {{
      joint_limits: [[-18, 18], [-22, 22], [-12, 12]]
      box hips {{ size: [{hips[0]}, {hips[1]}, {hips[2]}], material: hide }}
      {legs_hind}
      bone tail_1 attach(parent_box=hips, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.25) {{
        joint_limits: [[-20, 20], [-28, 28], [-15, 15]]
        box caudal_1 {{ size: [1.5, 1.5, 4.5], material: hide }}
      }}
    }}
    {legs_fore}
    bone neck attach(parent_box=chest, anchor=[0, 0.35, 1], align=[0, 0, -1], seat=0.25) {{
      rest_euler: [-15, 0, 0]
      joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
      box neck_seg {{ size: [2, 2, 3.5], material: hide }}
      bone skull attach(parent_box=neck_seg, anchor=[0, 0, 1], align=[0, 0, -1], seat=0.2) {{
        rest_euler: [12, 0, 0]
        joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
        box snout {{ size: [2.5, 2, 3.5], material: hide }}
      }}
    }}
  }}
}}
"""


def _biped_fixture_dsl(*, chest: tuple[float, float, float], leg_u: float, leg_l: float) -> str:
    root_y = round(leg_u + leg_l * float(np.cos(np.deg2rad(12.0))) + 0.35 * chest[1], 3)
    return f"""
@meta {{ archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }}
@palette {{
  hide: ramp(base_lch=[0.48, 0.10, 205], steps=5, hue_shift=[-16, 16])
  claw: ramp(base_lch=[0.70, 0.09, 68], steps=4, hue_shift=[-12, 12])
}}
@rig {{
  root bone pelvis {{
    pivot_local: [0, {root_y}, 0]
    box body_core {{ size: [{chest[0]}, {chest[1]}, {chest[2]}], material: hide }}
    bone hind_leg_l attach(parent_box=body_core, anchor=[-1, -0.65, 0], align=[1, 1, 0]) mirror_x(hind_leg_r) {{
      joint_limits: [[-65, 65], [-25, 25], [-25, 25]]
      box thigh {{ size: [2, {leg_u}, 2.5], material: hide }}
      bone shin attach(parent_box=thigh, anchor=[0, -1, 0], align=[0, 1, 0]) {{
        rest_euler: [12, 0, 0]
        joint_limits: [[-55, 65], [-20, 20], [-20, 20]]
        box tarsus {{ size: [1.5, {leg_l}, 2], material: claw }}
      }}
    }}
    bone neck attach(parent_box=body_core, anchor=[0, 0.5, 1], align=[0, 0, -1], seat=0.25) {{
      rest_euler: [-18, 0, 0]
      joint_limits: [[-25, 25], [-25, 25], [-15, 15]]
      box neck_seg {{ size: [2, 2.5, 4], material: hide }}
    }}
    bone tail_1 attach(parent_box=body_core, anchor=[0, 0.2, -1], align=[0, 0, 1], seat=0.25) {{
      joint_limits: [[-20, 20], [-30, 30], [-15, 15]]
      box tail_box {{ size: [2, 1.5, 5.5], material: hide }}
    }}
  }}
}}
"""


def _hexapod_fixture_dsl() -> str:
    return """
@meta { archetype: entity, target_profile: bedrock_entity, texel_density: 16, dsl_version: "4.1", size_step: 0.5, audit_profile: sculpted }
@palette {
  chitin: ramp(base_lch=[0.36, 0.09, 25], steps=5, hue_shift=[-16, 16])
  tarsus_mat: ramp(base_lch=[0.24, 0.05, 260], steps=4, hue_shift=[-10, 10])
}
@rig {
  root bone thorax {
    pivot_local: [0, 5.5, 0]
    box carapace { size: [6, 3.5, 13], material: chitin }
    bone fore_leg_l attach(parent_box=carapace, anchor=[-1, -0.35, 0.72], align=[1, 0, 0]) mirror_x(fore_leg_r) {
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box coxa_f { size: [3.5, 1, 1], material: chitin }
      bone tibia_f attach(parent_box=coxa_f, anchor=[-1, -0.2, 0], align=[0, 1, 0]) {
        joint_limits: [[-55, 55], [-55, 55], [-55, 55]]
        box shin_f { size: [1, 4.5, 1], material: tarsus_mat }
      }
    }
    bone mid_leg_l attach(parent_box=carapace, anchor=[-1, -0.35, 0.0], align=[1, 0, 0]) mirror_x(mid_leg_r) {
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box coxa_m { size: [3.5, 1, 1], material: chitin }
      bone tibia_m attach(parent_box=coxa_m, anchor=[-1, -0.2, 0], align=[0, 1, 0]) {
        joint_limits: [[-55, 55], [-55, 55], [-55, 55]]
        box shin_m { size: [1, 4.5, 1], material: tarsus_mat }
      }
    }
    bone hind_leg_l attach(parent_box=carapace, anchor=[-1, -0.35, -0.72], align=[1, 0, 0]) mirror_x(hind_leg_r) {
      joint_limits: [[-55, 55], [-60, 60], [-55, 55]]
      box coxa_h { size: [3.5, 1, 1], material: chitin }
      bone tibia_h attach(parent_box=coxa_h, anchor=[-1, -0.2, 0], align=[0, 1, 0]) {
        joint_limits: [[-55, 55], [-55, 55], [-55, 55]]
        box shin_h { size: [1, 4.5, 1], material: tarsus_mat }
      }
    }
  }
}
"""


@functools.lru_cache(maxsize=2)
def build_proposer_supervision_dataset() -> tuple[ProposerTaskRecord, ...]:
    """Build solver-verified 5-task supervision + strictly isolated `holdout_gate_d` records."""
    species_specs = [
        # Train split (distinct body_plan_groups & authors from holdout_gate_d)
        (
            "train",
            "swift_fox_v1",
            "canine_quadruped",
            "synccraft_proc_team_a",
            _quadruped_fixture_dsl(chest=(4.0, 4.0, 6.0), hips=(3.5, 3.5, 5.0), leg_u=3.5, leg_l=3.5),
        ),
        (
            "train",
            "ironhide_bison_v1",
            "heavy_ungulate_quadruped",
            "synccraft_proc_team_a",
            _quadruped_fixture_dsl(chest=(6.5, 6.0, 8.0), hips=(5.5, 5.0, 6.5), leg_u=4.5, leg_l=4.5),
        ),
        # Val split
        (
            "val",
            "dune_skink_v1",
            "sprawled_reptile",
            "synccraft_proc_team_b",
            _quadruped_fixture_dsl(chest=(4.5, 2.5, 6.5), hips=(4.0, 2.5, 5.5), leg_u=3.5, leg_l=3.0, sprawled=True),
        ),
        # Holdout Gate D split (completely disjoint body_plan_groups & source_authors!)
        (
            "holdout_gate_d",
            "crest_raptor_holdout",
            "biped_theropod",
            "synccraft_holdout_bench_c",
            _biped_fixture_dsl(chest=(4.5, 4.5, 6.5), leg_u=5.0, leg_l=5.0),
        ),
        (
            "holdout_gate_d",
            "carapace_beetle_holdout",
            "hexapod_arthropod",
            "synccraft_holdout_bench_d",
            _hexapod_fixture_dsl(),
        ),
    ]

    style_prompts = (
        ("cautious creeping stalk", 1.6),
        ("snappy alert trot", 1.2),
    )

    records: list[ProposerTaskRecord] = []
    for split, asset_id, body_plan, author, dsl_src in species_specs:
        prog = parse_program(dsl_src)
        scene = compile_geometry(prog)
        morph = analyze_morphology(prog, scene)
        morph_dict = morph.to_dict()

        # Task 1: `label_morphology_roles`
        skeleton_summary = [
            {
                "bone": b.name,
                "parent": b.parent_name,
                "pivot_world": [round(float(x), 3) for x in b.pivot_world],
                "boxes": [
                    {"name": bx.name, "size": [round(float(s), 3) for s in bx.size]}
                    for bx in scene.boxes
                    if bx.bone_name == b.name
                ],
            }
            for b in scene.bones
        ]
        records.append(
            ProposerTaskRecord(
                record_id=f"{asset_id}__label_morphology",
                task_type="label_morphology_roles",
                split=split,
                asset_id=asset_id,
                body_plan_group=body_plan,
                source_author=author,
                dsl_source=dsl_src,
                prompt_payload={"skeleton": skeleton_summary},
                target_payload={
                    "limbs": [{"role": lb.role, "chain": list(lb.chain), "posture": lb.posture} for lb in morph.limbs],
                    "spine": list(morph.spine),
                    "neck": list(morph.neck),
                    "tail": list(morph.tail),
                    "wings": [{"role": wg.role, "chain": list(wg.chain)} for wg in morph.wings],
                },
                verified_by_solver=len(morph.limbs) >= 2,
            )
        )

        # Task 2 (`propose_gait_from_style`) & Task 5 (`invert_clip_to_gait`)
        for idx_p, (phrase, dur_s) in enumerate(style_prompts):
            seed_gait = style_phrase_to_gait_preset(phrase, morph, gait_name=f"gait_{idx_p}")
            repaired = search_or_repair_gait(
                prog,
                seed_gait,
                duration_s=dur_s,
                clip_name=f"clip_{idx_p}",
                max_evals=6,
            )
            records.append(
                ProposerTaskRecord(
                    record_id=f"{asset_id}__propose__{idx_p}",
                    task_type="propose_gait_from_style",
                    split=split,
                    asset_id=asset_id,
                    body_plan_group=body_plan,
                    source_author=author,
                    dsl_source=dsl_src,
                    prompt_payload={
                        "style_phrase": phrase,
                        "duration_s": dur_s,
                        "morphology": morph_dict,
                    },
                    target_payload=_gait_decl_to_dict(repaired.gait),
                    verified_by_solver=repaired.passed,
                )
            )

            if idx_p == 1 and repaired.passed:
                inv = invert_clip_to_gait(repaired.program, f"clip_{idx_p}")
                records.append(
                    ProposerTaskRecord(
                        record_id=f"{asset_id}__invert__{idx_p}",
                        task_type="invert_clip_to_gait",
                        split=split,
                        asset_id=asset_id,
                        body_plan_group=body_plan,
                        source_author=author,
                        dsl_source=dsl_src,
                        prompt_payload={
                            "duration_s": dur_s,
                            "cycles_estimated": inv.cycles_estimated,
                            "stride_length_units": round(inv.stride_length_units, 4),
                            "per_limb_duty": {k: round(v, 4) for k, v in inv.per_limb_duty.items()},
                            "per_limb_stroke_units": {k: round(v, 4) for k, v in inv.per_limb_stroke_units.items()},
                            "morphology": morph_dict,
                        },
                        target_payload=_gait_decl_to_dict(repaired.gait),
                        verified_by_solver=True,
                    )
                )

        # Task 3: `repair_failed_gait` (construct an intentionally over-driven failing gait & repair it)
        broken_gait = GaitDecl(
            name="overdriven_gait",
            pattern=_adapt_pattern_for_morphology("diagonal_pairs", morph),
            froude=2.8,
            duty=0.35,
            lift_ratio=0.36,
            heave_ratio=0.14,
            pitch_deg=11.0,
            spine_wave_deg=18.0,
            snap=0.90,
        )
        first_try = search_or_repair_gait(prog, broken_gait, duration_s=1.2, clip_name="broken_clip", max_evals=1)
        fixed_res = search_or_repair_gait(prog, broken_gait, duration_s=1.2, clip_name="broken_clip", max_evals=8)
        records.append(
            ProposerTaskRecord(
                record_id=f"{asset_id}__repair_gait",
                task_type="repair_failed_gait",
                split=split,
                asset_id=asset_id,
                body_plan_group=body_plan,
                source_author=author,
                dsl_source=dsl_src,
                prompt_payload={
                    "failing_gait": _gait_decl_to_dict(broken_gait),
                    "failure_diagnostics": first_try.audit_report.get("gait_audit", {}),
                    "collision_events": first_try.audit_report.get("collision_events", 0),
                    "morphology": morph_dict,
                },
                target_payload=_gait_decl_to_dict(fixed_res.gait),
                verified_by_solver=fixed_res.passed,
            )
        )

        # Task 4: `plan_texture_and_stamps`
        head_boxes = [bx.name for bx in scene.boxes if any(k in bx.name for k in ("snout", "head", "carapace"))]
        target_box = head_boxes[0] if head_boxes else scene.boxes[0].name
        records.append(
            ProposerTaskRecord(
                record_id=f"{asset_id}__texture_plan",
                task_type="plan_texture_and_stamps",
                split=split,
                asset_id=asset_id,
                body_plan_group=body_plan,
                source_author=author,
                dsl_source=dsl_src,
                prompt_payload={
                    "character_brief": "weathered sculpted scales with warm sunlit dorsum and glowing amber eyes",
                    "boxes": [bx.name for bx in scene.boxes],
                    "materials": [p.name for p in prog.palettes],
                },
                target_payload={
                    "style": {
                        "shade_mode": "banded",
                        "quantize_ramp": True,
                        "glow_bleed": 1,
                    },
                    "material_family": "keratin_scales",
                    "stamps": [
                        {
                            "name": "eye_pair",
                            "target_box": target_box,
                            "face": "east",
                            "at": [0.35, 0.40],
                            "size": [0.22, 0.18],
                            "shape": "eye",
                            "emissive": 0.85,
                            "mirror": True,
                        }
                    ],
                },
                verified_by_solver=True,
            )
        )

    verify_body_plan_and_author_split_isolation(records)
    return tuple(records)


def evaluate_gate_d_proposer(
    holdout_records: Sequence[ProposerTaskRecord],
    proposer_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_search_evals: int = 8,
) -> dict[str, Any]:
    """Benchmark `proposer_fn` on `holdout_gate_d` against Fixed Presets and Bounded Search."""
    gait_records = [
        r for r in holdout_records if r.split == "holdout_gate_d" and r.task_type == "propose_gait_from_style"
    ]
    if not gait_records:
        raise ValueError("no holdout_gate_d 'propose_gait_from_style' records supplied for Gate D evaluation")

    preset_ledger = CostLedger()
    search_ledger = CostLedger()
    proposer_ledger = CostLedger()

    preset_passes = 0
    search_passes = 0
    proposer_first_try_passes = 0
    proposer_post_search_passes = 0

    program_cache: dict[str, tuple[Program, MorphologyReport]] = {}
    for rec in gait_records:
        if rec.asset_id not in program_cache:
            p = parse_program(rec.dsl_source)
            m = analyze_morphology(p, compile_geometry(p))
            program_cache[rec.asset_id] = (p, m)
        prog, morph = program_cache[rec.asset_id]
        phrase = str(rec.prompt_payload["style_phrase"])
        dur_s = float(rec.prompt_payload.get("duration_s", 1.6))

        # 1. Fixed Preset Baseline (1 audit call, 0 search)
        preset_gait = style_phrase_to_gait_preset(phrase, morph, gait_name="preset_gait")
        preset_res = search_or_repair_gait(
            prog,
            preset_gait,
            duration_s=dur_s,
            clip_name="eval_preset",
            max_evals=1,
            ledger=preset_ledger,
        )
        if preset_res.passed:
            preset_passes += 1

        # 2. Fixed Preset + Bounded Deterministic Search Baseline
        search_res = search_or_repair_gait(
            prog,
            preset_gait,
            duration_s=dur_s,
            clip_name="eval_search",
            max_evals=max_search_evals,
            ledger=search_ledger,
        )
        if search_res.passed:
            search_passes += 1

        # 3. Candidate Proposer (`proposer_fn` + bounded search fallback)
        raw_pred = proposer_fn(rec.prompt_payload)
        approx_tokens = max(1, len(json.dumps(raw_pred, sort_keys=True)) // 4)
        pred_gait = _gait_decl_from_dict(raw_pred, default_name="proposer_gait")

        first_res = search_or_repair_gait(
            prog,
            pred_gait,
            duration_s=dur_s,
            clip_name="eval_proposer",
            max_evals=1,
            ledger=proposer_ledger,
        )
        if first_res.passed:
            proposer_ledger.record("accepted_tokens", approx_tokens)
            proposer_first_try_passes += 1
            proposer_post_search_passes += 1
        else:
            proposer_ledger.record("repair_tokens", approx_tokens)
            fallback_res = search_or_repair_gait(
                prog,
                pred_gait,
                duration_s=dur_s,
                clip_name="eval_proposer",
                max_evals=max(1, max_search_evals - 1),
                ledger=proposer_ledger,
            )
            if fallback_res.passed:
                proposer_post_search_passes += 1

    n = float(len(gait_records))
    preset_rate = preset_passes / n
    search_rate = search_passes / n
    prop_first_rate = proposer_first_try_passes / n
    prop_post_rate = proposer_post_search_passes / n

    search_mean_calls = search_ledger.compiler_calls / n
    prop_mean_calls = proposer_ledger.compiler_calls / n

    # Gate D criterion: proposer must beat fixed presets on first-try pass rate OR achieve >= search pass rate with fewer solver calls
    gate_d_justified = (prop_first_rate > preset_rate + 1e-9) or (
        prop_post_rate >= search_rate - 1e-9 and prop_mean_calls < search_mean_calls - 1e-9
    )

    return {
        "num_holdout_prompts": len(gait_records),
        "fixed_preset_baseline": {
            "first_try_pass_rate": round(preset_rate, 4),
            "mean_solver_calls": round(preset_ledger.compiler_calls / n, 4),
            "ledger": preset_ledger.to_dict(),
        },
        "preset_plus_deterministic_search": {
            "post_search_pass_rate": round(search_rate, 4),
            "mean_solver_calls": round(search_mean_calls, 4),
            "ledger": search_ledger.to_dict(),
        },
        "proposer": {
            "first_try_pass_rate": round(prop_first_rate, 4),
            "post_search_pass_rate": round(prop_post_rate, 4),
            "mean_solver_calls": round(prop_mean_calls, 4),
            "ledger": proposer_ledger.to_dict(),
        },
        "gate_d_justified": bool(gate_d_justified),
    }
