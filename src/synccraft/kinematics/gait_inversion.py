"""Inverse Gait Fitting, Bounded Deterministic Gait Search, & Cross-Creature Retargeting (ADR 0013).

Provides three deterministic capabilities bridging authored Blockbench/Bedrock animations,
Block-DSL v4.1 `gait` statements, and Gate D proposer supervision:
1. `invert_clip_to_gait`: Recovers dimensionless `(pattern, froude, duty, phase, lift_ratio,
   heave_ratio, pitch_deg, spine_wave_deg, tail_lag, weight, snap)` from any animated clip
   (`keyframes`, `tracks`, `ik_drives`, or `gait`), supporting both root-motion clips and
   in-place treadmill animations.
2. `search_or_repair_gait`: Failure-directed bounded coordinate search over `GaitDecl`
   guided by the 8-rule biomechanical verifier (`audit_behavior_clip`) with explicit
   `CostLedger` accounting.
3. `retarget_clip_across_models`: Transfers locomotion and secondary axial channels across
   quadrupeds, bipeds, sprawled reptiles, and hexapods via derived `@morphology` roles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from synccraft.dsl.ast_nodes import (
    BehaviorDecl,
    BehaviorStateDecl,
    ClipDecl,
    GAIT_PATTERNS,
    GaitDecl,
    Program,
    TrackDecl,
    validate_program,
)
from synccraft.evaluation.budget import CostLedger
from synccraft.kinematics.gait_solver import (
    _default_gravity_units_s2,
    audit_gait_clip_contracts,
    evaluate_gait_clip,
    resolve_limb_phases,
)
from synccraft.kinematics.morphology import MorphologyReport, analyze_morphology
from synccraft.kinematics.musculoskeletal import audit_behavior_clip, evaluate_behavior_clip
from synccraft.kinematics.se3_chain import compile_geometry


def _circular_diff(a: float, b: float) -> float:
    """Shortest signed wrap-around difference in `[-0.5, 0.5)`."""
    d = (float(a) - float(b) + 0.5) % 1.0 - 0.5
    return d


@dataclass(frozen=True)
class InvertedGaitResult:
    """Recovered `GaitDecl` and diagnostic telemetry from `invert_clip_to_gait`."""

    gait: GaitDecl
    cycles_estimated: int
    stride_length_units: float
    implied_speed_u_s: float
    was_in_place_treadmill: bool
    pattern_phase_residual: float
    per_limb_duty: dict[str, float]
    per_limb_stroke_units: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gait": {
                "name": self.gait.name,
                "pattern": self.gait.pattern,
                "froude": round(float(self.gait.froude), 4),
                "duty": round(float(self.gait.duty), 4),
                "phase": [[k, round(float(v), 4)] for k, v in self.gait.phase],
                "lift_ratio": round(float(self.gait.lift_ratio), 4),
                "heave_ratio": round(float(self.gait.heave_ratio), 4),
                "pitch_deg": round(float(self.gait.pitch_deg), 4),
                "spine_wave_deg": round(float(self.gait.spine_wave_deg), 4),
                "tail_lag": round(float(self.gait.tail_lag), 4),
                "weight": round(float(self.gait.weight), 4),
                "snap": round(float(self.gait.snap), 4),
                "export_root_motion": bool(self.gait.export_root_motion),
            },
            "cycles_estimated": int(self.cycles_estimated),
            "stride_length_units": round(float(self.stride_length_units), 4),
            "implied_speed_u_s": round(float(self.implied_speed_u_s), 4),
            "was_in_place_treadmill": bool(self.was_in_place_treadmill),
            "pattern_phase_residual": round(float(self.pattern_phase_residual), 4),
            "per_limb_duty": {k: round(float(v), 4) for k, v in sorted(self.per_limb_duty.items())},
            "per_limb_stroke_units": {k: round(float(v), 4) for k, v in sorted(self.per_limb_stroke_units.items())},
        }


@dataclass(frozen=True)
class GaitSearchResult:
    """Outcome of bounded deterministic gait search/repair (`search_or_repair_gait`)."""

    passed: bool
    gait: GaitDecl
    program: Program
    audit_report: dict[str, Any]
    evaluations_used: int
    repair_steps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "evaluations_used": int(self.evaluations_used),
            "repair_steps": list(self.repair_steps),
            "gait": {
                "name": self.gait.name,
                "pattern": self.gait.pattern,
                "froude": round(float(self.gait.froude), 4),
                "duty": round(float(self.gait.duty), 4),
                "lift_ratio": round(float(self.gait.lift_ratio), 4),
                "heave_ratio": round(float(self.gait.heave_ratio), 4),
                "pitch_deg": round(float(self.gait.pitch_deg), 4),
                "spine_wave_deg": round(float(self.gait.spine_wave_deg), 4),
                "weight": round(float(self.gait.weight), 4),
                "snap": round(float(self.gait.snap), 4),
            },
            "audit_passed": bool(self.audit_report.get("passed", False)),
        }


def _classify_best_gait_pattern(
    recovered_phases: dict[str, float],
    morph: MorphologyReport,
) -> tuple[str, dict[str, float], float]:
    """Match recovered limb touchdown phases against canonical `GAIT_PATTERNS` up to a global phase shift."""
    roles = [lb.role for lb in morph.limbs]
    has_mid = any(r.startswith("mid") for r in roles)
    has_fore = any(r.startswith("fore") for r in roles)

    if not has_fore and not has_mid:
        candidates = ("biped_alternate",)
    elif has_mid:
        candidates = ("hexapod_tripod", "lateral_sequence")
    elif any(lb.posture == "sprawled" for lb in morph.limbs):
        candidates = ("sprawl_diagonal", "diagonal_pairs", "lateral_sequence")
    else:
        candidates = ("lateral_sequence", "diagonal_pairs", "rotary_gallop", "transverse_gallop")

    best_pat = candidates[0]
    best_res = float("inf")
    best_shifted: dict[str, float] = dict(recovered_phases)

    for pat in candidates:
        if pat not in GAIT_PATTERNS:
            continue
        proto_gait = GaitDecl(name="proto", pattern=pat)
        proto_map = resolve_limb_phases(proto_gait, morph)
        common = [r for r in roles if r in recovered_phases and r in proto_map]
        if not common:
            continue
        # Estimate optimal circular phase offset `delta` aligning `recovered + delta ≈ proto`
        angles = [2.0 * math.pi * (proto_map[r] - recovered_phases[r]) for r in common]
        mean_sin = sum(math.sin(a) for a in angles) / len(common)
        mean_cos = sum(math.cos(a) for a in angles) / len(common)
        delta = (math.atan2(mean_sin, mean_cos) / (2.0 * math.pi)) % 1.0
        shifted = {r: (recovered_phases[r] + delta) % 1.0 for r in recovered_phases}
        res = sum(abs(_circular_diff(shifted[r], proto_map[r])) for r in common) / len(common)
        if res < best_res:
            best_res = res
            best_pat = pat
            best_shifted = shifted

    return best_pat, best_shifted, best_res


def invert_clip_to_gait(
    program: Program,
    clip_name: str,
    *,
    num_samples: int = 48,
    gait_name: str | None = None,
) -> InvertedGaitResult:
    """Fit a compact `GaitDecl` to any animated clip via world/body-frame foot trajectory inversion."""
    if program.behavior is None:
        raise ValueError("program has no @behavior section")
    clip_map = {c.name: c for c in program.behavior.clips}
    if clip_name not in clip_map:
        raise ValueError(f"unknown clip {clip_name!r}")
    clip = clip_map[clip_name]
    n_s = max(16, int(num_samples))

    rest_scene = compile_geometry(program)
    morph = analyze_morphology(program, rest_scene)
    if not morph.limbs:
        raise ValueError("cannot invert clip to gait: morphology found 0 ground-contacting limbs")

    # Uniform periodic phase samples in [0, 1) plus endpoint 1.0 to measure net root travel cleanly
    phases = [float(i) / float(n_s) for i in range(n_s)] + [1.0]
    samples = evaluate_behavior_clip(program, clip_name, sample_phases=phases)
    periodic_samples = samples[:n_s]
    end_sample = samples[-1]

    root_bone_name = program.root.name
    rest_root_piv = rest_scene.bone_by_name[root_bone_name].pivot_world.copy()
    fwd_z = float(morph.forward_sign_z)
    mean_h = max(0.5, float(np.mean([lb.hip_height for lb in morph.limbs])))
    g_u = _default_gravity_units_s2(program)

    # Measure total root forward translation across the clip [0, 1]
    root_disp_all = np.zeros((n_s + 1, 3), dtype=np.float64)
    for idx, fr in enumerate(samples):
        cur_piv = fr.scene.bone_by_name[root_bone_name].pivot_world
        root_disp_all[idx] = cur_piv - rest_root_piv

    net_root_fwd = abs(float(root_disp_all[-1, 2] - root_disp_all[0, 2]))
    was_treadmill = net_root_fwd <= 0.20

    # Extract body-relative fore-aft Z (`z_rel`) and ground-relative Y (`y_rel`) per limb
    z_rel_by_role: dict[str, np.ndarray] = {}
    y_rel_by_role: dict[str, np.ndarray] = {}
    for lb in morph.limbs:
        eff_loc = np.asarray(lb.contact_local, dtype=np.float64)
        z_arr = np.zeros(n_s, dtype=np.float64)
        y_arr = np.zeros(n_s, dtype=np.float64)
        for idx, fr in enumerate(periodic_samples):
            tip_b = fr.scene.bone_by_name[lb.chain[-1]]
            p_w = tip_b.pivot_world + tip_b.rotation @ eff_loc
            p_body = p_w - root_disp_all[idx]
            z_arr[idx] = fwd_z * (float(p_body[2]) - float(lb.contact_world[2]))
            y_arr[idx] = float(p_w[1]) - float(morph.ground_y)
        z_rel_by_role[lb.role] = z_arr
        y_rel_by_role[lb.role] = y_arr

    # Estimate integer gait cycles in clip (`1..6`) via FFT harmonic power across all limbs
    max_harmonic = min(6, n_s // 4)
    harmonic_power = np.zeros(max_harmonic + 1, dtype=np.float64)
    for r, z_arr in z_rel_by_role.items():
        centered = z_arr - float(np.mean(z_arr))
        spec = np.fft.rfft(centered)
        for k_h in range(1, max_harmonic + 1):
            harmonic_power[k_h] += float(abs(spec[k_h]) ** 2)
    cycles_est = int(np.argmax(harmonic_power[1:]) + 1) if np.max(harmonic_power[1:]) > 1e-6 else 1

    # Phase-fold trajectories onto a single canonical cycle `[0, 1)` of resolution `M = n_s`
    m_bins = n_s
    per_limb_duty: dict[str, float] = {}
    per_limb_stroke: dict[str, float] = {}
    per_limb_lift: dict[str, float] = {}
    raw_phases: dict[str, float] = {}

    for lb in morph.limbs:
        z_arr = z_rel_by_role[lb.role]
        y_arr = y_rel_by_role[lb.role]
        stroke = float(np.max(z_arr) - np.min(z_arr))
        per_limb_stroke[lb.role] = stroke
        lift_max = max(0.0, float(np.max(y_arr) - np.min(y_arr)))
        per_limb_lift[lb.role] = lift_max

        if stroke <= 0.05:
            per_limb_duty[lb.role] = 0.70
            raw_phases[lb.role] = 0.0
            continue

        # Fold onto one cycle if `cycles_est > 1`
        z_cyc = np.zeros(m_bins, dtype=np.float64)
        y_cyc = np.zeros(m_bins, dtype=np.float64)
        counts = np.zeros(m_bins, dtype=np.float64)
        for idx in range(n_s):
            c_phi = ((float(idx) / float(n_s)) * float(cycles_est)) % 1.0
            b_idx = int(math.floor(c_phi * m_bins)) % m_bins
            z_cyc[b_idx] += z_arr[idx]
            y_cyc[b_idx] += y_arr[idx]
            counts[b_idx] += 1.0
        # Fill any empty bins by circular interpolation
        for b_idx in range(m_bins):
            if counts[b_idx] > 0:
                z_cyc[b_idx] /= counts[b_idx]
                y_cyc[b_idx] /= counts[b_idx]
            else:
                prev_b = (b_idx - 1) % m_bins
                z_cyc[b_idx] = z_cyc[prev_b]
                y_cyc[b_idx] = y_cyc[prev_b]

        # Backward body-frame stroke (`dz/dc_phi < 0`) near ground height defines stance
        dz = np.roll(z_cyc, -1) - np.roll(z_cyc, 1)
        y_min = float(np.min(y_cyc))
        y_thresh = y_min + max(0.18, 0.30 * lift_max)
        in_stance_mask = (dz <= 1e-4) & (y_cyc <= y_thresh)
        duty_r = float(np.mean(in_stance_mask))
        if duty_r < 0.25 or duty_r > 0.88:
            # Fallback to purely kinematic backward-stroke fraction when clip lacks vertical foot lift
            duty_r = float(np.clip(np.mean(dz <= 0.0), 0.32, 0.82))
        per_limb_duty[lb.role] = duty_r

        # Touchdown occurs at peak forward body-relative Z (`u = 0` -> `z_rel = +0.5 * duty * S`)
        td_bin = int(np.argmax(z_cyc))
        cyc_prog_td = float(td_bin) / float(m_bins)
        raw_phases[lb.role] = (-cyc_prog_td) % 1.0

    duty_est = float(np.clip(np.median(list(per_limb_duty.values())), 0.30, 0.85))
    mean_stroke = float(np.mean(list(per_limb_stroke.values())))
    if not was_treadmill:
        stride_len = net_root_fwd / float(max(1, cycles_est))
    else:
        # In body frame, foot stroke equals `duty * stride_length`
        stride_len = mean_stroke / max(0.30, duty_est)

    dur_s = max(0.1, float(clip.duration_s))
    # Estimate dimensionless Froude both from stride scaling (`S / h ≈ 2.3 * Fr^0.30`) and speed (`v^2 / (g * h)`)
    implied_speed = (stride_len * float(cycles_est)) / dur_s
    fr_from_speed = (implied_speed * implied_speed) / max(1e-6, g_u * mean_h)
    rel_stride = stride_len / max(0.5, mean_h)
    fr_from_stride = (max(0.05, rel_stride / 2.3)) ** (1.0 / 0.30)
    # Blend speed-based and Alexander-stride-based Froude so keyframed clips with arbitrary duration stay physical
    fr_est = float(np.clip(0.55 * fr_from_speed + 0.45 * fr_from_stride, 0.02, 3.5))

    best_pattern, aligned_phases, pat_res = _classify_best_gait_pattern(raw_phases, morph)

    # Secondary torso heave, pitch, spine wave, and tail lag telemetry
    root_y_series = root_disp_all[:n_s, 1]
    heave_amp = 0.5 * float(np.max(root_y_series) - np.min(root_y_series))
    heave_ratio = float(np.clip(heave_amp / max(0.5, mean_h), 0.0, 0.18))
    lift_ratio = float(np.clip(np.mean(list(per_limb_lift.values())) / max(0.5, mean_h), 0.04, 0.40))

    non_root_spine = [b for b in morph.spine if b != root_bone_name]
    if non_root_spine:
        yaw_sum = np.zeros(n_s, dtype=np.float64)
        pitch_sum = np.zeros(n_s, dtype=np.float64)
        for idx, fr in enumerate(periodic_samples):
            for sb in non_root_spine:
                eul = fr.pose_euler_deltas.get(sb, (0.0, 0.0, 0.0))
                pitch_sum[idx] += float(eul[0])
                yaw_sum[idx] += float(eul[1])
        spine_wave_deg = float(np.clip(0.5 * (np.max(yaw_sum) - np.min(yaw_sum)), 0.0, 22.0))
        pitch_deg = float(np.clip(0.5 * (np.max(pitch_sum) - np.min(pitch_sum)), 0.0, 14.0))
    else:
        root_pitch = np.array(
            [fr.pose_euler_deltas.get(root_bone_name, (0.0, 0.0, 0.0))[0] for fr in periodic_samples],
            dtype=np.float64,
        )
        pitch_deg = float(np.clip(np.max(root_pitch) - np.min(root_pitch), 0.0, 14.0))
        spine_wave_deg = 2.0

    # Keep explicit phase overrides only when residual from canonical pattern exceeds 0.04 cycles
    proto_phases = resolve_limb_phases(GaitDecl(name="proto", pattern=best_pattern), morph)
    explicit_phase: tuple[tuple[str, float], ...] = ()
    if pat_res > 0.04:
        explicit_phase = tuple(
            (lb.role, round(float(aligned_phases.get(lb.role, proto_phases.get(lb.role, 0.0))), 4))
            for lb in morph.limbs
        )

    weight_est = float(np.clip(0.35 + 0.8 * (duty_est - 0.55) + 2.0 * heave_ratio, 0.1, 1.0))
    snap_est = float(np.clip(0.25 + 0.35 * min(1.5, fr_est), 0.05, 0.95))

    recovered = GaitDecl(
        name=gait_name or f"{clip_name}_gait",
        pattern=best_pattern,
        froude=round(fr_est, 4),
        duty=round(duty_est, 4),
        phase=explicit_phase,
        lift_ratio=round(lift_ratio, 4),
        heave_ratio=round(heave_ratio, 4),
        pitch_deg=round(pitch_deg, 4),
        spine_wave_deg=round(spine_wave_deg, 4),
        tail_lag=0.20,
        weight=round(weight_est, 4),
        snap=round(snap_est, 4),
        export_root_motion=not was_treadmill,
    )
    return InvertedGaitResult(
        gait=recovered,
        cycles_estimated=cycles_est,
        stride_length_units=stride_len,
        implied_speed_u_s=implied_speed,
        was_in_place_treadmill=was_treadmill,
        pattern_phase_residual=pat_res,
        per_limb_duty=per_limb_duty,
        per_limb_stroke_units=per_limb_stroke,
    )


def _with_gait_clip(
    program: Program,
    clip_name: str,
    gait: GaitDecl,
    *,
    duration_s: float,
    extra_tracks: tuple[TrackDecl, ...] = (),
) -> Program:
    """Insert or replace `clip_name` on `program` with a single-`gait` `ClipDecl`."""
    new_clip = ClipDecl(
        name=clip_name,
        duration_s=float(duration_s),
        loop=True,
        tracks=extra_tracks,
        gaits=(gait,),
    )
    if program.behavior is None:
        beh = BehaviorDecl(clips=(new_clip,))
    else:
        kept = [c for c in program.behavior.clips if c.name != clip_name]
        kept.append(new_clip)
        beh = replace(program.behavior, clips=tuple(kept))
    updated = replace(program, behavior=beh)
    validate_program(updated)
    return updated


def search_or_repair_gait(
    program: Program,
    initial_gait: GaitDecl,
    *,
    duration_s: float = 1.6,
    clip_name: str = "gait_clip",
    extra_tracks: tuple[TrackDecl, ...] = (),
    num_audit_frames: int = 16,
    max_evals: int = 8,
    ledger: CostLedger | None = None,
) -> GaitSearchResult:
    """Deterministically repair or bound-search `GaitDecl` using 8-rule biomechanical audit diagnostics."""
    cur_gait = initial_gait
    steps_log: list[str] = []
    best_prog = _with_gait_clip(program, clip_name, cur_gait, duration_s=duration_s, extra_tracks=extra_tracks)
    best_aud: dict[str, Any] = {}
    best_gait = cur_gait
    best_score = -1e9

    max_it = max(1, int(max_evals))
    n_fast = max(8, int(num_audit_frames))
    for eval_idx in range(max_it):
        cand_prog = _with_gait_clip(program, clip_name, cur_gait, duration_s=duration_s, extra_tracks=extra_tracks)
        if ledger is not None:
            ledger.record("compiler_calls", 1)
            ledger.record("initial_audit_calls" if eval_idx == 0 else "reaudit_calls", 1)
            if eval_idx > 0:
                ledger.record("projection_steps", 1)

        clip_obj = next(c for c in cand_prog.behavior.clips if c.name == clip_name)  # type: ignore[union-attr]
        fast_frames = evaluate_gait_clip(cand_prog, clip_obj, num_frames=n_fast)
        fast_ga = audit_gait_clip_contracts(cand_prog, clip_obj, fast_frames)

        # Only run full adaptive swept `audit_behavior_clip` when the 8 fast contracts pass (or on the final step)
        if bool(fast_ga.get("passed", False)) or eval_idx + 1 == max_it:
            aud = audit_behavior_clip(cand_prog, clip_name, num_frames=n_fast)
            ga = aud.get("gait_audit") or fast_ga
        else:
            aud = {
                "passed": False,
                "step_bound_satisfied": True,
                "total_collision_events": 0 if fast_ga["checks"].get("clipping", True) else 1,
                "max_stance_slip_units": fast_ga.get("max_stance_slip_units", 0.0),
                "measured_max_step_deg": 0.0,
                "gait_audit": fast_ga,
            }
            ga = fast_ga

        checks = ga.get("checks") or {}
        n_checks_ok = sum(1 for v in checks.values() if v)
        col_ev = int(aud.get("total_collision_events", aud.get("collision_events", 0)))
        score = (
            100.0 * n_checks_ok
            - 5.0 * float(col_ev)
            - 20.0 * float(aud.get("max_stance_slip_units", 0.0))
            - (25.0 if not bool(aud.get("step_bound_satisfied", True)) else 0.0)
        )
        if score > best_score:
            best_score = score
            best_prog = cand_prog
            best_aud = aud
            best_gait = cur_gait

        if bool(aud.get("passed", False)):
            return GaitSearchResult(
                passed=True,
                gait=cur_gait,
                program=cand_prog,
                audit_report=aud,
                evaluations_used=eval_idx + 1,
                repair_steps=tuple(steps_log),
            )

        if eval_idx + 1 >= max_it:
            break

        # Deterministic failure-directed coordinate adjustment
        new_froude = float(cur_gait.froude)
        new_duty = float(cur_gait.duty)
        new_lift = float(cur_gait.lift_ratio)
        new_heave = float(cur_gait.heave_ratio)
        new_pitch = float(cur_gait.pitch_deg)
        new_spine = float(cur_gait.spine_wave_deg)
        new_snap = float(cur_gait.snap)
        new_pat = cur_gait.pattern
        reasons: list[str] = []

        if not checks.get("clipping", True) or col_ev > 0:
            new_froude = max(0.04, 0.65 * new_froude)
            new_lift = max(0.05, 0.70 * new_lift)
            new_spine = max(0.0, 0.55 * new_spine)
            new_pitch = max(0.0, 0.60 * new_pitch)
            new_heave = max(0.0, 0.60 * new_heave)
            reasons.append("reduce_stride_lift_and_axial_sway_for_clipping")

        if not checks.get("reach", True) or not checks.get("foot_slip", True):
            new_froude = max(0.04, 0.68 * new_froude)
            new_duty = min(0.78, new_duty + 0.06)
            new_heave = max(0.0, 0.55 * new_heave)
            reasons.append("shorten_froude_and_heave_for_reach_or_slip")

        if not checks.get("balance", True):
            new_duty = min(0.82, max(new_duty + 0.08, 0.68))
            if new_froude <= 0.30 and new_pat in ("rotary_gallop", "transverse_gallop"):
                new_pat = "lateral_sequence"
            reasons.append("widen_duty_support_for_balance")

        if not checks.get("ground", True):
            new_froude = max(0.05, 0.55 * new_froude)
            new_lift = float(np.clip(0.65 * new_lift, 0.06, 0.14))
            new_heave = max(0.0, 0.45 * new_heave)
            new_pitch = max(0.5, 0.45 * new_pitch)
            new_spine = max(1.0, 0.50 * new_spine)
            new_duty = min(0.74, max(0.55, new_duty + 0.08))
            new_snap = max(0.10, 0.55 * new_snap)
            reasons.append("clamp_froude_pitch_lift_and_heave_for_ground_clearance")

        if not bool(aud.get("step_bound_satisfied", True)):
            new_froude = max(0.04, 0.55 * new_froude)
            new_lift = max(0.05, 0.68 * new_lift)
            new_snap = max(0.05, 0.50 * new_snap)
            new_duty = min(0.76, new_duty + 0.05)
            reasons.append("soften_snap_and_froude_for_frame_step_bound")

        if not checks.get("joint_limits", True):
            new_spine = max(0.0, 0.50 * new_spine)
            new_pitch = max(0.0, 0.50 * new_pitch)
            new_froude = max(0.04, 0.80 * new_froude)
            reasons.append("tighten_axial_angles_for_joint_limits")

        if not reasons:
            new_froude = max(0.05, 0.80 * new_froude)
            new_duty = float(np.clip(new_duty, 0.50, 0.75))
            reasons.append("conservative_froude_backoff")

        steps_log.append("+".join(reasons))
        cur_gait = replace(
            cur_gait,
            pattern=new_pat,
            froude=round(new_froude, 4),
            duty=round(new_duty, 4),
            lift_ratio=round(new_lift, 4),
            heave_ratio=round(new_heave, 4),
            pitch_deg=round(new_pitch, 4),
            spine_wave_deg=round(new_spine, 4),
            snap=round(new_snap, 4),
        )

    return GaitSearchResult(
        passed=bool(best_aud.get("passed", False)),
        gait=best_gait,
        program=best_prog,
        audit_report=best_aud,
        evaluations_used=max_it,
        repair_steps=tuple(steps_log),
    )


def _adapt_pattern_for_morphology(pattern: str, dst_morph: MorphologyReport) -> str:
    """Adapt canonical gait pattern when retargeting across biped / quadruped / sprawled / hexapod rigs."""
    roles = [lb.role for lb in dst_morph.limbs]
    has_mid = any(r.startswith("mid") for r in roles)
    has_fore = any(r.startswith("fore") for r in roles)

    if not has_fore and not has_mid:
        return "biped_alternate"
    if has_mid:
        if pattern in ("diagonal_pairs", "sprawl_diagonal", "rotary_gallop", "transverse_gallop"):
            return "hexapod_tripod"
        return "lateral_sequence"
    if any(lb.posture == "sprawled" for lb in dst_morph.limbs) and pattern == "diagonal_pairs":
        return "sprawl_diagonal"
    if pattern in ("biped_alternate", "hexapod_tripod"):
        return "diagonal_pairs"
    return pattern


def retarget_clip_across_models(
    src_program: Program,
    clip_name: str,
    dst_program: Program,
    *,
    new_clip_name: str | None = None,
    repair_if_needed: bool = True,
    max_repair_evals: int = 8,
    ledger: CostLedger | None = None,
) -> GaitSearchResult:
    """Retarget a locomotion (+ secondary axial) clip from `src_program` onto `dst_program` via `@morphology`."""
    if src_program.behavior is None:
        raise ValueError("source program has no @behavior section")
    cmap = {c.name: c for c in src_program.behavior.clips}
    if clip_name not in cmap:
        raise ValueError(f"unknown source clip {clip_name!r}")
    src_clip = cmap[clip_name]
    out_name = new_clip_name or clip_name

    src_morph = analyze_morphology(src_program, compile_geometry(src_program))
    dst_scene = compile_geometry(dst_program)
    dst_morph = analyze_morphology(dst_program, dst_scene)
    if not dst_morph.limbs:
        raise ValueError("destination program has 0 ground-contacting limbs in @morphology")

    if getattr(src_clip, "gaits", ()):
        base_gait = src_clip.gaits[0]
    else:
        inv = invert_clip_to_gait(src_program, clip_name, gait_name=f"{out_name}_gait")
        base_gait = inv.gait

    adapted_pattern = _adapt_pattern_for_morphology(base_gait.pattern, dst_morph)
    # Drop source-specific explicit phase overrides when limb topology differs so `resolve_limb_phases` projects cleanly
    src_roles = {lb.role for lb in src_morph.limbs}
    dst_roles = {lb.role for lb in dst_morph.limbs}
    kept_phase = base_gait.phase if (base_gait.phase and src_roles == dst_roles) else ()

    dst_gait = replace(
        base_gait,
        name=f"{out_name}_gait",
        pattern=adapted_pattern,
        phase=kept_phase,
    )

    # Map secondary wing/ear/jaw non-limb `TrackDecl`s proportionally across matching `@morphology` chains
    from synccraft.kinematics.musculoskeletal import _collect_expanded_joint_limits

    limits_map = _collect_expanded_joint_limits(dst_program)
    dst_root = dst_program.root.name
    mapped_tracks: list[TrackDecl] = []

    def _safe_amp(bone_n: str, axis: str, amp: float, bias: float) -> tuple[float, float] | None:
        ax_i = {"x": 0, "y": 1, "z": 2}[axis]
        if bone_n in limits_map and limits_map[bone_n][0] is not None:
            jlim, ang_sign = limits_map[bone_n]
            lo, hi = float(jlim[ax_i][0]), float(jlim[ax_i][1])
            if math.isfinite(lo) and math.isfinite(hi):
                if float(ang_sign[ax_i]) < 0.0:
                    lo, hi = -hi, -lo
                half_span = 0.45 * (hi - lo)
                if half_span <= 0.25:
                    return None
                c_bias = float(np.clip(bias, lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)))
                max_a = min(abs(amp), max(0.0, hi - c_bias - 0.05), max(0.0, c_bias - lo - 0.05))
                if max_a <= 0.1:
                    return None
                return round(math.copysign(max_a, amp), 4), round(c_bias, 4)
        return round(float(amp), 4), round(float(bias), 4)

    src_wing_by_bone: dict[str, tuple[str, float]] = {}
    for wg in src_morph.wings:
        for i_w, wb in enumerate(wg.chain):
            src_wing_by_bone[wb] = (wg.role, float(i_w) / float(max(1, len(wg.chain) - 1)))
    dst_wing_by_role = {wg.role: wg.chain for wg in dst_morph.wings}

    for tr in src_clip.tracks:
        if tr.bone in src_wing_by_bone:
            w_role, rel_idx = src_wing_by_bone[tr.bone]
            if w_role in dst_wing_by_role and dst_wing_by_role[w_role]:
                d_chain = dst_wing_by_role[w_role]
                d_bone = d_chain[min(len(d_chain) - 1, int(round(rel_idx * (len(d_chain) - 1))))]
                if d_bone != dst_root:
                    bounded = _safe_amp(d_bone, tr.axis, tr.amp_deg, tr.bias_deg)
                    if bounded is not None:
                        mapped_tracks.append(
                            replace(tr, bone=d_bone, amp_deg=bounded[0], bias_deg=bounded[1])
                        )

    return search_or_repair_gait(
        dst_program,
        dst_gait,
        duration_s=float(src_clip.duration_s),
        clip_name=out_name,
        extra_tracks=tuple(mapped_tracks),
        max_evals=max_repair_evals if repair_if_needed else 1,
        ledger=ledger,
    )
