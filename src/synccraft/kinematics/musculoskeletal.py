"""Biomechanical Musculoskeletal, Tendon-Coupled & Closed-Loop Behavior Engine (v3.2).

Implements ADR 0006 reduced-order qualitative musculoskeletal control and multi-phase
motion synthesis atop SyncCraft's exact right-handed SE(3) chain (`compile_posed_geometry`):

1. Normalized Zajac (1989) / Hill agonist-antagonist muscle equilibrium (`@muscles`, H8):
   - Active Gaussian force-length curve ``f_L(L_tilde) = exp(-((L_tilde - 1) / w)^2)``
   - Passive parallel elastic curve ``f_PE(L_tilde) = max(0, L_tilde - 1)^2``
   - Bracketed torque-balance root-finding with activation- & stretch-modulated joint stiffness
   - Biarticular passive tendon pantograph couplings (`TendonCoupling`)
   - Exact 3D SE(3) origin-to-insertion fiber strain & isochoric `belly_box` `inflate` bulging
2. Multi-bone distributed gaze/aiming chains (`LookChainDecl`) and generalized N-bone
   (`N >= 2`) limb/tentacle IK (`IKChainDecl` + `solve_n_bone_ik`)
3. Analytical, keyframe (`key`, `linear` | `catmullrom`), and 2-parameter damped-spring
   (`spring`) follow-through clips (`@behavior`) plus `@controller` state transitions

Fidelity label: ``reduced_order_qualitative`` (normalized Zajac/Hill force-length torque
equilibrium + exact SE(3) kinematics; not a 3D volumetric FEM soft-tissue simulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from synccraft.dsl.ast_nodes import (
    Bone,
    ClipDecl,
    ControllerDecl,
    IKChainDecl,
    KeyframeTrackDecl,
    LookChainDecl,
    MuscleDecl,
    Program,
    RadialSpec,
    SpringDecl,
    Vec3,
)
from synccraft.kinematics.se3_chain import (
    Scene,
    _angle_sign_from_reflection,
    _clamp_delta_to_joint_limits,
    compile_geometry,
    compile_posed_geometry,
)

if TYPE_CHECKING:
    from synccraft.data_gen.ik_motion_templates import NBoneIKResult

MUSCULOSKELETAL_FIDELITY_LABEL: str = "reduced_order_qualitative"
HILL_ACTIVE_WIDTH_DEFAULT: float = 0.45
HILL_PASSIVE_STIFFNESS_DEFAULT: float = 1.5
HILL_PULLEY_STRAIN_GAIN: float = 0.35


def hill_active_force_length(norm_length: float, width: float = HILL_ACTIVE_WIDTH_DEFAULT) -> float:
    """Normalized Zajac (1989) / Hill active isometric force-length curve ``f_L(L_tilde)``.

    ``f_L(L_tilde) = exp(-((L_tilde - 1.0) / width) ** 2)``, peaking at ``f_L(1.0) == 1.0``.
    """
    if not math.isfinite(norm_length) or not math.isfinite(width) or width <= 0.0:
        raise ValueError("norm_length must be finite and width strictly positive")
    z = (float(norm_length) - 1.0) / float(width)
    return float(math.exp(-(z * z)))


def hill_passive_force_length(norm_length: float) -> float:
    """Normalized Zajac (1989) passive parallel-elastic curve ``f_PE(L_tilde) = max(0, L_tilde - 1)^2``."""
    if not math.isfinite(norm_length):
        raise ValueError("norm_length must be finite")
    stretch = max(0.0, float(norm_length) - 1.0)
    return float(stretch * stretch)


@dataclass(frozen=True)
class MuscleTelemetry:
    """Per-muscle biomechanical state evaluated in exact 3D SE(3) + Zajac/Hill curves."""

    name: str
    activation: float
    rest_length: float
    posed_length: float
    contraction_ratio: float
    strain: float
    applied_bulge_inflate: float
    active_force_fl: float = 1.0
    passive_force_fpe: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "activation": round(self.activation, 5),
            "rest_length": round(self.rest_length, 5),
            "posed_length": round(self.posed_length, 5),
            "contraction_ratio": round(self.contraction_ratio, 5),
            "strain": round(self.strain, 5),
            "applied_bulge_inflate": round(self.applied_bulge_inflate, 5),
            "active_force_fl": round(self.active_force_fl, 5),
            "passive_force_fpe": round(self.passive_force_fpe, 5),
        }


@dataclass(frozen=True)
class MusculoskeletalPoseResult:
    """Evaluated joint deltas, belly-box inflations, posed Scene, and Zajac/Hill telemetry."""

    pose_euler_deltas: dict[str, Vec3]
    box_inflate_deltas: dict[str, float]
    joint_stiffness: dict[str, float]
    muscle_telemetry: tuple[MuscleTelemetry, ...]
    scene: Scene
    fidelity_label: str = MUSCULOSKELETAL_FIDELITY_LABEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity_label": self.fidelity_label,
            "pose_euler_deltas": {k: list(v) for k, v in sorted(self.pose_euler_deltas.items())},
            "box_inflate_deltas": {k: round(v, 5) for k, v in sorted(self.box_inflate_deltas.items())},
            "joint_stiffness": {k: round(v, 5) for k, v in sorted(self.joint_stiffness.items())},
            "muscle_telemetry": [m.to_dict() for m in self.muscle_telemetry],
        }


@dataclass(frozen=True)
class BehaviorFrameSample:
    """Single sampled frame from a @behavior clip."""

    frame_index: int
    time_s: float
    phase: float
    pose_euler_deltas: dict[str, Vec3]
    box_inflate_deltas: dict[str, float]
    root_translation_delta: Vec3
    muscle_activations: dict[str, float]
    effector_world_positions: dict[str, Vec3]
    effector_in_stance: dict[str, bool]
    effector_stance_step_id: dict[str, int]
    scene: Scene
    pose_position_deltas: dict[str, Vec3] = field(default_factory=dict)
    pose_scale_factors: dict[str, Vec3] = field(default_factory=dict)
    root_motion_velocity: Vec3 = (0.0, 0.0, 0.0)


def _bone_anchor_point_world(scene: Scene, bone_name: str, preferred_box: str | None = None) -> np.ndarray:
    """Return representative 3D anatomical attachment point on bone_name in world space."""
    if preferred_box is not None and preferred_box in scene.box_by_name:
        bx = scene.box_by_name[preferred_box]
        if bx.bone_name == bone_name:
            return bx.center.copy()
    for bx in scene.boxes:
        if bx.bone_name == bone_name:
            return bx.center.copy()
    return scene.bone_by_name[bone_name].pivot_world.copy()


def _collect_expanded_joint_limits(program: Program) -> dict[str, tuple[Any, np.ndarray]]:
    """Map expanded bone name -> (joint_limits, angle_sign) matching compile_geometry (including bio_joint presets)."""
    from synccraft.kinematics.bio_constraints import resolve_bone_joint_limits

    out: dict[str, tuple[Any, np.ndarray]] = {}

    def _walk(
        bone: Bone,
        *,
        sign: np.ndarray,
        prefix: str | None = None,
        renamed_root: str | None = None,
    ) -> None:
        name = renamed_root or (f"{prefix}__{bone.name}" if prefix else bone.name)
        eff_lim = resolve_bone_joint_limits(bone)
        if eff_lim is not None:
            out[name] = (eff_lim, _angle_sign_from_reflection(sign))
        for item in bone.items:
            if isinstance(item, Bone):
                _walk(item, sign=sign, prefix=prefix)
                m_flip = (
                    np.array((-1.0, 1.0, 1.0), dtype=np.float64)
                    if item.mirror_axis == "x"
                    else np.array((1.0, 1.0, -1.0), dtype=np.float64)
                )
                if item.mirror_name:
                    m_prefix = f"{prefix}__{item.mirror_name}" if prefix else item.mirror_name
                    _walk(
                        item,
                        sign=sign * m_flip,
                        prefix=m_prefix,
                        renamed_root=m_prefix,
                    )
                if isinstance(item.radial, RadialSpec):
                    for k in range(1, item.radial.count):
                        rn = f"{item.name}_r{k}"
                        rp = f"{prefix}__{rn}" if prefix else rn
                        _walk(item, sign=sign, prefix=rp, renamed_root=rp)
                        if item.mirror_name:
                            rmn = f"{item.mirror_name}_r{k}"
                            rmp = f"{prefix}__{rmn}" if prefix else rmn
                            _walk(item, sign=sign * m_flip, prefix=rmp, renamed_root=rmp)

    _walk(program.root, sign=np.ones(3, dtype=np.float64))
    return out


def _clamp_pose_vec(
    bone_name: str,
    vec: np.ndarray,
    limits_map: dict[str, tuple[Any, np.ndarray]],
) -> np.ndarray:
    if bone_name in limits_map:
        jlim, ang_sign = limits_map[bone_name]
        return _clamp_delta_to_joint_limits(vec, jlim, angle_sign=ang_sign)
    return vec


def _pulley_norm_fiber_length(moment_arm_deg: float, total_joint_angle_deg: float) -> float:
    """Normalized muscle fiber length L_tilde(theta) along signed moment-arm pulley."""
    theta_rad = math.radians(float(total_joint_angle_deg))
    r_norm = float(moment_arm_deg) / 90.0
    return max(0.25, 1.0 - HILL_PULLEY_STRAIN_GAIN * r_norm * theta_rad)


def _solve_hill_axis_equilibrium(
    m_group: list[MuscleDecl],
    act_map: dict[str, float],
    base_angle_deg: float,
    *,
    width: float = HILL_ACTIVE_WIDTH_DEFAULT,
    k_pe: float = HILL_PASSIVE_STIFFNESS_DEFAULT,
) -> tuple[float, float]:
    """Solve Zajac/Hill active f_L + passive f_PE torque balance for delta_theta_deg.

    Finds unique root of:
      tau_net(dtheta) = sum_m (a_m * f_L(L_m) + k_pe * f_PE(L_m)) * r_m
                        - (k_lig + sum_m a_m * f_L(L_m)) * dtheta == 0
    via 60-step bracketed bisection on [-180, 180] degrees.
    """
    max_a = max(act_map[m.name] for m in m_group)
    k_lig = max(0.0, 1.0 - max_a)

    def _tau_and_stiffness(dtheta_deg: float) -> tuple[float, float]:
        theta_tot = base_angle_deg + dtheta_deg
        drive = 0.0
        active_k = 0.0
        passive_k = 0.0
        for m in m_group:
            l_tilde = _pulley_norm_fiber_length(m.moment_arm_deg, theta_tot)
            fl = hill_active_force_length(l_tilde, width=width)
            fpe = hill_passive_force_length(l_tilde)
            a_m = act_map[m.name]
            drive += (a_m * fl + k_pe * fpe) * float(m.moment_arm_deg)
            active_k += a_m * fl
            passive_k += k_pe * fpe
        k_eff = max(1e-12, k_lig + active_k)
        tau = drive - k_eff * dtheta_deg
        return tau, k_eff + passive_k

    tau_0, stiff_0 = _tau_and_stiffness(0.0)
    if abs(tau_0) <= 1e-12:
        return 0.0, stiff_0

    lo, hi = -180.0, 180.0
    tau_lo, _ = _tau_and_stiffness(lo)
    tau_hi, _ = _tau_and_stiffness(hi)
    if tau_lo <= 0.0:
        _, st = _tau_and_stiffness(lo)
        return lo, st
    if tau_hi >= 0.0:
        _, st = _tau_and_stiffness(hi)
        return hi, st

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        tau_mid, _ = _tau_and_stiffness(mid)
        if abs(tau_mid) <= 1e-13:
            lo = hi = mid
            break
        if tau_mid > 0.0:
            lo = mid
        else:
            hi = mid

    sol = 0.5 * (lo + hi)
    _, stiff_sol = _tau_and_stiffness(sol)
    return sol, stiff_sol


def _compute_muscle_bulge_and_telemetry(
    rest_scene: Scene,
    kin_scene: Scene,
    muscles: tuple[MuscleDecl, ...],
    act_map: dict[str, float],
    poses: dict[str, Vec3],
) -> tuple[dict[str, float], tuple[MuscleTelemetry, ...]]:
    """Measure exact 3D SE(3) origin-to-insertion fiber strain, Zajac f_L/f_PE, and belly bulge."""
    inflate_deltas: dict[str, float] = {}
    telemetry: list[MuscleTelemetry] = []
    axis_map = {"x": 0, "y": 1, "z": 2}

    for m in muscles:
        a_val = float(np.clip(act_map.get(m.name, 0.0), 0.0, 1.0))
        p_orig_0 = _bone_anchor_point_world(rest_scene, m.origin_bone)
        p_ins_0 = _bone_anchor_point_world(rest_scene, m.insertion_bone, preferred_box=m.belly_box)
        rest_len = float(np.linalg.norm(p_ins_0 - p_orig_0))

        p_orig_q = _bone_anchor_point_world(kin_scene, m.origin_bone)
        p_ins_q = _bone_anchor_point_world(kin_scene, m.insertion_bone, preferred_box=m.belly_box)
        posed_len = float(np.linalg.norm(p_ins_q - p_orig_q))

        ins_pose = poses.get(m.insertion_bone, (0.0, 0.0, 0.0))
        ins_theta = float(ins_pose[axis_map[m.action_axis]])
        pulley_l = _pulley_norm_fiber_length(m.moment_arm_deg, ins_theta)

        if rest_len > 1e-5:
            chord_ratio = posed_len / max(1e-4, rest_len)
            # When belly_box resides on origin_bone (e.g. bicep_belly on upper arm), chord_ratio is 1.0;
            # blend geometric chord ratio with signed pulley fiber ratio so f_L/f_PE reflect real excursion.
            ratio = chord_ratio if abs(chord_ratio - 1.0) > 1e-4 else pulley_l
        else:
            ratio = pulley_l

        strain = ratio - 1.0
        fl_val = hill_active_force_length(pulley_l)
        fpe_val = hill_passive_force_length(pulley_l)
        shortening_gain = float(np.clip(0.68 + 0.32 * np.clip((1.0 - ratio) / 0.25, -0.5, 1.0), 0.35, 1.0))
        bulge = float(a_val * fl_val * m.bulge_inflate * shortening_gain) if m.belly_box is not None else 0.0
        if m.belly_box is not None and bulge > 0.0:
            inflate_deltas[m.belly_box] = min(0.35, max(inflate_deltas.get(m.belly_box, 0.0), bulge))

        telemetry.append(
            MuscleTelemetry(
                name=m.name,
                activation=a_val,
                rest_length=rest_len,
                posed_length=posed_len,
                contraction_ratio=ratio,
                strain=strain,
                applied_bulge_inflate=bulge,
                active_force_fl=fl_val,
                passive_force_fpe=fpe_val,
            )
        )
    return inflate_deltas, tuple(telemetry)


def evaluate_muscles_at_activations(
    program: Program,
    activations: dict[str, float] | None = None,
    *,
    base_pose_deltas: dict[str, Vec3] | None = None,
    root_translation_delta: Vec3 | None = None,
    pose_position_deltas: dict[str, Vec3] | None = None,
    pose_scale_factors: dict[str, Vec3] | None = None,
    validate: bool = True,
    _rest_scene: Scene | None = None,
    _limits_map: dict[str, tuple[Any, np.ndarray]] | None = None,
    _skip_scene: bool = False,
) -> MusculoskeletalPoseResult:
    """Compute Zajac/Hill force-length joint equilibrium, tendon couplings, and belly bulges."""
    rest_scene = _rest_scene if _rest_scene is not None else compile_geometry(program, validate=validate)
    limits_map = _limits_map if _limits_map is not None else _collect_expanded_joint_limits(program)
    act_in = activations or {}
    poses: dict[str, np.ndarray] = {
        b: np.asarray(v, dtype=np.float64).copy() for b, v in (base_pose_deltas or {}).items()
    }
    stiffness_out: dict[str, float] = {}
    axis_map = {"x": 0, "y": 1, "z": 2}

    muscles: tuple[MuscleDecl, ...] = program.muscles
    act_map: dict[str, float] = {}
    by_insertion_axis: dict[tuple[str, str], list[MuscleDecl]] = {}
    for m in muscles:
        a_val = float(np.clip(act_in.get(m.name, 0.0), 0.0, 1.0))
        act_map[m.name] = a_val
        by_insertion_axis.setdefault((m.insertion_bone, m.action_axis), []).append(m)

    # 1. Per-axis Zajac (1989) / Hill active f_L + passive f_PE torque balance
    for (bone_name, ax_char), m_group in by_insertion_axis.items():
        ax_i = axis_map[ax_char]
        cur_vec = poses.setdefault(bone_name, np.zeros(3, dtype=np.float64))
        base_theta = float(cur_vec[ax_i])
        delta_eq, k_eff = _solve_hill_axis_equilibrium(m_group, act_map, base_theta)
        cur_vec[ax_i] = base_theta + delta_eq
        stiffness_out[bone_name] = max(stiffness_out.get(bone_name, 0.0), float(k_eff))

    # Enforce anatomical joint_limits before propagating passive tendon pantographs
    for b_name in list(poses):
        poses[b_name] = _clamp_pose_vec(b_name, poses[b_name], limits_map)

    # 2. Biarticular passive tendon pantograph couplings (coupled from clamped primary excursion)
    coupled_counts: dict[tuple[str, str], list[MuscleDecl]] = {}
    for m in muscles:
        if m.tendon_coupling is not None:
            coupled_counts.setdefault((m.insertion_bone, m.action_axis), []).append(m)

    for m in muscles:
        if m.tendon_coupling is not None:
            tc = m.tendon_coupling
            src_vec = poses.get(m.insertion_bone, np.zeros(3, dtype=np.float64))
            src_angle = float(src_vec[axis_map[m.action_axis]])
            peer_coupled = coupled_counts[(m.insertion_bone, m.action_axis)]
            if len(peer_coupled) <= 1:
                weight = 1.0
            else:
                denom = sum(act_map[p.name] for p in peer_coupled)
                weight = (act_map[m.name] / denom) if denom > 1e-6 else (1.0 / len(peer_coupled))
            tgt_vec = poses.setdefault(tc.bone, np.zeros(3, dtype=np.float64))
            tgt_vec[axis_map[tc.axis]] += tc.ratio * src_angle * weight

    for b_name in list(poses):
        poses[b_name] = _clamp_pose_vec(b_name, poses[b_name], limits_map)

    pose_tuples: dict[str, Vec3] = {
        b: (float(v[0]), float(v[1]), float(v[2])) for b, v in poses.items()
    }

    if _skip_scene and not muscles:
        return MusculoskeletalPoseResult(
            pose_euler_deltas=pose_tuples,
            box_inflate_deltas={},
            joint_stiffness=stiffness_out,
            muscle_telemetry=(),
            scene=rest_scene,
            fidelity_label=MUSCULOSKELETAL_FIDELITY_LABEL,
        )

    # 3. Pre-bulge posed kinematics to measure exact 3D SE(3) fiber shortening
    kin_scene = compile_posed_geometry(
        program,
        pose_tuples,
        root_translation_delta=root_translation_delta,
        pose_position_deltas=pose_position_deltas,
        pose_scale_factors=pose_scale_factors,
        validate=validate,
    )

    inflate_deltas, telemetry = _compute_muscle_bulge_and_telemetry(
        rest_scene,
        kin_scene,
        muscles,
        act_map,
        pose_tuples,
    )

    final_scene = (
        compile_posed_geometry(
            program,
            pose_tuples,
            box_inflate_deltas=inflate_deltas,
            root_translation_delta=root_translation_delta,
            pose_position_deltas=pose_position_deltas,
            pose_scale_factors=pose_scale_factors,
            validate=validate,
        )
        if inflate_deltas
        else kin_scene
    )

    return MusculoskeletalPoseResult(
        pose_euler_deltas=pose_tuples,
        box_inflate_deltas=inflate_deltas,
        joint_stiffness=stiffness_out,
        muscle_telemetry=telemetry,
        scene=final_scene,
        fidelity_label=MUSCULOSKELETAL_FIDELITY_LABEL,
    )


def solve_look_chain(
    program: Program,
    look_chain: LookChainDecl | str,
    target_world: Vec3 | Any,
    *,
    base_pose_deltas: dict[str, Vec3] | None = None,
    forward_axis: str = "-z",
    max_total_yaw_deg: float = 75.0,
    max_total_pitch_deg: float = 55.0,
) -> dict[str, Vec3]:
    """Distribute yaw/pitch gaze rotation smoothly across a multi-bone neck/head chain."""
    if isinstance(look_chain, str):
        if program.kinematics is None:
            raise KeyError(f"Program has no @kinematics section for look_chain {look_chain!r}")
        lk_map = {c.name: c for c in program.kinematics.look_chains}
        if look_chain not in lk_map:
            raise KeyError(f"Unknown look_chain {look_chain!r}")
        look_chain = lk_map[look_chain]
    poses: dict[str, Vec3] = dict(base_pose_deltas or {})
    scene = compile_posed_geometry(program, poses)
    tip_bone = scene.bone_by_name[look_chain.bones[-1]]
    tgt = np.asarray(target_world, dtype=np.float64)
    rel_world = tgt - tip_bone.pivot_world
    dist = float(np.linalg.norm(rel_world))
    if dist < 1e-6:
        return poses

    first_bone = scene.bone_by_name[look_chain.bones[0]]
    base_rot = scene.bone_by_name[first_bone.parent_name].rotation if first_bone.parent_name else np.eye(3)
    rel_local = base_rot.T @ (rel_world / dist)

    fwd_sign = -1.0 if forward_axis.startswith("-") else 1.0
    fwd_ax = forward_axis[-1]
    if fwd_ax == "z":
        fwd_comp = max(1e-6, abs(float(rel_local[2])))
        yaw_deg = float(
            np.clip(
                fwd_sign * math.degrees(math.atan2(float(rel_local[0]), fwd_comp)),
                -max_total_yaw_deg,
                max_total_yaw_deg,
            )
        )
        pitch_deg = float(
            np.clip(
                -fwd_sign * math.degrees(math.asin(np.clip(float(rel_local[1]), -1.0, 1.0))),
                -max_total_pitch_deg,
                max_total_pitch_deg,
            )
        )
        delta_vec = np.array([pitch_deg, yaw_deg, 0.0], dtype=np.float64)
    elif fwd_ax == "y":
        fwd_comp = max(1e-6, abs(float(rel_local[1])))
        rx_deg = float(
            np.clip(
                fwd_sign * math.degrees(math.atan2(float(rel_local[2]), fwd_comp)),
                -max_total_pitch_deg,
                max_total_pitch_deg,
            )
        )
        rz_deg = float(
            np.clip(
                -fwd_sign * math.degrees(math.asin(np.clip(float(rel_local[0]), -1.0, 1.0))),
                -max_total_yaw_deg,
                max_total_yaw_deg,
            )
        )
        delta_vec = np.array([rx_deg, 0.0, rz_deg], dtype=np.float64)
    else:
        fwd_comp = max(1e-6, abs(float(rel_local[0])))
        yaw_deg = float(
            np.clip(
                -fwd_sign * math.degrees(math.atan2(float(rel_local[2]), fwd_comp)),
                -max_total_yaw_deg,
                max_total_yaw_deg,
            )
        )
        pitch_deg = float(
            np.clip(
                fwd_sign * math.degrees(math.asin(np.clip(float(rel_local[1]), -1.0, 1.0))),
                -max_total_pitch_deg,
                max_total_pitch_deg,
            )
        )
        delta_vec = np.array([0.0, yaw_deg, pitch_deg], dtype=np.float64)

    limits_map = _collect_expanded_joint_limits(program)
    n_bones = len(look_chain.bones)
    raw_weights = np.array(
        [look_chain.weight_decay ** (n_bones - 1 - idx) for idx in range(n_bones)],
        dtype=np.float64,
    )
    weights = raw_weights / np.sum(raw_weights)
    for b_name, w in zip(look_chain.bones, weights):
        cur = np.asarray(poses.get(b_name, (0.0, 0.0, 0.0)), dtype=np.float64)
        upd = _clamp_pose_vec(b_name, cur + w * delta_vec, limits_map)
        poses[b_name] = (float(upd[0]), float(upd[1]), float(upd[2]))
    return poses


def _solve_ik_chain_with_root_delta(
    program: Program,
    chain: IKChainDecl,
    target_world: Vec3,
    *,
    base_pose_deltas: dict[str, Vec3],
    root_translation_delta: Vec3,
    validate: bool = True,
    _limits_cache: dict[str, tuple[Any, np.ndarray]] | None = None,
) -> NBoneIKResult:
    """Solve generalized N-bone IK when the root bone has translated by root_translation_delta."""
    from synccraft.data_gen.ik_motion_templates import NBoneIKResult, solve_n_bone_ik

    shifted_target = (
        float(target_world[0] - root_translation_delta[0]),
        float(target_world[1] - root_translation_delta[1]),
        float(target_world[2] - root_translation_delta[2]),
    )
    res = solve_n_bone_ik(
        program,
        chain.bones,
        chain.effector,
        shifted_target,
        base_pose_deltas=base_pose_deltas,
        hinge_axis=chain.hinge_axis,
        pole_world=chain.pole,
        plane=getattr(chain, "plane", None),
        validate=validate,
        _limits_cache=_limits_cache,
    )
    ach = (
        float(res.achieved_world[0] + root_translation_delta[0]),
        float(res.achieved_world[1] + root_translation_delta[1]),
        float(res.achieved_world[2] + root_translation_delta[2]),
    )
    clamped = (
        float(res.clamped_target_world[0] + root_translation_delta[0]),
        float(res.clamped_target_world[1] + root_translation_delta[1]),
        float(res.clamped_target_world[2] + root_translation_delta[2]),
    )
    pole_w = (
        (
            float(chain.pole[0] + root_translation_delta[0]),
            float(chain.pole[1] + root_translation_delta[1]),
            float(chain.pole[2] + root_translation_delta[2]),
        )
        if chain.pole is not None
        else None
    )
    return NBoneIKResult(
        bones=res.bones,
        pose_euler_deltas=res.pose_euler_deltas,
        achieved_world=ach,
        target_world=target_world,
        clamped_target_world=clamped,
        reachable=res.reachable,
        position_error=res.position_error,
        iterations=res.iterations,
        pole_world=pole_w,
    )


def evaluate_keyframe_track(
    track: KeyframeTrackDecl,
    time_s: float,
) -> Vec3:
    """Evaluate a multi-keyframe rotation/position/scale channel using linear or Catmull-Rom interpolation."""
    keys = track.keys
    n_k = len(keys)
    if n_k == 0:
        return (1.0, 1.0, 1.0) if track.channel == "scale" else (0.0, 0.0, 0.0)
    if n_k == 1 or time_s <= float(keys[0][0]):
        v0 = keys[0][1]
        return (float(v0[0]), float(v0[1]), float(v0[2]))
    if time_s >= float(keys[-1][0]):
        vn = keys[-1][1]
        return (float(vn[0]), float(vn[1]), float(vn[2]))

    seg_i = 0
    for idx in range(n_k - 1):
        if float(keys[idx][0]) <= time_s <= float(keys[idx + 1][0]):
            seg_i = idx
            break
    t0 = float(keys[seg_i][0])
    t1 = float(keys[seg_i + 1][0])
    dt = max(1e-9, t1 - t0)
    u = float(np.clip((time_s - t0) / dt, 0.0, 1.0))

    p1 = np.asarray(keys[seg_i][1], dtype=np.float64)
    p2 = np.asarray(keys[seg_i + 1][1], dtype=np.float64)
    if track.interpolation == "linear" or n_k == 2:
        out = (1.0 - u) * p1 + u * p2
    else:
        p0 = np.asarray(keys[max(0, seg_i - 1)][1], dtype=np.float64)
        p3 = np.asarray(keys[min(n_k - 1, seg_i + 2)][1], dtype=np.float64)
        u2 = u * u
        u3 = u2 * u
        out = 0.5 * (
            2.0 * p1
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
        )
    if track.channel == "scale":
        out = np.maximum(out, 1e-3)
    return (float(out[0]), float(out[1]), float(out[2]))


def evaluate_controller_transition(
    controller: ControllerDecl,
    current_state: str | None,
    parameters: dict[str, float],
) -> tuple[str, float]:
    """Evaluate deterministic @controller state transition rules, returning (next_state, blend_s)."""
    state = current_state or controller.initial
    if state is None:
        state = controller.transitions[0].from_state if controller.transitions else "idle"

    for rule in controller.transitions:
        if rule.from_state != state:
            continue
        val = float(parameters.get(rule.parameter, 0.0))
        thresh = float(rule.threshold)
        op = rule.operator
        matched = (
            (op == ">" and val > thresh)
            or (op == "<" and val < thresh)
            or (op == ">=" and val >= thresh)
            or (op == "<=" and val <= thresh)
            or (op == "==" and abs(val - thresh) <= 1e-9)
            or (op == "!=" and abs(val - thresh) > 1e-9)
        )
        if matched:
            return rule.to_state, float(rule.blend_s)
    return state, 0.0


def _evaluate_track_angle(
    law: str,
    time_s: float,
    duration_s: float,
    amp_deg: float,
    freq_hz: float,
    phase_deg: float,
    bias_deg: float,
) -> float:
    u = (time_s * freq_hz + phase_deg / 360.0) % 1.0
    two_pi_u = 2.0 * math.pi * u
    if law == "wave":
        return bias_deg + amp_deg * math.sin(two_pi_u)
    if law == "flap":
        return bias_deg + amp_deg * (math.sin(two_pi_u) + 0.20 * math.sin(2.0 * two_pi_u))
    if law == "hinge":
        return bias_deg + amp_deg * (0.5 - 0.5 * math.cos(two_pi_u))
    if law == "impulse":
        norm_t = min(1.0, max(0.0, time_s / max(1e-6, duration_s)))
        env = math.sin(math.pi * norm_t) ** 2
        return bias_deg + amp_deg * env * math.sin(two_pi_u)
    if law == "spin":
        return bias_deg + amp_deg * (time_s * freq_hz + phase_deg / 360.0)
    return bias_deg


def _integrate_spring_follow_through(
    program: Program,
    rest_scene: Scene,
    springs: list[SpringDecl],
    times_s: list[float],
    loop: bool,
    duration_s: float,
    raw_poses_per_frame: list[dict[str, Vec3]],
    root_deltas_per_frame: list[Vec3],
    limits_map: dict[str, tuple[Any, np.ndarray]],
) -> list[dict[str, Vec3]]:
    """Integrate 2-parameter damped harmonic secondary follow-through via implicit Euler substepping."""
    n_f = len(times_s)
    if n_f < 2 or not springs:
        return raw_poses_per_frame

    bone_by_name = rest_scene.bone_by_name
    out_poses = [dict(p) for p in raw_poses_per_frame]

    # Collect ancestor chain for parent inertial forcing
    def _ancestor_euler_sum(pose_map: dict[str, Vec3], bname: str) -> np.ndarray:
        acc = np.zeros(3, dtype=np.float64)
        cur_p = bone_by_name[bname].parent_name if bname in bone_by_name else None
        while cur_p is not None and cur_p in bone_by_name:
            if cur_p in pose_map:
                acc = acc + np.asarray(pose_map[cur_p], dtype=np.float64)
            cur_p = bone_by_name[cur_p].parent_name
        return acc

    frame_dt = (duration_s / float(n_f)) if loop else ((times_s[-1] - times_s[0]) / float(max(1, n_f - 1)))
    frame_dt = max(1e-4, frame_dt)
    n_sub = max(4, int(math.ceil(frame_dt / 0.004)))
    h = frame_dt / float(n_sub)

    for sp in springs:
        bname = sp.bone
        k = float(sp.stiffness)
        c = float(sp.damping)
        targets = [np.asarray(p.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64) for p in out_poses]
        anc_angles = [_ancestor_euler_sum(p, bname) for p in out_poses]
        root_pos = [np.asarray(rd, dtype=np.float64) for rd in root_deltas_per_frame]

        # Parent angular velocity + root linear velocity drive passive secondary follow-through
        forcing: list[np.ndarray] = []
        for idx in range(n_f):
            prev_i = (idx - 1) % n_f if loop else max(0, idx - 1)
            d_anc = (anc_angles[idx] - anc_angles[prev_i]) / frame_dt
            d_root = (root_pos[idx] - root_pos[prev_i]) / frame_dt
            # Forward Z/heave Y motion pitches trailing attachments along X; lateral X rolls Z
            lin_coupled = np.array([-3.5 * d_root[2] + 2.5 * d_root[1], 0.0, 3.5 * d_root[0]], dtype=np.float64)
            forcing.append(-0.45 * d_anc + lin_coupled)

        theta = targets[0].copy()
        vel = np.zeros(3, dtype=np.float64)
        passes = 2 if loop else 1
        recorded: list[np.ndarray] = [theta.copy() for _ in range(n_f)]

        for pass_idx in range(passes):
            for idx in range(n_f):
                recorded[idx] = _clamp_pose_vec(bname, theta, limits_map)
                next_i = (idx + 1) % n_f if loop else min(n_f - 1, idx + 1)
                tgt_0, tgt_1 = targets[idx], targets[next_i]
                f_0, f_1 = forcing[idx], forcing[next_i]
                for s_i in range(1, n_sub + 1):
                    alpha = float(s_i) / float(n_sub)
                    tgt_sub = (1.0 - alpha) * tgt_0 + alpha * tgt_1
                    f_sub = (1.0 - alpha) * f_0 + alpha * f_1
                    # Unconditionally stable backward-Euler step for x'' + c x' + k (x - tgt) = f_sub
                    denom = 1.0 + h * c + h * h * k
                    vel = (vel + h * (f_sub - k * (theta - tgt_sub))) / denom
                    theta = theta + h * vel
            if not loop or pass_idx == passes - 1:
                break

        for idx in range(n_f):
            v = recorded[idx]
            out_poses[idx][bname] = (float(v[0]), float(v[1]), float(v[2]))

    return out_poses


def resolve_clip_effect_events(program: Program, clip: ClipDecl) -> list[dict[str, Any]]:
    """Expand explicit timestamp and stance-touchdown `effect` triggers into sorted event dicts (ADR 0013)."""
    effects = getattr(clip, "effects", ())
    if not effects:
        return []
    events: list[dict[str, Any]] = []
    touchdown_times: list[tuple[float, str | None]] | None = None

    for eff in effects:
        if eff.trigger == "time":
            events.append(
                {
                    "time_s": round(float(eff.time_s), 4),
                    "sound": eff.sound,
                    "particle": eff.particle,
                    "locator": eff.locator,
                    "instruction": eff.instruction,
                }
            )
        elif eff.trigger == "touchdown":
            if touchdown_times is None:
                touchdown_times = []
                if getattr(clip, "gaits", ()):
                    from synccraft.kinematics.gait_solver import (
                        analyze_morphology,
                        compute_froude_kinematics,
                        resolve_limb_phases,
                    )

                    rs = compile_geometry(program, validate=False)
                    morph = analyze_morphology(program, rs)
                    gd = clip.gaits[0]
                    l_phases = resolve_limb_phases(gd, morph)
                    kin = compute_froude_kinematics(program, gd, morph, float(clip.duration_s))
                    n_cyc = max(1, int(kin["cycles_in_clip"]))
                    period_s = float(clip.duration_s) / float(n_cyc)
                    for lb in morph.limbs:
                        ph_off = l_phases.get(lb.role, 0.0)
                        base_u = (-ph_off) % 1.0
                        for c_i in range(n_cyc):
                            t_td = (c_i + base_u) * period_s
                            if 0.0 <= t_td <= float(clip.duration_s) + 1e-6:
                                touchdown_times.append((round(t_td, 4), eff.locator or lb.role))
                elif clip.ik_drives:
                    for ikd in clip.ik_drives:
                        u_td = (-float(ikd.phase_deg) / 360.0) % 1.0
                        touchdown_times.append((round(u_td * float(clip.duration_s), 4), eff.locator or ikd.chain))
                else:
                    touchdown_times.append((0.0, eff.locator))
                touchdown_times.sort(key=lambda item: (item[0], item[1] or ""))
            for t_td, loc_hint in touchdown_times:
                events.append(
                    {
                        "time_s": t_td,
                        "sound": eff.sound,
                        "particle": eff.particle,
                        "locator": eff.locator or loc_hint,
                        "instruction": eff.instruction,
                    }
                )
    events.sort(key=lambda ev: (ev["time_s"], ev.get("sound") or "", ev.get("particle") or ""))
    return events


def _batch_sample_clip_layer_deltas(
    program: Program,
    layer_clip_name: str,
    times_s: list[float],
    weight: float,
    *,
    parent_clip: ClipDecl | None = None,
) -> list[tuple[dict[str, Vec3], dict[str, Vec3]]]:
    """Batch-sample `(euler_deltas, pos_deltas)` across all `times_s` with a single `compile_geometry` pass."""
    if not times_s or program.behavior is None or abs(float(weight)) <= 1e-9:
        return [({}, {}) for _ in times_s]
    cmap = {c.name: c for c in program.behavior.clips}
    src = cmap[layer_clip_name]
    dur = max(1e-6, float(src.duration_s))
    phis: list[float] = []
    if parent_clip is not None and parent_clip.loop and src.loop:
        p_dur = max(1e-6, float(parent_clip.duration_s))
        n_cyc = max(1, int(round(p_dur / dur)))
        for t_s in times_s:
            phis.append((((float(t_s) / p_dur) * float(n_cyc)) % 1.0))
    else:
        for t_s in times_s:
            phis.append(((float(t_s) / dur) % 1.0) if src.loop else min(1.0, max(0.0, float(t_s) / dur)))

    w = float(weight)
    # Fast analytic path when `src` contains only `tracks` / `keyframes` (no IK, muscles, springs, or nested layers)
    if (
        not src.muscle_drives
        and not src.ik_drives
        and not getattr(src, "slide_drives", ())
        and not getattr(src, "gaits", ())
        and not getattr(src, "layers", ())
        and not (program.behavior.springs or (program.kinematics and program.kinematics.springs))
    ):
        ax_map = {"x": 0, "y": 1, "z": 2}
        out_fast: list[tuple[dict[str, Vec3], dict[str, Vec3]]] = []
        for phi in phis:
            ts_src = phi * dur
            e_map: dict[str, list[float]] = {}
            p_map: dict[str, list[float]] = {}
            for tr in src.tracks:
                ang = _evaluate_track_angle(
                    tr.law, ts_src, dur, tr.amp_deg, tr.freq_hz, tr.phase_deg, tr.bias_deg
                )
                v = e_map.setdefault(tr.bone, [0.0, 0.0, 0.0])
                v[ax_map[tr.axis]] += ang
            for kf in src.keyframes:
                kv = evaluate_keyframe_track(kf, ts_src)
                if kf.channel == "rotation":
                    v = e_map.setdefault(kf.bone, [0.0, 0.0, 0.0])
                    v[0] += float(kv[0])
                    v[1] += float(kv[1])
                    v[2] += float(kv[2])
                elif kf.channel == "position":
                    vp = p_map.setdefault(kf.bone, [0.0, 0.0, 0.0])
                    vp[0] += float(kv[0])
                    vp[1] += float(kv[1])
                    vp[2] += float(kv[2])
            out_fast.append(
                (
                    {b: (w * v[0], w * v[1], w * v[2]) for b, v in e_map.items()},
                    {b: (w * v[0], w * v[1], w * v[2]) for b, v in p_map.items()},
                )
            )
        return out_fast

    subs = evaluate_behavior_clip(program, layer_clip_name, sample_phases=phis, _validate=False)
    return [
        (
            {b: (w * float(v[0]), w * float(v[1]), w * float(v[2])) for b, v in sub.pose_euler_deltas.items()},
            {b: (w * float(v[0]), w * float(v[1]), w * float(v[2])) for b, v in sub.pose_position_deltas.items()},
        )
        for sub in subs
    ]


def evaluate_behavior_clip(
    program: Program,
    clip_name: str,
    *,
    num_frames: int = 12,
    active_state: str | None = None,
    sample_phases: Sequence[float] | None = None,
    _validate: bool = True,
    _pre_spring_cache: dict[float, Any] | None = None,
) -> list[BehaviorFrameSample]:
    """Synthesize exact posed, muscle-inflated, keyframe, spring, gait, and IK-locked frames for a @behavior clip."""
    from synccraft.dsl.ast_nodes import validate_program

    if _validate:
        validate_program(program)
    if program.behavior is None:
        raise ValueError("Program has no @behavior section")
    clip_map: dict[str, ClipDecl] = {c.name: c for c in program.behavior.clips}
    if clip_name not in clip_map:
        raise KeyError(f"Unknown @behavior clip {clip_name!r}; available: {sorted(clip_map)}")
    clip = clip_map[clip_name]

    if getattr(clip, "flights", ()):
        from synccraft.kinematics.flight_solver import evaluate_flight_clip

        return evaluate_flight_clip(
            program,
            clip,
            num_frames=num_frames,
            active_state=active_state,
            sample_phases=sample_phases,
            _phase_cache=_pre_spring_cache,
        )

    if getattr(clip, "gaits", ()):
        from synccraft.kinematics.gait_solver import evaluate_gait_clip

        return evaluate_gait_clip(
            program,
            clip,
            num_frames=num_frames,
            active_state=active_state,
            sample_phases=sample_phases,
            _phase_cache=_pre_spring_cache,
        )

    state_pose_bias: dict[str, Vec3] = {}
    state_muscle_tone: dict[str, float] = {}
    if active_state is not None:
        st_map = {s.name: s for s in program.behavior.states}
        if active_state not in st_map:
            raise KeyError(f"Unknown @behavior state {active_state!r}")
        st = st_map[active_state]
        state_pose_bias = dict(st.pose_biases)
        state_muscle_tone = dict(st.muscle_tones)

    rest_scene = compile_geometry(program, validate=False)
    explicit_limits = _collect_expanded_joint_limits(program)
    auto_rig = bool(getattr(clip, "auto_rig", False))
    com_g = float(getattr(clip, "com_balance", 0.0)) or (0.50 if auto_rig else 0.0)
    grasp_g = float(getattr(clip, "digit_grasp", 0.0))
    antag_g = float(getattr(clip, "antagonist_bulge", 0.0)) or (0.65 if auto_rig else 0.0)
    neck_aim_tgt = getattr(clip, "neck_aim", None)
    spine_coil_deg = float(getattr(clip, "spine_coil", 0.0))
    pin_spec = getattr(clip, "ground_pin_limbs", False)
    has_ground_pin = bool(pin_spec is True or (isinstance(pin_spec, (tuple, list)) and len(pin_spec) > 0))
    use_auto_bio = bool(getattr(clip, "auto_bio_limits", False) or auto_rig or has_ground_pin)
    need_rig_asm = bool(
        auto_rig
        or has_ground_pin
        or com_g > 1e-4
        or grasp_g > 1e-4
        or antag_g > 1e-4
        or neck_aim_tgt is not None
        or abs(spine_coil_deg) > 1e-4
    )
    rig_asm = None
    if need_rig_asm:
        from synccraft.kinematics.whole_body_solver import auto_assemble_biomechanical_rig

        rig_asm = auto_assemble_biomechanical_rig(program, rest_scene, explicit_limits=explicit_limits)
        limits_map = rig_asm.bio_constraints if use_auto_bio else explicit_limits
    elif use_auto_bio:
        from synccraft.kinematics.bio_constraints import infer_universal_bio_constraints

        limits_map = infer_universal_bio_constraints(program, rest_scene, explicit_limits=explicit_limits)
    else:
        limits_map = explicit_limits
    prismatic_map: dict[str, tuple[str, tuple[float, float], Vec3 | None]] = {}
    if program.kinematics is not None:
        for pj in getattr(program.kinematics, "prismatic_joints", ()):
            prismatic_map[pj.bone] = (pj.axis, (float(pj.limits[0]), float(pj.limits[1])), pj.axis_vec)
    ik_chain_map: dict[str, IKChainDecl] = (
        {c.name: c for c in program.kinematics.ik_chains} if program.kinematics is not None else {}
    )
    ik_rest_world: dict[str, np.ndarray] = {}
    for cname, cdecl in ik_chain_map.items():
        tip_b = rest_scene.bone_by_name[cdecl.bones[-1]]
        ik_rest_world[cname] = tip_b.pivot_world + tip_b.rotation @ np.asarray(cdecl.effector, dtype=np.float64)

    all_springs: list[SpringDecl] = []
    seen_sp_bones: set[str] = set()
    for sp_src in (
        *(program.kinematics.springs if program.kinematics is not None else ()),
        *program.behavior.springs,
    ):
        if sp_src.bone not in seen_sp_bones:
            seen_sp_bones.add(sp_src.bone)
            all_springs.append(sp_src)

    if clip.ik_drives:
        mean_stride_z = float(np.mean([d.stride for d in clip.ik_drives]))
    else:
        mean_stride_z = 0.0
    mean_stride_vec = np.array([0.0, 0.0, mean_stride_z], dtype=np.float64)
    duty_factor = 0.60

    if sample_phases is not None:
        phases = np.asarray(list(sample_phases), dtype=np.float64)
        n_f = len(phases)
    else:
        n_f = max(2, int(num_frames))
        phases = np.linspace(0.0, 1.0, n_f, endpoint=not clip.loop, dtype=np.float64)
    axis_idx = {"x": 0, "y": 1, "z": 2}

    times_list: list[float] = []
    phases_list: list[float] = []
    root_deltas_list: list[Vec3] = []
    acts_list: list[dict[str, float]] = []
    pos_deltas_list: list[dict[str, Vec3]] = []
    scale_factors_list: list[dict[str, Vec3]] = []
    pre_spring_poses_list: list[dict[str, Vec3]] = []
    pre_spring_ms_res_list: list[MusculoskeletalPoseResult] = []
    eff_pos_list: list[dict[str, Vec3]] = []
    eff_stance_list: list[dict[str, bool]] = []
    eff_step_list: list[dict[str, int]] = []

    for phi_raw in phases:
        phi = float(phi_raw)
        phi_key = round(phi, 8)
        if _pre_spring_cache is not None and phi_key in _pre_spring_cache:
            (
                t_s,
                root_delta,
                acts,
                frame_pos_deltas,
                frame_scale_factors,
                cur_poses,
                ms_res,
                eff_positions,
                eff_stance,
                eff_step_ids,
            ) = _pre_spring_cache[phi_key]
            times_list.append(t_s)
            phases_list.append(phi)
            root_deltas_list.append(root_delta)
            acts_list.append(acts)
            pos_deltas_list.append(frame_pos_deltas)
            scale_factors_list.append(frame_scale_factors)
            pre_spring_poses_list.append(cur_poses)
            pre_spring_ms_res_list.append(ms_res)
            eff_pos_list.append(eff_positions)
            eff_stance_list.append(eff_stance)
            eff_step_list.append(eff_step_ids)
            continue

        t_s = phi * clip.duration_s
        times_list.append(t_s)
        phases_list.append(phi)

        heave_y = 0.0
        if clip.ik_drives and abs(mean_stride_z) > 1e-6:
            max_lift = max(d.lift for d in clip.ik_drives)
            heave_y = -0.10 * max_lift * (0.5 - 0.5 * math.cos(4.0 * math.pi * phi))
        root_delta_arr = phi * mean_stride_vec + np.array([0.0, heave_y, 0.0], dtype=np.float64)
        root_delta: Vec3 = (float(root_delta_arr[0]), float(root_delta_arr[1]), float(root_delta_arr[2]))
        root_deltas_list.append(root_delta)

        # 1. Neural muscle excitation curves
        acts = dict(state_muscle_tone)
        for md in clip.muscle_drives:
            u_m = (t_s * md.freq_hz + md.phase_deg / 360.0) % 1.0
            val = md.baseline + (md.peak - md.baseline) * (0.5 - 0.5 * math.cos(2.0 * math.pi * u_m))
            acts[md.muscle] = float(np.clip(max(acts.get(md.muscle, 0.0), val), 0.0, 1.0))
        acts_list.append(acts)

        # 2. Analytic bone tracks + multi-keyframe channels + state pose_biases
        track_poses: dict[str, np.ndarray] = {
            b: np.asarray(v, dtype=np.float64).copy() for b, v in state_pose_bias.items()
        }
        frame_pos_deltas = {}
        frame_scale_factors = {}

        for tr in clip.tracks:
            ang = _evaluate_track_angle(
                tr.law,
                t_s,
                clip.duration_s,
                tr.amp_deg,
                tr.freq_hz,
                tr.phase_deg,
                tr.bias_deg,
            )
            vec = track_poses.setdefault(tr.bone, np.zeros(3, dtype=np.float64))
            vec[axis_idx[tr.axis]] += ang

        for kf in clip.keyframes:
            kv = evaluate_keyframe_track(kf, t_s)
            if kf.channel == "rotation":
                vec = track_poses.setdefault(kf.bone, np.zeros(3, dtype=np.float64))
                vec += np.asarray(kv, dtype=np.float64)
            elif kf.channel == "position":
                prev_p = frame_pos_deltas.get(kf.bone, (0.0, 0.0, 0.0))
                frame_pos_deltas[kf.bone] = (prev_p[0] + kv[0], prev_p[1] + kv[1], prev_p[2] + kv[2])
            elif kf.channel == "scale":
                prev_s = frame_scale_factors.get(kf.bone, (1.0, 1.0, 1.0))
                frame_scale_factors[kf.bone] = (prev_s[0] * kv[0], prev_s[1] * kv[1], prev_s[2] * kv[2])

        for lyr in getattr(clip, "layers", ()):
            l_eul, l_pos = _batch_sample_clip_layer_deltas(
                program, lyr.source_clip, [t_s], float(lyr.weight), parent_clip=clip
            )[0]
            for lb_n, lv in l_eul.items():
                vec = track_poses.setdefault(lb_n, np.zeros(3, dtype=np.float64))
                vec += np.asarray(lv, dtype=np.float64)
            for lb_n, lpv in l_pos.items():
                prev_p = frame_pos_deltas.get(lb_n, (0.0, 0.0, 0.0))
                frame_pos_deltas[lb_n] = (prev_p[0] + lpv[0], prev_p[1] + lpv[1], prev_p[2] + lpv[2])

        for sd in getattr(clip, "slide_drives", ()):
            u_sd = (t_s * float(sd.freq_hz) + float(sd.phase_deg) / 360.0) % 1.0
            if sd.law in ("piston", "wave"):
                disp = float(sd.bias_units) + float(sd.amp_units) * math.sin(2.0 * math.pi * u_sd)
            elif sd.law == "hinge":
                disp = float(sd.bias_units) + float(sd.amp_units) * (0.5 - 0.5 * math.cos(2.0 * math.pi * u_sd))
            elif sd.law == "impulse":
                disp = float(sd.bias_units) + float(sd.amp_units) * (math.sin(math.pi * u_sd) ** 4)
            else:
                raise ValueError(f"Unsupported slide law {sd.law!r}")
            p_arr = list(frame_pos_deltas.get(sd.bone, (0.0, 0.0, 0.0)))
            if sd.bone in prismatic_map:
                p_axis, (lo_u, hi_u), p_vec = prismatic_map[sd.bone]
                if p_axis == sd.axis:
                    disp = float(np.clip(disp, lo_u, hi_u))
                if p_vec is not None and p_axis == sd.axis:
                    p_arr[0] += disp * float(p_vec[0])
                    p_arr[1] += disp * float(p_vec[1])
                    p_arr[2] += disp * float(p_vec[2])
                else:
                    p_arr[axis_idx[sd.axis]] += disp
            else:
                p_arr[axis_idx[sd.axis]] += disp
            frame_pos_deltas[sd.bone] = (float(p_arr[0]), float(p_arr[1]), float(p_arr[2]))

        from synccraft.kinematics.flight_solver import _apply_muscle_scale_bulge

        _apply_muscle_scale_bulge(program, rest_scene, acts, frame_scale_factors)
        pos_deltas_list.append(frame_pos_deltas)
        scale_factors_list.append(frame_scale_factors)

        base_pose_tuples: dict[str, Vec3] = {
            b: (float(v[0]), float(v[1]), float(v[2])) for b, v in track_poses.items()
        }

        # 3. Musculoskeletal + biarticular tendon solve
        clip_stab = max(float(getattr(clip, "head_stabilize", 0.0)), 0.70 if auto_rig else 0.0)
        needs_post_scene = bool(clip.ik_drives or all_springs or clip_stab > 1e-4 or rig_asm is not None)
        ms_res = evaluate_muscles_at_activations(
            program,
            acts,
            base_pose_deltas=base_pose_tuples,
            root_translation_delta=root_delta,
            pose_position_deltas=frame_pos_deltas or None,
            pose_scale_factors=frame_scale_factors or None,
            validate=False,
            _rest_scene=rest_scene,
            _limits_map=limits_map,
            _skip_scene=needs_post_scene,
        )
        cur_poses = dict(ms_res.pose_euler_deltas)
        if rig_asm is not None:
            from synccraft.kinematics.whole_body_solver import (
                solve_com_balance_and_counterweight,
                solve_foot_roll_and_digit_articulation,
                solve_full_body_antagonist_muscle_bulge,
                solve_serpentine_cervical_wave,
                solve_spine_neck_spline_ik,
            )

            if auto_rig and neck_aim_tgt is None and abs(spine_coil_deg) <= 1e-4:
                for na in rig_asm.neck_assemblies:
                    if len(na.chain) >= 3:
                        cur_poses = solve_serpentine_cervical_wave(
                            rest_scene,
                            na.chain,
                            cur_poses,
                            phase_rad=2.0 * math.pi * phi,
                            pitch_amp_deg=3.5,
                            yaw_amp_deg=3.2,
                            roll_amp_deg=1.4,
                            limits_map=limits_map,
                        )
            if neck_aim_tgt is not None or abs(spine_coil_deg) > 1e-4:
                for na in rig_asm.neck_assemblies:
                    cur_poses = solve_spine_neck_spline_ik(
                        program,
                        rest_scene,
                        na.chain,
                        cur_poses,
                        aim_target_world=neck_aim_tgt,
                        s_coil_deg=spine_coil_deg,
                        limits_map=limits_map,
                        root_translation_delta=root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                        forward_sign_z=float(rig_asm.morphology.forward_sign_z),
                    )
            if com_g > 1e-4:
                cur_poses, root_delta = solve_com_balance_and_counterweight(
                    program,
                    rest_scene,
                    rig_asm,
                    cur_poses,
                    root_delta,
                    strength=com_g,
                    in_place=bool(getattr(clip, "in_place", False)),
                    grounded=True,
                    limits_map=limits_map,
                    pose_position_deltas=frame_pos_deltas or None,
                    pose_scale_factors=frame_scale_factors or None,
                )
                root_deltas_list[-1] = root_delta
            if grasp_g > 1e-4 or auto_rig:
                pin_stance_map: dict[str, bool] | None = None
                if has_ground_pin:
                    from synccraft.kinematics.whole_body_solver import limb_matches_pin_spec

                    grounded_las = [la for la in rig_asm.limb_assemblies if not la.is_free]
                    pin_stance_map = {
                        la.role: limb_matches_pin_spec(la, lb, pin_spec)
                        for la, lb in zip(grounded_las, rig_asm.morphology.limbs)
                    }
                cur_poses = solve_foot_roll_and_digit_articulation(
                    rig_asm,
                    cur_poses,
                    limb_in_stance=pin_stance_map,
                    foot_roll_strength=0.0,
                    digit_strength=0.45 if auto_rig else 0.0,
                    grasp_amount=grasp_g,
                    aerial_tuck=False,
                    limits_map=limits_map,
                )
            if antag_g > 1e-4:
                frame_scale_factors = solve_full_body_antagonist_muscle_bulge(
                    rig_asm,
                    cur_poses,
                    frame_scale_factors,
                    strength=antag_g,
                    preserve_grounded_limb_length=bool(clip.ik_drives or has_ground_pin),
                )
                scale_factors_list[-1] = frame_scale_factors
        if clip_stab > 1e-4 and neck_aim_tgt is None:
            from synccraft.kinematics.bio_constraints import solve_vestibulo_ocular_stabilization
            from synccraft.kinematics.morphology import analyze_morphology

            if program.kinematics is not None and program.kinematics.look_chains:
                for lkc in program.kinematics.look_chains:
                    stab_bones = lkc.bones[-2:] if len(lkc.bones) >= 4 else lkc.bones
                    eff_stab = min(0.38, clip_stab) if len(lkc.bones) >= 4 else clip_stab
                    cur_poses = solve_vestibulo_ocular_stabilization(
                        program,
                        rest_scene,
                        stab_bones,
                        cur_poses,
                        strength=eff_stab,
                        weight_decay=float(lkc.weight_decay),
                        limits_map=limits_map,
                        root_translation_delta=root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                    )
            else:
                morph_c = analyze_morphology(program, rest_scene)
                for n_ch in (morph_c.neck_chains or ((morph_c.neck,) if morph_c.neck else ())):
                    stab_bones = n_ch[-2:] if len(n_ch) >= 4 else n_ch
                    eff_stab = min(0.38, clip_stab) if len(n_ch) >= 4 else clip_stab
                    cur_poses = solve_vestibulo_ocular_stabilization(
                        program,
                        rest_scene,
                        stab_bones,
                        cur_poses,
                        strength=eff_stab,
                        limits_map=limits_map,
                        root_translation_delta=root_delta,
                        pose_position_deltas=frame_pos_deltas or None,
                        pose_scale_factors=frame_scale_factors or None,
                    )
        eff_positions = {}
        eff_stance = {}
        eff_step_ids = {}

        # 4. Closed-loop world-anchored foot_cycle N-bone IK drives (< 0.02u stance slip)
        for ikd in clip.ik_drives:
            chain = ik_chain_map[ikd.chain]
            eff_rest_w = ik_rest_world[ikd.chain]
            stride_vec = np.array([0.0, 0.0, float(ikd.stride)], dtype=np.float64)
            phase_off = (ikd.phase_deg / 360.0) % 1.0

            if abs(ikd.stride) <= 1e-6 and abs(ikd.lift) <= 1e-6:
                tgt_w = eff_rest_w.copy()
                in_stance = True
                step_id = 0
            else:
                raw_phase = phi + phase_off
                wrap_k = math.floor(raw_phase)
                u = raw_phase - wrap_k
                stance_anchor = eff_rest_w + (wrap_k - phase_off + 0.5 * duty_factor) * stride_vec
                if u < duty_factor:
                    tgt_w = stance_anchor.copy()
                    in_stance = True
                    step_id = int(wrap_k)
                else:
                    s = (u - duty_factor) / max(1e-6, 1.0 - duty_factor)
                    sigma = 0.5 - 0.5 * math.cos(math.pi * s)
                    lift_y = ikd.lift * (math.sin(math.pi * s) ** 2)
                    tgt_w = stance_anchor + sigma * stride_vec + np.array([0.0, lift_y, 0.0], dtype=np.float64)
                    in_stance = False
                    step_id = -1

            ik_sol = _solve_ik_chain_with_root_delta(
                program,
                chain,
                (float(tgt_w[0]), float(tgt_w[1]), float(tgt_w[2])),
                base_pose_deltas=cur_poses,
                root_translation_delta=root_delta,
                validate=False,
                _limits_cache=limits_map,
            )
            cur_poses = ik_sol.pose_euler_deltas
            eff_positions[ikd.chain] = ik_sol.achieved_world
            eff_stance[ikd.chain] = in_stance
            eff_step_ids[ikd.chain] = step_id

        # 4b. Standing Limb IK Pinning (`ground_pin_limbs: true` or `[hind_l, hind_r]`)
        if has_ground_pin and rig_asm is not None and not all_springs:
            from synccraft.kinematics.whole_body_solver import solve_standing_limb_ik_pinning

            cur_poses, pin_pos, pin_st, pin_sid = solve_standing_limb_ik_pinning(
                program,
                rest_scene,
                rig_asm,
                cur_poses,
                pin_spec=pin_spec,
                root_translation_delta=root_delta,
                pose_position_deltas=frame_pos_deltas or None,
                pose_scale_factors=frame_scale_factors or None,
                limits_map=limits_map,
            )
            eff_positions.update(pin_pos)
            eff_stance.update(pin_st)
            eff_step_ids.update(pin_sid)

        if _pre_spring_cache is not None:
            _pre_spring_cache[phi_key] = (
                t_s,
                root_delta,
                acts,
                frame_pos_deltas,
                frame_scale_factors,
                cur_poses,
                ms_res,
                eff_positions,
                eff_stance,
                eff_step_ids,
            )
        pre_spring_poses_list.append(cur_poses)
        pre_spring_ms_res_list.append(ms_res)
        eff_pos_list.append(eff_positions)
        eff_stance_list.append(eff_stance)
        eff_step_list.append(eff_step_ids)

    # 5. Secondary 2-parameter damped-spring follow-through (when declared)
    if all_springs:
        final_poses_list = _integrate_spring_follow_through(
            program,
            rest_scene,
            all_springs,
            times_list,
            clip.loop,
            clip.duration_s,
            pre_spring_poses_list,
            root_deltas_list,
            limits_map,
        )
    else:
        final_poses_list = pre_spring_poses_list

    samples: list[BehaviorFrameSample] = []
    for f_idx in range(n_f):
        cur_poses = final_poses_list[f_idx]
        root_delta = root_deltas_list[f_idx]
        acts = acts_list[f_idx]
        ms_res = pre_spring_ms_res_list[f_idx]
        f_pos = pos_deltas_list[f_idx]
        f_scl = scale_factors_list[f_idx]

        if has_ground_pin and rig_asm is not None and all_springs:
            from synccraft.kinematics.whole_body_solver import solve_standing_limb_ik_pinning

            cur_poses, pin_pos, pin_st, pin_sid = solve_standing_limb_ik_pinning(
                program,
                rest_scene,
                rig_asm,
                cur_poses,
                pin_spec=pin_spec,
                root_translation_delta=root_delta,
                pose_position_deltas=f_pos or None,
                pose_scale_factors=f_scl or None,
                limits_map=limits_map,
            )
            eff_pos_list[f_idx].update(pin_pos)
            eff_stance_list[f_idx].update(pin_st)
            eff_step_list[f_idx].update(pin_sid)

        if clip.ik_drives or all_springs or float(getattr(clip, "head_stabilize", 0.0)) > 1e-4 or rig_asm is not None:
            final_cache_key = ("__final__", round(phases_list[f_idx], 8))
            if not all_springs and _pre_spring_cache is not None and final_cache_key in _pre_spring_cache:
                frame_inflate_deltas, final_scene = _pre_spring_cache[final_cache_key]
            else:
                kin_scene = compile_posed_geometry(
                    program,
                    cur_poses,
                    root_translation_delta=root_delta,
                    pose_position_deltas=f_pos or None,
                    pose_scale_factors=f_scl or None,
                    validate=False,
                )
                if program.muscles:
                    frame_inflate_deltas, _ = _compute_muscle_bulge_and_telemetry(
                        rest_scene,
                        kin_scene,
                        program.muscles,
                        acts,
                        cur_poses,
                    )
                else:
                    frame_inflate_deltas = ms_res.box_inflate_deltas
                final_scene = (
                    compile_posed_geometry(
                        program,
                        cur_poses,
                        box_inflate_deltas=frame_inflate_deltas,
                        root_translation_delta=root_delta,
                        pose_position_deltas=f_pos or None,
                        pose_scale_factors=f_scl or None,
                        validate=False,
                    )
                    if frame_inflate_deltas
                    else kin_scene
                )
                if not all_springs and _pre_spring_cache is not None:
                    _pre_spring_cache[final_cache_key] = (frame_inflate_deltas, final_scene)
        else:
            frame_inflate_deltas = ms_res.box_inflate_deltas
            final_scene = ms_res.scene

        samples.append(
            BehaviorFrameSample(
                frame_index=f_idx,
                time_s=times_list[f_idx],
                phase=phases_list[f_idx],
                pose_euler_deltas=cur_poses,
                box_inflate_deltas=frame_inflate_deltas,
                root_translation_delta=root_delta,
                muscle_activations=acts,
                effector_world_positions=eff_pos_list[f_idx],
                effector_in_stance=eff_stance_list[f_idx],
                effector_stance_step_id=eff_step_list[f_idx],
                scene=final_scene,
                pose_position_deltas=f_pos,
                pose_scale_factors=f_scl,
            )
        )

    return samples


def _measure_max_interframe_steps(frames: list[BehaviorFrameSample], loop: bool) -> tuple[float, float]:
    """Return ``(max_step_deg, max_step_units)`` across consecutive sampled clip frames."""
    max_deg = 0.0
    max_units = 0.0
    for idx in range(1, len(frames)):
        prev_f = frames[idx - 1]
        cur_f = frames[idx]
        for bname in set(prev_f.pose_euler_deltas) | set(cur_f.pose_euler_deltas):
            v0 = np.asarray(prev_f.pose_euler_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            v1 = np.asarray(cur_f.pose_euler_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            max_deg = max(max_deg, float(np.max(np.abs(v1 - v0))))
        r0 = np.asarray(prev_f.root_translation_delta, dtype=np.float64)
        r1 = np.asarray(cur_f.root_translation_delta, dtype=np.float64)
        max_units = max(max_units, float(np.linalg.norm(r1 - r0)))
        for bname in set(prev_f.pose_position_deltas) | set(cur_f.pose_position_deltas):
            p0 = np.asarray(prev_f.pose_position_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            p1 = np.asarray(cur_f.pose_position_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            max_units = max(max_units, float(np.linalg.norm(p1 - p0)))
    return max_deg, max_units


def _subdivide_clip_phases(
    frames: list[BehaviorFrameSample],
    *,
    max_step_deg: float,
    max_step_units: float,
    max_frames: int,
) -> list[float]:
    """Locally subdivide only intervals whose angular or linear step exceeds ``max_step_deg`` / ``max_step_units``."""
    budget_left = max(0, int(max_frames) - len(frames))
    if budget_left == 0 or len(frames) < 2:
        return [f.phase for f in frames]

    interval_needs: list[int] = []
    for idx in range(1, len(frames)):
        prev_f = frames[idx - 1]
        cur_f = frames[idx]
        step_deg = 0.0
        for bname in set(prev_f.pose_euler_deltas) | set(cur_f.pose_euler_deltas):
            v0 = np.asarray(prev_f.pose_euler_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            v1 = np.asarray(cur_f.pose_euler_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            step_deg = max(step_deg, float(np.max(np.abs(v1 - v0))))
        r0 = np.asarray(prev_f.root_translation_delta, dtype=np.float64)
        r1 = np.asarray(cur_f.root_translation_delta, dtype=np.float64)
        step_u = float(np.linalg.norm(r1 - r0))
        for bname in set(prev_f.pose_position_deltas) | set(cur_f.pose_position_deltas):
            p0 = np.asarray(prev_f.pose_position_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            p1 = np.asarray(cur_f.pose_position_deltas.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            step_u = max(step_u, float(np.linalg.norm(p1 - p0)))
        sub_k = max(
            1,
            int(
                math.ceil(
                    max(
                        step_deg / max(1e-6, 0.90 * max_step_deg),
                        step_u / max(1e-6, 0.90 * max_step_units),
                    )
                )
            ),
        )
        # Gradual local refinement (<= 2 inserts per interval per pass) focuses budget on swing peaks
        interval_needs.append(min(sub_k - 1, 2))

    total_extra = sum(interval_needs)
    if total_extra == 0:
        return [f.phase for f in frames]

    scale = min(1.0, budget_left / float(total_extra))
    extras = [int(math.floor(er * scale)) for er in interval_needs]
    rem_budget = max(0, budget_left - sum(extras))
    if rem_budget > 0:
        ranked = sorted(range(len(interval_needs)), key=lambda idx: interval_needs[idx], reverse=True)
        for idx in ranked:
            if rem_budget <= 0:
                break
            if interval_needs[idx] > extras[idx]:
                extras[idx] += 1
                rem_budget -= 1
    new_phases: list[float] = [frames[0].phase]
    for idx, extra in enumerate(extras):
        p0 = frames[idx].phase
        p1 = frames[idx + 1].phase
        sub_k = extra + 1
        for k in range(1, sub_k + 1):
            new_phases.append(p0 + (p1 - p0) * (k / float(sub_k)))
    return new_phases


def audit_behavior_clip(
    program: Program,
    clip_name: str,
    *,
    num_frames: int = 12,
    active_state: str | None = None,
    max_allowed_stance_slip: float = 0.02,
    collision_tolerance: float = 1e-7,
    adaptive_subdivision: bool | None = None,
    max_step_deg: float = 10.0,
    max_step_units: float = 0.5,
    max_frames: int = 144,
    audit_profile: str | None = None,
) -> dict[str, Any]:
    """Run unified adaptive discrete-step swept 15-axis OBB SAT + stance foot-skating audit on a @behavior clip."""
    from synccraft.audit.sync_audit import audit_scene

    clip_obj = next((c for c in (program.behavior.clips if program.behavior else ()) if c.name == clip_name), None)
    duration_s = float(clip_obj.duration_s) if clip_obj is not None else 0.0
    is_loop = bool(clip_obj.loop) if clip_obj is not None else True
    use_adaptive = True if adaptive_subdivision is None else bool(adaptive_subdivision)
    eff_max_frames = (
        max(int(max_frames), int(math.ceil(120.0 * duration_s)))
        if int(max_frames) == 144 and duration_s > 1.2
        else int(max_frames)
    )
    gait_cycles_in_clip = 1
    if clip_obj is not None and getattr(clip_obj, "gaits", ()) and int(max_frames) == 144:
        from synccraft.kinematics.gait_solver import analyze_morphology, compute_froude_kinematics

        _g_rest = compile_geometry(program, validate=False)
        _g_morph = analyze_morphology(program, _g_rest)
        _g_kin = compute_froude_kinematics(program, clip_obj.gaits[0], _g_morph, duration_s)
        gait_cycles_in_clip = max(1, int(_g_kin["cycles_in_clip"]))
        eff_max_frames = max(eff_max_frames, 72 * gait_cycles_in_clip)

    pre_spring_cache: dict[Any, Any] = {}
    if use_adaptive and clip_obj is not None:
        base_n = max(2, int(num_frames))
        if getattr(clip_obj, "gaits", ()):
            base_n = min(max(2, eff_max_frames - 1), max(base_n, 28 * gait_cycles_in_clip))
        seed_phases_set: set[float] = set(
            round(float(x), 8)
            for x in np.linspace(0.0, 1.0, base_n + (1 if is_loop else 0), endpoint=True, dtype=np.float64)
        )
        if duration_s > 1e-9:
            for kf_tr in getattr(clip_obj, "keyframes", ()):
                for t_k, _ in kf_tr.keys:
                    phi_k = float(t_k) / duration_s
                    if 0.0 <= phi_k <= 1.0:
                        seed_phases_set.add(round(phi_k, 8))
        seed_phases = sorted(seed_phases_set)
        if len(seed_phases) > eff_max_frames:
            seed_phases = [
                float(x) for x in np.linspace(0.0, 1.0, eff_max_frames, endpoint=True, dtype=np.float64)
            ]
        frames = evaluate_behavior_clip(
            program,
            clip_name,
            active_state=active_state,
            sample_phases=seed_phases,
            _pre_spring_cache=pre_spring_cache,
        )
    else:
        frames = evaluate_behavior_clip(
            program,
            clip_name,
            num_frames=num_frames,
            active_state=active_state,
            _pre_spring_cache=pre_spring_cache,
        )
    measured_max_deg, measured_max_units = _measure_max_interframe_steps(frames, is_loop)
    if use_adaptive:
        for _ in range(9):
            if len(frames) >= eff_max_frames or (measured_max_deg <= max_step_deg and measured_max_units <= max_step_units):
                break
            refined_phases = _subdivide_clip_phases(
                frames,
                max_step_deg=max_step_deg,
                max_step_units=max_step_units,
                max_frames=eff_max_frames,
            )
            if len(refined_phases) <= len(frames):
                break
            frames = evaluate_behavior_clip(
                program,
                clip_name,
                active_state=active_state,
                sample_phases=refined_phases,
                _validate=False,
                _pre_spring_cache=pre_spring_cache,
            )
            measured_max_deg, measured_max_units = _measure_max_interframe_steps(frames, is_loop)
    colliding_frames: list[dict[str, Any]] = []
    total_collision_events = 0

    eff_profile = audit_profile
    if eff_profile is None:
        meta_profile = str(getattr(getattr(program, "meta", None), "audit_profile", "") or "")
        meta_style = str(getattr(getattr(program, "meta", None), "style", "") or "")
        has_sg = bool(frames) and any(bool(getattr(b, "sculpt_group", ())) for b in frames[0].scene.boxes)
        if meta_profile in ("sculpted", "authored"):
            eff_profile = meta_profile
        elif "sculpted" in meta_style or has_sg:
            eff_profile = "sculpted"
        else:
            eff_profile = meta_profile or "strict"

    for fr in frames:
        aud = audit_scene(
            fr.scene,
            collision_tolerance=collision_tolerance,
            enforce_physical_contact=False,
            audit_profile=eff_profile,
        )
        if aud["collision_pairs"]:
            total_collision_events += len(aud["collision_pairs"])
            colliding_frames.append(
                {
                    "frame_index": fr.frame_index,
                    "phase": round(fr.phase, 4),
                    "collision_count": len(aud["collision_pairs"]),
                    "pairs": [c["pair"] for c in aud["collision_pairs"]],
                }
            )

    max_stance_slip = 0.0
    per_chain_slip: dict[str, float] = {}
    for idx in range(1, len(frames)):
        prev_f = frames[idx - 1]
        cur_f = frames[idx]
        for cname, in_st in cur_f.effector_in_stance.items():
            cur_sid = cur_f.effector_stance_step_id.get(cname, -1)
            prev_sid = prev_f.effector_stance_step_id.get(cname, -2)
            if in_st and prev_f.effector_in_stance.get(cname, False) and cur_sid >= 0 and cur_sid == prev_sid:
                p0 = np.asarray(prev_f.effector_world_positions[cname], dtype=np.float64)
                p1 = np.asarray(cur_f.effector_world_positions[cname], dtype=np.float64)
                horiz_slip = float(math.hypot(p1[0] - p0[0], p1[2] - p0[2]))
                per_chain_slip[cname] = max(per_chain_slip.get(cname, 0.0), horiz_slip)
                max_stance_slip = max(max_stance_slip, horiz_slip)

    bone_excursion_deg: dict[str, float] = {}
    for idx in range(1, len(frames)):
        prev_p = frames[idx - 1].pose_euler_deltas
        cur_p = frames[idx].pose_euler_deltas
        for bname in set(prev_p) | set(cur_p):
            v0 = np.asarray(prev_p.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            v1 = np.asarray(cur_p.get(bname, (0.0, 0.0, 0.0)), dtype=np.float64)
            bone_excursion_deg[bname] = bone_excursion_deg.get(bname, 0.0) + float(np.linalg.norm(v1 - v0))

    stance_ok = max_stance_slip <= max_allowed_stance_slip
    step_bound_satisfied = (
        True
        if not use_adaptive
        else (measured_max_deg <= max_step_deg + 1e-4 and measured_max_units <= max_step_units + 1e-4)
    )
    passed = (len(colliding_frames) == 0) and stance_ok and step_bound_satisfied
    result_dict: dict[str, Any] = {
        "clip_name": clip_name,
        "active_state": active_state,
        "motion_audit_method": "adaptive_discrete_step" if use_adaptive else "fixed_uniform_sampling",
        "time_range_s": [0.0, round(duration_s, 4)],
        "num_frames": len(frames),
        "max_frames_cap": eff_max_frames,
        "measured_max_step_deg": round(measured_max_deg, 4),
        "measured_max_step_units": round(measured_max_units, 5),
        "step_bound_satisfied": step_bound_satisfied,
        "passed": passed,
        "swept_collision_free": len(colliding_frames) == 0,
        "swept_collision_frame_count": len(colliding_frames),
        "total_collision_events": total_collision_events,
        "colliding_frames": colliding_frames,
        "stance_slip_passed": stance_ok,
        "max_stance_slip_units": round(max_stance_slip, 6),
        "max_allowed_stance_slip": max_allowed_stance_slip,
        "per_chain_stance_slip": {k: round(v, 6) for k, v in sorted(per_chain_slip.items())},
        "bone_angular_excursion_deg": {k: round(v, 3) for k, v in sorted(bone_excursion_deg.items())},
    }
    if clip_obj is not None and getattr(clip_obj, "flights", ()):
        from synccraft.kinematics.flight_solver import audit_flight_clip_contracts

        flight_audit = audit_flight_clip_contracts(
            program,
            clip_obj,
            frames,
            precomputed_collision_events=total_collision_events,
        )
        result_dict["flight_audit"] = flight_audit
        result_dict["root_motion"] = flight_audit.get("root_motion")
        if not flight_audit["passed"]:
            result_dict["passed"] = False
    if clip_obj is not None and getattr(clip_obj, "gaits", ()):
        from synccraft.kinematics.gait_solver import audit_gait_clip_contracts

        gait_audit = audit_gait_clip_contracts(
            program,
            clip_obj,
            frames,
            precomputed_collision_events=total_collision_events,
        )
        result_dict["gait_audit"] = gait_audit
        result_dict["root_motion"] = gait_audit.get("root_motion")
        if not gait_audit["passed"]:
            result_dict["passed"] = False
    return result_dict
