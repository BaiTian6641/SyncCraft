"""Biomechanical Musculoskeletal, Tendon-Coupled & Closed-Loop Behavior Engine.

Implements ADR 0006 reduced-order musculoskeletal control and multi-phase motion
synthesis atop SyncCraft's exact right-handed SE(3) chain (`compile_posed_geometry`):

1. Hill-type agonist-antagonist muscle equilibrium (`@muscles`):
   - Activation-modulated joint stiffness and net moment-arm excursion
   - Biarticular passive tendon pantograph couplings (`TendonCoupling`)
   - Exact 3D SE(3) origin-to-insertion fiber strain & isochoric `belly_box`
     `inflate` bulging (preserving integer Box-UV dimensions)
2. Multi-bone distributed gaze/aiming chains (`LookChainDecl`) and closed-loop
   2-bone damped-Levenberg-Marquardt limb IK (`IKChainDecl`)
3. Analytical world-anchored gait & behavior clips (`@behavior`):
   - C^1-continuous stance-locked foot trajectories with < 0.02u horizontal slip
   - Multi-segment travelling spine/tail waves, asymmetric wing flaps, and state
     modulation (`BehaviorStateDecl`)
   - Multi-phase swept 15-axis OBB SAT collision & stance-slip auditing

Fidelity label: ``reduced_order_model`` (quasi-static Hill equilibrium + analytical
kinematics; not a claim of 3D volumetric finite-element soft-tissue elasticity).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from synccraft.dsl.ast_nodes import (
    Bone,
    ClipDecl,
    IKChainDecl,
    LookChainDecl,
    MuscleDecl,
    Program,
    Vec3,
)
from synccraft.kinematics.se3_chain import (
    Scene,
    _clamp_delta_to_joint_limits,
    compile_geometry,
    compile_posed_geometry,
)

if TYPE_CHECKING:
    from synccraft.data_gen.ik_motion_templates import TwoBoneIKResult


@dataclass(frozen=True)
class MuscleTelemetry:
    """Per-muscle biomechanical state evaluated in exact 3D SE(3)."""

    name: str
    activation: float
    rest_length: float
    posed_length: float
    contraction_ratio: float
    strain: float
    applied_bulge_inflate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "activation": round(self.activation, 5),
            "rest_length": round(self.rest_length, 5),
            "posed_length": round(self.posed_length, 5),
            "contraction_ratio": round(self.contraction_ratio, 5),
            "strain": round(self.strain, 5),
            "applied_bulge_inflate": round(self.applied_bulge_inflate, 5),
        }


@dataclass(frozen=True)
class MusculoskeletalPoseResult:
    """Evaluated joint deltas, belly-box inflations, posed Scene, and telemetry."""

    pose_euler_deltas: dict[str, Vec3]
    box_inflate_deltas: dict[str, float]
    joint_stiffness: dict[str, float]
    muscle_telemetry: tuple[MuscleTelemetry, ...]
    scene: Scene

    def to_dict(self) -> dict[str, Any]:
        return {
            "fidelity_label": "reduced_order_model",
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


def _collect_expanded_joint_limits(program: Program) -> dict[str, tuple[Any, bool]]:
    """Map expanded bone name -> (joint_limits, mirrored) matching compile_geometry."""
    out: dict[str, tuple[Any, bool]] = {}

    def _walk(
        bone: Bone,
        *,
        mirrored: bool = False,
        prefix: str | None = None,
        renamed_root: str | None = None,
    ) -> None:
        name = renamed_root or (f"{prefix}__{bone.name}" if prefix else bone.name)
        if bone.joint_limits is not None:
            out[name] = (bone.joint_limits, mirrored)
        for item in bone.items:
            if isinstance(item, Bone):
                _walk(item, mirrored=mirrored, prefix=prefix)
                if item.mirror_name:
                    m_prefix = f"{prefix}__{item.mirror_name}" if prefix else item.mirror_name
                    _walk(
                        item,
                        mirrored=not mirrored,
                        prefix=m_prefix,
                        renamed_root=m_prefix,
                    )

    _walk(program.root)
    return out


def _clamp_pose_vec(
    bone_name: str,
    vec: np.ndarray,
    limits_map: dict[str, tuple[Any, bool]],
) -> np.ndarray:
    if bone_name in limits_map:
        jlim, is_mir = limits_map[bone_name]
        return _clamp_delta_to_joint_limits(vec, jlim, mirrored=is_mir)
    return vec


def _compute_muscle_bulge_and_telemetry(
    rest_scene: Scene,
    kin_scene: Scene,
    muscles: tuple[MuscleDecl, ...],
    act_map: dict[str, float],
) -> tuple[dict[str, float], tuple[MuscleTelemetry, ...]]:
    """Measure exact 3D SE(3) origin-to-insertion fiber shortening and belly_box bulge_inflate."""
    inflate_deltas: dict[str, float] = {}
    telemetry: list[MuscleTelemetry] = []
    for m in muscles:
        a_val = float(np.clip(act_map.get(m.name, 0.0), 0.0, 1.0))
        p_orig_0 = _bone_anchor_point_world(rest_scene, m.origin_bone)
        p_ins_0 = _bone_anchor_point_world(rest_scene, m.insertion_bone, preferred_box=m.belly_box)
        rest_len = float(np.linalg.norm(p_ins_0 - p_orig_0))

        p_orig_q = _bone_anchor_point_world(kin_scene, m.origin_bone)
        p_ins_q = _bone_anchor_point_world(kin_scene, m.insertion_bone, preferred_box=m.belly_box)
        posed_len = float(np.linalg.norm(p_ins_q - p_orig_q))

        ratio = posed_len / max(1e-4, rest_len) if rest_len > 1e-5 else 1.0
        strain = ratio - 1.0
        shortening_gain = float(np.clip(0.68 + 0.32 * np.clip((1.0 - ratio) / 0.25, -0.5, 1.0), 0.35, 1.0))
        bulge = float(a_val * m.bulge_inflate * shortening_gain) if m.belly_box is not None else 0.0
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
            )
        )
    return inflate_deltas, tuple(telemetry)


def evaluate_muscles_at_activations(
    program: Program,
    activations: dict[str, float] | None = None,
    *,
    base_pose_deltas: dict[str, Vec3] | None = None,
    root_translation_delta: Vec3 | None = None,
) -> MusculoskeletalPoseResult:
    """Compute Hill-type agonist-antagonist joint equilibrium, tendon couplings, and belly bulges."""
    rest_scene = compile_geometry(program)
    limits_map = _collect_expanded_joint_limits(program)
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

    # 1. Per-axis quasi-static Hill agonist-antagonist joint equilibrium
    for (bone_name, ax_char), m_group in by_insertion_axis.items():
        ax_i = axis_map[ax_char]
        max_a = max(act_map[m.name] for m in m_group)
        k_passive = max(0.0, 1.0 - max_a)
        k_active = sum(act_map[m.name] for m in m_group)
        k_total = max(1e-8, k_passive + k_active)
        net_drive_ax = sum(act_map[m.name] * float(m.moment_arm_deg) for m in m_group)
        cur_vec = poses.setdefault(bone_name, np.zeros(3, dtype=np.float64))
        cur_vec[ax_i] += net_drive_ax / k_total
        stiffness_out[bone_name] = max(stiffness_out.get(bone_name, 0.0), float(k_total))

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

    # 3. Pre-bulge posed kinematics to measure exact 3D SE(3) fiber shortening
    kin_scene = compile_posed_geometry(
        program,
        pose_tuples,
        root_translation_delta=root_translation_delta,
    )

    inflate_deltas, telemetry = _compute_muscle_bulge_and_telemetry(
        rest_scene,
        kin_scene,
        muscles,
        act_map,
    )

    final_scene = (
        compile_posed_geometry(
            program,
            pose_tuples,
            box_inflate_deltas=inflate_deltas,
            root_translation_delta=root_translation_delta,
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
        # R_y(ry) @ [0,0,s] = [s*sin(ry), 0, s*cos(ry)] -> ry has sign of s * x
        # R_x(rx) @ [0,0,s] = [0, -s*sin(rx), s*cos(rx)] -> rx has sign of -s * y
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
        # R_x(rx) @ [0,s,0] = [0, s*cos(rx), s*sin(rx)] -> rx has sign of s * z
        # R_z(rz) @ [0,s,0] = [-s*sin(rz), s*cos(rz), 0] -> rz has sign of -s * x
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
        # fwd_ax == "x":
        # R_y(ry) @ [s,0,0] = [s*cos(ry), 0, -s*sin(ry)] -> ry has sign of -s * z
        # R_z(rz) @ [s,0,0] = [s*cos(rz), s*sin(rz), 0] -> rz has sign of s * y
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


def _solve_ik_with_root_delta(
    program: Program,
    upper_bone: str,
    lower_bone: str,
    effector_local: Vec3,
    target_world: Vec3,
    *,
    base_pose_deltas: dict[str, Vec3],
    root_translation_delta: Vec3,
    hinge_axis: str = "x",
) -> TwoBoneIKResult:
    """Solve 2-bone IK when the root bone has translated by root_translation_delta."""
    from synccraft.data_gen.ik_motion_templates import TwoBoneIKResult, solve_two_bone_ik

    shifted_target = (
        float(target_world[0] - root_translation_delta[0]),
        float(target_world[1] - root_translation_delta[1]),
        float(target_world[2] - root_translation_delta[2]),
    )
    res = solve_two_bone_ik(
        program,
        upper_bone,
        lower_bone,
        effector_local,
        shifted_target,
        base_pose_deltas=base_pose_deltas,
        hinge_axis=hinge_axis,
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
    return TwoBoneIKResult(
        pose_euler_deltas=res.pose_euler_deltas,
        achieved_world=ach,
        target_world=target_world,
        clamped_target_world=clamped,
        reachable=res.reachable,
        position_error=res.position_error,
        iterations=res.iterations,
    )


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
    return bias_deg


def evaluate_behavior_clip(
    program: Program,
    clip_name: str,
    *,
    num_frames: int = 12,
    active_state: str | None = None,
) -> list[BehaviorFrameSample]:
    """Synthesize exact posed, muscle-inflated, and stance-IK-locked frames for a @behavior clip."""
    if program.behavior is None:
        raise ValueError("Program has no @behavior section")
    clip_map: dict[str, ClipDecl] = {c.name: c for c in program.behavior.clips}
    if clip_name not in clip_map:
        raise KeyError(f"Unknown @behavior clip {clip_name!r}; available: {sorted(clip_map)}")
    clip = clip_map[clip_name]

    state_pose_bias: dict[str, Vec3] = {}
    state_muscle_tone: dict[str, float] = {}
    if active_state is not None:
        st_map = {s.name: s for s in program.behavior.states}
        if active_state not in st_map:
            raise KeyError(f"Unknown @behavior state {active_state!r}")
        st = st_map[active_state]
        state_pose_bias = dict(st.pose_biases)
        state_muscle_tone = dict(st.muscle_tones)

    rest_scene = compile_geometry(program)
    ik_chain_map: dict[str, IKChainDecl] = (
        {c.name: c for c in program.kinematics.ik_chains} if program.kinematics is not None else {}
    )
    ik_rest_world: dict[str, np.ndarray] = {}
    for cname, cdecl in ik_chain_map.items():
        lo_b = rest_scene.bone_by_name[cdecl.lower]
        ik_rest_world[cname] = lo_b.pivot_world + lo_b.rotation @ np.asarray(cdecl.effector, dtype=np.float64)

    # Forward gait progression along Z from declared ik_drives
    if clip.ik_drives:
        mean_stride_z = float(np.mean([d.stride for d in clip.ik_drives]))
    else:
        mean_stride_z = 0.0
    mean_stride_vec = np.array([0.0, 0.0, mean_stride_z], dtype=np.float64)
    duty_factor = 0.60

    n_f = max(2, int(num_frames))
    phases = np.linspace(0.0, 1.0, n_f, endpoint=not clip.loop, dtype=np.float64)
    axis_idx = {"x": 0, "y": 1, "z": 2}
    samples: list[BehaviorFrameSample] = []

    for f_idx, phi_raw in enumerate(phases):
        phi = float(phi_raw)
        t_s = phi * clip.duration_s

        heave_y = 0.0
        if clip.ik_drives and abs(mean_stride_z) > 1e-6:
            max_lift = max(d.lift for d in clip.ik_drives)
            heave_y = -0.10 * max_lift * (0.5 - 0.5 * math.cos(4.0 * math.pi * phi))
        root_delta_arr = phi * mean_stride_vec + np.array([0.0, heave_y, 0.0], dtype=np.float64)
        root_delta: Vec3 = (float(root_delta_arr[0]), float(root_delta_arr[1]), float(root_delta_arr[2]))

        # 1. Neural muscle excitation curves
        acts: dict[str, float] = dict(state_muscle_tone)
        for md in clip.muscle_drives:
            u_m = (t_s * md.freq_hz + md.phase_deg / 360.0) % 1.0
            val = md.baseline + (md.peak - md.baseline) * (0.5 - 0.5 * math.cos(2.0 * math.pi * u_m))
            acts[md.muscle] = float(np.clip(max(acts.get(md.muscle, 0.0), val), 0.0, 1.0))

        # 2. Analytic bone tracks + state pose_biases
        track_poses: dict[str, np.ndarray] = {
            b: np.asarray(v, dtype=np.float64).copy() for b, v in state_pose_bias.items()
        }
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
        base_pose_tuples: dict[str, Vec3] = {
            b: (float(v[0]), float(v[1]), float(v[2])) for b, v in track_poses.items()
        }

        # 3. Musculoskeletal + biarticular tendon solve
        ms_res = evaluate_muscles_at_activations(
            program,
            acts,
            base_pose_deltas=base_pose_tuples,
            root_translation_delta=root_delta,
        )
        cur_poses = dict(ms_res.pose_euler_deltas)
        eff_positions: dict[str, Vec3] = {}
        eff_stance: dict[str, bool] = {}
        eff_step_ids: dict[str, int] = {}

        # 4. Closed-loop world-anchored foot_cycle IK drives (< 0.02u stance slip)
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

            ik_sol = _solve_ik_with_root_delta(
                program,
                chain.upper,
                chain.lower,
                chain.effector,
                (float(tgt_w[0]), float(tgt_w[1]), float(tgt_w[2])),
                base_pose_deltas=cur_poses,
                root_translation_delta=root_delta,
                hinge_axis=chain.hinge_axis,
            )
            cur_poses = ik_sol.pose_euler_deltas
            eff_positions[ikd.chain] = ik_sol.achieved_world
            eff_stance[ikd.chain] = in_stance
            eff_step_ids[ikd.chain] = step_id

        if clip.ik_drives:
            ik_kin_scene = compile_posed_geometry(
                program,
                cur_poses,
                root_translation_delta=root_delta,
            )
            if program.muscles:
                frame_inflate_deltas, _ = _compute_muscle_bulge_and_telemetry(
                    rest_scene,
                    ik_kin_scene,
                    program.muscles,
                    acts,
                )
            else:
                frame_inflate_deltas = ms_res.box_inflate_deltas
            final_scene = (
                compile_posed_geometry(
                    program,
                    cur_poses,
                    box_inflate_deltas=frame_inflate_deltas,
                    root_translation_delta=root_delta,
                )
                if frame_inflate_deltas
                else ik_kin_scene
            )
        else:
            frame_inflate_deltas = ms_res.box_inflate_deltas
            final_scene = ms_res.scene

        samples.append(
            BehaviorFrameSample(
                frame_index=f_idx,
                time_s=t_s,
                phase=phi,
                pose_euler_deltas=cur_poses,
                box_inflate_deltas=frame_inflate_deltas,
                root_translation_delta=root_delta,
                muscle_activations=acts,
                effector_world_positions=eff_positions,
                effector_in_stance=eff_stance,
                effector_stance_step_id=eff_step_ids,
                scene=final_scene,
            )
        )

    return samples


def audit_behavior_clip(
    program: Program,
    clip_name: str,
    *,
    num_frames: int = 12,
    active_state: str | None = None,
    max_allowed_stance_slip: float = 0.02,
    collision_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Run multi-phase swept 15-axis OBB SAT + stance foot-skating audit on a @behavior clip."""
    from synccraft.audit.sync_audit import audit_scene

    frames = evaluate_behavior_clip(
        program,
        clip_name,
        num_frames=num_frames,
        active_state=active_state,
    )
    colliding_frames: list[dict[str, Any]] = []
    total_collision_events = 0

    for fr in frames:
        aud = audit_scene(
            fr.scene,
            collision_tolerance=collision_tolerance,
            enforce_physical_contact=False,
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
    passed = (len(colliding_frames) == 0) and stance_ok
    return {
        "clip_name": clip_name,
        "active_state": active_state,
        "num_frames": len(frames),
        "passed": passed,
        "swept_collision_free": len(colliding_frames) == 0,
        "total_collision_events": total_collision_events,
        "colliding_frames": colliding_frames,
        "stance_slip_passed": stance_ok,
        "max_stance_slip_units": round(max_stance_slip, 6),
        "max_allowed_stance_slip": max_allowed_stance_slip,
        "per_chain_stance_slip": {k: round(v, 6) for k, v in sorted(per_chain_slip.items())},
        "bone_angular_excursion_deg": {k: round(v, 3) for k, v in sorted(bone_excursion_deg.items())},
    }
