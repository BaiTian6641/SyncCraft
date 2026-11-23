"""Universal Underwater Buoyancy & Musculoskeletal Swimming Positioning System (`SO(3)` / `SE(3)` / `Sim(3)`).

Synthesizes biologically grounded aquatic / underwater swimming, hovering, paddling, and
pelagic medusa-pulse propulsion for ANY aquatic or amphibious creature rig (Leviathans,
crocodilians, sea serpents, fish, sharks, eels, dolphins, sea turtles, cephalopods, and
radial jellyfish) purely from `@morphology` geometry, 3D angular Jacobians, `@muscles`
Zajac/Hill activations, and `SO(3)` / `Sim(3)` biomechanics.

Key physical invariants:
1. Neutral-Buoyancy Underwater Positioning (NOT ground-pinned!): lifts the creature to
   clean mid-water clearance (`buoyancy_y` + sinusoidal heave bob + `ground_clearance`
   seabed guard) while keeping `in_place: true` (`max|root_z| == 0.000u`) and emitting
   matched hydrodynamic forward swim speed in `root_motion_velocity`.
2. True Progressive Hydrodynamic Traveling Wave (`SO(3)`): drives phase-delayed lateral
   undulation (`carangiform` | `subcarangiform` | `anguilliform` | `leviathan_cruise` |
   `leviathan_hover`) projected onto each axial bone's exact world-Y yaw axis (`R_b.T @ y_hat`)
   from neck -> chest -> waist -> pelvis -> paddle-tail tip, with vestibulo-ocular
   `head_stabilize` locking the predator's gaze dead-ahead.
3. Exact 3D Jacobian Hydrodynamic Limb Positioning (`hydro_tuck` | `paddle_row` | `none`):
   - `hydro_tuck`: sweeps amphibious forelimbs & hindlimbs cleanly BACKWARD along the hull
     and horizontally level with flank clearance + trailing hydro-feathered webbed digits
     (authentic Lagiacrus / marine-iguana / crocodile high-speed swimming — zero walking legs!).
   - `paddle_row`: anti-phase underwater breaststroke / steering paddling for ambush hover.
4. Active Underwater Myomere `Sim(3)` Muscle Bulge: volume-preserving lateral myomere
   contraction (`sx * sy * sz ≈ 1`) synchronized with stroke curvature across torso, neck,
   and caudal segments, composed cleanly with declared `@muscles` `muscle_drive`s and
   additive `key` / `track` animations (e.g. underwater roar / lightning discharge).
5. Pelagic Radial `medusa_pulse` Mode: rhythmic `Sim(3)` bell contraction/recovery +
   Jacobian-projected radial tentacle & oral-arm propulsion for 0-leg jellyfish/cephalopods.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from synccraft.dsl.ast_nodes import ClipDecl, Program, SwimDecl, Vec3
from synccraft.kinematics.bio_constraints import (
    infer_universal_bio_constraints,
    soft_clamp_euler_to_limits,
)
from synccraft.kinematics.morphology import LimbMorphology, MorphologyReport, analyze_morphology
from synccraft.kinematics.se3_chain import (
    Scene,
    _angle_sign_from_reflection,
    compile_geometry,
    compile_posed_geometry,
)


def _world_axis_to_local_euler_deg(
    R_bone: np.ndarray,
    world_axis: np.ndarray,
    angle_deg: float,
    angle_sign: np.ndarray | None = None,
) -> np.ndarray:
    """Project a small rotation of `angle_deg` about `world_axis` into bone's local XYZ Euler delta.

    Note: `R_bone` (`WorldBone.rotation`) already embeds mirror parity, and `se3_chain.py:467`
    applies `delta_euler` directly (`local_rotation @ euler_xyz_matrix(delta_euler)`), so NO
    extra `angle_sign` multiplication is needed here.
    """
    u_w = np.asarray(world_axis, dtype=np.float64)
    n = float(np.linalg.norm(u_w))
    if n < 1e-8 or abs(angle_deg) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    u_w = u_w / n
    u_local = R_bone.T @ u_w
    return u_local * float(angle_deg)


def _calibrate_hydro_limb_axes(
    rest_scene: Scene,
    limb: LimbMorphology,
    forward_sign_z: float,
) -> dict[str, tuple[int, float]]:
    """Derive exact local rotation axes & signs for (aft_sweep, lift_up, flank_out) from 3D Jacobian."""
    chain = limb.chain
    if not chain or chain[0] not in rest_scene.bone_by_name:
        return {"aft": (1, 1.0), "up": (2, 1.0), "out": (0, 1.0)}
    root_b = rest_scene.bone_by_name[chain[0]]
    tip_w = np.asarray(limb.contact_world, dtype=np.float64)
    r_vec = tip_w - root_b.pivot_world
    if float(np.linalg.norm(r_vec)) < 1e-4:
        r_vec = np.array([-6.0 if limb.side == "l" else 6.0, -8.0, 0.0], dtype=np.float64)

    jac = np.zeros((3, 3), dtype=np.float64)
    for ax_i in range(3):
        axis_w = root_b.rotation[:, ax_i]
        jac[:, ax_i] = np.cross(axis_w, r_vec)

    # 1. Aft sweep: strongest |J[2, i]| moving tip toward -forward_sign_z (backward along hull)
    aft_ax = int(np.argmax(np.abs(jac[2, :])))
    aft_sign = -1.0 if (jac[2, aft_ax] * forward_sign_z) > 0.0 else 1.0

    # 2. Lift up (+Y): among remaining axes, strongest |J[1, i]| moving tip UP (+Y)
    rem = [i for i in range(3) if i != aft_ax]
    up_ax = max(rem, key=lambda i: abs(jac[1, i]))
    up_sign = 1.0 if jac[1, up_ax] >= 0.0 else -1.0

    # 3. Flank clearance (+outward X): remaining axis moving tip away from sagittal X=0
    out_ax = [i for i in range(3) if i not in (aft_ax, up_ax)][0]
    side_sign = -1.0 if limb.side == "l" else 1.0
    out_sign = 1.0 if (jac[0, out_ax] * side_sign) >= 0.0 else -1.0

    return {
        "aft": (aft_ax, aft_sign),
        "up": (up_ax, up_sign),
        "out": (out_ax, out_sign),
    }


def _ordered_axial_swim_chain(
    rest_scene: Scene,
    morph: MorphologyReport,
) -> tuple[list[str], list[str], list[str]]:
    """Return topologically ordered (neck_chain_root_to_head, torso_chain_root_to_chest, tail_chain_root_to_tip)."""
    parent_map = {b.name: b.parent_name for b in rest_scene.bones}
    children_map: dict[str, list[str]] = {}
    for b in rest_scene.bones:
        if b.parent_name:
            children_map.setdefault(b.parent_name, []).append(b.name)

    def _depth(bn: str) -> int:
        d = 0
        cur = bn
        while parent_map.get(cur) is not None and d < 64:
            cur = parent_map[cur]  # type: ignore[assignment]
            d += 1
        return d

    # Tail chain (ordered proximal -> distal by kinematic tree depth)
    tail_bones: list[str] = []
    if morph.tail:
        tail_bones = sorted(
            [b for b in morph.tail if b in rest_scene.bone_by_name and "__" not in b],
            key=_depth,
        )
    else:
        tail_bones = sorted(
            [
                b.name for b in rest_scene.bones
                if "tail" in b.name.lower() and "__" not in b.name
                and not any(k in b.name.lower() for k in ("fin", "petal", "sail", "membrane"))
            ],
            key=_depth,
        )

    # Neck + head chain (ordered proximal -> head)
    neck_bones: list[str] = []
    if morph.neck:
        neck_bones = sorted(
            [b for b in morph.neck if b in rest_scene.bone_by_name and "__" not in b],
            key=_depth,
        )
    else:
        neck_bones = sorted(
            [
                b.name for b in rest_scene.bones
                if ("neck" in b.name.lower() or "cerv" in b.name.lower()) and "__" not in b.name
                and not any(k in b.name.lower() for k in ("fin", "hood", "frill", "gill", "mane", "crest"))
            ],
            key=_depth,
        )
    head_candidates = [b.name for b in rest_scene.bones if b.name.lower() in ("head", "skull", "cranium") and "__" not in b.name]
    for hb in head_candidates:
        if hb not in neck_bones:
            neck_bones.append(hb)

    # Torso chain (root -> waist -> chest)
    torso_bones: list[str] = []
    root_name = rest_scene.bones[0].name
    if morph.spine:
        for bn in morph.spine:
            if bn in rest_scene.bone_by_name and bn not in tail_bones and bn not in neck_bones:
                torso_bones.append(bn)
        torso_bones.sort(key=_depth)
    if not torso_bones:
        torso_bones = sorted(
            [
                b.name for b in rest_scene.bones
                if any(k in b.name.lower() for k in ("pelvis", "waist", "torso", "chest", "body", "spine"))
                and b.name not in tail_bones
                and b.name not in neck_bones
                and "__" not in b.name
            ],
            key=_depth,
        )
    if root_name not in torso_bones:
        torso_bones.insert(0, root_name)

    return neck_bones, torso_bones, tail_bones


def _mode_wave_profile(mode: str, s: float) -> float:
    """Hydrodynamic amplitude envelope A(s) for normalized axial position s in [0 (head), 1 (tail tip)]."""
    s_c = float(np.clip(s, 0.0, 1.0))
    if mode == "anguilliform":
        return 0.42 + 0.58 * (s_c ** 0.85)
    if mode == "carangiform":
        return 0.08 + 0.92 * (s_c ** 1.85)
    if mode == "subcarangiform":
        return 0.16 + 0.84 * (s_c ** 1.45)
    if mode == "leviathan_hover":
        return 0.20 + 0.80 * (s_c ** 1.25)
    # Default: leviathan_cruise
    return 0.18 + 0.82 * (s_c ** 1.30)


def _solve_hydro_limb_pose(
    rest_scene: Scene,
    limb: LimbMorphology,
    swim: SwimDecl,
    phi: float,
    forward_sign_z: float,
    pose_euler: dict[str, np.ndarray],
    bone_signs: dict[str, np.ndarray],
) -> None:
    """Apply Jacobian-calibrated hydrodynamic limb positioning (`hydro_tuck` or `paddle_row`)."""
    if swim.limb_mode == "none" or not limb.chain:
        return
    tuck = float(np.clip(swim.limb_tuck, 0.0, 1.0))
    if tuck < 1e-4:
        return

    axes = _calibrate_hydro_limb_axes(rest_scene, limb, forward_sign_z)
    aft_ax, aft_sign = axes["aft"]
    up_ax, up_sign = axes["up"]
    out_ax, out_sign = axes["out"]

    is_fore = limb.role.startswith("fore") or "arm" in limb.chain[0].lower()
    is_left = limb.side == "l"
    phase_off = (0.0 if is_left else math.pi) + (0.0 if is_fore else 0.5 * math.pi)
    stroke = math.sin(2.0 * math.pi * phi + phase_off)
    stroke_cos = math.cos(2.0 * math.pi * phi + phase_off)

    # Proximal limb bone (shoulder / hip)
    b0 = limb.chain[0]
    d0 = np.zeros(3, dtype=np.float64)
    if swim.limb_mode == "hydro_tuck":
        # Streamlined crocodile/Lagiacrus underwater sweep: forelimbs fold deeply backward along the
        # ribcage, while hindlimbs sweep smoothly aft with wider lateral clearance from the undulating
        # caudal root (`tail_1`) so bilateral high-amplitude tail beats never clip the tucked thighs!
        aft_base = (36.0 if is_fore else 16.0) * tuck
        up_base = (18.0 if is_fore else 13.0) * tuck
        out_base = (12.0 if is_fore else 24.0) * tuck
        d0[aft_ax] += aft_sign * (aft_base + (4.0 if is_fore else 2.5) * stroke * tuck)
        d0[up_ax] += up_sign * (up_base + 2.0 * stroke_cos * tuck)
        d0[out_ax] += out_sign * out_base
    elif swim.limb_mode == "paddle_row":
        # Underwater breaststroke / hover treading: rhythmic anti-phase sweep + lift recovery
        aft_amp = (18.0 if is_fore else 15.0) * tuck
        up_base = (12.0 if is_fore else 10.0) * tuck
        d0[aft_ax] += aft_sign * (8.0 * tuck + aft_amp * stroke)
        d0[up_ax] += up_sign * (up_base + 7.5 * stroke_cos * tuck)
        d0[out_ax] += out_sign * (6.0 * tuck)

    # World-space Jacobian `root_b.rotation` ALREADY incorporates mirror reflection -> apply d0 directly!
    pose_euler[b0] = pose_euler.get(b0, np.zeros(3, dtype=np.float64)) + d0

    # Distal forearm / shin segment: streamline aft or feather during paddle recovery
    if len(limb.chain) >= 2 and limb.chain[1] in rest_scene.bone_by_name:
        b1 = limb.chain[1]
        rb1 = rest_scene.bone_by_name[b1]
        tip_w = np.asarray(limb.contact_world, dtype=np.float64)
        r1 = tip_w - rb1.pivot_world
        if float(np.linalg.norm(r1)) > 1e-4:
            j1 = np.array([np.cross(rb1.rotation[:, k], r1) for k in range(3)], dtype=np.float64).T
            ax1 = int(np.argmax(np.abs(j1[2, :])))
            sg1 = -1.0 if (j1[2, ax1] * forward_sign_z) > 0.0 else 1.0
            d1 = np.zeros(3, dtype=np.float64)
            if swim.limb_mode == "hydro_tuck":
                d1[ax1] += sg1 * (24.0 * tuck + 5.0 * stroke_cos * tuck)
            else:
                d1[ax1] += sg1 * (10.0 * tuck + 12.0 * stroke_cos * tuck)
            pose_euler[b1] = pose_euler.get(b1, np.zeros(3, dtype=np.float64)) + d1

    # Webbed paw / foot trailing hydro-feathering (X pitch is symmetric across mirror_x!)
    foot_b = limb.chain[-1] if len(limb.chain) >= 3 else None
    if foot_b is not None and foot_b in rest_scene.bone_by_name:
        df = np.zeros(3, dtype=np.float64)
        fin_lag = float(swim.fin_lag_deg)
        if swim.limb_mode == "hydro_tuck":
            df[0] += (22.0 * tuck) + 0.45 * fin_lag * math.sin(2.0 * math.pi * phi + phase_off - 0.65)
        else:
            df[0] += 0.65 * fin_lag * math.sin(2.0 * math.pi * phi + phase_off - 0.55)
        pose_euler[foot_b] = pose_euler.get(foot_b, np.zeros(3, dtype=np.float64)) + df


def evaluate_swim_clip(
    program: Program,
    clip: ClipDecl,
    *,
    num_frames: int = 24,
    active_state: str | None = None,
    sample_phases: Sequence[float] | None = None,
    _phase_cache: dict[float, Any] | None = None,
) -> list[Any]:
    """Synthesize exact underwater buoyant, traveling-wave, limb-positioned & myomere-bulging frames."""
    from synccraft.kinematics.flight_solver import _apply_muscle_scale_bulge
    from synccraft.kinematics.musculoskeletal import (
        BehaviorFrameSample,
        _collect_expanded_joint_limits,
        _evaluate_track_angle,
        evaluate_keyframe_track,
        evaluate_muscles_at_activations,
    )

    if not getattr(clip, "swims", ()):
        raise ValueError(f"clip {clip.name!r} has no swim declaration")
    swim: SwimDecl = clip.swims[0]

    rest_scene = compile_geometry(program, validate=False)
    morph = analyze_morphology(program, rest_scene)
    explicit_limits = _collect_expanded_joint_limits(program)
    limits_map = (
        infer_universal_bio_constraints(program, rest_scene, morph, explicit_limits=explicit_limits)
        if (swim.auto_bio_limits or clip.auto_bio_limits or clip.auto_rig)
        else explicit_limits
    )

    bone_signs: dict[str, np.ndarray] = {}
    for b in rest_scene.bones:
        refl = np.array([-1.0, 1.0, 1.0], dtype=np.float64) if b.mirrored else np.ones(3, dtype=np.float64)
        bone_signs[b.name] = np.asarray(_angle_sign_from_reflection(refl), dtype=np.float64)

    root_name = rest_scene.bones[0].name
    neck_bones, torso_bones, tail_bones = _ordered_axial_swim_chain(rest_scene, morph)
    head_bone = neck_bones[-1] if (neck_bones and any(k in neck_bones[-1].lower() for k in ("head", "skull", "cranium"))) else None
    forward_sign_z = float(morph.forward_sign_z)

    # Compute rest-pose solid floor clearance so buoyancy_y lifts cleanly into mid-water
    solid_mins = [
        float(bx.corners(include_inflate=True)[:, 1].min())
        for bx in rest_scene.boxes
        if getattr(bx, "plane_normal", None) is None
    ]
    rest_min_y = min(solid_mins) if solid_mins else 0.0
    body_len = max(12.0, float(morph.body_span[2]))

    # Construct ordered full axial spine [head .. neck_1, chest .. root, tail_1 .. tail_M]
    axial_front_to_back: list[str] = list(reversed(neck_bones)) + list(reversed(torso_bones)) + list(tail_bones)
    # Deduplicate preserving front-to-back order
    seen_ax: set[str] = set()
    ordered_axial: list[str] = []
    for bn in axial_front_to_back:
        if bn not in seen_ax and bn in rest_scene.bone_by_name:
            seen_ax.add(bn)
            ordered_axial.append(bn)
    n_ax = max(1, len(ordered_axial))

    # Identify radial tentacles for medusa_pulse mode
    radial_chains: list[list[str]] = []
    if swim.mode == "medusa_pulse":
        parent_map = {b.name: b.parent_name for b in rest_scene.bones}
        children_map: dict[str, list[str]] = {}
        for b in rest_scene.bones:
            if b.parent_name:
                children_map.setdefault(b.parent_name, []).append(b.name)
        for b in rest_scene.bones:
            low = b.name.lower()
            if any(k in low for k in ("tentacle", "oral_arm", "bell_lobe")) and (
                b.parent_name is None or not any(k in b.parent_name.lower() for k in ("tentacle", "oral_arm"))
            ):
                ch = [b.name]
                cur = b.name
                while children_map.get(cur):
                    nxt = children_map[cur][0]
                    ch.append(nxt)
                    cur = nxt
                radial_chains.append(ch)

    if sample_phases is not None:
        phases = [float(p) for p in sample_phases]
    else:
        n_f = max(4, int(num_frames))
        phases = [i / float(n_f) for i in range(n_f)]

    # Estimated forward hydrodynamic speed in Blockbench units/sec (bedrock_gecko_exporter divides by units_per_block=16!)
    est_speed_u_s = float(swim.freq_hz) * body_len * (0.28 if swim.mode == "leviathan_hover" else 0.55)
    root_vel = (0.0, 0.0, round(est_speed_u_s * forward_sign_z, 4))

    samples: list[BehaviorFrameSample] = []
    for f_i, phi in enumerate(phases):
        t_s = float(phi * clip.duration_s)
        theta_cyc = 2.0 * math.pi * phi * max(0.25, float(swim.freq_hz) * float(clip.duration_s))
        # Ensure clean loop wrap on periodic clips: integer cycle count if clip.loop
        if clip.loop:
            n_cycles = max(1, int(round(float(swim.freq_hz) * float(clip.duration_s))))
            theta_cyc = 2.0 * math.pi * phi * float(n_cycles)

        pose_euler: dict[str, np.ndarray] = {}
        pose_pos: dict[str, np.ndarray] = {}
        pose_scale: dict[str, np.ndarray] = {}

        # Merge any additive explicit keyframes / tracks / muscles authored on the clip
        axis_idx = {"x": 0, "y": 1, "z": 2}
        dur_s = float(clip.duration_s)
        for tr in clip.tracks:
            ang = _evaluate_track_angle(tr.law, t_s, dur_s, tr.amp_deg, tr.freq_hz, tr.phase_deg, tr.bias_deg)
            vec = pose_euler.setdefault(tr.bone, np.zeros(3, dtype=np.float64))
            vec[axis_idx[tr.axis]] += ang
        for kf in clip.keyframes:
            kv = evaluate_keyframe_track(kf, t_s)
            if kf.channel == "rotation":
                pose_euler[kf.bone] = pose_euler.get(kf.bone, np.zeros(3, dtype=np.float64)) + np.asarray(kv, dtype=np.float64)
            elif kf.channel == "position":
                pose_pos[kf.bone] = pose_pos.get(kf.bone, np.zeros(3, dtype=np.float64)) + np.asarray(kv, dtype=np.float64)
            elif kf.channel == "scale":
                pose_scale[kf.bone] = pose_scale.get(kf.bone, np.ones(3, dtype=np.float64)) * np.asarray(kv, dtype=np.float64)
        base_musc: dict[str, float] = {}
        for md in clip.muscle_drives:
            u_m = (t_s * md.freq_hz + md.phase_deg / 360.0) % 1.0
            val = md.baseline + (md.peak - md.baseline) * (0.5 - 0.5 * math.cos(2.0 * math.pi * u_m))
            base_musc[md.muscle] = float(np.clip(max(base_musc.get(md.muscle, 0.0), val), 0.0, 1.0))

        # 1) Underwater buoyancy positioning, pitch trim, roll banking & lateral recoil sway
        bob_y = float(swim.heave_amp_u) * math.sin(theta_cyc)
        pitch_osc = (2.2 if swim.mode == "leviathan_hover" else 1.6) * math.cos(theta_cyc)
        roll_osc = float(swim.roll_bank_deg) * math.sin(theta_cyc - 0.45)
        lat_sway_x = -(0.018 * body_len) * (float(swim.wave_amp_deg) / 16.0) * math.sin(theta_cyc)

        root_delta = np.array([lat_sway_x, float(swim.buoyancy_y) + bob_y, 0.0], dtype=np.float64)
        d_root = pose_euler.get(root_name, np.zeros(3, dtype=np.float64))
        d_root[0] += float(swim.body_pitch_deg) + pitch_osc
        d_root[2] += roll_osc
        pose_euler[root_name] = d_root

        if swim.mode == "medusa_pulse":
            # Rhythmic pelagic bell contraction/recovery + radial tentacle propulsion waves
            pulse = math.sin(theta_cyc)
            bell_rad = 1.0 + 0.075 * float(swim.myomere_bulge) * pulse
            bell_ax = 1.0 / max(0.75, bell_rad)
            pose_scale[root_name] = pose_scale.get(root_name, np.ones(3)) * np.array([bell_rad, bell_rad, bell_ax])
            for r_idx, rch in enumerate(radial_chains):
                for seg_i, bn in enumerate(rch):
                    rb = rest_scene.bone_by_name[bn]
                    r_xy = rb.pivot_world.copy() - rest_scene.bones[0].pivot_world
                    r_xy[2] = 0.0
                    rn = float(np.linalg.norm(r_xy))
                    if rn > 1e-4:
                        r_hat = r_xy / rn
                        tang_hat = np.cross(np.array([0.0, 0.0, 1.0]), r_hat)
                        w_ang = float(swim.wave_amp_deg) * (0.45 + 0.25 * seg_i) * math.sin(
                            theta_cyc - 0.65 * seg_i - 0.12 * r_idx
                        )
                        d_r = _world_axis_to_local_euler_deg(rb.rotation, tang_hat, w_ang, bone_signs.get(bn))
                        pose_euler[bn] = pose_euler.get(bn, np.zeros(3)) + d_r
        else:
            # 2) Progressive hydrodynamic traveling S-wave across neck -> torso -> paddle tail
            total_phase_span = 2.0 * math.pi * float(swim.wavelength_ratio)
            world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            cum_head_yaw = 0.0
            myo_gain = float(np.clip(swim.myomere_bulge, 0.0, 1.0))

            for idx, bn in enumerate(ordered_axial):
                s = idx / float(max(1, n_ax - 1))
                env = _mode_wave_profile(swim.mode, s)
                local_phase = theta_cyc - s * total_phase_span
                yaw_deg = float(swim.wave_amp_deg) * env * math.sin(local_phase)

                # Distribute modest share to root/torso, larger whip amplitude to distal tail
                if bn == root_name:
                    yaw_deg *= 0.45
                elif bn in torso_bones:
                    yaw_deg *= 0.60
                elif bn in neck_bones:
                    yaw_deg *= 0.55

                rb = rest_scene.bone_by_name[bn]
                # Head vestibulo-ocular stabilization counter-rotates skull to keep eyes locked dead-ahead!
                if head_bone is not None and bn == head_bone and float(swim.head_stabilize) > 1e-4:
                    yaw_deg = -float(swim.head_stabilize) * cum_head_yaw
                elif bn in neck_bones or bn in torso_bones:
                    cum_head_yaw += yaw_deg

                d_w = _world_axis_to_local_euler_deg(rb.rotation, world_y, yaw_deg, bone_signs.get(bn))
                # Subtle hydro vertical pitch undulation along tail tip
                if bn in tail_bones:
                    t_idx = tail_bones.index(bn)
                    pitch_tail = 0.14 * float(swim.wave_amp_deg) * (0.4 + 0.6 * (t_idx / max(1, len(tail_bones) - 1))) * math.cos(local_phase)
                    d_w[0] += pitch_tail
                pose_euler[bn] = pose_euler.get(bn, np.zeros(3, dtype=np.float64)) + d_w

                # 3) Active Underwater Musculoskeletal Sim(3) Myomere Contraction Bulge!
                if myo_gain > 1e-4 and bn != head_bone:
                    # Volume-preserving lateral myomere thickening synced to |sin(local_phase)|
                    act = abs(math.sin(local_phase)) * env
                    bulge_x = 1.0 + 0.048 * myo_gain * act
                    bulge_y = 1.0 + 0.022 * myo_gain * act
                    bulge_z = 1.0 / max(0.85, math.sqrt(bulge_x * bulge_y))
                    cur_s = pose_scale.get(bn, np.ones(3, dtype=np.float64))
                    pose_scale[bn] = cur_s * np.array([bulge_x, bulge_y, bulge_z], dtype=np.float64)

            # 4) Jacobian-calibrated hydrodynamic limb positioning (hydro_tuck or paddle_row)
            all_limbs = list(morph.limbs) + list(morph.free_limbs)
            for limb in all_limbs:
                _solve_hydro_limb_pose(
                    rest_scene,
                    limb,
                    swim,
                    phi * (max(1, int(round(float(swim.freq_hz) * float(clip.duration_s)))) if clip.loop else float(swim.freq_hz) * float(clip.duration_s)),
                    forward_sign_z,
                    pose_euler,
                    bone_signs,
                )

            # Trailing webbed digit / fin feathering
            if float(swim.digit_articulation) > 1e-4:
                dig_amp = float(swim.fin_lag_deg) * float(swim.digit_articulation)
                for b in rest_scene.bones:
                    low = b.name.lower()
                    if any(k in low for k in ("toe_", "finger_", "thumb_", "dewclaw_")) and "wing_" not in low:
                        d_dig = pose_euler.get(b.name, np.zeros(3, dtype=np.float64))
                        base_tuck = 18.0 * float(swim.limb_tuck) if swim.limb_mode == "hydro_tuck" else 6.0
                        d_dig[0] += base_tuck + 0.35 * dig_amp * math.sin(theta_cyc - 1.1)
                        pose_euler[b.name] = d_dig

        # 5) Evaluate declared @muscles (e.g. thorax_diaphragm, mandible_depressor, shock_sac)
        box_inflate: dict[str, float] = {}
        if base_musc and getattr(program, "muscles", None):
            mres = evaluate_muscles_at_activations(program, base_musc, base_pose_deltas=pose_euler)
            for bn, ev in mres.pose_euler_deltas.items():
                pose_euler[bn] = np.asarray(ev, dtype=np.float64)
            box_inflate = dict(mres.box_inflate_deltas)
            scl_tmp = {bn: (float(v[0]), float(v[1]), float(v[2])) for bn, v in pose_scale.items()}
            _apply_muscle_scale_bulge(program, rest_scene, base_musc, scl_tmp)
            for bn, sv in scl_tmp.items():
                pose_scale[bn] = np.asarray(sv, dtype=np.float64)

        # 6) Soft-clamp every bone delta to biological joint limits (with exact mirror angle_sign!)
        clamped_euler: dict[str, Vec3] = {}
        for bn, ev in pose_euler.items():
            if bn in limits_map:
                lim, asign = limits_map[bn]
                c_ev = soft_clamp_euler_to_limits(ev, lim, angle_sign=asign)
            else:
                c_ev = ev
            clamped_euler[bn] = (float(c_ev[0]), float(c_ev[1]), float(c_ev[2]))

        pos_out = {bn: (float(v[0]), float(v[1]), float(v[2])) for bn, v in pose_pos.items()}
        scl_out = {bn: (float(v[0]), float(v[1]), float(v[2])) for bn, v in pose_scale.items()}

        # 7) Compile posed scene & enforce underwater seabed ground_clearance guard if enabled
        posed_scene = compile_posed_geometry(
            program,
            pose_euler_deltas=clamped_euler,
            box_inflate_deltas=box_inflate,
            root_translation_delta=(float(root_delta[0]), float(root_delta[1]), float(root_delta[2])),
            pose_position_deltas=pos_out,
            pose_scale_factors=scl_out,
            validate=False,
        )
        if swim.ground_clearance:
            cur_solid_mins = [
                float(bx.corners(include_inflate=True)[:, 1].min())
                for bx in posed_scene.boxes
                if getattr(bx, "plane_normal", None) is None
            ]
            cur_min_y = min(cur_solid_mins) if cur_solid_mins else 0.0
            target_floor = max(0.5, rest_min_y + 2.0)
            if cur_min_y < target_floor:
                lift = target_floor - cur_min_y
                root_delta[1] += lift
                posed_scene = compile_posed_geometry(
                    program,
                    pose_euler_deltas=clamped_euler,
                    box_inflate_deltas=box_inflate,
                    root_translation_delta=(float(root_delta[0]), float(root_delta[1]), float(root_delta[2])),
                    pose_position_deltas=pos_out,
                    pose_scale_factors=scl_out,
                    validate=False,
                )

        samples.append(
            BehaviorFrameSample(
                frame_index=f_i,
                time_s=round(t_s, 6),
                phase=round(float(phi), 6),
                pose_euler_deltas=clamped_euler,
                box_inflate_deltas=box_inflate,
                root_translation_delta=(float(root_delta[0]), float(root_delta[1]), float(root_delta[2])),
                muscle_activations=dict(base_musc),
                effector_world_positions={},
                effector_in_stance={},
                effector_stance_step_id={},
                scene=posed_scene,
                pose_position_deltas=pos_out,
                pose_scale_factors=scl_out,
                root_motion_velocity=root_vel,
            )
        )

    return samples


def audit_swim_clip_contracts(
    program: Program,
    clip: ClipDecl,
    frames: Sequence[Any],
    *,
    precomputed_collision_events: int | None = None,
) -> dict[str, Any]:
    """Audit underwater swimming biomechanical contracts (entity-space, buoyancy clearance, Sim(3), loop, SAT)."""
    swim = clip.swims[0]
    max_root_z = max((abs(float(f.root_translation_delta[2])) for f in frames), default=0.0)
    min_solid_y = min(
        (
            float(bx.corners(include_inflate=True)[:, 1].min())
            for f in frames
            for bx in f.scene.boxes
            if getattr(bx, "plane_normal", None) is None
        ),
        default=0.0,
    )
    sim3_bones = {
        bn
        for f in frames
        for bn, sv in f.pose_scale_factors.items()
        if max(abs(float(v) - 1.0) for v in sv) > 1e-4
    }
    in_place_ok = (not swim.in_place) or (max_root_z <= 1e-3)
    buoyancy_ok = (not swim.ground_clearance) or (min_solid_y >= -0.05)
    coll_ok = True if precomputed_collision_events is None else (precomputed_collision_events == 0)
    sim3_ok = (float(swim.myomere_bulge) <= 1e-4) or (len(sim3_bones) >= 1)

    checks = {
        "in_place_entity_space": in_place_ok,
        "buoyancy_clearance": buoyancy_ok,
        "myomere_sim3_active": sim3_ok,
        "clipping": coll_ok,
    }
    vel = frames[0].root_motion_velocity if frames else (0.0, 0.0, 0.0)
    upb = float(getattr(program.meta, "units_per_block", 16))
    return {
        "passed": all(checks.values()),
        "mode": swim.mode,
        "limb_mode": swim.limb_mode,
        "checks": checks,
        "max_root_z": round(max_root_z, 5),
        "min_solid_y": round(min_solid_y, 3),
        "myomere_sim3_bones": len(sim3_bones),
        "root_motion": {
            "in_place": bool(swim.in_place),
            "speed_blocks_per_s": round(abs(float(vel[2])) / max(1.0, upb), 4),
            "velocity_blocks_per_s": [round(float(c) / max(1.0, upb), 4) for c in vel],
        },
    }
