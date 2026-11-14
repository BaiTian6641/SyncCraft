"""P1-03d Action-Conditioned 3D World Transition Engine: S_{t+1} = F(S_t, a_t, E_t, dt).

Integrates geometry, kinematics, surface material fields, tribological sliding wear,
and reduced-order environmental weathering under one pure transition contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np

from synccraft.audit.sync_audit import audit_scene
from synccraft.data_gen.ik_motion_templates import sample_joint_trajectory, solve_two_bone_ik
from synccraft.dsl.ast_nodes import Meta, Program, Vec3, validate_program
from synccraft.dsl.parser import format_program, parse_program
from synccraft.kinematics.musculoskeletal import (
    audit_behavior_clip,
    evaluate_behavior_clip,
    evaluate_muscles_at_activations,
)
from synccraft.shading.uv_packer import allocate_uv_layout
from synccraft.surface.contracts import (
    EnvironmentState,
    FaceKey,
    MaterialProgram,
    MaterialSpec,
    StampOverride,
    WorldState,
)
from synccraft.surface.domain import (
    SurfaceDomain,
    build_surface_domain,
    evaluate_posed_surface_kinematics,
    remap_surface_state,
)
from synccraft.surface.processes import (
    integrate_gray_scott_on_state,
    step_pure_diffusion,
)


# ---------------------------------------------------------------------------
# Typed Action Specifications (JSON-serializable for World-Model datasets)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoOpEnvironmentAction:
    """Advance environmental exposure (humidity, oxidation, biofilm) over dt."""

    evolve_rd: bool = False
    kind: str = "environment_exposure"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "evolve_rd": self.evolve_rd}


@dataclass(frozen=True)
class SetPoseAction:
    """Sweep joints smoothly from current pose to target pose_euler_deltas."""

    pose_euler_deltas: dict[str, Vec3]
    contact_normal_load: float = 1.0
    substeps: int = 8
    contact_threshold: float = 0.55
    kind: str = "set_pose"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "pose_euler_deltas": {k: list(v) for k, v in sorted(self.pose_euler_deltas.items())},
            "contact_normal_load": self.contact_normal_load,
            "substeps": self.substeps,
            "contact_threshold": self.contact_threshold,
        }


@dataclass(frozen=True)
class JointTrajectoryAction:
    """Execute a multi-keyframe joint trajectory (e.g. reciprocated hinge wear cycle)."""

    keyframes: tuple[tuple[float, dict[str, Vec3]], ...]
    contact_normal_load: float = 1.25
    num_samples: int = 16
    contact_threshold: float = 0.55
    kind: str = "joint_trajectory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "keyframes": [
                [t, {k: list(v) for k, v in sorted(pose.items())}]
                for t, pose in self.keyframes
            ],
            "contact_normal_load": self.contact_normal_load,
            "num_samples": self.num_samples,
            "contact_threshold": self.contact_threshold,
        }


@dataclass(frozen=True)
class TwoBoneIKAction:
    """Solve 2-bone limb IK to a world target and sweep the limb into position."""

    upper_bone: str
    lower_bone: str
    effector_local: Vec3
    target_world: Vec3
    hinge_axis: str = "x"
    contact_normal_load: float = 1.0
    substeps: int = 8
    kind: str = "two_bone_ik"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "upper_bone": self.upper_bone,
            "lower_bone": self.lower_bone,
            "effector_local": list(self.effector_local),
            "target_world": list(self.target_world),
            "hinge_axis": self.hinge_axis,
            "contact_normal_load": self.contact_normal_load,
            "substeps": self.substeps,
        }


@dataclass(frozen=True)
class ApplyRubbingAction:
    """Localized tribological rubbing/abrasion intervention on (box_name, face)."""

    box_name: str
    face: str
    u_center: float
    v_center: float
    radius: float = 1.0
    normal_load: float = 1.5
    sliding_distance: float = 2.0
    kind: str = "apply_rubbing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "box_name": self.box_name,
            "face": self.face,
            "u_center": self.u_center,
            "v_center": self.v_center,
            "radius": self.radius,
            "normal_load": self.normal_load,
            "sliding_distance": self.sliding_distance,
        }


@dataclass(frozen=True)
class EditMaterialAction:
    """Authoring intervention updating a MaterialSpec or StampOverrides."""

    material_spec: MaterialSpec
    stamps: tuple[StampOverride, ...] | None = None
    reset_rd: bool = True
    kind: str = "edit_material"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "material_spec": self.material_spec.to_dict(),
            "stamps": [s.to_dict() for s in self.stamps] if self.stamps is not None else None,
            "reset_rd": self.reset_rd,
        }


@dataclass(frozen=True)
class EditGeometryAction:
    """Authoring intervention updating the @rig Program with explicit state remapping."""

    new_program: Program
    remap_policy: str = "preserve_matching_parts"
    kind: str = "edit_geometry"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "new_program_dsl": format_program(self.new_program),
            "remap_policy": self.remap_policy,
        }


@dataclass(frozen=True)
class SetTexelDensityAction:
    """Change target pixel resolution without altering solver-space persistent fields."""

    texel_density: int
    kind: str = "set_texel_density"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "texel_density": self.texel_density}


@dataclass(frozen=True)
class SetLightingAction:
    """Change external lighting conditions without altering intrinsic materials or state."""

    light_dir_world: Vec3
    ambient_intensity: float | None = None
    sun_intensity: float | None = None
    kind: str = "set_lighting"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "light_dir_world": list(self.light_dir_world),
            "ambient_intensity": self.ambient_intensity,
            "sun_intensity": self.sun_intensity,
        }


@dataclass(frozen=True)
class RepackUVAction:
    """Repack UV island layout (order, max_width, padding) leaving WorldState invariant."""

    max_width: int = 256
    padding: int = 2
    reverse_boxes: bool = True
    reverse_faces: bool = True
    kind: str = "repack_uv"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "max_width": self.max_width,
            "padding": self.padding,
            "reverse_boxes": self.reverse_boxes,
            "reverse_faces": self.reverse_faces,
        }


@dataclass(frozen=True)
class ExecuteBehaviorClipAction:
    """Execute a musculoskeletal & closed-loop IK @behavior clip with swept wear & stance audit."""

    clip_name: str
    active_state: str | None = None
    cycles: int = 1
    num_frames_per_cycle: int = 12
    contact_normal_load: float = 1.2
    contact_threshold: float = 0.60
    kind: str = "execute_behavior_clip"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "clip_name": self.clip_name,
            "active_state": self.active_state,
            "cycles": self.cycles,
            "num_frames_per_cycle": self.num_frames_per_cycle,
            "contact_normal_load": self.contact_normal_load,
            "contact_threshold": self.contact_threshold,
        }


@dataclass(frozen=True)
class ActivateBehaviorStateAction:
    """Transition entity into a named @behavior state (pose_bias, muscle_tone, material_mods)."""

    state_name: str
    substeps: int = 6
    contact_normal_load: float = 1.0
    kind: str = "activate_behavior_state"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "state_name": self.state_name,
            "substeps": self.substeps,
            "contact_normal_load": self.contact_normal_load,
        }


WorldAction = (
    NoOpEnvironmentAction
    | SetPoseAction
    | JointTrajectoryAction
    | TwoBoneIKAction
    | ApplyRubbingAction
    | EditMaterialAction
    | EditGeometryAction
    | SetTexelDensityAction
    | SetLightingAction
    | RepackUVAction
    | ExecuteBehaviorClipAction
    | ActivateBehaviorStateAction
)


def action_from_dict(data: dict[str, Any]) -> WorldAction:
    """Deserialize a typed WorldAction from JSON dictionary."""
    kind = data["kind"]
    if kind == "environment_exposure":
        return NoOpEnvironmentAction(evolve_rd=bool(data.get("evolve_rd", False)))
    if kind == "set_pose":
        poses = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in data["pose_euler_deltas"].items()}
        return SetPoseAction(
            pose_euler_deltas=poses,
            contact_normal_load=float(data.get("contact_normal_load", 1.0)),
            substeps=int(data.get("substeps", 8)),
            contact_threshold=float(data.get("contact_threshold", 0.55)),
        )
    if kind == "joint_trajectory":
        kfs = tuple(
            (float(t), {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in p.items()})
            for t, p in data["keyframes"]
        )
        return JointTrajectoryAction(
            keyframes=kfs,
            contact_normal_load=float(data.get("contact_normal_load", 1.25)),
            num_samples=int(data.get("num_samples", 16)),
            contact_threshold=float(data.get("contact_threshold", 0.55)),
        )
    if kind == "two_bone_ik":
        eff = data["effector_local"]
        tgt = data["target_world"]
        return TwoBoneIKAction(
            upper_bone=str(data["upper_bone"]),
            lower_bone=str(data["lower_bone"]),
            effector_local=(float(eff[0]), float(eff[1]), float(eff[2])),
            target_world=(float(tgt[0]), float(tgt[1]), float(tgt[2])),
            hinge_axis=str(data.get("hinge_axis", "x")),
            contact_normal_load=float(data.get("contact_normal_load", 1.0)),
            substeps=int(data.get("substeps", 8)),
        )
    if kind == "apply_rubbing":
        return ApplyRubbingAction(
            box_name=str(data["box_name"]),
            face=str(data["face"]),
            u_center=float(data["u_center"]),
            v_center=float(data["v_center"]),
            radius=float(data.get("radius", 1.0)),
            normal_load=float(data.get("normal_load", 1.5)),
            sliding_distance=float(data.get("sliding_distance", 2.0)),
        )
    if kind == "edit_material":
        spec = MaterialSpec.from_dict(data["material_spec"])
        stamps = (
            tuple(StampOverride.from_dict(s) for s in data["stamps"])
            if data.get("stamps") is not None
            else None
        )
        return EditMaterialAction(material_spec=spec, stamps=stamps, reset_rd=bool(data.get("reset_rd", True)))
    if kind == "edit_geometry":
        prog = parse_program(str(data["new_program_dsl"]))
        return EditGeometryAction(new_program=prog, remap_policy=str(data.get("remap_policy", "preserve_matching_parts")))
    if kind == "set_texel_density":
        return SetTexelDensityAction(texel_density=int(data["texel_density"]))
    if kind == "set_lighting":
        lw = data["light_dir_world"]
        return SetLightingAction(
            light_dir_world=(float(lw[0]), float(lw[1]), float(lw[2])),
            ambient_intensity=float(data["ambient_intensity"]) if data.get("ambient_intensity") is not None else None,
            sun_intensity=float(data["sun_intensity"]) if data.get("sun_intensity") is not None else None,
        )
    if kind == "repack_uv":
        return RepackUVAction(
            max_width=int(data.get("max_width", 256)),
            padding=int(data.get("padding", 2)),
            reverse_boxes=bool(data.get("reverse_boxes", True)),
            reverse_faces=bool(data.get("reverse_faces", True)),
        )
    if kind == "execute_behavior_clip":
        return ExecuteBehaviorClipAction(
            clip_name=str(data["clip_name"]),
            active_state=str(data["active_state"]) if data.get("active_state") is not None else None,
            cycles=int(data.get("cycles", 1)),
            num_frames_per_cycle=int(data.get("num_frames_per_cycle", 12)),
            contact_normal_load=float(data.get("contact_normal_load", 1.2)),
            contact_threshold=float(data.get("contact_threshold", 0.60)),
        )
    if kind == "activate_behavior_state":
        return ActivateBehaviorStateAction(
            state_name=str(data["state_name"]),
            substeps=int(data.get("substeps", 6)),
            contact_normal_load=float(data.get("contact_normal_load", 1.0)),
        )
    raise ValueError(f"Unknown WorldAction kind {kind!r}")


@dataclass
class TransitionResult:
    """Output of S_{t+1} = F(S_t, a_t, E_t, dt) with causal & cost audit records."""

    next_state: WorldState
    swept_audit_passed: bool
    substeps_audited: int
    max_illegal_overlap_depth: float
    total_sliding_wear_added: float
    total_oxidation_added: float
    total_biofilm_added: float
    repacked_uv_layout: dict[str, Any] | None = None
    ik_metadata: dict[str, Any] | None = None
    cost_counters: dict[str, int] = field(default_factory=dict)
    reduced_model_notes: tuple[str, ...] = (
        "Archard sliding wear, moisture relaxation, exposed-metal oxidation, and biofilm "
        "colonization are calibrated reduced-order mechanisms; SAT checks swept overlap, not contact forces.",
    )


def _extract_audit_status(aud: dict[str, Any]) -> tuple[bool, float]:
    """Return (no_illegal_collisions_and_connected, max_illegal_overlap) from audit_scene()."""
    collisions = aud.get("collision_pairs", [])
    errors = aud.get("errors", [])
    disconnected = any(v.get("violation") == "disconnected_physical_components" for v in aud.get("contact_violations", []))
    ok = (len(collisions) == 0) and (len(errors) == 0) and not disconnected
    peak_overlap = max((float(c.get("raw_min_overlap", 0.0)) for c in collisions), default=0.0)
    return ok, max(0.0, peak_overlap)


def _accumulate_swept_sliding_wear(
    domain: SurfaceDomain,
    program: Program,
    material_program: MaterialProgram,
    surface_state: Any,
    pose_sequence: list[dict[str, Vec3]],
    *,
    contact_normal_load: float,
    contact_threshold: float,
) -> tuple[bool, int, float, float, int, int]:
    """Audit sub-stepped poses via SAT and accumulate relative tangential Archard sliding wear."""
    pal_map = {p.name: p for p in program.palettes}
    swept_ok = True
    max_overlap = 0.0
    wear_added = 0.0
    compiler_calls = 0
    audit_calls = 0

    prev_scene, prev_kin = evaluate_posed_surface_kinematics(
        domain,
        program,
        pose_sequence[0],
        contact_threshold=contact_threshold,
    )
    compiler_calls += 1
    audit_0 = audit_scene(prev_scene, enforce_physical_contact=True)
    audit_calls += 1
    ok_0, ov_0 = _extract_audit_status(audit_0)
    if not ok_0:
        swept_ok = False
    max_overlap = max(max_overlap, ov_0)

    for step_pose in pose_sequence[1:]:
        cur_scene, cur_kin = evaluate_posed_surface_kinematics(
            domain,
            program,
            step_pose,
            contact_threshold=contact_threshold,
        )
        compiler_calls += 1
        aud = audit_scene(cur_scene, enforce_physical_contact=True)
        audit_calls += 1
        ok_m, ov_m = _extract_audit_status(aud)
        if not ok_m:
            swept_ok = False
        max_overlap = max(max_overlap, ov_m)

        prev_boxes = prev_scene.box_by_name
        cur_boxes = cur_scene.box_by_name

        for fk, grid in domain.faces.items():
            box_name = grid.box_name
            self_box_cur = cur_boxes[box_name]
            spec = material_program.spec_for(grid.material_name, pal_map.get(grid.material_name))
            if spec.wear_susceptibility <= 0.0 or contact_normal_load <= 0.0:
                continue

            w_cur = cur_kin[fk].world_pos
            w_prev = prev_kin[fk].world_pos
            n_cur = cur_kin[fk].world_normal

            p_lo = w_cur.reshape(-1, 3).min(axis=0) - contact_threshold
            p_hi = w_cur.reshape(-1, 3).max(axis=0) + contact_threshold

            patch_wear_delta = np.zeros(grid.shape, dtype=np.float64)

            for other_cur in cur_scene.boxes:
                if other_cur.name == box_name or other_cur.bone_name == self_box_cur.bone_name:
                    continue
                o_corners = other_cur.corners(include_inflate=True)
                if np.any(o_corners.max(axis=0) < p_lo) or np.any(o_corners.min(axis=0) > p_hi):
                    continue

                o_rot_cur = np.asarray(other_cur.rotation, dtype=np.float64)
                o_ctr_cur = np.asarray(other_cur.center, dtype=np.float64)
                o_half = other_cur.half_extents

                q_other = (w_cur - o_ctr_cur) @ o_rot_cur
                q_abs = np.abs(q_other) - o_half
                outside = np.maximum(q_abs, 0.0)
                inside = np.minimum(np.max(q_abs, axis=-1), 0.0)
                signed_d = np.linalg.norm(outside, axis=-1) + inside

                near_mask = signed_d <= contact_threshold
                if not np.any(near_mask):
                    continue

                psi = np.clip(1.0 - np.maximum(0.0, signed_d) / max(1e-6, contact_threshold), 0.0, 1.0)
                other_prev = prev_boxes[other_cur.name]
                w_partner_prev = q_other @ np.asarray(other_prev.rotation, dtype=np.float64).T + np.asarray(other_prev.center, dtype=np.float64)

                # Relative displacement of self sample vs partner rigid body
                delta_rel = w_partner_prev - w_prev
                n_comp = np.sum(delta_rel * n_cur, axis=-1, keepdims=True) * n_cur
                slide_tan = np.linalg.norm(delta_rel - n_comp, axis=-1)

                dw = spec.wear_susceptibility * contact_normal_load * psi * slide_tan * 0.22
                patch_wear_delta += np.where(near_mask, dw, 0.0)

            if np.any(patch_wear_delta > 0.0):
                st_patch = surface_state.patches[fk]
                st_patch.wear = st_patch.wear + patch_wear_delta
                if spec.coating_thickness > 0.0:
                    st_patch.coating_damage = st_patch.coating_damage + patch_wear_delta * 0.85
                wear_added += float(patch_wear_delta.sum())

        prev_scene, prev_kin = cur_scene, cur_kin

    return swept_ok, len(pose_sequence), max_overlap, wear_added, compiler_calls, audit_calls


def _evolve_environmental_fields(
    domain: SurfaceDomain,
    program: Program,
    material_program: MaterialProgram,
    surface_state: Any,
    posed_kin: dict[FaceKey, Any],
    env: EnvironmentState,
    dt: float,
    *,
    evolve_rd: bool = False,
) -> tuple[float, float, int]:
    """Advance moisture, exposed-metal oxidation, biofilm, and optional RD over dt > 0."""
    if dt <= 0.0:
        return 0.0, 0.0, 0

    pal_map = {p.name: p for p in program.palettes}
    ox_added = 0.0
    bio_added = 0.0
    pde_substeps = 0

    # 1. Moisture equilibration + conservative surface diffusion across manifold seams
    moisture_map: dict[FaceKey, np.ndarray] = {}
    alpha_m = 1.0 - math.exp(-0.55 * dt)
    for fk, grid in domain.faces.items():
        spec = material_program.spec_for(grid.material_name, pal_map.get(grid.material_name))
        st = surface_state.patches[fk]
        pk = posed_kin[fk]
        poros_factor = 0.35 + 0.65 * spec.porosity
        shelter_factor = 1.15 - 0.30 * pk.cavity_ao
        m_eq = np.clip(env.humidity * poros_factor * shelter_factor, 0.0, 1.0)
        moisture_map[fk] = np.clip(st.moisture + alpha_m * (m_eq - st.moisture), 0.0, 1.0)

    diff_dt = min(dt, 2.5)
    moisture_diffused = step_pure_diffusion(domain, moisture_map, diffusivity=0.015, total_time=diff_dt)
    pde_substeps += max(1, int(math.ceil(0.015 * diff_dt / (0.18 * domain.dx * domain.dx))))
    for fk, m_arr in moisture_diffused.items():
        surface_state.patches[fk].moisture = np.clip(m_arr, 0.0, 1.0)

    # 2. Selective oxidation on exposed metallic substrates & biofilm colonization
    temp_factor = math.exp(0.025 * (env.temperature_c - 20.0))
    for fk, grid in domain.faces.items():
        spec = material_program.spec_for(grid.material_name, pal_map.get(grid.material_name))
        st = surface_state.patches[fk]
        pk = posed_kin[fk]
        total_wear = st.wear + st.coating_damage

        if spec.family == "painted_metal":
            coat_rem = np.clip(spec.coating_thickness - total_wear, 0.0, max(1e-6, spec.coating_thickness))
            coat_cov = np.clip(coat_rem / max(0.08, 0.35 * max(0.2, spec.coating_thickness)), 0.0, 1.0)
            exposed_metal = 1.0 - coat_cov
        elif spec.family == "brushed_metal":
            exposed_metal = np.full(grid.shape, spec.substrate_metalness, dtype=np.float64)
        else:
            exposed_metal = np.zeros(grid.shape, dtype=np.float64)

        if np.any(exposed_metal > 1e-4) and spec.oxidation_rate > 0.0 and env.oxygen_activity > 0.0:
            moisture_boost = 0.08 + 1.92 * st.moisture
            d_ox = (
                spec.oxidation_rate
                * exposed_metal
                * env.oxygen_activity
                * moisture_boost
                * temp_factor
                * dt
                * (1.0 - st.oxidation)
            )
            d_ox = np.clip(d_ox, 0.0, 1.0 - st.oxidation)
            st.oxidation = st.oxidation + d_ox
            ox_added += float(d_ox.sum())

        if env.spore_load > 0.0 and spec.biofilm_affinity > 0.0:
            damp = np.maximum(0.0, st.moisture - 0.30)
            upward = np.clip(0.5 + 0.5 * pk.world_normal[..., 1], 0.0, 1.0)
            d_bio = (
                spec.biofilm_affinity
                * env.spore_load
                * damp
                * (0.3 + 0.7 * upward)
                * dt
                * (1.0 - st.biofilm)
            )
            d_bio = np.clip(d_bio, 0.0, 1.0 - st.biofilm)
            st.biofilm = st.biofilm + d_bio
            bio_added += float(d_bio.sum())

    # 3. Optional reaction-diffusion continuation
    if evolve_rd:
        rd_by_mat: dict[str, set[FaceKey]] = {}
        for fk, grid in domain.faces.items():
            spec = material_program.spec_for(grid.material_name, pal_map.get(grid.material_name))
            if spec.family == "patterned_skin":
                rd_by_mat.setdefault(grid.material_name, set()).add(fk)
        for mat_name, fk_set in rd_by_mat.items():
            spec = material_program.spec_for(mat_name, pal_map.get(mat_name))
            pde_substeps += integrate_gray_scott_on_state(domain, surface_state, spec, fk_set, duration=dt)

    return ox_added, bio_added, pde_substeps


def step_world(
    state: WorldState,
    action: WorldAction,
    env: EnvironmentState | None = None,
    dt: float = 1.0,
) -> TransitionResult:
    """Pure world transition S_{t+1} = F(S_t, a_t, E_t, dt)."""
    if dt < 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be a finite nonnegative number")

    cur_env = env if env is not None else state.environment
    cur_prog = state.program
    cur_mat = state.material_program
    cur_style = state.style_config
    cur_pose = dict(state.pose_euler_deltas)
    subdiv = max(4, state.surface_state.subdivisions_per_unit)

    domain = build_surface_domain(cur_prog, cur_mat, subdivisions_per_unit=subdiv)
    next_surf = remap_surface_state(state.surface_state, domain)

    swept_ok = True
    substeps_audited = 1
    max_overlap = 0.0
    wear_added = 0.0
    repacked_layout: dict[str, Any] | None = None
    ik_meta: dict[str, Any] | None = None
    compiler_calls = 1  # build_surface_domain compiled rest geometry
    audit_calls = 0
    pde_substeps = 0

    if isinstance(action, SetPoseAction):
        kfs = [(0.0, cur_pose), (max(1e-3, dt), dict(action.pose_euler_deltas))]
        traj = sample_joint_trajectory(kfs, num_samples=max(2, action.substeps))
        pose_seq = [p for _, p in traj]
        swept_ok, substeps_audited, max_overlap, wear_added, cc, ac = _accumulate_swept_sliding_wear(
            domain,
            cur_prog,
            cur_mat,
            next_surf,
            pose_seq,
            contact_normal_load=action.contact_normal_load,
            contact_threshold=action.contact_threshold,
        )
        compiler_calls += cc
        audit_calls += ac
        cur_pose = dict(action.pose_euler_deltas)

    elif isinstance(action, JointTrajectoryAction):
        traj = sample_joint_trajectory(list(action.keyframes), num_samples=max(2, action.num_samples))
        pose_seq = [cur_pose] + [p for _, p in traj]
        swept_ok, substeps_audited, max_overlap, wear_added, cc, ac = _accumulate_swept_sliding_wear(
            domain,
            cur_prog,
            cur_mat,
            next_surf,
            pose_seq,
            contact_normal_load=action.contact_normal_load,
            contact_threshold=action.contact_threshold,
        )
        compiler_calls += cc
        audit_calls += ac
        cur_pose = dict(pose_seq[-1])

    elif isinstance(action, TwoBoneIKAction):
        ik_res = solve_two_bone_ik(
            cur_prog,
            action.upper_bone,
            action.lower_bone,
            action.effector_local,
            action.target_world,
            base_pose_deltas=cur_pose,
            hinge_axis=action.hinge_axis,
        )
        compiler_calls += ik_res.iterations * 5 + 2
        ik_meta = {
            "achieved_world": list(ik_res.achieved_world),
            "target_world": list(ik_res.target_world),
            "clamped_target_world": list(ik_res.clamped_target_world),
            "reachable": ik_res.reachable,
            "position_error": ik_res.position_error,
            "iterations": ik_res.iterations,
        }
        kfs = [(0.0, cur_pose), (max(1e-3, dt), ik_res.pose_euler_deltas)]
        traj = sample_joint_trajectory(kfs, num_samples=max(2, action.substeps))
        pose_seq = [p for _, p in traj]
        swept_ok, substeps_audited, max_overlap, wear_added, cc, ac = _accumulate_swept_sliding_wear(
            domain,
            cur_prog,
            cur_mat,
            next_surf,
            pose_seq,
            contact_normal_load=action.contact_normal_load,
            contact_threshold=0.55,
        )
        compiler_calls += cc
        audit_calls += ac
        cur_pose = dict(ik_res.pose_euler_deltas)

    elif isinstance(action, ApplyRubbingAction):
        fk = (action.box_name, action.face)
        if fk not in domain.faces:
            raise KeyError(f"Face {fk!r} not found in SurfaceDomain")
        grid = domain.faces[fk]
        pal_map = {p.name: p for p in cur_prog.palettes}
        spec = cur_mat.spec_for(grid.material_name, pal_map.get(grid.material_name))
        d2 = (grid.u_units - action.u_center) ** 2 + (grid.v_units - action.v_center) ** 2
        sigma2 = max(1e-4, 0.5 * action.radius * action.radius)
        kernel = np.exp(-d2 / (2.0 * sigma2)) * (d2 <= (action.radius * 1.35) ** 2)
        dw = spec.wear_susceptibility * action.normal_load * action.sliding_distance * kernel * 0.45
        st_patch = next_surf.patches[fk]
        st_patch.wear = st_patch.wear + dw
        if spec.coating_thickness > 0.0:
            st_patch.coating_damage = st_patch.coating_damage + dw * 0.92
        wear_added += float(dw.sum())

    elif isinstance(action, EditMaterialAction):
        cur_mat = cur_mat.with_material(action.material_spec)
        if action.stamps is not None:
            cur_mat = replace(cur_mat, stamps=action.stamps)
        domain = build_surface_domain(cur_prog, cur_mat, subdivisions_per_unit=subdiv)
        compiler_calls += 1
        if action.reset_rd:
            for fk, grid in domain.faces.items():
                if grid.material_name == action.material_spec.name:
                    next_surf.patches[fk].rd_initialized = False

    elif isinstance(action, EditGeometryAction):
        validate_program(action.new_program)
        cur_prog = action.new_program
        domain = build_surface_domain(cur_prog, cur_mat, subdivisions_per_unit=subdiv)
        compiler_calls += 1
        next_surf = remap_surface_state(next_surf, domain, policy=action.remap_policy)
        valid_bones = {b.name for b in domain.rest_scene.bones}
        cur_pose = {k: v for k, v in cur_pose.items() if k in valid_bones}

    elif isinstance(action, SetTexelDensityAction):
        new_meta = Meta(cur_prog.meta.archetype, cur_prog.meta.target_profile, action.texel_density)
        cur_prog = validate_program(replace(cur_prog, meta=new_meta))

    elif isinstance(action, SetLightingAction):
        cur_env = EnvironmentState(
            humidity=cur_env.humidity,
            oxygen_activity=cur_env.oxygen_activity,
            temperature_c=cur_env.temperature_c,
            spore_load=cur_env.spore_load,
            light_dir_world=action.light_dir_world,
            ambient_intensity=action.ambient_intensity if action.ambient_intensity is not None else cur_env.ambient_intensity,
            sun_intensity=action.sun_intensity if action.sun_intensity is not None else cur_env.sun_intensity,
        )

    elif isinstance(action, RepackUVAction):
        posed_scene, _ = evaluate_posed_surface_kinematics(domain, cur_prog, cur_pose)
        compiler_calls += 1
        box_names = [b.name for b in posed_scene.boxes]
        if action.reverse_boxes:
            box_names = list(reversed(box_names))
        face_order = ("down", "up", "west", "east", "south", "north") if action.reverse_faces else None
        repacked_layout = allocate_uv_layout(
            cur_prog,
            posed_scene,
            max_width=action.max_width,
            padding=action.padding,
            box_order=box_names,
            face_order=face_order,
        )

    elif isinstance(action, ExecuteBehaviorClipAction):
        n_frames = max(2, int(action.num_frames_per_cycle))
        n_cycles = max(1, int(action.cycles))
        frames = evaluate_behavior_clip(
            cur_prog,
            action.clip_name,
            num_frames=n_frames,
            active_state=action.active_state,
        )
        clip_audit = audit_behavior_clip(
            cur_prog,
            action.clip_name,
            num_frames=n_frames,
            active_state=action.active_state,
        )
        compiler_calls += n_frames * 3
        audit_calls += n_frames
        pose_seq = [cur_pose] + [fr.pose_euler_deltas for fr in frames]
        swept_w_ok, substeps_audited, max_overlap, wear_added, cc, ac = _accumulate_swept_sliding_wear(
            domain,
            cur_prog,
            cur_mat,
            next_surf,
            pose_seq,
            contact_normal_load=action.contact_normal_load * float(n_cycles),
            contact_threshold=action.contact_threshold,
        )
        compiler_calls += cc
        audit_calls += ac
        swept_ok = bool(swept_w_ok and clip_audit["passed"])
        cur_pose = dict(frames[-1].pose_euler_deltas)
        ik_meta = {
            "clip_name": action.clip_name,
            "active_state": action.active_state,
            "cycles": n_cycles,
            "clip_audit": clip_audit,
        }

    elif isinstance(action, ActivateBehaviorStateAction):
        if cur_prog.behavior is None:
            raise ValueError("Program has no @behavior section")
        st_map = {s.name: s for s in cur_prog.behavior.states}
        if action.state_name not in st_map:
            raise KeyError(f"Unknown @behavior state {action.state_name!r}")
        st = st_map[action.state_name]
        ms_res = evaluate_muscles_at_activations(
            cur_prog,
            dict(st.muscle_tones),
            base_pose_deltas=dict(st.pose_biases),
        )
        compiler_calls += 2
        kfs = [(0.0, cur_pose), (max(1e-3, dt), ms_res.pose_euler_deltas)]
        traj = sample_joint_trajectory(kfs, num_samples=max(2, action.substeps))
        pose_seq = [p for _, p in traj]
        swept_ok, substeps_audited, max_overlap, wear_added, cc, ac = _accumulate_swept_sliding_wear(
            domain,
            cur_prog,
            cur_mat,
            next_surf,
            pose_seq,
            contact_normal_load=action.contact_normal_load,
            contact_threshold=0.55,
        )
        compiler_calls += cc
        audit_calls += ac
        cur_pose = dict(ms_res.pose_euler_deltas)

        # Apply causal state-driven material & persistent surface field interventions
        pal_map = {p.name: p for p in cur_prog.palettes}
        for mod in st.material_mods:
            spec = cur_mat.spec_for(mod.material, pal_map.get(mod.material))
            if abs(mod.emissive) > 1e-9:
                new_em = float(np.clip(max(spec.emissive_strength, mod.emissive), 0.0, 2.0))
                cur_mat = cur_mat.with_material(replace(spec, emissive_strength=new_em))
            for fk, grid in domain.faces.items():
                if grid.material_name == mod.material:
                    st_patch = next_surf.patches[fk]
                    if abs(mod.moisture) > 1e-9:
                        st_patch.moisture = np.clip(np.maximum(st_patch.moisture, mod.moisture), 0.0, 1.0)
                    if abs(mod.wear_delta) > 1e-9:
                        dw_mod = np.maximum(0.0, st_patch.wear + mod.wear_delta) - st_patch.wear
                        st_patch.wear = st_patch.wear + dw_mod
                        wear_added += float(np.maximum(0.0, dw_mod).sum())

        ik_meta = {
            "activated_state": action.state_name,
            "musculoskeletal_equilibrium": ms_res.to_dict(),
        }

    # Final posed evaluation & SAT check if not already audited along a trajectory
    final_scene, final_kin = evaluate_posed_surface_kinematics(domain, cur_prog, cur_pose)
    compiler_calls += 1
    if audit_calls == 0:
        aud_f = audit_scene(final_scene, enforce_physical_contact=True)
        audit_calls += 1
        ok_f, ov_f = _extract_audit_status(aud_f)
        if not ok_f:
            swept_ok = False
        max_overlap = max(max_overlap, ov_f)

    evolve_rd_flag = isinstance(action, NoOpEnvironmentAction) and action.evolve_rd
    ox_added, bio_added, pde_steps = _evolve_environmental_fields(
        domain,
        cur_prog,
        cur_mat,
        next_surf,
        final_kin,
        cur_env,
        dt,
        evolve_rd=evolve_rd_flag,
    )
    pde_substeps += pde_steps

    next_ws = WorldState(
        program=cur_prog,
        material_program=cur_mat,
        surface_state=next_surf,
        pose_euler_deltas=cur_pose,
        style_config=cur_style,
        environment=cur_env,
        time_sec=state.time_sec + dt,
        step_index=state.step_index + 1,
    )

    return TransitionResult(
        next_state=next_ws,
        swept_audit_passed=swept_ok,
        substeps_audited=substeps_audited,
        max_illegal_overlap_depth=max_overlap,
        total_sliding_wear_added=wear_added,
        total_oxidation_added=ox_added,
        total_biofilm_added=bio_added,
        repacked_uv_layout=repacked_layout,
        ik_metadata=ik_meta,
        cost_counters={
            "compiler_calls": compiler_calls,
            "audit_calls": audit_calls,
            "pde_substeps": pde_substeps,
        },
    )
