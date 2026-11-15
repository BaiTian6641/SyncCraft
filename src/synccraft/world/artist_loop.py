"""Closed-Loop Rule-Grounded Artist Critique, Musculoskeletal Upgrade & Progressive Refinement Engine (ADR 0006).

Provides two deterministic, non-diffusion artist-assistant capabilities:
1. Project Intake, Quantitative Critique & Looped Enhancement (`analyze_artist_project`,
   `enhance_artist_project_step`, `run_looped_artist_enhancement`):
   Ingests an artist's existing Block-DSL, `.bbmodel`, Java `.json`, or Bedrock
   `.geo.json` (+ `.animation.json`) project, audits 5 physical/anatomical/material/motion
   pillars, emits ranked actionable suggestions, and executes rule-verified upgrades.
2. Progressive From-Scratch Construction (`build_progressive_from_scratch_trajectory`):
   Constructs complex entities/assets in 5 artist-like rule-governed stages
   (Stage 0 rough gesture blockout -> Stage 1 skeletal hierarchy -> Stage 2 secondary
   anatomy & sockets -> Stage 3 scientific welded materials -> Stage 4 musculoskeletal
   & stance-IK behavior), with 100% SAT + P1-02 verification at every intermediate stage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Any

import numpy as np

from synccraft.audit.sync_audit import _evaluate_declared_contact, audit_scene
from synccraft.dsl.ast_nodes import (
    ANIMATED_PROFILES,
    Attach,
    BehaviorDecl,
    BehaviorStateDecl,
    Bone,
    Box,
    ClipDecl,
    IKChainDecl,
    IKDriveDecl,
    KinematicsDecl,
    Locator,
    LookChainDecl,
    MaterialDecl,
    MaterialModDecl,
    MuscleDecl,
    MuscleDriveDecl,
    Program,
    StyleDecl,
    TendonCoupling,
    TrackDecl,
    Vec3,
    validate_program,
)
from synccraft.dsl.parser import format_program, parse_program
from synccraft.kinematics.musculoskeletal import (
    audit_behavior_clip,
    evaluate_muscles_at_activations,
)
from synccraft.kinematics.sat_potential import sat_obb
from synccraft.kinematics.se3_chain import Scene, compile_geometry
from synccraft.world.inverse_inference import (
    BANNED_PROVENANCE_TOKENS,
    LICENSE_POLICIES,
    AssetProvenance,
    LicenseGateError,
    _decompile_bbmodel,
    _decompile_bedrock_geojson,
    _decompile_java_json,
    decompile_bedrock_animation_to_behavior,
    ingest_external_asset,
)

LOCAL_WORKSPACE_LICENSE_POLICY = "local_workspace_unverified"
_NOAI_RESTRICTED_MARKERS: tuple[str, ...] = (
    "noai",
    "no-ai",
    "no machine learning",
    "do not train",
    "no generative ai",
    "tdm-reservation: 1",
    "ai-generated",
)


@dataclass(frozen=True)
class ArtistSuggestion:
    """Actionable, mechanistically explained improvement proposal for an artist's asset."""

    suggestion_id: str
    priority: str  # "critical" | "high" | "medium" | "low"
    category: str  # "geometry_contact" | "rig_hierarchy" | "musculoskeletal" | "surface_material" | "behavior_motion"
    target_elements: tuple[str, ...]
    title: str
    rationale: str
    proposed_upgrade_op: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "priority": self.priority,
            "category": self.category,
            "target_elements": list(self.target_elements),
            "title": self.title,
            "rationale": self.rationale,
            "proposed_upgrade_op": self.proposed_upgrade_op,
        }


@dataclass(frozen=True)
class ArtistCritiqueReport:
    """Multi-pillar quantitative audit and prioritized enhancement plan for a 3D asset."""

    readiness_score: float
    pillar_scores: dict[str, float]
    box_count: int
    bone_count: int
    locator_count: int
    max_bone_depth: int
    static_audit_passed: bool
    behavior_clips_audited: int
    behavior_clips_passed: int
    suggestions: tuple[ArtistSuggestion, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "readiness_score": round(self.readiness_score, 4),
            "pillar_scores": {k: round(v, 4) for k, v in sorted(self.pillar_scores.items())},
            "box_count": self.box_count,
            "bone_count": self.bone_count,
            "locator_count": self.locator_count,
            "max_bone_depth": self.max_bone_depth,
            "static_audit_passed": self.static_audit_passed,
            "behavior_clips_audited": self.behavior_clips_audited,
            "behavior_clips_passed": self.behavior_clips_passed,
            "suggestions": [s.to_dict() for s in self.suggestions],
        }


@dataclass(frozen=True)
class ProgressiveStageRecord:
    """Single rule-verified stage in a 5-stage from-scratch construction trajectory."""

    stage_index: int
    stage_name: str
    description: str
    program: Program
    program_dsl: str
    box_count: int
    bone_count: int
    static_audit_passed: bool
    readiness_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "description": self.description,
            "box_count": self.box_count,
            "bone_count": self.bone_count,
            "static_audit_passed": self.static_audit_passed,
            "readiness_score": round(self.readiness_score, 4),
            "program_dsl": self.program_dsl,
        }


def _compute_max_bone_depth(bone: Bone, depth: int = 0) -> int:
    child_depths = [
        _compute_max_bone_depth(it, depth + 1) for it in bone.items if isinstance(it, Bone)
    ]
    return max([depth, *child_depths])


def _count_ast_joint_limits(bone: Bone) -> tuple[int, int]:
    """Return (non_root_bones_with_limits, total_non_root_bones)."""
    with_lim = 0
    total = 0
    for it in bone.items:
        if isinstance(it, Bone):
            total += 1
            if it.joint_limits is not None:
                with_lim += 1
            sub_w, sub_t = _count_ast_joint_limits(it)
            with_lim += sub_w
            total += sub_t
    return with_lim, total


def build_unlabeled_workspace_provenance(
    asset_id: str = "artist_workspace_input",
) -> AssetProvenance:
    """Return conservative NOASSERTION provenance for unlabeled local workspace inputs (H7).

    Unlabeled workspace inputs are NEVER stamped permissive (e.g. CC-BY-4.0) by default
    and are strictly flagged ``commercial_weights_eligible=False``.
    """
    return AssetProvenance(
        asset_id=asset_id,
        source_url="local://artist_workspace",
        author="workspace_artist",
        license_spdx="NOASSERTION",
        attribution_notice="Unverified local workspace input; ineligible for training or commercial weights",
    )


def _check_local_workspace_restrictions(payload: dict[str, Any], prov: AssetProvenance) -> None:
    """Reject Mojang/proprietary tokens and NoAI/TDM markers even in local workspace mode."""
    meta_blob = " ".join(
        str(getattr(prov, attr, ""))
        for attr in ("asset_id", "source_url", "author", "license_spdx", "attribution_notice", "title", "changes_made_note")
    )
    doc_meta = json.dumps(
        {
            k: payload.get(k)
            for k in ("name", "credit", "author", "license", "description", "meta", "minecraft:geometry")
            if k in payload
        },
        sort_keys=True,
        default=str,
    )
    combined = f"{meta_blob} {doc_meta}".lower()
    for banned in BANNED_PROVENANCE_TOKENS:
        if banned in combined:
            raise LicenseGateError(f"Asset provenance/metadata contains prohibited source marker {banned!r}")
    for marker in _NOAI_RESTRICTED_MARKERS:
        if marker in combined:
            raise LicenseGateError(f"Asset metadata contains restricted AI opt-out marker {marker!r}")
    if "-nd" in str(prov.license_spdx).lower():
        raise LicenseGateError(f"NoDerivatives license {prov.license_spdx!r} is forbidden")


def ingest_artist_project_record(
    project: Program | str | dict[str, Any],
    *,
    format_kind: str = "blockdsl",
    animation_payload: dict[str, Any] | None = None,
    provenance: AssetProvenance | None = None,
    license_policy: str | None = None,
) -> dict[str, Any]:
    """Ingest an artist input and return Program alongside explicit H7 provenance & eligibility flags."""
    prov = provenance if provenance is not None else build_unlabeled_workspace_provenance()
    effective_policy = (
        license_policy
        if license_policy is not None
        else (LOCAL_WORKSPACE_LICENSE_POLICY if provenance is None else "clean_room_permissive")
    )

    if isinstance(project, Program):
        prog = validate_program(project)
        eligible = bool(provenance is not None and effective_policy == "clean_room_permissive" and prov.license_spdx != "NOASSERTION")
        return {
            "program": prog,
            "provenance": prov,
            "license_spdx": prov.license_spdx,
            "license_policy": effective_policy,
            "license_track": effective_policy if provenance is not None else LOCAL_WORKSPACE_LICENSE_POLICY,
            "commercial_weights_eligible": eligible,
        }
    if isinstance(project, str):
        prog = parse_program(project)
        eligible = bool(provenance is not None and effective_policy == "clean_room_permissive" and prov.license_spdx != "NOASSERTION")
        return {
            "program": prog,
            "provenance": prov,
            "license_spdx": prov.license_spdx,
            "license_policy": effective_policy,
            "license_track": effective_policy if provenance is not None else LOCAL_WORKSPACE_LICENSE_POLICY,
            "commercial_weights_eligible": eligible,
        }
    if isinstance(project, dict):
        _check_local_workspace_restrictions(project, prov)
        if effective_policy == LOCAL_WORKSPACE_LICENSE_POLICY:
            if LOCAL_WORKSPACE_LICENSE_POLICY in LICENSE_POLICIES:
                rec = ingest_external_asset(
                    project,
                    format_kind=format_kind,
                    provenance=prov,
                    animation_payload=animation_payload,
                    license_policy=LOCAL_WORKSPACE_LICENSE_POLICY,
                )
                return {
                    "program": rec.program,
                    "provenance": prov,
                    "license_spdx": prov.license_spdx,
                    "license_policy": LOCAL_WORKSPACE_LICENSE_POLICY,
                    "license_track": LOCAL_WORKSPACE_LICENSE_POLICY,
                    "commercial_weights_eligible": False,
                }
            raw_to_safe_bone: dict[str, str] | None = None
            if format_kind == "bbmodel":
                prog, _, _, _ = _decompile_bbmodel(project)
            elif format_kind == "java_json":
                prog, _, _ = _decompile_java_json(project)
            elif format_kind == "bedrock_geojson":
                prog, _, _, raw_to_safe_bone = _decompile_bedrock_geojson(project)
            else:
                raise ValueError(
                    f"Unsupported external format_kind {format_kind!r}; expected 'bbmodel', 'java_json', or 'bedrock_geojson'"
                )
            if animation_payload is not None:
                beh = decompile_bedrock_animation_to_behavior(
                    animation_payload,
                    prog,
                    raw_to_safe_bone=raw_to_safe_bone,
                )
                prog = validate_program(replace(prog, behavior=beh))
            return {
                "program": prog,
                "provenance": prov,
                "license_spdx": prov.license_spdx,
                "license_policy": LOCAL_WORKSPACE_LICENSE_POLICY,
                "license_track": LOCAL_WORKSPACE_LICENSE_POLICY,
                "commercial_weights_eligible": False,
            }

        rec = ingest_external_asset(
            project,
            format_kind=format_kind,
            provenance=prov,
            animation_payload=animation_payload,
            license_policy=effective_policy,
        )
        return {
            "program": rec.program,
            "provenance": rec.provenance,
            "license_spdx": rec.provenance.license_spdx,
            "license_policy": effective_policy,
            "license_track": rec.license_track,
            "commercial_weights_eligible": bool(rec.commercial_weights_eligible and rec.provenance.license_spdx != "NOASSERTION"),
        }
    raise TypeError(f"Unsupported artist project input type {type(project).__name__}")


def ingest_artist_project_to_program(
    project: Program | str | dict[str, Any],
    *,
    format_kind: str = "blockdsl",
    animation_payload: dict[str, Any] | None = None,
    provenance: AssetProvenance | None = None,
    license_policy: str | None = None,
) -> Program:
    """Normalize an artist input (Program, Block-DSL text, .bbmodel, .json, or .geo.json) into a validated Program.

    Unlabeled inputs default to ``license_spdx="NOASSERTION"`` under ``local_workspace_unverified``
    with ``commercial_weights_eligible=False``; passing ``license_policy="clean_room_permissive"`` or
    ``"research_nc"`` without verified provenance raises ``LicenseGateError``.
    """
    bundle = ingest_artist_project_record(
        project,
        format_kind=format_kind,
        animation_payload=animation_payload,
        provenance=provenance,
        license_policy=license_policy,
    )
    return bundle["program"]


def analyze_artist_project(
    project: Program | str | dict[str, Any],
    *,
    format_kind: str = "blockdsl",
    animation_payload: dict[str, Any] | None = None,
    provenance: AssetProvenance | None = None,
    license_policy: str | None = None,
) -> ArtistCritiqueReport:
    """Perform 5-pillar physical, anatomical, musculoskeletal, surface, and motion critique."""
    program = ingest_artist_project_to_program(
        project,
        format_kind=format_kind,
        animation_payload=animation_payload,
        provenance=provenance,
        license_policy=license_policy,
    )
    scene = compile_geometry(program)
    static_audit = audit_scene(scene, enforce_physical_contact=True)

    suggestions: list[ArtistSuggestion] = []
    n_boxes = len(scene.boxes)
    n_bones = len(scene.bones)
    n_locs = len(scene.locators)
    max_depth = _compute_max_bone_depth(program.root)
    is_entity = program.meta.archetype == "entity"
    is_animated_profile = program.meta.target_profile in ANIMATED_PROFILES

    # 1. Pillar: Geometry & Contact Physics
    n_col = len(static_audit["collision_pairs"])
    n_cviol = len(static_audit["contact_violations"])
    if static_audit["passed"]:
        geom_score = 1.0
    else:
        geom_score = max(0.0, 1.0 - 0.25 * n_col - 0.20 * n_cviol)
        bad_targets = tuple(
            sorted(
                {
                    p
                    for c in static_audit["collision_pairs"]
                    for p in c["pair"]
                }
                | {
                    p
                    for v in static_audit["contact_violations"]
                    for p in v.get("pair", [])
                }
            )
        )[:8]
        suggestions.append(
            ArtistSuggestion(
                suggestion_id="fix_static_sat_and_contacts",
                priority="critical",
                category="geometry_contact",
                target_elements=bad_targets or (program.root.name,),
                title="Resolve rest-pose OBB collisions and P1-02 joint contact gaps",
                rationale=(
                    f"15-axis SAT audit detected {n_col} non-exempt OBB collision(s) and "
                    f"{n_cviol} contact/connectivity violation(s) at rest."
                ),
                proposed_upgrade_op="repair_rest_contacts",
            )
        )

    # 2. Pillar: Rig Hierarchy, Joint Limits & Semantic Sockets
    with_lim, total_non_root = _count_ast_joint_limits(program.root)
    rig_score = 1.0
    if is_entity and n_boxes >= 8 and n_bones < 3:
        rig_score -= 0.45
        suggestions.append(
            ArtistSuggestion(
                suggestion_id="articulate_flat_rig",
                priority="high",
                category="rig_hierarchy",
                target_elements=(program.root.name,),
                title="Decompose overloaded root bone into articulated skeletal chains",
                rationale=(
                    f"Entity has {n_boxes} boxes packed into only {n_bones} bone(s), preventing "
                    "limb IK, spinal wave propagation, and secondary motion."
                ),
                proposed_upgrade_op="articulate_hierarchy",
            )
        )
    if total_non_root > 0 and with_lim < total_non_root:
        lim_frac = with_lim / max(1, total_non_root)
        rig_score -= 0.25 * (1.0 - lim_frac)
        suggestions.append(
            ArtistSuggestion(
                suggestion_id="add_anatomical_joint_limits",
                priority="medium",
                category="rig_hierarchy",
                target_elements=tuple(b.name for b in scene.bones[1:5]),
                title="Add anatomical ROM joint_limits to articulating child bones",
                rationale=(
                    f"Only {with_lim}/{total_non_root} non-root AST bone(s) declare joint_limits, "
                    "leaving joints vulnerable to hyperextension during IK or procedural motion."
                ),
                proposed_upgrade_op="add_joint_limits",
            )
        )
    if is_entity and n_locs == 0:
        rig_score -= 0.20
        suggestions.append(
            ArtistSuggestion(
                suggestion_id="add_semantic_locators",
                priority="medium",
                category="rig_hierarchy",
                target_elements=(program.root.name,),
                title="Add semantic 3D locators (camera_focus, rider_seat, vfx_emitter, ik_effector)",
                rationale=(
                    "Entity declares 0 3D locators; game engines (Cobblemon/Bedrock/GeckoLib) "
                    "rely on explicit socket locators for mounts, head tracking, VFX, and hitboxes."
                ),
                proposed_upgrade_op="add_locators",
            )
        )
    rig_score = float(np.clip(rig_score, 0.0, 1.0))

    # 3. Pillar: Scientific Materials & Cross-Box Continuity
    mat_score = 1.0
    if not program.materials:
        mat_score -= 0.55
        suggestions.append(
            ArtistSuggestion(
                suggestion_id="bind_physical_materials",
                priority="high",
                category="surface_material",
                target_elements=tuple(p.name for p in program.palettes),
                title="Bind palettes to physically grounded @materials processes",
                rationale=(
                    "Program defines color ramps in @palette without @materials process bindings "
                    "(roughness, IOR, subsurface, reaction-diffusion, or wear susceptibility)."
                ),
                proposed_upgrade_op="synthesize_scientific_materials",
            )
        )
    else:
        organic_unwelded = [
            m.name
            for m in program.materials
            if m.family in ("patterned_skin", "keratin_scales") and m.weld_group is None and n_boxes >= 4
        ]
        if organic_unwelded:
            mat_score -= 0.25
            suggestions.append(
                ArtistSuggestion(
                    suggestion_id="enable_cross_box_weld_group",
                    priority="medium",
                    category="surface_material",
                    target_elements=tuple(organic_unwelded),
                    title="Enable cross-box weld_group continuity on organic skin/scale materials",
                    rationale=(
                        f"Organic material(s) {organic_unwelded} omit weld_group, causing Turing "
                        "morphogen patterns and moisture diffusion to break at adjacent box seams."
                    ),
                    proposed_upgrade_op="add_material_weld_groups",
                )
            )
    if program.style is None:
        mat_score -= 0.15
        suggestions.append(
            ArtistSuggestion(
                suggestion_id="configure_pbr_style",
                priority="low",
                category="surface_material",
                target_elements=("@style",),
                title="Add explicit @style PBR encoding and bevel highlight contract",
                rationale="Explicit @style guarantees reproducible LabPBR/glTF/vanilla quantization.",
                proposed_upgrade_op="add_style_decl",
            )
        )
    mat_score = float(np.clip(mat_score, 0.0, 1.0))

    # 4. Pillar: Musculoskeletal & Kinematic Control
    ms_score = 1.0
    if is_animated_profile and n_bones >= 3:
        if not program.muscles:
            ms_score -= 0.50
            suggestions.append(
                ArtistSuggestion(
                    suggestion_id="synthesize_musculoskeletal_units",
                    priority="high" if is_entity else "medium",
                    category="musculoskeletal",
                    target_elements=tuple(b.name for b in scene.bones[1:5]),
                    title="Add Hill-type @muscles with tendon couplings and belly bulge_inflate",
                    rationale=(
                        "Articulated rig has no @muscles declarations; antagonist muscle pairs and "
                        "biarticular tendon couplings unify limb posture and secondary volume bulge."
                    ),
                    proposed_upgrade_op="synthesize_muscles_and_kinematics",
                )
            )
        if program.kinematics is None:
            ms_score -= 0.40
            suggestions.append(
                ArtistSuggestion(
                    suggestion_id="synthesize_ik_and_look_chains",
                    priority="high" if is_entity else "medium",
                    category="musculoskeletal",
                    target_elements=tuple(b.name for b in scene.bones[1:5]),
                    title="Declare @kinematics ik_chain and look_chain descriptors",
                    rationale=(
                        "Missing @kinematics chains prevents closed-loop stance foot locking and "
                        "distributed multi-bone neck/head gaze tracking."
                    ),
                    proposed_upgrade_op="synthesize_muscles_and_kinematics",
                )
            )
    ms_score = float(np.clip(ms_score, 0.0, 1.0))

    # 5. Pillar: Dynamic @behavior Clips & Swept Stance-Slip Audit
    clips_audited = 0
    clips_passed = 0
    if not is_animated_profile:
        beh_score = 1.0
    elif program.behavior is None or not program.behavior.clips:
        beh_score = 0.15 if n_bones >= 2 else 0.70
        if n_bones >= 2:
            suggestions.append(
                ArtistSuggestion(
                    suggestion_id="synthesize_swept_behavior_clips",
                    priority="high",
                    category="behavior_motion",
                    target_elements=tuple(b.name for b in scene.bones[1:4]),
                    title="Synthesize swept-SAT & zero-skating @behavior clips and reactive states",
                    rationale=(
                        "Animated target profile has 0 @behavior clips; adding swept-audited idle/gait "
                        "clips and reactive states enables dynamic world-model simulation."
                    ),
                    proposed_upgrade_op="synthesize_behavior",
                )
            )
    else:
        for cl in program.behavior.clips:
            clips_audited += 1
            rep = audit_behavior_clip(program, cl.name, num_frames=10)
            if rep["passed"]:
                clips_passed += 1
            else:
                suggestions.append(
                    ArtistSuggestion(
                        suggestion_id=f"repair_clip_{cl.name}",
                        priority="critical",
                        category="behavior_motion",
                        target_elements=(cl.name,),
                        title=f"Repair dynamic collision or foot slip in clip {cl.name!r}",
                        rationale=(
                            f"Clip {cl.name!r} had {rep['total_collision_events']} swept SAT collision(s) "
                            f"and {rep['max_stance_slip_units']:.4f}u stance slip (limit 0.02u)."
                        ),
                        proposed_upgrade_op="repair_behavior_amplitudes",
                    )
                )
        beh_score = clips_passed / max(1, clips_audited)

    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    ranked_suggestions = tuple(
        sorted(suggestions, key=lambda s: (priority_rank.get(s.priority, 9), s.suggestion_id))
    )

    pillar_scores = {
        "geometry_contact": geom_score,
        "rig_hierarchy": rig_score,
        "surface_material": mat_score,
        "musculoskeletal": ms_score,
        "behavior_motion": beh_score,
    }
    weights = {
        "geometry_contact": 0.32,
        "rig_hierarchy": 0.18,
        "surface_material": 0.18,
        "musculoskeletal": 0.16,
        "behavior_motion": 0.16,
    }
    readiness = sum(pillar_scores[k] * weights[k] for k in weights)

    return ArtistCritiqueReport(
        readiness_score=float(np.clip(readiness, 0.0, 1.0)),
        pillar_scores=pillar_scores,
        box_count=n_boxes,
        bone_count=n_bones,
        locator_count=n_locs,
        max_bone_depth=max_depth,
        static_audit_passed=bool(static_audit["passed"]),
        behavior_clips_audited=clips_audited,
        behavior_clips_passed=clips_passed,
        suggestions=ranked_suggestions,
    )


def _infer_material_family_from_palette(name: str, lch: Vec3, is_entity: bool) -> tuple[str, float, float, str | None]:
    """Return (family, roughness, metalness, weld_group) from palette semantics & Oklch."""
    low = name.lower()
    l_val, c_val, h_val = lch
    if any(k in low for k in ("blade", "steel", "iron", "chrome", "alloy", "metal", "silver", "gold", "brass")):
        return ("brushed_metal", 0.24, 1.0, None)
    if any(k in low for k in ("armor", "plate", "mech", "hull", "enamel", "painted")):
        return ("painted_metal", 0.32, 0.0, None)
    if any(k in low for k in ("wing", "carapace", "elytra", "crystal", "irid", "gem", "shell")):
        return ("thin_film_iridescence", 0.20, 0.0, "carapace_continuum" if is_entity else None)
    if any(k in low for k in ("scale", "wyvern", "drake", "horn", "claw", "osteoderm", "fang", "bone")):
        return ("keratin_scales", 0.38, 0.0, "dermis_continuum" if is_entity else None)
    if any(k in low for k in ("skin", "hide", "belly", "flesh", "body", "pelt", "fur", "coat")):
        return ("patterned_skin", 0.52, 0.0, "dermis_continuum")
    if any(k in low for k in ("wood", "timber", "oak", "handle", "shaft", "bark", "botanical")):
        return ("solid_wood", 0.58, 0.0, None)
    if any(k in low for k in ("stone", "rock", "slate", "basalt", "brick", "relic")):
        return ("weathered_stone", 0.72, 0.0, None)
    if any(k in low for k in ("cloth", "strap", "wrap", "fabric", "cape", "banner")):
        return ("woven_fabric", 0.68, 0.0, None)
    if any(k in low for k in ("moss", "lichen", "vine", "leaf", "bio")):
        return ("lichen_biofilm", 0.76, 0.0, None)
    if is_entity:
        return ("patterned_skin" if c_val >= 0.06 else "keratin_scales", 0.45, 0.0, "dermis_continuum")
    return ("painted_metal" if l_val > 0.45 else "polymer_dielectric", 0.38, 0.0, None)


def _add_joint_limits_and_locators(program: Program) -> Program:
    """Populate missing anatomical joint_limits and semantic 3D locators cleanly."""
    scene = compile_geometry(program)
    existing_ids: set[str] = (
        {b.name for b in scene.bones}
        | {bx.name for bx in scene.boxes}
        | {loc.name for loc in scene.locators}
        | {p.name for p in program.palettes}
        | {m.name for m in program.muscles}
    )
    if program.kinematics is not None:
        existing_ids |= {c.name for c in program.kinematics.ik_chains} | {c.name for c in program.kinematics.look_chains}
    if program.behavior is not None:
        existing_ids |= {c.name for c in program.behavior.clips} | {s.name for s in program.behavior.states}

    def _fresh(base: str) -> str:
        cand = base
        idx = 2
        while cand in existing_ids:
            cand = f"{base}_{idx}"
            idx += 1
        existing_ids.add(cand)
        return cand

    need_locs = len(scene.locators) == 0 and program.meta.archetype in ("entity", "weapon_3d", "block_multipart")
    added_root_loc = False

    def _upgrade_bone(b: Bone, is_root: bool) -> Bone:
        nonlocal added_root_loc
        lim = b.joint_limits
        if not is_root and lim is None:
            low = b.name.lower()
            if "wing" in low:
                lim = ((-55.0, 55.0), (-45.0, 45.0), (-75.0, 75.0))
            elif any(k in low for k in ("jaw", "lid", "door", "hinge")):
                lim = ((-45.0, 55.0), (-15.0, 15.0), (-15.0, 15.0))
            elif any(k in low for k in ("neck", "head", "tail", "spine")):
                lim = ((-45.0, 45.0), (-55.0, 55.0), (-35.0, 35.0))
            else:
                lim = ((-65.0, 65.0), (-45.0, 45.0), (-55.0, 55.0))

        new_items: list[Box | Locator | Bone] = []
        first_box: Box | None = None
        for it in b.items:
            if isinstance(it, Box):
                if first_box is None:
                    first_box = it
                new_items.append(it)
            elif isinstance(it, Locator):
                new_items.append(it)
            elif isinstance(it, Bone):
                new_items.append(_upgrade_bone(it, is_root=False))

        # Add semantic locators to root & head/jaw/tip bones when absent
        if need_locs and first_box is not None:
            low = b.name.lower()
            if is_root and not added_root_loc:
                added_root_loc = True
                role = "rider_seat" if program.meta.archetype == "entity" else "socket"
                loc_y = round(float(first_box.size[1]) * 0.5, 2)
                # Place after boxes, before child bones so AST order stays canonical
                box_part = [x for x in new_items if not isinstance(x, Bone)]
                bone_part = [x for x in new_items if isinstance(x, Bone)]
                box_part.append(
                    Locator(
                        name=_fresh("mount_seat_socket" if role == "rider_seat" else "primary_socket"),
                        offset=(0.0, loc_y, 0.0),
                        role=role,
                    )
                )
                new_items = [*box_part, *bone_part]
            elif any(k in low for k in ("head", "skull", "cranium")):
                box_part = [x for x in new_items if not isinstance(x, Bone)]
                bone_part = [x for x in new_items if isinstance(x, Bone)]
                box_part.append(
                    Locator(
                        name=_fresh(f"{b.name}_focus"),
                        offset=(0.0, round(float(first_box.size[1]) * 0.25, 2), round(float(first_box.size[2]) * 0.5, 2)),
                        role="camera_focus",
                    )
                )
                new_items = [*box_part, *bone_part]
            elif any(k in low for k in ("jaw", "snout", "muzzle", "emitter", "cannon")):
                box_part = [x for x in new_items if not isinstance(x, Bone)]
                bone_part = [x for x in new_items if isinstance(x, Bone)]
                box_part.append(
                    Locator(
                        name=_fresh(f"{b.name}_vfx"),
                        offset=(0.0, 0.0, round(float(first_box.size[2]) * 0.5, 2)),
                        role="vfx_emitter",
                    )
                )
                new_items = [*box_part, *bone_part]

        return replace(b, items=new_items, joint_limits=lim)

    return validate_program(replace(program, root=_upgrade_bone(program.root, is_root=True)))


def _upgrade_scientific_materials_and_style(program: Program) -> Program:
    """Synthesize or upgrade @materials (with cross-box weld_groups) and @style."""
    is_entity = program.meta.archetype == "entity"
    existing_by_name = {m.name: m for m in program.materials}
    upgraded_mats: list[MaterialDecl] = []

    for idx, pal in enumerate(program.palettes):
        if pal.name in existing_by_name:
            m = existing_by_name[pal.name]
            if m.family in ("patterned_skin", "keratin_scales") and m.weld_group is None:
                m = replace(m, weld_group="dermis_continuum")
            upgraded_mats.append(m)
        else:
            fam, rough, metal, wg = _infer_material_family_from_palette(pal.name, pal.base_lch, is_entity)
            upgraded_mats.append(
                MaterialDecl(
                    name=pal.name,
                    family=fam,
                    roughness=rough,
                    metalness=metal,
                    coating=0.55 if fam == "painted_metal" else 0.0,
                    ior=1.52,
                    scale=1.15,
                    seed=101 + idx * 37,
                    anisotropy=1.6 if fam in ("brushed_metal", "solid_wood", "keratin_scales") else 1.0,
                    subsurface=0.35 if fam in ("patterned_skin", "keratin_scales") else 0.0,
                    porosity=0.28 if fam in ("weathered_stone", "solid_wood", "woven_fabric") else 0.08,
                    emissive=1.0 if pal.emissive else 0.0,
                    wear=0.42 if fam in ("painted_metal", "keratin_scales") else 0.25,
                    oxidation=0.32 if fam in ("painted_metal", "brushed_metal") else 0.05,
                    biofilm=0.24 if fam in ("weathered_stone", "solid_wood") else 0.10,
                    weld_group=wg,
                )
            )

    style = program.style or StyleDecl(
        target_encoding="labpbr_1_3",
        quantize_ramp=False,
        bevel_highlight=0.10,
        view_deg=22.0,
    )
    return validate_program(replace(program, materials=tuple(upgraded_mats), style=style))


def _synthesize_muscles_kinematics_and_behavior(program: Program) -> Program:
    """Synthesize SAT-verified @muscles, @kinematics, and @behavior on an articulated Program."""
    if program.meta.target_profile not in ANIMATED_PROFILES:
        return program

    scene = compile_geometry(program)
    if len(scene.bones) < 2:
        return program

    root_name = program.root.name
    bone_map = scene.bone_by_name
    children_of: dict[str, list[str]] = {b.name: [] for b in scene.bones}
    boxes_of: dict[str, list[str]] = {b.name: [] for b in scene.bones}
    for b in scene.bones:
        if b.parent_name is not None and b.parent_name in children_of:
            children_of[b.parent_name].append(b.name)
    for bx in scene.boxes:
        boxes_of[bx.bone_name].append(bx.name)

    used_names: set[str] = (
        set(bone_map)
        | {bx.name for bx in scene.boxes}
        | {loc.name for loc in scene.locators}
        | {p.name for p in program.palettes}
    )

    def _claim_id(base: str) -> str:
        cand = base
        k = 2
        while cand in used_names:
            cand = f"{base}_{k}"
            k += 1
        used_names.add(cand)
        return cand

    # Discover 2-bone limb/wing chains and multi-bone neck/tail/spine chains
    ik_chains: list[IKChainDecl] = list(program.kinematics.ik_chains) if program.kinematics else []
    look_chains: list[LookChainDecl] = list(program.kinematics.look_chains) if program.kinematics else []
    leg_ik_names: list[tuple[str, float]] = []

    if not ik_chains:
        for b in scene.bones:
            if b.name == root_name:
                continue
            low = b.name.lower()
            is_limb = any(tok in low for tok in ("leg", "thigh", "femur", "hip", "arm", "humerus", "foreleg", "hindleg"))
            ch_list = children_of.get(b.name, [])
            if is_limb and ch_list:
                lo_name = ch_list[0]
                lo_bone = bone_map[lo_name]
                # Distal effector vector in lower bone's local frame
                lo_boxes = [scene.box_by_name[n] for n in boxes_of.get(lo_name, [])]
                if lo_boxes:
                    bx0 = lo_boxes[0]
                    eff_vec = np.asarray(bx0.local_center, dtype=np.float64) * 1.65
                    if float(np.linalg.norm(eff_vec)) < 0.5:
                        eff_vec = np.array([0.0, -float(bx0.size[1]), 0.0], dtype=np.float64)
                else:
                    eff_vec = np.array([0.0, -2.0, 0.0], dtype=np.float64)
                ik_name = _claim_id(f"ik_{b.name}")
                ik_chains.append(
                    IKChainDecl(
                        name=ik_name,
                        upper=b.name,
                        lower=lo_name,
                        effector=(round(float(eff_vec[0]), 3), round(float(eff_vec[1]), 3), round(float(eff_vec[2]), 3)),
                        hinge_axis="x",
                    )
                )
                if any(tok in low for tok in ("leg", "thigh", "femur", "hip", "foreleg", "hindleg")):
                    phase = 180.0 if lo_bone.mirrored else 0.0
                    if "hind" in low or "rear" in low or "back" in low:
                        phase = (phase + 90.0) % 360.0
                    leg_ik_names.append((ik_name, phase))

    if not look_chains:
        neck_head = [
            b.name
            for b in scene.bones
            if b.name != root_name and not b.mirrored and any(k in b.name.lower() for k in ("neck", "head", "skull"))
        ][:4]
        if neck_head:
            look_chains.append(LookChainDecl(name=_claim_id("gaze_chain"), bones=tuple(neck_head), weight_decay=0.72))
        elif len(scene.bones) >= 2:
            look_chains.append(LookChainDecl(name=_claim_id("posture_chain"), bones=(scene.bones[1].name,), weight_decay=0.8))

    kin_decl = KinematicsDecl(ik_chains=tuple(ik_chains), look_chains=tuple(look_chains))

    # Synthesize SAT-safe Hill-type @muscles (testing bulge_inflate via audit_scene!)
    muscles: list[MuscleDecl] = list(program.muscles)
    if not muscles:
        candidate_bones = [b for b in scene.bones if b.name != root_name][:6]
        for b in candidate_bones:
            parent_b = b.parent_name or root_name
            low = b.name.lower()
            ax = "z" if "wing" in low else ("y" if "tail" in low else "x")
            sign_m = -1.0 if (b.mirrored and ax in ("y", "z")) else 1.0
            belly = boxes_of[b.name][0] if boxes_of.get(b.name) else None
            ch_list = children_of.get(b.name, [])
            tendon = TendonCoupling(bone=ch_list[0], axis=ax, ratio=-0.55) if ch_list else None

            # Test candidate muscle excursion & belly bulge against 15-axis SAT
            chosen_arm = 0.0
            chosen_bulge = 0.0
            for arm_try, bulge_try in ((16.0, 0.08), (10.0, 0.05), (5.0, 0.02), (2.0, 0.0)):
                cand_m = MuscleDecl(
                    name=_claim_id(f"m_{b.name}_flex"),
                    origin_bone=parent_b,
                    insertion_bone=b.name,
                    action_axis=ax,
                    moment_arm_deg=round(sign_m * arm_try, 2),
                    tendon_coupling=tendon,
                    belly_box=belly if bulge_try > 0 else None,
                    bulge_inflate=bulge_try if belly else 0.0,
                )
                used_names.discard(cand_m.name)
                temp_prog = replace(program, muscles=(*muscles, cand_m), kinematics=kin_decl)
                res_m = evaluate_muscles_at_activations(temp_prog, {cand_m.name: 1.0})
                aud_m = audit_scene(res_m.scene, enforce_physical_contact=False)
                if not aud_m["collision_pairs"]:
                    chosen_arm = arm_try
                    chosen_bulge = bulge_try if belly else 0.0
                    _claim_id(cand_m.name)
                    muscles.append(cand_m)
                    break

    prog_with_ms = validate_program(replace(program, muscles=tuple(muscles), kinematics=kin_decl))

    # Synthesize swept-SAT & zero-skating @behavior clips + reactive state
    if prog_with_ms.behavior is not None and prog_with_ms.behavior.clips:
        # Repair any failing clip amplitudes if needed
        repaired_clips: list[ClipDecl] = []
        for cl in prog_with_ms.behavior.clips:
            rep = audit_behavior_clip(prog_with_ms, cl.name, num_frames=10)
            if rep["passed"]:
                repaired_clips.append(cl)
            else:
                scaled_cl = cl
                for scale in (0.6, 0.35, 0.15, 0.0):
                    cand_cl = replace(
                        cl,
                        tracks=tuple(replace(t, amp_deg=round(t.amp_deg * scale, 3), bias_deg=round(t.bias_deg * scale, 3)) for t in cl.tracks),
                        muscle_drives=tuple(replace(md, peak=round(md.peak * scale, 3), baseline=round(md.baseline * scale, 3)) for md in cl.muscle_drives),
                        ik_drives=tuple(replace(ikd, stride=round(ikd.stride * scale, 3), lift=round(ikd.lift * scale, 3)) for ikd in cl.ik_drives),
                    )
                    cand_beh = replace(prog_with_ms.behavior, clips=(cand_cl,))
                    cand_p = replace(prog_with_ms, behavior=cand_beh)
                    if audit_behavior_clip(cand_p, cand_cl.name, num_frames=10)["passed"]:
                        scaled_cl = cand_cl
                        break
                repaired_clips.append(scaled_cl)
        return validate_program(replace(prog_with_ms, behavior=replace(prog_with_ms.behavior, clips=tuple(repaired_clips))))

    dur = 1.6
    freq = round(1.0 / dur, 4)

    # Build idle/action clip and binary-scale amplitudes until 100% swept-SAT & stance-IK clean
    tracks_cand: list[TrackDecl] = []
    for idx_b, b in enumerate(scene.bones[1:7]):
        low = b.name.lower()
        if "wing" in low:
            sgn = -1.0 if b.mirrored else 1.0
            tracks_cand.append(TrackDecl(bone=b.name, law="flap", axis="z", amp_deg=18.0 * sgn, freq_hz=freq, phase_deg=0.0))
        elif "tail" in low or "spine" in low or "neck" in low:
            sgn = -1.0 if b.mirrored else 1.0
            tracks_cand.append(
                TrackDecl(bone=b.name, law="wave", axis="y", amp_deg=10.0 * sgn, freq_hz=freq, phase_deg=round((idx_b * -35.0) % 360.0, 1))
            )
        elif not any(ik.upper == b.name or ik.lower == b.name for ik in ik_chains):
            tracks_cand.append(TrackDecl(bone=b.name, law="hinge", axis="x", amp_deg=8.0, freq_hz=freq, phase_deg=0.0))

    m_drives_cand = [
        MuscleDriveDecl(muscle=m.name, peak=0.75, freq_hz=freq, phase_deg=round((i * 90.0) % 360.0, 1), baseline=0.05)
        for i, m in enumerate(muscles[:4])
    ]
    ik_drives_cand = [
        IKDriveDecl(chain=ik_n, stride=1.6, lift=0.8, phase_deg=ph)
        for ik_n, ph in leg_ik_names[:6]
    ]

    if not tracks_cand and not m_drives_cand and not ik_drives_cand:
        tracks_cand.append(TrackDecl(bone=scene.bones[1].name, law="hinge", axis="x", amp_deg=5.0, freq_hz=freq))

    clip_id = _claim_id("locomotion_cycle" if ik_drives_cand else "active_cycle")
    chosen_clip: ClipDecl | None = None
    for scale in (1.0, 0.70, 0.45, 0.25, 0.10, 0.0):
        cand_clip = ClipDecl(
            name=clip_id,
            duration_s=dur,
            loop=True,
            tracks=tuple(replace(t, amp_deg=round(t.amp_deg * scale, 3)) for t in tracks_cand),
            muscle_drives=tuple(replace(md, peak=round(md.peak * scale, 3), baseline=round(min(md.baseline, md.peak * scale), 3)) for md in m_drives_cand),
            ik_drives=tuple(replace(ikd, stride=round(ikd.stride * scale, 3), lift=round(ikd.lift * scale, 3)) for ikd in ik_drives_cand),
        )
        test_prog = replace(prog_with_ms, behavior=BehaviorDecl(clips=(cand_clip,)))
        if audit_behavior_clip(test_prog, clip_id, num_frames=12)["passed"]:
            chosen_clip = cand_clip
            break
    if chosen_clip is None:
        chosen_clip = ClipDecl(
            name=clip_id,
            duration_s=dur,
            loop=True,
            tracks=(TrackDecl(bone=scene.bones[1].name, law="wave", axis="y", amp_deg=0.0, freq_hz=freq),),
        )

    # Reactive entity/asset state (e.g. Monster Hunter enraged / charged mode)
    first_mat = prog_with_ms.palettes[0].name
    state_tones = tuple((m.name, 0.55) for m in muscles[:2])
    state_decl = BehaviorStateDecl(
        name=_claim_id("awakened_state"),
        pose_biases=() if state_tones else ((scene.bones[1].name, (0.0, 0.0, 0.0)),),
        muscle_tones=state_tones,
        material_mods=(MaterialModDecl(material=first_mat, emissive=0.45, moisture=0.25, wear_delta=0.05),),
    )

    beh_decl = BehaviorDecl(clips=(chosen_clip,), states=(state_decl,))
    return validate_program(replace(prog_with_ms, behavior=beh_decl))


def _repair_rest_geometry_if_needed(program: Program) -> Program:
    """Repair rest-pose OBB collisions or separated declared contacts via bounded rule adjustments."""
    sc = compile_geometry(program)
    aud = audit_scene(sc, enforce_physical_contact=True)
    if aud["passed"]:
        return program

    # Zero out excessive rest_euler collisions or clamp detached/offset drifts on child bones
    def _relax_bone(b: Bone, is_root: bool, scale_euler: float) -> Bone:
        new_items = [_relax_bone(it, False, scale_euler) if isinstance(it, Bone) else it for it in b.items]
        if is_root:
            return replace(b, items=new_items)
        att = b.attach
        if att is not None and not att.detached:
            att = replace(att, offset=(0.0, 0.0, 0.0))
        re = (
            round(b.rest_euler[0] * scale_euler, 3),
            round(b.rest_euler[1] * scale_euler, 3),
            round(b.rest_euler[2] * scale_euler, 3),
        )
        return replace(b, rest_euler=re, attach=att, items=new_items)

    for s_factor in (1.0, 0.65, 0.35, 0.0):
        cand = validate_program(replace(program, root=_relax_bone(program.root, True, s_factor)))
        if audit_scene(compile_geometry(cand), enforce_physical_contact=True)["passed"]:
            return cand
    return program


def _articulate_flat_hierarchy(program: Program) -> Program:
    """Decompose an overloaded flat root bone into articulated anatomical child chains with zero vertex drift."""
    scene = compile_geometry(program)
    root = program.root
    direct_boxes = [it for it in root.items if isinstance(it, Box)]
    if len(direct_boxes) <= 1:
        return program

    used_ids: set[str] = (
        {b.name for b in scene.bones}
        | {bx.name for bx in scene.boxes}
        | {loc.name for loc in scene.locators}
        | {p.name for p in program.palettes}
        | {m.name for m in program.muscles}
    )
    if program.kinematics is not None:
        used_ids |= {c.name for c in program.kinematics.ik_chains}
        used_ids |= {c.name for c in program.kinematics.look_chains}
    if program.behavior is not None:
        used_ids |= {c.name for c in program.behavior.clips}
        used_ids |= {s.name for s in program.behavior.states}

    def _fresh_bone_id(box_name: str) -> str:
        base = f"bone_{box_name}"
        cand = base
        idx = 2
        while cand in used_ids:
            cand = f"{base}_{idx}"
            idx += 1
        used_ids.add(cand)
        return cand

    primary_box = direct_boxes[0]
    direct_order = [bx.name for bx in direct_boxes]
    direct_set = set(direct_order)
    root_rot = scene.bone_by_name[root.name].rotation

    children_boxes_of: dict[str, list[Box]] = {bx.name: [] for bx in direct_boxes}
    for idx in range(1, len(direct_boxes)):
        bx = direct_boxes[idx]
        w_c = scene.box_by_name[bx.name]
        nom_parent = (
            bx.place.base_box
            if (bx.place is not None and bx.place.base_box in direct_set and direct_order.index(bx.place.base_box) < idx)
            else primary_box.name
        )
        sat_nom = sat_obb(scene.box_by_name[nom_parent], w_c)
        chosen_parent = nom_parent
        if not (-1e-4 <= sat_nom.raw_min_overlap <= 0.5):
            best_gap = float("inf")
            for prev_name in direct_order[:idx]:
                sat_cand = sat_obb(scene.box_by_name[prev_name], w_c)
                if -1e-4 <= sat_cand.raw_min_overlap <= 0.5:
                    gap_score = abs(sat_cand.raw_min_overlap)
                    if gap_score < best_gap:
                        best_gap = gap_score
                        chosen_parent = prev_name
        children_boxes_of[chosen_parent].append(bx)

    existing_bones_by_parent: dict[str, list[Bone]] = {}
    root_kept_bones: list[Bone] = []
    for it in root.items:
        if isinstance(it, Bone):
            if it.attach is not None and it.attach.parent_box in direct_set and it.attach.parent_box != primary_box.name:
                existing_bones_by_parent.setdefault(it.attach.parent_box, []).append(it)
            else:
                root_kept_bones.append(it)

    def _build_articulated_subbone(parent_box_name: str, child_box: Box) -> Bone:
        w_p = scene.box_by_name[parent_box_name]
        w_c = scene.box_by_name[child_box.name]
        diff = root_rot.T @ (w_c.center - w_p.center)
        s_p = w_p.size
        s_c = w_c.size
        norm_sep = np.abs(diff) / np.maximum(1e-6, 0.5 * (s_p + s_c))
        dom = int(np.argmax(norm_sep))

        a_arr = np.zeros(3, dtype=np.float64)
        al_arr = np.zeros(3, dtype=np.float64)
        off_arr = np.zeros(3, dtype=np.float64)
        sgn = 1.0 if float(diff[dom]) >= 0.0 else -1.0
        a_arr[dom] = sgn
        al_arr[dom] = -sgn
        off_arr[dom] = float(diff[dom]) - 0.5 * float(s_p[dom]) * sgn - 0.5 * float(s_c[dom]) * sgn

        for j in range(3):
            if j == dom:
                continue
            a_arr[j] = float(np.clip(diff[j] / max(1e-6, 0.5 * float(s_p[j])), -1.0, 1.0))
            rem_j = float(diff[j]) - 0.5 * float(s_p[j]) * a_arr[j]
            al_arr[j] = float(np.clip(-rem_j / max(1e-6, 0.5 * float(s_c[j])), -1.0, 1.0))
            off_arr[j] = rem_j + 0.5 * float(s_c[j]) * al_arr[j]

        sat_pair = sat_obb(w_p, w_c)
        pair_key = (w_p.name, w_c.name) if w_p.name <= w_c.name else (w_c.name, w_p.name)
        contact_err = _evaluate_declared_contact(
            w_p,
            w_c,
            pair_key,
            "attach",
            sat_pair.raw_min_overlap,
            collision_tolerance=1e-7,
            max_contact_gap=1e-4,
            max_inset_depth=0.5,
        )
        can_connect = (contact_err is None) and all(abs(float(x)) <= 0.5 for x in off_arr)

        att = Attach(
            parent_box=parent_box_name,
            anchor=(float(a_arr[0]), float(a_arr[1]), float(a_arr[2])),
            align=(float(al_arr[0]), float(al_arr[1]), float(al_arr[2])),
            offset=(float(off_arr[0]), float(off_arr[1]), float(off_arr[2])),
            detached=not can_connect,
        )
        moved_box = replace(child_box, place=None)
        sub_items: list[Box | Locator | Bone] = [moved_box]
        for grand_bx in children_boxes_of.get(child_box.name, []):
            sub_items.append(_build_articulated_subbone(child_box.name, grand_bx))
        sub_items.extend(existing_bones_by_parent.get(child_box.name, []))
        return Bone(
            name=_fresh_bone_id(child_box.name),
            rest_euler=(0.0, 0.0, 0.0),
            attach=att,
            items=sub_items,
        )

    new_root_items: list[Box | Locator | Bone] = [primary_box]
    new_root_items.extend(it for it in root.items if isinstance(it, Locator))
    for ch_bx in children_boxes_of[primary_box.name]:
        new_root_items.append(_build_articulated_subbone(primary_box.name, ch_bx))
    new_root_items.extend(root_kept_bones)

    return validate_program(replace(program, root=replace(root, items=new_root_items)))


def _apply_single_upgrade_op(program: Program, op: str) -> Program:
    if op == "repair_rest_contacts":
        return _repair_rest_geometry_if_needed(program)
    if op == "articulate_hierarchy":
        return _add_joint_limits_and_locators(_articulate_flat_hierarchy(program))
    if op in ("add_joint_limits", "add_locators"):
        return _add_joint_limits_and_locators(program)
    if op in ("synthesize_scientific_materials", "add_material_weld_groups", "add_style_decl"):
        return _upgrade_scientific_materials_and_style(program)
    if op in ("synthesize_muscles_and_kinematics", "synthesize_behavior", "repair_behavior_amplitudes"):
        return _synthesize_muscles_kinematics_and_behavior(program)
    return program


def enhance_artist_project_step(
    program: Program,
    *,
    seen_dsls: set[str] | None = None,
    precomputed_critique: ArtistCritiqueReport | None = None,
) -> tuple[Program, str]:
    """Execute one deterministic rule-grounded enhancement step guided by `analyze_artist_project`."""
    critique = precomputed_critique if precomputed_critique is not None else analyze_artist_project(program)
    if not critique.suggestions:
        return program, "noop_already_optimal"

    cur_dsl = format_program(program)
    blocked = set(seen_dsls or ()) | {cur_dsl}
    tried_ops: set[str] = set()

    for sug in critique.suggestions:
        op = sug.proposed_upgrade_op
        if op in tried_ops:
            continue
        tried_ops.add(op)
        cand = _apply_single_upgrade_op(program, op)
        if format_program(cand) not in blocked:
            return cand, op

    return program, "noop_converged"


def run_looped_artist_enhancement(
    project: Program | str | dict[str, Any],
    *,
    format_kind: str = "blockdsl",
    animation_payload: dict[str, Any] | None = None,
    provenance: AssetProvenance | None = None,
    license_policy: str | None = None,
    max_iterations: int = 4,
) -> dict[str, Any]:
    """Run closed-loop critique -> rule upgrade -> re-audit iterations on an artist project."""
    intake_bundle = ingest_artist_project_record(
        project,
        format_kind=format_kind,
        animation_payload=animation_payload,
        provenance=provenance,
        license_policy=license_policy,
    )
    cur_prog: Program = intake_bundle["program"]
    initial_critique = analyze_artist_project(cur_prog)
    cur_critique = initial_critique
    history: list[dict[str, Any]] = [
        {
            "iteration": 0,
            "applied_op": "initial_intake",
            "critique": initial_critique.to_dict(),
        }
    ]

    seen_dsls = {format_program(cur_prog)}
    for step_i in range(1, max(1, max_iterations) + 1):
        next_prog, op_used = enhance_artist_project_step(
            cur_prog,
            seen_dsls=seen_dsls,
            precomputed_critique=cur_critique,
        )
        if op_used.startswith("noop"):
            break
        next_dsl = format_program(next_prog)
        next_critique = analyze_artist_project(next_prog)
        history.append(
            {
                "iteration": step_i,
                "applied_op": op_used,
                "critique": next_critique.to_dict(),
            }
        )
        cur_prog = next_prog
        cur_critique = next_critique
        seen_dsls.add(next_dsl)
        if not next_critique.suggestions:
            break

    final_critique = cur_critique
    return {
        "initial_readiness_score": round(initial_critique.readiness_score, 4),
        "final_readiness_score": round(final_critique.readiness_score, 4),
        "readiness_delta": round(final_critique.readiness_score - initial_critique.readiness_score, 4),
        "iterations_run": len(history) - 1,
        "final_program": cur_prog,
        "final_program_dsl": format_program(cur_prog),
        "final_critique": final_critique.to_dict(),
        "license_spdx": intake_bundle["license_spdx"],
        "license_track": intake_bundle["license_track"],
        "commercial_weights_eligible": intake_bundle["commercial_weights_eligible"],
        "history": history,
    }


def _bone_subtree_has_box(bone: Bone) -> bool:
    for it in bone.items:
        if isinstance(it, Box):
            return True
        if isinstance(it, Bone) and _bone_subtree_has_box(it):
            return True
    return False


def _strip_locators_and_limits_from_bone(bone: Bone) -> Bone:
    boxes_in_bone = [it for it in bone.items if isinstance(it, Box)]
    locators_in_bone = [it for it in bone.items if isinstance(it, Locator)]
    stripped_children: dict[str, Bone] = {}
    for it in bone.items:
        if isinstance(it, Bone):
            stripped_child = _strip_locators_and_limits_from_bone(it)
            if _bone_subtree_has_box(stripped_child):
                stripped_children[it.name] = stripped_child

    referenced_locators = {
        ch.attach.parent_box
        for ch in stripped_children.values()
        if ch.attach is not None
    }
    need_structural_locator = (not boxes_in_bone) and bool(locators_in_bone)
    preserved_loc_names = set(referenced_locators)
    if need_structural_locator and not (preserved_loc_names & {loc.name for loc in locators_in_bone}):
        preserved_loc_names.add(locators_in_bone[0].name)

    items: list[Box | Locator | Bone] = []
    for it in bone.items:
        if isinstance(it, Box):
            items.append(it)
        elif isinstance(it, Locator) and it.name in preserved_loc_names:
            items.append(it)
        elif isinstance(it, Bone) and it.name in stripped_children:
            items.append(stripped_children[it.name])
    return replace(bone, items=items, joint_limits=None)


def build_progressive_from_scratch_trajectory(
    master_program: Program,
) -> tuple[ProgressiveStageRecord, ...]:
    """Construct a 5-stage rule-governed progressive refinement trajectory from rough sketch to masterwork.

    Every stage (Stage 0 -> Stage 4) is an explicit, deterministic, 100% SAT + P1-02 audited
    Block-DSL program—modeling how a skilled creature/game-asset artist blocks out primary
    masses, adds skeletal joints, sculpts secondary anatomy & sockets, applies scientific
    surface materials, and binds musculoskeletal animation without black-box diffusion noise.
    """
    from synccraft.data_gen.curriculum_and_assistant import (
        _prune_unused_palettes_and_materials,
        coarsen_program_to_tier,
    )

    full_upgraded = _synthesize_muscles_kinematics_and_behavior(
        _upgrade_scientific_materials_and_style(
            _add_joint_limits_and_locators(_repair_rest_geometry_if_needed(master_program))
        )
    )

    # Stage 2: Full geometry + joint_limits + semantic locators (pure clay/palette, pre-materials/motion)
    stage_2_prog = validate_program(
        replace(
            full_upgraded,
            materials=(),
            style=None,
            muscles=(),
            kinematics=None,
            behavior=None,
        )
    )

    # Clean rig without locators/limits for Stage 0 and Stage 1 silhouette coarsening
    bare_rig_prog = _prune_unused_palettes_and_materials(
        replace(stage_2_prog, root=_strip_locators_and_limits_from_bone(stage_2_prog.root))
    )

    # Stage 0: Rough conceptual gesture blockout (tier_1_simple primary masses, single clay/base palette)
    stage_0_prog = validate_program(coarsen_program_to_tier(bare_rig_prog, "tier_1_simple"))

    # Stage 1: Articulated skeletal hierarchy (tier_2_medium primary limb/wing/tail chains)
    stage_1_prog = validate_program(coarsen_program_to_tier(bare_rig_prog, "tier_2_medium"))

    # Stage 3: Full anatomy + scientific welded @materials + @style
    stage_3_prog = validate_program(
        replace(
            full_upgraded,
            muscles=(),
            kinematics=None,
            behavior=None,
        )
    )

    # Stage 4: Full musculoskeletal @muscles + @kinematics + swept-SAT @behavior masterwork
    stage_4_prog = full_upgraded

    raw_stages = [
        (
            0,
            "stage_0_conceptual_gesture_blockout",
            "Rough primary-mass silhouette blockout establishing core proportions and ground/root frame.",
            stage_0_prog,
        ),
        (
            1,
            "stage_1_articulated_skeletal_hierarchy",
            "Primary skeletal chain subdivision with bilateral mirror_x limb/wing branching.",
            stage_1_prog,
        ),
        (
            2,
            "stage_2_secondary_anatomy_and_sockets",
            "Full secondary anatomy (osteoderms, phalanges, sub-assemblies), ROM joint_limits, and 3D locators.",
            stage_2_prog,
        ),
        (
            3,
            "stage_3_scientific_surface_materials",
            "Physically grounded @materials processes with cross-box weld_group continuity and LabPBR @style.",
            stage_3_prog,
        ),
        (
            4,
            "stage_4_musculoskeletal_and_behavior_masterwork",
            "Hill-type @muscles with tendon couplings & belly bulge, @kinematics IK/gaze chains, and swept-SAT @behavior.",
            stage_4_prog,
        ),
    ]

    records: list[ProgressiveStageRecord] = []
    for idx, sname, desc, prog in raw_stages:
        sc = compile_geometry(prog)
        aud = audit_scene(sc, enforce_physical_contact=True)
        crit = analyze_artist_project(prog)
        records.append(
            ProgressiveStageRecord(
                stage_index=idx,
                stage_name=sname,
                description=desc,
                program=prog,
                program_dsl=format_program(prog),
                box_count=len(sc.boxes),
                bone_count=len(sc.bones),
                static_audit_passed=bool(aud["passed"]),
                readiness_score=crit.readiness_score,
            )
        )
    return tuple(records)


def export_artist_project_bundle(
    program: Program,
    *,
    export_format: str | None = None,
    atlas: dict[str, Any] | None = None,
    name: str = "SyncCraft enhanced asset",
    identifier: str = "geometry.synccraft.enhanced",
    num_animation_frames: int = 12,
    animation_mode: str = "keyframes",
) -> dict[str, Any]:
    """Export an enhanced Program to Bedrock, GeckoLib, or animated Blockbench 5.0 with roundtrip re-audit."""
    from synccraft.backends.bbmodel5_exporter import export_bbmodel
    from synccraft.backends.bedrock_gecko_exporter import export_bedrock, export_geckolib

    prog = validate_program(program)
    scene = compile_geometry(prog)
    static_audit = audit_scene(scene, enforce_physical_contact=True)

    target_fmt = export_format or prog.meta.target_profile
    if target_fmt not in ("bedrock_entity", "geckolib", "bbmodel_5"):
        target_fmt = "bbmodel_5"

    clips_total = len(prog.behavior.clips) if prog.behavior is not None else 0
    clips_passed = 0
    if prog.behavior is not None:
        for cl in prog.behavior.clips:
            if audit_behavior_clip(prog, cl.name, num_frames=max(6, num_animation_frames))["passed"]:
                clips_passed += 1

    rt_prov = AssetProvenance(
        asset_id="roundtrip_reaudit",
        source_url="local://synccraft_roundtrip_reaudit",
        author="synccraft_auditor",
        license_spdx="Apache-2.0",
        attribution_notice="Internal post-export roundtrip verification",
    )

    if target_fmt == "bedrock_entity":
        bundle = export_bedrock(
            scene,
            program=prog,
            atlas=atlas,
            identifier=identifier,
            profile="bedrock_entity",
            num_animation_frames=num_animation_frames,
            animation_mode=animation_mode,
        )
        rt_rec = ingest_external_asset(
            bundle["geometry"],
            "bedrock_geojson",
            rt_prov,
            animation_payload=bundle["animations"],
        )
        rt_err = max(float(bundle["serialized_vertex_max_error"]), float(rt_rec.vertex_reconstruction_error))
    elif target_fmt == "geckolib":
        bundle = export_geckolib(
            scene,
            program=prog,
            atlas=atlas,
            identifier=identifier,
            profile="geckolib",
            num_animation_frames=num_animation_frames,
            animation_mode=animation_mode,
        )
        rt_rec = ingest_external_asset(
            bundle["geometry"],
            "bedrock_geojson",
            rt_prov,
            animation_payload=bundle["animations"],
        )
        rt_err = max(float(bundle["serialized_vertex_max_error"]), float(rt_rec.vertex_reconstruction_error))
    else:
        bb_doc = export_bbmodel(
            scene,
            name=name,
            atlas=atlas,
            program=prog,
            num_animation_frames=num_animation_frames,
        )
        bundle = {"bbmodel": bb_doc, "runtime_import_verified": False}
        rt_rec = ingest_external_asset(bb_doc, "bbmodel", rt_prov)
        rt_err = float(rt_rec.vertex_reconstruction_error)

    rt_sc = compile_geometry(rt_rec.program)
    for bx in scene.boxes:
        if bx.name in rt_sc.box_by_name:
            for inc_inf in (False, True):
                c_err = float(
                    np.max(
                        np.linalg.norm(
                            bx.corners(include_inflate=inc_inf) - rt_sc.box_by_name[bx.name].corners(include_inflate=inc_inf),
                            axis=-1,
                        )
                    )
                )
                rt_err = max(rt_err, c_err)

    rt_clips = len(rt_rec.program.behavior.clips) if rt_rec.program.behavior is not None else 0
    return {
        "export_format": target_fmt,
        "runtime_import_verified": False,
        "static_audit_passed": bool(static_audit["passed"]),
        "behavior_clips_total": clips_total,
        "behavior_clips_passed": clips_passed,
        "roundtrip_vertex_max_error": round(rt_err, 10),
        "roundtrip_locator_count": len(rt_sc.locators),
        "roundtrip_clip_count": rt_clips,
        "payload": bundle,
    }
