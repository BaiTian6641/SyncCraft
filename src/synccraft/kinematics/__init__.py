"""Shared deterministic geometry, SAT audit, and Zajac/Hill biomechanics."""

from .se3_chain import (
    Scene,
    WorldBone,
    WorldBox,
    WorldLocator,
    compile_geometry,
    compile_posed_geometry,
    euler_xyz_matrix,
)
from .sat_potential import SATResult, sat_obb
from .musculoskeletal import (
    HILL_ACTIVE_WIDTH_DEFAULT,
    HILL_PASSIVE_STIFFNESS_DEFAULT,
    MUSCULOSKELETAL_FIDELITY_LABEL,
    BehaviorFrameSample,
    MuscleTelemetry,
    MusculoskeletalPoseResult,
    audit_behavior_clip,
    evaluate_behavior_clip,
    evaluate_controller_transition,
    evaluate_keyframe_track,
    evaluate_muscles_at_activations,
    hill_active_force_length,
    hill_passive_force_length,
    solve_look_chain,
)

__all__ = [
    "Scene",
    "WorldBone",
    "WorldBox",
    "WorldLocator",
    "compile_geometry",
    "compile_posed_geometry",
    "euler_xyz_matrix",
    "SATResult",
    "sat_obb",
    "HILL_ACTIVE_WIDTH_DEFAULT",
    "HILL_PASSIVE_STIFFNESS_DEFAULT",
    "MUSCULOSKELETAL_FIDELITY_LABEL",
    "BehaviorFrameSample",
    "MuscleTelemetry",
    "MusculoskeletalPoseResult",
    "audit_behavior_clip",
    "evaluate_behavior_clip",
    "evaluate_controller_transition",
    "evaluate_keyframe_track",
    "evaluate_muscles_at_activations",
    "hill_active_force_length",
    "hill_passive_force_length",
    "solve_look_chain",
]
