"""Shared deterministic geometry for compiler, audit, and future projection."""

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
    BehaviorFrameSample,
    MuscleTelemetry,
    MusculoskeletalPoseResult,
    audit_behavior_clip,
    evaluate_behavior_clip,
    evaluate_muscles_at_activations,
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
    "BehaviorFrameSample",
    "MuscleTelemetry",
    "MusculoskeletalPoseResult",
    "audit_behavior_clip",
    "evaluate_behavior_clip",
    "evaluate_muscles_at_activations",
    "solve_look_chain",
]
