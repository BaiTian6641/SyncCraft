"""Action-conditioned 3D World Model transitions, exemplars, interventions, inverse inference, and datasets."""

from .exemplars import build_articulated_mechanism_world, build_biological_creature_world
from .interventions import run_seven_interventions_suite
from .inverse_inference import (
    APPROVED_PERMISSIVE_LICENSES,
    AssetProvenance,
    IngestedAssetRecord,
    LicenseGateError,
    MaterialHypothesis,
    infer_material_posterior,
    ingest_external_asset,
    verify_asset_provenance,
)
from .trajectory_dataset import (
    COMPOSITIONAL_HELDOUT_TRIPLES,
    WORLD_DATASET_PRESETS,
    build_intervention_trajectory_dataset,
    evaluate_matched_budget_world_baselines,
)
from .transitions import (
    ApplyRubbingAction,
    EditGeometryAction,
    EditMaterialAction,
    JointTrajectoryAction,
    NoOpEnvironmentAction,
    RepackUVAction,
    SetLightingAction,
    SetPoseAction,
    SetTexelDensityAction,
    TransitionResult,
    TwoBoneIKAction,
    WorldAction,
    action_from_dict,
    step_world,
)

__all__ = [
    "APPROVED_PERMISSIVE_LICENSES",
    "COMPOSITIONAL_HELDOUT_TRIPLES",
    "WORLD_DATASET_PRESETS",
    "ApplyRubbingAction",
    "AssetProvenance",
    "EditGeometryAction",
    "EditMaterialAction",
    "IngestedAssetRecord",
    "JointTrajectoryAction",
    "LicenseGateError",
    "MaterialHypothesis",
    "NoOpEnvironmentAction",
    "RepackUVAction",
    "SetLightingAction",
    "SetPoseAction",
    "SetTexelDensityAction",
    "TransitionResult",
    "TwoBoneIKAction",
    "WorldAction",
    "action_from_dict",
    "build_articulated_mechanism_world",
    "build_biological_creature_world",
    "build_intervention_trajectory_dataset",
    "evaluate_matched_budget_world_baselines",
    "infer_material_posterior",
    "ingest_external_asset",
    "run_seven_interventions_suite",
    "step_world",
    "verify_asset_provenance",
]
