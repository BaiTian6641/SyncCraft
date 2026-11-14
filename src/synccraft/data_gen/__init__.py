"""Procedural Block-DSL v3 dataset generation and multi-core SAT/P1-02 shard builder."""

from .static_families import (
    DATASET_PRESETS,
    GENERATOR_REVISION,
    MANDATORY_ROW_FIELDS,
    SEEN_STATIC_FAMILIES,
    SYNCCRAFT_ATTRIBUTION,
    UNSEEN_STATIC_FAMILIES,
    DatasetPreset,
    build_verified_dataset,
    generate_candidate_for_family,
    verify_and_package_candidate,
)

__all__ = [
    "DATASET_PRESETS",
    "GENERATOR_REVISION",
    "MANDATORY_ROW_FIELDS",
    "SEEN_STATIC_FAMILIES",
    "SYNCCRAFT_ATTRIBUTION",
    "UNSEEN_STATIC_FAMILIES",
    "DatasetPreset",
    "build_verified_dataset",
    "generate_candidate_for_family",
    "verify_and_package_candidate",
]
