"""Procedural multi-domain Block-DSL v3 dataset generation and multi-core SAT/P1-02 shard builder."""

from .diversity_engine import (
    MACRO_DOMAINS,
    NAMING_REGISTERS,
    compose_multilevel_prompt,
    compute_qd_signature,
    diversify_program_ast,
    halton_scalar,
    halton_vector,
    summarize_corpus_diversity,
)
from .multi_domain_taxonomy import ENTITY_BODY_PLANS, FAMILY_DOMAINS
from .static_families import (
    DATASET_PRESETS,
    GENERATOR_REVISION,
    MANDATORY_ROW_FIELDS,
    SEEN_FAMILIES,
    SEEN_STATIC_FAMILIES,
    SYNCCRAFT_ATTRIBUTION,
    UNSEEN_FAMILIES,
    UNSEEN_STATIC_FAMILIES,
    DatasetPreset,
    build_verified_dataset,
    generate_candidate_for_family,
    verify_and_package_candidate,
)

__all__ = [
    "DATASET_PRESETS",
    "ENTITY_BODY_PLANS",
    "FAMILY_DOMAINS",
    "GENERATOR_REVISION",
    "MACRO_DOMAINS",
    "MANDATORY_ROW_FIELDS",
    "NAMING_REGISTERS",
    "SEEN_FAMILIES",
    "SEEN_STATIC_FAMILIES",
    "SYNCCRAFT_ATTRIBUTION",
    "UNSEEN_FAMILIES",
    "UNSEEN_STATIC_FAMILIES",
    "DatasetPreset",
    "build_verified_dataset",
    "compose_multilevel_prompt",
    "compute_qd_signature",
    "diversify_program_ast",
    "generate_candidate_for_family",
    "halton_scalar",
    "halton_vector",
    "summarize_corpus_diversity",
    "verify_and_package_candidate",
]
