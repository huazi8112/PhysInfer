"""Authoritative PhysInfer model and reproducibility utilities."""

from .bic_annotation import (
    SecondaryBICAnnotation,
    annotate_secondary_bic_preference,
    bic_from_nll,
    rate_separation_degree,
)
from .features import FEATURE_NAMES, extract_features15
from .models import (
    SequentialSingleOn,
    adaptive_stationary_pmf,
    promoter_generator,
    sample_observed_counts,
)
from .order_selection import SelectiveDecision, joint_bootstrap_decision

__all__ = [
    "FEATURE_NAMES",
    "SecondaryBICAnnotation",
    "SequentialSingleOn",
    "SelectiveDecision",
    "adaptive_stationary_pmf",
    "annotate_secondary_bic_preference",
    "bic_from_nll",
    "extract_features15",
    "joint_bootstrap_decision",
    "promoter_generator",
    "rate_separation_degree",
    "sample_observed_counts",
]
