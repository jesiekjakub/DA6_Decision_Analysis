"""Public surface of the AHP package."""

from .alternative_matrices import build_alternative_matrix
from .consistency import MatrixDiscrepancy, max_discrepancy, reconstruct_matrix
from .dm_matrices import A_economic, A_geography, A_goal, A_social, MATRICES
from .global_weights import compute_global_weights
from .hierarchy_setup import (
    CATEGORIES,
    CATEGORY_CRITERIA,
    CRITERIA,
    CRITERION_CATEGORY,
    DIRECTIONS,
)
from .scoring import compute_ahp_scores
from .thresholds import THRESHOLDS, diff_to_score
from .weights import AhpWeightsResult, ahp_weights

__all__ = [
    "A_economic",
    "A_geography",
    "A_goal",
    "A_social",
    "AhpWeightsResult",
    "CATEGORIES",
    "CATEGORY_CRITERIA",
    "CRITERIA",
    "CRITERION_CATEGORY",
    "DIRECTIONS",
    "MATRICES",
    "MatrixDiscrepancy",
    "THRESHOLDS",
    "ahp_weights",
    "build_alternative_matrix",
    "compute_ahp_scores",
    "compute_global_weights",
    "diff_to_score",
    "max_discrepancy",
    "reconstruct_matrix",
]
