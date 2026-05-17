"""PROMETHEE I / II computation pipeline.

The functions follow the standard PROMETHEE construction: a marginal
preference function per criterion (V-shape with indifference and preference
thresholds), aggregated into a comprehensive preference index, then summed
into positive, negative and net flows. Partial and complete rankings are
constructed from those flows as outranking matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import CriterionType


def difference_function(
    alternative_a: float,
    alternative_b: float,
    criterion_type: CriterionType,
) -> float:
    """Signed criterion difference ``a − b`` for gain, ``b − a`` for cost."""
    if criterion_type == CriterionType.GAIN:
        return alternative_a - alternative_b
    return alternative_b - alternative_a


def marginal_preference_function(
    diff: float,
    indifference_threshold: float,
    preference_threshold: float,
) -> float:
    """V-shape preference function.

    Returns 0 below the indifference threshold ``q``, 1 above the preference
    threshold ``p``, and a linear ramp in between. ``p == q`` collapses to a
    step at ``q``.
    """
    if diff > preference_threshold:
        return 1.0
    if diff <= indifference_threshold:
        return 0.0
    return (diff - indifference_threshold) / (preference_threshold - indifference_threshold)


def calculate_marginal_preference_matrix(
    dataset: pd.DataFrame, preference_information: pd.DataFrame
) -> np.ndarray:
    """Three-dimensional marginal preference indices over (alt, alt, criterion)."""
    n_alts = len(dataset)
    n_criteria = len(dataset.columns)
    result = np.zeros((n_alts, n_alts, n_criteria))

    q = preference_information["q"].to_numpy()
    p = preference_information["p"].to_numpy()
    types = preference_information["type"].to_numpy()
    values = dataset.to_numpy()

    for i in range(n_alts):
        for j in range(n_alts):
            for k in range(n_criteria):
                diff = difference_function(values[i, k], values[j, k], types[k])
                result[i, j, k] = marginal_preference_function(diff, q[k], p[k])
    return result


def calculate_comprehensive_preference_index(
    marginal_preference_matrix: np.ndarray, preference_information: pd.DataFrame
) -> np.ndarray:
    """Weight-normalised sum of marginal indices across criteria."""
    weights = preference_information["w"].to_numpy()
    return np.sum(marginal_preference_matrix * weights, axis=2) / weights.sum()


def calculate_positive_flow(
    comprehensive_preference_matrix: np.ndarray, alternatives: pd.Index
) -> pd.Series:
    """``φ⁺(a) = Σ_b π(a, b)`` — how much ``a`` dominates the rest."""
    return pd.Series(comprehensive_preference_matrix.sum(axis=1), index=alternatives)


def calculate_negative_flow(
    comprehensive_preference_matrix: np.ndarray, alternatives: pd.Index
) -> pd.Series:
    """``φ⁻(a) = Σ_b π(b, a)`` — how much ``a`` is dominated."""
    return pd.Series(comprehensive_preference_matrix.sum(axis=0), index=alternatives)


def calculate_net_flow(
    positive_flow: pd.Series, negative_flow: pd.Series
) -> pd.Series:
    """``φ(a) = φ⁺(a) − φ⁻(a)`` — the PROMETHEE II ranking score."""
    return positive_flow - negative_flow


def create_partial_ranking(
    positive_flow: pd.Series, negative_flow: pd.Series
) -> pd.DataFrame:
    """PROMETHEE I outranking matrix.

    Entry ``[i, j] = 1`` iff alternative ``i`` is at least as good as ``j``
    on *both* flows. Pairs that disagree are left at 0, expressing the
    "incomparable" relation of PROMETHEE I.
    """
    alts = positive_flow.index
    n = len(alts)
    matrix = np.zeros((n, n), dtype=int)
    pos = positive_flow.to_numpy()
    neg = negative_flow.to_numpy()
    for i in range(n):
        for j in range(n):
            if pos[i] >= pos[j] and neg[i] <= neg[j]:
                matrix[i, j] = 1
    return pd.DataFrame(matrix, index=alts, columns=alts)


def create_complete_ranking(net_flow: pd.Series) -> pd.DataFrame:
    """PROMETHEE II outranking matrix — total order on net flow."""
    alts = net_flow.index
    n = len(alts)
    matrix = np.zeros((n, n), dtype=int)
    values = net_flow.to_numpy()
    for i in range(n):
        for j in range(n):
            if values[i] >= values[j]:
                matrix[i, j] = 1
    return pd.DataFrame(matrix, index=alts, columns=alts)
