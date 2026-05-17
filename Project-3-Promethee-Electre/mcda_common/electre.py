"""ELECTRE Tri-B sorting pipeline.

The functions compute marginal concordance, comprehensive concordance,
marginal discordance, the credibility index, the boolean outranking
relation, and finally the pessimistic and optimistic class assignments
defined in the ELECTRE Tri-B method by Roy (1991).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .promethee import difference_function
from .types import CriterionType


def calculate_marginal_concordance_index(
    diff: float,
    indifference_threshold: float,
    preference_threshold: float,
) -> float:
    """Marginal concordance: 1 above ``-q``, 0 below ``-p``, linear in between."""
    if diff >= -indifference_threshold:
        return 1.0
    if diff <= -preference_threshold:
        return 0.0
    return (preference_threshold + diff) / (preference_threshold - indifference_threshold)


def calculate_marginal_concordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    preference_information: pd.DataFrame,
) -> np.ndarray:
    """4-D marginal concordance: ``[direction, alt, profile, criterion]``.

    ``direction = 0`` is the alternative-outranks-profile case (``aSb``);
    ``direction = 1`` is the reverse (``bSa``). Both are needed to construct
    the four-valued outranking relation downstream.
    """
    n_alts = len(dataset)
    n_profiles = len(boundary_profiles)
    n_criteria = len(dataset.columns)
    result = np.zeros((2, n_alts, n_profiles, n_criteria))

    q = preference_information["q"].to_numpy()
    p = preference_information["p"].to_numpy()
    types = preference_information["type"].to_numpy()
    alt_values = dataset.to_numpy()
    profile_values = boundary_profiles.to_numpy()

    for i in range(n_alts):
        for j in range(n_profiles):
            for k in range(n_criteria):
                diff_ab = difference_function(alt_values[i, k], profile_values[j, k], types[k])
                result[0, i, j, k] = calculate_marginal_concordance_index(diff_ab, q[k], p[k])
                diff_ba = difference_function(profile_values[j, k], alt_values[i, k], types[k])
                result[1, i, j, k] = calculate_marginal_concordance_index(diff_ba, q[k], p[k])
    return result


def calculate_comprehensive_concordance_matrix(
    marginal_concordance_matrix: np.ndarray,
    preference_information: pd.DataFrame,
) -> np.ndarray:
    """Weight-normalised concordance across criteria."""
    weights = preference_information["w"].to_numpy()
    return np.sum(marginal_concordance_matrix * weights, axis=3) / weights.sum()


def calculate_marginal_discordance_index(
    diff: float, preference_threshold: float, veto_threshold: float
) -> float:
    """Marginal discordance: 1 below ``-v``, 0 above ``-p``, linear in between."""
    if diff <= -veto_threshold:
        return 1.0
    if diff >= -preference_threshold:
        return 0.0
    return (-diff - preference_threshold) / (veto_threshold - preference_threshold)


def calculate_marginal_discordance_matrix(
    dataset: pd.DataFrame,
    boundary_profiles: pd.DataFrame,
    preference_thresholds: pd.DataFrame,
    veto_thresholds: pd.DataFrame,
    criterion_types: pd.DataFrame,
) -> np.ndarray:
    """4-D marginal discordance: ``[direction, alt, profile, criterion]``.

    Criteria without a veto threshold (``NaN`` in the ``v`` column) are
    silently skipped — their entries remain at the array default of 0,
    matching the original ELECTRE Tri-B convention.
    """
    n_alts = len(dataset)
    n_profiles = len(boundary_profiles)
    n_criteria = len(dataset.columns)
    result = np.zeros((2, n_alts, n_profiles, n_criteria))

    p = preference_thresholds["p"].to_numpy()
    v = veto_thresholds["v"].to_numpy()
    types = criterion_types["type"].to_numpy()
    alt_values = dataset.to_numpy()
    profile_values = boundary_profiles.to_numpy()

    for i in range(n_alts):
        for j in range(n_profiles):
            for k in range(n_criteria):
                if pd.isna(v[k]):
                    continue
                diff_ab = difference_function(alt_values[i, k], profile_values[j, k], types[k])
                result[0, i, j, k] = calculate_marginal_discordance_index(diff_ab, p[k], v[k])
                diff_ba = difference_function(profile_values[j, k], alt_values[i, k], types[k])
                result[1, i, j, k] = calculate_marginal_discordance_index(diff_ba, p[k], v[k])
    return result


def calculate_credibility_index(
    comprehensive_concordance_matrix: np.ndarray,
    marginal_discordance_matrix: np.ndarray,
) -> np.ndarray:
    """ELECTRE Tri-B credibility index ``σ(a, b)``.

    Concordance is multiplicatively damped by ``(1 − d_k) / (1 − c)`` for
    every criterion whose discordance exceeds the comprehensive concordance.
    Criteria where ``d_k ≤ c`` contribute neutrally (factor 1).
    """
    C = comprehensive_concordance_matrix
    D = marginal_discordance_matrix
    credibility = np.zeros_like(C)
    n_criteria = D.shape[3]
    for direction in range(2):
        for i in range(C.shape[1]):
            for j in range(C.shape[2]):
                c = C[direction, i, j]
                product = 1.0
                for k in range(n_criteria):
                    d = D[direction, i, j, k]
                    if d > c:
                        # 1 - c can only equal 0 when c == 1, but the
                        # discordance > c condition guarantees c < 1 there,
                        # so no division-by-zero guard is needed.
                        product *= (1.0 - d) / (1.0 - c)
                credibility[direction, i, j] = c * product
    return credibility


def calculate_outranking_relation_matrix(
    credibility_index: np.ndarray, credibility_threshold: float
) -> np.ndarray:
    """Boolean cut: ``σ(a, b) ≥ λ`` defines the crisp outranking relation."""
    return credibility_index >= credibility_threshold


def build_relation_dataframe(
    outranking_matrix: np.ndarray,
    alternatives: list[str],
    profiles: list[str],
) -> pd.DataFrame:
    """Collapse the two-direction outranking arrays into a four-valued relation.

    Cells take values ``"I"`` (indifference), ``">"`` (alternative outranks
    profile), ``"<"`` (profile outranks alternative), or ``"?"`` (incomparable).
    """
    relations = pd.DataFrame(index=alternatives, columns=profiles, dtype=str)
    for i, alt in enumerate(alternatives):
        for j, profile in enumerate(profiles):
            a_out_b = bool(outranking_matrix[0, i, j])
            b_out_a = bool(outranking_matrix[1, i, j])
            if a_out_b and b_out_a:
                relations.loc[alt, profile] = "I"
            elif a_out_b:
                relations.loc[alt, profile] = ">"
            elif b_out_a:
                relations.loc[alt, profile] = "<"
            else:
                relations.loc[alt, profile] = "?"
    return relations


def calculate_pessimistic_assignment(relation: pd.DataFrame) -> pd.DataFrame:
    """Pessimistic assignment rule: scan profiles top-down, assign first that the alt outranks."""
    profiles = relation.columns.tolist()
    assignments: dict[str, int] = {}
    for alt in relation.index:
        assigned_category = 1
        for h in range(len(profiles) - 1, -1, -1):
            if relation.loc[alt, profiles[h]] in (">", "I"):
                assigned_category = h + 2
                break
        assignments[alt] = assigned_category
    return pd.DataFrame(
        list(assignments.values()),
        index=list(assignments.keys()),
        columns=["pessimistic_assignment"],
    )


def calculate_optimistic_assignment(relation: pd.DataFrame) -> pd.DataFrame:
    """Optimistic assignment rule: scan profiles bottom-up, assign first that outranks the alt."""
    profiles = relation.columns.tolist()
    n_categories = len(profiles) + 1
    assignments: dict[str, int] = {}
    for alt in relation.index:
        assigned_category = n_categories
        for h in range(len(profiles)):
            if relation.loc[alt, profiles[h]] in ("<", "I"):
                assigned_category = h + 1
                break
        assignments[alt] = assigned_category
    return pd.DataFrame(
        list(assignments.values()),
        index=list(assignments.keys()),
        columns=["optimistic_assignment"],
    )


# Backward-compatible aliases for the typo'd names used in the original notebook.
calculate_pessimistic_assigment = calculate_pessimistic_assignment
calculate_optimistic_assigment = calculate_optimistic_assignment


__all__ = [
    "build_relation_dataframe",
    "calculate_comprehensive_concordance_matrix",
    "calculate_credibility_index",
    "calculate_marginal_concordance_index",
    "calculate_marginal_concordance_matrix",
    "calculate_marginal_discordance_index",
    "calculate_marginal_discordance_matrix",
    "calculate_optimistic_assigment",
    "calculate_optimistic_assignment",
    "calculate_outranking_relation_matrix",
    "calculate_pessimistic_assigment",
    "calculate_pessimistic_assignment",
]
