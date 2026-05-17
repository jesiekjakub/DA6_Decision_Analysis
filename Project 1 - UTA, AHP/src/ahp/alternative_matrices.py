"""Pairwise comparison matrices for alternatives, one per criterion.

For each criterion the raw values are turned into a reciprocal Saaty matrix:
the magnitude of each pairwise difference is mapped through the criterion's
threshold table, and the sign of the (direction-adjusted) difference picks
which side of the matrix gets the strong score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ahp.hierarchy_setup import DIRECTIONS
from ahp.thresholds import THRESHOLDS, diff_to_score


def build_alternative_matrix(
    criterion: str, df: pd.DataFrame
) -> NDArray[np.float64]:
    """Construct the n×n reciprocal Saaty matrix for ``criterion``."""
    vals = df[criterion].to_numpy()
    direction = DIRECTIONS[criterion]
    table = THRESHOLDS[criterion]
    n = len(vals)
    A = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            score = diff_to_score(abs(vals[i] - vals[j]), table)
            signed_diff = (vals[i] - vals[j]) * direction
            if signed_diff > 0:
                A[i, j], A[j, i] = score, 1.0 / score
            elif signed_diff < 0:
                A[i, j], A[j, i] = 1.0 / score, score
    return A
