"""Aggregate per-criterion AHP priorities into a single ranking score per alternative."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from ahp.alternative_matrices import build_alternative_matrix
from ahp.hierarchy_setup import CRITERIA
from ahp.weights import ahp_weights


def compute_ahp_scores(
    df: pd.DataFrame, global_weights: Mapping[str, float]
) -> pd.Series:
    """Return the weighted sum of criterion priorities, indexed by country.

    The per-criterion local priorities come from running the eigenvector
    method on each alternative-comparison matrix; the global criterion weight
    scales each contribution before summation.
    """
    scores = np.zeros(len(df), dtype=float)
    for crit in CRITERIA:
        local = ahp_weights(build_alternative_matrix(crit, df)).weights
        scores += global_weights[crit] * local

    return pd.Series(scores, index=df["Country"], name="Score")
