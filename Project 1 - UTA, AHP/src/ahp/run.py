"""End-to-end AHP pipeline: hierarchy weights → alternative scoring → ranking.

Mirrors the structure of :mod:`uta_discrimination.solver`. Runs the
eigenvector method on each pairwise matrix, combines the goal-level and
criteria-level priorities into a global weight per criterion, scores every
alternative, writes the ranking to CSV, and reports the most inconsistent
judgment in the (intentionally inconsistent) goal-level matrix.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from ahp.consistency import max_discrepancy, reconstruct_matrix
from ahp.dm_matrices import MATRICES
from ahp.global_weights import compute_global_weights
from ahp.hierarchy_setup import CATEGORIES, CATEGORY_CRITERIA, CRITERIA
from ahp.scoring import compute_ahp_scores
from ahp.weights import AhpWeightsResult, ahp_weights
from common.config import OUTPUT_DIR
from common.data_loading import load_data
from common.logging_utils import get_logger

_LOG: Final = get_logger(__name__)
_AHP_RANKING_FILE: Final = OUTPUT_DIR / "ahp_ranking.csv"


def _log_weights(label: str, result: AhpWeightsResult, items: list[str]) -> None:
    _LOG.info(
        "%s — lambda_max=%.4f, CI=%.4f, RI=%.4f, CR=%.4f (%s)",
        label,
        result.lambda_max,
        result.consistency_index,
        result.random_index,
        result.consistency_ratio,
        "consistent" if result.is_consistent else "INCONSISTENT",
    )
    for name, w in zip(items, result.weights):
        _LOG.info("    %-40s %.4f", name, w)


def main() -> None:
    """CLI entry point for the AHP ranking."""
    df, _ = load_data()

    _LOG.info("Goal level (categories)")
    goal_result = ahp_weights(MATRICES["Goal"])
    _log_weights("Goal", goal_result, CATEGORIES)

    if not goal_result.is_consistent:
        discrepancy = max_discrepancy(
            MATRICES["Goal"],
            reconstruct_matrix(goal_result.weights),
            CATEGORIES,
        )
        _LOG.warning(
            "Most inconsistent judgment in the Goal matrix: %s vs %s — "
            "DM said %.3f, consistent value would be %.3f (|Δ|=%.3f)",
            discrepancy.row_label,
            discrepancy.col_label,
            discrepancy.dm_value,
            discrepancy.reconstructed_value,
            discrepancy.abs_diff,
        )

    w_categories: dict[str, list[float]] = {}
    for cat in CATEGORIES:
        cat_result = ahp_weights(MATRICES[cat])
        _log_weights(cat, cat_result, CATEGORY_CRITERIA[cat])
        w_categories[cat] = cat_result.weights.tolist()

    global_weights = compute_global_weights(goal_result.weights.tolist(), w_categories)
    _LOG.info("Global criterion weights:")
    for crit, w in sorted(global_weights.items(), key=lambda kv: -kv[1]):
        _LOG.info("    %-40s %.4f", crit, w)

    scores = compute_ahp_scores(df, global_weights)
    ranking = (
        scores.sort_values(ascending=False)
        .to_frame(name="Score")
        .reset_index()
    )
    ranking.index += 1
    ranking.index.name = "Rank"

    _LOG.info("Final AHP ranking:\n%s", ranking.to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(_AHP_RANKING_FILE)
    _LOG.info("Ranking saved to: %s", _AHP_RANKING_FILE)


if __name__ == "__main__":
    main()
