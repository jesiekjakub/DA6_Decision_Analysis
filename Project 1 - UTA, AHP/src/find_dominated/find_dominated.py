"""Pareto-dominance scan over the processed dataset.

An alternative ``a`` dominates ``b`` iff it is weakly better on every criterion
and strictly better on at least one. After sign-flipping cost criteria so all
columns point in the "higher is better" direction, this reduces to two
broadcast comparisons.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

from common.data_loading import load_data
from common.logging_utils import get_logger

_LOG: Final = get_logger(__name__)


def find_dominated_pairs(
    df: pd.DataFrame, directions: dict[str, int]
) -> list[tuple[str, str]]:
    """Return every ``(dominator, dominated)`` country pair under Pareto dominance."""
    criteria = list(directions.keys())
    # Sign-flip cost criteria so every column is now "higher = better".
    signs = np.array([directions[c] for c in criteria], dtype=float)
    values = df[criteria].to_numpy(dtype=float) * signs

    # Broadcast (n, 1, m) vs (1, n, m) → (n, n, m) per-criterion comparisons.
    weakly_better = (values[:, None, :] >= values[None, :, :]).all(axis=2)
    strictly_better = (values[:, None, :] > values[None, :, :]).any(axis=2)
    dominates = weakly_better & strictly_better
    np.fill_diagonal(dominates, False)

    countries = df["Country"].to_numpy()
    dom_idx, sub_idx = np.where(dominates)
    return [(countries[i], countries[j]) for i, j in zip(dom_idx, sub_idx)]


def main() -> None:
    """CLI entry point: print all dominance relationships and per-country summaries."""
    df, directions = load_data()
    pairs = find_dominated_pairs(df, directions)

    if not pairs:
        _LOG.info("No dominated alternatives found.")
        return

    _LOG.info("Found %d dominance relationships:", len(pairs))
    for dominator, dominated in sorted(pairs):
        _LOG.info("  %s dominates %s", dominator, dominated)

    grouped: dict[str, list[str]] = {}
    for dominator, dominated in pairs:
        grouped.setdefault(dominated, []).append(dominator)

    _LOG.info("Dominated alternatives (%d):", len(grouped))
    for country, dominators in sorted(grouped.items(), key=lambda item: -len(item[1])):
        _LOG.info(
            "  %s — dominated by %d: %s",
            country,
            len(dominators),
            ", ".join(sorted(dominators)),
        )


if __name__ == "__main__":
    main()
