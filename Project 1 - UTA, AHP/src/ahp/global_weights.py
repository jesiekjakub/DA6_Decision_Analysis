"""Combine Goal-level and Criteria-level AHP weights into a single weight per criterion."""

from __future__ import annotations

from typing import Mapping, Sequence

from ahp.hierarchy_setup import CATEGORIES, CATEGORY_CRITERIA


def compute_global_weights(
    w_goal: Sequence[float],
    w_categories: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    """Multiply category priorities by criterion-within-category priorities.

    Args:
        w_goal: Priority weights at the Goal level, one per category in the
            same order as :data:`hierarchy_setup.CATEGORIES`.
        w_categories: Per-category priority weights, in the same order as the
            criteria listed under that category in
            :data:`hierarchy_setup.CATEGORY_CRITERIA`.

    Returns:
        Mapping from criterion name to its normalised global weight.
        Renormalised to sum to 1 even when the inputs do — defensive against
        small numerical drift from the eigenvector solver.
    """
    global_w: dict[str, float] = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        for crit_idx, crit in enumerate(CATEGORY_CRITERIA[cat]):
            global_w[crit] = w_goal[cat_idx] * w_categories[cat][crit_idx]

    total = sum(global_w.values())
    return {c: v / total for c, v in global_w.items()}
