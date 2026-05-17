"""Inconsistency resolution for the UTA preference model (task 2.1).

Given a set of pairwise comparisons, find every *minimal* subset whose removal
makes the remaining preferences representable by an additive value function
under the standard C1–C5 constraints. The MILP minimises the number of
removals; iterative cuts force each subsequent solve to find a different
minimal subset until the problem becomes infeasible.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import pulp

from common.config import DELTA, GAMMA
from common.data_loading import load_data, load_preferences
from common.logging_utils import get_logger
from common.uta_core import (
    add_anti_flatness_constraints,
    add_monotonicity_constraints,
    add_normalization_constraints,
    add_weight_bound_constraints,
    compute_characteristic_points,
    compute_utility,
    create_marginal_value_variables,
    solve_model,
)

_LOG: Final = get_logger(__name__)


def build_inconsistency_milp(
    df: pd.DataFrame,
    directions: dict[str, int],
    preferences: list[tuple[str, str]],
    cuts: list[frozenset[int]],
) -> tuple[pulp.LpProblem, dict[str, list[pulp.LpVariable]], list[pulp.LpVariable]]:
    """Build the MILP whose optimum identifies the smallest removal set.

    Args:
        df: Processed dataset (wide form) with one row per alternative.
        directions: Per-criterion +1 (gain) / -1 (cost) mapping.
        preferences: DM pairwise comparisons.
        cuts: Removal sets already discovered; each is excluded from the
            feasible region via a no-good cut so subsequent solves return a
            different minimal set.

    Returns:
        The PuLP problem, the marginal-value variable matrix, and the binary
        removal-indicator vector.
    """
    criteria = list(directions.keys())
    char_points = compute_characteristic_points(df, directions)

    model = pulp.LpProblem("UTA_inconsistency", pulp.LpMinimize)
    u = create_marginal_value_variables(criteria)
    v = [pulp.LpVariable(f"v_{k}", cat=pulp.LpBinary) for k in range(len(preferences))]
    model += pulp.lpSum(v), "minimize_removals"

    add_normalization_constraints(model, u, criteria)
    add_monotonicity_constraints(model, u, criteria)
    add_weight_bound_constraints(model, u, criteria)
    add_anti_flatness_constraints(model, u, criteria)

    # C4 with a binary relaxation: when v[k]=1 the right-hand side becomes
    # DELTA - 1, slack enough to absorb any feasible utility ordering, so
    # the preference is effectively dropped. v[k]=0 enforces it normally.
    for k, (preferred, over) in enumerate(preferences):
        u_preferred = compute_utility(preferred, df, criteria, char_points, u)
        u_over = compute_utility(over, df, criteria, char_points, u)
        model += (
            u_preferred - u_over >= DELTA - v[k],
            f"pref_{k}_{preferred}_over_{over}",
        )

    for cut_idx, removal_set in enumerate(cuts):
        # No-good cut: forbid this exact removal set without forbidding its
        # supersets (which may still be minimal in a different sense).
        model += (
            pulp.lpSum(v[k] for k in removal_set) <= len(removal_set) - 1,
            f"cut_{cut_idx}",
        )

    return model, u, v


def find_all_minimal_removals(
    df: pd.DataFrame,
    directions: dict[str, int],
    preferences: list[tuple[str, str]],
) -> list[frozenset[int]]:
    """Iteratively solve the MILP, accumulating cuts until no more solutions exist."""
    cuts: list[frozenset[int]] = []
    removals: list[frozenset[int]] = []

    for iteration in range(1, len(preferences) + 2):
        model, _, v = build_inconsistency_milp(df, directions, preferences, cuts)
        solve_model(model)

        if model.status != pulp.constants.LpStatusOptimal:
            _LOG.info("Iteration %d: infeasible — all minimal removal sets found.", iteration)
            break

        removal_set = frozenset(k for k in range(len(preferences)) if v[k].varValue > 0.5)
        obj_val = int(round(pulp.value(model.objective)))
        _LOG.info("Iteration %d: V* = %d, removed = %s", iteration, obj_val, sorted(removal_set))

        removals.append(removal_set)
        cuts.append(removal_set)

    return removals


def print_results(
    removals: list[frozenset[int]],
    preferences: list[tuple[str, str]],
) -> None:
    """Pretty-print every minimal removal set and the consistent subset it implies."""
    _LOG.info("Found %d minimal removal set(s)", len(removals))
    for idx, removal_set in enumerate(removals, start=1):
        _LOG.info("--- Removal set %d ---", idx)
        _LOG.info("Removed comparisons:")
        for k in sorted(removal_set):
            pref, over = preferences[k]
            _LOG.info("  [%d] %s > %s", k, pref, over)
        _LOG.info("Consistent subset (kept comparisons):")
        for k in range(len(preferences)):
            if k not in removal_set:
                pref, over = preferences[k]
                _LOG.info("  [%d] %s > %s", k, pref, over)


def main() -> None:
    """CLI entry point for task 2.1."""
    df, directions = load_data()
    preferences = load_preferences()
    criteria = list(directions.keys())

    _LOG.info("Dataset: %d alternatives, %d criteria", len(df), len(criteria))
    _LOG.info("Preferences: %d pairwise comparisons", len(preferences))

    for k, (pref, over) in enumerate(preferences):
        _LOG.info("  [%d] %s > %s", k, pref, over)

    char_points = compute_characteristic_points(df, directions)
    _LOG.info("Characteristic points (gamma=%d, %d points per criterion):", GAMMA, GAMMA + 1)
    for c in criteria:
        nature = "gain" if directions[c] == 1 else "cost"
        pts = " -> ".join(f"{p:.1f}" for p in char_points[c])
        _LOG.info("  %s (%s): %s", c, nature, pts)

    removals = find_all_minimal_removals(df, directions, preferences)
    print_results(removals, preferences)


if __name__ == "__main__":
    main()
