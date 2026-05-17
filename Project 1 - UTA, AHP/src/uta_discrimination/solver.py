"""Most-discriminant value function (task 2.2).

Given the consistent subset of preferences left over from task 2.1, find the
additive value function that maximises the smallest utility gap ``epsilon``
across all kept pairs. The model is the same UTA LP minus the binary removal
variables.
"""

from __future__ import annotations

import math
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd
import pulp

from common.config import GAMMA, OUTPUT_DIR
from common.data_loading import load_data, load_preferences, load_removal_indices
from common.exceptions import SolverError
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


def get_consistent_preferences(
    preferences: list[tuple[str, str]],
    removal_indices: set[int],
) -> list[tuple[int, tuple[str, str]]]:
    """Drop the indices flagged in task 2.1, preserving original positions."""
    return [(i, pref) for i, pref in enumerate(preferences) if i not in removal_indices]


def build_discrimination_model(
    df: pd.DataFrame,
    directions: dict[str, int],
    consistent_prefs: list[tuple[int, tuple[str, str]]],
) -> tuple[pulp.LpProblem, dict[str, list[pulp.LpVariable]], pulp.LpVariable]:
    """Build the LP that maximises ``epsilon`` subject to C1–C5 + C4."""
    criteria = list(directions.keys())
    char_points = compute_characteristic_points(df, directions)

    model = pulp.LpProblem("UTA_discrimination", pulp.LpMaximize)
    u = create_marginal_value_variables(criteria)
    epsilon = pulp.LpVariable("epsilon", lowBound=0)
    model += epsilon, "maximize_discrimination"

    add_normalization_constraints(model, u, criteria)
    add_monotonicity_constraints(model, u, criteria)
    add_weight_bound_constraints(model, u, criteria)
    add_anti_flatness_constraints(model, u, criteria)

    for orig_k, (preferred, over) in consistent_prefs:
        u_pref = compute_utility(preferred, df, criteria, char_points, u)
        u_over = compute_utility(over, df, criteria, char_points, u)
        model += (
            u_pref - u_over >= epsilon,
            f"pref_{orig_k}_{preferred}_over_{over}",
        )

    return model, u, epsilon


def log_model_details(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    epsilon: pulp.LpVariable,
    criteria: list[str],
    char_points: dict[str, list[float]],
) -> None:
    """Emit the full model dump (constraints, marginal values, criterion weights)."""
    _LOG.info("Objective: maximize epsilon")
    _LOG.info("Constraints (%d):", len(model.constraints))
    for name, constraint in model.constraints.items():
        _LOG.info("  %s: %s", name, constraint)

    _LOG.info("epsilon* = %.6f", epsilon.varValue)
    _LOG.info("Marginal values u_i(x_i^j):")
    for c in criteria:
        _LOG.info("  %s:", c)
        for j in range(GAMMA + 1):
            label = "worst" if j == 0 else ("best" if j == GAMMA else "")
            _LOG.info("    u(%.1f) = %.6f  %s", char_points[c][j], u[c][j].varValue, label)

    _LOG.info("Criterion weights (u_i(beta_i)):")
    for c in criteria:
        _LOG.info("  %s: %.6f", c, u[c][GAMMA].varValue)


def rank_alternatives(
    df: pd.DataFrame,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
    char_points: dict[str, list[float]],
) -> pd.DataFrame:
    """Score every alternative with the solved marginal values and return a sorted ranking."""
    rows = []
    for _, row in df.iterrows():
        country = row["Country"]
        utility_val = pulp.value(compute_utility(country, df, criteria, char_points, u))
        rows.append({"Country": country, "U(a)": round(utility_val, 6)})

    ranking = pd.DataFrame(rows).sort_values("U(a)", ascending=False).reset_index(drop=True)
    ranking.index += 1
    ranking.index.name = "Rank"
    return ranking


def plot_marginal_value_functions(
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
    char_points: dict[str, list[float]],
    directions: dict[str, int],
    *,
    ncols: int = 4,
) -> plt.Figure:
    """Grid plot of every marginal value function, sharing a y-axis range."""
    n = len(criteria)
    nrows = math.ceil(n / ncols)
    max_weight = max(u[c][GAMMA].varValue for c in criteria)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    flat_axes = axes.flatten() if n > 1 else [axes]

    for idx, c in enumerate(criteria):
        ax = flat_axes[idx]
        pts = char_points[c]
        vals = [u[c][j].varValue for j in range(GAMMA + 1)]

        # Cost criteria are stored with worst→best matching descending raw
        # values; flipping here makes the x-axis always read low→high.
        if directions[c] == -1:
            pts, vals = list(reversed(pts)), list(reversed(vals))

        ax.plot(pts, vals, "o-", linewidth=2, markersize=6, color="steelblue")
        ax.set_ylim(0, max_weight * 1.1)
        ax.set_title(c, fontsize=10, fontweight="bold")
        ax.set_xlabel("Raw value", fontsize=9)
        ax.set_ylabel("Marginal value", fontsize=9)
        ax.grid(True, alpha=0.3)

        nature = "gain" if directions[c] == 1 else "cost"
        ax.text(
            0.02, 0.95, f"({nature})", transform=ax.transAxes,
            fontsize=8, verticalalignment="top", color="gray",
        )

    for idx in range(n, nrows * ncols):
        flat_axes[idx].axis("off")

    fig.tight_layout()
    return fig


def main() -> None:
    """CLI entry point for task 2.2."""
    df, directions = load_data()
    preferences = load_preferences()
    criteria = list(directions.keys())

    _LOG.info("Dataset: %d alternatives, %d criteria", len(df), len(criteria))
    _LOG.info("Total preferences: %d", len(preferences))

    removal_indices = load_removal_indices()
    _LOG.info("Removed preference indices: %s", sorted(removal_indices))
    for idx in sorted(removal_indices):
        pref, over = preferences[idx]
        _LOG.info("  [%d] %s > %s", idx, pref, over)

    consistent_prefs = get_consistent_preferences(preferences, removal_indices)
    _LOG.info("Consistent subset: %d preferences", len(consistent_prefs))

    char_points = compute_characteristic_points(df, directions)
    model, u, epsilon = build_discrimination_model(df, directions, consistent_prefs)

    status = solve_model(model)
    _LOG.info("Solver status: %s", pulp.LpStatus[status])
    if status != pulp.constants.LpStatusOptimal:
        raise SolverError(pulp.LpStatus[status])

    log_model_details(model, u, epsilon, criteria, char_points)

    ranking = rank_alternatives(df, u, criteria, char_points)
    _LOG.info("Ranking of all alternatives:\n%s", ranking.to_string())

    fig = plot_marginal_value_functions(u, criteria, char_points, directions)
    output_path = OUTPUT_DIR / "marginal_value_functions.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("Plot saved to: %s", output_path)


if __name__ == "__main__":
    main()
