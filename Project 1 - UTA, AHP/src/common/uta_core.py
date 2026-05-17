"""Building blocks shared by the UTA inconsistency resolver and the
discrimination solver: characteristic points, piecewise-linear interpolation,
the marginal value variables, and the C1–C5 constraint families.

The math follows the standard UTA formulation by Jacquet-Lagrèze & Siskos
(1982): each criterion is divided into ``GAMMA`` equal-width segments, the
marginal value function is anchored at the worst point and monotonically
increasing, and the comprehensive utility ``U(a)`` is the sum of interpolated
marginal values.
"""

from __future__ import annotations

import pandas as pd
import pulp

from .config import (
    GAMMA,
    MIN_SEGMENT_SHARE,
    NON_LINEARITY_THRESHOLD,
    WEIGHT_LB,
    WEIGHT_UB,
)
from .exceptions import SolverError

# Comparison tolerance for "this raw value sits exactly on a characteristic
# point." Tightened a couple of orders below typical csv-rounding noise.
_RAW_VALUE_TOL: float = 1e-9
# Degenerate-segment threshold for interpolation: when two adjacent
# characteristic points are within this distance, return the endpoint
# variable instead of risking division by an effectively-zero span.
_DEGENERATE_SEGMENT_TOL: float = 1e-12
# Big-M for the disjunctive non-linearity constraint. Marginal values are
# normalised to [0, 1] by construction so 1.0 is the tightest valid M.
_NON_LINEARITY_BIG_M: float = 1.0


def compute_characteristic_points(
    df: pd.DataFrame, directions: dict[str, int]
) -> dict[str, list[float]]:
    """Equally-spaced characteristic points for each criterion, worst → best.

    For a gain criterion the worst value is ``min`` and the best is ``max``;
    for a cost criterion the convention is flipped. Returns a list of
    ``GAMMA + 1`` points per criterion.
    """
    char_points: dict[str, list[float]] = {}
    for criterion, direction in directions.items():
        col_min = float(df[criterion].min())
        col_max = float(df[criterion].max())
        if direction == 1:
            alpha, beta = col_min, col_max
        else:
            alpha, beta = col_max, col_min
        char_points[criterion] = [
            alpha + j / GAMMA * (beta - alpha) for j in range(GAMMA + 1)
        ]
    return char_points


def interpolate_value(
    raw_val: float,
    char_points: list[float],
    u_vars: list[pulp.LpVariable],
) -> pulp.LpAffineExpression:
    """Piecewise-linear interpolation expressed as a PuLP affine expression.

    Returns ``u(x_k) + t * (u(x_{k+1}) - u(x_k))`` where ``[x_k, x_{k+1}]`` is
    the segment containing ``raw_val``. Boundary values that round-trip to
    floating point land near a characteristic point are snapped to the nearest
    endpoint variable rather than rejected.
    """
    ascending = char_points[-1] > char_points[0]

    for k in range(len(char_points) - 1):
        lo, hi = char_points[k], char_points[k + 1]
        lo_raw, hi_raw = (lo, hi) if ascending else (hi, lo)
        if lo_raw - _RAW_VALUE_TOL <= raw_val <= hi_raw + _RAW_VALUE_TOL:
            span = abs(hi - lo)
            if span < _DEGENERATE_SEGMENT_TOL:
                return u_vars[k]
            t = abs(raw_val - lo) / span
            return u_vars[k] + t * (u_vars[k + 1] - u_vars[k])

    if abs(raw_val - char_points[0]) < _RAW_VALUE_TOL:
        return u_vars[0]
    if abs(raw_val - char_points[-1]) < _RAW_VALUE_TOL:
        return u_vars[-1]

    raise ValueError(
        f"Value {raw_val} outside characteristic-point range "
        f"[{min(char_points)}, {max(char_points)}]"
    )


def compute_utility(
    country: str,
    df: pd.DataFrame,
    criteria: list[str],
    char_points: dict[str, list[float]],
    u_vars: dict[str, list[pulp.LpVariable]],
) -> pulp.LpAffineExpression:
    """Comprehensive utility ``U(a) = Σ_i u_i(g_i(a))`` for one country."""
    row = df.loc[df["Country"] == country].iloc[0]
    expr = pulp.LpAffineExpression()
    for criterion in criteria:
        expr += interpolate_value(row[criterion], char_points[criterion], u_vars[criterion])
    return expr


def create_marginal_value_variables(
    criteria: list[str],
) -> dict[str, list[pulp.LpVariable]]:
    """Create the ``u_i(x_i^j)`` continuous variables for each criterion."""
    return {
        criterion: [
            pulp.LpVariable(f"u_{criterion}_x{j}", lowBound=0)
            for j in range(GAMMA + 1)
        ]
        for criterion in criteria
    }


def add_normalization_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C1: ``u_i(α_i) = 0`` per criterion and ``Σ_i u_i(β_i) = 1``."""
    for criterion in criteria:
        model += u[criterion][0] == 0, f"norm_worst_{criterion}"
    model += (
        pulp.lpSum(u[criterion][GAMMA] for criterion in criteria) == 1,
        "norm_sum_best",
    )


def add_monotonicity_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C2: marginal value is non-decreasing across segments."""
    for criterion in criteria:
        for j in range(GAMMA):
            model += (
                u[criterion][j + 1] - u[criterion][j] >= 0,
                f"mono_{criterion}_seg{j}",
            )


def add_weight_bound_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C3: ``WEIGHT_LB ≤ u_i(β_i) ≤ WEIGHT_UB``."""
    for criterion in criteria:
        model += u[criterion][GAMMA] <= WEIGHT_UB, f"weight_ub_{criterion}"
        model += u[criterion][GAMMA] >= WEIGHT_LB, f"weight_lb_{criterion}"


def add_anti_flatness_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C5: each segment carries at least ``MIN_SEGMENT_SHARE`` of the criterion
    weight, plus a disjunctive non-linearity constraint that forces either the
    first or the last segment to be larger than the other by at least
    ``NON_LINEARITY_THRESHOLD * u_i(β_i)``. Without these the LP frequently
    collapses to nearly flat marginal value functions that ignore the data.
    """
    for criterion in criteria:
        for j in range(GAMMA):
            model += (
                u[criterion][j + 1] - u[criterion][j]
                >= MIN_SEGMENT_SHARE * u[criterion][GAMMA],
                f"min_seg_{criterion}_seg{j}",
            )

    for criterion in criteria:
        seg_first = u[criterion][1] - u[criterion][0]
        seg_last = u[criterion][GAMMA] - u[criterion][GAMMA - 1]
        threshold = NON_LINEARITY_THRESHOLD * u[criterion][GAMMA]
        # Binary disjunction: d=0 selects the concave branch (first > last),
        # d=1 selects the convex branch (last > first). Big-M deactivates the
        # other constraint.
        d = pulp.LpVariable(f"d_{criterion}", cat=pulp.LpBinary)
        model += (
            seg_first - seg_last >= threshold - _NON_LINEARITY_BIG_M * d,
            f"nonlin_{criterion}_concave",
        )
        model += (
            seg_last - seg_first >= threshold - _NON_LINEARITY_BIG_M * (1 - d),
            f"nonlin_{criterion}_convex",
        )


def solve_model(model: pulp.LpProblem, *, raise_on_failure: bool = False) -> int:
    """Run the model. Prefer GLPK; CBC is the fallback when GLPK isn't installed.

    Args:
        model: A PuLP problem with objective and constraints attached.
        raise_on_failure: If True, raise :class:`SolverError` on any status
            other than Optimal. Iterative callers (the inconsistency resolver)
            need to detect infeasibility without an exception, so the default
            is False.

    Returns:
        The PuLP status code (1=Optimal, -1=Infeasible, etc.).
    """
    try:
        model.solve(pulp.GLPK_CMD(msg=0))
    except pulp.PulpSolverError:
        model.solve(pulp.PULP_CBC_CMD(msg=0))

    if raise_on_failure and model.status != pulp.constants.LpStatusOptimal:
        raise SolverError(pulp.LpStatus[model.status])
    return model.status
