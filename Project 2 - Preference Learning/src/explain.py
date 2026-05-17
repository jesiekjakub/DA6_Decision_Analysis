"""Decision-explanation helpers for the ANN-UTADIS model.

These match the analyses in section 2.1 of the report notebook:

- per-criterion contributions to the comprehensive utility (the additive
  decomposition that makes ANN-UTADIS interpretable),
- the smallest single-criterion change that flips an alternative's predicted
  class — both an analytical computation (using the learned thresholds and
  marginal utilities) and an empirical verification by grid-scanning the
  criterion.

The functions take a fitted module and a single alternative ``x`` as input
and return plain dictionaries / dataclasses, with no plotting side effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# Resolution of the empirical grid scan when searching for a class flip on
# the [0, 1] criterion axis. 1001 points spaces the candidate values by 1e-3
# which is below the smallest meaningful change in the original dataset
# (criterion values are quantised to 0.25 steps).
_GRID_RESOLUTION: int = 1001


def per_criterion_contributions(
    uta_model: nn.Module, x: torch.Tensor
) -> np.ndarray:
    """Return the marginal utility :math:`u_j(g_j(a))` for each criterion.

    The ANN-UTADIS comprehensive utility is the row-sum of the monotonic
    block's output; this function returns the row itself, so the elements
    sum to the alternative's utility.

    Args:
        uta_model: A fitted ANN-UTADIS module whose ``.monotonic_layer``
            attribute is the spread → activation → combine stack.
        x: Tensor of shape ``(num_criteria,)`` or ``(1, num_criteria)``
            containing the alternative's criterion values.

    Returns:
        Numpy array of shape ``(num_criteria,)`` with the per-criterion
        contributions.
    """
    x = x.view(1, -1)
    with torch.no_grad():
        per_criterion = uta_model.monotonic_layer(x)
    return per_criterion.squeeze(0).cpu().numpy()


@dataclass(frozen=True)
class MinChangeResult:
    """Smallest single-criterion change that flips the predicted class."""

    criterion_index: int
    direction: str  # "increase" or "decrease"
    delta: float
    original_class: int
    new_class: int


def _predict_class(
    classifier: nn.Module, uta_model: nn.Module, x: torch.Tensor
) -> int:
    with torch.no_grad():
        utility = uta_model(x.view(1, -1))
        probs = classifier(utility)
    return int(torch.argmax(probs, dim=1).item())


def empirical_min_change(
    classifier: nn.Module,
    uta_model: nn.Module,
    x: torch.Tensor,
    *,
    target_class: int | None = None,
    grid_resolution: int = _GRID_RESOLUTION,
) -> list[MinChangeResult]:
    """Per-criterion empirical minimum change via grid scan over ``[0, 1]``.

    For each criterion, the value is swept from 0 to 1 in
    ``grid_resolution`` equally-spaced steps; the smallest delta on either
    side of the original value that changes the predicted class is recorded.

    Args:
        classifier: Module turning utility into class probabilities (typically
            ``NormLayer`` wrapping ``OrdinalThresholdLayer``, or just the
            threshold layer if the utility is already normalised).
        uta_model: Comprehensive utility model.
        x: Original alternative, shape ``(num_criteria,)``.
        target_class: If specified, only changes that land in this class are
            considered. Otherwise any class flip qualifies.
        grid_resolution: Number of candidate values per criterion.

    Returns:
        List of :class:`MinChangeResult`, one per criterion. Entries whose
        criterion cannot be flipped within ``[0, 1]`` have ``delta = inf``.
    """
    x = x.view(-1)
    num_criteria = x.shape[0]
    original_class = _predict_class(classifier, uta_model, x)

    grid = torch.linspace(0.0, 1.0, grid_resolution)
    results: list[MinChangeResult] = []

    for j in range(num_criteria):
        best: MinChangeResult | None = None
        for val in grid:
            if torch.isclose(val, x[j], atol=1e-6):
                continue
            x_alt = x.clone()
            x_alt[j] = val
            new_class = _predict_class(classifier, uta_model, x_alt)
            if new_class == original_class:
                continue
            if target_class is not None and new_class != target_class:
                continue
            delta = float((val - x[j]).item())
            if best is None or abs(delta) < abs(best.delta):
                best = MinChangeResult(
                    criterion_index=j,
                    direction="increase" if delta > 0 else "decrease",
                    delta=delta,
                    original_class=original_class,
                    new_class=new_class,
                )
        results.append(
            best
            or MinChangeResult(
                criterion_index=j,
                direction="none",
                delta=float("inf"),
                original_class=original_class,
                new_class=original_class,
            )
        )
    return results


def analytical_min_change(
    uta_model: nn.Module,
    threshold_layer: nn.Module,
    x: torch.Tensor,
    *,
    target_class: int,
    grid_resolution: int = _GRID_RESOLUTION,
) -> list[MinChangeResult]:
    """Analytical minimum change against the nearest threshold.

    Closed-form analysis is hard for the ANN-UTADIS marginal utility because
    the spread/combine layers are arbitrary monotonic functions, not linear.
    Instead, for each criterion we evaluate the model on a dense grid of
    values, take the resulting utility curve, and find the smallest delta
    whose utility crosses the target threshold. This is "analytical" in the
    sense that it uses the *utility curve* of the model rather than the
    classifier's discrete decision boundary — useful when the threshold is
    close to the alternative's utility and small empirical perturbations
    don't yet flip the class.

    Args:
        uta_model: Comprehensive utility model.
        threshold_layer: Module exposing ``.thresholds()`` returning the
            ``K - 1`` ordered class boundaries.
        x: Original alternative, shape ``(num_criteria,)``.
        target_class: The class we want to push the alternative into.
        grid_resolution: Number of candidate values per criterion.

    Returns:
        One :class:`MinChangeResult` per criterion. ``delta = inf`` when the
        target threshold is unreachable by changing that criterion alone.
    """
    x = x.view(-1)
    num_criteria = x.shape[0]
    with torch.no_grad():
        thresholds = threshold_layer.thresholds()
    if target_class == 0:
        target_utility = float(thresholds[0].item())
        direction_sign = -1
    elif target_class >= thresholds.shape[0]:
        target_utility = float(thresholds[-1].item())
        direction_sign = +1
    else:
        target_utility = float(thresholds[target_class - 1].item())
        direction_sign = +1

    grid = torch.linspace(0.0, 1.0, grid_resolution)
    results: list[MinChangeResult] = []
    original_class = int(
        torch.argmax(threshold_layer(uta_model(x.view(1, -1)))).item()
    )

    for j in range(num_criteria):
        x_batch = x.unsqueeze(0).expand(grid_resolution, -1).clone()
        x_batch[:, j] = grid
        with torch.no_grad():
            utility = uta_model(x_batch)
        sign = (utility - target_utility) * direction_sign
        # First grid point that satisfies the inequality determines the delta.
        crossings = torch.where(sign >= 0)[0]
        if crossings.numel() == 0:
            results.append(
                MinChangeResult(
                    criterion_index=j,
                    direction="none",
                    delta=float("inf"),
                    original_class=original_class,
                    new_class=original_class,
                )
            )
            continue
        # Pick the crossing with the smallest absolute change relative to x[j].
        deltas = grid[crossings] - x[j]
        best_idx = int(torch.argmin(deltas.abs()).item())
        delta = float(deltas[best_idx].item())
        results.append(
            MinChangeResult(
                criterion_index=j,
                direction="increase" if delta > 0 else ("decrease" if delta < 0 else "none"),
                delta=delta,
                original_class=original_class,
                new_class=target_class,
            )
        )
    return results
