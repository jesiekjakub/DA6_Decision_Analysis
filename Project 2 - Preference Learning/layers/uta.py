"""Top-level ANN-UTADIS comprehensive value function."""

from __future__ import annotations

import torch
import torch.nn as nn

from .monotonic_layer import MonotonicLayer


class Uta(nn.Module):
    """Comprehensive utility :math:`U(a) = \\sum_j u_j(g_j(a))`.

    A single monotonic block followed by a sum over criteria. The output is
    *unnormalized* and *unthresholded* — wrap it in
    :class:`~layers.norm_layer.NormLayer` for UTADIS anchoring and an
    :class:`~layers.threshold_layer.OrdinalThresholdLayer` for class
    probabilities.

    Args:
        num_criteria: Number of criteria m in the dataset.
        num_hidden_components: Number L of hidden components per criterion.
            Controls the resolution of the learned piecewise-linear marginal
            utility functions.
        slope: Initial slope passed to the :class:`LeakyHardSigmoid` inside
            the monotonic block.
        input_range: Range of raw criterion values, used to initialise the
            spread-layer biases.
    """

    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        slope: float = 0.01,
        input_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__()
        self.monotonic_layer = MonotonicLayer(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
            slope=slope,
            input_range=input_range,
        )

    def set_slope(self, val: float) -> None:
        self.monotonic_layer.set_slope(val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, num_criteria)`` inputs to ``(batch,)`` utilities."""
        return self.monotonic_layer(x).sum(dim=1)
