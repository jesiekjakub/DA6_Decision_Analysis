"""Three-stage monotonic block: spread → leaky-hard-sigmoid → combine."""

from __future__ import annotations

import torch
import torch.nn as nn

from .criterion_layer_combine import CriterionLayerCombine
from .criterion_layer_spread import CriterionLayerSpread
from .leaky_hard_sigmoid import LeakyHardSigmoid


class MonotonicLayer(nn.Sequential):
    """One marginal utility value per criterion, with guaranteed monotonicity.

    Stages:

    1. :class:`CriterionLayerSpread` — affine projections with non-negative weights.
    2. :class:`LeakyHardSigmoid` — monotonic activation.
    3. :class:`CriterionLayerCombine` — non-negative mixture over hidden components.

    Each stage is individually monotonic on its input, so the composition is
    monotonic with respect to every criterion. The inherited ``forward`` from
    :class:`nn.Sequential` is sufficient — no override is needed.

    Args:
        num_criteria: Number of criteria m.
        num_hidden_components: Number L of hidden components per criterion.
        slope: Initial slope of the leaky-hard-sigmoid activation.
        input_range: Range of raw input values used to initialise the
            spread-layer biases.
        min_combine_weight: Floor applied to the combine-layer weights.
    """

    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        slope: float = 0.01,
        input_range: tuple[float, float] = (0.0, 1.0),
        min_combine_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.criterion_layer_spread = CriterionLayerSpread(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
            input_range=input_range,
        )
        self.activation_function = LeakyHardSigmoid(slope=slope)
        self.criterion_layer_combine = CriterionLayerCombine(
            num_criteria=num_criteria,
            num_hidden_components=num_hidden_components,
            min_weight=min_combine_weight,
        )

    def set_slope(self, val: float) -> None:
        self.activation_function.set_slope(val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # nn.Sequential's default forward would also chain the children; the
        # explicit type-annotated wrapper is here so editors and mypy report
        # the actual signature instead of nn.Module's permissive default.
        return super().forward(x)
