"""First stage of the ANN-UTADIS monotonic block."""

from __future__ import annotations

import torch
import torch.nn as nn

# Wide initial weight range keeps gradients alive through the LeakyHardSigmoid
# plateau at the start of training; tighter ranges (e.g. uniform(0, 1)) tend to
# park the hidden components in the linear interior of the activation and learn
# slowly.
_WEIGHT_INIT_LOW: float = 1.0
_WEIGHT_INIT_HIGH: float = 10.0
# Floor applied to weights that drift below zero during training. Monotonicity
# of the whole block depends on these weights staying non-negative.
_WEIGHT_FLOOR: float = 0.0


class CriterionLayerSpread(nn.Module):
    """Spread each criterion into L trainable affine projections.

    The first stage of the monotonic block: for each criterion :math:`g_j(a)`
    the layer outputs ``num_hidden_components`` values of the form
    :math:`w_{l,j}\\,(g_j(a) + b_{l,j})`. These are passed through
    :class:`LeakyHardSigmoid` to form a piecewise-linear marginal utility.
    """

    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        input_range: tuple[float, float] = (0.0, 1.0),
        normalize_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_criteria = num_criteria
        # Negated to make the additive form ``x + b`` shift inputs into the
        # activation's working range when ``b`` is drawn from this interval.
        negated = (-input_range[0], -input_range[1])
        self.max_bias = max(negated)
        self.min_bias = min(negated)
        self.normalize_bias = normalize_bias

        self.bias = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.weight = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, _WEIGHT_INIT_LOW, _WEIGHT_INIT_HIGH)
        nn.init.uniform_(self.bias, self.min_bias, self.max_bias)

    def compute_bias(self) -> torch.Tensor:
        if self.normalize_bias:
            return torch.clamp(self.bias, self.min_bias, self.max_bias)
        return self.bias

    def compute_weight(self) -> torch.Tensor:
        # Hard floor at zero so the block stays monotonic. ``torch.clamp``
        # returns a non-leaf temporary; the underlying parameter is left
        # untouched, gradient still flows through the clamp.
        return torch.clamp(self.weight, min=_WEIGHT_FLOOR)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1, self.num_criteria)
        return (x + self.compute_bias()) * self.compute_weight()
