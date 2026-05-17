"""Min-max normalizer that anchors U(0) = 0 and U(1) = 1."""

from __future__ import annotations

import torch
import torch.nn as nn

# Floor on the denominator (U(1) - U(0)) to keep the gradient finite when the
# wrapped network briefly collapses to a constant during training.
_DENOMINATOR_EPS: float = 1e-12


class NormLayer(nn.Module):
    """Re-scale the wrapped module's output to satisfy UTADIS anchoring.

    Subtracts :math:`U(\\mathbf{0})` and divides by :math:`U(\\mathbf{1}) - U(\\mathbf{0})`,
    so that the comprehensive utility starts at 0 and ends at 1. This makes
    the learned class thresholds directly comparable across runs and gives
    the marginal utility plots a natural axis.

    The wrapped module is queried three times per forward pass — once with
    the real batch, once with all-zeros, once with all-ones. Caching the
    anchoring values would shave a small amount of compute but require
    invalidation on every ``set_slope`` call, so it is left to the future.
    """

    def __init__(self, method_instance: nn.Module, num_criteria: int) -> None:
        super().__init__()
        self.method_instance = method_instance
        self.num_criteria = num_criteria

    def set_slope(self, slope: float) -> None:
        self.method_instance.set_slope(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.method_instance(x)

        zero_input = torch.zeros(1, self.num_criteria, device=out.device, dtype=x.dtype)
        one_input = torch.ones(1, self.num_criteria, device=out.device, dtype=x.dtype)
        zero = self.method_instance(zero_input)
        one = self.method_instance(one_input)

        return (out - zero) / (one - zero + _DENOMINATOR_EPS)
