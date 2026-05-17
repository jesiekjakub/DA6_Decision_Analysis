"""Leaky hard sigmoid — the monotonic activation used inside ANN-UTADIS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakyHardSigmoid(nn.Module):
    """Monotonic, piecewise-linear approximation of the sigmoid.

    Built from two leaky-ReLU operations, which keeps the function strictly
    monotonic while avoiding the vanishing gradients of the standard hard
    sigmoid in its saturated regions.
    """

    def __init__(self, slope: float = 0.01) -> None:
        super().__init__()
        self.slope = slope

    def set_slope(self, val: float) -> None:
        self.slope = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(1.0 - F.leaky_relu(1.0 - x, self.slope), self.slope)
