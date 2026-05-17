"""Third stage of the ANN-UTADIS monotonic block."""

from __future__ import annotations

import torch
import torch.nn as nn

# Bounds for the uniform weight initialisation. The resulting weights are
# renormalised to sum to one across the whole tensor; the choice here just
# avoids initialising components to exactly zero (which kills their gradient
# through the multiplicative mixture).
_WEIGHT_INIT_LOW: float = 0.2
_WEIGHT_INIT_HIGH: float = 1.0


class CriterionLayerCombine(nn.Module):
    """Aggregate L hidden-component outputs back to one value per criterion.

    Uses non-negative weights so the whole monotonic block stays monotonic.
    Weights are clamped at ``min_weight`` from below in :meth:`compute_weight`
    — the clamp operates on a temporary, so the underlying parameter is left
    free for the optimiser to update naturally.
    """

    def __init__(
        self,
        num_criteria: int,
        num_hidden_components: int,
        min_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.min_weight = min_weight
        self.weight = nn.Parameter(torch.empty(num_hidden_components, num_criteria))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, _WEIGHT_INIT_LOW, _WEIGHT_INIT_HIGH)
        with torch.no_grad():
            self.weight.data.div_(self.weight.data.sum())

    def compute_weight(self) -> torch.Tensor:
        # Lower clamp keeps a small but non-zero contribution from every
        # hidden component; otherwise once a weight crosses zero the
        # downstream gradient through that component is permanently zero.
        return torch.clamp(self.weight, min=self.min_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.compute_weight()).sum(dim=1)
