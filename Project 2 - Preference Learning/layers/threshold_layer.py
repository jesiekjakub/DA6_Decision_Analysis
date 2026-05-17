"""Binary and multiclass-ordinal threshold heads for ANN-UTADIS."""

from __future__ import annotations

import torch
import torch.nn as nn

# Lower bound on the input to softplus⁻¹. softplus⁻¹(t) = log(exp(t) - 1)
# blows up as t → 0⁺ and is undefined for t ≤ 0; clamping the raw initial
# step away from zero keeps the initial parameter finite even in the unusual
# case of ``num_classes = 1`` requested with one threshold per class.
_SOFTPLUS_INVERSE_MIN: float = 1e-6


def _softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    """Inverse of ``softplus`` for strictly positive inputs."""
    safe = torch.clamp(value, min=_SOFTPLUS_INVERSE_MIN)
    return torch.log(torch.expm1(safe))


class ThresholdLayer(nn.Module):
    """Single learnable scalar threshold for the binary ANN-UTADIS variant.

    Args:
        threshold: Fixed initial value. ``None`` initialises uniformly in
            ``(0.1, 0.9)``.
        requires_grad: Whether the threshold is trainable.
    """

    def __init__(
        self,
        threshold: float | None = None,
        requires_grad: bool = True,
    ) -> None:
        super().__init__()
        if threshold is None:
            init = torch.empty(1).uniform_(0.1, 0.9)
        else:
            init = torch.tensor([threshold], dtype=torch.float32)
        self.threshold = nn.Parameter(init, requires_grad=requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.threshold


class OrdinalThresholdLayer(nn.Module):
    """Multiclass ordinal thresholds for K-class sorting.

    Holds ``K - 1`` thresholds :math:`t_1 < t_2 < \\dots < t_{K-1}` and
    assigns an alternative with utility :math:`U(a)` to class :math:`k` iff
    :math:`t_{k-1} \\le U(a) < t_k` (with :math:`t_0 = -\\infty`,
    :math:`t_K = +\\infty`).

    Monotonic ordering is enforced via a cumulative-softplus parameterisation:
    only the first raw value is unconstrained, and each subsequent threshold
    is the previous one plus a non-negative gap obtained as
    ``softplus(raw[k])``. This keeps the thresholds strictly ordered for any
    real-valued ``raw`` vector and makes the parameterisation differentiable
    everywhere — useful both for backprop and for SHAP attributions on the
    logits.

    The forward pass produces a proper probability distribution by reusing
    the cumulative-sigmoid construction of CORAL-style ordinal regression:

    .. code-block::

        P(y >= k | U) = sigmoid((U - t_{k-1}) / temperature)
        P(y == k | U) = P(y >= k) - P(y >= k+1)

    At low ``temperature`` the distribution concentrates on the bucket
    selected by :meth:`predict`, so ``argmax`` over the probabilities
    matches a hard bucketize on the thresholds.

    Args:
        num_classes: Number of classes K. Creates ``K - 1`` thresholds.
        temperature: Sharpness of the cumulative sigmoids; lower is sharper.
    """

    def __init__(
        self,
        num_classes: int,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.temperature = temperature

        # Initialise thresholds at evenly-spaced points 1/K, 2/K, …, (K-1)/K.
        # Only the first raw value is the literal threshold; subsequent raw
        # values feed softplus → gap, so the inverse-softplus of the step
        # size makes the *initial* effective threshold spacing equal to step.
        step = 1.0 / num_classes
        raw_init = torch.empty(self.num_thresholds)
        raw_init[0] = step
        if self.num_thresholds > 1:
            raw_init[1:] = _softplus_inverse(torch.tensor(step))
        self.raw = nn.Parameter(raw_init)

    def thresholds(self) -> torch.Tensor:
        """The current K-1 strictly-ordered thresholds."""
        if self.num_thresholds == 1:
            return self.raw
        gaps = torch.nn.functional.softplus(self.raw[1:])
        return torch.cat([self.raw[:1], self.raw[:1] + torch.cumsum(gaps, dim=0)])

    def forward(self, utility: torch.Tensor) -> torch.Tensor:
        """Class probabilities from a batch of utilities.

        Args:
            utility: Shape ``(batch,)`` comprehensive utilities.

        Returns:
            Tensor of shape ``(batch, num_classes)`` with per-class probabilities.
        """
        t = self.thresholds()
        u = utility.unsqueeze(1)
        tau = self.temperature
        cumulative = torch.sigmoid((u - t.unsqueeze(0)) / tau)  # P(y > k) for k = 0..K-2
        one = torch.ones_like(cumulative[:, :1])
        zero = torch.zeros_like(cumulative[:, :1])
        p_ge = torch.cat([one, cumulative], dim=1)
        p_gt = torch.cat([cumulative, zero], dim=1)
        return p_ge - p_gt
