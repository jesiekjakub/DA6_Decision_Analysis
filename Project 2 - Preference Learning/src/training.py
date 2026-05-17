"""Generic training loop used by both the ANN-UTADIS and the deep MLP models.

The notebook trains each model with a near-identical loop; this module is the
deduplicated version. The loop is intentionally minimal — no progress bars,
no curriculum tricks beyond the optional linear slope schedule for
ANN-UTADIS — because the report focuses on the comparison, not on training
infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EarlyStopping:
    """Patience-based early stopping over a validation loss series.

    The implementation is deliberately tiny: it keeps the best loss seen so
    far and the parameters that produced it, and signals stop once
    ``patience`` consecutive epochs fail to improve on the best by at least
    ``min_delta``.
    """

    patience: int = 20
    min_delta: float = 0.0
    best_loss: float = float("inf")
    counter: int = 0
    best_state: Optional[dict] = field(default=None, repr=False)
    should_stop: bool = False

    def step(self, current_loss: float, model: nn.Module) -> None:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _linear_slope(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """Linear interpolation from ``start`` to ``end`` across ``total_epochs``."""
    if total_epochs <= 1:
        return end
    t = epoch / (total_epochs - 1)
    return start + (end - start) * t


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    epochs: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    slope_schedule: Optional[tuple[float, float]] = None,
    early_stopping: Optional[EarlyStopping] = None,
    device: str = "cpu",
) -> list[float]:
    """Generic supervised training loop with optional slope and LR schedules.

    Args:
        model: Module to train. If ``slope_schedule`` is given, the model
            must expose ``set_slope(float)``.
        train_loader: Yields ``(features, target)`` tensors.
        val_loader: Optional validation loader; required when
            ``early_stopping`` is provided.
        optimizer: Pre-configured optimiser.
        loss_fn: Callable computing the scalar loss from ``(pred, target)``.
        epochs: Number of training epochs.
        scheduler: Optional LR scheduler stepped once per epoch.
        slope_schedule: ``(start, end)`` linear schedule fed to
            ``model.set_slope`` at the start of each epoch.
        early_stopping: Optional early-stopping policy on the validation loss.
        device: Torch device string.

    Returns:
        Per-epoch validation losses (training loss when no validation loader
        is provided).
    """
    history: list[float] = []
    model.to(device)

    for epoch in range(epochs):
        if slope_schedule is not None and hasattr(model, "set_slope"):
            model.set_slope(_linear_slope(epoch, epochs, *slope_schedule))

        model.train()
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(features)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            model.eval()
            losses: list[float] = []
            loader = val_loader or train_loader
            for features, target in loader:
                features = features.to(device)
                target = target.to(device)
                losses.append(loss_fn(model(features), target).item())
            epoch_loss = sum(losses) / len(losses)
            history.append(epoch_loss)

        if early_stopping is not None:
            early_stopping.step(epoch_loss, model)
            if early_stopping.should_stop:
                break

    if early_stopping is not None:
        early_stopping.restore(model)
    return history
