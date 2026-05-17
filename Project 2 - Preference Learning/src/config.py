"""Hyperparameter configurations for the three comparison models.

Each dataclass is a frozen named bundle of the magic numbers the notebook
scatters across cells. Importing these from the notebook is optional — the
notebook is self-contained — but using them keeps the numbers in one place
and makes the comparison reproducible from a fresh kernel.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Loader and splitter parameters."""

    csv_path: str = "../data/lectures evaluation.csv"
    criteria_columns: tuple[str, ...] = ("c1", "c2", "c3", "c4")
    target_column: str = "quality"
    class_merge: tuple[int, ...] = (0, 0, 1, 2, 2)
    seed: int = 1234
    test_size: float = 0.2


@dataclass(frozen=True)
class XgbConfig:
    """Hyperparameters for the monotone XGBoost baseline."""

    max_depth: int = 3
    learning_rate: float = 0.1
    n_estimators: int = 100
    objective: str = "multi:softprob"
    # Monotonicity constraint applied to every criterion. Tuple length must
    # match the number of feature columns.
    monotone_constraints: tuple[int, ...] = (1, 1, 1, 1)
    seed: int = 1234


@dataclass(frozen=True)
class AnnUtadisConfig:
    """Hyperparameters for the ANN-UTADIS interpretable model."""

    num_hidden_components: int = 50
    learning_rate: float = 1e-2
    epochs: int = 400
    weight_decay: float = 1e-2
    # Linear slope schedule from `slope_start` at epoch 0 to `slope_end` at
    # the final epoch. Decaying the slope sharpens the leaky-hard-sigmoid as
    # training converges, which empirically tightens the learned marginal
    # utility functions in the report.
    slope_start: float = 1e-2
    slope_end: float = 3e-3
    ordinal_temperature: float = 5e-2
    seed: int = 1234


@dataclass(frozen=True)
class MlpConfig:
    """Hyperparameters for the deep MLP baseline."""

    hidden_sizes: tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.2
    learning_rate: float = 5e-3
    epochs: int = 600
    weight_decay: float = 1e-4
    seed: int = 1234
