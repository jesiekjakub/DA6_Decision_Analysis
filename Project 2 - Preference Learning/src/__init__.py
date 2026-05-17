"""Reusable training, explanation, and configuration helpers for the report notebook.

The notebook in :mod:`notebooks/report.ipynb` defines its training loops,
SHAP setups, and minimum-change analyses inline. The helpers here are the
importable library version of that code — same algorithms, no notebook
plumbing — so they can be reused from scripts and from future notebooks
without copy/paste.
"""

from .config import AnnUtadisConfig, DatasetConfig, MlpConfig, XgbConfig
from .explain import (
    MinChangeResult,
    analytical_min_change,
    empirical_min_change,
    per_criterion_contributions,
)
from .training import EarlyStopping, train_classifier

__all__ = [
    "AnnUtadisConfig",
    "DatasetConfig",
    "EarlyStopping",
    "MinChangeResult",
    "MlpConfig",
    "XgbConfig",
    "analytical_min_change",
    "empirical_min_change",
    "per_criterion_contributions",
    "train_classifier",
]
