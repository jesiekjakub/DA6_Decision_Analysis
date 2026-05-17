"""CSV loaders for the PROMETHEE and ELECTRE Tri-B exercises.

The exercise datasets share a common layout: one ``dataset.csv`` indexed by
alternative, one ``preference.csv`` indexed by criterion (with a ``type``
column on ``CriterionType``), and — for ELECTRE Tri-B — one
``boundary_profiles.csv`` indexed by profile.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .types import CriterionType


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the alternatives × criteria matrix from ``dataset_path / dataset.csv``."""
    return pd.read_csv(dataset_path / "dataset.csv", index_col=0)


def load_preference_information(dataset_path: Path) -> pd.DataFrame:
    """Load the per-criterion preference parameters.

    The ``type`` column is converted to a pandas ``Categorical`` over the
    :class:`CriterionType` values so downstream comparisons against
    ``CriterionType.GAIN`` / ``CriterionType.COST`` work without re-coercion.
    """
    preferences = pd.read_csv(dataset_path / "preference.csv", index_col=0)
    preferences["type"] = pd.Categorical(preferences["type"], list(CriterionType))
    return preferences


def load_boundary_profiles(dataset_path: Path) -> pd.DataFrame:
    """Load ELECTRE Tri-B boundary profiles from ``dataset_path / boundary_profiles.csv``."""
    return pd.read_csv(dataset_path / "boundary_profiles.csv", index_col=0)
