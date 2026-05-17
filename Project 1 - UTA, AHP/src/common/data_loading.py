"""Readers for the processed dataset, the DM preferences, and the consistency subset.

All loaders raise :class:`MissingDataError` for absent files and
:class:`InconsistentDataError` when the column schema does not match what the
solvers expect, so failures surface with a useful path instead of a generic
``FileNotFoundError`` or ``KeyError`` deep inside pandas.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_FILE, METADATA_FILE, PREFERENCES_FILE, SELECTED_SUBSET_FILE
from .exceptions import InconsistentDataError, MissingDataError


def _read_csv(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise MissingDataError(path)
    df = pd.read_csv(path)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise InconsistentDataError(
            f"{path.name} is missing required columns: {missing} "
            f"(found: {list(df.columns)})"
        )
    return df


def load_data() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the processed dataset and the per-criterion gain/cost map.

    Returns:
        A pair ``(df, directions)`` where ``df`` is the wide-format dataset
        with one row per country, and ``directions`` maps each criterion name
        to ``+1`` (gain — higher is better) or ``-1`` (cost — lower is better).
    """
    df = _read_csv(DATASET_FILE, required_columns=("Country",))
    meta = _read_csv(METADATA_FILE, required_columns=("criterion", "nature"))

    directions: dict[str, int] = {}
    for _, row in meta.iterrows():
        nature = row["nature"]
        if nature not in {"gain", "cost"}:
            raise InconsistentDataError(
                f"Unknown criterion nature '{nature}' in {METADATA_FILE.name}"
            )
        directions[row["criterion"]] = 1 if nature == "gain" else -1
    return df, directions


def load_preferences() -> list[tuple[str, str]]:
    """Load the 20 DM pairwise comparisons as ``(preferred, over)`` tuples."""
    df = _read_csv(PREFERENCES_FILE, required_columns=("preferred", "over"))
    return list(zip(df["preferred"].tolist(), df["over"].tolist()))


def load_removal_indices() -> set[int]:
    """Indices of preferences removed in task 2.1 to restore consistency."""
    df = _read_csv(SELECTED_SUBSET_FILE, required_columns=("removed_index",))
    return set(df["removed_index"].astype(int).tolist())
