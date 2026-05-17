"""ETL: raw OECD Better Life Index CSV → wide-format MCDA dataset.

Pipeline:

1. Locate the raw CSV in ``data/raw/`` (the user supplies it from Kaggle).
2. Filter to European OECD members and the eight chosen indicators.
3. Drop non-Total inequality rows when an Inequality column is present.
4. Pivot long → wide.
5. Resolve NaNs by row-drop first, falling back to median imputation if the
   row-drop would push the alternative count below the spec minimum.
6. Append the haversine distance from Poznań to each capital.
7. Sanity-check sizes and write ``data/processed/dataset.csv``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Final

import pandas as pd

from common.config import CAPITALS_FILE, DATASET_FILE, RAW_DIR
from common.exceptions import InconsistentDataError, MissingDataError
from common.logging_utils import get_logger

_LOG: Final = get_logger(__name__)

POZNAN_LAT: Final[float] = 52.4064
POZNAN_LON: Final[float] = 16.9252
EARTH_RADIUS_KM: Final[float] = 6371.0

# Sanity-check bounds from the project specification.
_MIN_ALTERNATIVES: Final[int] = 12
_MAX_ALTERNATIVES: Final[int] = 50
_MIN_CRITERIA: Final[int] = 4
_MAX_CRITERIA: Final[int] = 9

SELECTED_INDICATORS: Final[tuple[str, ...]] = (
    "Employment rate",
    "Long-term unemployment rate",
    "Personal earnings",
    "Life expectancy",
    "Life satisfaction",
    "Employees working very long hours",
    "Air pollution",
)

EUROPEAN_COUNTRIES: Final[tuple[str, ...]] = (
    "Austria", "Belgium", "Czech Republic", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Iceland",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Netherlands", "Norway", "Poland", "Portugal", "Slovak Republic",
    "Slovenia", "Spain", "Sweden", "Switzerland", "United Kingdom",
)

_INEQUALITY_COLUMNS: Final[tuple[str, ...]] = ("Inequality", "INEQUALITY")
_TOTAL_VALUES: Final[tuple[str, ...]] = ("Total", "TOT")


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_raw_csv() -> Path:
    """Locate the single Kaggle CSV in ``data/raw/`` or fail with a useful message."""
    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise MissingDataError(
            RAW_DIR,
            hint=(
                "no CSV found. Download "
                "https://www.kaggle.com/datasets/joebeachcapital/oecd-better-life-index "
                "and place the file in this directory."
            ),
        )
    if len(csvs) > 1:
        _LOG.warning("Multiple CSV files in %s; using %s", RAW_DIR, csvs[0].name)
    return csvs[0]


def explore_dataset(df: pd.DataFrame) -> None:
    """Log raw-dataset shape and the indicator/country coverage at INFO level."""
    _LOG.info("Raw dataset shape: %s", df.shape)
    _LOG.info("Columns: %s", list(df.columns))

    if "Indicator" in df.columns:
        indicators = sorted(df["Indicator"].unique())
        _LOG.info("Indicators in raw data (%d):", len(indicators))
        for i, ind in enumerate(indicators, 1):
            marker = " <-- SELECTED" if ind in SELECTED_INDICATORS else ""
            _LOG.info("  %2d. %s%s", i, ind, marker)

    if "Country" in df.columns:
        countries = sorted(df["Country"].unique())
        _LOG.info("Countries in raw data (%d):", len(countries))
        for c in countries:
            marker = " <-- EUROPEAN" if c in EUROPEAN_COUNTRIES else ""
            _LOG.info("  - %s%s", c, marker)

    for col in ("Inequality", "INEQUALITY", "Measure", "MEASURE"):
        if col in df.columns:
            _LOG.info("Unique values in '%s': %s", col, sorted(df[col].unique()))


def _filter_to_totals(df: pd.DataFrame) -> pd.DataFrame:
    inequality_col = next((c for c in _INEQUALITY_COLUMNS if c in df.columns), None)
    if inequality_col is None:
        return df
    present = set(df[inequality_col].unique())
    keep = next((v for v in _TOTAL_VALUES if v in present), None)
    if keep is None:
        _LOG.warning(
            "No 'Total' or 'TOT' value in column '%s'; keeping all rows", inequality_col
        )
        return df
    return df[df[inequality_col] == keep]


def _resolve_nans(df_wide: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaNs if doing so keeps enough alternatives, else median-impute."""
    indicator_cols = list(SELECTED_INDICATORS)
    nan_counts = df_wide[indicator_cols].isna().sum()
    if not nan_counts.any():
        return df_wide

    _LOG.info("Missing values per indicator:\n%s", nan_counts[nan_counts > 0])
    rows_with_nan = df_wide[indicator_cols].isna().any(axis=1)
    _LOG.info(
        "Countries with missing data: %s",
        df_wide.loc[rows_with_nan, "Country"].tolist(),
    )

    df_clean = df_wide.dropna(subset=indicator_cols)
    if len(df_clean) >= _MIN_ALTERNATIVES:
        _LOG.info("Dropping NaN rows; %d countries remain", len(df_clean))
        return df_clean

    _LOG.warning(
        "Only %d countries after dropna (< %d). Falling back to median imputation.",
        len(df_clean),
        _MIN_ALTERNATIVES,
    )
    for col in indicator_cols:
        median_val = df_wide[col].median()
        n_filled = df_wide[col].isna().sum()
        if n_filled > 0:
            df_wide[col] = df_wide[col].fillna(median_val)
            _LOG.warning("Filled %d NaN in '%s' with median=%.2f", n_filled, col, median_val)
    return df_wide


def _attach_distances(df_wide: pd.DataFrame) -> pd.DataFrame:
    if not CAPITALS_FILE.is_file():
        raise MissingDataError(CAPITALS_FILE)
    capitals = json.loads(CAPITALS_FILE.read_text())

    distances: list[float] = []
    for country in df_wide["Country"]:
        if country not in capitals:
            raise InconsistentDataError(
                f"Country '{country}' missing from {CAPITALS_FILE.name}"
            )
        cap = capitals[country]
        distances.append(round(haversine(POZNAN_LAT, POZNAN_LON, cap["lat"], cap["lon"]), 1))
    df_wide["Distance from Poznan (km)"] = distances
    return df_wide


def _validate_final(df: pd.DataFrame, criteria: list[str]) -> None:
    n_alts, n_crit = len(df), len(criteria)
    if not _MIN_ALTERNATIVES <= n_alts <= _MAX_ALTERNATIVES:
        raise InconsistentDataError(
            f"Alternatives count {n_alts} outside [{_MIN_ALTERNATIVES}, {_MAX_ALTERNATIVES}]"
        )
    if not _MIN_CRITERIA <= n_crit <= _MAX_CRITERIA:
        raise InconsistentDataError(
            f"Criteria count {n_crit} outside [{_MIN_CRITERIA}, {_MAX_CRITERIA}]"
        )
    if df[criteria].isna().any().any():
        raise InconsistentDataError("NaN values remain in the final dataset")


def main() -> None:
    """CLI entry point: run the full ETL pipeline and write ``dataset.csv``."""
    raw_csv = find_raw_csv()
    _LOG.info("Reading: %s", raw_csv.name)
    df = pd.read_csv(raw_csv)

    explore_dataset(df)

    if "Country" not in df.columns:
        raise InconsistentDataError(f"'Country' column missing from {raw_csv.name}")
    if "Indicator" not in df.columns:
        raise InconsistentDataError(f"'Indicator' column missing from {raw_csv.name}")

    df_europe = df[df["Country"].isin(EUROPEAN_COUNTRIES)]
    found = set(df_europe["Country"].unique())
    missing = set(EUROPEAN_COUNTRIES) - found
    if missing:
        _LOG.warning("European countries missing from dataset: %s", sorted(missing))
    _LOG.info("European countries found: %d", len(found))

    available_indicators = set(df["Indicator"].unique())
    missing_indicators = set(SELECTED_INDICATORS) - available_indicators
    if missing_indicators:
        _LOG.warning("Indicators missing from dataset: %s", sorted(missing_indicators))
        for mi in missing_indicators:
            keyword = mi.split()[0].lower()
            matches = [i for i in available_indicators if keyword in i.lower()]
            _LOG.warning("  '%s' -> possible matches: %s", mi, matches)

    df_filtered = _filter_to_totals(df_europe[df_europe["Indicator"].isin(SELECTED_INDICATORS)])

    # aggfunc="first" because the long-format rows for a (country, indicator)
    # pair are duplicates after filtering to 'Total' inequality; using pivot()
    # without an aggregator would raise on the duplicates that still exist
    # before the Total filter has fully reduced the slice.
    df_wide = df_filtered.pivot_table(
        index="Country",
        columns="Indicator",
        values="Value",
        aggfunc="first",
    ).reset_index()
    _LOG.info("Pivoted dataset shape: %s", df_wide.shape)

    df_wide = _resolve_nans(df_wide)
    df_wide = _attach_distances(df_wide)

    final_columns = ["Country", *SELECTED_INDICATORS, "Distance from Poznan (km)"]
    df_wide = df_wide[final_columns].sort_values("Country").reset_index(drop=True)
    _validate_final(df_wide, final_columns[1:])

    _LOG.info(
        "Final dataset: %d alternatives, %d criteria",
        len(df_wide),
        len(final_columns) - 1,
    )
    _LOG.info("\n%s", df_wide.to_string(index=False))

    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(DATASET_FILE, index=False)
    _LOG.info("Dataset saved to: %s", DATASET_FILE)


if __name__ == "__main__":
    try:
        main()
    except (MissingDataError, InconsistentDataError) as err:
        _LOG.error("%s", err)
        sys.exit(1)
