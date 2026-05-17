"""Project-wide file paths and UTA model parameters.

The model constants are kept in :class:`UtaConfig` so they can be passed
around explicitly when needed; the module level also re-exports them for the
common ``from common.config import GAMMA`` access pattern used in scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
PREFERENCES_DIR: Path = DATA_DIR / "preferences"
OUTPUT_DIR: Path = DATA_DIR / "output"

DATASET_FILE: Path = PROCESSED_DIR / "dataset.csv"
METADATA_FILE: Path = PROCESSED_DIR / "criteria_metadata.csv"
PREFERENCES_FILE: Path = PREFERENCES_DIR / "preferences.csv"
SELECTED_SUBSET_FILE: Path = PREFERENCES_DIR / "selected_consistent_subset.csv"
CAPITALS_FILE: Path = PROCESSED_DIR / "european_capitals.json"


@dataclass(frozen=True)
class UtaConfig:
    """Numerical parameters of the UTA additive value-function model.

    Bounds come from the project specification: a single criterion may carry
    at most half of the total weight and at least 1/(2n) of it, where n is
    the number of criteria. ``MIN_SEGMENT_SHARE`` and ``NON_LINEARITY_THRESHOLD``
    together prevent degenerate piecewise-linear functions (flat segments and
    near-linear shapes that don't reflect the DM's actual preferences).
    """

    gamma: int = 4
    weight_ub: float = 0.5
    weight_lb: float = 0.0625
    delta: float = 1e-3
    min_segment_share: float = 0.15
    non_linearity_threshold: float = 0.25

    @property
    def num_characteristic_points(self) -> int:
        return self.gamma + 1


DEFAULT_CONFIG: UtaConfig = UtaConfig()

GAMMA: int = DEFAULT_CONFIG.gamma
WEIGHT_UB: float = DEFAULT_CONFIG.weight_ub
WEIGHT_LB: float = DEFAULT_CONFIG.weight_lb
DELTA: float = DEFAULT_CONFIG.delta
MIN_SEGMENT_SHARE: float = DEFAULT_CONFIG.min_segment_share
NON_LINEARITY_THRESHOLD: float = DEFAULT_CONFIG.non_linearity_threshold
