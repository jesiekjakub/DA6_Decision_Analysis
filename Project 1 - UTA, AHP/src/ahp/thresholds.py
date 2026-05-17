"""Per-criterion difference → Saaty-score lookup tables.

Each entry maps an upper-exclusive ``|diff|`` boundary to a Saaty intensity in
[1, 9]. The boundaries are calibrated to each criterion's natural scale —
employment-rate differences are measured in percentage points, earnings in
dollars, distance in kilometres, etc. — so a single fixed bin size would
either flatten the small-range criteria or saturate the large-range ones.
A ``|diff|`` greater than every listed boundary maps to the extreme score 9.
"""

from __future__ import annotations

from typing import Final

# Tuples are ``(upper_bound, score)``. Lookup walks the list in order and
# returns the first matching score; the trailing 9 falls out of the lookup
# when nothing matches.
_ThresholdTable = list[tuple[float, int]]

THRESHOLDS: Final[dict[str, _ThresholdTable]] = {
    "Employment rate": [
        (2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6), (16, 7), (20, 8),
    ],
    "Long-term unemployment rate": [
        (0.5, 1), (1.0, 2), (2.0, 3), (3.0, 4), (4.0, 5), (5.0, 6), (6.0, 7), (8.0, 8),
    ],
    "Personal earnings": [
        (2000, 1), (5000, 2), (10000, 3), (15000, 4),
        (20000, 5), (25000, 6), (30000, 7), (35000, 8),
    ],
    "Life expectancy": [
        (0.5, 1), (1.0, 2), (2.0, 3), (3.0, 4), (4.0, 5), (5.5, 6), (6.5, 7), (7.5, 8),
    ],
    "Life satisfaction": [
        (0.1, 1), (0.2, 2), (0.4, 3), (0.6, 4), (0.8, 5), (1.0, 6), (1.2, 7), (1.6, 8),
    ],
    "Employees working very long hours": [
        (0.5, 1), (1.5, 2), (2.5, 3), (3.5, 4), (5.0, 5), (6.5, 6), (8.0, 7), (9.5, 8),
    ],
    "Air pollution": [
        (1.0, 1), (2.0, 2), (4.0, 3), (6.0, 4), (8.0, 5), (10.0, 6), (13.0, 7), (16.0, 8),
    ],
    "Distance from Poznan (km)": [
        (100, 1), (250, 2), (450, 3), (650, 4), (900, 5), (1200, 6), (1600, 7), (2000, 8),
    ],
}

_MAX_SAATY_SCORE: Final[int] = 9


def diff_to_score(abs_diff: float, table: _ThresholdTable) -> int:
    """Saaty intensity score for an absolute criterion difference."""
    for upper, score in table:
        if abs_diff < upper:
            return score
    return _MAX_SAATY_SCORE
