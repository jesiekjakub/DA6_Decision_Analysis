"""Decision-maker pairwise comparison matrices on the Saaty 1-9 scale.

The Goal-level matrix carries an intentional cycle that violates the
consistency ratio threshold (CR < 0.10) — Economic ≈ 5 × Social and
Social ≈ 2 × Geography would imply Economic ≈ 10 × Geography, but the DM
recorded 3 × instead. The mismatch is deliberate: the task exercises the
consistency diagnostics implemented in :mod:`ahp.consistency`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Goal-level: [Economic, Social/Health, Geography/Environment]
A_goal: NDArray[np.float64] = np.array(
    [
        [1,    5,    3  ],
        [1/5,  1,    2  ],
        [1/3,  1/2,  1  ],
    ],
    dtype=float,
)

# Economic: [Employment rate, Long-term unemployment rate, Personal earnings]
A_economic: NDArray[np.float64] = np.array(
    [
        [1,    1/2,  1/5],
        [2,    1,    1/5],
        [5,    5,    1  ],
    ],
    dtype=float,
)

# Social/Health: [Life expectancy, Life satisfaction, Working long hours]
A_social: NDArray[np.float64] = np.array(
    [
        [1,    1/3,  1/2],
        [3,    1,    2  ],
        [2,    1/2,  1  ],
    ],
    dtype=float,
)

# Geography/Environment: [Air pollution, Distance from Poznan]
A_geography: NDArray[np.float64] = np.array(
    [
        [1,    1/3],
        [3,    1  ],
    ],
    dtype=float,
)

MATRICES: dict[str, NDArray[np.float64]] = {
    "Goal":                  A_goal,
    "Economic":              A_economic,
    "Social/Health":         A_social,
    "Geography/Environment": A_geography,
}
