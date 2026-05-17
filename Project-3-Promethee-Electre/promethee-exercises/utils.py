"""Re-exports from :mod:`mcda_common` so the notebook's ``import utils`` keeps working.

The PROMETHEE notebook executes from this directory, so ``mcda_common`` —
sitting one level up at the project root — is not on the import path by
default. The first few lines fix that without requiring a ``pip install`` of
the surrounding project.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from mcda_common.io import load_dataset, load_preference_information  # noqa: E402
from mcda_common.types import CriterionType  # noqa: E402
from mcda_common.viz import (  # noqa: E402
    display_ranking,
    find_nodes_groups,
    test_marginal_preference_function,
    test_single_marginal_preference_function,
)

__all__ = [
    "CriterionType",
    "display_ranking",
    "find_nodes_groups",
    "load_dataset",
    "load_preference_information",
    "test_marginal_preference_function",
    "test_single_marginal_preference_function",
]
