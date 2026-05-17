"""Consistency diagnostics for AHP pairwise matrices.

Reconstructs the perfectly-consistent matrix implied by a priority vector
``w`` (the outer product ``w_i / w_j``) and reports the entry that diverges
furthest from the DM's recorded value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MatrixDiscrepancy:
    """The single largest pairwise discrepancy between observed and reconstructed."""

    row_label: str
    col_label: str
    dm_value: float
    reconstructed_value: float
    abs_diff: float


def reconstruct_matrix(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """The perfectly-consistent reciprocal matrix implied by a priority vector."""
    return np.outer(weights, 1.0 / weights)


def max_discrepancy(
    A_orig: NDArray[np.float64],
    A_rec: NDArray[np.float64],
    labels: list[str],
) -> MatrixDiscrepancy:
    """Locate the most inconsistent off-diagonal judgment in ``A_orig`` vs ``A_rec``."""
    diff = np.abs(A_orig - A_rec)
    np.fill_diagonal(diff, -1.0)  # exclude the trivially-1.0 diagonal
    i, j = np.unravel_index(int(np.argmax(diff)), diff.shape)
    return MatrixDiscrepancy(
        row_label=labels[i],
        col_label=labels[j],
        dm_value=float(A_orig[i, j]),
        reconstructed_value=float(A_rec[i, j]),
        abs_diff=float(diff[i, j]),
    )
