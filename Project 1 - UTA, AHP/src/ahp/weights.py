"""Saaty's eigenvector method for AHP priorities, plus consistency ratio.

The principal-eigenvector of a reciprocal pairwise matrix gives the priority
vector; ``λ_max`` indexes the consistency of the DM's judgments. Below
``CR < 0.10`` the matrix is conventionally considered acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Saaty's tabulated Random Index, n -> RI. The 1.49 fallback is the published
# value for n=10; it's used for larger matrices as a conservative default
# because the table doesn't extend further in any standard source.
_RANDOM_INDEX: dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
}
_RI_FALLBACK: float = 1.49
_CR_THRESHOLD: float = 0.10
_RECIPROCAL_TOL: float = 1e-9


@dataclass(frozen=True)
class AhpWeightsResult:
    """Output of :func:`ahp_weights`."""

    weights: NDArray[np.float64]
    lambda_max: float
    consistency_index: float
    random_index: float
    consistency_ratio: float
    is_consistent: bool


def _check_reciprocal(A: NDArray[np.float64]) -> None:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {A.shape}")
    n = A.shape[0]
    for i in range(n):
        if abs(A[i, i] - 1.0) > _RECIPROCAL_TOL:
            raise ValueError(f"Diagonal entry A[{i},{i}] = {A[i, i]} is not 1")
        for j in range(i + 1, n):
            if abs(A[i, j] * A[j, i] - 1.0) > _RECIPROCAL_TOL:
                raise ValueError(
                    f"Matrix is not reciprocal at ({i},{j}): "
                    f"A[i,j]*A[j,i] = {A[i, j] * A[j, i]}"
                )


def ahp_weights(A: NDArray[np.float64]) -> AhpWeightsResult:
    """Run the eigenvector method on a reciprocal pairwise matrix.

    Args:
        A: Square reciprocal matrix on Saaty's scale (1–9 and reciprocals).

    Returns:
        :class:`AhpWeightsResult` carrying the normalised priority vector, the
        Perron eigenvalue, the consistency index/ratio, and a boolean flag.
    """
    _check_reciprocal(A)
    n = A.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = int(np.argmax(eigenvalues.real))
    lambda_max = float(eigenvalues[idx].real)

    # The eigenvector is real up to floating noise and sign-ambiguous; ``abs``
    # picks the positive representative before normalisation.
    w = np.abs(eigenvectors[:, idx].real)
    w = w / w.sum()

    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = _RANDOM_INDEX.get(n, _RI_FALLBACK)
    cr = ci / ri if ri > 0 else 0.0

    return AhpWeightsResult(
        weights=w,
        lambda_max=lambda_max,
        consistency_index=ci,
        random_index=ri,
        consistency_ratio=cr,
        is_consistent=cr < _CR_THRESHOLD,
    )
