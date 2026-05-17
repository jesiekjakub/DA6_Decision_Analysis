"""Domain exceptions raised by the data-loading and modeling layers."""

from __future__ import annotations

from pathlib import Path


class DataLayerError(Exception):
    """Base exception for everything raised by the loaders/solvers."""


class MissingDataError(DataLayerError):
    """A file the pipeline depends on is not on disk."""

    def __init__(self, path: Path, hint: str | None = None) -> None:
        msg = f"Required file not found: {path}"
        if hint:
            msg = f"{msg} ({hint})"
        super().__init__(msg)
        self.path = path


class InconsistentDataError(DataLayerError):
    """A loaded file is missing required columns or violates expected structure."""


class SolverError(DataLayerError):
    """The LP/MILP solver returned a non-optimal status."""

    def __init__(self, status: str) -> None:
        super().__init__(f"Solver did not reach an optimal solution (status={status})")
        self.status = status
