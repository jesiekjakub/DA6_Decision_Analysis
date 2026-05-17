"""Shared enumerations used by both PROMETHEE and ELECTRE."""

from __future__ import annotations

from enum import StrEnum, auto


class CriterionType(StrEnum):
    """Direction in which a criterion is to be optimised."""

    GAIN = auto()
    COST = auto()
