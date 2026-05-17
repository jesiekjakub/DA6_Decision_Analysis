"""Static AHP hierarchy: categories, the criteria within each, and gain/cost directions."""

from __future__ import annotations

from typing import Final

CATEGORIES: Final[list[str]] = ["Economic", "Social/Health", "Geography/Environment"]

CATEGORY_CRITERIA: Final[dict[str, list[str]]] = {
    "Economic":              ["Employment rate", "Long-term unemployment rate", "Personal earnings"],
    "Social/Health":         ["Life expectancy", "Life satisfaction", "Employees working very long hours"],
    "Geography/Environment": ["Air pollution", "Distance from Poznan (km)"],
}

CRITERION_CATEGORY: Final[dict[str, str]] = {
    crit: cat for cat, crit_list in CATEGORY_CRITERIA.items() for crit in crit_list
}

CRITERIA: Final[list[str]] = [
    crit for cat in CATEGORIES for crit in CATEGORY_CRITERIA[cat]
]

DIRECTIONS: Final[dict[str, int]] = {
    "Employment rate":                   +1,
    "Long-term unemployment rate":       -1,
    "Personal earnings":                 +1,
    "Life expectancy":                   +1,
    "Life satisfaction":                 +1,
    "Employees working very long hours": -1,
    "Air pollution":                     -1,
    "Distance from Poznan (km)":         -1,
}
