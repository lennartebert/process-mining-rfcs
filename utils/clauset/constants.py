"""Model ids, defaults, and GOF metadata for Clauset-style power-law fits."""

from __future__ import annotations

from typing import Any

# Distribution folder / label names used in outputs.
FULL_RANGE_POWER_LAW = "full_range_power_law"
LOWER_BOUNDED_POWER_LAW = "lower_bounded_power_law"
DOUBLY_BOUNDED_POWER_LAW = "doubly_bounded_power_law"

DISTRIBUTION_NAMES: tuple[str, ...] = (
    FULL_RANGE_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    DOUBLY_BOUNDED_POWER_LAW,
)

# xmin=None means estimate xmin; search_xmax selects among head-exclusion xmax values.
DISTRIBUTION_SPECS: dict[str, dict[str, Any]] = {
    FULL_RANGE_POWER_LAW: {"xmin": 1, "search_xmax": False},
    LOWER_BOUNDED_POWER_LAW: {"xmin": None, "search_xmax": False},
    DOUBLY_BOUNDED_POWER_LAW: {"xmin": None, "search_xmax": True},
}

# (distribution_name, nested) for powerlaw.Fit.distribution_compare.
EXTERNAL_ALTERNATIVES: list[tuple[str, bool]] = [
    ("lognormal", False),
    ("exponential", False),
    ("stretched_exponential", False),
    ("truncated_power_law", True),
]

# Number of highest-valued observations to exclude when setting inclusive xmax.
EXCLUDED_HEAD_VARIANTS: tuple[int, ...] = (0, 1, 2, 3, 5, 10)

DEFAULT_SIGNIFICANCE_LEVEL = 0.10
DEFAULT_MINIMUM_FITTED_TYPES = 30

GOF_KIND_REFIT_ALL_PARAMETERS = "refit_all_parameters"
GOF_KIND_SKIPPED_INVALID_FIT = "skipped_invalid_fit"

_ALPHA_UPPER_BOUND = 4.0
_ALPHA_BOUNDARY_TOL = 1e-6
# Package default is [0, 3]; raise the upper bound so discrete full-range MLE
# is not pinned at alpha=3 (which makes pdf()/ccdf() return ~1e-307 sentinels).
ALPHA_PARAMETER_RANGE: list[float] = [0.0, 4.0]
