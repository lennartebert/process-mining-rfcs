"""Clauset–Shalizi–Newman power-law fitting, GOF, and model comparison.

Public surface for analyses and CLI. Low-level samplers live in
``utils.clauset.sampling`` (underscore-prefixed; used by tests).
"""

from __future__ import annotations

from .compare import compare_distribution
from .constants import (
    DEFAULT_MINIMUM_FITTED_TYPES,
    DEFAULT_SIGNIFICANCE_LEVEL,
    DISTRIBUTION_NAMES,
    DISTRIBUTION_SPECS,
    DOUBLY_BOUNDED_POWER_LAW,
    EXCLUDED_HEAD_VARIANTS,
    EXTERNAL_ALTERNATIVES,
    FULL_RANGE_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
)
from .data import (
    descriptive_frequency_stats,
    descriptive_observation_stats,
    to_frequency_array,
    to_observation_array,
    validate_frequency_series,
)
from .evaluate import evaluate_doubly_bounded_power_law, evaluate_power_law_fit
from .fitting import fit_power_law
from .pipeline import (
    analyze_powerlaw_data,
    append_and_checkpoint,
    empty_powerlaw_results,
)
from .results import (
    best_other_distribution,
    build_summary_row,
    classify_power_law_result,
    empty_comparison_frame,
    empty_summary_row,
    gof_row_from_evaluation,
    human_model_label,
)

__all__ = [
    "DEFAULT_MINIMUM_FITTED_TYPES",
    "DEFAULT_SIGNIFICANCE_LEVEL",
    "DISTRIBUTION_NAMES",
    "DISTRIBUTION_SPECS",
    "DOUBLY_BOUNDED_POWER_LAW",
    "EXCLUDED_HEAD_VARIANTS",
    "EXTERNAL_ALTERNATIVES",
    "FULL_RANGE_POWER_LAW",
    "LOWER_BOUNDED_POWER_LAW",
    "analyze_powerlaw_data",
    "append_and_checkpoint",
    "best_other_distribution",
    "build_summary_row",
    "classify_power_law_result",
    "compare_distribution",
    "descriptive_frequency_stats",
    "descriptive_observation_stats",
    "empty_comparison_frame",
    "empty_powerlaw_results",
    "empty_summary_row",
    "evaluate_doubly_bounded_power_law",
    "evaluate_power_law_fit",
    "fit_power_law",
    "gof_row_from_evaluation",
    "human_model_label",
    "to_frequency_array",
    "to_observation_array",
    "validate_frequency_series",
]
