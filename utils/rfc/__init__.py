"""Reusable rank-frequency-curve utilities."""

from .fit_models import (
    DEFAULT_BOUNDED_MIN_FREQUENCY,
    compute_fit_statistics,
    compute_powerlaw_statistics,
    fit_exponential,
    fit_linear,
    fit_power_law,
    negative_log_likelihood,
)
from .plotting import (
    create_all_plots,
    create_dataset_plots,
    create_dataset_unfitted_plots,
    create_loglog_rfc_plot,
    create_loglog_subfigure_plot,
    normalize_for_plot,
)
from .rank_frequency_extraction import (
    extract_frequency_counts,
    extract_rank_frequencies,
    extract_rank_frequencies_from_counts,
    extract_rank_frequencies_from_event_log,
)
from .statistical_tests import (
    build_summary_row,
    classify_power_law_result,
    clauset_gof_bootstrap,
    compare_power_law_alternatives,
    descriptive_frequency_stats,
    fit_discrete_power_law,
    interpret_comparison,
    preferred_base_model,
    preferred_model_label,
    to_frequency_array,
    validate_frequency_series,
)
from .types import FitStatistics

__all__ = [
    "DEFAULT_BOUNDED_MIN_FREQUENCY",
    "FitStatistics",
    "build_summary_row",
    "classify_power_law_result",
    "clauset_gof_bootstrap",
    "compare_power_law_alternatives",
    "compute_fit_statistics",
    "compute_powerlaw_statistics",
    "create_all_plots",
    "create_dataset_plots",
    "create_dataset_unfitted_plots",
    "create_loglog_rfc_plot",
    "create_loglog_subfigure_plot",
    "descriptive_frequency_stats",
    "extract_frequency_counts",
    "extract_rank_frequencies",
    "extract_rank_frequencies_from_counts",
    "extract_rank_frequencies_from_event_log",
    "fit_discrete_power_law",
    "fit_exponential",
    "fit_linear",
    "fit_power_law",
    "interpret_comparison",
    "negative_log_likelihood",
    "normalize_for_plot",
    "preferred_base_model",
    "preferred_model_label",
    "to_frequency_array",
    "validate_frequency_series",
]
