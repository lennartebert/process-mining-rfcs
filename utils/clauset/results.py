"""CSV row builders and classification helpers for Clauset analysis outputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_MINIMUM_FITTED_TYPES,
    DEFAULT_SIGNIFICANCE_LEVEL,
    EXTERNAL_ALTERNATIVES,
    GOF_KIND_REFIT_ALL_PARAMETERS,
    GOF_KIND_SKIPPED_INVALID_FIT,
)


def _to_cutoff_int(value: Any) -> int | float:
    """Cast a finite xmin/xmax cutoff to int; otherwise return NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number):
        return float("nan")
    return int(number)


def gof_row_from_evaluation(
    *,
    log_name: str,
    evaluation: Mapping[str, Any],
    selected: bool | None = None,
) -> dict[str, Any]:
    """Build one gof.csv row from an evaluation result.

    Invalid fits still emit a row with ``gof_kind=skipped_invalid_fit``.
    """
    fit_valid = bool(evaluation.get("fit_valid", True))
    gof_kind = (
        GOF_KIND_SKIPPED_INVALID_FIT
        if not fit_valid
        else evaluation.get("gof_kind", GOF_KIND_REFIT_ALL_PARAMETERS)
    )
    row: dict[str, Any] = {
        "log_name": log_name,
        "xmin": _to_cutoff_int(evaluation.get("xmin")),
        "xmax": _to_cutoff_int(evaluation.get("xmax")),
        "alpha": evaluation.get("alpha"),
        "KS_D": evaluation.get("KS_D"),
        "n_fitted_types": evaluation.get("n_fitted_types"),
        "fitted_type_share": evaluation.get("fitted_type_share"),
        "fitted_occurrence_share": evaluation.get("fitted_occurrence_share"),
        "log_range": evaluation.get("log_range"),
        "gof_p": evaluation.get("gof_p"),
        "n_success": evaluation.get("n_success"),
        "n_failed": evaluation.get("n_failed"),
        "alpha_at_boundary": evaluation.get("alpha_at_boundary"),
        "fit_valid": fit_valid,
        "gof_kind": gof_kind,
    }
    if "excluded_head_variants" in evaluation:
        row["excluded_head_variants"] = evaluation["excluded_head_variants"]
    if "actual_excluded_variants" in evaluation:
        row["actual_excluded_variants"] = evaluation["actual_excluded_variants"]
    if selected is not None:
        row["selected"] = bool(selected)
    return row


def _valid_comparison_frame(comparison_df: pd.DataFrame | None) -> pd.DataFrame:
    """Return comparison rows with finite R and p."""
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame(
            columns=[
                "log_name",
                "n_fitted_types",
                "model_1",
                "model_2",
                "R",
                "p",
                "preferred_model",
                "interpretation",
            ]
        )
    r = pd.to_numeric(comparison_df["R"], errors="coerce")
    p = pd.to_numeric(comparison_df["p"], errors="coerce")
    valid = r.notna() & p.notna() & np.isfinite(r) & np.isfinite(p)
    return comparison_df.loc[valid].copy()


def best_other_distribution(
    comparison_df: pd.DataFrame | None,
) -> str | None:
    """Return the strongest competing ``model_2`` (most negative R)."""
    base = _valid_comparison_frame(comparison_df)
    if base.empty:
        return None
    best_idx = base["R"].astype(float).idxmin()
    return str(base.loc[best_idx, "model_2"])


def _comparison_preference_label(
    comparison_df: pd.DataFrame | None,
    *,
    significance_level: float,
) -> str:
    """Map pairwise comparisons to preferred / alternatives preferred / inconclusive."""
    base = _valid_comparison_frame(comparison_df)
    if base.empty:
        return "inconclusive"
    alternative_significantly_preferred = bool(
        ((base["R"] < 0) & (base["p"] < significance_level)).any()
    )
    focal_significantly_preferred = bool(
        ((base["R"] > 0) & (base["p"] < significance_level)).any()
    )
    if alternative_significantly_preferred:
        return "alternatives preferred"
    if focal_significantly_preferred:
        return "preferred over alternatives"
    return "inconclusive"


def classify_power_law_result(
    n_fitted_types: int,
    gof_p: float,
    comparison_df: pd.DataFrame,
    *,
    model_label: str,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    fit_valid: bool = True,
    pathology_flags: Sequence[str] | None = None,
) -> str:
    """Return a three-step classification string for one distribution.

    Step (1) reports whether the fitted region has enough types.
    Reject when ``gof_p < significance_level``; do not reject when
    ``gof_p >= significance_level``. Pathological fits (e.g. alpha at the
    package bound) suppress alpha and all downstream findings.
    """
    if not fit_valid:
        flags = ", ".join(pathology_flags or []) or "invalid fit diagnostics"
        return f"{model_label} invalid ({flags}); fit findings suppressed"

    if n_fitted_types >= minimum_fitted_types:
        fitted_step = f"(1) >= {minimum_fitted_types} fitted types"
    else:
        fitted_step = f"(1) < {minimum_fitted_types} fitted types"

    preference = _comparison_preference_label(
        comparison_df, significance_level=significance_level
    )
    if not np.isfinite(gof_p):
        return f"{fitted_step}; (2) {model_label} GOF unavailable; (3) {preference}"
    if gof_p < significance_level:
        gof_step = f"(2) {model_label} not plausible (p<{significance_level:g})"
    else:
        gof_step = f"(2) {model_label} plausible (p>={significance_level:g})"
    return f"{fitted_step}; {gof_step}; (3) {preference}"


def build_summary_row(
    *,
    log_name: str,
    input_path: str,
    descriptive_stats: Mapping[str, Any] | pd.Series | pd.DataFrame,
    evaluation: Mapping[str, Any],
    classification: str,
    best_other_distribution: str | None = None,
) -> dict[str, Any]:
    """Assemble one per-distribution summary row."""
    if isinstance(descriptive_stats, pd.DataFrame):
        stats = descriptive_stats.iloc[0].to_dict()
    elif isinstance(descriptive_stats, pd.Series):
        stats = descriptive_stats.to_dict()
    else:
        stats = dict(descriptive_stats)

    def _optional_float(value: Any) -> float | pd.NA:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return pd.NA
        return value if np.isfinite(value) else pd.NA

    def _optional_int(value: Any) -> int | pd.NA:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return pd.NA
        try:
            if pd.isna(value):
                return pd.NA
        except (TypeError, ValueError):
            pass
        return int(value)

    row: dict[str, Any] = {
        "log_name": log_name,
        "input_path": str(input_path),
        "n_occurrences": int(stats["n_occurrences"]),
        "n_types": int(stats["n_types"]),
        "n_singletons": int(stats["n_singletons"]),
        "singleton_share": float(stats["singleton_share"]),
        "xmin": _optional_float(evaluation.get("xmin")),
        "xmax": _optional_float(evaluation.get("xmax")),
        "n_fitted_types": _optional_int(evaluation.get("n_fitted_types")),
        "fitted_type_share": _optional_float(evaluation.get("fitted_type_share")),
        "fitted_occurrence_share": _optional_float(
            evaluation.get("fitted_occurrence_share")
        ),
        "log_range": _optional_float(evaluation.get("log_range")),
        "fit_valid": evaluation.get("fit_valid", pd.NA),
        "classification": classification,
        "best_other_distribution": pd.NA,
    }
    fit_valid = bool(evaluation.get("fit_valid", True))
    if fit_valid:
        row["alpha"] = _optional_float(evaluation.get("alpha"))
        row["KS_D"] = _optional_float(evaluation.get("KS_D"))
        row["gof_p"] = _optional_float(evaluation.get("gof_p"))
        row["n_success"] = _optional_int(evaluation.get("n_success"))
        row["n_failed"] = _optional_int(evaluation.get("n_failed"))
        row["gof_kind"] = evaluation.get("gof_kind", pd.NA)
        row["best_other_distribution"] = (
            pd.NA if best_other_distribution is None else best_other_distribution
        )
    else:
        row["alpha"] = pd.NA
        row["KS_D"] = pd.NA
        row["gof_p"] = pd.NA
        row["n_success"] = pd.NA
        row["n_failed"] = pd.NA
        row["gof_kind"] = GOF_KIND_SKIPPED_INVALID_FIT
    if "excluded_head_variants" in evaluation:
        row["excluded_head_variants"] = _optional_int(
            evaluation.get("excluded_head_variants")
        )
    if "actual_excluded_variants" in evaluation:
        row["actual_excluded_variants"] = _optional_int(
            evaluation.get("actual_excluded_variants")
        )
    return row


def empty_summary_row(
    *,
    log_name: str,
    input_path: str,
    classification: str,
    include_exclusion_fields: bool = False,
) -> dict[str, Any]:
    """Return a summary row with unavailable metrics left empty."""
    row = {
        "log_name": log_name,
        "input_path": str(input_path),
        "n_occurrences": pd.NA,
        "n_types": pd.NA,
        "n_singletons": pd.NA,
        "singleton_share": pd.NA,
        "alpha": pd.NA,
        "xmin": pd.NA,
        "xmax": pd.NA,
        "KS_D": pd.NA,
        "n_fitted_types": pd.NA,
        "fitted_type_share": pd.NA,
        "fitted_occurrence_share": pd.NA,
        "log_range": pd.NA,
        "gof_p": pd.NA,
        "n_success": pd.NA,
        "n_failed": pd.NA,
        "gof_kind": pd.NA,
        "fit_valid": pd.NA,
        "classification": classification,
        "best_other_distribution": pd.NA,
    }
    if include_exclusion_fields:
        row["excluded_head_variants"] = pd.NA
        row["actual_excluded_variants"] = pd.NA
    return row


def empty_comparison_frame(log_name: str, model_1: str) -> pd.DataFrame:
    """Return empty comparison rows for one focal distribution."""
    rows = [
        {
            "log_name": log_name,
            "n_fitted_types": pd.NA,
            "model_1": model_1,
            "model_2": model_2,
            "R": pd.NA,
            "p": pd.NA,
            "preferred_model": pd.NA,
            "interpretation": pd.NA,
        }
        for model_2, _nested in EXTERNAL_ALTERNATIVES
    ]
    return pd.DataFrame(rows)


def human_model_label(distribution_name: str) -> str:
    """Map folder/distribution id to a short classification label."""
    return distribution_name.replace("_", " ")
