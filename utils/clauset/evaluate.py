"""High-level evaluate: fit + Clauset GOF for each power-law model variant."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    EXCLUDED_HEAD_VARIANTS,
    GOF_KIND_REFIT_ALL_PARAMETERS,
    GOF_KIND_SKIPPED_INVALID_FIT,
)
from .data import to_observation_array
from .fitting import (
    fit_doubly_bounded_xmax_candidates,
    fit_power_law,
    select_xmax_by_min_ks,
)
from .gof import (
    _gof_bootstrap_doubly_bounded_selection,
    _gof_bootstrap_refit_all_parameters,
)


def _evaluation_from_fit_and_gof(
    fit_result: Mapping[str, Any],
    gof: Mapping[str, Any],
    *,
    gof_kind: str,
) -> dict[str, Any]:
    """Merge a fit dict with bootstrap GOF fields (shared by all models)."""
    return {
        "fit": fit_result["fit"],
        "alpha": fit_result["alpha"],
        "xmin": fit_result["xmin"],
        "xmax": fit_result["xmax"],
        "KS_D": fit_result["KS_D"],
        "n_fitted_types": fit_result["n_fitted_types"],
        "fitted_type_share": fit_result["fitted_type_share"],
        "fitted_occurrence_share": fit_result["fitted_occurrence_share"],
        "log_range": fit_result["log_range"],
        "alpha_at_boundary": fit_result["alpha_at_boundary"],
        "pathology_flags": fit_result["pathology_flags"],
        "fit_valid": fit_result["fit_valid"],
        "gof_p": gof["gof_p"],
        "n_success": gof["n_success"],
        "n_failed": gof["n_failed"],
        "empirical_d": gof["empirical_d"],
        "simulated_distances": gof.get("simulated_distances"),
        "gof_kind": gof_kind,
        **{
            key: fit_result[key]
            for key in (
                "excluded_head_variants",
                "actual_excluded_variants",
            )
            if key in fit_result
        },
    }


def _evaluation_from_invalid_fit(fit_result: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluation for a pathological fit: keep diagnostics, suppress findings."""
    return {
        "fit": fit_result["fit"],
        "alpha": fit_result["alpha"],
        "xmin": fit_result["xmin"],
        "xmax": fit_result["xmax"],
        "KS_D": fit_result["KS_D"],
        "n_fitted_types": fit_result["n_fitted_types"],
        "fitted_type_share": fit_result["fitted_type_share"],
        "fitted_occurrence_share": fit_result["fitted_occurrence_share"],
        "log_range": fit_result["log_range"],
        "alpha_at_boundary": fit_result["alpha_at_boundary"],
        "pathology_flags": list(fit_result.get("pathology_flags") or []),
        "fit_valid": False,
        "gof_p": np.nan,
        "n_success": 0,
        "n_failed": 0,
        "empirical_d": np.nan,
        "simulated_distances": np.asarray([], dtype=float),
        "gof_kind": GOF_KIND_SKIPPED_INVALID_FIT,
        **{
            key: fit_result[key]
            for key in (
                "excluded_head_variants",
                "actual_excluded_variants",
            )
            if key in fit_result
        },
    }


def evaluate_power_law_fit(
    data: np.ndarray,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    fit_result: Mapping[str, Any] | None = None,
    discrete: bool = True,
) -> dict[str, Any]:
    """Fit a power law and run Clauset GOF with free-parameter refits.

    Shared path for full-range and lower-bounded models. Optionally reuse an
    existing ``fit_result`` from ``fit_power_law``.

    If the fit is pathological (``fit_valid=False``), bootstrap GOF is skipped
    and fit findings (alpha, KS, gof_p, …) are not treated as reportable.
    """
    data = to_observation_array(data, discrete=discrete)
    if fit_result is None:
        fit_result = fit_power_law(data, xmin=xmin, xmax=xmax, discrete=discrete)
    if not bool(fit_result.get("fit_valid", True)):
        return _evaluation_from_invalid_fit(fit_result)
    gof = _gof_bootstrap_refit_all_parameters(
        data,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        xmin=xmin,
        xmax=xmax,
        discrete=discrete,
    )
    return _evaluation_from_fit_and_gof(
        fit_result, gof, gof_kind=GOF_KIND_REFIT_ALL_PARAMETERS
    )


def evaluate_doubly_bounded_power_law(
    data: np.ndarray,
    *,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    excluded_head_variants: Sequence[int] = EXCLUDED_HEAD_VARIANTS,
    discrete: bool = True,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select best xmax by min KS, then GOF only for that selected fit.

    Returns ``(selected_evaluation_or_None, candidate_fit_rows_without_gof)``.
    Invalid selected fits skip GOF and suppress reportable fit findings.
    """
    data = to_observation_array(data, discrete=discrete)
    candidate_fits = fit_doubly_bounded_xmax_candidates(
        data, excluded_head_variants=excluded_head_variants, discrete=discrete
    )
    selected_fit = select_xmax_by_min_ks(candidate_fits)
    if selected_fit is None:
        return None, candidate_fits
    if not bool(selected_fit.get("fit_valid", True)):
        return _evaluation_from_invalid_fit(selected_fit), candidate_fits

    gof = _gof_bootstrap_doubly_bounded_selection(
        data,
        selected_evaluation=selected_fit,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        excluded_head_variants=excluded_head_variants,
        discrete=discrete,
    )
    evaluation = _evaluation_from_fit_and_gof(
        selected_fit, gof, gof_kind=GOF_KIND_REFIT_ALL_PARAMETERS
    )
    return evaluation, candidate_fits
