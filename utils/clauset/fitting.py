"""MLE power-law fitting via the ``powerlaw`` package (Clauset et al.)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import powerlaw

from .constants import (
    ALPHA_PARAMETER_RANGE,
    EXCLUDED_HEAD_VARIANTS,
    _ALPHA_BOUNDARY_TOL,
    _ALPHA_UPPER_BOUND,
)
from .data import to_observation_array


def _powerlaw_fit(
    data: np.ndarray,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
    discrete: bool = True,
) -> powerlaw.Fit:
    """Create a ``powerlaw.Fit`` with alpha search range [0, 4]."""
    return powerlaw.Fit(
        data,
        discrete=bool(discrete),
        xmin=xmin,
        xmax=xmax,
        verbose=False,
        parameter_ranges={"alpha": list(ALPHA_PARAMETER_RANGE)},
    )


def _fitted_region_mask(
    frequencies: np.ndarray,
    xmin: float,
    xmax: float | None,
) -> np.ndarray:
    """Boolean mask for observations in [xmin, xmax] (xmax inclusive when set)."""
    mask = frequencies >= xmin
    if xmax is not None and np.isfinite(xmax):
        mask &= frequencies <= xmax
    return mask


def _log_range(xmin: float, xmax: float | None) -> float:
    """Return log10(xmax / xmin) when both are finite and positive."""
    if xmax is None or not np.isfinite(xmax) or not np.isfinite(xmin):
        return float("nan")
    if xmin <= 0 or xmax <= 0:
        return float("nan")
    return float(np.log10(xmax / xmin))


def _alpha_at_parameter_boundary(alpha: float) -> bool:
    """True when alpha sits on the configured upper search bound."""
    return bool(
        np.isfinite(alpha)
        and abs(float(alpha) - _ALPHA_UPPER_BOUND) <= _ALPHA_BOUNDARY_TOL
    )


def _pathological_fit_diagnostics(alpha: float) -> list[str]:
    """Return human-readable pathology flags for a fitted power law."""
    flags: list[str] = []
    if _alpha_at_parameter_boundary(alpha):
        flags.append("alpha_at_parameter_boundary")
    return flags


def _extract_power_law_stats(
    fit: powerlaw.Fit,
    data: np.ndarray,
    *,
    discrete: bool = True,
) -> dict[str, Any]:
    """Extract unprefixed power-law fit statistics for the fitted support."""
    model = fit.power_law
    alpha = float(getattr(model, "alpha", np.nan))
    xmin = float(getattr(model, "xmin", np.nan))
    xmax_raw = getattr(fit, "xmax", None)
    xmax = (
        float(xmax_raw) if xmax_raw is not None and np.isfinite(float(xmax_raw)) else None
    )
    ks_d = float(getattr(model, "D", np.nan))
    log_range = _log_range(xmin, xmax if xmax is not None else float(np.max(data)))

    if discrete:
        n_occurrences = int(np.asarray(data).sum())
        n_types = int(np.asarray(data).size)
    else:
        n_occurrences = int(np.asarray(data).size)
        n_types = n_occurrences
    n_fitted_types = np.nan
    fitted_type_share = np.nan
    fitted_occurrence_share = np.nan
    if np.isfinite(xmin):
        fitted_mask = _fitted_region_mask(data, xmin, xmax)
        n_fitted_types = int(np.sum(fitted_mask))
        if discrete:
            n_fitted_occurrences = int(np.asarray(data)[fitted_mask].sum())
        else:
            n_fitted_occurrences = n_fitted_types
        fitted_type_share = n_fitted_types / n_types if n_types else np.nan
        fitted_occurrence_share = (
            n_fitted_occurrences / n_occurrences if n_occurrences else np.nan
        )

    pathology_flags = _pathological_fit_diagnostics(alpha)
    return {
        "alpha": alpha,
        "xmin": xmin,
        "xmax": xmax if xmax is not None else np.nan,
        "KS_D": ks_d,
        "n_fitted_types": n_fitted_types,
        "fitted_type_share": fitted_type_share,
        "fitted_occurrence_share": fitted_occurrence_share,
        "log_range": log_range,
        "alpha_at_boundary": _alpha_at_parameter_boundary(alpha),
        "pathology_flags": pathology_flags,
        "fit_valid": len(pathology_flags) == 0,
    }


def fit_power_law(
    data: np.ndarray,
    xmin: float | None = None,
    xmax: float | None = None,
    *,
    discrete: bool = True,
) -> dict[str, Any]:
    """Fit a power law with optional fixed xmin/xmax (Clauset MLE via ``powerlaw``)."""
    data = to_observation_array(data, discrete=discrete)
    fit = _powerlaw_fit(data, xmin=xmin, xmax=xmax, discrete=discrete)
    stats = _extract_power_law_stats(fit, data, discrete=discrete)
    return {"fit": fit, **stats}


def xmax_candidates_from_excluded_head(
    data: np.ndarray,
    excluded_head_variants: Sequence[int] = EXCLUDED_HEAD_VARIANTS,
    *,
    discrete: bool = True,
) -> list[tuple[int, float, int]]:
    """Return (excluded_head_variants, xmax, actual_excluded) candidates.

    ``excluded_head_variants = k`` sets ``xmax = ranked_values[k]``
    (descending ranks). Because xmax is inclusive, observations with
    value == xmax remain in the fit; ``actual_excluded_variants`` counts
    values strictly greater than xmax (ties may exclude more than k).

    Identical xmax values from ties keep the smallest requested k only.
    """
    data = to_observation_array(data, discrete=discrete)
    ranked = np.sort(data)[::-1]
    candidates: list[tuple[int, float, int]] = []
    seen_xmax: set[float] = set()
    for k in excluded_head_variants:
        if k < 0 or k >= ranked.size:
            continue
        xmax = float(ranked[k])
        if xmax in seen_xmax:
            continue
        seen_xmax.add(xmax)
        actual_excluded = int(np.sum(data > xmax))
        candidates.append((int(k), xmax, actual_excluded))
    return candidates


def select_xmax_by_min_ks(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Select xmax like ``powerlaw.Fit.find_xmin``, but for an upper cutoff.

    Among discrete head-exclusion candidates with a finite KS distance, choose
    the one with the **smallest** ``KS_D``. Ties prefer fewer excluded head
    variants, then more fitted types.
    """
    usable: list[Mapping[str, Any]] = []
    for row in candidate_rows:
        ks_d = row.get("KS_D", float("nan"))
        try:
            ks_d_f = float(ks_d)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(ks_d_f):
            continue
        usable.append(row)
    if not usable:
        return None

    def sort_key(row: Mapping[str, Any]) -> tuple[float, int, int]:
        ks_d = float(row["KS_D"])
        excl_k = int(row.get("excluded_head_variants", 10**9) or 10**9)
        n_fitted = int(row.get("n_fitted_types", 0) or 0)
        return (ks_d, excl_k, -n_fitted)

    return min(usable, key=sort_key)


def _fit_doubly_bounded_candidate(
    data: np.ndarray,
    *,
    excluded_head_variants: int,
    xmax: float,
    actual_excluded_variants: int,
    discrete: bool = True,
) -> dict[str, Any]:
    """Fit one doubly bounded candidate (no GOF)."""
    fit_result = fit_power_law(data, xmin=None, xmax=xmax, discrete=discrete)
    log_range = _log_range(float(fit_result["xmin"]), float(xmax))
    return {
        **fit_result,
        "log_range": log_range,
        "excluded_head_variants": int(excluded_head_variants),
        "actual_excluded_variants": int(actual_excluded_variants),
        "xmax": float(xmax),
    }


def fit_doubly_bounded_xmax_candidates(
    data: np.ndarray,
    *,
    excluded_head_variants: Sequence[int] = EXCLUDED_HEAD_VARIANTS,
    discrete: bool = True,
) -> list[dict[str, Any]]:
    """Fit all xmax candidates (KS only; no bootstrap GOF)."""
    data = to_observation_array(data, discrete=discrete)
    rows: list[dict[str, Any]] = []
    for excl_k, xmax, actual_excl in xmax_candidates_from_excluded_head(
        data, excluded_head_variants=excluded_head_variants, discrete=discrete
    ):
        try:
            rows.append(
                _fit_doubly_bounded_candidate(
                    data,
                    excluded_head_variants=excl_k,
                    xmax=xmax,
                    actual_excluded_variants=actual_excl,
                    discrete=discrete,
                )
            )
        except Exception:
            # Some (xmin, xmax) pairs leave an empty support; skip that candidate.
            continue
    return rows
