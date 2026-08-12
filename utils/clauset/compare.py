"""Likelihood-ratio comparisons of a fitted power law vs alternatives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import powerlaw

from .constants import DEFAULT_SIGNIFICANCE_LEVEL, EXTERNAL_ALTERNATIVES


def _interpret_comparison(
    r_value: float,
    p_value: float,
    model_1: str,
    model_2: str,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> str:
    """Return a short interpretation of one likelihood-ratio comparison."""
    if np.isfinite(r_value) and r_value == 0.0 and not np.isfinite(p_value):
        return "no preference (identical likelihoods)"
    if not np.isfinite(r_value) or not np.isfinite(p_value):
        return "comparison failed"

    # Distinguishable only when p < alpha (strict).
    if p_value >= significance_level:
        direction = (
            f"{model_1} numerically favored"
            if r_value > 0
            else (
                f"{model_2} numerically favored"
                if r_value < 0
                else "no numerical preference"
            )
        )
        return f"inconclusive ({direction})"

    if r_value > 0:
        return f"{model_1} favored (distinguishable)"
    if r_value < 0:
        return f"{model_2} favored (distinguishable)"
    return "no preference (distinguishable tie)"


def _preferred_model_label(r_value: float, model_1: str, model_2: str) -> str:
    """Map the sign of R to a preferred-model label."""
    if not np.isfinite(r_value):
        return "unavailable"
    if r_value > 0:
        return model_1
    if r_value < 0:
        return model_2
    return "tie"


def _comparison_row(
    *,
    log_name: str,
    model_1: str,
    model_2: str,
    r_value: float,
    p_value: float,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    interpretation: str | None = None,
) -> dict[str, Any]:
    """Build one comparison CSV row."""
    preferred = _preferred_model_label(r_value, model_1, model_2)
    if interpretation is None:
        interpretation = _interpret_comparison(
            r_value, p_value, model_1, model_2, significance_level=significance_level
        )
    return {
        "log_name": log_name,
        "model_1": model_1,
        "model_2": model_2,
        "R": r_value,
        "p": p_value,
        "preferred_model": preferred,
        "interpretation": interpretation,
    }


def _failed_comparison_row(
    *,
    log_name: str,
    model_1: str,
    model_2: str,
    exc: Exception,
) -> dict[str, Any]:
    message = str(exc).strip() or exc.__class__.__name__
    return _comparison_row(
        log_name=log_name,
        model_1=model_1,
        model_2=model_2,
        r_value=float("nan"),
        p_value=float("nan"),
        interpretation=f"error during comparison ({message})",
    )


def _compare_against_alternatives(
    fit: powerlaw.Fit,
    log_name: str,
    model_1: str,
    alternatives: list[tuple[str, bool]] | None = None,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> pd.DataFrame:
    """Compare a fitted power law against alternatives on the same Fit support."""
    if alternatives is None:
        alternatives = EXTERNAL_ALTERNATIVES

    rows: list[dict[str, Any]] = []
    for model_2, nested in alternatives:
        try:
            r_value, p_value = fit.distribution_compare(
                "power_law",
                model_2,
                nested=nested,
            )
            rows.append(
                _comparison_row(
                    log_name=log_name,
                    model_1=model_1,
                    model_2=model_2,
                    r_value=float(r_value),
                    p_value=float(p_value),
                    significance_level=significance_level,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch resilient
            rows.append(
                _failed_comparison_row(
                    log_name=log_name,
                    model_1=model_1,
                    model_2=model_2,
                    exc=exc,
                )
            )
    return pd.DataFrame(rows)


def compare_distribution(
    *,
    log_name: str,
    model_1: str,
    fit_1: powerlaw.Fit,
    alternatives: list[tuple[str, bool]] | None = None,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> pd.DataFrame:
    """Compare one power-law Fit against alternatives on identical support.

    Cross-range power-law variant comparisons are intentionally omitted:
    range selection and distribution-family selection stay separate.
    """
    return _compare_against_alternatives(
        fit_1,
        log_name=log_name,
        model_1=model_1,
        alternatives=alternatives,
        significance_level=significance_level,
    )
