"""Curve fitting and fit-quality metrics for RFC series."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .types import FitStatistics

DEFAULT_BOUNDED_MIN_FREQUENCY = 2.0


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = a*x + b and return (a, b)."""
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def fit_exponential(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit y = a*exp(b*x) and return (a, b)."""
    mask = (y > 0) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    x_fit = x[mask]
    y_fit = y[mask]
    b, log_a = np.polyfit(x_fit, np.log(y_fit), 1)
    return float(np.exp(log_a)), float(b)


def fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
) -> Tuple[float, float]:
    """Fit y = C*x^(-alpha) and return (C, alpha)."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if min_frequency is not None:
        mask &= y >= float(min_frequency)
    if max_frequency is not None:
        mask &= y <= float(max_frequency)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    x_fit = x[mask]
    y_fit = y[mask]
    slope, log_c = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
    return float(np.exp(log_c)), float(-slope)


def negative_log_likelihood(observed_freq: np.ndarray, predicted_values: np.ndarray) -> float:
    """Compute negative log-likelihood for observed frequencies."""
    eps = 1e-12
    observed = np.asarray(observed_freq, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    if observed.size == 0 or predicted.size != observed.size:
        return np.inf
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        return np.inf

    predicted = np.maximum(predicted, eps)
    pred_sum = predicted.sum()
    if pred_sum <= 0:
        return np.inf
    predicted_prob = predicted / pred_sum
    return float(-np.sum(observed * np.log(predicted_prob + eps)))


def _fit_and_score_power_law(
    ranks: np.ndarray,
    frequencies: np.ndarray,
    min_fit_frequency: float | None = None,
) -> Tuple[float, float, float]:
    """Fit on an optional frequency subset; score NLL on all points."""
    c, alpha = fit_power_law(ranks, frequencies, min_frequency=min_fit_frequency)
    pred = (
        c * (ranks ** (-alpha))
        if np.isfinite(c) and np.isfinite(alpha)
        else np.full_like(ranks, np.nan, dtype=float)
    )
    return c, alpha, negative_log_likelihood(frequencies, pred)


def compute_powerlaw_statistics(
    ranks: np.ndarray,
    frequencies: np.ndarray,
    bounded_min_frequency: float = DEFAULT_BOUNDED_MIN_FREQUENCY,
) -> dict:
    """Fit unbounded and bounded power laws; bounded fit excludes low frequencies, NLL uses all."""
    power_c, power_alpha, power_nll = _fit_and_score_power_law(ranks, frequencies)
    bounded_power_c, bounded_power_alpha, bounded_power_nll = _fit_and_score_power_law(
        ranks,
        frequencies,
        min_fit_frequency=bounded_min_frequency,
    )
    return {
        "power_c": power_c,
        "power_alpha": power_alpha,
        "power_nll": power_nll,
        "bounded_power_min_frequency": bounded_min_frequency,
        "bounded_power_c": bounded_power_c,
        "bounded_power_alpha": bounded_power_alpha,
        "bounded_power_nll": bounded_power_nll,
    }


def compute_fit_statistics(ranks: np.ndarray, frequencies: np.ndarray) -> FitStatistics:
    """Compute fitted parameters and NLL values for major RFC curve models."""
    linear_a, linear_b = fit_linear(ranks, frequencies)
    exp_a, exp_b = fit_exponential(ranks, frequencies)
    power_stats = compute_powerlaw_statistics(ranks, frequencies)

    linear_pred = linear_a * ranks + linear_b
    exp_pred = (
        exp_a * np.exp(exp_b * ranks)
        if np.isfinite(exp_a) and np.isfinite(exp_b)
        else np.full_like(ranks, np.nan, dtype=float)
    )

    linear_nll = negative_log_likelihood(frequencies, linear_pred)
    exponential_nll = negative_log_likelihood(frequencies, exp_pred)
    nll_values = {
        "linear": linear_nll,
        "exponential": exponential_nll,
        "power_law": power_stats["power_nll"],
        "bounded_power_law": power_stats["bounded_power_nll"],
    }
    finite_nll = {name: value for name, value in nll_values.items() if np.isfinite(value)}
    best_fit = min(finite_nll, key=finite_nll.get) if finite_nll else "none"

    return {
        "linear_a": linear_a,
        "linear_b": linear_b,
        "linear_nll": linear_nll,
        "exponential_a": exp_a,
        "exponential_b": exp_b,
        "exponential_nll": exponential_nll,
        "best_fit": best_fit,
        **power_stats,
    }
