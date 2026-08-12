"""Clauset-style power-law statistical tests for observation or frequency data."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import powerlaw

# Distribution folder / label names used in outputs.
FULL_RANGE_POWER_LAW = "full_range_power_law"
LOWER_BOUNDED_POWER_LAW = "lower_bounded_power_law"
DOUBLY_BOUNDED_POWER_LAW = "doubly_bounded_power_law"

DISTRIBUTION_NAMES: tuple[str, ...] = (
    FULL_RANGE_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    DOUBLY_BOUNDED_POWER_LAW,
)

DISTRIBUTION_SPECS: dict[str, dict[str, Any]] = {
    FULL_RANGE_POWER_LAW: {"xmin": 1, "search_xmax": False},
    LOWER_BOUNDED_POWER_LAW: {"xmin": None, "search_xmax": False},
    DOUBLY_BOUNDED_POWER_LAW: {"xmin": None, "search_xmax": True},
}

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
DEFAULT_MINIMUM_ORDERS_OF_MAGNITUDE = 1.0

GOF_KIND_REFIT_ALL_PARAMETERS = "refit_all_parameters"
GOF_KIND_SKIPPED_INVALID_FIT = "skipped_invalid_fit"

_ALPHA_UPPER_BOUND = 4.0
_ALPHA_BOUNDARY_TOL = 1e-6
# Package default is [0, 3]; raise the upper bound so discrete full-range MLE
# is not pinned at alpha=3 (which makes pdf()/ccdf() return ~1e-307 sentinels).
ALPHA_PARAMETER_RANGE: list[float] = [0.0, 4.0]


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


def validate_frequency_series(
    frequency_df: pd.DataFrame,
    frequency_column: str,
) -> pd.Series:
    """Validate a frequency column and return it as a numeric Series."""
    if frequency_column not in frequency_df.columns:
        raise KeyError(
            f"Configured frequency_column={frequency_column!r} not found. "
            f"Available columns: {frequency_df.columns.tolist()}"
        )

    freq_series = frequency_df[frequency_column]
    if freq_series.isna().any():
        n_missing = int(freq_series.isna().sum())
        raise ValueError(
            f"Frequency column {frequency_column!r} contains {n_missing} missing value(s)."
        )

    freq_numeric = pd.to_numeric(freq_series, errors="coerce")
    if freq_numeric.isna().any():
        bad_values = freq_series[freq_numeric.isna()].unique()[:5]
        raise TypeError(
            f"Frequency column {frequency_column!r} contains non-numeric values. "
            f"Examples: {list(bad_values)}"
        )

    if not np.all(np.isfinite(freq_numeric.to_numpy(dtype=float))):
        raise ValueError(
            f"Frequency column {frequency_column!r} contains non-finite values."
        )

    freq_values = freq_numeric.to_numpy()
    if not np.allclose(freq_values, np.round(freq_values)):
        raise ValueError(
            f"Frequency column {frequency_column!r} must contain integers only."
        )

    return freq_numeric


def to_frequency_array(
    frequency_df: pd.DataFrame,
    frequency_column: str = "frequency",
) -> np.ndarray:
    """Validate a frequency column and return a 1-D positive integer array."""
    freq_numeric = validate_frequency_series(frequency_df, frequency_column)
    frequencies = np.asarray(np.round(freq_numeric.to_numpy()), dtype=int)

    if frequencies.ndim != 1:
        raise ValueError(
            f"Frequencies must be one-dimensional; got shape {frequencies.shape}."
        )
    if frequencies.size == 0:
        raise ValueError("Frequency array is empty after validation.")
    if np.any(frequencies <= 0):
        n_nonpositive = int(np.sum(frequencies <= 0))
        raise ValueError(
            "All frequencies must be positive integers; "
            f"found {n_nonpositive} non-positive value(s)."
        )
    return frequencies


def to_observation_array(
    data: np.ndarray | Sequence[Any],
    *,
    discrete: bool = True,
) -> np.ndarray:
    """Validate and return a 1-D positive observation array for power-law fits.

    Parameters
    ----------
    discrete :
        If True, require positive integers (frequency / count data).
        If False, require positive finite floats (continuous magnitudes).
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"data must be one-dimensional; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Observation array is empty after validation.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Observation array contains non-finite values.")
    if np.any(arr <= 0):
        n_nonpositive = int(np.sum(arr <= 0))
        raise ValueError(
            "All observations must be positive; "
            f"found {n_nonpositive} non-positive value(s)."
        )
    if discrete:
        if not np.allclose(arr, np.round(arr)):
            raise ValueError("Discrete observations must be integers.")
        return np.asarray(np.round(arr), dtype=int)
    return arr


def descriptive_frequency_stats(frequencies: np.ndarray) -> pd.DataFrame:
    """Return a one-row dataframe of basic frequency-distribution statistics."""
    frequencies = np.asarray(frequencies, dtype=int)
    n_occurrences = int(frequencies.sum())
    n_types = int(frequencies.size)
    n_singletons = int(np.sum(frequencies == 1))
    n_at_most_five = int(np.sum(frequencies <= 5))
    return pd.DataFrame(
        [
            {
                "n_occurrences": n_occurrences,
                "n_types": n_types,
                "min_frequency": int(frequencies.min()),
                "max_frequency": int(frequencies.max()),
                "mean_frequency": float(np.mean(frequencies)),
                "median_frequency": float(np.median(frequencies)),
                "n_singletons": n_singletons,
                "singleton_share": n_singletons / n_types,
                "n_at_most_five": n_at_most_five,
                "at_most_five_share": n_at_most_five / n_types,
            }
        ]
    )


def descriptive_observation_stats(
    data: np.ndarray,
    *,
    discrete: bool = True,
) -> pd.DataFrame:
    """Descriptive stats with schema compatible with ``build_summary_row``.

    Discrete: treat *data* as type frequencies (``n_occurrences = sum``).
    Continuous: treat *data* as raw magnitudes (``n_occurrences = n_obs``).
    """
    if discrete:
        return descriptive_frequency_stats(data)
    values = np.asarray(data, dtype=float)
    n_obs = int(values.size)
    unique, counts = np.unique(values, return_counts=True)
    n_singletons = int(np.sum(counts == 1))
    n_unique = int(unique.size)
    return pd.DataFrame(
        [
            {
                "n_occurrences": n_obs,
                "n_types": n_obs,
                "min_frequency": float(np.min(values)),
                "max_frequency": float(np.max(values)),
                "mean_frequency": float(np.mean(values)),
                "median_frequency": float(np.median(values)),
                "n_singletons": n_singletons,
                "singleton_share": (n_singletons / n_unique) if n_unique else 0.0,
                "n_at_most_five": int(np.sum(counts <= 5)),
                "at_most_five_share": (
                    float(np.sum(counts[counts <= 5]) / n_obs) if n_obs else 0.0
                ),
            }
        ]
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
    """True when alpha sits on powerlaw's default upper bound of 3."""
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


def _to_cutoff_int(value: Any) -> int | float:
    """Cast a finite xmin/xmax cutoff to int; otherwise return NaN."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(number):
        return float("nan")
    return int(number)


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
    xmax = float(xmax_raw) if xmax_raw is not None and np.isfinite(float(xmax_raw)) else None
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
    """Fit a power law with optional fixed xmin/xmax."""
    data = to_observation_array(data, discrete=discrete)
    fit = _powerlaw_fit(data, xmin=xmin, xmax=xmax, discrete=discrete)
    stats = _extract_power_law_stats(fit, data, discrete=discrete)
    return {"fit": fit, **stats}


def fit_discrete_power_law(
    frequencies: np.ndarray,
    xmin: float | None = None,
    xmax: float | None = None,
) -> dict[str, Any]:
    """Fit a discrete power law (alias for ``fit_power_law(..., discrete=True)``)."""
    return fit_power_law(frequencies, xmin=xmin, xmax=xmax, discrete=True)


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


def interpret_comparison(
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


def preferred_model_label(r_value: float, model_1: str, model_2: str) -> str:
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
    preferred = preferred_model_label(r_value, model_1, model_2)
    if interpretation is None:
        interpretation = interpret_comparison(
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


def compare_against_alternatives(
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
    return compare_against_alternatives(
        fit_1,
        log_name=log_name,
        model_1=model_1,
        alternatives=alternatives,
        significance_level=significance_level,
    )


def _failed_gof_result(
    *,
    xmin: float = float("nan"),
    xmax: float = float("nan"),
    alpha: float = float("nan"),
    empirical_d: float = float("nan"),
    n_bootstraps: int,
) -> dict[str, Any]:
    """Return an empty GOF result when bootstrap cannot be completed."""
    return {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "alpha": float(alpha),
        "empirical_d": float(empirical_d),
        "gof_p": float("nan"),
        "n_success": 0,
        "n_failed": int(n_bootstraps),
        "simulated_distances": np.asarray([], dtype=float),
    }


def _sample_discrete_power_law_lower_bounded(
    n: int,
    *,
    alpha: float,
    xmin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` integers from a discrete power law on ``x >= xmin`` (no upper bound)."""
    if n <= 0:
        return np.asarray([], dtype=int)
    xmin_i = int(np.floor(xmin))
    dist = powerlaw.Power_Law(
        xmin=xmin_i,
        parameters=[float(alpha)],
        discrete=True,
    )
    out: list[int] = []
    # Approximate sampler is fast; reject any rare draws below xmin (never clip).
    while len(out) < n:
        need = n - len(out)
        batch = np.asarray(
            dist.generate_random(need, estimate_discrete=False),
            dtype=float,
        )
        batch = np.asarray(np.round(batch), dtype=int)
        batch = batch[batch >= xmin_i]
        if batch.size:
            out.extend(batch.tolist())
    return np.asarray(out[:n], dtype=int)


def _sample_discrete_power_law_truncated(
    n: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` integers from a discrete PL normalized on ``{xmin,...,xmax}``."""
    if n <= 0:
        return np.asarray([], dtype=int)
    xmin_i = int(np.floor(xmin))
    xmax_i = int(np.floor(xmax))
    if xmax_i < xmin_i:
        raise ValueError(f"xmax ({xmax_i}) < xmin ({xmin_i})")
    support = np.arange(xmin_i, xmax_i + 1, dtype=int)
    if support.size == 1:
        return np.full(n, support[0], dtype=int)

    # Direct categorical sampling on the truncated support (no clipping).
    # For extremely wide supports fall back to rejection from the lower-bounded PL.
    max_direct_support = 100_000
    if support.size <= max_direct_support:
        log_w = -float(alpha) * np.log(support.astype(float))
        log_w -= float(np.max(log_w))
        weights = np.exp(log_w)
        probs = weights / weights.sum()
        return rng.choice(support, size=n, replace=True, p=probs).astype(int)

    out: list[int] = []
    while len(out) < n:
        need = n - len(out)
        # Oversample: acceptance rate is the truncated mass under the lower-bounded PL.
        batch = _sample_discrete_power_law_lower_bounded(
            max(need * 2, need),
            alpha=alpha,
            xmin=xmin_i,
            rng=rng,
        )
        batch = batch[batch <= xmax_i]
        if batch.size:
            out.extend(batch.tolist())
    return np.asarray(out[:n], dtype=int)


def _sample_continuous_power_law_lower_bounded(
    n: int,
    *,
    alpha: float,
    xmin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` values from a continuous power law on ``x >= xmin``."""
    if n <= 0:
        return np.asarray([], dtype=float)
    # Inverse CDF: x = xmin * U^(-1/(alpha-1)) for alpha > 1.
    alpha = float(alpha)
    xmin = float(xmin)
    if alpha <= 1.0:
        # Degenerate / heavy regime: fall back to powerlaw's generator.
        dist = powerlaw.Power_Law(xmin=xmin, parameters=[alpha], discrete=False)
        return np.asarray(dist.generate_random(n), dtype=float)
    u = rng.random(n)
    return xmin * np.power(u, -1.0 / (alpha - 1.0))


def _sample_continuous_power_law_truncated(
    n: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` values from a continuous PL truncated to ``[xmin, xmax]``."""
    if n <= 0:
        return np.asarray([], dtype=float)
    xmin = float(xmin)
    xmax = float(xmax)
    alpha = float(alpha)
    if xmax < xmin:
        raise ValueError(f"xmax ({xmax}) < xmin ({xmin})")
    # Inverse CDF of truncated continuous power law.
    if alpha == 1.0:
        # Limit form: uniform in log-space between xmin and xmax.
        u = rng.random(n)
        return xmin * np.exp(u * np.log(xmax / xmin))
    if alpha < 1.0:
        out: list[float] = []
        while len(out) < n:
            need = n - len(out)
            batch = _sample_continuous_power_law_lower_bounded(
                max(need * 2, need),
                alpha=alpha,
                xmin=xmin,
                rng=rng,
            )
            batch = batch[batch <= xmax]
            if batch.size:
                out.extend(batch.tolist())
        return np.asarray(out[:n], dtype=float)
    # CDF(x) = 1 - (x/xmin)^(1-alpha); truncate and invert.
    cmin = xmin ** (1.0 - alpha)
    cmax = xmax ** (1.0 - alpha)
    u = rng.random(n)
    return np.power(cmin - u * (cmin - cmax), 1.0 / (1.0 - alpha))


def _sample_fitted_power_law(
    n: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float | None,
    rng: np.random.Generator,
    discrete: bool,
) -> np.ndarray:
    """Sample from the fitted PL on its proper support (no clipping)."""
    if xmax is not None and np.isfinite(xmax):
        if discrete:
            return _sample_discrete_power_law_truncated(
                n, alpha=alpha, xmin=xmin, xmax=xmax, rng=rng
            )
        return _sample_continuous_power_law_truncated(
            n, alpha=alpha, xmin=xmin, xmax=xmax, rng=rng
        )
    if discrete:
        return _sample_discrete_power_law_lower_bounded(
            n, alpha=alpha, xmin=xmin, rng=rng
        )
    return _sample_continuous_power_law_lower_bounded(
        n, alpha=alpha, xmin=xmin, rng=rng
    )


def _resample_empirical(
    values: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    discrete: bool,
) -> np.ndarray:
    """Resample ``n`` observations from an empirical region (with replacement)."""
    if n <= 0:
        return np.asarray([], dtype=int if discrete else float)
    if values.size == 0:
        raise ValueError("cannot resample from an empty empirical region")
    drawn = rng.choice(values, size=n, replace=True)
    return drawn.astype(int if discrete else float, copy=False)


def _simulate_semiparametric_sample(
    *,
    data: np.ndarray,
    empirical_model: Any,
    empirical_xmin: float,
    empirical_xmax: float,
    rng: np.random.Generator,
    discrete: bool = True,
) -> np.ndarray:
    """Build one Clauset semiparametric synthetic sample of size ``K=len(data)``.

    The empirically fitted model (alpha, xmin, and xmax if present) is the fixed
    null. Each of the ``K`` synthetic observations is drawn independently from a
    mixture over empirical regions, so region counts vary across replicates while
    the total sample size stays fixed. Fitted-range values are sampled from the
    correctly normalized (truncated) power law — never by clipping unbounded draws.
    """
    data = np.asarray(data)
    if discrete:
        data = np.asarray(data, dtype=int)
    else:
        data = np.asarray(data, dtype=float)

    k_total = int(data.size)
    if k_total == 0:
        return data.copy()

    alpha = float(getattr(empirical_model, "alpha"))
    xmin = float(empirical_xmin)
    has_xmax = bool(np.isfinite(empirical_xmax))
    xmax = float(empirical_xmax) if has_xmax else None

    if has_xmax:
        assert xmax is not None
        below = data[data < xmin]
        mid = data[(data >= xmin) & (data <= xmax)]
        above = data[data > xmax]
        shares = np.asarray(
            [below.size, mid.size, above.size], dtype=float
        ) / float(k_total)
        n_below, n_mid, n_above = (int(x) for x in rng.multinomial(k_total, shares))
        parts: list[np.ndarray] = []
        if n_below:
            parts.append(
                _resample_empirical(below, n_below, rng=rng, discrete=discrete)
            )
        if n_mid:
            parts.append(
                _sample_fitted_power_law(
                    n_mid,
                    alpha=alpha,
                    xmin=xmin,
                    xmax=xmax,
                    rng=rng,
                    discrete=discrete,
                )
            )
        if n_above:
            parts.append(
                _resample_empirical(above, n_above, rng=rng, discrete=discrete)
            )
        if not parts:
            return data[:0].copy()
        return np.concatenate(parts)

    below = data[data < xmin]
    n_fit = int(np.sum(data >= xmin))
    p_fit = n_fit / float(k_total)
    n_tail = int(rng.binomial(k_total, p_fit))
    n_body = k_total - n_tail
    parts = []
    if n_body:
        parts.append(_resample_empirical(below, n_body, rng=rng, discrete=discrete))
    if n_tail:
        parts.append(
            _sample_fitted_power_law(
                n_tail,
                alpha=alpha,
                xmin=xmin,
                xmax=None,
                rng=rng,
                discrete=discrete,
            )
        )
    if not parts:
        return data[:0].copy()
    return np.concatenate(parts)


def clauset_gof_bootstrap_refit_all_parameters(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    xmin: float | None = None,
    xmax: float | None = None,
    *,
    discrete: bool = True,
) -> dict[str, Any]:
    """Clauset GOF that refits free cutoffs and alpha on each bootstrap sample.

    Synthetic samples use the Clauset semiparametric mixture (empirical outside
    regions + fitted PL on the fitted support) with fixed total size ``K`` and
    varying region counts. On each replicate, free parameters are re-estimated:
    ``xmin is None`` / ``xmax is None`` means that cutoff is refit; a provided
    cutoff is treated as a model constraint.
    """
    data = to_observation_array(data, discrete=discrete)

    try:
        empirical_fit = _powerlaw_fit(
            data, xmin=xmin, xmax=xmax, discrete=discrete
        )
        empirical_model = empirical_fit.power_law
        empirical_xmin = float(empirical_model.xmin)
        empirical_alpha = float(empirical_model.alpha)
        empirical_d = float(empirical_model.D)
        xmax_raw = getattr(empirical_fit, "xmax", None)
        empirical_xmax = (
            float(xmax_raw)
            if xmax_raw is not None and np.isfinite(float(xmax_raw))
            else float("nan")
        )
    except Exception:
        return _failed_gof_result(n_bootstraps=n_bootstraps)

    if not (
        np.isfinite(empirical_xmin)
        and np.isfinite(empirical_alpha)
        and np.isfinite(empirical_d)
    ):
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    in_tail = data >= empirical_xmin
    if np.isfinite(empirical_xmax):
        in_tail &= data <= empirical_xmax
    n_tail = int(np.sum(in_tail))
    if n_tail < 2:
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    # Model constraints stay fixed; free cutoffs (None) are refit each replicate.
    synth_xmin = xmin
    synth_xmax = xmax if (xmax is not None and np.isfinite(float(xmax))) else None

    rng = np.random.default_rng(random_seed)
    simulated_distances: list[float] = []
    n_failed = 0

    for _ in range(n_bootstraps):
        try:
            synthetic_data = _simulate_semiparametric_sample(
                data=data,
                empirical_model=empirical_model,
                empirical_xmin=empirical_xmin,
                empirical_xmax=empirical_xmax,
                rng=rng,
                discrete=discrete,
            )
            synthetic_fit = _powerlaw_fit(
                synthetic_data,
                xmin=synth_xmin,
                xmax=synth_xmax,
                discrete=discrete,
            )
            simulated_d = float(synthetic_fit.power_law.D)
            if not np.isfinite(simulated_d):
                raise ValueError("non-finite synthetic KS distance")
            simulated_distances.append(simulated_d)
        except Exception:
            n_failed += 1

    simulated_distances_arr = np.asarray(simulated_distances, dtype=float)
    n_success = int(simulated_distances_arr.size)
    if n_success == 0:
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    n_at_least = int(np.sum(simulated_distances_arr >= empirical_d))
    p_value = n_at_least / n_success

    return {
        "xmin": empirical_xmin,
        "xmax": empirical_xmax,
        "alpha": empirical_alpha,
        "empirical_d": empirical_d,
        "gof_p": float(p_value),
        "n_success": n_success,
        "n_failed": int(n_failed),
        "simulated_distances": simulated_distances_arr,
    }


# Public alias used by callers that only need the default GOF path.
clauset_gof_bootstrap = clauset_gof_bootstrap_refit_all_parameters


def select_xmax_by_min_ks(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Select xmax like ``powerlaw.Fit.find_xmin``, but for an upper cutoff.

    Among discrete head-exclusion candidates with a finite KS distance, choose
    the one with the **smallest** ``KS_D`` (best match of empirical vs fitted
    CDF on that candidate's support). Ties prefer fewer excluded head variants,
    then more fitted types.
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


# Backwards-compatible alias (previous coverage-first selector).
def select_broadest_plausible_interval(
    candidate_rows: Sequence[Mapping[str, Any]],
    **_kwargs: Any,
) -> Mapping[str, Any] | None:
    """Deprecated alias for ``select_xmax_by_min_ks`` (kwargs ignored)."""
    return select_xmax_by_min_ks(candidate_rows)


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
    gof = clauset_gof_bootstrap_refit_all_parameters(
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

    Mirrors lower-bounded flow: one official bootstrap GOF on the chosen model.
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

    gof = clauset_gof_bootstrap_doubly_bounded_selection(
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


def evaluate_doubly_bounded_candidates(
    data: np.ndarray,
    *,
    log_name: str,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    excluded_head_variants: Sequence[int] = EXCLUDED_HEAD_VARIANTS,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    minimum_orders_of_magnitude: float = DEFAULT_MINIMUM_ORDERS_OF_MAGNITUDE,
    discrete: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, list[dict[str, Any]]]:
    """Compatibility wrapper: GOF only for min-KS xmax; candidates are fits only.

    Returns ``(gof_rows, selected_evaluation_or_None, candidate_fits)``.
    ``gof_rows`` contains at most one row (the selected model), like lower-bounded.
    """
    del significance_level, minimum_fitted_types, minimum_orders_of_magnitude
    selected, candidate_fits = evaluate_doubly_bounded_power_law(
        data,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        excluded_head_variants=excluded_head_variants,
        discrete=discrete,
    )
    if selected is None:
        return [], None, candidate_fits
    gof_row = gof_row_from_evaluation(
        log_name=log_name,
        evaluation=selected,
        selected=True,
    )
    return [gof_row], selected, candidate_fits


def clauset_gof_bootstrap_doubly_bounded_selection(
    data: np.ndarray,
    *,
    selected_evaluation: Mapping[str, Any],
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    excluded_head_variants: Sequence[int] = EXCLUDED_HEAD_VARIANTS,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    minimum_orders_of_magnitude: float = DEFAULT_MINIMUM_ORDERS_OF_MAGNITUDE,
    discrete: bool = True,
) -> dict[str, Any]:
    """GOF that re-runs min-KS xmax selection inside each bootstrap."""
    del significance_level, minimum_fitted_types, minimum_orders_of_magnitude

    data = to_observation_array(data, discrete=discrete)
    fit = selected_evaluation["fit"]
    empirical_model = fit.power_law
    empirical_xmin = float(selected_evaluation["xmin"])
    empirical_xmax = float(selected_evaluation["xmax"])
    empirical_d = float(selected_evaluation["KS_D"])
    empirical_alpha = float(selected_evaluation["alpha"])

    if not (np.isfinite(empirical_xmin) and np.isfinite(empirical_xmax)):
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    in_tail = (data >= empirical_xmin) & (data <= empirical_xmax)
    if int(np.sum(in_tail)) < 2:
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    rng = np.random.default_rng(random_seed)
    simulated_distances: list[float] = []
    n_failed = 0

    for _ in range(n_bootstraps):
        try:
            synthetic_data = _simulate_semiparametric_sample(
                data=data,
                empirical_model=empirical_model,
                empirical_xmin=empirical_xmin,
                empirical_xmax=empirical_xmax,
                rng=rng,
                discrete=discrete,
            )
            cand_rows = fit_doubly_bounded_xmax_candidates(
                synthetic_data,
                excluded_head_variants=excluded_head_variants,
                discrete=discrete,
            )
            chosen = select_xmax_by_min_ks(cand_rows)
            if chosen is None:
                raise ValueError("no usable candidate in bootstrap replication")
            simulated_d = float(chosen["KS_D"])
            if not np.isfinite(simulated_d):
                raise ValueError("non-finite selected synthetic KS distance")
            simulated_distances.append(simulated_d)
        except Exception:
            n_failed += 1

    simulated_distances_arr = np.asarray(simulated_distances, dtype=float)
    n_success = int(simulated_distances_arr.size)
    if n_success == 0:
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    n_at_least = int(np.sum(simulated_distances_arr >= empirical_d))
    p_value = n_at_least / n_success
    return {
        "xmin": empirical_xmin,
        "xmax": empirical_xmax,
        "alpha": empirical_alpha,
        "empirical_d": empirical_d,
        "gof_p": float(p_value),
        "n_success": n_success,
        "n_failed": int(n_failed),
        "simulated_distances": simulated_distances_arr,
    }


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
