"""Clauset-style discrete power-law statistical tests for frequency vectors."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import powerlaw

DEFAULT_ALTERNATIVES: list[tuple[str, bool]] = [
    ("lognormal", False),
    ("exponential", False),
    ("stretched_exponential", False),
    ("truncated_power_law", True),
]


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


def descriptive_frequency_stats(frequencies: np.ndarray) -> pd.DataFrame:
    """Return a one-row dataframe of basic frequency-distribution statistics."""
    frequencies = np.asarray(frequencies, dtype=int)
    n_cases = int(frequencies.sum())
    n_variants = int(frequencies.size)
    n_singletons = int(np.sum(frequencies == 1))
    n_at_most_five = int(np.sum(frequencies <= 5))
    return pd.DataFrame(
        [
            {
                "n_cases": n_cases,
                "n_variants": n_variants,
                "min_frequency": int(frequencies.min()),
                "max_frequency": int(frequencies.max()),
                "mean_frequency": float(np.mean(frequencies)),
                "median_frequency": float(np.median(frequencies)),
                "n_singletons": n_singletons,
                "singleton_share": n_singletons / n_variants,
                "n_at_most_five": n_at_most_five,
                "at_most_five_share": n_at_most_five / n_variants,
            }
        ]
    )


def fit_discrete_power_law(frequencies: np.ndarray) -> dict[str, Any]:
    """Fit a discrete power law and return the fit object plus fitted-region metrics."""
    frequencies = np.asarray(frequencies, dtype=int)
    n_cases = int(frequencies.sum())
    n_variants = int(frequencies.size)

    fit = powerlaw.Fit(frequencies, discrete=True, verbose=False)
    alpha = float(fit.power_law.alpha)
    xmin = float(fit.power_law.xmin)
    ks_d = float(fit.power_law.D)

    fitted_mask = frequencies >= xmin
    n_fitted_variants = int(np.sum(fitted_mask))
    n_fitted_cases = int(frequencies[fitted_mask].sum())

    return {
        "fit": fit,
        "alpha": alpha,
        "xmin": xmin,
        "KS_D": ks_d,
        "n_fitted_variants": n_fitted_variants,
        "fitted_variant_share": n_fitted_variants / n_variants,
        "n_fitted_cases": n_fitted_cases,
        "fitted_case_share": n_fitted_cases / n_cases,
        "summary": pd.DataFrame(
            [
                {
                    "alpha": alpha,
                    "xmin": xmin,
                    "KS_D": ks_d,
                    "n_fitted_variants": n_fitted_variants,
                    "fitted_variant_share": n_fitted_variants / n_variants,
                    "n_fitted_cases": n_fitted_cases,
                    "fitted_case_share": n_fitted_cases / n_cases,
                }
            ]
        ),
    }


def interpret_comparison(r_value: float, p_value: float, alternative: str) -> str:
    """Return a short interpretation of one powerlaw.distribution_compare result."""
    if not np.isfinite(r_value) or not np.isfinite(p_value):
        return "comparison failed"

    if p_value >= 0.10:
        direction = (
            "power law numerically favored"
            if r_value > 0
            else (
                f"{alternative} numerically favored"
                if r_value < 0
                else "no numerical preference"
            )
        )
        return f"inconclusive ({direction})"

    if r_value > 0:
        return "power law favored (distinguishable)"
    if r_value < 0:
        return f"{alternative} favored (distinguishable)"
    return "no preference (distinguishable tie)"


def preferred_model_label(r_value: float, alternative: str) -> str:
    """Map the sign of R to a preferred-model label."""
    if not np.isfinite(r_value):
        return "unavailable"
    if r_value > 0:
        return "power_law"
    if r_value < 0:
        return alternative
    return "tie"


def compare_power_law_alternatives(
    fit: powerlaw.Fit,
    log_name: str,
    alternatives: list[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    """Compare a fitted power law against common alternative distributions."""
    if alternatives is None:
        alternatives = DEFAULT_ALTERNATIVES

    comparison_rows: list[dict[str, Any]] = []
    for alternative, nested in alternatives:
        try:
            r_value, p_value = fit.distribution_compare(
                "power_law",
                alternative,
                nested=nested,
            )
            r_value = float(r_value)
            p_value = float(p_value)
        except Exception:
            r_value = float("nan")
            p_value = float("nan")

        comparison_rows.append(
            {
                "log_name": log_name,
                "model_1": "power_law",
                "model_2": alternative,
                "R": r_value,
                "p": p_value,
                "preferred_model": preferred_model_label(r_value, alternative),
                "interpretation": interpret_comparison(r_value, p_value, alternative),
            }
        )
    return pd.DataFrame(comparison_rows)


def clauset_gof_bootstrap(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Clauset-style semiparametric bootstrap GOF for a discrete power law."""
    data = np.asarray(data, dtype=int)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("data must be a non-empty one-dimensional integer array.")

    empirical_fit = powerlaw.Fit(data, discrete=True, verbose=False)
    empirical_xmin = float(empirical_fit.power_law.xmin)
    empirical_alpha = float(empirical_fit.power_law.alpha)
    empirical_d = float(empirical_fit.power_law.D)

    body = data[data < empirical_xmin]
    n_tail = int(np.sum(data >= empirical_xmin))
    n_body = int(body.size)

    if n_tail < 2:
        raise ValueError(
            f"Too few observations in the fitted region (n_tail={n_tail}) for bootstrap GOF."
        )

    rng = np.random.default_rng(random_seed)
    simulated_distances: list[float] = []
    n_failed = 0

    for _ in range(n_bootstraps):
        try:
            synthetic_tail = empirical_fit.power_law.generate_random(
                n_tail,
                estimate_discrete=True,
            )
            synthetic_tail = np.asarray(synthetic_tail, dtype=float)
            synthetic_tail = np.maximum(np.round(synthetic_tail), empirical_xmin).astype(int)

            if n_body > 0:
                synthetic_body = rng.choice(body, size=n_body, replace=True).astype(int)
                synthetic_data = np.concatenate([synthetic_body, synthetic_tail])
            else:
                synthetic_data = synthetic_tail

            synthetic_fit = powerlaw.Fit(synthetic_data, discrete=True, verbose=False)
            simulated_d = float(synthetic_fit.power_law.D)
            if not np.isfinite(simulated_d):
                raise ValueError("non-finite synthetic KS distance")
            simulated_distances.append(simulated_d)
        except Exception:
            n_failed += 1

    simulated_distances_arr = np.asarray(simulated_distances, dtype=float)
    n_success = int(simulated_distances_arr.size)
    if n_success == 0:
        raise RuntimeError("All bootstrap iterations failed; cannot compute GOF p-value.")

    n_at_least = int(np.sum(simulated_distances_arr >= empirical_d))
    # Add-one correction; denominator uses configured n_bootstraps, not only successes.
    p_value = (1 + n_at_least) / (n_bootstraps + 1)

    return {
        "empirical_xmin": empirical_xmin,
        "empirical_alpha": empirical_alpha,
        "empirical_d": empirical_d,
        "gof_p": float(p_value),
        "n_success": n_success,
        "n_failed": int(n_failed),
        "simulated_distances": simulated_distances_arr,
    }


def classify_power_law_result(
    n_fitted_variants: int,
    gof_p: float,
    comparison_df: pd.DataFrame,
    minimum_fitted_variants: int = 50,
    significance_level: float = 0.10,
) -> str:
    """Return a conservative textual classification of power-law evidence."""
    alternative_significantly_preferred = bool(
        (
            (comparison_df["R"] < 0)
            & (comparison_df["p"] < significance_level)
            & comparison_df["R"].notna()
            & comparison_df["p"].notna()
        ).any()
    )
    power_law_significantly_preferred = bool(
        (
            (comparison_df["R"] > 0)
            & (comparison_df["p"] < significance_level)
            & comparison_df["R"].notna()
            & comparison_df["p"].notna()
        ).any()
    )

    if n_fitted_variants < minimum_fitted_variants:
        return "insufficient fitted observations"
    if gof_p < significance_level:
        return "power law rejected"
    if alternative_significantly_preferred:
        return "power law not rejected, but alternative preferred"
    if power_law_significantly_preferred:
        return "power law plausible and preferred over alternatives"
    return "power law plausible, comparisons inconclusive"


def build_summary_row(
    *,
    concept_name: str,
    log_name: str,
    input_path: str,
    descriptive_stats: Mapping[str, Any] | pd.Series | pd.DataFrame,
    fit_result: Mapping[str, Any],
    gof_p: float,
    classification: str,
) -> dict[str, Any]:
    """Assemble the final one-row summary used by notebook and batch CSV output."""
    if isinstance(descriptive_stats, pd.DataFrame):
        stats = descriptive_stats.iloc[0].to_dict()
    elif isinstance(descriptive_stats, pd.Series):
        stats = descriptive_stats.to_dict()
    else:
        stats = dict(descriptive_stats)

    return {
        "concept_name": concept_name,
        "log_name": log_name,
        "input_path": str(input_path),
        "n_cases": int(stats["n_cases"]),
        "n_variants": int(stats["n_variants"]),
        "n_singletons": int(stats["n_singletons"]),
        "singleton_share": float(stats["singleton_share"]),
        "alpha": float(fit_result["alpha"]),
        "xmin": float(fit_result["xmin"]),
        "KS_D": float(fit_result["KS_D"]),
        "gof_p": float(gof_p),
        "n_fitted_variants": int(fit_result["n_fitted_variants"]),
        "fitted_variant_share": float(fit_result["fitted_variant_share"]),
        "fitted_case_share": float(fit_result["fitted_case_share"]),
        "classification": classification,
    }
