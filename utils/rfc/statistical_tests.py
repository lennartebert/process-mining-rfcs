"""Clauset-style discrete power-law statistical tests for frequency vectors."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import powerlaw

DEFAULT_ALTERNATIVES: list[tuple[str, bool]] = [
    ("lognormal", False),
    ("lognormal_positive", False),
    ("exponential", False),
    ("stretched_exponential", False),
    ("truncated_power_law", True),
]
BASE_MODEL_CANDIDATES = ("power_law", "truncated_power_law")


def _extract_model_fit_stats(
    fit: powerlaw.Fit,
    model_name: str,
    frequencies: np.ndarray,
    n_cases: int,
    n_variants: int,
) -> dict[str, Any]:
    """Extract per-model fit statistics with model-name-prefixed keys."""
    prefix = f"{model_name}_"
    model = getattr(fit, model_name)
    alpha = float(getattr(model, "alpha", np.nan))
    xmin = float(getattr(model, "xmin", np.nan))
    ks_d = float(getattr(model, "D", np.nan))

    n_fitted_variants = np.nan
    fitted_variant_share = np.nan
    fitted_case_share = np.nan
    if np.isfinite(xmin):
        fitted_mask = frequencies >= xmin
        n_fitted_variants = int(np.sum(fitted_mask))
        n_fitted_cases = int(frequencies[fitted_mask].sum())
        fitted_variant_share = n_fitted_variants / n_variants
        fitted_case_share = n_fitted_cases / n_cases

    return {
        f"{prefix}alpha": alpha,
        f"{prefix}xmin": xmin,
        f"{prefix}KS_D": ks_d,
        f"{prefix}n_fitted_variants": n_fitted_variants,
        f"{prefix}fitted_variant_share": fitted_variant_share,
        f"{prefix}fitted_case_share": fitted_case_share,
    }


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
    power_law_stats = _extract_model_fit_stats(
        fit, "power_law", frequencies, n_cases=n_cases, n_variants=n_variants
    )
    truncated_stats = _extract_model_fit_stats(
        fit, "truncated_power_law", frequencies, n_cases=n_cases, n_variants=n_variants
    )
    model_stats = {**power_law_stats, **truncated_stats}

    return {
        "fit": fit,
        **model_stats,
        "summary": pd.DataFrame([model_stats]),
    }


def interpret_comparison(
    r_value: float, p_value: float, model_1: str, model_2: str
) -> str:
    """Return a short interpretation of one powerlaw.distribution_compare result."""
    if not np.isfinite(r_value) or not np.isfinite(p_value):
        return "comparison failed"

    if p_value >= 0.10:
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


def compare_power_law_alternatives(
    fit: powerlaw.Fit,
    log_name: str,
    alternatives: list[tuple[str, bool]] | None = None,
) -> pd.DataFrame:
    """Compare power-law-family candidates against common alternatives."""
    if alternatives is None:
        alternatives = DEFAULT_ALTERNATIVES

    model_1_candidates = BASE_MODEL_CANDIDATES
    comparison_rows: list[dict[str, Any]] = []
    for model_1 in model_1_candidates:
        for model_2, nested in alternatives:
            if model_1 == model_2:
                continue

            try:
                r_value, p_value = fit.distribution_compare(
                    model_1,
                    model_2,
                    nested=nested,
                )
                r_value = float(r_value)
                p_value = float(p_value)
                preferred = preferred_model_label(r_value, model_1, model_2)
                interpretation = interpret_comparison(
                    r_value, p_value, model_1, model_2
                )
            except Exception as exc:
                r_value = float("nan")
                p_value = float("nan")
                preferred = "unavailable"
                message = str(exc).strip() or exc.__class__.__name__
                interpretation = f"error during comparison ({message})"

            comparison_rows.append(
                {
                    "log_name": log_name,
                    "model_1": model_1,
                    "model_2": model_2,
                    "R": r_value,
                    "p": p_value,
                    "preferred_model": preferred,
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(comparison_rows)


def preferred_base_model(
    comparison_df: pd.DataFrame, significance_level: float = 0.10
) -> str:
    """Infer preferred base model from direct power-law vs truncated comparison."""
    direct_mask = (
        comparison_df["model_1"].isin(BASE_MODEL_CANDIDATES)
        & comparison_df["model_2"].isin(BASE_MODEL_CANDIDATES)
        & comparison_df["p"].notna()
        & comparison_df["R"].notna()
    )
    direct_df = comparison_df.loc[direct_mask].copy()
    if direct_df.empty:
        return "power_law"

    significant_df = direct_df[direct_df["p"] < significance_level]
    if significant_df.empty:
        return "power_law"

    preferred_labels = set(significant_df["preferred_model"].dropna().astype(str))
    if "truncated_power_law" in preferred_labels and "power_law" not in preferred_labels:
        return "truncated_power_law"
    if "power_law" in preferred_labels and "truncated_power_law" not in preferred_labels:
        return "power_law"
    return "power_law"


def _failed_gof_result(
    *,
    model_name: str,
    n_bootstraps: int,
    empirical_xmin: float = float("nan"),
    empirical_alpha: float = float("nan"),
    empirical_d: float = float("nan"),
) -> dict[str, Any]:
    """Return an empty GOF result when bootstrap cannot be completed."""
    return {
        "empirical_xmin": float(empirical_xmin),
        "empirical_alpha": float(empirical_alpha),
        "empirical_d": float(empirical_d),
        "model_name": model_name,
        "gof_p": float("nan"),
        "n_success": 0,
        "n_failed": int(n_bootstraps),
        "simulated_distances": np.asarray([], dtype=float),
    }


def clauset_gof_bootstrap(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    model_name: str = "power_law",
) -> dict[str, Any]:
    """Clauset-style semiparametric bootstrap GOF for a fitted tail model.

    Returns NaN ``gof_p`` (instead of raising) when the bootstrap cannot be
    completed for the requested model.
    """
    data = np.asarray(data, dtype=int)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("data must be a non-empty one-dimensional integer array.")

    try:
        empirical_fit = powerlaw.Fit(data, discrete=True, verbose=False)
        empirical_model = getattr(empirical_fit, model_name)
        empirical_xmin = float(empirical_model.xmin)
        empirical_alpha = float(empirical_model.alpha)
        empirical_d = float(empirical_model.D)
    except Exception:
        return _failed_gof_result(model_name=model_name, n_bootstraps=n_bootstraps)

    if not (
        np.isfinite(empirical_xmin)
        and np.isfinite(empirical_alpha)
        and np.isfinite(empirical_d)
    ):
        return _failed_gof_result(
            model_name=model_name,
            n_bootstraps=n_bootstraps,
            empirical_xmin=empirical_xmin,
            empirical_alpha=empirical_alpha,
            empirical_d=empirical_d,
        )

    body = data[data < empirical_xmin]
    n_tail = int(np.sum(data >= empirical_xmin))
    n_body = int(body.size)

    if n_tail < 2:
        return _failed_gof_result(
            model_name=model_name,
            n_bootstraps=n_bootstraps,
            empirical_xmin=empirical_xmin,
            empirical_alpha=empirical_alpha,
            empirical_d=empirical_d,
        )

    rng = np.random.default_rng(random_seed)
    simulated_distances: list[float] = []
    n_failed = 0

    for _ in range(n_bootstraps):
        try:
            synthetic_tail = empirical_model.generate_random(
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
            simulated_model = getattr(synthetic_fit, model_name)
            simulated_d = float(simulated_model.D)
            if not np.isfinite(simulated_d):
                raise ValueError("non-finite synthetic KS distance")
            simulated_distances.append(simulated_d)
        except Exception:
            n_failed += 1

    simulated_distances_arr = np.asarray(simulated_distances, dtype=float)
    n_success = int(simulated_distances_arr.size)
    if n_success == 0:
        return _failed_gof_result(
            model_name=model_name,
            n_bootstraps=n_bootstraps,
            empirical_xmin=empirical_xmin,
            empirical_alpha=empirical_alpha,
            empirical_d=empirical_d,
        )

    n_at_least = int(np.sum(simulated_distances_arr >= empirical_d))
    # Add-one correction; denominator uses configured n_bootstraps, not only successes.
    p_value = (1 + n_at_least) / (n_bootstraps + 1)

    return {
        "empirical_xmin": empirical_xmin,
        "empirical_alpha": empirical_alpha,
        "empirical_d": empirical_d,
        "model_name": model_name,
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
    preferred_model: str | None = None,
) -> str:
    """Return a conservative textual classification of tail-model evidence."""
    preferred_model = preferred_model or preferred_base_model(
        comparison_df, significance_level=significance_level
    )
    model_label = (
        "truncated power law"
        if preferred_model == "truncated_power_law"
        else "power law"
    )
    base_comparisons = comparison_df[
        (comparison_df["model_1"] == preferred_model)
        & (~comparison_df["model_2"].isin(BASE_MODEL_CANDIDATES))
    ]
    alternative_significantly_preferred = bool(
        (
            (base_comparisons["R"] < 0)
            & (base_comparisons["p"] < significance_level)
            & base_comparisons["R"].notna()
            & base_comparisons["p"].notna()
        ).any()
    )
    power_law_significantly_preferred = bool(
        (
            (base_comparisons["R"] > 0)
            & (base_comparisons["p"] < significance_level)
            & base_comparisons["R"].notna()
            & base_comparisons["p"].notna()
        ).any()
    )

    if n_fitted_variants < minimum_fitted_variants:
        return "insufficient fitted observations"
    if not np.isfinite(gof_p):
        return f"{model_label} GOF unavailable"
    if gof_p < significance_level:
        return f"{model_label} rejected"
    if alternative_significantly_preferred:
        return f"{model_label} not rejected, but alternative preferred"
    if power_law_significantly_preferred:
        return f"{model_label} plausible and preferred over alternatives"
    return f"{model_label} plausible, comparisons inconclusive"


def build_summary_row(
    *,
    concept_name: str,
    log_name: str,
    input_path: str,
    descriptive_stats: Mapping[str, Any] | pd.Series | pd.DataFrame,
    fit_result: Mapping[str, Any],
    gof_p: float,
    power_law_gof_p: float,
    truncated_power_law_gof_p: float,
    power_law_gof_n_success: int,
    power_law_gof_n_failed: int,
    truncated_power_law_gof_n_success: int,
    truncated_power_law_gof_n_failed: int,
    power_law_classification: str,
    truncated_power_law_classification: str,
    truncated_power_law_preferred_over_power_law: bool | None = None,
) -> dict[str, Any]:
    """Assemble the final one-row summary used by notebook and batch CSV output."""
    if isinstance(descriptive_stats, pd.DataFrame):
        stats = descriptive_stats.iloc[0].to_dict()
    elif isinstance(descriptive_stats, pd.Series):
        stats = descriptive_stats.to_dict()
    else:
        stats = dict(descriptive_stats)

    def _optional_float(value: float) -> float | pd.NA:
        value = float(value)
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

    return {
        "concept_name": concept_name,
        "log_name": log_name,
        "input_path": str(input_path),
        "n_cases": int(stats["n_cases"]),
        "n_variants": int(stats["n_variants"]),
        "n_singletons": int(stats["n_singletons"]),
        "singleton_share": float(stats["singleton_share"]),
        "power_law_alpha": _optional_float(fit_result["power_law_alpha"]),
        "power_law_xmin": _optional_float(fit_result["power_law_xmin"]),
        "power_law_KS_D": _optional_float(fit_result["power_law_KS_D"]),
        "power_law_n_fitted_variants": _optional_int(
            fit_result["power_law_n_fitted_variants"]
        ),
        "power_law_fitted_variant_share": _optional_float(
            fit_result["power_law_fitted_variant_share"]
        ),
        "power_law_fitted_case_share": _optional_float(
            fit_result["power_law_fitted_case_share"]
        ),
        "truncated_power_law_alpha": _optional_float(
            fit_result["truncated_power_law_alpha"]
        ),
        "truncated_power_law_xmin": _optional_float(
            fit_result["truncated_power_law_xmin"]
        ),
        "truncated_power_law_KS_D": _optional_float(
            fit_result["truncated_power_law_KS_D"]
        ),
        "truncated_power_law_n_fitted_variants": _optional_int(
            fit_result["truncated_power_law_n_fitted_variants"]
        ),
        "truncated_power_law_fitted_variant_share": _optional_float(
            fit_result["truncated_power_law_fitted_variant_share"]
        ),
        "truncated_power_law_fitted_case_share": _optional_float(
            fit_result["truncated_power_law_fitted_case_share"]
        ),
        "gof_p": _optional_float(gof_p),
        "power_law_gof_p": _optional_float(power_law_gof_p),
        "truncated_power_law_gof_p": _optional_float(truncated_power_law_gof_p),
        "power_law_gof_n_success": int(power_law_gof_n_success),
        "power_law_gof_n_failed": int(power_law_gof_n_failed),
        "truncated_power_law_gof_n_success": int(truncated_power_law_gof_n_success),
        "truncated_power_law_gof_n_failed": int(truncated_power_law_gof_n_failed),
        "truncated_power_law_preferred_over_power_law": (
            pd.NA
            if truncated_power_law_preferred_over_power_law is None
            else bool(truncated_power_law_preferred_over_power_law)
        ),
        "power_law_classification": power_law_classification,
        "truncated_power_law_classification": truncated_power_law_classification,
    }
