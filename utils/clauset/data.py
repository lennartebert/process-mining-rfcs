"""Input validation and descriptive statistics for Clauset fits."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


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
