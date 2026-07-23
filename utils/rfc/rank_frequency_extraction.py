"""Rank/frequency extraction helpers for RFC analyses."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def extract_frequency_counts(attachments_df: pd.DataFrame) -> np.ndarray:
    """Return sorted absolute counts from attachment node_id frequencies."""
    counts = attachments_df["node_id"].value_counts(dropna=False).to_numpy()
    return np.array(sorted(counts, reverse=True))


def extract_rank_frequencies(attachments_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return rank array and absolute frequencies from attachment rows."""
    counts = extract_frequency_counts(attachments_df)
    if counts.size == 0:
        return np.array([]), np.array([])
    ranks = np.arange(1, len(counts) + 1)
    return ranks, counts.astype(float)


def extract_rank_frequencies_from_counts(
    counts: np.ndarray | list[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ranks and sorted frequencies from in-memory counts."""
    arr = np.asarray(counts, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return np.array([]), np.array([])
    sorted_counts = np.array(sorted(arr, reverse=True), dtype=float)
    ranks = np.arange(1, len(sorted_counts) + 1, dtype=float)
    return ranks, sorted_counts


def extract_rank_frequencies_from_event_log(
    event_log_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build RFC ranks/frequencies from per-case activity-sequence variants."""
    required_cols = {"case", "activity"}
    missing = required_cols.difference(event_log_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in event log: {sorted(missing)}")
    variant_counts = event_log_df.groupby("case")["activity"].apply(tuple).value_counts()
    return extract_rank_frequencies_from_counts(variant_counts.to_numpy(dtype=float))
