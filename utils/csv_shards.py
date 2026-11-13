"""Concatenate per-log CSV shards without upcasting bool and int columns."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def filter_shards_by_log(
    matches: Sequence[Path],
    *,
    prefix: str,
    datasets: Sequence[str] | None,
) -> list[Path]:
    """Keep shards named ``{prefix}<LOG>.csv`` when ``datasets`` is set."""
    if datasets is None:
        return list(matches)
    allowed = set(datasets)
    kept: list[Path] = []
    for path in matches:
        name = path.name
        if not (name.startswith(prefix) and name.endswith(".csv")):
            continue
        log_name = name[len(prefix) : -len(".csv")]
        if log_name in allowed:
            kept.append(path)
    return kept


def _bool_and_int_columns(frames: Sequence[pd.DataFrame]) -> tuple[set[str], set[str]]:
    """Columns to restore after concat with sparse error shards.

    Empty cells in a failed-fit shard make pandas upcast ``bool`` to ``1.0``
    and ``int`` to ``398.0``. Keep a column boolean if any shard inferred it
    as bool, and integral if any shard inferred it as int and no shard holds
    a fractional value.
    """
    bool_cols: set[str] = set()
    int_cols: set[str] = set()
    fractional_cols: set[str] = set()
    for frame in frames:
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_bool_dtype(series):
                bool_cols.add(str(column))
                continue
            if pd.api.types.is_integer_dtype(series):
                int_cols.add(str(column))
                continue
            if not pd.api.types.is_float_dtype(series):
                continue
            values = series.dropna()
            if values.empty:
                continue
            if not (values == values.round()).all():
                fractional_cols.add(str(column))
    return bool_cols, int_cols - fractional_cols - bool_cols


def _restore_bool_column(series: pd.Series) -> pd.Series:
    def convert(value: object) -> object:
        if value is None or pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
            return bool(int(value))
        text = str(value).strip().lower()
        if text in {"true", "1", "1.0", "yes"}:
            return True
        if text in {"false", "0", "0.0", "no"}:
            return False
        return pd.NA

    return series.map(convert).astype("boolean")


def _restore_concat_dtypes(
    combined: pd.DataFrame, frames: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    bool_cols, int_cols = _bool_and_int_columns(frames)
    restored = combined.copy()
    for column in bool_cols:
        if column in restored.columns:
            restored[column] = _restore_bool_column(restored[column])
    for column in int_cols:
        if column not in restored.columns:
            continue
        numeric = pd.to_numeric(restored[column], errors="coerce")
        restored[column] = numeric.round().astype("Int64")
    return restored


def concat_sorted_csvs(
    paths: Sequence[Path],
    *,
    sort_cols: Sequence[str],
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"Warning: skipping empty shard {path}")
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return None
    combined = _restore_concat_dtypes(pd.concat(frames, ignore_index=True), frames)
    present = [column for column in sort_cols if column in combined.columns]
    if present:
        combined = combined.sort_values(list(present)).reset_index(drop=True)
    return combined
