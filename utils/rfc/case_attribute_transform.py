"""Transform case attributes using datatype + binning configuration.

Binning applies to continuous attributes for relevance / RFC only.
Continuous magnitude power-law fits use raw values via
``extract_continuous_values``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .case_attribute_inventory import MISSING_LABEL, _is_missing, _to_float

NONPOSITIVE_LABEL = "__NONPOSITIVE__"


class TransformError(ValueError):
    """Raised when an attribute cannot be transformed under its config."""


def resolve_source_series(case_df: pd.DataFrame, attribute_name: str) -> pd.Series:
    if attribute_name not in case_df.columns:
        raise TransformError(
            f"Attribute column {attribute_name!r} not found in case table"
        )
    return case_df[attribute_name]


def apply_missing_rule(
    series: pd.Series,
    missing_value_handling: str = "separate_category",
) -> pd.Series:
    if missing_value_handling == "drop_cases":
        return series[~series.map(_is_missing)].reset_index(drop=True)
    if missing_value_handling == "separate_category":
        return series.map(lambda v: MISSING_LABEL if _is_missing(v) else v)
    if missing_value_handling == "exclude_attribute":
        raise TransformError("missing_value_handling is exclude_attribute")
    raise TransformError(f"Unknown missing_value_handling: {missing_value_handling}")


def _numeric_series(series: pd.Series) -> pd.Series:
    return series.map(lambda v: np.nan if _is_missing(v) else _to_float(v))


def log_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return ``n_bins + 1`` log-spaced edges over positive finite values.

    Mirrors ``powerlaw.statistics.pdf`` log spacing in spirit, but keeps float
    edges (no integer flooring) and uses a fixed bin count for relevance.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size < 1:
        raise TransformError("log_bins requires at least one positive finite value")
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmin <= 0:
        raise TransformError("log_bins requires positive values")
    if xmin == xmax:
        # Degenerate range: tiny symmetric padding in log space.
        pad = max(xmin * 1e-6, 1e-12)
        xmin = max(xmin - pad, np.nextafter(0.0, 1.0))
        xmax = xmax + pad
    edges = np.logspace(np.log10(xmin), np.log10(xmax), num=int(n_bins) + 1)
    edges = np.unique(edges)
    if edges.size < 2:
        raise TransformError("log_bins produced fewer than 2 unique edges")
    # Ensure the observed min/max are covered under pd.cut right-closed bins.
    edges[0] = min(edges[0], xmin)
    edges[-1] = max(edges[-1], xmax)
    return edges


def linear_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return ``n_bins + 1`` linearly spaced edges over finite values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 1:
        raise TransformError("linear_bins requires at least one finite value")
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmin == xmax:
        pad = max(abs(xmin) * 1e-6, 1e-12)
        xmin -= pad
        xmax += pad
    edges = np.linspace(xmin, xmax, num=int(n_bins) + 1)
    edges = np.unique(edges)
    if edges.size < 2:
        raise TransformError("linear_bins produced fewer than 2 unique edges")
    return edges


def _label_binned(
    numeric: pd.Series,
    edges: np.ndarray,
    *,
    nonpositive_as_category: bool,
) -> pd.Series:
    labels = pd.Series(index=numeric.index, dtype=object)
    missing_mask = numeric.isna()
    labels.loc[missing_mask] = MISSING_LABEL

    finite = numeric[~missing_mask]
    if nonpositive_as_category:
        nonpos = finite <= 0
        labels.loc[finite.index[nonpos.to_numpy()]] = NONPOSITIVE_LABEL
        positive = finite[~nonpos]
    else:
        positive = finite

    if positive.empty:
        return labels.map(lambda v: MISSING_LABEL if pd.isna(v) else str(v))

    binned = pd.cut(positive, bins=edges, include_lowest=True, duplicates="drop")
    labels.loc[positive.index] = binned.map(
        lambda v: MISSING_LABEL if pd.isna(v) else str(v)
    )
    return labels.map(str)


def _apply_quantile_bins(series: pd.Series, n_bins: int) -> pd.Series:
    numeric = _numeric_series(series)
    try:
        binned = pd.qcut(numeric, q=int(n_bins), duplicates="drop")
    except ValueError as exc:
        raise TransformError(f"quantile_bins failed: {exc}") from exc
    return binned.map(lambda v: MISSING_LABEL if pd.isna(v) else str(v))


def _apply_fixed_bins(series: pd.Series, bin_edges: list[float]) -> pd.Series:
    numeric = _numeric_series(series)
    binned = pd.cut(numeric, bins=bin_edges, include_lowest=True)
    return binned.map(lambda v: MISSING_LABEL if pd.isna(v) else str(v))


def _apply_edge_bins(
    series: pd.Series,
    *,
    method: str,
    n_bins: int,
) -> pd.Series:
    numeric = _numeric_series(series)
    values = numeric.to_numpy(dtype=float)
    if method == "log_bins":
        edges = log_bin_edges(values, int(n_bins))
        return _label_binned(numeric, edges, nonpositive_as_category=True)
    if method == "linear_bins":
        edges = linear_bin_edges(values, int(n_bins))
        return _label_binned(numeric, edges, nonpositive_as_category=False)
    raise TransformError(f"Unsupported edge binning method: {method}")


def transform_by_datatype(
    case_df: pd.DataFrame,
    attribute_name: str,
    *,
    datatype: str,
    binning: Mapping[str, Any] | None = None,
    missing_value_handling: str = "separate_category",
) -> pd.Series:
    """Return transformed case-level discrete labels under datatype + binning."""
    binning = dict(binning or {})
    series = resolve_source_series(case_df, attribute_name)

    if datatype in {"unsupported", "datetime", "identifier-like"}:
        raise TransformError(
            f"datatype {datatype!r} cannot be transformed for frequency analysis"
        )

    if datatype in {"categorical", "boolean"}:
        return apply_missing_rule(series, missing_value_handling)

    if datatype == "continuous":
        method = binning.get("method")
        if method in {"log_bins", "linear_bins"}:
            n_bins = binning.get("n_bins")
            if n_bins is None:
                raise TransformError(f"{method} requires n_bins")
            if missing_value_handling == "drop_cases":
                series = series[~series.map(_is_missing)].reset_index(drop=True)
                return _apply_edge_bins(series, method=method, n_bins=int(n_bins))
            transformed = _apply_edge_bins(series, method=method, n_bins=int(n_bins))
            return apply_missing_rule(transformed, missing_value_handling)
        if method == "quantile_bins":
            n_bins = binning.get("n_bins")
            if n_bins is None:
                raise TransformError("quantile_bins requires n_bins")
            if missing_value_handling == "drop_cases":
                series = series[~series.map(_is_missing)].reset_index(drop=True)
                return _apply_quantile_bins(series, int(n_bins))
            transformed = _apply_quantile_bins(series, int(n_bins))
            return apply_missing_rule(transformed, missing_value_handling)
        if method == "fixed_bins":
            edges = binning.get("bin_edges")
            if edges is None:
                raise TransformError("fixed_bins requires bin_edges")
            if missing_value_handling == "drop_cases":
                series = series[~series.map(_is_missing)].reset_index(drop=True)
                return _apply_fixed_bins(series, list(edges))
            transformed = _apply_fixed_bins(series, list(edges))
            return apply_missing_rule(transformed, missing_value_handling)
        raise TransformError(
            "continuous datatype requires binning.method "
            "log_bins, linear_bins, quantile_bins, or fixed_bins"
        )

    raise TransformError(f"Unsupported datatype: {datatype}")


def extract_continuous_values(
    case_df: pd.DataFrame,
    attribute_name: str,
    *,
    drop_nonpositive: bool = True,
) -> np.ndarray:
    """Return finite numeric raw values for continuous magnitude power-law fits."""
    series = resolve_source_series(case_df, attribute_name)
    numeric = _numeric_series(series).to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if drop_nonpositive:
        numeric = numeric[numeric > 0]
    return np.asarray(numeric, dtype=float)


def value_frequency_counts(values: pd.Series | list[Any]) -> np.ndarray:
    """Return sorted positive integer counts of distinct transformed values."""
    series = pd.Series(list(values), dtype=object)
    if series.empty:
        return np.asarray([], dtype=int)
    counts = series.map(lambda v: str(v)).value_counts().to_numpy(dtype=int)
    return np.array(sorted(counts, reverse=True), dtype=int)


def missing_category_share(values: pd.Series | list[Any]) -> float:
    series = pd.Series(list(values), dtype=object)
    if series.empty:
        return 0.0
    n_missing = int((series.map(str) == MISSING_LABEL).sum())
    return n_missing / len(series)
