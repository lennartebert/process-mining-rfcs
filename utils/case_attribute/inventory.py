"""Case-attribute inventory, type inference, and recommendations."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.io.case_tables import (
    is_default_excluded_attribute,
    is_event_aggregation_column,
)

MISSING_LABEL = "__MISSING__"

DATATYPES = (
    "categorical",
    "continuous",
    "boolean",
    "datetime",
    "identifier-like",
    "unsupported",
)

BINNING_METHODS = ("log_bins", "linear_bins", "quantile_bins", "fixed_bins")
# Minimum finite observations required to suggest continuous magnitude power-law tests.
DEFAULT_MIN_OBSERVATIONS_CONTINUOUS_POWERLAW = 50

IDENTIFIER_NAME_PATTERNS = (
    re.compile(r"(^|[_:])id($|[_:])", re.IGNORECASE),
    re.compile(r"uuid", re.IGNORECASE),
    re.compile(r"case", re.IGNORECASE),
    re.compile(r"concept:name", re.IGNORECASE),
    re.compile(r"\bkey\b", re.IGNORECASE),
)


@dataclass
class RecommendationThresholds:
    """Thresholds driving automatic include/datatype recommendations."""

    near_constant_largest_share: float = 0.98
    identifier_distinct_share: float = 0.95
    identifier_distinct_share_large: float = 0.90
    identifier_n_distinct_large: int = 1000
    excessive_missing_share: float = 0.50
    minimum_distinct_for_powerlaw: int = 50
    continuous_min_distinct: int = 20
    strong_relevance_ner: float = 0.05
    trace_encoding_ner: float = 0.99
    default_n_bins: int = 10
    minimum_observations_for_continuous_powerlaw: int = (
        DEFAULT_MIN_OBSERVATIONS_CONTINUOUS_POWERLAW
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttributeInventory:
    """Descriptive inventory for one attribute."""

    attribute_name: str
    source_level: str
    inferred_type: str
    n_cases: int
    n_missing: int
    missing_share: float
    n_distinct: int
    distinct_share: float
    most_frequent_value: Any
    largest_value_share: float
    min_value: Any = None
    max_value: Any = None
    median_value: Any = None
    mean_value: Any = None
    example_values: list[Any] = field(default_factory=list)
    is_constant: bool = False
    is_near_constant: bool = False
    is_identifier_like: bool = False
    suitable_for_categorical_frequency: bool = False
    special_handling_required: bool = False


@dataclass
class DatatypeRecommendation:
    """Step-1 suggestions: include for relevance, datatype, and binning."""

    include: bool
    datatype: str
    reason: str
    binning_method: str | None = None
    n_bins: int | None = None
    bin_edges: list[float] | None = None


@dataclass
class PowerlawIncludeRecommendation:
    """Step-2 suggestion: whether to run power-law testing."""

    include: bool
    reason: str
    discrete: bool = True


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_nested(value: Any) -> bool:
    return isinstance(value, (list, dict, set, tuple))


def _looks_like_bool(values: Sequence[Any]) -> bool:
    normalized: set[str] = set()
    for value in values:
        if _is_missing(value):
            continue
        if isinstance(value, (bool, np.bool_)):
            normalized.add(str(bool(value)).lower())
            continue
        text = str(value).strip().lower()
        if text in {"true", "false", "0", "1"}:
            normalized.add("true" if text in {"true", "1"} else "false")
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            normalized.add(str(bool(int(value))).lower())
        else:
            return False
    return 1 <= len(normalized) <= 2


def _try_parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return pd.to_datetime(text, utc=False).to_pydatetime()
    except (TypeError, ValueError, OverflowError):
        return None


def _majority_datetime(values: Sequence[Any]) -> bool:
    non_missing = [v for v in values if not _is_missing(v)]
    if not non_missing:
        return False
    parsed = sum(1 for v in non_missing if _try_parse_datetime(v) is not None)
    return parsed / len(non_missing) >= 0.8


def _is_numeric_series(values: Sequence[Any]) -> bool:
    non_missing = [v for v in values if not _is_missing(v)]
    if not non_missing:
        return False
    ok = 0
    for value in non_missing:
        if isinstance(value, (bool, np.bool_)):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            if np.isfinite(float(value)):
                ok += 1
            continue
        try:
            float(str(value).replace(",", ""))
            ok += 1
        except (TypeError, ValueError):
            return False
    return ok == len(non_missing)


def _to_float(value: Any) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return float(str(value).replace(",", ""))


def _name_suggests_identifier(attribute_name: str) -> bool:
    return any(pattern.search(attribute_name) for pattern in IDENTIFIER_NAME_PATTERNS)


def infer_attribute_type(
    values: Sequence[Any],
    attribute_name: str,
    *,
    thresholds: RecommendationThresholds | None = None,
) -> str:
    """Conservatively infer an attribute datatype from observed values."""
    thresholds = thresholds or RecommendationThresholds()
    non_missing = [v for v in values if not _is_missing(v)]
    if any(_is_nested(v) for v in non_missing):
        return "unsupported"
    if not non_missing:
        return "unsupported"
    if _looks_like_bool(non_missing):
        return "boolean"
    if _majority_datetime(non_missing):
        return "datetime"

    n_nonmissing = len(non_missing)
    n_distinct = len({str(v) for v in non_missing})
    distinct_share = n_distinct / n_nonmissing if n_nonmissing else 0.0

    identifier_by_share = distinct_share >= thresholds.identifier_distinct_share
    identifier_by_large = (
        distinct_share >= thresholds.identifier_distinct_share_large
        and n_distinct >= thresholds.identifier_n_distinct_large
    )
    if identifier_by_share or identifier_by_large or _name_suggests_identifier(
        attribute_name
    ):
        if (
            n_distinct >= max(10, int(0.5 * n_nonmissing))
            or identifier_by_share
            or identifier_by_large
        ):
            return "identifier-like"

    if _is_numeric_series(non_missing):
        # Low-cardinality integers stay categorical codes; otherwise continuous.
        if n_distinct >= thresholds.continuous_min_distinct:
            return "continuous"
        return "categorical"

    return "categorical"


def describe_attribute(
    values: Sequence[Any] | pd.Series,
    *,
    attribute_name: str,
    source_level: str,
    thresholds: RecommendationThresholds | None = None,
    inferred_type: str | None = None,
    n_example_values: int = 5,
) -> AttributeInventory:
    """Compute descriptive inventory statistics for one attribute."""
    thresholds = thresholds or RecommendationThresholds()
    series = pd.Series(list(values), dtype=object)
    n_cases = int(len(series))
    missing_mask = series.map(_is_missing)
    n_missing = int(missing_mask.sum())
    non_missing = series[~missing_mask]
    n_nonmissing = int(len(non_missing))

    inferred = inferred_type or infer_attribute_type(
        series.tolist(), attribute_name, thresholds=thresholds
    )

    if n_nonmissing == 0:
        n_distinct = 0
        distinct_share = 0.0
        most_frequent = None
        largest_share = 0.0
        example_values: list[Any] = []
    else:
        value_counts = non_missing.map(lambda v: str(v)).value_counts()
        n_distinct = int(value_counts.size)
        distinct_share = n_distinct / n_cases if n_cases else 0.0
        top_key = value_counts.index[0]
        largest_share = float(value_counts.iloc[0] / n_cases) if n_cases else 0.0
        most_frequent = next(
            (v for v in non_missing.tolist() if str(v) == top_key),
            top_key,
        )
        example_values = []
        seen: set[str] = set()
        for value in non_missing.tolist():
            key = str(value)
            if key in seen:
                continue
            seen.add(key)
            example_values.append(_json_safe(value))
            if len(example_values) >= n_example_values:
                break

    min_value = max_value = median_value = mean_value = None
    if inferred == "continuous" and n_nonmissing > 0:
        numeric = np.asarray(
            [_to_float(v) for v in non_missing.tolist()], dtype=float
        )
        min_value = float(np.min(numeric))
        max_value = float(np.max(numeric))
        median_value = float(np.median(numeric))
        mean_value = float(np.mean(numeric))

    is_constant = n_distinct <= 1
    is_near_constant = largest_share >= thresholds.near_constant_largest_share
    is_identifier_like = inferred == "identifier-like"
    suitable = (
        inferred in {"categorical", "boolean"}
        and not is_constant
        and not is_identifier_like
        and (
            source_level == "case"
            or (
                source_level == "event"
                and is_event_aggregation_column(attribute_name)
            )
        )
    )
    special_handling = (
        inferred
        in {
            "continuous",
            "datetime",
            "unsupported",
            "identifier-like",
        }
        or (
            source_level == "event"
            and not is_event_aggregation_column(attribute_name)
        )
    )

    return AttributeInventory(
        attribute_name=attribute_name,
        source_level=source_level,
        inferred_type=inferred,
        n_cases=n_cases,
        n_missing=n_missing,
        missing_share=(n_missing / n_cases) if n_cases else 0.0,
        n_distinct=n_distinct,
        distinct_share=distinct_share,
        most_frequent_value=_json_safe(most_frequent),
        largest_value_share=largest_share,
        min_value=min_value,
        max_value=max_value,
        median_value=median_value,
        mean_value=mean_value,
        example_values=example_values,
        is_constant=is_constant,
        is_near_constant=is_near_constant,
        is_identifier_like=is_identifier_like,
        suitable_for_categorical_frequency=suitable,
        special_handling_required=special_handling,
    )


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return str(value)
    if _is_nested(value):
        return str(value)
    return str(value)


def recommend_datatype(
    inventory: AttributeInventory,
    *,
    thresholds: RecommendationThresholds | None = None,
) -> DatatypeRecommendation:
    """Step-1 recommendations: include for relevance analysis, datatype, binning."""
    thresholds = thresholds or RecommendationThresholds()
    datatype = inventory.inferred_type
    include = True
    reasons: list[str] = []
    binning_method: str | None = None
    n_bins: int | None = None
    bin_edges = None

    if datatype == "unsupported":
        include = False
        reasons.append("Unsupported nested or structured values.")
    elif datatype == "datetime":
        include = False
        reasons.append(
            "Datetime attribute excluded without an explicit transformation."
        )
    elif datatype == "identifier-like" or inventory.is_identifier_like:
        include = False
        reasons.append("Identifier or near-identifier attribute.")
    elif inventory.is_constant or inventory.is_near_constant:
        include = False
        reasons.append("Constant or near-constant attribute.")
    elif is_default_excluded_attribute(inventory.attribute_name):
        include = False
        if str(inventory.attribute_name).startswith("org:resource"):
            reasons.append("org:resource excluded from default inclusion.")
        else:
            reasons.append(
                "Non-first event aggregations (__constant/__last/__mode/"
                "__n_unique) are excluded from default inclusion."
            )
    elif inventory.source_level == "event" and not is_event_aggregation_column(
        inventory.attribute_name
    ):
        include = False
        reasons.append(
            "Event-level attribute without a selected case-level aggregation column."
        )
    elif inventory.missing_share >= thresholds.excessive_missing_share:
        include = False
        reasons.append(f"Excessive missingness ({inventory.missing_share:.1%}).")
    elif datatype == "continuous":
        include = True
        binning_method = "log_bins"
        n_bins = thresholds.default_n_bins
        reasons.append(
            "Continuous attribute; suggested log binning for relevance / RFC "
            "(power-law magnitude fits use raw values)."
        )
    elif datatype == "boolean":
        include = True
        reasons.append("Boolean case attribute.")
    else:
        include = True
        reasons.append("Categorical case attribute.")

    if (
        inventory.source_level == "event"
        and is_event_aggregation_column(inventory.attribute_name)
        and include
    ):
        reasons.append("Case-level aggregation of an event attribute.")

    return DatatypeRecommendation(
        include=include,
        datatype=datatype,
        reason=" ".join(reasons).strip() or "No automatic recommendation notes.",
        binning_method=binning_method,
        n_bins=n_bins,
        bin_edges=bin_edges,
    )


def values_are_integer_valued(
    values: Sequence[Any] | np.ndarray,
    *,
    atol: float = 1e-9,
) -> bool:
    """Return True if all finite values are integers within ``atol``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return False
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(np.all(np.abs(finite - np.round(finite)) <= atol))


def recommend_powerlaw_discrete(
    *,
    datatype: str,
    values: Sequence[Any] | np.ndarray | None = None,
) -> bool:
    """Suggest ``powerlaw.Fit(..., discrete=...)`` for the observation vector.

    Categorical / boolean frequency fits are always discrete.
    Continuous magnitude fits are discrete when values are integer-valued;
    otherwise continuous (``discrete=False``).
    """
    if datatype in {"categorical", "boolean"}:
        return True
    if datatype == "continuous":
        if values is None:
            return False
        return values_are_integer_valued(values)
    # datetime / identifier-like / unsupported: unused when include=False
    return True


def recommend_powerlaw_include(
    *,
    datatype: str,
    n_distinct_transformed: int,
    is_near_constant: bool,
    step1_included: bool,
    association: Mapping[str, Any] | None = None,
    thresholds: RecommendationThresholds | None = None,
    n_observations: int | None = None,
    n_raw_distinct: int | None = None,
    raw_values: Sequence[Any] | np.ndarray | None = None,
) -> PowerlawIncludeRecommendation:
    """Step-2 suggestion for power-law testing inclusion."""
    thresholds = thresholds or RecommendationThresholds()
    association = association or {}
    reasons: list[str] = []
    discrete = recommend_powerlaw_discrete(datatype=datatype, values=raw_values)

    if not step1_included:
        return PowerlawIncludeRecommendation(
            include=False,
            reason="Not included in step-1 datatype selection.",
            discrete=discrete,
        )
    if datatype in {"datetime", "identifier-like", "unsupported"}:
        return PowerlawIncludeRecommendation(
            include=False,
            reason=(
                f"Datatype {datatype!r} is not suitable for "
                "power-law testing."
            ),
            discrete=discrete,
        )
    if is_near_constant:
        return PowerlawIncludeRecommendation(
            include=False,
            reason="Near-constant attribute after transformation.",
            discrete=discrete,
        )

    ner = association.get("normalized_entropy_reduction")
    try:
        ner_f = (
            float(ner)
            if ner is not None and not (isinstance(ner, float) and math.isnan(ner))
            else None
        )
    except (TypeError, ValueError):
        ner_f = None
    if ner_f is not None and ner_f >= thresholds.strong_relevance_ner:
        reasons.append("Behaviorally relevant to trace variants.")
    if ner_f is not None and ner_f >= thresholds.trace_encoding_ner:
        reasons.append(
            "Normalized entropy reduction is near 1.0; may encode the variant directly."
        )

    if datatype == "continuous":
        n_obs = int(
            n_observations
            if n_observations is not None
            else association.get("n_cases") or 0
        )
        min_obs = int(thresholds.minimum_observations_for_continuous_powerlaw)
        if n_obs < min_obs:
            return PowerlawIncludeRecommendation(
                include=False,
                reason=(
                    f"Only {n_obs} observations for continuous magnitude fit "
                    f"(<{min_obs})."
                ),
                discrete=discrete,
            )
        raw_distinct = (
            int(n_raw_distinct)
            if n_raw_distinct is not None
            else int(n_distinct_transformed)
        )
        support = "discrete" if discrete else "continuous"
        reasons.append(
            f"{n_obs} observations ({raw_distinct} distinct raw values); "
            f"eligible for magnitude power-law fitting ({support} MLE)."
        )
        return PowerlawIncludeRecommendation(
            include=True,
            reason=" ".join(reasons).strip(),
            discrete=discrete,
        )

    if n_distinct_transformed < thresholds.minimum_distinct_for_powerlaw:
        return PowerlawIncludeRecommendation(
            include=False,
            reason=(
                f"Only {n_distinct_transformed} distinct transformed values "
                f"(<{thresholds.minimum_distinct_for_powerlaw})."
            ),
            discrete=discrete,
        )
    reasons.append(
        f"{n_distinct_transformed} distinct transformed values; "
        "eligible for discrete frequency distribution fitting."
    )
    return PowerlawIncludeRecommendation(
        include=True,
        reason=" ".join(reasons).strip(),
        discrete=discrete,
    )


def inventory_to_row(inventory: AttributeInventory) -> dict[str, Any]:
    """Flatten an inventory dataclass to a CSV-friendly dict."""
    row = asdict(inventory)
    row["example_values"] = "|".join(str(v) for v in inventory.example_values)
    return row
