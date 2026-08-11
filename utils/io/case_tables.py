"""Build case-level attribute tables from PM4Py event logs."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pm4py.objects.log.obj import EventLog, Trace

from .activity_labels import (
    DEFAULT_ACTIVITY_FIELD,
    resolve_activity_classifier_fields,
    trace_activity_sequence,
)

CASE_ID_KEY = "concept:name"
VARIANT_COLUMN = "variant"
CASE_ID_COLUMN = "case_id"

# Aggregations materialised from event-level attributes onto the case table.
# Only ``__first`` is emitted by default to keep HITL inventories manageable.
EVENT_AGGREGATION_SUFFIXES = ("__first",)

# Suffixes that must never be suggested for inclusion (also not materialised).
EXCLUDED_EVENT_AGGREGATION_SUFFIXES = (
    "__constant",
    "__last",
    "__mode",
    "__n_unique",
)

# Attribute bases excluded from default inclusion (any ``__*`` aggregation too).
DEFAULT_EXCLUDED_ATTRIBUTE_BASES = frozenset({"org:resource"})


def is_event_aggregation_column(name: str) -> bool:
    """Return True when *name* is a case-table projection of an event attribute."""
    return any(str(name).endswith(suffix) for suffix in EVENT_AGGREGATION_SUFFIXES)


def attribute_base_name(name: str) -> str:
    """Strip a known aggregation suffix, if present."""
    text = str(name)
    for suffix in EVENT_AGGREGATION_SUFFIXES + EXCLUDED_EVENT_AGGREGATION_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def is_default_excluded_attribute(name: str) -> bool:
    """Return True for attributes excluded from default inclusion suggestions."""
    text = str(name)
    if any(text.endswith(suffix) for suffix in EXCLUDED_EVENT_AGGREGATION_SUFFIXES):
        return True
    return attribute_base_name(text) in DEFAULT_EXCLUDED_ATTRIBUTE_BASES


# Standard XES / activity keys excluded from candidate attributes.
ALWAYS_EXCLUDED_EVENT_KEYS = frozenset(
    {
        "time:timestamp",
        DEFAULT_ACTIVITY_FIELD,
        "lifecycle:transition",
    }
)


def _is_missing(value: Any) -> bool:
    """Return True when *value* should be treated as missing."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_value(value: Any) -> Any:
    """Normalize a raw attribute value for tabular storage."""
    if _is_missing(value):
        return None
    if isinstance(value, (list, dict, set, tuple)):
        return value
    return value


def _event_noise_keys(classifier_fields: list[str] | None) -> frozenset[str]:
    """Keys that should not become candidate attributes."""
    noise = set(ALWAYS_EXCLUDED_EVENT_KEYS)
    if classifier_fields:
        noise.update(classifier_fields)
    return frozenset(noise)


def _trace_case_id(trace: Trace, fallback_index: int) -> str:
    """Return the case identifier from trace attributes, with a stable fallback."""
    attributes = getattr(trace, "attributes", None) or {}
    raw = attributes.get(CASE_ID_KEY)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return f"case_{fallback_index}"
    return str(raw)


def _first_non_missing(values: list[Any]) -> Any:
    for value in values:
        if not _is_missing(value):
            return value
    return None


def build_case_attribute_table(
    event_log: EventLog,
    *,
    include_event_aggregates: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return one row per non-empty trace plus source-level map for attributes.

    Columns always include ``case_id`` and ``variant``. Case-level attributes
    come from ``trace.attributes`` (excluding ``concept:name``). Event-level
    attributes are materialised as ``{attr}__first`` when
    ``include_event_aggregates`` is True (``source_level="event"``).

    Bare event-attribute names are never used as columns.

    Returns
    -------
    case_df :
        Case-level DataFrame.
    source_levels :
        Mapping attribute_name -> ``case`` | ``event`` | ``unknown``.
    """
    classifier_fields = resolve_activity_classifier_fields(event_log)
    noise_keys = _event_noise_keys(classifier_fields)

    rows: list[dict[str, Any]] = []
    event_keys_seen: set[str] = set()

    for index, trace in enumerate(event_log):
        if not trace:
            continue

        row: dict[str, Any] = {
            CASE_ID_COLUMN: _trace_case_id(trace, index),
            VARIANT_COLUMN: str(trace_activity_sequence(trace, classifier_fields)),
        }

        case_attrs = getattr(trace, "attributes", None) or {}
        for key, value in case_attrs.items():
            if key == CASE_ID_KEY:
                continue
            row[str(key)] = _normalize_value(value)

        per_event: dict[str, list[Any]] = {}
        for event in trace:
            for key, value in event.items():
                key_str = str(key)
                if key_str in noise_keys:
                    continue
                event_keys_seen.add(key_str)
                per_event.setdefault(key_str, []).append(_normalize_value(value))

        if include_event_aggregates:
            for key_str, values in per_event.items():
                row[f"{key_str}__first"] = _first_non_missing(values)

        rows.append(row)

    case_df = pd.DataFrame(rows)
    if case_df.empty:
        case_df = pd.DataFrame(columns=[CASE_ID_COLUMN, VARIANT_COLUMN])

    source_levels = _infer_source_levels(
        case_df=case_df,
        case_attr_keys={
            str(k)
            for trace in event_log
            if trace
            for k in (getattr(trace, "attributes", None) or {})
            if k != CASE_ID_KEY
        },
        event_keys_seen=event_keys_seen,
        include_event_aggregates=include_event_aggregates,
    )
    return case_df, source_levels


def _infer_source_levels(
    *,
    case_df: pd.DataFrame,
    case_attr_keys: set[str],
    event_keys_seen: set[str],
    include_event_aggregates: bool,
) -> dict[str, str]:
    """Assign source_level labels for every attribute column."""
    source_levels: dict[str, str] = {}
    reserved = {CASE_ID_COLUMN, VARIANT_COLUMN}

    for column in case_df.columns:
        if column in reserved:
            continue
        if column in case_attr_keys:
            source_levels[column] = "case"
            continue
        if is_event_aggregation_column(column):
            source_levels[column] = "event"
            continue
        source_levels[column] = "unknown"

    if include_event_aggregates:
        for key in event_keys_seen:
            source_levels.setdefault(key, "event")

    return source_levels


def candidate_attribute_names(
    case_df: pd.DataFrame,
    source_levels: dict[str, str],
) -> list[str]:
    """Return sorted candidate attribute names present as case-table columns."""
    del source_levels  # retained for API symmetry with callers
    reserved = {CASE_ID_COLUMN, VARIANT_COLUMN}
    return sorted(col for col in case_df.columns if col not in reserved)
