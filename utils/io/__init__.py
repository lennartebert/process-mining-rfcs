"""I/O helpers for event logs and attachments."""

from __future__ import annotations

import pandas as pd

from .activity_labels import (
    ACTIVITY_CLASSIFIER_NAME,
    format_event_activity,
    parse_classifier_keys,
    resolve_activity_classifier_fields,
    trace_activity_sequence,
    variant_and_activity_counts,
)
from .attachments import (
    REQUIRED_ATTACHMENT_COLUMNS,
    extract_attachments,
    extract_attachments_from_trace_data,
    load_attachments,
    parse_dataset_input,
    parse_dataset_inputs,
    save_attachments,
    trace_completion_data,
)
from .case_tables import (
    CASE_ID_COLUMN,
    VARIANT_COLUMN,
    build_case_attribute_table,
    candidate_attribute_names,
    is_default_excluded_attribute,
    is_event_aggregation_column,
)
from .event_logs import (
    get_data_dictionary,
    get_event_log_from_dictionary,
    get_event_log_from_path,
)


def parse_count(value: object) -> float:
    """Parse a count stored as a number or a comma-formatted string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text.replace(",", ""))


__all__ = [
    "ACTIVITY_CLASSIFIER_NAME",
    "CASE_ID_COLUMN",
    "REQUIRED_ATTACHMENT_COLUMNS",
    "VARIANT_COLUMN",
    "build_case_attribute_table",
    "candidate_attribute_names",
    "is_default_excluded_attribute",
    "is_event_aggregation_column",
    "extract_attachments",
    "extract_attachments_from_trace_data",
    "get_data_dictionary",
    "get_event_log_from_dictionary",
    "get_event_log_from_path",
    "format_event_activity",
    "load_attachments",
    "parse_classifier_keys",
    "parse_count",
    "parse_dataset_input",
    "parse_dataset_inputs",
    "resolve_activity_classifier_fields",
    "save_attachments",
    "trace_activity_sequence",
    "trace_completion_data",
    "variant_and_activity_counts",
]
