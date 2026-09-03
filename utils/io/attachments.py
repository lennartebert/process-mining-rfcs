"""Attachment extraction and CSV I/O helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from pm4py.objects.log.obj import EventLog

from .activity_labels import (
    resolve_activity_classifier_fields,
    trace_activity_sequence,
)

REQUIRED_ATTACHMENT_COLUMNS = ["attachment_time", "attachment_index", "node_id"]

# Boundary labels prepended/appended when extracting n-grams with n >= 2.
# Change these if they collide with real activity names in a log.
START = "START"
END = "END"
NGRAM_CONCEPTS = tuple(f"n{i}" for i in range(1, 11))


def parse_dataset_input(raw_value: str) -> Tuple[str, Path]:
    """Parse one CLI input pair in the form <dataset>=<attachments_csv_path>."""
    if "=" not in raw_value:
        raise ValueError(
            f"Invalid --input value '{raw_value}'. Expected <dataset>=<attachments_csv_path>."
        )
    dataset_name, csv_path = raw_value.split("=", 1)
    dataset_name = dataset_name.strip()
    csv_path = csv_path.strip()
    if not dataset_name or not csv_path:
        raise ValueError(
            f"Invalid --input value '{raw_value}'. Expected non-empty dataset and file path."
        )
    return dataset_name, Path(csv_path)


def parse_dataset_inputs(raw_values: List[str]) -> List[Tuple[str, Path]]:
    """Parse all dataset input pairs and guard against duplicate dataset names."""
    pairs = [parse_dataset_input(raw) for raw in raw_values]
    names = [name for name, _ in pairs]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"duplicated dataset names in --inputs: {duplicated}")
    return pairs


def load_attachments(attachments_path: Path) -> pd.DataFrame:
    """Load and validate one attachment CSV/CSV.GZ file."""
    df = pd.read_csv(attachments_path, compression="infer")
    missing = [col for col in REQUIRED_ATTACHMENT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {attachments_path}: {missing}")
    df["attachment_index"] = pd.to_numeric(df["attachment_index"])
    return df.sort_values("attachment_index").reset_index(drop=True)


def save_attachments(attachments_df: pd.DataFrame, attachments_path: Path) -> None:
    """Write attachment rows to CSV/CSV.GZ."""
    attachments_path.parent.mkdir(parents=True, exist_ok=True)
    attachments_df.to_csv(attachments_path, index=False, compression="infer")


def trace_completion_data(event_log: EventLog) -> List[Dict[str, object]]:
    """Return traces with completion timestamp and activity sequence."""
    classifier_fields = resolve_activity_classifier_fields(event_log)
    trace_rows: List[Dict[str, object]] = []
    for trace in event_log:
        if not trace:
            continue
        completion_timestamp = trace[-1].get("time:timestamp")
        if completion_timestamp is None:
            continue
        activity_sequence = trace_activity_sequence(trace, classifier_fields)
        trace_rows.append(
            {
                "completion_timestamp": completion_timestamp,
                "activity_sequence": activity_sequence,
            }
        )
    return trace_rows


def start_end_labels_in_activity_set(
    trace_rows: Sequence[Dict[str, object]],
) -> Tuple[str, ...]:
    """Return START/END labels that already occur as activities in ``trace_rows``."""
    activities = {
        activity
        for row in trace_rows
        for activity in row["activity_sequence"]
    }
    return tuple(label for label in (START, END) if label in activities)


def warn_if_start_end_in_activity_set(
    trace_rows: Sequence[Dict[str, object]],
    *,
    dataset_name: str | None = None,
) -> Tuple[str, ...]:
    """Print a warning if START or END already appears in the log's activity set.

    Returns the colliding labels (empty if none).
    """
    collisions = start_end_labels_in_activity_set(trace_rows)
    if not collisions:
        return collisions
    where = f" in {dataset_name}" if dataset_name else ""
    collided = ", ".join(repr(label) for label in collisions)
    print(
        f"Warning: activity set{where} already contains n-gram boundary "
        f"label(s) {collided}. Wrapping traces with {START!r}/{END!r} will mix "
        "pads with real activities. Change START/END at the top of "
        "utils/io/attachments.py if needed."
    )
    return collisions


def _extended_trace(
    seq: Tuple[str, ...], n: int, add_start_end: bool
) -> Tuple[str, ...]:
    """Return the sequence n-grams are taken from.

    For n >= 2 with add_start_end, wrap with START/END. n1 never wraps.
    """
    if add_start_end and n >= 2:
        return (START, *seq, END)
    return seq


def _ngram_nodes(
    seq: Tuple[str, ...], n: int, add_start_end: bool = True
) -> List[str]:
    """Return n-gram node ids for one activity sequence.

    Assumptions:
    - nk means contiguous subtraces of length k (n1 ≡ activities; n2 ≡ DFR pairs).
    - For n >= 2 and add_start_end, work on the extended trace (START, *seq, END).
    - n1 never includes START/END.
    - If n >= |work|, emit the entire work sequence as one node (no padding).
    - If n < |work|, emit sliding windows of length n.
    - n1 node ids are bare activity strings; n2..n10 use str(tuple(...)).
    """
    work = _extended_trace(seq, n, add_start_end)
    if n == 1:
        return [str(activity) for activity in work]
    if n >= len(work):
        return [str(work)]
    return [str(work[i : i + n]) for i in range(len(work) - n + 1)]


def _node_ids_for_concept(
    activity_sequence: Tuple[str, ...],
    concept: str,
    add_start_end: bool = True,
) -> List[str]:
    """Return node ids contributed by one trace for the selected concept."""
    if concept == "variants":
        return [str(activity_sequence)]
    if concept == "activities":
        return [str(activity) for activity in activity_sequence]
    if concept == "dfrs":
        return [str((source, target)) for source, target in zip(activity_sequence, activity_sequence[1:])]
    if concept in NGRAM_CONCEPTS:
        return _ngram_nodes(activity_sequence, int(concept[1:]), add_start_end)
    raise ValueError(f"Unsupported concept: {concept}")


def extract_attachments(
    event_log: EventLog,
    concept: str,
    add_start_end: bool = True,
) -> pd.DataFrame:
    """Return attachment rows with time, index, and concept-specific node id."""
    trace_rows = trace_completion_data(event_log)
    trace_rows.sort(key=lambda item: item["completion_timestamp"])
    if add_start_end and concept in NGRAM_CONCEPTS and concept != "n1":
        warn_if_start_end_in_activity_set(trace_rows)
    return extract_attachments_from_trace_data(
        trace_rows, concept, add_start_end=add_start_end
    )


def extract_attachments_from_trace_data(
    trace_rows: List[Dict[str, object]],
    concept: str,
    add_start_end: bool = True,
) -> pd.DataFrame:
    """Return attachment rows for one concept from precomputed trace rows."""
    attachments = []
    attachment_index = 0
    for row in trace_rows:
        completion_ts = row["completion_timestamp"]
        timestamp_str = completion_ts.isoformat() if isinstance(completion_ts, datetime) else str(completion_ts)
        activity_sequence = row["activity_sequence"]
        for node_id in _node_ids_for_concept(
            activity_sequence, concept, add_start_end=add_start_end
        ):
            attachments.append(
                {
                    "attachment_time": timestamp_str,
                    "attachment_index": attachment_index,
                    "node_id": node_id,
                }
            )
            attachment_index += 1
    return pd.DataFrame(attachments)
