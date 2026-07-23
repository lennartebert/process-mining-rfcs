"""Helpers to merge and pair simulated event logs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import ActivitySetOverlap


def relabel_cases(event_log: pd.DataFrame, start_case_index: int = 1) -> tuple[pd.DataFrame, int]:
    """Relabel case IDs to avoid collisions when concatenating multiple logs."""
    case_ids = event_log["case"].drop_duplicates().tolist()
    mapping = {
        old_case_id: f"case_{start_case_index + idx}"
        for idx, old_case_id in enumerate(case_ids)
    }
    relabeled = event_log.copy()
    relabeled["case"] = relabeled["case"].map(mapping)
    next_index = start_case_index + len(case_ids)
    return relabeled, next_index


def combine_logs(logs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine logs while preserving unique case IDs across segments."""
    relabeled_logs = []
    next_case_idx = 1
    for log in logs:
        relabeled, next_case_idx = relabel_cases(log, start_case_index=next_case_idx)
        relabeled_logs.append(relabeled)
    if not relabeled_logs:
        return pd.DataFrame(columns=["case", "activity", "timestamp"])
    return pd.concat(relabeled_logs, ignore_index=True)


def extract_traces(event_log: pd.DataFrame) -> list[list[str]]:
    """Return one activity list per case, in case order."""
    traces = []
    for case_id in event_log["case"].drop_duplicates():
        activities = event_log.loc[event_log["case"] == case_id, "activity"].tolist()
        if activities:
            traces.append(activities)
    return traces


def prefix_activities(activities: list[str], prefix: str) -> list[str]:
    """Prefix each activity name (e.g. ``a:activity``)."""
    return [f"{prefix}:{activity}" for activity in activities]


def prefix_event_log(event_log: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return a copy of the log with prefixed activity names."""
    prefixed = event_log.copy()
    prefixed["activity"] = prefixed["activity"].apply(lambda name: f"{prefix}:{name}")
    return prefixed


def pair_and_concat_traces(
    traces_a: list[list[str]],
    traces_b: list[list[str]],
    seed: int,
    prefix_a: str | None = None,
    prefix_b: str | None = None,
) -> list[list[str]]:
    """Pair traces 1:1 via a random permutation of *b*; each trace used exactly once."""
    if len(traces_a) != len(traces_b):
        raise ValueError(
            f"Trace counts must match for pairing (got {len(traces_a)} and {len(traces_b)})."
        )
    if not traces_a:
        return []

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(traces_b))
    combined: list[list[str]] = []
    for idx_a in range(len(traces_a)):
        idx_b = int(order[idx_a])
        left = traces_a[idx_a]
        right = traces_b[idx_b]
        if prefix_a is not None:
            left = prefix_activities(left, prefix_a)
        if prefix_b is not None:
            right = prefix_activities(right, prefix_b)
        combined.append(left + right)
    return combined


def traces_to_event_log(traces: list[list[str]]) -> pd.DataFrame:
    """Build a minimal event log from trace activity lists."""
    rows = []
    for case_idx, activities in enumerate(traces, start=1):
        case_id = f"case_{case_idx}"
        for activity in activities:
            rows.append({"case": case_id, "activity": activity})
    return pd.DataFrame(rows)


def prepare_logs_for_drift(
    log_a: pd.DataFrame,
    log_b: pd.DataFrame,
    s: ActivitySetOverlap,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply overlapping-activity-set policy before concatenating drift halves."""
    if s == "non_overlapping":
        return prefix_event_log(log_a, "a"), prefix_event_log(log_b, "b")
    return log_a.copy(), log_b.copy()
