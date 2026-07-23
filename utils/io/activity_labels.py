"""Activity labels from XES Activity classifier metadata on loaded event logs."""

from __future__ import annotations

from pm4py.objects.log.obj import EventLog, Trace

ACTIVITY_CLASSIFIER_NAME = "Activity classifier"
DEFAULT_ACTIVITY_FIELD = "concept:name"


def parse_classifier_keys(keys: str | list[str]) -> list[str]:
    """Normalize XES classifier keys (string or list from PM4Py)."""
    if isinstance(keys, list):
        return [str(key) for key in keys]
    return keys.split()


def resolve_activity_classifier_fields(event_log: EventLog) -> list[str] | None:
    """Return Activity classifier field list, or None to use concept:name only."""
    classifiers = getattr(event_log, "classifiers", None) or {}
    if ACTIVITY_CLASSIFIER_NAME not in classifiers:
        return None
    return parse_classifier_keys(classifiers[ACTIVITY_CLASSIFIER_NAME])


def format_event_activity(event: dict, fields: list[str] | None) -> str:
    """Build one activity label from classifier fields or concept:name fallback."""
    if fields:
        return "+".join(str(event.get(key, "")) for key in fields)
    return str(event.get(DEFAULT_ACTIVITY_FIELD, ""))


def trace_activity_sequence(trace: Trace, fields: list[str] | None) -> tuple[str, ...]:
    """Return per-event activity labels for one trace."""
    return tuple(format_event_activity(event, fields) for event in trace)


def variant_and_activity_counts(event_log: EventLog) -> tuple[int, int]:
    """Return distinct variant count and distinct activity label count."""
    fields = resolve_activity_classifier_fields(event_log)
    sequences: list[tuple[str, ...]] = []
    activities: set[str] = set()
    for trace in event_log:
        if not trace:
            continue
        sequence = trace_activity_sequence(trace, fields)
        sequences.append(sequence)
        activities.update(sequence)
    return len(set(sequences)), len(activities)
