from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.obj import EventLog

from utils.constants import DATA_DICTIONARY_PATH


def get_data_dictionary(
    path: Path | None = None,
    get_real: bool = True,
    get_synthetic: bool = False,
) -> dict[str, Any]:
    """Return data-dictionary entries filtered by requested dataset types."""
    data_dictionary_path = path or DATA_DICTIONARY_PATH
    with open(data_dictionary_path, "r", encoding="utf-8") as file:
        data_dictionary: dict[str, Any] = json.load(file)

    allowed_types = set()
    if get_real:
        allowed_types.add("real")
    if get_synthetic:
        allowed_types.add("synthetic")
    if not allowed_types:
        return {}

    filtered: dict[str, Any] = {
        dataset_name: dataset_info
        for dataset_name, dataset_info in data_dictionary.items()
        if dataset_info.get("type") in allowed_types
    }
    for dataset_info in filtered.values():
        dataset_info.setdefault("activity_key", "concept:name")
    return filtered


def get_event_log_from_path(log_path: Path, *, timestamp_sort: bool = True) -> EventLog:
    """Load one XES or XES.GZ event log from disk."""
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: timestamp_sort}
    return xes_importer.apply(str(log_path), variant=variant, parameters=parameters)


def get_event_log_from_dictionary(
    dataset_name: str,
    data_dictionary: dict[str, Any],
    *,
    timestamp_sort: bool = True,
) -> EventLog:
    """Load a dataset event log using a data-dictionary entry."""
    if dataset_name not in data_dictionary:
        raise KeyError(f"Dataset '{dataset_name}' not found in data dictionary.")
    log_path = Path(data_dictionary[dataset_name]["path"])
    return get_event_log_from_path(log_path, timestamp_sort=timestamp_sort)
