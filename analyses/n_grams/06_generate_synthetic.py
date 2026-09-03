"""Standalone first-order DFG simulator for n-grams analysis.

For each source log, count directly-follows relations (with START/END pads),
build a per-activity CDF of next activities, and sample the same number of
traces as the source. Writes ``data/synthetic/<LOG>_sim/<LOG>_sim.xes.gz``
(without START/END) and upserts ``data/data_dictionary.json``.

Not wired into ``main.py``. Run directly:

    python analyses/n_grams/06_generate_synthetic.py
    python analyses/n_grams/06_generate_synthetic.py --datasets ACCRE
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import Event, EventLog, Trace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ALL_REAL_LOGS_TOKEN,
    DATA_DICTIONARY_PATH,
    DATA_DIR,
)
from utils.io.activity_labels import (
    resolve_activity_classifier_fields,
    trace_activity_sequence,
)
from utils.io.event_logs import get_data_dictionary, get_event_log_from_path

START = "START"
END = "END"
CUTOFF_MULTIPLIER = 10
SIM_SUFFIX = "_sim"

NextActivitiesMap = dict[str, list[tuple[float, str]]]


def _resolve_datasets(raw: Sequence[str]) -> List[str]:
    if len(raw) == 1 and raw[0] == ALL_REAL_LOGS_TOKEN:
        return list(ALL_REAL_LOG_DATASETS)
    return list(raw)


def _source_sequences(event_log: EventLog) -> list[tuple[str, ...]]:
    fields = resolve_activity_classifier_fields(event_log)
    sequences: list[tuple[str, ...]] = []
    for trace in event_log:
        if not trace:
            continue
        sequences.append(trace_activity_sequence(trace, fields))
    return sequences


def build_next_activities_map(
    sequences: Iterable[Sequence[str]],
) -> NextActivitiesMap:
    """Build ``current -> [(cum_p, next), ...]`` from padded DF counts."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence in sequences:
        if not sequence:
            continue
        padded = (START, *sequence, END)
        for current, nxt in zip(padded, padded[1:]):
            counts[current][nxt] += 1

    next_map: NextActivitiesMap = {}
    for current, counter in counts.items():
        total = sum(counter.values())
        pairs: list[tuple[float, str]] = []
        items = list(counter.items())
        running = 0.0
        for i, (nxt, count) in enumerate(items):
            if i == len(items) - 1:
                running = 1.0
            else:
                running += count / total
            pairs.append((running, nxt))
        next_map[current] = pairs
    return next_map


def sample_next_activity(
    current: str,
    next_map: NextActivitiesMap,
    rng: random.Random,
) -> str:
    pairs = next_map[current]
    index = bisect_left(pairs, rng.random(), key=lambda pair: pair[0])
    if index >= len(pairs):
        index = len(pairs) - 1
    return pairs[index][1]


def sample_trace(
    next_map: NextActivitiesMap,
    rng: random.Random,
    cutoff: int,
) -> list[str]:
    """Walk START -> END; return activities without START/END."""
    activities: list[str] = []
    current = START
    while current != END and len(activities) < cutoff:
        if current not in next_map:
            raise RuntimeError(f"No outgoing DF relations for activity {current!r}")
        current = sample_next_activity(current, next_map, rng)
        if current != END:
            activities.append(current)
    return activities


def traces_to_event_log(traces: Sequence[Sequence[str]]) -> EventLog:
    event_log = EventLog()
    start_time = datetime(2020, 1, 1)
    for case_index, activities in enumerate(traces, start=1):
        trace = Trace()
        trace.attributes["concept:name"] = f"case_{case_index}"
        timestamp = start_time
        for activity in activities:
            event = Event(
                {"concept:name": activity, "time:timestamp": timestamp}
            )
            trace.append(event)
            timestamp += timedelta(hours=1)
        event_log.append(trace)
    return event_log


def sim_log_path(dataset_name: str) -> Path:
    sim_name = f"{dataset_name}{SIM_SUFFIX}"
    return DATA_DIR / "synthetic" / sim_name / f"{sim_name}.xes.gz"


def _upsert_dictionary_entry(
    dictionary_path: Path,
    key: str,
    entry: Mapping[str, str],
) -> None:
    with open(dictionary_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data[key] = dict(entry)
    tmp_path = dictionary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(dictionary_path)


def generate_for_dataset(
    dataset_name: str,
    data_dictionary: Mapping[str, Mapping[str, object]],
    *,
    rng: random.Random,
    force: bool,
) -> Path:
    if dataset_name not in data_dictionary:
        raise SystemExit(
            f"Error: dataset {dataset_name!r} not found in data dictionary."
        )

    out_path = sim_log_path(dataset_name)
    sim_name = f"{dataset_name}{SIM_SUFFIX}"
    source_name = str(data_dictionary[dataset_name].get("name", dataset_name))
    relative_path = out_path.as_posix()
    entry = {
        "name": f"{source_name} (DFG simulated)",
        "short_name": sim_name,
        "type": "synthetic",
        "path": relative_path,
    }

    if out_path.exists() and not force:
        print(f"Skipping {dataset_name}: {out_path} exists (use --force to overwrite)")
        _upsert_dictionary_entry(DATA_DICTIONARY_PATH, sim_name, entry)
        return out_path

    log_path = Path(str(data_dictionary[dataset_name]["path"]))
    print(f"Loading {dataset_name} from {log_path}")
    event_log = get_event_log_from_path(log_path)
    sequences = _source_sequences(event_log)
    if not sequences:
        raise SystemExit(f"Error: {dataset_name} has no non-empty traces.")

    max_trace_length = max(len(seq) for seq in sequences)
    cutoff = CUTOFF_MULTIPLIER * max_trace_length
    n_traces = len(event_log)
    next_map = build_next_activities_map(sequences)
    if START not in next_map:
        raise SystemExit(f"Error: {dataset_name} produced an empty DFG.")

    print(
        f"  traces={n_traces} max_len={max_trace_length} "
        f"cutoff={cutoff} states={len(next_map)}"
    )
    sampled = [sample_trace(next_map, rng, cutoff) for _ in range(n_traces)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xes_exporter.apply(traces_to_event_log(sampled), str(out_path))
    _upsert_dictionary_entry(DATA_DICTIONARY_PATH, sim_name, entry)
    print(f"  wrote {out_path} and registered {sim_name}")
    return out_path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate first-order DFG logs for n-grams analysis "
            "(writes data/synthetic/<LOG>_sim/ and updates the data dictionary)"
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[ALL_REAL_LOGS_TOKEN],
        help=(
            "Source dataset names (default: ALL_REAL_LOGS). "
            "Simulated logs are named <LOG>_sim."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed used independently for each source log (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing simulated .xes.gz",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = _resolve_datasets(args.datasets)
    data_dictionary = get_data_dictionary(
        DATA_DICTIONARY_PATH, get_real=True, get_synthetic=True
    )
    unknown = [name for name in datasets if name not in data_dictionary]
    if unknown:
        raise SystemExit(
            "Error: unknown dataset(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(sorted(data_dictionary))
        )

    print(f"Generating DFG-simulated logs for {len(datasets)} dataset(s)...")
    for dataset_name in datasets:
        rng = random.Random(args.random_seed)
        generate_for_dataset(
            dataset_name, data_dictionary, rng=rng, force=args.force
        )
    print("Done.")


if __name__ == "__main__":
    main()
