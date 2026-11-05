"""Generate iid "monkey typing" synthetic event logs for Clauset analysis.

Each trace has a fixed length; each event is drawn independently and uniformly
from a finite activity alphabet. Writes
``data/synthetic/<NAME>/<NAME>.xes.gz`` and upserts ``data/data_dictionary.json``.

Not wired into ``main.py``. Run directly:

    python analyses/powerlaw_statistics/07_generate_synthetic_monkey.py \\
        --n-activities 24 --n-traces 20000 --trace-length 20
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Mapping, Sequence

from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import Event, EventLog, Trace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import DATA_DICTIONARY_PATH, DATA_DIR


def activity_alphabet(n_activities: int) -> list[str]:
    if n_activities < 1:
        raise ValueError("n_activities must be >= 1")
    if n_activities <= len(string.ascii_lowercase):
        return list(string.ascii_lowercase[:n_activities])
    return [f"a{i}" for i in range(n_activities)]


def monkey_dataset_name(
    n_activities: int,
    n_traces: int,
    trace_length: int,
) -> str:
    return f"MONKEY_A{n_activities}_N{n_traces}_L{trace_length}"


def monkey_log_path(dataset_name: str) -> Path:
    return DATA_DIR / "synthetic" / dataset_name / f"{dataset_name}.xes.gz"


def sample_traces(
    activities: Sequence[str],
    *,
    n_traces: int,
    trace_length: int,
    rng: random.Random,
) -> list[list[str]]:
    return [
        [rng.choice(activities) for _ in range(trace_length)]
        for _ in range(n_traces)
    ]


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


def generate_monkey_log(
    *,
    n_activities: int,
    n_traces: int,
    trace_length: int,
    rng: random.Random,
    force: bool,
    dataset_name: str | None = None,
) -> Path:
    if n_traces < 1:
        raise SystemExit("Error: --n-traces must be >= 1")
    if trace_length < 1:
        raise SystemExit("Error: --trace-length must be >= 1")

    name = dataset_name or monkey_dataset_name(
        n_activities, n_traces, trace_length
    )
    out_path = monkey_log_path(name)
    relative_path = out_path.as_posix()
    entry = {
        "name": (
            f"Monkey model (iid; {n_activities} activities, "
            f"{n_traces} traces, length {trace_length})"
        ),
        "short_name": name,
        "type": "synthetic",
        "path": relative_path,
    }

    if out_path.exists() and not force:
        print(f"Skipping: {out_path} exists (use --force to overwrite)")
        _upsert_dictionary_entry(DATA_DICTIONARY_PATH, name, entry)
        return out_path

    activities = activity_alphabet(n_activities)
    print(
        f"Sampling monkey log {name}: "
        f"activities={n_activities} traces={n_traces} length={trace_length}"
    )
    sampled = sample_traces(
        activities,
        n_traces=n_traces,
        trace_length=trace_length,
        rng=rng,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xes_exporter.apply(traces_to_event_log(sampled), str(out_path))
    _upsert_dictionary_entry(DATA_DICTIONARY_PATH, name, entry)
    print(f"  wrote {out_path} and registered {name}")
    return out_path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an iid monkey-typing synthetic XES log "
            "(writes data/synthetic/MONKEY_*/ and updates the data dictionary)"
        )
    )
    parser.add_argument(
        "--n-activities",
        type=int,
        required=True,
        help="Alphabet size (a.. for <=26; a0,a1,... otherwise)",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        required=True,
        help="Number of traces to sample",
    )
    parser.add_argument(
        "--trace-length",
        type=int,
        required=True,
        help="Fixed number of events per trace",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Dataset / folder name "
            "(default: MONKEY_A{n}_N{traces}_L{length})"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing .xes.gz",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    generate_monkey_log(
        n_activities=args.n_activities,
        n_traces=args.n_traces,
        trace_length=args.trace_length,
        rng=random.Random(args.random_seed),
        force=args.force,
        dataset_name=args.name,
    )
    print("Done.")


if __name__ == "__main__":
    main()
