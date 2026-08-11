"""Ordered runner for the n-gram analysis pipeline.

Runs numbered steps 01 -> 02 -> 03 by default. Use --only / --from / --to to
select a subset. Each step script remains runnable on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ALL_REAL_LOGS_TOKEN,
)

STEPS: list[tuple[str, str]] = [
    ("01", "01_describe.py"),
    ("02", "02_extract_ngrams.py"),
    ("03", "03_clauset_tests.py"),
]


def _load_step(module_filename: str):
    path = Path(__file__).resolve().parent / module_filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_datasets(raw: Sequence[str]) -> List[str]:
    if len(raw) == 1 and raw[0] == ALL_REAL_LOGS_TOKEN:
        return list(ALL_REAL_LOG_DATASETS)
    return list(raw)


def _selected_steps(args: argparse.Namespace) -> list[tuple[str, str]]:
    ids = [step_id for step_id, _ in STEPS]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - set(ids)
        if unknown:
            raise SystemExit(f"Unknown --only step(s): {sorted(unknown)}")
        return [(sid, name) for sid, name in STEPS if sid in wanted]

    start = args.from_step or ids[0]
    end = args.to_step or ids[-1]
    if start not in ids or end not in ids:
        raise SystemExit(f"--from/--to must be one of {ids}")
    i0, i1 = ids.index(start), ids.index(end)
    if i0 > i1:
        raise SystemExit("--from must be <= --to")
    return STEPS[i0 : i1 + 1]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the n-gram analysis pipeline (01->02->03)"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--from",
        dest="from_step",
        default=None,
        help="First step id to run (01, 02, or 03)",
    )
    parser.add_argument(
        "--to",
        dest="to_step",
        default=None,
        help="Last step id to run (01, 02, or 03)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated step ids to run (e.g. 01,03)",
    )
    parser.add_argument(
        "--passthrough",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to each selected step (after --)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    datasets = _resolve_datasets(args.datasets)
    selected = _selected_steps(args)
    base_argv = ["--datasets", *datasets, *args.passthrough]

    for step_id, filename in selected:
        module = _load_step(filename)
        print(f"$ python analyses/n_grams/{filename} {' '.join(base_argv)}")
        module.main(base_argv)


if __name__ == "__main__":
    main()
