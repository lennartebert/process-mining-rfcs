"""Ordered runner for the n-gram analysis pipeline.

Runs numbered steps 01 -> 02 -> 03 by default. Use --only / --from / --to to
select a subset. Each step script remains runnable on its own.

``--test`` runs TEST_BPIC12 with concepts variants, n1, n2 and fewer bootstraps.

``--parallel`` writes per-log CSV shards (no LaTeX) for Slurm array jobs; requires
exactly one dataset. Combine shards later with ``05_combine_parallel.py``.
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
    TEST_ATTACHMENTS_DIR,
    TEST_DATASET,
    TEST_N_GRAMS_DIR,
)
from utils.io.attachments import NGRAM_CONCEPTS

STEPS: list[tuple[str, str]] = [
    ("01", "01_describe.py"),
    ("02", "02_extract_ngrams.py"),
    ("03", "03_clauset_tests.py"),
    ("04", "04_compose_results.py"),
]

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
TEST_CONCEPTS = ["n1", "n2", "variants"]
TEST_N_BOOTSTRAPS = 50
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]


def _load_step(module_filename: str):
    path = Path(__file__).resolve().parent / module_filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_datasets(raw: Sequence[str] | None) -> List[str]:
    if raw is None:
        raise SystemExit("Error: --datasets is required unless --test is set.")
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


def _argv_for_step(
    step_id: str,
    *,
    datasets: List[str],
    concepts: List[str],
    test_mode: bool,
    parallel: bool,
    n_bootstraps: int | None,
    random_seed: int | None,
    passthrough: List[str],
) -> List[str]:
    """Build argv for one step (01 does not accept concepts/bootstraps)."""
    argv = ["--datasets", *datasets]
    if test_mode and step_id == "01":
        argv.extend(["--output-dir", str(TEST_N_GRAMS_DIR)])
    if step_id in {"02", "03", "04"}:
        argv.extend(["--concepts", *concepts])
    if test_mode and step_id == "02":
        argv.extend(["--output-dir", str(TEST_ATTACHMENTS_DIR)])
    if step_id == "03":
        if test_mode:
            argv.extend(
                [
                    "--attachments-dir",
                    str(TEST_ATTACHMENTS_DIR),
                    "--output-dir",
                    str(TEST_N_GRAMS_DIR),
                    "--n-bootstraps",
                    str(TEST_N_BOOTSTRAPS if n_bootstraps is None else n_bootstraps),
                ]
            )
        elif n_bootstraps is not None:
            argv.extend(["--n-bootstraps", str(n_bootstraps)])
        if random_seed is not None:
            argv.extend(["--random-seed", str(random_seed)])
    if test_mode and step_id == "04":
        argv.extend(["--output-dir", str(TEST_N_GRAMS_DIR)])
    # Step 02 attachments are already per-log; no --parallel needed there.
    if parallel and step_id in {"01", "03", "04"}:
        argv.append("--parallel")
    argv.extend(passthrough)
    return argv


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the n-gram analysis pipeline (01->02->03->04)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names (ignored when --test is set)",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        choices=CONCEPT_CHOICES,
        help=(
            "Concepts for steps 02-04 (default: n1..n10 + variants; "
            f"with --test default: {', '.join(TEST_CONCEPTS)})"
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            f"Quick smoke run: dataset {TEST_DATASET}, concepts "
            f"{', '.join(TEST_CONCEPTS)} unless --concepts is set, "
            f"attachments under {TEST_ATTACHMENTS_DIR}, "
            f"Clauset/compose under {TEST_N_GRAMS_DIR}, "
            f"and n-bootstraps={TEST_N_BOOTSTRAPS}"
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        default=None,
        help="First step id to run (01, 02, 03, or 04)",
    )
    parser.add_argument(
        "--to",
        dest="to_step",
        default=None,
        help="Last step id to run (01, 02, 03, or 04)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated step ids to run (e.g. 01,03)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Per-log CSV shards (no LaTeX) for Slurm array jobs. "
            "Requires exactly one dataset. Combine later with 05_combine_parallel.py."
        ),
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=None,
        help="Bootstrap iterations for step 03 (overrides --test default when set)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for step 03 bootstrap resampling",
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
    datasets = [TEST_DATASET] if args.test else _resolve_datasets(args.datasets)
    if args.parallel and len(datasets) != 1:
        raise SystemExit(
            "Error: --parallel requires exactly one dataset "
            f"(got {len(datasets)}: {datasets})."
        )
    if args.concepts is not None:
        concepts = list(args.concepts)
    elif args.test:
        concepts = list(TEST_CONCEPTS)
    else:
        concepts = list(DEFAULT_CONCEPTS)
    selected = _selected_steps(args)

    for step_id, filename in selected:
        module = _load_step(filename)
        step_argv = _argv_for_step(
            step_id,
            datasets=datasets,
            concepts=concepts,
            test_mode=args.test,
            parallel=args.parallel,
            n_bootstraps=args.n_bootstraps,
            random_seed=args.random_seed,
            passthrough=args.passthrough,
        )
        print(f"$ python analyses/n_grams/{filename} {' '.join(step_argv)}")
        module.main(step_argv)


if __name__ == "__main__":
    main()
