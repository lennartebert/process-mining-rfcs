"""Ordered runner for the n-gram analysis pipeline.

Default (serial): ``01 -> 02 -> 03 -> 05`` (describe, extract, Clauset, compose).
Step 04 (combine) is skipped because there are no parallel shards.

``--parallel`` (one dataset): ``01 -> 02 -> 03`` only, writing per-log shards.
After all array tasks finish, run ``--combine`` (or ``.slurm/n_grams_combine.sh``)
to execute ``04 -> 05`` (merge shards, then compose).

``--test`` runs TEST_BPIC12 with concepts variants, n1, n2 and fewer bootstraps.
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
    ("04", "04_combine_parallel.py"),
    ("05", "05_compose_results.py"),
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
        raise SystemExit(
            "Error: --datasets is required unless --test or --combine is set."
        )
    if len(raw) == 1 and raw[0] == ALL_REAL_LOGS_TOKEN:
        return list(ALL_REAL_LOG_DATASETS)
    return list(raw)


def _selected_steps(args: argparse.Namespace) -> list[tuple[str, str]]:
    ids = [step_id for step_id, _ in STEPS]
    if args.combine:
        if args.only or args.from_step or args.to_step:
            raise SystemExit(
                "Error: --combine cannot be combined with --only/--from/--to."
            )
        if args.parallel:
            raise SystemExit("Error: --combine cannot be used with --parallel.")
        return [(sid, name) for sid, name in STEPS if sid in {"04", "05"}]

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = wanted - set(ids)
        if unknown:
            raise SystemExit(f"Unknown --only step(s): {sorted(unknown)}")
        return [(sid, name) for sid, name in STEPS if sid in wanted]

    if args.from_step is None and args.to_step is None:
        if args.parallel:
            return [(sid, name) for sid, name in STEPS if sid in {"01", "02", "03"}]
        # Serial default: skip combine (no shards).
        return [(sid, name) for sid, name in STEPS if sid != "04"]

    start = args.from_step or ids[0]
    end = args.to_step or ids[-1]
    if start not in ids or end not in ids:
        raise SystemExit(f"--from/--to must be one of {ids}")
    i0, i1 = ids.index(start), ids.index(end)
    if i0 > i1:
        raise SystemExit("--from must be <= --to")
    selected = STEPS[i0 : i1 + 1]
    if args.parallel:
        selected = [(sid, name) for sid, name in selected if sid in {"01", "02", "03"}]
    return selected


def _results_dir(args: argparse.Namespace) -> str | None:
    if args.output_dir:
        return args.output_dir
    if args.test:
        return str(TEST_N_GRAMS_DIR)
    return None


def _argv_for_step(
    step_id: str,
    *,
    datasets: List[str] | None,
    concepts: List[str],
    test_mode: bool,
    parallel: bool,
    n_bootstraps: int | None,
    random_seed: int | None,
    results_dir: str | None,
    passthrough: List[str],
) -> List[str]:
    """Build argv for one step."""
    argv: List[str] = []

    if step_id == "04":
        if results_dir:
            argv.extend(["--output-dir", results_dir])
        if datasets is not None:
            argv.extend(["--datasets", *datasets])
        argv.extend(["--concepts", *concepts])
        argv.extend(passthrough)
        return argv

    if step_id == "05":
        # Datasets optional: omit to compose all logs present in summaries.
        if datasets is not None:
            argv.extend(["--datasets", *datasets])
        argv.extend(["--concepts", *concepts])
        if results_dir:
            argv.extend(["--output-dir", results_dir])
        argv.extend(passthrough)
        return argv

    if datasets is None:
        raise SystemExit(f"Error: --datasets is required for step {step_id}.")
    argv.extend(["--datasets", *datasets])

    if step_id in {"02", "03"}:
        argv.extend(["--concepts", *concepts])

    if step_id == "02" and test_mode:
        argv.extend(["--output-dir", str(TEST_ATTACHMENTS_DIR)])

    if step_id in {"01", "03"} and results_dir:
        argv.extend(["--output-dir", results_dir])

    if step_id == "03":
        if test_mode:
            argv.extend(
                [
                    "--attachments-dir",
                    str(TEST_ATTACHMENTS_DIR),
                    "--n-bootstraps",
                    str(TEST_N_BOOTSTRAPS if n_bootstraps is None else n_bootstraps),
                ]
            )
        elif n_bootstraps is not None:
            argv.extend(["--n-bootstraps", str(n_bootstraps)])
        if random_seed is not None:
            argv.extend(["--random-seed", str(random_seed)])

    if parallel and step_id in {"01", "03"}:
        argv.append("--parallel")

    argv.extend(passthrough)
    return argv


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the n-gram analysis pipeline "
            "(serial 01->02->03->05; parallel 01->02->03; --combine 04->05)"
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names (ignored when --test is set; optional with --combine)",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=None,
        choices=CONCEPT_CHOICES,
        help=(
            "Concepts for steps 02-05 (default: n1..n10 + variants; "
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
        help="First step id to run (01..05)",
    )
    parser.add_argument(
        "--to",
        dest="to_step",
        default=None,
        help="Last step id to run (01..05)",
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
            "Per-log CSV shards for steps 01+03 (Slurm array). "
            "Requires exactly one dataset. Runs 01->02->03 only; "
            "afterwards use --combine."
        ),
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "After parallel shards exist: run step 04 (merge log_info + Clauset) "
            "then step 05 (compose tables/plots)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="N-grams results root (default: results/n_grams; with --test: results/test/n_grams)",
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
    if args.test:
        datasets: List[str] | None = [TEST_DATASET]
    elif args.combine and args.datasets is None:
        datasets = None
    else:
        datasets = _resolve_datasets(args.datasets)

    if args.parallel and (datasets is None or len(datasets) != 1):
        raise SystemExit(
            "Error: --parallel requires exactly one dataset "
            f"(got {datasets!r})."
        )
    if args.concepts is not None:
        concepts = list(args.concepts)
    elif args.test:
        concepts = list(TEST_CONCEPTS)
    else:
        concepts = list(DEFAULT_CONCEPTS)

    results_dir = _results_dir(args)
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
            results_dir=results_dir,
            passthrough=args.passthrough,
        )
        print(f"$ python analyses/n_grams/{filename} {' '.join(step_argv)}")
        module.main(step_argv)


if __name__ == "__main__":
    main()
