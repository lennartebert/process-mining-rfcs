"""Step 04: combine per-log parallel CSV shards from describe + Clauset.

Merges shards produced by ``main.py --parallel`` (steps 01 and 03):

- ``<experiment>/log_info/log_info_<LOG>.csv`` → ``log_info.csv`` + ``.tex``
- ``<experiment>/<xmin>/raw_results/<concept>/<model>/{gof,comparison,summary}_<LOG>.csv``
  → unsuffixed CSVs

Optional ``--datasets`` keeps only shards for the listed logs.

Does not run saturation, paired GOF, or compose. After shards are merged,
``main.py --combine`` continues with step 05 (cross-log) then step 06 (compose).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.log_info import IDENTITY_COLUMNS, _write_latex_from_csv
from utils.constants import PL_STATS_REAL_DIR
from utils.csv_shards import concat_sorted_csvs, filter_shards_by_log
from utils.io.attachments import NGRAM_CONCEPTS
from analyses.powerlaw_statistics.analysis_helpers.paths import concept_dirname, log_info_dir, raw_results_dir
from utils.powerlaw import DISTRIBUTION_NAMES

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]
CLASET_STEMS = ("gof", "comparison", "summary")


def _combine_glob(
    directory: Path,
    pattern: str,
    out_path: Path,
    *,
    sort_cols: Sequence[str],
    prefix: str,
    datasets: Sequence[str] | None,
) -> bool:
    matches = filter_shards_by_log(
        list(directory.glob(pattern)), prefix=prefix, datasets=datasets
    )
    if not matches:
        return False
    combined = concat_sorted_csvs(matches, sort_cols=sort_cols)
    if combined is None:
        print(f"Warning: no rows in shards matching {directory / pattern}")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Combined {len(matches)} shard(s) -> {out_path}")
    return True


def combine_log_info(
    output_dir: Path, *, datasets: Sequence[str] | None = None
) -> None:
    matches = filter_shards_by_log(
        list(output_dir.glob("log_info_*.csv")),
        prefix="log_info_",
        datasets=datasets,
    )
    if not matches:
        print(f"Warning: no log_info_*.csv under {output_dir}")
        return
    combined = concat_sorted_csvs(matches, sort_cols=["Log"])
    if combined is None:
        print("Warning: log_info shards were empty")
        return
    csv_path = output_dir / "log_info.csv"
    tex_path = output_dir / "log_info.tex"
    combined.to_csv(csv_path, index=False)
    print(f"Combined {len(matches)} shard(s) -> {csv_path}")
    stats = [column for column in combined.columns if column not in IDENTITY_COLUMNS]
    _write_latex_from_csv(combined, tex_path, stats)


def combine_clauset(
    output_dir: Path,
    concepts: Sequence[str],
    *,
    datasets: Sequence[str] | None = None,
) -> None:
    for concept in concepts:
        concept_dir = output_dir / concept_dirname(concept)
        if not concept_dir.is_dir():
            continue
        for model_dir in sorted(path for path in concept_dir.iterdir() if path.is_dir()):
            if model_dir.name not in DISTRIBUTION_NAMES:
                continue
            for stem in CLASET_STEMS:
                sort_cols = ["log_name"]
                if stem == "comparison":
                    sort_cols = ["log_name", "model_2"]
                elif stem == "gof":
                    sort_cols = ["log_name", "doubly_bounded_exclude_head_variants"]
                _combine_glob(
                    model_dir,
                    f"{stem}_*.csv",
                    model_dir / f"{stem}.csv",
                    sort_cols=sort_cols,
                    prefix=f"{stem}_",
                    datasets=datasets,
                )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-log power-law statistics CSV shards"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PL_STATS_REAL_DIR),
        help=f"Experiment directory containing shards (default: {PL_STATS_REAL_DIR})",
    )
    parser.add_argument(
        "--xmin",
        type=int,
        default=None,
        help="xmin folder to combine (default: xmin-adaptive)",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=DEFAULT_CONCEPTS,
        choices=CONCEPT_CHOICES,
        help="Concepts whose Clauset shards to merge (default: n1..n20 + variants)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional log filter: only merge shards for these datasets",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_dir = Path(args.output_dir)
    if not experiment_dir.is_dir():
        raise SystemExit(f"Error: output directory not found: {experiment_dir}")

    datasets = list(args.datasets) if args.datasets else None
    info_dir = log_info_dir(experiment_dir)
    raw_dir = raw_results_dir(experiment_dir, args.xmin)
    print(f"Combining parallel shards under: {experiment_dir}")
    print(f"  log info: {info_dir}")
    print(f"  raw results: {raw_dir}")
    if datasets is not None:
        print(f"Dataset filter: {datasets}")
    combine_log_info(info_dir, datasets=datasets)
    combine_clauset(raw_dir, args.concepts, datasets=datasets)
    print(f"Combine complete: {experiment_dir}")


if __name__ == "__main__":
    main()
