"""Step 04: combine per-log parallel CSV shards from describe + Clauset.

Merges shards produced by ``main.py --parallel`` (steps 01 and 03) under a
power-law statistics results root:

- ``log_info_<LOG>.csv`` → ``log_info.csv`` + ``.tex``
- ``<concept>/<model>/{gof,comparison,summary}_<LOG>.csv`` → unsuffixed CSVs

Optional ``--datasets`` keeps only shards for the listed logs.

Does not compose tables/plots; run ``05_compose_results.py`` afterwards
(``main.py --combine`` runs 04 then 05).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.log_info import IDENTITY_COLUMNS, _write_latex_from_csv
from utils.constants import POWERLAW_STATISTICS_REAL_DIR
from utils.io.attachments import NGRAM_CONCEPTS
from utils.powerlaw import DISTRIBUTION_NAMES

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]
CLASET_STEMS = ("gof", "comparison", "summary")


def _filter_shards_by_log(
    matches: Sequence[Path],
    *,
    prefix: str,
    datasets: Sequence[str] | None,
) -> list[Path]:
    """Keep shards named ``{prefix}<LOG>.csv`` when ``datasets`` is set."""
    if datasets is None:
        return list(matches)
    allowed = set(datasets)
    kept: list[Path] = []
    for path in matches:
        name = path.name
        if not (name.startswith(prefix) and name.endswith(".csv")):
            continue
        log_name = name[len(prefix) : -len(".csv")]
        if log_name in allowed:
            kept.append(path)
    return kept


def _concat_sorted_csvs(
    paths: Sequence[Path],
    *,
    sort_cols: Sequence[str],
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            print(f"Warning: skipping empty shard {path}")
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    present = [c for c in sort_cols if c in combined.columns]
    if present:
        combined = combined.sort_values(list(present)).reset_index(drop=True)
    return combined


def _combine_glob(
    directory: Path,
    pattern: str,
    out_path: Path,
    *,
    sort_cols: Sequence[str],
    prefix: str,
    datasets: Sequence[str] | None,
) -> bool:
    matches = _filter_shards_by_log(
        directory.glob(pattern), prefix=prefix, datasets=datasets
    )
    if not matches:
        return False
    combined = _concat_sorted_csvs(matches, sort_cols=sort_cols)
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
    matches = _filter_shards_by_log(
        output_dir.glob("log_info_*.csv"),
        prefix="log_info_",
        datasets=datasets,
    )
    if not matches:
        print(f"Warning: no log_info_*.csv under {output_dir}")
        return
    combined = _concat_sorted_csvs(matches, sort_cols=["Log"])
    if combined is None:
        print("Warning: log_info shards were empty")
        return
    csv_path = output_dir / "log_info.csv"
    tex_path = output_dir / "log_info.tex"
    combined.to_csv(csv_path, index=False)
    print(f"Combined {len(matches)} shard(s) -> {csv_path}")
    stats = [c for c in combined.columns if c not in IDENTITY_COLUMNS]
    _write_latex_from_csv(combined, tex_path, stats)


def combine_clauset(
    output_dir: Path,
    concepts: Sequence[str],
    *,
    datasets: Sequence[str] | None = None,
) -> None:
    for concept in concepts:
        concept_dir = output_dir / concept
        if not concept_dir.is_dir():
            continue
        for model_dir in sorted(p for p in concept_dir.iterdir() if p.is_dir()):
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
        description="Combine power-law statistics parallel shards (log_info + Clauset) into aggregates"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(POWERLAW_STATISTICS_REAL_DIR),
        help=f"Power-law statistics results root containing shards (default: {POWERLAW_STATISTICS_REAL_DIR})",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=DEFAULT_CONCEPTS,
        choices=CONCEPT_CHOICES,
        help="Concepts whose Clauset shards to merge (default: n1..n10 + variants)",
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
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"Error: output directory not found: {output_dir}")

    datasets = list(args.datasets) if args.datasets else None
    print(f"Combining parallel shards under: {output_dir}")
    if datasets is not None:
        print(f"Dataset filter: {datasets}")
    combine_log_info(output_dir, datasets=datasets)
    combine_clauset(output_dir, args.concepts, datasets=datasets)
    print(f"Combine complete: {output_dir}")


if __name__ == "__main__":
    main()
