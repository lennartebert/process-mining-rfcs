"""Combine per-log parallel CSV shards into aggregate tables (+ LaTeX).

Reads shards produced by ``main.py --parallel`` under an n-grams results root
and writes the usual unsuffixed aggregates. Only this script emits LaTeX for
``log_info``, ``variant_power_law``, and ``log_n_scaling``.

Does not touch attachments or plots.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.log_info import IDENTITY_COLUMNS, _write_latex_from_csv
from utils.constants import N_GRAMS_DIR
from utils.io.attachments import NGRAM_CONCEPTS
from utils.powerlaw import DISTRIBUTION_NAMES

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]
CLASET_STEMS = ("gof", "comparison", "summary")


def _load_compose_module():
    path = Path(__file__).resolve().parent / "04_compose_results.py"
    spec = importlib.util.spec_from_file_location("n_grams_compose_results", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
) -> bool:
    matches = list(directory.glob(pattern))
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


def combine_log_info(output_dir: Path) -> None:
    matches = list(output_dir.glob("log_info_*.csv"))
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


def combine_clauset(output_dir: Path, concepts: Sequence[str]) -> None:
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
                )


def combine_compose_tables(output_dir: Path, compose_mod) -> None:
    variant_matches = list(output_dir.glob("variant_power_law_*.csv"))
    if variant_matches:
        combined = _concat_sorted_csvs(variant_matches, sort_cols=["Log"])
        if combined is not None:
            compose_mod.write_table_csv_and_tex(
                combined,
                output_dir / "variant_power_law.csv",
                output_dir / "variant_power_law.tex",
                caption="Trace variant power-law results",
            )
    else:
        print(f"Warning: no variant_power_law_*.csv under {output_dir}")

    fitted_matches = list(output_dir.glob("log_n_fitted_types_*.csv"))
    fitted_combined = None
    if fitted_matches:
        fitted_combined = _concat_sorted_csvs(fitted_matches, sort_cols=["Log"])
        if fitted_combined is not None:
            compose_mod.write_table_csv(
                fitted_combined,
                output_dir / "log_n_fitted_types.csv",
            )
    else:
        print(f"Warning: no log_n_fitted_types_*.csv under {output_dir}")

    scaling_matches = list(output_dir.glob("log_n_scaling_*.csv"))
    if scaling_matches:
        scaling_combined = _concat_sorted_csvs(scaling_matches, sort_cols=["Log"])
        if scaling_combined is not None:
            if fitted_combined is None:
                fitted_combined = pd.DataFrame(columns=scaling_combined.columns)
            compose_mod.write_table_csv_and_tex(
                scaling_combined,
                output_dir / "log_n_scaling.csv",
                output_dir / "log_n_scaling.tex",
                caption="Log x n power-law scaling",
                legend=compose_mod.SCALING_LEGEND,
                latex_body=compose_mod.scaling_dataframe_to_latex(
                    scaling_combined, fitted_combined
                ),
                footnote=compose_mod.SCALING_GRAY_FOOTNOTE,
            )
    else:
        print(f"Warning: no log_n_scaling_*.csv under {output_dir}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine n-grams --parallel CSV shards into aggregates + LaTeX"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(N_GRAMS_DIR),
        help=f"N-grams results root containing shards (default: {N_GRAMS_DIR})",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=DEFAULT_CONCEPTS,
        choices=CONCEPT_CHOICES,
        help="Concepts whose Clauset shards to merge (default: n1..n10 + variants)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"Error: output directory not found: {output_dir}")

    compose_mod = _load_compose_module()
    print(f"Combining parallel shards under: {output_dir}")
    combine_log_info(output_dir)
    combine_clauset(output_dir, args.concepts)
    combine_compose_tables(output_dir, compose_mod)
    print(f"Combine complete: {output_dir}")


if __name__ == "__main__":
    main()
