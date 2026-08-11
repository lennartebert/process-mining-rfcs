"""Build consolidated log-info tables for all datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pm4py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import DATA_DICTIONARY_PATH, LOG_INFO_DIR
from utils.io import get_data_dictionary, get_event_log_from_path, variant_and_activity_counts
from utils.parsing import parse_count


IDENTITY_COLUMNS: List[str] = ["Log", "Description"]

DEFAULT_STATS: List[str] = [
    "# Cases",
    "# Events",
    "# Variants",
    "# Activities",
]

ALL_STATS: List[str] = [
    *DEFAULT_STATS,
    "Median Trace Length",
    "Max Trace Length",
]

INTEGER_STATS = frozenset(
    {
        "# Cases",
        "# Events",
        "# Variants",
        "# Activities",
        "Median Trace Length",
        "Max Trace Length",
    }
)

LATEX_COLUMN_RENAMES = {
    "# Cases": r"\# Cases",
    "# Events": r"\# Events",
    "# Variants": r"\# Variants",
    "# Activities": r"\# Activities",
    "Median Trace Length": r"Median Trace Length",
    "Max Trace Length": r"Max Trace Length",
}


def _output_columns(stats: Sequence[str]) -> List[str]:
    return [*IDENTITY_COLUMNS, *stats]


def _format_stat_for_latex(value: object, *, integer_like: bool) -> str:
    """Format a numeric stat for LaTeX tables."""
    parsed = parse_count(value)
    if pd.isna(parsed):
        return ""
    if integer_like:
        return format(int(round(parsed)), ",")
    text = f"{parsed:.1f}"
    if text.endswith(".0"):
        return format(int(parsed), ",")
    return text


def _coerce_integer_stats(df: pd.DataFrame, stats: Sequence[str]) -> pd.DataFrame:
    """Ensure integer metric columns use nullable Int64 (CSV writes without .0)."""
    out = df.copy()
    for col in stats:
        if col not in INTEGER_STATS or col not in out.columns:
            continue
        out[col] = (
            pd.to_numeric(out[col], errors="coerce").round().astype("Int64")
        )
    return out


def _write_latex_from_csv(
    details_df: pd.DataFrame, tex_path: Path, stats: Sequence[str]
) -> None:
    """Write LaTeX output from an existing details DataFrame."""
    latex_df = details_df.copy()
    for col in stats:
        latex_df[col] = latex_df[col].map(
            lambda value, c=col: _format_stat_for_latex(
                value, integer_like=c in INTEGER_STATS
            )
        )
    rename = {col: LATEX_COLUMN_RENAMES[col] for col in stats if col in LATEX_COLUMN_RENAMES}
    latex_df = latex_df.rename(columns=rename)
    tex_path.write_text(latex_df.to_latex(index=False, escape=False), encoding="utf-8")
    print(f"Saved: {tex_path}")


def _rows_for_checkpoint(
    sorted_names: Sequence[str],
    prefix_rows: List[dict],
    cached_by_log: Dict[str, dict],
) -> List[dict]:
    """Prefix rows plus any already-known rows for later logs (survives partial overwrites)."""
    rows = list(prefix_rows)
    for name in sorted_names[len(prefix_rows) :]:
        if name in cached_by_log:
            rows.append(cached_by_log[name])
    return rows


def _load_csv_by_log(csv_path: Path) -> Dict[str, dict]:
    """Load existing CSV rows keyed by Log name."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    for col in ALL_STATS:
        if col in df.columns:
            df[col] = df[col].map(parse_count)
    by_log: Dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["Log"])
        by_log[name] = row.to_dict()
    return by_log


def _cache_row_has_stats(row: dict, stats: Sequence[str]) -> bool:
    """True when the cached row contains every requested stat column."""
    return all(stat in row for stat in stats)


def _trace_lengths(event_log) -> List[int]:
    return [len(trace) for trace in event_log]


def _compute_row_for_dataset(
    dataset_name: str,
    dataset_info: dict,
    stats: Sequence[str],
) -> dict:
    """Compute one log-info row for the requested stats."""
    log_path = Path(dataset_info["path"])
    description = str(dataset_info.get("description", ""))
    row: dict = {
        "Log": dataset_name,
        "Description": description,
    }
    for stat in stats:
        row[stat] = pd.NA

    if not log_path.exists():
        print(f"Warning: log file missing for {dataset_name}: {log_path}")
        return row

    event_log = get_event_log_from_path(log_path)
    need_variants_or_activities = (
        "# Variants" in stats or "# Activities" in stats
    )
    need_events = "# Events" in stats
    need_trace_lengths = (
        "Median Trace Length" in stats or "Max Trace Length" in stats
    )

    if "# Cases" in stats:
        row["# Cases"] = int(len(event_log))

    if need_events:
        log_df = pm4py.convert_to_dataframe(event_log)
        row["# Events"] = int(len(log_df))

    if need_variants_or_activities:
        variant_count, activity_count = variant_and_activity_counts(event_log)
        if "# Variants" in stats:
            row["# Variants"] = int(variant_count)
        if "# Activities" in stats:
            row["# Activities"] = int(activity_count)

    if need_trace_lengths:
        lengths = _trace_lengths(event_log)
        if lengths:
            if "Median Trace Length" in stats:
                row["Median Trace Length"] = int(round(float(np.median(lengths))))
            if "Max Trace Length" in stats:
                row["Max Trace Length"] = int(max(lengths))
        # else leave as pd.NA

    return row



def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for log-info generation."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated log-info outputs (CSV + LaTeX)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names from data dictionary",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=list(DEFAULT_STATS),
        choices=ALL_STATS,
        help=(
            "Log stats to extract (default: "
            + ", ".join(f"'{s}'" for s in DEFAULT_STATS)
            + "). Optional: 'Median Trace Length', 'Max Trace Length'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(LOG_INFO_DIR),
        help=f"Output directory for log_info.csv/.tex (default: {LOG_INFO_DIR})",
    )
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Ignore cached CSV rows and recompute all metrics from event logs",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Generate log-info outputs with incremental CSV checkpoints and LaTeX."""
    args = parse_args(argv)
    stats: List[str] = list(args.stats)
    columns = _output_columns(stats)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "log_info.csv"
    tex_path = output_dir / "log_info.tex"

    cached_by_log = {} if args.force_recalculate else _load_csv_by_log(csv_path)

    data_dictionary = get_data_dictionary(
        DATA_DICTIONARY_PATH, get_real=True, get_synthetic=True
    )
    invalid_datasets: List[str] = [ds for ds in args.datasets if ds not in data_dictionary]
    if invalid_datasets:
        print(f"Error: Invalid dataset names: {invalid_datasets}")
        print(f"Available datasets: {sorted(data_dictionary.keys())}")
        sys.exit(1)
    sorted_names = sorted(args.datasets)

    out_rows: List[dict] = []
    for dataset_name in sorted_names:
        dataset_info = data_dictionary[dataset_name]

        cached = cached_by_log.get(dataset_name)
        if (
            cached is not None
            and not args.force_recalculate
            and _cache_row_has_stats(cached, stats)
        ):
            # Keep only identity + requested stats in the written table.
            out_rows.append({col: cached.get(col, pd.NA) for col in columns})
            continue

        row = _compute_row_for_dataset(dataset_name, dataset_info, stats)
        cached_by_log[dataset_name] = row
        out_rows.append(row)

        checkpoint_rows = _rows_for_checkpoint(sorted_names, out_rows, cached_by_log)
        # Restrict checkpoint rows to the requested columns when possible.
        checkpoint_normalized = [
            {col: r.get(col, pd.NA) for col in columns} for r in checkpoint_rows
        ]
        prefix_df = _coerce_integer_stats(
            pd.DataFrame(checkpoint_normalized, columns=columns), stats
        )
        prefix_df.to_csv(csv_path, index=False)
        print(
            f"Checkpoint saved ({len(out_rows)}/{len(sorted_names)} logs complete, "
            f"{len(checkpoint_normalized)} rows in file): {csv_path}"
        )

    details_df = _coerce_integer_stats(
        pd.DataFrame(out_rows, columns=columns), stats
    )
    # Ensure a final CSV write even when everything came from cache.
    details_df.to_csv(csv_path, index=False)
    _write_latex_from_csv(details_df, tex_path, stats)


if __name__ == "__main__":
    main()
