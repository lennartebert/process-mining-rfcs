"""Build consolidated log-info tables for all datasets."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import pm4py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import DATA_DICTIONARY_PATH, LOG_INFO_DIR
from utils.io import get_data_dictionary, get_event_log_from_path, variant_and_activity_counts
from utils.parsing import parse_count


LOG_INFO_COLUMNS: List[str] = [
    "Log",
    "Description",
    "# Cases",
    "# Events",
    "# Variants",
    "# Activities",
]

COUNT_COLUMNS = LOG_INFO_COLUMNS[2:]


def _format_count_for_latex(value: object) -> str:
    """Format numeric counts for LaTeX tables."""
    parsed = parse_count(value)
    if pd.isna(parsed):
        return ""
    return format(int(round(parsed)), ",")


def _write_latex_from_csv(details_df: pd.DataFrame, tex_path: Path) -> None:
    """Write LaTeX output from an existing details DataFrame."""
    latex_df = details_df.copy()
    for col in COUNT_COLUMNS:
        latex_df[col] = latex_df[col].map(_format_count_for_latex)
    latex_df = latex_df.rename(
        columns={
            "# Cases": r"\# Cases",
            "# Events": r"\# Events",
            "# Variants": r"\# Variants",
            "# Activities": r"\# Activities",
        }
    )
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
    for col in COUNT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(parse_count)
    by_log: Dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row["Log"])
        by_log[name] = row.to_dict()
    return by_log


def _compute_row_for_dataset(
    dataset_name: str,
    dataset_info: dict,
) -> dict:
    """Compute one log-info row with numeric counts for CSV output."""
    log_path = Path(dataset_info["path"])
    description = str(dataset_info.get("description", ""))

    if not log_path.exists():
        print(f"Warning: log file missing for {dataset_name}: {log_path}")
        return {
            "Log": dataset_name,
            "Description": description,
            "# Cases": pd.NA,
            "# Events": pd.NA,
            "# Variants": pd.NA,
            "# Activities": pd.NA,
        }

    event_log = get_event_log_from_path(log_path)
    log_df = pm4py.convert_to_dataframe(event_log)
    variant_count, activity_count = variant_and_activity_counts(event_log)
    row = {
        "Log": dataset_name,
        "Description": description,
        "# Cases": int(len(event_log)),
        "# Events": int(len(log_df)),
        "# Variants": int(variant_count),
        "# Activities": int(activity_count),
    }
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
        "--output-dir",
        type=str,
        default=str(LOG_INFO_DIR),
        help="Output directory for log-info tables (default: results/log_info)",
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

        if dataset_name in cached_by_log and not args.force_recalculate:
            out_rows.append(cached_by_log[dataset_name])
            continue

        row = _compute_row_for_dataset(dataset_name, dataset_info)
        cached_by_log[dataset_name] = row
        out_rows.append(row)

        checkpoint_rows = _rows_for_checkpoint(sorted_names, out_rows, cached_by_log)
        prefix_df = pd.DataFrame(checkpoint_rows, columns=LOG_INFO_COLUMNS)
        prefix_df.to_csv(csv_path, index=False)
        print(
            f"Checkpoint saved ({len(out_rows)}/{len(sorted_names)} logs complete, "
            f"{len(checkpoint_rows)} rows in file): {csv_path}"
        )

    details_df = pd.DataFrame(out_rows, columns=LOG_INFO_COLUMNS)
    _write_latex_from_csv(details_df, tex_path)


if __name__ == "__main__":
    main()
