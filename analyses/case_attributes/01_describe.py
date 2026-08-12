"""Step 1: describe case attributes and write editable inventory CSV.

Outputs per log under ``<output-dir>/<log_name>/``:

- ``attribute_inventory.csv`` (descriptive stats + suggested/override columns)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ALL_REAL_LOGS_TOKEN,
    CASE_ATTRIBUTES_DIR,
    DATA_DICTIONARY_PATH,
)
from utils.io import get_data_dictionary, get_event_log_from_path
from utils.io.case_tables import (
    build_case_attribute_table,
    candidate_attribute_names,
)
from utils.case_attribute.config import (
    INVENTORY_FILENAME,
    build_inventory_row,
    load_csv,
    merge_inventory_dataframe,
    write_csv,
)
from utils.case_attribute.inventory import (
    RecommendationThresholds,
    describe_attribute,
    recommend_datatype,
)


def _resolve_datasets(raw: List[str]) -> List[str]:
    if len(raw) == 1 and raw[0] == ALL_REAL_LOGS_TOKEN:
        return list(ALL_REAL_LOG_DATASETS)
    return list(raw)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe case attributes and write attribute_inventory.csv"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--data-dictionary",
        type=str,
        default=str(DATA_DICTIONARY_PATH),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CASE_ATTRIBUTES_DIR),
    )
    parser.add_argument("--continuous-min-distinct", type=int, default=20)
    parser.add_argument("--default-n-bins", type=int, default=10)
    parser.add_argument("--near-constant-largest-share", type=float, default=0.98)
    parser.add_argument("--excessive-missing-share", type=float, default=0.50)
    return parser.parse_args(argv)


def describe_one_log(
    *,
    log_name: str,
    log_path: Path,
    output_dir: Path,
    thresholds: RecommendationThresholds,
) -> dict[str, Any]:
    event_log = get_event_log_from_path(log_path)
    case_df, source_levels = build_case_attribute_table(event_log)
    attr_names = candidate_attribute_names(case_df, source_levels)

    suggested_rows: list[dict[str, Any]] = []
    for attr_name in attr_names:
        source_level = source_levels.get(attr_name, "unknown")
        values = case_df[attr_name].tolist()
        inventory = describe_attribute(
            values,
            attribute_name=attr_name,
            source_level=source_level,
            thresholds=thresholds,
        )
        recommendation = recommend_datatype(inventory, thresholds=thresholds)
        suggested_rows.append(build_inventory_row(inventory, recommendation))

    log_dir = output_dir / log_name
    log_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = log_dir / INVENTORY_FILENAME
    old_df = load_csv(inventory_path)
    inventory_df = merge_inventory_dataframe(old_df, suggested_rows)
    write_csv(inventory_df, inventory_path)

    return {
        "log_name": log_name,
        "n_cases": int(len(case_df)),
        "n_attributes": len(attr_names),
        "inventory_path": inventory_path,
        "inventory_df": inventory_df,
    }


def _print_summary(inventory_df: pd.DataFrame) -> None:
    if inventory_df.empty:
        print("  (no attributes)")
        return
    cols = [
        "attribute_name",
        "include_suggested",
        "type_suggested",
        "n_distinct",
        "source_level",
    ]
    present = [c for c in cols if c in inventory_df.columns]
    summary = inventory_df[present].sort_values(
        ["include_suggested", "n_distinct"], ascending=[False, False]
    )
    print(summary.to_string(index=False))


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = RecommendationThresholds(
        continuous_min_distinct=args.continuous_min_distinct,
        default_n_bins=args.default_n_bins,
        near_constant_largest_share=args.near_constant_largest_share,
        excessive_missing_share=args.excessive_missing_share,
    )
    data_dictionary = get_data_dictionary(
        Path(args.data_dictionary), get_real=True, get_synthetic=True
    )
    datasets = _resolve_datasets(args.datasets)

    print(f"Describing case attributes for {len(datasets)} dataset(s)...")
    for log_name in datasets:
        print(f"\n=== {log_name} ===")
        if log_name not in data_dictionary:
            print("  Warning: dataset not in data dictionary; skipping")
            continue
        log_path = Path(data_dictionary[log_name]["path"])
        if not log_path.exists():
            print(f"  Warning: log file missing: {log_path}")
            continue
        try:
            result = describe_one_log(
                log_name=log_name,
                log_path=log_path,
                output_dir=output_dir,
                thresholds=thresholds,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: failed on {log_name}: {exc}")
            continue
        print(f"  cases={result['n_cases']}, attributes={result['n_attributes']}")
        print(f"  wrote {result['inventory_path']}")
        _print_summary(result["inventory_df"])

    print(f"\nDone. Outputs under: {output_dir}")
    print(f"Edit {INVENTORY_FILENAME} override columns, then run step 2.")


if __name__ == "__main__":
    main()
