"""Verify consistency between log_info.csv and attachment extracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import LOG_INFO_DIR, RESULTS_DIR
from utils.io import load_attachments
from utils.parsing import parse_count

SUPPORTED_CONCEPTS = ("variants", "activities")


def _load_log_info_rows(datasets: List[str], log_info_path: Path) -> pd.DataFrame:
    if not log_info_path.exists():
        raise FileNotFoundError(f"log_info file not found: {log_info_path}")
    df = pd.read_csv(log_info_path)
    by_log = df.set_index("Log")
    missing_logs = [name for name in datasets if name not in by_log.index]
    if missing_logs:
        raise ValueError(f"log_info.csv missing rows for: {missing_logs}")
    return by_log


def _check_variants(
    dataset: str,
    row: pd.Series,
    attachments_path: Path,
) -> List[str]:
    errors: List[str] = []
    if not attachments_path.exists():
        return [f"{dataset}: missing {attachments_path}"]

    expected_variants = parse_count(row["# Variants"])
    expected_cases = parse_count(row["# Cases"])
    if pd.isna(expected_variants) or pd.isna(expected_cases):
        return [f"{dataset}: log_info has missing # Variants or # Cases"]

    df = load_attachments(attachments_path)
    unique_nodes = int(df["node_id"].nunique())
    row_count = int(len(df))
    expected_variants_int = int(round(expected_variants))
    expected_cases_int = int(round(expected_cases))

    if unique_nodes != expected_variants_int:
        errors.append(
            f"{dataset}: # Variants in log_info ({expected_variants_int}) "
            f"!= unique nodes in {attachments_path} ({unique_nodes})"
        )
    if row_count != expected_cases_int:
        errors.append(
            f"{dataset}: # Cases in log_info ({expected_cases_int}) "
            f"!= rows in {attachments_path} ({row_count})"
        )
    return errors


def _check_activities(
    dataset: str,
    row: pd.Series,
    attachments_path: Path,
) -> List[str]:
    errors: List[str] = []
    if not attachments_path.exists():
        return [f"{dataset}: missing {attachments_path}"]

    expected_events = parse_count(row["# Events"])
    expected_activities = parse_count(row["# Activities"])
    if pd.isna(expected_events) or pd.isna(expected_activities):
        return [f"{dataset}: log_info has missing # Events or # Activities"]

    df = load_attachments(attachments_path)
    unique_nodes = int(df["node_id"].nunique())
    row_count = int(len(df))
    expected_events_int = int(round(expected_events))
    expected_activities_int = int(round(expected_activities))

    if row_count != expected_events_int:
        errors.append(
            f"{dataset}: # Events in log_info ({expected_events_int}) "
            f"!= rows in {attachments_path} ({row_count})"
        )
    if unique_nodes != expected_activities_int:
        errors.append(
            f"{dataset}: # Activities in log_info ({expected_activities_int}) "
            f"!= unique nodes in {attachments_path} ({unique_nodes})"
        )
    return errors


def run_checks(
    datasets: List[str],
    concept: str,
    output_root: Path,
    log_info_path: Path,
) -> List[str]:
    """Run integrity checks; return human-readable error messages."""
    if concept not in SUPPORTED_CONCEPTS:
        print(
            f"Integrity checks: skipped (no checks defined for concept '{concept}')."
        )
        return []

    log_info_df = _load_log_info_rows(datasets, log_info_path)
    all_errors: List[str] = []
    checker = _check_variants if concept == "variants" else _check_activities

    for dataset in datasets:
        row = log_info_df.loc[dataset]
        attachments_path = output_root / concept / dataset / "attachments.csv.gz"
        all_errors.extend(checker(dataset, row, attachments_path))

    return all_errors


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify log_info.csv matches attachment extracts for the selected concept."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names to check",
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        choices=["variants", "activities", "dfrs"],
        help="Concept whose attachments are compared to log_info",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root containing <concept>/<dataset>/attachments.csv.gz (default: results)",
    )
    parser.add_argument(
        "--log-info-path",
        type=str,
        default=None,
        help="Path to log_info.csv (default: results/log_info/log_info.csv)",
    )
    return parser.parse_args(argv)


def _write_report(output_root: Path, concept: str, lines: List[str]) -> Path:
    report_path = output_root / concept / "integrity_checks.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    log_info_path = Path(args.log_info_path) if args.log_info_path else LOG_INFO_DIR / "log_info.csv"

    report_lines = [
        f"Running integrity checks (concept={args.concept}, datasets={len(args.datasets)})..."
    ]
    print(report_lines[0])
    errors = run_checks(args.datasets, args.concept, output_root, log_info_path)

    if not errors and args.concept in SUPPORTED_CONCEPTS:
        report_lines.append(f"Integrity checks passed for {len(args.datasets)} dataset(s).")
        for line in report_lines[1:]:
            print(line)
        report_path = _write_report(output_root, args.concept, report_lines)
        print(f"Saved: {report_path}")
        return

    if errors:
        report_lines.append("Integrity check failures:")
        print(report_lines[-1])
        for message in errors:
            line = f"  - {message}"
            report_lines.append(line)
            print(line)
        report_path = _write_report(output_root, args.concept, report_lines)
        print(f"Saved: {report_path}")
        raise SystemExit(1)

    report_path = _write_report(output_root, args.concept, report_lines)
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
