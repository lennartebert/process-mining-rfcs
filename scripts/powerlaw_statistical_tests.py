"""Batch Clauset-style discrete power-law statistical tests from attachments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RESULTS_DIR
from utils.io import load_attachments, parse_dataset_inputs
from utils.rfc import (
    build_summary_row,
    classify_power_law_result,
    clauset_gof_bootstrap,
    compare_power_law_alternatives,
    descriptive_frequency_stats,
    extract_frequency_counts,
    fit_discrete_power_law,
    to_frequency_array,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for batch power-law statistical tests."""
    parser = argparse.ArgumentParser(description="Power-law statistical tests")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Dataset/attachments pairs: <dataset>=<attachments_csv_path>",
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        required=True,
        help="Subfolder name under results/statistical_tests for this run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results)",
    )
    parser.add_argument(
        "--concept-name",
        type=str,
        default=None,
        help="Concept label stored in summary rows (default: --analysis-name)",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Number of Clauset GOF bootstrap iterations (default: 1000)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42)",
    )
    parser.add_argument(
        "--minimum-fitted-variants",
        type=int,
        default=50,
        help="Minimum fitted-region variants required for classification",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.10,
        help="Significance level for GOF and model comparisons (default: 0.10)",
    )
    return parser.parse_args(argv)


def analyze_one_dataset(
    *,
    dataset_name: str,
    attachments_path: Path,
    concept_name: str,
    n_bootstraps: int,
    random_seed: int,
    minimum_fitted_variants: int,
    significance_level: float,
) -> tuple[dict, pd.DataFrame]:
    """Run statistical tests for one attachments file."""
    attachments_df = load_attachments(attachments_path)
    counts = extract_frequency_counts(attachments_df)
    if counts.size < 2:
        raise ValueError("not enough unique node frequencies to fit a power law")

    frequency_df = pd.DataFrame({"frequency": counts.astype(int)})
    frequencies = to_frequency_array(frequency_df, frequency_column="frequency")
    descriptive_stats = descriptive_frequency_stats(frequencies)
    fit_result = fit_discrete_power_law(frequencies)
    comparison_df = compare_power_law_alternatives(
        fit_result["fit"],
        log_name=dataset_name,
    )
    gof_result = clauset_gof_bootstrap(
        frequencies,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
    )
    classification = classify_power_law_result(
        n_fitted_variants=int(fit_result["n_fitted_variants"]),
        gof_p=float(gof_result["gof_p"]),
        comparison_df=comparison_df,
        minimum_fitted_variants=minimum_fitted_variants,
        significance_level=significance_level,
    )
    summary_row = build_summary_row(
        concept_name=concept_name,
        log_name=dataset_name,
        input_path=str(attachments_path),
        descriptive_stats=descriptive_stats,
        fit_result=fit_result,
        gof_p=float(gof_result["gof_p"]),
        classification=classification,
    )
    summary_row["gof_n_success"] = int(gof_result["n_success"])
    summary_row["gof_n_failed"] = int(gof_result["n_failed"])
    return summary_row, comparison_df


def main(argv: List[str] | None = None) -> None:
    """Run batch power-law statistical tests end-to-end."""
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    analysis_dir = output_root / "statistical_tests" / args.analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    concept_name = (args.concept_name or args.analysis_name).strip() or args.analysis_name

    try:
        input_pairs = parse_dataset_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    summary_rows: List[dict] = []
    comparison_frames: List[pd.DataFrame] = []

    print(f"Processing {len(input_pairs)} dataset(s)...")
    print(
        f"Bootstrap settings: n_bootstraps={args.n_bootstraps}, "
        f"random_seed={args.random_seed}"
    )

    for dataset_name, attachments_path in input_pairs:
        print(f"\nProcessing {dataset_name}...")
        if not attachments_path.exists():
            print(f"  Warning: attachments file not found: {attachments_path}")
            continue

        try:
            summary_row, comparison_df = analyze_one_dataset(
                dataset_name=dataset_name,
                attachments_path=attachments_path,
                concept_name=concept_name,
                n_bootstraps=args.n_bootstraps,
                random_seed=args.random_seed,
                minimum_fitted_variants=args.minimum_fitted_variants,
                significance_level=args.significance_level,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch resilient per dataset
            print(f"  Warning: failed on {dataset_name}: {exc}")
            continue

        print(
            f"  alpha={summary_row['alpha']:.4f}, xmin={summary_row['xmin']:g}, "
            f"gof_p={summary_row['gof_p']:.4f}, classification={summary_row['classification']}"
        )
        summary_rows.append(summary_row)
        comparison_frames.append(comparison_df)

    if not summary_rows:
        print("Error: no datasets processed successfully")
        raise SystemExit(1)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("log_name")
        .reset_index(drop=True)
    )
    comparisons_df = (
        pd.concat(comparison_frames, ignore_index=True)
        .sort_values(["log_name", "model_2"])
        .reset_index(drop=True)
    )

    summary_path = analysis_dir / "summary.csv"
    comparisons_path = analysis_dir / "comparisons.csv"
    summary_df.to_csv(summary_path, index=False)
    comparisons_df.to_csv(comparisons_path, index=False)

    print(f"\nSaved: {summary_path}")
    print(f"Saved: {comparisons_path}")
    print(f"All statistical-test outputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
