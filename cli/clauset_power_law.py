"""Shared CLI: Clauset-style discrete power-law evaluation from attachments.

For a given attachments path (``dataset=path`` pairs), run GOF and model
comparisons and write ``results/statistical_tests/<analysis-name>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RESULTS_DIR
from utils.io import load_attachments, parse_dataset_inputs
from utils.rfc import extract_frequency_counts
from utils.rfc.powerlaw_pipeline import (
    analyze_powerlaw_data,
    append_and_checkpoint,
    empty_powerlaw_results,
)
from utils.rfc.statistical_tests import (
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
)


def _classify_dataset_error(exc: Exception) -> str:
    """Map per-dataset failures to a compact summary classification."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if any(
        token in lowered
        for token in (
            "not enough",
            "too few",
            "no valid",
            "insufficient",
            "cannot fit",
            "xmin",
        )
    ):
        return f"insufficient observations ({message})"
    return f"error during fitting ({message})"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for batch power-law statistical tests."""
    parser = argparse.ArgumentParser(
        description="Clauset-style power-law evaluation from attachments"
    )
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
        help="Minimum fitted-region variants required for eligibility/classification",
    )
    parser.add_argument(
        "--minimum-orders-of-magnitude",
        type=float,
        default=1.0,
        help="Minimum log10(xmax/xmin) for doubly bounded eligibility",
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
    n_bootstraps: int,
    random_seed: int,
    minimum_fitted_variants: int,
    minimum_orders_of_magnitude: float,
    significance_level: float,
) -> dict[str, dict[str, Any]]:
    """Run statistical tests for one attachments file."""
    del minimum_orders_of_magnitude  # retained for CLI compatibility; unused
    attachments_df = load_attachments(attachments_path)
    counts = extract_frequency_counts(attachments_df)
    return analyze_powerlaw_data(
        counts,
        log_name=dataset_name,
        input_path=str(attachments_path),
        discrete=True,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        minimum_fitted_variants=minimum_fitted_variants,
        significance_level=significance_level,
    )


def _error_results(
    *,
    dataset_name: str,
    attachments_path: Path,
    classification: str,
) -> dict[str, dict[str, Any]]:
    """Build empty per-distribution outputs for a failed dataset."""
    return empty_powerlaw_results(
        log_name=dataset_name,
        input_path=str(attachments_path),
        classification=classification,
    )


def main(argv: List[str] | None = None) -> None:
    """Run batch power-law statistical tests end-to-end."""
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    analysis_dir = output_root / "statistical_tests" / args.analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_pairs = parse_dataset_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    accumulated: dict[str, dict[str, list]] = {
        name: {"gof_rows": [], "comparison_frames": [], "summary_rows": []}
        for name in DISTRIBUTION_NAMES
    }

    print(f"Processing {len(input_pairs)} dataset(s)...")
    print(
        f"Bootstrap settings: n_bootstraps={args.n_bootstraps}, "
        f"random_seed={args.random_seed}"
    )
    print(
        f"Classification thresholds: significance_level={args.significance_level}, "
        f"minimum_fitted_variants={args.minimum_fitted_variants}; "
        f"doubly bounded xmax = min KS among EXCLUDED_HEAD_VARIANTS "
        f"(min log-range arg={args.minimum_orders_of_magnitude} unused for selection)"
    )

    for dataset_name, attachments_path in input_pairs:
        print(f"\nProcessing {dataset_name}...")
        if not attachments_path.exists():
            print(f"  Warning: attachments file not found: {attachments_path}")
            dataset_results = _error_results(
                dataset_name=dataset_name,
                attachments_path=attachments_path,
                classification="error during fitting (attachments file not found)",
            )
        else:
            try:
                dataset_results = analyze_one_dataset(
                    dataset_name=dataset_name,
                    attachments_path=attachments_path,
                    n_bootstraps=args.n_bootstraps,
                    random_seed=args.random_seed,
                    minimum_fitted_variants=args.minimum_fitted_variants,
                    minimum_orders_of_magnitude=args.minimum_orders_of_magnitude,
                    significance_level=args.significance_level,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch resilient
                print(f"  Warning: failed on {dataset_name}: {exc}")
                dataset_results = _error_results(
                    dataset_name=dataset_name,
                    attachments_path=attachments_path,
                    classification=_classify_dataset_error(exc),
                )

        for name in DISTRIBUTION_NAMES:
            summary = dataset_results[name]["summary_row"]
            alpha = summary.get("alpha")
            gof_p = summary.get("gof_p")
            if pd.notna(alpha) and pd.notna(gof_p):
                extra = ""
                if name == DOUBLY_BOUNDED_POWER_LAW and pd.notna(
                    summary.get("excluded_head_variants")
                ):
                    extra = (
                        f", excluded_head_variants="
                        f"{int(summary['excluded_head_variants'])}"
                    )
                print(
                    f"  {name}: alpha={float(alpha):.4f}, "
                    f"gof_p={float(gof_p):.4f}, "
                    f"classification={summary['classification']}{extra}"
                )
            else:
                print(f"  {name}: classification={summary['classification']}")

        append_and_checkpoint(
            analysis_dir=analysis_dir,
            accumulated=accumulated,
            dataset_results=dataset_results,
        )
        print(f"  Checkpointed under: {analysis_dir}")

    if not any(accumulated[name]["summary_rows"] for name in DISTRIBUTION_NAMES):
        print("Error: no datasets processed successfully")
        raise SystemExit(1)

    print(f"\nAll statistical-test outputs saved to: {analysis_dir}")
    for name in DISTRIBUTION_NAMES:
        print(f"  {analysis_dir / name}/{{gof,comparison,summary}}.csv")


if __name__ == "__main__":
    main()
