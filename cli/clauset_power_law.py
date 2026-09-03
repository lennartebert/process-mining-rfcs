"""Shared CLI: Clauset-style discrete power-law evaluation from attachments.

For a given attachments path (``dataset=path`` pairs), run GOF and model
comparisons and write ``results/<analysis-name>/<model>/``.
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

from utils.powerlaw import (
    DEFAULT_MINIMUM_FITTED_TYPES,
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
    accumulate_and_write_csvs,
    empty_clauset_result,
    run_clauset_pipeline,
)
from utils.constants import RESULTS_DIR
from utils.io import load_attachments, parse_dataset_inputs
from utils.rfc import extract_frequency_counts


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
        help="Subfolder name under the output root for this run "
        "(e.g. n1 -> results/powerlaw_statistics/real/n1/<model>/)",
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
        "--minimum-fitted-types",
        type=int,
        default=DEFAULT_MINIMUM_FITTED_TYPES,
        help=(
            "Minimum fitted-region types for classification step (1) "
            f"(default: {DEFAULT_MINIMUM_FITTED_TYPES})"
        ),
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.10,
        help="Significance level for GOF and model comparisons (default: 0.10)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DISTRIBUTION_NAMES),
        choices=list(DISTRIBUTION_NAMES),
        help=(
            "Power-law models to fit (default: all three). "
            "Each model is run as a separate pipeline call. "
            "Example: --models lower_bounded_power_law"
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Write per-log shards gof_<log>.csv / comparison_<log>.csv / "
            "summary_<log>.csv instead of aggregated CSVs"
        ),
    )
    return parser.parse_args(argv)


def analyze_one_dataset(
    *,
    dataset_name: str,
    attachments_path: Path,
    n_bootstraps: int,
    random_seed: int,
    minimum_fitted_types: int,
    significance_level: float,
    models: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Run one Clauset pipeline call per selected model."""
    attachments_df = load_attachments(attachments_path)
    counts = extract_frequency_counts(attachments_df)
    results: dict[str, dict[str, Any]] = {}
    for model in models:
        results[model] = run_clauset_pipeline(
            counts,
            model=model,
            log_name=dataset_name,
            input_path=str(attachments_path),
            discrete=True,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
            minimum_fitted_types=minimum_fitted_types,
            significance_level=significance_level,
        )
    return results


def _error_results(
    *,
    dataset_name: str,
    attachments_path: Path,
    classification: str,
    models: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Build empty per-model outputs for a failed dataset."""
    return {
        model: empty_clauset_result(
            log_name=dataset_name,
            input_path=str(attachments_path),
            model=model,
            classification=classification,
        )
        for model in models
    }


def main(argv: List[str] | None = None) -> None:
    """Run batch power-law statistical tests end-to-end."""
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    analysis_dir = output_root / args.analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_pairs = parse_dataset_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    selected_models = tuple(args.models)
    accumulated: dict[str, dict[str, list]] = {
        name: {"gof_rows": [], "comparison_frames": [], "summary_rows": []}
        for name in selected_models
    }
    any_success = False

    print(f"Processing {len(input_pairs)} dataset(s)...")
    print(
        f"Bootstrap settings: n_bootstraps={args.n_bootstraps}, "
        f"random_seed={args.random_seed}"
    )
    print(f"Models: {', '.join(selected_models)}")
    print(
        f"Classification thresholds: significance_level={args.significance_level}, "
        f"minimum_fitted_types={args.minimum_fitted_types}"
    )
    if args.parallel:
        print("Parallel mode: writing per-log CSV shards")

    for dataset_name, attachments_path in input_pairs:
        print(f"\nProcessing {dataset_name}...")
        if not attachments_path.exists():
            print(f"  Warning: attachments file not found: {attachments_path}")
            dataset_results = _error_results(
                dataset_name=dataset_name,
                attachments_path=attachments_path,
                classification="error during fitting (attachments file not found)",
                models=selected_models,
            )
        else:
            try:
                dataset_results = analyze_one_dataset(
                    dataset_name=dataset_name,
                    attachments_path=attachments_path,
                    n_bootstraps=args.n_bootstraps,
                    random_seed=args.random_seed,
                    minimum_fitted_types=args.minimum_fitted_types,
                    significance_level=args.significance_level,
                    models=selected_models,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch resilient
                print(f"  Warning: failed on {dataset_name}: {exc}")
                dataset_results = _error_results(
                    dataset_name=dataset_name,
                    attachments_path=attachments_path,
                    classification=_classify_dataset_error(exc),
                    models=selected_models,
                )

        for name in selected_models:
            summary = dataset_results[name]["summary_row"]
            alpha = summary.get("alpha")
            gof_p = summary.get("gof_p")
            if pd.notna(alpha) and pd.notna(gof_p):
                extra = ""
                if name == DOUBLY_BOUNDED_POWER_LAW and pd.notna(
                    summary.get("doubly_bounded_exclude_head_variants")
                ):
                    extra = (
                        f", doubly_bounded_exclude_head_variants="
                        f"{int(summary['doubly_bounded_exclude_head_variants'])}"
                    )
                print(
                    f"  {name}: alpha={float(alpha):.4f}, "
                    f"gof_p={float(gof_p):.4f}, "
                    f"classification={summary['classification']}{extra}"
                )
            else:
                print(f"  {name}: classification={summary['classification']}")

        if args.parallel:
            shard_acc: dict[str, dict[str, list]] = {
                name: {"gof_rows": [], "comparison_frames": [], "summary_rows": []}
                for name in selected_models
            }
            accumulate_and_write_csvs(
                analysis_dir=analysis_dir,
                accumulated=shard_acc,
                dataset_results=dataset_results,
                file_suffix=f"_{dataset_name}",
            )
            any_success = any(
                shard_acc[name]["summary_rows"] for name in selected_models
            ) or any_success
            print(
                f"  Checkpointed shards under: {analysis_dir}/"
                f"{{model}}/{{gof,comparison,summary}}_{dataset_name}.csv"
            )
        else:
            accumulate_and_write_csvs(
                analysis_dir=analysis_dir,
                accumulated=accumulated,
                dataset_results=dataset_results,
            )
            print(f"  Checkpointed under: {analysis_dir}")

    if args.parallel:
        if not any_success:
            print("Error: no datasets processed successfully")
            raise SystemExit(1)
        print(f"\nAll statistical-test shards saved under: {analysis_dir}")
        for name in selected_models:
            print(f"  {analysis_dir / name}/{{gof,comparison,summary}}_<log>.csv")
    else:
        if not any(accumulated[name]["summary_rows"] for name in selected_models):
            print("Error: no datasets processed successfully")
            raise SystemExit(1)
        print(f"\nAll statistical-test outputs saved to: {analysis_dir}")
        for name in selected_models:
            print(f"  {analysis_dir / name}/{{gof,comparison,summary}}.csv")


if __name__ == "__main__":
    main()
