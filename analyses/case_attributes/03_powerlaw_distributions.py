"""Step 3: Clauset power-law analysis of selected case attributes.

Reads ``attribute_selection_for_powerlaw.csv`` (step 2). Writes the same
``gof`` / ``comparison`` / ``summary`` CSV layout as the variant statistical
tests under ``results/statistical_tests/{analysis_name}/``.
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
    DATA_DICTIONARY_PATH,
    RESULTS_DIR,
)
from utils.io import get_data_dictionary, get_event_log_from_path
from utils.io.case_tables import build_case_attribute_table
from utils.rfc.case_attribute_config import (
    POWERLAW_SELECTION_FILENAME,
    load_csv,
    resolve_powerlaw_row,
)
from utils.rfc.case_attribute_transform import (
    TransformError,
    extract_continuous_values,
    transform_by_datatype,
    value_frequency_counts,
)
from utils.rfc.powerlaw_pipeline import (
    analyze_powerlaw_data,
    append_and_checkpoint,
    empty_powerlaw_results,
)
from utils.rfc.statistical_tests import (
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
)


def _resolve_datasets(raw: List[str]) -> List[str]:
    if len(raw) == 1 and raw[0] == ALL_REAL_LOGS_TOKEN:
        return list(ALL_REAL_LOG_DATASETS)
    return list(raw)


def _classify_error(exc: Exception) -> str:
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
    parser = argparse.ArgumentParser(
        description="Power-law analysis of selected case-attribute distributions"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(RESULTS_DIR / "case_attribute_analysis"),
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        default="case_attributes",
        help="Subfolder under results/statistical_tests (default: case_attributes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results)",
    )
    parser.add_argument(
        "--data-dictionary",
        type=str,
        default=str(DATA_DICTIONARY_PATH),
    )
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--minimum-fitted-variants", type=int, default=50)
    parser.add_argument("--significance-level", type=float, default=0.10)
    return parser.parse_args(argv)


def analyze_one_attribute(
    *,
    dataset_name: str,
    attribute_name: str,
    case_df: pd.DataFrame,
    entry: dict[str, Any],
    source_log_path: str,
    n_bootstraps: int,
    random_seed: int,
    minimum_fitted_variants: int,
    significance_level: float,
) -> dict[str, dict[str, Any]]:
    """Run shared Clauset stack for one included attribute."""
    effective_cfg = resolve_powerlaw_row(entry)
    log_name = f"{dataset_name}::{attribute_name}"

    if not effective_cfg["include"]:
        return empty_powerlaw_results(
            log_name=log_name,
            input_path=source_log_path,
            classification="skipped_not_included",
        )

    datatype = effective_cfg["datatype"]
    discrete = bool(effective_cfg["discrete"])
    if datatype == "continuous":
        values = extract_continuous_values(case_df, attribute_name)
        return analyze_powerlaw_data(
            values,
            log_name=log_name,
            input_path=source_log_path,
            discrete=discrete,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
            minimum_fitted_variants=minimum_fitted_variants,
            significance_level=significance_level,
        )

    if datatype in {"datetime", "identifier-like", "unsupported"}:
        return empty_powerlaw_results(
            log_name=log_name,
            input_path=source_log_path,
            classification=f"skipped_handling (datatype {datatype!r})",
        )

    transformed = transform_by_datatype(
        case_df,
        attribute_name,
        datatype=datatype,
        binning=effective_cfg["binning"],
    )
    counts = value_frequency_counts(transformed)
    return analyze_powerlaw_data(
        counts,
        log_name=log_name,
        input_path=source_log_path,
        discrete=discrete,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        minimum_fitted_variants=minimum_fitted_variants,
        significance_level=significance_level,
    )


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    analysis_dir = output_root / "statistical_tests" / args.analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    config_dir = Path(args.config_dir)

    data_dictionary = get_data_dictionary(
        Path(args.data_dictionary), get_real=True, get_synthetic=True
    )
    datasets = _resolve_datasets(args.datasets)

    accumulated: dict[str, dict[str, list]] = {
        name: {"gof_rows": [], "comparison_frames": [], "summary_rows": []}
        for name in DISTRIBUTION_NAMES
    }

    print(f"Processing {len(datasets)} dataset(s) for case-attribute power laws...")
    print(
        f"Bootstrap settings: n_bootstraps={args.n_bootstraps}, "
        f"random_seed={args.random_seed}"
    )
    print(f"Outputs under: {analysis_dir}")

    n_units = 0
    for dataset_name in datasets:
        print(f"\n=== {dataset_name} ===")
        csv_path = config_dir / dataset_name / POWERLAW_SELECTION_FILENAME
        selection_df = load_csv(csv_path)
        if selection_df is None or selection_df.empty:
            print(f"  Warning: missing selection CSV: {csv_path}")
            continue

        if dataset_name not in data_dictionary:
            print("  Warning: dataset not in data dictionary; skipping")
            continue
        log_path = Path(data_dictionary[dataset_name]["path"])
        if not log_path.exists():
            print(f"  Warning: log file missing: {log_path}")
            continue

        try:
            event_log = get_event_log_from_path(log_path)
            case_df, _ = build_case_attribute_table(event_log)
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: failed loading {dataset_name}: {exc}")
            continue

        for _, raw_row in selection_df.iterrows():
            entry = raw_row.to_dict()
            attribute_name = str(entry.get("attribute_name"))
            effective_cfg = resolve_powerlaw_row(entry)
            if not effective_cfg["include"]:
                continue

            unit_name = f"{dataset_name}::{attribute_name}"
            print(
                f"  Attribute: {attribute_name} "
                f"({effective_cfg['datatype']}, discrete={effective_cfg['discrete']})"
            )
            try:
                dataset_results = analyze_one_attribute(
                    dataset_name=dataset_name,
                    attribute_name=attribute_name,
                    case_df=case_df,
                    entry=entry,
                    source_log_path=str(log_path),
                    n_bootstraps=args.n_bootstraps,
                    random_seed=args.random_seed,
                    minimum_fitted_variants=args.minimum_fitted_variants,
                    significance_level=args.significance_level,
                )
            except (TransformError, Exception) as exc:  # noqa: BLE001
                print(f"    Warning: failed on {unit_name}: {exc}")
                dataset_results = empty_powerlaw_results(
                    log_name=unit_name,
                    input_path=str(log_path),
                    classification=_classify_error(exc),
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
                        f"    {name}: alpha={float(alpha):.4f}, "
                        f"gof_p={float(gof_p):.4f}, "
                        f"classification={summary['classification']}{extra}"
                    )
                else:
                    print(
                        f"    {name}: classification={summary['classification']}"
                    )

            append_and_checkpoint(
                analysis_dir=analysis_dir,
                accumulated=accumulated,
                dataset_results=dataset_results,
            )
            n_units += 1

    if n_units == 0:
        print("Error: no attributes processed successfully")
        raise SystemExit(1)

    print(f"\nAll statistical-test outputs saved to: {analysis_dir}")
    for name in DISTRIBUTION_NAMES:
        print(f"  {analysis_dir / name}/{{gof,comparison,summary}}.csv")


if __name__ == "__main__":
    main()
