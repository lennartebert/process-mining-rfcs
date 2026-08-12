"""Step 3: Clauset power-law analysis of selected case attributes.

Reads ``attribute_selection_for_powerlaw.csv`` (step 2). Writes the same
``gof`` / ``comparison`` / ``summary`` CSV layout under
``results/case_attributes/powerlaw/<model>/``.
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

from utils.powerlaw import (
    DEFAULT_MINIMUM_FITTED_TYPES,
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
    accumulate_and_write_csvs,
    empty_clauset_result,
    run_clauset_pipeline,
)
from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ALL_REAL_LOGS_TOKEN,
    CASE_ATTRIBUTES_DIR,
    DATA_DICTIONARY_PATH,
)
from utils.io import get_data_dictionary, get_event_log_from_path
from utils.io.case_tables import build_case_attribute_table
from utils.case_attribute.config import (
    POWERLAW_SELECTION_FILENAME,
    load_csv,
    resolve_powerlaw_row,
)
from utils.case_attribute.transform import (
    TransformError,
    extract_continuous_values,
    transform_by_datatype,
    value_frequency_counts,
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


def _empty_all_models(
    *,
    log_name: str,
    input_path: str,
    classification: str,
) -> dict[str, dict[str, Any]]:
    return {
        model: empty_clauset_result(
            log_name=log_name,
            input_path=input_path,
            model=model,
            classification=classification,
        )
        for model in DISTRIBUTION_NAMES
    }


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Power-law analysis of selected case-attribute distributions"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(CASE_ATTRIBUTES_DIR),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Case-attributes results root (default: {CASE_ATTRIBUTES_DIR})",
    )
    parser.add_argument(
        "--data-dictionary",
        type=str,
        default=str(DATA_DICTIONARY_PATH),
    )
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--minimum-fitted-types",
        type=int,
        default=DEFAULT_MINIMUM_FITTED_TYPES,
    )
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
    minimum_fitted_types: int,
    significance_level: float,
) -> dict[str, dict[str, Any]]:
    """Run Clauset once per model for one included attribute."""
    effective_cfg = resolve_powerlaw_row(entry)
    log_name = f"{dataset_name}::{attribute_name}"

    if not effective_cfg["include"]:
        return _empty_all_models(
            log_name=log_name,
            input_path=source_log_path,
            classification="skipped_not_included",
        )

    datatype = effective_cfg["datatype"]
    discrete = bool(effective_cfg["discrete"])

    if datatype == "continuous":
        values = extract_continuous_values(case_df, attribute_name)
        observations = values
    elif datatype in {"datetime", "identifier-like", "unsupported"}:
        return _empty_all_models(
            log_name=log_name,
            input_path=source_log_path,
            classification=f"skipped_handling (datatype {datatype!r})",
        )
    else:
        transformed = transform_by_datatype(
            case_df,
            attribute_name,
            datatype=datatype,
            binning=effective_cfg["binning"],
        )
        observations = value_frequency_counts(transformed)

    results: dict[str, dict[str, Any]] = {}
    for model in DISTRIBUTION_NAMES:
        results[model] = run_clauset_pipeline(
            observations,
            model=model,
            log_name=log_name,
            input_path=source_log_path,
            discrete=discrete,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
            minimum_fitted_types=minimum_fitted_types,
            significance_level=significance_level,
        )
    return results


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else CASE_ATTRIBUTES_DIR
    analysis_dir = output_root / "powerlaw"
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
                    minimum_fitted_types=args.minimum_fitted_types,
                    significance_level=args.significance_level,
                )
            except (TransformError, Exception) as exc:  # noqa: BLE001
                print(f"    Warning: failed on {unit_name}: {exc}")
                dataset_results = _empty_all_models(
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
                        summary.get("doubly_bounded_exclude_head_variants")
                    ):
                        extra = (
                            f", doubly_bounded_exclude_head_variants="
                            f"{int(summary['doubly_bounded_exclude_head_variants'])}"
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

            accumulate_and_write_csvs(
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
