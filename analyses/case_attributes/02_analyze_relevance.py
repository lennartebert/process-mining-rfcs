"""Step 2: attribute–variant relevance, plots, power-law selection CSV.

Reads ``attribute_inventory.csv`` (step 1). Writes:

- ``attribute_selection_for_powerlaw.csv`` (association metrics + override columns)
- ``plots/<attribute>/{rfc_loglog,pdf_loglog[,ccdf_loglog]}.pdf`` (included attrs only)
"""

from __future__ import annotations

import argparse
import re
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
from utils.io.case_tables import VARIANT_COLUMN, build_case_attribute_table
from utils.rfc.case_attribute_association import associate_attribute_with_variant
from utils.rfc.case_attribute_config import (
    INVENTORY_FILENAME,
    POWERLAW_SELECTION_FILENAME,
    ConfigValidationError,
    build_powerlaw_selection_row,
    load_csv,
    merge_powerlaw_selection_dataframe,
    resolve_inventory_row,
    write_csv,
)
from utils.rfc.case_attribute_inventory import (
    RecommendationThresholds,
    recommend_powerlaw_include,
)
from utils.rfc.case_attribute_plotting import (
    plot_attribute_pdf_loglog,
    plot_attribute_rfc_loglog,
    plot_continuous_ccdf_loglog,
    plot_continuous_pdf_loglog,
)
from utils.rfc.case_attribute_transform import (
    TransformError,
    extract_continuous_values,
    transform_by_datatype,
    value_frequency_counts,
)


def _resolve_datasets(raw: List[str]) -> List[str]:
    if len(raw) == 1 and raw[0] == ALL_REAL_LOGS_TOKEN:
        return list(ALL_REAL_LOG_DATASETS)
    return list(raw)


def _safe_dirname(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name.strip())
    return cleaned or "attribute"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute attribute–variant relevance and write "
            "attribute_selection_for_powerlaw.csv"
        )
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(CASE_ATTRIBUTES_DIR),
    )
    parser.add_argument(
        "--data-dictionary",
        type=str,
        default=str(DATA_DICTIONARY_PATH),
    )
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--minimum-distinct-for-powerlaw", type=int, default=50)
    return parser.parse_args(argv)


def analyze_one_log(
    *,
    log_name: str,
    log_path: Path,
    config_dir: Path,
    thresholds: RecommendationThresholds,
    n_permutations: int,
    random_seed: int,
) -> dict[str, Any]:
    inventory_path = config_dir / log_name / INVENTORY_FILENAME
    inventory_df = load_csv(inventory_path)
    if inventory_df is None or inventory_df.empty:
        raise FileNotFoundError(f"Missing step-1 inventory CSV: {inventory_path}")

    event_log = get_event_log_from_path(log_path)
    case_df, _source_levels = build_case_attribute_table(event_log)
    variants = case_df[VARIANT_COLUMN].tolist() if not case_df.empty else []

    selection_rows: list[dict[str, Any]] = []
    plots_dir = config_dir / log_name / "plots"

    old_selection = load_csv(config_dir / log_name / POWERLAW_SELECTION_FILENAME)

    for _, raw_row in inventory_df.iterrows():
        row = raw_row.to_dict()
        attr_name = str(row.get("attribute_name"))
        try:
            resolved = resolve_inventory_row(row)
        except ConfigValidationError as exc:
            print(f"  Warning: invalid config for {attr_name}: {exc}")
            continue

        datatype = resolved["datatype"]
        binning = resolved["binning"]
        metadata = resolved["metadata"]
        step1_included = bool(resolved["include"])

        transformed = None
        counts = None
        if step1_included:
            try:
                transformed = transform_by_datatype(
                    case_df,
                    attr_name,
                    datatype=datatype,
                    binning=binning,
                )
                counts = value_frequency_counts(transformed)
                attr_plot_dir = plots_dir / _safe_dirname(str(attr_name))
                try:
                    plot_attribute_rfc_loglog(
                        counts,
                        attr_plot_dir / "rfc_loglog.pdf",
                        title=f"{log_name} / {attr_name}",
                    )
                    if datatype == "continuous":
                        raw = extract_continuous_values(case_df, attr_name)
                        plot_continuous_pdf_loglog(
                            raw,
                            attr_plot_dir / "pdf_loglog.pdf",
                            title=f"{log_name} / {attr_name}",
                        )
                        plot_continuous_ccdf_loglog(
                            raw,
                            attr_plot_dir / "ccdf_loglog.pdf",
                            title=f"{log_name} / {attr_name}",
                        )
                    else:
                        plot_attribute_pdf_loglog(
                            counts,
                            attr_plot_dir / "pdf_loglog.pdf",
                            title=f"{log_name} / {attr_name}",
                        )
                except Exception as plot_exc:  # noqa: BLE001
                    print(f"  Warning: plot failed for {attr_name}: {plot_exc}")
            except TransformError as exc:
                print(f"  Skip transform/plots for {attr_name}: {exc}")

        if step1_included and transformed is not None:
            association = associate_attribute_with_variant(
                variants,
                transformed.tolist(),
                attribute_name=attr_name,
                inferred_type="categorical",
                source_level=str(metadata.get("source_level", "case")),
                n_permutations=n_permutations,
                random_seed=random_seed,
            )
            association["inferred_type"] = datatype
            association["n_distinct_transformed"] = (
                int(counts.size) if counts is not None else None
            )
        elif not step1_included:
            association = {
                "attribute_name": attr_name,
                "status": "skipped",
                "reason": "not_included_in_step1",
                "n_distinct_transformed": None,
            }
        else:
            association = {
                "attribute_name": attr_name,
                "status": "not_applicable",
                "reason": "untransformable",
                "n_distinct_transformed": None,
            }

        n_distinct_transformed = int(counts.size) if counts is not None else 0
        is_near_constant = bool(metadata.get("is_near_constant", False))
        if counts is not None and counts.size > 0:
            largest_share = float(counts.max() / counts.sum())
            is_near_constant = is_near_constant or largest_share >= 0.98

        n_obs = int(metadata.get("n_cases") or association.get("n_cases") or 0)
        n_raw_distinct = int(metadata.get("n_distinct") or n_distinct_transformed)
        near_constant_for_pl = (
            bool(metadata.get("is_near_constant", False))
            if datatype == "continuous"
            else is_near_constant
        )
        raw_values = None
        if datatype == "continuous" and step1_included:
            try:
                raw_values = extract_continuous_values(case_df, attr_name)
            except Exception:  # noqa: BLE001
                raw_values = None
        pl_rec = recommend_powerlaw_include(
            datatype=datatype,
            n_distinct_transformed=n_distinct_transformed,
            is_near_constant=near_constant_for_pl,
            step1_included=step1_included,
            association=association,
            thresholds=thresholds,
            n_observations=n_obs,
            n_raw_distinct=n_raw_distinct,
            raw_values=raw_values,
        )

        selection_rows.append(
            build_powerlaw_selection_row(
                attribute_name=attr_name,
                resolved_step1=resolved,
                association=association,
                recommendation=pl_rec,
            )
        )

    log_dir = config_dir / log_name
    selection_df = merge_powerlaw_selection_dataframe(old_selection, selection_rows)
    selection_path = log_dir / POWERLAW_SELECTION_FILENAME
    write_csv(selection_df, selection_path)

    return {
        "selection_path": selection_path,
        "n_attributes": len(selection_rows),
        "n_included": int(
            sum(
                1
                for row in selection_rows
                if row.get("status") not in {"skipped", "not_applicable"}
            )
        ),
    }


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    config_dir = Path(args.config_dir)
    thresholds = RecommendationThresholds(
        minimum_distinct_for_powerlaw=args.minimum_distinct_for_powerlaw,
    )
    data_dictionary = get_data_dictionary(
        Path(args.data_dictionary), get_real=True, get_synthetic=True
    )
    datasets = _resolve_datasets(args.datasets)

    print(f"Analyzing attribute relevance for {len(datasets)} dataset(s)...")
    print(
        f"Permutations: n_permutations={args.n_permutations}, "
        f"random_seed={args.random_seed}"
    )

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
            result = analyze_one_log(
                log_name=log_name,
                log_path=log_path,
                config_dir=config_dir,
                thresholds=thresholds,
                n_permutations=args.n_permutations,
                random_seed=args.random_seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  Warning: failed on {log_name}: {exc}")
            continue
        print(f"  attributes={result['n_attributes']} analyzed={result['n_included']}")
        print(f"  wrote {result['selection_path']}")

    print(f"\nDone. Edit {POWERLAW_SELECTION_FILENAME} overrides, then run step 3.")


if __name__ == "__main__":
    main()
