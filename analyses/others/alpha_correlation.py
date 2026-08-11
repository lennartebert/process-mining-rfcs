"""Compute correlations between log-size metrics and power-law alpha exponents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import LOG_INFO_DIR, RFCS_DIR
from utils.parsing import parse_count

LOG_METRICS: List[str] = [
    "# Cases",
    "# Events",
    "# Variants",
    "# Activities",
]

POWER_ALPHA_COLS = ("power_alpha", "power_law_exponent_alpha")
BOUNDED_POWER_ALPHA_COLS = ("bounded_power_alpha", "bounded_power_law_exponent_alpha")


def _resolve_alpha_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first present alpha column name from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_alpha_columns(static_df: pd.DataFrame) -> pd.DataFrame:
    """Extract dataset and both alpha columns with normalized names."""
    power_col = _resolve_alpha_column(static_df, POWER_ALPHA_COLS)
    bounded_col = _resolve_alpha_column(static_df, BOUNDED_POWER_ALPHA_COLS)
    if power_col is None or bounded_col is None:
        missing = []
        if power_col is None:
            missing.append("power_alpha / power_law_exponent_alpha")
        if bounded_col is None:
            missing.append("bounded_power_alpha / bounded_power_law_exponent_alpha")
        raise ValueError(f"Static analysis CSV is missing required columns: {', '.join(missing)}")

    return pd.DataFrame(
        {
            "dataset": static_df["dataset"].astype(str),
            "power_law_exponent_alpha": pd.to_numeric(static_df[power_col], errors="coerce"),
            "bounded_power_law_exponent_alpha": pd.to_numeric(
                static_df[bounded_col], errors="coerce"
            ),
        }
    )


def _pearson_with_pvalue(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Pearson r and two-sided p-value; NaN if undefined."""
    paired = pd.concat([x, y], axis=1).dropna()
    if len(paired) < 2:
        return float("nan"), float("nan")
    r, p_value = pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
    return float(r), float(p_value)


def _compute_correlations(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation and p-value of each log metric with both alpha columns."""
    rows: List[dict] = []
    for metric in LOG_METRICS:
        valid = merged_df[
            [metric, "power_law_exponent_alpha", "bounded_power_law_exponent_alpha"]
        ].dropna()
        if len(valid) < 2:
            print(
                f"Warning: fewer than 2 valid rows for '{metric}'; "
                "writing NaN correlations and p-values."
            )
            power_corr = float("nan")
            power_p = float("nan")
            bounded_corr = float("nan")
            bounded_p = float("nan")
        else:
            power_corr, power_p = _pearson_with_pvalue(
                valid[metric], valid["power_law_exponent_alpha"]
            )
            bounded_corr, bounded_p = _pearson_with_pvalue(
                valid[metric], valid["bounded_power_law_exponent_alpha"]
            )
        rows.append(
            {
                "log_metric": metric,
                "correlation_to_alpha_power_law": power_corr,
                "p_value_to_alpha_power_law": power_p,
                "correlation_to_alpha_power_law_bounded": bounded_corr,
                "p_value_to_alpha_power_law_bounded": bounded_p,
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for alpha correlation analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute Pearson correlations and p-values between log-size metrics "
            "and power-law alpha exponents."
        )
    )
    parser.add_argument(
        "--log-info-path",
        type=str,
        default=None,
        help=f"Path to log_info.csv (default: {LOG_INFO_DIR / 'log_info.csv'})",
    )
    parser.add_argument(
        "--static-analysis-path",
        type=str,
        default=None,
        help=(
            "Path to rfc_static_analysis.csv "
            "(default: <output-dir>/<analysis-name>/rfc_static_analysis.csv)"
        ),
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        default="variants",
        help="Analysis subfolder under output-dir when static path is not set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RFCS_DIR),
        help=f"Output root directory (default: {RFCS_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Load inputs, compute correlations, and write correlations.csv."""
    args = parse_args(argv)
    output_root = Path(args.output_dir)
    analysis_name = args.analysis_name.strip() or "variants"
    analysis_dir = output_root / analysis_name

    log_info_path = (
        Path(args.log_info_path)
        if args.log_info_path
        else LOG_INFO_DIR / "log_info.csv"
    )
    static_path = (
        Path(args.static_analysis_path)
        if args.static_analysis_path
        else analysis_dir / "rfc_static_analysis.csv"
    )

    if not log_info_path.exists():
        print(f"Error: log info file not found: {log_info_path}", file=sys.stderr)
        return
    if not static_path.exists():
        print(f"Error: static analysis file not found: {static_path}", file=sys.stderr)
        return

    log_info_df = pd.read_csv(log_info_path)
    try:
        static_df = pd.read_csv(static_path)
    except Exception as exc:
        print(f"Error: failed to read static analysis file: {static_path}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return

    if "dataset" not in static_df.columns:
        print(
            f"Error: static analysis file missing 'dataset' column: {static_path}",
            file=sys.stderr,
        )
        return
    if "Log" not in log_info_df.columns:
        print(
            f"Error: log info file missing 'Log' column: {log_info_path}",
            file=sys.stderr,
        )
        return

    try:
        alpha_df = _load_alpha_columns(static_df)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return

    for metric in LOG_METRICS:
        if metric not in log_info_df.columns:
            print(
                f"Error: log info file missing '{metric}' column: {log_info_path}",
                file=sys.stderr,
            )
            return

    log_metrics_df = log_info_df.copy()
    log_metrics_df["Log"] = log_metrics_df["Log"].astype(str)
    for metric in LOG_METRICS:
        log_metrics_df[metric] = log_metrics_df[metric].map(parse_count)

    merged = log_metrics_df.merge(
        alpha_df,
        left_on="Log",
        right_on="dataset",
        how="inner",
    )
    if merged.empty:
        print(
            "Error: no overlapping logs between log_info and static analysis files.",
            file=sys.stderr,
        )
        return

    correlations_df = _compute_correlations(merged)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output_path = analysis_dir / "correlations.csv"
    correlations_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(correlations_df.to_string(index=False))


if __name__ == "__main__":
    main()
