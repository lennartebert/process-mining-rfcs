"""PDF-based power-law analysis from pre-extracted attachments."""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import powerlaw
import matplotlib

matplotlib.use("Agg")  # Headless backend for CLI/batch runs.

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RFCS_DIR
from utils.io import load_attachments, parse_dataset_inputs
from utils.powerlaw.plotting import create_multiplot
from utils.rfc import extract_frequency_counts


def _as_float(value: object) -> float:
    """Convert a value to float, returning nan when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_distribution_attr(fit: powerlaw.Fit, distribution_name: str, attribute_name: str) -> float:
    """Read a distribution attribute from powerlaw, guarding lazy-fit failures."""
    try:
        distribution = getattr(fit, distribution_name)
        return _as_float(getattr(distribution, attribute_name))
    except Exception:
        return float("nan")


def _safe_distribution_compare(
    fit: powerlaw.Fit, left_distribution: str, right_distribution: str
) -> tuple[float, float]:
    """Run powerlaw distribution comparison and guard against fit failures."""
    try:
        r_value, p_value = fit.distribution_compare(left_distribution, right_distribution)
        return _as_float(r_value), _as_float(p_value)
    except Exception:
        return float("nan"), float("nan")


def _collect_fit_row(dataset_name: str, counts: np.ndarray, fit: powerlaw.Fit) -> dict:
    """Collect one PDF/power-law summary row for a dataset."""
    alpha = _safe_distribution_attr(fit, "power_law", "alpha")
    sigma = _safe_distribution_attr(fit, "power_law", "sigma")
    xmin = _safe_distribution_attr(fit, "power_law", "xmin")
    xmax = _safe_distribution_attr(fit, "power_law", "xmax")
    d_value = _safe_distribution_attr(fit, "power_law", "D")
    n_tail = _as_float(getattr(fit, "n_tail", np.nan))

    exp_lambda = _safe_distribution_attr(fit, "exponential", "Lambda")
    exp_xmin = _safe_distribution_attr(fit, "exponential", "xmin")

    lognormal_mu = _safe_distribution_attr(fit, "lognormal", "mu")
    lognormal_sigma = _safe_distribution_attr(fit, "lognormal", "sigma")
    lognormal_xmin = _safe_distribution_attr(fit, "lognormal", "xmin")

    r_exp, p_exp = _safe_distribution_compare(fit, "power_law", "exponential")
    r_lognormal, p_lognormal = _safe_distribution_compare(fit, "power_law", "lognormal")

    return {
        "dataset": dataset_name,
        "unique_nodes": int(len(counts)),
        "attachment_count": int(counts.sum()),
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "count_mean": float(np.mean(counts)),
        "count_median": float(np.median(counts)),
        "xmin": xmin,
        "xmax": xmax,
        "alpha": alpha,
        "sigma": sigma,
        "D": d_value,
        "n_tail": n_tail,
        "power_law_exponent_alpha": alpha,
        "exponential_lambda": exp_lambda,
        "exponential_xmin": exp_xmin,
        "lognormal_mu": lognormal_mu,
        "lognormal_sigma": lognormal_sigma,
        "lognormal_xmin": lognormal_xmin,
        "power_law_vs_exponential_R": r_exp,
        "power_law_vs_exponential_p": p_exp,
        "power_law_vs_lognormal_R": r_lognormal,
        "power_law_vs_lognormal_p": p_lognormal,
    }


def create_distribution_comparison_tables(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Create CSV and LaTeX table with both R and p per compared distribution."""
    comparison_df = pd.DataFrame(
        {
            "dataset": summary_df["dataset"],
            "exponential_R": summary_df["power_law_vs_exponential_R"],
            "exponential_p": summary_df["power_law_vs_exponential_p"],
            "lognormal_R": summary_df["power_law_vs_lognormal_R"],
            "lognormal_p": summary_df["power_law_vs_lognormal_p"],
        }
    )

    csv_path = output_dir / "pdf_distribution_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path.name}")

    latex_path = output_dir / "pdf_distribution_comparison.tex"
    latex_table = comparison_df.to_latex(index=False, float_format=lambda value: f"{value:.3f}")
    latex_path.write_text(latex_table, encoding="utf-8")
    print(f"  Saved: {latex_path.name}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for PDF power-law analysis."""
    parser = argparse.ArgumentParser(description="PDF power-law analysis")
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
        help="Subfolder name under the output root for this run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output root directory (default: {RFCS_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Run PDF power-law analysis end-to-end."""
    args = parse_args(argv)
    base_output_dir = Path(args.output_dir) if args.output_dir else RFCS_DIR
    base_output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = base_output_dir / args.analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_pairs = parse_dataset_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    summary_rows: List[dict] = []
    dataset_fits: List[tuple[str, powerlaw.Fit]] = []

    print(f"Processing {len(input_pairs)} dataset(s)...")
    for dataset_name, attachments_path in input_pairs:
        print(f"\nProcessing {dataset_name}...")
        if not attachments_path.exists():
            print(f"  Warning: attachments file not found: {attachments_path}")
            continue

        try:
            attachments_df = load_attachments(attachments_path)
        except ValueError as exc:
            print(f"  Warning: {exc}")
            continue

        counts = extract_frequency_counts(attachments_df)
        if counts.size < 2:
            print("  Warning: not enough unique node frequencies to fit power-law")
            continue

        print(f"  Loaded {len(attachments_df)} attachments, {len(counts)} unique node frequencies")
        fit = powerlaw.Fit(counts, discrete=True, verbose=False)
        summary_rows.append(_collect_fit_row(dataset_name, counts, fit))
        dataset_fits.append((dataset_name, fit))

    if not summary_rows:
        print("Error: no datasets processed successfully")
        raise SystemExit(1)

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    summary_path = analysis_dir / "pdf_static_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path.name}")

    create_distribution_comparison_tables(summary_df, analysis_dir)
    multiplot_path = analysis_dir / "pdf_powerlaw_fits.pdf"
    create_multiplot(dataset_fits, multiplot_path)
    print(f"  Saved: {multiplot_path.name}")
    print(f"\nAll PDF power-law outputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
