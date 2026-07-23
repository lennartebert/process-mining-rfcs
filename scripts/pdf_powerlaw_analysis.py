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
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RESULTS_DIR
from utils.io import load_attachments, parse_dataset_inputs
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


def create_multiplot(dataset_fits: List[tuple[str, powerlaw.Fit]], output_path: Path) -> None:
    """Create one multi-panel log-log PDF plot with fitted lines."""
    if not dataset_fits:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(dataset_fits) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), squeeze=False)
    flat_axes = axes.flatten()

    for idx, (dataset_name, fit) in enumerate(dataset_fits):
        ax = flat_axes[idx]
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        is_bottom_row = row_idx == (n_rows - 1)
        is_left_column = col_idx == 0
        try:
            fit.plot_pdf(
                ax=ax,
                color="blue",
                marker="o",
                markersize=6.4,
                markerfacecolor="none",
                markeredgecolor="blue",
                markeredgewidth=1.4,
                linestyle="None",
                label="Observed PDF",
            )
            fit.power_law.plot_pdf(ax=ax, color="black", linestyle="--", linewidth=2.6, label="Power law fit")
            fit.lognormal.plot_pdf(ax=ax, color="red", linestyle=":", linewidth=2.6, label="Lognormal fit")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=1)
        except Exception:
            ax.text(0.5, 0.5, "fit unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(dataset_name, fontsize=11)
        ax.set_xlabel("Frequency" if is_bottom_row else "")
        ax.set_ylabel("p(Frequency)" if is_left_column else "")
        ax.grid(True, alpha=0.35, linewidth=1.15)
        ax.tick_params(axis="both", which="both", width=1.35, labelsize=9.5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.35)

    for idx in range(len(dataset_fits), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.004))
        fig.tight_layout(rect=(0, 0.033, 1, 1))
    else:
        fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


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
        help="Output root directory (default: results)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Run PDF power-law analysis end-to-end."""
    args = parse_args(argv)
    base_output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
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
    create_multiplot(dataset_fits, analysis_dir / "pdf_powerlaw_fits.pdf")
    print(f"\nAll PDF power-law outputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()
