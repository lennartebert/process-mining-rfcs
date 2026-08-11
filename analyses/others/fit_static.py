"""Rank-frequency-curve (RFC) analysis from pre-extracted attachments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RESULTS_DIR
from utils.io import load_attachments, parse_dataset_inputs
from utils.rfc import (
    compute_fit_statistics,
    create_all_plots,
    create_dataset_plots,
    create_dataset_unfitted_plots,
    create_loglog_rfc_plot,
    create_loglog_subfigure_plot,
    extract_rank_frequencies,
    extract_rank_frequencies_from_counts,
    extract_rank_frequencies_from_event_log,
)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for RFC analysis."""
    parser = argparse.ArgumentParser(description="RFC analysis")
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
    """Run RFC analysis end-to-end."""
    plt.switch_backend("Agg")
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

    all_rank_freqs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    static_rows: List[dict] = []
    static_stats_by_dataset: Dict[str, dict] = {}
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
        ranks, frequencies = extract_rank_frequencies(attachments_df)
        all_rank_freqs[dataset_name] = (ranks, frequencies)
        if len(ranks) > 0:
            stats = compute_fit_statistics(ranks, frequencies)
            row = {"dataset": dataset_name, **stats}
            static_rows.append(row)
            static_stats_by_dataset[dataset_name] = row
        print(f"  Loaded {len(attachments_df)} attachments and found {len(ranks)} ranked nodes")

    if not all_rank_freqs:
        print("Error: no datasets processed successfully")
        raise SystemExit(1)

    max_rank = max(len(ranks) for ranks, _ in all_rank_freqs.values())
    rank_freq_data = {"rank": np.arange(1, max_rank + 1)}
    for dataset_name, (_, frequencies) in all_rank_freqs.items():
        padded = np.full(max_rank, np.nan)
        padded[: len(frequencies)] = frequencies
        rank_freq_data[dataset_name] = padded
    rank_freq_df = pd.DataFrame(rank_freq_data)
    rank_freq_df.to_csv(analysis_dir / "rfc_rank_frequencies.csv", index=False)

    cum_data = {"rank": rank_freq_df["rank"].values}
    for dataset_name in [c for c in rank_freq_df.columns if c != "rank"]:
        frequencies = rank_freq_df[dataset_name].values
        cum = np.nancumsum(np.nan_to_num(frequencies, nan=0.0))
        cum[np.isnan(frequencies)] = np.nan
        cum_data[dataset_name] = cum
    pd.DataFrame(cum_data).to_csv(analysis_dir / "rfc_cumulative_rank_frequencies.csv", index=False)

    if static_rows:
        static_df = pd.DataFrame(static_rows).sort_values("dataset").reset_index(drop=True)
        static_df.to_csv(analysis_dir / "rfc_static_analysis.csv", index=False)
        summary_df = static_df[
            [
                "dataset",
                "linear_nll",
                "exponential_nll",
                "power_nll",
                "bounded_power_nll",
                "best_fit",
                "power_alpha",
                "bounded_power_alpha",
            ]
        ].rename(
            columns={
                "power_alpha": "power_law_exponent_alpha",
                "bounded_power_alpha": "bounded_power_law_exponent_alpha",
            }
        )
        summary_df.to_csv(analysis_dir / "rfc_static_analysis_summary.csv", index=False)
        summary_df_latex = summary_df.rename(
            columns={
                "dataset": "Dataset",
                "linear_nll": "Lin. NLL",
                "exponential_nll": "Exp. NLL",
                "power_nll": "PL NLL",
                "bounded_power_nll": "BPL NLL",
                "best_fit": "Best fit",
                "power_law_exponent_alpha": r"PL \alpha",
                "bounded_power_law_exponent_alpha": r"BPL \alpha",
            }
        ).copy()
        summary_df_latex["Best fit"] = summary_df_latex["Best fit"].replace(
            {
                "linear": "Lin.",
                "exponential": "Exp.",
                "power_law": "PL",
                "bounded_power_law": "BPL",
            }
        )
        summary_latex = summary_df_latex.to_latex(
            index=False,
            float_format="%.2f",
            escape=False,
        )
        summary_latex += (
            "\n"
            r"\noindent\textit{Note: Lin.: linear \quad Exp.: exponential \quad "
            r"PL: power-law \quad BPL: bounded power-law \quad NLL: negative log-likelihood.}"
            "\n"
        )
        (analysis_dir / "rfc_static_analysis_summary.tex").write_text(summary_latex)

    create_all_plots(rank_freq_df, analysis_dir)
    create_loglog_subfigure_plot(rank_freq_df, analysis_dir, static_stats_by_dataset)
    create_loglog_subfigure_plot(
        rank_freq_df,
        analysis_dir,
        static_stats_by_dataset,
        output_filename="rfc_all_loglog_subfigures_aligned.pdf",
        aligned_axes=True,
        aligned_axis_cap=1e5,
    )
    for dataset_name, _ in input_pairs:
        if dataset_name in all_rank_freqs and dataset_name in static_stats_by_dataset:
            create_dataset_plots(
                rank_freq_df,
                analysis_dir,
                dataset_name,
                static_stats_by_dataset[dataset_name],
            )
            create_dataset_unfitted_plots(
                rank_freq_df,
                analysis_dir,
                dataset_name,
            )
    print(f"\nAll RFC outputs saved to: {analysis_dir}")


__all__ = [
    "compute_fit_statistics",
    "create_all_plots",
    "create_dataset_plots",
    "create_loglog_rfc_plot",
    "create_loglog_subfigure_plot",
    "extract_rank_frequencies",
    "extract_rank_frequencies_from_counts",
    "extract_rank_frequencies_from_event_log",
]


if __name__ == "__main__":
    main()
