"""Dynamic preferential-attachment (PA) analysis from attachment CSV inputs.

This implements the measurement procedure from:

  Preferential Attachment in Online Networks: Measurement and Explanations
  (Kunegis, Blattner, and Moser, 2013; Web Science / arXiv:1303.6271v1)

Attachment rows use columns: attachment_time, attachment_index, node_id.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import RESULTS_DIR
from utils.io import load_attachments, parse_dataset_inputs

DEFAULT_LAMBDA_REG = 0.1
LABEL_DIRECTION_DEFAULT = "top_right"
LABEL_DIRECTION_EXCEPTIONS = {
    "BPIC20_rfp": "bottom_left",
    "BPIC15_1": "top_right",
    "BPIC15_2": "bottom_left",
    "BPIC15_4": "bottom_right",
    "BPIC15_5": "top_left",
    "RTFMP": "top",
    "BPIC17": "left",
    "BPIC12": "right",
    "SEPSIS": "left",
    "BPIC14": "right",
    "BPIC20_tpd": "left",
    "MOBIS": "left",
    "ITHD": "right",
    "BPIC20_dd": "top",
    "BPIC11": "top",
    "BPIC18": "right",
    "BPIC13_i": "right",
    "BPIC20_ptc": "top",
    "BPIC20_id": "right",
    "ACCRE": "right",
}
LABEL_DIRECTION_OFFSETS = {
    "top_right": (9, 9),
    "top_left": (-9, 9),
    "bottom_right": (9, -9),
    "bottom_left": (-9, -9),
    "right": (12, 0),
    "left": (-12, 0),
    "top": (0, 12),
}


def _split_old_new(df: pd.DataFrame, split_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split edges into E1 (old) and E\\E1 (new) by temporal order."""
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be between 0 and 1")
    split_idx = int(np.floor(len(df) * split_ratio))
    old_df = df.iloc[:split_idx].copy()
    new_df = df.iloc[split_idx:].copy()
    return old_df, new_df


NodeSetMode = Literal["old_only", "union"]


def _compute_d1_d2(old_df: pd.DataFrame, new_df: pd.DataFrame, node_set_mode: NodeSetMode) -> pd.DataFrame:
    """Compute d1 (degree in E1) and d2 (attachments after t1).

    Modes:
    - old_only: keep only nodes present before t1 (historical behavior).
    - union: use nodes from t1 union t2; nodes first appearing after t1 get d1=0.
    """
    d1 = old_df["node_id"].value_counts()
    d2 = new_df["node_id"].value_counts()
    if node_set_mode == "old_only":
        all_nodes = pd.Index(d1.index)
    elif node_set_mode == "union":
        all_nodes = pd.Index(d1.index).union(pd.Index(d2.index))
    else:
        raise ValueError(f"Unsupported node_set_mode: {node_set_mode}")
    result = pd.DataFrame({"node_id": all_nodes})
    result["d1"] = result["node_id"].map(d1).fillna(0).astype(float)
    result["d2"] = result["node_id"].map(d2).fillna(0).astype(float)
    return result


def _fit_alpha_beta_least_squares(d_df: pd.DataFrame, lambda_reg: float, x_offset: float) -> Tuple[float, float, float]:
    """Eq. (2) least squares: y ~ alpha + beta*log(1+d1) with y = log(lambda + d2)."""
    if lambda_reg < 0:
        raise ValueError("lambda_reg must be >= 0")
    if x_offset < 0:
        raise ValueError("x_offset must be >= 0")
    x_raw = x_offset + d_df["d1"].values
    y_raw = lambda_reg + d_df["d2"].values
    valid_mask = (x_raw > 0) & (y_raw > 0)
    if not np.any(valid_mask):
        raise ValueError(
            "No valid points for log-log fit. "
            "With lambda-reg=0 and/or x-offset=0, all transformed values must be strictly positive."
        )
    x = np.log(x_raw[valid_mask])
    y = np.log(y_raw[valid_mask])
    design = np.column_stack([np.ones_like(x), x])
    params, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    alpha = float(params[0])
    beta = float(params[1])
    residual = alpha + beta * x - y
    epsilon = float(np.exp(np.sqrt(np.mean(residual**2))))
    return alpha, beta, epsilon


def _make_scatter_plot(
    d_df: pd.DataFrame, alpha: float, beta: float, lambda_reg: float, x_offset: float, output_path: Path
) -> None:
    """Log-log fit plot with grouped boxplots by regularized degree."""
    x_raw = x_offset + d_df["d1"].values
    y_raw = lambda_reg + d_df["d2"].values
    valid_mask = (x_raw > 0) & (y_raw > 0)
    x_plot = x_raw[valid_mask]
    y_plot = y_raw[valid_mask]
    order = np.argsort(x_plot)
    x_sorted = x_plot[order]
    y_fit_sorted = np.exp(alpha) * (x_sorted**beta)

    unique_x = np.unique(x_plot)
    grouped_x_positions: List[float] = []
    grouped_y_values: List[np.ndarray] = []
    if unique_x.size <= 200:
        for x_val in unique_x:
            vals = y_plot[x_plot == x_val]
            if vals.size > 0:
                grouped_x_positions.append(float(x_val))
                grouped_y_values.append(vals)
    else:
        quantile_edges = np.unique(np.quantile(x_plot, np.linspace(0, 1, 121)))
        for i in range(len(quantile_edges) - 1):
            lo = quantile_edges[i]
            hi = quantile_edges[i + 1]
            if i == len(quantile_edges) - 2:
                mask = (x_plot >= lo) & (x_plot <= hi)
            else:
                mask = (x_plot >= lo) & (x_plot < hi)
            vals = y_plot[mask]
            if vals.size > 0:
                x_mid = float(np.sqrt(max(lo, 1e-12) * max(hi, 1e-12)))
                grouped_x_positions.append(x_mid)
                grouped_y_values.append(vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    if grouped_y_values:
        widths = [max(pos * 0.08, 0.03) for pos in grouped_x_positions]
        ax.boxplot(
            grouped_y_values,
            positions=grouped_x_positions,
            widths=widths,
            showmeans=False,
            manage_ticks=False,
            patch_artist=True,
            boxprops={"facecolor": "#a6cee3", "alpha": 0.45, "edgecolor": "#1f78b4"},
            whiskerprops={"color": "#1f78b4", "alpha": 0.8},
            capprops={"color": "#1f78b4", "alpha": 0.8},
            medianprops={"color": "#08306b", "linewidth": 1.5},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.2, "markeredgecolor": "#666666"},
        )
    (fit_handle,) = ax.plot(
        x_sorted,
        y_fit_sorted,
        color="red",
        linewidth=2,
        label=f"fit: exp(alpha)*x^beta, exp(alpha)={np.exp(alpha):.6f}, beta={beta:.6f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Regularized degree at t1: {x_offset:g} + d1(u)")
    ax.set_ylabel("Regularized new edges after t1: lambda + d2(u)")
    ax.grid(True, alpha=0.3)
    box_proxy = Patch(
        facecolor="#a6cee3",
        edgecolor="#1f78b4",
        alpha=0.45,
        label="Boxplot per degree bin (box=IQR, center line=median)",
    )
    ax.legend(handles=[box_proxy, fit_handle], loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_figure_beta_vs_epsilon(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create figure: beta vs epsilon."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary_df["beta"], summary_df["epsilon"], alpha=0.8)
    for _, row in summary_df.iterrows():
        ax.text(row["beta"], row["epsilon"], str(row["dataset"]), fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Preferential attachment exponent (beta)")
    ax.set_ylabel("Root-mean-square logarithmic error (epsilon)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_figure_beta_vs_power_law(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Create figure: beta vs static power-law exponent alpha."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = summary_df[np.isfinite(summary_df["power_law_exponent_alpha"])]
    ax.scatter(valid["power_law_exponent_alpha"], valid["beta"], alpha=0.8)
    for _, row in valid.iterrows():
        dataset_name = str(row["dataset"])
        direction = LABEL_DIRECTION_EXCEPTIONS.get(dataset_name, LABEL_DIRECTION_DEFAULT)
        dx, dy = LABEL_DIRECTION_OFFSETS.get(direction, LABEL_DIRECTION_OFFSETS[LABEL_DIRECTION_DEFAULT])
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
        ax.annotate(
            dataset_name,
            xy=(row["power_law_exponent_alpha"], row["beta"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.65},
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.8, "alpha": 0.8},
        )
    ax.set_xlabel("Power law exponent alpha (static)")
    ax.set_ylabel("Preferential attachment exponent (beta)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI for dynamic PA analysis."""
    parser = argparse.ArgumentParser(description="Dynamic preferential-attachment analysis from attachment CSV files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Dataset/attachments pairs: <dataset>=<attachments_csv_path>",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results)",
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        default="dynamic_pa_analysis",
        help="Subfolder name under the output root for this run (e.g. same as concept or a batch label)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.75,
        help="Temporal split ratio for t1 (default: 0.75, Kunegis et al. 2013)",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=DEFAULT_LAMBDA_REG,
        help="Regularization lambda (default: 0.1, Kunegis et al. 2013)",
    )
    parser.add_argument(
        "--x-offset",
        type=float,
        default=1.0,
        help="Offset for x-axis transform and fit: x = x_offset + d1(u) (default: 1.0)",
    )
    parser.add_argument(
        "--node-set-mode",
        type=str,
        default="old_only",
        choices=["old_only", "union"],
        help=(
            "Node population used for d1/d2 table: "
            "'old_only' keeps only nodes seen before t1; "
            "'union' includes nodes first seen after t1 with d1=0."
        ),
    )
    parser.add_argument(
        "--static-summary-path",
        type=str,
        default=None,
        help=(
            "Optional path to static_analysis_summary.csv. "
            "When provided, creates figure_beta_vs_power_law_exponent.pdf using "
            "power_law_exponent_alpha from static analysis."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Run dynamic analysis for selected attachment inputs."""
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)
    analysis_root = output_root / args.analysis_name
    analysis_root.mkdir(parents=True, exist_ok=True)

    try:
        input_pairs = parse_dataset_inputs(args.inputs)
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc

    summary_rows: List[Dict[str, float]] = []
    for dataset_name, attachments_path in input_pairs:
        if not attachments_path.exists():
            print(f"Skipping {dataset_name}: missing {attachments_path}")
            continue

        print(f"\nProcessing {dataset_name}...")
        try:
            df = load_attachments(attachments_path)
        except ValueError as exc:
            print(f"Skipping {dataset_name}: {exc}")
            continue
        old_df, new_df = _split_old_new(df, split_ratio=args.split_ratio)
        d_df = _compute_d1_d2(old_df, new_df, node_set_mode=args.node_set_mode)
        try:
            alpha, beta, epsilon = _fit_alpha_beta_least_squares(
                d_df, lambda_reg=args.lambda_reg, x_offset=args.x_offset
            )
        except ValueError as exc:
            print(f"Skipping {dataset_name}: {exc}")
            continue

        dataset_dir = analysis_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        d_df.to_csv(dataset_dir / "d1_d2_table.csv", index=False)
        _make_scatter_plot(
            d_df,
            alpha=alpha,
            beta=beta,
            lambda_reg=args.lambda_reg,
            x_offset=args.x_offset,
            output_path=dataset_dir / "dynamic_fit.pdf",
        )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "num_edges_total": float(len(df)),
                "num_edges_old": float(len(old_df)),
                "num_edges_new": float(len(new_df)),
                "num_nodes": float(d_df["node_id"].nunique()),
                "split_ratio": float(args.split_ratio),
                "lambda_reg": float(args.lambda_reg),
                "x_offset": float(args.x_offset),
                "node_set_mode": str(args.node_set_mode),
                "alpha": float(alpha),
                "beta": float(beta),
                "epsilon": float(epsilon),
            }
        )
        print(f"  node_set_mode={args.node_set_mode}, beta={beta:.6f}, epsilon={epsilon:.6f}")
        print(f"  Saved: {dataset_dir}")

    if not summary_rows:
        print("\nNo datasets processed.")
        raise SystemExit(1)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = analysis_root / "dynamic_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    _make_figure_beta_vs_epsilon(summary_df, analysis_root / "figure_beta_vs_epsilon.pdf")

    if args.static_summary_path:
        static_summary_path = Path(args.static_summary_path)
        if not static_summary_path.exists():
            print(f"Warning: static summary file not found: {static_summary_path}")
        else:
            static_summary_df = pd.read_csv(static_summary_path)
            required_cols = {"dataset", "power_law_exponent_alpha"}
            missing_cols = required_cols.difference(static_summary_df.columns)
            if missing_cols:
                print(
                    f"Warning: static summary missing required columns {sorted(missing_cols)}; "
                    "skipping figure_beta_vs_power_law_exponent.pdf"
                )
            else:
                merged = summary_df.merge(
                    static_summary_df[["dataset", "power_law_exponent_alpha"]],
                    on="dataset",
                    how="left",
                )
                _make_figure_beta_vs_power_law(merged, analysis_root / "figure_beta_vs_power_law_exponent.pdf")
                print("Created: figure_beta_vs_power_law_exponent.pdf")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
