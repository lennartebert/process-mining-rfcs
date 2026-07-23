"""RFC plotting helpers for scripts and notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .fit_models import DEFAULT_BOUNDED_MIN_FREQUENCY, compute_powerlaw_statistics


def _apply_loglog_axes_style(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Apply the same log-log axis styling used in RFC subfigure plots."""
    local_axis_max = max(1.0, float(np.nanmax([np.nanmax(x), np.nanmax(y)])))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=0.9, right=local_axis_max)
    ax.set_ylim(bottom=0.9, top=local_axis_max)
    ax.grid(True, alpha=0.3)


def _format_nll(nll: float) -> str:
    """Format NLL for figure annotations: integer with comma thousands separator."""
    if not np.isfinite(nll):
        return "—"
    return f"{round(nll):,}"


def _add_powerlaw_metrics_annotations(
    ax: plt.Axes,
    power_alpha: float,
    power_nll: float,
    bounded_power_alpha: float,
    bounded_power_nll: float,
) -> None:
    """Add in-plot alpha/NLL annotations for unbounded and bounded power-law fits."""
    x0 = 0.50
    y0 = 0.96
    line_dx = 0.07
    row_gap = 0.08
    text_x = x0 + line_dx + 0.015
    text_bbox = {"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": 0.72}

    ax.plot([x0, x0 + line_dx], [y0, y0], transform=ax.transAxes, color="black", linestyle=":", linewidth=2.0)
    ax.text(
        text_x,
        y0,
        rf"$\alpha$: {power_alpha:.2f}, NLL: {_format_nll(power_nll)}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        bbox=text_bbox,
    )
    y1 = y0 - row_gap
    ax.plot([x0, x0 + line_dx], [y1, y1], transform=ax.transAxes, color="#1f77b4", linestyle="--", linewidth=1.8)
    ax.text(
        text_x,
        y1,
        rf"$\alpha$: {bounded_power_alpha:.2f}, NLL: {_format_nll(bounded_power_nll)}",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        bbox=text_bbox,
    )


def normalize_for_plot(values: np.ndarray) -> np.ndarray:
    """Normalize values to relative frequencies for cross-dataset plotting."""
    arr = np.asarray(values, dtype=float)
    total = np.nansum(arr)
    if total <= 0:
        return arr
    return arr / total


def create_loglog_rfc_plot(
    ranks: np.ndarray,
    frequencies: np.ndarray,
    bounded_min_frequency: float = DEFAULT_BOUNDED_MIN_FREQUENCY,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_powerlaw_fits: bool = True,
    x_max: float | None = None,
    y_max: float | None = None,
) -> Tuple[plt.Figure, plt.Axes, dict]:
    """Create one log-log RFC plot with unbounded and bounded power-law fits."""
    ranks_arr = np.asarray(ranks, dtype=float)
    frequencies_arr = np.asarray(frequencies, dtype=float)
    mask = (
        np.isfinite(ranks_arr)
        & np.isfinite(frequencies_arr)
        & (ranks_arr > 0)
        & (frequencies_arr > 0)
    )
    x = ranks_arr[mask]
    y = frequencies_arr[mask]
    stats: dict = {}

    if ax is None:
        fig = Figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    if x.size == 0:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.set_title(title or "RFC log-log")
        ax.grid(True, alpha=0.3)
        return fig, ax, stats

    stats = compute_powerlaw_statistics(x, y, bounded_min_frequency)

    ax.plot(x, y, "-", linewidth=1.8, color="red", alpha=0.9, label="Observed data")
    if show_powerlaw_fits and np.isfinite(stats["power_c"]) and np.isfinite(stats["power_alpha"]):
        y_fit = np.maximum(stats["power_c"] * (x ** (-stats["power_alpha"])), 1e-12)
        ax.plot(x, y_fit, ":", linewidth=2.0, color="black", alpha=0.95, label="Power-law fit")
    if show_powerlaw_fits and np.isfinite(stats["bounded_power_c"]) and np.isfinite(stats["bounded_power_alpha"]):
        y_fit_bounded = np.maximum(
            stats["bounded_power_c"] * (x ** (-stats["bounded_power_alpha"])),
            1e-12,
        )
        bounded_mask = y >= float(bounded_min_frequency)
        if bounded_mask.any():
            ax.plot(
                x[bounded_mask],
                y_fit_bounded[bounded_mask],
                "--",
                linewidth=1.8,
                color="#1f77b4",
                alpha=0.95,
                label=f"Power-law fit (freq>={bounded_min_frequency:g})",
            )

    data_axis_max = max(1.0, float(np.nanmax([np.nanmax(x), np.nanmax(y)])))
    axis_x_max = float(x_max) if x_max is not None else data_axis_max
    axis_y_max = float(y_max) if y_max is not None else data_axis_max
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=0.9, right=axis_x_max)
    ax.set_ylim(bottom=0.9, top=axis_y_max)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title(title or "RFC log-log")
    ax.grid(True, alpha=0.3)

    if show_powerlaw_fits:
        _add_powerlaw_metrics_annotations(
            ax=ax,
            power_alpha=float(stats["power_alpha"]),
            power_nll=float(stats["power_nll"]),
            bounded_power_alpha=float(stats["bounded_power_alpha"]),
            bounded_power_nll=float(stats["bounded_power_nll"]),
        )
    ax.legend(loc="lower left", fontsize=9)
    return fig, ax, stats


def create_all_plots(rank_freq_df: pd.DataFrame, output_dir: Path) -> None:
    """Create aggregate RFC plots across all datasets and axis scalings."""
    rank_col = rank_freq_df["rank"].values
    dataset_cols = [col for col in rank_freq_df.columns if col != "rank"]
    scales = [("linear", False, False), ("logy", False, True), ("logx", True, False), ("loglog", True, True)]

    for scale_name, log_x, log_y in scales:
        fig, ax = plt.subplots(figsize=(10, 6))
        for col in dataset_cols:
            frequencies = rank_freq_df[col].values
            mask = ~np.isnan(frequencies)
            if mask.any():
                relative = normalize_for_plot(frequencies[mask])
                ax.plot(rank_col[mask], relative, "-", linewidth=1.0, alpha=0.6, label=col)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        plt.tight_layout()
        out_path = output_dir / f"rfc_all_{scale_name}.pdf"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def create_loglog_subfigure_plot(
    rank_freq_df: pd.DataFrame,
    output_dir: Path,
    fit_stats_by_dataset: Dict[str, dict],
    output_filename: str = "rfc_all_loglog_subfigures.pdf",
    aligned_axes: bool = False,
    aligned_axis_cap: float = 1e5,
) -> None:
    """Create a multi-panel log-log RFC figure with power-law overlays."""
    rank_col = rank_freq_df["rank"].values
    dataset_cols = [col for col in rank_freq_df.columns if col != "rank"]
    if not dataset_cols:
        return
    aligned_axis_max = 1.0
    if aligned_axes:
        max_candidates = []
        for dataset_name in dataset_cols:
            frequencies = rank_freq_df[dataset_name].values
            mask = (~np.isnan(frequencies)) & np.isfinite(rank_col) & np.isfinite(frequencies) & (rank_col > 0) & (frequencies > 0)
            if mask.any():
                max_candidates.append(float(np.nanmax(rank_col[mask])))
                max_candidates.append(float(np.nanmax(frequencies[mask])))
        if max_candidates:
            aligned_axis_max = max(1.0, min(float(aligned_axis_cap), max(max_candidates)))
    n_cols = 4
    n_rows = int(np.ceil(len(dataset_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.8 * n_rows), squeeze=False)
    flat_axes = axes.flatten()
    for idx, dataset_name in enumerate(dataset_cols):
        ax = flat_axes[idx]
        frequencies = rank_freq_df[dataset_name].values
        mask = (~np.isnan(frequencies)) & (frequencies > 0)
        if mask.any():
            ranks = rank_col[mask]
            observed = frequencies[mask]
            ax.plot(ranks, observed, "-", linewidth=1.8, color="red", alpha=0.9)
            stats = fit_stats_by_dataset.get(dataset_name, {})
            power_c = float(stats.get("power_c", np.nan))
            power_alpha = float(stats.get("power_alpha", np.nan))
            if np.isfinite(power_c) and np.isfinite(power_alpha):
                y_fit = np.maximum(power_c * (ranks ** (-power_alpha)), 1e-12)
                ax.plot(ranks, y_fit, ":", linewidth=2.0, color="black", alpha=0.95)
            bounded_power_c = float(stats.get("bounded_power_c", np.nan))
            bounded_power_alpha = float(stats.get("bounded_power_alpha", np.nan))
            bounded_min_freq = float(stats.get("bounded_power_min_frequency", np.nan))
            power_nll = float(stats.get("power_nll", np.nan))
            bounded_power_nll = float(stats.get("bounded_power_nll", np.nan))
            if np.isfinite(bounded_power_c) and np.isfinite(bounded_power_alpha):
                y_fit_bounded = np.maximum(bounded_power_c * (ranks ** (-bounded_power_alpha)), 1e-12)
                interval_mask = np.isfinite(observed) & (observed > 0)
                if np.isfinite(bounded_min_freq):
                    interval_mask &= observed >= bounded_min_freq
                if interval_mask.any():
                    ax.plot(ranks[interval_mask], y_fit_bounded[interval_mask], "--", linewidth=1.8, color="#1f77b4")
            if aligned_axes:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(left=0.9, right=aligned_axis_max)
                ax.set_ylim(bottom=0.9, top=aligned_axis_max)
                ax.grid(True, alpha=0.3)
            else:
                _apply_loglog_axes_style(ax, ranks, observed)
            _add_powerlaw_metrics_annotations(
                ax=ax,
                power_alpha=power_alpha,
                power_nll=power_nll,
                bounded_power_alpha=bounded_power_alpha,
                bounded_power_nll=bounded_power_nll,
            )
        row_idx = idx // n_cols
        col_idx = idx % n_cols
        if row_idx == (n_rows - 1):
            ax.set_xlabel("Rank", fontsize=12)
        if col_idx == 0:
            ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(dataset_name, fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
    for idx in range(len(dataset_cols), len(flat_axes)):
        flat_axes[idx].set_visible(False)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="red", linewidth=1.8, linestyle="-", label="Observed data"),
        Line2D([0], [0], color="black", linewidth=2.0, linestyle=":", label="Power-law fit"),
        Line2D([0], [0], color="#1f77b4", linewidth=1.8, linestyle="--", label="Power-law fit (freq>=2)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015))
    plt.tight_layout(rect=(0, 0.035, 1, 1))
    out_path = output_dir / output_filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_dataset_plots(
    rank_freq_df: pd.DataFrame,
    base_output_dir: Path,
    dataset_name: str,
    fit_stats: dict,
) -> None:
    """Create per-dataset RFC plots (linear/log variants) with fitted curves."""
    dataset_dir = base_output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rank_col = rank_freq_df["rank"].values
    freqs = rank_freq_df[dataset_name].values
    mask = ~np.isnan(freqs)
    ranks = rank_col[mask]
    y = freqs[mask]
    if len(ranks) == 0:
        return

    linear_a = float(fit_stats["linear_a"])
    linear_b = float(fit_stats["linear_b"])
    exp_a = float(fit_stats["exponential_a"])
    exp_b = float(fit_stats["exponential_b"])
    power_c = float(fit_stats["power_c"])
    power_alpha = float(fit_stats["power_alpha"])
    bounded_power_c = float(fit_stats.get("bounded_power_c", np.nan))
    bounded_power_alpha = float(fit_stats.get("bounded_power_alpha", np.nan))
    x_plot = np.linspace(ranks.min(), ranks.max(), 1000)

    scales = [
        ("linear", False, False),
        # ("logy", False, True),  # Temporarily disabled per current analysis request.
        # ("logx", True, False),  # Temporarily disabled per current analysis request.
        ("loglog", True, True),
    ]
    for scale_name, log_x, log_y in scales:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ranks, y, "-", linewidth=2.0, color="red", alpha=0.9, label="Observed")
        y_linear = linear_a * x_plot + linear_b
        y_exp = exp_a * np.exp(exp_b * x_plot) if np.isfinite(exp_a) and np.isfinite(exp_b) else None
        y_power = power_c * (x_plot ** (-power_alpha)) if np.isfinite(power_c) and np.isfinite(power_alpha) else None
        y_power_bounded = (
            bounded_power_c * (x_plot ** (-bounded_power_alpha))
            if np.isfinite(bounded_power_c) and np.isfinite(bounded_power_alpha)
            else None
        )
        if log_y:
            y_linear = np.maximum(y_linear, 1e-12)
            if y_exp is not None:
                y_exp = np.maximum(y_exp, 1e-12)
            if y_power is not None:
                y_power = np.maximum(y_power, 1e-12)
            if y_power_bounded is not None:
                y_power_bounded = np.maximum(y_power_bounded, 1e-12)

        ax.plot(x_plot, y_linear, "--", linewidth=2.0, alpha=0.75, label="Linear fit")
        if y_exp is not None:
            ax.plot(x_plot, y_exp, "--", linewidth=2.0, alpha=0.75, label="Exponential fit")
        if y_power is not None:
            ax.plot(x_plot, y_power, "--", linewidth=2.0, alpha=0.75, label="Power-law fit")
        if y_power_bounded is not None:
            ax.plot(
                x_plot,
                y_power_bounded,
                "--",
                linewidth=2.0,
                alpha=0.75,
                color="#1f77b4",
                label="Power-law fit (freq>=2)",
            )
        ax.set_xlabel("Rank")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=10)
        plt.tight_layout()
        out_path = dataset_dir / f"rfc_fitted_{scale_name}.pdf"
        fig.savefig(out_path, dpi=360, bbox_inches="tight")
        plt.close(fig)


def create_dataset_unfitted_plots(
    rank_freq_df: pd.DataFrame,
    base_output_dir: Path,
    dataset_name: str,
) -> None:
    """Create per-dataset RFC plots without fitted lines (linear and log-log)."""
    dataset_dir = base_output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rank_col = rank_freq_df["rank"].values
    freqs = rank_freq_df[dataset_name].values
    mask = (~np.isnan(freqs)) & (freqs > 0)
    ranks = rank_col[mask]
    y = freqs[mask]
    if len(ranks) == 0:
        return

    # Linear scale without fitted lines.
    fig_linear, ax_linear = plt.subplots(figsize=(8, 5))
    ax_linear.plot(ranks, y, "-", linewidth=2.0, color="red", alpha=0.9, label="Observed")
    ax_linear.set_xlabel("Rank")
    ax_linear.set_ylabel("Frequency")
    ax_linear.grid(True, alpha=0.3)
    ax_linear.legend(fontsize=10)
    fig_linear.tight_layout()
    fig_linear.savefig(dataset_dir / "rfc_linear.pdf", dpi=360, bbox_inches="tight")
    plt.close(fig_linear)

    # Log-log scale without fitted lines, using subplot styling.
    fig_loglog, ax_loglog = plt.subplots(figsize=(8, 5))
    ax_loglog.plot(ranks, y, "-", linewidth=1.8, color="red", alpha=0.9, label="Observed")
    _apply_loglog_axes_style(ax_loglog, ranks, y)
    ax_loglog.set_xlabel("Rank")
    ax_loglog.set_ylabel("Frequency")
    ax_loglog.legend(fontsize=10)
    fig_loglog.tight_layout()
    fig_loglog.savefig(dataset_dir / "rfc_loglog.pdf", dpi=360, bbox_inches="tight")
    plt.close(fig_loglog)
