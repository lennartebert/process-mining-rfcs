"""Log-log PDF/CCDF plotting helpers for power-law fits."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import powerlaw


def plot_pdf_loglog(
    frequencies: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Write an empirical PDF of frequencies on log-log axes via ``powerlaw``."""
    frequencies = np.asarray(frequencies, dtype=float)
    frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 0)]
    # powerlaw.plot_pdf needs enough distinct observations to bin.
    if frequencies.size < 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        powerlaw.plot_pdf(
            frequencies, ax=ax, color="red", linewidth=1.8, label="Observed"
        )
    except (IndexError, ValueError, RuntimeError):
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_pdf_loglog(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
    fit: powerlaw.Fit | None = None,
) -> None:
    """Write PDF of continuous magnitudes (powerlaw log bins + optional fit)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if fit is not None:
            fit.plot_pdf(ax=ax, color="red", linewidth=1.8, label="Observed")
            fit.power_law.plot_pdf(
                ax=ax, color="black", linestyle="--", linewidth=1.5, label="Power law"
            )
        else:
            powerlaw.plot_pdf(
                values, ax=ax, color="red", linewidth=1.8, label="Observed"
            )
    except (IndexError, ValueError, RuntimeError, AttributeError):
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Value")
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_continuous_ccdf_loglog(
    values: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
    fit: powerlaw.Fit | None = None,
) -> None:
    """Write CCDF of continuous magnitudes via ``powerlaw``."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if fit is not None:
            fit.plot_ccdf(ax=ax, color="red", linewidth=1.8, label="Observed")
            fit.power_law.plot_ccdf(
                ax=ax, color="black", linestyle="--", linewidth=1.5, label="Power law"
            )
        else:
            powerlaw.plot_ccdf(
                values, ax=ax, color="red", linewidth=1.8, label="Observed"
            )
    except (IndexError, ValueError, RuntimeError, AttributeError):
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Value")
    ax.set_ylabel("CCDF")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def _plot_alternative_overlay(
    fit: powerlaw.Fit,
    ax,
    *,
    kind: str,
    alternative_name: str | None,
) -> None:
    """Overlay power-law and optional alternative PDF/CCDF curves on ``ax``."""
    if kind == "pdf":
        fit.power_law.plot_pdf(
            ax=ax, color="black", linestyle="--", linewidth=1.5, label="Power law"
        )
    else:
        fit.power_law.plot_ccdf(
            ax=ax, color="black", linestyle="--", linewidth=1.5, label="Power law"
        )
    if not alternative_name:
        return
    try:
        alt = getattr(fit, alternative_name)
    except Exception:
        return
    plot_fn = alt.plot_pdf if kind == "pdf" else alt.plot_ccdf
    try:
        plot_fn(
            ax=ax,
            color="tab:blue",
            linestyle=":",
            linewidth=1.5,
            label=alternative_name.replace("_", " "),
        )
    except Exception:
        return


def plot_fit_pdf_with_alternative(
    values: np.ndarray,
    output_path: Path,
    *,
    fit: powerlaw.Fit,
    alternative_name: str | None = None,
    title: str | None = None,
    xlabel: str = "Frequency",
) -> None:
    """Log-log PDF: empirical + power law + optional alternative distribution."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        fit.plot_pdf(ax=ax, color="red", linewidth=1.8, label="Observed")
        _plot_alternative_overlay(
            fit, ax, kind="pdf", alternative_name=alternative_name
        )
    except (IndexError, ValueError, RuntimeError, AttributeError):
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_fit_ccdf_with_alternative(
    values: np.ndarray,
    output_path: Path,
    *,
    fit: powerlaw.Fit,
    alternative_name: str | None = None,
    title: str | None = None,
    xlabel: str = "Frequency",
) -> None:
    """Log-log CCDF: empirical + power law + optional alternative distribution."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        fit.plot_ccdf(ax=ax, color="red", linewidth=1.8, label="Observed")
        _plot_alternative_overlay(
            fit, ax, kind="ccdf", alternative_name=alternative_name
        )
    except (IndexError, ValueError, RuntimeError, AttributeError):
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CCDF")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def create_multiplot(
    dataset_fits: Sequence[tuple[str, powerlaw.Fit]],
    output_path: Path,
) -> None:
    """Create one multi-panel log-log PDF plot with fitted lines."""
    if not dataset_fits:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(dataset_fits) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows), squeeze=False
    )
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
            fit.power_law.plot_pdf(
                ax=ax, color="black", linestyle="--", linewidth=2.6, label="Power law fit"
            )
            fit.lognormal.plot_pdf(
                ax=ax, color="red", linestyle=":", linewidth=2.6, label="Lognormal fit"
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(left=1)
        except Exception:
            ax.text(
                0.5,
                0.5,
                "fit unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
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
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.004),
        )
        fig.tight_layout(rect=(0, 0.033, 1, 1))
    else:
        fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
