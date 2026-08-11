"""Log-log RFC and PDF/CCDF plots for attribute distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import powerlaw

from .plotting import _apply_loglog_axes_style


def plot_attribute_rfc_loglog(
    frequencies: np.ndarray,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Write a log-log rank-frequency plot for positive integer counts."""
    frequencies = np.asarray(frequencies, dtype=float)
    frequencies = frequencies[np.isfinite(frequencies) & (frequencies > 0)]
    if frequencies.size < 1:
        return
    ranks = np.arange(1, frequencies.size + 1, dtype=float)
    y = np.sort(frequencies)[::-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, y, "-", linewidth=1.8, color="red", alpha=0.9, label="Observed")
    _apply_loglog_axes_style(ax, ranks, y)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)


def plot_attribute_pdf_loglog(
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
