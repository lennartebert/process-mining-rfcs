"""Log-log RFC plots for attribute (and other count) distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.rfc.plotting import _apply_loglog_axes_style


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
