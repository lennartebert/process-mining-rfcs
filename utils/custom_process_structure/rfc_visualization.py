"""RFC rendering helpers for simulated event logs."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils.rfc import (
    DEFAULT_BOUNDED_MIN_FREQUENCY,
    create_loglog_rfc_plot,
    extract_rank_frequencies_from_event_log,
)


def build_plot_title(settings: dict) -> str:
    """Build a compact RFC plot title for one experiment setting."""
    q = settings["q"]
    alpha_part = ""
    if q == "power_law" and settings.get("alpha") is not None:
        alpha_part = f" (alpha={settings['alpha']:.2f})"
    return (
        f"N={settings['N']}, n={settings['n']}, o={settings['o']}, q={q}{alpha_part}, "
        f"m={settings['m']}, p={settings['p']:.2f}"
    )


def render_rfc_from_event_log(
    event_log: pd.DataFrame,
    settings: dict,
    save_pdf_path: Path | None = None,
    show_powerlaw_fits: bool = True,
    suppress_title: bool = False,
    x_max: float | None = None,
    y_max: float | None = None,
) -> dict:
    """Render RFC figure from an event log and return image bytes and fit stats."""
    ranks, frequencies = extract_rank_frequencies_from_event_log(event_log)
    plot_title = build_plot_title(settings)
    fig, ax, fit_stats = create_loglog_rfc_plot(
        ranks=ranks,
        frequencies=frequencies,
        bounded_min_frequency=DEFAULT_BOUNDED_MIN_FREQUENCY,
        title=plot_title,
        show_powerlaw_fits=show_powerlaw_fits,
        x_max=x_max,
        y_max=y_max,
    )
    fig.set_size_inches(6.6, 4.2)
    if suppress_title:
        ax.set_title("")
    else:
        ax.set_title(plot_title, fontsize=8)
    rfc_png_buffer = io.BytesIO()
    fig.savefig(rfc_png_buffer, format="png", dpi=220, bbox_inches="tight")
    rfc_png_buffer.seek(0)
    if save_pdf_path is not None:
        fig.savefig(save_pdf_path, format="pdf", bbox_inches="tight")
    png_bytes = rfc_png_buffer.getvalue()
    plt.close(fig)
    return {"rfc_png": png_bytes, "fit_stats": fit_stats}
