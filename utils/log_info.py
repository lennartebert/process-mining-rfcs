"""Shared log-info table columns and LaTeX export.

Used by ``cli/log_info.py`` and by the parallel-shard combine step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from utils.io import parse_count

IDENTITY_COLUMNS: list[str] = ["Log", "Description"]

INTEGER_STATS = frozenset(
    {
        "# Cases",
        "# Events",
        "# Variants",
        "# Activities",
        "Median Trace Length",
        "Max Trace Length",
    }
)

_LATEX_COLUMN_RENAMES = {
    "# Cases": r"\# Cases",
    "# Events": r"\# Events",
    "# Variants": r"\# Variants",
    "# Activities": r"\# Activities",
    "Median Trace Length": r"Median Trace Length",
    "Max Trace Length": r"Max Trace Length",
}


def _format_stat_for_latex(value: object, *, integer_like: bool) -> str:
    """Format a numeric stat for LaTeX tables."""
    parsed = parse_count(value)
    if pd.isna(parsed):
        return ""
    if integer_like:
        return format(int(round(parsed)), ",")
    text = f"{parsed:.1f}"
    if text.endswith(".0"):
        return format(int(parsed), ",")
    return text


def write_log_info_latex(
    details_df: pd.DataFrame, tex_path: Path, stats: Sequence[str]
) -> None:
    """Write a log-info LaTeX table from an existing details frame."""
    latex_df = details_df.copy()
    for col in stats:
        latex_df[col] = latex_df[col].map(
            lambda value, c=col: _format_stat_for_latex(
                value, integer_like=c in INTEGER_STATS
            )
        )
    rename = {
        col: _LATEX_COLUMN_RENAMES[col]
        for col in stats
        if col in _LATEX_COLUMN_RENAMES
    }
    latex_df = latex_df.rename(columns=rename)
    tex_path.write_text(latex_df.to_latex(index=False, escape=False), encoding="utf-8")
    print(f"Saved: {tex_path}")
