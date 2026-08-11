"""Small parsing helpers shared by CLIs and analyses."""

from __future__ import annotations

import pandas as pd


def parse_count(value: object) -> float:
    """Parse a count stored as a number or a comma-formatted string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return float("nan")
    return float(text.replace(",", ""))
