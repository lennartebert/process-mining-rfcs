"""Lightweight wall-clock timing helpers for CLI / analysis printouts."""

from __future__ import annotations

from datetime import datetime


def timestamp() -> str:
    """Local wall-clock stamp for timing lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    """Human-readable duration (e.g. ``1h02m03.4s``, ``12m03.4s``, ``3.4s``)."""
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    frac = seconds - total
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    sec_str = f"{secs + frac:.1f}s"
    if hours:
        return f"{hours}h{minutes:02d}m{sec_str}"
    if minutes:
        return f"{minutes}m{sec_str}"
    return sec_str


def timing_print(message: str) -> None:
    """Print a ``[ts] TIMING: ...`` line."""
    print(f"[{timestamp()}] TIMING: {message}", flush=True)
