"""Type definitions for RFC analysis outputs."""

from __future__ import annotations

from typing import TypedDict


class FitStatistics(TypedDict, total=False):
    """Fitted-curve parameters and scoring metrics for one RFC series."""

    linear_a: float
    linear_b: float
    linear_nll: float
    exponential_a: float
    exponential_b: float
    exponential_nll: float
    power_c: float
    power_alpha: float
    power_nll: float
    bounded_power_min_frequency: float
    bounded_power_c: float
    bounded_power_alpha: float
    bounded_power_nll: float
    best_fit: str
