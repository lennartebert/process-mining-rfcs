"""Resolve optional / null experiment parameters from notebook dictionaries."""

from __future__ import annotations

from typing import TypeVar

from .types import (
    CompositionExperimentSettings,
    ControlFlowSettings,
    DriftExperimentSettings,
    ExperimentSettings,
    XorProbabilityDistribution,
)

T = TypeVar("T")

DEFAULT_ALPHA = 1.5
DEFAULT_SEED = 123


def resolve(settings: dict, key: str, default: T) -> T:
    """Return *default* when *key* is missing or explicitly ``None``."""
    value = settings.get(key, default)
    return default if value is None else value


def alpha_for_simulation(settings: dict) -> float:
    """Numeric ``alpha`` for model/BPMN code (ignored unless ``q`` is power-law)."""
    return float(resolve(settings, "alpha", DEFAULT_ALPHA))


def normalize_control_flow(cf: ControlFlowSettings) -> ControlFlowSettings:
    """Fill in defaults; ``alpha=None`` is allowed when ``q`` is not ``power_law``."""
    q: XorProbabilityDistribution = resolve(cf, "q", "equal")
    normalized: ControlFlowSettings = {
        "m": int(resolve(cf, "m", 1)),
        "n": int(resolve(cf, "n", 1)),
        "o": int(resolve(cf, "o", 3)),
        "p": float(resolve(cf, "p", 0.0)),
        "q": q,
    }
    if q == "power_law":
        normalized["alpha"] = float(resolve(cf, "alpha", DEFAULT_ALPHA))
    return normalized


def normalize_simple_settings(settings: ExperimentSettings) -> ExperimentSettings:
    """Normalize a simple-process experiment dict before simulation."""
    q: XorProbabilityDistribution = resolve(settings, "q", "equal")
    normalized: ExperimentSettings = {
        "N": int(resolve(settings, "N", 5000)),
        "m": int(resolve(settings, "m", 1)),
        "n": int(resolve(settings, "n", 1)),
        "o": int(resolve(settings, "o", 3)),
        "p": float(resolve(settings, "p", 0.0)),
        "q": q,
        "seed": int(resolve(settings, "seed", DEFAULT_SEED)),
    }
    if q == "power_law":
        normalized["alpha"] = float(resolve(settings, "alpha", DEFAULT_ALPHA))
    if "name" in settings and settings["name"] is not None:
        normalized["name"] = settings["name"]
    return normalized


def normalize_composition_settings(
    settings: CompositionExperimentSettings,
) -> CompositionExperimentSettings:
    """Normalize composition experiment dict."""
    normalized: CompositionExperimentSettings = {
        "N": int(resolve(settings, "N", 5000)),
        "a": normalize_control_flow(settings["a"]),
        "b": normalize_control_flow(settings["b"]),
        "seed": int(resolve(settings, "seed", DEFAULT_SEED)),
    }
    if "name" in settings and settings["name"] is not None:
        normalized["name"] = settings["name"]
    return normalized


def normalize_drift_settings(settings: DriftExperimentSettings) -> DriftExperimentSettings:
    """Normalize drift experiment dict."""
    normalized: DriftExperimentSettings = {
        "N": int(resolve(settings, "N", 5000)),
        "s": resolve(settings, "s", "overlapping"),
        "a": normalize_control_flow(settings["a"]),
        "b": normalize_control_flow(settings["b"]),
        "seed": int(resolve(settings, "seed", DEFAULT_SEED)),
    }
    if "name" in settings and settings["name"] is not None:
        normalized["name"] = settings["name"]
    return normalized
