"""Type definitions for custom process structure experiments."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

import pandas as pd

XorProbabilityDistribution = Literal["equal", "majority", "power_law"]
ActivitySetOverlap = Literal["overlapping", "non_overlapping"]


class ControlFlowSettings(TypedDict, total=False):
    """Control-flow parameters for one synthetic process."""

    m: int
    n: int
    o: int
    p: float
    q: XorProbabilityDistribution
    alpha: Optional[float]  # omit or None when q is not power_law


class ExperimentSettings(TypedDict, total=False):
    """Input parameters for one simple synthetic process experiment."""

    name: str
    N: int
    m: int
    n: int
    o: int
    p: float
    q: XorProbabilityDistribution
    alpha: Optional[float]  # omit or None when q is not power_law
    seed: int


class CompositionExperimentSettings(TypedDict, total=False):
    """Settings for process-composition experiments."""

    name: str
    N: int
    a: ControlFlowSettings
    b: ControlFlowSettings
    seed: int


class DriftExperimentSettings(TypedDict, total=False):
    """Settings for sudden and gradual drift experiments."""

    name: str
    N: int
    s: ActivitySetOverlap
    a: ControlFlowSettings
    b: ControlFlowSettings
    seed: int


class SingleExperimentResult(TypedDict):
    """Output payload of one experiment execution."""

    event_log: pd.DataFrame
    fit_stats: dict[str, Any]
    rfc_png: bytes
    bpmn_png: bytes
    sample_traces: list[tuple[str, list[str]]]


class TwoProcessExperimentResult(SingleExperimentResult):
    """Composition/drift result with separate process a and b artifacts."""

    bpmn_a_png: bytes
    bpmn_b_png: bytes
    rfc_a_png: bytes
    rfc_b_png: bytes
