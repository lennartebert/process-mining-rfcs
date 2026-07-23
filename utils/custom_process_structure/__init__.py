"""Synthetic custom-process-structure modeling and experiment utilities."""

from .bpmn_generation import build_bpmn_from_instruction, render_bpmn_bytes
from .event_log_simulation import sample_parallel_iteration_trace, simulate_event_log
from .experiment_execution import (
    control_flow_to_settings,
    run_composition_batch,
    run_composition_experiment,
    run_drift_batch,
    run_drift_experiment,
    run_simple_batch,
    run_simple_experiment,
)
from .log_combination import (
    combine_logs,
    extract_traces,
    pair_and_concat_traces,
    prefix_activities,
    prepare_logs_for_drift,
    relabel_cases,
    traces_to_event_log,
)
from .model_definition import create_parallel_xor_model, make_xor_probabilities
from .rfc_visualization import build_plot_title, render_rfc_from_event_log
from .types import (
    ActivitySetOverlap,
    CompositionExperimentSettings,
    ControlFlowSettings,
    DriftExperimentSettings,
    ExperimentSettings,
    SingleExperimentResult,
    TwoProcessExperimentResult,
    XorProbabilityDistribution,
)

__all__ = [
    "ActivitySetOverlap",
    "CompositionExperimentSettings",
    "ControlFlowSettings",
    "DriftExperimentSettings",
    "ExperimentSettings",
    "SingleExperimentResult",
    "TwoProcessExperimentResult",
    "XorProbabilityDistribution",
    "build_bpmn_from_instruction",
    "build_plot_title",
    "combine_logs",
    "control_flow_to_settings",
    "create_parallel_xor_model",
    "extract_traces",
    "make_xor_probabilities",
    "pair_and_concat_traces",
    "prefix_activities",
    "prepare_logs_for_drift",
    "relabel_cases",
    "render_bpmn_bytes",
    "render_rfc_from_event_log",
    "run_composition_batch",
    "run_composition_experiment",
    "run_drift_batch",
    "run_drift_experiment",
    "run_simple_batch",
    "run_simple_experiment",
    "sample_parallel_iteration_trace",
    "simulate_event_log",
    "traces_to_event_log",
]
