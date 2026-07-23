"""Experiment orchestration for custom process structure simulations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bpmn_generation import build_bpmn_from_instruction, render_bpmn_bytes
from .event_log_simulation import simulate_event_log
from .log_combination import (
    combine_logs,
    extract_traces,
    pair_and_concat_traces,
    prepare_logs_for_drift,
    traces_to_event_log,
)
from .model_definition import create_parallel_xor_model
from .rfc_visualization import render_rfc_from_event_log
from .settings_resolution import (
    alpha_for_simulation,
    normalize_composition_settings,
    normalize_control_flow,
    normalize_drift_settings,
    normalize_simple_settings,
    resolve,
)
from .types import (
    CompositionExperimentSettings,
    ControlFlowSettings,
    DriftExperimentSettings,
    ExperimentSettings,
    SingleExperimentResult,
    TwoProcessExperimentResult,
)


def control_flow_to_settings(cf: ControlFlowSettings, N: int, seed: int) -> ExperimentSettings:
    """Merge control-flow dict with general parameters."""
    cf_norm = normalize_control_flow(cf)
    settings: ExperimentSettings = {
        "N": N,
        "m": cf_norm["m"],
        "n": cf_norm["n"],
        "o": cf_norm["o"],
        "p": cf_norm["p"],
        "q": cf_norm["q"],
        "seed": seed,
    }
    if "alpha" in cf_norm:
        settings["alpha"] = cf_norm["alpha"]
    return settings


def _artifact_path(save_prefix: Path, stem: str) -> Path:
    """Build ``{save_prefix.name}_{stem}.pdf`` under the same directory."""
    return save_prefix.with_name(f"{save_prefix.name}_{stem}").with_suffix(".pdf")


def _simulate_log(settings: ExperimentSettings) -> pd.DataFrame:
    model = create_parallel_xor_model(
        n=settings["n"],
        o=settings["o"],
        m=settings["m"],
        q=settings["q"],
        alpha=alpha_for_simulation(settings),
    )
    return simulate_event_log(
        model=model,
        N=settings["N"],
        p=settings["p"],
        seed=resolve(settings, "seed", 123),
    )


def _sample_traces(event_log, trace_count: int) -> list[tuple[str, list[str]]]:
    sample_traces = []
    for case_id in event_log["case"].drop_duplicates().head(trace_count):
        activities = event_log[event_log["case"] == case_id]["activity"].tolist()
        sample_traces.append((case_id, activities))
    return sample_traces


def _save_bpmn_pdf(
    settings: ExperimentSettings,
    save_prefix: Path | None,
    label: str | None = None,
) -> bytes | None:
    """Render BPMN PNG; optionally persist PDF as ``*_bpmn`` or ``*_bpmn_{label}``."""
    try:
        bpmn_model = build_bpmn_from_instruction(
            n=settings["n"],
            o=settings["o"],
            m=settings["m"],
            q=settings["q"],
            alpha=alpha_for_simulation(settings),
            p=settings["p"],
        )
        bpmn_png = render_bpmn_bytes(bpmn_model, image_format="png", dpi=300)
    except Exception as exc:
        if save_prefix is not None:
            raise
        print(f"Warning: BPMN rendering skipped ({exc})")
        return None
    if save_prefix is not None:
        stem = f"bpmn_{label}" if label else "bpmn"
        bpmn_pdf = render_bpmn_bytes(bpmn_model, image_format="pdf", dpi=300)
        _artifact_path(save_prefix, stem).write_bytes(bpmn_pdf)
    return bpmn_png


def _save_rfc_pdf(
    event_log: pd.DataFrame,
    settings: ExperimentSettings,
    save_prefix: Path | None,
    label: str,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> dict:
    """Render RFC PNG/PDF for one log segment."""
    pdf_path = _artifact_path(save_prefix, f"rfc_{label}") if save_prefix is not None else None
    return render_rfc_from_event_log(
        event_log=event_log,
        settings=settings,
        save_pdf_path=pdf_path,
        x_max=x_max_in_graphs,
        y_max=y_max_in_graphs,
    )


def run_simple_experiment(
    settings: ExperimentSettings,
    save_prefix: Path | None = None,
    trace_count: int = 15,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> SingleExperimentResult:
    """Execute one simple synthetic experiment."""
    settings = normalize_simple_settings(settings)
    event_log = _simulate_log(settings)
    rfc_data = render_rfc_from_event_log(
        event_log=event_log,
        settings=settings,
        save_pdf_path=save_prefix.with_suffix(".pdf") if save_prefix is not None else None,
        x_max=x_max_in_graphs,
        y_max=y_max_in_graphs,
    )
    bpmn_png = _save_bpmn_pdf(settings, save_prefix)
    return {
        "event_log": event_log,
        "fit_stats": rfc_data["fit_stats"],
        "rfc_png": rfc_data["rfc_png"],
        "bpmn_png": bpmn_png or b"",
        "sample_traces": _sample_traces(event_log, trace_count),
    }


def _run_two_process_experiment(
    settings_a: ExperimentSettings,
    settings_b: ExperimentSettings,
    log_a: pd.DataFrame,
    log_b: pd.DataFrame,
    combined_log: pd.DataFrame,
    save_prefix: Path | None,
    trace_count: int,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> TwoProcessExperimentResult:
    """Persist per-process BPMN/RFC plus combined RFC."""
    bpmn_a_png = _save_bpmn_pdf(settings_a, save_prefix, label="a")
    bpmn_b_png = _save_bpmn_pdf(settings_b, save_prefix, label="b")
    rfc_a_data = _save_rfc_pdf(
        log_a, settings_a, save_prefix, label="a", x_max_in_graphs=x_max_in_graphs, y_max_in_graphs=y_max_in_graphs
    )
    rfc_b_data = _save_rfc_pdf(
        log_b, settings_b, save_prefix, label="b", x_max_in_graphs=x_max_in_graphs, y_max_in_graphs=y_max_in_graphs
    )

    combined_settings: ExperimentSettings = {
        "N": settings_a["N"] + settings_b["N"],
        "m": settings_a["m"],
        "n": settings_a["n"],
        "o": settings_a["o"],
        "p": settings_a["p"],
        "q": settings_a["q"],
    }
    if "alpha" in settings_a:
        combined_settings["alpha"] = settings_a["alpha"]
    rfc_combined_data = _save_rfc_pdf(
        combined_log,
        combined_settings,
        save_prefix,
        label="combined",
        x_max_in_graphs=x_max_in_graphs,
        y_max_in_graphs=y_max_in_graphs,
    )

    return {
        "event_log": combined_log,
        "fit_stats": rfc_combined_data["fit_stats"],
        "rfc_png": rfc_combined_data["rfc_png"],
        "bpmn_png": bpmn_a_png or b"",
        "bpmn_a_png": bpmn_a_png or b"",
        "bpmn_b_png": bpmn_b_png or b"",
        "rfc_a_png": rfc_a_data["rfc_png"],
        "rfc_b_png": rfc_b_data["rfc_png"],
        "sample_traces": _sample_traces(combined_log, trace_count),
    }


def run_composition_experiment(
    settings: CompositionExperimentSettings,
    save_prefix: Path | None = None,
    trace_count: int = 15,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> TwoProcessExperimentResult:
    """Simulate process a and b, pair traces, and render per-process + combined RFCs."""
    settings = normalize_composition_settings(settings)
    seed = settings["seed"]
    N = settings["N"]
    settings_a = control_flow_to_settings(settings["a"], N=N, seed=seed)
    settings_b = control_flow_to_settings(settings["b"], N=N, seed=seed + 1)

    log_a = _simulate_log(settings_a)
    log_b = _simulate_log(settings_b)
    combined_traces = pair_and_concat_traces(
        extract_traces(log_a),
        extract_traces(log_b),
        seed=seed,
        prefix_a="a",
        prefix_b="b",
    )
    combined_log = traces_to_event_log(combined_traces)

    return _run_two_process_experiment(
        settings_a,
        settings_b,
        log_a,
        log_b,
        combined_log,
        save_prefix,
        trace_count,
        x_max_in_graphs=x_max_in_graphs,
        y_max_in_graphs=y_max_in_graphs,
    )


def run_drift_experiment(
    settings: DriftExperimentSettings,
    save_prefix: Path | None = None,
    trace_count: int = 15,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> TwoProcessExperimentResult:
    """Simulate sudden 50/50 drift between process a and b."""
    settings = normalize_drift_settings(settings)
    seed = settings["seed"]
    N = settings["N"]
    N_a = N // 2
    N_b = N - N_a

    settings_a = control_flow_to_settings(settings["a"], N=N_a, seed=seed)
    settings_b = control_flow_to_settings(settings["b"], N=N_b, seed=seed + 1)

    log_a = _simulate_log(settings_a)
    log_b = _simulate_log(settings_b)
    log_a_for_combine, log_b_for_combine = prepare_logs_for_drift(log_a, log_b, settings["s"])
    combined_log = combine_logs([log_a_for_combine, log_b_for_combine])

    return _run_two_process_experiment(
        settings_a,
        settings_b,
        log_a,
        log_b,
        combined_log,
        save_prefix,
        trace_count,
        x_max_in_graphs=x_max_in_graphs,
        y_max_in_graphs=y_max_in_graphs,
    )


def _run_batch(
    runner,
    experiments: list,
    results_dir: Path,
    trace_count: int,
    two_process: bool = False,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> list[dict]:
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict] = []
    for experiment in experiments:
        save_prefix = results_dir / experiment["name"]
        result = runner(
            experiment,
            save_prefix=save_prefix,
            trace_count=trace_count,
            x_max_in_graphs=x_max_in_graphs,
            y_max_in_graphs=y_max_in_graphs,
        )
        if two_process:
            outputs.append(
                {
                    "name": experiment["name"],
                    "result": result,
                    "rfc_pdf_a": _artifact_path(save_prefix, "rfc_a"),
                    "rfc_pdf_b": _artifact_path(save_prefix, "rfc_b"),
                    "rfc_pdf_combined": _artifact_path(save_prefix, "rfc_combined"),
                    "bpmn_pdf_a": _artifact_path(save_prefix, "bpmn_a"),
                    "bpmn_pdf_b": _artifact_path(save_prefix, "bpmn_b"),
                }
            )
        else:
            outputs.append(
                {
                    "name": experiment["name"],
                    "result": result,
                    "rfc_pdf": save_prefix.with_suffix(".pdf"),
                    "bpmn_pdf": _artifact_path(save_prefix, "bpmn"),
                }
            )
    return outputs


def run_simple_batch(
    experiments: list[ExperimentSettings],
    results_dir: Path,
    trace_count: int = 15,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> list[dict]:
    """Run simple experiments and persist PDFs."""
    return _run_batch(
        run_simple_experiment,
        experiments,
        results_dir,
        trace_count,
        x_max_in_graphs=x_max_in_graphs,
        y_max_in_graphs=y_max_in_graphs,
    )


def run_composition_batch(
    experiments: list[CompositionExperimentSettings],
    results_dir: Path,
    trace_count: int = 15,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> list[dict]:
    """Run composition experiments and persist PDFs."""
    return _run_batch(
        run_composition_experiment,
        experiments,
        results_dir,
        trace_count,
        two_process=True,
        x_max_in_graphs=x_max_in_graphs,
        y_max_in_graphs=y_max_in_graphs,
    )


def run_drift_batch(
    experiments: list[DriftExperimentSettings],
    results_dir: Path,
    trace_count: int = 15,
    x_max_in_graphs: float | None = None,
    y_max_in_graphs: float | None = None,
) -> list[dict]:
    """Run drift experiments and persist PDFs."""
    return _run_batch(
        run_drift_experiment,
        experiments,
        results_dir,
        trace_count,
        two_process=True,
        x_max_in_graphs=x_max_in_graphs,
        y_max_in_graphs=y_max_in_graphs,
    )
