"""Paths for power-law statistics results under ``results/pl_stats``.

An experiment directory (``real``, ``real43``, ``synth``, ``synth_dfg``) holds
the bootstrap seed and a ``log_info`` directory. Log description and saturation
counts both live under ``log_info`` (saturation in ``log_info/saturation``).
Each xmin policy is a subdirectory of the experiment. Raw Clauset CSVs live
under that subdirectory. Classification thresholds live one level further
down, in ``clauset_results``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

XMIN_ADAPTIVE = "xmin-adaptive"
RAW_RESULTS_DIRNAME = "raw_results"
CLAUSE_RESULTS_DIRNAME = "clauset_results"
LOG_INFO_DIRNAME = "log_info"
SATURATION_DIRNAME = "saturation"
EXPLORE_DIRNAME = "explore_gof_changes"
SEED_FILENAME = "seed.json"
PLOTS_DIRNAME = "plots"


def xmin_dirname(xmin: int | None) -> str:
    """Folder for an xmin policy. ``None`` is the adaptive fit."""
    if xmin is None:
        return XMIN_ADAPTIVE
    value = int(xmin)
    if value < 0:
        raise ValueError(f"xmin must be non-negative, got {xmin}")
    return f"xmin{value:02d}"


def xmin_setting_from_dirname(name: str) -> int | None:
    """Inverse of :func:`xmin_dirname`."""
    if name == XMIN_ADAPTIVE:
        return None
    if name.startswith("xmin") and len(name) == 6 and name[4:].isdigit():
        return int(name[4:])
    raise ValueError(f"not an xmin directory name: {name}")


def concept_dirname(concept: str) -> str:
    """Zero-pad n-gram concept ids for result folders (``n1`` -> ``n001``)."""
    text = str(concept)
    if text.startswith("n") and text[1:].isdigit():
        return f"n{int(text[1:]):03d}"
    return text


def concept_id(dirname: str) -> str:
    """Inverse of :func:`concept_dirname` (``n001`` -> ``n1``)."""
    text = str(dirname)
    if text.startswith("n") and text[1:].isdigit():
        return f"n{int(text[1:])}"
    return text


def config_dirname(*, min_types: int, gof_p: float, comparison_p: float) -> str:
    """Clauset configuration folder, e.g. ``min[30]-gofp[0.10]-pcomp[0.10]``."""
    return (
        f"min[{int(min_types):02d}]-"
        f"gofp[{float(gof_p):.2f}]-"
        f"pcomp[{float(comparison_p):.2f}]"
    )


def log_info_dir(experiment: Path) -> Path:
    return Path(experiment) / LOG_INFO_DIRNAME


def saturation_dir(experiment: Path) -> Path:
    """Cross-log saturation tables for one experiment (``log_info/saturation``)."""
    return log_info_dir(experiment) / SATURATION_DIRNAME


def xmin_dir(experiment: Path, xmin: int | None) -> Path:
    return Path(experiment) / xmin_dirname(xmin)


def raw_results_dir(experiment: Path, xmin: int | None) -> Path:
    return xmin_dir(experiment, xmin) / RAW_RESULTS_DIRNAME


def concept_results_dir(raw_results: Path, concept: str) -> Path:
    return Path(raw_results) / concept_dirname(concept)


def clauset_config_dir(
    experiment: Path,
    xmin: int | None,
    *,
    min_types: int,
    gof_p: float,
    comparison_p: float,
) -> Path:
    return (
        xmin_dir(experiment, xmin)
        / CLAUSE_RESULTS_DIRNAME
        / config_dirname(
            min_types=min_types, gof_p=gof_p, comparison_p=comparison_p
        )
    )


def plots_dir(experiment: Path, xmin: int | None) -> Path:
    return raw_results_dir(experiment, xmin) / PLOTS_DIRNAME


def explore_dir(experiment: Path, xmin: int | None) -> Path:
    return xmin_dir(experiment, xmin) / EXPLORE_DIRNAME


def write_seed(experiment: Path, seed: int) -> Path:
    """Record the bootstrap seed shared by every xmin folder in this experiment."""
    path = Path(experiment) / SEED_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seed": int(seed)}, indent=2) + "\n", encoding="utf-8"
    )
    return path


def stack_raw_summaries(raw_results: Path) -> pd.DataFrame:
    """Stack per-concept ``summary.csv`` files, adding ``concept`` and ``model``.

    Concept ids are the unpadded names (``n1``, ``variants``). The prose
    classification column is not created here.
    """
    root = Path(raw_results)
    frames: list[pd.DataFrame] = []
    if not root.is_dir():
        return pd.DataFrame()
    for concept_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if concept_dir.name == PLOTS_DIRNAME:
            continue
        concept = concept_id(concept_dir.name)
        for model_dir in sorted(path for path in concept_dir.iterdir() if path.is_dir()):
            summary_path = model_dir / "summary.csv"
            if not summary_path.is_file():
                continue
            try:
                frame = pd.read_csv(summary_path)
            except pd.errors.EmptyDataError:
                continue
            if frame.empty:
                continue
            frame.insert(0, "concept", concept)
            frame.insert(1, "model", model_dir.name)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
