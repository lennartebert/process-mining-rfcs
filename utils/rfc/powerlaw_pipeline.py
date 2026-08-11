"""Shared Clauset-style power-law analysis for discrete or continuous data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .statistical_tests import (
    DISTRIBUTION_NAMES,
    DISTRIBUTION_SPECS,
    DOUBLY_BOUNDED_POWER_LAW,
    FULL_RANGE_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    best_other_distribution,
    build_summary_row,
    classify_power_law_result,
    compare_distribution,
    descriptive_observation_stats,
    empty_comparison_frame,
    empty_summary_row,
    evaluate_doubly_bounded_power_law,
    evaluate_power_law_fit,
    gof_row_from_evaluation,
    human_model_label,
    to_observation_array,
)


def analyze_powerlaw_data(
    data: np.ndarray,
    *,
    log_name: str,
    input_path: str,
    discrete: bool = True,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    minimum_fitted_variants: int = 50,
    significance_level: float = 0.10,
) -> dict[str, dict[str, Any]]:
    """Run full-range, lower-bounded, and doubly-bounded tests on observations.

    Parameters
    ----------
    data :
        Discrete: one positive integer per distinct item (variant / category count).
        Continuous: one positive finite magnitude per observation.
    discrete :
        Pass through to ``powerlaw.Fit`` and semiparametric bootstrap.
    log_name / input_path :
        Labels copied into summary / GOF / comparison rows.
    """
    data = to_observation_array(data, discrete=discrete)
    if data.size < 2:
        raise ValueError("not enough observations to fit a power law")

    descriptive_stats = descriptive_observation_stats(data, discrete=discrete)

    evaluations: dict[str, dict[str, Any] | None] = {}
    gof_rows_by_dist: dict[str, list[dict[str, Any]]] = {}

    for name in (FULL_RANGE_POWER_LAW, LOWER_BOUNDED_POWER_LAW):
        xmin = DISTRIBUTION_SPECS[name]["xmin"]
        evaluation = evaluate_power_law_fit(
            data,
            xmin=xmin,
            xmax=None,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
            discrete=discrete,
        )
        evaluations[name] = evaluation
        gof_row = gof_row_from_evaluation(log_name=log_name, evaluation=evaluation)
        gof_rows_by_dist[name] = [] if gof_row is None else [gof_row]

    db_selected, candidate_fits = evaluate_doubly_bounded_power_law(
        data,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        discrete=discrete,
    )
    evaluations[DOUBLY_BOUNDED_POWER_LAW] = db_selected
    if db_selected is None:
        gof_rows_by_dist[DOUBLY_BOUNDED_POWER_LAW] = []
    else:
        gof_row = gof_row_from_evaluation(
            log_name=log_name,
            evaluation=db_selected,
            selected=True,
        )
        gof_rows_by_dist[DOUBLY_BOUNDED_POWER_LAW] = (
            [] if gof_row is None else [gof_row]
        )

    results: dict[str, dict[str, Any]] = {}
    for name in DISTRIBUTION_NAMES:
        evaluation = evaluations[name]
        if evaluation is None:
            results[name] = {
                "gof_rows": gof_rows_by_dist[name],
                "comparison_df": empty_comparison_frame(log_name, model_1=name),
                "summary_row": empty_summary_row(
                    log_name=log_name,
                    input_path=input_path,
                    classification="no eligible doubly bounded interval",
                    include_exclusion_fields=(name == DOUBLY_BOUNDED_POWER_LAW),
                ),
                "candidate_fits": candidate_fits if name == DOUBLY_BOUNDED_POWER_LAW else [],
                "descriptive_stats": descriptive_stats,
            }
            continue

        fit_valid = bool(evaluation.get("fit_valid", True))
        if fit_valid:
            comparison_df = compare_distribution(
                log_name=log_name,
                model_1=name,
                fit_1=evaluation["fit"],
                significance_level=significance_level,
            )
            best_other = best_other_distribution(comparison_df)
        else:
            comparison_df = empty_comparison_frame(log_name, model_1=name)
            best_other = None

        n_fitted = evaluation.get("n_fitted_variants")
        n_fitted_variants = 0 if n_fitted is None or pd.isna(n_fitted) else int(n_fitted)
        gof_p_raw = evaluation.get("gof_p")
        try:
            gof_p = float(gof_p_raw)
        except (TypeError, ValueError):
            gof_p = float("nan")
        classification = classify_power_law_result(
            n_fitted_variants=n_fitted_variants,
            gof_p=gof_p,
            comparison_df=comparison_df,
            model_label=human_model_label(name),
            minimum_fitted_variants=minimum_fitted_variants,
            significance_level=significance_level,
            fit_valid=fit_valid,
            pathology_flags=evaluation.get("pathology_flags") or [],
        )
        summary_row = build_summary_row(
            log_name=log_name,
            input_path=input_path,
            descriptive_stats=descriptive_stats,
            evaluation=evaluation,
            classification=classification,
            best_other_distribution=best_other,
        )
        results[name] = {
            "gof_rows": gof_rows_by_dist[name],
            "comparison_df": comparison_df,
            "summary_row": summary_row,
            "candidate_fits": candidate_fits if name == DOUBLY_BOUNDED_POWER_LAW else [],
            "descriptive_stats": descriptive_stats,
            "evaluation": evaluation,
        }
    return results


def empty_powerlaw_results(
    *,
    log_name: str,
    input_path: str,
    classification: str,
) -> dict[str, dict[str, Any]]:
    """Build empty per-distribution outputs for a failed analysis unit."""
    results: dict[str, dict[str, Any]] = {}
    for name in DISTRIBUTION_NAMES:
        results[name] = {
            "gof_rows": [],
            "comparison_df": empty_comparison_frame(log_name, model_1=name),
            "summary_row": empty_summary_row(
                log_name=log_name,
                input_path=input_path,
                classification=classification,
                include_exclusion_fields=(name == DOUBLY_BOUNDED_POWER_LAW),
            ),
            "candidate_fits": [],
            "descriptive_stats": pd.DataFrame(),
        }
    return results


def append_and_checkpoint(
    *,
    analysis_dir: Path,
    accumulated: dict[str, dict[str, list]],
    dataset_results: dict[str, dict[str, Any]],
) -> None:
    """Append one unit's results and checkpoint CSVs per distribution."""
    for name in DISTRIBUTION_NAMES:
        dist_dir = analysis_dir / name
        dist_dir.mkdir(parents=True, exist_ok=True)

        accumulated[name]["gof_rows"].extend(dataset_results[name]["gof_rows"])
        accumulated[name]["comparison_frames"].append(
            dataset_results[name]["comparison_df"]
        )
        accumulated[name]["summary_rows"].append(dataset_results[name]["summary_row"])

        gof_df = pd.DataFrame(accumulated[name]["gof_rows"])
        if not gof_df.empty:
            sort_cols = ["log_name"]
            if "excluded_head_variants" in gof_df.columns:
                sort_cols.append("excluded_head_variants")
            gof_df = gof_df.sort_values(sort_cols).reset_index(drop=True)

        comparison_df = (
            pd.concat(accumulated[name]["comparison_frames"], ignore_index=True)
            .sort_values(["log_name", "model_2"])
            .reset_index(drop=True)
        )
        summary_df = (
            pd.DataFrame(accumulated[name]["summary_rows"])
            .sort_values("log_name")
            .reset_index(drop=True)
        )

        gof_df.to_csv(dist_dir / "gof.csv", index=False)
        comparison_df.to_csv(dist_dir / "comparison.csv", index=False)
        summary_df.to_csv(dist_dir / "summary.csv", index=False)
