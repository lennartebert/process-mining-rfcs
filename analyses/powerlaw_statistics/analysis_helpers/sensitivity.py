"""One-at-a-time sensitivity reporting from saved power-law results.

Reads the adaptive real-log Clauset summaries and likelihood-ratio tables. It
does not refit, rerun likelihood-ratio comparisons, or rerun goodness-of-fit
bootstraps. A configuration's ``xmin`` selects the saved folder, not the fitted
xmin stored in a row. A missing result stays missing.

Eligibility and paired GOF reuse ``paired_gof.assess_endpoint`` and
``paired_gof.build_paired_gof_tables``. Trace-variant categories A–E reuse
``compact_classification``. Outputs are ``configurations.csv``,
``sensitivity_summary.csv``, and matching ``.tex`` tables.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.powerlaw_statistics.analysis_helpers.paired_gof import (
    MODEL,
    PAPER_KS,
    REFERENCE_CONCEPT,
    EndpointAssessment,
    assess_endpoint,
    build_paired_gof_tables,
    concept_for_k,
    load_concept_results,
)
from analyses.powerlaw_statistics.analysis_helpers.paths import (
    concept_dirname,
    raw_results_dir,
    xmin_dir,
)
from utils.constants import ALL_REAL_LOG_DATASETS, PL_STATS_REAL_DIR
from utils.csv_shards import concat_sorted_csvs, filter_shards_by_log
from utils.io.attachments import NGRAM_CONCEPTS
from utils.powerlaw import compact_classification
from utils.powerlaw.analyze import _finite_preference_rows

OUTPUT_DIR = PL_STATS_REAL_DIR / "sensitivity"
CONCEPTS: tuple[str, ...] = (*NGRAM_CONCEPTS, REFERENCE_CONCEPT)
SIGNIFICANCE = 0.05
ZERO_TOL_PP = 1e-9
SUMMARY_COLUMNS: tuple[str, ...] = (
    "setting",
    "input_status",
    "tv_category_transitions",
    "tv_A_count",
    "tv_B_count",
    "tv_C_count",
    "tv_D_count",
    "tv_E_count",
    "tv_eligible_n",
    "tv_gof_nonreject_n",
    "tv_gof_nonreject_pct",
    "paired_n_by_k",
    "paired_n_range",
    "paired_eligibility_changes",
    "paired_difference_range_pp",
    "paired_sign_reversal_orders",
    "paired_zero_difference_orders",
    "paired_unavailable_orders",
    "paired_availability_changes",
    "paired_holm_significant_orders",
    "paired_holm_significance_changes",
)
OBSOLETE_OUTPUT_NAMES: tuple[str, ...] = (
    "paired_gof_sensitivity.csv",
    "changes_vs_baseline.csv",
    "category_sensitivity.csv",
    "README.md",
)


@dataclass(frozen=True)
class Configuration:
    setting: str
    xmin: int | None
    min_types: int
    p_gof: float
    p_comp: float

    @property
    def xmin_label(self) -> str:
        if self.xmin is None:
            return "adaptive"
        return str(int(self.xmin))

    @property
    def appendix_heading(self) -> str:
        """Human-readable appendix heading for the varied threshold."""
        if self.setting == "baseline":
            return "Baseline"
        if self.setting.startswith("min_types_"):
            return f"Minimum fitted types = {int(self.min_types)}"
        if self.setting.startswith("gof_"):
            return f"GOF threshold = {float(self.p_gof):.2f}"
        if self.setting.startswith("comp_"):
            return f"Comparison threshold = {float(self.p_comp):.2f}"
        return self.setting


# Authoritative one-at-a-time grid (order preserved).
CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration("baseline", None, 50, 0.10, 0.10),
    Configuration("min_types_0", None, 0, 0.10, 0.10),
    Configuration("min_types_15", None, 15, 0.10, 0.10),
    Configuration("min_types_30", None, 30, 0.10, 0.10),
    Configuration("gof_005", None, 50, 0.05, 0.10),
    Configuration("gof_020", None, 50, 0.20, 0.10),
    Configuration("comp_005", None, 50, 0.10, 0.05),
    Configuration("comp_020", None, 50, 0.10, 0.20),
)

# Kept for tests / callers that still distinguish paired vs category OAT rows.
PAIRED_SETTING_NAMES: tuple[str, ...] = tuple(
    config.setting
    for config in CONFIGURATIONS
    if config.setting not in {"comp_005", "comp_020"}
)
CATEGORY_SETTING_NAMES: tuple[str, ...] = ("baseline", "comp_005", "comp_020")


@dataclass
class SensitivityResult:
    configurations: pd.DataFrame
    summary: pd.DataFrame
    warnings: list[str]


def configurations_frame() -> pd.DataFrame:
    rows = [
        {
            "setting": config.setting,
            "xmin": config.xmin_label,
            "min_types": int(config.min_types),
            "p_gof": f"{config.p_gof:.2f}",
            "p_comp": f"{config.p_comp:.2f}",
        }
        for config in CONFIGURATIONS
    ]
    return pd.DataFrame(rows, columns=["setting", "xmin", "min_types", "p_gof", "p_comp"])


def _config(setting: str) -> Configuration:
    for config in CONFIGURATIONS:
        if config.setting == setting:
            return config
    raise KeyError(setting)


def _ks_from_concepts(concepts: Sequence[str]) -> tuple[int, ...]:
    ks: list[int] = []
    for concept in concepts:
        text = str(concept)
        if text.startswith("n") and text[1:].isdigit():
            ks.append(int(text[1:]))
    if ks != sorted(ks) or len(ks) != len(set(ks)):
        raise ValueError(f"n-gram concepts must be unique and increasing, got {concepts}")
    return tuple(ks)


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: object) -> float | None:
    if _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _text(value: object) -> str | None:
    if _missing(value):
        return None
    text = str(value).strip()
    return text or None


def _close(left: float | None, right: float | None, tol: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= tol


def _comparison_signature(rows: Sequence[Mapping[str, object]]) -> tuple | None:
    seen: dict[str, tuple] = {}
    for row in rows:
        model = _text(row.get("model_2"))
        if model is None:
            return None
        if model in seen:
            return None
        seen[model] = (
            model,
            _finite_float(row.get("R")),
            _finite_float(row.get("p")),
            _text(row.get("preferred_model")),
        )
    return tuple(sorted(seen.values()))


def _signatures_agree(left: tuple | None, right: tuple | None) -> bool:
    if left is None or right is None or len(left) != len(right):
        return False
    for one, other in zip(left, right):
        if one[0] != other[0] or one[3] != other[3]:
            return False
        if not _close(one[1], other[1]) or not _close(one[2], other[2]):
            return False
    return True


def _frame_from_rows(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame | None:
    if not rows or _comparison_signature(rows) is None:
        return None
    return pd.DataFrame(list(rows))


def load_concept_comparisons(
    xmin_directory: Path,
    concept: str,
    study_logs: Sequence[str],
) -> dict[str, pd.DataFrame | None]:
    """Load saved LLR rows for one concept.

    Shard and ``comparison.csv`` rows must agree when both contain a log.
    A log with no finite table, a repeated alternative, or disagreeing sources
    is ``None``.
    """
    model_dir = Path(xmin_directory) / "raw_results" / concept_dirname(concept) / MODEL
    empty = {log: None for log in study_logs}
    if not model_dir.is_dir():
        return empty
    allowed = set(study_logs)
    shard_paths = filter_shards_by_log(
        list(model_dir.glob("comparison_*.csv")),
        prefix="comparison_",
        datasets=study_logs,
    )
    shard_frame = concat_sorted_csvs(shard_paths, sort_cols=["log_name"]) if shard_paths else None
    summary_path = model_dir / "comparison.csv"
    summary_frame = pd.read_csv(summary_path) if summary_path.is_file() else None

    def _grouped(frame: pd.DataFrame | None) -> dict[str, list[dict]]:
        grouped = {log: [] for log in study_logs}
        if frame is None or frame.empty or "log_name" not in frame.columns:
            return grouped
        for record in frame.to_dict(orient="records"):
            name = _text(record.get("log_name"))
            if name in allowed:
                grouped[name].append(record)
        return grouped

    from_shards = _grouped(shard_frame)
    from_summary = _grouped(summary_frame)
    loaded: dict[str, pd.DataFrame | None] = {}
    for log in study_logs:
        shards = from_shards[log]
        combined = from_summary[log]
        if shards and combined:
            if _signatures_agree(_comparison_signature(shards), _comparison_signature(combined)):
                loaded[log] = _frame_from_rows(shards)
            else:
                loaded[log] = None
        elif shards:
            loaded[log] = _frame_from_rows(shards)
        else:
            loaded[log] = _frame_from_rows(combined)
    return loaded


def classify_trace_variant(
    endpoint: EndpointAssessment,
    comparison: pd.DataFrame | None,
    *,
    min_types: int,
    p_gof: float,
    p_comp: float,
) -> str:
    """Paper A–E code for one trace-variant endpoint (no refit)."""
    if not endpoint.eligible:
        return "A"
    if endpoint.outcome != 1:
        return "B"
    comparison_df = comparison if comparison is not None else pd.DataFrame()
    code = compact_classification(
        n_fitted_types=int(endpoint.n_fitted_types or 0),
        gof_p=float(endpoint.gof_p),
        comparison_df=comparison_df,
        minimum_fitted_types=int(min_types),
        gof_p_threshold=float(p_gof),
        comparison_p_threshold=float(p_comp),
        fit_valid=True,
    )
    if code not in {"C", "D", "E"}:
        raise AssertionError(
            "eligible non-rejected endpoint produced category "
            f"{code} at min_types={min_types}, p_gof={p_gof}, p_comp={p_comp}"
        )
    return code


def _assess_all(
    experiment_dir: Path,
    configs: Sequence[Configuration],
    concepts: Sequence[str],
    study_logs: Sequence[str],
) -> tuple[dict[str, dict[str, dict[str, EndpointAssessment]]], dict]:
    loaded_by_xmin: dict[int | None, dict] = {}
    for xmin in {config.xmin for config in configs}:
        directory = xmin_dir(experiment_dir, xmin)
        loaded_by_xmin[xmin] = {
            concept: load_concept_results(directory, concept, study_logs)
            for concept in concepts
        }
    assessed: dict[str, dict[str, dict[str, EndpointAssessment]]] = {}
    for config in configs:
        per_concept: dict[str, dict[str, EndpointAssessment]] = {}
        for concept in concepts:
            loaded = loaded_by_xmin[config.xmin][concept]
            per_concept[concept] = {
                log: assess_endpoint(
                    loaded.by_log.get(log, []),
                    sources_disagree=log in loaded.sources_disagree,
                    min_fitted_types=config.min_types,
                    gof_threshold=config.p_gof,
                    xmin_setting=config.xmin,
                )
                for log in study_logs
            }
        assessed[config.setting] = per_concept
    return assessed, loaded_by_xmin


def _input_status(experiment_dir: Path, config: Configuration) -> str:
    directory = xmin_dir(experiment_dir, config.xmin)
    if not directory.is_dir():
        return f"incomplete: missing {directory.relative_to(experiment_dir)}"
    raw = raw_results_dir(experiment_dir, config.xmin)
    if not raw.is_dir():
        return f"incomplete: missing {raw.relative_to(experiment_dir)}"
    variants = raw / concept_dirname(REFERENCE_CONCEPT) / MODEL
    if not variants.is_dir():
        return (
            "incomplete: missing "
            f"{variants.relative_to(experiment_dir)}"
        )
    summary = variants / "summary.csv"
    shards = list(variants.glob("summary_*.csv"))
    if not summary.is_file() and not shards:
        return (
            "incomplete: missing summary tables under "
            f"{variants.relative_to(experiment_dir)}"
        )
    return "ok"


def _format_orders(values: Sequence[int]) -> str:
    if not values:
        return "None"
    return ", ".join(str(int(value)) for value in values)


def _format_log_list(values: Sequence[str]) -> str:
    if not values:
        return "None"
    return ", ".join(values)


def _difference_sign_pp(value: float | None) -> int | None:
    if value is None:
        return None
    if abs(value) <= ZERO_TOL_PP:
        return 0
    return 1 if value > 0 else -1


def _paired_membership(
    audit: pd.DataFrame,
    *,
    k: int,
    study_logs: Sequence[str],
) -> list[str] | None:
    if audit is None or audit.empty:
        return None
    if "included_in_paired" not in audit.columns or "log_name" not in audit.columns:
        return None
    block = audit[audit["k"].astype(int) == int(k)]
    if block.empty:
        return None
    included = {
        str(row["log_name"])
        for _, row in block.iterrows()
        if bool(row["included_in_paired"])
    }
    return [log for log in study_logs if log in included]


def _paired_key(config: Configuration) -> tuple[int | None, int, float]:
    return (config.xmin, int(config.min_types), float(config.p_gof))


def _build_paired_by_key(
    experiment_dir: Path,
    configs: Sequence[Configuration],
    study_logs: Sequence[str],
    ks: Sequence[int],
) -> dict[tuple[int | None, int, float], tuple[pd.DataFrame, pd.DataFrame]]:
    out: dict[tuple[int | None, int, float], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for key in {_paired_key(config) for config in configs}:
        xmin, min_types, p_gof = key
        summary, audit = build_paired_gof_tables(
            experiment_dir,
            study_logs=study_logs,
            xmin_settings=(xmin,),
            ks=ks,
            min_fitted_types_grid=(min_types,),
            gof_thresholds=(p_gof,),
        )
        out[key] = (summary.reset_index(drop=True), audit.reset_index(drop=True))
    return out


def _summary_by_k(summary: pd.DataFrame) -> dict[int, pd.Series]:
    return {int(row["k"]): row for _, row in summary.iterrows()}


def _tv_categories_for_config(
    assessed: Mapping[str, Mapping[str, Mapping[str, EndpointAssessment]]],
    comparisons_by_xmin: Mapping[int | None, Mapping[str, pd.DataFrame | None]],
    config: Configuration,
    study_logs: Sequence[str],
) -> dict[str, str]:
    endpoints = assessed[config.setting][REFERENCE_CONCEPT]
    comparisons = comparisons_by_xmin[config.xmin]
    return {
        log: classify_trace_variant(
            endpoints[log],
            comparisons.get(log),
            min_types=config.min_types,
            p_gof=config.p_gof,
            p_comp=config.p_comp,
        )
        for log in study_logs
    }


def _category_transitions(
    baseline: Mapping[str, str],
    current: Mapping[str, str],
    study_logs: Sequence[str],
) -> str:
    parts: list[str] = []
    for log in study_logs:
        before = baseline[log]
        after = current[log]
        if before != after:
            parts.append(f"{log} {before}→{after}")
    return "None" if not parts else " | ".join(parts)


def _paired_n_by_k(by_k: Mapping[int, pd.Series], ks: Sequence[int]) -> str:
    parts: list[str] = []
    for k in ks:
        row = by_k.get(int(k))
        n = 0 if row is None else int(row["N"])
        parts.append(f"{int(k)}={n}")
    return " | ".join(parts)


def _paired_n_range(by_k: Mapping[int, pd.Series], ks: Sequence[int]) -> str:
    values = [
        int(by_k[k]["N"])
        for k in ks
        if k in by_k and int(by_k[k]["N"]) > 0 and _finite_float(by_k[k]["paired_difference"]) is not None
    ]
    if not values:
        return "NA"
    return f"{min(values)} to {max(values)}"


def _eligibility_changes(
    baseline_audit: pd.DataFrame | None,
    current_audit: pd.DataFrame | None,
    *,
    ks: Sequence[int],
    study_logs: Sequence[str],
) -> str:
    if baseline_audit is None or current_audit is None:
        return "Unavailable"
    parts: list[str] = []
    for k in ks:
        before = _paired_membership(baseline_audit, k=k, study_logs=study_logs)
        after = _paired_membership(current_audit, k=k, study_logs=study_logs)
        if before is None or after is None:
            return "Unavailable"
        added = [log for log in after if log not in set(before)]
        removed = [log for log in before if log not in set(after)]
        if not added and not removed:
            continue
        parts.append(
            f"k={int(k)} added [{_format_log_list(added)}], "
            f"removed [{_format_log_list(removed)}]"
        )
    return "None" if not parts else " | ".join(parts)


def _difference_range_pp(by_k: Mapping[int, pd.Series], ks: Sequence[int]) -> str:
    values = [
        100.0 * float(by_k[k]["paired_difference"])
        for k in ks
        if k in by_k and _finite_float(by_k[k]["paired_difference"]) is not None
    ]
    if not values:
        return "NA"
    return f"{min(values):.1f} to {max(values):.1f}"


def _sign_reversals(
    baseline_by_k: Mapping[int, pd.Series],
    current_by_k: Mapping[int, pd.Series],
    ks: Sequence[int],
) -> list[int]:
    found: list[int] = []
    for k in ks:
        if k not in baseline_by_k or k not in current_by_k:
            continue
        left = _finite_float(baseline_by_k[k]["paired_difference"])
        right = _finite_float(current_by_k[k]["paired_difference"])
        if left is None or right is None:
            continue
        left_pp = 100.0 * left
        right_pp = 100.0 * right
        left_sign = _difference_sign_pp(left_pp)
        right_sign = _difference_sign_pp(right_pp)
        if left_sign is None or right_sign is None:
            continue
        if left_sign == 0 or right_sign == 0:
            continue
        if left_sign * right_sign < 0:
            found.append(int(k))
    return found


def _zero_orders(by_k: Mapping[int, pd.Series], ks: Sequence[int]) -> list[int]:
    found: list[int] = []
    for k in ks:
        if k not in by_k:
            continue
        value = _finite_float(by_k[k]["paired_difference"])
        if value is None:
            continue
        if _difference_sign_pp(100.0 * value) == 0:
            found.append(int(k))
    return found


def _unavailable_orders(by_k: Mapping[int, pd.Series], ks: Sequence[int]) -> list[int]:
    found: list[int] = []
    for k in ks:
        if k not in by_k or _finite_float(by_k[k]["paired_difference"]) is None:
            found.append(int(k))
    return found


def _availability_changes(
    baseline_by_k: Mapping[int, pd.Series],
    current_by_k: Mapping[int, pd.Series],
    ks: Sequence[int],
) -> str:
    became_available: list[int] = []
    became_unavailable: list[int] = []
    for k in ks:
        before = k in baseline_by_k and _finite_float(baseline_by_k[k]["paired_difference"]) is not None
        after = k in current_by_k and _finite_float(current_by_k[k]["paired_difference"]) is not None
        if after and not before:
            became_available.append(int(k))
        if before and not after:
            became_unavailable.append(int(k))
    if not became_available and not became_unavailable:
        return "None"
    return (
        f"available [{_format_orders(became_available)}] | "
        f"unavailable [{_format_orders(became_unavailable)}]"
    )


def _holm_significant(by_k: Mapping[int, pd.Series], ks: Sequence[int]) -> list[int] | None:
    any_valid = False
    found: list[int] = []
    for k in ks:
        if k not in by_k:
            continue
        p_holm = _finite_float(by_k[k]["mcnemar_p_holm"])
        if p_holm is None:
            continue
        any_valid = True
        if p_holm < SIGNIFICANCE:
            found.append(int(k))
    if not any_valid:
        return None
    return found


def _holm_changes(
    baseline_by_k: Mapping[int, pd.Series],
    current_by_k: Mapping[int, pd.Series],
    ks: Sequence[int],
) -> str:
    gained: list[int] = []
    lost: list[int] = []
    for k in ks:
        if k not in baseline_by_k or k not in current_by_k:
            continue
        before = _finite_float(baseline_by_k[k]["mcnemar_p_holm"])
        after = _finite_float(current_by_k[k]["mcnemar_p_holm"])
        if before is None or after is None:
            continue
        before_sig = before < SIGNIFICANCE
        after_sig = after < SIGNIFICANCE
        if after_sig and not before_sig:
            gained.append(int(k))
        if before_sig and not after_sig:
            lost.append(int(k))
    if not gained and not lost:
        return "None"
    return (
        f"newly significant [{_format_orders(gained)}] | "
        f"no longer significant [{_format_orders(lost)}]"
    )


def _incomplete_row(setting: str, status: str) -> dict[str, object]:
    row = {column: "NA" for column in SUMMARY_COLUMNS}
    row["setting"] = setting
    row["input_status"] = status
    return row


def _build_summary_row(
    config: Configuration,
    *,
    input_status: str,
    categories: Mapping[str, str],
    baseline_categories: Mapping[str, str],
    paired_summary: pd.DataFrame,
    paired_audit: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    baseline_audit: pd.DataFrame,
    study_logs: Sequence[str],
    ks: Sequence[int],
    is_baseline: bool,
) -> dict[str, object]:
    counts = {code: 0 for code in ("A", "B", "C", "D", "E")}
    for log in study_logs:
        counts[categories[log]] += 1
    eligible_n = counts["B"] + counts["C"] + counts["D"] + counts["E"]
    nonreject_n = counts["C"] + counts["D"] + counts["E"]
    if eligible_n == 0:
        pct: object = "NA"
    else:
        pct = 100.0 * nonreject_n / eligible_n

    by_k = _summary_by_k(paired_summary)
    baseline_by_k = _summary_by_k(baseline_summary)
    significant = _holm_significant(by_k, ks)

    if is_baseline:
        transitions = "Reference"
        eligibility = "Reference"
        reversals = "Reference"
        availability = "Reference"
        holm_changes = "Reference"
    else:
        transitions = _category_transitions(baseline_categories, categories, study_logs)
        eligibility = _eligibility_changes(
            baseline_audit,
            paired_audit,
            ks=ks,
            study_logs=study_logs,
        )
        reversals = _format_orders(_sign_reversals(baseline_by_k, by_k, ks))
        availability = _availability_changes(baseline_by_k, by_k, ks)
        holm_changes = _holm_changes(baseline_by_k, by_k, ks)

    return {
        "setting": config.setting,
        "input_status": input_status,
        "tv_category_transitions": transitions,
        "tv_A_count": counts["A"],
        "tv_B_count": counts["B"],
        "tv_C_count": counts["C"],
        "tv_D_count": counts["D"],
        "tv_E_count": counts["E"],
        "tv_eligible_n": eligible_n,
        "tv_gof_nonreject_n": nonreject_n,
        "tv_gof_nonreject_pct": pct,
        "paired_n_by_k": _paired_n_by_k(by_k, ks),
        "paired_n_range": _paired_n_range(by_k, ks),
        "paired_eligibility_changes": eligibility,
        "paired_difference_range_pp": _difference_range_pp(by_k, ks),
        "paired_sign_reversal_orders": reversals,
        "paired_zero_difference_orders": _format_orders(_zero_orders(by_k, ks)),
        "paired_unavailable_orders": _format_orders(_unavailable_orders(by_k, ks)),
        "paired_availability_changes": availability,
        "paired_holm_significant_orders": (
            "NA" if significant is None else _format_orders(significant)
        ),
        "paired_holm_significance_changes": holm_changes,
    }


def _validate_summary(
    summary: pd.DataFrame,
    *,
    study_logs: Sequence[str],
    categories_by_setting: Mapping[str, Mapping[str, str]],
    paired_by_setting: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> list[str]:
    warnings_out: list[str] = []
    n_logs = len(study_logs)
    baseline = summary[summary["setting"] == "baseline"].iloc[0]
    baseline_cats = categories_by_setting["baseline"]
    baseline_summary, baseline_audit = paired_by_setting["baseline"]
    baseline_by_k = _summary_by_k(baseline_summary)

    for _, row in summary.iterrows():
        setting = str(row["setting"])
        if str(row["input_status"]) != "ok":
            warnings_out.append(f"{setting}: {row['input_status']}")
            continue
        total = (
            int(row["tv_A_count"])
            + int(row["tv_B_count"])
            + int(row["tv_C_count"])
            + int(row["tv_D_count"])
            + int(row["tv_E_count"])
        )
        if total != n_logs:
            warnings_out.append(
                f"{setting}: category counts sum to {total}, expected {n_logs}"
            )
        eligible = int(row["tv_B_count"]) + int(row["tv_C_count"]) + int(row["tv_D_count"]) + int(
            row["tv_E_count"]
        )
        nonreject = int(row["tv_C_count"]) + int(row["tv_D_count"]) + int(row["tv_E_count"])
        if int(row["tv_eligible_n"]) != eligible:
            warnings_out.append(
                f"{setting}: tv_eligible_n={row['tv_eligible_n']} != B+C+D+E={eligible}"
            )
        if int(row["tv_gof_nonreject_n"]) != nonreject:
            warnings_out.append(
                f"{setting}: tv_gof_nonreject_n={row['tv_gof_nonreject_n']} != C+D+E={nonreject}"
            )

    for setting in ("comp_005", "comp_020"):
        if setting not in categories_by_setting or setting not in paired_by_setting:
            continue
        if str(summary.set_index("setting").loc[setting, "input_status"]) != "ok":
            continue
        cats = categories_by_setting[setting]
        for log in study_logs:
            before = baseline_cats[log]
            after = cats[log]
            if before in {"A", "B"} or after in {"A", "B"}:
                if before != after:
                    warnings_out.append(
                        f"{setting}: A/B category changed for {log}: {before}→{after}"
                    )
            # GOF non-rejection membership (C/D/E vs not) must match baseline.
            before_nr = before in {"C", "D", "E"}
            after_nr = after in {"C", "D", "E"}
            if before_nr != after_nr:
                warnings_out.append(
                    f"{setting}: GOF non-rejection changed for {log}: {before}→{after}"
                )
        current_summary, current_audit = paired_by_setting[setting]
        current_by_k = _summary_by_k(current_summary)
        for k in PAPER_KS:
            before_m = _paired_membership(baseline_audit, k=k, study_logs=study_logs)
            after_m = _paired_membership(current_audit, k=k, study_logs=study_logs)
            if before_m != after_m:
                warnings_out.append(f"{setting}: paired membership changed at k={k}")
            if k not in baseline_by_k or k not in current_by_k:
                continue
            for column in (
                "paired_difference",
                "ci95_lower",
                "ci95_upper",
                "mcnemar_p",
                "mcnemar_p_holm",
                "N",
            ):
                left = _finite_float(baseline_by_k[k][column])
                right = _finite_float(current_by_k[k][column])
                if not _close(left, right):
                    warnings_out.append(
                        f"{setting}: paired {column} changed at k={k}"
                    )
                    break
    return warnings_out


def build_sensitivity(
    experiment_dir: Path,
    *,
    study_logs: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
    check_saved_baseline: bool = False,
) -> SensitivityResult:
    """Build the sensitivity summary from one experiment directory."""
    del check_saved_baseline  # retained for call-site compatibility
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {experiment_dir}")
    logs = list(study_logs) if study_logs is not None else list(ALL_REAL_LOG_DATASETS)
    selected = list(concepts) if concepts is not None else list(CONCEPTS)
    if REFERENCE_CONCEPT not in selected:
        raise ValueError(f"concepts must include {REFERENCE_CONCEPT}")
    ks = _ks_from_concepts(selected)
    if not ks:
        raise ValueError("concepts must include at least one n-gram order")

    statuses = {config.setting: _input_status(experiment_dir, config) for config in CONFIGURATIONS}
    complete = [config for config in CONFIGURATIONS if statuses[config.setting] == "ok"]
    incomplete = [config for config in CONFIGURATIONS if statuses[config.setting] != "ok"]

    assessed, _loaded = _assess_all(experiment_dir, complete, selected, logs)
    comparisons_by_xmin: dict[int | None, dict[str, pd.DataFrame | None]] = {}
    for xmin in {config.xmin for config in complete}:
        comparisons_by_xmin[xmin] = load_concept_comparisons(
            xmin_dir(experiment_dir, xmin), REFERENCE_CONCEPT, logs
        )

    paired_by_key = _build_paired_by_key(experiment_dir, complete, logs, ks)
    categories_by_setting: dict[str, dict[str, str]] = {}
    paired_by_setting: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for config in complete:
        categories_by_setting[config.setting] = _tv_categories_for_config(
            assessed, comparisons_by_xmin, config, logs
        )
        paired_by_setting[config.setting] = paired_by_key[_paired_key(config)]

    if "baseline" not in paired_by_setting:
        raise RuntimeError("baseline inputs are incomplete; cannot build sensitivity summary")

    baseline_categories = categories_by_setting["baseline"]
    baseline_summary, baseline_audit = paired_by_setting["baseline"]

    rows: list[dict[str, object]] = []
    warning_messages: list[str] = []
    for config in CONFIGURATIONS:
        status = statuses[config.setting]
        if status != "ok":
            warning_messages.append(f"{config.setting}: {status}")
            print(f"WARNING: {config.setting}: {status}", file=sys.stderr)
            rows.append(_incomplete_row(config.setting, status))
            continue
        paired_summary, paired_audit = paired_by_setting[config.setting]
        rows.append(
            _build_summary_row(
                config,
                input_status=status,
                categories=categories_by_setting[config.setting],
                baseline_categories=baseline_categories,
                paired_summary=paired_summary,
                paired_audit=paired_audit,
                baseline_summary=baseline_summary,
                baseline_audit=baseline_audit,
                study_logs=logs,
                ks=ks,
                is_baseline=config.setting == "baseline",
            )
        )

    summary = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
    warning_messages.extend(
        _validate_summary(
            summary,
            study_logs=logs,
            categories_by_setting=categories_by_setting,
            paired_by_setting=paired_by_setting,
        )
    )
    for message in warning_messages:
        if message.startswith(tuple(f"{config.setting}:" for config in incomplete)):
            continue
        print(f"WARNING: {message}", file=sys.stderr)
    return SensitivityResult(configurations_frame(), summary, warning_messages)


def _format_csv_cell(value: object) -> str:
    if isinstance(value, str):
        if any(character in value for character in ",\"\n"):
            return '"' + value.replace('"', '""') + '"'
        return value
    if _missing(value):
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    number = float(value)
    if math.isnan(number):
        return "NA"
    if float(number).is_integer() and abs(number) < 1e12:
        # Keep integer counts unquoted; percentages may be integral too.
        return repr(number) if abs(number - round(number)) > 0 else str(int(round(number)))
    return repr(number)


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(str(column) for column in frame.columns)
    lines = [header]
    for _, row in frame.iterrows():
        lines.append(",".join(_format_csv_cell(row[column]) for column in frame.columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_SENSITIVITY_LATEX_UNICODE = (
    ("→", r"$\rightarrow$"),
    ("←", r"$\leftarrow$"),
    ("−", r"--"),
    ("–", r"--"),
    ("—", r"---"),
)


def _escape_latex(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    for src, dst in _SENSITIVITY_LATEX_UNICODE:
        text = text.replace(src, "\0" + dst + "\0")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    # Temporarily protect injected math / dashes, then escape, then restore.
    protected: list[str] = []
    chunks: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\0":
            end = text.find("\0", i + 1)
            if end < 0:
                chunks.append(replacements.get(text[i], text[i]))
                i += 1
                continue
            protected.append(text[i + 1 : end])
            chunks.append(f"\x01{len(protected) - 1}\x01")
            i = end + 1
            continue
        chunks.append(text[i])
        i += 1
    joined = "".join(chunks)
    escaped = "".join(replacements.get(ch, ch) for ch in joined)
    for index, payload in enumerate(protected):
        escaped = escaped.replace(f"\x01{index}\x01", payload)
    return escaped


SUMMARY_TEX_COLUMNS: tuple[str, ...] = (
    "setting",
    "tv_category_transitions",
    "tv_A_count",
    "tv_B_count",
    "tv_C_count",
    "tv_D_count",
    "tv_E_count",
    "tv_eligible_n",
    "tv_gof_nonreject_n",
    "tv_gof_nonreject_pct",
    "paired_n_range",
    "paired_difference_range_pp",
    "paired_holm_significant_orders",
)

SUMMARY_TEX_HEADERS: Mapping[str, str] = {
    "setting": "Setting",
    "tv_category_transitions": (
        r"\begin{tabular}{@{}c@{}}TV category\\changes vs\\baseline\end{tabular}"
    ),
    "tv_A_count": "A",
    "tv_B_count": "B",
    "tv_C_count": "C",
    "tv_D_count": "D",
    "tv_E_count": "E",
    "tv_eligible_n": (
        r"\begin{tabular}{@{}c@{}}Elig\-\\ible\end{tabular}"
    ),
    "tv_gof_nonreject_n": (
        r"\begin{tabular}{@{}c@{}}GOF\\NR\end{tabular}"
    ),
    "tv_gof_nonreject_pct": (
        r"\begin{tabular}{@{}c@{}}GOF\\NR\%\end{tabular}"
    ),
    "paired_n_range": (
        r"\begin{tabular}{@{}c@{}}Paired $N$\\range\end{tabular}"
    ),
    "paired_difference_range_pp": (
        r"\begin{tabular}{@{}c@{}}Diff.\ range\\(pp)\end{tabular}"
    ),
    "paired_holm_significant_orders": (
        r"\begin{tabular}{@{}c@{}}Holm-sig.\\$k$\end{tabular}"
    ),
}

# Column specs for portrait layout (p-columns allow line breaks).
SUMMARY_TEX_COL_SPEC: Mapping[str, str] = {
    "setting": "l",
    "tv_category_transitions": r">{\raggedright\arraybackslash}p{2.6cm}",
    "tv_A_count": "c",
    "tv_B_count": "c",
    "tv_C_count": "c",
    "tv_D_count": "c",
    "tv_E_count": "c",
    "tv_eligible_n": "c",
    "tv_gof_nonreject_n": "c",
    "tv_gof_nonreject_pct": "c",
    "paired_n_range": r">{\centering\arraybackslash}p{1.35cm}",
    "paired_difference_range_pp": r">{\centering\arraybackslash}p{1.55cm}",
    "paired_holm_significant_orders": r">{\raggedright\arraybackslash}p{2.4cm}",
}


def _format_wrapped_tex_lines(value: object, *, sep: str = "|") -> str:
    """Split a pipe-/comma-separated field into a top-aligned tabular."""
    if _missing(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parts = [part.strip() for part in text.split(sep) if part.strip()]
    if len(parts) <= 1:
        return _escape_latex(text)
    body = r" \\ ".join(_escape_latex(part) for part in parts)
    return rf"\begin{{tabular}}[t]{{@{{}}l@{{}}}}{body}\end{{tabular}}"


def _format_summary_tex_cell(column: str, value: object) -> str:
    if _missing(value):
        return ""
    if column == "tv_category_transitions":
        return _format_wrapped_tex_lines(value, sep="|")
    if column == "paired_holm_significant_orders":
        # Keep significant k values comma-separated (as in the CSV).
        return _escape_latex(value)
    if column in {"paired_n_range", "paired_difference_range_pp"}:
        text = str(value).strip().replace(" to ", " -- ")
        return _escape_latex(text)
    if column == "tv_gof_nonreject_pct":
        number = float(value)
        return _escape_latex(f"{number:.1f}")
    if column.endswith("_count") or column.endswith("_n"):
        try:
            return _escape_latex(str(int(value)))
        except (TypeError, ValueError):
            return _escape_latex(value)
    return _escape_latex(value)


def configurations_dataframe_to_latex(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    headers = {
        "setting": "Setting",
        "xmin": "$x_{\\min}$ policy",
        "min_types": "Min.\\ fitted types",
        "p_gof": "GOF $p$ threshold",
        "p_comp": "Comparison $p$ threshold",
    }
    header_cells = [
        headers[c] if c in headers else _escape_latex(c) for c in columns
    ]
    col_spec = "l" + ("c" * max(len(columns) - 1, 0))
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column in {"p_gof", "p_comp"}:
                cells.append(_escape_latex(f"{float(value):.2f}"))
            elif column == "min_types":
                cells.append(_escape_latex(str(int(value))))
            else:
                cells.append(_escape_latex(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def sensitivity_summary_dataframe_to_latex(frame: pd.DataFrame) -> str:
    columns = [c for c in SUMMARY_TEX_COLUMNS if c in frame.columns]
    col_spec = "".join(
        SUMMARY_TEX_COL_SPEC.get(c, "c") for c in columns
    )
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(SUMMARY_TEX_HEADERS.get(c, _escape_latex(c)) for c in columns)
        + r" \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        cells = [_format_summary_tex_cell(c, row[c]) for c in columns]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return (
        "\n".join(lines)
        + "\n"
        + "\\begin{flushleft}\\footnotesize\n"
        "TV = trace-variant abstraction. Category counts A--E use the "
        "configuration thresholds. GOF NR = number (and percent) of eligible "
        "logs with GOF non-rejection. Paired columns summarise McNemar "
        "comparisons of GOF non-rejection versus trace variants across "
        "$k=1,\\ldots,20$. Full text fields "
        "(eligibility changes, availability changes, etc.) remain in "
        "\\texttt{sensitivity\\_summary.csv}.\n"
        "\\end{flushleft}\n"
    )


def write_table_tex(body: str, path: Path, *, caption: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    if caption:
        parts.append(f"% {caption}")
    parts.append(body.rstrip())
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_sensitivity_tex(result: SensitivityResult, output_dir: Path) -> None:
    """Write LaTeX companions next to sensitivity CSV tables."""
    output_dir = Path(output_dir)
    write_table_tex(
        configurations_dataframe_to_latex(result.configurations),
        output_dir / "configurations.tex",
        caption=(
            "One-at-a-time sensitivity configurations. "
            "The baseline uses adaptive $x_{\\min}$, 50 fitted types, "
            "and GOF / comparison thresholds of 0.10."
        ),
    )
    write_table_tex(
        sensitivity_summary_dataframe_to_latex(result.summary),
        output_dir / "sensitivity_summary.tex",
        caption=(
            "Sensitivity summary across configurations: trace-variant category "
            "counts, GOF non-rejection, and paired GOF comparison diagnostics."
        ),
    )


def write_sensitivity_tex_from_csv(output_dir: Path) -> list[Path]:
    """Regenerate sensitivity ``.tex`` files from existing CSV outputs."""
    output_dir = Path(output_dir)
    written: list[Path] = []
    config_csv = output_dir / "configurations.csv"
    summary_csv = output_dir / "sensitivity_summary.csv"
    if config_csv.is_file():
        frame = pd.read_csv(config_csv)
        tex_path = output_dir / "configurations.tex"
        write_table_tex(
            configurations_dataframe_to_latex(frame),
            tex_path,
            caption=(
                "One-at-a-time sensitivity configurations. "
                "The baseline uses adaptive $x_{\\min}$, 50 fitted types, "
                "and GOF / comparison thresholds of 0.10."
            ),
        )
        written.append(tex_path)
    if summary_csv.is_file():
        frame = pd.read_csv(summary_csv)
        tex_path = output_dir / "sensitivity_summary.tex"
        write_table_tex(
            sensitivity_summary_dataframe_to_latex(frame),
            tex_path,
            caption=(
                "Sensitivity summary across configurations: trace-variant category "
                "counts, GOF non-rejection, and paired GOF comparison diagnostics."
            ),
        )
        written.append(tex_path)
    return written


def cleanup_obsolete_outputs(output_dir: Path) -> list[str]:
    """Delete superseded sensitivity presentation files. Returns removed paths."""
    output_dir = Path(output_dir)
    removed: list[str] = []
    for name in OBSOLETE_OUTPUT_NAMES:
        path = output_dir / name
        if path.is_file():
            path.unlink()
            removed.append(str(path))
    leftover_dir = output_dir / "paired_gof_comparison"
    if leftover_dir.is_dir():
        # Only plots previously generated under sensitivity/; safe to remove.
        for path in sorted(leftover_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
            elif path.is_dir():
                path.rmdir()
        leftover_dir.rmdir()
        removed.append(str(leftover_dir))
    return removed


def write_sensitivity(result: SensitivityResult, output_dir: Path) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    removed = cleanup_obsolete_outputs(output_dir)
    write_table(result.configurations, output_dir / "configurations.csv")
    write_table(result.summary, output_dir / "sensitivity_summary.csv")
    write_sensitivity_tex(result, output_dir)
    return removed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sensitivity summary of saved power-law GOF and category results. "
            "Does not refit or rerun GOF or likelihood-ratio tests."
        )
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=PL_STATS_REAL_DIR,
        help="Experiment directory (default: results/pl_stats/real)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: results/pl_stats/real/sensitivity)",
    )
    args = parser.parse_args(argv)
    result = build_sensitivity(args.experiment, check_saved_baseline=False)
    removed = write_sensitivity(result, args.output_dir)
    ok = result.summary[result.summary["input_status"] == "ok"]
    incomplete = result.summary[result.summary["input_status"] != "ok"]
    print(
        f"Wrote {args.output_dir / 'sensitivity_summary.csv'} "
        f"({len(ok)} complete, {len(incomplete)} incomplete)."
    )
    print(f"Settings processed: {', '.join(result.summary['setting'].astype(str))}")
    if incomplete.shape[0]:
        print("Missing / incomplete inputs:")
        for _, row in incomplete.iterrows():
            print(f"  - {row['setting']}: {row['input_status']}")
    if result.warnings:
        print(f"Validation warnings: {len(result.warnings)}")
        for message in result.warnings:
            print(f"  - {message}")
    else:
        print("Validation warnings: none")
    if removed:
        print("Obsolete sensitivity outputs removed:")
        for path in removed:
            print(f"  - {path}")
    else:
        print("Obsolete sensitivity outputs removed: none")


if __name__ == "__main__":
    main()
