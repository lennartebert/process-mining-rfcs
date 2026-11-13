"""Sensitivity grid: thresholds and fitted-type cutoffs."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.powerlaw_statistics.analysis_helpers.paired_gof import MODEL, PAPER_KS
from analyses.powerlaw_statistics.analysis_helpers.paths import concept_dirname
from analyses.powerlaw_statistics.analysis_helpers.sensitivity import (
    CATEGORY_SETTING_NAMES,
    CONCEPTS,
    CONFIGURATIONS,
    PAIRED_SETTING_NAMES,
    SUMMARY_COLUMNS,
    _ks_from_concepts,
    build_sensitivity,
    load_concept_comparisons,
    write_sensitivity,
)

STUDY_LOGS = (
    "BORDER",
    "D0",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "INVALID",
    "LOW",
    "NO_LLR",
    "ONLY_ADAPTIVE",
)
DISCORDANT = tuple(f"D{index}" for index in range(8))
CONCEPTS_FIXTURE = ("n1", "n2", "variants")


def _summary_row(
    log: str,
    *,
    xmin: float,
    n_fitted: int,
    gof_p: float | None,
    fit_valid: bool = True,
) -> dict:
    return {
        "log_name": log,
        "xmin": xmin,
        "n_fitted_types": n_fitted,
        "fit_valid": fit_valid,
        "gof_p": gof_p,
        "gof_kind": "refit_all_parameters" if fit_valid else "skipped_invalid_fit",
    }


def _write_summary(root: Path, xmin_name: str, concept: str, rows: list[dict]) -> None:
    model = (
        root
        / xmin_name
        / "raw_results"
        / concept_dirname(concept)
        / "lower_bounded_power_law"
    )
    model.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(model / "summary.csv", index=False)


def _comparison_rows(log: str, pattern: str) -> list[dict]:
    if pattern == "alternative":
        specs = (
            ("exponential", 1.0, 0.40, MODEL),
            ("lognormal", -1.0, 0.07, "lognormal"),
            ("stretched_exponential", 1.0, 0.40, MODEL),
            ("truncated_power_law", -5.0, 0.01, "truncated_power_law"),
        )
    elif pattern == "power_law":
        specs = (
            ("exponential", 2.0, 0.12, MODEL),
            ("lognormal", 2.0, 0.12, MODEL),
            ("stretched_exponential", 2.0, 0.12, MODEL),
            ("truncated_power_law", -5.0, 0.01, "truncated_power_law"),
        )
    elif pattern == "missing":
        specs = (
            ("exponential", math.nan, math.nan, None),
            ("lognormal", math.nan, math.nan, None),
            ("stretched_exponential", math.nan, math.nan, None),
            ("truncated_power_law", math.nan, math.nan, None),
        )
    else:
        raise AssertionError(pattern)
    rows = []
    for model_2, r_value, p_value, preferred in specs:
        rows.append(
            {
                "log_name": log,
                "model_1": MODEL,
                "model_2": model_2,
                "R": r_value,
                "p": p_value,
                "preferred_model": preferred,
            }
        )
    return rows


def _pattern(log: str) -> str:
    if log == "NO_LLR":
        return "missing"
    if log in {"BORDER", "D4", "D5", "D6", "D7"}:
        return "power_law"
    return "alternative"


def _adaptive_gof(log: str, concept: str) -> float | None:
    if log == "INVALID":
        return None
    if log == "BORDER" and concept == "n1":
        return 0.099
    if log in DISCORDANT and concept != "variants":
        return 0.50
    return 0.50 if concept == "variants" or log not in DISCORDANT else 0.01


def _fitted_count(log: str) -> int:
    if log == "LOW":
        return 10
    if log == "INVALID":
        return 80
    # Eligible under the baseline min_types=50 cutoff and the min_types_30 contrast.
    return 60


def _rows_for(logs: tuple[str, ...], *, xmin: float) -> dict[str, list[dict]]:
    by_concept = {concept: [] for concept in CONCEPTS_FIXTURE}
    for log in logs:
        for concept in CONCEPTS_FIXTURE:
            by_concept[concept].append(
                _summary_row(
                    log,
                    xmin=xmin,
                    n_fitted=_fitted_count(log),
                    gof_p=_adaptive_gof(log, concept),
                    fit_valid=log != "INVALID",
                )
            )
    return by_concept


def _write_fixture(root: Path) -> None:
    adaptive = _rows_for(STUDY_LOGS, xmin=4)
    for concept, rows in adaptive.items():
        _write_summary(root, "xmin-adaptive", concept, rows)
    comparison_rows: list[dict] = []
    for log in STUDY_LOGS:
        if log == "INVALID":
            continue
        comparison_rows.extend(_comparison_rows(log, _pattern(log)))
    for concept in CONCEPTS_FIXTURE:
        model = (
            root
            / "xmin-adaptive"
            / "raw_results"
            / concept_dirname(concept)
            / "lower_bounded_power_law"
        )
        pd.DataFrame(comparison_rows).to_csv(model / "comparison.csv", index=False)


class SensitivityGridTests(unittest.TestCase):
    def test_published_grid_and_holm_family(self) -> None:
        settings = tuple(config.setting for config in CONFIGURATIONS)
        self.assertEqual(settings[: len(PAIRED_SETTING_NAMES)], PAIRED_SETTING_NAMES)
        self.assertEqual(
            settings,
            PAIRED_SETTING_NAMES + ("comp_005", "comp_020"),
        )
        self.assertEqual(CATEGORY_SETTING_NAMES, ("baseline", "comp_005", "comp_020"))
        baseline = CONFIGURATIONS[0]
        self.assertEqual(
            (baseline.xmin, baseline.min_types, baseline.p_gof, baseline.p_comp),
            (None, 50, 0.10, 0.10),
        )
        alt_min = next(config for config in CONFIGURATIONS if config.setting == "min_types_30")
        self.assertEqual(alt_min.min_types, 30)
        self.assertEqual(_ks_from_concepts(CONCEPTS), PAPER_KS)
        self.assertEqual(list(SUMMARY_COLUMNS)[0], "setting")


class SensitivityFixtureTests(unittest.TestCase):
    def test_thresholds_and_cutoffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root)
            result = build_sensitivity(
                root,
                study_logs=STUDY_LOGS,
                concepts=CONCEPTS_FIXTURE,
            )
            output = root / "out"
            removed = write_sensitivity(result, output)
            self.assertFalse((output / "paired_gof_sensitivity.csv").exists())
            self.assertFalse((output / "README.md").exists())
            self.assertEqual(removed, [])
            self._assert_summary(result)
            self._assert_comp_invariants(result)
            self._assert_roundtrip(output)

    def _assert_summary(self, result) -> None:
        summary = result.summary.set_index("setting")
        self.assertTrue((summary["input_status"] == "ok").all())
        self.assertEqual(int(summary.loc["baseline", "tv_eligible_n"]), 11)
        self.assertEqual(int(summary.loc["baseline", "tv_gof_nonreject_n"]), 11)
        self.assertEqual(int(summary.loc["min_types_0", "tv_eligible_n"]), 12)
        self.assertEqual(int(summary.loc["min_types_15", "tv_eligible_n"]), 11)
        self.assertEqual(int(summary.loc["min_types_30", "tv_eligible_n"]), 11)
        self.assertEqual(summary.loc["baseline", "tv_category_transitions"], "Reference")
        self.assertEqual(summary.loc["baseline", "paired_eligibility_changes"], "Reference")
        self.assertEqual(summary.loc["baseline", "paired_zero_difference_orders"], "2")
        # NO_LLR is non-rejected with missing LLR → compose-equivalent E.
        self.assertEqual(int(summary.loc["baseline", "tv_E_count"]), 6)
        self.assertIn("C→E", summary.loc["comp_005", "tv_category_transitions"])
        self.assertIn("E→D", summary.loc["comp_020", "tv_category_transitions"])
        self.assertEqual(
            int(summary.loc["baseline", "tv_A_count"])
            + int(summary.loc["baseline", "tv_B_count"])
            + int(summary.loc["baseline", "tv_C_count"])
            + int(summary.loc["baseline", "tv_D_count"])
            + int(summary.loc["baseline", "tv_E_count"]),
            len(STUDY_LOGS),
        )

    def _assert_comp_invariants(self, result) -> None:
        summary = result.summary.set_index("setting")
        for setting in ("comp_005", "comp_020"):
            self.assertEqual(
                int(summary.loc[setting, "tv_eligible_n"]),
                int(summary.loc["baseline", "tv_eligible_n"]),
            )
            self.assertEqual(
                int(summary.loc[setting, "tv_gof_nonreject_n"]),
                int(summary.loc["baseline", "tv_gof_nonreject_n"]),
            )
            self.assertEqual(summary.loc[setting, "paired_eligibility_changes"], "None")
            self.assertEqual(summary.loc[setting, "paired_availability_changes"], "None")
            self.assertEqual(
                summary.loc[setting, "paired_n_by_k"],
                summary.loc["baseline", "paired_n_by_k"],
            )
            self.assertEqual(
                summary.loc[setting, "paired_difference_range_pp"],
                summary.loc["baseline", "paired_difference_range_pp"],
            )

    def _assert_roundtrip(self, output: Path) -> None:
        path = output / "sensitivity_summary.csv"
        text = path.read_text(encoding="utf-8")
        self.assertIn("input_status", text)
        frame = pd.read_csv(path)
        self.assertEqual(list(frame["setting"]), [config.setting for config in CONFIGURATIONS])
        configs = pd.read_csv(output / "configurations.csv")
        self.assertEqual(list(configs["setting"]), list(frame["setting"]))


class ComparisonSourceTests(unittest.TestCase):
    def test_disagreeing_sources_are_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = (
                root
                / "raw_results"
                / concept_dirname("variants")
                / "lower_bounded_power_law"
            )
            model.mkdir(parents=True)
            summary = pd.DataFrame(_comparison_rows("ACCRE", "alternative"))
            shard = pd.DataFrame(_comparison_rows("ACCRE", "power_law"))
            summary.to_csv(model / "comparison.csv", index=False)
            shard.to_csv(model / "comparison_ACCRE.csv", index=False)
            loaded = load_concept_comparisons(root, "variants", ["ACCRE"])
            self.assertIsNone(loaded["ACCRE"])


if __name__ == "__main__":
    unittest.main()
