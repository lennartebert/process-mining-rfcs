"""Paired GOF comparisons: Newcombe method 10, McNemar, and saved-result rules."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import ALL_REAL_LOG_DATASETS, PL_STATS_DIR
from analyses.powerlaw_statistics.analysis_helpers.paths import concept_dirname


def _load_paired():
    path = (
        REPO_ROOT
        / "analyses"
        / "powerlaw_statistics"
        / "analysis_helpers"
        / "paired_gof.py"
    )
    spec = importlib.util.spec_from_file_location("paired_gof_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PAIRED = _load_paired()


def _row(
    log: str,
    *,
    concept: str,
    xmin: float | None = 1.0,
    n_fitted_types: int | None = 40,
    fit_valid: object = True,
    gof_p: float | None = 0.5,
    gof_kind: str | None = "refit_all_parameters",
    classification: str = "A = ignored label",
) -> dict:
    return {
        "log_name": log,
        "input_path": f"results/attachments/{concept}/{log}/attachments.csv.gz",
        "xmin": xmin,
        "n_fitted_types": n_fitted_types,
        "fit_valid": fit_valid,
        "gof_p": gof_p,
        "gof_kind": gof_kind,
        "classification": classification,
        "best_other_distribution": "stretched_exponential",
    }


def _write_rows(root: Path, xmin_name: str, concept: str, rows: list[dict]) -> None:
    model = (
        root
        / xmin_name
        / "raw_results"
        / concept_dirname(concept)
        / "lower_bounded_power_law"
    )
    model.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame["log_name"].nunique() == len(frame):
        for record in rows:
            pd.DataFrame([record]).to_csv(
                model / f"summary_{record['log_name']}.csv", index=False
            )
    else:
        frame.to_csv(model / "summary.csv", index=False)


class NewcombeMethod10Tests(unittest.TestCase):
    def test_table_iii_examples(self) -> None:
        for e, f, g, h, lower, upper in PAIRED.NEWCOMBE_1998_TABLE_III_METHOD_10:
            got_lower, got_upper = PAIRED.newcombe_method10_from_cells(e, f, g, h)
            self.assertLess(abs(got_lower - lower), 1e-4, msg=f"{(e, f, g, h)} lower")
            self.assertLess(abs(got_upper - upper), 1e-4, msg=f"{(e, f, g, h)} upper")

    def test_orientation_is_variants_minus_k(self) -> None:
        # n01=6, n10=0 -> difference (6-0)/6 = 1, which is the paper's (f-g)/n
        # with e=h=0, f=6, g=0.
        lower, upper = PAIRED.newcombe_method10_variants_minus_k(0, 6, 0, 0)
        paper_lower, paper_upper = PAIRED.newcombe_method10_from_cells(0, 6, 0, 0)
        self.assertEqual((lower, upper), (paper_lower, paper_upper))
        self.assertGreater(lower, 0.0)

    def test_swapping_endpoints_reverses_the_interval(self) -> None:
        forward = PAIRED.paired_effect(4, 6, 1, 3)
        swapped = PAIRED.paired_effect(4, 1, 6, 3)
        self.assertAlmostEqual(swapped["paired_difference"], -forward["paired_difference"])
        self.assertAlmostEqual(swapped["ci95_lower"], -forward["ci95_upper"])
        self.assertAlmostEqual(swapped["ci95_upper"], -forward["ci95_lower"])
        self.assertAlmostEqual(swapped["mcnemar_p"], forward["mcnemar_p"])


class McNemarAndHolmTests(unittest.TestCase):
    def test_six_versus_zero_discordant_pairs(self) -> None:
        pvalue = PAIRED.exact_mcnemar_p(0, 6, 0, 0)
        self.assertEqual(pvalue, 0.03125)

    def test_no_discordant_pairs_report_p_equals_one(self) -> None:
        self.assertEqual(PAIRED.exact_mcnemar_p(5, 0, 0, 4), 1.0)
        effect = PAIRED.paired_effect(5, 0, 0, 4)
        self.assertEqual(effect["mcnemar_p"], 1.0)
        self.assertEqual(effect["paired_difference"], 0.0)
        self.assertIsNotNone(effect["ci95_lower"])

    def test_empty_sample_is_unavailable(self) -> None:
        effect = PAIRED.paired_effect(0, 0, 0, 0)
        self.assertEqual(effect["N"], 0)
        self.assertIsNone(effect["proportion_k"])
        self.assertIsNone(effect["paired_difference"])
        self.assertIsNone(effect["ci95_lower"])
        self.assertIsNone(effect["mcnemar_p"])

    def test_holm_keeps_a_family_of_five_when_comparisons_are_missing(self) -> None:
        adjusted = PAIRED.holm_adjust([0.03125, None, None, None, None])
        self.assertAlmostEqual(adjusted[0], 0.15625)
        self.assertTrue(all(value is None for value in adjusted[1:]))

    def test_two_difference_formulas_agree(self) -> None:
        effect = PAIRED.paired_effect(3, 6, 1, 2)
        from_margins = effect["proportion_variants"] - effect["proportion_k"]
        from_discordance = (6 - 1) / 12
        self.assertAlmostEqual(effect["paired_difference"], from_margins)
        self.assertAlmostEqual(effect["paired_difference"], from_discordance)


class SavedResultComparisonTests(unittest.TestCase):
    def _build(self, root: Path, **kwargs):
        summary, audit = PAIRED.build_paired_gof_tables(root, **kwargs)
        PAIRED.validate_paired_tables(summary, audit)
        return summary, audit

    def test_eligibility_cutoff_and_ignored_category_labels(self) -> None:
        root = Path(self.id().replace(".", "_"))
        # Use a temp dir under the system temp via unittest's addCleanup pattern.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = ["LOW", "HIGH", "MISSING_P"]
            for concept, gof_p, n_fitted in (
                ("n1", 0.40, 20),
                ("variants", 0.40, 20),
            ):
                rows = [
                    _row(log, concept=concept, xmin=3, n_fitted_types=n_fitted, gof_p=gof_p)
                    for log in ("LOW", "HIGH")
                ]
                rows.append(
                    _row(
                        "MISSING_P",
                        concept=concept,
                        xmin=3,
                        n_fitted_types=40,
                        gof_p=None,
                        classification="(1) < 30 fitted types",
                    )
                )
                # HIGH has enough types for both cutoffs.
                rows[1]["n_fitted_types"] = 40
                _write_rows(root, "xmin-adaptive", concept, rows)
            summary, audit = self._build(
                root,
                study_logs=logs,
                xmin_settings=(None,),
                ks=(1,),
                min_fitted_types_grid=(15, 30),
                gof_thresholds=(0.05, 0.10),
            )
            low_cutoff = summary[summary["min_fitted_types"] == 15].iloc[0]
            high_cutoff = summary[summary["min_fitted_types"] == 30].iloc[0]
            self.assertEqual(int(low_cutoff["N"]), 2)
            self.assertEqual(int(high_cutoff["N"]), 1)
            self.assertLessEqual(int(high_cutoff["N"]), int(low_cutoff["N"]))
            missing = audit[
                (audit["log_name"] == "MISSING_P") & (audit["min_fitted_types"] == 15)
            ].iloc[0]
            self.assertIn("gof_p_unavailable", missing["exclusion_reason_k"])
            self.assertFalse(bool(missing["included_in_paired"]))
            labelled = audit[
                (audit["log_name"] == "LOW") & (audit["min_fitted_types"] == 15)
            ].iloc[0]
            self.assertTrue(bool(labelled["included_in_paired"]))
            self.assertEqual(int(labelled["outcome_k"]), 1)

    def test_gof_threshold_changes_outcomes_not_eligibility(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = ["BORDER"]
            for concept in ("n1", "variants"):
                _write_rows(
                    root,
                    "xmin-adaptive",
                    concept,
                    [_row("BORDER", concept=concept, xmin=4, gof_p=0.07, n_fitted_types=40)],
                )
            summary, audit = self._build(
                root,
                study_logs=logs,
                xmin_settings=(None,),
                ks=(1,),
                min_fitted_types_grid=(15,),
                gof_thresholds=(0.05, 0.10),
            )
            self.assertEqual(set(summary["N"].astype(int)), {1})
            outcomes = {
                float(row.gof_threshold): int(row.outcome_k)
                for row in audit.itertuples(index=False)
            }
            self.assertEqual(outcomes[0.05], 1)
            self.assertEqual(outcomes[0.10], 0)
            self.assertTrue(bool(audit["eligible_k"].all()))

    def test_percentage_gof_p_is_not_rescaled(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for concept in ("n1", "variants"):
                _write_rows(
                    root,
                    "xmin-adaptive",
                    concept,
                    [_row("PCT", concept=concept, xmin=2, gof_p=5.6)],
                )
            _, audit = self._build(
                root,
                study_logs=["PCT"],
                xmin_settings=(None,),
                ks=(1,),
                min_fitted_types_grid=(15,),
                gof_thresholds=(0.05,),
            )
            row = audit.iloc[0]
            self.assertIn("gof_p_out_of_range", row["exclusion_reason_k"])
            self.assertAlmostEqual(float(row["gof_p_k"]), 5.6)
            self.assertFalse(bool(row["included_in_paired"]))

    def test_duplicates_and_disagreement_are_not_averaged(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "xmin-adaptive" / "raw_results" / "n001" / "lower_bounded_power_law"
            model.mkdir(parents=True)
            duplicated = pd.DataFrame(
                [
                    _row("DUP", concept="n1", gof_p=0.0),
                    _row("DUP", concept="n1", gof_p=1.0),
                ]
            )
            duplicated.to_csv(model / "summary.csv", index=False)
            variant = root / "xmin-adaptive" / "raw_results" / "variants" / "lower_bounded_power_law"
            variant.mkdir(parents=True)
            pd.DataFrame([_row("DUP", concept="variants", gof_p=0.2)]).to_csv(
                variant / "summary_DUP.csv", index=False
            )
            # Shard and combined summary disagree for a second log's variants.
            pd.DataFrame([_row("MIX", concept="n1", gof_p=0.2)]).to_csv(
                model / "summary_MIX.csv", index=False
            )
            pd.DataFrame([_row("MIX", concept="variants", gof_p=0.2)]).to_csv(
                variant / "summary_MIX.csv", index=False
            )
            pd.DataFrame([_row("MIX", concept="variants", gof_p=0.9)]).to_csv(
                variant / "summary.csv", index=False
            )
            summary, audit = self._build(
                root,
                study_logs=["DUP", "MIX"],
                xmin_settings=(None,),
                ks=(1,),
                min_fitted_types_grid=(15,),
                gof_thresholds=(0.05,),
            )
            self.assertEqual(int(summary.iloc[0]["N"]), 0)
            dup = audit[audit["log_name"] == "DUP"].iloc[0]
            mix = audit[audit["log_name"] == "MIX"].iloc[0]
            self.assertIn("duplicate_results", dup["exclusion_reason_k"])
            self.assertTrue(math.isnan(float(dup["gof_p_k"])))
            self.assertIn("incompatible_run_configuration", mix["exclusion_reason_variants"])
            self.assertTrue(math.isnan(float(mix["gof_p_variants"])))

    def test_fixed_and_adaptive_experiments_are_not_mixed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Adaptive fit happened to select xmin=1, with a high p-value.
            for concept in ("n1", "variants"):
                _write_rows(
                    root,
                    "xmin-adaptive",
                    concept,
                    [_row("ADAPT", concept=concept, xmin=1, gof_p=0.90)],
                )
                _write_rows(
                    root,
                    "xmin01",
                    concept,
                    [
                        _row("ADAPT", concept=concept, xmin=1, gof_p=0.01),
                        _row("FOREIGN", concept=concept, xmin=5, gof_p=0.90),
                    ],
                )
            summary, audit = self._build(
                root,
                study_logs=["ADAPT", "FOREIGN"],
                xmin_settings=(1, None),
                ks=(1,),
                min_fitted_types_grid=(15,),
                gof_thresholds=(0.05,),
            )
            adaptive = audit[
                (audit["xmin_setting"] == "None") & (audit["log_name"] == "ADAPT")
            ].iloc[0]
            fixed = audit[
                (audit["xmin_setting"] == "1") & (audit["log_name"] == "ADAPT")
            ].iloc[0]
            foreign = audit[
                (audit["xmin_setting"] == "1") & (audit["log_name"] == "FOREIGN")
            ].iloc[0]
            self.assertEqual(adaptive["source_experiment"], "xmin-adaptive")
            self.assertEqual(fixed["source_experiment"], "xmin01")
            self.assertAlmostEqual(float(adaptive["gof_p_k"]), 0.90)
            self.assertAlmostEqual(float(fixed["gof_p_k"]), 0.01)
            self.assertEqual(int(adaptive["outcome_k"]), 1)
            self.assertEqual(int(fixed["outcome_k"]), 0)
            self.assertIn("incompatible_xmin_configuration", foreign["exclusion_reason_k"])
            self.assertFalse(bool(foreign["included_in_paired"]))
            missing_foreign = audit[
                (audit["xmin_setting"] == "None") & (audit["log_name"] == "FOREIGN")
            ].iloc[0]
            self.assertIn("missing_result", missing_foreign["exclusion_reason_k"])

    def test_missing_experiment_is_reported_not_replaced(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for concept in ("n1", "variants"):
                _write_rows(
                    root,
                    "xmin-adaptive",
                    concept,
                    [_row("ONLY", concept=concept, xmin=4, gof_p=0.4)],
                )
            summary, _audit = self._build(
                root,
                study_logs=["ONLY"],
                xmin_settings=(1,),
                ks=(1,),
                min_fitted_types_grid=(15,),
                gof_thresholds=(0.05,),
            )
            row = summary.iloc[0]
            self.assertEqual(row["status"], "unavailable")
            self.assertEqual(int(row["N"]), 0)
            self.assertIn("Missing xmin directory xmin01", row["notes"])
            self.assertTrue(pd.isna(row["mcnemar_p"]))
            self.assertTrue(pd.isna(row["paired_difference"]))

    def test_holm_family_stays_five_for_a_missing_k(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = [f"L{i}" for i in range(6)]
            for concept in ("n1", "variants"):
                rows = []
                for log in logs:
                    # Rejected at k=1, not rejected at variants.
                    gof_p = 0.01 if concept == "n1" else 0.80
                    rows.append(_row(log, concept=concept, xmin=2, gof_p=gof_p))
                _write_rows(root, "xmin02", concept, rows)
            summary, _audit = self._build(
                root,
                study_logs=logs,
                xmin_settings=(2,),
                ks=(1, 2, 3, 4, 20),
                min_fitted_types_grid=(15,),
                gof_thresholds=(0.05,),
            )
            available = summary[summary["k"] == 1].iloc[0]
            missing = summary[summary["k"] == 20].iloc[0]
            self.assertEqual(available["mcnemar_p"], 0.03125)
            self.assertAlmostEqual(float(available["mcnemar_p_holm"]), 0.15625)
            self.assertEqual(missing["status"], "unavailable")
            self.assertTrue(pd.isna(missing["mcnemar_p"]))
            self.assertTrue(pd.isna(missing["mcnemar_p_holm"]))

    def test_endpoint_specific_exclusion(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_rows(
                root,
                "xmin-adaptive",
                "n1",
                [_row("ONE", concept="n1", xmin=8, n_fitted_types=10, gof_p=0.5)],
            )
            _write_rows(
                root,
                "xmin-adaptive",
                "variants",
                [_row("ONE", concept="variants", xmin=8, n_fitted_types=40, gof_p=0.5)],
            )
            summary, audit = self._build(
                root,
                study_logs=["ONE"],
                xmin_settings=(None,),
                ks=(1,),
                min_fitted_types_grid=(15,),
                gof_thresholds=(0.05,),
            )
            row = audit.iloc[0]
            self.assertIn("below_min_fitted_types", row["exclusion_reason_k"])
            self.assertEqual(row["exclusion_reason_variants"], "")
            self.assertTrue(bool(row["eligible_variants"]))
            self.assertFalse(bool(row["included_in_paired"]))
            self.assertEqual(int(summary.iloc[0]["N"]), 0)
            self.assertEqual(int(summary.iloc[0]["n_eligible_variants"]), 1)
            self.assertEqual(int(summary.iloc[0]["n_eligible_k"]), 0)


class PaperTableTests(unittest.TestCase):
    def test_holm_families_are_recomputed_from_raw_pvalues(self) -> None:
        raw = [0.01, 0.02, 0.03] + [0.4] * 17
        rows = []
        for k, pvalue in enumerate(raw, start=1):
            rows.append(
                {
                    "k": k,
                    "N": 10,
                    "paired_difference": 0.1,
                    "ci95_lower": -0.05,
                    "ci95_upper": 0.25,
                    "mcnemar_p": pvalue,
                }
            )
        rows[3]["N"] = 0
        rows[3]["paired_difference"] = None
        rows[3]["ci95_lower"] = None
        rows[3]["ci95_upper"] = None
        rows[3]["mcnemar_p"] = None
        summary = pd.DataFrame(rows)
        table = PAIRED.paper_table_from_summary(summary)
        PAIRED.validate_paper_table(table, summary)
        raw_for_holm = [None if k == 4 else raw[k - 1] for k in range(1, 21)]
        holm_20 = PAIRED.holm_adjust(raw_for_holm)
        holm_3 = PAIRED.holm_adjust(raw_for_holm[:3])
        self.assertNotAlmostEqual(holm_20[0], holm_3[0])
        by_stat = table.set_index("statistic")
        self.assertEqual(list(table.columns), ["statistic", *[f"k={k}" for k in range(1, 21)]])
        self.assertAlmostEqual(float(by_stat.loc["mcnemar_p_holm_k_le_3", "k=1"]), holm_3[0])
        self.assertAlmostEqual(float(by_stat.loc["mcnemar_p_holm_k_le_20", "k=1"]), holm_20[0])
        self.assertNotAlmostEqual(
            float(by_stat.loc["mcnemar_p_holm_k_le_3", "k=1"]),
            float(by_stat.loc["mcnemar_p_holm_k_le_20", "k=1"]),
        )
        for k in range(4, 21):
            self.assertTrue(pd.isna(by_stat.loc["mcnemar_p_holm_k_le_3", f"k={k}"]))
        for k in range(11, 21):
            self.assertTrue(pd.isna(by_stat.loc["mcnemar_p_holm_k_le_10", f"k={k}"]))
        self.assertEqual(int(by_stat.loc["paired_N", "k=4"]), 0)
        self.assertTrue(pd.isna(by_stat.loc["difference_pp", "k=4"]))
        self.assertTrue(pd.isna(by_stat.loc["mcnemar_p_raw", "k=4"]))
        self.assertAlmostEqual(float(by_stat.loc["difference_pp", "k=1"]), 10.0)
        self.assertAlmostEqual(float(by_stat.loc["ci95_lower_pp", "k=1"]), -5.0)
        self.assertAlmostEqual(float(by_stat.loc["ci95_upper_pp", "k=1"]), 25.0)

    def test_csv_writes_literal_nan_and_integer_counts(self) -> None:
        rows = []
        for k in range(1, 21):
            rows.append(
                {
                    "k": k,
                    "N": 0 if k == 11 else 6,
                    "paired_difference": None if k == 11 else 1.0,
                    "ci95_lower": None if k == 11 else 0.2,
                    "ci95_upper": None if k == 11 else 1.0,
                    "mcnemar_p": None if k == 11 else 0.03125,
                }
            )
        table = PAIRED.paper_table_from_summary(pd.DataFrame(rows))
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired_gof_paper_table.csv"
            PAIRED._write_paper_csv(table, path)
            text = path.read_text(encoding="utf-8")
        header = text.splitlines()[0]
        self.assertEqual(header, "statistic," + ",".join(f"k={k}" for k in range(1, 21)))
        self.assertIn("NaN", text)
        paired = next(line for line in text.splitlines() if line.startswith("paired_N,"))
        self.assertIn(",0,", paired)
        self.assertNotIn("0.0", paired)
        self.assertNotRegex(text, r"\*")
        self.assertNotIn("%", text)


class RealStudyResultsTests(unittest.TestCase):
    def test_saved_real_experiments_satisfy_the_checks(self) -> None:
        root = REPO_ROOT / PL_STATS_DIR / "real"
        if not (root / "xmin-adaptive").is_dir():
            self.skipTest("saved real experiments are not present")
        summary, audit = PAIRED.build_paired_gof_tables(root)
        PAIRED.validate_paired_tables(summary, audit)
        self.assertEqual(len(summary), 3 * 3 * 2 * 5)
        self.assertEqual(len(audit), len(summary) * len(ALL_REAL_LOG_DATASETS))
        self.assertNotIn("TEST_BPIC12", set(audit["log_name"]))
        self.assertFalse(audit["log_name"].astype(str).str.endswith("_sim").any())
        adaptive = audit[audit["xmin_setting"] == "None"]
        self.assertTrue((adaptive["source_experiment"] == "xmin-adaptive").all())
        fixed = audit[audit["xmin_setting"] == "1"]
        self.assertTrue((fixed["source_experiment"] == "xmin01").all())


if __name__ == "__main__":
    unittest.main()
