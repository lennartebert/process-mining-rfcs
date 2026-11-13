"""Saturation diagnostics: window weights, identities, and CSV tables."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.io.attachments import _ngram_nodes


def _load_saturation():
    path = (
        REPO_ROOT
        / "analyses"
        / "powerlaw_statistics"
        / "analysis_helpers"
        / "saturation.py"
    )
    spec = importlib.util.spec_from_file_location("saturation_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SAT = _load_saturation()


def _profile(name: str, sequences: list[tuple[str, ...]]):
    return SAT.profile_from_sequences(name, sequences)


class WeightTests(unittest.TestCase):
    def test_weight_matches_extractor_windows(self) -> None:
        for length in range(1, 31):
            sequence = tuple(f"a{i}" for i in range(length))
            for k in (1, 5, 10, 15, 20):
                self.assertEqual(
                    SAT.occurrence_weight(length, k),
                    len(_ngram_nodes(sequence, k, True)),
                )

    def test_weight_rejects_unknown_order(self) -> None:
        with self.assertRaises(ValueError):
            SAT.occurrence_weight(4, 0)

    def test_k1_completion_threshold_is_one_activity(self) -> None:
        self.assertEqual(SAT.max_complete_trace_length(1), 1)
        self.assertEqual(SAT.occurrence_weight(1, 1), 1)
        self.assertEqual(SAT.max_complete_trace_length(5), 3)


class DiagnosticTests(unittest.TestCase):
    def test_identities_monotonicity_and_saturation(self) -> None:
        sequences = [("a",), ("a",), ("b", "c", "d", "e")]
        profile = _profile("LOG", sequences)
        frame = SAT.build_saturation_frame([profile])
        SAT.validate_saturation_frame(frame)
        at1 = frame[(frame.log_name == "LOG") & (frame.k == 1)].iloc[0]
        self.assertEqual(int(at1.max_complete_trace_length), 1)
        self.assertEqual(int(at1.n_complete_cases), 2)
        self.assertEqual(int(at1.n_complete_variants), 1)
        self.assertEqual(int(at1.total_occurrences), 6)
        self.assertEqual(int(at1.excess_occurrences), 3)
        self.assertAlmostEqual(float(at1.complete_occurrence_share), 2 / 6)
        self.assertAlmostEqual(float(at1.excess_occurrences_per_case), 1.0)
        at5 = frame[(frame.log_name == "LOG") & (frame.k == 5)].iloc[0]
        self.assertEqual(int(at5.n_cases), 3)
        self.assertEqual(int(at5.n_variants), 2)
        self.assertEqual(float(at5.median_trace_length), 1.0)
        self.assertEqual(int(at5.max_trace_length), 4)
        self.assertEqual(int(at5.n_complete_cases), 2)
        self.assertEqual(int(at5.n_complete_variants), 1)
        self.assertEqual(int(at5.total_occurrences), 4)
        self.assertEqual(int(at5.excess_occurrences), 1)
        self.assertAlmostEqual(float(at5.complete_occurrence_share), 0.5)
        self.assertAlmostEqual(float(at5.excess_occurrences_per_case), 1 / 3)
        for k in (10, 15, 20):
            row = frame[(frame.log_name == "LOG") & (frame.k == k)].iloc[0]
            self.assertEqual(int(row.total_occurrences), 3)
            self.assertEqual(int(row.excess_occurrences), 0)
            self.assertAlmostEqual(float(row.complete_case_share), 1.0)
            self.assertAlmostEqual(float(row.complete_variant_share), 1.0)
            self.assertAlmostEqual(float(row.complete_occurrence_share), 1.0)

    def test_extracted_mismatch_is_rejected(self) -> None:
        profile = _profile("LOG", [("a", "b", "c", "d", "e", "f")])
        frame = SAT.build_saturation_frame(
            [profile], {("LOG", k): (1, "summary n") for k in SAT.SATURATION_KS}
        )
        with self.assertRaises(AssertionError):
            SAT.validate_saturation_frame(frame)

    def test_missing_and_empty_stay_unknown(self) -> None:
        missing = SAT.LengthProfile("GONE", SAT.STATUS_MISSING, "Event log file is missing.")
        empty = _profile("EMPTY", [])
        present = _profile("ONE", [("a",), ("b", "b")])
        frame = SAT.build_saturation_frame([missing, empty, present])
        SAT.validate_saturation_frame(frame)
        for name in ("GONE", "EMPTY"):
            rows = frame[frame.log_name == name]
            self.assertTrue(rows["n_cases"].isna().all())
            self.assertTrue(rows["total_occurrences"].isna().all())
            self.assertFalse((rows["n_cases"].fillna(0) == 0).all() and rows["n_cases"].notna().any())
        table = SAT.build_k_table(frame, 5)
        gone = table[table["Log"] == "GONE"].iloc[0]
        empty = table[table["Log"] == "EMPTY"].iloc[0]
        self.assertTrue(pd.isna(gone["Cases"]))
        self.assertTrue(pd.isna(empty["Cases"]))
        self.assertTrue(pd.isna(gone["Cases complete (%)"]))
        # Both lengths are <= 3, so the one usable log is fully complete.
        # Missing logs do not pull the median down.
        median = table[table["Log"] == SAT.MEDIAN_ROW_LABEL].iloc[0]
        self.assertTrue(pd.isna(median["Cases"]))
        self.assertTrue(pd.isna(median["Median length"]))
        self.assertAlmostEqual(float(median["Cases complete (%)"]), 100.0)
        self.assertAlmostEqual(float(median["Excess occurrences per case"]), 0.0)

    def test_median_row_is_unweighted_across_logs(self) -> None:
        short = _profile("MANY", [("a",)] * 100)
        long = _profile("ONE", [("z",) * 100])
        frame = SAT.build_saturation_frame([short, long])
        SAT.validate_saturation_frame(frame)
        table = SAT.build_k_table(frame, 5)
        median = table[table["Log"] == SAT.MEDIAN_ROW_LABEL].iloc[0]
        # Shares are 1 and 0. Their median is 50. A pooled share would be about 99.
        self.assertAlmostEqual(float(median["Cases complete (%)"]), 50.0)
        self.assertNotAlmostEqual(float(median["Cases complete (%)"]), 99.0)
        self.assertTrue(pd.isna(median["Cases"]))
        self.assertTrue(pd.isna(median["Max length"]))


class TableTests(unittest.TestCase):
    def test_k_tables_keep_log_order_and_half_step_median(self) -> None:
        profile = _profile("ACCRE", [("a",)] * 1000)
        other = _profile("BPIC12", [("a", "b")])
        frame = SAT.build_saturation_frame([profile, other])
        for k in SAT.SATURATION_KS:
            table = SAT.build_k_table(frame, k)
            self.assertEqual(list(table.columns), SAT.K_TABLE_COLUMNS)
            self.assertEqual(list(table["Log"]), ["ACCRE", "BPIC12", SAT.MEDIAN_ROW_LABEL])
            self.assertEqual(int(table.iloc[0]["Cases"]), 1000)
        half = SAT.build_k_table(
            SAT.build_saturation_frame([_profile("HALF", [("a",), ("b", "b")])]),
            5,
        )
        self.assertAlmostEqual(float(half.iloc[0]["Median length"]), 1.5)

    def test_cross_log_threshold_rows(self) -> None:
        short = _profile("SHORT", [("a",)] * 5)
        mostly = _profile("MOSTLY", [("a",)] * 4 + [("z",) * 100])
        long = _profile("LONG", [("z",) * 30])
        missing = SAT.LengthProfile("GONE", SAT.STATUS_MISSING, "Event log file is missing.")
        frame = SAT.build_saturation_frame([short, mostly, long, missing])
        SAT.validate_saturation_frame(frame)
        table = SAT.build_cross_log_table(frame)
        self.assertEqual(
            list(table["Log"]),
            [
                "SHORT",
                "MOSTLY",
                "LONG",
                "GONE",
                "Logs with ≥50% of cases complete",
                "Logs with ≥60% of cases complete",
                "Logs with ≥70% of cases complete",
                "Logs with ≥80% of cases complete",
                "Logs with ≥90% of cases complete",
                "Logs with ≥100% of cases complete",
            ],
        )
        mostly_row = table[table["Log"] == "MOSTLY"].iloc[0]
        self.assertEqual(mostly_row["k=5"], "80.0%")
        self.assertEqual(mostly_row["k=10"], "80.0%")
        self.assertEqual(mostly_row["k=20"], "80.0%")
        # Four complete cases contribute 1 each; the length-100 case contributes 83.
        self.assertEqual(mostly_row[SAT.CROSS_LOG_OCCURRENCE_COLUMN], "4.6%")
        self.assertEqual(mostly_row[SAT.CROSS_LOG_EXCESS_COLUMN], "16.40")
        short_row = table[table["Log"] == "SHORT"].iloc[0]
        self.assertEqual(short_row["k=5"], "100.0%")
        self.assertEqual(short_row[SAT.CROSS_LOG_OCCURRENCE_COLUMN], "100.0%")
        self.assertEqual(short_row[SAT.CROSS_LOG_EXCESS_COLUMN], "0.00")
        long_row = table[table["Log"] == "LONG"].iloc[0]
        self.assertEqual(long_row["k=20"], "0.0%")
        self.assertEqual(long_row[SAT.CROSS_LOG_EXCESS_COLUMN], "12.00")
        gone = table[table["Log"] == "GONE"].iloc[0]
        self.assertEqual(gone["k=5"], "")
        self.assertEqual(gone[SAT.CROSS_LOG_EXCESS_COLUMN], "")
        ge50 = table[table["Log"] == "Logs with ≥50% of cases complete"].iloc[0]
        ge60 = table[table["Log"] == "Logs with ≥60% of cases complete"].iloc[0]
        ge70 = table[table["Log"] == "Logs with ≥70% of cases complete"].iloc[0]
        ge80 = table[table["Log"] == "Logs with ≥80% of cases complete"].iloc[0]
        ge90 = table[table["Log"] == "Logs with ≥90% of cases complete"].iloc[0]
        ge100 = table[table["Log"] == "Logs with ≥100% of cases complete"].iloc[0]
        self.assertEqual(ge50["k=5"], "2/3 (66.7%)")
        self.assertEqual(ge60["k=5"], "2/3 (66.7%)")
        self.assertEqual(ge70["k=5"], "2/3 (66.7%)")
        self.assertEqual(ge50["k=20"], "2/3 (66.7%)")
        self.assertEqual(ge50[SAT.CROSS_LOG_OCCURRENCE_COLUMN], "—")
        self.assertEqual(ge50[SAT.CROSS_LOG_EXCESS_COLUMN], "—")
        self.assertEqual(ge100[SAT.CROSS_LOG_EXCESS_COLUMN], "—")
        self.assertEqual(ge80["k=5"], "2/3 (66.7%)")
        self.assertEqual(ge90["k=5"], "1/3 (33.3%)")
        self.assertEqual(ge100["k=5"], "1/3 (33.3%)")
        self.assertEqual(ge100["k=20"], "1/3 (33.3%)")
        self.assertEqual(ge80["k=20"], "2/3 (66.7%)")
        self.assertEqual(ge80[SAT.CROSS_LOG_OCCURRENCE_COLUMN], "—")
        self.assertEqual(ge80[SAT.CROSS_LOG_EXCESS_COLUMN], "—")
        self.assertEqual(ge90[SAT.CROSS_LOG_EXCESS_COLUMN], "—")

    def test_k1_table_counts_single_activity_cases(self) -> None:
        mostly = _profile("MOSTLY", [("a",)] * 4 + [("z",) * 100])
        table = SAT.build_k_table(SAT.build_saturation_frame([mostly]), 1)
        row = table.iloc[0]
        self.assertEqual(int(row["Cases"]), 5)
        self.assertAlmostEqual(float(row["Cases complete (%)"]), 80.0)
        self.assertAlmostEqual(float(row["Excess occurrences per case"]), 99 / 5)

    def test_writer_emits_csvs_and_drops_text_report(self) -> None:
        frame = SAT.build_saturation_frame(
            [_profile("ACCRE", [("a",)] * 4), _profile("BPIC12", [("a", "b")])]
        )
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            stale = destination / SAT.STALE_TEXT_FILENAME
            stale.write_text("old report\n", encoding="utf-8")
            SAT.write_saturation_tables(frame, destination)
            self.assertFalse(stale.exists())
            self.assertTrue((destination / SAT.SUMMARY_FILENAME).is_file())
            self.assertTrue((destination / SAT.CROSS_LOG_FILENAME).is_file())
            for k in SAT.SATURATION_KS:
                path = destination / SAT.k_table_filename(k)
                self.assertTrue(path.is_file(), path.name)
                written = pd.read_csv(path)
                self.assertEqual(
                    list(written["Log"]),
                    ["ACCRE", "BPIC12", SAT.MEDIAN_ROW_LABEL],
                )
            summary = pd.read_csv(destination / SAT.SUMMARY_FILENAME)
            self.assertEqual(set(summary["k"]), set(SAT.SATURATION_KS))

    def test_saturation_dir_is_under_log_info(self) -> None:
        from analyses.powerlaw_statistics.analysis_helpers.paths import saturation_dir

        self.assertEqual(
            saturation_dir(Path("results/pl_stats/real")),
            Path("results/pl_stats/real/log_info/saturation"),
        )


class RealLogOccurrenceTests(unittest.TestCase):
    def test_accre_matches_saved_n5_total(self) -> None:
        summary = (
            REPO_ROOT
            / "results"
            / "pl_stats"
            / "real"
            / "xmin-adaptive"
            / "raw_results"
            / "n005"
            / "lower_bounded_power_law"
            / "summary.csv"
        )
        if not summary.exists():
            self.skipTest("saved ACCRE n5 summary is not present")
        profiles = SAT.load_study_profiles(["ACCRE"])
        extracted = SAT.load_extracted_occurrences(
            summary.parents[2], ["ACCRE"], [5]
        )
        frame = SAT.build_saturation_frame(profiles, extracted, ks=(5,))
        SAT.validate_saturation_frame(frame, ks=(5,))
        saved = pd.read_csv(summary)
        saved_total = int(saved.loc[saved.log_name == "ACCRE", "n_occurrences"].iloc[0])
        self.assertEqual(int(frame.iloc[0].total_occurrences), saved_total)


if __name__ == "__main__":
    unittest.main()
