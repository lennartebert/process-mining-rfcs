"""Saturation diagnostics: window weights, identities, and readable tables."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.io.attachments import _ngram_nodes


def _load_saturation():
    path = REPO_ROOT / "analyses" / "powerlaw_statistics" / "saturation.py"
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
            for k in (5, 10, 15, 20):
                self.assertEqual(
                    SAT.occurrence_weight(length, k),
                    len(_ngram_nodes(sequence, k, True)),
                )

    def test_weight_rejects_unwrapped_order(self) -> None:
        with self.assertRaises(ValueError):
            SAT.occurrence_weight(4, 1)


class DiagnosticTests(unittest.TestCase):
    def test_identities_monotonicity_and_saturation(self) -> None:
        sequences = [("a",), ("a",), ("b", "c", "d", "e")]
        profile = _profile("LOG", sequences)
        frame = SAT.build_saturation_frame([profile])
        SAT.validate_saturation_frame(frame)
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
        text = SAT.format_saturation_results(frame)
        self.assertIn("GONE: missing_log. Event log file is missing.", text)
        self.assertIn("EMPTY: empty_log. No traces with a completion timestamp.", text)
        # The one usable log has case-completion share 0.5 at k=5 (length 2 is complete, length 1 is complete).
        # Both lengths are <= 3, so the median is 100, and the missing logs do not pull it down.
        self.assertIn(SAT.MEDIAN_ROW_LABEL, text)
        k5 = text.split("k = 10.", 1)[0]
        median_line = next(line for line in k5.splitlines() if line.startswith(SAT.MEDIAN_ROW_LABEL))
        self.assertTrue(median_line.rstrip().endswith("0.00"))
        self.assertIn("100.0", median_line)

    def test_median_row_is_unweighted_across_logs(self) -> None:
        short = _profile("MANY", [("a",)] * 100)
        long = _profile("ONE", [("z",) * 100])
        frame = SAT.build_saturation_frame([short, long])
        SAT.validate_saturation_frame(frame)
        text = SAT.format_saturation_results(frame)
        k5 = text.split("k = 10.", 1)[0]
        median_line = next(line for line in k5.splitlines() if line.startswith(SAT.MEDIAN_ROW_LABEL))
        # Shares are 1 and 0. Their median is 50. A pooled share would be about 99.
        self.assertIn("50.0", median_line)
        self.assertNotIn("99.0", median_line)
        # Count and length columns stay blank on the median row.
        self.assertNotIn("100", median_line)
        self.assertTrue(median_line.startswith(SAT.MEDIAN_ROW_LABEL))


class ReadableTableTests(unittest.TestCase):
    def test_formatting_alignment_and_shared_log_order(self) -> None:
        sequences = [("a",)] * 1000
        profile = _profile("ACCRE", sequences)
        other = _profile("BPIC12", [("a", "b")])
        frame = SAT.build_saturation_frame([profile, other])
        text = SAT.format_saturation_results(frame)
        for k, bound in ((5, 3), (10, 8), (15, 13), (20, 18)):
            self.assertIn(f"k = {k}. A case or variant is complete when L <= {bound}", text)
        self.assertIn("(O_k - N) / N", text)
        self.assertIn("does not imply that most extracted occurrences come from complete cases", text)
        header = next(line for line in text.splitlines() if line.startswith("Log"))
        data = next(line for line in text.splitlines() if line.startswith("ACCRE"))
        self.assertIn("1,000", data)
        self.assertEqual(
            header.index("Cases") + len("Cases"),
            data.index("1,000") + len("1,000"),
        )
        self.assertTrue(data.startswith("ACCRE"))
        order = []
        for line in text.splitlines():
            if line.startswith("ACCRE") or line.startswith("BPIC12"):
                order.append(line.split("  ", 1)[0].strip())
        self.assertEqual(order, ["ACCRE", "BPIC12"] * 4)
        # Even case count keeps the half-step median (lengths 1 and 2).
        half = _profile("HALF", [("a",), ("b", "b")])
        half_line = next(
            line
            for line in SAT.format_saturation_results(
                SAT.build_saturation_frame([half])
            ).splitlines()
            if line.startswith("HALF")
        )
        self.assertIn("1.5", half_line)


class RealLogOccurrenceTests(unittest.TestCase):
    def test_accre_matches_saved_n5_total(self) -> None:
        summary = (
            REPO_ROOT
            / "results"
            / "powerlaw_statistics"
            / "real"
            / "n5"
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
