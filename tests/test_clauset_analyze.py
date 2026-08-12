"""Unit tests for Clauset analysis helpers, fit dispatch, and GOF plumbing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from utils.powerlaw.analyze import (
    DOUBLY_BOUNDED_POWER_LAW,
    FULL_RANGE_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    _classify,
    _descriptive_stats,
    _failed_gof,
    _fitted_region_mask,
    _log_range,
    _powerlaw_fit_with_stats,
    _powerlaw_fit_with_stats_for_model,
    _select_xmax_by_min_ks,
    _validate_observations,
    _xmax_candidates,
    bootstrap_gof,
    decide_and_record,
)
from utils.powerlaw.sampling import _sample_fitted_power_law


class ValidateObservationsTests(unittest.TestCase):
    def test_accepts_positive_integer_valued(self) -> None:
        out = _validate_observations([1, 2.0, 3])
        np.testing.assert_array_equal(out, np.asarray([1, 2, 3]))

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            _validate_observations([])

    def test_rejects_non_positive(self) -> None:
        with self.assertRaises(ValueError):
            _validate_observations([1, 0, 2])
        with self.assertRaises(ValueError):
            _validate_observations([1, -1, 2])

    def test_rejects_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            _validate_observations([1.0, np.nan, 2.0])

    def test_rejects_non_integer(self) -> None:
        with self.assertRaises(ValueError):
            _validate_observations([1.0, 1.5, 2.0])

    def test_rejects_non_1d(self) -> None:
        with self.assertRaises(ValueError):
            _validate_observations([[1, 2], [3, 4]])

    def test_continuous_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            _validate_observations([1, 2, 3], discrete=False)


class DescriptiveAndMaskTests(unittest.TestCase):
    def test_descriptive_stats(self) -> None:
        data = np.asarray([1, 1, 2, 5], dtype=int)
        stats = _descriptive_stats(data)
        self.assertEqual(stats["n_occurrences"], 9)
        self.assertEqual(stats["n_types"], 4)
        self.assertEqual(stats["n_singletons"], 2)
        self.assertAlmostEqual(stats["singleton_share"], 0.5)

    def test_fitted_region_mask(self) -> None:
        data = np.asarray([1, 2, 3, 4, 10], dtype=int)
        mask = _fitted_region_mask(data, xmin=2, xmax=4)
        np.testing.assert_array_equal(mask, np.asarray([False, True, True, True, False]))
        mask_open = _fitted_region_mask(data, xmin=3, xmax=None)
        np.testing.assert_array_equal(
            mask_open, np.asarray([False, False, True, True, True])
        )

    def test_log_range(self) -> None:
        self.assertAlmostEqual(_log_range(10.0, 1000.0), 2.0)
        self.assertTrue(np.isnan(_log_range(1.0, None)))
        self.assertTrue(np.isnan(_log_range(1.0, float("nan"))))
        self.assertTrue(np.isnan(_log_range(-1.0, 10.0)))


class XmaxSelectionTests(unittest.TestCase):
    def test_xmax_candidates_rank_and_dedupe(self) -> None:
        # Descending: 50, 20, 20, 10, 5, 1 → k=0→50, k=1→20, k=2→20 skipped, k=3→10
        data = np.asarray([1, 5, 10, 20, 20, 50], dtype=int)
        cands = _xmax_candidates(data, (0, 1, 2, 3, 99))
        self.assertEqual(
            cands,
            [
                (0, 50.0, 0),
                (1, 20.0, 1),
                (3, 10.0, 3),
            ],
        )

    def test_select_xmax_by_min_ks_and_ties(self) -> None:
        candidates = [
            {
                "KS_D": 0.2,
                "doubly_bounded_exclude_head_variants": 0,
                "n_fitted_types": 10,
                "xmax": 100,
            },
            {
                "KS_D": 0.1,
                "doubly_bounded_exclude_head_variants": 5,
                "n_fitted_types": 8,
                "xmax": 50,
            },
            {
                "KS_D": 0.1,
                "doubly_bounded_exclude_head_variants": 2,
                "n_fitted_types": 7,
                "xmax": 40,
            },
        ]
        chosen = _select_xmax_by_min_ks(candidates)
        assert chosen is not None
        self.assertEqual(chosen["xmax"], 40)
        self.assertEqual(chosen["doubly_bounded_exclude_head_variants"], 2)

    def test_select_xmax_tie_prefers_more_fitted_types(self) -> None:
        candidates = [
            {
                "KS_D": 0.1,
                "doubly_bounded_exclude_head_variants": 1,
                "n_fitted_types": 5,
                "xmax": 30,
            },
            {
                "KS_D": 0.1,
                "doubly_bounded_exclude_head_variants": 1,
                "n_fitted_types": 12,
                "xmax": 25,
            },
        ]
        chosen = _select_xmax_by_min_ks(candidates)
        assert chosen is not None
        self.assertEqual(chosen["xmax"], 25)

    def test_select_xmax_all_nan_returns_none(self) -> None:
        self.assertIsNone(
            _select_xmax_by_min_ks([{"KS_D": float("nan")}, {"KS_D": "x"}])
        )


def _comparison_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class ClassifyTests(unittest.TestCase):
    def test_invalid_fit(self) -> None:
        text = _classify(
            n_fitted_types=50,
            gof_p=0.5,
            comparison_df=_comparison_df(
                [{"R": 1.0, "p": 0.01, "model_2": "lognormal"}]
            ),
            model_label="lower bounded power law",
            minimum_fitted_types=30,
            significance_level=0.1,
            fit_valid=False,
            pathology_flags=["alpha_at_parameter_boundary"],
        )
        self.assertIn("invalid", text)
        self.assertIn("alpha_at_parameter_boundary", text)

    def test_plausible_and_preferred(self) -> None:
        text = _classify(
            n_fitted_types=40,
            gof_p=0.4,
            comparison_df=_comparison_df(
                [{"R": 2.0, "p": 0.01, "model_2": "lognormal"}]
            ),
            model_label="full range power law",
            minimum_fitted_types=30,
            significance_level=0.1,
            fit_valid=True,
            pathology_flags=[],
        )
        self.assertIn("(1) >= 30 fitted types", text)
        self.assertIn("plausible", text)
        self.assertIn("preferred over alternatives", text)

    def test_not_plausible_alternatives_preferred(self) -> None:
        text = _classify(
            n_fitted_types=10,
            gof_p=0.01,
            comparison_df=_comparison_df(
                [{"R": -1.5, "p": 0.02, "model_2": "exponential"}]
            ),
            model_label="lower bounded power law",
            minimum_fitted_types=30,
            significance_level=0.1,
            fit_valid=True,
            pathology_flags=[],
        )
        self.assertIn("(1) < 30 fitted types", text)
        self.assertIn("not plausible", text)
        self.assertIn("alternatives preferred", text)

    def test_gof_unavailable_inconclusive(self) -> None:
        text = _classify(
            n_fitted_types=35,
            gof_p=float("nan"),
            comparison_df=_comparison_df(
                [{"R": 0.1, "p": 0.5, "model_2": "lognormal"}]
            ),
            model_label="doubly bounded power law",
            minimum_fitted_types=30,
            significance_level=0.1,
            fit_valid=True,
            pathology_flags=[],
        )
        self.assertIn("GOF unavailable", text)
        self.assertIn("inconclusive", text)


class FitDispatchSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(42)
        cls.lower_data = _sample_fitted_power_law(
            400, alpha=2.2, xmin=1, xmax=None, rng=rng
        )
        # Heavy head so doubly-bounded xmax selection has room to exclude.
        head = np.asarray([500, 300, 200, 150, 120], dtype=int)
        body = _sample_fitted_power_law(
            200, alpha=2.0, xmin=1, xmax=80, rng=np.random.default_rng(7)
        )
        cls.doubly_data = np.concatenate([body, head])

    def test_full_range(self) -> None:
        fit = _powerlaw_fit_with_stats_for_model(
            self.lower_data,
            FULL_RANGE_POWER_LAW,
            doubly_bounded_exclude_head_variants=(0, 1, 2),
        )
        assert fit is not None
        self.assertEqual(fit["xmin"], 1)
        self.assertTrue(np.isnan(fit["xmax"]))
        self.assertTrue(np.isfinite(fit["alpha"]))
        self.assertTrue(1.2 <= fit["alpha"] <= 3.5)
        self.assertTrue(fit["fit_valid"])

    def test_lower_bounded(self) -> None:
        fit = _powerlaw_fit_with_stats_for_model(
            self.lower_data,
            LOWER_BOUNDED_POWER_LAW,
            doubly_bounded_exclude_head_variants=(0, 1, 2),
        )
        assert fit is not None
        self.assertTrue(np.isfinite(fit["xmin"]))
        self.assertGreaterEqual(fit["xmin"], 1)
        self.assertTrue(np.isnan(fit["xmax"]))
        self.assertTrue(fit["fit_valid"])

    def test_doubly_bounded(self) -> None:
        fit = _powerlaw_fit_with_stats_for_model(
            self.doubly_data,
            DOUBLY_BOUNDED_POWER_LAW,
            doubly_bounded_exclude_head_variants=(0, 1, 2, 3),
        )
        assert fit is not None
        self.assertTrue(np.isfinite(fit["xmax"]))
        self.assertTrue(np.isfinite(fit["KS_D"]))
        self.assertIn("doubly_bounded_exclude_head_variants", fit)
        self.assertTrue(fit["fit_valid"])


class BootstrapGofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(123)
        cls.data = _sample_fitted_power_law(
            250, alpha=2.3, xmin=1, xmax=None, rng=rng
        )
        cls.fit = _powerlaw_fit_with_stats(cls.data, xmin=None, xmax=None)

    def test_failed_gof_nonfinite_params(self) -> None:
        bad = {
            **self.fit,
            "xmin": float("nan"),
            "alpha": float("nan"),
            "KS_D": float("nan"),
        }
        gof = bootstrap_gof(
            self.data, bad, LOWER_BOUNDED_POWER_LAW, n_bootstraps=8, random_seed=1
        )
        self.assertTrue(np.isnan(gof["gof_p"]))
        self.assertEqual(gof["n_success"], 0)
        self.assertEqual(gof["n_failed"], 8)

    def test_failed_gof_too_few_on_support(self) -> None:
        data = np.asarray([1, 2, 3, 4, 5], dtype=int)
        empirical = {
            "xmin": 5.0,
            "xmax": float("nan"),
            "KS_D": 0.2,
            "alpha": 2.0,
            "fit": SimpleNamespace(power_law=SimpleNamespace(alpha=2.0)),
        }
        gof = bootstrap_gof(
            data, empirical, LOWER_BOUNDED_POWER_LAW, n_bootstraps=5, random_seed=2
        )
        self.assertTrue(np.isnan(gof["gof_p"]))
        self.assertEqual(gof["n_failed"], 5)

    def test_happy_path_structure_and_reproducibility(self) -> None:
        gof_a = bootstrap_gof(
            self.data,
            self.fit,
            LOWER_BOUNDED_POWER_LAW,
            n_bootstraps=10,
            random_seed=42,
        )
        gof_b = bootstrap_gof(
            self.data,
            self.fit,
            LOWER_BOUNDED_POWER_LAW,
            n_bootstraps=10,
            random_seed=42,
        )
        self.assertGreaterEqual(gof_a["gof_p"], 0.0)
        self.assertLessEqual(gof_a["gof_p"], 1.0)
        self.assertEqual(gof_a["n_success"] + gof_a["n_failed"], 10)
        self.assertEqual(len(gof_a["simulated_distances"]), gof_a["n_success"])
        self.assertEqual(gof_a["gof_p"], gof_b["gof_p"])
        np.testing.assert_array_equal(
            gof_a["simulated_distances"], gof_b["simulated_distances"]
        )

    def test_doubly_bounded_smoke(self) -> None:
        head = np.asarray([400, 250, 180], dtype=int)
        body = _sample_fitted_power_law(
            120, alpha=2.0, xmin=1, xmax=60, rng=np.random.default_rng(9)
        )
        data = np.concatenate([body, head])
        fit = _powerlaw_fit_with_stats_for_model(
            data,
            DOUBLY_BOUNDED_POWER_LAW,
            doubly_bounded_exclude_head_variants=(0, 1, 2),
        )
        assert fit is not None
        gof = bootstrap_gof(
            data,
            fit,
            DOUBLY_BOUNDED_POWER_LAW,
            n_bootstraps=3,
            random_seed=7,
            doubly_bounded_exclude_head_variants=(0, 1, 2),
        )
        self.assertEqual(gof["n_success"] + gof["n_failed"], 3)
        if gof["n_success"] > 0:
            self.assertGreaterEqual(gof["gof_p"], 0.0)
            self.assertLessEqual(gof["gof_p"], 1.0)


class DecideAndRecordTests(unittest.TestCase):
    def _base_descriptive(self) -> dict:
        return {
            "n_occurrences": 20,
            "n_types": 5,
            "n_singletons": 2,
            "singleton_share": 0.4,
        }

    def test_invalid_fit_blanks_inference_fields(self) -> None:
        fit = {
            "fit": MagicMock(),
            "alpha": 4.0,
            "xmin": 1,
            "xmax": float("nan"),
            "KS_D": 0.1,
            "n_fitted_types": 5,
            "fitted_type_share": 1.0,
            "fitted_occurrence_share": 1.0,
            "log_range": float("nan"),
            "alpha_at_boundary": True,
            "pathology_flags": ["alpha_at_parameter_boundary"],
            "fit_valid": False,
        }
        gof = _failed_gof(n_bootstraps=10, xmin=1.0, xmax=float("nan"), alpha=4.0)
        comparison = _comparison_df(
            [
                {
                    "log_name": "toy",
                    "model_1": LOWER_BOUNDED_POWER_LAW,
                    "model_2": "lognormal",
                    "R": 1.0,
                    "p": 0.2,
                }
            ]
        )
        out = decide_and_record(
            log_name="toy",
            input_path="/tmp/toy.csv",
            model=LOWER_BOUNDED_POWER_LAW,
            descriptive=self._base_descriptive(),
            fit=fit,
            gof=gof,
            comparison_df=comparison,
            minimum_fitted_types=30,
            significance_level=0.1,
        )
        self.assertTrue(pd.isna(out["summary_row"]["alpha"]))
        self.assertTrue(pd.isna(out["summary_row"]["gof_p"]))
        self.assertEqual(out["evaluation"]["gof_kind"], "skipped_invalid_fit")
        self.assertEqual(int(out["comparison_df"]["n_fitted_types"].iloc[0]), 5)
        self.assertIn("invalid", out["summary_row"]["classification"])

    def test_valid_path_schema(self) -> None:
        data = _sample_fitted_power_law(
            100, alpha=2.2, xmin=1, xmax=None, rng=np.random.default_rng(3)
        )
        fit = _powerlaw_fit_with_stats(data, xmin=1, xmax=None)
        gof = {
            "gof_p": 0.55,
            "n_success": 10,
            "n_failed": 0,
            "empirical_d": fit["KS_D"],
            "simulated_distances": np.asarray([0.1, 0.2], dtype=float),
        }
        comparison = _comparison_df(
            [
                {
                    "log_name": "toy",
                    "model_1": FULL_RANGE_POWER_LAW,
                    "model_2": "lognormal",
                    "R": 1.2,
                    "p": 0.3,
                },
                {
                    "log_name": "toy",
                    "model_1": FULL_RANGE_POWER_LAW,
                    "model_2": "exponential",
                    "R": -0.5,
                    "p": 0.4,
                },
            ]
        )
        out = decide_and_record(
            log_name="toy",
            input_path="/tmp/toy.csv",
            model=FULL_RANGE_POWER_LAW,
            descriptive=_descriptive_stats(data),
            fit=fit,
            gof=gof,
            comparison_df=comparison,
            minimum_fitted_types=30,
            significance_level=0.1,
        )
        self.assertIn("gof_rows", out)
        self.assertIn("comparison_df", out)
        self.assertIn("summary_row", out)
        self.assertIn("evaluation", out)
        self.assertEqual(len(out["gof_rows"]), 1)
        self.assertEqual(out["gof_rows"][0]["log_name"], "toy")
        self.assertTrue(out["summary_row"]["fit_valid"])
        self.assertEqual(out["summary_row"]["best_other_distribution"], "exponential")
        self.assertIn("n_fitted_types", out["comparison_df"].columns)


if __name__ == "__main__":
    unittest.main()
