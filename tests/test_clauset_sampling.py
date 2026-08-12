"""Unit tests for Clauset discrete sampling helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from scipy.special import zeta

from utils.powerlaw.sampling import (
    _resample_empirical,
    _sample_fitted_power_law,
    _simulate_semiparametric_sample,
)


class SampleFittedPowerLawTruncatedTests(unittest.TestCase):
    def test_draws_stay_on_integer_support(self) -> None:
        rng = np.random.default_rng(0)
        draws = _sample_fitted_power_law(
            500, alpha=2.0, xmin=2, xmax=7, rng=rng
        )
        self.assertEqual(draws.dtype, np.dtype(int))
        self.assertTrue(np.all(draws >= 2))
        self.assertTrue(np.all(draws <= 7))

    def test_frequencies_match_normalized_power_law(self) -> None:
        xmin, xmax, alpha = 1, 5, 2.0
        support = np.arange(xmin, xmax + 1)
        expected = support.astype(float) ** (-alpha)
        expected /= expected.sum()

        rng = np.random.default_rng(1)
        n = 20_000
        draws = _sample_fitted_power_law(
            n, alpha=alpha, xmin=xmin, xmax=xmax, rng=rng
        )
        counts = np.bincount(draws, minlength=xmax + 1)[xmin : xmax + 1]
        empirical = counts / n
        self.assertTrue(np.max(np.abs(empirical - expected)) < 0.02)

    def test_n_zero_returns_empty(self) -> None:
        rng = np.random.default_rng(2)
        out = _sample_fitted_power_law(0, alpha=2.0, xmin=1, xmax=3, rng=rng)
        self.assertEqual(out.size, 0)
        self.assertEqual(out.dtype, np.dtype(int))

    def test_xmax_less_than_xmin_raises(self) -> None:
        rng = np.random.default_rng(3)
        with self.assertRaises(ValueError):
            _sample_fitted_power_law(10, alpha=2.0, xmin=5, xmax=3, rng=rng)

    def test_non_integer_bounds_raise(self) -> None:
        rng = np.random.default_rng(4)
        with self.assertRaises(ValueError):
            _sample_fitted_power_law(10, alpha=2.0, xmin=1.5, xmax=5, rng=rng)
        with self.assertRaises(ValueError):
            _sample_fitted_power_law(10, alpha=2.0, xmin=1, xmax=5.5, rng=rng)

    def test_continuous_not_implemented(self) -> None:
        rng = np.random.default_rng(5)
        with self.assertRaises(NotImplementedError):
            _sample_fitted_power_law(
                10, alpha=2.0, xmin=1, xmax=5, rng=rng, discrete=False
            )


class SampleFittedPowerLawUnboundedTests(unittest.TestCase):
    def test_draws_at_least_xmin(self) -> None:
        rng = np.random.default_rng(10)
        draws = _sample_fitted_power_law(
            300, alpha=2.5, xmin=3, xmax=None, rng=rng
        )
        self.assertTrue(np.all(draws >= 3))

    def test_alpha_le_one_raises(self) -> None:
        rng = np.random.default_rng(11)
        with self.assertRaises(ValueError):
            _sample_fitted_power_law(10, alpha=1.0, xmin=1, xmax=None, rng=rng)

    def test_empirical_survival_matches_hurwitz(self) -> None:
        alpha, xmin = 2.5, 1
        n = 8_000
        rng = np.random.default_rng(12)
        draws = _sample_fitted_power_law(
            n, alpha=alpha, xmin=xmin, xmax=None, rng=rng
        )
        norm = float(zeta(alpha, xmin))
        for x in (1, 2, 5, 10):
            empirical = float(np.mean(draws > x))
            theoretical = float(zeta(alpha, x + 1)) / norm
            self.assertAlmostEqual(empirical, theoretical, delta=0.05)

    def test_seed_reproducibility(self) -> None:
        a = _sample_fitted_power_law(
            50, alpha=2.2, xmin=1, xmax=None, rng=np.random.default_rng(99)
        )
        b = _sample_fitted_power_law(
            50, alpha=2.2, xmin=1, xmax=None, rng=np.random.default_rng(99)
        )
        np.testing.assert_array_equal(a, b)


class ResampleEmpiricalTests(unittest.TestCase):
    def test_values_subset_of_source(self) -> None:
        source = np.asarray([2, 4, 7, 9], dtype=int)
        rng = np.random.default_rng(20)
        drawn = _resample_empirical(source, 40, rng=rng)
        self.assertTrue(np.all(np.isin(drawn, source)))

    def test_n_nonpositive_empty(self) -> None:
        source = np.asarray([1, 2], dtype=int)
        rng = np.random.default_rng(21)
        self.assertEqual(_resample_empirical(source, 0, rng=rng).size, 0)
        self.assertEqual(_resample_empirical(source, -3, rng=rng).size, 0)

    def test_empty_source_raises(self) -> None:
        rng = np.random.default_rng(22)
        with self.assertRaises(ValueError):
            _resample_empirical(np.asarray([], dtype=int), 5, rng=rng)


class _AlphaModel(SimpleNamespace):
    """Minimal stand-in for powerlaw.Fit.power_law (only ``alpha`` is read)."""


class SimulateSemiparametricSampleTests(unittest.TestCase):
    def test_doubly_bounded_length_and_support(self) -> None:
        data = np.asarray(
            [1, 1, 1, 2, 2, 3, 4, 5, 8, 12, 20, 50],
            dtype=int,
        )
        xmin, xmax = 3.0, 12.0
        model = _AlphaModel(alpha=2.0)
        rng = np.random.default_rng(30)
        out = _simulate_semiparametric_sample(
            data=data,
            empirical_model=model,
            empirical_xmin=xmin,
            empirical_xmax=xmax,
            rng=rng,
        )
        self.assertEqual(out.size, data.size)

        below = set(data[data < xmin].tolist())
        above = set(data[data > xmax].tolist())
        for v in out:
            if v < xmin:
                self.assertIn(int(v), below)
            elif v > xmax:
                self.assertIn(int(v), above)
            else:
                self.assertGreaterEqual(int(v), int(xmin))
                self.assertLessEqual(int(v), int(xmax))

    def test_doubly_bounded_no_above_region_does_not_crash(self) -> None:
        # All mass <= xmax: above region empty; mid draws stay in [xmin, xmax].
        data = np.asarray([1, 1, 2, 3, 4, 5, 5, 6], dtype=int)
        xmin, xmax = 3.0, 6.0
        model = _AlphaModel(alpha=2.1)
        rng = np.random.default_rng(31)
        out = _simulate_semiparametric_sample(
            data=data,
            empirical_model=model,
            empirical_xmin=xmin,
            empirical_xmax=xmax,
            rng=rng,
        )
        self.assertEqual(out.size, data.size)
        self.assertTrue(np.all(out <= xmax))

    def test_unbounded_body_and_tail_support(self) -> None:
        data = np.asarray([1, 1, 1, 2, 2, 3, 5, 8, 13, 21], dtype=int)
        xmin = 3.0
        model = _AlphaModel(alpha=2.3)
        rng = np.random.default_rng(32)
        out = _simulate_semiparametric_sample(
            data=data,
            empirical_model=model,
            empirical_xmin=xmin,
            empirical_xmax=float("nan"),
            rng=rng,
        )
        self.assertEqual(out.size, data.size)
        body_vals = set(data[data < xmin].tolist())
        for v in out:
            if v < xmin:
                self.assertIn(int(v), body_vals)
            else:
                self.assertGreaterEqual(int(v), int(xmin))

    def test_empty_data_returns_empty_copy(self) -> None:
        data = np.asarray([], dtype=int)
        out = _simulate_semiparametric_sample(
            data=data,
            empirical_model=_AlphaModel(alpha=2.0),
            empirical_xmin=1.0,
            empirical_xmax=float("nan"),
            rng=np.random.default_rng(33),
        )
        self.assertEqual(out.size, 0)

    def test_average_region_shares_near_empirical(self) -> None:
        data = np.asarray(
            [1, 1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 15, 20, 40],
            dtype=int,
        )
        xmin, xmax = 3.0, 10.0
        k = float(data.size)
        share_below = float(np.sum(data < xmin)) / k
        share_mid = float(np.sum((data >= xmin) & (data <= xmax))) / k
        share_above = float(np.sum(data > xmax)) / k

        model = _AlphaModel(alpha=2.0)
        n_reps = 200
        counts = np.zeros(3, dtype=float)
        for seed in range(n_reps):
            out = _simulate_semiparametric_sample(
                data=data,
                empirical_model=model,
                empirical_xmin=xmin,
                empirical_xmax=xmax,
                rng=np.random.default_rng(1000 + seed),
            )
            counts[0] += np.sum(out < xmin)
            counts[1] += np.sum((out >= xmin) & (out <= xmax))
            counts[2] += np.sum(out > xmax)
        shares = counts / (n_reps * k)
        self.assertAlmostEqual(shares[0], share_below, delta=0.05)
        self.assertAlmostEqual(shares[1], share_mid, delta=0.05)
        self.assertAlmostEqual(shares[2], share_above, delta=0.05)


if __name__ == "__main__":
    unittest.main()
