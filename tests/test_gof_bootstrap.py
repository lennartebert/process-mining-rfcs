"""Tests for Clauset semiparametric GOF bootstrap sample generation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from utils.rfc.statistical_tests import (
    _sample_discrete_power_law_truncated,
    _simulate_semiparametric_sample,
)


class SemiparametricBootstrapGenerationTests(unittest.TestCase):
    def test_lower_bounded_sample_size_fixed_and_region_counts_vary(self) -> None:
        rng = np.random.default_rng(0)
        data = np.asarray(
            [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40],
            dtype=int,
        )
        xmin = 3.0
        model = SimpleNamespace(alpha=2.0)
        k = data.size
        n_fit_emp = int(np.sum(data >= xmin))
        self.assertGreater(n_fit_emp, 0)
        self.assertLess(n_fit_emp, k)

        mid_counts: list[int] = []
        for _ in range(40):
            sample = _simulate_semiparametric_sample(
                data=data,
                empirical_model=model,
                empirical_xmin=xmin,
                empirical_xmax=float("nan"),
                rng=rng,
                discrete=True,
            )
            self.assertEqual(sample.size, k)
            mid_counts.append(int(np.sum(sample >= xmin)))

        self.assertGreater(len(set(mid_counts)), 1)
        self.assertNotEqual(mid_counts, [n_fit_emp] * len(mid_counts))

    def test_doubly_bounded_sample_size_fixed_and_region_counts_vary(self) -> None:
        rng = np.random.default_rng(1)
        data = np.asarray(
            [1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 22, 30, 45, 60, 80],
            dtype=int,
        )
        xmin, xmax = 3.0, 20.0
        model = SimpleNamespace(alpha=2.0)
        k = data.size
        n_mid_emp = int(np.sum((data >= xmin) & (data <= xmax)))
        self.assertGreater(n_mid_emp, 0)
        self.assertLess(n_mid_emp, k)

        mid_counts: list[int] = []
        for _ in range(40):
            sample = _simulate_semiparametric_sample(
                data=data,
                empirical_model=model,
                empirical_xmin=xmin,
                empirical_xmax=xmax,
                rng=rng,
                discrete=True,
            )
            self.assertEqual(sample.size, k)
            mid = sample[(sample >= xmin) & (sample <= xmax)]
            below = sample[sample < xmin]
            above = sample[sample > xmax]
            self.assertEqual(below.size + mid.size + above.size, k)
            mid_counts.append(int(mid.size))
            # Fitted-region draws must lie inside [xmin, xmax].
            self.assertTrue(np.all(mid >= xmin))
            self.assertTrue(np.all(mid <= xmax))

        self.assertGreater(len(set(mid_counts)), 1)
        self.assertNotEqual(mid_counts, [n_mid_emp] * len(mid_counts))

    def test_truncated_draws_never_outside_support(self) -> None:
        rng = np.random.default_rng(2)
        xmin, xmax, alpha = 5, 25, 2.3
        draws = _sample_discrete_power_law_truncated(
            5000, alpha=alpha, xmin=xmin, xmax=xmax, rng=rng
        )
        self.assertEqual(draws.size, 5000)
        self.assertTrue(np.all(draws >= xmin))
        self.assertTrue(np.all(draws <= xmax))

    def test_truncated_sampler_has_no_clipping_mass_at_xmax(self) -> None:
        """Clipping unbounded draws onto xmax piles mass at the boundary.

        Exact truncated probabilities should match the normalized x^{-alpha}
        weights, so P(X=xmax) stays near its theoretical share.
        """
        rng = np.random.default_rng(3)
        xmin, xmax, alpha = 2, 10, 1.8
        support = np.arange(xmin, xmax + 1, dtype=int)
        weights = support.astype(float) ** (-alpha)
        p_xmax = float(weights[-1] / weights.sum())

        n = 20_000
        draws = _sample_discrete_power_law_truncated(
            n, alpha=alpha, xmin=xmin, xmax=xmax, rng=rng
        )
        emp = float(np.mean(draws == xmax))
        # Clipping mass would be sum_{x>=xmax} p_unbounded(x) >> p_xmax.
        self.assertLess(abs(emp - p_xmax), 0.02)

        # Explicitly contrast with clipping: pile-up at xmax is much larger.
        unbounded = []
        while len(unbounded) < n:
            need = n - len(unbounded)
            # Reuse lower-bounded helper via truncated rejection path's sibling.
            from utils.rfc.statistical_tests import (
                _sample_discrete_power_law_lower_bounded,
            )

            batch = _sample_discrete_power_law_lower_bounded(
                need, alpha=alpha, xmin=xmin, rng=rng
            )
            unbounded.extend(batch.tolist())
        unbounded_arr = np.asarray(unbounded[:n], dtype=int)
        clipped = np.minimum(unbounded_arr, xmax)
        clipped_mass = float(np.mean(clipped == xmax))
        self.assertGreater(clipped_mass, p_xmax + 0.05)
        self.assertLess(emp, clipped_mass - 0.03)


if __name__ == "__main__":
    unittest.main()
