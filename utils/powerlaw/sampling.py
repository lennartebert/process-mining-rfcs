"""Discrete power-law samplers and Clauset semiparametric synthetic samples.

Synthetic replicates keep total size ``K`` fixed while resampling empirical
mass outside the fitted support and drawing from a correctly normalized
(power-law / truncated) distribution on the fitted support - never by clipping
draws onto xmax.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import powerlaw
from scipy.special import zeta


def _sample_fitted_power_law(
    n: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float | None,
    rng: np.random.Generator,
    discrete: bool = True,
) -> np.ndarray:
    """Sample from a fitted discrete power law on [xmin, xmax].

    xmin=None means xmin=1.
    xmax=None means no upper bound.
    """
    if not discrete:
        raise NotImplementedError(
            "continuous power-law sampling (discrete=False) is not implemented yet"
        )

    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return np.empty(0, dtype=int)

    alpha = float(alpha)

    def _integer_bound(name: str, value: float | None, default: int) -> int:
        if value is None:
            return default
        value = float(value)
        if not np.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer or None")
        return int(value)

    xmin_i = _integer_bound("xmin", xmin, 1)

    if xmin_i < 1:
        raise ValueError("xmin must be >= 1")

    # Finite support: exact categorical sampling.
    if xmax is not None:
        xmax_i = _integer_bound("xmax", xmax, 1)

        if xmax_i < xmin_i:
            raise ValueError("xmax must be >= xmin")

        support = np.arange(xmin_i, xmax_i + 1)

        log_weights = -alpha * np.log(support)
        log_weights -= log_weights.max()

        probabilities = np.exp(log_weights)
        probabilities /= probabilities.sum()

        return rng.choice(
            support,
            size=n,
            replace=True,
            p=probabilities,
        )

    # Infinite support requires alpha > 1.
    if alpha <= 1:
        raise ValueError("Unbounded discrete power law requires alpha > 1")

    # Exact inverse-CDF sampling using the Hurwitz zeta function.
    normalization = float(zeta(alpha, xmin_i))

    def probability_above(x: int) -> float:
        """P(X > x) under the fitted lower-bounded power law."""
        return float(zeta(alpha, x + 1)) / normalization

    draws = np.empty(n, dtype=int)

    for i, u in enumerate(rng.random(n)):
        if u == 0:
            draws[i] = xmin_i
            continue

        target = 1.0 - u

        # Bracket the inverse CDF.
        lo = xmin_i - 1
        hi = xmin_i

        while probability_above(hi) > target:
            hi *= 2

        # Find smallest x with F(x) >= u.
        while hi - lo > 1:
            mid = (lo + hi) // 2

            if probability_above(mid) <= target:
                hi = mid
            else:
                lo = mid

        draws[i] = hi

    return draws


def _resample_empirical(
    values: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    discrete: bool = True,
) -> np.ndarray:
    """Resample ``n`` observations from an empirical region (with replacement)."""
    if not discrete:
        raise NotImplementedError(
            "continuous empirical resampling (discrete=False) is not implemented yet"
        )
    if n <= 0:
        return np.asarray([], dtype=int)
    if values.size == 0:
        raise ValueError("cannot resample from an empty empirical region")
    drawn = rng.choice(values, size=n, replace=True)
    return drawn.astype(int, copy=False)


def _simulate_semiparametric_sample(
    *,
    data: np.ndarray,
    empirical_model: Any,
    empirical_xmin: float,
    empirical_xmax: float,
    rng: np.random.Generator,
    discrete: bool = True,
) -> np.ndarray:
    """Build one Clauset semiparametric synthetic sample of size ``K=len(data)``.

    The empirically fitted model (alpha, xmin, and xmax if present) is the fixed
    null. Each of the ``K`` synthetic observations is drawn independently from a
    mixture over empirical regions, so region counts vary across replicates while
    the total sample size stays fixed.
    """
    if not discrete:
        raise NotImplementedError(
            "continuous semiparametric sampling (discrete=False) is not implemented yet"
        )

    data = np.asarray(data, dtype=int)
    k_total = int(data.size)
    if k_total == 0:
        return data.copy()

    alpha = float(getattr(empirical_model, "alpha"))
    xmin = float(empirical_xmin)
    xmax = float(empirical_xmax) if np.isfinite(empirical_xmax) else None

    if xmax is not None:
        # Doubly bounded: three regions - below xmin, fitted [xmin, xmax], above xmax.
        below = data[data < xmin]
        mid = data[(data >= xmin) & (data <= xmax)]
        above = data[data > xmax]
        # Region counts -> Multinomial with empirical shares; total size stays K.
        shares = np.asarray(
            [below.size, mid.size, above.size], dtype=float
        ) / float(k_total)
        n_below, n_mid, n_above = (int(x) for x in rng.multinomial(k_total, shares))
        parts: list[np.ndarray] = []
        if n_below:
            # Outside the fit: resample from the empirical body.
            parts.append(
                _resample_empirical(below, n_below, rng=rng, discrete=True)
            )
        if n_mid:
            # Fitted support: draw from the truncated power law (no clipping).
            parts.append(
                _sample_fitted_power_law(
                    n_mid,
                    alpha=alpha,
                    xmin=xmin,
                    xmax=xmax,
                    rng=rng,
                    discrete=True,
                )
            )
        if n_above:
            # Outside the fit: resample from the empirical head above xmax.
            parts.append(
                _resample_empirical(above, n_above, rng=rng, discrete=True)
            )
        if not parts:
            return data[:0].copy()
        return np.concatenate(parts)

    # No xmax: body below xmin (empirical) vs tail x >= xmin (power law).
    below = data[data < xmin]
    n_fit = int(np.sum(data >= xmin))
    p_fit = n_fit / float(k_total)
    n_tail = int(rng.binomial(k_total, p_fit))
    n_body = k_total - n_tail
    parts = []
    if n_body:
        parts.append(_resample_empirical(below, n_body, rng=rng, discrete=True))
    if n_tail:
        parts.append(
            _sample_fitted_power_law(
                n_tail,
                alpha=alpha,
                xmin=xmin,
                xmax=None,
                rng=rng,
                discrete=True,
            )
        )
    if not parts:
        return data[:0].copy()
    return np.concatenate(parts)
