"""Power-law samplers and Clauset semiparametric synthetic samples.

Synthetic replicates keep total size ``K`` fixed while resampling empirical
mass outside the fitted support and drawing from a correctly normalized
(power-law / truncated) distribution on the fitted support — never by clipping
unbounded draws.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import powerlaw


def _sample_discrete_power_law_lower_bounded(
    n: int,
    *,
    alpha: float,
    xmin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` integers from a discrete power law on ``x >= xmin`` (no upper bound)."""
    if n <= 0:
        return np.asarray([], dtype=int)
    xmin_i = int(np.floor(xmin))
    dist = powerlaw.Power_Law(
        xmin=xmin_i,
        parameters=[float(alpha)],
        discrete=True,
    )
    out: list[int] = []
    # Approximate sampler is fast; reject any rare draws below xmin (never clip).
    while len(out) < n:
        need = n - len(out)
        batch = np.asarray(
            dist.generate_random(need, estimate_discrete=False),
            dtype=float,
        )
        batch = np.asarray(np.round(batch), dtype=int)
        batch = batch[batch >= xmin_i]
        if batch.size:
            out.extend(batch.tolist())
    return np.asarray(out[:n], dtype=int)


def _sample_discrete_power_law_truncated(
    n: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` integers from a discrete PL normalized on ``{xmin,...,xmax}``."""
    if n <= 0:
        return np.asarray([], dtype=int)
    xmin_i = int(np.floor(xmin))
    xmax_i = int(np.floor(xmax))
    if xmax_i < xmin_i:
        raise ValueError(f"xmax ({xmax_i}) < xmin ({xmin_i})")
    support = np.arange(xmin_i, xmax_i + 1, dtype=int)
    if support.size == 1:
        return np.full(n, support[0], dtype=int)

    # Direct categorical sampling on the truncated support (no clipping).
    # For extremely wide supports fall back to rejection from the lower-bounded PL.
    max_direct_support = 100_000
    if support.size <= max_direct_support:
        log_w = -float(alpha) * np.log(support.astype(float))
        log_w -= float(np.max(log_w))
        weights = np.exp(log_w)
        probs = weights / weights.sum()
        return rng.choice(support, size=n, replace=True, p=probs).astype(int)

    out: list[int] = []
    while len(out) < n:
        need = n - len(out)
        # Oversample: acceptance rate is the truncated mass under the lower-bounded PL.
        batch = _sample_discrete_power_law_lower_bounded(
            max(need * 2, need),
            alpha=alpha,
            xmin=xmin_i,
            rng=rng,
        )
        batch = batch[batch <= xmax_i]
        if batch.size:
            out.extend(batch.tolist())
    return np.asarray(out[:n], dtype=int)


def _sample_continuous_power_law_lower_bounded(
    n: int,
    *,
    alpha: float,
    xmin: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` values from a continuous power law on ``x >= xmin``."""
    if n <= 0:
        return np.asarray([], dtype=float)
    # Inverse CDF: x = xmin * U^(-1/(alpha-1)) for alpha > 1.
    alpha = float(alpha)
    xmin = float(xmin)
    if alpha <= 1.0:
        # Degenerate / heavy regime: fall back to powerlaw's generator.
        dist = powerlaw.Power_Law(xmin=xmin, parameters=[alpha], discrete=False)
        return np.asarray(dist.generate_random(n), dtype=float)
    u = rng.random(n)
    return xmin * np.power(u, -1.0 / (alpha - 1.0))


def _sample_continuous_power_law_truncated(
    n: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n`` values from a continuous PL truncated to ``[xmin, xmax]``."""
    if n <= 0:
        return np.asarray([], dtype=float)
    xmin = float(xmin)
    xmax = float(xmax)
    alpha = float(alpha)
    if xmax < xmin:
        raise ValueError(f"xmax ({xmax}) < xmin ({xmin})")
    # Inverse CDF of truncated continuous power law.
    if alpha == 1.0:
        # Limit form: uniform in log-space between xmin and xmax.
        u = rng.random(n)
        return xmin * np.exp(u * np.log(xmax / xmin))
    if alpha < 1.0:
        out: list[float] = []
        while len(out) < n:
            need = n - len(out)
            batch = _sample_continuous_power_law_lower_bounded(
                max(need * 2, need),
                alpha=alpha,
                xmin=xmin,
                rng=rng,
            )
            batch = batch[batch <= xmax]
            if batch.size:
                out.extend(batch.tolist())
        return np.asarray(out[:n], dtype=float)
    # CDF(x) = 1 - (x/xmin)^(1-alpha); truncate and invert.
    cmin = xmin ** (1.0 - alpha)
    cmax = xmax ** (1.0 - alpha)
    u = rng.random(n)
    return np.power(cmin - u * (cmin - cmax), 1.0 / (1.0 - alpha))


def _sample_fitted_power_law(
    n: int,
    *,
    alpha: float,
    xmin: float,
    xmax: float | None,
    rng: np.random.Generator,
    discrete: bool,
) -> np.ndarray:
    """Sample from the fitted PL on its proper support (no clipping)."""
    if xmax is not None and np.isfinite(xmax):
        if discrete:
            return _sample_discrete_power_law_truncated(
                n, alpha=alpha, xmin=xmin, xmax=xmax, rng=rng
            )
        return _sample_continuous_power_law_truncated(
            n, alpha=alpha, xmin=xmin, xmax=xmax, rng=rng
        )
    if discrete:
        return _sample_discrete_power_law_lower_bounded(
            n, alpha=alpha, xmin=xmin, rng=rng
        )
    return _sample_continuous_power_law_lower_bounded(
        n, alpha=alpha, xmin=xmin, rng=rng
    )


def _resample_empirical(
    values: np.ndarray,
    n: int,
    *,
    rng: np.random.Generator,
    discrete: bool,
) -> np.ndarray:
    """Resample ``n`` observations from an empirical region (with replacement)."""
    if n <= 0:
        return np.asarray([], dtype=int if discrete else float)
    if values.size == 0:
        raise ValueError("cannot resample from an empty empirical region")
    drawn = rng.choice(values, size=n, replace=True)
    return drawn.astype(int if discrete else float, copy=False)


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
    data = np.asarray(data)
    if discrete:
        data = np.asarray(data, dtype=int)
    else:
        data = np.asarray(data, dtype=float)

    k_total = int(data.size)
    if k_total == 0:
        return data.copy()

    alpha = float(getattr(empirical_model, "alpha"))
    xmin = float(empirical_xmin)
    has_xmax = bool(np.isfinite(empirical_xmax))
    xmax = float(empirical_xmax) if has_xmax else None

    if has_xmax:
        assert xmax is not None
        below = data[data < xmin]
        mid = data[(data >= xmin) & (data <= xmax)]
        above = data[data > xmax]
        shares = np.asarray(
            [below.size, mid.size, above.size], dtype=float
        ) / float(k_total)
        n_below, n_mid, n_above = (int(x) for x in rng.multinomial(k_total, shares))
        parts: list[np.ndarray] = []
        if n_below:
            parts.append(
                _resample_empirical(below, n_below, rng=rng, discrete=discrete)
            )
        if n_mid:
            parts.append(
                _sample_fitted_power_law(
                    n_mid,
                    alpha=alpha,
                    xmin=xmin,
                    xmax=xmax,
                    rng=rng,
                    discrete=discrete,
                )
            )
        if n_above:
            parts.append(
                _resample_empirical(above, n_above, rng=rng, discrete=discrete)
            )
        if not parts:
            return data[:0].copy()
        return np.concatenate(parts)

    below = data[data < xmin]
    n_fit = int(np.sum(data >= xmin))
    p_fit = n_fit / float(k_total)
    n_tail = int(rng.binomial(k_total, p_fit))
    n_body = k_total - n_tail
    parts = []
    if n_body:
        parts.append(_resample_empirical(below, n_body, rng=rng, discrete=discrete))
    if n_tail:
        parts.append(
            _sample_fitted_power_law(
                n_tail,
                alpha=alpha,
                xmin=xmin,
                xmax=None,
                rng=rng,
                discrete=discrete,
            )
        )
    if not parts:
        return data[:0].copy()
    return np.concatenate(parts)
