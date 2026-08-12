"""Clauset–Shalizi–Newman goodness-of-fit via semiparametric bootstrap."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import EXCLUDED_HEAD_VARIANTS
from .data import to_observation_array
from .fitting import (
    _powerlaw_fit,
    fit_doubly_bounded_xmax_candidates,
    select_xmax_by_min_ks,
)
from .sampling import _simulate_semiparametric_sample


def _failed_gof_result(
    *,
    xmin: float = float("nan"),
    xmax: float = float("nan"),
    alpha: float = float("nan"),
    empirical_d: float = float("nan"),
    n_bootstraps: int,
) -> dict[str, Any]:
    """Return an empty GOF result when bootstrap cannot be completed."""
    return {
        "xmin": float(xmin),
        "xmax": float(xmax),
        "alpha": float(alpha),
        "empirical_d": float(empirical_d),
        "gof_p": float("nan"),
        "n_success": 0,
        "n_failed": int(n_bootstraps),
        "simulated_distances": np.asarray([], dtype=float),
    }


def _gof_from_simulated_distances(
    *,
    empirical_xmin: float,
    empirical_xmax: float,
    empirical_alpha: float,
    empirical_d: float,
    simulated_distances: list[float],
    n_failed: int,
    n_bootstraps: int,
) -> dict[str, Any]:
    """Compute p-value from collected synthetic KS distances."""
    simulated_distances_arr = np.asarray(simulated_distances, dtype=float)
    n_success = int(simulated_distances_arr.size)
    if n_success == 0:
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )
    n_at_least = int(np.sum(simulated_distances_arr >= empirical_d))
    return {
        "xmin": empirical_xmin,
        "xmax": empirical_xmax,
        "alpha": empirical_alpha,
        "empirical_d": empirical_d,
        "gof_p": float(n_at_least / n_success),
        "n_success": n_success,
        "n_failed": int(n_failed),
        "simulated_distances": simulated_distances_arr,
    }


def _run_bootstrap_ks(
    *,
    data: np.ndarray,
    empirical_model: Any,
    empirical_xmin: float,
    empirical_xmax: float,
    n_bootstraps: int,
    random_seed: int,
    discrete: bool,
    synthetic_ks: Callable[[np.ndarray, np.random.Generator], float],
) -> tuple[list[float], int]:
    """Shared bootstrap loop: simulate, measure KS, count failures."""
    rng = np.random.default_rng(random_seed)
    simulated_distances: list[float] = []
    n_failed = 0
    for _ in range(n_bootstraps):
        try:
            synthetic_data = _simulate_semiparametric_sample(
                data=data,
                empirical_model=empirical_model,
                empirical_xmin=empirical_xmin,
                empirical_xmax=empirical_xmax,
                rng=rng,
                discrete=discrete,
            )
            simulated_d = float(synthetic_ks(synthetic_data, rng))
            if not np.isfinite(simulated_d):
                raise ValueError("non-finite synthetic KS distance")
            simulated_distances.append(simulated_d)
        except Exception:
            n_failed += 1
    return simulated_distances, n_failed


def _gof_bootstrap_refit_all_parameters(
    data: np.ndarray,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    xmin: float | None = None,
    xmax: float | None = None,
    *,
    discrete: bool = True,
) -> dict[str, Any]:
    """Clauset GOF that refits free cutoffs and alpha on each bootstrap sample.

    ``xmin is None`` / ``xmax is None`` means that cutoff is refit; a provided
    cutoff is treated as a model constraint.
    """
    data = to_observation_array(data, discrete=discrete)

    try:
        empirical_fit = _powerlaw_fit(data, xmin=xmin, xmax=xmax, discrete=discrete)
        empirical_model = empirical_fit.power_law
        empirical_xmin = float(empirical_model.xmin)
        empirical_alpha = float(empirical_model.alpha)
        empirical_d = float(empirical_model.D)
        xmax_raw = getattr(empirical_fit, "xmax", None)
        empirical_xmax = (
            float(xmax_raw)
            if xmax_raw is not None and np.isfinite(float(xmax_raw))
            else float("nan")
        )
    except Exception:
        return _failed_gof_result(n_bootstraps=n_bootstraps)

    if not (
        np.isfinite(empirical_xmin)
        and np.isfinite(empirical_alpha)
        and np.isfinite(empirical_d)
    ):
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    in_tail = data >= empirical_xmin
    if np.isfinite(empirical_xmax):
        in_tail &= data <= empirical_xmax
    if int(np.sum(in_tail)) < 2:
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    synth_xmin = xmin
    synth_xmax = xmax if (xmax is not None and np.isfinite(float(xmax))) else None

    def synthetic_ks(synthetic_data: np.ndarray, _rng: np.random.Generator) -> float:
        synthetic_fit = _powerlaw_fit(
            synthetic_data,
            xmin=synth_xmin,
            xmax=synth_xmax,
            discrete=discrete,
        )
        return float(synthetic_fit.power_law.D)

    simulated_distances, n_failed = _run_bootstrap_ks(
        data=data,
        empirical_model=empirical_model,
        empirical_xmin=empirical_xmin,
        empirical_xmax=empirical_xmax,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        discrete=discrete,
        synthetic_ks=synthetic_ks,
    )
    return _gof_from_simulated_distances(
        empirical_xmin=empirical_xmin,
        empirical_xmax=empirical_xmax,
        empirical_alpha=empirical_alpha,
        empirical_d=empirical_d,
        simulated_distances=simulated_distances,
        n_failed=n_failed,
        n_bootstraps=n_bootstraps,
    )


def _gof_bootstrap_doubly_bounded_selection(
    data: np.ndarray,
    *,
    selected_evaluation: Mapping[str, Any],
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    excluded_head_variants: Sequence[int] = EXCLUDED_HEAD_VARIANTS,
    discrete: bool = True,
) -> dict[str, Any]:
    """GOF that re-runs min-KS xmax selection inside each bootstrap."""
    data = to_observation_array(data, discrete=discrete)
    fit = selected_evaluation["fit"]
    empirical_model = fit.power_law
    empirical_xmin = float(selected_evaluation["xmin"])
    empirical_xmax = float(selected_evaluation["xmax"])
    empirical_d = float(selected_evaluation["KS_D"])
    empirical_alpha = float(selected_evaluation["alpha"])

    if not (np.isfinite(empirical_xmin) and np.isfinite(empirical_xmax)):
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    in_tail = (data >= empirical_xmin) & (data <= empirical_xmax)
    if int(np.sum(in_tail)) < 2:
        return _failed_gof_result(
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
            n_bootstraps=n_bootstraps,
        )

    def synthetic_ks(synthetic_data: np.ndarray, _rng: np.random.Generator) -> float:
        cand_rows = fit_doubly_bounded_xmax_candidates(
            synthetic_data,
            excluded_head_variants=excluded_head_variants,
            discrete=discrete,
        )
        chosen = select_xmax_by_min_ks(cand_rows)
        if chosen is None:
            raise ValueError("no usable candidate in bootstrap replication")
        return float(chosen["KS_D"])

    simulated_distances, n_failed = _run_bootstrap_ks(
        data=data,
        empirical_model=empirical_model,
        empirical_xmin=empirical_xmin,
        empirical_xmax=empirical_xmax,
        n_bootstraps=n_bootstraps,
        random_seed=random_seed,
        discrete=discrete,
        synthetic_ks=synthetic_ks,
    )
    return _gof_from_simulated_distances(
        empirical_xmin=empirical_xmin,
        empirical_xmax=empirical_xmax,
        empirical_alpha=empirical_alpha,
        empirical_d=empirical_d,
        simulated_distances=simulated_distances,
        n_failed=n_failed,
        n_bootstraps=n_bootstraps,
    )
