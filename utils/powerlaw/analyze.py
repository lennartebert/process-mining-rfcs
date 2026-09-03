"""Clauset-Shalizi-Newman power-law analysis (one model per run).

Top-to-bottom story:

1. validate observations
2. MLE for the chosen model (free vs fixed cutoffs)
3. bootstrap GOF reusing the empirical fit (refit only free params)
4. LLR vs fixed alternative families
5. classify + build CSV rows
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import powerlaw

from .sampling import _simulate_semiparametric_sample

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FULL_RANGE_POWER_LAW = "full_range_power_law"
LOWER_BOUNDED_POWER_LAW = "lower_bounded_power_law"
DOUBLY_BOUNDED_POWER_LAW = "doubly_bounded_power_law"

DISTRIBUTION_NAMES: tuple[str, ...] = (
    FULL_RANGE_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    DOUBLY_BOUNDED_POWER_LAW,
)

DEFAULT_SIGNIFICANCE_LEVEL = 0.10
DEFAULT_MINIMUM_FITTED_TYPES = 30
DEFAULT_DOUBLY_BOUNDED_EXCLUDE_HEAD_VARIANTS: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    5,
    10,
)

# Vuong/LLR competitors for powerlaw.Fit.distribution_compare (not a user knob).
_ALTERNATIVES: list[tuple[str, bool]] = [ # (model name, nested)
    ("lognormal", False),
    ("exponential", False),
    ("stretched_exponential", False),
    ("truncated_power_law", True),
]
# Preference (Other preferred / PL best) ignores nested truncated power law.
CLASSIFICATION_ALTERNATIVES: tuple[str, ...] = (
    "exponential",
    "lognormal",
    "stretched_exponential",
)

# Compact five-way labels used by compose tables (letter codes A–E).
CLASSIFICATION_CODES: tuple[str, ...] = ("A", "B", "C", "D", "E")
CLASSIFICATION_LABELS: dict[str, str] = {
    "A": "Insufficient support",
    "B": "PL not plausible",
    "C": "PL plausible, alternative preferred",
    "D": "PL plausible, PL best",
    "E": "PL plausible, alternatives inconclusive",
}
CLASSIFICATION_CELL_COLORS: dict[str, str] = {
    "A": "FFFFFF",
    "B": "D55E00",
    "C": "E69F00",
    "D": "009E73",
    "E": "56B4E9",
}

# powerlaw default alpha upper bound is 3; raise so discrete MLE is not pinned.
_ALPHA_PARAMETER_RANGE: list[float] = [0.0, 4.0]
_ALPHA_UPPER_BOUND = 4.0
_ALPHA_BOUNDARY_TOL = 1e-6

_GOF_KIND_REFIT = "refit_all_parameters"
_GOF_KIND_SKIPPED = "skipped_invalid_fit"


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def _validate_observations(
    data: np.ndarray | Sequence[Any],
    *,
    discrete: bool = True,
) -> np.ndarray:
    """Return a 1-D positive observation array. ``discrete=False`` not implemented."""
    if not discrete:
        raise NotImplementedError(
            "continuous Clauset fits (discrete=False) are not implemented yet"
        )
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"data must be one-dimensional; got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("Observation array is empty after validation.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Observation array contains non-finite values.")
    if np.any(arr <= 0):
        raise ValueError("All observations must be positive.")
    if not np.allclose(arr, np.round(arr)):
        raise ValueError("Discrete observations must be integers.")
    return np.asarray(np.round(arr), dtype=int)


def _descriptive_stats(data: np.ndarray) -> dict[str, Any]:
    """Basic counts for the summary row (discrete frequencies)."""
    n_types = int(data.size)
    n_singletons = int(np.sum(data == 1))
    return {
        "n_occurrences": int(data.sum()),
        "n_types": n_types,
        "n_singletons": n_singletons,
        "singleton_share": n_singletons / n_types if n_types else 0.0,
    }


# ---------------------------------------------------------------------------
# MLE
# ---------------------------------------------------------------------------


def _powerlaw_fit(
    data: np.ndarray,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> powerlaw.Fit:
    """Wrapper for ``powerlaw.Fit`` with our alpha search range (discrete only)."""
    return powerlaw.Fit(
        data,
        discrete=True,
        xmin=xmin,
        xmax=xmax,
        verbose=False,
        parameter_ranges={"alpha": list(_ALPHA_PARAMETER_RANGE)},
    )


def _fitted_region_mask(
    data: np.ndarray, xmin: float, xmax: float | None
) -> np.ndarray:
    """Boolean mask for observations in [xmin, xmax] (xmax inclusive when set)."""
    mask = data >= xmin
    if xmax is not None and np.isfinite(xmax):
        mask &= data <= xmax
    return mask


def _log_range(xmin: float, xmax: float | None) -> float:
    """Return log10(xmax / xmin) when both cutoffs are finite and positive.

    Informative statistic only (fitted-support width); not used in GOF/LLR/classify.
    """
    if xmax is None or not np.isfinite(xmax) or not np.isfinite(xmin):
        return float("nan")
    if xmin <= 0 or xmax <= 0:
        return float("nan")
    return float(np.log10(xmax / xmin))


def _powerlaw_fit_with_stats(
    data: np.ndarray,
    *,
    xmin: float | None = None,
    xmax: float | None = None,
) -> dict[str, Any]:
    """Run ``_powerlaw_fit`` and attach summary stats (alpha, KS, shares, …).

    ``xmin=None`` means estimate xmin; a finite ``xmin`` is held fixed.
    ``xmax=None`` means no upper cutoff (not estimated); a finite ``xmax`` is held fixed.
    """
    fit = _powerlaw_fit(data, xmin=xmin, xmax=xmax)
    pl = fit.power_law
    alpha = float(getattr(pl, "alpha", np.nan))
    # xmin may be estimated (xmin=None) or fixed; always read the value used by the fit.
    xmin_hat = float(getattr(pl, "xmin", np.nan))
    # xmax is never estimated by powerlaw - Fit only stores the constraint we passed.
    assert getattr(fit, "xmax", None) == xmax
    # Discrete fits only: cutoffs are None/missing or integer-valued.
    if np.isfinite(xmin_hat):
        assert float(xmin_hat).is_integer(), (
            f"xmin must be an integer when discrete, got {xmin_hat!r}"
        )
        xmin_out: int | float = int(xmin_hat)
    else:
        xmin_out = float("nan")
    if xmax is None:
        xmax_out: int | float = float("nan")
    else:
        assert float(xmax).is_integer(), (
            f"xmax must be an integer when discrete, got {xmax!r}"
        )
        xmax_out = int(xmax)
    ks_d = float(getattr(pl, "D", np.nan))
    log_range = _log_range(
        xmin_out, float(xmax_out) if np.isfinite(xmax_out) else float(np.max(data))
    )

    n_occurrences = int(data.sum())
    n_types = int(data.size)
    n_fitted_types = np.nan
    fitted_type_share = np.nan
    fitted_occurrence_share = np.nan
    if np.isfinite(xmin_out):
        fitted_mask = _fitted_region_mask(data, float(xmin_out), xmax)
        n_fitted_types = int(np.sum(fitted_mask))
        n_fitted_occ = int(data[fitted_mask].sum())
        fitted_type_share = n_fitted_types / n_types if n_types else np.nan
        fitted_occurrence_share = (
            n_fitted_occ / n_occurrences if n_occurrences else np.nan
        )

    alpha_at_boundary = bool(
        np.isfinite(alpha) and abs(alpha - _ALPHA_UPPER_BOUND) <= _ALPHA_BOUNDARY_TOL
    )
    pathology_flags = (
        ["alpha_at_parameter_boundary"] if alpha_at_boundary else []
    )
    return {
        "fit": fit,
        "alpha": alpha,
        "xmin": xmin_out,
        "xmax": xmax_out,
        "KS_D": ks_d,
        "n_fitted_types": n_fitted_types,
        "fitted_type_share": fitted_type_share,
        "fitted_occurrence_share": fitted_occurrence_share,
        "log_range": log_range,
        "alpha_at_boundary": alpha_at_boundary,
        "pathology_flags": pathology_flags,
        "fit_valid": len(pathology_flags) == 0,
    }


def _xmax_candidates(
    data: np.ndarray, doubly_bounded_exclude_head_variants: Sequence[int]
) -> list[tuple[int, float, int]]:
    """(exclude_k, xmax, actual_excluded) from ranked head-frequency exclusion."""
    ranked = np.sort(data)[::-1]
    out: list[tuple[int, float, int]] = []
    seen: set[float] = set()
    for k in doubly_bounded_exclude_head_variants:
        if k < 0 or k >= ranked.size:
            continue
        xmax = float(ranked[k])
        if xmax in seen:
            continue
        seen.add(xmax)
        out.append((int(k), xmax, int(np.sum(data > xmax))))
    return out


def _select_xmax_by_min_ks(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Pick candidate with smallest KS_D (ties: fewer exclusions, more types)."""
    usable = []
    for row in candidates:
        try:
            ks = float(row.get("KS_D", float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(ks):
            usable.append(row)
    if not usable:
        return None

    def key(row: Mapping[str, Any]) -> tuple[float, int, int]:
        return (
            float(row["KS_D"]),
            int(row.get("doubly_bounded_exclude_head_variants", 10**9) or 10**9),
            -int(row.get("n_fitted_types", 0) or 0),
        )

    return min(usable, key=key)


def _fit_doubly_bounded_candidates(
    data: np.ndarray, doubly_bounded_exclude_head_variants: Sequence[int]
) -> list[dict[str, Any]]:
    """Fit all xmax candidates (KS only). xmin free, xmax fixed per candidate."""
    rows: list[dict[str, Any]] = []
    for excl_k, xmax, actual_excl in _xmax_candidates(data, doubly_bounded_exclude_head_variants):
        try:
            # xmin free (None); xmax fixed at this candidate
            fit_result = _powerlaw_fit_with_stats(data, xmin=None, xmax=xmax)
            rows.append(
                {
                    **fit_result,
                    "log_range": _log_range(float(fit_result["xmin"]), float(xmax)),
                    "doubly_bounded_exclude_head_variants": int(excl_k),
                    "actual_excluded_variants": int(actual_excl),
                    "xmax": float(xmax),
                }
            )
        except Exception:
            continue
    return rows


def _powerlaw_fit_with_stats_for_model(
    data: np.ndarray,
    model: str,
    *,
    doubly_bounded_exclude_head_variants: Sequence[int],
) -> dict[str, Any] | None:
    """Dispatch cutoffs for ``model``, then ``_powerlaw_fit_with_stats``.

    Returns None if doubly-bounded has no usable xmax candidate.
    """
    if model == FULL_RANGE_POWER_LAW:
        # xmin fixed at 1; xmax none (unbounded above)
        return _powerlaw_fit_with_stats(data, xmin=1, xmax=None)
    if model == LOWER_BOUNDED_POWER_LAW:
        # xmin free; xmax none
        return _powerlaw_fit_with_stats(data, xmin=None, xmax=None)
    if model == DOUBLY_BOUNDED_POWER_LAW:
        # xmin free; xmax chosen by min KS among doubly_bounded_exclude_head_variants
        candidates = _fit_doubly_bounded_candidates(data, doubly_bounded_exclude_head_variants)
        selected = _select_xmax_by_min_ks(candidates)
        if selected is None:
            return None
        return dict(selected)
    raise ValueError(
        f"unknown model {model!r}; expected one of {list(DISTRIBUTION_NAMES)}"
    )


# ---------------------------------------------------------------------------
# GOF (single function; reuses empirical fit)
# ---------------------------------------------------------------------------


def _failed_gof(
    *,
    n_bootstraps: int,
    xmin: float = float("nan"),
    xmax: float = float("nan"),
    alpha: float = float("nan"),
    empirical_d: float = float("nan"),
) -> dict[str, Any]:
    """Empty GOF result when bootstrap cannot run or all replicates fail."""
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


def bootstrap_gof(
    data: np.ndarray,
    empirical_fit: Mapping[str, Any],
    model: str,
    *,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    doubly_bounded_exclude_head_variants: Sequence[int] = DEFAULT_DOUBLY_BOUNDED_EXCLUDE_HEAD_VARIANTS,
) -> dict[str, Any]:
    """Clauset semiparametric GOF; does **not** re-fit the original sample.

    Free cutoffs are re-estimated on each synthetic replicate according to
    ``model``; fixed cutoffs stay locked.
    """
    # Null hypothesis: use the already-fitted empirical parameters (no re-MLE here).
    empirical_xmin = float(empirical_fit["xmin"])
    empirical_xmax = float(empirical_fit["xmax"])
    empirical_d = float(empirical_fit["KS_D"])
    empirical_alpha = float(empirical_fit["alpha"])
    empirical_model = empirical_fit["fit"].power_law

    if not (
        np.isfinite(empirical_xmin)
        and np.isfinite(empirical_alpha)
        and np.isfinite(empirical_d)
    ):
        return _failed_gof(
            n_bootstraps=n_bootstraps,
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
        )

    # Need enough points on the fitted support to define a KS distance.
    in_tail = data >= empirical_xmin
    if np.isfinite(empirical_xmax):
        in_tail &= data <= empirical_xmax
    if int(np.sum(in_tail)) < 2:
        return _failed_gof(
            n_bootstraps=n_bootstraps,
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
        )

    # Synthetic refits: which cutoffs are free vs fixed for this model?
    if model == FULL_RANGE_POWER_LAW:
        # xmin fixed at 1; xmax none
        synth_xmin: float | None = 1.0
        synth_xmax: float | None = None
        reselect_xmax = False
    elif model == LOWER_BOUNDED_POWER_LAW:
        # xmin free; xmax none
        synth_xmin = None
        synth_xmax = None
        reselect_xmax = False
    elif model == DOUBLY_BOUNDED_POWER_LAW:
        # xmin free; xmax re-selected via min-KS each replicate
        if not np.isfinite(empirical_xmax):
            return _failed_gof(
                n_bootstraps=n_bootstraps,
                xmin=empirical_xmin,
                xmax=empirical_xmax,
                alpha=empirical_alpha,
                empirical_d=empirical_d,
            )
        synth_xmin = None
        synth_xmax = None
        reselect_xmax = True
    else:
        raise ValueError(f"unknown model {model!r}")

    rng = np.random.default_rng(random_seed)
    simulated: list[float] = []
    n_failed = 0
    for _ in range(n_bootstraps):
        try:
            # Draw one synthetic sample under the empirical null (semiparametric mixture).
            synthetic = _simulate_semiparametric_sample(
                data=data,
                empirical_model=empirical_model,
                empirical_xmin=empirical_xmin,
                empirical_xmax=empirical_xmax,
                rng=rng,
                discrete=True,
            )
            if reselect_xmax:
                # Doubly-bounded: re-select xmax from doubly_bounded_exclude_head_variants.
                cands = _fit_doubly_bounded_candidates(synthetic, doubly_bounded_exclude_head_variants)
                chosen = _select_xmax_by_min_ks(cands)
                if chosen is None:
                    raise ValueError("no usable candidate in bootstrap")
                sim_d = float(chosen["KS_D"])
            else:
                # Full-/lower-bounded: refit free cutoffs only, then read KS.
                sim_fit = _powerlaw_fit(
                    synthetic, xmin=synth_xmin, xmax=synth_xmax
                )
                sim_d = float(sim_fit.power_law.D)
            if not np.isfinite(sim_d):
                raise ValueError("non-finite synthetic KS")
            simulated.append(sim_d)
        except Exception:
            n_failed += 1

    arr = np.asarray(simulated, dtype=float)
    n_success = int(arr.size)
    if n_success == 0:
        return _failed_gof(
            n_bootstraps=n_bootstraps,
            xmin=empirical_xmin,
            xmax=empirical_xmax,
            alpha=empirical_alpha,
            empirical_d=empirical_d,
        )
    # p = fraction of synthetic KS distances at least as large as the empirical D.
    return {
        "xmin": empirical_xmin,
        "xmax": empirical_xmax,
        "alpha": empirical_alpha,
        "empirical_d": empirical_d,
        "gof_p": float(np.sum(arr >= empirical_d) / n_success),
        "n_success": n_success,
        "n_failed": int(n_failed),
        "simulated_distances": arr,
    }


# ---------------------------------------------------------------------------
# LLR vs alternatives
# ---------------------------------------------------------------------------


def _interpret_llr(
    r: float, p: float, model_1: str, model_2: str, alpha: float
) -> str:
    """Short text label for one Vuong LLR comparison."""
    if np.isfinite(r) and r == 0.0 and not np.isfinite(p):
        return "no preference (identical likelihoods)"
    if not np.isfinite(r) or not np.isfinite(p):
        return "comparison failed"
    if p >= alpha:
        if r > 0:
            direction = f"{model_1} numerically favored"
        elif r < 0:
            direction = f"{model_2} numerically favored"
        else:
            direction = "no numerical preference"
        return f"inconclusive ({direction})"
    if r > 0:
        return f"{model_1} favored (distinguishable)"
    if r < 0:
        return f"{model_2} favored (distinguishable)"
    return "no preference (distinguishable tie)"


def compare_to_alternatives(
    fit: powerlaw.Fit,
    *,
    log_name: str,
    model_1: str,
    significance_level: float,
) -> pd.DataFrame:
    """Vuong LLR of power_law vs fixed alternative families on the same support."""
    rows: list[dict[str, Any]] = []
    for model_2, nested in _ALTERNATIVES:
        try:
            r_value, p_value = fit.distribution_compare(
                "power_law", model_2, nested=nested
            )
            r_f, p_f = float(r_value), float(p_value)
            if np.isfinite(r_f) and r_f > 0:
                preferred = model_1
            elif np.isfinite(r_f) and r_f < 0:
                preferred = model_2
            elif np.isfinite(r_f):
                preferred = "tie"
            else:
                preferred = "unavailable"
            rows.append(
                {
                    "log_name": log_name,
                    "model_1": model_1,
                    "model_2": model_2,
                    "R": r_f,
                    "p": p_f,
                    "preferred_model": preferred,
                    "interpretation": _interpret_llr(
                        r_f, p_f, model_1, model_2, significance_level
                    ),
                }
            )
        except Exception as exc:
            msg = str(exc).strip() or exc.__class__.__name__
            rows.append(
                {
                    "log_name": log_name,
                    "model_1": model_1,
                    "model_2": model_2,
                    "R": float("nan"),
                    "p": float("nan"),
                    "preferred_model": "unavailable",
                    "interpretation": f"error during comparison ({msg})",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decide and record (classification + CSV rows)
# ---------------------------------------------------------------------------


def _optional_float(value: Any) -> float | pd.NA:
    """Finite float or ``pd.NA`` for CSV cells."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return pd.NA
    return value if np.isfinite(value) else pd.NA


def _optional_int(value: Any) -> int | pd.NA:
    """Finite int or ``pd.NA`` for CSV cells."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    return int(value)


def _finite_preference_rows(comparison_df: pd.DataFrame | None) -> pd.DataFrame:
    """Finite (R, p) rows for alternatives that count in preference."""
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame(columns=["model_2", "R", "p"])
    rows = comparison_df.copy()
    r = pd.to_numeric(rows["R"], errors="coerce")
    p = pd.to_numeric(rows["p"], errors="coerce")
    models = rows["model_2"].astype(str)
    ok = (
        r.notna()
        & p.notna()
        & np.isfinite(r)
        & np.isfinite(p)
        & models.isin(CLASSIFICATION_ALTERNATIVES)
    )
    out = rows.loc[ok].copy()
    out["R"] = r.loc[ok]
    out["p"] = p.loc[ok]
    return out


def llr_preference(
    comparison_df: pd.DataFrame | None,
    *,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> str:
    """Step-3 label: ``alternative preferred``, ``PL preferred``, or ``alternatives inconclusive``.

    Truncated power law is ignored. Alternative preferred if any counted
    alternative has R<0 and p<significance. PL preferred only if every counted
    alternative has R>0 and p<significance.
    """
    valid = _finite_preference_rows(comparison_df)
    if valid.empty:
        return "alternatives inconclusive"
    if ((valid["R"] < 0) & (valid["p"] < significance_level)).any():
        return "alternative preferred"
    by_model: dict[str, tuple[float, float]] = {}
    for _, row in valid.iterrows():
        by_model[str(row["model_2"])] = (float(row["R"]), float(row["p"]))
    if all(
        name in by_model
        and by_model[name][0] > 0
        and by_model[name][1] < significance_level
        for name in CLASSIFICATION_ALTERNATIVES
    ):
        return "PL preferred"
    return "alternatives inconclusive"


def select_best_other_distribution(
    comparison_df: pd.DataFrame | None,
) -> str | None:
    """Most negative R among counted alternatives (excludes truncated power law)."""
    valid = _finite_preference_rows(comparison_df)
    if valid.empty:
        return None
    return str(valid.loc[valid["R"].astype(float).idxmin(), "model_2"])


def compact_classification(
    *,
    n_fitted_types: int,
    gof_p: float,
    comparison_df: pd.DataFrame | None,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    gof_p_threshold: float | None = None,
    comparison_p_threshold: float | None = None,
    fit_valid: bool = True,
) -> str:
    """Return an A–E code from ``CLASSIFICATION_LABELS``.

    ``significance_level`` is used for both GOF and LLR when the dedicated
    thresholds are omitted (existing callers). Compose may pass
    ``gof_p_threshold`` and ``comparison_p_threshold`` independently.
    """
    gof_thr = significance_level if gof_p_threshold is None else gof_p_threshold
    cmp_thr = (
        significance_level if comparison_p_threshold is None else comparison_p_threshold
    )
    if not fit_valid:
        return "A"
    if n_fitted_types < minimum_fitted_types:
        return "A"
    if not np.isfinite(gof_p):
        return "A"
    if gof_p < gof_thr:
        return "B"
    preference = llr_preference(comparison_df, significance_level=cmp_thr)
    if preference == "alternative preferred":
        return "C"
    if preference == "PL preferred":
        return "D"
    return "E"


def _classify(
    *,
    n_fitted_types: int,
    gof_p: float,
    comparison_df: pd.DataFrame,
    model_label: str,
    minimum_fitted_types: int,
    significance_level: float,
    fit_valid: bool,
    pathology_flags: Sequence[str],
) -> str:
    """Three-step classification: fitted size, GOF plausibility, LLR preference."""
    if not fit_valid:
        flags = ", ".join(pathology_flags) or "invalid fit diagnostics"
        return f"{model_label} invalid ({flags}); fit findings suppressed"

    fitted_step = (
        f"(1) >= {minimum_fitted_types} fitted types"
        if n_fitted_types >= minimum_fitted_types
        else f"(1) < {minimum_fitted_types} fitted types"
    )
    preference = llr_preference(
        comparison_df, significance_level=significance_level
    )

    if not np.isfinite(gof_p):
        return f"{fitted_step}; (2) {model_label} GOF unavailable; (3) {preference}"
    if gof_p < significance_level:
        gof_step = f"(2) {model_label} not plausible (p<{significance_level:g})"
    else:
        gof_step = f"(2) {model_label} plausible (p>={significance_level:g})"
    return f"{fitted_step}; {gof_step}; (3) {preference}"


def decide_and_record(
    *,
    log_name: str,
    input_path: str,
    model: str,
    descriptive: Mapping[str, Any],
    fit: Mapping[str, Any],
    gof: Mapping[str, Any],
    comparison_df: pd.DataFrame,
    minimum_fitted_types: int,
    significance_level: float,
) -> dict[str, Any]:
    """Merge fit+GOF, classify, and build gof/comparison/summary outputs."""
    fit_valid = bool(fit.get("fit_valid", True))
    # Single dict used for classification and for gof/summary CSV fields.
    evaluation = {
        **{k: fit[k] for k in fit if k != "fit"},
        "fit": fit["fit"],
        # Pathological fits skip GOF; keep placeholders so downstream stays uniform.
        "gof_p": gof["gof_p"] if fit_valid else np.nan,
        "n_success": gof["n_success"] if fit_valid else 0,
        "n_failed": gof["n_failed"] if fit_valid else 0,
        "empirical_d": gof.get("empirical_d", np.nan),
        "simulated_distances": (
            gof.get("simulated_distances")
            if fit_valid
            else np.asarray([], dtype=float)
        ),
        "gof_kind": _GOF_KIND_REFIT if fit_valid else _GOF_KIND_SKIPPED,
        "fit_valid": fit_valid,
    }

    n_fitted = evaluation.get("n_fitted_types")
    n_fitted_types = 0 if n_fitted is None or pd.isna(n_fitted) else int(n_fitted)

    # Stamp n_fitted_types and put identifying columns first for CSV export.
    comparison_df = comparison_df.copy()
    comparison_df["n_fitted_types"] = n_fitted_types
    comparison_df = comparison_df.loc[
        :,
        ["log_name", "n_fitted_types"]
        + [c for c in comparison_df.columns if c not in ("log_name", "n_fitted_types")],
    ]

    try:
        gof_p = float(evaluation.get("gof_p"))
    except (TypeError, ValueError):
        gof_p = float("nan")

    model_label = model.replace("_", " ")
    classification = _classify(
        n_fitted_types=n_fitted_types,
        gof_p=gof_p,
        comparison_df=comparison_df,
        model_label=model_label,
        minimum_fitted_types=minimum_fitted_types,
        significance_level=significance_level,
        fit_valid=fit_valid,
        pathology_flags=list(fit.get("pathology_flags") or []),
    )

    # Strongest counted alternative under Vuong R: most negative R (TPL excluded).
    best_other: str | None = None
    if fit_valid:
        best_other = select_best_other_distribution(comparison_df)

    # Detailed GOF metrics row (one per model run).
    gof_row: dict[str, Any] = {
        "log_name": log_name,
        "xmin": evaluation.get("xmin"),
        "xmax": evaluation.get("xmax"),
        "alpha": evaluation.get("alpha"),
        "KS_D": evaluation.get("KS_D"),
        "n_fitted_types": evaluation.get("n_fitted_types"),
        "fitted_type_share": evaluation.get("fitted_type_share"),
        "fitted_occurrence_share": evaluation.get("fitted_occurrence_share"),
        "log_range": evaluation.get("log_range"),
        "gof_p": evaluation.get("gof_p"),
        "n_success": evaluation.get("n_success"),
        "n_failed": evaluation.get("n_failed"),
        "alpha_at_boundary": evaluation.get("alpha_at_boundary"),
        "fit_valid": fit_valid,
        "gof_kind": evaluation["gof_kind"],
    }
    # Doubly-bounded only: record selected exclude-k / xmax.
    if "doubly_bounded_exclude_head_variants" in evaluation:
        gof_row["doubly_bounded_exclude_head_variants"] = evaluation["doubly_bounded_exclude_head_variants"]
        gof_row["selected"] = True
    if "actual_excluded_variants" in evaluation:
        gof_row["actual_excluded_variants"] = evaluation["actual_excluded_variants"]

    # Compact decision row for the summary CSV (_optional_* maps missing to pd.NA).
    summary: dict[str, Any] = {
        "log_name": log_name,
        "input_path": str(input_path),
        "n_occurrences": int(descriptive["n_occurrences"]),
        "n_types": int(descriptive["n_types"]),
        "n_singletons": int(descriptive["n_singletons"]),
        "singleton_share": float(descriptive["singleton_share"]),
        "xmin": _optional_float(evaluation.get("xmin")),
        "xmax": _optional_float(evaluation.get("xmax")),
        "n_fitted_types": _optional_int(evaluation.get("n_fitted_types")),
        "fitted_type_share": _optional_float(evaluation.get("fitted_type_share")),
        "fitted_occurrence_share": _optional_float(
            evaluation.get("fitted_occurrence_share")
        ),
        "log_range": _optional_float(evaluation.get("log_range")),
        "fit_valid": fit_valid,
        "classification": classification,
        "best_other_distribution": pd.NA,
    }
    if fit_valid:
        summary["alpha"] = _optional_float(evaluation.get("alpha"))
        summary["KS_D"] = _optional_float(evaluation.get("KS_D"))
        summary["gof_p"] = _optional_float(evaluation.get("gof_p"))
        summary["n_success"] = _optional_int(evaluation.get("n_success"))
        summary["n_failed"] = _optional_int(evaluation.get("n_failed"))
        summary["gof_kind"] = evaluation.get("gof_kind", pd.NA)
        summary["best_other_distribution"] = (
            pd.NA if best_other is None else best_other
        )
    else:
        # Invalid fit: blank inference fields so CSV does not look like a result.
        summary["alpha"] = pd.NA
        summary["KS_D"] = pd.NA
        summary["gof_p"] = pd.NA
        summary["n_success"] = pd.NA
        summary["n_failed"] = pd.NA
        summary["gof_kind"] = _GOF_KIND_SKIPPED
    if "doubly_bounded_exclude_head_variants" in evaluation:
        summary["doubly_bounded_exclude_head_variants"] = _optional_int(
            evaluation.get("doubly_bounded_exclude_head_variants")
        )
    if "actual_excluded_variants" in evaluation:
        summary["actual_excluded_variants"] = _optional_int(
            evaluation.get("actual_excluded_variants")
        )

    return {
        "gof_rows": [gof_row],
        "comparison_df": comparison_df,
        "summary_row": summary,
        "evaluation": evaluation,
    }


def empty_clauset_result(
    *,
    log_name: str,
    input_path: str,
    model: str,
    classification: str,
) -> dict[str, Any]:
    """Placeholder outputs when a run fails before fitting."""
    comparison_rows = [
        {
            "log_name": log_name,
            "n_fitted_types": pd.NA,
            "model_1": model,
            "model_2": m2,
            "R": pd.NA,
            "p": pd.NA,
            "preferred_model": pd.NA,
            "interpretation": pd.NA,
        }
        for m2, _ in _ALTERNATIVES
    ]
    summary = {
        "log_name": log_name,
        "input_path": str(input_path),
        "n_occurrences": pd.NA,
        "n_types": pd.NA,
        "n_singletons": pd.NA,
        "singleton_share": pd.NA,
        "alpha": pd.NA,
        "xmin": pd.NA,
        "xmax": pd.NA,
        "KS_D": pd.NA,
        "n_fitted_types": pd.NA,
        "fitted_type_share": pd.NA,
        "fitted_occurrence_share": pd.NA,
        "log_range": pd.NA,
        "gof_p": pd.NA,
        "n_success": pd.NA,
        "n_failed": pd.NA,
        "gof_kind": pd.NA,
        "fit_valid": pd.NA,
        "classification": classification,
        "best_other_distribution": pd.NA,
    }
    if model == DOUBLY_BOUNDED_POWER_LAW:
        summary["doubly_bounded_exclude_head_variants"] = pd.NA
        summary["actual_excluded_variants"] = pd.NA
    return {
        "gof_rows": [],
        "comparison_df": pd.DataFrame(comparison_rows),
        "summary_row": summary,
    }


# ---------------------------------------------------------------------------
# Public orchestration
# ---------------------------------------------------------------------------


def run_clauset_pipeline(
    observations: np.ndarray | Sequence[Any],
    *,
    model: str,
    log_name: str,
    input_path: str,
    discrete: bool = True,
    n_bootstraps: int = 1000,
    random_seed: int = 42,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    compare_alternatives: bool = True,
    doubly_bounded_exclude_head_variants: Sequence[int] = (
        DEFAULT_DOUBLY_BOUNDED_EXCLUDE_HEAD_VARIANTS
    ),
) -> dict[str, Any]:
    """Run Clauset analysis for **one** power-law model.

    To analyse several models, call this function once per model.

    ``doubly_bounded_exclude_head_variants`` is only used for the
    doubly-bounded model (xmax candidates); other models ignore it.
    Set ``compare_alternatives=False`` to skip Vuong/LLR comparisons (fit + GOF only).
    """
    if model not in DISTRIBUTION_NAMES:
        raise ValueError(
            f"unknown model {model!r}; expected one of {list(DISTRIBUTION_NAMES)}"
        )

    data = _validate_observations(observations, discrete=discrete)
    if data.size < 2:
        raise ValueError("not enough observations to fit a power law")

    descriptive = _descriptive_stats(data)
    fit = _powerlaw_fit_with_stats_for_model(
        data, model, doubly_bounded_exclude_head_variants=doubly_bounded_exclude_head_variants
    )
    if fit is None:
        return empty_clauset_result(
            log_name=log_name,
            input_path=input_path,
            model=model,
            classification="no eligible doubly bounded interval",
        )

    if not bool(fit.get("fit_valid", True)):
        gof = _failed_gof(
            n_bootstraps=0,
            xmin=float(fit["xmin"]),
            xmax=float(fit["xmax"]) if np.isfinite(fit["xmax"]) else float("nan"),
            alpha=float(fit["alpha"]),
            empirical_d=float(fit["KS_D"]),
        )
        comparison_df = empty_clauset_result(
            log_name=log_name, input_path=input_path, model=model, classification=""
        )["comparison_df"]
    else:
        gof = bootstrap_gof(
            data,
            fit,
            model,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
            doubly_bounded_exclude_head_variants=doubly_bounded_exclude_head_variants,
        )
        if compare_alternatives:
            comparison_df = compare_to_alternatives(
                fit["fit"],
                log_name=log_name,
                model_1=model,
                significance_level=significance_level,
            )
        else:
            comparison_df = empty_clauset_result(
                log_name=log_name, input_path=input_path, model=model, classification=""
            )["comparison_df"]

    return decide_and_record(
        log_name=log_name,
        input_path=input_path,
        model=model,
        descriptive=descriptive,
        fit=fit,
        gof=gof,
        comparison_df=comparison_df,
        minimum_fitted_types=minimum_fitted_types,
        significance_level=significance_level,
    )


def accumulate_and_write_csvs(
    *,
    analysis_dir: Path,
    accumulated: dict[str, dict[str, list]],
    dataset_results: dict[str, dict[str, Any]],
    file_suffix: str = "",
) -> None:
    """Fold one log/unit into ``accumulated`` and rewrite result CSVs.

    For each model in ``accumulated``:

    1. Append that model's rows from ``dataset_results`` (as returned by
       ``run_clauset_pipeline``) into the in-memory lists.
    2. Write sorted ``gof{suffix}.csv``, ``comparison{suffix}.csv``, and
       ``summary{suffix}.csv`` under ``analysis_dir/<model>/``.

    Call after each unit so a crash mid-batch still leaves partial CSVs.
    Mutates ``accumulated`` in place.
    """
    for name in accumulated:
        dist_dir = analysis_dir / name
        dist_dir.mkdir(parents=True, exist_ok=True)

        accumulated[name]["gof_rows"].extend(dataset_results[name]["gof_rows"])
        accumulated[name]["comparison_frames"].append(
            dataset_results[name]["comparison_df"]
        )
        accumulated[name]["summary_rows"].append(dataset_results[name]["summary_row"])

        gof_df = pd.DataFrame(accumulated[name]["gof_rows"])
        if not gof_df.empty:
            sort_cols = ["log_name"]
            if "doubly_bounded_exclude_head_variants" in gof_df.columns:
                sort_cols.append("doubly_bounded_exclude_head_variants")
            gof_df = gof_df.sort_values(sort_cols).reset_index(drop=True)

        comparison_df = (
            pd.concat(accumulated[name]["comparison_frames"], ignore_index=True)
            .sort_values(["log_name", "model_2"])
            .reset_index(drop=True)
        )
        summary_df = (
            pd.DataFrame(accumulated[name]["summary_rows"])
            .sort_values("log_name")
            .reset_index(drop=True)
        )

        gof_df.to_csv(dist_dir / f"gof{file_suffix}.csv", index=False)
        comparison_df.to_csv(dist_dir / f"comparison{file_suffix}.csv", index=False)
        summary_df.to_csv(dist_dir / f"summary{file_suffix}.csv", index=False)
