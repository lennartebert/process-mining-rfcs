"""Synthetic Zipf E2E (attachments + step 03) and GOF-implementation Monte Carlo.

The slow GOF-implementation test (Type I error, alpha bias) is skipped unless
RUN_PL_GOF_TEST=1.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import TEST_ATTACHMENTS_DIR, TEST_N_GRAMS_DIR, TEST_RESULTS_DIR
from utils.io.attachments import (
    REQUIRED_ATTACHMENT_COLUMNS,
    load_attachments,
    save_attachments,
)
from utils.powerlaw import (
    DEFAULT_SIGNIFICANCE_LEVEL,
    FULL_RANGE_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    run_clauset_pipeline,
)
from utils.rfc import extract_frequency_counts

# Seed 42 yields ~9.5k rows (not clipped). Variants only: same Zipf vector, all three models.
LOG_NAME = "TEST"
N_TYPES = 1000
ALPHA_TRUE = 2.0
XMIN_TRUE = 1
SEED = 42
N_BOOTSTRAPS_E2E = 1000
ATTACHMENT_TIME = "1970-01-01T00:00:00"

ATTACHMENTS_ROOT = REPO_ROOT / TEST_ATTACHMENTS_DIR
N_GRAMS_ROOT = REPO_ROOT / TEST_N_GRAMS_DIR
TRUTH_PATH = REPO_ROOT / TEST_RESULTS_DIR / "synthetic_truth.json"
CONCEPTS = ["variants"]


def sample_type_frequencies(n_types: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Independent Zipf draws: one frequency per type. No clipping."""
    return np.asarray(rng.zipf(float(alpha), size=int(n_types)), dtype=int)


def frequencies_to_attachments(frequencies: np.ndarray) -> pd.DataFrame:
    """Expand type frequencies into production attachment rows."""
    node_ids: list[str] = []
    for i, freq in enumerate(frequencies, start=1):
        node_ids.extend([f"type_{i:04d}"] * int(freq))
    n = len(node_ids)
    return pd.DataFrame(
        {
            "attachment_time": [ATTACHMENT_TIME] * n,
            "attachment_index": np.arange(n, dtype=int),
            "node_id": node_ids,
        }
    )


def _attachments_path(concept: str) -> Path:
    return ATTACHMENTS_ROOT / concept / LOG_NAME / "attachments.csv.gz"


def _gof_reps_path(output_dir: Path, n_types: int) -> Path:
    return output_dir / f"gof_reps_types{int(n_types)}.csv"


def _gof_summary_path(output_dir: Path, n_types: int) -> Path:
    return output_dir / f"gof_summary_types{int(n_types)}.csv"


def _gof_combined_summary_path(output_dir: Path) -> Path:
    return output_dir / "gof_summary.csv"


def _recombine_gof_implementation_summary(output_dir: Path) -> Path:
    summary_paths = sorted(output_dir.glob("gof_summary_types*.csv"))
    rows: list[pd.DataFrame] = []
    for path in summary_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "types" not in df.columns:
            continue
        rows.append(df)
    if not rows:
        return _gof_combined_summary_path(output_dir)
    combined = pd.concat(rows, ignore_index=True)
    combined["types"] = pd.to_numeric(combined["types"], errors="coerce")
    combined = combined[np.isfinite(combined["types"])].copy()
    combined["types"] = combined["types"].astype(int)
    combined = combined.sort_values("types").reset_index(drop=True)
    out = _gof_combined_summary_path(output_dir)
    combined.to_csv(out, index=False)
    return out


def _load_or_write_attachments() -> np.ndarray:
    """Reuse existing variant attachments, or generate Zipf frequencies and save."""
    path = _attachments_path("variants")
    if path.exists():
        df = load_attachments(path)
        missing = [col for col in REQUIRED_ATTACHMENT_COLUMNS if col not in df.columns]
        assert missing == [], f"missing columns in {path}"
        frequencies = extract_frequency_counts(df)
        assert np.all(frequencies >= 1)
        return np.asarray(frequencies, dtype=int)

    frequencies = sample_type_frequencies(N_TYPES, ALPHA_TRUE, np.random.default_rng(SEED))
    assert np.all(frequencies >= 1)
    assert np.issubdtype(frequencies.dtype, np.integer)
    attachments = frequencies_to_attachments(frequencies)
    for concept in CONCEPTS:
        out_path = _attachments_path(concept)
        save_attachments(attachments, out_path)
        df = load_attachments(out_path)
        missing = [col for col in REQUIRED_ATTACHMENT_COLUMNS if col not in df.columns]
        assert missing == [], f"missing columns in {out_path}"
        assert int(df["node_id"].nunique()) == int(len(frequencies))
        assert len(df) == int(frequencies.sum())
        recovered = extract_frequency_counts(df)
        np.testing.assert_array_equal(np.sort(recovered), np.sort(frequencies))
    return frequencies


def run_synthetic_e2e(*, n_bootstraps: int = N_BOOTSTRAPS_E2E) -> None:
    """Load or generate Zipf attachments, run production step 03, and sanity-check."""
    frequencies = _load_or_write_attachments()
    n_types = int(len(frequencies))
    truth = {
        "log_name": LOG_NAME,
        "alpha_true": ALPHA_TRUE,
        "xmin_true": XMIN_TRUE,
        "xmax_true": None,
        "types": n_types,
        "seed": SEED,
        "n_rows": int(frequencies.sum()),
        "clipped": False,
        "concepts": list(CONCEPTS),
        "n_bootstraps": int(n_bootstraps),
    }
    TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRUTH_PATH.write_text(json.dumps(truth, indent=2) + "\n")

    step_path = REPO_ROOT / "analyses" / "n_grams" / "03_clauset_tests.py"
    spec = importlib.util.spec_from_file_location(step_path.stem, step_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {step_path}")
    clauset_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clauset_tests)
    clauset_tests.main(
        [
            "--datasets",
            LOG_NAME,
            "--concepts",
            *CONCEPTS,
            "--attachments-dir",
            str(ATTACHMENTS_ROOT),
            "--output-dir",
            str(N_GRAMS_ROOT),
            "--n-bootstraps",
            str(int(n_bootstraps)),
            "--random-seed",
            str(SEED),
        ]
    )

    rows: list[dict] = []
    for concept in CONCEPTS:
        for model in clauset_tests.models_for_concept(concept):
            summary_path = N_GRAMS_ROOT / concept / model / "summary.csv"
            assert summary_path.exists(), f"missing {summary_path}"
            summary = pd.read_csv(summary_path)
            assert len(summary) == 1
            src = summary.iloc[0]
            assert str(src["log_name"]) == LOG_NAME
            assert bool(src["fit_valid"])
            assert int(src["n_types"]) == n_types
            alpha = float(src["alpha"])
            gof_p = float(src["gof_p"])
            assert np.isfinite(alpha)
            assert np.isfinite(float(src["KS_D"]))
            assert np.isfinite(gof_p) and 0.0 <= gof_p <= 1.0
            assert abs(alpha - ALPHA_TRUE) <= 0.5, (
                f"{model} alpha {alpha:.3f} diverges more than 0.5 from {ALPHA_TRUE}"
            )

            comparison = pd.read_csv(N_GRAMS_ROOT / concept / model / "comparison.csv")
            p = pd.to_numeric(comparison["p"], errors="coerce")
            finite_p = p[np.isfinite(p)]
            if not finite_p.empty:
                assert ((finite_p >= 0.0) & (finite_p <= 1.0)).all()

            if concept == "variants" and model == FULL_RANGE_POWER_LAW:
                assert int(src["n_fitted_types"]) == n_types
                assert int(src["xmin"]) == XMIN_TRUE
                assert pd.isna(src["xmax"])

            rows.append(
                {
                    "concept": concept,
                    "model": model,
                    "alpha_true": truth["alpha_true"],
                    "xmin_true": truth["xmin_true"],
                    "types": truth["types"],
                    "seed": truth["seed"],
                    "n_bootstraps": int(n_bootstraps),
                    "alpha": src.get("alpha"),
                    "xmin": src.get("xmin"),
                    "xmax": src.get("xmax"),
                    "n_types": src.get("n_types"),
                    "n_fitted_types": src.get("n_fitted_types"),
                    "gof_p": src.get("gof_p"),
                    "KS_D": src.get("KS_D"),
                    "classification": src.get("classification"),
                    "best_other_distribution": src.get("best_other_distribution"),
                }
            )

    N_GRAMS_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(N_GRAMS_ROOT / "synthetic_validation.csv", index=False)


class SyntheticFullRangeE2ETests(unittest.TestCase):
    def test_extracted_counts_match_generated_frequencies_then_pipeline(self) -> None:
        run_synthetic_e2e(n_bootstraps=N_BOOTSTRAPS_E2E)


def run_gof_implementation_test(
    *,
    n_reps: int = 200,
    n_bootstraps: int = 1000,
    n_types: int = N_TYPES,
    alpha_true: float = ALPHA_TRUE,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    output_dir: Path | None = None,
) -> dict:
    """Monte Carlo Type I test of lower-bounded Clauset GOF on true Zipf data.

    Each replicate draws ``n_types`` Zipf(alpha_true) frequencies and runs
    production lower-bounded Clauset GOF with no alternative-model comparisons.

    Checks (nominal significance 0.10 unless overridden):
    1. Type I error: P(gof_p < alpha) about 0.10 (binomial ±2 SE band).
    2. Alpha bias: |mean(alpha_hat) - alpha_true| <= 0.2.

    Writes gof_reps_types{N}.csv and gof_summary_types{N}.csv, and recombines
    all per-types summaries into gof_summary.csv.
    """
    output_dir = Path(output_dir) if output_dir else REPO_ROOT / TEST_RESULTS_DIR / "gof_implementation"
    output_dir.mkdir(parents=True, exist_ok=True)

    gof_rows: list[dict] = []
    n_reject = 0
    n_failed = 0
    alphas: list[float] = []
    xmins: list[float] = []
    for i in range(n_reps):
        frequencies = sample_type_frequencies(n_types, alpha_true, np.random.default_rng(i))
        out = run_clauset_pipeline(
            frequencies,
            model=LOWER_BOUNDED_POWER_LAW,
            log_name=LOG_NAME,
            input_path="synthetic-frequencies",
            discrete=True,
            n_bootstraps=n_bootstraps,
            random_seed=i,
            compare_alternatives=False,
        )
        summary = out["summary_row"]
        try:
            alpha_f = float(summary.get("alpha"))
            gof_f = float(summary.get("gof_p"))
            xmin_f = float(summary.get("xmin"))
        except (TypeError, ValueError):
            alpha_f, gof_f, xmin_f = float("nan"), float("nan"), float("nan")
        if not (np.isfinite(alpha_f) and np.isfinite(gof_f)):
            n_failed += 1
            continue
        alphas.append(alpha_f)
        if np.isfinite(xmin_f):
            xmins.append(xmin_f)
        rejected = gof_f < significance_level
        n_reject += int(rejected)
        gof_rows.append(
            {
                "seed": i,
                "alpha": alpha_f,
                "xmin": xmin_f,
                "gof_p": gof_f,
                "rejected": rejected,
            }
        )

    n_ok = n_reps - n_failed
    reject_rate = n_reject / n_ok if n_ok else float("nan")
    mean_alpha = float(np.mean(alphas)) if alphas else float("nan")
    mean_xmin = float(np.mean(xmins)) if xmins else float("nan")
    se = (
        float(np.sqrt(significance_level * (1.0 - significance_level) / n_ok))
        if n_ok
        else float("nan")
    )
    lo, hi = significance_level - 2.0 * se, significance_level + 2.0 * se

    checks = {
        "n_reps": n_reps,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "reject_rate": reject_rate,
        "reject_lo_2SE": lo,
        "reject_hi_2SE": hi,
        "mean_alpha": mean_alpha,
        "mean_xmin": mean_xmin,
        "types": int(n_types),
    }
    pd.DataFrame(gof_rows).to_csv(_gof_reps_path(output_dir, n_types), index=False)
    pd.DataFrame([checks]).to_csv(_gof_summary_path(output_dir, n_types), index=False)
    _recombine_gof_implementation_summary(output_dir)

    failures: list[str] = []
    if n_ok < max(20, n_reps // 2):
        failures.append(f"too many failed fits: {n_failed}/{n_reps}")
    if not (lo <= reject_rate <= hi):
        failures.append(
            f"GOF rejection rate {reject_rate:.3f} outside [{lo:.3f}, {hi:.3f}]"
        )
    if abs(mean_alpha - alpha_true) > 0.2:
        failures.append(f"mean alpha {mean_alpha:.3f} far from {alpha_true}")
    checks["failures"] = failures
    checks["ok"] = not failures
    return checks


@unittest.skipUnless(
    os.environ.get("RUN_PL_GOF_TEST") == "1",
    "set RUN_PL_GOF_TEST=1 to run the slow GOF-implementation test",
)
class SyntheticPLGofImplementationTests(unittest.TestCase):
    def test_gof_on_true_zipf_lower_bounded(self) -> None:
        result = run_gof_implementation_test()
        self.assertTrue(result["ok"], msg="; ".join(result["failures"]))


def _e2e_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic Zipf E2E (attachments + step 03)"
    )
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS_E2E)
    args = parser.parse_args(argv)
    run_synthetic_e2e(n_bootstraps=args.n_bootstraps)
    return 0


def _gof_implementation_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Type I test of lower-bounded Clauset GOF (true Zipf)"
    )
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--n-bootstraps", type=int, default=1000)
    parser.add_argument("--types", type=int, nargs="+", default=[N_TYPES])
    parser.add_argument("--alpha-true", type=float, default=ALPHA_TRUE)
    parser.add_argument("--significance-level", type=float, default=DEFAULT_SIGNIFICANCE_LEVEL)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip type counts with existing per-types outputs.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force recomputation even when per-types outputs already exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / TEST_RESULTS_DIR / "gof_implementation"),
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    type_counts = list(dict.fromkeys(args.types))
    combined_failures: list[str] = []
    ran_any = False
    for n_types in type_counts:
        summary_path = _gof_summary_path(output_dir, n_types)
        reps_path = _gof_reps_path(output_dir, n_types)
        if args.skip_existing and summary_path.exists() and reps_path.exists():
            print(f"Skipping types={n_types}: found {summary_path.name} and {reps_path.name}")
            continue
        ran_any = True
        result = run_gof_implementation_test(
            n_reps=args.n_reps,
            n_bootstraps=args.n_bootstraps,
            n_types=n_types,
            alpha_true=args.alpha_true,
            significance_level=args.significance_level,
            output_dir=output_dir,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "failures"}, indent=2, default=str))
        if result["failures"]:
            combined_failures.append(f"types={n_types}: " + "; ".join(result["failures"]))

    summary_out = _recombine_gof_implementation_summary(output_dir)
    print(f"Combined summary written to: {summary_out}")
    if not ran_any:
        print("No type counts recomputed.")
    if combined_failures:
        print("FAILURES:", " | ".join(combined_failures))
        return 1
    return 0


if __name__ == "__main__":
    if "--gof-test" in sys.argv:
        sys.argv.remove("--gof-test")
        raise SystemExit(_gof_implementation_cli(sys.argv[1:]))
    if "--e2e-test" in sys.argv:
        sys.argv.remove("--e2e-test")
        raise SystemExit(_e2e_cli(sys.argv[1:]))
    unittest.main()
