"""Synthetic full-range power-law E2E: generate attachments, run step 03, sanity-check.

The slow GOF-implementation test (Type I error, alpha bias, precision, ranking)
is skipped unless RUN_PL_GOF_TEST=1.
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
    run_clauset_pipeline,
)
from utils.powerlaw.analyze import _powerlaw_fit_with_stats
from utils.rfc import extract_frequency_counts

# Seed 42 yields ~9.5k rows (not clipped). Variants only: same Zipf vector, all three models.
LOG_NAME = "TEST"
K = 1000
ALPHA_TRUE = 2.0
XMIN_TRUE = 1
SEED = 42
N_BOOTSTRAPS_E2E = 1000
ALPHA_TOLERANCE = 0.5
ATTACHMENT_TIME = "1970-01-01T00:00:00"

ATTACHMENTS_ROOT = REPO_ROOT / TEST_ATTACHMENTS_DIR
N_GRAMS_ROOT = REPO_ROOT / TEST_N_GRAMS_DIR
TRUTH_PATH = REPO_ROOT / TEST_RESULTS_DIR / "synthetic_truth.json"
CONCEPTS = ["variants"]


def sample_type_frequencies(k: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Independent Zipf draws: one frequency per type. No clipping."""
    return np.asarray(rng.zipf(float(alpha), size=int(k)), dtype=int)


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


class SyntheticFullRangeE2ETests(unittest.TestCase):
    def test_extracted_counts_match_generated_frequencies_then_pipeline(self) -> None:
        frequencies = sample_type_frequencies(K, ALPHA_TRUE, np.random.default_rng(SEED))
        self.assertTrue(np.all(frequencies >= 1))
        self.assertTrue(np.issubdtype(frequencies.dtype, np.integer))
        truth = {
            "log_name": LOG_NAME,
            "alpha_true": ALPHA_TRUE,
            "xmin_true": XMIN_TRUE,
            "xmax_true": None,
            "K": int(K),
            "seed": SEED,
            "n_rows": int(frequencies.sum()),
            "clipped": False,
            "concepts": list(CONCEPTS),
        }
        TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRUTH_PATH.write_text(json.dumps(truth, indent=2) + "\n")
        attachments = frequencies_to_attachments(frequencies)

        # Save → load → extract_frequency_counts must recover the generated frequencies.
        for concept in CONCEPTS:
            path = ATTACHMENTS_ROOT / concept / LOG_NAME / "attachments.csv.gz"
            save_attachments(attachments, path)
            df = load_attachments(path)
            missing = [col for col in REQUIRED_ATTACHMENT_COLUMNS if col not in df.columns]
            self.assertEqual(missing, [], f"missing columns in {path}")
            self.assertEqual(int(df["node_id"].nunique()), int(len(frequencies)))
            self.assertEqual(len(df), int(frequencies.sum()))
            recovered = extract_frequency_counts(df)
            np.testing.assert_array_equal(np.sort(recovered), np.sort(frequencies))

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
                str(N_BOOTSTRAPS_E2E),
                "--random-seed",
                str(SEED),
            ]
        )

        rows: list[dict] = []
        for concept in CONCEPTS:
            for model in clauset_tests.models_for_concept(concept):
                summary_path = N_GRAMS_ROOT / concept / model / "summary.csv"
                self.assertTrue(summary_path.exists(), f"missing {summary_path}")
                summary = pd.read_csv(summary_path)
                self.assertEqual(len(summary), 1)
                src = summary.iloc[0]
                self.assertEqual(str(src["log_name"]), LOG_NAME)
                self.assertTrue(bool(src["fit_valid"]))
                self.assertEqual(int(src["n_types"]), K)
                alpha = float(src["alpha"])
                gof_p = float(src["gof_p"])
                self.assertTrue(np.isfinite(alpha))
                self.assertTrue(np.isfinite(float(src["KS_D"])))
                self.assertTrue(np.isfinite(gof_p) and 0.0 <= gof_p <= 1.0)

                comparison = pd.read_csv(N_GRAMS_ROOT / concept / model / "comparison.csv")
                p = pd.to_numeric(comparison["p"], errors="coerce")
                finite_p = p[np.isfinite(p)]
                if not finite_p.empty:
                    self.assertTrue(((finite_p >= 0.0) & (finite_p <= 1.0)).all())

                if concept == "variants" and model == FULL_RANGE_POWER_LAW:
                    self.assertEqual(int(src["n_fitted_types"]), K)
                    self.assertEqual(int(src["xmin"]), XMIN_TRUE)
                    self.assertTrue(pd.isna(src["xmax"]))
                    self.assertLess(abs(alpha - ALPHA_TRUE), ALPHA_TOLERANCE)

                rows.append(
                    {
                        "concept": concept,
                        "model": model,
                        "alpha_true": truth["alpha_true"],
                        "xmin_true": truth["xmin_true"],
                        "K": truth["K"],
                        "seed": truth["seed"],
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


def run_gof_implementation_test(
    *,
    n_reps: int = 200,
    n_bootstraps: int = 1000,
    k: int = K,
    k_small: int = 200,
    alpha_true: float = ALPHA_TRUE,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    output_dir: Path | None = None,
) -> dict:
    """Monte Carlo test of the Clauset GOF implementation on true full-range PL data.

    Each replicate draws K type frequencies from Zipf(alpha_true) and runs the
    production full-range Clauset pipeline on that vector (no attachment expansion).
    A single E2E seed cannot assert GOF p-values or model ranking; this checks
    the *distribution* of those stochastic outcomes.

    Checks (nominal significance 0.10 unless overridden):
    1. Type I error: P(gof_p < alpha) should be about 0.10 (binomial 3-SE band).
    2. Alpha bias: mean(alpha_hat) should stay close to alpha_true.
    3. Precision: std(alpha_hat) at large K should be smaller than at k_small
       (MLE only; GOF is skipped for this sweep).
    4. Ranking: "alternatives preferred" / Other better should not dominate.

    Writes gof_reps.csv and gof_implementation_summary.csv under output_dir.
    Returns a dict with rates, bands, and a ``failures`` list (empty iff ok).
    """
    output_dir = Path(output_dir) if output_dir else REPO_ROOT / TEST_RESULTS_DIR / "gof_implementation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- GOF + LLR over n_reps independent Zipf samples ---
    gof_rows: list[dict] = []
    n_reject = 0
    n_other_better = 0
    n_failed = 0
    alphas: list[float] = []
    for i in range(n_reps):
        frequencies = sample_type_frequencies(k, alpha_true, np.random.default_rng(i))
        out = run_clauset_pipeline(
            frequencies,
            model=FULL_RANGE_POWER_LAW,
            log_name=LOG_NAME,
            input_path="synthetic-frequencies",
            discrete=True,
            n_bootstraps=n_bootstraps,
            random_seed=i,
        )
        summary = out["summary_row"]
        alpha = summary.get("alpha")
        gof_p = summary.get("gof_p")
        classification = str(summary.get("classification") or "")
        try:
            alpha_f = float(alpha)
            gof_f = float(gof_p)
        except (TypeError, ValueError):
            alpha_f, gof_f = float("nan"), float("nan")
        if not (np.isfinite(alpha_f) and np.isfinite(gof_f)):
            n_failed += 1
            continue
        alphas.append(alpha_f)
        # Clauset GOF: reject the PL null when p is below the nominal level.
        rejected = gof_f < significance_level
        n_reject += int(rejected)
        # Production classification step (3): Vuong LLR prefers an alternative.
        other_better = "alternatives preferred" in classification.lower()
        n_other_better += int(other_better)
        gof_rows.append(
            {
                "seed": i,
                "alpha": alpha_f,
                "gof_p": gof_f,
                "rejected": rejected,
                "other_better": other_better,
                "classification": classification,
            }
        )

    n_ok = n_reps - n_failed
    reject_rate = n_reject / n_ok if n_ok else float("nan")
    other_rate = n_other_better / n_ok if n_ok else float("nan")
    mean_alpha = float(np.mean(alphas)) if alphas else float("nan")
    # Null rejection rate ~ Binomial(n_ok, significance_level); 3-SE Monte Carlo band.
    se = float(np.sqrt(significance_level * (1.0 - significance_level) / n_ok)) if n_ok else float("nan")
    lo, hi = significance_level - 3.0 * se, significance_level + 3.0 * se

    # --- MLE precision vs sample size (no bootstrap GOF) ---
    small_alphas = []
    large_alphas = []
    for i in range(n_reps):
        small = sample_type_frequencies(k_small, alpha_true, np.random.default_rng(10_000 + i))
        large = sample_type_frequencies(k, alpha_true, np.random.default_rng(20_000 + i))
        small_alphas.append(float(_powerlaw_fit_with_stats(small, xmin=1, xmax=None)["alpha"]))
        large_alphas.append(float(_powerlaw_fit_with_stats(large, xmin=1, xmax=None)["alpha"]))
    std_small = float(np.nanstd(small_alphas))
    std_large = float(np.nanstd(large_alphas))

    checks = {
        "n_reps": n_reps,
        "n_ok": n_ok,
        "n_failed": n_failed,
        "reject_rate": reject_rate,
        "reject_lo": lo,
        "reject_hi": hi,
        "mean_alpha": mean_alpha,
        "alpha_true": alpha_true,
        "other_better_rate": other_rate,
        "std_alpha_k_small": std_small,
        "std_alpha_k": std_large,
        "k": k,
        "k_small": k_small,
    }
    pd.DataFrame(gof_rows).to_csv(output_dir / "gof_reps.csv", index=False)
    pd.DataFrame([checks]).to_csv(output_dir / "gof_implementation_summary.csv", index=False)

    # Fail if GOF Type I error, alpha, precision, or ranking look wrong on true PL data.
    failures: list[str] = []
    if n_ok < max(20, n_reps // 2):
        failures.append(f"too many failed fits: {n_failed}/{n_reps}")
    if not (lo <= reject_rate <= hi):
        failures.append(
            f"GOF rejection rate {reject_rate:.3f} outside [{lo:.3f}, {hi:.3f}]"
        )
    if abs(mean_alpha - alpha_true) > 0.2:
        failures.append(f"mean alpha {mean_alpha:.3f} far from {alpha_true}")
    if not (std_large < std_small):
        failures.append(f"precision did not improve: std(K={k})={std_large:.4f} vs std(K={k_small})={std_small:.4f}")
    if other_rate > 0.25:
        failures.append(f"Other better dominates: rate={other_rate:.3f}")
    checks["failures"] = failures
    checks["ok"] = not failures
    return checks


@unittest.skipUnless(
    os.environ.get("RUN_PL_GOF_TEST") == "1",
    "set RUN_PL_GOF_TEST=1 to run the slow GOF-implementation test",
)
class SyntheticPLGofImplementationTests(unittest.TestCase):
    def test_gof_on_true_full_range_pl(self) -> None:
        result = run_gof_implementation_test()
        self.assertTrue(result["ok"], msg="; ".join(result["failures"]))


def _gof_implementation_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monte Carlo test of the Clauset GOF implementation (true full-range PL)"
    )
    parser.add_argument("--n-reps", type=int, default=200)
    parser.add_argument("--n-bootstraps", type=int, default=200)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--k-small", type=int, default=200)
    parser.add_argument("--alpha-true", type=float, default=ALPHA_TRUE)
    parser.add_argument("--significance-level", type=float, default=DEFAULT_SIGNIFICANCE_LEVEL)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / TEST_RESULTS_DIR / "gof_implementation"),
    )
    args = parser.parse_args(argv)
    result = run_gof_implementation_test(
        n_reps=args.n_reps,
        n_bootstraps=args.n_bootstraps,
        k=args.k,
        k_small=args.k_small,
        alpha_true=args.alpha_true,
        significance_level=args.significance_level,
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}, indent=2, default=str))
    if result["failures"]:
        print("FAILURES:", "; ".join(result["failures"]))
        return 1
    return 0


if __name__ == "__main__":
    if "--gof-test" in sys.argv:
        sys.argv.remove("--gof-test")
        raise SystemExit(_gof_implementation_cli(sys.argv[1:]))
    unittest.main()
