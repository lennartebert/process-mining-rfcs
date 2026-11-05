"""Step 03: Clauset power-law tests on n-gram/variant attachments.

Runs ``cli.clauset_power_law`` once per concept (variants, n1..n20 by default),
reading attachments from ``results/attachments/`` and writing under
``results/powerlaw_statistics/real/<concept>/<model>/``.

Model policy:
- ``variants``: full_range, lower_bounded, and doubly_bounded
- other concepts (n-grams, …): lower_bounded only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli import clauset_power_law
from utils.constants import ATTACHMENTS_DIR, POWERLAW_STATISTICS_REAL_DIR
from utils.io.attachments import NGRAM_CONCEPTS
from utils.powerlaw import DISTRIBUTION_NAMES, LOWER_BOUNDED_POWER_LAW
from utils.timing import format_duration, timing_print

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]
VARIANT_MODELS = list(DISTRIBUTION_NAMES)
NGRAM_MODELS = [LOWER_BOUNDED_POWER_LAW]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clauset power-law evaluation for variants and n-grams "
            "(wraps cli/clauset_power_law.py)"
        )
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=DEFAULT_CONCEPTS,
        choices=CONCEPT_CHOICES,
        help="Concepts whose attachments to test (default: n1..n20 + variants)",
    )
    parser.add_argument(
        "--attachments-dir",
        type=str,
        default=None,
        help=f"Root of attachments tree (default: {ATTACHMENTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Power-law statistics results root for Clauset outputs (default: {POWERLAW_STATISTICS_REAL_DIR})",
    )
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=1000,
        help="Bootstrap iterations passed to clauset_power_law (default: 1000)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling (default: 42)",
    )
    parser.add_argument(
        "--xmin",
        type=int,
        default=None,
        help=(
            "Lock xmin for empirical fit and bootstrap refits "
            "(default: estimate / model policy)"
        ),
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Forward --parallel to clauset_power_law (per-log CSV shards)",
    )
    return parser.parse_args(argv)


def models_for_concept(concept: str) -> list[str]:
    """Return which power-law models to fit for a concept."""
    if concept == "variants":
        return VARIANT_MODELS
    return NGRAM_MODELS


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    attachments_root = (
        Path(args.attachments_dir) if args.attachments_dir else ATTACHMENTS_DIR
    )
    output_root = Path(args.output_dir) if args.output_dir else POWERLAW_STATISTICS_REAL_DIR

    step_t0 = time.perf_counter()
    concept_elapsed: list[tuple[str, float]] = []
    for concept in args.concepts:
        models = models_for_concept(concept)
        inputs = [
            f"{dataset}={attachments_root / concept / dataset / 'attachments.csv.gz'}"
            for dataset in args.datasets
        ]
        forward = [
            "--inputs",
            *inputs,
            "--analysis-name",
            concept,
            "--output-dir",
            str(output_root),
            "--n-bootstraps",
            str(args.n_bootstraps),
            "--random-seed",
            str(args.random_seed),
            "--models",
            *models,
        ]
        if args.xmin is not None:
            forward.extend(["--xmin", str(args.xmin)])
        if args.parallel:
            forward.append("--parallel")
        print(f"$ python cli/clauset_power_law.py {' '.join(forward)}")
        timing_print(
            f"concept={concept} start "
            f"(datasets={len(args.datasets)}, models={','.join(models)}, "
            f"n_bootstraps={args.n_bootstraps})"
        )
        concept_t0 = time.perf_counter()
        clauset_power_law.main(forward)
        elapsed = time.perf_counter() - concept_t0
        concept_elapsed.append((concept, elapsed))
        timing_print(f"concept={concept} done elapsed={format_duration(elapsed)}")

    timing_print(
        "clauset_tests total "
        f"elapsed={format_duration(time.perf_counter() - step_t0)} "
        f"({len(concept_elapsed)} concepts)"
    )
    for concept, elapsed in concept_elapsed:
        timing_print(f"  summary concept={concept} elapsed={format_duration(elapsed)}")


if __name__ == "__main__":
    main()
