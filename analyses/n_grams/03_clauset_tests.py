"""Step 03: Clauset power-law tests on n-gram/variant attachments.

Runs ``cli.clauset_power_law`` once per concept (variants, n1..n10 by default),
writing under ``results/statistical_tests/<concept>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli import clauset_power_law
from utils.constants import RESULTS_DIR
from utils.io.attachments import NGRAM_CONCEPTS

DEFAULT_CONCEPTS = ["variants", *NGRAM_CONCEPTS]


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
        choices=["variants", "activities", "dfrs", *NGRAM_CONCEPTS],
        help="Concepts whose attachments to test (default: variants + n1..n10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root containing <concept>/<dataset>/attachments.csv.gz "
        "(default: results)",
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
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    for concept in args.concepts:
        inputs = [
            f"{dataset}={output_root / concept / dataset / 'attachments.csv.gz'}"
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
        ]
        print(f"$ python cli/clauset_power_law.py {' '.join(forward)}")
        clauset_power_law.main(forward)


if __name__ == "__main__":
    main()
