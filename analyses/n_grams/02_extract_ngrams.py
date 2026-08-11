"""Step 02: extract n-gram and variant attachments (cli.extract_attachments)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli import extract_attachments
from utils.io.attachments import NGRAM_CONCEPTS

DEFAULT_CONCEPTS = ["variants", *NGRAM_CONCEPTS]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract attachments for variants and n-grams n1..n10 "
            "via cli/extract_attachments.py"
        )
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute attachments even when attachments.csv.gz already exists",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=DEFAULT_CONCEPTS,
        choices=["variants", "activities", "dfrs", *NGRAM_CONCEPTS],
        help="Concepts to extract (default: variants + n1..n10)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    forward = ["--datasets", *args.datasets, "--concepts", *args.concepts]
    if args.output_dir:
        forward.extend(["--output-dir", args.output_dir])
    if args.force:
        forward.append("--force")
    print(f"$ python cli/extract_attachments.py {' '.join(forward)}")
    extract_attachments.main(forward)


if __name__ == "__main__":
    main()
