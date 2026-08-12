"""Step 01: describe event logs (wrapper around cli.log_info)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli import log_info
from cli.log_info import ALL_STATS, DEFAULT_STATS
from utils.constants import N_GRAMS_DIR

# n_grams describe always includes trace-length stats by default.
N_GRAMS_DEFAULT_STATS: List[str] = [
    *DEFAULT_STATS,
    "Median Trace Length",
    "Max Trace Length",
]


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe event logs via cli/log_info.py"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--stats",
        nargs="+",
        default=list(N_GRAMS_DEFAULT_STATS),
        choices=ALL_STATS,
        help=(
            "Log stats to extract (default: "
            + ", ".join(f"'{s}'" for s in N_GRAMS_DEFAULT_STATS)
            + ")."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(N_GRAMS_DIR),
        help=f"Output directory for log_info.csv/.tex (default: {N_GRAMS_DIR})",
    )
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Ignore cached CSV rows and recompute metrics from event logs",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Forward --parallel to cli/log_info.py (per-log CSV shard, no LaTeX)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    forward = [
        "--datasets",
        *args.datasets,
        "--stats",
        *args.stats,
        "--output-dir",
        args.output_dir,
    ]
    if args.force_recalculate:
        forward.append("--force-recalculate")
    if args.parallel:
        forward.append("--parallel")
    print(f"$ python cli/log_info.py {' '.join(forward)}")
    log_info.main(forward)


if __name__ == "__main__":
    main()
