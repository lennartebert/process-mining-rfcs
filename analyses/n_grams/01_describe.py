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
from utils.constants import LOG_INFO_DIR


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe event logs via cli/log_info.py"
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(LOG_INFO_DIR),
        help="Output directory for log-info tables (default: results/log_info)",
    )
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Ignore cached CSV rows and recompute metrics from event logs",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    forward = ["--datasets", *args.datasets, "--output-dir", args.output_dir]
    if args.force_recalculate:
        forward.append("--force-recalculate")
    print(f"$ python cli/log_info.py {' '.join(forward)}")
    log_info.main(forward)


if __name__ == "__main__":
    main()
