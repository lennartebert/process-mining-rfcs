"""Step 05: cross-log analyses from saved (combined) Clauset results.

Runs, in order:

1. saturation tables under ``<experiment>/log_info/saturation/``
2. paired GOF comparisons under the Clauset configuration folder
3. per-log GOF-change exploration under ``<xmin>/explore_gof_changes/``

Fits, likelihood-ratio tests, and GOF bootstraps are not rerun. Saturation
ignores xmin and GOF thresholds. Paired GOF uses ``min_types`` and ``gof_p``;
``comparison_p`` only selects the destination folder name. Explore uses the
xmin folder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.powerlaw_statistics.analysis_helpers import explore_gof_changes
from analyses.powerlaw_statistics.analysis_helpers import saturation as saturation_diagnostics
from analyses.powerlaw_statistics.analysis_helpers.paired_gof import write_paired_gof_for_config
from utils.constants import PL_STATS_REAL_DIR
from utils.powerlaw import DEFAULT_MINIMUM_FITTED_TYPES, DEFAULT_SIGNIFICANCE_LEVEL


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-log saturation, paired GOF, and GOF-change exploration "
            "from saved Clauset results"
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional log filter for paired GOF / explore (default: all study logs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Experiment directory (default: {PL_STATS_REAL_DIR})",
    )
    parser.add_argument(
        "--xmin",
        type=int,
        default=None,
        help="xmin folder for paired GOF and explore (default: xmin-adaptive)",
    )
    parser.add_argument(
        "--min-types",
        dest="min_types",
        type=int,
        default=DEFAULT_MINIMUM_FITTED_TYPES,
        help=(
            "Minimum fitted types for paired GOF eligibility "
            f"(default: {DEFAULT_MINIMUM_FITTED_TYPES})"
        ),
    )
    parser.add_argument(
        "--gof-p",
        dest="gof_p",
        type=float,
        default=DEFAULT_SIGNIFICANCE_LEVEL,
        help=(
            "GOF non-rejection threshold for paired GOF "
            f"(default: {DEFAULT_SIGNIFICANCE_LEVEL})"
        ),
    )
    parser.add_argument(
        "--comparison-p",
        dest="comparison_p",
        type=float,
        default=DEFAULT_SIGNIFICANCE_LEVEL,
        help=(
            "Comparison-p folder label for paired GOF outputs "
            f"(default: {DEFAULT_SIGNIFICANCE_LEVEL}; does not enter the paired test)"
        ),
    )
    parser.add_argument(
        "--skip-explore-gof-changes",
        action="store_true",
        help="Skip per-log GOF-change exploration artifacts",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_dir = Path(args.output_dir) if args.output_dir else PL_STATS_REAL_DIR
    datasets = list(args.datasets) if args.datasets else None
    print(
        "Cross-log analysis: "
        f"experiment={experiment_dir}, xmin={args.xmin!r}, "
        f"min_types={int(args.min_types)}, gof_p={float(args.gof_p):g}, "
        f"comparison_p={float(args.comparison_p):g}"
    )
    saturation_diagnostics.write_saturation_diagnostics(experiment_dir)
    write_paired_gof_for_config(
        experiment_dir,
        xmin_setting=args.xmin,
        min_fitted_types=int(args.min_types),
        gof_threshold=float(args.gof_p),
        comparison_p=float(args.comparison_p),
        study_logs=datasets,
    )
    if args.skip_explore_gof_changes:
        print("Skipping explore_gof_changes (--skip-explore-gof-changes)")
        return
    if not experiment_dir.name.startswith("real"):
        print(
            "Skipping explore_gof_changes "
            f"({experiment_dir.name} is not a real-log experiment)"
        )
        return
    xmin_token = "none" if args.xmin is None else str(int(args.xmin))
    explore_argv = ["--xmin", xmin_token]
    if datasets is not None:
        explore_argv.extend(["--datasets", *datasets])
    # explore_gof_changes resolves paths under results/pl_stats/real/<xmin>.
    # When the experiment is a non-default real* folder, still run against that
    # xmin policy under the default real tree only if experiment is exactly "real".
    if experiment_dir.resolve() != PL_STATS_REAL_DIR.resolve():
        print(
            "Skipping explore_gof_changes "
            f"(experiment {experiment_dir} is not {PL_STATS_REAL_DIR})"
        )
        return
    explore_gof_changes.main(explore_argv)


if __name__ == "__main__":
    main()
