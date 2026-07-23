"""Single CLI entrypoint for the RFC study pipeline.

Pipeline order:
1) log_info.py (unless --skip-log-info)
2) extract_attachments.py (unless --skip-extract)
3) rfc_powerlaw_analysis.py (unless --skip-static)
4) pdf_powerlaw_analysis.py (optional, via --run-pdf)
5) dynamic_rfc_analysis.py (optional, via --run-dynamic)
6) alpha_correlationy.py (optional, via --run-alpha-correlation)
7) integrity_checks.py (always last)
"""

import argparse
from pathlib import Path
from typing import List

from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ALL_REAL_LOGS_TOKEN,
    RESULTS_DIR,
    TEST_DATASET,
    TEST_RESULTS_DIR,
)


def _build_attachment_inputs(
    datasets: List[str], output_dir: str | None, concept: str
) -> List[str]:
    """Build <dataset>=<attachments.csv.gz> input pairs for analysis scripts."""
    output_root = Path(output_dir) if output_dir else RESULTS_DIR
    return [
        f"{dataset}={output_root / concept / dataset / 'attachments.csv.gz'}"
        for dataset in datasets
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI args for full pipeline execution."""
    parser = argparse.ArgumentParser(description="Run full RFC study pipeline")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Dataset names from data dictionary (ignored when --test is set). "
            f"Use '{ALL_REAL_LOGS_TOKEN}' to run the standard real-log benchmark set."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run standardized test mode (dataset TEST_BPIC12, output root results/test)",
    )
    parser.add_argument(
        "--analysis-name",
        type=str,
        default=None,
        help="Subfolder for static/dynamic outputs (default: same as --concept; omit dataset from this label)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory (default: results)",
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="variants",
        choices=["variants", "activities", "dfrs"],
        help="Concept used for extraction and static analysis inputs",
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.75, help="Split ratio for dynamic PA analysis"
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.1,
        help="Lambda regularization for dynamic PA analysis",
    )
    parser.add_argument(
        "--x-offset", type=float, default=1.0, help="Offset for dynamic x transform: x_offset + d1(u)"
    )
    parser.add_argument(
        "--node-set-mode",
        type=str,
        default="old_only",
        choices=["old_only", "union"],
        help=(
            "Node population for dynamic d1/d2 table: "
            "'old_only' (nodes in t1 only) or "
            "'union' (t1 union t2, t2-new nodes get d1=0)."
        ),
    )
    parser.add_argument("--skip-static", action="store_true", help="Skip static RFC stage")
    parser.add_argument(
        "--skip-log-info",
        action="store_true",
        help="Skip log-info table generation stage",
    )
    parser.add_argument(
        "--force-recalculate",
        action="store_true",
        help="Recompute log-info metrics from logs (ignore cached log_info.csv rows)",
    )
    parser.add_argument(
        "--skip-extract", action="store_true", help="Skip attachment extraction stage"
    )
    parser.add_argument(
        "--run-pdf",
        action="store_true",
        help="Run optional PDF power-law stage (default: off)",
    )
    parser.add_argument(
        "--run-dynamic",
        action="store_true",
        help="Run optional dynamic RFC stage (default: off)",
    )
    parser.add_argument(
        "--run-alpha-correlation",
        action="store_true",
        help="Run optional alpha correlation stage (default: off)",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the selected stages in sequence."""
    args = parse_args()
    if args.test:
        datasets = [TEST_DATASET]
        output_dir = str(TEST_RESULTS_DIR)
    else:
        if not args.datasets:
            raise SystemExit("Error: --datasets is required unless --test is set.")
        datasets = (
            list(ALL_REAL_LOG_DATASETS)
            if len(args.datasets) == 1 and args.datasets[0] == ALL_REAL_LOGS_TOKEN
            else args.datasets
        )
        output_dir = args.output_dir

    analysis_name = (args.analysis_name or args.concept).strip() or "run"

    # Build shared dataset arg list once so all stages stay consistent.
    dataset_args = ["--datasets", *datasets]
    output_args = ["--output-dir", output_dir] if output_dir else []

    if not args.skip_log_info:
        from scripts import log_info

        log_info_argv = [*dataset_args, "--output-dir", str(RESULTS_DIR / "log_info")]
        if args.force_recalculate:
            log_info_argv.append("--force-recalculate")
        print(f"$ {' '.join(['python', 'scripts/log_info.py', *log_info_argv])}")
        log_info.main(log_info_argv)

    if not args.skip_extract:
        from scripts import extract_attachments

        # extract_attachments skips existing attachments.csv.gz by default (unless --force is used).
        extract_argv = [
            *dataset_args,
            "--concepts",
            args.concept,
            *output_args,
        ]
        print(f"$ {' '.join(['python', 'scripts/extract_attachments.py', *extract_argv])}")
        extract_attachments.main(extract_argv)

    if not args.skip_static:
        from scripts import rfc_powerlaw_analysis

        static_inputs = _build_attachment_inputs(
            datasets, output_dir, args.concept
        )
        rfc_argv = [
            "--inputs",
            *static_inputs,
            "--analysis-name",
            analysis_name,
            *output_args,
        ]
        print(f"$ {' '.join(['python', 'scripts/rfc_powerlaw_analysis.py', *rfc_argv])}")
        rfc_powerlaw_analysis.main(rfc_argv)

        if args.run_pdf:
            try:
                from scripts import pdf_powerlaw_analysis
            except ModuleNotFoundError as exc:
                if exc.name == "powerlaw":
                    print(
                        "Warning: skipping pdf_powerlaw_analysis stage because "
                        "the 'powerlaw' package is not installed."
                    )
                else:
                    raise
            else:
                pdf_argv = [
                    "--inputs",
                    *static_inputs,
                    "--analysis-name",
                    analysis_name,
                    *output_args,
                ]
                print(f"$ {' '.join(['python', 'scripts/pdf_powerlaw_analysis.py', *pdf_argv])}")
                pdf_powerlaw_analysis.main(pdf_argv)

    if args.run_dynamic:
        from scripts import dynamic_rfc_analysis

        dynamic_inputs = _build_attachment_inputs(
            datasets, output_dir, args.concept
        )
        pdf_summary_path = (
            Path(output_dir) if output_dir else RESULTS_DIR
        ) / analysis_name / "pdf_static_analysis_summary.csv"
        dynamic_argv = [
            "--inputs",
            *dynamic_inputs,
            "--analysis-name",
            analysis_name,
            "--split-ratio",
            str(args.split_ratio),
            "--lambda-reg",
            str(args.lambda_reg),
            "--x-offset",
            str(args.x_offset),
            "--node-set-mode",
            str(args.node_set_mode),
            *output_args,
        ]
        if args.run_pdf:
            dynamic_argv.extend(["--static-summary-path", str(pdf_summary_path)])
        print(f"$ {' '.join(['python', 'scripts/dynamic_rfc_analysis.py', *dynamic_argv])}")
        dynamic_rfc_analysis.main(dynamic_argv)

    if args.run_alpha_correlation:
        from scripts import alpha_correlationy

        output_root = Path(output_dir) if output_dir else RESULTS_DIR
        static_path = output_root / analysis_name / "rfc_static_analysis.csv"
        corr_argv = [
            "--static-analysis-path",
            str(static_path),
            "--analysis-name",
            analysis_name,
            "--output-dir",
            str(output_root),
        ]
        print(
            f"$ {' '.join(['python', 'scripts/alpha_correlationy.py', *corr_argv])}"
        )
        alpha_correlationy.main(corr_argv)

    from scripts import integrity_checks

    integrity_argv = [
        *dataset_args,
        "--concept",
        args.concept,
    ]
    if output_dir:
        integrity_argv.extend(["--output-dir", output_dir])
    print(
        f"$ {' '.join(['python', 'scripts/integrity_checks.py', *integrity_argv])}"
    )
    integrity_checks.main(integrity_argv)

    print("\nRFC study pipeline completed.")


if __name__ == "__main__":
    main()
