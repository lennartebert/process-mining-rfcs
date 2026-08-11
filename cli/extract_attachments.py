"""Extract concept-specific attachment tables from event logs.

Run from the repository root. Dataset paths in the data dictionary and default
`results/` output are relative to the current working directory. Importing
`utils` requires the repo root on PYTHONPATH (this module inserts it when run
as a script; otherwise `export PYTHONPATH="$PWD"`).

Outputs are written as:
`<output-root>/<concept>/<dataset>/attachments.csv.gz`
where concept is one of: variants, activities, dfrs, n1..n10.
"""

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import DATA_DICTIONARY_PATH, RESULTS_DIR
from utils.io import (
    extract_attachments_from_trace_data,
    get_data_dictionary,
    get_event_log_from_path,
    save_attachments,
    trace_completion_data,
)
from utils.io.attachments import NGRAM_CONCEPTS


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI args for attachment extraction."""
    parser = argparse.ArgumentParser(description="Extract trace attachments to CSV files")
    parser.add_argument("--datasets", nargs="+", required=True, help="Dataset names from data dictionary")
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
        default=["variants"],
        choices=["variants", "activities", "dfrs", *NGRAM_CONCEPTS],
        help="Concepts for attachment extraction (n1..n10 = length-k activity n-grams)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    """Run extraction for all requested datasets."""
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    # Load project dataset registry and validate requested names.
    data_dictionary = get_data_dictionary(
        DATA_DICTIONARY_PATH, get_real=True, get_synthetic=True
    )

    print(f"Processing {len(args.datasets)} dataset(s)...")
    for dataset_name in args.datasets:
        concept_outputs = []
        for concept in args.concepts:
            dataset_concept_dir = output_root / concept / dataset_name
            dataset_concept_dir.mkdir(parents=True, exist_ok=True)
            attachment_path = dataset_concept_dir / "attachments.csv.gz"
            concept_outputs.append((concept, attachment_path))

        pending_outputs = []
        for concept, attachment_path in concept_outputs:
            if attachment_path.exists() and not args.force:
                print(f"  Skipping {concept}/{dataset_name}: {attachment_path} already exists")
                continue
            pending_outputs.append((concept, attachment_path))

        if not pending_outputs:
            print(f"Skipping dataset {dataset_name}: all requested concepts already extracted")
            continue

        if dataset_name not in data_dictionary:
            missing = ", ".join(concept for concept, _ in pending_outputs)
            print(
                f"Warning: cannot extract {dataset_name} ({missing}): "
                "dataset not in data dictionary and attachments.csv.gz is missing"
            )
            continue

        log_path = Path(data_dictionary[dataset_name]["path"])
        if not log_path.exists():
            print(f"Warning: Log file missing for {dataset_name}: {log_path}")
            continue

        print(f"Extracting {dataset_name} from {log_path}...")
        event_log = get_event_log_from_path(log_path)
        trace_data = trace_completion_data(event_log)
        # Sort once and reuse for all requested concepts to avoid repeated log traversal.
        trace_data.sort(key=lambda item: item["completion_timestamp"])
        for concept, attachment_path in pending_outputs:
            attachments_df = extract_attachments_from_trace_data(trace_data, concept)
            save_attachments(attachments_df, attachment_path)
            unique_nodes = attachments_df["node_id"].nunique() if not attachments_df.empty else 0
            print(
                f"  Saved {concept}/{dataset_name}/attachments.csv.gz "
                f"(rows={len(attachments_df)}, nodes={unique_nodes})"
            )

    print(f"\nAttachment extraction complete. Output root: {output_root}")


if __name__ == "__main__":
    main()
