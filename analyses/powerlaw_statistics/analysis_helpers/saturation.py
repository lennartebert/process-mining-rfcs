"""Cross-log saturation of trace n-grams at selected orders.

Trace length and variant identity follow attachment extraction
(``trace_completion_data`` and the activity classifier). No power-law fit
or goodness-of-fit test is run.

A case is complete when it contributes one occurrence. Order k = 1 does not
wrap the trace, so that is activity length L = 1 and the case contributes L
occurrences. For k >= 2 the extractor wraps each trace as (START, activities,
END) and emits one node when L + 2 <= k, or L + 3 - k sliding windows
otherwise. That is ``max(1, L + 3 - k)`` occurrences per case.
"""

from __future__ import annotations

import gzip
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ATTACHMENTS_DIR,
    DATA_DICTIONARY_PATH,
)
from analyses.powerlaw_statistics.analysis_helpers.paths import (
    SATURATION_DIRNAME,
    concept_dirname,
    raw_results_dir,
    saturation_dir,
)
from utils.powerlaw import LOWER_BOUNDED_POWER_LAW

SATURATION_KS = (1, 5, 10, 15, 20)
CROSS_LOG_KS = (5, 10, 15, 20)
STATUS_OK = "ok"
STATUS_MISSING = "missing_log"
STATUS_EMPTY = "empty_log"
MEDIAN_ROW_LABEL = "Median across logs"
SUMMARY_FILENAME = "saturation_summary.csv"
CROSS_LOG_FILENAME = "cross_log_saturation.csv"
STALE_TEXT_FILENAME = "saturation_results.txt"
CROSS_LOG_LOG_COLUMN = "Log"
CROSS_LOG_OCCURRENCE_COLUMN = "Complete-case occurrence share, k=20"
CROSS_LOG_EXCESS_COLUMN = "Excess occurrences per case, k=20"
CROSS_LOG_SUMMARY_BLANK = "—"
CROSS_LOG_THRESHOLDS = (50, 60, 70, 80, 90, 100)

K_TABLE_COLUMNS = [
    "Log",
    "Cases",
    "Trace variants",
    "Median length",
    "Max length",
    "Cases complete (%)",
    "Variants complete (%)",
    "Occurrences from complete cases (%)",
    "Excess occurrences per case",
]

SUMMARY_COLUMNS = [
    "log_name",
    "k",
    "max_complete_trace_length",
    "n_cases",
    "n_variants",
    "median_trace_length",
    "max_trace_length",
    "n_complete_cases",
    "complete_case_share",
    "n_complete_variants",
    "complete_variant_share",
    "total_occurrences",
    "complete_occurrence_share",
    "excess_occurrences",
    "excess_occurrences_per_case",
    "status",
    "notes",
    "extracted_occurrences",
    "extracted_occurrences_source",
]

_COUNT_COLUMNS = (
    "n_cases",
    "n_variants",
    "max_trace_length",
    "n_complete_cases",
    "n_complete_variants",
    "total_occurrences",
    "excess_occurrences",
    "extracted_occurrences",
    "max_complete_trace_length",
)
_SHARE_COLUMNS = (
    "complete_case_share",
    "complete_variant_share",
    "complete_occurrence_share",
    "excess_occurrences_per_case",
    "median_trace_length",
)

@dataclass(frozen=True)
class LengthProfile:
    """Per-case and per-variant lengths for one study log."""

    log_name: str
    status: str
    notes: str
    case_lengths: tuple[int, ...] = ()
    variant_lengths: tuple[int, ...] = ()


def occurrence_weight(length: int, k: int) -> int:
    """Occurrences one case of activity-length ``length`` contributes at order k.

    Order 1 does not wrap: the weight is L. Orders k >= 2 wrap with START and
    END and contribute ``max(1, L + 3 - k)`` windows.
    """
    length = int(length)
    order = int(k)
    if length < 0:
        raise ValueError(f"trace length must be non-negative, got {length}")
    if order == 1:
        return length
    if order < 2:
        raise ValueError(
            "occurrence weight is defined for k = 1 and wrapped n-grams k >= 2, "
            f"got k={order}"
        )
    return max(1, length + 3 - order)


def max_complete_trace_length(k: int) -> int:
    """Largest activity length that contributes one occurrence at order k.

    k = 1 keeps a trace of one activity. k >= 2 keeps L <= k - 2, because the
    wrapped trace has length L + 2.
    """
    order = int(k)
    if order == 1:
        return 1
    if order < 2:
        raise ValueError(f"completion threshold is defined for k >= 1, got k={order}")
    return order - 2


def _occurrence_weights(lengths: np.ndarray, k: int) -> np.ndarray:
    count = int(lengths.size)
    return np.fromiter(
        (occurrence_weight(int(length), k) for length in lengths),
        dtype=int,
        count=count,
    )


def profile_from_sequences(
    log_name: str, sequences: Sequence[Sequence[str]]
) -> LengthProfile:
    """Build a profile from activity sequences already selected for extraction."""
    if len(sequences) == 0:
        return LengthProfile(
            log_name=log_name,
            status=STATUS_EMPTY,
            notes="No traces with a completion timestamp.",
        )
    case_lengths = tuple(len(sequence) for sequence in sequences)
    variant_lengths = tuple(len(sequence) for sequence in set(map(tuple, sequences)))
    return LengthProfile(
        log_name=log_name,
        status=STATUS_OK,
        notes="",
        case_lengths=case_lengths,
        variant_lengths=variant_lengths,
    )


def diagnose_profile(profile: LengthProfile, k: int) -> dict[str, object]:
    """One log × order row. Unknown logs stay missing rather than zero."""
    row: dict[str, object] = {
        "log_name": profile.log_name,
        "k": int(k),
        "max_complete_trace_length": max_complete_trace_length(k),
        "status": profile.status,
        "notes": profile.notes,
        "extracted_occurrences": pd.NA,
        "extracted_occurrences_source": pd.NA,
    }
    for column in (
        "n_cases",
        "n_variants",
        "median_trace_length",
        "max_trace_length",
        "n_complete_cases",
        "complete_case_share",
        "n_complete_variants",
        "complete_variant_share",
        "total_occurrences",
        "complete_occurrence_share",
        "excess_occurrences",
        "excess_occurrences_per_case",
    ):
        row[column] = pd.NA
    if profile.status != STATUS_OK:
        return row

    lengths = np.asarray(profile.case_lengths, dtype=int)
    variants = np.asarray(profile.variant_lengths, dtype=int)
    n_cases = int(lengths.size)
    n_variants = int(variants.size)
    if n_cases == 0 or n_variants == 0:
        raise ValueError(f"{profile.log_name} is marked ok but has no lengths")
    weights = _occurrence_weights(lengths, k)
    variant_weights = _occurrence_weights(variants, k)
    # Complete means the case or variant contributes exactly one occurrence.
    n_complete_cases = int(np.count_nonzero(weights == 1))
    n_complete_variants = int(np.count_nonzero(variant_weights == 1))
    total = int(weights.sum())
    if total <= 0:
        raise ValueError(f"{profile.log_name} k={k} has no occurrences")
    excess = total - n_cases
    row.update(
        {
            "n_cases": n_cases,
            "n_variants": n_variants,
            "median_trace_length": float(np.median(lengths)),
            "max_trace_length": int(lengths.max()),
            "n_complete_cases": n_complete_cases,
            "complete_case_share": n_complete_cases / n_cases,
            "n_complete_variants": n_complete_variants,
            "complete_variant_share": n_complete_variants / n_variants,
            "total_occurrences": total,
            "complete_occurrence_share": n_complete_cases / total,
            "excess_occurrences": excess,
            "excess_occurrences_per_case": excess / n_cases,
        }
    )
    return row


def build_saturation_frame(
    profiles: Sequence[LengthProfile],
    extracted: Mapping[tuple[str, int], tuple[int, str]] | None = None,
    *,
    ks: Sequence[int] = SATURATION_KS,
) -> pd.DataFrame:
    """One row per profile × k, in profile order then k order."""
    extracted = extracted or {}
    rows = []
    for profile in profiles:
        for k in ks:
            row = diagnose_profile(profile, k)
            found = extracted.get((profile.log_name, int(k)))
            if found is not None and profile.status != STATUS_OK:
                total, source = found
                raise AssertionError(
                    f"{profile.log_name} k={k} is {profile.status} but a saved "
                    f"occurrence total exists ({total}, {source}). {profile.notes}"
                )
            if found is not None:
                total, source = found
                row["extracted_occurrences"] = int(total)
                row["extracted_occurrences_source"] = source
                if profile.status == STATUS_OK:
                    match = int(row["total_occurrences"]) == int(total)
                    note = (
                        f"Extracted occurrence total matches ({source})."
                        if match
                        else (
                            f"Extracted occurrence total {total} ({source}) "
                            f"disagrees with formula total {row['total_occurrences']}."
                        )
                    )
                    row["notes"] = _join_notes(row["notes"], note)
            elif profile.status == STATUS_OK:
                row["notes"] = _join_notes(
                    row["notes"], "Extracted occurrence total is unavailable."
                )
            rows.append(row)
    frame = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return _coerce_columns(frame)


def validate_saturation_frame(
    frame: pd.DataFrame, *, ks: Sequence[int] = SATURATION_KS
) -> None:
    """Check identities, monotonicity, and agreement with extracted totals."""
    ordered_k = list(ks)
    for log_name, group in frame.groupby("log_name", sort=False):
        group = group.set_index("k", drop=False).loc[list(ordered_k)]
        status = set(group["status"])
        if status != {STATUS_OK}:
            if status <= {STATUS_MISSING, STATUS_EMPTY} and len(status) == 1:
                for column in (
                    "n_cases",
                    "n_variants",
                    "median_trace_length",
                    "max_trace_length",
                    "n_complete_cases",
                    "complete_case_share",
                    "n_complete_variants",
                    "complete_variant_share",
                    "total_occurrences",
                    "complete_occurrence_share",
                    "excess_occurrences",
                    "excess_occurrences_per_case",
                ):
                    if group[column].notna().any():
                        raise AssertionError(
                            f"{log_name} status {status} has a numeric value in {column}"
                        )
                continue
            raise AssertionError(f"{log_name} has mixed or unknown status {status}")

        previous = None
        for k in ordered_k:
            row = group.loc[k]
            n_cases = int(row["n_cases"])
            n_variants = int(row["n_variants"])
            total = int(row["total_occurrences"])
            n_complete_cases = int(row["n_complete_cases"])
            n_complete_variants = int(row["n_complete_variants"])
            excess = int(row["excess_occurrences"])
            if n_cases <= 0 or n_variants <= 0 or total <= 0:
                raise AssertionError(f"{log_name} k={k} has a non-positive count")
            if excess != total - n_cases:
                raise AssertionError(f"{log_name} k={k} excess does not equal O_k - N")
            if abs(float(row["excess_occurrences_per_case"]) - excess / n_cases) > 1e-12:
                raise AssertionError(f"{log_name} k={k} excess per case disagrees")
            if abs(float(row["complete_case_share"]) - n_complete_cases / n_cases) > 1e-12:
                raise AssertionError(f"{log_name} k={k} case share disagrees")
            if abs(float(row["complete_variant_share"]) - n_complete_variants / n_variants) > 1e-12:
                raise AssertionError(f"{log_name} k={k} variant share disagrees")
            if abs(float(row["complete_occurrence_share"]) - n_complete_cases / total) > 1e-12:
                raise AssertionError(f"{log_name} k={k} occurrence share disagrees")
            for share_name in (
                "complete_case_share",
                "complete_variant_share",
                "complete_occurrence_share",
            ):
                share = float(row[share_name])
                if share < -1e-12 or share > 1 + 1e-12:
                    raise AssertionError(f"{log_name} k={k} {share_name} is outside [0, 1]")
            if int(row["max_complete_trace_length"]) != max_complete_trace_length(int(k)):
                raise AssertionError(f"{log_name} k={k} completion threshold disagrees")
            max_length = int(row["max_trace_length"])
            saturated = (
                int(k) == 1 and max_length <= 1 and total == n_cases
            ) or (int(k) >= 2 and int(k) >= max_length + 2)
            if saturated:
                if not (
                    n_complete_cases == n_cases
                    and n_complete_variants == n_variants
                    and total == n_cases
                    and excess == 0
                    and abs(float(row["complete_case_share"]) - 1) < 1e-12
                    and abs(float(row["complete_variant_share"]) - 1) < 1e-12
                    and abs(float(row["complete_occurrence_share"]) - 1) < 1e-12
                ):
                    raise AssertionError(
                        f"{log_name} k={k} is past the longest trace but is not saturated"
                    )
            if pd.notna(row["extracted_occurrences"]):
                if int(row["extracted_occurrences"]) != total:
                    raise AssertionError(
                        f"{log_name} k={k} formula total {total} != extracted "
                        f"{int(row['extracted_occurrences'])} "
                        f"({row['extracted_occurrences_source']})"
                    )
            if previous is not None:
                if n_complete_cases < previous["n_complete_cases"]:
                    raise AssertionError(f"{log_name} complete cases decrease in k")
                if n_complete_variants < previous["n_complete_variants"]:
                    raise AssertionError(f"{log_name} complete variants decrease in k")
                if float(row["complete_case_share"]) + 1e-12 < previous["complete_case_share"]:
                    raise AssertionError(f"{log_name} case-completion share decreases in k")
                if float(row["complete_variant_share"]) + 1e-12 < previous["complete_variant_share"]:
                    raise AssertionError(f"{log_name} variant-completion share decreases in k")
                if total > previous["total_occurrences"]:
                    raise AssertionError(f"{log_name} occurrence total increases in k")
                if excess > previous["excess_occurrences"]:
                    raise AssertionError(f"{log_name} excess occurrences increase in k")
            previous = {
                "n_complete_cases": n_complete_cases,
                "n_complete_variants": n_complete_variants,
                "complete_case_share": float(row["complete_case_share"]),
                "complete_variant_share": float(row["complete_variant_share"]),
                "total_occurrences": total,
                "excess_occurrences": excess,
            }


def k_table_filename(k: int) -> str:
    """CSV name for one order, e.g. ``k1.csv`` or ``k20.csv``."""
    return f"k{int(k)}.csv"


def build_k_table(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    """One order's table: a row per log, then the unweighted median across logs.

    Shares are percentage points. Count and length cells are blank for a log
    with no usable traces, and on the median row. The median is not a pooled
    percentage: each usable log counts once.
    """
    order = int(k)
    block = _block_for_k(frame, order)
    rows = [_k_table_row(row) for _, row in block.iterrows()]
    rows.append(_k_table_median_row(block))
    table = pd.DataFrame(rows, columns=K_TABLE_COLUMNS)
    for column in ("Cases", "Trace variants", "Max length"):
        table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    for column in (
        "Median length",
        "Cases complete (%)",
        "Variants complete (%)",
        "Occurrences from complete cases (%)",
        "Excess occurrences per case",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def build_cross_log_table(
    frame: pd.DataFrame, *, ks: Sequence[int] = CROSS_LOG_KS
) -> pd.DataFrame:
    """Wide case-completion table plus threshold summary rows.

    Each study log is one row. The k columns are the percent of cases complete
    at that order, formatted to one decimal place. The last two columns are
    the complete-case occurrence share and the excess occurrences per case at
    k = 20. Summary rows count logs whose displayed case-completion percent
    meets 50, 60, 70, 80, 90, and 100. A log with no usable traces is listed with blank
    cells and is left out of those counts.
    """
    orders = [int(k) for k in ks]
    if 20 not in orders:
        raise ValueError(f"cross-log table needs k=20, got {orders}")
    columns = [
        CROSS_LOG_LOG_COLUMN,
        *[f"k={k}" for k in orders],
        CROSS_LOG_OCCURRENCE_COLUMN,
        CROSS_LOG_EXCESS_COLUMN,
    ]
    by_k = {k: _block_for_k(frame, k) for k in orders}
    log_names = _log_order(frame)
    rows: list[dict[str, str]] = []
    for log_name in log_names:
        row = {column: "" for column in columns}
        row[CROSS_LOG_LOG_COLUMN] = log_name
        at_20 = by_k[20].loc[log_name]
        if at_20["status"] == STATUS_OK:
            row[CROSS_LOG_OCCURRENCE_COLUMN] = _format_percent_cell(
                at_20["complete_occurrence_share"]
            )
            row[CROSS_LOG_EXCESS_COLUMN] = _format_excess_cell(
                at_20["excess_occurrences_per_case"]
            )
        for k in orders:
            record = by_k[k].loc[log_name]
            if record["status"] == STATUS_OK:
                row[f"k={k}"] = _format_percent_cell(record["complete_case_share"])
        rows.append(row)
    for threshold in CROSS_LOG_THRESHOLDS:
        summary = {column: "" for column in columns}
        summary[CROSS_LOG_LOG_COLUMN] = f"Logs with ≥{threshold}% of cases complete"
        summary[CROSS_LOG_OCCURRENCE_COLUMN] = CROSS_LOG_SUMMARY_BLANK
        summary[CROSS_LOG_EXCESS_COLUMN] = CROSS_LOG_SUMMARY_BLANK
        for k in orders:
            block = by_k[k]
            usable = block[block["status"] == STATUS_OK]
            n_logs = int(len(usable))
            n_meeting = int(
                sum(
                    _shown_percent_value(share) >= threshold
                    for share in usable["complete_case_share"]
                )
            )
            summary[f"k={k}"] = _format_threshold_cell(n_meeting, n_logs)
        rows.append(summary)
    return pd.DataFrame(rows, columns=columns)


def write_saturation_tables(
    frame: pd.DataFrame,
    destination: Path,
    *,
    ks: Sequence[int] = SATURATION_KS,
) -> None:
    """Write the long summary, one CSV per order, and the wide cross-log table."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / SUMMARY_FILENAME, index=False)
    for k in ks:
        build_k_table(frame, int(k)).to_csv(
            destination / k_table_filename(k), index=False
        )
    if all(int(k) in {int(order) for order in ks} for k in CROSS_LOG_KS):
        build_cross_log_table(frame).to_csv(
            destination / CROSS_LOG_FILENAME, index=False
        )
    stale = destination / STALE_TEXT_FILENAME
    if stale.is_file():
        stale.unlink()


def _raw_results_for_counts(experiment_dir: Path) -> Path | None:
    """Prefer the adaptive xmin folder; occurrence totals do not depend on xmin."""
    preferred = raw_results_dir(experiment_dir, None)
    if preferred.is_dir():
        return preferred
    matches = sorted(experiment_dir.glob("xmin*/raw_results"))
    for candidate in matches:
        if candidate.is_dir():
            return candidate
    return None


def write_saturation_diagnostics(output_dir: Path) -> Path | None:
    """Write saturation tables for a real-log experiment directory.

    The counts do not depend on xmin or on the GOF grid. They are written
    once, under ``<experiment>/log_info/saturation/``, not inside an xmin folder.
    """
    output_dir = Path(output_dir)
    if not output_dir.name.startswith("real"):
        print(
            "Skipping saturation diagnostics "
            f"({output_dir.name} is not a real-log experiment)"
        )
        return None
    print(f"Saturation diagnostics: loading study logs for {output_dir.name}")
    profiles = load_study_profiles(ALL_REAL_LOG_DATASETS)
    raw_results = _raw_results_for_counts(output_dir)
    extracted = load_extracted_occurrences(
        raw_results if raw_results is not None else output_dir,
        [profile.log_name for profile in profiles],
        SATURATION_KS,
    )
    frame = build_saturation_frame(profiles, extracted)
    validate_saturation_frame(frame)
    _migrate_legacy_saturation_dir(output_dir)
    destination = saturation_dir(output_dir)
    write_saturation_tables(frame, destination)
    print(f"Saturation diagnostics: {len(frame)} log-order rows -> {destination}")
    return destination


def load_study_profiles(log_names: Sequence[str]) -> list[LengthProfile]:
    """Load length profiles through the attachment-extraction trace selection."""
    from utils.io.attachments import trace_completion_data
    from utils.io.event_logs import get_data_dictionary, get_event_log_from_dictionary

    dictionary = get_data_dictionary(
        DATA_DICTIONARY_PATH, get_real=True, get_synthetic=True
    )
    profiles = []
    for log_name in log_names:
        print(f"  {log_name}")
        info = dictionary.get(log_name)
        if info is None:
            profiles.append(
                LengthProfile(
                    log_name=log_name,
                    status=STATUS_MISSING,
                    notes="Log is not in the data dictionary.",
                )
            )
            continue
        log_path = Path(info["path"])
        if not log_path.exists():
            profiles.append(
                LengthProfile(
                    log_name=log_name,
                    status=STATUS_MISSING,
                    notes=f"Event log file is missing: {log_path}",
                )
            )
            continue
        try:
            event_log = get_event_log_from_dictionary(log_name, dictionary)
            trace_rows = trace_completion_data(event_log)
        except Exception as exc:
            profiles.append(
                LengthProfile(
                    log_name=log_name,
                    status=STATUS_MISSING,
                    notes=f"Event log could not be read: {exc}",
                )
            )
            continue
        sequences = [tuple(row["activity_sequence"]) for row in trace_rows]
        del event_log, trace_rows
        profiles.append(profile_from_sequences(log_name, sequences))
    return profiles


def load_extracted_occurrences(
    output_dir: Path,
    log_names: Sequence[str],
    ks: Sequence[int],
    *,
    attachments_dir: Path = ATTACHMENTS_DIR,
) -> dict[tuple[str, int], tuple[int, str]]:
    """Saved occurrence totals, preferring the Clauset summary when it has one.

    A summary total and an attachment-row total for the same log are both
    read when the attachment file exists. They must agree.
    """
    found: dict[tuple[str, int], tuple[int, str]] = {}
    for k in ks:
        summary_totals = _summary_occurrence_totals(output_dir, int(k))
        for log_name in log_names:
            attachment_path = attachments_dir / f"n{k}" / log_name / "attachments.csv.gz"
            attachment_total = (
                _count_attachment_rows(attachment_path) if attachment_path.exists() else None
            )
            summary_total = summary_totals.get(log_name)
            if summary_total is not None and attachment_total is not None:
                if int(summary_total) != int(attachment_total):
                    raise AssertionError(
                        f"{log_name} n{k}: summary n_occurrences {summary_total} "
                        f"!= attachment rows {attachment_total}"
                    )
                found[(log_name, int(k))] = (int(summary_total), f"summary n{k}")
            elif summary_total is not None:
                found[(log_name, int(k))] = (int(summary_total), f"summary n{k}")
            elif attachment_total is not None:
                found[(log_name, int(k))] = (
                    int(attachment_total),
                    f"attachments n{k}",
                )
    return found


def _summary_occurrence_totals(output_dir: Path, k: int) -> dict[str, int]:
    path = (
        output_dir
        / concept_dirname(f"n{k}")
        / LOWER_BOUNDED_POWER_LAW
        / "summary.csv"
    )
    if not path.exists():
        return {}
    frame = pd.read_csv(path, usecols=["log_name", "n_occurrences"])
    if frame["log_name"].duplicated().any():
        duplicated = sorted(set(frame.loc[frame["log_name"].duplicated(), "log_name"]))
        raise ValueError(f"duplicate summary rows for n{k}: {duplicated}")
    totals = {}
    for _, row in frame.iterrows():
        if pd.isna(row["n_occurrences"]):
            continue
        totals[str(row["log_name"])] = int(row["n_occurrences"])
    return totals


def _count_attachment_rows(path: Path) -> int:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"empty attachment file: {path}")
        return sum(1 for line in handle if line.strip())


def _migrate_legacy_saturation_dir(experiment: Path) -> None:
    """Move a pre-log_info ``<experiment>/saturation`` folder under ``log_info``."""
    legacy = Path(experiment) / SATURATION_DIRNAME
    destination = saturation_dir(experiment)
    if not legacy.is_dir():
        return
    if legacy.resolve() == destination.resolve():
        return
    if destination.exists():
        raise FileExistsError(
            f"cannot move {legacy} to {destination}: destination already exists"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    legacy.rename(destination)
    print(f"Saturation diagnostics: moved {legacy} -> {destination}")


def _block_for_k(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    order = int(k)
    block = frame[frame["k"] == order]
    if block.empty:
        raise ValueError(f"saturation frame has no rows for k={order}")
    log_order = _log_order(frame)
    indexed = block.set_index("log_name", drop=False)
    missing = [log_name for log_name in log_order if log_name not in indexed.index]
    if missing:
        raise ValueError(f"k={order} is missing logs: {missing}")
    return indexed.loc[log_order]


def _k_table_row(row: pd.Series) -> dict[str, object]:
    blank: dict[str, object] = {column: pd.NA for column in K_TABLE_COLUMNS}
    blank["Log"] = str(row["log_name"])
    if row["status"] != STATUS_OK:
        return blank
    return {
        "Log": str(row["log_name"]),
        "Cases": int(row["n_cases"]),
        "Trace variants": int(row["n_variants"]),
        "Median length": float(row["median_trace_length"]),
        "Max length": int(row["max_trace_length"]),
        "Cases complete (%)": 100 * float(row["complete_case_share"]),
        "Variants complete (%)": 100 * float(row["complete_variant_share"]),
        "Occurrences from complete cases (%)": 100 * float(row["complete_occurrence_share"]),
        "Excess occurrences per case": float(row["excess_occurrences_per_case"]),
    }


def _k_table_median_row(block: pd.DataFrame) -> dict[str, object]:
    usable = block[block["status"] == STATUS_OK]
    return {
        "Log": MEDIAN_ROW_LABEL,
        "Cases": pd.NA,
        "Trace variants": pd.NA,
        "Median length": pd.NA,
        "Max length": pd.NA,
        "Cases complete (%)": _percent_points(_unweighted_median(usable["complete_case_share"])),
        "Variants complete (%)": _percent_points(
            _unweighted_median(usable["complete_variant_share"])
        ),
        "Occurrences from complete cases (%)": _percent_points(
            _unweighted_median(usable["complete_occurrence_share"])
        ),
        "Excess occurrences per case": _unweighted_median(
            usable["excess_occurrences_per_case"]
        ),
    }


def _unweighted_median(values: Iterable[object]) -> float | None:
    usable = [float(value) for value in values if pd.notna(value)]
    if not usable:
        return None
    return float(np.median(usable))


def _percent_points(share: float | None) -> float:
    if share is None or pd.isna(share):
        return pd.NA
    return 100 * float(share)


def _shown_percent_value(share: object) -> float:
    """One-decimal percentage, matching the cross-log cell text."""
    return float(f"{100 * float(share):.1f}")


def _format_percent_cell(share: object) -> str:
    if share is None or pd.isna(share):
        return ""
    return f"{_shown_percent_value(share):.1f}%"


def _format_excess_cell(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _format_threshold_cell(n_meeting: int, n_logs: int) -> str:
    if n_logs <= 0:
        return ""
    percent = 100 * n_meeting / n_logs
    return f"{n_meeting}/{n_logs} ({percent:.1f}%)"


def _join_notes(existing: object, extra: str) -> str:
    text = "" if existing is None or pd.isna(existing) else str(existing).strip()
    if not text:
        return extra
    if text.endswith("."):
        return f"{text} {extra}"
    return f"{text}. {extra}"


def _log_order(frame: pd.DataFrame) -> list[str]:
    seen = []
    for log_name in frame["log_name"]:
        if log_name not in seen:
            seen.append(str(log_name))
    return seen


def _coerce_columns(frame: pd.DataFrame) -> pd.DataFrame:
    coerced = frame.copy()
    for column in _COUNT_COLUMNS:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype("Int64")
    for column in _SHARE_COLUMNS:
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    return coerced
