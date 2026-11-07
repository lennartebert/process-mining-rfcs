"""Cross-log saturation of trace n-grams at selected orders.

Trace length and variant identity follow attachment extraction
(``trace_completion_data`` and the activity classifier). No power-law fit
or goodness-of-fit test is run. For order k >= 2 the extractor wraps each
trace as (START, activities, END) and emits one node when k >= L + 2, or
L + 3 - k sliding windows otherwise. That is ``max(1, L + 3 - k)``
occurrences per case.
"""

from __future__ import annotations

import gzip
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ATTACHMENTS_DIR,
    DATA_DICTIONARY_PATH,
)
from utils.powerlaw import LOWER_BOUNDED_POWER_LAW

SATURATION_KS = (5, 10, 15, 20)
REAL_EXPERIMENT_DIRECTORIES = frozenset({"real", "real_xmin1", "real_xmin2"})
STATUS_OK = "ok"
STATUS_MISSING = "missing_log"
STATUS_EMPTY = "empty_log"
MEDIAN_ROW_LABEL = "Median across logs"

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

METHODS_NOTE = """\
Cross-log saturation of trace n-grams.

Trace length L counts activity events and excludes the START and END markers
that extraction adds for orders k >= 2. N is the number of traces attachment
extraction keeps (non-empty traces with a completion timestamp). V is the
number of distinct activity sequences among those traces. Median length is
case-weighted: each case counts once. The median row is the unweighted median
across logs, not a pooled percentage. Case counts, variant counts, and
trace-length columns are left blank on that row.

Every real-world study log is included. The requested xmin setting, fitted-type
cutoff, and GOF threshold do not enter these counts. A log that cannot be read,
or that has no usable traces, is left unknown rather than filled with zeros.

At order k a case is complete when L + 2 <= k, and a variant is complete when
its length meets the same bound. Each case contributes max(1, L + 3 - k)
occurrences. A complete trace contributes one occurrence per case, and that
contribution stays one at every higher order. Excess occurrences per case
equals (O_k - N) / N. A high share of complete cases does not mean that most
extracted occurrences come from complete cases, because an incomplete case
contributes more than one occurrence.
"""


@dataclass(frozen=True)
class LengthProfile:
    """Per-case and per-variant lengths for one study log."""

    log_name: str
    status: str
    notes: str
    case_lengths: tuple[int, ...] = ()
    variant_lengths: tuple[int, ...] = ()


def occurrence_weight(length: int, k: int) -> int:
    """Occurrences one case of activity-length ``length`` contributes at order k."""
    if k < 2:
        raise ValueError(f"occurrence weight is defined for wrapped n-grams, got k={k}")
    return max(1, int(length) + 3 - int(k))


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
        "max_complete_trace_length": int(k) - 2,
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
    weights = np.fromiter(
        (occurrence_weight(int(length), k) for length in lengths),
        dtype=int,
        count=n_cases,
    )
    n_complete_cases = int(np.count_nonzero(lengths + 2 <= k))
    n_complete_variants = int(np.count_nonzero(variants + 2 <= k))
    total = int(weights.sum())
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
            if int(row["max_complete_trace_length"]) != int(k) - 2:
                raise AssertionError(f"{log_name} k={k} completion threshold disagrees")
            max_length = int(row["max_trace_length"])
            if int(k) >= max_length + 2:
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


def format_saturation_results(
    frame: pd.DataFrame, *, ks: Sequence[int] = SATURATION_KS
) -> str:
    """Four readable tables, one per order, in the frame's log order."""
    lines = [METHODS_NOTE.rstrip(), ""]
    log_order = _log_order(frame)
    for k in ks:
        block = frame[frame["k"] == int(k)].set_index("log_name", drop=False)
        block = block.loc[log_order]
        lines.append(_format_k_table(block, int(k)))
        lines.append("")
    lines.append(_extraction_footer(frame))
    return "\n".join(lines).rstrip() + "\n"


def write_saturation_diagnostics(output_dir: Path) -> Path | None:
    """Write saturation tables for a real-log experiment directory.

    The counts do not depend on xmin or on the GOF grid. They are written
    into the experiment directory that combine is merging.
    """
    output_dir = Path(output_dir)
    if output_dir.name not in REAL_EXPERIMENT_DIRECTORIES:
        print(
            "Skipping saturation diagnostics "
            f"({output_dir.name} is not a real-log xmin experiment)"
        )
        return None
    print(f"Saturation diagnostics: loading study logs for {output_dir.name}")
    profiles = load_study_profiles(ALL_REAL_LOG_DATASETS)
    extracted = load_extracted_occurrences(
        output_dir, [profile.log_name for profile in profiles], SATURATION_KS
    )
    frame = build_saturation_frame(profiles, extracted)
    validate_saturation_frame(frame)
    destination = output_dir / "saturation"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "saturation_summary.csv"
    text_path = destination / "saturation_results.txt"
    frame.to_csv(csv_path, index=False)
    text_path.write_text(format_saturation_results(frame), encoding="utf-8")
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
    path = output_dir / f"n{k}" / LOWER_BOUNDED_POWER_LAW / "summary.csv"
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


def _format_k_table(block: pd.DataFrame, k: int) -> str:
    threshold = k - 2
    caption = (
        f"k = {k}. A case or variant is complete when L <= {threshold} "
        f"(L + 2 <= {k}). Complete traces contribute one occurrence per case "
        "and remain unchanged at higher orders. Excess occurrences per case "
        "equals (O_k - N) / N. High case-completion coverage does not imply "
        "that most extracted occurrences come from complete cases."
    )
    headers = [
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
    numeric = [False, True, True, True, True, True, True, True, True]
    body = [_display_row(row) for _, row in block.iterrows()]
    body.append(_median_row(block))
    lines = [caption, _align_table(headers, body, numeric)]
    unavailable = block[block["status"] != STATUS_OK]
    if not unavailable.empty:
        lines.append("Unavailable logs (values left unknown):")
        for _, row in unavailable.iterrows():
            lines.append(f"- {row['log_name']}: {row['status']}. {row['notes']}")
    return "\n".join(lines)


def _display_row(row: pd.Series) -> list[str]:
    if row["status"] != STATUS_OK:
        blank = [""] * 8
        return [str(row["log_name"]), *blank]
    return [
        str(row["log_name"]),
        _format_count(row["n_cases"]),
        _format_count(row["n_variants"]),
        _format_length(row["median_trace_length"]),
        _format_count(row["max_trace_length"]),
        _format_percent(row["complete_case_share"]),
        _format_percent(row["complete_variant_share"]),
        _format_percent(row["complete_occurrence_share"]),
        _format_excess(row["excess_occurrences_per_case"]),
    ]


def _median_row(block: pd.DataFrame) -> list[str]:
    usable = block[block["status"] == STATUS_OK]
    return [
        MEDIAN_ROW_LABEL,
        "",
        "",
        "",
        "",
        _format_percent(_unweighted_median(usable["complete_case_share"])),
        _format_percent(_unweighted_median(usable["complete_variant_share"])),
        _format_percent(_unweighted_median(usable["complete_occurrence_share"])),
        _format_excess(_unweighted_median(usable["excess_occurrences_per_case"])),
    ]


def _align_table(
    headers: Sequence[str], rows: Sequence[Sequence[str]], numeric: Sequence[bool]
) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(cells: Sequence[str]) -> str:
        padded = []
        for index, cell in enumerate(cells):
            if numeric[index]:
                padded.append(cell.rjust(widths[index]))
            else:
                padded.append(cell.ljust(widths[index]))
        return "  ".join(padded).rstrip()

    rule = ["-" * width for width in widths]
    lines = [render(headers), render(rule)]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def _extraction_footer(frame: pd.DataFrame) -> str:
    ok = frame[frame["status"] == STATUS_OK]
    unavailable = ok[ok["extracted_occurrences"].isna()]
    if unavailable.empty and not ok.empty:
        return (
            "Formula occurrence totals match the saved extracted totals "
            "for every study log and order where a saved total exists."
        )
    if unavailable.empty:
        return "No study log had a usable trace set, so no extracted total was compared."
    labels = [
        f"{row.log_name} k={int(row.k)}"
        for row in unavailable.itertuples(index=False)
    ]
    return (
        "Extracted occurrence totals were unavailable for: " + ", ".join(labels) + "."
    )


def _unweighted_median(values: Iterable[object]) -> float | None:
    usable = [float(value) for value in values if pd.notna(value)]
    if not usable:
        return None
    return float(np.median(usable))


def _format_count(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def _format_length(value: object) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.1f}"


def _format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{100 * float(value):.1f}"


def _format_excess(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


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
