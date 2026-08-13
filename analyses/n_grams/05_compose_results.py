"""Step 04: compose n-gram Clauset results into tables and plots.

Reads ``<concept>/<power-law-model>/`` summaries under the n-grams results root
and writes tables/plots at that same root (default: ``results/n_grams/``):

- variant power-law table (CSV + LaTeX)
- per-log RFC / PDF / CCDF plots for variants
- log x n fitted-types matrix (CSV)
- log x n scaling matrix (CSV + LaTeX with legend; gray cells when fitted types < 30)
- per-scale PL outcome summary counts (CSV + LaTeX) and stacked-area chart (PDF)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.powerlaw import (
    DEFAULT_MINIMUM_FITTED_TYPES,
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
)
from utils.constants import N_GRAMS_DIR
from utils.io.attachments import NGRAM_CONCEPTS

VARIANT_TABLE_COLUMNS = [
    "Log",
    "alpha",
    "x_min",
    "fitted types",
    "GOF p",
    "best alternative",
    "LLR",
    "p",
    "Classification",
]

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]

SCALING_LEGEND = (
    "? = missing data / error; "
    "Rejected = PL rejected; "
    "Other better = PL plausible but other significantly better; "
    "PL plausible = PL plausible but other cannot be rejected; "
    "PL best = PL plausible and significantly better than others"
)
SCALING_GRAY_FOOTNOTE = (
    f"Gray cells: fewer than {DEFAULT_MINIMUM_FITTED_TYPES} fitted types"
)

# Compact scaling-cell label -> outcome-summary / stacked-area series.
# Inconclusive ``PL plausible`` is folded into ``PL best`` so the four series
# sum to the number of logs (GOF passed and alternatives not preferred).
PL_OUTCOME_COLUMNS = [
    "PL rejected",
    "PL plausible but other better",
    "PL best",
    "Missing data",
]
_SCALING_CELL_TO_OUTCOME = {
    "Rejected": "PL rejected",
    "Other better": "PL plausible but other better",
    "PL plausible": "PL best",
    "PL best": "PL best",
    "?": "Missing data",
}
_PL_OUTCOME_STACK_COLORS = (
    "#1a1a1a",  # rejected — near black
    "#666666",  # other better
    "#b0b0b0",  # PL best
    "#e8e8e8",  # missing — light gray
)
_PL_OUTCOME_STACK_HATCHES = (
    "...",  # rejected
    "///",  # other better
    "xxx",  # PL best
    "",  # missing (solid light fill)
)


def scaling_columns_for_concepts(concepts: Sequence[str]) -> list[str]:
    """Map selected concepts to scaling-matrix columns (n-grams + optional trace)."""
    columns: list[str] = []
    for concept in NGRAM_CONCEPTS:
        if concept in concepts:
            columns.append(concept)
    if "variants" in concepts:
        columns.append("trace")
    return columns


def concept_for_scaling_column(column: str) -> str:
    return "variants" if column == "trace" else column



def escape_latex(value: object) -> str:
    """Escape LaTeX special characters in a cell value."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _optional_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _optional_int(value: object) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _summary_path(tests_root: Path, concept: str, model: str) -> Path:
    return tests_root / concept / model / "summary.csv"


def _comparison_path(tests_root: Path, concept: str, model: str) -> Path:
    return tests_root / concept / model / "comparison.csv"


def _best_alternative_llr_p(
    comparison_df: pd.DataFrame | None,
    *,
    log_name: str,
    best_other: str | None,
    model: str,
) -> tuple[float | None, float | None]:
    if comparison_df is None or not best_other:
        return None, None
    rows = comparison_df
    if "log_name" in rows.columns:
        rows = rows.loc[rows["log_name"].astype(str) == str(log_name)]
    mask = (rows["model_2"].astype(str) == str(best_other)) & (
        rows["model_1"].astype(str) == str(model)
    )
    matched = rows.loc[mask]
    if matched.empty:
        # Some rows may only store model_2 as the alternative.
        matched = rows.loc[rows["model_2"].astype(str) == str(best_other)]
    if matched.empty:
        return None, None
    row = matched.iloc[0]
    return _optional_float(row.get("R")), _optional_float(row.get("p"))


def build_variant_power_law_table(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the trace-variant power-law summary table."""
    summary = _read_csv_if_exists(_summary_path(tests_root, "variants", model))
    comparison = _read_csv_if_exists(_comparison_path(tests_root, "variants", model))
    if summary is None or summary.empty:
        return pd.DataFrame(columns=VARIANT_TABLE_COLUMNS)

    rows: list[dict[str, Any]] = []
    for _, src in summary.iterrows():
        log_name = str(src["log_name"])
        if datasets is not None and log_name not in datasets:
            continue
        best_other = src.get("best_other_distribution")
        if best_other is not None and not (isinstance(best_other, float) and pd.isna(best_other)):
            best_other_str = str(best_other)
        else:
            best_other_str = None
        llr, p_val = _best_alternative_llr_p(
            comparison,
            log_name=log_name,
            best_other=best_other_str,
            model=model,
        )
        rows.append(
            {
                "Log": log_name,
                "alpha": src.get("alpha"),
                "x_min": src.get("xmin"),
                "fitted types": src.get("n_fitted_types"),
                "GOF p": src.get("gof_p"),
                "best alternative": best_other_str if best_other_str else pd.NA,
                "LLR": llr if llr is not None else pd.NA,
                "p": p_val if p_val is not None else pd.NA,
                "Classification": src.get("classification"),
            }
        )
    return pd.DataFrame(rows, columns=VARIANT_TABLE_COLUMNS)


def classification_to_scaling_cell(classification: object) -> str:
    """Map a Clauset classification string to a compact scaling-matrix label."""
    if classification is None or (isinstance(classification, float) and pd.isna(classification)):
        return "?"
    text = str(classification).strip()
    if not text:
        return "?"
    lowered = text.lower()
    if (
        "invalid" in lowered
        or "gof unavailable" in lowered
        or "unresolved" in lowered
        or "error" in lowered
    ):
        return "?"
    if "not plausible" in lowered:
        return "Rejected"
    if "plausible" in lowered and "alternatives preferred" in lowered:
        return "Other better"
    if "plausible" in lowered and "preferred over alternatives" in lowered:
        return "PL best"
    if "plausible" in lowered and "inconclusive" in lowered:
        return "PL plausible"
    if "plausible" in lowered:
        return "PL plausible"
    return "?"


def scale_label_for_column(column: str) -> str:
    """Human-readable scale label: ``n2`` -> ``n=2``, ``trace`` -> ``variant``."""
    if column == "trace":
        return "variant"
    if column.startswith("n") and column[1:].isdigit():
        return f"n={column[1:]}"
    return column


def build_pl_outcome_counts(scaling_df: pd.DataFrame) -> pd.DataFrame:
    """Absolute per-scale outcome counts (rows sum to the number of logs)."""
    value_cols = [c for c in scaling_df.columns if c != "Log"]
    rows: list[dict[str, object]] = []
    for column in value_cols:
        counts = {name: 0 for name in PL_OUTCOME_COLUMNS}
        for value in scaling_df[column]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                cell = "?"
            else:
                cell = str(value).strip() or "?"
            outcome = _SCALING_CELL_TO_OUTCOME.get(cell, "Missing data")
            counts[outcome] += 1
        row: dict[str, object] = {"Scale": scale_label_for_column(column)}
        row.update(counts)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Scale", *PL_OUTCOME_COLUMNS])


def build_pl_outcome_summary(scaling_df: pd.DataFrame) -> pd.DataFrame:
    """Count scaling-matrix outcomes per scale as ``count/total`` fractions."""
    counts_df = build_pl_outcome_counts(scaling_df)
    total = int(len(scaling_df))
    denom = str(total)
    rows: list[dict[str, str]] = []
    for _, src in counts_df.iterrows():
        row = {"Scale": str(src["Scale"])}
        for name in PL_OUTCOME_COLUMNS:
            row[name] = f"{int(src[name])}/{denom}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["Scale", *PL_OUTCOME_COLUMNS])


def plot_pl_outcome_stacked_area(
    counts_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Stacked area chart of outcome counts across n-gram scales up to variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if counts_df.empty:
        return

    scales = counts_df["Scale"].astype(str).tolist()
    x = np.arange(len(scales), dtype=float)
    series = [
        counts_df[name].to_numpy(dtype=float) for name in PL_OUTCOME_COLUMNS
    ]
    total = float(np.max(np.sum(np.vstack(series), axis=0))) if series else 0.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    # Hatches need a backend that draws edges; keep grayscale fills + patterns.
    stacks = ax.stackplot(
        x,
        *series,
        colors=_PL_OUTCOME_STACK_COLORS,
        alpha=0.95,
    )
    for poly, hatch in zip(stacks, _PL_OUTCOME_STACK_HATCHES):
        poly.set_hatch(hatch)
        poly.set_edgecolor("#000000")
        poly.set_linewidth(0.4)
    ax.set_xlim(0, max(len(scales) - 1, 0))
    ax.set_xticks(x)
    ax.set_xticklabels(scales, rotation=45, ha="right")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Number of logs")
    if total > 0:
        ax.set_ylim(0, total)
        ax.set_yticks(np.arange(0, total + 1, max(1, int(total // 6) or 1)))
    ax.grid(True, axis="y", alpha=0.3)
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="#000000",
            hatch=hatch,
            label=label,
            linewidth=0.4,
        )
        for color, hatch, label in zip(
            _PL_OUTCOME_STACK_COLORS,
            _PL_OUTCOME_STACK_HATCHES,
            PL_OUTCOME_COLUMNS,
        )
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def write_pl_outcome_summary(
    scaling_df: pd.DataFrame,
    *,
    compose_dir: Path,
) -> pd.DataFrame:
    """Build and write per-scale PL outcome summary (CSV + LaTeX + stacked area)."""
    counts_df = build_pl_outcome_counts(scaling_df)
    total = int(len(scaling_df))
    denom = str(total)
    summary_rows: list[dict[str, str]] = []
    for _, src in counts_df.iterrows():
        row = {"Scale": str(src["Scale"])}
        for name in PL_OUTCOME_COLUMNS:
            row[name] = f"{int(src[name])}/{denom}"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows, columns=["Scale", *PL_OUTCOME_COLUMNS])
    write_table_csv_and_tex(
        summary_df,
        compose_dir / "pl_outcome_summary.csv",
        compose_dir / "pl_outcome_summary.tex",
        caption="Power-law outcome counts by scale",
    )
    plot_pl_outcome_stacked_area(
        counts_df,
        compose_dir / "pl_outcome_summary.pdf",
    )
    return summary_df


def _load_concept_summaries(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None,
    concepts: Sequence[str],
) -> tuple[list[str], dict[str, pd.DataFrame], list[str]]:
    """Load per-concept summaries keyed by scaling column name."""
    scaling_columns = scaling_columns_for_concepts(concepts)
    per_concept: dict[str, pd.DataFrame] = {}
    logs: set[str] = set()
    for column in scaling_columns:
        concept = concept_for_scaling_column(column)
        summary = _read_csv_if_exists(_summary_path(tests_root, concept, model))
        if summary is None or summary.empty:
            continue
        if datasets is not None:
            summary = summary.loc[summary["log_name"].astype(str).isin(datasets)]
        per_concept[column] = summary.set_index(summary["log_name"].astype(str))
        logs.update(per_concept[column].index.astype(str).tolist())
    ordered_logs = sorted(logs) if datasets is None else [d for d in datasets if d in logs]
    return ordered_logs, per_concept, scaling_columns


def build_log_n_scaling_table(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build log x n scaling matrix with compact cell labels."""
    selected = list(concepts) if concepts is not None else list(DEFAULT_CONCEPTS)
    ordered_logs, per_concept, scaling_columns = _load_concept_summaries(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=selected,
    )
    rows: list[dict[str, str]] = []
    for log_name in ordered_logs:
        row: dict[str, str] = {"Log": log_name}
        for column in scaling_columns:
            summary = per_concept.get(column)
            if summary is None or log_name not in summary.index:
                row[column] = "?"
                continue
            classification = summary.loc[log_name, "classification"]
            if isinstance(classification, pd.Series):
                classification = classification.iloc[0]
            row[column] = classification_to_scaling_cell(classification)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Log", *scaling_columns])


def build_log_n_fitted_types_table(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build log x n matrix of fitted-type counts (or ``?`` if missing)."""
    selected = list(concepts) if concepts is not None else list(DEFAULT_CONCEPTS)
    ordered_logs, per_concept, scaling_columns = _load_concept_summaries(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=selected,
    )
    rows: list[dict[str, str]] = []
    for log_name in ordered_logs:
        row: dict[str, str] = {"Log": log_name}
        for column in scaling_columns:
            summary = per_concept.get(column)
            if summary is None or log_name not in summary.index:
                row[column] = "?"
                continue
            value = summary.loc[log_name, "n_fitted_types"]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            n_fitted = _optional_int(value)
            row[column] = "?" if n_fitted is None else str(n_fitted)
        rows.append(row)
    return pd.DataFrame(rows, columns=["Log", *scaling_columns])


def _dataframe_to_latex_escaped(df: pd.DataFrame) -> str:
    """Render a DataFrame as LaTeX with escaped cell text."""
    escaped = df.copy()
    for col in escaped.columns:
        escaped[col] = escaped[col].map(escape_latex)
    # escape=True would double-escape; values are already escaped.
    return escaped.to_latex(index=False, escape=False)


def _gray_latex_cell(text: str) -> str:
    return f"\\textcolor{{gray}}{{{text}}}"


def scaling_dataframe_to_latex(
    scaling_df: pd.DataFrame,
    fitted_types_df: pd.DataFrame,
    *,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
) -> str:
    """Render scaling matrix LaTeX, graying cells with fitted types below threshold."""
    fitted_indexed = (
        fitted_types_df.set_index("Log")
        if "Log" in fitted_types_df.columns
        else fitted_types_df
    )
    escaped = scaling_df.copy()
    value_cols = [c for c in escaped.columns if c != "Log"]
    for idx, row in escaped.iterrows():
        log_name = str(row["Log"])
        escaped.at[idx, "Log"] = escape_latex(log_name)
        for col in value_cols:
            cell = escape_latex(row[col])
            n_fitted = None
            if log_name in fitted_indexed.index and col in fitted_indexed.columns:
                n_fitted = _optional_int(fitted_indexed.loc[log_name, col])
            if n_fitted is not None and n_fitted < minimum_fitted_types:
                cell = _gray_latex_cell(cell)
            escaped.at[idx, col] = cell
    return escaped.to_latex(index=False, escape=False)


def write_table_csv(df: pd.DataFrame, csv_path: Path) -> None:
    """Write a DataFrame to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def write_table_tex(
    df: pd.DataFrame,
    tex_path: Path,
    *,
    caption: str | None = None,
    legend: str | None = None,
    latex_body: str | None = None,
    footnote: str | None = None,
) -> None:
    """Write a DataFrame (or pre-rendered body) as escaped LaTeX."""
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    latex = latex_body if latex_body is not None else _dataframe_to_latex_escaped(df)
    parts = []
    if caption:
        parts.append(f"% {caption}")
    parts.append(latex.rstrip())
    if legend:
        parts.append("% Legend:")
        parts.append(f"% {legend}")
        parts.append(
            "\\begin{flushleft}\\footnotesize "
            + escape_latex(legend)
            + "\\end{flushleft}"
        )
    if footnote:
        parts.append(f"% {footnote}")
        parts.append(
            "\\begin{flushleft}\\footnotesize "
            + escape_latex(footnote)
            + "\\end{flushleft}"
        )
    tex_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Saved: {tex_path}")


def write_table_csv_and_tex(
    df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    *,
    caption: str | None = None,
    legend: str | None = None,
    latex_body: str | None = None,
    footnote: str | None = None,
) -> None:
    write_table_csv(df, csv_path)
    write_table_tex(
        df,
        tex_path,
        caption=caption,
        legend=legend,
        latex_body=latex_body,
        footnote=footnote,
    )


def write_log_n_scaling_outputs(
    scaling_df: pd.DataFrame,
    fitted_types_df: pd.DataFrame,
    *,
    compose_dir: Path,
) -> None:
    """Write fitted-types CSV, scaling CSV + LaTeX, and PL outcome summary."""
    write_table_csv(fitted_types_df, compose_dir / "log_n_fitted_types.csv")
    write_table_csv_and_tex(
        scaling_df,
        compose_dir / "log_n_scaling.csv",
        compose_dir / "log_n_scaling.tex",
        caption="Log x n power-law scaling",
        legend=SCALING_LEGEND,
        latex_body=scaling_dataframe_to_latex(scaling_df, fitted_types_df),
        footnote=SCALING_GRAY_FOOTNOTE,
    )
    write_pl_outcome_summary(scaling_df, compose_dir=compose_dir)


def write_variant_plots(
    tests_root: Path,
    plots_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
) -> None:
    summary = _read_csv_if_exists(_summary_path(tests_root, "variants", model))
    if summary is None or summary.empty:
        print("Warning: no variants summary found; skipping plots.")
        return

    for _, src in summary.iterrows():
        log_name = str(src["log_name"])
        if datasets is not None and log_name not in datasets:
            continue
        input_path = Path(str(src["input_path"]))
        if not input_path.exists():
            print(f"Warning: attachments missing for {log_name}: {input_path}")
            continue

        plot_dir = plots_root / log_name
        plot_dir.mkdir(parents=True, exist_ok=True)
        xmin = _optional_float(src.get("xmin"))
        xmax = _optional_float(src.get("xmax"))
        best_other = src.get("best_other_distribution")
        if best_other is not None and not (isinstance(best_other, float) and pd.isna(best_other)):
            alt_name = str(best_other)
        else:
            alt_name = None

        # Isolate matplotlib savefig in a child process: some Windows/env setups
        # segfault inside Agg savefig and would otherwise abort the whole compose.
        script = f"""
import sys
from pathlib import Path
sys.path.insert(0, {str(REPO_ROOT)!r})
from utils.io import load_attachments
import powerlaw
from utils.rfc import extract_frequency_counts
from utils.case_attribute.plotting import plot_attribute_rfc_loglog
from utils.powerlaw.plotting import (
    plot_fit_ccdf_with_alternative,
    plot_fit_pdf_with_alternative,
)
frequencies = extract_frequency_counts(load_attachments(Path({str(input_path)!r})))
plot_dir = Path({str(plot_dir)!r})
plot_attribute_rfc_loglog(frequencies, plot_dir / "rfc_loglog.pdf", title={log_name!r} + " RFC")
xmin = {xmin!r}
xmax = {xmax!r}
model = {model!r}
alt_name = {alt_name!r}
if model == {DOUBLY_BOUNDED_POWER_LAW!r}:
    fit = powerlaw.Fit(frequencies, discrete=True, xmin=xmin, xmax=xmax, verbose=False, parameter_ranges={{"alpha": [0.0, 4.0]}})
elif model == {LOWER_BOUNDED_POWER_LAW!r}:
    fit = powerlaw.Fit(frequencies, discrete=True, xmin=xmin, xmax=None, verbose=False, parameter_ranges={{"alpha": [0.0, 4.0]}})
else:
    fit = powerlaw.Fit(frequencies, discrete=True, xmin=1.0, xmax=None, verbose=False, parameter_ranges={{"alpha": [0.0, 4.0]}})
if fit is not None:
    plot_fit_pdf_with_alternative(
        frequencies, plot_dir / "pdf_loglog.pdf", fit=fit,
        alternative_name=alt_name, title={log_name!r} + " PDF",
    )
    plot_fit_ccdf_with_alternative(
        frequencies, plot_dir / "ccdf_loglog.pdf", fit=fit,
        alternative_name=alt_name, title={log_name!r} + " CCDF",
    )
print("plots-ok")
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            print(
                f"Warning: plotting failed for {log_name} "
                f"(exit={completed.returncode}). stderr={completed.stderr[-500:]}"
            )
        else:
            print(f"Saved plots under: {plot_dir}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose n-gram Clauset results into tables and plots"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional log filter (default: all logs present in summaries)",
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=DEFAULT_CONCEPTS,
        choices=CONCEPT_CHOICES,
        help="Concepts to include in compose tables/plots (default: n1..n10 + variants)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"N-grams results root (default: {N_GRAMS_DIR})",
    )
    parser.add_argument(
        "--tests-root",
        type=str,
        default=None,
        help="Root containing <concept>/<model>/ summaries (default: <output-dir>)",
    )
    parser.add_argument(
        "--power-law-model",
        type=str,
        default=LOWER_BOUNDED_POWER_LAW,
        choices=list(DISTRIBUTION_NAMES),
        help="Power-law family folder under <concept>/",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else N_GRAMS_DIR
    tests_root = Path(args.tests_root) if args.tests_root else output_root
    compose_dir = output_root
    compose_dir.mkdir(parents=True, exist_ok=True)
    model = args.power_law_model
    datasets = list(args.datasets) if args.datasets else None
    concepts = list(args.concepts)

    print(f"Composing from: {tests_root} (model={model}, concepts={concepts})")

    if "variants" in concepts:
        variant_df = build_variant_power_law_table(
            tests_root,
            model=model,
            datasets=datasets,
        )
        write_table_csv_and_tex(
            variant_df,
            compose_dir / "variant_power_law.csv",
            compose_dir / "variant_power_law.tex",
            caption="Trace variant power-law results",
        )
        write_variant_plots(
            tests_root,
            compose_dir / "plots",
            model=model,
            datasets=datasets,
        )
    else:
        print("Skipping variant table/plots (variants not in --concepts)")

    fitted_types_df = build_log_n_fitted_types_table(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=concepts,
    )
    scaling_df = build_log_n_scaling_table(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=concepts,
    )
    write_log_n_scaling_outputs(
        scaling_df,
        fitted_types_df,
        compose_dir=compose_dir,
    )
    print(f"Compose outputs written under: {compose_dir}")


if __name__ == "__main__":
    main()
