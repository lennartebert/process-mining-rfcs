"""Step 04: compose n-gram Clauset results into tables and plots.

Reads ``<concept>/<power-law-model>/`` summaries under the n-grams results root
and writes tables/plots at that same root (default: ``results/n_grams/``):

- variant power-law table (CSV + LaTeX)
- per-log RFC / PDF / CCDF plots for variants
- log x n scaling matrix (CSV + LaTeX with legend)
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

from utils.constants import N_GRAMS_DIR
from utils.io.attachments import NGRAM_CONCEPTS
from utils.rfc import (
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
)

VARIANT_TABLE_COLUMNS = [
    "Log",
    "alpha",
    "x_min",
    "fitted variants",
    "GOF p",
    "best alternative",
    "LLR",
    "p",
    "Classification",
]

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]

SCALING_LEGEND = (
    "? = insufficient evidence / unresolved; "
    "Rejected = PL rejected; "
    "Other better = PL plausible but other significantly better; "
    "PL plausible = PL plausible but other cannot be rejected; "
    "PL best = PL plausible and significantly better than others"
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
                "fitted variants": src.get("n_fitted_variants"),
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
        "insufficient" in lowered
        or "invalid" in lowered
        or "gof unavailable" in lowered
        or "unresolved" in lowered
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


def build_log_n_scaling_table(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build log x n scaling matrix with compact cell labels."""
    selected = list(concepts) if concepts is not None else list(DEFAULT_CONCEPTS)
    scaling_columns = scaling_columns_for_concepts(selected)
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


def _dataframe_to_latex_escaped(df: pd.DataFrame) -> str:
    """Render a DataFrame as LaTeX with escaped cell text."""
    escaped = df.copy()
    for col in escaped.columns:
        escaped[col] = escaped[col].map(escape_latex)
    # escape=True would double-escape; values are already escaped.
    return escaped.to_latex(index=False, escape=False)


def write_table_csv_and_tex(
    df: pd.DataFrame,
    csv_path: Path,
    tex_path: Path,
    *,
    caption: str | None = None,
    legend: str | None = None,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    latex = _dataframe_to_latex_escaped(df)
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
    tex_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


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
from utils.rfc import extract_frequency_counts, fit_discrete_power_law
from utils.rfc.case_attribute_plotting import (
    plot_attribute_rfc_loglog,
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
    fit_result = fit_discrete_power_law(frequencies, xmin=xmin, xmax=xmax)
elif model == {LOWER_BOUNDED_POWER_LAW!r}:
    fit_result = fit_discrete_power_law(frequencies, xmin=xmin, xmax=None)
else:
    fit_result = fit_discrete_power_law(frequencies, xmin=1.0, xmax=None)
fit = fit_result.get("fit")
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
        import subprocess

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
            tests_root, model=model, datasets=datasets
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

    scaling_df = build_log_n_scaling_table(
        tests_root, model=model, datasets=datasets, concepts=concepts
    )
    write_table_csv_and_tex(
        scaling_df,
        compose_dir / "log_n_scaling.csv",
        compose_dir / "log_n_scaling.tex",
        caption="Log x n power-law scaling",
        legend=SCALING_LEGEND,
    )
    print(f"Compose outputs written under: {compose_dir}")


if __name__ == "__main__":
    main()
