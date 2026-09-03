"""Step 04: compose n-gram Clauset results into tables and plots.

Reads ``<concept>/<power-law-model>/`` summaries under the power-law statistics results root
and writes tables to a categorization subfolder
(default: ``results/powerlaw_statistics/real/min-types-30_gof-p-0.1_comparison-p-0.1``).
Per-log RFC / PDF / CCDF plots are independent of categorization and always
land under ``<output-dir>/plots/<log>/`` (use ``--skip-plots`` to omit them).

- variant power-law table (CSV + LaTeX)
- per-log RFC / PDF / CCDF plots for variants (shared ``plots/``)
- log x n fitted-types matrix (CSV)
- log x n scaling matrix (CSV + LaTeX with legend)
- per-scale PL outcome summary counts (CSV + LaTeX), stacked-area chart, and category bar chart (PDF)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.powerlaw import (
    CLASSIFICATION_CELL_COLORS,
    CLASSIFICATION_CODES,
    CLASSIFICATION_LABELS,
    DEFAULT_MINIMUM_FITTED_TYPES,
    DEFAULT_SIGNIFICANCE_LEVEL,
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    compact_classification,
    llr_preference,
    select_best_other_distribution,
)
from utils.constants import POWERLAW_STATISTICS_REAL_DIR
from utils.io.attachments import NGRAM_CONCEPTS

VARIANT_TABLE_COLUMNS = [
    "Log",
    "alpha",
    "x_min",
    "fitted types",
    "fitted type %",
    "GOF p",
    "best alternative",
    "R",
    "p",
    "Result",
    "Classification",
]

DEFAULT_CONCEPTS = [*NGRAM_CONCEPTS, "variants"]
CONCEPT_CHOICES = ["variants", "activities", "dfrs", *NGRAM_CONCEPTS]

def _format_threshold(value: float) -> str:
    return f"{value:g}"


def compose_settings_dirname(
    *,
    min_types: int,
    gof_p: float,
    comparison_p: float,
) -> str:
    return (
        f"min-types-{min_types}_"
        f"gof-p-{_format_threshold(gof_p)}_"
        f"comparison-p-{_format_threshold(comparison_p)}"
    )


def build_scaling_legend(
    *,
    minimum_fitted_types: int,
    gof_p: float,
    comparison_p: float,
) -> str:
    gof = _format_threshold(gof_p)
    cmp_p = _format_threshold(comparison_p)
    return (
        f"A = {CLASSIFICATION_LABELS['A']} = fit invalid, GOF missing, or fewer than "
        f"{minimum_fitted_types} fitted types; "
        f"B = {CLASSIFICATION_LABELS['B']} = GOF p<{gof}; "
        f"C = {CLASSIFICATION_LABELS['C']} = GOF p>={gof} and some of Exp/LN/StExp "
        f"significantly better than PL (comparison p<{cmp_p}); "
        f"D = {CLASSIFICATION_LABELS['D']} = GOF p>={gof} and PL significantly better than "
        f"Exp, LN, and StExp (comparison p<{cmp_p}); "
        f"E = {CLASSIFICATION_LABELS['E']} = GOF p>={gof} but no considered "
        "alternative significantly outperforms PL and PL does not significantly "
        "outperform all alternatives. "
        "TPL is excluded from preference."
    )


def build_scaling_tex_footnote() -> str:
    items = ", ".join(
        f"{code} = {CLASSIFICATION_LABELS[code]}" for code in CLASSIFICATION_CODES
    )
    return (
        "\\begin{center}\\footnotesize\n"
        f"Classification: {items}.\n"
        "\\end{center}"
    )


def build_variant_tex_footnote() -> str:
    items = ", ".join(
        f"{code} = {CLASSIFICATION_LABELS[code]}" for code in CLASSIFICATION_CODES
    )
    return (
        "\\begin{center}\\footnotesize\n"
        "Models: LN = Log normal, Exp = Exponential, StExp = Stretched exponential; "
        f"\\quad Category: {items}.\n"
        "\\end{center}"
    )


VARIANT_ALTERNATIVE_ABBREV = {
    "truncated_power_law": "TPL",
    "lognormal": "LN",
    "exponential": "Exp",
    "stretched_exponential": "StExp",
}

# Compact scaling-cell labels (A–E), same set as log_n_scaling.
PL_OUTCOME_COLUMNS = list(CLASSIFICATION_CODES)
_PL_OUTCOME_STACK_COLORS = tuple(
    f"#{CLASSIFICATION_CELL_COLORS[code]}" for code in CLASSIFICATION_CODES
)
BAR_CHART_CODES = [code for code in CLASSIFICATION_CODES if code != "A"]
_BAR_LABEL_TEXT_COLOR = {
    "B": "#FFFFFF",
    "C": "#111111",
    "D": "#FFFFFF",
    "E": "#111111",
}


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


def _comparison_for_log(
    comparison_df: pd.DataFrame | None, log_name: str
) -> pd.DataFrame:
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()
    if "log_name" not in comparison_df.columns:
        return comparison_df
    return comparison_df.loc[comparison_df["log_name"].astype(str) == str(log_name)]


def _compact_from_summary_row(
    src: Mapping[str, Any],
    comparison_df: pd.DataFrame | None,
    *,
    log_name: str,
    minimum_fitted_types: int,
    gof_p_threshold: float,
    comparison_p_threshold: float,
) -> str:
    n_fitted = _optional_int(src.get("n_fitted_types")) or 0
    gof_p = _optional_float(src.get("gof_p"))
    gof_p_f = float("nan") if gof_p is None else gof_p
    fit_valid = src.get("fit_valid", True)
    if not isinstance(fit_valid, bool):
        fit_valid = str(fit_valid).strip().lower() in {"true", "1", "yes"}
    return compact_classification(
        n_fitted_types=n_fitted,
        gof_p=gof_p_f,
        comparison_df=_comparison_for_log(comparison_df, log_name),
        minimum_fitted_types=minimum_fitted_types,
        gof_p_threshold=gof_p_threshold,
        comparison_p_threshold=comparison_p_threshold,
        fit_valid=bool(fit_valid),
    )


def build_variant_power_law_table(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    gof_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
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
        log_comparison = _comparison_for_log(comparison, log_name)
        best_other_str = select_best_other_distribution(log_comparison)
        r_val, p_val = _best_alternative_llr_p(
            log_comparison,
            log_name=log_name,
            best_other=best_other_str,
            model=model,
        )
        share = _optional_float(src.get("fitted_type_share"))
        fitted_pct = pd.NA if share is None else 100.0 * share
        rows.append(
            {
                "Log": log_name,
                "alpha": src.get("alpha"),
                "x_min": src.get("xmin"),
                "fitted types": src.get("n_fitted_types"),
                "fitted type %": fitted_pct,
                "GOF p": src.get("gof_p"),
                "best alternative": best_other_str if best_other_str else pd.NA,
                "R": r_val if r_val is not None else pd.NA,
                "p": p_val if p_val is not None else pd.NA,
                "Result": llr_result_label(
                    log_comparison, comparison_p_threshold=comparison_p_threshold
                ),
                "Classification": _compact_from_summary_row(
                    src,
                    log_comparison,
                    log_name=log_name,
                    minimum_fitted_types=minimum_fitted_types,
                    gof_p_threshold=gof_p_threshold,
                    comparison_p_threshold=comparison_p_threshold,
                ),
            }
        )
    return pd.DataFrame(rows, columns=VARIANT_TABLE_COLUMNS)


def classification_to_scaling_cell(classification: object) -> str:
    """Map a stored classification string to an A–E code (legacy fallback)."""
    fallback = "A"
    if classification is None or (isinstance(classification, float) and pd.isna(classification)):
        return fallback
    text = str(classification).strip()
    if not text:
        return fallback
    if text in CLASSIFICATION_LABELS or text == "?":
        return text
    for code, label in CLASSIFICATION_LABELS.items():
        if text == label:
            return code
    lowered = text.lower()
    if (
        "invalid" in lowered
        or "gof unavailable" in lowered
        or "unresolved" in lowered
        or "error" in lowered
        or "insufficient" in lowered
        or "< 30 fitted types" in lowered
        or ("< 30" in lowered and "fitted types" in lowered)
    ):
        return "A"
    if "not plausible" in lowered:
        return "B"
    if "plausible" in lowered and (
        "alternative preferred" in lowered or "other preferred" in lowered
    ):
        return "C"
    if "plausible" in lowered and (
        "pl best" in lowered
        or "pl preferred" in lowered
        or "preferred over alternatives" in lowered
    ):
        return "D"
    if "plausible" in lowered:
        return "E"
    return fallback


def _format_tex_percent(value: object, digits: int = 1) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _format_tex_decimal(value: object, digits: int = 2) -> str:
    number = _optional_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _format_tex_int(value: object) -> str:
    number = _optional_int(value)
    if number is None:
        float_number = _optional_float(value)
        if float_number is None:
            return ""
        number = int(round(float_number))
    return str(number)


def _yes_no(flag: bool | None) -> str:
    if flag is None:
        return ""
    return "Yes" if flag else "No"


def abbreviate_alternative_model(name: object) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text or text.lower() == "nan":
        return ""
    return VARIANT_ALTERNATIVE_ABBREV.get(text, text)


def llr_result_label(
    comparison_df: pd.DataFrame | None,
    *,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> str:
    """Compact step-3 outcome from counted alternatives."""
    preference = llr_preference(
        comparison_df, significance_level=comparison_p_threshold
    )
    if preference == "alternative preferred":
        return "Alt. preferred"
    if preference == "PL preferred":
        return "PL preferred"
    return "Inconclusive"


def _category_cellcolor_tex(code: str) -> str:
    hex_color = CLASSIFICATION_CELL_COLORS.get(code, CLASSIFICATION_CELL_COLORS["A"])
    return f"\\cellcolor[HTML]{{{hex_color}}} {escape_latex(code)}"


def variant_power_law_dataframe_to_latex(
    df: pd.DataFrame,
    *,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    gof_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> str:
    """Two-row overview header for the trace-variant power-law table."""
    gof = _format_threshold(gof_p_threshold)
    lines = [
        r"\begin{tabular}{lrrrrcrclrrll}",
        r"\toprule",
        (
            r" & \multicolumn{2}{c}{Fitting result}"
            r" & \multicolumn{3}{c}{1. Fitted types}"
            r" & \multicolumn{2}{c}{2. GOF}"
            r" & \multicolumn{4}{c}{3. Best alternative}"
            r" & Category \\"
        ),
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}\cmidrule(lr){7-8}\cmidrule(lr){9-12}",
        (
            f"Log & $\\alpha$ & x-min & \\# & \\% & $\\geq {minimum_fitted_types}$?"
            f" & $p$ & $p \\geq {gof}$ & Model & $R$ & $p$ & Result & \\"
        ),
        r"\midrule",
    ]
    for _, row in df.iterrows():
        n_fitted = _optional_int(row.get("fitted types"))
        gof_p = _optional_float(row.get("GOF p"))
        geq_threshold = None if n_fitted is None else n_fitted >= minimum_fitted_types
        gof_ok = None if gof_p is None else gof_p >= gof_p_threshold
        classification = classification_to_scaling_cell(row.get("Classification"))
        cells = [
            escape_latex(row.get("Log")),
            _format_tex_decimal(row.get("alpha")),
            _format_tex_int(row.get("x_min")),
            _format_tex_int(n_fitted),
            _format_tex_percent(row.get("fitted type %")),
            _yes_no(geq_threshold),
            _format_tex_decimal(gof_p),
            _yes_no(gof_ok),
            escape_latex(abbreviate_alternative_model(row.get("best alternative"))),
            _format_tex_decimal(row.get("R")),
            _format_tex_decimal(row.get("p")),
            escape_latex(row.get("Result")),
            _category_cellcolor_tex(classification),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def bar_chart_scale_label(scale: str) -> str:
    """``n=1`` -> ``k=1``, ``variant`` -> ``trace``."""
    if scale == "variant":
        return "trace"
    if scale.startswith("n="):
        return f"k={scale[2:]}"
    return scale


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
            cell = classification_to_scaling_cell(value)
            if cell not in counts:
                cell = "A"
            counts[cell] += 1
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
    stacks = ax.stackplot(
        x,
        *series,
        colors=_PL_OUTCOME_STACK_COLORS,
        alpha=0.95,
    )
    for poly in stacks:
        poly.set_edgecolor("#333333")
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
            edgecolor="#333333",
            label=f"{code}: {CLASSIFICATION_LABELS[code]}",
            linewidth=0.4,
        )
        for color, code in zip(_PL_OUTCOME_STACK_COLORS, PL_OUTCOME_COLUMNS)
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


def plot_category_bar_chart(
    counts_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """100% stacked bars of non-A categories; n on top is the non-A count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if counts_df.empty:
        return

    codes = BAR_CHART_CODES
    labels = [
        bar_chart_scale_label(str(scale)) for scale in counts_df["Scale"].astype(str)
    ]
    counts = counts_df[codes].to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    percents = np.zeros_like(counts)
    nonzero = totals > 0
    percents[nonzero] = counts[nonzero] / totals[nonzero, None] * 100.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels), dtype=float)
    if len(x) > 1:
        x[-1] += 0.45
    bottoms = np.zeros(len(labels), dtype=float)
    for i, code in enumerate(codes):
        heights = percents[:, i]
        color = f"#{CLASSIFICATION_CELL_COLORS[code]}"
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=color,
            edgecolor="#333333",
            linewidth=0.4,
            width=0.72,
        )
        text_color = _BAR_LABEL_TEXT_COLOR.get(code, "#111111")
        for j, (height, bottom) in enumerate(zip(heights, bottoms)):
            if height < 4.0:
                continue
            ax.text(
                x[j],
                bottom + height / 2.0,
                f"{height:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
        bottoms += heights

    for j, total in enumerate(totals):
        ax.text(
            x[j],
            100.8,
            f"n={int(total)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlim(-0.6, float(x[-1]) + 0.6 if len(x) else 0.6)
    ax.set_ylim(0, 107)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Behavioral representation")
    ax.set_ylabel("Power-law assessment outcomes among eligible logs (%)")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, axis="y", alpha=0.3)
    legend_handles = [
        Patch(
            facecolor=f"#{CLASSIFICATION_CELL_COLORS[code]}",
            edgecolor="#333333",
            label=f"{code}: {CLASSIFICATION_LABELS[code]}",
            linewidth=0.4,
        )
        for code in codes
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


def pl_outcome_summary_dataframe_to_latex(df: pd.DataFrame) -> str:
    """Render PL outcome summary with cell-colored A–E headers and count cells."""
    codes = [c for c in df.columns if c in CLASSIFICATION_LABELS]
    spec = "l" + "c" * len(codes)
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(["Scale", *[_category_cellcolor_tex(code) for code in codes]])
        + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [escape_latex(row.get("Scale"))]
        for code in codes:
            hex_color = CLASSIFICATION_CELL_COLORS[code]
            cells.append(
                f"\\cellcolor[HTML]{{{hex_color}}} {escape_latex(row.get(code))}"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_pl_outcome_summary(
    scaling_df: pd.DataFrame,
    *,
    compose_dir: Path,
    legend: str,
) -> pd.DataFrame:
    """Build and write per-scale PL outcome summary (CSV + LaTeX + stacked area + bar)."""
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
        legend=legend,
        latex_body=pl_outcome_summary_dataframe_to_latex(summary_df),
    )
    plot_pl_outcome_stacked_area(
        counts_df,
        compose_dir / "pl_outcome_summary.pdf",
    )
    plot_category_bar_chart(
        counts_df,
        compose_dir / "category_bar_chart.pdf",
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


def _load_concept_comparisons(
    tests_root: Path,
    *,
    model: str,
    scaling_columns: Sequence[str],
) -> dict[str, pd.DataFrame | None]:
    per_comparison: dict[str, pd.DataFrame | None] = {}
    for column in scaling_columns:
        concept = concept_for_scaling_column(column)
        per_comparison[column] = _read_csv_if_exists(
            _comparison_path(tests_root, concept, model)
        )
    return per_comparison


def build_log_n_scaling_table(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    gof_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> pd.DataFrame:
    """Build log x n scaling matrix with compact cell labels."""
    selected = list(concepts) if concepts is not None else list(DEFAULT_CONCEPTS)
    ordered_logs, per_concept, scaling_columns = _load_concept_summaries(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=selected,
    )
    per_comparison = _load_concept_comparisons(
        tests_root, model=model, scaling_columns=scaling_columns
    )
    rows: list[dict[str, str]] = []
    for log_name in ordered_logs:
        row: dict[str, str] = {"Log": log_name}
        for column in scaling_columns:
            summary = per_concept.get(column)
            if summary is None or log_name not in summary.index:
                row[column] = "?"
                continue
            src = summary.loc[log_name]
            if isinstance(src, pd.DataFrame):
                src = src.iloc[0]
            row[column] = _compact_from_summary_row(
                src,
                per_comparison.get(column),
                log_name=log_name,
                minimum_fitted_types=minimum_fitted_types,
                gof_p_threshold=gof_p_threshold,
                comparison_p_threshold=comparison_p_threshold,
            )
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


def scaling_dataframe_to_latex(scaling_df: pd.DataFrame) -> str:
    """Render scaling matrix LaTeX with colored A–E cells."""
    escaped = scaling_df.copy()
    value_cols = [c for c in escaped.columns if c != "Log"]
    for idx, row in escaped.iterrows():
        log_name = str(row["Log"])
        escaped.at[idx, "Log"] = escape_latex(log_name)
        for col in value_cols:
            raw = row[col]
            if str(raw).strip() == "?":
                escaped.at[idx, col] = escape_latex("?")
                continue
            code = classification_to_scaling_cell(raw)
            escaped.at[idx, col] = _category_cellcolor_tex(code)
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
    legend_tex: str | None = None,
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
    if legend_tex:
        parts.append(legend_tex.rstrip())
    elif legend:
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
    legend_tex: str | None = None,
    latex_body: str | None = None,
    footnote: str | None = None,
) -> None:
    write_table_csv(df, csv_path)
    write_table_tex(
        df,
        tex_path,
        caption=caption,
        legend=legend,
        legend_tex=legend_tex,
        latex_body=latex_body,
        footnote=footnote,
    )


def write_log_n_scaling_outputs(
    scaling_df: pd.DataFrame,
    fitted_types_df: pd.DataFrame,
    *,
    compose_dir: Path,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    gof_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> None:
    """Write fitted-types CSV, scaling CSV + LaTeX, and PL outcome summary."""
    scaling_legend = build_scaling_legend(
        minimum_fitted_types=minimum_fitted_types,
        gof_p=gof_p_threshold,
        comparison_p=comparison_p_threshold,
    )
    write_table_csv(fitted_types_df, compose_dir / "log_n_fitted_types.csv")
    write_table_csv_and_tex(
        scaling_df,
        compose_dir / "log_n_scaling.csv",
        compose_dir / "log_n_scaling.tex",
        caption="Log x n power-law scaling",
        legend_tex=build_scaling_tex_footnote(),
        latex_body=scaling_dataframe_to_latex(scaling_df),
    )
    write_pl_outcome_summary(
        scaling_df,
        compose_dir=compose_dir,
        legend=scaling_legend,
    )


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
        help=f"Power-law statistics results root (default: {POWERLAW_STATISTICS_REAL_DIR})",
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
    parser.add_argument(
        "--min-types",
        "--minimum-fitted-types",
        dest="min_types",
        type=int,
        default=DEFAULT_MINIMUM_FITTED_TYPES,
        help=(
            "Minimum fitted types threshold for classification/legends "
            f"(default: {DEFAULT_MINIMUM_FITTED_TYPES})"
        ),
    )
    parser.add_argument(
        "--gof-p",
        dest="gof_p",
        type=float,
        default=DEFAULT_SIGNIFICANCE_LEVEL,
        help=(
            "GOF p-value threshold for power-law plausibility "
            f"(default: {DEFAULT_SIGNIFICANCE_LEVEL})"
        ),
    )
    parser.add_argument(
        "--comparison-p",
        dest="comparison_p",
        type=float,
        default=DEFAULT_SIGNIFICANCE_LEVEL,
        help=(
            "Vuong/LLR p-value threshold for alternative preference "
            f"(default: {DEFAULT_SIGNIFICANCE_LEVEL})"
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help=(
            "Skip per-log RFC / PDF / CCDF plots "
            "(tables and the PL-outcome stacked-area chart are still written)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_dir) if args.output_dir else POWERLAW_STATISTICS_REAL_DIR
    tests_root = Path(args.tests_root) if args.tests_root else output_root
    min_types = int(args.min_types)
    gof_p_threshold = float(args.gof_p)
    comparison_p_threshold = float(args.comparison_p)
    compose_dir = output_root / compose_settings_dirname(
        min_types=min_types,
        gof_p=gof_p_threshold,
        comparison_p=comparison_p_threshold,
    )
    compose_dir.mkdir(parents=True, exist_ok=True)
    model = args.power_law_model
    datasets = list(args.datasets) if args.datasets else None
    concepts = list(args.concepts)

    print(
        "Composing from: "
        f"{tests_root} (model={model}, concepts={concepts}, "
        f"min_types={min_types}, gof_p={gof_p_threshold:g}, "
        f"comparison_p={comparison_p_threshold:g})"
    )

    if "variants" in concepts:
        variant_df = build_variant_power_law_table(
            tests_root,
            model=model,
            datasets=datasets,
            minimum_fitted_types=min_types,
            gof_p_threshold=gof_p_threshold,
            comparison_p_threshold=comparison_p_threshold,
        )
        write_table_csv_and_tex(
            variant_df,
            compose_dir / "variant_power_law.csv",
            compose_dir / "variant_power_law.tex",
            caption="Trace variant power-law results",
            legend_tex=build_variant_tex_footnote(),
            latex_body=variant_power_law_dataframe_to_latex(
                variant_df,
                minimum_fitted_types=min_types,
                gof_p_threshold=gof_p_threshold,
            ),
        )
        if args.skip_plots:
            print("Skipping per-log RFC/PDF/CCDF plots (--skip-plots)")
        else:
            write_variant_plots(
                tests_root,
                output_root / "plots",
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
        minimum_fitted_types=min_types,
        gof_p_threshold=gof_p_threshold,
        comparison_p_threshold=comparison_p_threshold,
    )
    write_log_n_scaling_outputs(
        scaling_df,
        fitted_types_df,
        compose_dir=compose_dir,
        minimum_fitted_types=min_types,
        gof_p_threshold=gof_p_threshold,
        comparison_p_threshold=comparison_p_threshold,
    )
    print(f"Compose outputs written under: {compose_dir}")


if __name__ == "__main__":
    main()
