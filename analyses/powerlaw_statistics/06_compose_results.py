"""Step 06: compose n-gram Clauset results into tables and plots.

Reads ``raw_results/<concept>/<power-law-model>/`` summaries under an xmin folder
and writes tables to a configuration subfolder
(default: ``results/pl_stats/real/xmin-adaptive/clauset_results/min[30]-gofp[0.10]-pcomp[0.10]``).
Per-log RFC / PDF / CCDF plots are independent of categorization and always
land under ``raw_results/plots/<log>/`` (use ``--skip-plots`` to omit them).
The fitted-type count matrix is also threshold-free and is written next to the raw fits.

- variant power-law table (CSV + LaTeX)
- trace-variant pairwise model-comparison table (LaTeX beside ``comparison.csv``)
- summary counts of those comparisons for logs outside categories A and B
- per-log RFC / PDF / CCDF plots for variants (shared ``plots/``)
- log x n fitted-types matrix (CSV)
- log x n scaling matrix (CSV + LaTeX with legend)
- per-scale PL outcome summary counts (CSV + LaTeX), stacked-area chart, and category bar chart (PDF + PNG)
- paired GOF comparison plot (PDF + PNG) and caption from existing paired tables
- transition overview LaTeX tables next to existing ``explore_gof_changes/*/transition_overview.csv``
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
    CLASSIFICATION_ALTERNATIVES,
    CLASSIFICATION_CELL_COLORS,
    CLASSIFICATION_CODES,
    CLASSIFICATION_LABEL_TEXT_COLORS,
    CLASSIFICATION_LABELS,
    DEFAULT_MINIMUM_FITTED_TYPES,
    DEFAULT_SIGNIFICANCE_LEVEL,
    DISTRIBUTION_NAMES,
    DOUBLY_BOUNDED_POWER_LAW,
    LOWER_BOUNDED_POWER_LAW,
    compact_classification,
    format_classification_label,
    llr_preference,
    select_best_distribution,
    select_best_other_distribution,
)
from utils.constants import PL_STATS_REAL_DIR
from utils.io.attachments import NGRAM_CONCEPTS
from analyses.powerlaw_statistics.analysis_helpers.paths import (
    concept_dirname,
    config_dirname,
    explore_dir,
    plots_dir,
    raw_results_dir,
    xmin_dirname,
)
from analyses.powerlaw_statistics.analysis_helpers.paired_gof_plot import (
    PAPER_TABLE_NAME,
    PairedPlotSetting,
    compose_paired_gof_plots,
    paired_frame_from_paper_table,
    write_paired_gof_plot_bundle,
)

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
    return config_dirname(
        min_types=min_types, gof_p=gof_p, comparison_p=comparison_p
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
        f"{format_classification_label('A')} = fit invalid, GOF missing, or fewer than "
        f"{minimum_fitted_types} fitted types; "
        f"{format_classification_label('B')} = GOF p<{gof}; "
        f"{format_classification_label('C')} = GOF p>={gof} and some of Exp/LN/StExp "
        f"significantly better than PL (comparison p<{cmp_p}); "
        f"{format_classification_label('D')} = GOF p>={gof} and PL significantly better than "
        f"Exp, LN, and StExp (comparison p<{cmp_p}); "
        f"{format_classification_label('E')} = GOF p>={gof} but no considered "
        "alternative significantly outperforms PL and PL does not significantly "
        "outperform all alternatives. "
        "TPL is excluded from preference."
    )


def build_scaling_tex_footnote() -> str:
    items = ", ".join(
        format_classification_label(code) for code in CLASSIFICATION_CODES
    )
    return (
        "\\begin{center}\\footnotesize\n"
        f"TV: Trace variant; \\quad Category: {items}.\n"
        "\\end{center}"
    )


def build_variant_tex_footnote() -> str:
    items = ", ".join(
        format_classification_label(code) for code in CLASSIFICATION_CODES
    )
    return (
        "\\begin{center}\\footnotesize\n"
        "Models: LN = Log normal, Exp = Exponential, StExp = Stretched exponential; "
        f"\\quad Category: {items}.\n"
        "\\end{center}"
    )


# Likelihood-ratio statistic in generated tables (Clauset et al. use script R).
LLR_STATISTIC_TEX = r"$\mathcal{R}$"

VARIANT_ALTERNATIVE_ABBREV = {
    "power_law": "PL",
    "full_range_power_law": "PL",
    "lower_bounded_power_law": "PL",
    "doubly_bounded_power_law": "PL",
    "truncated_power_law": "TPL",
    "lognormal": "LN",
    "exponential": "Exp",
    "stretched_exponential": "StExp",
}

# Compact scaling-cell labels (A–E), same set as across_logs_categories.
PL_OUTCOME_COLUMNS = list(CLASSIFICATION_CODES)
_PL_OUTCOME_STACK_COLORS = tuple(
    f"#{CLASSIFICATION_CELL_COLORS[code]}" for code in CLASSIFICATION_CODES
)
BAR_CHART_CODES = [code for code in CLASSIFICATION_CODES if code != "A"]


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



_LATEX_UNICODE_MAP = (
    ("→", "\0ARROW\0"),
    ("←", "\0LARROW\0"),
    ("−", "\0MINUS\0"),
    ("–", "\0MINUS\0"),
    ("—", "\0EMDASH\0"),
    ("≥", "\0GEQ\0"),
    ("≤", "\0LEQ\0"),
    ("×", "\0TIMES\0"),
)
_LATEX_UNICODE_TOKENS = {
    "\0ARROW\0": r"$\rightarrow$",
    "\0LARROW\0": r"$\leftarrow$",
    "\0MINUS\0": r"--",
    "\0EMDASH\0": r"---",
    "\0GEQ\0": r"$\geq$",
    "\0LEQ\0": r"$\leq$",
    "\0TIMES\0": r"$\times$",
}


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
    for src, dst in _LATEX_UNICODE_MAP:
        text = text.replace(src, dst)
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
    out: list[str] = []
    for ch in text:
        out.append(replacements.get(ch, ch))
    escaped = "".join(out)
    for token, latex in _LATEX_UNICODE_TOKENS.items():
        escaped = escaped.replace(token, latex)
    return escaped


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
    return tests_root / concept_dirname(concept) / model / "summary.csv"


def _comparison_path(tests_root: Path, concept: str, model: str) -> Path:
    return tests_root / concept_dirname(concept) / model / "comparison.csv"


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
                "fitted types": _optional_int(src.get("n_fitted_types")),
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
    table = pd.DataFrame(rows, columns=VARIANT_TABLE_COLUMNS)
    # Keep whole counts as integers when some logs have no fit (empty cells).
    table["fitted types"] = pd.to_numeric(table["fitted types"], errors="coerce").astype("Int64")
    return table


def classification_to_scaling_cell(classification: object) -> str:
    """Map a stored classification string to an A–E code."""
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
        or "ineligible" in lowered
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
        return escape_latex("N/A")
    return f"{number:.{digits}f}"


def _format_tex_decimal(value: object, digits: int = 2) -> str:
    number = _optional_float(value)
    if number is None:
        return escape_latex("N/A")
    return f"{number:.{digits}f}"


def _format_tex_int(value: object) -> str:
    number = _optional_int(value)
    if number is None:
        float_number = _optional_float(value)
        if float_number is None:
            return escape_latex("N/A")
        number = int(round(float_number))
    return str(number)


def _yes_no(flag: bool | None) -> str:
    if flag is None:
        return escape_latex("N/A")
    return "Yes" if flag else "No"


def abbreviate_alternative_model(name: object) -> str:
    if name is None or pd.isna(name):
        return "N/A"
    text = str(name).strip()
    if not text or text.lower() == "nan":
        return "N/A"
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
            f"Log & $\\alpha$ & $x_{{\\min}}$ & \\# & \\% & $\\geq {minimum_fitted_types}$?"
            f" & $p_{{\\text{{GOF}}}}$ & $p_{{\\text{{GOF}}}} \\geq {gof}$"
            f" & Model & {LLR_STATISTIC_TEX} & $p_{{\\text{{comp.}}}}$ & Result & "
            r"\\"
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
    tabular = "\n".join(lines)
    return "\\resizebox{\\linewidth}{!}{%\n" + tabular + "\n}\n"


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


def _savefig_pdf_and_png(fig: Any, output_path: Path) -> None:
    """Write ``output_path`` (typically ``.pdf``) and a sibling ``.png``."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    png_path = output_path.with_suffix(".png")
    if png_path != output_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {png_path}")
    plt.close(fig)


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
            label=format_classification_label(code, style="paren"),
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
    _savefig_pdf_and_png(fig, output_path)


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
        text_color = CLASSIFICATION_LABEL_TEXT_COLORS.get(
            code, CLASSIFICATION_LABEL_TEXT_COLORS["A"]
        )
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
            label=format_classification_label(code, style="paren"),
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
    _savefig_pdf_and_png(fig, output_path)


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
    return build_across_logs_summary_field_table(
        tests_root,
        field="n_fitted_types",
        model=model,
        datasets=datasets,
        concepts=concepts,
        as_int=True,
    )


def build_across_logs_summary_field_table(
    tests_root: Path,
    *,
    field: str,
    model: str,
    datasets: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
    as_int: bool = False,
) -> pd.DataFrame:
    """Build a log × scale matrix from one summary column."""
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
            value = summary.loc[log_name, field]
            if isinstance(value, pd.Series):
                value = value.iloc[0]
            if as_int:
                number = _optional_int(value)
                row[column] = "?" if number is None else str(number)
            else:
                number = _optional_float(value)
                if number is None:
                    row[column] = "?"
                elif float(number).is_integer():
                    row[column] = str(int(number))
                else:
                    row[column] = f"{number:g}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["Log", *scaling_columns])


def build_across_logs_best_alternative_tables(
    tests_root: Path,
    *,
    model: str,
    datasets: Sequence[str] | None = None,
    concepts: Sequence[str] | None = None,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build best-model, R, p, and significance matrices for each log × scale.

    The name matrix uses ``power_law`` or an alternative model id. R and p are
    blank when power law wins. The significance matrix holds ``True``/``False``
    strings for LaTeX bolding.
    """
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
    name_rows: list[dict[str, str]] = []
    r_rows: list[dict[str, str]] = []
    p_rows: list[dict[str, str]] = []
    sig_rows: list[dict[str, str]] = []
    for log_name in ordered_logs:
        name_row: dict[str, str] = {"Log": log_name}
        r_row: dict[str, str] = {"Log": log_name}
        p_row: dict[str, str] = {"Log": log_name}
        sig_row: dict[str, str] = {"Log": log_name}
        for column in scaling_columns:
            summary = per_concept.get(column)
            if summary is None or log_name not in summary.index:
                name_row[column] = "?"
                r_row[column] = "?"
                p_row[column] = "?"
                sig_row[column] = "False"
                continue
            comparison = _comparison_for_log(per_comparison.get(column), log_name)
            name, r_val, p_val, significant = select_best_distribution(
                comparison, significance_level=comparison_p_threshold
            )
            name_row[column] = "?" if name is None else name
            r_row[column] = "?" if r_val is None else f"{r_val:g}"
            p_row[column] = "?" if p_val is None else f"{p_val:g}"
            sig_row[column] = "True" if significant else "False"
        name_rows.append(name_row)
        r_rows.append(r_row)
        p_rows.append(p_row)
        sig_rows.append(sig_row)
    columns = ["Log", *scaling_columns]
    return (
        pd.DataFrame(name_rows, columns=columns),
        pd.DataFrame(r_rows, columns=columns),
        pd.DataFrame(p_rows, columns=columns),
        pd.DataFrame(sig_rows, columns=columns),
    )


def _dataframe_to_latex_escaped(df: pd.DataFrame) -> str:
    """Render a DataFrame as LaTeX with escaped cell text."""
    escaped = df.copy()
    for col in escaped.columns:
        escaped[col] = escaped[col].map(escape_latex)
    # escape=True would double-escape; values are already escaped.
    return escaped.to_latex(index=False, escape=False)


def _across_logs_column_layout(
    df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str], list[str], int]:
    """Return ordered value columns and header pieces for across-logs tables."""
    value_cols = [c for c in df.columns if c != "Log"]
    ngram_cols = [
        c for c in value_cols if str(c).startswith("n") and str(c)[1:].isdigit()
    ]
    has_trace = "trace" in value_cols
    other_cols = [c for c in value_cols if c not in ngram_cols and c != "trace"]
    ordered_cols = [*ngram_cols, *(["trace"] if has_trace else []), *other_cols]
    n_k = len(ngram_cols)

    header1_parts = ["Log"]
    if n_k:
        header1_parts.append(rf"\multicolumn{{{n_k}}}{{c}}{{$k$}}")
    if has_trace:
        header1_parts.append("TV")
    header1_parts.extend(escape_latex(c) for c in other_cols)

    header2_parts = [""]
    header2_parts.extend(str(int(str(c)[1:])) for c in ngram_cols)
    if has_trace:
        header2_parts.append("")
    header2_parts.extend("" for _ in other_cols)
    return ordered_cols, header1_parts, header2_parts, ngram_cols, n_k


def scaling_dataframe_to_latex(scaling_df: pd.DataFrame) -> str:
    """Render category matrix LaTeX with a two-row ``k`` / ``TV`` header."""
    ordered_cols, header1_parts, header2_parts, _ngram_cols, n_k = (
        _across_logs_column_layout(scaling_df)
    )
    # Center k / TV headers and category letters under each column.
    col_spec = "l" + ("c" * len(ordered_cols))
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header1_parts) + r" \\",
    ]
    if n_k:
        lines.append(rf"\cmidrule(lr){{2-{1 + n_k}}}")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    for _, row in scaling_df.iterrows():
        cells = [escape_latex(row.get("Log"))]
        for col in ordered_cols:
            raw = row[col]
            if str(raw).strip() == "?":
                cells.append(escape_latex("?"))
                continue
            code = classification_to_scaling_cell(raw)
            cells.append(_category_cellcolor_tex(code))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tabular = "\n".join(lines)
    return "\\resizebox{\\linewidth}{!}{%\n" + tabular + "\n}\n"


def build_best_alternative_tex_footnote(
    *,
    comparison_p: float = DEFAULT_SIGNIFICANCE_LEVEL,
    include_unavailable: bool = False,
    break_before_bold: bool = False,
) -> str:
    cmp_p = _format_threshold(comparison_p)
    unavailable = (
        "; ``?'' denotes unavailable or ineligible results"
        if include_unavailable
        else ""
    )
    joiner = ";\\\\\n" if break_before_bold else "; "
    return (
        "\\begin{center}\\footnotesize\n"
        "Models: PL~=~Power law, LN~=~Log-normal, Exp~=~Exponential, "
        "StExp~=~Stretched exponential"
        f"{joiner}Bold: p-value $< {cmp_p}${unavailable}\n"
        "\\end{center}"
    )


def _comparison_model_label(model_1: object, model_2: object) -> str:
    """Short-form label for one pairwise comparison, e.g. ``PL vs LN``."""
    return (
        f"{abbreviate_alternative_model(model_1)} vs "
        f"{abbreviate_alternative_model(model_2)}"
    )


def _preferred_model_tex(
    name: object,
    p_value: float | None,
    *,
    comparison_p_threshold: float,
) -> str:
    """Preferred-model abbreviation, bold when the comparison is significant."""
    abbrev = abbreviate_alternative_model(name)
    if abbrev == "N/A":
        return abbrev
    escaped = escape_latex(abbrev)
    if p_value is not None and p_value < comparison_p_threshold:
        return rf"\textbf{{{escaped}}}"
    return escaped


# Row order within each log in the trace-variant comparison table.
VARIANT_COMPARISON_MODEL_ORDER = (
    "exponential",
    "stretched_exponential",
    "lognormal",
)

VARIANT_COMPARISON_CAPTION = (
    r"\caption[Pairwise model comparisons at the trace-variant level]"
    "{Pairwise likelihood-ratio comparisons of the power law against each "
    "alternative at the trace-variant level. "
    "N/A denotes an unavailable comparison.}"
)


def variant_model_comparison_to_latex(
    comparison_df: pd.DataFrame,
    *,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> str:
    """One row per log and pairwise alternative comparison.

    Returns the interior of a ``longtable`` (headers, rows, and foot).
    Log names follow the CSV order and are printed only on the first row of
    each log. Within a log the alternatives are Exp, StExp, then LN.
    Truncated power law is omitted. The preferred-model cell is bold when
    ``p`` is below ``comparison_p_threshold``.
    """
    header = (
        f"Log & Model & {LLR_STATISTIC_TEX} & $p$ & Preferred model " + r"\\"
    )
    lines = [
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{@{}l}{\footnotesize\textit{Continued from previous page}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    if comparison_df is None or comparison_df.empty or "model_2" not in comparison_df.columns:
        return "\n".join(lines)

    frame = comparison_df.loc[
        comparison_df["model_2"].astype(str).isin(CLASSIFICATION_ALTERNATIVES)
    ]
    if "log_name" in frame.columns:
        log_names = list(dict.fromkeys(frame["log_name"].astype(str)))
    else:
        log_names = [""]
        frame = frame.copy()
        frame["log_name"] = ""

    model_rank = {
        name: index for index, name in enumerate(VARIANT_COMPARISON_MODEL_ORDER)
    }
    first_log = True
    for log_name in log_names:
        rows = frame.loc[frame["log_name"].astype(str) == log_name]
        if rows.empty:
            continue
        rows = rows.assign(
            _model_order=rows["model_2"].astype(str).map(
                lambda name: model_rank.get(name, len(model_rank))
            )
        ).sort_values("_model_order", kind="stable")
        if not first_log:
            lines.append(r"\midrule")
        first_log = False
        first_row = True
        for _, row in rows.iterrows():
            p_value = _optional_float(row.get("p"))
            cells = [
                escape_latex(log_name) if first_row else "",
                escape_latex(
                    _comparison_model_label(row.get("model_1"), row.get("model_2"))
                ),
                _format_tex_decimal(row.get("R")),
                _format_tex_decimal(p_value),
                _preferred_model_tex(
                    row.get("preferred_model"),
                    p_value,
                    comparison_p_threshold=comparison_p_threshold,
                ),
            ]
            first_row = False
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def write_variant_model_comparison_tex(
    comparison_csv: Path,
    *,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    datasets: Sequence[str] | None = None,
    tex_path: Path | None = None,
) -> Path | None:
    """Write ``comparison.tex`` next to a trace-variant ``comparison.csv``."""
    csv_path = Path(comparison_csv)
    if not csv_path.is_file():
        print(f"No comparison CSV at {csv_path}; skipping comparison TeX")
        return None
    destination = Path(tex_path) if tex_path is not None else csv_path.with_suffix(".tex")
    frame = pd.read_csv(csv_path)
    if datasets is not None and "log_name" in frame.columns:
        wanted = {str(name) for name in datasets}
        frame = frame.loc[frame["log_name"].astype(str).isin(wanted)]
    body = variant_model_comparison_to_latex(
        frame, comparison_p_threshold=comparison_p_threshold
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Full longtable, not a fragment: ``\input`` is not expandable, so a body
    # that starts with ``\toprule`` cannot be inserted after ``\\``.
    destination.write_text(
        "\n".join(
            [
                "% Pairwise likelihood-ratio comparisons. Do not edit by hand.",
                r"{\small",
                r"\begin{longtable}{@{}llrrl@{}}",
                VARIANT_COMPARISON_CAPTION,
                r"\label{tab:variant-model-comparisons}\\",
                body,
                r"\end{longtable}",
                build_best_alternative_tex_footnote(
                    comparison_p=comparison_p_threshold,
                    break_before_bold=True,
                ),
                r"}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved: {destination}")
    return destination


_COMPARISON_SUMMARY_OUTCOMES = (
    ("other", "Other distribution significantly better"),
    ("inconclusive", "Inconclusive"),
    ("pl", "PL significantly better"),
)


def _pairwise_comparison_outcome(
    r_value: float | None,
    p_value: float | None,
    *,
    comparison_p_threshold: float,
) -> str:
    """One cell of the A/B-excluded summary: other, pl, or inconclusive."""
    if (
        r_value is None
        or p_value is None
        or not np.isfinite(r_value)
        or not np.isfinite(p_value)
    ):
        return "inconclusive"
    if p_value < comparison_p_threshold and r_value < 0:
        return "other"
    if p_value < comparison_p_threshold and r_value > 0:
        return "pl"
    return "inconclusive"


def variant_comparison_summary_counts(
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    minimum_fitted_types: int,
    gof_p_threshold: float,
    comparison_p_threshold: float,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Count pairwise outcomes for logs that are not in category A or B.

    Categories use ``compact_classification`` with the given thresholds.
    Each column is one alternative in ``VARIANT_COMPARISON_MODEL_ORDER``.
    A missing or non-finite comparison counts as inconclusive.
    """
    counts = {
        model: {key: 0 for key, _label in _COMPARISON_SUMMARY_OUTCOMES}
        for model in VARIANT_COMPARISON_MODEL_ORDER
    }
    if (
        summary_df is None
        or summary_df.empty
        or "log_name" not in summary_df.columns
    ):
        return [], counts

    eligible: list[str] = []
    for _, src in summary_df.iterrows():
        log_name = str(src["log_name"])
        code = _compact_from_summary_row(
            src,
            comparison_df,
            log_name=log_name,
            minimum_fitted_types=minimum_fitted_types,
            gof_p_threshold=gof_p_threshold,
            comparison_p_threshold=comparison_p_threshold,
        )
        if code in {"A", "B"}:
            continue
        eligible.append(log_name)
        rows = _comparison_for_log(comparison_df, log_name)
        for model in VARIANT_COMPARISON_MODEL_ORDER:
            if rows.empty or "model_2" not in rows.columns:
                matched = rows.iloc[0:0]
            else:
                matched = rows.loc[rows["model_2"].astype(str) == model]
            if matched.empty:
                outcome = "inconclusive"
            else:
                row = matched.iloc[0]
                outcome = _pairwise_comparison_outcome(
                    _optional_float(row.get("R")),
                    _optional_float(row.get("p")),
                    comparison_p_threshold=comparison_p_threshold,
                )
            counts[model][outcome] += 1
    return eligible, counts


def variant_comparison_summary_to_latex(
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    minimum_fitted_types: int,
    gof_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> str:
    """Full ``table`` environment summarising logs outside categories A and B."""
    eligible, counts = variant_comparison_summary_counts(
        comparison_df,
        summary_df,
        minimum_fitted_types=minimum_fitted_types,
        gof_p_threshold=gof_p_threshold,
        comparison_p_threshold=comparison_p_threshold,
    )
    n_logs = len(eligible)
    log_word = "log" if n_logs == 1 else "logs"
    gof = _format_threshold(gof_p_threshold)
    cmp_p = _format_threshold(comparison_p_threshold)
    headers = [
        _comparison_model_label("lower_bounded_power_law", model)
        for model in VARIANT_COMPARISON_MODEL_ORDER
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (
            r"\caption{Summary of alternative-model tests for "
            f"{n_logs} {log_word} not in category A or B.}}"
        ),
        r"\label{tab:variant-comparison-summary}",
        r"{\small",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        " & " + " & ".join(escape_latex(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for key, label in _COMPARISON_SUMMARY_OUTCOMES:
        cells = [escape_latex(label)]
        cells.extend(str(counts[model][key]) for model in VARIANT_COMPARISON_MODEL_ORDER)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{center}\footnotesize",
            (
                f"Excluded: category A (invalid fit, missing GOF, or fewer than "
                f"{minimum_fitted_types} fitted types) and category B "
                f"(GOF $p<{gof}$).\\\\"
            ),
            f"Significant: likelihood-ratio $p<{cmp_p}$.",
            r"\end{center}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_variant_comparison_summary_tex(
    comparison_csv: Path,
    *,
    summary_csv: Path | None = None,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    gof_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    datasets: Sequence[str] | None = None,
    tex_path: Path | None = None,
) -> Path | None:
    """Write ``comparison_summary.tex`` next to a trace-variant ``comparison.csv``."""
    csv_path = Path(comparison_csv)
    summary_path = (
        Path(summary_csv) if summary_csv is not None else csv_path.with_name("summary.csv")
    )
    if not csv_path.is_file() or not summary_path.is_file():
        print(
            "No comparison/summary CSV for the category summary "
            f"({csv_path}, {summary_path}); skipping"
        )
        return None
    comparison_df = pd.read_csv(csv_path)
    summary_df = pd.read_csv(summary_path)
    if datasets is not None:
        wanted = {str(name) for name in datasets}
        if "log_name" in comparison_df.columns:
            comparison_df = comparison_df.loc[
                comparison_df["log_name"].astype(str).isin(wanted)
            ]
        if "log_name" in summary_df.columns:
            summary_df = summary_df.loc[summary_df["log_name"].astype(str).isin(wanted)]
    destination = (
        Path(tex_path) if tex_path is not None else csv_path.with_name("comparison_summary.tex")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "% Counts of pairwise comparisons for logs outside categories A and B. "
        "Do not edit by hand.\n"
        + variant_comparison_summary_to_latex(
            comparison_df,
            summary_df,
            minimum_fitted_types=minimum_fitted_types,
            gof_p_threshold=gof_p_threshold,
            comparison_p_threshold=comparison_p_threshold,
        ),
        encoding="utf-8",
    )
    print(f"Saved: {destination}")
    return destination


def best_alternative_dataframe_to_latex(
    names_df: pd.DataFrame,
    significant_df: pd.DataFrame,
) -> str:
    """Render best-distribution abbreviations; bold when significant."""
    ordered_cols, header1_parts, header2_parts, _ngram_cols, n_k = (
        _across_logs_column_layout(names_df)
    )
    # Center k / TV headers and model abbreviations under each column.
    col_spec = "l" + ("c" * len(ordered_cols))
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header1_parts) + r" \\",
    ]
    if n_k:
        lines.append(rf"\cmidrule(lr){{2-{1 + n_k}}}")
    lines.append(" & ".join(header2_parts) + r" \\")
    lines.append(r"\midrule")

    sig_by_log = significant_df.set_index(significant_df["Log"].astype(str))
    for _, row in names_df.iterrows():
        log_name = str(row.get("Log"))
        cells = [escape_latex(log_name)]
        sig_row = sig_by_log.loc[log_name] if log_name in sig_by_log.index else None
        if isinstance(sig_row, pd.DataFrame):
            sig_row = sig_row.iloc[0]
        for col in ordered_cols:
            raw = row[col]
            text = str(raw).strip()
            if not text or text == "?":
                cells.append(escape_latex("?"))
                continue
            abbrev = VARIANT_ALTERNATIVE_ABBREV.get(text, text)
            bold = False
            if sig_row is not None:
                bold = str(sig_row.get(col)).strip().lower() in {"true", "1", "yes"}
            escaped = escape_latex(abbrev)
            cells.append(rf"\textbf{{{escaped}}}" if bold else escaped)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tabular = "\n".join(lines)
    return "\\resizebox{\\linewidth}{!}{%\n" + tabular + "\n}\n"


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


_TRANSITION_ROW_LABEL_TEX = {
    "Types >= xmin": r"Types $\geq x_{\min}$",
    "Types ≥ xmin": r"Types $\geq x_{\min}$",
    "Types <= xmin": r"Types $\leq x_{\min}$",
    "Types ≤ xmin": r"Types $\leq x_{\min}$",
    "Types < xmin": r"Types $< x_{\min}$",
    "Occurrences >= xmin": r"Occurrences $\geq x_{\min}$",
    "Occurrences ≥ xmin": r"Occurrences $\geq x_{\min}$",
    "Occurrences <= xmin": r"Occurrences $\leq x_{\min}$",
    "Occurrences ≤ xmin": r"Occurrences $\leq x_{\min}$",
    "Occurrences < xmin": r"Occurrences $< x_{\min}$",
    "− Excluded because < xmin": r"-- Excluded because $< x_{\min}$",
    "- Excluded because < xmin": r"-- Excluded because $< x_{\min}$",
    "Excluded because < xmin": r"Excluded because $< x_{\min}$",
    "xmin": r"$x_{\min}$",
}


def _format_transition_row_label(value: object) -> str:
    """Render a transition row label, with math-mode $x_{\\min}$ comparisons."""
    text = "" if value is None else str(value).strip()
    if text in _TRANSITION_ROW_LABEL_TEX:
        return _TRANSITION_ROW_LABEL_TEX[text]
    # Normalise unicode minus / dash prefixes before the generic escape path.
    normalised = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("≥", ">=")
        .replace("≤", "<=")
    )
    if normalised in _TRANSITION_ROW_LABEL_TEX:
        return _TRANSITION_ROW_LABEL_TEX[normalised]
    return escape_latex(text)


def _format_transition_cell(value: object) -> str:
    """Format a transition-overview cell for LaTeX (thousands separators)."""
    if value is None:
        return escape_latex("N/A")
    try:
        if pd.isna(value):
            return escape_latex("N/A")
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return escape_latex("N/A")
    number = _optional_float(value)
    if number is not None and float(number).is_integer():
        return escape_latex(f"{int(round(number)):,}")
    if number is not None:
        return escape_latex(f"{number:g}")
    return escape_latex(text)


def _normalize_transition_label(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("≥", ">=")
        .replace("≤", "<=")
    )


# Appendix presentation order (CSV keeps the full explore table).
_TRANSITION_TYPE_BLOCK = (
    "Types before",
    "+ Extra from branching",
    "- Exhausted types",
    "+ Introduced types",
    "Types after",
    "- Excluded because < xmin",
    "Types >= xmin",
)
_TRANSITION_OCC_BLOCK = (
    "Occurrences before",
    "- No successor",
    "+ Introduced occurrences",
    "Occurrences after",
    "- Excluded because < xmin",
    "Occurrences >= xmin",
)
# Greyscale for appendix reconciliation rows (None = white / no \rowcolor).
_TRANSITION_GRAY_TOTAL = "0.85"  # before / after / >= xmin
_TRANSITION_GRAY_DELTA = "0.93"  # + / - accounting rows
_TRANSITION_TOTAL_LABELS = {
    "Types before",
    "Types after",
    "Types >= xmin",
    "Occurrences before",
    "Occurrences after",
    "Occurrences >= xmin",
}


def _transition_row_gray(label: str) -> str | None:
    """White for xmin; darker grey for totals; lighter grey for +/- rows."""
    if label == "xmin":
        return None
    if label in _TRANSITION_TOTAL_LABELS:
        return _TRANSITION_GRAY_TOTAL
    return _TRANSITION_GRAY_DELTA


def _transition_row_index_map(df: pd.DataFrame, row_col: str) -> dict[str, list[Any]]:
    """Map normalised labels to row positions (excluded appears twice)."""
    mapping: dict[str, list[Any]] = {}
    for idx, row in df.iterrows():
        key = _normalize_transition_label(row.get(row_col))
        mapping.setdefault(key, []).append(idx)
    return mapping


def _transition_appendix_row_plan(
    df: pd.DataFrame, row_col: str
) -> list[tuple[str, Any | None, str | None]]:
    """Return (kind, source_index|None, gray) rows for the appendix table."""
    mapping = _transition_row_index_map(df, row_col)
    plan: list[tuple[str, Any | None, str | None]] = []

    xmin_ids = mapping.get("xmin", [])
    if not xmin_ids:
        return plan
    plan.append(("data", xmin_ids[0], _transition_row_gray("xmin")))
    plan.append(("spacer", None, None))

    excluded_ids = list(mapping.get("- Excluded because < xmin", []))
    for label in _TRANSITION_TYPE_BLOCK:
        if label == "- Excluded because < xmin":
            if not excluded_ids:
                continue
            idx = excluded_ids.pop(0)
        else:
            ids = mapping.get(label, [])
            if not ids:
                continue
            idx = ids[0]
        plan.append(("data", idx, _transition_row_gray(label)))
    plan.append(("spacer", None, None))

    for label in _TRANSITION_OCC_BLOCK:
        if label == "- Excluded because < xmin":
            if not excluded_ids:
                continue
            idx = excluded_ids.pop(0)
        else:
            ids = mapping.get(label, [])
            if not ids:
                continue
            idx = ids[0]
        plan.append(("data", idx, _transition_row_gray(label)))
    return plan


def transition_overview_dataframe_to_latex(df: pd.DataFrame) -> str:
    """Render the appendix type/occurrence reconciliation table."""
    if df.empty:
        return (
            "\\begin{tabular}{l}\n\\toprule\n"
            "No transition data\n\\bottomrule\n\\end{tabular}\n"
        )
    columns = list(df.columns)
    row_col = columns[0]
    value_cols = columns[1:]
    # Corner cell labels the column axis (abstraction order k / TV).
    header_cells = ["$k$"]
    for col in value_cols:
        label = str(col)
        if label.startswith("k="):
            header_cells.append(escape_latex(label[2:]))
        elif label == "variants":
            header_cells.append("TV")
        else:
            header_cells.append(escape_latex(label))
    col_spec = "l" + ("r" * len(value_cols))
    n_cols = 1 + len(value_cols)
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    plan = _transition_appendix_row_plan(df, row_col)
    if not plan:
        # Fall back to the raw frame if expected labels are missing.
        plan = [
            ("data", idx, "0.95" if i % 2 else "0.90")
            for i, idx in enumerate(df.index)
        ]
    for kind, idx, gray in plan:
        if kind == "spacer":
            lines.append(r"\addlinespace[0.35em]")
            lines.append((" & " * (n_cols - 1)) + r" \\")
            lines.append(r"\addlinespace[0.15em]")
            continue
        row = df.loc[idx]
        cells = [_format_transition_row_label(row.get(row_col))]
        cells.extend(_format_transition_cell(row.get(col)) for col in value_cols)
        prefix = rf"\rowcolor[gray]{{{gray}}} " if gray else ""
        lines.append(prefix + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tabular = "\n".join(lines)
    return "\\resizebox{\\linewidth}{!}{%\n" + tabular + "\n}\n"


def build_transition_overview_tex_footnote() -> str:
    return (
        "\\begin{flushleft}\\footnotesize\n"
        "Top block: fitted $x_{\\min}$. Middle block: type-count reconciliation "
        "from abstraction order $k-1$ to $k$. Bottom block: occurrence "
        "reconciliation; \\quad TV = Trace Variants; \\quad N/A marks "
        "inapplicable cells.\n"
        "\\end{flushleft}"
    )


def write_transition_overview_tex(
    csv_path: Path,
    tex_path: Path | None = None,
    *,
    log_name: str | None = None,
) -> Path | None:
    """Write ``transition_overview.tex`` next to an existing CSV."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return None
    tex_path = Path(tex_path) if tex_path is not None else csv_path.with_suffix(".tex")
    df = pd.read_csv(csv_path)
    label = log_name or csv_path.parent.name
    write_table_tex(
        df,
        tex_path,
        caption=(
            f"Type-count reconciliation for {label}: type and occurrence "
            "transitions across abstraction orders $k$ and trace variants."
        ),
        legend_tex=build_transition_overview_tex_footnote(),
        latex_body=transition_overview_dataframe_to_latex(df),
    )
    return tex_path


def write_explore_transition_overview_tex(
    experiment: Path,
    xmin: int | None = None,
) -> list[Path]:
    """Write LaTeX for every ``transition_overview.csv`` under explore_gof_changes."""
    root = explore_dir(Path(experiment), xmin)
    written: list[Path] = []
    if not root.is_dir():
        print(f"No explore_gof_changes directory at {root}; skipping transition TeX")
        return written
    for log_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        csv_path = log_dir / "transition_overview.csv"
        tex_path = write_transition_overview_tex(csv_path, log_name=log_dir.name)
        if tex_path is not None:
            written.append(tex_path)
    return written


def write_across_logs_outputs(
    *,
    categories_df: pd.DataFrame,
    types_df: pd.DataFrame,
    fitted_types_df: pd.DataFrame,
    xmin_df: pd.DataFrame,
    best_alt_df: pd.DataFrame,
    best_alt_r_df: pd.DataFrame,
    best_alt_p_df: pd.DataFrame,
    best_alt_significant_df: pd.DataFrame,
    compose_dir: Path,
    minimum_fitted_types: int = DEFAULT_MINIMUM_FITTED_TYPES,
    gof_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
    comparison_p_threshold: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> None:
    """Write across-logs category / count / best-alternative tables and charts."""
    scaling_legend = build_scaling_legend(
        minimum_fitted_types=minimum_fitted_types,
        gof_p=gof_p_threshold,
        comparison_p=comparison_p_threshold,
    )
    write_table_csv(types_df, compose_dir / "across_logs_types.csv")
    write_table_csv(fitted_types_df, compose_dir / "across_logs_fitted_types.csv")
    write_table_csv(xmin_df, compose_dir / "across_logs_xmin.csv")
    write_table_csv(best_alt_r_df, compose_dir / "across_logs_best_alternative_r.csv")
    write_table_csv(best_alt_p_df, compose_dir / "across_logs_best_alternative_p.csv")
    write_table_csv_and_tex(
        categories_df,
        compose_dir / "across_logs_categories.csv",
        compose_dir / "across_logs_categories.tex",
        caption="Log x n power-law categories",
        legend_tex=build_scaling_tex_footnote(),
        latex_body=scaling_dataframe_to_latex(categories_df),
    )
    write_table_csv_and_tex(
        best_alt_df,
        compose_dir / "across_logs_best_alternative.csv",
        compose_dir / "across_logs_best_alternative.tex",
        caption="Log x n best-fitting distribution",
        legend_tex=build_best_alternative_tex_footnote(
            comparison_p=comparison_p_threshold,
            include_unavailable=True,
        ),
        latex_body=best_alternative_dataframe_to_latex(
            best_alt_df, best_alt_significant_df
        ),
    )
    # Remove superseded filenames from older compose runs.
    for stale in (
        compose_dir / "log_n_scaling.csv",
        compose_dir / "log_n_scaling.tex",
    ):
        if stale.exists():
            stale.unlink()
            print(f"Removed stale: {stale}")
    write_pl_outcome_summary(
        categories_df,
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
        help="Concepts to include in compose tables/plots (default: n1..n20 + variants)",
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
        help="xmin folder to compose (default: xmin-adaptive)",
    )
    parser.add_argument(
        "--tests-root",
        type=str,
        default=None,
        help="Root containing <concept>/<model>/ summaries (default: the xmin raw_results folder)",
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
    parser.add_argument(
        "--all-paired-gof-plots",
        action="store_true",
        help=(
            "Only write paired GOF comparison plots for every discoverable "
            "clauset_results setting under --output-dir "
            "(uses existing paired tables, no refits)"
        ),
    )
    return parser.parse_args(argv)


def write_compose_paired_gof_plot(
    compose_dir: Path,
    *,
    min_types: int,
    gof_p: float,
    comparison_p: float,
    xmin: int | None,
) -> None:
    """Write the paired GOF plot for one compose output directory if possible."""
    paper = Path(compose_dir) / PAPER_TABLE_NAME
    if not paper.is_file():
        print(
            f"Skipping paired GOF plot for {compose_dir.name} "
            f"(missing {PAPER_TABLE_NAME})"
        )
        return
    xmin_label = "adaptive" if xmin is None else str(int(xmin))
    setting = PairedPlotSetting(
        label=f"{xmin_dirname(xmin)}/{compose_dir.name}",
        output_dir=Path(compose_dir),
        frame=paired_frame_from_paper_table(paper),
        min_types=int(min_types),
        gof_p=float(gof_p),
        comparison_p=float(comparison_p),
        xmin_label=xmin_label,
        source=str(paper),
    )
    write_paired_gof_plot_bundle(setting)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_dir = Path(args.output_dir) if args.output_dir else PL_STATS_REAL_DIR

    if args.all_paired_gof_plots:
        settings, skipped = compose_paired_gof_plots(experiment_dir)
        print(
            f"Paired GOF plots written for {len(settings)} setting(s) "
            f"under {experiment_dir}"
        )
        for setting in settings:
            print(f"  wrote {setting.output_dir / 'paired_gof_comparison.pdf'}")
        if skipped:
            print(f"Skipped {len(skipped)} setting(s):")
            for item in skipped:
                print(f"  {item}")
        return

    raw_root = raw_results_dir(experiment_dir, args.xmin)
    tests_root = Path(args.tests_root) if args.tests_root else raw_root
    min_types = int(args.min_types)
    gof_p_threshold = float(args.gof_p)
    comparison_p_threshold = float(args.comparison_p)
    compose_dir = (
        raw_root.parent
        / "clauset_results"
        / compose_settings_dirname(
            min_types=min_types,
            gof_p=gof_p_threshold,
            comparison_p=comparison_p_threshold,
        )
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
        write_variant_model_comparison_tex(
            _comparison_path(tests_root, "variants", model),
            comparison_p_threshold=comparison_p_threshold,
            datasets=datasets,
        )
        write_variant_comparison_summary_tex(
            _comparison_path(tests_root, "variants", model),
            summary_csv=_summary_path(tests_root, "variants", model),
            minimum_fitted_types=min_types,
            gof_p_threshold=gof_p_threshold,
            comparison_p_threshold=comparison_p_threshold,
            datasets=datasets,
        )
        if args.skip_plots:
            print("Skipping per-log RFC/PDF/CCDF plots (--skip-plots)")
        else:
            write_variant_plots(
                tests_root,
                plots_dir(experiment_dir, args.xmin),
                model=model,
                datasets=datasets,
            )
    else:
        print("Skipping variant table/plots (variants not in --concepts)")

    categories_df = build_log_n_scaling_table(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=concepts,
        minimum_fitted_types=min_types,
        gof_p_threshold=gof_p_threshold,
        comparison_p_threshold=comparison_p_threshold,
    )
    types_df = build_across_logs_summary_field_table(
        tests_root,
        field="n_types",
        model=model,
        datasets=datasets,
        concepts=concepts,
        as_int=True,
    )
    fitted_types_df = build_log_n_fitted_types_table(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=concepts,
    )
    xmin_df = build_across_logs_summary_field_table(
        tests_root,
        field="xmin",
        model=model,
        datasets=datasets,
        concepts=concepts,
        as_int=False,
    )
    (
        best_alt_df,
        best_alt_r_df,
        best_alt_p_df,
        best_alt_significant_df,
    ) = build_across_logs_best_alternative_tables(
        tests_root,
        model=model,
        datasets=datasets,
        concepts=concepts,
        comparison_p_threshold=comparison_p_threshold,
    )
    write_across_logs_outputs(
        categories_df=categories_df,
        types_df=types_df,
        fitted_types_df=fitted_types_df,
        xmin_df=xmin_df,
        best_alt_df=best_alt_df,
        best_alt_r_df=best_alt_r_df,
        best_alt_p_df=best_alt_p_df,
        best_alt_significant_df=best_alt_significant_df,
        compose_dir=compose_dir,
        minimum_fitted_types=min_types,
        gof_p_threshold=gof_p_threshold,
        comparison_p_threshold=comparison_p_threshold,
    )
    write_compose_paired_gof_plot(
        compose_dir,
        min_types=min_types,
        gof_p=gof_p_threshold,
        comparison_p=comparison_p_threshold,
        xmin=args.xmin,
    )
    transition_tex = write_explore_transition_overview_tex(
        experiment_dir, xmin=args.xmin
    )
    if transition_tex:
        print(
            f"Wrote {len(transition_tex)} transition overview TeX "
            f"table(s) under {explore_dir(experiment_dir, args.xmin)}"
        )
    print(f"Compose outputs written under: {compose_dir}")


if __name__ == "__main__":
    main()
