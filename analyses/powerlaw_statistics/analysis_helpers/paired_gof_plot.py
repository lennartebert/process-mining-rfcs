"""Paired GOF difference plot (trace variants minus finite order).

Reads existing paired comparison tables. Does not refit, bootstrap, or
recompute McNemar tests.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from analyses.powerlaw_statistics.analysis_helpers.paths import (
    CLAUSE_RESULTS_DIRNAME,
    xmin_setting_from_dirname,
)

PAPER_TABLE_NAME = "paired_gof_paper_table.csv"
PLOT_STEM = "paired_gof_comparison"
CAPTION_NAME = "paired_gof_comparison_caption.tex"
HOLM_ROW = "mcnemar_p_holm_k_le_20"
SIGNIFICANCE = 0.05
MAX_P_ANNOTATIONS = 4
KS = tuple(range(1, 21))
FIGSIZE_IN = (14.5 / 2.54, 7.0 / 2.54)
DEFAULT_YLIM = (-20.0, 100.0)
AXIS_NUMBER_FONTSIZE = 8
CONFIG_DIR_RE = re.compile(
    r"^min\[(\d+)\]-gofp\[([0-9.]+)\]-pcomp\[([0-9.]+)\]$"
)


@dataclass(frozen=True)
class PairedPlotSetting:
    """One plottable paired-GOF configuration."""

    label: str
    output_dir: Path
    frame: pd.DataFrame
    min_types: int
    gof_p: float
    comparison_p: float
    xmin_label: str
    source: str


def parse_config_dirname(name: str) -> tuple[int, float, float]:
    """Parse ``min[50]-gofp[0.10]-pcomp[0.10]`` into thresholds."""
    match = CONFIG_DIR_RE.match(str(name))
    if match is None:
        raise ValueError(f"not a clauset config directory name: {name}")
    return int(match.group(1)), float(match.group(2)), float(match.group(3))


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def paired_frame_from_paper_table(path: Path) -> pd.DataFrame:
    """Load the wide paper table into a long frame for plotting."""
    table = pd.read_csv(path).set_index("statistic")
    rows: list[dict[str, object]] = []
    for k in KS:
        column = f"k={k}"
        if column not in table.columns:
            rows.append(
                {
                    "k": k,
                    "paired_N": 0,
                    "difference_pp": math.nan,
                    "ci95_lower_pp": math.nan,
                    "ci95_upper_pp": math.nan,
                    "p_holm": math.nan,
                }
            )
            continue
        paired_n = _finite(table.at["paired_N", column])
        rows.append(
            {
                "k": k,
                "paired_N": 0 if paired_n is None else int(round(paired_n)),
                "difference_pp": _finite(table.at["difference_pp", column]),
                "ci95_lower_pp": _finite(table.at["ci95_lower_pp", column]),
                "ci95_upper_pp": _finite(table.at["ci95_upper_pp", column]),
                "p_holm": _finite(table.at[HOLM_ROW, column]),
            }
        )
    return pd.DataFrame(rows)


def estimate_available(row: Mapping[str, object] | pd.Series) -> bool:
    return (
        _finite(row["difference_pp"]) is not None
        and _finite(row["ci95_lower_pp"]) is not None
        and _finite(row["ci95_upper_pp"]) is not None
    )


def shared_ylim(frames: Sequence[pd.DataFrame]) -> tuple[float, float]:
    """Common y-limits from finite CI endpoints and zero, with padding."""
    values: list[float] = [0.0]
    for frame in frames:
        for column in ("difference_pp", "ci95_lower_pp", "ci95_upper_pp"):
            for value in frame[column]:
                number = _finite(value)
                if number is not None:
                    values.append(number)
    low = min(values)
    high = max(values)
    span = max(high - low, 1.0)
    pad = max(4.0, 0.06 * span)
    # Extra headroom for possible Holm annotations above upper CI caps.
    return low - pad, high + pad + 6.0


def format_holm_p(p_value: float) -> str:
    if p_value < 0.001:
        return "Holm p < 0.001"
    return f"Holm p = {p_value:.3f}"


def build_paired_gof_caption(
    *,
    min_types: int,
    gof_p: float,
    comparison_p: float,
    xmin_label: str,
) -> str:
    """Caption prose without colons, semicolons, or en-dashes."""
    return (
        "Paired GOF comparison of trace variants versus finite-order n-grams. "
        "Differences are trace variants minus finite order. "
        "Only logs eligible at both levels enter each comparison. "
        f"Eligibility requires at least {int(min_types)} fitted types. "
        f"GOF non-rejection uses threshold {float(gof_p):g}. "
        f"The folder also records comparison threshold {float(comparison_p):g}, "
        "which does not enter the paired GOF outcome. "
        f"The xmin policy is {xmin_label}. "
        "Error bars are pointwise 95 percent Newcombe confidence intervals "
        "for paired proportions. "
        "Filled points indicate two-sided exact McNemar p-values below 0.05 "
        "after Holm adjustment for the planned comparison family of "
        "abstraction orders 1 through 20. "
        "N gives the number of paired logs. "
        "NA indicates an unavailable estimate. "
        "The intervals are pointwise. "
        "An interval that excludes zero need not remain significant after "
        "Holm adjustment."
    )


def write_paired_gof_caption(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"Saved: {path}")


def plot_paired_gof_comparison(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Write ``paired_gof_comparison.pdf`` and ``.png`` for one setting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ordered = frame.set_index("k").reindex(list(KS)).reset_index()
    if ylim is None:
        ylim = DEFAULT_YLIM
    y0, y1 = ylim

    fig, ax = plt.subplots(figsize=FIGSIZE_IN)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#d0d0d0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(0.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)

    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(y0, y1)
    ax.set_xticks(list(KS))
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=3, pad=2)
    # Dense grid every 10 pp; labeled ticks every 20 pp (…, -20%, 0%, 20%, …).
    grid_ticks = np.arange(y0, y1 + 1e-9, 10.0)
    label_ticks = np.arange(y0, y1 + 1e-9, 20.0)
    ax.set_yticks(grid_ticks, minor=False)
    ax.set_yticklabels(
        [f"{int(t)}%" if t in set(label_ticks) else "" for t in grid_ticks]
    )
    ax.tick_params(axis="y", labelsize=AXIS_NUMBER_FONTSIZE)

    significant_ks: list[int] = []
    for _, row in ordered.iterrows():
        k = int(row["k"])
        if not estimate_available(row):
            ax.text(
                k,
                0.0,
                "NA",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#b0b0b0",
                zorder=3,
            )
            continue
        diff = float(row["difference_pp"])
        lo = float(row["ci95_lower_pp"])
        hi = float(row["ci95_upper_pp"])
        p_holm = _finite(row["p_holm"])
        is_sig = p_holm is not None and p_holm < SIGNIFICANCE
        if is_sig:
            significant_ks.append(k)
        ax.errorbar(
            k,
            diff,
            yerr=np.array([[diff - lo], [hi - diff]], dtype=float),
            fmt="none",
            ecolor="#4a4a4a",
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=2,
        )
        ax.plot(
            k,
            diff,
            marker="o",
            markersize=5.5,
            markerfacecolor="black" if is_sig else "white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            linestyle="None",
            zorder=4,
        )

    annotate = 0 < len(significant_ks) <= MAX_P_ANNOTATIONS
    if annotate:
        for _, row in ordered.iterrows():
            k = int(row["k"])
            p_holm = _finite(row["p_holm"])
            if p_holm is None or p_holm >= SIGNIFICANCE or not estimate_available(row):
                continue
            hi = float(row["ci95_upper_pp"])
            ax.text(
                k,
                hi + 0.03 * (y1 - y0),
                format_holm_p(p_holm),
                ha="center",
                va="bottom",
                fontsize=7,
                color="black",
                zorder=5,
            )

    ax.set_ylabel(
        "Difference in GOF non-rejection share\n"
        "compared to trace variant level",
        fontsize=8,
        labelpad=4,
        linespacing=1.15,
    )
    ax.set_xlabel("")

    # Two labeled rows beneath the axis: k values, then N values.
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    k_row_y = -0.08
    n_row_y = -0.22
    label_x = -0.01
    ax.text(
        label_x,
        k_row_y,
        r"$k$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=AXIS_NUMBER_FONTSIZE,
        clip_on=False,
    )
    ax.text(
        label_x,
        n_row_y,
        r"$N$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=AXIS_NUMBER_FONTSIZE,
        clip_on=False,
    )
    for _, row in ordered.iterrows():
        k = int(row["k"])
        ax.text(
            k,
            k_row_y,
            str(k),
            transform=trans,
            ha="center",
            va="top",
            fontsize=AXIS_NUMBER_FONTSIZE,
            clip_on=False,
        )
        ax.text(
            k,
            n_row_y,
            str(int(row["paired_N"])),
            transform=trans,
            ha="center",
            va="top",
            fontsize=AXIS_NUMBER_FONTSIZE,
            clip_on=False,
        )

    if not any(estimate_available(row) for _, row in ordered.iterrows()):
        ax.text(
            0.5,
            0.5,
            "No paired estimates available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#666666",
        )

    fig.subplots_adjust(left=0.14, right=0.99, top=0.90, bottom=0.28)
    fig.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    png_path = output_path.with_suffix(".png")
    if png_path != output_path:
        fig.savefig(png_path, dpi=300)
        print(f"Saved: {png_path}")
    plt.close(fig)


def write_paired_gof_plot_bundle(
    setting: PairedPlotSetting,
    *,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Write plot PDF/PNG and caption for one setting."""
    plot_paired_gof_comparison(
        setting.frame,
        setting.output_dir / f"{PLOT_STEM}.pdf",
        ylim=ylim,
    )
    caption = build_paired_gof_caption(
        min_types=setting.min_types,
        gof_p=setting.gof_p,
        comparison_p=setting.comparison_p,
        xmin_label=setting.xmin_label,
    )
    write_paired_gof_caption(setting.output_dir / CAPTION_NAME, caption)


def discover_clauset_paired_settings(
    experiment_dir: Path,
) -> tuple[list[PairedPlotSetting], list[str]]:
    """Settings that already have ``paired_gof_paper_table.csv``."""
    experiment_dir = Path(experiment_dir)
    found: list[PairedPlotSetting] = []
    skipped: list[str] = []
    for xmin_path in sorted(experiment_dir.glob("xmin*")):
        if not xmin_path.is_dir():
            continue
        try:
            xmin_value = xmin_setting_from_dirname(xmin_path.name)
        except ValueError:
            continue
        clauset_root = xmin_path / CLAUSE_RESULTS_DIRNAME
        if not clauset_root.is_dir():
            continue
        for config_dir in sorted(clauset_root.iterdir()):
            if not config_dir.is_dir():
                continue
            paper = config_dir / PAPER_TABLE_NAME
            if not paper.is_file():
                skipped.append(
                    f"{xmin_path.name}/{config_dir.name} (missing {PAPER_TABLE_NAME})"
                )
                continue
            try:
                min_types, gof_p, comparison_p = parse_config_dirname(config_dir.name)
            except ValueError as exc:
                skipped.append(f"{xmin_path.name}/{config_dir.name} ({exc})")
                continue
            xmin_label = "adaptive" if xmin_value is None else str(int(xmin_value))
            found.append(
                PairedPlotSetting(
                    label=f"{xmin_path.name}/{config_dir.name}",
                    output_dir=config_dir,
                    frame=paired_frame_from_paper_table(paper),
                    min_types=min_types,
                    gof_p=gof_p,
                    comparison_p=comparison_p,
                    xmin_label=xmin_label,
                    source=str(paper),
                )
            )
    return found, skipped


def discover_all_paired_plot_settings(
    experiment_dir: Path,
) -> tuple[list[PairedPlotSetting], list[str]]:
    """Clauset ``clauset_results`` folders that already have a paper table."""
    return discover_clauset_paired_settings(experiment_dir)


def compose_paired_gof_plots(
    experiment_dir: Path,
) -> tuple[list[PairedPlotSetting], list[str]]:
    """Write paired GOF plots for every clauset_results setting under an experiment."""
    settings, skipped = discover_all_paired_plot_settings(experiment_dir)
    if not settings:
        return settings, skipped
    for setting in settings:
        print(
            "Paired GOF plot: "
            f"{setting.label} (source={setting.source})"
        )
        write_paired_gof_plot_bundle(setting, ylim=DEFAULT_YLIM)
    return settings, skipped
