"""Batch explore GOF jumps / abstraction-order diagnostics for real logs.

For each log under a Clauset results root (``real``, ``real_xmin1``, or
``real_xmin2``), write:

1. Trace-length distribution (box + CDF)
2. GOF ``p`` vs abstraction order ``k`` (integer x ticks)
3. Empirical vs saved-α CDF panels with max-|Δ| markers
4. Type / occurrence transition overview CSV (variants column = types/occ only)

Not wired into ``main.py``. Run:

    python analyses/powerlaw_statistics/08_explore_gof_jumps.py --xmin 2
    python analyses/powerlaw_statistics/08_explore_gof_jumps.py --xmin 1
    python analyses/powerlaw_statistics/08_explore_gof_jumps.py --xmin none
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import powerlaw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.constants import (
    ALL_REAL_LOG_DATASETS,
    ALL_REAL_LOGS_TOKEN,
    ATTACHMENTS_DIR,
    DATA_DICTIONARY_PATH,
    POWERLAW_STATISTICS_DIR,
)
from utils.io import get_data_dictionary, get_event_log_from_dictionary, load_attachments
from utils.io.attachments import START, _extended_trace, _ngram_nodes, trace_completion_data
from utils.powerlaw.analyze import LOWER_BOUNDED_POWER_LAW
from utils.rfc import extract_frequency_counts

SOURCE_CATS = [
    "Extended 1→1",
    "Extended 1→many",
    "Newly completed 1→1",
    "Exhausted 1→0",
    "Complete and retained 1→1",
]
CAT_DISPLAY = {
    "Extended 1→1": "Extended 1→1",
    "Extended 1→many": "Branched 1→many",
    "Newly completed 1→1": "Newly completed 1→1",
    "Exhausted 1→0": "Exhausted 1→0",
    "Complete and retained 1→1": "Complete and retained",
}
MODEL = LOWER_BOUNDED_POWER_LAW


def parse_xmin_token(raw: str) -> int | None:
    text = str(raw).strip().lower()
    if text in {"none", "null", "free", "nan"}:
        return None
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("--xmin must be >= 1 or 'none'")
    return value


def results_root_for_xmin(xmin: int | None) -> Path:
    if xmin is None:
        return REPO_ROOT / POWERLAW_STATISTICS_DIR / "real"
    return REPO_ROOT / POWERLAW_STATISTICS_DIR / f"real_xmin{xmin}"


def concept_sort_key(concept: str) -> tuple[int, int | str]:
    if isinstance(concept, str) and concept.startswith("n") and concept[1:].isdigit():
        return (0, int(concept[1:]))
    return (1, str(concept))


def k_from_concept(concept: str) -> int | None:
    if isinstance(concept, str) and concept.startswith("n") and concept[1:].isdigit():
        return int(concept[1:])
    return None


def attachment_path(attachments_root: Path, concept: str, log_name: str) -> Path:
    return attachments_root / concept / log_name / "attachments.csv.gz"


def load_frequencies(attachments_root: Path, concept: str, log_name: str) -> np.ndarray:
    path = attachment_path(attachments_root, concept, log_name)
    if not path.is_file():
        raise FileNotFoundError(path)
    return extract_frequency_counts(load_attachments(path))


def load_log_summary(summary_all: Path, log_name: str) -> pd.DataFrame:
    df = pd.read_csv(summary_all)
    df = df[(df["log_name"] == log_name) & (df["model"] == MODEL)].copy()
    df["concept"] = df["concept"].astype(str)
    return df.sort_values("concept", key=lambda s: s.map(concept_sort_key)).reset_index(
        drop=True
    )


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Trace-length distribution
# ---------------------------------------------------------------------------


def write_trace_length_analysis(
    log_name: str,
    out_dir: Path,
    *,
    data_dictionary: Mapping[str, Mapping[str, object]],
) -> list[dict[str, Any]]:
    event_log = get_event_log_from_dictionary(log_name, data_dictionary)
    trace_rows = trace_completion_data(event_log)
    case_lengths = np.asarray(
        [len(row["activity_sequence"]) for row in trace_rows],
        dtype=float,
    )
    n_traces = int(case_lengths.size)
    if n_traces == 0:
        raise SystemExit(f"Error: {log_name} has no traces.")

    len_mean = float(np.mean(case_lengths))
    len_median = float(np.median(case_lengths))
    len_q1, len_q3 = np.quantile(case_lengths, [0.25, 0.75])
    stats = {
        "log_name": log_name,
        "n_traces": n_traces,
        "mean": len_mean,
        "median": len_median,
        "q1": float(len_q1),
        "q3": float(len_q3),
        "min": float(case_lengths.min()),
        "max": float(case_lengths.max()),
    }
    pd.DataFrame([stats]).to_csv(out_dir / "trace_length_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.boxplot(
        case_lengths,
        vert=True,
        widths=0.45,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35},
    )
    ax.scatter([1], [len_mean], marker="D", s=45, color="tab:red", zorder=3, label=f"mean={len_mean:.2f}")
    ax.plot([], [], color="C0", lw=1.2, label=f"median={len_median:.2f}")
    ax.set_xticks([1])
    ax.set_xticklabels([log_name])
    ax.set_ylabel("trace length")
    ax.set_title(f"{log_name} trace-length box plot")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    lengths_sorted = np.sort(case_lengths)
    cdf_pct = 100.0 * np.arange(1, n_traces + 1) / n_traces
    ax.step(lengths_sorted, cdf_pct, where="post", color="tab:blue", lw=1.6)
    ax.axvline(len_median, color="0.35", ls="--", lw=1.0, label=f"median={len_median:.2f}")
    ax.axvline(len_mean, color="tab:red", ls=":", lw=1.2, label=f"mean={len_mean:.2f}")
    ax.set_xlabel("trace length")
    ax.set_ylabel("cumulative percentage of traces")
    ax.set_ylim(0, 100)
    ax.set_title(f"{log_name} trace-length CDF")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"{log_name} case trace lengths (n={n_traces})", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir / "trace_length_distribution.pdf")
    return trace_rows


# ---------------------------------------------------------------------------
# 2. GOF p vs k
# ---------------------------------------------------------------------------


def write_gof_p_vs_k(
    log_name: str,
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    experiment_id: str,
) -> None:
    plot_df = summary.copy()
    plot_df["k"] = plot_df["concept"].map(k_from_concept)
    plot_df = plot_df.dropna(subset=["k", "gof_p"]).sort_values("k")
    plot_df["k"] = plot_df["k"].astype(int)
    plot_df[["concept", "k", "gof_p", "KS_D", "alpha", "xmin", "n_fitted_types"]].to_csv(
        out_dir / "gof_p_vs_k.csv", index=False
    )
    if plot_df.empty:
        print(f"  {log_name}: no n-gram GOF rows to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(plot_df["k"], plot_df["gof_p"], "o-", color="tab:blue", lw=1.8, ms=5)
    ax.axhline(0.10, color="gray", ls=":", lw=1, label="p = 0.10")
    ax.set_xlabel("k")
    ax.set_ylabel(r"$p_{\mathrm{GOF}}$")
    ax.set_ylim(0.0, 1.0)
    ks = plot_df["k"].to_numpy(dtype=int)
    ax.set_xticks(ks)
    ax.xaxis.set_major_locator(mticker.FixedLocator(ks.tolist()))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.set_xlim(float(ks.min()) - 0.5, float(ks.max()) + 0.5)
    ax.set_title(f"{log_name} ({experiment_id}): GOF p-value vs abstraction order")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(fig, out_dir / "gof_p_vs_k.pdf")


# ---------------------------------------------------------------------------
# 3. Empirical vs saved-α CDF with max |Δ|
# ---------------------------------------------------------------------------


def _max_deviation_point(
    kind: str, tail_data: np.ndarray, model: Any, saved_xmin: int
) -> tuple[float, float, float, float]:
    if kind == "cdf":
        x, y_emp = powerlaw.cdf(tail_data, xmin=saved_xmin)
        y_model = np.asarray(model.cdf(np.asarray(x, dtype=float)), dtype=float)
    elif kind == "ccdf":
        x, y_emp = powerlaw.ccdf(tail_data, xmin=saved_xmin)
        y_model = np.asarray(model.ccdf(np.asarray(x, dtype=float)), dtype=float)
    else:
        raise ValueError(kind)
    x = np.asarray(x, dtype=float)
    y_emp = np.asarray(y_emp, dtype=float)
    resid = y_emp - y_model
    i_star = int(np.nanargmax(np.abs(resid)))
    return float(x[i_star]), float(y_emp[i_star]), float(y_model[i_star]), float(resid[i_star])


def _label_xmin_frequency(ax: plt.Axes, saved_xmin: int, *, y_linear: bool, x_left: float = 1.0) -> None:
    saved_xmin_f = float(saved_xmin)
    ax.axvline(saved_xmin_f, color="0.45", ls=":", lw=1.0, zorder=2)
    ymin, ymax = ax.get_ylim()
    if y_linear:
        y_text = ymin + 0.04 * (ymax - ymin)
    else:
        y_text = ymin * ((ymax / ymin) ** 0.04) if ymin > 0 else ymax * 0.05
    ax.text(
        saved_xmin_f,
        y_text,
        f"$x_{{\\mathrm{{min}}}}={int(saved_xmin)}$",
        rotation=90,
        va="bottom",
        ha="right",
        fontsize=8,
        color="0.35",
        clip_on=False,
    )
    ticks = [t for t in ax.get_xticks() if np.isfinite(t) and t >= x_left]
    if not any(np.isclose(t, saved_xmin_f) for t in ticks):
        ticks = sorted(ticks + [saved_xmin_f])
    ax.set_xticks(ticks)

    def _fmt(x: float, _pos: Any = None, xmin: float = saved_xmin_f) -> str:
        if not np.isfinite(x) or x <= 0:
            return ""
        if np.isclose(x, xmin):
            return f"{int(xmin)}\n(=xmin)"
        if abs(x - round(x)) < 1e-8:
            return str(int(round(x)))
        return f"{x:g}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
    _, x_right = ax.get_xlim()
    ax.set_xlim(x_left, max(x_right, saved_xmin_f * 1.01))
    ax.set_xmargin(0.0)


def plot_saved_pl_cdf(
    ax: plt.Axes,
    frequencies: np.ndarray,
    saved_alpha: float,
    saved_xmin: float,
    *,
    color: str,
    title: str,
) -> dict[str, float]:
    frequencies = np.asarray(frequencies, dtype=int)
    saved_xmin_i = int(saved_xmin)
    tail_data = frequencies[frequencies >= saved_xmin_i]
    if tail_data.size == 0:
        ax.set_visible(False)
        return {"x": float("nan"), "delta": float("nan"), "abs_delta": float("nan")}
    x_grid = np.arange(saved_xmin_i, int(tail_data.max()) + 1)

    powerlaw.plot_cdf(tail_data, ax=ax, color=color, linewidth=1.8, label="Empirical CDF")
    model = powerlaw.Power_Law(
        xmin=saved_xmin_i,
        parameters=[float(saved_alpha)],
        discrete=True,
    )
    model.plot_cdf(
        data=x_grid,
        ax=ax,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Saved PL (α={float(saved_alpha):.4f})",
    )

    x_star, y_emp, y_model, resid = _max_deviation_point("cdf", tail_data, model, saved_xmin_i)
    ax.plot([x_star, x_star], [y_emp, y_model], color="crimson", lw=1.6, zorder=4, label="Max |Δ|")
    ax.scatter([x_star], [y_emp], color=color, s=45, zorder=5, edgecolor="crimson")
    ax.scatter([x_star], [y_model], color="black", s=45, zorder=5, edgecolor="crimson")
    ax.annotate(
        f"max |Δ|={abs(resid):.4f}\nat x={x_star:.0f}",
        xy=(x_star, 0.5 * (y_emp + y_model)),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=8,
        color="crimson",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", alpha=0.85),
    )

    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.set_ylim(0.0, 1.0)
    _label_xmin_frequency(ax, saved_xmin_i, y_linear=True, x_left=1.0)
    ax.set_xlabel("x (Occurrence Frequency)")
    ax.set_ylabel(r"$P(X < x \mid X \geq x_{\mathrm{min}})$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(1.0, ax.get_xlim()[1])
    return {"x": x_star, "delta": resid, "abs_delta": abs(resid)}


def write_cdf_grids(
    log_name: str,
    summary: pd.DataFrame,
    out_dir: Path,
    *,
    attachments_root: Path,
) -> None:
    ngram = summary[summary["concept"].map(k_from_concept).notna()].copy()
    ngram = ngram.sort_values("concept", key=lambda s: s.map(concept_sort_key))
    if ngram.empty:
        print(f"  {log_name}: no n-gram rows for CDF grid")
        return

    panel_colors = plt.cm.tab10.colors
    n_panels = len(ngram)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.6 * n_rows), sharex=False, sharey=True)
    axes_flat = np.asarray(axes).ravel()

    rows_out: list[dict[str, Any]] = []
    for i, (_, csv_row) in enumerate(ngram.iterrows()):
        ax = axes_flat[i]
        concept = str(csv_row["concept"])
        k = k_from_concept(concept)
        try:
            freqs = load_frequencies(attachments_root, concept, log_name)
        except FileNotFoundError:
            ax.set_visible(False)
            print(f"  {log_name} CDF {concept}: missing attachments — skipped")
            continue
        if pd.isna(csv_row["alpha"]) or pd.isna(csv_row["xmin"]):
            ax.set_visible(False)
            continue
        saved_alpha = float(csv_row["alpha"])
        saved_xmin = float(csv_row["xmin"])
        color = panel_colors[i % len(panel_colors)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = plot_saved_pl_cdf(
                ax,
                freqs,
                saved_alpha,
                saved_xmin,
                color=color,
                title=(
                    f"k={k}  |  α={saved_alpha:.3f}, "
                    f"n_fit={int(csv_row['n_fitted_types'])}, "
                    f"KS={float(csv_row['KS_D']):.4f}, "
                    f"p={100.0 * float(csv_row['gof_p']):.1f}%"
                ),
            )
        ks = float(csv_row["KS_D"])
        rows_out.append(
            {
                "log_name": log_name,
                "concept": concept,
                "k": k,
                "x_star": info["x"],
                "delta": info["delta"],
                "abs_delta": info["abs_delta"],
                "reported_KS_D": ks,
                "matches_KS_D": bool(
                    np.isfinite(info["abs_delta"])
                    and np.isclose(info["abs_delta"], ks, rtol=0, atol=1e-10)
                ),
            }
        )

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{log_name} empirical vs saved-α CDF", y=1.01)
    fig.tight_layout()
    for _ax in axes_flat:
        if _ax.get_visible():
            _ax.set_xlim(1.0, max(_ax.get_xlim()[1], 1.0))
    _savefig(fig, out_dir / "cdf_grid.pdf")
    pd.DataFrame(rows_out).to_csv(out_dir / "cdf_max_deviation.csv", index=False)


# ---------------------------------------------------------------------------
# 4. Type / occurrence transition overview
# ---------------------------------------------------------------------------


def _emit_occurrences(case_id: int, seq: tuple[str, ...], k: int) -> list[dict[str, Any]]:
    work = _extended_trace(seq, k, True)
    if k == 1:
        return [
            {
                "case_id": case_id,
                "pos": i,
                "node_id": str(activity),
                "is_complete": False,
                "tokens": (activity,),
            }
            for i, activity in enumerate(work)
        ]
    if k >= len(work):
        return [
            {
                "case_id": case_id,
                "pos": 0,
                "node_id": str(work),
                "is_complete": True,
                "tokens": tuple(work),
            }
        ]
    return [
        {
            "case_id": case_id,
            "pos": i,
            "node_id": str(work[i : i + k]),
            "is_complete": False,
            "tokens": tuple(work[i : i + k]),
        }
        for i in range(len(work) - k + 1)
    ]


def _classify_source_type(
    *,
    n_children: int,
    child_ids: set[str],
    src_is_complete: bool,
    tgt_complete_types: set[str],
) -> str:
    if n_children == 0:
        return "Exhausted 1→0"
    if n_children == 1:
        kid = next(iter(child_ids))
        kid_complete = kid in tgt_complete_types
        if src_is_complete and kid_complete:
            return "Complete and retained 1→1"
        if (not src_is_complete) and kid_complete:
            return "Newly completed 1→1"
        return "Extended 1→1"
    return "Extended 1→many"


def analyze_transition(k: int, sequences: list[tuple[str, ...]]) -> dict[str, Any]:
    src: list[dict[str, Any]] = []
    tgt: list[dict[str, Any]] = []
    for cid, seq in enumerate(sequences):
        src.extend(_emit_occurrences(cid, seq, k))
        tgt.extend(_emit_occurrences(cid, seq, k + 1))

    src_by_pos = {(o["case_id"], o["pos"]): o for o in src}
    src_by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for o in src:
        src_by_case[o["case_id"]].append(o)

    parent_children: dict[str, set[str]] = defaultdict(set)
    parent_child_occ: Counter[str] = Counter()
    src_successor_keys: set[tuple[int, int]] = set()
    introduced_type_set: set[str] = set()
    introduced_occ = 0
    tgt_parent_types: dict[str, set[str]] = defaultdict(set)

    for t in tgt:
        cid = t["case_id"]
        parent = None
        if k == 1:
            pref = t["tokens"][0]
            if pref == START:
                introduced_occ += 1
                introduced_type_set.add(t["node_id"])
                continue
            parent = src_by_pos.get((cid, t["pos"] - 1))
        else:
            src_complete = [o for o in src_by_case[cid] if o["is_complete"]]
            if t["is_complete"] and src_complete:
                parent = src_complete[0]
            elif t["is_complete"]:
                parent = src_by_pos.get((cid, 0))
            else:
                parent = src_by_pos.get((cid, t["pos"]))

        if parent is None:
            continue
        parent_children[parent["node_id"]].add(t["node_id"])
        parent_child_occ[parent["node_id"]] += 1
        src_successor_keys.add((parent["case_id"], parent["pos"]))
        tgt_parent_types[t["node_id"]].add(parent["node_id"])

    src_types = {o["node_id"] for o in src}
    tgt_types = {o["node_id"] for o in tgt}
    src_complete_types = {o["node_id"] for o in src if o["is_complete"]}
    tgt_complete_types = {o["node_id"] for o in tgt if o["is_complete"]}
    src_occ_by_type = Counter(o["node_id"] for o in src)
    src_no_succ_by_type = Counter(
        o["node_id"] for o in src if (o["case_id"], o["pos"]) not in src_successor_keys
    )

    category_of = {
        nid: _classify_source_type(
            n_children=len(parent_children.get(nid, set())),
            child_ids=parent_children.get(nid, set()),
            src_is_complete=nid in src_complete_types,
            tgt_complete_types=tgt_complete_types,
        )
        for nid in src_types
    }
    cat_types = {c: {n for n, cat in category_of.items() if cat == c} for c in SOURCE_CATS}

    extra_branch = int(sum(max(len(kids) - 1, 0) for kids in parent_children.values()))
    n_exhausted = len(cat_types["Exhausted 1→0"])
    n_introduced = len(introduced_type_set)
    newly_complete_targets = {
        tid for tid in tgt_complete_types if tid not in src_complete_types
    }
    complete_tgt_occ = sum(1 for o in tgt if o["is_complete"])

    return {
        "k": k,
        "cat_types": cat_types,
        "types_before": len(src_types),
        "extra_branch": extra_branch,
        "n_exhausted": n_exhausted,
        "n_introduced": n_introduced,
        "types_after": len(tgt_types),
        "occ_before": len(src),
        "occ_no_succ": int(sum(src_no_succ_by_type.values())),
        "introduced_occ": introduced_occ,
        "occ_after": len(tgt),
        "newly_complete_target_types": len(newly_complete_targets),
        "complete_target_types": len(tgt_complete_types),
        "complete_target_occurrences": complete_tgt_occ,
        "src_occ_by_type": src_occ_by_type,
        "src_no_succ_by_type": src_no_succ_by_type,
        "parent_children": parent_children,
        "parent_child_occ": parent_child_occ,
    }


def write_transition_overview(
    log_name: str,
    out_dir: Path,
    *,
    trace_rows: Sequence[Mapping[str, Any]],
    ngram_ks: Sequence[int],
) -> None:
    sequences = [tuple(row["activity_sequence"]) for row in trace_rows]
    k_values = sorted(int(k) for k in ngram_ks)
    if not k_values:
        print(f"  {log_name}: no n-gram k values for transition table")
        return

    indep_totals: dict[Any, dict[str, int]] = {}
    for k in k_values:
        nodes: list[str] = []
        for seq in sequences:
            nodes.extend(_ngram_nodes(seq, k, True))
        indep_totals[k] = {"n_types": len(set(nodes)), "n_occ": len(nodes)}
    variant_nodes = [str(seq) for seq in sequences]
    indep_totals["variants"] = {
        "n_types": len(set(variant_nodes)),
        "n_occ": len(variant_nodes),
    }

    transition_by_dest: dict[int, dict[str, Any]] = {}
    for k in k_values[:-1]:
        transition_by_dest[k + 1] = analyze_transition(k, sequences)

    def _cell(k: int, field: str) -> Any:
        if field in {"types_after", "occ_after"}:
            return indep_totals[k]["n_types" if field == "types_after" else "n_occ"]
        if field == "types_before":
            return indep_totals[k - 1]["n_types"] if k >= 2 else None
        if field == "occ_before":
            return indep_totals[k - 1]["n_occ"] if k >= 2 else None
        if k == 1 and field == "introduced_occ":
            return indep_totals[1]["n_occ"]
        if k == 1 and field == "n_introduced":
            return indep_totals[1]["n_types"]
        if k < 2 or k not in transition_by_dest:
            return None
        res = transition_by_dest[k]
        if field.startswith("cat:"):
            return len(res["cat_types"][field[len("cat:") :]])
        return res[field]

    row_specs = (
        [(CAT_DISPLAY[cat], f"cat:{cat}") for cat in SOURCE_CATS]
        + [
            ("Types before", "types_before"),
            ("+ Extra from branching", "extra_branch"),
            ("− Exhausted types", "n_exhausted"),
            ("+ Introduced types", "n_introduced"),
            ("Types after", "types_after"),
            ("Occurrences before", "occ_before"),
            ("− No successor", "occ_no_succ"),
            ("+ Introduced occurrences", "introduced_occ"),
            ("Occurrences after", "occ_after"),
            ("Newly complete target types", "newly_complete_target_types"),
            ("Complete target types in total", "complete_target_types"),
            ("Occurrences in complete targets", "complete_target_occurrences"),
        ]
    )

    overview_rows = []
    for label, field in row_specs:
        row: dict[str, Any] = {"row": label}
        for k in k_values:
            value = _cell(k, field)
            row[f"k={k}"] = value if value is not None else "N/A"
        # Variants endpoint: only types/occurrences after; everything else N/A.
        if field == "types_after":
            row["variants"] = indep_totals["variants"]["n_types"]
        elif field == "occ_after":
            row["variants"] = indep_totals["variants"]["n_occ"]
        else:
            row["variants"] = "N/A"
        overview_rows.append(row)

    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(out_dir / "transition_overview.csv", index=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def process_log(
    log_name: str,
    *,
    results_root: Path,
    attachments_root: Path,
    data_dictionary: Mapping[str, Mapping[str, object]],
    experiment_id: str,
) -> None:
    summary_all = results_root / "summary_all_concepts.csv"
    if not summary_all.is_file():
        raise SystemExit(f"Error: missing {summary_all}")

    summary = load_log_summary(summary_all, log_name)
    if summary.empty:
        print(f"Skipping {log_name}: no lower_bounded rows in {summary_all}")
        return

    out_dir = results_root / "explore_gof_jumps" / log_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{experiment_id}] {log_name} -> {out_dir}")

    print(f"  trace-length analysis…")
    trace_rows = write_trace_length_analysis(
        log_name, out_dir, data_dictionary=data_dictionary
    )

    print(f"  GOF p vs k…")
    write_gof_p_vs_k(log_name, summary, out_dir, experiment_id=experiment_id)

    print(f"  CDF grids…")
    write_cdf_grids(log_name, summary, out_dir, attachments_root=attachments_root)

    ngram_ks = sorted(
        k
        for k in (k_from_concept(c) for c in summary["concept"].astype(str))
        if k is not None
    )
    print(f"  transition overview (k={ngram_ks[0]}..{ngram_ks[-1]} + variants)…")
    write_transition_overview(
        log_name, out_dir, trace_rows=trace_rows, ngram_ks=ngram_ks
    )
    print(f"  done {log_name}")


def resolve_datasets(raw: Sequence[str] | None, summary_all: Path) -> list[str]:
    if raw is None or (len(raw) == 1 and raw[0] == ALL_REAL_LOGS_TOKEN):
        df = pd.read_csv(summary_all, usecols=["log_name"])
        present = set(df["log_name"].astype(str))
        return [name for name in ALL_REAL_LOG_DATASETS if name in present]
    return list(raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write per-log GOF-jump exploration artifacts under "
            "results/powerlaw_statistics/<real|real_xminN>/explore_gof_jumps/<LOG>/"
        )
    )
    parser.add_argument(
        "--xmin",
        type=parse_xmin_token,
        required=True,
        help="Fixed xmin used for the Clauset run: 1, 2, or none (free xmin -> real/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[ALL_REAL_LOGS_TOKEN],
        help="Log names or ALL_REAL_LOGS (default: all logs present in summary_all_concepts.csv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results_root = results_root_for_xmin(args.xmin)
    experiment_id = results_root.name
    summary_all = results_root / "summary_all_concepts.csv"
    if not results_root.is_dir():
        raise SystemExit(f"Error: results root not found: {results_root}")
    if not summary_all.is_file():
        raise SystemExit(
            f"Error: missing {summary_all}. Run combine for this results root first."
        )

    attachments_root = (REPO_ROOT / ATTACHMENTS_DIR).resolve()
    data_dictionary = get_data_dictionary(
        REPO_ROOT / DATA_DICTIONARY_PATH, get_real=True, get_synthetic=False
    )
    datasets = resolve_datasets(args.datasets, summary_all)
    print(
        f"explore_gof_jumps: xmin={args.xmin!r} root={results_root} "
        f"n_logs={len(datasets)}"
    )
    for log_name in datasets:
        process_log(
            log_name,
            results_root=results_root,
            attachments_root=attachments_root,
            data_dictionary=data_dictionary,
            experiment_id=experiment_id,
        )
    print("Done.")


if __name__ == "__main__":
    main()
