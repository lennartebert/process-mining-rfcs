"""Paired comparisons of saved power-law GOF outcomes.

Compares selected abstraction orders (k = 1, 2, 3, 4, 20) with trace variants
on the same real-world study logs. Fits and GOF tests are not recomputed.

The unit of pairing is the log. A log enters a comparison only when both
endpoints are eligible. Eligibility is recomputed from the saved fit:

    eligible = valid fit/GOF result AND n_fitted_types >= min_fitted_types

Existing A–E labels are not used. The binary outcome is GOF non-rejection,

    gof_not_rejected = (gof_p >= gof_threshold),

with ``gof_p`` read as a probability in [0, 1]. That threshold is separate
from the significance level of the paired test. Preference for an alternative
distribution does not enter the outcome.

The difference is the non-rejection proportion at trace variants minus the
proportion at k, on the jointly eligible logs. Its 95% interval is Newcombe's
paired method 10 (Statistics in Medicine 1998; PMID 9839354) and is pointwise,
not multiplicity-adjusted. The two-sided exact McNemar test is Holm-adjusted
across the five k-versus-variants comparisons within each sensitivity
configuration. All configurations are reported.

A separate paper table, ``paired_gof_paper_table.csv``, compares k = 1, ..., 20
with trace variants at adaptive xmin, 30 fitted types, and GOF threshold 0.10.
Its three Holm rows are alternative family definitions (k <= 20, k <= 10, and
k <= 3). Each adjustment is recomputed from the raw p-values. They are not
three independent confirmations of a finding.

Different logs are treated as independent units. Related benchmark logs and
the non-random study collection limit generalisation. GOF non-rejection does
not establish that a power law is true or best, and a non-significant
difference does not establish equivalence. These intervals describe
differences across logs, not Monte Carlo uncertainty in a single GOF p-value.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from utils.constants import ALL_REAL_LOG_DATASETS
from utils.powerlaw import LOWER_BOUNDED_POWER_LAW

# Sensitivity grid. ``None`` is the adaptively estimated xmin experiment.
XMIN_SETTINGS: tuple[int | None, ...] = (1, 2, None)
COMPARISON_KS: tuple[int, ...] = (1, 2, 3, 4, 20)
MIN_FITTED_TYPES_GRID: tuple[int, ...] = (15, 30, 50)
GOF_THRESHOLDS: tuple[float, ...] = (0.05, 0.10)
REFERENCE_CONCEPT = "variants"
PLANNED_FAMILY_SIZE = len(COMPARISON_KS)

# Paper table: one adaptive configuration, every order against trace variants.
# The three Holm rows are alternative families, recomputed from raw p-values.
PAPER_KS: tuple[int, ...] = tuple(range(1, 21))
PAPER_MIN_FITTED_TYPES = 30
PAPER_GOF_THRESHOLD = 0.10
PAPER_XMIN_SETTING = None
PAPER_TABLE_NAME = "paired_gof_paper_table.csv"
PAPER_STATISTIC_ROWS = (
    "paired_N",
    "difference_pp",
    "ci95_lower_pp",
    "ci95_upper_pp",
    "mcnemar_p_raw",
    "mcnemar_p_holm_k_le_20",
    "mcnemar_p_holm_k_le_10",
    "mcnemar_p_holm_k_le_3",
)

# Results-directory names are the saved xmin policy. Fitted xmin is not used
# to choose the experiment: an adaptive fit that selected xmin=1 stays in
# ``real`` and is not treated as the fixed-xmin=1 run.
EXPERIMENT_DIRECTORY = {
    1: "real_xmin1",
    2: "real_xmin2",
    None: "real",
}
XMIN_POLICY = {1: "fixed", 2: "fixed", None: "adaptive"}

CI_METHOD = "newcombe_1998_method_10"
CI_MULTIPLICITY = "pointwise"
GOF_BOUNDARY = "gof_p >= gof_threshold"
MODEL = LOWER_BOUNDED_POWER_LAW
PROVENANCE_DIRECTORY = "results_directory_name"
PROVENANCE_SIDECAR = "experiment_provenance.json"
SIDECAR_NAME = "experiment_provenance.json"

# Newcombe 1998, Table III, method 10. Cells (e, f, g, h) follow the paper:
# e = both positive, f = first only, g = second only, h = neither.
# The interval is for (f - g) / n = p_first - p_second, to 4 decimal places.
NEWCOMBE_1998_TABLE_III_METHOD_10: tuple[tuple[int, int, int, int, float, float], ...] = (
    (36, 12, 2, 0, 0.0569, 0.3404),
    (20, 12, 2, 16, 0.0562, 0.3292),
    (18, 12, 2, 18, 0.0562, 0.3290),
    (36, 14, 0, 0, 0.1528, 0.4167),
    (35, 14, 0, 1, 0.1461, 0.4175),
    (18, 14, 0, 18, 0.1441, 0.3963),
    (2, 97, 1, 0, 0.8721, 0.9854),
    (1, 97, 1, 1, 0.8736, 0.9850),
    (0, 29, 1, 0, 0.6666, 0.9882),
    (2, 98, 0, 0, 0.9178, 0.9945),
    (1, 98, 0, 1, 0.9171, 0.9916),
    (0, 30, 0, 0, 0.8395, 1.0),
    (54, 0, 0, 0, -0.0664, 0.0664),
    (53, 0, 0, 1, -0.0729, 0.0729),
    (30, 0, 0, 24, -0.0358, 0.0358),
    (29, 0, 0, 25, -0.0354, 0.0354),
    (28, 0, 0, 26, -0.0352, 0.0352),
    (27, 0, 0, 27, -0.0351, 0.0351),
)

_REASON_PRIORITY = (
    "duplicate_results",
    "incompatible_run_configuration",
    "incompatible_xmin_configuration",
    "missing_result",
    "invalid_fit",
    "n_fitted_types_missing",
    "below_min_fitted_types",
    "gof_p_out_of_range",
    "gof_p_unavailable",
)
_SKIPPED_GOF_KINDS = {"skipped_invalid_fit", "skipped"}
_COMPARE_FIELDS = ("xmin", "n_fitted_types", "fit_valid", "gof_p", "gof_kind")

SUMMARY_COLUMNS = [
    "xmin_setting",
    "xmin_policy",
    "min_fitted_types",
    "gof_threshold",
    "gof_boundary",
    "k",
    "concept",
    "reference_concept",
    "model",
    "n_intended_logs",
    "n_eligible_k",
    "n_eligible_variants",
    "n_jointly_eligible",
    "N",
    "n00",
    "n01",
    "n10",
    "n11",
    "proportion_k",
    "proportion_variants",
    "paired_difference",
    "ci95_lower",
    "ci95_upper",
    "ci_method",
    "ci_multiplicity",
    "mcnemar_p",
    "mcnemar_p_holm",
    "status",
    "notes",
    "source_experiment",
    "provenance",
]

AUDIT_COLUMNS = [
    "log_name",
    "xmin_setting",
    "xmin_policy",
    "min_fitted_types",
    "gof_threshold",
    "gof_boundary",
    "k",
    "concept",
    "reference_concept",
    "model",
    "source_experiment",
    "provenance",
    "input_path_k",
    "input_path_variants",
    "fitted_xmin_k",
    "fitted_xmin_variants",
    "n_fitted_types_k",
    "n_fitted_types_variants",
    "gof_p_k",
    "gof_p_variants",
    "fit_valid_k",
    "fit_valid_variants",
    "gof_kind_k",
    "gof_kind_variants",
    "eligible_k",
    "eligible_variants",
    "included_in_paired",
    "exclusion_reason_k",
    "exclusion_reason_variants",
    "outcome_k",
    "outcome_variants",
]

METHODS_NOTE = """\
Paired comparison of power-law GOF non-rejection.

The unit of pairing is the event log. Each row uses only logs that are
eligible at both the selected abstraction order and the trace-variant level.
A log that is ineligible at either endpoint is excluded from both sides.
Eligibility is recomputed from the saved fit and GOF result
(valid fit/GOF and n_fitted_types at or above the support cutoff). It is not
taken from previously stored A–E labels, so a result labelled A at a cutoff
of 30 can be eligible at 15. A saved result with no GOF p-value is unavailable.

GOF non-rejection means gof_p >= the configuration's GOF threshold. gof_p is
a probability in [0, 1], not a percentage. This threshold is separate from
the significance level of the McNemar test. Whether an alternative
distribution is preferred does not change the binary outcome.

The difference is the non-rejection proportion at trace variants minus the
proportion at k, in percentage points in this view. A positive value means
non-rejection is more frequent at the trace-variant level. The 95% interval
is Newcombe's paired score method, method 10 of Newcombe (Statistics in
Medicine, 1998; PMID 9839354). Intervals are pointwise. They are not adjusted
for the five comparisons, and an interval that excludes zero need not agree
with the Holm-adjusted exact test.

The test is the two-sided exact McNemar test. Holm adjustment is applied
across the five k-versus-variants comparisons within each of the 18
configurations. The family size stays five when a comparison is unavailable.
Where significance is marked (*), it is Holm-adjusted p < 0.05. The 18
configurations are a sensitivity analysis; all of them are reported.

Different logs are treated as independent units. Related benchmark logs and
non-random selection of the study collection limit generalisation. GOF
non-rejection does not establish that a power law is true or that it is the
best description. A non-significant difference does not establish equivalence.
These intervals concern differences across logs, not Monte Carlo uncertainty
in the individual GOF p-values.

Experiment identity comes from the saved results directory (real = adaptive
xmin, real_xmin1 = fixed xmin 1, real_xmin2 = fixed xmin 2), not from the
fitted xmin. An adaptive fit that selected xmin=1 is not a fixed-xmin=1 run.
Trace variants are taken from the same experiment as k. Missing results are
reported and are not replaced by another experiment.
"""


@lru_cache(maxsize=1)
def _combine_module():
    """Load step 04 so shard concatenation stays the combine-stage loader."""
    path = Path(__file__).resolve().parent / "04_combine_parallel.py"
    spec = importlib.util.spec_from_file_location("powerlaw_combine_parallel", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look the module up in sys.modules while the class is created.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def format_xmin_setting(xmin_setting: int | None) -> str:
    """Serialise a grid setting. ``None`` is the adaptive experiment."""
    if xmin_setting is None:
        return "None"
    return str(int(xmin_setting))


def concept_for_k(k: int) -> str:
    return f"n{int(k)}"


def experiment_directory(statistics_root: Path, xmin_setting: int | None) -> Path:
    try:
        name = EXPERIMENT_DIRECTORY[xmin_setting]
    except KeyError as exc:
        raise ValueError(f"unsupported xmin_setting {xmin_setting!r}") from exc
    return Path(statistics_root) / name


def identify_xmin_setting(experiment_dir: Path) -> tuple[int | None, str]:
    """Return ``(xmin_setting, provenance)`` for a results directory.

    The policy is the saved experiment configuration. Fitted xmin values are
    not consulted. A sidecar ``experiment_provenance.json`` overrides the
    directory name when present; a conflict between the two is an error.
    """
    directory = Path(experiment_dir)
    from_name = EXPERIMENT_DIRECTORY_SETTING.get(directory.name)
    sidecar = directory / SIDECAR_NAME
    if sidecar.is_file():
        setting = _xmin_setting_from_sidecar(sidecar)
        if from_name is not _ABSENT and from_name != setting:
            raise ValueError(
                f"incompatible xmin provenance for {directory}: "
                f"directory name implies {format_xmin_setting(from_name)}, "
                f"{SIDECAR_NAME} implies {format_xmin_setting(setting)}"
            )
        return setting, PROVENANCE_SIDECAR
    if from_name is _ABSENT:
        raise ValueError(
            f"no saved xmin provenance for {directory}; "
            "refusing to infer the experiment from fitted xmin"
        )
    return from_name, PROVENANCE_DIRECTORY


_ABSENT = object()
EXPERIMENT_DIRECTORY_SETTING = {name: setting for setting, name in EXPERIMENT_DIRECTORY.items()}


def _xmin_setting_from_sidecar(path: Path) -> int | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policy = str(payload.get("xmin_policy", "")).strip().lower()
    if policy == "adaptive":
        if payload.get("xmin") not in (None, ""):
            raise ValueError(f"{path} sets xmin_policy=adaptive but also sets xmin")
        return None
    if policy == "fixed":
        xmin = payload.get("xmin")
        if xmin is None:
            raise ValueError(f"{path} sets xmin_policy=fixed without xmin")
        return int(xmin)
    raise ValueError(f"{path} has unrecognised xmin_policy {policy!r}")


def wilson_score_interval(successes: int, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for a single binomial proportion."""
    if n <= 0:
        raise ValueError("Wilson interval requires n > 0")
    probability = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (probability + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(probability * (1.0 - probability) / n + z2 / (4.0 * n * n)) / denom
    return centre - half, centre + half


def newcombe_method10_from_cells(
    e: int,
    f: int,
    g: int,
    h: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Newcombe 1998 method 10 interval for ``(f - g) / n``.

    Method 10 is method 8 with a continuity correction applied to a positive
    phi coefficient: the numerator ``eh - fg`` is replaced by
    ``max(eh - fg - n/2, 0)`` when ``eh > fg``. Phi is 0 when any margin is 0.
    """
    counts = [int(e), int(f), int(g), int(h)]
    if any(count < 0 for count in counts):
        raise ValueError("cell counts must be non-negative")
    n = sum(counts)
    if n <= 0:
        raise ValueError("Newcombe method 10 requires N > 0")
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p1 = (e + f) / n
    p2 = (e + g) / n
    theta = (f - g) / n
    l1, u1 = wilson_score_interval(e + f, n, z)
    l2, u2 = wilson_score_interval(e + g, n, z)
    numerator = float(e * h - f * g)
    if numerator > 0.0:
        numerator = max(numerator - n / 2.0, 0.0)
    margin_product = (e + f) * (g + h) * (e + g) * (f + h)
    phi = 0.0 if margin_product == 0 else numerator / math.sqrt(margin_product)
    dl1 = p1 - l1
    du1 = u1 - p1
    dl2 = p2 - l2
    du2 = u2 - p2
    lower_width = _nonnegative_sqrt(dl1 * dl1 - 2.0 * phi * dl1 * du2 + du2 * du2)
    upper_width = _nonnegative_sqrt(du1 * du1 - 2.0 * phi * du1 * dl2 + dl2 * dl2)
    lower = theta - lower_width
    upper = theta + upper_width
    return _bound_limit(lower), _bound_limit(upper)


def newcombe_method10_variants_minus_k(
    n00: int,
    n01: int,
    n10: int,
    n11: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """95% interval for the variants-minus-k non-rejection difference.

    ``n01`` is rejected at k and not rejected at variants. ``n10`` is the
    opposite discordant cell. The returned interval has the same orientation.
    """
    # Paper cells with the first margin equal to trace variants.
    return newcombe_method10_from_cells(n11, n01, n10, n00, alpha=alpha)


def _nonnegative_sqrt(value: float) -> float:
    if value < -1e-12:
        raise ValueError(f"Newcombe method 10 radicand is negative ({value})")
    return math.sqrt(max(value, 0.0))


def _bound_limit(value: float) -> float:
    if value < -1.0:
        if value < -1.0 - 1e-9:
            raise ValueError(f"Newcombe limit {value} is below -1")
        return -1.0
    if value > 1.0:
        if value > 1.0 + 1e-9:
            raise ValueError(f"Newcombe limit {value} is above 1")
        return 1.0
    return value


def exact_mcnemar_p(n00: int, n01: int, n10: int, n11: int) -> float:
    """Two-sided exact McNemar p-value.

    Uses ``statsmodels`` with ``exact=True`` on
    ``[[n00, n01], [n10, n11]]``. When there are no discordant pairs the
    p-value is 1. ``N = 0`` is not a test and must be handled by the caller.
    """
    if n01 + n10 == 0:
        return 1.0
    result = mcnemar([[int(n00), int(n01)], [int(n10), int(n11)]], exact=True)
    return float(result.pvalue)


def holm_adjust(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm adjustment that keeps the planned family size.

    Unavailable comparisons contribute p = 1 for the adjustment only. Their
    returned adjusted p-values stay missing.
    """
    if not pvalues:
        return []
    filled = [1.0 if pvalue is None else float(pvalue) for pvalue in pvalues]
    adjusted = multipletests(filled, method="holm")[1]
    return [
        None if original is None else float(corrected)
        for original, corrected in zip(pvalues, adjusted)
    ]


def paired_effect(
    n00: int, n01: int, n10: int, n11: int
) -> dict[str, float | None]:
    """Proportions and difference on one jointly eligible set of logs."""
    total = int(n00) + int(n01) + int(n10) + int(n11)
    if total == 0:
        return {
            "N": 0,
            "proportion_k": None,
            "proportion_variants": None,
            "paired_difference": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "mcnemar_p": None,
        }
    proportion_k = (n10 + n11) / total
    proportion_variants = (n01 + n11) / total
    difference = proportion_variants - proportion_k
    lower, upper = newcombe_method10_variants_minus_k(n00, n01, n10, n11)
    return {
        "N": total,
        "proportion_k": proportion_k,
        "proportion_variants": proportion_variants,
        "paired_difference": difference,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "mcnemar_p": exact_mcnemar_p(n00, n01, n10, n11),
    }


@dataclass(frozen=True)
class EndpointAssessment:
    """One log at one concept, under one support cutoff and GOF threshold."""

    eligible: bool
    reasons: tuple[str, ...]
    outcome: int | None
    fit_valid: bool | None
    n_fitted_types: int | None
    gof_p: float | None
    fitted_xmin: float | None
    gof_kind: str | None
    input_path: str | None

    @property
    def exclusion_reason(self) -> str:
        return "; ".join(self.reasons)

    @property
    def primary_reason(self) -> str | None:
        if not self.reasons:
            return None
        rank = {reason: index for index, reason in enumerate(_REASON_PRIORITY)}
        return min(self.reasons, key=lambda reason: rank.get(reason, len(rank)))


@dataclass(frozen=True)
class LoadedConcept:
    by_log: Mapping[str, list[dict]]
    sources_disagree: frozenset[str]
    directory_found: bool
    results_found: bool


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_bool(value: object) -> bool | None:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "1.0", "yes"}:
        return True
    if text in {"false", "0", "0.0", "no"}:
        return False
    return None


def _parse_int(value: object) -> int | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or abs(number - round(number)) > 1e-6:
        return None
    return int(round(number))


def _parse_prob(value: object) -> tuple[float | None, str | None]:
    """Return ``(probability, problem)``. Problem is ``out_of_range`` or ``missing``."""
    if _is_missing(value):
        return None, "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "missing"
    if not math.isfinite(number):
        return None, "missing"
    if number < 0.0 or number > 1.0:
        return number, "out_of_range"
    return number, None


def _parse_xmin(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_text(value: object) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    return text or None


def _values_equal(field: str, left: object, right: object) -> bool:
    if field == "fit_valid":
        return _parse_bool(left) == _parse_bool(right)
    if field == "gof_kind":
        return _parse_text(left) == _parse_text(right)
    if field == "n_fitted_types":
        return _parse_int(left) == _parse_int(right)
    if field in {"xmin", "gof_p"}:
        a = _parse_xmin(left)
        b = _parse_xmin(right)
        if a is None or b is None:
            return a is None and b is None
        return abs(a - b) <= 1e-8
    return left == right


def _records_agree(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return all(_values_equal(field, left.get(field), right.get(field)) for field in _COMPARE_FIELDS)


def load_concept_results(
    experiment_dir: Path,
    concept: str,
    study_logs: Sequence[str],
) -> LoadedConcept:
    """Load one concept with the combine stage's shard reader.

    Shards and ``summary.csv`` must agree when both contain a log. Duplicate
    rows are retained so the caller can reject them instead of choosing one.
    """
    model_dir = Path(experiment_dir) / concept / MODEL
    allowed = set(study_logs)
    if not model_dir.is_dir():
        return LoadedConcept({}, frozenset(), Path(experiment_dir).is_dir(), False)

    combine = _combine_module()
    shard_paths = combine._filter_shards_by_log(
        model_dir.glob("summary_*.csv"),
        prefix="summary_",
        datasets=study_logs,
    )
    if shard_paths:
        # Shard dtypes are restored after concat. Silence the interim cast warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Behavior when concatenating bool-dtype",
                category=FutureWarning,
            )
            shard_frame = combine._concat_sorted_csvs(shard_paths, sort_cols=["log_name"])
    else:
        shard_frame = None
    summary_path = model_dir / "summary.csv"
    summary_frame = pd.read_csv(summary_path) if summary_path.is_file() else None

    def _grouped(frame: pd.DataFrame | None) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {log: [] for log in study_logs}
        if frame is None or frame.empty or "log_name" not in frame.columns:
            return grouped
        for record in frame.to_dict(orient="records"):
            name = _parse_text(record.get("log_name"))
            if name in allowed:
                grouped[name].append(record)
        return grouped

    from_shards = _grouped(shard_frame)
    from_summary = _grouped(summary_frame)
    by_log: dict[str, list[dict]] = {}
    disagree: set[str] = set()
    for log in study_logs:
        shards = from_shards[log]
        combined = from_summary[log]
        if shards and combined:
            if (
                len(shards) == 1
                and len(combined) == 1
                and _records_agree(shards[0], combined[0])
            ):
                by_log[log] = shards
            else:
                disagree.add(log)
                by_log[log] = [*shards, *combined]
        elif shards:
            by_log[log] = shards
        else:
            by_log[log] = combined
    found = shard_frame is not None or summary_frame is not None
    return LoadedConcept(by_log, frozenset(disagree), True, found)


def assess_endpoint(
    rows: Sequence[Mapping[str, object]],
    *,
    sources_disagree: bool,
    min_fitted_types: int,
    gof_threshold: float,
    xmin_setting: int | None,
) -> EndpointAssessment:
    """Eligibility and the binary GOF outcome for one saved endpoint."""
    reasons: list[str] = []
    if sources_disagree or len(rows) > 1:
        # Two disagreeing singleton sources are not a within-file duplicate.
        # More than two rows means at least one source repeated the log.
        if sources_disagree:
            reasons.append("incompatible_run_configuration")
        if len(rows) > 1 and (not sources_disagree or len(rows) > 2):
            reasons.append("duplicate_results")
        return _blank_assessment(reasons)

    if len(rows) == 0:
        return _blank_assessment(["missing_result"])

    row = rows[0]
    fit_valid = _parse_bool(row.get("fit_valid"))
    n_fitted = _parse_int(row.get("n_fitted_types"))
    gof_p, gof_problem = _parse_prob(row.get("gof_p"))
    fitted_xmin = _parse_xmin(row.get("xmin"))
    gof_kind = _parse_text(row.get("gof_kind"))
    input_path = _parse_text(row.get("input_path"))

    if xmin_setting is not None:
        locked = float(xmin_setting)
        if fitted_xmin is not None and abs(fitted_xmin - locked) > 1e-6:
            reasons.append("incompatible_xmin_configuration")
        elif fit_valid is True and fitted_xmin is None:
            reasons.append("incompatible_xmin_configuration")

    if fit_valid is not True or (gof_kind in _SKIPPED_GOF_KINDS):
        reasons.append("invalid_fit")
    if n_fitted is None:
        reasons.append("n_fitted_types_missing")
    elif n_fitted < int(min_fitted_types):
        reasons.append("below_min_fitted_types")
    if gof_problem == "out_of_range":
        reasons.append("gof_p_out_of_range")
    elif gof_p is None:
        reasons.append("gof_p_unavailable")

    eligible = not reasons
    outcome = None
    if eligible:
        assert gof_p is not None
        outcome = 1 if gof_p >= float(gof_threshold) else 0
    return EndpointAssessment(
        eligible,
        tuple(reasons),
        outcome,
        fit_valid,
        n_fitted,
        gof_p,
        fitted_xmin,
        gof_kind,
        input_path,
    )


def _blank_assessment(reasons: list[str]) -> EndpointAssessment:
    return EndpointAssessment(False, tuple(reasons), None, None, None, None, None, None, None)


def _tally(reasons: Sequence[str | None]) -> str:
    counts = Counter(reason for reason in reasons if reason)
    if not counts:
        return "none"
    ordered = [reason for reason in _REASON_PRIORITY if reason in counts]
    ordered.extend(sorted(set(counts) - set(ordered)))
    return ", ".join(f"{reason}={counts[reason]}" for reason in ordered)


def build_paired_gof_tables(
    statistics_root: Path,
    *,
    study_logs: Sequence[str] | None = None,
    xmin_settings: Sequence[int | None] = XMIN_SETTINGS,
    ks: Sequence[int] = COMPARISON_KS,
    min_fitted_types_grid: Sequence[int] = MIN_FITTED_TYPES_GRID,
    gof_thresholds: Sequence[float] = GOF_THRESHOLDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the summary and audit tables from saved results."""
    logs = list(study_logs) if study_logs is not None else list(ALL_REAL_LOG_DATASETS)
    root = Path(statistics_root)
    summary_rows: list[dict] = []
    audit_rows: list[dict] = []

    for xmin_setting in xmin_settings:
        directory = experiment_directory(root, xmin_setting)
        try:
            confirmed, provenance = identify_xmin_setting(directory)
        except ValueError as exc:
            confirmed, provenance = xmin_setting, PROVENANCE_DIRECTORY
            provenance_error = str(exc)
        else:
            provenance_error = ""
            if confirmed != xmin_setting:
                provenance_error = (
                    f"experiment {directory.name} is xmin setting "
                    f"{format_xmin_setting(confirmed)}, not "
                    f"{format_xmin_setting(xmin_setting)}"
                )
        policy = XMIN_POLICY[xmin_setting]
        setting_label = format_xmin_setting(xmin_setting)
        needed = [concept_for_k(k) for k in ks] + [REFERENCE_CONCEPT]
        loaded = {
            concept: load_concept_results(directory, concept, logs) for concept in needed
        }
        for min_fitted in min_fitted_types_grid:
            for gof_threshold in gof_thresholds:
                family_rows: list[dict] = []
                family_pvalues: list[float | None] = []
                for k in ks:
                    concept = concept_for_k(k)
                    k_loaded = loaded[concept]
                    v_loaded = loaded[REFERENCE_CONCEPT]
                    included_outcomes: list[tuple[int, int]] = []
                    k_primary: list[str | None] = []
                    v_primary: list[str | None] = []
                    n_eligible_k = 0
                    n_eligible_v = 0
                    for log in logs:
                        k_end = assess_endpoint(
                            k_loaded.by_log.get(log, []),
                            sources_disagree=log in k_loaded.sources_disagree,
                            min_fitted_types=min_fitted,
                            gof_threshold=gof_threshold,
                            xmin_setting=xmin_setting,
                        )
                        v_end = assess_endpoint(
                            v_loaded.by_log.get(log, []),
                            sources_disagree=log in v_loaded.sources_disagree,
                            min_fitted_types=min_fitted,
                            gof_threshold=gof_threshold,
                            xmin_setting=xmin_setting,
                        )
                        if provenance_error:
                            k_end = _force_incompatible(k_end)
                            v_end = _force_incompatible(v_end)
                        included = bool(k_end.eligible and v_end.eligible)
                        if included:
                            assert k_end.outcome is not None and v_end.outcome is not None
                            included_outcomes.append((k_end.outcome, v_end.outcome))
                        if k_end.eligible:
                            n_eligible_k += 1
                        else:
                            k_primary.append(k_end.primary_reason)
                        if v_end.eligible:
                            n_eligible_v += 1
                        else:
                            v_primary.append(v_end.primary_reason)
                        audit_rows.append(
                            _audit_row(
                                log=log,
                                xmin_setting=setting_label,
                                policy=policy,
                                min_fitted=min_fitted,
                                gof_threshold=gof_threshold,
                                k=k,
                                concept=concept,
                                source=directory.name,
                                provenance=provenance,
                                k_end=k_end,
                                v_end=v_end,
                                included=included,
                            )
                        )
                    n00, n01, n10, n11 = _contingency(included_outcomes)
                    effect = paired_effect(n00, n01, n10, n11)
                    status, notes = _comparison_status(
                        effect=effect,
                        n01=n01,
                        n10=n10,
                        directory=directory,
                        k_loaded=k_loaded,
                        v_loaded=v_loaded,
                        concept=concept,
                        provenance_error=provenance_error,
                        k_primary=k_primary,
                        v_primary=v_primary,
                        n_intended=len(logs),
                    )
                    row = {
                        "xmin_setting": setting_label,
                        "xmin_policy": policy,
                        "min_fitted_types": int(min_fitted),
                        "gof_threshold": float(gof_threshold),
                        "gof_boundary": GOF_BOUNDARY,
                        "k": int(k),
                        "concept": concept,
                        "reference_concept": REFERENCE_CONCEPT,
                        "model": MODEL,
                        "n_intended_logs": len(logs),
                        "n_eligible_k": n_eligible_k,
                        "n_eligible_variants": n_eligible_v,
                        "n_jointly_eligible": int(effect["N"]),
                        "N": int(effect["N"]),
                        "n00": n00,
                        "n01": n01,
                        "n10": n10,
                        "n11": n11,
                        "proportion_k": effect["proportion_k"],
                        "proportion_variants": effect["proportion_variants"],
                        "paired_difference": effect["paired_difference"],
                        "ci95_lower": effect["ci95_lower"],
                        "ci95_upper": effect["ci95_upper"],
                        "ci_method": CI_METHOD if effect["N"] else pd.NA,
                        "ci_multiplicity": CI_MULTIPLICITY if effect["N"] else pd.NA,
                        "mcnemar_p": effect["mcnemar_p"],
                        "mcnemar_p_holm": None,
                        "status": status,
                        "notes": notes,
                        "source_experiment": directory.name,
                        "provenance": provenance,
                    }
                    family_rows.append(row)
                    family_pvalues.append(
                        None if effect["mcnemar_p"] is None else float(effect["mcnemar_p"])
                    )
                for row, adjusted in zip(family_rows, holm_adjust(family_pvalues)):
                    row["mcnemar_p_holm"] = pd.NA if adjusted is None else adjusted
                    summary_rows.append(row)

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    return summary, audit


def _force_incompatible(endpoint: EndpointAssessment) -> EndpointAssessment:
    reasons = tuple(dict.fromkeys((*endpoint.reasons, "incompatible_run_configuration")))
    return EndpointAssessment(
        False,
        reasons,
        None,
        endpoint.fit_valid,
        endpoint.n_fitted_types,
        endpoint.gof_p,
        endpoint.fitted_xmin,
        endpoint.gof_kind,
        endpoint.input_path,
    )


def _contingency(outcomes: Sequence[tuple[int, int]]) -> tuple[int, int, int, int]:
    n00 = n01 = n10 = n11 = 0
    for outcome_k, outcome_variants in outcomes:
        if outcome_k == 0 and outcome_variants == 0:
            n00 += 1
        elif outcome_k == 0 and outcome_variants == 1:
            n01 += 1
        elif outcome_k == 1 and outcome_variants == 0:
            n10 += 1
        elif outcome_k == 1 and outcome_variants == 1:
            n11 += 1
        else:
            raise ValueError(f"binary outcomes required, got {(outcome_k, outcome_variants)}")
    return n00, n01, n10, n11


def _comparison_status(
    *,
    effect: Mapping[str, float | None],
    n01: int,
    n10: int,
    directory: Path,
    k_loaded: LoadedConcept,
    v_loaded: LoadedConcept,
    concept: str,
    provenance_error: str,
    k_primary: Sequence[str | None],
    v_primary: Sequence[str | None],
    n_intended: int,
) -> tuple[str, str]:
    notes: list[str] = []
    if provenance_error:
        notes.append(provenance_error if provenance_error.endswith(".") else provenance_error + ".")
    elif not directory.is_dir():
        notes.append(f"Missing experiment directory {directory.name}.")
    else:
        if not k_loaded.results_found:
            notes.append(f"Missing {MODEL} results for {concept} in {directory.name}.")
        if not v_loaded.results_found:
            notes.append(
                f"Missing {MODEL} results for {REFERENCE_CONCEPT} in {directory.name}."
            )
    total = int(effect["N"] or 0)
    notes.append(
        f"Jointly eligible {total}/{n_intended}. "
        f"k not eligible: {_tally(k_primary)}. "
        f"Variants not eligible: {_tally(v_primary)}."
    )
    if total == 0:
        notes.append("No jointly eligible logs; estimates and tests unavailable.")
        return "unavailable", " ".join(notes)
    if n01 + n10 == 0:
        notes.append("No discordant pairs; two-sided exact McNemar p=1.")
        return "no_discordance", " ".join(notes)
    return "ok", " ".join(notes)


def _audit_row(
    *,
    log: str,
    xmin_setting: str,
    policy: str,
    min_fitted: int,
    gof_threshold: float,
    k: int,
    concept: str,
    source: str,
    provenance: str,
    k_end: EndpointAssessment,
    v_end: EndpointAssessment,
    included: bool,
) -> dict:
    return {
        "log_name": log,
        "xmin_setting": xmin_setting,
        "xmin_policy": policy,
        "min_fitted_types": int(min_fitted),
        "gof_threshold": float(gof_threshold),
        "gof_boundary": GOF_BOUNDARY,
        "k": int(k),
        "concept": concept,
        "reference_concept": REFERENCE_CONCEPT,
        "model": MODEL,
        "source_experiment": source,
        "provenance": provenance,
        "input_path_k": k_end.input_path,
        "input_path_variants": v_end.input_path,
        "fitted_xmin_k": k_end.fitted_xmin,
        "fitted_xmin_variants": v_end.fitted_xmin,
        "n_fitted_types_k": k_end.n_fitted_types,
        "n_fitted_types_variants": v_end.n_fitted_types,
        "gof_p_k": k_end.gof_p,
        "gof_p_variants": v_end.gof_p,
        "fit_valid_k": k_end.fit_valid,
        "fit_valid_variants": v_end.fit_valid,
        "gof_kind_k": k_end.gof_kind,
        "gof_kind_variants": v_end.gof_kind,
        "eligible_k": k_end.eligible,
        "eligible_variants": v_end.eligible,
        "included_in_paired": included,
        "exclusion_reason_k": k_end.exclusion_reason,
        "exclusion_reason_variants": v_end.exclusion_reason,
        "outcome_k": k_end.outcome,
        "outcome_variants": v_end.outcome,
    }


def validate_paired_tables(summary: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Check counting identities, the sensitivity grid, and experiment identity."""
    if summary.empty:
        raise AssertionError("paired summary is empty")
    group_cols = ["xmin_setting", "min_fitted_types", "gof_threshold"]
    for keys, family in summary.groupby(group_cols, sort=False):
        pvalues = [
            None if _is_missing(value) else float(value) for value in family["mcnemar_p"]
        ]
        adjusted = holm_adjust(pvalues)
        for (_, row), corrected in zip(family.iterrows(), adjusted):
            _check_summary_row(row)
            observed = row["mcnemar_p_holm"]
            if corrected is None:
                if not _is_missing(observed):
                    raise AssertionError(f"Holm p-value should be missing for {keys}")
            elif _is_missing(observed) or abs(float(observed) - corrected) > 1e-12:
                raise AssertionError(f"Holm p-value mismatch for {keys} k={row['k']}")
        if len(family) != len(set(family["k"])):
            raise AssertionError(f"duplicate k in family {keys}")

    _check_cutoff_monotonicity(summary)
    _check_threshold_does_not_change_eligibility(summary, audit)
    _check_audit_matches_summary(summary, audit)
    _check_experiments_are_not_mixed(summary, audit)


def _check_summary_row(row: pd.Series) -> None:
    n00, n01, n10, n11 = (int(row[name]) for name in ("n00", "n01", "n10", "n11"))
    total = n00 + n01 + n10 + n11
    if total != int(row["N"]) or total != int(row["n_jointly_eligible"]):
        raise AssertionError(f"counts do not sum to N for k={row['k']}")
    effect = paired_effect(n00, n01, n10, n11)
    if total == 0:
        for column in (
            "proportion_k",
            "proportion_variants",
            "paired_difference",
            "ci95_lower",
            "ci95_upper",
            "mcnemar_p",
            "mcnemar_p_holm",
        ):
            if not _is_missing(row[column]):
                raise AssertionError(f"{column} should be missing when N=0")
        if row["status"] != "unavailable":
            raise AssertionError("N=0 must be status unavailable")
        return
    for column in ("proportion_k", "proportion_variants", "paired_difference", "mcnemar_p"):
        if abs(float(row[column]) - float(effect[column])) > 1e-12:
            raise AssertionError(f"{column} mismatch")
    difference_from_margins = float(row["proportion_variants"]) - float(row["proportion_k"])
    difference_from_discordance = (n01 - n10) / total
    if abs(float(row["paired_difference"]) - difference_from_margins) > 1e-12:
        raise AssertionError("difference does not match the two proportions")
    if abs(float(row["paired_difference"]) - difference_from_discordance) > 1e-12:
        raise AssertionError("the two difference formulas disagree")
    lower, upper = newcombe_method10_variants_minus_k(n00, n01, n10, n11)
    if abs(float(row["ci95_lower"]) - lower) > 1e-12 or abs(float(row["ci95_upper"]) - upper) > 1e-12:
        raise AssertionError("stored interval does not match Newcombe method 10")
    if row["ci_method"] != CI_METHOD or row["ci_multiplicity"] != CI_MULTIPLICITY:
        raise AssertionError("CI method must be recorded as pointwise Newcombe method 10")
    swapped = paired_effect(n00, n10, n01, n11)
    if abs(float(swapped["paired_difference"]) + float(row["paired_difference"])) > 1e-12:
        raise AssertionError("swapping endpoints did not negate the difference")
    if (
        abs(float(swapped["ci95_lower"]) + float(row["ci95_upper"])) > 1e-12
        or abs(float(swapped["ci95_upper"]) + float(row["ci95_lower"])) > 1e-12
    ):
        raise AssertionError("swapping endpoints did not reverse the interval")
    if abs(float(swapped["mcnemar_p"]) - float(row["mcnemar_p"])) > 1e-12:
        raise AssertionError("swapping endpoints changed the two-sided McNemar p-value")
    if n01 + n10 == 0:
        if float(row["mcnemar_p"]) != 1.0 or row["status"] != "no_discordance":
            raise AssertionError("no-discordance case must report p=1")
        if abs(float(row["paired_difference"])) > 1e-12:
            raise AssertionError("no discordant pairs must have difference 0")


def _check_cutoff_monotonicity(summary: pd.DataFrame) -> None:
    columns = ["N", "n_eligible_k", "n_eligible_variants", "n_jointly_eligible"]
    groups = summary.groupby(
        ["xmin_setting", "gof_threshold", "k", "source_experiment"], sort=False
    )
    for keys, frame in groups:
        ordered = frame.sort_values("min_fitted_types")
        for column in columns:
            values = [int(value) for value in ordered[column]]
            if values != sorted(values, reverse=True):
                raise AssertionError(
                    f"raising the support cutoff increased {column} for {keys}: {values}"
                )


def _check_threshold_does_not_change_eligibility(
    summary: pd.DataFrame, audit: pd.DataFrame
) -> None:
    groups = summary.groupby(
        ["xmin_setting", "min_fitted_types", "k", "source_experiment"], sort=False
    )
    for keys, frame in groups:
        for column in ("N", "n_eligible_k", "n_eligible_variants", "n_jointly_eligible"):
            if frame[column].nunique() != 1:
                raise AssertionError(
                    f"GOF threshold changed {column} for {keys}"
                )
    audit_groups = audit.groupby(
        ["xmin_setting", "min_fitted_types", "k", "log_name", "source_experiment"],
        sort=False,
    )
    for keys, frame in audit_groups:
        if frame["eligible_k"].nunique() != 1 or frame["eligible_variants"].nunique() != 1:
            raise AssertionError(f"GOF threshold changed eligibility for {keys}")
        if frame["included_in_paired"].nunique() != 1:
            raise AssertionError(f"GOF threshold changed paired inclusion for {keys}")


def _check_audit_matches_summary(summary: pd.DataFrame, audit: pd.DataFrame) -> None:
    keys = [
        "xmin_setting",
        "min_fitted_types",
        "gof_threshold",
        "k",
        "source_experiment",
    ]
    for _, row in summary.iterrows():
        mask = pd.Series(True, index=audit.index)
        for key in keys:
            mask &= audit[key] == row[key]
        block = audit.loc[mask]
        if len(block) != int(row["n_intended_logs"]):
            raise AssertionError("audit does not have one row per intended log")
        included = block[block["included_in_paired"] == True]  # noqa: E712
        if len(included) != int(row["N"]):
            raise AssertionError("included audit rows do not equal N")
        if not included.empty:
            outcome_k = included["outcome_k"].astype(int)
            outcome_v = included["outcome_variants"].astype(int)
            n11 = int(((outcome_k == 1) & (outcome_v == 1)).sum())
            n10 = int(((outcome_k == 1) & (outcome_v == 0)).sum())
            n01 = int(((outcome_k == 0) & (outcome_v == 1)).sum())
            n00 = int(((outcome_k == 0) & (outcome_v == 0)).sum())
            if (n00, n01, n10, n11) != (
                int(row["n00"]),
                int(row["n01"]),
                int(row["n10"]),
                int(row["n11"]),
            ):
                raise AssertionError("audit outcomes do not match paired counts")
            if int((outcome_k == 1).sum()) != int(row["n10"]) + int(row["n11"]):
                raise AssertionError("proportion_k uses a different log set")
            if int((outcome_v == 1).sum()) != int(row["n01"]) + int(row["n11"]):
                raise AssertionError("proportion_variants uses a different log set")
        eligible_k = int((block["eligible_k"] == True).sum())  # noqa: E712
        eligible_v = int((block["eligible_variants"] == True).sum())  # noqa: E712
        if eligible_k != int(row["n_eligible_k"]) or eligible_v != int(row["n_eligible_variants"]):
            raise AssertionError("eligible counts do not match the audit")
        both = (block["eligible_k"] == True) & (block["eligible_variants"] == True)  # noqa: E712
        included_flag = block["included_in_paired"] == True  # noqa: E712
        if not both.equals(included_flag):
            raise AssertionError("inclusion is not the intersection of the two endpoints")


def _check_experiments_are_not_mixed(summary: pd.DataFrame, audit: pd.DataFrame) -> None:
    for _, row in summary.iterrows():
        expected = EXPERIMENT_DIRECTORY[
            None if row["xmin_setting"] == "None" else int(row["xmin_setting"])
        ]
        if row["source_experiment"] != expected:
            raise AssertionError(
                f"xmin setting {row['xmin_setting']} used experiment {row['source_experiment']}"
            )
        if row["xmin_policy"] != XMIN_POLICY[
            None if row["xmin_setting"] == "None" else int(row["xmin_setting"])
        ]:
            raise AssertionError("xmin policy does not match the experiment")
    included = audit[audit["included_in_paired"] == True]  # noqa: E712
    for _, row in included.iterrows():
        expected = EXPERIMENT_DIRECTORY[
            None if row["xmin_setting"] == "None" else int(row["xmin_setting"])
        ]
        if row["source_experiment"] != expected:
            raise AssertionError("paired log uses a different experiment from its xmin setting")
        if row["xmin_setting"] == "None":
            continue
        locked = float(row["xmin_setting"])
        for column in ("fitted_xmin_k", "fitted_xmin_variants"):
            if _is_missing(row[column]) or abs(float(row[column]) - locked) > 1e-6:
                raise AssertionError(
                    "fixed-xmin comparison includes a fitted xmin from another policy"
                )


def format_paired_results(summary: pd.DataFrame) -> str:
    """Readable view organised by xmin setting, support cutoff, and GOF threshold."""
    lines = [METHODS_NOTE.rstrip(), ""]
    for xmin_setting in _unique(summary["xmin_setting"]):
        policy = summary.loc[summary["xmin_setting"] == xmin_setting, "xmin_policy"].iloc[0]
        if xmin_setting == "None":
            heading = "xmin setting: None (adaptive; xmin estimated for each fit)"
        else:
            heading = f"xmin setting: {xmin_setting} (fixed xmin={xmin_setting})"
        lines.append(f"== {heading}; policy={policy} ==")
        subset = summary[summary["xmin_setting"] == xmin_setting]
        for min_fitted in _unique(subset["min_fitted_types"]):
            lines.append(f"-- support cutoff: min_fitted_types = {int(min_fitted)} --")
            cutoff = subset[subset["min_fitted_types"] == min_fitted]
            for threshold in _unique(cutoff["gof_threshold"]):
                lines.append(
                    f"---- GOF threshold {float(threshold):.2f} "
                    f"(non-rejection if {GOF_BOUNDARY.replace('gof_threshold', f'{float(threshold):.2f}')}) ----"
                )
                block = cutoff[cutoff["gof_threshold"] == threshold].sort_values("k")
                lines.extend(_format_block(block))
                lines.append("")
        lines.append("")
    lines.append(
        "* Holm-adjusted p < 0.05. Confidence intervals are pointwise and are "
        "not multiplicity-adjusted."
    )
    return "\n".join(lines).rstrip() + "\n"


def _unique(series: pd.Series) -> list:
    seen = []
    for value in series:
        if value not in seen:
            seen.append(value)
    return seen


def _format_block(block: pd.DataFrame) -> list[str]:
    headers = [
        "Comparison",
        "Paired N",
        "Not rejected at k",
        "Not rejected at variants",
        "Rejected→not rejected",
        "Not rejected→rejected",
        "Difference, pp [95% CI]",
        "Raw p",
        "Holm-adjusted p",
    ]
    body = []
    notes = []
    for _, row in block.iterrows():
        total = int(row["N"])
        if total == 0:
            diff = "NA"
            raw_p = "NA"
            holm_p = "NA"
            not_k = "NA"
            not_v = "NA"
            to_variant = "NA"
            to_k = "NA"
        else:
            diff = (
                f"{100 * float(row['paired_difference']):+.2f} "
                f"[{100 * float(row['ci95_lower']):+.2f}, "
                f"{100 * float(row['ci95_upper']):+.2f}]"
            )
            raw_p = _format_p(row["mcnemar_p"])
            significant = (
                not _is_missing(row["mcnemar_p_holm"]) and float(row["mcnemar_p_holm"]) < 0.05
            )
            holm_p = _format_p(row["mcnemar_p_holm"], star=significant)
            not_k = str(int(row["n10"]) + int(row["n11"]))
            not_v = str(int(row["n01"]) + int(row["n11"]))
            to_variant = str(int(row["n01"]))
            to_k = str(int(row["n10"]))
        body.append(
            [
                f"k={int(row['k'])} vs variants",
                str(total),
                not_k,
                not_v,
                to_variant,
                to_k,
                diff,
                raw_p,
                holm_p,
            ]
        )
        notes.append(f"k={int(row['k'])}: {row['status']}. {row['notes']}")
    widths = [
        max(len(headers[index]), *(len(cells[index]) for cells in body))
        for index in range(len(headers))
    ]
    def _render(cells: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(cells)).rstrip()

    rendered = [_render(headers), _render(["-" * width for width in widths])]
    rendered.extend(_render(cells) for cells in body)
    rendered.append("Notes:")
    rendered.extend(f"- {note}" for note in notes)
    return rendered


def _format_p(value: object, *, star: bool = False) -> str:
    if _is_missing(value):
        return "NA"
    text = f"{float(value):.4g}"
    if star:
        text += " *"
    return text


def paper_table_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Wide paper table. Holm families are recomputed from the raw p-values.

    ``mcnemar_p_holm_k_le_10`` and ``mcnemar_p_holm_k_le_3`` are not slices of
    the 20-comparison adjustment. Unavailable tests stay missing in the table
    and contribute p = 1 only inside ``holm_adjust``.
    """
    ordered = summary.sort_values("k")
    ks = [int(k) for k in ordered["k"]]
    if ks != list(PAPER_KS):
        raise ValueError(f"paper table requires k=1..20 in order, got {ks}")
    if ordered["k"].duplicated().any():
        raise ValueError("paper table has duplicate k")
    raw = [
        None if _is_missing(value) else float(value) for value in ordered["mcnemar_p"]
    ]
    holm_20 = holm_adjust(raw)
    holm_10 = holm_adjust(raw[:10])
    holm_3 = holm_adjust(raw[:3])
    columns = ["statistic", *[f"k={k}" for k in PAPER_KS]]
    values = {
        "paired_N": [int(value) for value in ordered["N"]],
        "difference_pp": [_to_percentage_points(value) for value in ordered["paired_difference"]],
        "ci95_lower_pp": [_to_percentage_points(value) for value in ordered["ci95_lower"]],
        "ci95_upper_pp": [_to_percentage_points(value) for value in ordered["ci95_upper"]],
        "mcnemar_p_raw": [_nan_if_missing(value) for value in raw],
        "mcnemar_p_holm_k_le_20": [_nan_if_missing(value) for value in holm_20],
        "mcnemar_p_holm_k_le_10": [
            *[_nan_if_missing(value) for value in holm_10],
            *[math.nan] * (len(PAPER_KS) - 10),
        ],
        "mcnemar_p_holm_k_le_3": [
            *[_nan_if_missing(value) for value in holm_3],
            *[math.nan] * (len(PAPER_KS) - 3),
        ],
    }
    frame = pd.DataFrame(
        [[name, *values[name]] for name in PAPER_STATISTIC_ROWS],
        columns=columns,
    )
    return frame


def validate_paper_table(table: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Check column order, separate Holm families, and a single pp conversion."""
    expected = ["statistic", *[f"k={k}" for k in PAPER_KS]]
    if list(table.columns) != expected:
        raise AssertionError(f"paper columns must be k=1..20, got {list(table.columns)}")
    if list(table["statistic"]) != list(PAPER_STATISTIC_ROWS):
        raise AssertionError(f"paper rows differ from {PAPER_STATISTIC_ROWS}")
    ordered = summary.sort_values("k")
    raw = [
        None if _is_missing(value) else float(value) for value in ordered["mcnemar_p"]
    ]
    holm_20 = holm_adjust(list(raw))
    holm_10 = holm_adjust(list(raw[:10]))
    holm_3 = holm_adjust(list(raw[:3]))
    by_stat = table.set_index("statistic")
    for index, (_, row) in enumerate(ordered.iterrows()):
        column = f"k={int(row['k'])}"
        paired_n = by_stat.loc["paired_N", column]
        if int(paired_n) != int(row["N"]):
            raise AssertionError(f"{column} paired_N does not match N")
        _check_pp(by_stat.loc["difference_pp", column], row["paired_difference"], column, "difference")
        _check_pp(by_stat.loc["ci95_lower_pp", column], row["ci95_lower"], column, "ci95_lower")
        _check_pp(by_stat.loc["ci95_upper_pp", column], row["ci95_upper"], column, "ci95_upper")
        _check_same_p(by_stat.loc["mcnemar_p_raw", column], raw[index], column, "raw")
        _check_same_p(by_stat.loc["mcnemar_p_holm_k_le_20", column], holm_20[index], column, "holm20")
        if index < 10:
            _check_same_p(
                by_stat.loc["mcnemar_p_holm_k_le_10", column], holm_10[index], column, "holm10"
            )
        elif not _is_missing(by_stat.loc["mcnemar_p_holm_k_le_10", column]):
            raise AssertionError(f"{column} holm k<=10 must be NaN")
        if index < 3:
            _check_same_p(
                by_stat.loc["mcnemar_p_holm_k_le_3", column], holm_3[index], column, "holm3"
            )
        elif not _is_missing(by_stat.loc["mcnemar_p_holm_k_le_3", column]):
            raise AssertionError(f"{column} holm k<=3 must be NaN")
        if int(row["N"]) == 0:
            for statistic in (
                "difference_pp",
                "ci95_lower_pp",
                "ci95_upper_pp",
                "mcnemar_p_raw",
                "mcnemar_p_holm_k_le_20",
            ):
                if not _is_missing(by_stat.loc[statistic, column]):
                    raise AssertionError(f"{column} {statistic} must be NaN when N=0")


def write_paired_gof_paper_table(
    output_dir: Path, *, statistics_root: Path | None = None
) -> Path | None:
    """Write the k=1..20 paper CSV for adaptive xmin, 30 types, threshold 0.10.

    The three Holm rows are alternative comparison families. A paper has to
    justify which family supports its main inference. Choosing the family
    because that adjustment is significant is not a confirmation.
    """
    output_dir = Path(output_dir)
    if output_dir.name not in EXPERIMENT_DIRECTORY_SETTING:
        print(
            "Skipping paired GOF paper table "
            f"({output_dir.name} is not a real-log xmin experiment)"
        )
        return None
    root = Path(statistics_root) if statistics_root is not None else output_dir.parent
    summary, audit = build_paired_gof_tables(
        root,
        xmin_settings=(PAPER_XMIN_SETTING,),
        ks=PAPER_KS,
        min_fitted_types_grid=(PAPER_MIN_FITTED_TYPES,),
        gof_thresholds=(PAPER_GOF_THRESHOLD,),
    )
    validate_paired_tables(summary, audit)
    table = paper_table_from_summary(summary)
    validate_paper_table(table, summary)
    destination = output_dir / "paired_gof"
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / PAPER_TABLE_NAME
    _write_paper_csv(table, csv_path)
    print(
        "Paired GOF paper table: xmin=None (adaptive), "
        f"min_fitted_types={PAPER_MIN_FITTED_TYPES}, "
        f"gof_threshold={PAPER_GOF_THRESHOLD:.2f}, k=1..20 vs variants."
    )
    print(
        "Holm rows k_le_20, k_le_10, and k_le_3 are alternative family "
        "definitions recomputed from the raw p-values. They are not "
        "independent confirmations of a finding."
    )
    for _, row in summary.sort_values("k").iterrows():
        print(f"  k={int(row['k'])}: {row['status']}. {row['notes']}")
    print(f"Paired GOF paper table -> {csv_path}")
    return csv_path


def _to_percentage_points(value: object) -> float:
    if _is_missing(value):
        return math.nan
    return 100.0 * float(value)


def _nan_if_missing(value: object) -> float:
    if _is_missing(value):
        return math.nan
    return float(value)


def _check_pp(observed: object, probability: object, column: str, name: str) -> None:
    if _is_missing(probability):
        if not _is_missing(observed):
            raise AssertionError(f"{column} {name} should be NaN")
        return
    expected = 100.0 * float(probability)
    if _is_missing(observed) or abs(float(observed) - expected) > 1e-9:
        raise AssertionError(f"{column} {name} was not converted to percentage points once")
    if abs(float(observed) / 100.0 - float(probability)) > 1e-12:
        raise AssertionError(f"{column} {name} does not invert to the probability-scale value")


def _check_same_p(observed: object, expected: object, column: str, name: str) -> None:
    if _is_missing(expected):
        if not _is_missing(observed):
            raise AssertionError(f"{column} {name} should be NaN")
        return
    if _is_missing(observed) or abs(float(observed) - float(expected)) > 1e-12:
        raise AssertionError(f"{column} {name} does not match the raw-test adjustment")


def _write_paper_csv(table: pd.DataFrame, path: Path) -> None:
    """Write numeric cells with literal NaN and full float precision."""
    header = ",".join(str(column) for column in table.columns)
    lines = [header]
    for _, row in table.iterrows():
        cells = [str(row["statistic"])]
        integer = row["statistic"] == "paired_N"
        for column in table.columns[1:]:
            cells.append(_format_paper_cell(row[column], integer=integer))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_paper_cell(value: object, *, integer: bool) -> str:
    if _is_missing(value):
        return "NaN"
    if integer:
        return str(int(value))
    return repr(float(value))


def write_paired_gof_comparisons(output_dir: Path) -> Path | None:
    """Write paired GOF outputs when ``output_dir`` is a real xmin experiment.

    The tables always cover the full xmin grid, using sibling experiment
    directories under the same power-law statistics root. Smoke-test and
    synthetic logs are not included.
    """
    output_dir = Path(output_dir)
    if output_dir.name not in EXPERIMENT_DIRECTORY_SETTING:
        print(
            "Skipping paired GOF comparisons "
            f"({output_dir.name} is not a real-log xmin experiment)"
        )
        return None
    statistics_root = output_dir.parent
    summary, audit = build_paired_gof_tables(statistics_root)
    validate_paired_tables(summary, audit)
    destination = output_dir / "paired_gof"
    destination.mkdir(parents=True, exist_ok=True)
    summary_path = destination / "paired_gof_summary.csv"
    audit_path = destination / "paired_gof_audit.csv"
    text_path = destination / "paired_gof_results.txt"
    summary.to_csv(summary_path, index=False)
    audit.to_csv(audit_path, index=False)
    text_path.write_text(format_paired_results(summary), encoding="utf-8")
    print(
        "Paired GOF comparisons: "
        f"{len(summary)} configuration-comparison rows, "
        f"{len(audit)} audit rows -> {destination}"
    )
    write_paired_gof_paper_table(output_dir, statistics_root=statistics_root)
    return destination
