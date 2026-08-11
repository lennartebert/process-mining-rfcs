"""Association between case attributes and complete trace variants.

Uses Shannon entropy reduction / mutual information at case level.
This is intentionally not labelled as a generic correlation coefficient.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .case_attribute_inventory import MISSING_LABEL, _is_missing
from utils.io.case_tables import is_event_aggregation_column


def shannon_entropy(labels: Sequence[Any], *, base: float = 2.0) -> float:
    """Return Shannon entropy of a discrete label sequence."""
    if len(labels) == 0:
        return 0.0
    series = pd.Series(list(labels), dtype=object).map(_label_key)
    counts = series.value_counts().to_numpy(dtype=float)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs) / np.log(base)).sum())


def _label_key(value: Any) -> str:
    if _is_missing(value):
        return MISSING_LABEL
    return str(value)


def conditional_entropy(
    target: Sequence[Any],
    given: Sequence[Any],
    *,
    base: float = 2.0,
) -> float:
    """Return H(target | given)."""
    if len(target) == 0:
        return 0.0
    if len(target) != len(given):
        raise ValueError("target and given must have the same length")

    df = pd.DataFrame(
        {
            "target": [_label_key(v) for v in target],
            "given": [_label_key(v) for v in given],
        }
    )
    n = len(df)
    h = 0.0
    for _, group in df.groupby("given", sort=False):
        weight = len(group) / n
        h += weight * shannon_entropy(group["target"].tolist(), base=base)
    return float(h)


def entropy_reduction_metrics(
    variant_labels: Sequence[Any],
    attribute_values: Sequence[Any],
    *,
    missing_value_handling: str = "separate_category",
    base: float = 2.0,
) -> dict[str, Any]:
    """Compute entropy reduction of variants given an attribute.

    Parameters
    ----------
    missing_value_handling :
        ``separate_category`` keeps missing as ``__MISSING__``.
        ``drop_cases`` drops pairs where the attribute is missing.
    """
    variants = list(variant_labels)
    attributes = list(attribute_values)
    if len(variants) != len(attributes):
        raise ValueError("variant_labels and attribute_values length mismatch")

    if missing_value_handling == "drop_cases":
        pairs = [
            (v, a)
            for v, a in zip(variants, attributes)
            if not _is_missing(a)
        ]
        if not pairs:
            return _empty_association(status="skipped", reason="no_cases_after_drop")
        variants = [v for v, _ in pairs]
        attributes = [a for _, a in pairs]
    elif missing_value_handling == "exclude_attribute":
        return _empty_association(status="skipped", reason="exclude_attribute")
    elif missing_value_handling != "separate_category":
        raise ValueError(f"Unknown missing_value_handling: {missing_value_handling}")

    n = len(variants)
    n_distinct_attr = len({_label_key(a) for a in attributes})
    n_distinct_var = len({_label_key(v) for v in variants})
    if n < 2 or n_distinct_attr < 2 or n_distinct_var < 2:
        return {
            "status": "not_applicable",
            "reason": "insufficient_variation",
            "n_cases": n,
            "n_distinct_attribute": n_distinct_attr,
            "n_distinct_variant": n_distinct_var,
            "entropy_variant": shannon_entropy(variants, base=base) if n else 0.0,
            "entropy_attribute": shannon_entropy(attributes, base=base) if n else 0.0,
            "entropy_conditional": None,
            "entropy_reduction": 0.0,
            "normalized_entropy_reduction": None,
            "nmi": None,
            "permutation_p": None,
        }

    h_v = shannon_entropy(variants, base=base)
    h_a = shannon_entropy(attributes, base=base)
    h_v_given_a = conditional_entropy(variants, attributes, base=base)
    er = h_v - h_v_given_a
    # Numerical safety for tiny negative values from floating point.
    er = max(0.0, float(er))
    ner = (er / h_v) if h_v > 0 else None
    denom = h_v + h_a
    nmi = (2.0 * er / denom) if denom > 0 else None
    return {
        "status": "computed",
        "reason": None,
        "n_cases": n,
        "n_distinct_attribute": n_distinct_attr,
        "n_distinct_variant": n_distinct_var,
        "entropy_variant": h_v,
        "entropy_attribute": h_a,
        "entropy_conditional": h_v_given_a,
        "entropy_reduction": er,
        "normalized_entropy_reduction": ner,
        "nmi": nmi,
        "permutation_p": None,
    }


def permutation_p_value(
    variant_labels: Sequence[Any],
    attribute_values: Sequence[Any],
    *,
    n_permutations: int = 200,
    random_seed: int = 42,
    missing_value_handling: str = "separate_category",
    observed_entropy_reduction: float | None = None,
) -> float | None:
    """Return a one-sided permutation p-value for entropy reduction."""
    if n_permutations <= 0:
        return None

    if observed_entropy_reduction is None:
        observed = entropy_reduction_metrics(
            variant_labels,
            attribute_values,
            missing_value_handling=missing_value_handling,
        )
        if observed["status"] != "computed":
            return None
        observed_er = float(observed["entropy_reduction"])
        # Align with the same filtered vectors used for the observed metric.
        variants = list(variant_labels)
        attributes = list(attribute_values)
        if missing_value_handling == "drop_cases":
            pairs = [
                (v, a)
                for v, a in zip(variants, attributes)
                if not _is_missing(a)
            ]
            variants = [v for v, _ in pairs]
            attributes = [a for _, a in pairs]
    else:
        observed_er = float(observed_entropy_reduction)
        variants = list(variant_labels)
        attributes = list(attribute_values)
        if missing_value_handling == "drop_cases":
            pairs = [
                (v, a)
                for v, a in zip(variants, attributes)
                if not _is_missing(a)
            ]
            variants = [v for v, _ in pairs]
            attributes = [a for _, a in pairs]

    rng = np.random.default_rng(random_seed)
    attr_arr = np.asarray(attributes, dtype=object)
    n_ge = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(attr_arr)
        perm = entropy_reduction_metrics(
            variants,
            shuffled.tolist(),
            missing_value_handling="separate_category",
        )
        if float(perm["entropy_reduction"]) >= observed_er:
            n_ge += 1
    return float((1 + n_ge) / (n_permutations + 1))


def associate_attribute_with_variant(
    variant_labels: Sequence[Any],
    attribute_values: Sequence[Any],
    *,
    attribute_name: str,
    inferred_type: str,
    source_level: str,
    missing_value_handling: str = "separate_category",
    n_permutations: int = 0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Compute variant association when meaningful; otherwise mark skipped."""
    base = {
        "attribute_name": attribute_name,
        "inferred_type": inferred_type,
        "source_level": source_level,
    }

    if inferred_type in {"unsupported", "datetime", "identifier-like"}:
        return {
            **base,
            **_empty_association(
                status="not_applicable",
                reason=f"type_{inferred_type}",
            ),
        }
    if source_level == "event" and not is_event_aggregation_column(attribute_name):
        return {
            **base,
            **_empty_association(
                status="not_applicable",
                reason="event_level_without_aggregation",
            ),
        }

    metrics = entropy_reduction_metrics(
        variant_labels,
        attribute_values,
        missing_value_handling=missing_value_handling,
    )
    if metrics["status"] == "computed" and n_permutations > 0:
        metrics["permutation_p"] = permutation_p_value(
            variant_labels,
            attribute_values,
            n_permutations=n_permutations,
            random_seed=random_seed,
            missing_value_handling=missing_value_handling,
            observed_entropy_reduction=float(metrics["entropy_reduction"]),
        )
    return {**base, **metrics}


def _empty_association(*, status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "n_cases": None,
        "n_distinct_attribute": None,
        "n_distinct_variant": None,
        "entropy_variant": None,
        "entropy_attribute": None,
        "entropy_conditional": None,
        "entropy_reduction": None,
        "normalized_entropy_reduction": None,
        "nmi": None,
        "permutation_p": None,
    }
