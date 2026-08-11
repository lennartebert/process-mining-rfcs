"""CSV configuration for case-attribute HITL (schema v3).

Step 1 writes ``attribute_inventory.csv`` with descriptive columns plus
suggested/override pairs. Step 2 writes ``attribute_selection_for_powerlaw.csv``
with association metrics plus the same override pattern for power-law selection.

Effective values are resolved in memory as override-if-present else suggested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .case_attribute_inventory import (
    BINNING_METHODS,
    DATATYPES,
    AttributeInventory,
    DatatypeRecommendation,
    PowerlawIncludeRecommendation,
)

SCHEMA_VERSION = 3
INVENTORY_FILENAME = "attribute_inventory.csv"
POWERLAW_SELECTION_FILENAME = "attribute_selection_for_powerlaw.csv"

# Backwards-compatible aliases (deprecated YAML names).
DATATYPES_FILENAME = INVENTORY_FILENAME

INVENTORY_OVERRIDE_COLUMNS = (
    "include_override",
    "type_override",
    "binning_override",
    "binning_n_bins_override",
)

POWERLAW_OVERRIDE_COLUMNS = (
    "include_override",
    "type_override",
    "binning_override",
    "binning_n_bins_override",
    "discrete_override",
    "reason_override",
)


class ConfigValidationError(ValueError):
    """Raised when a CSV override is invalid."""


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if pd.isna(value):
        return True
    return False


def resolve_field(suggested: Any, override: Any = None, *, default: Any = None) -> Any:
    """Return effective value: override if present, else suggested, else default."""
    if not _is_blank(override):
        return override
    if not _is_blank(suggested):
        return suggested
    return default


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ConfigValidationError(f"Invalid boolean for {field_name}: {value!r}")


def validate_datatype(value: Any) -> str:
    text = str(value).strip()
    if text not in DATATYPES:
        raise ConfigValidationError(
            f"Invalid datatype {value!r}; allowed: {DATATYPES}"
        )
    return text


def validate_binning(
    datatype: str,
    method: Any,
    n_bins: Any,
    bin_edges: Any = None,
) -> dict[str, Any]:
    """Validate binning against datatype; return cleaned effective params."""
    del bin_edges  # fixed_bins edges are not exposed in the CSV schema
    if datatype != "continuous":
        return {"method": None, "n_bins": None, "bin_edges": None}
    if _is_blank(method):
        raise ConfigValidationError(
            "continuous datatype requires binning "
            f"(one of {BINNING_METHODS})"
        )
    method_s = str(method).strip()
    if method_s not in BINNING_METHODS:
        raise ConfigValidationError(
            f"Invalid binning method {method!r}; allowed: {BINNING_METHODS}"
        )
    if method_s == "fixed_bins":
        raise ConfigValidationError(
            "fixed_bins requires bin_edges; use log_bins/linear_bins/quantile_bins "
            "in the CSV workflow"
        )
    if _is_blank(n_bins) or int(n_bins) < 2:
        raise ConfigValidationError(f"{method_s} requires binning_n_bins >= 2")
    return {"method": method_s, "n_bins": int(n_bins), "bin_edges": None}


def inventory_metadata_columns(inventory: AttributeInventory) -> dict[str, Any]:
    return {
        "attribute_name": inventory.attribute_name,
        "source_level": inventory.source_level,
        "inferred_type": inventory.inferred_type,
        "n_cases": inventory.n_cases,
        "n_missing": inventory.n_missing,
        "missing_share": inventory.missing_share,
        "n_distinct": inventory.n_distinct,
        "distinct_share": inventory.distinct_share,
        "most_frequent_value": inventory.most_frequent_value,
        "largest_value_share": inventory.largest_value_share,
        "min_value": inventory.min_value,
        "max_value": inventory.max_value,
        "median_value": inventory.median_value,
        "mean_value": inventory.mean_value,
        "example_values": "|".join(str(v) for v in inventory.example_values),
        "is_constant": inventory.is_constant,
        "is_near_constant": inventory.is_near_constant,
        "is_identifier_like": inventory.is_identifier_like,
        "suitable_for_categorical_frequency": (
            inventory.suitable_for_categorical_frequency
        ),
        "special_handling_required": inventory.special_handling_required,
    }


def build_inventory_row(
    inventory: AttributeInventory,
    recommendation: DatatypeRecommendation,
) -> dict[str, Any]:
    """Build one step-1 inventory CSV row (suggested values; overrides blank)."""
    row = inventory_metadata_columns(inventory)
    row.update(
        {
            "include_suggested": bool(recommendation.include),
            "include_override": None,
            "type_suggested": recommendation.datatype,
            "type_override": None,
            "binning_suggested": recommendation.binning_method,
            "binning_override": None,
            "binning_n_bins_suggested": recommendation.n_bins,
            "binning_n_bins_override": None,
            "reason_suggested": recommendation.reason,
        }
    )
    return row


def merge_inventory_dataframe(
    old_df: pd.DataFrame | None,
    new_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge newly suggested inventory rows, preserving user override columns."""
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        return new_df
    if old_df is None or old_df.empty or "attribute_name" not in old_df.columns:
        return new_df.sort_values("attribute_name").reset_index(drop=True)

    old_by_name = old_df.set_index("attribute_name", drop=False)
    merged_rows: list[dict[str, Any]] = []
    for row in new_df.to_dict(orient="records"):
        name = row["attribute_name"]
        if name in old_by_name.index:
            old = old_by_name.loc[name]
            if isinstance(old, pd.DataFrame):
                old = old.iloc[0]
            for col in INVENTORY_OVERRIDE_COLUMNS:
                if col in old_by_name.columns and not _is_blank(old.get(col)):
                    row[col] = old.get(col)
        merged_rows.append(row)
    return (
        pd.DataFrame(merged_rows)
        .sort_values("attribute_name")
        .reset_index(drop=True)
    )


def resolve_inventory_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve effective include/datatype/binning from one inventory CSV row."""
    include_raw = resolve_field(
        row.get("include_suggested"), row.get("include_override"), default=False
    )
    include = _parse_bool(include_raw, field_name="include")
    datatype = validate_datatype(
        resolve_field(
            row.get("type_suggested"),
            row.get("type_override"),
            default="unsupported",
        )
    )
    method = resolve_field(
        row.get("binning_suggested"), row.get("binning_override"), default=None
    )
    n_bins = resolve_field(
        row.get("binning_n_bins_suggested"),
        row.get("binning_n_bins_override"),
        default=None,
    )
    if include and datatype == "continuous":
        cleaned = validate_binning(datatype, method, n_bins)
    elif datatype == "continuous" and not _is_blank(method):
        cleaned = validate_binning(datatype, method, n_bins)
    else:
        cleaned = {
            "method": None if _is_blank(method) else str(method),
            "n_bins": None if _is_blank(n_bins) else int(n_bins),
            "bin_edges": None,
        }
    return {
        "attribute_name": str(row.get("attribute_name")),
        "include": include,
        "datatype": datatype,
        "binning": cleaned,
        "metadata": {
            key: row.get(key)
            for key in (
                "source_level",
                "inferred_type",
                "n_cases",
                "n_missing",
                "missing_share",
                "n_distinct",
                "distinct_share",
                "most_frequent_value",
                "largest_value_share",
                "min_value",
                "max_value",
                "median_value",
                "mean_value",
                "example_values",
                "is_constant",
                "is_near_constant",
                "is_identifier_like",
                "suitable_for_categorical_frequency",
                "special_handling_required",
            )
        },
        "reason_suggested": row.get("reason_suggested"),
    }


def build_powerlaw_selection_row(
    *,
    attribute_name: str,
    resolved_step1: Mapping[str, Any],
    association: Mapping[str, Any],
    recommendation: PowerlawIncludeRecommendation,
) -> dict[str, Any]:
    """Build one step-2 selection CSV row."""
    binning = resolved_step1.get("binning") or {}
    return {
        "attribute_name": attribute_name,
        "source_level": (resolved_step1.get("metadata") or {}).get("source_level"),
        "n_cases": (resolved_step1.get("metadata") or {}).get("n_cases"),
        "n_distinct": (resolved_step1.get("metadata") or {}).get("n_distinct"),
        "status": association.get("status"),
        "association_reason": association.get("reason"),
        "entropy_variant": association.get("entropy_variant"),
        "entropy_attribute": association.get("entropy_attribute"),
        "entropy_conditional": association.get("entropy_conditional"),
        "entropy_reduction": association.get("entropy_reduction"),
        "normalized_entropy_reduction": association.get(
            "normalized_entropy_reduction"
        ),
        "nmi": association.get("nmi"),
        "permutation_p": association.get("permutation_p"),
        "n_distinct_transformed": association.get("n_distinct_transformed"),
        "include_suggested": bool(recommendation.include),
        "include_override": None,
        "type_suggested": resolved_step1.get("datatype"),
        "type_override": None,
        "binning_suggested": binning.get("method"),
        "binning_override": None,
        "binning_n_bins_suggested": binning.get("n_bins"),
        "binning_n_bins_override": None,
        "discrete_suggested": bool(recommendation.discrete),
        "discrete_override": None,
        "reason_suggested": recommendation.reason,
        "reason_override": None,
    }


def merge_powerlaw_selection_dataframe(
    old_df: pd.DataFrame | None,
    new_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge step-2 selection rows, preserving override columns."""
    new_df = pd.DataFrame(new_rows)
    if new_df.empty:
        return new_df
    if old_df is None or old_df.empty or "attribute_name" not in old_df.columns:
        return new_df.sort_values("attribute_name").reset_index(drop=True)

    old_by_name = old_df.set_index("attribute_name", drop=False)
    merged_rows: list[dict[str, Any]] = []
    for row in new_df.to_dict(orient="records"):
        name = row["attribute_name"]
        if name in old_by_name.index:
            old = old_by_name.loc[name]
            if isinstance(old, pd.DataFrame):
                old = old.iloc[0]
            for col in POWERLAW_OVERRIDE_COLUMNS:
                if col in old_by_name.columns and not _is_blank(old.get(col)):
                    row[col] = old.get(col)
        merged_rows.append(row)
    return (
        pd.DataFrame(merged_rows)
        .sort_values("attribute_name")
        .reset_index(drop=True)
    )


def resolve_powerlaw_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve effective power-law selection settings from one CSV row."""
    include_raw = resolve_field(
        row.get("include_suggested"), row.get("include_override"), default=False
    )
    include = _parse_bool(include_raw, field_name="include")
    datatype = validate_datatype(
        resolve_field(
            row.get("type_suggested"),
            row.get("type_override"),
            default="unsupported",
        )
    )
    method = resolve_field(
        row.get("binning_suggested"), row.get("binning_override"), default=None
    )
    n_bins = resolve_field(
        row.get("binning_n_bins_suggested"),
        row.get("binning_n_bins_override"),
        default=None,
    )
    if datatype == "continuous" and not _is_blank(method):
        cleaned = validate_binning(datatype, method, n_bins)
    elif datatype == "continuous" and include:
        cleaned = validate_binning(datatype, method, n_bins)
    else:
        cleaned = {"method": None, "n_bins": None, "bin_edges": None}

    reason = resolve_field(
        row.get("reason_suggested"), row.get("reason_override"), default=""
    )
    discrete_raw = resolve_field(
        row.get("discrete_suggested"),
        row.get("discrete_override"),
        default=None,
    )
    if _is_blank(discrete_raw):
        # Legacy CSVs without discrete_*: continuous magnitudes used continuous MLE.
        discrete = datatype != "continuous"
    else:
        discrete = _parse_bool(discrete_raw, field_name="discrete")
    return {
        "attribute_name": str(row.get("attribute_name")),
        "include": include,
        "datatype": datatype,
        "discrete": discrete,
        "binning": cleaned,
        "reason": "" if _is_blank(reason) else str(reason),
        "metadata": {
            "source_level": row.get("source_level"),
            "n_cases": row.get("n_cases"),
            "n_distinct": row.get("n_distinct"),
            "is_near_constant": None,
        },
        "behavioral_relevance": {
            "status": row.get("status"),
            "reason": row.get("association_reason"),
            "entropy_variant": row.get("entropy_variant"),
            "entropy_attribute": row.get("entropy_attribute"),
            "entropy_conditional": row.get("entropy_conditional"),
            "entropy_reduction": row.get("entropy_reduction"),
            "normalized_entropy_reduction": row.get("normalized_entropy_reduction"),
            "nmi": row.get("nmi"),
            "permutation_p": row.get("permutation_p"),
            "n_distinct_transformed": row.get("n_distinct_transformed"),
        },
        "attribute_status": "present",
    }


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
