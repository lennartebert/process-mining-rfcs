"""Unit tests for the three-step case-attribute HITL pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.rfc.case_attribute_association import (
    associate_attribute_with_variant,
    entropy_reduction_metrics,
    shannon_entropy,
)
from utils.rfc.case_attribute_config import (
    build_inventory_row,
    build_powerlaw_selection_row,
    merge_inventory_dataframe,
    resolve_field,
    resolve_inventory_row,
    resolve_powerlaw_row,
)
from utils.rfc.case_attribute_inventory import (
    RecommendationThresholds,
    describe_attribute,
    infer_attribute_type,
    recommend_datatype,
    recommend_powerlaw_discrete,
    recommend_powerlaw_include,
    values_are_integer_valued,
)
from utils.rfc.case_attribute_transform import (
    TransformError,
    extract_continuous_values,
    log_bin_edges,
    transform_by_datatype,
    value_frequency_counts,
)
from utils.io.case_tables import is_default_excluded_attribute
from utils.rfc.powerlaw_pipeline import analyze_powerlaw_data


class TypeInferenceTests(unittest.TestCase):
    def test_low_cardinality_integers_are_categorical(self) -> None:
        values = [1, 2, 1, 2, 3, 1, 2, 3, 1, 2] * 5
        self.assertEqual(infer_attribute_type(values, "LoanCode"), "categorical")

    def test_amount_like_numeric_is_continuous(self) -> None:
        values = ([5000] * 50 + [7000] * 30 + list(range(1000, 1600)))
        self.assertEqual(infer_attribute_type(values, "AMOUNT_REQ"), "continuous")

    def test_high_cardinality_numeric_is_continuous(self) -> None:
        values = list(range(40)) * 3
        self.assertEqual(infer_attribute_type(values, "Amount"), "continuous")

    def test_boolean_inference(self) -> None:
        values = [True, False, True, False, True]
        self.assertEqual(infer_attribute_type(values, "Accepted"), "boolean")

    def test_nested_unsupported(self) -> None:
        values = [{"a": 1}, {"a": 2}, {"a": 1}]
        self.assertEqual(infer_attribute_type(values, "Payload"), "unsupported")

    def test_unique_ids_are_identifier_like(self) -> None:
        values = [f"id-{i}" for i in range(100)]
        self.assertEqual(infer_attribute_type(values, "CustomerId"), "identifier-like")


class AssociationTests(unittest.TestCase):
    def test_perfect_dependence_ner_is_one(self) -> None:
        variants = ["A", "A", "B", "B"]
        attrs = ["x", "x", "y", "y"]
        metrics = entropy_reduction_metrics(variants, attrs)
        self.assertEqual(metrics["status"], "computed")
        self.assertAlmostEqual(metrics["normalized_entropy_reduction"], 1.0, places=6)
        self.assertAlmostEqual(
            metrics["entropy_reduction"], shannon_entropy(variants), places=6
        )

    def test_binned_continuous_association(self) -> None:
        df = pd.DataFrame({"Amount": list(range(1, 101))})
        transformed = transform_by_datatype(
            df,
            "Amount",
            datatype="continuous",
            binning={"method": "log_bins", "n_bins": 5},
        )
        variants = ["A", "B"] * 50
        result = associate_attribute_with_variant(
            variants,
            transformed.tolist(),
            attribute_name="Amount",
            inferred_type="categorical",
            source_level="case",
            n_permutations=0,
        )
        self.assertEqual(result["status"], "computed")


class ConfigResolveTests(unittest.TestCase):
    def test_override_wins(self) -> None:
        self.assertFalse(resolve_field(True, False))

    def test_blank_override_keeps_suggested(self) -> None:
        self.assertEqual(resolve_field("categorical", None), "categorical")
        self.assertEqual(resolve_field("categorical", float("nan")), "categorical")

    def test_inventory_csv_merge_preserves_overrides(self) -> None:
        inventory = describe_attribute(
            list(range(40)) * 3,
            attribute_name="Amount",
            source_level="case",
        )
        rec = recommend_datatype(inventory)
        suggested = [build_inventory_row(inventory, rec)]
        first = merge_inventory_dataframe(None, suggested)
        self.assertTrue(bool(first.loc[0, "include_suggested"]))
        self.assertEqual(first.loc[0, "type_suggested"], "continuous")
        self.assertEqual(first.loc[0, "binning_suggested"], "log_bins")
        self.assertEqual(int(first.loc[0, "binning_n_bins_suggested"]), 10)

        first.loc[0, "include_override"] = False
        first.loc[0, "type_override"] = "continuous"
        first.loc[0, "binning_override"] = "quantile_bins"
        first.loc[0, "binning_n_bins_override"] = 8

        regenerated = merge_inventory_dataframe(first, suggested)
        resolved = resolve_inventory_row(regenerated.iloc[0].to_dict())
        self.assertFalse(resolved["include"])
        self.assertEqual(resolved["datatype"], "continuous")
        self.assertEqual(resolved["binning"]["method"], "quantile_bins")
        self.assertEqual(resolved["binning"]["n_bins"], 8)

    def test_org_resource_excluded_by_default(self) -> None:
        self.assertTrue(is_default_excluded_attribute("org:resource"))
        self.assertTrue(is_default_excluded_attribute("org:resource__first"))
        inventory = describe_attribute(
            [f"r{i % 5}" for i in range(50)],
            attribute_name="org:resource__first",
            source_level="event",
        )
        rec = recommend_datatype(inventory)
        self.assertFalse(rec.include)

    def test_continuous_powerlaw_include_ignores_bin_count(self) -> None:
        rec = recommend_powerlaw_include(
            datatype="continuous",
            n_distinct_transformed=10,
            is_near_constant=False,
            step1_included=True,
            association={"n_cases": 200, "normalized_entropy_reduction": 0.1},
            n_observations=200,
            n_raw_distinct=150,
            raw_values=np.arange(1, 201, dtype=float),
        )
        self.assertTrue(rec.include)
        self.assertTrue(rec.discrete)

    def test_integer_continuous_suggests_discrete_mle(self) -> None:
        self.assertTrue(values_are_integer_valued([5000, 7000, 10000.0]))
        self.assertFalse(values_are_integer_valued([1.5, 2.0, 3.25]))
        self.assertTrue(
            recommend_powerlaw_discrete(
                datatype="continuous", values=[5000, 7000, 15000]
            )
        )
        self.assertFalse(
            recommend_powerlaw_discrete(
                datatype="continuous", values=[1.5, 2.7, 10.0]
            )
        )
        self.assertTrue(recommend_powerlaw_discrete(datatype="categorical"))

    def test_discrete_override_in_selection_csv(self) -> None:
        row = {
            "attribute_name": "AMOUNT_REQ",
            "include_suggested": True,
            "include_override": None,
            "type_suggested": "continuous",
            "type_override": None,
            "binning_suggested": "log_bins",
            "binning_override": None,
            "binning_n_bins_suggested": 10,
            "binning_n_bins_override": None,
            "discrete_suggested": True,
            "discrete_override": False,
            "reason_suggested": "test",
            "reason_override": None,
        }
        resolved = resolve_powerlaw_row(row)
        self.assertFalse(resolved["discrete"])
        row["discrete_override"] = None
        self.assertTrue(resolve_powerlaw_row(row)["discrete"])
        # Legacy row without discrete_* columns
        legacy = {
            "attribute_name": "AMOUNT_REQ",
            "include_suggested": True,
            "type_suggested": "continuous",
            "binning_suggested": "log_bins",
            "binning_n_bins_suggested": 10,
        }
        self.assertFalse(resolve_powerlaw_row(legacy)["discrete"])

    def test_powerlaw_selection_row_includes_discrete(self) -> None:
        from utils.rfc.case_attribute_inventory import PowerlawIncludeRecommendation

        rec = PowerlawIncludeRecommendation(
            include=True, reason="ok", discrete=True
        )
        built = build_powerlaw_selection_row(
            attribute_name="AMOUNT_REQ",
            resolved_step1={
                "datatype": "continuous",
                "binning": {"method": "log_bins", "n_bins": 10},
                "metadata": {"source_level": "case", "n_cases": 100, "n_distinct": 20},
            },
            association={"status": "computed"},
            recommendation=rec,
        )
        self.assertTrue(built["discrete_suggested"])
        self.assertIsNone(built["discrete_override"])


class TransformTests(unittest.TestCase):
    def test_categorical_frequency_counts(self) -> None:
        df = pd.DataFrame({"Goal": ["A", "A", "B", "C", "A", "B"]})
        values = transform_by_datatype(df, "Goal", datatype="categorical")
        counts = value_frequency_counts(values)
        self.assertEqual(counts.tolist(), [3, 2, 1])

    def test_continuous_log_bins(self) -> None:
        df = pd.DataFrame({"Amount": list(range(1, 101))})
        values = transform_by_datatype(
            df,
            "Amount",
            datatype="continuous",
            binning={"method": "log_bins", "n_bins": 5},
        )
        counts = value_frequency_counts(values)
        self.assertGreaterEqual(counts.size, 2)
        self.assertEqual(int(counts.sum()), 100)

    def test_continuous_quantile_bins_still_supported(self) -> None:
        df = pd.DataFrame({"Amount": list(range(100))})
        values = transform_by_datatype(
            df,
            "Amount",
            datatype="continuous",
            binning={"method": "quantile_bins", "n_bins": 5},
        )
        counts = value_frequency_counts(values)
        self.assertGreaterEqual(counts.size, 2)
        self.assertEqual(int(counts.sum()), 100)

    def test_log_bin_edges_count(self) -> None:
        edges = log_bin_edges(np.arange(1, 101, dtype=float), n_bins=10)
        self.assertEqual(edges.size, 11)

    def test_extract_continuous_drops_nonpositive(self) -> None:
        df = pd.DataFrame({"Amount": [0, -1, 2.5, 10.0, None]})
        values = extract_continuous_values(df, "Amount")
        self.assertTrue(np.allclose(values, [2.5, 10.0]))

    def test_continuous_without_binning_raises(self) -> None:
        df = pd.DataFrame({"Amount": list(range(20))})
        with self.assertRaises(TransformError):
            transform_by_datatype(df, "Amount", datatype="continuous", binning={})


class PipelineSmokeTests(unittest.TestCase):
    def test_analyze_powerlaw_discrete_smoke(self) -> None:
        ranks = np.arange(1, 80)
        freqs = np.maximum(1, (1000 / ranks**1.5).astype(int))
        results = analyze_powerlaw_data(
            freqs,
            log_name="synthetic",
            input_path="memory",
            discrete=True,
            n_bootstraps=2,
            random_seed=0,
            minimum_fitted_variants=10,
            significance_level=0.10,
        )
        self.assertIn("full_range_power_law", results)
        self.assertIn("summary_row", results["full_range_power_law"])

    def test_analyze_powerlaw_continuous_smoke(self) -> None:
        rng = np.random.default_rng(0)
        values = (rng.pareto(2.5, size=500) + 1.0) * 10.0
        results = analyze_powerlaw_data(
            values,
            log_name="synthetic::Amount",
            input_path="memory",
            discrete=False,
            n_bootstraps=2,
            random_seed=0,
            minimum_fitted_variants=10,
            significance_level=0.10,
        )
        self.assertIn("full_range_power_law", results)
        self.assertIn("alpha", results["full_range_power_law"]["summary_row"])


class CaseTableConstantTests(unittest.TestCase):
    def test_only_first_event_aggregation_is_materialised(self) -> None:
        from pm4py.objects.log.obj import Event, EventLog, Trace

        from utils.io.case_tables import build_case_attribute_table

        trace = Trace()
        trace.attributes["concept:name"] = "c1"
        e1 = Event(
            {
                "concept:name": "Create Fine",
                "vehicleClass": "A",
                "time:timestamp": "2013-01-01",
            }
        )
        e2 = Event(
            {
                "concept:name": "Send Fine",
                "vehicleClass": "A",
                "time:timestamp": "2013-01-02",
            }
        )
        trace.append(e1)
        trace.append(e2)
        log = EventLog([trace])

        case_df, source_levels = build_case_attribute_table(log)
        self.assertNotIn("vehicleClass", case_df.columns)
        self.assertIn("vehicleClass__first", case_df.columns)
        self.assertNotIn("vehicleClass__constant", case_df.columns)
        self.assertNotIn("vehicleClass__last", case_df.columns)
        self.assertNotIn("vehicleClass__mode", case_df.columns)
        self.assertNotIn("vehicleClass__n_unique", case_df.columns)
        self.assertEqual(case_df.loc[0, "vehicleClass__first"], "A")
        self.assertEqual(source_levels["vehicleClass__first"], "event")
        self.assertEqual(source_levels.get("vehicleClass"), "event")


if __name__ == "__main__":
    unittest.main()
