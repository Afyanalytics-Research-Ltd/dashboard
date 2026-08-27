"""
Tests for the retrieval-driven analytics pipeline (agents/schema_validation.py,
agents/retrieval.py) added in the Planner -> Retriever -> Query rewrite.

agents/tests.py is unrelated dead code — it tests a next_agent/agent_outputs/
iteration_count multi-agent architecture that no longer exists anywhere in
this codebase. Don't extend it; this file is the pipeline's own test suite.

Most of what's tested here is pure-function logic with no DB/network
dependency, so SimpleTestCase (no database setup) keeps these fast.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from django.test import SimpleTestCase, TestCase

from agents import catalog, catalog_sync, derived_metrics, expr_eval, retrieval, schema_validation, schema_writer
from agents.models import MetricDefinition


class ValidateFiltersTests(SimpleTestCase):
    """Regression tests for the two bugs this session's earlier fixes addressed:
    a date-range operator attached to a non-time field (crashed Cube.js by
    trying to cast a facility code to a timestamp), and a filter referencing
    a field that doesn't belong to the matched metric at all."""

    ALLOWED = {"fact_x.sex", "fact_x.source_schema", "fact_x.admission_month"}
    TIME_MEMBERS = {"fact_x.admission_month"}

    def test_drops_filter_on_field_not_belonging_to_metric(self):
        filters = [{"member": "fact_x.unrelated_field", "operator": "equals", "values": ["a"]}]
        result = schema_validation.validate_filters(filters, allowed_members=self.ALLOWED, time_members=self.TIME_MEMBERS)
        self.assertEqual(result, [])

    def test_drops_date_range_operator_on_non_time_field(self):
        """The exact bug: inDateRange attached to a facility-code string field."""
        filters = [{
            "member": "fact_x.source_schema", "operator": "inDateRange",
            "values": ["2023-09-01", "2023-09-30"],
        }]
        result = schema_validation.validate_filters(filters, allowed_members=self.ALLOWED, time_members=self.TIME_MEMBERS)
        self.assertEqual(result, [])

    def test_keeps_date_range_operator_on_real_time_field(self):
        filters = [{
            "member": "fact_x.admission_month", "operator": "inDateRange",
            "values": ["2023-09-01", "2023-09-30"],
        }]
        result = schema_validation.validate_filters(filters, allowed_members=self.ALLOWED, time_members=self.TIME_MEMBERS)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["member"], "fact_x.admission_month")

    def test_keeps_equals_filter_on_allowed_dimension(self):
        filters = [{"member": "fact_x.sex", "operator": "equals", "values": ["F"]}]
        result = schema_validation.validate_filters(filters, allowed_members=self.ALLOWED, time_members=self.TIME_MEMBERS)
        self.assertEqual(len(result), 1)

    def test_drops_filter_with_invalid_operator(self):
        filters = [{"member": "fact_x.sex", "operator": "regexMatches", "values": ["F"]}]
        result = schema_validation.validate_filters(filters, allowed_members=self.ALLOWED, time_members=self.TIME_MEMBERS)
        self.assertEqual(result, [])

    def test_no_allowed_members_means_no_field_restriction(self):
        """allowed_members=None (legacy call shape) skips the field-membership check."""
        filters = [{"member": "anything.goes", "operator": "equals", "values": ["x"]}]
        result = schema_validation.validate_filters(filters)
        self.assertEqual(len(result), 1)


class ValidMembersForTests(SimpleTestCase):
    def test_valid_members_for_includes_measures_dimensions_and_time(self):
        metric = {
            "cube_query": {
                "measures": ["fact_x.count"],
                "dimensions": ["fact_x.sex"],
                "timeDimensions": [{"dimension": "fact_x.admission_month"}],
            }
        }
        self.assertEqual(
            schema_validation.valid_members_for(metric),
            {"fact_x.count", "fact_x.sex", "fact_x.admission_month"},
        )

    def test_valid_time_members_for_empty_when_no_time_dimension(self):
        metric = {"cube_query": {"measures": ["fact_x.count"], "dimensions": [], "timeDimensions": []}}
        self.assertEqual(schema_validation.valid_time_members_for(metric), set())


class PromoteDateFiltersTests(SimpleTestCase):
    def test_moves_indaterange_filter_into_existing_time_dimension(self):
        query = {"timeDimensions": [{"dimension": "fact_x.admission_month", "granularity": "month"}]}
        filters = [{"member": "fact_x.admission_month", "operator": "inDateRange", "values": ["2023-09-01", "2023-09-30"]}]
        updated_query, remaining = schema_validation.promote_date_filters(query, filters)
        self.assertEqual(remaining, [])
        self.assertEqual(updated_query["timeDimensions"][0]["dateRange"], ["2023-09-01", "2023-09-30"])

    def test_leaves_non_date_filter_untouched(self):
        query = {"timeDimensions": []}
        filters = [{"member": "fact_x.sex", "operator": "equals", "values": ["F"]}]
        updated_query, remaining = schema_validation.promote_date_filters(query, filters)
        self.assertEqual(remaining, filters)


class RetrievalRankingTests(SimpleTestCase):
    """Verifies the cosine-similarity ranking math in agents/retrieval.py
    against a small synthetic in-memory index — bypasses real file I/O and
    the OpenAI embeddings call entirely."""

    def _make_index(self):
        # Three orthonormal-ish 3-D vectors so similarity scores are exact
        # and unambiguous to reason about.
        vectors = np.array([
            [1.0, 0.0, 0.0],   # "closest" to the query vector below
            [0.0, 1.0, 0.0],   # orthogonal — score 0
            [0.7071, 0.7071, 0.0],  # 45 degrees off — partial match
        ], dtype=np.float32)
        return retrieval._EmbeddingsIndex(
            vectors=vectors,
            ids=["measure::a", "dimension::b", "metric::c"],
            sources=["measure", "dimension", "metric"],
            metadata=[
                {"cube": "fact_x", "field": "fact_x.a", "kind": "measure", "label": "A", "description": "", "metric_id": None, "glossary_term": None},
                {"cube": "fact_x", "field": "fact_x.b", "kind": "dimension", "label": "B", "description": "", "metric_id": None, "glossary_term": None},
                {"cube": None, "field": None, "kind": "metric", "label": "C", "description": "", "metric_id": "c", "glossary_term": None},
            ],
        )

    def test_retrieve_ranks_by_cosine_similarity_descending(self):
        index = self._make_index()
        query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        with patch.object(retrieval, "_load_index", return_value=index), \
             patch.object(retrieval, "_embed_query", return_value=query_vec):
            results = retrieval.retrieve("anything", top_k=3)

        self.assertEqual([r["id"] for r in results], ["measure::a", "metric::c", "dimension::b"])
        self.assertAlmostEqual(results[0]["score"], 1.0, places=3)
        self.assertAlmostEqual(results[2]["score"], 0.0, places=3)

    def test_retrieve_many_keeps_max_score_across_queries(self):
        index = self._make_index()

        def fake_embed(text):
            # First call favors "b", second favors "a" — retrieve_many should
            # keep the max score seen for each id across both calls.
            return np.array([0.0, 1.0, 0.0], dtype=np.float32) if text == "q1" else np.array([1.0, 0.0, 0.0], dtype=np.float32)

        with patch.object(retrieval, "_load_index", return_value=index), \
             patch.object(retrieval, "_embed_query", side_effect=fake_embed):
            results = retrieval.retrieve_many(["q1", "q2"], top_k=3)

        top_ids = [r["id"] for r in results[:2]]
        self.assertIn("measure::a", top_ids)
        self.assertIn("dimension::b", top_ids)


class ExprEvalTests(SimpleTestCase):
    """agents/expr_eval.py — the one piece of the derived-metrics pipeline
    that's pure Python with no I/O, so it gets the most rigorous coverage."""

    def test_basic_arithmetic(self):
        self.assertEqual(expr_eval.evaluate("a / b", {"a": 10, "b": 2}), 5.0)
        self.assertEqual(expr_eval.evaluate("a - b", {"a": 5, "b": 8}), -3)
        self.assertEqual(expr_eval.evaluate("a + b", {"a": 2, "b": 3}), 5)
        self.assertEqual(expr_eval.evaluate("a * b", {"a": 4, "b": 5}), 20)
        self.assertEqual(expr_eval.evaluate("-a", {"a": 4}), -4)

    def test_division_by_zero_raises(self):
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate("a / b", {"a": 10, "b": 0})

    def test_unbound_variable_raises(self):
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate("a + c", {"a": 1, "b": 2})

    def test_rejects_function_calls(self):
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate("__import__('os').system('x')", {})

    def test_rejects_power_operator(self):
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate("a ** b", {"a": 2, "b": 3})

    def test_rejects_comprehensions(self):
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate("[x for x in range(a)]", {"a": 1})

    def test_rejects_attribute_access(self):
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate("a.bit_length", {"a": 1})

    def test_rejects_string_literals_in_the_expression_itself(self):
        # Note: expr_eval trusts the caller to bind only numeric values in
        # `variables` (nodes_query._inject_computed_field always converts
        # via float() first) — this checks the expression string itself
        # can't smuggle in a string literal.
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate('"just a string"', {})

    def test_rejects_invalid_syntax(self):
        with self.assertRaises(expr_eval.ExprEvalError):
            expr_eval.evaluate("a +", {"a": 1})


class DerivedMetricsHelperTests(TestCase):
    """Mostly pure helper logic in agents/derived_metrics.py — grouping and
    shape checks that don't require an LLM or live Cube schema.
    TestCase (not SimpleTestCase) because _build_from_glossary_formula ->
    _base_query_for -> find_catalog_metric_containing_measure now queries
    MetricDefinition (agents/catalog.py is DB-backed, no longer YAML)."""

    def test_candidate_cube_prefers_explicit_cube_field(self):
        self.assertEqual(derived_metrics._candidate_cube({"cube": "fact_x", "field": "fact_y.count"}), "fact_x")

    def test_candidate_cube_falls_back_to_field_prefix(self):
        self.assertEqual(derived_metrics._candidate_cube({"cube": None, "field": "fact_x.count"}), "fact_x")

    def test_candidate_cube_none_when_nothing_to_go_on(self):
        self.assertIsNone(derived_metrics._candidate_cube({"cube": None, "field": None}))

    def test_measure_shaped_true_for_bare_measure(self):
        self.assertTrue(derived_metrics._measure_shaped({"source": "measure", "field": "fact_x.count"}))

    def test_measure_shaped_true_for_glossary_maps_to(self):
        self.assertTrue(derived_metrics._measure_shaped({"source": "glossary", "field": "fact_x.count", "formula": None}))

    def test_measure_shaped_false_for_glossary_formula(self):
        self.assertFalse(derived_metrics._measure_shaped({"source": "glossary", "field": "fact_x.count", "formula": "a/b"}))

    def test_measure_shaped_false_for_dimension(self):
        self.assertFalse(derived_metrics._measure_shaped({"source": "dimension", "field": "fact_x.sex"}))

    def test_build_from_glossary_formula_rejects_cross_cube_variables(self):
        """A curation bug (formula spanning two cubes) must be refused, not
        silently sent to Cube as an invalid same-cube query."""
        candidate = {
            "glossary_term": "bad formula", "formula": "a / b",
            "variables": {"a": "fact_x.count", "b": "fact_y.total"},
            "label": "bad", "description": "", "score": 0.9,
        }
        self.assertIsNone(derived_metrics._build_from_glossary_formula(candidate))

    def test_build_from_glossary_formula_rejects_unsafe_expression(self):
        candidate = {
            "glossary_term": "sneaky", "formula": "__import__('os')",
            "variables": {"a": "fact_x.count"}, "label": "x", "description": "", "score": 0.9,
        }
        self.assertIsNone(derived_metrics._build_from_glossary_formula(candidate))

    def test_build_from_glossary_formula_accepts_valid_same_cube_formula(self):
        candidate = {
            "glossary_term": "discount rate", "formula": "a / b",
            "variables": {"a": "fact_x.discount", "b": "fact_x.total"},
            "label": "Discount Rate", "description": "desc", "score": 0.8,
        }
        derived = derived_metrics._build_from_glossary_formula(candidate)
        self.assertIsNotNone(derived)
        self.assertEqual(derived["base_cube"], "fact_x")
        self.assertEqual(set(derived["cube_query"]["measures"]), {"fact_x.discount", "fact_x.total"})
        self.assertEqual(derived["variables"], {"a": "fact_x.discount", "b": "fact_x.total"})


class SchemaWriterTests(SimpleTestCase):
    """Pure logic in agents/schema_writer.py. The real Snowflake cardinality
    query and the real YAML file write are NOT exercised here — no safe way
    to do that without a live Snowflake connection and a disposable Cube
    schema copy; verify those manually per the implementation plan."""

    def test_fan_out_refuses_many_to_many_always(self):
        self.assertTrue(schema_writer.would_fan_out_beyond_tolerance("many_to_many", {"a": "sum"}))
        self.assertTrue(schema_writer.would_fan_out_beyond_tolerance("many_to_many", {}))

    def test_fan_out_refuses_sum_on_fanned_out_side(self):
        self.assertTrue(schema_writer.would_fan_out_beyond_tolerance("one_to_many", {"a": "sum"}))
        self.assertTrue(schema_writer.would_fan_out_beyond_tolerance("many_to_one", {"a": "count"}))

    def test_fan_out_allows_one_to_one(self):
        self.assertFalse(schema_writer.would_fan_out_beyond_tolerance("one_to_one", {"a": "sum"}))

    def test_fan_out_allows_non_additive_agg_on_fanned_out_side(self):
        self.assertFalse(schema_writer.would_fan_out_beyond_tolerance("one_to_many", {"a": "avg"}))

    def test_resolve_join_direction_one_to_many(self):
        owner, referenced, rel = schema_writer._resolve_join_direction("base", "target", "one_to_many")
        self.assertEqual((owner, referenced, rel), ("target", "base", "many_to_one"))

    def test_resolve_join_direction_many_to_one(self):
        owner, referenced, rel = schema_writer._resolve_join_direction("base", "target", "many_to_one")
        self.assertEqual((owner, referenced, rel), ("base", "target", "many_to_one"))

    def test_resolve_join_direction_one_to_one(self):
        owner, referenced, rel = schema_writer._resolve_join_direction("base", "target", "one_to_one")
        self.assertEqual((owner, referenced, rel), ("target", "base", "one_to_one"))

    def test_resolve_join_direction_rejects_many_to_many(self):
        with self.assertRaises(schema_writer.SchemaWriterError):
            schema_writer._resolve_join_direction("base", "target", "many_to_many")

    def test_find_candidate_join_key_prefers_patient_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "cube_a.yml").write_text(
                "cubes:\n  - name: cube_a\n    dimensions:\n"
                "      - name: patient_id\n        type: string\n"
                "      - name: facility\n        type: string\n"
            )
            (cubes_dir / "cube_b.yml").write_text(
                "cubes:\n  - name: cube_b\n    dimensions:\n"
                "      - name: patient_id\n        type: string\n"
                "      - name: facility\n        type: string\n"
            )
            with patch.object(schema_writer, "CUBES_DIR", cubes_dir):
                key = schema_writer.find_candidate_join_key("cube_a", "cube_b")
        self.assertEqual(key, "patient_id")

    def test_find_candidate_join_key_none_when_no_shared_dimension(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "cube_a.yml").write_text(
                "cubes:\n  - name: cube_a\n    dimensions:\n      - name: only_in_a\n        type: string\n"
            )
            (cubes_dir / "cube_b.yml").write_text(
                "cubes:\n  - name: cube_b\n    dimensions:\n      - name: only_in_b\n        type: string\n"
            )
            with patch.object(schema_writer, "CUBES_DIR", cubes_dir):
                key = schema_writer.find_candidate_join_key("cube_a", "cube_b")
        self.assertIsNone(key)

    def test_find_candidate_join_key_missing_file_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(schema_writer, "CUBES_DIR", Path(tmp)):
                key = schema_writer.find_candidate_join_key("nope_a", "nope_b")
        self.assertIsNone(key)


class CatalogDbTests(TestCase):
    """agents/catalog.py is DB-backed (MetricDefinition), not YAML-backed —
    verifies get_all()/get_by_id()/as_context() keep the exact shape every
    consumer (nodes_query.py, nodes.py, derived_metrics.py) already relies
    on, now sourced from the database instead of catalog/metrics.yaml."""

    def setUp(self):
        MetricDefinition.objects.create(
            metric_id="active_metric",
            name="Active Metric",
            description="An active metric.",
            cube_query={
                "measures": ["fact_x.count"],
                "dimensions": ["fact_x.sex"],
                "timeDimensions": [{"dimension": "fact_x.admission_month", "granularity": "month"}],
                "filters": [],
                "limit": 500,
            },
        )
        MetricDefinition.objects.create(
            metric_id="inactive_metric",
            name="Inactive Metric",
            description="Should not surface.",
            cube_query={},
            is_active=False,
        )

    def test_get_all_returns_only_active_metrics_in_dict_shape(self):
        result = catalog.get_all()
        ids = {m["id"] for m in result}
        self.assertIn("active_metric", ids)
        self.assertNotIn("inactive_metric", ids)
        active = next(m for m in result if m["id"] == "active_metric")
        self.assertEqual(set(active.keys()), {"id", "name", "description", "cube_query"})

    def test_get_by_id_returns_none_for_inactive_metric(self):
        self.assertIsNone(catalog.get_by_id("inactive_metric"))

    def test_get_by_id_returns_none_for_unknown_metric(self):
        self.assertIsNone(catalog.get_by_id("does_not_exist"))

    def test_get_by_id_returns_matching_active_metric(self):
        metric = catalog.get_by_id("active_metric")
        self.assertEqual(metric["name"], "Active Metric")

    def test_as_context_lists_dimension_and_date_fields(self):
        text = catalog.as_context()
        self.assertIn("[active_metric] Active Metric", text)
        self.assertIn("fact_x.sex", text)
        self.assertIn("fact_x.admission_month", text)

    def test_reload_is_a_harmless_noop(self):
        catalog.reload()  # must not raise — existing call site (nodes.re_classify) still calls it


class ValidateColumnExistsTests(SimpleTestCase):
    """agents/catalog_sync.py's live-Snowflake safety check for a proposed
    measure's sql_expression — the check that justifies staging
    PendingCubeMeasure for approval instead of auto-writing it."""

    def test_count_measure_needs_no_sql_passes_without_querying_snowflake(self):
        ok, msg = catalog_sync.validate_column_exists("some_cube", "")
        self.assertTrue(ok)

    def test_unquoted_expression_fails_to_extract_a_column(self):
        ok, msg = catalog_sync.validate_column_exists("some_cube", "{CUBE}.no_quotes_here")
        self.assertFalse(ok)

    def test_valid_column_passes_when_snowflake_query_succeeds(self):
        with patch("warehouse.services.snowflake.SnowflakeClient.query") as mock_query:
            mock_query.return_value = None
            ok, msg = catalog_sync.validate_column_exists("rpt_bed_occupancy", '{CUBE}."SOME_COLUMN"')
        self.assertTrue(ok)
        mock_query.assert_called_once()
        called_sql = mock_query.call_args[0][0]
        self.assertIn('"REPORTING"."RPT_BED_OCCUPANCY"', called_sql)
        self.assertIn('"SOME_COLUMN"', called_sql)

    def test_missing_column_fails_when_snowflake_query_raises(self):
        with patch("warehouse.services.snowflake.SnowflakeClient.query") as mock_query:
            mock_query.side_effect = Exception("column not found")
            ok, msg = catalog_sync.validate_column_exists("rpt_bed_occupancy", '{CUBE}."NOPE"')
        self.assertFalse(ok)
        self.assertIn("NOPE", msg)

    def test_cube_keyword_alone_is_not_treated_as_a_calculated_measure(self):
        """{CUBE} is Cube's own SQL-table-alias keyword, not a member
        reference — a plain {CUBE}."COLUMN" expression must still go
        through the Snowflake column check, not the member-existence one."""
        with patch("warehouse.services.snowflake.SnowflakeClient.query") as mock_query:
            mock_query.return_value = None
            ok, msg = catalog_sync.validate_column_exists("rpt_bed_occupancy", '{CUBE}."SOME_COLUMN"')
        self.assertTrue(ok)
        mock_query.assert_called_once()

    def test_calculated_measure_passes_when_all_referenced_members_exist(self):
        """The real bug hit in production: btr's sql (\"{total_admissions} /
        NULLIF({bed_count}, 0)\") has no quoted column at all, so it must be
        checked as a calculated measure — each {member} verified against
        the cube's own already-defined measures/dimensions instead."""
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    dimensions: []\n"
                "    measures:\n      - name: total_admissions\n        type: sum\n"
                "      - name: bed_count\n        type: max\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync.validate_column_exists(
                    "test_cube", "{total_admissions} / NULLIF({bed_count}, 0)"
                )
        self.assertTrue(ok)
        self.assertIn("total_admissions", msg)
        self.assertIn("bed_count", msg)

    def test_calculated_measure_fails_when_a_referenced_member_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: total_admissions\n        type: sum\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync.validate_column_exists(
                    "test_cube", "{total_admissions} / NULLIF({bed_count}, 0)"
                )
        self.assertFalse(ok)
        self.assertIn("bed_count", msg)

    def test_calculated_measure_fails_gracefully_when_cube_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(catalog_sync, "CUBES_DIR", Path(tmp)):
                ok, msg = catalog_sync.validate_column_exists(
                    "nonexistent_cube", "{total_admissions} / {bed_count}"
                )
        self.assertFalse(ok)


class _FakePendingMeasure:
    """Duck-typed stand-in for PendingCubeMeasure — write_pending_measure_to_yaml
    only ever reads these attributes (never queries the DB), so a plain
    stub avoids needing a real User row for requested_by/reviewed_by just
    to exercise the pure splice logic."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class WritePendingMeasureToYamlTests(SimpleTestCase):
    """The measures: splice — deliberately harder than
    schema_writer._splice_join_into_yaml's joins: [] replacement, since
    every cube file already has at least one measure (see
    agents/catalog_sync.py's write_pending_measure_to_yaml docstring)."""

    def _pending(self, **overrides):
        defaults = dict(
            cube_name="test_cube", measure_name="new_measure", measure_type="sum",
            sql_expression='{CUBE}."NEW_COLUMN"', title="New Measure",
            description="A test measure.", requested_by="tester",
        )
        defaults.update(overrides)
        return _FakePendingMeasure(**defaults)

    def test_inserts_new_measure_as_first_item_and_stays_valid_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
                "    pre_aggregations:\n      # placeholder\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir), \
                 patch.object(catalog_sync, "_notify_analytics_team_of_measure"):
                ok, msg = catalog_sync.write_pending_measure_to_yaml(self._pending())

            self.assertTrue(ok)
            text = (cubes_dir / "test_cube.yml").read_text()
            data = yaml.safe_load(text)
            measures = data["cubes"][0]["measures"]
            self.assertEqual([m["name"] for m in measures], ["new_measure", "count"])
            self.assertEqual(measures[0]["sql"], '{CUBE}."NEW_COLUMN"')
            self.assertEqual(measures[0]["type"], "sum")

    def test_refuses_when_cube_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(catalog_sync, "CUBES_DIR", Path(tmp)):
                ok, msg = catalog_sync.write_pending_measure_to_yaml(self._pending(cube_name="missing_cube"))
        self.assertFalse(ok)
        self.assertIn("does not exist", msg)

    def test_refuses_when_measures_key_appears_more_than_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
                "    pre_aggregations:\n      measures:\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync.write_pending_measure_to_yaml(self._pending())
        self.assertFalse(ok)
        self.assertIn("found 2", msg)

    def test_refuses_gracefully_when_file_is_not_writable(self):
        """The exact bug hit in production: the Cube container (running as
        root) touches these bind-mounted files, leaving them owned by a
        different user than the one running this app — write_text() then
        raises PermissionError, which must surface as (False, message), not
        an unhandled 500 that also skips the helpful diagnosis."""
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            cube_file = cubes_dir / "test_cube.yml"
            cube_file.write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
            )
            cube_file.chmod(0o444)
            try:
                with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                    ok, msg = catalog_sync.write_pending_measure_to_yaml(self._pending())
            finally:
                cube_file.chmod(0o644)  # tempdir cleanup needs write-back permission
        self.assertFalse(ok)
        self.assertIn("could not write", msg)

    def test_description_with_a_colon_is_safely_yaml_quoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir), \
                 patch.object(catalog_sync, "_notify_analytics_team_of_measure"):
                ok, msg = catalog_sync.write_pending_measure_to_yaml(
                    self._pending(description="Admissions per bed: per ward per month.")
                )
            self.assertTrue(ok)
            data = yaml.safe_load((cubes_dir / "test_cube.yml").read_text())
            self.assertEqual(
                data["cubes"][0]["measures"][0]["description"],
                "Admissions per bed: per ward per month.",
            )

    def test_yaml_scalar_never_line_wraps_regardless_of_length(self):
        """Direct regression guard on the actual root cause: yaml.safe_dump's
        default width=80 inserts a real newline mid-scalar for anything
        longer than 80 chars, confirmed via yaml.safe_dump(long_val,
        default_flow_style=True) producing '...THEN\\n  7 ELSE...' — proven
        to fail this exact assertion before the width=1_000_000 fix."""
        long_sql = (
            "CASE {CUBE}.\"WARD_NAME\" WHEN 'General Female' THEN 7 "
            "WHEN 'General Maternity' THEN 7 WHEN 'Pediatric General' THEN 6 "
            "WHEN 'General Male' THEN 4 ELSE NULL END"
        )
        self.assertGreater(len(long_sql), 80)
        result = catalog_sync._yaml_scalar(long_sql)
        self.assertNotIn("\n", result)
        self.assertEqual(result, long_sql)

    def test_long_sql_expression_does_not_get_line_wrapped_into_invalid_yaml(self):
        """The exact bug hit in production: yaml.safe_dump's default
        width=80 line-wraps a long CASE expression, and the wrapped
        continuation line lands at the wrong indentation once spliced into
        the cube file — producing YAML that fails to parse at all. Real
        example: a 7-branch CASE WHEN ward_name -> bed_count expression."""
        long_sql = (
            '{CUBE}."WARD_NAME" > 0 AND ('
            "CASE {CUBE}.\"WARD_NAME\" WHEN 'General Female' THEN 7 "
            "WHEN 'General Maternity' THEN 7 WHEN 'Pediatric General' THEN 6 "
            "WHEN 'General Male' THEN 4 WHEN 'Private Male' THEN 3 "
            "WHEN 'Private Female' THEN 3 WHEN 'Private Maternity' THEN 2 "
            "ELSE NULL END)"
        )
        self.assertGreater(len(long_sql), 80)  # actually exercises the wrap-prone path

        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir), \
                 patch.object(catalog_sync, "_notify_analytics_team_of_measure"):
                ok, msg = catalog_sync.write_pending_measure_to_yaml(
                    self._pending(sql_expression=long_sql, measure_type="max")
                )
            self.assertTrue(ok, msg)

            text = (cubes_dir / "test_cube.yml").read_text()
            data = yaml.safe_load(text)  # must not raise ScannerError
            self.assertEqual(data["cubes"][0]["measures"][0]["sql"], long_sql)

    def test_calculated_measure_sql_starting_with_brace_stays_valid_yaml(self):
        """A calculated measure's sql (e.g. btr's "{total_admissions} /
        NULLIF({bed_count}, 0)") starts with "{" — a YAML flow-mapping
        indicator — so it MUST come out quoted, not as a bare plain scalar,
        or it'd either fail to parse or be misread as a mapping instead of
        a string. yaml.safe_dump handles this automatically; this pins that
        behavior down as a regression guard rather than trusting it."""
        calculated_sql = "{total_admissions} / NULLIF({bed_count}, 0)"
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir), \
                 patch.object(catalog_sync, "_notify_analytics_team_of_measure"):
                ok, msg = catalog_sync.write_pending_measure_to_yaml(
                    self._pending(sql_expression=calculated_sql, measure_type="number")
                )
            self.assertTrue(ok, msg)

            data = yaml.safe_load((cubes_dir / "test_cube.yml").read_text())
            sql_value = data["cubes"][0]["measures"][0]["sql"]
            self.assertIsInstance(sql_value, str)
            self.assertEqual(sql_value, calculated_sql)


class ClassifyColumnTests(SimpleTestCase):
    """agents/catalog_sync.py's _classify_column — the heuristic behind
    sync_cube_schemas_from_snowflake. Real Snowflake DATA_TYPE values
    (TEXT/NUMBER/FLOAT/DATE/etc.), not guessed ones — confirmed against
    live REPORTING tables (RPT_DOCTOR_PERFORMANCE, RPT_BED_OCCUPANCY,
    FACT_INPATIENT_ADMISSIONS) before writing this classifier."""

    def test_bare_id_excluded(self):
        self.assertIsNone(catalog_sync._classify_column("ID", "NUMBER"))

    def test_id_suffix_excluded(self):
        self.assertIsNone(catalog_sync._classify_column("PATIENT_ID", "NUMBER"))

    def test_key_suffix_excluded(self):
        self.assertIsNone(catalog_sync._classify_column("FACILITY_KEY", "NUMBER"))

    def test_code_suffix_excluded(self):
        self.assertIsNone(catalog_sync._classify_column("WARD_CODE", "TEXT"))

    def test_date_becomes_time_dimension(self):
        self.assertEqual(catalog_sync._classify_column("ADMISSION_MONTH", "DATE"), ("time", "time"))

    def test_timestamp_variant_becomes_time_dimension(self):
        self.assertEqual(
            catalog_sync._classify_column("CREATED_AT", "TIMESTAMP_NTZ"), ("time", "time")
        )

    def test_plain_numeric_becomes_sum_measure(self):
        self.assertEqual(catalog_sync._classify_column("EVALUATIONS", "NUMBER"), ("measure", "sum"))

    def test_avg_prefix_becomes_avg_measure(self):
        self.assertEqual(catalog_sync._classify_column("AVG_LOS_DAYS", "NUMBER"), ("measure", "avg"))

    def test_rate_suffix_becomes_avg_measure(self):
        self.assertEqual(
            catalog_sync._classify_column("CONVERSION_RATE", "NUMBER"), ("measure", "avg")
        )

    def test_pct_suffix_becomes_avg_measure(self):
        self.assertEqual(
            catalog_sync._classify_column("CONVERSION_RATE_PCT", "NUMBER"), ("measure", "avg")
        )

    def test_percent_anywhere_becomes_avg_measure(self):
        self.assertEqual(
            catalog_sync._classify_column("STOCKOUT_PERCENTAGE", "NUMBER"), ("measure", "avg")
        )

    def test_text_becomes_string_dimension(self):
        self.assertEqual(catalog_sync._classify_column("USERNAME", "TEXT"), ("dimension", "string"))

    def test_boolean_becomes_boolean_dimension(self):
        self.assertEqual(
            catalog_sync._classify_column("IS_STOCKOUT", "BOOLEAN"), ("dimension", "boolean")
        )

    def test_semi_structured_type_is_unclassified(self):
        self.assertIsNone(catalog_sync._classify_column("METADATA_BLOB", "VARIANT"))


class SpliceNewFieldsIntoCubeYamlTests(SimpleTestCase):
    """agents/catalog_sync.py's _splice_new_fields_into_cube_yaml — batches
    one or more new measures AND dimensions into a single file write,
    generalizing write_pending_measure_to_yaml's proven single-measure
    splice (same regression coverage: must stay valid, parseable YAML)."""

    def test_adds_measure_and_dimension_in_one_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n"
                "    dimensions:\n      - name: existing_dim\n        type: string\n"
                "    measures:\n      - name: count\n        type: count\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync._splice_new_fields_into_cube_yaml(
                    "test_cube",
                    new_measures=[{"name": "evaluations", "sql": '{CUBE}."EVALUATIONS"', "type": "sum"}],
                    new_dimensions=[{"name": "username", "sql": '{CUBE}."USERNAME"', "type": "string"}],
                )
            self.assertTrue(ok, msg)

            data = yaml.safe_load((cubes_dir / "test_cube.yml").read_text())
            cube = data["cubes"][0]
            measure_names = {m["name"] for m in cube["measures"]}
            dim_names = {d["name"] for d in cube["dimensions"]}
            self.assertIn("evaluations", measure_names)
            self.assertIn("count", measure_names)  # existing measure untouched
            self.assertIn("username", dim_names)
            self.assertIn("existing_dim", dim_names)  # existing dimension untouched

    def test_noop_when_nothing_to_add(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
            )
            original = (cubes_dir / "test_cube.yml").read_text()
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync._splice_new_fields_into_cube_yaml("test_cube", [], [])
            self.assertTrue(ok)
            self.assertEqual((cubes_dir / "test_cube.yml").read_text(), original)

    def test_refuses_when_cube_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(catalog_sync, "CUBES_DIR", Path(tmp)):
                ok, msg = catalog_sync._splice_new_fields_into_cube_yaml(
                    "missing_cube", [{"name": "x", "sql": "", "type": "sum"}], []
                )
        self.assertFalse(ok)
        self.assertIn("does not exist", msg)

    def test_refuses_when_dimensions_key_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n"
                "    dimensions:\n      - name: existing_dim\n        type: string\n"
                "    measures:\n      - name: count\n        type: count\n"
                "    pre_aggregations:\n      dimensions:\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync._splice_new_fields_into_cube_yaml(
                    "test_cube", [], [{"name": "username", "sql": "", "type": "string"}]
                )
        self.assertFalse(ok)
        self.assertIn("dimensions", msg)


class SyncCubeSchemasFromSnowflakeTests(SimpleTestCase):
    """agents/catalog_sync.py's sync_cube_schemas_from_snowflake — the full
    orchestrator. Mocks SnowflakeClient.get_columns (never hits real
    Snowflake) with a temp CUBES_DIR standing in for model/cubes/."""

    def _columns_df(self, rows):
        return pd.DataFrame(rows, columns=["COLUMN_NAME", "DATA_TYPE", "NUMERIC_SCALE"])

    def test_adds_missing_columns_and_skips_already_declared(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "cube_a.yml").write_text(
                "cubes:\n  - name: cube_a\n"
                "    dimensions:\n      - name: username\n        type: string\n"
                "    measures:\n      - name: count\n        type: count\n"
            )
            columns = self._columns_df([
                ("USERNAME", "TEXT", None),       # already declared — must not be duplicated
                ("PATIENT_ID", "NUMBER", 0),      # identifier — excluded
                ("EVALUATIONS", "NUMBER", 0),     # new — sum measure
                ("CONVERSION_RATE_PCT", "NUMBER", 1),  # new — avg measure
                ("METADATA_BLOB", "VARIANT", None),    # unclassified
            ])
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir), \
                 patch("warehouse.services.snowflake.SnowflakeClient.get_columns", return_value=columns), \
                 patch.object(catalog_sync, "_notify_analytics_team_of_schema_sync"):
                summary = catalog_sync.sync_cube_schemas_from_snowflake(dry_run=False)

            self.assertEqual(summary["cubes_updated"], ["cube_a"])
            self.assertEqual(set(summary["fields_added"]["cube_a"]), {"evaluations", "conversion_rate_pct"})
            self.assertEqual(summary["skipped_unclassified"]["cube_a"], ["METADATA_BLOB"])
            self.assertEqual(summary["errors"], {})

            data = yaml.safe_load((cubes_dir / "cube_a.yml").read_text())
            cube = data["cubes"][0]
            measure_names = {m["name"] for m in cube["measures"]}
            dim_names = {d["name"] for d in cube["dimensions"]}
            self.assertIn("evaluations", measure_names)
            evaluations = next(m for m in cube["measures"] if m["name"] == "evaluations")
            self.assertEqual(evaluations["type"], "sum")
            conversion = next(m for m in cube["measures"] if m["name"] == "conversion_rate_pct")
            self.assertEqual(conversion["type"], "avg")
            self.assertNotIn("patient_id", measure_names)
            self.assertNotIn("patient_id", dim_names)
            self.assertEqual(dim_names, {"username"})  # unchanged — username was already declared

    def test_dry_run_computes_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            original = (
                "cubes:\n  - name: cube_a\n    measures:\n      - name: count\n        type: count\n"
            )
            (cubes_dir / "cube_a.yml").write_text(original)
            columns = self._columns_df([("EVALUATIONS", "NUMBER", 0)])
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir), \
                 patch("warehouse.services.snowflake.SnowflakeClient.get_columns", return_value=columns), \
                 patch.object(catalog_sync, "_notify_analytics_team_of_schema_sync") as mock_notify:
                summary = catalog_sync.sync_cube_schemas_from_snowflake(dry_run=True)

            self.assertEqual(summary["cubes_updated"], ["cube_a"])
            self.assertEqual((cubes_dir / "cube_a.yml").read_text(), original)  # untouched
            mock_notify.assert_not_called()  # no email for a preview

    def test_one_cube_error_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "cube_a.yml").write_text(
                "cubes:\n  - name: cube_a\n    measures:\n      - name: count\n        type: count\n"
            )
            (cubes_dir / "cube_b.yml").write_text(
                "cubes:\n  - name: cube_b\n    measures:\n      - name: count\n        type: count\n"
            )
            columns = self._columns_df([("EVALUATIONS", "NUMBER", 0)])

            def _get_columns_side_effect(schema, table):
                if table == "CUBE_A":
                    raise Exception("Snowflake connection reset")
                return columns

            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir), \
                 patch(
                     "warehouse.services.snowflake.SnowflakeClient.get_columns",
                     side_effect=_get_columns_side_effect,
                 ), \
                 patch.object(catalog_sync, "_notify_analytics_team_of_schema_sync"):
                summary = catalog_sync.sync_cube_schemas_from_snowflake(dry_run=False)

            self.assertIn("cube_a", summary["errors"])
            self.assertEqual(summary["cubes_updated"], ["cube_b"])


class GetCubeMeasureDefinitionTests(SimpleTestCase):
    """agents/catalog_sync.py's get_cube_measure_definition — reads a
    measure's current sql/type/title/description straight from the cube's
    own YAML (Cube's /meta doesn't expose a measure's SQL), used to
    pre-fill the Edit Measure form."""

    def test_returns_full_definition_for_existing_measure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n"
                "      - name: revenue\n        sql: '{CUBE}.\"REVENUE\"'\n"
                "        type: sum\n        title: Revenue\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                result = catalog_sync.get_cube_measure_definition("test_cube", "revenue")
        self.assertEqual(result["sql"], '{CUBE}."REVENUE"')
        self.assertEqual(result["type"], "sum")
        self.assertEqual(result["title"], "Revenue")
        self.assertEqual(result["description"], "")

    def test_returns_none_for_unknown_measure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "test_cube.yml").write_text(
                "cubes:\n  - name: test_cube\n    measures:\n      - name: count\n        type: count\n"
            )
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                result = catalog_sync.get_cube_measure_definition("test_cube", "nonexistent")
        self.assertIsNone(result)

    def test_returns_none_for_missing_cube_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(catalog_sync, "CUBES_DIR", Path(tmp)):
                result = catalog_sync.get_cube_measure_definition("nonexistent_cube", "count")
        self.assertIsNone(result)


class ReplaceFieldInCubeYamlTests(SimpleTestCase):
    """agents/catalog_sync.py's _replace_field_in_cube_yaml — the edit
    counterpart to the insert-new-item splice, exercised here against the
    same kind of real, multi-measure file that caught earlier splice bugs
    (see WritePendingMeasureToYamlTests) rather than a trivial fixture."""

    _REAL_CUBE_YAML = (
        "cubes:\n"
        "  - name: rpt_bed_occupancy\n"
        "    dimensions:\n"
        "      - name: ward_name\n        type: string\n"
        "    measures:\n"
        "      - name: bti_days\n"
        "        sql: ({bed_count} * {days_in_month} - {total_bed_days}) / NULLIF({discharged_admissions}, 0)\n"
        "        type: number\n"
        "      - name: bed_count\n"
        "        sql: CASE {CUBE}.\"WARD_NAME\" WHEN 'General Female' THEN 7 ELSE NULL END\n"
        "        type: max\n"
        "      - name: count\n"
        "        type: count\n"
        "      - name: avg_admission_cost\n"
        "        sql: '{CUBE}.\"AVG_ADMISSION_COST\"'\n"
        "        type: sum\n"
        "    pre_aggregations:\n"
        "      # placeholder\n"
    )

    def test_edits_a_middle_measure_leaving_others_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "rpt_bed_occupancy.yml").write_text(self._REAL_CUBE_YAML)
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync._replace_field_in_cube_yaml(
                    "rpt_bed_occupancy", "measures", "count",
                    sql_expression="", field_type="count", title="Total Rows", description="",
                )
            self.assertTrue(ok, msg)

            data = yaml.safe_load((cubes_dir / "rpt_bed_occupancy.yml").read_text())
            measures = {m["name"]: m for m in data["cubes"][0]["measures"]}
            self.assertEqual(set(measures), {"bti_days", "bed_count", "count", "avg_admission_cost"})
            self.assertEqual(measures["count"]["title"], "Total Rows")
            self.assertEqual(measures["count"]["type"], "count")
            self.assertNotIn("sql", measures["count"])
            # untouched neighbors, including calculated-measure braces and CASE expr
            self.assertEqual(
                measures["bti_days"]["sql"],
                "({bed_count} * {days_in_month} - {total_bed_days}) / NULLIF({discharged_admissions}, 0)",
            )
            self.assertIn("CASE", measures["bed_count"]["sql"])
            self.assertEqual(data["cubes"][0]["dimensions"][0]["name"], "ward_name")

    def test_edits_the_last_measure_without_eating_pre_aggregations(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "rpt_bed_occupancy.yml").write_text(self._REAL_CUBE_YAML)
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync._replace_field_in_cube_yaml(
                    "rpt_bed_occupancy", "measures", "avg_admission_cost",
                    sql_expression='{CUBE}."AVG_ADMISSION_COST"', field_type="avg",
                    title="", description="",
                )
            self.assertTrue(ok, msg)
            data = yaml.safe_load((cubes_dir / "rpt_bed_occupancy.yml").read_text())
            self.assertIn("pre_aggregations", data["cubes"][0])
            self.assertEqual(len(data["cubes"][0]["measures"]), 4)
            edited = next(m for m in data["cubes"][0]["measures"] if m["name"] == "avg_admission_cost")
            self.assertEqual(edited["type"], "avg")

    def test_refuses_when_measure_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            cubes_dir = Path(tmp)
            (cubes_dir / "rpt_bed_occupancy.yml").write_text(self._REAL_CUBE_YAML)
            with patch.object(catalog_sync, "CUBES_DIR", cubes_dir):
                ok, msg = catalog_sync._replace_field_in_cube_yaml(
                    "rpt_bed_occupancy", "measures", "does_not_exist",
                    sql_expression="", field_type="sum", title="", description="",
                )
        self.assertFalse(ok)
        self.assertIn("could not find", msg)

    def test_refuses_when_cube_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(catalog_sync, "CUBES_DIR", Path(tmp)):
                ok, msg = catalog_sync._replace_field_in_cube_yaml(
                    "missing_cube", "measures", "count",
                    sql_expression="", field_type="count", title="", description="",
                )
        self.assertFalse(ok)
        self.assertIn("does not exist", msg)
