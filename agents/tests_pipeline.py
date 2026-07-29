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
from django.test import SimpleTestCase

from agents import derived_metrics, expr_eval, retrieval, schema_validation, schema_writer


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


class DerivedMetricsHelperTests(SimpleTestCase):
    """Pure helper logic in agents/derived_metrics.py — grouping and shape
    checks that don't require an LLM or live Cube schema."""

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
