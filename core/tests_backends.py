"""Tests for the platform backend switches (warehouse / LLM provider).

Airflow and Redash are mocked; the real end-to-end checks live in
afya_backends/tests (Databend) and the staging verification.
"""

import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings

from afya_backends import toggles

from . import backends
from .models import AuditLog, Notification, SystemSettings

User = get_user_model()
SWITCH_URL = "/core/settings/backends/"
STATUS_URL = "/core/settings/backends/status/"
SETTINGS_URL = "/core/settings/"
TRANSLATE_URL = "/core/api/warehouse/translate/"


class _Base(TestCase):
    def setUp(self):
        cache.clear()
        toggles.clear_cache()
        self.admin = User.objects.create_superuser("root", "root@example.com", "pw")
        self.other_admin = User.objects.create_superuser("root2", "root2@example.com", "pw")
        self.staff = User.objects.create_user("staff", "s@example.com", "pw", is_staff=True)
        # No real propagation in unit tests (PropagationTests re-enable it).
        self._prop_patcher = mock.patch.object(backends, "_propagate")
        self.propagate = self._prop_patcher.start()
        self.addCleanup(mock.patch.stopall)
        self.addCleanup(toggles.clear_cache)

    def real_propagation(self):
        self._prop_patcher.stop()


class SwitchTests(_Base):
    def test_default_is_snowflake_and_api(self):
        self.assertEqual(toggles.warehouse_backend(), "snowflake")
        self.assertEqual(toggles.llm_provider(), "api")

    def test_superuser_switches_warehouse(self):
        self.client.force_login(self.admin)
        r = self.client.post(SWITCH_URL, {"switch": "warehouse_backend", "value": "databend"})
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertEqual((body["previous"], body["value"]), ("snowflake", "databend"))
        self.assertEqual(SystemSettings.get("warehouse_backend"), "databend")
        toggles.clear_cache()
        self.assertEqual(toggles.warehouse_backend(), "databend")
        self.propagate.assert_called_once()
        self.assertTrue(AuditLog.objects.filter(resource="PlatformBackend", resource_id="warehouse_backend").exists())
        # other superusers are told
        self.assertTrue(Notification.objects.filter(user=self.other_admin, title__icontains="Databend").exists())
        self.assertFalse(Notification.objects.filter(user=self.admin, title__icontains="Databend").exists())

    def test_llm_switch_and_aliases(self):
        self.client.force_login(self.admin)
        r = self.client.post(SWITCH_URL, {"switch": "llm_provider", "value": "local"})
        self.assertEqual(r.json()["value"], "ollama")
        toggles.clear_cache()
        self.assertEqual(toggles.llm_provider(), "ollama")

    def test_invalid_value_rejected(self):
        self.client.force_login(self.admin)
        r = self.client.post(SWITCH_URL, {"switch": "warehouse_backend", "value": "postgres"})
        self.assertEqual(r.status_code, 400)
        self.assertIsNone(SystemSettings.get("warehouse_backend"))

    def test_unknown_switch_rejected(self):
        self.client.force_login(self.admin)
        r = self.client.post(SWITCH_URL, {"switch": "everything", "value": "databend"})
        self.assertEqual(r.status_code, 400)

    def test_non_superuser_cannot_switch(self):
        self.client.force_login(self.staff)
        r = self.client.post(SWITCH_URL, {"switch": "warehouse_backend", "value": "databend"})
        self.assertIn(r.status_code, (302, 403))
        self.assertIsNone(SystemSettings.get("warehouse_backend"))

    def test_anonymous_cannot_switch(self):
        r = self.client.post(SWITCH_URL, {"switch": "warehouse_backend", "value": "databend"})
        self.assertIn(r.status_code, (302, 403))

    def test_generic_settings_editor_routes_through_switch(self):
        self.client.force_login(self.admin)
        r = self.client.post(SETTINGS_URL, {"key": "warehouse_backend", "value": '"databend"'})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["value"], "databend")
        self.propagate.assert_called_once()
        bad = self.client.post(SETTINGS_URL, {"key": "warehouse_backend", "value": "oracle"})
        self.assertEqual(bad.status_code, 400)
        # ordinary keys still work as before
        ok = self.client.post(SETTINGS_URL, {"key": "max_export_rows", "value": "500"})
        self.assertEqual(ok.json()["value"], 500)

    def test_settings_page_renders_switches(self):
        self.client.force_login(self.admin)
        SystemSettings.set("warehouse_backend", "databend")
        toggles.clear_cache()
        r = self.client.get(SETTINGS_URL)
        self.assertEqual(r.status_code, 200)
        html = r.content.decode()
        self.assertIn("Platform backends", html)
        self.assertIn('data-value="databend"\n                  class="active"', html)
        self.assertIn("Ollama (local)", html)


class PropagationTests(_Base):
    def test_propagate_runs_steps_and_records_progress(self):
        self.real_propagation()
        with mock.patch.object(backends, "_sync_airflow", return_value="afya_warehouse_backend = databend") as af, \
                mock.patch.object(backends, "sync_redash", return_value="3 queries now on databend") as rd:
            backends.switch("warehouse_backend", "databend", user=self.admin, propagate_async=False)
        af.assert_called_once_with("warehouse_backend", "databend")
        rd.assert_called_once_with("databend")
        prog = cache.get(backends.PROGRESS_KEY)
        self.assertTrue(prog["done"])
        self.assertEqual([s["name"] for s in prog["steps"]],
                         ["Saved setting", "Airflow Variable", "Redash queries", "Cube"])
        self.assertTrue(all(s["ok"] for s in prog["steps"]))

    def test_failed_step_is_reported_not_raised(self):
        self.real_propagation()
        with mock.patch.object(backends, "_sync_airflow", side_effect=RuntimeError("Airflow down")), \
                mock.patch.object(backends, "sync_redash", return_value="ok"):
            res = backends.switch("llm_provider", "ollama", user=self.admin, propagate_async=False)
        self.assertEqual(SystemSettings.get("llm_provider"), "ollama")  # the switch itself held
        steps = cache.get(backends.PROGRESS_KEY)["steps"]
        af = next(s for s in steps if s["name"] == "Airflow Variable")
        self.assertFalse(af["ok"])
        self.assertIn("Airflow down", af["detail"])
        self.assertEqual(res["value"], "ollama")

    def test_llm_switch_does_not_touch_redash(self):
        self.real_propagation()
        with mock.patch.object(backends, "_sync_airflow", return_value="x"), \
                mock.patch.object(backends, "sync_redash") as rd:
            backends.switch("llm_provider", "ollama", user=self.admin, propagate_async=False)
        rd.assert_not_called()

    def test_airflow_variable_create_when_missing(self):
        client = mock.Mock()
        resp404 = mock.Mock(status_code=404, ok=False)
        resp200 = mock.Mock(status_code=200, ok=True)
        client.request.side_effect = [resp404, resp200]
        with mock.patch("airflow_ui.client.get_client", return_value=client):
            out = backends._sync_airflow("warehouse_backend", "databend")
        self.assertEqual(out, "afya_warehouse_backend = databend")
        self.assertEqual(client.request.call_args_list[0].args[:2], ("PATCH", "/variables/afya_warehouse_backend"))
        self.assertEqual(client.request.call_args_list[1].args[:2], ("POST", "/variables"))

    def test_redash_twins_and_moves_queries_both_ways(self):
        rd = mock.Mock()
        sources = [{"id": 1, "type": "snowflake", "name": "reporting"}, {"id": 2, "type": "pg", "name": "other"}]
        queries = {"count": 3, "results": [
            {"id": 10, "data_source_id": 1, "version": 1},
            {"id": 11, "data_source_id": 1, "version": 4},
            {"id": 12, "data_source_id": 2, "version": 1},
        ]}
        moves = []

        def request(method, path, json=None, **kw):
            if path == "/api/data_sources" and method == "POST":
                self.assertEqual(json["type"], backends.DATABEND_RUNNER_TYPE)
                return mock.Mock(ok=True, json=lambda: {"id": 7})
            moves.append((path, json["data_source_id"]))
            return mock.Mock(ok=True)

        def get_json(path, **params):
            if path == "/api/data_sources":
                return sources
            return queries
        rd.get_json.side_effect = get_json
        rd.request.side_effect = request
        with mock.patch.object(backends, "_redash", return_value=rd):
            msg = backends.sync_redash("databend")
            self.assertEqual(sorted(moves), [("/api/queries/10", 7), ("/api/queries/11", 7)])
            self.assertIn("2 queries", msg)
            # back to Snowflake: everything on the twin returns to source 1
            for q in queries["results"][:2]:
                q["data_source_id"] = 7
            queries["results"].append({"id": 13, "data_source_id": 7, "version": 1})  # created on Databend
            moves.clear()
            backends.sync_redash("snowflake")
        self.assertEqual(sorted(moves), [("/api/queries/10", 1), ("/api/queries/11", 1), ("/api/queries/13", 1)])
        self.assertEqual(SystemSettings.get(backends.REDASH_MAP_KEY)["twins"], {"1": 7})


class StatusTests(_Base):
    def test_status_shallow(self):
        self.client.force_login(self.admin)
        r = self.client.get(STATUS_URL + "?deep=0")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["warehouse_backend"], "snowflake")

    def test_status_requires_superuser(self):
        self.client.force_login(self.staff)
        self.assertIn(self.client.get(STATUS_URL).status_code, (302, 403))


@override_settings(AFYA_INTERNAL_API_TOKEN="s3cret")
class TranslateAPITests(_Base):
    def post(self, body, token="s3cret"):
        return self.client.post(TRANSLATE_URL, data=json.dumps(body), content_type="application/json",
                                HTTP_X_AFYA_TOKEN=token)

    def test_requires_token(self):
        self.assertEqual(self.post({"sql": "select 1"}, token="nope").status_code, 401)
        self.assertEqual(self.client.get(TRANSLATE_URL).status_code, 401)

    @override_settings(AFYA_INTERNAL_API_TOKEN="")
    def test_disabled_without_configured_token(self):
        self.assertEqual(self.post({"sql": "select 1"}, token="").status_code, 401)

    def test_translates(self):
        r = self.post({"sql": "select id from HOSPITALS.REPORTING.RPT_X where name ilike 'a%'; USE WAREHOUSE W"})
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertEqual(body["statements"], ['SELECT "ID" FROM "REPORTING"."RPT_X" WHERE LOWER("NAME") LIKE LOWER(\'a%\')'])
        self.assertEqual(body["actions"], ["sql", "noop"])

    def test_get_reports_active_switches(self):
        SystemSettings.set("warehouse_backend", "databend")
        toggles.clear_cache()
        r = self.client.get(TRANSLATE_URL, HTTP_X_AFYA_TOKEN="s3cret")
        self.assertEqual(r.json()["warehouse_backend"], "databend")

    def test_csrf_exempt(self):
        from django.test import Client
        c = Client(enforce_csrf_checks=True)
        r = c.post(TRANSLATE_URL, data=json.dumps({"sql": "select 1"}), content_type="application/json",
                   HTTP_X_AFYA_TOKEN="s3cret")
        self.assertEqual(r.status_code, 200)


class AsyncSourceTests(_Base):
    def test_settings_source_works_inside_event_loop(self):
        """Channels consumers read the switch from async code: the ORM call
        must run on a worker thread, not raise SynchronousOnlyOperation."""
        import asyncio
        import threading
        seen = {}

        def fake_get(key, default=None):
            seen["thread"] = threading.current_thread()
            return "ollama"

        async def read():
            seen["loop_thread"] = threading.current_thread()
            return backends._settings_source(toggles.LLM)
        with mock.patch.object(SystemSettings, "get", side_effect=fake_get):
            val = asyncio.run(read())
        self.assertEqual(val, "ollama")
        self.assertIsNot(seen["thread"], seen["loop_thread"])

    def test_settings_source_sync_path(self):
        SystemSettings.set("llm_provider", "ollama")
        self.assertEqual(backends._settings_source(toggles.LLM), "ollama")
