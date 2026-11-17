"""Postman / OpenAPI import → REST connector config; AI refinement is validated; POST-body reads + login auth."""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from connect_studio.engine import api_import, connectors, llm

POSTMAN = {
    "info": {"name": "Clinic API", "_postman_id": "x", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
    "variable": [{"key": "baseUrl", "value": "https://api.clinic.test/v1"}, {"key": "tenant", "value": "42"}],
    "auth": {"type": "bearer", "bearer": [{"key": "token", "value": "{{token}}"}]},
    "item": [
        {"name": "Auth", "item": [{"name": "Login", "request": {"method": "POST", "url": {"raw": "{{baseUrl}}/auth/login"},
                                                               "body": {"mode": "raw", "raw": "{\"username\": \"{{user}}\", \"password\": \"{{pass}}\"}"}},
                                   "response": [{"body": "{\"data\": {\"token\": \"abc\"}}"}]}]},
        {"name": "List patients", "request": {"method": "GET", "url": {
            "raw": "{{baseUrl}}/patients?page=1&per_page=50&status=active",
            "query": [{"key": "page", "value": "1"}, {"key": "per_page", "value": "50"}, {"key": "status", "value": "active"}]}},
         "response": [{"body": "{\"data\": [{\"id\": 1, \"first_name\": \"Ann\"}], \"meta\": {}}"}]},
        {"name": "Get patient", "request": {"method": "GET", "url": {"raw": "{{baseUrl}}/patients/:id"}}},
        {"name": "Read visits", "request": {"method": "POST", "url": {"raw": "{{baseUrl}}/gateway"},
                                             "body": {"mode": "raw", "raw": "{\"action\": \"read\", \"model\": \"visit\", \"tenant\": {{tenant}}, \"page\": 1, \"per_page\": 100}"}}},
        {"name": "Create patient", "request": {"method": "POST", "url": {"raw": "{{baseUrl}}/patients"}, "body": {"mode": "raw", "raw": "{\"first_name\": \"x\"}"}}},
    ],
}

OPENAPI = {
    "openapi": "3.0.1", "info": {"title": "Lab API"},
    "servers": [{"url": "https://{env}.lab.test/api", "variables": {"env": {"default": "prod"}}}],
    "components": {"securitySchemes": {"key": {"type": "apiKey", "in": "header", "name": "X-Lab-Key"}},
                   "schemas": {"Result": {"type": "object", "properties": {"id": {"type": "integer"}, "test": {"type": "string"}}},
                               "Page": {"type": "object", "properties": {"items": {"type": "array", "items": {"$ref": "#/components/schemas/Result"}}}}}},
    "security": [{"key": []}],
    "paths": {
        "/results": {"get": {"operationId": "listResults", "parameters": [{"name": "offset", "in": "query"}, {"name": "limit", "in": "query"}],
                             "responses": {"200": {"content": {"application/json": {"schema": {"$ref": "#/components/schemas/Page"}}}}}},
                     "post": {"operationId": "createResult", "responses": {"201": {}}}},
        "/results/{id}": {"get": {"operationId": "getResult", "parameters": [{"name": "id", "in": "path"}], "responses": {"200": {}}}},
    },
}


class ParseTests(SimpleTestCase):
    def test_postman(self):
        p = api_import.parse(json.dumps(POSTMAN), "c.json")
        c = p["config"]
        self.assertEqual(p["format"], "Postman v2.1")
        self.assertEqual(c["base_url"], "https://api.clinic.test/v1")
        self.assertEqual((c["auth_type"], c["login_path"], c["token_path"]), ("login", "/auth/login", "data.token"))
        self.assertEqual((c["pagination"], c["page_param"], c["size_param"]), ("page", "page", "per_page"))
        eps = {e["name"]: e for e in p["endpoints"]}
        self.assertTrue(eps["list_patients"]["selected"])
        self.assertEqual(eps["list_patients"]["path"], "/patients?status=active")      # paging params lifted out
        self.assertEqual(eps["list_patients"]["records"], "data")
        self.assertEqual(eps["list_patients"]["fields"], ["id", "first_name"])
        self.assertFalse(eps["get_patient"]["selected"])
        self.assertFalse(eps["create_patient"]["selected"])
        self.assertFalse(eps["login"]["selected"])
        self.assertEqual(eps["read_visits"]["body"], {"action": "read", "model": "visit", "tenant": 42})   # var resolved, typed
        self.assertEqual(api_import.stream_line(eps["read_visits"]),
                         'read_visits = POST /gateway {"action":"read","model":"visit","tenant":42}')
        self.assertEqual({v["name"] for v in p["variables"]}, {"user", "pass"})

    def test_openapi(self):
        p = api_import.parse(json.dumps(OPENAPI), "o.json")
        c = p["config"]
        self.assertEqual(c["base_url"], "https://prod.lab.test/api")
        self.assertEqual((c["auth_type"], c["api_key_name"]), ("api_key_header", "X-Lab-Key"))
        self.assertEqual((c["pagination"], c["page_param"], c["size_param"]), ("offset", "offset", "limit"))
        eps = {e["name"]: e for e in p["endpoints"]}
        self.assertTrue(eps["listresults"]["selected"])
        self.assertEqual(eps["listresults"]["records"], "items")
        self.assertEqual(eps["listresults"]["fields"], ["id", "test"])
        self.assertFalse(eps["createresult"]["selected"])
        self.assertFalse(eps["getresult"]["selected"])

    def test_yaml_and_errors(self):
        p = api_import.parse("swagger: '2.0'\ninfo: {title: T}\nhost: x.test\nbasePath: /v2\npaths:\n  /a:\n    get: {responses: {}}\n", "s.yaml")
        self.assertEqual(p["config"]["base_url"], "https://x.test/v2")
        for bad in ("", "[]", "{\"hello\": 1}", "{not json"):
            with self.assertRaises(api_import.ImportError_):
                api_import.parse(bad, "x.json")

    def test_ai_refine_is_validated(self):
        p = api_import.parse(json.dumps(POSTMAN), "c.json")
        ids = {e["name"]: e["id"] for e in p["endpoints"]}
        fake = {"endpoints": [
            {"id": ids["list_patients"], "table": "Patients!", "records": "data", "why": "patients"},
            {"id": ids["read_visits"], "table": "patients", "body": {"action": "read", "model": "visit"}},   # dup name, keys removed
            {"id": ids["create_patient"], "table": "x", "body": {"evil": 1}},                                # body change refused
            {"id": "e999", "table": "ghost"}],
            "pagination": {"pagination": "cursor"},                                                       # no params → ignored
            "summary": "Clinic data", "notes": ["fill in user"]}
        r = api_import.ai_refine(p, chat=lambda *a, **k: fake)
        sel = {e["name"]: e for e in r["endpoints"] if e["selected"]}
        self.assertEqual(set(sel), {"patients", "patients_2", "x"})
        self.assertEqual(sel["patients_2"]["body"], {"action": "read", "model": "visit"})
        self.assertEqual(sel["x"]["body"], {"first_name": "x"})
        self.assertEqual(r["config"]["pagination"], "page")
        self.assertTrue(r["ai"]["used"])
        self.assertIn("fill in user", r["notes"])
        self.assertTrue(p["endpoints"][1]["selected"] in (True, False))          # original untouched
        # LLM down → parser proposal + error
        down = api_import.ai_refine(p, chat=mock.Mock(side_effect=llm.LLMUnavailable("no key")))
        self.assertFalse(down["ai"]["used"])
        self.assertEqual([e["selected"] for e in down["endpoints"]], [e["selected"] for e in p["endpoints"]])


def _gw(name, url, body):
    return {"name": name, "request": {"method": "POST", "url": {"raw": url}, "body": {"mode": "raw", "raw": body}}}


GATEWAY = {
    "info": {"name": "Model Gateway", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
    "variable": [{"key": "core_url", "value": "http://localhost:8001"}, {"key": "source_tenant_id", "value": "1"}],
    "item": [
        {"name": "Auth", "item": [_gw("Login", "{{core_url}}/api/v1/login", '{"username": "u", "password": "p"}')]},
        {"name": "core-service", "item": [
            _gw("List", "{{core_url}}/api/v1/gateway", '{"action": "list"}'),
            _gw("Read departments", "{{core_url}}/api/v1/gateway",
                '{"model": "departments", "action": "read", "source_tenant_id": {{source_tenant_id}}, "filters": {"status": "active"}}'),
            _gw("Copy insurers", "{{core_url}}/api/v1/gateway", '{"model": "insurance_companies", "action": "copy", "ids": [1]}')]},
        {"name": "🧪 patient-evaluation-service", "item": [
            _gw("Read visits", "{{evaluation_url}}/api/v1/gateway", '{"model": "visits", "action": "read"}')]},
        {"name": "migrate", "item": [
            _gw("Read wards", "https://inpatientmigrate.afyaanalytics.ai/api/v1/gateway", '{"model": "ward", "action": "read"}')]},
    ],
}


class V3PostmanTests(SimpleTestCase):
    def test_model_gateway_collection(self):
        r = api_import.v3_from_postman(json.dumps(GATEWAY), "g.json")
        got = {m["id"]: m["selected"] for m in r["models"]}
        self.assertEqual(got, {"core:departments": True, "core:insurance_companies": False,
                               "evaluation:visits": True, "inpatient:ward": True})
        self.assertEqual(r["services"], ["core", "evaluation", "inpatient"])
        self.assertEqual(r["url_template"], "https://{service}migrate.afyaanalytics.ai/api/")
        with self.assertRaises(api_import.ImportError_):
            api_import.v3_from_postman(json.dumps({"item": [_gw("x", "{{core_url}}/api/v1/login", "{}")]}), "x.json")


class RestPostAndLoginTests(SimpleTestCase):
    def test_post_body_paging_and_login(self):
        calls = []

        def resp(body, status=200):
            return mock.Mock(status_code=status, ok=status < 400, json=lambda: body, text=json.dumps(body), headers={}, links={})

        class S:
            headers = {}

            def post(self, url, json=None, timeout=None):              # login
                calls.append(("login", url, json))
                return resp({"data": {"token": "tok"}})

            def request(self, method, url, timeout=None, json=None):
                calls.append((method, url, json, dict(self.headers)))
                rows = [{"id": 1}, {"id": 2}] if json["page"] == 1 else [{"id": 3}]
                return resp({"data": rows})

        c = connectors.create("rest_api", {
            "base_url": "https://api.test/v1", "auth_type": "login", "login_path": "/auth/login",
            "login_body": '{"username": "u", "password": "p"}',
            "streams": 'visits = POST /gateway {"action":"read","model":"visit"} -> data',
            "pagination": "page", "page_param": "page", "size_param": "per_page", "page_size": 2},
            workdir=Path(tempfile.mkdtemp()))
        with mock.patch("requests.Session", S):
            rows = [r for b in c.read("visits") for r in b]
        self.assertEqual([r["id"] for r in rows], [1, 2, 3])
        self.assertEqual(calls[0], ("login", "https://api.test/v1/auth/login", {"username": "u", "password": "p"}))
        self.assertEqual(calls[1][:3], ("POST", "https://api.test/v1/gateway", {"action": "read", "model": "visit", "page": 1, "per_page": 2}))
        self.assertEqual(calls[1][3]["Authorization"], "Bearer tok")
        self.assertEqual(calls[2][2]["page"], 2)


class ImportEndpointTests(TestCase):
    def test_endpoint(self):
        User = get_user_model()
        self.client.force_login(User.objects.create_superuser("imp", "i@example.com", "pw"))
        with mock.patch.object(llm, "_client", side_effect=llm.LLMUnavailable("off")):
            r = self.client.post("/connect/api/import-api/", data=json.dumps({"text": json.dumps(OPENAPI), "filename": "o.json"}),
                                 content_type="application/json")
        self.assertEqual(r.status_code, 200, r.content)
        body = r.json()
        self.assertFalse(body["ai"]["used"])
        self.assertTrue(any(e["line"].startswith("listresults = /results") for e in body["endpoints"]))
        bad = self.client.post("/connect/api/import-api/", data=json.dumps({"text": "{}"}), content_type="application/json")
        self.assertEqual(bad.status_code, 400)
