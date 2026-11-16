"""Connect Studio JSON API — real Django stack; background jobs run inline; Groq off."""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from connect_studio import crypto
from connect_studio.engine import llm
from connect_studio.models import Connection, Job, Mapping, Source

from .fixtures import make_hmis, target_json

User = get_user_model()
TMP = tempfile.mkdtemp()


class _InlineThread:
    def __init__(self, target, **_kw):
        self.target = target

    def start(self):
        self.target()


@override_settings(MEDIA_ROOT=TMP)
class ApiTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser("csadmin", "a@example.com", "pw")
        self.client.force_login(self.admin)
        patches = [
            mock.patch("connect_studio.jobs._Thread", _InlineThread),
            mock.patch("connect_studio.jobs.db_connection"),
            mock.patch("connect_studio.jobs.close_old_connections"),
            mock.patch.object(llm, "_client", side_effect=llm.LLMUnavailable("off in tests")),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.tmp = Path(tempfile.mkdtemp())

    def api(self, method, path, body=None, **kw):
        fn = getattr(self.client, method.lower())
        if body is not None:
            return fn(f"/connect/api/{path}", data=json.dumps(body), content_type="application/json", **kw)
        return fn(f"/connect/api/{path}", **kw)

    def make_source(self, kind="sqlite", **extra):
        r = self.api("POST", "sources/", {"kind": kind, "name": f"{kind} src", **extra})
        self.assertEqual(r.status_code, 201, r.content)
        return r.json()["source"]["id"]

    def upload(self, sid, name, content):
        r = self.client.post(f"/connect/api/sources/{sid}/files/", {"files": SimpleUploadedFile(name, content)})
        self.assertEqual(r.status_code, 201, r.content)

    def test_permissions(self):
        self.client.logout()
        self.assertEqual(self.api("GET", "meta/").status_code, 401)
        User.objects.create_user("plain", "p@example.com", "pw")
        self.client.login(username="plain", password="pw")
        self.assertEqual(self.api("GET", "meta/").status_code, 403)
        self.assertEqual(self.client.get("/connect/").status_code, 403)

    def test_secrets_never_returned(self):
        sid = self.make_source("postgres", config={"host": "db", "database": "x", "username": "u", "password": "s3cret"})
        body = self.api("GET", f"sources/{sid}/").content.decode()
        self.assertNotIn("s3cret", body)
        self.assertIn("password", self.api("GET", f"sources/{sid}/").json()["source"]["secrets_set"])
        self.assertEqual(crypto.decrypt(Source.objects.get(pk=sid).secrets)["password"], "s3cret")
        # blank secret on update keeps the stored one
        self.api("PATCH", f"sources/{sid}/", {"config": {"host": "db2", "database": "x", "username": "u", "password": ""}})
        self.assertEqual(crypto.decrypt(Source.objects.get(pk=sid).secrets)["password"], "s3cret")

    def test_end_to_end(self):
        # source: upload sqlite, test, extract
        src = self.make_source("sqlite")
        self.upload(src, "hmis.sqlite", make_hmis(self.tmp / "h.sqlite").read_bytes())
        self.assertEqual(self.api("POST", f"sources/{src}/test/").status_code, 202)
        self.assertEqual(Source.objects.get(pk=src).status, "ok")
        r = self.api("POST", f"sources/{src}/extract/", {"sample_rows": 100})
        job = Job.objects.get(pk=r.json()["job"]["id"])
        self.assertEqual(job.status, "succeeded", job.error)
        self.assertIn("Snapshot #", job.message)
        # target: upload schema.json
        tgt = self.make_source("schema_file")
        self.upload(tgt, "v3.json", target_json().encode())
        self.assertEqual(self.api("POST", f"sources/{tgt}/extract/").status_code, 202)
        # mapping (rule-based: Groq is off)
        r = self.api("POST", "mappings/", {"source": src, "target": tgt, "ai": True})
        self.assertEqual(r.status_code, 201, r.content)
        mid = r.json()["mapping"]["id"]
        m = Mapping.objects.get(pk=mid)
        self.assertEqual(m.status, "review")
        self.assertGreater(m.summary["mapped"], 0)
        detail = self.api("GET", f"mappings/{mid}/?snapshots=1").json()["mapping"]
        self.assertIsNotNone(detail["source_schema"])
        # review: create a split, a code map; reject one; delete one
        r = self.api("PATCH", f"mappings/{mid}/fields/", {"create": [
            {"target_table": "patients", "target": "first_name", "source": "othernames", "kind": "one column divided", "rule": {"part": 1}},
            {"target_table": "visits", "target": "department_name", "source": "department", "kind": "looked up in another table",
             "rule": {"table": "departments", "key": "id", "value": "name"}}]})
        self.assertEqual(r.status_code, 200, r.content)
        # gender ← sex was already matched by the rule pass (synonyms): creating it again is a conflict
        dup = self.api("PATCH", f"mappings/{mid}/fields/", {"create": [{"target_table": "patients", "target": "gender", "source": "sex"}]})
        self.assertEqual(dup.status_code, 409)
        bad = self.api("PATCH", f"mappings/{mid}/fields/", {"updates": [{"id": "patients::gender", "source": "nope"}]})
        self.assertEqual(bad.status_code, 400)
        r = self.api("PATCH", f"mappings/{mid}/fields/", {"updates": [
            {"id": "patients::gender", "source": "sex", "kind": "codes converted", "rule": {"map": {"M": "male", "F": "female"}}}]})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["fields"][0]["made_by"], "human")
        fid = "patients::last_name"
        self.assertEqual(self.api("PATCH", f"mappings/{mid}/fields/", {"updates": [{"id": fid, "status": "rejected"}]}).status_code, 200)
        self.assertEqual(self.api("PATCH", f"mappings/{mid}/fields/", {"updates": [{"id": fid, "status": "approved"}]}).status_code, 200)
        r = self.api("POST", f"mappings/{mid}/bulk/", {"action": "approve", "min_confidence": 0})
        self.assertGreater(r.json()["summary"]["approved"], 0)
        # preview + export
        prev = self.api("POST", f"mappings/{mid}/preview/", {"source_table": "patients", "source": "othernames",
                                                             "kind": "one column divided", "rule": {"part": 1}}).json()
        self.assertIn({"in": "Stephen Ombaka", "out": "Stephen"}, prev["preview"])
        lk = self.api("POST", f"mappings/{mid}/preview/", {"source_table": "visits", "source": "department",
                                                           "kind": "looked up in another table", "rule": {"table": "departments", "key": "id", "value": "name"}}).json()
        self.assertIn("OPD", [x["out"] for x in lk["preview"]])
        exp = self.api("GET", f"mappings/{mid}/export/?flavor=afya").json()
        self.assertEqual(exp["problems"], [])
        dl = self.api("GET", f"mappings/{mid}/export/?flavor=generic&download=1")
        self.assertIn("attachment", dl["Content-Disposition"])
        # sync into DuckDB
        dest = self.make_source("duckdb", config={"path": str(self.tmp / "dest.duckdb")})
        r = self.api("POST", "connections/", {"mapping": mid, "destination": dest, "mode": "full_refresh_overwrite",
                                              "schedule_minutes": 60, "run_now": True})
        self.assertEqual(r.status_code, 201, r.content)
        c = Connection.objects.get(pk=r.json()["connection"]["id"])
        self.assertEqual(c.last_status, "succeeded")
        self.assertEqual(c.last_rows, 6)
        self.assertIsNotNone(c.next_run_at)
        # a schema-file source is not a sync destination
        self.assertEqual(self.api("POST", "connections/", {"mapping": mid, "destination": tgt}).status_code, 400)
        ov = self.api("GET", "overview/").json()
        self.assertEqual(ov["counts"]["connections"], 1)
        kinds = sorted(j["kind"] for j in self.api("GET", "jobs/").json()["jobs"])
        self.assertEqual(kinds, ["extract", "extract", "map", "sync", "test"], kinds)

    def test_shell_renders(self):
        r = self.client.get("/connect/mappings/123")
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'id="cs-root"')
