"""Regressions for bugs found by the 2026-11-17 end-to-end run on production."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import openpyxl
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings

from connect_studio import services
from connect_studio.engine import connectors, llm

from .fixtures import make_hmis

User = get_user_model()


class ServerDatabaseExtractionTests(SimpleTestCase):
    def test_generic_sqlalchemy_url_is_extracted_through_dbatlas(self):
        # every server database (postgres, mysql, redshift, any URL) goes through the "sqlalchemy" adapter,
        # which dbatlas does not register itself — on prod this failed with "No adapter registered"
        db = make_hmis(Path(tempfile.mkdtemp()) / "hmis.sqlite")
        c = connectors.create("sqlalchemy", {"url": f"sqlite:///{db}"}, workdir=Path(tempfile.mkdtemp()))
        snap = c.discover({"profile": False})
        self.assertEqual({t["name"] for t in snap["tables"]}, {"departments", "patients", "visits"})
        self.assertTrue(any(r["from"].endswith("visits") and r["to"].endswith("patients") for r in snap["relationships"]))


class FilesConnectorTests(SimpleTestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def conn(self, *files, **cfg):
        return connectors.create("files", cfg, files=list(files), workdir=Path(tempfile.mkdtemp()))

    def write(self, name, data):
        p = self.dir / name
        p.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
        return p

    def test_table_named_after_original_upload_and_bom_stripped(self):
        f = self.write("3fa1b2c4_patients.csv", "﻿id,phone,dob,active\n1,+254712345678,1990-01-02,true\n2,0700111222,1985-05-05,false\n")
        c = self.conn(f)
        self.assertEqual(c.streams(), ["patients"])
        rows = next(c.iter_rows("patients"))
        self.assertEqual(list(rows[0]), ["id", "phone", "dob", "active"])
        self.assertEqual(rows[0]["phone"], "+254712345678")   # not 254712345678.0
        self.assertEqual(rows[1]["phone"], "0700111222")      # leading zero kept
        self.assertEqual(rows[0]["id"], 1)
        self.assertIs(rows[1]["active"], False)
        snap = c.discover({})
        cols = {col["name"]: col["family"] for col in snap["tables"][0]["columns"]}
        self.assertEqual(cols["phone"], "text")
        self.assertEqual(cols["dob"], "date")

    def test_header_only_csv_keeps_columns(self):
        c = self.conn(self.write("a1b2c3d4_empty.csv", "id,name,amount\n"), self.write("x.csv", "id\n1\n"))
        snap = c.discover({})
        t = next(t for t in snap["tables"] if t["name"] == "empty")
        self.assertEqual([col["name"] for col in t["columns"]], ["id", "name", "amount"])

    def test_empty_sheet_skipped_and_unicode_name(self):
        p = self.dir / "Facturación ü.xlsx"
        wb = openpyxl.Workbook()
        wb.active.title = "Invoices"
        wb.active.append(["id", "amount"])
        wb.active.append([1, 10.5])
        wb.create_sheet("Notes")
        wb.save(p)
        self.assertEqual(self.conn(p).streams(), ["facturacion_u"])

    def test_same_file_name_twice(self):
        a = self.write("11111111_visits.csv", "id\n1\n")
        sub = self.dir / "b"
        sub.mkdir()
        b = sub / "22222222_visits.csv"
        b.write_text("id\n2\n")
        self.assertEqual(self.conn(a, b).streams(), ["visits", "visits_2"])

    def test_corrupt_file_fails_the_connection_test(self):
        with self.assertRaises(connectors.ConnectorError) as e:
            self.conn(self.write("bad.json", '{"data": [1, 2, ')).test()
        self.assertIn("bad", str(e.exception))

    def test_pasted_schema_counts_as_input(self):
        self.assertEqual(connectors.validate_config("schema_file", {"schema_text": "create table t (id int)"}), [])
        self.assertTrue(connectors.validate_config("schema_file", {}))


class MergeHumanDecisionsTests(SimpleTestCase):
    def test_human_fields_tables_and_filters_survive_regeneration(self):
        old = {"tables": [{"target": "client", "source": "people", "made_by": "human", "where": {"column": "x", "in": [1]}},
                          {"target": "bill", "source": "invoices", "made_by": "ai"}],
               "fields": [{"id": "client::sex", "target_table": "client", "target": "sex", "source_table": "people",
                           "source": "gender", "kind": "codes converted", "rule": {"map": {"M": "male"}}, "made_by": "human"},
                          {"id": "bill::no", "target_table": "bill", "target": "no", "source_table": "invoices",
                           "source": "gone_column", "kind": "copied", "rule": {}, "made_by": "human"}]}
        new = {"tables": [{"target": "client", "source": "patients", "made_by": "ai"}, {"target": "bill", "source": "invoices"}],
               "fields": [{"id": "client::sex", "target_table": "client", "target": "sex", "source_table": "patients",
                           "source": "sex", "kind": "copied", "made_by": "ai"}],
               "unmapped_fields": [{"target_table": "client", "target": "mrn", "reason": "x"}]}
        cols = {"people": {"gender"}, "patients": {"sex"}, "invoices": {"no"}}
        kept = services.merge_human_decisions(old, new, cols)
        client = next(p for p in new["tables"] if p["target"] == "client")
        self.assertEqual((client["source"], client["made_by"], client["where"]), ("people", "human", {"column": "x", "in": [1]}))
        f = {x["id"]: x for x in new["fields"]}
        self.assertEqual(f["client::sex"]["kind"], "codes converted")
        self.assertNotIn("bill::no", f)   # its source column no longer exists
        self.assertEqual(kept, 2)


class GroqJsonRetryTests(SimpleTestCase):
    def test_json_validate_failed_is_retried(self):
        class BadRequest(Exception):
            status_code = 400
            code = "json_validate_failed"

        calls = []

        class Completions:
            def create(self, **kw):
                calls.append(kw.get("temperature"))
                if len(calls) == 1:
                    raise BadRequest("Error code: 400 - Failed to generate JSON. Please adjust your prompt.")
                resp = mock.Mock()
                resp.choices = [mock.Mock(message=mock.Mock(content='{"ok": true}'))]
                resp.usage = None
                return resp

        client = mock.Mock()
        client.chat.completions = Completions()
        with mock.patch.object(llm, "_client", return_value=client), mock.patch.object(llm.time, "sleep"):
            self.assertEqual(llm.chat_json("s", "u"), {"ok": True})
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], 0.0)


class _InlineThread:
    def __init__(self, target, **_kw):
        self.target = target

    def start(self):
        self.target()


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class ApiHardeningTests(TestCase):
    def setUp(self):
        self.client.force_login(User.objects.create_superuser("csadmin", "a@example.com", "pw"))
        for p in (mock.patch("connect_studio.jobs._Thread", _InlineThread), mock.patch("connect_studio.jobs.db_connection"),
                  mock.patch("connect_studio.jobs.close_old_connections"),
                  mock.patch.object(llm, "_client", side_effect=llm.LLMUnavailable("off in tests"))):
            p.start()
            self.addCleanup(p.stop)

    def test_bad_query_values_are_400(self):
        for qs in ("limit=abc", "source=abc", "offset=-5"):
            r = self.client.get(f"/connect/api/jobs/?{qs}")
            self.assertEqual(r.status_code, 400, (qs, r.content))

    @override_settings(DATA_UPLOAD_MAX_MEMORY_SIZE=1000)
    def test_oversized_json_body_is_413(self):
        r = self.client.post("/connect/api/import-api/", data=json.dumps({"text": "x" * 5000}), content_type="application/json")
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.json()["error"]["code"], "too_large")

    def test_regenerate_that_matches_nothing_keeps_the_mapping(self):
        from connect_studio import crypto
        from connect_studio.models import Mapping, Source
        src = Source.objects.create(name="s", kind="schema_file", config={"schema_text": '{"tables":[{"name":"alpha","columns":[{"name":"a","type":"int"}]}]}'},
                                    secrets=crypto.encrypt({}), status="ok")
        tgt = Source.objects.create(name="t", kind="schema_file", config={"schema_text": '{"tables":[{"name":"zulu","columns":[{"name":"z","type":"int"}]}]}'},
                                    secrets=crypto.encrypt({}), status="ok")
        doc = {"tables": [{"target": "zulu", "source": "alpha", "made_by": "human"}],
               "fields": [{"id": "zulu::z", "target_table": "zulu", "target": "z", "source_table": "alpha", "source": "a",
                           "kind": "copied", "rule": {}, "transform": "", "confidence": 1, "made_by": "human", "status": "approved"}]}
        m = Mapping.objects.create(name="m", source=src, target=tgt, document=doc, settings={"ai": False}, status="review")
        r = self.client.post(f"/connect/api/mappings/{m.pk}/generate/", data="{}", content_type="application/json")
        self.assertEqual(r.status_code, 202)
        m.refresh_from_db()
        # the human table choice is kept, so the regenerated document still maps zulu ← alpha
        self.assertEqual(m.document["tables"][0]["source"], "alpha")
        self.assertTrue(any(f["id"] == "zulu::z" and f["made_by"] == "human" for f in m.document["fields"]))
