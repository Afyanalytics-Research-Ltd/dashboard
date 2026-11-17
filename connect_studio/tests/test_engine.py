"""Engine tests — connectors, snapshots, rules, mapper (Groq mocked), export, sync. No network."""

import json
import tempfile
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

from connect_studio.engine import connectors, document, llm, mapper, rules, snapshot, sync

from .fixtures import make_hmis, target_json


class SnapshotParsingTests(SimpleTestCase):
    def test_tables_list_with_fk_enum_and_inference(self):
        snap = snapshot.parse_schema_document(target_json(), "schema.json")
        self.assertEqual(snap["stats"]["tables"], 2)
        visits = snapshot.table_index(snap)["visits"]
        self.assertEqual(visits["primary_key"], ["id"])
        status = snapshot.column_index(visits)["status"]
        self.assertEqual(status["allowed_values"], ["waiting", "completed"])
        self.assertEqual(status["family"], "text")
        rel = snap["relationships"][0]
        self.assertEqual((rel["from"], rel["to"], rel["kind"]), ("visits", "patients", "foreign_key"))

    def test_table_map_and_inferred_relationship(self):
        doc = {"patients": {"id": "int", "name": "text"}, "visits": {"id": "int", "patient_id": "int"}}
        snap = snapshot.parse_schema_document(json.dumps(doc), "s.json")
        self.assertEqual({t["key"] for t in snap["tables"]}, {"patients", "visits"})
        inferred = [r for r in snap["relationships"] if r["kind"] == "inferred"]
        self.assertTrue(any(r["from"] == "visits" and r["to"] == "patients" for r in inferred), snap["relationships"])

    def test_json_schema(self):
        doc = {"$defs": {"patient": {"type": "object", "required": ["id"], "properties": {
            "id": {"type": "integer"}, "dob": {"type": ["string", "null"], "format": "date"},
            "sex": {"type": "string", "enum": ["M", "F"]}}}}}
        snap = snapshot.parse_schema_document(json.dumps(doc), "s.json")
        cols = snapshot.column_index(snap["tables"][0])
        self.assertEqual(cols["id"]["family"], "integer")
        self.assertFalse(cols["id"]["nullable"])
        self.assertEqual(cols["dob"]["family"], "date")
        self.assertEqual(cols["sex"]["allowed_values"], ["M", "F"])

    def test_sql_ddl(self):
        ddl = """CREATE TABLE wards (id INT PRIMARY KEY, name VARCHAR(50) NOT NULL, kind ENUM('male','female'));
                 CREATE TABLE beds (id INT, ward_id INT, PRIMARY KEY (id), FOREIGN KEY (ward_id) REFERENCES wards(id));"""
        snap = snapshot.parse_schema_document(ddl, "schema.sql")
        wards = snapshot.table_index(snap)["wards"]
        self.assertEqual(wards["primary_key"], ["id"])
        self.assertFalse(snapshot.column_index(wards)["name"]["nullable"])
        self.assertEqual(snapshot.column_index(wards)["kind"]["allowed_values"], ["male", "female"])
        self.assertIn(("beds", "wards"), {(r["from"], r["to"]) for r in snap["relationships"]})

    def test_mappings_json_target_side(self):
        doc = {"connection": {"id": 1}, "tables": [{"v3 model": "App\\Models\\X", "v3 table": "xs", "source table": "a"}],
               "fields": [{"model": "App\\Models\\X", "v3_table": "xs", "table": "a", "field": "id"},
                          {"model": "App\\Models\\X", "v3_table": "xs", "table": "a", "field": "name"}]}
        snap = snapshot.parse_schema_document(json.dumps(doc), "m.json")
        t = snapshot.table_index(snap)["xs"]
        self.assertEqual([c["name"] for c in t["columns"]], ["id", "name"])
        self.assertEqual(t["model"], "App\\Models\\X")

    def test_bad_documents(self):
        with self.assertRaises(snapshot.SchemaDocumentError):
            snapshot.parse_schema_document("{nope", "s.json")
        with self.assertRaises(snapshot.SchemaDocumentError):
            snapshot.parse_schema_document("", "s.json")

    def test_diff(self):
        a = snapshot.parse_schema_document(json.dumps({"t": {"id": "int", "x": "text"}}), "a.json")
        b = snapshot.parse_schema_document(json.dumps({"t": {"id": "bigint", "y": "text"}, "u": {"id": "int"}}), "b.json")
        d = snapshot.diff(a, b)
        self.assertEqual(d["added_tables"], ["u"])
        ch = d["changed_tables"][0]
        self.assertEqual((ch["added_columns"], ch["removed_columns"]), (["y"], ["x"]))
        self.assertEqual(ch["type_changes"][0]["column"], "id")


class RuleTests(SimpleTestCase):
    def test_apply_every_kind(self):
        A = rules.apply
        self.assertEqual(A({"kind": "copied", "rule": {}, "transform": "uppercase"}, "abc"), "ABC")
        self.assertEqual(A({"kind": "worked out", "rule": {"steps": [{"op": "trim"}, {"op": "slug"}]}}, "  Hello World "), "hello-world")
        f = {"kind": "codes converted", "rule": {"map": {"M": "male", "F": "female"}, "default": "unknown"}}
        self.assertEqual((A(f, "m"), A(f, "F"), A(f, "X")), ("male", "female", "unknown"))
        self.assertEqual(A({"kind": "one column divided", "rule": {"part": 1, "delimiter": " "}}, "Ali Hassan Omar"), "Ali")
        self.assertEqual(A({"kind": "one column divided", "rule": {"part": "rest", "delimiter": " "}}, "Ali Hassan Omar"), "Hassan Omar")
        lk = {"kind": "looked up in another table", "rule": {"table": "d", "key": "id", "value": "name"}}
        self.assertEqual(A(lk, 1, {("d", "id", "name"): {"1": "OPD"}}), "OPD")
        self.assertEqual(A({"kind": "constant", "rule": {"value": 7}}, None), 7)

    def test_validate(self):
        tables = {"p": {"columns": [{"name": "id"}, {"name": "sex"}]}, "d": {"columns": [{"name": "id"}, {"name": "name"}]}}
        ok = {"kind": "looked up in another table", "source_table": "p", "source": "id", "rule": {"table": "d", "key": "id", "value": "name"}}
        self.assertIsNone(rules.validate(ok, tables))
        self.assertIn("does not exist", rules.validate({**ok, "source": "nope"}, tables))
        self.assertIn("lookup", rules.validate({**ok, "rule": {"table": "zzz", "key": "id", "value": "x"}}, tables))
        self.assertIn("map", rules.validate({"kind": "codes converted", "source_table": "p", "source": "sex", "rule": {}}, tables))

    def test_normalize_loader_spellings(self):
        f = rules.normalize({"kind": "looked up in another table", "rule": json.dumps({"lookup": {"table": "d", "match": "id", "value": "name"}, "column": "x"})})
        self.assertEqual(f["rule"], {"table": "d", "key": "id", "value": "name"})


class _Env(SimpleTestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = make_hmis(self.tmp / "hmis.sqlite")
        self.src = connectors.create("sqlite", {}, files=[self.db], workdir=self.tmp / "w")
        self.s_snap = self.src.discover({"sample_rows": 100})
        self.t_snap = snapshot.parse_schema_document(target_json(), "schema.json")

    def tearDown(self):
        self.src.close()


class ConnectorTests(_Env):
    def test_sqlite_discover_profiles_and_fk(self):
        s = self.s_snap["stats"]
        self.assertEqual(s["tables"], 3)
        self.assertTrue(s["profiled"])
        sex = snapshot.column_index(snapshot.table_index(self.s_snap)["patients"])["sex"]
        self.assertEqual({t["v"] for t in sex["profile"]["top"]}, {"M", "F"})
        self.assertIn(("visits", "patients"), {(r["from"], r["to"]) for r in self.s_snap["relationships"]})
        self.assertTrue(self.src.test()["ok"])

    def test_files_connector(self):
        csv = self.tmp / "people.csv"
        csv.write_text("person_id,full_name\n1,Ann Lee\n2,Bob Ray\n")
        js = self.tmp / "orders.json"
        js.write_text(json.dumps({"data": [{"id": 1, "person_id": 1, "meta": {"channel": "web"}}]}))
        c = connectors.create("files", {}, files=[csv, js], workdir=self.tmp / "f")
        snap = c.discover({})
        idx = snapshot.table_index(snap)
        self.assertEqual(set(idx), {"people", "orders"})
        self.assertIn("meta__channel", snapshot.column_index(idx["orders"]))
        rows = [r for b in c.read("people") for r in b]
        self.assertEqual(rows[0]["full_name"], "Ann Lee")

    def test_rest_connector_pagination(self):
        pages = {1: {"data": [{"id": 1}, {"id": 2}]}, 2: {"data": [{"id": 3}]}}
        c = connectors.create("rest_api", {"base_url": "https://api.example.com", "streams": "items = /items -> data",
                                           "pagination": "page", "page_size": 2}, workdir=self.tmp / "r")

        def fake_get(session, url):
            page = int(url.split("page=")[1].split("&")[0])
            return mock.Mock(links={}), pages[page]

        with mock.patch.object(connectors.RestConnector, "_get", side_effect=fake_get):
            rows = [r for b in c.read("items") for r in b]
        self.assertEqual([r["id"] for r in rows], [1, 2, 3])

    def test_afya_platform_v3(self):
        """Two-step login, model discovery with alias clashes, tenant-scoped paged reads."""
        calls = []

        def resp(body, status=200):
            return mock.Mock(status_code=status, ok=status < 400, json=lambda: body, text=json.dumps(body), headers={})

        def post(url, json=None, headers=None, timeout=None):
            calls.append((url, json))
            if url.endswith("/v1/login"):
                if "facility_id" not in json:
                    return resp({"step": 2, "facilities": [{"id": 5}, {"id": 6}]})
                return resp({"access_token": "t", "tenant": {"id": 42}, "facility": {"id": json["facility_id"]}})
            svc = url.split("//")[1].split(".")[0]
            if json["action"] == "list":
                models = {"reception": [{"alias": "patient", "operations": ["read"]}],
                          "finance": [{"alias": "patient"}, {"alias": "payment"}, {"alias": "x", "operations": ["insert"]}]}
                return resp({"data": models.get(svc, [])})
            data = [{"id": i} for i in range(1, 3)] if json["page"] == 1 else [{"id": 3}]
            return resp({"data": data, "meta": {"last_page": 2}})

        c = connectors.create("afya_platform", {"username": "u", "password": "p", "facility_id": 6,
                                                "services": "reception, finance"}, workdir=self.tmp / "v3")
        c._session = lambda: mock.Mock(post=post)
        self.assertIn("organization 42, facility 6", c.test()["message"])
        self.assertEqual(sorted(c.streams()), ["finance__patient", "payment", "reception__patient"])
        c.PER_PAGE = 2
        rows = [r for b in c.read("payment") for r in b]
        self.assertEqual([r["id"] for r in rows], [1, 2, 3])
        reads = [b for u, b in calls if b and b.get("action") == "read"]
        self.assertTrue(all(b["source_tenant_id"] == 42 for b in reads))
        self.assertTrue(all("//finance." in u for u, b in calls if b and b.get("action") == "read"))
        with self.assertRaises(connectors.ConnectorError):
            connectors.create("afya_platform", {"username": "u", "password": "p", "models": "patient"}).streams()

    def test_afya_platform_v3_structure_from_describe(self):
        """Structure only: columns/types/PK/FK/enums come from describe, nothing is read."""
        def resp(body):
            return mock.Mock(status_code=200, ok=True, json=lambda: body, text="", headers={})

        desc = {
            "patient": {"class": r"App\Models\Patient", "table": "patients", "primary_key": ["id"], "columns": [
                {"name": "id", "type": "bigint unsigned", "nullable": False, "primary": True},
                {"name": "gender", "type": "enum('male','female')", "nullable": True, "comment": "sex at birth"}]},
            "visit": {"class": r"App\Models\Visit", "table": "visits", "primary_key": ["id"], "columns": [
                {"name": "id", "type": "bigint unsigned", "nullable": False, "primary": True},
                {"name": "patient_id", "type": "bigint unsigned", "nullable": False, "references": "patients.id"}]},
        }
        reads = []

        def post(url, json=None, headers=None, timeout=None):
            if url.endswith("/v1/login"):
                return resp({"access_token": "t", "tenant": {"id": 7}})
            if json["action"] == "read":
                reads.append(json)
            return resp({"data": desc[json["model"]]})

        c = connectors.create("afya_platform", {"username": "u", "password": "p", "sample_data": False,
                                                "models": "reception:patient, reception:visit"}, workdir=self.tmp / "v3d")
        c._session = lambda: mock.Mock(post=post)
        snap = c.discover({})
        self.assertEqual(reads, [])
        idx = snapshot.table_index(snap)
        self.assertEqual((idx["patient"]["name"], idx["patient"]["model"], idx["patient"]["service"]),
                         ("patients", r"App\Models\Patient", "reception"))
        gender = snapshot.column_index(idx["patient"])["gender"]
        self.assertEqual(gender["allowed_values"], ["male", "female"])
        self.assertEqual(gender["comment"], "sex at birth")
        self.assertIn({"from": "visit", "from_columns": ["patient_id"], "to": "patient", "to_columns": ["id"],
                       "confidence": 1.0, "kind": "foreign_key"}, snap["relationships"])

    def test_catalog_and_validation(self):
        keys = {c["key"] for c in connectors.catalog()}
        self.assertTrue({"postgres", "mysql", "snowflake", "files", "rest_api", "afya_extraction", "schema_file"} <= keys)
        self.assertIn("Host is required.", connectors.validate_config("postgres", {}))
        self.assertEqual(connectors.validate_config("postgres", {"host": "h", "database": "d", "username": "u"}), [])


class MapperTests(_Env):
    def test_rule_based_only(self):
        with mock.patch.object(llm, "_client", side_effect=llm.LLMUnavailable("off")):
            doc = mapper.generate(self.s_snap, self.t_snap, use_ai=True)
        self.assertFalse(doc["settings"]["ai"])
        pairs = {p["target"]: p["source"] for p in doc["tables"]}
        self.assertEqual(pairs, {"patients": "patients", "visits": "visits"})
        f = {(x["target_table"], x["target"]): x for x in doc["fields"]}
        self.assertEqual(f[("patients", "date_of_birth")]["source"], "dob")  # synonym
        self.assertEqual(f[("patients", "last_name")]["source"], "surname")

    def test_ai_suggestions_are_validated(self):
        def fake_chat(system, user, **kw):
            if "TARGET tables" in system:
                return {"matches": [{"target": "patients", "source": "patients", "confidence": 0.9, "reason": "same"},
                                    {"target": "visits", "source": "visits", "confidence": 0.9, "reason": "same"}]}
            if "TARGET TABLE patients" in user:
                return {"fields": [
                    {"target": "first_name", "source": "othernames", "kind": "one column divided", "rule": {"part": 1, "delimiter": " "}, "confidence": 0.9},
                    {"target": "gender", "source": "sex", "kind": "codes converted", "rule": {"map": {"M": "male", "F": "female"}}, "confidence": 0.95},
                    {"target": "last_name", "source": "no_such_column", "kind": "copied", "confidence": 0.9}]}
            return {"fields": [{"target": "department_name", "source": "department", "kind": "looked up in another table",
                                "rule": {"table": "departments", "key": "id", "value": "name"}, "confidence": 0.9}]}

        with mock.patch.object(llm, "_client", return_value=object()), mock.patch.object(llm, "chat_json", side_effect=fake_chat):
            doc = mapper.generate(self.s_snap, self.t_snap, use_ai=True)
        f = {(x["target_table"], x["target"]): x for x in doc["fields"]}
        self.assertEqual(f[("patients", "first_name")]["kind"], "one column divided")
        self.assertEqual(f[("patients", "gender")]["rule"]["map"]["M"], "male")
        self.assertEqual(f[("patients", "last_name")]["source"], "surname")  # invalid AI pick ignored, rule match kept
        self.assertEqual(f[("visits", "department_name")]["rule"]["table"], "departments")
        return doc


class ExportAndSyncTests(_Env):
    def _doc(self):
        doc = MapperTests.test_ai_suggestions_are_validated(self)
        doc["fields"].append({"id": "visits::status", "target_table": "visits", "target": "status", "source_table": "visits",
                              "source": "status", "kind": "codes converted", "rule": {"map": {"served": "completed", "new": "waiting"}},
                              "transform": "", "confidence": 1, "why": "", "made_by": "human", "status": "approved"})
        doc["fields"] = [f for f in doc["fields"] if not (f["target"] == "status" and f["made_by"] != "human")]
        return doc

    def test_export_flavors_validate(self):
        doc = self._doc()
        meta = {"id": 3, "name": "hmis", "kind": "sqlite"}
        afya = document.export(doc, source_meta=meta, target_meta=meta, target=self.t_snap, source=self.s_snap, flavor="afya", connection_id=59)
        self.assertEqual(document.validate_export(afya, "afya"), [])
        self.assertEqual(afya["connection"]["id"], 59)
        self.assertTrue(all("v3 model" in t for t in afya["tables"]))
        split = next(x for x in afya["fields"] if x["field"] == "first_name")
        self.assertEqual(json.loads(split["rule"]), {"part": 1, "delimiter": " ", "column": "othernames"})
        generic = document.export(doc, source_meta=meta, target_meta=meta, target=self.t_snap, source=self.s_snap)
        self.assertEqual(document.validate_export(generic, "generic"), [])
        self.assertEqual(generic["format"], "mappings/1")
        s = document.summary(doc, self.t_snap)
        self.assertEqual(s["target_tables"], 2)

    def test_sync_full_and_incremental(self):
        doc = self._doc()
        dest = connectors.create("duckdb", {"path": str(self.tmp / "dest.duckdb")})
        rep = sync.run(source=self.src, destination=dest, doc=doc, source_snap=self.s_snap, target_snap=self.t_snap)
        self.assertEqual(rep["failed"], 0, rep)
        self.assertEqual(rep["rows"], 6)
        with dest.engine().connect() as conn:
            from sqlalchemy import text
            pts = conn.execute(text("select first_name, gender from patients order by id")).fetchall()
            vis = conn.execute(text("select department_name, status from visits order by id")).fetchall()
        self.assertEqual(pts[0], ("Stephen", "male"))
        self.assertEqual(vis[0], ("OPD", "completed"))
        self.assertEqual(vis[1], ("Lab", "waiting"))
        rep2 = sync.run(source=self.src, destination=dest, doc=doc, source_snap=self.s_snap, target_snap=self.t_snap,
                        mode="incremental_dedupe", state={})
        self.assertEqual(rep2["failed"], 0, rep2)
        self.assertTrue(rep2["state"]["patients"]["cursor"].startswith("2026-01-03"))
        rep3 = sync.run(source=self.src, destination=dest, doc=doc, source_snap=self.s_snap, target_snap=self.t_snap,
                        mode="incremental_dedupe", state=rep2["state"])
        self.assertEqual(rep3["rows"], 0)  # nothing new past the cursor
        with dest.engine().connect() as conn:
            self.assertEqual(conn.execute(text("select count(*) from patients")).scalar(), 3)
        dest.close()

    def test_append_modes_on_keyed_and_unkeyed_tables(self):
        # prod regression: append into a table created with the target's PK failed with duplicate key errors
        from sqlalchemy import text
        doc = self._doc()
        dest = connectors.create("duckdb", {"path": str(self.tmp / "append.duckdb")})
        run = lambda mode, **kw: sync.run(source=self.src, destination=dest, doc=doc, source_snap=self.s_snap,  # noqa: E731
                                          target_snap=self.t_snap, mode=mode, **kw)
        self.assertEqual(run("full_refresh_overwrite")["failed"], 0)          # creates patients with PK id
        rep = run("full_refresh_append")                                      # keyed → merged, not failed
        self.assertEqual(rep["failed"], 0, rep)
        with dest.engine().connect() as conn:
            self.assertEqual(conn.execute(text("select count(*) from patients")).scalar(), 3)
            conn.execute(text("drop table patients"))
            conn.execute(text("drop table visits"))
            conn.commit()
        self.assertEqual(run("full_refresh_append")["failed"], 0)             # fresh tables: no PK
        self.assertEqual(run("full_refresh_append")["failed"], 0)             # → every version kept
        with dest.engine().connect() as conn:
            self.assertEqual(conn.execute(text("select count(*) from patients")).scalar(), 6)
        self.assertEqual(run("full_refresh_overwrite", settings={"tables": ["patients"], "row_limit": 2})["rows"], 2)
        dest.close()

    def test_merge_dedupes_keys_within_one_batch(self):
        from sqlalchemy import text
        dest = connectors.create("duckdb", {"path": str(self.tmp / "merge.duckdb")})
        w = dest.writer()
        t = w.ensure_table("t", [{"name": "id", "family": "integer"}, {"name": "v", "family": "text"}], primary_key=["id"])
        w.write(t, [{"id": 1, "v": "old"}, {"id": 1, "v": "new"}, {"id": 2, "v": "x"}], merge_keys=["id"])
        with dest.engine().connect() as conn:
            self.assertEqual(conn.execute(text("select v from t where id = 1")).fetchall(), [("new",)])
        dest.close()

    def test_coerce(self):
        self.assertEqual(sync.coerce("12", "integer"), 12)
        self.assertIs(sync.coerce("yes", "boolean"), True)
        self.assertIsNone(sync.coerce("not a date", "date"))
        self.assertEqual(sync.coerce({"a": 1}, "text"), '{"a": 1}')


class LLMHelperTests(SimpleTestCase):
    def test_parse_json_tolerates_fences(self):
        self.assertEqual(llm.parse_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(llm.parse_json('sure! {"a": 2} hope that helps'), {"a": 2})
