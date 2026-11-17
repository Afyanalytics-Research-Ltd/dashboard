"""SQL dump connector: dumps from five database engines restored into DuckDB, and the clean-up after a migration.

The fixtures in tests/dumps/ are the shapes each tool really writes: pg_dump 16 output (COPY and --inserts, made
from a real server), sqlite3 .dump, phpMyAdmin and mysqldump scripts, an SSMS "Generate Scripts" file (UTF-16,
CRLF, GO batches) and a SQL Developer export. No real patient data.
"""

import bz2
import gzip
import json
import shutil
import tempfile
import zipfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from unittest import mock

import duckdb
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings

from connect_studio.engine import connectors, sqldump
from connect_studio.models import Job, Snapshot, Source

DUMPS = Path(__file__).parent / "dumps"
User = get_user_model()


def restore(*names, **kw):
    tmp = Path(tempfile.mkdtemp())
    paths = [n if isinstance(n, Path) else DUMPS / n for n in names]
    meta = sqldump.restore(paths, tmp / "db.duckdb", workdir=tmp, **kw)
    return meta, duckdb.connect(str(tmp / "db.duckdb"), read_only=True)


def types(con, table):
    return dict(con.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
                            [table]).fetchall())


class EngineDialects(SimpleTestCase):
    def test_phpmyadmin(self):
        meta, con = restore("mysql_phpmyadmin.sql")
        self.assertEqual(meta["dialect"], "mysql")
        rows = con.execute("SELECT id, patient_no, name, dob, gender, balance, note FROM patients ORDER BY id").fetchall()
        self.assertEqual(rows[0], (1, "947/26", "Hannah Nduta", None, "Female", Decimal("1250.50"), "kagaa"))   # 0000-00-00 → NULL
        self.assertEqual(rows[1][2], 'O\'Brien "Jr"')
        self.assertEqual(rows[1][6], "line one\nline two; with a semicolon")
        self.assertEqual(rows[2][2], "Back\\slash")
        t = meta["tables"]["patients"]
        self.assertEqual(t["primary_key"], ["id"])                          # from ALTER TABLE … ADD PRIMARY KEY
        self.assertEqual(t["comment"], "Registered patients")
        self.assertEqual(t["columns"]["patient_no"]["comment"], "Hospital number")
        self.assertEqual(t["columns"]["gender"]["enum"], ["Male", "Female"])
        self.assertTrue(t["columns"]["id"]["auto"])                         # ALTER … MODIFY … AUTO_INCREMENT
        self.assertIn(["patient_no"], t["uniques"])
        self.assertEqual(meta["tables"]["visits"]["foreign_keys"],
                         [{"columns": ["patient_id"], "table": "patients", "ref_columns": ["id"]}])
        ty = types(con, "visits")
        self.assertEqual(ty["visit_date"], "TIMESTAMP")
        self.assertEqual(ty["amount"], "VARCHAR")                           # 'n/a' doesn't fit decimal(15,2)
        self.assertTrue(any("visits.amount" in w and "'n/a'" in w for w in meta["warnings"]))
        self.assertIsNone(con.execute("SELECT visit_date FROM visits WHERE id = 11").fetchone()[0])

    def test_mysqldump_with_routines(self):
        meta, con = restore("mysqldump.sql")
        self.assertEqual(sorted(meta["tables"]), ["items", "orders"])
        self.assertEqual(con.execute("SELECT count(*) FROM orders").fetchone()[0], 2)    # the routine's INSERT is not data
        rows = con.execute("SELECT id, code, price, photo, updated FROM items ORDER BY id").fetchall()
        self.assertEqual(rows[1][1], "B;2")
        self.assertEqual(rows[2][1], "C'3")
        self.assertEqual(rows[1][2], 1000.0)
        self.assertEqual(rows[0][3], "PNG\0data")                           # _binary introducer dropped
        self.assertEqual(rows[1][3], "0x89504E47")
        self.assertEqual(rows[2][4], datetime(2026, 1, 2, 11, 30, 0, 125000))
        self.assertEqual(meta["tables"]["items"]["primary_key"], ["id"])
        self.assertEqual(meta["tables"]["orders"]["foreign_keys"][0]["table"], "items")

    def test_pg_dump_copy_and_inserts_agree(self):
        m1, c1 = restore("pg_dump_copy.sql")
        m2, c2 = restore("pg_dump_inserts.sql")
        for meta in (m1, m2):
            self.assertEqual(meta["dialect"], "postgres")
            self.assertEqual(sorted(meta["tables"]), ["lookup", "patients", "visits"])   # one non-default schema: no prefix
            self.assertEqual(meta["tables"]["visits"]["foreign_keys"][0]["table"], "patients")
            self.assertEqual(meta["tables"]["patients"]["comment"], "Registered patients")
            self.assertEqual(meta["tables"]["patients"]["columns"]["dob"]["comment"], "Date of birth")
            self.assertEqual(meta["skipped"].get("procedures / functions / triggers"), 1)
        q = "SELECT id, name, sex, dob, tags, extra, balance FROM patients ORDER BY id"
        self.assertEqual(c1.execute(q).fetchall(), c2.execute(q).fetchall())
        rows = c1.execute(q).fetchall()
        self.assertEqual(rows[0], (1, "Amina O'Brien", "female", date(1990, 5, 1), "{a,b}", '{"k": "v;w"}', Decimal("1250.50")))
        self.assertEqual(rows[1][1], "Tab\there and back\\slash")
        self.assertEqual(rows[2][1], "Line\nbreak; with semicolon")
        self.assertEqual(c1.execute("SELECT count(*), sum(paid::int) FROM visits").fetchone(), (5, 2))
        self.assertEqual(types(c1, "visits")["paid"], "BOOLEAN")

    def test_sqlite_dump(self):
        meta, con = restore("sqlite_dump.sql")
        self.assertEqual(meta["dialect"], "sqlite")
        self.assertEqual(sorted(meta["tables"]), ["bed", "ward"])             # no sqlite_sequence, no view
        self.assertEqual(con.execute("SELECT * FROM bed ORDER BY id").fetchall(),
                         [(1, 1, "B1", 1500.5, True), (2, 2, None, None, False), (3, 1, "x'y", 2000.0, True)])
        self.assertEqual(meta["tables"]["bed"]["foreign_keys"][0]["table"], "ward")

    def test_ssms_script_utf16(self):
        self.assertTrue((DUMPS / "tsql_ssms.sql").read_bytes().startswith(b"\xff\xfe"))
        meta, con = restore("tsql_ssms.sql")
        self.assertEqual(meta["dialect"], "tsql")
        rows = con.execute("SELECT * FROM Patients ORDER BY PatientID").fetchall()
        self.assertEqual(rows[0][:6], (1, "Amina Wanjiru", date(1990, 5, 1), datetime(2024, 1, 15, 9, 30), 1250.5, True))
        self.assertEqual(rows[1][3], datetime(2023, 11, 12, 10, 0))          # CAST(0x0000B0B800A4CB80 AS DateTime)
        self.assertEqual(rows[2][2], date(2011, 1, 6))                       # CAST(0xBE330B00 AS Date)
        self.assertEqual(rows[2][6], "GO\r\nnot a batch separator inside a string")
        self.assertEqual(con.execute("SELECT count(*) FROM Visits").fetchone()[0], 2)    # the procedure's INSERT isn't data
        p = meta["tables"]["Patients"]
        self.assertEqual(p["primary_key"], ["PatientID"])
        self.assertEqual(p["comment"], "Patient register")
        self.assertEqual(p["columns"]["DOB"]["comment"], "Date of birth")
        self.assertEqual(p["columns"]["IsActive"]["default"], "1")
        self.assertEqual(meta["tables"]["Visits"]["foreign_keys"][0]["table"], "Patients")
        self.assertEqual(types(con, "Visits")["Fee"], "DECIMAL(10,2)")

    def test_oracle_export(self):
        meta, con = restore("oracle_sqldev.sql")
        self.assertEqual(meta["dialect"], "oracle")
        self.assertEqual(sorted(meta["tables"]), ["PATIENTS", "VISITS"])
        self.assertEqual(con.execute("SELECT ID, NAME, DOB, BALANCE FROM PATIENTS ORDER BY ID").fetchall(),
                         [(1, "Amina O'Brien", datetime(1990, 5, 1), Decimal("1250.50")), (2, "Juma", datetime(2001, 12, 31), None)])
        self.assertEqual(con.execute("SELECT ID, VISITED_AT FROM VISITS ORDER BY ID").fetchall(),
                         [(1, datetime(2024, 1, 15, 9, 30)), (2, None)])         # NUMBER with integers → BIGINT
        self.assertFalse(meta["tables"]["PATIENTS"]["columns"]["NAME"]["nullable"])
        self.assertEqual(meta["tables"]["VISITS"]["foreign_keys"][0]["table"], "PATIENTS")
        self.assertNotIn(99, [r[0] for r in con.execute("SELECT ID FROM VISITS").fetchall()])


class EngineFiles(SimpleTestCase):
    def test_detection(self):
        for name, d in (("mysql_phpmyadmin.sql", "mysql"), ("mysqldump.sql", "mysql"), ("pg_dump_copy.sql", "postgres"),
                        ("sqlite_dump.sql", "sqlite"), ("oracle_sqldev.sql", "oracle")):
            self.assertEqual(sqldump.detect_dialect((DUMPS / name).read_text()), d, name)
        self.assertEqual(sqldump.detect_dialect((DUMPS / "tsql_ssms.sql").read_text(encoding="utf-16")), "tsql")

    def test_compressed_and_zipped(self):
        tmp = Path(tempfile.mkdtemp())
        gz = tmp / "dump.sql.gz"
        gz.write_bytes(gzip.compress((DUMPS / "mysqldump.sql").read_bytes()))
        bz = tmp / "dump.sql.bz2"
        bz.write_bytes(bz2.compress((DUMPS / "sqlite_dump.sql").read_bytes()))
        zp = tmp / "both.zip"
        with zipfile.ZipFile(zp, "w") as z:
            z.write(DUMPS / "pg_dump_inserts.sql", "export/pg.sql")
            z.write(DUMPS / "oracle_sqldev.sql", "export/ora.sql")
            z.writestr("__MACOSX/._pg.sql", "junk")
        meta, con = restore(gz, bz, zp)
        # two non-default schemas (Oracle HR, Postgres clinic) → those tables are prefixed; public / unqualified aren't
        self.assertEqual(sorted(meta["tables"]), ["HR__PATIENTS", "HR__VISITS", "bed", "clinic__patients", "clinic__visits",
                                                  "items", "lookup", "orders", "ward"])
        self.assertEqual(meta["tables"]["clinic__visits"]["foreign_keys"][0]["table"], "clinic__patients")
        self.assertEqual(meta["dialects"], ["mysql", "oracle", "postgres", "sqlite"])
        self.assertEqual(con.execute("SELECT count(*) FROM items").fetchone()[0], 3)

    def test_data_before_ddl_and_table_filter(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "1_data.sql").write_text("INSERT INTO `w` VALUES (1,'a'),(2,'b');\n")
        (tmp / "2_schema.sql").write_text("CREATE TABLE `w` (`id` int NOT NULL, `name` varchar(5), PRIMARY KEY (`id`));\n")
        meta, con = restore(tmp / "1_data.sql", tmp / "2_schema.sql", dialect="mysql")
        self.assertEqual(con.execute("SELECT id, name FROM w ORDER BY id").fetchall(), [(1, "a"), (2, "b")])
        self.assertEqual(meta["tables"]["w"]["primary_key"], ["id"])
        meta, con = restore("mysqldump.sql", tables=["orders"], max_rows=1)
        self.assertEqual(list(meta["tables"]), ["orders"])
        self.assertEqual(con.execute("SELECT count(*) FROM orders").fetchone()[0], 1)

    def test_untyped_keeps_text(self):
        meta, con = restore("pg_dump_copy.sql", typed=False)
        self.assertEqual(set(types(con, "patients").values()), {"VARCHAR"})

    def test_binary_dumps_explain_themselves(self):
        tmp = Path(tempfile.mkdtemp())
        pgc = tmp / "db.dump"
        pgc.write_bytes(b"PGDMP\x01\x0e\x00" + b"\x00" * 64)
        bak = tmp / "db.bak"
        bak.write_bytes(b"TAPE\x00" + b"\x00" * 64)
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(sqldump.DumpError, "custom-format dump.*pg_dump --format=plain"):
                restore(pgc)
        with self.assertRaisesRegex(sqldump.DumpError, r"SQL Server backup \(\.bak\).*Generate Scripts"):
            restore(bak)
        empty = tmp / "notes.sql"
        empty.write_text("-- nothing here\nSET NAMES utf8;\n")
        with self.assertRaisesRegex(sqldump.DumpError, "No tables found"):
            restore(empty)

    def test_pg_custom_format_goes_through_pg_restore(self):
        tmp = Path(tempfile.mkdtemp())
        dump = tmp / "clinic.dump"
        dump.write_bytes(b"PGDMP\x01\x10\x00" + b"\x00" * 64)       # a custom-format header
        calls = []

        def fake_run(cmd, **kw):                                      # pg_restore -f <out> <in> → plain SQL
            calls.append(cmd)
            Path(cmd[cmd.index("-f") + 1]).write_text((DUMPS / "pg_dump_copy.sql").read_text())
            return mock.Mock(returncode=0, stderr="")
        with mock.patch("shutil.which", return_value="/usr/bin/pg_restore"), mock.patch("subprocess.run", fake_run):
            meta, con = restore(dump)
        self.assertEqual(calls[0][:3], ["/usr/bin/pg_restore", "--no-owner", "--no-privileges"])
        self.assertEqual(meta["dialect"], "postgres")
        self.assertEqual(con.execute("SELECT count(*) FROM visits").fetchone()[0], 5)

    def test_value_parser_wrappers(self):
        vp = sqldump.ValueParser("tsql")
        row = next(vp.rows("(CONVERT(varchar(10), '2020-01-01', 120), CAST(5 AS DECIMAL(10,2)), N'x', DEFAULT, -3)"))
        self.assertEqual(row, ["2020-01-01", "5", "x", None, "-3"])
        self.assertEqual(sqldump._oracle_datetime("15-JAN-24 09.30.00.000000000 AM", "DD-MON-RR HH.MI.SSXFF AM"),
                         "2024-01-15 09:30:00")
        self.assertEqual(sqldump._mysql_datetime("31/12/2020", "%d/%m/%Y"), "2020-12-31")
        self.assertEqual(next(sqldump.ValueParser("postgres").rows("('2020-01-01'::date, E'a\\tb', 1.5e3)")),
                         ["2020-01-01", "a\tb", "1.5e3"])


class ConnectorTests(SimpleTestCase):
    def make(self, *names, **config):
        work = Path(tempfile.mkdtemp())
        files = []
        for n in names:
            shutil.copy(DUMPS / n, work / n)
            files.append(work / n)
        return connectors.create("sql_dump", config, files=files, workdir=work / "work")

    def test_test_discover_read_and_cache(self):
        c = self.make("mysql_phpmyadmin.sql")
        res = c.test()
        self.assertIn("MySQL / MariaDB dump → 2 tables, 6 rows", res["message"])
        first = json.loads((c.workdir / "dump.meta.json").read_text())["restored_at"]
        snap = c.discover({})
        t = {x["name"]: x for x in snap["tables"]}
        self.assertEqual(t["patients"]["primary_key"], ["id"])
        self.assertEqual(t["patients"]["rows"], 3)
        cols = {x["name"]: x for x in t["patients"]["columns"]}
        self.assertEqual(cols["dob"]["type"], "date")
        self.assertEqual(cols["gender"]["allowed_values"], ["Male", "Female"])
        self.assertTrue(cols["id"]["pk"])
        self.assertEqual(cols["patient_no"]["comment"], "Hospital number")
        visits = {x["name"]: x for x in t["visits"]["columns"]}
        self.assertEqual(visits["amount"]["type"], "VARCHAR")               # kept as text: declared type not claimed
        self.assertEqual(visits["amount"]["declared_type"], "decimal(15,2)")
        fk = [r for r in snap["relationships"] if r["kind"] == "foreign_key"]
        self.assertEqual([(r["from"], r["to"]) for r in fk], [("visits", "patients")])
        self.assertEqual(snap["dump"]["dialect"], "mysql")
        rows = [r for batch in c.read("patients", columns=["id", "name"]) for r in batch]
        self.assertEqual(len(rows), 3)
        c.close()
        # unchanged uploads + options → the same restore is reused
        c2 = connectors.create("sql_dump", {}, files=c.files, workdir=c.workdir)
        c2.test()
        self.assertEqual(json.loads((c.workdir / "dump.meta.json").read_text())["restored_at"], first)
        # an option that changes the restore → rebuilt
        c3 = connectors.create("sql_dump", {"tables": "visits"}, files=c.files, workdir=c.workdir)
        self.assertIn("1 tables", c3.test()["message"])
        removed = c3.cleanup()
        self.assertIn("dump.duckdb", removed)
        self.assertFalse((c.workdir / "dump.duckdb").exists())

    def test_read_only_and_missing_file(self):
        c = self.make("sqlite_dump.sql")
        with self.assertRaises(connectors.ConnectorError):
            c.writer()
        gone = connectors.create("sql_dump", {"path": "/nonexistent/x.sql"}, files=[], workdir=Path(tempfile.mkdtemp()))
        with self.assertRaisesRegex(connectors.ConnectorError, "Not found on the server"):
            gone.test()


class _InlineThread:
    def __init__(self, target, **_kw):
        self.target = target

    def start(self):
        self.target()


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class CleanupApiTests(TestCase):
    def setUp(self):
        self.client.force_login(User.objects.create_superuser("dumpadmin", "d@example.com", "pw"))
        for p in (mock.patch("connect_studio.jobs._Thread", _InlineThread), mock.patch("connect_studio.jobs.db_connection"),
                  mock.patch("connect_studio.jobs.close_old_connections")):
            p.start()
            self.addCleanup(p.stop)

    def api(self, method, path, body=None):
        fn = getattr(self.client, method.lower())
        if body is not None:
            return fn(f"/connect/api/{path}", data=json.dumps(body), content_type="application/json")
        return fn(f"/connect/api/{path}")

    def source_with_dump(self):
        sid = self.api("POST", "sources/", {"kind": "sql_dump", "name": "Hospital dump"}).json()["source"]["id"]
        gz = gzip.compress((DUMPS / "mysql_phpmyadmin.sql").read_bytes())
        r = self.client.post(f"/connect/api/sources/{sid}/files/", {"files": SimpleUploadedFile("hospital.sql.gz", gz)})
        self.assertEqual(r.status_code, 201, r.content)
        r = self.api("POST", f"sources/{sid}/extract/", {})
        self.assertEqual(r.status_code, 202, r.content)
        self.assertEqual(Job.objects.get(pk=r.json()["job"]["id"]).status, "succeeded")
        return Source.objects.get(pk=sid)

    def test_free_space_keeps_snapshots_and_mappings(self):
        s = self.source_with_dump()
        upload = Path(s.files.get().file.path)
        storage = self.api("GET", f"sources/{s.pk}/").json()["source"]["storage"]
        self.assertEqual(storage["local"], 0)                               # the job's workspace is already gone
        self.assertGreater(storage["uploads"], 0)
        r = self.api("POST", f"sources/{s.pk}/free-space/", {})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(r.json()["freed_bytes"], storage["total"])
        self.assertEqual(r.json()["files_deleted"], ["hospital.sql.gz"])
        self.assertFalse(upload.exists())
        self.assertFalse(s.work_path().exists())
        self.assertEqual(Snapshot.objects.filter(source=s).count(), 1)
        detail = self.api("GET", f"sources/{s.pk}/").json()["source"]
        self.assertEqual(detail["storage"]["total"], 0)
        self.assertTrue(detail["storage"]["freed_at"])
        # testing again says what happened instead of a bare "upload a file"
        r = self.api("POST", f"sources/{s.pk}/test/", {})
        self.assertEqual(r.status_code, 400)
        self.assertIn("deleted on", r.json()["error"]["message"])
        # uploading again clears the flag and works
        self.client.post(f"/connect/api/sources/{s.pk}/files/",
                         {"files": SimpleUploadedFile("again.sql", (DUMPS / "sqlite_dump.sql").read_bytes())})
        self.assertNotIn("freed_at", Source.objects.get(pk=s.pk).config)
        r = self.api("POST", f"sources/{s.pk}/test/", {})
        self.assertEqual(Job.objects.get(pk=r.json()["job"]["id"]).status, "succeeded")

    def test_delete_source_leaves_nothing_on_disk(self):
        s = self.source_with_dump()
        upload, work = Path(s.files.get().file.path), s.work_path()
        (work / "leftover").mkdir(parents=True, exist_ok=True)              # e.g. from a crashed job
        r = self.api("DELETE", f"sources/{s.pk}/")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertGreater(r.json()["freed_bytes"], 0)
        self.assertFalse(upload.exists())
        self.assertFalse(work.exists())

    def test_busy_source_is_not_touched(self):
        s = self.source_with_dump()
        Job.objects.create(kind="extract", status="running", source=s)
        for method, path in (("POST", f"sources/{s.pk}/free-space/"), ("DELETE", f"sources/{s.pk}/")):
            r = self.api(method, path, {} if method == "POST" else None)
            self.assertEqual(r.status_code, 409, r.content)
        self.assertTrue(Path(s.files.get().file.path).exists())

    def test_jobs_leave_no_workspace_behind(self):
        s = self.source_with_dump()                                          # ran an extraction
        self.assertEqual([p for p in s.work_path().rglob("*") if p.is_file()] if s.work_path().exists() else [], [])
