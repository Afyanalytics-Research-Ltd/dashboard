"""Uploads in S3, scratch workspaces per job, chunked / resumable uploads, and the clean-up around them.

S3 is an in-memory fake that behaves like the real API for the calls storage.py makes (multipart parts are
checked against their Content-MD5, server-side encryption must be requested, missing keys raise)."""

import base64
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.utils import timezone

from connect_studio import services, storage
from connect_studio.engine import sqldump
from connect_studio.models import ChunkedUpload, Job, Snapshot, Source, SourceFile

DUMPS = Path(__file__).parent / "dumps"
User = get_user_model()


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.multipart: dict[str, dict] = {}
        self.calls = []

    def _sse(self, extra):
        assert extra.get("ServerSideEncryption") == "AES256", "objects must be written encrypted"

    def create_multipart_upload(self, Bucket, Key, **kw):
        self._sse(kw)
        uid = f"mp-{len(self.multipart) + 1}"
        self.multipart[uid] = {"key": Key, "parts": {}}
        return {"UploadId": uid}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, ContentMD5):
        assert base64.b64encode(hashlib.md5(Body).digest()).decode() == ContentMD5, "Content-MD5 mismatch"
        etag = f'"{hashlib.md5(Body).hexdigest()}"'
        self.multipart[UploadId]["parts"][PartNumber] = (etag, Body)
        return {"ETag": etag}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):
        mp = self.multipart.pop(UploadId)
        data = b""
        for p in MultipartUpload["Parts"]:
            etag, body = mp["parts"][p["PartNumber"]]
            assert etag == p["ETag"]
            data += body
        self.objects[Key] = data

    def abort_multipart_upload(self, Bucket, Key, UploadId):
        self.multipart.pop(UploadId, None)

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        self._sse(ExtraArgs or {})
        self.objects[Key] = Path(Filename).read_bytes()

    def upload_fileobj(self, Fileobj, Bucket, Key, ExtraArgs=None):
        self._sse(ExtraArgs or {})
        self.objects[Key] = Fileobj.read()

    def download_file(self, Bucket, Key, Filename):
        self.calls.append(("download", Key))
        if Key not in self.objects:
            raise KeyError(Key)
        Path(Filename).write_bytes(self.objects[Key])

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        return {"Contents": [{"Key": k, "Size": len(v)} for k, v in sorted(self.objects.items()) if k.startswith(Prefix)],
                "IsTruncated": False}

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop(o["Key"], None)


class _InlineThread:
    def __init__(self, target, **_kw):
        self.target = target

    def start(self):
        self.target()


def files_under(path: Path) -> list[Path]:
    return [p for p in path.rglob("*") if p.is_file()] if path.exists() else []


class _Base(TestCase):
    s3 = True

    def setUp(self):
        self.media = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.media, True)
        ov = override_settings(MEDIA_ROOT=str(self.media))
        ov.enable()
        self.addCleanup(ov.disable)
        self.fake = FakeS3()
        env = {"CONNECT_STUDIO_S3_BUCKET": "cs-test" if self.s3 else "", "CONNECT_STUDIO_S3_PREFIX": "cs"}
        for p in (mock.patch.dict(os.environ, env), mock.patch.object(storage, "_client", self.fake),
                  mock.patch("connect_studio.jobs._Thread", _InlineThread), mock.patch("connect_studio.jobs.db_connection"),
                  mock.patch("connect_studio.jobs.close_old_connections"), mock.patch.object(services, "CHUNK_SIZE", 64 * 1024)):
            p.start()
            self.addCleanup(p.stop)
        self.client.force_login(User.objects.create_superuser("s3admin", "s@example.com", "pw"))

    def api(self, method, path, body=None):
        fn = getattr(self.client, method.lower())
        if body is not None:
            return fn(f"/connect/api/{path}", data=json.dumps(body), content_type="application/json")
        return fn(f"/connect/api/{path}")

    def new_source(self, kind="sql_dump"):
        return self.api("POST", "sources/", {"kind": kind, "name": f"{kind} src"}).json()["source"]["id"]

    def chunked(self, sid, name, data: bytes, *, last_modified=1, skip=(), corrupt=()):
        r = self.api("POST", f"sources/{sid}/uploads/", {"name": name, "size": len(data), "last_modified": last_modified})
        self.assertEqual(r.status_code, 201, r.content)
        u = r.json()["upload"]
        for i in range(u["total"]):
            if i in skip or i in u["received"]:
                continue
            part = data[i * u["chunk_size"]:(i + 1) * u["chunk_size"]]
            sha = hashlib.sha256(part).hexdigest()
            body = part[:-1] + b"X" if i in corrupt else part
            r = self.client.put(f"/connect/api/uploads/{u['id']}/chunks/{i}/", data=body,
                                content_type="application/octet-stream", HTTP_X_CHUNK_SHA256=sha)
            if i in corrupt:
                self.assertEqual(r.status_code, 422, r.content)
            else:
                self.assertEqual(r.status_code, 200, r.content)
        return u

    def finish(self, u):
        return self.api("POST", f"uploads/{u['id']}/", {})

    def run_job(self, r):
        self.assertEqual(r.status_code, 202, r.content)
        return Job.objects.get(pk=r.json()["job"]["id"])


class ChunkedUploadS3Tests(_Base):
    def test_chunks_go_straight_to_s3_and_resume(self):
        sid = self.new_source()
        data = gzip.compress((DUMPS / "mysqldump.sql").read_bytes() * 150, compresslevel=0)  # ~6 chunks of 64 KB
        self.assertGreater(len(data), 4 * 64 * 1024)
        u = self.chunked(sid, "shop.sql.gz", data, skip={2})
        # the connection dropped before chunk 2: choosing the same file again resumes the same upload
        again = self.api("POST", f"sources/{sid}/uploads/", {"name": "shop.sql.gz", "size": len(data), "last_modified": 1}).json()["upload"]
        self.assertEqual(again["id"], u["id"])
        self.assertNotIn(2, again["received"])
        self.assertEqual(self.finish(u).status_code, 409)                # a chunk is missing
        self.chunked(sid, "shop.sql.gz", data)                           # sends only chunk 2
        r = self.finish(u)
        self.assertEqual(r.status_code, 201, r.content)
        sf = SourceFile.objects.get(source_id=sid)
        self.assertEqual((sf.storage, sf.size), ("s3", len(data)))
        self.assertEqual(self.fake.objects[sf.key], data)
        self.assertTrue(sf.key.startswith(f"cs/sources/{sid}/files/"))
        self.assertEqual(files_under(self.media), [])                    # nothing on this server
        self.assertFalse(ChunkedUpload.objects.exists())

    def test_bad_chunks_are_refused(self):
        sid = self.new_source()
        data = os.urandom(200 * 1024)
        u = self.chunked(sid, "x.sql", data, corrupt={1})
        r = self.client.put(f"/connect/api/uploads/{u['id']}/chunks/0/", data=b"short", content_type="application/octet-stream")
        self.assertEqual(r.status_code, 422)
        self.assertIn("expected", r.json()["error"]["message"])
        r = self.client.put(f"/connect/api/uploads/{u['id']}/chunks/99/", data=b"x", content_type="application/octet-stream")
        self.assertEqual(r.status_code, 422)
        r = self.api("POST", f"sources/{sid}/uploads/", {"name": "x.exe", "size": 10})
        self.assertEqual(r.status_code, 400)
        r = self.api("POST", f"sources/{sid}/uploads/", {"name": "x.sql", "size": 50 * 1024 ** 4})
        self.assertIn("limit", r.json()["error"]["message"])
        self.assertEqual(self.api("DELETE", f"uploads/{u['id']}/").status_code, 200)
        self.assertEqual(self.fake.multipart, {})                        # aborted in S3 too

    def test_single_request_upload_also_lands_in_s3(self):
        sid = self.new_source()
        r = self.client.post(f"/connect/api/sources/{sid}/files/",
                             {"files": SimpleUploadedFile("pg.sql", (DUMPS / "pg_dump_copy.sql").read_bytes())})
        self.assertEqual(r.status_code, 201, r.content)
        self.assertEqual(SourceFile.objects.get(source_id=sid).storage, "s3")
        self.assertEqual(files_under(self.media), [])


class WorkspaceS3Tests(_Base):
    def dump_source(self):
        sid = self.new_source()
        self.assertEqual(self.finish(self.chunked(sid, "clinic.sql", (DUMPS / "mysql_phpmyadmin.sql").read_bytes())).status_code, 201)
        return Source.objects.get(pk=sid)

    def test_jobs_leave_nothing_on_disk_and_reuse_the_restore(self):
        s = self.dump_source()
        real_restore = sqldump.restore
        with mock.patch.object(sqldump, "restore", side_effect=real_restore) as restore:
            self.assertEqual(self.run_job(self.api("POST", f"sources/{s.pk}/test/", {})).status, "succeeded")
            self.assertEqual(files_under(self.media), [])                # workspace gone after the job
            cache = [k for k in self.fake.objects if k.startswith(f"cs/sources/{s.pk}/cache/")]
            self.assertEqual(sorted(Path(k).name for k in cache), ["dump.duckdb", "dump.meta.json"])
            job = self.run_job(self.api("POST", f"sources/{s.pk}/extract/", {}))
            self.assertEqual(job.status, "succeeded", job.error)
            self.assertEqual(restore.call_count, 1)                      # the extract downloaded the cached restore
            self.assertTrue(any("Reused the restored copy" in line["msg"] for line in job.log))
        self.assertEqual(files_under(self.media), [])
        snap = Snapshot.objects.get(source=s).data
        self.assertEqual({t["name"]: t["rows"] for t in snap["tables"]}, {"patients": 3, "visits": 3})
        storage_info = self.api("GET", f"sources/{s.pk}/").json()["source"]["storage"]
        self.assertEqual(storage_info["local"], 0)
        self.assertGreater(storage_info["cache"], 0)
        self.assertEqual(storage_info["where"], "s3")

    def test_a_failing_job_cleans_up_too(self):
        s = self.dump_source()
        with mock.patch("connect_studio.engine.connectors.SQLDumpConnector.discover", side_effect=RuntimeError("boom")):
            job = self.run_job(self.api("POST", f"sources/{s.pk}/extract/", {}))
        self.assertEqual(job.status, "failed")
        self.assertEqual(files_under(self.media), [])

    def test_new_upload_invalidates_the_s3_cache(self):
        s = self.dump_source()
        self.run_job(self.api("POST", f"sources/{s.pk}/test/", {}))
        self.assertTrue(any("/cache/" in k for k in self.fake.objects))
        self.finish(self.chunked(s.pk, "more.sql", (DUMPS / "sqlite_dump.sql").read_bytes()))
        self.assertFalse(any("/cache/" in k for k in self.fake.objects))

    def test_free_space_and_delete_empty_s3(self):
        s = self.dump_source()
        self.run_job(self.api("POST", f"sources/{s.pk}/test/", {}))
        self.chunked(s.pk, "half.sql", os.urandom(300 * 1024), skip={1, 2, 3})       # an unfinished upload
        prefix = f"cs/sources/{s.pk}/"
        self.assertTrue(any(k.startswith(prefix) for k in self.fake.objects))
        r = self.api("POST", f"sources/{s.pk}/free-space/", {})
        self.assertEqual(r.status_code, 200, r.content)
        self.assertFalse(any(k.startswith(prefix) for k in self.fake.objects))
        self.assertEqual(self.fake.multipart, {})
        s2 = self.dump_source()
        self.run_job(self.api("POST", f"sources/{s2.pk}/test/", {}))
        self.assertEqual(self.api("DELETE", f"sources/{s2.pk}/").status_code, 200)
        self.assertFalse(any(k.startswith(f"cs/sources/{s2.pk}/") for k in self.fake.objects))
        self.assertEqual(files_under(self.media), [])

    def test_uploaded_database_destination_is_written_back(self):
        sid = self.new_source("duckdb")
        import duckdb

        db = Path(tempfile.mkdtemp()) / "dest.duckdb"
        duckdb.connect(str(db)).close()
        self.client.post(f"/connect/api/sources/{sid}/files/", {"files": SimpleUploadedFile("dest.duckdb", db.read_bytes())})
        s = Source.objects.get(pk=sid)
        with services.opened(s, "t", write_back=True) as conn:
            w = conn.writer()
            table = w.ensure_table("t", [{"name": "a", "type": "integer", "family": "integer"}])
            w.write(table, [{"a": 1}, {"a": 2}])
        sf = SourceFile.objects.get(source=s)
        back = Path(tempfile.mkdtemp()) / "back.duckdb"
        back.write_bytes(self.fake.objects[sf.key])
        self.assertEqual(duckdb.connect(str(back)).execute("select count(*) from t").fetchone()[0], 2)
        self.assertEqual(files_under(self.media), [])


class JanitorTests(_Base):
    def test_janitor_clears_stale_workspaces_and_uploads(self):
        sid = self.new_source()
        old = self.media / "connect_studio" / "work" / str(sid) / "extract9-old"
        old.mkdir(parents=True)
        (old / "dump.duckdb").write_bytes(b"x" * 1000)
        legacy = self.media / "connect_studio" / "work" / str(sid) / "stage.duckdb"     # pre-S3 leftover
        legacy.write_bytes(b"y" * 10)
        fresh = self.media / "connect_studio" / "work" / str(sid) / "test10-running"
        fresh.mkdir()
        past = time.time() - 7 * 3600
        for p in (old, old / "dump.duckdb", legacy):
            os.utime(p, (past, past))
        u = self.chunked(sid, "a.sql", os.urandom(100 * 1024), skip={1})
        ChunkedUpload.objects.filter(pk=u["id"]).update(updated_at=timezone.now() - timezone.timedelta(hours=30))
        res = services.janitor()
        self.assertEqual(res["workspaces"], 2)
        self.assertEqual(res["uploads"], 1)
        self.assertFalse(old.exists())
        self.assertFalse(legacy.exists())
        self.assertTrue(fresh.exists())                                   # too recent: may belong to a running job
        self.assertEqual(self.fake.multipart, {})
        call_command("connect_studio_cleanup", stdout=io.StringIO())

    def test_move_local_uploads_to_s3(self):
        with mock.patch.dict(os.environ, {"CONNECT_STUDIO_S3_BUCKET": ""}):
            sid = self.new_source()
            self.client.post(f"/connect/api/sources/{sid}/files/",
                             {"files": SimpleUploadedFile("old.sql", (DUMPS / "sqlite_dump.sql").read_bytes())})
        sf = SourceFile.objects.get(source_id=sid)
        self.assertEqual(sf.storage, "local")
        local = Path(sf.file.path)
        call_command("connect_studio_cleanup", "--move-to-s3", stdout=io.StringIO())
        sf.refresh_from_db()
        self.assertEqual(sf.storage, "s3")
        self.assertFalse(local.exists())
        self.assertEqual(self.fake.objects[sf.key], (DUMPS / "sqlite_dump.sql").read_bytes())


class LocalFallbackTests(_Base):
    s3 = False

    def test_without_a_bucket_chunks_build_a_local_file_and_jobs_still_clean_up(self):
        sid = self.new_source()
        data = (DUMPS / "pg_dump_inserts.sql").read_bytes()
        r = self.finish(self.chunked(sid, "pg.sql", data))
        self.assertEqual(r.status_code, 201, r.content)
        sf = SourceFile.objects.get(source_id=sid)
        self.assertEqual(sf.storage, "local")
        self.assertEqual(Path(sf.file.path).read_bytes(), data)
        self.assertFalse(list((self.media / "connect_studio" / "uploads").glob("*.part")))
        self.assertEqual(self.run_job(self.api("POST", f"sources/{sid}/test/", {})).status, "succeeded")
        self.assertEqual(files_under(self.media / "connect_studio" / "work"), [])
        self.assertEqual(self.fake.objects, {})
