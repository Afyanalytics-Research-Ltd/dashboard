#!/usr/bin/env python3
# pipelines/platform_backup.py
"""
Full platform backup to S3: every Postgres database, the Databend
warehouse, and Snowflake.

Driven by dags/platform_backup_to_s3_dag.py (daily). It also runs by hand:

    python platform_backup.py setup-bucket          # create/harden the bucket (idempotent)
    python platform_backup.py postgres|databend|snowflake [--dry-run]
    python platform_backup.py all [--dry-run]

S3 LAYOUT  s3://$BACKUP_S3_BUCKET/<source>/<YYYY-MM-DD>/
    postgres/<target>/_globals.sql.gz        roles + tablespaces (pg_dumpall -g)
    postgres/<target>/<db>.dump              pg_dump custom format  -> pg_restore
    databend/<db>/_ddl.sql                   SHOW CREATE TABLE/VIEW for the db
    databend/<db>/<table>/data_*.parquet     COPY INTO s3 (server side)
    snowflake/<DB>/_ddl.sql                  GET_DDL('DATABASE', db, TRUE)
    snowflake/<DB>/<SCHEMA>/<TABLE>/part-NNNNN.parquet   (data days only)
    manifest.json                            every object, size, sha256, status

Retention is not done here: the bucket's lifecycle rule (setup-bucket)
expires everything after BACKUP_RETENTION_DAYS (30). Versioning is on, so a
same-day re-run never destroys the earlier copy.

Nothing is spooled to local disk except Snowflake parquet parts (one part at a
time). Passwords go to pg_dump through PGPASSWORD, never argv.

A target whose hostname does not resolve is "skipped" (that service is not
deployed on this host, e.g. Django's postgres-container on the WSL box). A
target that resolves but fails is "failed", and the DAG run fails.
Snowflake is "skipped" rather than "failed" when the account refuses the
connection (e.g. the 2026-11-10 ACCOUNTADMIN system lock), unless
BACKUP_SNOWFLAKE_REQUIRED=1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import logging
import os
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("platform_backup")

try:  # same .env the other pipeline scripts use when run by hand
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass

# ── config ───────────────────────────────────────────────────────────────
ACCOUNT_ID = "707559309638"
BUCKET = os.getenv("BACKUP_S3_BUCKET", f"afya-platform-backups-{ACCOUNT_ID}")
REGION = os.getenv("BACKUP_S3_REGION") or os.getenv("AWS_REGION") or "eu-north-1"
SOURCE = os.getenv("BACKUP_SOURCE_LABEL", "datahub")       # which machine these backups came from
RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
# Full Snowflake table data is exported on this weekday only (Mon=0 .. Sun=6);
# DDL is exported every run. A daily full scan of every table costs credits
# and looks exactly like bulk exfiltration in QUERY_HISTORY - keep it weekly.
SNOWFLAKE_DATA_WEEKDAY = int(os.getenv("BACKUP_SNOWFLAKE_DATA_WEEKDAY", "6"))
SNOWFLAKE_PART_ROWS = int(os.getenv("BACKUP_SNOWFLAKE_PART_ROWS", "2000000"))
SNOWFLAKE_QUERY_TAG = "platform_backup"
SNOWFLAKE_SKIP_DATABASES = {"SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"}

# Postgres servers. Credentials are env-var NAMES (resolved at run time).
# databases=None -> every non-template database on the server.
POSTGRES_TARGETS = [
    # Not POSTGRES_HOST ("postgres"): the Airflow worker sits on two networks that
    # both have a service called postgres. The container name is unique.
    {"name": "django", "host": "postgres-container", "user_env": "POSTGRES_USERNAME",
     "password_env": "POSTGRES_PASSWORD"},
    {"name": "airflow", "host": "airflow-postgres", "user": "airflow", "password": "airflow"},
    # Pre-2026-11-04 Airflow stack's database; still running, kept until removed.
    {"name": "airflow-legacy", "host": "airflow-db", "user_env": "AIRFLOW_DB_USER",
     "password_env": "AIRFLOW_DB_PASSWORD"},
    {"name": "redash", "host": "redash-postgres", "user_env": "REDASH_DB_USER", "password_env": "REDASH_DB_PASSWORD"},
    {"name": "superset", "host": "superset-postgres", "user_env": "SUPERSET_DB_USER",
     "password_env": "SUPERSET_DB_PASSWORD"},
    {"name": "mlflow", "host": "mlflow-db", "user_env": "MLFLOW_DB_USER", "password_env": "MLFLOW_DB_PASSWORD"},
    {"name": "marquez", "host": "marquez-db", "user_env": "MARQUEZ_DB_USER", "password_env": "MARQUEZ_DB_PASSWORD"},
]


# ── helpers ──────────────────────────────────────────────────────────────
def run_prefix(day: dt.date | None = None) -> str:
    return f"{SOURCE}/{(day or dt.date.today()).isoformat()}"


def aws_keys() -> tuple[str, str]:
    """BACKUP_AWS_* if set (a dedicated key, so the backup never disturbs the
    host's own AWS_* settings), else the standard AWS_* variables."""
    return ((os.getenv("BACKUP_AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID") or "").strip(),
            (os.getenv("BACKUP_AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip())


def s3():
    import boto3
    key_id, secret = aws_keys()
    if key_id and secret:
        return boto3.client("s3", region_name=REGION, aws_access_key_id=key_id, aws_secret_access_key=secret)
    return boto3.client("s3", region_name=REGION)   # instance role / default chain


def _resolve(target: dict) -> dict:
    def pick(key, default=None):
        env = target.get(f"{key}_env")
        val = os.getenv(env) if env else None
        val = (val or "").strip().strip("'\"") or target.get(key) or default
        return val
    return {"name": target["name"], "host": pick("host"), "port": pick("port"),
            "user": pick("user"), "password": pick("password"),
            "databases": target.get("databases")}


def _resolvable(host: str | None) -> bool:
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


class _HashingReader:
    """File-like wrapper: sha256 + byte count of whatever boto3 reads through it."""

    def __init__(self, raw):
        self.raw, self.size, self.sha = raw, 0, hashlib.sha256()

    def read(self, n=-1):
        data = self.raw.read(n)
        self.size += len(data)
        self.sha.update(data)
        return data


def _stream_cmd_to_s3(cmd: str, env_extra: dict, key: str, *, dry_run: bool) -> dict:
    """Run a shell pipeline and multipart-upload its stdout to s3://BUCKET/key."""
    if dry_run:
        log.info("[dry-run] %s -> s3://%s/%s", cmd.split("|")[0].strip()[:120], BUCKET, key)
        return {"key": key, "status": "dry_run"}
    from boto3.s3.transfer import TransferConfig
    started = time.time()
    proc = subprocess.Popen(["bash", "-o", "pipefail", "-c", cmd], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, **env_extra})
    reader = _HashingReader(proc.stdout)
    client = s3()
    upload_error = None
    try:
        client.upload_fileobj(reader, BUCKET, key,
                              ExtraArgs={"ServerSideEncryption": "AES256"},
                              Config=TransferConfig(multipart_chunksize=64 * 1024 * 1024, max_concurrency=4))
    except Exception as e:  # noqa: BLE001 - reported below together with the dump's stderr
        upload_error = e
        proc.kill()
    rc = proc.wait()
    stderr = proc.stderr.read().decode(errors="replace").strip()
    if rc != 0 or upload_error:
        try:
            client.delete_object(Bucket=BUCKET, Key=key)  # never leave a truncated dump as "the backup"
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"{key}: exit={rc} upload_error={upload_error!r} stderr={stderr[-1500:]}")
    if reader.size == 0:
        raise RuntimeError(f"{key}: dump produced 0 bytes")
    return {"key": key, "status": "ok", "bytes": reader.size, "sha256": reader.sha.hexdigest(),
            "seconds": round(time.time() - started, 1)}


def _put_bytes(key: str, body: bytes, *, dry_run: bool) -> dict:
    if dry_run:
        log.info("[dry-run] put s3://%s/%s (%d bytes)", BUCKET, key, len(body))
        return {"key": key, "status": "dry_run"}
    s3().put_object(Bucket=BUCKET, Key=key, Body=body, ServerSideEncryption="AES256")
    return {"key": key, "status": "ok", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def _skipped(engine, name, reason):
    log.warning("%s/%s skipped: %s", engine, name, reason)
    return {"engine": engine, "target": name, "status": "skipped", "reason": reason, "objects": []}


# ── bucket setup ─────────────────────────────────────────────────────────
def setup_bucket() -> dict:
    """Create (if missing) and harden the backup bucket. Idempotent."""
    from botocore.exceptions import ClientError
    c = s3()
    try:
        c.head_bucket(Bucket=BUCKET)
        created = False
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("404", "NoSuchBucket"):
            raise
        c.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        c.get_waiter("bucket_exists").wait(Bucket=BUCKET)
        created = True
    c.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    c.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
    c.put_bucket_encryption(Bucket=BUCKET, ServerSideEncryptionConfiguration={"Rules": [{
        "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}, "BucketKeyEnabled": True}]})
    c.put_bucket_lifecycle_configuration(Bucket=BUCKET, LifecycleConfiguration={"Rules": [{
        "ID": f"expire-after-{RETENTION_DAYS}d", "Status": "Enabled", "Filter": {"Prefix": ""},
        "Expiration": {"Days": RETENTION_DAYS},
        "NoncurrentVersionExpiration": {"NoncurrentDays": 7},
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 2},
    }, {
        # warehouse_bisync's bulk staging area: files are purged after each load;
        # this only catches leftovers of a failed run.
        "ID": "bisync-stage-2d", "Status": "Enabled", "Filter": {"Prefix": "_bisync_stage/"},
        "Expiration": {"Days": 2}, "NoncurrentVersionExpiration": {"NoncurrentDays": 1},
    }, {
        "ID": "expired-delete-markers", "Status": "Enabled", "Filter": {"Prefix": ""},
        "Expiration": {"ExpiredObjectDeleteMarker": True},
    }]})
    c.put_bucket_tagging(Bucket=BUCKET, Tagging={"TagSet": [
        {"Key": "Purpose", "Value": "platform-backups"}, {"Key": "ManagedBy", "Value": "platform_backup.py"}]})
    log.info("bucket s3://%s ready (created=%s, region=%s, retention=%sd)", BUCKET, created, REGION, RETENTION_DAYS)
    return {"bucket": BUCKET, "created": created, "region": REGION, "retention_days": RETENTION_DAYS}


def check_bucket() -> dict:
    """Preflight: the bucket exists, is versioned and has the expiry rule."""
    c = s3()
    c.head_bucket(Bucket=BUCKET)
    if c.get_bucket_versioning(Bucket=BUCKET).get("Status") != "Enabled":
        raise RuntimeError(f"s3://{BUCKET} versioning is not enabled - run `platform_backup.py setup-bucket`")
    rules = c.get_bucket_lifecycle_configuration(Bucket=BUCKET).get("Rules", [])
    if not any(r.get("Expiration", {}).get("Days") for r in rules):
        raise RuntimeError(f"s3://{BUCKET} has no expiry lifecycle rule - run `platform_backup.py setup-bucket`")
    return {"bucket": BUCKET, "region": REGION}


# ── Postgres ─────────────────────────────────────────────────────────────
def backup_postgres(prefix: str, *, dry_run: bool = False) -> list[dict]:
    import psycopg2
    results = []
    for t in map(_resolve, POSTGRES_TARGETS):
        name, host, port = t["name"], t["host"], t["port"] or "5432"
        if not _resolvable(host):
            results.append(_skipped("postgres", name, f"host {host!r} not on this network"))
            continue
        res = {"engine": "postgres", "target": name, "host": host, "status": "ok", "objects": [], "errors": []}
        env = {"PGPASSWORD": t["password"] or ""}
        conn_args = f"-h {shlex.quote(host)} -p {shlex.quote(port)} -U {shlex.quote(t['user'] or '')}"
        try:
            dbs = t["databases"]
            if not dbs:
                with psycopg2.connect(host=host, port=port, user=t["user"], password=t["password"],
                                      dbname="postgres", connect_timeout=15) as c, c.cursor() as cur:
                    cur.execute("SELECT datname FROM pg_database WHERE NOT datistemplate AND datallowconn ORDER BY 1")
                    dbs = [r[0] for r in cur.fetchall()]
            res["databases"] = dbs
            base = f"{prefix}/postgres/{name}"
            try:
                res["objects"].append(_stream_cmd_to_s3(
                    f"pg_dumpall {conn_args} --globals-only --no-password | gzip -6", env,
                    f"{base}/_globals.sql.gz", dry_run=dry_run))
            except Exception as e:  # noqa: BLE001 - needs superuser; the db dumps still matter more
                res["errors"].append(f"globals: {e}")
            for db in dbs:
                try:
                    res["objects"].append(_stream_cmd_to_s3(
                        f"pg_dump {conn_args} --no-password -Fc -Z 6 -d {shlex.quote(db)}", env,
                        f"{base}/{db}.dump", dry_run=dry_run))
                except Exception as e:  # noqa: BLE001
                    res["errors"].append(f"{db}: {e}")
        except Exception as e:  # noqa: BLE001
            res["errors"].append(str(e))
        if res["errors"]:
            res["status"] = "failed"
            log.error("postgres/%s errors: %s", name, res["errors"])
        results.append(res)
    return results


# ── Databend ─────────────────────────────────────────────────────────────
class _Databend:
    """Raw Databend HTTP query API (no Snowflake-dialect translation)."""

    def __init__(self):
        import requests
        self.url = (os.getenv("DATABEND_HTTP_URL") or "http://databend:8000").rstrip("/")
        self.http = requests.Session()
        self.http.auth = (os.getenv("DATABEND_USER", "databend"), os.getenv("DATABEND_PASSWORD", ""))

    def query(self, sql: str, timeout: int = 3600) -> list[list]:
        r = self.http.post(f"{self.url}/v1/query", json={"sql": sql, "pagination": {"wait_time_secs": 30}},
                           timeout=60)
        r.raise_for_status()
        body, rows, deadline = r.json(), [], time.time() + timeout
        while True:
            if body.get("error"):
                raise RuntimeError(f"databend: {body['error'].get('message', body['error'])}"[:2000])
            rows.extend(body.get("data") or [])
            nxt = body.get("next_uri")
            if not nxt:
                return rows
            if "/final" in nxt:  # all pages read; closing the query frees it server side
                self.http.get(self.url + nxt, timeout=60)
                return rows
            if time.time() > deadline:
                raise TimeoutError(f"databend query exceeded {timeout}s: {sql[:200]}")
            r = self.http.get(self.url + nxt, timeout=120)
            r.raise_for_status()
            body = r.json()


def _bq(ident: str) -> str:
    return "`" + ident.replace("`", "``") + "`"


def backup_databend(prefix: str, *, dry_run: bool = False) -> list[dict]:
    db_url = os.getenv("DATABEND_HTTP_URL") or "http://databend:8000"
    host = db_url.split("://", 1)[-1].split(":", 1)[0].split("/", 1)[0]
    if not _resolvable(host):
        return [_skipped("databend", "warehouse", f"host {host!r} not on this network")]
    res = {"engine": "databend", "target": "warehouse", "status": "ok", "objects": [], "errors": [], "tables": 0}
    try:
        bend = _Databend()
        tables = bend.query(
            "SELECT database, name, engine FROM system.tables "
            "WHERE database NOT IN ('system', 'information_schema') ORDER BY database, name")
        by_db: dict[str, list] = {}
        for db, name, engine in tables:
            by_db.setdefault(db, []).append((name, engine))
        key_id, secret = aws_keys()
        if not dry_run and not (key_id and secret):
            raise RuntimeError("BACKUP_AWS_ACCESS_KEY_ID/SECRET (or AWS_*) not set - Databend writes to S3 itself")
        conn = (f"CONNECTION = (ACCESS_KEY_ID = '{key_id}', SECRET_ACCESS_KEY = '{secret}', "
                f"REGION = '{REGION}')")
        for db, objs in by_db.items():
            ddl = []
            for name, engine in objs:
                fq = f"{_bq(db)}.{_bq(name)}"
                try:
                    # Databend has no SHOW CREATE VIEW; SHOW CREATE TABLE returns a view's CREATE VIEW too.
                    ddl.append(bend.query(f"SHOW CREATE TABLE {fq}")[0][1].rstrip(";") + ";")
                except Exception as e:  # noqa: BLE001
                    res["errors"].append(f"ddl {db}.{name}: {e}")
                if engine.upper() in ("VIEW", "STREAM") or engine.upper().startswith("SYSTEM"):
                    continue
                key_prefix = f"{prefix}/databend/{db}/{name}/"
                if dry_run:
                    log.info("[dry-run] COPY INTO s3://%s/%s FROM %s", BUCKET, key_prefix, fq)
                    continue
                try:
                    out = bend.query(f"COPY INTO 's3://{BUCKET}/{key_prefix}' {conn} FROM {fq} "
                                     "FILE_FORMAT = (TYPE = PARQUET) DETAILED_OUTPUT = TRUE")
                    for row in out:  # file_name (relative to the location), file_size, row_count
                        rel = str(row[0]).split(key_prefix, 1)[-1].lstrip("/")
                        res["objects"].append({"key": key_prefix + rel, "status": "ok",
                                               "bytes": int(row[1]), "rows": int(row[2])})
                    res["tables"] += 1
                except Exception as e:  # noqa: BLE001
                    res["errors"].append(f"data {db}.{name}: {str(e).replace(secret, '***') if secret else e}")
            res["objects"].append(_put_bytes(f"{prefix}/databend/{db}/_ddl.sql",
                                             ("\n\n".join(ddl) + "\n").encode(), dry_run=dry_run))
    except Exception as e:  # noqa: BLE001
        res["errors"].append(str(e))
    if res["errors"]:
        res["status"] = "failed"
        log.error("databend errors: %s", res["errors"][:20])
    return [res]


# ── Snowflake ────────────────────────────────────────────────────────────
def _snowflake_connect():
    import snowflake.connector
    kw = dict(user=os.getenv("SNOWFLAKE_USER", "").strip(), account=os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
              warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "").strip() or None,
              role=os.getenv("SNOWFLAKE_ROLE", "").strip() or None, login_timeout=60,
              session_parameters={"QUERY_TAG": SNOWFLAKE_QUERY_TAG, "TIMEZONE": "UTC"})
    key_path = (os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH") or "").strip()
    if key_path:
        kw["private_key_file"] = key_path
    elif os.getenv("SNOWFLAKE_PASSWORD"):
        kw["password"] = os.getenv("SNOWFLAKE_PASSWORD").strip()
    else:
        raise RuntimeError("set SNOWFLAKE_PRIVATE_KEY_PATH or SNOWFLAKE_PASSWORD")
    return snowflake.connector.connect(**kw)


def _sq(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'


def backup_snowflake(prefix: str, *, with_data: bool | None = None, dry_run: bool = False) -> list[dict]:
    required = os.getenv("BACKUP_SNOWFLAKE_REQUIRED", "").strip().lower() in ("1", "true", "yes")
    if with_data is None:
        with_data = dt.date.today().weekday() == SNOWFLAKE_DATA_WEEKDAY
    try:
        conn = _snowflake_connect()
    except Exception as e:  # noqa: BLE001 - e.g. 250001 "Access prevented by a system lock"
        if required:
            return [{"engine": "snowflake", "target": "account", "status": "failed", "objects": [],
                     "errors": [f"connect: {e}"]}]
        return [_skipped("snowflake", "account", f"cannot connect ({str(e)[:300]})")]

    res = {"engine": "snowflake", "target": "account", "status": "ok", "objects": [], "errors": [],
           "with_data": with_data, "tables": 0, "rows": 0}
    try:
        cur = conn.cursor()
        cur.execute("SHOW DATABASES")
        cols = [d[0] for d in cur.description]
        dbs = [r[cols.index("name")] for r in cur.fetchall()
               if r[cols.index("name")] not in SNOWFLAKE_SKIP_DATABASES
               and not (r[cols.index("origin")] if "origin" in cols else "")       # shares / imported
               and (r[cols.index("kind")] if "kind" in cols else "STANDARD") == "STANDARD"]
        res["databases"] = dbs
        for db in dbs:
            try:
                cur.execute(f"SELECT GET_DDL('DATABASE', '{_sq(db)}', TRUE)")
                res["objects"].append(_put_bytes(f"{prefix}/snowflake/{db}/_ddl.sql",
                                                 cur.fetchone()[0].encode(), dry_run=dry_run))
            except Exception as e:  # noqa: BLE001
                res["errors"].append(f"ddl {db}: {e}")
            if not with_data:
                continue
            cur.execute(f"SELECT table_schema, table_name FROM {_sq(db)}.INFORMATION_SCHEMA.TABLES "
                        "WHERE table_type = 'BASE TABLE' AND table_schema <> 'INFORMATION_SCHEMA' ORDER BY 1, 2")
            for schema, table in cur.fetchall():
                try:
                    res["objects"].extend(_snowflake_table(conn, db, schema, table, prefix, dry_run=dry_run))
                    res["tables"] += 1
                except Exception as e:  # noqa: BLE001
                    res["errors"].append(f"data {db}.{schema}.{table}: {e}")
        res["rows"] = sum(o.get("rows", 0) for o in res["objects"])
    except Exception as e:  # noqa: BLE001
        res["errors"].append(str(e))
    finally:
        conn.close()
    if res["errors"]:
        res["status"] = "failed"
        log.error("snowflake errors: %s", res["errors"][:20])
    return [res]


def _snowflake_table(conn, db, schema, table, prefix, *, dry_run) -> list[dict]:
    """Stream one table out as parquet parts of <= SNOWFLAKE_PART_ROWS rows."""
    import pyarrow.parquet as pq
    fq = f"{_sq(db)}.{_sq(schema)}.{_sq(table)}"
    base = f"{prefix}/snowflake/{db}/{schema}/{table}"
    if dry_run:
        log.info("[dry-run] SELECT * FROM %s -> s3://%s/%s/part-*.parquet", fq, BUCKET, base)
        return [{"key": f"{base}/", "status": "dry_run"}]
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {fq}")
    objects, part, writer, rows_in_part, tmp = [], 0, None, 0, None

    def flush():
        nonlocal writer, tmp, part, rows_in_part
        if writer is None:
            return
        writer.close()
        body = Path(tmp).read_bytes()
        obj = _put_bytes(f"{base}/part-{part:05d}.parquet", body, dry_run=False)
        obj["rows"] = rows_in_part
        objects.append(obj)
        os.unlink(tmp)
        writer, tmp, rows_in_part = None, None, 0
        part += 1

    try:
        for batch in cur.fetch_arrow_batches():
            if writer is None:
                fd, tmp = tempfile.mkstemp(suffix=".parquet", prefix="sfbackup-")
                os.close(fd)
                writer = pq.ParquetWriter(tmp, batch.schema, compression="zstd")
            writer.write_table(batch.cast(writer.schema) if batch.schema != writer.schema else batch)
            rows_in_part += batch.num_rows
            if rows_in_part >= SNOWFLAKE_PART_ROWS:
                flush()
        flush()
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    if not objects:  # empty table: keep a marker so a restore knows it existed
        objects.append(_put_bytes(f"{base}/_EMPTY", b"", dry_run=False))
    return objects


# ── manifest / verification ──────────────────────────────────────────────
def write_manifest(prefix: str, results: list[dict], *, dry_run: bool = False) -> dict:
    """Check every uploaded object is really there at the recorded size, write
    manifest.json, and return a summary. Raises if anything failed."""
    c = None if dry_run else s3()
    missing = []
    for r in results:
        for o in r.get("objects", []):
            if o.get("status") != "ok" or c is None:
                continue
            try:
                size = c.head_object(Bucket=BUCKET, Key=o["key"])["ContentLength"]
                if "bytes" in o and size != o["bytes"]:
                    missing.append(f"{o['key']}: size {size} != {o['bytes']}")
            except Exception as e:  # noqa: BLE001
                missing.append(f"{o['key']}: {e}")
    failed = [f"{r['engine']}/{r['target']}" for r in results if r.get("status") == "failed"]
    summary = {
        "bucket": BUCKET, "prefix": prefix, "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "objects": sum(len(r.get("objects", [])) for r in results),
        "bytes": sum(o.get("bytes", 0) for r in results for o in r.get("objects", [])),
        "ok": [f"{r['engine']}/{r['target']}" for r in results if r.get("status") == "ok"],
        "skipped": {f"{r['engine']}/{r['target']}": r.get("reason") for r in results if r.get("status") == "skipped"},
        "failed": failed, "missing_objects": missing,
    }
    _put_bytes(f"{prefix}/manifest.json",
               json.dumps({"summary": summary, "results": results}, indent=2, default=str).encode(), dry_run=dry_run)
    log.info("manifest: %s", json.dumps(summary, indent=2))
    if failed or missing:
        raise RuntimeError(f"backup incomplete - failed={failed} missing={missing[:20]} "
                           f"(see s3://{BUCKET}/{prefix}/manifest.json)")
    return summary


ENGINES = {"postgres": backup_postgres, "databend": backup_databend,
           "snowflake": backup_snowflake}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["setup-bucket", "check-bucket", "all", *ENGINES])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--snowflake-data", action="store_true", help="export Snowflake table data today")
    a = ap.parse_args()
    if a.what == "setup-bucket":
        print(json.dumps(setup_bucket(), indent=2))
        return
    if a.what == "check-bucket":
        print(json.dumps(check_bucket(), indent=2))
        return
    prefix = run_prefix()
    results = []
    for name in (ENGINES if a.what == "all" else [a.what]):
        kw = {"with_data": True} if (name == "snowflake" and a.snowflake_data) else {}
        results += ENGINES[name](prefix, dry_run=a.dry_run, **kw)
    write_manifest(prefix, results, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
