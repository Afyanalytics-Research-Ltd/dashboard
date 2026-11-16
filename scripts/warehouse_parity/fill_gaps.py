"""Careful, additive Snowflake -> Databend gap fill for the two HOSPITALS tables
the audit found short (prod, 2026-11-16).

  python fill_gaps.py            # dry run: report what would be copied, write nothing
  python fill_gaps.py --apply    # copy

Never drops, replaces or deletes anything in Databend:
  * ANALYTICS.AFYA_HMIS_PATIENT_COUNTS (append-only snapshots keyed by RUN_ID):
    add the columns Databend lacks (nullable), then append only Snowflake runs
    Databend doesn't have, ~BATCH_ROWS rows per batch, each batch verified
    per RUN_ID before the next. A run present in Databend with a different
    row count stops the script (no repair attempted).
  * REPORTING.CANONICAL_PRODUCT_TAXONOMY: created + loaded only if absent.
Throttle: pauses while host load1 > MAX_LOAD or MemAvailable < MIN_MEM_MB.
"""
import os, sys, time
sys.path.insert(0, "/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airflow_dashboard.settings")
import django
django.setup()
import snowflake.connector
from afya_backends.warehouse.dbapi import Connection
from afya_backends.warehouse.pandas_tools import write_pandas
from afya_backends.warehouse.sync import databend_type, _q
from core.warehouse_creds import snowflake_kwargs

APPLY = "--apply" in sys.argv
BATCH_ROWS, MAX_LOAD, MIN_MEM_MB, PAUSE = 20_000, 2.5, 1024, 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wait_for_headroom():
    while True:
        load1 = float(open("/proc/loadavg").read().split()[0])
        mem = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
        if load1 <= MAX_LOAD and mem >= MIN_MEM_MB:
            return load1, mem
        log("  throttle: load1=%.2f mem_avail=%dMB - waiting 30s" % (load1, mem))
        time.sleep(30)


kw = snowflake_kwargs("SNOWFLAKE_ADMIN_"); kw.pop("schema", None)
sf = snowflake.connector.connect(**kw, login_timeout=60, session_parameters={
    "QUERY_TAG": "databend-gap-fill", "STATEMENT_TIMEOUT_IN_SECONDS": 300})
c = sf.cursor()
db = Connection(database="HOSPITALS")
run = db._t.run
log("mode:", "APPLY" if APPLY else "DRY RUN")

# ── 1. AFYA_HMIS_PATIENT_COUNTS ────────────────────────────────────────────
SFT = "HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS"
DBT = '"ANALYTICS"."AFYA_HMIS_PATIENT_COUNTS"'
sf_cols = c.execute("SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA='ANALYTICS' AND TABLE_NAME='AFYA_HMIS_PATIENT_COUNTS' ORDER BY ORDINAL_POSITION").fetchall()
db_cols = [r[0] for r in run("SELECT name FROM system.columns WHERE database='ANALYTICS' AND table='AFYA_HMIS_PATIENT_COUNTS'").rows]
missing_cols = [col for col in sf_cols if col[0] not in db_cols]
for name, dt, p, s in missing_cols:
    ddl = "ALTER TABLE %s ADD COLUMN %s %s NULL" % (DBT, _q(name), databend_type(dt, p, s).replace(" NULL", ""))
    log("column:", ddl)
    if APPLY:
        run(ddl)

sf_runs = dict(c.execute(f"SELECT RUN_ID, COUNT(*) FROM {SFT} GROUP BY 1").fetchall())
db_runs = {r[0]: int(r[1]) for r in run(f'SELECT "RUN_ID", COUNT(*) FROM {DBT} GROUP BY 1').rows}
bad = [(r, sf_runs[r], db_runs[r]) for r in sf_runs if r in db_runs and db_runs[r] != sf_runs[r]]
if bad:
    sys.exit("STOP: runs present in both with different row counts (not touching): %s" % bad[:5])
todo = sorted(r for r in sf_runs if r not in db_runs)
log("patient counts: snowflake runs=%d, databend has %d of them, to copy=%d runs / %d rows"
    % (len(sf_runs), len(sf_runs) - len(todo), len(todo), sum(sf_runs[r] for r in todo)))

batches, cur_b, n = [], [], 0
for r in todo:
    if cur_b and n + sf_runs[r] > BATCH_ROWS:
        batches.append(cur_b); cur_b, n = [], 0
    cur_b.append(r); n += sf_runs[r]
if cur_b:
    batches.append(cur_b)

copied = 0
if APPLY:
    for i, b in enumerate(batches, 1):
        load1, mem = wait_for_headroom()
        df = c.execute(f"SELECT * FROM {SFT} WHERE RUN_ID IN (%s)" % ",".join(["%s"] * len(b)), b).fetch_pandas_all()
        write_pandas(db, df, "AFYA_HMIS_PATIENT_COUNTS", database="HOSPITALS", schema="ANALYTICS", quote_identifiers=True)
        got = {r[0]: int(r[1]) for r in run(f'SELECT "RUN_ID", COUNT(*) FROM {DBT} WHERE "RUN_ID" IN (%s) GROUP BY 1'
                                            % ",".join("'%s'" % x.replace("'", "''") for x in b)).rows}
        wrong = [(r, sf_runs[r], got.get(r)) for r in b if got.get(r) != sf_runs[r]]
        if wrong:
            sys.exit("STOP: batch %d verification failed: %s" % (i, wrong[:5]))
        copied += len(df)
        log("  batch %d/%d: %d runs, %d rows ok (total %d) load1=%.2f mem=%dMB" % (i, len(batches), len(b), len(df), copied, load1, mem))
        del df
        time.sleep(PAUSE)
else:
    log("would copy in %d batches of <= %d rows" % (len(batches), BATCH_ROWS))

# ── 2. CANONICAL_PRODUCT_TAXONOMY (only if absent) ─────────────────────────
exists = run("SELECT COUNT(*) FROM system.tables WHERE database='REPORTING' AND name='CANONICAL_PRODUCT_TAXONOMY'").rows[0][0]
if int(exists):
    log("taxonomy: already in Databend - not touching")
else:
    cols = c.execute("SELECT COLUMN_NAME, DATA_TYPE, NUMERIC_PRECISION, NUMERIC_SCALE FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS "
                     "WHERE TABLE_SCHEMA='REPORTING' AND TABLE_NAME='CANONICAL_PRODUCT_TAXONOMY' ORDER BY ORDINAL_POSITION").fetchall()
    ddl = 'CREATE TABLE "REPORTING"."CANONICAL_PRODUCT_TAXONOMY" (%s)' % ", ".join(
        "%s %s" % (_q(x[0]), databend_type(x[1], x[2], x[3])) for x in cols)
    sf_n = c.execute("SELECT COUNT(*) FROM HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY").fetchone()[0]
    log("taxonomy:", ddl[:160], "... rows:", sf_n)
    if APPLY:
        wait_for_headroom()
        run(ddl)
        df = c.execute("SELECT * FROM HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY").fetch_pandas_all()
        write_pandas(db, df, "CANONICAL_PRODUCT_TAXONOMY", database="HOSPITALS", schema="REPORTING", quote_identifiers=True)
        db_n = int(run('SELECT COUNT(*) FROM "REPORTING"."CANONICAL_PRODUCT_TAXONOMY"').rows[0][0])
        log("taxonomy: snowflake=%d databend=%d %s" % (sf_n, db_n, "OK" if db_n == sf_n else "MISMATCH"))

sf.close(); db.close()
log("done")
