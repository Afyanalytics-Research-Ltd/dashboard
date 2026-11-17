"""Step 2 of the careful prod Snowflake -> Databend fill (2026-11-16).

  python fill2.py            # dry run (translate only, write nothing)
  python fill2.py --apply

A. HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY -> Databend REPORTING.CANONICAL_PRODUCT_TAXONOMY
   (table already created empty by fill_gaps.py; loads only while it has 0 rows).
   LOADED_AT is corrupt in Snowflake (epoch microseconds stored as seconds ->
   year 56,560,319); per the user it is loaded decoded: epoch_second / 1e6.
B. HOSPITALS views -> Databend views, via the Snowflake->Databend translator.
   CREATE VIEW IF NOT EXISTS only; each new view compile-checked with
   SELECT * ... LIMIT 0; a view that doesn't compile is dropped again (only
   views created by this run) and retried in later passes (dependency order).
   Never materialised. Untranslatable views are listed.
"""
import json, os, re, sys, time
sys.path.insert(0, "/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airflow_dashboard.settings")
import django
django.setup()
import snowflake.connector
from afya_backends.warehouse import sql_translate as tr
from afya_backends.warehouse.dbapi import Connection
from afya_backends.warehouse.pandas_tools import write_pandas
from core.warehouse_creds import snowflake_kwargs

APPLY = "--apply" in sys.argv
MAX_LOAD, MIN_MEM_MB = 2.5, 1024


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wait_for_headroom():
    while True:
        load1 = float(open("/proc/loadavg").read().split()[0])
        mem = next(int(l.split()[1]) // 1024 for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
        if load1 <= MAX_LOAD and mem >= MIN_MEM_MB:
            return load1
        log("  throttle: load1=%.2f mem=%dMB - waiting 30s" % (load1, mem))
        time.sleep(30)


kw = snowflake_kwargs("SNOWFLAKE_ADMIN_"); kw.pop("schema", None)
sf = snowflake.connector.connect(**kw, login_timeout=60, session_parameters={
    "QUERY_TAG": "databend-gap-fill", "STATEMENT_TIMEOUT_IN_SECONDS": 300})
c = sf.cursor()
db = Connection(database="HOSPITALS"); run = db._t.run
log("mode:", "APPLY" if APPLY else "DRY RUN")

# ── A. taxonomy ────────────────────────────────────────────────────────────
n_db = int(run('SELECT COUNT(*) FROM "REPORTING"."CANONICAL_PRODUCT_TAXONOMY"').rows[0][0])
cols = [r[0] for r in c.execute("SELECT COLUMN_NAME FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='REPORTING' "
                                "AND TABLE_NAME='CANONICAL_PRODUCT_TAXONOMY' ORDER BY ORDINAL_POSITION").fetchall()]
sel = ", ".join("TO_TIMESTAMP_NTZ(DATE_PART(EPOCH_SECOND, LOADED_AT) / 1000000) AS LOADED_AT" if x == "LOADED_AT" else '"%s"' % x for x in cols)
sf_n = c.execute("SELECT COUNT(*) FROM HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY").fetchone()[0]
if n_db:
    log("taxonomy: Databend already has %d rows - not touching" % n_db)
elif APPLY:
    wait_for_headroom()
    df = c.execute("SELECT %s FROM HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY" % sel).fetch_pandas_all()
    write_pandas(db, df, "CANONICAL_PRODUCT_TAXONOMY", database="HOSPITALS", schema="REPORTING", quote_identifiers=True)
    r = run('SELECT COUNT(*), MIN("LOADED_AT"), MAX("LOADED_AT"), COUNT(DISTINCT "PRODUCT_ID") FROM "REPORTING"."CANONICAL_PRODUCT_TAXONOMY"').rows[0]
    s = c.execute("SELECT COUNT(DISTINCT PRODUCT_ID) FROM HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY").fetchone()[0]
    log("taxonomy: snowflake=%d databend=%s loaded_at=%s..%s distinct product_id sf=%d db=%s %s"
        % (sf_n, r[0], r[1], r[2], s, r[3], "OK" if int(r[0]) == sf_n and int(r[3]) == s else "MISMATCH"))
else:
    log("taxonomy: would load %d rows (LOADED_AT decoded)" % sf_n)

# ── B. views ───────────────────────────────────────────────────────────────
existing = {(d.upper(), n.upper()) for d, n in run("SELECT database, name FROM system.tables").rows}
views = c.execute("SELECT TABLE_SCHEMA, TABLE_NAME, VIEW_DEFINITION FROM HOSPITALS.INFORMATION_SCHEMA.VIEWS "
                  "WHERE TABLE_SCHEMA <> 'INFORMATION_SCHEMA' ORDER BY 1, 2").fetchall()
sf.close()
primary = tr.default_primary_database()
plan, untranslatable = [], []
for schema, name, definition in views:
    target_db = tr.map_database("HOSPITALS", schema, primary)
    if (target_db.upper(), name.upper()) in existing:
        continue
    if not definition:
        untranslatable.append((schema, name, "no definition visible")); continue
    try:
        ctx = tr.Context(database="HOSPITALS", schema=schema, primary_database=primary)
        stmts = [s for s in tr.translate(definition, ctx) if s.action == "sql"]
        if len(stmts) != 1 or not re.match(r"\s*CREATE\b", stmts[0].sql, re.I):
            raise ValueError("translated to %d statements" % len(stmts))
        sql = re.sub(r"^\s*CREATE\s+(OR\s+REPLACE\s+)?(SECURE\s+)?(RECURSIVE\s+)?VIEW\s+(IF\s+NOT\s+EXISTS\s+)?",
                     "CREATE VIEW IF NOT EXISTS ", stmts[0].sql, count=1, flags=re.I)
        if not sql.upper().startswith("CREATE VIEW IF NOT EXISTS"):
            raise ValueError("unexpected DDL: " + sql[:80])
        plan.append((schema, name, target_db, sql))
    except Exception as e:
        untranslatable.append((schema, name, str(e)[:150]))
log("views: %d in Snowflake, %d to create, %d untranslatable" % (len(views), len(plan), len(untranslatable)))

dbs_now = {r[0].upper() for r in run("SELECT name FROM system.databases").rows}
need_dbs = sorted({p[2] for p in plan if p[2].upper() not in dbs_now})
log("databases to create (schemas with only views):", need_dbs)
if APPLY:
    for d in need_dbs:
        run('CREATE DATABASE IF NOT EXISTS "%s"' % d.replace('"', ''))

created, failed = [], {}
if APPLY:
    pending = plan
    for p in range(1, 6):
        nxt = []
        for schema, name, target_db, sql in pending:
            wait_for_headroom()
            ref = '"%s"."%s"' % (target_db, name)
            try:
                run(sql)
            except Exception as e:
                failed[(schema, name)] = "create: " + str(e)[:200]; nxt.append((schema, name, target_db, sql)); continue
            try:
                run("SELECT * FROM %s LIMIT 0" % ref)
                created.append((schema, name)); failed.pop((schema, name), None)
            except Exception as e:
                failed[(schema, name)] = "compile: " + str(e)[:200]
                run("DROP VIEW IF EXISTS %s" % ref)
                nxt.append((schema, name, target_db, sql))
            time.sleep(0.2)
        log("  pass %d: created so far %d, still failing %d" % (p, len(created), len(nxt)))
        if not nxt or len(nxt) == len(pending):
            break
        pending = nxt

out = {"created": created, "failed": {"%s.%s" % k: v for k, v in failed.items()},
       "untranslatable": untranslatable, "planned": len(plan)}
json.dump(out, open("/tmp/fill2_result.json", "w"), indent=1)
log("created=%d failed=%d untranslatable=%d" % (len(created), len(failed), len(untranslatable)))
from collections import Counter
print("failure reasons (top):", Counter(re.sub(r"[\"'`].*?[\"'`]", "?", v)[:90] for v in failed.values()).most_common(8))
print("untranslatable (top):", Counter(r[2][:90] for r in untranslatable).most_common(8))
db.close()
log("done")
