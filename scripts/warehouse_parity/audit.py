"""Read-only Snowflake vs Databend inventory audit (metadata only).

Snowflake: SHOW DATABASES / SHOW TABLES / SHOW VIEWS - served by cloud
services, no warehouse resumed, no table scans.
Databend: system.tables - catalog metadata, no table scans.
Writes nothing to either warehouse. Output: JSON to argv[1], summary on stdout.
"""
import json, os, sys, time
sys.path.insert(0, "/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airflow_dashboard.settings")
import django
django.setup()

import snowflake.connector
from afya_backends.warehouse import sql_translate as tr
from afya_backends.warehouse.dbapi import Connection
from core.warehouse_creds import snowflake_kwargs

SKIP_DBS = {"SNOWFLAKE", "SNOWFLAKE_SAMPLE_DATA"}
primary = tr.default_primary_database()

kw = snowflake_kwargs("SNOWFLAKE_ADMIN_")
kw.pop("warehouse", None)            # SHOW commands need no warehouse - never resume one
kw.pop("schema", None)
sf = snowflake.connector.connect(**kw, login_timeout=60, session_parameters={"QUERY_TAG": "databend-audit-readonly"})
cur = sf.cursor(snowflake.connector.DictCursor)

cur.execute("SHOW DATABASES")
dbs = [r for r in cur.fetchall() if r["name"] not in SKIP_DBS]
sf_objs = []
for d in dbs:
    if d.get("kind") not in (None, "STANDARD") or d.get("origin"):
        sf_objs.append({"db": d["name"], "note": "skipped: %s %s" % (d.get("kind"), d.get("origin"))})
        continue
    cur.execute('SHOW TABLES IN DATABASE "%s"' % d["name"])
    for r in cur.fetchall():
        if r["schema_name"] == "INFORMATION_SCHEMA":
            continue
        sf_objs.append({"db": d["name"], "schema": r["schema_name"], "name": r["name"], "type": "TABLE",
                        "kind": r.get("kind"), "rows": r.get("rows"), "bytes": r.get("bytes"),
                        "external": r.get("is_external"), "last_altered": str(r.get("last_altered") or "")})
    cur.execute('SHOW VIEWS IN DATABASE "%s"' % d["name"])
    for r in cur.fetchall():
        if r["schema_name"] == "INFORMATION_SCHEMA":
            continue
        sf_objs.append({"db": d["name"], "schema": r["schema_name"], "name": r["name"], "type": "VIEW",
                        "materialized": r.get("is_materialized")})
sf.close()

db = Connection()
res = db._t.run("SELECT database, name, engine, num_rows FROM system.tables "
                "WHERE database NOT IN ('system','information_schema','default')")
dbt = {}
for database, name, engine, num_rows in res.rows:
    dbt[(database.upper(), name.upper())] = {"database": database, "name": name, "engine": engine,
                                             "rows": int(num_rows) if num_rows not in (None, "", "NULL") else None}
try:
    res = db._t.run("SELECT database, name FROM system.views WHERE database NOT IN ('system','information_schema')")
    for database, name in res.rows:
        dbt.setdefault((database.upper(), name.upper()), {"database": database, "name": name, "engine": "VIEW", "rows": None})
except Exception as e:
    print("system.views not available:", str(e)[:100])
db.close()

report = []
matched = set()
for o in sf_objs:
    if "note" in o:
        report.append(o); continue
    target_db = tr.map_database(o["db"], o["schema"], primary)
    key = (target_db.upper(), o["name"].upper())
    d = dbt.get(key)
    if d:
        matched.add(key)
    st = "missing"
    if d:
        if o["type"] == "VIEW":
            st = "present"
        elif d["rows"] is None:
            st = "present(rows?)"
        elif o["rows"] == d["rows"]:
            st = "equal"
        elif (o["rows"] or 0) > d["rows"]:
            st = "databend_fewer"
        else:
            st = "databend_more"
    report.append(dict(o, target_db=target_db, databend=d, status=st))
extra = [v for k, v in dbt.items() if k not in matched]

json.dump({"at": time.time(), "snowflake": report, "databend_only": extra}, open(sys.argv[1], "w"), indent=1, default=str)

from collections import Counter
print("snowflake objects:", sum(1 for r in report if "status" in r), "| databend objects:", len(dbt))
for r in report:
    if "note" in r: print("  ", r["db"], r["note"])
print("status counts:", dict(Counter((r["type"], r["status"]) for r in report if "status" in r)))
for st in ("missing", "databend_fewer", "databend_more", "present(rows?)"):
    rows = [r for r in report if r.get("status") == st]
    if rows:
        print("\n== %s (%d)" % (st, len(rows)))
        for r in sorted(rows, key=lambda r: -(r.get("rows") or 0))[:60]:
            print("   %-60s %-6s sf_rows=%-12s db_rows=%s" % ("%s.%s.%s" % (r["db"], r["schema"], r["name"]), r["type"],
                  r.get("rows"), (r["databend"] or {}).get("rows")))
        if len(rows) > 60: print("   ... %d more" % (len(rows) - 60))
print("\nDatabend-only objects:", len(extra))
