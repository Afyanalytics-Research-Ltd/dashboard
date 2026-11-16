"""Copy HOSPITALS tables that exist in Snowflake but not in Databend (only if absent)."""
import os, sys
sys.path.insert(0, "/")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "airflow_dashboard.settings")
import django
django.setup()
import snowflake.connector
from afya_backends.warehouse.dbapi import Connection
from afya_backends.warehouse.sync import Syncer
from core.warehouse_creds import snowflake_kwargs

TABLES = [("REPORTING", "RPT_KSH_V3_OPD_STATION_TAT")]
kw = snowflake_kwargs("SNOWFLAKE_ADMIN_"); kw.pop("schema", None)
sf = snowflake.connector.connect(**kw, login_timeout=60, session_parameters={"QUERY_TAG": "databend-gap-fill"})
db = Connection(database="HOSPITALS")
s = Syncer(sf, db, database="HOSPITALS", state_file="/tmp/fill3_state.json", log=print)
for schema, table in TABLES:
    n = int(db._t.run("SELECT COUNT(*) FROM system.tables WHERE database='%s' AND name='%s'" % (schema, table)).rows[0][0])
    if n:
        print(schema, table, "already exists in Databend - not touching"); continue
    s.sync_table(schema, table, schema)   # verifies Snowflake vs Databend row count
sf.close(); db.close()
