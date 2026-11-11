# Platform backups: what, where, and how to restore

DAG `platform_backup_to_s3` (`dags/platform_backup_to_s3_dag.py`) runs every night at 01:00 EAT and calls `pipelines/platform_backup.py`.

| | |
|---|---|
| Bucket | `s3://afya-platform-backups-707559309638` (eu-north-1) |
| Protection | Public access blocked, SSE-S3 (AES256) encryption, versioning on |
| Retention | Lifecycle rule deletes objects after **30 days**; overwritten versions after 7 days |
| Layout | `<source>/<YYYY-MM-DD>/{postgres,databend,snowflake}/…` plus `manifest.json` |

`<source>` is `BACKUP_SOURCE_LABEL`, default `datahub`. Set a different label on each machine that runs the DAG (for example `datahub` for the prod EC2 host and `wsl-dev` for a laptop) so their backups don't mix.

## What gets backed up

| Engine | Targets | Format | Frequency |
|---|---|---|---|
| Postgres | django (`postgres-container`), airflow, airflow-legacy (`airflow-db`), redash, superset, mlflow, marquez: every non-template database, plus roles (`_globals.sql.gz`) | `pg_dump -Fc` | daily |
| Databend | every table (Parquet, written by Databend itself with `COPY INTO`) plus `_ddl.sql` for each database | Parquet + SQL | daily |
| Snowflake | every user database except SNOWFLAKE and SAMPLE_DATA: `GET_DDL` (all objects) plus each table as Parquet parts | SQL + Parquet | DDL daily; data on **Sundays** |

A target whose host doesn't exist on the machine running the DAG is **skipped**, not failed. For example, `postgres-container` exists only on prod. `manifest.json` lists exactly what was `ok`, `skipped` or `failed`. If anything failed, or an uploaded object is missing or the wrong size, the `manifest` task fails the run.

Snowflake is **skipped** while the account refuses connections (since 2026-11-10: first the system lock, then `JWT token is invalid`). To make that fail the run instead, set `BACKUP_SNOWFLAKE_REQUIRED=1`. Its queries carry `QUERY_TAG = 'platform_backup'`, so they are easy to identify in QUERY_HISTORY.

## Manual runs

```bash
# inside the Airflow worker (repo pipelines/ is mounted at /opt/airflow/pipelines)
python platform_backup.py check-bucket
python platform_backup.py all --dry-run          # list what would be uploaded
python platform_backup.py postgres               # one engine
python platform_backup.py snowflake --snowflake-data   # include table data today
python platform_backup.py setup-bucket           # re-apply bucket settings (idempotent)
```

You can also trigger the DAG from the UI with params `engines`, `snowflake_data` (auto/yes/no) and `dry_run`.

A manual single-engine run rewrites that day's `manifest.json` with only that engine's results. The earlier version is kept, because versioning is on.

## Restore

```bash
B=s3://afya-platform-backups-707559309638/datahub/2026-11-11
```

**Postgres:**
```bash
aws s3 cp $B/postgres/redash/_globals.sql.gz - | gunzip | psql -h HOST -U postgres          # roles first (optional)
aws s3 cp $B/postgres/redash/redash.dump /tmp/redash.dump
createdb -h HOST -U USER redash_restored
pg_restore -h HOST -U USER -d redash_restored --no-owner --jobs 4 /tmp/redash.dump
```
Restore into a new database first and check it, then swap names.

**Databend:**
```sql
-- 1. recreate objects: run databend/<DB>/_ddl.sql
-- 2. reload each table
COPY INTO "DB"."TABLE" FROM 's3://afya-platform-backups-707559309638/datahub/2026-11-11/databend/DB/TABLE/'
  CONNECTION = (ACCESS_KEY_ID = '…', SECRET_ACCESS_KEY = '…', REGION = 'eu-north-1')
  FILE_FORMAT = (TYPE = PARQUET);
```

**Snowflake:**
```sql
-- 1. run snowflake/<DB>/_ddl.sql (recreates schemas, tables, views, procs…)
-- 2. stage the parquet parts and load
CREATE STAGE restore_stg URL = 's3://afya-platform-backups-707559309638/datahub/<DATE>/snowflake/DB/'
  STORAGE_INTEGRATION = <integration>  FILE_FORMAT = (TYPE = PARQUET);
COPY INTO DB.SCHEMA.TABLE FROM @restore_stg/SCHEMA/TABLE/
  MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;
```
For anything changed or dropped in the last 1–90 days, Snowflake Time Travel (`UNDROP`, `AT(TIMESTAMP => …)`) is faster than a restore from S3.

## Not covered (yet)

- **MySQL** (the local MySQL 8 on the Windows host: tenri, hospital_db, kakamega, lodwar, platform): excluded by decision on 2026-11-11.
- **Docker volumes that aren't databases**: n8n workflows (`dashboard-n8n_n8n-data`, SQLite), wandb (`dashboard_wandb-data`), Databend raw storage. These need host-level access to Docker, which the Airflow worker deliberately doesn't have.
- **`.env` files and key files**: deliberately excluded. Keep secrets in a password manager or secrets manager, not in a backup bucket.
- **Object Lock (WORM)**: not enabled. Versioning protects against overwrites, but anyone with delete-version rights can still purge backups. Consider a separate AWS account or Object Lock for ransomware-grade protection.
