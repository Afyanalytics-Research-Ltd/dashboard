# Connect Studio

Fivetran/Airbyte-style data movement for Afya DataHub, at **`/connect/`** (superusers):

1. **Sources**: connect a database, files, a REST API or the Afya Extraction API, or upload a target schema.
2. **Schema extraction**: [dbatlas](https://github.com/jordansgoodman/dbatlas) introspects tables, keys and
   constraints, profiles a row sample (in DuckDB) and infers relationships nobody declared. Every extraction is a
   versioned snapshot; re-extracting shows schema drift.
3. **AI mapping**: a source snapshot is mapped onto a target snapshot. Rule-based matching runs first, then Groq
   (`openai/gpt-oss-120b`) matches tables and writes column rules. Every suggestion is validated against the real
   snapshots before anyone sees it.
4. **Review**: in the mapping studio, source columns are wired to target columns. You can drag and drop to map,
   edit rules in the inspector with a live preview on sampled values, approve with keyboard shortcuts, and export
   **mappings.json**.
5. **Syncs**: mapping + destination + mode (full refresh / incremental, append / deduped) + schedule.

## Layout

```
connect_studio/
  engine/            framework-free; usable from Airflow or a script
    connectors.py    declarative connector registry (specs → UI forms), SQL/staged/schema connectors, SQL writer
    snapshot.py      normalized snapshot format; dbatlas ↔ snapshot; schema.json / JSON Schema / DDL / mappings.json parsers; diff
    mapper.py        candidates → Groq table matching → Groq field mapping → validation
    rules.py         the rule grammar (same as pipelines/from_json_mappings_to_snowflake.py) + apply/validate
    document.py      coverage summary; export (generic | afya) + loader shape validation
    sync.py          source rows → rules → typed destination rows; modes + cursor state
    llm.py           Groq JSON calls: retries, rate-limit hints, token accounting
  models.py          Source, SourceFile, Snapshot, Mapping, Connection, Job
  services.py        jobs (test/extract/map/remap/sync), serializers, scheduler
  jobs.py            background threads that stream progress + logs into Job rows
  api.py / urls.py   JSON API under /connect/api/
static/connect_studio/   single-page UI (shared ui/js router + components, own design system)
templates/connect_studio/app.html
```

## Connectors

| Family | Connectors |
|---|---|
| SQL (SQLAlchemy) | PostgreSQL, MySQL, MariaDB, SQL Server*, Oracle*, Redshift, Snowflake, SQLite, DuckDB, any SQLAlchemy URL |
| Staged → DuckDB | Files (CSV/TSV/Excel/JSON/NDJSON/Parquet), REST API (declarative streams, auth incl. login requests, pagination, POST-body reads), Afya Extraction API |
| Afya Platform (V3) | `{service}.afyaanalytics.ai` gateways: structure from `describe` (types, PKs, FKs, enums, comments; complete even for empty models), optional row samples for profiles. Source and target |
| SQL dump → DuckDB | `.sql` scripts from MySQL / MariaDB, PostgreSQL, SQL Server, SQLite and Oracle — plain, `.gz` / `.bz2` / `.xz` or zipped (see below) |
| Schema only | schema.json (many shapes), JSON Schema, SQL DDL (any sqlglot dialect), dbatlas.json, mappings.json (target side) |

\* needs a driver that isn't installed here (`pyodbc` + msodbcsql18, `oracledb`). The catalog shows "driver missing".

SQL connectors can also be **sync destinations**. To add a connector, call `register(ConnectorSpec(...), cls)` in
`engine/connectors.py`. The UI builds its form from the spec: groups, advanced fields, `show_if` and secrets.

### SQL dumps

`engine/sqldump.py` restores a dump without its database server: it streams the script, splits statements with
each engine's quoting rules, and loads every table into `work/<source>/dump.duckdb`. From there the source is an
ordinary DuckDB database (test, dbatlas extraction, full-table reads for syncs).

| Engine | What it reads |
|---|---|
| MySQL / MariaDB | mysqldump and phpMyAdmin: backtick names, `\` escapes, extended INSERTs, `/*!…*/` code comments, keys added later with `ALTER TABLE`, `DELIMITER` blocks, `_binary` / `X''` values |
| PostgreSQL | pg_dump plain: `COPY … FROM stdin` or `--inserts`, `E''` strings, `$$` bodies, `\connect` lines, `COMMENT ON`. Custom / tar format (`-Fc`, `-Ft`) goes through `pg_restore` (installed in the web image) |
| SQL Server | SSMS "Generate Scripts" / sqlcmd: UTF-16, `GO` batches, statements without `;`, `[dbo].[x]`, `N''`, `CAST(N'…' AS DateTime)` and the old binary `CAST(0x… AS DateTime)`, `sp_addextendedproperty` descriptions |
| SQLite | `.dump` (its `sqlite_sequence` bookkeeping is skipped) |
| Oracle | SQL Developer exports: `TO_DATE` / `TO_TIMESTAMP` with their format masks, `/` terminators, PL/SQL blocks skipped |

- Every table is loaded as text first, then each column gets its declared type if every value converts.
  `0000-00-00` becomes NULL. Columns that don't convert stay text, and the warning names the column with sample values.
- The DuckDB copy has no constraints, because dumps routinely break their own keys. Declared primary keys, foreign
  keys, uniques, comments, defaults and enum values go into `dump.meta.json` and from there into the snapshot.
- Routines, triggers and views are skipped and counted. So are `INSERT … SELECT` statements and system tables.
- **Not supported, with a clear error:** SQL Server `.bak` and Oracle `.dmp` (binary backups: script them out),
  and SQLite database files (use the SQLite connector).
- **Re-use and rebuilds:** a restore is reused until the uploads or the restore options change. Options are engine,
  tables, rows per table and typing. A file lock keeps parallel jobs from restoring the same source twice.
- **Large dumps:** upload them directly. Chunked uploads go up to 20 GB, and compressing first (~10× smaller) only
  makes it faster. A file already on the server works too, via *…or a dump already on the server*.

### Storage and clean-up

With `CONNECT_STUDIO_S3_BUCKET` set, S3 is the only durable copy of an upload, and the server's disk is scratch
space for one job at a time:

- **Uploads:** files of any size, up to `CONNECT_STUDIO_MAX_UPLOAD_GB`, go up in 16 MB chunks as an S3 multipart
  upload, so they never touch the server's disk.
  - The browser sends 3 chunks at a time. Each carries a SHA-256 that the server checks, and S3 checks the part's
    Content-MD5.
  - A failed chunk is retried with backoff. If it keeps failing, the upload stops and the error says so.
  - Choosing the same file again (same name, size and modified time) resumes the upload and sends only the missing
    chunks.
  - API: `POST sources/<id>/uploads/` → `PUT uploads/<uuid>/chunks/<n>/` → `POST uploads/<uuid>/` (`DELETE` aborts).
- **Jobs** (test, extract, sync, and the mapping lookup preview) run in `work/<source>/<job>-…/`, via
  `services.opened`.
  - The job downloads the uploads it needs into that folder, and the folder is deleted when the job ends, whether it
    succeeded, failed or was cancelled.
  - A restored SQL dump is cached in S3, under `…/sources/<id>/cache/`, so the next job downloads it instead of
    parsing the dump again.
  - An uploaded SQLite or DuckDB file used as a sync destination is uploaded back after a successful sync.
- **Janitor:** runs every 15 min in the scheduler, or with `manage.py connect_studio_cleanup`. It deletes workspaces
  idle for more than 6 h (left by a crashed job) and uploads abandoned for more than 24 h.
- **Moving older uploads:** `connect_studio_cleanup --move-to-s3` moves uploads made before S3 was configured into
  S3 and deletes the local copies.
- **Free up space** (source page, `POST sources/<id>/free-space/`) deletes the uploads, unfinished uploads and the S3
  cache. Snapshots and mappings are kept. **Deleting a source** removes all of that too.
- Uploading or removing a file clears the cache, because its contents belonged to the old files.
- None of this touches data a sync has already written to a destination, and all of it is refused while a job is
  using the source.

Without a bucket, uploads stay in the media folder. Jobs still use a scratch workspace that is deleted afterwards,
but there's no cache, so every job restores the dump again.

### Importing Postman / OpenAPI

On a new REST API source, drop a Postman collection (v2.0/2.1) or an OpenAPI 3 / Swagger 2 file (JSON or YAML).
`engine/api_import.py` reads the base URL, auth (a login request becomes `auth_type=login`), pagination, record
paths and endpoints. Then Groq (`ai_refine`) chooses the endpoints that are entity tables and names them. Its answer
is validated: only known endpoint ids are accepted, a body can only lose keys, and a pagination setting needs its
parameter names. You review the picker, then **Apply to form**. The same parser runs through
`POST /connect/api/import-api/`.

## Configuration

| Env | Default | |
|---|---|---|
| `GROQ_API_KEY` / `GROQ_MODEL` | (platform) | AI mapping. Without a key the mapper runs rule-based only |
| `CONNECT_STUDIO_REASONING_EFFORT` | `medium` | gpt-oss reasoning effort for mapping calls |
| `CONNECT_STUDIO_KEY` | derived from `SECRET_KEY` | Fernet key for stored secrets (passwords/tokens/keys) |
| `CONNECT_STUDIO_SCHEDULER` | `1` | in-process scheduler for scheduled syncs (web server only) |
| `AFYA_EXTRACTION_*` | | default credentials for the Afya Extraction API connector |
| `CONNECT_STUDIO_S3_BUCKET` | (unset = local) | private bucket for uploads and cached restores (see *Storage*) |
| `CONNECT_STUDIO_S3_PREFIX` | `connect-studio` | key prefix in the bucket |
| `CONNECT_STUDIO_S3_REGION` / `_ENDPOINT_URL` | SDK default | region; an S3-compatible endpoint for local testing |
| `CONNECT_STUDIO_AWS_ACCESS_KEY_ID` / `_SECRET_ACCESS_KEY` | AWS_* / instance role | Connect Studio's own key (IAM user `connect-studio-<env>`); needs `s3:PutObject`, `GetObject`, `DeleteObject`, `ListBucket`, `AbortMultipartUpload`, `ListMultipartUploadParts` on the bucket / prefix |
| `CONNECT_STUDIO_MAX_UPLOAD_GB` | `20` | largest single upload |

Scheduled syncs claim their slot atomically, so several web workers never double-run one. With the in-process
scheduler off, run `python manage.py connect_studio_run_due` from cron or Airflow.

## mappings.json

Export from the studio (**Export ▸ …**) or `GET /connect/api/mappings/<id>/export/?flavor=generic|afya&download=1`.
The `afya` flavor has the exact shape of `dags/jsons/*_data_mappings.json`, so
`pipelines/from_json_mappings_to_snowflake.py` runs it unchanged. `constant` rules (which that loader doesn't
support) go to `special_actions`.

## Tests

```
python manage.py test connect_studio
```

These need no network: Groq is mocked, and the tests use SQLite/DuckDB/files plus a mocked REST API.
