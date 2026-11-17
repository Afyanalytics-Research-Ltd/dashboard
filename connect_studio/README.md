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
- **Large dumps:** compress them (they shrink ~10×) to stay under the 100 MB upload limit, or put the file on the
  server and set *…or a dump already on the server*.

### Disk clean-up

- Everything a source stores lives in its uploads and its work directory, `work/<id>`.
  Staged samples and restored dumps are rebuilt from the uploads when needed.
- **Free up space** (source page, `POST /connect/api/sources/<id>/free-space/`) deletes the uploads and the work
  directory. Snapshots and mappings are kept.
- **Deleting a source** removes its uploads and its work directory too.
- Uploading or removing a file clears the work directory, because its contents belonged to the old files.
- None of these touch data a sync has already written to a destination.
- They are all refused while a job is reading or writing the source.

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
