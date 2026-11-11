# Warehouse Studio

A Snowflake workspace inside Afya DataHub at **`/warehouse-studio/`** (superusers only):
SQL worksheets, a cost center, and a performance optimizer driven by the
cost & performance stored procedures in `HOSPITALS.ANALYTICS`.

| Page | What it adds over Snowsight |
|---|---|
| **Overview** | Account pulse: health score, month-to-date spend vs. last month, projected month-end, fleet state, what to fix next, what just ran |
| **Performance & Optimizer** | 12 detector procedures runnable on demand (with live progress) or daily; findings explorer with **triage** (acknowledge / in progress / resolve / ignore + notes, bulk); $-ranked impact map; "fastest path to healthy"; report procedures; schedule switch |
| **Cost Center** | Spend by service / warehouse / day, month burn-up with forecast, weekday×hour heat-map, attribution by user/role/warehouse/database/query tag, costliest query *shapes*, storage overhead with fix SQL |
| **Warehouses** | Fleet cards with idle burn and $-quantified recommendations; **what-if simulator** (size, auto-suspend) |
| **Worksheets** | Multi-tab CodeMirror editor, schema-aware autocomplete, run statement/selection/all, async with cancel, error line markers, **cost preview before running** (EXPLAIN), virtualized grid (10k rows) with column profiles, chart builder, operator profile + automatic diagnosis per query |
| **Query History** | Real-time (7 d) or account-wide (30 d) history, timeline zoom, badges (spill, pruning, queueing, compile), **query compare with SQL diff** |
| **Data Explorer / Procedures** | Catalog browser with preview, DDL, storage; typed forms for stored procedures |

Every query the Studio runs is tagged `warehouse-studio:<kind>:<user>` (`QUERY_TAG`) so its own cost shows up in attribution.

## Setup

1. **Connection** — the Studio reuses the platform's `SNOWFLAKE_*` variables
   (`ACCOUNT`, `USER`, `ROLE`, `WAREHOUSE`, `DATABASE`, `SCHEMA`, and
   `PRIVATE_KEY_PATH` or `PASSWORD`). Override any of them for the Studio only
   with `WAREHOUSE_STUDIO_SNOWFLAKE_<NAME>`.
2. **Toolkit** — install once, as ACCOUNTADMIN:

   ```bash
   python manage.py warehouse_studio_install            # uses SNOWFLAKE_ADMIN_* from .env
   python manage.py warehouse_studio_install --env-file /secure/admin.env
   python manage.py warehouse_studio_install --dry-run  # print the 82 statements
   python manage.py warehouse_studio_install --verify   # what exists
   python manage.py warehouse_studio_install --uninstall [--purge]
   ```

   Source: `snowflake/warehouse_studio/install.sql` (idempotent; tables use
   `IF NOT EXISTS`, findings survive re-installs).
3. `python manage.py migrate warehouse_studio`

### Settings (env var → default)

| Variable | Default | Meaning |
|---|---|---|
| `WAREHOUSE_STUDIO_TOOLKIT_DATABASE` / `_SCHEMA` | `HOSPITALS` / `ANALYTICS` | where the toolkit lives |
| `WAREHOUSE_STUDIO_CREDIT_PRICE_USD` | `2.60` | $/credit for every dollar figure (users can override locally in Cost Center) |
| `WAREHOUSE_STUDIO_STORAGE_PRICE_PER_TB_USD` | `23` | $/TB-month |
| `WAREHOUSE_STUDIO_ALLOW_WRITES` | off | allow DML/DDL and non-toolkit procedures after a per-run unlock |
| `WAREHOUSE_STUDIO_MAX_RESULT_ROWS` | `10000` | rows kept for the grid (exports stream up to `MAX_EXPORT_ROWS`, 1,000,000) |
| `WAREHOUSE_STUDIO_STATEMENT_TIMEOUT` | `900` | seconds |
| `WAREHOUSE_STUDIO_POOL_SIZE` | `6` | pooled Snowflake sessions per process |

## The toolkit and its security model

* The 12 detectors and 6 reports are ported from `TUNING_LAB.LAB_OWNER` and
  `OPTIMIZATIONS.TPC_OWNER`, fully qualified and de-duplicating, created
  **`EXECUTE AS OWNER`** so the app role (`DATAANALYSTS`) can run them
  without being granted `SNOWFLAKE.ACCOUNT_USAGE`. The app role receives only
  `USAGE` on those procedures and `SELECT` on the toolkit views/tables.
* Telemetry views (`WS_*`) expose 30–365 days of account usage; `WS_QUERY_HISTORY`
  redacts statements that can carry credentials.
* Nothing in the toolkit alters a warehouse or table. Procedures from the lab that
  do (`SP_RECLUSTER_WINDOW`, `SP_OPTIMIZE_WAREHOUSE_AUTOSUSPEND`, …) were
  deliberately **not** ported with owner's rights; the UI shows fix SQL to copy.
* The daily task `T_WS_TUNING_DETECTORS` is created **suspended**; switch it on
  from Optimizer → Runs & schedule.
* Worksheets are read-only by default: a real SQL lexer (`services/sql_guard.py`)
  classifies each statement; account/security administration (users, roles,
  grants, integrations) is refused even with writes enabled; `USE`/`SET` are
  refused (context comes from the pickers, sessions are pooled).

## Architecture

```
warehouse_studio/
  api.py              JSON API (async views; handlers run in a thread pool under ASGI)
  services/
    connection.py     session pool keyed by (role, warehouse, database, schema)
    executor.py       execute_async / get_query_status / get_results_from_sfqid / cancel / streaming export
    sql_guard.py      lexer, statement splitter, read/write/admin classifier
    sources.py        toolkit view → direct ACCOUNT_USAGE → "install the toolkit"
    cost.py history.py optimizer.py diagnostics.py catalog.py session.py installer.py
static/warehouse_studio/js
  app.js              shell, context bar, ⌘K command palette, g-chord shortcuts
  lib/                api, store, charts (ECharts), ui components, query inspector,
                      pure modules: fmt, metrics, layout, sqlsplit, gridcore, costcore, explorercore
  pages/              overview, optimizer, cost, warehouses, worksheets, history, data, procedures
```

## Tests

```bash
DEBUG=True python manage.py test warehouse_studio          # ~340 tests: Python + the Node JS suites
node --test warehouse_studio/tests/js/*.test.mjs           # JS only (Node ≥ 18)
WAREHOUSE_STUDIO_LIVE_TESTS=1 DEBUG=True python manage.py test warehouse_studio.tests.test_live   # real Snowflake, read-mostly
```

The live suite runs one detector (`SP_DETECT_WAREHOUSE_SETTINGS`), which records findings; everything else only reads.
