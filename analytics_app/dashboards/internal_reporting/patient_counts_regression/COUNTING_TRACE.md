# Current counting logic: trace, dependencies and invariants

Source: `analytics_app/dashboards/internal_reporting/snowflake_patient_counts.sql`, procedure
`AFYA_REFRESH_PATIENT_COUNTS(PUBLISH_RESULTS, ACCEPT_REVIEW, FULL_RECOUNT)`, L151-403 as of 2026-11-05.

The SHA-1 of the file at review time is `94e5b84b7bc043e4777768b68dd073b0d49250f2`. The file is **untracked in git**
(see "Risks").

The executable model is `counting_reference.py`, with characterisation tests in `test_counting_reference.py`
(24/24 pass).

## 1. Calculations

### 1.1 Normalised view of each source row (`src_cte`, L232-239)

| Field | Expression | Notes |
|---|---|---|
| `uk` | `NULLIF(TRIM(<UPD_KEY>::STRING), '')` | Only blank becomes NULL. `'null'`, `'0'` and `'UNKNOWN'` are real keys (test N08) |
| `is_del` | `FALSE` if no `DELETED_COLUMN`, else `TRIM(COALESCE(<col>::STRING,'')) NOT IN ('', 'null', 'NULL', 'None', 'none', 'false', 'FALSE', '0')` | Case-sensitive list: `'False'`, `'N'` and `'yes'` mean deleted (N02). An `IS_ACTIVE` column would invert the meaning (N03) |
| `recv` | `NULL::DATE`, or `TRY_TO_TIMESTAMP(<RECEIVED_COLUMN>::STRING)::DATE` | Used only for receipts |
| `rn` | `1` if no `VERSION_COLUMN`; else `1` when `<RECORD_KEY>` is NULL, else `ROW_NUMBER() OVER (PARTITION BY <RECORD_KEY> ORDER BY TRY_TO_TIMESTAMP(<VERSION_COLUMN>::STRING) DESC NULLS LAST)` | A record key without a version column does nothing (N07). An unparseable version sorts last (N05). Ties are broken nondeterministically |

### 1.2 Per-source statistics (L253-256), one scan

| Run column | Expression |
|---|---|
| `ROWS_READ` | `COUNT(*)` |
| `ROWS_SUPERSEDED` | `COUNT_IF(rn > 1)` |
| `ROWS_DELETED` | `COUNT_IF(rn = 1 AND is_del)` |
| **`PATIENT_RECORDS`** | **`COUNT_IF(rn = 1 AND NOT is_del)`** |
| **`UPDS`** | **`COUNT(DISTINCT IFF(rn = 1 AND NOT is_del, uk, NULL))`**: distinct keys over *kept* records only (N09) |
| `RECORDS_NO_UPD_KEY` | `COUNT_IF(rn = 1 AND NOT is_del AND uk IS NULL)` |

### 1.3 Reuse instead of rescanning (L224-251)

- **Only base tables** are reused. `altered` is the `LAST_ALTERED` of a `BASE TABLE`; views get NULL, so they are
  always recounted.
- **When:** `NOT FULL_RECOUNT AND altered IS NOT NULL`, and the latest run has the same `SOURCE`, the same
  `CONFIG_SIG`, `SOURCE_ALTERED = altered` and a status other than `failed`.
- **`CONFIG_SIG`** = `logic_version | obj | uk | rk | dc | vc` (L192-193). `logic_version = '2026-11-04.3'`.
  `RECEIVED_COLUMN` is not part of it.

### 1.4 Checks (L264-278), which decide `ok` or `review`

- **Zero records.**
- **`UPDS > PATIENT_RECORDS`.** This can't fire (K01).
- **Missing keys:** `RECORDS_NO_UPD_KEY > 5%` of records.
- **Large moves:** a change of more than 20% in records or UPDs against the last *published* run of the same
  `SOURCE` (`MAX_BY(.., RUN_AT) WHERE PUBLISHED`).
- A new source has no publish history, so only the zero and missing-key checks apply (K04).

### 1.5 Publish (L349-394), one transaction; on error, ROLLBACK and RAISE

1. **MERGE** into `AFYA_PATIENT_COUNTS` from this run's `RUNS` rows where `STATUS = 'ok'` (or `review` with
   `ACCEPT_REVIEW`).
   - It sets `PATIENT_RECORDS`, `UPDS`, `RUN_ID` and `COUNTED_AT`.
   - `CHANGED_AT` changes only when a figure changed.
2. **`PUBLISHED = TRUE`** on those run rows.
3. **DELETE** from `COUNTS` any row whose `SOURCE` is not an active row in `SOURCES`.
4. **Catch-up receipts** (2.2).

A source that fails keeps its previous published figures, because failed rows are not merged.

## 2. Historical receipts (`AFYA_PATIENT_RECORD_RECEIPTS`)

### 2.1 First publish of a source that has a `RECEIVED_COLUMN` (L280-295)

- **Condition:** only if the source has no receipts yet, and the run is publishing an `ok` (or accepted) status.
- **Records** are dated by their own `recv` day.
- **Each UPD** is dated by the earliest `recv` among its kept records. Undated rows get `RECEIVED_DATE NULL`.
- **Method:** `'load time'` (or `'undated'`). The sums equal the published figures (E01, E02).
- **Not transactional:** these rows are written **before** the run row, outside the publish transaction. A later
  publish rollback leaves them in place.

### 2.2 Catch-up after every publish (L367-388)

- **Which sources:** each published source whose `PATIENT_RECORDS` or `UPDS` differ from `SUM(RECORDS_ADDED)` /
  `SUM(UPDS_ADDED)` in its receipts.
- **Amount:** one row carrying the *difference*. It can be negative (E05).
- **Date and method:**

| Situation | `RECEIVED_DATE` | `METHOD` |
|---|---|---|
| No receipts yet | `COALESCE(RAW_MAIN_LOAD, SOURCE_CREATED, run date)` | `raw event load date` / `table created` / `first counted` (E03, E08, E09) |
| `RAW_LAST_LOAD` newer than the last receipt date | `RAW_LAST_LOAD` | `raw event load date` (E04) |
| Otherwise | the run date | `run increment` (E05) |

- **Raw dates** (L297-330) apply only to `*_CLEAN` objects with no `RECEIVED_COLUMN`, where `<X>_RAW.EVENTS_RAW` exists.
  - `RAW_LAST_LOAD = MAX(INGESTED_AT)::DATE` `WHERE SOURCE_TABLE = LOWER(<table>)`.
  - `RAW_MAIN_LOAD` is the day with the most events, where array payloads count their elements. It is computed only
    on the source's first publish.
  - Errors are swallowed, so the dates are lost but the count is not.
- **Invariant:** for every published source, the receipts sum to `COUNTS` (E07; regression check R3).

## 3. Dependencies

| Consumer | Depends on |
|---|---|
| Refresh cursor (L175-179) | `SOURCES` columns `SOURCE`, `SOURCE_OBJECT`, `UPD_KEY`, `RECORD_KEY`, `DELETED_COLUMN`, `VERSION_COLUMN`, `RECEIVED_COLUMN`; only `STATUS = 'active'` |
| Publish DELETE (L364-365) | `SOURCES.STATUS = 'active'` by `SOURCE`. **Any other status value means "remove from the dashboard"** |
| Reuse, 20% check and receipts | `SOURCE` (the dashboard name) as the join key across `RUNS`, `COUNTS` and `RECEIPTS`. Renaming a source orphans its history |
| Dashboard `data.py` | `COUNTS(SOURCE, PATIENT_RECORDS, UPDS, COUNTED_AT)`; `RECEIPTS` summed by `SOURCE` and `RECEIVED_DATE` |
| Overview tab | `UPDs = SUM(COUNTS.UPDS)`, `patient records = SUM(COUNTS.PATIENT_RECORDS)` |
| Return string (L399-401) | `n_proposed`, currently the count of `proposed` rows in `SOURCES`. **This is the only line the redesign changes** (migration diff 05) |
| Discovery (current) | `CALL AFYA_DISCOVER_PATIENT_DATASETS()` at L181, unguarded |

## 4. What must stay unchanged during the redesign

| # | Invariant | How it's protected |
|---|---|---|
| I1 | `src_cte` and the statistics statement (1.1-1.2) are byte-identical | diff 05 touches only L181-182; R7 compares the deployed body with the repo plus diff 05 |
| I2 | `logic_version` and `CONFIG_SIG` composition | not bumped. R2: same `CONFIG_SIG` per source before and after |
| I3 | Reuse rule | R2: base tables still `REUSED = TRUE` on the first dry run after cut-over |
| I4 | Checks, thresholds and statuses | R1 |
| I5 | Publish transaction, MERGE filter and the non-active DELETE | R4, R5 |
| I6 | Receipts algorithm and the sum invariant | R3 |
| I7 | `SOURCES` columns, `STATUS` vocabulary and `SOURCE` naming: discovery never writes to `SOURCES`; only the human approval procedure does, with `STATUS = 'active'` | migration V5 hash; R6 |
| I8 | Error masking (L342) | unchanged code |
| I9 | Approved candidates use only uppercase, unquoted identifiers (`UPPER(TRIM())` at L186-191 and the identifier regexes at L199-205 would otherwise fail them) | approval guard (report 4.4) |

## 5. Risks found while tracing

1. **The script is untracked in git** (`?? …/snowflake_patient_counts.sql`). Rollback by "redeploy the previous body
   from git" is impossible until the current body is committed. Commit it, or keep a copy of the deployed DDL
   (01 P8), **before** cut-over.
2. **Approving a candidate that has duplicate `SOURCE_OBJECT` rows, or the same `SOURCE`**, makes the MERGE
   non-deterministic, and Snowflake raises by default. The whole publish rolls back, so no source is published that
   week. The approval guard must prevent it; R6 detects it.
3. **Load-time receipts are written outside the publish transaction** (2.1). A publish failure in the same run leaves
   them, and they block future `'load time'` dating (`n_have > 0`). That's an existing behaviour, not introduced by
   the redesign.
4. **`TRY_TO_TIMESTAMP` on integer strings interprets magnitude row by row.** Mixed formats would date receipts
   inconsistently, without any error.
5. **The placeholder list used by discovery** (`'0'`, `'null'`, `'UNKNOWN'`, …) is **not** applied by counting (N08).
   A discovery profile's missing-key share can therefore be higher than the refresh's `RECORDS_NO_UPD_KEY` for the
   same source. Don't "fix" this in counting as part of the redesign: it would change published UPDs.
