# Canary runs

A canary sends a few records through a pipeline's real path into the real
tables, reads every one of them back, and fails if any did not land intact.
Run one before a full load to find mapping, type and permission problems on
3 records instead of 300,000.

It works for the three load pipelines and for all three in sequence:

| Pipeline | Script / DAG | What a canary writes |
|---|---|---|
| JSON mappings → Snowflake | `from_json_mappings_to_snowflake.py`, DAG `from_json_mappings_to_snowflake` | N records per V3 table into `<NAME>_RAW.EVENTS_RAW` |
| Documents → Snowflake | DAG `gdrive_landingai_extract` (`dags/document_pipeline.py`) | the first N records of each file into the extraction table |
| Snowflake → V3 | `snowflake_to_v3_migration.py`, DAG `snowflake_to_v3_migration` | N records per table into the facility's V3 org |
| All of the above | DAG `canary_end_to_end` | each stage's canary, one after another |

N is any number: `canary=3` is the usual choice. `canary=0`, the default, is a
normal full run.

## Where canary data goes

Canary records go into **the same schemas, tables and V3 org as a full run**,
so the canary tests the real destination. The full run afterwards doesn't
write them twice:

- **EVENTS_RAW.** The canary upserts by record id: it loads its rows, then
  deletes older rows with the same ids. It never replaces or clears a table
  and never prunes, so the rest of an earlier load stays. The full run
  replaces each table as usual, canary rows included.
- **Document table.** The canary replaces only its own rows (`RECORD_KEY`).
  The file's other rows stay. It never marks a file done or moves it out of
  the source folder. The full run reuses the parsed pages, skips
  re-extracting canary records that group on the same pages under the same
  schema, then replaces the file's rows.
- **V3.** Canary records are real inserts. They are marked migrated, and each
  table's job is left open, so the full run skips them and posts the rest.

## Running a canary

### From the command line (WSL, repo root)

```bash
source .venv/bin/activate

# JSON mappings → Snowflake: 3 records per V3 table
python from_json_mappings_to_snowflake.py --mappings dags/jsons/silverwood_data_mappings.json --canary 3

# Snowflake → V3: 3 records per table, parents first
python snowflake_to_v3_migration.py --facility silverwood --canary 3
python snowflake_to_v3_migration.py --facility silverwood --canary 3 --table patients visits
```

Both scripts print the canary report at the end and exit with code 1 when the
canary failed. `--record-limit` still works as the old name for `--canary` on
the migration script. Add `--dry-run` to the JSON script to see which records
a canary would take without writing anything.

The document pipeline runs only in Airflow.

### From Airflow

Set the **Canary records** param when you trigger the DAG:

| DAG | Param | Notes |
|---|---|---|
| `from_json_mappings_to_snowflake` | `canary` | `tables` narrows it to some source tables |
| `gdrive_landingai_extract` | `canary` | Records per file. One file unless `max_files` is set. Use `file_ids` / `name_contains` to choose it. |
| `snowflake_to_v3_migration` | `canary` | Forces `order=by_tier`. `record_limit` is the old name; the larger value wins. |
| `canary_end_to_end` | `canary` (default 3) | See below |

Example trigger conf:

```json
{"mappings": "dags/jsons/wema_medical_data_mappings.json", "canary": 3}
```

## The end-to-end DAG

`canary_end_to_end` runs every stage's canary and only starts a stage when
the one before it passed:

```
plan ─► json_to_snowflake ─► flatten ─► verify_clean ─► migrate_to_v3 ─┐
  └──► documents (if run_documents) ───────────────────────────────────┴► summary, done
```

| Task | Does |
|---|---|
| `plan` | Reads the mappings, resolves the facility (schema) name and lists the V3 tables the canary touches |
| `json_to_snowflake` | Triggers `from_json_mappings_to_snowflake` with `canary=N` and waits for it |
| `flatten` | Triggers `snowflake_flatten_clean` for those V3 tables (their `<FAC>_CLEAN` views) and waits |
| `verify_clean` | Checks that each record the JSON canary landed is in its CLEAN view, by id |
| `migrate_to_v3` | Triggers `snowflake_to_v3_migration` with `canary=N` and waits (skipped when `migrate` is off) |
| `documents` | Triggers `gdrive_landingai_extract` with `document_conf` plus `canary=N` (only when `run_documents` is on) |
| `summary` | Logs every stage's canary report in one place, whatever happened |
| `done` | Runs only when no stage failed, so the DAG run's state is the verdict |

Params: `mappings`, `name` (blank = the connection's name in the extraction
tool; it must also be a facility the V3 migration knows), `canary`, `tables`,
`dedup` (run dedup in the flatten step, as a full run does), `migrate`,
`run_documents`, `document_conf` (for example
`{"schema_path": "dags/jsons/schema.json", "name_contains": "siaya"}`).

Each triggered run is an ordinary run of that DAG with the run id
`canary_e2e__<this run id>`, with its own logs. Clearing a stage's task in
`canary_end_to_end` re-runs that same triggered run.

## What is checked

Every record the canary writes is read back from the destination and compared
with what was sent. Values are compared in a normalised form, so a
destination's own spelling isn't a difference. For example, `12` and `"12.00"`
are equal, as are `true` and `1`, a midnight timestamp and its date, and the
same uuid in different case.

| Pipeline | Read back from | A record fails when |
|---|---|---|
| JSON → Snowflake | `EVENTS_RAW`, rows of this load (`ingested_at`) | it is missing; a mapped value isn't stored as sent; its id is stored more than once |
| Flatten (end-to-end only) | `<FAC>_CLEAN.<table>` by `ID` | it is missing from the view |
| Documents → Snowflake | the extraction table, by `RECORD_KEY` | its extraction failed; it is missing or stored twice; a field set in the extraction is NULL in its column (the value didn't fit the column type, which `sf_cast` hides in a full run) |
| Snowflake → V3 | gateway `read` of the model (newest pages) | it was dead-lettered; it is missing; a field sent is stored differently, left empty, or not returned |

Each target table or model ends up **passed**, **failed** or **untested**.
Untested means there was nothing to check, for example an empty source or
every record waiting on a parent, and the report says why. Untested isn't a
failure on its own, but a canary where *every* target is untested fails,
because it checked nothing.

### How the V3 canary picks records

Tables run tier by tier, parents first. Within a table the canary takes
records whose required parents are already in V3, which includes parents the
canary itself just posted, and passes over the rest. That way a child table
tests real inserts instead of reporting WAITING. If none of the first
`CANARY_SCAN_PAGES` pages has such a record, the table is untested with
"no record has its parents in V3 yet".

## When a canary fails

**Alerting.** A failed canary fails its Airflow task with an
`AirflowFailException` carrying the report. The central failure callback
(`config/airflow_local_settings.py`, see `PIPELINE_MONITORING_README.md`)
sends it to Sentry tagged `monitor_type:task_failure`, like any other failed
task. Nothing extra needs setting up. Failing tasks:

- JSON: the `report` task, plus each failing source table's `extract_load`
  result.
- Documents: `verify_canary`.
- V3: each failing table's `migrate` / `tier_N` task, then `canary_report`
  with every table.
- End-to-end: the stage that failed, then `done`.

**The report.** Each canary is saved as JSON:

```
output/canary/<pipeline>/<run_id>.json
output/canary/<pipeline>/<facility>_latest.json
```

The task also pushes it as the `pipeline_reconciliation` XCom, following the
contract in `PIPELINE_MONITORING_README.md`. The summary names each failing
table, the field, and up to 5 record ids:

```
Canary snowflake_to_v3_migration [silverwood] · 3 record(s) per target · run …: FAILED (4 passed, 1 failed, 2 untested)
  FAILED   visits: 3/3 landed · visit_date not stored as sent on 3 (ids 9f1c…, 7a2e…, 1b0d…)
  UNTESTED inpatient_vitals: 0/0 landed · untested — no record has its parents in V3 yet (12 passed over …)
```

Reports carry table, field and record ids only, never values. Use the ids to
look the records up at the source and the destination.

**Where to start.**

| Symptom | Usual cause |
|---|---|
| JSON: `<field> not stored as sent` | A mapping rule (lookup, code map, split) produced a value that didn't survive the JSON encoding. Check the field's rule in the mappings. |
| JSON: `stored more than once` | Two loads of the same ids under different `ingested_at`; an interrupted load. Re-run the canary: it replaces its own ids. |
| Flatten: missing from the CLEAN view | Dedup merged the record away (`HOSPITALS.DEDUP.CROSSWALK`), or the view filters it out |
| Documents: `<field> not stored as sent` | The extracted value doesn't fit the column type, e.g. text in a FLOAT column. Fix the extraction schema or the column. |
| Documents: `extraction failed` | LandingAI error for that record (in the `process_file` log) |
| V3: dead-lettered | V3 rejected the record. The `error_ids` and response are in the task log and the dead-letter file. |
| V3: `not stored as sent` | V3 rewrote, dropped or ignores the field. A field V3 legitimately rewrites goes into `CANARY_SKIP_FIELDS`. A field it drops is a mapping error. |
| V3: missing | Posted but not found in the newest `CANARY_VERIFY_PAGES` pages. Raise that setting if the model is busy or not ordered by id. |
| Every target untested | Nothing to check: an empty source, everything already migrated, or no parents in V3. Run the parent tables first. |

After a fix, run the canary again. It replaces its own rows (Snowflake) or
skips records already in V3, so re-running is safe.

## Settings

| Env var | Default | Effect |
|---|---|---|
| `CANARY_MAX_PAGES` | 20 | JSON: pages read per source table looking for N rows past a V3 table's `where` filter |
| `CANARY_SCAN_PAGES` | 10 | V3: pages of a CLEAN view read looking for records whose parents are in V3 |
| `CANARY_VERIFY_PAGES` | 4 | V3: newest pages of a model read back (the whole model when it is that small) |
| `CANARY_SKIP_FIELDS` | `id,updated_at` | V3: fields V3 is expected to rewrite, never compared. The model's facility column is always skipped. |
| `CANARY_DIR` | `<repo>/output/canary` | where reports are saved |

## Limits

- **Record selection.** The JSON canary takes the first N rows of each source
  table. It doesn't choose children that belong to the canary's parents, so a
  child table may have none whose parents were canaried. In the V3 stage that
  child table is then untested on a fresh facility, unless its parents
  already exist in V3.
- **V3 read-back.** The gateway can't read by id, so the canary reads the
  newest pages of each model. A record posted to an existing V3 row (matched
  on uuid) is found by its V3 id.
- **Documents.** Only the first N complete records of each file are
  extracted. A record the page limit may have cut off is left out.
- **Canary writes stay.** Canary records remain in the destination, as they
  would after a full run. A canary is not a dry run: use `--dry-run` /
  `mode=dry_run` for that.

## Code

- `canary.py`: the shared parts: the `canary` count, value normalisation,
  `TargetResult` / `CanaryReport`, saving, and `finish_in_airflow`.
- `from_json_mappings_to_snowflake.py`: `canary_rows`, `Warehouse.upsert_records`
  / `read_back` / `duplicate_ids`, `verify_canary`, `canary_report`.
- `dags/document_pipeline.py`: `canary_records`, `canary_record_keys`,
  `load_to_snowflake(record_keys=…)`, `verify_snowflake_canary`.
- `snowflake_to_v3_migration.py`: `_canary_pick`, `_read_back_v3`,
  `_verify_canary`, `_finish_canary`, `canary_report`.
- `dags/canary_end_to_end_dag.py`: the end-to-end DAG.
- Tests: `tests/test_canary.py`

```bash
.venv/bin/python -m pytest tests/test_canary.py
```
