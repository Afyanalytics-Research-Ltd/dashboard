# Snowflake → Databend parity check (prod)

Scripts used on 2026-11-16 to check that everything in Snowflake `HOSPITALS`
is in Databend, and to fill the gaps without straining the server. Run any of
them with `run_py.sh` (low priority, inside the prod web container, using the
container's own credentials).

| Script | Writes? | What it does |
|---|---|---|
| `audit.py OUT.json` | no | Table/view inventory of both warehouses from **metadata only** (Snowflake `SHOW TABLES/VIEWS`, which needs no warehouse; Databend `system.tables`). Compares row counts per table. |
| `fill_gaps.py [--apply]` | with `--apply` | `ANALYTICS.AFYA_HMIS_PATIENT_COUNTS`: adds the columns Databend lacks and **appends** the Snowflake runs (`RUN_ID`) Databend doesn't have, 20k-row batches, each verified. Creates `REPORTING.CANONICAL_PRODUCT_TAXONOMY` if absent. |
| `fill2.py [--apply]` | with `--apply` | Loads the taxonomy table (its `LOADED_AT` is corrupt in Snowflake, so it's loaded decoded, epoch µs → timestamp). Creates missing HOSPITALS views via the translator, each compile-checked. |
| `fill3.py` | yes | Copies a named list of tables that exist only in Snowflake (only if absent). |

All of them are dry-run by default where they write, never drop or replace
existing Databend data, pause while host load > 2.5 or free memory < 1 GB, and
can be re-run safely (they only add what is missing).

Ongoing sync is the `warehouse_bisync` DAG (hourly), which since this change
also handles append-only tables by key and keeps Databend's views in step;
these scripts are for checking it or for a one-off catch-up.
