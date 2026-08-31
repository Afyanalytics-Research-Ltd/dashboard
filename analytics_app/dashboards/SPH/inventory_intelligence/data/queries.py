"""Parameterized SQL builders.

Contract points these builders enforce:

- **Bind parameters for values.** Facility discriminators, dates, and other
  values travel as ``%(name)s`` binds (connector pyformat); only schema/table
  identifiers are interpolated, and those are validated against a strict
  identifier pattern first.
- **No category filters.** ``product_category`` is a returned dimension,
  never a ``WHERE`` literal (design law).
- **No canonical_product_id joins.** It is renumbered per taxonomy pipeline
  run; product joins use ``(facility, product_id)`` and therapeutic class is
  keyed by ``inn_name`` (attached at taxonomy load — handoff gotcha #4).
- **v2 is all-TEXT with systematic leading whitespace** → every v2 column is
  ``TRIM``-ed in SQL; booleans compared via ``LOWER(x)='true'``; timestamps
  via ``TRY_TO_TIMESTAMP``; quantities via ``TRY_TO_DOUBLE(TRIM(...))``.
- **No ``TRY_TO_*`` inside UNION branches** (Snowflake reconciles branches
  with ``TRY_CAST``, which is illegal between FLOAT and NUMBER — handoff
  gotcha #1). These builders avoid SQL UNIONs across sources entirely: v1/v2
  stitching happens in Python (``ingestion.py``), so each statement is
  single-source. Any future UNION must use plain ``CAST`` on both branches.
- **SQL fetches raw typed rows only** — no CASE tiers, no averages
  the Python layer would re-derive, no ``CURRENT_DATE`` anchors (horizons
  come from ``utils.facility.sql_ref_date``).
"""

from __future__ import annotations

import re

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Fully-qualified source objects (all pre-existing — reuse, never rebuild).
FACT_DISPENSING = "HOSPITALS.REPORTING.FACT_DISPENSING"
FACT_PROCUREMENT = "HOSPITALS.REPORTING.FACT_PROCUREMENT"
RPT_STOCK_STATUS = "HOSPITALS.REPORTING.RPT_STOCK_STATUS"
TAXONOMY = "HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY"
V2_SCHEMA = "HOSPITALS.ORTHOPEDIC_CLEAN_V2"
V1_LEDGER = "HOSPITALS.ORTHOPEDIC_CLEAN.ORTHO_INVENTORYLEDGERENTRIES"


def _check_ident(name: str) -> str:
    """Validate an identifier destined for f-string interpolation.

    Identifiers (schema/table names) cannot be bind parameters in Snowflake,
    so interpolation is unavoidable — but only after a strict allowlist match.
    """
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


# ── Consumption: v1 (FACT_DISPENSING) ────────────────────────────────────────

def consumption_v1_sql() -> str:
    """v1 dispensing history for one facility. Bind: ``facility``.

    Emits the ``load_consumption`` column contract directly. The SPH
    branch of the fact carries ``soh_before`` on every row — the censoring
    signal. ``store``/``batch`` are not exposed by the fact's SPH
    branch, so they are NULL here (v2 supplies them).
    """
    return f"""
        SELECT
            f.product_id                                            AS item_key,
            COALESCE(t.canonical_name, f.product_id)                AS canonical_name,
            t.inn_name                                              AS inn_name,
            t.therapeutic_class                                     AS therapeutic_class,
            COALESCE(t.product_category, 'uncategorized')           AS product_category,
            f.dispensed_at                                          AS dispensed_at,
            CAST(f.quantity_dispensed AS FLOAT)                     AS quantity,
            CAST(f.line_total AS FLOAT)
                / NULLIF(CAST(f.quantity_dispensed AS FLOAT), 0)    AS unit_price,
            'v1'                                                    AS source_system,
            CAST(f.soh_before AS FLOAT)                             AS soh_before,
            CAST(f.soh_after_raw AS FLOAT)                          AS soh_after,
            f.is_stockout_dispense                                  AS is_stockout_dispense,
            CAST(f.patient_id AS VARCHAR)                           AS patient_id,
            CAST(NULL AS VARCHAR)                                   AS store,
            CAST(NULL AS VARCHAR)                                   AS batch
        FROM {FACT_DISPENSING} f
        LEFT JOIN {TAXONOMY} t
            ON UPPER(t.facility) = UPPER(f.source_schema)
           AND t.product_id      = f.product_id
        WHERE f.source_schema = %(facility)s
        ORDER BY f.product_id, f.dispensed_at
    """


# ── Consumption: v2 (PHARMREQUESTS) ──────────────────────────────────────────

def consumption_v2_sql(table: str = "PHARMREQUESTS") -> str:
    """v2 dispensing lines with the full v2 cleaning contract.

    - Real dispenses only: ``TRIM("dispensed") IN ('1','3')``.
    - Timestamps from ``"stamp"`` (ISO) via ``TRY_TO_TIMESTAMP`` — never the
      ``DD/MM/YYYY`` ``"date"`` column.
    - Every TEXT column is TRIM-ed (systematic leading whitespace in source).
    - Item identity is (itcode, itname); bridging to the v1 master spine
      happens in Python (``ingestion.py``), so no cross-source UNION exists
      here and ``TRY_TO_*`` is safe.
    """
    table = _check_ident(table)
    return f"""
        SELECT
            NULLIF(TRIM("itcode"), '')                  AS itcode,
            NULLIF(TRIM("itname"), '')                  AS itname,
            TRY_TO_TIMESTAMP(TRIM("stamp"))             AS dispensed_at,
            TRY_TO_DOUBLE(TRIM("qty"))                  AS quantity,
            TRY_TO_DOUBLE(TRIM("price"))                AS unit_price,
            NULLIF(TRIM("category"), '')                AS category,
            NULLIF(TRIM("batches"), '')                 AS batch,
            NULLIF(TRIM("userstore"), '')               AS store,
            NULLIF(TRIM("patid"), '')                   AS patient_id
        FROM {V2_SCHEMA}."{table}"
        WHERE TRIM("dispensed") IN ('1', '3')
        ORDER BY 1, 3
    """


# ── Receipts ─────────────────────────────────────────────────────────────────

def receipts_procurement_sql() -> str:
    """Goods-received lines from the procurement gold layer. Bind: ``facility``.

    ``doc_type='RECEIPT'`` = supplier-invoice receiving — at SPH most
    purchasing bypasses formal POs, so receipts (not orders) are the
    replenishment-event stream. Receipt unit cost is best-effort in
    source (single vs bulk basis) and is treated as such downstream.
    """
    return f"""
        SELECT
            p.item_id                                   AS item_key,
            p.doc_at                                    AS received_at,
            CAST(p.quantity AS FLOAT)                   AS quantity,
            p.supplier_name                             AS supplier,
            CAST(p.unit_price AS FLOAT)                 AS unit_cost,
            p.batch_expiry                              AS batch_expiry,
            COALESCE(p.product_category, 'uncategorized') AS product_category,
            'procurement'                               AS receipt_source
        FROM {FACT_PROCUREMENT} p
        WHERE p.source_schema = %(facility)s
          AND p.doc_type = 'RECEIPT'
        ORDER BY p.item_id, p.doc_at
    """


def receipts_ledger_in_sql() -> str:
    """v1 stock-ledger IN movements (SPH branch pattern) — the second receipt
    stream, complementing invoice receipts for inter-arrival modeling.

    Semantics verified by the ledger backtest: text booleans compared with
    ``LOWER(x)='true'``; ``units_moved`` (raw ``containing``) = 0 rows are
    ghosts and excluded; ``movement_direction`` derives from the sign of the
    movement. Single-source statement → ``TRY_TO_*`` free anyway (plain CAST
    used for the numeric).
    """
    return f"""
        SELECT
            le.item_id                                  AS item_key,
            CAST(le.created_at AS TIMESTAMP_NTZ)        AS received_at,
            CAST(le.units_moved AS FLOAT)               AS quantity,
            CAST(NULL AS VARCHAR)                       AS supplier,
            CAST(le.unit_price AS FLOAT)                AS unit_cost,
            CAST(NULL AS TIMESTAMP_NTZ)                 AS batch_expiry,
            CAST(NULL AS VARCHAR)                       AS product_category,
            'ledger_in'                                 AS receipt_source
        FROM {V1_LEDGER} le
        WHERE le.movement_direction = 'IN'
          AND LOWER(le.is_committed) = 'true'
          AND COALESCE(LOWER(le.is_discarded), 'false') <> 'true'
          AND CAST(le.units_moved AS FLOAT) <> 0
        ORDER BY le.item_id, le.created_at
    """


# ── Stock on hand ────────────────────────────────────────────────────────────

def stock_status_sql() -> str:
    """Latest SOH per product from ``RPT_STOCK_STATUS``. Bind: ``facility``.

    Raw and floor-at-zero SOH are both returned; the display tiers in the
    view are NOT selected — banding is computed statistically in Python.
    ``soh_as_of`` is attached in Python
    from ``FacilityMeta.stock_as_of`` because the freeze date is facility
    metadata, not a fact column.
    """
    return f"""
        SELECT
            s.product_id                                AS item_key,
            s.canonical_name                            AS canonical_name,
            s.inn_name                                  AS inn_name,
            s.therapeutic_class                         AS therapeutic_class,
            COALESCE(s.product_category, 'uncategorized') AS product_category,
            CAST(s.current_soh_raw AS FLOAT)            AS soh_raw,
            CAST(s.current_soh AS FLOAT)                AS soh,
            s.last_dispensed_at                         AS last_dispensed_at,
            CAST(s.days_since_last_dispense AS FLOAT)   AS days_since_last_dispense,
            s.stock_status                              AS stock_status
        FROM {RPT_STOCK_STATUS} s
        WHERE UPPER(s.facility) = UPPER(%(facility)s)
    """


def batches_sql() -> str:
    """Procurement lines carrying a batch expiry date (expiry_risk source).
    Bind: ``facility``.

    SPH has no batch-level stock ledger, so this is procurement-grain — the
    quantity *received* on a dated batch, roughly 5% of lines carry an expiry —
    an expiry *watchlist*, not remaining-on-shelf. Columns: item_key, batch,
    qty, expiry.
    """
    return f"""
        SELECT
            p.item_id                                     AS item_key,
            COALESCE(p.doc_code, CAST(p.doc_id AS VARCHAR)) AS batch,
            CAST(p.quantity AS FLOAT)                     AS qty,
            CAST(p.batch_expiry AS DATE)                  AS expiry
        FROM {FACT_PROCUREMENT} p
        WHERE p.source_schema = %(facility)s
          AND p.batch_expiry IS NOT NULL
    """


def item_cost_sql() -> str:
    """Per-item acquisition cost — median non-zero procurement unit price across
    all document types. Bind: ``facility``.

    This is the COST basis for inventory valuation and turnover (what SPH paid
    to acquire stock), distinct from the billed *selling* price carried on
    dispensing rows. Inventory is valued at cost, never at selling price.
    """
    return f"""
        SELECT
            CAST(p.item_id AS VARCHAR)            AS item_key,
            MEDIAN(CAST(p.unit_price AS FLOAT))   AS unit_cost
        FROM {FACT_PROCUREMENT} p
        WHERE p.source_schema = %(facility)s
          AND p.unit_price > 0
        GROUP BY 1
    """


# ── Product catalog / taxonomy ───────────────────────────────────────────────

def catalog_sql() -> str:
    """Canonical taxonomy rows for one facility. Bind: ``facility``.

    Therapeutic class arrives on the taxonomy row already keyed by
    ``inn_name`` (stable across pipeline runs); nothing here or downstream
    joins by ``canonical_product_id`` (renumbered per run — gotcha #4). It is
    returned for reference only.
    """
    return f"""
        SELECT
            t.product_id                                AS item_key,
            t.product_name                              AS product_name,
            t.canonical_name                            AS canonical_name,
            t.inn_name                                  AS inn_name,
            t.strength_canonical                        AS strength_canonical,
            t.form_canonical                            AS form_canonical,
            t.therapeutic_class                         AS therapeutic_class,
            t.therapeutic_subclass                      AS therapeutic_subclass,
            COALESCE(t.product_category, 'uncategorized') AS product_category,
            t.inn_map_status                            AS inn_map_status,
            t.canonical_product_id                      AS canonical_product_id
        FROM {TAXONOMY} t
        WHERE UPPER(t.facility) = UPPER(%(facility)s)
    """


# ── Theatre procedures ───────────────────────────────────────────────────────

def theatre_procedures_sql(table: str = "THEATREQUESTS") -> str:
    """Theatre/surgery requests (v2). Raw procedure text plus timestamps for
    weekly counts feeding procedure-driven demand. Full v2 TRIM contract applies.
    """
    table = _check_ident(table)
    return f"""
        SELECT
            NULLIF(TRIM("procedures"), '')              AS procedures,
            NULLIF(TRIM("procnotes"), '')               AS procnotes,
            TRY_TO_TIMESTAMP(TRIM("stamp"))             AS requested_at,
            NULLIF(TRIM("patid"), '')                   AS patient_id,
            NULLIF(TRIM("visitno"), '')                 AS visit_no,
            NULLIF(TRIM("prescid"), '')                 AS presc_id
        FROM {V2_SCHEMA}."{table}"
        ORDER BY 3
    """
