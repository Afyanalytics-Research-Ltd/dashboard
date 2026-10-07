"""
Core Snowflake queries feeding the intelligence layer and all pages.
All functions take a facility schema string (e.g. 'kisumu').
SQL IN-clause produced by utils.facility.sql_schema_filter().
"""

from __future__ import annotations

import pandas as pd

from utils.snowflake_conn import run_query


# ── Full dispensing history (feeds intelligence engines) ─────────────────────

def get_dispensing_history(
    schema: str,
    days_back: int = 730,
    ref_date: str = "CURRENT_DATE",
) -> pd.DataFrame:
    """
    All dispensing records for the facility within the lookback window.
    Used to train DemandEngine, AnomalyEngine, and LeadTimeEngine.

    ref_date: SQL date expression used as the anchor for the lookback window.
    Pass sql_ref_date(fac) from utils.facility — CURRENT_DATE for live
    facilities, a literal date string for historical ones.
    """
    return run_query(f"""
        SELECT
            f.product_id,
            f.canonical_product_id,
            f.quantity_dispensed,
            f.line_total,
            f.dispensed_at,
            f.soh_before,
            f.soh_after_raw,
            f.soh_after_display,
            f.is_stockout_dispense,
            f.dispensed_from_negative_stock,
            f.patient_id,
            f.raw_dispensing_id,
            COALESCE(t.canonical_name, f.product_id::VARCHAR) AS canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            t.product_category,
            t.inn_map_status
        FROM HOSPITALS.REPORTING.FACT_DISPENSING f
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER(f.source_schema) = UPPER(t.facility)
            AND f.product_id = t.product_id
        WHERE f.source_schema = '{schema}'
          AND t.product_category = 'pharma'
          AND f.dispensed_at >= DATEADD('day', -{days_back}, {ref_date})
        ORDER BY f.product_id, f.dispensed_at
    """)


# ── Current stock on hand ─────────────────────────────────────────────────────

def get_current_soh(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """
    Latest SOH per product with taxonomy and 90-day average daily consumption.
    One row per product.

    ref_date: SQL date expression used as the anchor for consumption windows
    and days-since-last-dispense. Pass sql_ref_date(fac) from utils.facility.
    """
    return run_query(f"""
        WITH latest AS (
            SELECT
                product_id,
                canonical_product_id,
                soh_after_raw        AS current_soh,
                soh_after_display    AS current_soh_display,
                dispensed_at         AS last_dispensed_at,
                ROW_NUMBER() OVER (
                    PARTITION BY product_id
                    ORDER BY dispensed_at DESC
                ) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
        ),
        consumption AS (
            SELECT
                product_id,
                SUM(quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0) AS avg_daily_units,
                STDDEV(quantity_dispensed)                                      AS std_daily_units
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -90, {ref_date})
            GROUP BY product_id
        )
        SELECT
            l.product_id,
            l.canonical_product_id,
            COALESCE(t.canonical_name, l.product_id::VARCHAR) AS canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            t.product_category,
            l.current_soh,
            l.current_soh_display,
            l.last_dispensed_at,
            DATEDIFF('day', l.last_dispensed_at, {ref_date}) AS days_since_last_dispense,
            c.avg_daily_units,
            c.std_daily_units,
            CASE
                WHEN c.avg_daily_units > 0
                THEN ROUND(l.current_soh_display / c.avg_daily_units, 1)
                ELSE NULL
            END AS days_of_stock,
            CASE
                WHEN l.current_soh < 0                                          THEN 'negative'
                WHEN l.current_soh = 0                                          THEN 'zero'
                WHEN l.current_soh_display / NULLIF(c.avg_daily_units, 0) < 7  THEN 'critical'
                WHEN l.current_soh_display / NULLIF(c.avg_daily_units, 0) < 30 THEN 'low'
                ELSE 'adequate'
            END AS stock_status
        FROM latest l
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER('{schema}') = UPPER(t.facility)
            AND l.product_id = t.product_id
        LEFT JOIN consumption c ON l.product_id = c.product_id
        WHERE l.rn = 1
          AND t.product_category = 'pharma'
        ORDER BY
            CASE stock_status
                WHEN 'negative' THEN 1 WHEN 'zero' THEN 2
                WHEN 'critical' THEN 3 WHEN 'low'  THEN 4 ELSE 5
            END,
            l.current_soh_display ASC
    """)


# ── KPI summary row ───────────────────────────────────────────────────────────

def get_kpi_summary(schema: str, ref_date: str = "CURRENT_DATE") -> pd.DataFrame:
    """
    Single-row aggregate KPIs for the Command Centre header.

    Product universe matches get_dos_watchlist(): pharma products with at least
    one dispensing event in the last 90 days. This ensures Briefing KPI counts
    are directly comparable to Stockout Watch and Order Workbench figures.

    ref_date: SQL date expression anchoring all lookback windows.
    Pass sql_ref_date(fac) from utils.facility.
    """
    return run_query(f"""
        WITH consumption AS (
            SELECT
                product_id,
                SUM(quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0) AS avg_daily_units
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -90, {ref_date})
            GROUP BY product_id
        ),
        current_soh AS (
            SELECT
                product_id,
                soh_after_raw        AS current_soh,
                soh_after_display    AS current_soh_display,
                ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY dispensed_at DESC) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
        ),
        active_pharma AS (
            -- Active pharma products: dispensed in last 90 days, in taxonomy as pharma.
            -- Matches the product universe used by get_dos_watchlist().
            SELECT
                cs.product_id,
                cs.current_soh,
                cs.current_soh_display,
                c.avg_daily_units
            FROM consumption c
            JOIN current_soh cs  ON c.product_id = cs.product_id AND cs.rn = 1
            JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
                ON UPPER('{schema}') = UPPER(t.facility) AND c.product_id = t.product_id
            WHERE t.product_category = 'pharma'
        ),
        status_agg AS (
            SELECT
                COUNT(*)                                                               AS total_products,
                COUNT_IF(current_soh <= 0)                                             AS active_stockouts,
                COUNT_IF(current_soh_display / NULLIF(avg_daily_units, 0) < 7
                         AND current_soh > 0)                                          AS critical_count,
                COUNT_IF(current_soh_display / NULLIF(avg_daily_units, 0) BETWEEN 7 AND 30) AS low_count
            FROM active_pharma
        ),
        value_90d AS (
            SELECT
                SUM(line_total)         AS total_dispensing_value_90d,
                SUM(quantity_dispensed) AS total_units_90d
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -90, {ref_date})
        ),
        patient_risk AS (
            SELECT
                COUNT_IF(has_chronic_drug = 1
                    AND last_dispensed_at >= DATEADD('day', -90, {ref_date})) AS chronic_patients_active,
                COUNT_IF(has_opioid = 1
                    AND last_dispensed_at >= DATEADD('day', -90, {ref_date})) AS opioid_patients_active
            FROM HOSPITALS.REPORTING.FACT_PATIENT_DISPENSING
            WHERE source_schema = '{schema}'
        )
        SELECT
            sa.total_products,
            sa.active_stockouts,
            sa.critical_count,
            sa.low_count,
            v.total_dispensing_value_90d,
            v.total_units_90d,
            pr.chronic_patients_active,
            pr.opioid_patients_active
        FROM status_agg sa
        CROSS JOIN value_90d v
        CROSS JOIN patient_risk pr
    """)


# ── DOS watchlist ─────────────────────────────────────────────────────────────

def get_dos_watchlist(
    schema: str,
    window_days: int = 90,
    ref_date: str = "CURRENT_DATE",
) -> pd.DataFrame:
    """
    Days of stock per product with P50 and P90 depletion forecasts.
    P90 uses elevated demand scenario: mean + 1.28 * stddev.

    ref_date: SQL date expression anchoring consumption windows and predicted
    stockout dates. Pass sql_ref_date(fac) from utils.facility.
    """
    return run_query(f"""
        WITH consumption AS (
            SELECT
                product_id,
                canonical_product_id,
                SUM(quantity_dispensed) /
                    NULLIF(COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0) AS avg_daily_units,
                STDDEV(quantity_dispensed)                                      AS stddev_daily_units
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
              AND dispensed_at >= DATEADD('day', -{window_days}, {ref_date})
            GROUP BY product_id, canonical_product_id
        ),
        current_soh AS (
            SELECT
                product_id,
                soh_after_raw        AS current_soh,
                soh_after_display    AS current_soh_display,
                dispensed_at         AS last_dispensed_at,
                ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY dispensed_at DESC) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
        ),
        episode_count AS (
            SELECT product_id, COUNT(*) AS stockout_episode_count
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}' AND is_stockout_dispense = TRUE
            GROUP BY product_id
        )
        SELECT
            c.product_id,
            COALESCE(t.canonical_name, c.product_id::VARCHAR) AS canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            cs.current_soh,
            cs.current_soh_display,
            cs.last_dispensed_at,
            c.avg_daily_units,
            c.stddev_daily_units,
            ROUND(cs.current_soh_display / NULLIF(c.avg_daily_units, 0), 1)   AS days_of_stock_p50,
            ROUND(cs.current_soh_display /
                NULLIF(c.avg_daily_units + 1.28 * COALESCE(c.stddev_daily_units, 0), 0), 1) AS days_of_stock_p90,
            DATEADD('day',
                ROUND(cs.current_soh_display / NULLIF(c.avg_daily_units, 0), 0),
                {ref_date}) AS predicted_stockout_p50,
            DATEADD('day',
                ROUND(cs.current_soh_display /
                    NULLIF(c.avg_daily_units + 1.28 * COALESCE(c.stddev_daily_units, 0), 0), 0),
                {ref_date}) AS predicted_stockout_p90,
            COALESCE(e.stockout_episode_count, 0) AS stockout_episode_count,
            CASE
                WHEN cs.current_soh <= 0 THEN 'red'
                WHEN cs.current_soh_display / NULLIF(c.avg_daily_units, 0) < 7  THEN 'red'
                WHEN cs.current_soh_display / NULLIF(c.avg_daily_units, 0) < 30 THEN 'amber'
                ELSE 'green'
            END AS dos_status
        FROM consumption c
        JOIN current_soh cs ON c.product_id = cs.product_id AND cs.rn = 1
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER('{schema}') = UPPER(t.facility) AND c.product_id = t.product_id
        LEFT JOIN episode_count e ON c.product_id = e.product_id
        WHERE t.product_category = 'pharma'
        ORDER BY
            CASE dos_status WHEN 'red' THEN 1 WHEN 'amber' THEN 2 ELSE 3 END,
            days_of_stock_p50 ASC NULLS LAST
    """)


# ── Dead stock candidates ─────────────────────────────────────────────────────

def get_dead_stock(
    schema: str,
    idle_threshold_days: int = 30,
    ref_date: str = "CURRENT_DATE",
) -> pd.DataFrame:
    """
    ref_date: SQL date expression used to calculate days_idle
    (DATEDIFF from last_dispensed_at to ref_date). Pass sql_ref_date(fac).
    """
    return run_query(f"""
        WITH activity AS (
            SELECT
                product_id,
                canonical_product_id,
                MAX(dispensed_at)                                       AS last_dispensed_at,
                DATEDIFF('day', MAX(dispensed_at), {ref_date})          AS days_idle,
                SUM(line_total)                                         AS total_historical_value,
                SUM(quantity_dispensed)                                 AS total_historical_units
            FROM HOSPITALS.REPORTING.FACT_DISPENSING
            WHERE source_schema = '{schema}'
            GROUP BY product_id, canonical_product_id
        ),
        soh AS (
            SELECT product_id, soh_after_display AS current_soh,
                   ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY dispensed_at DESC) AS rn
            FROM HOSPITALS.REPORTING.FACT_DISPENSING WHERE source_schema = '{schema}'
        )
        SELECT
            COALESCE(t.canonical_name, a.product_id::VARCHAR) AS canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            a.last_dispensed_at,
            a.days_idle,
            a.total_historical_value,
            a.total_historical_units,
            s.current_soh,
            CASE WHEN a.days_idle >= 90 THEN 'dead' ELSE 'slow' END AS idle_category
        FROM activity a
        JOIN soh s ON a.product_id = s.product_id AND s.rn = 1
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER('{schema}') = UPPER(t.facility) AND a.product_id = t.product_id
        WHERE a.days_idle >= {idle_threshold_days}
          AND s.current_soh > 0
          AND t.product_category = 'pharma'
        ORDER BY a.days_idle DESC, a.total_historical_value DESC
    """)


# ── Monthly dispensing trends ─────────────────────────────────────────────────

def get_monthly_trends(schema: str) -> pd.DataFrame:
    return run_query(f"""
        WITH monthly AS (
            SELECT
                f.canonical_product_id,
                DATE_TRUNC('month', f.dispensed_at)         AS dispensing_month,
                SUM(f.quantity_dispensed)                   AS total_units_dispensed,
                SUM(f.line_total)                           AS total_dispensing_value,
                COUNT(DISTINCT f.patient_id)                AS unique_patients,
                COUNT(DISTINCT f.raw_dispensing_id)         AS dispensing_events,
                COUNT_IF(f.is_stockout_dispense = TRUE)     AS stockout_dispenses
            FROM HOSPITALS.REPORTING.FACT_DISPENSING f
            WHERE f.source_schema = '{schema}'
              AND f.canonical_product_id IS NOT NULL
            GROUP BY f.canonical_product_id, DATE_TRUNC('month', f.dispensed_at)
        )
        SELECT
            m.*,
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            LAG(m.total_units_dispensed) OVER (
                PARTITION BY m.canonical_product_id ORDER BY m.dispensing_month
            ) AS prev_month_units,
            CASE
                WHEN LAG(m.total_units_dispensed) OVER (
                    PARTITION BY m.canonical_product_id ORDER BY m.dispensing_month) > 0
                THEN ROUND((m.total_units_dispensed -
                    LAG(m.total_units_dispensed) OVER (
                        PARTITION BY m.canonical_product_id ORDER BY m.dispensing_month)) * 100.0 /
                    LAG(m.total_units_dispensed) OVER (
                        PARTITION BY m.canonical_product_id ORDER BY m.dispensing_month), 1)
            END AS mom_change_pct
        FROM monthly m
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_KEYS t ON m.canonical_product_id = t.canonical_product_id
        ORDER BY t.canonical_name, m.dispensing_month
    """)


# ── Compliance: deficit dispenses ─────────────────────────────────────────────

def get_deficit_dispenses(schema: str) -> pd.DataFrame:
    return run_query(f"""
        SELECT
            COALESCE(t.canonical_name, f.product_id::VARCHAR) AS canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            h.dispensed_by_user_id,
            f.dispensed_at,
            f.soh_before,
            f.quantity_dispensed,
            f.soh_after_raw,
            f.line_total
        FROM HOSPITALS.REPORTING.FACT_DISPENSING f
        JOIN HOSPITALS.STAGING.STG_DISPENSING_HEADER h
            ON f.source_schema = h.source_schema
            AND f.raw_dispensing_id = h.raw_dispensing_id
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER(f.source_schema) = UPPER(t.facility)
            AND f.product_id = t.product_id
        WHERE f.source_schema = '{schema}'
          AND f.dispensed_from_negative_stock = TRUE
          AND t.product_category = 'pharma'
        ORDER BY f.dispensed_at DESC
    """)
