"""
Days-of-stock watchlist queries — feeds Page 2 (Stockout Engine).
"""

import pandas as pd
from utils.snowflake_conn import run_query


def get_dos_watchlist(facility_filter: str, window_days: int = 90) -> pd.DataFrame:
    """Days of stock remaining per product based on recent consumption velocity.

    Args:
        facility_filter: SQL IN clause string from utils.sql_in_filter()
        window_days: lookback window for average daily consumption (default 90)

    Returns columns:
        source_schema, canonical_name, therapeutic_class, therapeutic_subclass,
        current_soh, avg_daily_units, days_of_stock, predicted_stockout_date,
        dos_status (red | amber | green),
        stockout_episode_count (historical frequency)
    """
    return run_query(f"""
        WITH consumption AS (
            SELECT
                source_schema,
                product_id,
                canonical_product_id,
                SUM(quantity_dispensed)                             AS total_units,
                COUNT(DISTINCT DATE_TRUNC('day', dispensed_at))     AS active_days,
                -- avg over active dispensing days only, not calendar days
                SUM(quantity_dispensed) / NULLIF(
                    COUNT(DISTINCT DATE_TRUNC('day', dispensed_at)), 0
                )                                                   AS avg_daily_units,
                STDDEV(quantity_dispensed)                          AS stddev_daily_units
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
              AND dispensed_at >= DATEADD('day', -{window_days}, CURRENT_DATE)
            GROUP BY source_schema, product_id, canonical_product_id
        ),
        current_soh AS (
            SELECT
                source_schema,
                product_id,
                soh_after_raw                                       AS current_soh,
                soh_after_display                                   AS current_soh_display,
                dispensed_at                                        AS last_dispensed_at,
                ROW_NUMBER() OVER (
                    PARTITION BY source_schema, product_id
                    ORDER BY dispensed_at DESC
                )                                                   AS rn
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
        ),
        episode_count AS (
            SELECT
                source_schema,
                product_id,
                COUNT(*)                                            AS stockout_episode_count
            FROM HOSPITALS.REPORTING.fact_dispensing
            WHERE source_schema IN {facility_filter}
              AND is_stockout_dispense = TRUE
            GROUP BY source_schema, product_id
        )
        SELECT
            c.source_schema,
            t.canonical_name,
            t.therapeutic_class,
            t.therapeutic_subclass,
            cs.current_soh,
            cs.current_soh_display,
            cs.last_dispensed_at,
            c.avg_daily_units,
            c.stddev_daily_units,
            -- P50 days of stock (expected case)
            CASE
                WHEN c.avg_daily_units > 0
                THEN ROUND(cs.current_soh_display / c.avg_daily_units, 1)
                ELSE NULL
            END                                                     AS days_of_stock_p50,
            -- P90 days of stock (elevated demand scenario: mean + 1.28 * stddev)
            CASE
                WHEN (c.avg_daily_units + 1.28 * COALESCE(c.stddev_daily_units, 0)) > 0
                THEN ROUND(cs.current_soh_display /
                        (c.avg_daily_units + 1.28 * COALESCE(c.stddev_daily_units, 0)), 1)
                ELSE NULL
            END                                                     AS days_of_stock_p90,
            -- Predicted stockout date at P50 and P90
            DATEADD('day',
                ROUND(cs.current_soh_display / NULLIF(c.avg_daily_units, 0), 0),
                CURRENT_DATE
            )                                                       AS predicted_stockout_p50,
            DATEADD('day',
                ROUND(cs.current_soh_display /
                    NULLIF(c.avg_daily_units + 1.28 * COALESCE(c.stddev_daily_units, 0), 0), 0),
                CURRENT_DATE
            )                                                       AS predicted_stockout_p90,
            COALESCE(e.stockout_episode_count, 0)                  AS stockout_episode_count,
            CASE
                WHEN cs.current_soh_display / NULLIF(c.avg_daily_units, 0) < 7
                  OR cs.current_soh <= 0                           THEN 'red'
                WHEN cs.current_soh_display / NULLIF(c.avg_daily_units, 0) < 30 THEN 'amber'
                ELSE                                                    'green'
            END                                                     AS dos_status
        FROM consumption c
        JOIN current_soh cs
            ON c.source_schema = cs.source_schema
            AND c.product_id   = cs.product_id
            AND cs.rn = 1
        LEFT JOIN HOSPITALS.REPORTING.CANONICAL_PRODUCT_TAXONOMY t
            ON UPPER(c.source_schema) = UPPER(t.facility)
            AND c.product_id          = t.product_id
        LEFT JOIN episode_count e
            ON c.source_schema = e.source_schema
            AND c.product_id   = e.product_id
        WHERE t.product_category = 'pharma'
        ORDER BY
            CASE dos_status
                WHEN 'red'   THEN 1
                WHEN 'amber' THEN 2
                ELSE              3
            END,
            days_of_stock_p50 ASC NULLS LAST
    """)
