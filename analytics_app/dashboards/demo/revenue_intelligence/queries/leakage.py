"""
Revenue-leakage queries — flow vectors only on the Leakage view
(pharmacy, theatre, credit notes, unbilled consultations). The AR-balance
vectors (undispatched / unknown insurer) belong to the Receivables view and
are reconciled centrally in analytics.exposure_bridge() to avoid double-counting.
"""

from __future__ import annotations

import pandas as pd

from utils.snowflake_conn import run_query


def get_leakage_summary() -> pd.DataFrame:
    """All six leakage vectors (undispatched consultations row has NULL KES)."""
    return run_query("""
        SELECT leakage_vector, event_count, leakage_kes, snapshot_date
        FROM HOSPITALS.REPORTING.RPT_REV_LEAKAGE_SUMMARY
        ORDER BY leakage_kes DESC NULLS LAST
    """)


# ── Pharmacy ──────────────────────────────────────────────────────────────────

def get_pharmacy_trend() -> pd.DataFrame:
    """Monthly pharmacy leakage — always use the excl-outlier column for trend."""
    return run_query("""
        SELECT
            prescription_month,
            SUM(leakage_count)             AS leakage_events,
            SUM(leakage_kes)               AS leakage_kes_incl_outlier,
            SUM(leakage_kes_excl_outlier)  AS leakage_kes,
            SUM(total_prescriptions)       AS total_prescriptions,
            ROUND(SUM(leakage_count) * 100.0
                  / NULLIF(SUM(total_prescriptions), 0), 1) AS leakage_rate_pct,
            MAX(CASE WHEN contains_outlier THEN 1 ELSE 0 END) AS has_outlier
        FROM HOSPITALS.REPORTING.RPT_REV_PRESCRIPTION_LEAKAGE
        GROUP BY 1
        ORDER BY 1
    """)


def get_pharmacy_fulfillment() -> pd.DataFrame:
    return run_query("""
        SELECT
            SUM(dispensed_count)         AS dispensed,
            SUM(cancelled_count)         AS cancelled,
            SUM(filled_then_cancelled)   AS filled_then_cancelled,
            SUM(leakage_count)           AS leakage,
            SUM(total_prescriptions)     AS total,
            ROUND(SUM(leakage_count) * 100.0
                  / NULLIF(SUM(total_prescriptions), 0), 1) AS leakage_rate_pct
        FROM HOSPITALS.REPORTING.RPT_REV_PRESCRIPTION_LEAKAGE
    """)


def get_top_drugs(limit: int = 15) -> pd.DataFrame:
    return run_query(f"""
        SELECT
            drug_name, store_name,
            SUM(leakage_count)             AS leakage_events,
            SUM(leakage_kes_excl_outlier)  AS leakage_kes,
            SUM(total_prescriptions)       AS total_prescriptions,
            ROUND(SUM(leakage_count) * 100.0
                  / NULLIF(SUM(total_prescriptions), 0), 1) AS leakage_rate_pct
        FROM HOSPITALS.REPORTING.RPT_REV_PRESCRIPTION_LEAKAGE
        GROUP BY 1, 2
        ORDER BY leakage_kes DESC
        LIMIT {int(limit)}
    """)


def get_doctor_leakage(limit: int = 12) -> pd.DataFrame:
    return run_query(f"""
        SELECT
            prescribed_by,
            SUM(total_prescriptions)       AS prescriptions_written,
            SUM(leakage_count)             AS unfilled_count,
            SUM(leakage_kes_excl_outlier)  AS unfilled_kes,
            ROUND(SUM(leakage_count) * 100.0
                  / NULLIF(SUM(total_prescriptions), 0), 1) AS unfilled_rate_pct
        FROM HOSPITALS.REPORTING.RPT_REV_PRESCRIPTION_LEAKAGE
        WHERE prescribed_by IS NOT NULL
        GROUP BY 1
        HAVING SUM(total_prescriptions) >= 10
        ORDER BY unfilled_kes DESC
        LIMIT {int(limit)}
    """)


def get_pharmacy_outlier() -> pd.DataFrame:
    """The excluded outlier rows (Nutriflex) for transparent disclosure."""
    return run_query("""
        SELECT
            prescription_month, drug_name, prescribed_by, leakage_count,
            leakage_kes - leakage_kes_excl_outlier AS outlier_value_kes
        FROM HOSPITALS.REPORTING.RPT_REV_PRESCRIPTION_LEAKAGE
        WHERE contains_outlier = TRUE
        ORDER BY outlier_value_kes DESC
    """)


# ── Theatre ───────────────────────────────────────────────────────────────────

def get_theatre_summary() -> pd.DataFrame:
    return run_query("""
        SELECT
            SUM(booking_count) AS booked,
            SUM(CASE WHEN is_scheduled THEN booking_count ELSE 0 END) AS scheduled,
            SUM(CASE WHEN has_operation THEN booking_count ELSE 0 END) AS completed,
            SUM(CASE WHEN has_operation AND is_billed THEN booking_count ELSE 0 END) AS billed,
            SUM(completed_unbilled_kes) AS completed_unbilled_kes,
            ROUND(SUM(CASE WHEN has_operation THEN booking_count ELSE 0 END) * 100.0
                  / NULLIF(SUM(booking_count), 0), 1) AS completion_rate_pct,
            ROUND(SUM(CASE WHEN has_operation AND is_billed THEN booking_count ELSE 0 END) * 100.0
                  / NULLIF(SUM(CASE WHEN has_operation THEN booking_count ELSE 0 END), 0), 1)
                  AS billing_capture_pct
        FROM HOSPITALS.REPORTING.RPT_REV_THEATRE_FUNNEL
    """)


def get_theatre_rejections() -> pd.DataFrame:
    return run_query("""
        SELECT
            reason_summary,
            SUM(booking_count)         AS bookings,
            SUM(expected_revenue_kes)  AS lost_revenue_kes
        FROM HOSPITALS.REPORTING.RPT_REV_THEATRE_FUNNEL
        WHERE booking_status = 'rejected'
           OR (booking_status = 'booked' AND has_operation = FALSE AND is_scheduled = FALSE)
        GROUP BY 1
        ORDER BY lost_revenue_kes DESC NULLS LAST
        LIMIT 10
    """)


# ── Data quality register ─────────────────────────────────────────────────────

def get_data_quality() -> pd.DataFrame:
    return run_query("""
        SELECT metric, value, unit, snapshot_date
        FROM HOSPITALS.REPORTING.RPT_REV_DATA_QUALITY
        ORDER BY metric
    """)
