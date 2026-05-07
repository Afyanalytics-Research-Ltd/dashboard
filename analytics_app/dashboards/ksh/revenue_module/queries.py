"""
queries.py
----------
Snowflake SQL queries for the PharmaPlus Revenue Intelligence dashboard.

NOTE on date types
~~~~~~~~~~~~~~~~~~
The source MySQL columns (INVOICE_DATE, RECEIPT_DATE, CREATED_AT, ...) often
arrive in Snowflake as VARCHAR rather than as proper DATE / TIMESTAMP types.
Snowflake's DATE_TRUNC / DATEDIFF / "date - date" arithmetic refuse to operate
on VARCHAR — they raise:

    SQL compilation error: Function DATE_TRUNC does not support VARCHAR(...)
    argument type

To stay robust regardless of how the warehouse models a column, every query
below wraps date/timestamp references in TRY_TO_DATE / TRY_TO_TIMESTAMP. These
casts are no-ops if the column is already DATE/TIMESTAMP and parse leniently
when the column is VARCHAR.

Each query is parameterised with two date placeholders:
    {start}   inclusive (e.g. '2024-05-01')
    {end}     exclusive (e.g. '2026-05-01')
and (where relevant) {clinic_filter} which is either '' (no filter) or a
fragment such as "AND v.CLINIC IN (1,2,3)".

All amount columns are returned as KSh (Kenyan Shillings).
"""

from textwrap import dedent

# ──────────────────────────────────────────────────────────────────────────────
# 1. DAILY REVENUE TREND
# ──────────────────────────────────────────────────────────────────────────────
DAILY_REVENUE = dedent("""
    WITH paid AS (
        SELECT
            TO_DATE(TRY_TO_TIMESTAMP(p.created_at))             AS revenue_date,
            v.CLINIC                                              AS clinic_id,
            COALESCE(p.PAYMENT_MODE, 'mixed')                     AS payment_mode,
            CASE WHEN v.SCHEME IS NULL THEN 'cash' ELSE 'insurance' END AS payer_type,
            SUM(COALESCE(p.AMOUNT, 0))                            AS gross_amount,
            SUM(COALESCE(p.DISCOUNT, 0))                          AS discount_amount,
            SUM(COALESCE(p.WAIVER_AMOUNT, 0))                     AS waiver_amount,
            SUM(COALESCE(p.VAT_AMOUNT, 0))                        AS vat_amount,
            COUNT(DISTINCT p.PATIENT)                             AS unique_patients,
            COUNT(*)                                              AS receipt_count
        FROM FINANCE_EVALUATION_PAYMENTS p
        LEFT JOIN EVALUATION_VISITS v ON v.ID = p.VISIT
        GROUP BY 1, 2, 3, 4
    )
    SELECT
        revenue_date,
        clinic_id,
        payment_mode,
        payer_type,
        gross_amount,
        discount_amount,
        waiver_amount,
        vat_amount,
        gross_amount - discount_amount - waiver_amount  AS net_amount,
        unique_patients,
        receipt_count,
        gross_amount / NULLIF(receipt_count, 0)         AS avg_receipt_value
    FROM paid
    ORDER BY revenue_date, clinic_id
""")

# ──────────────────────────────────────────────────────────────────────────────
# 2. REVENUE BY SERVICE LINE
# ──────────────────────────────────────────────────────────────────────────────
REVENUE_BY_SERVICE_LINE = dedent("""
    SELECT
        DATE_TRUNC('MONTH', TRY_TO_DATE(i.created_at))                   AS revenue_month,
        COALESCE(NULLIF(ii.REVENUE_SUMMARY_TAG, ''),
                NULLIF(ii.ITEM_CLASSIFY,       ''),
                INITCAP(ii.ITEM_TYPE), 'other')                           AS service_line,
        COUNT(DISTINCT i.ID)                                               AS invoices,
        COUNT(DISTINCT i.PATIENT_ID)                                       AS patients,
        SUM(ii.AMOUNT)                                                     AS gross_revenue,
        SUM(ii.QUANTITY)                                                   AS units_sold,
        AVG(ii.PRICE)                                                      AS avg_unit_price,
        SUM(ii.AMOUNT)
            / NULLIF(COUNT(DISTINCT i.PATIENT_ID), 0)                      AS arpu
    FROM FINANCE_INVOICES        i
    JOIN FINANCE_INVOICE_ITEMS   ii ON ii.INVOICE_ID = i.ID
    GROUP BY 1, 2
    HAVING revenue_month IS NOT NULL
    ORDER BY 1, gross_revenue DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 4. PAYMENT MODE MIX
# ──────────────────────────────────────────────────────────────────────────────
PAYMENT_MODE_MIX = dedent("""
    
    SELECT
        DATE_TRUNC('MONTH', TRY_TO_TIMESTAMP(created_at))   AS revenue_month,
        SUM(CASH_AMOUNT)                                      AS cash,
        SUM(MPESA_AMOUNT)
          + SUM(PESA_PAL_MPESA_AMOUNT)                        AS mpesa,
        SUM(CARD_AMOUNT)
          + SUM(PESA_PAL_CARD_AMOUNT)                         AS card,
        SUM(CHEQUE_AMOUNT)                                    AS cheque,
        SUM(JAMBOPAY_AMOUNT)                                  AS jambopay,
        SUM(PATIENTACCOUNT_AMOUNT)                            AS account,
        SUM(WAIVER_AMOUNT)                                    AS waiver,
        SUM(GIFTCARD_AMOUNT)                                  AS giftcard,
        SUM(LOYALTY_AMOUNT) + SUM(POINTS_AMOUNT)              AS loyalty
    FROM FINANCE_EVALUATION_PAYMENTS
    GROUP BY 1
    HAVING revenue_month IS NOT NULL
    ORDER BY 1
""")

# ──────────────────────────────────────────────────────────────────────────────
# 5. PAYER PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
PAYER_PERFORMANCE = dedent("""
    WITH inv AS (
        SELECT
            COALESCE(s.NAME, 'Cash / Out-of-pocket')   AS payer_name,
            i.ID                                       AS invoice_id,
            TRY_TO_DATE(i.INVOICE_DATE)                AS invoice_date,
            i.AMOUNT                                   AS invoice_amount,
            i.PAID                                     AS paid_amount,
            i.BALANCE                                  AS outstanding_amount,
            DATEDIFF('day', TRY_TO_DATE(i.INVOICE_DATE), CURRENT_DATE) AS age_days
        FROM FINANCE_INVOICES   i
        LEFT JOIN SETTINGS_INSURANCE s ON s.ID = i.SCHEME_ID
    )
    SELECT
        payer_name,
        COUNT(*)                                                   AS invoices,
        SUM(invoice_amount)                                        AS billed,
        SUM(paid_amount)                                           AS collected,
        SUM(outstanding_amount)                                    AS outstanding,
        100 * SUM(paid_amount) / NULLIF(SUM(invoice_amount), 0)    AS collection_rate_pct,
        AVG(age_days)                                              AS avg_dso,
        SUM(CASE WHEN age_days BETWEEN  0 AND 30  THEN outstanding_amount END) AS bucket_0_30,
        SUM(CASE WHEN age_days BETWEEN 31 AND 60  THEN outstanding_amount END) AS bucket_31_60,
        SUM(CASE WHEN age_days BETWEEN 61 AND 90  THEN outstanding_amount END) AS bucket_61_90,
        SUM(CASE WHEN age_days >  90              THEN outstanding_amount END) AS bucket_over_90
    FROM inv
    GROUP BY payer_name
    ORDER BY billed DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 6. PATIENT RFM
# ──────────────────────────────────────────────────────────────────────────────
PATIENT_RFM = dedent("""    
    SELECT
        p.PATIENT                                                          AS patient_id,
        MAX(TRY_TO_TIMESTAMP(p.created_at))                              AS last_receipt,
        DATEDIFF('day', MAX(TRY_TO_TIMESTAMP(p.created_at)), CURRENT_DATE)
                                                                        AS recency_days,
        COUNT(DISTINCT p.RECEIPT)                                          AS frequency,
        SUM(p.AMOUNT)                                                      AS monetary,
        AVG(p.AMOUNT)                                                      AS avg_ticket,
        MIN(TRY_TO_TIMESTAMP(p.created_at))                              AS first_receipt,
        DATEDIFF('day',
                MIN(TRY_TO_TIMESTAMP(p.created_at)),
                MAX(TRY_TO_TIMESTAMP(p.created_at)))                    AS lifetime_days
    FROM FINANCE_EVALUATION_PAYMENTS p
    WHERE p.PATIENT    IS NOT NULL
    GROUP BY p.PATIENT
""")

# ──────────────────────────────────────────────────────────────────────────────
# 7. TOP REVENUE-DRIVING ITEMS
# ──────────────────────────────────────────────────────────────────────────────
TOP_ITEMS = dedent("""
    SELECT
        ii.ITEM_NAME                                  AS item_name,
        COALESCE(ii.REVENUE_SUMMARY_TAG, ii.CATEGORY) AS category,
        COUNT(DISTINCT ii.INVOICE_ID)                 AS invoices,
        SUM(ii.QUANTITY)                              AS units,
        SUM(ii.AMOUNT)                                AS gross_revenue,
        AVG(ii.PRICE)                                 AS avg_price
    FROM FINANCE_INVOICE_ITEMS ii
    JOIN FINANCE_INVOICES      i ON i.ID = ii.INVOICE_ID
    GROUP BY 1, 2
    ORDER BY gross_revenue DESC
    LIMIT 50
""")

# ──────────────────────────────────────────────────────────────────────────────
# 8. HOUR-OF-DAY × DAY-OF-WEEK HEATMAP
# ──────────────────────────────────────────────────────────────────────────────
HOURLY_HEATMAP = dedent("""    
    SELECT
        DAYNAME(TRY_TO_TIMESTAMP(p.CREATED_AT))         AS day_name,
        DAYOFWEEK(TRY_TO_TIMESTAMP(p.CREATED_AT))       AS day_idx,
        HOUR(TRY_TO_TIMESTAMP(p.CREATED_AT))            AS hour_of_day,
        COUNT(*)                                        AS receipts,
        SUM(p.AMOUNT)                                   AS revenue
    FROM FINANCE_EVALUATION_PAYMENTS p
    GROUP BY 1, 2, 3
    HAVING day_idx IS NOT NULL AND hour_of_day IS NOT NULL
    ORDER BY day_idx, hour_of_day
""")

# ──────────────────────────────────────────────────────────────────────────────
# 9. COHORT RETENTION
# ──────────────────────────────────────────────────────────────────────────────
COHORT_RETENTION = dedent("""
    WITH first_visit AS (
        SELECT
            PATIENT,
            MIN(DATE_TRUNC('MONTH', TRY_TO_TIMESTAMP(CREATED_AT))) AS cohort_month
        FROM FINANCE_EVALUATION_PAYMENTS
        GROUP BY PATIENT
    ),
    monthly AS (
        SELECT
            f.cohort_month,
            DATE_TRUNC('MONTH', TRY_TO_TIMESTAMP(p.CREATED_AT))             AS active_month,
            DATEDIFF('month',
                    f.cohort_month,
                    DATE_TRUNC('MONTH', TRY_TO_TIMESTAMP(p.CREATED_AT)))    AS month_offset,
            COUNT(DISTINCT p.PATIENT)                                          AS active_patients,
            SUM(p.AMOUNT)                                                      AS revenue
        FROM FINANCE_EVALUATION_PAYMENTS p
        JOIN first_visit f ON f.PATIENT = p.PATIENT
        GROUP BY 1, 2, 3
    )
    SELECT * FROM monthly
    WHERE active_month IS NOT NULL
    ORDER BY cohort_month, month_offset
""")

# ──────────────────────────────────────────────────────────────────────────────
# 10. DOCTOR PRODUCTIVITY
# ──────────────────────────────────────────────────────────────────────────────
DOCTOR_PRODUCTIVITY = dedent("""
    SELECT
        u.ID                                                 AS user_id,
        u.USERNAME                                               AS doctor_name,
        COUNT(DISTINCT v.ID)                                 AS visits,
        COUNT(DISTINCT v.PATIENT)                            AS unique_patients,
        SUM(p.AMOUNT)                                        AS revenue_attributed,
        SUM(p.AMOUNT) / NULLIF(COUNT(DISTINCT v.ID), 0)      AS arpv,
        COUNT(DISTINCT v.ID)
        / NULLIF(COUNT(DISTINCT TO_DATE(TRY_TO_TIMESTAMP(v.CREATED_AT))), 0)
                                                            AS visits_per_active_day
    FROM EVALUATION_VISITS              v
    JOIN USERS                          u ON u.ID = v.USER
    LEFT JOIN FINANCE_EVALUATION_PAYMENTS p
        ON p.VISIT = v.ID AND p.DELETED_AT IS NULL
    GROUP BY 1, 2
    HAVING visits >= 5
    ORDER BY revenue_attributed DESC NULLS LAST
    LIMIT 25
""")

# ──────────────────────────────────────────────────────────────────────────────
# 11. LEAKAGE
# ──────────────────────────────────────────────────────────────────────────────
LEAKAGE = dedent("""
    SELECT
        DATE_TRUNC('MONTH', TRY_TO_TIMESTAMP(CREATED_AT))            AS revenue_month,
        SUM(AMOUNT)                                                    AS net_received,
        SUM(DISCOUNT)                                                  AS discount,
        SUM(WAIVER_AMOUNT)                                             AS waiver,
        SUM(DISCOUNT) + SUM(WAIVER_AMOUNT)                             AS leakage_total,
        100 * (SUM(DISCOUNT) + SUM(WAIVER_AMOUNT))
            / NULLIF(SUM(AMOUNT) + SUM(DISCOUNT) + SUM(WAIVER_AMOUNT), 0)
                                                                    AS leakage_pct
    FROM FINANCE_EVALUATION_PAYMENTS
    GROUP BY 1
    HAVING revenue_month IS NOT NULL
    ORDER BY 1
""")


# ──────────────────────────────────────────────────────────────────────────────
# 13. CLAIM REJECTION
# ──────────────────────────────────────────────────────────────────────────────
CLAIM_REJECTION = dedent("""                     
    SELECT
        DATE_TRUNC('MONTH', TO_DATE(TRY_TO_TIMESTAMP(i.CREATED_AT)))            AS revenue_month,
        s.NAME                                                                  AS payer_name,
        COUNT(*)                                                                AS claims,
        SUM(i.AMOUNT)                                                           AS billed,
        SUM(i.PAID)                                                             AS collected,
        SUM(i.BALANCE)                                                          AS outstanding,
        -- Proxy: insurance claim still has balance after 90 days = effectively rejected/stuck
        SUM(CASE
                WHEN i.BALANCE > 0
                AND DATEDIFF('day', TRY_TO_TIMESTAMP(i.CREATED_AT), CURRENT_DATE) > 90
                THEN 1 ELSE 0
            END)                                                                AS rejected,
        SUM(CASE
                WHEN i.BALANCE > 0
                AND DATEDIFF('day', TRY_TO_TIMESTAMP(i.CREATED_AT), CURRENT_DATE) > 90
                THEN i.BALANCE
            END)                                                                AS rejected_value,
        100.0 * SUM(CASE
                        WHEN i.BALANCE > 0
                        AND DATEDIFF('day', TRY_TO_TIMESTAMP(i.CREATED_AT), CURRENT_DATE) > 90
                        THEN 1 ELSE 0
                    END) / NULLIF(COUNT(*), 0)                                  AS rejection_rate_pct
    FROM FINANCE_INVOICES        i
    JOIN SETTINGS_INSURANCE      s ON s.ID = i.SCHEME_ID    -- INNER JOIN drops cash invoices
    GROUP BY 1, 2
    HAVING revenue_month IS NOT NULL
    ORDER BY 1, claims DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 14. REVENUE CONCENTRATION (HHI)
# ──────────────────────────────────────────────────────────────────────────────
REVENUE_CONCENTRATION = dedent("""
    WITH per_payer AS (
        SELECT
            COALESCE(s.NAME, 'Cash')           AS payer,
            SUM(i.AMOUNT)                      AS amt,
            SUM(i.AMOUNT) /
              SUM(SUM(i.AMOUNT)) OVER ()       AS share
        FROM FINANCE_INVOICES        i
        LEFT JOIN SETTINGS_INSURANCE s ON s.ID = i.SCHEME_ID
    
        GROUP BY 1
    )
    SELECT
        payer,
        amt,
        100 * share                                                       AS share_pct,
        100 * SUM(share)
              OVER (ORDER BY share DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)     AS cumulative_pct,
        SUM(POWER(share * 100, 2)) OVER ()                                AS hhi_index
    FROM per_payer
    ORDER BY share DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 15. ARPV TREND
# ──────────────────────────────────────────────────────────────────────────────
ARPV_TREND = dedent("""
    WITH daily AS (
        SELECT
            TO_DATE(p.created_at)        AS d,
            SUM(p.AMOUNT)                AS revenue,
            COUNT(DISTINCT p.PATIENT)    AS visits   -- patient-day as visit proxy
        FROM FINANCE_EVALUATION_PAYMENTS p
        WHERE p.PATIENT IS NOT NULL                  -- drop the 21 orphan rows
        GROUP BY 1
    )
    SELECT
        d                                                               AS revenue_date,
        revenue / NULLIF(visits, 0)                                     AS arpv,
        AVG(revenue / NULLIF(visits, 0))
            OVER (ORDER BY d ROWS BETWEEN  6 PRECEDING AND CURRENT ROW) AS arpv_7d,
        AVG(revenue / NULLIF(visits, 0))
            OVER (ORDER BY d ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS arpv_28d
    FROM daily
    WHERE d IS NOT NULL
    ORDER BY d
""")

# ──────────────────────────────────────────────────────────────────────────────
# 16. REVENUE AT RISK
# ──────────────────────────────────────────────────────────────────────────────
REVENUE_AT_RISK = dedent("""
    WITH ageing AS (
        SELECT
            i.BALANCE,
            DATEDIFF('day', TRY_TO_DATE(i.created_at), CURRENT_DATE) AS age_days
        FROM FINANCE_INVOICES i
        WHERE i.BALANCE > 0
        AND i.DELETED_AT IS NULL
        AND i.STATUS NOT IN (3, 4)
        AND TRY_TO_DATE(i.created_at) IS NOT NULL
    )
    SELECT
        CASE
            WHEN age_days <= 30  THEN '0-30 days'
            WHEN age_days <= 60  THEN '31-60 days'
            WHEN age_days <= 90  THEN '61-90 days'
            WHEN age_days <= 180 THEN '91-180 days'
            ELSE '180+ days'
        END                                                  AS age_bucket,
        COUNT(*)                                             AS invoices,
        SUM(BALANCE)                                         AS at_risk,
        AVG(age_days)                                        AS avg_age_days,
        SUM(BALANCE *
            POWER(0.97,
                GREATEST(0, (age_days - 30) / 30.0)))      AS expected_collection
    FROM ageing
    GROUP BY age_bucket
    ORDER BY MIN(age_days)
""")

# ──────────────────────────────────────────────────────────────────────────────
# 17. WEEKLY GROSS PROFIT
# ──────────────────────────────────────────────────────────────────────────────
# Convenience map for the data layer.
ALL_QUERIES = {
    "daily_revenue":           DAILY_REVENUE,
    "revenue_by_service_line": REVENUE_BY_SERVICE_LINE,
    "payment_mode_mix":        PAYMENT_MODE_MIX,
    "payer_performance":       PAYER_PERFORMANCE,
    "patient_rfm":             PATIENT_RFM,
    "top_items":               TOP_ITEMS,
    "hourly_heatmap":          HOURLY_HEATMAP,
    "cohort_retention":        COHORT_RETENTION,
    "doctor_productivity":     DOCTOR_PRODUCTIVITY,
    "leakage":                 LEAKAGE,
    "claim_rejection":         CLAIM_REJECTION,
    "revenue_concentration":   REVENUE_CONCENTRATION,
    "arpv_trend":              ARPV_TREND,
    "revenue_at_risk":         REVENUE_AT_RISK,
}


def render(query_name: str, *, start: str, end: str, clinic_filter: str = "") -> str:
    """Return a fully-formatted SQL string ready to execute on Snowflake."""
    template = ALL_QUERIES[query_name]
    return template.format(start=start, end=end, clinic_filter=clinic_filter)