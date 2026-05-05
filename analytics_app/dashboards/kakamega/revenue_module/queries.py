"""
queries.py
----------
Snowflake SQL queries for the PharmaPlus Revenue Intelligence dashboard.

These assume the operational MySQL schema has been replicated into Snowflake
into the schema referenced by SnowflakeClient (default PUBLIC). Tables are
referenced unqualified — Snowflake will resolve them against the configured
DATABASE.SCHEMA.

Each query is parameterised with two date placeholders:
    {start}   inclusive (e.g. '2024-05-01')
    {end}     exclusive (e.g. '2026-05-01')
and (where relevant) {clinic_filter} which is either '' (no filter) or a
fragment such as "AND v.CLINIC IN (1,2,3)".

All amount columns are returned as KSh (Kenyan Shillings).
"""

from textwrap import dedent

# ──────────────────────────────────────────────────────────────────────────────
# 1. DAILY REVENUE TREND  (the workhorse time-series)
# ──────────────────────────────────────────────────────────────────────────────
DAILY_REVENUE = dedent("""
    WITH paid AS (
        SELECT
            DATE(p.RECEIPT_DATE)                              AS revenue_date,
            v.CLINIC                                          AS clinic_id,
            COALESCE(p.PAYMENT_MODE, 'mixed')                 AS payment_mode,
            CASE WHEN v.SCHEME IS NULL THEN 'cash' ELSE 'insurance' END AS payer_type,
            SUM(COALESCE(p.AMOUNT, 0))                        AS gross_amount,
            SUM(COALESCE(p.DISCOUNT, 0))                      AS discount_amount,
            SUM(COALESCE(p.WAIVER_AMOUNT, 0))                 AS waiver_amount,
            SUM(COALESCE(p.VAT_AMOUNT, 0))                    AS vat_amount,
            COUNT(DISTINCT p.PATIENT)                         AS unique_patients,
            COUNT(*)                                          AS receipt_count
        FROM FINANCE_EVALUATION_PAYMENTS p
        LEFT JOIN EVALUATION_VISITS v ON v.ID = p.VISIT
        WHERE p.RECEIPT_DATE >= '{start}'
          AND p.RECEIPT_DATE <  '{end}'
          AND p.DELETED_AT IS NULL
          AND COALESCE(p.STATUS, '') NOT IN ('cancelled', 'voided')
          {clinic_filter}
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
# 2. REVENUE BY SERVICE LINE  (consultation / pharmacy / lab / theatre / etc.)
# ──────────────────────────────────────────────────────────────────────────────
REVENUE_BY_SERVICE_LINE = dedent("""
    SELECT
        DATE_TRUNC('MONTH', i.INVOICE_DATE)                                AS revenue_month,
        COALESCE(NULLIF(ii.REVENUE_SUMMARY_TAG, ''),
                 NULLIF(ii.ITEM_CLASSIFY, ''),
                 INITCAP(ii.ITEM_TYPE), 'other')                           AS service_line,
        COUNT(DISTINCT i.ID)                                               AS invoices,
        COUNT(DISTINCT i.PATIENT_ID)                                       AS patients,
        SUM(ii.AMOUNT)                                                     AS gross_revenue,
        SUM(ii.QUANTITY)                                                   AS units_sold,
        AVG(ii.PRICE)                                                      AS avg_unit_price,
        SUM(ii.AMOUNT) /
            NULLIF(COUNT(DISTINCT i.PATIENT_ID), 0)                        AS arpu
    FROM FINANCE_INVOICES        i
    JOIN FINANCE_INVOICE_ITEMS   ii ON ii.INVOICE_ID = i.ID
    WHERE i.INVOICE_DATE >= '{start}'
      AND i.INVOICE_DATE <  '{end}'
      AND i.DELETED_AT IS NULL
      AND ii.DELETED_AT IS NULL
    GROUP BY 1, 2
    ORDER BY 1, gross_revenue DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 3. REVENUE BY BRANCH  (clinic ranking + month-over-month deltas)
# ──────────────────────────────────────────────────────────────────────────────
REVENUE_BY_BRANCH = dedent("""
    WITH base AS (
        SELECT
            DATE_TRUNC('MONTH', p.RECEIPT_DATE)               AS revenue_month,
            sc.ID                                             AS clinic_id,
            sc.NAME                                           AS clinic_name,
            sc.TOWN                                           AS town,
            SUM(p.AMOUNT)                                     AS revenue,
            COUNT(DISTINCT p.PATIENT)                         AS patients,
            COUNT(DISTINCT p.VISIT)                           AS visits
        FROM FINANCE_EVALUATION_PAYMENTS p
        JOIN EVALUATION_VISITS  v  ON v.ID  = p.VISIT
        JOIN SETTINGS_CLINICS   sc ON sc.ID = v.CLINIC
        WHERE p.RECEIPT_DATE >= '{start}'
          AND p.RECEIPT_DATE <  '{end}'
          AND p.DELETED_AT IS NULL
          AND sc.STATUS = 'active'
        GROUP BY 1, 2, 3, 4
    )
    SELECT
        revenue_month,
        clinic_id,
        clinic_name,
        town,
        revenue,
        patients,
        visits,
        revenue / NULLIF(visits,   0)  AS arpv,            -- avg revenue per visit
        revenue / NULLIF(patients, 0)  AS arpu,            -- avg revenue per unique patient
        revenue
          - LAG(revenue) OVER (PARTITION BY clinic_id ORDER BY revenue_month)
                                       AS mom_change_abs,
        100 * (revenue
          - LAG(revenue) OVER (PARTITION BY clinic_id ORDER BY revenue_month))
          / NULLIF(LAG(revenue) OVER (PARTITION BY clinic_id ORDER BY revenue_month), 0)
                                       AS mom_change_pct
    FROM base
    ORDER BY revenue_month, revenue DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 4. PAYMENT MODE MIX  (cash / mpesa / card / cheque / insurance)
# ──────────────────────────────────────────────────────────────────────────────
PAYMENT_MODE_MIX = dedent("""
    SELECT
        DATE_TRUNC('MONTH', RECEIPT_DATE) AS revenue_month,
        SUM(CASH_AMOUNT)                  AS cash,
        SUM(MPESA_AMOUNT)
          + SUM(PESA_PAL_MPESA_AMOUNT)    AS mpesa,
        SUM(CARD_AMOUNT)
          + SUM(PESA_PAL_CARD_AMOUNT)     AS card,
        SUM(CHEQUE_AMOUNT)                AS cheque,
        SUM(JAMBOPAY_AMOUNT)              AS jambopay,
        SUM(PATIENTACCOUNT_AMOUNT)        AS account,
        SUM(WAIVER_AMOUNT)                AS waiver,
        SUM(GIFTCARD_AMOUNT)              AS giftcard,
        SUM(LOYALTY_AMOUNT)
          + SUM(POINTS_AMOUNT)            AS loyalty
    FROM FINANCE_EVALUATION_PAYMENTS
    WHERE RECEIPT_DATE >= '{start}'
      AND RECEIPT_DATE <  '{end}'
      AND DELETED_AT IS NULL
    GROUP BY 1
    ORDER BY 1
""")

# ──────────────────────────────────────────────────────────────────────────────
# 5. PAYER (INSURANCE SCHEME) PERFORMANCE — DSO, claim mix, AR ageing
# ──────────────────────────────────────────────────────────────────────────────
PAYER_PERFORMANCE = dedent("""
    WITH inv AS (
        SELECT
            COALESCE(s.NAME, 'Cash / Out-of-pocket')   AS payer_name,
            i.ID                                       AS invoice_id,
            i.INVOICE_DATE                             AS invoice_date,
            i.AMOUNT                                   AS invoice_amount,
            i.PAID                                     AS paid_amount,
            i.BALANCE                                  AS outstanding_amount,
            CURRENT_DATE - i.INVOICE_DATE              AS age_days
        FROM FINANCE_INVOICES   i
        LEFT JOIN SETTINGS_INSURANCE s ON s.ID = i.SCHEME_ID
        WHERE i.INVOICE_DATE >= '{start}'
          AND i.INVOICE_DATE <  '{end}'
          AND i.DELETED_AT IS NULL
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
# 6. PATIENT RFM TABLE  (recency, frequency, monetary)
# ──────────────────────────────────────────────────────────────────────────────
PATIENT_RFM = dedent("""
    SELECT
        p.PATIENT                                                 AS patient_id,
        MAX(p.RECEIPT_DATE)                                       AS last_receipt,
        DATEDIFF('day', MAX(p.RECEIPT_DATE), CURRENT_DATE)        AS recency_days,
        COUNT(DISTINCT p.RECEIPT)                                 AS frequency,
        SUM(p.AMOUNT)                                             AS monetary,
        AVG(p.AMOUNT)                                             AS avg_ticket,
        MIN(p.RECEIPT_DATE)                                       AS first_receipt,
        DATEDIFF('day', MIN(p.RECEIPT_DATE),
                        MAX(p.RECEIPT_DATE))                      AS lifetime_days
    FROM FINANCE_EVALUATION_PAYMENTS p
    WHERE p.RECEIPT_DATE >= '{start}'
      AND p.RECEIPT_DATE <  '{end}'
      AND p.DELETED_AT IS NULL
      AND p.PATIENT IS NOT NULL
    GROUP BY p.PATIENT
""")

# ──────────────────────────────────────────────────────────────────────────────
# 7. TOP REVENUE-DRIVING PROCEDURES / ITEMS  (Pareto)
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
    WHERE i.INVOICE_DATE >= '{start}'
      AND i.INVOICE_DATE <  '{end}'
      AND i.DELETED_AT IS NULL
      AND ii.DELETED_AT IS NULL
    GROUP BY 1, 2
    ORDER BY gross_revenue DESC
    LIMIT 50
""")

# ──────────────────────────────────────────────────────────────────────────────
# 8. HOUR-OF-DAY × DAY-OF-WEEK HEATMAP  (operations / staffing analysis)
# ──────────────────────────────────────────────────────────────────────────────
HOURLY_HEATMAP = dedent("""
    SELECT
        DAYNAME(p.CREATED_AT)        AS day_name,
        DAYOFWEEK(p.CREATED_AT)      AS day_idx,
        HOUR(p.CREATED_AT)           AS hour_of_day,
        COUNT(*)                     AS receipts,
        SUM(p.AMOUNT)                AS revenue
    FROM FINANCE_EVALUATION_PAYMENTS p
    WHERE p.RECEIPT_DATE >= '{start}'
      AND p.RECEIPT_DATE <  '{end}'
      AND p.DELETED_AT IS NULL
    GROUP BY 1, 2, 3
    ORDER BY day_idx, hour_of_day
""")

# ──────────────────────────────────────────────────────────────────────────────
# 9. NEW vs RETURNING PATIENT REVENUE  (cohort retention)
# ──────────────────────────────────────────────────────────────────────────────
COHORT_RETENTION = dedent("""
    WITH first_visit AS (
        SELECT PATIENT, MIN(DATE_TRUNC('MONTH', RECEIPT_DATE)) AS cohort_month
        FROM FINANCE_EVALUATION_PAYMENTS
        WHERE DELETED_AT IS NULL AND PATIENT IS NOT NULL
        GROUP BY PATIENT
    ),
    monthly AS (
        SELECT
            f.cohort_month,
            DATE_TRUNC('MONTH', p.RECEIPT_DATE)                         AS active_month,
            DATEDIFF('month', f.cohort_month,
                              DATE_TRUNC('MONTH', p.RECEIPT_DATE))      AS month_offset,
            COUNT(DISTINCT p.PATIENT)                                   AS active_patients,
            SUM(p.AMOUNT)                                               AS revenue
        FROM FINANCE_EVALUATION_PAYMENTS p
        JOIN first_visit f ON f.PATIENT = p.PATIENT
        WHERE p.RECEIPT_DATE >= '{start}'
          AND p.RECEIPT_DATE <  '{end}'
          AND p.DELETED_AT IS NULL
        GROUP BY 1, 2, 3
    )
    SELECT * FROM monthly ORDER BY cohort_month, month_offset
""")

# ──────────────────────────────────────────────────────────────────────────────
# 10. DOCTOR (PROVIDER) PRODUCTIVITY  — revenue, visits, ARPV
# ──────────────────────────────────────────────────────────────────────────────
DOCTOR_PRODUCTIVITY = dedent("""
    SELECT
        u.ID                                          AS user_id,
        u.NAME                                        AS doctor_name,
        COUNT(DISTINCT v.ID)                          AS visits,
        COUNT(DISTINCT v.PATIENT)                     AS unique_patients,
        SUM(p.AMOUNT)                                 AS revenue_attributed,
        SUM(p.AMOUNT) / NULLIF(COUNT(DISTINCT v.ID), 0) AS arpv,
        COUNT(DISTINCT v.ID)
          / NULLIF(COUNT(DISTINCT DATE(v.CREATED_AT)), 0) AS visits_per_active_day
    FROM EVALUATION_VISITS              v
    JOIN USERS                          u ON u.ID = v.USER
    LEFT JOIN FINANCE_EVALUATION_PAYMENTS p ON p.VISIT = v.ID AND p.DELETED_AT IS NULL
    WHERE v.CREATED_AT >= '{start}'
      AND v.CREATED_AT <  '{end}'
      AND v.DELETED_AT IS NULL
    GROUP BY 1, 2
    HAVING visits >= 5
    ORDER BY revenue_attributed DESC
    LIMIT 25
""")

# ──────────────────────────────────────────────────────────────────────────────
# 11. DISCOUNT / WAIVER LEAKAGE  — revenue lost to discounts and write-offs
# ──────────────────────────────────────────────────────────────────────────────
LEAKAGE = dedent("""
    SELECT
        DATE_TRUNC('MONTH', RECEIPT_DATE)                              AS revenue_month,
        SUM(AMOUNT)                                                    AS net_received,
        SUM(DISCOUNT)                                                  AS discount,
        SUM(WAIVER_AMOUNT)                                             AS waiver,
        SUM(DISCOUNT) + SUM(WAIVER_AMOUNT)                             AS leakage_total,
        100 * (SUM(DISCOUNT) + SUM(WAIVER_AMOUNT))
            / NULLIF(SUM(AMOUNT)
                   + SUM(DISCOUNT)
                   + SUM(WAIVER_AMOUNT), 0)                            AS leakage_pct
    FROM FINANCE_EVALUATION_PAYMENTS
    WHERE RECEIPT_DATE >= '{start}'
      AND RECEIPT_DATE <  '{end}'
      AND DELETED_AT IS NULL
    GROUP BY 1
    ORDER BY 1
""")

# ──────────────────────────────────────────────────────────────────────────────
# 12. INVENTORY MARGIN  — selling price vs cost (top SKUs)
# ──────────────────────────────────────────────────────────────────────────────
INVENTORY_MARGIN = dedent("""
    SELECT
        ip.NAME                                                  AS product,
        ic.NAME                                                  AS category,
        ip.SELLING_PRICE                                         AS selling_price,
        ip.INSURANCE_PRICE                                       AS insurance_price,
        AVG(ib.UNIT_COST)                                        AS avg_cost,
        ip.SELLING_PRICE - AVG(ib.UNIT_COST)                     AS margin_abs,
        100 * (ip.SELLING_PRICE - AVG(ib.UNIT_COST))
              / NULLIF(ip.SELLING_PRICE, 0)                      AS margin_pct,
        SUM(ibs.AMOUNT)                                          AS revenue_ttm
    FROM INVENTORY_PRODUCTS  ip
    JOIN INVENTORY_CATEGORIES ic ON ic.ID = ip.CATEGORY
    LEFT JOIN INVENTORY_BATCHES      ib  ON ib.PRODUCT_ID = ip.ID
    LEFT JOIN INVENTORY_BATCH_SALES  ibs ON ibs.ID         = ib.SALE_ID
        AND ibs.CREATED_AT >= '{start}'
        AND ibs.CREATED_AT <  '{end}'
    WHERE ip.ACTIVE = 1
      AND ip.DELETED_AT IS NULL
    GROUP BY 1, 2, 3, 4
    ORDER BY revenue_ttm DESC NULLS LAST
    LIMIT 100
""")

# ──────────────────────────────────────────────────────────────────────────────
# 13. CLAIM REJECTION / DENIAL RATE
# ──────────────────────────────────────────────────────────────────────────────
CLAIM_REJECTION = dedent("""
    SELECT
        DATE_TRUNC('MONTH', i.INVOICE_DATE)            AS revenue_month,
        s.NAME                                         AS payer_name,
        COUNT(*)                                       AS claims,
        SUM(CASE WHEN i.STATUS = 4 THEN 1 ELSE 0 END)  AS rejected,
        100 * SUM(CASE WHEN i.STATUS = 4 THEN 1 ELSE 0 END)
              / NULLIF(COUNT(*), 0)                    AS rejection_rate_pct,
        SUM(CASE WHEN i.STATUS = 4 THEN i.AMOUNT END)  AS rejected_value
    FROM FINANCE_INVOICES         i
    LEFT JOIN SETTINGS_INSURANCE  s ON s.ID = i.SCHEME_ID
    WHERE i.INVOICE_DATE >= '{start}'
      AND i.INVOICE_DATE <  '{end}'
      AND i.SCHEME_ID    IS NOT NULL
      AND i.DELETED_AT   IS NULL
    GROUP BY 1, 2
    ORDER BY 1, claims DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 14. REVENUE CONCENTRATION (HHI / top-N share)
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
        WHERE i.INVOICE_DATE >= '{start}'
          AND i.INVOICE_DATE <  '{end}'
          AND i.DELETED_AT IS NULL
        GROUP BY 1
    )
    SELECT
        payer,
        amt,
        100 * share                                            AS share_pct,
        100 * SUM(share)
              OVER (ORDER BY share DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)  AS cumulative_pct,
        SUM(POWER(share * 100, 2)) OVER ()                     AS hhi_index
    FROM per_payer
    ORDER BY share DESC
""")

# ──────────────────────────────────────────────────────────────────────────────
# 15. AVERAGE REVENUE PER VISIT (ARPV) — daily, smoothed
# ──────────────────────────────────────────────────────────────────────────────
ARPV_TREND = dedent("""
    WITH daily AS (
        SELECT
            DATE(p.RECEIPT_DATE)          AS d,
            SUM(p.AMOUNT)                 AS revenue,
            COUNT(DISTINCT p.VISIT)       AS visits
        FROM FINANCE_EVALUATION_PAYMENTS p
        WHERE p.RECEIPT_DATE >= '{start}'
          AND p.RECEIPT_DATE <  '{end}'
          AND p.DELETED_AT IS NULL
        GROUP BY 1
    )
    SELECT
        d                                                            AS revenue_date,
        revenue / NULLIF(visits, 0)                                  AS arpv,
        AVG(revenue / NULLIF(visits, 0))
            OVER (ORDER BY d ROWS BETWEEN  6 PRECEDING AND CURRENT ROW) AS arpv_7d,
        AVG(revenue / NULLIF(visits, 0))
            OVER (ORDER BY d ROWS BETWEEN 27 PRECEDING AND CURRENT ROW) AS arpv_28d
    FROM daily
    ORDER BY d
""")

# ──────────────────────────────────────────────────────────────────────────────
# 16. REVENUE AT RISK  — overdue invoices weighted by age
# ──────────────────────────────────────────────────────────────────────────────
REVENUE_AT_RISK = dedent("""
    SELECT
        CASE
            WHEN CURRENT_DATE - i.INVOICE_DATE <= 30  THEN '0-30 days'
            WHEN CURRENT_DATE - i.INVOICE_DATE <= 60  THEN '31-60 days'
            WHEN CURRENT_DATE - i.INVOICE_DATE <= 90  THEN '61-90 days'
            WHEN CURRENT_DATE - i.INVOICE_DATE <= 180 THEN '91-180 days'
            ELSE '180+ days'
        END                                                  AS age_bucket,
        COUNT(*)                                             AS invoices,
        SUM(i.BALANCE)                                       AS at_risk,
        AVG(CURRENT_DATE - i.INVOICE_DATE)                   AS avg_age_days,
        -- discount future receivables: ~3% per month past 30 days
        SUM(i.BALANCE *
            POWER(0.97,
              GREATEST(0, (CURRENT_DATE - i.INVOICE_DATE - 30) / 30.0))) AS expected_collection
    FROM FINANCE_INVOICES i
    WHERE i.BALANCE > 0
      AND i.DELETED_AT IS NULL
      AND i.STATUS NOT IN (3, 4)        -- exclude voided / rejected
    GROUP BY age_bucket
    ORDER BY MIN(CURRENT_DATE - i.INVOICE_DATE)
""")

# ──────────────────────────────────────────────────────────────────────────────
# 17. WEEKLY GROSS PROFIT (proxy)  — revenue minus inventory cost
# ──────────────────────────────────────────────────────────────────────────────
GROSS_PROFIT_WEEKLY = dedent("""
    SELECT
        DATE_TRUNC('WEEK', ibs.CREATED_AT)               AS week,
        SUM(ibs.AMOUNT)                                  AS revenue,
        SUM(ib.UNIT_COST * ib.QUANTITY)                  AS cogs,
        SUM(ibs.AMOUNT)
          - SUM(ib.UNIT_COST * ib.QUANTITY)              AS gross_profit,
        100 * (SUM(ibs.AMOUNT) - SUM(ib.UNIT_COST * ib.QUANTITY))
              / NULLIF(SUM(ibs.AMOUNT), 0)               AS gross_margin_pct
    FROM INVENTORY_BATCH_SALES ibs
    JOIN INVENTORY_BATCHES     ib  ON ib.SALE_ID = ibs.ID
    WHERE ibs.CREATED_AT >= '{start}'
      AND ibs.CREATED_AT <  '{end}'
    GROUP BY 1
    ORDER BY 1
""")

# Convenience map so the data layer can iterate over them.
ALL_QUERIES = {
    "daily_revenue":           DAILY_REVENUE,
    "revenue_by_service_line": REVENUE_BY_SERVICE_LINE,
    "revenue_by_branch":       REVENUE_BY_BRANCH,
    "payment_mode_mix":        PAYMENT_MODE_MIX,
    "payer_performance":       PAYER_PERFORMANCE,
    "patient_rfm":             PATIENT_RFM,
    "top_items":               TOP_ITEMS,
    "hourly_heatmap":          HOURLY_HEATMAP,
    "cohort_retention":        COHORT_RETENTION,
    "doctor_productivity":     DOCTOR_PRODUCTIVITY,
    "leakage":                 LEAKAGE,
    "inventory_margin":        INVENTORY_MARGIN,
    "claim_rejection":         CLAIM_REJECTION,
    "revenue_concentration":   REVENUE_CONCENTRATION,
    "arpv_trend":              ARPV_TREND,
    "revenue_at_risk":         REVENUE_AT_RISK,
    "gross_profit_weekly":     GROSS_PROFIT_WEEKLY,
}


def render(query_name: str, *, start: str, end: str, clinic_filter: str = "") -> str:
    """Return a fully-formatted SQL string ready to execute on Snowflake."""
    template = ALL_QUERIES[query_name]
    return template.format(start=start, end=end, clinic_filter=clinic_filter)