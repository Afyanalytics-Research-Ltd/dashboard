from ai_foundation.contracts import InvestigationCard, InvestigationStep, MetricDefinition, Trigger

# Weights live in config, not engine code — auditable without reading Python
SEVERITY_WEIGHTS: dict[str, int] = {
    "Critical": 3,
    "Warning":  2,
    "Info":     1,
}

IMPACT_WEIGHTS: dict[str, int] = {
    "patient_flow":     3,
    "clinical_safety":  3,
    "capacity":         2,
    "efficiency":       2,
    "data_quality":     1,
}

_VALUE_QUERY = """
WITH latest AS (
    SELECT MAX(period_date) AS d FROM HOSPITALS.REPORTING.mart_operational_kpis
)
SELECT
    ROUND(AVG(m.cons_p50_mins), 0)  AS value,
    SUM(m.cons_n)                   AS n,
    MAX(m.period_date)              AS freshness_date
FROM HOSPITALS.REPORTING.mart_operational_kpis m
WHERE m.period_date >= (SELECT DATEADD('day', -28, d) FROM latest)
  AND m.period_date <= (SELECT d FROM latest)
"""

_BASELINE_QUERY = """
WITH latest AS (
    SELECT MAX(period_date) AS d FROM HOSPITALS.REPORTING.mart_operational_kpis
)
SELECT
    ROUND(AVG(m.cons_p50_mins), 0) AS baseline
FROM HOSPITALS.REPORTING.mart_operational_kpis m
WHERE m.period_date >= (SELECT DATEADD('day', -56, d) FROM latest)
  AND m.period_date <  (SELECT DATEADD('day', -28, d) FROM latest)
"""

CONSULT_P50 = MetricDefinition(
    metric_id="consult_p50",
    value_query=_VALUE_QUERY,
    baseline_query=_BASELINE_QUERY,
    freshness_requirement_hours=1440,  # 60 days — covers data horizon lag; proper fix needs mart_refresh_ts column
    minimum_sample=100,
)

# VALIDATION_THRESHOLD_UNCALIBRATED — not a hospital operating threshold
CONSULT_P50_TRIGGER = Trigger(
    metric_id="consult_p50",
    left_ref="change",
    operator="gt",
    threshold_type="absolute",
    threshold_val=0.10,
)

_DEPT_ATTRIBUTION_QUERY = """
WITH latest AS (
    SELECT MAX(pj.visit_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    WHERE pj.source_system = 'EMR_V2'
)
SELECT
    pj.dept,
    COUNT(*)                                                        AS visits,
    COUNT_IF(pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
        AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480) AS valid_n,
    ROUND(MEDIAN(
        CASE WHEN pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
             AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
        THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
    ), 0)                                                           AS p50_mins
FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
WHERE pj.source_system = 'EMR_V2'
  AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
  AND pj.dept IS NOT NULL
GROUP BY 1
HAVING valid_n >= 10
ORDER BY p50_mins DESC NULLS LAST
"""

_TEMPORAL_QUERY = """
WITH latest AS (
    SELECT MAX(pj.visit_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    WHERE pj.source_system = 'EMR_V2'
)
SELECT
    DAYOFWEEK(pj.arrival_ts)                                        AS dow_num,
    DAYNAME(pj.arrival_ts)                                          AS day_name,
    HOUR(pj.arrival_ts)                                             AS hour_of_day,
    COUNT(*)                                                        AS visit_count,
    ROUND(MEDIAN(
        CASE WHEN pj.cons_ts IS NOT NULL
              AND pj.cons_ts >= pj.arrival_ts
              AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
        THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
    ), 0)                                                           AS median_wait_mins
FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
WHERE pj.source_system = 'EMR_V2'
  AND pj.dept = %s
  AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
  AND pj.arrival_ts IS NOT NULL
  AND HOUR(pj.arrival_ts) BETWEEN 6 AND 22
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 3
ORDER BY 1, 3
"""

_VOLUME_QUERY = """
WITH latest AS (
    SELECT MAX(visit_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
    WHERE source_system = 'EMR_V2'
),
hourly AS (
    SELECT
        HOUR(arrival_ts)                                            AS hour_of_day,
        COUNT(*)                                                    AS total_arrivals,
        COUNT(DISTINCT visit_date)                                  AS days_observed,
        ROUND(COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT visit_date), 0), 1) AS avg_daily_arrivals
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
    WHERE source_system = 'EMR_V2'
      AND dept = %s
      AND visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND arrival_ts IS NOT NULL
      AND HOUR(arrival_ts) BETWEEN 6 AND 22
    GROUP BY 1
)
SELECT
    hour_of_day,
    total_arrivals,
    days_observed,
    avg_daily_arrivals,
    ROUND(AVG(avg_daily_arrivals) OVER (), 1)                      AS overall_avg_hourly_arrivals,
    ROUND(avg_daily_arrivals / NULLIF(AVG(avg_daily_arrivals) OVER (), 0), 2) AS volume_ratio
FROM hourly
ORDER BY 1
"""

_DOWNSTREAM_INCOMPLETE_CARE_QUERY = """
WITH latest AS (
    SELECT MAX(pj.visit_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    WHERE pj.source_system = 'EMR_V2'
),
cohort_care AS (
    SELECT
        COUNT(*)                                                            AS cohort_visits,
        COUNT_IF(m.incomplete_care = 1)                                    AS cohort_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(*), 0), 1)                                      AS cohort_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON pj.visit_id = m.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.dept = %s
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
),
baseline_care AS (
    SELECT
        COUNT(*)                                                            AS baseline_visits,
        COUNT_IF(m.incomplete_care = 1)                                    AS baseline_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(*), 0), 1)                                      AS baseline_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON pj.visit_id = m.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.dept IS NOT NULL
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
)
SELECT
    c.cohort_visits,
    c.cohort_incomplete_n,
    c.cohort_incomplete_pct,
    b.baseline_visits,
    b.baseline_incomplete_n,
    b.baseline_incomplete_pct
FROM cohort_care c
CROSS JOIN baseline_care b
"""

_DOWNSTREAM_PHARMACY_QUERY = """
WITH latest AS (
    SELECT MAX(pj.visit_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    WHERE pj.source_system = 'EMR_V2'
),
cohort_pharm AS (
    SELECT
        ROUND(MEDIAN(ph.tat_mins), 0) AS cohort_p50_mins,
        COUNT(DISTINCT pj.visit_id)   AS cohort_visit_n,
        COUNT(ph.visit_id)            AS cohort_item_n
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    INNER JOIN HOSPITALS.REPORTING.rpt_ortho_pharmacy ph
        ON ph.visit_id = pj.visit_id AND ph.source_system = 'EMR_V2'
    WHERE pj.source_system = 'EMR_V2'
      AND pj.dept = %s
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND ph.tat_mins IS NOT NULL
      AND ph.tat_mins BETWEEN 1 AND 1440
),
baseline_pharm AS (
    SELECT
        ROUND(MEDIAN(ph.tat_mins), 0) AS baseline_p50_mins,
        COUNT(DISTINCT pj.visit_id)   AS baseline_visit_n
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    INNER JOIN HOSPITALS.REPORTING.rpt_ortho_pharmacy ph
        ON ph.visit_id = pj.visit_id AND ph.source_system = 'EMR_V2'
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND pj.visit_type <> 'Inpatient'
      AND ph.tat_mins IS NOT NULL
      AND ph.tat_mins BETWEEN 1 AND 1440
)
SELECT
    c.cohort_p50_mins,
    c.cohort_visit_n,
    c.cohort_item_n,
    b.baseline_p50_mins,
    b.baseline_visit_n
FROM cohort_pharm c
CROSS JOIN baseline_pharm b
"""

# Peak window established by Step 3 temporal findings
_PEAK_WINDOW_HOURS = (9, 10, 11)

# Thresholds
_VOLUME_SPIKE_RATIO = 1.20               # peak window must be 20%+ above average to qualify as volume-driven
_DOWNSTREAM_ELEVATION_RATIO = 1.20      # cohort pharmacy P50 must be 20%+ above baseline to count as elevated
_DOWNSTREAM_INCOMPLETE_ELEVATION = 1.20 # cohort incomplete_pct must be 20%+ above OPD baseline to count as elevated

CONSULT_P50_CARD = InvestigationCard(
    id="consult_p50_card",
    trigger_metric_id="consult_p50",
    severity="Warning",
    impact_domain="patient_flow",
    sample_label="consult records",
    steps=[
        InvestigationStep(
            step_id="quantify",
            purpose="Confirm elevation magnitude and sample size from MetricState",
            query=None,
        ),
        InvestigationStep(
            step_id="dept_attribution",
            purpose="Identify which department drives the excess TAT",
            query=_DEPT_ATTRIBUTION_QUERY,
        ),
        InvestigationStep(
            step_id="temporal_pattern",
            purpose="Identify when the attributed department's excess is concentrated (DOW × hour)",
            query=_TEMPORAL_QUERY,
            uses_cohort=True,
        ),
        InvestigationStep(
            step_id="mechanism_test",
            purpose="Test volume, capacity, and scheduling as candidate mechanisms for the temporal concentration",
            query=_VOLUME_QUERY,
            uses_cohort=True,
        ),
        InvestigationStep(
            step_id="downstream_pharmacy",
            purpose="Determine whether the attributed cohort shows elevated pharmacy TAT downstream",
            query=_DOWNSTREAM_PHARMACY_QUERY,
            uses_cohort=True,
        ),
        InvestigationStep(
            step_id="downstream_incomplete_care",
            purpose="Determine whether the attributed cohort shows elevated incomplete care rate vs OPD baseline",
            query=_DOWNSTREAM_INCOMPLETE_CARE_QUERY,
            uses_cohort=True,
        ),
    ],
)

# ── Pharmacy P50 ─────────────────────────────────────────────────────────────

_PHARM_VALUE_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
),
w AS (
    SELECT tat_mins, DATE(request_stamp) AS req_date
    FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
    WHERE source_system = 'EMR_V2'
      AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND DATE(request_stamp) <= (SELECT d FROM latest)
      AND tat_mins IS NOT NULL
      AND tat_mins BETWEEN 1 AND 1440
)
SELECT
    ROUND(MEDIAN(tat_mins), 0)  AS value,
    COUNT(*)                    AS n,
    MAX(req_date)               AS freshness_date
FROM w
"""

_PHARM_BASELINE_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
)
SELECT ROUND(MEDIAN(tat_mins), 0) AS baseline
FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
WHERE source_system = 'EMR_V2'
  AND DATE(request_stamp) >= (SELECT DATEADD('day', -56, d) FROM latest)
  AND DATE(request_stamp) <  (SELECT DATEADD('day', -28, d) FROM latest)
  AND tat_mins IS NOT NULL
  AND tat_mins BETWEEN 1 AND 1440
"""

PHARMACY_P50 = MetricDefinition(
    metric_id="pharmacy_p50",
    value_query=_PHARM_VALUE_QUERY,
    baseline_query=_PHARM_BASELINE_QUERY,
    freshness_requirement_hours=1440,
    minimum_sample=50,
)

PHARMACY_P50_TRIGGER = Trigger(
    metric_id="pharmacy_p50",
    left_ref="change",
    operator="gt",
    threshold_type="absolute",
    threshold_val=0.10,
)

# Single-queue attribution — returns one row so _attribution_result finds a top row above baseline
_PHARM_ATTRIBUTION_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
)
SELECT
    'PHARMACY DISPENSING'   AS dept,
    COUNT(*)                AS visits,
    COUNT_IF(tat_mins IS NOT NULL AND tat_mins BETWEEN 1 AND 1440)     AS valid_n,
    ROUND(MEDIAN(CASE WHEN tat_mins IS NOT NULL AND tat_mins BETWEEN 1 AND 1440
                      THEN tat_mins END), 0)                           AS p50_mins
FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
WHERE source_system = 'EMR_V2'
  AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
"""

# DOW × hour distribution from queue-entry timestamp (request_stamp)
_PHARM_TEMPORAL_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
)
SELECT
    DAYOFWEEK(request_stamp)                                            AS dow_num,
    DAYNAME(request_stamp)                                              AS day_name,
    HOUR(request_stamp)                                                 AS hour_of_day,
    COUNT(*)                                                            AS visit_count,
    ROUND(MEDIAN(CASE WHEN tat_mins BETWEEN 1 AND 1440
                      THEN tat_mins END), 0)                           AS median_wait_mins
FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
WHERE source_system = 'EMR_V2'
  AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
  AND request_stamp IS NOT NULL
  AND HOUR(request_stamp) BETWEEN 6 AND 22
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 3
ORDER BY 1, 3
"""

# Hourly prescription demand — same column contract as consult mechanism query
_PHARM_MECHANISM_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
),
hourly AS (
    SELECT
        HOUR(request_stamp)                                              AS hour_of_day,
        COUNT(*)                                                         AS total_arrivals,
        COUNT(DISTINCT DATE(request_stamp))                              AS days_observed,
        ROUND(COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT DATE(request_stamp)), 0), 1) AS avg_daily_arrivals
    FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
    WHERE source_system = 'EMR_V2'
      AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND request_stamp IS NOT NULL
      AND tat_mins IS NOT NULL AND tat_mins BETWEEN 1 AND 1440
      AND HOUR(request_stamp) BETWEEN 6 AND 22
    GROUP BY 1
)
SELECT
    hour_of_day,
    total_arrivals,
    days_observed,
    avg_daily_arrivals,
    ROUND(AVG(avg_daily_arrivals) OVER (), 1)                           AS overall_avg_hourly_arrivals,
    ROUND(avg_daily_arrivals / NULLIF(AVG(avg_daily_arrivals) OVER (), 0), 2) AS volume_ratio
FROM hourly
ORDER BY 1
"""

# Downstream: pharmacy patients' incomplete care rate vs OPD baseline
# Cohort = all OPD visits with pharmacy records in the 28-day window (uses_cohort=False)
_PHARM_DOWNSTREAM_INCOMPLETE_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(pj.visit_date)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    WHERE pj.source_system = 'EMR_V2'
),
cohort_care AS (
    SELECT
        COUNT(DISTINCT pj.visit_id)                                     AS cohort_visits,
        COUNT_IF(m.incomplete_care = 1)                                 AS cohort_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(DISTINCT pj.visit_id), 0), 1)               AS cohort_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    INNER JOIN HOSPITALS.REPORTING.rpt_ortho_pharmacy ph
        ON ph.visit_id = pj.visit_id AND ph.source_system = 'EMR_V2'
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON m.visit_id = pj.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND ph.tat_mins IS NOT NULL
),
baseline_care AS (
    SELECT
        COUNT(DISTINCT pj.visit_id)                                     AS baseline_visits,
        COUNT_IF(m.incomplete_care = 1)                                 AS baseline_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(DISTINCT pj.visit_id), 0), 1)               AS baseline_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON m.visit_id = pj.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.dept IS NOT NULL
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
)
SELECT
    c.cohort_visits,
    c.cohort_incomplete_n,
    c.cohort_incomplete_pct,
    b.baseline_visits,
    b.baseline_incomplete_n,
    b.baseline_incomplete_pct
FROM cohort_care c
CROSS JOIN baseline_care b
"""

# Peak window: broader than consult (post-consultation prescription rush expected 9–13h)
_PHARM_PEAK_WINDOW_HOURS = (9, 10, 11, 12, 13)

PHARMACY_P50_CARD = InvestigationCard(
    id="pharmacy_p50_card",
    trigger_metric_id="pharmacy_p50",
    severity="Warning",
    impact_domain="efficiency",
    sample_label="pharmacy dispensing records",
    steps=[
        InvestigationStep(
            step_id="quantify",
            purpose="Confirm dispensing TAT elevation magnitude and sample size from MetricState",
            query=None,
        ),
        InvestigationStep(
            step_id="dept_attribution",
            purpose="Establish pharmacy dispensing queue as single-queue attribution anchor",
            query=_PHARM_ATTRIBUTION_QUERY,
        ),
        InvestigationStep(
            step_id="temporal_pattern",
            purpose="Identify when dispensing TAT is concentrated by day-of-week and hour of queue entry",
            query=_PHARM_TEMPORAL_QUERY,
            uses_cohort=False,
            meta={"temporal_anchor": "MAX(request_stamp) from rpt_ortho_pharmacy"},
        ),
        InvestigationStep(
            step_id="mechanism_test",
            purpose="Test volume as candidate mechanism for temporal concentration in pharmacy queue",
            query=_PHARM_MECHANISM_QUERY,
            uses_cohort=False,
            meta={
                "peak_window_hours": _PHARM_PEAK_WINDOW_HOURS,
                "capacity_not_applicable_reason": (
                    "Pharmacy SHIFTS shift-end times unreliable (Inv 108 — median 23h); "
                    "per-hour staffing availability cannot be determined from shift-initiation data alone"
                ),
                "scheduling_not_applicable_reason": (
                    "No appointment or session table for pharmacy dispensing — "
                    "demand is walk-in prescription flow with no pre-scheduled slots"
                ),
            },
        ),
        InvestigationStep(
            step_id="downstream_incomplete_care",
            purpose="Determine whether pharmacy patients show elevated incomplete care rate vs OPD baseline",
            query=_PHARM_DOWNSTREAM_INCOMPLETE_QUERY,
            uses_cohort=False,
        ),
    ],
)

# ── Lab Collect P50 ──────────────────────────────────────────────────────────

_LAB_VALUE_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_lab
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
),
w AS (
    SELECT
        DATEDIFF('minute', request_stamp, collection_stamp) AS collect_mins,
        DATE(request_stamp) AS req_date
    FROM HOSPITALS.REPORTING.rpt_ortho_lab
    WHERE source_system = 'EMR_V2'
      AND collection_stamp IS NOT NULL
      AND collection_stamp > request_stamp
      AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND DATE(request_stamp) <= (SELECT d FROM latest)
      AND DATEDIFF('minute', request_stamp, collection_stamp) BETWEEN 1 AND 480
)
SELECT
    ROUND(MEDIAN(collect_mins), 0) AS value,
    COUNT(*)                       AS n,
    MAX(req_date)                  AS freshness_date
FROM w
"""

_LAB_BASELINE_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_lab
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
)
SELECT ROUND(MEDIAN(DATEDIFF('minute', request_stamp, collection_stamp)), 0) AS baseline
FROM HOSPITALS.REPORTING.rpt_ortho_lab
WHERE source_system = 'EMR_V2'
  AND collection_stamp IS NOT NULL
  AND collection_stamp > request_stamp
  AND DATE(request_stamp) >= (SELECT DATEADD('day', -56, d) FROM latest)
  AND DATE(request_stamp) <  (SELECT DATEADD('day', -28, d) FROM latest)
  AND DATEDIFF('minute', request_stamp, collection_stamp) BETWEEN 1 AND 480
"""

LAB_COLLECT_P50 = MetricDefinition(
    metric_id="lab_collect_p50",
    value_query=_LAB_VALUE_QUERY,
    baseline_query=_LAB_BASELINE_QUERY,
    freshness_requirement_hours=1440,
    minimum_sample=50,
)

LAB_COLLECT_P50_TRIGGER = Trigger(
    metric_id="lab_collect_p50",
    left_ref="change",
    operator="gt",
    threshold_type="absolute",
    threshold_val=0.10,
)

# Test-type attribution — test_name aliased as dept so _attribution_result handler works unchanged
_LAB_ATTRIBUTION_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_lab
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
)
SELECT
    UPPER(TRIM(test_name))                                          AS dept,
    COUNT(*)                                                        AS visits,
    COUNT_IF(DATEDIFF('minute', request_stamp, collection_stamp) BETWEEN 1 AND 480)
                                                                    AS valid_n,
    ROUND(MEDIAN(
        CASE WHEN DATEDIFF('minute', request_stamp, collection_stamp) BETWEEN 1 AND 480
             THEN DATEDIFF('minute', request_stamp, collection_stamp) END
    ), 0)                                                           AS p50_mins
FROM HOSPITALS.REPORTING.rpt_ortho_lab
WHERE source_system = 'EMR_V2'
  AND collection_stamp IS NOT NULL
  AND collection_stamp > request_stamp
  AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
  AND test_name IS NOT NULL
GROUP BY 1
HAVING valid_n >= 20
ORDER BY p50_mins DESC NULLS LAST
"""

# DOW × hour distribution of order → collection wait, scoped to attributed test type
_LAB_TEMPORAL_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_lab
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
)
SELECT
    DAYOFWEEK(request_stamp)                                        AS dow_num,
    DAYNAME(request_stamp)                                          AS day_name,
    HOUR(request_stamp)                                             AS hour_of_day,
    COUNT(*)                                                        AS visit_count,
    ROUND(MEDIAN(
        CASE WHEN DATEDIFF('minute', request_stamp, collection_stamp) BETWEEN 1 AND 480
             THEN DATEDIFF('minute', request_stamp, collection_stamp) END
    ), 0)                                                           AS median_wait_mins
FROM HOSPITALS.REPORTING.rpt_ortho_lab
WHERE source_system = 'EMR_V2'
  AND collection_stamp IS NOT NULL
  AND collection_stamp > request_stamp
  AND UPPER(TRIM(test_name)) = %s
  AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
  AND request_stamp IS NOT NULL
  AND HOUR(request_stamp) BETWEEN 6 AND 22
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 3
ORDER BY 1, 3
"""

# Hourly order volume for attributed test type — same column contract as consult mechanism query
_LAB_MECHANISM_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(request_stamp)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_lab
    WHERE source_system = 'EMR_V2' AND request_stamp IS NOT NULL
),
hourly AS (
    SELECT
        HOUR(request_stamp)                                              AS hour_of_day,
        COUNT(*)                                                         AS total_arrivals,
        COUNT(DISTINCT DATE(request_stamp))                              AS days_observed,
        ROUND(COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT DATE(request_stamp)), 0), 1) AS avg_daily_arrivals
    FROM HOSPITALS.REPORTING.rpt_ortho_lab
    WHERE source_system = 'EMR_V2'
      AND collection_stamp IS NOT NULL
      AND collection_stamp > request_stamp
      AND UPPER(TRIM(test_name)) = %s
      AND DATE(request_stamp) >= (SELECT DATEADD('day', -28, d) FROM latest)
      AND request_stamp IS NOT NULL
      AND HOUR(request_stamp) BETWEEN 6 AND 22
    GROUP BY 1
)
SELECT
    hour_of_day,
    total_arrivals,
    days_observed,
    avg_daily_arrivals,
    ROUND(AVG(avg_daily_arrivals) OVER (), 1)                           AS overall_avg_hourly_arrivals,
    ROUND(avg_daily_arrivals / NULLIF(AVG(avg_daily_arrivals) OVER (), 0), 2) AS volume_ratio
FROM hourly
ORDER BY 1
"""

# Downstream: visits with ANY V2 lab order vs OPD baseline
# Uses had_lab_order flag on patient_journey — no visit_id join to rpt_ortho_lab needed
_LAB_DOWNSTREAM_INCOMPLETE_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(pj.visit_date)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    WHERE pj.source_system = 'EMR_V2'
),
cohort_care AS (
    SELECT
        COUNT(DISTINCT pj.visit_id)                                     AS cohort_visits,
        COUNT_IF(m.incomplete_care = 1)                                 AS cohort_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(DISTINCT pj.visit_id), 0), 1)               AS cohort_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON m.visit_id = pj.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.had_lab_order = 1
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
),
baseline_care AS (
    SELECT
        COUNT(DISTINCT pj.visit_id)                                     AS baseline_visits,
        COUNT_IF(m.incomplete_care = 1)                                 AS baseline_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(DISTINCT pj.visit_id), 0), 1)               AS baseline_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON m.visit_id = pj.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.dept IS NOT NULL
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
)
SELECT
    c.cohort_visits,
    c.cohort_incomplete_n,
    c.cohort_incomplete_pct,
    b.baseline_visits,
    b.baseline_incomplete_n,
    b.baseline_incomplete_pct
FROM cohort_care c
CROSS JOIN baseline_care b
"""

# Morning lab ordering window — post-consultation prescription peak; tune after first run
_LAB_PEAK_WINDOW_HOURS = (8, 9, 10, 11, 12)

LAB_COLLECT_P50_CARD = InvestigationCard(
    id="lab_collect_p50_card",
    trigger_metric_id="lab_collect_p50",
    severity="Warning",
    impact_domain="patient_flow",
    sample_label="lab orders with collection timestamps",
    scope_note=(
        "Stage 1 of 2 — order to specimen collection (phlebotomy queue wait only). "
        "Stage 2 — specimen collection to lab bench result — is a separate process and workforce; not covered by this card."
    ),
    steps=[
        InvestigationStep(
            step_id="quantify",
            purpose="Confirm order-to-collection TAT elevation magnitude and sample size",
            query=None,
        ),
        InvestigationStep(
            step_id="dept_attribution",
            purpose="Identify which test type has the highest order-to-collection wait P50",
            query=_LAB_ATTRIBUTION_QUERY,
        ),
        InvestigationStep(
            step_id="temporal_pattern",
            purpose="Identify when the attributed test type's collection wait is concentrated (DOW × hour of order)",
            query=_LAB_TEMPORAL_QUERY,
            uses_cohort=True,
            meta={"temporal_anchor": "MAX(request_stamp) from rpt_ortho_lab"},
        ),
        InvestigationStep(
            step_id="mechanism_test",
            purpose="Test volume as candidate mechanism for temporal concentration in phlebotomy queue",
            query=_LAB_MECHANISM_QUERY,
            uses_cohort=True,
            meta={
                "peak_window_hours": _LAB_PEAK_WINDOW_HOURS,
                "capacity_not_applicable_reason": (
                    "Lab phlebotomy staffing data not in schema — "
                    "SHIFTS covers OPD and Pharmacy only; lab collection is a separate workforce"
                ),
                "scheduling_not_applicable_reason": (
                    "Lab collection is physician-demand-driven from OPD consultation; "
                    "no appointment system exists for phlebotomy"
                ),
            },
        ),
        InvestigationStep(
            step_id="downstream_incomplete_care",
            purpose="Determine whether visits with lab orders show elevated incomplete care rate vs OPD baseline",
            query=_LAB_DOWNSTREAM_INCOMPLETE_QUERY,
            uses_cohort=False,
        ),
    ],
)

# ══════════════════════════════════════════════════════════════════════════════
# imaging_tat_p50 — order → radiology arrival, V2 only
# ══════════════════════════════════════════════════════════════════════════════

_IMG_VALUE_QUERY = """
WITH latest AS (
    SELECT MAX(request_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_imaging
    WHERE source_system = 'EMR_V2' AND request_date IS NOT NULL
)
SELECT
    ROUND(MEDIAN(result_tat_mins), 0)  AS value,
    COUNT(*)                           AS n,
    MAX(request_date)                  AS freshness_date
FROM HOSPITALS.REPORTING.rpt_ortho_imaging
WHERE source_system = 'EMR_V2'
  AND result_tat_mins IS NOT NULL
  AND result_tat_mins BETWEEN 1 AND 1440
  AND request_date >= (SELECT DATEADD('day', -28, d) FROM latest)
  AND request_date <= (SELECT d FROM latest)
"""

_IMG_BASELINE_QUERY = """
WITH latest AS (
    SELECT MAX(request_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_imaging
    WHERE source_system = 'EMR_V2' AND request_date IS NOT NULL
)
SELECT ROUND(MEDIAN(result_tat_mins), 0) AS baseline
FROM HOSPITALS.REPORTING.rpt_ortho_imaging
WHERE source_system = 'EMR_V2'
  AND result_tat_mins IS NOT NULL
  AND result_tat_mins BETWEEN 1 AND 1440
  AND request_date >= (SELECT DATEADD('day', -56, d) FROM latest)
  AND request_date <  (SELECT DATEADD('day', -28, d) FROM latest)
"""

IMAGING_TAT_P50 = MetricDefinition(
    metric_id="imaging_tat_p50",
    value_query=_IMG_VALUE_QUERY,
    baseline_query=_IMG_BASELINE_QUERY,
    freshness_requirement_hours=1440,
    minimum_sample=50,
)

IMAGING_TAT_P50_TRIGGER = Trigger(
    metric_id="imaging_tat_p50",
    left_ref="change",
    operator="gt",
    threshold_type="absolute",
    threshold_val=0.10,
)

# Modality attribution — top modality by P50 wait, 28-day window, V2 only
# valid_n >= 20 matches diagnostics page stability threshold; prevents low-volume noise attribution
_IMG_ATTRIBUTION_QUERY = """
WITH latest AS (
    SELECT MAX(request_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_imaging
    WHERE source_system = 'EMR_V2' AND request_date IS NOT NULL
)
SELECT
    UPPER(TRIM(COALESCE(modality_group, 'OTHER')))   AS dept,
    COUNT(*)                                          AS visits,
    COUNT_IF(result_tat_mins BETWEEN 1 AND 1440)     AS valid_n,
    ROUND(MEDIAN(
        CASE WHEN result_tat_mins BETWEEN 1 AND 1440
             THEN result_tat_mins END
    ), 0)                                             AS p50_mins
FROM HOSPITALS.REPORTING.rpt_ortho_imaging
WHERE source_system = 'EMR_V2'
  AND modality_group IS NOT NULL
  AND request_date >= (SELECT DATEADD('day', -28, d) FROM latest)
GROUP BY 1
HAVING valid_n >= 20
ORDER BY p50_mins DESC NULLS LAST
"""

# DOW × hour of request_stamp, scoped to attributed modality
_IMG_TEMPORAL_QUERY = """
WITH latest AS (
    SELECT MAX(request_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_imaging
    WHERE source_system = 'EMR_V2' AND request_date IS NOT NULL
)
SELECT
    DAYOFWEEK(request_stamp)                          AS dow_num,
    DAYNAME(request_stamp)                            AS day_name,
    HOUR(request_stamp)                               AS hour_of_day,
    COUNT(*)                                          AS visit_count,
    ROUND(MEDIAN(
        CASE WHEN result_tat_mins BETWEEN 1 AND 1440
             THEN result_tat_mins END
    ), 0)                                             AS median_wait_mins
FROM HOSPITALS.REPORTING.rpt_ortho_imaging
WHERE source_system = 'EMR_V2'
  AND UPPER(TRIM(COALESCE(modality_group, 'OTHER'))) = %s
  AND request_stamp IS NOT NULL
  AND HOUR(request_stamp) BETWEEN 6 AND 22
  AND request_date >= (SELECT DATEADD('day', -28, d) FROM latest)
GROUP BY 1, 2, 3
HAVING COUNT(*) >= 3
ORDER BY 1, 3
"""

# Hourly request volume for attributed modality — same column contract as other mechanism queries
_IMG_MECHANISM_QUERY = """
WITH latest AS (
    SELECT MAX(request_date) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_imaging
    WHERE source_system = 'EMR_V2' AND request_date IS NOT NULL
),
hourly AS (
    SELECT
        HOUR(request_stamp)                                                          AS hour_of_day,
        COUNT(*)                                                                     AS total_arrivals,
        COUNT(DISTINCT request_date)                                                 AS days_observed,
        ROUND(COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT request_date), 0), 1)          AS avg_daily_arrivals
    FROM HOSPITALS.REPORTING.rpt_ortho_imaging
    WHERE source_system = 'EMR_V2'
      AND UPPER(TRIM(COALESCE(modality_group, 'OTHER'))) = %s
      AND request_stamp IS NOT NULL
      AND HOUR(request_stamp) BETWEEN 6 AND 22
      AND request_date >= (SELECT DATEADD('day', -28, d) FROM latest)
    GROUP BY 1
)
SELECT
    hour_of_day,
    total_arrivals,
    days_observed,
    avg_daily_arrivals,
    ROUND(AVG(avg_daily_arrivals) OVER (), 1)                                        AS overall_avg_hourly_arrivals,
    ROUND(avg_daily_arrivals / NULLIF(AVG(avg_daily_arrivals) OVER (), 0), 2)        AS volume_ratio
FROM hourly
ORDER BY 1
"""

# Downstream: visits with any imaging order vs OPD baseline
# Uses had_imaging flag on patient_journey — no join to rpt_ortho_imaging needed
_IMG_DOWNSTREAM_INCOMPLETE_QUERY = """
WITH latest AS (
    SELECT MAX(DATE(pj.visit_date)) AS d
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    WHERE pj.source_system = 'EMR_V2'
),
cohort_care AS (
    SELECT
        COUNT(DISTINCT pj.visit_id)                                     AS cohort_visits,
        COUNT_IF(m.incomplete_care = 1)                                 AS cohort_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(DISTINCT pj.visit_id), 0), 1)               AS cohort_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON m.visit_id = pj.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.had_imaging = TRUE
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
),
baseline_care AS (
    SELECT
        COUNT(DISTINCT pj.visit_id)                                     AS baseline_visits,
        COUNT_IF(m.incomplete_care = 1)                                 AS baseline_incomplete_n,
        ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
            / NULLIF(COUNT(DISTINCT pj.visit_id), 0), 1)               AS baseline_incomplete_pct
    FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
    JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON m.visit_id = pj.visit_id
    WHERE pj.source_system = 'EMR_V2'
      AND pj.visit_type <> 'Inpatient'
      AND pj.arrival_ts IS NOT NULL
      AND pj.dept IS NOT NULL
      AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
)
SELECT
    c.cohort_visits,
    c.cohort_incomplete_n,
    c.cohort_incomplete_pct,
    b.baseline_visits,
    b.baseline_incomplete_n,
    b.baseline_incomplete_pct
FROM cohort_care c
CROSS JOIN baseline_care b
"""

# Mid-morning imaging window — post-consultation ordering peak
_IMG_PEAK_WINDOW_HOURS = (9, 10, 11, 12, 13)

IMAGING_TAT_P50_CARD = InvestigationCard(
    id="imaging_tat_p50_card",
    trigger_metric_id="imaging_tat_p50",
    severity="Warning",
    impact_domain="patient_flow",
    sample_label="imaging orders",
    scope_note=(
        "Measures order to radiology arrival (V2 definition); "
        "radiology reporting stage (arrival to result) is a separate process not covered by this card. "
        "Identifies the currently worsening modality by 28-day change — not the absolute slowest modality all-time. "
        "A different modality may show a higher absolute TAT on the diagnostics dashboard; "
        "this card fires when any modality deteriorates relative to its own recent baseline."
    ),
    steps=[
        InvestigationStep(
            step_id="quantify",
            purpose="Confirm order-to-radiology-arrival TAT elevation magnitude and sample size",
            query=None,
        ),
        InvestigationStep(
            step_id="dept_attribution",
            purpose="Identify which imaging modality has the highest order-to-arrival wait P50 in the last 28 days",
            query=_IMG_ATTRIBUTION_QUERY,
        ),
        InvestigationStep(
            step_id="temporal_pattern",
            purpose="Identify when the attributed modality's wait is concentrated (DOW × hour of request)",
            query=_IMG_TEMPORAL_QUERY,
            uses_cohort=True,
            meta={"temporal_anchor": "MAX(request_date) from rpt_ortho_imaging"},
        ),
        InvestigationStep(
            step_id="mechanism_test",
            purpose="Test volume as candidate mechanism for temporal concentration in imaging queue",
            query=_IMG_MECHANISM_QUERY,
            uses_cohort=True,
            meta={
                "peak_window_hours": _IMG_PEAK_WINDOW_HOURS,
                "capacity_not_applicable_reason": (
                    "Radiology staffing data not in schema — "
                    "SHIFTS covers OPD and Pharmacy only; radiology workforce is a separate department"
                ),
                "scheduling_not_applicable_reason": (
                    "Imaging booking and equipment availability data not in schema — "
                    "no radiology scheduling or equipment utilisation table available"
                ),
            },
        ),
        InvestigationStep(
            step_id="downstream_incomplete_care",
            purpose="Determine whether visits with imaging orders show elevated incomplete care rate vs OPD baseline",
            query=_IMG_DOWNSTREAM_INCOMPLETE_QUERY,
            uses_cohort=False,
        ),
    ],
)

CARD_REGISTRY: dict[str, InvestigationCard] = {
    "consult_p50": CONSULT_P50_CARD,
    "pharmacy_p50": PHARMACY_P50_CARD,
    "lab_collect_p50": LAB_COLLECT_P50_CARD,
    "imaging_tat_p50": IMAGING_TAT_P50_CARD,
}

# Expose constants for engine
MECHANISM_PEAK_WINDOW_HOURS = _PEAK_WINDOW_HOURS
MECHANISM_VOLUME_SPIKE_RATIO = _VOLUME_SPIKE_RATIO
DOWNSTREAM_PHARMACY_ELEVATION_RATIO = _DOWNSTREAM_ELEVATION_RATIO
DOWNSTREAM_INCOMPLETE_ELEVATION_RATIO = _DOWNSTREAM_INCOMPLETE_ELEVATION
