from .db import run_query_df

_V1    = "source_system = 'EMR_V1'"
_V2    = "source_system = 'EMR_V2'"
_V2_OPD = "source_system = 'EMR_V2' AND visit_type <> 'Inpatient'"


# ── OPD ──────────────────────────────────────────────────────────────────────

def q_opd_monthly():
    """Monthly visit counts — all systems. Vertical cutoff line at 2025-02-01 in chart."""
    return run_query_df("""
        SELECT
            visit_month,
            COUNT(*) AS visits
        FROM HOSPITALS.REPORTING.rpt_ortho_opd
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_summary():
    """Top-level OPD KPIs — all systems. Gender normalized to M/F in stg_visits."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                    AS total_visits,
            COUNT_IF(gender = 'F')                                      AS female_visits,
            COUNT_IF(gender = 'M')                                      AS male_visits,
            ROUND(COUNT_IF(gender = 'F') * 100.0
                  / NULLIF(COUNT(gender), 0), 1)                       AS female_pct,
            ROUND(COUNT_IF(gender = 'M') * 100.0
                  / NULLIF(COUNT(gender), 0), 1)                       AS male_pct,
            MIN(visit_date)                                             AS data_from,
            MAX(visit_date)                                             AS data_to
        FROM HOSPITALS.REPORTING.rpt_ortho_opd
    """)


def q_opd_dow():
    """Visit count by day of week — all systems."""
    return run_query_df("""
        SELECT
            visit_dow,
            visit_day_name,
            COUNT(*) AS visits
        FROM HOSPITALS.REPORTING.rpt_ortho_opd
        GROUP BY 1, 2
        ORDER BY 1
    """)


def q_opd_gender():
    """Gender breakdown — all systems. Normalized to M/F."""
    return run_query_df("""
        SELECT
            COALESCE(gender, 'Unknown')                                 AS gender,
            COUNT(*)                                                    AS visits,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)         AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_opd
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_opd_hour():
    """Visit arrival hour — V2 only, for current operational staffing patterns."""
    return run_query_df("""
        SELECT
            arrival_hour,
            COUNT(*) AS visits
        FROM HOSPITALS.REPORTING.rpt_ortho_opd
        WHERE source_system = 'EMR_V2'
          AND arrival_hour IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)


def q_opd_dow_v2():
    """Visit count by day of week — V2 only. Replaces q_opd_dow() in Peak Ops section."""
    return run_query_df("""
        SELECT
            visit_dow,
            visit_day_name,
            COUNT(*) AS visits
        FROM HOSPITALS.REPORTING.rpt_ortho_opd
        WHERE source_system = 'EMR_V2'
        GROUP BY 1, 2
        ORDER BY 1
    """)


def q_peak_stage_tat():
    """Stage TAT medians by hour band — V2 OPD, stable baseline Mar–Dec 2025.
    Three stages: pre-triage (arrival→triage), triage→consult, consult→pharmacy.
    Returns one row per hour band with median and valid-n for each stage.
    BASELINE (Mar–Dec 2025) — date range hardcoded; re-run if baseline changes."""
    return run_query_df("""
        WITH base AS (
            SELECT
                CASE
                    WHEN HOUR(arrival_ts) BETWEEN 7  AND  9 THEN '1 Early (07-10)'
                    WHEN HOUR(arrival_ts) BETWEEN 10 AND 12 THEN '2 Peak (10-13)'
                    WHEN HOUR(arrival_ts) BETWEEN 13 AND 16 THEN '3 After-peak (13-17)'
                END                                                         AS hour_band,
                CASE WHEN triage_ts IS NOT NULL
                     AND DATEDIFF('minute', arrival_ts, triage_ts) BETWEEN 1 AND 240
                THEN DATEDIFF('minute', arrival_ts, triage_ts) END          AS pretriage_mins,
                CASE WHEN cons_ts IS NOT NULL AND triage_ts IS NOT NULL
                     AND DATEDIFF('minute', triage_ts, cons_ts) BETWEEN 1 AND 480
                THEN DATEDIFF('minute', triage_ts, cons_ts) END             AS triage_cons_mins,
                CASE WHEN pharm_ts IS NOT NULL AND cons_ts IS NOT NULL
                     AND DATEDIFF('minute', cons_ts, pharm_ts) BETWEEN 1 AND 240
                THEN DATEDIFF('minute', cons_ts, pharm_ts) END              AS cons_pharm_mins
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
              AND visit_type   <> 'Inpatient'
              AND arrival_ts   IS NOT NULL
              AND visit_date   BETWEEN '2025-03-01' AND '2025-12-31'
        )
        SELECT
            hour_band,
            COUNT_IF(pretriage_mins    IS NOT NULL) AS pretriage_n,
            ROUND(MEDIAN(pretriage_mins),    0)     AS pretriage_median_mins,
            COUNT_IF(triage_cons_mins  IS NOT NULL) AS triage_cons_n,
            ROUND(MEDIAN(triage_cons_mins),  0)     AS triage_cons_median_mins,
            COUNT_IF(cons_pharm_mins   IS NOT NULL) AS cons_pharm_n,
            ROUND(MEDIAN(cons_pharm_mins),   0)     AS cons_pharm_median_mins
        FROM base
        WHERE hour_band IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)


def q_peak_lwbs_by_band():
    """No-care rate by hour band — V2 OPD, stable baseline Mar–Dec 2025.
    Joins spine (arrival_ts for hour band) + mart (incomplete_care signal).
    BASELINE (Mar–Dec 2025) — date range hardcoded; re-run if baseline changes."""
    return run_query_df("""
        WITH base AS (
            SELECT
                CASE
                    WHEN HOUR(pj.arrival_ts) BETWEEN 7  AND  9 THEN '1 Early (07-10)'
                    WHEN HOUR(pj.arrival_ts) BETWEEN 10 AND 12 THEN '2 Peak (10-13)'
                    WHEN HOUR(pj.arrival_ts) BETWEEN 13 AND 16 THEN '3 After-peak (13-17)'
                END                                                         AS hour_band,
                m.incomplete_care
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey     pj
            JOIN HOSPITALS.REPORTING.mart_pathway_analysis          m
                ON pj.visit_id = m.visit_id
            WHERE pj.source_system = 'EMR_V2'
              AND pj.visit_type   <> 'Inpatient'
              AND pj.arrival_ts   IS NOT NULL
              AND pj.visit_date   BETWEEN '2025-03-01' AND '2025-12-31'
        )
        SELECT
            hour_band,
            CASE hour_band
                WHEN '1 Early (07-10)'      THEN 1
                WHEN '2 Peak (10-13)'       THEN 2
                WHEN '3 After-peak (13-17)' THEN 3
            END                                                             AS band_sort,
            COUNT(*)                                                        AS total_n,
            COUNT_IF(incomplete_care = 1)                                   AS incomplete_n,
            ROUND(100.0 * COUNT_IF(incomplete_care = 1)
                / NULLIF(COUNT(*), 0), 1)                                   AS incomplete_pct
        FROM base
        WHERE hour_band IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 2
    """)


def q_peak_volume_corr():
    """Weekly peak-hour volume vs triage→consult TAT and no-care rate — V2 OPD, Mar–Dec 2025.
    Grain: one row per week. Scatter plots use flat cloud to show no volume→delay relationship.
    BASELINE (Mar–Dec 2025, peak hours 10–12) — hardcoded; re-run if baseline changes."""
    return run_query_df("""
        WITH weekly AS (
            SELECT
                DATE_TRUNC('week', pj.visit_date)                       AS week,
                COUNT(DISTINCT pj.visit_id)                             AS weekly_visits,
                ROUND(MEDIAN(
                    CASE WHEN pj.cons_ts   IS NOT NULL
                              AND pj.triage_ts IS NOT NULL
                              AND DATEDIFF('minute', pj.triage_ts, pj.cons_ts) BETWEEN 1 AND 480
                         THEN DATEDIFF('minute', pj.triage_ts, pj.cons_ts) END
                ), 0)                                                   AS median_triage_cons_mins,
                ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
                      / NULLIF(COUNT(*), 0), 1)                         AS nocare_rate_pct
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey  pj
            JOIN HOSPITALS.REPORTING.mart_pathway_analysis       m
                ON pj.visit_id = m.visit_id
            WHERE pj.source_system      = 'EMR_V2'
              AND pj.visit_type        <> 'Inpatient'
              AND pj.arrival_ts        IS NOT NULL
              AND HOUR(pj.arrival_ts) BETWEEN 10 AND 12
              AND pj.visit_date        BETWEEN '2025-03-01' AND '2025-12-31'
            GROUP BY 1
        )
        SELECT week, weekly_visits, median_triage_cons_mins, nocare_rate_pct
        FROM weekly
        WHERE weekly_visits >= 5
        ORDER BY 1
    """)


def q_peak_lwbs_by_stage():
    """LWBS count by hour band × drop_off_stage — V2 OPD, Mar–Dec 2025.
    Restricted to incomplete_care=1 patients and post-registration/post-triage stages.
    Shows WHERE in the pathway patients exit, segmented by time of day.
    BASELINE (Mar–Dec 2025) — date range hardcoded; re-run if baseline changes."""
    return run_query_df("""
        WITH base AS (
            SELECT
                CASE
                    WHEN HOUR(pj.arrival_ts) BETWEEN 7  AND  9 THEN '1 Early (07-10)'
                    WHEN HOUR(pj.arrival_ts) BETWEEN 10 AND 12 THEN '2 Peak (10-13)'
                    WHEN HOUR(pj.arrival_ts) BETWEEN 13 AND 16 THEN '3 After-peak (13-17)'
                END                                                         AS hour_band,
                m.drop_off_stage
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey  pj
            JOIN HOSPITALS.REPORTING.mart_pathway_analysis       m
                ON pj.visit_id = m.visit_id
            WHERE pj.source_system  = 'EMR_V2'
              AND pj.visit_type    <> 'Inpatient'
              AND pj.arrival_ts    IS NOT NULL
              AND pj.visit_date    BETWEEN '2025-03-01' AND '2025-12-31'
              AND m.incomplete_care = 1
              AND m.drop_off_stage  IN ('post-registration', 'post-triage')
        )
        SELECT
            hour_band,
            CASE hour_band
                WHEN '1 Early (07-10)'      THEN 1
                WHEN '2 Peak (10-13)'       THEN 2
                WHEN '3 After-peak (13-17)' THEN 3
            END                                                             AS band_sort,
            drop_off_stage,
            COUNT(*)                                                        AS n
        FROM base
        WHERE hour_band IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 2, 3
    """)


def q_opd_peak_conversion():
    """Conversion rate by OPD visit DOW — V1 only.
    Numerator: OPD visits that preceded a V1 admission within 7 days (via prior_opd_date).
    Denominator: all V1 OPD visits on that DOW."""
    return run_query_df(f"""
        WITH opd_by_dow AS (
            SELECT
                visit_dow,
                visit_day_name,
                COUNT(*) AS total_opd_visits
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            WHERE {_V1}
            GROUP BY 1, 2
        ),
        converted_by_dow AS (
            SELECT
                DAYOFWEEK(prior_opd_date)               AS visit_dow,
                COUNT(DISTINCT prior_opd_visit_id)      AS converted_visits
            FROM HOSPITALS.REPORTING.rpt_ortho_conversion
            WHERE {_V1}
              AND has_prior_opd_7d     = TRUE
              AND prior_opd_date       IS NOT NULL
            GROUP BY 1
        )
        SELECT
            o.visit_dow,
            o.visit_day_name,
            o.total_opd_visits,
            COALESCE(c.converted_visits, 0)             AS converted_visits,
            ROUND(COALESCE(c.converted_visits, 0) * 100.0
                  / NULLIF(o.total_opd_visits, 0), 2)  AS conversion_pct
        FROM opd_by_dow o
        LEFT JOIN converted_by_dow c ON o.visit_dow = c.visit_dow
        ORDER BY o.visit_dow
    """)


def q_opd_peak_funnel():
    """Peak patient return funnel — V1 only. Inv 122/123 (2026-07-23).
    Peak window: Weekday 07:00-08:59 (data-derived — highest-pressure slot).
    Returns a single aggregate row: non-admitted count, never_returned, returned,
    admitted_of_returned, never_returned_pct, admitted_of_returned_pct, median_days_to_return."""
    return run_query_df(f"""
        WITH peak_visits AS (
            SELECT visit_id, composite_patient_id, visit_date
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            WHERE {_V1}
              AND visit_dow   IN (1, 2, 3, 4, 5)
              AND hour_of_day IN (7, 8)
        ),
        admitted_from_peak AS (
            SELECT DISTINCT prior_opd_visit_id AS visit_id
            FROM HOSPITALS.REPORTING.rpt_ortho_conversion
            WHERE {_V1}
              AND has_prior_opd_7d    = TRUE
              AND prior_opd_visit_id IS NOT NULL
        ),
        peak_non_admitted AS (
            SELECT p.*
            FROM peak_visits p
            LEFT JOIN admitted_from_peak a ON p.visit_id = a.visit_id
            WHERE a.visit_id IS NULL
        ),
        next_opd AS (
            SELECT
                pna.visit_id        AS peak_visit_id,
                pna.composite_patient_id,
                pna.visit_date      AS peak_date,
                MIN(opd.visit_date) AS next_opd_date
            FROM peak_non_admitted pna
            INNER JOIN HOSPITALS.REPORTING.rpt_ortho_opd opd
                ON  opd.composite_patient_id = pna.composite_patient_id
                AND opd.visit_date            > pna.visit_date
                AND opd.source_system         = 'EMR_V1'
            GROUP BY 1, 2, 3
        ),
        returned_then_admitted AS (
            SELECT DISTINCT no.peak_visit_id
            FROM next_opd no
            INNER JOIN HOSPITALS.REPORTING.rpt_ortho_conversion conv
                ON  conv.composite_patient_id = no.composite_patient_id
                AND conv.admission_date       >= no.next_opd_date
                AND conv.admission_date       <= DATEADD('day', 7, no.next_opd_date)
                AND conv.has_prior_opd_7d      = TRUE
                AND conv.source_system         = 'EMR_V1'
        )
        SELECT
            COUNT(DISTINCT pna.visit_id)                                AS peak_non_admitted,
            COUNT(DISTINCT no.peak_visit_id)                            AS returned,
            COUNT(DISTINCT pna.visit_id)
                - COUNT(DISTINCT no.peak_visit_id)                      AS never_returned,
            COUNT(DISTINCT rta.peak_visit_id)                           AS returned_then_admitted,
            ROUND(
                (COUNT(DISTINCT pna.visit_id)
                    - COUNT(DISTINCT no.peak_visit_id)) * 100.0
                / NULLIF(COUNT(DISTINCT pna.visit_id), 0), 1
            )                                                           AS never_returned_pct,
            ROUND(
                COUNT(DISTINCT rta.peak_visit_id) * 100.0
                / NULLIF(COUNT(DISTINCT no.peak_visit_id), 0), 1
            )                                                           AS admitted_of_returned_pct,
            ROUND(MEDIAN(
                DATEDIFF('day', no.peak_date, no.next_opd_date)
            ), 0)                                                       AS median_days_to_return
        FROM peak_non_admitted pna
        LEFT JOIN next_opd no                ON pna.visit_id = no.peak_visit_id
        LEFT JOIN returned_then_admitted rta ON pna.visit_id = rta.peak_visit_id
    """)


def q_opd_peak_funnel_by_dow():
    """Deferred admissions broken down by DOW within the peak window — V1 only.
    Confirms which weekday within Mon-Fri 07:00-08:59 has the highest deferred admission rate."""
    return run_query_df(f"""
        WITH peak_visits AS (
            SELECT visit_id, composite_patient_id, visit_date, visit_dow, visit_day_name
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            WHERE {_V1}
              AND visit_dow   IN (1, 2, 3, 4, 5)
              AND hour_of_day IN (7, 8)
        ),
        admitted_from_peak AS (
            SELECT DISTINCT prior_opd_visit_id AS visit_id
            FROM HOSPITALS.REPORTING.rpt_ortho_conversion
            WHERE {_V1}
              AND has_prior_opd_7d    = TRUE
              AND prior_opd_visit_id IS NOT NULL
        ),
        peak_non_admitted AS (
            SELECT p.*
            FROM peak_visits p
            LEFT JOIN admitted_from_peak a ON p.visit_id = a.visit_id
            WHERE a.visit_id IS NULL
        ),
        next_opd AS (
            SELECT
                pna.visit_id        AS peak_visit_id,
                pna.composite_patient_id,
                pna.visit_date      AS peak_date,
                pna.visit_dow,
                pna.visit_day_name,
                MIN(opd.visit_date) AS next_opd_date
            FROM peak_non_admitted pna
            INNER JOIN HOSPITALS.REPORTING.rpt_ortho_opd opd
                ON  opd.composite_patient_id = pna.composite_patient_id
                AND opd.visit_date            > pna.visit_date
                AND opd.source_system         = 'EMR_V1'
            GROUP BY 1, 2, 3, 4, 5
        ),
        returned_then_admitted AS (
            SELECT
                no.peak_visit_id,
                no.visit_dow,
                no.visit_day_name,
                DATEDIFF('day', no.peak_date, no.next_opd_date) AS days_to_return
            FROM next_opd no
            INNER JOIN HOSPITALS.REPORTING.rpt_ortho_conversion conv
                ON  conv.composite_patient_id = no.composite_patient_id
                AND conv.admission_date       >= no.next_opd_date
                AND conv.admission_date       <= DATEADD('day', 7, no.next_opd_date)
                AND conv.has_prior_opd_7d      = TRUE
                AND conv.source_system         = 'EMR_V1'
        )
        SELECT
            pna.visit_dow,
            pna.visit_day_name,
            COUNT(DISTINCT pna.visit_id)                            AS peak_non_admitted,
            COUNT(DISTINCT rta.peak_visit_id)                       AS deferred_admissions,
            ROUND(COUNT(DISTINCT rta.peak_visit_id) * 100.0
                  / NULLIF(COUNT(DISTINCT pna.visit_id), 0), 1)    AS deferred_pct,
            ROUND(MEDIAN(rta.days_to_return), 0)                   AS median_days_to_return
        FROM peak_non_admitted pna
        LEFT JOIN returned_then_admitted rta ON pna.visit_id = rta.peak_visit_id
        GROUP BY 1, 2
        ORDER BY 1
    """)


def q_opd_tat_by_dow():
    """Median non-zero TAT by day of week — all systems. NULL rows excluded (batch-close artefact)."""
    return run_query_df("""
        SELECT
            visit_dow,
            visit_day_name,
            COUNT(*)                                                    AS visits_with_tat,
            ROUND(MEDIAN(total_visit_mins), 0)                         AS median_tat_mins,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY total_visit_mins), 0)                      AS p75_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_tat
        WHERE total_visit_mins IS NOT NULL
          AND total_visit_mins  > 0
        GROUP BY 1, 2
        ORDER BY 1
    """)


def q_opd_deferred_monthly():
    """Tuesday peak deferred admissions monthly trend — V1 only. Inv 123 extension (2026-07-23).
    90-day return-visit cap eliminates spurious long-lag matches from pre-2022 data.
    Filtered to 2022-06+ where operational data is reliable."""
    return run_query_df(f"""
        WITH peak_visits AS (
            SELECT visit_id, composite_patient_id, visit_date
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            WHERE {_V1}
              AND visit_dow   = 2
              AND hour_of_day IN (7, 8)
        ),
        admitted_from_peak AS (
            SELECT DISTINCT prior_opd_visit_id AS visit_id
            FROM HOSPITALS.REPORTING.rpt_ortho_conversion
            WHERE {_V1}
              AND has_prior_opd_7d    = TRUE
              AND prior_opd_visit_id IS NOT NULL
        ),
        peak_non_admitted AS (
            SELECT p.*
            FROM peak_visits p
            LEFT JOIN admitted_from_peak a ON p.visit_id = a.visit_id
            WHERE a.visit_id IS NULL
        ),
        next_opd AS (
            SELECT
                pna.visit_id          AS peak_visit_id,
                pna.composite_patient_id,
                pna.visit_date        AS peak_date,
                MIN(opd.visit_date)   AS next_opd_date
            FROM peak_non_admitted pna
            INNER JOIN HOSPITALS.REPORTING.rpt_ortho_opd opd
                ON  opd.composite_patient_id = pna.composite_patient_id
                AND opd.visit_date            > pna.visit_date
                AND opd.source_system         = 'EMR_V1'
                AND DATEDIFF('day', pna.visit_date, opd.visit_date) <= 90
            GROUP BY 1, 2, 3
        ),
        returned_then_admitted AS (
            SELECT DISTINCT no.peak_visit_id
            FROM next_opd no
            INNER JOIN HOSPITALS.REPORTING.rpt_ortho_conversion conv
                ON  conv.composite_patient_id = no.composite_patient_id
                AND conv.admission_date       >= no.next_opd_date
                AND conv.admission_date       <= DATEADD('day', 7, no.next_opd_date)
                AND conv.has_prior_opd_7d      = TRUE
                AND conv.source_system         = 'EMR_V1'
        )
        SELECT
            DATE_TRUNC('month', pna.visit_date)                              AS visit_month,
            COUNT(DISTINCT pna.visit_id)                                     AS tue_peak_non_admitted,
            COUNT(DISTINCT rta.peak_visit_id)                                AS deferred_admissions,
            ROUND(COUNT(DISTINCT rta.peak_visit_id) * 100.0
                  / NULLIF(COUNT(DISTINCT pna.visit_id), 0), 1)              AS deferred_pct,
            ROUND(MEDIAN(DATEDIFF('day', no.peak_date, no.next_opd_date)), 0) AS median_days_to_return
        FROM peak_non_admitted pna
        LEFT JOIN next_opd no                ON pna.visit_id = no.peak_visit_id
        LEFT JOIN returned_then_admitted rta ON pna.visit_id = rta.peak_visit_id
        WHERE DATE_TRUNC('month', pna.visit_date) >= '2022-06-01'
        GROUP BY 1
        ORDER BY 1
    """)


# ── TAT ──────────────────────────────────────────────────────────────────────

def q_tat_summary():
    """TAT KPI summary — all systems. Threshold metrics (under 60 min, over 4 hr), P50/P90 non-zero."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                    AS total_visits,
            COUNT_IF(total_visit_mins IS NOT NULL)                      AS with_recorded_tat,
            COUNT_IF(total_visit_mins = 0)                              AS zero_tat_count,
            COUNT_IF(total_visit_mins > 0)                              AS nonzero_count,
            ROUND(COUNT_IF(total_visit_mins = 0) * 100.0
                  / NULLIF(COUNT_IF(total_visit_mins IS NOT NULL), 0), 1) AS zero_pct,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY CASE WHEN total_visit_mins > 0
                                 THEN total_visit_mins END), 0)        AS p50_nonzero_mins,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                  (ORDER BY CASE WHEN total_visit_mins > 0
                                 THEN total_visit_mins END), 0)        AS p90_nonzero_mins,
            ROUND(COUNT_IF(total_visit_mins > 0 AND total_visit_mins < 60) * 100.0
                  / NULLIF(COUNT_IF(total_visit_mins > 0), 0), 1)      AS under_60_pct,
            ROUND(COUNT_IF(total_visit_mins >= 240) * 100.0
                  / NULLIF(COUNT_IF(total_visit_mins > 0), 0), 1)      AS over_4hr_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_tat
    """)


def q_tat_distribution():
    """TAT bucket distribution — all systems. Shows batch-close artefact."""
    return run_query_df("""
        SELECT
            CASE
                WHEN total_visit_mins IS NULL          THEN 'NULL'
                WHEN total_visit_mins = 0              THEN '0 min'
                WHEN total_visit_mins BETWEEN 1 AND 4  THEN '1–4 min'
                WHEN total_visit_mins BETWEEN 5 AND 30 THEN '5–30 min'
                WHEN total_visit_mins BETWEEN 31 AND 120 THEN '31–120 min'
                WHEN total_visit_mins BETWEEN 121 AND 480 THEN '2–8 hr'
                WHEN total_visit_mins > 480            THEN '>8 hr'
            END                                         AS tat_bucket,
            COUNT(*)                                    AS visits,
            ROUND(COUNT(*) * 100.0
                  / SUM(COUNT(*)) OVER (), 1)           AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_tat
        GROUP BY 1
        ORDER BY
            CASE tat_bucket
                WHEN 'NULL'      THEN 0 WHEN '0 min'     THEN 1
                WHEN '1–4 min'   THEN 2 WHEN '5–30 min'  THEN 3
                WHEN '31–120 min' THEN 4 WHEN '2–8 hr'   THEN 5
                WHEN '>8 hr'     THEN 6
            END
    """)


def q_tat_nonzero_monthly():
    """Monthly median TAT — all systems, zero and NULL excluded (artefact rows removed).
    V1: total_visit_mins = close−open. V2: total_visit_mins = max stage proxy."""
    return run_query_df("""
        SELECT
            visit_month,
            COUNT(*)                                                    AS visits_with_tat,
            MEDIAN(total_visit_mins)                                    AS median_tat_mins,
            PERCENTILE_CONT(0.75) WITHIN GROUP
                (ORDER BY total_visit_mins)                             AS p75_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_tat
        WHERE total_visit_mins > 0
        GROUP BY 1
        ORDER BY 1
    """)


# ── Theatre ───────────────────────────────────────────────────────────────────

def q_theatre_monthly():
    """Monthly procedure count — V1+V2. Grain: procedure_month + source_system."""
    return run_query_df("""
        SELECT
            source_system,
            procedure_month,
            COUNT(*)                        AS procedures,
            COUNT(DISTINCT visit_id)        AS distinct_visits,
            COUNT_IF(from_opd = TRUE)       AS from_opd,
            COUNT_IF(is_completed = TRUE)   AS completed
        FROM HOSPITALS.REPORTING.rpt_ortho_theatre
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)


def q_theatre_v1_procedure_hours():
    """V1 procedure × total theatre hours — reads from reporting view (Inv 142/148).
    procedure_name now sourced via ORTHO_THEATRE → stg_procedures → rpt_ortho_theatre.
    Excludes durations >720 min (timing errors). Returns top 20 by total hours."""
    return run_query_df("""
        SELECT
            procedure_name,
            COUNT(*)                                                        AS case_count,
            ROUND(MEDIAN(surgery_duration_mins), 0)                         AS median_mins,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY surgery_duration_mins), 0)                     AS p90_mins,
            ROUND(SUM(surgery_duration_mins) / 60.0, 1)                    AS total_hours,
            ROUND(SUM(surgery_duration_mins) * 100.0
                  / NULLIF(SUM(SUM(surgery_duration_mins)) OVER (), 0), 1) AS pct_of_total_hours
        FROM HOSPITALS.REPORTING.vw_rpt_ortho_theatre
        WHERE source_system = 'EMR_V1'
          AND procedure_name IS NOT NULL
          AND surgery_duration_mins IS NOT NULL
          AND surgery_duration_mins <= 720
        GROUP BY 1
        ORDER BY total_hours DESC
        LIMIT 20
    """)


def q_theatre_summary():
    """Top-level theatre KPIs — V1+V2. Duration/NHIF naturally self-filter to V1 (NULL for V2)."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                        AS total_procedures,
            COUNT(DISTINCT visit_id)                                        AS distinct_visits,
            COUNT_IF(from_opd = TRUE)                                       AS from_opd_count,
            ROUND(COUNT_IF(from_opd = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                 AS from_opd_pct,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY
                CASE WHEN surgery_duration_mins <= 720
                     THEN surgery_duration_mins END), 0)                    AS median_duration_mins,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY
                CASE WHEN surgery_duration_mins <= 720
                     THEN surgery_duration_mins END), 0)                    AS p90_duration_mins,
            ROUND(COUNT_IF(mode_of_payment = 'NHIF') * 100.0
                  / NULLIF(COUNT_IF(mode_of_payment IS NOT NULL), 0), 1)    AS nhif_pct,
            MIN(procedure_date)                                             AS data_from,
            MAX(procedure_date)                                             AS data_to
        FROM HOSPITALS.REPORTING.rpt_ortho_theatre
    """)


def q_theatre_duration():
    """Surgery duration in buckets — V1 only (V2 starttime unavailable). Excludes >720 min errors."""
    return run_query_df("""
        SELECT
            CASE
                WHEN surgery_duration_mins < 60   THEN '< 60 min'
                WHEN surgery_duration_mins < 120  THEN '60–120 min'
                WHEN surgery_duration_mins < 180  THEN '120–180 min'
                WHEN surgery_duration_mins < 300  THEN '180–300 min'
                WHEN surgery_duration_mins < 480  THEN '300–480 min'
                ELSE '> 480 min'
            END                                   AS duration_bucket,
            MIN(surgery_duration_mins)            AS bucket_min,
            COUNT(*)                              AS cases
        FROM HOSPITALS.REPORTING.rpt_ortho_theatre
        WHERE source_system = 'EMR_V1'
          AND surgery_duration_mins IS NOT NULL
          AND surgery_duration_mins <= 720
        GROUP BY duration_bucket
        ORDER BY bucket_min
    """)


def q_theatre_anaesthesia():
    """Anaesthesia type distribution — V1 only (V2 anaesthesia_type = 0% coverage)."""
    return run_query_df("""
        SELECT
            COALESCE(anaesthesia_type, 'Unknown')           AS anaesthesia_type,
            COUNT(*)                                         AS cases,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_theatre
        WHERE source_system = 'EMR_V1'
        GROUP BY 1
        ORDER BY cases DESC
    """)


def q_theatre_procedures_v2():
    """Top V2 procedures by volume — procedure_name (96% coverage). V2-only.
    avg_sched_days / median_sched_days removed — request_to_planned_days is 97.7% zeros (Inv 144)."""
    return run_query_df("""
        SELECT
            procedure_name,
            COUNT(*)                                            AS procedures,
            COUNT_IF(is_elective = TRUE)                        AS elective,
            COUNT_IF(is_elective = FALSE)                       AS emergency
        FROM HOSPITALS.REPORTING.rpt_ortho_theatre
        WHERE source_system = 'EMR_V2'
          AND procedure_name IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 15
    """)


def q_theatre_elective():
    """V2 elective vs emergency split. V2-only (is_elective 96% coverage).
    median_sched_days / p90_sched_days removed — request_to_planned_days is 97.7% zeros (Inv 144)."""
    return run_query_df("""
        SELECT
            CASE
                WHEN is_elective = TRUE  THEN 'Elective'
                WHEN is_elective = FALSE THEN 'Emergency'
                ELSE 'Unknown'
            END                                                 AS case_type,
            COUNT(*)                                            AS procedures,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_theatre
        WHERE source_system = 'EMR_V2'
          AND is_elective IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_theatre_access_kpis():
    """V2 same-day theatre wait KPIs — request_to_service_mins (64.8% coverage, Inv 143).
    Elective-only (emergency avg ≈7 min, not a bottleneck). Capped at 480 min (Inv 150):
    values above are request timestamps logged at shift start, not actual booking times.
    Capped: P50=25 min · P90=339 min · 19.8% over 4hrs · 865 cases (140 artefacts excluded)."""
    return run_query_df(f"""
        SELECT
            COUNT(request_to_service_mins)                                  AS n_with_wait,
            ROUND(COUNT(request_to_service_mins) * 100.0
                  / NULLIF(COUNT_IF(is_elective = TRUE), 0), 1)            AS pct_coverage,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                  (ORDER BY request_to_service_mins), 0)                   AS p50_mins,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY request_to_service_mins), 0)                   AS p90_mins,
            COUNT_IF(request_to_service_mins > 240)                        AS over_4hrs,
            ROUND(COUNT_IF(request_to_service_mins > 240) * 100.0
                  / NULLIF(COUNT(request_to_service_mins), 0), 1)          AS pct_over_4hrs
        FROM HOSPITALS.REPORTING.vw_rpt_ortho_theatre
        WHERE source_system = 'EMR_V2'
          AND is_elective = TRUE
          AND request_to_service_mins IS NOT NULL
          AND request_to_service_mins <= 480
    """)


def q_theatre_overdue():
    """V2 overdue scheduled procedures — status=1 with past planned_date (Inv 145).
    Proxy non-completion: booked but not executed on planned date. Not confirmed cancellations."""
    return run_query_df("""
        SELECT COUNT(*) AS overdue_count
        FROM HOSPITALS.ORTHOPEDIC_CLEAN_V2.ORTHO_PROCEDURE
        WHERE is_sentinel = FALSE
          AND status = '1'
          AND planned_date < CURRENT_DATE()
    """)


def q_theatre_emergency_kpis():
    """V2 emergency theatre access — cleaned average wait + long-wait flag (Inv 143/149).
    Distribution is bimodal: 12 cases cluster at 1–28 min (typical fast access);
    5 cases at 338–670 min (distinct long-wait group); 3 zeros are recording artefacts.
    Cleaned average excludes zeros and waits >120 min — shown separately as n_long_wait.
    IQR method fails on this distribution (Q3=338, upper fence=842, excludes nothing)."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                         AS total_emergency,
            COUNT(request_to_service_mins)                                   AS n_with_wait,
            COUNT_IF(request_to_service_mins > 0
                     AND request_to_service_mins <= 120)                     AS n_clean,
            ROUND(AVG(CASE
                WHEN request_to_service_mins > 0
                 AND request_to_service_mins <= 120
                THEN request_to_service_mins END), 0)                        AS avg_clean_mins,
            COUNT_IF(request_to_service_mins > 120)                          AS n_long_wait,
            COUNT_IF(request_to_service_mins = 0)                            AS n_zero
        FROM HOSPITALS.REPORTING.vw_rpt_ortho_theatre
        WHERE source_system = 'EMR_V2'
          AND is_elective = FALSE
    """)


def q_theatre_emergency_monthly():
    """Monthly emergency procedure count V2 — trend tracking for ED TAT section.
    Volume trend is measurable even where full pathway TAT is constrained."""
    return run_query_df("""
        SELECT
            procedure_month,
            COUNT(*) AS emergency_cases
        FROM HOSPITALS.REPORTING.rpt_ortho_theatre
        WHERE source_system = 'EMR_V2'
          AND is_elective = FALSE
        GROUP BY 1
        ORDER BY 1
    """)


def q_theatre_access_distribution():
    """V2 elective same-day wait in time buckets — for distribution chart (Inv 143/150).
    Capped at 480 min (one session): '> 8 hrs' bucket is entirely artefactual (Inv 150).
    Excludes NULL wait (35.2% missing) and artefact values >480 min (13.9% of uncapped set)."""
    return run_query_df("""
        SELECT
            CASE
                WHEN request_to_service_mins <  60  THEN '< 1 hr'
                WHEN request_to_service_mins < 120  THEN '1–2 hrs'
                WHEN request_to_service_mins < 240  THEN '2–4 hrs'
                ELSE                                     '4–8 hrs'
            END                                                             AS wait_bucket,
            COUNT(*)                                                        AS cases
        FROM HOSPITALS.REPORTING.vw_rpt_ortho_theatre
        WHERE source_system = 'EMR_V2'
          AND is_elective = TRUE
          AND request_to_service_mins IS NOT NULL
          AND request_to_service_mins <= 480
        GROUP BY 1
        ORDER BY MIN(request_to_service_mins)
    """)


def q_theatre_v2_procedure_wait():
    """V2 procedures ranked by median same-day wait — Inv 147/150.
    Cap at 480 min (one 8-hr session): values above this reflect request timestamps
    logged at shift start rather than actual booking time, confirmed same-day
    (99.7% of cases have request_date = planned_date — Inv 150).
    REMOVAL OF PLATES SHORT BONES drops off after cap (only 1 of 5 cases clean)."""
    return run_query_df("""
        SELECT
            procedure_name,
            COUNT(request_to_service_mins)                                  AS n,
            ROUND(MEDIAN(request_to_service_mins), 0)                       AS median_wait_mins,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY request_to_service_mins), 0)                   AS p90_wait_mins,
            COUNT_IF(request_to_service_mins > 240)                        AS over_4hrs,
            ROUND(COUNT_IF(request_to_service_mins > 240) * 100.0
                  / NULLIF(COUNT(request_to_service_mins), 0), 1)          AS pct_over_4hrs
        FROM HOSPITALS.REPORTING.vw_rpt_ortho_theatre
        WHERE source_system = 'EMR_V2'
          AND is_elective = TRUE
          AND procedure_name IS NOT NULL
          AND request_to_service_mins IS NOT NULL
          AND request_to_service_mins <= 480
        GROUP BY 1
        HAVING COUNT(request_to_service_mins) >= 5
        ORDER BY median_wait_mins DESC
        LIMIT 15
    """)


# ── Admissions ────────────────────────────────────────────────────────────────

def q_admissions_monthly():
    """Monthly admissions + reliable LOS — V2 only. Grain: admission_month."""
    return run_query_df(f"""
        SELECT
            admission_month,
            COUNT(*)                                AS admissions,
            COUNT_IF(is_discharge_reliable = TRUE)  AS reliable_discharge,
            ROUND(AVG(CASE WHEN is_discharge_reliable = TRUE
                           THEN los_days END), 1)   AS avg_los_days
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
        GROUP BY 1
        ORDER BY 1
    """)


def q_admissions_summary():
    """Top-level admissions KPIs — V2 only."""
    return run_query_df(f"""
        SELECT
            COUNT(*)                                                AS total_admissions,
            COUNT_IF(is_discharge_reliable = TRUE)                 AS reliable_discharges,
            ROUND(COUNT_IF(is_discharge_reliable = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                        AS reliable_pct,
            ROUND(AVG(CASE WHEN is_los_plausible = TRUE
                           THEN los_days END), 1)                  AS avg_los_days,
            COUNT_IF(is_los_plausible = TRUE AND los_days > 14)    AS long_stay_count,
            MIN(admission_date)                                     AS data_from,
            MAX(admission_date)                                     AS data_to
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
    """)


def q_admissions_ward():
    """Ward admission volume + OPD routing split — V2 only (100% ward coverage)."""
    return run_query_df(f"""
        SELECT
            ward,
            COUNT(*)                                                    AS admissions,
            COUNT_IF(has_prior_opd_90d = TRUE)                         AS via_opd,
            COUNT_IF(has_prior_opd_90d = FALSE)                        AS direct,
            ROUND(COUNT_IF(has_prior_opd_90d = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                            AS opd_pct,
            ROUND(COUNT_IF(has_prior_opd_90d = FALSE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                            AS direct_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
          AND ward IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_admissions_los_by_ward():
    """Median LOS by ward — V1. Reliable + plausible discharges; wards with >= 20 cases only."""
    return run_query_df(f"""
        WITH ward_los AS (
            SELECT
                ward,
                COUNT(*)                                                AS reliable_count,
                ROUND(MEDIAN(los_days), 1)                              AS median_los,
                ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                      (ORDER BY los_days), 1)                           AS p90_los,
                COUNT_IF(los_days > 14)                                 AS long_stay_count,
                ROUND(COUNT_IF(los_days > 14) * 100.0
                      / NULLIF(COUNT(*), 0), 1)                        AS long_stay_pct
            FROM HOSPITALS.REPORTING.rpt_ortho_admissions
            WHERE {_V1}
              AND is_discharge_reliable = TRUE
              AND is_los_plausible      = TRUE
              AND ward                  IS NOT NULL
            GROUP BY 1
        )
        SELECT *
        FROM ward_los
        WHERE reliable_count >= 20
        ORDER BY median_los DESC
    """)


def q_admissions_gender():
    """Gender split — V2 only."""
    return run_query_df(f"""
        SELECT
            CASE
                WHEN UPPER(TRIM(gender)) IN ('M', 'MALE')   THEN 'M'
                WHEN UPPER(TRIM(gender)) IN ('F', 'FEMALE') THEN 'F'
                ELSE 'Unknown'
            END                                                     AS gender,
            COUNT(*)                                                AS admissions,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)     AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_admissions_dow():
    """Admissions by day of week — V2 only. Sorted Mon–Sun."""
    return run_query_df(f"""
        SELECT
            CASE WHEN admission_dow = 0 THEN 7 ELSE admission_dow END AS dow_sort,
            admission_day_name                                          AS day_name,
            COUNT(*)                                                    AS admissions
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
          AND admission_dow IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1
    """)


def q_admissions_los_dist():
    """LOS bucket distribution — V2 only, plausible discharges only."""
    return run_query_df(f"""
        SELECT
            CASE
                WHEN los_days BETWEEN 0  AND 7   THEN '0–7d'
                WHEN los_days BETWEEN 8  AND 14  THEN '8–14d'
                WHEN los_days BETWEEN 15 AND 30  THEN '15–30d'
                WHEN los_days BETWEEN 31 AND 90  THEN '31–90d'
                ELSE '> 90d'
            END                                                     AS los_bucket,
            CASE
                WHEN los_days BETWEEN 0  AND 7   THEN 1
                WHEN los_days BETWEEN 8  AND 14  THEN 2
                WHEN los_days BETWEEN 15 AND 30  THEN 3
                WHEN los_days BETWEEN 31 AND 90  THEN 4
                ELSE 5
            END                                                     AS bucket_order,
            COUNT(*)                                                AS patients,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)     AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
          AND is_los_plausible = TRUE
        GROUP BY 1, 2
        ORDER BY 2
    """)


def q_admissions_long_stay_trend():
    """Monthly long-stay rate (>14d) — V2 only, plausible discharges only."""
    return run_query_df(f"""
        SELECT
            admission_month,
            COUNT_IF(is_los_plausible = TRUE)                               AS plausible_count,
            COUNT_IF(is_los_plausible = TRUE AND los_days > 14)             AS long_stay_count,
            ROUND(
                COUNT_IF(is_los_plausible = TRUE AND los_days > 14) * 100.0
                / NULLIF(COUNT_IF(is_los_plausible = TRUE), 0), 1
            )                                                               AS long_stay_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
        GROUP BY 1
        ORDER BY 1
    """)


def q_admissions_routing_trend():
    """Monthly OPD-routed vs direct admission counts — V2 only."""
    return run_query_df(f"""
        SELECT
            admission_month,
            COUNT(*)                                                    AS admissions,
            COUNT_IF(has_prior_opd_90d = TRUE)                         AS via_opd,
            COUNT_IF(has_prior_opd_90d = FALSE)                        AS direct,
            ROUND(COUNT_IF(has_prior_opd_90d = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                            AS opd_pct,
            ROUND(COUNT_IF(has_prior_opd_90d = FALSE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                            AS direct_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
        GROUP BY 1
        ORDER BY 1
    """)


def q_admissions_discharge_hour():
    """Discharge time-of-day distribution — V2 only. Hour 0–23 with count + pct."""
    return run_query_df(f"""
        SELECT
            discharge_hour                                                  AS hour,
            COUNT(*)                                                        AS discharges,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)             AS pct
        FROM HOSPITALS.REPORTING.vw_rpt_ortho_admissions
        WHERE {_V2}
          AND discharge_hour IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)


def q_admissions_long_stay_by_routing():
    """Long-stay rate by OPD routing — V2 only, plausible discharges only."""
    return run_query_df(f"""
        SELECT
            CASE WHEN has_prior_opd_90d = TRUE THEN 'Via OPD' ELSE 'Direct' END AS routing,
            COUNT_IF(is_los_plausible = TRUE)                               AS total,
            COUNT_IF(is_los_plausible = TRUE AND los_days > 14)             AS long_stay,
            ROUND(
                COUNT_IF(is_los_plausible = TRUE AND los_days > 14) * 100.0
                / NULLIF(COUNT_IF(is_los_plausible = TRUE), 0), 1
            )                                                               AS long_stay_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
        GROUP BY 1
        ORDER BY long_stay_pct DESC
    """)


# ── Conversion ────────────────────────────────────────────────────────────────

def q_conversion_summary():
    """Conversion KPI summary — all systems. Admission rate, same-day %, median lag, direct %."""
    return run_query_df("""
        WITH opd_count AS (
            SELECT COUNT(*) AS opd_visits
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
        ),
        conv_stats AS (
            SELECT
                COUNT(*)                                                AS total_admissions,
                COUNT_IF(has_prior_opd_7d = TRUE)                      AS opd_converted,
                COUNT_IF(is_direct_admission = TRUE)                   AS direct_admissions,
                COUNT_IF(opd_to_admission_days = 0)                    AS same_day_admissions,
                ROUND(MEDIAN(
                    CASE WHEN has_prior_opd_7d = TRUE
                              AND opd_to_admission_days IS NOT NULL
                         THEN opd_to_admission_days END
                ), 1)                                                  AS median_days_to_admit
            FROM HOSPITALS.REPORTING.rpt_ortho_conversion
        )
        SELECT
            o.opd_visits,
            c.total_admissions,
            c.opd_converted,
            c.direct_admissions,
            c.same_day_admissions,
            c.median_days_to_admit,
            ROUND(c.total_admissions * 100.0
                  / NULLIF(o.opd_visits, 0), 2)                        AS admission_rate,
            ROUND(c.same_day_admissions * 100.0
                  / NULLIF(c.opd_converted, 0), 1)                    AS same_day_pct,
            ROUND(c.direct_admissions * 100.0
                  / NULLIF(c.total_admissions, 0), 1)                 AS direct_pct
        FROM opd_count o
        CROSS JOIN conv_stats c
    """)


def q_conversion_monthly():
    """Monthly admission rate — all systems. Rate = admissions / OPD visits per month."""
    return run_query_df("""
        WITH opd_monthly AS (
            SELECT visit_month AS month, COUNT(*) AS opd_visits
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            GROUP BY 1
        ),
        adm_monthly AS (
            SELECT
                admission_month AS month,
                COUNT(*) AS admissions,
                COUNT_IF(has_prior_opd_7d = TRUE) AS converted,
                COUNT_IF(is_direct_admission = TRUE) AS direct
            FROM HOSPITALS.REPORTING.rpt_ortho_conversion
            GROUP BY 1
        )
        SELECT
            a.month                                                     AS admission_month,
            a.admissions,
            a.converted,
            a.direct,
            o.opd_visits,
            ROUND(a.admissions * 100.0 / NULLIF(o.opd_visits, 0), 2)  AS admission_rate
        FROM adm_monthly a
        LEFT JOIN opd_monthly o ON a.month = o.month
        ORDER BY 1
    """)


def q_conversion_lag():
    """OPD-to-admission lag distribution (days) — all systems, OPD-converted only."""
    return run_query_df("""
        SELECT
            opd_to_admission_days,
            COUNT(*) AS admissions
        FROM HOSPITALS.REPORTING.rpt_ortho_conversion
        WHERE has_prior_opd_7d = TRUE
          AND opd_to_admission_days IS NOT NULL
          AND opd_to_admission_days BETWEEN 0 AND 7
        GROUP BY 1
        ORDER BY 1
    """)


def q_conv_v2_summary():
    """OPD->IPD conversion summary — V2 only. Total admissions, OPD-trigger rate, date range."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                        AS total_admissions,
            COUNT_IF(has_prior_opd_7d = TRUE)                               AS opd_triggered,
            COUNT_IF(is_direct_admission = TRUE)                            AS direct,
            ROUND(COUNT_IF(has_prior_opd_7d = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                 AS opd_trigger_pct,
            MIN(admission_date)                                             AS data_from,
            MAX(admission_date)                                             AS data_to
        FROM HOSPITALS.REPORTING.rpt_ortho_conversion
        WHERE source_system = 'EMR_V2'
    """)


def q_conv_v2_monthly():
    """Monthly V2 admissions + OPD visit volume + conversion rate (admissions / OPD visits)."""
    return run_query_df("""
        WITH admissions AS (
            SELECT
                admission_month,
                COUNT(*)                                                    AS admissions,
                COUNT_IF(has_prior_opd_7d = TRUE)                           AS opd_triggered,
                COUNT_IF(is_direct_admission = TRUE)                        AS direct,
                ROUND(COUNT_IF(has_prior_opd_7d = TRUE) * 100.0
                      / NULLIF(COUNT(*), 0), 1)                             AS opd_trigger_pct
            FROM HOSPITALS.REPORTING.rpt_ortho_conversion
            WHERE source_system = 'EMR_V2'
            GROUP BY 1
        ),
        opd_vol AS (
            SELECT
                DATE_TRUNC('month', visit_date)                             AS visit_month,
                COUNT(*)                                                    AS opd_visits
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            WHERE source_system = 'EMR_V2'
            GROUP BY 1
        )
        SELECT
            a.admission_month,
            a.admissions,
            a.opd_triggered,
            a.direct,
            a.opd_trigger_pct,
            o.opd_visits,
            ROUND(a.admissions * 100.0 / NULLIF(o.opd_visits, 0), 2)      AS conversion_rate
        FROM admissions a
        LEFT JOIN opd_vol o ON a.admission_month = o.visit_month
        ORDER BY 1
    """)


def q_conv_v2_ward():
    """V2 admissions by ward — ranked by volume, with share and OPD-trigger %."""
    return run_query_df("""
        SELECT
            COALESCE(ward, 'Unknown')                                       AS ward,
            COUNT(*)                                                        AS admissions,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)             AS pct,
            COUNT_IF(has_prior_opd_7d = TRUE)                               AS opd_triggered,
            ROUND(COUNT_IF(has_prior_opd_7d = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                 AS opd_trigger_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_conversion
        WHERE source_system = 'EMR_V2'
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_conv_v2_type():
    """V2 conversion_type breakdown — New Patient / Revisit / Direct / Walk-In."""
    return run_query_df("""
        SELECT
            COALESCE(conversion_type, 'Unknown')                            AS conversion_type,
            COUNT(*)                                                        AS n,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)             AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_conversion
        WHERE source_system = 'EMR_V2'
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_conv_v2_dow():
    """V2 admissions by day of week — volume + OPD-trigger %."""
    return run_query_df("""
        SELECT
            DAYOFWEEK(admission_date)                                       AS dow,
            DAYNAME(admission_date)                                         AS day_name,
            COUNT(*)                                                        AS admissions,
            ROUND(COUNT_IF(has_prior_opd_7d = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                 AS opd_trigger_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_conversion
        WHERE source_system = 'EMR_V2'
        GROUP BY 1, 2
        ORDER BY 1
    """)


# ── Lab ───────────────────────────────────────────────────────────────────────

def q_lab_monthly():
    """Monthly lab test volume + result rate — V1+V2.
    Groups by request_date (not visit_month) — visit_month NULL for V2 Feb 2026+ due to Issue 86."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', request_date)                   AS visit_month,
            COUNT(*)                                            AS tests,
            COUNT(DISTINCT visit_id)                           AS distinct_visits,
            COUNT_IF(has_result = TRUE)                        AS with_result,
            ROUND(COUNT_IF(has_result = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                    AS result_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_lab
        WHERE request_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)


def q_lab_top_tests():
    """Top 15 lab tests by volume — V1+V2. UPPER(TRIM()) collapses case variants (e.g. Urinalysis/URINALYSIS)."""
    return run_query_df("""
        SELECT
            UPPER(TRIM(test_name))                      AS test_name,
            COUNT(*)                                    AS tests,
            COUNT_IF(has_result = TRUE)                 AS with_result,
            ROUND(COUNT_IF(has_result = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)             AS result_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_lab
        WHERE test_name IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT 15
    """)


def q_lab_summary():
    """Lab KPI summary — V1+V2. Total orders, completion rate, P50/P90 TAT."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                AS total_orders,
            COUNT_IF(has_result = TRUE)                             AS resulted,
            COUNT_IF(has_result = FALSE)                            AS unresulted,
            ROUND(COUNT_IF(has_result = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                        AS completion_pct,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY result_tat_mins), 0)                   AS p50_tat_mins,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                  (ORDER BY result_tat_mins), 0)                   AS p90_tat_mins,
            MIN(request_date)                                      AS data_from,
            MAX(request_date)                                      AS data_to
        FROM HOSPITALS.REPORTING.rpt_ortho_lab
    """)


def q_lab_tat_dist():
    """Lab TAT bucket distribution — V1+V2, resulted orders only. Both systems: order → result recorded."""
    return run_query_df("""
        SELECT
            CASE
                WHEN result_tat_mins <=  30 THEN '≤30 min'
                WHEN result_tat_mins <=  60 THEN '31–60 min'
                WHEN result_tat_mins <= 120 THEN '1–2 hrs'
                WHEN result_tat_mins <= 240 THEN '2–4 hrs'
                WHEN result_tat_mins <= 480 THEN '4–8 hrs'
                ELSE '>8 hrs'
            END                                                     AS tat_bucket,
            CASE
                WHEN result_tat_mins <=  30 THEN 1
                WHEN result_tat_mins <=  60 THEN 2
                WHEN result_tat_mins <= 120 THEN 3
                WHEN result_tat_mins <= 240 THEN 4
                WHEN result_tat_mins <= 480 THEN 5
                ELSE 6
            END                                                     AS bucket_order,
            COUNT(*)                                                AS orders,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)     AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_lab
        WHERE result_tat_mins IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 2
    """)


def q_lab_bridge():
    """Bridge KPI: V1 patients with unresulted OPD lab who reached theatre within 30 days."""
    return run_query_df("""
        SELECT COUNT(DISTINCT l.composite_patient_id) AS patients_at_risk
        FROM HOSPITALS.REPORTING.rpt_ortho_lab l
        INNER JOIN HOSPITALS.STAGING.stg_procedures p
            ON l.composite_patient_id = p.composite_patient_id
           AND p.operation_date BETWEEN l.visit_date
               AND DATEADD('day', 30, l.visit_date)
        WHERE l.has_result = FALSE
          AND l.source_system = 'EMR_V1'
          AND p.source_system = 'EMR_V1'
          AND p.operation_date IS NOT NULL
    """)


# ── Imaging ───────────────────────────────────────────────────────────────────

def q_imaging_monthly():
    """Monthly imaging study volume by modality — V1+V2."""
    return run_query_df("""
        SELECT
            visit_month,
            modality_group,
            COUNT(*)                        AS studies,
            COUNT(DISTINCT visit_id)        AS distinct_visits
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
        WHERE modality_group IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
    """)


def q_imaging_modality_mix():
    """All-time modality breakdown — V1+V2."""
    return run_query_df("""
        SELECT
            modality_group,
            COUNT(*)                                    AS studies,
            ROUND(COUNT(*) * 100.0
                  / SUM(COUNT(*)) OVER (), 1)           AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
        WHERE modality_group IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_imaging_summary():
    """Imaging KPI summary — V1+V2. Total orders, completion rate, P50/P90 TAT.
    V1 TAT = order→physician review (seenon). V2 TAT = order→radiology arrival (recstamp)."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                AS total_orders,
            COUNT_IF(has_result = TRUE)                             AS resulted,
            COUNT_IF(has_result = FALSE)                            AS unresulted,
            ROUND(COUNT_IF(has_result = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                        AS completion_pct,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY result_tat_mins), 0)                   AS p50_tat_mins,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                  (ORDER BY result_tat_mins), 0)                   AS p90_tat_mins,
            MIN(request_date)                                      AS data_from,
            MAX(request_date)                                      AS data_to
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
    """)


def q_lab_mom():
    """Lab orders: last substantially complete month + MoM % change.
    Excludes months where max data day < 25 (avoids partial months from ingestion lag)."""
    return run_query_df("""
        WITH monthly AS (
            SELECT
                DATE_TRUNC('month', request_date)   AS order_month,
                COUNT(*)                            AS orders
            FROM HOSPITALS.REPORTING.rpt_ortho_lab
            WHERE request_date IS NOT NULL
            GROUP BY 1
            HAVING DAY(MAX(request_date)) >= 25
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY order_month DESC) AS rn
            FROM monthly
        )
        SELECT
            r1.order_month                                                          AS last_month,
            r1.orders                                                               AS last_month_orders,
            r2.orders                                                               AS prev_month_orders,
            ROUND((r1.orders - r2.orders) * 100.0 / NULLIF(r2.orders, 0), 1)       AS mom_pct
        FROM ranked r1
        LEFT JOIN ranked r2 ON r2.rn = r1.rn + 1
        WHERE r1.rn = 1
    """)


def q_imaging_mom():
    """Imaging orders: last substantially complete month + MoM % change.
    Excludes months where max data day < 25 (avoids partial months from ingestion lag)."""
    return run_query_df("""
        WITH monthly AS (
            SELECT
                DATE_TRUNC('month', request_date)   AS order_month,
                COUNT(*)                            AS orders
            FROM HOSPITALS.REPORTING.rpt_ortho_imaging
            WHERE request_date IS NOT NULL
            GROUP BY 1
            HAVING DAY(MAX(request_date)) >= 25
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (ORDER BY order_month DESC) AS rn
            FROM monthly
        )
        SELECT
            r1.order_month                                                          AS last_month,
            r1.orders                                                               AS last_month_orders,
            r2.orders                                                               AS prev_month_orders,
            ROUND((r1.orders - r2.orders) * 100.0 / NULLIF(r2.orders, 0), 1)       AS mom_pct
        FROM ranked r1
        LEFT JOIN ranked r2 ON r2.rn = r1.rn + 1
        WHERE r1.rn = 1
    """)


def q_imaging_modality_completion():
    """Completion rate + volume by modality — V1+V2."""
    return run_query_df("""
        SELECT
            COALESCE(modality_group, 'Other')                       AS modality_group,
            COUNT(*)                                                AS orders,
            COUNT_IF(has_result = TRUE)                             AS resulted,
            ROUND(COUNT_IF(has_result = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                        AS completion_pct,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)     AS pct_of_total
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_imaging_completion_monthly():
    """Monthly imaging completion rate — V1+V2, overall (not split by modality).
    Groups by request_date (not visit_month) — visit_month NULL for V2 Feb 2026+ due to Issue 86."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', request_date)                       AS visit_month,
            COUNT(*)                                                AS total_orders,
            COUNT_IF(has_result = TRUE)                             AS resulted,
            ROUND(COUNT_IF(has_result = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                        AS completion_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
        WHERE request_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)


def q_imaging_tat_dist():
    """Imaging TAT bucket distribution — order→scan completion.
    V2: all records. V1: 2024+ only (pre-2024 = physician review, different stage — Inv 137).
    Capped at 1440 min (24 hrs) to exclude V1 outliers (Issue 97)."""
    return run_query_df("""
        SELECT
            CASE
                WHEN result_tat_mins <=  30 THEN '≤30 min'
                WHEN result_tat_mins <=  60 THEN '31–60 min'
                WHEN result_tat_mins <= 120 THEN '1–2 hrs'
                WHEN result_tat_mins <= 240 THEN '2–4 hrs'
                WHEN result_tat_mins <= 480 THEN '4–8 hrs'
                ELSE '>8 hrs'
            END                                                     AS tat_bucket,
            CASE
                WHEN result_tat_mins <=  30 THEN 1
                WHEN result_tat_mins <=  60 THEN 2
                WHEN result_tat_mins <= 120 THEN 3
                WHEN result_tat_mins <= 240 THEN 4
                WHEN result_tat_mins <= 480 THEN 5
                ELSE 6
            END                                                     AS bucket_order,
            COUNT(*)                                                AS orders,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)     AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
        WHERE result_tat_mins IS NOT NULL
          AND result_tat_mins <= 1440
          AND (
              source_system = 'EMR_V2'
              OR (source_system = 'EMR_V1' AND YEAR(request_date) >= 2024)
          )
        GROUP BY 1, 2
        ORDER BY 2
    """)


def q_imaging_bridge():
    """Bridge KPI: V1 patients with unresulted OPD imaging who reached theatre within 30 days."""
    return run_query_df("""
        SELECT COUNT(DISTINCT i.composite_patient_id) AS patients_at_risk
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging i
        INNER JOIN HOSPITALS.STAGING.stg_procedures p
            ON i.composite_patient_id = p.composite_patient_id
           AND p.operation_date BETWEEN i.visit_date
               AND DATEADD('day', 30, i.visit_date)
        WHERE i.has_result = FALSE
          AND i.source_system = 'EMR_V1'
          AND p.source_system = 'EMR_V1'
          AND p.operation_date IS NOT NULL
    """)


def q_diag_demand_monthly():
    """Monthly demand: lab orders + imaging orders + OPD visits → orders per 100 visits.
    Uses request_date (not visit_month) — visit_month NULL for V2 Feb 2026+ due to Issue 86."""
    return run_query_df("""
        WITH lab AS (
            SELECT DATE_TRUNC('month', request_date) AS month, COUNT(*) AS lab_orders
            FROM HOSPITALS.REPORTING.rpt_ortho_lab
            WHERE request_date IS NOT NULL
            GROUP BY 1
        ),
        img AS (
            SELECT DATE_TRUNC('month', request_date) AS month, COUNT(*) AS imaging_orders
            FROM HOSPITALS.REPORTING.rpt_ortho_imaging
            WHERE request_date IS NOT NULL
            GROUP BY 1
        ),
        opd AS (
            SELECT DATE_TRUNC('month', visit_date) AS month, COUNT(*) AS opd_visits
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            WHERE visit_date IS NOT NULL
            GROUP BY 1
        )
        SELECT
            COALESCE(l.month, i.month, o.month)         AS month,
            COALESCE(l.lab_orders, 0)                   AS lab_orders,
            COALESCE(i.imaging_orders, 0)               AS imaging_orders,
            COALESCE(l.lab_orders, 0)
                + COALESCE(i.imaging_orders, 0)         AS total_orders,
            COALESCE(o.opd_visits, 0)                   AS opd_visits,
            ROUND(COALESCE(l.lab_orders, 0) * 100.0
                  / NULLIF(o.opd_visits, 0), 1)         AS lab_per_100,
            ROUND(COALESCE(i.imaging_orders, 0) * 100.0
                  / NULLIF(o.opd_visits, 0), 1)         AS imaging_per_100
        FROM lab l
        FULL OUTER JOIN img i ON l.month = i.month
        FULL OUTER JOIN opd o ON COALESCE(l.month, i.month) = o.month
        ORDER BY 1
    """)


def q_imaging_modality_tat():
    """Imaging TAT by modality — P50, P90, % within 60 min.
    V2 + V1 2024+ only (same stage: order→scan completion). Capped at 1440 min (Issue 97)."""
    return run_query_df("""
        SELECT
            COALESCE(modality_group, 'Other')                       AS modality_group,
            COUNT(*)                                                AS orders,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY result_tat_mins), 0)                   AS p50_mins,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                  (ORDER BY result_tat_mins), 0)                   AS p90_mins,
            COUNT_IF(result_tat_mins <= 60)                        AS within_60,
            ROUND(COUNT_IF(result_tat_mins <= 60) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                        AS pct_within_60
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
        WHERE result_tat_mins IS NOT NULL
          AND result_tat_mins <= 1440
          AND (
              source_system = 'EMR_V2'
              OR (source_system = 'EMR_V1' AND YEAR(request_date) >= 2024)
          )
        GROUP BY 1
        ORDER BY 3 DESC
    """)


def q_lab_tat_by_test():
    """Lab TAT by test type — P50, P90 for top 10 tests by volume. V1+V2 (consistent definition)."""
    return run_query_df("""
        WITH top_tests AS (
            SELECT UPPER(TRIM(test_name)) AS test_name, COUNT(*) AS n
            FROM HOSPITALS.REPORTING.rpt_ortho_lab
            WHERE test_name IS NOT NULL AND result_tat_mins IS NOT NULL
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 10
        )
        SELECT
            UPPER(TRIM(l.test_name))                                AS test_name,
            COUNT(*)                                                AS orders,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY l.result_tat_mins), 0)                 AS p50_mins,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                  (ORDER BY l.result_tat_mins), 0)                 AS p90_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_lab l
        INNER JOIN top_tests t ON UPPER(TRIM(l.test_name)) = t.test_name
        WHERE l.result_tat_mins IS NOT NULL
        GROUP BY 1
        ORDER BY 2 DESC
    """)


def q_lab_collect_wait_by_test():
    """Order→collect and collect→result P50 per test — V2 only. Shows which tests drive phlebotomy wait."""
    return run_query_df(f"""
        WITH top_tests AS (
            SELECT UPPER(TRIM(test_name)) AS test_name, COUNT(*) AS n
            FROM HOSPITALS.REPORTING.rpt_ortho_lab
            WHERE {_V2}
              AND collection_stamp IS NOT NULL
              AND collection_stamp > request_stamp
              AND test_name IS NOT NULL
            GROUP BY 1
            HAVING COUNT(*) >= 20
            ORDER BY 2 DESC
            LIMIT 12
        )
        SELECT
            UPPER(TRIM(l.test_name))                                            AS test_name,
            COUNT(*)                                                             AS orders,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                  DATEDIFF('minute', l.request_stamp, l.collection_stamp)), 0)  AS p50_order_to_collect,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                  DATEDIFF('minute', l.collection_stamp, l.result_stamp)), 0)   AS p50_collect_to_result
        FROM HOSPITALS.REPORTING.rpt_ortho_lab l
        INNER JOIN top_tests t ON UPPER(TRIM(l.test_name)) = t.test_name
        WHERE {_V2}
          AND l.collection_stamp IS NOT NULL
          AND l.collection_stamp > l.request_stamp
          AND l.result_stamp IS NOT NULL
        GROUP BY 1
        ORDER BY p50_order_to_collect DESC
    """)


def q_lab_chain_tat():
    """V2 lab 3-stage chain: order→collection TAT + collection→result TAT.
    Shows where delay occurs: pre-collection (queue) vs post-collection (processing)."""
    return run_query_df(f"""
        SELECT
            COUNT(*)                                                        AS total_v2,
            COUNT(collection_stamp)                                         AS with_collection,
            COUNT(result_stamp)                                             AS with_result,
            COUNT_IF(collection_stamp IS NOT NULL
                     AND result_stamp IS NOT NULL)                          AS full_chain,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                  DATEDIFF('minute', request_stamp, collection_stamp)), 0)  AS p50_order_to_collect,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY
                  DATEDIFF('minute', request_stamp, collection_stamp)), 0)  AS p90_order_to_collect,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                  DATEDIFF('minute', collection_stamp, result_stamp)), 0)   AS p50_collect_to_result,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY
                  DATEDIFF('minute', collection_stamp, result_stamp)), 0)   AS p90_collect_to_result
        FROM HOSPITALS.REPORTING.rpt_ortho_lab
        WHERE {_V2}
          AND collection_stamp IS NOT NULL
          AND collection_stamp > request_stamp
    """)


def q_imaging_tat_by_hour():
    """Imaging TAT by hour of day and modality — tests whether afternoon drives the tail.
    V2 + V1 2024+ only. Capped at 1440 min."""
    return run_query_df("""
        SELECT
            COALESCE(modality_group, 'Other')                       AS modality_group,
            HOUR(request_stamp)                                     AS request_hour,
            COUNT(*)                                                AS orders,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY result_tat_mins), 0)                   AS p50_mins,
            ROUND(COUNT_IF(result_tat_mins <= 60) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                        AS pct_within_60
        FROM HOSPITALS.REPORTING.rpt_ortho_imaging
        WHERE result_tat_mins IS NOT NULL
          AND result_tat_mins <= 1440
          AND request_stamp IS NOT NULL
          AND (
              source_system = 'EMR_V2'
              OR (source_system = 'EMR_V1' AND YEAR(request_date) >= 2024)
          )
        GROUP BY 1, 2
        HAVING COUNT(*) >= 10
        ORDER BY 1, 2
    """)


# ── Pharmacy ──────────────────────────────────────────────────────────────────

def q_pharm_dispensing_summary():
    """Pharmacy dispensing KPIs — V1+V2, by source_system.
    V1: is_served = boolean flag (66.6%). V2: is_served = status=2 (93.6%).
    Rates differ by field definition — do not blend (Rule B)."""
    return run_query_df("""
        SELECT
            source_system,
            COUNT(*)                                                    AS total_orders,
            COUNT_IF(is_served = TRUE)                                  AS dispensed,
            COUNT_IF(is_served = FALSE OR is_served IS NULL)            AS not_dispensed,
            ROUND(COUNT_IF(is_served = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                             AS fulfillment_rate,
            MIN(order_date)                                             AS data_from,
            MAX(order_date)                                             AS data_to
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        GROUP BY source_system
        ORDER BY source_system
    """)


def q_pharm_dispensing_monthly():
    """Monthly pharmacy dispensing — V1+V2, by source_system. order_month = dispensing month.
    Returns numerators + denominators; Python computes MoM per system."""
    return run_query_df("""
        SELECT
            source_system,
            order_month,
            COUNT(*)                                                    AS total_orders,
            COUNT_IF(is_served = TRUE)                                  AS dispensed,
            ROUND(COUNT_IF(is_served = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                             AS fulfillment_rate
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE order_month IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)


def q_pharm_class_breakdown():
    """Fulfillment by item class — V1+V2, by source_system. V2: all items classify as Drug."""
    return run_query_df("""
        SELECT
            source_system,
            COALESCE(item_class, 'Uncategorized')                       AS item_class,
            COUNT(*)                                                    AS total_orders,
            COUNT_IF(is_served = TRUE)                                  AS dispensed,
            COUNT_IF(is_served = FALSE OR is_served IS NULL)            AS not_dispensed,
            ROUND(COUNT_IF(is_served = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                             AS fulfillment_rate
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
    """)


def q_pharm_speed_summary():
    """Pharmacy dispensing interval summary — V1+V2, one row per source_system.
    OPD join removed — both systems use request_stamp→dispensed_stamp directly.
    V2 coverage ~61% (dispensed_stamp population). V1 includes all orders (not OPD-scoped)."""
    return run_query_df("""
        SELECT
            source_system,
            COUNT(*)                                                        AS total_orders,
            COUNT_IF(tat_mins IS NOT NULL)                                  AS orders_with_timestamps,
            COUNT_IF(tat_mins IS NOT NULL AND tat_mins < 240)               AS orders_in_window,
            COUNT_IF(tat_mins >= 240)                                       AS excluded_long_interval,
            ROUND(COUNT_IF(tat_mins IS NOT NULL AND tat_mins < 240) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                 AS coverage_pct,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP
                  (ORDER BY CASE WHEN tat_mins IS NOT NULL AND tat_mins < 240
                                 THEN tat_mins END), 0)                     AS p50_mins,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP
                  (ORDER BY CASE WHEN tat_mins IS NOT NULL AND tat_mins < 240
                                 THEN tat_mins END), 0)                     AS p90_mins,
            COUNT_IF(tat_mins >= 120 AND tat_mins < 240)                    AS btw_2_4hr_count,
            ROUND(COUNT_IF(tat_mins >= 120 AND tat_mins < 240) * 100.0
                  / NULLIF(COUNT_IF(tat_mins IS NOT NULL AND tat_mins < 240), 0), 1)
                                                                            AS btw_2_4hr_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        GROUP BY 1
        ORDER BY 1
    """)


def q_pharm_speed_distribution():
    """Pharmacy dispensing interval distribution — V1+V2, 3 buckets, tat_mins < 240.
    pct denominator partitioned by source_system. bucket_order for deterministic sort."""
    return run_query_df("""
        SELECT
            source_system,
            CASE
                WHEN tat_mins <  60  THEN 'Under 1 hr'
                WHEN tat_mins < 120  THEN '1–2 hrs'
                ELSE                      '2–4 hrs'
            END                                                         AS tat_bucket,
            CASE
                WHEN tat_mins <  60  THEN 1
                WHEN tat_mins < 120  THEN 2
                ELSE                      3
            END                                                         AS bucket_order,
            COUNT(*)                                                    AS orders,
            ROUND(COUNT(*) * 100.0
                  / SUM(COUNT(*)) OVER (PARTITION BY source_system), 1) AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE tat_mins IS NOT NULL
          AND tat_mins < 240
        GROUP BY 1, 2, 3
        ORDER BY 1, 3
    """)


def q_pharm_speed_monthly():
    """Monthly pharmacy dispensing interval trend — V1+V2, P50 + P75, tat_mins < 240."""
    return run_query_df("""
        SELECT
            source_system,
            order_month,
            COUNT(*)                                                        AS orders,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tat_mins), 0) AS p50_mins,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY tat_mins), 0) AS p75_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE tat_mins IS NOT NULL
          AND tat_mins < 240
          AND order_month IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)


def q_pharm_workload_summary():
    """V2 pharmacy workload summary — total orders, intensity vs OPD visits, non-dispensing count.
    orders_per_opd_visit = total V2 pharmacy orders / total V2 OPD visits (demand intensity)."""
    return run_query_df("""
        WITH pharm AS (
            SELECT
                COUNT(*)                        AS total_orders,
                COUNT_IF(is_served = FALSE)      AS not_dispensed,
                MIN(order_date)                  AS data_from,
                MAX(order_date)                  AS data_to
            FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
            WHERE source_system = 'EMR_V2'
        ),
        opd AS (
            SELECT COUNT(*) AS opd_visits
            FROM HOSPITALS.REPORTING.rpt_ortho_opd
            WHERE source_system = 'EMR_V2'
        )
        SELECT
            p.total_orders,
            p.not_dispensed,
            ROUND(p.not_dispensed * 100.0
                  / NULLIF(p.total_orders, 0), 1)               AS not_dispensed_pct,
            o.opd_visits,
            ROUND(p.total_orders * 1.0
                  / NULLIF(o.opd_visits, 0), 1)                 AS orders_per_opd_visit,
            p.data_from,
            p.data_to
        FROM pharm p
        CROSS JOIN opd o
    """)


def q_pharm_throughput_monthly():
    """Monthly V2 pharmacy order volume + dispensing TAT — for dual-axis volume/speed chart.
    tat_mins capped at 240 min. Months with <30 timed orders excluded (volatile)."""
    return run_query_df("""
        SELECT
            order_month,
            COUNT(*)                                                            AS total_orders,
            COUNT_IF(tat_mins IS NOT NULL AND tat_mins > 0 AND tat_mins < 240) AS timed_orders,
            ROUND(MEDIAN(CASE WHEN tat_mins > 0 AND tat_mins < 240
                              THEN tat_mins END), 0)                            AS p50_tat_mins,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY CASE WHEN tat_mins > 0 AND tat_mins < 240
                                 THEN tat_mins END), 0)                         AS p90_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE source_system = 'EMR_V2'
          AND order_month IS NOT NULL
        GROUP BY 1
        HAVING COUNT_IF(tat_mins IS NOT NULL AND tat_mins > 0 AND tat_mins < 240) >= 30
        ORDER BY 1
    """)


def q_pharm_tat_dist():
    """V2 dispensing TAT in 4 operational buckets — <30, 30-60, 60-120, >120 min.
    Capped at 240 min; orders above this threshold excluded as non-same-session."""
    return run_query_df("""
        SELECT
            CASE
                WHEN tat_mins <  30  THEN '< 30 min'
                WHEN tat_mins <  60  THEN '30–60 min'
                WHEN tat_mins < 120  THEN '1–2 hrs'
                ELSE                      '> 2 hrs'
            END                                                                 AS tat_bucket,
            CASE
                WHEN tat_mins <  30  THEN 1
                WHEN tat_mins <  60  THEN 2
                WHEN tat_mins < 120  THEN 3
                ELSE                      4
            END                                                                 AS bucket_order,
            COUNT(*)                                                            AS orders,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1)                 AS pct
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE source_system = 'EMR_V2'
          AND tat_mins IS NOT NULL
          AND tat_mins > 0
          AND tat_mins < 240
        GROUP BY 1, 2
        ORDER BY 2
    """)


def q_pharm_class_tat():
    """V2 item class × median dispensing TAT + volume — for priority medication groups.
    Excludes blank/null class and classes with <50 orders."""
    return run_query_df("""
        SELECT
            item_class,
            COUNT(*)                                                            AS total_orders,
            COUNT_IF(is_served = TRUE)                                          AS dispensed,
            ROUND(COUNT_IF(is_served = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                     AS dispensed_pct,
            ROUND(MEDIAN(CASE WHEN tat_mins > 0 AND tat_mins < 240
                              THEN tat_mins END), 0)                            AS median_tat_mins,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY CASE WHEN tat_mins > 0 AND tat_mins < 240
                                 THEN tat_mins END), 0)                         AS p90_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE source_system = 'EMR_V2'
          AND item_class IS NOT NULL
          AND item_class != ''
        GROUP BY 1
        HAVING COUNT(*) >= 50
        ORDER BY median_tat_mins DESC NULLS LAST
    """)


def q_pharm_top_items():
    """V2 top slowest items by median dispensing TAT — ≥30 orders, top 15.
    Capped at 240 min. over_2hr_pct flags items where tail delays are concentrated."""
    return run_query_df("""
        SELECT
            item_name,
            item_class,
            COUNT(*)                                                            AS total_orders,
            ROUND(MEDIAN(CASE WHEN tat_mins > 0 AND tat_mins < 240
                              THEN tat_mins END), 0)                            AS median_tat_mins,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP
                  (ORDER BY CASE WHEN tat_mins > 0 AND tat_mins < 240
                                 THEN tat_mins END), 0)                         AS p90_tat_mins,
            COUNT_IF(tat_mins >= 120 AND tat_mins < 240)                        AS over_2hr_count,
            ROUND(COUNT_IF(tat_mins >= 120 AND tat_mins < 240) * 100.0
                  / NULLIF(COUNT_IF(tat_mins IS NOT NULL
                                    AND tat_mins > 0
                                    AND tat_mins < 240), 0), 1)                AS over_2hr_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE source_system = 'EMR_V2'
          AND item_name IS NOT NULL
        GROUP BY 1, 2
        HAVING COUNT(*) >= 30
           AND MEDIAN(CASE WHEN tat_mins > 0 AND tat_mins < 240
                           THEN tat_mins END) IS NOT NULL
        ORDER BY median_tat_mins DESC NULLS LAST
        LIMIT 15
    """)


def q_pharm_hour():
    """V2 dispensing TAT by hour prescriptions are written (ORDER_HOUR) — peak analysis.
    Median TAT + order volume by hour. Capped at 240 min. Hours with <10 timed orders excluded."""
    return run_query_df("""
        SELECT
            order_hour,
            COUNT(*)                                                            AS total_orders,
            COUNT_IF(tat_mins IS NOT NULL AND tat_mins > 0 AND tat_mins < 240) AS timed_orders,
            ROUND(MEDIAN(CASE WHEN tat_mins > 0 AND tat_mins < 240
                              THEN tat_mins END), 0)                            AS median_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE source_system = 'EMR_V2'
          AND order_hour IS NOT NULL
        GROUP BY 1
        HAVING COUNT_IF(tat_mins IS NOT NULL AND tat_mins > 0 AND tat_mins < 240) >= 10
        ORDER BY 1
    """)


def q_pharm_dow():
    """V2 dispensing TAT by day of week — median TAT + volume. Mon–Sun sort."""
    return run_query_df("""
        SELECT
            order_day_name,
            CASE order_day_name
                WHEN 'Mon' THEN 1  WHEN 'Tue' THEN 2  WHEN 'Wed' THEN 3
                WHEN 'Thu' THEN 4  WHEN 'Fri' THEN 5  WHEN 'Sat' THEN 6
                WHEN 'Sun' THEN 7  ELSE 8
            END                                                                 AS dow_sort,
            COUNT(*)                                                            AS total_orders,
            ROUND(MEDIAN(CASE WHEN tat_mins > 0 AND tat_mins < 240
                              THEN tat_mins END), 0)                            AS median_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE source_system = 'EMR_V2'
          AND order_day_name IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 2
    """)


# ── Physician Dependence ──────────────────────────────────────────────────────

def q_physician_kpis():
    """Dependency KPIs — V2 only. total_physicians, top1_share, top3_share, HHI."""
    return run_query_df(f"""
        WITH base AS (
            SELECT
                doctor_hash,
                COUNT(*)                            AS admissions,
                SUM(COUNT(*)) OVER ()               AS grand_total
            FROM HOSPITALS.REPORTING.rpt_ortho_physician_dependence
            WHERE {_V2}
              AND doctor_hash IS NOT NULL
            GROUP BY 1
        ),
        shares AS (
            SELECT
                doctor_hash,
                admissions,
                admissions * 1.0 / grand_total      AS share,
                ROW_NUMBER() OVER (ORDER BY admissions DESC) AS rnk
            FROM base
        )
        SELECT
            COUNT(*)                                                    AS total_physicians,
            ROUND(MAX(CASE WHEN rnk = 1 THEN share END) * 100, 1)     AS top1_share,
            ROUND(SUM(CASE WHEN rnk <= 3 THEN share END) * 100, 1)    AS top3_share,
            ROUND(SUM(POWER(share, 2)), 4)                              AS hhi
        FROM shares
    """)


def q_physician_workload():
    """Per-physician ranked admissions — V2 only. Top 25, ≥ 5 admissions. doctor_hash only (Issue 72)."""
    return run_query_df(f"""
        WITH base AS (
            SELECT
                doctor_hash,
                COUNT(*)                            AS admissions,
                SUM(COUNT(*)) OVER ()               AS grand_total
            FROM HOSPITALS.REPORTING.rpt_ortho_physician_dependence
            WHERE {_V2}
              AND doctor_hash IS NOT NULL
            GROUP BY 1
        )
        SELECT
            doctor_hash,
            admissions,
            ROUND(admissions * 100.0 / grand_total, 1)  AS share_pct
        FROM base
        WHERE admissions >= 5
        ORDER BY admissions DESC
        LIMIT 25
    """)


def q_physician_efficiency():
    """Volume vs median LOS scatter — V2 only. ≥ 20 admissions, LOS capped 0–60 days."""
    return run_query_df(f"""
        SELECT
            doctor_hash,
            COUNT(*)                    AS admissions,
            ROUND(MEDIAN(los_days), 1)  AS median_los
        FROM HOSPITALS.REPORTING.rpt_ortho_physician_dependence
        WHERE {_V2}
          AND doctor_hash IS NOT NULL
          AND los_days BETWEEN 0 AND 60
        GROUP BY 1
        HAVING COUNT(*) >= 20
        ORDER BY admissions DESC
    """)


def q_physician_trend():
    """Monthly top-1 and top-3 physician share — V2 only. Is concentration changing?"""
    return run_query_df(f"""
        WITH monthly AS (
            SELECT
                admission_month,
                doctor_hash,
                COUNT(*)                                            AS admissions,
                SUM(COUNT(*)) OVER (PARTITION BY admission_month)  AS month_total
            FROM HOSPITALS.REPORTING.rpt_ortho_physician_dependence
            WHERE {_V2}
              AND doctor_hash IS NOT NULL
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT
                admission_month,
                admissions,
                month_total,
                ROW_NUMBER() OVER (PARTITION BY admission_month ORDER BY admissions DESC) AS rnk,
                admissions * 100.0 / month_total                    AS share_pct
            FROM monthly
            WHERE month_total >= 20
        )
        SELECT
            admission_month,
            ROUND(MAX(CASE WHEN rnk = 1 THEN share_pct END), 1)    AS top1_share,
            ROUND(SUM(CASE WHEN rnk <= 3 THEN share_pct END), 1)   AS top3_share,
            MAX(month_total)                                         AS total_admissions
        FROM ranked
        GROUP BY 1
        ORDER BY 1
    """)


def q_physician_continuity():
    """Admitting physician vs treating physician match — V2 only."""
    return run_query_df(f"""
        SELECT
            COUNT_IF(doctor_hash IS NOT NULL
                     AND admitted_by_hash IS NOT NULL)              AS both_recorded,
            COUNT_IF(doctor_hash = admitted_by_hash)                AS same_physician,
            COUNT_IF(doctor_hash <> admitted_by_hash
                     AND doctor_hash IS NOT NULL
                     AND admitted_by_hash IS NOT NULL)              AS transferred,
            ROUND(
                COUNT_IF(doctor_hash = admitted_by_hash) * 100.0 /
                NULLIF(COUNT_IF(doctor_hash IS NOT NULL
                                AND admitted_by_hash IS NOT NULL), 0),
            1)                                                      AS match_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_physician_dependence
        WHERE {_V2}
    """)


def q_physician_rank_trend():
    """Monthly rank of top 5 overall physicians — V2 only. For bump chart."""
    return run_query_df(f"""
        WITH all_monthly AS (
            SELECT
                admission_month,
                doctor_hash,
                COUNT(*)                                            AS admissions,
                SUM(COUNT(*)) OVER (PARTITION BY admission_month)  AS month_total
            FROM HOSPITALS.REPORTING.rpt_ortho_physician_dependence
            WHERE {_V2}
              AND doctor_hash IS NOT NULL
            GROUP BY 1, 2
        ),
        top5_overall AS (
            SELECT
                doctor_hash,
                ROW_NUMBER() OVER (ORDER BY SUM(admissions) DESC)  AS overall_rank
            FROM all_monthly
            GROUP BY 1
            QUALIFY ROW_NUMBER() OVER (ORDER BY SUM(admissions) DESC) <= 5
        ),
        monthly_ranked AS (
            SELECT
                am.admission_month,
                am.doctor_hash,
                am.admissions,
                ROW_NUMBER() OVER (PARTITION BY am.admission_month ORDER BY am.admissions DESC) AS monthly_rank
            FROM all_monthly am
            WHERE am.month_total >= 20
        )
        SELECT
            mr.admission_month,
            t.overall_rank,
            mr.monthly_rank,
            mr.admissions
        FROM monthly_ranked mr
        JOIN top5_overall t ON mr.doctor_hash = t.doctor_hash
        ORDER BY 1, 2
    """)


# ── Busy Periods ─────────────────────────────────────────────────────────────

def q_busy_dow_summary():
    """OPD weekly volume distribution by day — from rpt_ortho_busy_periods V1."""
    return run_query_df("""
        SELECT
            visit_day_name,
            visit_dow,
            SUM(total_visits)                                       AS total_visits,
            ROUND(SUM(pct_of_total), 1)                             AS pct_of_weekly_total,
            SUM(CASE WHEN is_peak_slot THEN 1 ELSE 0 END)           AS peak_hours_count
        FROM HOSPITALS.REPORTING.rpt_ortho_busy_periods
        GROUP BY 1, 2
        ORDER BY 2
    """)


def q_busy_peak_window():
    """Peak operating window — top 10 weekday slots by avg_per_day.
    Uses top-N rather than P75 range to avoid spanning the full working day.
    Returns the hour window where the highest-intensity visits concentrate."""
    return run_query_df("""
        WITH top_slots AS (
            SELECT hour_of_day, avg_per_day
            FROM HOSPITALS.REPORTING.rpt_ortho_busy_periods
            WHERE visit_dow BETWEEN 1 AND 5
              AND hour_of_day >= 7
            ORDER BY avg_per_day DESC
            LIMIT 10
        )
        SELECT
            MIN(hour_of_day)           AS peak_hour_start,
            MAX(hour_of_day) + 1       AS peak_hour_end,
            ROUND(AVG(avg_per_day), 1) AS avg_per_slot,
            ROUND(MAX(avg_per_day), 1) AS max_per_slot,
            COUNT(*)                   AS peak_slot_count
        FROM top_slots
    """)


# ── Shared ────────────────────────────────────────────────────────────────────

def q_data_freshness():
    """Latest visit date in each V1 reporting view — for sidebar freshness label."""
    return run_query_df(f"""
        SELECT MAX(visit_date)::DATE AS max_date
        FROM HOSPITALS.REPORTING.rpt_ortho_opd
        WHERE {_V1}
    """)


# ── Revenue Leakage (V2 mart) ────────────────────────────────────────────────

def q_leakage_summary():
    """Leakage KPIs — clinical procedures only. V2 only (mart is V2 by design)."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                    AS total_clinical,
            COUNT_IF(is_leakage = 1)                                    AS leakage_count,
            ROUND(100.0 * COUNT_IF(is_leakage = 1) / COUNT(*), 1)      AS leakage_rate_pct,
            ROUND(100.0 * COUNT_IF(is_leakage = 0) / COUNT(*), 1)      AS collection_rate_pct,
            COALESCE(SUM(uncollected_value), 0)                         AS total_uncollected_kes,
            MIN(request_date)                                           AS data_from,
            MAX(request_date)                                           AS data_to
        FROM HOSPITALS.REPORTING.mart_revenue_leakage
        WHERE proc_category = 'Clinical'
    """)


def q_leakage_by_procedure():
    """Leakage by procedure type — sorted by uncollected value desc. Pareto share included."""
    return run_query_df("""
        WITH proc_agg AS (
            SELECT
                request_name,
                COUNT(*)                                                AS total_requests,
                COUNT_IF(is_leakage = 1)                                AS leakage_count,
                ROUND(100.0 * COUNT_IF(is_leakage = 1) / COUNT(*), 1)  AS leakage_pct,
                COALESCE(SUM(uncollected_value), 0)                     AS uncollected_kes
            FROM HOSPITALS.REPORTING.mart_revenue_leakage
            WHERE proc_category = 'Clinical'
            GROUP BY 1
        )
        SELECT
            *,
            ROUND(100.0 * uncollected_kes
                / NULLIF(SUM(uncollected_kes) OVER (), 0), 1)           AS share_of_total_pct
        FROM proc_agg
        ORDER BY uncollected_kes DESC NULLS LAST
    """)


def q_leakage_monthly():
    """Monthly clinical collection rate + uncollected KES. V2 only."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', request_date)                           AS request_month,
            COUNT(*)                                                    AS clinical_requests,
            COUNT_IF(is_leakage = 1)                                    AS leakage_count,
            ROUND(100.0 * COUNT_IF(is_leakage = 0) / COUNT(*), 1)      AS collection_rate_pct,
            COALESCE(SUM(uncollected_value), 0)                         AS uncollected_kes
        FROM HOSPITALS.REPORTING.mart_revenue_leakage
        WHERE proc_category = 'Clinical'
          AND request_date IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)


def q_leakage_prev_month():
    """Previous complete calendar month leakage KPIs. V2 only. Used as the 'last closed month' KPI card."""
    return run_query_df("""
        SELECT
            TO_CHAR(
                DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE)),
                'Mon YYYY'
            )                                                           AS month_label,
            COUNT(*)                                                    AS clinical_requests,
            COUNT_IF(is_leakage = 1)                                    AS leakage_count,
            ROUND(100.0 * COUNT_IF(is_leakage = 1) / COUNT(*), 1)      AS leakage_rate_pct,
            ROUND(100.0 * COUNT_IF(is_leakage = 0) / COUNT(*), 1)      AS collection_rate_pct,
            COALESCE(SUM(uncollected_value), 0)                         AS uncollected_kes
        FROM HOSPITALS.REPORTING.mart_revenue_leakage
        WHERE proc_category = 'Clinical'
          AND DATE_TRUNC('month', request_date)
                = DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE))
    """)


# ── Patient Waiting (V2 mart) ────────────────────────────────────────────────

def q_waiting_rbi_summary():
    """Per-stage RBI + TAT comparison for insight card and RBI table.
    4 stages: Consult, Pharmacy, Lab, Imaging. Sorted by RBI score desc (highest = top priority).
    current_p50 = avg of daily P50 over recent 28d from MAX(period_date) anchor.
    prior_p50 = avg over preceding 28d. Anchor = MAX(period_date) — mart is not refreshed daily;
    CURRENT_DATE anchor left only 1-2 rows in the current window (same bug as q_cc_pipeline)."""
    return run_query_df("""
        WITH latest AS (
            SELECT MAX(period_date) AS d FROM HOSPITALS.REPORTING.mart_operational_kpis
        ),
        base AS (
            SELECT * FROM HOSPITALS.REPORTING.mart_operational_kpis
            WHERE period_date >= (SELECT DATEADD('day', -56, d) FROM latest)
              AND period_date <= (SELECT d FROM latest)
        ),
        cons AS (
            SELECT
                'Consult'                                           AS stage,
                'Clinical'                                         AS operational_owner,
                'Review clinician capacity and clinic scheduling'  AS recommended_action,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN cons_p50_mins      END), 0) AS current_p50_mins,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN cons_p50_mins      END), 0) AS prior_p50_mins,
                MAX_BY(cons_rbi,       period_date)                                                                             AS rbi_score,
                MAX_BY(cons_rbi_label, period_date)                                                                             AS rbi_label,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN cons_coverage_pct  END), 1) AS coverage_pct,
                COALESCE(SUM(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN cons_n          END), 0) AS coverage_n,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN opd_visits         END), 0) AS current_avg_vol,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN opd_visits         END), 0) AS prior_avg_vol
            FROM base
        ),
        pharm AS (
            SELECT
                'Pharmacy'                                         AS stage,
                'Pharmacy'                                         AS operational_owner,
                'Review dispensing workflow and staffing'          AS recommended_action,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN pharm_dispensing_p50_mins  END), 0) AS current_p50_mins,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN pharm_dispensing_p50_mins  END), 0) AS prior_p50_mins,
                MAX_BY(pharm_rbi,       period_date)                                                                                   AS rbi_score,
                MAX_BY(pharm_rbi_label, period_date)                                                                                   AS rbi_label,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN pharm_disp_coverage_pct    END), 1) AS coverage_pct,
                COALESCE(SUM(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN pharm_disp_n            END), 0) AS coverage_n,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN opd_visits              END), 0) AS current_avg_vol,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN opd_visits              END), 0) AS prior_avg_vol
            FROM base
        ),
        lab AS (
            SELECT
                'Lab'                                              AS stage,
                'Laboratory'                                       AS operational_owner,
                'Review specimen processing capacity'              AS recommended_action,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN lab_p50_mins      END), 0) AS current_p50_mins,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN lab_p50_mins      END), 0) AS prior_p50_mins,
                MAX_BY(lab_rbi,       period_date)                                                                             AS rbi_score,
                MAX_BY(lab_rbi_label, period_date)                                                                             AS rbi_label,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN lab_coverage_pct  END), 1) AS coverage_pct,
                COALESCE(SUM(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN lab_n          END), 0) AS coverage_n,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN lab_denominator   END), 0) AS current_avg_vol,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN lab_denominator   END), 0) AS prior_avg_vol
            FROM base
        ),
        img AS (
            SELECT
                'Imaging'                                          AS stage,
                'Radiology'                                        AS operational_owner,
                'Review radiology scheduling and capacity'         AS recommended_action,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN imaging_p50_mins      END), 0) AS current_p50_mins,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN imaging_p50_mins      END), 0) AS prior_p50_mins,
                MAX_BY(imaging_rbi,       period_date)                                                                             AS rbi_score,
                MAX_BY(imaging_rbi_label, period_date)                                                                             AS rbi_label,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN imaging_coverage_pct  END), 1) AS coverage_pct,
                COALESCE(SUM(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN imaging_n          END), 0) AS coverage_n,
                ROUND(AVG(CASE WHEN period_date >= (SELECT DATEADD('day',-28,d) FROM latest) THEN imaging_denominator   END), 0) AS current_avg_vol,
                ROUND(AVG(CASE WHEN period_date <  (SELECT DATEADD('day',-28,d) FROM latest) THEN imaging_denominator   END), 0) AS prior_avg_vol
            FROM base
        ),
        all_stages AS (
            SELECT * FROM cons
            UNION ALL SELECT * FROM pharm
            UNION ALL SELECT * FROM lab
            UNION ALL SELECT * FROM img
        )
        SELECT *,
            ROUND(100.0 * (current_p50_mins - prior_p50_mins)
                  / NULLIF(prior_p50_mins, 0), 1)                  AS pct_change_28d
        FROM all_stages
        ORDER BY rbi_score DESC NULLS LAST
    """)


def q_waiting_tat_trend():
    """Daily TAT trend per stage, previous complete calendar month. V2 OPD dates only (opd_visits > 0)."""
    return run_query_df("""
        SELECT
            period_date,
            opd_visits,
            cons_p50_mins,
            cons_coverage_pct,
            pharm_tat_p50_mins,
            pharm_tat_coverage_pct,
            lab_p50_mins,
            lab_coverage_pct,
            imaging_p50_mins,
            imaging_coverage_pct,
            cycle_p50_mins,
            cycle_coverage_pct
        FROM HOSPITALS.REPORTING.mart_operational_kpis
        WHERE period_date >= DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE))
          AND period_date <  DATE_TRUNC('month', CURRENT_DATE)
          AND opd_visits  >  0
        ORDER BY period_date
    """)


def q_waiting_dow_scatter():
    """Average volume × average median consult TAT by day of week, last 90 days.
    Used for the Volume × TAT quadrant scatter — identifies which DOW needs staffing intervention."""
    return run_query_df("""
        SELECT
            DAYOFWEEK(period_date)              AS dow_num,
            DAYNAME(period_date)                AS day_name,
            ROUND(AVG(opd_visits), 0)           AS avg_daily_visits,
            ROUND(AVG(cons_p50_mins), 0)        AS avg_cons_p50_mins,
            ROUND(AVG(cons_coverage_pct), 1)    AS avg_coverage_pct,
            COUNT(*)                            AS observation_days
        FROM HOSPITALS.REPORTING.mart_operational_kpis
        WHERE period_date >= DATEADD('day', -90, CURRENT_DATE)
          AND period_date <  CURRENT_DATE
          AND opd_visits  >  0
          AND cons_p50_mins IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1
    """)


def q_waiting_dept_tat():
    """Consult TAT by department — last 28 days from MAX anchor.
    Excludes depts with fewer than 10 timed consults."""
    return run_query_df(f"""
        WITH latest AS (
            SELECT MAX(pj.visit_date) AS d
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
            WHERE pj.source_system = 'EMR_V2'
        )
        SELECT
            pj.dept,
            COUNT(*)                                                    AS visits,
            COUNT_IF(pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480) AS valid_n,
            ROUND(MEDIAN(
                CASE WHEN pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                     AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
            ), 0)                                                       AS p50_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_date >= (SELECT DATEADD('day', -28, d) FROM latest)
          AND pj.dept IS NOT NULL
        GROUP BY 1
        HAVING valid_n >= 10
        ORDER BY p50_mins DESC NULLS LAST
    """)


def q_waiting_heatmap():
    """DOW × hour-of-day heatmap from spine. Color = median consult TAT. V2 OPD only.
    Grain: dow_num × hour_of_day aggregated over the previous complete calendar month. Min 3 visits per cell."""
    return run_query_df("""
        SELECT
            DAYOFWEEK(arrival_ts)                                           AS dow_num,
            DAYNAME(arrival_ts)                                             AS day_name,
            HOUR(arrival_ts)                                                AS hour_of_day,
            COUNT(*)                                                        AS visit_count,
            COUNT_IF(cons_ts IS NOT NULL AND cons_ts >= arrival_ts)         AS tat_n,
            ROUND(
                MEDIAN(CASE
                    WHEN cons_ts IS NOT NULL
                     AND cons_ts >= arrival_ts
                     AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                    THEN DATEDIFF('minute', arrival_ts, cons_ts)
                END)
            , 0)                                                            AS median_cons_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE source_system = 'EMR_V2'
          AND arrival_ts IS NOT NULL
          AND arrival_ts >= DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE))
          AND arrival_ts <  DATE_TRUNC('month', CURRENT_DATE)
          AND HOUR(arrival_ts) BETWEEN 6 AND 22
        GROUP BY 1, 2, 3
        HAVING COUNT(*) >= 3
        ORDER BY 1, 3
    """)


def q_waiting_weekly_tat():
    """Weekly OPD volume + P50 consult TAT — last 12 complete weeks.
    Baseline = 12-week average. Classification per week:
      'capacity' — TAT > 90 min AND volume > 15% above baseline
      'process'  — TAT > 90 min, volume at or below baseline (possible staffing/process constraint)
      'normal'   — TAT ≤ 90 min"""
    return run_query_df("""
        WITH weeks AS (
            SELECT
                DATE_TRUNC('week', visit_date)                              AS week_start,
                COUNT(*)                                                    AS weekly_visits,
                COUNT_IF(
                    cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                    AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                )                                                           AS valid_n,
                ROUND(MEDIAN(
                    CASE WHEN cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                         AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                    THEN DATEDIFF('minute', arrival_ts, cons_ts) END
                ), 0)                                                       AS p50_tat_mins
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
              AND visit_type   <> 'Inpatient'
              AND arrival_ts   IS NOT NULL
              AND visit_date   >= DATEADD('week', -12, DATE_TRUNC('week', CURRENT_DATE))
              AND visit_date   <  DATE_TRUNC('week', CURRENT_DATE)
            GROUP BY 1
            HAVING weekly_visits >= 5
        ),
        baseline AS (
            SELECT
                AVG(weekly_visits) AS avg_visits,
                AVG(p50_tat_mins)  AS avg_tat
            FROM weeks
        )
        SELECT
            w.week_start,
            w.weekly_visits,
            w.valid_n,
            w.p50_tat_mins,
            ROUND(b.avg_visits, 0)                                          AS avg_visits_baseline,
            ROUND(b.avg_tat, 0)                                             AS avg_tat_baseline,
            ROUND(100.0 * (w.weekly_visits - b.avg_visits)
                / NULLIF(b.avg_visits, 0), 1)                               AS volume_delta_pct,
            ROUND(100.0 * (w.p50_tat_mins  - b.avg_tat)
                / NULLIF(b.avg_tat, 0), 1)                                  AS tat_delta_pct,
            CASE
                WHEN w.p50_tat_mins > 90
                     AND (w.weekly_visits / NULLIF(b.avg_visits, 0)) > 1.15
                THEN 'capacity'
                WHEN w.p50_tat_mins > 90
                THEN 'process'
                ELSE 'normal'
            END                                                             AS flag_type
        FROM weeks w
        CROSS JOIN baseline b
        ORDER BY w.week_start
    """)


def q_waiting_heatmap_flagged():
    """DOW × hour heatmap scoped only to weeks where P50 consult TAT > 90 min.
    Same column structure as q_waiting_heatmap. Returns empty if no flagged weeks.
    Covers last 12 complete weeks. Min 3 visits per cell."""
    return run_query_df("""
        WITH flagged_weeks AS (
            SELECT DATE_TRUNC('week', visit_date) AS week_start
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
              AND visit_type   <> 'Inpatient'
              AND arrival_ts   IS NOT NULL
              AND cons_ts      IS NOT NULL
              AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
              AND visit_date   >= DATEADD('week', -12, DATE_TRUNC('week', CURRENT_DATE))
              AND visit_date   <  DATE_TRUNC('week', CURRENT_DATE)
            GROUP BY 1
            HAVING COUNT_IF(DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480) >= 5
               AND MEDIAN(DATEDIFF('minute', arrival_ts, cons_ts)) > 90
        )
        SELECT
            DAYOFWEEK(pj.arrival_ts)                                        AS dow_num,
            DAYNAME(pj.arrival_ts)                                          AS day_name,
            HOUR(pj.arrival_ts)                                             AS hour_of_day,
            COUNT(*)                                                        AS visit_count,
            COUNT_IF(pj.cons_ts IS NOT NULL AND pj.cons_ts >= pj.arrival_ts) AS tat_n,
            ROUND(MEDIAN(
                CASE WHEN pj.cons_ts IS NOT NULL
                      AND pj.cons_ts >= pj.arrival_ts
                      AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
            ), 0)                                                           AS median_cons_tat_mins
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        JOIN flagged_weeks fw ON DATE_TRUNC('week', pj.visit_date) = fw.week_start
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_type   <> 'Inpatient'
          AND pj.arrival_ts   IS NOT NULL
          AND HOUR(pj.arrival_ts) BETWEEN 6 AND 22
        GROUP BY 1, 2, 3
        HAVING visit_count >= 3
        ORDER BY 1, 3
    """)


def q_waiting_dept_pressure():
    """Per-dept P50 consult TAT — 28-day baseline vs flagged weeks (P50 > 90 min, last 12 weeks).
    Only depts present in BOTH windows with ≥5 timed consults each.
    delta_mins = flagged P50 − baseline P50. Large positive delta = pressure-sensitive dept.
    Returns empty if no flagged weeks exist."""
    return run_query_df("""
        WITH latest AS (
            SELECT MAX(visit_date) AS d
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
        ),
        flagged_weeks AS (
            SELECT DATE_TRUNC('week', visit_date) AS week_start
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
              AND visit_type   <> 'Inpatient'
              AND arrival_ts   IS NOT NULL
              AND cons_ts      IS NOT NULL
              AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
              AND visit_date   >= DATEADD('week', -12, DATE_TRUNC('week', CURRENT_DATE))
              AND visit_date   <  DATE_TRUNC('week', CURRENT_DATE)
            GROUP BY 1
            HAVING COUNT_IF(DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480) >= 5
               AND MEDIAN(DATEDIFF('minute', arrival_ts, cons_ts)) > 90
        ),
        baseline AS (
            SELECT
                pj.dept,
                COUNT_IF(
                    pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                    AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                )                                                           AS valid_n,
                ROUND(MEDIAN(
                    CASE WHEN pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                         AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                    THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
                ), 0)                                                       AS p50_baseline
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
            WHERE pj.source_system = 'EMR_V2'
              AND pj.visit_type   <> 'Inpatient'
              AND pj.arrival_ts   IS NOT NULL
              AND pj.dept         IS NOT NULL
              AND pj.visit_date   >= (SELECT DATEADD('day', -28, d) FROM latest)
            GROUP BY 1
            HAVING valid_n >= 5
        ),
        flagged AS (
            SELECT
                pj.dept,
                COUNT_IF(
                    pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                    AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                )                                                           AS valid_n,
                ROUND(MEDIAN(
                    CASE WHEN pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                         AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                    THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
                ), 0)                                                       AS p50_flagged
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
            JOIN flagged_weeks fw ON DATE_TRUNC('week', pj.visit_date) = fw.week_start
            WHERE pj.source_system = 'EMR_V2'
              AND pj.visit_type   <> 'Inpatient'
              AND pj.arrival_ts   IS NOT NULL
              AND pj.dept         IS NOT NULL
            GROUP BY 1
            HAVING valid_n >= 5
        )
        SELECT
            b.dept,
            b.p50_baseline,
            f.p50_flagged,
            (f.p50_flagged - b.p50_baseline)                               AS delta_mins,
            ROUND(100.0 * (f.p50_flagged - b.p50_baseline)
                / NULLIF(b.p50_baseline, 0), 1)                            AS delta_pct
        FROM baseline b
        JOIN flagged f ON b.dept = f.dept
        ORDER BY delta_mins DESC NULLS LAST
    """)


def q_waiting_spillover_summary():
    """Operational metrics: bottleneck days vs normal days — last 12 complete weeks.
    Bottleneck day: daily P50 consult TAT > 90 min, ≥10 valid timed consults.
    Returns 2 rows (normal first, bottleneck second — ORDER BY day_type DESC).
    Columns: day_type, days_n, avg_p50_consult_mins, avg_daily_visits,
             avg_ancillary_completion_pct, avg_pharmacy_pct, avg_p50_pharm_wait_mins."""
    return run_query_df("""
        WITH daily_consult AS (
            SELECT
                visit_date,
                COUNT_IF(
                    cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                    AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                )                                                           AS valid_n,
                ROUND(MEDIAN(
                    CASE WHEN cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                         AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                    THEN DATEDIFF('minute', arrival_ts, cons_ts) END
                ), 0)                                                       AS p50_consult_mins,
                CASE
                    WHEN COUNT_IF(
                        cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                        AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                    ) >= 10
                    AND MEDIAN(
                        CASE WHEN cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                             AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                        THEN DATEDIFF('minute', arrival_ts, cons_ts) END
                    ) > 90
                    THEN 'bottleneck'
                    ELSE 'normal'
                END                                                         AS day_type
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
              AND visit_type   <> 'Inpatient'
              AND arrival_ts   IS NOT NULL
              AND visit_date   >= DATEADD('week', -12, DATE_TRUNC('week', CURRENT_DATE))
              AND visit_date   <  DATE_TRUNC('week', CURRENT_DATE)
            GROUP BY 1
        ),
        daily_downstream AS (
            SELECT
                visit_date,
                COUNT(*)                                                    AS daily_visits,
                ROUND(100.0 * COUNT_IF(had_pharmacy = 1)
                    / NULLIF(COUNT(*), 0), 1)                               AS pharmacy_pct,
                ROUND(100.0 * COUNT_IF(
                    had_pharmacy = 1 OR had_lab_order = 1
                    OR had_imaging = 1 OR had_proc_request = 1
                ) / NULLIF(COUNT(*), 0), 1)                                AS ancillary_completion_pct,
                ROUND(MEDIAN(
                    CASE WHEN pharm_ts IS NOT NULL AND cons_ts IS NOT NULL
                         AND DATEDIFF('minute', cons_ts, pharm_ts) BETWEEN 1 AND 480
                    THEN DATEDIFF('minute', cons_ts, pharm_ts) END
                ), 0)                                                       AS p50_pharm_wait_mins
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
              AND visit_type   <> 'Inpatient'
              AND arrival_ts   IS NOT NULL
              AND visit_date   >= DATEADD('week', -12, DATE_TRUNC('week', CURRENT_DATE))
              AND visit_date   <  DATE_TRUNC('week', CURRENT_DATE)
            GROUP BY 1
        )
        SELECT
            dc.day_type,
            COUNT(dc.visit_date)                                            AS days_n,
            ROUND(AVG(dc.p50_consult_mins), 0)                             AS avg_p50_consult_mins,
            ROUND(AVG(dd.daily_visits), 0)                                 AS avg_daily_visits,
            ROUND(AVG(dd.ancillary_completion_pct), 1)                     AS avg_ancillary_completion_pct,
            ROUND(AVG(dd.pharmacy_pct), 1)                                 AS avg_pharmacy_pct,
            ROUND(AVG(dd.p50_pharm_wait_mins), 0)                          AS avg_p50_pharm_wait_mins
        FROM daily_consult dc
        JOIN daily_downstream dd ON dc.visit_date = dd.visit_date
        GROUP BY 1
        ORDER BY 1 DESC
    """)


def q_waiting_service_breakdown():
    """Per-service downstream completion rate — bottleneck vs normal days — last 12 complete weeks.
    Returns 2 rows (normal first, bottleneck second).
    Columns: day_type, total_visits, pharmacy_pct, lab_pct, imaging_pct, proc_pct."""
    return run_query_df("""
        WITH daily_consult AS (
            SELECT
                visit_date,
                CASE
                    WHEN COUNT_IF(
                        cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                        AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                    ) >= 10
                    AND MEDIAN(
                        CASE WHEN cons_ts IS NOT NULL AND arrival_ts IS NOT NULL
                             AND DATEDIFF('minute', arrival_ts, cons_ts) BETWEEN 1 AND 480
                        THEN DATEDIFF('minute', arrival_ts, cons_ts) END
                    ) > 90
                    THEN 'bottleneck'
                    ELSE 'normal'
                END                                                         AS day_type
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE source_system = 'EMR_V2'
              AND visit_type   <> 'Inpatient'
              AND arrival_ts   IS NOT NULL
              AND visit_date   >= DATEADD('week', -12, DATE_TRUNC('week', CURRENT_DATE))
              AND visit_date   <  DATE_TRUNC('week', CURRENT_DATE)
            GROUP BY 1
        )
        SELECT
            dc.day_type,
            COUNT(*)                                                         AS total_visits,
            ROUND(100.0 * COUNT_IF(pj.had_pharmacy = 1)
                / NULLIF(COUNT(*), 0), 1)                                   AS pharmacy_pct,
            ROUND(100.0 * COUNT_IF(pj.had_lab_order = 1)
                / NULLIF(COUNT(*), 0), 1)                                   AS lab_pct,
            ROUND(100.0 * COUNT_IF(pj.had_imaging = 1)
                / NULLIF(COUNT(*), 0), 1)                                   AS imaging_pct,
            ROUND(100.0 * COUNT_IF(pj.had_proc_request = 1)
                / NULLIF(COUNT(*), 0), 1)                                   AS proc_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        JOIN daily_consult dc ON pj.visit_date = dc.visit_date
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_type   <> 'Inpatient'
          AND pj.arrival_ts   IS NOT NULL
          AND pj.visit_date   >= DATEADD('week', -12, DATE_TRUNC('week', CURRENT_DATE))
          AND pj.visit_date   <  DATE_TRUNC('week', CURRENT_DATE)
        GROUP BY 1
        ORDER BY 1 DESC
    """)


# ── Drop-off / Pathway ───────────────────────────────────────────────────────

def q_dropoff_kpis():
    """Headline KPIs for Care Pathway Completion page. Single row.
    Base = all V2 non-Inpatient arrivals (visit_type <> 'Inpatient', arrival_ts IS NOT NULL)
    — same population as q_dropoff_sankey_v2 so KPI denominator matches Sankey TOTAL.
    Previously used disposition = 'OPD' which excluded ~1,800 OPD visits that progressed
    to admission, creating a denominator mismatch with the Sankey's 'All Arrivals' node."""
    return run_query_df("""
        SELECT
            ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
                / NULLIF(COUNT(*), 0), 1)                           AS opd_incomplete_pct,
            COUNT(*)                                                AS opd_v2_n,
            ROUND(100.0 * COUNT_IF(m.pathway_complete = 1)
                / NULLIF(COUNT(*), 0), 1)                           AS received_care_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey  j
        JOIN HOSPITALS.REPORTING.mart_pathway_analysis      m ON j.visit_id = m.visit_id
        WHERE j.source_system = 'EMR_V2'
          AND j.visit_type   <> 'Inpatient'
          AND j.arrival_ts   IS NOT NULL
    """)


def q_dropoff_sankey():
    """Patient pathway flow edge counts for Sankey. V2 OPD only (arrival_ts IS NOT NULL).
    Returns a single row — Python transforms into Sankey source/target/value arrays.
    7 nodes: Arrival → Consult | No Consult → Ancillary | Direct→Admitted | Direct→Theatre | OPD Exit
    → Admission | Theatre | OPD Exit."""
    return run_query_df("""
        SELECT
            COUNT(*)                                                            AS total_arrivals,
            COUNT_IF(cons_ts IS NOT NULL)                                       AS arrived_to_consult,
            COUNT_IF(cons_ts IS NULL)                                           AS arrived_no_consult,
            COUNT_IF(cons_ts IS NOT NULL
                AND (had_lab_order = 1 OR had_imaging = 1))                    AS consult_to_ancillary,
            COUNT_IF(cons_ts IS NOT NULL
                AND had_lab_order = 0 AND had_imaging = 0
                AND had_admission = 1 AND had_theatre = 0)                     AS consult_direct_admission,
            COUNT_IF(cons_ts IS NOT NULL
                AND had_lab_order = 0 AND had_imaging = 0
                AND had_theatre = 1)                                            AS consult_direct_theatre,
            COUNT_IF(cons_ts IS NOT NULL
                AND had_lab_order = 0 AND had_imaging = 0
                AND had_admission = 0 AND had_theatre = 0)                     AS consult_to_opd_exit,
            COUNT_IF(cons_ts IS NOT NULL AND (had_lab_order = 1 OR had_imaging = 1)
                AND had_admission = 1 AND had_theatre = 0)                     AS ancillary_to_admission,
            COUNT_IF(cons_ts IS NOT NULL AND (had_lab_order = 1 OR had_imaging = 1)
                AND had_theatre = 1)                                            AS ancillary_to_theatre,
            COUNT_IF(cons_ts IS NOT NULL AND (had_lab_order = 1 OR had_imaging = 1)
                AND had_admission = 0 AND had_theatre = 0)                     AS ancillary_to_opd_exit
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE source_system = 'EMR_V2'
          AND arrival_ts IS NOT NULL
    """)


def q_dropoff_by_disposition():
    """Incomplete care rate by disposition. V2 only. Includes only meaningful dispositions
    (excludes OPD→Admitted, OPD→Theatre — always pathway_complete by construction)."""
    return run_query_df("""
        SELECT
            disposition,
            COUNT(*) AS total_v2,
            COUNT_IF(incomplete_care = 1) AS incomplete_n,
            ROUND(100.0 * COUNT_IF(incomplete_care = 1) / NULLIF(COUNT(*), 0), 1) AS incomplete_pct,
            ROUND(100.0 * COUNT_IF(pathway_complete = 1) / NULLIF(COUNT(*), 0), 1) AS complete_pct
        FROM HOSPITALS.REPORTING.mart_pathway_analysis
        WHERE source_system = 'EMR_V2'
          AND disposition IN ('OPD', 'Inpatient-Only')
        GROUP BY 1
        ORDER BY incomplete_pct DESC NULLS LAST
    """)


def q_dropoff_conversion_monthly():
    """Monthly OPD→Admission conversion rate. V2 only.
    Denominator = OPD arrivals per month. Numerator = had_admission = TRUE (spine, Issue 84 fixed)."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', visit_date)                                     AS visit_month,
            COUNT(*)                                                            AS opd_visits,
            COUNT_IF(had_admission = TRUE)                                      AS admitted_from_opd,
            ROUND(100.0 * COUNT_IF(had_admission = TRUE)
                / NULLIF(COUNT(*), 0), 1)                                       AS conversion_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE source_system = 'EMR_V2'
          AND visit_type   <> 'Inpatient'
          AND arrival_ts IS NOT NULL
          AND visit_date >= DATEADD('month', -18, CURRENT_DATE)
          AND visit_date <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 1
        ORDER BY 1
    """)


def q_dropoff_funnel():
    """OPD→Admission→Theatre funnel counts. V2 only. Single row.
    had_admission / had_theatre populated directly on OPD rows (spine Issue 84 fixed)."""
    return run_query_df("""
        SELECT
            COUNT(*)                                AS total_arrivals,
            COUNT_IF(cons_ts IS NOT NULL)           AS arrived_to_consult,
            COUNT_IF(cons_ts IS NULL)               AS arrived_no_consult,
            COUNT_IF(had_admission = TRUE)          AS opd_to_admission,
            COUNT_IF(had_theatre  = TRUE)           AS opd_to_theatre
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE source_system = 'EMR_V2'
          AND visit_type   <> 'Inpatient'
          AND arrival_ts IS NOT NULL
    """)


def q_dropoff_sankey_v2():
    """Clinical pathway Sankey with pharmacy node. V2 OPD only (arrival_ts IS NOT NULL).
    Joins spine (cons_ts) + mart (pathway_complete, had_pharmacy, had_admission, had_theatre).
    Mutually exclusive consult splits — priority: theatre > admitted > pharmacy > OPD exit."""
    return run_query_df("""
        WITH base AS (
            SELECT
                j.cons_ts,
                j.had_admission,
                j.had_theatre,
                m.pathway_complete,
                m.had_pharmacy
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey  j
            JOIN HOSPITALS.REPORTING.mart_pathway_analysis       m ON j.visit_id = m.visit_id
            WHERE j.source_system = 'EMR_V2'
              AND j.visit_type   <> 'Inpatient'
              AND j.arrival_ts   IS NOT NULL
        )
        SELECT
            COUNT(*)                                                                        AS total,
            COUNT_IF(cons_ts IS NOT NULL)                                                   AS to_consult,
            COUNT_IF(cons_ts IS NULL AND pathway_complete = 1)                             AS to_ancillary,
            COUNT_IF(pathway_complete = 0)                                                  AS to_dropoff,
            -- Consult → theatre direct (not via admission)
            COUNT_IF(cons_ts IS NOT NULL AND had_theatre = TRUE
                     AND had_admission = FALSE)                                             AS consult_theatre_direct,
            -- Consult → admitted (priority over pharmacy)
            COUNT_IF(cons_ts IS NOT NULL AND had_admission = TRUE)                         AS consult_admitted,
            -- Consult → pharmacy (not admitted, not theatre)
            COUNT_IF(cons_ts IS NOT NULL AND had_pharmacy = TRUE
                     AND had_admission = FALSE AND had_theatre = FALSE)                     AS consult_pharmacy,
            -- Consult → OPD exit (no pharmacy, not admitted, not theatre)
            COUNT_IF(cons_ts IS NOT NULL AND had_pharmacy = FALSE
                     AND had_admission = FALSE AND had_theatre = FALSE)                     AS consult_opd_exit,
            -- Ancillary-only → admitted
            COUNT_IF(cons_ts IS NULL AND pathway_complete = 1
                     AND had_admission = TRUE)                                              AS ancillary_admitted,
            -- Ancillary-only → exit
            COUNT_IF(cons_ts IS NULL AND pathway_complete = 1
                     AND had_admission = FALSE)                                             AS ancillary_exit,
            -- Admitted → theatre
            COUNT_IF(had_admission = TRUE AND had_theatre = TRUE)                          AS admitted_theatre,
            -- Admitted → ward discharge
            COUNT_IF(had_admission = TRUE AND had_theatre = FALSE)                         AS admitted_discharge
        FROM base
    """)


def q_dropoff_stage_responsibility():
    """Drop-off stage distribution for incomplete V2 visits. Excludes 'unknown'.
    Used for Stage Accountability table — owner mapping applied in Python."""
    return run_query_df("""
        SELECT
            drop_off_stage,
            COUNT(*)                                                        AS drop_off_n,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)             AS drop_off_pct
        FROM HOSPITALS.REPORTING.mart_pathway_analysis
        WHERE source_system = 'EMR_V2'
          AND incomplete_care = 1
          AND drop_off_stage IS NOT NULL
          AND drop_off_stage <> 'unknown'
        GROUP BY 1
        ORDER BY drop_off_n DESC
    """)


def q_dropoff_dept_breakdown():
    """Post-registration ghost visits by registered department.
    Excludes WALK-IN (registration channel, not a clinical destination).
    Top 5 departments by volume. Pct renormalised over non-Walk-In ghosts only."""
    return run_query_df("""
        SELECT
            pj.dept                                                         AS dept,
            COUNT(*)                                                        AS n,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)             AS pct
        FROM HOSPITALS.REPORTING.mart_pathway_analysis       m
        JOIN HOSPITALS.REPORTING.rpt_ortho_patient_journey   pj
            ON pj.visit_id = m.visit_id
        WHERE m.source_system   = 'EMR_V2'
          AND m.incomplete_care = 1
          AND m.drop_off_stage  = 'post-registration'
          AND pj.dept           IS NOT NULL
          AND pj.dept           != 'WALK-IN'
        GROUP BY 1
        ORDER BY n DESC
        LIMIT 5
    """)


# ── Drop-off Why ─────────────────────────────────────────────────────────────

def q_dropoff_hour_of_day():
    """Drop-off counts by hour of arrival — previous complete calendar month.
    post-registration and post-triage only. Pivoted: one row per hour."""
    return run_query_df("""
        SELECT
            HOUR(pj.arrival_ts)                                             AS arrival_hour,
            COUNT_IF(m.drop_off_stage = 'post-registration')               AS ghost_n,
            COUNT_IF(m.drop_off_stage = 'post-triage')                     AS post_triage_n
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        JOIN HOSPITALS.REPORTING.mart_pathway_analysis m ON pj.visit_id = m.visit_id
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_type   <> 'Inpatient'
          AND pj.arrival_ts   IS NOT NULL
          AND m.drop_off_stage IN ('post-registration', 'post-triage')
          AND pj.visit_date >= DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE))
          AND pj.visit_date <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 1
        ORDER BY 1
    """)


def q_dropoff_monthly_trend():
    """Monthly ghost rate trend. OPD V2 only. Excludes current incomplete month."""
    return run_query_df("""
        SELECT
            DATE_TRUNC('month', visit_date)                                 AS visit_month,
            COUNT(*)                                                        AS total_opd,
            COUNT_IF(drop_off_stage = 'post-registration')                 AS ghost_n,
            COUNT_IF(drop_off_stage = 'post-triage')                       AS post_triage_n,
            ROUND(100.0 * COUNT_IF(drop_off_stage = 'post-registration')
                / NULLIF(COUNT(*), 0), 2)                                   AS ghost_pct,
            ROUND(100.0 * COUNT_IF(incomplete_care = 1)
                / NULLIF(COUNT(*), 0), 2)                                   AS incomplete_pct
        FROM HOSPITALS.REPORTING.mart_pathway_analysis
        WHERE source_system = 'EMR_V2'
          AND disposition    = 'OPD'
          AND visit_date     <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 1
        ORDER BY 1
    """)


def q_dropoff_dept_tat():
    """Arrival-to-consult TAT + incomplete rate by department — previous complete month.
    Only depts with ≥5 timed consults. Sorted by P50 wait DESC.
    Dual signal: bar length = wait time, incomplete_pct shows who's losing patients."""
    return run_query_df("""
        SELECT
            pj.dept,
            COUNT(*)                                                            AS total_visits,
            COUNT_IF(
                pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
            )                                                                   AS valid_n,
            ROUND(MEDIAN(
                CASE WHEN pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                     AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
            ), 0)                                                               AS p50_wait_mins,
            ROUND(PERCENTILE_CONT(0.9) WITHIN GROUP (
                ORDER BY
                CASE WHEN pj.cons_ts IS NOT NULL AND pj.arrival_ts IS NOT NULL
                     AND DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) BETWEEN 1 AND 480
                THEN DATEDIFF('minute', pj.arrival_ts, pj.cons_ts) END
            ), 0)                                                               AS p90_wait_mins,
            COUNT_IF(m.incomplete_care = 1)                                    AS incomplete_n,
            ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
                / NULLIF(COUNT(*), 0), 1)                                       AS incomplete_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        JOIN HOSPITALS.REPORTING.mart_pathway_analysis m
            ON pj.visit_id = m.visit_id
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_type   <> 'Inpatient'
          AND pj.arrival_ts   IS NOT NULL
          AND pj.dept         IS NOT NULL
          AND pj.visit_date   >= DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE))
          AND pj.visit_date   <  DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 1
        HAVING valid_n >= 5
        ORDER BY p50_wait_mins DESC NULLS LAST
    """)


def q_dropoff_service_line():
    """Incomplete rate by clinical service line — last complete data month.
    Walk-In excluded (registration channel, not a clinical destination).
    Partial-month guard: if MAX(visit_date) day < 25, steps back one month.
    Minimum 10 total visits. Sorted by incomplete_pct DESC."""
    return run_query_df("""
        WITH data_max AS (
            SELECT
                DATE_TRUNC('month', MAX(pj2.visit_date)) AS max_month,
                DAY(MAX(pj2.visit_date))                 AS max_day
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj2
            WHERE pj2.source_system = 'EMR_V2'
              AND pj2.visit_type   <> 'Inpatient'
        ),
        ref AS (
            SELECT
                CASE
                    WHEN max_day < 25 THEN DATEADD('month', -1, max_month)
                    ELSE max_month
                END AS ref_start
            FROM data_max
        )
        SELECT
            pj.dept,
            DATE_TRUNC('month', MIN(pj.visit_date))                             AS ref_month,
            COUNT(*)                                                            AS total_visits,
            COUNT_IF(m.incomplete_care = 1)                                     AS incomplete_n,
            ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
                / NULLIF(COUNT(*), 0), 1)                                       AS incomplete_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        JOIN HOSPITALS.REPORTING.mart_pathway_analysis m
            ON pj.visit_id = m.visit_id
        CROSS JOIN ref
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_type   <> 'Inpatient'
          AND pj.arrival_ts   IS NOT NULL
          AND pj.dept         IS NOT NULL
          AND UPPER(pj.dept)  <> 'WALK-IN'
          AND pj.visit_date   >= ref.ref_start
          AND pj.visit_date   <  DATEADD('month', 1, ref.ref_start)
        GROUP BY 1
        HAVING total_visits >= 10
        ORDER BY incomplete_pct DESC NULLS LAST
    """)


def q_dropoff_factor_breakdown():
    """Dynamic factor analysis: auto-detects top 2 service lines by incomplete_pct
    over stable baseline (March 2025 – second-most-recent complete month, min 50 visits).
    Returns incomplete_pct by dept_group × time band.
    dept_group values: top-1 dept name, top-2 dept name, 'All Other'.
    Used to determine whether service line, time-of-day, or their interaction is the
    dominant measurable factor — no dept names are hardcoded."""
    return run_query_df("""
        WITH dept_ranks AS (
            SELECT
                pj.dept,
                ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
                    / NULLIF(COUNT(*), 0), 1)                           AS dept_incomplete_pct,
                ROW_NUMBER() OVER (
                    ORDER BY COUNT_IF(m.incomplete_care = 1) * 1.0
                             / NULLIF(COUNT(*), 0) DESC NULLS LAST)     AS dept_rank
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey  pj
            JOIN HOSPITALS.REPORTING.mart_pathway_analysis      m
                ON m.visit_id  = pj.visit_id
            WHERE pj.source_system = 'EMR_V2'
              AND pj.visit_type   <> 'Inpatient'
              AND pj.arrival_ts   IS NOT NULL
              AND pj.dept          IS NOT NULL
              AND UPPER(pj.dept)  <> 'WALK-IN'
              AND pj.visit_date   >= '2025-03-01'
              AND pj.visit_date   <  DATEADD('month', -1, DATE_TRUNC('month', CURRENT_DATE))
            GROUP BY 1
            HAVING COUNT(*) >= 50
            QUALIFY dept_rank <= 2
        ),
        base AS (
            SELECT
                COALESCE(dr.dept, 'All Other')                          AS dept_group,
                CASE
                    WHEN HOUR(pj.arrival_ts) BETWEEN 6  AND 11
                        THEN '1 Morning (06-12)'
                    WHEN HOUR(pj.arrival_ts) BETWEEN 12 AND 14
                        THEN '2 Midday (12-15)'
                    WHEN HOUR(pj.arrival_ts) BETWEEN 15 AND 18
                        THEN '3 Afternoon (15-19)'
                    ELSE '4 Other'
                END                                                     AS time_band,
                m.incomplete_care
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey  pj
            JOIN HOSPITALS.REPORTING.mart_pathway_analysis      m
                ON m.visit_id  = pj.visit_id
            LEFT JOIN dept_ranks                                    dr
                ON dr.dept     = pj.dept
            WHERE pj.source_system = 'EMR_V2'
              AND pj.visit_type   <> 'Inpatient'
              AND pj.arrival_ts   IS NOT NULL
              AND pj.dept          IS NOT NULL
              AND UPPER(pj.dept)  <> 'WALK-IN'
              AND HOUR(pj.arrival_ts) BETWEEN 6 AND 18
              AND pj.visit_date   >= '2025-03-01'
              AND pj.visit_date   <  DATEADD('month', -1, DATE_TRUNC('month', CURRENT_DATE))
        )
        SELECT
            dept_group,
            time_band,
            COUNT(*)                                                    AS total_visits,
            COUNT_IF(incomplete_care = 1)                               AS incomplete_n,
            ROUND(100.0 * COUNT_IF(incomplete_care = 1)
                / NULLIF(COUNT(*), 0), 1)                               AS incomplete_pct
        FROM base
        GROUP BY 1, 2
        ORDER BY 1, 2
    """)


def q_dropoff_ghost_stage():
    """Stage of exit for incomplete visits — March 2025+ baseline, top clinical depts.
    Pure ghost = arrived, no triage, no care signal.
    Post-triage ghost = triaged but no care recorded.
    Walk-In excluded. Min 20 incomplete visits."""
    return run_query_df("""
        SELECT
            pj.dept,
            COUNT(*)                                                                AS incomplete_n,
            ROUND(100.0 * COUNT_IF(NOT pj.had_triage
                         AND NOT pj.had_service_charge
                         AND NOT pj.had_clinical_notes)
                / NULLIF(COUNT(*), 0), 1)                                           AS pure_ghost_pct,
            ROUND(100.0 * COUNT_IF(pj.had_triage
                         AND NOT pj.had_service_charge
                         AND NOT pj.had_clinical_notes)
                / NULLIF(COUNT(*), 0), 1)                                           AS post_triage_ghost_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        JOIN HOSPITALS.REPORTING.mart_pathway_analysis m
            ON pj.visit_id = m.visit_id
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_type   <> 'Inpatient'
          AND pj.arrival_ts   IS NOT NULL
          AND pj.dept         IS NOT NULL
          AND UPPER(pj.dept)  <> 'WALK-IN'
          AND m.incomplete_care = 1
          AND pj.visit_date   >= '2025-03-01'
        GROUP BY 1
        HAVING incomplete_n >= 20
        ORDER BY incomplete_n DESC
    """)


def q_dropoff_dept_hourly():
    """Arrival-hour incomplete rate by dept — March 2025+.
    Returns all depts (Walk-In excluded) with ≥30 total visits per hour.
    Used to show Physio afternoon gradient and Pharmacy late-evening spike (Inv 154)."""
    return run_query_df("""
        SELECT
            pj.dept,
            HOUR(pj.arrival_ts)                                                     AS arrival_hour,
            COUNT(*)                                                                AS total_n,
            COUNT_IF(m.incomplete_care = 1)                                         AS incomplete_n,
            ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
                / NULLIF(COUNT(*), 0), 1)                                           AS incomplete_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey pj
        JOIN HOSPITALS.REPORTING.mart_pathway_analysis m
            ON pj.visit_id = m.visit_id
        WHERE pj.source_system = 'EMR_V2'
          AND pj.visit_type   <> 'Inpatient'
          AND pj.arrival_ts   IS NOT NULL
          AND pj.dept         IS NOT NULL
          AND UPPER(pj.dept)  <> 'WALK-IN'
          AND pj.visit_date   >= '2025-03-01'
        GROUP BY 1, 2
        HAVING total_n >= 30
        ORDER BY 1, 2
    """)


def q_dropoff_volume_corr():
    """Weekly OPD volume vs incomplete rate Pearson correlation — stable baseline
    March 2025 – second-most-recent complete month.
    Tests capacity pressure hypothesis: does high volume drive higher incomplete rates?
    Returns single row: corr coefficient, weeks_n, avg_weekly_visits, avg_incomplete_pct.
    r > 0.5 = volume explanatory; |r| < 0.3 = volume not the factor."""
    return run_query_df("""
        WITH weekly AS (
            SELECT
                DATE_TRUNC('week', pj.visit_date)                       AS visit_week,
                COUNT(*)                                                 AS total_visits,
                ROUND(100.0 * COUNT_IF(m.incomplete_care = 1)
                    / NULLIF(COUNT(*), 0), 4)                            AS incomplete_pct
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey  pj
            JOIN HOSPITALS.REPORTING.mart_pathway_analysis      m
                ON m.visit_id = pj.visit_id
            WHERE pj.source_system = 'EMR_V2'
              AND pj.visit_type   <> 'Inpatient'
              AND pj.arrival_ts   IS NOT NULL
              AND pj.visit_date   >= '2025-03-01'
              AND pj.visit_date   <  DATEADD('month', -1, DATE_TRUNC('month', CURRENT_DATE))
            GROUP BY 1
        )
        SELECT
            ROUND(CORR(total_visits, incomplete_pct), 3)                AS volume_incomplete_corr,
            COUNT(*)                                                     AS weeks_n,
            ROUND(AVG(total_visits), 0)                                  AS avg_weekly_visits,
            ROUND(AVG(incomplete_pct), 2)                                AS avg_incomplete_pct
        FROM weekly
    """)


# ── Capacity Pressure ────────────────────────────────────────────────────────

def q_capacity_snapshot():
    """Latest day's operational pressure. Single row.
    Ward / Theatre / Lab: current load, 14-day baseline, Δ% from baseline."""
    return run_query_df("""
        SELECT
            period_date,
            total_census                                                        AS ward_current,
            ROUND(census_14d_avg,   1)                                          AS ward_baseline,
            ROUND(100.0 * (total_census   - census_14d_avg)
                / NULLIF(census_14d_avg,   0), 1)                               AS ward_delta_pct,
            theatre_cases                                                       AS theatre_current,
            ROUND(theatre_14d_avg,  1)                                          AS theatre_baseline,
            ROUND(100.0 * (theatre_cases  - theatre_14d_avg)
                / NULLIF(theatre_14d_avg,  0), 1)                               AS theatre_delta_pct,
            lab_orders                                                          AS lab_current,
            ROUND(lab_14d_avg,      1)                                          AS lab_baseline,
            ROUND(100.0 * (lab_orders     - lab_14d_avg)
                / NULLIF(lab_14d_avg,      0), 1)                               AS lab_delta_pct
        FROM HOSPITALS.REPORTING.mart_capacity
        WHERE period_date = (SELECT MAX(period_date) FROM HOSPITALS.REPORTING.mart_capacity)
    """)


def q_capacity_trend():
    """Daily Δ% from 14-day baseline — Ward, Theatre, Lab. Last 90 days.
    Raw current + baseline values included for hover tooltips."""
    return run_query_df("""
        SELECT
            period_date,
            total_census                                                        AS ward_current,
            ROUND(census_14d_avg,   1)                                          AS ward_baseline,
            ROUND(100.0 * (total_census   - census_14d_avg)
                / NULLIF(census_14d_avg,   0), 1)                               AS ward_delta_pct,
            theatre_cases                                                       AS theatre_current,
            ROUND(theatre_14d_avg,  1)                                          AS theatre_baseline,
            ROUND(100.0 * (theatre_cases  - theatre_14d_avg)
                / NULLIF(theatre_14d_avg,  0), 1)                               AS theatre_delta_pct,
            lab_orders                                                          AS lab_current,
            ROUND(lab_14d_avg,      1)                                          AS lab_baseline,
            ROUND(100.0 * (lab_orders     - lab_14d_avg)
                / NULLIF(lab_14d_avg,      0), 1)                               AS lab_delta_pct
        FROM HOSPITALS.REPORTING.mart_capacity
        WHERE period_date >= DATEADD('day', -90, CURRENT_DATE)
          AND period_date <  CURRENT_DATE
        ORDER BY period_date
    """)


# ── V2 OPD — Live Operations ─────────────────────────────────────────────────

def q_v2_opd_summary():
    """V2 OPD top-level KPIs. Visit mix % uses steady-state period (Aug 2025+)
    to exclude ramp-up artefact (Feb–Jul 2025: catch-up revisit registrations from old system)."""
    return run_query_df(f"""
        SELECT
            COUNT(*)                                                          AS total_visits,
            MIN(visit_date)                                                   AS data_from,
            MAX(visit_date)                                                   AS data_to,
            ROUND(100.0 * COUNT_IF(visit_type = 'New Patient'
                AND visit_date >= '2025-08-01')
                / NULLIF(COUNT_IF(visit_date >= '2025-08-01'), 0), 1)        AS ss_new_pct,
            ROUND(100.0 * COUNT_IF(visit_type = 'Revisit'
                AND visit_date >= '2025-08-01')
                / NULLIF(COUNT_IF(visit_date >= '2025-08-01'), 0), 1)        AS ss_revisit_pct,
            ROUND(100.0 * COUNT_IF(visit_type = 'Walk-In'
                AND visit_date >= '2025-08-01')
                / NULLIF(COUNT_IF(visit_date >= '2025-08-01'), 0), 1)        AS ss_walk_in_pct,
            ROUND(100.0 * COUNT_IF(had_admission = TRUE) / COUNT(*), 1)      AS conversion_pct,
            ROUND(100.0 * COUNT_IF(cons_ts IS NOT NULL) / COUNT(*), 1)       AS cons_ts_coverage_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE {_V2_OPD}
    """)


def q_v2_opd_visit_mix_monthly():
    """V2 OPD visit type mix by month — New Patient / Revisit / Walk-In.
    is_ramp_period = TRUE for Feb–Jul 2025 (catch-up registrations from old system inflate revisit share).
    Steady state from Aug 2025: New ~37%, Revisit ~40%, Walk-In ~22%.
    Excludes current partial month."""
    return run_query_df(f"""
        SELECT
            DATE_TRUNC('month', visit_date)                              AS visit_month,
            COUNT(*)                                                     AS total,
            COUNT_IF(visit_type = 'New Patient')                         AS new_patient,
            COUNT_IF(visit_type = 'Revisit')                             AS revisit,
            COUNT_IF(visit_type = 'Walk-In')                             AS walk_in,
            ROUND(100.0 * COUNT_IF(visit_type = 'New Patient') / COUNT(*), 1) AS new_pct,
            ROUND(100.0 * COUNT_IF(visit_type = 'Revisit')     / COUNT(*), 1) AS revisit_pct,
            ROUND(100.0 * COUNT_IF(visit_type = 'Walk-In')     / COUNT(*), 1) AS walk_in_pct,
            CASE WHEN DATE_TRUNC('month', visit_date) < '2025-08-01'
                 THEN TRUE ELSE FALSE END                                AS is_ramp_period
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE {_V2_OPD}
          AND visit_date < DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 1
        ORDER BY 1
    """)


def q_v2_opd_consult_tat_monthly():
    """V2 OPD arrival→consult completion: monthly P50, P75, coverage %.
    Uses CTE pre-filter — PERCENTILE_CONT does not support FILTER clause in Snowflake.
    WARNING (Issue 86): coverage drops 60%→33% in Feb–May 2026; TAT drop in that window
    is a documentation artefact (survivorship bias). low_coverage = TRUE when coverage < 50%.
    Excludes current partial month."""
    return run_query_df(f"""
        WITH tat_valid AS (
            SELECT
                DATE_TRUNC('month', visit_date)         AS visit_month,
                DATEDIFF('minute', arrival_ts, cons_ts) AS tat_mins
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE {_V2_OPD}
              AND arrival_ts IS NOT NULL
              AND cons_ts    IS NOT NULL
              AND cons_ts     > arrival_ts
              AND visit_date  < DATE_TRUNC('month', CURRENT_DATE)
        ),
        monthly_totals AS (
            SELECT
                DATE_TRUNC('month', visit_date)         AS visit_month,
                COUNT(*)                                AS opd_visits,
                COUNT_IF(cons_ts IS NOT NULL)           AS has_cons_ts
            FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
            WHERE {_V2_OPD}
              AND visit_date < DATE_TRUNC('month', CURRENT_DATE)
            GROUP BY 1
        )
        SELECT
            m.visit_month,
            m.opd_visits,
            m.has_cons_ts,
            ROUND(100.0 * m.has_cons_ts / NULLIF(m.opd_visits, 0), 1)  AS coverage_pct,
            CASE WHEN ROUND(100.0 * m.has_cons_ts / NULLIF(m.opd_visits, 0), 1) < 50
                 THEN TRUE ELSE FALSE END                                AS low_coverage,
            MEDIAN(v.tat_mins)                                          AS p50_mins,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY v.tat_mins)    AS p75_mins
        FROM monthly_totals m
        LEFT JOIN tat_valid v ON v.visit_month = m.visit_month
        GROUP BY m.visit_month, m.opd_visits, m.has_cons_ts
        ORDER BY 1
    """)


def q_v2_opd_arrival_hour():
    """V2 OPD arrivals and consult completion rate by hour of day.
    V2 peak window: 10:00–11:00 (differs from V1 07:00–08:59 — INV-V2-OPD-4).
    Consult rate falls at peak: 65% at 07:00 → 53% at 11:00 → 44% at 14:00–17:00."""
    return run_query_df(f"""
        SELECT
            HOUR(arrival_ts)                                             AS hour_of_day,
            COUNT(*)                                                     AS arrivals,
            COUNT_IF(cons_ts IS NOT NULL)                                AS reached_consult,
            ROUND(100.0 * COUNT_IF(cons_ts IS NOT NULL) / COUNT(*), 1)  AS consult_rate_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE {_V2_OPD}
          AND arrival_ts IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)


def q_v2_opd_conversion_monthly():
    """V2 cross-visit OPD→Admission conversion rate by month.
    had_admission uses xv_inpatient CTE (within 30 days, Issue 84 fix).
    Steady-state band: 2.4–3.3% from Aug 2025.
    is_ramp_period = TRUE for Feb–Jul 2025. Excludes current partial month."""
    return run_query_df(f"""
        SELECT
            DATE_TRUNC('month', visit_date)                              AS visit_month,
            COUNT(*)                                                     AS opd_visits,
            COUNT_IF(had_admission = TRUE)                               AS converted,
            ROUND(100.0 * COUNT_IF(had_admission = TRUE) / COUNT(*), 1) AS conversion_pct,
            CASE WHEN DATE_TRUNC('month', visit_date) < '2025-08-01'
                 THEN TRUE ELSE FALSE END                                AS is_ramp_period
        FROM HOSPITALS.REPORTING.rpt_ortho_patient_journey
        WHERE {_V2_OPD}
          AND visit_date < DATE_TRUNC('month', CURRENT_DATE)
        GROUP BY 1
        ORDER BY 1
    """)


# ── Command Center ────────────────────────────────────────────────────────────

def q_cc_pipeline():
    """Command Center pipeline: 7-day avg per stage vs prior 21-day baseline.
    Anchor = MAX(period_date) — works regardless of ingestion lag relative to today.
    Sources mart_operational_kpis (arrivals/consults/ancillary) + mart_capacity (ward/theatre).
    Single row."""
    return run_query_df("""
        WITH latest AS (
            SELECT MAX(period_date) AS d FROM HOSPITALS.REPORTING.mart_operational_kpis
        ),
        kpi AS (
            SELECT period_date, opd_visits, cons_n,
                COALESCE(lab_n, 0) + COALESCE(imaging_n, 0) AS ancillary_n
            FROM HOSPITALS.REPORTING.mart_operational_kpis
            WHERE period_date >= (SELECT DATEADD('day', -28, d) FROM latest)
              AND period_date <= (SELECT d FROM latest)
        ),
        cap AS (
            SELECT period_date, total_census, theatre_cases
            FROM HOSPITALS.REPORTING.mart_capacity
            WHERE period_date >= (SELECT DATEADD('day', -28, d) FROM latest)
              AND period_date <= (SELECT d FROM latest)
        )
        SELECT
            (SELECT d FROM latest)                                                                                   AS anchor_date,
            ROUND(AVG(CASE WHEN k.period_date >= (SELECT DATEADD('day',-7,d) FROM latest) THEN k.opd_visits    END),0) AS arr_7d,
            ROUND(AVG(CASE WHEN k.period_date <  (SELECT DATEADD('day',-7,d) FROM latest) THEN k.opd_visits    END),0) AS arr_base,
            ROUND(AVG(CASE WHEN k.period_date >= (SELECT DATEADD('day',-7,d) FROM latest) THEN k.cons_n        END),0) AS cons_7d,
            ROUND(AVG(CASE WHEN k.period_date <  (SELECT DATEADD('day',-7,d) FROM latest) THEN k.cons_n        END),0) AS cons_base,
            ROUND(AVG(CASE WHEN k.period_date >= (SELECT DATEADD('day',-7,d) FROM latest) THEN k.ancillary_n   END),0) AS anc_7d,
            ROUND(AVG(CASE WHEN k.period_date <  (SELECT DATEADD('day',-7,d) FROM latest) THEN k.ancillary_n   END),0) AS anc_base,
            ROUND(AVG(CASE WHEN c.period_date >= (SELECT DATEADD('day',-7,d) FROM latest) THEN c.total_census  END),1) AS ward_7d,
            ROUND(AVG(CASE WHEN c.period_date <  (SELECT DATEADD('day',-7,d) FROM latest) THEN c.total_census  END),1) AS ward_base,
            ROUND(AVG(CASE WHEN c.period_date >= (SELECT DATEADD('day',-7,d) FROM latest) THEN c.theatre_cases END),0) AS theatre_7d,
            ROUND(AVG(CASE WHEN c.period_date <  (SELECT DATEADD('day',-7,d) FROM latest) THEN c.theatre_cases END),0) AS theatre_base
        FROM kpi k
        LEFT JOIN cap c ON k.period_date = c.period_date
    """)


def q_cc_pharm_dispensing():
    """V2 pharmacy dispensing P50 for Home pulse card.
    Uses rpt_ortho_pharmacy (request_stamp → dispensed_stamp) — the real dispensing interval.
    pharm_tat_p50_mins in mart = consult→queue (~2-4 min) — wrong metric for this card (Issue 91)."""
    return run_query_df("""
        SELECT
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tat_mins), 0) AS p50_dispensing_mins,
            ROUND(COUNT_IF(tat_mins IS NOT NULL AND tat_mins < 240) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                  AS coverage_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
        WHERE source_system = 'EMR_V2'
          AND tat_mins IS NOT NULL
          AND tat_mins < 240
    """)


def q_cc_pharm_fulfillment():
    """V2 pharmacy fulfillment KPI for the Home flow node.
    Uses the same status=2 definition, numerator, denominator, and monthly comparison
    as the Pharmacy page's V2 Fulfillment Rate card."""
    return run_query_df("""
        WITH v2_orders AS (
            SELECT
                order_date,
                COUNT(*) AS total_orders,
                COUNT_IF(is_served = TRUE) AS dispensed
            FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
            WHERE source_system = 'EMR_V2'
            GROUP BY 1
        ),
        monthly AS (
            SELECT
                DATE_TRUNC('month', order_date) AS order_month,
                ROUND(COUNT_IF(is_served = TRUE) * 100.0 / NULLIF(COUNT(*), 0), 1)
                    AS fulfillment_rate
            FROM HOSPITALS.REPORTING.rpt_ortho_pharmacy
            WHERE source_system = 'EMR_V2'
              AND order_date IS NOT NULL
            GROUP BY 1
        ),
        current_month AS (
            SELECT fulfillment_rate
            FROM monthly
            QUALIFY ROW_NUMBER() OVER (ORDER BY order_month DESC) = 1
        ),
        prior_month AS (
            SELECT fulfillment_rate
            FROM monthly
            QUALIFY ROW_NUMBER() OVER (ORDER BY order_month DESC) = 2
        )
        SELECT
            ROUND(SUM(dispensed) * 100.0 / NULLIF(SUM(total_orders), 0), 1) AS fulfillment_rate,
            SUM(dispensed) AS dispensed_orders,
            SUM(total_orders) AS total_orders,
            MIN(order_date) AS data_from,
            MAX(order_date) AS data_to,
            (SELECT fulfillment_rate FROM current_month)
              - (SELECT fulfillment_rate FROM prior_month) AS mom_delta_pp
        FROM v2_orders
    """)


def q_cc_discharge_los():
    """V2 median LOS for Home Discharge flow node. V2 only, plausible discharges."""
    return run_query_df(f"""
        SELECT
            ROUND(MEDIAN(los_days), 1)                                      AS median_los_days,
            ROUND(COUNT_IF(is_discharge_reliable = TRUE) * 100.0
                  / NULLIF(COUNT(*), 0), 1)                                 AS discharge_coverage_pct
        FROM HOSPITALS.REPORTING.rpt_ortho_admissions
        WHERE {_V2}
          AND is_discharge_reliable = TRUE
          AND is_los_plausible      = TRUE
    """)


def q_cc_lab_completion():
    """V2 lab completion rate for Home Lab flow node and pulse card.
    28-day avg from MAX(request_date) anchor. Uses has_result (status='4' for V2 — Issue 89 fix)."""
    return run_query_df(f"""
        WITH latest AS (
            SELECT MAX(request_date) AS d
            FROM HOSPITALS.REPORTING.rpt_ortho_lab
            WHERE {_V2}
        )
        SELECT
            ROUND(COUNT_IF(has_result) * 100.0 / NULLIF(COUNT(*), 0), 1)   AS completion_pct,
            COUNT(*)                                                         AS total_orders,
            COUNT_IF(has_result)                                             AS completed_orders
        FROM HOSPITALS.REPORTING.rpt_ortho_lab
        WHERE {_V2}
          AND request_date >= (SELECT DATEADD('day', -28, d) FROM latest)
    """)


def q_cc_freshness():
    """Latest period_date in mart_operational_kpis — V2 data age for Command Center header."""
    return run_query_df("""
        SELECT MAX(period_date) AS v2_latest_date
        FROM HOSPITALS.REPORTING.mart_operational_kpis
    """)


# ── Background preload ────────────────────────────────────────────────────────

def preload_all():
    """Fire all queries in parallel to warm the cache. Call once from a background thread."""
    import concurrent.futures

    _all = [
        q_opd_monthly, q_opd_summary, q_opd_dow, q_opd_gender, q_opd_hour,
        q_opd_peak_conversion, q_opd_peak_funnel, q_opd_peak_funnel_by_dow,
        q_opd_tat_by_dow, q_opd_deferred_monthly,
        q_tat_summary, q_tat_distribution, q_tat_nonzero_monthly,
        q_theatre_monthly, q_theatre_summary,
        q_theatre_duration, q_theatre_anaesthesia,
        q_admissions_monthly, q_admissions_summary, q_admissions_ward,
        q_admissions_los_by_ward, q_admissions_gender, q_admissions_dow,
        q_admissions_los_dist, q_admissions_long_stay_trend,
        q_admissions_routing_trend, q_admissions_long_stay_by_routing,
        q_conversion_summary, q_conversion_monthly, q_conversion_lag,
        q_lab_monthly, q_lab_top_tests, q_lab_summary, q_lab_tat_dist, q_lab_bridge,
        q_imaging_monthly, q_imaging_modality_mix, q_imaging_summary,
        q_imaging_modality_completion, q_imaging_completion_monthly,
        q_imaging_tat_dist, q_imaging_bridge,
        q_pharm_dispensing_summary, q_pharm_dispensing_monthly,
        q_pharm_class_breakdown, q_pharm_speed_summary,
        q_pharm_speed_distribution, q_pharm_speed_monthly,
        q_physician_kpis, q_physician_workload, q_physician_efficiency,
        q_physician_trend, q_physician_continuity, q_physician_rank_trend,
        q_busy_dow_summary, q_busy_peak_window,
        q_data_freshness,
        q_leakage_summary, q_leakage_by_procedure, q_leakage_monthly, q_leakage_prev_month,
        q_waiting_rbi_summary, q_waiting_tat_trend, q_waiting_dow_scatter, q_waiting_heatmap,
        q_dropoff_kpis, q_dropoff_sankey_v2, q_dropoff_by_disposition,
        q_dropoff_conversion_monthly, q_dropoff_stage_responsibility,
        q_dropoff_hour_of_day, q_dropoff_monthly_trend, q_dropoff_dept_breakdown,
        q_capacity_snapshot, q_capacity_trend,
        q_v2_opd_summary, q_v2_opd_visit_mix_monthly, q_v2_opd_consult_tat_monthly,
        q_v2_opd_arrival_hour, q_v2_opd_conversion_monthly,
        q_cc_pipeline, q_cc_freshness, q_cc_pharm_dispensing, q_cc_pharm_fulfillment,
        q_cc_lab_completion,
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fn) for fn in _all]
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
