-- DRAFT - NOT EXECUTED.  Counting regression checks for the discovery cut-over (migration step 6).
-- All statements are read-only, except the two dry-run CALLs, which write RUNS rows only (no publish; this is the
-- procedure's existing dry-run behaviour). The CALLs need explicit approval at the time; they read source data.
--
-- Procedure:
--   1. BEFORE cut-over: CALL dry run (A), note its RUN_ID as :before_run.
--   2. Apply diff 05 (deploy through the guard).
--   3. AFTER cut-over, immediately: CALL dry run (B), note its RUN_ID as :after_run.
--   4. Run R1-R7. Any non-empty "expect 0 rows" result = regression; roll back with 99_rollback.sql R2.
-- Running A and B back-to-back minimises genuine source changes between them (views are always recounted).

USE ROLE DATAANALYSTS;

-- (A) / (B) dry runs - require approval when executed
-- CALL HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS();
-- SET before_run = (SELECT MAX_BY(RUN_ID, RUN_AT) FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS);
-- ... deploy diff 05 ...
-- CALL HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS();
-- SET after_run = (SELECT MAX_BY(RUN_ID, RUN_AT) FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS);

-- R1. identical figures and statuses per source (expect 0 rows)
WITH b AS (SELECT * FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS WHERE RUN_ID = $before_run),
     a AS (SELECT * FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS WHERE RUN_ID = $after_run)
SELECT COALESCE(a.SOURCE, b.SOURCE) AS source,
       b.STATUS AS st_before, a.STATUS AS st_after,
       b.ROWS_READ AS rows_before, a.ROWS_READ AS rows_after,
       b.PATIENT_RECORDS AS rec_before, a.PATIENT_RECORDS AS rec_after,
       b.UPDS AS upds_before, a.UPDS AS upds_after,
       b.ROWS_SUPERSEDED AS sup_before, a.ROWS_SUPERSEDED AS sup_after,
       b.ROWS_DELETED AS del_before, a.ROWS_DELETED AS del_after,
       b.RECORDS_NO_UPD_KEY AS nokey_before, a.RECORDS_NO_UPD_KEY AS nokey_after
FROM b FULL OUTER JOIN a ON a.SOURCE = b.SOURCE
WHERE a.SOURCE IS NULL OR b.SOURCE IS NULL
   OR a.STATUS IS DISTINCT FROM b.STATUS
   OR a.ROWS_READ IS DISTINCT FROM b.ROWS_READ OR a.PATIENT_RECORDS IS DISTINCT FROM b.PATIENT_RECORDS
   OR a.UPDS IS DISTINCT FROM b.UPDS OR a.ROWS_SUPERSEDED IS DISTINCT FROM b.ROWS_SUPERSEDED
   OR a.ROWS_DELETED IS DISTINCT FROM b.ROWS_DELETED OR a.RECORDS_NO_UPD_KEY IS DISTINCT FROM b.RECORDS_NO_UPD_KEY;
-- A difference on a VIEW source can be a genuine data change between A and B: re-run A/B once; a persistent
-- difference on a BASE TABLE source with unchanged SOURCE_ALTERED is always a regression.

-- R2. same CONFIG_SIG, and base-table reuse still works (expect 0 rows)
SELECT a.SOURCE, b.CONFIG_SIG AS sig_before, a.CONFIG_SIG AS sig_after, a.SOURCE_ALTERED, a.REUSED
FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS a
JOIN HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS b ON b.SOURCE = a.SOURCE AND b.RUN_ID = $before_run
WHERE a.RUN_ID = $after_run
  AND (a.CONFIG_SIG IS DISTINCT FROM b.CONFIG_SIG
       OR (a.SOURCE_ALTERED IS NOT NULL AND a.SOURCE_ALTERED = b.SOURCE_ALTERED AND NOT a.REUSED));

-- R3. receipts still sum to the published figures (expect 0 rows) - run any time, before and after
SELECT c.SOURCE, c.PATIENT_RECORDS, c.UPDS, x.records, x.upds
FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS c
LEFT JOIN (SELECT SOURCE, SUM(RECORDS_ADDED) AS records, SUM(UPDS_ADDED) AS upds
           FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_RECORD_RECEIPTS GROUP BY 1) x ON x.SOURCE = c.SOURCE
WHERE x.SOURCE IS NULL OR x.records <> c.PATIENT_RECORDS OR x.upds <> c.UPDS;

-- R4. dashboard table equals the latest published run per source (expect 0 rows)
SELECT c.SOURCE, c.PATIENT_RECORDS, c.UPDS, r.PATIENT_RECORDS AS run_records, r.UPDS AS run_upds
FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS c
LEFT JOIN (SELECT SOURCE, PATIENT_RECORDS, UPDS FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS WHERE PUBLISHED
           QUALIFY ROW_NUMBER() OVER (PARTITION BY SOURCE ORDER BY RUN_AT DESC) = 1) r ON r.SOURCE = c.SOURCE
WHERE r.SOURCE IS NULL OR r.PATIENT_RECORDS <> c.PATIENT_RECORDS OR r.UPDS <> c.UPDS;

-- R5. every active source is on the dashboard and nothing else is (expect 0 rows)
SELECT 'active_not_published' AS problem, s.SOURCE FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES s
LEFT JOIN HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS c ON c.SOURCE = s.SOURCE
WHERE s.STATUS = 'active' AND c.SOURCE IS NULL
UNION ALL
SELECT 'published_not_active', c.SOURCE FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS c
LEFT JOIN HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES s ON s.SOURCE = c.SOURCE AND s.STATUS = 'active'
WHERE s.SOURCE IS NULL;

-- R6. registry integrity that the MERGE relies on (expect 0 rows)
SELECT 'duplicate_active_name' AS problem, SOURCE AS item FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
WHERE STATUS = 'active' GROUP BY SOURCE HAVING COUNT(*) > 1
UNION ALL
SELECT 'duplicate_active_object', UPPER(TRIM(SOURCE_OBJECT)) FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
WHERE STATUS = 'active' GROUP BY 2 HAVING COUNT(*) > 1
UNION ALL
SELECT 'non_uppercase_identifier', SOURCE FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
WHERE STATUS = 'active' AND (UPD_KEY <> UPPER(UPD_KEY) OR SOURCE_OBJECT <> UPPER(SOURCE_OBJECT));

-- R7. deployed refresh = repo + diff 05 only. Compare offline: the deployed text below must equal the repo body
--     after applying 05_refresh_discovery_decoupling.diff (same byte-match the deploy guard performs).
SELECT GET_DDL('PROCEDURE', 'HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS(BOOLEAN, BOOLEAN, BOOLEAN)') AS deployed_refresh;
