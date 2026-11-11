-- =============================================================================
-- Warehouse Studio — cost & performance toolkit for {{DB}}.{{SCHEMA}}
-- =============================================================================
--
-- Installs the tuning detector suite (ported from TUNING_LAB.LAB_OWNER and
-- OPTIMIZATIONS.TPC_OWNER) into {{DB}}.{{SCHEMA}}, plus the read-only
-- telemetry views Warehouse Studio's Cost Center and Optimizer pages read.
--
-- Run it as ACCOUNTADMIN (it reads SNOWFLAKE.ACCOUNT_USAGE):
--     python manage.py warehouse_studio_install --env-file /path/to/admin.env
--
-- Design notes
--   * Every procedure is EXECUTE AS OWNER so the app role ({{APP_ROLE}}) can
--     run detectors without being granted ACCOUNT_USAGE itself.  The app role
--     only gets USAGE on these exact procedures and SELECT on these views.
--   * Nothing here ALTERs a warehouse or table.  The lab procedures that do
--     (SP_RECLUSTER_WINDOW, SP_OPTIMIZE_WAREHOUSE_AUTOSUSPEND,
--     SP_OPTIMIZE_TIME_TRAVEL, SP_ENABLE_ATTR_SOS) are deliberately NOT ported
--     with owner's rights: that would hand the app role ACCOUNTADMIN-level
--     DDL.  The Studio shows copy-able fix SQL for a human to run instead.
--   * Every object name is fully qualified: owner's-rights procedures must
--     not depend on the caller's current database/schema.
--   * Tables use CREATE ... IF NOT EXISTS so re-running keeps findings.
--     Everything else is CREATE OR REPLACE, so the script is idempotent.
--   * Detectors de-duplicate, so re-running one never double counts.
--   * Query text in WS_QUERY_HISTORY is redacted for statements that can
--     carry credentials (users, passwords, keys, secrets, tokens).
-- =============================================================================

USE ROLE ACCOUNTADMIN;

-- ---------------------------------------------------------------- storage

CREATE TABLE IF NOT EXISTS {{DB}}.{{SCHEMA}}.TUNING_FINDINGS (
    FINDING_ID      NUMBER AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
    DETECTOR        VARCHAR NOT NULL,
    DETECTED_AT     TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
    SEVERITY        VARCHAR,
    OBJECT_TYPE     VARCHAR,
    OBJECT_NAME     VARCHAR,
    OCCURRED_AT     TIMESTAMP_LTZ,
    METRIC_NAME     VARCHAR,
    METRIC_VALUE    FLOAT,
    WASTED_SECONDS  FLOAT,
    DETAIL          VARIANT,
    RECOMMENDATION  VARCHAR
)
COMMENT = 'Warehouse Studio: one row per cost/performance finding written by the SP_DETECT_* procedures';

CREATE TABLE IF NOT EXISTS {{DB}}.{{SCHEMA}}.TUNING_SCAN_LOG (
    QUERY_ID    VARCHAR,
    DETECTOR    VARCHAR,
    STATUS      VARCHAR,
    SCANNED_AT  TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
)
COMMENT = 'Warehouse Studio: queries already inspected by operator-level detectors';

CREATE TABLE IF NOT EXISTS {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (
    RUN_ID        NUMBER AUTOINCREMENT START 1 INCREMENT 1 NOORDER,
    BATCH_ID      VARCHAR,
    DETECTOR      VARCHAR,
    STARTED_AT    TIMESTAMP_LTZ,
    FINISHED_AT   TIMESTAMP_LTZ,
    STATUS        VARCHAR,
    RESULT        VARIANT,
    ERROR_MESSAGE VARCHAR
)
COMMENT = 'Warehouse Studio: per-detector outcome of every SP_RUN_ALL_DETECTORS batch';

-- ---------------------------------------------------------------- findings views

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.V_TUNING_FINDINGS
COMMENT = 'Warehouse Studio: findings from the last 30 days'
AS
SELECT finding_id,
       detector,
       severity,
       object_type,
       object_name,
       occurred_at,
       metric_name,
       ROUND(metric_value, 3)   AS metric_value,
       ROUND(wasted_seconds, 1) AS wasted_seconds,
       recommendation,
       detail,
       detected_at
FROM   {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
WHERE  detected_at >= DATEADD('day', -30, CURRENT_TIMESTAMP());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.V_TUNING_PRIORITIES
COMMENT = 'Warehouse Studio: findings rolled up by detector and severity, ranked by wasted time'
AS
SELECT detector,
       severity,
       object_type,
       COUNT(*)                             AS findings,
       ROUND(SUM(wasted_seconds))           AS total_wasted_seconds,
       ROUND(SUM(wasted_seconds) / 3600, 2) AS total_wasted_hours,
       MAX(detected_at)                     AS last_detected_at,
       ANY_VALUE(recommendation)            AS recommendation
FROM   {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
WHERE  detected_at >= DATEADD('day', -30, CURRENT_TIMESTAMP())
GROUP BY 1, 2, 3;

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.V_CLUSTERING_CANDIDATES
COMMENT = 'Warehouse Studio: repeated query shapes that scan most of a large table'
AS
WITH scans AS (
    SELECT query_parameterized_hash,
           COUNT(*)                                              AS executions,
           AVG(partitions_scanned / NULLIF(partitions_total, 0)) AS avg_scan_ratio,
           AVG(partitions_total)                                 AS avg_partitions_total,
           SUM(execution_time) / 1000                            AS total_execution_secs,
           ANY_VALUE(query_id)                                   AS sample_query_id,
           ANY_VALUE(LEFT(query_text, 200))                      AS sql_preview
    FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
    WHERE  start_time       >= DATEADD('day', -7, CURRENT_TIMESTAMP())
    AND    execution_status  = 'SUCCESS'
    AND    warehouse_size    IS NOT NULL
    AND    partitions_total  > 0
    GROUP BY query_parameterized_hash
)
SELECT *
FROM   scans
WHERE  avg_scan_ratio       > 0.8
AND    avg_partitions_total > 1000
AND    executions          >= 5;

-- ---------------------------------------------------------------- telemetry views

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_QUERY_HISTORY
COMMENT = 'Warehouse Studio: 30 days of query history, credential-bearing statements redacted'
AS
SELECT query_id,
       CASE
         WHEN query_type ILIKE '%USER%'
           OR query_type ILIKE '%INTEGRATION%'
           OR query_type ILIKE '%SECRET%'
           OR REGEXP_INSTR(query_text, 'password|private_key|rsa_public_key|secret|token|credentials', 1, 1, 0, 'i') > 0
         THEN '/* redacted by Warehouse Studio: statement may carry credentials */'
         ELSE query_text
       END                                  AS query_text,
       query_type,
       query_tag,
       query_parameterized_hash,
       database_name,
       schema_name,
       user_name,
       role_name,
       warehouse_name,
       warehouse_size,
       warehouse_type,
       cluster_number,
       execution_status,
       error_code,
       error_message,
       start_time,
       end_time,
       total_elapsed_time,
       compilation_time,
       execution_time,
       queued_provisioning_time,
       queued_repair_time,
       queued_overload_time,
       transaction_blocked_time,
       bytes_scanned,
       percentage_scanned_from_cache,
       bytes_written,
       bytes_spilled_to_local_storage,
       bytes_spilled_to_remote_storage,
       bytes_sent_over_the_network,
       rows_produced,
       rows_inserted,
       rows_updated,
       rows_deleted,
       partitions_scanned,
       partitions_total,
       credits_used_cloud_services,
       query_acceleration_bytes_scanned,
       is_client_generated_statement
FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
WHERE  start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_QUERY_ATTRIBUTION
COMMENT = 'Warehouse Studio: compute credits attributed to individual queries (30 days)'
AS
SELECT query_id,
       parent_query_id,
       root_query_id,
       warehouse_name,
       query_hash,
       query_parameterized_hash,
       query_tag,
       user_name,
       start_time,
       end_time,
       credits_attributed_compute,
       credits_used_query_acceleration
FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_ATTRIBUTION_HISTORY
WHERE  start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_WAREHOUSE_METERING
COMMENT = 'Warehouse Studio: hourly warehouse credits (365 days)'
AS
SELECT start_time,
       end_time,
       warehouse_name,
       credits_used,
       credits_used_compute,
       credits_used_cloud_services
FROM   SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE  start_time >= DATEADD('day', -365, CURRENT_TIMESTAMP());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_METERING_DAILY
COMMENT = 'Warehouse Studio: daily credits by service type, as billed (365 days)'
AS
SELECT usage_date,
       service_type,
       credits_used_compute,
       credits_used_cloud_services,
       credits_used,
       credits_adjustment_cloud_services,
       credits_billed
FROM   SNOWFLAKE.ACCOUNT_USAGE.METERING_DAILY_HISTORY
WHERE  usage_date >= DATEADD('day', -365, CURRENT_DATE());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_WAREHOUSE_LOAD
COMMENT = 'Warehouse Studio: 5-minute warehouse load (running / queued / blocked), 30 days'
AS
SELECT start_time,
       end_time,
       warehouse_name,
       avg_running,
       avg_queued_load,
       avg_queued_provisioning,
       avg_blocked
FROM   SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_LOAD_HISTORY
WHERE  start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_WAREHOUSE_EVENTS
COMMENT = 'Warehouse Studio: warehouse resume / suspend / resize events, 30 days'
AS
SELECT timestamp,
       warehouse_name,
       cluster_number,
       event_name,
       event_reason,
       event_state,
       user_name,
       role_name
FROM   SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_EVENTS_HISTORY
WHERE  timestamp >= DATEADD('day', -30, CURRENT_TIMESTAMP());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_STORAGE_DAILY
COMMENT = 'Warehouse Studio: account storage by day (table, stage, fail-safe), 365 days'
AS
SELECT usage_date,
       storage_bytes,
       stage_bytes,
       failsafe_bytes
FROM   SNOWFLAKE.ACCOUNT_USAGE.STORAGE_USAGE
WHERE  usage_date >= DATEADD('day', -365, CURRENT_DATE());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_DATABASE_STORAGE
COMMENT = 'Warehouse Studio: storage per database by day, 90 days'
AS
SELECT usage_date,
       database_name,
       average_database_bytes,
       average_failsafe_bytes
FROM   SNOWFLAKE.ACCOUNT_USAGE.DATABASE_STORAGE_USAGE_HISTORY
WHERE  usage_date >= DATEADD('day', -90, CURRENT_DATE());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_TABLE_STORAGE
COMMENT = 'Warehouse Studio: live tables with active / time-travel / fail-safe / clone bytes'
AS
SELECT id                                                      AS table_id,
       table_catalog || '.' || table_schema || '.' || table_name AS path_to_object,
       table_catalog                                           AS database_name,
       table_schema                                            AS schema_name,
       table_name,
       is_transient,
       table_created,
       active_bytes,
       time_travel_bytes,
       failsafe_bytes,
       retained_for_clone_bytes,
       active_bytes + time_travel_bytes + failsafe_bytes + retained_for_clone_bytes AS total_bytes,
       ROUND((time_travel_bytes + failsafe_bytes) / NULLIF(active_bytes, 0), 2)      AS overhead_multiple
FROM   SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS
WHERE  deleted = FALSE
AND    table_dropped IS NULL
AND    schema_dropped IS NULL
AND    catalog_dropped IS NULL;

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_TASK_HISTORY
COMMENT = 'Warehouse Studio: task runs, 30 days'
AS
SELECT name,
       database_name,
       schema_name,
       state,
       error_code,
       error_message,
       scheduled_time,
       query_start_time,
       completed_time,
       query_id,
       root_task_id,
       run_id
FROM   SNOWFLAKE.ACCOUNT_USAGE.TASK_HISTORY
WHERE  scheduled_time >= DATEADD('day', -30, CURRENT_TIMESTAMP());

CREATE OR REPLACE VIEW {{DB}}.{{SCHEMA}}.WS_CLUSTERING_HISTORY
COMMENT = 'Warehouse Studio: automatic clustering credits, 90 days'
AS
SELECT start_time,
       end_time,
       database_name,
       schema_name,
       table_name,
       credits_used,
       num_bytes_reclustered,
       num_rows_reclustered
FROM   SNOWFLAKE.ACCOUNT_USAGE.AUTOMATIC_CLUSTERING_HISTORY
WHERE  start_time >= DATEADD('day', -90, CURRENT_TIMESTAMP());

-- ---------------------------------------------------------------- detectors

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_LONG_COMPILATION(P_LOOKBACK_DAYS NUMBER, P_MIN_COMPILE_MS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: compilation time longer than execution time'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    SELECT 'sp_detect_long_compilation',
           CASE WHEN q.compilation_time > 10000 THEN 'HIGH' ELSE 'MEDIUM' END,
           'QUERY',
           q.query_id,
           q.start_time,
           'compile_to_execute_ratio',
           q.compilation_time / NULLIF(q.execution_time, 0),
           q.compilation_time / 1000,
           OBJECT_CONSTRUCT('compilation_ms',   q.compilation_time,
                            'execution_ms',     q.execution_time,
                            'query_text_bytes', LENGTH(q.query_text),
                            'warehouse',        q.warehouse_name,
                            'warehouse_size',   q.warehouse_size,
                            'user',             q.user_name,
                            'sql_preview',      LEFT(q.query_text, 200)),
           'Shorten the SQL text: bind variables instead of inlined literals, ' ||
           'huge IN lists into a temp table join, flatten nested views. A bigger ' ||
           'warehouse will not help - compilation runs in cloud services.'
    FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
    WHERE  q.start_time       >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
    AND    q.execution_status  = 'SUCCESS'
    AND    q.compilation_time >= COALESCE(:P_MIN_COMPILE_MS, 1000)
    AND    q.compilation_time  > q.execution_time
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.object_name = q.query_id
                       AND   f.detector    = 'sp_detect_long_compilation');

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_long_compilation',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_LONG_EXECUTION(P_LOOKBACK_DAYS NUMBER, P_MIN_EXECUTION_MS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: long-running queries with a diagnosis of why'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    SELECT 'sp_detect_long_execution',
           CASE WHEN q.execution_time > 300000 THEN 'HIGH' ELSE 'MEDIUM' END,
           'QUERY',
           q.query_id,
           q.start_time,
           'execution_seconds',
           q.execution_time / 1000,
           q.execution_time / 1000,
           OBJECT_CONSTRUCT(
               'warehouse',           q.warehouse_name,
               'warehouse_size',      q.warehouse_size,
               'partitions_scanned',  q.partitions_scanned,
               'partitions_total',    q.partitions_total,
               'spill_local_bytes',   q.bytes_spilled_to_local_storage,
               'spill_remote_bytes',  q.bytes_spilled_to_remote_storage,
               'queued_overload_ms',  q.queued_overload_time,
               'rows_produced',       q.rows_produced,
               'user',                q.user_name,
               'sql_preview',         LEFT(q.query_text, 200)),
           CASE
             WHEN q.bytes_spilled_to_remote_storage > 0
                  THEN 'Remote spill. Size the warehouse up - this is not a query problem.'
             WHEN q.bytes_spilled_to_local_storage > 0
                  THEN 'Local spill. Size up one step, or reduce the working set.'
             WHEN q.queued_overload_time > q.execution_time * 0.5
                  THEN 'Mostly queueing. Concurrency problem, not query shape.'
             WHEN q.partitions_scanned / NULLIF(q.partitions_total, 0) > 0.8
                  AND q.partitions_total > 1000
                  THEN 'Scanning most of the table. Check pruning and cluster key.'
             ELSE 'Compute bound. Inspect the operator profile for the most expensive node.'
           END
    FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
    WHERE  q.start_time      >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
    AND    q.execution_status = 'SUCCESS'
    AND    q.warehouse_size   IS NOT NULL
    AND    q.bytes_scanned    > 0
    AND    q.execution_time  >= COALESCE(:P_MIN_EXECUTION_MS, 60000)
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.object_name = q.query_id
                       AND   f.detector    = 'sp_detect_long_execution');

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_long_execution',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_POOR_PRUNING(P_LOOKBACK_DAYS NUMBER, P_MIN_PARTITIONS NUMBER, P_MIN_RATIO FLOAT)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: queries scanning most partitions of a large table'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    SELECT 'sp_detect_poor_pruning',
           CASE WHEN q.partitions_scanned / NULLIF(q.partitions_total, 0) > 0.9
                THEN 'HIGH' ELSE 'MEDIUM' END,
           'QUERY',
           q.query_id,
           q.start_time,
           'partition_scan_ratio',
           q.partitions_scanned / NULLIF(q.partitions_total, 0),
           q.execution_time / 1000,
           OBJECT_CONSTRUCT('partitions_scanned', q.partitions_scanned,
                            'partitions_total',   q.partitions_total,
                            'bytes_scanned',      q.bytes_scanned,
                            'warehouse',          q.warehouse_name,
                            'warehouse_size',     q.warehouse_size,
                            'user',               q.user_name,
                            'sql_preview',        LEFT(q.query_text, 200)),
           'Predicates are not pruning. Check for functions or casts wrapped ' ||
           'around filter columns, range predicates (BETWEEN, LIKE, <>), or a ' ||
           'cluster key that does not match the WHERE clause.'
    FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
    WHERE  q.start_time       >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
    AND    q.execution_status  = 'SUCCESS'
    AND    q.warehouse_size    IS NOT NULL
    AND    q.bytes_scanned     > 0
    AND    q.partitions_total >= COALESCE(:P_MIN_PARTITIONS, 1000)
    AND    q.partitions_scanned / NULLIF(q.partitions_total, 0) > COALESCE(:P_MIN_RATIO, 0.5)
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.object_name = q.query_id
                       AND   f.detector    = 'sp_detect_poor_pruning');

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_poor_pruning',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_SPILL_AND_OOM(P_LOOKBACK_DAYS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: local / remote spill and out-of-memory failures'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    SELECT 'sp_detect_spill_and_oom',
           CASE
             WHEN q.execution_status = 'FAIL'                       THEN 'CRITICAL'
             WHEN q.bytes_spilled_to_remote_storage > 0             THEN 'CRITICAL'
             WHEN q.bytes_spilled_to_local_storage > POWER(1024, 3) THEN 'HIGH'
             ELSE 'LOW'
           END,
           'QUERY',
           q.query_id,
           q.start_time,
           'spill_gb',
           (COALESCE(q.bytes_spilled_to_local_storage, 0)
            + COALESCE(q.bytes_spilled_to_remote_storage, 0)) / POWER(1024, 3),
           q.execution_time / 1000,
           OBJECT_CONSTRUCT('warehouse',        q.warehouse_name,
                            'warehouse_size',   q.warehouse_size,
                            'status',           q.execution_status,
                            'error_message',    q.error_message,
                            'spill_local',      q.bytes_spilled_to_local_storage,
                            'spill_remote',     q.bytes_spilled_to_remote_storage,
                            'network_bytes',    q.bytes_sent_over_the_network,
                            'bytes_scanned',    q.bytes_scanned,
                            'user',             q.user_name,
                            'sql_preview',      LEFT(q.query_text, 200)),
           CASE
             WHEN q.bytes_spilled_to_remote_storage > 0
                  THEN 'Remote spill. Size up. Each step doubles credits/hour, but removing ' ||
                       'remote spill often more than halves runtime, so the larger warehouse ' ||
                       'is frequently cheaper overall.'
             WHEN q.bytes_sent_over_the_network > POWER(1024, 3)
                  THEN 'High network bytes alongside spill suggests join-key skew. ' ||
                       'Sizing up masks this; a rewrite fixes it.'
             ELSE 'Local spill. Size up one step or shrink the working set (project fewer ' ||
                  'columns, filter earlier, avoid ORDER BY on huge sets).'
           END
    FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
    WHERE  q.start_time     >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
    AND    q.warehouse_size  IS NOT NULL
    AND    (   q.bytes_spilled_to_local_storage  > 0
            OR q.bytes_spilled_to_remote_storage > 0
            OR (q.execution_status = 'FAIL' AND q.error_message ILIKE '%memory%'))
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.object_name = q.query_id
                       AND   f.detector    = 'sp_detect_spill_and_oom');

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_spill_and_oom',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_QUEUEING(P_LOOKBACK_DAYS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: warehouse-hours with more than a minute of overload queueing'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    WITH hourly AS (
        SELECT q.warehouse_name,
               DATE_TRUNC('hour', q.start_time)    AS usage_hour,
               SUM(q.queued_overload_time) / 1000  AS queued_seconds,
               COUNT(*)                            AS queued_queries,
               ANY_VALUE(q.warehouse_size)         AS warehouse_size,
               ARRAY_AGG(DISTINCT q.role_name)     AS roles
        FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
        WHERE  q.start_time          >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
        AND    q.queued_overload_time > 0
        GROUP BY 1, 2
        HAVING SUM(q.queued_overload_time) / 1000 > 60
    )
    SELECT 'sp_detect_queueing',
           CASE WHEN h.queued_seconds > 3600 THEN 'HIGH'
                WHEN h.queued_seconds > 600  THEN 'MEDIUM'
                ELSE 'LOW' END,
           'WAREHOUSE',
           h.warehouse_name,
           h.usage_hour,
           'queued_seconds_in_hour',
           h.queued_seconds,
           h.queued_seconds,
           OBJECT_CONSTRUCT('hour',           TO_CHAR(h.usage_hour, 'YYYYMMDD_HH24'),
                            'queued_queries', h.queued_queries,
                            'warehouse',      h.warehouse_name,
                            'warehouse_size', h.warehouse_size,
                            'roles',          h.roles),
           'Queueing is concurrency, not query speed. Reduce MAX_CONCURRENCY_LEVEL, ' ||
           're-time the batch, or consolidate workloads. A little queueing means the ' ||
           'warehouse is well loaded - do not tune it to zero.'
    FROM   hourly h
    WHERE  NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.detector    = 'sp_detect_queueing'
                       AND   f.object_name = h.warehouse_name
                       AND   f.occurred_at = h.usage_hour);

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_queueing',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_BLOCKED_TRANSACTIONS(P_LOOKBACK_DAYS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: queries that waited on a lock held by another transaction'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    SELECT 'sp_detect_blocked_transactions',
           CASE WHEN q.transaction_blocked_time > 300000 THEN 'HIGH' ELSE 'MEDIUM' END,
           'QUERY',
           q.query_id,
           q.start_time,
           'blocked_seconds',
           q.transaction_blocked_time / 1000,
           q.transaction_blocked_time / 1000,
           OBJECT_CONSTRUCT('warehouse',   q.warehouse_name,
                            'query_type',  q.query_type,
                            'user',        q.user_name,
                            'sql_preview', LEFT(q.query_text, 200)),
           'Object locking. Serialised access to one table, often a shared log or ' ||
           'control table. Use an EVENT table for logging, segment the target table, ' ||
           'or batch the DML. A bigger warehouse changes nothing.'
    FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
    WHERE  q.start_time              >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
    AND    q.transaction_blocked_time > 0
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.object_name = q.query_id
                       AND   f.detector    = 'sp_detect_blocked_transactions');

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_blocked_transactions',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_STORAGE_OVERHEAD(P_MIN_ACTIVE_GB FLOAT, P_RATE_PER_TB_MONTH FLOAT)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: tables whose time-travel + fail-safe outweigh their live data'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    SELECT 'sp_detect_storage_overhead',
           CASE WHEN (t.time_travel_bytes + t.failsafe_bytes) / NULLIF(t.active_bytes, 0) > 3
                THEN 'HIGH' ELSE 'MEDIUM' END,
           'TABLE',
           t.table_catalog || '.' || t.table_schema || '.' || t.table_name,
           CURRENT_TIMESTAMP(),
           'retention_overhead_multiple',
           (t.time_travel_bytes + t.failsafe_bytes) / NULLIF(t.active_bytes, 0),
           NULL,
           OBJECT_CONSTRUCT(
               'table_id',        t.id,
               'is_transient',    t.is_transient,
               'active_gb',       ROUND(t.active_bytes             / POWER(1024, 3), 3),
               'time_travel_gb',  ROUND(t.time_travel_bytes        / POWER(1024, 3), 3),
               'failsafe_gb',     ROUND(t.failsafe_bytes           / POWER(1024, 3), 3),
               'clone_gb',        ROUND(t.retained_for_clone_bytes / POWER(1024, 3), 3),
               'est_monthly_usd', ROUND((t.active_bytes + t.time_travel_bytes
                                        + t.failsafe_bytes + t.retained_for_clone_bytes)
                                        / POWER(1024, 4)
                                        * COALESCE(:P_RATE_PER_TB_MONTH, 23), 2)),
           'High-churn table. Lower DATA_RETENTION_TIME_IN_DAYS, or make it TRANSIENT ' ||
           'if it is reloadable (transient tables carry no Fail-safe at all). Batch the ' ||
           'DML rather than running it row at a time.'
    FROM   SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS t
    WHERE  t.deleted = FALSE
    AND    t.table_dropped IS NULL
    AND    t.active_bytes / POWER(1024, 3) > COALESCE(:P_MIN_ACTIVE_GB, 1)
    AND    (t.time_travel_bytes + t.failsafe_bytes) / NULLIF(t.active_bytes, 0) > 1
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.detector    = 'sp_detect_storage_overhead'
                       AND   f.object_name = t.table_catalog || '.' || t.table_schema || '.' || t.table_name
                       AND   f.detected_at >= DATEADD('day', -1, CURRENT_TIMESTAMP()));

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_storage_overhead',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_WAREHOUSE_WASTE(P_LOOKBACK_DAYS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: metered warehouse-hours with under five minutes of work'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    WITH metered AS (
        SELECT warehouse_name,
               DATE_TRUNC('hour', start_time) AS usage_hour,
               SUM(credits_used_compute)      AS credits
        FROM   SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
        WHERE  start_time >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
        GROUP BY 1, 2
    ),
    worked AS (
        SELECT warehouse_name,
               DATE_TRUNC('hour', start_time) AS usage_hour,
               SUM(execution_time) / 1000     AS execution_seconds,
               COUNT(*)                       AS queries
        FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE  start_time     >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
        AND    warehouse_size  IS NOT NULL
        GROUP BY 1, 2
    )
    SELECT 'sp_detect_warehouse_waste',
           CASE WHEN COALESCE(w.execution_seconds, 0) < 60 THEN 'HIGH' ELSE 'MEDIUM' END,
           'WAREHOUSE',
           m.warehouse_name,
           m.usage_hour,
           'credits_per_execution_minute',
           m.credits / NULLIF(COALESCE(w.execution_seconds, 0) / 60, 0),
           GREATEST(3600 - COALESCE(w.execution_seconds, 0), 0),
           OBJECT_CONSTRUCT('usage_hour',        TO_CHAR(m.usage_hour, 'YYYYMMDD_HH24'),
                            'warehouse',         m.warehouse_name,
                            'credits',           ROUND(m.credits, 4),
                            'execution_seconds', ROUND(COALESCE(w.execution_seconds, 0), 1),
                            'queries',           COALESCE(w.queries, 0)),
           'Warehouse metering with little or no query execution. Set AUTO_SUSPEND = 60, ' ||
           'and check for sessions or BI tools holding the warehouse awake.'
    FROM   metered m
    LEFT JOIN worked w
      ON   w.warehouse_name = m.warehouse_name
     AND   w.usage_hour     = m.usage_hour
    WHERE  m.credits > 0.05
    AND    COALESCE(w.execution_seconds, 0) < 300
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.detector    = 'sp_detect_warehouse_waste'
                       AND   f.object_name = m.warehouse_name
                       AND   f.occurred_at = m.usage_hour);

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_warehouse_waste',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_WAREHOUSE_SETTINGS()
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: warehouses with AUTO_SUSPEND unset or above 60 seconds'
EXECUTE AS OWNER
AS
$$
BEGIN
    -- SHOW inside scripting runs through EXECUTE IMMEDIATE; RESULT_SCAN reads it.
    -- Nothing else may run between the two.
    EXECUTE IMMEDIATE 'SHOW WAREHOUSES';

    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    SELECT 'sp_detect_warehouse_settings',
           CASE WHEN "auto_suspend" IS NULL OR "auto_suspend" = 0 THEN 'HIGH'
                WHEN "auto_suspend" > 600                         THEN 'HIGH'
                ELSE 'MEDIUM' END,
           'WAREHOUSE',
           "name",
           CURRENT_TIMESTAMP(),
           'auto_suspend_seconds',
           COALESCE("auto_suspend", 999999),
           NULL,
           OBJECT_CONSTRUCT('size',         "size",
                            'auto_suspend', "auto_suspend",
                            'auto_resume',  "auto_resume",
                            'state',        "state",
                            'type',         "type",
                            'fix_sql',      'ALTER WAREHOUSE "' || "name" || '" SET AUTO_SUSPEND = 60;'),
           'AUTO_SUSPEND should be 60 unless you have a stated reason. Billing has a ' ||
           '60-second minimum then bills per second, so 60 wastes nothing. Exceptions: ' ||
           'a warehouse in continual use, or one kept warm deliberately for BI cache reuse.'
    FROM   TABLE(RESULT_SCAN(LAST_QUERY_ID())) w
    WHERE  ("auto_suspend" IS NULL OR "auto_suspend" = 0 OR "auto_suspend" > 60)
    AND    NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.detector    = 'sp_detect_warehouse_settings'
                       AND   f.object_name = w."name"
                       AND   f.detected_at >= DATEADD('day', -1, CURRENT_TIMESTAMP()));

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_warehouse_settings',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_SELECT_STAR(P_LOOKBACK_DAYS NUMBER, P_MIN_EXECUTION_MS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: slow SELECT * query shapes'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    WITH shapes AS (
        SELECT q.query_parameterized_hash             AS shape,
               MIN(q.start_time)                      AS first_seen,
               SUM(q.execution_time) / 1000           AS execution_seconds,
               COUNT(*)                               AS executions,
               ROUND(SUM(q.bytes_scanned) / POWER(1024, 3), 2) AS gb_scanned,
               ANY_VALUE(LEFT(q.query_text, 200))     AS sql_preview,
               ANY_VALUE(q.query_id)                  AS sample_query_id,
               ANY_VALUE(q.warehouse_name)            AS warehouse
        FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
        WHERE  q.start_time      >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
        AND    q.execution_status = 'SUCCESS'
        AND    q.warehouse_size   IS NOT NULL
        AND    q.query_type       = 'SELECT'
        AND    q.execution_time  >= COALESCE(:P_MIN_EXECUTION_MS, 10000)
        -- REGEXP_INSTR, not RLIKE: RLIKE matches the whole string.
        AND    REGEXP_INSTR(q.query_text, 'select\\s+\\*', 1, 1, 0, 'is') > 0
        GROUP BY q.query_parameterized_hash
    )
    SELECT 'sp_detect_select_star',
           'LOW',
           'QUERY',
           s.shape,
           s.first_seen,
           'total_execution_seconds',
           s.execution_seconds,
           s.execution_seconds,
           OBJECT_CONSTRUCT('executions',      s.executions,
                            'gb_scanned',      s.gb_scanned,
                            'warehouse',       s.warehouse,
                            'sql_preview',     s.sql_preview,
                            'sample_query_id', s.sample_query_id),
           'SELECT * reads every column from every micro-partition. Name the columns ' ||
           'you need. Free to fix, and it compounds across every execution of the shape.'
    FROM   shapes s
    WHERE  NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.detector    = 'sp_detect_select_star'
                       AND   f.object_name = s.shape
                       AND   f.detected_at >= DATEADD('day', -1, CURRENT_TIMESTAMP()));

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_select_star',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_REPEAT_OFFENDERS(P_LOOKBACK_DAYS NUMBER, P_MIN_EXECUTIONS NUMBER, P_TOP_N NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: query shapes with the highest total execution time'
EXECUTE AS OWNER
AS
$$
BEGIN
    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
          (detector, severity, object_type, object_name, occurred_at,
           metric_name, metric_value, wasted_seconds, detail, recommendation)
    WITH shapes AS (
        SELECT query_parameterized_hash                AS shape,
               MIN(start_time)                         AS first_seen,
               SUM(execution_time) / 1000              AS execution_seconds,
               COUNT(*)                                AS executions,
               ROUND(AVG(execution_time) / 1000, 1)    AS avg_execution_secs,
               ROUND(SUM(compilation_time) / 1000, 1)  AS total_compile_secs,
               ANY_VALUE(warehouse_name)               AS warehouse,
               ANY_VALUE(warehouse_size)               AS warehouse_size,
               ANY_VALUE(LEFT(query_text, 200))        AS sql_preview,
               ANY_VALUE(query_id)                     AS sample_query_id
        FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE  start_time      >= DATEADD('day', -COALESCE(:P_LOOKBACK_DAYS, 7), CURRENT_TIMESTAMP())
        AND    execution_status = 'SUCCESS'
        AND    warehouse_size   IS NOT NULL
        GROUP BY query_parameterized_hash
        HAVING COUNT(*) >= COALESCE(:P_MIN_EXECUTIONS, 5)
        QUALIFY ROW_NUMBER() OVER (ORDER BY SUM(execution_time) DESC) <= COALESCE(:P_TOP_N, 20)
    )
    SELECT 'sp_detect_repeat_offenders',
           'HIGH',
           'QUERY',
           s.shape,
           s.first_seen,
           'total_execution_seconds',
           s.execution_seconds,
           s.execution_seconds,
           OBJECT_CONSTRUCT('executions',         s.executions,
                            'avg_execution_secs', s.avg_execution_secs,
                            'total_compile_secs', s.total_compile_secs,
                            'warehouse',          s.warehouse,
                            'warehouse_size',     s.warehouse_size,
                            'sql_preview',        s.sql_preview,
                            'sample_query_id',    s.sample_query_id),
           'Highest aggregate cost by query shape. Start here: one fix multiplies ' ||
           'across every execution.'
    FROM   shapes s
    WHERE  NOT EXISTS (SELECT 1 FROM {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
                       WHERE f.detector    = 'sp_detect_repeat_offenders'
                       AND   f.object_name = s.shape
                       AND   f.detected_at >= DATEADD('day', -1, CURRENT_TIMESTAMP()));

    RETURN OBJECT_CONSTRUCT('detector', 'sp_detect_repeat_offenders',
                            'findings', SQLROWCOUNT)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_JOIN_EXPLOSION(P_LOOKBACK_DAYS NUMBER, P_MIN_ELAPSED_MS NUMBER, P_MAX_QUERIES NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio detector: Cartesian joins found in the operator profile of slow queries'
EXECUTE AS OWNER
AS
$$
DECLARE
    v_days    NUMBER DEFAULT 1;
    v_scanned NUMBER DEFAULT 0;
    v_found   NUMBER DEFAULT 0;
    v_failed  NUMBER DEFAULT 0;
    v_qid     VARCHAR;
    v_sql     VARCHAR;
BEGIN
    -- Operator stats live for 14 days. Asking for more is wasted effort.
    v_days := LEAST(GREATEST(COALESCE(:P_LOOKBACK_DAYS, 1), 1), 14);

    LET rs RESULTSET := (
        SELECT q.query_id
        FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
        WHERE  q.start_time        >= DATEADD('day', -:v_days, CURRENT_TIMESTAMP())
        AND    q.query_type         IN ('SELECT', 'CREATE_TABLE_AS_SELECT', 'INSERT', 'MERGE')
        AND    q.execution_status   = 'SUCCESS'
        AND    q.warehouse_size     IS NOT NULL
        AND    q.bytes_scanned      > 0
        AND    q.total_elapsed_time > COALESCE(:P_MIN_ELAPSED_MS, 60000)
        AND    NOT EXISTS (SELECT 1
                           FROM   {{DB}}.{{SCHEMA}}.TUNING_SCAN_LOG s
                           WHERE  s.query_id = q.query_id
                           AND    s.detector = 'sp_detect_join_explosion')
        QUALIFY ROW_NUMBER() OVER (ORDER BY q.total_elapsed_time DESC) <= COALESCE(:P_MAX_QUERIES, 200)
    );

    LET c1 CURSOR FOR rs;

    FOR rec IN c1 DO
        v_qid := rec.query_id;

        -- Guard the concatenation: query ids are UUID-shaped, nothing else.
        IF (RLIKE(v_qid, '[0-9a-fA-F-]{36}')) THEN
            v_sql := 'INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
                          (detector, severity, object_type, object_name,
                           occurred_at, metric_name, metric_value,
                           wasted_seconds, detail, recommendation)
                      SELECT ''sp_detect_join_explosion'',
                             ''CRITICAL'',
                             ''QUERY'',
                             ''' || v_qid || ''',
                             CURRENT_TIMESTAMP(),
                             ''row_multiple'',
                             operator_statistics:output_rows::FLOAT
                                / NULLIF(operator_statistics:input_rows::FLOAT, 0),
                             NULL,
                             OBJECT_CONSTRUCT(
                                 ''operator_id'',         operator_id,
                                 ''operator_type'',       operator_type,
                                 ''output_rows'',         operator_statistics:output_rows,
                                 ''input_rows'',          operator_statistics:input_rows,
                                 ''operator_attributes'', operator_attributes),
                             ''Cartesian join. Count your join predicates: they should equal '' ||
                             ''(number of tables - 1).''
                      FROM   TABLE(GET_QUERY_OPERATOR_STATS(''' || v_qid || '''))
                      WHERE  operator_type = ''CartesianJoin''';
            BEGIN
                EXECUTE IMMEDIATE :v_sql;
                v_found   := v_found + SQLROWCOUNT;
                v_scanned := v_scanned + 1;
                INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_SCAN_LOG (query_id, detector, status)
                VALUES (:v_qid, 'sp_detect_join_explosion', 'SCANNED');
            EXCEPTION
                WHEN OTHER THEN
                    v_failed := v_failed + 1;
                    INSERT INTO {{DB}}.{{SCHEMA}}.TUNING_SCAN_LOG (query_id, detector, status)
                    VALUES (:v_qid, 'sp_detect_join_explosion', 'ERROR');
            END;
        END IF;
    END FOR;

    -- Attribute elapsed time so the priority view can rank Cartesian joins.
    UPDATE {{DB}}.{{SCHEMA}}.TUNING_FINDINGS f
    SET    wasted_seconds = q.total_elapsed_time / 1000,
           occurred_at    = q.start_time
    FROM   SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY q
    WHERE  q.query_id       = f.object_name
    AND    f.detector       = 'sp_detect_join_explosion'
    AND    f.wasted_seconds IS NULL
    AND    q.start_time    >= DATEADD('day', -:v_days, CURRENT_TIMESTAMP());

    RETURN OBJECT_CONSTRUCT('detector',        'sp_detect_join_explosion',
                            'lookback_days',   :v_days,
                            'queries_scanned', :v_scanned,
                            'findings',        :v_found,
                            'failed',          :v_failed)::VARCHAR;
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_RUN_ALL_DETECTORS(P_LOOKBACK_DAYS NUMBER)
RETURNS VARCHAR
LANGUAGE SQL
COMMENT = 'Warehouse Studio: run every detector; one failure never stops the batch'
EXECUTE AS OWNER
AS
$$
DECLARE
    v_days   NUMBER DEFAULT 7;
    v_start  TIMESTAMP_LTZ;
    v_t0     TIMESTAMP_LTZ;
    v_total  NUMBER DEFAULT 0;
    v_failed NUMBER DEFAULT 0;
    v_batch  VARCHAR;
    v_res    VARCHAR;
BEGIN
    v_days  := COALESCE(:P_LOOKBACK_DAYS, 7);
    v_start := CURRENT_TIMESTAMP();
    v_batch := UUID_STRING();

    -- One block per detector so a permissions gap on one ACCOUNT_USAGE view
    -- leaves the rest working.  Each outcome lands in WS_DETECTOR_RUNS.
    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_JOIN_EXPLOSION(LEAST(:v_days, 14), 60000, 200) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_join_explosion', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_join_explosion', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_LONG_COMPILATION(:v_days, 1000) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_long_compilation', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_long_compilation', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_LONG_EXECUTION(:v_days, 60000) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_long_execution', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_long_execution', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_POOR_PRUNING(:v_days, 1000, 0.5) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_poor_pruning', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_poor_pruning', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_SPILL_AND_OOM(:v_days) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_spill_and_oom', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_spill_and_oom', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_QUEUEING(:v_days) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_queueing', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_queueing', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_BLOCKED_TRANSACTIONS(:v_days) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_blocked_transactions', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_blocked_transactions', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_STORAGE_OVERHEAD(1, 23) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_storage_overhead', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_storage_overhead', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_WAREHOUSE_WASTE(:v_days) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_warehouse_waste', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_warehouse_waste', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_WAREHOUSE_SETTINGS() INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_warehouse_settings', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_warehouse_settings', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_SELECT_STAR(:v_days, 10000) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_select_star', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_select_star', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    v_t0 := CURRENT_TIMESTAMP();
    BEGIN
        CALL {{DB}}.{{SCHEMA}}.SP_DETECT_REPEAT_OFFENDERS(:v_days, 5, 20) INTO :v_res;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, result)
            SELECT :v_batch, 'sp_detect_repeat_offenders', :v_t0, CURRENT_TIMESTAMP(), 'SUCCESS', TRY_PARSE_JSON(:v_res);
    EXCEPTION WHEN OTHER THEN
        v_failed := v_failed + 1;
        INSERT INTO {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS (batch_id, detector, started_at, finished_at, status, error_message)
            VALUES (:v_batch, 'sp_detect_repeat_offenders', :v_t0, CURRENT_TIMESTAMP(), 'ERROR', :SQLERRM);
    END;

    SELECT COUNT(*) INTO :v_total
    FROM   {{DB}}.{{SCHEMA}}.TUNING_FINDINGS
    WHERE  detected_at >= :v_start;

    RETURN OBJECT_CONSTRUCT('batch_id',      :v_batch,
                            'run_started',   :v_start,
                            'lookback_days', :v_days,
                            'new_findings',  :v_total,
                            'failed',        :v_failed)::VARCHAR;
END;
$$;

-- ---------------------------------------------------------------- reports (return tables)

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_RUN_FULL_OPTIMIZATION_ANALYSIS(P_DAYS_BACK NUMBER DEFAULT 7, P_TOP_N NUMBER DEFAULT 5)
RETURNS TABLE (ANALYSIS_TYPE VARCHAR, FINDING VARCHAR, METRIC_VALUE VARCHAR, RECOMMENDATION VARCHAR)
LANGUAGE SQL
COMMENT = 'Warehouse Studio report: long-running, spilling, queued and poorly pruned work in one table'
EXECUTE AS OWNER
AS
$$
DECLARE
    v_days  NUMBER        DEFAULT LEAST(GREATEST(COALESCE(:P_DAYS_BACK, 7), 1), 365);
    v_top   NUMBER        DEFAULT LEAST(GREATEST(COALESCE(:P_TOP_N, 5), 1), 100);
    v_since TIMESTAMP_LTZ DEFAULT DATEADD(day, :v_days * -1, CURRENT_TIMESTAMP());
    res     RESULTSET;
BEGIN
    res := (
        WITH qh AS (
            SELECT query_id, warehouse_name, warehouse_size, total_elapsed_time,
                   bytes_spilled_to_remote_storage, bytes_spilled_to_local_storage,
                   queued_overload_time, partitions_scanned, partitions_total
            FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
            WHERE start_time >= :v_since
              AND warehouse_size IS NOT NULL
        )
        SELECT 'LONG_RUNNING'::VARCHAR                                   AS ANALYSIS_TYPE,
               ('Query ' || query_id)::VARCHAR                           AS FINDING,
               (TO_VARCHAR(ROUND(total_elapsed_time / 1000, 1)) || ' s')::VARCHAR AS METRIC_VALUE,
               'Review the operator profile: join order and pruning'::VARCHAR AS RECOMMENDATION
        FROM qh
        WHERE total_elapsed_time > 60000
        QUALIFY ROW_NUMBER() OVER (ORDER BY total_elapsed_time DESC) <= :v_top
        UNION ALL
        SELECT 'SPILLING'::VARCHAR,
               ('Query ' || query_id)::VARCHAR,
               (TO_VARCHAR(ROUND(bytes_spilled_to_remote_storage / POWER(1024, 3), 2))
                || ' GB remote / '
                || TO_VARCHAR(ROUND(COALESCE(bytes_spilled_to_local_storage, 0) / POWER(1024, 3), 2))
                || ' GB local')::VARCHAR,
               'Resize the warehouse - remote spilling is the strongest signal'::VARCHAR
        FROM qh
        WHERE bytes_spilled_to_remote_storage > 0
        QUALIFY ROW_NUMBER() OVER (ORDER BY bytes_spilled_to_remote_storage DESC) <= :v_top
        UNION ALL
        SELECT 'QUEUED'::VARCHAR,
               ('Warehouse ' || warehouse_name)::VARCHAR,
               (TO_VARCHAR(ROUND(SUM(queued_overload_time) / 1000, 1))
                || ' s queued across ' || TO_VARCHAR(COUNT(*)) || ' queries')::VARCHAR,
               'Enable multi-cluster scaling or resize the warehouse'::VARCHAR
        FROM qh
        WHERE queued_overload_time > 0
          AND warehouse_name IS NOT NULL
        GROUP BY warehouse_name
        HAVING SUM(queued_overload_time) > 10000
        QUALIFY ROW_NUMBER() OVER (ORDER BY SUM(queued_overload_time) DESC) <= :v_top
        UNION ALL
        SELECT 'HIGH_SCAN_RATIO'::VARCHAR,
               ('Query ' || query_id)::VARCHAR,
               (TO_VARCHAR(ROUND(partitions_scanned * 100.0 / partitions_total, 1))
                || '% of ' || TO_VARCHAR(partitions_total) || ' partitions')::VARCHAR,
               'Consider clustering, search optimization, or a tighter filter'::VARCHAR
        FROM qh
        WHERE partitions_total >= 1000
          AND partitions_scanned * 100.0 / partitions_total > 80
        QUALIFY ROW_NUMBER() OVER (ORDER BY partitions_scanned DESC) <= :v_top
        ORDER BY 1, 3
    );
    RETURN TABLE(res);
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_FIND_HIGH_TIME_TRAVEL_TABLES(P_TOP_N NUMBER DEFAULT 20, P_MIN_TIME_TRAVEL_GB FLOAT DEFAULT 1)
RETURNS TABLE (TABLE_NAME VARCHAR, TABLE_TYPE VARCHAR, IS_TRANSIENT VARCHAR, RETENTION_DAYS NUMBER,
               ACTIVE_GB FLOAT, TIME_TRAVEL_GB FLOAT, FAILSAFE_GB FLOAT, CLONE_RETAINED_GB FLOAT, TIME_TRAVEL_PCT FLOAT)
LANGUAGE SQL
COMMENT = 'Warehouse Studio report: tables carrying the most time-travel storage'
EXECUTE AS OWNER
AS
$$
DECLARE
    res RESULTSET;
BEGIN
    res := (
        SELECT (m.table_catalog || '.' || m.table_schema || '.' || m.table_name)::VARCHAR AS TABLE_NAME,
               t.table_type::VARCHAR                                AS TABLE_TYPE,
               m.is_transient::VARCHAR                              AS IS_TRANSIENT,
               t.retention_time::NUMBER                             AS RETENTION_DAYS,
               (m.active_bytes             / POWER(1024, 3))::FLOAT AS ACTIVE_GB,
               (m.time_travel_bytes        / POWER(1024, 3))::FLOAT AS TIME_TRAVEL_GB,
               (m.failsafe_bytes           / POWER(1024, 3))::FLOAT AS FAILSAFE_GB,
               (m.retained_for_clone_bytes / POWER(1024, 3))::FLOAT AS CLONE_RETAINED_GB,
               (100 * m.time_travel_bytes / NULLIF(m.active_bytes + m.time_travel_bytes, 0))::FLOAT AS TIME_TRAVEL_PCT
        FROM SNOWFLAKE.ACCOUNT_USAGE.TABLE_STORAGE_METRICS m
        LEFT JOIN SNOWFLAKE.ACCOUNT_USAGE.TABLES t
               ON t.table_id = m.id
              AND t.deleted IS NULL
        WHERE m.deleted = FALSE
          AND m.table_dropped   IS NULL
          AND m.schema_dropped  IS NULL
          AND m.catalog_dropped IS NULL
          AND m.time_travel_bytes >= COALESCE(:P_MIN_TIME_TRAVEL_GB, 1) * POWER(1024, 3)
        QUALIFY ROW_NUMBER() OVER (ORDER BY m.time_travel_bytes DESC) <= COALESCE(:P_TOP_N, 20)
        ORDER BY m.time_travel_bytes DESC
    );
    RETURN TABLE(res);
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_MONITOR_CLUSTERING_COSTS(P_DAYS_BACK NUMBER DEFAULT 7)
RETURNS TABLE (TABLE_NAME VARCHAR, CREDITS_USED FLOAT, BYTES_RECLUSTERED NUMBER, ROWS_RECLUSTERED NUMBER)
LANGUAGE SQL
COMMENT = 'Warehouse Studio report: automatic clustering credits per table'
EXECUTE AS OWNER
AS
$$
DECLARE
    res RESULTSET;
BEGIN
    res := (
        SELECT (database_name || '.' || schema_name || '.' || table_name)::VARCHAR AS TABLE_NAME,
               SUM(credits_used)::FLOAT          AS CREDITS_USED,
               SUM(num_bytes_reclustered)::NUMBER AS BYTES_RECLUSTERED,
               SUM(num_rows_reclustered)::NUMBER  AS ROWS_RECLUSTERED
        FROM SNOWFLAKE.ACCOUNT_USAGE.AUTOMATIC_CLUSTERING_HISTORY
        WHERE start_time >= DATEADD(DAY, -COALESCE(:P_DAYS_BACK, 7), CURRENT_TIMESTAMP())
        GROUP BY 1
        ORDER BY CREDITS_USED DESC
    );
    RETURN TABLE(res);
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_CLUSTERING_REPORT(P_TABLE VARCHAR, P_KEY VARCHAR)
RETURNS VARIANT
LANGUAGE SQL
COMMENT = 'Warehouse Studio report: clustering depth / overlap for a table and optional key'
EXECUTE AS OWNER
AS
$$
DECLARE
    v_json VARIANT;
BEGIN
    -- Arguments go to the system function as values, never spliced into SQL.
    IF (:P_TABLE IS NULL OR NOT RLIKE(:P_TABLE, '[A-Za-z_][A-Za-z0-9_$]*([.][A-Za-z_][A-Za-z0-9_$]*){0,2}')) THEN
        RETURN OBJECT_CONSTRUCT('error', 'P_TABLE must be an unquoted [db.][schema.]table identifier');
    END IF;
    IF (:P_KEY IS NULL OR TRIM(:P_KEY) = '') THEN
        v_json := PARSE_JSON(SYSTEM$CLUSTERING_INFORMATION(:P_TABLE));
    ELSE
        v_json := PARSE_JSON(SYSTEM$CLUSTERING_INFORMATION(:P_TABLE, :P_KEY));
    END IF;
    RETURN OBJECT_CONSTRUCT(
        'table',                          :P_TABLE,
        'key',                            COALESCE(NULLIF(TRIM(:P_KEY), ''), '(declared)'),
        'cluster_by_keys',                v_json:cluster_by_keys,
        'total_partition_count',          v_json:total_partition_count,
        'total_constant_partition_count', v_json:total_constant_partition_count,
        'constant_ratio',                 ROUND(v_json:total_constant_partition_count::FLOAT
                                                / NULLIF(v_json:total_partition_count::FLOAT, 0), 4),
        'average_depth',                  v_json:average_depth,
        'average_overlaps',               v_json:average_overlaps,
        'partition_depth_histogram',      v_json:partition_depth_histogram,
        'notes',                          v_json:notes);
EXCEPTION
    WHEN OTHER THEN
        RETURN OBJECT_CONSTRUCT('error', SQLERRM);
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_ESTIMATE_SOS_COST(P_TABLE_NAME VARCHAR)
RETURNS VARIANT
LANGUAGE SQL
COMMENT = 'Warehouse Studio report: estimated build + maintenance cost of search optimization'
EXECUTE AS OWNER
AS
$$
DECLARE
    v_json VARIANT;
BEGIN
    IF (:P_TABLE_NAME IS NULL OR NOT RLIKE(:P_TABLE_NAME, '[A-Za-z_][A-Za-z0-9_$]*([.][A-Za-z_][A-Za-z0-9_$]*){0,2}')) THEN
        RETURN OBJECT_CONSTRUCT('error', 'P_TABLE_NAME must be an unquoted [db.][schema.]table identifier');
    END IF;
    v_json := PARSE_JSON(SYSTEM$ESTIMATE_SEARCH_OPTIMIZATION_COSTS(:P_TABLE_NAME));
    RETURN v_json;
EXCEPTION
    WHEN OTHER THEN
        RETURN OBJECT_CONSTRUCT('error', SQLERRM);
END;
$$;

-- ---------------------------------------------------------------- studio helpers

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_WAREHOUSES()
RETURNS TABLE (NAME VARCHAR, STATE VARCHAR, TYPE VARCHAR, SIZE VARCHAR, MIN_CLUSTER_COUNT NUMBER,
               MAX_CLUSTER_COUNT NUMBER, STARTED_CLUSTERS NUMBER, RUNNING NUMBER, QUEUED NUMBER,
               AUTO_SUSPEND NUMBER, AUTO_RESUME VARCHAR, RESOURCE_MONITOR VARCHAR, SCALING_POLICY VARCHAR,
               ENABLE_QUERY_ACCELERATION VARCHAR, CREATED_ON TIMESTAMP_LTZ, RESUMED_ON TIMESTAMP_LTZ,
               OWNER VARCHAR, COMMENT VARCHAR)
LANGUAGE SQL
COMMENT = 'Warehouse Studio: every warehouse in the account with its settings and live state'
EXECUTE AS OWNER
AS
$$
DECLARE
    res RESULTSET;
BEGIN
    EXECUTE IMMEDIATE 'SHOW WAREHOUSES';
    res := (
        SELECT "name"::VARCHAR, "state"::VARCHAR, "type"::VARCHAR, "size"::VARCHAR,
               "min_cluster_count"::NUMBER, "max_cluster_count"::NUMBER, "started_clusters"::NUMBER,
               "running"::NUMBER, "queued"::NUMBER, TRY_TO_NUMBER("auto_suspend"::VARCHAR),
               "auto_resume"::VARCHAR, "resource_monitor"::VARCHAR, "scaling_policy"::VARCHAR,
               "enable_query_acceleration"::VARCHAR, "created_on"::TIMESTAMP_LTZ,
               "resumed_on"::TIMESTAMP_LTZ, "owner"::VARCHAR, "comment"::VARCHAR
        FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
        ORDER BY 1
    );
    RETURN TABLE(res);
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_RESOURCE_MONITORS()
RETURNS TABLE (NAME VARCHAR, CREDIT_QUOTA FLOAT, USED_CREDITS FLOAT, REMAINING_CREDITS FLOAT,
               LEVEL VARCHAR, FREQUENCY VARCHAR, START_TIME TIMESTAMP_LTZ, END_TIME TIMESTAMP_LTZ,
               NOTIFY_AT VARCHAR, SUSPEND_AT VARCHAR, SUSPEND_IMMEDIATELY_AT VARCHAR)
LANGUAGE SQL
COMMENT = 'Warehouse Studio: resource monitors with quota and consumption'
EXECUTE AS OWNER
AS
$$
DECLARE
    res RESULTSET;
BEGIN
    EXECUTE IMMEDIATE 'SHOW RESOURCE MONITORS';
    res := (
        SELECT "name"::VARCHAR, TRY_TO_DOUBLE("credit_quota"::VARCHAR), TRY_TO_DOUBLE("used_credits"::VARCHAR),
               TRY_TO_DOUBLE("remaining_credits"::VARCHAR), "level"::VARCHAR, "frequency"::VARCHAR,
               TRY_TO_TIMESTAMP_LTZ("start_time"::VARCHAR), TRY_TO_TIMESTAMP_LTZ("end_time"::VARCHAR),
               "notify_at"::VARCHAR, "suspend_at"::VARCHAR, "suspend_immediately_at"::VARCHAR
        FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
        ORDER BY 1
    );
    RETURN TABLE(res);
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_QUERY_PROFILE(P_QUERY_ID VARCHAR)
RETURNS TABLE (QUERY_ID VARCHAR, STEP_ID NUMBER, OPERATOR_ID NUMBER, PARENT_OPERATORS ARRAY,
               OPERATOR_TYPE VARCHAR, OPERATOR_STATISTICS VARIANT, EXECUTION_TIME_BREAKDOWN VARIANT,
               OPERATOR_ATTRIBUTES VARIANT)
LANGUAGE SQL
COMMENT = 'Warehouse Studio: operator-level profile of any query from the last 14 days'
EXECUTE AS OWNER
AS
$$
DECLARE
    res RESULTSET;
    v_sql VARCHAR;
BEGIN
    -- GET_QUERY_OPERATOR_STATS needs a literal id: allow only the UUID shape.
    IF (:P_QUERY_ID IS NULL OR NOT RLIKE(:P_QUERY_ID, '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')) THEN
        res := (SELECT NULL::VARCHAR, NULL::NUMBER, NULL::NUMBER, NULL::ARRAY, NULL::VARCHAR,
                       NULL::VARIANT, NULL::VARIANT, NULL::VARIANT WHERE FALSE);
        RETURN TABLE(res);
    END IF;
    v_sql := 'SELECT query_id::VARCHAR, step_id::NUMBER, operator_id::NUMBER, parent_operators::ARRAY,
                     operator_type::VARCHAR, operator_statistics::VARIANT,
                     execution_time_breakdown::VARIANT, operator_attributes::VARIANT
              FROM TABLE(GET_QUERY_OPERATOR_STATS(''' || :P_QUERY_ID || '''))
              ORDER BY step_id, operator_id';
    res := (EXECUTE IMMEDIATE :v_sql);
    RETURN TABLE(res);
END;
$$;

CREATE OR REPLACE PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_DETECTOR_SCHEDULE(P_ACTION VARCHAR)
RETURNS VARIANT
LANGUAGE SQL
COMMENT = 'Warehouse Studio: STATUS / RESUME / SUSPEND the daily detector task'
EXECUTE AS OWNER
AS
$$
DECLARE
    v_action VARCHAR DEFAULT UPPER(COALESCE(:P_ACTION, 'STATUS'));
    v_state  VARCHAR;
    v_sched  VARCHAR;
    v_wh     VARCHAR;
BEGIN
    IF (v_action NOT IN ('STATUS', 'RESUME', 'SUSPEND')) THEN
        RETURN OBJECT_CONSTRUCT('error', 'P_ACTION must be STATUS, RESUME or SUSPEND');
    END IF;
    IF (v_action = 'RESUME') THEN
        ALTER TASK {{DB}}.{{SCHEMA}}.T_WS_TUNING_DETECTORS RESUME;
    ELSEIF (v_action = 'SUSPEND') THEN
        ALTER TASK {{DB}}.{{SCHEMA}}.T_WS_TUNING_DETECTORS SUSPEND;
    END IF;
    EXECUTE IMMEDIATE 'SHOW TASKS LIKE ''T_WS_TUNING_DETECTORS'' IN SCHEMA {{DB}}.{{SCHEMA}}';
    SELECT "state", "schedule", "warehouse" INTO :v_state, :v_sched, :v_wh
    FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()));
    RETURN OBJECT_CONSTRUCT('task', '{{DB}}.{{SCHEMA}}.T_WS_TUNING_DETECTORS',
                            'state', v_state, 'schedule', v_sched, 'warehouse', v_wh,
                            'action', v_action);
END;
$$;

-- ---------------------------------------------------------------- schedule

-- Created suspended: turning on recurring spend is a decision for a human,
-- made from the Optimizer page (SP_WS_DETECTOR_SCHEDULE('RESUME')).
CREATE TASK IF NOT EXISTS {{DB}}.{{SCHEMA}}.T_WS_TUNING_DETECTORS
    WAREHOUSE = {{TASK_WAREHOUSE}}
    SCHEDULE  = 'USING CRON 0 5 * * * UTC'
    COMMENT   = 'Warehouse Studio: daily run of every cost/performance detector'
AS
    CALL {{DB}}.{{SCHEMA}}.SP_RUN_ALL_DETECTORS(2);

-- ---------------------------------------------------------------- grants

GRANT USAGE ON DATABASE {{DB}} TO ROLE {{APP_ROLE}};
GRANT USAGE ON SCHEMA {{DB}}.{{SCHEMA}} TO ROLE {{APP_ROLE}};

GRANT SELECT ON TABLE {{DB}}.{{SCHEMA}}.TUNING_FINDINGS  TO ROLE {{APP_ROLE}};
GRANT SELECT ON TABLE {{DB}}.{{SCHEMA}}.TUNING_SCAN_LOG  TO ROLE {{APP_ROLE}};
GRANT SELECT ON TABLE {{DB}}.{{SCHEMA}}.WS_DETECTOR_RUNS TO ROLE {{APP_ROLE}};

GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.V_TUNING_FINDINGS       TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.V_TUNING_PRIORITIES     TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.V_CLUSTERING_CANDIDATES TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_QUERY_HISTORY        TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_QUERY_ATTRIBUTION    TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_WAREHOUSE_METERING   TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_METERING_DAILY       TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_WAREHOUSE_LOAD       TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_WAREHOUSE_EVENTS     TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_STORAGE_DAILY        TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_DATABASE_STORAGE     TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_TABLE_STORAGE        TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_TASK_HISTORY         TO ROLE {{APP_ROLE}};
GRANT SELECT ON VIEW {{DB}}.{{SCHEMA}}.WS_CLUSTERING_HISTORY   TO ROLE {{APP_ROLE}};

GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_LONG_COMPILATION(NUMBER, NUMBER)               TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_LONG_EXECUTION(NUMBER, NUMBER)                 TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_POOR_PRUNING(NUMBER, NUMBER, FLOAT)            TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_SPILL_AND_OOM(NUMBER)                          TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_QUEUEING(NUMBER)                               TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_BLOCKED_TRANSACTIONS(NUMBER)                   TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_STORAGE_OVERHEAD(FLOAT, FLOAT)                 TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_WAREHOUSE_WASTE(NUMBER)                        TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_WAREHOUSE_SETTINGS()                           TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_SELECT_STAR(NUMBER, NUMBER)                    TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_REPEAT_OFFENDERS(NUMBER, NUMBER, NUMBER)       TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_DETECT_JOIN_EXPLOSION(NUMBER, NUMBER, NUMBER)         TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_RUN_ALL_DETECTORS(NUMBER)                             TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_RUN_FULL_OPTIMIZATION_ANALYSIS(NUMBER, NUMBER)        TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_FIND_HIGH_TIME_TRAVEL_TABLES(NUMBER, FLOAT)           TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_MONITOR_CLUSTERING_COSTS(NUMBER)                      TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_CLUSTERING_REPORT(VARCHAR, VARCHAR)                   TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_ESTIMATE_SOS_COST(VARCHAR)                            TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_WAREHOUSES()                                       TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_RESOURCE_MONITORS()                                TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_QUERY_PROFILE(VARCHAR)                             TO ROLE {{APP_ROLE}};
GRANT USAGE ON PROCEDURE {{DB}}.{{SCHEMA}}.SP_WS_DETECTOR_SCHEDULE(VARCHAR)                         TO ROLE {{APP_ROLE}};
