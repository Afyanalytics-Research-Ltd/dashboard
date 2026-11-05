-- Patient counts for the internal reporting dashboard (one-time setup; creates AFYA_* objects in HOSPITALS.ANALYTICS).
--   Patient Records = rows left after dropping superseded versions and soft-deleted records.
--   UPDs            = distinct UPD_KEY values, within one source (no cross-facility matching).

USE ROLE DATAANALYSTS;

-- Sources. STATUS: proposed (found by discovery, awaiting approval) | active | rejected (never proposed again).
CREATE TABLE HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES (
    SOURCE           STRING NOT NULL,   -- name shown on the dashboard
    SOURCE_OBJECT    STRING NOT NULL,   -- DATABASE.SCHEMA.OBJECT
    UPD_KEY          STRING NOT NULL,
    RECORD_KEY       STRING,            -- record identifier, needed with VERSION_COLUMN
    DELETED_COLUMN   STRING,            -- a non-empty value means deleted
    VERSION_COLUMN   STRING,            -- keep the latest row per RECORD_KEY
    RECEIVED_COLUMN  STRING,            -- per-record warehouse load time
    STATUS           STRING NOT NULL DEFAULT 'active',
    PROFILE          VARIANT,           -- discovery statistics (counts only)
    UPDATED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_BY       STRING DEFAULT CURRENT_USER(),
    CONSTRAINT PK_AFYA_PATIENT_COUNT_SOURCES PRIMARY KEY (SOURCE)
);

-- One row per source per run. STATUS: ok | review | failed.
CREATE TABLE HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS (
    RUN_ID              STRING        NOT NULL,
    RUN_AT              TIMESTAMP_NTZ NOT NULL,
    SOURCE              STRING        NOT NULL,
    CONFIG_SIG          STRING,
    SOURCE_ALTERED      TIMESTAMP_LTZ,
    SOURCE_CREATED      DATE,
    RAW_MAIN_LOAD       DATE,            -- day most of the source's raw events arrived (first publish only)
    RAW_LAST_LOAD       DATE,            -- latest raw event load for the source
    ROWS_READ           INTEGER,
    ROWS_SUPERSEDED     INTEGER,
    ROWS_DELETED        INTEGER,
    PATIENT_RECORDS     INTEGER,
    UPDS                INTEGER,
    RECORDS_NO_UPD_KEY  INTEGER,
    STATUS              STRING        NOT NULL,
    CHECK_NOTES         STRING,
    REUSED              BOOLEAN DEFAULT FALSE,
    PUBLISHED           BOOLEAN DEFAULT FALSE,
    ERROR               STRING
);

-- What arrived per source and day (RECEIVED_DATE NULL = undated). Sums equal the published figures.
CREATE TABLE HOSPITALS.ANALYTICS.AFYA_PATIENT_RECORD_RECEIPTS (
    SOURCE         STRING        NOT NULL,
    RECEIVED_DATE  DATE,
    RECORDS_ADDED  INTEGER       NOT NULL,
    UPDS_ADDED     INTEGER       NOT NULL,
    METHOD         STRING        NOT NULL,   -- load time | table created | first counted | run increment | backfill
    RUN_ID         STRING,
    RECORDED_AT    TIMESTAMP_NTZ NOT NULL
);

INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
    (SOURCE, SOURCE_OBJECT, UPD_KEY, RECORD_KEY, DELETED_COLUMN, VERSION_COLUMN, RECEIVED_COLUMN, STATUS) VALUES
    ('Kisumu Specialist Hospital (KSH)',       'HOSPITALS.KISUMU_V3_CLEAN.EVALUATION_VISITS',                     'PATIENT',              'ID', 'DELETED_AT', 'UPDATED_AT', NULL,           'active'),
    ('Tenri',                                  'HOSPITALS.TENRI.EVALUATION_VISITS',                               'PATIENT',              'ID', 'DELETED_AT', 'UPDATED_AT', '_INGESTED_AT', 'active'),
    ('Lodwar',                                 'HOSPITALS.LODWAR_CLEAN.EVALUATION_VISITS',                        'PATIENT',              'ID', 'DELETED_AT', 'UPDATED_AT', NULL,           'active'),
    ('Kakamega',                               'HOSPITALS.KAKAMEGA_CLEAN.EVALUATION_VISITS',                      'PATIENT',              'ID', 'DELETED_AT', 'UPDATED_AT', NULL,           'active'),
    ('Kakamega HIV data',                      'HOSPITALS.KAKAMEGA_CLEAN.HIV_HTS_DATA',                           'PATIENT_ID',           NULL, NULL,         NULL,         NULL,           'active'),
    ('St. Peter''s Orthopedics Hospital (SPH)', 'HOSPITALS.STAGING.STG_VISITS',                                   'COMPOSITE_PATIENT_ID', NULL, NULL,         NULL,         NULL,           'active'),
    ('Nandi',                                  'HOSPITALS.NANDI_CLEAN.NANDI_OUTPATIENT_WIDE',                     'UNIQUE_PATIENT_ID',    NULL, NULL,         NULL,         NULL,           'active'),
    ('Migori HIV',                             'HOSPITALS.MIGORI_CLEAN.AROMBE_HIV_RAW',                           'PATIENT_ID_HASH',      NULL, NULL,         NULL,         NULL,           'active'),
    ('Mother Francisca historical data',       'HOSPITALS.MOTHERFRANSICA_HISTORICAL_DATA.LIMPATIENTVISITDETAILS', 'REGID',                NULL, NULL,         NULL,         NULL,           'active'),
    ('Mother Francisca medical camp',          'HOSPITALS.MOTHERFRANSICA_MEDICAL_CAMP_CLEAN.PATIENT_DATA',        'PATIENT_ID',           NULL, NULL,         NULL,         NULL,           'active'),
    ('Silverwood',                             'HOSPITALS.SILVERWOOD_CLEAN.VISITS',                               'PATIENT',              NULL, NULL,         NULL,         NULL,           'active'),
    ('Siaya medical camp',                     'HOSPITALS.SIAYA_MEDICAL_CAMP.SIAYA_MEDICAL_CAMP_COMPLETE_DATA_CLEANED', 'PATIENT_ID',     NULL, NULL,         NULL,         NULL,           'active'),
    -- already covered by the sources above, so discovery must not propose them
    ('KSH superseded views',                   'HOSPITALS.KISUMU_CLEAN.VISITS',                                   'PATIENT',              NULL, NULL,         NULL,         NULL,           'rejected'),
    ('SPH V1 feed',                            'HOSPITALS.ORTHOPEDIC_CLEAN.ORTHO_ORDERS',                         'PATIENT_ID',           NULL, NULL,         NULL,         NULL,           'rejected'),
    ('SPH V2 feed',                            'HOSPITALS.ORTHOPEDIC_CLEAN_V2.ORTHO_VISITS',                      'PATIENT_ID',           NULL, NULL,         NULL,         NULL,           'rejected');

-- Finds the most visit-like table, with a patient identifier, in each *_CLEAN schema that has no source yet,
-- profiles it with one aggregate scan, and adds it as 'proposed' with the columns it inferred.
CREATE PROCEDURE HOSPITALS.ANALYTICS.AFYA_DISCOVER_PATIENT_DATASETS()
RETURNS STRING
LANGUAGE SQL
EXECUTE AS OWNER
AS
$$
DECLARE
    n_proposed INTEGER DEFAULT 0;
    candidates CURSOR FOR
        WITH tabs AS (
            SELECT c.TABLE_SCHEMA AS sch, c.TABLE_NAME AS tbl, MAX(t.ROW_COUNT) AS n_rows,
                   MIN_BY(c.COLUMN_NAME, CASE c.COLUMN_NAME WHEN 'UNIQUE_PATIENT_ID' THEN 1 WHEN 'COMPOSITE_PATIENT_ID' THEN 2
                          WHEN 'PATIENT_ID' THEN 3 WHEN 'PATIENT' THEN 4 WHEN 'PATIENT_ID_HASH' THEN 5 WHEN 'REGID' THEN 6 END) AS uk,
                   MIN_BY(c.COLUMN_NAME, CASE c.COLUMN_NAME WHEN 'VISIT_ID' THEN 1 WHEN 'ENCOUNTER_ID' THEN 2
                          WHEN 'PRESCID' THEN 3 WHEN 'APTID' THEN 4 WHEN 'ID' THEN 5 END) AS rk,
                   MIN_BY(c.COLUMN_NAME, CASE c.COLUMN_NAME WHEN 'DELETED_AT' THEN 1 WHEN 'IS_DELETED' THEN 2 END) AS dc,
                   MIN_BY(c.COLUMN_NAME, CASE c.COLUMN_NAME WHEN 'UPDATED_AT' THEN 1 END) AS vc,
                   MIN_BY(c.COLUMN_NAME, CASE c.COLUMN_NAME WHEN '_INGESTED_AT' THEN 1 WHEN 'INGESTED_AT' THEN 2
                          WHEN '_RAW_INGESTED_AT' THEN 3 WHEN 'INGESTION_DATE' THEN 4 WHEN '_LOADED_AT' THEN 5 END) AS rc
            FROM HOSPITALS.INFORMATION_SCHEMA.COLUMNS c
            JOIN HOSPITALS.INFORMATION_SCHEMA.TABLES t ON t.TABLE_SCHEMA = c.TABLE_SCHEMA AND t.TABLE_NAME = c.TABLE_NAME
            WHERE ENDSWITH(c.TABLE_SCHEMA, '_CLEAN')
            GROUP BY 1, 2)
        SELECT sch, tbl, uk, rk, dc, vc, rc
        FROM tabs
        WHERE uk IS NOT NULL
          AND sch NOT IN (SELECT SPLIT_PART(SOURCE_OBJECT, '.', 2) FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES)
        QUALIFY ROW_NUMBER() OVER (PARTITION BY sch ORDER BY
            CASE WHEN REGEXP_LIKE(tbl, '(.*_)?(VISITS?|ENCOUNTERS?|ATTENDANCES?|OUTPATIENTS?)') THEN 3
                 WHEN CONTAINS(tbl, 'VISIT') OR CONTAINS(tbl, 'ENCOUNTER') THEN 2
                 WHEN CONTAINS(tbl, 'PATIENT') THEN 1 ELSE 0 END DESC,
            n_rows DESC NULLS LAST, tbl) = 1;
BEGIN
    FOR c IN candidates DO
        LET obj STRING := 'HOSPITALS.' || c.sch || '.' || c.tbl;
        LET uk STRING := c.uk;
        LET rk STRING := c.rk;
        LET dc STRING := c.dc;
        LET vc STRING := c.vc;
        LET rc STRING := c.rc;
        LET rows_n INTEGER := 0; LET upd_n INTEGER := 0; LET no_upd INTEGER := 0; LET key_n INTEGER := 0;
        LET profile VARIANT := NULL;
        BEGIN
            LET stmt STRING := 'SELECT COUNT(*), COUNT(DISTINCT ' || uk || '), COUNT_IF(' || uk || ' IS NULL), '
                            || IFF(rk IS NULL, '0', 'COUNT(DISTINCT ' || rk || ')') || ' FROM ' || obj;
            LET rs RESULTSET := (EXECUTE IMMEDIATE :stmt);
            LET c1 CURSOR FOR rs;
            OPEN c1;
            FETCH c1 INTO rows_n, upd_n, no_upd, key_n;
            CLOSE c1;
            -- versions are only deduplicated on a key that is (near) unique per record
            IF (rk IS NULL OR key_n < 0.99 * rows_n AND vc IS NULL) THEN
                rk := NULL;
            END IF;
            profile := OBJECT_CONSTRUCT('rows', rows_n, 'distinct_upd_key', upd_n, 'rows_without_upd_key', no_upd,
                                        'distinct_record_key', key_n, 'discovered_at', CURRENT_TIMESTAMP());
        EXCEPTION
            WHEN OTHER THEN
                profile := OBJECT_CONSTRUCT('error', LEFT(REGEXP_REPLACE(SQLERRM, '''[^'']*''', '''<value>'''), 300));
        END;
        INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
            (SOURCE, SOURCE_OBJECT, UPD_KEY, RECORD_KEY, DELETED_COLUMN, VERSION_COLUMN, RECEIVED_COLUMN, STATUS, PROFILE)
        SELECT :obj, :obj, :uk, :rk, :dc, IFF(:rk IS NULL, NULL, :vc), :rc, 'proposed', :profile;
        n_proposed := n_proposed + 1;
    END FOR;
    RETURN n_proposed || ' dataset(s) proposed';
END;
$$;

-- Discovers, then counts every active source into AFYA_PATIENT_COUNT_RUNS and checks the result. With
-- PUBLISH_RESULTS, updates AFYA_PATIENT_COUNTS and the receipts in one transaction: 'ok' sources are published,
-- 'review' sources only with ACCEPT_REVIEW, failed sources keep their previous figures.
-- FULL_RECOUNT = TRUE rescans unchanged base tables too.
CREATE PROCEDURE HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS(
    PUBLISH_RESULTS BOOLEAN DEFAULT FALSE,
    ACCEPT_REVIEW   BOOLEAN DEFAULT FALSE,
    FULL_RECOUNT    BOOLEAN DEFAULT FALSE)
RETURNS STRING
LANGUAGE SQL
EXECUTE AS OWNER
AS
$$
DECLARE
    logic_version STRING DEFAULT '2026-11-04.3';   -- bump when the counting SQL changes, to force recounts
    raw_main DATE; raw_last DATE; n_raw INTEGER;
    max_change    FLOAT  DEFAULT 0.20;   -- a figure moving more than 20% since the last publish needs review
    max_no_key    FLOAT  DEFAULT 0.05;   -- more than 5% of records without a UPD key needs review
    run_id        STRING        DEFAULT UUID_STRING();
    run_at        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP();
    n_ok INTEGER DEFAULT 0; n_review INTEGER DEFAULT 0; n_failed INTEGER DEFAULT 0;
    n_reused INTEGER DEFAULT 0; n_published INTEGER DEFAULT 0; n_proposed INTEGER DEFAULT 0;
    discovery_error STRING DEFAULT NULL;
    src_cte STRING; stmt STRING; notes STRING; status STRING; reused BOOLEAN;
    altered TIMESTAMP_LTZ; created DATE; n_want INTEGER; n_found INTEGER; n_have INTEGER;
    rows_read INTEGER; rows_superseded INTEGER; rows_deleted INTEGER; records INTEGER;
    upds INTEGER; no_uk INTEGER; p_rec INTEGER; p_upd INTEGER;
    bad_config     EXCEPTION (-20001, 'Invalid configuration in AFYA_PATIENT_COUNT_SOURCES');
    missing_column EXCEPTION (-20002, 'A configured column does not exist in SOURCE_OBJECT');
    sources CURSOR FOR
        SELECT SOURCE, SOURCE_OBJECT, UPD_KEY, RECORD_KEY, DELETED_COLUMN, VERSION_COLUMN, RECEIVED_COLUMN
        FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
        WHERE STATUS = 'active'
        ORDER BY SOURCE;
BEGIN
    BEGIN   -- a discovery failure must never stop the count refresh
        CALL HOSPITALS.ANALYTICS.AFYA_DISCOVER_PATIENT_DATASETS();
    EXCEPTION
        WHEN OTHER THEN
            discovery_error := LEFT(REGEXP_REPLACE(SQLERRM, '''[^'']*''', '''<value>'''), 300);
    END;
    SELECT COUNT(*) INTO :n_proposed FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES WHERE STATUS = 'proposed';

    FOR s IN sources DO
        LET src STRING := s.SOURCE;
        LET obj STRING := UPPER(TRIM(s.SOURCE_OBJECT));
        LET uk  STRING := UPPER(TRIM(s.UPD_KEY));
        LET rk  STRING := NULLIF(UPPER(TRIM(s.RECORD_KEY)), '');
        LET dc  STRING := NULLIF(UPPER(TRIM(s.DELETED_COLUMN)), '');
        LET vc  STRING := NULLIF(UPPER(TRIM(s.VERSION_COLUMN)), '');
        LET rc  STRING := NULLIF(UPPER(TRIM(s.RECEIVED_COLUMN)), '');
        LET sig STRING := logic_version || '|' || COALESCE(obj, '') || '|' || COALESCE(uk, '') || '|'
                       || COALESCE(rk, '') || '|' || COALESCE(dc, '') || '|' || COALESCE(vc, '');
        rows_read := NULL; rows_superseded := NULL; rows_deleted := NULL; records := NULL;
        upds := NULL; no_uk := NULL; altered := NULL; created := NULL; reused := FALSE; notes := '';
        raw_main := NULL; raw_last := NULL;
        BEGIN
            -- names are spliced into SQL, so only plain identifiers are accepted
            IF (NOT REGEXP_LIKE(COALESCE(obj, ''), '^[A-Z0-9_$]+[.][A-Z0-9_$]+[.][A-Z0-9_$]+$')
                OR (vc IS NOT NULL AND rk IS NULL)
                OR NOT REGEXP_LIKE(COALESCE(uk, ''), '^[A-Z_][A-Z0-9_$]*$')
                OR NOT REGEXP_LIKE(COALESCE(rk, 'X'), '^[A-Z_][A-Z0-9_$]*$')
                OR NOT REGEXP_LIKE(COALESCE(dc, 'X'), '^[A-Z_][A-Z0-9_$]*$')
                OR NOT REGEXP_LIKE(COALESCE(vc, 'X'), '^[A-Z_][A-Z0-9_$]*$')
                OR NOT REGEXP_LIKE(COALESCE(rc, 'X'), '^[A-Z_][A-Z0-9_$]*$')) THEN
                RAISE bad_config;
            END IF;

            LET db    STRING := SPLIT_PART(obj, '.', 1);
            LET sch   STRING := SPLIT_PART(obj, '.', 2);
            LET tname STRING := SPLIT_PART(obj, '.', 3);
            LET info_cols STRING := db || '.INFORMATION_SCHEMA.COLUMNS';
            LET info_tabs STRING := db || '.INFORMATION_SCHEMA.TABLES';

            SELECT ARRAY_SIZE(ARRAY_DISTINCT(ARRAY_CONSTRUCT_COMPACT(:uk, :rk, :dc, :vc, :rc))) INTO :n_want;
            SELECT COUNT(DISTINCT COLUMN_NAME) INTO :n_found
            FROM IDENTIFIER(:info_cols)
            WHERE TABLE_SCHEMA = :sch AND TABLE_NAME = :tname
              AND ARRAY_CONTAINS(COLUMN_NAME::VARIANT, ARRAY_CONSTRUCT_COMPACT(:uk, :rk, :dc, :vc, :rc));
            IF (n_found < n_want) THEN
                RAISE missing_column;
            END IF;

            -- views have no reliable change or load time, so they are always recounted
            SELECT MAX(IFF(TABLE_TYPE = 'BASE TABLE', LAST_ALTERED, NULL)),
                   MAX(IFF(TABLE_TYPE = 'BASE TABLE', CREATED, NULL))::DATE
            INTO :altered, :created
            FROM IDENTIFIER(:info_tabs)
            WHERE TABLE_SCHEMA = :sch AND TABLE_NAME = :tname;

            -- kept records are rn = 1 AND NOT is_del: latest version per record key, not soft-deleted
            src_cte := 'WITH src AS (SELECT '
                    || 'NULLIF(TRIM(' || uk || '::STRING), '''') AS uk, '
                    || IFF(dc IS NULL, 'FALSE', 'TRIM(COALESCE(' || dc || '::STRING, '''')) NOT IN '
                       || '('''', ''null'', ''NULL'', ''None'', ''none'', ''false'', ''FALSE'', ''0'')') || ' AS is_del, '
                    || IFF(rc IS NULL, 'NULL::DATE', 'TRY_TO_TIMESTAMP(' || rc || '::STRING)::DATE') || ' AS recv, '
                    || IFF(vc IS NULL, '1', 'CASE WHEN ' || rk || ' IS NULL THEN 1 ELSE ROW_NUMBER() OVER '
                       || '(PARTITION BY ' || rk || ' ORDER BY TRY_TO_TIMESTAMP(' || vc || '::STRING) DESC NULLS LAST) END')
                    || ' AS rn FROM ' || obj || ')';

            IF (NOT FULL_RECOUNT AND altered IS NOT NULL) THEN
                SELECT MAX_BY(ROWS_READ, RUN_AT), MAX_BY(ROWS_SUPERSEDED, RUN_AT), MAX_BY(ROWS_DELETED, RUN_AT),
                       MAX_BY(PATIENT_RECORDS, RUN_AT), MAX_BY(UPDS, RUN_AT), MAX_BY(RECORDS_NO_UPD_KEY, RUN_AT)
                INTO :rows_read, :rows_superseded, :rows_deleted, :records, :upds, :no_uk
                FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS
                WHERE SOURCE = :src AND CONFIG_SIG = :sig AND SOURCE_ALTERED = :altered AND STATUS <> 'failed';
            END IF;

            IF (rows_read IS NOT NULL) THEN
                reused := TRUE;
                n_reused := n_reused + 1;
            ELSE
                stmt := src_cte
                     || ' SELECT COUNT(*), COUNT_IF(rn > 1), COUNT_IF(rn = 1 AND is_del), COUNT_IF(rn = 1 AND NOT is_del), '
                     || 'COUNT(DISTINCT IFF(rn = 1 AND NOT is_del, uk, NULL)), '
                     || 'COUNT_IF(rn = 1 AND NOT is_del AND uk IS NULL) FROM src';
                LET rs RESULTSET := (EXECUTE IMMEDIATE :stmt);
                LET c1 CURSOR FOR rs;
                OPEN c1;
                FETCH c1 INTO rows_read, rows_superseded, rows_deleted, records, upds, no_uk;
                CLOSE c1;
            END IF;

            -- checks against impossible results and against the last published figures
            SELECT MAX_BY(PATIENT_RECORDS, RUN_AT), MAX_BY(UPDS, RUN_AT) INTO :p_rec, :p_upd
            FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS WHERE SOURCE = :src AND PUBLISHED;
            IF (records = 0) THEN notes := notes || 'no records; '; END IF;
            IF (upds > records) THEN notes := notes || 'UPDs exceed patient records; '; END IF;
            IF (records > 0 AND no_uk > max_no_key * records) THEN
                notes := notes || ROUND(100 * no_uk / records, 1) || '% of records have no UPD key; ';
            END IF;
            IF (p_rec > 0 AND ABS(records - p_rec) > max_change * p_rec) THEN
                notes := notes || 'patient records changed ' || ROUND(100 * (records - p_rec) / p_rec) || '%; ';
            END IF;
            IF (p_upd > 0 AND ABS(upds - p_upd) > max_change * p_upd) THEN
                notes := notes || 'UPDs changed ' || ROUND(100 * (upds - p_upd) / p_upd) || '%; ';
            END IF;
            status := IFF(notes = '', 'ok', 'review');

            -- first publish of a source with per-record load times: date its records day by day
            -- (before the run row, so a failure here leaves only a 'failed' row for this source)
            IF (PUBLISH_RESULTS AND rc IS NOT NULL AND (status = 'ok' OR ACCEPT_REVIEW)) THEN
                SELECT COUNT(*) INTO :n_have FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_RECORD_RECEIPTS WHERE SOURCE = :src;
                IF (n_have = 0) THEN
                    stmt := 'INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_RECORD_RECEIPTS '
                         || '(SOURCE, RECEIVED_DATE, RECORDS_ADDED, UPDS_ADDED, METHOD, RUN_ID, RECORDED_AT) '
                         || 'SELECT ?, recv, SUM(r), SUM(u), IFF(recv IS NULL, ''undated'', ''load time''), ?, ? '
                         || 'FROM (' || src_cte
                         || ', kept AS (SELECT uk, recv FROM src WHERE rn = 1 AND NOT is_del)'
                         || ', pat AS (SELECT MIN(recv) AS recv FROM kept WHERE uk IS NOT NULL GROUP BY uk) '
                         || 'SELECT recv, COUNT(*) AS r, 0 AS u FROM kept GROUP BY recv '
                         || 'UNION ALL SELECT recv, 0, COUNT(*) FROM pat GROUP BY recv) GROUP BY recv';
                    EXECUTE IMMEDIATE :stmt USING (src, run_id, run_at);
                END IF;
            END IF;

            -- raw events behind a *_CLEAN table: latest load every run; on first publish, the main load day
            -- (the day carrying most records; array events count their elements)
            LET raw_sch STRING := IFF(ENDSWITH(sch, '_CLEAN'), REGEXP_REPLACE(sch, '_CLEAN$', '_RAW'), NULL);
            LET raw_tbl STRING := LOWER(tname);
            n_raw := 0;
            BEGIN   -- a raw-table problem only loses the date, never the count
            IF (raw_sch IS NOT NULL AND rc IS NULL) THEN
                SELECT COUNT(*) INTO :n_raw FROM IDENTIFIER(:info_tabs) WHERE TABLE_SCHEMA = :raw_sch AND TABLE_NAME = 'EVENTS_RAW';
            END IF;
            IF (n_raw > 0) THEN
                LET raw_obj STRING := db || '.' || raw_sch || '.EVENTS_RAW';
                stmt := 'SELECT MAX(INGESTED_AT)::DATE FROM ' || raw_obj || ' WHERE SOURCE_TABLE = ?';
                LET rs2 RESULTSET := (EXECUTE IMMEDIATE :stmt USING (raw_tbl));
                LET c2 CURSOR FOR rs2;
                OPEN c2;
                FETCH c2 INTO raw_last;
                CLOSE c2;
                SELECT COUNT(*) INTO :n_have FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_RECORD_RECEIPTS WHERE SOURCE = :src;
                IF (PUBLISH_RESULTS AND raw_last IS NOT NULL AND n_have = 0) THEN
                    stmt := 'SELECT MAX_BY(d, n) FROM (SELECT INGESTED_AT::DATE AS d, '
                         || 'SUM(IFF(IS_ARRAY(PAYLOAD), ARRAY_SIZE(PAYLOAD), 1)) AS n FROM ' || raw_obj
                         || ' WHERE SOURCE_TABLE = ? GROUP BY 1)';
                    LET rs3 RESULTSET := (EXECUTE IMMEDIATE :stmt USING (raw_tbl));
                    LET c3 CURSOR FOR rs3;
                    OPEN c3;
                    FETCH c3 INTO raw_main;
                    CLOSE c3;
                END IF;
            END IF;
            EXCEPTION
                WHEN OTHER THEN
                    raw_main := NULL;
                    raw_last := NULL;
            END;

            INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS
                (RUN_ID, RUN_AT, SOURCE, CONFIG_SIG, SOURCE_ALTERED, SOURCE_CREATED, RAW_MAIN_LOAD, RAW_LAST_LOAD,
                 ROWS_READ, ROWS_SUPERSEDED, ROWS_DELETED, PATIENT_RECORDS, UPDS, RECORDS_NO_UPD_KEY, STATUS,
                 CHECK_NOTES, REUSED)
            VALUES (:run_id, :run_at, :src, :sig, :altered, :created, :raw_main, :raw_last, :rows_read,
                    :rows_superseded, :rows_deleted, :records, :upds, :no_uk, :status, NULLIF(:notes, ''), :reused);
            IF (status = 'ok') THEN n_ok := n_ok + 1; ELSE n_review := n_review + 1; END IF;
        EXCEPTION
            WHEN OTHER THEN
                -- quoted values in Snowflake errors can be source data, so they are masked before storing
                LET msg STRING := LEFT(REGEXP_REPLACE(SQLERRM, '''[^'']*''', '''<value>'''), 500);
                n_failed := n_failed + 1;
                INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS (RUN_ID, RUN_AT, SOURCE, CONFIG_SIG, STATUS, ERROR)
                VALUES (:run_id, :run_at, :src, :sig, 'failed', :msg);
        END;
    END FOR;

    IF (PUBLISH_RESULTS) THEN
        BEGIN
            BEGIN TRANSACTION;
            MERGE INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS t
            USING (SELECT SOURCE, PATIENT_RECORDS, UPDS FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS
                   WHERE RUN_ID = :run_id AND (STATUS = 'ok' OR (STATUS = 'review' AND :ACCEPT_REVIEW))) r
            ON t.SOURCE = r.SOURCE
            WHEN MATCHED THEN UPDATE SET
                CHANGED_AT = IFF(t.PATIENT_RECORDS IS DISTINCT FROM r.PATIENT_RECORDS
                                 OR t.UPDS IS DISTINCT FROM r.UPDS, :run_at, t.CHANGED_AT),
                PATIENT_RECORDS = r.PATIENT_RECORDS, UPDS = r.UPDS, RUN_ID = :run_id, COUNTED_AT = :run_at
            WHEN NOT MATCHED THEN INSERT (SOURCE, PATIENT_RECORDS, UPDS, RUN_ID, COUNTED_AT, CHANGED_AT)
            VALUES (r.SOURCE, r.PATIENT_RECORDS, r.UPDS, :run_id, :run_at, :run_at);
            UPDATE HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS SET PUBLISHED = TRUE
            WHERE RUN_ID = :run_id AND (STATUS = 'ok' OR (STATUS = 'review' AND :ACCEPT_REVIEW));
            DELETE FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS
            WHERE SOURCE NOT IN (SELECT SOURCE FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES WHERE STATUS = 'active');

            -- receipts catch up to the published figures. A source's first entry is dated to its main raw load,
            -- else its table's creation, else the run; later differences go to the latest raw load if it is new,
            -- else the run date
            INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_RECORD_RECEIPTS
                (SOURCE, RECEIVED_DATE, RECORDS_ADDED, UPDS_ADDED, METHOD, RUN_ID, RECORDED_AT)
            SELECT r.SOURCE,
                   CASE WHEN x.SOURCE IS NULL THEN COALESCE(r.RAW_MAIN_LOAD, r.SOURCE_CREATED, :run_at::DATE)
                        WHEN r.RAW_LAST_LOAD > x.LAST_DATE THEN r.RAW_LAST_LOAD
                        ELSE :run_at::DATE END,
                   r.PATIENT_RECORDS - COALESCE(x.RECORDS, 0), r.UPDS - COALESCE(x.UPDS, 0),
                   CASE WHEN x.SOURCE IS NULL THEN CASE WHEN r.RAW_MAIN_LOAD IS NOT NULL THEN 'raw event load date'
                                                        WHEN r.SOURCE_CREATED IS NOT NULL THEN 'table created'
                                                        ELSE 'first counted' END
                        WHEN r.RAW_LAST_LOAD > x.LAST_DATE THEN 'raw event load date'
                        ELSE 'run increment' END,
                   :run_id, :run_at
            FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS r
            LEFT JOIN (SELECT SOURCE, SUM(RECORDS_ADDED) AS RECORDS, SUM(UPDS_ADDED) AS UPDS,
                              MAX(RECEIVED_DATE) AS LAST_DATE
                       FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_RECORD_RECEIPTS GROUP BY SOURCE) x ON x.SOURCE = r.SOURCE
            WHERE r.RUN_ID = :run_id AND r.PUBLISHED
              AND (r.PATIENT_RECORDS <> COALESCE(x.RECORDS, 0) OR r.UPDS <> COALESCE(x.UPDS, 0));
            COMMIT;
        EXCEPTION
            WHEN OTHER THEN
                ROLLBACK;
                RAISE;
        END;
        SELECT COUNT_IF(PUBLISHED) INTO :n_published
        FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS WHERE RUN_ID = :run_id;
    END IF;

    RETURN 'Run ' || run_id || ': ' || n_ok || ' ok, ' || n_review || ' to review, ' || n_failed || ' failed, '
        || n_reused || ' reused; ' || n_proposed || ' proposed dataset(s) awaiting approval; '
        || IFF(PUBLISH_RESULTS, n_published || ' published', 'dry run, dashboard not updated')
        || IFF(discovery_error IS NULL, '', '; discovery failed: ' || discovery_error);
END;
$$;

-- Dry run (discovers, counts and checks; dashboard and receipts untouched):
-- CALL HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS();

-- Latest run per source:
-- SELECT SOURCE, STATUS, PATIENT_RECORDS, UPDS, CHECK_NOTES, ERROR
-- FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS
-- WHERE RUN_ID = (SELECT MAX_BY(RUN_ID, RUN_AT) FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_RUNS) ORDER BY SOURCE;

-- Proposed datasets, then approve (name it for the dashboard) or reject one:
-- SELECT SOURCE_OBJECT, UPD_KEY, RECORD_KEY, DELETED_COLUMN, VERSION_COLUMN, RECEIVED_COLUMN, PROFILE
-- FROM HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES WHERE STATUS = 'proposed';
-- UPDATE HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
-- SET STATUS = 'active', SOURCE = '<Dashboard name>', UPDATED_AT = CURRENT_TIMESTAMP() WHERE SOURCE_OBJECT = '<object>';
-- UPDATE HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES
-- SET STATUS = 'rejected', UPDATED_AT = CURRENT_TIMESTAMP() WHERE SOURCE_OBJECT = '<object>';

-- Dashboard table (create once), then publish:
-- CREATE OR REPLACE TRANSIENT TABLE HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS (
--     SOURCE STRING NOT NULL, PATIENT_RECORDS INTEGER, UPDS INTEGER,
--     RUN_ID STRING, COUNTED_AT TIMESTAMP_NTZ, CHANGED_AT TIMESTAMP_NTZ,
--     CONSTRAINT PK_AFYA_PATIENT_COUNTS PRIMARY KEY (SOURCE)
-- ) DATA_RETENTION_TIME_IN_DAYS = 0;
-- CALL HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS(PUBLISH_RESULTS => TRUE);

-- Weekly schedule (needs CREATE TASK here and EXECUTE TASK on the account); ~0.1 X-Small credit per run.
-- CREATE TASK HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS_WEEKLY
--   WAREHOUSE = COMPUTE_WH
--   SCHEDULE  = 'USING CRON 0 5 * * MON Africa/Nairobi'
-- AS CALL HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS(PUBLISH_RESULTS => TRUE);
-- ALTER TASK HOSPITALS.ANALYTICS.AFYA_REFRESH_PATIENT_COUNTS_WEEKLY RESUME;
