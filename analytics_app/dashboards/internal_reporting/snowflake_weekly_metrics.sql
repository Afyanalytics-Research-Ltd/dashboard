-- Afyanalytics internal metrics tables (one-time setup). Creates only the AFYA_* tables in
-- HOSPITALS.ANALYTICS;

USE ROLE DATAANALYSTS;

-- One row per reporting week; figures are running totals.
CREATE TABLE HOSPITALS.ANALYTICS.AFYA_WEEKLY_METRICS (
    PERIOD                 STRING        NOT NULL,   -- e.g. 'Q3 2026'
    WEEK_ENDING            DATE          NOT NULL,
    TEAM_FULL_TIME         INTEGER,
    TEAM_CONSULTANTS       INTEGER,
    UPDS_TARGET            INTEGER,
    UPDS_PROCESSED         INTEGER,
    MOU_TOTAL_COUNTIES     INTEGER,
    MOU_SIGNED             INTEGER,
    FACILITIES_TARGET      INTEGER,
    FACILITIES_PRIVATE     INTEGER,
    DATA_RECEIVED          INTEGER,
    DATA_PROCESSED         INTEGER,
    REVENUE_TARGET         NUMBER(18,2),
    REVENUE_BOOKED         NUMBER(18,2),
    APPRENTICES_TARGET     INTEGER,
    APPRENTICES_ONBOARDED  INTEGER,
    UPDATED_AT             TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_AFYA_WEEKLY_METRICS PRIMARY KEY (PERIOD, WEEK_ENDING)
);

-- Patients per data source (current snapshot, one row per source).
CREATE TABLE HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS (
    SOURCE           STRING  NOT NULL,
    PATIENTS         INTEGER,
    UNIQUE_PATIENTS  INTEGER,
    UPDATED_AT       TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_AFYA_PATIENT_COUNTS PRIMARY KEY (SOURCE)
);

-- Seed
INSERT INTO HOSPITALS.ANALYTICS.AFYA_WEEKLY_METRICS
    (PERIOD, WEEK_ENDING, TEAM_FULL_TIME, TEAM_CONSULTANTS, UPDS_TARGET, UPDS_PROCESSED,
     MOU_TOTAL_COUNTIES, MOU_SIGNED, FACILITIES_TARGET, FACILITIES_PRIVATE,
     DATA_RECEIVED, DATA_PROCESSED, REVENUE_TARGET, REVENUE_BOOKED,
     APPRENTICES_TARGET, APPRENTICES_ONBOARDED)
VALUES
    ('Q3 2026', '2026-07-10', 11, 11, 3000000, 1053372, 14, 8, 200, 13,
     1139547, 1053372, 300000, 0, 100, 33);

INSERT INTO HOSPITALS.ANALYTICS.AFYA_PATIENT_COUNTS (SOURCE, PATIENTS, UNIQUE_PATIENTS) VALUES
    ('Kisumu Specialist Hospital (KSH)',      40532,  13769),
    ('Tenri',                                696743, 248785),
    ('Lodwar',                               158878, 104627),
    ('Kakamega',                              72077,  43894),
    ('St. Peter''s Orthopedics Hospital (SPH)', 423386, 94972),
    ('Nandi',                                158878, 158878),
    ('Migori HIV',                             1287,   1287),
    ('Mother Francisca historical data',      12165,  12165),
    ('Silverwood',                               18,     18),
    ('Siaya medical camp',                      917,    915),
    ('Kakamega HIV data',                      6894,   6536),
    ('Mother Francisca medical camp',          5000,   3032);
