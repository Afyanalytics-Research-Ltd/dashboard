"""Snowflake storage: HOSPITALS.ANALYTICS.AFYA_WEEKLY_METRICS, AFYA_PATIENT_COUNTS and AFYA_PATIENT_RECORD_RECEIPTS."""

import os
import sys
from datetime import date, datetime, timedelta

import streamlit as st

SF_DATABASE = os.environ.get("AFYA_METRICS_SF_DATABASE", "HOSPITALS").strip()
SF_SCHEMA = os.environ.get("AFYA_METRICS_SF_SCHEMA", "ANALYTICS").strip()
SF_FQN = f"{SF_DATABASE}.{SF_SCHEMA}.AFYA_WEEKLY_METRICS"
# Published by the AFYA_REFRESH_PATIENT_COUNTS procedure (snowflake_patient_counts.sql); read-only here.
PC_FQN = f"{SF_DATABASE}.{SF_SCHEMA}.AFYA_PATIENT_COUNTS"
RC_FQN = f"{SF_DATABASE}.{SF_SCHEMA}.AFYA_PATIENT_RECORD_RECEIPTS"
# Posted by the HMIS (one row per facility per run); read-only here. See hmis.py.
HM_FQN = f"{SF_DATABASE}.{SF_SCHEMA}.AFYA_HMIS_PATIENT_COUNTS"
SF_TTL = int(os.environ.get("AFYA_METRICS_TTL", "600"))

DASH_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLS = [
    "PERIOD", "WEEK_ENDING",
    "TEAM_FULL_TIME", "TEAM_CONSULTANTS",
    "UPDS_TARGET", "UPDS_PROCESSED",
    "MOU_TOTAL_COUNTIES", "MOU_SIGNED",
    "FACILITIES_TARGET", "FACILITIES_PRIVATE",
    "DATA_RECEIVED", "DATA_PROCESSED",
    "REVENUE_TARGET", "REVENUE_BOOKED",
    "APPRENTICES_TARGET", "APPRENTICES_ONBOARDED",
]
NUM_COLS = COLS[2:]


def _num(x):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return 0


def _date(v):
    if isinstance(v, datetime):
        return v.date()
    if hasattr(v, "to_pydatetime"):
        return v.to_pydatetime().date()
    return v


def period_for(d):
    """date -> 'Q3 2026'."""
    return f"Q{(d.month - 1) // 3 + 1} {d.year}"


def period_bounds(period):
    """'Q3 2026' -> (1 Jul 2026, 30 Sep 2026); (None, None) if not a quarter label."""
    try:
        q, y = int(period.strip().upper()[1]), int(period.split()[-1])
        start = date(y, 3 * q - 2, 1)
        end = date(y + (q == 4), (3 * q) % 12 + 1, 1) - timedelta(days=1)
        return start, end
    except (ValueError, IndexError):
        return None, None


def next_week_ending(rows):
    return rows[-1]["WEEK_ENDING"] + timedelta(days=7) if rows else date.today()


def _client():
    if DASH_DIR not in sys.path:
        sys.path.insert(0, DASH_DIR)
    from snowflake_service.snowflake_client import SnowflakeClient
    return SnowflakeClient()


def _fetch(sql):
    # JSON result format: the Arrow reader DLL can be blocked by endpoint policy.
    cur = _client().conn.cursor()
    try:
        cur.execute("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'JSON'")
        cur.execute(sql)
        names = [d[0].upper() for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]
    finally:
        cur.close()


def _write(statements):
    """Run (sql, params) pairs in one transaction."""
    cur = _client().conn.cursor()
    try:
        cur.execute("BEGIN")
        for sql, params in statements:
            cur.execute(sql, params)
        cur.execute("COMMIT")
    except Exception:
        cur.execute("ROLLBACK")
        raise
    finally:
        cur.close()


def _updated(recs):
    stamps = [r["UPDATED_AT"] for r in recs if r.get("UPDATED_AT")]
    return _date(max(stamps)) if stamps else None


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_rows():
    """All weeks, oldest first, and the last-updated date."""
    recs = _fetch(f"SELECT * FROM {SF_FQN} ORDER BY WEEK_ENDING")
    rows = []
    for rec in recs:
        row = {"PERIOD": str(rec["PERIOD"]).strip(), "WEEK_ENDING": _date(rec["WEEK_ENDING"])}
        row.update({c: _num(rec.get(c)) for c in NUM_COLS})
        rows.append(row)
    return rows, _updated(recs)


def upsert(row):
    """Insert or replace one week."""
    _write([
        (f"DELETE FROM {SF_FQN} WHERE PERIOD = %s AND WEEK_ENDING = %s", (row["PERIOD"], row["WEEK_ENDING"])),
        (f"INSERT INTO {SF_FQN} ({', '.join(COLS)}) VALUES ({', '.join(['%s'] * len(COLS))})",
         tuple(row[c] for c in COLS)),
    ])
    read_rows.clear()


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_patient_counts():
    """Sources, largest first, and the date of the latest count."""
    recs = _fetch(f"SELECT SOURCE, PATIENT_RECORDS, UPDS, COUNTED_AT FROM {PC_FQN} ORDER BY PATIENT_RECORDS DESC")
    rows = [{"SOURCE": str(r["SOURCE"]).strip(), "PATIENT_RECORDS": _num(r["PATIENT_RECORDS"]),
             "UPDS": _num(r["UPDS"]), "COUNTED_AT": _date(r["COUNTED_AT"])} for r in recs]
    stamps = [r["COUNTED_AT"] for r in rows if r["COUNTED_AT"]]
    return rows, max(stamps) if stamps else None


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_hmis_environments():
    """{environment: last captured} in the HMIS counts table."""
    return {r["ENVIRONMENT"]: r["LAST"] for r in _fetch(f"SELECT ENVIRONMENT, MAX(CAPTURED_AT) AS LAST FROM {HM_FQN} "
                                                        f"GROUP BY 1")}


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_hmis(env):
    """The latest run of environment `env` and the last run at least a day older (WHICH = 'latest' | 'previous'),
    facility and unassigned rows only. `env` must be a name returned by hmis.resolve / pick_environment."""
    from . import hmis as H
    if not env or not H._SAFE.match(str(env)):
        return []
    recs = _fetch(f"""
        WITH runs AS (SELECT RUN_ID, MAX(CAPTURED_AT) AS AT FROM {HM_FQN} WHERE ENVIRONMENT = '{env}' GROUP BY 1),
        latest AS (SELECT RUN_ID, AT FROM runs QUALIFY ROW_NUMBER() OVER (ORDER BY AT DESC) = 1),
        previous AS (SELECT r.RUN_ID FROM runs r JOIN latest l ON r.AT <= DATEADD('hour', -24, l.AT)
                     QUALIFY ROW_NUMBER() OVER (ORDER BY r.AT DESC) = 1)
        SELECT IFF(t.RUN_ID IN (SELECT RUN_ID FROM latest), 'latest', 'previous') AS WHICH, t.CAPTURED_AT,
               t.ORGANIZATION_ID, t.ORGANIZATION_NAME, t.FACILITY_ID, t.FACILITY_NAME, t.PATIENT_COUNT, t.SCOPE,
               t.COUNT_METHOD
        FROM {HM_FQN} t
        WHERE t.ENVIRONMENT = '{env}' AND t.SCOPE IN ('facility', 'unassigned')
          AND t.RUN_ID IN (SELECT RUN_ID FROM latest UNION ALL SELECT RUN_ID FROM previous)""")
    return [{**r, "PATIENT_COUNT": _num(r["PATIENT_COUNT"]), "SCOPE": str(r["SCOPE"]).lower(),
             "COUNT_METHOD": str(r["COUNT_METHOD"]).strip()} for r in recs]


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_receipts():
    """What arrived per source and received date (None = undated)."""
    recs = _fetch(f"SELECT SOURCE, RECEIVED_DATE, SUM(RECORDS_ADDED) AS RECORDS, SUM(UPDS_ADDED) AS UPDS "
                  f"FROM {RC_FQN} GROUP BY 1, 2 ORDER BY 2 NULLS FIRST")
    return [{"SOURCE": str(r["SOURCE"]).strip(), "DATE": _date(r["RECEIVED_DATE"]), "RECORDS": _num(r["RECORDS"]),
             "UPDS": _num(r["UPDS"])} for r in recs]
