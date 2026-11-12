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


@st.cache_resource(show_spinner=False)
def _reader():
    """One Snowflake connection per process for reads: a new connection takes 1-3 s, a query on an open one well
    under a second. Writes keep their own connection (_write) so their transactions never share it."""
    return _client()


def _run(conn, sql):
    cur = conn.cursor()
    try:
        # JSON result format: the Arrow reader DLL can be blocked by endpoint policy.
        cur.execute("ALTER SESSION SET PYTHON_CONNECTOR_QUERY_RESULT_FORMAT = 'JSON'")
        cur.execute(sql)
        names = [d[0].upper() for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]
    finally:
        cur.close()


def _fetch(sql):
    client = _reader()
    try:
        if client.conn.is_closed():
            raise ConnectionError("closed")
        return _run(client.conn, sql)
    except Exception as exc:    # closed, or idle long enough for the session to expire: reconnect once and retry
        if type(exc).__name__ == "ProgrammingError" and getattr(exc, "errno", None) != 390114:
            raise               # a problem with the query itself: reconnecting would not help
        try:
            client.conn.close()
        except Exception:
            pass
        client.conn = client._connect()
        return _run(client.conn, sql)


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
    from . import hmis as H
    return {r["ENVIRONMENT"]: r["LAST"] for r in _fetch(
        f"SELECT ENVIRONMENT, CONVERT_TIMEZONE('UTC', '{H.TZ}', MAX(CAPTURED_AT)) AS LAST FROM {HM_FQN} GROUP BY 1")}


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_hmis(env):
    """The latest run of environment `env` and the last run at least a day older (WHICH = 'latest' | 'previous'),
    facility and unassigned rows only, with CAPTURED_AT in East Africa Time. `env` must be a name returned by
    hmis.resolve."""
    from . import hmis as H
    if not env or not H._SAFE.match(str(env)):
        return []
    recs = _fetch(f"""
        WITH runs AS (SELECT RUN_ID, MAX(CAPTURED_AT) AS AT FROM {HM_FQN} WHERE ENVIRONMENT = '{env}' GROUP BY 1),
        latest AS (SELECT RUN_ID, AT FROM runs QUALIFY ROW_NUMBER() OVER (ORDER BY AT DESC) = 1),
        previous AS (SELECT r.RUN_ID FROM runs r JOIN latest l ON r.AT <= DATEADD('hour', -24, l.AT)
                     QUALIFY ROW_NUMBER() OVER (ORDER BY r.AT DESC) = 1)
        SELECT IFF(t.RUN_ID IN (SELECT RUN_ID FROM latest), 'latest', 'previous') AS WHICH,
               CONVERT_TIMEZONE('UTC', '{H.TZ}', t.CAPTURED_AT) AS CAPTURED_AT,
               t.ORGANIZATION_ID, t.ORGANIZATION_NAME, t.FACILITY_ID, t.FACILITY_NAME, t.PATIENT_COUNT, t.SCOPE,
               t.COUNT_METHOD, t.VISIT_COUNT
        FROM {HM_FQN} t
        WHERE t.ENVIRONMENT = '{env}' AND t.SCOPE IN ('facility', 'unassigned')
          AND t.RUN_ID IN (SELECT RUN_ID FROM latest UNION ALL SELECT RUN_ID FROM previous)""")
    return [{**r, "PATIENT_COUNT": _num(r["PATIENT_COUNT"]), "SCOPE": str(r["SCOPE"]).lower(),
             "COUNT_METHOD": str(r["COUNT_METHOD"]).strip(),
             "VISIT_COUNT": None if r.get("VISIT_COUNT") is None else _num(r["VISIT_COUNT"])} for r in recs]


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_hmis_daily(env):
    """Each facility's patients as of the end of each day (East Africa Time) it was posted (the day's last run), with the change since
    its previous day: the first day carries the whole count. Summed over the days, the changes give the latest
    count, so dated arrivals and the running total agree."""
    from . import hmis as H
    if not env or not H._SAFE.match(str(env)):
        return []
    methods = ", ".join("'" + m.replace("'", "") + "'" for m in sorted(H.METHODS))
    recs = _fetch(f"""
        WITH daily AS (
            SELECT ORGANIZATION_ID, FACILITY_ID, CONVERT_TIMEZONE('UTC', '{H.TZ}', CAPTURED_AT)::DATE AS DAY, MAX_BY(PATIENT_COUNT, CAPTURED_AT) AS N,
                   COALESCE(MAX_BY(VISIT_COUNT, CAPTURED_AT), 0) AS V
            FROM {HM_FQN}
            WHERE ENVIRONMENT = '{env}' AND SCOPE = 'facility' AND FACILITY_ID IS NOT NULL
              AND COUNT_METHOD IN ({methods})
            GROUP BY 1, 2, 3)
        SELECT ORGANIZATION_ID, FACILITY_ID, DAY,
               N - COALESCE(LAG(N) OVER (PARTITION BY ORGANIZATION_ID, FACILITY_ID ORDER BY DAY), 0) AS ADDED,
               V - COALESCE(LAG(V) OVER (PARTITION BY ORGANIZATION_ID, FACILITY_ID ORDER BY DAY), 0) AS VISITS_ADDED
        FROM daily""")
    return [{"ORGANIZATION_ID": r["ORGANIZATION_ID"], "FACILITY_ID": r["FACILITY_ID"], "DATE": _date(r["DAY"]),
             "ADDED": _num(r["ADDED"]), "VISITS_ADDED": _num(r["VISITS_ADDED"])} for r in recs]


@st.cache_data(ttl=SF_TTL, show_spinner=False)
def read_receipts():
    """What arrived per source and received date (None = undated)."""
    recs = _fetch(f"SELECT SOURCE, RECEIVED_DATE, SUM(RECORDS_ADDED) AS RECORDS, SUM(UPDS_ADDED) AS UPDS "
                  f"FROM {RC_FQN} GROUP BY 1, 2 ORDER BY 2 NULLS FIRST")
    return [{"SOURCE": str(r["SOURCE"]).strip(), "DATE": _date(r["RECEIVED_DATE"]), "RECORDS": _num(r["RECORDS"]),
             "UPDS": _num(r["UPDS"])} for r in recs]
