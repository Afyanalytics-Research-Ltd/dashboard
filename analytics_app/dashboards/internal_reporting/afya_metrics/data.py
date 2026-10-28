"""Snowflake storage: HOSPITALS.ANALYTICS.AFYA_WEEKLY_METRICS and AFYA_PATIENT_COUNTS."""

import os
import sys
from datetime import date, datetime, timedelta

import streamlit as st

SF_DATABASE = os.environ.get("AFYA_METRICS_SF_DATABASE", "HOSPITALS").strip()
SF_SCHEMA = os.environ.get("AFYA_METRICS_SF_SCHEMA", "ANALYTICS").strip()
SF_FQN = f"{SF_DATABASE}.{SF_SCHEMA}.AFYA_WEEKLY_METRICS"
PC_FQN = f"{SF_DATABASE}.{SF_SCHEMA}.AFYA_PATIENT_COUNTS"
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
    """Sources, largest first, and the last-updated date."""
    recs = _fetch(f"SELECT * FROM {PC_FQN} ORDER BY PATIENTS DESC")
    rows = [{"SOURCE": str(r["SOURCE"]).strip(), "PATIENTS": _num(r["PATIENTS"]),
             "UNIQUE_PATIENTS": _num(r["UNIQUE_PATIENTS"])} for r in recs]
    return rows, _updated(recs)


def save_patient_counts(upserts, deletes=()):
    stmts = [(f"DELETE FROM {PC_FQN} WHERE SOURCE = %s", (src,))
             for src in list(deletes) + [r["SOURCE"] for r in upserts]]
    stmts += [(f"INSERT INTO {PC_FQN} (SOURCE, PATIENTS, UNIQUE_PATIENTS) VALUES (%s, %s, %s)",
               (r["SOURCE"], r["PATIENTS"], r["UNIQUE_PATIENTS"])) for r in upserts]
    _write(stmts)
    read_patient_counts.clear()
