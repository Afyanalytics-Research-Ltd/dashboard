"""Executable reference of the CURRENT counting semantics of AFYA_REFRESH_PATIENT_COUNTS.

It mirrors, statement for statement, the src CTE and the per-source statistics (snowflake_patient_counts.sql
L231-256) and the receipts logic (L280-295 first-publish load-time receipts; L367-388 catch-up receipts),
translated to SQLite / Python. Its only purpose is to pin down today's behaviour (including quirks) so the
discovery redesign can be shown not to change it. It is not a re-implementation to be deployed.

Translation notes:
  x::STRING                -> CAST(x AS TEXT)
  TRY_TO_TIMESTAMP(s)      -> try_ts(s): ISO 'YYYY-MM-DD[ T]HH:MM[:SS[.fff]]', or an integer epoch interpreted by
                              magnitude (s / ms / us / ns) as documented for TO_TIMESTAMP; else NULL
  COUNT_IF(c)              -> SUM(CASE WHEN c THEN 1 ELSE 0 END)
  IFF(c, a, b)             -> CASE WHEN c THEN a ELSE b END
"""
import sqlite3
from datetime import datetime, timezone

NOT_DELETED = ("", "null", "NULL", "None", "none", "false", "FALSE", "0")   # L234-235


def try_ts(s):
    if s is None:
        return None
    s = str(s).strip()
    if s.lstrip("-").isdigit():
        n = int(s)
        for div in (1, 1e3, 1e6, 1e9):          # seconds, ms, us, ns by magnitude
            if abs(n / div) < 31536000000:      # < year ~3000 in seconds
                return datetime.fromtimestamp(n / div, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def connect():
    conn = sqlite3.connect(":memory:")
    conn.create_function("try_ts", 1, try_ts)
    return conn


def q(c):
    return '"' + c.replace('"', '""') + '"'


def src_cte(obj, uk, rk=None, dc=None, vc=None, rc=None):
    """L232-239 verbatim in structure."""
    is_del = "0" if dc is None else f"TRIM(COALESCE(CAST({q(dc)} AS TEXT), '')) NOT IN {NOT_DELETED}"
    recv = "NULL" if rc is None else f"date(try_ts(CAST({q(rc)} AS TEXT)))"
    rn = "1" if vc is None else (f"CASE WHEN {q(rk)} IS NULL THEN 1 ELSE ROW_NUMBER() OVER "
                                 f"(PARTITION BY {q(rk)} ORDER BY try_ts(CAST({q(vc)} AS TEXT)) DESC NULLS LAST) END")
    return (f"WITH src AS (SELECT NULLIF(TRIM(CAST({q(uk)} AS TEXT)), '') AS uk, ({is_del}) AS is_del, "
            f"{recv} AS recv, {rn} AS rn FROM {q(obj)})")


def count_source(conn, obj, uk, rk=None, dc=None, vc=None, rc=None):
    """L253-256: rows_read, rows_superseded, rows_deleted, records, upds, no_uk."""
    if vc is not None and rk is None:
        raise ValueError("bad_config: VERSION_COLUMN requires RECORD_KEY (L200)")
    sql = src_cte(obj, uk, rk, dc, vc, rc) + """
SELECT COUNT(*),
       SUM(CASE WHEN rn > 1 THEN 1 ELSE 0 END),
       SUM(CASE WHEN rn = 1 AND is_del THEN 1 ELSE 0 END),
       SUM(CASE WHEN rn = 1 AND NOT is_del THEN 1 ELSE 0 END),
       COUNT(DISTINCT CASE WHEN rn = 1 AND NOT is_del THEN uk END),
       SUM(CASE WHEN rn = 1 AND NOT is_del AND uk IS NULL THEN 1 ELSE 0 END)
FROM src"""
    keys = ["rows_read", "rows_superseded", "rows_deleted", "records", "upds", "no_uk"]
    return dict(zip(keys, [v or 0 for v in conn.execute(sql).fetchone()]))


def checks(records, upds, no_uk, p_rec=None, p_upd=None, max_change=0.20, max_no_key=0.05):
    """L264-278. Returns (status, notes)."""
    notes = ""
    if records == 0:
        notes += "no records; "
    if upds > records:
        notes += "UPDs exceed patient records; "
    if records > 0 and no_uk > max_no_key * records:
        notes += f"{round(100 * no_uk / records, 1)}% of records have no UPD key; "
    if p_rec and abs(records - p_rec) > max_change * p_rec:
        notes += f"patient records changed {round(100 * (records - p_rec) / p_rec)}%; "
    if p_upd and abs(upds - p_upd) > max_change * p_upd:
        notes += f"UPDs changed {round(100 * (upds - p_upd) / p_upd)}%; "
    return ("ok" if notes == "" else "review"), notes


def load_time_receipts(conn, obj, uk, rk=None, dc=None, vc=None, rc=None):
    """L285-292: first publish of a source with RECEIVED_COLUMN. Records by receive day; each UPD on the day of its
    earliest kept record. Returns {date: (records, upds)}."""
    sql = src_cte(obj, uk, rk, dc, vc, rc) + """
, kept AS (SELECT uk, recv FROM src WHERE rn = 1 AND NOT is_del)
, pat AS (SELECT MIN(recv) AS recv FROM kept WHERE uk IS NOT NULL GROUP BY uk)
SELECT recv, SUM(r), SUM(u) FROM (
    SELECT recv, COUNT(*) AS r, 0 AS u FROM kept GROUP BY recv
    UNION ALL SELECT recv, 0, COUNT(*) FROM pat GROUP BY recv) GROUP BY recv"""
    return {d: (r, u) for d, r, u in conn.execute(sql).fetchall()}


class Receipts:
    """L367-388 catch-up: after each publish, one row per source whose published totals differ from the receipts."""

    def __init__(self):
        self.rows = []      # (source, date, records_added, upds_added, method)

    def totals(self, source):
        mine = [r for r in self.rows if r[0] == source]
        return (sum(r[2] for r in mine), sum(r[3] for r in mine), max((r[1] for r in mine if r[1]), default=None)) \
            if mine else None

    def publish(self, source, records, upds, run_date, raw_main=None, raw_last=None, created=None):
        x = self.totals(source)
        if x and records == x[0] and upds == x[1]:
            return None
        if x is None:
            d = raw_main or created or run_date
            m = "raw event load date" if raw_main else ("table created" if created else "first counted")
        elif raw_last and x[2] and raw_last > x[2]:
            d, m = raw_last, "raw event load date"
        else:
            d, m = run_date, "run increment"
        row = (source, d, records - (x[0] if x else 0), upds - (x[1] if x else 0), m)
        self.rows.append(row)
        return row
