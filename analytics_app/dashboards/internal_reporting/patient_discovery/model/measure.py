"""Aggregate measurements. Every query returns counts, shares, min/max dates or value shapes (letters -> A, digits
-> 9) - never an identifier value. Identifier columns are compared after the same normalisation the profile uses.

Statements per measurement (each is one scan of each object it names):
  screen       1 scan: rows; record-ID count and distinct; date range; exact-duplicate rows (no record ID only; the
               row hash leaves out personal columns);
               per candidate: non-missing, distinct, distinct (record, value) pairs, rows where value = record ID
  profile      1-2 scans: the existing per-value profile (profile_sql) for one candidate
  pair         3 scans: distinct a, b, (a, b); max and count of a with several b; and the reverse
  shape        2 scans: the 3 most common value shapes and the total
  containment  A: 1 scan, B: 2 scans: distinct A, distinct A found in B (left join of distinct values); distinct B
  window       1 scan: distinct values up to a date (reconciling a new version with its predecessor)
  categories   1 scan: distinct counts of short text columns (to find grouping columns)
  conditional  1 scan: rows in groups where the candidate is almost always present, and where almost never
  align        A, B: 2 scans each: records per side, records shared by record ID, and how many shared records
               carry the same identifier value (agree) or a different one (disagree)
  formats      A: 1 scan, B: 1 scan: distinct values of A, and how many have a format (length and character
               classes; 'num' for numbers) that occurs among B's values

The SQLite dialect runs the tests; the Snowflake dialect is compiled with `python -m patient_discovery.run compile`.
"""
import re

from .. import profile_sql as P

# part of every observation's fingerprint: a change to any measurement's SQL or normalisation re-measures
MEASURE_VERSION = "2026-11-09.2"
from ..classify import Classified


def shape(value):
    if value is None:
        return None
    s = re.sub(r"[A-Z]", "A", str(value).upper())
    return re.sub(r"[0-9]", "9", s)[:16]


def fmt(value):
    """Coarse format of a raw value: 'num' for numbers, else length + character classes (A letters, 9 digits,
    P anything else). 'KSH-000123' -> '10:A9P'; a 14-character walk-in code 'W1234567890123' -> '14:A9'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return "num"
    s = str(value).strip().upper()
    if not s:
        return None
    try:
        float(s)
        return "num"
    except ValueError:
        pass
    return f"{len(s)}:" + ("A" if re.search(r"[A-Z]", s) else "") + ("9" if re.search(r"[0-9]", s) else "") + \
        ("P" if re.search(r"[^A-Z0-9]", s) else "")


class SQLite:
    name = "sqlite"

    @staticmethod
    def key(col, t):
        return P.SQLite.key(P.q(col), t)

    @staticmethod
    def shape(expr):
        return f"shape({expr})"

    @staticmethod
    def rel(fqn):
        return P.q(fqn)

    @staticmethod
    def date(col):
        return P.SQLite.date(P.q(col))

    @staticmethod
    def text(col):
        return f"CAST({P.q(col)} AS TEXT)"

    @staticmethod
    def row_signature(columns):
        return " || '|' || ".join(f"COALESCE(CAST({P.q(c)} AS TEXT), '')" for c, _ in columns)

    @staticmethod
    def fmt(col):
        return f"fmt({P.q(col)})"


class Snowflake:
    name = "snowflake"

    @staticmethod
    def key(col, t):
        return P.Snowflake.key(P.q(col), t)

    @staticmethod
    def shape(expr):
        return f"LEFT(REGEXP_REPLACE(REGEXP_REPLACE(UPPER({expr}), '[A-Z]', 'A'), '[0-9]', '9'), 16)"

    @staticmethod
    def rel(fqn):
        return ".".join(P.q(p) for p in fqn.split("."))

    @staticmethod
    def date(col):
        return P.Snowflake.date(P.q(col))

    @staticmethod
    def text(col):
        return f"{P.q(col)}::STRING"

    @staticmethod
    def row_signature(columns):
        return f"HASH({', '.join(P.q(c) for c, _ in columns)})"

    @staticmethod
    def fmt(col):
        t = f"UPPER(TRIM({P.q(col)}::STRING))"
        return (f"IFF(TRY_TO_DOUBLE({t}) IS NOT NULL, 'num', LENGTH({t}) || ':' || IFF(REGEXP_LIKE({t}, '.*[A-Z].*'), "
                f"'A', '') || IFF(REGEXP_LIKE({t}, '.*[0-9].*'), '9', '') || IFF(REGEXP_LIKE({t}, '.*[^A-Z0-9].*'), "
                f"'P', ''))")


# ------------------------------------------------------------------ SQL builders
def screen_sql(D, fqn, cands, rk, date_col, columns):
    rk_expr = D.key(rk[0], rk[1]) if rk else "NULL"
    dt = D.date(date_col) if date_col else "NULL"
    rs = D.row_signature(signature_columns(columns)) if not rk else "NULL"
    proj = [f"{rk_expr} AS rk", f"{dt} AS dt", f"{rs} AS rs"] + [f"{D.key(c, t)} AS k{i}" for i, (c, t) in enumerate(cands)]
    agg = ["COUNT(*)", "COUNT(rk)", "COUNT(DISTINCT rk)", "MIN(dt)", "MAX(dt)", "COUNT(DISTINCT rs)"]
    for i in range(len(cands)):
        agg += [f"COUNT(k{i})", f"COUNT(DISTINCT k{i})", f"COUNT(DISTINCT rk || '|' || k{i})",
                f"SUM(CASE WHEN k{i} = rk THEN 1 ELSE 0 END)"]
    return [f"SELECT {', '.join(agg)} FROM (SELECT {', '.join(proj)} FROM {D.rel(fqn)}) t"]


def signature_columns(columns):
    """Columns hashed to detect exact duplicate rows: every column except personal ones (names, phones, dates of
    birth, national IDs, addresses). Only the distinct count of the hash leaves the warehouse."""
    from .interpret import PRIVACY
    return [(c, t) for c, t in columns if not any(p in c.upper() for p in PRIVACY)]


def parse_screen(rows, cands, has_rk):
    r = rows[0][0]
    v = dict(rows=int(r[0] or 0), rk_rows=int(r[1] or 0), rk_distinct=int(r[2] or 0),
             date_min=str(r[3]) if r[3] is not None else None, date_max=str(r[4]) if r[4] is not None else None,
             distinct_rows=None if has_rk else int(r[5] or 0), cands={})
    for i, (c, _) in enumerate(cands):
        n, d, p, e = r[6 + 4 * i: 10 + 4 * i]
        v["cands"][c] = dict(nonnull=int(n or 0), distinct=int(d or 0), rk_pairs=int(p or 0), eq_rk=int(e or 0))
    return v


def pair_sql(D, fqn, a, b):
    inner = f"SELECT {D.key(*a)} AS a, {D.key(*b)} AS b FROM {D.rel(fqn)}"
    both = f"FROM ({inner}) t WHERE a IS NOT NULL AND b IS NOT NULL"
    return [f"SELECT COUNT(DISTINCT a), COUNT(DISTINCT b), COUNT(DISTINCT a || '|' || b) {both}",
            f"SELECT MAX(n), SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) FROM (SELECT a, COUNT(DISTINCT b) AS n {both} GROUP BY a) g",
            f"SELECT MAX(n), SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) FROM (SELECT b, COUNT(DISTINCT a) AS n {both} GROUP BY b) g"]


def parse_pair(rows):
    (da, db, dab), (max_b, a_many), (max_a, b_many) = (r[0] for r in rows)
    return dict(da=int(da or 0), db=int(db or 0), dab=int(dab or 0), max_b_per_a=int(max_b or 0),
                a_with_many_b=int(a_many or 0), max_a_per_b=int(max_a or 0), b_with_many_a=int(b_many or 0))


def shape_sql(D, fqn, col):
    inner = f"SELECT {D.key(*col)} AS k FROM {D.rel(fqn)}"
    return [f"SELECT s, COUNT(*) AS n FROM (SELECT {D.shape('k')} AS s FROM ({inner}) t WHERE k IS NOT NULL) u "
            f"GROUP BY s ORDER BY n DESC, s LIMIT 3",
            f"SELECT COUNT(*) FROM ({inner}) t WHERE k IS NOT NULL"]


def parse_shape(rows):
    total = int(rows[1][0][0] or 0)
    return dict(total=total, top=[(s, (n / total) if total else 0.0) for s, n in rows[0]])


def containment_sql(D, a_fqn, a, b_fqn, b):
    ka = f"SELECT DISTINCT {D.key(*a)} AS k FROM {D.rel(a_fqn)}"
    kb = f"SELECT DISTINCT {D.key(*b)} AS k FROM {D.rel(b_fqn)}"
    return [f"SELECT COUNT(a.k), COUNT(b.k) FROM ({ka}) a LEFT JOIN ({kb}) b ON a.k = b.k WHERE a.k IS NOT NULL",
            f"SELECT COUNT(k) FROM ({kb}) t WHERE k IS NOT NULL"]


def parse_containment(rows):
    (da, inter), (db,) = rows[0][0], rows[1][0]
    da, inter, db = int(da or 0), int(inter or 0), int(db or 0)
    return dict(da=da, db=db, inter=inter, a_in_b=inter / da if da else 0.0, b_in_a=inter / db if db else 0.0)


def window_sql(D, fqn, col, date_col, until):
    until = str(until).replace("'", "")
    return [f"SELECT COUNT(DISTINCT k) FROM (SELECT {D.key(*col)} AS k, {D.date(date_col)} AS d FROM {D.rel(fqn)}) t "
            f"WHERE k IS NOT NULL AND d <= '{until}'"]


def categories_sql(D, fqn, cols):
    return [f"SELECT {', '.join(f'COUNT(DISTINCT {P.q(c)})' for c in cols)} FROM {D.rel(fqn)}"]


def conditional_sql(D, fqn, col, group):
    inner = f"SELECT {D.text(group)} AS g, {D.key(*col)} AS k FROM {D.rel(fqn)}"
    return [f"SELECT SUM(CASE WHEN nn * 10 >= n * 9 THEN n ELSE 0 END), SUM(CASE WHEN nn * 10 <= n THEN n ELSE 0 END), "
            f"SUM(n), COUNT(*) FROM (SELECT g, COUNT(*) AS n, COUNT(k) AS nn FROM ({inner}) t GROUP BY g) x"]


def align_sql(D, a_fqn, a_rk, a_key, b_fqn, b_rk, b_key):
    """Per record ID, one identifier value per side (MIN over versions); join on the record ID."""
    side = lambda fqn, rk, key: (f"SELECT rk, MIN(k) AS k FROM (SELECT {D.key(*rk)} AS rk, {D.key(*key)} AS k "
                                 f"FROM {D.rel(fqn)}) t WHERE rk IS NOT NULL GROUP BY rk")
    a, b = side(a_fqn, a_rk, a_key), side(b_fqn, b_rk, b_key)
    return [f"SELECT COUNT(*), SUM(CASE WHEN a.k = b.k THEN 1 ELSE 0 END), "
            f"SUM(CASE WHEN a.k <> b.k THEN 1 ELSE 0 END) FROM ({a}) a JOIN ({b}) b ON a.rk = b.rk",
            f"SELECT COUNT(*) FROM ({a}) a", f"SELECT COUNT(*) FROM ({b}) b"]


def parse_align(rows):
    (shared, agree, disagree), (ra,), (rb,) = rows[0][0], rows[1][0], rows[2][0]
    return dict(shared=int(shared or 0), agree=int(agree or 0), disagree=int(disagree or 0), ra=int(ra or 0),
                rb=int(rb or 0))


def formats_sql(D, a_fqn, a, b_fqn, b):
    fa = f"SELECT DISTINCT {D.key(*a)} AS k, {D.fmt(a[0])} AS f FROM {D.rel(a_fqn)}"
    fb = f"SELECT DISTINCT {D.fmt(b[0])} AS f FROM {D.rel(b_fqn)}"
    return [f"SELECT COUNT(DISTINCT a.k), COUNT(DISTINCT CASE WHEN b.f IS NOT NULL THEN a.k END) "
            f"FROM ({fa}) a LEFT JOIN ({fb}) b ON a.f = b.f WHERE a.k IS NOT NULL"]


# ------------------------------------------------------------------ execution
def _profile(wh, D, cat, gap):
    o = cat[gap.subject[0]]
    (col, t), rk, deleted, version, date_col = gap.args
    c = Classified(fqn=o["fqn"], schema=o["schema"], name=o["name"], object_type=o["type"], row_count=o["row_count"],
                   last_altered=o["last_altered"], columns=o["columns"], key=col, key_type=t, record_key=rk,
                   deleted_column=deleted, version_column=version, date_column=date_col)
    return P.parse([wh.query(s)[0] for s in P.profile_statements(c, D.name)])


def run(wh, D, cat, gap):
    """Execute one gap's measurement; returns its values. Every query goes through wh.query (counted by callers)."""
    k, s, a = gap.kind, gap.subject, gap.args
    if k == "screen":
        cands, rk, date_col = a
        o = cat[s[0]]
        return parse_screen([wh.query(q) for q in screen_sql(D, s[0], cands, rk, date_col, o["columns"])], cands, bool(rk))
    if k == "profile":
        return _profile(wh, D, cat, gap)
    if k == "pair":
        return parse_pair([wh.query(q) for q in pair_sql(D, s[0], a[0], a[1])])
    if k == "shape":
        return parse_shape([wh.query(q) for q in shape_sql(D, s[0], a[0])])
    if k == "containment":
        return parse_containment([wh.query(q) for q in containment_sql(D, s[0], a[0], s[1], a[1])])
    if k == "window":
        return dict(distinct=int(wh.query(window_sql(D, s[0], a[0], a[1], a[2])[0])[0][0] or 0))
    if k == "categories":
        r = wh.query(categories_sql(D, s[0], a)[0])[0]
        return dict(distinct={c: int(n or 0) for c, n in zip(a, r)})
    if k == "conditional":
        present, absent, n, groups = wh.query(conditional_sql(D, s[0], a[0], a[1])[0])[0]
        return dict(present_rows=int(present or 0), absent_rows=int(absent or 0), rows=int(n or 0),
                    groups=int(groups or 0))
    if k == "align":
        return parse_align([wh.query(q) for q in align_sql(D, s[0], a[0], a[1], s[1], a[2], a[3])])
    if k == "formats":
        d, covered = wh.query(formats_sql(D, s[0], a[0], s[1], a[1])[0])[0]
        return dict(distinct=int(d or 0), covered=int(covered or 0))
    raise ValueError(k)


def statements_for(D, cat, gap):
    """The SQL a gap would run (used to check the Snowflake dialect builds, and for cost accounting)."""
    k, s, a = gap.kind, gap.subject, gap.args
    if k == "screen":
        return screen_sql(D, s[0], a[0], a[1], a[2], cat[s[0]]["columns"])
    if k == "pair":
        return pair_sql(D, s[0], a[0], a[1])
    if k == "shape":
        return shape_sql(D, s[0], a[0])
    if k == "containment":
        return containment_sql(D, s[0], a[0], s[1], a[1])
    if k == "window":
        return window_sql(D, s[0], a[0], a[1], a[2])
    if k == "categories":
        return categories_sql(D, s[0], a)
    if k == "conditional":
        return conditional_sql(D, s[0], a[0], a[1])
    if k == "align":
        return align_sql(D, s[0], a[0], a[1], s[1], a[2], a[3])
    if k == "formats":
        return formats_sql(D, s[0], a[0], s[1], a[1])
    if k == "profile":
        o = cat[s[0]]
        (col, t), rk, deleted, version, date_col = a
        c = Classified(fqn=o["fqn"], schema=o["schema"], name=o["name"], object_type=o["type"],
                       row_count=o["row_count"], last_altered=o["last_altered"], columns=o["columns"], key=col,
                       key_type=t, record_key=rk, deleted_column=deleted, version_column=version, date_column=date_col)
        return P.profile_statements(c, D.name)
    raise ValueError(k)
