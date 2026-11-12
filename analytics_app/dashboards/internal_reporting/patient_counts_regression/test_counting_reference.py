"""Characterisation tests for the CURRENT counting semantics (expected values worked out by hand from the SQL).

These are not "desired behaviour" tests: several pin down quirks (marked QUIRK) that the discovery redesign must
NOT change silently. Run:  python -I test_counting_reference.py   (add --report to write counting_results.md)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import counting_reference as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "counting_results.md"
RESULTS = []


def check(tid, title, actual, expected, note=""):
    RESULTS.append((tid, title, actual == expected, expected, actual, note))


def table(conn, name, cols, rows):
    conn.execute(f"CREATE TABLE {R.q(name)} ({', '.join(R.q(c) for c in cols)})")
    conn.executemany(f"INSERT INTO {R.q(name)} VALUES ({', '.join('?' * len(cols))})", rows)


def main():
    c = R.connect()

    # ---------------- Patient Records / UPDs
    table(c, "basic", ["ID", "PATIENT"], [(1, "A"), (2, "A"), (3, "B"), (4, "B"), (5, "B")])
    r = R.count_source(c, "basic", "PATIENT")
    check("N01", "Plain rows: records = rows, UPDs = distinct key", (r["records"], r["upds"]), (5, 2))

    vals = [None, "", "null", "None", "false", "0", "2025-01-01", "False", "N", "yes", "NULL", "none", "FALSE"]
    table(c, "dels", ["ID", "PATIENT", "DELETED_AT"], [(i, f"P{i}", v) for i, v in enumerate(vals)])
    r = R.count_source(c, "dels", "PATIENT", dc="DELETED_AT")
    check("N02", "Deletion predicate: only '2025-01-01', 'False', 'N', 'yes' count as deleted", r["rows_deleted"], 4,
          "QUIRK: 'False' and 'N' are treated as deleted (case-sensitive list, L234-235)")

    table(c, "active", ["ID", "PATIENT", "IS_ACTIVE"], [(1, "A", "true"), (2, "B", "true"), (3, "C", "false")])
    r = R.count_source(c, "active", "PATIENT", dc="IS_ACTIVE")
    check("N03", "IS_ACTIVE configured as DELETED_COLUMN inverts: active rows deleted", (r["records"], r["rows_deleted"]),
          (1, 2), "QUIRK: why discovery must never suggest IS_ACTIVE as DELETED_COLUMN")

    table(c, "vers", ["ID", "PATIENT", "DELETED_AT", "UPDATED_AT"],
          [(1, "A", None, "2025-01-01"), (1, "A", None, "2025-02-01"), (1, "A", "2025-03-01", "2025-03-01"),
           (2, "B", None, "2025-01-01")])
    r = R.count_source(c, "vers", "PATIENT", rk="ID", dc="DELETED_AT", vc="UPDATED_AT")
    check("N04", "Versions: latest version decides; latest deleted -> record dropped",
          (r["rows_read"], r["rows_superseded"], r["rows_deleted"], r["records"], r["upds"]), (4, 2, 1, 1, 1))

    table(c, "badts", ["ID", "PATIENT", "DELETED_AT", "UPDATED_AT"],
          [(1, "A", None, "2025-01-01"), (1, "A", "2025-06-01", "not a date")])
    r = R.count_source(c, "badts", "PATIENT", rk="ID", dc="DELETED_AT", vc="UPDATED_AT")
    check("N05", "Unparseable version sorts last: older live version survives", (r["records"], r["rows_deleted"]), (1, 0),
          "QUIRK: a deletion recorded with a bad timestamp is ignored (NULLS LAST)")

    table(c, "nullrk", ["ID", "PATIENT", "UPDATED_AT"], [(None, "A", "2025-01-01"), (None, "A", "2025-01-02"), (1, "B", "x")])
    r = R.count_source(c, "nullrk", "PATIENT", rk="ID", vc="UPDATED_AT")
    check("N06", "Rows with NULL record key are each kept (rn = 1)", (r["records"], r["rows_superseded"]), (3, 0))

    table(c, "novc", ["ID", "PATIENT"], [(1, "A"), (1, "A"), (2, "B")])
    r = R.count_source(c, "novc", "PATIENT", rk="ID")
    check("N07", "Record key without version column is ignored: duplicates counted", r["records"], 3,
          "QUIRK: RECORD_KEY alone does not deduplicate (rn = 1 when VERSION_COLUMN is NULL, L237)")

    table(c, "keys", ["ID", "PATIENT"], [(1, " P1 "), (2, "P1"), (3, ""), (4, "   "), (5, None), (6, "null"), (7, "0")])
    r = R.count_source(c, "keys", "PATIENT")
    check("N08", "Key normalisation: TRIM + blank->NULL only", (r["upds"], r["no_uk"]), (3, 3),
          "QUIRK: 'null' and '0' count as patient identities (discovery's placeholder list must not leak into counting)")

    table(c, "delonly", ["ID", "PATIENT", "DELETED_AT"], [(1, "A", "2025-01-01"), (2, "B", None)])
    r = R.count_source(c, "delonly", "PATIENT", dc="DELETED_AT")
    check("N09", "A patient seen only on deleted records is not a UPD", r["upds"], 1)

    table(c, "num", ["ID", "PATIENT"], [(1, 123), (2, "123"), (3, 123.0)])
    r = R.count_source(c, "num", "PATIENT")
    check("N10", "Numeric vs text keys after ::STRING (SQLite: 123.0 -> '123.0')", r["upds"], 2,
          "UNVERIFIED in Snowflake: FLOAT 123.0::STRING may render '123' -> 1 UPD. Check on an AFYA_ test table")

    try:
        R.count_source(c, "vers", "PATIENT", vc="UPDATED_AT")
        check("N11", "VERSION_COLUMN without RECORD_KEY is rejected", "accepted", "bad_config")
    except ValueError:
        check("N11", "VERSION_COLUMN without RECORD_KEY is rejected", "bad_config", "bad_config")

    # ---------------- checks (L264-278)
    check("K01", "UPDs > records can never fire for real data (distinct over kept rows <= kept rows)",
          R.checks(10, 10, 0)[0], "ok", "QUIRK: check is unreachable (adversarially noted in report 2.4)")
    check("K02", "> 5% missing key -> review", R.checks(100, 90, 6)[0], "review")
    check("K03", "> 20% move vs last published -> review", R.checks(130, 50, 0, p_rec=100, p_upd=50)[0], "review")
    check("K04", "First publish of a new source: no history -> only zero/missing checks apply",
          R.checks(5000, 3032, 0)[0], "ok")

    # ---------------- receipts (L280-295 load-time; L367-388 catch-up)
    table(c, "recv", ["ID", "PATIENT", "LOADED"],
          [(1, "A", "2025-01-01"), (2, "A", "2025-01-02"), (3, "B", "2025-01-02"), (4, "C", None)])
    lt = R.load_time_receipts(c, "recv", "PATIENT", rc="LOADED")
    check("E01", "Load-time receipts: records by day, each UPD on its first day, undated kept", lt,
          {None: (1, 1), "2025-01-01": (1, 1), "2025-01-02": (2, 1)})
    check("E02", "Load-time receipts sum to the published figures",
          (sum(v[0] for v in lt.values()), sum(v[1] for v in lt.values())),
          (R.count_source(c, "recv", "PATIENT")["records"], R.count_source(c, "recv", "PATIENT")["upds"]))

    rc = R.Receipts()
    a = rc.publish("S", 100, 40, run_date="2026-11-04", raw_main="2026-03-27", created="2026-03-01")
    check("E03", "First publish dated to main raw load", a, ("S", "2026-03-27", 100, 40, "raw event load date"))
    b = rc.publish("S", 110, 42, run_date="2026-11-11", raw_last="2026-11-08")
    check("E04", "Growth with a newer raw load is dated to that load", b, ("S", "2026-11-08", 10, 2, "raw event load date"))
    d = rc.publish("S", 105, 42, run_date="2026-11-18", raw_last="2026-11-08")
    check("E05", "Shrinkage (deletions) without a newer load: negative run increment", d,
          ("S", "2026-11-18", -5, 0, "run increment"), "QUIRK: receipts can be negative")
    e = rc.publish("S", 105, 42, run_date="2026-11-25")
    check("E06", "No change: no receipt row", e, None)
    check("E07", "Invariant: receipts sum to the last published figures", rc.totals("S")[:2], (105, 42))
    f = rc.publish("T", 50, 20, run_date="2026-11-04", created="2026-02-01")
    check("E08", "First publish without raw load: dated to table creation", f[1:], ("2026-02-01", 50, 20, "table created"))
    g = rc.publish("V", 50, 20, run_date="2026-11-04")
    check("E09", "First publish of a view without raw load: dated to the run", g[1:], ("2026-11-04", 50, 20, "first counted"))

    ok = sum(r[2] for r in RESULTS)
    lines = ["# Counting characterisation results", "",
             "Generated by `patient_counts_regression/test_counting_reference.py` (SQLite reference of the current refresh SQL).",
             f"**{ok} / {len(RESULTS)} pass.** A failure here means the reference model disagrees with the hand-derived",
             "reading of the production SQL, i.e. one of the two is wrong and must be checked before relying on it.", "",
             "| Test | Behaviour pinned | Result | Expected | Actual | Note |", "|---|---|---|---|---|---|"]
    for tid, title, passed, exp, act, note in RESULTS:
        lines.append(f"| {tid} | {title} | {'pass' if passed else 'FAIL'} | `{exp}` | `{act}` | {note} |")
    if "--report" in sys.argv:
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for tid, title, passed, exp, act, _ in RESULTS:
        if not passed:
            print("FAIL", tid, title, "expected", exp, "actual", act)
    print(f"{ok}/{len(RESULTS)} pass")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
