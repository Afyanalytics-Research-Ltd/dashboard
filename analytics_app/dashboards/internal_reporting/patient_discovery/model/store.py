"""Shared storage for people's answers and the current questions, in HOSPITALS.ANALYTICS.

AFYA_PATIENT_DISCOVERY_QUESTIONS   the current question pack; replaced by every discovery run
AFYA_PATIENT_DISCOVERY_STATEMENTS  answers as statements; append-only; the latest row per STATEMENT_ID wins

Used by the command line (patient_discovery.run) and the dashboard's Discovery tab. Functions take `fetch(sql)` and
`write([(sql, params)])` callables so each caller keeps its own connection. Nothing here touches the registry or
the counts.
"""
import json
import re

from .. import config as C

QUESTIONS = f"{C.DATABASE}.ANALYTICS.AFYA_PATIENT_DISCOVERY_QUESTIONS"
STATEMENTS = f"{C.DATABASE}.ANALYTICS.AFYA_PATIENT_DISCOVERY_STATEMENTS"
TABLES = (QUESTIONS, STATEMENTS)
REGISTRY = f"{C.DATABASE}.ANALYTICS.AFYA_PATIENT_COUNT_SOURCES"
_LITERAL = re.compile(r"'(?:[^'\\]|\\.|'')*'")

DDL = [
    f"""CREATE TABLE IF NOT EXISTS {QUESTIONS} (
    QUESTION_ID  STRING NOT NULL,
    TYPE         STRING NOT NULL,      -- meaning | scope | a question code
    FACILITY     STRING,
    BODY         VARIANT NOT NULL,     -- the question as generated: ask, options, members/objects, evidence
    GENERATED_AT TIMESTAMP_NTZ NOT NULL)      -- Africa/Nairobi time""",
    f"""CREATE TABLE IF NOT EXISTS {STATEMENTS} (
    STATEMENT_ID STRING NOT NULL,      -- N:<facility>:<column> | F:<facility> | X:<object>
    KIND         STRING NOT NULL,      -- namespace | facility | not_a_source
    SUBJECT      STRING NOT NULL,
    BODY         VARIANT NOT NULL,     -- the full statement (assertion with provenance)
    QUESTION_ID  STRING,
    ANSWERED_BY  STRING NOT NULL,
    ANSWERED_AT  TIMESTAMP_NTZ NOT NULL,
    BASIS        STRING)""",
]


def read_statements(fetch):
    """Latest statement per id, as dicts in the statements.json shape. [] if the table doesn't exist yet."""
    try:
        recs = fetch(f"SELECT BODY FROM {STATEMENTS} "
                     f"QUALIFY ROW_NUMBER() OVER (PARTITION BY STATEMENT_ID ORDER BY ANSWERED_AT DESC) = 1")
    except Exception as e:
        if "does not exist" in str(e) or "not authorized" in str(e):
            return []
        raise
    return [json.loads(r["BODY"]) if isinstance(r["BODY"], str) else r["BODY"] for r in recs]


def read_questions(fetch):
    try:
        recs = fetch(f"SELECT BODY, GENERATED_AT FROM {QUESTIONS} ORDER BY QUESTION_ID")
    except Exception as e:
        if "does not exist" in str(e):
            return [], None
        raise
    body = [json.loads(r["BODY"]) if isinstance(r["BODY"], str) else r["BODY"] for r in recs]
    return body, (max(r["GENERATED_AT"] for r in recs) if recs else None)


def answered_questions(fetch):
    """{question id: (answered by, at)} for answers recorded since the questions were generated."""
    try:
        recs = fetch(f"SELECT QUESTION_ID, ANSWERED_BY, ANSWERED_AT FROM {STATEMENTS} WHERE QUESTION_ID IS NOT NULL")
    except Exception as e:
        if "does not exist" in str(e):
            return {}
        raise
    return {r["QUESTION_ID"]: (r["ANSWERED_BY"], r["ANSWERED_AT"]) for r in recs}


def publish_questions(groups):
    """(sql, params) pairs that replace the question pack."""
    out = [(f"DELETE FROM {QUESTIONS}", None)]
    for g in groups:
        out.append((f"INSERT INTO {QUESTIONS} (QUESTION_ID, TYPE, FACILITY, BODY, GENERATED_AT) "
                    f"SELECT %s, %s, %s, PARSE_JSON(%s), CONVERT_TIMEZONE('Africa/Nairobi', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ",
                    (g["id"], g["type"], g.get("facility"), json.dumps(g, default=str))))
    return out


def merge(new, existing):
    """A namespace answered again gains members (with its history) instead of replacing them; other facts already
    recorded are not recorded twice. existing: {statement id: statement dict}."""
    out = []
    for s in new:
        old = existing.get(s["id"])
        if old and s["kind"] == "namespace" and old["assertion"].get("meaning") == s["assertion"]["meaning"]:
            members = list(old["assertion"]["members"])
            added = [m for m in s["assertion"]["members"] if m not in members]
            if not added:
                continue                                   # the same fact is not recorded twice
            members += added
            s = dict(s, assertion=dict(s["assertion"], members=members,
                                        history=old["assertion"].get("history", []) +
                                        [old["assertion"].get("provenance")]))
        elif old and old["assertion"] == s["assertion"]:
            continue
        out.append(s)
    return out


def record_statements(statements):
    """(sql, params) pairs that append statements."""
    out = []
    for s in statements:
        p = s["assertion"].get("provenance") or {}
        out.append((f"INSERT INTO {STATEMENTS} (STATEMENT_ID, KIND, SUBJECT, BODY, QUESTION_ID, ANSWERED_BY, "
                    f"ANSWERED_AT, BASIS) SELECT %s, %s, %s, PARSE_JSON(%s), %s, %s, CONVERT_TIMEZONE('Africa/Nairobi', CURRENT_TIMESTAMP())::TIMESTAMP_NTZ, %s",
                    (s["id"], s["kind"], s["subject"], json.dumps(s, default=str), p.get("question"),
                     p.get("by") or "unknown", p.get("basis"))))
    return out


def register_sql(reg):
    """Insert one accepted source into the counting registry, only if neither its object nor its name exists.
    The provenance (question, answers, who) goes in the registry's PROFILE column."""
    c = reg["config"]
    return (f"INSERT INTO {REGISTRY} (SOURCE, SOURCE_OBJECT, UPD_KEY, RECORD_KEY, DELETED_COLUMN, VERSION_COLUMN, "
            f"RECEIVED_COLUMN, STATUS, PROFILE, UPDATED_BY) SELECT %s, %s, %s, %s, %s, %s, %s, 'active', PARSE_JSON(%s), %s "
            f"WHERE NOT EXISTS (SELECT 1 FROM {REGISTRY} WHERE UPPER(TRIM(SOURCE_OBJECT)) = %s OR SOURCE = %s)",
            (reg["name"], reg["object"], c["key"], c.get("record"), c.get("deleted"), c.get("version"),
             c.get("received"), json.dumps(dict(discovery=reg["provenance"]), default=str),
             reg["provenance"].get("by") or "discovery", reg["object"].upper(), reg["name"]))


def registered_statement(reg):
    p = dict(reg["provenance"])
    return dict(id=f"G:{reg['object']}", kind="registered", subject=reg["object"], origin="system",
                assertion=dict(name=reg["name"], config=reg["config"], provenance=p))


def check_writes(pairs):
    """Discovery may write only: CREATE TABLE IF NOT EXISTS / INSERT / DELETE on its two tables, and the guarded
    insert-if-absent into the registry. Anything else (any UPDATE or DELETE of the registry, other tables, several
    statements) is refused."""
    own = "|".join(re.escape(t) for t in TABLES)
    reg = re.escape(REGISTRY)
    allowed = [re.compile(rf"^(CREATE TABLE IF NOT EXISTS|INSERT INTO|DELETE FROM)\s+({own})\b", re.I),
               re.compile(rf"^INSERT INTO\s+{reg}\s+\(SOURCE, SOURCE_OBJECT,.*\bWHERE NOT EXISTS \(SELECT 1 FROM {reg}\b",
                          re.I | re.S)]
    forbidden = re.compile(rf"\b(UPDATE|MERGE|DROP|TRUNCATE|ALTER|GRANT|REVOKE)\b|\bDELETE\s+FROM\s+{reg}\b", re.I)
    for sql, _ in pairs:
        s = sql.strip()
        code = _LITERAL.sub("''", s)
        if ";" in code or not any(a.match(s) for a in allowed) or forbidden.search(code):
            raise PermissionError(f"refused write: {s[:90]}")
