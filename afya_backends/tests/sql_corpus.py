"""SQL corpus check: translate every Snowflake statement in the platform and
have Databend parse the result.

    DATABEND_HTTP_URL=http://localhost:8008 python -m afya_backends.tests.sql_corpus [--redash] [-v]

Sources: string literals / f-strings in Python that look like SQL (f-string
holes become a placeholder identifier), *.sql files, and optionally every
saved Redash query (REDASH_BASE_URL + REDASH_ADMIN_API_KEY).  Each statement
is translated (sql_translate) and sent to Databend as ``EXPLAIN SYNTAX`` -
parse only, so it works before any table exists.  Reports translation
failures and Databend syntax errors with their source location.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {"env", "node_modules", "static", "logs", ".git", "afya_backends", "__pycache__", "redash", "media"}
SQL_START = re.compile(r"^\s*(?:\(\s*)?(SELECT|WITH|INSERT|MERGE|CREATE|UPDATE|DELETE|COPY|ALTER|DROP|TRUNCATE|"
                       r"SHOW|DESC|DESCRIBE|USE|PUT)\b", re.I)
HOLE = "AFYA_HOLE"


def _fstring_text(node: ast.JoinedStr) -> str:
    out = []
    for v in node.values:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out.append(v.value)
        else:
            out.append(HOLE)
    return "".join(out)


# Structural signatures of real SQL (prose like "Create per-facility RAW
# schema..." or "Insert a joins: entry..." has none of them).
SQL_SHAPE = re.compile(
    r"\bSELECT\b[\s\S]+\bFROM\b|\bINSERT\s+(OVERWRITE\s+)?INTO\b|\bMERGE\s+INTO\b|\bUPDATE\s+\S+\s+SET\b|"
    r"\bDELETE\s+FROM\b|\bCREATE\s+(OR\s+REPLACE\s+)?(TRANSIENT\s+|TEMP\w*\s+|SECURE\s+)?(TABLE|VIEW|SCHEMA|STAGE|"
    r"DATABASE|SEQUENCE)\b|\bALTER\s+(TABLE|SESSION|WAREHOUSE)\b|\bDROP\s+(TABLE|VIEW|SCHEMA|STAGE)\b|"
    r"\bCOPY\s+INTO\b|\bTRUNCATE\s+TABLE\b|\bWITH\s+\w+\s+AS\s*\(|\bUSE\s+(WAREHOUSE|ROLE|SCHEMA|DATABASE)\b|"
    r"\bSHOW\s+(TABLES|SCHEMAS|COLUMNS|VIEWS)\b|\bDESC(RIBE)?\s+TABLE\b|^\s*SELECT\s+\S", re.I)
# Strings used with MySQL/Postgres clients, not the warehouse.
NOT_WAREHOUSE = re.compile(r"ENGINE\s*=\s*InnoDB|ON DUPLICATE KEY|AUTO_INCREMENT|`\w+`|\bSERIAL\b|"
                           r"pg_catalog|information_schema\.processlist", re.I)


def _docstring_nodes(tree):
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def python_sql():
    for f in sorted(ROOT.rglob("*.py")):
        rel = f.relative_to(ROOT)
        if any(p in SKIP for p in rel.parts) or f.is_symlink() or "tests" in rel.parts:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        docs = _docstring_nodes(tree)
        for node in ast.walk(tree):
            text = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                text = node.value
            elif isinstance(node, ast.JoinedStr):
                text = _fstring_text(node)
            if not text or len(text) < 12 or not SQL_START.match(text):
                continue
            if not SQL_SHAPE.search(text) or NOT_WAREHOUSE.search(text):
                continue
            yield f"{rel}:{getattr(node, 'lineno', 0)}", text


def sql_files():
    for f in sorted(ROOT.rglob("*.sql")):
        rel = f.relative_to(ROOT)
        if any(p in SKIP for p in rel.parts) or "target" in rel.parts:
            continue
        yield str(rel), f.read_text(encoding="utf-8", errors="replace")


def redash_queries():
    import requests
    base = os.getenv("REDASH_BASE_URL", "").rstrip("/")
    key = os.getenv("REDASH_ADMIN_API_KEY", "")
    if not (base and key):
        return
    page = 1
    while True:
        r = requests.get(f"{base}/api/queries", params={"page": page, "page_size": 100},
                         headers={"Authorization": f"Key {key}"}, timeout=60).json()
        for q in r.get("results", []):
            yield f"redash:{q['id']}:{q['name'][:40]}", q.get("query") or ""
        if page * 100 >= r.get("count", 0):
            break
        page += 1


def main(argv):
    import requests
    from afya_backends.warehouse import sql_translate as tr
    from afya_backends.warehouse.dbapi import bind
    url = os.getenv("DATABEND_HTTP_URL", "http://localhost:8008").rstrip("/")
    auth = (os.getenv("DATABEND_USER", "databend"), os.getenv("DATABEND_PASSWORD", ""))
    verbose = "-v" in argv
    sources = list(python_sql()) + list(sql_files())
    if "--redash" in argv:
        sources += list(redash_queries())
    ctx = tr.Context(database="HOSPITALS", schema="PUBLIC", primary_database="HOSPITALS")
    stats = Counter()
    failures = {"translate": [], "databend": []}
    http = requests.Session()
    http.auth = auth
    from afya_backends.warehouse.dbapi import _quote_aware_scan
    failures["fstring_fragment"] = []
    for loc, text in sources:
        # Bind parameters the way the call site would - but only real
        # placeholders (outside quotes); '%spine%' in a LIKE is not one.
        sample = text
        outside = "".join(text[i] for i in _quote_aware_scan(text, lambda i: i))
        try:
            if re.search(r"%\(\w+\)s", outside):
                names = re.findall(r"%\((\w+)\)s", sample)
                sample = bind(sample, {n: "x" for n in names})
            elif "%s" in outside.replace("%%", ""):
                sample = bind(sample, ["x"] * sample.replace("%%", "").count("%s"))
        except Exception:
            pass
        has_hole = HOLE in sample
        for stmt in tr.split_statements(sample):
            if not SQL_START.match(stmt):
                continue
            stats["statements"] += 1
            try:
                out = tr.translate(stmt, ctx)
            except Exception as exc:
                bucket = "fstring_fragment" if has_hole else "translate"
                stats[f"{bucket}_error"] += 1
                failures[bucket].append((loc, str(exc)[:160], stmt[:200]))
                continue
            for st in out:
                stats[f"action:{st.action}"] += 1
                if st.action not in ("sql", "copy"):
                    continue
                if st.sql.lstrip().upper().startswith(("COPY", "LIST", "REMOVE")):
                    continue  # EXPLAIN SYNTAX doesn't take COPY; covered by live tests
                r = http.post(f"{url}/v1/query", json={"sql": "EXPLAIN SYNTAX " + st.sql}, timeout=60).json()
                if r.get("error"):
                    msg = " ".join(r["error"].get("message", "").split())
                    bucket = "fstring_fragment" if has_hole and HOLE in st.sql else "databend"
                    stats[f"{bucket}_syntax_error"] += 1
                    failures[bucket].append((loc, msg[-220:], st.sql[:300]))
                else:
                    stats["ok"] += 1
    print(json.dumps(stats, indent=1, sort_keys=True))
    for kind in ("translate", "databend", "fstring_fragment"):
        if kind == "fstring_fragment" and not verbose:
            print(f"\n==== {kind} (f-string holes, informational): {len(failures[kind])}")
            continue
        items = failures[kind]
        print(f"\n==== {kind} failures: {len(items)}")
        for loc, msg, sql in (items if verbose else items[:60]):
            print(f"- {loc}\n    {msg}\n    SQL: {sql!r}")
    print("\n==== most common Databend error tails")
    for msg, n in Counter(m[-90:] for _, m, _ in failures["databend"]).most_common(25):
        print(f"{n:4d}  {msg}")
    return 0 if not (failures["translate"] or failures["databend"]) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
