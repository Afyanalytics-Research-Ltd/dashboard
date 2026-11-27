"""Databend: serve the platform's views as tables (refreshed copies) so queries and metadata run at table speed.

Why: Databend re-plans a view - and every view beneath it - on each query, and even to list its columns
(DBeaver does that constantly). Measured on prod 2026-11-27: one REPORTING view's column list 5-8 s, all of
REPORTING's columns 131-142 s, COUNT(*) of RPT_KSH_V3_INV_CONTROL (801 rows) 31 s - against 0.03 s for a table.

What it does, per managed view <DB>.<NAME> (a view in one of `databases` with no owner role, i.e. created
by the platform - views people create themselves are owned by their role and never touched):

  first time  CREATE VIEW  <NAME>__SRC AS <the view's SQL>            the definition is kept, unchanged
              CREATE TABLE <NAME>__MAT_NEW AS SELECT * FROM <NAME>__SRC
              DROP VIEW <NAME>; ALTER TABLE <NAME>__MAT_NEW RENAME TO <NAME>
  refresh     CREATE OR REPLACE TABLE <NAME> AS SELECT * FROM <NAME>__SRC

Views are refreshed in dependency order (a view that reads another managed view comes after it), so each
one reads already-refreshed tables instead of re-running the chain below it. A view that fails keeps its last
good copy (or stays a view); the rest go on. To change a materialized view's SQL, change <NAME>__SRC.
`revert` turns them back into plain views.

    from afya_backends.warehouse import materialize
    report = materialize.run(conn, databases=("REPORTING", "STAGING", "BRONZE"))
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

SRC_SUFFIX = "__SRC"
NEW_SUFFIX = "__MAT_NEW"
DEFAULT_DATABASES = ("REPORTING", "STAGING", "BRONZE")
_REF = re.compile(r'"([A-Za-z0-9_$]+)"\s*\.\s*"([A-Za-z0-9_$]+)"')


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _fq(db: str, name: str) -> str:
    return f"{_q(db)}.{_q(name)}"


@dataclass
class Item:
    db: str
    name: str
    sql: str                      # the SELECT that defines it
    state: str                    # "view" (not yet materialized) | "materialized"
    refs: set = field(default_factory=set)

    @property
    def key(self) -> tuple:
        return (self.db, self.name)


def references(sql: str) -> set:
    """("DB", "NAME") pairs a definition reads (quoted, qualified - as the translator writes them)."""
    return {(a, b) for a, b in _REF.findall(sql or "")}


def order(items: dict) -> list:
    """Dependencies first. A cycle (shouldn't happen with views) is appended at the end, as is."""
    done, out, temp = set(), [], set()

    def visit(k, depth=0):
        if k in done or k in temp or depth > 200:
            return
        temp.add(k)
        for r in sorted(items[k].refs):
            if r in items and r != k:
                visit(r, depth + 1)
        temp.discard(k)
        done.add(k)
        out.append(items[k])

    for k in sorted(items):
        visit(k)
    return out


def discover(run: Callable[[str], list], databases: Iterable[str]) -> dict:
    """{(db, name): Item} - platform views (no owner) and already-materialized ones (<NAME>__SRC + table)."""
    dbs = ", ".join("'" + d.replace("'", "''") + "'" for d in databases)
    views = run(f"SELECT database, name, view_query, owner FROM system.views WHERE database IN ({dbs})")
    tables = {(r[0], r[1]) for r in run(
        f"SELECT database, name FROM system.tables WHERE database IN ({dbs}) AND engine NOT IN ('VIEW', 'STREAM')")}
    items = {}
    for db, name, sql, owner in views:
        if name.endswith(SRC_SUFFIX):
            base = name[: -len(SRC_SUFFIX)]
            if (db, base) in tables:
                items[(db, base)] = Item(db, base, sql, "materialized", references(sql))
            continue
        if owner:                      # someone's own view (owned by their role): not ours
            continue
        if (db, name + SRC_SUFFIX) in {(v[0], v[1]) for v in views}:
            continue                   # half-converted earlier: the __SRC pass above handles it
        items[(db, name)] = Item(db, name, sql, "view", references(sql))
    return items


def _refresh(run, it: Item) -> None:
    src = _fq(it.db, it.name + SRC_SUFFIX)
    if it.state == "view":
        run(f"CREATE OR REPLACE VIEW {src} AS {it.sql}")
        new = _fq(it.db, it.name + NEW_SUFFIX)
        run(f"CREATE OR REPLACE TABLE {new} AS SELECT * FROM {src}")
        run(f"DROP VIEW {_fq(it.db, it.name)}")
        run(f"ALTER TABLE {new} RENAME TO {_q(it.name)}")
    else:
        run(f"CREATE OR REPLACE TABLE {_fq(it.db, it.name)} AS SELECT * FROM {src}")


def run(conn, databases: Iterable[str] = DEFAULT_DATABASES, *, only: Optional[set] = None, dry_run: bool = False,
        log: Callable[[str], None] = None) -> dict:
    """Materialize / refresh every managed view. `conn`: a Databend afya_backends dbapi Connection (raw SQL)."""
    log = log or logger.info
    raw = lambda sql: conn._t.run(sql).rows                       # Databend SQL as is, no translation
    items = discover(raw, databases)
    plan = [it for it in order(items) if not only or ".".join(it.key) in only or it.name in only]
    report = {"materialized": [], "refreshed": [], "failed": [], "seconds": {}, "planned": len(plan)}
    t0 = time.monotonic()
    for it in plan:
        name = ".".join(it.key)
        if dry_run:
            log(f"[dry-run] {name}: {'materialize' if it.state == 'view' else 'refresh'}")
            continue
        s = time.monotonic()
        try:
            _refresh(raw, it)
            report["materialized" if it.state == "view" else "refreshed"].append(name)
        except Exception as e:  # noqa: BLE001 - one view must not stop the rest; its last copy stays
            report["failed"].append(f"{name}: {str(e)[:300]}")
            # a table that stopped refreshing is going stale (a regression); a view Databend can't copy yet
            # just stays a (slow) view
            report.setdefault("stale" if it.state == "materialized" else "still_views", []).append(name)
            log(f"{name}: FAILED {str(e)[:300]}")
            if it.state == "view":       # a half-done first conversion: make sure the view itself still exists
                try:
                    if not raw(f"SELECT 1 FROM system.views WHERE database = '{it.db}' AND name = '{it.name}'"):
                        raw(f"CREATE VIEW {_fq(it.db, it.name)} AS {it.sql}")
                    raw(f"DROP TABLE IF EXISTS {_fq(it.db, it.name + NEW_SUFFIX)}")
                    raw(f"DROP VIEW IF EXISTS {_fq(it.db, it.name + SRC_SUFFIX)}")
                except Exception as e2:  # noqa: BLE001
                    log(f"{name}: cleanup after failure also failed: {str(e2)[:200]}")
        report["seconds"][name] = round(time.monotonic() - s, 1)
    report["total_seconds"] = round(time.monotonic() - t0, 1)
    log(f"materialize: {len(report['materialized'])} materialized, {len(report['refreshed'])} refreshed, "
        f"{len(report['failed'])} failed in {report['total_seconds']} s")
    return report


def revert(conn, databases: Iterable[str] = DEFAULT_DATABASES, *, log: Callable[[str], None] = None) -> list:
    """Back to plain views: <NAME> = the __SRC definition again."""
    log = log or logger.info
    raw = lambda sql: conn._t.run(sql).rows
    done = []
    for it in discover(raw, databases).values():
        if it.state != "materialized":
            continue
        raw(f"DROP TABLE {_fq(it.db, it.name)}")
        raw(f"CREATE VIEW {_fq(it.db, it.name)} AS {it.sql}")
        raw(f"DROP VIEW {_fq(it.db, it.name + SRC_SUFFIX)}")
        done.append(".".join(it.key))
        log(f"reverted {'.'.join(it.key)}")
    return done


def materialized_keys(view_keys: Iterable[tuple]) -> set:
    """(SCHEMA, NAME) upper of materialized views, from a set of view keys (their __SRC views)."""
    return {(s, v[: -len(SRC_SUFFIX)]) for s, v in view_keys if v.endswith(SRC_SUFFIX)}
