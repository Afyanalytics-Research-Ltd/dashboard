"""Sync engine: source rows → mapping rules → destination tables.

Modes (Fivetran / Airbyte vocabulary)
  full_refresh_overwrite   rebuild every destination table from scratch
  full_refresh_append      read everything, append
  incremental_append       read rows whose cursor column is past the saved cursor, append
  incremental_dedupe       incremental + replace rows with the same primary key

State is ``{target_table: {"cursor": <last value>, "cursor_column": "..."}}``;
the caller persists it between runs.
"""

from __future__ import annotations

import datetime as dt
import decimal
import time
from typing import Any, Callable

from . import rules

MODES = ["full_refresh_overwrite", "full_refresh_append", "incremental_append", "incremental_dedupe"]
CURSOR_HINTS = ("updated_at", "modified_at", "last_modified", "date_updated", "updated", "modified", "created_at",
                "date_created", "created", "inserted_at", "timestamp", "id")

Progress = Callable[..., None]


class SyncCancelled(Exception):
    pass


def pick_cursor(src_table: dict) -> str | None:
    names = {c["name"].lower(): c["name"] for c in src_table["columns"]}
    for hint in CURSOR_HINTS:
        if hint in names:
            return names[hint]
    return None


def _gt(a: Any, b: Any) -> bool:
    """a > b, tolerating a saved (string) cursor compared with a live typed value."""
    try:
        return a > b
    except TypeError:
        return str(a) > str(b)


_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off", ""}


def coerce(value: Any, family: str) -> Any:
    if value is None:
        return None
    try:
        if family == "integer":
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int,)):
                return value
            f = float(str(value).strip())
            return int(f)
        if family == "decimal":
            if isinstance(value, (int, float, decimal.Decimal)):
                return value
            return float(str(value).replace(",", "").strip())
        if family == "boolean":
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            return True if s in _TRUE else False if s in _FALSE else None
        if family in ("date", "datetime"):
            if isinstance(value, (dt.datetime, dt.date)):
                return value if family == "datetime" or not isinstance(value, dt.datetime) else value.date()
            import pandas as pd
            ts = pd.to_datetime(str(value), errors="coerce")
            if pd.isna(ts):
                return None
            return ts.to_pydatetime() if family == "datetime" else ts.date()
        if family == "json":
            return value
        if family in ("text", "uuid", "other", "unknown"):
            if isinstance(value, (dict, list)):
                import json
                return json.dumps(value, default=str)
            return value if isinstance(value, str) else str(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value


def run(*, source, destination, doc: dict, source_snap: dict, target_snap: dict, mode: str = "full_refresh_overwrite",
        state: dict | None = None, settings: dict | None = None, progress: Progress = lambda **k: None,
        cancelled: Callable[[], bool] = lambda: False, batch_size: int = 2000, row_limit: int | None = None) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown sync mode {mode}")
    state = dict(state or {})
    settings = settings or {}
    only = set(settings.get("tables") or [])
    cursor_cols = settings.get("cursors") or {}
    src_idx = {t["key"]: t for t in source_snap["tables"]}
    tgt_idx = {t["key"]: t for t in target_snap["tables"]}
    writer = destination.writer()
    incremental = mode.startswith("incremental")
    pairs = [p for p in doc.get("tables", []) if p.get("source") and (not only or p["target"] in only)]
    report = {"mode": mode, "tables": [], "rows": 0, "started_at": dt.datetime.utcnow().isoformat() + "Z"}

    for n, pair in enumerate(pairs, start=1):
        if cancelled():
            raise SyncCancelled()
        t0 = time.time()
        tkey, skey = pair["target"], pair["source"]
        fields = [f for f in doc["fields"] if f["target_table"] == tkey and f["status"] != "rejected"]
        entry = {"target": tkey, "source": skey, "rows": 0, "status": "ok", "error": None}
        report["tables"].append(entry)
        if not fields:
            entry.update(status="skipped", error="no mapped columns")
            continue
        tt, st = tgt_idx.get(tkey), src_idx.get(skey)
        if not tt or not st:
            entry.update(status="failed", error="table missing from the snapshots")
            continue
        tcols = {c["name"]: c for c in tt["columns"]}
        out_cols = [{"name": f["target"], "family": (tcols.get(f["target"]) or {}).get("family") or "text"} for f in fields]
        pk = [c for c in (tt.get("primary_key") or []) if c in {f["target"] for f in fields}]
        progress(msg=f"[{n}/{len(pairs)}] {skey} → {tkey}", phase="sync", pct=100 * (n - 1) / max(len(pairs), 1))
        try:
            table = writer.ensure_table(tt["name"], out_cols, primary_key=pk, replace=(mode == "full_refresh_overwrite"))
            # lookups: read each referenced source table once
            lookups = {}
            for f in fields:
                if f["kind"] == "looked up in another table":
                    r = f["rule"]
                    key = (r.get("table"), r.get("key"), r.get("value"))
                    if key not in lookups:
                        found = {}
                        for batch in source.read(r["table"], columns=[r["key"], r["value"]], batch_size=10000):
                            for row in batch:
                                if row.get(r["key"]) not in (None, ""):
                                    found[str(row[r["key"]])] = row.get(r["value"])
                        lookups[key] = found
            needed = sorted({f["source"] for f in fields if f.get("source")} |
                            ({pair["where"]["column"]} if pair.get("where") else set()))
            cursor_col = None
            cursor_val = None
            if incremental:
                cursor_col = cursor_cols.get(tkey) or (state.get(tkey) or {}).get("cursor_column") or pick_cursor(st)
                if not cursor_col:
                    raise ValueError("incremental sync needs a cursor column (set one in the connection settings)")
                if cursor_col not in needed:
                    needed.append(cursor_col)
                cursor_val = (state.get(tkey) or {}).get("cursor")
            where = pair.get("where")
            allowed = {str(v).lower() for v in where["in"]} if where else None
            max_cursor = cursor_val
            written = 0
            for batch in source.read(skey, columns=needed, cursor_column=cursor_col, cursor_value=cursor_val,
                                     batch_size=batch_size, limit=row_limit):
                if cancelled():
                    raise SyncCancelled()
                out = []
                for row in batch:
                    if allowed is not None and str(row.get(where["column"])).lower() not in allowed:
                        continue
                    rec = {}
                    for f in fields:
                        val = rules.apply(f, row.get(f["source"]) if f.get("source") else None, lookups)
                        rec[f["target"]] = coerce(val, (tcols.get(f["target"]) or {}).get("family") or "text")
                    out.append(rec)
                    if cursor_col is not None:
                        cv = row.get(cursor_col)
                        if cv is not None and (max_cursor is None or _gt(cv, max_cursor)):
                            max_cursor = cv
                written += writer.write(table, out, merge_keys=pk if mode == "incremental_dedupe" else None)
                progress(msg=f"{tkey}: {written:,} rows", level="debug", detail={"target": tkey, "rows": written})
            entry["rows"] = written
            report["rows"] += written
            if cursor_col:
                state[tkey] = {"cursor": max_cursor.isoformat() if hasattr(max_cursor, "isoformat") else max_cursor,
                               "cursor_column": cursor_col}
                entry["cursor"] = state[tkey]["cursor"]
        except SyncCancelled:
            raise
        except Exception as e:
            entry.update(status="failed", error=str(getattr(e, "orig", None) or e)[:500])
            progress(msg=f"{tkey} failed: {entry['error']}", level="error")
        entry["seconds"] = round(time.time() - t0, 2)
    report["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
    report["failed"] = sum(1 for t in report["tables"] if t["status"] == "failed")
    report["state"] = state
    return report
