"""
In-memory stand-ins for v3_to_snowflake_sync's two edges, shared by its unit and Snowflake tests.

FakeV3Client — a V3 tenant whose rows a test can add, edit and delete between runs.
FakeSink     — a Snowflake target with the same contract as SnowflakeSink (MERGE only rewrites a row whose
               hash changed or that was flagged deleted; flag_deleted honours the guard). The real SQL is
               checked against Snowflake in test_v3_to_snowflake_sync_snowflake.py.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import v3_to_snowflake_sync as S


class FakeV3Client:
    def __init__(self, tables: dict[tuple[str, str], list[dict]] | None = None, organization_id: int = 6,
                 fail_on: set[tuple[str, str]] | None = None, fail_after: int | None = None):
        self.tables = {k: [dict(r) for r in v] for k, v in (tables or {}).items()}
        self.organization_id = organization_id
        self.fail_on = fail_on or set()
        self.fail_after = fail_after          # raise after yielding this many rows of a failing model
        self.reads: list[tuple[str, str]] = []

    def token(self) -> str:
        return "token"

    def models(self, service: str) -> list[dict]:
        return [{"alias": a, "operations": ["read", "insert"]} for (s, a) in self.tables if s == service]

    def read_all(self, service: str, alias: str, per_page: int = 500, workers: int = 1):
        self.reads.append((service, alias))
        for i, row in enumerate(self.tables.get((service, alias), [])):
            if (service, alias) in self.fail_on and i == (self.fail_after or 0):
                raise RuntimeError(f"gateway 500 on {service}/{alias}")
            yield deepcopy(row)

    # test helpers ----------------------------------------------------------------
    def rows(self, service: str, alias: str) -> list[dict]:
        return self.tables.setdefault((service, alias), [])

    def edit(self, service: str, alias: str, row_id, **changes) -> None:
        for r in self.rows(service, alias):
            if r.get("id") == row_id:
                r.update(changes)
                return
        raise KeyError(row_id)

    def delete(self, service: str, alias: str, row_id) -> dict:
        rows = self.rows(service, alias)
        for i, r in enumerate(rows):
            if r.get("id") == row_id:
                return rows.pop(i)
        raise KeyError(row_id)


class FakeSink:
    def __init__(self):
        self.tables: dict[str, dict[tuple[int, str], dict]] = {}
        self.columns: dict[str, dict[str, str]] = {}          # table → {column: type}, like the real table
        self.state: dict[tuple[int, str], dict] = {}
        self.merges: list[tuple[str, int]] = []       # (table, batch size) per MERGE call
        self.fail_merge_calls: set[int] = set()       # MERGE call numbers (1-based) that raise
        self.ensure_schema_calls = 0

    def ensure_schema(self) -> None:
        self.ensure_schema_calls += 1

    def ensure_table(self, table: str, v3_columns: dict[str, str]) -> dict[str, str]:
        self.tables.setdefault(table, {})
        cols = self.columns.setdefault(table, {})
        for name, sf_type in v3_columns.items():
            cols.setdefault(S.column_name(name), sf_type)
        return cols

    def get_watermark(self, tenant_id: int, model: str) -> datetime | None:
        return (self.state.get((tenant_id, model)) or {}).get("watermark")

    def save_state(self, tenant_id, model, source, *, watermark, status, error, started, counts) -> None:
        self.state[(tenant_id, model.name)] = {"watermark": watermark, "status": status, "error": error,
                                               "source": source, "counts": dict(counts), "table": model.table}

    def merge(self, table: str, tenant_id: int, source: str, rows: list[S.StagedRow]) -> tuple[int, int]:
        self.merges.append((table, len(rows)))
        if len(self.merges) in self.fail_merge_calls:
            raise RuntimeError("snowflake: warehouse suspended")
        self.ensure_table(table, S.column_types(r.row for r in rows))
        t = self.tables[table]
        inserted = updated = 0
        for r in rows:
            current = t.get((tenant_id, r.key))
            if current is None:
                t[(tenant_id, r.key)] = {"row": r, "deleted": False, "source": source}
                inserted += 1
            elif current["row"].row_hash != r.row_hash or current["deleted"]:
                t[(tenant_id, r.key)] = {"row": r, "deleted": False, "source": source}
                updated += 1
        return inserted, updated

    def flag_deleted(self, table: str, tenant_id: int, seen_keys: list[str], max_fraction: float):
        t = self.tables.setdefault(table, {})
        seen = set(seen_keys)
        live = [k for k, v in t.items() if k[0] == tenant_id and not v["deleted"]]
        gone = [k for k in live if k[1] not in seen]
        skipped, flagged = None, 0
        if gone and live and len(gone) / len(live) > max_fraction:
            skipped = f"{len(gone)} of {len(live)} live rows were not read back — over the guard"
        else:
            for k in gone:
                t[k]["deleted"] = True
            flagged = len(gone)
        restored = 0
        for k, v in t.items():
            if k[0] == tenant_id and v["deleted"] and k[1] in seen:
                v["deleted"] = False
                restored += 1
        return flagged, restored, skipped

    # test helpers ----------------------------------------------------------------
    def live_keys(self, table: str, tenant_id: int = 6) -> set[str]:
        return {k[1] for k, v in self.tables.get(table, {}).items() if k[0] == tenant_id and not v["deleted"]}

    def deleted_keys(self, table: str, tenant_id: int = 6) -> set[str]:
        return {k[1] for k, v in self.tables.get(table, {}).items() if k[0] == tenant_id and v["deleted"]}


def patient(i: int, updated: str = "2026-10-01T08:00:00.000000Z", **extra) -> dict:
    return {"id": i, "uuid": f"00000000-0000-0000-0000-{i:012d}", "first_name": f"P{i}", "organization_id": 6,
            "created_at": "2026-09-01T08:00:00.000000Z", "updated_at": updated, "deleted_at": None, **extra}
