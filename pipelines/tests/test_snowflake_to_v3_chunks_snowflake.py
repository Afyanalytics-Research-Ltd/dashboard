"""
snowflake_to_v3_migration.iter_clean_chunks against REAL Snowflake — the dedupe window and the keyset
paging SQL. Read-only: the "CLEAN view" is an inline VALUES table; nothing is created or written.

    S2V3_SNOWFLAKE_IT=1 python -m pytest tests/test_snowflake_to_v3_chunks_snowflake.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import snowflake_to_v3_migration as m

pytestmark = pytest.mark.skipif(os.getenv("S2V3_SNOWFLAKE_IT") != "1",
                                reason="set S2V3_SNOWFLAKE_IT=1 to run against real Snowflake")

# (ID, UUID, UPDATED_AT, NAME) — every awkward case at once
ROWS = [
    (1, "a", "2026-01-01", "old"), (1, "a", "2026-02-01", "new"),      # re-extracted snapshot: newest wins
    (2, None, "2026-01-01", "pre-uuid"), (2, "b", "2026-01-02", "b"),  # same record before/after V2's uuids
    (3, "c", "2026-01-01", "c"), (3, "d", "2026-01-01", "d"),          # two uuids share an id
    (4, None, "2026-01-01", "no uuid ever"),
    (5, "h", "2026-01-01", "h"),
    (None, "e", "2026-01-01", "e"), (None, "f", "2026-01-01", "f"), (None, "g", "2026-01-01", "g"),
]
EXPECTED = {(1, "a", "new"), (2, "b", "b"), (3, "c", "c"), (3, "d", "d"), (4, None, "no uuid ever"),
            (5, "h", "h"), (None, "e", "e"), (None, "f", "f"), (None, "g", "g")}


def _values_sql() -> str:
    def lit(v):
        return "NULL" if v is None else (str(v) if isinstance(v, int) else f"'{v}'")
    body = ", ".join(f"({', '.join(lit(v) for v in r)})" for r in ROWS)
    return f'SELECT * FROM (VALUES {body}) AS t("ID", "UUID", "UPDATED_AT", "NAME")'


@pytest.fixture(scope="module", autouse=True)
def _env():
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")
    os.chdir(root)


@pytest.fixture
def clean_view(monkeypatch):
    monkeypatch.setattr(m, "sf_schema", lambda f, layer: "HOSPITALS.UNUSED_CLEAN")
    monkeypatch.setattr(m, "_without_twin_rows", lambda cur, f, t, base: _values_sql())


@pytest.mark.parametrize("chunk_rows", [1, 2, 3, 4, 100])
def test_every_record_exactly_once_whatever_the_page_size(clean_view, chunk_rows):
    pages = list(m.iter_clean_chunks("f", "any_table", chunk_rows=chunk_rows))
    rows = [r for page in pages for r in page]
    got = [(r["id"], r["uuid"], r["name"]) for r in rows]
    assert len(got) == len(set(got)) == len(EXPECTED), got          # nothing twice, nothing lost
    assert {(int(i) if i is not None else None, u, n) for i, u, n in got} == EXPECTED
    assert all(len(p) <= chunk_rows for p in pages)
    assert not any(m._SORT_KEY.lower() in r or m._SORT_KEY in r for r in rows)   # paging key never leaks
