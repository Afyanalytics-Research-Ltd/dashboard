"""pgvector store + hybrid retriever for the support knowledge base.

Retrieval = vector search (bge-small, cosine) and keyword search (Postgres full text)
fused with reciprocal-rank fusion, then re-ranked with a local cross-encoder.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

import psycopg2
import psycopg2.extras

DIM = 384  # BAAI/bge-small-en-v1.5

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS kb_chunks (
    id           text PRIMARY KEY,
    source       text NOT NULL,
    title        text NOT NULL,
    breadcrumb   text NOT NULL,
    url_path     text NOT NULL DEFAULT '',
    content      text NOT NULL,
    content_hash text NOT NULL,
    embedding    vector({DIM}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (
                   setweight(to_tsvector('english', title || ' ' || breadcrumb || ' ' || url_path), 'A') ||
                   setweight(to_tsvector('english', content), 'B')) STORED,
    updated_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx ON kb_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS kb_chunks_tsv_idx ON kb_chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS kb_chunks_source_idx ON kb_chunks (source);

CREATE TABLE IF NOT EXISTS chat_history (
    id         bigserial PRIMARY KEY,
    session_id text NOT NULL,
    role       text NOT NULL,
    content    text NOT NULL,
    meta       jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_history_session_idx ON chat_history (session_id, id);
"""


def db_url() -> str:
    url = os.getenv("SUPPORT_KB_DB_URL", "").strip()
    if not url:
        raise RuntimeError("SUPPORT_KB_DB_URL is not set")
    return url


def connect():
    return psycopg2.connect(db_url())


def _vec(v: Iterable[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA)
    conn.commit()


def existing_hashes(conn) -> dict[str, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, content_hash FROM kb_chunks")
        return dict(cur.fetchall())


def upsert(conn, rows: list[tuple]) -> None:
    """rows: (id, source, title, breadcrumb, url_path, content, content_hash, embedding list)"""
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            INSERT INTO kb_chunks (id, source, title, breadcrumb, url_path, content, content_hash, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source, title = EXCLUDED.title,
              breadcrumb = EXCLUDED.breadcrumb, url_path = EXCLUDED.url_path, content = EXCLUDED.content,
              content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding, updated_at = now()
        """, [r[:7] + (_vec(r[7]),) for r in rows], template="(%s,%s,%s,%s,%s,%s,%s,%s::vector)")
    conn.commit()


def delete_missing(conn, keep_ids: set[str]) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM kb_chunks")
        stale = [r[0] for r in cur.fetchall() if r[0] not in keep_ids]
        if stale:
            cur.execute("DELETE FROM kb_chunks WHERE id = ANY(%s)", (stale,))
    conn.commit()
    return len(stale)


_STOP = set("a an the is are was were be to of in on for and or how do does i we you my our can cant "
            "can't what why when where which who it this that with from at by as not no".split())


def _keyword_query(q: str) -> str:
    """OR-query of the meaningful words (prefix-matched), so one rare word is enough to hit."""
    words = [w for w in re.findall(r"[a-zA-Z0-9_.]{2,}", q.lower()) if w not in _STOP][:12]
    return " | ".join(f"{re.sub(r'[^a-z0-9_]', '', w)}:*" for w in words if re.sub(r'[^a-z0-9_]', '', w))


HYBRID_SQL = """
WITH v AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(emb)s::vector) AS r
    FROM kb_chunks {where}
    ORDER BY embedding <=> %(emb)s::vector LIMIT %(k)s
), kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, q) DESC) AS r
    FROM kb_chunks, to_tsquery('english', %(tsq)s) q
    WHERE %(tsq)s <> '' AND tsv @@ q {and_where}
    ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %(k)s
)
SELECT c.id, c.source, c.title, c.breadcrumb, c.url_path, c.content,
       COALESCE(1.0 / (60 + v.r), 0) + COALESCE(1.0 / (60 + kw.r), 0) AS rrf
FROM kb_chunks c
LEFT JOIN v USING (id) LEFT JOIN kw USING (id)
WHERE v.id IS NOT NULL OR kw.id IS NOT NULL
ORDER BY rrf DESC
LIMIT %(k)s
"""


def search(query: str, *, top_k: int = 6, candidates: int = 20, sources: list[str] | None = None,
           rerank: bool = True) -> list[dict]:
    from afya_backends import llm

    emb = llm.local_embed([query], query=True)[0]
    params = {"emb": _vec(emb), "tsq": _keyword_query(query), "k": candidates}
    where = and_where = ""
    if sources:
        params["sources"] = list(sources)
        where, and_where = "WHERE source = ANY(%(sources)s)", "AND source = ANY(%(sources)s)"
    try:
        with connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(HYBRID_SQL.format(where=where, and_where=and_where), params)
            rows = [dict(r) for r in cur.fetchall()]
    except psycopg2.Error:
        # a malformed keyword query must never take retrieval down: fall back to vector only
        params["tsq"] = ""
        with connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(HYBRID_SQL.format(where=where, and_where=and_where), params)
            rows = [dict(r) for r in cur.fetchall()]
    if rerank and rows:
        # the cross-encoder reads ~512 tokens; the context header + first ~1300 chars is what it can use
        scores = llm.local_rerank(query, [r["content"][:1300] for r in rows])
        for r, s in zip(rows, scores):
            r["score"] = s
        rows.sort(key=lambda r: -r["score"])
    else:
        for r in rows:
            r["score"] = float(r["rrf"])
    for r in rows:
        r["rrf"] = float(r["rrf"])
    return rows[:top_k]
