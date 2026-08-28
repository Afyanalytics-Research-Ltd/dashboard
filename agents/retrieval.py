"""
Runtime semantic retriever.

Loads catalog/embeddings.npz (built offline by catalog/build_embeddings.py)
once per process and answers "what measures/dimensions/metrics/glossary
terms are semantically related to this text?" via brute-force cosine
similarity — no vector DB needed at this catalog's scale (low hundreds of
rows total), see the retrieval-store decision in the implementation plan.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from django.conf import settings
from openai import OpenAI

logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path(__file__).resolve().parent.parent / "catalog" / "embeddings.npz"
EMBEDDING_MODEL = "text-embedding-3-small"

_openai_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


@dataclass
class _EmbeddingsIndex:
    vectors: np.ndarray   # shape (N, D), float32, L2-normalized at build time
    ids: list             # length N
    sources: list          # length N, one of: metric | measure | dimension | glossary
    metadata: list         # length N, parallel list of dicts


_index_cache: _EmbeddingsIndex | None = None
_index_cache_mtime: float | None = None


def _load_index() -> _EmbeddingsIndex:
    global _index_cache, _index_cache_mtime

    if not EMBEDDINGS_PATH.exists():
        raise RuntimeError(
            f"{EMBEDDINGS_PATH} not found — run "
            f"'python manage.py rebuild_embeddings' first to build the semantic-retrieval index."
        )

    # A rebuild (Settings page / `manage.py rebuild_embeddings`) overwrites
    # this file on disk from a DIFFERENT process than the one serving
    # requests — nothing else in this codebase calls reload_index() to bust
    # a live web worker's in-memory cache, so without this stat() check a
    # freshly-curated metric/measure description (or a filter-value fix)
    # silently never reaches retrieval until every worker process happens
    # to restart. The stat() itself is cheap (page-cache hit) compared to
    # the embeddings call this feeds into, so checking on every call is fine.
    current_mtime = EMBEDDINGS_PATH.stat().st_mtime
    if _index_cache is None or current_mtime != _index_cache_mtime:
        with np.load(EMBEDDINGS_PATH, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata"]))
            _index_cache = _EmbeddingsIndex(
                vectors=data["vectors"],
                ids=list(data["ids"]),
                sources=list(data["sources"]),
                metadata=metadata,
            )
        _index_cache_mtime = current_mtime
        logger.info("retrieval: loaded %d embeddings from %s", len(_index_cache.ids), EMBEDDINGS_PATH)
    return _index_cache


def reload_index() -> None:
    """Bust the in-process cache — call after re-running build_embeddings.py without a restart."""
    global _index_cache
    _index_cache = None


def _embed_query(text: str) -> np.ndarray:
    resp = _openai().embeddings.create(model=EMBEDDING_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm else v


def _to_candidate(index: _EmbeddingsIndex, i: int, score: float) -> dict:
    meta = index.metadata[i]
    return {
        "source": index.sources[i],
        "id": index.ids[i],
        "cube": meta.get("cube"),
        "field": meta.get("field"),
        "kind": meta.get("kind"),
        "cube_measure_type": meta.get("cube_measure_type", ""),
        "label": meta.get("label", ""),
        "description": meta.get("description", ""),
        "score": score,
        "metric_id": meta.get("metric_id"),
        "glossary_term": meta.get("glossary_term"),
        "formula": meta.get("formula"),
        "variables": meta.get("variables"),
    }


def retrieve(query_text: str, top_k: int = 8) -> list[dict]:
    """
    Brute-force cosine-similarity search over the embeddings index.
    Returns up to top_k candidates, sorted by score descending.
    """
    index = _load_index()
    qvec = _embed_query(query_text)
    scores = index.vectors @ qvec  # cosine similarity, since both sides are unit-normed
    k = min(top_k, len(scores))
    top_idx = np.argsort(-scores)[:k]
    return [_to_candidate(index, int(i), float(scores[i])) for i in top_idx]


def retrieve_many(query_texts: list[str], top_k: int = 8) -> list[dict]:
    """
    Run retrieve() for each text (e.g. the raw question plus each of the
    intent planner's candidate_terms) and merge results, keeping the
    highest score seen for each candidate id, sorted descending.
    """
    best: dict[str, dict] = {}
    for text in query_texts:
        if not text:
            continue
        for candidate in retrieve(text, top_k=top_k):
            existing = best.get(candidate["id"])
            if existing is None or candidate["score"] > existing["score"]:
                best[candidate["id"]] = candidate
    return sorted(best.values(), key=lambda c: -c["score"])[:top_k]
