"""
Business glossary loader.

Reads catalog/glossary.yaml — same load/cache/reload pattern as
agents/catalog.py — and provides:
  - get_all()          → full list of glossary entries
  - find_by_term(text) → exact/case-insensitive term-or-alias match
  - as_context()       → compact string, usable for LLM-prompt injection
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

GLOSSARY_PATH = Path(__file__).resolve().parent.parent / "catalog" / "glossary.yaml"


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    """Load and cache the glossary. Call reload() after editing the file."""
    with open(GLOSSARY_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data.get("terms", [])


def reload() -> None:
    """Bust the cache — call after an analyst edits catalog/glossary.yaml."""
    _load.cache_clear()


def get_all() -> list[dict]:
    return _load()


def find_by_term(text: str) -> Optional[dict]:
    """
    Exact (case-insensitive) match against a term or one of its aliases.
    Semantic/fuzzy matching is the retriever's job (agents/retrieval.py) —
    this is only the cheap, deterministic exact-match path.
    """
    needle = text.strip().lower()
    if not needle:
        return None
    for entry in get_all():
        if entry.get("term", "").strip().lower() == needle:
            return entry
        aliases = [a.strip().lower() for a in (entry.get("aliases") or [])]
        if needle in aliases:
            return entry
    return None


def as_context() -> str:
    """Compact string listing every glossary term, for optional LLM-prompt injection."""
    lines = []
    for entry in get_all():
        aliases = ", ".join(entry.get("aliases") or [])
        target = entry.get("maps_to") or f"formula: {entry.get('formula')}"
        lines.append(f"[{entry['term']}] (aka {aliases}) → {target} — {entry.get('description', '')}")
    return "\n".join(lines)
