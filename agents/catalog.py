"""
Metric catalog loader.

Reads catalog/metrics.yaml and provides:
  - get_all()      → full list (for LLM context)
  - get_by_id(id)  → single metric dict (for resume path)
  - as_context()   → compact string injected into LLM prompt
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog" / "metrics.yaml"


@lru_cache(maxsize=1)
def _load() -> list[dict]:
    """Load and cache the catalog. Call reload() to bust the cache."""
    # import pdb;pdb.set_trace()
    with open(CATALOG_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data.get("metrics", [])


def reload() -> None:
    """Bust the cache — call after analytics team adds a new metric."""
    _load.cache_clear()


def get_all() -> list[dict]:
    return _load()


def get_by_id(metric_id: str) -> Optional[dict]:
    print(get_all())
    for m in get_all():
        if m["id"] == metric_id:
            return m
    return None


def as_context() -> str:
    """
    Returns a compact string listing all metrics, suitable for injection
    into an LLM system prompt.

    Example output:
        [total_revenue] Total Revenue — Total revenue across all products…
        [monthly_active_users] Monthly Active Users — Count of distinct users…
    """
    lines = []
    for m in get_all():
        desc = m["description"].strip().replace("\n", " ")
        lines.append(f"[{m['id']}] {m['name']} — {desc}")
    return "\n".join(lines)