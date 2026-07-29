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

    Each metric's real filterable field names (dimensions + timeDimensions)
    are listed alongside it — without these, a filter-extracting LLM has no
    grounding for what "CubeName.dimensionName" values are actually valid,
    and will fall back to copying/adapting whatever example it was last
    shown (e.g. inventing "Dispensing.date" from a "Dispensing.facility"
    example, even for a metric with no such cube at all).

    Date fields are called out separately from other dimensions — a plain
    dimension (e.g. a facility code) must never be targeted by a date-range
    operator, and a metric with no date field at all must never get a
    date-range filter applied to it, so the model needs to see the
    distinction rather than one flat list of "filterable fields."

    Example output:
        [total_revenue] Total Revenue — Total revenue across all products…
          Dimension fields: total_revenue.region
          Date field (for date-range filters only): total_revenue.month
    """
    lines = []
    for m in get_all():
        desc = m["description"].strip().replace("\n", " ")
        cube_query = m.get("cube_query") or {}
        dims = list(cube_query.get("dimensions") or [])
        date_fields = [
            td["dimension"] for td in (cube_query.get("timeDimensions") or []) if td.get("dimension")
        ]
        dims_text = ", ".join(dims) if dims else "none"
        date_text = ", ".join(date_fields) if date_fields else "none — do not apply a date-range filter to this metric"
        lines.append(
            f"[{m['id']}] {m['name']} — {desc}\n"
            f"  Dimension fields: {dims_text}\n"
            f"  Date field (for date-range filters only): {date_text}"
        )
    return "\n".join(lines)