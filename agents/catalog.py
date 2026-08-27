"""
Metric catalog loader.

Reads MetricDefinition rows from the database (managed via the Agent
Configuration settings page / Django admin — see agents/views.py,
agents/admin.py) and provides:
  - get_all()      → full list (for LLM context)
  - get_by_id(id)  → single metric dict (for resume path)
  - as_context()   → compact string injected into LLM prompt

catalog/metrics.yaml was the source of truth before this; it was imported
once into the DB via agents/migrations/0004_import_metrics_yaml.py and is no
longer read here.

No caching: writes to MetricDefinition now come from an ordinary web
request (the settings page), not only from the re_classify resume path that
used to call reload() right after editing the YAML file in the same
process — so an in-process cache would go stale in every OTHER worker
process until it happened to resume a thread, which may never happen for a
fresh query. The table is small (curated metrics, not raw data), so a plain
per-call query is cheap and removes that whole bug class instead of trying
to keep a cache correctly invalidated across processes.
"""

from __future__ import annotations

from typing import Optional

from .models import MetricDefinition


def reload() -> None:
    """
    No-op — kept only so existing call sites (agents/nodes.py's
    re_classify) don't need to change. There's no cache to bust anymore;
    get_all()/get_by_id() always query the DB directly.
    """


def _as_dict(m: MetricDefinition) -> dict:
    return {
        "id": m.metric_id,
        "name": m.name,
        "description": m.description,
        "cube_query": m.cube_query or {},
    }


def get_all() -> list[dict]:
    return [_as_dict(m) for m in MetricDefinition.objects.filter(is_active=True)]


def get_by_id(metric_id: str) -> Optional[dict]:
    m = MetricDefinition.objects.filter(metric_id=metric_id, is_active=True).first()
    return _as_dict(m) if m else None


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
