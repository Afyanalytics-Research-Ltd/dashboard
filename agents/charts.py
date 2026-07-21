"""
Chart generation from Cube.dev semantic-layer results.

Two things live here:
  - is_heavy() / build_chart() — pure functions, used by nodes.format_result()
    to decide whether to *offer* a chart, and to actually render one.
  - get_chart_for_thread() — pulls a previous query's raw Cube result back out
    of the LangGraph checkpoint (keyed by thread_id) and renders it. This is
    what lets "yes, show me a chart" work as a follow-up turn without
    re-running the query against Cube.
"""

from __future__ import annotations

import base64
import io
import logging
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Below this many rows, a chart adds little over the text summary already given.
HEAVY_ROW_THRESHOLD = 15

# Shared across every channel (chat websocket, WhatsApp, and any future REST
# caller) so "does this message mean yes / show me a chart" is answered the
# same way everywhere, rather than three slightly-different regexes drifting
# apart over time.
_AFFIRMATIVE_REPLIES = frozenset({
    'yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay', 'please', 'please do',
    'go ahead', 'do it', 'visualize', 'visualise', 'chart', 'show me', 'show it',
})

_VISUALIZE_PATTERNS = [
    r'\bgraph(s|ic)?\b', r'\bchart(s|ing)?\b', r'\bplot\b',
    r'\bvisuali[sz]e\b', r'\bvisuali[sz]ation\b',
    r'\bshowcase\b', r'\bdisplay\b', r'\bpictures?\b', r'\bimage\b',
    r'\bshow\s+(it|that|me|this|here)\b', r'\bsee\s+(it|that)\b', r'\blet\s+me\s+see\b',
]


def is_affirmative_reply(text: str) -> bool:
    """True if `text` reads as a plain "yes" to a just-made offer."""
    return text.strip().lower().rstrip('.!') in _AFFIRMATIVE_REPLIES


def wants_visualization(text: str) -> bool:
    """True if `text` explicitly asks for a graph/chart/plot, unprompted."""
    q = text.lower()
    return any(re.search(p, q) for p in _VISUALIZE_PATTERNS)


def is_heavy(raw_result: dict | None) -> bool:
    """True if the result set is large enough that a chart is worth offering."""
    rows = (raw_result or {}).get("data") or []
    return len(rows) > HEAVY_ROW_THRESHOLD


def _annotation_title(raw_result: dict, key: str, field: str) -> str:
    ann = (raw_result.get("annotation") or {}).get(key, {})
    return (ann.get(field) or {}).get("title", "")


def _clean_label(raw_result: dict, field_key: str, section: str) -> str:
    """
    Human-friendly label for a Cube measure/dimension key, e.g.
    "fact_inpatient_admissions.sex" -> "Sex".

    Cube auto-generates a title of "{Cube Title} {Field Title}" whenever no
    explicit `title` is set in the cube's schema (e.g. "Fact Inpatient
    Admissions Sex") — that's an internal-naming artifact, not something an
    analyst actually wrote, and reads badly in a chart meant for end users.
    If the annotation title is exactly that auto-generated pattern, use the
    shorter field-only name instead; otherwise trust the (real, custom)
    title as given.
    """
    cube_name, _, field_name = field_key.partition(".")
    field_title = field_name.replace("_", " ").title()
    cube_title = cube_name.replace("_", " ").title()

    annotation_title = _annotation_title(raw_result, section, field_key)
    if not annotation_title:
        return field_title
    if annotation_title.strip().lower() == f"{cube_title} {field_title}".strip().lower():
        return field_title
    return annotation_title


_PALETTE = ["#0072CE", "#0BB99F", "#F59E0B", "#EF4444", "#8B5CF6"]


def _label(value) -> str:
    return "Unknown" if value is None else str(value)


def build_chart(raw_result: dict, metric_name: str = "") -> dict | None:
    """
    Render a chart image from a Cube API response.

    Chart shape follows the query's own shape (as resolved by Cube, not
    guessed from free text):
      - a timeDimension + 1+ measures -> line chart over time (one line per measure)
      - a dimension + 1+ measures     -> bar chart (grouped bars if >1 measure)
      - anything else                 -> None (not chart-worthy — e.g. a
                                          single scalar KPI with no axis at all)

    Dimensions already pinned to one value by a filter (e.g.
    "source_schema = KISUMU_CLEAN") carry no visual information, so they're
    skipped in favour of a dimension that actually varies across rows.

    Returns:
        {"image_base64": "...", "mime": "image/png", "caption": "..."},
        or None if this result's shape doesn't suit a simple chart.
    """
    query = raw_result.get("query") or {}
    rows = raw_result.get("data") or []
    if not rows:
        return None

    measures = query.get("measures") or []
    dimensions = query.get("dimensions") or []
    time_dims = [td["dimension"] for td in (query.get("timeDimensions") or []) if td.get("dimension")]

    if not measures:
        logger.info("build_chart: query has no measures — nothing to chart")
        return None

    varying_dims = [d for d in dimensions if len({r.get(d) for r in rows}) > 1]

    if time_dims:
        x_key, chart_kind, x_section = time_dims[0], "line", "timeDimensions"
    elif varying_dims:
        x_key, chart_kind, x_section = varying_dims[0], "bar", "dimensions"
    elif dimensions:
        # No dimension actually varies (e.g. a single-row result) — still
        # usable as a labelled x-axis, just not a very interesting one.
        x_key, chart_kind, x_section = dimensions[0], "bar", "dimensions"
    else:
        logger.info("build_chart: no dimension/timeDimension to use as an axis")
        return None

    x_label = _clean_label(raw_result, x_key, x_section)
    x_values = [_label(r.get(x_key)) for r in rows]

    series: dict[str, list[float]] = {}
    for m in measures:
        try:
            series[m] = [float(r.get(m) or 0) for r in rows]
        except (TypeError, ValueError) as exc:
            logger.warning("build_chart: could not coerce values for %s — %s", m, exc)
            return None

    measure_labels = {m: _clean_label(raw_result, m, "measures") for m in measures}

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)

    if chart_kind == "line":
        for i, m in enumerate(measures):
            ax.plot(x_values, series[m], marker="o", label=measure_labels[m], color=_PALETTE[i % len(_PALETTE)])
    else:
        n = len(measures)
        positions = list(range(len(x_values)))
        width = 0.8 / n
        for i, m in enumerate(measures):
            offsets = [p + (i - (n - 1) / 2) * width for p in positions]
            ax.bar(offsets, series[m], width=width, label=measure_labels[m], color=_PALETTE[i % len(_PALETTE)])
        ax.set_xticks(positions)
        rotate = len(x_values) > 8
        ax.set_xticklabels(x_values, rotation=45 if rotate else 0, ha="right" if rotate else "center")

    if len(measures) > 1:
        ax.legend(fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(measure_labels[measures[0]] if len(measures) == 1 else "Value")
    ax.set_title(metric_name or (measure_labels[measures[0]] if len(measures) == 1 else " / ".join(measure_labels.values())))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    caption_metric = metric_name or ", ".join(measure_labels.values())
    return {
        "image_base64": base64.b64encode(buf.read()).decode("ascii"),
        "mime": "image/png",
        "caption": f"{caption_metric} by {x_label}",
    }


def get_chart_for_thread(thread_id: str) -> tuple[dict | None, str | None]:
    """
    Look up a previous query's result by thread_id and render a chart.

    Works for both the REST flow (/api/query/ then /api/visualize/) and the
    chat flow (agent graph invocation then a "yes" follow-up), since both
    invoke the same graph/checkpointer keyed by thread_id.

    Returns (chart, error) — exactly one is non-None.
    """
    from .graph import graph  # local import: avoids a module-load cycle with graph.py

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        return None, "Thread not found."

    raw_result = snapshot.values.get("raw_result")
    if not raw_result:
        return None, "No result available to chart."

    metric = snapshot.values.get("matched_metric") or {}
    chart = build_chart(raw_result, metric_name=metric.get("name", ""))
    if not chart:
        return None, (
            "This result isn't a good fit for a chart (try asking a follow-up "
            "question with a breakdown by date or category instead)."
        )
    return chart, None
