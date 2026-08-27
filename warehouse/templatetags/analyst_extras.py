"""Template filters for the spreadsheet analyst pages.

Model output is Markdown. It is rendered server-side and *sanitised* - the
text comes from an LLM, and an LLM's output is untrusted input like any
other. With `bleach` installed the HTML is whitelisted; without it, the
filter falls back to escaped plain text rather than shipping raw HTML to the
browser.
"""

from __future__ import annotations

from django import template
from django.utils.html import escape, linebreaks
from django.utils.safestring import mark_safe

register = template.Library()

ALLOWED_TAGS = [
    "p", "br", "strong", "em", "code", "pre", "blockquote",
    "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td", "hr", "a",
]
ALLOWED_ATTRS = {"a": ["href", "title"]}


@register.filter(name="render_markdown")
def render_markdown(text: str) -> str:
    """Markdown -> sanitised HTML. Degrades safely if deps are missing."""
    if not text:
        return ""

    try:
        import markdown as md
    except ImportError:
        return mark_safe(linebreaks(escape(text)))  # noqa: S308 - escaped first

    html = md.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html",
    )

    try:
        import bleach
    except ImportError:
        # No sanitiser available - do not trust model output with raw HTML.
        return mark_safe(linebreaks(escape(text)))  # noqa: S308 - escaped first

    cleaned = bleach.clean(
        html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True
    )
    return mark_safe(bleach.linkify(cleaned))  # noqa: S308 - sanitised above


@register.filter(name="tool_label")
def tool_label(name: str) -> str:
    return {
        "list_sheets": "Listed sheets",
        "profile_sheet": "Profiled a sheet",
        "run_python": "Ran pandas",
        "make_chart": "Drew a chart",
        "export_table": "Exported a table",
        "write_report": "Wrote the report",
    }.get(name, name)


@register.filter(name="tool_icon")
def tool_icon(name: str) -> str:
    return {
        "list_sheets": "bi-list-columns-reverse",
        "profile_sheet": "bi-clipboard-data",
        "run_python": "bi-code-slash",
        "make_chart": "bi-bar-chart-line",
        "export_table": "bi-file-earmark-excel",
        "write_report": "bi-file-earmark-text",
    }.get(name, "bi-gear")
