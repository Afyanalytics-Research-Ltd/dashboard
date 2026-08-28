"""
Dynamic chart generation: an LLM writes matplotlib code tailored to the
user's actual request and the shape of the data, instead of the fixed
"bar-or-line" heuristic in charts.py:build_chart().

Safety model — this is the sensitive part, so it is deliberately layered:

1. The LLM only ever sees a pandas DataFrame built from a Cube.js result
   that has ALREADY been fetched under the requesting user's row/column
   access scope (RLS/CLS applied upstream, before this module runs). It
   never gets DB, network, or filesystem access of any kind.
2. The code it writes is executed through warehouse.agent.sandbox.execute()
   — the exact AST-validated, restricted-builtins, wall-clock-guarded
   sandbox already used (and tested) for the spreadsheet analyst feature.
   Imports, dunder access, eval/exec/open/__import__ etc. are rejected
   before a single line runs, regardless of what the model tried to write.
3. ANY failure — validation rejection, a runtime exception, a timeout, or
   the code simply not producing a usable `fig` — is caught here and
   reported as "no dynamic chart", never raised. The caller (charts.py)
   always has the deterministic build_chart() to fall back to, so a
   hostile or broken LLM response can degrade the chart, but can never
   crash the request or escape the sandbox.

This module has no view/consumer-layer knowledge — it takes data in and
returns a chart dict (or None) out, same shape as build_chart().
"""

from __future__ import annotations

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from openai import OpenAI

from warehouse.agent.sandbox import execute as sandbox_execute

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        from django.conf import settings

        api_key = getattr(settings, "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _setting(name: str, default):
    from django.conf import settings

    return getattr(settings, name, default)


_SYSTEM_PROMPT = """You write ONE short Python snippet that builds a matplotlib chart from a \
pandas DataFrame named `df`, already loaded in your execution namespace, in response to a \
specific user request.

Rules:
- `df`, `pd` (pandas), `np` (numpy) and `plt` (matplotlib.pyplot) are already available.
  Do not import anything — there is no import statement available to you at all.
- Assign the finished Figure object to a variable named exactly `fig`. Do not call
  plt.show() or plt.savefig() yourself.
- Pick whichever chart type best matches the user's request and the data's shape: bar,
  horizontal bar, line, pie, scatter, grouped/stacked bar, or histogram. If the request
  doesn't specify a type, default to the simplest chart that makes the data easy to read.
- Use ONLY the exact column names given to you below — never invent or guess one.
- Reference columns with df['Column Name'] bracket notation, since names may contain spaces.
- Label the axes, add a legend when there is more than one series, and rotate long x tick
  labels 45 degrees if there are more than ~8 of them.
- Keep it to one Figure with one set of axes.
- Output ONLY the Python code. No explanation, no markdown code fences, no comments needed.
"""


def _dataframe_from_raw_result(
    raw_result: dict,
    computed_measure: str | None,
    computed_label: str | None,
    max_rows: int,
) -> tuple[pd.DataFrame, bool] | tuple[None, bool]:
    """Build a DataFrame with human-friendly column names from a Cube result.

    Returns (df, truncated). df is None if there is nothing chartable.
    """
    from .charts import _clean_label  # reuse the same title-cleaning logic

    rows = raw_result.get("data") or []
    if not rows:
        return None, False

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    df = pd.DataFrame(rows)
    if df.empty:
        return None, False

    query = raw_result.get("query") or {}
    dimensions = query.get("dimensions") or []
    time_dims = [td["dimension"] for td in (query.get("timeDimensions") or []) if td.get("dimension")]
    measures = [computed_measure] if computed_measure else (query.get("measures") or [])

    if not measures:
        # Nothing to plot on a value axis — same "not chart-worthy" call
        # build_chart() makes, and worth making here too so an unchartable
        # result never costs an OpenAI call for a chart that can't exist.
        return None, False

    rename: dict[str, str] = {}
    for key in dimensions:
        rename[key] = _clean_label(raw_result, key, "dimensions")
    for key in time_dims:
        rename[key] = _clean_label(raw_result, key, "timeDimensions")
    for key in measures:
        if key is None:
            continue
        if computed_measure and key == computed_measure:
            rename[key] = computed_label or key.replace("_", " ").title()
        else:
            rename[key] = _clean_label(raw_result, key, "measures")

    # Coerce measure columns to numeric — Cube returns them as strings.
    for key in measures:
        if key and key in df.columns:
            df[key] = pd.to_numeric(df[key], errors="coerce")

    # Guard against a rename collision (two source keys cleaning to the same
    # label) silently merging two distinct columns into one.
    seen: dict[str, int] = {}
    deduped_rename: dict[str, str] = {}
    for key, label in rename.items():
        if key not in df.columns:
            continue
        count = seen.get(label, 0)
        seen[label] = count + 1
        deduped_rename[key] = label if count == 0 else f"{label} ({count + 1})"

    df = df.rename(columns=deduped_rename)
    # Drop any raw Cube key not covered above (e.g. an ungrouped id column)
    # so the LLM only ever sees clean, chartable columns.
    keep = list(deduped_rename.values())
    df = df[[c for c in keep if c in df.columns]]

    return df, truncated


def _generate_chart_code(df: pd.DataFrame, question: str, metric_name: str, model: str) -> str | None:
    """Ask the LLM for a matplotlib snippet. Returns None on any API failure."""
    columns_desc = "\n".join(f"- {c!r} (dtype: {df[c].dtype})" for c in df.columns)
    sample = df.head(5).to_string(index=False)

    user_prompt = f"""User's request: {question or '(no specific chart type requested — pick the clearest one for this data)'}

Metric context: {metric_name or '(none given)'}

DataFrame `df` has {len(df):,} row(s) and these columns:
{columns_desc}

First few rows:
{sample}
"""

    try:
        completion = _openai().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=700,
            temperature=0.2,
        )
    except Exception:
        logger.exception("chart_codegen: OpenAI call failed")
        return None

    code = (completion.choices[0].message.content or "").strip()
    if not code:
        return None

    # The model is told not to, but strip markdown fences defensively —
    # they would otherwise be a guaranteed SyntaxError in the sandbox.
    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()

    if code:
        # Logged unconditionally — before validation/execution — so the
        # exact snippet the model wrote is visible in the logs regardless
        # of whether the sandbox later accepts, rejects, or fails it. The
        # WARNING logged on rejection (in render_dynamic_chart) has the
        # error; this has the code that produced it.
        logger.info(
            "chart_codegen: generated snippet (model=%s, question=%r):\n%s",
            model, question, code,
        )

    return code or None


def render_dynamic_chart(
    raw_result: dict,
    question: str = "",
    metric_name: str = "",
    computed_measure: str | None = None,
    computed_label: str | None = None,
) -> dict | None:
    """
    Try to build a chart via LLM-generated matplotlib code, sandboxed.

    Returns a chart dict identical in shape to build_chart()'s
    ({"image_base64", "mime", "caption"}), or None on ANY failure —
    the caller is expected to fall back to build_chart() in that case.
    Never raises.
    """
    try:
        max_rows = _setting("CHART_CODEGEN_MAX_ROWS", 2000)
        df, truncated = _dataframe_from_raw_result(raw_result, computed_measure, computed_label, max_rows)
        if df is None or df.empty:
            return None

        model = _setting("CHART_CODEGEN_MODEL", "gpt-4o-mini")
        code = _generate_chart_code(df, question, metric_name, model)
        if not code:
            return None

        namespace = {"df": df, "pd": pd, "np": np, "plt": plt}
        timeout = _setting("CHART_CODEGEN_TIMEOUT", 8)

        result = sandbox_execute(code, namespace, timeout=timeout)
        if not result.ok:
            logger.warning("chart_codegen: sandboxed snippet rejected/failed: %s", result.error)
            return None

        fig = namespace.get("fig")
        if not isinstance(fig, Figure):
            logger.warning(
                "chart_codegen: generated code did not assign a Figure to `fig` (got %r)",
                type(fig).__name__,
            )
            return None

        import base64
        import io

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        buf.seek(0)

        caption = metric_name or (question.strip() or "Chart")
        if truncated:
            caption += f" (first {max_rows:,} rows)"

        logger.info("chart_codegen: dynamic chart succeeded (caption=%r)", caption)
        return {
            "image_base64": base64.b64encode(buf.read()).decode("ascii"),
            "mime": "image/png",
            "caption": caption,
        }
    except Exception:
        # Defense in depth: nothing above should raise, but a broken or
        # adversarial snippet must never be able to take the chat down.
        logger.exception("chart_codegen: unexpected error building dynamic chart")
        return None
    finally:
        # matplotlib.pyplot's figure registry is process-global — anything
        # the sandboxed code opened (via plt.figure()/plt.subplots(), even
        # ones it never assigned to `fig`) must be closed here, on every
        # path, or repeated chat requests leak memory indefinitely.
        plt.close("all")
