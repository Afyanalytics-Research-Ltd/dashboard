"""Reusable dashboard building blocks: headers, KPI tiles, priority labels,
plain data-quality notes, and table column configs."""
from __future__ import annotations

import html
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from inventory_intelligence.dashboard import data_access, theme


def priority_words(scores: pd.Series) -> pd.Series:
    """Continuous urgency score → plain priority words, from the top slices of
    the values on screen (no fixed thresholds)."""
    numeric = pd.to_numeric(scores, errors="coerce")
    ranks = numeric.rank(pct=True)
    out = pd.Series("Fine for now", index=scores.index, dtype=object)
    out[ranks > 0.5] = "Keep watching"
    out[ranks > 0.8] = "Order soon"
    out[ranks > 0.95] = "Order now"
    out[numeric.isna()] = "—"
    return out


def page_header(title: str, subtitle: str = "") -> None:
    bits = []
    as_of = data_access.soh_as_of()
    written = data_access.analytics_written_at()
    if as_of:
        bits.append(f"Data through {data_access.pretty_date(as_of)}")
    if written is not None:
        bits.append(f"updated {written:%d %b %Y}")
    meta = f'<div class="sph-meta">{" · ".join(bits)}</div>' if bits else ""
    sub = f'<p class="sph-sub">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="sph-header"><div class="sph-title">{html.escape(title)}</div>'
        f"{sub}{meta}</div>",
        unsafe_allow_html=True,
    )


def section_header(text: str, finding: Optional[str] = None) -> None:
    """An analytical chapter header. ``finding`` optionally renders the section's
    headline result as an accent sub-line, so the hierarchy itself carries the
    argument (e.g. ``finding="82% sits in just 3 items"``)."""
    find = f'<div class="sph-sec-find">{html.escape(finding)}</div>' if finding else ""
    st.markdown(
        f'<div class="sph-sec"><div class="sph-sec-title">{html.escape(text)}</div>{find}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list[dict]) -> None:
    """Row of KSH-style KPI tiles. Each: {label, value, detail?, accent?, help?}."""
    tiles = []
    for m in metrics:
        accent = m.get("accent", theme.accent())
        detail = f'<div class="sph-tile-detail">{html.escape(str(m["detail"]))}</div>' if m.get("detail") else ""
        title_attr = f' title="{html.escape(str(m["help"]))}"' if m.get("help") else ""
        tiles.append(
            f'<div class="sph-tile" style="--accent:{accent}"{title_attr}>'
            f'<div class="sph-tile-label">{html.escape(str(m["label"]))}</div>'
            f'<div class="sph-tile-value">{html.escape(str(m["value"]))}</div>'
            f"{detail}</div>"
        )
    st.markdown(f'<div class="sph-kpis">{"".join(tiles)}</div>', unsafe_allow_html=True)


def badge(text: str, color: Optional[str] = None) -> str:
    color = color or theme.accent()
    return f'<span class="sph-badge" style="background:{color}">{html.escape(str(text))}</span>'


def cost_assumptions_note() -> None:
    if data_access.placeholder_inputs():
        st.caption(
            "Cost figures use provisional cost and holding assumptions, "
            "pending sign-off from finance."
        )


def missing_analytics_stop() -> None:
    st.info("The latest figures aren't ready yet. Please check back shortly.")
    st.stop()


def warehouse_required() -> None:
    if not data_access.snowflake_available():
        st.info("Live spending data is temporarily unavailable. Please try again shortly.")
        st.stop()


def empty_state(message: str, note: Optional[str] = None) -> None:
    st.info(message)
    if note:
        st.caption(note)


def category_sidebar_filter(df: Optional[pd.DataFrame] = None) -> list[str]:
    options = data_access.category_options(df)
    return st.sidebar.multiselect(
        "Product type", options=options, default=[], placeholder="All product types",
    )


def chance_column(label: str, help_text: str = "") -> "st.column_config.ProgressColumn":
    return st.column_config.ProgressColumn(
        label, help=help_text or "Chance of running out", min_value=0.0, max_value=1.0, format="percent",
    )


def kes_column(label: str, help_text: str = "") -> "st.column_config.NumberColumn":
    return st.column_config.NumberColumn(label, help=help_text, format="localized")
