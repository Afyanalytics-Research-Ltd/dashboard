"""
Reusable Streamlit UI components.
All components that need custom styling use st.markdown with injected CSS classes.
Call inject_css() once per page before rendering any components.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from utils.formatting import (
    ACTION_COLORS,
    COLOR_BORDER,
    COLOR_PRIMARY,
    COLOR_RED,
    COLOR_TEXT_MUTED,
    CONFIDENCE_COLORS,
    PRIORITY_COLORS,
)


# ── Global CSS ────────────────────────────────────────────────────────────────

_CSS = """
<style>
/* ── Hide Streamlit auto-nav ───────────────────────── */
section[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"] { display: none !important; }

/* Use more of the viewport — trim Streamlit's default side/top whitespace */
.main .block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    max-width: 1600px;
    /* top clears Streamlit's 60px fixed header so the page title stays visible */
    padding: 4.5rem 2.5rem 3rem;
}

/* ── Page header ───────────────────────────────────── */
.page-header {
    padding-bottom: 14px;
    margin-bottom: 20px;
    border-bottom: 1px solid #E5E7EB;
}
.page-title {
    font-size: 28px;
    font-weight: 800;
    color: #111827;
    margin: 0 0 4px;
    line-height: 1.2;
}
.page-subtitle { font-size: 14px; color: #9CA3AF; margin: 0; line-height: 1.5; }

/* ── Section header ────────────────────────────────── */
.section-header {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9CA3AF;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #E5E7EB;
}

/* ── KPI tiles ─────────────────────────────────────── */
.kpi-tile {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-top: 3px solid #E5E7EB;  /* accent overridden inline */
    border-radius: 10px;
    padding: 14px 14px 12px;
}
.kpi-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9CA3AF;
    margin-bottom: 6px;
    line-height: 1.4;              /* allow wrapping — no truncation */
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
    word-break: break-word;        /* allow wrapping — no truncation */
}
.kpi-delta {
    font-size: 12px;
    font-weight: 600;
    margin-top: 4px;
}

/* ── Action cards ──────────────────────────────────── */
.action-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 3px solid #E5E7EB;  /* accent overridden inline */
    border-radius: 0 10px 10px 0;
    padding: 11px 14px;
    margin-bottom: 6px;
}
.action-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.action-body { flex: 1; min-width: 0; }
.action-drug {
    font-weight: 600;
    font-size: 14px;
    color: #111827;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.action-reason {
    font-size: 13px;
    color: #9CA3AF;
    margin-top: 2px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.action-badges {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 3px;
    flex-shrink: 0;
}
.action-badge {
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    color: #fff;
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.priority-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 3px;
    color: #fff;
    text-transform: uppercase;
    opacity: 0.85;
}

/* ── AI summary ────────────────────────────────────── */
.ai-summary {
    background: #F0FAF6;
    border: 1px solid #C3E8D8;
    border-left: 4px solid #0F6E56;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 15px;
    line-height: 1.65;
    color: #111827;
    margin-bottom: 16px;
}
.ai-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #0F6E56;
    margin-bottom: 6px;
}

/* ── Anomaly banner ────────────────────────────────── */
.anomaly-banner {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 3px solid #D97706;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.anomaly-title {
    font-weight: 700;
    font-size: 13px;
    color: #92400E;
    margin-bottom: 3px;
}

/* ── Empty state ───────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: #9CA3AF;
    font-size: 14px;
    background: #FAFAFA;
    border: 1.5px dashed #E5E7EB;
    border-radius: 10px;
}
.empty-state-icon { font-size: 28px; margin-bottom: 10px; }

/* ── Inline badges ─────────────────────────────────── */
.badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 4px;
    color: #fff;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Generic cards ─────────────────────────────────── */
.afya-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.afya-card-accent {
    background: #FFFFFF;
    border-left: 4px solid #0F6E56;
    border-top: 1px solid #E5E7EB;
    border-right: 1px solid #E5E7EB;
    border-bottom: 1px solid #E5E7EB;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 8px;
}

/* ── Stat strip (briefing page KPIs) ───────────────────── */
.stat-strip {
    display: flex;
    background: #fff;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 16px;
}
.stat-item {
    flex: 1;
    padding: 12px 16px 10px;
    border-right: 1px solid #E5E7EB;
    min-width: 0;
}
.stat-item:last-child { border-right: none; }
.stat-label {
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #9CA3AF;
    margin-bottom: 5px;
    line-height: 1.3;
    min-height: 2.3em;            /* reserve 2 lines so values stay aligned */
    white-space: normal;          /* wrap the full label instead of truncating */
}
.stat-value {
    font-size: 27px;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}
.stat-hint {
    font-size: 12px;
    font-weight: 600;
    margin-top: 3px;
}

/* ── AI Decision cards ─────────────────────────────────── */
.decision-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 4px solid #E5E7EB;  /* accent overridden inline */
    border-radius: 0 10px 10px 0;
    padding: 12px 14px 10px;
    margin-bottom: 2px;
}
.decision-drug {
    font-size: 15px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 3px;
}
.decision-meta {
    font-size: 12px;
    color: #6B7280;
    margin-bottom: 6px;
    line-height: 1.4;
}
.decision-narrative {
    font-size: 13px;
    color: #374151;
    line-height: 1.55;
    margin-bottom: 0;
}
.decision-ai-badge {
    font-size: 10px;
    font-weight: 700;
    color: #0F6E56;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Anomaly analysis box ───────────────────────────── */
.anomaly-analysis {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-top: 4px;
    font-size: 13px;
    color: #374151;
    line-height: 1.65;
}
.anomaly-analysis-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #D97706;
    margin-bottom: 5px;
}

/* ── Traceability card ─────────────────────────────── */
.trace-card {
    background: #FAFBFF;
    border: 1px solid #C7D2FE;
    border-left: 4px solid #3730A3;
    border-radius: 0 10px 10px 0;
    padding: 16px 20px;
    margin-top: 8px;
}
.trace-header-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.10em;
    color: #3730A3;
    margin-bottom: 6px;
}
.trace-drug-name {
    font-size: 15px;
    font-weight: 700;
    color: #1E1B4B;
    margin-bottom: 14px;
}
.trace-cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    /* retained for future use if 2-col layout is re-enabled */
}
.trace-section {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin: 10px 0 5px;
    padding-bottom: 3px;
    border-bottom: 1px solid #E5E7EB;
}
.trace-section:first-child { margin-top: 0; }
.trace-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 2px 0;
    border-bottom: 1px dotted #F3F4F6;
}
.trace-key  { color: #6B7280; }
.trace-val  { font-weight: 600; color: #111827; text-align: right; }
.trace-formula-line {
    font-size: 12.5px;
    font-family: 'Consolas', 'Courier New', monospace;
    color: #374151;
    background: #EEF2FF;
    border-radius: 4px;
    padding: 4px 8px;
    margin: 4px 0 2px;
}
.trace-formula-step {
    font-size: 12px;
    color: #6B7280;
    font-family: 'Consolas', 'Courier New', monospace;
    padding: 1px 8px;
}
.trace-formula-answer {
    font-size: 14px;
    font-weight: 700;
    color: #1E1B4B;
    font-family: 'Consolas', 'Courier New', monospace;
    padding: 4px 8px;
    background: #C7D2FE;
    border-radius: 4px;
    margin-top: 4px;
    display: inline-block;
}
.trace-confidence-pill {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
/* ── Data quality banner ────────────────────────────── */
.dq-banner {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 4px solid #D97706;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    margin-bottom: 12px;
    font-size: 13px;
}
.dq-banner-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #92400E;
    margin-bottom: 4px;
}
.dq-banner-item { color: #78350F; padding: 1px 0; }

/* ── Insight cards (Phase 2) ────────────────────────── */
.insight-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 4px solid #E5E7EB;   /* accent overridden inline */
    border-radius: 0 10px 10px 0;
    padding: 12px 16px 10px;
    margin-bottom: 8px;
}
.insight-severity {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 5px;
}
.insight-headline {
    font-size: 13px;
    font-weight: 700;
    color: #111827;
    line-height: 1.4;
    margin-bottom: 4px;
}
.insight-narration {
    font-size: 12px;
    color: #374151;
    line-height: 1.5;
    margin-bottom: 6px;
    font-style: italic;
}
.insight-facts {
    margin: 0 0 8px;
    padding: 0;
    list-style: none;
}
.insight-facts li {
    font-size: 11px;
    color: #6B7280;
    padding: 1px 0;
}
.insight-facts li::before {
    content: "· ";
    color: #9CA3AF;
}
.insight-action-chip {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    background: #F0FDF4;
    color: #166534;
    border: 1px solid #86EFAC;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Sidebar ───────────────────────────────────────── */
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
.sidebar-facility {
    font-size: 13px;
    font-weight: 700;
    color: #0F6E56;
    padding: 2px 0;
    line-height: 1.4;
}
.sidebar-date { font-size: 11px; color: #9CA3AF; }
</style>
"""


def inject_css() -> None:
    """Inject global design-system CSS. Call once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Page chrome ───────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str, facility_label: str = "", is_live: bool = True) -> None:
    st.markdown(
        f"""
        <div class="page-header">
          <div class="page-title">{title}</div>
          <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ── KPI tiles ─────────────────────────────────────────────────────────────────

def kpi_row(metrics: list[dict]) -> None:
    """
    Render a horizontal KPI strip.

    Each metric dict supports:
      label        str   — KPI label
      value        str   — formatted value string
      delta        str   — optional sub-label line
      delta_good   bool  — True → teal delta, False → red delta
      accent_color str   — top-border colour (auto-derived from delta_good if omitted)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            # Derive accent colour: explicit override → delta_good fallback → neutral
            if "accent_color" in m:
                accent = m["accent_color"]
            elif "delta_good" in m:
                accent = COLOR_PRIMARY if m["delta_good"] else COLOR_RED
            else:
                accent = COLOR_BORDER

            delta_html = ""
            if m.get("delta"):
                good = m.get("delta_good", True)
                delta_color = COLOR_PRIMARY if good else COLOR_RED
                delta_html = (
                    f'<div class="kpi-delta" style="color:{delta_color}">'
                    f'{m["delta"]}</div>'
                )

            st.markdown(
                f"""
                <div class="kpi-tile" style="border-top-color:{accent}">
                  <div>
                    <div class="kpi-label">{m['label']}</div>
                    <div class="kpi-value">{m['value']}</div>
                  </div>
                  {delta_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Stat strip ───────────────────────────────────────────────────────────────

def stat_strip(metrics: list[dict]) -> None:
    """
    Render a flat horizontal stat strip.
    Each metric dict: label, value, hint (optional), hint_good (bool), accent_color (optional).
    Designed for briefing-page KPIs — replaces the 2-row kpi_row grid.
    """
    items_html = ""
    for m in metrics:
        accent = m.get("accent_color", "#111827")
        hint_html = ""
        if m.get("hint"):
            good = m.get("hint_good", True)
            hc = COLOR_PRIMARY if good else COLOR_RED
            hint_html = f'<div class="stat-hint" style="color:{hc}">{m["hint"]}</div>'
        items_html += (
            f'<div class="stat-item">'
            f'<div class="stat-label">{m["label"]}</div>'
            f'<div class="stat-value" style="color:{accent}">{m["value"]}</div>'
            f'{hint_html}'
            f'</div>'
        )
    st.markdown(f'<div class="stat-strip">{items_html}</div>', unsafe_allow_html=True)


# ── AI summary ────────────────────────────────────────────────────────────────

def ai_summary_box(text: str) -> None:
    st.markdown(
        f'<div class="ai-summary">'
        f'<div class="ai-label">✦ &nbsp;Situation summary</div>'
        f'{text}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Action cards ──────────────────────────────────────────────────────────────

def action_cards(actions: list[dict]) -> None:
    """
    Render a vertical list of action cards.
    Each dict: action, canonical_name, reason, clinical_priority (optional)

    Design: coloured left-border + dot indicate severity; no emoji icons.
    Action badge + priority badge stacked on the right.
    """
    for a in actions:
        act   = a.get("action", "MONITOR")
        color = ACTION_COLORS.get(act, "#888780")
        cp    = a.get("clinical_priority", "")

        priority_html = ""
        if cp:
            cp_color = PRIORITY_COLORS.get(cp, "#888780")
            priority_html = (
                f'<span class="priority-badge" style="background:{cp_color}">{cp}</span>'
            )

        st.markdown(
            f"""
            <div class="action-card" style="border-left-color:{color}">
              <div class="action-dot" style="background:{color}"></div>
              <div class="action-body">
                <div class="action-drug">{a.get('canonical_name', '—')}</div>
                <div class="action-reason">{a.get('reason', '')}</div>
              </div>
              <div class="action-badges">
                <span class="action-badge" style="background:{color}">{act}</span>
                {priority_html}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── AI Decision cards ─────────────────────────────────────────────────────────

def decision_card_ai(
    canonical_name: str,
    action: str,
    dos_remaining: Optional[float],
    order_qty: int,
    cost_estimate_kes: Optional[float],
    stockout_gap_days: int,
    narrative: str,
    is_ai: bool,
    color: str,
    confidence: Optional[str] = None,
) -> None:
    """
    Compact AI decision card: first sentence visible, full reasoning behind expander.
    confidence: HIGH | MEDIUM | LOW — shown as a coloured badge if provided.
    """
    from utils.formatting import fmt_int
    dos_str  = f"{dos_remaining:.0f}d remaining" if (dos_remaining and dos_remaining > 0) else "Stocked out"
    qty_str  = f"Order {fmt_int(order_qty)} units" if order_qty > 0 else "Qty: estimate in Workbench"
    cost_str = f" · ~KES {cost_estimate_kes:,.0f}" if cost_estimate_kes else ""
    gap_str  = f" · {stockout_gap_days}d gap during delivery" if stockout_gap_days > 0 else ""
    ai_badge = '<span class="decision-ai-badge">✦ AI</span>' if is_ai else ""

    # Confidence badge — coloured chip showing forecast data quality.
    # Prefixed with "CONF:" so it can't be mistaken for clinical priority.
    # HIGH=green (reliable forecast), MEDIUM=amber, LOW=red (sparse/stocked-out data).
    _conf_styles = {
        "HIGH":   "background:#DCFCE7;color:#166534;border:1px solid #86EFAC",
        "MEDIUM": "background:#FEF3C7;color:#92400E;border:1px solid #FDE68A",
        "LOW":    "background:#FEE2E2;color:#991B1B;border:1px solid #FECACA",
    }
    conf_badge = ""
    if confidence:
        _lvl = confidence.upper()
        _style = _conf_styles.get(_lvl, "background:#F3F4F6;color:#6B7280;border:1px solid #E5E7EB")
        conf_badge = (
            f'<span style="display:inline-block;{_style};border-radius:4px;'
            f'padding:1px 7px;font-size:9px;font-weight:700;letter-spacing:0.05em;'
            f'margin-left:6px;vertical-align:middle">CONF: {_lvl}</span>'
        )

    # First sentence only for the compact view
    dot_idx = narrative.find(".")
    if 0 < dot_idx < len(narrative) - 1:
        first_sentence = narrative[: dot_idx + 1].strip()
        remainder      = narrative[dot_idx + 1 :].strip()
    else:
        first_sentence = narrative
        remainder      = ""

    _card_html = (
        f'<div class="decision-card" style="border-left-color:{color}">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:3px">'
        f'<div class="decision-drug">{canonical_name}{conf_badge}</div>'
        f'{ai_badge}'
        f'</div>'
        f'<div class="decision-meta">{dos_str} · {qty_str}{cost_str}{gap_str}</div>'
        f'<div class="decision-narrative">{first_sentence}</div>'
        f'</div>'
    )
    st.markdown(_card_html, unsafe_allow_html=True)
    if remainder:
        with st.expander("Full reasoning →"):
            st.markdown(
                f'<div style="font-size:12px;color:#374151;line-height:1.65">{remainder}</div>',
                unsafe_allow_html=True,
            )


# ── Inline badges ─────────────────────────────────────────────────────────────

def status_badge(status: str) -> str:
    """Return inline HTML badge string for stock status."""
    from utils.formatting import STATUS_COLORS
    color = STATUS_COLORS.get(status.lower(), "#888780")
    return f'<span class="badge" style="background:{color}">{status.upper()}</span>'


def priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority.upper(), "#888780")
    return f'<span class="badge" style="background:{color}">{priority}</span>'


def confidence_pill(level: str) -> str:
    color = CONFIDENCE_COLORS.get(level.upper(), "#888780")
    return f'<span class="badge" style="background:{color};opacity:0.85">{level}</span>'


# ── Anomaly banner ────────────────────────────────────────────────────────────

def anomaly_banner(canonical_name: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="anomaly-banner">
          <div class="anomaly-title">⚠ &nbsp;{canonical_name}</div>
          <div style="font-size:12px;color:#78350F">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Insight cards (Phase 2) ──────────────────────────────────────────────────

# Severity → left-border colour
_INSIGHT_SEV_COLORS = {
    "CRITICAL": "#DC2626",   # red-600
    "HIGH":     "#D97706",   # amber-600
    "MEDIUM":   "#0F6E56",   # Afya teal
}
_INSIGHT_SEV_LABEL_COLORS = {
    "CRITICAL": "#991B1B",
    "HIGH":     "#92400E",
    "MEDIUM":   "#065F46",
}


def insight_card(row: "Any") -> None:
    """
    Render a compact Phase 2 InsightCard for Today's Briefing.
    Shows: severity badge · headline · top 2 facts · action chip.

    Args:
        row: An InsightRow from intelligence.insight_engine.detect_all()
    """
    sev    = str(getattr(row, "severity", "MEDIUM")).upper()
    head   = str(getattr(row, "headline", ""))
    facts  = list(getattr(row, "supporting_facts", []))[:2]   # max 2 bullets
    action = str(getattr(row, "recommended_action", "Review"))

    border_col = _INSIGHT_SEV_COLORS.get(sev, "#9CA3AF")
    label_col  = _INSIGHT_SEV_LABEL_COLORS.get(sev, "#374151")

    facts_items = "".join(f"<li>{f}</li>" for f in facts)
    facts_block = f'<ul class="insight-facts">{facts_items}</ul>' if facts_items else ""

    card_html = (
        f'<div class="insight-card" style="border-left-color:{border_col}">'
        + f'<div class="insight-severity" style="color:{label_col}">{sev}</div>'
        + f'<div class="insight-headline">{head}</div>'
        + facts_block
        + f'<span class="insight-action-chip">{action}</span>'
        + '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ── Empty state ───────────────────────────────────────────────────────────────

def empty_state(message: str, icon: str = "📭") -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-state-icon">{icon}</div>
          <div style="max-width:360px;margin:0 auto">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar navigation (shared across all pages) ─────────────────────────────

def sidebar_nav(fac=None) -> None:
    """
    Render the full navigation sidebar.
    Call once at the top of every page. fac = FacilityMeta or None (benchmark).
    """
    import os
    with st.sidebar:
        # ── Logo ─────────────────────────────────────────────
        if os.path.exists("ksh_logo.png"):
            st.image("ksh_logo.png", use_container_width=True)
            st.markdown(
                "<hr style='margin:8px 0 6px;border:none;border-top:1px solid #E5E7EB'>",
                unsafe_allow_html=True,
            )

        # ── Facility header ───────────────────────────────────
        if fac is not None:
            live_dot  = "●" if fac.is_live else "◷"
            dot_color = COLOR_PRIMARY if fac.is_live else COLOR_TEXT_MUTED
            st.markdown(
                f"""
                <div class="sidebar-facility">
                  <span style="color:{dot_color}">{live_dot}</span>
                  &nbsp;{fac.label}
                </div>
                <div class="sidebar-date">{fac.date_range}</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="sidebar-facility">Afyanalytics</div>'
                f'<div class="sidebar-date">Cross-facility view</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            "<hr style='margin:10px 0 6px;border:none;border-top:1px solid #E5E7EB'>",
            unsafe_allow_html=True,
        )

        # ── Navigation ────────────────────────────────────────
        st.page_link("ksh_inventory_intelligence.py", label="Today's Briefing",   icon="📋")
        st.page_link("pages/1_order_workbench.py",  label="Order Workbench",    icon="🛒")
        st.page_link("pages/2_stockout_watch.py",   label="Stockout Watch",     icon="⚠️")
        st.page_link("pages/3_dead_stock.py",       label="Dead Stock Actions", icon="📦")
        st.page_link("pages/4_patient_risk.py",     label="Patient Risk",       icon="🩺")
        st.page_link("pages/5_demand_insights.py",  label="Demand Insights",    icon="📈")
        st.page_link("pages/6_compliance_log.py",   label="Compliance Log",     icon="📜")

        st.markdown(
            "<hr style='margin:6px 0 10px;border:none;border-top:1px solid #E5E7EB'>",
            unsafe_allow_html=True,
        )

        # ── AI provider status ────────────────────────────────
        try:
            from intelligence.ai_client import get_provider, last_error
            _provider = get_provider()
            _provider_label = {
                "groq":   ("✦ Groq",   "#F55036", "#FEF0EE"),
                "grok":   ("✦ Grok",   "#0F6E56", "#E6F4EE"),
                "claude": ("✦ Claude", "#6B48FF", "#F0EEFF"),
                "none":   ("○ AI offline", "#9CA3AF", "#F5F6FA"),
            }.get(_provider, ("○ AI offline", "#9CA3AF", "#F5F6FA"))
            st.markdown(
                f"<div style='font-size:10px;font-weight:700;color:{_provider_label[1]};"
                f"background:{_provider_label[2]};padding:3px 8px;border-radius:4px;"
                f"text-align:center;margin-bottom:6px;letter-spacing:.04em'>"
                f"{_provider_label[0]}</div>",
                unsafe_allow_html=True,
            )
            _ai_err = last_error()
            if _ai_err:
                st.warning(f"AI error: {_ai_err}", icon="⚠️")
        except Exception:
            pass

        # ── Footer controls ───────────────────────────────────
        if st.button(
            "↺  Refresh data",
            use_container_width=True,
            key="_nav_refresh",
            type="secondary",
        ):
            st.cache_data.clear()
            st.rerun()


# ── Traceability card ─────────────────────────────────────────────────────────

def traceability_card(
    drug_name: str,
    clinical_priority: str,
    # Demand inputs
    avg_daily_units: float,
    std_daily_units: float,
    cv: float,
    demand_type: str = "UNKNOWN",
    adi: float = 0.0,
    cv_nz: float = 0.0,
    data_months: int = 0,
    confidence: str = "LOW",
    trend_direction: str = "STABLE",
    # Lead time inputs
    lt_mean: float = 14.0,
    lt_std: float = 0.0,
    lt_source: str = "",
    # Safety stock & order
    safety_stock_units: float = 0.0,
    z_value: float = 1.645,
    service_level: float = 0.95,
    rop: float = 0.0,
    current_soh: float = 0.0,
    target_cover_days: int = 30,
    order_qty: float = 0.0,
) -> None:
    """
    Full calculation trace panel for an order recommendation.
    Shows every labeled input and every formula step — nothing hidden.
    """
    import math

    # Derived values
    lt_p90 = lt_mean + 1.645 * lt_std
    trend_arrow = {"UP": "↑", "DOWN": "↓", "STABLE": "→"}.get(trend_direction.upper(), "→")
    trend_color = {"UP": "#D97706", "DOWN": "#DC2626", "STABLE": "#6B7280"}.get(trend_direction.upper(), "#6B7280")

    conf_colors = {"HIGH": ("#166534", "#DCFCE7"), "MEDIUM": ("#92400E", "#FEF3C7"), "LOW": ("#991B1B", "#FEE2E2")}
    cc, cbg = conf_colors.get(confidence.upper(), ("#6B7280", "#F3F4F6"))

    cp_colors = {"CRITICAL": "#A32D2D", "HIGH": "#854F0B", "STANDARD": "#0C447C"}
    cp_color = cp_colors.get(clinical_priority.upper(), "#6B7280")

    dt_colors = {
        "SMOOTH":       ("#166534", "#DCFCE7"),
        "ERRATIC":      ("#854F0B", "#FEF3C7"),
        "INTERMITTENT": ("#1D4ED8", "#EFF6FF"),
        "LUMPY":        ("#991B1B", "#FEE2E2"),
    }
    dt_c, dt_bg = dt_colors.get(demand_type.upper(), ("#6B7280", "#F3F4F6"))
    dt_desc = {
        "SMOOTH":       "frequent · stable quantity",
        "ERRATIC":      "frequent · variable quantity",
        "INTERMITTENT": "infrequent · stable quantity",
        "LUMPY":        "infrequent · variable quantity",
    }.get(demand_type.upper(), "")
    adi_label = f"{adi:.1f}d avg between dispenses" if adi > 0 else "—"

    # Formula components
    ss_var1 = lt_mean * std_daily_units ** 2
    ss_var2 = avg_daily_units ** 2 * lt_std ** 2
    ss_inner = math.sqrt(max(0.0, ss_var1 + ss_var2))
    cover_units = (target_cover_days + lt_mean) * avg_daily_units
    order_calc = max(0.0, cover_units - current_soh)  # SS sets WHEN to order (ROP), not HOW MUCH

    lt_obs_label = f"({lt_source})" if lt_source else ""

    inputs_html = f"""
<div class="trace-section">Demand</div>
<div class="trace-row"><span class="trace-key">Avg daily (μ)</span>
  <span class="trace-val">{avg_daily_units:.2f} units/day</span></div>
<div class="trace-row"><span class="trace-key">Demand σ</span>
  <span class="trace-val">{std_daily_units:.2f} units</span></div>
<div class="trace-row"><span class="trace-key">Demand pattern</span>
  <span class="trace-val">
    <span class="trace-confidence-pill" style="background:{dt_bg};color:{dt_c}">{demand_type}</span>
    &nbsp;<span style="color:#6B7280;font-weight:400;font-size:11px">{dt_desc}</span>
  </span></div>
<div class="trace-row"><span class="trace-key">Demand interval</span>
  <span class="trace-val">{adi_label}</span></div>
<div class="trace-row"><span class="trace-key">Trend</span>
  <span class="trace-val" style="color:{trend_color}">{trend_arrow} {trend_direction}</span></div>
<div class="trace-row"><span class="trace-key">Data history</span>
  <span class="trace-val">{data_months} months &nbsp;
    <span class="trace-confidence-pill" style="background:{cbg};color:{cc}">{confidence}</span>
  </span></div>

<div class="trace-section" style="margin-top:12px">Lead Time</div>
<div class="trace-row"><span class="trace-key">P50 (used in calc)</span>
  <span class="trace-val">{lt_mean:.1f}d &nbsp;<span style="color:#9CA3AF;font-size:10px">{lt_obs_label}</span></span></div>
<div class="trace-row"><span class="trace-key">P90 worst case</span>
  <span class="trace-val">{lt_p90:.1f}d</span></div>
<div class="trace-row"><span class="trace-key">Lead time σ</span>
  <span class="trace-val">{lt_std:.1f}d</span></div>

<div class="trace-section" style="margin-top:12px">Service Level</div>
<div class="trace-row"><span class="trace-key">Target</span>
  <span class="trace-val">{int(service_level * 100)}%</span></div>
<div class="trace-row"><span class="trace-key">Z-value</span>
  <span class="trace-val">{z_value:.3f}</span></div>
<div class="trace-row"><span class="trace-key">Current SOH</span>
  <span class="trace-val">{current_soh:,.0f} units</span></div>
<div class="trace-row"><span class="trace-key">Target cover</span>
  <span class="trace-val">{target_cover_days}d</span></div>
"""

    formula_html = f"""
<div class="trace-section">Safety Stock</div>
<div class="trace-formula-line">SS = Z × √(LT × σ²_d + μ² × σ²_LT)</div>
<div class="trace-formula-step">= {z_value:.3f} × √({lt_mean:.1f}×{std_daily_units:.2f}² + {avg_daily_units:.2f}²×{lt_std:.1f}²)</div>
<div class="trace-formula-step">= {z_value:.3f} × √({ss_var1:.0f} + {ss_var2:.0f})</div>
<div class="trace-formula-step">= {z_value:.3f} × {ss_inner:.1f}</div>
<div class="trace-formula-answer">SS = {safety_stock_units:.0f} units</div>

<div class="trace-section" style="margin-top:14px">Reorder Point</div>
<div class="trace-formula-line">ROP = μ × LT + SS</div>
<div class="trace-formula-step">= {avg_daily_units:.1f} × {lt_mean:.0f} + {safety_stock_units:.0f}</div>
<div class="trace-formula-answer">ROP = {rop:.0f} units</div>

<div class="trace-section" style="margin-top:14px">Order Quantity</div>
<div class="trace-formula-line">Qty = (Cover + LT) × μ − SOH</div>
<div class="trace-formula-step" style="color:#9CA3AF;font-size:10px">SS determines <em>when</em> to order (via ROP), not how much</div>
<div class="trace-formula-step">= ({target_cover_days} + {lt_mean:.1f}) × {avg_daily_units:.2f} − {current_soh:.0f}</div>
<div class="trace-formula-step">= {cover_units:.1f} − {current_soh:.0f} = {order_calc:.1f} → ceil</div>
<div class="trace-formula-answer">Order = {order_qty:.0f} units ✓</div>
"""

    # Always-visible: header + demand/LT inputs
    # NOTE: no blank lines inside the f-string — Streamlit markdown parser treats
    # blank lines before closing tags as paragraph breaks, rendering </div> as literal text.
    _drug_header = (
        f'<div class="trace-card">'
        f'<div class="trace-header-label">▸ Calculation trace</div>'
        f'<div class="trace-drug-name">{drug_name}'
        f'<span style="font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;'
        f'color:#fff;background:{cp_color};margin-left:8px">{clinical_priority}</span>'
        f'</div>'
        + inputs_html.strip()
        + '</div>'
    )
    st.markdown(_drug_header, unsafe_allow_html=True)

    # Formula steps: collapsed by default — one click to verify the maths
    with st.expander("Show formula breakdown →", expanded=False):
        st.markdown(
            '<div style="padding:4px 8px">' + formula_html.strip() + '</div>',
            unsafe_allow_html=True,
        )


# ── Data quality banner ────────────────────────────────────────────────────────

def data_quality_banner(flags: list[str]) -> None:
    """
    Render a collapsible data quality warning banner.
    flags: list of human-readable issue strings.
    Show nothing if flags is empty.
    """
    if not flags:
        return
    items_html = "".join(f'<div class="dq-banner-item">· {f}</div>' for f in flags)
    st.markdown(
        f"""
        <div class="dq-banner">
          <div class="dq-banner-title">⚠ Data quality flags ({len(flags)} issue{'s' if len(flags) != 1 else ''})</div>
          {items_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Historical data notice ────────────────────────────────────────────────────

def historical_notice(date_range: str) -> None:
    st.info(
        f"**Historical data** ({date_range}). Live alerts and order actions are disabled. "
        "Use this facility for analysis and benchmarking.",
        icon="📅",
    )
