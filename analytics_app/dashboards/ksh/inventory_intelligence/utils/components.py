"""
Reusable Streamlit UI components.
All components that need custom styling use st.markdown with injected CSS classes.
Call inject_css() once per page before rendering any components.
"""

from __future__ import annotations

from typing import Optional

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

.main .block-container { padding-top: 0.5rem; }

/* ── Page header ───────────────────────────────────── */
.page-header {
    padding-bottom: 14px;
    margin-bottom: 20px;
    border-bottom: 1px solid #E5E7EB;
}
.page-title {
    font-size: 26px;
    font-weight: 800;
    color: #111827;
    margin: 0 0 4px;
    line-height: 1.2;
}
.page-subtitle { font-size: 13px; color: #9CA3AF; margin: 0; line-height: 1.5; }

/* ── Section header ────────────────────────────────── */
.section-header {
    font-size: 11px;
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
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9CA3AF;
    margin-bottom: 6px;
    line-height: 1.4;              /* allow wrapping — no truncation */
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
    word-break: break-word;        /* allow wrapping — no truncation */
}
.kpi-delta {
    font-size: 11px;
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
    font-size: 13px;
    color: #111827;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.action-reason {
    font-size: 12px;
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
    font-size: 10px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    color: #fff;
    white-space: nowrap;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.priority-badge {
    font-size: 9px;
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
    font-size: 14px;
    line-height: 1.65;
    color: #111827;
    margin-bottom: 16px;
}
.ai-label {
    font-size: 10px;
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
    font-size: 12px;
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
    font-size: 10px;
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
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #9CA3AF;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
}
.stat-hint {
    font-size: 10px;
    font-weight: 600;
    margin-top: 2px;
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
    font-size: 13px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 3px;
}
.decision-meta {
    font-size: 11px;
    color: #6B7280;
    margin-bottom: 6px;
    line-height: 1.4;
}
.decision-narrative {
    font-size: 12px;
    color: #374151;
    line-height: 1.55;
    margin-bottom: 0;
}
.decision-ai-badge {
    font-size: 9px;
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
    font-size: 12px;
    color: #374151;
    line-height: 1.65;
}
.anomaly-analysis-label {
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #D97706;
    margin-bottom: 5px;
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
) -> None:
    """
    Compact AI decision card: first sentence visible, full reasoning behind expander.
    """
    from utils.formatting import fmt_int
    dos_str  = f"{dos_remaining:.0f}d remaining" if (dos_remaining and dos_remaining > 0) else "Stocked out"
    qty_str  = f"Order {fmt_int(order_qty)} units" if order_qty > 0 else "Qty: estimate in Workbench"
    cost_str = f" · ~KES {cost_estimate_kes:,.0f}" if cost_estimate_kes else ""
    gap_str  = f" · {stockout_gap_days}d gap during delivery" if stockout_gap_days > 0 else ""
    ai_badge = '<span class="decision-ai-badge">✦ AI</span>' if is_ai else ""

    # First sentence only for the compact view
    dot_idx = narrative.find(".")
    if 0 < dot_idx < len(narrative) - 1:
        first_sentence = narrative[: dot_idx + 1].strip()
        remainder      = narrative[dot_idx + 1 :].strip()
    else:
        first_sentence = narrative
        remainder      = ""

    st.markdown(
        f"""
        <div class="decision-card" style="border-left-color:{color}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:3px">
            <div class="decision-drug">{canonical_name}</div>
            {ai_badge}
          </div>
          <div class="decision-meta">{dos_str} · {qty_str}{cost_str}{gap_str}</div>
          <div class="decision-narrative">{first_sentence}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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


# ── Historical data notice ────────────────────────────────────────────────────

def historical_notice(date_range: str) -> None:
    st.info(
        f"**Historical data** ({date_range}). Live alerts and order actions are disabled. "
        "Use this facility for analysis and benchmarking.",
        icon="📅",
    )
