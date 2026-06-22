"""
ui_template.py — Afya Clinical Analytics · Design system
=========================================================
Single source of truth for colours, typography, layout tokens,
Plotly theme, CSS injection, and UI component helpers.
"""

import json
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _st_components

# ── DESIGN TOKENS — primary palette ──────────────────────────────────────────
PRIMARY    = "#0F6E56"   # teal   — positive / within reference
BLUE       = "#0C447C"   # blue   — informational
AMBER      = "#854F0B"   # amber  — warning / monitor
RED        = "#A32D2D"   # red    — critical / below reference
NEUTRAL    = "#888780"   # grey   — inactive / unknown
DEEP_RED   = "#791F1F"   # deep red — negative

# Status variants (lighter fills)
AMBER_LIGHT = "#D97706"
RED_LIGHT   = "#DC2626"

# Surface
BG         = "#F5F6FA"
SURFACE    = "#FFFFFF"
BORDER     = "#E5E7EB"
TEXT       = "#111827"
TEXT_MUTED = "#6B7280"
TEXT_HINT  = "#9CA3AF"

# KPI accent colours — pass as accent_color to kpi_row()
ACCENT_CRITICAL = "#A32D2D"   # rate above threshold / below reference
ACCENT_MONITOR  = "#D97706"   # elevated but not critical
ACCENT_POSITIVE = "#0F6E56"   # within or above reference
ACCENT_INFO     = "#0C447C"   # volume counts / no threshold
ACCENT_NEUTRAL  = "#E5E7EB"   # no clinical signal

# Typography
FONT_FAMILY     = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
FONT_PAGE_TITLE = 22
FONT_SECTION    = 11
FONT_KPI_LABEL  = 10
FONT_KPI_VALUE  = 24
FONT_BODY       = 14
FONT_CHART      = 12
FONT_CAPTION    = 11

# ── BACKWARD-COMPAT ALIASES (used by existing views.py code) ─────────────────
AFYA_BLUE = "#0072CE"
TEAL      = "#0BB99F"
COOL_BLUE = "#003467"
ORANGE    = "#f5a623"
CORAL     = "#e05c5c"
PURPLE    = "#7b5ea7"
GRAY      = "#adb5bd"
MUTED     = "#6B8CAE"
BG_LIGHT  = "#F4F8FC"
SEQ       = [TEAL, AFYA_BLUE, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY]
GREEN     = "#38A169"
AMBER     = "#D97706"
RED       = "#C53030"
CA_BLUE   = "#185FA5"
CA_GREEN  = "#0F6E56"
CA_RED    = "#E24B4A"
CA_AMBER  = "#BA7517"
CA_PURPLE = "#534AB7"
CA_PINK   = "#D4537E"
CA_MUTED  = "#888780"

# ── PLOTLY TEMPLATE ───────────────────────────────────────────────────────────
pio.templates["afya"] = pio.templates["plotly_white"]
_t = pio.templates["afya"].layout
_t.font        = dict(family=FONT_FAMILY, color=TEXT_MUTED, size=FONT_CHART)
_t.legend.font = dict(family=FONT_FAMILY, color=TEXT_MUTED, size=FONT_CHART)
_t.xaxis.tickfont   = dict(color=TEXT_HINT, size=FONT_CHART, family=FONT_FAMILY)
_t.xaxis.title.font = dict(color=TEXT_MUTED, size=FONT_CHART, family=FONT_FAMILY)
_t.yaxis.tickfont   = dict(color=TEXT_HINT, size=FONT_CHART, family=FONT_FAMILY)
_t.yaxis.title.font = dict(color=TEXT_MUTED, size=FONT_CHART, family=FONT_FAMILY)
_t.xaxis.gridcolor  = "rgba(0,0,0,0.05)"
_t.yaxis.gridcolor  = "rgba(0,0,0,0.05)"
_t.paper_bgcolor    = "rgba(0,0,0,0)"
_t.plot_bgcolor     = "rgba(0,0,0,0)"
pio.templates.default = "afya"

CHART_FONT = dict(family=FONT_FAMILY, size=FONT_CHART, color=TEXT_MUTED)

CHART_LAYOUT = dict(
    font          = CHART_FONT,
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    autosize      = True,
    margin        = dict(t=8, b=48, l=0, r=16),
    hoverlabel    = dict(
        font_size=FONT_CHART, font_family=FONT_FAMILY,
        bgcolor=SURFACE, bordercolor=BORDER,
    ),
    legend = dict(
        font=dict(family=FONT_FAMILY, size=11, color=TEXT_MUTED),
        bgcolor="rgba(0,0,0,0)",
        orientation="h",
        y=-0.18,
        x=0.5,
        xanchor="center",
    ),
    colorway=SEQ,
)

# Pass to every st.plotly_chart call: width follows container, chart resizes on drag
PC_CFG = {"responsive": True, "displayModeBar": False, "useResizeHandler": True}

AXIS_STYLE = dict(
    gridcolor     = "rgba(0,0,0,0.05)",
    zerolinecolor = "rgba(0,0,0,0.1)",
    tickfont      = dict(family=FONT_FAMILY, size=FONT_CHART, color=TEXT_HINT),
    title_font    = dict(family=FONT_FAMILY, size=FONT_CHART, color=TEXT_MUTED),
    showline      = False,
    linecolor     = BORDER,
)

# Legacy alias
AXIS = dict(
    showgrid   = True,
    gridcolor  = "rgba(0,0,0,0.05)",
    zeroline   = False,
    color      = TEXT_MUTED,
    tickfont   = dict(color=TEXT_HINT, size=FONT_CHART, family=FONT_FAMILY),
    title_font = dict(color=TEXT_MUTED, size=FONT_CHART, family=FONT_FAMILY),
    title_standoff = 8,
)


def _ax(**overrides):
    return {**AXIS, **overrides}


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset & base ─────────────────────────────────────────────────────────── */
* { box-sizing: border-box; }
html, body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stMarkdown, .stMetric, .stDataFrame, .stSelectbox, .stRadio, .stCaption,
.element-container, .block-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
/* ── Restore Material Symbols Rounded font on ALL icon elements ─────────────
   Must appear after the html/body rule so !important specificity wins.
   Targets both the generic icon element (stIconMaterial) and expander
   variants (stExpanderIcon) which share the same inner span structure. ──── */
[data-testid="stIconMaterial"],
[data-testid="stExpanderIcon"],
[data-testid="stExpanderIconCheck"],
[data-testid="stExpanderIconError"],
[data-testid="stExpanderIconSpinner"] {
    font-family: 'Material Symbols Rounded' !important;
    font-style: normal !important;
    font-weight: 400 !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}
.main .block-container { padding-top: 0.75rem !important; }
section[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"] { display: none !important; }
[data-testid="stMetricValue"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #111827 !important;
}

/* ── App background ───────────────────────────────────────────────────────── */
.stApp { background: #F5F6FA !important; }
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}

/* ── Page header ──────────────────────────────────────────────────────────── */
.page-header {
    padding-bottom: 14px;
    margin-bottom: 20px;
    border-bottom: 1px solid #E5E7EB;
}
.page-title {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #111827 !important;
    margin: 0 0 4px !important;
    line-height: 1.2 !important;
    font-family: Inter, -apple-system, sans-serif !important;
}
.page-subtitle {
    font-size: 12px !important;
    color: #9CA3AF !important;
    margin: 0 !important;
    line-height: 1.5 !important;
    font-family: Inter, -apple-system, sans-serif !important;
}

/* ── Section header ───────────────────────────────────────────────────────── */
.section-header, .sh {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #9CA3AF !important;
    margin: 22px 0 10px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #E5E7EB !important;
    font-family: Inter, -apple-system, sans-serif !important;
}

/* ── KPI tiles ────────────────────────────────────────────────────────────── */
.kpi-tile {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-top: 3px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px 14px 12px;
    font-family: Inter, -apple-system, sans-serif;
}
.kpi-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9CA3AF;
    margin-bottom: 6px;
    line-height: 1.4;
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #111827;
    line-height: 1.1;
    word-break: break-word;
}
.kpi-delta { font-size: 11px; font-weight: 600; margin-top: 4px; }

/* ── legacy kpi-card ──────────────────────────────────────────────────────── */
.kpi-card {
    background: #fff;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 14px 14px 10px;
}

/* ── Insight bars ─────────────────────────────────────────────────────────── */
.insight-bar {
    padding: 10px 14px;
    font-size: 13px;
    line-height: 1.65;
    margin: 10px 0;
    border-radius: 0 4px 4px 0;
    font-family: Inter, -apple-system, sans-serif;
    color: #374151;
}
.insight-bar.teal  { border-left: 3px solid #0F6E56; background: #F0FAF6; }
.insight-bar.blue  { border-left: 3px solid #0C447C; background: #EEF3FA; }
.insight-bar.amber { border-left: 3px solid #854F0B; background: #FFFBEB; }
.insight-bar.red   { border-left: 3px solid #A32D2D; background: #FEF2F2; }
.insight-bar ul { margin: 4px 0 0 0 !important; padding-left: 16px !important; list-style: disc !important; }
.insight-bar ul li { font-size: 13px !important; line-height: 1.6 !important; color: #374151 !important; margin-bottom: 3px !important; font-family: Inter, -apple-system, sans-serif !important; }
.insight-bar ul li:last-child { margin-bottom: 0 !important; }
.insight-bar ul li strong { font-weight: 600 !important; }

/* legacy variant classes */
.insight-blue   { background: #EEF3FA; border-left: 3px solid #0C447C; border-radius: 0 4px 4px 0; padding: 10px 13px; margin-bottom: 10px; }
.insight-teal   { background: #F0FAF6; border-left: 3px solid #0F6E56; border-radius: 0 4px 4px 0; padding: 10px 13px; margin-bottom: 10px; }
.insight-amber  { background: #FFFBEB; border-left: 3px solid #854F0B; border-radius: 0 4px 4px 0; padding: 10px 13px; margin-bottom: 10px; }
.insight-red    { background: #FEF2F2; border-left: 3px solid #A32D2D; border-radius: 0 4px 4px 0; padding: 10px 13px; margin-bottom: 10px; }
.insight-purple { background: #F5F0FF; border-left: 3px solid #7B5EA7; border-radius: 0 4px 4px 0; padding: 10px 13px; margin-bottom: 10px; }
.insight-lbl  { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 4px; }
.insight-txt  { font-size: 13px; color: #374151; line-height: 1.65; }

/* ── Anomaly banner ───────────────────────────────────────────────────────── */
.anomaly-banner {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 3px solid #D97706;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0 16px;
    font-family: Inter, -apple-system, sans-serif;
}
.anomaly-title { font-weight: 700; font-size: 12px; color: #92400E; margin-bottom: 3px; }
.anomaly-body  { font-size: 13px; color: #78350F; line-height: 1.55; }

/* ── afya-card chart container ────────────────────────────────────────────── */
.afya-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.afya-card-accent {
    background: #FFFFFF;
    border-left: 4px solid #0F6E56;
    border-top: 1px solid #E5E7EB;
    border-right: 1px solid #E5E7EB;
    border-bottom: 1px solid #E5E7EB;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.chart-title { font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 3px; font-family: Inter, -apple-system, sans-serif; }
.chart-sub   { font-size: 11px; color: #9CA3AF; margin-bottom: 10px; line-height: 1.4; font-family: Inter, -apple-system, sans-serif; }

/* ── Stat strip (Briefing headline row) ───────────────────────────────────── */
.stat-strip { display: flex; background: #fff; border: 1px solid #E5E7EB; border-radius: 10px; overflow: hidden; margin-bottom: 16px; }
.stat-item  { flex: 1; padding: 12px 16px 10px; border-right: 1px solid #E5E7EB; }
.stat-item:last-child { border-right: none; }
.stat-label { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: #9CA3AF; margin-bottom: 4px; font-family: Inter, -apple-system, sans-serif; }
.stat-value { font-size: 22px; font-weight: 700; color: #111827; line-height: 1.1; font-family: Inter, -apple-system, sans-serif; }
.stat-hint  { font-size: 10px; font-weight: 600; margin-top: 2px; font-family: Inter, -apple-system, sans-serif; }

/* ── Action cards ──────────────────────────────────────────────────────────── */
.action-card   { display: flex; align-items: center; gap: 10px; background: #fff; border: 1px solid #E5E7EB; border-left: 3px solid #E5E7EB; border-radius: 0 10px 10px 0; padding: 11px 14px; margin-bottom: 6px; }
.action-dot    { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.action-body   { flex: 1; min-width: 0; }
.action-drug   { font-weight: 600; font-size: 13px; color: #111827; font-family: Inter, -apple-system, sans-serif; }
.action-reason { font-size: 12px; color: #9CA3AF; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: Inter, -apple-system, sans-serif; }
.action-badges { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; flex-shrink: 0; }
.action-badge  { font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 4px; color: #fff; text-transform: uppercase; font-family: Inter, -apple-system, sans-serif; }

/* ── Callout strip (benchmark chart) ─────────────────────────────────────── */
.callout-strip { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
.callout { border-radius: 8px; padding: 11px 14px; font-size: 12px; line-height: 1.6; font-family: Inter, -apple-system, sans-serif; }
.callout.green { background: rgba(15,110,86,0.07); border-left: 3px solid #0F6E56; color: #374151; }
.callout.red   { background: rgba(163,45,45,0.07); border-left: 3px solid #A32D2D; color: #374151; }
.callout strong { display: block; font-size: 12px; font-weight: 600; margin-bottom: 3px; color: #111827; }

/* ── Badges ───────────────────────────────────────────────────────────────── */
.badge { display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; color: #fff; text-transform: uppercase; letter-spacing: 0.04em; font-family: Inter, -apple-system, sans-serif; }
.badge-diag  { background: #EEF3FA; color: #0C447C; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
.badge-pred  { background: #FFFBEB; color: #854F0B; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
.badge-presc { background: #F0FAF6; color: #0F6E56; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
.fbadge-high  { background: #FEE2E2; color: #991B1B; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
.fbadge-med   { background: #FEF3C7; color: #92400E; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
.fbadge-watch { background: #DBEAFE; color: #1E40AF; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
.fbadge-ok    { background: #D1FAE5; color: #065F46; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
/* Within / below reference segment badges */
.bw { background: #E1F5EE; color: #085041; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 6px; font-family: Inter, -apple-system, sans-serif; }
.bb { background: #FCEBEB; color: #791F1F; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; display: inline-block; margin-top: 6px; font-family: Inter, -apple-system, sans-serif; }

/* ── Empty state ──────────────────────────────────────────────────────────── */
.empty-state { text-align: center; padding: 40px 20px; color: #9CA3AF; font-size: 14px; background: #FAFAFA; border: 1.5px dashed #E5E7EB; border-radius: 10px; font-family: Inter, -apple-system, sans-serif; }

/* ── Sidebar nav radio ────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important; align-items: center !important;
    padding: 8px 10px !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #374151 !important; cursor: pointer !important;
    margin-bottom: 2px !important; width: 100% !important;
    font-family: Inter, -apple-system, sans-serif !important;
    transition: background 0.1s !important; background: transparent !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background: #F5F6FA !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #F0FAF6 !important; color: #0F6E56 !important; font-weight: 600 !important;
}
/* Hide the radio circle indicator (first child div) only */
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label input[type="radio"] { display: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: inherit !important; font-size: 13px !important;
    margin: 0 !important; font-family: Inter,-apple-system,sans-serif !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 1px !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none !important; }

/* ── Nav section labels ───────────────────────────────────────────────────── */
.nav-section {
    font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: #D1D5DB; padding: 10px 10px 4px;
    font-family: Inter, -apple-system, sans-serif;
}

/* ── Role toggle (sidebar columns + buttons) ──────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    background: #F5F6FA !important; border-radius: 8px !important;
    padding: 3px !important; gap: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div { padding: 0 !important; }
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    border-radius: 6px !important; border: none !important;
    font-size: 11px !important; font-weight: 500 !important;
    padding: 5px 8px !important; line-height: 1.4 !important;
    width: 100% !important; transition: all 0.1s !important;
    font-family: Inter,-apple-system,sans-serif !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="secondary"] {
    background: transparent !important; color: #6B7280 !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
    background: #fff !important; color: #0F6E56 !important;
    font-weight: 600 !important; box-shadow: 0 1px 3px rgba(0,0,0,.08) !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
[data-baseweb="tab"], [data-baseweb="tab"] p, [data-baseweb="tab"] span,
button[role="tab"], button[role="tab"] p {
    font-family: Inter, -apple-system, sans-serif !important;
    font-weight: 600 !important; font-size: 13px !important;
    color: #6B7280 !important; letter-spacing: 0.01em !important;
}
[aria-selected="true"], [aria-selected="true"] p, [aria-selected="true"] span {
    color: #0F6E56 !important; border-bottom-color: #0F6E56 !important;
}

/* ── Hide Streamlit chrome (status widget + deploy button only) ─────────────
   stToolbar and #MainMenu are intentionally kept — they contain the theme
   toggle, Rerun, and Clear-cache options the user needs. ──────────────────── */
[data-testid="stStatusWidget"],
[data-testid="stDecoration"],
[data-testid="stDeployButton"] { visibility: hidden !important; height: 0 !important; }

/* ── Sidebar filter expanders ────────────────────────────────────────────────
   Match approved design: white card, 1px border, compact arrow on right. ─── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    box-shadow: none !important;
    margin-bottom: 6px !important;
    overflow: hidden !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 13px !important;
    font-family: Inter, -apple-system, sans-serif !important;
    font-weight: 500 !important;
    color: #374151 !important;
    padding: 9px 12px !important;
    background: #FFFFFF !important;
    border-radius: 8px !important;
    min-height: 40px !important;
    align-items: center !important;
}
/* Multiselect chips in sidebar — match green teal primary */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background-color: #0F6E56 !important;
    border-radius: 20px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #FFFFFF !important;
    font-size: 12px !important;
    font-family: Inter, -apple-system, sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] [data-testid="stMultiSelectDeleteButton"] {
    color: #FFFFFF !important;
    opacity: 0.8 !important;
}

/* ── Scrollbar ────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 10px; }

/* ── Plotly chart containers — fill width, resize on container change ─────── */
[data-testid="stPlotlyChart"] {
    width: 100% !important;
}
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plotly {
    width: 100% !important;
}
.svg-container { width: 100% !important; }

/* ── Responsive breakpoints ───────────────────────────────────────────────── */
/* Tablet ≤ 900px */
@media (max-width: 900px) {
    .kpi-value { font-size: 20px !important; }
    .kpi-label { font-size: 9px !important; }
    .page-title { font-size: 18px !important; }
    .afya-card { padding: 10px 12px !important; }
}

/* Mobile ≤ 640px — stack Streamlit columns */
@media (max-width: 640px) {
    [data-testid="column"] {
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Stack KPI tiles */
    .kpi-tile { margin-bottom: 8px !important; }
    /* Collapse callout grid to single column */
    .callout-strip { grid-template-columns: 1fr !important; }
    /* Reduce chart margins */
    [data-testid="stPlotlyChart"] { margin: 0 !important; }
    .afya-card { padding: 8px 10px !important; border-radius: 8px !important; }
    .page-title { font-size: 16px !important; }
    .page-subtitle { font-size: 10px !important; }
}

/* ── Section-lbl legacy ───────────────────────────────────────────────────── */
.section-lbl {
    font-family: Inter, -apple-system, sans-serif !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
    color: #9CA3AF !important; padding-bottom: 6px !important;
    border-bottom: 1px solid #E5E7EB !important; margin-bottom: 12px !important;
}
.freshness { background: #F0FAF6; border: 1px solid #A7F3D0; border-radius: 6px; padding: 8px 10px; font-size: 12px; color: #065F46; margin-bottom: 12px; }
"""


def inject_global_css():
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ── PAGE HEADER ───────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "") -> None:
    """Tab-level title. subtitle is optional — only used on Today's Briefing."""
    sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-title">{title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── SECTION HEADER ────────────────────────────────────────────────────────────
def section_header(title: str) -> None:
    """Uppercase muted section label with bottom border."""
    st.markdown(
        f'<div class="section-header">{title}</div>',
        unsafe_allow_html=True,
    )


# ── KPI CARD (single, legacy) ─────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str = "", delta: str = "",
             delta_color: str = TEXT_HINT, color: str = TEXT):
    delta_html = (
        f'<div style="font-size:11px;color:{delta_color};margin-top:3px;font-weight:600">{delta}</div>'
        if delta else ""
    )
    sub_html = (
        f'<div style="font-size:11px;color:{TEXT_HINT};margin-top:2px">{sub}</div>'
        if sub else ""
    )
    st.markdown(
        f'<div class="kpi-tile" style="border-top-color:{ACCENT_NEUTRAL};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'{delta_html}{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── KPI ROW (border-top accent tiles) ─────────────────────────────────────────
def kpi_row(cards: list) -> None:
    """Horizontal row of KPI tiles with coloured border-top accent.

    Each card dict:
      label        str  — displayed uppercase automatically
      value        str  — formatted value string
      delta        str  — sub-line / context text  (optional)
      delta_good   bool — True=green, False=red, absent=muted  (optional)
      accent_color str  — border-top hex (use ACCENT_* constants; default ACCENT_NEUTRAL)
    """
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        accent = c.get("accent_color", ACCENT_NEUTRAL)
        delta  = c.get("delta", "")
        if "delta_good" in c:
            delta_color = ACCENT_POSITIVE if c["delta_good"] else ACCENT_CRITICAL
        else:
            delta_color = TEXT_HINT
        delta_html = (
            f'<div class="kpi-delta" style="color:{delta_color}">{delta}</div>'
            if delta else ""
        )
        val_color = accent if accent not in (ACCENT_NEUTRAL, BORDER) else TEXT
        col.markdown(
            f'<div class="kpi-tile" style="border-top-color:{accent};">'
            f'<div class="kpi-label">{c["label"]}</div>'
            f'<div class="kpi-value" style="color:{val_color}">{c["value"]}</div>'
            f'{delta_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── CHART CARD ────────────────────────────────────────────────────────────────
def chart_card(title: str, subtitle: str = "") -> None:
    """Opens an afya-card with title and subtitle above the chart.
    Must be followed by st.plotly_chart() then chart_card_close()."""
    sub_html = f'<div class="chart-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="afya-card">'
        f'<div class="chart-title">{title}</div>'
        f'{sub_html}',
        unsafe_allow_html=True,
    )


def chart_card_close() -> None:
    """Closes the afya-card div opened by chart_card()."""
    st.markdown("</div>", unsafe_allow_html=True)


# ── INSIGHT BAR ───────────────────────────────────────────────────────────────
def insight_bar(content, variant: str = "blue") -> None:
    """Renders a coloured insight bar after a chart or section.

    content: str (single paragraph) OR list[str] (bullet points, max 3)
    variant: teal | blue | amber | red
    """
    icons = {"teal": "✦", "blue": "ℹ", "amber": "⚠", "red": "⚡"}
    icon = icons.get(variant, "ℹ")
    if isinstance(content, list):
        items_html = "".join(f"<li>{item}</li>" for item in content)
        body_html = f"<ul>{items_html}</ul>"
    else:
        body_html = content
    st.markdown(
        f'<div class="insight-bar {variant}">'
        f'<span style="font-weight:600;">{icon}</span>&nbsp;&nbsp;'
        f'{body_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── INSIGHT CARD (legacy wrapper) ─────────────────────────────────────────────
def insight_card(text: str, label: str = "Key insight", variant: str = "blue") -> None:
    """Legacy helper — wraps insight_bar with a label prefix."""
    label_color = {
        "blue": BLUE, "teal": PRIMARY, "amber": AMBER, "red": RED,
    }.get(variant, BLUE)
    st.markdown(
        f'<div class="insight-{variant}">'
        f'<div class="insight-lbl" style="color:{label_color}">{label}</div>'
        f'<div class="insight-txt">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── ANOMALY BANNER ────────────────────────────────────────────────────────────
def stat_strip(items: list) -> None:
    """Single-row stat strip for Today's Briefing headline section.

    Each item dict:
      label        str  — uppercase label
      value        str  — formatted value
      hint         str  — sub-line text  (optional)
      hint_good    bool — True=green, False=red, absent=muted  (optional)
      accent_color str  — colour for value and hint (optional)
    """
    cells = ""
    for it in items:
        accent = it.get("accent_color", TEXT)
        hint   = it.get("hint", "")
        if "hint_good" in it:
            hint_color = ACCENT_POSITIVE if it["hint_good"] else ACCENT_CRITICAL
        else:
            hint_color = TEXT_MUTED
        hint_html = (
            f'<div class="stat-hint" style="color:{hint_color}">{hint}</div>'
            if hint else ""
        )
        cells += (
            f'<div class="stat-item">'
            f'<div class="stat-label">{it["label"]}</div>'
            f'<div class="stat-value" style="color:{accent}">{it["value"]}</div>'
            f'{hint_html}'
            f'</div>'
        )
    st.markdown(f'<div class="stat-strip">{cells}</div>', unsafe_allow_html=True)


def anomaly_banner(metric: str, message: str,
                   color: str = AMBER_LIGHT, bg: str = "#FFFBEB") -> None:
    """Warning strip — only call when threshold is actually crossed."""
    st.markdown(
        f'<div class="anomaly-banner" style="border-left-color:{color};background:{bg};">'
        f'<div class="anomaly-title">⚠ &nbsp;{metric}</div>'
        f'<div class="anomaly-body">{message}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── ACTION CARDS ──────────────────────────────────────────────────────────────
def action_cards(items: list) -> None:
    """Row of coloured action cards for recommendations sections.

    Item dict keys:
      action           — 'ORDER NOW' | 'ORDER THIS WEEK' | 'MONITOR'
      canonical_name   — short action title
      reason           — detail / reference
      clinical_priority — 'CRITICAL' | 'HIGH' | 'STANDARD'
    """
    _STYLES = {
        "ORDER NOW":       (RED,         "#FEF2F2"),
        "ORDER THIS WEEK": (AMBER_LIGHT, "#FFFBEB"),
        "MONITOR":         (BLUE,        "#EEF3FA"),
    }
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        action = item.get("action", "MONITOR")
        border, bg = _STYLES.get(action, (NEUTRAL, "#F9FAFB"))
        col.markdown(
            f'<div style="background:{bg};border-left:4px solid {border};'
            f'border-radius:0 8px 8px 0;padding:14px 16px;height:100%;">'
            f'<div style="font-size:10px;font-weight:700;color:{border};'
            f'text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px;">{action}</div>'
            f'<div style="font-size:13px;font-weight:600;color:{TEXT};margin-bottom:6px;">'
            f'{item.get("canonical_name","")}</div>'
            f'<div style="font-size:12px;color:{TEXT_MUTED};line-height:1.6;">'
            f'{item.get("reason","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── FLAG BADGE ────────────────────────────────────────────────────────────────
def flag_badge(level: str, text: str) -> str:
    cls = {"high": "fbadge-high", "med": "fbadge-med",
           "watch": "fbadge-watch", "ok": "fbadge-ok"}.get(level, "fbadge-watch")
    return f'<span class="{cls}">{text}</span>'


# ── DATA FRESHNESS BAR ────────────────────────────────────────────────────────
def freshness_bar(date_str: str, schema: str = ""):
    schema_html = f' &nbsp;·&nbsp; <code style="font-size:10px">{schema}</code>' if schema else ""
    st.markdown(
        f'<div class="freshness">Data as of <strong>{date_str}</strong>{schema_html}</div>',
        unsafe_allow_html=True,
    )


# ── NUMBER FORMATTERS ─────────────────────────────────────────────────────────
def fmt_num(v, suffix="") -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        if abs(f) >= 1_000_000:
            return f"{f/1_000_000:.1f}M{suffix}"
        if abs(f) >= 1_000:
            return f"{f/1_000:.1f}K{suffix}"
        return f"{f:,.0f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def fmt_pct(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}%"
    except (TypeError, ValueError):
        return str(v)


def fmt_delta(v, positive_good: bool = True) -> tuple:
    if v is None:
        return "—", TEXT_HINT
    try:
        f = float(v)
        sign  = "+" if f > 0 else ""
        color = (ACCENT_POSITIVE if (f > 0) == positive_good else ACCENT_CRITICAL) if f != 0 else TEXT_HINT
        return f"{sign}{f:.1f}%", color
    except (TypeError, ValueError):
        return str(v), TEXT_HINT


# ── STANDARD CHART HELPERS ────────────────────────────────────────────────────
def horizontal_bar(df, label_col: str, value_col: str,
                   color: str = PRIMARY, height: int = 280,
                   title: str = "", xaxis_suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(
        y=df[label_col], x=df[value_col], orientation="h",
        marker_color=color,
        text=df[value_col].apply(lambda v: f"{v:.1f}{xaxis_suffix}"),
        textposition="outside", textfont=dict(size=10, color=TEXT_MUTED),
    ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        xaxis={**_ax(), "showgrid": False},
        yaxis={**_ax(), "showgrid": False, "autorange": "reversed"},
        showlegend=False,
    )
    return fig


def line_chart(df, x_col: str, y_cols: list, labels: list = None,
               colors: list = None, height: int = 280,
               title: str = "", yaxis_suffix: str = "") -> go.Figure:
    colors = colors or SEQ
    labels = labels or y_cols
    fig = go.Figure()
    for i, (col, lbl) in enumerate(zip(y_cols, labels)):
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col], name=lbl,
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        xaxis=_ax(), yaxis={**_ax(), "ticksuffix": yaxis_suffix},
    )
    return fig


def stacked_bar(df, x_col: str, y_cols: list, labels: list = None,
                colors: list = None, height: int = 280, title: str = "") -> go.Figure:
    colors = colors or SEQ
    labels = labels or y_cols
    fig = go.Figure()
    for i, (col, lbl) in enumerate(zip(y_cols, labels)):
        fig.add_trace(go.Bar(
            x=df[x_col], y=df[col], name=lbl,
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        barmode="stack", xaxis=_ax(), yaxis=_ax(),
    )
    return fig


def donut_chart(labels: list, values: list, colors: list = None,
                height: int = 220, title: str = "") -> go.Figure:
    colors = colors or SEQ
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.55,
        marker=dict(colors=colors), textinfo="percent",
        textfont=dict(size=10, family=FONT_FAMILY),
    ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title,
           "legend": dict(orientation="v", x=1.05, y=0.5,
                          font=dict(size=10, family=FONT_FAMILY))},
        showlegend=True,
    )
    return fig


def heatmap_chart(z: list, x_labels: list, y_labels: list,
                  height: int = 220, title: str = "",
                  colorscale: str = "Blues") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=y_labels, colorscale=colorscale,
        showscale=False, xgap=2, ygap=2,
    ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        xaxis={**_ax(), "showgrid": False},
        yaxis={**_ax(), "showgrid": False},
    )
    return fig


# ── BENCHMARK CHART (OPD → IPD Section B) ────────────────────────────────────
def build_benchmark_chart(df_bench) -> go.Figure:
    """Diverging gap bar chart. df_bench columns (lowercase):
       diagnosis_label | actual_rate_pct | ref_lower | ref_upper
    Sorted descending by gap before calling."""
    import pandas as pd
    df = df_bench.copy()
    df.columns = [c.lower() for c in df.columns]
    ref_col = "ref_lower" if "ref_lower" in df.columns else (
        "ref_floor" if "ref_floor" in df.columns else df.columns[2]
    )
    lbl_col = "diagnosis_label" if "diagnosis_label" in df.columns else (
        "segment" if "segment" in df.columns else df.columns[0]
    )
    df["gap"] = (df["actual_rate_pct"] - df[ref_col]).round(1)
    df = df.sort_values("gap", ascending=False).reset_index(drop=True)

    GREEN_FILL = "rgba(15,110,86,0.82)"
    RED_FILL   = "rgba(163,45,45,0.82)"
    GREEN_ZONE = "rgba(15,110,86,0.04)"
    RED_ZONE   = "rgba(163,45,45,0.04)"
    AMBER_LINE = "#D97706"

    bar_colors      = [GREEN_FILL if g >= 0 else RED_FILL for g in df["gap"]]
    text_colors     = [PRIMARY    if g >= 0 else RED       for g in df["gap"]]
    value_labels    = [f"{g:+.1f} pp" for g in df["gap"]]

    _x_min = min(df["gap"].min() - 8, -10)
    _x_max = max(df["gap"].max() + 12, 25)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df[lbl_col], x=df["gap"], orientation="h",
        marker=dict(color=bar_colors, cornerradius=3),
        width=0.62, showlegend=False,
        text=value_labels,
        textposition="outside",
        textfont=dict(size=11, family=FONT_FAMILY, color=text_colors),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Gap: %{x:+.1f} pct pts<extra></extra>",
    ))

    # Zone backgrounds and separator line
    first_below = next((i for i, g in enumerate(df["gap"]) if g < 0), None)
    if first_below is not None and first_below > 0:
        sep_y = first_below - 0.5
        fig.add_shape(type="rect", xref="x", yref="y",
                      x0=_x_min, x1=_x_max, y0=sep_y, y1=len(df) - 0.5,
                      fillcolor=GREEN_ZONE, line_width=0, layer="below")
        fig.add_shape(type="rect", xref="x", yref="y",
                      x0=_x_min, x1=_x_max, y0=-0.5, y1=sep_y,
                      fillcolor=RED_ZONE, line_width=0, layer="below")
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=_x_min, x1=_x_max, y0=sep_y, y1=sep_y,
                      line=dict(color="rgba(0,0,0,0.12)", width=1, dash="dot"))
        # Labels at right edge, inside zones — no overlap with bars
        fig.add_annotation(x=_x_max * 0.92, y=sep_y + 0.4,
                           text="Below reference floor", showarrow=False,
                           font=dict(size=9, family=FONT_FAMILY, color=RED),
                           xanchor="right", bgcolor="rgba(255,255,255,0.7)")
        fig.add_annotation(x=_x_max * 0.92, y=sep_y - 0.4,
                           text="Above reference floor", showarrow=False,
                           font=dict(size=9, family=FONT_FAMILY, color=PRIMARY),
                           xanchor="right", bgcolor="rgba(255,255,255,0.7)")

    # Reference floor marker at x=0
    fig.add_shape(type="line", xref="x", yref="paper",
                  x0=0, x1=0, y0=0, y1=1,
                  line=dict(color=AMBER_LINE, width=1.5, dash="dot"))
    fig.add_annotation(x=0, y=1.01, xref="x", yref="paper",
                       text="Reference floor", showarrow=False,
                       font=dict(size=10, family=FONT_FAMILY, color=AMBER_LINE),
                       xanchor="center", yanchor="bottom")

    fig.update_layout(
        height=max(360, min(len(df) * 26 + 60, 680)),
        margin=dict(l=190, r=80, t=30, b=50),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family=FONT_FAMILY),
        xaxis={**AXIS_STYLE, "range": [_x_min, _x_max], "ticksuffix": " pp",
               "zeroline": False,
               "title": dict(text="Gap from reference floor (pct pts)",
                             font=dict(size=FONT_CHART, family=FONT_FAMILY, color=TEXT_MUTED))},
        yaxis={**AXIS_STYLE, "showgrid": False,
               "categoryorder": "array",
               "categoryarray": df[lbl_col].tolist(),
               "tickfont": dict(size=11)},
        bargap=0.35,
    )
    return fig


def render_benchmark_callouts(df_bench) -> None:
    """Two callout cards (above / below reference) below the benchmark chart."""
    df = df_bench.copy()
    df.columns = [c.lower() for c in df.columns]
    ref_col = "ref_lower" if "ref_lower" in df.columns else (
        "ref_floor" if "ref_floor" in df.columns else df.columns[2]
    )
    lbl_col = "diagnosis_label" if "diagnosis_label" in df.columns else (
        "segment" if "segment" in df.columns else df.columns[0]
    )
    df["gap"] = (df["actual_rate_pct"] - df[ref_col]).round(1)
    above = df[df["gap"] >= 0].sort_values("gap", ascending=False)
    below = df[df["gap"] < 0].sort_values("gap", ascending=True)
    best  = above.iloc[0] if len(above) > 0 else None
    worst = below.iloc[0] if len(below) > 0 else None
    c1, c2 = st.columns(2)
    with c1:
        if best is not None:
            st.markdown(
                f'<div class="callout green">'
                f'<strong>{len(above)} condition{"s" if len(above)!=1 else ""} above reference</strong>'
                f'Best: {best[lbl_col]} ({best["gap"]:+.1f} pp). '
                f'Maintain current admission protocols for these conditions.'
                f'</div>',
                unsafe_allow_html=True,
            )
    with c2:
        if worst is not None:
            urgent = below.head(3)[lbl_col].tolist()
            st.markdown(
                f'<div class="callout red">'
                f'<strong>{len(below)} condition{"s" if len(below)!=1 else ""} below reference</strong>'
                f'Most urgent: {", ".join(urgent)}. '
                f'Review OPD assessment and admission criteria. See Section F for actions.'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── SORTABLE TABLE ────────────────────────────────────────────────────────────
def render_sortable_table(
    df,
    height: int = 400,
    highlight_rules: list = None,
    badge_columns: dict = None,
    key: str = "table",
):
    """Custom HTML table with sticky header, scroll, click-to-sort, and badges."""
    rows_json      = json.dumps(df.to_dict(orient="records"), default=str)
    cols_json      = json.dumps(list(df.columns))
    highlight_json = json.dumps(highlight_rules or [])
    badge_json     = json.dumps(badge_columns   or {})

    html = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
.tbl-wrap {{
    overflow-y: auto; overflow-x: auto;
    height: {height}px;
    border: 0.5px solid rgba(128,128,128,0.18);
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
}}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
thead th {{
    position: sticky; top: 0; z-index: 2;
    background: #F9FAFB; color: #6B7280;
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; padding: 8px 12px;
    border-bottom: 1px solid #E5E7EB; cursor: pointer;
    white-space: nowrap; user-select: none;
}}
thead th:hover {{ background: #F3F4F6; color: #374151; }}
tbody tr {{ border-bottom: 1px solid #F3F4F6; }}
tbody tr:last-child {{ border-bottom: none; }}
tbody tr:hover {{ background: #FAFAFA; }}
tbody td {{
    padding: 8px 12px; color: #111827;
    font-size: 12px; white-space: nowrap;
}}
.row-red    {{ background: rgba(254,242,242,0.6); }}
.row-amber  {{ background: rgba(255,251,235,0.6); }}
.row-green  {{ background: rgba(240,250,246,0.6); }}
.sort-asc::after  {{ content: ' ↑'; opacity: .6; }}
.sort-desc::after {{ content: ' ↓'; opacity: .6; }}
</style>
<div class="tbl-wrap" id="tbl_{key}">
<table id="t_{key}">
<thead id="thead_{key}"></thead>
<tbody id="tbody_{key}"></tbody>
</table>
</div>
<script>
(function(){{
  const rows = {rows_json};
  const cols = {cols_json};
  const hRules = {highlight_json};
  const bRules = {badge_json};
  const tbody = document.getElementById('tbody_{key}');
  const thead = document.getElementById('thead_{key}');
  let sortCol = null, sortAsc = true;

  function badge(col, val) {{
    const rules = bRules[col] || [];
    const n = parseFloat(val);
    for (const r of rules) {{
      if (!isNaN(n) && n >= r.min && n < r.max) {{
        return `<span style="background:${{r.bg}};color:${{r.text}};font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;">${{val}}</span>`;
      }}
    }}
    return val;
  }}

  function rowClass(row) {{
    for (const r of hRules) {{
      try {{ if (eval(`const val=${{JSON.stringify(row[r.column])}};${{r.js_condition}}`)) return r.row_class; }} catch(e){{}}
    }}
    return '';
  }}

  function render(data) {{
    const trh = document.createElement('tr');
    cols.forEach((c,i) => {{
      const th = document.createElement('th');
      th.textContent = c;
      if (sortCol === i) th.className = sortAsc ? 'sort-asc' : 'sort-desc';
      th.onclick = () => {{ sortAsc = sortCol === i ? !sortAsc : true; sortCol = i; render(rows); }};
      trh.appendChild(th);
    }});
    thead.innerHTML = ''; thead.appendChild(trh);

    const sorted = [...data];
    if (sortCol !== null) {{
      sorted.sort((a,b) => {{
        const av = a[cols[sortCol]], bv = b[cols[sortCol]];
        const an = parseFloat(av), bn = parseFloat(bv);
        return sortAsc
          ? (!isNaN(an)&&!isNaN(bn) ? an-bn : String(av).localeCompare(String(bv)))
          : (!isNaN(an)&&!isNaN(bn) ? bn-an : String(bv).localeCompare(String(av)));
      }});
    }}

    tbody.innerHTML = '';
    sorted.forEach(row => {{
      const tr = document.createElement('tr');
      tr.className = rowClass(row);
      cols.forEach(c => {{
        const td = document.createElement('td');
        td.innerHTML = badge(c, row[c] ?? '');
        tr.appendChild(td);
      }});
      tbody.appendChild(tr);
    }});
  }}

  render(rows);
}})();
</script>
"""
    _st_components.html(html, height=height + 30, scrolling=False)
