"""
sph/opd_ipd_module/ui_template.py
===================================
Design system built from Template A (Overview) and Template B (Standard Tab).

TOKEN SYSTEM
------------
Define once per project — swap the hex values below, nothing else changes.

  token-primary    → PRIMARY        informational, headers, KPI numbers
  token-success    → SUCCESS        positive findings, client-name block
  token-danger     → DANGER         critical gaps, negative findings
  token-warning    → WARNING        caution-level findings
  token-neutral    → NEUTRAL        inactive, unknown, no signal
  token-surface-0  → SURFACE_0      page background
  token-surface-1  → SURFACE_1      card / panel background
  token-surface-2  → SURFACE_2      active nav item background
  token-text-primary   → TEXT_PRI   body and header text
  token-text-secondary → TEXT_SEC   supporting text
  token-text-muted     → TEXT_MUT   captions, labels, hints
  token-border     → BORDER         default hairline borders

COMPONENT CATALOGUE
-------------------
  Page structure
    inject_css()               — call once at app startup
    page_header(title, sub)    — tab title + one subtitle line
    section_header(title)      — uppercase muted section label

  Template A (Overview only)
    kpi_row(cards)             — 4–6 KPI tiles, no sparklines
    stage_row(stages)          — mission-stage funnel strip
    issues_table(rows)         — ranked priority list, text-first
    overview_synthesis(text)   — closing synthesis paragraph

  Template B (every standard tab)
    kpi_row(cards)             — same component, sparkline slot added
    chart_container_open(title, note)   — Pattern A/B/C chart wrapper open
    chart_container_close()             — closes the wrapper
    insight_bar(bullets, action, variant) — full-width below-chart bar
    sharp_finding_card(eyebrow, stat, context, sub)  — Pattern D
    tab_synthesis(items)       — three-tile closing synthesis strip

  Shared utilities
    fmt_num(v)       — 1,234 / 12.3K / 1.2M
    fmt_pct(v)       — "12.3%"
    fmt_delta(v, positive_good) — "+3.1%", colour tuple
    PC_CFG           — standard plotly_chart config dict

PLOTLY THEME
------------
All charts should use:
    fig.update_layout(**CHART_LAYOUT)
    fig.update_xaxes(**AXIS_X)
    fig.update_yaxes(**AXIS_Y)

Heights:
    H_SINGLE = 280   full-width chart
    H_PAIRED = 260   both charts in a two-column row (apply to BOTH columns)
"""

import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN VALUES  — change only here, never elsewhere in any file
# ─────────────────────────────────────────────────────────────────────────────

# Core palette — SPH_Dashboard_Color_Standard.md (St. Peter's Orthopedic branding)
PRIMARY  = "#1B8A82"   # token-primary  — teal, brand / informational
SUCCESS  = "#3B6D11"   # token-success  — green, status: good / on target
DANGER   = "#A32D2D"   # token-danger   — red, status: needs attention now
WARNING  = "#854F0B"   # token-warning  — amber, status: monitor / developing
NEUTRAL  = "#8A93A6"   # token-neutral  — cool grey, uncategorized

# Brand secondary + dark neutral (spec section 1)
SECONDARY = "#C13868"   # raspberry — general surgery / OBGYN category color
DARK_NAVY = "#141F3D"   # deep navy — headers, footers, dark text
ACCENT_PINK = "#E91E63" # sparing single-standout callouts only

# Status borders (spec section 3 — fill/text/border triad per status)
DANGER_BORDER  = "#E24B4A"
WARNING_BORDER = "#EF9F27"
SUCCESS_BORDER = "#639922"

# Derived / semantic aliases kept for backward compat with views.py
BLUE     = PRIMARY
RED      = DANGER
AMBER    = WARNING
GREEN    = SUCCESS

# Accent colour constants for kpi_row cards
ACCENT_INFO     = PRIMARY
ACCENT_POSITIVE = SUCCESS
ACCENT_CRITICAL = DANGER
ACCENT_MONITOR  = WARNING
ACCENT_NEUTRAL  = "#D3D6DE"

# CA-prefix chart colour aliases — used by the Clinical Activity tab
# (sph/clinical_activity_module) so no view file hardcodes hex values.
# Category rule (spec §4): Teal = orthopedics-related, Raspberry = general
# surgery/OBGYN-related, Navy = male, Raspberry-light = female.
CA_BLUE   = PRIMARY      # orthopedics category
CA_GREEN  = SUCCESS
CA_RED    = DANGER
CA_AMBER  = WARNING
CA_MUTED  = NEUTRAL
CA_PINK   = SECONDARY    # general surgery / OBGYN category

# Teal family (darkest → lightest) — magnitude-only bars/lines, category ramps
TEAL_1 = "#1B8A82"
TEAL_2 = "#4FADA5"
TEAL_3 = "#8FCFC8"
TEAL_4 = "#E1F5EE"

# Raspberry family (darkest → lightest)
RASP_1 = "#C13868"
RASP_2 = "#D6698C"
RASP_3 = "#EBA3B8"
RASP_4 = "#FBEAF0"

# Grey family (darkest → lightest)
GREY_1 = "#141F3D"
GREY_2 = "#5C6478"
GREY_3 = "#8A93A6"
GREY_4 = "#D3D6DE"

# Surfaces (spec section 2 — light-theme surfaces)
SURFACE_0 = "#F4F6FA"   # token-surface-0  page bg
SURFACE_1 = "#FFFFFF"   # token-surface-1  card / panel
SURFACE_2 = TEAL_4       # token-surface-2  active nav bg
SURFACE   = SURFACE_1    # alias

# Tab title accent — deep navy per spec
TITLE_NAVY = DARK_NAVY

# Text
TEXT_PRI  = "#141F3D"   # token-text-primary
TEXT_SEC  = "#5C6478"   # token-text-secondary
TEXT_MUT  = "#8A93A6"   # token-text-muted
TEXT      = TEXT_PRI    # alias
TEXT_MUTED= TEXT_MUT    # alias
TEXT_HINT = TEXT_MUT    # alias

# Border / gridlines
BORDER    = "#E4E7ED"   # token-border
GRIDLINE  = "#EDEFF3"   # chart axes — recessive, never black

# Insight / callout bars (spec section 4 — Informational / Attention / Monitor / Positive)
_BG = {
    "primary": ("#F4F6FA", DARK_NAVY),
    "success": ("#EAF3DE", SUCCESS),
    "danger":  ("#FCEBEB", DANGER),
    "warning": ("#FAEEDA", WARNING),
    "neutral": ("#F4F6FA", NEUTRAL),
}

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────

_FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
_FONT_SIZE   = 12

pio.templates["afya_template"] = pio.templates["plotly_white"]
_t = pio.templates["afya_template"].layout
_t.font                 = dict(family=_FONT_FAMILY, color=TEXT_MUT, size=_FONT_SIZE)
_t.paper_bgcolor        = "rgba(0,0,0,0)"
_t.plot_bgcolor         = "rgba(0,0,0,0)"
_t.xaxis.gridcolor      = "rgba(0,0,0,0.04)"
_t.yaxis.gridcolor      = "rgba(0,0,0,0.04)"
_t.xaxis.tickfont       = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_MUT)
_t.yaxis.tickfont       = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_MUT)
_t.xaxis.title.font     = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_SEC)
_t.yaxis.title.font     = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_SEC)
pio.templates.default   = "afya_template"

# Standard layout dict — spread into fig.update_layout(**CHART_LAYOUT)
CHART_LAYOUT = dict(
    font          = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_MUT),
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    autosize      = True,
    margin        = dict(t=8, b=52, l=0, r=8),
    hoverlabel    = dict(
        font_size=_FONT_SIZE, font_family=_FONT_FAMILY,
        bgcolor=SURFACE_1, bordercolor=BORDER,
    ),
    legend=dict(
        font=dict(family=_FONT_FAMILY, size=11, color=TEXT_MUT),
        bgcolor="rgba(0,0,0,0)",
        orientation="h",
        y=-0.22,
        x=0.5,
        xanchor="center",
    ),
)

# Axis dicts — spread into fig.update_xaxes(**AXIS_X) etc.
AXIS_X = dict(
    showgrid     = False,
    zeroline     = False,
    tickfont     = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_MUT),
    title_font   = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_SEC),
    linecolor    = BORDER,
    showline     = False,
    title_standoff = 8,
)
AXIS_Y = dict(
    showgrid     = True,
    gridcolor    = "rgba(0,0,0,0.04)",
    zeroline     = False,
    tickfont     = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_MUT),
    title_font   = dict(family=_FONT_FAMILY, size=_FONT_SIZE, color=TEXT_SEC),
    showline     = False,
    title_standoff = 8,
)

# Alias used by the Clinical Activity tab build spec
AXIS_STYLE = AXIS_Y

# Legacy alias used in some existing views
AXIS = {**AXIS_Y, "showgrid": True, "color": TEXT_MUT}

# Standard chart heights (apply the same value to both columns in Pattern B)
H_SINGLE = 280   # full-width chart
H_PAIRED = 260   # two-column paired chart — both columns get this value

# Standard plotly_chart config
PC_CFG = {"responsive": True, "displayModeBar": False, "useResizeHandler": True}


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Reset ─────────────────────────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, .stMarkdown, .stMetric, .stDataFrame, .stSelectbox, .stRadio,
.stCaption, .element-container, .block-container, button, input, select,
textarea, label, p {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

/* ── Preserve Material Icons on expander elements ───────────────────────────── */
[data-testid="stIconMaterial"],
[data-testid="stExpanderIcon"],
[data-testid="stExpanderIconCheck"],
[data-testid="stExpanderIconError"],
[data-testid="stExpanderIconSpinner"] {{
    font-family: 'Material Symbols Rounded' !important;
    font-style: normal !important;
    font-weight: 400 !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}}

/* ── App chrome ─────────────────────────────────────────────────────────────── */
.stApp {{ background: {SURFACE_0} !important; }}
.main .block-container {{ padding-top: 0.75rem !important; }}

/* Light sidebar — SPH_Dashboard_Color_Standard.md surfaces */
[data-testid="stSidebar"] {{
    background: {SURFACE_0} !important;
    border-right: 1px solid {BORDER} !important;
}}

/* All text inside sidebar defaults to secondary text grey */
[data-testid="stSidebar"] * {{
    color: {TEXT_SEC} !important;
}}

/* Streamlit button shell — strip all default button chrome */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    width: 100% !important;
    text-align: left !important;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* The <p> inside each button is the visible nav item */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p {{
    font-size: 13px !important;
    font-weight: 400 !important;
    text-align: left !important;
    margin: 0 !important;
    padding: 9px 10px !important;
    color: {TEXT_PRI} !important;
    border-radius: 7px !important;
    border-left: 2.5px solid transparent !important;
    width: 100% !important;
    transition: background 0.12s ease, color 0.12s ease;
}}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p:hover {{
    background: {TEAL_4} !important;
    color: {PRIMARY} !important;
}}

/* Focus ring — remove it, it looks wrong on dark bg */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:focus:not(:active) {{
    box-shadow: none !important;
    border-color: transparent !important;
    outline: none !important;
}}

/* Sidebar divider line */
[data-testid="stSidebar"] hr {{
    border-color: {BORDER} !important;
}}

/* Sidebar section group labels */
[data-testid="stSidebar"] .at-nav-group {{
    color: {TEXT_MUT} !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 14px 10px 6px !important;
}}

/* Hide the default Streamlit nav generated by multi-page apps */
section[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"] {{ display: none !important; }}

/* ── Metric value override ──────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {{
    font-size: 24px !important;
    font-weight: 700 !important;
    color: {TEXT_PRI} !important;
}}

/* ── Plotly chart containers — fill width ──────────────────────────────────── */
[data-testid="stPlotlyChart"] {{ width: 100% !important; }}
[data-testid="stPlotlyChart"] > div,
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plotly {{ width: 100% !important; }}
.svg-container {{ width: 100% !important; }}

/* ── Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-thumb {{ background: #D3D6DE; border-radius: 10px; }}

/* ── Page header ───────────────────────────────────────────────────────────── */
.at-page-header {{
    padding-bottom: 14px;
    margin-bottom: 20px;
    border-bottom: 1px solid {BORDER};
}}
.at-page-title {{
    font-size: 20px;
    font-weight: 600;
    color: {TEXT_PRI};
    margin: 0 0 4px;
    line-height: 1.2;
}}
.at-page-sub {{
    font-size: 12px;
    color: {TEXT_MUT};
    margin: 0;
    line-height: 1.5;
}}

/* ── Section label ─────────────────────────────────────────────────────────── */
.at-section-lbl {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {TEXT_MUT};
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid {BORDER};
}}

/* ── KPI tile ──────────────────────────────────────────────────────────────── */
/* token-primary border-top = informational
   token-success             = positive
   token-danger              = critical
   token-warning             = caution
   token-neutral / none      = no signal
   Rule: top-border accent only, never full colored background */
.at-kpi {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-top: 3px solid {BORDER};    /* default — overridden per tile */
    border-radius: 10px;
    padding: 14px 16px 12px;
}}
.at-kpi-lbl {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUT};
    margin-bottom: 6px;
    line-height: 1.4;
}}
.at-kpi-val {{
    font-size: 24px;
    font-weight: 700;
    line-height: 1.1;
    word-break: break-word;
}}
.at-kpi-delta {{
    font-size: 11px;
    font-weight: 600;
    margin-top: 4px;
}}
.at-kpi-sparkline {{
    display: flex;
    align-items: flex-end;
    gap: 3px;
    height: 20px;
    margin-top: 8px;
}}
.at-kpi-spark-bar {{
    flex: 1;
    border-radius: 1px 1px 0 0;
    min-width: 4px;
}}

/* ── Stage row (Template A, element 2) ──────────────────────────────────────── */
.at-stage-strip {{
    display: flex;
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 4px;
}}
.at-stage-item {{
    flex: 1;
    padding: 12px 16px 10px;
    border-right: 1px solid {BORDER};
    position: relative;
}}
.at-stage-item:last-child {{ border-right: none; }}
.at-stage-eyebrow {{
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {TEXT_MUT};
    margin-bottom: 4px;
}}
.at-stage-val {{
    font-size: 20px;
    font-weight: 700;
    line-height: 1.1;
}}
.at-stage-lbl {{
    font-size: 11px;
    color: {TEXT_SEC};
    margin-top: 2px;
}}

/* ── Issues table (Template A, element 3) ───────────────────────────────────── */
.at-issues {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
}}
.at-issues-hdr {{
    display: grid;
    border-bottom: 1px solid {BORDER};
    background: {SURFACE_0};
    padding: 8px 14px;
}}
.at-issues-row {{
    display: grid;
    border-bottom: 1px solid {BORDER};
    padding: 9px 14px;
    align-items: center;
}}
.at-issues-row:last-child {{ border-bottom: none; }}
.at-issues-rank {{
    font-size: 12px;
    font-weight: 700;
    color: {TEXT_MUT};
}}
.at-issues-issue {{
    font-size: 12px;
    color: {TEXT_SEC};
    line-height: 1.45;
}}
.at-issues-where {{
    font-size: 11px;
    color: {TEXT_MUT};
}}

/* ── Badge ─────────────────────────────────────────────────────────────────── */
.at-badge {{
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.at-badge-danger  {{ background: rgba(163,45,45,.1);  color: #A32D2D; }}
.at-badge-warning {{ background: rgba(133,79,11,.1);  color: #854F0B; }}
.at-badge-success {{ background: rgba(15,110,86,.1);  color: #3B6D11; }}
.at-badge-neutral {{ background: rgba(136,135,128,.12); color: #5C6478; }}

/* ── Chart card wrapper (Pattern A / B / C) ─────────────────────────────────── */
/* The wrapper provides the white card background and title area.
   The chart itself renders via st.plotly_chart — it fills the container.
   chart_container_open() opens this; chart_container_close() closes it.
   Do NOT put insight bars inside this div — they go after chart_container_close(). */
.at-chart-card {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px 12px;
    margin-bottom: 0;
}}
.at-chart-title {{
    font-size: 12px;
    font-weight: 600;
    color: {TEXT_SEC};
    margin-bottom: 2px;
}}
.at-chart-note {{
    font-size: 11px;
    font-style: italic;
    color: {TEXT_MUT};
    margin-bottom: 10px;
    line-height: 1.4;
}}

/* ── Insight bar (Pattern A / B / C — always full-width below chart row) ─────── */
/* Variants map to token severity:
   primary = informational  (blue)
   success = positive       (teal)
   danger  = critical       (red)
   warning = caution        (amber) */
.at-insight {{
    padding: 10px 14px;
    font-size: 13px;
    line-height: 1.65;
    border-radius: 0 6px 6px 0;
    margin: 8px 0 16px;
}}
.at-insight.primary {{ border-left: 3px solid {DARK_NAVY}; background: {SURFACE_0}; }}
.at-insight.success {{ border-left: 3px solid {SUCCESS_BORDER}; background: #EAF3DE; }}
.at-insight.danger  {{ border-left: 3px solid {DANGER_BORDER};  background: #FCEBEB; }}
.at-insight.warning {{ border-left: 3px solid {WARNING_BORDER}; background: #FAEEDA; }}
.at-insight ul {{
    margin: 4px 0 0 0 !important;
    padding-left: 16px !important;
    list-style: disc !important;
}}
.at-insight ul li {{
    font-size: 13px !important;
    line-height: 1.6 !important;
    color: {TEXT_SEC} !important;
    margin-bottom: 3px !important;
}}
.at-insight ul li:last-child {{ margin-bottom: 0 !important; }}
/* Bold spans were only bold — same muted color as the surrounding text —
   so the key numbers/phrases didn't actually stand out on a skim. Each
   variant's <strong> now uses that variant's accent color instead. */
.at-insight.primary ul li strong {{ color: {DARK_NAVY}; }}
.at-insight.success ul li strong {{ color: {SUCCESS_BORDER}; }}
.at-insight.danger  ul li strong {{ color: {DANGER_BORDER}; }}
.at-insight.warning ul li strong {{ color: {WARNING_BORDER}; }}
.at-insight-action {{
    font-size: 12px;
    font-weight: 600;
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px solid rgba(0,0,0,0.06);
}}
.at-insight.primary .at-insight-action {{ color: {PRIMARY}; }}
.at-insight.success .at-insight-action {{ color: {SUCCESS}; }}
.at-insight.danger  .at-insight-action {{ color: {DANGER};  }}
.at-insight.warning .at-insight-action {{ color: {WARNING}; }}

/* ── Sharp-finding card (Pattern D) ─────────────────────────────────────────── */
/* One per section max. Left border only — no rounded corners on that side. */
.at-sharp-card {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-left: 4px solid {DANGER};
    border-radius: 0 10px 10px 0;
    padding: 16px 18px;
    margin-bottom: 12px;
}}
.at-sharp-eyebrow {{
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {DANGER};
    margin-bottom: 4px;
}}
.at-sharp-stat {{
    font-size: 28px;
    font-weight: 700;
    color: {DANGER};
    line-height: 1.1;
}}
.at-sharp-ctx {{
    font-size: 12px;
    color: {TEXT_SEC};
    margin-top: 4px;
    line-height: 1.5;
}}
.at-sharp-sub {{
    font-size: 11px;
    color: {TEXT_MUT};
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px solid {BORDER};
    line-height: 1.45;
}}

/* ── Tab synthesis (closing three-tile strip — Template B) ──────────────────── */
/* Always the last element on every tab page. Bordered with token-primary. */
.at-synthesis {{
    border: 2px solid {PRIMARY};
    border-radius: 10px;
    padding: 16px 18px;
    background: rgba(12,68,124,.04);
    margin-top: 8px;
}}
.at-synthesis-lead {{
    font-size: 12px;
    font-weight: 600;
    color: {PRIMARY};
    margin-bottom: 12px;
}}
.at-synthesis-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
}}
.at-syn-tile {{
    padding: 10px 12px;
    background: {SURFACE_0};
    border-radius: 6px;
    border-left: 3px solid {NEUTRAL};
}}
.at-syn-tile.danger  {{ border-left-color: {DANGER};  }}
.at-syn-tile.warning {{ border-left-color: {WARNING}; }}
.at-syn-tile.success {{ border-left-color: {SUCCESS}; }}
.at-syn-tile.primary {{ border-left-color: {PRIMARY}; }}
.at-syn-tile-lbl {{
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUT};
    margin-bottom: 4px;
}}
.at-syn-tile-body {{
    font-size: 11px;
    color: {TEXT_SEC};
    line-height: 1.5;
}}

/* ── Overview synthesis (Template A, element 4 — single paragraph) ──────────── */
.at-overview-synthesis {{
    border: 2px solid {PRIMARY};
    border-radius: 10px;
    padding: 16px 18px;
    background: rgba(12,68,124,.04);
}}
.at-overview-synthesis-lead {{
    font-size: 13px;
    font-weight: 600;
    color: {PRIMARY};
    margin-bottom: 6px;
}}
.at-overview-synthesis-body {{
    font-size: 13px;
    color: {TEXT_SEC};
    line-height: 1.65;
}}

/* ── Data reliability traffic-light card ────────────────────────────────────── */
.at-reliability-card {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}}
.at-reliability-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin: 0 auto 6px;
}}
.at-reliability-domain {{
    font-size: 11px;
    font-weight: 600;
    color: {TEXT_SEC};
    margin-bottom: 4px;
}}
.at-reliability-note {{
    font-size: 10px;
    color: {TEXT_MUT};
    line-height: 1.4;
}}

/* ── Recommendation card ─────────────────────────────────────────────────────── */
.at-rec-card {{
    background: {SURFACE_1};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    height: 100%;
}}
.at-rec-num {{
    font-size: 10px;
    font-weight: 700;
    color: {TEXT_MUT};
    margin-bottom: 4px;
}}
.at-rec-title {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT_PRI};
    margin-bottom: 6px;
}}
.at-rec-body {{
    font-size: 12px;
    color: {TEXT_SEC};
    line-height: 1.6;
}}
.at-rec-source {{
    font-size: 10px;
    color: {TEXT_MUT};
    margin-top: 8px;
}}

/* ── Freshness bar ───────────────────────────────────────────────────────────── */
.at-freshness {{
    background: #EAF3DE;
    border: 1px solid #639922;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 12px;
    color: #3B6D11;
    margin-bottom: 12px;
}}

/* ── Clinical Activity: tab header with caveat chip ─────────────────────────── */
.ca-tab-header {{
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 24px; padding-bottom: 16px; border-bottom: 2px solid #E4E7ED;
}}
.ca-tab-title {{ font-size: 20px; font-weight: 700; color: #0D2B5E; letter-spacing: -0.3px; }}
.ca-tab-subtitle {{ font-size: 12px; color: #8A93A6; margin-top: 3px; }}
.ca-caveat-chip {{
    display: inline-flex; align-items: center; gap: 5px; flex-shrink: 0;
    background: #F4F6FA; color: #5C6478; padding: 5px 12px;
    border-radius: 20px; font-size: 11px; font-weight: 500;
    white-space: nowrap; margin-top: 2px;
}}

/* ── Clinical Activity: synthesis (key findings) cards ───────────────────────── */
.ca-synthesis-row {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 16px 0 0;
}}
.ca-synthesis-card {{
    background: #fff; border: 1px solid {BORDER}; border-top: 3px solid {PRIMARY};
    border-radius: 10px; padding: 16px 16px 14px;
}}
.ca-synth-label {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: {TEXT_MUT}; margin-bottom: 6px;
}}
.ca-synth-value {{ font-size: 15px; font-weight: 700; color: {PRIMARY}; margin-bottom: 8px; line-height: 1.3; }}
.ca-synth-body {{ font-size: 12px; color: {TEXT_SEC}; line-height: 1.65; }}

/* ── Clinical Activity: DQ callout ────────────────────────────────────────────── */
.ca-dq-callout {{
    background: #FAEEDA; border: 1.5px solid #EF9F27; border-left: 4px solid {WARNING};
    border-radius: 8px; padding: 12px 16px; margin: 10px 0 16px;
    font-size: 12px; color: {WARNING}; line-height: 1.65;
}}
.ca-dq-callout strong {{ font-weight: 700; }}

/* ── Clinical Activity: explain block ─────────────────────────────────────────── */
.ca-explain {{
    background: {SURFACE_0}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 12px 16px; margin: 8px 0 12px; font-size: 12px;
    color: {TEXT_SEC}; line-height: 1.65;
}}
.ca-explain strong {{ color: {TEXT_PRI}; }}

/* ── Clinical Activity: KPI chip (inline metric in card) ──────────────────────── */
.ca-kpi-chip {{
    background: #EAF3DE; border: 1px solid #639922; border-radius: 8px;
    padding: 10px 14px; margin-top: 8px; display: flex; align-items: center; gap: 10px;
}}
.ca-kpi-chip-val {{ font-size: 22px; font-weight: 700; color: {SUCCESS}; }}
.ca-kpi-chip-label {{ font-size: 12px; color: {TEXT_SEC}; }}

/* ── Clinical Activity: ward label with rate badge inline ─────────────────────── */
.ca-ward-label {{ font-size: 13px; font-weight: 600; color: {TEXT_PRI}; }}
.ca-ward-rate-outlier {{ font-size: 11px; font-weight: 400; color: #E24B4A; margin-left: 6px; }}
.ca-ward-rate-normal  {{ font-size: 11px; font-weight: 400; color: {TEXT_MUT}; margin-left: 6px; }}

/* ── Clinical Activity: recommendation cards ──────────────────────────────────── */
.ca-rec-card {{
    background: #fff; border: 1px solid {BORDER}; border-radius: 10px; padding: 16px;
}}
.ca-rec-card.urgent   {{ border-left: 4px solid #E24B4A; }}
.ca-rec-card.moderate {{ border-left: 4px solid #EF9F27; }}
.ca-rec-card.structural {{ border-left: 4px solid {PRIMARY}; }}
.ca-rec-priority {{ font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }}
.ca-rec-priority.urgent   {{ color: {DANGER}; }}
.ca-rec-priority.moderate {{ color: {WARNING}; }}
.ca-rec-priority.structural {{ color: {PRIMARY}; }}
.ca-rec-title {{ font-size: 13px; font-weight: 700; color: {TEXT_PRI}; margin-bottom: 4px; }}
.ca-rec-stat {{
    font-size: 12px; font-weight: 600; color: {TEXT_SEC};
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid {SURFACE_0};
}}
.ca-rec-action-label {{
    font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; color: {TEXT_MUT}; margin-bottom: 4px;
}}
.ca-rec-action {{ font-size: 12px; color: #5C6478; line-height: 1.65; }}

/* ── Clinical Activity: limitation cards ──────────────────────────────────────── */
.ca-lim-card {{
    background: #FAEEDA; border: 1px solid #EF9F27;
    border-radius: 10px; padding: 14px;
}}
.ca-lim-title {{ font-size: 12px; font-weight: 700; color: #854F0B; margin-bottom: 6px; }}
.ca-lim-detail {{ font-size: 12px; color: #854F0B; line-height: 1.6; margin-bottom: 8px; }}
.ca-lim-fix {{
    font-size: 11px; font-weight: 600; color: #3B6D11;
    background: #EAF3DE; padding: 6px 10px; border-radius: 6px;
}}

/* ── Clinical Activity: section divider label ─────────────────────────────────── */
.ca-divider-label {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: #8A93A6; margin: 16px 0 10px;
    padding-bottom: 6px; border-bottom: 1px solid #E4E7ED;
}}

/* ── Clinical Activity: empty state ───────────────────────────────────────────── */
.ca-empty-state {{
    background: #F4F6FA; border: 1.5px dashed #E4E7ED; border-radius: 8px;
    padding: 32px; text-align: center; color: #8A93A6; font-size: 13px;
}}

/* ── Responsive ─────────────────────────────────────────────────────────────── */
@media (max-width: 900px) {{
    .at-kpi-val {{ font-size: 20px !important; }}
    .at-page-title {{ font-size: 17px !important; }}
    .at-chart-card {{ padding: 10px 12px !important; }}
    .at-synthesis-grid {{ grid-template-columns: 1fr !important; }}
}}
@media (max-width: 640px) {{
    [data-testid="column"] {{
        width: 100% !important;
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }}
    .at-stage-strip {{ flex-direction: column !important; }}
    .at-stage-item {{ border-right: none !important; border-bottom: 1px solid {BORDER} !important; }}
}}
"""


def inject_css() -> None:
    """Call once at the top of every dashboard entry point."""
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
        '&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# Legacy alias used by older views
def inject_global_css() -> None:
    inject_css()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "") -> None:
    """
    Tab-level title + one subtitle line — styling matches the Flow and
    Retention tab's reference header exactly (20px/700/navy title, 2px
    border, 24px bottom margin).

    Template B rule: subtitle names St. Peter's Orthopaedic Hospital and
    briefly states the tab's scope in one sentence — e.g. "St. Peter's
    Orthopaedic Hospital — Patient retention and follow-up scheduling,
    trailing 12 months."
    Template A rule: subtitle is always "St. Peter's Orthopaedic Hospital —
    one-page summary across all tabs, see individual tabs for full detail."
    """
    sub_html = (
        f'<div style="font-size:12px;color:{TEXT_MUT};margin-top:3px;line-height:1.5">'
        f'{subtitle}</div>' if subtitle else ""
    )
    st.markdown(
        f'<div style="padding-bottom:16px;margin-bottom:24px;'
        f'border-bottom:2px solid {BORDER};font-family:Inter,sans-serif">'
        f'<div style="font-size:20px;font-weight:700;color:{TITLE_NAVY};'
        f'letter-spacing:-0.3px;line-height:1.2">{title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    """
    Uppercase muted section label with bottom border.
    Placed above every chart or content block.
    """
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.08em;color:{TEXT_MUT};margin:24px 0 12px;'
        f'padding-bottom:6px;border-bottom:1px solid {BORDER};'
        f'font-family:Inter,sans-serif">{title}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW  (Template A + B)
# ─────────────────────────────────────────────────────────────────────────────

def kpi_row(cards: list) -> None:
    """
    Horizontal row of KPI tiles with coloured border-top accent.

    Template B rule: 3–6 tiles, equal width, equal height, one row.
    Severity accent is the top border only — never a full colored background.

    Each card dict:
      label        str   — displayed uppercase automatically
      value        str   — formatted value string
      delta        str   — sub-line / context text  (optional)
      delta_good   bool  — True=success color, False=danger color  (optional)
      accent_color str   — border-top hex. Use ACCENT_* constants.
                           Default: ACCENT_NEUTRAL (grey — no clinical signal)
      sparkline    list  — optional list of (height_pct, hex_color) tuples
                           for the breakdown mini-bar strip. Only add where
                           the spread across a dimension is real and material.
    """
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        accent = c.get("accent_color", ACCENT_NEUTRAL)
        delta  = c.get("delta", "")
        sparks = c.get("sparkline", [])

        if "delta_good" in c:
            delta_color = SUCCESS if c["delta_good"] else DANGER
        else:
            delta_color = TEXT_MUT

        delta_html = (
            f'<div style="font-size:11px;font-weight:600;margin-top:4px;'
            f'color:{delta_color}">{delta}</div>'
            if delta else ""
        )

        # Sparkline mini-bars — only rendered if the caller provides them
        if sparks:
            bars_html = "".join(
                f'<div style="flex:1;border-radius:1px 1px 0 0;min-width:4px;'
                f'height:{h}%;background:{color}"></div>'
                for h, color in sparks
            )
            spark_html = (
                f'<div style="display:flex;align-items:flex-end;gap:3px;'
                f'height:20px;margin-top:8px">{bars_html}</div>'
            )
        else:
            spark_html = ""

        # Value colour follows accent when accent is a real signal
        val_color = accent if accent not in (ACCENT_NEUTRAL, BORDER, "#E4E7ED") else TEXT_PRI

        col.markdown(
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
            f'border-top:3px solid {accent};border-radius:10px;'
            f'padding:14px 16px 12px;font-family:Inter,sans-serif">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:{TEXT_MUT};margin-bottom:6px;'
            f'line-height:1.4">{c["label"]}</div>'
            f'<div style="font-size:24px;font-weight:700;line-height:1.1;'
            f'word-break:break-word;color:{val_color}">{c["value"]}</div>'
            f'{delta_html}'
            f'{spark_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE A — OVERVIEW-ONLY COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def stage_row(stages: list) -> None:
    """
    Mission-stage funnel strip (Template A, element 2).

    Only render this when the subject genuinely has a stage structure.
    Do not force a funnel where one doesn't exist.

    Each stage dict:
      eyebrow      str   — "Stage 1", "Q1", "Step: Screened" etc.
      value        str   — the headline number or rate
      label        str   — plain-language label for the stage
      accent_color str   — value colour. Use ACCENT_* constants.
    """
    cells = ""
    for stage in stages:
        accent = stage.get("accent_color", TEXT_PRI)
        cells += (
            f'<div style="flex:1;padding:12px 16px 10px;'
            f'border-right:1px solid {BORDER};font-family:Inter,sans-serif">'
            f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.07em;color:{TEXT_MUT};margin-bottom:4px">'
            f'{stage.get("eyebrow","")}</div>'
            f'<div style="font-size:20px;font-weight:700;line-height:1.1;'
            f'color:{accent}">{stage["value"]}</div>'
            f'<div style="font-size:11px;color:{TEXT_SEC};margin-top:2px">'
            f'{stage.get("label","")}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:flex;background:{SURFACE_1};border:1px solid {BORDER};'
        f'border-radius:10px;overflow:hidden;margin-bottom:4px">{cells}</div>',
        unsafe_allow_html=True,
    )


def issues_table(rows: list, col_widths: str = "32px 1fr 160px 90px") -> None:
    """
    Ranked priority list — text-first, not a chart (Template A, element 3).

    Template A rule: this is a scannable list, not a bar chart.
    Deliberately plain — rank, issue text, location, severity badge.

    Each row dict:
      rank         int or str
      issue        str   — plain-language description
      where        str   — location or scope of the finding
      severity     str   — "critical" | "warning" | "success" | "neutral"
      severity_lbl str   — display label for the badge (optional, defaults to severity)
    """
    _BADGE_STYLES = {
        "critical": ("rgba(163,45,45,.1)",   "#A32D2D"),
        "warning":  ("rgba(133,79,11,.1)",   "#854F0B"),
        "success":  ("rgba(15,110,86,.1)",   "#3B6D11"),
        "neutral":  ("rgba(136,135,128,.12)","#5C6478"),
    }

    hdr = (
        f'<div style="display:grid;grid-template-columns:{col_widths};'
        f'border-bottom:1px solid {BORDER};background:{SURFACE_0};padding:8px 14px">'
        f'<span style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.06em;color:{TEXT_MUT}">#</span>'
        f'<span style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.06em;color:{TEXT_MUT}">Issue</span>'
        f'<span style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.06em;color:{TEXT_MUT}">Where</span>'
        f'<span style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.06em;color:{TEXT_MUT}">Severity</span>'
        f'</div>'
    )

    body = ""
    for r in rows:
        sev   = r.get("severity", "neutral")
        label = r.get("severity_lbl", sev.capitalize())
        bg, fg = _BADGE_STYLES.get(sev, _BADGE_STYLES["neutral"])
        body += (
            f'<div style="display:grid;grid-template-columns:{col_widths};'
            f'border-bottom:1px solid {BORDER};padding:9px 14px;align-items:center">'
            f'<span style="font-size:12px;font-weight:700;color:{TEXT_MUT}">'
            f'{r["rank"]}</span>'
            f'<span style="font-size:12px;color:{TEXT_SEC};line-height:1.45">'
            f'{r["issue"]}</span>'
            f'<span style="font-size:11px;color:{TEXT_MUT}">{r.get("where","")}</span>'
            f'<span><span style="display:inline-block;font-size:10px;font-weight:700;'
            f'padding:2px 8px;border-radius:4px;text-transform:uppercase;'
            f'letter-spacing:.04em;background:{bg};color:{fg}">{label}</span></span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
        f'border-radius:10px;overflow:hidden;font-family:Inter,sans-serif">'
        f'{hdr}{body}</div>',
        unsafe_allow_html=True,
    )


def overview_synthesis(lead: str, body: str) -> None:
    """
    Closing synthesis paragraph for the Overview page (Template A, element 4).

    Template A rule: bold lead-in, 3–5 sentences maximum.
    This is the single paragraph a reader should remember if they
    read nothing else on the dashboard.
    """
    st.markdown(
        f'<div style="border:2px solid {PRIMARY};border-radius:10px;'
        f'padding:16px 18px;background:rgba(12,68,124,.04);'
        f'font-family:Inter,sans-serif">'
        f'<div style="font-size:13px;font-weight:600;color:{PRIMARY};'
        f'margin-bottom:6px">{lead}</div>'
        f'<div style="font-size:13px;color:{TEXT_SEC};line-height:1.65">'
        f'{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW PAGE (v2) — tinted KPI row, per-tab section, closing key findings
# ─────────────────────────────────────────────────────────────────────────────

def kpi_row_tinted(cards: list) -> None:
    """
    Horizontal row of KPI tiles with a TINTED BACKGROUND matching severity
    (not just a top border) — denser read for the Overview page.

    Each card dict:
      label        str   — displayed uppercase automatically
      value        str   — formatted value string
      sub          str   — sub-line / context text (optional)
      severity     str   — "danger" | "warning" | "success"  (default "warning")
    """
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        sev = c.get("severity", "warning")
        bg, border = _BG.get(sev, _BG["warning"])
        fg = {"danger": DANGER, "warning": WARNING, "success": SUCCESS,
              "neutral": DARK_NAVY}.get(sev, WARNING)
        sub_html = (
            f'<div style="font-size:11px;font-weight:600;color:{TEXT_MUT};margin-top:4px;'
            f'line-height:1.4">{c.get("sub","")}</div>'
        ) if c.get("sub") else ""
        col.markdown(
            f'<div style="background:{bg};border:1px solid {border};'
            f'border-radius:10px;padding:14px 16px 12px;font-family:Inter,sans-serif;'
            f'height:100%;min-height:132px;box-sizing:border-box">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:{TEXT_MUT};margin-bottom:6px;'
            f'line-height:1.4">{c["label"]}</div>'
            f'<div style="font-size:24px;font-weight:700;line-height:1.1;'
            f'word-break:break-word;color:{fg}">{c["value"]}</div>'
            f'{sub_html}'
            f'</div>',
            unsafe_allow_html=True,
        )


def overview_tab_section(tab_name: str, tab_tag: str, chart_title: str, issues: list,
                          chart_fn=None, chart_note: str = "") -> None:
    """
    One Overview-page tab section: label row (tab name + stat tag), then a
    two-column [chart | 3-issue stack] row (spec section 5).

    tab_name:    str  — e.g. "Case mix"
    tab_tag:     str  — one-line headline stat, right-aligned small tag
    chart_title: str  — chart card title
    chart_fn:    callable(height) — draws the plotly chart via st.plotly_chart
                 (called inside the chart card, left column)
    chart_note:  optional small italic note above the chart canvas
    issues:      list of dicts: {severity: "danger"|"warning"|"success",
                                  tag: str, body: str}  — exactly 3 expected
    """
    st.markdown(
        f'<div style="display:flex;align-items:baseline;justify-content:space-between;'
        f'margin:20px 0 8px;font-family:Inter,sans-serif">'
        f'<span style="font-size:13px;font-weight:600;color:{TEXT_PRI}">{tab_name}</span>'
        f'<span style="font-size:11px;color:{TEXT_MUT};font-weight:500">{tab_tag}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_chart, col_issues = st.columns([1, 1.3])

    with col_chart:
        chart_container_open(chart_title, chart_note)
        if chart_fn is not None:
            chart_fn()
        chart_container_close()

    with col_issues:
        # White card + top-border accent — same pattern as kpi_row(), not a
        # full tinted background (that was a leftover from before the KPI
        # strips were brought in line with the rest of the dashboard).
        items_html = ""
        for it in issues:
            sev = it.get("severity", "warning")
            fg = {"danger": DANGER, "warning": WARNING, "success": SUCCESS}.get(sev, WARNING)
            items_html += (
                f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
                f'border-top:3px solid {fg};border-radius:10px;'
                f'padding:10px 12px;margin-bottom:8px;font-family:Inter,sans-serif">'
                f'<span style="display:block;font-size:10px;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.06em;color:{fg};'
                f'margin-bottom:4px">{it.get("tag","")}</span>'
                f'<span style="font-size:12px;color:{TEXT_SEC};line-height:1.5">'
                f'{it.get("body","")}</span>'
                f'</div>'
            )
        st.markdown(
            f'<div style="display:flex;flex-direction:column;justify-content:center;'
            f'height:100%">{items_html}</div>',
            unsafe_allow_html=True,
        )


def overview_key_findings(tiles: list) -> None:
    """
    Closing three-tile key-findings row (spec section 12) — success-tinted
    (informational, not severity-coded), title + body only, no lead prefix.

    Each tile dict: {num: "01", title: str, body: str}
    """
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;'
        f'margin-top:20px">'
        + "".join(
            f'<div style="background:{_BG["success"][0]};border:1px solid {_BG["success"][1]};'
            f'border-radius:10px;padding:14px 16px;font-family:Inter,sans-serif">'
            f'<div style="font-size:10px;font-weight:700;color:{SUCCESS};'
            f'margin-bottom:4px">{t.get("num","")}</div>'
            f'<div style="font-size:13px;font-weight:600;color:{TEXT_PRI};'
            f'margin-bottom:6px">{t["title"]}</div>'
            f'<div style="font-size:12px;color:{TEXT_SEC};line-height:1.6">'
            f'{t["body"]}</div>'
            f'</div>'
            for t in tiles
        )
        + '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE B — CHART CONTAINER (Pattern A / B / C)
# ─────────────────────────────────────────────────────────────────────────────

def chart_container_open(title: str, note: str = "") -> None:
    """
    Opens the white chart card container (Pattern A / B / C).

    Call this, then st.plotly_chart(), then chart_container_close().
    Do NOT put insight_bar() inside this — it must go after close().

    Pattern B rule: both columns must use identical Plotly height values.
    Set H_PAIRED on both chart calls — the container does not enforce this.
    """
    note_html = (
        f'<div style="font-size:11px;font-style:italic;color:{TEXT_MUT};'
        f'margin-bottom:10px;line-height:1.4">{note}</div>'
    ) if note else ""
    st.markdown(
        f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
        f'border-radius:10px;padding:14px 16px 12px;margin-bottom:0;'
        f'font-family:Inter,sans-serif">'
        f'<div style="font-size:12px;font-weight:600;color:{TEXT_SEC};'
        f'margin-bottom:2px">{title}</div>'
        f'{note_html}',
        unsafe_allow_html=True,
    )


def chart_container_close() -> None:
    """Closes the chart card div opened by chart_container_open()."""
    st.markdown("</div>", unsafe_allow_html=True)


# Backward-compat aliases used by existing views.py
def chart_card(title: str, subtitle: str = "") -> None:
    chart_container_open(title, subtitle)


def chart_card_close() -> None:
    chart_container_close()


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE B — INSIGHT BAR (Pattern A / B / C)
# ─────────────────────────────────────────────────────────────────────────────

def insight_bar(
    bullets: list | str,
    action:  str = "",
    variant: str = "primary",
) -> None:
    """
    Full-width insight bar, always placed BELOW a chart or chart row.

    Template B rule: insight bars span the full width below the chart row.
    Never place inside an individual card in a paired row.
    The 'action' line (bold, coloured) is for the single most important
    next step — omit if none applies.

    bullets: list[str] or single str paragraph
    action:  bold "Action:" line (optional)
    variant: "primary" | "success" | "danger" | "warning" | "neutral"
    """
    _STYLES = {
        "primary": (DARK_NAVY, SURFACE_0),
        "success": (SUCCESS_BORDER, "#EAF3DE"),
        "danger":  (DANGER_BORDER,  "#FCEBEB"),
        "warning": (WARNING_BORDER, "#FAEEDA"),
        "neutral": (NEUTRAL, SURFACE_0),
    }
    border_color, bg_color = _STYLES.get(variant, _STYLES["primary"])

    if isinstance(bullets, str):
        body_html = (
            f'<p style="margin:0;font-size:13px;color:{TEXT_SEC};line-height:1.65">'
            f'{bullets}</p>'
        )
    else:
        items = "".join(
            f'<li style="font-size:13px;color:{TEXT_SEC};line-height:1.6;'
            f'margin-bottom:3px">{b}</li>'
            for b in bullets
        )
        body_html = (
            f'<ul style="margin:4px 0 0 0;padding-left:18px;list-style:disc">'
            f'{items}</ul>'
        )

    action_html = (
        f'<div style="font-size:12px;font-weight:600;color:{border_color};'
        f'margin-top:8px;padding-top:6px;border-top:1px solid rgba(0,0,0,0.07)">'
        f'{action}</div>'
    ) if action else ""

    st.markdown(
        f'<div style="padding:10px 14px;border-left:3px solid {border_color};'
        f'background:{bg_color};border-radius:0 6px 6px 0;'
        f'margin:8px 0 16px;font-family:Inter,sans-serif">'
        f'{body_html}'
        f'{action_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE B — PATTERN D: SHARP-FINDING CARD
# ─────────────────────────────────────────────────────────────────────────────

def sharp_finding_card(
    eyebrow: str,
    stat:    str,
    context: str,
    sub:     str = "",
) -> None:
    """
    Pattern D sharp-finding card — one entity, one severe finding.

    Template B rule: use once or twice per tab at most.
    Left border in token-danger. No chart inside.
    'sub' is one small muted line — a cross-reference or caveat.
    """
    sub_html = (
        f'<div style="font-size:11px;color:{TEXT_MUT};margin-top:8px;'
        f'padding-top:6px;border-top:1px solid {BORDER};line-height:1.45">'
        f'{sub}</div>'
    ) if sub else ""

    st.markdown(
        f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
        f'border-left:4px solid {DANGER};border-radius:0 10px 10px 0;'
        f'padding:16px 18px;margin-bottom:12px;font-family:Inter,sans-serif">'
        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.07em;color:{DANGER};margin-bottom:4px">{eyebrow}</div>'
        f'<div style="font-size:28px;font-weight:700;color:{DANGER};'
        f'line-height:1.1">{stat}</div>'
        f'<div style="font-size:12px;color:{TEXT_SEC};margin-top:4px;'
        f'line-height:1.5">{context}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE B — CLOSING SYNTHESIS (tab summary)
# ─────────────────────────────────────────────────────────────────────────────

def tab_synthesis(lead: str, tiles: list) -> None:
    """
    Three-tile closing synthesis strip — required on every tab page.

    Template B rule: full-width, final element on the page, bordered
    in the strongest accent token. Content is tab-specific: what this
    tab found, why it matters, what to do next.

    lead:  bold header line above the tiles
    tiles: list of dicts:
      label   str  — short uppercase label (e.g. "Biggest clinical question")
      body    str  — 2–3 sentence finding
      variant str  — "danger" | "warning" | "success" | "primary" | "neutral"
    """
    _BORDER = {
        "danger":  DANGER,
        "warning": WARNING,
        "success": SUCCESS,
        "primary": PRIMARY,
        "neutral": NEUTRAL,
    }
    tiles_html = ""
    for tile in tiles:
        bc = _BORDER.get(tile.get("variant", "neutral"), NEUTRAL)
        tiles_html += (
            f'<div style="padding:10px 12px;background:{SURFACE_0};border-radius:6px;'
            f'border-left:3px solid {bc};font-family:Inter,sans-serif">'
            f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:{TEXT_MUT};margin-bottom:4px">{tile["label"]}</div>'
            f'<div style="font-size:11px;color:{TEXT_SEC};line-height:1.5">{tile["body"]}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="border:2px solid {PRIMARY};border-radius:10px;'
        f'padding:16px 18px;background:rgba(12,68,124,.04);'
        f'font-family:Inter,sans-serif">'
        f'<div style="font-size:12px;font-weight:600;color:{PRIMARY};'
        f'margin-bottom:12px">{lead}</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">'
        f'{tiles_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILITY COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def reliability_strip(items: list) -> None:
    """
    Traffic-light data reliability strip (Section 9 pattern).

    Each item dict:
      domain     str  — short domain name
      note       str  — one-sentence limitation
      status     str  — "red" | "amber" | "green"
    """
    _DOT = {"red": DANGER, "amber": WARNING, "green": SUCCESS}
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        dot_color = _DOT.get(item.get("status", "amber"), NEUTRAL)
        col.markdown(
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
            f'border-radius:8px;padding:10px 12px;text-align:center;'
            f'font-family:Inter,sans-serif">'
            f'<div style="width:10px;height:10px;border-radius:50%;'
            f'background:{dot_color};margin:0 auto 6px"></div>'
            f'<div style="font-size:11px;font-weight:600;color:{TEXT_SEC};'
            f'margin-bottom:4px">{item["domain"]}</div>'
            f'<div style="font-size:10px;color:{TEXT_MUT};line-height:1.4">'
            f'{item["note"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def recommendation_cards(recs: list) -> None:
    """
    Two-column grid of recommendation cards (Section 10 pattern).

    Each rec dict:
      num     str  — "01", "02" …
      title   str  — short imperative title
      body    str  — 2–3 sentences
      source  str  — "Sections 3, 4, 5"
    """
    col_a, col_b = st.columns(2)
    for i, rec in enumerate(recs):
        col = col_a if i % 2 == 0 else col_b
        col.markdown(
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
            f'border-radius:10px;padding:14px 16px;margin-bottom:10px;'
            f'font-family:Inter,sans-serif">'
            f'<div style="font-size:10px;font-weight:700;color:{TEXT_MUT};'
            f'margin-bottom:4px">{rec["num"]}</div>'
            f'<div style="font-size:13px;font-weight:600;color:{TEXT_PRI};'
            f'margin-bottom:6px">{rec["title"]}</div>'
            f'<div style="font-size:12px;color:{TEXT_SEC};line-height:1.6">'
            f'{rec["body"]}</div>'
            f'<div style="font-size:10px;color:{TEXT_MUT};margin-top:8px">'
            f'{rec["source"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


_PRIORITY_SEVERITY_COLOR = {
    "critical": DANGER,   # red   — needs action now
    "monitor":  WARNING,  # amber — keep watching / not yet urgent
    "okay":     SUCCESS,  # green — working as intended, no action needed
}


def priority_cards(cards: list) -> None:
    """
    Two-column grid of severity-colored priority/recommendation cards —
    the standard "what this tab found, and what to do next" pattern used
    across every tab's closing section.

    Each card dict:
      label     str — small colored heading, e.g. "PRIORITY 1" or
                       "Urgent — Clinical Quality"
      title     str — bold title, one line
      body      str — 1-3 sentences
      severity  str — "critical" | "monitor" | "okay" -> red / amber / green
                       left border and label color. Defaults to "monitor"
                       if omitted.
      source    str — optional small gray footer, e.g. "Sections 3, 4, 5"
    """
    # Fresh columns per row (not one long stack per side) so each row's two
    # cards actually sit level with each other; min-height on the card itself
    # keeps them aligned even when body text length differs between the two.
    for row_start in range(0, len(cards), 2):
        row_cards = cards[row_start:row_start + 2]
        row_cols = st.columns(2)
        for col, c in zip(row_cols, row_cards):
            color = _PRIORITY_SEVERITY_COLOR.get(c.get("severity", "monitor"), WARNING)
            source_html = (
                f'<div style="font-size:10px;color:{TEXT_MUT};margin-top:8px">{c["source"]}</div>'
                if c.get("source") else ""
            )
            col.markdown(
                f'<div style="background:{SURFACE_1};border:1px solid {BORDER};border-left:4px solid {color};'
                f'border-radius:8px;padding:14px 16px;margin-bottom:12px;font-family:Inter,sans-serif;'
                f'min-height:230px;box-sizing:border-box;display:flex;flex-direction:column">'
                f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;'
                f'color:{color};margin-bottom:6px">{c["label"]}</div>'
                f'<div style="font-size:14px;font-weight:700;color:{TEXT_PRI};margin-bottom:6px;'
                f'line-height:1.35">{c["title"]}</div>'
                f'<div style="font-size:12px;color:{TEXT_SEC};line-height:1.6">{c["body"]}</div>'
                f'<div style="flex:1"></div>'
                f'{source_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


def key_findings_cards(items: list) -> None:
    """
    Two-column grid of "key findings" cards — same visual design as
    priority_cards() (colored left border, uppercase eyebrow label), reused
    for tabs whose closing section reports findings rather than action items.

    Each item dict:
      num    str — "01", "02" …
      title  str — short label ("Most specific finding")
      body   str — 2–3 sentences
    """
    # Fresh columns per row — see priority_cards() for why.
    for row_start in range(0, len(items), 2):
        row_items = items[row_start:row_start + 2]
        row_cols = st.columns(2)
        for col, item in zip(row_cols, row_items):
            source_html = (
                f'<div style="font-size:10px;color:{TEXT_MUT};margin-top:8px">{item["source"]}</div>'
                if item.get("source") else ""
            )
            col.markdown(
                f'<div style="background:{SURFACE_1};border:1px solid {BORDER};border-left:4px solid {PRIMARY};'
                f'border-radius:8px;padding:14px 16px;margin-bottom:12px;font-family:Inter,sans-serif;'
                f'min-height:168px;box-sizing:border-box;display:flex;flex-direction:column">'
                f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;'
                f'color:{PRIMARY};margin-bottom:6px">FINDING {item["num"]}</div>'
                f'<div style="font-size:14px;font-weight:700;color:{TEXT_PRI};margin-bottom:6px;'
                f'line-height:1.35">{item["title"]}</div>'
                f'<div style="font-size:12px;color:{TEXT_SEC};line-height:1.6">{item["body"]}</div>'
                f'<div style="flex:1"></div>'
                f'{source_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


def freshness_bar(date_str: str, schema: str = "") -> None:
    """Data freshness indicator strip."""
    schema_html = (
        f' &nbsp;·&nbsp; <code style="font-size:10px">{schema}</code>'
        if schema else ""
    )
    st.markdown(
        f'<div style="background:#EAF3DE;border:1px solid #639922;border-radius:6px;'
        f'padding:7px 10px;font-size:12px;color:#3B6D11;margin-bottom:12px">'
        f'Data as of <strong>{date_str}</strong>{schema_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# NUMBER FORMATTERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_num(v, suffix: str = "") -> str:
    """Format a number as 1,234 / 12.3K / 1.2M. Returns '—' for None."""
    if v is None:
        return "—"
    try:
        f = float(v)
        if abs(f) >= 1_000_000:
            return f"{f / 1_000_000:.1f}M{suffix}"
        if abs(f) >= 1_000:
            return f"{f / 1_000:.1f}K{suffix}"
        return f"{f:,.0f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def fmt_pct(v, decimals: int = 1) -> str:
    """Format as percentage string. Returns '—' for None."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{decimals}f}%"
    except (TypeError, ValueError):
        return str(v)


def fmt_delta(v, positive_good: bool = True) -> tuple[str, str]:
    """
    Return (formatted_string, color_hex) for a delta value.
    positive_good=True means a positive number is green (growth).
    positive_good=False means a positive number is red (e.g. error rate).
    """
    if v is None:
        return "—", TEXT_MUT
    try:
        f = float(v)
        sign  = "+" if f > 0 else ""
        good  = (f > 0) == positive_good
        color = SUCCESS if good else DANGER if f != 0 else TEXT_MUT
        return f"{sign}{f:.1f}%", color
    except (TypeError, ValueError):
        return str(v), TEXT_MUT