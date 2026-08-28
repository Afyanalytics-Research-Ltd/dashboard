import sys
import os
# Add dashboards/ to path so 'import ksh.clinical_module.X' resolves correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta

import ksh.clinical_module.queries as Q
import ksh.clinical_module.views as V
from ksh.clinical_module.queries import run_query
from ksh.clinical_module.ui_template import inject_global_css

st.set_page_config(
    page_title="Afya Clinical Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
inject_global_css()
st.markdown("<style>" + """
html, body, .stMarkdown, .stMetric, .stDataFrame, .stSelectbox, .stRadio,
.stCaption, .element-container, .block-container, button, input, select,
textarea, label, p {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
[data-testid="stIconMaterial"],
[data-testid="stExpanderIcon"],
[data-testid="stExpanderIconCheck"],
[data-testid="stExpanderIconError"],
[data-testid="stExpanderIconSpinner"] {
    font-family: 'Material Symbols Rounded' !important;
    font-style: normal !important;
    font-weight: 400 !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}
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
    min-height: 40px !important;
}
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
/* Section F accordion: arrow toggle buttons in narrow right columns */
[data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child button,
[data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-of-type button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #9CA3AF !important;
    font-size: 16px !important;
    padding: 8px 4px !important;
    min-height: unset !important;
    line-height: 1 !important;
}
[data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child button:hover,
[data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="column"]:last-of-type button:hover {
    color: #374151 !important;
    background: rgba(0,0,0,0.04) !important;
    border-radius: 4px !important;
}
""" + "</style>", unsafe_allow_html=True)

# ── PALETTE ───────────────────────────────────────────────────────────────────
AFYA_BLUE = "#0072CE"
TEAL      = "#0BB99F"
COOL_BLUE = "#003467"
ORANGE    = "#F5A623"
CORAL     = "#E05C5C"
PURPLE    = "#7B5EA7"
GRAY      = "#ADB5BD"
MUTED     = "#6B8CAE"
BORDER    = "#D6E4F0"
BG_LIGHT  = "#F4F8FC"
SEQ       = [TEAL, AFYA_BLUE, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY]
GREEN     = "#38A169"
AMBER     = "#D97706"
RED       = "#C53030"

# Plotly template and chart defaults are set by ui_template.py (imported above)
CHART_LAYOUT = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    autosize      = True,
    font          = dict(family="Inter, -apple-system, sans-serif", size=12, color="#6B7280"),
    margin        = dict(t=8, b=0, l=0, r=16),
    legend        = dict(orientation="h", y=1.08, x=0,
                         font=dict(family="Inter, -apple-system, sans-serif", size=11),
                         bgcolor="rgba(0,0,0,0)"),
    colorway      = SEQ,
)
AXIS = dict(
    showgrid    = True,
    gridcolor   = "rgba(0,0,0,0.05)",
    zeroline    = False,
    color       = "#6B7280",
    tickfont    = dict(color="#9CA3AF", size=11, family="Inter, -apple-system, sans-serif"),
    title_font  = dict(color="#6B7280", size=11, family="Inter, -apple-system, sans-serif"),
    title_standoff = 8,
)
def _ax(**overrides): return {**AXIS, **overrides}

# ── UI HELPERS ────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", delta="", delta_color=MUTED, color=COOL_BLUE):
    delta_html = f'<div style="font-family:Montserrat,sans-serif;font-size:12px;color:{delta_color};margin-top:3px">{delta}</div>' if delta else ""
    sub_html   = f'<div style="font-family:Inter,sans-serif;font-size:11px;color:{MUTED};margin-top:2px">{sub}</div>' if sub else ""
    st.markdown(
        f'<div style="background:#fff;border:1px solid {BORDER};border-radius:8px;padding:14px 14px 10px">'
        f'<div style="font-family:Montserrat,sans-serif;font-size:11px;font-weight:600;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:1.2px;margin-bottom:5px">{label}</div>'
        f'<div style="font-family:Montserrat,sans-serif;font-size:26px;font-weight:700;color:{color};line-height:1.1">{value}</div>'
        f'{delta_html}{sub_html}</div>', unsafe_allow_html=True)

def section_header(title):
    st.markdown(f'<div class="sh">{title}</div>', unsafe_allow_html=True)

def insight_card(text, label="Key insight", variant="blue"):
    color_map = {"blue":"#0072CE","teal":"#0BB99F","amber":"#D97706","red":"#C53030","purple":"#7B5EA7"}
    lbl_color = color_map.get(variant, "#0072CE")
    st.markdown(
        f'<div class="insight-{variant}">'
        f'<div class="insight-lbl" style="color:{lbl_color}">{label}</div>'
        f'<div class="insight-txt">{text}</div></div>', unsafe_allow_html=True)

def prescriptive_card(text, action=""):
    action_html = f'<div style="font-size:13px;font-weight:600;color:#0F6E56;margin-top:6px">→ {action}</div>' if action else ""
    st.markdown(
        f'<div style="background:#E8F7F4;border-left:4px solid #0BB99F;border-radius:0 6px 6px 0;padding:10px 13px;margin-bottom:10px">'
        f'<div class="insight-lbl" style="color:#0BB99F">Recommended action</div>'
        f'<div class="insight-txt">{text}</div>{action_html}</div>', unsafe_allow_html=True)

def diagnostic_card(text):
    st.markdown(
        f'<div style="background:#EBF5FF;border-left:4px solid #0072CE;border-radius:0 6px 6px 0;padding:10px 13px;margin-bottom:10px">'
        f'<div class="insight-lbl" style="color:#0072CE">Why this is happening</div>'
        f'<div class="insight-txt">{text}</div></div>', unsafe_allow_html=True)

def predictive_card(text):
    st.markdown(
        f'<div style="background:#FFFBEB;border-left:4px solid #D97706;border-radius:0 6px 6px 0;padding:10px 13px;margin-bottom:10px">'
        f'<div class="insight-lbl" style="color:#D97706">Predictive signal</div>'
        f'<div class="insight-txt">{text}</div></div>', unsafe_allow_html=True)

def flag_badge(level, text):
    cls = {"high":"fbadge-high","med":"fbadge-med","watch":"fbadge-watch","ok":"fbadge-ok"}.get(level,"fbadge-watch")
    return f'<span class="{cls}">{text}</span>'

def fmt_num(v, suffix=""):
    if v is None: return "—"
    try:
        f = float(v)
        if abs(f) >= 1_000_000: return f"{f/1_000_000:.1f}M{suffix}"
        if abs(f) >= 1_000:     return f"{f/1_000:.1f}K{suffix}"
        return f"{f:,.0f}{suffix}"
    except: return str(v)

def fmt_pct(v, d=1):
    if v is None: return "—"
    try: return f"{float(v):.{d}f}%"
    except: return str(v)

def fmt_delta(v, positive_good=True):
    if v is None: return "—", MUTED
    try:
        f = float(v)
        sign  = "+" if f > 0 else ""
        color = (GREEN if (f > 0) == positive_good else RED) if f != 0 else MUTED
        return f"{sign}{f:.1f}%", color
    except: return str(v), MUTED

def horizontal_bar(df, label_col, value_col, color=AFYA_BLUE, height=280, title="", xaxis_suffix=""):
    fig = go.Figure(go.Bar(
        y=df[label_col], x=df[value_col], orientation="h", marker_color=color,
        text=df[value_col].apply(lambda v: f"{v:.1f}{xaxis_suffix}"),
        textposition="outside", textfont=dict(size=12, color=COOL_BLUE),
    ))
    fig.update_layout(**{**CHART_LAYOUT,"height":height,"title_text":title},
                      xaxis={**_ax(),"showgrid":False},
                      yaxis={**_ax(),"showgrid":False,"autorange":"reversed"},
                      showlegend=False)
    return fig

def line_chart(df, x_col, y_cols, labels=None, colors=None, height=280, title="", yaxis_suffix=""):
    colors = colors or SEQ; labels = labels or y_cols
    fig = go.Figure()
    for i,(col,lbl) in enumerate(zip(y_cols,labels)):
        fig.add_trace(go.Scatter(x=df[x_col],y=df[col],name=lbl,mode="lines+markers",
                                 line=dict(color=colors[i%len(colors)],width=2),marker=dict(size=4)))
    fig.update_layout(**{**CHART_LAYOUT,"height":height,"title_text":title},
                      xaxis=_ax(),yaxis={**_ax(),"ticksuffix":yaxis_suffix})
    return fig

def stacked_bar(df, x_col, y_cols, labels=None, colors=None, height=280, title=""):
    colors = colors or SEQ; labels = labels or y_cols
    fig = go.Figure()
    for i,(col,lbl) in enumerate(zip(y_cols,labels)):
        fig.add_trace(go.Bar(x=df[x_col],y=df[col],name=lbl,marker_color=colors[i%len(colors)]))
    fig.update_layout(**{**CHART_LAYOUT,"height":height,"title_text":title},
                      barmode="stack",xaxis=_ax(),yaxis=_ax())
    return fig

def donut_chart(labels, values, colors=None, height=220, title=""):
    colors = colors or SEQ
    fig = go.Figure(go.Pie(labels=labels,values=values,hole=0.55,
                           marker=dict(colors=colors),textinfo="percent",
                           textfont=dict(size=12,family="Montserrat, sans-serif")))
    fig.update_layout(**{**CHART_LAYOUT,"height":height,"title_text":title,
                         "legend":dict(orientation="v",x=1.05,y=0.5,
                                       font=dict(size=10,family="Montserrat, sans-serif"))},
                      showlegend=True)
    return fig

def heatmap_chart(z, x_labels, y_labels, height=220, title="", colorscale="Blues"):
    fig = go.Figure(go.Heatmap(z=z,x=x_labels,y=y_labels,colorscale=colorscale,
                               showscale=False,xgap=2,ygap=2))
    fig.update_layout(**{**CHART_LAYOUT,"height":height,"title_text":title},
                      xaxis={**_ax(),"showgrid":False},yaxis={**_ax(),"showgrid":False})
    return fig

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state["role"] = "Head of Clinician"
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Today's Briefing"
if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = None
if "selected_schema" not in st.session_state:
    st.session_state["selected_schema"] = None

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DATA_START = "2024-09-01"

SCHEMA_DISPLAY = {
    "kisumu": "Kisumu Specialists",
    "tenri":  "Tendri",
}

def _display_name(schema: str) -> str:
    return SCHEMA_DISPLAY.get(schema.lower(), schema)

def _display_to_schemas(display_names: list) -> list:
    inverse: dict[str, list] = {}
    for raw, display in SCHEMA_DISPLAY.items():
        inverse.setdefault(display, []).append(raw)
    result = []
    for d in display_names:
        result.extend(inverse.get(d, [d]))
    return result

def _facility_options(schemas: list) -> list:
    try:
        if schemas:
            quoted = ", ".join(f"'{s}'" for s in schemas)
            sql = (
                f"SELECT DISTINCT clinic "
                f"FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS "
                f"WHERE source_schema IN ({quoted}) "
                f"AND clinic IS NOT NULL ORDER BY 1"
            )
        else:
            sql = (
                "SELECT DISTINCT clinic "
                "FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS "
                "WHERE clinic IS NOT NULL ORDER BY 1"
            )
        df = run_query(sql)
        df.columns = [c.lower() for c in df.columns]
        return df["clinic"].tolist() if not df.empty else []
    except:
        return []

# ── NAV OPTIONS ───────────────────────────────────────────────────────────────
def _nav_options(role: str) -> list:
    if role == "Clinician":
        return ["🩺  Patient Card"]
    return [
        "🗓  Today's Briefing",
        "📊  OPD → IPD Conversion",
        "🏥  Clinical Activity",
        "👥  Patient Acquisition",
        "🔄  Flow and Retention",
        "🦠  Disease Burden",
    ]


def _page_clean(page: str) -> str:
    """Strip leading emoji + whitespace from nav label to get the plain page name."""
    return page.split("  ", 1)[-1].strip()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # ── Logo ──────────────────────────────────────────────────────────
        st.markdown(
            '<div style="padding:14px 4px 6px;">'
            '<span style="font-size:18px;font-weight:800;color:#0F6E56;font-family:Inter,-apple-system,sans-serif;'
            'letter-spacing:-0.02em;">Afya</span>'
            '<span style="font-size:18px;font-weight:800;color:#111827;font-family:Inter,-apple-system,sans-serif;'
            'letter-spacing:-0.02em;">Analytics</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Facility badge ─────────────────────────────────────────────────
        st.markdown(
            '<div style="background:#F0FAF6;border:1px solid #A7F3D0;border-radius:6px;'
            'padding:7px 10px;margin:2px 0 12px;">'
            '<div style="font-size:12px;font-weight:700;color:#0F6E56;font-family:Inter,-apple-system,sans-serif;">'
            '🏥 Kisumu Specialists</div>'
            '<div style="font-size:11px;color:#6B7280;margin-top:1px;font-family:Inter,-apple-system,sans-serif;">'
            'Sep 2024 – present</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Navigation ────────────────────────────────────────────────────
        role = st.session_state.get("role", "Head of Clinician")
        nav_label = "Patient view" if role == "Clinician" else "Clinical tabs"
        st.markdown(f'<div class="nav-section">{nav_label}</div>', unsafe_allow_html=True)
        _nav_opts = _nav_options(role)
        # _nav_to is set by the role-toggle buttons as a pending redirect;
        # we consume it here (before the widget is created) to set the index.
        _pending = st.session_state.pop("_nav_to", None)
        if _pending and _pending in _nav_opts:
            _nav_idx = _nav_opts.index(_pending)
        else:
            _current = st.session_state.get("nav_page", _nav_opts[0])
            _nav_idx = _nav_opts.index(_current) if _current in _nav_opts else 0
        page = st.radio(
            label="Navigation",
            options=_nav_opts,
            index=_nav_idx,
            label_visibility="collapsed",
            key="nav_page",
        )

        st.markdown('<hr style="margin:10px 0;border:none;border-top:1px solid #E5E7EB">', unsafe_allow_html=True)

        # ── Hospital filter ───────────────────────────────────────────────
        with st.expander("🏥 Hospital", expanded=True):
            schema_opts = list(dict.fromkeys(SCHEMA_DISPLAY.values()))
            _hosp_default = ["Kisumu Specialists"] if "Kisumu Specialists" in schema_opts else []
            selected_display = st.multiselect(
                "Select hospital", options=schema_opts,
                default=_hosp_default, placeholder="All hospitals",
                label_visibility="collapsed")
            selected_schemas = _display_to_schemas(selected_display)

        # ── Facility filter ───────────────────────────────────────────────
        with st.expander("🏢 Facility", expanded=False):
            facility_opts = _facility_options(selected_schemas)
            _fac_default = ["1"] if "1" in facility_opts else (facility_opts[:1] if facility_opts else [])
            selected_facilities = st.multiselect(
                "Select facilities", options=facility_opts,
                default=_fac_default,
                placeholder="All facilities" if facility_opts else "Select a hospital first",
                label_visibility="collapsed",
                disabled=not facility_opts)

        st.markdown('<hr style="margin:10px 0;border:none;border-top:1px solid #E5E7EB">', unsafe_allow_html=True)

        # ── Role selector (segmented toggle) ──────────────────────────
        st.markdown('<div class="nav-section">View</div>', unsafe_allow_html=True)
        _c1, _c2 = st.columns(2, gap="small")
        with _c1:
            if st.button(
                "Head of Clinician",
                key="btn_hoc",
                use_container_width=True,
                type="primary" if role == "Head of Clinician" else "secondary",
            ):
                st.session_state["role"] = "Head of Clinician"
                st.session_state["_nav_to"] = "🗓  Today's Briefing"
                st.rerun()
        with _c2:
            if st.button(
                "Clinician",
                key="btn_clin",
                use_container_width=True,
                type="primary" if role == "Clinician" else "secondary",
            ):
                st.session_state["role"] = "Clinician"
                st.session_state["_nav_to"] = "🩺  Patient Card"
                st.rerun()

        st.markdown('<hr style="margin:10px 0;border:none;border-top:1px solid #E5E7EB">', unsafe_allow_html=True)

        if st.button("↺ Refresh data", use_container_width=True, type="secondary"):
            st.cache_data.clear()
            st.rerun()

    filters = {
        "source_schemas":  selected_schemas,
        "schema_display":  selected_display,
        "facilities":      selected_facilities,
        "date_range":      "Custom",
        "date_from":       DATA_START,
        "date_to":         None,
    }
    st.session_state["filters"] = filters
    return filters, page

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "") -> None:
    sub_html = f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="page-header">'
        f'<div class="page-title">{title}</div>'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── HELPERS DICT — consumed by views.py ──────────────────────────────────────
st.session_state["helpers"] = {
    "gap":      lambda px=10: st.markdown(f'<div style="margin:{px}px 0"></div>', unsafe_allow_html=True),
    "sh":       lambda t, mt=0: st.markdown(f'<div class="sh" style="margin-top:{mt}px">{t}</div>', unsafe_allow_html=True),
    "kpi_card": kpi_card,
    "pc":       lambda f: st.plotly_chart(f, use_container_width=True, config={"responsive": True, "displayModeBar": False, "useResizeHandler": True}),
    "note":     lambda t, warn=False: st.markdown(
                    f'<div style="background:{"#FFFBEB" if warn else "#F4F8FC"};'
                    f'border-left:3px solid {"#D97706" if warn else "#0072CE"};'
                    f'padding:7px 11px;font-size:15px;margin-top:5px;border-radius:0 4px 4px 0">{t}</div>',
                    unsafe_allow_html=True),
    "fmt_num":  fmt_num,
    "fmt_pct":  fmt_pct,
    "fmt_kes":  lambda v: fmt_num(v, suffix=""),
}

# ── MAIN ─────────────────────────────────────────────────────────────────────
import importlib
importlib.reload(V)

filters, page = render_sidebar()
role  = st.session_state.get("role", "Head of Clinician")
_page = _page_clean(page)

VIEW_MAP = {
    "Today's Briefing":     lambda: V.render_briefing(filters, run_query),
    "OPD → IPD Conversion": lambda: V.render_tab_opd_ipd(filters, run_query),
    "Clinical Activity":    lambda: V.render_tab_clinical_activity(filters, run_query),
    "Patient Acquisition":  lambda: V.render_tab2_patient_acquisition(filters, run_query),
    "Flow and Retention":   lambda: V.render_tab3_retention(filters, run_query),
    "Disease Burden":       lambda: V.render_tab4_disease_burden(filters, run_query),
    "Patient Card":         lambda: V.render_clinician_view(filters, run_query),
}

CLINICAL_TABS = {
    "Today's Briefing", "OPD → IPD Conversion", "Clinical Activity",
    "Patient Acquisition", "Flow and Retention", "Disease Burden",
}

if _page == "Patient Card" and role == "Head of Clinician":
    st.warning("Patient Card is only available in Clinician view.")
    st.stop()

if _page in CLINICAL_TABS and role == "Clinician":
    st.info("Switch to Head of Clinician view to access clinical dashboards.")
    st.stop()

render_fn = VIEW_MAP.get(_page)
if render_fn:
    render_fn()
