"""
app.py — Afya Clinical Analytics
==================================
6-tab structure:
  Tab 1: Clinical Overview   — volume, growth, what is driving it
  Tab 2: Patient Flow        — retention, churn, re-engagement
  Tab 3: Patient Segmentation — who they are + clinical revenue link
  Tab 4: Disease Burden      — sub-tabbed deep dive
  Tab 5: Clinical Quality    — workload, investigations, medication
  Tab 6: Patient Card        — clinician + HoC only

Role visibility:
  Head of Clinician:  Tabs 1-6 (full detail everywhere)
  Clinician:          All patients across facility + who treated them + patient card
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_module"))

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta

import queries as Q
from queries import run_query
import views as V

st.set_page_config(
    page_title="Afya Clinical Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

pio.templates["afya"] = pio.templates["plotly_white"]
_t = pio.templates["afya"].layout
_t.font        = dict(family="Montserrat, sans-serif", color=COOL_BLUE, size=11)
_t.legend.font = dict(family="Montserrat, sans-serif", color=COOL_BLUE, size=10)
_t.xaxis.tickfont   = dict(color=MUTED, size=10)
_t.xaxis.title.font = dict(color=COOL_BLUE, size=11)
_t.yaxis.tickfont   = dict(color=MUTED, size=10)
_t.yaxis.title.font = dict(color=COOL_BLUE, size=11)
_t.xaxis.gridcolor  = "#EBF3FB"
_t.yaxis.gridcolor  = "#EBF3FB"
_t.paper_bgcolor    = "#fff"
_t.plot_bgcolor     = "#fff"
pio.templates.default = "afya"

CHART_LAYOUT = dict(
    plot_bgcolor="#fff", paper_bgcolor="#fff",
    font=dict(family="Montserrat, sans-serif", size=11, color=COOL_BLUE),
    margin=dict(t=10, b=10, l=0, r=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(family="Montserrat, sans-serif", size=10, color=COOL_BLUE),
                bgcolor="rgba(0,0,0,0)"),
    colorway=SEQ,
)
AXIS = dict(
    showgrid=True, gridcolor="#EBF3FB", zeroline=False, color=COOL_BLUE,
    tickfont=dict(color=MUTED, size=10, family="Montserrat, sans-serif"),
    title_font=dict(color=COOL_BLUE, size=11, family="Montserrat, sans-serif"),
    title_standoff=8,
)
def _ax(**overrides): return {**AXIS, **overrides}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html, body { font-family: 'Montserrat', sans-serif; background: #fff; color: #003467; }
.stApp { background: #F4F8FC; }
[data-testid="stSidebar"] { background: #F4F8FC; border-right: 1px solid #D6E4F0; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stMarkdown { color: #003467; font-family: 'Montserrat', sans-serif; }
[data-testid="stSidebar"] span[class*="material"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
.sh { font-size:11px; font-weight:700; color:#0072CE; text-transform:uppercase;
      letter-spacing:2px; padding:8px 0 6px; border-bottom:2px solid #EBF3FB; margin-bottom:10px; }
/* Diagnostic / predictive / prescriptive insight styles */
.insight-blue  { background:#F4F8FC; border-left:4px solid #0072CE; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-teal  { background:#E8F7F4; border-left:4px solid #0BB99F; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-amber { background:#FFFBEB; border-left:4px solid #D97706; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-red   { background:#FFF5F5; border-left:4px solid #E53E3E; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-purple { background:#F5F0FF; border-left:4px solid #7B5EA7; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-lbl   { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px; }
.insight-txt   { font-size:13px; color:#003467; line-height:1.6; }
/* Analytics tier badges */
.badge-diag  { background:#EBF5FF; color:#185FA5; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; }
.badge-pred  { background:#FFFBEB; color:#854F0B; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; }
.badge-presc { background:#E8F7F4; color:#0F6E56; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; }
/* KPI & flag cards */
.kpi-card { background:#fff; border:1px solid #D6E4F0; border-radius:8px; padding:14px 14px 10px; }
.fbadge-high  { background:#FED7D7; color:#9B2C2C; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; }
.fbadge-med   { background:#FEEBC8; color:#7B341E; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; }
.fbadge-watch { background:#BEE3F8; color:#2C5282; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; }
.fbadge-ok    { background:#C6F6D5; color:#276749; font-size:10px; font-weight:700; padding:2px 8px; border-radius:20px; }
.freshness { background:#E8F7F4; border:1px solid #9FE1CB; border-radius:6px;
             padding:8px 10px; font-size:11px; color:#0F6E56; margin-bottom:12px; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#B0C8E0; border-radius:10px; }
[data-baseweb="tab"] { font-family:'Montserrat',sans-serif !important; font-weight:600 !important;
                        font-size:13px !important; color:#6B8CAE !important; }
[aria-selected="true"] { color:#0072CE !important; border-bottom-color:#0072CE !important; }
</style>
""", unsafe_allow_html=True)

# ── UI HELPERS ────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub="", delta="", delta_color=MUTED, color=COOL_BLUE):
    delta_html = f'<div style="font-size:11px;color:{delta_color};margin-top:3px">{delta}</div>' if delta else ""
    sub_html   = f'<div style="font-size:10px;color:{MUTED};margin-top:2px">{sub}</div>' if sub else ""
    st.markdown(
        f'<div style="background:#fff;border:1px solid {BORDER};border-radius:8px;padding:14px 14px 10px">'
        f'<div style="font-size:10px;font-weight:600;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:1.2px;margin-bottom:5px">{label}</div>'
        f'<div style="font-size:22px;font-weight:700;color:{color};line-height:1.1">{value}</div>'
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
    action_html = f'<div style="font-size:11px;font-weight:600;color:#0F6E56;margin-top:6px">→ {action}</div>' if action else ""
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

def freshness_bar(date_str, schema=""):
    schema_html = f' &nbsp;·&nbsp; <code style="font-size:10px">{schema}</code>' if schema else ""
    st.markdown(f'<div class="freshness">📅 Data as of <strong>{date_str}</strong>{schema_html} · Anchored to MAX(visit_date)</div>', unsafe_allow_html=True)

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
        textposition="outside", textfont=dict(size=10, color=COOL_BLUE),
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
                           textfont=dict(size=10,family="Montserrat, sans-serif")))
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
if "filters" not in st.session_state:
    st.session_state["filters"] = {
        "source_schemas": [], "schema_display": [], "facilities": [],
        "visit_type": "All", "payer_type": "All",
        "age_group": "All", "disease_group": "All",
        "date_range": "Last 12 months",
        "date_from": None, "date_to": None,
    }
if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = None
if "selected_schema" not in st.session_state:
    st.session_state["selected_schema"] = None

# ── TOPBAR ────────────────────────────────────────────────────────────────────
def _get_freshness():
    try:
        df = run_query("SELECT MAX(created_at)::DATE AS max_date FROM HOSPITALS.STAGING.STG_EVALUATION_VISITS")
        if not df.empty and df["max_date"].iloc[0] is not None:
            return str(df["max_date"].iloc[0])[:10]
    except: pass
    return "—"

def render_topbar():
    c1, c2, c3 = st.columns([3, 4, 3])
    with c1:
        schemas = st.session_state.get("filters", {}).get("schema_display", [])
        schema_display = " · ".join(schemas) if schemas else "All hospitals"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding-top:6px">'
            f'<div style="width:9px;height:9px;border-radius:50%;background:#0BB99F;flex-shrink:0"></div>'
            f'<span style="font-size:15px;font-weight:700;color:#003467">Afya Clinical Analytics</span>'
            f'<code style="font-size:10px;background:#F4F8FC;border:1px solid #D6E4F0;'
            f'border-radius:4px;padding:2px 7px;color:#6B8CAE">{schema_display}</code>'
            f'</div>', unsafe_allow_html=True)
    with c2:
        freshness = _get_freshness()
        st.markdown(
            f'<div style="text-align:center;padding-top:10px;font-size:11px;color:#6B8CAE">'
            f'📅 Data as of <strong style="color:#0F6E56">{freshness}</strong>'
            f'&nbsp;·&nbsp;anchored to MAX(visit_date)</div>', unsafe_allow_html=True)
    with c3:
        _roles = ["Head of Clinician", "Clinician"]
        _cur   = st.session_state.get("role", _roles[0])
        role = st.selectbox(
            "Role", _roles,
            index=_roles.index(_cur) if _cur in _roles else 0,
            label_visibility="collapsed", key="role_selector")
        st.session_state["role"] = role
    st.divider()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# Maps raw source_schema values to human-readable display names.
# Multiple raw schemas can map to the same display name (deduplication).
SCHEMA_DISPLAY = {
    "kisumu":      "Kisumu Specialists",
    "tenri":       "Tendri",
}

def _display_name(schema: str) -> str:
    return SCHEMA_DISPLAY.get(schema.lower(), schema)

def _display_to_schemas(display_names: list) -> list:
    """Expand selected display names back to all matching raw schemas."""
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

def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-size:11px;font-weight:700;color:#0072CE;text-transform:uppercase;'
            'letter-spacing:2px;padding:8px 0 6px;border-bottom:2px solid #EBF3FB;margin-bottom:12px">'
            'FILTERS</div>', unsafe_allow_html=True)

        # ── Date range ────────────────────────────────────────────────────
        with st.expander("📅 Date range", expanded=True):
            col_from, col_to = st.columns(2)
            with col_from:
                date_from = st.date_input("From", value=None, key="date_from",
                                          label_visibility="visible")
            with col_to:
                date_to = st.date_input("To", value=None, key="date_to",
                                        label_visibility="visible")
            quick = st.radio("Quick range",
                ["Last 12 months", "Last 6 months", "Last 90 days", "Custom"],
                index=0, label_visibility="collapsed")
            if quick != "Custom":
                effective_from = None
                effective_to   = None
            else:
                effective_from = str(date_from) if date_from else None
                effective_to   = str(date_to)   if date_to   else None

        # ── Hospital — fixed display names, no DB query ───────────────────
        with st.expander("🏥 Hospital", expanded=True):
            schema_opts = list(SCHEMA_DISPLAY.values())
            _hosp_default = ["Kisumu Specialists"] if "Kisumu Specialists" in schema_opts else []
            selected_display = st.multiselect(
                "Select hospital", options=schema_opts,
                default=_hosp_default, placeholder="All hospitals",
                label_visibility="collapsed")
            selected_schemas = _display_to_schemas(selected_display)

        # ── Facility — reads from clinic column ───────────────────────────
        with st.expander("🏢 Facility", expanded=True):
            facility_opts = _facility_options(selected_schemas)
            _fac_default = ["1"] if "1" in facility_opts else (facility_opts[:1] if facility_opts else [])
            selected_facilities = st.multiselect(
                "Select facilities", options=facility_opts,
                default=_fac_default, placeholder="All facilities" if facility_opts else "Select a hospital first",
                label_visibility="collapsed",
                disabled=not facility_opts)

        with st.expander("🚪 Visit type", expanded=False):
            visit_type = st.radio("Visit type", ["All", "Outpatient", "Inpatient"],
                index=0, label_visibility="collapsed")

        with st.expander("💳 Payer type", expanded=False):
            payer_type = st.radio("Payer", ["All", "NHIF / SHA", "Private insurance", "Cash"],
                index=0, label_visibility="collapsed")

        with st.expander("👥 Age group", expanded=False):
            age_group = st.radio("Age",
                ["All", "Paediatric (<18)", "Adult (18–64)", "Senior (65+)"],
                index=0, label_visibility="collapsed")

        with st.expander("🦠 Disease group", expanded=False):
            disease_group = st.radio("Disease group",
                ["All", "NCD / Chronic", "Communicable", "MNCH", "Injury"],
                index=0, label_visibility="collapsed")

        st.divider()

        with st.expander("📖 Abbreviations", expanded=False):
            st.markdown("""
<div style="font-size:11px;line-height:1.8;color:#003467">
<b>ANC</b> — Antenatal Care<br><b>BP</b> — Blood Pressure<br>
<b>HTN</b> — Hypertension<br><b>LTFU</b> — Lost to Follow-Up (&gt;180 days)<br>
<b>MNCH</b> — Maternal, Newborn &amp; Child Health<br>
<b>MoM</b> — Month-on-Month<br><b>NCD</b> — Non-Communicable Disease<br>
<b>NHIF / SHA</b> — Kenya public insurer<br><b>PNC</b> — Postnatal Care<br>
<b>pp</b> — Percentage points<br><b>YoY</b> — Year-on-Year
</div>""", unsafe_allow_html=True)

        st.divider()
        if st.button("✕ Clear all filters", width='stretch'):
            st.session_state["filters"] = {
                "source_schemas": [], "schema_display": [], "facilities": [],
                "visit_type": "All", "payer_type": "All",
                "age_group": "All", "disease_group": "All",
                "date_range": "Last 12 months",
                "date_from": None, "date_to": None,
            }
            st.rerun()

    filters = {
        "source_schemas":  selected_schemas,
        "schema_display":  selected_display,
        "facilities":      selected_facilities,
        "visit_type":      visit_type,
        "payer_type":      payer_type,
        "age_group":       age_group,
        "disease_group":   disease_group,
        "date_range":      quick if quick != "Custom" else "Last 12 months",
        "date_from":       effective_from,
        "date_to":         effective_to,
    }
    st.session_state["filters"] = filters
    return filters

# ── MAIN ─────────────────────────────────────────────────────────────────────
# ── HELPERS DICT — consumed by views.py ──────────────────────────────────────
st.session_state["helpers"] = {
    "gap":      lambda px=10: st.markdown(f'<div style="margin:{px}px 0"></div>', unsafe_allow_html=True),
    "sh":       lambda t, mt=0: st.markdown(f'<div class="sh" style="margin-top:{mt}px">{t}</div>', unsafe_allow_html=True),
    "kpi_card": kpi_card,
    "pc":       lambda f: st.plotly_chart(f, use_container_width=True),
    "note":     lambda t, warn=False: st.markdown(
                    f'<div style="background:{"#FFFBEB" if warn else "#F4F8FC"};'
                    f'border-left:3px solid {"#D97706" if warn else "#0072CE"};'
                    f'padding:7px 11px;font-size:13px;margin-top:5px;border-radius:0 4px 4px 0">{t}</div>',
                    unsafe_allow_html=True),
    "fmt_num":  fmt_num,
    "fmt_pct":  fmt_pct,
    "fmt_kes":  lambda v: fmt_num(v, suffix=""),
}

render_topbar()
filters = render_sidebar()
role    = st.session_state.get("role", "Head of Clinician")

if role == "Clinician":
    V.render_clinician_view(filters, run_query)

else:  # Head of Clinician — full access to all tabs including patient card
    tabs = st.tabs([
        "📈  Operations",
        "🧑‍🤝‍🧑  Patient Demographics",
        "📊  Flow and Retention",
        "🦠  Disease Burden",
        "🩺  Clinical Quality & Workload",
        "👤  Patient Card",
    ])
    with tabs[0]: V.render_tab1_operations(filters, run_query)
    with tabs[1]: V.render_tab2_segmentation(filters, run_query)
    with tabs[2]: V.render_tab3_retention(filters, run_query)
    with tabs[3]: V.render_tab4_disease_burden(filters, run_query)
    with tabs[4]: V.render_tab5_workload(filters, run_query)
    with tabs[5]: V.render_clinician_view(filters, run_query)