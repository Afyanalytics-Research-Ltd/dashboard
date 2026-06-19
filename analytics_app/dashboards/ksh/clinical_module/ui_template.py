"""
ui_template.py — Afya Clinical Analytics · Plotly theme + shared UI helpers
============================================================================
Mirrors the XanaLife ui_template pattern.
Import at the top of every view file.
"""

import json
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _st_components

# ── PALETTE ───────────────────────────────────────────────────────────────────
AFYA_BLUE = "#0072CE"
TEAL      = "#0BB99F"
COOL_BLUE = "#003467"
ORANGE    = "#f5a623"
CORAL     = "#e05c5c"
PURPLE    = "#7b5ea7"
GRAY      = "#adb5bd"
MUTED     = "#6B8CAE"
BORDER    = "#D6E4F0"
BG_LIGHT  = "#F4F8FC"
SEQ       = [TEAL, AFYA_BLUE, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY]

# Status colours
GREEN  = "#38A169"
AMBER  = "#D97706"
RED    = "#C53030"

# Clinical Activity palette
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
_t.font        = dict(family="Montserrat, sans-serif", color=COOL_BLUE, size=13)
_t.legend.font = dict(family="Montserrat, sans-serif", color=COOL_BLUE, size=12)
_t.xaxis.tickfont   = dict(color=MUTED, size=12)
_t.xaxis.title.font = dict(color=COOL_BLUE, size=13)
_t.yaxis.tickfont   = dict(color=MUTED, size=12)
_t.yaxis.title.font = dict(color=COOL_BLUE, size=13)
_t.xaxis.gridcolor  = "#EBF3FB"
_t.yaxis.gridcolor  = "#EBF3FB"
_t.paper_bgcolor    = "#fff"
_t.plot_bgcolor     = "#fff"
pio.templates.default = "afya"


# ── SHARED LAYOUT ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    plot_bgcolor  = "#fff",
    paper_bgcolor = "#fff",
    font          = dict(family="Montserrat, sans-serif", size=13, color=COOL_BLUE),
    margin        = dict(t=10, b=10, l=0, r=10),
    legend        = dict(
        orientation = "h",
        yanchor     = "bottom", y=1.02,
        xanchor     = "right",  x=1,
        font        = dict(family="Montserrat, sans-serif", size=12, color=COOL_BLUE),
        bgcolor     = "rgba(0,0,0,0)",
    ),
    colorway = SEQ,
)

AXIS = dict(
    showgrid     = True,
    gridcolor    = "#EBF3FB",
    zeroline     = False,
    color        = COOL_BLUE,
    tickfont     = dict(color=MUTED,     size=12, family="Montserrat, sans-serif"),
    title_font   = dict(color=COOL_BLUE, size=13, family="Montserrat, sans-serif"),
    title_standoff = 8,
)


# ── HELPER: axis dict merge ────────────────────────────────────────────────────
def _ax(**overrides):
    return {**AXIS, **overrides}


# ── GLOBAL CSS INJECTION ──────────────────────────────────────────────────────
def inject_global_css():
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stMetric,
.stDataFrame, .stSelectbox, .stRadio, .stCaption,
.element-container, .block-container {
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
html, body { background: #fff; color: #003467; }
h1, h2, h3, h4,
.section-label, .metric-label,
[data-testid="stMetricLabel"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 24px !important;
}
.section-lbl {
    font-family: 'Montserrat', sans-serif !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--text-color) !important;
    opacity: 0.45 !important;
    padding-bottom: 6px !important;
    border-bottom: 0.5px solid rgba(128,128,128,0.2) !important;
    margin-bottom: 12px !important;
}
.insight-bar {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    line-height: 1.6 !important;
}
.js-plotly-plot .plotly .gtitle {
    font-family: 'Montserrat', sans-serif !important;
}
.stApp { background: #F4F8FC; }
[data-testid="stSidebar"] { background: #F4F8FC; border-right: 1px solid #D6E4F0; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] .stMarkdown { color: #003467; font-family: 'Montserrat', sans-serif; }
[data-testid="stSidebar"] span[class*="material"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important; }
.sh { font-size:13px; font-weight:700; color:#0072CE; text-transform:uppercase;
      letter-spacing:2px; padding:8px 0 6px; border-bottom:2px solid #EBF3FB; margin-bottom:10px; }
.insight-blue   { background:#F4F8FC; border-left:4px solid #0072CE; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-teal   { background:#E8F7F4; border-left:4px solid #0BB99F; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-amber  { background:#FFFBEB; border-left:4px solid #D97706; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-red    { background:#FFF5F5; border-left:4px solid #E53E3E; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-purple { background:#F5F0FF; border-left:4px solid #7B5EA7; border-radius:0 6px 6px 0; padding:10px 13px; margin-bottom:10px; }
.insight-lbl { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px; }
.insight-txt { font-size:15px; color:#003467; line-height:1.6; }
.badge-diag  { background:#EBF5FF; color:#185FA5; font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; }
.badge-pred  { background:#FFFBEB; color:#854F0B; font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; }
.badge-presc { background:#E8F7F4; color:#0F6E56; font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; }
.kpi-card { background:#fff; border:1px solid #D6E4F0; border-radius:8px; padding:14px 14px 10px; }
.fbadge-high  { background:#FED7D7; color:#9B2C2C; font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; }
.fbadge-med   { background:#FEEBC8; color:#7B341E; font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; }
.fbadge-watch { background:#BEE3F8; color:#2C5282; font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; }
.fbadge-ok    { background:#C6F6D5; color:#276749; font-size:12px; font-weight:700; padding:2px 8px; border-radius:20px; }
.freshness { background:#E8F7F4; border:1px solid #9FE1CB; border-radius:6px;
             padding:8px 10px; font-size:13px; color:#0F6E56; margin-bottom:12px; }
[data-testid="stCaptionContainer"] p,
.stCaption p { font-size: 14px !important; line-height: 1.6; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#B0C8E0; border-radius:10px; }
[data-baseweb="tab"],
[data-baseweb="tab"] p,
[data-baseweb="tab"] span,
button[role="tab"],
button[role="tab"] p {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    color: #6B8CAE !important;
    letter-spacing: 0.01em !important;
}
[aria-selected="true"],
[aria-selected="true"] p,
[aria-selected="true"] span {
    color: #0072CE !important;
    border-bottom-color: #0072CE !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ── KPI CARD ──────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str = "", delta: str = "",
             delta_color: str = MUTED, color: str = COOL_BLUE):
    delta_html = (
        f'<div style="font-family:Montserrat,sans-serif;font-size:12px;color:{delta_color};margin-top:3px">{delta}</div>'
        if delta else ""
    )
    sub_html = (
        f'<div style="font-family:Inter,sans-serif;font-size:11px;color:{MUTED};margin-top:2px">{sub}</div>'
        if sub else ""
    )
    st.markdown(
        f'<div style="background:#fff;border:1px solid {BORDER};border-radius:8px;padding:14px 14px 10px">'
        f'<div style="font-family:Montserrat,sans-serif;font-size:11px;font-weight:600;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:1.2px;margin-bottom:5px">{label}</div>'
        f'<div style="font-family:Montserrat,sans-serif;font-size:26px;font-weight:700;color:{color};line-height:1.1">{value}</div>'
        f'{delta_html}{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── SECTION HEADER ────────────────────────────────────────────────────────────
def section_header(title: str):
    st.markdown(
        f'<div class="sh">{title}</div>',
        unsafe_allow_html=True,
    )


# ── INSIGHT CARD ──────────────────────────────────────────────────────────────
def insight_card(text: str, label: str = "Key insight", variant: str = "blue"):
    """variant: blue | teal | amber | red"""
    st.markdown(
        f'<div class="insight-{variant}">'
        f'<div class="insight-lbl" style="color:{"#0072CE" if variant=="blue" else "#0BB99F" if variant=="teal" else "#D97706" if variant=="amber" else "#C53030"}">'
        f'{label}</div>'
        f'<div class="insight-txt">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── FLAG BADGE ────────────────────────────────────────────────────────────────
def flag_badge(level: str, text: str) -> str:
    """Returns HTML string for inline use. level: high | med | watch | ok"""
    cls = {"high": "fbadge-high", "med": "fbadge-med",
           "watch": "fbadge-watch", "ok": "fbadge-ok"}.get(level, "fbadge-watch")
    return f'<span class="{cls}">{text}</span>'


# ── DATA FRESHNESS BAR ────────────────────────────────────────────────────────
def freshness_bar(date_str: str, schema: str = ""):
    schema_html = f' &nbsp;·&nbsp; <code style="font-size:10px">{schema}</code>' if schema else ""
    st.markdown(
        f'<div class="freshness">📅 Data as of <strong>{date_str}</strong>{schema_html} '
        f'· Anchored to MAX(visit_date)</div>',
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


def fmt_delta(v, positive_good: bool = True) -> tuple[str, str]:
    """Returns (formatted_string, color) for use in kpi_card."""
    if v is None:
        return "—", MUTED
    try:
        f = float(v)
        sign = "+" if f > 0 else ""
        color = (GREEN if (f > 0) == positive_good else RED) if f != 0 else MUTED
        return f"{sign}{f:.1f}%", color
    except (TypeError, ValueError):
        return str(v), MUTED


# ── STANDARD BAR CHART ────────────────────────────────────────────────────────
def horizontal_bar(
    df,
    label_col: str,
    value_col: str,
    color: str = AFYA_BLUE,
    height: int = 280,
    title: str = "",
    xaxis_suffix: str = "",
) -> go.Figure:
    fig = go.Figure(go.Bar(
        y=df[label_col],
        x=df[value_col],
        orientation="h",
        marker_color=color,
        text=df[value_col].apply(lambda v: f"{v:.1f}{xaxis_suffix}"),
        textposition="outside",
        textfont=dict(size=10, color=COOL_BLUE),
    ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        xaxis={**_ax(), "showgrid": False},
        yaxis={**_ax(), "showgrid": False, "autorange": "reversed"},
        showlegend=False,
    )
    return fig


# ── MULTI-LINE CHART ──────────────────────────────────────────────────────────
def line_chart(
    df,
    x_col: str,
    y_cols: list[str],
    labels: list[str] = None,
    colors: list[str] = None,
    height: int = 280,
    title: str = "",
    yaxis_suffix: str = "",
) -> go.Figure:
    colors = colors or SEQ
    labels = labels or y_cols
    fig = go.Figure()
    for i, (col, lbl) in enumerate(zip(y_cols, labels)):
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col],
            name=lbl,
            mode="lines+markers",
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        xaxis=_ax(),
        yaxis={**_ax(), "ticksuffix": yaxis_suffix},
    )
    return fig


# ── STACKED BAR CHART ─────────────────────────────────────────────────────────
def stacked_bar(
    df,
    x_col: str,
    y_cols: list[str],
    labels: list[str] = None,
    colors: list[str] = None,
    height: int = 280,
    title: str = "",
) -> go.Figure:
    colors = colors or SEQ
    labels = labels or y_cols
    fig = go.Figure()
    for i, (col, lbl) in enumerate(zip(y_cols, labels)):
        fig.add_trace(go.Bar(
            x=df[x_col], y=df[col],
            name=lbl,
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        barmode="stack",
        xaxis=_ax(),
        yaxis=_ax(),
    )
    return fig


# ── DONUT CHART ───────────────────────────────────────────────────────────────
def donut_chart(
    labels: list[str],
    values: list[float],
    colors: list[str] = None,
    height: int = 220,
    title: str = "",
) -> go.Figure:
    colors = colors or SEQ
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="percent",
        textfont=dict(size=10, family="Montserrat, sans-serif"),
    ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title,
           "legend": dict(orientation="v", x=1.05, y=0.5,
                          font=dict(size=10, family="Montserrat, sans-serif"))},
        showlegend=True,
    )
    return fig


# ── HEATMAP ───────────────────────────────────────────────────────────────────
def heatmap_chart(
    z: list,
    x_labels: list,
    y_labels: list,
    height: int = 220,
    title: str = "",
    colorscale: str = "Blues",
) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        showscale=False,
        xgap=2,
        ygap=2,
    ))
    fig.update_layout(
        **{**CHART_LAYOUT, "height": height, "title_text": title},
        xaxis={**_ax(), "showgrid": False},
        yaxis={**_ax(), "showgrid": False},
    )
    return fig


# ── SORTABLE TABLE ────────────────────────────────────────────────────────────
def render_sortable_table(
    df,
    height: int = 400,
    highlight_rules: list = None,
    badge_columns: dict = None,
    key: str = "table",
):
    """
    Custom HTML table with sticky header, scroll, click-to-sort, badges, and row highlights.

    Parameters
    ----------
    df              : DataFrame to display
    height          : fixed pixel height (scroll activates beyond this)
    highlight_rules : list of dicts:
        [{'column': 'col', 'js_condition': 'val <= 30', 'row_class': 'row-amber'}]
    badge_columns   : dict of column_name → list of badge rules:
        {'Rate': [{'min': 10, 'max': 999, 'bg': '#FCEBEB', 'text': '#791F1F'}]}
    key             : unique identifier for the component
    """
    import pandas as pd
    rows_json      = json.dumps(df.to_dict(orient="records"), default=str)
    cols_json      = json.dumps(list(df.columns))
    highlight_json = json.dumps(highlight_rules or [])
    badge_json     = json.dumps(badge_columns   or {})

    html = f"""
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@500;600&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
.tbl-wrap {{
    overflow-y: auto;
    overflow-x: auto;
    height: {height}px;
    border: 0.5px solid rgba(128,128,128,0.18);
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    min-width: 500px;
}}
thead th {{
    position: sticky;
    top: 0;
    z-index: 2;
    background: #ffffff;
    font-family: 'Montserrat', sans-serif;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #888780;
    text-align: left;
    padding: 8px 12px;
    border-bottom: 0.5px solid rgba(128,128,128,0.18);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
}}
thead th:hover {{ color: #444441; }}
thead th .sort-icon {{ display:inline-block; margin-left:4px; opacity:0.4; font-size:10px; }}
thead th.sort-asc  .sort-icon::after {{ content: ' ↑'; opacity:1; }}
thead th.sort-desc .sort-icon::after {{ content: ' ↓'; opacity:1; }}
thead th:not(.sort-asc):not(.sort-desc) .sort-icon::after {{ content: ' ↕'; }}
tbody td {{
    padding: 9px 12px;
    border-bottom: 0.5px solid rgba(128,128,128,0.1);
    color: #2C2C2A;
    vertical-align: middle;
    line-height: 1.4;
}}
tbody tr:last-child td {{ border-bottom: none; }}
tbody tr:hover td {{ background: rgba(24,95,165,0.03) !important; }}
.badge {{
    display: inline-block;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 500;
    white-space: nowrap;
    font-family: 'Inter', sans-serif;
}}
.row-amber td {{ background: #FEF3E2 !important; }}
.row-blue  td {{ background: #EBF3FB !important; }}
.row-red   td {{ background: #FCEBEB !important; }}
.muted {{ color: #B4B2A9; }}
</style>

<div class="tbl-wrap" id="tw_{key}">
  <table id="t_{key}">
    <thead><tr id="hdr_{key}"></tr></thead>
    <tbody id="body_{key}"></tbody>
  </table>
</div>

<script>
(function() {{
    const cols       = {cols_json};
    const allRows    = {rows_json};
    const highlights = {highlight_json};
    const badges     = {badge_json};

    let sortCol  = null;
    let sortAsc  = true;
    let curRows  = [...allRows];

    const hdr = document.getElementById('hdr_{key}');
    cols.forEach((col, i) => {{
        const th = document.createElement('th');
        th.innerHTML = col + '<span class="sort-icon"></span>';
        th.addEventListener('click', () => {{
            if (sortCol === i) {{ sortAsc = !sortAsc; }}
            else {{ sortCol = i; sortAsc = true; }}
            hdr.querySelectorAll('th').forEach((h, j) => {{
                h.classList.remove('sort-asc', 'sort-desc');
                if (j === i) h.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
            }});
            sortAndRender();
        }});
        hdr.appendChild(th);
    }});

    function badgeHtml(col, val) {{
        if (!badges[col]) {{
            if (val === null || val === undefined || val === '') return '<span class="muted">—</span>';
            return String(val);
        }}
        const rules = badges[col];
        for (const r of rules) {{
            const num = parseFloat(val);
            if (!isNaN(num) && num >= r.min && num < r.max) {{
                const label = (r.label !== undefined && r.label !== null)
                    ? r.label
                    : (typeof val === 'number' ? val.toFixed(1) + '%' : String(val));
                return `<span class="badge" style="background:${{r.bg}};color:${{r.text}}">${{label}}</span>`;
            }}
        }}
        if (val === null || val === undefined || val === '') return '<span class="muted">—</span>';
        return String(val);
    }}

    function rowClass(row) {{
        for (const rule of highlights) {{
            const val = row[rule.column];
            if (val === undefined) continue;
            try {{
                const cond = rule.js_condition.replace(/\\bval\\b/g, JSON.stringify(val));
                if (eval(cond)) return rule.row_class || '';
            }} catch(e) {{}}
        }}
        return '';
    }}

    function renderBody(rows) {{
        const body = document.getElementById('body_{key}');
        body.innerHTML = '';
        rows.forEach(row => {{
            const tr = document.createElement('tr');
            const rc = rowClass(row);
            if (rc) tr.className = rc;
            cols.forEach(col => {{
                const td = document.createElement('td');
                const val = row[col];
                if (val === null || val === undefined || val === '') {{
                    td.innerHTML = '<span class="muted">—</span>';
                }} else {{
                    td.innerHTML = badgeHtml(col, val);
                }}
                tr.appendChild(td);
            }});
            body.appendChild(tr);
        }});
    }}

    function sortAndRender() {{
        if (sortCol === null) {{ renderBody(curRows); return; }}
        const col = cols[sortCol];
        curRows = [...curRows].sort((a, b) => {{
            const av = a[col], bv = b[col];
            if (av === null && bv === null) return 0;
            if (av === null) return 1;
            if (bv === null) return -1;
            const an = parseFloat(av), bn = parseFloat(bv);
            if (!isNaN(an) && !isNaN(bn)) return sortAsc ? an - bn : bn - an;
            return sortAsc
                ? String(av).localeCompare(String(bv))
                : String(bv).localeCompare(String(av));
        }});
        renderBody(curRows);
    }}

    renderBody(curRows);
}})();
</script>
"""
    _st_components.html(html, height=height + 20, scrolling=False)
