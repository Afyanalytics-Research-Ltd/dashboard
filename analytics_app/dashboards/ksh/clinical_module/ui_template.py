"""
ui_template.py — Afya Clinical Analytics · Plotly theme + shared UI helpers
============================================================================
Mirrors the XanaLife ui_template pattern.
Import at the top of every view file.
"""

import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st

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

# ── PLOTLY TEMPLATE ───────────────────────────────────────────────────────────
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


# ── SHARED LAYOUT ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    plot_bgcolor  = "#fff",
    paper_bgcolor = "#fff",
    font          = dict(family="Montserrat, sans-serif", size=11, color=COOL_BLUE),
    margin        = dict(t=10, b=10, l=0, r=10),
    legend        = dict(
        orientation = "h",
        yanchor     = "bottom", y=1.02,
        xanchor     = "right",  x=1,
        font        = dict(family="Montserrat, sans-serif", size=10, color=COOL_BLUE),
        bgcolor     = "rgba(0,0,0,0)",
    ),
    colorway = SEQ,
)

AXIS = dict(
    showgrid     = True,
    gridcolor    = "#EBF3FB",
    zeroline     = False,
    color        = COOL_BLUE,
    tickfont     = dict(color=MUTED,     size=10, family="Montserrat, sans-serif"),
    title_font   = dict(color=COOL_BLUE, size=11, family="Montserrat, sans-serif"),
    title_standoff = 8,
)


# ── HELPER: axis dict merge ────────────────────────────────────────────────────
def _ax(**overrides):
    return {**AXIS, **overrides}


# ── KPI CARD ──────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str = "", delta: str = "",
             delta_color: str = MUTED, color: str = COOL_BLUE):
    delta_html = (
        f'<div style="font-size:11px;color:{delta_color};margin-top:3px">{delta}</div>'
        if delta else ""
    )
    sub_html = (
        f'<div style="font-size:10px;color:{MUTED};margin-top:2px">{sub}</div>'
        if sub else ""
    )
    st.markdown(
        f'<div style="background:#fff;border:1px solid {BORDER};border-radius:8px;padding:14px 14px 10px">'
        f'<div style="font-size:10px;font-weight:600;color:{MUTED};text-transform:uppercase;'
        f'letter-spacing:1.2px;margin-bottom:5px">{label}</div>'
        f'<div style="font-size:22px;font-weight:700;color:{color};line-height:1.1">{value}</div>'
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
