"""
afya_theme.py
=============
Afya Analytics · Plotly theme
Import this once at the top of any dashboard to apply the Afya brand template.

Usage:
    from afya_theme import AFYA_BLUE, TEAL, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY
    from afya_theme import CHART_LAYOUT, AXIS
    # Template is registered and set as default on import — no extra call needed.
"""

import plotly.io as pio

# ── PALETTE ───────────────────────────────────────────────────────────────────
AFYA_BLUE = "#0072CE"
TEAL      = "#0BB99F"
COOL_BLUE = "#003467"
ORANGE    = "#f5a623"
CORAL     = "#e05c5c"
PURPLE    = "#7b5ea7"
GRAY      = "#adb5bd"
MUTED     = "#0072CE"
BORDER    = "#D6E4F0"
BG_LIGHT  = "#F4F8FC"
SEQ       = [TEAL, AFYA_BLUE, COOL_BLUE, ORANGE, CORAL, PURPLE]

# ── PLOTLY TEMPLATE ───────────────────────────────────────────────────────────
pio.templates["afya"] = pio.templates["plotly_white"]

_t = pio.templates["afya"].layout

_t.font        = dict(family="Montserrat, sans-serif", color=AFYA_BLUE, size=11)
_t.legend.font = dict(family="Montserrat, sans-serif", color=AFYA_BLUE, size=10)

_t.xaxis.tickfont   = dict(color=AFYA_BLUE, size=10)
_t.xaxis.title.font = dict(color=AFYA_BLUE, size=11)
_t.yaxis.tickfont   = dict(color=AFYA_BLUE, size=10)
_t.yaxis.title.font = dict(color=AFYA_BLUE, size=11)
_t.xaxis.gridcolor  = "#EBF3FB"
_t.yaxis.gridcolor  = "#EBF3FB"

_t.paper_bgcolor = "#fff"
_t.plot_bgcolor  = "#fff"

pio.templates.default = "afya"

# ── SHARED LAYOUT DICT ────────────────────────────────────────────────────────
# Pass as **CHART_LAYOUT in every fig.update_layout() call
CHART_LAYOUT = dict(
    plot_bgcolor="#fff",
    paper_bgcolor="#fff",
    font=dict(family="Montserrat, sans-serif", size=11, color=AFYA_BLUE),
    margin=dict(t=10, b=10, l=0, r=10),
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right",  x=1,
        font=dict(family="Montserrat, sans-serif", size=10, color=AFYA_BLUE),
        title=dict(font=dict(family="Montserrat, sans-serif", size=10, color=AFYA_BLUE)),
        bgcolor="rgba(0,0,0,0)",
    ),
    colorway=SEQ,
)

# ── SHARED AXIS DICT ──────────────────────────────────────────────────────────
# Pass as **AXIS in every fig.update_xaxes() / fig.update_yaxes() call
AXIS = dict(
    showgrid=True,
    gridcolor="#EBF3FB",
    zeroline=False,
    color=AFYA_BLUE,
    tickfont=dict(color=AFYA_BLUE,  size=10, family="Montserrat, sans-serif"),
    title_font=dict(color=AFYA_BLUE, size=11, family="Montserrat, sans-serif"),
    title_standoff=8,
)