import os
import numpy as np
import pandas as pd
import streamlit as st

_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")

COLORS = {
    "primary": "#0072CE",
    "success": "#0BB99F",
    "warning": "#D97706",
    "danger":  "#E11D48",
    "muted":   "#6B8CAE",
    "dark":    "#003467",
    "purple":  "#7F77DD",
    "coral":   "#D85A30",
    "green":   "#1D9E75",
}

CHART_LAYOUT = dict(
    paper_bgcolor="#fff",
    plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

_CSS = """
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
    background: #fff;
    color: #003467;
}
.stApp { background: #fff; }

[data-testid="stSidebar"] {
    background: #F0F5FA !important;
    border-right: 1px solid #D6E4F0 !important;
}
[data-testid="stSidebar"] *:not(i) {
    font-family: 'Montserrat', sans-serif !important;
}

[data-testid="stSidebarNav"] { display: none !important; }

[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {
    font-family: 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga';
}

[data-testid="stSidebar"] a:hover {
    background: #DDE9F5 !important;
    text-decoration: none !important;
}

.sb-label {
    font-size: 9px;
    font-weight: 700;
    color: #8BAAC5;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    padding: 0 4px;
    margin-bottom: 4px;
}

.sh {
    font-size: 10px;
    font-weight: 800;
    color: #0072CE;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    padding: 8px 0;
    border-bottom: 2px solid #EBF3FB;
    margin-bottom: 16px;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
}
.stButton button {
    background: #0072CE !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 8px 18px !important;
    border-radius: 6px !important;
}
.stButton button:hover { background: #003467 !important; }

[data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    color: #6B8CAE !important;
    font-size: 12px !important;
}
[aria-selected="true"] {
    color: #0072CE !important;
    border-bottom-color: #0072CE !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #B0C8E0; border-radius: 10px; }
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


_NAV = [
    ("overview",   "fa-solid fa-house",                     "Overview",       "/"),
    (None,         None,                                     "Camp Report",    None),
    ("conditions", "fa-solid fa-stethoscope",                "Disease Burden", "/Disease_Burden"),
    ("response",   "fa-solid fa-kit-medical",                "Camp Response",  "/Camp_Response"),
    ("priorities", "fa-solid fa-triangle-exclamation",      "Priorities",     "/Priorities"),
]


def render_sidebar(active: str = "overview"):
    with st.sidebar:
        if os.path.exists(_LOGO_PATH):
            logo_col = st.columns([1, 2, 1])
            with logo_col[1]:
                st.image(_LOGO_PATH, use_container_width=True)

        st.markdown(
            '<div style="text-align:center;padding:10px 0 16px">'
            '<div style="font-size:9px;font-weight:700;color:#8BAAC5;'
            'text-transform:uppercase;letter-spacing:2px;margin-bottom:5px">'
            'Mother Francisca Mission</div>'
            '<div style="font-size:17px;font-weight:800;color:#003467;line-height:1.2">'
            'Camp Analytics</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-label" style="margin-bottom:6px">Navigate</div>',
                    unsafe_allow_html=True)

        items_html = '<div style="display:flex;flex-direction:column;gap:2px;margin-bottom:4px">'
        for page_id, icon, label, href in _NAV:
            if page_id is None:
                items_html += (
                    f'<div style="margin:10px 0 4px;padding:0 4px;font-size:9px;font-weight:700;'
                    f'color:#8BAAC5;text-transform:uppercase;letter-spacing:1.8px;'
                    f'border-top:1px solid #D6E4F0;padding-top:10px">{label}</div>'
                )
                continue
            is_active   = page_id == active
            is_disabled = href is None
            icon_color  = "#fff" if is_active else ("#8BAAC5" if is_disabled else "#0072CE")
            bg          = "background:#003467;" if is_active else ""
            txt_color   = "#fff" if is_active else ("#8BAAC5" if is_disabled else "#1E3A55")
            weight      = "800" if is_active else "600"
            opacity     = "opacity:0.45;" if is_disabled else ""
            cursor      = "cursor:default;" if (is_active or is_disabled) else "cursor:pointer;"

            inner = (
                f'<i class="{icon}" style="width:18px;text-align:center;'
                f'font-size:14px;color:{icon_color};flex-shrink:0"></i>'
                f'<span style="font-size:13px;font-weight:{weight};color:{txt_color}">{label}</span>'
            )

            row_style = (
                f'display:flex;align-items:center;gap:12px;padding:10px 14px;'
                f'border-radius:8px;text-decoration:none;{bg}{opacity}{cursor}'
            )

            if is_active or is_disabled:
                items_html += f'<div style="{row_style}">{inner}</div>'
            else:
                items_html += (
                    f'<a href="{href}" target="_self" style="{row_style}">'
                    f'{inner}</a>'
                )

        items_html += '</div>'
        st.markdown(items_html, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
        st.divider()
        st.markdown(
            '<div class="sb-label">Data Scope</div>'
            '<div style="font-size:10px;color:#6B8CAE;line-height:1.85;padding:2px 4px 0">'
            '<b style="color:#003467">Period</b> Jul – Sep 2026 (provisional)<br>'
            '<b style="color:#003467">Source</b> 796 PDFs · 5,135 pages<br>'
            '<b style="color:#003467">Encounters</b> 2,575 blocks<br>'
            '<b style="color:#003467">Linked records</b> 2,447'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown(
            '<div class="sb-label">Abbreviations</div>'
            '<div style="font-size:10px;color:#6B8CAE;line-height:1.85;padding:2px 4px 0">'
            '<b style="color:#003467">MF</b> — Mother Francisca Mission Maternity<br>'
            '<b style="color:#003467">OCR</b> — Optical Character Recognition<br>'
            '<b style="color:#003467">Encounter block</b> — one clinical encounter from a PDF page<br>'
            '<b style="color:#003467">Linked record</b> — deduped via probabilistic name match<br>'
            '<b style="color:#003467">ICD-10</b> — International Classification of Diseases, 10th revision<br>'
            '<b style="color:#003467">OTHER</b> — residual; garbled or unclassifiable OCR terms'
            '</div>',
            unsafe_allow_html=True,
        )


def cl(**kw):
    return {**CHART_LAYOUT, **kw}


def kpi_card(label, value, sub="", color="#003467", icon=""):
    _accent = {COLORS["danger"], COLORS["warning"], COLORS["success"]}
    bl = f"border-left:4px solid {color};" if color in _accent else ""
    icon_html = f'<span style="font-size:13px;margin-right:5px">{icon}</span>' if icon else ""
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
        f'padding:24px 20px;{bl}">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">{icon_html}{label}</div>'
        f'<div style="font-size:42px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:12px;color:#6B8CAE;margin-top:8px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)


def info_card(text, border_color="#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;border-left:3px solid {border_color};'
        f'border-radius:4px;font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True,
    )


def page_header(title, subtitle=None, center=False):
    _align = "center" if center else "left"
    sub_html = (
        f'<div style="font-size:12px;color:#6B8CAE;margin-top:4px;text-align:{_align}">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:8px;text-align:{_align}">'
        f'  <div style="font-size:9px;font-weight:700;color:#6B8CAE;'
        f'       text-transform:uppercase;letter-spacing:2px">'
        f'    Mother Francisca Mission · Nandi County</div>'
        f'  <div style="font-size:24px;font-weight:800;color:#003467;margin-top:4px">{title}</div>'
        f'  {sub_html}'
        f'</div>'
        f'<div style="border-bottom:1px solid #EBF3FB;margin:16px 0 24px"></div>',
        unsafe_allow_html=True,
    )


def insight_panel(title, items, footer=None, border_color=None):
    if border_color is None:
        border_color = COLORS["primary"]
    _items_html = ""
    for item in items:
        if isinstance(item, tuple):
            text, source = item
            _items_html += (
                f'<li style="margin-bottom:10px;display:flex;justify-content:space-between;'
                f'align-items:flex-start;gap:12px">'
                f'  <span style="color:#003467">{text}</span>'
                f'  <span style="font-size:10px;color:#9BAEC8;white-space:nowrap;'
                f'       margin-top:2px;flex-shrink:0">→ {source}</span>'
                f'</li>'
            )
        else:
            _items_html += f'<li style="margin-bottom:10px;color:#003467">{item}</li>'
    _footer_html = (
        f'<div style="font-size:11px;color:#6B8CAE;border-top:1px solid #EBF3FB;'
        f'padding-top:10px;margin-top:4px;font-style:italic">{footer}</div>'
        if footer else ""
    )
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;'
        f'border-left:3px solid {border_color};border-radius:6px;'
        f'padding:18px 20px;margin:16px 0">'
        f'  <div style="font-size:10px;font-weight:800;color:{border_color};'
        f'       text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">'
        f'    {title}</div>'
        f'  <ul style="margin:0;padding-left:16px;font-size:13px;line-height:1.7">'
        f'  {_items_html}</ul>'
        f'  {_footer_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def dq_note(text):
    st.markdown(
        f'<div style="background:#F4F8FC;border-left:3px solid #B0C8E0;border-radius:4px;'
        f'padding:8px 12px;margin:10px 0;font-size:12px;color:#003467;line-height:1.5">'
        f'<span style="font-weight:700;color:#6B8CAE">Data note · </span>{text}</div>',
        unsafe_allow_html=True,
    )


def _add_rolling_mean(fig, x_series, y_series, n=3, name="3-mo avg", color=None, dash="dot"):
    if color is None:
        color = COLORS["muted"]
    roll = pd.Series(
        y_series.values if hasattr(y_series, "values") else y_series
    ).rolling(n, min_periods=2).mean()
    fig.add_scatter(
        x=x_series, y=roll,
        mode="lines", name=name,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"<b>{name}</b>: %{{y:.1f}}<extra></extra>",
    )
