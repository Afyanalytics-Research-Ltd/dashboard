"""
Shared design system for the XanaLife dashboard.
"""

import os
import streamlit as st
import numpy as np

# ── Brand colors ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":  "#0072CE",
    "success":  "#313D3B",
    "warning":  "#D97706",
    "danger":   "#E11D48",
    "muted":    "#6B8CAE",
    "purple":   "#7F77DD",
    "pink":     "#D4537E",
    "coral":    "#D85A30",
    "green":    "#1D9E75",
}

STATUS_COLORS = {
    "Stockout":       "#E11D48",
    "Critical":       "#f97316",
    "Warning":        "#D97706",
    "Monitor":        "#eab308",
    "Healthy":        "#0BB99F",
    "Overstocked":    "#0072CE",
    "No demand data": "#6B8CAE",
}

# ── Plotly defaults ────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, sans-serif", size=11, color="#003467"),
    margin=dict(l=0, r=0, t=24, b=10),
    xaxis=dict(
        gridcolor="#EBF3FB", linecolor="#D6E4F0",
        tickfont=dict(size=10, color="#003467"),
        title_font=dict(size=11, color="#003467"),
    ),
    yaxis=dict(
        gridcolor="#EBF3FB", linecolor="#D6E4F0",
        tickfont=dict(size=10, color="#003467"),
        title_font=dict(size=11, color="#003467"),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#003467"),
    ),
)

# ── CSS ────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #fff;
    color: #003467;
}
.stApp { background: #fff; }

/* ── Hide Streamlit chrome ── */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
header     { visibility: hidden; }

/* ── Hide auto-generated sidebar nav so we render our own ── */
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar base ── */
[data-testid="stSidebar"] {
    background: #F4F8FC;
    border-right: 1px solid #D6E4F0;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}
[data-testid="stSidebar"] * {
    color: #003467 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Sidebar filter labels ── */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stTextInput label {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: #6B8CAE !important;
}

/* ── Page links (navigation) ── */
[data-testid="stPageLink"] a {
    text-decoration: none !important;
    color: #003467 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 10px !important;
    border-radius: 6px !important;
    display: block !important;
    background: transparent !important;
    transition: background 0.15s !important;
}
[data-testid="stPageLink"] a:hover {
    background: #EBF3FB !important;
    color: #0072CE !important;
}
[data-testid="stPageLink-active"] a {
    background: #EBF3FB !important;
    color: #0072CE !important;
    font-weight: 700 !important;
}
[data-testid="stPageLink"] p {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #003467 !important;
    margin: 0 !important;
}

/* ── Section headers ── */
.sh {
    font-size: 10px;
    font-weight: 700;
    color: #0072CE;
    text-transform: uppercase;
    letter-spacing: 2px;
    padding: 6px 0;
    border-bottom: 1px solid #D6E4F0;
    margin-bottom: 14px;
}

/* ── Buttons ── */
.stButton button {
    background: #0072CE !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 9px 22px !important;
    border-radius: 6px !important;
    transition: background 0.15s !important;
}
.stButton button:hover { background: #005fad !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: #F4F8FC !important;
    color: #0072CE !important;
    border: 1px solid #B0C8E0 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 7px 20px !important;
    border-radius: 6px !important;
}
[data-testid="stDownloadButton"] button:hover { background: #EBF3FB !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: #F4F8FC;
    border: 1px solid #D6E4F0;
    padding: 4px;
    border-radius: 8px;
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    color: #6B8CAE !important;
    border-radius: 6px !important;
    padding: 7px 20px !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #003467 !important;
    background: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 1px 4px rgba(0,52,103,0.10) !important;
}

/* ── Dataframe column headers ── */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] [role="columnheader"],
div[class*="glideDataEditor"] .headerRow .headerCell,
.dvn-stack [role="columnheader"] {
    background: #EBF3FB !important;
    color: #003467 !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    border-bottom: 2px solid #B0C8E0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: #B0C8E0; border-radius: 10px; }

/* ── Hide native metric widget ── */
div[data-testid="metric-container"] { display: none; }
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_nav():
    """Logo + navigation links — call once at top of every page's sidebar block."""
    # ── Logo ──
    logo_path = "Logo.png"
    if os.path.exists(logo_path):
        st.markdown('<div style="padding:20px 16px 12px">', unsafe_allow_html=True)
        st.image(logo_path, width=148)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="padding:20px 16px 12px">'
            '<span style="font-size:18px;font-weight:800;color:#0072CE">XanaLife</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Nav divider ──
    st.markdown(
        '<div style="border-bottom:1px solid #D6E4F0;margin:0 0 12px"></div>',
        unsafe_allow_html=True,
    )

    # ── Navigation links ──
    st.markdown(
        '<div style="font-size:9px;font-weight:700;color:#8AABCC;text-transform:uppercase;'
        'letter-spacing:1.5px;padding:0 4px;margin-bottom:6px">Menu</div>',
        unsafe_allow_html=True,
    )
    st.page_link("xanalife/cross_sell_inventory_dashboard.py",
                 label="Home", icon="🏠")
    st.page_link("xanalife/1_CSUC.py",
                 label="Cross-Sell Intelligence", icon="📊")
    st.page_link("xanalife/3_Stockout_Prediction.py",
                 label="Inventory Risk", icon="📦")
    st.markdown(
        '<div style="border-bottom:1px solid #D6E4F0;margin:12px 0 16px"></div>',
        unsafe_allow_html=True,
    )


def page_banner(title: str, subtitle: str, tag: str = ""):
    """Clean top-of-page header."""
    tag_html = (
        f'<span style="display:inline-block;background:#EBF3FB;color:#0072CE;'
        f'font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.5px;'
        f'padding:3px 10px;border-radius:4px;margin-bottom:12px">{tag}</span><br>'
        if tag else ""
    )
    st.markdown(
        f'{tag_html}'
        f'<h1 style="font-size:24px;font-weight:800;color:#003467;margin:0 0 8px;line-height:1.25">'
        f'{title}</h1>'
        f'<p style="font-size:13px;color:#6B8CAE;margin:0 0 24px;line-height:1.6;max-width:740px">'
        f'{subtitle}</p>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", color: str = "#003467"):
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;'
        f'border-top:3px solid {color};border-radius:8px;padding:18px 16px 14px">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:11px;color:#8AABCC;margin-top:6px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str, margin_top: int = 24):
    st.markdown(
        f'<div class="sh" style="margin-top:{margin_top}px">{text}</div>',
        unsafe_allow_html=True,
    )


def info_card(text: str, border_color: str = "#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;'
        f'border-left:3px solid {border_color};border-radius:4px;'
        f'font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True,
    )


def page_header(module_name: str):
    """Legacy — kept for compatibility."""
    st.markdown(
        f'<p style="font-size:10px;font-weight:700;letter-spacing:2px;'
        f'text-transform:uppercase;color:#0072CE;margin-bottom:4px">'
        f'XanaLife &nbsp;·&nbsp; {module_name}</p>',
        unsafe_allow_html=True,
    )


def fmt_kes(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"KES {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"KES {v/1_000:.1f}K"
    return f"KES {v:.0f}"
