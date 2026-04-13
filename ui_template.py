"""
template_module.py

Starter file for any of our your module.
Copy this file, rename it (e.g. app_pricing.py, app_forecasting.py),
and build your content inside the tab sections.

You only need to touch:
    1. PAGE_TITLE — the name of your module
    2. The sidebar section — your filters and controls
    3. The tab definitions — rename/add tabs for your module
    4. The content inside each tab — your analysis goes here

Everything else (CSS, fonts, colors, KPI cards, layout) is handled for you.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ─── PAGE CONFIG — change PAGE_TITLE only ────────────────────────────────────

PAGE_TITLE = "Module Name"   # e.g. "New Venture Headstart" 

st.set_page_config(
    page_title=f"PharmaPlus · {PAGE_TITLE}",
    page_icon="⚕", #unnecessary
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SHARED THEME ────────────────────────────────────────────
# This is the same CSS used across all modules.
# Copy it exactly as-is into every module file.

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif;background:#fff;color:#003467}
.stApp{background:#fff}
[data-testid="stSidebar"]{background:#F4F8FC;border-right:1px solid #D6E4F0}
[data-testid="stSidebar"] *{color:#003467!important;font-family:'Montserrat',sans-serif!important}
.sh{font-size:10px;font-weight:800;color:#0072CE;text-transform:uppercase;
    letter-spacing:2.5px;padding:8px 0;border-bottom:2px solid #EBF3FB;margin-bottom:16px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
.stButton button{background:#0072CE!important;color:#fff!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;font-size:11px!important;font-weight:700!important;
  letter-spacing:1px!important;padding:8px 18px!important;border-radius:6px!important}
.stButton button:hover{background:#003467!important}
[data-baseweb="tab"]{font-family:'Montserrat',sans-serif!important;font-weight:600!important;
  color:#6B8CAE!important;font-size:12px!important}
[aria-selected="true"]{color:#0072CE!important;border-bottom-color:#0072CE!important}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:#B0C8E0;border-radius:10px}
</style>
""", unsafe_allow_html=True)

# ─── SHARED HELPERS — copy these into every module ────────────────────────────

def fmt_ksh(v):
    """Format a number as KSh with K/M suffix."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"KSh {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"KSh {v/1_000:.1f}K"
    return f"KSh {v:.0f}"


def kpi_card(label, value, sub="", color="#003467"):
    """
    Renders a KPI card matching the IBR module style.

    Usage:
        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Products tracked", "2,488", "across 5 categories", "#0072CE")
        with col2:
            kpi_card("Avg price gap", "18.4%", "vs nearest competitor", "#D97706")
        with col3:
            kpi_card("Priced above market", "312 SKUs", "needs review", "#E11D48")
    """
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;'
        f'border-radius:8px;padding:18px 16px">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text, margin_top=0):
    """Renders a section label matching the IBR module style."""
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(
        f'<div class="sh" style="{style}">{text}</div>',
        unsafe_allow_html=True,
    )


def info_card(text, border_color="#0072CE"):
    """
    A simple highlighted info strip — use for insights, alerts, or callouts.

    Usage:
        info_card("18 SKUs are priced more than 30% above Goodlife in this category.")
        info_card("Demand is accelerating in Two Rivers for supplements.", "#0BB99F")
    """
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;'
        f'border-left:3px solid {border_color};border-radius:4px;'
        f'font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True,
    )


# Plotly chart defaults — apply to every figure for visual consistency
CHART_LAYOUT = dict(
    paper_bgcolor="#fff",
    plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

# Brand colors for charts — use these instead of arbitrary colors
COLORS = {
    "primary":  "#0072CE",
    "success":  "#0BB99F",
    "warning":  "#D97706",
    "danger":   "#E11D48",
    "muted":    "#6B8CAE",
    "purple":   "#7F77DD",
    "pink":     "#D4537E",
    "coral":    "#D85A30",
    "green":    "#1D9E75",
}


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
# Put your module's filters and controls here.

with st.sidebar:
    try:
        st.image("assets/pharmaplus_logo.png", width=160)
    except:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#0072CE;'
            'padding:8px 0 16px">PharmaPlus</div>',
            unsafe_allow_html=True,
        )

    # ── Replace this section with your module's controls ──────────────────
    section_header("Filters")

    # Example filter — replace with what your module needs
    # category_filter = st.multiselect("Category", options=[...], default=[...])
    # date_range = st.date_input("Date range", ...)
    # branch_filter = st.multiselect("Branch", options=[...])

    st.info("Add your module's filters and controls here.")
    # ── End of your section ───────────────────────────────────────────────


# ─── PAGE HEADER ─────────────────────────────────────────────────────────────

st.markdown(
    f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    f'text-transform:uppercase;color:#0072CE;margin-bottom:16px">'
    f'PharmaPlus · {PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)

# ── KPI row — replace with your module's metrics ─────────────────────────────
# Pattern: I'm using only4 cards, always the same visual weight

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Metric one", "—", "description", COLORS["primary"])
with c2:
    kpi_card("Metric two", "—", "description", COLORS["success"])
with c3:
    kpi_card("Metric three", "—", "description", COLORS["warning"])
with c4:
    kpi_card("Metric four", "—", "description", COLORS["danger"])

st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────
# Rename these tabs to match your module's content.
# Keep to 3–4 tabs maximum.

tab1, tab2, tab3 = st.tabs([
    "◉  Main view",
    "△  Secondary view",
    "∑  Analytics",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — your main content
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    section_header("Your main section title here")

    # ── Your content goes here ────────────────────────────────────────────
    # Examples of what you can use:

    # A callout:
    # info_card("Key insight about your data goes here.", COLORS["success"])

    # A dataframe:
    # st.dataframe(your_df, hide_index=True, use_container_width=True)

    # A chart (apply CHART_LAYOUT for consistency):
    # fig = go.Figure(...)
    # fig.update_layout(**CHART_LAYOUT, height=300)
    # st.plotly_chart(fig, use_container_width=True)

    st.info("Your tab 1 content goes here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — secondary view
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    section_header("Your secondary section title here")
    st.info("Your tab 2 content goes here.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — analytics
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    section_header("Analytics")

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        section_header("Chart title")
        # fig = go.Figure(...)
        # fig.update_layout(**CHART_LAYOUT, height=300)
        # st.plotly_chart(fig, use_container_width=True)
        st.info("Left chart goes here.")

    with col_r:
        section_header("Chart title")
        st.info("Right chart goes here.")
