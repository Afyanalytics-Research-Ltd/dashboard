import streamlit as st

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

