"""
Revenue Leakage — pages/6_leakage.py
Source: mart_revenue_leakage · V2 only · Feb 2025 – present
Grain: one row per procedure request (clinical procedures only).

Decision question: Where are clinical procedures being performed but not collected —
and which procedure types are driving it?

Story: Leakage is concentrated (Physio ~73%), not distributed. Fix is targeted billing
workflow intervention at specific procedure desks, not a general process change.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Revenue Leakage · SPH Ortho",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.theme import (
    apply_theme, render_sidebar, kpi_card, section_header, info_card,
    page_header, insight_panel, COLORS, cl,
)
from dashboard.queries import (
    q_leakage_summary, q_leakage_by_procedure, q_leakage_prev_month,
)

apply_theme()
render_sidebar("leakage")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _act(label):
    st.markdown(
        f'<div style="border-top:1.5px solid #D6E4F0;margin:40px 0 20px 0;padding-top:14px">'
        f'<span style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:2.5px">{label}</span></div>',
        unsafe_allow_html=True,
    )


def _caption(text):
    st.markdown(
        f'<div style="font-size:9px;color:#9BAEC8;margin-top:-6px;margin-bottom:4px">{text}</div>',
        unsafe_allow_html=True,
    )


def _fmt_kes(val):
    """Format KES value: M for millions, K for thousands, raw otherwise."""
    if val is None:
        return "—"
    if val >= 1_000_000:
        return f"KES {val / 1_000_000:.2f}M"
    if val >= 1_000:
        return f"KES {val / 1_000:.0f}K"
    return f"KES {val:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

summary_df  = q_leakage_summary()
proc_df     = q_leakage_by_procedure()
prev_mo_df  = q_leakage_prev_month()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════

_s = summary_df.iloc[0]
_data_from = str(_s["DATA_FROM"])[:7] if _s["DATA_FROM"] else "Feb 2025"
_data_to   = str(_s["DATA_TO"])[:7]   if _s["DATA_TO"]   else "present"
_period    = f"{_data_from} – {_data_to}"

page_header(
    "Where Are We Losing Revenue?",
    subtitle="Clinical procedure requests with outstanding balances",
    period=_period,
    mode="snapshot",
)

info_card(
    "V2 data only · Clinical procedures only (Physiotherapy, Cannulation, Minor Dressing, "
    "K-Wire Removal, etc.). Billing fee rows (Theatre Fee, Surgeon Fees, Anaesthetist Fee) "
    "are excluded — those are administrative charges with a different collection channel. "
    "Leakage = procedure requested + unit price on record + not yet collected. "
    "This mart does not confirm procedure completion — a requested-but-uncollected procedure "
    "may reflect a billing gap, a cancellation not recorded, or a collection timing lag."
)


# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════

_total_clinical  = int(_s["TOTAL_CLINICAL"])
_leakage_count   = int(_s["LEAKAGE_COUNT"])
_leakage_rate    = float(_s["LEAKAGE_RATE_PCT"])
_collection_rate = float(_s["COLLECTION_RATE_PCT"])
_uncollected_kes = float(_s["TOTAL_UNCOLLECTED_KES"])

_collection_color = (
    COLORS["success"] if _collection_rate >= 90
    else COLORS["warning"] if _collection_rate >= 80
    else COLORS["danger"]
)

# prev-month KPI values
_pm = prev_mo_df.iloc[0] if not prev_mo_df.empty else None
_pm_label        = str(_pm["MONTH_LABEL"])   if _pm is not None else "—"
_pm_kes          = float(_pm["UNCOLLECTED_KES"]) if _pm is not None else 0.0
_pm_leakage_rate = float(_pm["LEAKAGE_RATE_PCT"]) if _pm is not None else 0.0
_pm_requests     = int(_pm["CLINICAL_REQUESTS"]) if _pm is not None else 0
_pm_color = (
    COLORS["danger"]  if _pm_leakage_rate >= 20
    else COLORS["warning"] if _pm_leakage_rate >= 10
    else COLORS["success"]
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card(
        "Observed Uncollected Value",
        _fmt_kes(_uncollected_kes),
        f"{_leakage_count:,} procedure requests · {_period}",
        color=COLORS["danger"],
    )
with k2:
    kpi_card(
        "Clinical Leakage Rate",
        f"{_leakage_rate:.1f}%",
        f"of {_total_clinical:,} clinical procedure requests unpaid",
        color=COLORS["danger"] if _leakage_rate >= 20 else COLORS["warning"],
    )
with k3:
    kpi_card(
        "Collection Rate",
        f"{_collection_rate:.1f}%",
        "of clinical procedures collected at request",
        color=_collection_color,
    )
with k4:
    kpi_card(
        "Sessions Affected",
        f"{_leakage_count:,}",
        f"procedure requests with outstanding balance · {_period}",
        color=COLORS["muted"],
    )
with k5:
    kpi_card(
        f"Last Complete Month ({_pm_label})",
        _fmt_kes(_pm_kes),
        f"{_pm_leakage_rate:.1f}% leakage rate · {_pm_requests:,} requests",
        color=_pm_color,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LEAKAGE BY PROCEDURE TYPE + PARETO CONCENTRATION
# ══════════════════════════════════════════════════════════════════════════════

_act("Leakage by Procedure Type")
_caption(
    "Clinical procedures only. Sorted by observed uncollected value (KES). "
    "Leakage % = uncollected ÷ total requests per procedure type."
)

col_bars, col_pareto = st.columns([3, 2])

with col_bars:
    if not proc_df.empty:
        pd_plot = proc_df[proc_df["UNCOLLECTED_KES"] > 0].copy()
        pd_plot = pd_plot.nlargest(10, "UNCOLLECTED_KES")
        pd_plot = pd_plot.sort_values("UNCOLLECTED_KES", ascending=True)

        _bar_colors = []
        for _, row in pd_plot.iterrows():
            if row["SHARE_OF_TOTAL_PCT"] >= 50:
                _bar_colors.append(COLORS["danger"])
            elif row["LEAKAGE_PCT"] >= 80:
                _bar_colors.append(COLORS["coral"])
            else:
                _bar_colors.append(COLORS["warning"])

        _custom_text = [
            f"{_fmt_kes(r['UNCOLLECTED_KES'])} · {int(r['LEAKAGE_COUNT'])} sessions · {r['LEAKAGE_PCT']:.0f}% leakage"
            for _, r in pd_plot.iterrows()
        ]

        fig_proc = go.Figure(go.Bar(
            y=pd_plot["REQUEST_NAME"].tolist(),
            x=pd_plot["UNCOLLECTED_KES"].tolist(),
            orientation="h",
            marker_color=_bar_colors,
            customdata=list(zip(
                pd_plot["LEAKAGE_COUNT"].tolist(),
                pd_plot["LEAKAGE_PCT"].tolist(),
                pd_plot["TOTAL_REQUESTS"].tolist(),
                pd_plot["SHARE_OF_TOTAL_PCT"].tolist(),
            )),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Uncollected: KES %{x:,.0f}<br>"
                "Sessions uncollected: %{customdata[0]}<br>"
                "Leakage rate: %{customdata[1]:.1f}%<br>"
                "Total requests: %{customdata[2]}<br>"
                "Share of total leakage: %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
            text=[_fmt_kes(v) for v in pd_plot["UNCOLLECTED_KES"].tolist()],
            textposition="outside",
            textfont=dict(size=10, color="#003467"),
        ))
        fig_proc.update_layout(**cl(
            height=max(200, len(pd_plot) * 48),
            showlegend=False,
            xaxis=dict(title="Uncollected Value (KES)", gridcolor="#EBF3FB"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=80, t=10, b=20),
        ))
        st.plotly_chart(fig_proc, use_container_width=True)

with col_pareto:
    if not proc_df.empty:
        # Top 3 procedures + Other
        top3    = proc_df[proc_df["UNCOLLECTED_KES"] > 0].head(3).copy()
        other_share = max(0.0, 100.0 - float(top3["SHARE_OF_TOTAL_PCT"].sum()))

        pareto_labels = top3["REQUEST_NAME"].tolist()
        pareto_shares = top3["SHARE_OF_TOTAL_PCT"].tolist()
        if other_share > 0.5:
            pareto_labels.append("Other")
            pareto_shares.append(round(other_share, 1))

        _pareto_colors = [
            COLORS["danger"],
            COLORS["coral"],
            COLORS["warning"],
            COLORS["muted"],
        ][:len(pareto_labels)]

        fig_pareto = go.Figure(go.Bar(
            y=pareto_labels[::-1],
            x=pareto_shares[::-1],
            orientation="h",
            marker_color=_pareto_colors[::-1],
            text=[f"{v:.0f}%" for v in pareto_shares[::-1]],
            textposition="outside",
            textfont=dict(size=11, color="#003467"),
            hovertemplate="<b>%{y}</b>: %{x:.1f}% of total leakage<extra></extra>",
        ))
        fig_pareto.update_layout(**cl(
            height=max(200, len(pareto_labels) * 56),
            showlegend=False,
            title=dict(
                text="Share of Total Leakage",
                font=dict(size=10, color="#6B8CAE"),
                x=0,
            ),
            xaxis=dict(title="% of total uncollected value", range=[0, 110], gridcolor="#EBF3FB"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=50, t=30, b=20),
        ))
        st.plotly_chart(fig_pareto, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# INSIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
insight_panel(
    "Operational implication",
    [
        "Physiotherapy non-collection is systemic — one procedure type accounts for the majority "
        "of leakage with near-zero collection. The fix is a targeted billing workflow change at "
        "the physio desk, not a general process intervention.",
        "K-Wire Removal shows 100% non-collection across all sessions recorded. This pattern "
        "suggests a billing setup gap — the procedure may not have an active billing step attached "
        "in the system, not a patient payment failure.",
        "TAT correlation is flat — patients who waited longer did not leave more uncollected "
        "procedures. Leakage is not a downstream effect of consult delays. The problem is at the "
        "billing and collection step for specific procedure types.",
        "All clinical leakage is OPD-associated. Inpatient procedures show zero leakage because "
        "they are bundled through a different billing channel. Ward-level analysis is not applicable.",
    ],
    footer=f"Source: mart_revenue_leakage · V2 only · {_period}",
    border_color=COLORS["danger"],
)
