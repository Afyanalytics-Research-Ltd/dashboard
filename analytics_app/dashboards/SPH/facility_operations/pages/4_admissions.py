"""
Admissions & Theatre — pages/4_admissions.py
Tabs: Theatre · OPD->IPD Conversion

Theatre: rpt_ortho_theatre · V1+V2 · 2,341 procedures total
  Central question: Is theatre access constrained by demand, scheduling, or execution?
  V1: 749 procedures · Jun 2022–Jun 2023 · procedure_name (via SINGLEORDERITEMS) + duration
  V2: 1,592 procedures · Feb 2025–Jun 2026 · procedure_name + elective flag + same-day wait
  Same-day wait (request→service): 64.8% V2 coverage · elective P50=25 min · P90=339 min (capped 480 min, Inv 150)
  Scheduling lag (request→planned): 97.7% zeros — dead metric (Inv 144)
  Room utilisation: 84.8% room field missing — not feasible (Inv 146)

OPD->IPD Conversion: rpt_ortho_conversion · V2 only (Feb 2025+)
  2,324 admissions · 87.2% OPD-triggered · 5 wards · Feb 2025 – Jun 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Admissions & Theatre · SPH Ortho",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import (
    apply_theme, render_sidebar, kpi_card, section_header, info_card,
    page_header, COLORS, cl,
)

from facility_operations.dashboard.queries import (
    q_theatre_summary, q_theatre_monthly, q_theatre_elective,
    q_theatre_access_kpis, q_theatre_overdue, q_theatre_access_distribution,
    q_theatre_v1_procedure_hours, q_theatre_v2_procedure_wait,
    q_theatre_emergency_kpis, q_theatre_emergency_monthly,
    q_conv_v2_summary, q_conv_v2_monthly, q_conv_v2_ward,
    q_conv_v2_type, q_conv_v2_dow,
)

apply_theme()
render_sidebar("admissions")

# ── Header ─────────────────────────────────────────────────────────────────────

page_header("Admissions & Theatre", subtitle="Theatre · OPD → IPD Conversion")

# ── Load data ──────────────────────────────────────────────────────────────────

try:
    theatre_kpis     = q_theatre_summary().iloc[0]
    monthly_df       = q_theatre_monthly()
    elective_df      = q_theatre_elective()
    access_kpis_df   = q_theatre_access_kpis()
    overdue_df       = q_theatre_overdue()
    access_dist_df   = q_theatre_access_distribution()
    v1_proc_hrs_df   = q_theatre_v1_procedure_hours()
    v2_proc_wait_df  = q_theatre_v2_procedure_wait()
    emrg_kpis_df     = q_theatre_emergency_kpis()
    emrg_monthly_df  = q_theatre_emergency_monthly()
    conv_kpis        = q_conv_v2_summary().iloc[0]
    conv_monthly_df  = q_conv_v2_monthly()
    conv_ward_df     = q_conv_v2_ward()
    conv_type_df     = q_conv_v2_type()
    conv_dow_df      = q_conv_v2_dow()
except Exception as e:
    st.error(f"Failed to load data. Check Snowflake connection.\n\n{e}")
    st.stop()

# ── Theatre helpers ────────────────────────────────────────────────────────────

def _safe_kpi(df, col):
    """Return int value from a single-row DataFrame, or None if missing."""
    if df.empty or col not in df.columns:
        return None
    v = df.iloc[0][col]
    return int(v) if v is not None and str(v) != "nan" else None

_ak = access_kpis_df  # alias
_p50      = _safe_kpi(_ak, "P50_MINS")
_p90      = _safe_kpi(_ak, "P90_MINS")
_pct_4h      = None if _ak.empty else round(float(_ak.iloc[0]["PCT_OVER_4HRS"]), 1)
_overdue     = _safe_kpi(overdue_df, "OVERDUE_COUNT") or 0
_over_4hrs_n = None if _ak.empty or "OVER_4HRS" not in _ak.columns else int(_ak.iloc[0]["OVER_4HRS"])
_over_8hrs_n = (
    int(access_dist_df.loc[access_dist_df["WAIT_BUCKET"] == "> 8 hrs", "CASES"].iloc[0])
    if not access_dist_df.empty and "> 8 hrs" in access_dist_df["WAIT_BUCKET"].values
    else None
)

_elec_row  = elective_df[elective_df["CASE_TYPE"] == "Elective"].iloc[0]  \
             if not elective_df.empty and "Elective" in elective_df["CASE_TYPE"].values else None
_emrg_row  = elective_df[elective_df["CASE_TYPE"] == "Emergency"].iloc[0] \
             if not elective_df.empty and "Emergency" in elective_df["CASE_TYPE"].values else None
_elec_pct  = round(float(_elec_row["PCT"]), 0) if _elec_row is not None else None
_emrg_n    = int(_emrg_row["PROCEDURES"])       if _emrg_row is not None else None

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_theatre, tab_conversion = st.tabs(["◉  Theatre", "△  OPD → IPD"])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Theatre
# ══════════════════════════════════════════════════════════════════════════════

with tab_theatre:

    info_card(
        "<b>Is theatre access constrained by demand, scheduling, or execution?</b> &nbsp;"
        "V1: Jun 2022–Jun 2023 (749 procedures) — procedure name + duration recorded. "
        "V2: Feb 2025–Jun 2026 (1,592 procedures) — procedure name + same-day wait recorded. "
        "20-month gap between systems. V1 and V2 are presented separately where data differs.",
        border_color=COLORS["muted"],
    )

    # ── Section 1: Demand ─────────────────────────────────────────────────────

    section_header("1  Demand — How much theatre activity is there?", margin_top=20)

    _data_from = pd.to_datetime(theatre_kpis["DATA_FROM"]).strftime("%b %Y")
    _data_to   = pd.to_datetime(theatre_kpis["DATA_TO"]).strftime("%b %Y")

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card(
            "Theatre Procedures",
            f"{int(theatre_kpis['TOTAL_PROCEDURES']):,}",
            f"V1 + V2 · {_data_from} – {_data_to}",
        )
    with k2:
        kpi_card(
            "Elective",
            f"{_elec_pct:.0f}%" if _elec_pct is not None else "—",
            "of V2 cases · 96% is_elective coverage",
            color=COLORS["primary"],
        )
    with k3:
        kpi_card(
            "Emergency",
            f"{_emrg_n:,}" if _emrg_n is not None else "—",
            "V2 cases · 2% of caseload · near-immediate access",
            color=COLORS["success"],
        )

    st.markdown("<br>", unsafe_allow_html=True)

    _elec_finding = (
        f"<b>98% of theatre activity is elective. Emergency demand is not driving delays.</b> "
        f"Of {int(theatre_kpis['TOTAL_PROCEDURES']):,} total procedures, "
        f"only {_emrg_n or 31} ({100 - (_elec_pct or 98):.0f}%) are emergency — "
        f"near-immediate access once booked. The access challenge is in the elective pathway."
    )
    info_card(_elec_finding, border_color=COLORS["primary"])

    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px;margin-top:12px">'
        'Monthly Procedure Volume — V1 (blue) + V2 (teal)</div>',
        unsafe_allow_html=True,
    )
    if not monthly_df.empty:
        tm = monthly_df.copy()
        tm["PROCEDURE_MONTH"] = pd.to_datetime(tm["PROCEDURE_MONTH"])
        tm = tm.sort_values(["SOURCE_SYSTEM", "PROCEDURE_MONTH"])
        tm_v1 = tm[tm["SOURCE_SYSTEM"] == "EMR_V1"]
        tm_v2 = tm[tm["SOURCE_SYSTEM"] == "EMR_V2"]
        fig_vol = go.Figure()
        if not tm_v1.empty:
            fig_vol.add_trace(go.Bar(
                x=tm_v1["PROCEDURE_MONTH"].tolist(),
                y=tm_v1["PROCEDURES"].tolist(),
                name="V1",
                marker_color=COLORS["primary"], opacity=0.80,
                text=tm_v1["PROCEDURES"].tolist(),
                textposition="outside",
                textfont=dict(size=8, color="#003467"),
                hovertemplate="<b>V1 · %{x|%b %Y}</b>: %{y} procedures<extra></extra>",
            ))
        if not tm_v2.empty:
            fig_vol.add_trace(go.Bar(
                x=tm_v2["PROCEDURE_MONTH"].tolist(),
                y=tm_v2["PROCEDURES"].tolist(),
                name="V2",
                marker_color="#0BB99F", opacity=0.80,
                text=tm_v2["PROCEDURES"].tolist(),
                textposition="outside",
                textfont=dict(size=8, color="#003467"),
                hovertemplate="<b>V2 · %{x|%b %Y}</b>: %{y} procedures<extra></extra>",
            ))
        _tc = pd.Timestamp("2025-02-01")
        fig_vol.add_shape(
            type="line", x0=_tc, x1=_tc, y0=0, y1=1, yref="paper",
            line=dict(dash="dot", color=COLORS["warning"], width=1.5),
        )
        fig_vol.add_annotation(
            x=_tc, y=1, yref="paper", text="V2 start",
            showarrow=False, xanchor="left", yanchor="top",
            font=dict(size=9, color="#003467"),
        )
        fig_vol.update_layout(**cl(
            height=260, showlegend=True,
            legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", dtick="M3",
                       tickformat="%b %Y", tickangle=-45),
            yaxis=dict(gridcolor="#EBF3FB", title="Procedures"),
            margin=dict(l=0, r=0, t=30, b=60),
        ))
        st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div style="font-size:9px;color:#9BAEC8;margin-top:-4px">'
            'V1 Jun 2022–Jun 2023 &nbsp;·&nbsp; 20-month gap &nbsp;·&nbsp; V2 Feb 2025–Jun 2026</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Access ─────────────────────────────────────────────────────

    section_header("2  Access — Are patients reaching theatre promptly?")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        kpi_card(
            "Same-Day Wait (Median)",
            f"{_p50} min" if _p50 is not None else "—",
            "Elective · request → procedure start",
            color=COLORS["primary"],
        )
    with a2:
        kpi_card(
            "Longest Same-Day Wait",
            f"{_p90} min" if _p90 is not None else "—",
            "1 in 10 elective cases waits this long",
            color=COLORS["danger"],
        )
    with a3:
        kpi_card(
            "Wait > 4 Hours",
            f"{_pct_4h:.0f}%" if _pct_4h is not None else "—",
            "of elective cases on scheduled day",
            color=COLORS["warning"],
        )
    with a4:
        kpi_card(
            "Overdue Bookings",
            f"{_overdue}",
            "Scheduled · past planned date · unresolved",
            color=COLORS["warning"],
        )

    if _over_4hrs_n is not None:
        _impact_txt = (
            f"<b>{_over_4hrs_n} elective patients waited more than 4 hours on their scheduled "
            f"procedure day</b>"
        )
        if _over_8hrs_n is not None:
            _impact_txt += f" — of these, <b>{_over_8hrs_n} waited more than 8 hours</b>."
        else:
            _impact_txt += "."
        _impact_txt += (
            f" Halving the >4-hour rate would directly improve access for approximately "
            f"<b>{round(_over_4hrs_n * 0.5)} patients</b> over the same period."
        )
        info_card(_impact_txt, border_color=COLORS["danger"])

    st.markdown("<br>", unsafe_allow_html=True)

    if not access_dist_df.empty:
        _WAIT_ORDER = ["< 1 hr", "1–2 hrs", "2–4 hrs", "4–8 hrs"]
        _WAIT_COLORS = {
            "< 1 hr":  COLORS["success"],
            "1–2 hrs": COLORS["primary"],
            "2–4 hrs": COLORS["warning"],
            "4–8 hrs": COLORS["danger"],
        }
        wd = access_dist_df.copy()
        wd["WAIT_BUCKET"] = pd.Categorical(
            wd["WAIT_BUCKET"], categories=_WAIT_ORDER, ordered=True,
        )
        wd = wd.sort_values("WAIT_BUCKET").reset_index(drop=True)
        fig_wait = go.Figure(go.Bar(
            x=wd["WAIT_BUCKET"].tolist(),
            y=wd["CASES"].tolist(),
            marker_color=[_WAIT_COLORS.get(b, COLORS["muted"]) for b in wd["WAIT_BUCKET"]],
            opacity=0.85,
            text=wd["CASES"].tolist(),
            textposition="outside",
            textfont=dict(size=10, color="#003467"),
            hovertemplate="<b>%{x}</b>: %{y} cases<extra></extra>",
        ))
        fig_wait.update_layout(**cl(
            height=260, showlegend=False,
            xaxis=dict(categoryorder="array", categoryarray=_WAIT_ORDER,
                       gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor="#EBF3FB", title="Cases"),
            margin=dict(l=0, r=0, t=20, b=20),
        ))
        st.plotly_chart(fig_wait, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div style="font-size:9px;color:#9BAEC8;margin-top:-8px">'
            'V2 elective cases only &nbsp;·&nbsp; time from theatre request to procedure start &nbsp;·&nbsp; '
            '64.8% of V2 cases have wait recorded &nbsp;·&nbsp; capped at 480 min (shift-start artefact) &nbsp;·&nbsp; emergency excluded (avg ≈7 min)'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── ED TAT subsection ─────────────────────────────────────────────────────

    st.markdown(
        '<div style="font-size:12px;font-weight:700;color:#003467;margin:20px 0 10px 0;'
        'letter-spacing:0.04em;text-transform:uppercase">Emergency Access</div>',
        unsafe_allow_html=True,
    )

    _ek = emrg_kpis_df
    _etotal     = _safe_kpi(_ek, "TOTAL_EMERGENCY")
    _eavg       = _safe_kpi(_ek, "AVG_CLEAN_MINS")
    _en_clean   = _safe_kpi(_ek, "N_CLEAN")
    _en_long    = _safe_kpi(_ek, "N_LONG_WAIT") or 0

    ec1, ec2 = st.columns([1, 3], gap="large")
    with ec1:
        _long_caveat = f" · {_en_long} cases waited 338–670 min (excluded)" if _en_long else ""
        kpi_card(
            "Emergency to Theatre (Avg)",
            f"{_eavg} min" if _eavg is not None else "—",
            f"Booking → start · {_en_clean} of 20 cases · zeros + waits >120 min removed{_long_caveat}",
            color=COLORS["success"],
        )
    with ec2:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:6px">'
            'Monthly Emergency Procedure Volume (V2)</div>',
            unsafe_allow_html=True,
        )
        if not emrg_monthly_df.empty:
            em = emrg_monthly_df.copy()
            em["PROCEDURE_MONTH"] = pd.to_datetime(em["PROCEDURE_MONTH"])
            fig_em = go.Figure(go.Bar(
                x=em["PROCEDURE_MONTH"].tolist(),
                y=em["EMERGENCY_CASES"].tolist(),
                marker_color=COLORS["warning"],
                opacity=0.80,
                text=em["EMERGENCY_CASES"].tolist(),
                textposition="outside",
                textfont=dict(size=9, color="#003467"),
                hovertemplate="<b>%{x|%b %Y}</b>: %{y} emergency cases<extra></extra>",
            ))
            fig_em.update_layout(**cl(
                height=160, showlegend=False,
                xaxis=dict(gridcolor="rgba(0,0,0,0)", tickformat="%b %Y", tickangle=-45,
                           dtick="M2"),
                yaxis=dict(gridcolor="#EBF3FB", title="Cases", dtick=1),
                margin=dict(l=0, r=0, t=20, b=40),
            ))
            st.plotly_chart(fig_em, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                '<div style="font-size:9px;color:#9BAEC8;margin-top:-8px">'
                f'V2 · {_etotal} total emergency cases · avg = {_eavg} min (clean cohort) · '
                f'{_en_long} cases at 338–670 min excluded from average · under investigation</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Procedure Throughput ──────────────────────────────────────

    section_header("3  Procedure Throughput — What is consuming theatre capacity?")

    col_hrs, col_wt = st.columns(2, gap="large")

    with col_hrs:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:4px">'
            'V1 · Procedures by Total Theatre Hours</div>'
            '<div style="font-size:9px;color:#9BAEC8;margin-bottom:8px">'
            'Historical load distribution · Jun 2022–Jun 2023</div>',
            unsafe_allow_html=True,
        )
        if not v1_proc_hrs_df.empty:
            ph = v1_proc_hrs_df.head(12).copy()
            ph = ph.sort_values("TOTAL_HOURS", ascending=True)
            fig_hrs = go.Figure(go.Bar(
                y=ph["PROCEDURE_NAME"].tolist(),
                x=ph["TOTAL_HOURS"].tolist(),
                orientation="h",
                marker_color=COLORS["primary"],
                opacity=0.85,
                text=[f"{h:.0f} h  ({p:.0f}%)" for h, p in
                      zip(ph["TOTAL_HOURS"], ph["PCT_OF_TOTAL_HOURS"])],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=9, color="#fff"),
                customdata=ph[["CASE_COUNT", "MEDIAN_MINS", "PCT_OF_TOTAL_HOURS"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>%{x:.1f} hrs total<br>"
                    "%{customdata[0]} cases · median %{customdata[1]} min<br>"
                    "%{customdata[2]:.1f}% of all theatre hours<extra></extra>"
                ),
            ))
            fig_hrs.update_layout(**cl(
                height=max(300, len(ph) * 30),
                showlegend=False,
                xaxis=dict(gridcolor="#EBF3FB", title="Total hours"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9)),
                margin=dict(l=0, r=10, t=10, b=10),
            ))
            st.plotly_chart(fig_hrs, use_container_width=True, config={"displayModeBar": False})

    with col_wt:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:4px">'
            'V2 · Procedures by Median Same-Day Wait</div>'
            '<div style="font-size:9px;color:#9BAEC8;margin-bottom:8px">'
            'Current access pattern · elective cases · Feb 2025–Jun 2026</div>',
            unsafe_allow_html=True,
        )
        if not v2_proc_wait_df.empty:
            pw = v2_proc_wait_df.head(12).copy()
            pw = pw.sort_values("MEDIAN_WAIT_MINS", ascending=True)
            _wait_bar_colors = [
                COLORS["danger"] if w > 240 else COLORS["warning"]
                for w in pw["MEDIAN_WAIT_MINS"]
            ]
            fig_wt = go.Figure(go.Bar(
                y=pw["PROCEDURE_NAME"].tolist(),
                x=pw["MEDIAN_WAIT_MINS"].tolist(),
                orientation="h",
                marker_color=_wait_bar_colors,
                opacity=0.85,
                text=[f"{int(w)} min" for w in pw["MEDIAN_WAIT_MINS"]],
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=9, color="#fff"),
                customdata=pw[["N", "PCT_OVER_4HRS", "P90_WAIT_MINS"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>Median wait: %{x} min<br>"
                    "%{customdata[0]} cases · %{customdata[1]:.0f}% wait >4 hrs<br>"
                    "90th pct: %{customdata[2]} min<extra></extra>"
                ),
            ))
            fig_wt.update_layout(**cl(
                height=max(300, len(pw) * 30),
                showlegend=False,
                xaxis=dict(gridcolor="#EBF3FB", title="Median wait (min)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=9)),
                margin=dict(l=0, r=10, t=10, b=10),
            ))
            st.plotly_chart(fig_wt, use_container_width=True, config={"displayModeBar": False})

    _med_dur = int(theatre_kpis["MEDIAN_DURATION_MINS"]) if theatre_kpis["MEDIAN_DURATION_MINS"] else 120
    _p90_dur = int(theatre_kpis["P90_DURATION_MINS"])    if theatre_kpis["P90_DURATION_MINS"]    else 300

    info_card(
        f"At V1 median duration ({_med_dur} min + 30 min overhead), one high-duration procedure "
        f"on a mixed list is consistent with the observed 556-min 90th percentile same-day wait for "
        f"subsequent patients. Duration-aware scheduling directly addresses this.",
        border_color=COLORS["primary"],
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 4: Operational Priorities ────────────────────────────────────

    section_header("4  Where should management intervene first?")

    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["dark"]};margin-bottom:16px;line-height:1.6">'
        f'Elective theatre delay is concentrated in high-duration procedures. '
        f'The three actions below address the constraint directly.'
        f'</div>',
        unsafe_allow_html=True,
    )

    _pri_rows = [
        ("1", "Duration-aware scheduling",
         "High-duration procedures (arthroplasty, major fixation) are associated with "
         "substantially longer same-day waits. Scheduling these cases first in the session "
         "or in dedicated slots would reduce the 90th percentile wait.",
         "Theatre scheduling / Clinical lead",
         COLORS["danger"]),
        ("2", "Overdue booking resolution",
         f"{_overdue} cases are scheduled with a past planned date and no recorded completion. "
         "Each represents a patient whose procedure did not proceed on the planned day with "
         "no visible follow-up action.",
         "Theatre coordinator",
         COLORS["warning"]),
        ("3", "Procedure-specific queue monitoring",
         "ANKLE ORIF (67 cases, 37% wait >4 hrs) and TOTAL KNEE/HIP REPLACEMENT "
         "(50–52% wait >4 hrs) are the highest-volume procedures with the longest queues. "
         "Track these specifically rather than overall list performance.",
         "Theatre scheduling",
         COLORS["warning"]),
    ]

    _pri_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:11px">'
        '<thead><tr style="background:#EBF3FB">'
        '<th style="padding:8px 10px;text-align:left;color:#003467;width:3%">#</th>'
        '<th style="padding:8px 10px;text-align:left;color:#003467;width:22%">Action</th>'
        '<th style="padding:8px 10px;text-align:left;color:#003467;width:55%">Rationale</th>'
        '<th style="padding:8px 10px;text-align:left;color:#003467;width:20%">Owner</th>'
        '</tr></thead><tbody>'
    )
    for rank, title, rationale, owner, color in _pri_rows:
        _pri_html += (
            f'<tr style="border-bottom:1px solid #EBF3FB">'
            f'<td style="padding:8px 10px;font-weight:700;color:{color}">{rank}</td>'
            f'<td style="padding:8px 10px;font-weight:600;color:#003467">{title}</td>'
            f'<td style="padding:8px 10px;color:#4A5568;line-height:1.5">{rationale}</td>'
            f'<td style="padding:8px 10px;color:#6B8CAE">{owner}</td>'
            f'</tr>'
        )
    _pri_html += '</tbody></table>'
    st.markdown(_pri_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 5: Data Capability ────────────────────────────────────────────

    section_header("5  What data would improve future decisions?")

    _cap_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:11px">'
        '<thead><tr style="background:#F8FAFC">'
        '<th style="padding:8px 10px;text-align:left;color:#003467;border-bottom:2px solid #E2E8F0;width:22%">Metric</th>'
        '<th style="padding:8px 10px;text-align:left;color:#003467;border-bottom:2px solid #E2E8F0;width:30%">Variables needed</th>'
        '<th style="padding:8px 10px;text-align:left;color:#003467;border-bottom:2px solid #E2E8F0;width:48%">Constraint in V2</th>'
        '</tr></thead><tbody>'
        '<tr style="border-bottom:1px solid #EBF3FB;background:#FFFBF0">'
        '<td style="padding:8px 10px;font-weight:700;color:#003467">Room capture</td>'
        '<td style="padding:8px 10px;color:#4A5568">Room field at booking</td>'
        '<td style="padding:8px 10px;color:#4A5568">15.6% recorded · <b>unlocks 3 metrics below</b> · Owner: EMR / Theatre admin (Inv 146)</td>'
        '</tr>'
        '<tr style="border-bottom:1px solid #EBF3FB">'
        '<td style="padding:8px 10px;font-weight:600;color:#6B8CAE">Theatre utilisation</td>'
        '<td style="padding:8px 10px;color:#4A5568">Room + session hours denominator</td>'
        '<td style="padding:8px 10px;color:#4A5568">Blocked until room capture improves</td>'
        '</tr>'
        '<tr style="border-bottom:1px solid #EBF3FB">'
        '<td style="padding:8px 10px;font-weight:600;color:#6B8CAE">Turnover time</td>'
        '<td style="padding:8px 10px;color:#4A5568">Room + time_in + time_out, same record</td>'
        '<td style="padding:8px 10px;color:#4A5568">V1 has times, no room · V2 has room, no times (Inv 148)</td>'
        '</tr>'
        '<tr style="border-bottom:1px solid #EBF3FB">'
        '<td style="padding:8px 10px;font-weight:600;color:#6B8CAE">Cancellation rate</td>'
        '<td style="padding:8px 10px;color:#4A5568">Status code on cancelled bookings</td>'
        '<td style="padding:8px 10px;color:#4A5568">Cancellations deleted, not logged (Inv 145)</td>'
        '</tr>'
        '<tr>'
        '<td style="padding:8px 10px;font-weight:600;color:#6B8CAE">Emergency response time</td>'
        '<td style="padding:8px 10px;color:#4A5568">Arrival timestamp (cons_ts)</td>'
        '<td style="padding:8px 10px;color:#4A5568">Walk-in cons_ts 0.1% coverage · schema-supported (Inv 149)</td>'
        '</tr>'
        '</tbody></table>'
        '<div style="font-size:9px;color:#9BAEC8;margin-top:8px">'
        'All metrics blocked by recording gaps, not system design. '
        'Room capture is the highest-priority fix — it unblocks utilisation, turnover, and session TAT simultaneously.'
        '</div>'
    )
    st.markdown(_cap_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — OPD → IPD Conversion
# ══════════════════════════════════════════════════════════════════════════════

with tab_conversion:

    def _caption(text):
        st.markdown(
            f'<div style="font-size:9px;color:#9BAEC8;margin-top:-6px;margin-bottom:4px">{text}</div>',
            unsafe_allow_html=True,
        )

    info_card(
        "<b>V2 only · Feb 2025 – Jun 2026 · 2,324 admissions.</b> "
        "87.2% are OPD-triggered (prior OPD visit within 7 days). "
        "Feb 2025 is a ramp-up month (66% OPD-trigger); stable from Mar 2025 onward. "
        "Ward and conversion_type have 100% coverage. "
        "Pay mode excluded — 99.2% cash, no analytical signal.",
        border_color=COLORS["muted"],
    )

    _total_adm    = int(conv_kpis["TOTAL_ADMISSIONS"])
    _opd_trig_pct = float(conv_kpis["OPD_TRIGGER_PCT"])
    _direct_n     = int(conv_kpis["DIRECT"])
    _direct_pct   = round(_direct_n * 100.0 / _total_adm, 1) if _total_adm > 0 else 0
    _data_from    = pd.to_datetime(conv_kpis["DATA_FROM"]).strftime("%b %Y")
    _data_to      = pd.to_datetime(conv_kpis["DATA_TO"]).strftime("%b %Y")

    # ── Section 1: Scale ─────────────────────────────────────────────────────────

    section_header("1  How many admitted?", margin_top=20)

    # Prepare monthly data here — used for MoM KPI derivation + chart
    cm = pd.DataFrame()
    if not conv_monthly_df.empty:
        cm = conv_monthly_df.copy()
        cm["ADMISSION_MONTH"] = pd.to_datetime(cm["ADMISSION_MONTH"])
        cm = cm[cm["ADMISSION_MONTH"] < pd.Timestamp("2026-07-01")]
        cm = cm.sort_values("ADMISSION_MONTH")

    # Derive MoM conversion rate from latest two complete months
    _latest_rate = _prev_rate = _mom_delta = None
    _latest_month_str = _prev_month_str = ""
    if len(cm) >= 2 and "CONVERSION_RATE" in cm.columns:
        _lr = cm.iloc[-1]
        _pr = cm.iloc[-2]
        _latest_rate      = float(_lr["CONVERSION_RATE"]) if pd.notna(_lr["CONVERSION_RATE"]) else None
        _prev_rate        = float(_pr["CONVERSION_RATE"]) if pd.notna(_pr["CONVERSION_RATE"]) else None
        _latest_month_str = _lr["ADMISSION_MONTH"].strftime("%b %Y")
        _prev_month_str   = _pr["ADMISSION_MONTH"].strftime("%b %Y")
        if _latest_rate is not None and _prev_rate is not None:
            _mom_delta = round(_latest_rate - _prev_rate, 2)

    k1, k2, k3 = st.columns(3)
    with k1:
        if _latest_rate is not None and _mom_delta is not None:
            _arrow    = "↑" if _mom_delta > 0 else ("↓" if _mom_delta < 0 else "→")
            _delta_str = f"{_arrow} {abs(_mom_delta):.2f}pp vs {_prev_month_str}"
            kpi_card(
                f"Conversion Rate · {_latest_month_str}",
                f"{_latest_rate:.1f}%",
                _delta_str,
                color=COLORS["success"] if _mom_delta >= 0 else COLORS["danger"],
            )
        else:
            kpi_card("Total Admissions", f"{_total_adm:,}", f"V2 only · {_data_from} – {_data_to}")
    with k2:
        kpi_card(
            "OPD-Triggered",
            f"{_opd_trig_pct:.0f}%",
            "Prior OPD visit within 7 days",
            color=COLORS["primary"],
        )
    with k3:
        kpi_card(
            "Direct Admissions",
            f"{_direct_pct:.0f}%",
            f"{_direct_n:,} admissions — no preceding OPD visit",
            color=COLORS["muted"],
        )

    if not cm.empty:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=cm["ADMISSION_MONTH"].tolist(),
            y=cm["ADMISSIONS"].tolist(),
            name="Admissions",
            marker_color="#0BB99F",
            opacity=0.85,
            yaxis="y",
            hovertemplate="<b>%{x|%b %Y}</b>: %{y} admissions<extra></extra>",
        ))
        if "OPD_VISITS" in cm.columns:
            fig_trend.add_trace(go.Scatter(
                x=cm["ADMISSION_MONTH"].tolist(),
                y=cm["OPD_VISITS"].tolist(),
                name="OPD Visits",
                mode="lines+markers",
                line=dict(color=COLORS["muted"], width=2, dash="dot"),
                marker=dict(size=5),
                yaxis="y2",
                hovertemplate="<b>%{x|%b %Y}</b>: %{y:,} OPD visits<extra></extra>",
            ))
        fig_trend.update_layout(**cl(
            height=260,
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
            xaxis=dict(gridcolor="rgba(0,0,0,0)", dtick="M2", tickformat="%b %Y", tickangle=-45),
            yaxis=dict(title="Admissions", gridcolor="#EBF3FB"),
            yaxis2=dict(title="OPD Visits", overlaying="y", side="right",
                        showgrid=False, tickfont=dict(color=COLORS["muted"])),
            margin=dict(l=0, r=60, t=30, b=60),
        ))
        st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        _caption("Teal bars = admissions · dotted line = OPD visits (right axis) · V2 only · Jul 2026 excluded (partial month)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Where ─────────────────────────────────────────────────────────

    section_header("2  Where do they go?", margin_top=0)

    _WARD_LABELS = {
        "SIMBA-ORTHOPAEDIC":   "Orthopaedic (Simba)",
        "NDOVU-SURGICAL WARD": "Surgical (Ndovu)",
        "NYATI-MEDICAL WARD":  "Medical (Nyati)",
        "NYATI-OBS/GYN WARD":  "Obs/Gyn (Nyati)",
        "MATERNITY":           "Maternity",
    }

    if not conv_ward_df.empty:
        wd = conv_ward_df[conv_ward_df["WARD"] != "Unknown"].copy()
        wd["WARD_LABEL"] = wd["WARD"].map(_WARD_LABELS).fillna(wd["WARD"])
        wd = wd.sort_values("ADMISSIONS", ascending=True)

        fig_ward = go.Figure(go.Bar(
            x=wd["ADMISSIONS"].tolist(),
            y=wd["WARD_LABEL"].tolist(),
            orientation="h",
            marker_color=COLORS["primary"],
            opacity=0.85,
            text=[
                f"{int(r['ADMISSIONS'])}  ({r['PCT']:.0f}%)  OPD {r['OPD_TRIGGER_PCT']:.0f}%"
                for _, r in wd.iterrows()
            ],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=9, color="#fff"),
            customdata=wd[["PCT", "OPD_TRIGGER_PCT"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>%{x} admissions (%{customdata[0]:.0f}%)"
                "<br>OPD-trigger: %{customdata[1]:.0f}%<extra></extra>"
            ),
        ))
        fig_ward.update_layout(**cl(
            height=230,
            showlegend=False,
            xaxis=dict(gridcolor="#EBF3FB", visible=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
            margin=dict(l=0, r=10, t=10, b=10),
        ))
        st.plotly_chart(fig_ward, use_container_width=True, config={"displayModeBar": False})
        _caption("Bar width = admissions · text shows share and OPD-trigger rate per ward · V2 only")

        st.markdown(
            '<div style="background:#EBF3FB;border-left:4px solid #003467;'
            'border-radius:4px;padding:10px 14px;margin:14px 0 0 0;'
            'font-size:11px;font-weight:600;color:#003467">'
            'Orthopaedic and Surgical wards absorb 82% of all admissions. '
            'OPD-trigger rate is consistent across all three main wards (85&ndash;90%). '
            'Process improvements to the OPD admission pathway will have hospital-wide effect '
            '&mdash; Orthopaedic and Surgical should be the primary focus.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Who ───────────────────────────────────────────────────────────

    section_header("3  What type of visit converts?", margin_top=0)

    _TYPE_LABELS = {
        "New Patient Conversion": "New Patient",
        "Revisit Conversion":     "Revisit",
        "Direct Admission":       "Direct",
        "Walk-In Conversion":     "Walk-In",
    }
    _TYPE_COLORS = {
        "New Patient Conversion": COLORS["primary"],
        "Revisit Conversion":     COLORS["success"],
        "Direct Admission":       COLORS["warning"],
        "Walk-In Conversion":     COLORS["muted"],
    }

    if not conv_type_df.empty:
        ct = conv_type_df.copy()
        ct["TYPE_LABEL"] = ct["CONVERSION_TYPE"].map(_TYPE_LABELS).fillna(ct["CONVERSION_TYPE"])
        ct["COLOR"]      = ct["CONVERSION_TYPE"].map(_TYPE_COLORS).fillna(COLORS["muted"])
        ct = ct.sort_values("N", ascending=True)

        fig_type = go.Figure(go.Bar(
            x=ct["N"].tolist(),
            y=ct["TYPE_LABEL"].tolist(),
            orientation="h",
            marker_color=ct["COLOR"].tolist(),
            opacity=0.85,
            text=[f"{int(r['N']):,}  ({r['PCT']:.0f}%)" for _, r in ct.iterrows()],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="#fff"),
            customdata=ct["PCT"].tolist(),
            hovertemplate="<b>%{y}</b>: %{x:,} admissions (%{customdata:.0f}%)<extra></extra>",
        ))
        fig_type.update_layout(**cl(
            height=200,
            showlegend=False,
            xaxis=dict(gridcolor="#EBF3FB", visible=False),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
            margin=dict(l=0, r=10, t=10, b=10),
        ))
        st.plotly_chart(fig_type, use_container_width=True, config={"displayModeBar": False})
        _caption(
            "V2 only · 'Direct' = admitted without a prior OPD visit · "
            "New Patient Conversions account for 64% — the dominant admission trigger"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 4: When ──────────────────────────────────────────────────────────

    section_header("4  When do admissions peak?", margin_top=0)

    if not conv_dow_df.empty:
        dw = conv_dow_df.copy()
        _dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dw["DOW_SORT"] = dw["DAY_NAME"].map({d: i for i, d in enumerate(_dow_order)})
        dw = dw.sort_values("DOW_SORT")

        _dow_avg    = dw["ADMISSIONS"].mean()
        _bar_colors = [
            COLORS["warning"] if v > _dow_avg * 1.05 else COLORS["primary"]
            for v in dw["ADMISSIONS"]
        ]

        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(
            name="Admissions",
            x=dw["DAY_NAME"].tolist(),
            y=dw["ADMISSIONS"].tolist(),
            marker_color=_bar_colors,
            opacity=0.85,
            text=dw["ADMISSIONS"].tolist(),
            textposition="outside",
            textfont=dict(size=9, color="#003467"),
            hovertemplate="<b>%{x}</b>: %{y} admissions<extra></extra>",
        ))
        fig_dow.add_trace(go.Scatter(
            name="OPD-trigger %",
            x=dw["DAY_NAME"].tolist(),
            y=dw["OPD_TRIGGER_PCT"].tolist(),
            mode="lines+markers",
            yaxis="y2",
            line=dict(color=COLORS["success"], width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b>: %{y:.0f}% OPD-triggered<extra></extra>",
        ))
        fig_dow.add_hline(
            y=_dow_avg, line_dash="dot",
            line_color=COLORS["muted"], line_width=1,
            annotation_text=f"Avg {int(_dow_avg)}/day",
            annotation_position="top left",
            annotation_font=dict(size=9, color=COLORS["muted"]),
        )
        fig_dow.update_layout(**cl(
            height=270,
            showlegend=True,
            legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0)",
                categoryorder="array",
                categoryarray=_dow_order,
            ),
            yaxis=dict(gridcolor="#EBF3FB", title="Admissions"),
            yaxis2=dict(
                title="OPD-trigger %",
                overlaying="y",
                side="right",
                showgrid=False,
                ticksuffix="%",
                range=[70, 100],
            ),
            margin=dict(l=0, r=50, t=30, b=10),
        ))
        st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})
        _caption(
            "Orange bars = >5% above daily avg · dotted line = OPD-trigger % (right axis) · "
            "Mon peak volume; Sat/Sun lower volume but higher OPD-trigger rate (91%)"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 5: Implication ───────────────────────────────────────────────────

    section_header("5  Where should we act first?", margin_top=0)

    info_card(
        "<b>The OPD pathway is the dominant source of inpatient admissions (87%) across wards and weekdays.</b> "
        "Operational changes to OPD admission processes are therefore likely to have hospital-wide impact "
        "and should be prioritized over ward-specific process redesign."
        "<br><br>"
        "<b>Orthopaedic and Surgical wards account for 82% of all admissions.</b> "
        "Improvements to the OPD admission process in these wards will have the greatest operational impact.",
        border_color=COLORS["primary"],
    )
