"""
Pharmacy — pages/3_pharmacy.py
Central question: Is pharmacy the operational constraint?

V2 (Feb 2025–Jul 2026): 184,412 orders · 8 item classes
Dispensing TAT = request_stamp → dispensed_stamp (tat_mins, capped at 240 min)
Key finding: P50 fell from ~80 min (Mar 2025) to 27 min (Jun 2026) as volume grew 40%+
Non-dispensing rate 6.4% — reason not captured; labelled as "no dispensing recorded"
pharm_ts in patient journey = prescription-write time, not pharmacy arrival (Inv 151)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Pharmacy · SPH Ortho",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.theme import (
    apply_theme, render_sidebar, kpi_card, section_header, info_card,
    page_header, COLORS, cl,
)
from dashboard.queries import (
    q_pharm_workload_summary, q_pharm_throughput_monthly,
    q_pharm_tat_dist, q_pharm_class_tat, q_pharm_top_items,
    q_pharm_hour, q_pharm_dow, q_pharm_speed_summary,
)

apply_theme()
render_sidebar("pharmacy")

page_header("Pharmacy", subtitle="Is pharmacy the operational constraint?")

# ── Load data ──────────────────────────────────────────────────────────────────

try:
    workload_df   = q_pharm_workload_summary()
    throughput_df = q_pharm_throughput_monthly()
    tat_dist_df   = q_pharm_tat_dist()
    class_tat_df  = q_pharm_class_tat()
    top_items_df  = q_pharm_top_items()
    hour_df       = q_pharm_hour()
    dow_df        = q_pharm_dow()
    speed_df      = q_pharm_speed_summary()
except Exception as e:
    st.error(f"Failed to load data. Check Snowflake connection.\n\n{e}")
    st.stop()

# ── Helpers ────────────────────────────────────────────────────────────────────

_w = workload_df.iloc[0] if not workload_df.empty else None

_total_orders     = int(_w["TOTAL_ORDERS"])      if _w is not None else 0
_not_dispensed    = int(_w["NOT_DISPENSED"])      if _w is not None else 0
_not_disp_pct     = float(_w["NOT_DISPENSED_PCT"]) if _w is not None else 0.0
_opd_visits       = int(_w["OPD_VISITS"])         if _w is not None else 0
_orders_per_visit = float(_w["ORDERS_PER_OPD_VISIT"]) if _w is not None else 0.0
_data_from        = pd.to_datetime(_w["DATA_FROM"]).strftime("%b %Y") if _w is not None else ""
_data_to          = pd.to_datetime(_w["DATA_TO"]).strftime("%b %Y")   if _w is not None else ""

_sv2 = speed_df[speed_df["SOURCE_SYSTEM"] == "EMR_V2"].iloc[0] \
       if not speed_df.empty and "EMR_V2" in speed_df["SOURCE_SYSTEM"].values else None
_p50_overall = int(_sv2["P50_MINS"])      if _sv2 is not None else None
_p90_overall = int(_sv2["P90_MINS"])      if _sv2 is not None else None
_coverage    = float(_sv2["COVERAGE_PCT"]) if _sv2 is not None else None

# Monthly trend helpers — exclude partial months (<5,000 orders)
_tm = throughput_df.copy()
if not _tm.empty:
    _tm["ORDER_MONTH"] = pd.to_datetime(_tm["ORDER_MONTH"])
    _tm = _tm.sort_values("ORDER_MONTH")
    _tm_full = _tm[_tm["TOTAL_ORDERS"] >= 5000].reset_index(drop=True)
else:
    _tm_full = pd.DataFrame()

_first = _tm_full.iloc[0]  if not _tm_full.empty else None
_last  = _tm_full.iloc[-1] if not _tm_full.empty else None
_p50_start   = int(_first["P50_TAT_MINS"]) if _first is not None else None
_p50_end     = int(_last["P50_TAT_MINS"])  if _last  is not None else None
_p90_start   = int(_first["P90_TAT_MINS"]) if _first is not None else None
_p90_end     = int(_last["P90_TAT_MINS"])  if _last  is not None else None
_vol_start   = int(_first["TOTAL_ORDERS"]) if _first is not None else None
_vol_end     = int(_last["TOTAL_ORDERS"])  if _last  is not None else None
_vol_chg_pct = round((_vol_end - _vol_start) / _vol_start * 100) \
               if _vol_start and _vol_end and _vol_start > 0 else None

# Correlation (volume vs P50, complete months only)
_r = None
if not _tm_full.empty and len(_tm_full) >= 5:
    _valid = _tm_full[_tm_full["P50_TAT_MINS"].notna()]
    if len(_valid) >= 5:
        _r = float(np.corrcoef(_valid["TOTAL_ORDERS"], _valid["P50_TAT_MINS"])[0, 1])

# ── Section 1: Dispensing Workload ─────────────────────────────────────────────

section_header("1  Dispensing Workload — How much is pharmacy handling?", margin_top=20)

w1, w2, w3 = st.columns(3)
with w1:
    kpi_card(
        "Total V2 Orders",
        f"{_total_orders:,}",
        f"{_data_from} – {_data_to} · V2 system",
        color=COLORS["primary"],
    )
with w2:
    avg_monthly = round(_total_orders / max(len(_tm_full), 1))
    kpi_card(
        "Avg Orders / Month",
        f"{avg_monthly:,}",
        "Complete months only",
        color=COLORS["muted"],
    )
with w3:
    kpi_card(
        "Orders per OPD Visit",
        f"{_orders_per_visit:.1f}",
        f"V2 pharmacy orders ÷ {_opd_visits:,} V2 OPD visits",
        color=COLORS["muted"],
    )

st.markdown("<br>", unsafe_allow_html=True)

if not _tm.empty:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
        'Monthly Order Volume (V2)</div>',
        unsafe_allow_html=True,
    )
    fig_vol = go.Figure(go.Bar(
        x=_tm["ORDER_MONTH"].tolist(),
        y=_tm["TOTAL_ORDERS"].tolist(),
        marker_color=COLORS["primary"],
        opacity=0.75,
        text=_tm["TOTAL_ORDERS"].tolist(),
        textposition="outside",
        textfont=dict(size=8, color="#003467"),
        hovertemplate="<b>%{x|%b %Y}</b>: %{y:,} orders<extra></extra>",
    ))
    fig_vol.update_layout(**cl(
        height=220,
        xaxis=dict(gridcolor="rgba(0,0,0,0)", dtick="M1",
                   tickformat="%b %Y", tickangle=-45),
        yaxis=dict(gridcolor="#EBF3FB", title="Orders"),
        margin=dict(l=0, r=0, t=10, b=60),
    ))
    st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div style="font-size:9px;color:#9BAEC8;margin-top:-4px">'
        'V2 only · Feb 2025 – present · first and last months may be partial</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 2: Operational Throughput ─────────────────────────────────────────

section_header("2  Operational Throughput — How fast is dispensing?")

t1, t2, t3 = st.columns(3)
with t1:
    kpi_card(
        "Median Dispensing Time",
        f"{_p50_overall} min" if _p50_overall is not None else "—",
        "Median · V2 · request → dispensed (tat_mins < 240)",
        color=COLORS["success"],
    )
with t2:
    kpi_card(
        "Longest Dispensing Wait",
        f"{_p90_overall} min" if _p90_overall is not None else "—",
        "1 in 10 orders takes at least this long",
        color=COLORS["warning"],
    )
with t3:
    kpi_card(
        "TAT Coverage",
        f"{_coverage:.1f}%" if _coverage is not None else "—",
        "Orders with both request + dispensed timestamps",
        color=COLORS["muted"],
    )

st.markdown("<br>", unsafe_allow_html=True)

# Dual-axis chart: monthly orders (bar) + P50 (line)
if not _tm.empty:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
        'Monthly Order Volume vs Median Dispensing Time (V2)</div>',
        unsafe_allow_html=True,
    )
    _chart_df = _tm[_tm["TOTAL_ORDERS"] >= 5000].copy()  # exclude partial months
    fig_dual = go.Figure()
    fig_dual.add_trace(go.Bar(
        x=_chart_df["ORDER_MONTH"].tolist(),
        y=_chart_df["TOTAL_ORDERS"].tolist(),
        name="Monthly Orders",
        marker_color=COLORS["primary"],
        opacity=0.25,
        yaxis="y2",
        hovertemplate="<b>%{x|%b %Y}</b>: %{y:,} orders<extra></extra>",
    ))
    fig_dual.add_trace(go.Scatter(
        x=_chart_df["ORDER_MONTH"].tolist(),
        y=_chart_df["P50_TAT_MINS"].tolist(),
        name="Median TAT (min)",
        mode="lines+markers",
        line=dict(color=COLORS["success"], width=2.5),
        marker=dict(size=6),
        yaxis="y",
        hovertemplate="<b>%{x|%b %Y}</b> Median: %{y} min<extra></extra>",
    ))
    fig_dual.update_layout(**cl(
        height=280, showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=10)),
        yaxis=dict(title="Dispensing Time (min)", gridcolor="#EBF3FB"),
        yaxis2=dict(title="Monthly Orders", overlaying="y", side="right", showgrid=False),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", dtick="M2",
                   tickformat="%b %Y", tickangle=-45),
        margin=dict(l=0, r=60, t=30, b=60),
    ))
    st.plotly_chart(fig_dual, use_container_width=True, config={"displayModeBar": False})

    _r_label = ""
    if _r is not None:
        _r_label = (
            f"&nbsp;·&nbsp; r = {_r:.2f} — "
            + ("higher workload is associated with faster, not slower, dispensing"
               if _r < -0.3 else
               "no strong relationship between volume and dispensing speed")
        )
    st.markdown(
        f'<div style="font-size:9px;color:#9BAEC8;margin-top:-4px">'
        f'Partial months (Feb 2025, Jul 2026) excluded{_r_label}</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 3: Service Reliability ────────────────────────────────────────────

section_header("3  Service Reliability — Are delays concentrated or widespread?")

sr1, sr2 = st.columns([3, 1], gap="large")

with sr1:
    if not tat_dist_df.empty:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
            'Dispensing Time Distribution — V2 (orders with timestamps, tat < 4 hrs)</div>',
            unsafe_allow_html=True,
        )
        td = tat_dist_df.sort_values("BUCKET_ORDER")
        _BUCKET_COLORS = {
            "< 30 min":  COLORS["success"],
            "30–60 min": COLORS["primary"],
            "1–2 hrs":   COLORS["warning"],
            "> 2 hrs":   COLORS["danger"],
        }
        fig_dist = go.Figure(go.Bar(
            y=td["TAT_BUCKET"].tolist(),
            x=td["PCT"].tolist(),
            orientation="h",
            marker_color=[_BUCKET_COLORS.get(b, COLORS["muted"]) for b in td["TAT_BUCKET"]],
            opacity=0.85,
            text=[f"{p:.0f}%" for p in td["PCT"].tolist()],
            textposition="outside",
            textfont=dict(size=11, color="#003467"),
            customdata=td["ORDERS"].tolist(),
            hovertemplate="<b>%{y}</b>: %{x:.0f}% · %{customdata:,} orders<extra></extra>",
        ))
        _bucket_order = ["< 30 min", "30–60 min", "1–2 hrs", "> 2 hrs"]
        fig_dist.update_layout(**cl(
            height=230, showlegend=False,
            xaxis=dict(title="% of timed orders", gridcolor="#EBF3FB", ticksuffix="%"),
            yaxis=dict(categoryorder="array", categoryarray=list(reversed(_bucket_order)),
                       gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=60, t=10, b=20),
        ))
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

with sr2:
    kpi_card(
        "No Dispensing Recorded",
        f"{_not_disp_pct:.1f}%",
        f"{_not_dispensed:,} of {_total_orders:,} V2 orders",
        color=COLORS["muted"],
    )
    st.markdown(
        '<div style="font-size:10px;color:#6B8CAE;line-height:1.5;margin-top:8px;'
        'padding:10px 12px;background:#F8FAFC;border:1px solid #D6E4F0;border-radius:6px">'
        'Reason not captured in V2. May include stock-out, patient choice, '
        'prescription change, or documentation gap. Cannot be attributed '
        'to pharmacy performance alone.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 4: Priority Medication Groups ─────────────────────────────────────

section_header("4  Priority Medication Groups — Which prescriptions wait longest?")

pg1, pg2 = st.columns([1, 1], gap="large")

with pg1:
    if not class_tat_df.empty:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
            'Median TAT by Item Class (V2)</div>',
            unsafe_allow_html=True,
        )
        cd = class_tat_df.copy()
        cd = cd[cd["MEDIAN_TAT_MINS"].notna()].sort_values("MEDIAN_TAT_MINS", ascending=True)
        fig_class = go.Figure(go.Bar(
            y=cd["ITEM_CLASS"].tolist(),
            x=cd["MEDIAN_TAT_MINS"].tolist(),
            orientation="h",
            marker_color=COLORS["primary"],
            opacity=0.80,
            text=[f"{int(v)} min" for v in cd["MEDIAN_TAT_MINS"].tolist()],
            textposition="outside",
            textfont=dict(size=9, color="#003467"),
            customdata=cd["TOTAL_ORDERS"].tolist(),
            hovertemplate="<b>%{y}</b>: %{x} min median · %{customdata:,} orders<extra></extra>",
        ))
        fig_class.update_layout(**cl(
            height=max(220, len(cd) * 38),
            showlegend=False,
            xaxis=dict(title="Median TAT (min)", gridcolor="#EBF3FB"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=70, t=10, b=20),
        ))
        st.plotly_chart(fig_class, use_container_width=True, config={"displayModeBar": False})

with pg2:
    if not top_items_df.empty:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
            'Top 15 Slowest Items — ≥30 orders (V2)</div>',
            unsafe_allow_html=True,
        )
        ti = top_items_df[["ITEM_NAME", "ITEM_CLASS", "TOTAL_ORDERS",
                            "MEDIAN_TAT_MINS", "OVER_2HR_PCT"]].copy()
        ti.columns = ["Item", "Class", "Orders", "Median (min)", ">2hr %"]
        st.dataframe(
            ti.style.format({"Median (min)": "{:.0f}", ">2hr %": "{:.1f}"}),
            use_container_width=True,
            hide_index=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 5: Variation ───────────────────────────────────────────────────────

section_header("5  Where Does Pharmacy Slow Down?")

v1, v2 = st.columns(2, gap="large")

with v1:
    if not hour_df.empty:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
            'Median TAT by Hour Prescriptions Written (V2)</div>',
            unsafe_allow_html=True,
        )
        hd = hour_df.copy().sort_values("ORDER_HOUR")
        fig_hr = go.Figure(go.Bar(
            x=hd["ORDER_HOUR"].tolist(),
            y=hd["MEDIAN_TAT_MINS"].tolist(),
            marker_color=[
                COLORS["danger"] if v == hd["MEDIAN_TAT_MINS"].max() else COLORS["primary"]
                for v in hd["MEDIAN_TAT_MINS"].tolist()
            ],
            opacity=0.80,
            text=[f"{int(v)}" for v in hd["MEDIAN_TAT_MINS"].tolist()],
            textposition="outside",
            textfont=dict(size=8, color="#003467"),
            hovertemplate="<b>Hour %{x}:00</b>: %{y} min median<extra></extra>",
        ))
        fig_hr.update_layout(**cl(
            height=240,
            xaxis=dict(title="Hour of Day", gridcolor="rgba(0,0,0,0)",
                       tickmode="linear", tick0=0, dtick=1),
            yaxis=dict(title="Median TAT (min)", gridcolor="#EBF3FB"),
            margin=dict(l=0, r=0, t=10, b=30),
        ))
        st.plotly_chart(fig_hr, use_container_width=True, config={"displayModeBar": False})

with v2:
    if not dow_df.empty:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:8px">'
            'Median TAT by Day of Week (V2)</div>',
            unsafe_allow_html=True,
        )
        dd = dow_df.copy().sort_values("DOW_SORT")
        fig_dow = go.Figure(go.Bar(
            x=dd["ORDER_DAY_NAME"].tolist(),
            y=dd["MEDIAN_TAT_MINS"].tolist(),
            marker_color=[
                COLORS["danger"] if v == dd["MEDIAN_TAT_MINS"].max() else COLORS["primary"]
                for v in dd["MEDIAN_TAT_MINS"].tolist()
            ],
            opacity=0.80,
            text=[f"{int(v)}" for v in dd["MEDIAN_TAT_MINS"].tolist()],
            textposition="outside",
            textfont=dict(size=9, color="#003467"),
            hovertemplate="<b>%{x}</b>: %{y} min median<extra></extra>",
        ))
        fig_dow.update_layout(**cl(
            height=240,
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Median TAT (min)", gridcolor="#EBF3FB"),
            margin=dict(l=0, r=0, t=10, b=20),
        ))
        st.plotly_chart(fig_dow, use_container_width=True, config={"displayModeBar": False})

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 6: Operational Conclusion ─────────────────────────────────────────

section_header("6  Operational Conclusion")

_p50_delta = f"{_p50_start} → {_p50_end} min" if _p50_start and _p50_end else "substantially improved"
_p90_delta = f"{_p90_start} → {_p90_end} min" if _p90_start and _p90_end else "substantially improved"
_vol_delta = f"{_vol_chg_pct}%+" if _vol_chg_pct else "40%+"

st.markdown(
    f'<div style="background:#F0FAF9;border-left:4px solid #0BB99F;'
    f'border-radius:6px;padding:20px 24px;margin-top:8px">'
    f'<div style="font-size:14px;font-weight:700;color:#1A3A5C;margin-bottom:14px">'
    f'Pharmacy is currently maintaining throughput under existing demand'
    f'</div>'
    f'<div style="font-size:11px;color:#1A3A5C;line-height:1.8;margin-bottom:16px">'
    f'<b>Evidence:</b><ul style="margin:6px 0 0 0;padding-left:18px">'
    f'<li>Median dispensing time fell from <b>{_p50_delta}</b> (Mar 2025 → Jun 2026)</li>'
    f'<li>Longest waits (1 in 10 orders) reduced from <b>{_p90_delta}</b> over the same period</li>'
    f'<li>Monthly order volume increased by more than <b>{_vol_delta}</b></li>'
    f'<li>Improvements in speed occurred while demand was increasing, not despite stable demand</li>'
    f'</ul>'
    f'</div>'
    f'<div style="font-size:11px;font-weight:600;color:#003467;'
    f'border-top:1px solid #A7D9D4;padding-top:14px">'
    f'Management priority: Maintain current pharmacy performance. '
    f'Operational attention should remain on consultation and diagnostic bottlenecks, '
    f'where patient delays are substantially greater.'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)
