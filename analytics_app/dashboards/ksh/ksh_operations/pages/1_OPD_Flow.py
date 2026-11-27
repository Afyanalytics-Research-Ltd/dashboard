"""
KSH V3 â€” OPD Flow Â· Detail Page
Tabs: Overview (demographics) | Deep Analysis
Deep Analysis: Journey â†’ TAT comparison â†’ Structural proof â†’ Hourly queue â†’ Segments â†’ Volume
Silver: demographics, segments, station TAT dist, pharmacy hourly, volume band.
Gold: monthly aggregates, 8-week station TAT weekly trend.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from ksh_operations.dashboard.theme import apply_theme, render_sidebar, section_header, page_header, cl, COLORS
from ksh_operations.dashboard.queries import (
    q_opd_monthly,

    q_opd_lwbs_gold,
    q_opd_lwbs_hourly_gold,
    q_opd_lwbs_dow_gold,
    q_opd_lwbs_profile,
    q_opd_lab_conversion,
    q_pharmacy_homepage,
    q_opd_station_tat,
    q_opd_peak_baseline_summary,
    q_opd_peak_baseline_tat,
    q_opd_peak_hours,
    q_opd_journey_duration_monthly,
    q_opd_journey_sankey,
)

apply_theme()
render_sidebar(active="opd")
page_header("OPD Flow", subtitle="Outpatient Department Â· Full history")




_AX = dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE"))
def _ax(title="", **kw): return {**_AX, "title": title, **kw}

C_BLUE  = COLORS["primary"]
C_NAVY  = COLORS["dark"]
C_TEAL  = COLORS["secondary"]
C_MUTED = COLORS["muted"]
C_RED   = "#E11D48"

# â”€â”€ Load data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
monthly        = q_opd_monthly()

lwbs_data      = q_opd_lwbs_gold()
lwbs_hourly    = q_opd_lwbs_hourly_gold()
lwbs_dow       = q_opd_lwbs_dow_gold()
lwbs_profile   = q_opd_lwbs_profile()
lab_conv_data  = q_opd_lab_conversion()
pharm_hp       = q_pharmacy_homepage()
station_tat    = q_opd_station_tat()
peak_summary   = q_opd_peak_baseline_summary()
peak_tat       = q_opd_peak_baseline_tat()
peak_hrs       = q_opd_peak_hours()
journey_dur    = q_opd_journey_duration_monthly()
journey_sk_df  = q_opd_journey_sankey()

# â”€â”€ Derived scalars â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
monthly_clean = monthly[monthly["IS_PARTIAL"] == False].copy()
monthly_clean["MONTH_LABEL"] = pd.to_datetime(monthly_clean["MONTH"]).dt.strftime("%b %Y")
monthly_clean["rolling_3m"]  = monthly_clean["TOTAL_VISITS"].rolling(3, min_periods=2).mean()

# Volume growth: first vs last complete month
vol_growth_pct = None
if len(monthly_clean) >= 2:
    v0 = monthly_clean.iloc[0]["TOTAL_VISITS"]
    v1 = monthly_clean.iloc[-1]["TOTAL_VISITS"]
    vol_growth_pct = round((v1 - v0) / v0 * 100) if v0 > 0 else None

# Pharmacy TAT â€” prescription to dispensing (gold RPT_KSH_V3_PHARMACY_TAT)
_pharm = pharm_hp.copy()
_pharm.columns  = [c.upper() for c in _pharm.columns]
_pharm_row      = _pharm.iloc[0] if not _pharm.empty else None
pharm_tat_p50   = float(_pharm_row["TAT_P50"])     if _pharm_row is not None and pd.notna(_pharm_row.get("TAT_P50"))     else None
pharm_tat_delta = float(_pharm_row["TAT_MOM_PCT"]) if _pharm_row is not None and pd.notna(_pharm_row.get("TAT_MOM_PCT")) else None
pharm_tat_mon   = pd.to_datetime(_pharm_row["MONTH"]).strftime("%b %Y") if _pharm_row is not None else "â€”"

# Station TAT â€” weekly grain for bottleneck charts
tat = station_tat.copy()
tat.columns = [c.upper() for c in tat.columns]

# Journey scalars from last 3 complete months
_last3  = monthly_clean.tail(3)
_latest = monthly_clean.iloc[-1] if not monthly_clean.empty else None
visits_per_day  = int(_latest["TOTAL_VISITS"] / 30) if _latest is not None else None
partial_pct_val = _last3["PARTIAL_JOURNEY_PCT"].mean() if "PARTIAL_JOURNEY_PCT" in _last3.columns else None

lwbs_df = lwbs_data.copy()
lwbs_df.columns = [c.upper() for c in lwbs_df.columns]
lwbs_pct_val   = round(lwbs_df.iloc[-1]["LWBS_PCT"], 1) if not lwbs_df.empty else None
lwbs_pct_month = pd.to_datetime(lwbs_df.iloc[-1]["MONTH"]).strftime("%b %Y") if not lwbs_df.empty else None
lab_df = lab_conv_data.copy()
lab_df.columns = [c.upper() for c in lab_df.columns]
lab_referral_pct    = round(lab_df["REFERRAL_PCT"].mean(), 1) if not lab_df.empty else None
lab_fulfillment_pct = round(lab_df["FULFILLMENT_PCT"].mean(), 1) if not lab_df.empty else None

# Prior month for deltas
_prior = monthly_clean.iloc[-2] if len(monthly_clean) >= 2 else None

arrivals_delta   = (visits_per_day - int(_prior["TOTAL_VISITS"] / 30)) \
                   if visits_per_day and _prior is not None else None
lab_ref_delta    = round(float(lab_df["REFERRAL_PCT"].iloc[-1])    - float(lab_df["REFERRAL_PCT"].iloc[-2]),    1) \
                   if len(lab_df) >= 2 else None
lab_ful_delta    = round(float(lab_df["FULFILLMENT_PCT"].iloc[-1]) - float(lab_df["FULFILLMENT_PCT"].iloc[-2]), 1) \
                   if len(lab_df) >= 2 else None
lwbs_delta       = round(float(lwbs_df.iloc[-1]["LWBS_PCT"]) - float(lwbs_df.iloc[-2]["LWBS_PCT"]), 1) \
                   if len(lwbs_df) >= 2 else None

# Card 5 â€” Worst station P50 this week (most recent complete week in tat)
tat.columns = [c.upper() for c in tat.columns]
_this_wk = tat[tat["WEEK_RANK"] == 1] if not tat.empty else pd.DataFrame()
_worst_stn_row   = _this_wk.loc[_this_wk["WAIT_P50"].idxmax()] if not _this_wk.empty else None
worst_stn_name   = _worst_stn_row["STATION"].title() if _worst_stn_row is not None else None
worst_stn_p50    = int(_worst_stn_row["WAIT_P50"])   if _worst_stn_row is not None else None
worst_stn_base   = int(tat[tat["STATION"] == _worst_stn_row["STATION"]]["WAIT_P50"].median()) \
                   if _worst_stn_row is not None else None
worst_stn_delta  = (worst_stn_p50 - worst_stn_base) if worst_stn_p50 and worst_stn_base else None

# Card 6 â€” Admission conversion rate (latest complete month)
conv_latest = round(float(_latest["CONVERSION_RATE"]), 1) \
              if _latest is not None and pd.notna(_latest.get("CONVERSION_RATE")) else None
conv_prior  = round(float(_prior["CONVERSION_RATE"]),  1) \
              if _prior  is not None and pd.notna(_prior.get("CONVERSION_RATE"))  else None
conv_delta  = round(conv_latest - conv_prior, 1) if conv_latest and conv_prior else None

# Card 7 â€” Median journey duration (latest complete month)
_jd = journey_dur.copy()
_jd.columns = [c.upper() for c in _jd.columns]
_jd_clean    = _jd[_jd["IS_PARTIAL"] == False]
journey_p50  = int(_jd_clean.iloc[-1]["JOURNEY_P50_MIN"]) if not _jd_clean.empty else None
_jd_prior    = int(_jd_clean.iloc[-2]["JOURNEY_P50_MIN"]) if len(_jd_clean) >= 2  else None
journey_delta = (journey_p50 - _jd_prior)                 if journey_p50 and _jd_prior else None


# â”€â”€ Shared components â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _delta_html(val, unit="", good="down", label="vs prior mo"):
    """Colored arrow delta. good='down': lower is better. good='up': higher is better. good='neutral': grey."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if good == "neutral":
        color = "#6B8CAE"
    else:
        is_good = (val < 0) if good == "down" else (val > 0)
        color = "#16A34A" if is_good else "#E11D48"
    arrow = "â–¼" if val < 0 else "â–²"
    sign  = "+" if val > 0 else ""
    return (f'<div style="font-size:11px;font-weight:600;color:{color};margin-top:4px">'
            f'{arrow} {sign}{val}{unit} {label}</div>')

def _journey_card(col, label, value, sub=None, delta=None):
    delta_html = "" if delta is None else delta
    sub_html   = "" if sub is None else \
        f'<div style="font-size:10px;color:#6B8CAE;margin-top:5px;line-height:1.4">{sub}</div>'
    col.markdown(
        f'<div style="background:#FFFFFF;border:1px solid #0072CE;border-left:4px solid #0072CE;'
        f'border-radius:8px;padding:14px 16px;text-align:center;min-height:100px">'
        f'<div style="font-size:9px;font-weight:700;color:#0072CE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:#003467">{value}</div>'
        f'{delta_html}'
        f'{sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 1 â€” Patient Journey
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section_header("OPD At a Glance", margin_top=8)

jc1, jc3, jc4, jc5 = st.columns(4)
_journey_card(jc1, "Arrivals / Day",
    f"~{visits_per_day}" if visits_per_day else "â€”",
    "latest complete month",
    delta=_delta_html(arrivals_delta, unit="/day", good="neutral"))
_journey_card(jc3, "Lab Referral Rate",
    f"{lab_referral_pct}%" if lab_referral_pct is not None else "â€”",
    f"referred Â· {lab_fulfillment_pct}% results returned" if lab_fulfillment_pct is not None else "referred",
    delta=_delta_html(lab_ref_delta, unit="pp", good="neutral"))
_journey_card(jc4, "Pharmacy Wait",
    f"{int(pharm_tat_p50)} min" if pharm_tat_p50 else "â€”",
    f"prescription â†’ dispensing Â· P50 Â· {pharm_tat_mon}",
    delta=_delta_html(pharm_tat_delta, unit="%", good="down"))
_journey_card(jc5, "Left Without Seen",
    f"{lwbs_pct_val}%" if lwbs_pct_val is not None else "â€”",
    "left before doctor Â· latest month",
    delta=_delta_html(lwbs_delta, unit="pp", good="down"))

nc1, nc2, nc3 = st.columns(3)
_journey_card(
    nc1, "Worst Station â€” This Week",
    worst_stn_name if worst_stn_name else "â€”",
    f"{worst_stn_p50} min Â· P50 Â· 8-week median: {worst_stn_base} min" if worst_stn_p50 else "P50",
    delta=_delta_html(worst_stn_delta, unit=" min", good="down", label="vs 8-wk median"),
)
_journey_card(
    nc2, "Admission Conversion",
    f"{conv_latest}%" if conv_latest is not None else "â€”",
    "OPD â†’ admitted Â· latest complete month",
    delta=_delta_html(conv_delta, unit="pp", good="up"),
)
_journey_card(
    nc3, "Median Journey Duration",
    f"{journey_p50} min" if journey_p50 else "â€”",
    "first station â†’ last station Â· P50 Â· latest complete month",
    delta=_delta_html(journey_delta, unit=" min", good="down"),
)


# Monthly trendline + day-of-week pattern

fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(
    x=monthly_clean["MONTH_LABEL"], y=monthly_clean["TOTAL_VISITS"],
    name="Monthly visits", mode="lines+markers",
    line=dict(color=C_NAVY, width=2), marker=dict(size=5, color=C_BLUE),
))
fig_vol.add_trace(go.Scatter(
    x=monthly_clean["MONTH_LABEL"], y=monthly_clean["rolling_3m"],
    name="3-mo avg", mode="lines",
    line=dict(color=C_TEAL, width=2, dash="dot"),
))
fig_vol.update_layout(**cl(
    height=260, margin=dict(l=50, r=20, t=10, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=_ax("Monthly visits"),
    xaxis=_ax(tickangle=-45, nticks=12),
))
st.plotly_chart(fig_vol, use_container_width=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 2 â€” Patient Journey Sankey
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if not journey_sk_df.empty:
    _sk   = journey_sk_df.iloc[0]
    _tot  = int(_sk["TOTAL_VISITS"])
    _svc  = int(_sk["SERVICE_COUNT"])
    _lwbs = int(_sk["LWBS_COUNT"])
    _anc  = int(_sk["DIRECT_ANCILLARY_COUNT"])
    _inc  = int(_sk["INCOMPLETE_OPD_COUNT"])
    _adm  = int(_sk["ADMITTED_COUNT"])
    _adm_i = int(_sk["ADMITTED_INPATIENT_COUNT"])
    _adm_d = int(_sk["ADMITTED_DISCHARGED_COUNT"])
    _rxY  = int(_sk["WITH_RX_COUNT"])
    _rxN  = int(_sk["NO_RX_COUNT"])
    _doc  = _svc - _anc - _inc   # intermediate: patients who reached doctor pathway

    section_header("Patient Journey", margin_top=28)

    def _lbl(name, n):
        return f"{name}<br><b>{n:,}</b>  {n/_tot*100:.0f}%"

    # Nodes:
    # 0=All Arrivals  1=LWBS         2=Service Contact
    # 3=Direct Ancillary  4=Triaged No Doctor  5=Doctor Visit
    # 6=Admitted (pass-through)
    # 7=Inpatient  8=Discharged  9=With Rx  10=No Rx
    _node_labels = [
        _lbl("All Arrivals",          _tot),
        _lbl("LWBS",                  _lwbs),
        _lbl("Service Contact",       _svc),
        _lbl("Direct Ancillary",      _anc),
        _lbl("Triaged, No Doctor",    _inc),
        _lbl("Doctor Visit",          _doc),
        _lbl("Admitted",              _adm),
        _lbl("Still Inpatient",       _adm_i),
        _lbl("Admitted â€” Discharged", _adm_d),
        _lbl("With Prescription",     _rxY),
        _lbl("No Prescription",       _rxN),
    ]
    _node_colors = [
        "#003467",  # 0 All Arrivals
        "#E11D48",  # 1 LWBS
        "#0072CE",  # 2 Service Contact
        "#0E9F6E",  # 3 Direct Ancillary
        "#F59E0B",  # 4 Triaged, No Doctor
        "#0072CE",  # 5 Doctor Visit
        "#7C3AED",  # 6 Admitted
        "#7C3AED",  # 7 Still Inpatient
        "#117A65",  # 8 Admitted â€” Discharged
        "#0E9F6E",  # 9 With Prescription
        "#6B8CAE",  # 10 No Prescription
    ]
    # 5 stages:
    # x=0.001 All Arrivals
    # x=0.28  LWBS + Service Contact
    # x=0.56  Direct Ancillary + Triaged No Doctor + Doctor Visit
    # x=0.76  Admitted (pass-through)
    # x=0.999 Inpatient + Discharged + With Rx + No Rx
    _node_x = [0.001, 0.28, 0.28, 0.56, 0.56, 0.56, 0.76, 0.999, 0.999, 0.999, 0.999]
    _node_y = [0.50,  0.10, 0.68, 0.12, 0.35, 0.78, 0.14, 0.04,  0.22,  0.60,  0.88 ]

    # Edges
    _src = [0,     0,     2,    2,    2,    5,    5,    5,    6,      6     ]
    _tgt = [1,     2,     3,    4,    5,    6,    9,    10,   7,      8     ]
    _val = [_lwbs, _svc,  _anc, _inc, _doc, _adm, _rxY, _rxN, _adm_i, _adm_d]
    _link_colors = [
        "rgba(225,29,72,0.22)",    # â†’ LWBS
        "rgba(0,114,206,0.15)",    # â†’ Service Contact
        "rgba(14,159,110,0.22)",   # â†’ Direct Ancillary
        "rgba(245,158,11,0.22)",   # â†’ Triaged, No Doctor
        "rgba(0,114,206,0.15)",    # â†’ Doctor Visit
        "rgba(124,58,237,0.22)",   # â†’ Admitted
        "rgba(14,159,110,0.22)",   # â†’ With Rx
        "rgba(107,140,174,0.22)",  # â†’ No Rx
        "rgba(124,58,237,0.30)",   # â†’ Still Inpatient
        "rgba(17,122,101,0.22)",   # â†’ Admitted Discharged
    ]

    fig_sk = go.Figure(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=16, thickness=20,
            line=dict(color="#FFFFFF", width=0.8),
            label=_node_labels,
            color=_node_colors,
            x=_node_x,
            y=_node_y,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=_src, target=_tgt, value=_val,
            color=_link_colors,
            hovertemplate="%{source.label} â†’ %{target.label}<br>%{value:,} visits<extra></extra>",
        ),
    ))
    fig_sk.update_layout(**cl(
        height=500,
        font=dict(size=13, color="#1E3A55", family="Inter, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=16, b=10),
    ))
    st.plotly_chart(fig_sk, use_container_width=True)
    st.caption("All registered visits Â· all outcomes Â· settled months only Â· Descriptive.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 3 â€” Station TAT comparison
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cur4 = tat[tat["WEEK_RANK"] <= 4].groupby("STATION").agg(p50=("WAIT_P50", "median")).reset_index()
pri4 = tat[tat["WEEK_RANK"] > 4].groupby("STATION").agg(prior_p50=("WAIT_P50", "median")).reset_index()
ranking = cur4.merge(pri4, on="STATION", how="left")
ranking["pct_change"] = (
    (ranking["p50"] - ranking["prior_p50"]) / ranking["prior_p50"] * 100
).where(ranking["prior_p50"].notna())
ranking = ranking.sort_values("p50", ascending=True)

_worst = ranking.iloc[-1] if not ranking.empty else None

section_header("Station Wait Times â€” Last 28 Days", margin_top=28)

bar_colors = [C_RED if s == _worst["STATION"] else C_BLUE for s in ranking["STATION"]] if _worst is not None else [C_BLUE] * len(ranking)
bar_text   = ranking["p50"].apply(lambda x: f"{int(x)} min" if pd.notna(x) else "â€”")

fig_cmp = go.Figure(go.Bar(
    x=ranking["p50"], y=ranking["STATION"],
    orientation="h",
    marker_color=bar_colors,
    text=bar_text, textposition="outside",
    textfont=dict(size=13, color="#003467"),
))
fig_cmp.update_layout(**cl(
    height=240, margin=dict(l=90, r=80, t=10, b=20),
    xaxis=_ax("Median wait (min, last 28 days)", range=[0, 100]),
    yaxis=_ax(), showlegend=False,
))
st.plotly_chart(fig_cmp, use_container_width=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 3b â€” Bottleneck Analysis
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section_header("Bottleneck Analysis â€” 34 Weeks", margin_top=28)

_stn_colors = {
    "Doctor":     C_NAVY,
    "Pharmacy":   C_RED,
    "Laboratory": C_TEAL,
    "Radiology":  C_MUTED,
}
_stations = ["Doctor", "Pharmacy", "Laboratory", "Radiology"]

ta1, ta2, ta3, ta4 = st.columns(4)
for _col, _stn in zip([ta1, ta2, ta3, ta4], _stations):
    _grp = tat[tat["STATION"] == _stn].sort_values("WEEK_START")
    _fig = go.Figure(go.Scatter(
        x=_grp["WEEK_START"], y=_grp["WAIT_P50"],
        mode="lines+markers",
        line=dict(color=_stn_colors[_stn], width=2),
        marker=dict(size=4),
        hovertemplate="Week: %{x}<br>P50: %{y} min<extra></extra>",
    ))
    _fig.update_layout(**cl(
        height=200,
        title=dict(text=_stn, font=dict(size=11, color=_stn_colors[_stn]), x=0.5, xanchor="center"),
        margin=dict(l=40, r=10, t=36, b=40),
        xaxis=_ax(tickangle=-45, nticks=4),
        yaxis=_ax("min"),
        showlegend=False,
    ))
    _col.plotly_chart(_fig, use_container_width=True)

st.caption("Flat high = chronic bottleneck Â· rising slope = worsening Â· dip then rise = recovered then regressed. 34 settled weeks Â· Indicative.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 5 â€” LWBS (Left Without Being Seen)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lh = lwbs_hourly.copy();  lh.columns = [c.upper() for c in lh.columns]
ld = lwbs_dow.copy();     ld.columns = [c.upper() for c in ld.columns]
lp = lwbs_profile.copy();  lp.columns = [c.upper() for c in lp.columns]
pay = lp[lp["DIM"] == "payment"].copy().sort_values("LWBS_PCT", ascending=True)

_lwbs_header = (
    f"{lwbs_pct_val}% Left Without Being Seen â€” {lwbs_pct_month}"
    if lwbs_pct_val else "LWBS â€” Left Without Being Seen"
)
section_header(_lwbs_header, margin_top=28)

# Monthly trend scalars
cur_pct  = lwbs_df.iloc[-1]["LWBS_PCT"] if not lwbs_df.empty else None
prev_pct = lwbs_df.iloc[-2]["LWBS_PCT"] if len(lwbs_df) >= 2 else None
mom_delta = round(cur_pct - prev_pct, 1) if cur_pct is not None and prev_pct is not None else None
lwbs_df["MONTH_LABEL"] = pd.to_datetime(lwbs_df["MONTH"]).dt.strftime("%b %Y")

# Row 1: monthly trend chart + day-of-week breakdown
col_trend, col_dow = st.columns([2, 2])

with col_trend:
    bar_colors_m = [C_RED if i == len(lwbs_df) - 1 else C_MUTED for i in range(len(lwbs_df))]
    fig_trend = go.Figure(go.Bar(
        x=lwbs_df["MONTH_LABEL"],
        y=lwbs_df["LWBS_PCT"],
        marker_color=bar_colors_m,
        text=lwbs_df["LWBS_PCT"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
    ))
    if mom_delta is not None:
        arrow = "â–²" if mom_delta > 0 else "â–¼"
        color = C_RED if mom_delta > 0 else "#16A34A"
        fig_trend.add_annotation(
            x=lwbs_df["MONTH_LABEL"].iloc[-1],
            y=cur_pct,
            text=f'<span style="color:{color}">{arrow} {abs(mom_delta):.1f}pp vs prior</span>',
            showarrow=False,
            yshift=28,
            font=dict(size=10),
        )
    fig_trend.update_layout(**cl(
        height=240, margin=dict(l=20, r=20, t=36, b=30),
        title=dict(text="Monthly Trend", font=dict(size=12, color="#003467"), x=0, xanchor="left"),
        xaxis=_ax(),
        yaxis=_ax("LWBS %"),
        showlegend=False,
    ))
    st.plotly_chart(fig_trend, use_container_width=True)

with col_dow:
    if not ld.empty:
        avg_dow = ld["LWBS_PCT"].mean()
        dow_colors = [C_RED if p >= avg_dow else C_MUTED for p in ld["LWBS_PCT"]]
        fig_dow = go.Figure(go.Bar(
            x=ld["DAY_OF_WEEK"],
            y=ld["LWBS_PCT"],
            marker_color=dow_colors,
            text=ld["LWBS_PCT"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>LWBS: %{y:.1f}%<extra></extra>",
        ))
        fig_dow.update_layout(**cl(
            height=240, margin=dict(l=20, r=20, t=36, b=30),
            title=dict(text="By Day of Week", font=dict(size=12, color="#003467"), x=0, xanchor="left"),
            xaxis=_ax(),
            yaxis=_ax("LWBS %"),
            showlegend=False,
        ))
        st.plotly_chart(fig_dow, use_container_width=True)

# Row 2: hourly breakdown + payment mode
col_hr, col_pay = st.columns([3, 1.5])

with col_hr:
    avg_lwbs = lh["LWBS_PCT"].mean()
    hr_colors = [C_RED if p >= avg_lwbs else C_MUTED for p in lh["LWBS_PCT"]]
    fig_hr = go.Figure(go.Bar(
        x=[f"{int(h):02d}:00" for h in lh["ARRIVAL_HOUR"]],
        y=lh["LWBS_PCT"],
        marker_color=hr_colors,
        text=lh["LWBS_PCT"].apply(lambda x: f"{x:.0f}%"),
        textposition="outside",
    ))
    fig_hr.add_hline(
        y=avg_lwbs, line_dash="dot", line_color="#6B8CAE",
        annotation_text=f"avg {avg_lwbs:.0f}%",
        annotation_position="top left",
        annotation_font_size=10,
    )
    fig_hr.update_layout(**cl(
        height=260, margin=dict(l=20, r=20, t=36, b=40),
        title=dict(text="By Arrival Hour", font=dict(size=12, color="#003467"), x=0, xanchor="left"),
        xaxis=_ax("Arrival hour", tickangle=-45),
        yaxis=_ax("LWBS %"),
        showlegend=False,
    ))
    st.plotly_chart(fig_hr, use_container_width=True)

with col_pay:
    pay_colors = [C_RED if p == pay["LWBS_PCT"].max() else C_MUTED for p in pay["LWBS_PCT"]]
    fig_pay = go.Figure(go.Bar(
        x=pay["LWBS_PCT"], y=pay["SEGMENT"].str.capitalize(),
        orientation="h",
        marker_color=pay_colors,
        text=pay["LWBS_PCT"].apply(lambda x: f"{x:.0f}%"),
        textposition="outside",
    ))
    fig_pay.update_layout(**cl(
        height=260, margin=dict(l=80, r=60, t=36, b=40),
        title=dict(text="By Payment Mode", font=dict(size=12, color="#003467"), x=0, xanchor="left"),
        xaxis=_ax("LWBS %"),
        yaxis={**_AX, "autorange": "reversed"},
        showlegend=False,
    ))
    st.plotly_chart(fig_pay, use_container_width=True)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SECTION 6 â€” Peak Hours Analysis
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
section_header("When Is OPD Busiest?", margin_top=28)

_ph = peak_hrs.copy()
_ph.columns = [c.upper() for c in _ph.columns]
if not _ph.empty:
    _dow_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    _ph_pivot  = _ph.pivot_table(index="DOW_NAME", columns="ARRIVAL_HOUR",
                                  values="AVG_VISITS", fill_value=0)
    _ph_pivot  = _ph_pivot.reindex([d for d in _dow_order if d in _ph_pivot.index])
    for h in range(24):
        if h not in _ph_pivot.columns:
            _ph_pivot[h] = 0
    _ph_pivot  = _ph_pivot[sorted(_ph_pivot.columns)]

    _peak_row  = _ph.loc[_ph["AVG_VISITS"].idxmax()]
    _peak_desc = f"{_peak_row['DOW_NAME']} {int(_peak_row['ARRIVAL_HOUR']):02d}:00"

    fig_ph = go.Figure(go.Heatmap(
        z=_ph_pivot.values,
        x=[f"{h:02d}:00" for h in _ph_pivot.columns],
        y=_ph_pivot.index.tolist(),
        colorscale=[[0, "#EBF3FB"], [0.5, "#0072CE"], [1, "#003467"]],
        hovertemplate="<b>%{y} %{x}</b><br>Avg visits: %{z:.1f}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Avg visits", tickfont=dict(size=9, color="#6B8CAE"),
                      titlefont=dict(size=9, color="#6B8CAE")),
    ))
    fig_ph.update_layout(**cl(
        height=280,
        margin=dict(l=90, r=20, t=10, b=50),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#6B8CAE"),
                   tickmode="array",
                   tickvals=[f"{h:02d}:00" for h in range(0, 24, 2)],
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]),
        yaxis=dict(tickfont=dict(size=10, color="#6B8CAE")),
    ))
    st.plotly_chart(fig_ph, use_container_width=True)
    st.caption(f"Trailing 12 months Â· busiest slot: {_peak_desc} Â· Descriptive.")

_ps = peak_summary.copy()
_ps.columns = [c.upper() for c in _ps.columns]
_ps = _ps.set_index("BAND")

_pt = peak_tat.copy()
_pt.columns = [c.upper() for c in _pt.columns]

if "peak" in _ps.index and "baseline" in _ps.index:
    _base_journey  = int(_ps.loc["baseline", "JOURNEY_P50"])
    _peak_journey  = int(_ps.loc["peak",     "JOURNEY_P50"])
    _journey_chg   = round((_peak_journey - _base_journey) / _base_journey * 100)

    _base_conv     = float(_ps.loc["baseline", "CONVERSION_PCT"])
    _peak_conv     = float(_ps.loc["peak",     "CONVERSION_PCT"])
    _conv_chg      = round(_peak_conv - _base_conv, 1)

    section_header("Peak vs Baseline Impact", margin_top=28)

    def _comparison_card(col, label, base_val, peak_val, change, change_unit="%", good="down"):
        sign  = "+" if change > 0 else ""
        color = "#16A34A" if (change < 0 and good == "down") or (change > 0 and good == "up") else "#E11D48"
        col.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #0072CE;border-left:4px solid #0072CE;'
            f'border-radius:8px;padding:14px 16px;min-height:110px">'
            f'<div style="font-size:9px;font-weight:700;color:#0072CE;text-transform:uppercase;'
            f'letter-spacing:1.5px;margin-bottom:10px">{label}</div>'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-end">'
            f'<div style="text-align:center">'
            f'<div style="font-size:9px;color:#6B8CAE;margin-bottom:3px">BASELINE</div>'
            f'<div style="font-size:22px;font-weight:800;color:#003467">{base_val}</div>'
            f'</div>'
            f'<div style="text-align:center">'
            f'<div style="font-size:9px;color:#6B8CAE;margin-bottom:3px">PEAK</div>'
            f'<div style="font-size:22px;font-weight:800;color:#003467">{peak_val}</div>'
            f'</div>'
            f'<div style="text-align:center">'
            f'<div style="font-size:9px;color:#6B8CAE;margin-bottom:3px">CHANGE</div>'
            f'<div style="font-size:22px;font-weight:800;color:{color}">{sign}{change}{change_unit}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    pc1, pc2 = st.columns(2)
    _comparison_card(pc1, "Median Journey Duration",
                     f"{_base_journey} min", f"{_peak_journey} min",
                     _journey_chg, change_unit="%", good="down")
    _comparison_card(pc2, "Admission Conversion Rate",
                     f"{_base_conv}%", f"{_peak_conv}%",
                     _conv_chg, change_unit="pp", good="up")

    if not _pt.empty:
        _stn_order = ["doctor", "laboratory", "pharmacy", "radiology"]
        _pt_base   = _pt[_pt["BAND"] == "baseline"].set_index("STATION")["WAIT_P50"]
        _pt_peak   = _pt[_pt["BAND"] == "peak"].set_index("STATION")["WAIT_P50"]
        _stns      = [s for s in _stn_order if s in _pt_base.index and s in _pt_peak.index]

        fig_pvb = go.Figure()
        fig_pvb.add_trace(go.Bar(
            name="Baseline",
            x=[s.title() for s in _stns],
            y=[_pt_base[s] for s in _stns],
            marker_color="#C7D9EE",
            text=[f"{int(_pt_base[s])} min" for s in _stns],
            textposition="outside",
            hovertemplate="<b>%{x}</b> Â· Baseline<br>P50: %{y} min<extra></extra>",
        ))
        fig_pvb.add_trace(go.Bar(
            name="Peak",
            x=[s.title() for s in _stns],
            y=[_pt_peak[s] for s in _stns],
            marker_color=C_NAVY,
            text=[f"{int(_pt_peak[s])} min" for s in _stns],
            textposition="outside",
            hovertemplate="<b>%{x}</b> Â· Peak<br>P50: %{y} min<extra></extra>",
        ))
        fig_pvb.update_layout(**cl(
            barmode="group",
            height=300,
            margin=dict(l=50, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=_ax("Median wait (min)"),
            xaxis=_ax(),
        ))
        st.plotly_chart(fig_pvb, use_container_width=True)
        st.caption("Peak â‰¥8 arrivals/hr Â· Baseline â‰¤4 Â· trailing 12 months Â· Indicative.")


