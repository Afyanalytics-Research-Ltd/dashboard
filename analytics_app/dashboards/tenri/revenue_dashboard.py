"""
app_revenue.py
--------------
PharmaPlus · Revenue Intelligence

Live dashboard backed by Snowflake. SQL queries live in queries.py. Predictive
models (forecast, anomalies, RFM, drivers) live in predictive.py and run on
the dataframes returned from Snowflake. What-if scenario arithmetic lives in
simulator.py and is applied on top of live baselines.

Run with:
    streamlit run app_revenue.py
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath('__file__')),
        "analytics_app",
        "dashboards",
        "ksh",
        "revenue_module"
    )
)


from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ksh.revenue_module.data_layer as dl
import ksh.revenue_module.predictive as predictive
import ksh.revenue_module.whatif as whatif


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

PAGE_TITLE = "Revenue Intelligence"

st.set_page_config(
    page_title=f"PharmaPlus · {PAGE_TITLE}",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── SHARED THEME ────────────────────────────────────────────────────────────

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


# ─── HELPERS ────────────────────────────────────────────────────────────────

def fmt_ksh(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if abs(v) >= 1_000_000_000:
        return f"KSh {v/1_000_000_000:.2f}B"
    if abs(v) >= 1_000_000:
        return f"KSh {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"KSh {v/1_000:.1f}K"
    return f"KSh {v:.0f}"


def fmt_pct(v, digits=1):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{digits}f}%"


def kpi_card(label, value, sub="", color="#003467"):
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#F4F8FC 0%,#FFFFFF 100%);'
        f'border:1px solid #D6E4F0;border-radius:10px;padding:18px 16px;'
        f'box-shadow:0 1px 2px rgba(0,52,103,0.04)">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)


def info_card(text, border_color="#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;'
        f'border-left:3px solid {border_color};border-radius:4px;'
        f'font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True,
    )


CHART_LAYOUT = dict(
    paper_bgcolor="#fff",
    plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

COLORS = {
    "primary":  "#0072CE",
    "navy":     "#003467",
    "success":  "#0BB99F",
    "warning":  "#D97706",
    "danger":   "#E11D48",
    "muted":    "#6B8CAE",
    "purple":   "#7F77DD",
    "pink":     "#D4537E",
    "coral":    "#D85A30",
    "green":    "#1D9E75",
}

PALETTE = [
    "#0072CE", "#0BB99F", "#7F77DD", "#D97706", "#D4537E",
    "#1D9E75", "#003467", "#D85A30", "#6B8CAE", "#E11D48",
]


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# ─── SIDEBAR ────────────────────────────────────────────────────────────────

with st.sidebar:
    try:
        st.image("assets/pharmaplus_logo.png", width=160)
    except Exception:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#0072CE;'
            'padding:8px 0 16px">PharmaPlus</div>',
            unsafe_allow_html=True,
        )

    section_header("Date range")
    today = date.today()
    quick_pick = st.selectbox(
        "Period",
        ["Last 30 days", "Last 90 days", "Last 6 months", "Last 12 months",
         "Last 24 months", "Custom"],
        index=3,
    )
    end = today
    spans = {
        "Last 30 days": 30, "Last 90 days": 90, "Last 6 months": 183,
        "Last 12 months": 365, "Last 24 months": 730,
    }
    if quick_pick != "Custom":
        start = end - timedelta(days=spans[quick_pick])
    else:
        start = st.date_input("Start", end - timedelta(days=365))
        end   = st.date_input("End",   today)

    section_header("Filters", margin_top=12)
    try:
        clinics_df = dl.list_clinics()
        clinic_options = clinics_df["clinic_name"].tolist() if len(clinics_df) else []
    except Exception:
        clinics_df = pd.DataFrame()
        clinic_options = []

    selected_clinics = st.multiselect("Branch", clinic_options, default=clinic_options)
    clinic_filter = ""
    if selected_clinics and len(clinic_options) and len(selected_clinics) < len(clinic_options):
        ids = clinics_df.loc[clinics_df["clinic_name"].isin(selected_clinics), "clinic_id"].tolist()
        if ids:
            clinic_filter = f"AND v.CLINIC IN ({','.join(str(i) for i in ids)})"

    forecast_horizon = st.slider("Forecast horizon (days)", 30, 180, 90, step=30)

    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
    section_header("Data source")
    st.markdown(
        '<div style="font-size:11px;color:#6B8CAE;line-height:1.6">'
        'Live Snowflake · auto-refreshes every 15 min<br>'
        'Schema: <span style="color:#003467;font-weight:600">PHARMAPLUS_PROD</span></div>',
        unsafe_allow_html=True,
    )


# ─── PAGE HEADER ────────────────────────────────────────────────────────────

st.markdown(
    f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    f'text-transform:uppercase;color:#0072CE;margin-bottom:16px">'
    f'PharmaPlus · {PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)


# ─── DATA FETCH ─────────────────────────────────────────────────────────────

start_s = start.strftime("%Y-%m-%d")
end_s   = (end + timedelta(days=1)).strftime("%Y-%m-%d")    # exclusive

# All Snowflake calls happen here. If the connection fails, we surface a
# clear error and stop — there is no silent fallback.
try:
    daily        = dl.daily_revenue(start_s, end_s, clinic_filter)
    sl_monthly   = dl.revenue_by_service_line(start_s, end_s)
    pay_mix_df   = dl.payment_mode_mix(start_s, end_s)
    payer_df     = dl.payer_performance(start_s, end_s)
    rfm_df       = dl.patient_rfm(start_s, end_s)
    items_df     = dl.top_items(start_s, end_s)
    hourly_df    = dl.hourly_heatmap(start_s, end_s)
    cohort_df    = dl.cohort_retention(start_s, end_s)
    docs_df      = dl.doctor_productivity(start_s, end_s)
    leak_df      = dl.leakage(start_s, end_s)
    margin_df    = dl.inventory_margin(start_s, end_s)
    rejection_df = dl.claim_rejection(start_s, end_s)
    concent_df   = dl.revenue_concentration(start_s, end_s)
    arpv_df      = dl.arpv_trend(start_s, end_s)
    risk_df      = dl.revenue_at_risk(start_s, end_s)
    gp_df        = dl.gross_profit_weekly(start_s, end_s)
except Exception as exc:
    st.error(
        "Could not connect to Snowflake. Verify SNOWFLAKE_ACCOUNT, "
        "SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_PATH, and warehouse access."
    )
    st.exception(exc)
    st.stop()


# ─── KPI ROW ────────────────────────────────────────────────────────────────

total_revenue   = daily["gross_amount"].sum()
total_receipts  = daily["receipt_count"].sum()
unique_pat      = daily["unique_patients"].sum()
avg_daily       = daily.groupby("revenue_date")["gross_amount"].sum().mean()
collected_total = payer_df["collected"].sum() if len(payer_df) else 0
billed_total    = payer_df["billed"].sum()    if len(payer_df) else 0
collection_pct  = 100 * collected_total / billed_total if billed_total else 0

# month-over-month delta on total revenue (last full month vs prior)
by_day = daily.groupby("revenue_date", as_index=False)["gross_amount"].sum()
by_day["revenue_date"] = pd.to_datetime(by_day["revenue_date"])
by_day = by_day.sort_values("revenue_date")
by_day["m"] = by_day["revenue_date"].dt.to_period("M")
mom = by_day.groupby("m")["gross_amount"].sum()
mom_delta = None
if len(mom) >= 2:
    mom_delta = (mom.iloc[-1] - mom.iloc[-2]) / mom.iloc[-2] * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    sub = (f"{'▲' if (mom_delta or 0) >= 0 else '▼'} {abs(mom_delta or 0):.1f}% MoM"
           if mom_delta is not None else "—")
    kpi_card("Gross revenue", fmt_ksh(total_revenue), sub, COLORS["primary"])
with c2:
    kpi_card("Avg daily revenue", fmt_ksh(avg_daily),
             f"{int(total_receipts):,} receipts in window", COLORS["success"])
with c3:
    kpi_card("Unique patients", f"{int(unique_pat):,}",
             f"{(total_revenue / max(unique_pat, 1)):,.0f} KSh ARPU", COLORS["warning"])
with c4:
    kpi_card("Collection rate", fmt_pct(collection_pct),
             f"Outstanding {fmt_ksh(billed_total - collected_total)}",
             COLORS["danger"] if collection_pct < 80 else COLORS["green"])

st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)


# ─── TABS ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "◉  Revenue Pulse",
    "△  Forecast & Risk",
    "◇  Payer & Patient Mix",
    "∑  KPI Scorecard",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Revenue Pulse
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    section_header("Daily revenue · 7-day rolling average")

    by_day_chart = (
        daily.groupby("revenue_date", as_index=False)["gross_amount"].sum()
              .sort_values("revenue_date")
    )
    by_day_chart["revenue_date"] = pd.to_datetime(by_day_chart["revenue_date"])
    by_day_chart["roll7"] = by_day_chart["gross_amount"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=by_day_chart["revenue_date"], y=by_day_chart["gross_amount"],
        mode="lines", name="Daily",
        line=dict(color=hex_to_rgba(COLORS["primary"], 0.25), width=1),
        fill="tozeroy", fillcolor=hex_to_rgba(COLORS["primary"], 0.06),
        hovertemplate="<b>%{x|%a %d %b %Y}</b><br>%{y:,.0f} KSh<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=by_day_chart["revenue_date"], y=by_day_chart["roll7"],
        mode="lines", name="7-day avg",
        line=dict(color=COLORS["primary"], width=2.5, shape="spline", smoothing=0.6),
        hovertemplate="<b>%{x|%a %d %b %Y}</b><br>7-day avg: %{y:,.0f} KSh<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, height=320, showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    col_l, col_r = st.columns([1.2, 1], gap="large")

    # ── Service line treemap
    with col_l:
        section_header("Service line mix · selected period")
        sl_total = (sl_monthly.groupby("service_line", as_index=False)["gross_revenue"]
                    .sum().sort_values("gross_revenue", ascending=False))
        if len(sl_total):
            sl_total["share"] = 100 * sl_total["gross_revenue"] / sl_total["gross_revenue"].sum()
            fig = go.Figure(go.Treemap(
                labels=sl_total["service_line"],
                parents=[""] * len(sl_total),
                values=sl_total["gross_revenue"],
                textinfo="label+percent parent",
                texttemplate="<b>%{label}</b><br>%{percentParent}<br>%{value:,.0f}",
                hovertemplate="<b>%{label}</b><br>%{value:,.0f} KSh<br>%{percentParent}<extra></extra>",
                marker=dict(
                    colors=[PALETTE[i % len(PALETTE)] for i in range(len(sl_total))],
                    line=dict(color="#fff", width=2),
                ),
                textfont=dict(family="Montserrat", color="#fff", size=12),
            ))
            fig.update_layout(**CHART_LAYOUT, height=360, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No service-line data for this window.")

    # ── Branch ranking horizontal bar
    with col_r:
        section_header("Branch ranking · period total")
        br_total = (branches_df.groupby(["clinic_id", "clinic_name"], as_index=False)["revenue"]
                    .sum().sort_values("revenue"))
        if len(br_total):
            fig = go.Figure(go.Bar(
                x=br_total["revenue"],
                y=br_total["clinic_name"],
                orientation="h",
                marker=dict(
                    color=br_total["revenue"],
                    colorscale=[[0, "#D6E4F0"], [1, COLORS["primary"]]],
                    line=dict(width=0),
                ),
                text=[fmt_ksh(v) for v in br_total["revenue"]],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["navy"]),
                hovertemplate="<b>%{y}</b><br>%{x:,.0f} KSh<extra></extra>",
            ))
            fig.update_layout(
                **{**CHART_LAYOUT, "margin": dict(l=0, r=40, t=10, b=20)},
                height=360, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Hourly heatmap + payment mode area
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        section_header("Demand heatmap · hour × weekday")
        if len(hourly_df):
            order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            pivot = (hourly_df.pivot_table(index="day_name", columns="hour_of_day",
                                           values="revenue", aggfunc="sum")
                     .reindex(order))
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"{h:02d}" for h in pivot.columns],
                y=pivot.index,
                colorscale=[[0, "#F4F8FC"], [0.5, "#7FB1E0"], [1, COLORS["navy"]]],
                hovertemplate="<b>%{y} %{x}:00</b><br>%{z:,.0f} KSh<extra></extra>",
                showscale=False,
            ))
            fig.update_layout(
                **{**CHART_LAYOUT, "margin": dict(l=0, r=0, t=10, b=20)},
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section_header("Payment mode mix · monthly")
        if len(pay_mix_df):
            modes = ["cash", "mpesa", "card", "cheque", "jambopay", "account"]
            modes = [m for m in modes if m in pay_mix_df.columns]
            fig = go.Figure()
            cum = np.zeros(len(pay_mix_df))
            for i, m in enumerate(modes):
                vals = pay_mix_df[m].fillna(0).values
                fig.add_trace(go.Scatter(
                    x=pay_mix_df["revenue_month"], y=vals,
                    mode="lines", name=m.title(),
                    stackgroup="one", line=dict(width=0.5, color=PALETTE[i]),
                    fillcolor=hex_to_rgba(PALETTE[i], 0.85),
                    hovertemplate=f"<b>{m.title()}</b><br>"
                                  "%{x|%b %Y}<br>%{y:,.0f} KSh<extra></extra>",
                ))
                cum += vals
            fig.update_layout(**CHART_LAYOUT, height=300, hovermode="x unified",
                              legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

    # ── Top items Pareto
    section_header("Top revenue items · Pareto", margin_top=8)
    if len(items_df):
        top = items_df.head(20).copy()
        top["cum_share"] = 100 * top["gross_revenue"].cumsum() / items_df["gross_revenue"].sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top["item_name"], y=top["gross_revenue"],
            marker=dict(color=COLORS["primary"], line=dict(width=0)),
            name="Revenue", yaxis="y",
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} KSh<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=top["item_name"], y=top["cum_share"], mode="lines+markers",
            line=dict(color=COLORS["coral"], width=2.5, shape="spline"),
            marker=dict(size=6, color=COLORS["coral"]),
            yaxis="y2", name="Cumulative %",
            hovertemplate="<b>%{x}</b><br>Cumulative %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#6B8CAE")),
            yaxis=dict(title="Revenue (KSh)", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                        range=[0, 105], gridcolor="rgba(0,0,0,0)",
                        tickfont=dict(size=10, color=COLORS["coral"])),
            height=380, showlegend=False, margin=dict(l=0, r=0, t=10, b=80),
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast & Risk (predictive)
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    section_header(f"Forecast · next {forecast_horizon} days")

    # Build a single revenue series for the forecaster
    series = (daily.groupby("revenue_date", as_index=False)["gross_amount"].sum()
              .rename(columns={"gross_amount": "revenue"}))
    series["revenue_date"] = pd.to_datetime(series["revenue_date"])

    if len(series) >= 60:
        fc = predictive.forecast_revenue(series, horizon_days=forecast_horizon)
        hist = fc[fc["segment"] == "history"]
        fut  = fc[fc["segment"] == "forecast"]

        fig = go.Figure()
        # 80% interval band on the forecast
        fig.add_trace(go.Scatter(
            x=fut["date"].tolist() + fut["date"].tolist()[::-1],
            y=fut["upper_80"].tolist() + fut["lower_80"].tolist()[::-1],
            fill="toself", fillcolor=hex_to_rgba(COLORS["primary"], 0.12),
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["actual"], mode="lines", name="Actual",
            line=dict(color=COLORS["primary"], width=2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=fut["date"], y=fut["predicted"], mode="lines", name="Forecast",
            line=dict(color=COLORS["coral"], width=2.5, dash="dot", shape="spline"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**CHART_LAYOUT, height=340, hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

        forecast_total = fut["predicted"].sum()
        info_card(
            f"Projected revenue over next {forecast_horizon} days: "
            f"<b>{fmt_ksh(forecast_total)}</b> "
            f"(80% confidence band {fmt_ksh(fut['lower_80'].sum())} – "
            f"{fmt_ksh(fut['upper_80'].sum())}). "
            "Model: Ridge regression over calendar + lag features.",
            COLORS["primary"],
        )
    else:
        st.info("Need at least 60 days of history to fit a forecast.")

    # ── Anomaly detection
    col_l, col_r = st.columns([1.2, 1], gap="large")
    with col_l:
        section_header("Anomaly detection · Isolation Forest", margin_top=8)
        if len(series) >= 30:
            anom = predictive.detect_anomalies(series.rename(columns={"revenue_date": "revenue_date"}))
            normal = anom[~anom["is_anomaly"]]
            outliers = anom[anom["is_anomaly"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=normal["revenue_date"], y=normal["revenue"], mode="lines",
                name="Normal", line=dict(color=COLORS["muted"], width=1.5),
                hovertemplate="%{x|%d %b}<br>%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=outliers["revenue_date"], y=outliers["revenue"], mode="markers",
                name="Anomaly",
                marker=dict(color=COLORS["danger"], size=10, line=dict(color="#fff", width=1.5)),
                hovertemplate="<b>Anomaly</b><br>%{x|%d %b}<br>%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, height=300,
                              legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)
            if len(outliers):
                worst = outliers.nsmallest(3, "revenue")
                lines = "; ".join(
                    f"{pd.to_datetime(d).strftime('%d %b %Y')} ({fmt_ksh(v)})"
                    for d, v in zip(worst["revenue_date"], worst["revenue"])
                )
                info_card(f"<b>{len(outliers)} anomalous days flagged.</b> Worst dips: {lines}.",
                          COLORS["danger"])

    with col_r:
        section_header("Revenue drivers · feature importance", margin_top=8)
        if len(series) >= 30:
            drivers = predictive.revenue_drivers(series)
            drivers = drivers.sort_values("importance")
            fig = go.Figure(go.Bar(
                x=drivers["importance"], y=drivers["label"],
                orientation="h",
                marker=dict(color=drivers["importance"],
                            colorscale=[[0, "#D6E4F0"], [1, COLORS["purple"]]],
                            line=dict(width=0)),
                hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, height=300,
                              margin=dict(l=0, r=20, t=10, b=20),
                              xaxis=dict(title="", showgrid=True, gridcolor="#EBF3FB"))
            st.plotly_chart(fig, use_container_width=True)

    # ── AR ageing & at-risk
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        section_header("Revenue at risk · AR ageing", margin_top=8)
        if len(risk_df):
            fig = go.Figure()
            risk_colors = [COLORS["green"], COLORS["primary"], COLORS["warning"],
                           COLORS["coral"], COLORS["danger"]]
            fig.add_trace(go.Bar(
                x=risk_df["age_bucket"], y=risk_df["at_risk"],
                marker=dict(color=risk_colors[:len(risk_df)], line=dict(width=0)),
                name="At risk",
                hovertemplate="<b>%{x}</b><br>%{y:,.0f} KSh<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=risk_df["age_bucket"], y=risk_df["expected_collection"],
                mode="lines+markers", name="Expected collection",
                line=dict(color=COLORS["navy"], width=2, dash="dash"),
                marker=dict(size=8, color=COLORS["navy"]),
                hovertemplate="<b>%{x}</b><br>Expected %{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, height=300,
                              legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

    # ── What-if scenarios (uses simulator.py — not data simulation)
    with col_b:
        section_header("What-if levers · projected uplift", margin_top=8)
        baseline_avg = avg_daily if not np.isnan(avg_daily) else 0
        whatif_df = whatif.simulate_levers(baseline_avg, horizon_days=forecast_horizon)
        if len(whatif_df):
            base_value = baseline_avg * forecast_horizon
            fig = go.Figure(go.Waterfall(
                name="What-if",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(whatif_df) + ["total"],
                x=["Baseline"] + whatif_df["lever"].tolist() + ["With all levers"],
                text=[fmt_ksh(base_value)]
                     + [f"+{p:.1f}%" for p in whatif_df["uplift_pct"]]
                     + [fmt_ksh(whatif_df["cumulative"].iloc[-1])],
                y=[base_value] + whatif_df["uplift_value"].tolist() + [0],
                textposition="outside",
                textfont=dict(size=9, color=COLORS["navy"]),
                connector={"line": {"color": "#D6E4F0"}},
                increasing={"marker": {"color": COLORS["success"]}},
                decreasing={"marker": {"color": COLORS["danger"]}},
                totals={"marker": {"color": COLORS["primary"]}},
                hovertemplate="<b>%{x}</b><br>%{y:,.0f} KSh<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, height=320,
                              xaxis=dict(tickangle=-25, tickfont=dict(size=9, color="#6B8CAE")),
                              yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")))
            st.plotly_chart(fig, use_container_width=True)
            info_card(
                f"If all levers hit, {forecast_horizon}-day revenue rises to "
                f"<b>{fmt_ksh(whatif_df['cumulative'].iloc[-1])}</b> from "
                f"{fmt_ksh(base_value)}.",
                COLORS["success"],
            )

    # ── Two-lever elasticity heatmap
    section_header("Two-lever sensitivity · ARPV × Volume", margin_top=8)
    if not np.isnan(avg_daily):
        grid = whatif.elasticity_grid(avg_daily, horizon_days=forecast_horizon)
        pivot = grid.pivot(index="arpv_delta_pct", columns="volume_delta_pct",
                           values="projected_revenue")
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0, "#FBE6EB"], [0.5, "#F4F8FC"], [1, hex_to_rgba(COLORS["success"], 0.85)]],
            hovertemplate="ARPV %{y:+.1f}% · Volume %{x:+.1f}%<br>%{z:,.0f} KSh<extra></extra>",
            colorbar=dict(title="KSh", tickfont=dict(size=9, color="#6B8CAE")),
        ))
        fig.update_layout(**CHART_LAYOUT, height=300,
                          xaxis=dict(title="Visit volume Δ %", tickfont=dict(size=10, color="#6B8CAE")),
                          yaxis=dict(title="ARPV Δ %", tickfont=dict(size=10, color="#6B8CAE")))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Payer & Patient Mix (qualitative + segmentation)
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    section_header("Payer concentration · revenue share & cumulative")
    if len(concent_df):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=concent_df["payer"], y=concent_df["share_pct"],
            name="Share %", marker=dict(color=COLORS["primary"], line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}% share<extra></extra>",
        ))
        if "cumulative_pct" in concent_df.columns:
            fig.add_trace(go.Scatter(
                x=concent_df["payer"], y=concent_df["cumulative_pct"],
                mode="lines+markers", name="Cumulative %",
                line=dict(color=COLORS["coral"], width=2.5, shape="spline"),
                marker=dict(size=7, color=COLORS["coral"]),
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Cumulative %{y:.1f}%<extra></extra>",
            ))
        hhi = float(concent_df["hhi_index"].iloc[0]) if "hhi_index" in concent_df.columns else None
        fig.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            xaxis=dict(tickangle=-25, tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Share %", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                        range=[0, 105], gridcolor="rgba(0,0,0,0)",
                        tickfont=dict(size=10, color=COLORS["coral"])),
            height=320, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        if hhi is not None:
            verdict = ("HIGHLY concentrated" if hhi > 2500
                       else "moderately concentrated" if hhi > 1500
                       else "competitive (well-diversified)")
            info_card(
                f"Herfindahl-Hirschman Index = <b>{hhi:.0f}</b> · payer mix is {verdict}. "
                "Above 2500 indicates dangerous dependence on a single payer.",
                COLORS["warning"] if hhi > 1500 else COLORS["green"],
            )

    # ── Payer scoreboard with DSO and AR ageing
    col_l, col_r = st.columns([1.4, 1], gap="large")
    with col_l:
        section_header("Payer scoreboard", margin_top=8)
        if len(payer_df):
            scoreboard = payer_df.copy()
            scoreboard = scoreboard[
                ["payer_name", "billed", "collected", "outstanding",
                 "collection_rate_pct", "avg_dso"]
            ]
            scoreboard.columns = ["Payer", "Billed (KSh)", "Collected (KSh)",
                                  "Outstanding (KSh)", "Collection %", "Avg DSO"]
            st.dataframe(
                scoreboard.style
                    .format({
                        "Billed (KSh)": "{:,.0f}",
                        "Collected (KSh)": "{:,.0f}",
                        "Outstanding (KSh)": "{:,.0f}",
                        "Collection %": "{:.1f}%",
                        "Avg DSO": "{:.0f}",
                    })
                    .background_gradient(subset=["Collection %"], cmap="RdYlGn"),
                hide_index=True, use_container_width=True, height=320,
            )

    with col_r:
        section_header("AR ageing by payer", margin_top=8)
        if len(payer_df):
            buckets = ["bucket_0_30", "bucket_31_60", "bucket_61_90", "bucket_over_90"]
            buckets = [b for b in buckets if b in payer_df.columns]
            top6 = payer_df.head(6)
            fig = go.Figure()
            for i, b in enumerate(buckets):
                fig.add_trace(go.Bar(
                    x=top6["payer_name"], y=top6[b].fillna(0),
                    name=b.replace("bucket_", "").replace("_", "-"),
                    marker=dict(color=PALETTE[i], line=dict(width=0)),
                    hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
                ))
            fig.update_layout(**CHART_LAYOUT, height=320, barmode="stack",
                              xaxis=dict(tickangle=-20, tickfont=dict(size=9, color="#6B8CAE")),
                              legend=dict(orientation="h", y=-0.25, font=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)

    # ── Patient RFM segmentation (KMeans)
    section_header("Patient segmentation · RFM clustering (KMeans, k=5)", margin_top=8)
    if len(rfm_df):
        seg = predictive.segment_patients(rfm_df, k=5)
        risk = predictive.churn_risk(rfm_df)
        seg["churn_risk"] = risk["churn_risk"].values
        seg["risk_band"]  = risk["risk_band"].values

        col_a, col_b = st.columns([1.3, 1], gap="large")
        with col_a:
            sample = seg.sample(min(2500, len(seg)), random_state=42)
            seg_summary = (seg.groupby("segment")
                              .agg(patients=("patient_id", "count"),
                                   revenue=("monetary", "sum"),
                                   recency=("recency_days", "mean"),
                                   frequency=("frequency", "mean"))
                              .reset_index().sort_values("revenue", ascending=False))
            seg_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(seg_summary["segment"])}

            fig = go.Figure()
            for s in seg_summary["segment"]:
                sub = sample[sample["segment"] == s]
                fig.add_trace(go.Scatter(
                    x=sub["recency_days"], y=sub["frequency"],
                    mode="markers", name=s,
                    marker=dict(
                        size=np.clip(np.log1p(sub["monetary"]) * 1.6, 4, 22),
                        color=seg_color[s], line=dict(color="#fff", width=0.5),
                        opacity=0.7,
                    ),
                    hovertemplate=f"<b>{s}</b><br>Recency %{{x:.0f}}d<br>"
                                  "Visits %{y}<br>Spend %{marker.size:.0f}<extra></extra>",
                ))
            fig.update_layout(**CHART_LAYOUT, height=380,
                              xaxis=dict(title="Recency (days since last visit)",
                                         gridcolor="#EBF3FB",
                                         tickfont=dict(size=10, color="#6B8CAE")),
                              yaxis=dict(title="Frequency (visits)", gridcolor="#EBF3FB",
                                         tickfont=dict(size=10, color="#6B8CAE")),
                              legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            seg_summary["share_revenue"] = (
                100 * seg_summary["revenue"] / seg_summary["revenue"].sum()
            )
            fig = go.Figure(go.Bar(
                x=seg_summary["share_revenue"], y=seg_summary["segment"],
                orientation="h",
                marker=dict(color=[seg_color[s] for s in seg_summary["segment"]],
                            line=dict(width=0)),
                text=[f"{p:.1f}% · {int(n):,} pt" for p, n in
                      zip(seg_summary["share_revenue"], seg_summary["patients"])],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["navy"]),
                hovertemplate="<b>%{y}</b><br>%{x:.1f}% revenue<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, height=380,
                              margin=dict(l=0, r=80, t=10, b=20),
                              xaxis=dict(title="% of revenue", gridcolor="#EBF3FB"))
            st.plotly_chart(fig, use_container_width=True)

        critical = (risk["risk_band"] == "Critical").sum()
        info_card(
            f"<b>{int(critical):,}</b> patients are in the Critical churn band "
            f"({100*critical/len(risk):.1f}% of the active base). "
            f"Recovering 20% of them at average spend would add "
            f"<b>{fmt_ksh(0.2 * critical * rfm_df['avg_ticket'].median())}</b> per month.",
            COLORS["danger"],
        )

    # ── Cohort retention heatmap
    section_header("Cohort retention · revenue", margin_top=8)
    if len(cohort_df):
        ch = cohort_df.copy()
        pivot = ch.pivot_table(index="cohort_month", columns="month_offset",
                               values="revenue", aggfunc="sum")
        # Normalize each cohort row to its own month-0 revenue
        norm = pivot.div(pivot[0], axis=0) * 100 if 0 in pivot.columns else pivot
        fig = go.Figure(go.Heatmap(
            z=norm.values,
            x=[f"M{c}" for c in norm.columns],
            y=[d.strftime("%b %Y") for d in pd.to_datetime(norm.index)],
            colorscale=[[0, "#F4F8FC"], [0.5, "#7FB1E0"], [1, COLORS["primary"]]],
            hovertemplate="Cohort %{y}<br>%{x}<br>%{z:.0f}% of cohort month<extra></extra>",
            colorbar=dict(title="% of M0", tickfont=dict(size=9, color="#6B8CAE")),
        ))
        fig.update_layout(**CHART_LAYOUT, height=300, margin=dict(l=0, r=0, t=10, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — KPI Scorecard
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    section_header("Headline KPIs · selected period")

    # Compute all the headline KPIs
    weighted_dso = (
        (payer_df["billed"] * payer_df["avg_dso"]).sum() / payer_df["billed"].sum()
        if len(payer_df) and payer_df["billed"].sum() else None
    )
    leakage_pct  = leak_df["leakage_pct"].mean() if len(leak_df) else None
    margin_pct   = gp_df["gross_margin_pct"].mean() if len(gp_df) else None
    arpv_now     = arpv_df["arpv_28d"].dropna().iloc[-1] if len(arpv_df) else None
    rejection_rate = (rejection_df["rejection_rate_pct"].mean()
                      if len(rejection_df) else None)
    top_doctor_share = (
        100 * docs_df["revenue_attributed"].head(5).sum() / docs_df["revenue_attributed"].sum()
        if len(docs_df) and docs_df["revenue_attributed"].sum() else None
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Weighted DSO", f"{weighted_dso:.0f} d" if weighted_dso else "—",
                      "Across all payers", COLORS["warning"])
    with k2: kpi_card("Discount/waiver leakage", fmt_pct(leakage_pct),
                      "Of gross + leakage", COLORS["danger"])
    with k3: kpi_card("Gross margin (pharmacy)", fmt_pct(margin_pct),
                      "Avg over period", COLORS["success"])
    with k4: kpi_card("28-day ARPV", fmt_ksh(arpv_now),
                      "Smoothed average", COLORS["primary"])

    k5, k6, k7, k8 = st.columns(4)
    with k5: kpi_card("Claim rejection rate", fmt_pct(rejection_rate),
                      "Avg across payers", COLORS["danger"])
    with k6: kpi_card("Top-5 doctor share", fmt_pct(top_doctor_share),
                      "Of doctor-attributed revenue", COLORS["purple"])
    with k7: kpi_card("Active doctors", f"{len(docs_df):,}",
                      "With ≥5 visits in window", COLORS["navy"])
    with k8: kpi_card("Branches", f"{branches_df['clinic_id'].nunique() if len(branches_df) else 0}",
                      "With activity in window", COLORS["muted"])

    # ── KPI trends panel
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        section_header("ARPV trend · 7d & 28d smoothing", margin_top=8)
        if len(arpv_df):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=arpv_df["revenue_date"], y=arpv_df["arpv"],
                mode="lines", name="Daily ARPV",
                line=dict(color=hex_to_rgba(COLORS["primary"], 0.30), width=1),
                hovertemplate="%{x|%d %b}<br>%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=arpv_df["revenue_date"], y=arpv_df["arpv_7d"],
                mode="lines", name="7-day",
                line=dict(color=COLORS["primary"], width=2, shape="spline"),
                hovertemplate="7-day %{x|%d %b}<br>%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=arpv_df["revenue_date"], y=arpv_df["arpv_28d"],
                mode="lines", name="28-day",
                line=dict(color=COLORS["coral"], width=2.5, shape="spline"),
                hovertemplate="28-day %{x|%d %b}<br>%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, height=320, hovermode="x unified",
                              legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header("Discount + waiver leakage · monthly", margin_top=8)
        if len(leak_df):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=leak_df["revenue_month"], y=leak_df["discount"],
                name="Discount",
                marker=dict(color=COLORS["warning"], line=dict(width=0)),
                hovertemplate="%{x|%b %Y}<br>Discount %{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=leak_df["revenue_month"], y=leak_df["waiver"],
                name="Waiver",
                marker=dict(color=COLORS["danger"], line=dict(width=0)),
                hovertemplate="%{x|%b %Y}<br>Waiver %{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=leak_df["revenue_month"], y=leak_df["leakage_pct"],
                mode="lines+markers", name="Leakage %",
                line=dict(color=COLORS["navy"], width=2, shape="spline"),
                marker=dict(size=6, color=COLORS["navy"]),
                yaxis="y2",
                hovertemplate="%{x|%b %Y}<br>Leakage %{y:.2f}%<extra></extra>",
            ))
            fig.update_layout(
                **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                xaxis=dict(tickfont=dict(size=10, color="#6B8CAE")),
                yaxis=dict(title="KSh", gridcolor="#EBF3FB",
                           tickfont=dict(size=10, color="#6B8CAE")),
                yaxis2=dict(title="Leakage %", overlaying="y", side="right",
                            tickfont=dict(size=10, color=COLORS["navy"]),
                            gridcolor="rgba(0,0,0,0)"),
                barmode="stack", height=320,
                legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Doctor productivity polar
    section_header("Doctor productivity · top 12 (revenue & ARPV)", margin_top=8)
    if len(docs_df):
        top12 = docs_df.head(12).copy()
        # Normalise the two metrics to 0-100 for comparable bars
        top12["rev_norm"]  = 100 * top12["revenue_attributed"] / top12["revenue_attributed"].max()
        top12["arpv_norm"] = 100 * top12["arpv"] / top12["arpv"].max()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top12["doctor_name"], y=top12["rev_norm"],
            name="Revenue (indexed)",
            marker=dict(color=COLORS["primary"], line=dict(width=0)),
            text=[fmt_ksh(v) for v in top12["revenue_attributed"]],
            textposition="outside",
            textfont=dict(size=9, color=COLORS["navy"]),
            hovertemplate="<b>%{x}</b><br>Revenue %{text}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=top12["doctor_name"], y=top12["arpv_norm"],
            name="ARPV (indexed)", mode="lines+markers",
            line=dict(color=COLORS["coral"], width=2, shape="spline"),
            marker=dict(size=8, color=COLORS["coral"], line=dict(color="#fff", width=1)),
            hovertemplate="<b>%{x}</b><br>ARPV index %{y:.0f}<extra></extra>",
        ))
        fig.update_layout(**CHART_LAYOUT, height=380,
                          xaxis=dict(tickangle=-25, tickfont=dict(size=10, color="#6B8CAE")),
                          yaxis=dict(title="% of leader (max=100)", gridcolor="#EBF3FB",
                                     tickfont=dict(size=10, color="#6B8CAE")),
                          legend=dict(orientation="h", y=-0.3, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)

    # ── Branch radar comparison
    if len(branches_df):
        section_header("Branch comparison · multi-metric radar", margin_top=8)
        latest = branches_df.sort_values("revenue_month").groupby("clinic_name").last().reset_index()
        # Normalise metrics for the radar chart
        for col in ["revenue", "patients", "visits", "arpv", "arpu"]:
            if col in latest.columns:
                m = latest[col].max()
                latest[f"{col}_n"] = 100 * latest[col] / m if m else 0

        radar_metrics = [
            ("revenue_n",  "Revenue"),
            ("patients_n", "Patients"),
            ("visits_n",   "Visits"),
            ("arpv_n",     "ARPV"),
            ("arpu_n",     "ARPU"),
        ]
        radar_metrics = [m for m in radar_metrics if m[0] in latest.columns]

        fig = go.Figure()
        for i, (_, row) in enumerate(latest.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[row[m[0]] for m in radar_metrics] + [row[radar_metrics[0][0]]],
                theta=[m[1] for m in radar_metrics] + [radar_metrics[0][1]],
                fill="toself",
                name=row["clinic_name"],
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                fillcolor=hex_to_rgba(PALETTE[i % len(PALETTE)], 0.10),
                hovertemplate="<b>%{theta}</b><br>%{r:.1f}<extra></extra>",
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="#F4F8FC",
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor="#D6E4F0",
                                tickfont=dict(size=9, color="#6B8CAE")),
                angularaxis=dict(tickfont=dict(size=11, color="#003467")),
            ),
            paper_bgcolor="#fff",
            font=dict(family="Montserrat", color="#003467"),
            height=420,
            legend=dict(orientation="h", y=-0.1, font=dict(size=10)),
            margin=dict(l=20, r=20, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div style="margin-top:30px;padding-top:14px;border-top:1px solid #EBF3FB;'
    'font-size:10px;color:#6B8CAE;letter-spacing:1.5px;text-transform:uppercase">'
    'PharmaPlus · Revenue Intelligence · Powered by Snowflake'
    '</div>',
    unsafe_allow_html=True,
)