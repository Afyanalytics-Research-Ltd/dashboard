"""
app_revenue.py
--------------
lodwar · Revenue Intelligence

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
        "lodwar",
        "revenue_module"
    )
)


from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lodwar.revenue_module.data_layer as dl
import lodwar.revenue_module.predictive as predictive
import lodwar.revenue_module.whatif as whatif


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

PAGE_TITLE = "Revenue Intelligence"

st.set_page_config(
    page_title=f"lodwar · {PAGE_TITLE}",
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
        st.image("assets/lodwar_logo.png", width=160)
    except Exception:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#0072CE;'
            'padding:8px 0 16px">lodwar</div>',
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
        'Schema: <span style="color:#003467;font-weight:600">lodwar_PROD</span></div>',
        unsafe_allow_html=True,
    )


# ─── PAGE HEADER ────────────────────────────────────────────────────────────

st.markdown(
    f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    f'text-transform:uppercase;color:#0072CE;margin-bottom:16px">'
    f'lodwar · {PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)


# ─── DATA FETCH ─────────────────────────────────────────────────────────────

start_s = '1970-01-01'
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
    rejection_df = dl.claim_rejection(start_s, end_s)
    concent_df   = dl.revenue_concentration(start_s, end_s)
    arpv_df      = dl.arpv_trend(start_s, end_s)
    risk_df      = dl.revenue_at_risk(start_s, end_s)
except Exception as exc:
    st.error(
        "Could not connect to Snowflake. Verify SNOWFLAKE_ACCOUNT, "
        "SNOWFLAKE_USER, SNOWFLAKE_PRIVATE_KEY_PATH, and warehouse access."
    )
    st.exception(exc)
    st.stop()


print(daily)
# import pdb;pdb.set_trace()



# ─── KPI ROW ────────────────────────────────────────────────────────────────

total_revenue   = daily["gross_amount"].sum()
total_receipts  = daily["receipt_count"].sum()
unique_pat      = daily["unique_patients"].sum()
avg_daily       = daily.groupby("revenue_date")["gross_amount"].sum().mean()
collected_total = payer_df["collected"].sum() if len(payer_df) else 0
billed_total    = payer_df["billed"].sum()    if len(payer_df) else 0
collected_total = float(collected_total or 0)
billed_total    = float(billed_total or 0)
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
            fig.update_layout(**{**CHART_LAYOUT, "margin": dict(l=0, r=0, t=10, b=10)}, height=360)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No service-line data for this window.")

    
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
                cum += vals.astype(float)
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
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
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
        s = pd.Series(
            pd.to_numeric(series["revenue"], errors="coerce").values,
            index=pd.to_datetime(series["revenue_date"]),
        ).dropna()
        s_df = s.rename("revenue").rename_axis("revenue_date").reset_index()
        fc = predictive.forecast_revenue(s_df, horizon_days=forecast_horizon)
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

        if len(series) < 30:
            st.info("Need at least 30 days of data to estimate feature importance.")
        else:
            drivers = (
                predictive.revenue_drivers(series)
                        .sort_values("importance", ascending=True)
                        .reset_index(drop=True)
            )

            # Highlight the top driver with the accent colour, fade the rest
            top_idx = drivers["importance"].idxmax()
            bar_colors = [
                COLORS["purple"] if i == top_idx else "#B8CFE5"
                for i in range(len(drivers))
            ]

            # Pretty value labels (3 sig figs, no trailing zeros)
            max_imp = drivers["importance"].max() or 1
            text_labels = [f"{v:.3f}".rstrip("0").rstrip(".") for v in drivers["importance"]]

            fig = go.Figure(go.Bar(
                x=drivers["importance"],
                y=drivers["label"],
                orientation="h",
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0),
                ),
                text=text_labels,
                textposition="outside",
                textfont=dict(family="Montserrat", size=11, color="#0F2A47"),
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>importance %{x:.3f}<extra></extra>",
            ))

            # Subtle reference line at the median importance
            median_imp = float(drivers["importance"].median())
            fig.add_vline(
                x=median_imp,
                line=dict(color="#C8D6E5", width=1, dash="dot"),
                annotation_text="median",
                annotation_position="top",
                annotation_font=dict(family="Montserrat", size=9, color="#7A93B0"),
            )

            # Build the layout in a single dict so duplicates are impossible
            layout = {
                **CHART_LAYOUT,
                "height": 360,
                "margin": dict(l=10, r=60, t=20, b=20),
                "showlegend": False,
                "bargap": 0.35,
                "xaxis": dict(
                    title="",
                    showgrid=True,
                    gridcolor="#EBF3FB",
                    zeroline=False,
                    showline=False,
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    range=[0, max_imp * 1.18],   # leaves room for outside labels
                ),
                "yaxis": dict(
                    title="",
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    tickfont=dict(family="Montserrat", size=11, color="#0F2A47"),
                    automargin=True,
                ),
            }

            fig.update_layout(**layout)
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

        if not len(whatif_df) or baseline_avg <= 0:
            st.info("Not enough baseline revenue to project lever scenarios.")
        else:
            base_value      = baseline_avg * forecast_horizon
            cumulative_end  = float(whatif_df["cumulative"].iloc[-1])
            total_uplift    = cumulative_end - base_value
            total_uplift_pc = 100 * total_uplift / base_value if base_value else 0

            # Waterfall data
            x_labels = ["Baseline"] + whatif_df["lever"].tolist() + ["With all levers"]
            y_values = [base_value] + whatif_df["uplift_value"].tolist() + [0]
            text_lbl = (
                [fmt_ksh(base_value)]
                + [f"+{p:.1f}%" for p in whatif_df["uplift_pct"]]
                + [fmt_ksh(cumulative_end)]
            )

            fig = go.Figure(go.Waterfall(
                name="What-if",
                orientation="v",
                measure=["absolute"] + ["relative"] * len(whatif_df) + ["total"],
                x=x_labels,
                y=y_values,
                text=text_lbl,
                textposition="outside",
                textfont=dict(family="Montserrat", size=10, color=COLORS["navy"]),
                cliponaxis=False,
                connector=dict(
                    line=dict(color="#C8D6E5", width=1, dash="dot"),
                ),
                increasing=dict(marker=dict(color=COLORS["success"], line=dict(width=0))),
                decreasing=dict(marker=dict(color=COLORS["danger"],  line=dict(width=0))),
                totals    =dict(marker=dict(color=COLORS["primary"], line=dict(width=0))),
                hovertemplate="<b>%{x}</b><br>%{y:,.0f} KSh<extra></extra>",
            ))

            # Subtle baseline reference line
            fig.add_hline(
                y=base_value,
                line=dict(color="#C8D6E5", width=1, dash="dot"),
                annotation_text=f"baseline {fmt_ksh(base_value)}",
                annotation_position="top left",
                annotation_font=dict(family="Montserrat", size=9, color="#7A93B0"),
            )

            # Single dict → no duplicate-keyword errors possible
            layout = {
                **CHART_LAYOUT,
                "height": 360,
                "margin": dict(l=10, r=10, t=40, b=60),
                "showlegend": False,
                "bargap": 0.4,
                "xaxis": dict(
                    tickangle=-25,
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    showgrid=False,
                    showline=False,
                    zeroline=False,
                ),
                "yaxis": dict(
                    title=dict(text="Revenue (KSh)",
                            font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                    gridcolor="#EBF3FB",
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    zeroline=False,
                    showline=False,
                    rangemode="tozero",
                ),
            }
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            info_card(
                f"If all levers land, {forecast_horizon}-day revenue rises to "
                f"<b>{fmt_ksh(cumulative_end)}</b> from {fmt_ksh(base_value)} — "
                f"an uplift of <b>{fmt_ksh(total_uplift)}</b> "
                f"(<b>+{total_uplift_pc:.1f}%</b>).",
                COLORS["success"],
            )

    # ── Two-lever elasticity heatmap
    section_header("Two-lever sensitivity · ARPV × Volume", margin_top=8)

    if np.isnan(avg_daily) or avg_daily <= 0:
        st.info("Need a positive baseline to compute the sensitivity grid.")
    else:
        grid = whatif.elasticity_grid(avg_daily, horizon_days=forecast_horizon)
        pivot = grid.pivot(
            index="arpv_delta_pct",
            columns="volume_delta_pct",
            values="projected_revenue",
        ).sort_index().sort_index(axis=1)

        baseline_value = float(avg_daily * forecast_horizon)
        z_vals         = pivot.values.astype(float)
        z_min, z_max   = float(np.nanmin(z_vals)), float(np.nanmax(z_vals))

        # Diverging colorscale anchored at the baseline ("no change" point)
        colorscale = [
            [0.00, "#E85A6F"],   # deep coral — biggest losses
            [0.40, "#FBE6EB"],   # pale pink
            [0.50, "#F4F8FC"],   # neutral
            [0.60, "#D8EBE0"],   # pale mint
            [1.00, COLORS["success"]],
        ]

        fig = go.Figure(go.Heatmap(
            z=z_vals,
            x=pivot.columns,
            y=pivot.index,
            colorscale=colorscale,
            zmid=baseline_value,                          # anchors gradient at baseline
            zmin=z_min,
            zmax=z_max,
            xgap=2, ygap=2,                                # subtle separation between cells
            hovertemplate=(
                "<b>ARPV %{y:+.1f}%   ·   Volume %{x:+.1f}%</b>"
                "<br>Projected: %{z:,.0f} KSh"
                "<br>Δ vs baseline: %{customdata:+,.0f} KSh<extra></extra>"
            ),
            customdata=z_vals - baseline_value,
            colorbar=dict(
                title=dict(text="KSh", font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                tickfont=dict(family="Montserrat", size=9, color="#6B8CAE"),
                tickformat=",.0f",
                outlinewidth=0,
                thickness=12,
                len=0.85,
            ),
        ))

        # Crosshair marking the "no change" baseline point
        if 0 in pivot.index and 0 in pivot.columns:
            fig.add_hline(y=0, line=dict(color="#0F2A47", width=1, dash="dot"), opacity=0.45)
            fig.add_vline(x=0, line=dict(color="#0F2A47", width=1, dash="dot"), opacity=0.45)
            fig.add_annotation(
                x=0, y=0,
                text=f"<b>baseline</b><br>{fmt_ksh(baseline_value)}",
                showarrow=False,
                font=dict(family="Montserrat", size=9, color="#0F2A47"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#C8D6E5",
                borderwidth=1,
                borderpad=3,
            )

        layout = {
            **CHART_LAYOUT,
            "height": 360,
            "margin": dict(l=10, r=10, t=20, b=40),
            "xaxis": dict(
                title=dict(text="Visit volume Δ %",
                        font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                ticksuffix="%",
                showgrid=False,
                zeroline=False,
                showline=False,
                constrain="domain",
            ),
            "yaxis": dict(
                title=dict(text="ARPV Δ %",
                        font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                ticksuffix="%",
                showgrid=False,
                zeroline=False,
                showline=False,
                scaleanchor="x",      # square cells
                scaleratio=1,
            ),
        }
        fig.update_layout(**layout)
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
                        "Billed (KSh)":      "{:,.0f}",
                        "Collected (KSh)":   "{:,.0f}",
                        "Outstanding (KSh)": "{:,.0f}",
                        "Collection %":      "{:.1f}%",
                        "Avg DSO":           "{:.0f}",
                    }, na_rep="—")
                    .background_gradient(subset=["Collection %"], cmap="RdYlGn"),
                hide_index=True, use_container_width=True, height=320,
            )

    with col_r:
        section_header("AR ageing by payer", margin_top=8)

        bucket_meta = [
            ("bucket_0_30",     "0–30 days",   "#4FB286"),  # mint — current
            ("bucket_31_60",    "31–60 days",  "#F4C95D"),  # amber — watch
            ("bucket_61_90",    "61–90 days",  "#E8945A"),  # orange — concern
            ("bucket_over_90",  "90+ days",    "#D45B6E"),  # coral — risk
        ]
        available = [(col, label, c) for (col, label, c) in bucket_meta if col in payer_df.columns]

        if not len(payer_df) or not available:
            st.info("No accounts-receivable data to age for this window.")
        else:
            # Sort by total outstanding so the worst payers lead
            bucket_cols = [b[0] for b in available]
            plot_df = payer_df.copy()
            for c in bucket_cols:
                plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce").fillna(0)
            plot_df["_total"] = plot_df[bucket_cols].sum(axis=1)
            plot_df = plot_df.sort_values("_total", ascending=False).head(6)

            # Long y-tick labels truncated nicely
            x_labels = [
                (n if len(n) <= 18 else n[:16] + "…")
                for n in plot_df["payer_name"]
            ]

            fig = go.Figure()
            for col, label, colour in available:
                fig.add_trace(go.Bar(
                    name=label,
                    x=x_labels,
                    y=plot_df[col],
                    marker=dict(color=colour, line=dict(width=0)),
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:,.0f}} KSh<extra></extra>",
                ))

            # Total label sitting above each stack
            for x, total in zip(x_labels, plot_df["_total"]):
                fig.add_annotation(
                    x=x,
                    y=total,
                    text=fmt_ksh(total),
                    showarrow=False,
                    yshift=10,
                    font=dict(family="Montserrat", size=10, color="#0F2A47"),
                )

            layout = {
                **CHART_LAYOUT,
                "height": 360,
                "margin": dict(l=10, r=10, t=20, b=80),
                "barmode": "stack",
                "bargap": 0.35,
                "showlegend": True,
                "legend": dict(
                    orientation="h",
                    y=-0.32,
                    x=0.5,
                    xanchor="center",
                    font=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                "xaxis": dict(
                    tickangle=-20,
                    tickfont=dict(family="Montserrat", size=10, color="#0F2A47"),
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                ),
                "yaxis": dict(
                    title=dict(text="Outstanding (KSh)",
                            font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    gridcolor="#EBF3FB",
                    zeroline=False,
                    showline=False,
                    rangemode="tozero",
                ),
            }
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)
    # ── Patient RFM segmentation (KMeans)
    section_header("Patient segmentation · RFM clustering (KMeans, k=5)", margin_top=8)

    if not len(rfm_df):
        st.info("No patient activity in this window to segment.")
    else:
        seg  = predictive.segment_patients(rfm_df, k=5)
        risk = predictive.churn_risk(rfm_df)
        seg["churn_risk"] = risk["churn_risk"].values
        seg["risk_band"]  = risk["risk_band"].values

        # Coerce Decimals → float once
        for c in ("recency_days", "frequency", "monetary", "churn_risk"):
            if c in seg.columns:
                seg[c] = pd.to_numeric(seg[c], errors="coerce")

        col_a, col_b = st.columns([1.3, 1], gap="large")
        with col_a:
            sample = seg.sample(min(2500, len(seg)), random_state=42)

            seg_summary = (
                seg.groupby("segment")
                .agg(patients=("patient_id", "count"),
                        revenue=("monetary", "sum"),
                        recency=("recency_days", "mean"),
                        frequency=("frequency", "mean"))
                .reset_index()
                .sort_values("revenue", ascending=False)
            )
            seg_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(seg_summary["segment"])}

            fig = go.Figure()
            for s in seg_summary["segment"]:
                sub = sample[sample["segment"] == s].copy()
                if not len(sub):
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["recency_days"],
                    y=sub["frequency"],
                    mode="markers",
                    name=s,
                    marker=dict(
                        size=np.clip(np.log1p(sub["monetary"]) * 1.6, 4, 22),
                        color=seg_color[s],
                        line=dict(color="#ffffff", width=0.6),
                        opacity=0.65,
                    ),
                    customdata=np.stack([sub["monetary"].values,
                                        sub["recency_days"].values,
                                        sub["frequency"].values], axis=-1),
                    hovertemplate=(
                        f"<b>{s}</b>"
                        "<br>Recency: %{customdata[1]:.0f} days"
                        "<br>Visits: %{customdata[2]:.0f}"
                        "<br>Spend: %{customdata[0]:,.0f} KSh"
                        "<extra></extra>"
                    ),
                ))

            # Segment centroids — bigger, ringed, labelled
            fig.add_trace(go.Scatter(
                x=seg_summary["recency"],
                y=seg_summary["frequency"],
                mode="markers+text",
                text=seg_summary["segment"],
                textposition="top center",
                textfont=dict(family="Montserrat", size=10, color="#0F2A47"),
                marker=dict(
                    size=18,
                    color=[seg_color[s] for s in seg_summary["segment"]],
                    line=dict(color="#0F2A47", width=1.5),
                    symbol="circle",
                ),
                name="Segment centroid",
                hoverinfo="skip",
                showlegend=False,
            ))

            layout = {
                **CHART_LAYOUT,
                "height": 420,
                "margin": dict(l=10, r=10, t=20, b=80),
                "showlegend": True,
                "legend": dict(
                    orientation="h",
                    y=-0.22,
                    x=0.5,
                    xanchor="center",
                    font=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                "xaxis": dict(
                    title=dict(text="Recency · days since last visit",
                            font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    gridcolor="#EBF3FB",
                    zeroline=False,
                    showline=False,
                    rangemode="tozero",
                ),
                "yaxis": dict(
                    title=dict(text="Frequency · visits",
                            font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    gridcolor="#EBF3FB",
                    zeroline=False,
                    showline=False,
                    rangemode="tozero",
                    type="log" if sample["frequency"].max() > 50 else "linear",
                ),
            }
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            seg_summary = seg_summary.copy()
            seg_summary["revenue"]       = pd.to_numeric(seg_summary["revenue"], errors="coerce").fillna(0)
            seg_summary["patients"]      = pd.to_numeric(seg_summary["patients"], errors="coerce").fillna(0).astype(int)
            total_revenue                 = seg_summary["revenue"].sum()
            seg_summary["share_revenue"] = (
                100 * seg_summary["revenue"] / total_revenue if total_revenue else 0
            )

            # Sort ascending so the largest bar lands at the top of a horizontal chart
            seg_summary = seg_summary.sort_values("share_revenue", ascending=True)

            max_share = float(seg_summary["share_revenue"].max() or 0)

            bar_text = [
                f"{p:.1f}%   ·   {n:,} patients"
                for p, n in zip(seg_summary["share_revenue"], seg_summary["patients"])
            ]

            fig = go.Figure(go.Bar(
                x=seg_summary["share_revenue"],
                y=seg_summary["segment"],
                orientation="h",
                marker=dict(
                    color=[seg_color[s] for s in seg_summary["segment"]],
                    line=dict(width=0),
                ),
                text=bar_text,
                textposition="outside",
                textfont=dict(family="Montserrat", size=10, color=COLORS["navy"]),
                cliponaxis=False,
                customdata=seg_summary[["revenue", "patients"]].values,
                hovertemplate=(
                    "<b>%{y}</b>"
                    "<br>%{x:.1f}% of revenue"
                    "<br>%{customdata[0]:,.0f} KSh"
                    "<br>%{customdata[1]:,} patients"
                    "<extra></extra>"
                ),
            ))

            layout = {
                **CHART_LAYOUT,
                "height": 380,
                "margin": dict(l=10, r=110, t=20, b=20),     # right margin reserves room for outside labels
                "showlegend": False,
                "bargap": 0.4,
                "xaxis": dict(
                    title=dict(text="% of revenue",
                            font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    ticksuffix="%",
                    gridcolor="#EBF3FB",
                    zeroline=False,
                    showline=False,
                    range=[0, max_share * 1.25],             # leaves space for outside labels
                ),
                "yaxis": dict(
                    title="",
                    tickfont=dict(family="Montserrat", size=11, color="#0F2A47"),
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    automargin=True,
                ),
            }
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        # Critical-band info card (unchanged from your version, with safer math)
        critical          = int((risk["risk_band"] == "Critical").sum())
        critical_share    = (100 * critical / len(risk)) if len(risk) else 0
        median_ticket     = float(pd.to_numeric(rfm_df["avg_ticket"], errors="coerce").median() or 0)
        recovery_estimate = 0.2 * critical * median_ticket

        info_card(
            f"<b>{critical:,}</b> patients are in the Critical churn band "
            f"({critical_share:.1f}% of the active base). "
            f"Recovering 20% of them at median spend would add "
            f"<b>{fmt_ksh(recovery_estimate)}</b> per month.",
            COLORS["danger"],
        )
    # ── Cohort retention heatmap
    section_header("Cohort retention · revenue", margin_top=8)

    if not len(cohort_df):
        st.info("No cohort activity in this window.")
    else:
        ch = cohort_df.copy()
        ch["revenue"] = pd.to_numeric(ch["revenue"], errors="coerce").fillna(0)

        pivot = ch.pivot_table(
            index="cohort_month",
            columns="month_offset",
            values="revenue",
            aggfunc="sum",
        ).sort_index()

        if 0 not in pivot.columns or (pivot[0] == 0).all():
            st.info("Cohort month-0 baseline is missing — cannot compute retention %.")
        else:
            # Normalize each cohort to its own month-0 revenue (avoid div-by-zero)
            m0 = pivot[0].replace(0, np.nan)
            norm = pivot.div(m0, axis=0) * 100

            # Drop empty trailing offsets (no cohort has data there)
            norm = norm.dropna(axis=1, how="all")

            # Newest cohorts at the top — conventional for retention grids
            norm  = norm.sort_index(ascending=False)
            pivot = pivot.reindex(norm.index)

            x_labels = [f"M{int(c)}" for c in norm.columns]
            y_labels = [d.strftime("%b %Y") for d in pd.to_datetime(norm.index)]

            # Hover gets both % and absolute KSh
            revenue_grid = pivot.reindex(columns=norm.columns).values

            fig = go.Figure(go.Heatmap(
                z=norm.values,
                x=x_labels,
                y=y_labels,
                customdata=revenue_grid,
                colorscale=[
                    [0.00, "#F4F8FC"],
                    [0.25, "#D6E4F0"],
                    [0.50, "#9FC2E5"],
                    [0.75, "#5E94CC"],
                    [1.00, COLORS["primary"]],
                ],
                zmin=0,
                zmax=100,                      # clamps visual range; values >100% just saturate
                xgap=2, ygap=2,                # subtle cell separation
                hovertemplate=(
                    "<b>Cohort %{y}</b>"
                    "<br>%{x}"
                    "<br>Retention: %{z:.0f}% of M0"
                    "<br>Revenue: %{customdata:,.0f} KSh"
                    "<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(text="% of M0",
                            font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                    tickfont=dict(family="Montserrat", size=9, color="#6B8CAE"),
                    ticksuffix="%",
                    outlinewidth=0,
                    thickness=12,
                    len=0.85,
                ),
            ))

            # Annotate cells with their values when the grid is small enough to read
            if norm.shape[0] * norm.shape[1] <= 80:
                for i, cohort in enumerate(norm.index):
                    for j, off in enumerate(norm.columns):
                        val = norm.iloc[i, j]
                        if pd.isna(val):
                            continue
                        fig.add_annotation(
                            x=x_labels[j],
                            y=y_labels[i],
                            text=f"{val:.0f}",
                            showarrow=False,
                            font=dict(
                                family="Montserrat",
                                size=9,
                                color="#0F2A47" if val < 55 else "#FFFFFF",
                            ),
                        )

            layout = {
                **CHART_LAYOUT,
                "height": 360,
                "margin": dict(l=10, r=10, t=20, b=30),
                "xaxis": dict(
                    title="",
                    side="top",
                    tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                ),
                "yaxis": dict(
                    title="",
                    tickfont=dict(family="Montserrat", size=10, color="#0F2A47"),
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    automargin=True,
                ),
            }
            fig.update_layout(**layout)
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
    arpv_now     = arpv_df["arpv_28d"].dropna().iloc[-1] if len(arpv_df) else None
    rejection_rate = (rejection_df["rejection_rate_pct"].mean()
                      if len(rejection_df) else None)
    top_doctor_share = (
        100 * docs_df["revenue_attributed"].head(5).sum() / docs_df["revenue_attributed"].sum()
        if len(docs_df) and docs_df["revenue_attributed"].sum() else None
    )

    k1, k2, k3 = st.columns(3)
    with k1: kpi_card("Weighted DSO", f"{weighted_dso:.0f} d" if weighted_dso else "—",
                      "Across all payers", COLORS["warning"])
    with k2: kpi_card("Discount/waiver leakage", fmt_pct(leakage_pct),
                      "Of gross + leakage", COLORS["danger"])
    with k3: kpi_card("28-day ARPV", fmt_ksh(arpv_now),
                      "Smoothed average", COLORS["primary"])

    k5, k6, k7, k8 = st.columns(4)
    with k5: kpi_card("Claim rejection rate", fmt_pct(rejection_rate),
                      "Avg across payers", COLORS["danger"])
    with k6: kpi_card("Top-5 doctor share", fmt_pct(top_doctor_share),
                      "Of doctor-attributed revenue", COLORS["purple"])
    with k7: kpi_card("Active doctors", f"{len(docs_df):,}",
                      "With ≥5 visits in window", COLORS["navy"])
    
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

    if not len(docs_df):
        st.info("No doctor activity in this window.")
    else:
        top12 = docs_df.head(12).copy()

        # Coerce numerics once
        for c in ("revenue_attributed", "arpv", "visits", "unique_patients"):
            if c in top12.columns:
                top12[c] = pd.to_numeric(top12[c], errors="coerce").fillna(0)

        # Sort by revenue so the chart reads left → right "best to worst"
        top12 = top12.sort_values("revenue_attributed", ascending=False).reset_index(drop=True)

        # Index both metrics to their respective max so bars and line share a 0–100 axis
        rev_max  = float(top12["revenue_attributed"].max() or 1)
        arpv_max = float(top12["arpv"].max() or 1)
        top12["rev_norm"]  = 100 * top12["revenue_attributed"] / rev_max
        top12["arpv_norm"] = 100 * top12["arpv"]               / arpv_max

        # Truncate long doctor names for tick labels (full name still in hover)
        name_short = [
            n if len(n) <= 16 else n[:14] + "…"
            for n in top12["doctor_name"].astype(str)
        ]

        # Colour bars by ARPV index — darker = both high-revenue AND high-ARPV
        bar_colors = [
            f"rgba(74, 144, 226, {0.35 + 0.6 * (a / 100):.2f})"   # COLORS["primary"] family
            for a in top12["arpv_norm"]
        ]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=name_short,
            y=top12["rev_norm"],
            name="Revenue (indexed)",
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[fmt_ksh(v) for v in top12["revenue_attributed"]],
            textposition="outside",
            textfont=dict(family="Montserrat", size=9, color=COLORS["navy"]),
            cliponaxis=False,
            customdata=np.stack([
                top12["doctor_name"].astype(str).values,
                top12["revenue_attributed"].values,
                top12["arpv"].values,
                top12["visits"].values if "visits" in top12.columns else np.zeros(len(top12)),
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Revenue: %{customdata[1]:,.0f} KSh"
                "<br>ARPV: %{customdata[2]:,.0f} KSh"
                "<br>Visits: %{customdata[3]:,.0f}"
                "<extra></extra>"
            ),
        ))

        fig.add_trace(go.Scatter(
            x=name_short,
            y=top12["arpv_norm"],
            name="ARPV (indexed)",
            mode="lines+markers",
            line=dict(color=COLORS["coral"], width=2.5, shape="spline"),
            marker=dict(size=9, color=COLORS["coral"], line=dict(color="#ffffff", width=1.5)),
            customdata=np.stack([
                top12["doctor_name"].astype(str).values,
                top12["arpv"].values,
            ], axis=-1),
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>ARPV: %{customdata[1]:,.0f} KSh"
                "<br>Indexed: %{y:.0f}"
                "<extra></extra>"
            ),
        ))

        layout = {
            **CHART_LAYOUT,
            "height": 420,
            "margin": dict(l=10, r=10, t=40, b=110),
            "bargap": 0.35,
            "showlegend": True,
            "legend": dict(
                orientation="h",
                y=-0.32,
                x=0.5,
                xanchor="center",
                font=dict(family="Montserrat", size=10, color="#6B8CAE"),
                bgcolor="rgba(0,0,0,0)",
            ),
            "xaxis": dict(
                tickangle=-25,
                tickfont=dict(family="Montserrat", size=10, color="#0F2A47"),
                showgrid=False,
                zeroline=False,
                showline=False,
            ),
            "yaxis": dict(
                title=dict(text="% of leader (max = 100)",
                        font=dict(family="Montserrat", size=10, color="#6B8CAE")),
                tickfont=dict(family="Montserrat", size=10, color="#6B8CAE"),
                ticksuffix="%",
                gridcolor="#EBF3FB",
                zeroline=False,
                showline=False,
                range=[0, 115],                 # leaves room for outside text labels
            ),
        }
        fig.update_layout(**layout)

        # Tiny caption explaining the indexing — answers "100 of what?" without a hover
        fig.add_annotation(
            text=f"Leader: {fmt_ksh(rev_max)} revenue · ARPV {fmt_ksh(arpv_max)}",
            xref="paper", yref="paper", x=0.0, y=1.07,
            xanchor="left", showarrow=False,
            font=dict(family="Montserrat", size=10, color="#7A93B0"),
        )

        st.plotly_chart(fig, use_container_width=True)

    
# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div style="margin-top:30px;padding-top:14px;border-top:1px solid #EBF3FB;'
    'font-size:10px;color:#6B8CAE;letter-spacing:1.5px;text-transform:uppercase">'
    'lodwar · Revenue Intelligence · Powered by Snowflake'
    '</div>',
    unsafe_allow_html=True,
)