"""
app_revenue.py
--------------
tenri · Revenue Intelligence

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
        "tenri",
        "revenue_module"
    )
)


from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import tenri.revenue_module.data_layer as dl
import tenri.revenue_module.predictive as predictive
import tenri.revenue_module.whatif as whatif


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

PAGE_TITLE = "Revenue Intelligence"

st.set_page_config(
    page_title=f"tenri · {PAGE_TITLE}",
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
        st.image("assets/tenri_logo.png", width=160)
    except Exception:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#0072CE;'
            'padding:8px 0 16px">tenri</div>',
            unsafe_allow_html=True,
        )

    section_header("Date range")
    today = date.today()
    quick_pick = st.selectbox(
        "Period",
        ["All time", "Last 30 days", "Last 90 days", "Last 6 months",
        "Last 12 months", "Last 24 months", "Custom"],
        index=0,
    )

    spans = {
        "Last 30 days": 30, "Last 90 days": 90, "Last 6 months": 183,
        "Last 12 months": 365, "Last 24 months": 730,
    }

    if quick_pick == "All time":
        start = date(1970, 1, 1)
        end   = date(2025, 9, 30)
    elif quick_pick == "Custom":
        start = st.date_input(
            "Start",
            value=date(1970, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date(2100, 12, 31),
        )
        end = st.date_input(
            "End",
            value=date(2025, 9, 30),
            min_value=date(1900, 1, 1),
            max_value=date(2100, 12, 31),
        )
    else:
        end   = today
        start = end - timedelta(days=spans[quick_pick])
    
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
        'Schema: <span style="color:#003467;font-weight:600">tenri_PROD</span></div>',
        unsafe_allow_html=True,
    )


# ─── PAGE HEADER ────────────────────────────────────────────────────────────

st.markdown(
    f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    f'text-transform:uppercase;color:#0072CE;margin-bottom:16px">'
    f'tenri · {PAGE_TITLE}</p>',
    unsafe_allow_html=True,
)


# ─── DATA FETCH ─────────────────────────────────────────────────────────────
start_s = start.strftime("%Y-%m-%d")
end_s   = (end + timedelta(days=1)).strftime("%Y-%m-%d")    # exclusive

# All Snowflake calls happen here. If the connection fails, we surface a
# clear error and stop — there is no silent fallback.
try:
    daily        = dl.daily_revenue(start_s, end_s, clinic_filter)
    sl_monthly   = dl.revenue_by_service_line(start_s, end_s, clinic_filter)
    # sl_monthly.to_csv('service_line.csv', index=False)
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

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "◉  Revenue Pulse",
    "△  Forecast & Risk",
    "◇  Payer & Patient Mix",
    "∑  KPI Scorecard",
    "⚙  Simulation",
    "⚙ Payment Simulator"
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

    
    # ── Service line treemap
    section_header("Service line mix · selected period")
    sl_total = (sl_monthly.groupby("service_line", as_index=False)["gross_revenue"]
                .sum().sort_values("gross_revenue", ascending=False))
    if len(sl_total):
        import plotly.subplots as sp
        import plotly.graph_objects as go

        top   = sl_total.sort_values("gross_revenue", ascending=True)
        rest  = sl_total[sl_total["service_line"] != "Investigation"].sort_values("gross_revenue", ascending=True)

        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("All service lines", "Excluding Investigation"),
            horizontal_spacing=0.18,
        )
        fig.add_trace(go.Bar(
            x=top["gross_revenue"],
            y=top["service_line"],
            orientation="h",
            marker_color="#0072CE",
            text=top["gross_revenue"],          # raw numbers, no .apply()
            texttemplate="%{text:,.2s}",        # d3-format, runs in the browser
            textposition="outside",
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=rest["gross_revenue"],
            y=rest["service_line"],
            orientation="h",
            marker_color="#5BA7E8",
            text=rest["gross_revenue"],
            texttemplate="%{text:,.2s}",
            textposition="outside",
        ), row=1, col=2)
        fig.update_layout(showlegend=False, height=600, plot_bgcolor="white",
                        margin=dict(l=180, r=80, t=60, b=40))
        fig.update_xaxes(tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No service-line data for this window.")

    
    # ── Hourly heatmap + payment mode area
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        section_header("Demand heatmap · hour × weekday")
        hourly_df.to_csv('tenri_hourly_df.csv',index=False)
        if len(hourly_df):
            day_map = {"Mon":"Monday","Tue":"Tuesday","Wed":"Wednesday","Thu":"Thursday",
                    "Fri":"Friday","Sat":"Saturday","Sun":"Sunday"}
            hourly_df["day_name"] = hourly_df["day_name"].map(day_map)

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
        total = items_df["gross_revenue"].sum()
        top["cum_share"] = 100 * top["gross_revenue"].cumsum() / total

        leader      = top.iloc[0]
        leader_pct  = 100 * leader["gross_revenue"] / total
        rest        = top.iloc[1:]            # ranks #2 – #20

        col_l, col_r = st.columns([1, 3])

        # ── Left: the dominant item as a KPI tile ───────────────────────────
        with col_l:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#003467,#0072CE);
                            color:white;border-radius:12px;padding:20px 18px;
                            height:340px;display:flex;flex-direction:column;
                            justify-content:center;">
                    <div style="font-size:10px;letter-spacing:2px;
                                text-transform:uppercase;opacity:0.7;">
                        #1 revenue item
                    </div>
                    <div style="font-size:18px;font-weight:700;margin:6px 0 14px;
                                line-height:1.2;">
                        {leader['item_name']}
                    </div>
                    <div style="font-size:28px;font-weight:800;">
                        KSh {leader['gross_revenue']:,.0f}
                    </div>
                    <div style="font-size:13px;margin-top:8px;opacity:0.85;">
                        {leader_pct:.1f}% of top-20 revenue
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Right: ranks #2 – #20 on their own scale ────────────────────────
        with col_r:
            fig_rest = go.Figure()
            fig_rest.add_trace(go.Bar(
                x=rest["item_name"], y=rest["gross_revenue"],
                marker=dict(color=COLORS["primary"], line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>%{y:,.0f} KSh<extra></extra>",
            ))
            fig_rest.update_layout(
                **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
                xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#6B8CAE")),
                yaxis=dict(title="Revenue (KSh)", gridcolor="#EBF3FB",
                        tickfont=dict(size=10, color="#6B8CAE")),
                height=340, showlegend=False,
                margin=dict(l=0, r=0, t=10, b=80),
                title=dict(text="Items #2 – #20 by revenue",
                        font=dict(size=13, color="#003467"), x=0, xanchor="left"),
            )
            st.plotly_chart(fig_rest, use_container_width=True)

        # ── Cumulative-share Pareto line, full width below ──────────────────
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=top["item_name"], y=top["cum_share"],
            mode="lines+markers",
            line=dict(color=COLORS["coral"], width=2.5, shape="spline"),
            marker=dict(size=6, color=COLORS["coral"]),
            fill="tozeroy", fillcolor="rgba(255,127,80,0.08)",
            hovertemplate="<b>%{x}</b><br>Cumulative %{y:.1f}%<extra></extra>",
        ))
        fig_line.add_hline(
            y=80, line_dash="dash", line_color="#6B8CAE", line_width=1,
            annotation_text="80% threshold", annotation_position="right",
            annotation_font_size=10, annotation_font_color="#6B8CAE",
        )
        fig_line.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
            xaxis=dict(tickangle=-45, tickfont=dict(size=9, color="#6B8CAE")),
            yaxis=dict(title="Cumulative %", range=[0, 105], gridcolor="#EBF3FB",
                    tickfont=dict(size=10, color=COLORS["coral"]), ticksuffix="%"),
            height=300, showlegend=False, margin=dict(l=0, r=0, t=20, b=80),
            title=dict(text="Cumulative share of top-20 revenue",
                    font=dict(size=13, color="#003467"), x=0, xanchor="left"),
        )
        st.plotly_chart(fig_line, use_container_width=True)


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
        # Split: top payer goes left, the rest go right (rescaled so small bars are visible)
        top_row = concent_df.iloc[[0]].copy()
        rest_df = concent_df.iloc[1:].copy()

        # Rescale the "rest" so their shares sum to 100% within the right-hand chart
        if len(rest_df):
            rest_total = rest_df["share_pct"].sum()
            if rest_total > 0:
                rest_df["share_pct_rescaled"] = rest_df["share_pct"] / rest_total * 100
                rest_df["cumulative_rescaled"] = rest_df["share_pct_rescaled"].cumsum()
            else:
                rest_df["share_pct_rescaled"] = rest_df["share_pct"]
                rest_df["cumulative_rescaled"] = rest_df["share_pct"].cumsum()

        col_left, col_right = st.columns([1, 3])

        # ---- LEFT: dominant payer ----
        with col_left:
            top_payer = top_row["payer"].iloc[0]
            top_share = float(top_row["share_pct"].iloc[0])
            fig_top = go.Figure()
            fig_top.add_trace(go.Bar(
                x=[top_payer], y=[top_share],
                marker=dict(color=COLORS["primary"], line=dict(width=0)),
                hovertemplate="<b>%{x}</b><br>%{y:.1f}% share<extra></extra>",
                text=[f"{top_share:.1f}%"], textposition="outside",
            ))
            fig_top.update_layout(
                **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                xaxis=dict(tickfont=dict(size=11, color="#6B8CAE")),
                yaxis=dict(title="Share %", gridcolor="#EBF3FB",
                        range=[0, max(105, top_share * 1.15)],
                        tickfont=dict(size=10, color="#6B8CAE")),
                height=320, showlegend=False,
                title=dict(text="Dominant payer", x=0.5, xanchor="center",
                        font=dict(size=12, color="#6B8CAE")),
            )
            st.plotly_chart(fig_top, use_container_width=True)

        # ---- RIGHT: remaining payers (rescaled) ----
        with col_right:
            if len(rest_df):
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=rest_df["payer"], y=rest_df["share_pct_rescaled"],
                    name="Share % (of remainder)",
                    marker=dict(color=COLORS["primary"], line=dict(width=0)),
                    customdata=rest_df["share_pct"],
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "%{y:.1f}% of remainder<br>"
                        "%{customdata:.1f}% of total<extra></extra>"
                    ),
                ))
                fig.add_trace(go.Scatter(
                    x=rest_df["payer"], y=rest_df["cumulative_rescaled"],
                    mode="lines+markers", name="Cumulative %",
                    line=dict(color=COLORS["coral"], width=2.5, shape="spline"),
                    marker=dict(size=7, color=COLORS["coral"]),
                    yaxis="y2",
                    hovertemplate="<b>%{x}</b><br>Cumulative %{y:.1f}% of remainder<extra></extra>",
                ))
                fig.update_layout(
                    **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                    xaxis=dict(tickangle=-25, tickfont=dict(size=10, color="#6B8CAE")),
                    yaxis=dict(title="Share % (of remainder)", gridcolor="#EBF3FB",
                            tickfont=dict(size=10, color="#6B8CAE")),
                    yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                range=[0, 105], gridcolor="rgba(0,0,0,0)",
                                tickfont=dict(size=10, color=COLORS["coral"])),
                    height=320, showlegend=False,
                    title=dict(
                        text=f"Remaining payers (excl. {top_payer} — {top_share:.0f}% of total)",
                        x=0.5, xanchor="center", font=dict(size=12, color="#6B8CAE"),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Only one payer in the dataset.")

        # ---- HHI verdict (unchanged, uses original concent_df) ----
        hhi = float(concent_df["hhi_index"].iloc[0]) if "hhi_index" in concent_df.columns else None
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

    


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Simulation (actionable recommendations)
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown(
        '<p style="font-size:13px;color:#6B8CAE;margin:0 0 16px;line-height:1.5">'
        'Three levers, ranked by where your data says they\'ll bite hardest. '
        'Each panel detects the top performer in its category, then lets you '
        'project the revenue impact of acting on it.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────────────────────────────────
    # LEVER 1 — TOP DAY → Increase staffing / marketing
    # ────────────────────────────────────────────────────────────────────────
    section_header("Lever 1 · Top day → staffing & marketing push")

    # Detect top day-of-week from the hourly heatmap (already loaded).
    if not len(hourly_df) or "day_name" not in hourly_df.columns:
        st.info("No hourly data available to identify the top day.")
    else:
        day_rev = (hourly_df.groupby("day_name", as_index=False)["revenue"].sum()
                   .sort_values("revenue", ascending=False))
        top_day_name      = day_rev.iloc[0]["day_name"]
        top_day_revenue   = float(day_rev.iloc[0]["revenue"])
        avg_other_days    = float(day_rev.iloc[1:]["revenue"].mean()) if len(day_rev) > 1 else 0
        n_weeks_in_window = max((end - start).days / 7, 1)
        top_day_per_week  = top_day_revenue / n_weeks_in_window

        col_l, col_r = st.columns([1, 1.4], gap="large")

        with col_l:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#003467,#0072CE);
                            color:white;border-radius:12px;padding:20px 18px;
                            height:280px;display:flex;flex-direction:column;
                            justify-content:center;">
                    <div style="font-size:10px;letter-spacing:2px;
                                text-transform:uppercase;opacity:0.7;">
                        Top revenue day
                    </div>
                    <div style="font-size:32px;font-weight:800;margin:8px 0 12px;">
                        {top_day_name}
                    </div>
                    <div style="font-size:11px;opacity:0.75;margin-bottom:4px;">
                        Weekly average on {top_day_name}s
                    </div>
                    <div style="font-size:22px;font-weight:700;">
                        {fmt_ksh(top_day_per_week)}
                    </div>
                    <div style="font-size:12px;margin-top:10px;opacity:0.85;">
                        +{((top_day_per_week / avg_other_days - 1) * 100) if avg_other_days else 0:.0f}%
                        vs. average non-top day
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r:
            st.markdown(f"**Tune the {top_day_name} intervention:**")
            volume_lift_pct = st.slider(
                f"Expected volume lift on {top_day_name}s (%)",
                0, 40, 15, step=1, key="sim_day_volume",
                help="Marketing or extra-staff capacity should drive incremental visits.",
            )
            campaign_cost   = st.number_input(
                "Weekly intervention cost (KSh)",
                min_value=0, value=15_000, step=1_000, key="sim_day_cost",
                help="Combined cost of marketing + additional staff for one peak day.",
            )
            margin_pct      = st.slider(
                "Contribution margin on incremental revenue (%)",
                10, 80, 35, step=5, key="sim_day_margin",
                help="What fraction of incremental revenue drops to bottom line.",
            )

            weekly_uplift_rev    = top_day_per_week * (volume_lift_pct / 100)
            weekly_uplift_margin = weekly_uplift_rev * (margin_pct / 100)
            weekly_net           = weekly_uplift_margin - campaign_cost
            annual_net           = weekly_net * 52
            payback_weeks        = (campaign_cost / weekly_uplift_margin
                                    if weekly_uplift_margin > 0 else None)

            m1, m2, m3 = st.columns(3)
            with m1:
                kpi_card("Weekly revenue lift", fmt_ksh(weekly_uplift_rev),
                         f"On {top_day_name}s", COLORS["primary"])
            with m2:
                kpi_card("Weekly net contribution", fmt_ksh(weekly_net),
                         "After intervention cost",
                         COLORS["success"] if weekly_net > 0 else COLORS["danger"])
            with m3:
                kpi_card("Annualised net", fmt_ksh(annual_net),
                         f"Payback: {payback_weeks:.1f} wks" if payback_weeks else "Negative ROI",
                         COLORS["green"] if annual_net > 0 else COLORS["danger"])

            info_card(
                f"Adding <b>{volume_lift_pct}%</b> volume on {top_day_name}s yields "
                f"<b>{fmt_ksh(weekly_uplift_rev)}</b>/week in revenue. "
                f"At a {margin_pct}% margin and {fmt_ksh(campaign_cost)} weekly spend, "
                f"net contribution is <b>{fmt_ksh(weekly_net)}</b>/week "
                f"(<b>{fmt_ksh(annual_net)}</b> annualised).",
                COLORS["success"] if weekly_net > 0 else COLORS["danger"],
            )

    # ────────────────────────────────────────────────────────────────────────
    # LEVER 2 — TOP PRODUCT → Frequently-bought-together bundles
    # ────────────────────────────────────────────────────────────────────────
    section_header("Lever 2 · Top product → frequently-bought-together bundles",
                   margin_top=12)

    if not len(items_df):
        st.info("No item-level data available to build bundles.")
    else:
        top_items_for_bundle = items_df.head(5).copy()
        anchor_name      = top_items_for_bundle.iloc[0]["item_name"]
        anchor_revenue   = float(top_items_for_bundle.iloc[0]["gross_revenue"])
        # Use a representative unit price for the anchor (best-effort)
        if "units_sold" in top_items_for_bundle.columns and \
           top_items_for_bundle.iloc[0]["units_sold"]:
            anchor_units = float(top_items_for_bundle.iloc[0]["units_sold"])
            anchor_unit_price = anchor_revenue / max(anchor_units, 1)
        else:
            anchor_units = 0
            anchor_unit_price = anchor_revenue / max(unique_pat or 1, 1)

        col_l, col_r = st.columns([1, 1.4], gap="large")

        with col_l:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#0BB99F,#1D9E75);
                            color:white;border-radius:12px;padding:20px 18px;
                            height:280px;display:flex;flex-direction:column;
                            justify-content:center;">
                    <div style="font-size:10px;letter-spacing:2px;
                                text-transform:uppercase;opacity:0.7;">
                        Anchor product
                    </div>
                    <div style="font-size:18px;font-weight:700;margin:6px 0 14px;
                                line-height:1.2;">
                        {anchor_name}
                    </div>
                    <div style="font-size:11px;opacity:0.75;">In-period revenue</div>
                    <div style="font-size:22px;font-weight:800;">
                        {fmt_ksh(anchor_revenue)}
                    </div>
                    <div style="font-size:11px;margin-top:10px;opacity:0.85;">
                        Suggested attach candidates from the top-5:
                        {", ".join(top_items_for_bundle["item_name"].iloc[1:4].tolist())}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r:
            st.markdown(f"**Bundle design around {anchor_name}:**")
            attach_rate = st.slider(
                "Bundle attach rate (% of anchor buyers who add the bundle)",
                5, 70, 25, step=1, key="sim_bundle_attach",
                help="Industry benchmark for retail-style attach is 15–30%.",
            )
            attach_avg_price = st.number_input(
                "Average price of attached item (KSh)",
                min_value=0, value=int(max(anchor_unit_price * 0.4, 200)),
                step=50, key="sim_bundle_price",
                help="Mean price of the 2–3 items bundled with the anchor.",
            )
            bundle_discount = st.slider(
                "Bundle discount vs Bundling individual items, priced separately (%)",
                0, 30, 10, step=1, key="sim_bundle_discount",
                help="The price break that motivates the bundle uptake.",
            )

            # Anchor "transactions" approximated from receipt count / patient share
            est_anchor_transactions = (
                anchor_units if anchor_units > 0
                else anchor_revenue / max(anchor_unit_price, 1)
            )
            bundle_attaches  = est_anchor_transactions * (attach_rate / 100)
            attach_revenue   = bundle_attaches * attach_avg_price * (1 - bundle_discount / 100)
            uplift_pct_total = 100 * attach_revenue / anchor_revenue if anchor_revenue else 0

            m1, m2, m3 = st.columns(3)
            with m1:
                kpi_card("Bundle uptake",
                         f"{int(bundle_attaches):,}",
                         "Additional bundle attaches", COLORS["success"])
            with m2:
                kpi_card("Incremental revenue", fmt_ksh(attach_revenue),
                         f"+{uplift_pct_total:.1f}% vs anchor",
                         COLORS["primary"])
            with m3:
                annualised_bundle = attach_revenue * (365 / max((end - start).days, 1))
                kpi_card("Annualised", fmt_ksh(annualised_bundle),
                         "If pattern persists", COLORS["green"])

            info_card(
                f"At a <b>{attach_rate}%</b> attach rate and "
                f"<b>{fmt_ksh(attach_avg_price)}</b> add-on price (after a "
                f"{bundle_discount}% bundle discount), <b>{anchor_name}</b> "
                f"transactions generate <b>{fmt_ksh(attach_revenue)}</b> of "
                f"incremental bundle revenue in this window — "
                f"<b>{fmt_ksh(annualised_bundle)}</b> annualised.",
                COLORS["success"],
            )

    # ────────────────────────────────────────────────────────────────────────
    # LEVER 3 — TOP SERVICE → Subscription / premium tier
    # ────────────────────────────────────────────────────────────────────────
    section_header("Lever 3 · Top service → subscription or premium tier",
                   margin_top=12)

    if not len(sl_monthly):
        st.info("No service-line data available to size a subscription.")
    else:
        sl_total = (sl_monthly.groupby("service_line", as_index=False)["gross_revenue"]
                    .sum().sort_values("gross_revenue", ascending=False))
        top_service        = sl_total.iloc[0]["service_line"]
        top_service_rev    = float(sl_total.iloc[0]["gross_revenue"])
        # Estimate monthly run-rate for this service
        months_in_window   = max((end - start).days / 30.0, 1)
        service_monthly    = top_service_rev / months_in_window

        col_l, col_r = st.columns([1, 1.4], gap="large")

        with col_l:
            st.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#7F77DD,#0072CE);
                            color:white;border-radius:12px;padding:20px 18px;
                            height:280px;display:flex;flex-direction:column;
                            justify-content:center;">
                    <div style="font-size:10px;letter-spacing:2px;
                                text-transform:uppercase;opacity:0.7;">
                        Top service line
                    </div>
                    <div style="font-size:24px;font-weight:800;margin:6px 0 12px;">
                        {top_service}
                    </div>
                    <div style="font-size:11px;opacity:0.75;">Monthly run-rate</div>
                    <div style="font-size:22px;font-weight:700;">
                        {fmt_ksh(service_monthly)}
                    </div>
                    <div style="font-size:11px;margin-top:10px;opacity:0.85;">
                        Strong candidate for a recurring-revenue tier.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_r:
            st.markdown(f"**Design a subscription/premium tier for {top_service}:**")
            sub_price = st.number_input(
                "Subscription price (KSh / month)",
                min_value=0, value=2_500, step=100, key="sim_sub_price",
                help="Monthly fee for the recurring/premium tier.",
            )
            conversion_pct = st.slider(
                "Active-patient conversion rate (%)",
                1, 30, 8, step=1, key="sim_sub_conversion",
                help="Share of current active patients who'd subscribe in year 1.",
            )
            avg_retention_months = st.slider(
                "Average subscriber retention (months)",
                3, 36, 12, step=1, key="sim_sub_retention",
                help="Lifetime of a subscriber before churn.",
            )
            cannibalisation_pct = st.slider(
                "Cannibalisation of bundling individual items, priced separately (%)",
                0, 50, 20, step=5, key="sim_sub_cannibal",
                help="Share of subscription revenue that would have come in anyway.",
            )

            subscribers_y1   = (unique_pat or 0) * (conversion_pct / 100)
            mrr              = subscribers_y1 * sub_price
            arr_gross        = mrr * 12
            ltv_per_sub      = sub_price * avg_retention_months
            ltv_total        = subscribers_y1 * ltv_per_sub
            arr_net          = arr_gross * (1 - cannibalisation_pct / 100)

            m1, m2, m3 = st.columns(3)
            with m1:
                kpi_card("Year-1 subscribers", f"{int(subscribers_y1):,}",
                         f"{conversion_pct}% of active base", COLORS["primary"])
            with m2:
                kpi_card("Annual recurring revenue", fmt_ksh(arr_net),
                         f"Net of {cannibalisation_pct}% cannibalisation",
                         COLORS["success"])
            with m3:
                kpi_card("Cohort lifetime value", fmt_ksh(ltv_total),
                         f"At {avg_retention_months}-mo retention",
                         COLORS["purple"])

            info_card(
                f"At <b>{conversion_pct}%</b> conversion of {int(unique_pat or 0):,} "
                f"active patients, a <b>{fmt_ksh(sub_price)}/month</b> "
                f"{top_service} subscription generates <b>{fmt_ksh(mrr)}</b> MRR — "
                f"<b>{fmt_ksh(arr_net)}</b> net ARR after cannibalisation. "
                f"Total cohort LTV: <b>{fmt_ksh(ltv_total)}</b>.",
                COLORS["primary"],
            )

    # ────────────────────────────────────────────────────────────────────────
    # COMBINED SUMMARY — all three levers stacked
    # ────────────────────────────────────────────────────────────────────────
    section_header("Combined annualised impact · all three levers", margin_top=12)

    annual_day_lever      = locals().get("annual_net", 0) or 0
    annual_bundle_lever   = locals().get("annualised_bundle", 0) or 0
    annual_sub_lever      = locals().get("arr_net", 0) or 0
    annual_total          = annual_day_lever + annual_bundle_lever + annual_sub_lever

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["Top-day push", "Bundle attach", "Subscription tier", "Combined uplift"],
        y=[annual_day_lever, annual_bundle_lever, annual_sub_lever, 0],
        text=[fmt_ksh(annual_day_lever), fmt_ksh(annual_bundle_lever),
              fmt_ksh(annual_sub_lever), fmt_ksh(annual_total)],
        textposition="outside",
        textfont=dict(family="Montserrat", size=11, color=COLORS["navy"]),
        cliponaxis=False,
        connector=dict(line=dict(color="#C8D6E5", width=1, dash="dot")),
        increasing=dict(marker=dict(color=COLORS["success"], line=dict(width=0))),
        decreasing=dict(marker=dict(color=COLORS["danger"],  line=dict(width=0))),
        totals    =dict(marker=dict(color=COLORS["primary"], line=dict(width=0))),
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} KSh / year<extra></extra>",
    ))
    fig.update_layout(
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
        height=340,
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis=dict(tickfont=dict(size=11, color=COLORS["navy"]),
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Annualised KSh", gridcolor="#EBF3FB",
                   tickfont=dict(size=10, color="#6B8CAE"),
                   zeroline=False),
        showlegend=False,
        bargap=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)

    info_card(
        f"If all three levers land at the configured assumptions, "
        f"the combined annualised revenue uplift is <b>{fmt_ksh(annual_total)}</b>. "
        f"Sliders are independent — adjust each lever to stress-test the plan.",
        COLORS["primary"],
    )


with tab6:
    section_header("Payer simulator · what-if levers for revenue, mix & retention")

    # ──────────────────────────────────────────────────────────────────────────
    # 0 · Baselines (pulled from the same frames the other tabs use)
    # ──────────────────────────────────────────────────────────────────────────
    have_payers   = "payer_df"   in dir() and len(payer_df)
    have_concent  = "concent_df" in dir() and len(concent_df)
    have_rfm      = "rfm_df"     in dir() and len(rfm_df)
    have_cohort   = "cohort_df"  in dir() and len(cohort_df)

    if not have_payers or not have_concent:
        st.info("Payer simulator needs the payer scoreboard and concentration data.")
        st.stop()

    base = payer_df.copy()
    for c in ("billed", "collected", "outstanding", "collection_rate_pct", "avg_dso"):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

    # Align payer naming between concent_df ("payer") and payer_df ("payer_name")
    payer_key_concent = "payer"
    payer_key_score   = "payer_name"

    baseline_total_billed     = float(base["billed"].sum())
    baseline_total_collected  = float(base["collected"].sum())
    baseline_total_outstanding = float(base["outstanding"].sum())
    baseline_collection_rate  = (
        100 * baseline_total_collected / baseline_total_billed
        if baseline_total_billed else 0
    )
    baseline_dso = float((base["avg_dso"] * base["billed"]).sum() /
                         max(baseline_total_billed, 1))

    top_payer  = concent_df[payer_key_concent].iloc[0]
    top_share  = float(concent_df["share_pct"].iloc[0])
    baseline_hhi = (
        float(concent_df["hhi_index"].iloc[0])
        if "hhi_index" in concent_df.columns else
        float((concent_df["share_pct"] ** 2).sum())
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 1 · Levers (three columns, mapped to the three strategic goals)
    # ──────────────────────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='color:#6B8CAE;font-size:12px;margin-bottom:6px;'>"
        f"Baseline: <b style='color:#0F2A47'>{fmt_ksh(baseline_total_billed)}</b> billed · "
        f"<b style='color:#0F2A47'>{baseline_collection_rate:.1f}%</b> collected · "
        f"DSO <b style='color:#0F2A47'>{baseline_dso:.0f}</b> days · "
        f"HHI <b style='color:#0F2A47'>{baseline_hhi:.0f}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    lev1, lev2, lev3 = st.columns(3, gap="large")

    with lev1:
        st.markdown("**1 · Diversify payer mix**")
        divert_pct = st.slider(
            f"Shift % of {top_payer} revenue → other payers",
            min_value=0, max_value=60, value=0, step=5,
            help="Simulates targeted acquisition of underrepresented carriers.",
        )
        top_n_redistribute = st.slider(
            "Spread across top-N alternative payers",
            min_value=1, max_value=max(1, len(concent_df) - 1),
            value=min(3, max(1, len(concent_df) - 1)), step=1,
            help="Diverted revenue is distributed proportionally to these payers.",
        )

    with lev2:
        st.markdown("**2 · Contract & denial levers**")
        fee_uplift_pct = st.slider(
            "Fee-schedule uplift on top-3 payers (%)",
            0, 25, 0, step=1,
            help="Renegotiated rates on highest-volume CPT/CDT codes.",
        )
        denial_reduction_pct = st.slider(
            "Denial / write-off reduction (%)",
            0, 80, 0, step=5,
            help="Closes the gap between billed and collected.",
        )
        dso_reduction_days = st.slider(
            "DSO reduction (days)",
            0, 45, 0, step=1,
            help="Faster cash conversion frees working capital.",
        )

    with lev3:
        st.markdown("**3 · Patient retention levers**")
        critical_n = (
            int((risk["risk_band"] == "Critical").sum())
            if "risk" in dir() and len(risk) else 0
        )
        churn_recovery_pct = st.slider(
            f"Reactivate % of {critical_n:,} Critical-band patients",
            0, 60, 0, step=5,
            help="Re-engagement campaigns for high-value churned patients.",
        )
        m1_retention_uplift = st.slider(
            "Month-1 cohort retention uplift (pp)",
            0, 30, 0, step=2,
            help="Better onboarding / first-visit experience.",
        )
        nrr_uplift_pct = st.slider(
            "Net revenue retention uplift on active cohorts (%)",
            0, 25, 0, step=1,
            help="Cross-sell / upsell to existing patient cohorts.",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 2 · Apply levers
    # ──────────────────────────────────────────────────────────────────────────
    sim = base.copy()

    # --- Lever 1: diversification (move share from top payer to top-N others) ---
    sim_concent = concent_df.copy()
    sim_concent["share_pct_sim"] = sim_concent["share_pct"].astype(float)
    if divert_pct > 0 and len(sim_concent) > 1:
        moved = sim_concent.loc[0, "share_pct_sim"] * (divert_pct / 100.0)
        sim_concent.loc[0, "share_pct_sim"] -= moved
        recipients = sim_concent.iloc[1:1 + top_n_redistribute]
        if recipients["share_pct"].sum() > 0:
            weights = recipients["share_pct"] / recipients["share_pct"].sum()
            sim_concent.loc[recipients.index, "share_pct_sim"] += moved * weights.values
    sim_concent["cumulative_sim"] = sim_concent["share_pct_sim"].cumsum()
    sim_hhi = float((sim_concent["share_pct_sim"] ** 2).sum())

    # --- Lever 2a: fee uplift on the top-3 payers by billed ---
    top3 = sim.nlargest(3, "billed").index
    sim.loc[top3, "billed"]    *= (1 + fee_uplift_pct / 100.0)
    sim.loc[top3, "collected"] *= (1 + fee_uplift_pct / 100.0)

    # --- Lever 2b: denial reduction (close billed-vs-collected gap) ---
    gap = (sim["billed"] - sim["collected"]).clip(lower=0)
    sim["collected"] += gap * (denial_reduction_pct / 100.0)

    # --- Lever 2c: DSO reduction (compresses outstanding receivables) ---
    if dso_reduction_days > 0 and baseline_dso > 0:
        dso_factor = max(0.0, 1 - dso_reduction_days / baseline_dso)
        sim["outstanding"] *= dso_factor

    # --- Lever 3a: churn recovery → incremental collected revenue ---
    median_ticket = (
        float(pd.to_numeric(rfm_df["avg_ticket"], errors="coerce").median() or 0)
        if have_rfm else 0
    )
    recovered_patients = critical_n * (churn_recovery_pct / 100.0)
    churn_recovery_value = recovered_patients * median_ticket

    # --- Lever 3b + 3c: cohort uplift (retention + NRR) ---
    cohort_uplift_value = 0.0
    if have_cohort:
        ch = cohort_df.copy()
        ch["revenue"] = pd.to_numeric(ch["revenue"], errors="coerce").fillna(0)
        m1_rev = ch.loc[ch["month_offset"] == 1, "revenue"].sum()
        active_rev = ch.loc[ch["month_offset"] >= 1, "revenue"].sum()
        cohort_uplift_value = (
            m1_rev * (m1_retention_uplift / 100.0) +
            active_rev * (nrr_uplift_pct / 100.0)
        )

    # Distribute the patient-side uplift proportionally across payer mix
    if (churn_recovery_value + cohort_uplift_value) > 0 and sim["collected"].sum() > 0:
        weights = sim["collected"] / sim["collected"].sum()
        sim["collected"] += (churn_recovery_value + cohort_uplift_value) * weights
        sim["billed"]    += (churn_recovery_value + cohort_uplift_value) * weights

    # Refresh derived columns
    sim["collection_rate_pct"] = np.where(
        sim["billed"] > 0, 100 * sim["collected"] / sim["billed"], 0
    )
    sim["avg_dso"] = sim["avg_dso"] * (
        max(0.0, 1 - dso_reduction_days / baseline_dso) if baseline_dso else 1.0
    )

    sim_total_billed     = float(sim["billed"].sum())
    sim_total_collected  = float(sim["collected"].sum())
    sim_total_outstanding = float(sim["outstanding"].sum())
    sim_collection_rate  = (
        100 * sim_total_collected / sim_total_billed if sim_total_billed else 0
    )
    sim_dso = float((sim["avg_dso"] * sim["billed"]).sum() / max(sim_total_billed, 1))

    delta_collected   = sim_total_collected - baseline_total_collected
    delta_outstanding = sim_total_outstanding - baseline_total_outstanding
    delta_rate_pp     = sim_collection_rate - baseline_collection_rate
    delta_dso         = sim_dso - baseline_dso
    delta_hhi         = sim_hhi - baseline_hhi

    # ──────────────────────────────────────────────────────────────────────────
    # 3 · KPI strip (before → after)
    # ──────────────────────────────────────────────────────────────────────────
    def _kpi(label, before, after, delta, fmt="ksh", reverse=False):
        if fmt == "ksh":
            before_s, after_s = fmt_ksh(before), fmt_ksh(after)
            delta_s = ("+" if delta >= 0 else "") + fmt_ksh(delta)
        elif fmt == "pct":
            before_s, after_s = f"{before:.1f}%", f"{after:.1f}%"
            delta_s = f"{'+' if delta >= 0 else ''}{delta:.1f} pp"
        elif fmt == "days":
            before_s, after_s = f"{before:.0f} d", f"{after:.0f} d"
            delta_s = f"{'+' if delta >= 0 else ''}{delta:.0f} d"
        else:
            before_s, after_s = f"{before:.0f}", f"{after:.0f}"
            delta_s = f"{'+' if delta >= 0 else ''}{delta:.0f}"

        good = (delta >= 0) ^ reverse
        colour = COLORS["green"] if good else COLORS["danger"]
        return (
            f"<div style='border:1px solid #EBF3FB;border-radius:10px;"
            f"padding:10px 12px;background:#FAFCFE;'>"
            f"<div style='color:#6B8CAE;font-size:11px;text-transform:uppercase;"
            f"letter-spacing:0.4px;'>{label}</div>"
            f"<div style='color:#0F2A47;font-size:18px;font-weight:600;"
            f"margin-top:2px;'>{after_s}</div>"
            f"<div style='color:#6B8CAE;font-size:11px;margin-top:2px;'>"
            f"baseline {before_s} · <span style='color:{colour};font-weight:600;'>"
            f"{delta_s}</span></div>"
            f"</div>"
        )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(_kpi("Collected", baseline_total_collected, sim_total_collected,
                     delta_collected, "ksh"), unsafe_allow_html=True)
    k2.markdown(_kpi("Collection %", baseline_collection_rate, sim_collection_rate,
                     delta_rate_pp, "pct"), unsafe_allow_html=True)
    k3.markdown(_kpi("Outstanding", baseline_total_outstanding, sim_total_outstanding,
                     delta_outstanding, "ksh", reverse=True), unsafe_allow_html=True)
    k4.markdown(_kpi("Avg DSO", baseline_dso, sim_dso, delta_dso, "days",
                     reverse=True), unsafe_allow_html=True)
    k5.markdown(_kpi("HHI (mix risk)", baseline_hhi, sim_hhi, delta_hhi, "int",
                     reverse=True), unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 4 · Mix before/after + waterfall of uplift sources
    # ──────────────────────────────────────────────────────────────────────────
    col_mix, col_water = st.columns([1.2, 1], gap="large")

    with col_mix:
        section_header("Payer mix · baseline vs simulated", margin_top=8)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Baseline share",
            x=sim_concent[payer_key_concent],
            y=sim_concent["share_pct"],
            marker=dict(color="#D6E4F0", line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Baseline %{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Simulated share",
            x=sim_concent[payer_key_concent],
            y=sim_concent["share_pct_sim"],
            marker=dict(color=COLORS["primary"], line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>Simulated %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            barmode="group",
            bargap=0.25,
            height=340,
            xaxis=dict(tickangle=-25, tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="Share %", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center",
                        font=dict(size=10, color="#6B8CAE"),
                        bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_water:
        section_header("Where the uplift comes from", margin_top=8)
        # Approximate decomposition of the collected-revenue delta
        contrib_fee   = baseline_total_collected * (fee_uplift_pct / 100.0)
        contrib_deny  = (baseline_total_billed - baseline_total_collected) * \
                        (denial_reduction_pct / 100.0)
        contrib_churn = churn_recovery_value
        contrib_cohort = cohort_uplift_value
        labels  = ["Baseline", "Fee uplift", "Denial recovery",
                   "Churn recovery", "Cohort uplift", "Simulated"]
        values  = [baseline_total_collected, contrib_fee, contrib_deny,
                   contrib_churn, contrib_cohort, sim_total_collected]
        measure = ["absolute", "relative", "relative", "relative", "relative", "total"]

        fig = go.Figure(go.Waterfall(
            x=labels, y=values, measure=measure,
            connector=dict(line=dict(color="#D6E4F0")),
            increasing=dict(marker=dict(color=COLORS["green"])),
            decreasing=dict(marker=dict(color=COLORS["danger"])),
            totals=dict(marker=dict(color=COLORS["primary"])),
            text=[fmt_ksh(v) for v in values], textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} KSh<extra></extra>",
        ))
        fig.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
            height=340,
            xaxis=dict(tickfont=dict(size=10, color="#6B8CAE")),
            yaxis=dict(title="KSh collected", gridcolor="#EBF3FB",
                       tickfont=dict(size=10, color="#6B8CAE")),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 5 · Per-payer simulated scoreboard
    # ──────────────────────────────────────────────────────────────────────────
    section_header("Simulated payer scoreboard", margin_top=8)
    delta_view = sim[[payer_key_score, "billed", "collected", "outstanding",
                      "collection_rate_pct", "avg_dso"]].copy()
    delta_view["Δ collected"] = sim["collected"].values - base["collected"].values
    delta_view.columns = ["Payer", "Billed (KSh)", "Collected (KSh)",
                          "Outstanding (KSh)", "Collection %", "Avg DSO",
                          "Δ collected"]

    st.dataframe(
        delta_view.style.format({
            "Billed (KSh)":      "{:,.0f}",
            "Collected (KSh)":   "{:,.0f}",
            "Outstanding (KSh)": "{:,.0f}",
            "Collection %":      "{:.1f}%",
            "Avg DSO":           "{:.0f}",
            "Δ collected":       "{:+,.0f}",
        }, na_rep="—")
        .background_gradient(subset=["Δ collected"], cmap="Greens")
        .background_gradient(subset=["Collection %"], cmap="RdYlGn"),
        hide_index=True, use_container_width=True, height=320,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 6 · Verdict card
    # ──────────────────────────────────────────────────────────────────────────
    verdict_colour = (
        COLORS["green"]   if sim_hhi < 1500 else
        COLORS["warning"] if sim_hhi < 2500 else
        COLORS["danger"]
    )
    verdict_label = (
        "competitive (well-diversified)" if sim_hhi < 1500 else
        "moderately concentrated"        if sim_hhi < 2500 else
        "HIGHLY concentrated"
    )
    info_card(
        f"With these levers, monthly collections move from "
        f"<b>{fmt_ksh(baseline_total_collected)}</b> → "
        f"<b>{fmt_ksh(sim_total_collected)}</b> "
        f"(<b>{'+' if delta_collected >= 0 else ''}{fmt_ksh(delta_collected)}</b>). "
        f"Outstanding AR falls by <b>{fmt_ksh(-delta_outstanding)}</b>, "
        f"DSO shifts <b>{delta_dso:+.0f} days</b>, and the payer mix becomes "
        f"<b>{verdict_label}</b> (HHI {baseline_hhi:.0f} → {sim_hhi:.0f}).",
        verdict_colour,
    )

# ─── Footer ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div style="margin-top:30px;padding-top:14px;border-top:1px solid #EBF3FB;'
    'font-size:10px;color:#6B8CAE;letter-spacing:1.5px;text-transform:uppercase">'
    'tenri · Revenue Intelligence · Powered by Snowflake'
    '</div>',
    unsafe_allow_html=True,
)