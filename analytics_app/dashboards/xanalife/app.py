"""
Xanalife Analytics — Executive Dashboard
Run: streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import connection as conn
import cash_integrity_analysis as ci
import scr_analysis as scr
import sales_analysis as sa
import margin_analysis as ma
import overview_analysis as ov

st.set_page_config(page_title="Xanalife Analytics", layout="wide", initial_sidebar_state="expanded")

# ── Shared theme ───────────────────────────────────────────────────────────────

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

# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_ksh(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"KSh {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"KSh {v/1_000:.1f}K"
    return f"KSh {v:.0f}"

def kpi_card(label, value, sub="", color="#003467"):
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;padding:18px 16px">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">{sub}</div>'
        f'</div>', unsafe_allow_html=True)

def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)

def info_card(text, border_color="#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;border-left:3px solid {border_color};'
        f'border-radius:4px;font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

def cl(**overrides):
    """Merge CHART_LAYOUT with per-chart overrides — avoids duplicate keyword arg errors."""
    return {**CHART_LAYOUT, **overrides}

COLORS = {
    "primary": "#0072CE", "success": "#0BB99F", "warning": "#D97706",
    "danger":  "#E11D48", "muted":   "#6B8CAE", "purple":  "#7F77DD",
    "coral":   "#D85A30", "green":   "#1D9E75",
}

# Kenya school terms — for trend chart annotations
KE_SCHOOL_TERMS = [
    ("2025-09-01", "2025-11-14", "Term 3 2025"),
    ("2026-01-06", "2026-04-04", "Term 1 2026"),
    ("2026-05-04", "2026-08-14", "Term 2 2026"),
]

# ── Session state ──────────────────────────────────────────────────────────────

for key in ("conn", "ci_data", "scr_data", "sa_data", "ma_data", "ov_data", "stores_df"):
    if key not in st.session_state:
        st.session_state[key] = None if key in ("conn", "stores_df") else {}

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    try:
        st.image("download.png", width=150)
    except Exception:
        try:
            st.image("../download.png", width=150)
        except Exception:
            st.markdown(
                '<div style="font-size:16px;font-weight:800;color:#0072CE;padding:8px 0 16px">Xanalife</div>',
                unsafe_allow_html=True)

    section_header("Connection")
    passcode = st.text_input("TOTP Passcode", type="password", max_chars=6, placeholder="6-digit code")
    connect_btn = st.button("Connect", type="primary")

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    section_header("Module", margin_top=8)
    page = st.radio("", ["Overview", "Revenue Intelligence", "Cash Integrity", "SCR", "Margin Intelligence"],
                    label_visibility="collapsed")

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    section_header("Filters", margin_top=8)

    _PHARMACY_CODES = ["PHARMACY", "KP"]

    if st.session_state.stores_df is not None:
        location = st.radio("Location", ["All", "Syokimau", "Katani"], horizontal=True)
        division = st.radio("Division", ["All", "Supermarket", "Pharmacy"], horizontal=True)

        _pool = st.session_state.stores_df.copy()
        if location != "All":
            _pool = _pool[_pool["LOCATION"] == location]
        if division == "Pharmacy":
            _pool = _pool[_pool["STORE_CODE"].isin(_PHARMACY_CODES)]
        elif division == "Supermarket":
            _pool = _pool[~_pool["STORE_CODE"].isin(_PHARMACY_CODES)]

        _store_pool = _pool["STORE_NAME"].tolist()
        selected_stores = st.multiselect("Stores", _store_pool, default=_store_pool)
        st.caption(
            "Applies to: Revenue · Basket · Margin · SCR · Stockout\n"
            "Not filterable: Cash Integrity · Invoices · Loyalty"
        )
    else:
        selected_stores = []
        st.caption("Connect to enable filters.")

# ── Connection ─────────────────────────────────────────────────────────────────

@st.cache_resource
def get_conn(p: str):
    return conn.connect(p)

if connect_btn and passcode:
    try:
        st.session_state.conn = get_conn(passcode)
        stores_raw = ov.run_query(
            "SELECT id AS STORE_ID, MIN(name) AS STORE_NAME, MIN(code) AS STORE_CODE FROM hospitals.xanalife_clean.inventory_stores GROUP BY id ORDER BY MIN(name)",
            st.session_state.conn)
        stores_raw["LOCATION"] = stores_raw["STORE_ID"].apply(
            lambda x: "Syokimau" if x in {399, 400, 401, 402, 403, 404} else "Katani"
        )
        st.session_state.stores_df = stores_raw
        st.sidebar.success("Connected")
        st.rerun()
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

if st.session_state.conn is None:
    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:16px">Xanalife · Analytics</p>',
        unsafe_allow_html=True)
    info_card("Enter your TOTP passcode in the sidebar and click Connect to load the dashboard.", COLORS["muted"])
    st.stop()

# ── Store filter — resolve defaults and compute cache key ─────────────────────

if not selected_stores and st.session_state.stores_df is not None:
    selected_stores = st.session_state.stores_df["STORE_NAME"].tolist()

filter_key = frozenset(selected_stores)

# When all stores are selected, pass [] to get_analyses — bypasses the IN subquery
# and restores full revenue (51.11M). The IN subquery excludes ~231 transactions
# whose store_product_ids have no match in inventory_store_products (~1% of rows).
_all_store_names = (frozenset(st.session_state.stores_df["STORE_NAME"].tolist())
                    if st.session_state.stores_df is not None else frozenset())
effective_stores = [] if filter_key == _all_store_names else selected_stores

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "Overview":

    expected_ov = {label for label, _ in ov.get_analyses()}
    if (not st.session_state.ov_data
            or st.session_state.ov_data.get("_filter_key") != filter_key
            or not expected_ov.issubset(st.session_state.ov_data.keys())):
        with st.spinner("Loading overview…"):
            st.session_state.ov_data = {"_filter_key": filter_key}
            for label, sql in ov.get_analyses(effective_stores):
                st.session_state.ov_data[label] = ov.run_query(sql, st.session_state.conn)

    O = st.session_state.ov_data

    rev   = O["Revenue Summary"].iloc[0]
    trend = O["Revenue Trend"]
    cash  = O["Cash Summary"].iloc[0]
    stk   = O["Stockout Count"].iloc[0]
    inv   = O["Invoices Summary"].iloc[0]
    loy   = O["Loyalty Summary"].iloc[0]

    for col in ["TOTAL_REVENUE", "REVENUE_THIS_MONTH", "REVENUE_LAST_MONTH", "TOTAL_TRANSACTIONS"]:
        rev[col] = pd.to_numeric(rev[col], errors="coerce") or 0
    latest_month_label = pd.to_datetime(rev.get("LATEST_DATA_MONTH", pd.NaT))
    month_label = latest_month_label.strftime("%b %Y") if pd.notna(latest_month_label) else "Latest month"
    for col in ["NET_VARIANCE", "CASH_AT_RISK", "VARIANCE_PCT", "UNCLOSED_SHIFTS", "ANOMALOUS_SHIFTS"]:
        cash[col] = pd.to_numeric(cash[col], errors="coerce") or 0
    for col in ["PRODUCTS_AT_ZERO_STOCK"]:
        stk[col] = pd.to_numeric(stk[col], errors="coerce") or 0
    for col in ["TOTAL_INVOICES", "TOTAL_AMOUNT", "TOTAL_BALANCE"]:
        inv[col] = pd.to_numeric(inv[col], errors="coerce") or 0
    for col in ["TOTAL_EARNED", "REDEMPTION_COUNT", "TOTAL_REDEEMED", "CUSTOMERS_WITH_POINTS"]:
        loy[col] = pd.to_numeric(loy[col], errors="coerce") or 0
    trend["DAILY_REVENUE"] = pd.to_numeric(trend["DAILY_REVENUE"], errors="coerce").fillna(0)
    trend["SALE_DATE"]     = pd.to_datetime(trend["SALE_DATE"])

    mom_delta = rev["REVENUE_THIS_MONTH"] - rev["REVENUE_LAST_MONTH"]
    mom_pct   = (mom_delta / rev["REVENUE_LAST_MONTH"] * 100) if rev["REVENUE_LAST_MONTH"] > 0 else 0

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Xanalife · Executive Overview</p>',
        unsafe_allow_html=True)
    st.caption(f"Sep 2025 – present · As of today")
    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total Revenue (YTD)", fmt_ksh(rev["TOTAL_REVENUE"]),
                 f"{int(rev['TOTAL_TRANSACTIONS']):,} transactions", COLORS["primary"])
    with c2:
        kpi_card(month_label, fmt_ksh(rev["REVENUE_THIS_MONTH"]),
                 f"{'+' if mom_pct >= 0 else ''}{mom_pct:.1f}% vs prior month",
                 COLORS["success"] if mom_pct >= 0 else COLORS["danger"])
    with c3:
        kpi_card("Cash Shortfall", fmt_ksh(abs(cash["NET_VARIANCE"])),
                 f"{abs(cash['VARIANCE_PCT']):.2f}% of cash handled · {int(cash['ANOMALOUS_SHIFTS'])} anomalous shifts",
                 COLORS["danger"])
    with c4:
        kpi_card("Outstanding Invoices", fmt_ksh(inv["TOTAL_BALANCE"]),
                 f"{int(inv['TOTAL_INVOICES'])} invoices — none collected",
                 COLORS["warning"])
    with c5:
        kpi_card("Loyalty Points Issued", f"{int(loy['TOTAL_EARNED']):,}",
                 f"{int(loy['REDEMPTION_COUNT'])} redemptions · programme near-dormant",
                 COLORS["muted"])

    st.markdown("<div style='margin-bottom:24px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 2])

    with left:
        section_header("Act Now")

        alerts = []
        if cash["ANOMALOUS_SHIFTS"] > 0:
            alerts.append(("danger",
                f"🔴 {int(cash['ANOMALOUS_SHIFTS'])} shifts where variance exceeded total sales — "
                "impossible under honest operation. Pull receipts immediately."))
        if abs(cash["VARIANCE_PCT"]) > 1:
            alerts.append(("danger",
                f"🔴 Cash shortfall is {abs(cash['VARIANCE_PCT']):.1f}% of revenue. "
                f"{fmt_ksh(abs(cash['NET_VARIANCE']))} unaccounted across all closed shifts."))
        if cash["UNCLOSED_SHIFTS"] > 100:
            alerts.append(("warning",
                f"🟡 {int(cash['UNCLOSED_SHIFTS']):,} shifts never closed. "
                "Revenue in those shifts has no variance accountability."))
        if inv["TOTAL_BALANCE"] > 0:
            alerts.append(("warning",
                f"🟡 {fmt_ksh(inv['TOTAL_BALANCE'])} in outstanding invoices — "
                f"{int(inv['TOTAL_INVOICES'])} invoices, none collected. Finance action required."))
        if loy["REDEMPTION_COUNT"] < 10:
            alerts.append(("muted",
                f"⚪ Loyalty programme: {int(loy['TOTAL_EARNED']):,} points issued, "
                f"only {int(loy['REDEMPTION_COUNT'])} redeemed. "
                "Programme costs without retention benefit — review or relaunch."))

        for severity, text in alerts:
            info_card(text, COLORS[severity])

        if not alerts:
            info_card("No critical alerts at this time.", COLORS["success"])

    with right:
        section_header("Revenue — Last 60 Days")

        rolling = trend["DAILY_REVENUE"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["SALE_DATE"], y=trend["DAILY_REVENUE"],
            mode="lines", name="Daily", line=dict(color=COLORS["primary"], width=1),
            opacity=0.35,
            hovertemplate="%{x|%b %d}<br>KSh %{y:,.0f}<extra>Daily</extra>"))
        fig.add_trace(go.Scatter(
            x=trend["SALE_DATE"], y=rolling,
            mode="lines", name="7-day avg", line=dict(color=COLORS["primary"], width=2.5),
            hovertemplate="%{x|%b %d}<br>KSh %{y:,.0f}<extra>7-day avg</extra>"))
        fig.update_layout(**CHART_LAYOUT, height=280, showlegend=True,
                          legend=dict(orientation="h", y=1.08, x=0,
                                      font=dict(size=10, color=COLORS["muted"])))
        fig.update_yaxes(tickprefix="KSh ", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    section_header("Executive Recommendation", margin_top=8)
    info_card(
        f"Revenue is {'growing' if mom_pct > 0 else 'declining'} {abs(mom_pct):.1f}% month-on-month. "
        f"The {fmt_ksh(abs(cash['NET_VARIANCE']))} cash shortfall ({abs(cash['VARIANCE_PCT']):.1f}% of cash handled) "
        f"is the highest-priority operational risk — it is driven by individuals, not systems, and will not "
        f"resolve without direct management intervention. The {fmt_ksh(inv['TOTAL_BALANCE'])} in uncollected "
        f"invoices is a receivables gap that compounds every month it is not addressed.",
        COLORS["primary"])


# ══════════════════════════════════════════════════════════════════════════════
# CASH INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════

if page == "Cash Integrity":

    expected_ci = {label for label, _ in ci.ANALYSES}
    if not st.session_state.ci_data or not expected_ci.issubset(st.session_state.ci_data.keys()):
        with st.spinner("Loading…"):
            st.session_state.ci_data = {}
            for label, sql in ci.ANALYSES:
                st.session_state.ci_data[label] = ci.run_query(sql, st.session_state.conn)

    D        = st.session_state.ci_data
    pareto   = D["Analysis 1 — Pareto by Station"].copy()
    trend    = D["Analysis 2 — Monthly Variance Trend"].copy()
    anomaly  = D["Analysis 3 — Anomalous Shifts"].copy()
    exposure = D["Analysis 5 — Unclosed Shift Exposure"].copy()
    by_user  = D["Analysis 6 — Unclosed by Station"].copy()

    for col in ["NET_VARIANCE", "SHIFTS", "CASH_AT_RISK", "VARIANCE_PCT", "PCT_OF_LOSS_POOL"]:
        pareto[col] = pd.to_numeric(pareto[col], errors="coerce")
        if col in trend.columns:
            trend[col] = pd.to_numeric(trend[col], errors="coerce")
    for col in ["UNCLOSED_SHIFTS", "REVENUE_IN_UNCLOSED", "AVG_REVENUE_UNCLOSED"]:
        exposure[col] = pd.to_numeric(exposure[col], errors="coerce")
    for col in ["UNCLOSED_RATE_PCT", "REVENUE_UNCLOSED_KES", "UNCLOSED", "TOTAL_SHIFTS"]:
        by_user[col] = pd.to_numeric(by_user[col], errors="coerce")
    for col in ["SALES", "VARIANCE"]:
        if col in anomaly.columns:
            anomaly[col] = pd.to_numeric(anomaly[col], errors="coerce")

    total_var        = pareto["NET_VARIANCE"].sum()
    top1             = pareto.nsmallest(1, "NET_VARIANCE").iloc[0]
    top3             = pareto.nsmallest(3, "NET_VARIANCE")
    _loss_pool       = abs(pareto[pareto["NET_VARIANCE"] < 0]["NET_VARIANCE"].sum())
    top3_pct         = abs(top3["NET_VARIANCE"].sum() / _loss_pool * 100) if _loss_pool > 0 else 0
    unclosed_count   = int(exposure["UNCLOSED_SHIFTS"].iloc[0])
    revenue_unclosed = float(exposure["REVENUE_IN_UNCLOSED"].iloc[0])
    ts               = trend.sort_values("MONTH")
    first_var        = float(ts["VARIANCE_PCT"].iloc[0])
    last_var         = float(ts["VARIANCE_PCT"].iloc[-1])
    worsening        = last_var < first_var

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Xanalife · Cash Integrity</p>',
        unsafe_allow_html=True)
    st.caption("Sep 2025 – present · Closed shifts only · System/admin users excluded")
    info_card(
        "Note: supervisor accounts appear in raw shift data and are excluded from this analysis. "
        "These records hold no cashier intelligence — flag to engineer for system cleanup.",
        COLORS["muted"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Net Cash Shortfall", fmt_ksh(abs(total_var)),
                 f"Top 3 users = {top3_pct:.0f}% of all losses", COLORS["danger"])
    with c2:
        kpi_card("Anomalous Shifts", str(len(anomaly)),
                 "Variance exceeded total sales — impossible under honest operation", COLORS["danger"])
    with c3:
        kpi_card("Unreconciled Shifts", f"{unclosed_count:,}",
                 f"{fmt_ksh(revenue_unclosed)} revenue with no variance captured", COLORS["warning"])
    with c4:
        kpi_card("Variance Trend",
                 "Worsening ↓" if worsening else "Stable / Improving",
                 f"{first_var:.2f}% → {last_var:.2f}% since go-live",
                 COLORS["danger"] if worsening else COLORS["success"])

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["◉  Pareto", "△  Trend", "∑  Shift Audit"])

    with tab1:
        section_header("Variance by User — Pareto")
        top_n = pareto.nsmallest(15, "NET_VARIANCE")
        fig = go.Figure(go.Bar(
            x=top_n["NET_VARIANCE"],
            y=top_n["USER_ID"].astype(str),
            orientation="h",
            marker_color=[COLORS["danger"] if v < -50000 else COLORS["warning"]
                          for v in top_n["NET_VARIANCE"]],
            hovertemplate="User %{y}<br>Variance: KSh %{x:,.0f}<br><extra></extra>"))
        fig.update_layout(**cl(yaxis=dict(autorange="reversed", tickfont=dict(size=10))),
                          height=380, xaxis_title="Net Variance (KSh)")
        fig.update_xaxes(tickprefix="KSh ", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        section_header("Decisions", margin_top=16)
        decisions = [
            {"Severity": "Critical",
             "Finding": f"User {top1['USER_ID']} — {fmt_ksh(abs(top1['NET_VARIANCE']))} shortfall ({abs(top1['PCT_OF_LOSS_POOL']):.0f}% of total losses)",
             "Action": "Needs Assessment."},
            {"Severity": "Critical",
             "Finding": f"{len(anomaly)} shifts where variance exceeded total sales",
             "Action": "shift logs for all flagged shifts. Cross-reference opening balances."},
            {"Severity": "High",
             "Finding": f"{unclosed_count:,} shifts never closed — {fmt_ksh(revenue_unclosed)} unreconciled",
             "Action": "Enforce: no new shift opens until prior shift is closed."},
            {"Severity": "High" if worsening else "Monitor",
             "Finding": f"Variance trend {first_var:.2f}% → {last_var:.2f}% ({'worsening' if worsening else 'stable'})",
             "Action": ("Spot Audit Shift Records. "
                        "Remove system test from the data.")
             if worsening else "Reinforce current controls. Re-evaluate in 30 days."},
        ]
        st.dataframe(pd.DataFrame(decisions), use_container_width=True, hide_index=True)

    with tab2:
        section_header("Monthly Variance Trend")
        ts2 = trend.sort_values("MONTH")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ts2["MONTH"], y=ts2["VARIANCE_PCT"],
            mode="lines+markers", name="Variance %",
            line=dict(color=COLORS["danger"], width=2),
            marker=dict(size=6),
            hovertemplate="%{x|%b %Y}<br>Variance: %{y:.2f}%<extra></extra>"))
        fig2.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], line_width=1)
        fig2.update_layout(**CHART_LAYOUT, height=300, yaxis_title="Variance % of Revenue")
        fig2.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        section_header("Cash at Risk vs Variance", margin_top=16)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=ts2["MONTH"], y=ts2["CASH_AT_RISK"], name="Cash at Risk",
            marker_color=COLORS["primary"], opacity=0.7,
            hovertemplate="%{x|%b %Y}<br>Cash at Risk: KSh %{y:,.0f}<extra></extra>"))
        fig3.add_trace(go.Bar(
            x=ts2["MONTH"], y=ts2["NET_VARIANCE"].abs(), name="Abs Variance",
            marker_color=COLORS["danger"],
            hovertemplate="%{x|%b %Y}<br>Variance: KSh %{y:,.0f}<extra></extra>"))
        fig3.update_layout(**CHART_LAYOUT, height=280, barmode="overlay",
                           legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        fig3.update_yaxes(tickprefix="KSh ", tickformat=",.0f")
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        section_header(f"Anomalous Shifts — {len(anomaly)}")
        st.caption("Shifts where variance exceeded total sales. Share with HR and operations.")
        st.dataframe(anomaly, use_container_width=True, hide_index=True, height=250)

        section_header("Unreconciled Shifts by User", margin_top=16)
        st.caption("Users above 30% unclosed rate require immediate retraining.")

        if not by_user.empty:
            fig4 = go.Figure(go.Bar(
                x=by_user.sort_values("UNCLOSED_RATE_PCT", ascending=False)["USER_ID"].astype(str),
                y=by_user.sort_values("UNCLOSED_RATE_PCT", ascending=False)["UNCLOSED_RATE_PCT"],
                marker_color=[COLORS["danger"] if v > 30 else COLORS["warning"]
                              for v in by_user.sort_values("UNCLOSED_RATE_PCT", ascending=False)["UNCLOSED_RATE_PCT"]],
                hovertemplate="User %{x}<br>Unclosed rate: %{y:.1f}%<extra></extra>"))
            fig4.add_hline(y=30, line_dash="dash", line_color=COLORS["danger"], line_width=1,
                           annotation_text="30% threshold", annotation_position="top right")
            fig4.update_layout(**CHART_LAYOUT, height=280, yaxis_title="Unclosed Rate %")
            fig4.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Full unreconciled shifts table"):
            st.dataframe(by_user, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    section_header("Executive Recommendation", margin_top=8)
    info_card(
        f"The {fmt_ksh(abs(total_var))} shortfall is not random — top 3 users account for {top3_pct:.0f}% of all losses. "
        f"This is a pattern, not a coincidence. The {len(anomaly)} anomalous shifts (where variance exceeded sales) "
        f"are the highest-risk events in this dataset and warrant physical investigation. "
        f"{'Variance is worsening month-on-month — individual retraining will not fix a systemic drift. A policy change is required.' if worsening else 'Variance is stable. Maintain controls and re-evaluate monthly.'}",
        COLORS["danger"] if worsening else COLORS["primary"])

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    summary_dl = (pareto.merge(
        by_user[["USER_ID", "UNCLOSED", "UNCLOSED_RATE_PCT", "REVENUE_UNCLOSED_KES"]],
        on="USER_ID", how="outer").sort_values("NET_VARIANCE"))
    st.download_button("⬇ Download full user data (CSV)", summary_dl.to_csv(index=False),
                       "cash_integrity_user_summary.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# SCR — SUBSTITUTION CAPTURE RATE
# ══════════════════════════════════════════════════════════════════════════════

if page == "SCR":

    expected_scr = {label for label, _ in scr.get_analyses()}
    if (not st.session_state.scr_data
            or st.session_state.scr_data.get("_filter_key") != filter_key
            or not expected_scr.issubset(st.session_state.scr_data.keys())):
        with st.spinner("Loading…"):
            st.session_state.scr_data = {"_filter_key": filter_key}
            for label, sql in scr.get_analyses(effective_stores):
                st.session_state.scr_data[label] = scr.run_query(sql, st.session_state.conn)

    S         = st.session_state.scr_data
    summary   = S["SCR Summary — Stockouts + Top Substitute"].copy()
    cat_exp   = S["Category Exposure — Stockouts by Category"].copy()
    depletion = S["Depletion Risk — Days Until Stockout"].copy()

    for col in ["AVG_WEEKLY_UNITS", "GAP_DAYS", "GAP_OCCURRENCES",
                "SUBSTITUTE_UPLIFT", "REVENUE_AT_RISK_KES", "RECOMMENDED_PRESTOCK_QTY"]:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    cat_exp["STOCKOUT_PRODUCTS"] = pd.to_numeric(cat_exp["STOCKOUT_PRODUCTS"], errors="coerce")
    for col in ["DAYS_UNTIL_STOCKOUT", "CURRENT_STOCK", "AVG_DAILY_RATE"]:
        if col in depletion.columns:
            depletion[col] = pd.to_numeric(depletion[col], errors="coerce")

    total_kes    = float(summary["REVENUE_AT_RISK_KES"].sum())
    flagged      = len(summary)
    has_subs     = summary["SUBSTITUTE_UPLIFT"].notna().any() and summary["SUBSTITUTE_UPLIFT"].max() > 0
    best_sub     = summary.loc[summary["SUBSTITUTE_UPLIFT"].idxmax()] if has_subs else None
    top_cat      = cat_exp.iloc[0]["NAME"] if not cat_exp.empty else "—"

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Xanalife · Substitution Capture Rate</p>',
        unsafe_allow_html=True)
    st.caption("Stockout exposure, substitute intelligence, and depletion risk · Sep 2025 – present")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Revenue at Risk", fmt_ksh(total_kes),
                 f"Across {flagged} flagged products", COLORS["danger"] if total_kes > 0 else COLORS["muted"])
    with c2:
        kpi_card("Products Flagged in Stockout", str(flagged),
                 "Products with dispensing gap ≥ 14 days", COLORS["warning"])
    with c3:
        if best_sub is not None:
            kpi_card("Strongest Substitute", f"+{int(best_sub['SUBSTITUTE_UPLIFT'])} units",
                     f"{best_sub['TOP_SUBSTITUTE']} absorbed demand when {best_sub['PRODUCT_NAME']} was out",
                     COLORS["success"])
        else:
            kpi_card("Strongest Substitute", "—", "No substitute signal detected yet", COLORS["muted"])
    with c4:
        kpi_card("Most Exposed Category", str(top_cat),
                 "Category with highest number of stockout products", COLORS["warning"])

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["◉  Stockout Intelligence", "△  Depletion Watch"])

    with tab1:
        section_header("Category Exposure")
        if not cat_exp.empty:
            fig = go.Figure(go.Bar(
                x=cat_exp["STOCKOUT_PRODUCTS"],
                y=cat_exp["NAME"],
                orientation="h",
                marker_color=COLORS["warning"],
                hovertemplate="%{y}<br>%{x} products in stockout<extra></extra>"))
            fig.update_layout(**CHART_LAYOUT, height=max(200, len(cat_exp) * 28),
                              xaxis_title="Products with Stockout Gap")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        section_header("What to Order — and How Much", margin_top=16)
        st.caption("Ranked by revenue at risk. Pre-stock quantity includes 20% buffer over substitute absorption.")

        # Pagination (100 rows per page)
        page_size = 100
        n_pages   = max(1, int(np.ceil(len(summary) / page_size)))
        if "scr_page" not in st.session_state:
            st.session_state.scr_page = 0

        p_col1, p_col2, p_col3 = st.columns([1, 3, 1])
        with p_col1:
            if st.button("← Prev", disabled=st.session_state.scr_page == 0):
                st.session_state.scr_page -= 1
        with p_col2:
            st.caption(f"Page {st.session_state.scr_page + 1} of {n_pages} · {len(summary)} products total")
        with p_col3:
            if st.button("Next →", disabled=st.session_state.scr_page >= n_pages - 1):
                st.session_state.scr_page += 1

        start = st.session_state.scr_page * page_size
        display = (
            summary.iloc[start:start + page_size]
            [["PRODUCT_NAME", "AVG_WEEKLY_UNITS", "GAP_DAYS", "REVENUE_AT_RISK_KES",
              "TOP_SUBSTITUTE", "RECOMMENDED_PRESTOCK_QTY", "ACTION"]]
            .rename(columns={
                "PRODUCT_NAME": "Product", "AVG_WEEKLY_UNITS": "Avg Units/Week",
                "GAP_DAYS": "Gap (days)", "REVENUE_AT_RISK_KES": "Revenue at Risk (KES)",
                "TOP_SUBSTITUTE": "Top Substitute", "RECOMMENDED_PRESTOCK_QTY": "Pre-stock Qty",
                "ACTION": "Action"})
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

    with tab2:
        section_header("Depletion Risk — Days Until Stockout")
        st.caption("Products with < 30 days of stock at current dispensing rate.")

        if not depletion.empty:
            dep_sorted = depletion.sort_values("DAYS_UNTIL_STOCKOUT")
            color_map  = {"Critical": COLORS["danger"], "Warning": COLORS["warning"], "Watch": COLORS["success"]}
            colors_dep = [color_map.get(u, COLORS["muted"]) for u in dep_sorted.get("URGENCY", [])]

            fig2 = go.Figure(go.Bar(
                x=dep_sorted["PRODUCT_NAME"],
                y=dep_sorted["DAYS_UNTIL_STOCKOUT"],
                marker_color=colors_dep,
                hovertemplate="%{x}<br>Days left: %{y}<extra></extra>"))
            fig2.add_hline(y=7,  line_dash="dash", line_color=COLORS["danger"],
                           line_width=1, annotation_text="Critical (7d)")
            fig2.add_hline(y=14, line_dash="dash", line_color=COLORS["warning"],
                           line_width=1, annotation_text="Warning (14d)")
            fig2.update_layout(**cl(xaxis=dict(tickangle=-35, tickfont=dict(size=9))),
                               height=340, yaxis_title="Days Until Stockout")
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Full depletion table"):
            st.dataframe(
                depletion[["PRODUCT_NAME", "CURRENT_STOCK", "AVG_DAILY_RATE",
                           "DAYS_UNTIL_STOCKOUT", "URGENCY"]]
                .rename(columns={"PRODUCT_NAME": "Product", "CURRENT_STOCK": "Units in Stock",
                                 "AVG_DAILY_RATE": "Daily Rate", "DAYS_UNTIL_STOCKOUT": "Days Left",
                                 "URGENCY": "Urgency"}),
                use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    section_header("Executive Recommendation", margin_top=8)
    if total_kes > 0:
        top3_risk = summary.nlargest(3, "REVENUE_AT_RISK_KES")["PRODUCT_NAME"].tolist()
        info_card(
            f"{fmt_ksh(total_kes)} in revenue is exposed to stockout gaps across {flagged} products. "
            f"The top 3 by risk — {', '.join(top3_risk)} — likely account for the majority of that exposure. "
            f"A single procurement order covering these three resolves most of the risk. "
            f"Where a substitute is identified, pre-stock it before the primary product hits its reorder level — "
            f"don't wait for the gap to open.", COLORS["warning"])
    else:
        info_card("No active stockout revenue risk detected.", COLORS["success"])

    st.download_button("⬇ Download SCR data (CSV)", summary.to_csv(index=False),
                       "scr_stockouts_and_substitutes.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# REVENUE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

if page == "Revenue Intelligence":

    expected_sa = {label for label, _ in sa.get_analyses()}
    if (not st.session_state.sa_data
            or st.session_state.sa_data.get("_filter_key") != filter_key
            or not expected_sa.issubset(st.session_state.sa_data.keys())):
        with st.spinner("Loading…"):
            st.session_state.sa_data = {"_filter_key": filter_key}
            for label, sql in sa.get_analyses(effective_stores):
                st.session_state.sa_data[label] = sa.run_query(sql, st.session_state.conn)

    A              = st.session_state.sa_data
    daily_df       = A["Daily Revenue"].copy()
    basket_sum     = A["Basket Summary"].copy()
    top_alone      = A["Top Single-Item Products"].copy()
    by_store       = A["Basket by Store"].copy()
    peak_df        = A["Peak Revenue Heatmap"].copy()
    rev_by_store   = A["Revenue by Store"].copy()
    top_prods_loc  = A["Top Products by Location"].copy()
    loc_opp        = A["Location Opportunity"].copy()

    for col in ["TRANSACTIONS", "DAILY_REVENUE"]:
        daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
    for col in ["TOTAL_TRANSACTIONS", "AVG_ITEMS_PER_SALE", "AVG_BASKET_VALUE_KES",
                "SINGLE_ITEM_PCT", "MULTI_ITEM_PCT"]:
        basket_sum[col] = pd.to_numeric(basket_sum[col], errors="coerce")
    for col in ["TIMES_BOUGHT_ALONE", "PCT_OF_SINGLE_ITEM_SALES"]:
        top_alone[col] = pd.to_numeric(top_alone[col], errors="coerce")
    for col in ["TRANSACTIONS", "AVG_ITEMS", "AVG_BASKET_KES", "SINGLE_ITEM_PCT"]:
        by_store[col] = pd.to_numeric(by_store[col], errors="coerce")
    for col in ["DAY_NUM", "HOUR_OF_DAY", "TRANSACTIONS", "REVENUE"]:
        peak_df[col] = pd.to_numeric(peak_df[col], errors="coerce")
    for col in ["REVENUE", "TRANSACTIONS"]:
        rev_by_store[col] = pd.to_numeric(rev_by_store[col], errors="coerce")
    for col in ["REVENUE", "TRANSACTIONS"]:
        top_prods_loc[col] = pd.to_numeric(top_prods_loc[col], errors="coerce")
    for col in ["SYOKIMAU_REVENUE", "KATANI_REVENUE", "GAP"]:
        loc_opp[col] = pd.to_numeric(loc_opp[col], errors="coerce")

    hist_df, forecast_df = sa.build_forecast(daily_df)

    total_rev  = float(daily_df["DAILY_REVENUE"].sum())
    proj_30d   = float(forecast_df["FORECAST"].sum())
    single_pct = float(basket_sum["SINGLE_ITEM_PCT"].iloc[0]) if not basket_sum.empty else 0.0
    avg_basket = float(basket_sum["AVG_BASKET_VALUE_KES"].iloc[0]) if not basket_sum.empty else 0.0
    peak_row   = peak_df.loc[peak_df["REVENUE"].idxmax()] if not peak_df.empty else None
    peak_label = (f"{peak_row['DAY_NAME']} {int(peak_row['HOUR_OF_DAY']):02d}:00"
                  if peak_row is not None else "—")

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Xanalife · Revenue Intelligence</p>',
        unsafe_allow_html=True)
    st.caption("Sep 2025 – present · POS sales · All stores")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Revenue (YTD)", fmt_ksh(total_rev),
                 f"→ {fmt_ksh(proj_30d)} projected next 30 days", COLORS["primary"])
    with c2:
        kpi_card("Avg Basket Value", fmt_ksh(avg_basket),
                 f"{float(basket_sum['AVG_ITEMS_PER_SALE'].iloc[0]) if not basket_sum.empty else 0:.1f} items per transaction",
                 COLORS["success"])
    with c3:
        kpi_card("Single-Item Basket Rate", f"{single_pct:.1f}%",
                 "Transactions with only one product — cross-sell opportunity",
                 COLORS["warning"] if single_pct > 50 else COLORS["muted"])
    with c4:
        kpi_card("Peak Revenue Slot", peak_label,
                 f"KSh {float(peak_df['REVENUE'].max()):,.0f} cumulative revenue" if peak_row is not None else "",
                 COLORS["purple"])

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["◉  Revenue Trend", "△  Basket Analysis", "∑  Peak Windows", "◎  By Location"])

    with tab1:
        section_header("Revenue Trend + 30-Day Forecast")
        st.caption("Daily actual · 7-day rolling average · Linear projection · Red lines = public holidays · Bands = school terms")

        fig = go.Figure()

        # School term shaded bands
        for start_d, end_d, label_t in KE_SCHOOL_TERMS:
            fig.add_vrect(x0=start_d, x1=end_d,
                          fillcolor=COLORS["primary"], opacity=0.04,
                          line_width=0,
                          annotation_text=label_t,
                          annotation_position="top left",
                          annotation_font_size=9,
                          annotation_font_color=COLORS["muted"])

        fig.add_trace(go.Scatter(
            x=hist_df["SALE_DATE"], y=hist_df["DAILY_REVENUE"],
            mode="lines", name="Daily", line=dict(color=COLORS["primary"], width=1), opacity=0.3,
            hovertemplate="%{x|%b %d}<br>KSh %{y:,.0f}<extra>Daily</extra>"))
        fig.add_trace(go.Scatter(
            x=hist_df["SALE_DATE"], y=hist_df["ROLLING_7"],
            mode="lines", name="7-day avg", line=dict(color=COLORS["primary"], width=2.5),
            hovertemplate="%{x|%b %d}<br>KSh %{y:,.0f}<extra>7-day avg</extra>"))
        fig.add_trace(go.Scatter(
            x=forecast_df["SALE_DATE"], y=forecast_df["FORECAST"],
            mode="lines", name="Forecast", line=dict(color=COLORS["warning"], width=2, dash="dash"),
            hovertemplate="%{x|%b %d}<br>KSh %{y:,.0f}<extra>Forecast</extra>"))

        for hd in sa.KE_PUBLIC_HOLIDAYS:
            fig.add_vline(x=str(hd.date()), line_dash="dot",
                          line_color=COLORS["danger"], line_width=1.5, opacity=0.7)

        fig.update_layout(**CHART_LAYOUT, height=360,
                          legend=dict(orientation="h", y=1.08, font=dict(size=10)))
        fig.update_yaxes(tickprefix="KSh ", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        st.download_button(
            "⬇ Daily revenue + forecast (CSV)",
            pd.concat([
                hist_df[["SALE_DATE", "DAILY_REVENUE", "ROLLING_7", "TREND"]].assign(type="actual"),
                forecast_df.rename(columns={"FORECAST": "DAILY_REVENUE"}).assign(type="forecast"),
            ]).to_csv(index=False),
            "revenue_trend_forecast.csv", "text/csv")

    with tab2:
        section_header("Basket Intelligence")

        avg_items = float(basket_sum["AVG_ITEMS_PER_SALE"].iloc[0]) if not basket_sum.empty else 0.0
        multi_pct = float(basket_sum["MULTI_ITEM_PCT"].iloc[0]) if not basket_sum.empty else 0.0

        lift_1item = ((avg_items + 1) / avg_items - 1) * 100 if avg_items > 0 else 0
        info_card(
            f"Average basket is <b>{avg_items:.1f} items</b> — a 1-item lift per transaction would increase basket revenue "
            f"by ~{lift_1item:.0f}%. With {int(basket_sum['TOTAL_TRANSACTIONS'].iloc[0]):,} transactions, "
            f"that is {fmt_ksh((avg_basket / avg_items) * basket_sum['TOTAL_TRANSACTIONS'].iloc[0])} "
            f"in additional annual revenue at current volume. The products below are the lever.",
            COLORS["primary"])

        b1, b2 = st.columns([1, 2])
        with b1:
            # Basket composition donut
            multi_count  = float(basket_sum["MULTI_ITEM_PCT"].iloc[0]) if not basket_sum.empty else 0
            single_count = 100 - multi_count
            fig_donut = go.Figure(go.Pie(
                labels=["Multi-item (3+)", "1–2 items"],
                values=[multi_count, single_count],
                hole=0.55,
                marker_colors=[COLORS["success"], COLORS["warning"]],
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                textinfo="percent+label", textfont=dict(size=10)))
            fig_donut.update_layout(**cl(margin=dict(l=0, r=0, t=20, b=0)),
                                    height=240, showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
            kpi_card("Multi-item rate", f"{multi_pct:.1f}%", "transactions with 3+ items", COLORS["success"])

        with b2:
            section_header("Top Products Bought Alone — Cross-Sell Targets")
            top_alone_d = top_alone.rename(columns={
                "PRODUCT_NAME": "Product",
                "TIMES_BOUGHT_ALONE": "Solo transactions",
                "PCT_OF_SINGLE_ITEM_SALES": "% of single-item sales"})
            st.dataframe(top_alone_d, use_container_width=True, hide_index=True, height=240)

        info_card(
            "<b>What to do:</b> The products above are bought alone more than any others. "
            "Place a companion product display at the point of sale next to the top 3 SKUs. "
            "Brief cashiers on one suggested add-on per product. Even a 10% conversion rate "
            "on solo transactions generates measurable basket lift within a week.",
            COLORS["success"])

        with st.expander("Basket metrics by store"):
            store_display = by_store.rename(columns={
                "STORE_NAME": "Store", "TRANSACTIONS": "Transactions",
                "AVG_ITEMS": "Avg items", "AVG_BASKET_KES": "Avg basket (KSh)",
                "SINGLE_ITEM_PCT": "Single-item %"})
            st.dataframe(store_display, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇ Basket + peak data (CSV)",
            pd.concat([top_alone.assign(table="cross_sell"),
                       by_store.assign(table="by_store")]).to_csv(index=False),
            "basket_peak_data.csv", "text/csv")

    with tab3:
        section_header("Peak Revenue Heatmap — Day × Hour")
        st.caption("Cumulative revenue by day of week and hour. Darker = more revenue.")

        if not peak_df.empty:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            pivot = (peak_df.pivot_table(
                         index="HOUR_OF_DAY", columns="DAY_NAME",
                         values="REVENUE", aggfunc="sum")
                     .fillna(0)
                     .reindex(columns=[d for d in day_order if d in peak_df["DAY_NAME"].values]))

            hm_text = [
                [f"{v/1000:.0f}K" if v >= 1000 else (f"{int(v)}" if v > 0 else "")
                 for v in row]
                for row in pivot.values.tolist()
            ]
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[[0, "#FEFCE8"], [0.5, "#FCD34D"], [1, "#15803D"]],
                text=hm_text,
                texttemplate="%{text}",
                textfont=dict(size=8, color="#003467"),
                hovertemplate="Day: %{x}<br>Hour: %{y}:00<br>Revenue: KSh %{z:,.0f}<extra></extra>",
                showscale=True))
            fig_hm.update_layout(**cl(yaxis=dict(title="Hour of day", dtick=2,
                                                 tickfont=dict(size=10))),
                                 height=480)
            fig_hm.update_xaxes(title="")
            st.plotly_chart(fig_hm, use_container_width=True, config={"displayModeBar": False})

        # Auto-generate peak insight card
        if peak_row is not None:
            second_peak = peak_df.nlargest(2, "REVENUE").iloc[1]
            info_card(
                f"<b>Your peak revenue slot is {peak_label}</b> — {fmt_ksh(float(peak_row['REVENUE']))} cumulative revenue. "
                f"Second highest: {second_peak['DAY_NAME']} {int(second_peak['HOUR_OF_DAY']):02d}:00. "
                f"These are the windows where a staffing gap or shift handover costs the most. "
                f"Lock these slots — no shift changes, no training sessions, full coverage.",
                COLORS["purple"])

    with tab4:
        _KATANI_CODES = {"KW", "KP"}

        # ── Revenue split: Katani vs Syokimau ─────────────────────────────────
        section_header("Revenue by Location — Katani vs Syokimau")

        if not rev_by_store.empty:
            rev_by_store["LOCATION"] = rev_by_store["STORE_CODE"].apply(
                lambda c: "Katani" if c in _KATANI_CODES else "Syokimau")
            loc_rev = rev_by_store.groupby("LOCATION")[["REVENUE", "TRANSACTIONS"]].sum().reset_index()

            lc1, lc2 = st.columns([1, 2])
            with lc1:
                fig_donut = go.Figure(go.Pie(
                    labels=loc_rev["LOCATION"],
                    values=loc_rev["REVENUE"],
                    hole=0.55,
                    marker_colors=[COLORS["primary"], COLORS["success"]],
                    hovertemplate="%{label}<br>KSh %{value:,.0f} (%{percent})<extra></extra>",
                    textinfo="percent+label",
                    textfont=dict(size=11)))
                fig_donut.update_layout(**cl(margin=dict(l=0, r=0, t=20, b=0)),
                                        height=260, showlegend=False)
                st.plotly_chart(fig_donut, use_container_width=True,
                                config={"displayModeBar": False})

            with lc2:
                stores_sorted = rev_by_store.sort_values("REVENUE", ascending=True)
                bar_colors = [COLORS["success"] if c in _KATANI_CODES else COLORS["primary"]
                              for c in stores_sorted["STORE_CODE"]]
                fig_stores = go.Figure(go.Bar(
                    x=stores_sorted["REVENUE"],
                    y=stores_sorted["STORE_NAME"],
                    orientation="h",
                    marker_color=bar_colors,
                    hovertemplate="%{y}<br>KSh %{x:,.0f}<extra></extra>"))
                fig_stores.update_layout(**cl(yaxis=dict(tickfont=dict(size=10))),
                                         height=260, xaxis_title="Revenue (KSh)")
                fig_stores.update_xaxes(tickprefix="KSh ", tickformat=",.0f")
                st.plotly_chart(fig_stores, use_container_width=True,
                                config={"displayModeBar": False})

        # ── Top products driving revenue at each location ──────────────────────
        section_header("What's Driving Revenue — Top 15 Products by Location", margin_top=16)
        st.caption("These products account for the largest share of revenue at each location.")

        if not top_prods_loc.empty:
            kat_top = top_prods_loc[top_prods_loc["LOCATION"] == "Katani"].head(15)
            syo_top = top_prods_loc[top_prods_loc["LOCATION"] == "Syokimau"].head(15)

            tp1, tp2 = st.columns(2)
            for col_ctx, df_loc, loc_name, color in [
                (tp1, syo_top, "Syokimau", COLORS["primary"]),
                (tp2, kat_top, "Katani",   COLORS["success"])
            ]:
                with col_ctx:
                    st.markdown(
                        f'<div style="font-size:11px;font-weight:700;color:{color};'
                        f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">'
                        f'{loc_name}</div>', unsafe_allow_html=True)
                    if not df_loc.empty:
                        fig_tp = go.Figure(go.Bar(
                            x=df_loc["REVENUE"],
                            y=df_loc["PRODUCT_NAME"],
                            orientation="h",
                            marker_color=color,
                            hovertemplate="%{y}<br>KSh %{x:,.0f}<extra></extra>"))
                        fig_tp.update_layout(
                            **cl(yaxis=dict(autorange="reversed",
                                           tickfont=dict(size=9)),
                                 margin=dict(l=0, r=0, t=4, b=20)),
                            height=360, xaxis_title="")
                        fig_tp.update_xaxes(tickprefix="KSh ", tickformat=".2s")
                        st.plotly_chart(fig_tp, use_container_width=True,
                                        config={"displayModeBar": False})
                    else:
                        st.caption(f"No data for {loc_name} with current filter.")

        # ── Cross-location opportunity ─────────────────────────────────────────
        section_header("Cross-Location Opportunity", margin_top=16)
        st.caption(
            "Products with significant revenue gap between locations. "
            "Positive gap = Syokimau stronger → growth opportunity at Katani. "
            "Negative gap = Katani stronger → growth opportunity at Syokimau.")

        if not loc_opp.empty:
            _months = max((pd.Timestamp.now() - pd.Timestamp("2025-09-01")).days / 30.44, 1)

            opp_display = loc_opp.copy()
            opp_display["monthly_upside"] = (opp_display["GAP"].abs() / _months).round(0)
            opp_display["direction"] = opp_display["GAP"].apply(
                lambda g: "Grow at Katani" if g > 0 else "Grow at Syokimau")

            fig_opp = go.Figure(go.Bar(
                x=opp_display["GAP"],
                y=opp_display["PRODUCT_NAME"],
                orientation="h",
                marker_color=[COLORS["primary"] if g > 0 else COLORS["success"]
                              for g in opp_display["GAP"]],
                hovertemplate="%{y}<br>Gap: KSh %{x:,.0f}<extra></extra>"))
            fig_opp.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
            fig_opp.update_layout(
                **cl(yaxis=dict(autorange="reversed", tickfont=dict(size=9))),
                height=max(300, len(opp_display) * 22),
                xaxis_title="Revenue Gap (KSh) — positive = Syokimau leads")
            fig_opp.update_xaxes(tickprefix="KSh ", tickformat=",.0f")
            st.plotly_chart(fig_opp, use_container_width=True,
                            config={"displayModeBar": False})

            # Prescription cards — top 2 each direction
            syokimau_leads = opp_display[opp_display["GAP"] > 0].head(2)
            katani_leads   = opp_display[opp_display["GAP"] < 0].head(2)

            if not syokimau_leads.empty:
                for _, row in syokimau_leads.iterrows():
                    info_card(
                        f"<b>Grow at Katani → {row['PRODUCT_NAME']}</b> — "
                        f"Syokimau generates {fmt_ksh(row['SYOKIMAU_REVENUE'])} vs "
                        f"{fmt_ksh(row['KATANI_REVENUE'])} at Katani. "
                        f"Closing that gap at Katani's run rate adds an estimated "
                        f"<b>{fmt_ksh(row['monthly_upside'])}/month</b>. "
                        f"Ensure stock is available and visible at Katani.",
                        COLORS["primary"])

            if not katani_leads.empty:
                for _, row in katani_leads.iterrows():
                    info_card(
                        f"<b>Grow at Syokimau → {row['PRODUCT_NAME']}</b> — "
                        f"Katani generates {fmt_ksh(row['KATANI_REVENUE'])} vs "
                        f"{fmt_ksh(row['SYOKIMAU_REVENUE'])} at Syokimau. "
                        f"Closing that gap at Syokimau's run rate adds an estimated "
                        f"<b>{fmt_ksh(row['monthly_upside'])}/month</b>. "
                        f"Review shelf placement and stock levels at Syokimau.",
                        COLORS["success"])

            with st.expander("Full opportunity table"):
                st.dataframe(
                    opp_display[["PRODUCT_NAME", "PRODUCT_CATEGORY",
                                 "SYOKIMAU_REVENUE", "KATANI_REVENUE",
                                 "GAP", "monthly_upside", "direction"]]
                    .rename(columns={
                        "PRODUCT_NAME": "Product", "PRODUCT_CATEGORY": "Category",
                        "SYOKIMAU_REVENUE": "Syokimau (KSh)", "KATANI_REVENUE": "Katani (KSh)",
                        "GAP": "Gap (KSh)", "monthly_upside": "Est. Monthly Upside (KSh)",
                        "direction": "Action"}),
                    use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    section_header("Executive Recommendation", margin_top=8)
    info_card(
        f"At current trajectory, projected 30-day revenue is {fmt_ksh(proj_30d)}. "
        f"{single_pct:.0f}% of transactions are single-item — this is the highest-ROI "
        f"improvement available without additional stock or headcount. "
        f"A cross-sell prompt at checkout on the top 5 solo products costs nothing to implement "
        f"and can move basket size materially within 2 weeks.",
        COLORS["primary"])


# ══════════════════════════════════════════════════════════════════════════════
# MARGIN INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

if page == "Margin Intelligence":

    expected_ma = {label for label, _ in ma.get_analyses()}
    if (not st.session_state.ma_data
            or st.session_state.ma_data.get("_filter_key") != filter_key
            or not expected_ma.issubset(st.session_state.ma_data.keys())):
        with st.spinner("Loading…"):
            st.session_state.ma_data = {"_filter_key": filter_key}
            for label, sql in ma.get_analyses(effective_stores):
                st.session_state.ma_data[label] = ma.run_query(sql, st.session_state.conn)

    M        = st.session_state.ma_data
    overall  = M["MVaR — Overall"].copy()
    by_store_m = M["MVaR — By Store"].copy()
    distrib  = M["MVaR — Distribution"].copy()

    for col in ["TOTAL_TRANSACTIONS", "AVG_MARGIN_PCT", "MVAR_5PCT", "MVAR_10PCT",
                "MEDIAN_MARGIN_PCT", "P25_MARGIN_PCT", "P75_MARGIN_PCT",
                "MIN_MARGIN_PCT", "MAX_MARGIN_PCT",
                "LOSS_MAKING_TRANSACTIONS", "LOSS_MAKING_PCT"]:
        overall[col] = pd.to_numeric(overall[col], errors="coerce")
    for col in ["TRANSACTIONS", "AVG_MARGIN_PCT", "MVAR_5PCT", "MVAR_10PCT",
                "MEDIAN_MARGIN_PCT", "LOSS_MAKING_COUNT"]:
        by_store_m[col] = pd.to_numeric(by_store_m[col], errors="coerce")
    for col in ["BUCKET_START", "TRANSACTION_COUNT"]:
        distrib[col] = pd.to_numeric(distrib[col], errors="coerce")

    ov_row      = overall.iloc[0]
    avg_m       = float(ov_row["AVG_MARGIN_PCT"])
    mvar5       = float(ov_row["MVAR_5PCT"])
    mvar10      = float(ov_row["MVAR_10PCT"])
    loss_pct    = float(ov_row["LOSS_MAKING_PCT"])
    loss_count  = int(ov_row["LOSS_MAKING_TRANSACTIONS"])

    st.markdown(
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin-bottom:4px">Xanalife · Margin Intelligence</p>',
        unsafe_allow_html=True)
    st.caption("Margin Value-at-Risk (MVaR) · Sep 2025 – present · 98.6% cost coverage")
    info_card(
        "<b>What is MVaR?</b> It measures your margin <i>floor</i> — not the average, but the worst. "
        "MVaR(5%) = in the bottom 5% of transactions, margin falls to this level or below. "
        "A low floor means discounts, pricing errors, or specific products are destroying value at the transaction level. "
        "The average looks healthy; MVaR shows you what's hiding underneath it.",
        COLORS["muted"])
    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Avg Transaction Margin", f"{avg_m:.1f}%",
                 f"Median: {float(ov_row['MEDIAN_MARGIN_PCT']):.1f}%", COLORS["primary"])
    with c2:
        kpi_card("MVaR (5%)", f"{mvar5:.1f}%",
                 "Floor margin in your worst 5% of transactions",
                 COLORS["danger"] if mvar5 < 0 else COLORS["warning"])
    with c3:
        kpi_card("MVaR (10%)", f"{mvar10:.1f}%",
                 "Floor margin in your worst 10% of transactions",
                 COLORS["warning"])
    with c4:
        kpi_card("Loss-Making Transactions", f"{loss_count:,}",
                 f"{loss_pct:.1f}% of all transactions — actively sold below cost",
                 COLORS["danger"] if loss_count > 0 else COLORS["success"])

    st.markdown("<div style='margin-bottom:16px'></div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["◉  Margin Distribution", "△  By Store"])

    with tab1:
        section_header("Transaction Margin Distribution")
        st.caption("Each bar = a 5% margin bucket. Percentile lines show MVaR thresholds.")

        if not distrib.empty:
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=distrib["BUCKET_START"],
                y=distrib["TRANSACTION_COUNT"],
                name="Transactions",
                marker_color=[COLORS["danger"] if b < 0 else COLORS["primary"]
                              for b in distrib["BUCKET_START"]],
                hovertemplate="Margin %{x}%–%{x:.0f}+5%<br>%{y:,} transactions<extra></extra>"))
            for val, label_v, color_v in [
                (mvar5,  "MVaR 5%",  COLORS["danger"]),
                (mvar10, "MVaR 10%", COLORS["warning"]),
                (avg_m,  "Avg",      COLORS["success"])]:
                fig_d.add_vline(x=val, line_dash="dash", line_color=color_v, line_width=2,
                                annotation_text=f"{label_v}: {val:.1f}%",
                                annotation_position="top",
                                annotation_font_size=10,
                                annotation_font_color=color_v)
            fig_d.update_layout(**CHART_LAYOUT, height=360,
                                xaxis_title="Margin %", yaxis_title="Transaction Count")
            fig_d.update_xaxes(ticksuffix="%")
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

        st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
        section_header("Margin Percentile Summary", margin_top=8)
        pct_df = pd.DataFrame({
            "Percentile": ["5th (MVaR)", "10th (MVaR)", "25th", "50th (Median)", "75th", "Avg"],
            "Margin %":   [mvar5, mvar10,
                           float(ov_row["P25_MARGIN_PCT"]),
                           float(ov_row["MEDIAN_MARGIN_PCT"]),
                           float(ov_row["P75_MARGIN_PCT"]),
                           avg_m]
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True)

    with tab2:
        section_header("MVaR by Store")

        if not by_store_m.empty:
            stores_sorted = by_store_m.sort_values("AVG_MARGIN_PCT", ascending=False)

            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(
                name="Avg Margin",
                x=stores_sorted["STORE_NAME"],
                y=stores_sorted["AVG_MARGIN_PCT"],
                marker_color=COLORS["primary"],
                hovertemplate="%{x}<br>Avg margin: %{y:.1f}%<extra></extra>"))
            fig_s.add_trace(go.Bar(
                name="MVaR 5%",
                x=stores_sorted["STORE_NAME"],
                y=stores_sorted["MVAR_5PCT"],
                marker_color=COLORS["danger"],
                hovertemplate="%{x}<br>MVaR 5%%: %{y:.1f}%<extra></extra>"))
            fig_s.add_trace(go.Bar(
                name="MVaR 10%",
                x=stores_sorted["STORE_NAME"],
                y=stores_sorted["MVAR_10PCT"],
                marker_color=COLORS["warning"],
                hovertemplate="%{x}<br>MVaR 10%%: %{y:.1f}%<extra></extra>"))
            fig_s.add_hline(y=0, line_dash="solid", line_color=COLORS["muted"], line_width=1)
            fig_s.update_layout(**CHART_LAYOUT, height=340, barmode="group",
                                legend=dict(orientation="h", y=1.08, font=dict(size=10)))
            fig_s.update_yaxes(ticksuffix="%", title="Margin %")
            st.plotly_chart(fig_s, use_container_width=True, config={"displayModeBar": False})

            with st.expander("Full MVaR by store table"):
                st.dataframe(
                    by_store_m.rename(columns={
                        "STORE_NAME": "Store", "TRANSACTIONS": "Transactions",
                        "AVG_MARGIN_PCT": "Avg Margin %", "MVAR_5PCT": "MVaR 5%",
                        "MVAR_10PCT": "MVaR 10%", "MEDIAN_MARGIN_PCT": "Median %",
                        "LOSS_MAKING_COUNT": "Loss-Making Txns"}),
                    use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-bottom:8px'></div>", unsafe_allow_html=True)
    section_header("Executive Recommendation", margin_top=8)

    if mvar5 < 0:
        info_card(
            f"<b>MVaR(5%) is {mvar5:.1f}% — negative.</b> This means in your worst 5% of transactions, "
            f"you are selling below cost. {loss_count:,} transactions ({loss_pct:.1f}%) are actively loss-making. "
            f"This is not a rounding issue — it is a pricing or cost-data problem. "
            f"Pull the lowest-margin transactions, identify the product lines, and review whether "
            f"unit costs are correctly set or whether those products are being discounted past their floor.",
            COLORS["danger"])
    else:
        worst_store = by_store_m.loc[by_store_m["MVAR_5PCT"].idxmin()] if not by_store_m.empty else None
        info_card(
            f"MVaR(5%) is {mvar5:.1f}% — on your worst 5% of transaction days, effective margin holds above zero. "
            f"Average margin is {avg_m:.1f}%. "
            + (f"Store '{worst_store['STORE_NAME']}' has the lowest floor margin ({worst_store['MVAR_5PCT']:.1f}%) — "
               f"review pricing on its lowest-margin product lines." if worst_store is not None else "")
            + f" Use MVaR as a benchmark when evaluating any new discount or promotion — "
            f"if it pushes floor margin below {mvar5:.1f}%, the promotion is destroying value on your worst days.",
            COLORS["primary"])

    st.download_button("⬇ Download margin data (CSV)",
                       pd.concat([overall.assign(table="overall"),
                                  by_store_m.assign(table="by_store")]).to_csv(index=False),
                       "margin_intelligence.csv", "text/csv")
