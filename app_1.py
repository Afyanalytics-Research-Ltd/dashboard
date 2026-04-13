import streamlit as st
import os
import pandas as pd
import numpy as np
from snowflake.snowpark import Session
from utils import (
    load_events_table,
    parse_payload_column,
    get_module_dataframe,
    get_module_tables,
    get_source_table,
    normalize_all_payloads,
)

st.set_page_config(page_title="Revenue Intelligence", layout="wide", page_icon="📊")

st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 16px;
    border-left: 4px solid #7c3aed;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a78bfa;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Revenue Intelligence Dashboard")
st.caption("15 deep-dive analyses built from Finance module data")

connection_parameters = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT", "").strip(),
    "user": os.getenv("SNOWFLAKE_USER", "").strip(),
    "password": os.getenv("SNOWFLAKE_PASSWORD", "").strip(),
    "role": os.getenv("SNOWFLAKE_ROLE", "").strip(),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "").strip(),
    "database": os.getenv("SNOWFLAKE_DATABASE", "").strip(),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "").strip(),
}

missing = [k for k, v in connection_parameters.items() if not v]
if missing:
    st.error(f"Missing configuration: {', '.join(missing)}")
    st.stop()


@st.cache_resource
def get_session():
    return Session.builder.configs(connection_parameters).create()


@st.cache_data
def load_all_finance_data():
    session = get_session()
    raw_df = load_events_table(session, "HOSPITALS.AFYA_API_AUTH_RAW.EVENTS_RAW")
    finance_df = get_module_dataframe(raw_df, "Finance")
    tables = get_module_tables(finance_df)

    loaded = {}
    for table in tables:
        try:
            src = get_source_table(raw_df, "Finance", table)
            src = parse_payload_column(src)
            normalized = normalize_all_payloads(src)
            if not normalized.empty:
                loaded[table] = normalized
        except Exception:
            pass
    return loaded, list(tables)


def to_numeric_safe(series):
    return pd.to_numeric(series, errors="coerce")


def to_datetime_safe(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


with st.spinner("Connecting to Snowflake and loading Finance data…"):
    try:
        finance_tables, table_names = load_all_finance_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

st.success(f"Loaded {len(finance_tables)} Finance tables: {', '.join(finance_tables.keys())}")

invoices = finance_tables.get("finance_invoices", pd.DataFrame()).copy()
waivers = finance_tables.get("finance_waivers", pd.DataFrame()).copy()

if invoices.empty:
    st.error("finance_invoices table not found or empty.")
    st.stop()

for col in ["balance", "amount", "paid", "company_id", "total_amount", "amount_paid"]:
    if col in invoices.columns:
        invoices[col] = to_numeric_safe(invoices[col])

for col in ["amount", "invoice_id"]:
    if col in waivers.columns:
        waivers[col] = to_numeric_safe(waivers[col])

for col in ["created_at", "due_date", "updated_at", "paid_at"]:
    if col in invoices.columns:
        invoices[col] = to_datetime_safe(invoices[col])

if "company_id" not in invoices.columns and "id" in invoices.columns:
    invoices["company_id"] = invoices["id"]

amount_col = next((c for c in ["amount", "total_amount"] if c in invoices.columns), None)
paid_col = next((c for c in ["paid", "amount_paid", "paid_amount"] if c in invoices.columns), None)
balance_col = "balance" if "balance" in invoices.columns else None
id_col = "id" if "id" in invoices.columns else invoices.columns[0]
date_col = "created_at" if "created_at" in invoices.columns else None
due_col = "due_date" if "due_date" in invoices.columns else None

merged = invoices.copy()
if not waivers.empty and "invoice_id" in waivers.columns and "amount" in waivers.columns:
    waiver_agg = waivers.groupby("invoice_id")["amount"].sum().reset_index().rename(
        columns={"amount": "waiver_amount", "invoice_id": id_col}
    )
    waiver_agg[id_col] = to_numeric_safe(waiver_agg[id_col])
    if id_col in merged.columns:
        merged[id_col] = to_numeric_safe(merged[id_col])
    merged = merged.merge(waiver_agg, on=id_col, how="left")
    merged["waiver_amount"] = merged["waiver_amount"].fillna(0)
else:
    merged["waiver_amount"] = 0


tabs = st.tabs([
    "① Leakage",
    "② Aging",
    "③ DSO",
    "④ Waterfall",
    "⑤ Pareto",
    "⑥ Waiver vs Volume",
    "⑦ MoM Momentum",
    "⑧ Collection Efficiency",
    "⑨ Revenue at Risk",
    "⑩ Payment Clustering",
    "⑪ Waiver Trend",
    "⑫ Invoice Distribution",
    "⑬ Bad Debt Forecast",
    "⑭ Top vs Delinquent",
    "⑮ Rolling 90-Day",
])


# ─── 1. Revenue Leakage Analysis ───────────────────────────────────────────────
with tabs[0]:
    st.subheader("① Revenue Leakage Analysis")
    st.markdown(
        "Measures how much gross-billed revenue silently bleeds out through waivers before it ever reaches collection. "
        "Most finance teams track waivers — few track *leakage rate* per company."
    )
    if amount_col and merged["waiver_amount"].sum() > 0:
        leakage = (
            merged.groupby("company_id")
            .agg(
                gross_billed=(amount_col, "sum"),
                total_waivers=("waiver_amount", "sum"),
            )
            .dropna()
        )
        leakage["leakage_rate_pct"] = (leakage["total_waivers"] / leakage["gross_billed"].replace(0, np.nan) * 100).round(2)
        leakage["net_billable"] = leakage["gross_billed"] - leakage["total_waivers"]
        leakage = leakage.sort_values("leakage_rate_pct", ascending=False).reset_index()

        col1, col2, col3 = st.columns(3)
        total_gross = leakage["gross_billed"].sum()
        total_waived = leakage["total_waivers"].sum()
        col1.metric("Total Gross Billed", f"{total_gross:,.0f}")
        col2.metric("Total Waived", f"{total_waived:,.0f}")
        col3.metric("Avg Leakage Rate", f"{(total_waived/total_gross*100) if total_gross else 0:.1f}%")

        st.markdown("**Leakage Rate by Company** — sorted by worst offenders")
        st.dataframe(leakage.style.background_gradient(subset=["leakage_rate_pct"], cmap="Reds"), width="stretch")
    else:
        st.info("Waiver data not available or amounts are zero.")


# ─── 2. Invoice Aging Pyramid ──────────────────────────────────────────────────
with tabs[1]:
    st.subheader("② Invoice Aging Pyramid")
    st.markdown(
        "Buckets unpaid invoices by how overdue they are. The shape of this pyramid predicts future bad debt. "
        "A heavy '90+ days' bucket is a silent write-off building up."
    )
    if due_col and balance_col:
        now = pd.Timestamp.now(tz="UTC")
        aging = merged[merged[balance_col] > 0].copy()
        aging["days_overdue"] = (now - aging[due_col]).dt.days
        aging["aging_bucket"] = pd.cut(
            aging["days_overdue"],
            bins=[-np.inf, 0, 30, 60, 90, np.inf],
            labels=["Not Due", "0–30 days", "31–60 days", "61–90 days", "90+ days"],
        )
        bucket_summary = (
            aging.groupby("aging_bucket", observed=True)
            .agg(invoice_count=("id", "count") if "id" in aging.columns else (balance_col, "count"),
                 total_outstanding=(balance_col, "sum"))
            .reset_index()
        )
        bucket_summary["% of total"] = (bucket_summary["total_outstanding"] / bucket_summary["total_outstanding"].sum() * 100).round(1)

        col1, col2 = st.columns(2)
        col1.dataframe(bucket_summary, width="stretch")
        over90 = bucket_summary[bucket_summary["aging_bucket"] == "90+ days"]["total_outstanding"].sum()
        col2.metric("Outstanding 90+ Days", f"{over90:,.0f}", delta="High Risk", delta_color="inverse")
    elif date_col and balance_col:
        now = pd.Timestamp.now(tz="UTC")
        aging = merged[merged[balance_col] > 0].copy()
        aging["days_since_invoice"] = (now - aging[date_col]).dt.days
        aging["aging_bucket"] = pd.cut(
            aging["days_since_invoice"],
            bins=[-np.inf, 30, 60, 90, np.inf],
            labels=["< 30 days", "31–60 days", "61–90 days", "90+ days"],
        )
        bucket_summary = (
            aging.groupby("aging_bucket", observed=True)
            .agg(total_outstanding=(balance_col, "sum"))
            .reset_index()
        )
        st.dataframe(bucket_summary, width="stretch")
    else:
        st.info("Date and balance columns needed for aging analysis.")


# ─── 3. Days Sales Outstanding (DSO) ──────────────────────────────────────────
with tabs[2]:
    st.subheader("③ Days Sales Outstanding (DSO)")
    st.markdown(
        "DSO is the average number of days between invoicing and payment. Low DSO = fast cash collection. "
        "High DSO per company = a credit risk or friction in their billing workflow."
    )
    paid_at_col = "paid_at" if "paid_at" in invoices.columns else None
    if date_col and (paid_at_col or paid_col):
        if paid_at_col:
            dso_df = merged.dropna(subset=[date_col, paid_at_col]).copy()
            dso_df["days_to_pay"] = (dso_df[paid_at_col] - dso_df[date_col]).dt.days
        elif amount_col and paid_col:
            dso_df = merged.dropna(subset=[date_col, amount_col, paid_col]).copy()
            dso_df = dso_df[dso_df[amount_col] > 0]
            dso_df["days_to_pay"] = (dso_df[paid_col] / dso_df[amount_col] * 30).round(0)

        dso_by_company = (
            dso_df.groupby("company_id")["days_to_pay"]
            .agg(["mean", "median", "max", "count"])
            .rename(columns={"mean": "avg_days", "median": "median_days", "max": "worst_case", "count": "invoices"})
            .round(1)
            .sort_values("avg_days", ascending=False)
            .reset_index()
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Avg DSO", f"{dso_df['days_to_pay'].mean():.1f} days")
        col2.metric("Best Company DSO", f"{dso_by_company['avg_days'].min():.1f} days")
        col3.metric("Worst Company DSO", f"{dso_by_company['avg_days'].max():.1f} days")
        st.dataframe(dso_by_company.style.background_gradient(subset=["avg_days"], cmap="RdYlGn_r"), width="stretch")
    else:
        st.info("Payment date or paid amount columns not found. Cannot calculate DSO.")


# ─── 4. Net Revenue Waterfall ──────────────────────────────────────────────────
with tabs[3]:
    st.subheader("④ Net Revenue Waterfall")
    st.markdown(
        "Traces revenue from gross invoiced → waived → collected → outstanding. "
        "Most dashboards show only the end number. This shows where it *disappears*."
    )
    if amount_col:
        gross = merged[amount_col].sum()
        waived = merged["waiver_amount"].sum()
        collected = merged[paid_col].sum() if paid_col else 0
        outstanding = merged[balance_col].sum() if balance_col else gross - waived - collected
        unaccounted = gross - waived - collected - outstanding

        waterfall = pd.DataFrame({
            "Stage": ["Gross Invoiced", "Less: Waivers", "Less: Collected", "Less: Outstanding", "Unaccounted / Error"],
            "Amount": [gross, -waived, -collected, -outstanding, unaccounted],
            "Cumulative": [gross, gross - waived, gross - waived - collected, gross - waived - collected - outstanding, 0],
        })

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Gross Invoiced", f"{gross:,.0f}")
        col2.metric("Waivers", f"-{waived:,.0f}", delta=f"{waived/gross*100:.1f}% of gross" if gross else "0%", delta_color="inverse")
        col3.metric("Collected", f"{collected:,.0f}", delta=f"{collected/gross*100:.1f}% collection rate" if gross else "0%")
        col4.metric("Outstanding", f"{outstanding:,.0f}", delta="Uncollected", delta_color="inverse")

        st.dataframe(waterfall, width="stretch")
    else:
        st.info("Amount column not found.")


# ─── 5. Pareto (Revenue Concentration) ────────────────────────────────────────
with tabs[4]:
    st.subheader("⑤ Revenue Concentration — Pareto Analysis")
    st.markdown(
        "Which 20% of companies generate 80% of your revenue? "
        "Concentration risk here means losing one client could crater total revenue."
    )
    if amount_col and "company_id" in merged.columns:
        pareto = (
            merged.groupby("company_id")[amount_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={amount_col: "revenue"})
        )
        pareto["cumulative_revenue"] = pareto["revenue"].cumsum()
        pareto["cumulative_pct"] = pareto["cumulative_revenue"] / pareto["revenue"].sum() * 100
        pareto["revenue_share_pct"] = pareto["revenue"] / pareto["revenue"].sum() * 100
        pareto["company_pct"] = (pareto.index + 1) / len(pareto) * 100

        top20_companies = pareto[pareto["company_pct"] <= 20]
        top20_revenue_share = top20_companies["revenue"].sum() / pareto["revenue"].sum() * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Companies", len(pareto))
        col2.metric("Top 20% Companies Revenue Share", f"{top20_revenue_share:.1f}%")
        col3.metric("Top Company Revenue", f"{pareto['revenue'].iloc[0]:,.0f}")

        st.markdown("**Revenue share per company (sorted)**")
        st.dataframe(pareto[["company_id", "revenue", "revenue_share_pct", "cumulative_pct"]].round(2), width="stretch")
    else:
        st.info("Amount and company_id columns required.")


# ─── 6. Waiver Rate vs Invoice Volume ─────────────────────────────────────────
with tabs[5]:
    st.subheader("⑥ Waiver Rate vs Invoice Volume")
    st.markdown(
        "Companies with high invoice volume AND high waiver rates are double-draining revenue. "
        "This catches deals that look busy but generate no real revenue."
    )
    if amount_col and merged["waiver_amount"].sum() > 0:
        combo = (
            merged.groupby("company_id")
            .agg(
                total_invoiced=(amount_col, "sum"),
                total_waived=("waiver_amount", "sum"),
                invoice_count=(amount_col, "count"),
            )
            .reset_index()
        )
        combo["waiver_rate_pct"] = (combo["total_waived"] / combo["total_invoiced"].replace(0, np.nan) * 100).round(2)
        combo["revenue_per_invoice"] = (combo["total_invoiced"] / combo["invoice_count"]).round(2)
        combo["net_revenue"] = combo["total_invoiced"] - combo["total_waived"]
        combo = combo.sort_values("waiver_rate_pct", ascending=False).reset_index(drop=True)

        st.dataframe(combo.style.background_gradient(subset=["waiver_rate_pct"], cmap="YlOrRd"), width="stretch")
    else:
        st.info("Waiver data not available.")


# ─── 7. Month-over-Month Revenue Momentum ─────────────────────────────────────
with tabs[6]:
    st.subheader("⑦ Month-over-Month Revenue Momentum")
    st.markdown(
        "Revenue momentum is not just growth — it's the *rate of change* of growth. "
        "Decelerating MoM growth is a leading indicator of future decline before it shows up in totals."
    )
    if date_col and amount_col:
        mom = merged.dropna(subset=[date_col, amount_col]).copy()
        mom["month"] = mom[date_col].dt.to_period("M")
        monthly = (
            mom.groupby("month")[amount_col]
            .sum()
            .reset_index()
            .rename(columns={amount_col: "revenue"})
            .sort_values("month")
        )
        monthly["prev_revenue"] = monthly["revenue"].shift(1)
        monthly["mom_growth_pct"] = ((monthly["revenue"] - monthly["prev_revenue"]) / monthly["prev_revenue"].replace(0, np.nan) * 100).round(2)
        monthly["acceleration"] = monthly["mom_growth_pct"].diff().round(2)
        monthly["signal"] = monthly["acceleration"].apply(
            lambda x: "🚀 Accelerating" if x > 0 else ("⚠️ Decelerating" if x < 0 else "— Stable") if pd.notna(x) else "—"
        )
        monthly["month"] = monthly["month"].astype(str)

        col1, col2, col3 = st.columns(3)
        last = monthly.dropna(subset=["mom_growth_pct"]).iloc[-1] if len(monthly) > 1 else None
        if last is not None:
            col1.metric("Latest Month Revenue", f"{last['revenue']:,.0f}")
            col2.metric("MoM Growth", f"{last['mom_growth_pct']:.1f}%")
            col3.metric("Momentum Signal", last["signal"])

        st.dataframe(monthly, width="stretch")
    else:
        st.info("Date and amount columns required.")


# ─── 8. Collection Efficiency Score ───────────────────────────────────────────
with tabs[7]:
    st.subheader("⑧ Collection Efficiency Score")
    st.markdown(
        "The ratio of what was actually collected vs what was invoiced. "
        "A company scoring < 70% is funding operations through uncollected receivables."
    )
    if amount_col and paid_col:
        eff = merged.groupby("company_id").agg(
            invoiced=(amount_col, "sum"),
            collected=(paid_col, "sum"),
        ).reset_index()
        eff["efficiency_score"] = (eff["collected"] / eff["invoiced"].replace(0, np.nan) * 100).round(1)
        eff["grade"] = eff["efficiency_score"].apply(
            lambda x: "A (≥90%)" if x >= 90 else "B (75–89%)" if x >= 75 else "C (60–74%)" if x >= 60 else "D (<60%)" if pd.notna(x) else "N/A"
        )
        eff = eff.sort_values("efficiency_score", ascending=False).reset_index(drop=True)

        grade_counts = eff["grade"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall Efficiency", f"{(eff['collected'].sum()/eff['invoiced'].sum()*100):.1f}%")
        col2.metric("Companies Graded A", str(grade_counts.get("A (≥90%)", 0)))
        col3.metric("Companies Graded D", str(grade_counts.get("D (<60%)", 0)))

        st.dataframe(eff.style.background_gradient(subset=["efficiency_score"], cmap="RdYlGn"), width="stretch")
    else:
        st.info("Paid and amount columns required.")


# ─── 9. Revenue at Risk ────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("⑨ Revenue at Risk")
    st.markdown(
        "Combines outstanding balance with aging to score which balances are most at risk of never being collected. "
        "Higher age × higher balance = highest danger."
    )
    if balance_col and date_col:
        now = pd.Timestamp.now(tz="UTC")
        rar = merged[merged[balance_col] > 0].copy()
        rar["age_days"] = (now - rar[date_col]).dt.days.clip(lower=0)
        rar["risk_score"] = (rar[balance_col] * np.log1p(rar["age_days"])).round(2)
        rar["risk_tier"] = pd.qcut(rar["risk_score"], q=4, labels=["Low", "Medium", "High", "Critical"], duplicates="drop")

        risk_by_company = (
            rar.groupby("company_id")
            .agg(
                total_at_risk=(balance_col, "sum"),
                avg_age_days=("age_days", "mean"),
                avg_risk_score=("risk_score", "mean"),
                invoice_count=(balance_col, "count"),
            )
            .round(1)
            .sort_values("avg_risk_score", ascending=False)
            .reset_index()
        )

        col1, col2 = st.columns(2)
        col1.metric("Total Revenue at Risk", f"{rar[balance_col].sum():,.0f}")
        col2.metric("Critical Risk Invoices", str((rar["risk_tier"] == "Critical").sum()))

        st.dataframe(risk_by_company.style.background_gradient(subset=["avg_risk_score"], cmap="Reds"), width="stretch")
    else:
        st.info("Balance and date columns required.")


# ─── 10. Payment Behavior Clustering ──────────────────────────────────────────
with tabs[9]:
    st.subheader("⑩ Payment Behavior Clustering")
    st.markdown(
        "Groups companies into behavioral segments based on how they pay: Champions, Reliable, Slow, and At-Risk. "
        "Allows targeted credit and collection strategies per segment."
    )
    if amount_col and paid_col and balance_col:
        cluster_df = merged.groupby("company_id").agg(
            total_invoiced=(amount_col, "sum"),
            total_paid=(paid_col, "sum"),
            total_outstanding=(balance_col, "sum"),
            invoice_count=(amount_col, "count"),
        ).reset_index()

        cluster_df["pay_rate"] = (cluster_df["total_paid"] / cluster_df["total_invoiced"].replace(0, np.nan)).round(3)
        cluster_df["outstanding_rate"] = (cluster_df["total_outstanding"] / cluster_df["total_invoiced"].replace(0, np.nan)).round(3)
        cluster_df["avg_invoice_value"] = (cluster_df["total_invoiced"] / cluster_df["invoice_count"]).round(2)

        def assign_segment(row):
            if row["pay_rate"] >= 0.90 and row["outstanding_rate"] <= 0.10:
                return "🏆 Champion"
            elif row["pay_rate"] >= 0.75:
                return "✅ Reliable"
            elif row["pay_rate"] >= 0.50:
                return "🐢 Slow Payer"
            else:
                return "🚨 At-Risk"

        cluster_df["segment"] = cluster_df.apply(assign_segment, axis=1)

        segment_summary = cluster_df.groupby("segment").agg(
            companies=("company_id", "count"),
            total_invoiced=("total_invoiced", "sum"),
            total_paid=("total_paid", "sum"),
        ).reset_index()

        col1, col2 = st.columns(2)
        col1.markdown("**Segment Distribution**")
        col1.dataframe(segment_summary, width="stretch")
        col2.markdown("**Company-level Clustering**")
        col2.dataframe(cluster_df.sort_values("pay_rate", ascending=False).reset_index(drop=True), width="stretch")
    else:
        st.info("Amount, paid, and balance columns required.")


# ─── 11. Waiver Trend Over Time ───────────────────────────────────────────────
with tabs[10]:
    st.subheader("⑪ Waiver Trend Over Time")
    st.markdown(
        "Tracks how generously or aggressively waivers are being granted month by month. "
        "Rising waiver amounts signal policy drift or collection weakness being masked by write-offs."
    )
    if date_col and merged["waiver_amount"].sum() > 0:
        wt = merged.dropna(subset=[date_col]).copy()
        wt["month"] = wt[date_col].dt.to_period("M").astype(str)
        waiver_trend = (
            wt.groupby("month")
            .agg(
                total_waived=("waiver_amount", "sum"),
                total_invoiced=(amount_col, "sum") if amount_col else ("waiver_amount", "count"),
            )
            .reset_index()
        )
        if amount_col:
            waiver_trend["waiver_rate_pct"] = (waiver_trend["total_waived"] / waiver_trend["total_invoiced"].replace(0, np.nan) * 100).round(2)
        waiver_trend["mom_waiver_change"] = waiver_trend["total_waived"].pct_change().mul(100).round(2)

        st.dataframe(waiver_trend, width="stretch")
        st.metric("Peak Waiver Month", str(waiver_trend.loc[waiver_trend["total_waived"].idxmax(), "month"]))
    else:
        st.info("Waiver data or date column not available.")


# ─── 12. Invoice Value Distribution ───────────────────────────────────────────
with tabs[11]:
    st.subheader("⑫ Invoice Value Distribution")
    st.markdown(
        "Shows the statistical spread of invoice amounts. Outlier invoices (extreme values) "
        "disproportionately affect averages and can hide the true 'typical' invoice behaviour."
    )
    if amount_col:
        desc = merged[amount_col].dropna().describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        desc_df = desc.reset_index().rename(columns={"index": "Statistic", amount_col: "Value"})
        desc_df["Value"] = desc_df["Value"].round(2)

        q1 = merged[amount_col].quantile(0.25)
        q3 = merged[amount_col].quantile(0.75)
        iqr = q3 - q1
        outliers = merged[
            (merged[amount_col] < q1 - 1.5 * iqr) | (merged[amount_col] > q3 + 1.5 * iqr)
        ]

        col1, col2, col3 = st.columns(3)
        col1.metric("Median Invoice", f"{merged[amount_col].median():,.0f}")
        col2.metric("Mean Invoice", f"{merged[amount_col].mean():,.0f}")
        col3.metric("Outlier Invoices", str(len(outliers)))

        col1, col2 = st.columns(2)
        col1.markdown("**Statistical Summary**")
        col1.dataframe(desc_df, width="stretch")
        col2.markdown("**Outlier Invoices (IQR Method)**")
        col2.dataframe(outliers[[id_col, amount_col, "company_id"]].head(20) if "company_id" in outliers.columns else outliers[[id_col, amount_col]].head(20), width="stretch")
    else:
        st.info("Amount column not found.")


# ─── 13. Bad Debt Forecast ────────────────────────────────────────────────────
with tabs[12]:
    st.subheader("⑬ Bad Debt Forecast")
    st.markdown(
        "Scores every outstanding invoice by its likelihood of becoming bad debt, "
        "using age, balance, and payment history. Gives finance teams a prioritised collection list."
    )
    if balance_col and date_col:
        now = pd.Timestamp.now(tz="UTC")
        bad_debt = merged[merged[balance_col] > 0].copy()
        bad_debt["age_days"] = (now - bad_debt[date_col]).dt.days.clip(lower=0)

        pay_rate_map = {}
        if paid_col and amount_col:
            rates = merged.groupby("company_id").apply(
                lambda g: g[paid_col].sum() / g[amount_col].replace(0, np.nan).sum()
            ).fillna(0)
            pay_rate_map = rates.to_dict()

        if "company_id" in bad_debt.columns:
            bad_debt["company_pay_rate"] = bad_debt["company_id"].map(pay_rate_map).fillna(0.5)
        else:
            bad_debt["company_pay_rate"] = 0.5

        bad_debt["bad_debt_probability"] = (
            (bad_debt["age_days"] / 365).clip(0, 1) * 0.5
            + (1 - bad_debt["company_pay_rate"]) * 0.5
        ).round(3)
        bad_debt["expected_loss"] = (bad_debt[balance_col] * bad_debt["bad_debt_probability"]).round(2)

        output_cols = ["company_id", balance_col, "age_days", "company_pay_rate", "bad_debt_probability", "expected_loss"]
        output_cols = [c for c in output_cols if c in bad_debt.columns]

        total_expected_loss = bad_debt["expected_loss"].sum()
        high_risk = bad_debt[bad_debt["bad_debt_probability"] >= 0.7]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Outstanding", f"{bad_debt[balance_col].sum():,.0f}")
        col2.metric("Expected Bad Debt", f"{total_expected_loss:,.0f}")
        col3.metric("High-Risk Invoices (≥70%)", str(len(high_risk)))

        st.dataframe(
            bad_debt[output_cols].sort_values("bad_debt_probability", ascending=False).head(50)
            .style.background_gradient(subset=["bad_debt_probability"], cmap="Reds"),
            width="stretch"
        )
    else:
        st.info("Balance and date columns required.")


# ─── 14. Top Revenue vs Top Delinquency ───────────────────────────────────────
with tabs[13]:
    st.subheader("⑭ Top Revenue vs Top Delinquency")
    st.markdown(
        "The dangerous overlap: companies that generate the most revenue AND owe the most. "
        "Losing one of these clients to a collections dispute would cause a double hit."
    )
    if amount_col and balance_col and "company_id" in merged.columns:
        rev = merged.groupby("company_id")[amount_col].sum().reset_index().rename(columns={amount_col: "total_revenue"})
        owe = merged.groupby("company_id")[balance_col].sum().reset_index().rename(columns={balance_col: "total_outstanding"})
        combined = rev.merge(owe, on="company_id")
        combined["revenue_rank"] = combined["total_revenue"].rank(ascending=False).astype(int)
        combined["delinquency_rank"] = combined["total_outstanding"].rank(ascending=False).astype(int)
        combined["danger_score"] = (combined["total_outstanding"] / combined["total_revenue"].replace(0, np.nan) * 100).round(1)
        combined = combined.sort_values("danger_score", ascending=False).reset_index(drop=True)
        combined["profile"] = combined.apply(
            lambda r: "🔥 Top Revenue & Delinquent" if r["revenue_rank"] <= 5 and r["delinquency_rank"] <= 5
            else "⭐ Top Revenue" if r["revenue_rank"] <= 5
            else "⚠️ High Delinquency" if r["delinquency_rank"] <= 5
            else "Normal",
            axis=1
        )

        st.dataframe(combined.style.background_gradient(subset=["danger_score"], cmap="RdYlGn_r"), width="stretch")
    else:
        st.info("Amount, balance, and company_id columns required.")


# ─── 15. Rolling 90-Day Revenue Trend ─────────────────────────────────────────
with tabs[14]:
    st.subheader("⑮ Rolling 90-Day Revenue Trend")
    st.markdown(
        "A rolling 90-day window smooths out month-end spikes and invoice timing quirks to reveal "
        "the true underlying revenue trajectory. The gap between actual and rolling average signals volatility."
    )
    if date_col and amount_col:
        roll = merged.dropna(subset=[date_col, amount_col]).copy()
        roll["date"] = roll[date_col].dt.date
        daily = roll.groupby("date")[amount_col].sum().reset_index().sort_values("date")
        daily["rolling_90d"] = daily[amount_col].rolling(90, min_periods=1).mean().round(2)
        daily["rolling_30d"] = daily[amount_col].rolling(30, min_periods=1).mean().round(2)
        daily["volatility_gap"] = (daily[amount_col] - daily["rolling_90d"]).round(2)
        daily["trend"] = daily["rolling_90d"].diff().apply(
            lambda x: "↑ Up" if x > 0 else "↓ Down" if x < 0 else "→ Flat" if pd.notna(x) else "—"
        )
        daily.rename(columns={amount_col: "daily_revenue"}, inplace=True)

        col1, col2, col3 = st.columns(3)
        last_row = daily.iloc[-1]
        col1.metric("Latest Daily Revenue", f"{last_row['daily_revenue']:,.0f}")
        col2.metric("90-Day Rolling Avg", f"{last_row['rolling_90d']:,.0f}")
        col3.metric("Trend Direction", last_row["trend"])

        st.dataframe(daily.tail(90), width="stretch")
    else:
        st.info("Date and amount columns required.")


st.divider()
with st.expander("🔍 Raw Finance Tables Explorer"):
    selected_table = st.selectbox("Select a Finance table to inspect", options=list(finance_tables.keys()))
    if selected_table:
        tbl = finance_tables[selected_table]
        st.caption(f"{len(tbl):,} rows × {len(tbl.columns)} columns")
        st.dataframe(tbl.head(50), width="stretch")
