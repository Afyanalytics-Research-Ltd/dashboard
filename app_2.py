import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.signal import periodogram
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter

from data_loader import (
    load_raw_events, load_module_tables_map, load_table,
    num_cols, date_cols, cat_cols,
    best_amount_col, best_date_col, best_id_col, safe_show, ALL_MODULES,
)

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Hospital Intelligence Engine", layout="wide", page_icon="🏥")

ACCENT = "#7c3aed"
CMAP   = "plasma"

# ── helpers ──────────────────────────────────────────────────────────────────

def fig_to_st(fig, key=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

def no_data(msg="Not enough data for this analysis."):
    st.info(f"⚠️ {msg}")

def section(n, title, desc):
    st.markdown(f"<h4 style='color:#a78bfa;margin-top:1.2rem'>#{n} — {title}</h4>", unsafe_allow_html=True)
    st.caption(desc)

def gini(arr):
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) - (n + 1) * arr.sum()) / (n * arr.sum())

def shannon_entropy(series):
    p = series.value_counts(normalize=True)
    return float(-(p * np.log2(p + 1e-12)).sum())

def benfords_expected():
    return pd.Series({d: np.log10(1 + 1 / d) for d in range(1, 10)})

def first_digits(series):
    s = series.dropna()
    s = s[s > 0].astype(str).str.replace(r"^\d+\.", "", regex=True).str.lstrip("0").str[0]
    return pd.to_numeric(s, errors="coerce").dropna().astype(int)

def cv(series):
    m = series.mean()
    return series.std() / m if m != 0 else np.nan

# ── sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🏥 Hospital Intelligence")
st.sidebar.markdown("100 deep analyses across all modules")

with st.sidebar:
    with st.spinner("Mapping modules…"):
        try:
            table_map = load_module_tables_map()
        except Exception as e:
            st.error(str(e)); st.stop()

module_choice = st.sidebar.selectbox("Module", ["🔀 Cross-Module"] + ALL_MODULES)
st.sidebar.markdown("---")

def pick_table(module):
    tables = table_map.get(module, [])
    if not tables:
        return None, []
    selected = st.sidebar.selectbox(f"{module} table", tables)
    return selected, tables

# ══════════════════════════════════════════════════════════════════════════════
#  FINANCE  (analyses 1-12)
# ══════════════════════════════════════════════════════════════════════════════

if module_choice == "Finance":
    st.title("💰 Finance — 12 Hardcore Analyses")
    tables = table_map.get("Finance", [])

    @st.cache_data(show_spinner=False)
    def load_finance():
        dfs = {t: load_table("Finance", t) for t in tables}
        return dfs

    with st.spinner("Loading Finance tables…"):
        fdfs = load_finance()

    st.success(f"Tables: {', '.join(tables)}")

    inv = fdfs.get("finance_invoices", pd.DataFrame())
    wav = fdfs.get("finance_waivers",  pd.DataFrame())

    for c in ["amount", "balance", "paid", "company_id", "id"]:
        if c in inv.columns: inv[c] = pd.to_numeric(inv[c], errors="coerce")
    for c in ["amount", "invoice_id"]:
        if c in wav.columns: wav[c] = pd.to_numeric(wav[c], errors="coerce")

    if not inv.empty and "invoice_id" in wav.columns and "amount" in wav.columns:
        wagg = wav.groupby("invoice_id")["amount"].sum().reset_index()
        wagg.columns = ["id", "waiver_amount"]
        inv = inv.merge(wagg, on="id", how="left")
        inv["waiver_amount"] = inv["waiver_amount"].fillna(0)
    else:
        inv["waiver_amount"] = 0

    # ── 1. Benford's Law Fraud Detection ─────────────────────────────────────
    section(1, "Benford's Law Fraud Detection",
            "First-digit frequency test against expected logarithmic distribution. "
            "Significant deviation (p<0.05) flags potential fabricated invoices.")
    if "amount" in inv.columns:
        fd = first_digits(inv["amount"].dropna())
        obs = fd.value_counts(normalize=True).sort_index().reindex(range(1,10)).fillna(0)
        exp = benfords_expected()
        chi2, p = stats.chisquare(obs, exp)
        fig, ax = plt.subplots(figsize=(9,4))
        x = np.arange(1,10)
        ax.bar(x - 0.2, obs.values, 0.4, label="Observed", color=ACCENT, alpha=0.85)
        ax.bar(x + 0.2, exp.values, 0.4, label="Expected (Benford)", color="#f97316", alpha=0.85)
        ax.set_xticks(x); ax.set_xlabel("First Digit"); ax.set_ylabel("Frequency")
        ax.set_title(f"Benford's Law · χ²={chi2:.2f}  p={p:.4f}  {'🚨 ANOMALY' if p < 0.05 else '✅ NORMAL'}")
        ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    # ── 2. Revenue Lorenz Curve & Gini ────────────────────────────────────────
    section(2, "Revenue Lorenz Curve & Gini Coefficient",
            "Lorenz curve shows inequality of revenue across companies. "
            "Gini=0 is perfect equality; Gini=1 is one company holds all revenue.")
    if "amount" in inv.columns and "company_id" in inv.columns:
        rev = inv.groupby("company_id")["amount"].sum().sort_values().dropna()
        cum = rev.cumsum() / rev.sum()
        G = gini(rev.values)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(np.linspace(0,1,len(cum)), cum.values, color=ACCENT, lw=2, label=f"Lorenz (Gini={G:.3f})")
        ax.plot([0,1],[0,1], "k--", lw=1, label="Perfect Equality")
        ax.fill_between(np.linspace(0,1,len(cum)), cum.values, np.linspace(0,1,len(cum)), alpha=0.15, color=ACCENT)
        ax.set_xlabel("Cumulative % of Companies"); ax.set_ylabel("Cumulative % of Revenue")
        ax.set_title("Revenue Lorenz Curve"); ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    # ── 3. Invoice Cohort Net Dollar Retention ────────────────────────────────
    section(3, "Invoice Cohort Net Dollar Retention (NDR)",
            "NDR tracks how much of a cohort's original billed revenue survives (or expands) "
            "in subsequent months — the gold metric for sustainable revenue.")
    dc = best_date_col(inv)
    if dc and "company_id" in inv.columns and "amount" in inv.columns:
        df3 = inv.dropna(subset=[dc, "company_id", "amount"]).copy()
        df3["period"]  = df3[dc].dt.to_period("M")
        df3["cohort"]  = df3.groupby("company_id")["period"].transform("min")
        df3["offset"]  = (df3["period"] - df3["cohort"]).apply(lambda x: x.n)
        pivot = df3.pivot_table(index="cohort", columns="offset", values="amount",
                                aggfunc="sum", fill_value=0)
        pivot = pivot.div(pivot[0].replace(0, np.nan), axis=0) * 100
        fig, ax = plt.subplots(figsize=(12,5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=200)
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, fontsize=7)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels([str(c) for c in pivot.index], fontsize=7)
        ax.set_title("NDR % by Cohort × Months Since First Invoice")
        plt.colorbar(im, ax=ax, label="% of Month-0 Revenue"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    # ── 4. Payment Velocity Survival (Empirical CDF) ──────────────────────────
    section(4, "Payment Velocity — Empirical CDF of Collection Lag",
            "How many invoices are paid by day N? The steeper the curve, "
            "the faster cash conversion. Long tails indicate chronic late payers.")
    if dc and "paid" in inv.columns and "amount" in inv.columns:
        paid_inv = inv[(inv["paid"] > 0) & inv[dc].notna()].copy()
        if "paid_at" in paid_inv.columns:
            paid_inv["lag"] = (pd.to_datetime(paid_inv["paid_at"], utc=True, errors="coerce")
                               - paid_inv[dc]).dt.days
        else:
            paid_inv["lag"] = (paid_inv["paid"] / paid_inv["amount"].replace(0,np.nan) * 30).clip(0, 365)
        lag = paid_inv["lag"].dropna().sort_values()
        ecdf = np.arange(1, len(lag)+1) / len(lag)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(lag, ecdf, color=ACCENT, lw=2)
        for p, c in [(0.5,"#10b981"),(0.75,"#f59e0b"),(0.9,"#ef4444")]:
            v = lag.quantile(p)
            ax.axvline(v, ls="--", color=c, lw=1.2, label=f"P{int(p*100)}={v:.0f}d")
        ax.set_xlabel("Days to Payment"); ax.set_ylabel("Cumulative Fraction")
        ax.set_title("Payment Velocity ECDF"); ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    # ── 5. Multi-Dim Revenue Heatmap (Company × Month) ────────────────────────
    section(5, "Revenue Density Heatmap — Company × Month",
            "Pivot of gross billed revenue per company per month. "
            "Dark cells = high billing months; white gaps = churn or seasonality.")
    if dc and "company_id" in inv.columns and "amount" in inv.columns:
        df5 = inv.dropna(subset=[dc,"company_id","amount"]).copy()
        df5["ym"] = df5[dc].dt.to_period("M").astype(str)
        pivot5 = df5.pivot_table(index="company_id", columns="ym", values="amount",
                                 aggfunc="sum", fill_value=0)
        top20 = pivot5.sum(axis=1).nlargest(20).index
        pivot5 = pivot5.loc[top20]
        fig, ax = plt.subplots(figsize=(14,6))
        im = ax.imshow(pivot5.values, aspect="auto", cmap=CMAP)
        ax.set_xticks(range(pivot5.shape[1])); ax.set_xticklabels(pivot5.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(pivot5.shape[0])); ax.set_yticklabels([str(i) for i in pivot5.index], fontsize=7)
        ax.set_title("Revenue Heatmap — Top 20 Companies × Month")
        plt.colorbar(im, ax=ax); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    # ── 6. Waiver Entropy per Company ─────────────────────────────────────────
    section(6, "Waiver Entropy per Company",
            "Shannon entropy of waiver amounts per company. "
            "High entropy = unpredictable, scattered waivers (policy inconsistency). "
            "Low entropy = predictable fixed waivers.")
    if not wav.empty and "amount" in wav.columns and "invoice_id" in wav.columns:
        if "company_id" in inv.columns and "id" in inv.columns:
            wav2 = wav.merge(inv[["id","company_id"]].drop_duplicates(), left_on="invoice_id", right_on="id", how="left")
        else:
            wav2 = wav.copy(); wav2["company_id"] = wav2.get("company_id", pd.Series(["unknown"]*len(wav)))
        ent = (wav2.dropna(subset=["company_id","amount"])
               .groupby("company_id")["amount"]
               .apply(lambda s: shannon_entropy(s.round(0).astype(int).astype(str)))
               .reset_index(name="entropy").sort_values("entropy", ascending=False).head(30))
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(ent["company_id"].astype(str), ent["entropy"], color=ACCENT)
        ax.set_xlabel("Shannon Entropy (bits)"); ax.set_title("Waiver Amount Entropy per Company")
        ax.invert_yaxis(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    # ── 7. Revenue Autocorrelation (Seasonality Detection) ───────────────────
    section(7, "Revenue Autocorrelation — Seasonality Fingerprint",
            "Lag-1 to lag-30 autocorrelation of daily revenue. "
            "Spikes at lag-7 = weekly cycle; lag-30 = monthly billing cycle.")
    if dc and "amount" in inv.columns:
        ts7 = (inv.dropna(subset=[dc,"amount"])
               .set_index(dc).resample("D")["amount"].sum()
               .fillna(0))
        if len(ts7) > 30:
            lags = range(1, 31)
            acf = [ts7.autocorr(lag=l) for l in lags]
            fig, ax = plt.subplots(figsize=(10,4))
            ax.bar(list(lags), acf, color=[ACCENT if abs(a)>0.2 else "#cbd5e1" for a in acf])
            ax.axhline(0, color="black", lw=0.8)
            ax.axhline(0.2, color="red", ls="--", lw=1, label="±0.2 threshold")
            ax.axhline(-0.2, color="red", ls="--", lw=1)
            ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
            ax.set_title("Daily Revenue Autocorrelation Function (ACF)"); ax.legend()
            fig.tight_layout(); fig_to_st(fig)
        else: no_data("Need >30 days of data")
    else: no_data()

    # ── 8. Invoice Lifecycle State Machine (Transition Matrix) ───────────────
    section(8, "Invoice State Transition Probability Matrix",
            "Cross-tabulation of invoice status pairs to build a stochastic "
            "transition matrix. Reveals which states are absorbing (e.g., waived never returns).")
    if "status" in inv.columns:
        ct = pd.crosstab(inv["status"], inv["status"].shift(-1).fillna("END"), normalize="index").round(3)
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(ct.values, cmap="Blues")
        ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels(ct.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(ct.index)));   ax.set_yticklabels(ct.index)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                ax.text(j, i, f"{ct.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title("Invoice Status Transition Matrix"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data("No 'status' column found.")

    # ── 9. Spectral Analysis (FFT on Revenue) ────────────────────────────────
    section(9, "Fourier Spectral Analysis of Revenue",
            "FFT decomposes daily revenue into dominant frequency components. "
            "Peaks identify the exact cycle lengths driving revenue patterns.")
    if dc and "amount" in inv.columns:
        ts9 = (inv.dropna(subset=[dc,"amount"]).set_index(dc)
               .resample("D")["amount"].sum().fillna(0))
        if len(ts9) > 60:
            f, pxx = periodogram(ts9.values, fs=1)
            periods = 1 / (f[1:] + 1e-9)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.semilogy(periods[:len(periods)//2], pxx[1:len(periods)//2+1], color=ACCENT)
            ax.set_xlim(2, min(180, len(ts9)//2))
            ax.set_xlabel("Period (days)"); ax.set_ylabel("Power Spectral Density (log)")
            ax.set_title("FFT Periodogram — Revenue Cycle Detection")
            ax.axvline(7, color="red", ls="--", lw=1, label="7-day cycle")
            ax.axvline(30, color="orange", ls="--", lw=1, label="30-day cycle")
            ax.legend(); fig.tight_layout(); fig_to_st(fig)
        else: no_data("Need >60 days of data")
    else: no_data()

    # ── 10. Expected Value Decomposition ─────────────────────────────────────
    section(10, "Expected Value Decomposition — Frequency × Avg Invoice",
            "Total revenue = invoice frequency × average invoice value. "
            "Plots both axes — top-right quadrant is the ideal company profile.")
    if "company_id" in inv.columns and "amount" in inv.columns:
        ev = inv.groupby("company_id")["amount"].agg(freq="count", avg="mean").reset_index()
        ev["expected_rev"] = ev["freq"] * ev["avg"]
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(ev["freq"], ev["avg"], c=ev["expected_rev"],
                        cmap=CMAP, s=60, alpha=0.8, edgecolors="none")
        ax.axvline(ev["freq"].median(), ls="--", color="gray", lw=1)
        ax.axhline(ev["avg"].median(), ls="--", color="gray", lw=1)
        ax.set_xlabel("Invoice Frequency"); ax.set_ylabel("Average Invoice Value")
        ax.set_title("EV Decomposition — Frequency × Avg Invoice"); plt.colorbar(sc, ax=ax, label="Expected Revenue")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    # ── 11. Z-Score Anomaly Detection on Payments ─────────────────────────────
    section(11, "Z-Score Anomaly Detection on Payment Amounts",
            "Any payment amount with |Z|>3 is a statistical outlier. "
            "These are candidates for manual audit — over-billing, duplicate, or fraud.")
    if "amount" in inv.columns:
        df11 = inv[["id","company_id","amount"]].dropna() if "id" in inv.columns else inv[["company_id","amount"]].dropna()
        df11["z"] = stats.zscore(df11["amount"].fillna(0))
        anomalies = df11[df11["z"].abs() > 3].sort_values("z", key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.scatter(range(len(df11)), df11["amount"], s=8, alpha=0.4, color="#94a3b8", label="Normal")
        idx = df11["z"].abs() > 3
        ax.scatter(np.where(idx)[0], df11.loc[idx, "amount"], s=40, color="#ef4444", zorder=5, label="Anomaly (|Z|>3)")
        ax.set_xlabel("Invoice Index"); ax.set_ylabel("Amount")
        ax.set_title(f"Payment Anomaly Detection — {idx.sum()} anomalies found"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
        st.dataframe(safe_show(anomalies), use_container_width=True)
    else: no_data()

    # ── 12. Marginal Collection Rate by Aging Bucket ─────────────────────────
    section(12, "Marginal Collection Rate by Aging Bucket",
            "At each aging bucket, what fraction of outstanding balance is actually collected? "
            "The slope of this curve is your effective cash recovery function.")
    if "balance" in inv.columns and "paid" in inv.columns and dc:
        now = pd.Timestamp.now(tz="UTC")
        df12 = inv.dropna(subset=["balance","paid",dc]).copy()
        df12["age_days"] = (now - df12[dc]).dt.days.clip(lower=0)
        df12["bucket"] = pd.cut(df12["age_days"], bins=[0,30,60,90,180,365,9999],
                                labels=["0-30","31-60","61-90","91-180","181-365","365+"])
        mcr = df12.groupby("bucket", observed=True).agg(
            total_balance=("balance","sum"), total_paid=("paid","sum")
        ).reset_index()
        mcr["collection_rate"] = mcr["total_paid"] / (mcr["total_balance"] + mcr["total_paid"]).replace(0,np.nan)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.bar(mcr["bucket"].astype(str), mcr["collection_rate"]*100, color=ACCENT)
        ax.set_xlabel("Age Bucket (days)"); ax.set_ylabel("Collection Rate (%)")
        ax.set_title("Marginal Collection Rate by Invoice Age"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  INPATIENT  (analyses 13-22)
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice == "Inpatient":
    st.title("🛏️ Inpatient — 10 Hardcore Analyses")
    tables = table_map.get("Inpatient", [])
    sel, _ = pick_table("Inpatient")
    if not sel: st.warning("No tables found."); st.stop()

    with st.spinner(f"Loading {sel}…"):
        df = load_table("Inpatient", sel)
    st.success(f"{len(df):,} rows  ·  {len(df.columns)} columns  ·  Table: {sel}")
    dc = best_date_col(df); nc = num_cols(df); cc = cat_cols(df)

    section(13, "Length-of-Stay Survival Distribution",
            "Empirical survival function P(LOS > t). Convex = many short stays; concave = chronic care dominant.")
    los_col = next((c for c in ["length_of_stay","los","days","duration"] if c in df.columns), None)
    if los_col:
        los = pd.to_numeric(df[los_col], errors="coerce").dropna().clip(lower=0)
        sorted_los = np.sort(los)
        surv = 1 - np.arange(1,len(sorted_los)+1)/len(sorted_los)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.step(sorted_los, surv, color=ACCENT, lw=2)
        ax.set_xlabel("Length of Stay (days)"); ax.set_ylabel("P(LOS > t)")
        ax.set_title("LOS Empirical Survival Curve"); ax.grid(True, alpha=0.3); fig.tight_layout(); fig_to_st(fig)
    else: no_data("No LOS column found.")

    section(14, "Admission Surge Detection via FFT",
            "Fourier transform on daily admissions detects periodic surge cycles (e.g., post-weekend surges).")
    if dc:
        ts14 = df.set_index(dc).resample("D").size().fillna(0)
        if len(ts14) > 30:
            f, pxx = periodogram(ts14.values, fs=1)
            periods = 1/(f[1:]+1e-9)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.semilogy(periods[:len(periods)//2], pxx[1:len(periods)//2+1], color=ACCENT)
            ax.set_xlim(2, 60); ax.set_xlabel("Period (days)")
            ax.set_ylabel("Power (log)"); ax.set_title("Admission Surge — FFT Periodogram")
            for v,l,c in [(7,"7d","red"),(14,"14d","orange"),(30,"30d","green")]:
                ax.axvline(v, ls="--", color=c, lw=1, label=l)
            ax.legend(); fig.tight_layout(); fig_to_st(fig)
        else: no_data()
    else: no_data()

    section(15, "Ward/Department Load Imbalance (Gini)",
            "Gini coefficient applied to patient volume per ward. "
            "High Gini = most patients routed to same ward, others idle.")
    ward_col = next((c for c in cc if any(w in c.lower() for w in ["ward","dept","unit","room"])), cc[0] if cc else None)
    if ward_col:
        wc = df[ward_col].value_counts()
        G = gini(wc.values)
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        axes[0].barh(wc.index[:20].astype(str), wc.values[:20], color=ACCENT)
        axes[0].set_title("Patient Volume by Ward (Top 20)"); axes[0].invert_yaxis()
        axes[1].pie(wc.values[:8], labels=[str(x) for x in wc.index[:8]], autopct="%1.1f%%")
        axes[1].set_title(f"Ward Distribution  (Gini={G:.3f})")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(16, "Patient Acuity Drift Over Time",
            "Rolling average of any numeric acuity/severity score. "
            "Rising trend = sicker patients over time; declining = healthier case mix.")
    sev_col = next((c for c in nc if any(w in c.lower() for w in ["score","severity","acuity","level","grade"]) ), nc[0] if nc else None)
    if sev_col and dc:
        df16 = df.dropna(subset=[dc, sev_col]).set_index(dc).sort_index()
        roll = df16[sev_col].rolling("30D").mean()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(df16.index, df16[sev_col], alpha=0.2, color=ACCENT, lw=0.5)
        ax.plot(roll.index, roll.values, color="#f97316", lw=2, label="30-day rolling avg")
        ax.set_ylabel(sev_col); ax.set_title("Patient Acuity Drift (30D Rolling)")
        ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(17, "Multi-Dimensional Box Plot — Numerics by Category",
            "Violin + box plot of all numeric columns stratified by top categorical column. "
            "Reveals distributional differences hidden in simple means.")
    if nc and cc:
        top_cc = max(cc, key=lambda c: df[c].nunique())
        top_cats = df[top_cc].value_counts().index[:6].tolist()
        subset = df[df[top_cc].isin(top_cats)]
        cols_to_plot = nc[:4]
        fig, axes = plt.subplots(1, len(cols_to_plot), figsize=(14, 5), sharey=False)
        for ax, col in zip(axes, cols_to_plot):
            data = [subset[subset[top_cc]==cat][col].dropna().values for cat in top_cats]
            vp = ax.violinplot(data, showmedians=True)
            for body in vp["bodies"]: body.set_alpha(0.7); body.set_color(ACCENT)
            ax.set_xticks(range(1,len(top_cats)+1))
            ax.set_xticklabels([str(c)[:10] for c in top_cats], rotation=45, ha="right", fontsize=7)
            ax.set_title(col)
        fig.suptitle(f"Distribution by '{top_cc}'"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(18, "Readmission Risk — Time Between Visits",
            "Inter-admission interval distribution per patient. "
            "Bimodal distribution signals two distinct readmission pathways.")
    pid_col = next((c for c in df.columns if "patient" in c.lower() and "id" in c.lower()), None)
    if pid_col and dc:
        df18 = df.dropna(subset=[pid_col,dc]).sort_values(dc)
        df18["prev_date"] = df18.groupby(pid_col)[dc].shift(1)
        df18["interval"] = (df18[dc] - df18["prev_date"]).dt.days.dropna()
        intervals = df18["interval"].dropna().clip(0,365)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(intervals, bins=50, color=ACCENT, edgecolor="white", linewidth=0.3)
        ax.axvline(intervals.mean(), color="red", ls="--", lw=1.5, label=f"Mean={intervals.mean():.1f}d")
        ax.axvline(30, color="orange", ls="--", lw=1, label="30-day mark")
        ax.set_xlabel("Days Between Admissions"); ax.set_title("Readmission Interval Distribution")
        ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(19, "Discharge Delay Cascade (Days Overdue)",
            "Histogram of stays exceeding expected duration. "
            "The right tail of this distribution is pure operational waste.")
    exp_col = next((c for c in df.columns if "expected" in c.lower() or "planned" in c.lower()), None)
    if los_col and exp_col:
        df19 = df[[los_col, exp_col]].dropna()
        df19["overstay"] = pd.to_numeric(df19[los_col],errors="coerce") - pd.to_numeric(df19[exp_col],errors="coerce")
        delays = df19["overstay"].dropna()
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(delays, bins=40, color=ACCENT)
        ax.axvline(0, color="red", ls="--", lw=1.5, label="On-time")
        ax.set_xlabel("Days over expected"); ax.set_title("Discharge Delay Distribution"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    elif nc and len(nc) >= 2:
        fig, ax = plt.subplots(figsize=(9,4))
        diff = df[nc[0]].dropna() - df[nc[1]].dropna()
        ax.hist(diff.dropna(), bins=40, color=ACCENT)
        ax.set_xlabel(f"{nc[0]} - {nc[1]}"); ax.set_title("Numeric Difference Distribution")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(20, "Inpatient Volume Heatmap — Hour × Day of Week",
            "Event density across hour-of-day and day-of-week. "
            "Identifies shift patterns, weekend dips, and overnight peaks.")
    if dc:
        df20 = df.dropna(subset=[dc]).copy()
        df20["hour"] = df20[dc].dt.hour
        df20["dow"]  = df20[dc].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hm = df20.pivot_table(index="hour", columns="dow", values=nc[0] if nc else dc,
                               aggfunc="count", fill_value=0)
        hm = hm.reindex(columns=[d for d in dow_order if d in hm.columns])
        fig, ax = plt.subplots(figsize=(12,6))
        im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(24)); ax.set_yticklabels(range(24), fontsize=7)
        ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns, fontsize=9)
        ax.set_title("Activity Heatmap — Hour × Day of Week"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(21, "Hierarchical Clustering of Numeric Features",
            "Dendrogram of patient/record clusters based on all numeric columns. "
            "Reveals natural sub-populations in the data.")
    if len(nc) >= 2:
        sample = df[nc].dropna().sample(min(300, len(df)), random_state=42)
        Z = linkage(sample.values, method="ward")
        fig, ax = plt.subplots(figsize=(12,5))
        dendrogram(Z, ax=ax, truncate_mode="lastp", p=20,
                   leaf_font_size=8, color_threshold=0.7*max(Z[:,2]))
        ax.set_title("Hierarchical Clustering Dendrogram"); ax.set_xlabel("Records")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(22, "Correlation Matrix of All Numeric Columns",
            "Full Pearson correlation heatmap. Values near ±1 are collinear risks "
            "or strong predictive relationships worth modelling.")
    if len(nc) >= 2:
        corr = df[nc].corr()
        fig, ax = plt.subplots(figsize=(max(6,len(nc)), max(5,len(nc)-1)))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(nc))); ax.set_xticklabels(nc, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(nc))); ax.set_yticklabels(nc, fontsize=8)
        for i in range(len(nc)):
            for j in range(len(nc)):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(corr.values[i,j]) > 0.5 else "black")
        ax.set_title("Numeric Correlation Matrix"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  THEATRE  (analyses 23-32)
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice == "Theatre":
    st.title("🏥 Theatre — 10 Hardcore Analyses")
    sel, _ = pick_table("Theatre")
    if not sel: st.warning("No tables."); st.stop()
    with st.spinner(): df = load_table("Theatre", sel)
    st.success(f"{len(df):,} rows · Table: {sel}")
    dc = best_date_col(df); nc = num_cols(df); cc = cat_cols(df)

    section(23, "Theatre Utilisation Heatmap (Hour × Day)",
            "Which slots are over/under-booked? High utilisation at unusual hours reveals scheduling inefficiencies.")
    if dc:
        df23 = df.dropna(subset=[dc]).copy()
        df23["hour"] = df23[dc].dt.hour; df23["dow"] = df23[dc].dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hm = df23.pivot_table(index="hour", columns="dow", aggfunc="size", fill_value=0)
        hm = hm.reindex(columns=[d for d in dow_order if d in hm.columns])
        fig, ax = plt.subplots(figsize=(12,6))
        im = ax.imshow(hm.values, aspect="auto", cmap="hot_r")
        ax.set_yticks(range(24)); ax.set_yticklabels(range(24), fontsize=7)
        ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns)
        ax.set_title("Theatre Utilisation Heatmap"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(24, "Surgeon/Doctor Performance Z-Score",
            "Each doctor's average case duration vs specialty mean. Z>2 consistently = systematic over-runner.")
    doc_col  = next((c for c in cc if any(w in c.lower() for w in ["doctor","surgeon","physician","provider"])), cc[0] if cc else None)
    dur_col  = next((c for c in nc if any(w in c.lower() for w in ["duration","time","minutes","hours"])), nc[0] if nc else None)
    if doc_col and dur_col:
        perf = df.groupby(doc_col)[dur_col].agg(["mean","std","count"]).reset_index()
        overall_mean = df[dur_col].mean(); overall_std = df[dur_col].std()
        perf["z"] = (perf["mean"] - overall_mean) / (overall_std + 1e-9)
        perf = perf.sort_values("z", ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(10,6))
        colors = ["#ef4444" if z > 1 else "#10b981" if z < -1 else "#94a3b8" for z in perf["z"]]
        ax.barh(perf[doc_col].astype(str), perf["z"], color=colors)
        ax.axvline(0, color="black", lw=1); ax.axvline(1,"red","--",lw=1); ax.axvline(-1,"green","--",lw=1)
        ax.set_title("Doctor Performance Z-Score (Duration)"); ax.invert_yaxis()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(25, "Case Complexity Distribution by Procedure/Type",
            "Violin plot of case duration by procedure type — procedure-level SLA benchmarking.")
    proc_col = next((c for c in cc if any(w in c.lower() for w in ["procedure","case","type","operation"])), cc[0] if cc else None)
    if proc_col and dur_col:
        top_procs = df[proc_col].value_counts().index[:8].tolist()
        sub = df[df[proc_col].isin(top_procs)]
        fig, ax = plt.subplots(figsize=(12,5))
        data = [sub[sub[proc_col]==p][dur_col].dropna().values for p in top_procs]
        vp = ax.violinplot(data, showmedians=True)
        for b in vp["bodies"]: b.set_alpha(0.7); b.set_color(ACCENT)
        ax.set_xticks(range(1,len(top_procs)+1))
        ax.set_xticklabels([str(p)[:15] for p in top_procs], rotation=30, ha="right")
        ax.set_ylabel(dur_col); ax.set_title("Case Duration Distribution by Procedure")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(26, "Theatre Cancellation Root Cause Pareto",
            "80/20 decomposition of cancellation reasons. The top 2-3 reasons typically account for 80% of lost slots.")
    cancel_col = next((c for c in cc if "cancel" in c.lower() or "reason" in c.lower()), None)
    status_col = next((c for c in cc if "status" in c.lower()), None)
    reason_col = cancel_col or status_col
    if reason_col:
        counts = df[reason_col].value_counts()
        cum = counts.cumsum() / counts.sum() * 100
        fig, ax = plt.subplots(figsize=(10,5))
        ax2 = ax.twinx()
        ax.bar(range(len(counts)), counts.values, color=ACCENT, alpha=0.8)
        ax2.plot(range(len(counts)), cum.values, color="red", lw=2, marker="o", ms=4)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(x)[:20] for x in counts.index], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Count"); ax2.set_ylabel("Cumulative %")
        ax.set_title("Pareto — Case Cancellation / Status Reasons")
        ax2.axhline(80, color="orange", ls="--", lw=1)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(27, "First-Case Delay Cascade — How Overruns Propagate",
            "Distribution of case start time vs scheduled time. Positive = delayed start. "
            "Cascade: one delayed first case shifts every subsequent case in the day.")
    sched_col  = next((c for c in df.columns if "schedul" in c.lower() or "plan" in c.lower()), None)
    actual_col = next((c for c in df.columns if "actual" in c.lower() or "start" in c.lower()), None)
    if sched_col and actual_col:
        s = pd.to_datetime(df[sched_col], utc=True, errors="coerce")
        a = pd.to_datetime(df[actual_col], utc=True, errors="coerce")
        delay = (a - s).dt.total_seconds().div(60).dropna()
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(delay, bins=50, color=ACCENT)
        ax.axvline(0, color="red", ls="--", lw=1.5, label="On-time")
        ax.axvline(delay.mean(), color="orange", ls="--", lw=1, label=f"Mean={delay.mean():.1f}min")
        ax.set_xlabel("Minutes (actual - scheduled)"); ax.set_title("Case Start Delay Distribution"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    elif nc:
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(df[nc[0]].dropna(), bins=40, color=ACCENT)
        ax.set_title(f"Distribution: {nc[0]}"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(28, "Turnover Time Distribution by Hour of Day",
            "Turnover = time between cases. Peaks in afternoon hours reveal compounding delays. "
            "Each wasted minute of turnover multiplies across all remaining cases.")
    if dc and dur_col:
        df28 = df.dropna(subset=[dc, dur_col]).sort_values(dc).copy()
        df28["turnover"] = df28[dc].diff().dt.total_seconds().div(60) - df28[dur_col]
        df28 = df28[df28["turnover"].between(0, 180)]
        df28["hour"] = df28[dc].dt.hour
        turn_by_hr = df28.groupby("hour")["turnover"].median()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(turn_by_hr.index, turn_by_hr.values, color=ACCENT)
        ax.set_xlabel("Hour of Day"); ax.set_ylabel("Median Turnover (min)")
        ax.set_title("Turnover Time by Hour of Day"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(29, "Revenue per Theatre Hour (Efficiency Frontier)",
            "If revenue data is available: revenue-per-hour by doctor/procedure. "
            "Otherwise: case volume per hour slot — the throughput efficiency frontier.")
    if doc_col and dur_col:
        eff = df.groupby(doc_col).agg(cases=(dur_col,"count"), total_time=(dur_col,"sum")).reset_index()
        eff["cases_per_hour"] = (eff["cases"] / eff["total_time"].replace(0,np.nan) * 60).round(2)
        eff = eff.nlargest(20,"cases_per_hour")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(eff[doc_col].astype(str), eff["cases_per_hour"], color=ACCENT)
        ax.set_xlabel("Cases per Hour"); ax.set_title("Throughput Efficiency Frontier (Top 20 Doctors)")
        ax.invert_yaxis(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(30, "Case Over-Run Detection — Boxplot by Category",
            "Statistical outlier cases by category. Outliers (IQR method) are cases to audit for billing discrepancies.")
    if dur_col and cc:
        cat = max(cc, key=lambda c: df[c].nunique())
        top = df[cat].value_counts().index[:6]
        sub = df[df[cat].isin(top)]
        fig, ax = plt.subplots(figsize=(11,5))
        groups = [sub[sub[cat]==c][dur_col].dropna().values for c in top]
        bp = ax.boxplot(groups, patch_artist=True, notch=True)
        for patch in bp["boxes"]: patch.set_facecolor(ACCENT); patch.set_alpha(0.7)
        ax.set_xticks(range(1,len(top)+1)); ax.set_xticklabels([str(c)[:15] for c in top], rotation=30, ha="right")
        ax.set_ylabel(dur_col); ax.set_title("Case Duration Boxplot by Category (Outlier Detection)")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(31, "Rolling 7-Day Theatre Volume Trend",
            "Smoothed trend line of daily case volume. Dips identify understaffed/underbooked periods.")
    if dc:
        ts31 = df.set_index(dc).resample("D").size().fillna(0)
        roll7 = ts31.rolling(7).mean()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.fill_between(ts31.index, ts31.values, alpha=0.3, color=ACCENT)
        ax.plot(roll7.index, roll7.values, color="#f97316", lw=2, label="7D rolling avg")
        ax.set_title("Daily Theatre Case Volume + 7D Rolling Average"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(32, "Procedure Mix Shift — Period-over-Period Comparison",
            "Compares procedure type distribution between two equal time periods. "
            "Shift in mix impacts average revenue per case.")
    if dc and proc_col:
        ts = df.dropna(subset=[dc, proc_col]).sort_values(dc)
        mid = ts[dc].quantile(0.5)
        h1 = ts[ts[dc] < mid][proc_col].value_counts(normalize=True)
        h2 = ts[ts[dc] >= mid][proc_col].value_counts(normalize=True)
        mix = pd.DataFrame({"Period 1": h1, "Period 2": h2}).fillna(0).head(10)
        fig, ax = plt.subplots(figsize=(11,5))
        x = np.arange(len(mix))
        ax.bar(x-0.2, mix["Period 1"]*100, 0.4, label="Period 1", color=ACCENT, alpha=0.8)
        ax.bar(x+0.2, mix["Period 2"]*100, 0.4, label="Period 2", color="#f97316", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([str(i)[:15] for i in mix.index], rotation=30, ha="right")
        ax.set_ylabel("% of Cases"); ax.set_title("Procedure Mix Shift — H1 vs H2"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  RECEPTION  (analyses 33-42)
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice == "Reception":
    st.title("🏨 Reception — 10 Hardcore Analyses")
    sel, _ = pick_table("Reception")
    if not sel: st.warning("No tables."); st.stop()
    with st.spinner(): df = load_table("Reception", sel)
    st.success(f"{len(df):,} rows · Table: {sel}")
    dc = best_date_col(df); nc = num_cols(df); cc = cat_cols(df)

    section(33, "Arrival Intensity Heatmap — Hour × Day",
            "Patient arrival density matrix. Saturated cells drive queueing pressure and missed SLAs.")
    if dc:
        df33 = df.dropna(subset=[dc]).copy()
        df33["hour"] = df33[dc].dt.hour; df33["dow"] = df33[dc].dt.day_name()
        dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hm = df33.groupby(["hour","dow"]).size().unstack(fill_value=0).reindex(columns=[d for d in dow if d in df33["dow"].unique()])
        fig, ax = plt.subplots(figsize=(12,6))
        im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(24)); ax.set_yticklabels(range(24), fontsize=7)
        ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns)
        ax.set_title("Arrival Intensity Heatmap"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(34, "Wait Time Percentile Curves (P50/P75/P90/P95)",
            "Shows the full wait time distribution — not just the average. "
            "P95 is what your worst-off 5% experience.")
    wait_col = next((c for c in nc if "wait" in str(c).lower() or "queue" in str(c).lower()), nc[0] if nc else None)
    if wait_col:
        w = df[wait_col].dropna().clip(lower=0)
        fig, axes = plt.subplots(1,2,figsize=(13,4))
        axes[0].hist(w, bins=60, color=ACCENT)
        for p,c in [(50,"green"),(75,"orange"),(90,"red"),(95,"darkred")]:
            v = np.percentile(w, p)
            axes[0].axvline(v, color=c, ls="--", lw=1.2, label=f"P{p}={v:.0f}")
        axes[0].legend(fontsize=8); axes[0].set_title("Wait Time Distribution with Percentiles")
        axes[1].plot(np.sort(w), np.linspace(0,1,len(w)), color=ACCENT, lw=2)
        axes[1].set_title("Wait Time ECDF"); axes[1].set_xlabel("Wait Time")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(35, "Registration-to-Treatment Funnel",
            "Counts how many patients move through each stage. Drop-offs at each stage quantify leakage.")
    stages = [c for c in df.columns if any(w in c.lower() for w in ["register","triage","admit","assign","treat","discharge"])]
    if len(stages) >= 2:
        counts = {s: df[s].notna().sum() for s in stages}
        names = list(counts.keys()); vals = list(counts.values())
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(names[::-1], vals[::-1], color=[ACCENT]*len(names))
        for i,(n,v) in enumerate(zip(names[::-1],vals[::-1])):
            ax.text(v, i, f" {v:,}", va="center", fontsize=9)
        ax.set_title("Patient Flow Funnel"); fig.tight_layout(); fig_to_st(fig)
    elif cc:
        counts = df[cc[0]].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.barh(counts.index.astype(str), counts.values, color=ACCENT)
        ax.set_title(f"{cc[0]} distribution"); ax.invert_yaxis()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(36, "Appointment Lead Time Distribution",
            "Days between booking and appointment date. "
            "Heavy right tail = patients booking months out; likely to no-show or cancel.")
    book_col = next((c for c in df.columns if "book" in c.lower() or "created" in c.lower()), None)
    appt_col = next((c for c in df.columns if "appt" in c.lower() or "schedul" in c.lower() or "visit" in c.lower()), None)
    if book_col and appt_col:
        b = pd.to_datetime(df[book_col], utc=True, errors="coerce")
        a = pd.to_datetime(df[appt_col], utc=True, errors="coerce")
        lead = (a - b).dt.days.dropna().clip(0, 180)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(lead, bins=50, color=ACCENT)
        ax.axvline(lead.mean(), color="red", ls="--", lw=1.5, label=f"Mean={lead.mean():.1f}d")
        ax.set_xlabel("Lead Days"); ax.set_title("Appointment Lead Time Distribution"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    elif dc:
        ts = df.set_index(dc).resample("D").size()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(ts.index, ts.values, color=ACCENT, lw=1.5)
        ax.set_title("Daily Registration Volume"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(37, "No-Show / Status Breakdown Stacked Time Series",
            "Monthly breakdown of appointment statuses stacked. Rising 'missed' category = worsening adherence.")
    status_col = next((c for c in cc if "status" in c.lower()), cc[0] if cc else None)
    if status_col and dc:
        df37 = df.dropna(subset=[dc, status_col]).copy()
        df37["ym"] = df37[dc].dt.to_period("M").astype(str)
        pivot = df37.pivot_table(index="ym", columns=status_col, aggfunc="size", fill_value=0)
        fig, ax = plt.subplots(figsize=(13,5))
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap=CMAP)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=7)
        ax.set_title("Monthly Status Breakdown (Stacked)"); ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(38, "Queue Depth Timeline (Rolling 15-min equivalent)",
            "Rolling count of 'open' patients — approximates live queue depth. "
            "Peaks reveal when the front desk is most overloaded.")
    if dc:
        ts38 = df.set_index(dc).resample("H").size()
        roll = ts38.rolling(3).mean()
        fig, ax = plt.subplots(figsize=(13,4))
        ax.fill_between(ts38.index, ts38.values, alpha=0.3, color=ACCENT)
        ax.plot(roll.index, roll.values, color="#f97316", lw=2, label="3H Rolling Avg")
        ax.set_title("Hourly Registration Volume (Queue Proxy)"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(39, "Referral / Source Channel Attribution",
            "Which referral source brings the most patients? And which source has the highest subsequent conversion?")
    ref_col = next((c for c in cc if any(w in c.lower() for w in ["referral","source","channel","refer"])), None)
    if ref_col:
        counts = df[ref_col].value_counts().head(20)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(counts.index.astype(str), counts.values, color=ACCENT)
        ax.set_title("Patient Volume by Referral Source"); ax.invert_yaxis()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data("No referral/source column found.")

    section(40, "Reception Throughput — Records per Hour by Staff",
            "Output rate per clerk/user. Outliers above norm = champion processors; below = training gap.")
    staff_col = next((c for c in cc if any(w in c.lower() for w in ["user","staff","clerk","created_by","attendant"])), None)
    if staff_col and dc:
        df40 = df.dropna(subset=[dc, staff_col]).copy()
        df40["hour"] = df40[dc].dt.floor("H")
        thru = df40.groupby(staff_col)["hour"].count().reset_index(name="records")
        hours_worked = df40.groupby(staff_col)["hour"].nunique().reset_index(name="hours")
        thru = thru.merge(hours_worked, on=staff_col)
        thru["records_per_hour"] = (thru["records"] / thru["hours"].replace(0,np.nan)).round(2)
        thru = thru.nlargest(20, "records_per_hour")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(thru[staff_col].astype(str), thru["records_per_hour"], color=ACCENT)
        ax.set_title("Reception Throughput per Staff Member"); ax.invert_yaxis()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(41, "Patient Volume Forecast — EWMA Extrapolation",
            "Exponentially weighted moving average extrapolates the next 30 days of patient volume. "
            "This is a zero-parameter stochastic smoother — no overfitting.")
    if dc:
        ts41 = df.set_index(dc).resample("D").size().fillna(0)
        ewma = ts41.ewm(span=14).mean()
        future_idx = pd.date_range(ts41.index[-1], periods=31, freq="D", tz="UTC")[1:]
        last_ewma = ewma.iloc[-1]
        decay = last_ewma * np.exp(-np.arange(1,31)/14)
        fig, ax = plt.subplots(figsize=(13,4))
        ax.plot(ts41.index, ts41.values, alpha=0.4, color=ACCENT, lw=1, label="Actual")
        ax.plot(ewma.index, ewma.values, color=ACCENT, lw=2, label="EWMA")
        ax.plot(future_idx, decay, color="#f97316", ls="--", lw=2, label="Forecast (30D)")
        ax.axvline(ts41.index[-1], color="gray", ls=":", lw=1)
        ax.set_title("Patient Volume EWMA Forecast"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(42, "Repeat Visit Pattern — Inter-Visit Interval Histogram",
            "How long between a patient's visits? Bimodal = two distinct return drivers. "
            "Very short intervals (<3 days) may indicate unresolved complaints.")
    pid_col = next((c for c in df.columns if "patient" in c.lower() and "id" in c.lower()), None)
    if pid_col and dc:
        df42 = df.dropna(subset=[pid_col, dc]).sort_values(dc)
        df42["prev"] = df42.groupby(pid_col)[dc].shift(1)
        df42["interval"] = (df42[dc] - df42["prev"]).dt.days
        ivl = df42["interval"].dropna().clip(0,365)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(ivl, bins=60, color=ACCENT)
        ax.axvline(7, color="red", ls="--", lw=1, label="7d")
        ax.axvline(30, color="orange", ls="--", lw=1, label="30d")
        ax.set_xlabel("Days Between Visits"); ax.set_title("Inter-Visit Interval Distribution"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  INVENTORY  (analyses 43-52)
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice == "Inventory":
    st.title("📦 Inventory — 10 Hardcore Analyses")
    sel, _ = pick_table("Inventory")
    if not sel: st.warning("No tables."); st.stop()
    with st.spinner(): df = load_table("Inventory", sel)
    st.success(f"{len(df):,} rows · Table: {sel}")
    dc = best_date_col(df); nc = num_cols(df); cc = cat_cols(df)
    qty_col  = next((c for c in nc if any(w in c.lower() for w in ["qty","quantity","stock","units"])), nc[0] if nc else None)
    val_col  = next((c for c in nc if any(w in c.lower() for w in ["cost","price","value","amount"])), nc[1] if len(nc)>1 else None)
    item_col = next((c for c in cc if any(w in c.lower() for w in ["item","product","drug","name","sku"])), cc[0] if cc else None)

    section(43, "ABC-XYZ Classification Matrix",
            "ABC = value classification (A=top 70%, B=next 20%, C=bottom 10%). "
            "XYZ = demand variability (X=stable, Y=variable, Z=erratic). "
            "Combined matrix drives replenishment strategy per SKU.")
    if item_col and val_col and qty_col:
        abc = df.groupby(item_col).agg(total_val=(val_col,"sum"), cv_qty=(qty_col, cv)).reset_index().dropna()
        abc = abc.sort_values("total_val", ascending=False)
        abc["cum_pct"] = abc["total_val"].cumsum() / abc["total_val"].sum()
        abc["ABC"] = abc["cum_pct"].apply(lambda x: "A" if x<=0.7 else "B" if x<=0.9 else "C")
        abc["XYZ"] = abc["cv_qty"].apply(lambda x: "X" if x<0.1 else "Y" if x<0.25 else "Z")
        abc["class"] = abc["ABC"] + abc["XYZ"]
        matrix = pd.crosstab(abc["ABC"], abc["XYZ"], values=abc["total_val"], aggfunc="sum").fillna(0)
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(matrix.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(matrix.columns))); ax.set_xticklabels(matrix.columns)
        ax.set_yticks(range(len(matrix.index)));   ax.set_yticklabels(matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j,i,f"{matrix.values[i,j]:,.0f}",ha="center",va="center",fontsize=9)
        ax.set_title("ABC-XYZ Matrix (Total Value by Segment)"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(44, "Inventory Turnover & Days of Supply",
            "Days of supply = stock / daily usage rate. Items with >90 DOS are candidates for stock reduction. "
            "Items with <7 DOS are stockout risk.")
    if qty_col and dc:
        daily_usage = df.groupby(dc)[qty_col].sum().resample("D").mean().mean() if item_col else df[qty_col].mean()
        if item_col:
            stock_df = df.groupby(item_col)[qty_col].sum().reset_index(name="stock")
            stock_df["dos"] = (stock_df["stock"] / (daily_usage + 1e-9)).round(1)
            stock_df["risk"] = stock_df["dos"].apply(lambda x: "🚨 Stockout" if x<7 else "⚠️ Excess" if x>90 else "✅ OK")
            fig, ax = plt.subplots(figsize=(10,5))
            colors = {"🚨 Stockout":"#ef4444","⚠️ Excess":"#f97316","✅ OK":"#10b981"}
            for risk, grp in stock_df.groupby("risk"):
                ax.scatter(grp["stock"], grp["dos"], label=risk, color=colors[risk], alpha=0.7, s=40)
            ax.axhline(7, color="red", ls="--", lw=1)
            ax.axhline(90, color="orange", ls="--", lw=1)
            ax.set_xlabel("Total Stock"); ax.set_ylabel("Days of Supply")
            ax.set_title("Inventory DOS vs Stock Level"); ax.legend()
            fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(45, "Dead Stock Aging Distribution",
            "Stock items with zero movement in 30/60/90/180 days. "
            "Value locked in dead stock is working capital trapped in the warehouse.")
    if dc and item_col:
        now = pd.Timestamp.now(tz="UTC")
        last_seen = df.dropna(subset=[dc]).groupby(item_col)[dc].max().reset_index()
        last_seen["days_idle"] = (now - last_seen[dc]).dt.days.clip(lower=0)
        bins = [0,30,60,90,180,365,9999]
        labels = ["<30d","30-60d","60-90d","90-180d","180-365d","365d+"]
        last_seen["bucket"] = pd.cut(last_seen["days_idle"], bins=bins, labels=labels)
        bc = last_seen.groupby("bucket", observed=True).size()
        fig, ax = plt.subplots(figsize=(9,4))
        ax.bar(bc.index.astype(str), bc.values, color=[ACCENT if i<3 else "#ef4444" for i in range(len(bc))])
        ax.set_xlabel("Days Since Last Movement"); ax.set_ylabel("SKU Count")
        ax.set_title("Dead Stock Aging Buckets"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(46, "Demand Volatility Coefficient of Variation",
            "CV = std/mean of demand per SKU. High CV = erratic demand = dangerous to stock. "
            "Sorted distribution shows which SKUs you can't forecast reliably.")
    if item_col and qty_col:
        cv_df = df.groupby(item_col)[qty_col].agg(cv).reset_index(name="cv").dropna().sort_values("cv", ascending=False)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(range(len(cv_df)), cv_df["cv"].values, color=ACCENT, lw=1.5)
        ax.fill_between(range(len(cv_df)), cv_df["cv"].values, alpha=0.2, color=ACCENT)
        ax.axhline(cv_df["cv"].median(), color="red", ls="--", lw=1, label=f"Median CV={cv_df['cv'].median():.2f}")
        ax.set_xlabel("SKU Rank (by CV)"); ax.set_ylabel("Coefficient of Variation")
        ax.set_title("Demand Volatility Distribution across SKUs"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(47, "Supplier Concentration Risk (Herfindahl Index)",
            "HHI = sum of squared market shares. HHI>0.25 = highly concentrated supply chain. "
            "Single-supplier dependency is an operational cliff edge.")
    sup_col = next((c for c in cc if any(w in c.lower() for w in ["supplier","vendor","manufacturer"])), None)
    if sup_col and val_col:
        sup_share = df.groupby(sup_col)[val_col].sum()
        share_pct = sup_share / sup_share.sum()
        hhi = (share_pct**2).sum()
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        share_pct.nlargest(10).plot(kind="bar", ax=axes[0], color=ACCENT)
        axes[0].set_title("Top 10 Supplier Value Share"); axes[0].set_ylabel("Share")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
        axes[1].pie(share_pct.nlargest(6).values, labels=[str(i)[:15] for i in share_pct.nlargest(6).index], autopct="%1.1f%%")
        axes[1].set_title(f"Supplier Concentration  (HHI={hhi:.3f})")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(48, "Economic Order Quantity Estimation",
            "EOQ = sqrt(2DS/H) per SKU. Orders too large = excess carrying cost; too small = high ordering frequency. "
            "This shows the optimal order size implied by the data.")
    if item_col and qty_col and val_col:
        eoq_df = df.groupby(item_col).agg(D=(qty_col,"sum"), H=(val_col,"mean")).reset_index()
        S = 50
        eoq_df["EOQ"] = np.sqrt(2 * eoq_df["D"] * S / (eoq_df["H"].replace(0,np.nan) + 1e-9)).round(0)
        eoq_df = eoq_df.dropna().nlargest(20,"D")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(range(len(eoq_df)), eoq_df["EOQ"], color=ACCENT, alpha=0.8)
        ax.bar(range(len(eoq_df)), eoq_df["D"]/12, color="#f97316", alpha=0.4, label="Avg Monthly Demand")
        ax.set_xticks(range(len(eoq_df)))
        ax.set_xticklabels([str(i)[:12] for i in eoq_df[item_col]], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Units"); ax.set_title("EOQ vs Monthly Demand (Top 20 SKUs)"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(49, "Inventory Value Concentration — Lorenz Curve",
            "What % of SKUs hold what % of inventory value? "
            "If top 10% of SKUs = 90% of value, that's where your carrying cost risk lives.")
    if item_col and val_col:
        val_per_item = df.groupby(item_col)[val_col].sum().sort_values()
        G = gini(val_per_item.values)
        cum = val_per_item.cumsum() / val_per_item.sum()
        x = np.linspace(0,1,len(cum))
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(x, cum.values, color=ACCENT, lw=2, label=f"Lorenz Curve (Gini={G:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1,label="Equality")
        ax.fill_between(x, cum.values, x, alpha=0.15, color=ACCENT)
        ax.set_xlabel("Cumulative % SKUs"); ax.set_ylabel("Cumulative % Value")
        ax.set_title("Inventory Value Lorenz Curve"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(50, "Reorder Breach Frequency — Stockout Probability",
            "% of time each item's recorded stock fell below a simulated reorder point. "
            "Frequency of breaches proxy how often you're gambling on stockouts.")
    if item_col and qty_col:
        rp_df = df.groupby(item_col)[qty_col].agg(["mean","std","min","count"]).reset_index()
        rp_df["reorder_point"] = rp_df["mean"] - rp_df["std"]
        rp_df["breach_prob"] = (rp_df["min"] < rp_df["reorder_point"]).astype(int)
        rp_df = rp_df.sort_values("breach_prob", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(rp_df[item_col].astype(str), rp_df["mean"], color="#94a3b8", label="Avg Stock")
        ax.barh(rp_df[item_col].astype(str), rp_df["reorder_point"], color=ACCENT, alpha=0.6, label="Reorder Point")
        ax.set_title("Stock Level vs Reorder Point (Top 20 At-Risk)"); ax.legend(); ax.invert_yaxis()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(51, "Purchase Order Cycle Time Distribution",
            "Distribution of time between consecutive orders per supplier/item. "
            "Right tail = slow replenishment cycles; left tail = over-ordering.")
    if dc and (item_col or sup_col):
        grp_col = item_col or sup_col
        df51 = df.dropna(subset=[dc, grp_col]).sort_values(dc)
        df51["prev_order"] = df51.groupby(grp_col)[dc].shift(1)
        df51["cycle_days"] = (df51[dc] - df51["prev_order"]).dt.days.clip(0,365)
        cycle = df51["cycle_days"].dropna()
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(cycle, bins=50, color=ACCENT)
        ax.axvline(cycle.median(), color="red", ls="--", lw=1.5, label=f"Median={cycle.median():.0f}d")
        ax.set_xlabel("Days Between Orders"); ax.set_title("PO Cycle Time Distribution"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(52, "Inventory Value Burn-Rate Over Time",
            "Rolling 30-day sum of inventory value consumed. "
            "Acceleration = demand surge; deceleration = stockpiling or demand drop.")
    if dc and val_col:
        burn = df.set_index(dc)[val_col].resample("D").sum().fillna(0)
        roll30 = burn.rolling(30).sum()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.fill_between(burn.index, burn.values, alpha=0.25, color=ACCENT)
        ax.plot(roll30.index, roll30.values, color="#f97316", lw=2, label="30D Rolling Burn")
        ax.set_title("Inventory Value Burn Rate (30D Rolling)"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  USERS  (analyses 53-62)
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice == "Users":
    st.title("👤 Users — 10 Hardcore Analyses")
    sel, _ = pick_table("Users")
    if not sel: st.warning("No tables."); st.stop()
    with st.spinner(): df = load_table("Users", sel)
    st.success(f"{len(df):,} rows · Table: {sel}")
    dc = best_date_col(df); nc = num_cols(df); cc = cat_cols(df)
    role_col = next((c for c in cc if "role" in c.lower()), cc[0] if cc else None)
    uid_col  = next((c for c in df.columns if "user" in c.lower() and "id" in c.lower()), best_id_col(df))

    section(53, "User Activity Entropy — Who is the Most Unpredictable?",
            "Shannon entropy of action types per user. High entropy = diverse, unpredictable activity. "
            "Low entropy = hyper-specialised. Outlier entropy users need auditing.")
    action_col = next((c for c in cc if any(w in c.lower() for w in ["action","event","type","activity"])), status_col := cc[0] if cc else None)
    if uid_col and action_col:
        ent = df.dropna(subset=[uid_col, action_col]).groupby(uid_col)[action_col].apply(
            lambda s: shannon_entropy(s)).reset_index(name="entropy").sort_values("entropy", ascending=False)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(range(len(ent)), ent["entropy"].values, color=ACCENT, lw=1.5)
        ax.fill_between(range(len(ent)), ent["entropy"].values, alpha=0.2, color=ACCENT)
        ax.axhline(ent["entropy"].mean(), color="red", ls="--", lw=1, label=f"Mean={ent['entropy'].mean():.2f}")
        ax.set_xlabel("User Rank (by Entropy)"); ax.set_ylabel("Shannon Entropy")
        ax.set_title("User Action Entropy Distribution"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(54, "Role Distribution & Permission Coverage",
            "How many users per role? Unbalanced role distribution concentrates risk. "
            "A role with 1 user is a single point of failure.")
    if role_col:
        rc = df[role_col].value_counts()
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        axes[0].bar(rc.index.astype(str), rc.values, color=ACCENT)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
        axes[0].set_title("Users per Role")
        axes[1].pie(rc.values, labels=[str(x)[:15] for x in rc.index], autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Role Distribution Share")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(55, "Login Session Heatmap — Hour × Day",
            "When are users most active? Off-hours activity (midnight logins) is a security signal.")
    if dc:
        df55 = df.dropna(subset=[dc]).copy()
        df55["hour"] = df55[dc].dt.hour; df55["dow"] = df55[dc].dt.day_name()
        dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hm = df55.groupby(["hour","dow"]).size().unstack(fill_value=0)
        hm = hm.reindex(columns=[d for d in dow if d in hm.columns])
        fig, ax = plt.subplots(figsize=(12,6))
        im = ax.imshow(hm.values, aspect="auto", cmap="Blues")
        ax.set_yticks(range(24)); ax.set_yticklabels(range(24), fontsize=7)
        ax.set_xticks(range(len(hm.columns))); ax.set_xticklabels(hm.columns)
        ax.set_title("User Login Heatmap — Hour × Day"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(56, "User Tenure Cohort Activity Analysis",
            "Users grouped by account creation cohort. Do newer cohorts engage less? "
            "Declining activity in young cohorts = onboarding failure.")
    create_col = next((c for c in df.columns if "creat" in c.lower() and ("at" in c.lower() or "date" in c.lower())), None)
    if create_col and dc and uid_col:
        df56 = df.dropna(subset=[create_col, dc, uid_col]).copy()
        df56["cohort"] = pd.to_datetime(df56[create_col], utc=True, errors="coerce").dt.to_period("M")
        df56["active_month"] = df56[dc].dt.to_period("M")
        df56["offset"] = (df56["active_month"] - df56["cohort"]).apply(lambda x: x.n)
        pivot = df56.pivot_table(index="cohort", columns="offset", values=uid_col,
                                  aggfunc="nunique", fill_value=0)
        fig, ax = plt.subplots(figsize=(12,5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(pivot.shape[1])); ax.set_xticklabels(pivot.columns, fontsize=7)
        ax.set_yticks(range(pivot.shape[0])); ax.set_yticklabels([str(c) for c in pivot.index], fontsize=7)
        ax.set_title("User Cohort Activity (Unique Active Users by Month Offset)"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(57, "Department Load Imbalance — Lorenz Curve",
            "Are users distributed evenly across departments? "
            "High Gini = one department hoards staff; others starved.")
    dept_col = next((c for c in cc if any(w in c.lower() for w in ["dept","department","unit","branch"])), role_col)
    if dept_col:
        dept_counts = df[dept_col].value_counts()
        G = gini(dept_counts.values)
        cum = np.cumsum(np.sort(dept_counts.values)) / dept_counts.sum()
        x = np.linspace(0,1,len(cum))
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(x, cum, color=ACCENT, lw=2, label=f"Lorenz (Gini={G:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1,label="Equal")
        ax.fill_between(x, cum, x, alpha=0.15, color=ACCENT)
        ax.set_title("Department Staff Concentration Lorenz Curve"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(58, "User Churn Signals — Declining Activity Detection",
            "Computes each user's activity in last 30d vs prior 30d. "
            "Users with >50% decline are churn risks who need re-engagement.")
    if dc and uid_col:
        now = pd.Timestamp.now(tz="UTC")
        df58 = df.dropna(subset=[dc, uid_col]).copy()
        last30 = df58[df58[dc] >= now - pd.Timedelta("30D")].groupby(uid_col).size().rename("last30")
        prev30 = df58[(df58[dc] >= now - pd.Timedelta("60D")) & (df58[dc] < now - pd.Timedelta("30D"))].groupby(uid_col).size().rename("prev30")
        churn = pd.concat([last30,prev30],axis=1).fillna(0)
        churn["change_pct"] = ((churn["last30"] - churn["prev30"]) / churn["prev30"].replace(0,np.nan) * 100).round(1)
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(churn["change_pct"].dropna(), bins=40, color=ACCENT)
        ax.axvline(0, color="red", ls="--", lw=1.5, label="No Change")
        ax.axvline(-50, color="orange", ls="--", lw=1, label="-50% (churn risk)")
        ax.set_xlabel("Activity Change %"); ax.set_title("User Activity Change Distribution (30D vs Prior 30D)"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(59, "Off-Hours Access Detection",
            "Logins outside 08:00-18:00 are flagged. Volume and user count of off-hours access. "
            "High off-hours = potential security risk or shift workers to plan for.")
    if dc and uid_col:
        df59 = df.dropna(subset=[dc, uid_col]).copy()
        df59["hour"] = df59[dc].dt.hour
        df59["off_hours"] = ~df59["hour"].between(8,17)
        oh = df59.groupby(uid_col)["off_hours"].agg(["sum","count"]).reset_index()
        oh["off_pct"] = (oh["sum"]/oh["count"]*100).round(1)
        oh = oh[oh["off_pct"]>0].sort_values("off_pct",ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(oh[uid_col].astype(str), oh["off_pct"], color=["#ef4444" if x>50 else ACCENT for x in oh["off_pct"]])
        ax.set_xlabel("% of Logins Off-Hours"); ax.set_title("Top 20 Users by Off-Hours Access Rate")
        ax.invert_yaxis(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(60, "User Action Sequence — Most Common Trigrams",
            "Most frequent three-step action sequences per user. "
            "Reveals workflow patterns — and anti-patterns — embedded in daily usage.")
    if uid_col and action_col:
        df60 = df.dropna(subset=[uid_col, action_col]).sort_values(dc if dc else action_col)
        trigrams = []
        for _, grp in df60.groupby(uid_col)[action_col]:
            acts = grp.tolist()
            trigrams += list(zip(acts[:-2], acts[1:-1], acts[2:]))
        top_tri = pd.Series(Counter(trigrams)).nlargest(20).reset_index()
        top_tri.columns = ["trigram","count"]
        top_tri["trigram"] = top_tri["trigram"].apply(lambda t: " → ".join(str(x)[:10] for x in t))
        fig, ax = plt.subplots(figsize=(11,6))
        ax.barh(top_tri["trigram"], top_tri["count"], color=ACCENT)
        ax.set_title("Top 20 Action Trigrams (3-Step Sequences)"); ax.invert_yaxis()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(61, "Peak Usage Hour by Role",
            "For each role, at what hour are users most active? "
            "Misaligned peaks vs service demand = resource scheduling gap.")
    if role_col and dc:
        df61 = df.dropna(subset=[role_col, dc]).copy()
        df61["hour"] = df61[dc].dt.hour
        role_peak = df61.groupby([role_col,"hour"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(13,max(5,len(role_peak)*0.5+2)))
        im = ax.imshow(role_peak.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(24)); ax.set_xticklabels(range(24), fontsize=7)
        ax.set_yticks(range(len(role_peak))); ax.set_yticklabels([str(r)[:20] for r in role_peak.index], fontsize=7)
        ax.set_title("Activity by Role × Hour of Day"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(62, "User Growth Velocity & Retention Curve",
            "Cumulative new users over time. Slope = growth velocity. "
            "Flattening curve = acquisition slowdown or user base saturation.")
    create_col2 = next((c for c in df.columns if "creat" in c.lower()), dc)
    if create_col2:
        ts62 = df.dropna(subset=[create_col2]).set_index(create_col2).resample("W").size().cumsum()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(ts62.index, ts62.values, color=ACCENT, lw=2)
        ax.fill_between(ts62.index, ts62.values, alpha=0.2, color=ACCENT)
        ax.set_title("Cumulative User Growth (Weekly)"); ax.set_ylabel("Total Users")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION  (analyses 63-72)
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice == "Evaluation":
    st.title("📋 Evaluation — 10 Hardcore Analyses")
    sel, _ = pick_table("Evaluation")
    if not sel: st.warning("No tables."); st.stop()
    with st.spinner(): df = load_table("Evaluation", sel)
    st.success(f"{len(df):,} rows · Table: {sel}")
    dc = best_date_col(df); nc = num_cols(df); cc = cat_cols(df)
    score_col = next((c for c in nc if any(w in c.lower() for w in ["score","grade","mark","rating","result"])), nc[0] if nc else None)
    eval_col  = next((c for c in cc if any(w in c.lower() for w in ["evaluator","assessor","reviewer","rater"])), cc[0] if cc else None)
    eval_ee   = next((c for c in cc if any(w in c.lower() for w in ["staff","employee","doctor","student","subject"])), cc[1] if len(cc)>1 else None)

    section(63, "Evaluator Bias Detection — Score Distribution by Evaluator",
            "Box plot of scores given by each evaluator. "
            "Evaluators whose median deviates >1σ from the global mean are systematically biased.")
    if eval_col and score_col:
        global_mean = df[score_col].mean(); global_std = df[score_col].std()
        top_evals = df[eval_col].value_counts().index[:12]
        sub = df[df[eval_col].isin(top_evals)]
        fig, ax = plt.subplots(figsize=(12,5))
        groups = [sub[sub[eval_col]==e][score_col].dropna().values for e in top_evals]
        bp = ax.boxplot(groups, patch_artist=True, notch=True)
        for patch,e in zip(bp["boxes"],top_evals):
            m = sub[sub[eval_col]==e][score_col].median()
            patch.set_facecolor("#ef4444" if abs(m-global_mean)>global_std else ACCENT)
            patch.set_alpha(0.7)
        ax.axhline(global_mean, color="black", ls="--", lw=1.5, label=f"Global Mean={global_mean:.2f}")
        ax.axhline(global_mean+global_std, color="orange", ls=":", lw=1)
        ax.axhline(global_mean-global_std, color="orange", ls=":", lw=1)
        ax.set_xticks(range(1,len(top_evals)+1))
        ax.set_xticklabels([str(e)[:12] for e in top_evals], rotation=30, ha="right")
        ax.set_title("Score Distribution by Evaluator (Red = Biased)"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(64, "Grade Inflation/Deflation Trend Over Time",
            "Rolling average score over time per evaluator. "
            "Rising trend = grade inflation; falling = increasing rigor — or evaluator fatigue.")
    if dc and score_col and eval_col:
        df64 = df.dropna(subset=[dc, score_col, eval_col]).set_index(dc).sort_index()
        top5 = df64[eval_col].value_counts().index[:5]
        fig, ax = plt.subplots(figsize=(12,5))
        for e in top5:
            sub = df64[df64[eval_col]==e][score_col].resample("M").mean()
            ax.plot(sub.index, sub.values, lw=2, marker="o", ms=4, label=str(e)[:15])
        ax.set_title("Monthly Average Score by Top 5 Evaluators"); ax.legend(fontsize=8)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(65, "Performance Trajectory Clustering — Improving vs Declining",
            "Computes slope of score over time per individual. "
            "Positive slope = improving; negative = declining. Clusters staff into trajectories.")
    if dc and score_col and eval_ee:
        df65 = df.dropna(subset=[dc, score_col, eval_ee]).copy()
        df65["t"] = (df65[dc] - df65[dc].min()).dt.days.astype(float)
        slopes = df65.groupby(eval_ee).apply(
            lambda g: np.polyfit(g["t"], g[score_col], 1)[0] if len(g) >= 2 else np.nan
        ).reset_index(name="slope").dropna()
        slopes["trajectory"] = slopes["slope"].apply(
            lambda x: "📈 Improving" if x>0.01 else "📉 Declining" if x<-0.01 else "➡️ Stable"
        )
        traj_counts = slopes["trajectory"].value_counts()
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        axes[0].hist(slopes["slope"], bins=30, color=ACCENT)
        axes[0].axvline(0, color="red", ls="--", lw=1.5)
        axes[0].set_title("Slope Distribution (Performance Trajectory)"); axes[0].set_xlabel("Score Slope")
        axes[1].pie(traj_counts.values, labels=traj_counts.index, autopct="%1.0f%%", startangle=90)
        axes[1].set_title("Trajectory Breakdown")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(66, "Evaluation Cycle Time — Lag from Event to Assessment",
            "If an event date and evaluation date exist: distribution of time-to-evaluate. "
            "Long lags = stale feedback; feedback loses impact after 72 hours.")
    event_col = next((c for c in df.columns if "event" in c.lower() or "incident" in c.lower() or "case" in c.lower()), None)
    if dc and event_col:
        ev = pd.to_datetime(df[event_col], utc=True, errors="coerce")
        lag = (df[dc] - ev).dt.days.clip(0,365).dropna()
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(lag, bins=50, color=ACCENT)
        ax.axvline(3, color="red", ls="--", lw=1, label="3d SLA")
        ax.axvline(lag.median(), color="orange", ls="--", lw=1, label=f"Median={lag.median():.0f}d")
        ax.set_title("Evaluation Cycle Time Distribution"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    elif score_col and dc:
        ts66 = df.dropna(subset=[dc,score_col]).set_index(dc).resample("M")[score_col].count()
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(ts66.index.astype(str), ts66.values, color=ACCENT)
        ax.set_title("Monthly Evaluation Count"); ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(67, "Weighted Multi-Dimensional KPI Score",
            "Composite score = weighted sum of all numeric evaluation dimensions. "
            "Rank staff by composite — a single number hiding a multidimensional truth.")
    if len(nc) >= 2:
        weights = {c: 1.0/len(nc) for c in nc}
        kpi = df[nc].copy()
        for c in nc: kpi[c] = (kpi[c] - kpi[c].min()) / (kpi[c].max() - kpi[c].min() + 1e-9)
        kpi["composite"] = sum(kpi[c]*w for c,w in weights.items())
        if eval_ee and eval_ee in df.columns:
            kpi[eval_ee] = df[eval_ee]
            ranked = kpi.groupby(eval_ee)["composite"].mean().sort_values(ascending=False).reset_index().head(25)
            fig, ax = plt.subplots(figsize=(10,6))
            ax.barh(ranked[eval_ee].astype(str), ranked["composite"], color=ACCENT)
            ax.set_title("Composite KPI Score Ranking (Top 25)"); ax.invert_yaxis()
            fig.tight_layout(); fig_to_st(fig)
        else:
            fig, ax = plt.subplots(figsize=(9,4))
            ax.hist(kpi["composite"].dropna(), bins=40, color=ACCENT)
            ax.set_title("Composite KPI Score Distribution"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(68, "Evaluation Completeness Heatmap",
            "Cross-matrix of evaluator × evaluatee. Missing cells = unevaluated relationships. "
            "Dark spots = over-evaluated pairs; white = gaps in coverage.")
    if eval_col and eval_ee:
        ct = pd.crosstab(df[eval_col], df[eval_ee])
        ct = ct.iloc[:20,:20]
        fig, ax = plt.subplots(figsize=(14,7))
        im = ax.imshow(ct.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(ct.columns))); ax.set_xticklabels([str(c)[:10] for c in ct.columns], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(ct.index)));   ax.set_yticklabels([str(i)[:15] for i in ct.index], fontsize=7)
        ax.set_title("Evaluation Coverage Matrix (Evaluator × Subject)"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(69, "Department Performance Benchmark",
            "Median score per department with 25th-75th percentile band. "
            "Departments below the lower quartile have a systemic performance problem.")
    dept_col = next((c for c in cc if any(w in c.lower() for w in ["dept","department","unit","section"])), cc[-1] if cc else None)
    if dept_col and score_col:
        depts = df.groupby(dept_col)[score_col].agg(["median","mean",
            lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]).reset_index()
        depts.columns = [dept_col,"median","mean","q25","q75"]
        depts = depts.sort_values("median", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(11,5))
        x = range(len(depts))
        ax.bar(x, depts["median"], color=ACCENT, label="Median", alpha=0.85)
        ax.errorbar(x, depts["median"],
                    yerr=[depts["median"]-depts["q25"], depts["q75"]-depts["median"]],
                    fmt="none", color="black", capsize=3, lw=1)
        ax.set_xticks(x); ax.set_xticklabels([str(d)[:12] for d in depts[dept_col]], rotation=30, ha="right")
        ax.axhline(df[score_col].median(), color="red", ls="--", lw=1, label="Global Median")
        ax.set_title("Department Performance Benchmark"); ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(70, "Outlier Evaluator Statistical Test",
            "One-sample t-test per evaluator: is their mean score significantly different from global mean? "
            "p<0.05 = statistically biased evaluator.")
    if eval_col and score_col:
        global_mean = df[score_col].mean()
        results = []
        for e, grp in df.groupby(eval_col)[score_col]:
            if len(grp) >= 5:
                t, p = stats.ttest_1samp(grp.dropna(), global_mean)
                results.append({"evaluator": e, "n": len(grp), "mean": grp.mean(), "t": t, "p": p})
        if results:
            res = pd.DataFrame(results).sort_values("p")
            res["significant"] = res["p"] < 0.05
            fig, ax = plt.subplots(figsize=(10,5))
            colors = ["#ef4444" if sig else ACCENT for sig in res["significant"]]
            ax.scatter(res["mean"] - global_mean, -np.log10(res["p"]+1e-10), c=colors, s=60, alpha=0.8)
            ax.axhline(-np.log10(0.05), color="red", ls="--", lw=1, label="p=0.05")
            ax.axvline(0, color="gray", ls=":", lw=1)
            ax.set_xlabel("Mean Score Deviation from Global"); ax.set_ylabel("-log10(p)")
            ax.set_title("Evaluator Bias Volcano Plot"); ax.legend()
            fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(71, "Evaluation Frequency vs Performance Correlation",
            "Do more frequently evaluated staff perform better? "
            "Negative correlation = evaluation burden; positive = coaching effect.")
    if eval_ee and score_col:
        freq_score = df.groupby(eval_ee).agg(freq=(score_col,"count"), avg_score=(score_col,"mean")).reset_index()
        r, p = stats.pearsonr(freq_score["freq"], freq_score["avg_score"])
        fig, ax = plt.subplots(figsize=(8,5))
        ax.scatter(freq_score["freq"], freq_score["avg_score"], alpha=0.6, color=ACCENT, s=40)
        m, b = np.polyfit(freq_score["freq"], freq_score["avg_score"], 1)
        ax.plot(sorted(freq_score["freq"]), [m*x+b for x in sorted(freq_score["freq"])],
                color="#f97316", lw=2, label=f"r={r:.3f}, p={p:.3f}")
        ax.set_xlabel("Evaluation Frequency"); ax.set_ylabel("Avg Score")
        ax.set_title("Evaluation Frequency vs Performance (Pearson Correlation)"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(72, "Score Distribution Shift — Period over Period",
            "KDE comparison of score distributions in two time periods. "
            "Shift in the curve reveals systemic change in evaluation standards.")
    if dc and score_col:
        df72 = df.dropna(subset=[dc, score_col]).sort_values(dc)
        mid = df72[dc].quantile(0.5)
        h1 = df72[df72[dc] < mid][score_col].dropna()
        h2 = df72[df72[dc] >= mid][score_col].dropna()
        ks_stat, ks_p = stats.ks_2samp(h1, h2)
        fig, ax = plt.subplots(figsize=(9,4))
        for s, l, c in [(h1,"Period 1",ACCENT),(h2,"Period 2","#f97316")]:
            s.plot.kde(ax=ax, label=l, color=c, lw=2)
        ax.set_title(f"Score KDE — Period Comparison  (KS p={ks_p:.4f} {'📌 SHIFT DETECTED' if ks_p<0.05 else '✅ stable'})")
        ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTS / CORE / SETTINGS — generic deep analytics
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice in ("Reports", "Core", "Settings"):
    st.title(f"🔧 {module_choice} — Deep Data Analytics")
    sel, _ = pick_table(module_choice)
    if not sel: st.warning("No tables."); st.stop()
    with st.spinner(): df = load_table(module_choice, sel)
    st.success(f"{len(df):,} rows · Table: {sel}")
    dc = best_date_col(df); nc = num_cols(df); cc = cat_cols(df)

    section(73, "Entity Creation Velocity & Growth Rate",
            "Daily creation rate with EWMA. Acceleration = rapid adoption or data migration. "
            "Deceleration = saturation or process slowdown.")
    if dc:
        ts = df.set_index(dc).resample("D").size().fillna(0)
        ewma = ts.ewm(span=7).mean()
        growth = ts.pct_change().rolling(7).mean() * 100
        fig, (ax1,ax2) = plt.subplots(2,1,figsize=(13,7),sharex=True)
        ax1.fill_between(ts.index, ts.values, alpha=0.3, color=ACCENT)
        ax1.plot(ewma.index, ewma.values, color="#f97316", lw=2, label="7D EWMA")
        ax1.set_title("Entity Creation Volume"); ax1.legend()
        ax2.fill_between(growth.index, growth.values, 0, where=growth>0, alpha=0.5, color="#10b981", label="Growth")
        ax2.fill_between(growth.index, growth.values, 0, where=growth<=0, alpha=0.5, color="#ef4444", label="Decline")
        ax2.set_title("7D Rolling Growth Rate (%)"); ax2.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(74, "Data Completeness Score per Column",
            "% of non-null values per column. Columns below 70% are unreliable. "
            "Sorted to instantly identify the most incomplete fields.")
    completeness = (df.notna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(5, len(completeness)*0.3)))
    colors = ["#ef4444" if v<50 else "#f97316" if v<80 else "#10b981" for v in completeness.values]
    ax.barh(completeness.index, completeness.values, color=colors)
    ax.axvline(70, color="orange", ls="--", lw=1.5, label="70% threshold")
    ax.axvline(100, color="green", ls=":", lw=1)
    ax.set_xlabel("% Complete"); ax.set_title("Column Completeness Score"); ax.legend()
    fig.tight_layout(); fig_to_st(fig)

    section(75, "Cardinality vs Completeness Scatter",
            "Each column plotted by unique value count vs completeness. "
            "High cardinality + low completeness = problematic free-text field.")
    card = pd.DataFrame({
        "column": df.columns,
        "cardinality": [df[c].nunique() for c in df.columns],
        "completeness": [df[c].notna().mean()*100 for c in df.columns],
    })
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(card["cardinality"], card["completeness"], c=range(len(card)), cmap=CMAP, s=60, alpha=0.8)
    for _, row in card.iterrows():
        ax.annotate(row["column"][:12], (row["cardinality"], row["completeness"]),
                    textcoords="offset points", xytext=(4,4), fontsize=6)
    ax.set_xlabel("Cardinality (unique values)"); ax.set_ylabel("Completeness (%)")
    ax.set_title("Column Cardinality vs Completeness"); plt.colorbar(sc, ax=ax)
    fig.tight_layout(); fig_to_st(fig)

    section(76, "Multi-Variable Correlation Heatmap",
            "Full Pearson correlation matrix on all numeric columns. "
            "Perfect correlations (±1.0) are collinear redundancies; strong ones are prediction signals.")
    if len(nc) >= 2:
        corr = df[nc].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(max(7,len(nc)), max(6,len(nc)-1)))
        im = ax.imshow(np.ma.array(corr.values, mask=mask), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(nc))); ax.set_xticklabels(nc, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(nc))); ax.set_yticklabels(nc, fontsize=8)
        for i in range(len(nc)):
            for j in range(i):
                ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=7)
        plt.colorbar(im, ax=ax); ax.set_title("Numeric Correlation Heatmap (Lower Triangle)")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(77, "Category Distribution — Top 5 Columns",
            "Horizontal bar charts for the top 5 categorical columns. "
            "Skewed distributions reveal dominant categories that bias all aggregate stats.")
    cat5 = cc[:5] if cc else []
    if cat5:
        fig, axes = plt.subplots(1, len(cat5), figsize=(14, 5))
        if len(cat5) == 1: axes = [axes]
        for ax, col in zip(axes, cat5):
            vc = df[col].value_counts().head(10)
            ax.barh(vc.index.astype(str), vc.values, color=ACCENT)
            ax.set_title(col, fontsize=9); ax.invert_yaxis()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(78, "Numeric Percentile Profile (Full Distribution)",
            "For every numeric column: P5, P25, P50, P75, P95 plotted together. "
            "Compares the full distributional shape across all fields in one view.")
    if nc:
        percs = df[nc].quantile([0.05,0.25,0.5,0.75,0.95])
        norm = (percs - percs.min()) / (percs.max() - percs.min() + 1e-9)
        fig, ax = plt.subplots(figsize=(12,5))
        for i,(idx,row) in enumerate(norm.iterrows()):
            ax.plot(range(len(nc)), row.values, marker="o", ms=5, lw=1.5,
                    label=f"P{int(idx*100)}")
        ax.set_xticks(range(len(nc))); ax.set_xticklabels(nc, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Normalised Value"); ax.set_title("Percentile Profile Across Numeric Columns")
        ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(79, "Time-Series Decomposition — Trend + Residual",
            "Rolling median as trend; raw minus trend as residual. "
            "Structural breaks in the residual signal operational anomalies.")
    if dc:
        ts79 = df.set_index(dc).resample("D").size().fillna(0)
        if len(ts79) > 14:
            trend = ts79.rolling(7, center=True).median()
            residual = ts79 - trend
            fig, axes = plt.subplots(3,1,figsize=(13,9),sharex=True)
            axes[0].plot(ts79.index, ts79.values, color=ACCENT, lw=1, label="Raw"); axes[0].set_title("Raw"); axes[0].legend()
            axes[1].plot(trend.index, trend.values, color="#f97316", lw=2, label="Trend (7D Median)"); axes[1].set_title("Trend"); axes[1].legend()
            axes[2].fill_between(residual.index, residual.values, 0, where=residual>0, alpha=0.5, color="#10b981")
            axes[2].fill_between(residual.index, residual.values, 0, where=residual<=0, alpha=0.5, color="#ef4444")
            axes[2].set_title("Residual"); fig.tight_layout(); fig_to_st(fig)
        else: no_data()
    else: no_data()

    section(80, "Record Duplication Probability (Fuzzy)",
            "Identifies likely duplicate records based on exact matching of top categorical columns. "
            "High duplication rate = master data quality failure.")
    if cc:
        key_cols = cc[:min(3,len(cc))]
        dup_df = df[key_cols].dropna()
        dup_count = dup_df.duplicated().sum()
        dup_groups = dup_df[dup_df.duplicated(keep=False)].groupby(key_cols).size().reset_index(name="count")
        dup_groups = dup_groups.sort_values("count",ascending=False)
        st.metric("Duplicate Records Found", f"{dup_count:,}", delta=f"{dup_count/len(df)*100:.1f}% of dataset", delta_color="inverse")
        st.dataframe(safe_show(dup_groups), use_container_width=True)
    else: no_data()


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-MODULE  (analyses 81-100)
# ══════════════════════════════════════════════════════════════════════════════

elif module_choice == "🔀 Cross-Module":
    st.title("🔀 Cross-Module — 20 System-Wide Intelligence Analyses")
    st.info("Loading all modules. This may take a minute on first run — data is cached afterwards.")

    @st.cache_data(show_spinner=True)
    def load_all():
        out = {}
        for mod in ALL_MODULES:
            out[mod] = {}
            for tbl in table_map.get(mod,[]):
                d = load_table(mod, tbl)
                if not d.empty: out[mod][tbl] = d
        return out

    all_data = load_all()

    def first_df(module):
        dfs = list(all_data.get(module,{}).values())
        return dfs[0] if dfs else pd.DataFrame()

    fin  = first_df("Finance")
    inp  = first_df("Inpatient")
    rec  = first_df("Reception")
    thr  = first_df("Theatre")
    inv  = first_df("Inventory")
    usr  = first_df("Users")
    ev   = first_df("Evaluation")

    section(81, "Module Data Volume Comparison",
            "Total row count per module × table. Reveals which modules are data-heavy vs starved. "
            "Imbalance suggests logging gaps or integration failures.")
    rows = []
    for mod, tbls in all_data.items():
        for tbl, df2 in tbls.items():
            rows.append({"module":mod,"table":tbl,"rows":len(df2),"cols":len(df2.columns)})
    vol = pd.DataFrame(rows)
    if not vol.empty:
        fig, ax = plt.subplots(figsize=(13,6))
        pivot_vol = vol.pivot_table(index="module", columns="table", values="rows", aggfunc="sum", fill_value=0)
        pivot_vol.plot(kind="bar", stacked=True, ax=ax, colormap=CMAP)
        ax.set_title("Data Volume per Module × Table (Stacked)"); ax.legend(fontsize=6, loc="upper right")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(82, "Column Coverage Heatmap Across All Modules",
            "Which column names appear across multiple modules? "
            "Shared columns (e.g., patient_id) are the join keys for cross-module analytics.")
    all_cols = {}
    for mod, tbls in all_data.items():
        for tbl, df2 in tbls.items():
            for col in df2.columns:
                all_cols.setdefault(col,[]).append(f"{mod}")
    col_presence = pd.DataFrame({
        "column": list(all_cols.keys()),
        "module_count": [len(set(v)) for v in all_cols.values()],
        "modules": [", ".join(set(v)) for v in all_cols.values()],
    }).sort_values("module_count", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.barh(col_presence["column"], col_presence["module_count"], color=ACCENT)
    ax.set_xlabel("Number of Modules Containing This Column")
    ax.set_title("Cross-Module Column Coverage (Top 40 columns)"); ax.invert_yaxis()
    fig.tight_layout(); fig_to_st(fig)

    section(83, "Data Freshness Index — Latest Record Date per Module",
            "How recent is each module's data? Stale modules indicate broken pipelines. "
            "A module that stopped receiving events 7 days ago is likely broken.")
    freshness = []
    for mod, tbls in all_data.items():
        for tbl, df2 in tbls.items():
            dc2 = best_date_col(df2)
            if dc2:
                last = df2[dc2].max()
                if pd.notna(last):
                    freshness.append({"module":mod,"table":tbl,"latest_record":last,
                                       "days_stale":(pd.Timestamp.now(tz="UTC")-last).days})
    if freshness:
        fr = pd.DataFrame(freshness).sort_values("days_stale")
        fig, ax = plt.subplots(figsize=(11,max(5,len(fr)*0.4)))
        colors = ["#ef4444" if d>7 else "#f97316" if d>3 else "#10b981" for d in fr["days_stale"]]
        ax.barh(fr["module"]+"/"+fr["table"], fr["days_stale"], color=colors)
        ax.set_xlabel("Days Since Latest Record"); ax.set_title("Data Freshness per Module/Table (Red = Stale)")
        ax.invert_yaxis(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(84, "Cross-Module Activity Correlation Matrix",
            "Daily event counts per module correlated with each other. "
            "High correlation = modules move together (same patients drive both). "
            "Low correlation = modules are operationally independent.")
    daily_series = {}
    for mod, tbls in all_data.items():
        series_list = []
        for tbl, df2 in tbls.items():
            dc2 = best_date_col(df2)
            if dc2:
                ts = df2.set_index(dc2).resample("D").size()
                series_list.append(ts)
        if series_list:
            combined = pd.concat(series_list, axis=1).sum(axis=1)
            daily_series[mod] = combined
    if len(daily_series) >= 2:
        cross_df = pd.DataFrame(daily_series).fillna(0)
        corr = cross_df.corr()
        fig, ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.index)
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=8,
                        color="white" if abs(corr.values[i,j])>0.5 else "black")
        ax.set_title("Cross-Module Daily Activity Correlation"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(85, "Multi-Module Throughput Stacked Timeline",
            "All modules' daily event volumes stacked over time. "
            "Reveals system-wide activity patterns and coordinated dips (holidays, outages).")
    if daily_series:
        stacked = pd.DataFrame(daily_series).fillna(0).resample("W").sum()
        fig, ax = plt.subplots(figsize=(14,6))
        stacked.plot(kind="area", stacked=True, ax=ax, colormap=CMAP, alpha=0.8)
        ax.set_title("Weekly Event Volume — All Modules (Stacked Area)"); ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(86, "Finance × Inpatient — Billing Lag Analysis",
            "Difference between first inpatient event date and first Finance invoice date per shared entity. "
            "Long lag = delayed billing = cash flow risk.")
    inp_dc = best_date_col(inp); fin_dc = best_date_col(fin)
    inp_id = next((c for c in inp.columns if "patient" in c.lower() and "id" in c.lower()), None)
    fin_id = next((c for c in fin.columns if "patient" in c.lower() and "id" in c.lower()), None)
    if not inp.empty and not fin.empty and inp_id and fin_id and inp_dc and fin_dc:
        inp_first = inp.groupby(inp_id)[inp_dc].min().reset_index().rename(columns={inp_dc:"admit_date",inp_id:"pid"})
        fin_first = fin.groupby(fin_id)[fin_dc].min().reset_index().rename(columns={fin_dc:"invoice_date",fin_id:"pid"})
        merged_bf = inp_first.merge(fin_first, on="pid")
        merged_bf["billing_lag"] = (merged_bf["invoice_date"] - merged_bf["admit_date"]).dt.days.clip(-10,180)
        lag_data = merged_bf["billing_lag"].dropna()
        fig, ax = plt.subplots(figsize=(9,4))
        ax.hist(lag_data, bins=40, color=ACCENT)
        ax.axvline(0, color="red", ls="--", lw=1.5, label="Day 0 (admit)")
        ax.axvline(lag_data.median(), color="orange", ls="--", lw=1, label=f"Median={lag_data.median():.0f}d")
        ax.set_xlabel("Days from Admission to Invoice"); ax.set_title("Finance-Inpatient Billing Lag"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data("Matching patient_id columns not found in Finance + Inpatient.")

    section(87, "Inventory × Finance — Cost Absorption Analysis",
            "Compares total inventory value consumed vs total invoiced revenue per month. "
            "Gap = unbilled consumed goods or billing discrepancy.")
    if not inv.empty and not fin.empty:
        inv_dc = best_date_col(inv); fin_dc2 = best_date_col(fin)
        inv_val = best_amount_col(inv); fin_amt = best_amount_col(fin)
        if inv_dc and fin_dc2 and inv_val and fin_amt:
            inv_mo = inv.set_index(inv_dc)[inv_val].resample("M").sum().rename("inventory_cost")
            fin_mo = fin.set_index(fin_dc2)[fin_amt].resample("M").sum().rename("invoiced_revenue")
            combined = pd.concat([inv_mo, fin_mo], axis=1).fillna(0)
            combined["gap"] = combined["invoiced_revenue"] - combined["inventory_cost"]
            fig, ax = plt.subplots(figsize=(13,5))
            ax.fill_between(combined.index, combined["invoiced_revenue"], label="Invoiced", alpha=0.6, color=ACCENT)
            ax.fill_between(combined.index, combined["inventory_cost"], label="Inventory Cost", alpha=0.6, color="#f97316")
            ax2 = ax.twinx()
            ax2.plot(combined.index, combined["gap"], color="#10b981", lw=2, ls="--", label="Gap")
            ax.set_title("Inventory Cost vs Invoiced Revenue (Monthly)"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
            fig.tight_layout(); fig_to_st(fig)
        else: no_data()
    else: no_data()

    section(88, "Users × Finance — Staff Revenue Attribution",
            "Revenue generated per active user/staff member using shared date periods. "
            "Reveals which staff periods correlate with highest billing volumes.")
    if not usr.empty and not fin.empty:
        usr_dc = best_date_col(usr); fin_dc3 = best_date_col(fin); fin_amt2 = best_amount_col(fin)
        if usr_dc and fin_dc3 and fin_amt2:
            usr_daily = usr.set_index(usr_dc).resample("D").size().rename("active_users")
            fin_daily = fin.set_index(fin_dc3)[fin_amt2].resample("D").sum().rename("revenue")
            combined2 = pd.concat([usr_daily,fin_daily],axis=1).fillna(0)
            combined2["rev_per_user"] = (combined2["revenue"] / combined2["active_users"].replace(0,np.nan)).round(2)
            r, p = stats.pearsonr(combined2["active_users"].dropna(), combined2["revenue"].dropna())
            fig, ax = plt.subplots(figsize=(9,5))
            ax.scatter(combined2["active_users"], combined2["revenue"], alpha=0.4, color=ACCENT, s=20)
            m,b = np.polyfit(combined2["active_users"].fillna(0), combined2["revenue"].fillna(0), 1)
            x_line = np.linspace(combined2["active_users"].min(), combined2["active_users"].max(), 100)
            ax.plot(x_line, m*x_line+b, color="#f97316", lw=2, label=f"r={r:.3f} p={p:.3f}")
            ax.set_xlabel("Daily Active Users"); ax.set_ylabel("Daily Revenue")
            ax.set_title("User Activity vs Revenue Correlation"); ax.legend()
            fig.tight_layout(); fig_to_st(fig)
        else: no_data()
    else: no_data()

    section(89, "Reception × Finance — Admission-to-Invoice Conversion Funnel",
            "How many Reception registrations end up with a Finance invoice? "
            "The drop-off rate is unbilled throughput — lost revenue opportunity.")
    if not rec.empty and not fin.empty:
        rec_vol = len(rec)
        fin_vol = len(fin)
        funnel = pd.DataFrame({"Stage":["Reception Registrations","Finance Invoices","Gap (Unbilled)"],
                               "Count":[rec_vol, fin_vol, max(0,rec_vol-fin_vol)]})
        fig, ax = plt.subplots(figsize=(8,5))
        colors = [ACCENT,"#10b981","#ef4444"]
        ax.barh(funnel["Stage"], funnel["Count"], color=colors)
        for i,(s,c) in enumerate(zip(funnel["Stage"],funnel["Count"])):
            ax.text(c, i, f" {c:,}", va="center", fontsize=10)
        ax.set_title("Reception → Finance Conversion Funnel"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(90, "Theatre × Finance — Revenue per Theatre Case",
            "If theatre and finance share a common case/patient ID: revenue attributed per theatre record. "
            "Otherwise: joint timeline comparison of theatre volume vs revenue.")
    if not thr.empty and not fin.empty:
        thr_dc = best_date_col(thr); fin_dc4 = best_date_col(fin); fin_amt4 = best_amount_col(fin)
        if thr_dc and fin_dc4 and fin_amt4:
            thr_wk = thr.set_index(thr_dc).resample("W").size().rename("theatre_cases")
            fin_wk = fin.set_index(fin_dc4)[fin_amt4].resample("W").sum().rename("revenue")
            comb = pd.concat([thr_wk,fin_wk],axis=1).fillna(0)
            comb["rev_per_case"] = (comb["revenue"]/comb["theatre_cases"].replace(0,np.nan)).fillna(0)
            fig, ax = plt.subplots(figsize=(13,5))
            ax2 = ax.twinx()
            ax.bar(comb.index, comb["theatre_cases"], width=5, color=ACCENT, alpha=0.5, label="Theatre Cases")
            ax2.plot(comb.index, comb["rev_per_case"], color="#f97316", lw=2, label="Revenue per Case")
            ax.set_title("Theatre Cases vs Revenue per Case (Weekly)"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
            fig.tight_layout(); fig_to_st(fig)
        else: no_data()
    else: no_data()

    section(91, "System-Wide Event Volume Anomaly Detection",
            "Z-score anomaly detection on total cross-module daily event volume. "
            "Days with |Z|>2.5 = system-wide spikes or crashes.")
    if daily_series:
        total = pd.DataFrame(daily_series).fillna(0).sum(axis=1).resample("D").sum()
        z = stats.zscore(total.values)
        anomaly_idx = np.abs(z) > 2.5
        fig, ax = plt.subplots(figsize=(13,4))
        ax.plot(total.index, total.values, color=ACCENT, lw=1.5, label="Total Events")
        ax.scatter(total.index[anomaly_idx], total.values[anomaly_idx],
                   color="#ef4444", s=50, zorder=5, label=f"Anomalies (|Z|>2.5): {anomaly_idx.sum()}")
        ax.set_title("System-Wide Event Volume + Anomaly Detection"); ax.legend()
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(92, "Cross-Module Operational Efficiency Radar",
            "Radar chart comparing 5 operational metrics across Finance, Inpatient, Theatre, Reception, Inventory. "
            "Each axis normalised to 0-1. A lopsided radar reveals system imbalance.")
    metrics = {}
    for mod, dfs_map in all_data.items():
        if not dfs_map: continue
        df_m = next(iter(dfs_map.values()))
        dc_m = best_date_col(df_m); nc_m = num_cols(df_m)
        completeness_score = df_m.notna().mean().mean()
        volume_score = min(1, len(df_m) / 10000)
        freshness_score = 0
        if dc_m:
            last = df_m[dc_m].max()
            if pd.notna(last):
                stale = (pd.Timestamp.now(tz="UTC") - last).days
                freshness_score = max(0, 1 - stale/30)
        numeric_richness = min(1, len(nc_m)/10)
        metrics[mod] = [completeness_score, volume_score, freshness_score, numeric_richness, len(dfs_map)/5]

    if metrics:
        labels = ["Completeness","Volume","Freshness","Numeric Richness","Table Coverage"]
        mods = list(metrics.keys())
        N = len(labels)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
        for mod, vals in metrics.items():
            v = vals + [vals[0]]
            ax.plot(angles, v, lw=1.5, label=mod)
            ax.fill(angles, v, alpha=0.05)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title("Cross-Module Operational Efficiency Radar", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(93, "Finance × Evaluation — Staff Performance vs Revenue",
            "Correlates evaluation scores with Finance billing volume in the same time period. "
            "High-performing staff should correlate with higher revenue throughput.")
    if not ev.empty and not fin.empty:
        ev_dc = best_date_col(ev); fin_dc5 = best_date_col(fin); fin_amt5 = best_amount_col(fin)
        score_col5 = next((c for c in num_cols(ev) if "score" in c.lower() or "grade" in c.lower() or "mark" in c.lower()), num_cols(ev)[0] if num_cols(ev) else None)
        if ev_dc and fin_dc5 and fin_amt5 and score_col5:
            ev_mo = ev.set_index(ev_dc)[score_col5].resample("M").mean().rename("avg_score")
            fin_mo2 = fin.set_index(fin_dc5)[fin_amt5].resample("M").sum().rename("revenue")
            comb5 = pd.concat([ev_mo,fin_mo2],axis=1).dropna()
            if len(comb5)>=3:
                r,p = stats.pearsonr(comb5["avg_score"], comb5["revenue"])
                fig, ax = plt.subplots(figsize=(8,5))
                ax.scatter(comb5["avg_score"], comb5["revenue"], color=ACCENT, s=50, alpha=0.8)
                m,b = np.polyfit(comb5["avg_score"],comb5["revenue"],1)
                ax.plot(sorted(comb5["avg_score"]),[m*x+b for x in sorted(comb5["avg_score"])],
                        color="#f97316",lw=2,label=f"r={r:.3f} p={p:.3f}")
                ax.set_xlabel("Avg Evaluation Score"); ax.set_ylabel("Monthly Revenue")
                ax.set_title("Evaluation Performance vs Revenue"); ax.legend()
                fig.tight_layout(); fig_to_st(fig)
            else: no_data("Not enough overlapping monthly data.")
        else: no_data()
    else: no_data()

    section(94, "All Modules — Weekday vs Weekend Volume Split",
            "For each module: what % of events occur on weekends? "
            "High weekend activity in Finance may indicate backdating or data entry delays.")
    weekend_data = []
    for mod, dfs_map in all_data.items():
        for tbl, df2 in dfs_map.items():
            dc2 = best_date_col(df2)
            if dc2:
                df2 = df2.dropna(subset=[dc2])
                weekend = df2[dc2].dt.dayofweek >= 5
                weekend_data.append({"module":mod,"table":tbl,"weekend_pct":weekend.mean()*100,"total":len(df2)})
    if weekend_data:
        wd = pd.DataFrame(weekend_data).sort_values("weekend_pct",ascending=False)
        fig, ax = plt.subplots(figsize=(11,5))
        ax.barh(wd["module"]+"/"+wd["table"], wd["weekend_pct"], color=[ACCENT if p<25 else "#f97316" if p<40 else "#ef4444" for p in wd["weekend_pct"]])
        ax.axvline(20, color="orange", ls="--", lw=1, label="20% reference")
        ax.set_xlabel("% of Events on Weekends"); ax.set_title("Weekend Activity Share by Module/Table")
        ax.invert_yaxis(); ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(95, "Data Volume Growth Projection — 90-Day Forecast per Module",
            "Linear regression on weekly row counts projected 90 days forward. "
            "Shows which modules are on growth trajectories vs plateauing.")
    if daily_series:
        fig, ax = plt.subplots(figsize=(13,6))
        future = pd.date_range(pd.Timestamp.now(tz="UTC"), periods=90, freq="D")
        for mod, ts in daily_series.items():
            wk = ts.resample("W").sum()
            if len(wk) < 4: continue
            x = np.arange(len(wk))
            m2,b2 = np.polyfit(x, wk.values, 1)
            x_fut = np.arange(len(wk), len(wk)+13)
            ax.plot(wk.index, wk.values, lw=1.5, label=mod)
            fut_dates = pd.date_range(wk.index[-1], periods=14, freq="W")[1:]
            ax.plot(fut_dates, [m2*xi+b2 for xi in x_fut], ls="--", lw=1)
        ax.set_title("Weekly Volume + 90-Day Linear Projection (solid=actual, dashed=forecast)")
        ax.legend(fontsize=7); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(96, "Cross-Module Missing Data Comparison",
            "Which module has the worst data completeness? "
            "Ranked by average % of null fields — the leakiest data pipeline wins worst prize.")
    comp_data = []
    for mod, dfs_map in all_data.items():
        for tbl, df2 in dfs_map.items():
            comp_data.append({"module":mod,"table":tbl,"completeness":df2.notna().mean().mean()*100})
    if comp_data:
        cd = pd.DataFrame(comp_data).sort_values("completeness")
        fig, ax = plt.subplots(figsize=(11,max(5,len(cd)*0.4)))
        colors = ["#ef4444" if v<70 else "#f97316" if v<85 else "#10b981" for v in cd["completeness"]]
        ax.barh(cd["module"]+"/"+cd["table"], cd["completeness"], color=colors)
        ax.axvline(80, color="orange", ls="--", lw=1, label="80% threshold")
        ax.set_xlabel("Average Column Completeness (%)"); ax.set_title("Data Quality by Module/Table")
        ax.invert_yaxis(); ax.legend(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(97, "Shared Entity Presence Score",
            "For each column name that appears in 2+ modules: how many total rows reference it? "
            "High-presence shared columns are the integration backbone of the system.")
    col_count = {}
    for mod, dfs_map in all_data.items():
        for tbl, df2 in dfs_map.items():
            for col in df2.columns:
                col_count[col] = col_count.get(col,0) + len(df2)
    col_mod_count = {}
    for mod, dfs_map in all_data.items():
        for tbl, df2 in dfs_map.items():
            for col in df2.columns:
                col_mod_count.setdefault(col,set()).add(mod)
    cross_cols = {k:v for k,v in col_count.items() if len(col_mod_count.get(k,set()))>=2}
    if cross_cols:
        cc_df = pd.DataFrame({"column":list(cross_cols.keys()),
                               "total_rows":list(cross_cols.values()),
                               "module_count":[len(col_mod_count[k]) for k in cross_cols]
                               }).sort_values("total_rows",ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(11,7))
        sc = ax.barh(cc_df["column"], cc_df["total_rows"], color=[ACCENT]*len(cc_df))
        ax.set_xlabel("Total Rows Containing This Column"); ax.set_title("Cross-Module Shared Column Presence (Top 30)")
        ax.invert_yaxis(); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(98, "All-Modules Numeric Distribution Heatmap",
            "For each module, normalised mean of all numeric columns stacked into a single heatmap. "
            "Instant view of how numeric profiles compare across the entire system.")
    mod_numeric_means = {}
    for mod, dfs_map in all_data.items():
        for tbl, df2 in dfs_map.items():
            nc2 = num_cols(df2)
            if nc2:
                means = df2[nc2].mean()
                normed = (means - means.min()) / (means.max() - means.min() + 1e-9)
                mod_numeric_means[f"{mod}/{tbl}"] = normed
    if mod_numeric_means:
        all_nc = sorted(set().union(*[set(v.index) for v in mod_numeric_means.values()]))[:30]
        hm_data = pd.DataFrame({k: v.reindex(all_nc).fillna(0) for k,v in mod_numeric_means.items()}).T
        fig, ax = plt.subplots(figsize=(14,max(5,len(mod_numeric_means)*0.5)))
        im = ax.imshow(hm_data.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(all_nc))); ax.set_xticklabels(all_nc, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(hm_data))); ax.set_yticklabels(hm_data.index, fontsize=7)
        ax.set_title("Normalised Numeric Means — All Modules × Columns"); plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(99, "Temporal Synchrony — Do Modules Peak Together?",
            "Cross-correlation matrix of module weekly volumes at different lags. "
            "Lag-1 correlation = one module reliably precedes another — a causal signal.")
    if len(daily_series) >= 3:
        weekly = pd.DataFrame({m: s.resample("W").sum() for m,s in daily_series.items()}).fillna(0)
        lags_to_check = [0,1,2,4]
        fig, axes = plt.subplots(1, len(lags_to_check), figsize=(16,4))
        mods_list = list(weekly.columns)
        for ax, lag in zip(axes, lags_to_check):
            shifted = weekly.shift(lag)
            corr = weekly.corrwith(shifted).values if lag > 0 else weekly.corr().values.diagonal()
            if lag == 0:
                corr_mat = weekly.corr()
            else:
                corr_mat = pd.concat([weekly,shifted.add_suffix("_lag")],axis=1).corr()
                corr_mat = corr_mat.loc[mods_list,[m+"_lag" for m in mods_list]]
            sub = weekly.corr() if lag==0 else corr_mat.values
            if hasattr(sub,"values"): sub = sub.values
            im = ax.imshow(sub, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_title(f"Lag={lag}W"); ax.set_xticks(range(len(mods_list))); ax.set_yticks(range(len(mods_list)))
            ax.set_xticklabels([m[:6] for m in mods_list], rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels([m[:6] for m in mods_list], fontsize=7)
        fig.suptitle("Cross-Module Temporal Synchrony (Cross-Correlation at Different Lags)"); fig.tight_layout(); fig_to_st(fig)
    else: no_data()

    section(100, "System Intelligence Score — Composite Hospital Health Index",
            "Final composite: data quality × freshness × volume × numeric richness × cross-module connectivity. "
            "A single number per module representing how 'analysis-ready' it is. "
            "Use this to prioritise data engineering investment.")
    if all_data:
        scores = []
        for mod, dfs_map in all_data.items():
            for tbl, df2 in dfs_map.items():
                dc2 = best_date_col(df2); nc2 = num_cols(df2); cc2 = cat_cols(df2)
                quality   = df2.notna().mean().mean()
                volume    = min(1, np.log1p(len(df2)) / np.log1p(10000))
                richness  = min(1, len(nc2) / 10)
                freshness = 0
                if dc2:
                    last = df2[dc2].max()
                    if pd.notna(last):
                        freshness = max(0, 1 - (pd.Timestamp.now(tz="UTC")-last).days/30)
                connectivity = min(1, len([c for c in df2.columns if len(col_mod_count.get(c,set()))>=2])/5)
                composite = (quality*0.25 + volume*0.20 + richness*0.20 + freshness*0.20 + connectivity*0.15)
                scores.append({"module":mod,"table":tbl,"quality":quality,"volume":volume,"richness":richness,
                               "freshness":freshness,"connectivity":connectivity,"composite":composite})
        sc_df = pd.DataFrame(scores).sort_values("composite",ascending=False)
        fig, axes = plt.subplots(1,2,figsize=(15,6))
        axes[0].barh(sc_df["module"]+"/"+sc_df["table"], sc_df["composite"],
                     color=[ACCENT if v>=0.7 else "#f97316" if v>=0.5 else "#ef4444" for v in sc_df["composite"]])
        axes[0].set_xlabel("Composite Intelligence Score (0-1)")
        axes[0].set_title("Hospital Module Intelligence Score (Higher = More Analysis-Ready)")
        axes[0].invert_yaxis()
        component_cols = ["quality","volume","richness","freshness","connectivity"]
        comp_pivot = sc_df.set_index("module")[component_cols]
        if not comp_pivot.empty:
            comp_pivot.plot(kind="bar", stacked=True, ax=axes[1], colormap=CMAP, alpha=0.85)
            axes[1].set_title("Score Component Breakdown by Module")
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha="right")
            axes[1].legend(fontsize=8)
        fig.tight_layout(); fig_to_st(fig)
        st.success(f"🏆 Top scoring module: **{sc_df.iloc[0]['module']}/{sc_df.iloc[0]['table']}** — Composite: {sc_df.iloc[0]['composite']:.3f}")
    else: no_data()

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
with st.expander("🔍 Raw Table Inspector"):
    mod_sel = st.selectbox("Module", ALL_MODULES, key="raw_mod")
    tbl_sel = st.selectbox("Table", table_map.get(mod_sel, []), key="raw_tbl")
    if tbl_sel:
        raw = load_table(mod_sel, tbl_sel)
        st.caption(f"{len(raw):,} rows × {len(raw.columns)} columns")
        st.dataframe(safe_show(raw, 100), use_container_width=True)
