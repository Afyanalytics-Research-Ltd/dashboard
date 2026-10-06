"""
The five decision-led views. Each takes a prepared context (ctx) built once in
app.py so data loads a single time per session.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils.components import inject_css, page_header, section_header, kpi_row
from utils.formatting import fmt_kes, fmt_kes_millions, fmt_int, fmt_pct
from queries import analytics as A

# ── Severity palette (semantic, not the brand accent) ─────────────────────────
_SEV = {"critical": "#B42318", "serious": "#B45309", "warning": "#B45309", "info": "#0C447C"}
_SEV_BG = {"critical": "#FEE4E2", "serious": "#FEF0C7", "warning": "#FEF0C7", "info": "#EAF1FB"}
_SEV_LABEL = {"critical": "CRITICAL", "serious": "HIGH", "warning": "MEDIUM", "info": "WATCH"}

_EXTRA_CSS = """
<style>
.callout{border-radius:0 8px 8px 0;padding:11px 15px;margin:6px 0 14px;font-size:13.5px;line-height:1.55}
.callout .c-t{font-weight:700;margin-bottom:2px}
.callout.info{background:#F0F6FC;border-left:4px solid #0C447C;color:#0f2540}
.callout.warn{background:#FFFBEB;border-left:4px solid #B45309;color:#5b3c0a}
.callout.crit{background:#FEF2F2;border-left:4px solid #B42318;color:#5a1512}
.diagnosis{display:flex;gap:0;background:#fff;border:1px solid #E5E7EB;border-radius:10px;
  overflow:hidden;margin:2px 0 18px}
.diagnosis .d-cell{flex:1;padding:12px 16px;border-right:1px solid #EEF0F2}
.diagnosis .d-cell:last-child{border-right:none}
.diagnosis .d-k{font-size:10.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:#9CA3AF}
.diagnosis .d-v{font-size:13.5px;font-weight:600;color:#111827;margin-top:3px;line-height:1.35}
.diagnosis .d-v .accent{color:#B42318}
.sig{display:flex;gap:11px;padding:11px 0;border-bottom:1px solid #F0F1F4}
.sig:last-child{border-bottom:none}
.sig .bar{width:3px;border-radius:2px;flex:none}
.sig .t{font-weight:700;font-size:13.5px;color:#111827}
.sig .d{font-size:12.5px;color:#4b5563;margin-top:3px;line-height:1.45}
.sig .m{font-size:11px;color:#9CA3AF;margin-top:4px;font-family:ui-monospace,Consolas,monospace}
.pill{display:inline-block;font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;
  text-transform:uppercase;letter-spacing:.04em;vertical-align:middle}
.work{display:flex;gap:14px;padding:14px 0;border-bottom:1px solid #F0F1F4}
.work:last-child{border-bottom:none}
.work .rk{width:26px;height:26px;border-radius:7px;background:#F3F4F6;display:grid;place-items:center;
  font-weight:800;font-size:13px;flex:none;color:#374151}
.work .amt{font-size:19px;font-weight:800;letter-spacing:-.02em;color:#111827;text-align:right;white-space:nowrap}
.work .amt small{display:block;font-size:10px;font-weight:600;color:#9CA3AF;letter-spacing:.05em;text-transform:uppercase}
.work .h{font-size:14px;font-weight:700;color:#111827}
.work .act{font-size:12.5px;color:#4b5563;margin-top:5px;line-height:1.5}
.work .meta{font-size:11px;color:#9CA3AF;margin-top:5px;font-family:ui-monospace,Consolas,monospace}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin-top:6px;font-size:12px;color:#4b5563}
.legend i{width:11px;height:11px;border-radius:3px;display:inline-block;margin-right:5px;vertical-align:middle}
.note{font-size:12px;color:#6B7280;line-height:1.5}
/* equal-height KPI tiles — reserve consistent space per part so tiles align
   regardless of how many lines each label/value/delta wraps to */
.kpi-tile{display:flex;flex-direction:column}
.kpi-label{min-height:34px}
.kpi-value{min-height:64px}
.kpi-delta{min-height:30px;margin-top:auto;padding-top:6px}
</style>
"""


def data_notes_expander(ctx):
    """Sidebar 'Data notes & caveats' — the analytical caveats live here, not on the pages."""
    with st.expander("Data notes & caveats"):
        cutoff = A.mon_label(ctx.cutoff)
        for t, d, col in [
            ("Dispatch collapse (Sep 2025 →)",
             "0% dispatch for the last seven data months is a real operational failure, not a data gap.", "#B42318"),
            ("Collection rate is not shown as performance",
             "The payment feed is incomplete (~KES 64M gap) and most AR isn't due yet, so a same-month "
             "collection rate would mislead. We show channel mix only.", "#B42318"),
            ("AR aging is anchored to the data cutoff",
             f"The reporting snapshot ages balances to today, but data ends {cutoff}; that ~{ctx.lag}-day lag "
             f"would push everything into '90+'. We re-age at invoice level to {cutoff} for a true distribution.", "#B45309"),
            ("Exposure is counted once",
             "Open AR (stock) is separated from operational leakage (flow); the unknown-insurer balance is a "
             "subset of AR, not an additional amount, so nothing is double-counted.", "#B45309"),
            ("Theatre billing capture is uncertain",
             "The booking funnel and the leakage table disagree on completed-but-unbilled theatre because "
             "visit-to-invoice linkage is incomplete. Treat the theatre figure as indicative.", "#B45309"),
            ("Pharmacy Mar-2025 outlier excluded",
             "A KES 21M Nutriflex data-entry error is removed from every pharmacy figure.", "#B45309"),
            ("Latest month is partial",
             f"The final month in the data is in-flight; headline figures use the last complete month "
             f"({A.mon_label(ctx.cur['REV_MONTH'])}).", "#0C447C"),
            ("Excluded periods",
             "The Apr–Aug 2024 pre-go-live ramp and the Oct-2025 anomaly are excluded from trends.", "#0C447C"),
        ]:
            st.markdown(f"<div style='font-size:12px;margin-bottom:9px'><b style='color:{col}'>{t}</b><br>"
                        f"<span style='color:#4b5563'>{d}</span></div>", unsafe_allow_html=True)


def setup():
    inject_css()
    st.markdown(_EXTRA_CSS, unsafe_allow_html=True)


# ── shared helpers ────────────────────────────────────────────────────────────

def callout(kind, title, body):
    st.markdown(f'<div class="callout {kind}"><div class="c-t">{title}</div>{body}</div>',
                unsafe_allow_html=True)


def pill(sev):
    return f'<span class="pill" style="background:{_SEV_BG[sev]};color:{_SEV[sev]}">{_SEV_LABEL[sev]}</span>'


def render_signals(signals):
    if not signals:
        st.markdown("<div style='font-size:13px;color:#0F6E56;padding:8px 0'>✓ No active alerts.</div>",
                    unsafe_allow_html=True)
        return
    for s in signals:
        st.markdown(
            f"<div class='sig'><div class='bar' style='background:{_SEV[s['sev']]}'></div>"
            f"<div style='flex:1;min-width:0'><div class='t'>{s['title']} &nbsp;{pill(s['sev'])}</div>"
            f"<div class='d'>{s['desc']}</div><div class='m'>{s['metric']} · {s['owner']}</div></div></div>",
            unsafe_allow_html=True)


def _svc_shares(svc, month):
    d = svc.copy()
    d["REV_MONTH"] = pd.to_datetime(d["REV_MONTH"])
    m = d[d["REV_MONTH"] == month]
    tot = m["REVENUE"].sum() or 1
    return {r.SERVICE_LINE: r.REVENUE / tot * 100 for r in m.itertuples()}


def _pharm_line(pt):
    fig = go.Figure(go.Scatter(
        x=[A.mon_label(m) for m in pt["REV_MONTH"]], y=pt["LEAKAGE_KES"], mode="lines+markers",
        line=dict(color="#B45309", width=2.4), fill="tozeroy", fillcolor="rgba(180,83,9,0.10)",
        marker=dict(size=5), hovertemplate="%{x}<br>KES %{y:,.0f}<extra></extra>"))
    fig.update_layout(**C._LAYOUT, height=300, showlegend=False,
                      xaxis=dict(showgrid=False, tickfont=dict(size=9)),
                      yaxis=dict(gridcolor=C.GRID, tickprefix="KES ", tickformat=".2s", tickfont=dict(size=10)))
    return fig


# ════════════════════════════════ 1 · EXECUTIVE ══════════════════════════════

def view_exec(ctx):
    exp, cur, prior = ctx.exp, ctx.cur, ctx.prior
    page_header("Executive Brief",
                "The facility is billing well — but not converting billings into cash. The problem is largely internal.")

    # One-line diagnosis: state → problem → cause → owner
    und_pct = exp.undispatched / exp.total_ar * 100
    st.markdown(
        f"<div class='diagnosis'>"
        f"<div class='d-cell'><div class='d-k'>Financial state</div>"
        f"<div class='d-v'>Revenue healthy · ~{fmt_kes_millions(cur['TOTAL_INVOICED'])}/mo</div></div>"
        f"<div class='d-cell'><div class='d-k'>The problem</div>"
        f"<div class='d-v'><span class='accent'>{fmt_kes_millions(exp.total_ar)}</span> owed · {und_pct:.0f}% never dispatched</div></div>"
        f"<div class='d-cell'><div class='d-k'>Root cause</div>"
        f"<div class='d-v'>Claim dispatch stopped <span class='accent'>Sep 2025</span></div></div>"
        f"<div class='d-cell'><div class='d-k'>Accountability</div>"
        f"<div class='d-v'>Mostly <span class='accent'>internal</span> (Billing)</div></div>"
        f"</div>", unsafe_allow_html=True)

    rev_mom = (cur["TOTAL_INVOICED"] - prior["TOTAL_INVOICED"]) / prior["TOTAL_INVOICED"] * 100 if prior is not None else None
    kpi_row([
        {"label": f"Gross revenue · {A.mon_label(cur['REV_MONTH'])}", "value": fmt_kes_millions(cur["TOTAL_INVOICED"]),
         "delta": (fmt_pct(rev_mom) + " vs prior month") if rev_mom is not None else "",
         "delta_good": (rev_mom or 0) >= 0, "accent_color": "#0F6E56"},
        {"label": "Open insurer receivables", "value": fmt_kes_millions(exp.total_ar),
         "delta": "the exposure", "delta_good": True, "accent_color": "#0C447C"},
        {"label": "Recoverable by internal action", "value": fmt_kes_millions(exp.recoverable_internal),
         "delta": f"{exp.recoverable_internal / exp.total_exposure * 100:.0f}% of total exposure · no insurer needed",
         "delta_good": True, "accent_color": "#0F6E56"},
        {"label": "Aged 90+ & undispatched", "value": fmt_kes_millions(ctx.ninety_plus),
         "delta": "past most submission windows", "delta_good": False, "accent_color": "#B42318"},
    ])

    left, right = st.columns([1.7, 1])
    with left:
        section_header("Revenue trend")
        st.plotly_chart(C.revenue_trend(ctx.dm), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='legend'><span><i style='background:#0F6E56'></i>Invoiced</span>"
                    "<span><i style='background:#1A1A2E'></i>3-mo average</span>"
                    "<span><i style='background:#B45309;border-radius:50%'></i>In-flight (partial)</span></div>",
                    unsafe_allow_html=True)
    with right:
        section_header("Payer mix")
        ins = cur["INSURER_PCT_OF_TOTAL"]
        st.plotly_chart(C.donut(["Insurer", "Cash"], [ins, 100 - ins], [C.TEAL, "#1D9E75"],
                                center_big=f"{ins:.0f}%", center_small="insurer", height=210),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='note' style='text-align:center'>~85% insurer-funded — which is why "
                    "collection speed, not billing, is the constraint.</div>", unsafe_allow_html=True)

    section_header("Where the money is stuck — one reconciled exposure")
    ce, cs = st.columns([1.7, 1])
    with ce:
        st.plotly_chart(C.exposure_split(exp.undispatched, exp.dispatched),
                        use_container_width=True, config={"displayModeBar": False})
        callout("info", "Two clocks, one number",
                f"Open AR is <b>{fmt_kes_millions(exp.total_ar)}</b>: "
                f"<b>{fmt_kes_millions(exp.undispatched)}</b> undispatched (our delay) + "
                f"<b>{fmt_kes_millions(exp.dispatched)}</b> dispatched but unpaid (insurer delay). "
                f"Separately, operational leakage — pharmacy, theatre, credit notes — adds "
                f"<b>{fmt_kes_millions(exp.flow_total)}</b> of recoverable value.")
    with cs:
        section_header("Signals")
        render_signals(ctx.signals)


# ════════════════════════════════ 2 · REVENUE ════════════════════════════════

def view_revenue(ctx):
    dm, cur = ctx.dm, ctx.cur
    page_header("Revenue Performance",
                "Revenue is growing on patient volume; case value drives the month-to-month swings. "
                "Read the trend, not the partial final month.")
    complete = dm[dm["IS_COMPLETE"]]
    peak = complete.loc[complete["TOTAL_INVOICED"].idxmax()]
    kpi_row([
        {"label": f"Latest complete month ({A.mon_label(cur['REV_MONTH'])})", "value": fmt_kes_millions(cur["TOTAL_INVOICED"]),
         "accent_color": "#0F6E56"},
        {"label": "Invoices", "value": fmt_int(cur["INVOICE_COUNT"]), "accent_color": "#0C447C"},
        {"label": "Avg invoice value", "value": fmt_kes(cur["AVG_INVOICE_AMOUNT"]), "accent_color": "#0C447C"},
        {"label": "Peak month", "value": fmt_kes_millions(peak["TOTAL_INVOICED"]),
         "delta": A.mon_label(peak["REV_MONTH"]), "delta_good": True, "accent_color": "#0F6E56"},
    ])

    section_header("Monthly billing trend")
    st.plotly_chart(C.revenue_trend(dm, roll_col="ROLL_6M", height=320),
                    use_container_width=True, config={"displayModeBar": False})
    st.markdown("<div class='legend'><span><i style='background:#0F6E56'></i>Invoiced</span>"
                "<span><i style='background:#1A1A2E'></i>6-mo average</span></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        section_header("Revenue drivers — volume vs case value")
        st.plotly_chart(C.driver_bars(dm), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='note'>Each month's change split into <b>volume</b> (more invoices) and "
                    "<b>intensity</b> (higher value per case). The swings are mostly intensity-led — case mix and "
                    "tariffs move revenue more than patient counts do — while the underlying base has grown on volume "
                    "(patient count up ~41% since go-live vs ~16% on case value).</div>", unsafe_allow_html=True)
    with c2:
        section_header("Service-line share shift")
        last_c = cur["REV_MONTH"]
        earlier = last_c - pd.DateOffset(months=6)
        svc = ctx.D["svc"].copy()
        svc["REV_MONTH"] = pd.to_datetime(svc["REV_MONTH"])
        avail = svc[svc["REV_MONTH"] <= earlier]["REV_MONTH"]
        base_m = avail.max() if not avail.empty else svc["REV_MONTH"].min()
        now, was = _svc_shares(svc, last_c), _svc_shares(svc, base_m)
        names = [k for k in now if k != "Copay"]
        shifts = [now[k] - was.get(k, 0) for k in names]
        st.plotly_chart(C.svc_shift(names, shifts), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='note'>{A.mon_label(last_c)} vs {A.mon_label(base_m)}. Lab is easing as higher-value "
                    "Procedure and Inpatient care grow — a mix shift, not a billing gap.</div>", unsafe_allow_html=True)

    d1, d2 = st.columns([1, 1])
    with d1:
        section_header("Revenue by day of week")
        dow = ctx.D["dow"].copy()
        st.plotly_chart(C.hbar(dow["DAY_OF_WEEK_NAME"].tolist(), dow["AVG_DAILY_INVOICED"].tolist(),
                               color=C.BLUE, value_fmt="kes", height=240),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='note'>Weekday-concentrated, as expected for outpatient-heavy volume — "
                    "relevant to weekend staffing and theatre scheduling.</div>", unsafe_allow_html=True)
    with d2:
        section_header("Top revenue items (all-time)")
        ti = ctx.D["top_items"].copy()
        st.dataframe(pd.DataFrame({
            "Item": ti["ITEM_NAME"], "Line": ti["SERVICE_LINE"],
            "Billed": ti["TIMES_BILLED"].map(fmt_int),
            "Revenue": ti["TOTAL_REVENUE"].map(fmt_kes_millions),
        }).head(12), use_container_width=True, hide_index=True, height=240)


# ═══════════════════════════════ 3 · RECEIVABLES ═════════════════════════════

def view_ar(ctx):
    exp, D = ctx.exp, ctx.D
    page_header("Receivables & Cash Flow",
                "The money is owed to us — and mostly held up by us. Dispatch stopped entirely in September 2025.")

    kpi_row([
        {"label": "Total open AR", "value": fmt_kes_millions(exp.total_ar), "accent_color": "#0C447C"},
        {"label": "Undispatched (our delay)", "value": fmt_kes_millions(exp.undispatched),
         "delta": f"{exp.undispatched / exp.total_ar * 100:.0f}% of AR · healthy < 40%",
         "delta_good": False, "accent_color": "#B42318"},
        {"label": "Aged 90+ & undispatched", "value": fmt_kes_millions(ctx.ninety_plus),
         "delta": "submission-deadline risk", "delta_good": False, "accent_color": "#B42318"},
        {"label": "Insurer sitting time", "value": f"~{ctx.dispatched_sitting_days:.0f}d",
         "delta": "on dispatched, unpaid claims", "delta_good": False, "accent_color": "#B45309"},
    ])

    a1, a2 = st.columns([1.55, 1])
    with a1:
        section_header("Accountability by insurer — whose delay is it?")
        st.plotly_chart(C.two_clock(D["snap"], top=9), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='legend'><span><i style='background:#B42318'></i>Undispatched — our delay</span>"
                    "<span><i style='background:#B45309'></i>Dispatched, unpaid — insurer delay</span></div>",
                    unsafe_allow_html=True)
    with a2:
        section_header("Dispatch rate — the collapse")
        st.plotly_chart(C.dispatch_trend(D["dispatch"]), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='note'>Zero for the last seven months — an operational failure, not a data gap. "
                    "Every new insurer invoice now piles into undispatched AR.</div>", unsafe_allow_html=True)

    ag1, ag2 = st.columns([1, 1])
    with ag1:
        section_header("Undispatched AR by age — how urgent?")
        st.plotly_chart(C.aging_bar(ctx.aging_dist), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='note'>About <b>{ctx.ninety_plus/exp.undispatched*100:.0f}%</b> of undispatched AR is "
                    "90+ days old and past most insurers' submission windows; the fresher balances are the most "
                    "recoverable.</div>", unsafe_allow_html=True)
    with ag2:
        section_header("Concentration — where the AR sits")
        conc = D["conc"].copy()
        conc["TOTAL_INVOICED"] = pd.to_numeric(conc["TOTAL_INVOICED"], errors="coerce")
        top = conc.head(10)
        st.plotly_chart(C.pareto([A.clean_insurer(x) for x in top["PAYER_LABEL"]], top["TOTAL_INVOICED"].tolist(),
                                 height=260),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='note'>Collection effort should follow the money — a few payers hold most of it.</div>",
                    unsafe_allow_html=True)

    section_header("Receivables detail")
    snap = D["snap"].copy()
    snap["INSURER_LABEL"] = snap["INSURER_LABEL"].map(A.clean_insurer)
    piv = snap.pivot_table(index=["INSURER_LABEL", "PAYER_CLASS"], columns="AR_STATE",
                           values="OUTSTANDING_KES", aggfunc="sum", fill_value=0).reset_index()
    und_c = next((c for c in piv.columns if "Undispatched" in str(c)), None)
    dsp_c = next((c for c in piv.columns if "Dispatched" in str(c)), None)
    piv["Total AR"] = piv[[c for c in [und_c, dsp_c] if c]].sum(axis=1)
    piv = piv.sort_values("Total AR", ascending=False).head(15)
    st.dataframe(pd.DataFrame({
        "Insurer": piv["INSURER_LABEL"], "Class": piv["PAYER_CLASS"],
        "Undispatched": piv[und_c].map(fmt_kes_millions) if und_c else "—",
        "Dispatched-unpaid": piv[dsp_c].map(fmt_kes_millions) if dsp_c else "—",
        "Total AR": piv["Total AR"].map(fmt_kes_millions),
    }), use_container_width=True, hide_index=True, height=420)

    section_header("Cash coming in — mix, not performance")
    callout("info", "Why there is no collection-rate KPI here",
            "The payment feed is incomplete and most receivables are not yet due, so a collection rate would "
            "misrepresent performance. We show only what is reliable — the payment channel mix and the shift to digital.")
    m1, m2 = st.columns(2)
    with m1:
        section_header("Payment channel mix over time")
        st.plotly_chart(C.channel_mix(D["collections"]), use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div class='note'>M-Pesa dominant; cash effectively ended January 2025 — a real structural "
                    "shift to digital, not a data gap.</div>", unsafe_allow_html=True)
    with m2:
        section_header("Payer concentration — SHA")
        conc = D["conc"].copy(); conc["TOTAL_INVOICED"] = pd.to_numeric(conc["TOTAL_INVOICED"], errors="coerce")
        sha = conc[conc["PAYER_LABEL"].str.contains("SHA|Social Health", case=False, na=False)]
        sha_pct = sha["TOTAL_INVOICED"].sum() / conc["TOTAL_INVOICED"].sum() * 100 if not conc.empty else 0
        st.plotly_chart(C.donut(["SHA", "Other insurers"], [sha_pct, 100 - sha_pct], ["#0F6E56", "#CBD5D1"],
                                center_big=f"{sha_pct:.0f}%", center_small="SHA share", height=230),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<div class='note' style='text-align:center'>SHA is the single largest payer at {sha_pct:.0f}% of "
                    "insurer revenue — below the 35% single-payer guardrail, worth watching.</div>", unsafe_allow_html=True)


# ════════════════════════════════ 4 · LEAKAGE ════════════════════════════════

def view_leakage(ctx):
    exp, D = ctx.exp, ctx.D
    page_header("Revenue Leakage",
                "Smaller than the AR problem, but faster to recover. Flow vectors only — AR is excluded to avoid double-counting.")
    pf = D["pharm_fulfil"].iloc[0]
    th = D["theatre"].iloc[0]
    uc = D["leak"][D["leak"]["LEAKAGE_VECTOR"].str.contains("Consult", case=False, na=False)]
    uc_ct = int(pd.to_numeric(uc["EVENT_COUNT"].iloc[0])) if not uc.empty else 0

    kpi_row([
        {"label": "Operational leakage (flow)", "value": fmt_kes_millions(exp.flow_total), "accent_color": "#B45309"},
        {"label": "Pharmacy dispensed-unpaid", "value": fmt_kes_millions(exp.pharmacy_flow),
         "delta": f"{pf['LEAKAGE_RATE_PCT']:.0f}% of prescriptions", "delta_good": False, "accent_color": "#B45309"},
        {"label": "Theatre completed-unbilled", "value": fmt_kes_millions(exp.theatre_flow),
         "delta": "capture contested", "delta_good": False, "accent_color": "#B45309"},
        {"label": "Unbilled consultations", "value": fmt_int(uc_ct) + " visits",
         "delta": "no tariff — count only", "delta_good": False, "accent_color": "#0C447C"},
    ])

    outlier_kes = pd.to_numeric(D["outlier"]["OUTLIER_VALUE_KES"], errors="coerce").sum()
    callout("warn", "March 2025 outlier excluded",
            f"A <b>{fmt_kes_millions(outlier_kes)}</b> Nutriflex data-entry error is excluded from the pharmacy "
            "figures below.")

    p1, p2 = st.columns([1.6, 1])
    with p1:
        section_header("Pharmacy leakage trend (outlier excluded)")
        pt = D["pharm_trend"].copy()
        pt["REV_MONTH"] = pd.to_datetime(pt["PRESCRIPTION_MONTH"])
        pt = pt[pt["REV_MONTH"] >= A.GO_LIVE]
        st.plotly_chart(_pharm_line(pt), use_container_width=True, config={"displayModeBar": False})
    with p2:
        section_header("Fulfillment")
        st.plotly_chart(C.donut(["Dispensed & paid", "Leakage (unpaid)", "Cancelled"],
                                [pf["DISPENSED"], pf["LEAKAGE"], pf["CANCELLED"]],
                                ["#0F6E56", "#B45309", "#9CA3AF"],
                                center_big=f"{pf['LEAKAGE_RATE_PCT']:.0f}%", center_small="leakage", height=230),
                        use_container_width=True, config={"displayModeBar": False})

    d1, d2 = st.columns(2)
    with d1:
        section_header("Top drugs by leakage")
        td = D["top_drugs"].copy()
        st.dataframe(pd.DataFrame({
            "Drug": td["DRUG_NAME"], "Store": td["STORE_NAME"],
            "Events": td["LEAKAGE_EVENTS"].map(fmt_int),
            "Rate": td["LEAKAGE_RATE_PCT"].map(lambda v: f"{v:.0f}%"),
            "Leakage": td["LEAKAGE_KES"].map(fmt_kes_millions),
        }), use_container_width=True, hide_index=True, height=340)
    with d2:
        section_header("Prescriber accountability")
        dc = D["doctors"].copy()
        st.dataframe(pd.DataFrame({
            "Doctor": dc["PRESCRIBED_BY"], "Written": dc["PRESCRIPTIONS_WRITTEN"].map(fmt_int),
            "Unfilled": dc["UNFILLED_COUNT"].map(fmt_int),
            "Rate": dc["UNFILLED_RATE_PCT"].map(lambda v: f"{v:.0f}%"),
            "Value": dc["UNFILLED_KES"].map(fmt_kes_millions),
        }), use_container_width=True, hide_index=True, height=340)
        st.markdown("<div class='note'>Doctor names are not normalised at source (case variants) — indicative until merged.</div>",
                    unsafe_allow_html=True)

    section_header("Theatre — high value per case")
    t1, t2 = st.columns([1.4, 1])
    with t1:
        st.plotly_chart(C.theatre_funnel(int(th["BOOKED"]), int(th["SCHEDULED"]),
                                         int(th["COMPLETED"]), int(th["BILLED"])),
                        use_container_width=True, config={"displayModeBar": False})
    with t2:
        callout("warn", "Billing capture is uncertain",
                f"The booking funnel and the leakage measure disagree on completed-but-unbilled theatre "
                f"({fmt_kes_millions(exp.theatre_flow)}), so treat this figure as indicative until the "
                "theatre-to-invoice link is confirmed.")
        rej = D["theatre_rej"].copy()
        if not rej.empty:
            st.dataframe(pd.DataFrame({
                "Rejection reason": rej["REASON_SUMMARY"],
                "Bookings": rej["BOOKINGS"].map(fmt_int),
                "Lost rev.": rej["LOST_REVENUE_KES"].map(fmt_kes_millions),
            }), use_container_width=True, hide_index=True, height=180)


# ═══════════════════════════════ 5 · ACTION CENTER ══════════════════════════

def view_action(ctx):
    exp, actions = ctx.exp, ctx.actions
    page_header("Action Center",
                "Every recoverable opportunity, ranked so management can answer one question: what do we act on first?")

    total = sum(a["kes"] for a in actions)
    top3 = sum(a["kes"] for a in actions[:3])
    critical = sum(a["kes"] for a in actions if a["sev"] == "critical")
    kpi_row([
        {"label": "Total identifiable opportunity", "value": fmt_kes_millions(total), "accent_color": "#0F6E56"},
        {"label": "Recoverable by internal action", "value": fmt_kes_millions(exp.recoverable_internal),
         "delta": "no insurer needed", "delta_good": True, "accent_color": "#0F6E56"},
        {"label": "Top 3 actions unlock", "value": fmt_kes_millions(top3),
         "delta": f"{top3 / total * 100:.0f}% of the total", "delta_good": True, "accent_color": "#0C447C"},
        {"label": "Critical — act this week", "value": fmt_kes_millions(critical),
         "delta": f"{sum(1 for a in actions if a['sev']=='critical')} items", "delta_good": False, "accent_color": "#B42318"},
    ])

    callout("info", "How this is ranked",
            "Ordered by <b>recoverable value</b>, then read alongside <b>urgency</b> (how aged), "
            "<b>concentration</b> (fewer payers means easier), and a clear <b>owner</b> for each action.")

    section_header("Prioritised recovery worklist")
    maxk = max(a["kes"] for a in actions)
    for a in actions:
        bar = int(a["kes"] / maxk * 100)
        st.markdown(
            f"<div class='work'><div class='bar' style='width:3px;border-radius:2px;background:{_SEV[a['sev']]};flex:none'></div>"
            f"<div class='rk'>{a['rank']}</div>"
            f"<div style='flex:1;min-width:0'>"
            f"<div class='h'>{a['title']} &nbsp;{pill(a['sev'])} "
            f"<span class='pill' style='background:#F3F4F6;color:#374151'>{a['owner']}</span> "
            f"<span class='pill' style='background:#EAF1FB;color:#0C447C'>Recoverable: {a['recover']}</span></div>"
            f"<div class='act'>{a['action']}</div>"
            f"<div class='meta'>{a['vol']} · {a['age']} · concentration: {a['conc']}</div>"
            f"<div style='height:6px;border-radius:3px;background:#F3F4F6;margin-top:8px;max-width:340px'>"
            f"<div style='height:100%;width:{bar}%;border-radius:3px;background:{_SEV[a['sev']]}'></div></div>"
            f"</div>"
            f"<div class='amt'>{fmt_kes_millions(a['kes'])}<small>exposure</small></div></div>",
            unsafe_allow_html=True)

    o1, o2 = st.columns(2)
    with o1:
        section_header("By owner")
        by = {}
        for a in actions:
            by[a["owner"]] = by.get(a["owner"], 0) + a["kes"]
        st.plotly_chart(C.hbar(list(by.keys()), list(by.values()), color=C.TEAL, height=230),
                        use_container_width=True, config={"displayModeBar": False})
    with o2:
        section_header("The 90-day play")
        for badge, bcol, title, body in [
            ("Week 1–2", "#B42318", "Restart dispatch",
             f"Clear the undispatched backlog oldest-first, SHA first. Unblocks {fmt_kes_millions(exp.attributed_undispatched)} and stops daily accrual of unrecoverable age."),
            ("Week 2–4", "#B45309", "Fix the NULL-insurer defect",
             f"Resolve the mid-2025 source mapping, back-attribute {fmt_kes_millions(exp.unknown)}, then dispatch."),
            ("Month 2–3", "#0C447C", "Escalate aged insurer claims",
             f"Formal reconciliation on the {fmt_kes_millions(exp.dispatched)} dispatched-but-unpaid; recover what's live, write down what isn't."),
            ("Ongoing", "#0F6E56", "Charge-capture at source",
             f"Pharmacy dispensing + theatre billing controls to stop the {fmt_kes_millions(exp.flow_total)}/period flow leakage recurring."),
        ]:
            st.markdown(
                f"<div style='margin-bottom:13px'><span class='pill' style='background:{bcol}20;color:{bcol}'>{badge}</span> "
                f"<b style='font-size:13.5px'>{title}</b>"
                f"<div class='note' style='margin-top:4px'>{body}</div></div>", unsafe_allow_html=True)

    callout("info", "The bigger prize is what we can't yet see",
            "This is built from billing, AR, pharmacy and theatre data alone. Integrating the rest of the revenue "
            "cycle — remittance advice, claim adjudication, denial reasons, payment reconciliation — would turn "
            "“money we can identify as stuck” into “money we can track to recovery.”")


VIEWS = {
    "Executive Brief": view_exec,
    "Revenue": view_revenue,
    "Receivables & Cash": view_ar,
    "Revenue Leakage": view_leakage,
    "Action Center": view_action,
}
