"""Demand & Availability — what is driving inventory pressure.

The chain the page follows is Demand → Availability → Risk: what is being used,
whether the stock position can support that use, and where it is most likely to
run short. A single item drilldown ties an item's usage and forecast to its own
running-out risk, so the reader never has to reconstruct the link by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from inventory_intelligence.dashboard import charts, components, data_access, theme  # noqa: E402

components.page_header(
    "Demand & availability",
    "What's driving inventory pressure — demand, availability, and risk",
)

if not data_access.analytics_available():
    components.missing_analytics_stop()

risk = data_access.with_names(data_access.load_table("stockout_risk"))
selected_categories = components.category_sidebar_filter(risk)
risk = data_access.apply_category_filter(risk, selected_categories)

if risk is None or risk.empty:
    components.empty_state("No demand or availability figures in the current view.")
    st.stop()

risk = risk.copy()
for col in ["p_stockout_30", "p_stockout_60", "p_stockout_90", "soh",
            "days_to_stockout_med", "days_to_stockout_q10", "days_to_stockout_q90",
            "expected_shortfall_nomove"]:
    if col in risk.columns:
        risk[col] = pd.to_numeric(risk[col], errors="coerce")
_leads = data_access.item_lead_times(data_access.selected_facility())
if _leads:
    risk["lead_time"] = risk["item_key"].astype(str).map(_leads)

at_risk = int((risk["p_stockout_30"] > 0.5).sum()) if "p_stockout_30" in risk.columns else 0
st.caption(
    f"**{at_risk:,} of {len(risk):,} items** have a better-than-even chance of running "
    "out within a month."
)

# ── 1. Where availability is at risk ──────────────────────────────────────────

components.section_header("Where availability is at risk")
st.caption("Chance of running out over 1 / 2 / 3 months, with real lead time. Sortable.")
table = risk.sort_values("p_stockout_30", ascending=False)[
    [c for c in ["display_name", "product_category", "soh", "lead_time",
                 "p_stockout_30", "p_stockout_60", "p_stockout_90"] if c in risk.columns]
]
st.dataframe(
    table, use_container_width=True, hide_index=True, height=380,
    column_config={
        "display_name": st.column_config.TextColumn("Item"),
        "product_category": st.column_config.TextColumn("Type"),
        "soh": st.column_config.NumberColumn("Stock on hand", format="%.0f"),
        "lead_time": st.column_config.NumberColumn(
            "Lead time (days)", format="%.0f",
            help="Median days from purchase order to goods-receipt, from matched "
                 "PO/receipt records. Blank where none are matched (most buying "
                 "bypasses formal POs)."),
        "p_stockout_30": components.chance_column("Chance of running out within 1 month"),
        "p_stockout_60": components.chance_column("Chance of running out within 2 months"),
        "p_stockout_90": components.chance_column("Chance of running out within 3 months"),
    },
)

# ── 1b. Financial exposure — where the at-risk value concentrates ─────────────
# The section above ranks by probability; this ranks by money at stake. They are
# different questions: a cheap near-certain stockout and an expensive likely one
# are different problems. value_at_risk = expected 30-day shortfall × cost under
# the do-nothing scenario (the existing validated figure) — not realised loss.

cost_risk = data_access.with_names(data_access.load_table("cost_risk"))
cost_risk = data_access.apply_category_filter(cost_risk, selected_categories)
if cost_risk is None or cost_risk.empty or "value_at_risk" not in cost_risk.columns:
    components.section_header("Financial exposure — where the at-risk value concentrates")
    components.empty_state("No value-at-risk figures in the current view.")
else:
    cr = cost_risk.copy()
    cr["value_at_risk"] = pd.to_numeric(cr["value_at_risk"], errors="coerce")
    cr = cr[cr["value_at_risk"] > 0].sort_values("value_at_risk", ascending=False)
    total_var = cr["value_at_risk"].sum()
    top5_share = (cr["value_at_risk"].head(5).sum() / total_var) if total_var else float("nan")
    components.section_header(
        "Financial exposure — where the at-risk value concentrates",
        finding=f"Top 5 items account for {theme.fmt_pct(top5_share, 0)} of "
                f"{theme.fmt_kes_compact(total_var)} modelled value at risk")
    cr = cr.merge(risk[[c for c in ["item_key", "p_stockout_30", "soh"] if c in risk.columns]],
                  on="item_key", how="left")
    st.caption(
        "Ranked by *money at stake*, not probability. **Priced inventory only** — "
        "modelled 30-day shortfall × cost, do-nothing scenario, not realised loss."
    )
    show = cr.head(10)[[c for c in ["display_name", "product_category", "p_stockout_30",
                                    "value_at_risk", "soh"] if c in cr.columns]]
    st.dataframe(
        show, use_container_width=True, hide_index=True,
        column_config={
            "display_name": st.column_config.TextColumn("Item"),
            "product_category": st.column_config.TextColumn("Type"),
            "p_stockout_30": components.chance_column(
                "Chance of running out", "Probability of running out within a month — the "
                "availability lens, separate from the money-at-stake below."),
            "value_at_risk": components.kes_column(
                "Value at risk (KES)", "Expected 30-day shortfall × cost, do-nothing scenario."),
            "soh": st.column_config.NumberColumn("Stock on hand", format="%.0f"),
        },
    )

# ── 2. One item — usage, forecast, and its risk ───────────────────────────────

components.section_header("One item — usage, forecast, and its own risk")
forecast = data_access.with_names(data_access.load_table("demand_forecast"))
forecast = data_access.apply_category_filter(forecast, selected_categories)

if forecast is None or forecast.empty:
    components.empty_state("No forecasts in the current view.")
else:
    items = forecast.drop_duplicates("item_key")
    _names = dict(zip(items["item_key"].astype(str), items["display_name"].astype(str)))
    choice = st.selectbox("Item", items["item_key"].tolist(),
                          format_func=lambda k: _names.get(str(k), str(k)))
    item_fc = forecast[forecast["item_key"].astype(str) == str(choice)]
    fc_row = item_fc.iloc[0]
    r_match = risk[risk["item_key"].astype(str) == str(choice)]
    r_row = r_match.iloc[0] if not r_match.empty else None
    horizon = int(fc_row.get("horizon", 0) or 0)

    components.kpi_row(
        [
            {"label": "Expected use next 4 weeks",
             "value": theme.fmt_compact(fc_row.get("q50")),
             "detail": f"likely {theme.fmt_compact(fc_row.get('q05'))} – "
                       f"{theme.fmt_compact(fc_row.get('q95'))} units"},
            {"label": "Chance of running out",
             "value": theme.fmt_pct(r_row.get("p_stockout_30")) if r_row is not None else "—",
             "detail": "within 1 month at current stock"},
            {"label": "Days of stock left",
             "value": theme.fmt_days(r_row.get("days_to_stockout_med")) if r_row is not None else "—",
             "detail": "expected, at the recent usage rate"},
            {"label": "Days stock was unavailable",
             "value": theme.fmt_pct(fc_row.get("censored_frac")),
             "detail": "share of days usage couldn't be seen (stock was out)"},
        ]
    )

    panel = data_access.demand_panel(data_access.selected_facility())
    left, right = st.columns([3, 2])
    with left:
        if panel is not None:
            daily = panel["daily"]
            item_daily = daily[daily["item_key"].astype(str) == str(choice)]
            if not item_daily.empty:
                st.plotly_chart(charts.demand_history(item_daily, title="Usage per day"),
                                use_container_width=True)
                st.caption("Shaded periods mark days stock was unavailable, so usage "
                           "couldn't be recorded.")
            else:
                components.empty_state("No usage history for this item.")
        else:
            components.empty_state("Usage history isn't available right now.")
    with right:
        st.plotly_chart(charts.forecast_fan(item_fc, title="Expected use (with likely range)"),
                        use_container_width=True)
        st.caption("The dark line is expected use; the bands show the likely (inner) "
                   "and wider (outer) range.")

    policy = data_access.load_table("inventory_policy")
    if policy is not None and not policy.empty:
        m = policy[policy["item_key"].astype(str) == str(choice)]
        if not m.empty:
            p = m.iloc[0]
            st.caption(
                "Suggested levels — reorder at "
                f"**{theme.fmt_compact(p.get('reorder_point'))}**, top up to "
                f"**{theme.fmt_compact(p.get('order_up_to'))}**, buffer "
                f"**{theme.fmt_compact(p.get('safety_stock'))}**. The **Ordering plan** "
                "turns this into a quantity to buy."
            )

# ── 3. Unusual usage ──────────────────────────────────────────────────────────

components.section_header("Unusual usage — what changed, and what to do")
anomalies = data_access.with_names(data_access.load_table("anomalies"))
anomalies = data_access.apply_category_filter(anomalies, selected_categories)

if anomalies is None or anomalies.empty:
    components.empty_state("Nothing to check yet.",
                           note="Usage is in line with the usual pattern, or this check hasn't run.")
else:
    anomalies = anomalies.copy()
    flagged = anomalies[anomalies["fdr_flag"].fillna(False).astype(bool)].copy()
    if flagged.empty:
        st.caption("Nothing is off-pattern in the current figures — every item is "
                   "tracking its own history. No action needed here.")
    else:
        st.caption(
            f"**{len(flagged):,} items** are off their usual usage — right = using more, "
            "left = using less. A surge may need a bigger order; a collapse, a recording gap."
        )
        st.plotly_chart(charts.anomaly_flagged_bars(flagged), use_container_width=True)
        rr = pd.to_numeric(flagged.get("rate_ratio_window"), errors="coerce")
        flagged["pct"] = (rr - 1.0) * 100.0
        flagged["move"] = flagged["pct"].map(lambda v: "Using more" if v > 0 else "Using less")
        flagged = flagged.reindex(flagged["pct"].abs().sort_values(ascending=False).index)
        show_cols = [c for c in ["display_name", "product_category", "move", "pct", "window"]
                     if c in flagged.columns]
        st.dataframe(
            flagged[show_cols].head(50), use_container_width=True, hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Item"),
                "product_category": st.column_config.TextColumn("Category"),
                "move": st.column_config.TextColumn("Direction"),
                "pct": st.column_config.NumberColumn("vs usual", format="%+.0f%%"),
                "window": st.column_config.TextColumn("Period"),
            },
        )

# ── Service lens: fulfilment (distinct from availability) ──────────────────────

with st.expander("A note on fulfilment vs availability"):
    st.caption(
        "Availability above asks *can our stock cover expected use?* — computed from "
        "cover and usage (v1). **Fulfilment** — *of what was prescribed, how much was "
        "actually dispensed?* — is a different, prescription-level question. The v2 "
        "prescription data can support it, but the raw rate is misleading (most "
        "un-dispensed lines are discharge / theatre / outpatient scripts never meant "
        "to be filled from this store; inpatient dispensing is ~99.5%). It is held back "
        "as a distinct service lens until the population is validated — see **Data quality**."
    )
