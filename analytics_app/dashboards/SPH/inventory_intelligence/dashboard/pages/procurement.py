"""Spending & suppliers — spend, supplier concentration, and the gap between
what was ordered and what was received.

Most purchasing here bypasses formal purchase orders, so goods received — not
orders — are the real replenishment signal.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from inventory_intelligence.dashboard import charts, components, data_access, theme  # noqa: E402

components.page_header(
    "Spending & suppliers",
    "Where the money goes, and who we buy from",
)

components.warehouse_required()

facility_code = data_access.selected_facility()
months = st.sidebar.slider(
    "Lookback (months)", 6, 60, 36,
    help="How far back to look.",
)

spend = data_access.procurement_spend(facility_code, months=months)
suppliers = data_access.supplier_summary(facility_code)

if spend is None or spend.empty:
    components.empty_state(
        "No spending records for this facility and time range."
    )
    st.stop()

selected_categories = components.category_sidebar_filter(spend)
spend_view = data_access.apply_category_filter(spend, selected_categories)

# ── Ordered vs received ───────────────────────────────────────────────────────

ordered_total = spend_view.loc[spend_view["doc_type"] == "ORDER", "total_value"].sum()
received_total = spend_view.loc[spend_view["doc_type"] == "RECEIPT", "total_value"].sum()

top5_share = float("nan")
if suppliers is not None and not suppliers.empty and "received_value" in suppliers.columns:
    rv = pd.to_numeric(suppliers["received_value"], errors="coerce")
    if rv.sum() > 0:
        top5_share = float(rv.nlargest(5).sum() / rv.sum())

components.kpi_row(
    [
        {
            "label": "Received value",
            "value": theme.fmt_kes_compact(received_total),
            "detail": f"goods received, last {months} months",
        },
        {
            "label": "Through a formal order",
            "value": theme.fmt_kes_compact(ordered_total),
            "detail": "the rest is received without a purchase order — a control gap, "
                      "not a delivery shortfall",
        },
        {
            "label": "Suppliers",
            "value": f"{len(suppliers):,}" if suppliers is not None else "—",
            "detail": "distinct medical-goods suppliers",
        },
        {
            "label": "Top 5 suppliers' share",
            "value": theme.fmt_pct(top5_share, 0),
            "detail": "of received value — how concentrated buying is",
        },
    ]
)

st.plotly_chart(charts.spend_trend(spend_view), use_container_width=True)

# ── Suppliers ─────────────────────────────────────────────────────────────────

st.subheader("Supplier concentration")
if suppliers is not None and not suppliers.empty:
    st.caption(
        "Spend through the largest suppliers — a steep curve means concentration "
        "(leverage to negotiate, but also dependency)."
    )
    st.plotly_chart(charts.supplier_pareto(suppliers), use_container_width=True)
    with st.expander("Supplier table"):
        st.dataframe(
            suppliers.sort_values("received_value", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "ordered_value": components.kes_column("ordered (KES)"),
                "received_value": components.kes_column("received (KES)"),
            },
        )
else:
    components.empty_state("No suppliers on record for this facility.")

# ── Supplier dependency — item-level, observed ────────────────────────────────

dep = data_access.supplier_dependency(facility_code)
if dep is None or dep.empty:
    components.section_header("Supplier dependency — items relying on one observed supplier")
    components.empty_state("No goods-receipt records to assess sourcing from.")
else:
    dep = dep.copy()
    for c in ("received_value", "receipt_occasions", "observed_suppliers", "span_days"):
        if c in dep.columns:
            dep[c] = pd.to_numeric(dep[c], errors="coerce")
    oneoff = dep[dep["receipt_occasions"] < 2]
    repeat = dep[dep["receipt_occasions"] >= 2]
    single = repeat[repeat["observed_suppliers"] == 1]
    repeat_val = repeat["received_value"].sum(skipna=True)
    single_val = single["received_value"].sum(skipna=True)
    share = (single_val / repeat_val) if repeat_val else float("nan")

    components.section_header(
        "Supplier dependency — items relying on one observed supplier",
        finding=f"{theme.fmt_kes_compact(single_val)} · {theme.fmt_pct(share, 0)} of "
                "repeat-purchased spend on one observed supplier")
    components.kpi_row(
        [
            {"label": "Repeat-purchased spend through a single observed supplier",
             "value": theme.fmt_kes_compact(single_val),
             "detail": f"{theme.fmt_pct(share, 0)} of {theme.fmt_kes_compact(repeat_val)} "
                       f"repeat-purchased received value · {len(single):,} items"},
            {"label": "Items assessed (repeat-purchased)", "value": theme.fmt_compact(len(repeat)),
             "detail": f"received on 2+ occasions · {len(oneoff):,} one-off purchases excluded"},
        ]
    )
    st.caption(
        "Assessed only on items received **2+ times** — the "
        f"**{len(oneoff):,} one-off purchases are excluded** (one receipt can't show sourcing). "
        "*Observed* = seen in records, not proof no alternative supplier exists."
    )
    st.plotly_chart(charts.observed_supplier_bars(repeat), use_container_width=True)

    # ── Exposure — dependency × inventory risk (the hero) ─────────────────────
    components.section_header(
        "Exposure — single-supplier items that also carry inventory risk")
    health = data_access.with_names(data_access.load_table("inventory_health"))
    sk = data_access.load_table("stockout_risk")
    if health is None or health.empty:
        components.empty_state("Inventory figures aren't available to cross-check exposure.")
    else:
        h = health.copy()
        h["item_key"] = h["item_key"].astype(str)
        keep = [c for c in ["item_key", "display_name", "product_category",
                            "inventory_value", "movement_class", "itr"] if c in h.columns]
        exp = single.merge(h[keep], on="item_key", how="inner")
        if sk is not None and "p_stockout_30" in sk.columns:
            sk = sk.copy()
            sk["item_key"] = sk["item_key"].astype(str)
            exp = exp.merge(sk[["item_key", "p_stockout_30"]], on="item_key", how="left")
        exp["p_stockout_30"] = pd.to_numeric(exp.get("p_stockout_30"), errors="coerce")
        exp["inventory_value"] = pd.to_numeric(exp.get("inventory_value"), errors="coerce")
        exp["itr"] = pd.to_numeric(exp.get("itr"), errors="coerce")
        active_itr_med = pd.to_numeric(
            h.loc[h["movement_class"] == "active", "itr"], errors="coerce").median()

        # Exposure = single observed supplier AND already at stockout risk (>=50%).
        # Transparent tiers (very high >=80% / high >=50%), then received value.
        at_risk = exp[exp["p_stockout_30"] >= 0.5].copy()
        _p = at_risk["p_stockout_30"]
        at_risk["risk_tier"] = np.where(_p >= 0.8, 2, 1)
        at_risk = at_risk.sort_values(["risk_tier", "received_value"], ascending=[False, False])
        very_high = int((at_risk["risk_tier"] == 2).sum())
        st.caption(
            f"Of **{len(exp):,} single-observed-supplier items in inventory**, "
            f"**{len(at_risk):,} are already at stockout risk** (**{very_high:,} very high, ≥80%**) "
            "— single supplier *and* about to run short. Ranked by risk, then value. SKU-level "
            "(a molecule may be multi-sourced under another brand)."
        )
        exp_cols = [c for c in ["display_name", "supplier_name", "receipt_occasions",
                                "span_days", "received_value", "p_stockout_30",
                                "inventory_value", "movement_class"] if c in at_risk.columns]
        st.dataframe(
            at_risk[exp_cols].head(100), use_container_width=True, hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Item"),
                "supplier_name": st.column_config.TextColumn("Observed supplier"),
                "receipt_occasions": st.column_config.NumberColumn("Receipts", format="%.0f"),
                "span_days": st.column_config.NumberColumn("Span (days)", format="%.0f"),
                "received_value": components.kes_column("Received (KES)"),
                "p_stockout_30": components.chance_column("Stockout risk (1 mo)"),
                "inventory_value": components.kes_column("Stock value (KES)"),
                "movement_class": st.column_config.TextColumn("Movement"),
            },
        )

        # ── Early warning — fast movers, single supplier, before risk emerges ──
        components.section_header(
            "Early supply vulnerabilities — fast movers, before risk emerges")
        ew = exp[(exp["movement_class"] == "active") & (exp["itr"] >= active_itr_med)
                 & (exp["p_stockout_30"] < 0.5)].sort_values("received_value", ascending=False)
        if ew.empty:
            components.empty_state(
                "No fast-moving single-supplier items outside the exposure list right now.")
        else:
            st.caption(
                f"**{len(ew):,} fast-moving, single-observed-supplier items *not yet* at risk** "
                "— fast movers turn into stockouts quickly, so pre-empt (diversify or buffer) "
                "before a supplier slip bites. Ranked by received value."
            )
            ew_cols = [c for c in ["display_name", "supplier_name", "receipt_occasions",
                                   "span_days", "itr", "p_stockout_30", "received_value"]
                       if c in ew.columns]
            st.dataframe(
                ew[ew_cols].head(50), use_container_width=True, hide_index=True,
                column_config={
                    "display_name": st.column_config.TextColumn("Item"),
                    "supplier_name": st.column_config.TextColumn("Observed supplier"),
                    "receipt_occasions": st.column_config.NumberColumn("Receipts", format="%.0f"),
                    "span_days": st.column_config.NumberColumn("Span (days)", format="%.0f"),
                    "itr": st.column_config.NumberColumn("Turnover / yr", format="%.1f"),
                    "p_stockout_30": components.chance_column("Stockout risk (1 mo)"),
                    "received_value": components.kes_column("Received (KES)"),
                },
            )
