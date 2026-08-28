"""Overview / Health — the executive entry point.

Top of the pyramid: one scorecard that answers *what is the current state of
inventory health?* — capital, how much of it is working vs idle, whether the
position can support demand (availability), and turnover — then routes the
reader to the page that explains each problem.
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
    "Overview",
    "The state of inventory health, at a glance",
)

if not data_access.analytics_available():
    components.missing_analytics_stop()

stockout = data_access.with_names(data_access.load_table("stockout_risk"))
cost_risk = data_access.with_names(data_access.load_table("cost_risk"))
health = data_access.with_names(data_access.load_table("inventory_health"))

selected_categories = components.category_sidebar_filter(stockout)
stockout = data_access.apply_category_filter(stockout, selected_categories)
cost_risk = data_access.apply_category_filter(cost_risk, selected_categories)
health = data_access.apply_category_filter(health, selected_categories)
health = data_access.enrich_surplus(health)  # consumption-based surplus for the capital split

# ── Scorecard ─────────────────────────────────────────────────────────────────

inv_total = (pd.to_numeric(health["inventory_value"], errors="coerce").sum()
             if health is not None and "inventory_value" in health.columns else float("nan"))
acv_total = (pd.to_numeric(health["annual_value"], errors="coerce").sum()
             if health is not None and "annual_value" in health.columns else float("nan"))
portfolio_itr = (acv_total / inv_total) if inv_total and inv_total > 0 else float("nan")

states = charts.capital_states(health) if health is not None else {}
working = states.get("working", float("nan"))
idle = (states.get("excess", 0) + states.get("slow", 0) + states.get("dead", 0)) if states else float("nan")
working_share = (working / inv_total) if inv_total and inv_total > 0 else float("nan")

chance = (pd.to_numeric(stockout["p_stockout_30"], errors="coerce")
          if stockout is not None and "p_stockout_30" in stockout.columns else pd.Series(dtype=float))
at_risk = int((chance > 0.5).sum())
n_items = len(stockout) if stockout is not None else 0
value_at_risk = (pd.to_numeric(cost_risk["value_at_risk"], errors="coerce").sum()
                 if cost_risk is not None and "value_at_risk" in cost_risk.columns else float("nan"))

components.kpi_row(
    [
        {"label": "Inventory value on hand", "value": theme.fmt_kes_compact(inv_total),
         "detail": "capital in stock, at acquisition cost"},
        {"label": "Capital at work", "value": theme.fmt_pct(working_share, 0),
         "detail": f"{theme.fmt_kes_compact(working)} working · "
                   f"{theme.fmt_kes_compact(idle)} idle or surplus"},
        {"label": "Items at risk of running out", "value": theme.fmt_compact(at_risk),
         "detail": f"of {n_items:,} tracked · better-than-even chance within a month"},
        {"label": "Value at risk of running short", "value": theme.fmt_kes_compact(value_at_risk),
         "detail": "unmet demand next month if nothing is ordered, at cost"},
        {"label": "Inventory turnover", "value":
            (f"{portfolio_itr:.1f}× / yr" if pd.notna(portfolio_itr) else "—"),
         "detail": "yearly use ÷ stock held"},
    ]
)

# ── Where the money sits ──────────────────────────────────────────────────────

if health is not None and not health.empty and "inventory_value" in health.columns:
    components.section_header("Where the capital sits — working vs idle")
    st.caption(
        "The same stock by how hard the capital works — green is in use; the rest is "
        "surplus, slow, or dead. **Movement & capital** breaks it down."
    )
    st.plotly_chart(charts.capital_decomposition_bar(health), use_container_width=True)

# ── What needs attention — routing into the story ─────────────────────────────

components.section_header("What needs attention now")
left, right = st.columns(2)

with left:
    st.markdown("**Running low — availability at risk**")
    st.caption("Highest chance of running out within a month.")
    if stockout is not None and not stockout.empty and "p_stockout_30" in stockout.columns:
        low = stockout.sort_values("p_stockout_30", ascending=False).head(8)
        cols = [c for c in ["display_name", "soh", "p_stockout_30"] if c in low.columns]
        st.dataframe(
            low[cols], use_container_width=True, hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Item"),
                "soh": st.column_config.NumberColumn("On hand", format="%.0f"),
                "p_stockout_30": components.chance_column("Chance of running out (1 month)"),
            },
        )
        st.caption("→ Full watchlist and the demand behind it: **Demand & availability**.")
    else:
        components.empty_state("No running-out figures in view.")

with right:
    st.markdown("**Trapped capital — value that isn't moving**")
    st.caption("Biggest capital in slow / dead stock.")
    if health is not None and not health.empty:
        trapped = health[health["movement_class"].isin(["slow", "dead"])].copy()
        trapped["inventory_value"] = pd.to_numeric(trapped["inventory_value"], errors="coerce")
        trapped = trapped.sort_values("inventory_value", ascending=False).head(8)
        cols = [c for c in ["display_name", "movement_class", "inventory_value"] if c in trapped.columns]
        st.dataframe(
            trapped[cols], use_container_width=True, hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Item"),
                "movement_class": st.column_config.TextColumn("Movement"),
                "inventory_value": components.kes_column("Capital tied up (KES)"),
            },
        )
        st.caption("→ The full working-vs-idle breakdown: **Movement & capital**.")
    else:
        components.empty_state("No stock-health figures in view.")

