"""Movement & Capital — where inventory capital is consumed or trapped.

The organising idea is velocity × inventory value = capital efficiency. Fast,
valuable stock must be protected; valuable stock that barely moves is trapped
capital to act on. The page opens with that map, then works down into the
specific capital to release: dead/slow, surplus above need, and expiry.
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
    "Movement & capital",
    "Where inventory capital is consumed — or trapped",
)

if not data_access.analytics_available():
    components.missing_analytics_stop()

health = data_access.with_names(data_access.load_table("inventory_health"))
selected_categories = components.category_sidebar_filter(health)
health = data_access.apply_category_filter(health, selected_categories)

if health is None or health.empty:
    components.empty_state("No stock-health figures yet.",
                           note="This needs a stock count and the latest run.")
    st.stop()

health = data_access.enrich_surplus(health)
for c in ["soh", "unit_price", "inventory_value", "annual_value", "itr",
          "days_of_supply", "days_since_last_dispense", "excess_units",
          "excess_value", "months_of_supply", "working_requirement", "years_of_supply"]:
    if c in health.columns:
        health[c] = pd.to_numeric(health[c], errors="coerce")
if "stock_suspect" in health.columns:
    health["stock_suspect"] = health["stock_suspect"].fillna(False).astype(bool)
else:
    health["stock_suspect"] = False

inv_total = health["inventory_value"].sum(skipna=True)
states = charts.capital_states(health)
slow_dead = float(states.get("slow", 0) + states.get("dead", 0))

components.kpi_row(
    [
        {"label": "Inventory value on hand", "value": theme.fmt_kes_compact(inv_total),
         "detail": "capital in stock, at acquisition cost"},
        {"label": "Capital genuinely working", "value": theme.fmt_kes_compact(states.get("working", float("nan"))),
         "detail": "active stock up to its top-up need"},
        {"label": "Slow-moving & dead capital", "value": theme.fmt_kes_compact(slow_dead),
         "detail": "value tied in stock that barely moves"},
    ]
)

# ── 1. Capital efficiency: velocity × value ───────────────────────────────────

components.section_header("Capital efficiency — velocity × value")

st.caption(
    "Each item by turnover (→) against capital held (↑), coloured by movement. "
    "**Upper-left = valuable but slow (trapped); upper-right = fast & valuable (protect).** "
    "Dotted lines mark the medians."
)
st.plotly_chart(charts.capital_efficiency_matrix(health), use_container_width=True)

# ── 2. Where the capital sits ─────────────────────────────────────────────────

components.section_header("Where the capital sits — and what's releasable")
_releasable = states.get("dead", 0) + states.get("slow", 0) + states.get("excess", 0)
st.caption(
    f"About **{theme.fmt_kes_compact(_releasable)}** could be freed — stepping from all "
    "stock through **dead** (365+ d), **slow** (180–365 d), and **surplus** (held above "
    "use) to the working base."
)
st.plotly_chart(charts.capital_waterfall(health), use_container_width=True)

# ── 3. Trapped capital — valuable but not moving ──────────────────────────────

# Exclude probable miscounts (>2y supply) so this reconciles with the KPI and the
# waterfall, which both quarantine suspect stock rather than book it as releasable.
_nonmoving_all = health[health["movement_class"].isin(["dead", "slow"])]
nonmoving = _nonmoving_all[~_nonmoving_all["stock_suspect"]].copy()
_susp_n = int(_nonmoving_all["stock_suspect"].sum())
if nonmoving.empty:
    components.section_header("Trapped capital — valuable stock that isn't moving")
    components.empty_state("Nothing dead or slow-moving in the current view.")
else:
    nonmoving = nonmoving.sort_values("inventory_value", ascending=False)
    _nm_total = nonmoving["inventory_value"].sum(skipna=True)
    _lead = nonmoving.iloc[0]
    _lead_val = float(_lead["inventory_value"])
    _top3 = nonmoving["inventory_value"].head(3).sum(skipna=True)
    components.section_header(
        "Trapped capital — valuable stock that isn't moving",
        finding=f"{theme.fmt_kes_compact(_nm_total)} · "
                f"{_top3 / _nm_total:.0%} sits in just 3 items")
    _susp_note = (f" A further **{_susp_n}** item(s) are held back as probable miscounts "
                  "(over two years' supply)." if _susp_n else "")
    st.caption(
        f"**{len(nonmoving):,} items** hold **{theme.fmt_kes_compact(_nm_total)}** that isn't "
        "moving — but it is **highly concentrated: the top 3 items account for "
        f"{_top3 / _nm_total:.0%}**. **{str(_lead['display_name'])} · "
        f"{theme.fmt_kes_compact(_lead_val)} · {_lead_val / _nm_total:.0%}** of trapped "
        "capital — the place to start (worth a physical-count check). Dead is a write-off / "
        "return decision; slow is a reduce-orders decision." + _susp_note
    )
    st.plotly_chart(charts.nonmoving_value_bars(nonmoving), use_container_width=True)
    with st.expander(f"View the list ({len(nonmoving):,} items)"):
        _label = {"dead": "Dead (365+ days)", "slow": "Slow (180–365 days)"}
        nonmoving["movement"] = nonmoving["movement_class"].map(_label)
        cols = [c for c in ["display_name", "product_category", "movement", "soh",
                            "days_since_last_dispense", "inventory_value"] if c in nonmoving.columns]
        st.dataframe(
            nonmoving[cols], use_container_width=True, hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Item"),
                "product_category": st.column_config.TextColumn("Category"),
                "movement": st.column_config.TextColumn("Movement"),
                "soh": st.column_config.NumberColumn("On hand", format="%.0f"),
                "days_since_last_dispense": st.column_config.NumberColumn("Days idle", format="%.0f"),
                "inventory_value": st.column_config.NumberColumn("Value tied up (KES)", format="%.0f"),
            },
        )

# ── 4. Surplus to release — held above what these items use ───────────────────

components.section_header("Surplus to release — held above what these items use")
release = (health[health["releasable"] == True].copy()  # noqa: E712
           if "releasable" in health.columns else health.iloc[0:0])
if release.empty:
    components.empty_state("Nothing is being held above its working requirement right now.")
else:
    receipts = data_access.item_max_receipt(data_access.selected_facility())
    release = data_access.classify_excess_cause(release, receipts)
    for c in ("excess_value", "excess_units", "working_requirement", "months_of_supply"):
        if c in release.columns:
            release[c] = pd.to_numeric(release[c], errors="coerce")
    excess_cap = release["excess_value"].sum(skipna=True)
    med_months = release["months_of_supply"].median(skipna=True)
    n_priced = int((pd.to_numeric(release["unit_price"], errors="coerce") > 0).sum())
    st.caption(
        f"**{len(release):,} active items** hold more than they'll use before their next "
        f"delivery — about **{theme.fmt_kes_compact(excess_cap)}** releasable"
        + (f", a median **{med_months:.0f} months** each" if pd.notna(med_months) else "")
        + ". Probable miscounts held back."
    )
    st.plotly_chart(charts.surplus_capital_split_bars(release), use_container_width=True)
    if n_priced < len(release):
        st.caption(f"Chart shows the {n_priced} priced items; the other "
                   f"{len(release) - n_priced} have no cost on record (see the list).")
    palette = charts.cause_palette()
    grouped = release.groupby("cause")
    tiles = []
    for key in ["demand_fell", "over_bought", "steady_overstock"]:
        if key not in grouped.groups:
            continue
        g = grouped.get_group(key)
        tiles.append({"label": data_access.EXCESS_CAUSE_LABELS[key],
                      "value": theme.fmt_kes_compact(g["excess_value"].sum(skipna=True)),
                      "detail": f"{len(g):,} items · {data_access.EXCESS_CAUSE_ACTIONS[key]}",
                      "accent": palette[key]})
    if tiles:
        components.kpi_row(tiles)
    with st.expander(f"View the list ({len(release):,} items)"):
        cols = [c for c in ["display_name", "product_category", "cause_label", "soh",
                            "months_of_supply", "excess_units", "excess_value", "cause_action"]
                if c in release.columns]
        st.dataframe(
            release.sort_values("excess_value", ascending=False, na_position="last")[cols].head(200),
            use_container_width=True, hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Item"),
                "product_category": st.column_config.TextColumn("Category"),
                "cause_label": st.column_config.TextColumn("Why it's surplus"),
                "soh": st.column_config.NumberColumn("On hand", format="%.0f"),
                "months_of_supply": st.column_config.NumberColumn("Months held", format="%.0f"),
                "excess_units": st.column_config.NumberColumn("Releasable (units)", format="%.0f"),
                "excess_value": st.column_config.NumberColumn("Releasable capital (KES)", format="%.0f"),
                "cause_action": st.column_config.TextColumn("What to do"),
            },
        )

# ── 5. Expiry watch ───────────────────────────────────────────────────────────

components.section_header("Expiry watch — use it or lose it")
expiry = data_access.with_names(data_access.load_table("expiry_risk"))
if expiry is None or expiry.empty:
    st.caption("No batch-expiry dates are recorded for this facility yet — this lights up "
               "when batch/expiry data is loaded.")
else:
    expiry = expiry.copy()
    for c in ["qty", "p_consumed_before_expiry", "days_to_expiry", "write_off_value"]:
        if c in expiry.columns:
            expiry[c] = pd.to_numeric(expiry[c], errors="coerce")
    at_risk = (expiry[expiry["p_consumed_before_expiry"] < 0.5]
               if "p_consumed_before_expiry" in expiry.columns else expiry)
    _wo = at_risk["write_off_value"].sum(skipna=True) if "write_off_value" in at_risk.columns else float("nan")
    _txt = (f"about **{theme.fmt_kes_compact(_wo)}** at risk of write-off"
            if pd.notna(_wo) and _wo > 0 else "nothing looks at risk right now")
    st.caption(
        f"**{len(at_risk):,} batch(es)** unlikely to be used before expiry — {_txt}. Only "
        "~1 in 20 lines has an expiry date, so treat as a watchlist, not a full ledger."
    )
    if not at_risk.empty:
        _sort = "days_to_expiry" if "days_to_expiry" in at_risk.columns else "p_consumed_before_expiry"
        at_risk = at_risk.sort_values(_sort)
        cols = [c for c in ["display_name", "batch", "qty", "expiry", "days_to_expiry", "write_off_value"]
                if c in at_risk.columns]
        st.dataframe(
            at_risk[cols].head(100), use_container_width=True, hide_index=True,
            column_config={
                "display_name": st.column_config.TextColumn("Item"),
                "batch": st.column_config.TextColumn("Batch / doc"),
                "qty": st.column_config.NumberColumn("On hand", format="%.0f"),
                "expiry": st.column_config.DatetimeColumn("Expires", format="DD MMM YYYY"),
                "days_to_expiry": st.column_config.NumberColumn("Days to expiry", format="%.0f"),
                "write_off_value": st.column_config.NumberColumn("Value at risk (KES)", format="%.0f"),
            },
        )

