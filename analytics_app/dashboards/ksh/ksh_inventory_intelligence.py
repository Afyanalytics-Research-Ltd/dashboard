"""
KSH Inventory Intelligence 
"""

import sys
import os
from pathlib import Path

ROOT = Path(os.path.abspath("analytics_app/dashboards/ksh/inventory_intelligence"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# ── Shared imports ─────────────────────────────────────────────────────────────
from intelligence import ai_client, order_intelligence
from intelligence.anomaly_engine import AnomalyEngine
from intelligence.briefing_writer import generate as write_briefing
from intelligence.config import DEFAULT_LEAD_TIME_DAYS, ABC_A_CUM_PCT, ABC_B_CUM_PCT
from intelligence.demand_engine import DemandEngine
from intelligence.lead_time_engine import LeadTimeEngine
from intelligence.priority_scorer import score_all, ORDER_NOW, ORDER_THIS_WEEK
from intelligence.priority_scorer import clinical_priority as get_clinical_priority
from intelligence import safety_stock as ss
from queries.core import (
    get_dispensing_history, get_kpi_summary, get_current_soh,
    get_dos_watchlist, get_dead_stock, get_deficit_dispenses,
    get_monthly_trends,
)
from queries.patient_risk import get_patient_risk_exposure, get_patient_risk_totals
from queries.receipts import get_kisumu_dispensing_for_lead_time
from utils.charts import (
    status_donut, dos_bar_chart, stockout_timeline,
    dead_stock_scatter, dispensing_trend, abc_pareto,
    anomaly_trend_chart,
)
from utils.components import (
    ai_summary_box, anomaly_banner, decision_card_ai,
    empty_state, inject_css, page_header, section_header, stat_strip,
)
from utils.facility import get_active_facility, set_active_facility, sql_ref_date
from utils.formatting import (
    ACTION_COLORS, CONFIDENCE_COLORS, DOS_COLORS,
    fmt_kes, fmt_kes_millions, fmt_int, fmt_days,
    fmt_drug_name, clean_drug_names,
    COLOR_RED,
)

# ── One-time setup ─────────────────────────────────────────────────────────────

inject_css()

fac = get_active_facility()
if fac is None:
    set_active_facility("kisumu")
    fac = get_active_facility()

_ref_date = sql_ref_date(fac)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    logo_path = str(ROOT / "ksh_logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
        st.markdown(
            "<hr style='margin:8px 0 6px;border:none;border-top:1px solid #E5E7EB'>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<div style='font-size:11px;color:#6B7280;padding:4px 0 12px'>"
        f"{'● Live' if fac.is_live else '◷ Historical'} · {fac.label}</div>",
        unsafe_allow_html=True,
    )

    page = option_menu(
        menu_title=None,
        options=[
            "Today's Briefing",
            "Order Workbench",
            "Stockout Watch",
            "Dead Stock",
            "Patient Risk",
            "Demand Insights",
            "Compliance Log",
        ],
        icons=[
            "clipboard2-pulse",
            "cart-check",
            "exclamation-triangle",
            "archive",
            "person-heart",
            "graph-up",
            "shield-check",
        ],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": "#0F6E56", "font-size": "13px"},
            "nav-link": {
                "font-size": "13px",
                "font-weight": "500",
                "color": "#374151",
                "padding": "8px 12px",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background-color": "#F0FAF6",
                "color": "#0F6E56",
                "font-weight": "700",
            },
        },
    )

    st.markdown(
        "<div style='font-size:10px;color:#9CA3AF;margin-top:16px;padding-top:8px;"
        "border-top:1px solid #E5E7EB'>Afyanalytics · Inventory Intelligence</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TODAY'S BRIEFING
# ══════════════════════════════════════════════════════════════════════════════

if page == "Today's Briefing":

    page_header(
        title="Today's Briefing",
        subtitle=f"Operational summary · {pd.Timestamp.now().strftime('%A, %d %b %Y')}",
        facility_label=fac.label,
        is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_load_dispensing(schema, ref_date):
        return get_dispensing_history(schema, days_back=730, ref_date=ref_date)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_load_kpis(schema, ref_date):
        df = get_kpi_summary(schema, ref_date=ref_date)
        return df.iloc[0].to_dict() if not df.empty else {}

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_load_soh(schema, ref_date):
        return get_current_soh(schema, ref_date=ref_date)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_compute_anomalies(dispensing_df, schema):
        if dispensing_df.empty:
            return pd.DataFrame()
        engine = AnomalyEngine(schema)
        engine.fit(dispensing_df)
        return engine.detect_all()

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_compute_unit_values(dispensing_df):
        df = dispensing_df.copy()
        df.columns = df.columns.str.lower()
        if "line_total" not in df.columns or "quantity_dispensed" not in df.columns:
            return {}
        grouped = df.groupby("canonical_name").agg(
            total_value=("line_total", "sum"),
            total_qty=("quantity_dispensed", "sum"),
        ).reset_index()
        grouped = grouped[grouped["total_qty"] > 0]
        grouped["unit_value"] = grouped["total_value"] / grouped["total_qty"]
        return dict(zip(grouped["canonical_name"], grouped["unit_value"]))

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_compute_demand_forecasts(dispensing_df):
        if dispensing_df.empty:
            return {}
        engine = DemandEngine("briefing")
        engine.fit(dispensing_df)
        forecasts_df = engine.forecast_all(dispensing_df)
        if forecasts_df.empty:
            return {}
        result = {}
        for _, row in forecasts_df.iterrows():
            name = str(row.get("canonical_name") or "").strip()
            if name:
                result[name] = {
                    "avg_daily_units": float(row.get("avg_daily_units", 0)),
                    "std_daily_units": float(row.get("std_daily_units", 0)),
                    "trend_direction": str(row.get("trend_direction", "STABLE")),
                    "confidence":      str(row.get("confidence", "LOW")),
                }
        return result

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_compute_lead_time_map(schema, soh_df):
        lt_eng = LeadTimeEngine(schema)
        try:
            lt_disp = get_kisumu_dispensing_for_lead_time()
            if not lt_disp.empty:
                lt_eng.fit_kisumu(lt_disp)
        except Exception:
            pass
        lt_df = lt_eng.get_all_product_lead_times()
        fac_mean, _, _ = lt_eng.get_lead_time()
        _soh = soh_df.copy()
        _soh.columns = _soh.columns.str.lower()
        if "product_id" not in _soh.columns or "canonical_name" not in _soh.columns:
            return {}
        if lt_df.empty:
            return {name: fac_mean for name in _soh["canonical_name"].dropna().unique()}
        merged = _soh[["product_id", "canonical_name"]].merge(lt_df, on="product_id", how="left")
        merged["lt_mean"] = merged["lt_mean"].fillna(fac_mean)
        return dict(zip(merged["canonical_name"], merged["lt_mean"]))

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_generate_decision_briefs(cache_key, top_drugs):
        briefs = []
        for d in top_drugs:
            brief = order_intelligence.generate(
                canonical_name=d["canonical_name"],
                dos_remaining=d["dos_remaining"],
                avg_daily_units=d["avg_daily_units"],
                current_soh=d["current_soh"],
                order_qty=d["order_qty"],
                target_cover_days=30,
                clinical_priority=d.get("clinical_priority", "STANDARD"),
                therapeutic_class=d.get("therapeutic_class", ""),
                patients_at_risk=d.get("patients_at_risk", 0),
                trend_direction=d.get("trend_direction", "STABLE"),
                confidence=d.get("confidence", "MEDIUM"),
                lead_time_days=d.get("lead_time_days", DEFAULT_LEAD_TIME_DAYS),
                avg_unit_value_kes=d.get("avg_unit_value_kes"),
            )
            briefs.append(brief.__dict__)
        return briefs

    with st.spinner("Loading facility data…"):
        dispensing_df = _brief_load_dispensing(fac.schema, _ref_date)
        kpi           = _brief_load_kpis(fac.schema, _ref_date)
        soh_df        = _brief_load_soh(fac.schema, _ref_date)

    if dispensing_df.empty:
        empty_state("No dispensing data found for this facility.", icon="📭")
        st.stop()

    soh_df.columns = soh_df.columns.str.upper()

    if "CANONICAL_NAME" in soh_df.columns:
        _has_name = soh_df["CANONICAL_NAME"].notna() & (soh_df["CANONICAL_NAME"].str.strip() != "")
        soh_df = soh_df[_has_name].copy()
        soh_df["_soh_sort"] = pd.to_numeric(soh_df["CURRENT_SOH"], errors="coerce").fillna(0)
        soh_df["_dos_sort"] = pd.to_numeric(soh_df.get("DAYS_OF_STOCK"), errors="coerce").fillna(9999)
        soh_df = (
            soh_df
            .sort_values(["_soh_sort", "_dos_sort"])
            .drop_duplicates(subset=["CANONICAL_NAME"], keep="first")
            .drop(columns=["_soh_sort", "_dos_sort"])
            .reset_index(drop=True)
        )

    with st.spinner("Running intelligence engines…"):
        actions_df   = score_all(soh_df)
        anomalies_df = _brief_compute_anomalies(dispensing_df, fac.schema)

    _stockouts = int(kpi.get("ACTIVE_STOCKOUTS") or 0)
    _critical  = int(kpi.get("CRITICAL_COUNT") or 0)
    _low       = int(kpi.get("LOW_COUNT") or 0)
    _total     = int(kpi.get("TOTAL_PRODUCTS") or 0)

    briefing_text = write_briefing(
        facility_name=fac.short,
        kpi_row=kpi,
        actions_df=actions_df,
        anomalies_df=anomalies_df if not anomalies_df.empty else None,
    )

    _at_risk = _stockouts + _critical
    _patients_total = int(kpi.get("CHRONIC_PATIENTS_ACTIVE") or 0) + int(kpi.get("OPIOID_PATIENTS_ACTIVE") or 0)
    _anom_parts = []
    if not anomalies_df.empty:
        for _, _ar in anomalies_df.head(2).iterrows():
            _sign = "+" if _ar["direction"] == "UP" else ""
            _anom_parts.append(f"{_ar['canonical_name']} ({_sign}{_ar['magnitude_pct']:.0f}%)")

    _headline_parts = [f"{_at_risk} items at immediate risk"]
    if _patients_total:
        _headline_parts.append(f"{_patients_total:,} patients affected")
    if _anom_parts:
        _headline_parts.append(f"Anomalies: {', '.join(_anom_parts)}")

    st.markdown(
        f'<div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;'
        f'padding:12px 16px;margin-bottom:4px">'
        f'<span style="font-size:13px;font-weight:600;color:#166534">'
        f'{" · ".join(_headline_parts)}</span></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Situation details →"):
        ai_summary_box(briefing_text)

    stat_strip([
        {"label": "Stocked out",         "value": fmt_int(_stockouts),
         "hint": "Immediate action" if _stockouts else "None", "hint_good": _stockouts == 0,
         "accent_color": "#991B1B" if _stockouts else "#111827"},
        {"label": "Critical  < 7d",      "value": fmt_int(_critical),
         "hint": "Order now" if _critical else "Clear",        "hint_good": _critical == 0,
         "accent_color": "#DC2626" if _critical else "#111827"},
        {"label": "Low  7–30d",          "value": fmt_int(_low),
         "hint": "Monitor" if _low else "Clear",               "hint_good": _low == 0,
         "accent_color": "#D97706" if _low else "#111827"},
        {"label": "Total products",      "value": fmt_int(_total),          "accent_color": "#111827"},
        {"label": "30d dispensing value","value": fmt_kes_millions(kpi.get("TOTAL_DISPENSING_VALUE_30D")), "accent_color": "#111827"},
        {"label": "Chronic patients",    "value": fmt_int(kpi.get("CHRONIC_PATIENTS_ACTIVE")),            "accent_color": "#111827"},
    ])

    left, right = st.columns([1.1, 1], gap="large")

    with right:
        _adequate = max(0, _total - _stockouts - _critical - _low)
        _status_counts = {}
        if _stockouts: _status_counts["stockout"] = _stockouts
        if _critical:  _status_counts["critical"] = _critical
        if _low:       _status_counts["low"]      = _low
        if _adequate:  _status_counts["adequate"] = _adequate

        if _status_counts:
            st.plotly_chart(status_donut(_status_counts), use_container_width=True)
        else:
            empty_state("Stock status data unavailable.")

        if not anomalies_df.empty:
            section_header("Consumption anomalies")
            from intelligence.anomaly_engine import build_anomaly_context
            from intelligence.order_intelligence import analyse_anomaly

            for _, row in anomalies_df.head(4).iterrows():
                _anom_name = str(row.get("canonical_name") or "").strip()
                if not _anom_name or _anom_name.lower() == "nan":
                    continue
                _anom_key = f"anomaly_v2_{fac.schema}_{_anom_name}"
                anomaly_banner(_anom_name, row["message"])

                _stored = st.session_state.get(_anom_key)
                if _stored:
                    _ctx, _action, _is_ai = _stored
                    _d = _ctx["days_at_current_rate"]
                    _n = _ctx["days_at_normal_rate"]
                    _spike_type = _ctx.get("spike_type", "SUSTAINED")
                    _type_styles = {
                        "SUSTAINED": ("background:#FEE2E2;border-color:#FECACA;color:#991B1B", "● SUSTAINED"),
                        "TRANSIENT": ("background:#FEF3C7;border-color:#FDE68A;color:#92400E", "◎ TRANSIENT"),
                        "DECLINING": ("background:#EFF6FF;border-color:#BFDBFE;color:#1D4ED8", "↘ DECLINING"),
                    }
                    _ts, _tl = _type_styles.get(_spike_type, _type_styles["SUSTAINED"])
                    _type_chip = (
                        f"<span style='display:inline-block;border:1px solid;border-radius:4px;"
                        f"padding:2px 8px;font-size:10px;font-weight:700;margin:0 4px 4px 0;{_ts}'>{_tl}</span>"
                    )
                    _cs = "display:inline-block;background:#F3F4F6;border:1px solid #E5E7EB;border-radius:4px;padding:2px 8px;font-size:10px;font-weight:600;color:#374151;margin:0 4px 4px 0"
                    _spike_txt = _ctx["spike_start"].strftime("Started %d %b") if _ctx["spike_start"] else "Onset: last 14d"
                    _stock_txt = (f"Stock: {_d}d at current rate vs {_n}d normal" if _d is not None and _n is not None else f"SOH: {_ctx['current_soh']:.0f} units")
                    _safe_txt  = f"Safe ADC: {_ctx['safe_order_adc']:.1f} u/day" if _ctx.get("safe_order_adc") else ""
                    _corr_txt  = f"Also ↑: {', '.join(_ctx['correlated_drugs'])}" if _ctx["correlated_drugs"] else ""
                    chips_html = _type_chip + "".join(f"<span style='{_cs}'>{t}</span>" for t in [_spike_txt, _stock_txt, _safe_txt, _corr_txt] if t)
                    st.markdown(f"<div style='margin:4px 0 6px'>{chips_html}</div>", unsafe_allow_html=True)
                    st.plotly_chart(
                        anomaly_trend_chart(
                            daily=_ctx["daily_series"], recent_start=_ctx["recent_start"],
                            baseline_avg=_ctx["baseline_avg"], recent_avg=_ctx["recent_avg"],
                            spike_start=_ctx.get("spike_start"),
                        ),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                    _ai_label = "✦ AI" if _is_ai else "○ Rule-based"
                    st.markdown(
                        f"<div class='anomaly-analysis'><div class='anomaly-analysis-label'>"
                        f"{_ai_label} &nbsp;Recommended action</div>{_action}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("↩ Collapse", key=f"clear_{_anom_key}", type="secondary"):
                        del st.session_state[_anom_key]
                        st.rerun()
                else:
                    if st.button("✦ Analyse", key=f"analyse_{_anom_key}", type="secondary"):
                        with st.spinner(f"Analysing {_anom_name}…"):
                            _ctx = build_anomaly_context(
                                dispensing_df=dispensing_df, soh_df=soh_df,
                                anomaly_row=row.to_dict(), anomalies_df=anomalies_df,
                            )
                            _action, _is_ai = analyse_anomaly(_ctx)
                        st.session_state[_anom_key] = (_ctx, _action, _is_ai)
                        st.rerun()
                st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='font-size:11px;color:#9CA3AF;margin-top:8px'>"
            f"Last run: {pd.Timestamp.now().strftime('%H:%M')} · Cached 1h</div>",
            unsafe_allow_html=True,
        )

    with left:
        _order_now_df = (
            actions_df[actions_df["action"] == ORDER_NOW].sort_values("urgency_score", ascending=False)
            if not actions_df.empty else pd.DataFrame()
        )
        _n_urgent = len(_order_now_df)
        section_header("Decisions needed today")

        if _order_now_df.empty:
            empty_state("No urgent actions — stock levels are healthy.", icon="✅")
        else:
            _soh_l = soh_df.copy()
            _soh_l.columns = _soh_l.columns.str.lower()
            _dos_col = "days_of_stock_p50" if "days_of_stock_p50" in _soh_l.columns else "days_of_stock"
            _soh_for_merge = _soh_l[["canonical_name", "current_soh", _dos_col, "therapeutic_class"]].rename(columns={_dos_col: "dos_remaining"})
            _merged = _order_now_df.head(5).merge(_soh_for_merge, on="canonical_name", how="left")

            _unit_values = _brief_compute_unit_values(dispensing_df)
            _demand_map  = _brief_compute_demand_forecasts(dispensing_df)
            _lt_map      = _brief_compute_lead_time_map(fac.schema, soh_df)

            _top_drugs_data = []
            for _, r in _merged.iterrows():
                soh_val = float(r.get("current_soh") or 0)
                dos_val = float(r.get("dos_remaining") or 0)
                name    = str(r.get("canonical_name", "Unknown"))
                lead_t  = max(1, round(_lt_map.get(name, DEFAULT_LEAD_TIME_DAYS)))
                _fc     = _demand_map.get(name, {})
                adc     = _fc.get("avg_daily_units", 0.0)
                qty     = max(0, int((30 + lead_t) * adc - soh_val))
                _top_drugs_data.append({
                    "canonical_name":     name,
                    "dos_remaining":      dos_val,
                    "avg_daily_units":    adc,
                    "current_soh":        soh_val,
                    "order_qty":          qty,
                    "lead_time_days":     lead_t,
                    "clinical_priority":  str(r.get("clinical_priority", "STANDARD")),
                    "therapeutic_class":  str(r.get("therapeutic_class", "") or ""),
                    "trend_direction":    _fc.get("trend_direction", "STABLE"),
                    "confidence":         _fc.get("confidence", "LOW"),
                    "avg_unit_value_kes": _unit_values.get(name),
                })

            _cache_key = f"{fac.schema}|{','.join(d['canonical_name'] for d in _top_drugs_data)}"
            _briefs = []
            if _top_drugs_data:
                with st.spinner("Generating AI recommendations…"):
                    _briefs = _brief_generate_decision_briefs(cache_key=_cache_key, top_drugs=_top_drugs_data)

            for brief_dict, drug_data in zip(_briefs, _top_drugs_data):
                decision_card_ai(
                    canonical_name=brief_dict["canonical_name"],
                    action=ORDER_NOW,
                    dos_remaining=drug_data["dos_remaining"],
                    order_qty=brief_dict["recommended_qty"],
                    cost_estimate_kes=brief_dict.get("cost_estimate_kes"),
                    stockout_gap_days=brief_dict["stockout_gap_days"],
                    narrative=brief_dict["narrative"],
                    is_ai=brief_dict["is_ai"],
                    color=ACTION_COLORS.get(ORDER_NOW, COLOR_RED),
                )

            if _n_urgent > 5:
                st.markdown(
                    f"<div style='font-size:12px;color:#6B7280;margin:4px 0 8px'>"
                    f"Showing top 5 of {_n_urgent} urgent items</div>",
                    unsafe_allow_html=True,
                )
            st.caption("→ Switch to **Order Workbench** in the sidebar for full reorder details")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ORDER WORKBENCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Order Workbench":

    with st.sidebar:
        st.subheader("Reorder settings")
        service_level = st.select_slider(
            "Service level", options=[0.90, 0.95, 0.99],
            value=st.session_state.get("service_level", 0.95),
            format_func=lambda x: f"{int(x*100)}%",
        )
        st.session_state["service_level"] = service_level
        lead_time_override = st.number_input(
            "Lead time override (days)", min_value=0, max_value=90,
            value=st.session_state.get("lead_time_override", 0),
            help="Set to 0 to use AI-inferred lead time.",
        )
        st.session_state["lead_time_override"] = lead_time_override
        target_cover = st.slider(
            "Target cover (days)", min_value=14, max_value=90,
            value=st.session_state.get("target_cover", 30),
        )
        st.session_state["target_cover"] = target_cover
        st.caption("Reorder qty covers target days + lead time, minus current SOH.")
        st.markdown("---")
        st.subheader("Filters")
        filter_priority = st.multiselect(
            "Clinical priority", ["CRITICAL", "HIGH", "STANDARD"],
            default=["CRITICAL", "HIGH", "STANDARD"],
        )
        filter_action = st.radio("Action type", ["All", ORDER_NOW, ORDER_THIS_WEEK], index=0)

    page_header(
        title="Order Workbench",
        subtitle="AI-calculated reorder quantities based on your consumption history",
        facility_label=fac.label, is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _wb_build_engines(schema, ref_date):
        disp = get_dispensing_history(schema, days_back=730, ref_date=ref_date)
        soh  = get_current_soh(schema, ref_date=ref_date)
        lt_disp = get_kisumu_dispensing_for_lead_time()
        demand_eng = DemandEngine(schema).fit(disp)
        forecasts_df = demand_eng.forecast_all(disp)
        lt_eng = LeadTimeEngine(schema)
        if not lt_disp.empty:
            lt_eng.fit_kisumu(lt_disp)
        return forecasts_df, lt_eng.get_all_product_lead_times(), soh, lt_eng

    with st.spinner("Building demand and lead time models…"):
        forecasts_df, lt_df, soh_df, lt_eng = _wb_build_engines(fac.schema, _ref_date)

    if forecasts_df.empty:
        empty_state("Not enough dispensing history to generate recommendations.", icon="📊")
        st.stop()

    rec = forecasts_df.copy()
    rec.columns = rec.columns.str.upper()

    if not soh_df.empty:
        rec = rec.merge(
            soh_df[["PRODUCT_ID", "CURRENT_SOH_DISPLAY", "CURRENT_SOH",
                    "THERAPEUTIC_CLASS", "THERAPEUTIC_SUBCLASS"]].copy(),
            on="PRODUCT_ID", how="left"
        )
    else:
        for col in ["CURRENT_SOH_DISPLAY", "CURRENT_SOH", "THERAPEUTIC_CLASS", "THERAPEUTIC_SUBCLASS"]:
            rec[col] = 0 if "SOH" in col else ""

    if not lt_df.empty:
        lt_up = lt_df.copy()
        lt_up.columns = lt_up.columns.str.upper()
        rec = rec.merge(lt_up, on="PRODUCT_ID", how="left")

    fac_lt_mean, fac_lt_std, _ = lt_eng.get_lead_time()

    if "LT_MEAN" not in rec.columns:
        rec["LT_MEAN"] = fac_lt_mean
        rec["LT_STD"]  = fac_lt_std
    else:
        override_val = float(lead_time_override) if lead_time_override > 0 else fac_lt_mean
        rec["LT_MEAN"] = rec["LT_MEAN"].fillna(override_val)
        rec["LT_STD"]  = rec["LT_STD"].fillna(fac_lt_std)

    if lead_time_override > 0:
        rec["LT_MEAN"] = float(lead_time_override)

    def _safe(v, default=0.0):
        try:
            f = float(v)
            return f if pd.notna(f) else default
        except (TypeError, ValueError):
            return default

    def _compute_row(row):
        lt_m = _safe(row.get("LT_MEAN"), fac_lt_mean)
        lt_s = _safe(row.get("LT_STD"),  fac_lt_std)
        adc  = _safe(row.get("AVG_DAILY_UNITS"))
        std  = _safe(row.get("STD_DAILY_UNITS"), adc * 0.3)
        soh  = _safe(row.get("CURRENT_SOH_DISPLAY"))
        return pd.Series({
            "rop":           round(ss.reorder_point(adc, lt_m, std, lt_s, service_level), 0),
            "safety_stock":  round(ss.safety_stock(std, lt_m, lt_s, adc, service_level), 0),
            "order_qty":     round(ss.recommended_order_quantity(soh, ss.reorder_point(adc, lt_m, std, lt_s, service_level), adc, lt_m, target_cover), 0),
            "dos_remaining": ss.days_of_stock(soh, adc),
            "needs_order":   soh <= ss.reorder_point(adc, lt_m, std, lt_s, service_level),
            "lt_mean_used":  round(lt_m, 1),
        })

    computed = rec.apply(_compute_row, axis=1)
    rec = pd.concat([rec, computed], axis=1)
    rec["CLINICAL_PRIORITY"] = rec.apply(
        lambda r: get_clinical_priority(
            str(r.get("CANONICAL_NAME", "") or ""),
            str(r.get("THERAPEUTIC_CLASS", "") or ""),
            str(r.get("THERAPEUTIC_SUBCLASS", "") or ""),
        ), axis=1,
    )
    rec["ACTION"] = rec["dos_remaining"].apply(
        lambda d: ORDER_NOW if (d is None or pd.isna(d) or d < 7) else ORDER_THIS_WEEK
    )

    n_order_now  = int((rec["dos_remaining"].fillna(999) < 7).sum())
    n_order_week = int(((rec["dos_remaining"].fillna(999) >= 7) & rec["needs_order"]).sum())
    total_qty    = int(rec.loc[rec["needs_order"], "order_qty"].sum())

    stat_strip([
        {"label": "Order now (<7d)",      "value": str(n_order_now),  "hint": "Immediate",
         "hint_good": n_order_now == 0,   "accent_color": "#991B1B" if n_order_now else "#111827"},
        {"label": "Order this week",      "value": str(n_order_week),
         "accent_color": "#D97706" if n_order_week else "#111827"},
        {"label": "Total units to order", "value": fmt_int(total_qty)},
        {"label": "Service level target", "value": f"{int(service_level*100)}%"},
        {"label": "Lead time (used)",     "value": fmt_days(lead_time_override if lead_time_override > 0 else fac_lt_mean)},
    ])

    section_header("Reorder recommendations")
    display = rec[rec["needs_order"] | (rec["dos_remaining"].fillna(999) < 7)].copy()
    display = display.sort_values("dos_remaining", na_position="first").reset_index(drop=True)

    if filter_priority:
        display = display[display["CLINICAL_PRIORITY"].isin(filter_priority)]
    if filter_action != "All":
        display = display[display["ACTION"] == filter_action]

    if display.empty:
        empty_state("No products match the current filters.", icon="✅")
        st.stop()

    table_df = pd.DataFrame({
        "Drug":          display["CANONICAL_NAME"].map(fmt_drug_name).values,
        "Action":        display["ACTION"].values,
        "Priority":      display["CLINICAL_PRIORITY"].values,
        "Class":         display["THERAPEUTIC_CLASS"].fillna("—").values,
        "DOS remaining": display["dos_remaining"].values,
        "Current SOH":   display["CURRENT_SOH_DISPLAY"].values,
        "Avg daily (u)": display["AVG_DAILY_UNITS"].values,
        "Order qty":     display["order_qty"].values,
        "Confidence":    display["CONFIDENCE"].values,
    })

    st.caption(f"**{len(table_df):,} products** need ordering · Click a row for formula breakdown")
    event = st.dataframe(
        table_df, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "Drug":          st.column_config.TextColumn(width="large"),
            "DOS remaining": st.column_config.NumberColumn(format="%.1f d"),
            "Current SOH":   st.column_config.NumberColumn(format="%,.0f"),
            "Avg daily (u)": st.column_config.NumberColumn(format="%.2f"),
            "Order qty":     st.column_config.NumberColumn(format="%,.0f"),
        },
    )

    selected = event.selection.rows if hasattr(event, "selection") else []
    if selected:
        row = display.iloc[selected[0]]
        adc  = _safe(row.get("AVG_DAILY_UNITS"))
        conf = str(row.get("CONFIDENCE", "LOW")).upper()
        conf_color = CONFIDENCE_COLORS.get(conf, "#888780")
        cp_color   = {"CRITICAL": "#A32D2D", "HIGH": "#854F0B", "STANDARD": "#0C447C"}.get(
            str(row.get("CLINICAL_PRIORITY", "")), "#6B7280"
        )
        drug_name = str(row.get("CANONICAL_NAME", "—"))

        st.markdown(
            f'<div style="background:#F0FAF6;border:1px solid #0F6E56;border-left:4px solid #0F6E56;'
            f'border-radius:0 10px 10px 0;padding:16px 20px;margin-top:8px">'
            f'<div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;'
            f'color:#0F6E56;margin-bottom:6px">Formula breakdown</div>'
            f'<div style="font-size:15px;font-weight:700;color:#1A1A2E;margin-bottom:12px">{drug_name}'
            f'<span style="font-size:11px;font-weight:700;padding:2px 8px;border-radius:4px;'
            f'color:#fff;background:{cp_color};margin-left:8px">{row.get("CLINICAL_PRIORITY","—")}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current SOH",   fmt_int(row.get("CURRENT_SOH_DISPLAY")))
        m2.metric("Reorder point", fmt_int(row.get("rop")))
        m3.metric("Safety stock",  fmt_int(row.get("safety_stock")))
        m4.metric("Lead time",     f"{row['lt_mean_used']:.0f}d")
        m5.metric("Order qty",     fmt_int(row.get("order_qty")))

        cover_units = (target_cover + row["lt_mean_used"]) * adc
        st.markdown(
            f'<div style="background:#F5F6FA;border-radius:8px;padding:12px 16px;'
            f'font-size:13px;margin-top:4px;line-height:1.8">'
            f'<b>Target:</b> {target_cover}d cover + {row["lt_mean_used"]:.0f}d lead time = {fmt_int(cover_units)} units<br>'
            f'<b>Safety stock</b> ({int(service_level*100)}% SL): {fmt_int(row["safety_stock"])} units<br>'
            f'<b>Current SOH:</b> {fmt_int(row.get("CURRENT_SOH_DISPLAY", 0))} → <b>Order: {fmt_int(row["order_qty"])} units</b><br>'
            f'<span style="color:{conf_color};font-weight:600">Confidence: {conf} · {row.get("DATA_MONTHS","?")} months data</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _brief_key = f"ai_brief_wb_{str(row.get('CANONICAL_NAME', ''))}"
        _col_ai, _ = st.columns([1, 2])
        with _col_ai:
            if st.button("✦ Generate AI reasoning", key=f"ai_btn_{selected[0]}", use_container_width=True):
                with st.spinner("Generating clinical recommendation…"):
                    _brief = order_intelligence.generate(
                        canonical_name=str(row.get("CANONICAL_NAME", "—")),
                        dos_remaining=_safe(row.get("dos_remaining"), 999),
                        avg_daily_units=_safe(row.get("AVG_DAILY_UNITS")),
                        current_soh=_safe(row.get("CURRENT_SOH_DISPLAY")),
                        order_qty=int(_safe(row.get("order_qty"))),
                        target_cover_days=target_cover,
                        clinical_priority=str(row.get("CLINICAL_PRIORITY", "STANDARD")),
                        therapeutic_class=str(row.get("THERAPEUTIC_CLASS", "") or ""),
                        patients_at_risk=0,
                        trend_direction="STABLE",
                        confidence=str(row.get("CONFIDENCE", "MEDIUM")),
                        lead_time_days=int(row.get("lt_mean_used", 7)),
                    )
                    st.session_state[_brief_key] = _brief

        if _brief_key in st.session_state:
            _b = st.session_state[_brief_key]
            st.markdown(
                f'<div style="background:#F0FAF6;border:1px solid #C3E8D8;border-left:3px solid #0F6E56;'
                f'border-radius:0 8px 8px 0;padding:12px 16px;margin-top:4px;font-size:13px;'
                f'line-height:1.65;color:#111827">'
                f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;'
                f'color:#0F6E56;margin-bottom:6px">{"✦ AI recommendation" if _b.is_ai else "Recommendation"}</div>'
                f'{_b.narrative}</div>',
                unsafe_allow_html=True,
            )
            if _b.stockout_gap_days > 0:
                st.warning(
                    f"⚠ {_b.stockout_gap_days}-day stockout gap — stock runs out before order arrives "
                    f"at estimated {_b.lead_time_days}-day lead time."
                )

    st.markdown("---")
    export_cols = {
        "CANONICAL_NAME": "Drug", "THERAPEUTIC_CLASS": "Class", "CLINICAL_PRIORITY": "Priority",
        "ACTION": "Action", "CURRENT_SOH_DISPLAY": "Current SOH", "AVG_DAILY_UNITS": "Avg daily (u)",
        "dos_remaining": "DOS remaining", "rop": "Reorder point", "safety_stock": "Safety stock",
        "order_qty": "Order qty", "CONFIDENCE": "Confidence",
    }
    export_df = display[[c for c in export_cols if c in display.columns]].rename(columns=export_cols)
    st.download_button(
        "📥 Export order list (CSV)", export_df.to_csv(index=False),
        file_name=f"order_list_{fac.schema}_{pd.Timestamp.now().date()}.csv",
        mime="text/csv", use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STOCKOUT WATCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Stockout Watch":

    with st.sidebar:
        show_filter  = st.selectbox("Show", ["Red + Amber", "Red only", "All products"], index=0)
        top_n        = st.slider("Products in chart", 10, 40, 20)
        window_days  = st.select_slider("Consumption window", [30, 60, 90, 180], value=90)
        st.caption("Consumption window affects daily average and DOS calculations.")

    page_header(
        title="Stockout Watch",
        subtitle="Days-of-stock watchlist with probabilistic depletion forecasts",
        facility_label=fac.label, is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _sw_load(schema, window, ref_date):
        return get_dos_watchlist(schema, window, ref_date=ref_date)

    with st.spinner("Loading watchlist…"):
        df = _sw_load(fac.schema, window_days, _ref_date)

    if df.empty:
        empty_state("No stock data available.", icon="📭")
        st.stop()

    df.columns = df.columns.str.upper()
    df = clean_drug_names(df)

    if show_filter == "Red only":
        view = df[df["DOS_STATUS"] == "red"]
    elif show_filter == "Red + Amber":
        view = df[df["DOS_STATUS"].isin(["red", "amber"])]
    else:
        view = df.copy()

    red_count   = int((df["DOS_STATUS"] == "red").sum())
    amber_count = int((df["DOS_STATUS"] == "amber").sum())
    zero_soh    = int((df.get("CURRENT_SOH", pd.Series([0])) <= 0).sum())
    avg_dos     = df["DAYS_OF_STOCK_P50"].dropna()
    avg_dos_val = avg_dos.mean() if not avg_dos.empty else None

    stat_strip([
        {"label": "Stocked out (SOH ≤ 0)", "value": fmt_int(zero_soh),
         "hint": "PPB risk",  "hint_good": zero_soh == 0,
         "accent_color": "#991B1B" if zero_soh else "#111827"},
        {"label": "Critical (<7d)",         "value": fmt_int(red_count),
         "hint": "Order now", "hint_good": red_count == 0,
         "accent_color": "#DC2626" if red_count else "#111827"},
        {"label": "Low stock (7–30d)",      "value": fmt_int(amber_count),
         "accent_color": "#D97706" if amber_count else "#111827"},
        {"label": "Average DOS (facility)", "value": fmt_days(avg_dos_val)},
        {"label": "Products assessed",      "value": fmt_int(len(df))},
    ])

    left, right = st.columns([1.4, 1])
    with left:
        section_header(f"Top {top_n} most at-risk products")
        if not view.empty:
            st.plotly_chart(dos_bar_chart(view, top_n=top_n), use_container_width=True)
        else:
            empty_state("No products in this filter.", icon="✅")

    with right:
        section_header("Predicted stockout dates")
        if not view.empty:
            pred_cols = ["CANONICAL_NAME", "DOS_STATUS", "DAYS_OF_STOCK_P50", "DAYS_OF_STOCK_P90",
                         "PREDICTED_STOCKOUT_P50", "PREDICTED_STOCKOUT_P90", "STOCKOUT_EPISODE_COUNT"]
            avail = [c for c in pred_cols if c in view.columns]
            st.dataframe(
                view[avail].copy().rename(columns={
                    "CANONICAL_NAME": "Drug", "DOS_STATUS": "Status",
                    "DAYS_OF_STOCK_P50": "DOS (avg)", "DAYS_OF_STOCK_P90": "DOS (high demand)",
                    "PREDICTED_STOCKOUT_P50": "Stockout by (avg)", "PREDICTED_STOCKOUT_P90": "Stockout by (high)",
                    "STOCKOUT_EPISODE_COUNT": "Past episodes",
                }).head(30),
                use_container_width=True, hide_index=True,
                column_config={
                    "Stockout by (avg)":  st.column_config.DateColumn(format="DD MMM YYYY"),
                    "Stockout by (high)": st.column_config.DateColumn(format="DD MMM YYYY"),
                    "DOS (avg)":          st.column_config.NumberColumn(format="%.1f d"),
                    "DOS (high demand)":  st.column_config.NumberColumn(format="%.1f d"),
                },
            )

    section_header("Full watchlist")
    search = st.text_input("Search drug name", placeholder="e.g. Metformin", label_visibility="collapsed")
    table_df = view.copy()
    if search:
        table_df = table_df[table_df["CANONICAL_NAME"].str.contains(search, case=False, na=False)]

    st.dataframe(
        table_df[[c for c in [
            "CANONICAL_NAME", "THERAPEUTIC_CLASS", "CURRENT_SOH_DISPLAY",
            "AVG_DAILY_UNITS", "DAYS_OF_STOCK_P50", "DOS_STATUS",
            "PREDICTED_STOCKOUT_P50", "STOCKOUT_EPISODE_COUNT",
        ] if c in table_df.columns]].rename(columns={
            "CANONICAL_NAME": "Drug", "THERAPEUTIC_CLASS": "Class",
            "CURRENT_SOH_DISPLAY": "SOH", "AVG_DAILY_UNITS": "Avg daily (u)",
            "DAYS_OF_STOCK_P50": "Days remaining", "DOS_STATUS": "Status",
            "PREDICTED_STOCKOUT_P50": "Stockout by", "STOCKOUT_EPISODE_COUNT": "# Stockouts",
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "Days remaining": st.column_config.NumberColumn(format="%.1f d"),
            "Avg daily (u)":  st.column_config.NumberColumn(format="%.1f"),
            "Stockout by":    st.column_config.DateColumn(format="DD MMM YYYY"),
        },
    )
    st.download_button(
        "📥 Export watchlist (CSV)", table_df.to_csv(index=False),
        file_name=f"dos_watchlist_{fac.schema}_{pd.Timestamp.now().date()}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEAD STOCK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Dead Stock":

    with st.sidebar:
        idle_threshold = st.slider(
            "Idle threshold (days)", 14, 90, 30,
            help="Products with no dispense activity beyond this threshold are flagged.",
        )

    page_header(
        title="Dead Stock Actions",
        subtitle="Idle inventory · Dead stock identification",
        facility_label=fac.label, is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _ds_load(schema, threshold, ref_date):
        return get_dead_stock(schema, threshold, ref_date=ref_date)

    with st.spinner("Loading dead stock data…"):
        dead_df = _ds_load(fac.schema, idle_threshold, _ref_date)

    dead_df.columns = dead_df.columns.str.upper()

    dead_only  = dead_df[dead_df.get("IDLE_CATEGORY", pd.Series(dtype=str)) == "dead"] if not dead_df.empty else pd.DataFrame()
    slow_only  = dead_df[dead_df.get("IDLE_CATEGORY", pd.Series(dtype=str)) == "slow"] if not dead_df.empty else pd.DataFrame()
    dead_value = dead_only["TOTAL_HISTORICAL_VALUE"].sum() if not dead_only.empty and "TOTAL_HISTORICAL_VALUE" in dead_only.columns else 0
    slow_value = slow_only["TOTAL_HISTORICAL_VALUE"].sum() if not slow_only.empty and "TOTAL_HISTORICAL_VALUE" in slow_only.columns else 0

    stat_strip([
        {"label": "Dead stock SKUs (90d+)", "value": fmt_int(len(dead_only)),
         "hint": "Capital at risk", "hint_good": len(dead_only) == 0,
         "accent_color": "#991B1B" if len(dead_only) else "#111827"},
        {"label": "Slow moving (30–90d)", "value": fmt_int(len(slow_only)),
         "accent_color": "#D97706" if len(slow_only) else "#111827"},
        {"label": "Dead stock value",   "value": fmt_kes_millions(dead_value)},
        {"label": "Slow moving value",  "value": fmt_kes_millions(slow_value)},
        {"label": "Total idle SKUs",    "value": fmt_int(len(dead_df))},
    ])

    left, right = st.columns([1.3, 1])
    with left:
        section_header("Idle inventory map")
        if not dead_df.empty:
            st.plotly_chart(dead_stock_scatter(dead_df), use_container_width=True)
        else:
            empty_state(f"No idle products beyond {idle_threshold} days.", icon="✅")

    with right:
        section_header("Dead stock detail")
        if not dead_df.empty:
            table_cols = [c for c in [
                "CANONICAL_NAME", "THERAPEUTIC_CLASS", "DAYS_IDLE",
                "IDLE_CATEGORY", "CURRENT_SOH", "TOTAL_HISTORICAL_VALUE",
            ] if c in dead_df.columns]
            display = dead_df[table_cols].copy()
            if "CANONICAL_NAME" in display.columns:
                display["CANONICAL_NAME"] = display["CANONICAL_NAME"].map(fmt_drug_name)
            st.dataframe(
                display.rename(columns={
                    "CANONICAL_NAME": "Drug", "THERAPEUTIC_CLASS": "Class",
                    "DAYS_IDLE": "Days idle", "IDLE_CATEGORY": "Category",
                    "CURRENT_SOH": "SOH", "TOTAL_HISTORICAL_VALUE": "KES value",
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "Days idle": st.column_config.NumberColumn(format="%d d"),
                    "KES value": st.column_config.NumberColumn(format="KES %,.0f"),
                    "SOH":       st.column_config.NumberColumn(format="%,.1f"),
                },
            )
            st.download_button(
                "📥 Export dead stock (CSV)", dead_df.to_csv(index=False),
                file_name=f"dead_stock_{fac.schema}_{pd.Timestamp.now().date()}.csv",
                mime="text/csv",
            )
        else:
            empty_state("No idle inventory found.", icon="✅")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PATIENT RISK
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Patient Risk":

    page_header(
        title="Patient Risk",
        subtitle="Patients on chronic or opioid medication affected by current stock shortfalls",
        facility_label=fac.label, is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _pr_load_exposure(schema, ref_date):
        return get_patient_risk_exposure(schema, ref_date=ref_date)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _pr_load_totals(schema, ref_date):
        df = get_patient_risk_totals(schema, ref_date=ref_date)
        if df.empty:
            return {}
        return {str(k).upper(): v for k, v in df.iloc[0].to_dict().items()}

    with st.spinner("Loading patient exposure data…"):
        exposure_df = _pr_load_exposure(fac.schema, _ref_date)
        totals      = _pr_load_totals(fac.schema, _ref_date)

    exposure_df.columns = exposure_df.columns.str.upper()
    exposure_df = clean_drug_names(exposure_df)

    def _pint(key):
        v = totals.get(key)
        return int(v) if v is not None else 0

    stat_strip([
        {"label": "Drugs with patient exposure", "value": fmt_int(len(exposure_df)),
         "hint": "At risk", "hint_good": len(exposure_df) == 0,
         "accent_color": "#991B1B" if len(exposure_df) else "#111827"},
        {"label": "Patients affected (total)",   "value": fmt_int(_pint("TOTAL_PATIENTS_AT_RISK")),
         "hint": "Unique patients", "hint_good": _pint("TOTAL_PATIENTS_AT_RISK") == 0,
         "accent_color": "#DC2626" if _pint("TOTAL_PATIENTS_AT_RISK") else "#111827"},
        {"label": "Chronic disease patients",    "value": fmt_int(_pint("CHRONIC_PATIENTS_AT_RISK")),
         "hint": "HTN, DM, epilepsy", "hint_good": _pint("CHRONIC_PATIENTS_AT_RISK") == 0,
         "accent_color": "#D97706" if _pint("CHRONIC_PATIENTS_AT_RISK") else "#111827"},
        {"label": "Opioid therapy patients",     "value": fmt_int(_pint("OPIOID_PATIENTS_AT_RISK")),
         "hint": "Morphine, pethidine", "hint_good": _pint("OPIOID_PATIENTS_AT_RISK") == 0,
         "accent_color": "#D97706" if _pint("OPIOID_PATIENTS_AT_RISK") else "#111827"},
        {"label": "Total active patients (90d)", "value": fmt_int(_pint("TOTAL_ACTIVE_PATIENTS"))},
    ])

    if exposure_df.empty:
        empty_state(
            "No patient-drug exposure records found. Either stock is healthy or patient data is unavailable.",
            icon="🩺",
        )
        st.stop()

    section_header("At-risk drug exposure detail")
    st.info(
        "Patients listed here were dispensed these drugs within the last 90 days and the drug supply is "
        "currently at risk. This does **not** mean they are actively harmed — it signals which stockouts "
        "require clinical escalation.",
        icon="ℹ️",
    )

    risk_filter = st.multiselect(
        "Filter by risk tier", options=["stockout", "critical", "low"],
        default=["stockout", "critical"],
    )
    view_df = exposure_df.copy()
    if risk_filter and "RISK_TIER" in view_df.columns:
        view_df = view_df[view_df["RISK_TIER"].isin(risk_filter)]

    show_cols = [c for c in [
        "CANONICAL_NAME", "THERAPEUTIC_CLASS", "THERAPEUTIC_SUBCLASS",
        "RISK_TIER", "DAYS_OF_STOCK", "TOTAL_PATIENTS_AT_RISK",
        "CHRONIC_PATIENTS", "OPIOID_PATIENTS",
    ] if c in view_df.columns]

    st.dataframe(
        view_df[show_cols].rename(columns={
            "CANONICAL_NAME": "Drug", "THERAPEUTIC_CLASS": "Class",
            "THERAPEUTIC_SUBCLASS": "Subclass", "RISK_TIER": "Risk tier",
            "DAYS_OF_STOCK": "DOS remaining", "TOTAL_PATIENTS_AT_RISK": "Patients affected",
            "CHRONIC_PATIENTS": "Chronic", "OPIOID_PATIENTS": "Opioid",
        }),
        use_container_width=True, hide_index=True,
        column_config={
            "DOS remaining":     st.column_config.NumberColumn(format="%.1f d"),
            "Patients affected": st.column_config.NumberColumn(),
            "Chronic":           st.column_config.NumberColumn(),
            "Opioid":            st.column_config.NumberColumn(),
        },
    )

    section_header("Clinical escalation guidance")
    st.markdown("""
| Situation | Recommended action |
|---|---|
| Opioid stockout | Notify prescribing clinician immediately. Document controlled substance shortage. |
| Chronic disease stockout (DM, HTN) | Identify substitute in formulary. Notify prescribers. Trigger urgent order. |
| Critical drug <7d DOS | Escalate to medical superintendent. Initiate emergency procurement. |
| Antimicrobial stockout | Review current antibiogram for alternatives. Notify pharmacist-in-charge. |
""")
    st.download_button(
        "📥 Export patient risk report (CSV)", view_df.to_csv(index=False),
        file_name=f"patient_risk_{fac.schema}_{pd.Timestamp.now().date()}.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Demand Insights":

    page_header(
        title="Demand Insights",
        subtitle="Consumption trends · ABC analysis · Demand volatility",
        facility_label=fac.label, is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _di_load_history(schema, ref_date):
        return get_dispensing_history(schema, days_back=730, ref_date=ref_date)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _di_load_monthly(schema):
        return get_monthly_trends(schema)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _di_build_forecasts(schema, ref_date):
        disp = _di_load_history(schema, ref_date)
        if disp.empty:
            return pd.DataFrame()
        return DemandEngine(schema).fit(disp).forecast_all(disp)

    with st.spinner("Loading demand data…"):
        history_df  = _di_load_history(fac.schema, _ref_date)
        monthly_df  = _di_load_monthly(fac.schema)
        forecast_df = _di_build_forecasts(fac.schema, _ref_date)

    history_df.columns = history_df.columns.str.upper()
    monthly_df.columns = monthly_df.columns.str.upper()
    history_df = clean_drug_names(history_df)
    monthly_df = clean_drug_names(monthly_df)

    # ── ABC Analysis ──────────────────────────────────────────────────────────
    section_header("ABC analysis — formulary by dispensing value")
    st.caption(
        f"A items = top {int(ABC_A_CUM_PCT*100)}% of value · "
        f"B items = next {int((ABC_B_CUM_PCT - ABC_A_CUM_PCT)*100)}% · "
        f"C items = remaining. Focus procurement attention on A items."
    )

    if not monthly_df.empty and "CANONICAL_NAME" in monthly_df.columns:
        abc_df = (
            monthly_df.groupby("CANONICAL_NAME")["TOTAL_DISPENSING_VALUE"]
            .sum().reset_index().sort_values("TOTAL_DISPENSING_VALUE", ascending=False)
        )
        abc_df["CUMULATIVE_PCT"] = abc_df["TOTAL_DISPENSING_VALUE"].cumsum() / abc_df["TOTAL_DISPENSING_VALUE"].sum()
        abc_df["ABC_CLASS"] = abc_df["CUMULATIVE_PCT"].apply(
            lambda x: "A" if x <= ABC_A_CUM_PCT else ("B" if x <= ABC_B_CUM_PCT else "C")
        )
        a_value = abc_df[abc_df["ABC_CLASS"] == "A"]["TOTAL_DISPENSING_VALUE"].sum()
        stat_strip([
            {"label": "A items (top value)", "value": fmt_int(int((abc_df["ABC_CLASS"] == "A").sum()))},
            {"label": "B items",             "value": fmt_int(int((abc_df["ABC_CLASS"] == "B").sum()))},
            {"label": "C items",             "value": fmt_int(int((abc_df["ABC_CLASS"] == "C").sum()))},
            {"label": "A-item value",        "value": fmt_kes_millions(a_value)},
            {"label": "Total products",      "value": fmt_int(len(abc_df))},
        ])
        st.plotly_chart(abc_pareto(abc_df), use_container_width=True)
        with st.expander("View full ABC table"):
            st.dataframe(
                abc_df.rename(columns={
                    "CANONICAL_NAME": "Drug", "TOTAL_DISPENSING_VALUE": "Total value (KES)",
                    "CUMULATIVE_PCT": "Cumulative %", "ABC_CLASS": "Class",
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "Total value (KES)": st.column_config.NumberColumn(format="KES %,.0f"),
                    "Cumulative %":      st.column_config.ProgressColumn(format="%.1%", min_value=0, max_value=1),
                },
            )
    else:
        empty_state("Monthly dispensing data not available for ABC analysis.", icon="📊")

    st.markdown("---")
    section_header("Monthly consumption trends")

    if not monthly_df.empty and "CANONICAL_NAME" in monthly_df.columns:
        drug_options = sorted(monthly_df["CANONICAL_NAME"].dropna().unique().tolist())
        selected_drugs = st.multiselect(
            "Select drugs to compare (up to 10)",
            options=drug_options,
            default=drug_options[:3] if len(drug_options) >= 3 else drug_options,
            max_selections=10,
        )
        if selected_drugs:
            st.plotly_chart(
                dispensing_trend(
                    monthly_df[monthly_df["CANONICAL_NAME"].isin(selected_drugs)],
                    y_col="TOTAL_UNITS_DISPENSED", color_col="CANONICAL_NAME",
                ),
                use_container_width=True,
            )
        else:
            empty_state("Select at least one drug above.", icon="📈")
    else:
        empty_state("No monthly trend data available.", icon="📊")

    st.markdown("---")
    section_header("Demand volatility — high CV signals erratic demand")
    st.caption("CV = coefficient of variation (std / mean). CV > 1.0 = highly erratic demand.")

    if not forecast_df.empty:
        forecast_df.columns = forecast_df.columns.str.upper()
        forecast_df = clean_drug_names(forecast_df)
        vol_df = forecast_df[[c for c in [
            "CANONICAL_NAME", "AVG_DAILY_UNITS", "STD_DAILY_UNITS",
            "CV", "TREND_DIRECTION", "CONFIDENCE", "DATA_MONTHS",
        ] if c in forecast_df.columns]].sort_values("CV", ascending=False)
        st.dataframe(
            vol_df.rename(columns={
                "CANONICAL_NAME": "Drug", "AVG_DAILY_UNITS": "Avg daily (u)",
                "STD_DAILY_UNITS": "Std dev (u)", "CV": "CV",
                "TREND_DIRECTION": "Trend", "CONFIDENCE": "Model confidence",
                "DATA_MONTHS": "Data (months)",
            }),
            use_container_width=True, hide_index=True,
            column_config={
                "CV":            st.column_config.NumberColumn(format="%.2f"),
                "Avg daily (u)": st.column_config.NumberColumn(format="%.2f"),
                "Std dev (u)":   st.column_config.NumberColumn(format="%.2f"),
                "Data (months)": st.column_config.NumberColumn(format="%d"),
            },
        )
    else:
        empty_state("Demand forecasts not available.", icon="📊")

    st.markdown("---")
    section_header("Top 20 movers — last 30 days")

    if not monthly_df.empty and "DISPENSING_MONTH" in monthly_df.columns:
        last_30 = monthly_df[monthly_df["DISPENSING_MONTH"] == monthly_df["DISPENSING_MONTH"].max()][[
            "CANONICAL_NAME", "TOTAL_UNITS_DISPENSED", "TOTAL_DISPENSING_VALUE",
            "UNIQUE_PATIENTS", "MOM_CHANGE_PCT",
        ]].copy()
        c1, c2 = st.columns(2)
        with c1:
            st.caption("By units dispensed")
            st.dataframe(
                last_30.nlargest(20, "TOTAL_UNITS_DISPENSED").rename(
                    columns={"CANONICAL_NAME": "Drug", "TOTAL_UNITS_DISPENSED": "Units", "MOM_CHANGE_PCT": "MoM %"}
                ),
                use_container_width=True, hide_index=True,
                column_config={"MoM %": st.column_config.NumberColumn(format="%+.1f%%")},
            )
        with c2:
            st.caption("By dispensing value (KES)")
            st.dataframe(
                last_30.nlargest(20, "TOTAL_DISPENSING_VALUE").rename(
                    columns={"CANONICAL_NAME": "Drug", "TOTAL_DISPENSING_VALUE": "KES value", "MOM_CHANGE_PCT": "MoM %"}
                ),
                use_container_width=True, hide_index=True,
                column_config={
                    "KES value": st.column_config.NumberColumn(format="KES %,.0f"),
                    "MoM %":     st.column_config.NumberColumn(format="%+.1f%%"),
                },
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: COMPLIANCE LOG
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Compliance Log":

    page_header(
        title="Compliance Log",
        subtitle="Deficit dispenses · Negative-stock events · PPB audit trail",
        facility_label=fac.label, is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cl_load_deficits(schema):
        return get_deficit_dispenses(schema)

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cl_load_watchlist(schema, ref_date):
        return get_dos_watchlist(schema, ref_date=ref_date)

    with st.spinner("Loading compliance data…"):
        deficit_df   = _cl_load_deficits(fac.schema)
        watchlist_df = _cl_load_watchlist(fac.schema, _ref_date)

    deficit_df.columns   = deficit_df.columns.str.upper()
    watchlist_df.columns = watchlist_df.columns.str.upper()
    deficit_df   = clean_drug_names(deficit_df)
    watchlist_df = clean_drug_names(watchlist_df)

    n_deficit_events = len(deficit_df)
    n_deficit_drugs  = deficit_df["CANONICAL_NAME"].nunique() if not deficit_df.empty and "CANONICAL_NAME" in deficit_df.columns else 0
    n_deficit_users  = deficit_df["DISPENSED_BY_USER_ID"].nunique() if not deficit_df.empty and "DISPENSED_BY_USER_ID" in deficit_df.columns else 0
    deficit_value    = deficit_df["LINE_TOTAL"].sum() if not deficit_df.empty and "LINE_TOTAL" in deficit_df.columns else 0
    negative_soh     = len(watchlist_df[watchlist_df.get("CURRENT_SOH", pd.Series([0])) < 0]) if not watchlist_df.empty else 0

    stat_strip([
        {"label": "Deficit dispense events",  "value": fmt_int(n_deficit_events),
         "hint": "Dispensed from zero/neg",   "hint_good": n_deficit_events == 0,
         "accent_color": "#991B1B" if n_deficit_events else "#111827"},
        {"label": "Drugs involved",           "value": fmt_int(n_deficit_drugs),
         "accent_color": "#DC2626" if n_deficit_drugs else "#111827"},
        {"label": "Pharmacists involved",     "value": fmt_int(n_deficit_users)},
        {"label": "Value dispensed at zero",  "value": fmt_kes(deficit_value)},
        {"label": "Products at negative SOH", "value": fmt_int(negative_soh),
         "hint": "PPB risk", "hint_good": negative_soh == 0,
         "accent_color": "#991B1B" if negative_soh else "#111827"},
    ])

    if n_deficit_events > 0:
        st.warning(
            f"**{n_deficit_events} dispense events** occurred when stock was at or below zero. "
            "These are flagged by the Pharmacy and Poisons Board (PPB) as compliance violations.",
            icon="⚠️",
        )
    else:
        st.success("No deficit dispenses on record. SOH has been maintained above zero.", icon="✅")

    section_header("Deficit dispense audit log")

    if deficit_df.empty:
        empty_state("No deficit dispenses recorded.", icon="✅")
    else:
        with st.sidebar:
            if "DISPENSED_AT" in deficit_df.columns:
                deficit_df["DISPENSED_AT"] = pd.to_datetime(deficit_df["DISPENSED_AT"])
                min_date = deficit_df["DISPENSED_AT"].min().date()
                max_date = deficit_df["DISPENSED_AT"].max().date()
                date_range = st.date_input(
                    "Date range", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date,
                )
                if len(date_range) == 2:
                    deficit_df = deficit_df[
                        (deficit_df["DISPENSED_AT"].dt.date >= date_range[0]) &
                        (deficit_df["DISPENSED_AT"].dt.date <= date_range[1])
                    ]
            drug_filter = st.text_input("Filter by drug name")

        if drug_filter and "CANONICAL_NAME" in deficit_df.columns:
            deficit_df = deficit_df[deficit_df["CANONICAL_NAME"].str.contains(drug_filter, case=False, na=False)]

        show_cols = [c for c in [
            "DISPENSED_AT", "CANONICAL_NAME", "THERAPEUTIC_SUBCLASS",
            "DISPENSED_BY_USER_ID", "SOH_BEFORE", "QUANTITY_DISPENSED",
            "SOH_AFTER_RAW", "LINE_TOTAL",
        ] if c in deficit_df.columns]

        st.dataframe(
            deficit_df[show_cols].rename(columns={
                "DISPENSED_AT": "Date / Time", "CANONICAL_NAME": "Drug",
                "THERAPEUTIC_SUBCLASS": "Subclass", "DISPENSED_BY_USER_ID": "Pharmacist ID",
                "SOH_BEFORE": "SOH before", "QUANTITY_DISPENSED": "Qty dispensed",
                "SOH_AFTER_RAW": "SOH after", "LINE_TOTAL": "Value (KES)",
            }),
            use_container_width=True, hide_index=True,
            column_config={
                "Date / Time": st.column_config.DatetimeColumn(format="DD MMM YYYY HH:mm"),
                "Value (KES)": st.column_config.NumberColumn(format="KES %,.2f"),
                "SOH before":  st.column_config.NumberColumn(format="%,.1f"),
                "SOH after":   st.column_config.NumberColumn(format="%,.1f"),
            },
        )
        st.download_button(
            "📥 Export compliance log (CSV)", deficit_df.to_csv(index=False),
            file_name=f"compliance_log_{fac.schema}_{pd.Timestamp.now().date()}.csv",
            mime="text/csv",
        )

    section_header("Stockout episode history")

    if not watchlist_df.empty and "FIRST_STOCKOUT_AT" in watchlist_df.columns:
        stockout_events = watchlist_df.dropna(subset=["FIRST_STOCKOUT_AT", "LAST_STOCKOUT_AT"])
        if not stockout_events.empty:
            st.plotly_chart(stockout_timeline(stockout_events), use_container_width=True)
        else:
            empty_state("No stockout episode history in watchlist data.", icon="✅")
    else:
        st.caption("Stockout episode timeline requires first/last stockout timestamps from the watchlist query.")
