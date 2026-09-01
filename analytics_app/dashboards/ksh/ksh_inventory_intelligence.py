"""
KSH Inventory Intelligence 
"""

import sys
import os
from pathlib import Path

ROOT = Path(os.path.abspath("analytics_app/dashboards/ksh/inventory_intelligence"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# dynamic_file_loader exec()s every dashboard inside one long-lived Streamlit
# process, and this dashboard's submodules import each other via bare
# top-level names ("queries", "intelligence", "utils") rather than a
# qualified package path. If a different dashboard was visited earlier in
# the same process and left its own module cached under one of those names,
# Python reuses that stale sys.modules entry instead of re-resolving via the
# ROOT just inserted above — surfacing as "ModuleNotFoundError: 'queries' is
# not a package" or similar. Drop any such stale entries so these names
# always resolve fresh, from this dashboard's own ROOT.
for _name in list(sys.modules):
    if _name in ("queries", "intelligence", "utils") or _name.startswith(
        ("queries.", "intelligence.", "utils.")
    ):
        del sys.modules[_name]

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# ── Shared imports ─────────────────────────────────────────────────────────────
from intelligence import ai_client, order_intelligence
from intelligence.anomaly_engine import AnomalyEngine
from intelligence.insight_engine import detect_all as detect_insights
from intelligence.config import DEFAULT_LEAD_TIME_DAYS, ABC_A_CUM_PCT, ABC_B_CUM_PCT, SERVICE_LEVEL_Z
from intelligence.seasonal_engine import (
    SeasonalEngine, get_climate_signal, parse_ref_date,
    compute_facility_seasonal_index,
    KISUMU_LAT, KISUMU_LON,
)
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
from queries.patient_risk import (
    get_patient_risk_exposure, get_patient_risk_totals, get_patient_refill_overdue,
    get_overdue_patient_list,
)
from queries.receipts import get_kisumu_dispensing_for_lead_time
from utils.charts import (
    status_donut, dos_bar_chart, stockout_timeline, stockout_risk_gantt,
    stockout_class_risk,
    dead_stock_scatter, dead_stock_action_bars,
    dispensing_trend, abc_pareto, anomaly_trend_chart,
)
from utils.components import (
    anomaly_banner, decision_card_ai,
    empty_state, inject_css, insight_card, page_header, section_header, stat_strip,
    traceability_card, data_quality_banner,
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
            "Demand Insights",
            "Dead Stock",
            "Patient Risk",
            "Compliance Log",
        ],
        icons=[
            "clipboard2-pulse",
            "cart-check",
            "exclamation-triangle",
            "graph-up",
            "archive",
            "person-heart",
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
# APP-LEVEL: SHARED ENGINE (runs once per session, cached 1h)
# Demand forecasts, lead times, SOH, and raw dispensing are computed here and
# shared across all pages. This ensures every module uses the same product
# universe, the same avg_daily_units, and the same demand classification.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _build_engines(schema, ref_date, _v=5):
    disp         = get_dispensing_history(schema, days_back=730, ref_date=ref_date)
    soh          = get_current_soh(schema, ref_date=ref_date)
    lt_disp      = get_kisumu_dispensing_for_lead_time()
    demand_eng   = DemandEngine(schema).fit(disp)
    forecasts    = demand_eng.forecast_all(disp)
    lt_eng       = LeadTimeEngine(schema)
    if not lt_disp.empty:
        lt_eng.fit_kisumu(lt_disp)
    return forecasts, lt_eng.get_all_product_lead_times(), soh, lt_eng, disp

with st.spinner("Loading facility intelligence…"):
    _forecasts_df, _lt_df, _soh_df, _lt_eng, _dispensing_df = _build_engines(fac.schema, _ref_date)


def _kpi_from_engines(forecasts_df: pd.DataFrame, soh_df: pd.DataFrame) -> dict:
    """
    Compute STOCKED_OUT / CRITICAL / LOW / TOTAL from the shared engine output.

    Universe mirrors the Order Workbench exactly:
    1. Forecastable products  — rows in forecasts_df (730d history, ≥14d data),
       EWM avg joined to current SOH for DOS computation.
    2. Unforecastable critical — products NOT in forecasts_df that have SQL
       days_of_stock < 7 (matches the Workbench unforecastable-critical block).
    """
    if forecasts_df.empty:
        return {"total_products": 0, "active_stockouts": 0, "critical_count": 0, "low_count": 0}

    _soh = soh_df.copy()
    _soh.columns = _soh.columns.str.upper()
    for _c in ["CURRENT_SOH", "CURRENT_SOH_DISPLAY", "AVG_DAILY_UNITS", "DAYS_OF_STOCK"]:
        if _c in _soh.columns:
            _soh[_c] = pd.to_numeric(_soh[_c], errors="coerce")

    # ── 1. Forecastable universe (start from forecasts, left-join SOH) ─────────
    _fc = forecasts_df[["product_id", "avg_daily_units"]].copy()
    _fc.columns = ["PRODUCT_ID", "EWM_AVG"]
    _fore = _fc.merge(
        _soh[["PRODUCT_ID", "CURRENT_SOH", "CURRENT_SOH_DISPLAY", "AVG_DAILY_UNITS"]],
        on="PRODUCT_ID", how="left",
    )
    _eff = _fore["EWM_AVG"].where(_fore["EWM_AVG"].notna(), _fore["AVG_DAILY_UNITS"]).replace(0, float("nan"))
    _fore["_dos"] = _fore["CURRENT_SOH_DISPLAY"] / _eff
    _fore_soh = _fore["CURRENT_SOH"].fillna(0)

    # ── 2. Unforecastable critical (not in forecasts, SQL DOS < 7) ─────────────
    _fcast_ids = set(_fc["PRODUCT_ID"].astype(str).str.upper())
    _soh["_PID_UP"] = _soh["PRODUCT_ID"].astype(str).str.upper()
    _unf = _soh[
        ~_soh["_PID_UP"].isin(_fcast_ids) &
        _soh["DAYS_OF_STOCK"].notna() &
        (_soh["DAYS_OF_STOCK"] < 7)
    ]
    _unf_soh = _unf["CURRENT_SOH"].fillna(0)

    return {
        "total_products":   len(_fore) + len(_unf),
        "active_stockouts": int((_fore_soh <= 0).sum()) + int((_unf_soh <= 0).sum()),
        "critical_count":   int(((_fore["_dos"] < 7) & (_fore_soh > 0)).sum())
                          + int(((_unf["DAYS_OF_STOCK"] < 7) & (_unf_soh > 0)).sum()),
        "low_count":        int(((_fore["_dos"] >= 7) & (_fore["_dos"] < 30)).sum()),
    }

_engine_kpis = _kpi_from_engines(_forecasts_df, _soh_df)


# ══════════════════════════════════════════════════════════════════════════════
# APP-LEVEL: SEASONAL ENGINE
# Climate signal cached daily (Open-Meteo API). Matching + multipliers computed
# from already-loaded _soh_df — no extra Snowflake calls.
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def _load_climate(lat: float, lon: float, ref_date_str: str):
    _rd = parse_ref_date(ref_date_str)
    return get_climate_signal(_rd)

_ref_date_parsed = parse_ref_date(_ref_date)

# Date the whole dashboard is anchored to — the most recent day of data, not the
# wall-clock date. Shown to users so figures read "as of last data" and aren't
# mistaken for real-time. Falls back to today when ref_date is CURRENT_DATE.
_as_of_date = _ref_date_parsed
_as_of_label = _as_of_date.strftime("%A, %d %b %Y")

# Seasonal engine is only meaningful for active facilities. Use the registry
# flag, not ref-date recency: KSH is live but its data can lag the calendar by
# months, so the ref_date (now anchored to MAX(dispensed_at)) may sit >90 days
# in the past while the facility is still active. Historical schemas (is_live
# False) stay gated off so they don't surface peaks from years ago.
_is_live_schema = fac.is_live

if _is_live_schema:
    _climate_signal    = _load_climate(KISUMU_LAT, KISUMU_LON, _ref_date)
    _seasonal_eng      = SeasonalEngine()
    # Phase 3.5: compute facility-specific seasonal index from dispensing history.
    # Returns None when insufficient data — engine falls back to calendar only.
    _facility_profile  = compute_facility_seasonal_index(
        _dispensing_df, _ref_date_parsed, facility_schema=fac.schema
    )
    _seasonal_summaries = _seasonal_eng.get_disease_summary(
        _soh_df, _ref_date_parsed, _climate_signal, facility_profile=_facility_profile
    )
    _seasonal_mult_map  = _seasonal_eng.get_seasonal_multipliers(
        _soh_df, _ref_date_parsed, _climate_signal, facility_profile=_facility_profile
    )
    # Per-product disease context: passed to AI narrative so it explains WHY quantities
    # are sized above baseline (disease name, timing, uplift %).
    _seasonal_context_map = _seasonal_eng.get_seasonal_context_map(
        _soh_df, _ref_date_parsed, _climate_signal, facility_profile=_facility_profile
    )
    _seasonal_outlook   = _seasonal_eng.get_outlook(_ref_date_parsed)
else:
    _climate_signal       = None
    _facility_profile     = None
    _seasonal_summaries   = []
    _seasonal_mult_map    = {}
    _seasonal_context_map = {}
    _seasonal_outlook     = pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TODAY'S BRIEFING
# ══════════════════════════════════════════════════════════════════════════════

if page == "Today's Briefing":

    page_header(
        title="Today's Briefing",
        subtitle=f"Operational summary · as of {_as_of_label} (latest data)",
        facility_label=fac.label,
        is_live=fac.is_live,
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_load_kpis(schema, ref_date):
        df = get_kpi_summary(schema, ref_date=ref_date)
        return df.iloc[0].to_dict() if not df.empty else {}

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
                seasonal_disease=d.get("seasonal_disease"),
                seasonal_weeks_to_peak=d.get("seasonal_weeks_to_peak"),
                seasonal_demand_mult=d.get("seasonal_demand_mult"),
                seasonal_too_late=d.get("seasonal_too_late", False),
            )
            briefs.append(brief.__dict__)
        return briefs

    with st.spinner("Loading facility data…"):
        dispensing_df = _dispensing_df
        kpi           = _brief_load_kpis(fac.schema, _ref_date)
        soh_df        = _soh_df

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

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_load_dead_stock(schema, ref_date):
        try:
            return get_dead_stock(schema, idle_threshold_days=60, ref_date=ref_date)
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=3600, show_spinner=False)
    def _brief_load_refill_overdue(schema, ref_date):
        try:
            return get_patient_refill_overdue(schema, ref_date=ref_date)
        except Exception:
            return pd.DataFrame()

    with st.spinner("Running intelligence engines…"):
        actions_df       = score_all(soh_df)
        anomalies_df     = _brief_compute_anomalies(dispensing_df, fac.schema)
        dead_stock_df    = _brief_load_dead_stock(fac.schema, _ref_date)
        refill_overdue_df = _brief_load_refill_overdue(fac.schema, _ref_date)

    # Stock status counts come from the shared engine (EWM-based DOS) so they
    # are directly comparable with Order Workbench ORDER NOW figures.
    # Value and patient figures still come from SQL (engine doesn't have those).
    _stockouts = _engine_kpis["active_stockouts"]
    _critical  = _engine_kpis["critical_count"]
    _low       = _engine_kpis["low_count"]
    _total     = _engine_kpis["total_products"]

    # ── KPI strip ────────────────────────────────────────────────────────────────
    stat_strip([
        {"label": "Stocked out",     "value": fmt_int(_stockouts),
         "hint": "Immediate action" if _stockouts else "None", "hint_good": _stockouts == 0,
         "accent_color": "#991B1B" if _stockouts else "#111827"},
        {"label": "Critical  < 7d",  "value": fmt_int(_critical),
         "hint": "Order now" if _critical else "Clear",        "hint_good": _critical == 0,
         "accent_color": "#DC2626" if _critical else "#111827"},
        {"label": "Low  7–30d",      "value": fmt_int(_low),
         "hint": "Monitor" if _low else "Clear",               "hint_good": _low == 0,
         "accent_color": "#D97706" if _low else "#111827"},
        {"label": "Total products",    "value": fmt_int(_total),                                            "accent_color": "#111827"},
        {"label": "90d dispensed (KES)","value": fmt_kes_millions(kpi.get("TOTAL_DISPENSING_VALUE_90D")),    "accent_color": "#111827"},
        {"label": "Chronic patients",  "value": fmt_int(kpi.get("CHRONIC_PATIENTS_ACTIVE")),
         "hint": "Active last 90d",                                                                          "accent_color": "#111827"},
    ])

    # ── Climate context banner (only when anomaly is significant) ────────────────
    if _climate_signal and abs(_climate_signal.anomaly_pct) >= 20:
        _rain_color = "#065A82" if _climate_signal.anomaly_pct > 0 else "#374151"
        _rain_icon  = "🌧" if _climate_signal.anomaly_pct > 0 else "☀"
        _rain_dir   = "above" if _climate_signal.anomaly_pct > 0 else "below"
        _rain_note  = (
            " Elevated malaria and diarrhoeal disease risk."
            if _climate_signal.anomaly_pct > 20 else ""
        )
        st.markdown(
            f"""<div style="background:{_rain_color}12;border-left:3px solid {_rain_color};
            padding:8px 14px;border-radius:4px;margin-bottom:14px;font-size:13px;color:#111827">
            {_rain_icon} <b>Kisumu climate —</b> {_climate_signal.current_month_name}:
            {abs(_climate_signal.anomaly_pct):.0f}% {_rain_dir} seasonal average
            ({_climate_signal.current_month_mm}mm vs {_climate_signal.historical_avg_mm}mm historical).
            {_rain_note}</div>""",
            unsafe_allow_html=True,
        )

    # ── Seasonal demand signals (dedicated section — not competing with clinical alerts) ──
    # Show AT PEAK (CRITICAL) and APPROACHING (HIGH) only — WATCH (MEDIUM, >3w away)
    # belongs in Demand Insights, not the daily briefing.
    # Also exclude diseases where facility data shows demand is not actually elevated
    # (blended multiplier ≤ 1.0 means the county calendar doesn't apply here).
    _actionable_summaries = [
        _s for _s in _seasonal_summaries
        if _s["demand_multiplier"] > 1.0 and _s["severity"] != "MEDIUM"
    ]
    if _actionable_summaries:
        _sev_styles = {
            "CRITICAL": ("background:#FEE2E2;color:#991B1B;border:1px solid #FECACA", "AT PEAK"),
            "HIGH":     ("background:#FEF3C7;color:#92400E;border:1px solid #FDE68A", "APPROACHING"),
            "MEDIUM":   ("background:#EFF6FF;color:#1E40AF;border:1px solid #BFDBFE", "WATCH"),
        }
        _chips_html = ""
        for _s in sorted(_actionable_summaries, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["severity"], 2)):
            _sty, _lbl = _sev_styles.get(_s["severity"], _sev_styles["MEDIUM"])
            _timing = "at peak" if _s["weeks_to_peak"] == 0 else f"{_s['weeks_to_peak']}w away"
            _uplift = round((_s["demand_multiplier"] - 1) * 100)
            _boost_icon = " ☁" if _s.get("climate_boosted") else ""
            _src_label = f" · {_s['signal_source']}" if _s.get("signal_source") else ""
            _chips_html += (
                f"<span style='display:inline-flex;align-items:center;gap:6px;"
                f"border-radius:6px;padding:5px 10px;margin:0 6px 6px 0;font-size:12px;{_sty}'>"
                f"<span style='font-weight:700;font-size:10px'>{_lbl}</span>"
                f"<span style='font-weight:600'>{_s['disease']}</span>"
                f"<span style='opacity:0.7'>· {_timing} · {_s['drugs_at_risk']} drug{'s' if _s['drugs_at_risk'] != 1 else ''} · +{_uplift}% demand{_boost_icon}{_src_label}</span>"
                f"</span>"
            )
        st.markdown(
            f"<div style='margin-bottom:4px;font-size:11px;font-weight:700;"
            f"color:#374151;text-transform:uppercase;letter-spacing:.06em'>"
            f"Seasonal demand signals</div>"
            f"<div style='margin-bottom:14px'>{_chips_html}</div>",
            unsafe_allow_html=True,
        )

    # ── Phase 2: Clinical alert detection (R3, R5 — operational signals only) ─────
    # R6 seasonal has its own dedicated section above; exclude it here.
    from intelligence.insight_engine import RULE_STOCKOUT, RULE_DEMAND_SPIKE, RULE_SEASONAL
    _insights = detect_insights(
        soh_df=soh_df,
        actions_df=actions_df,
        anomalies_df=anomalies_df if not anomalies_df.empty else pd.DataFrame(),
        dead_stock_df=dead_stock_df,
        patient_refill_df=refill_overdue_df,
        top_n=20,
        seasonal_summaries=None,  # seasonal handled in its own section above
    )
    _novel_insights = [
        i for i in _insights
        if i.rule_id not in (RULE_STOCKOUT, RULE_DEMAND_SPIKE, RULE_SEASONAL)
    ][:3]

    left, right = st.columns([1.1, 1], gap="large")

    with right:
        if _novel_insights:
            section_header("Clinical alerts")
            for _ins in _novel_insights:
                insight_card(_ins)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

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
            _soh_merge_cols = ["canonical_name", "current_soh", _dos_col, "therapeutic_class"]
            if "product_id" in _soh_l.columns:
                _soh_merge_cols.append("product_id")
            _soh_for_merge = _soh_l[_soh_merge_cols].rename(columns={_dos_col: "dos_remaining"})
            _merged = _order_now_df.head(5).merge(_soh_for_merge, on="canonical_name", how="left")

            _unit_values = _brief_compute_unit_values(dispensing_df)
            _demand_map  = {
                str(r.get("canonical_name") or "").strip(): {
                    "avg_daily_units": float(r.get("avg_daily_units", 0)),
                    "std_daily_units": float(r.get("std_daily_units", 0)),
                    "trend_direction": str(r.get("trend_direction", "STABLE")),
                    "confidence":      str(r.get("confidence", "LOW")),
                }
                for _, r in _forecasts_df.iterrows()
                if str(r.get("canonical_name") or "").strip()
            }
            _lt_map      = _brief_compute_lead_time_map(fac.schema, soh_df)

            _top_drugs_data = []
            for _, r in _merged.iterrows():
                soh_val = float(r.get("current_soh") or 0)
                dos_val = float(r.get("dos_remaining") or 0)
                name    = str(r.get("canonical_name", "Unknown"))
                lead_t  = max(1, round(_lt_map.get(name, DEFAULT_LEAD_TIME_DAYS)))
                _fc     = _demand_map.get(name, {})
                adc        = _fc.get("avg_daily_units", 0.0)
                _conf      = _fc.get("confidence", "LOW")
                # Don't project a specific quantity from a LOW-confidence forecast.
                # The demand estimate is too sparse to be actionable — send to Workbench.
                qty     = max(0, int((30 + lead_t) * adc - soh_val)) if _conf != "LOW" else 0

                # Seasonal context: look up by product_id → disease, timing, uplift.
                # too_late = lead time exceeds the remaining window to peak. The order
                # quantity is still sized at peak rate (covers the tail of the peak),
                # but the AI narrative escalates to urgency / expedite sourcing.
                _pid = str(r.get("product_id") or "").upper()
                _seas_ctx = _seasonal_context_map.get(_pid, {})
                _raw_mult = _seas_ctx.get("demand_mult") or 1.0
                _wtp = _seas_ctx.get("weeks_to_peak")
                _seas_too_late = (
                    _raw_mult > 1.0
                    and _wtp is not None
                    and lead_t > _wtp * 7
                )

                _top_drugs_data.append({
                    "canonical_name":         name,
                    "dos_remaining":          dos_val,
                    "avg_daily_units":        adc,
                    "current_soh":            soh_val,
                    "order_qty":              qty,
                    "lead_time_days":         lead_t,
                    "clinical_priority":      str(r.get("clinical_priority", "STANDARD")),
                    "therapeutic_class":      str(r.get("therapeutic_class", "") or ""),
                    "trend_direction":        _fc.get("trend_direction", "STABLE"),
                    "confidence":             _fc.get("confidence", "LOW"),
                    "avg_unit_value_kes":     _unit_values.get(name),
                    "seasonal_disease":       _seas_ctx.get("disease"),
                    "seasonal_weeks_to_peak": _wtp,
                    "seasonal_demand_mult":   _raw_mult if _raw_mult > 1.0 else None,
                    "seasonal_too_late":      _seas_too_late,
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
                    confidence=drug_data.get("confidence"),
                )

            if _n_urgent > 5:
                st.markdown(
                    f"<div style='font-size:12px;color:#6B7280;margin:4px 0 8px'>"
                    f"Showing top 5 of {_n_urgent} urgent items</div>",
                    unsafe_allow_html=True,
                )
            st.caption("→ Switch to **Order Workbench** in the sidebar for full reorder details")

    # ── Email digest ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            "<div style='font-size:11px;font-weight:700;color:#374151;margin-bottom:6px'>"
            "📬 Daily digest</div>",
            unsafe_allow_html=True,
        )
        if st.button("Send digest email", key="send_digest", use_container_width=True):
            from utils.notifier import send_daily_digest
            _order_now_count = len(actions_df[actions_df["action"] == ORDER_NOW]) if not actions_df.empty else 0
            _pr_count_row = {}
            try:
                from queries.patient_risk import get_patient_risk_totals as _gpr
                _pr_df = _gpr(fac.schema, ref_date=_ref_date)
                if not _pr_df.empty:
                    _pr_count_row = {k.lower(): v for k, v in _pr_df.iloc[0].to_dict().items()}
            except Exception:
                pass
            _pr_count = int(_pr_count_row.get("total_patients_at_risk", 0) or 0)
            # Use engine-derived stock counts (same source as the briefing KPI strip)
            # supplemented by SQL kpi for value/chronic fields.
            _digest_kpi = {
                "active_stockouts":       _engine_kpis["active_stockouts"],
                "critical_count":         _engine_kpis["critical_count"],
                "low_count":              _engine_kpis["low_count"],
                "chronic_patients_active": kpi.get("CHRONIC_PATIENTS_ACTIVE") or kpi.get("chronic_patients_active", 0),
            }
            with st.spinner("Sending digest…"):
                try:
                    sent = send_daily_digest(
                        facility_name=fac.label,
                        insights=_insights,
                        kpi=_digest_kpi,
                        order_count=_order_now_count,
                        patient_risk_count=_pr_count,
                        facility_slug=fac.schema,
                        force=True,
                        clinical_alerts=_novel_insights,
                        seasonal_signals=_actionable_summaries or [],
                    )
                    if sent:
                        st.success("Digest sent ✓")
                    else:
                        st.warning("Email not configured — add NOTIFY_EMAIL_TO and EMAIL_HOST_USER to .env")
                except Exception as _e:
                    st.error(f"Send failed: {_e}")


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

    forecasts_df = _forecasts_df
    lt_df        = _lt_df
    soh_df       = _soh_df
    lt_eng       = _lt_eng

    if forecasts_df.empty:
        empty_state("Not enough dispensing history to generate recommendations.", icon="📊")
        st.stop()

    rec = forecasts_df.copy()
    rec.columns = rec.columns.str.upper()

    # Guard against stale cached DataFrames that pre-date SB classification fields
    for _col, _default in [("DEMAND_TYPE", "UNKNOWN"), ("ADI", 0.0), ("CV_NZ", 0.0)]:
        if _col not in rec.columns:
            rec[_col] = _default

    if not soh_df.empty:
        rec = rec.merge(
            soh_df[["PRODUCT_ID", "CURRENT_SOH_DISPLAY", "CURRENT_SOH",
                    "THERAPEUTIC_CLASS", "THERAPEUTIC_SUBCLASS"]].copy(),
            on="PRODUCT_ID", how="left"
        )
    else:
        for col in ["CURRENT_SOH_DISPLAY", "CURRENT_SOH", "THERAPEUTIC_CLASS", "THERAPEUTIC_SUBCLASS"]:
            rec[col] = 0 if "SOH" in col else ""

    # Seasonal demand multiplier: applied to avg daily units before SS/ROP calculation.
    # Products in an approaching disease season get their order quantity sized to the
    # peak-demand rate rather than the current baseline.
    # Exception: if lead time exceeds the weeks remaining to peak, the order would
    # arrive after the season ends — uplift is suppressed so we don't over-order.
    rec["SEASONAL_MULT"] = (
        rec["PRODUCT_ID"].astype(str).str.upper()
        .map(lambda pid: _seasonal_mult_map.get(pid, 1.0))
    )
    rec["SEASONAL_WEEKS_TO_PEAK"] = (
        rec["PRODUCT_ID"].astype(str).str.upper()
        .map(lambda pid: _seasonal_context_map.get(pid, {}).get("weeks_to_peak"))
    )

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

    # Add critical products that have no forecast (< MIN_DAYS_FOR_FORECAST history).
    # These are visible in Briefing KPIs as CRITICAL but absent from forecasts_df.
    # Appending them ensures ORDER NOW count in the Workbench matches the Briefing.
    if not soh_df.empty:
        _fcast_ids = set(rec["PRODUCT_ID"].astype(str).str.upper())
        _soh_sup = soh_df.copy()
        _soh_sup.columns = _soh_sup.columns.str.upper()
        _unf = _soh_sup[
            ~_soh_sup["PRODUCT_ID"].astype(str).str.upper().isin(_fcast_ids) &
            _soh_sup["DAYS_OF_STOCK"].notna() &
            (_soh_sup["DAYS_OF_STOCK"] < 7)
        ].copy()
        if not _unf.empty:
            _unf = _unf.rename(columns={"DAYS_OF_STOCK": "dos_stub"})
            _unf["CONFIDENCE"]     = "LOW"
            _unf["DEMAND_TYPE"]    = "UNKNOWN"
            _unf["ADI"]            = 0.0
            _unf["CV_NZ"]          = 0.0
            _unf["CV"]             = 0.0
            _unf["DATA_MONTHS"]    = 0
            _unf["TREND_DIRECTION"] = "STABLE"
            _unf["LT_MEAN"]        = fac_lt_mean if lead_time_override == 0 else float(lead_time_override)
            _unf["LT_STD"]         = fac_lt_std
            rec = pd.concat([rec, _unf], ignore_index=True, sort=False)

    def _safe(v, default=0.0):
        try:
            f = float(v)
            return f if pd.notna(f) else default
        except (TypeError, ValueError):
            return default

    def _compute_row(row):
        lt_m  = _safe(row.get("LT_MEAN"), fac_lt_mean)
        lt_s  = _safe(row.get("LT_STD"),  fac_lt_std)
        adc   = _safe(row.get("AVG_DAILY_UNITS"))
        std   = _safe(row.get("STD_DAILY_UNITS"), adc * 0.3)
        soh   = _safe(row.get("CURRENT_SOH_DISPLAY"))
        smult = _safe(row.get("SEASONAL_MULT"), 1.0)
        # Apply seasonal uplift to demand rate used for SS/ROP/order qty.
        # Uplift is always applied regardless of lead time — a seasonal peak lasts
        # weeks, so even if stock arrives mid-peak it covers the remaining peak demand.
        # When lead time exceeds the peak window, the AI narrative escalates to urgency
        # (expedite / cross-facility redistribution) rather than shrinking the order.
        adc_s = adc * smult if smult > 1.0 else adc
        _rop  = ss.reorder_point(adc_s, lt_m, std, lt_s, service_level)
        # Flag when lead time exceeds remaining weeks to peak — stock can't arrive
        # before the season starts. Quantity is still sized at peak rate (the order
        # covers the tail), but the AI narrative escalates to urgency/expedite.
        _wtp = row.get("SEASONAL_WEEKS_TO_PEAK")
        _too_late = (
            smult > 1.0
            and _wtp is not None
            and not pd.isna(_wtp)
            and lt_m > float(_wtp) * 7
        )
        return pd.Series({
            "rop":                  round(_rop, 0),
            "safety_stock":         round(ss.safety_stock(std, lt_m, lt_s, adc_s, service_level), 0),
            "order_qty":            round(ss.recommended_order_quantity(soh, _rop, adc_s, lt_m, target_cover), 0),
            "dos_remaining":        ss.days_of_stock(soh, adc),   # baseline rate for display
            "needs_order":          soh <= _rop,
            "lt_mean_used":         round(lt_m, 1),
            "seasonal_uplift":      smult,
            "seasonal_too_late":    _too_late,
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
        "DOS remaining": display["dos_remaining"].apply(fmt_days).values,
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
            "DOS remaining": st.column_config.TextColumn(),
            "Current SOH":   st.column_config.NumberColumn(format="%,.0f"),
            "Avg daily (u)": st.column_config.NumberColumn(format="%.2f"),
            "Order qty":     st.column_config.NumberColumn(format="%,.0f"),
        },
    )

    selected = event.selection.rows if hasattr(event, "selection") else []
    if selected:
        row = display.iloc[selected[0]]
        drug_name = str(row.get("CANONICAL_NAME", "—"))

        _lt_n    = int(_safe(row.get("LT_N"), 0))
        _lt_src  = "observed from receipts" if _lt_n >= 3 else "facility average"
        _z_val   = SERVICE_LEVEL_Z.get(service_level, 1.645)
        traceability_card(
            drug_name=drug_name,
            clinical_priority=str(row.get("CLINICAL_PRIORITY", "STANDARD")),
            avg_daily_units=_safe(row.get("AVG_DAILY_UNITS")),
            std_daily_units=_safe(row.get("STD_DAILY_UNITS"), _safe(row.get("AVG_DAILY_UNITS")) * 0.3),
            cv=_safe(row.get("CV")),
            demand_type=str(row.get("DEMAND_TYPE", "UNKNOWN")).upper(),
            adi=_safe(row.get("ADI"), 0.0),
            cv_nz=_safe(row.get("CV_NZ"), 0.0),
            data_months=int(_safe(row.get("DATA_MONTHS"), 1)),
            confidence=str(row.get("CONFIDENCE", "LOW")).upper(),
            trend_direction=str(row.get("TREND_DIRECTION", "STABLE")).upper(),
            lt_mean=_safe(row.get("lt_mean_used"), fac_lt_mean),
            lt_std=_safe(row.get("LT_STD"), fac_lt_std),
            lt_source=_lt_src,
            safety_stock_units=_safe(row.get("safety_stock")),
            z_value=_z_val,
            service_level=service_level,
            rop=_safe(row.get("rop")),
            current_soh=_safe(row.get("CURRENT_SOH_DISPLAY")),
            target_cover_days=target_cover,
            order_qty=_safe(row.get("order_qty")),
        )

        _seas_uplift = _safe(row.get("seasonal_uplift"), 1.0)
        if _seas_uplift > 1.0:
            _uplift_pct = round((_seas_uplift - 1) * 100)
            _seas_disease = next(
                (s["disease"] for s in _seasonal_summaries
                 if any(
                     kw in str(row.get("THERAPEUTIC_SUBCLASS") or "").lower()
                     for kw in [s["disease"].lower().split()[0]]
                 )),
                "seasonal peak",
            )
            st.info(
                f"Seasonal demand uplift applied: **+{_uplift_pct}%** — "
                f"order quantity sized for approaching {_seas_disease} season. "
                f"DOS remaining shown at current baseline rate.",
                icon="🌡",
            )

        _brief_key = f"ai_brief_wb_{str(row.get('CANONICAL_NAME', ''))}"
        _col_ai, _ = st.columns([1, 2])
        with _col_ai:
            if st.button("✦ Generate AI reasoning", key=f"ai_btn_{selected[0]}", use_container_width=True):
                with st.spinner("Generating clinical recommendation…"):
                    _wb_pid = str(row.get("PRODUCT_ID") or "").upper()
                    _wb_seas = _seasonal_context_map.get(_wb_pid, {})
                    _wb_too_late = bool(row.get("seasonal_too_late", False))
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
                        seasonal_disease=_wb_seas.get("disease"),
                        seasonal_weeks_to_peak=_wb_seas.get("weeks_to_peak"),
                        seasonal_demand_mult=_wb_seas.get("demand_mult"),
                        seasonal_too_late=_wb_too_late,
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
    # Items whose CANONICAL_NAME is a raw product ID (no taxonomy entry) — try PRODUCT_NAME
    if "PRODUCT_NAME" in df.columns:
        _no_name = df["CANONICAL_NAME"].str.startswith("[#", na=False)
        df.loc[_no_name, "CANONICAL_NAME"] = df.loc[_no_name, "PRODUCT_NAME"].map(fmt_drug_name)

    # Snowflake returns decimal.Decimal for NUMBER columns; cast to float so Python
    # arithmetic in the forecast join below doesn't raise type errors.
    for _num_col in ["CURRENT_SOH", "CURRENT_SOH_DISPLAY", "AVG_DAILY_UNITS",
                     "STDDEV_DAILY_UNITS", "DAYS_OF_STOCK_P50", "DAYS_OF_STOCK_P90"]:
        if _num_col in df.columns:
            df[_num_col] = pd.to_numeric(df[_num_col], errors="coerce")

    # Join EWM avg_daily_units and demand_type from the shared forecast engine.
    # Where a forecast exists, use the EWM-based rate for DOS (more accurate than
    # the SQL 90d rolling avg, and consistent with the Order Workbench).
    # Products without a forecast retain the SQL avg and get CONFIDENCE = LOW.
    if not _forecasts_df.empty and "PRODUCT_ID" in df.columns:
        _fc_map = _forecasts_df[["product_id", "demand_type", "avg_daily_units", "confidence"]].copy()
        _fc_map.columns = ["PRODUCT_ID", "DEMAND_TYPE", "EWM_AVG_DAILY_UNITS", "CONFIDENCE"]
        df = df.merge(_fc_map, on="PRODUCT_ID", how="left")
        _ewm = df["EWM_AVG_DAILY_UNITS"].where(df["EWM_AVG_DAILY_UNITS"].notna(), df["AVG_DAILY_UNITS"])
        df["DAYS_OF_STOCK_P50"] = (
            df["CURRENT_SOH_DISPLAY"] / _ewm.replace(0, float("nan"))
        ).round(1)
        df["DAYS_OF_STOCK_P90"] = (
            df["CURRENT_SOH_DISPLAY"] /
            (_ewm + 1.28 * df["STDDEV_DAILY_UNITS"].fillna(0)).replace(0, float("nan"))
        ).round(1)
        df["DOS_STATUS"] = df.apply(
            lambda r: "red" if r["CURRENT_SOH"] <= 0 else
                      ("red"   if pd.notna(r["DAYS_OF_STOCK_P50"]) and r["DAYS_OF_STOCK_P50"] < 7  else
                      ("amber" if pd.notna(r["DAYS_OF_STOCK_P50"]) and r["DAYS_OF_STOCK_P50"] < 30 else "green")),
            axis=1,
        )
        df["DEMAND_TYPE"] = df["DEMAND_TYPE"].fillna("UNKNOWN")
        df["CONFIDENCE"]  = df["CONFIDENCE"].fillna("LOW")
        # Recompute predicted stockout dates to match the updated EWM-based DOS values
        # Use max(last_dispensed_at) as data-freshness anchor rather than today's date,
        # since even "live" feeds have ingestion lag.
        _anchor = (
            pd.Timestamp(_ref_date.strip("'"))
            if _ref_date != "CURRENT_DATE"
            else (
                pd.to_datetime(df["LAST_DISPENSED_AT"]).max().normalize()
                if "LAST_DISPENSED_AT" in df.columns and df["LAST_DISPENSED_AT"].notna().any()
                else pd.Timestamp.now().normalize()
            )
        )
        df["PREDICTED_STOCKOUT_P50"] = df["DAYS_OF_STOCK_P50"].apply(
            lambda d: _anchor + pd.Timedelta(days=int(round(d))) if pd.notna(d) and d >= 0 else pd.NaT
        )
        df["PREDICTED_STOCKOUT_P90"] = df["DAYS_OF_STOCK_P90"].apply(
            lambda d: _anchor + pd.Timedelta(days=int(round(d))) if pd.notna(d) and d >= 0 else pd.NaT
        )
    else:
        df["DEMAND_TYPE"] = "UNKNOWN"
        df["CONFIDENCE"]  = "LOW"

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

    section_header(f"Stock runway — next 45 days")
    st.caption(
        "Sorted by urgency — most critical at top. "
        "Dark red = already stocked out · Red = <7 days · Amber = 7–30 days · Teal = adequate stock. "
        "Dotted lines mark the 7d and 30d thresholds."
    )
    if not view.empty:
        st.plotly_chart(
            stockout_risk_gantt(view, anchor_date=_anchor, top_n=top_n, window_days=45),
            use_container_width=True,
        )
    else:
        empty_state("No products in this filter.", icon="✅")

    with st.expander("Full watchlist — search & export", expanded=False):
        search = st.text_input("Search drug name", placeholder="e.g. Metformin", label_visibility="collapsed")
        table_df = view.copy()
        if search:
            table_df = table_df[table_df["CANONICAL_NAME"].str.contains(search, case=False, na=False)]
        st.dataframe(
            table_df[[c for c in [
                "CANONICAL_NAME", "THERAPEUTIC_CLASS", "CURRENT_SOH_DISPLAY",
                "AVG_DAILY_UNITS", "DAYS_OF_STOCK_P50", "DOS_STATUS",
                "DEMAND_TYPE", "CONFIDENCE", "PREDICTED_STOCKOUT_P50", "STOCKOUT_EPISODE_COUNT",
            ] if c in table_df.columns]].rename(columns={
                "CANONICAL_NAME": "Drug", "THERAPEUTIC_CLASS": "Class",
                "CURRENT_SOH_DISPLAY": "SOH", "AVG_DAILY_UNITS": "Avg daily (u)",
                "DAYS_OF_STOCK_P50": "Days remaining", "DOS_STATUS": "Status",
                "DEMAND_TYPE": "Demand pattern", "CONFIDENCE": "Confidence",
                "PREDICTED_STOCKOUT_P50": "Stockout by",
                "STOCKOUT_EPISODE_COUNT": "# Stockouts",
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

    # ── Stockout intelligence ─────────────────────────────────────────────────
    _so_df = df[df["CURRENT_SOH"] <= 0].copy() if "CURRENT_SOH" in df.columns else pd.DataFrame()

    if not _so_df.empty:
        st.markdown("---")
        section_header("Stockout intelligence")

        # Duration tiers using last_dispensed_at as proxy for stockout date
        if "LAST_DISPENSED_AT" in _so_df.columns:
            _so_df["_last_disp"] = pd.to_datetime(_so_df["LAST_DISPENSED_AT"], errors="coerce")
            _so_df["_days_out"]  = (_anchor - _so_df["_last_disp"]).dt.days.clip(lower=0)

            _acute    = _so_df[_so_df["_days_out"] <  7]
            _serious  = _so_df[(_so_df["_days_out"] >= 7)  & (_so_df["_days_out"] < 30)]
            _chronic  = _so_df[_so_df["_days_out"] >= 30]

            _dur_cols = st.columns(3)
            for _col, _label, _subset, _clr, _hint in [
                (_dur_cols[0], "Acute (<7d out)",    _acute,   "#DC2626", "May have order in transit"),
                (_dur_cols[1], "Serious (7–30d out)", _serious, "#D97706", "Order overdue"),
                (_dur_cols[2], "Chronic (30d+ out)",  _chronic, "#7F1D1D", "Procurement process failure"),
            ]:
                _n = len(_subset)
                _col.markdown(
                    f"""
                    <div style="border:1px solid #E5E7EB;border-radius:8px;padding:16px;">
                      <div style="font-size:11px;font-weight:600;color:#6B7280;text-transform:uppercase;">{_label}</div>
                      <div style="font-size:32px;font-weight:700;color:{_clr};margin:4px 0 2px;">{_n}</div>
                      <div style="font-size:11px;color:#9CA3AF;">{_hint}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)

        # Volume at risk by therapeutic class
        if "THERAPEUTIC_CLASS" in _so_df.columns and "AVG_DAILY_UNITS" in _so_df.columns:
            st.plotly_chart(stockout_class_risk(_so_df), use_container_width=True)

        # Repeat offenders
        if "STOCKOUT_EPISODE_COUNT" in _so_df.columns:
            _repeat = (
                _so_df[pd.to_numeric(_so_df["STOCKOUT_EPISODE_COUNT"], errors="coerce").fillna(0) >= 3]
                .copy()
            )
            _repeat["STOCKOUT_EPISODE_COUNT"] = pd.to_numeric(_repeat["STOCKOUT_EPISODE_COUNT"], errors="coerce")
            _repeat = _repeat.sort_values("STOCKOUT_EPISODE_COUNT", ascending=False)

            if not _repeat.empty:
                with st.expander(f"Repeat offenders — {len(_repeat)} drugs stocked out 3+ times", expanded=True):
                    st.caption(
                        "These drugs have a systemic procurement problem. A single reorder won't fix it — "
                        "review order frequency, safety stock, or supplier reliability."
                    )
                    _rep_disp = _repeat[[c for c in [
                        "CANONICAL_NAME", "THERAPEUTIC_CLASS", "STOCKOUT_EPISODE_COUNT",
                        "AVG_DAILY_UNITS", "DAYS_OF_STOCK_P50",
                    ] if c in _repeat.columns]].copy()
                    if "CANONICAL_NAME" in _rep_disp.columns:
                        _rep_disp["CANONICAL_NAME"] = _rep_disp["CANONICAL_NAME"].map(fmt_drug_name)
                    st.dataframe(
                        _rep_disp.rename(columns={
                            "CANONICAL_NAME": "Drug", "THERAPEUTIC_CLASS": "Class",
                            "STOCKOUT_EPISODE_COUNT": "Times stocked out",
                            "AVG_DAILY_UNITS": "Avg daily (u)", "DAYS_OF_STOCK_P50": "DOS",
                        }),
                        use_container_width=True, hide_index=True,
                        column_config={
                            "Times stocked out": st.column_config.NumberColumn(format="%d×"),
                            "Avg daily (u)":     st.column_config.NumberColumn(format="%.1f"),
                            "DOS":               st.column_config.NumberColumn(format="%.0f d"),
                        },
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

    # Compute current stock value = SOH × avg historical unit price.
    # TOTAL_HISTORICAL_VALUE is cumulative dispensing revenue — wrong for a dead stock context.
    # avg_unit_price = TOTAL_HISTORICAL_VALUE / TOTAL_HISTORICAL_UNITS gives a per-unit price proxy.
    if not dead_df.empty:
        _thv = pd.to_numeric(dead_df.get("TOTAL_HISTORICAL_VALUE", 0), errors="coerce").fillna(0)
        _thu = pd.to_numeric(dead_df.get("TOTAL_HISTORICAL_UNITS",  0), errors="coerce").replace(0, float("nan"))
        _soh = pd.to_numeric(dead_df.get("CURRENT_SOH", 0),             errors="coerce").fillna(0)
        dead_df["_avg_unit_price"] = (_thv / _thu).fillna(0)
        dead_df["STOCK_VALUE"]     = (_soh * dead_df["_avg_unit_price"]).round(0)

    # ── History load: shared for ITR + root cause ─────────────────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def _ds_load_history(schema, ref_date):
        return get_dispensing_history(schema, days_back=730, ref_date=ref_date)

    _ds_history = _ds_load_history(fac.schema, _ref_date)
    _ds_h: pd.DataFrame = pd.DataFrame()
    _anchor_ds = (
        pd.Timestamp(_ref_date.strip("'"))
        if _ref_date != "CURRENT_DATE"
        else (
            pd.to_datetime(_ds_history["dispensed_at"]).max().normalize()
            if not _ds_history.empty and "dispensed_at" in _ds_history.columns
            else pd.Timestamp.now().normalize()
        )
    )

    if not _ds_history.empty:
        _ds_h = _ds_history.copy()
        _ds_h.columns = _ds_h.columns.str.upper()
        for _c in ("QUANTITY_DISPENSED", "SOH_BEFORE"):
            if _c in _ds_h.columns:
                _ds_h[_c] = pd.to_numeric(_ds_h[_c], errors="coerce")
        _ds_h["DISPENSED_AT"] = pd.to_datetime(_ds_h.get("DISPENSED_AT"), errors="coerce")
        _ds_h = _ds_h.dropna(subset=["DISPENSED_AT", "QUANTITY_DISPENSED"])
        if "CANONICAL_NAME" in _ds_h.columns:
            _ds_h = clean_drug_names(_ds_h)

    # ── ITR: per-drug lookback, both historical and recent (last 180d) ─────────
    # Fix: global lookback understates ITR for drugs with shorter activity windows.
    # Fix: historical ITR alone masks drugs that stopped dispensing — add recent_itr.
    _ds_itr_map:        dict = {}   # historical ITR (full 730d window)
    _ds_recent_itr_map: dict = {}   # recent ITR (last 180d only)
    _ds_median_itr:     float = 0.0

    if not _ds_h.empty:
        # Historical ITR: global lookback gives a stable facility-wide median.
        # Per-drug lookback inflates ITR for drugs with short activity windows and
        # skews the median — global lookback is more appropriate for the KPI strip.
        _global_lookback = max(1, (_ds_h["DISPENSED_AT"].max() - _ds_h["DISPENSED_AT"].min()).days)
        _cutoff_180      = _anchor_ds - pd.Timedelta(days=180)
        _ds_h_recent     = _ds_h[_ds_h["DISPENSED_AT"] >= _cutoff_180]
        _recent_lookback = max(1, min(180, _global_lookback))

        def _compute_hist_itr(df_grp: pd.DataFrame) -> float:
            _qty = float(df_grp["QUANTITY_DISPENSED"].sum())
            _soh = float(df_grp["SOH_BEFORE"][df_grp["SOH_BEFORE"] > 0].mean()) \
                   if "SOH_BEFORE" in df_grp.columns and (df_grp["SOH_BEFORE"] > 0).any() else 0.0
            return round((_qty * 365.0 / _global_lookback) / _soh, 1) if _soh > 0 else 0.0

        def _compute_recent_itr(df_grp: pd.DataFrame) -> float:
            # Per-drug lookback capped at 180d for the recent window.
            _span = max(1, min(_recent_lookback,
                               (df_grp["DISPENSED_AT"].max() - df_grp["DISPENSED_AT"].min()).days))
            _qty  = float(df_grp["QUANTITY_DISPENSED"].sum())
            _soh  = float(df_grp["SOH_BEFORE"][df_grp["SOH_BEFORE"] > 0].mean()) \
                    if "SOH_BEFORE" in df_grp.columns and (df_grp["SOH_BEFORE"] > 0).any() else 0.0
            return round((_qty * 365.0 / _span) / _soh, 1) if _soh > 0 else 0.0

        for _drug, _grp in _ds_h.groupby("CANONICAL_NAME"):
            _ds_itr_map[_drug] = _compute_hist_itr(_grp)

        for _drug, _grp in _ds_h_recent.groupby("CANONICAL_NAME"):
            _ds_recent_itr_map[_drug] = _compute_recent_itr(_grp)

        _valid_itrs    = [v for v in _ds_itr_map.values() if v > 0]
        _ds_median_itr = float(pd.Series(_valid_itrs).median()) if _valid_itrs else 0.0

    # ── Root cause classification ──────────────────────────────────────────────
    # Five categories with clear action implications:
    #
    #   Procurement gap   → recent_itr > 6× AND days_idle < 90: fast mover between orders.
    #                        Not dead stock — self-resolves. Shown separately, excluded from KPIs.
    #   Never dispensed   → no dispensing history in the 730d window at all.
    #   Demand ceased     → was dispensed, recent 90d activity = 0.
    #   Demand dropped    → recent monthly < 20% of smoothed peak (top-3 month average).
    #   Over-ordered      → recent monthly ≥ 20% of peak AND months-to-clear > 3.
    #
    # Peak: average of top-3 months (not single max) to smooth anomalous bulk events.
    # Months-to-clear: uses RECENT demand rate, not peak — shows real clearance horizon.

    _root_cause_map:       dict = {}
    _recent_monthly_map:   dict = {}
    _smoothed_peak_map:    dict = {}

    if not _ds_h.empty:
        _cutoff_90 = _anchor_ds - pd.Timedelta(days=90)
        # Pre-build min days-idle per canonical name (some drugs have duplicate rows)
        _days_idle_lookup: dict = (
            dead_df.groupby("CANONICAL_NAME")["DAYS_IDLE"]
            .min()
            .to_dict()
            if "DAYS_IDLE" in dead_df.columns and not dead_df.empty
            else {}
        )

        for _drug, _grp in _ds_h.groupby("CANONICAL_NAME"):
            _grp = _grp.sort_values("DISPENSED_AT")
            _monthly = _grp.set_index("DISPENSED_AT")["QUANTITY_DISPENSED"].resample("ME").sum()

            # Smoothed peak: average of top-3 months (min 1 month of data needed)
            _top3 = _monthly.nlargest(3)
            _smoothed_peak = float(_top3.mean()) if not _top3.empty else 0.0
            _smoothed_peak_map[_drug] = _smoothed_peak

            _recent_qty = float(_grp[_grp["DISPENSED_AT"] >= _cutoff_90]["QUANTITY_DISPENSED"].sum())

            # Divide by the window the drug was actually active, not the full 90 days.
            # A drug idle for N days was only dispensed during the first (90-N) days of the
            # window — dividing by 3 months deflates the rate and inflates months-to-clear.
            # Floor at 14 days (~a fortnight) to avoid runaway extrapolation from a single
            # dispensing event in the last few days before a drug went idle.
            _min_idle       = int(_days_idle_lookup.get(_drug, 0))
            _active_days    = max(14, 90 - _min_idle)
            _recent_monthly = _recent_qty / (_active_days / 30.0)
            _recent_monthly_map[_drug] = _recent_monthly

            _recent_itr     = _ds_recent_itr_map.get(_drug, 0.0)
            _hist_itr       = _ds_itr_map.get(_drug, 0.0)
            _days_idle_drug = float(_days_idle_lookup.get(_drug, 9999))

            if _smoothed_peak == 0:
                _root_cause_map[_drug] = "Never dispensed"
            elif _recent_itr > 6.0 and _days_idle_drug < 90:
                # Fast mover idle < 90 days: likely between order cycles, not structurally dead
                _root_cause_map[_drug] = "Procurement gap"
            elif _recent_monthly == 0:
                _root_cause_map[_drug] = "Demand ceased"
            elif _recent_monthly < 0.20 * _smoothed_peak:
                _root_cause_map[_drug] = "Demand dropped"
            else:
                _root_cause_map[_drug] = "Over-ordered"

    # Months to clear computed per-row so different SOH values for the same canonical name
    # each get their own figure rather than the last row overwriting the map.
    if not dead_df.empty and "CURRENT_SOH" in dead_df.columns:
        def _calc_months(row):
            _soh = float(row.get("CURRENT_SOH") or 0)
            _rm  = _recent_monthly_map.get(row.get("CANONICAL_NAME"), 0)
            return round(_soh / _rm, 1) if _rm > 0 and _soh > 0 else None
        dead_df["MONTHS_TO_CLEAR"] = dead_df.apply(_calc_months, axis=1)

    # Items whose root cause history comes from before the 730d window are "Dormant"
    # (dead stock query has no date cap on MAX(dispensed_at); _ds_h only covers 730d)
    if not dead_df.empty:
        _days_idle_col = pd.to_numeric(dead_df.get("DAYS_IDLE", pd.Series(dtype=float)), errors="coerce")
        _dormant_mask  = (_days_idle_col > 730) & (~dead_df["CANONICAL_NAME"].isin(_root_cause_map))
        dead_df.loc[_dormant_mask, "_override_cause"] = "Dormant (>2yr idle)"

    # ── Recommended action classification ─────────────────────────────────────
    def _recommend_action(days_idle) -> str:
        try:
            d = float(days_idle)
        except (TypeError, ValueError):
            return "Monitor"
        if d >= 180:
            return "Write off / Return"
        if d >= 90:
            return "Return to supplier"
        if d >= 60:
            return "Reduce next order"
        return "Monitor"

    # ── Drug group mapping (broad categories for scatter colour) ───────────────
    _GROUP_MAP = {
        "antibiotic": "Antimicrobials", "antiviral": "Antimicrobials",
        "antifungal": "Antimicrobials", "antimalarial": "Antimicrobials",
        "antiparasitic": "Antimicrobials", "antimicrobial": "Antimicrobials",
        "analgesic": "Analgesics & Pain", "anaesthetic": "Analgesics & Pain",
        "anesthetic": "Analgesics & Pain", "opioid": "Analgesics & Pain",
        "nsaid": "Analgesics & Pain",
        "cardiovascular": "Cardiovascular", "antihypertensive": "Cardiovascular",
        "diuretic": "Cardiovascular", "cardiac": "Cardiovascular",
        "diabetes": "Metabolic & Endocrine", "antidiabetic": "Metabolic & Endocrine",
        "thyroid": "Metabolic & Endocrine", "vitamin": "Metabolic & Endocrine",
        "supplement": "Metabolic & Endocrine", "mineral": "Metabolic & Endocrine",
        "hormonal": "Metabolic & Endocrine",
        "respiratory": "Respiratory", "bronchodilator": "Respiratory",
        "asthma": "Respiratory",
        "gastrointestinal": "GI & Nutritional", "antacid": "GI & Nutritional",
        "nutritional": "GI & Nutritional",
        "cns": "CNS & Psychiatry", "antiepileptic": "CNS & Psychiatry",
        "antidepressant": "CNS & Psychiatry", "anxiolytic": "CNS & Psychiatry",
        "antipsychotic": "CNS & Psychiatry", "sedative": "CNS & Psychiatry",
        "dermatolog": "Dermatology & Topical", "topical": "Dermatology & Topical",
        "skin": "Dermatology & Topical",
        "ophthalm": "Ophthalmology & ENT", "otic": "Ophthalmology & ENT",
        "antihistamine": "Ophthalmology & ENT",
        "haematin": "Haematology", "iron": "Haematology", "haematol": "Haematology",
        "oxytocic": "Obstetrics & Gynaecology", "gynaecol": "Obstetrics & Gynaecology",
        "contraceptive": "Obstetrics & Gynaecology",
    }

    def _drug_group(tc: str) -> str:
        tc_lower = str(tc or "").lower()
        for kw, grp in _GROUP_MAP.items():
            if kw in tc_lower:
                return grp
        return "Other"

    if not dead_df.empty:
        dead_df["DRUG_GROUP"]      = dead_df.get("THERAPEUTIC_CLASS", pd.Series(dtype=str)).map(_drug_group)
        dead_df["ITR"]             = dead_df["CANONICAL_NAME"].map(_ds_itr_map)
        dead_df["RECENT_ITR"]      = dead_df["CANONICAL_NAME"].map(_ds_recent_itr_map)
        dead_df["ACTION"]          = dead_df["DAYS_IDLE"].map(_recommend_action)
        dead_df["ROOT_CAUSE"]      = dead_df["CANONICAL_NAME"].map(_root_cause_map)
        # Apply dormant override for >730d items with no history in the analysis window
        if "_override_cause" in dead_df.columns:
            dead_df["ROOT_CAUSE"] = dead_df["_override_cause"].combine_first(dead_df["ROOT_CAUSE"])
            dead_df = dead_df.drop(columns=["_override_cause"])
        dead_df["ROOT_CAUSE"]      = dead_df["ROOT_CAUSE"].fillna("No history")
        # Override 1: procurement gap rows whose own DAYS_IDLE >= 90 — the canonical-name
        # lookup uses min(days_idle) so a high-idle sibling can get a false tag.
        _pg_mask = (dead_df["ROOT_CAUSE"] == "Procurement gap") & (pd.to_numeric(dead_df["DAYS_IDLE"], errors="coerce") >= 90)
        dead_df.loc[_pg_mask, "ROOT_CAUSE"] = "Over-ordered"

        # Override 2: procurement gap rows where corrected MTC > 12 months — ITR was measured
        # over 180 days but current 90-day demand is near-zero, indicating genuine over-stock
        # rather than a temporary order gap.  (Stress-tested: this catches only true outliers
        # — RIVAROXABAN 10MG at 107 mo — and creates zero false positives.)
        _pg_mtc_mask = (
            (dead_df["ROOT_CAUSE"] == "Procurement gap")
            & (pd.to_numeric(dead_df["MONTHS_TO_CLEAR"], errors="coerce") > 12)
        )
        dead_df.loc[_pg_mtc_mask, "ROOT_CAUSE"] = "Over-ordered"

        if "MONTHS_TO_CLEAR" not in dead_df.columns:
            dead_df["MONTHS_TO_CLEAR"] = None

    # Procurement gap items stay in dead_df and stat strip counts — they are visible to the
    # pharmacist but labelled so they understand these are likely self-resolving.
    # Excluding them from counts hides real idle inventory from the headline KPIs.
    dead_only  = dead_df[dead_df.get("IDLE_CATEGORY", pd.Series(dtype=str)) == "dead"] if not dead_df.empty else pd.DataFrame()
    slow_only  = dead_df[dead_df.get("IDLE_CATEGORY", pd.Series(dtype=str)) == "slow"] if not dead_df.empty else pd.DataFrame()
    dead_value = dead_only["STOCK_VALUE"].sum() if not dead_only.empty and "STOCK_VALUE" in dead_only.columns else 0
    slow_value = slow_only["STOCK_VALUE"].sum() if not slow_only.empty and "STOCK_VALUE" in slow_only.columns else 0
    _total_idle_value = dead_value + slow_value

    stat_strip([
        {"label": "Dead stock SKUs (90d+)", "value": fmt_int(len(dead_only)),
         "hint": "Capital at risk", "hint_good": len(dead_only) == 0,
         "accent_color": "#991B1B" if len(dead_only) else "#111827"},
        {"label": "Slow moving (30–90d)", "value": fmt_int(len(slow_only)),
         "accent_color": "#D97706" if len(slow_only) else "#111827"},
        {"label": "Dead stock value",     "value": fmt_kes_millions(dead_value)},
        {"label": "Slow moving value",    "value": fmt_kes_millions(slow_value)},
        {"label": "Total idle capital",   "value": fmt_kes_millions(_total_idle_value),
         "hint": "Dead + slow moving"},
        {"label": "Facility ITR (all)",   "value": f"{round(_ds_median_itr, 1)}×" if _ds_median_itr else "—",
         "hint": "Under 6x signals over-procurement"},
    ])

    # ── Root cause analysis (lead with the why) ───────────────────────────────
    _RC_ORDER = [
        "Procurement gap", "Never dispensed", "Demand ceased",
        "Demand dropped", "Over-ordered", "Dormant (>2yr idle)", "No history",
    ]
    _rc_colors = {
        "Procurement gap":       "#0369A1",
        "Never dispensed":       "#7F1D1D",
        "Demand ceased":         "#991B1B",
        "Demand dropped":        "#C2410C",
        "Over-ordered":          "#D97706",
        "Dormant (>2yr idle)":   "#6B21A8",
        "No history":            "#9CA3AF",
    }
    if not dead_df.empty and "ROOT_CAUSE" in dead_df.columns:
        section_header("Root cause breakdown")

        _rc_val_col = "STOCK_VALUE" if "STOCK_VALUE" in dead_df.columns else "TOTAL_HISTORICAL_VALUE"
        _rc_summary = (
            dead_df.groupby("ROOT_CAUSE")
            .agg(
                SKUs=("CANONICAL_NAME", "count"),
                Value=(_rc_val_col, "sum"),
                Avg_months_to_clear=("MONTHS_TO_CLEAR", "median"),
            )
            .reset_index()
        )
        _rc_summary["_sort"] = _rc_summary["ROOT_CAUSE"].map(
            lambda c: _RC_ORDER.index(c) if c in _RC_ORDER else 99
        )
        _rc_summary = _rc_summary.sort_values("_sort").drop(columns=["_sort"])

        # Categories where "months to clear" is semantically undefined
        _NO_CLEAR_CAUSES = {"Never dispensed", "Demand ceased", "No history", "Dormant (>2yr idle)"}

        _rc_cols = st.columns(max(len(_rc_summary), 1))
        for _col, (_, _rc_row) in zip(_rc_cols, _rc_summary.iterrows()):
            _cause  = _rc_row["ROOT_CAUSE"]
            _skus   = int(_rc_row["SKUs"])
            _val    = _rc_row["Value"]
            _months = _rc_row["Avg_months_to_clear"]
            _clr    = _rc_colors.get(_cause, "#6B7280")
            _show_clear = (
                _cause not in _NO_CLEAR_CAUSES
                and _months is not None
                and not pd.isna(_months)
            )
            _col.markdown(
                f"""
                <div style="border:1px solid #E5E7EB;border-radius:8px;padding:14px;">
                  <div style="font-size:10px;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:0.05em;">{_cause}</div>
                  <div style="font-size:26px;font-weight:700;color:{_clr};margin:4px 0 2px;">{_skus}</div>
                  <div style="font-size:11px;color:#374151;">SKUs · {fmt_kes_millions(_val)}</div>
                  {"<div style='font-size:10px;color:#9CA3AF;margin-top:3px;'>Clears in " + (">3yr" if _months > 36 else "~" + str(round(_months, 1)) + " mo") + "</div>" if _show_clear else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    section_header("Idle inventory map — by drug group")
    st.caption("X = days idle · Y = units on shelf · bubble size = stock value · top-right = highest priority · colour = drug group")
    if not dead_df.empty:
        st.plotly_chart(dead_stock_scatter(dead_df, color_col="DRUG_GROUP"), use_container_width=True)
    else:
        empty_state(f"No idle products beyond {idle_threshold} days.", icon="✅")

    section_header("Priority actions — by days idle")
    st.caption("Bar length = days since last dispensed · colour = recommended action · hover for stock value")
    if not dead_df.empty:
        st.plotly_chart(dead_stock_action_bars(dead_df), use_container_width=True)
    else:
        empty_state("No idle inventory found.", icon="✅")

    # ── Full detail — unified filterable table (replaces per-cause expanders) ─
    if not dead_df.empty:
        st.markdown("---")
        section_header("Full inventory detail")

        _available_causes = dead_df["ROOT_CAUSE"].unique().tolist() if "ROOT_CAUSE" in dead_df.columns else []
        _filter_options   = ["All"] + [c for c in _RC_ORDER if c in _available_causes]

        _f_col, _s_col = st.columns([2, 3])
        with _f_col:
            _rc_filter = st.selectbox(
                "Root cause", _filter_options,
                label_visibility="collapsed",
                key="ds_rc_filter",
            )
        with _s_col:
            _drug_search = st.text_input(
                "Search drug", placeholder="e.g. Phenytoin",
                label_visibility="collapsed",
                key="ds_drug_search",
            )

        _detail_df = dead_df.copy()
        if _rc_filter != "All" and "ROOT_CAUSE" in _detail_df.columns:
            _detail_df = _detail_df[_detail_df["ROOT_CAUSE"] == _rc_filter]
        if _drug_search:
            _detail_df = _detail_df[
                _detail_df["CANONICAL_NAME"].str.contains(_drug_search, case=False, na=False)
            ]
        _detail_df = _detail_df.sort_values("DAYS_IDLE", ascending=False)

        _show_cols = [c for c in [
            "CANONICAL_NAME", "DRUG_GROUP", "ROOT_CAUSE",
            "DAYS_IDLE", "CURRENT_SOH", "STOCK_VALUE",
            "MONTHS_TO_CLEAR", "RECENT_ITR", "ITR",
        ] if c in _detail_df.columns]
        _disp_detail = _detail_df[_show_cols].copy()
        if "CANONICAL_NAME" in _disp_detail.columns:
            _disp_detail["CANONICAL_NAME"] = _disp_detail["CANONICAL_NAME"].map(fmt_drug_name)

        # Cap display at 36 months (>3yr) — underlying value stays intact for sorting/export
        if "MONTHS_TO_CLEAR" in _disp_detail.columns:
            _mtc_num = pd.to_numeric(_disp_detail["MONTHS_TO_CLEAR"], errors="coerce")
            _disp_detail["MONTHS_TO_CLEAR"] = _mtc_num.where(_mtc_num <= 36, other=None)

        st.dataframe(
            _disp_detail.rename(columns={
                "CANONICAL_NAME": "Drug", "DRUG_GROUP": "Group",
                "ROOT_CAUSE": "Root cause",
                "DAYS_IDLE": "Days idle", "CURRENT_SOH": "SOH",
                "STOCK_VALUE": "Stock value (KES)",
                "MONTHS_TO_CLEAR": "Months to clear",
                "RECENT_ITR": "Recent ITR (6mo)",
                "ITR": "Hist. ITR (2yr)",
            }),
            use_container_width=True, hide_index=True,
            column_config={
                "Days idle":         st.column_config.NumberColumn(format="%d d"),
                "Stock value (KES)": st.column_config.NumberColumn(format="KES %,.0f"),
                "SOH":               st.column_config.NumberColumn(format="%,.1f"),
                "Months to clear":   st.column_config.NumberColumn(format="%.1f mo",
                                        help="At current 90-day demand rate · blank = >3yr or no recent demand"),
                "Recent ITR (6mo)":  st.column_config.NumberColumn(format="%.1f×",
                                        help="Inventory turns in last 6 months — 0 = no recent dispensing"),
                "Hist. ITR (2yr)":   st.column_config.NumberColumn(format="%.1f×",
                                        help="Inventory turns over full 2-year history"),
            },
        )

        _export_col_map = {
            "CANONICAL_NAME": "Drug", "DRUG_GROUP": "Group", "ROOT_CAUSE": "Root cause",
            "DAYS_IDLE": "Days idle", "CURRENT_SOH": "SOH", "STOCK_VALUE": "Stock value (KES)",
            "MONTHS_TO_CLEAR": "Months to clear", "RECENT_ITR": "Recent ITR (6mo)", "ITR": "Hist. ITR (2yr)",
        }
        _export_df = _detail_df[
            [c for c in _export_col_map if c in _detail_df.columns]
        ].rename(columns=_export_col_map)
        st.download_button(
            "📥 Export (CSV)", _export_df.to_csv(index=False),
            file_name=f"dead_stock_{fac.schema}_{pd.Timestamp.now().date()}.csv",
            mime="text/csv",
        )


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
         "hint": "All risk tiers", "hint_good": len(exposure_df) == 0,
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

    # ── Overdue refill contact list ───────────────────────────────────────────────
    section_header("Overdue refill — patient contact list")
    st.caption(
        "Patients overdue for a refill on a drug with less than 7 days of stock remaining. "
        "Use patient IDs to initiate follow-up."
    )

    @st.cache_data(ttl=3600, show_spinner=False)
    def _pr_load_overdue_list(schema, ref_date):
        try:
            return get_overdue_patient_list(schema, ref_date=ref_date)
        except Exception:
            return pd.DataFrame()

    with st.spinner("Loading overdue patient list…"):
        overdue_list_df = _pr_load_overdue_list(fac.schema, _ref_date)

    if overdue_list_df.empty:
        st.info("No overdue refill patients found with current data.", icon="✅")
    else:
        overdue_list_df.columns = overdue_list_df.columns.str.upper()

        # Drug filter
        _drug_opts = sorted(overdue_list_df["CANONICAL_NAME"].dropna().unique().tolist())
        _drug_filter = st.multiselect(
            "Filter by drug", options=_drug_opts,
            placeholder="All drugs",
        )
        _ol_view = overdue_list_df.copy()
        if _drug_filter:
            _ol_view = _ol_view[_ol_view["CANONICAL_NAME"].isin(_drug_filter)]

        _ol_show_cols = [c for c in [
            "PATIENT_ID", "CANONICAL_NAME", "THERAPEUTIC_CLASS",
            "LAST_DISPENSED", "LAST_QTY_DISPENSED", "ESTIMATED_SUPPLY_DAYS",
            "DAYS_OVERDUE", "DAYS_OF_COVER",
        ] if c in _ol_view.columns]

        st.dataframe(
            _ol_view[_ol_show_cols].rename(columns={
                "PATIENT_ID":              "Patient ID",
                "CANONICAL_NAME":          "Drug",
                "THERAPEUTIC_CLASS":       "Class",
                "LAST_DISPENSED":          "Last dispensed",
                "LAST_QTY_DISPENSED":      "Last qty",
                "ESTIMATED_SUPPLY_DAYS":   "Est. supply (d)",
                "DAYS_OVERDUE":            "Days overdue",
                "DAYS_OF_COVER":           "Drug cover (d)",
            }),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Last qty":       st.column_config.NumberColumn(format="%d units"),
                "Est. supply (d)":st.column_config.NumberColumn(
                    format="%d d",
                    help="(Last qty ÷ avg qty per visit) × mean days between visits. "
                         "Capped at 180d. Reflects the patient's personal refill rhythm, "
                         "not a facility-wide average."
                ),
                "Days overdue":   st.column_config.NumberColumn(
                    format="%d d",
                    help="Days past the estimated supply × 1.2 grace buffer."
                ),
                "Drug cover (d)": st.column_config.NumberColumn(format="%.1f d"),
            },
        )

        _dl_cols = st.columns(2)
        with _dl_cols[0]:
            st.download_button(
                "📥 Export contact list (CSV)",
                _ol_view[_ol_show_cols].to_csv(index=False),
                file_name=f"overdue_patients_{fac.schema}_{pd.Timestamp.now().date()}.csv",
                mime="text/csv",
            )

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
        f"C items = remaining."
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
                    "Total value (KES)":  st.column_config.NumberColumn(format="KES %,.0f"),
                    "Cumulative %":       st.column_config.ProgressColumn(format="%.1%", min_value=0, max_value=1),
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
    section_header("Seasonal demand outlook — Kisumu disease calendar")
    st.caption(
        "Historical disease peaks for Kisumu County mapped to affected drug classes. "
        "Values show expected demand uplift at peak vs baseline. "
        "Order Workbench automatically adjusts quantities for products in the warning window."
    )

    if not _seasonal_outlook.empty:
        # Style: teal highlight for peak months, plain for non-peak
        def _style_cell(val):
            if val == "—":
                return "color: #9CA3AF"
            return "color: #0F6E56; font-weight: 600"

        st.dataframe(
            _seasonal_outlook.style.map(_style_cell),
            use_container_width=True,
        )
        if _climate_signal:
            _anom = _climate_signal.anomaly_pct
            if abs(_anom) >= 20:
                st.caption(
                    f"Climate: Kisumu rainfall in {_climate_signal.current_month_name} is "
                    f"{abs(_anom):.0f}% {_climate_signal.anomaly_label} "
                    f"({_climate_signal.current_month_mm}mm vs {_climate_signal.historical_avg_mm}mm historical average). "
                    f"Source: {_climate_signal.data_source}."
                )
        if _seasonal_summaries:
            st.caption(
                f"**{len(_seasonal_summaries)} disease season(s) within warning window** — "
                "order quantities in the Workbench are already adjusted."
            )
    else:
        empty_state("No seasonal data available for this facility.", icon="📅")

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
