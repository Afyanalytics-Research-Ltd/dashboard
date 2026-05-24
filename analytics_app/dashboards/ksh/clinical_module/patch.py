"""
views.py — Afya Clinical Analytics
=====================================
Render functions only. All SQL lives in queries.py.
Each function calls queries.load_* and builds charts.

  render_tab1_operations
  render_tab2_segmentation
  render_tab3_retention
  render_tab4_disease_burden
  render_tab5_workload
  render_clinician_view
"""

import pandas as pd
import streamlit as st

import queries as Q
from ui_template import AFYA_BLUE, TEAL, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY, BG_LIGHT, BORDER
from charts import (
    line_chart, bar_chart, hbar_chart, stacked_bar, stacked_area,
    funnel_chart, heatmap, scatter, donut, table_fig, bullet, sparkline,
    BURDEN_COLORS, LIFECYCLE_COLORS,
)


# ─── SHARED HELPERS ───────────────────────────────────────────────────────────

def _H():
    return st.session_state.get("helpers", {})

def _gap(px=10):   _H().get("gap",      lambda px=10: None)(px)
def _sh(t, mt=0):  _H().get("sh",       lambda t, mt=0: None)(t, mt)
def _kpi(l, v, s="", color=AFYA_BLUE): _H().get("kpi_card", lambda l,v,s="",c=AFYA_BLUE: None)(l,v,s,color)
def _pc(fig):      _H().get("pc",       lambda f: st.plotly_chart(f, use_container_width=True))(fig)
def _note(t, w=False): _H().get("note", lambda t,warn=False: None)(t, w)
def _n(v):         return _H().get("fmt_num",  lambda v: str(v))(v)
def _p(v, d=1):    return _H().get("fmt_pct",  lambda v,d=1: str(v))(v, d)
def _k(v):         return _H().get("fmt_kes",  lambda v: str(v))(v)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_tab1_operations(filters: dict, run_query):
    # ── KPI ROW ───────────────────────────────────────────────────────────
    _sh("Key Metrics")
    try:
        df = Q.load_tab1_kpis(filters, run_query)
        if not df.empty:
            row = df.iloc[0]
            c1,c2,c3,c4 = st.columns(4)
            with c1: _kpi("Total Visits",    _n(row.get("total_visits")))
            with c2: _kpi("Inpatient",       _n(row.get("inpatient_visits")),
                          _p(row.get("inpatient_pct")) + " of visits", TEAL)
            with c3: _kpi("Outpatient",      _n(row.get("outpatient_visits")))
            with c4: _kpi("Avg OP Cost",     _k(row.get("avg_op_cost")))
    except Exception as e:
        st.warning(f"KPIs: {e}")

    _gap(16)

    # ── SECTION A: SERVICE GROWTH ─────────────────────────────────────────
    _sh("A — Service Growth by Type", mt=8)
    _note("Monthly visit volume split by service type proxy.")
    try:
        df = Q.load_service_growth(filters, run_query)
        if not df.empty:
            _pc(stacked_bar(
                df, x="visit_month",
                categories=["inpatient","outpatient_with_lab",
                             "outpatient_rx_only","consult_only"],
                color_map={
                    "inpatient":           AFYA_BLUE,
                    "outpatient_with_lab": TEAL,
                    "outpatient_rx_only":  ORANGE,
                    "consult_only":        GRAY,
                },
                y_label="Visits", x_label="Month", height=300,
            ))
    except Exception as e:
        st.warning(f"Service growth: {e}")

    _gap(12)

    # ── SECTION A: WARD BREAKDOWN ─────────────────────────────────────────
    _sh("Ward Breakdown", mt=8)
    try:
        df = Q.load_ward_breakdown(filters, run_query)
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                _pc(hbar_chart(df, x="admissions", y="ward",
                               color=AFYA_BLUE, x_label="Admissions",
                               height=300, top_n=10))
            with c2:
                _pc(bar_chart(
                    df.head(10), x="ward",
                    y=["avg_los_days","avg_discharge_latency_hrs"],
                    color_map={"avg_los_days": AFYA_BLUE,
                               "avg_discharge_latency_hrs": ORANGE},
                    y_label="Days / Hours", height=300,
                ))
    except Exception as e:
        st.warning(f"Ward breakdown: {e}")

    _gap(12)

    # ── SECTION B: SPIKE & DIP ────────────────────────────────────────────
    _sh("B — Volume Spikes & Dips", mt=8)
    _note("Months deviating >1.5 SD from the mean. Cause derived from disease mix.")
    try:
        df = Q.load_spike_dip(filters, run_query)
        if not df.empty:
            _pc(line_chart(df, x="visit_month", y="total_visits",
                           y_label="Visits", spike=True,
                           spike_col="is_spike", dip_col="is_dip", height=280))
            spikes = df[df["is_spike"] == 1]
            dips   = df[df["is_dip"]   == 1]
            if not spikes.empty or not dips.empty:
                c1, c2 = st.columns(2)
                with c1:
                    if not spikes.empty:
                        st.markdown("**Spike months**")
                        _pc(table_fig(
                            spikes[["visit_month","total_visits","z_score","spike_cause"]],
                            col_labels={"visit_month":"Month","total_visits":"Visits",
                                        "z_score":"Z","spike_cause":"Cause"},
                            height=180,
                        ))
                with c2:
                    if not dips.empty:
                        st.markdown("**Dip months**")
                        _pc(table_fig(
                            dips[["visit_month","total_visits","z_score","dip_cause"]],
                            col_labels={"visit_month":"Month","total_visits":"Visits",
                                        "z_score":"Z","dip_cause":"Cause"},
                            height=180,
                        ))
    except Exception as e:
        st.warning(f"Spike/dip: {e}")

    _gap(12)

    # ── SECTION C: JOURNEY TIMES ──────────────────────────────────────────
    _sh("C — Patient Journey Times", mt=8)
    _note("Inpatient journey starts from admission time. Caps: 12h OP, 48h IP.")
    try:
        df = Q.load_journey_times(filters, run_query)
        if not df.empty:
            stage_cols   = ["avg_hrs_to_triage","avg_hrs_triage_to_consult",
                            "avg_hrs_consult_to_lab","avg_hrs_lab_turnaround"]
            stage_labels = ["Arrival → Triage","Triage → Consult",
                            "Consult → Lab Result","Lab Turnaround"]
            c1, c2 = st.columns(2)
            for i, vtype in enumerate(df["visit_type"].unique()):
                row  = df[df["visit_type"] == vtype].iloc[0]
                vals = [row.get(c) for c in stage_cols]
                col  = c1 if i == 0 else c2
                with col:
                    st.markdown(f"**{vtype}**")
                    _pc(bar_chart(
                        pd.DataFrame({"stage": stage_labels, "hrs": vals}),
                        x="stage", y="hrs",
                        color=AFYA_BLUE if vtype == "Inpatient" else TEAL,
                        y_label="Avg Hours", height=240, show_text=True,
                    ))
    except Exception as e:
        st.warning(f"Journey times: {e}")

    _gap(12)

    # ── SECTION C: INPATIENT CONVERSION FUNNEL ────────────────────────────
    _sh("Inpatient Conversion Funnel", mt=8)
    _note(
        "Gate 2 — 'Saw a doctor' proxied by doctor note recorded. "
        "Tenri note coverage is lower due to recording behaviour, "
        "not necessarily lower consultation rate.",
        w=True,
    )
    try:
        df = Q.load_inpatient_funnel(filters, run_query)
        if not df.empty:
            row = df.iloc[0]
            g = [int(row.get(f"g{i}_{k}") or 0) for i, k in [
                (1,"total_visits"),(2,"note_recorded"),
                (3,"admitted"),(4,"with_investigation"),(5,"resulted_24h"),
            ]]
            c1, c2 = st.columns([1.2, 1])
            with c1:
                _pc(funnel_chart(
                    labels=["Visit Created","Saw a Doctor",
                             "Admitted","Investigation Ordered","Resulted 24h"],
                    values=g,
                    caveat=f"Note coverage: {row.get('g2_note_coverage_pct','?')}%",
                    height=360,
                ))
            with c2:
                _kpi("Consult → Not Admitted",
                     _n(row.get("dropoff_consult_not_admitted")))
                _gap(6)
                _kpi("Admitted → No Investigation",
                     _n(row.get("dropoff_no_investigation")))
                _gap(6)
                _kpi("Investigation Delayed >24h",
                     _n(row.get("dropoff_investigation_delayed")))
                _gap(6)
                _kpi("Consult → Admit Rate",
                     _p(row.get("g3_consult_to_admit_pct")), color=TEAL)
                _gap(6)
                _kpi("Avg Admission Cost",
                     _k(row.get("avg_admission_cost")))
    except Exception as e:
        st.warning(f"Funnel: {e}")

    _gap(12)

    # ── SECTION D: FORECAST ───────────────────────────────────────────────
    _sh("D — Encounter Forecast", mt=8)
    _note("Trend × seasonal index × disease-load adjustment.")
    try:
        df = Q.load_encounter_forecast(filters, run_query)
        if not df.empty:
            df["forecast"] = (df["trend_component"] * df["seasonal_index"]).round(0)
            _pc(line_chart(df, x="visit_month", y=["actual_visits","forecast"],
                           color_map={"actual_visits": AFYA_BLUE, "forecast": ORANGE},
                           y_label="Visits", height=300))
            latest = df.iloc[-1]
            c1,c2,c3 = st.columns(3)
            with c1: _kpi("Current Month",      _n(latest.get("actual_visits")))
            with c2: _kpi("Next Month Forecast",_n(latest.get("forecast")), color=ORANGE)
            with c3: _kpi("Seasonal Index",     f"{latest.get('seasonal_index',1.0):.2f}")
    except Exception as e:
        st.warning(f"Forecast: {e}")

    _gap(12)

    # ── SECTION E: CLINICIAN LOAD ─────────────────────────────────────────
    _sh("E — Clinician Load Variance", mt=8)
    try:
        df = Q.load_clinician_load(filters, run_query)
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                _pc(hbar_chart(df, x="avg_daily_patients", y="clinician",
                               color=AFYA_BLUE, x_label="Avg Daily Patients",
                               height=300))
            with c2:
                _pc(bar_chart(
                    df.head(15), x="clinician",
                    y=["vitals_pct_on_normal","vitals_pct_on_surge"],
                    color_map={"vitals_pct_on_normal": TEAL, "vitals_pct_on_surge": CORAL},
                    y_label="Vitals Recording %", height=300,
                ))
    except Exception as e:
        st.warning(f"Clinician load: {e}")

    _gap(12)

    # ── SECTION F: HEATMAP ────────────────────────────────────────────────
    _sh("F — Peak Demand Heatmap (EAT)", mt=8)
    try:
        df = Q.load_peak_demand_heatmap(filters, run_query)
        if not df.empty:
            day_order = ["Sunday","Monday","Tuesday","Wednesday",
                         "Thursday","Friday","Saturday"]
            _pc(heatmap(df, x="hour_of_day", y="day_name", z="visit_count",
                        day_order=day_order, height=280))
    except Exception as e:
        st.warning(f"Heatmap: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def render_tab2_segmentation(filters: dict, run_query):
    _sh("Patient Overview")
    try:
        df = Q.load_seg_kpis(filters, run_query)
        if not df.empty:
            row = df.iloc[0]
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: _kpi("Total Patients",  _n(row.get("total_patients")))
            with c2: _kpi("Chronic",         _n(row.get("chronic_patients")),
                          _p(row.get("chronic_rate_pct")), AFYA_BLUE)
            with c3: _kpi("Repeat Patients", _n(row.get("repeat_patients")),
                          _p(row.get("repeat_rate_pct")), TEAL)
            with c4: _kpi("Single Visit",    _n(row.get("single_visit")))
            with c5: _kpi("Avg Visits / Pt", str(row.get("avg_visits","—")))
    except Exception as e:
        st.warning(f"Seg KPIs: {e}")

    _gap(16)

    _sh("A — Demographics: Age × Sex × Chronic Status", mt=8)
    try:
        df = Q.load_demographics_age_sex(filters, run_query)
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                pivot = df.pivot_table(index="age_group", columns="sex",
                                       values="total", aggfunc="sum", fill_value=0)
                _pc(bar_chart(pivot.reset_index(), x="age_group",
                              y=list(pivot.columns),
                              color_map={"F":"#7b5ea7","FEMALE":"#7b5ea7",
                                         "M":TEAL,"MALE":TEAL},
                              y_label="Patients", height=300))
            with c2:
                pivot2 = df.pivot_table(index="age_group",
                                        values=["chronic","non_chronic"],
                                        aggfunc="sum", fill_value=0)
                _pc(stacked_bar(pivot2.reset_index(), x="age_group",
                                categories=["chronic","non_chronic"],
                                color_map={"chronic": AFYA_BLUE,
                                           "non_chronic": "#80b3e6"},
                                y_label="Patients", height=300))
    except Exception as e:
        st.warning(f"Demographics: {e}")

    _gap(12)

    _sh("B — New vs Returning Trend", mt=8)
    try:
        df = Q.load_new_vs_returning(filters, run_query)
        if not df.empty:
            df["new_pct"] = (
                df["new_patients"] / df["total_patients"].replace(0, float("nan")) * 100
            ).round(1)
            c1, c2 = st.columns(2)
            with c1:
                _pc(line_chart(df, x="visit_month",
                               y=["new_patients","returning_patients"],
                               color_map={"new_patients": ORANGE,
                                          "returning_patients": TEAL},
                               y_label="Patients", height=280))
            with c2:
                _pc(line_chart(df, x="visit_month", y="new_pct",
                               y_label="New Patient %", y_format="pct",
                               height=280))
    except Exception as e:
        st.warning(f"New vs returning: {e}")

    _gap(12)

    _sh("C — Payer Mix by Age Group", mt=8)
    try:
        df = Q.load_payer_mix(filters, run_query)
        if not df.empty:
            pivot = df.pivot_table(index="age_group", columns="payer_type",
                                   values="unique_patients", aggfunc="sum", fill_value=0)
            _pc(stacked_bar(pivot.reset_index(), x="age_group",
                            categories=list(pivot.columns),
                            color_map={"Cash": ORANGE, "NHIF / SHA": TEAL,
                                       "Insurance": AFYA_BLUE},
                            y_label="Patients", height=300))
    except Exception as e:
        st.warning(f"Payer mix: {e}")

    _gap(12)

    _sh("D — Revenue by Clinical Segment", mt=8)
    _note("Revenue anchored to clinical cause.")
    try:
        df = Q.load_revenue_by_segment(filters, run_query)
        if not df.empty:
            top10 = (df.groupby("primary_condition")["total_revenue"]
                     .sum().nlargest(10).reset_index())
            _pc(hbar_chart(top10, x="total_revenue", y="primary_condition",
                           x_label="Total Revenue (KES)", y_format="KES",
                           height=320))
    except Exception as e:
        st.warning(f"Revenue by segment: {e}")

    _gap(12)

    _sh("E — Revenue Concentration (Pareto)", mt=8)
    try:
        df = Q.load_pareto(filters, run_query)
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                _pc(donut(labels=df["revenue_tier"].tolist(),
                          values=df["tier_revenue"].tolist(),
                          color_map={"Top 10%": AFYA_BLUE, "Top 11–20%": TEAL,
                                     "Middle 21–50%": ORANGE, "Bottom 50%": GRAY},
                          height=280))
            with c2:
                _pc(table_fig(df,
                              col_labels={"revenue_tier":"Tier",
                                          "patient_count":"Patients",
                                          "tier_revenue":"Revenue",
                                          "revenue_share_pct":"Share %",
                                          "avg_spend":"Avg Spend",
                                          "avg_visits":"Avg Visits"},
                              fmt={"tier_revenue":"KES","avg_spend":"KES",
                                   "revenue_share_pct":"pct"},
                              height=220))
    except Exception as e:
        st.warning(f"Pareto: {e}")

    _gap(12)

    _sh("F — Age Cohort Growth", mt=8)
    try:
        df = Q.load_cohort_forecast(filters, run_query)
        if not df.empty:
            pivot = df.pivot_table(index="visit_month", columns="age_cohort",
                                   values="patient_count", aggfunc="sum", fill_value=0)
            cohort_colors = {"Paediatric (<18)": PURPLE,
                             "Young Adult (18–34)": TEAL,
                             "Adult (35–54)": AFYA_BLUE,
                             "Senior (55+)": ORANGE}
            _pc(stacked_area(pivot.reset_index(), x="visit_month",
                             categories=[c for c in cohort_colors if c in pivot.columns],
                             color_map=cohort_colors, y_label="Patients", height=300))
    except Exception as e:
        st.warning(f"Cohort forecast: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PATIENT FLOW & RETENTION
# ══════════════════════════════════════════════════════════════════════════════

def render_tab3_retention(filters: dict, run_query):
    _sh("Retention Overview")
    try:
        df = Q.load_retention_kpis(filters, run_query)
        if not df.empty:
            row = df.iloc[0]
            rr  = float(row.get("retention_rate_pct") or 0)
            c1,c2,c3,c4 = st.columns(4)
            with c1: _kpi("Chronic Patients", _n(row.get("chronic_patients")))
            with c2: _kpi("Retained (90d)",   _n(row.get("retained_patients")),
                          _p(rr) + " — benchmark 60%",
                          TEAL if rr >= 60 else CORAL)
            with c3: _kpi("LTFU",             _n(row.get("ltfu_patients")), color=CORAL)
            with c4: _pc(bullet(rr, 60, "Retention vs 60%", format="pct", height=100))
    except Exception as e:
        st.warning(f"Retention KPIs: {e}")

    _gap(16)

    _sh("A — Patient Lifecycle", mt=8)
    try:
        df = Q.load_lifecycle(filters, run_query)
        if not df.empty:
            c1, c2 = st.columns(2)
            with c1:
                _pc(bar_chart(df, x="lifecycle_status", y="patient_count",
                              color=AFYA_BLUE, y_label="Chronic Patients",
                              height=280, show_text=True))
            with c2:
                _pc(donut(labels=df["lifecycle_status"].tolist(),
                          values=df["patient_count"].tolist(),
                          color_map=LIFECYCLE_COLORS, height=280))
    except Exception as e:
        st.warning(f"Lifecycle: {e}")

    _gap(12)

    _sh("Retention by Payer Type", mt=8)
    _note("Cash patients typically have lower retention than insured patients.")
    try:
        df = Q.load_retention_by_payer(filters, run_query)
        if not df.empty:
            _pc(bar_chart(df, x="payer_type", y="retention_pct",
                          color_map={"NHIF / SHA": TEAL,
                                     "Cash": CORAL, "Insurance": AFYA_BLUE},
                          y_label="90-Day Retention %", y_format="pct",
                          height=260, show_text=True))
    except Exception as e:
        st.warning(f"Retention by payer: {e}")

    _gap(12)

    _sh("B — Dropout Cause Attribution", mt=8)
    _note("Four causes tested per LTFU patient. A patient can trigger multiple.")
    try:
        df = Q.load_dropout_causes(filters, run_query)
        if not df.empty and df.iloc[0].get("total_ltfu", 0):
            row   = df.iloc[0]
            total = int(row.get("total_ltfu") or 1)
            causes = {"Medication Gap": int(row.get("ltfu_rx_gap") or 0),
                      "Fragmented Care": int(row.get("ltfu_fragmented_care") or 0),
                      "Uncontrolled BP": int(row.get("ltfu_uncontrolled_bp") or 0)}
            cdf = pd.DataFrame({"cause": list(causes.keys()),
                                 "count": list(causes.values())})
            cdf["pct"] = (cdf["count"] / total * 100).round(1)
            _pc(hbar_chart(cdf.sort_values("pct"), x="pct", y="cause",
                           x_label="% of LTFU Patients", y_format="pct",
                           color=CORAL, height=220, show_text=True))
            _note(f"Total LTFU chronic patients: {total:,}")
    except Exception as e:
        st.warning(f"Dropout causes: {e}")

    _gap(12)

    _sh("C — Revenue at Risk from LTFU", mt=8)
    try:
        df = Q.load_revenue_at_risk(filters, run_query)
        if not df.empty:
            row = df.iloc[0]
            c1,c2,c3 = st.columns(3)
            with c1: _kpi("Chronic LTFU",       _n(row.get("chronic_ltfu")), color=CORAL)
            with c2: _kpi("Revenue at Risk",     _k(row.get("chronic_ltfu_revenue_at_risk")),
                          "annual estimate", CORAL)
            with c3: _kpi("Recoverable (31–90d)",_k(row.get("lapsing_revenue_recoverable")),
                          "still reachable", ORANGE)
    except Exception as e:
        st.warning(f"Revenue at risk: {e}")

    _gap(12)

    _sh("D — Re-engagement Outreach List", mt=8)
    _note("Campaign A: 30–60 days lapsed. B: 61–90 days. Sorted by priority score.")
    try:
        df = Q.load_outreach_list(filters, run_query)
        if not df.empty:
            camp_a = df[df["campaign"].str.contains("Campaign A")]
            camp_b = df[df["campaign"].str.contains("Campaign B")]
            c1, c2 = st.columns(2)
            for col, camp, label in [(c1, camp_a, "Campaign A"), (c2, camp_b, "Campaign B")]:
                with col:
                    st.markdown(f"**{label} — {len(camp)} patients**")
                    if not camp.empty:
                        _pc(table_fig(
                            camp[["patient","days_since","primary_condition",
                                  "rx_completion_pct","priority_score"]].head(20),
                            col_labels={"patient":"Patient","days_since":"Days",
                                        "primary_condition":"Condition",
                                        "rx_completion_pct":"Rx %",
                                        "priority_score":"Score"},
                            fmt={"rx_completion_pct":"pct"},
                            height=420,
                        ))
    except Exception as e:
        st.warning(f"Outreach list: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DISEASE BURDEN
# ══════════════════════════════════════════════════════════════════════════════

def render_tab4_disease_burden(filters: dict, run_query):
    st_a, st_b, st_c, st_d, st_e, st_f = st.tabs([
        "Overview","NCD & Chronic","RMNCH",
        "Communicable & HIV","Mental Health & Oncology",
        "Revenue & Investigations",
    ])

    with st_a:
        _sh("A — Disease Burden Overview")
        try:
            df = Q.load_burden_kpis(filters, run_query)
            if not df.empty:
                row = df.iloc[0]
                c1,c2,c3,c4,c5 = st.columns(5)
                with c1: _kpi("Diagnosed Visits",   _n(row.get("total_diagnosed")))
                with c2: _kpi("Comorbidity Rate",   _p(row.get("comorbidity_rate_pct")), color=ORANGE)
                with c3: _kpi("NCD Share",          _p(row.get("ncd_share_pct")), color=AFYA_BLUE)
                with c4: _kpi("Communicable Share", _p(row.get("communicable_share_pct")), color=TEAL)
                with c5: _kpi("Undetected NCD",     _n(row.get("undetected_ncd")),
                              "elevated vitals, no NCD code", CORAL)
        except Exception as e:
            st.warning(f"A1: {e}")

        _gap(12)
        _sh("Burden Group Monthly Trend", mt=8)
        try:
            df = Q.load_burden_trend(filters, run_query)
            if not df.empty:
                pivot = df.pivot_table(index="visit_month", columns="burden_group",
                                       values="visit_count", aggfunc="sum", fill_value=0)
                top6 = df.groupby("burden_group")["visit_count"].sum().nlargest(6).index.tolist()
                _pc(stacked_area(pivot.reset_index(), x="visit_month",
                                 categories=[g for g in top6 if g in pivot.columns],
                                 color_map=BURDEN_COLORS, y_label="Visits", height=320))
        except Exception as e:
            st.warning(f"A2: {e}")

        _gap(12)
        _sh("Top 10 Diagnoses", mt=8)
        try:
            df = Q.load_top_diagnoses(filters, run_query)
            if not df.empty:
                _pc(hbar_chart(df, x="visit_count", y="disease_group",
                               color=AFYA_BLUE, x_label="Visits", height=320))
        except Exception as e:
            st.warning(f"A3: {e}")

        _gap(12)
        _sh("Undetected NCD — Elevated Vitals Without NCD Code", mt=8)
        _note("Clinical miss and billing gap.", w=False)
        try:
            df = Q.load_undetected_ncd(filters, run_query)
            if not df.empty:
                _pc(bar_chart(df, x="age_group",
                              y=["elevated_visits","undetected"],
                              color_map={"elevated_visits": AFYA_BLUE,
                                         "undetected": CORAL},
                              y_label="Visits", height=280))
        except Exception as e:
            st.warning(f"A6: {e}")

    with st_b:
        _sh("B — NCD & Chronic Disease")
        try:
            df = Q.load_ncd_kpis(filters, run_query)
            if not df.empty:
                row = df.iloc[0]
                c1, c2 = st.columns(2)
                with c1: _kpi("NCD Patients", _n(row.get("ncd_patients")), color=AFYA_BLUE)
                with c2: _kpi("Controlled HTN",
                              _p(row.get("controlled_htn_pct")),
                              "avg BP <140/90",
                              TEAL if float(row.get("controlled_htn_pct") or 0) >= 60 else CORAL)
        except Exception as e:
            st.warning(f"B1: {e}")

        _gap(12)
        _sh("NCD by Age Group", mt=8)
        try:
            df = Q.load_ncd_by_age(filters, run_query)
            if not df.empty:
                top_conds = (df.groupby("chronic_condition")["patient_count"]
                             .sum().nlargest(6).index.tolist())
                b2t = df[df["chronic_condition"].isin(top_conds)]
                pivot = b2t.pivot_table(index="age_group", columns="chronic_condition",
                                        values="patient_count", aggfunc="sum", fill_value=0)
                _pc(stacked_bar(pivot.reset_index(), x="age_group",
                                categories=top_conds, y_label="Patients", height=300))
        except Exception as e:
            st.warning(f"B2: {e}")

        _gap(12)
        _sh("HTN Controlled vs Uncontrolled", mt=8)
        try:
            df = Q.load_htn_controlled(filters, run_query)
            if not df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    _pc(donut(labels=df["htn_status"].tolist(),
                              values=df["patient_count"].tolist(),
                              color_map={"Controlled": TEAL,
                                         "Uncontrolled": CORAL,
                                         "No BP Recorded": GRAY},
                              height=260))
                with c2:
                    _pc(table_fig(df,
                                  col_labels={"htn_status":"Status",
                                              "patient_count":"Patients",
                                              "avg_systolic":"Avg Systolic"},
                                  height=180))
        except Exception as e:
            st.warning(f"B5: {e}")

    with st_c:
        _sh("C — RMNCH")
        try:
            df = Q.load_anc_funnel(filters, run_query)
            if not df.empty:
                row = df.iloc[0]
                c1, c2 = st.columns(2)
                with c1:
                    _pc(funnel_chart(
                        labels=["ANC 1","ANC 2","ANC 3","ANC 4"],
                        values=[int(row.get("anc1") or 0), int(row.get("anc2") or 0),
                                int(row.get("anc3") or 0), int(row.get("anc4") or 0)],
                        height=320,
                    ))
                with c2:
                    _kpi("ANC4 Completion",
                         _p(row.get("anc4_completion_pct")),
                         "% completing 4 visits",
                         TEAL if float(row.get("anc4_completion_pct") or 0) >= 50 else CORAL)
        except Exception as e:
            st.warning(f"C2: {e}")

        _gap(12)
        _sh("Deliveries by Maternal Age Group", mt=8)
        try:
            df = Q.load_deliveries_by_age(filters, run_query)
            if not df.empty:
                _pc(bar_chart(df, x="maternal_age_group", y="delivery_count",
                              color=PURPLE, y_label="Deliveries", height=260))
        except Exception as e:
            st.warning(f"C3: {e}")

    with st_d:
        _sh("D — Communicable Disease & HIV")
        try:
            df = Q.load_communicable_trend(filters, run_query)
            if not df.empty:
                pivot = df.pivot_table(index="visit_month", columns="disease_group",
                                       values="visit_count", aggfunc="sum", fill_value=0)
                _pc(stacked_area(pivot.reset_index(), x="visit_month",
                                 categories=list(pivot.columns),
                                 y_label="Visits", height=320))
        except Exception as e:
            st.warning(f"D2: {e}")

        _gap(12)
        _sh("HIV Patient Profile", mt=8)
        try:
            df = Q.load_hiv_profile(filters, run_query)
            if not df.empty:
                row = df.iloc[0]
                c1,c2,c3,c4 = st.columns(4)
                with c1: _kpi("HIV Patients", _n(row.get("hiv_patients")))
                with c2: _kpi("Paediatric",   _n(row.get("paediatric")))
                with c3: _kpi("Female",        _n(row.get("female")))
                with c4: _kpi("Male",          _n(row.get("male")))
        except Exception as e:
            st.warning(f"D4: {e}")

    with st_e:
        _sh("E — Mental Health & Oncology")
        try:
            df = Q.load_mh_kpis(filters, run_query)
            if not df.empty:
                row = df.iloc[0]
                c1,c2,c3 = st.columns(3)
                with c1: _kpi("MH Visits",        _n(row.get("total_mh_visits")))
                with c2: _kpi("MH Patients",       _n(row.get("total_mh_patients")))
                with c3: _kpi("Inpatient Share",   _p(row.get("inpatient_share_pct")))
        except Exception as e:
            st.warning(f"E1: {e}")

        _gap(12)
        _sh("Mental Health by Age & Sex", mt=8)
        try:
            df = Q.load_mh_by_age_sex(filters, run_query)
            if not df.empty:
                pivot = df.pivot_table(index="age_group", columns="sex",
                                       values="patient_count", aggfunc="sum", fill_value=0)
                _pc(bar_chart(pivot.reset_index(), x="age_group",
                              y=list(pivot.columns),
                              color_map={"F":"#7b5ea7","FEMALE":"#7b5ea7",
                                         "M":TEAL,"MALE":TEAL},
                              y_label="Patients", height=280))
        except Exception as e:
            st.warning(f"E2: {e}")

    with st_f:
        _sh("F — Revenue by Disease Burden Group")
        try:
            df = Q.load_revenue_by_burden_group(filters, run_query)
            if not df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    _pc(hbar_chart(df, x="total_revenue", y="burden_group",
                                   color=AFYA_BLUE, x_label="Total Revenue (KES)",
                                   y_format="KES", height=300))
                with c2:
                    _pc(hbar_chart(df, x="avg_rev_per_visit", y="burden_group",
                                   color=TEAL, x_label="Avg Revenue / Visit (KES)",
                                   y_format="KES", height=300))
        except Exception as e:
            st.warning(f"F1: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CLINICAL WORKLOAD & QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def render_tab5_workload(filters: dict, run_query):
    _sh("Clinical Workload & Quality")
    _note("Scoped for Head of Clinical. "
          "Shortcut rate, BP omission, unplanned 72h returns.")

    _sh("Shortcut Rate — Single Diagnosis on Chronic Patients", mt=12)
    try:
        df = Q.load_shortcut_rate(filters, run_query)
        if not df.empty:
            _pc(hbar_chart(df, x="shortcut_rate_pct", y="clinician",
                           x_label="Shortcut Rate %", y_format="pct",
                           color=ORANGE,
                           height=max(280, len(df) * 26 + 40),
                           show_text=True))
    except Exception as e:
        st.warning(f"Shortcut rate: {e}")

    _gap(12)
    _sh("BP Omission Rate — Hypertension Visits Without BP Recorded", mt=8)
    try:
        df = Q.load_bp_omission_rate(filters, run_query)
        if not df.empty:
            _pc(hbar_chart(df, x="omission_pct", y="clinician",
                           x_label="BP Omission %", y_format="pct",
                           color=CORAL,
                           height=max(280, len(df) * 26 + 40),
                           show_text=True))
    except Exception as e:
        st.warning(f"BP omission: {e}")

    _gap(12)
    _sh("Unplanned 72h Return Rate by Clinician", mt=8)
    _note("Returns within 72h — proxy for incomplete management.")
    try:
        df = Q.load_return_72h(filters, run_query)
        if not df.empty:
            _pc(hbar_chart(df, x="return_72h_pct", y="clinician",
                           x_label="72h Return %", y_format="pct",
                           color=CORAL,
                           height=max(280, len(df) * 26 + 40),
                           show_text=True))
    except Exception as e:
        st.warning(f"72h return: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CLINICIAN VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_clinician_view(filters: dict, run_query, show_all_patients: bool = True):
    role = st.session_state.get("role", "Head of Clinician")

    st.markdown(
        f'<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
        f'text-transform:uppercase;color:#0072CE;margin-bottom:2px">'
        f'{"Facility Patient List" if show_all_patients else "Clinician View"}</p>'
        f'<p style="font-size:12px;color:#6B8CAE;margin-bottom:8px">'
        f'{"All patients across the facility · Treated by · Patient card"}'
        f'</p>',
        unsafe_allow_html=True,
    )

    _gap(8)
    _sh("Today's Patients")
    _note("'Today' = visits on the most recent date in the pipeline. "
          "HIGH patients sorted first.")

    try:
        cl1 = Q.load_todays_patients(filters, run_query)

        if cl1.empty:
            st.info("No visits found for today. Check hospital and facility filters.")
            return

        total   = len(cl1)
        high    = (cl1["risk_badge"] == "HIGH").sum()
        chronic = cl1["is_chronic"].sum()

        c1, c2, c3 = st.columns(3)
        with c1: _kpi("Today's Patients", _n(total))
        with c2: _kpi("HIGH Priority",    _n(high),    color=CORAL)
        with c3: _kpi("Chronic",          _n(chronic), color=AFYA_BLUE)

        _gap(12)

        # When showing all patients always include the treating clinician column
        display_cols = ["patient", "user", "sex", "age_group", "primary_condition",
                        "visit_type", "days_since_prev", "patient_status",
                        "risk_badge", "flag_reason"]
        col_labels = {
            "patient":        "Patient ID",
            "user":           "Treated by",
            "sex":            "Sex",
            "age_group":      "Age",
            "primary_condition": "Condition",
            "visit_type":     "Type",
            "days_since_prev":"Days Since Last",
            "patient_status": "Status",
            "risk_badge":     "Risk",
            "flag_reason":    "Flag",
        }

        _pc(table_fig(
            cl1[[c for c in display_cols if c in cl1.columns]],
            col_labels=col_labels,
            height=min(500, total * 28 + 60),
        ))

        _gap(16)
        _sh("Patient Card", mt=8)
        _note("Select a patient to load their vitals trend and medication continuity.")

        selected = st.selectbox(
            "Select patient",
            cl1["patient"].astype(str).tolist(),
            label_visibility="collapsed",
        )
        sel_schema = cl1.loc[
            cl1["patient"].astype(str) == selected, "source_schema"
        ].iloc[0] if not cl1.empty else (filters.get("schema") or "")

        if selected and sel_schema:
            _render_patient_card(selected, sel_schema, run_query)

    except Exception as e:
        st.warning(f"Patient list error: {e}")


def _render_patient_card(patient_id: str, source_schema: str, run_query):
    _gap(8)
    st.markdown(
        f'<div style="background:#0072CE;color:white;border-radius:8px;'
        f'padding:10px 14px;font-weight:700;font-size:13px;margin-bottom:10px">'
        f'Patient: {patient_id}</div>',
        unsafe_allow_html=True,
    )

    # CL2 — Vitals trend
    _sh("Vitals Trend — Last 6 Readings", mt=8)
    try:
        cl2 = Q.load_patient_vitals_trend(patient_id, source_schema, run_query)
        if not cl2.empty:
            row0   = cl2.iloc[0]
            signal = str(row0.get("clinical_signal", ""))
            sig_color = (CORAL   if "elevated" in signal.lower() or "rising" in signal.lower()
                         else TEAL if "expected" in signal.lower() else ORANGE)
            st.markdown(
                f'<div style="background:{sig_color};color:white;border-radius:6px;'
                f'padding:8px 14px;font-size:13px;font-weight:600;margin-bottom:10px">'
                f'🩺 {signal}</div>',
                unsafe_allow_html=True,
            )
            c1,c2,c3 = st.columns(3)
            for col, label, val_col, trend_col, vals_col in [
                (c1, "BP Systolic",  "recent_sys",   "systolic_trend",  "bp_systolic"),
                (c2, "BP Diastolic", "recent_dia",   "diastolic_trend", "bp_diastolic"),
                (c3, "Blood Sugar",  "recent_sugar",  "sugar_trend",     "blood_sugar"),
            ]:
                trend = str(row0.get(trend_col, ""))
                val   = row0.get(val_col)
                tc    = (TEAL if trend=="Improving" else CORAL if trend=="Worsening"
                         else AFYA_BLUE)
                arrow = "↑" if trend=="Worsening" else "↓" if trend=="Improving" else "→"
                with col:
                    st.markdown(
                        f'<div style="border:1px solid #D6E4F0;border-radius:8px;'
                        f'padding:10px 12px;background:#F4F8FC">'
                        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
                        f'text-transform:uppercase;margin-bottom:4px">{label}</div>'
                        f'<div style="font-size:20px;font-weight:800;color:{tc}">'
                        f'{f"{val:.0f}" if val else "—"}</div>'
                        f'<div style="font-size:11px;color:{tc};margin-top:2px">'
                        f'{arrow} {trend}</div></div>',
                        unsafe_allow_html=True,
                    )
                    spark_vals = cl2[vals_col].dropna().tolist()
                    if len(spark_vals) >= 2:
                        _pc(sparkline(spark_vals[::-1], trend=trend, height=60))
    except Exception as e:
        st.warning(f"CL2: {e}")

    _gap(12)

    # CL3 — Medication continuity
    _sh("Medication Continuity", mt=8)
    try:
        cl3 = Q.load_medication_continuity(patient_id, source_schema, run_query)
        if cl3.empty:
            _note("No chronic conditions with expected drug classes found.")
        else:
            gaps = int(cl3["is_gap"].sum())
            if gaps > 0:
                st.markdown(
                    f'<div style="background:#FEE2E2;border-left:4px solid {CORAL};'
                    f'border-radius:4px;padding:8px 14px;font-size:13px;'
                    f'font-weight:600;color:#991B1B;margin-bottom:10px">'
                    f'⚠ {gaps} medication gap{"s" if gaps > 1 else ""} detected</div>',
                    unsafe_allow_html=True,
                )
            _pc(table_fig(
                cl3[["condition","expected_drug_class","active_drug",
                      "days_since_prescribed","continuity_status"]],
                col_labels={"condition":"Condition","expected_drug_class":"Expected Class",
                            "active_drug":"Active Drug","days_since_prescribed":"Days Since Rx",
                            "continuity_status":"Status"},
                height=min(400, len(cl3) * 30 + 60),
            ))
    except Exception as e:
        st.warning(f"CL3: {e}")