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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import queries as Q
from ui_template import AFYA_BLUE, TEAL, COOL_BLUE, ORANGE, CORAL, PURPLE, GRAY, MUTED, BG_LIGHT, BORDER
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
def _kpi(l, v, s="", color=AFYA_BLUE):
    _H().get("kpi_card", lambda l, v, s="", c=AFYA_BLUE: None)(l, v, sub=s, color=color)
def _pc(fig):      _H().get("pc",       lambda f: st.plotly_chart(f, use_container_width=True))(fig)
def _note(t, w=False): _H().get("note", lambda t,warn=False: None)(t, w)
def _n(v):
    return _H().get("fmt_num", lambda v: str(v))(v)

def _p(v, d=1):
    return _H().get("fmt_pct", lambda v, d=1: str(v))(v, d)

def _k(v):
    return _H().get("fmt_kes", lambda v: str(v))(v)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def render_tab1_operations(filters: dict, run_query):
    # ── KPI ROW ───────────────────────────────────────────────────────────
    _sh("Key Metrics")
    try:
        df = Q.load_tab1_kpis(filters, run_query)
        df_cost = Q.load_avg_admission_cost_full(filters, run_query)
        if not df.empty:
            row  = df.iloc[0]
            crow = df_cost.iloc[0] if not df_cost.empty else {}
            ip_cost = crow.get("avg_ip_cost_full") or row.get("avg_admission_cost")
            op_cost = crow.get("avg_op_cost_full") or row.get("avg_op_cost")
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            with c1: _kpi("Total Visits",      _n(row.get("total_visits")))
            with c2: _kpi("Inpatient",         _n(row.get("inpatient_visits")),
                          str(_p(row.get("inpatient_pct")) or "—") + " of visits", TEAL)
            with c3: _kpi("Outpatient",        _n(row.get("outpatient_visits")))
            with c4: _kpi("Discharges",        _n(row.get("total_discharges")), color=TEAL)
            with c5: _kpi("Active Admissions", _n(row.get("active_admissions")), color=ORANGE)
            with c6: _kpi("Avg Admission Cost",_k(ip_cost),
                          f"Outpatient avg: {_k(op_cost)}")
    except Exception as e:
        st.warning(f"KPIs: {e}")

    _gap(16)

    # ── SECTION A: SERVICE GROWTH ─────────────────────────────────────────
    _sh("A — Service Growth by Type", mt=8)
    _note("Monthly visit volume by service type. Each panel uses its own scale.")
    try:
        df = Q.load_service_growth(filters, run_query)
        if not df.empty:
            df["outpatient"] = (
                df.get("outpatient_with_lab", pd.Series(0, index=df.index))
                + df.get("outpatient_rx_only", pd.Series(0, index=df.index))
                + df.get("consult_only", pd.Series(0, index=df.index))
            )
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["visit_month"], y=df["outpatient"],
                name="Outpatient", mode="lines+markers",
                line=dict(color=AFYA_BLUE, width=2), marker=dict(size=4),
            ))
            fig.add_trace(go.Scatter(
                x=df["visit_month"], y=df["inpatient"],
                name="Inpatient", mode="lines+markers",
                line=dict(color=TEAL, width=2), marker=dict(size=4),
            ))
            fig.update_layout(
                height=300, margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(
                    title="Visits", showgrid=True, gridcolor="#EBF3FB",
                    rangemode="tozero", fixedrange=False,
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            _pc(fig)
    except Exception as e:
        st.warning(f"Service growth: {e}")

    _gap(12)

    # ── SECTION A2: WARD BREAKDOWN ────────────────────────────────────────
    _sh("Ward Breakdown", mt=8)

    # Ward summary: % share of admissions
    try:
        df_wb = Q.load_ward_breakdown(filters, run_query)
        if not df_wb.empty:
            total_adm = df_wb["admissions"].sum()
            if total_adm > 0:
                df_wb["pct_share"] = (df_wb["admissions"] / total_adm * 100).round(1)
            else:
                df_wb["pct_share"] = 0.0
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ward Admission Share (%)**")
                _pc(hbar_chart(
                    df_wb.head(12), x="pct_share", y="ward",
                    x_label="% of Total Admissions", color=AFYA_BLUE, height=320,
                ))
            with c2:
                st.markdown("**Ward Summary**")
                _pc(table_fig(
                    df_wb[["ward", "admissions", "pct_share", "avg_los_days",
                            "avg_discharge_latency_hrs", "avg_admission_cost"]].head(12),
                    col_labels={"ward": "Ward", "admissions": "Admissions",
                                "pct_share": "Share %", "avg_los_days": "Avg LOS (d)",
                                "avg_discharge_latency_hrs": "Disch. Latency (h)",
                                "avg_admission_cost": "Avg Cost"},
                    fmt={"avg_admission_cost": "KES", "pct_share": "pct"},
                    height=320,
                ))
    except Exception as e:
        st.warning(f"Ward share: {e}")

    _gap(12)

    # Ward 1: Admission growth over time (stacked area)
    try:
        df_trend = Q.load_ward_admission_trend(filters, run_query)
        if not df_trend.empty:
            st.markdown("**Admission Growth by Ward (Top 6) — Stacked Area**")
            pivot = df_trend.pivot_table(
                index="visit_month", columns="ward",
                values="admissions", aggfunc="sum", fill_value=0,
            )
            ward_colors = [AFYA_BLUE, TEAL, ORANGE, CORAL, PURPLE, GRAY]
            color_map = {w: ward_colors[i % len(ward_colors)]
                         for i, w in enumerate(pivot.columns)}
            _pc(stacked_area(
                pivot.reset_index(), x="visit_month",
                categories=list(pivot.columns),
                color_map=color_map,
                y_label="Admissions", height=300,
            ))
    except Exception as e:
        st.warning(f"Ward trend: {e}")

    _gap(12)

    # Ward 2: Discharge latency vs patient count
    try:
        df_lat = Q.load_ward_discharge_latency(filters, run_query)
        if not df_lat.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Discharge Latency vs Patient Volume**")
                _note("Which ward takes longest to discharge after the decision is made.")
                _pc(scatter(
                    df_lat, x="patient_count", y="avg_discharge_latency_hrs",
                    label_col="ward",
                    x_label="Patients Discharged", y_label="Avg Discharge Latency (hrs)",
                    height=300,
                ))
            with c2:
                st.markdown("**Discharge Latency Ranking**")
                _pc(hbar_chart(
                    df_lat.sort_values("avg_discharge_latency_hrs", ascending=False).head(12),
                    x="avg_discharge_latency_hrs", y="ward",
                    x_label="Avg Latency (hrs)", color=ORANGE, height=300,
                ))
    except Exception as e:
        st.warning(f"Discharge latency: {e}")

    _gap(12)

    # Ward 3: Monthly admissions (bars) + Avg duration (line) — combined dual-axis
    try:
        df_active = Q.load_ward_active_vs_hours(filters, run_query)
        if not df_active.empty:
            bar_col = "total_admissions" if "total_admissions" in df_active.columns else "active_admissions"
            _note("Average admission time is calculated from completed admissions only. Ongoing admissions are excluded.")

            # Correlation annotation
            valid = df_active[[bar_col, "avg_admission_hours"]].dropna()
            if len(valid) >= 3:
                corr = valid[bar_col].corr(valid["avg_admission_hours"])
                if corr > 0.3:
                    corr_text = f"Higher-admission months correlate with longer average admission times (r = {corr:.2f}) — possible capacity pressure."
                elif corr < -0.3:
                    corr_text = f"Higher-admission months correlate with shorter average admission times (r = {corr:.2f}) — possible early discharge pressure."
                else:
                    corr_text = f"Admission volume shows no consistent relationship with average admission time (r = {corr:.2f})."
            else:
                corr_text = ""

            st.markdown("**Monthly Admissions vs Average Admission Time**")
            if corr_text:
                _note(corr_text)

            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(
                x=df_active["visit_month"], y=df_active[bar_col],
                name="Admissions", marker_color=AFYA_BLUE,
                yaxis="y1", opacity=0.75,
            ))
            fig_a.add_trace(go.Scatter(
                x=df_active["visit_month"], y=df_active["avg_admission_hours"],
                name="Avg Admission Time (hrs)", mode="lines+markers",
                line=dict(color=ORANGE, width=2), marker=dict(size=5),
                yaxis="y2",
            ))
            fig_a.update_layout(
                height=300, margin=dict(l=0, r=60, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Admissions", color=AFYA_BLUE,
                           showgrid=True, gridcolor="#EBF3FB"),
                yaxis2=dict(title="Avg Admission Time (hrs)", color=ORANGE,
                            overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            _pc(fig_a)
    except Exception as e:
        st.warning(f"Active admissions: {e}")

    _gap(12)

    # Ward 4: Cost per ward vs patient volumes — bubble chart with quadrants
    try:
        df_cost = Q.load_ward_cost_volume(filters, run_query)
        if not df_cost.empty:
            st.markdown("**Cost per Admission vs Volume by Ward**")
            _note(
                "Bubble size = total revenue. Cost = invoice line items per admitted visit. "
                "Quadrant lines = median admissions and median avg cost.",
                w=True,
            )
            med_x = df_cost["admissions"].median()
            med_y = df_cost["avg_admission_cost"].median()

            # Normalise bubble size: 10–50px range
            rev = df_cost["total_revenue"].fillna(0)
            rev_range = max(rev.max() - rev.min(), 1)
            bubble_sizes = ((rev - rev.min()) / rev_range * 40 + 10).tolist()

            fig_cost = go.Figure()

            # Quadrant reference lines
            fig_cost.add_shape(type="line", x0=med_x, x1=med_x,
                               y0=0, y1=1, yref="paper",
                               line=dict(color=GRAY, width=1, dash="dot"))
            fig_cost.add_shape(type="line", x0=0, x1=1, xref="paper",
                               y0=med_y, y1=med_y,
                               line=dict(color=GRAY, width=1, dash="dot"))

            # Quadrant labels
            quad_style = dict(xref="paper", yref="paper", showarrow=False,
                              font=dict(size=9, color=MUTED), bgcolor="rgba(255,255,255,0.7)")
            fig_cost.add_annotation(x=0.02, y=0.98, xanchor="left", yanchor="top",
                                    text="<b>Investigate</b><br><i>Inefficiency or niche premium?</i>",
                                    **quad_style)
            fig_cost.add_annotation(x=0.98, y=0.98, xanchor="right", yanchor="top",
                                    text="<b>Revenue engine</b><br><i>Protect and optimise</i>",
                                    **quad_style)
            fig_cost.add_annotation(x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                                    text="<b>Underutilised</b><br><i>Capacity opportunity</i>",
                                    **quad_style)
            fig_cost.add_annotation(x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                                    text="<b>Operational workhorse</b><br><i>Efficiency model</i>",
                                    **quad_style)

            fig_cost.add_trace(go.Scatter(
                x=df_cost["admissions"],
                y=df_cost["avg_admission_cost"],
                mode="markers+text",
                text=df_cost["ward"],
                textposition="top center",
                textfont=dict(size=9, family="Montserrat", color=COOL_BLUE),
                marker=dict(
                    size=bubble_sizes,
                    color=AFYA_BLUE,
                    opacity=0.75,
                    line=dict(color="white", width=0.5),
                ),
                customdata=df_cost[["total_revenue", "revenue_per_patient"]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Admissions: %{x:,}<br>"
                    "Avg Cost: KES %{y:,.0f}<br>"
                    "Total Revenue: KES %{customdata[0]:,.0f}<br>"
                    "Revenue/Patient: KES %{customdata[1]:,.0f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))

            fig_cost.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Admissions", showgrid=True,
                           gridcolor="#EBF3FB", zeroline=False),
                yaxis=dict(title="Avg Cost per Admission (KES)", showgrid=True,
                           gridcolor="#EBF3FB", zeroline=False,
                           tickformat=",.0f"),
            )
            _pc(fig_cost)
    except Exception as e:
        st.warning(f"Ward cost/volume: {e}")

    _gap(12)

    # Ward 5: Top-3 ward operational summary table
    try:
        df_ws = Q.load_top_ward_summary(filters, run_query)
        if not df_ws.empty:
            st.markdown("**What Are the Busiest Wards Treating?**")
            _note(
                "Top 3 wards by admissions. Pressure = recent 3-month avg vs prior 3-month avg. "
                "Seasonality: CV > 0.3 = seasonal pattern, < 0.15 = consistent."
            )

            for _c in ("admissions", "pressure_pct", "cv_seasonality", "top_condition_pct",
                       "new_pct", "returning_pct", "top_payer_pct",
                       "avg_los_days", "investigation_rate_pct",
                       "patients_per_clinician", "clinicians"):
                if _c in df_ws.columns:
                    df_ws[_c] = pd.to_numeric(df_ws[_c], errors="coerce")

            rows = []
            for _, r in df_ws.iterrows():
                pressure_pct = r.get("pressure_pct", 0) or 0
                pressure_sig = r.get("pressure_signal", "Stable")
                pressure_arrow = {"Rising": "↑", "Easing": "↓", "Stable": "→"}.get(
                    pressure_sig, "→"
                )
                cv = float(r.get("cv_seasonality") or 0)
                seasonality = "Seasonal" if cv > 0.30 else ("Consistent" if cv < 0.15 else "Mixed")

                top_cond     = r.get("top_condition") or "—"
                cond_pct     = r.get("top_condition_pct") or 0
                cond_pattern = r.get("condition_pattern") or ""
                cond_str     = f"{top_cond} ({int(cond_pct)}%)"
                if cond_pattern == "Varies":
                    cond_str += " + others"

                payer      = r.get("top_payer") or "—"
                payer_pct  = r.get("top_payer_pct") or 0
                payer_str  = f"{payer} ({int(payer_pct)}%)"

                los = r.get("avg_los_days")
                los_str = f"{float(los):.1f} days" if pd.notna(los) and los else "—"

                inv_rate = r.get("investigation_rate_pct")
                inv_str  = f"{int(inv_rate)}%" if pd.notna(inv_rate) and inv_rate else "—"

                ratio = r.get("patients_per_clinician")
                clin  = r.get("clinicians")
                ratio_str = (f"{float(ratio):.0f}:1  ({int(clin)} clinicians)"
                             if pd.notna(ratio) and ratio else "—")

                rows.append({
                    "Ward":              r.get("ward", "—"),
                    "Admissions":        f"{int(r.get('admissions', 0)):,}",
                    "Pressure":          f"{pressure_sig} {pressure_arrow} ({int(pressure_pct):+d}%)",
                    "Seasonality":       seasonality,
                    "Top Condition":     cond_str,
                    "New / Returning":   f"{int(r.get('new_pct', 0))}% new · {int(r.get('returning_pct', 0))}% returning",
                    "Top Payer":         payer_str,
                    "Avg Admission":     los_str,
                    "Investigation Rate": inv_str,
                    "Pts / Clinician":   ratio_str,
                })

            _pc(table_fig(
                pd.DataFrame(rows),
                col_labels={},
                height=max(200, len(rows) * 52 + 60),
            ))
    except Exception as e:
        st.warning(f"Ward summary: {e}")

    _gap(12)

    # ── DIAGNOSIS COST OUTLIERS ───────────────────────────────────────────
    _sh("Which Diagnoses Are Driving Disproportionate Cost?", mt=8)
    _note(
        "A ratio above 1.0 means this diagnosis consumes more cost share than its "
        "patient volume justifies. Use as a flag — deeper drug and investigation "
        "breakdowns belong in inpatient analytics."
    )
    try:
        df_dc = Q.load_diagnosis_cost_outliers(filters, run_query)
        if not df_dc.empty:
            for _c in ("visit_count", "avg_cost_per_case", "total_cost",
                       "volume_share_pct", "cost_share_pct", "cost_volume_ratio"):
                df_dc[_c] = pd.to_numeric(df_dc[_c], errors="coerce")

            overall_avg_cost = float(
                (df_dc["total_cost"].sum() / df_dc["visit_count"].sum())
                if df_dc["visit_count"].sum() > 0 else 0
            )

            def _ratio_color(r):
                if r >= 2.0:   return CORAL
                if r >= 1.5:   return ORANGE
                return TEAL

            colors = [_ratio_color(r) for r in df_dc["cost_volume_ratio"]]
            sizes  = df_dc["total_cost"]
            s_min, s_max = float(sizes.min()), float(sizes.max())
            s_range = s_max - s_min or 1
            bubble_sizes = [12 + (float(v) - s_min) / s_range * 28
                            for v in sizes]

            c1, c2 = st.columns([3, 2])
            with c1:
                fig_dc = go.Figure()
                fig_dc.add_trace(go.Scatter(
                    x=df_dc["visit_count"],
                    y=df_dc["avg_cost_per_case"],
                    mode="markers+text",
                    marker=dict(
                        color=colors,
                        size=bubble_sizes,
                        opacity=0.82,
                        line=dict(color="white", width=1),
                    ),
                    text=df_dc["diagnosis_group"],
                    textposition="top center",
                    textfont=dict(size=8, color=MUTED),
                    customdata=df_dc[["cost_volume_ratio", "cost_share_pct",
                                      "volume_share_pct", "total_cost"]].values,
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Visits: %{x:,}<br>"
                        "Avg cost/case: %{y:,.0f}<br>"
                        "Cost share: %{customdata[1]:.1f}%<br>"
                        "Volume share: %{customdata[2]:.1f}%<br>"
                        "Ratio: %{customdata[0]:.2f}x<extra></extra>"
                    ),
                ))
                # Overall avg cost reference line
                fig_dc.add_shape(
                    type="line",
                    xref="paper", yref="y",
                    x0=0, x1=1,
                    y0=overall_avg_cost, y1=overall_avg_cost,
                    line=dict(color=GRAY, width=1.5, dash="dash"),
                )
                fig_dc.add_annotation(
                    xref="paper", yref="y",
                    x=1.01, y=overall_avg_cost,
                    text="Avg cost",
                    xanchor="left", yanchor="middle",
                    showarrow=False,
                    font=dict(size=8, color=GRAY),
                )
                fig_dc.update_layout(
                    height=380,
                    margin=dict(l=0, r=60, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Visit Count", showgrid=False),
                    yaxis=dict(
                        title="Avg Cost per Case",
                        showgrid=True, gridcolor="#EBF3FB",
                    ),
                )
                _pc(fig_dc)
                _note(
                    "Bubble size = total cost. "
                    "Red = ratio ≥ 2×  ·  Orange = 1.5–2×  ·  Teal = below 1.5×. "
                    "Diagnoses above the dashed line cost more per case than the average."
                )

            with c2:
                tbl = df_dc[["diagnosis_group", "visit_count",
                              "avg_cost_per_case", "cost_volume_ratio"]].copy()
                tbl["cost_volume_ratio"] = tbl["cost_volume_ratio"].apply(
                    lambda v: f"{v:.2f}×" if pd.notna(v) else "—"
                )
                tbl["avg_cost_per_case"] = tbl["avg_cost_per_case"].apply(
                    lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                )
                tbl["visit_count"] = tbl["visit_count"].apply(
                    lambda v: f"{int(v):,}" if pd.notna(v) else "—"
                )
                _pc(table_fig(
                    tbl.head(12).rename(columns={
                        "diagnosis_group":   "Diagnosis",
                        "visit_count":       "Visits",
                        "avg_cost_per_case": "Avg Cost",
                        "cost_volume_ratio": "Ratio",
                    }),
                    col_labels={},
                    height=380,
                ))
    except Exception as e:
        st.warning(f"Diagnosis cost outliers: {e}")

    _gap(12)

    # ── SECTION B: SPIKE & DIP DETECTION ─────────────────────────────────
    _sh("B — When did volume behave unusually?", mt=8)

    try:
        df_vol = Q.load_monthly_volume_anomalies(filters, run_query)
        if not df_vol.empty:
            df_vol = df_vol.copy()
            df_vol["month_label"] = pd.to_datetime(
                df_vol["visit_month"], errors="coerce"
            ).dt.strftime("%Y-%m")

            # Drop the first partial month from chart — it skews visuals
            df_chart = df_vol[df_vol["month_type"] != "Partial"].copy()

            avg_vol = float(df_vol["avg_vol"].iloc[0])
            sd_vol  = float(df_vol["sd_vol"].iloc[0])

            spikes = df_chart[df_chart["month_type"] == "Spike"]
            dips   = df_chart[df_chart["month_type"] == "Dip"]

            # Fallback: no months cross 1.0 SD → use top-2 / bottom-2
            use_fallback = spikes.empty and dips.empty
            if use_fallback:
                spikes = df_chart.nlargest(2, "total_visits").copy()
                spikes["month_type"] = "Highest"
                dips = df_chart.nsmallest(2, "total_visits").copy()
                dips["month_type"] = "Lowest"
                _note(
                    "No months exceeded ±1 SD from the mean. "
                    "Showing the 2 highest and 2 lowest volume months instead."
                )

            # ── Time-series bar chart ──────────────────────────────────────
            type_color = {"Spike": ORANGE, "Dip": CORAL,
                          "Highest": ORANGE, "Lowest": CORAL, "Normal": AFYA_BLUE}
            bar_colors = [type_color.get(t, AFYA_BLUE) for t in df_chart["month_type"]]

            fig_vol = go.Figure()

            # ±1 SD shaded band (normal range)
            fig_vol.add_shape(
                type="rect", xref="paper", yref="y",
                x0=0, x1=1,
                y0=avg_vol - sd_vol, y1=avg_vol + sd_vol,
                fillcolor=GRAY, opacity=0.10, line_width=0,
            )

            # Spike threshold — upper edge of normal range
            fig_vol.add_hline(
                y=avg_vol + sd_vol,
                line=dict(color=ORANGE, width=1.5, dash="dash"),
                annotation_text=f"Spike threshold  ({avg_vol + sd_vol:,.0f})",
                annotation_position="right",
                annotation_font=dict(size=9, color=ORANGE),
            )

            # Dip threshold — lower edge of normal range
            fig_vol.add_hline(
                y=avg_vol - sd_vol,
                line=dict(color=CORAL, width=1.5, dash="dash"),
                annotation_text=f"Dip threshold  ({avg_vol - sd_vol:,.0f})",
                annotation_position="right",
                annotation_font=dict(size=9, color=CORAL),
            )

            fig_vol.add_trace(go.Bar(
                x=df_chart["month_label"],
                y=df_chart["total_visits"],
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{x}</b><br>Visits: %{y:,}<extra></extra>"
                ),
                showlegend=False,
            ))

            # Legend patches (manual)
            for label, color in [("Spike / Highest", ORANGE),
                                  ("Dip / Lowest", CORAL),
                                  ("Normal", AFYA_BLUE)]:
                fig_vol.add_trace(go.Bar(
                    x=[None], y=[None], name=label,
                    marker_color=color, showlegend=True,
                ))

            fig_vol.update_layout(
                height=300,
                margin=dict(l=0, r=60, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False),
                yaxis=dict(title="Total Visits", showgrid=True,
                           gridcolor="#EBF3FB"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                barmode="overlay",
            )
            _pc(fig_vol)

            # ── Summary tables with diagnosis context ─────────────────────
            spike_label = "Highest Volume Months" if use_fallback else "Spike Months"
            dip_label   = "Lowest Volume Months"  if use_fallback else "Dip Months"

            def _fmt_mom(row):
                if row.get("first_in_range") == 1:
                    return "prior period outside range"
                v = row.get("mom_pct")
                return f"{v:+.1f}%" if v is not None else "—"

            # Diagnosis context: top diagnosis + new/returning per anomalous month
            try:
                df_dx = Q.load_diagnosis_by_month(filters, run_query)
            except Exception:
                df_dx = pd.DataFrame()

            normal_months = df_chart[df_chart["month_type"] == "Normal"]["visit_month"].tolist()
            if not df_dx.empty and normal_months:
                baseline_new = (df_dx[df_dx["visit_month"].isin(normal_months)]
                                .groupby("visit_month")["new_patients"].sum().mean())
                baseline_ret = (df_dx[df_dx["visit_month"].isin(normal_months)]
                                .groupby("visit_month")["returning_patients"].sum().mean())
            else:
                baseline_new = baseline_ret = None

            def _enrich(df_in):
                rows = []
                for _, r in df_in.iterrows():
                    m = r["visit_month"]
                    mom = _fmt_mom(r)
                    if not df_dx.empty:
                        mdf = df_dx[df_dx["visit_month"] == m]
                        top_dx = (mdf.groupby("diagnosis_group")["visit_count"]
                                  .sum().idxmax() if not mdf.empty else "—")
                        new_pt = int(mdf["new_patients"].sum())
                        ret_pt = int(mdf["returning_patients"].sum())
                    else:
                        top_dx, new_pt, ret_pt = "—", None, None
                    rows.append({
                        "month_label":   r["month_label"],
                        "total_visits":  int(r["total_visits"]),
                        "z_score":       r["z_score"],
                        "mom_display":   mom,
                        "top_diagnosis": top_dx,
                        "new_patients":  new_pt,
                        "returning_pts": ret_pt,
                    })
                return pd.DataFrame(rows)

            col_labels = {
                "month_label":   "Month",
                "total_visits":  "Visits",
                "z_score":       "Z-Score",
                "mom_display":   "MoM Change",
                "top_diagnosis": "Top Diagnosis",
                "new_patients":  "New Patients",
                "returning_pts": "Returning Patients",
            }

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{spike_label}**")
                tdf = _enrich(spikes)
                _pc(table_fig(tdf, col_labels=col_labels,
                              height=max(120, len(tdf) * 36 + 60)))
            with c2:
                st.markdown(f"**{dip_label}**")
                tdf = _enrich(dips)
                _pc(table_fig(tdf, col_labels=col_labels,
                              height=max(120, len(tdf) * 36 + 60)))

            if baseline_new is not None:
                _note(
                    f"Baseline avg (normal months): "
                    f"{baseline_new:,.0f} new patients / month, "
                    f"{baseline_ret:,.0f} returning patients / month."
                )

    except Exception as e:
        st.warning(f"Volume anomalies: {e}")

    _gap(12)

    # ── SECTION C: JOURNEY TIMES ──────────────────────────────────────────
    _sh("C — Where are patients waiting longest?", mt=8)
    _note("Averages hide the patients who wait longest. This chart shows the full range.")
    try:
        df = Q.load_journey_times(filters, run_query)
        if not df.empty:
            # Stage definitions: (label, p50_col, p75_col, p90_col, pct_col, target_h, target_label)
            STAGES = [
                ("Arrival → Triage",       "p50_hrs_to_triage",         "p75_hrs_to_triage",         "p90_hrs_to_triage",         "pct_exceed_triage",  0.25, "Target: 15 min"),
                ("Triage → Consult",       "p50_hrs_triage_to_consult", "p75_hrs_triage_to_consult", "p90_hrs_triage_to_consult", "pct_exceed_consult", 1.0,  "Target: 1 h"),
                ("Consult → Lab Result",   "p50_hrs_consult_to_lab",    "p75_hrs_consult_to_lab",    "p90_hrs_consult_to_lab",    None,                 None, None),
                ("Lab Result → Discharge", "p50_hrs_lab_turnaround",    "p75_hrs_lab_turnaround",    "p90_hrs_lab_turnaround",    "pct_exceed_lab",     4.0,  "Target: 4 h"),
            ]

            vtypes = [v for v in ["Inpatient", "Outpatient"] if v in df["visit_type"].values]
            fig_jt = make_subplots(
                rows=1, cols=len(vtypes),
                subplot_titles=[
                    f"{v}  ({_n(df[df['visit_type']==v].iloc[0].get('total_visits'))} visits)"
                    for v in vtypes
                ],
                horizontal_spacing=0.08,
            )

            for col_idx, vtype in enumerate(vtypes, start=1):
                row = df[df["visit_type"] == vtype].iloc[0]
                stage_labels, p50s, p75_inc, p90_inc, grey_labels = [], [], [], [], []

                for label, c50, c75, c90, *_ in STAGES:
                    p50_val = row.get(c50)
                    p75_val = row.get(c75)
                    p90_val = row.get(c90)
                    stage_labels.append(label)

                    if pd.isna(p50_val) or p50_val is None:
                        p50s.append(None)
                        p75_inc.append(None)
                        p90_inc.append(None)
                        grey_labels.append(label)
                    else:
                        p75_val = p75_val or p50_val
                        p90_val = p90_val or p75_val
                        p50s.append(round(float(p50_val), 2))
                        p75_inc.append(round(max(float(p75_val) - float(p50_val), 0), 2))
                        p90_inc.append(round(max(float(p90_val) - float(p75_val), 0), 2))

                # P50 segment — teal
                fig_jt.add_trace(go.Bar(
                    y=stage_labels, x=p50s, orientation="h",
                    name="Median (P50)", marker_color=TEAL,
                    showlegend=(col_idx == 1),
                    hovertemplate="P50: %{x:.2f} h<extra></extra>",
                ), row=1, col=col_idx)

                # P50→P75 increment — amber
                fig_jt.add_trace(go.Bar(
                    y=stage_labels, x=p75_inc, orientation="h",
                    name="75th pct", marker_color=ORANGE,
                    showlegend=(col_idx == 1),
                    hovertemplate="P75 adds: %{x:.2f} h<extra></extra>",
                ), row=1, col=col_idx)

                # P75→P90 increment — red
                fig_jt.add_trace(go.Bar(
                    y=stage_labels, x=p90_inc, orientation="h",
                    name="90th pct", marker_color=CORAL,
                    showlegend=(col_idx == 1),
                    hovertemplate="P90 adds: %{x:.2f} h<extra></extra>",
                ), row=1, col=col_idx)

                # Grey placeholder bars for stages with no data
                grey_x = [0.5 if lbl in grey_labels else None for lbl in stage_labels]
                fig_jt.add_trace(go.Bar(
                    y=stage_labels, x=grey_x, orientation="h",
                    name="Not captured", marker_color=GRAY,
                    showlegend=(col_idx == 1),
                    hovertemplate="Lab TAT not currently captured<extra></extra>",
                ), row=1, col=col_idx)

                # Target reference lines and % exceeding annotations
                for label, c50, c75, c90, pct_col, target_h, target_label in STAGES:
                    if target_h is None:
                        continue
                    axis_key = "x" if col_idx == 1 else f"x{col_idx}"
                    fig_jt.add_shape(
                        type="line",
                        xref=axis_key, yref="paper",
                        x0=target_h, x1=target_h, y0=0, y1=1,
                        line=dict(color=PURPLE, width=1.5, dash="dash"),
                    )
                    # % exceeding annotation beside the bar
                    if pct_col and row.get(pct_col) is not None:
                        pct_val = row.get(pct_col)
                        fig_jt.add_annotation(
                            xref=axis_key, yref="y" if col_idx == 1 else f"y{col_idx}",
                            x=float(row.get(c90) or 0) * 1.02,
                            y=label,
                            text=f"  {pct_val}% > {target_label.split(': ')[1]}",
                            xanchor="left", yanchor="middle",
                            showarrow=False,
                            font=dict(size=8, color=MUTED),
                        )

            fig_jt.update_layout(
                barmode="stack",
                height=320,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Hours", showgrid=True, gridcolor="#EBF3FB"),
                legend=dict(orientation="h", yanchor="bottom", y=1.08,
                            xanchor="left", x=0),
            )
            if len(vtypes) > 1:
                fig_jt.update_layout(xaxis2=dict(title="Hours", showgrid=True,
                                                  gridcolor="#EBF3FB"))
            _pc(fig_jt)

            # Note for the grey stage
            _note(
                "Consult → Lab Result shown in grey: lab result timestamps are not currently "
                "captured end-to-end in the EMR. Recommend adding result receipt time at this stage."
            )
    except Exception as e:
        st.warning(f"Journey times: {e}")

    _gap(12)

    # Lab / investigation turnaround by clinical discipline
    _sh("Lab & Investigation Turnaround by Clinical Discipline", mt=8)
    try:
        # Discover available columns to find a procedure-name field
        inv_cols = Q.load_investigation_columns(run_query)
        _name_candidates = ["name", "investigation_name", "description",
                            "investigation_description", "test_name",
                            "procedure_name", "category"]
        name_col = next((c for c in _name_candidates if c in inv_cols), "")

        if name_col:
            _note(
                f"Grouped by clinical discipline derived from `{name_col}` — "
                "keyword-matched to Haematology, Clinical Chemistry, Microbiology, "
                "Immunology/Serology, Urinalysis, Pathology & Cytology, Radiology & Imaging."
            )
        else:
            _note(
                "No procedure-name column found — disciplines inferred from investigation_type "
                "(Lab → Clinical Chemistry/Haematology/Microbiology etc., Radiology, Ultrasound). "
                "Results show broad categories only."
            )

        df_lab = Q.load_lab_turnaround_by_discipline(filters, run_query, name_col=name_col)

        # Fallback: discipline query empty → use simpler by-type query
        _disc_mode = True
        if df_lab.empty:
            df_lab = Q.load_lab_turnaround_by_test(filters, run_query)
            _disc_mode = False
            if not df_lab.empty:
                df_lab = df_lab.rename(columns={"test_type": "discipline"})
                _note("Discipline grouping returned no data — showing results by investigation type.")

        if not df_lab.empty:
            # Coerce Snowflake Decimal columns to float
            for _c in ("test_count", "avg_turnaround_hrs", "median_turnaround_hrs", "result_rate_pct"):
                if _c in df_lab.columns:
                    df_lab[_c] = pd.to_numeric(df_lab[_c], errors="coerce")

            # ── per-discipline summary (weighted collapse to discipline level) ──
            rows = []
            for disc, g in df_lab.groupby("discipline"):
                total = float(g["test_count"].sum())
                avg_tat = g["avg_turnaround_hrs"].dropna()
                rows.append({
                    "discipline":            disc,
                    "test_count":            int(total),
                    "avg_turnaround_hrs":    round(
                        float((g["avg_turnaround_hrs"].fillna(0) * g["test_count"]).sum()) / total, 2
                    ) if total and not avg_tat.empty else None,
                    "median_turnaround_hrs": round(float(g["median_turnaround_hrs"].median()), 2)
                                             if g["median_turnaround_hrs"].notna().any() else None,
                    "result_rate_pct":       round(
                        float((g["result_rate_pct"].fillna(0) * g["test_count"]).sum()) / total, 1
                    ) if total else None,
                })
            disc_summary = pd.DataFrame(rows).reset_index(drop=True)
            has_tat = disc_summary["avg_turnaround_hrs"].notna().any()
            sort_col = "avg_turnaround_hrs" if has_tat else "test_count"
            disc_summary = disc_summary.sort_values(sort_col, ascending=False).reset_index(drop=True)

            # Colour map for disciplines
            _disc_colors = {
                "Haematology":            AFYA_BLUE,
                "Clinical Chemistry":     TEAL,
                "Microbiology":           ORANGE,
                "Immunology / Serology":  PURPLE,
                "Urinalysis":             COOL_BLUE,
                "Pathology & Cytology":   CORAL,
                "Radiology & Imaging":    GRAY,
                "Other / Unclassified":   "#AAAAAA",
            }
            bar_colors = [_disc_colors.get(d, AFYA_BLUE) for d in disc_summary["discipline"]]

            c1, c2 = st.columns(2)
            with c1:
                if has_tat:
                    x_vals = disc_summary["avg_turnaround_hrs"]
                    x_title = "Avg Turnaround (hrs)"
                    txt = disc_summary["avg_turnaround_hrs"].apply(
                        lambda v: f"{v:.1f}h" if pd.notna(v) else ""
                    )
                else:
                    x_vals = disc_summary["test_count"]
                    x_title = "Tests ordered"
                    txt = disc_summary["test_count"].apply(lambda v: f"{int(v):,}")
                    _note("Result timestamps not recorded — showing test volume by discipline.")

                fig_disc = go.Figure(go.Bar(
                    y=disc_summary["discipline"],
                    x=x_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=txt,
                    textposition="outside",
                ))
                fig_disc.update_layout(
                    height=320,
                    xaxis_title=x_title,
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=0, r=40, t=20, b=20),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                _pc(fig_disc)
            with c2:
                tbl_cols = {
                    "discipline":            "Discipline",
                    "test_count":            "Tests",
                    "avg_turnaround_hrs":    "Avg Hrs",
                    "median_turnaround_hrs": "Median Hrs",
                    "result_rate_pct":       "Result %",
                }
                _pc(table_fig(
                    disc_summary,
                    col_labels=tbl_cols,
                    fmt={"result_rate_pct": "pct"},
                    height=320,
                ))

            if _disc_mode:
                # ── drill-down: per-discipline x investigation_type breakdown ─
                _gap(8)
                selected_disc = st.selectbox(
                    "Drill into discipline",
                    disc_summary["discipline"].tolist(),
                    key="lab_disc_select",
                )
                df_drill = (
                    df_lab[df_lab["discipline"] == selected_disc]
                    .sort_values("avg_turnaround_hrs" if has_tat else "test_count",
                                 ascending=False)
                    .reset_index(drop=True)
                )
                if not df_drill.empty:
                    d1, d2 = st.columns(2)
                    with d1:
                        _pc(hbar_chart(
                            df_drill,
                            x="avg_turnaround_hrs" if has_tat else "test_count",
                            y="investigation_type",
                            x_label="Avg Turnaround (hrs)" if has_tat else "Tests ordered",
                            color=_disc_colors.get(selected_disc, AFYA_BLUE),
                            height=max(220, len(df_drill) * 30 + 60),
                            show_text=True,
                        ))
                    with d2:
                        _pc(table_fig(
                            df_drill,
                            col_labels={
                                "investigation_type":    "Test Type",
                                "test_count":            "Count",
                                "avg_turnaround_hrs":    "Avg Hrs",
                                "median_turnaround_hrs": "Median Hrs",
                                "result_rate_pct":       "Result %",
                            },
                            fmt={"result_rate_pct": "pct"},
                            height=max(220, len(df_drill) * 30 + 60),
                        ))
        else:
            _note("No investigation data found for the selected period.")
    except Exception as e:
        st.warning(f"Lab turnaround: {e}")

    _gap(12)

    # ── SECTION D: PEAK DEMAND ────────────────────────────────────────────
    _sh("D — Peak Demand: When, Who & Why", mt=8)

    # D1: Hour × Day heatmap with actual values
    try:
        df_hm = Q.load_peak_demand_heatmap(filters, run_query)
        if not df_hm.empty:
            day_order = ["Monday","Tuesday","Wednesday","Thursday",
                         "Friday","Saturday","Sunday"]
            hmap_sel = st.radio(
                "Visit type", ["Total", "Outpatient", "Inpatient"],
                horizontal=True, key="heatmap_type",
            )
            z_col = {"Total": "visit_count",
                     "Outpatient": "outpatient_count",
                     "Inpatient": "inpatient_count"}[hmap_sel]

            # Build heatmap with annotated values
            pivot = df_hm.pivot_table(
                index="day_name", columns="hour_of_day",
                values=z_col, aggfunc="sum", fill_value=0,
            )
            pivot = pivot.reindex([d for d in day_order if d in pivot.index])

            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[f"{h:02d}:00" for h in pivot.columns],
                y=list(pivot.index),
                colorscale="Blues",
                text=pivot.values,
                texttemplate="%{text:,}",
                textfont=dict(size=8),
                hovertemplate="<b>%{y} %{x}</b><br>Visits: %{z:,}<extra></extra>",
                colorbar=dict(thickness=12),
            ))
            fig_hm.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Hour of Day (EAT)"),
                yaxis=dict(title="Day", autorange="reversed"),
            )
            _pc(fig_hm)

            peak = df_hm.loc[df_hm["visit_count"].idxmax()]
            _note(
                f"Busiest slot: {peak['day_name']} at {int(peak['hour_of_day']):02d}:00 "
                f"({int(peak['visit_count']):,} total · "
                f"{int(peak['outpatient_count']):,} OP · "
                f"{int(peak['inpatient_count']):,} IP)."
            )
    except Exception as e:
        st.warning(f"Heatmap: {e}")

    _gap(12)

    # D2: Monthly peaks — which months and what contributed
    try:
        df_mo = Q.load_peak_demand_monthly(filters, run_query)
        if not df_mo.empty:
            st.markdown("**Monthly Volume: Peak Months & Composition**")
            c1, c2 = st.columns(2)
            with c1:
                fig_mb = go.Figure()
                colors_op = [ORANGE if p else AFYA_BLUE for p in df_mo["is_peak_month"]]
                fig_mb.add_trace(go.Bar(
                    x=df_mo["visit_month"], y=df_mo["outpatient_visits"],
                    name="Outpatient", marker_color=colors_op, opacity=0.9,
                ))
                fig_mb.add_trace(go.Bar(
                    x=df_mo["visit_month"], y=df_mo["inpatient_visits"],
                    name="Inpatient", marker_color=TEAL,
                ))
                fig_mb.update_layout(
                    barmode="stack", height=280,
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Month"),
                    yaxis=dict(title="Visits", showgrid=True, gridcolor="#EBF3FB"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1),
                )
                _note("Orange bars = peak months (>1.0 SD above mean).")
                _pc(fig_mb)
            with c2:
                st.markdown("**What Brought Patients in Anomalous Months?**")
                peak_rows = df_mo[df_mo["is_peak_month"] == 1].copy()
                fallback = peak_rows.empty
                if fallback:
                    peak_rows = df_mo.nlargest(2, "total_visits").copy()

                # Enrich with top diagnosis group for each highlighted month
                try:
                    df_dx = Q.load_diagnosis_by_month(filters, run_query)
                    if not df_dx.empty:
                        top_dx = (
                            df_dx.sort_values("visit_count", ascending=False)
                            .drop_duplicates("visit_month")[["visit_month", "diagnosis_group"]]
                            .rename(columns={"diagnosis_group": "Top Diagnosis"})
                        )
                        peak_rows = peak_rows.merge(top_dx, on="visit_month", how="left")
                except Exception:
                    pass

                tbl_cols = ["visit_month", "total_visits", "z_score",
                            "communicable_pct", "ncd_pct"]
                tbl_rename = {
                    "visit_month": "Month", "total_visits": "Visits",
                    "z_score": "Z-Score",
                    "communicable_pct": "Communicable %", "ncd_pct": "NCD %",
                }
                if "Top Diagnosis" in peak_rows.columns:
                    tbl_cols.append("Top Diagnosis")

                _pc(table_fig(
                    peak_rows[tbl_cols].rename(columns=tbl_rename),
                    col_labels={},
                    fmt={"Communicable %": "pct", "NCD %": "pct"},
                    height=260,
                ))

                if fallback:
                    _note("No statistical spike in selected period — showing top 2 months by volume.")
                else:
                    top = peak_rows.sort_values("communicable_pct", ascending=False).iloc[0]
                    if pd.to_numeric(top["communicable_pct"], errors="coerce") > 30:
                        _note(
                            f"Peak in {str(top['visit_month'])[:7]} was "
                            f"{top['communicable_pct']:.0f}% communicable disease — "
                            "likely outbreak driven.",
                            w=True,
                        )
    except Exception as e:
        st.warning(f"Monthly peaks: {e}")

    _gap(12)

    # ── SECTION E: PATIENT FLOW UNDER PRESSURE ───────────────────────────
    _sh("E — Patient Flow Under Pressure", mt=8)
    _note(
        "How does operational performance change between peak and quiet days? "
        "Do night-shift patients convert to admissions differently, and does volume stress "
        "affect how quickly patients are seen, investigated, and discharged?"
    )

    # E1: Night-shift A&E → morning admission conversion
    _gap(8)
    st.markdown("**E1 — Night-Shift A&E: How Many Convert to Morning Admissions?**")
    try:
        df_night = Q.load_night_ae_conversion(filters, run_query)
        if not df_night.empty:
            for _c in ("total_visits", "admitted", "conversion_rate_pct",
                       "morning_admit_pct", "avg_wait_to_admit_hrs",
                       "insurance_pct", "surgery_pct"):
                if _c in df_night.columns:
                    df_night[_c] = pd.to_numeric(df_night[_c], errors="coerce")

            cols = st.columns(len(df_night))
            for col, (_, row) in zip(cols, df_night.iterrows()):
                shift = row.get("shift", "")
                conv  = row.get("conversion_rate_pct", 0)
                morn  = row.get("morning_admit_pct", 0)
                wait  = row.get("avg_wait_to_admit_hrs", 0)
                total = row.get("total_visits", 0)
                adm   = row.get("admitted", 0)
                card_color = AFYA_BLUE if "Night" in str(shift) else TEAL

                with col:
                    st.markdown(
                        f'<div style="border-radius:10px;overflow:hidden;margin-bottom:12px;'
                        f'box-shadow:0 1px 4px rgba(0,0,0,0.08);">'
                        f'<div style="background:{card_color};padding:8px 14px;">'
                        f'<span style="color:white;font-size:12px;font-weight:700;letter-spacing:0.5px;">{shift}</span>'
                        f'</div>'
                        f'<div style="background:#FAFCFF;padding:12px 14px;">'
                        f'<div style="display:flex;align-items:baseline;gap:6px;margin-bottom:6px;">'
                        f'<span style="font-size:36px;font-weight:800;color:{card_color};line-height:1;">{_n(conv)}%</span>'
                        f'<span style="font-size:11px;color:#888;">admission rate</span>'
                        f'</div>'
                        f'<div style="font-size:11px;color:#666;margin-bottom:8px;">'
                        f'{int(total or 0):,} visits &rarr; {int(adm or 0):,} admitted'
                        f'</div>'
                        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">'
                        f'<div style="background:#EEF4FF;border-radius:6px;padding:6px 8px;text-align:center;">'
                        f'<div style="font-size:15px;font-weight:700;color:#333;">{_n(wait)} hrs</div>'
                        f'<div style="font-size:10px;color:#888;">avg wait</div>'
                        f'</div>'
                        f'<div style="background:#EEF4FF;border-radius:6px;padding:6px 8px;text-align:center;">'
                        f'<div style="font-size:15px;font-weight:700;color:#333;">{_n(morn)}%</div>'
                        f'<div style="font-size:10px;color:#888;">next morning</div>'
                        f'</div>'
                        f'</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            night_row = df_night[df_night["shift"].str.contains("Night", na=False)]
            if not night_row.empty:
                nr = night_row.iloc[0]
                day_conv = df_night[~df_night["shift"].str.contains("Night", na=False)]["conversion_rate_pct"].values
                day_conv_val = _n(day_conv[0]) if len(day_conv) else "—"
                _note(
                    f"Night: {_n(nr.get('conversion_rate_pct'))}% admission rate vs {day_conv_val}% day. "
                    f"{_n(nr.get('morning_admit_pct'))}% of night admissions recorded next morning. "
                    f"Avg wait: {_n(nr.get('avg_wait_to_admit_hrs'))} hrs."
                )
        else:
            _note("No admission data available for the selected period.", w=True)
    except Exception as e:
        st.warning(f"Night A&E conversion: {e}")

    _gap(12)

    # E2: Service delivery times — peak vs normal vs quiet days
    st.markdown("**E2 — Are Services Slower on Peak Days?**")
    _note(
        "Each visit is assigned to the load tier of its day. "
        "Peak = top 25% busiest days. Quiet = bottom 25%. Times are median minutes from arrival."
    )
    try:
        df_svc = Q.load_peak_day_service_times(filters, run_query)
        if not df_svc.empty:
            for _c in ("median_mins_to_triage", "p90_mins_to_triage",
                       "median_mins_to_consult", "p90_mins_to_consult",
                       "median_mins_to_inv", "p90_mins_to_inv",
                       "median_mins_to_rx", "p90_mins_to_rx",
                       "vitals_rate_pct", "notes_rate_pct", "inv_rate_pct", "rx_rate_pct",
                       "avg_clinicians_on_day", "avg_patients_per_clinician", "total_visits"):
                if _c in df_svc.columns:
                    df_svc[_c] = pd.to_numeric(df_svc[_c], errors="coerce")

            tier_order  = ["Peak (top 25%)", "Normal", "Quiet (bottom 25%)"]
            tier_colors = {"Peak (top 25%)": CORAL, "Normal": TEAL, "Quiet (bottom 25%)": COOL_BLUE}

            c_left, c_right = st.columns([1.1, 0.9])
            with c_left:
                # Grouped bar: median wait times by tier
                metrics = [
                    ("median_mins_to_triage",  "Triage (vitals)"),
                    ("median_mins_to_consult",  "Doctor consult"),
                    ("median_mins_to_inv",      "First investigation"),
                    ("median_mins_to_rx",       "Prescription dispensed"),
                ]
                fig_svc = go.Figure()
                tiers_present = [t for t in tier_order if t in df_svc["load_tier"].values]
                x_labels = [m[1] for m in metrics]

                for tier in tiers_present:
                    row = df_svc[df_svc["load_tier"] == tier].iloc[0]
                    y_vals = [row.get(m[0], 0) for m in metrics]
                    p90s   = [row.get(m[0].replace("median", "p90"), 0) for m in metrics]
                    fig_svc.add_trace(go.Bar(
                        name=tier,
                        x=x_labels,
                        y=y_vals,
                        marker_color=tier_colors.get(tier, GRAY),
                        text=[f"{int(float(v) if pd.notna(v) else 0)}m" for v in y_vals],
                        textposition="outside",
                        hovertemplate=(
                            f"<b>{tier}</b><br>"
                            "%{x}<br>"
                            "Median: %{y:.0f} min<br>"
                            "<extra></extra>"
                        ),
                    ))

                fig_svc.update_layout(
                    barmode="group",
                    height=320,
                    margin=dict(l=0, r=20, t=30, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Median minutes", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                )
                _pc(fig_svc)

            with c_right:
                # Summary table: all tiers with rate metrics + clinician ratio
                disp_cols = ["load_tier", "total_visits", "avg_patients_per_clinician",
                             "vitals_rate_pct", "notes_rate_pct", "inv_rate_pct", "rx_rate_pct",
                             "median_mins_to_triage", "median_mins_to_consult",
                             "median_mins_to_inv", "median_mins_to_rx"]
                df_svc_disp = df_svc[[c for c in disp_cols if c in df_svc.columns]].copy()
                _pc(table_fig(
                    df_svc_disp,
                    col_labels={
                        "load_tier": "Load Tier", "total_visits": "Visits",
                        "avg_patients_per_clinician": "Patients / Clinician",
                        "vitals_rate_pct": "Vitals %",
                        "notes_rate_pct": "Notes %",
                        "inv_rate_pct": "Investigation %",
                        "rx_rate_pct": "Prescription %",
                        "median_mins_to_triage": "Triage (min)",
                        "median_mins_to_consult": "Consult (min)",
                        "median_mins_to_inv": "Investigation (min)",
                        "median_mins_to_rx": "Prescription (min)",
                    },
                    fmt={"vitals_rate_pct": "pct", "notes_rate_pct": "pct",
                         "inv_rate_pct": "pct", "rx_rate_pct": "pct"},
                    height=220,
                ))

                # Clinician ratio on peak vs quiet — call it out
                peak_row  = df_svc[df_svc["load_tier"].str.startswith("Peak")]
                quiet_row = df_svc[df_svc["load_tier"].str.startswith("Quiet")]
                if not peak_row.empty and not quiet_row.empty:
                    pr = peak_row.iloc[0].get("avg_patients_per_clinician", None)
                    qr = quiet_row.iloc[0].get("avg_patients_per_clinician", None)
                    if pd.notna(pr) and pd.notna(qr) and qr > 0:
                        ratio_diff = ((float(pr) - float(qr)) / float(qr)) * 100
                        _note(
                            f"Clinician load: {_n(pr)} patients per clinician on peak days "
                            f"vs {_n(qr)} on quiet days — "
                            f"{'a {:.0f}% higher load when volume spikes.'.format(ratio_diff) if ratio_diff > 0 else 'no meaningful difference.'}"
                        )
        else:
            _note("Insufficient data to compute service time tiers.", w=True)
    except Exception as e:
        st.warning(f"Peak day service times: {e}")

    _gap(12)

    # E3: Off-peak investigation over-ordering by hour
    st.markdown("**E3 — Are Off-peak Clinicians Over-ordering Investigations?**")
    _note(
        "Blue bars show what % of visits had an investigation ordered each hour. "
        "The red line shows visit volume. Where the bar stays high but the line drops, "
        "clinicians are ordering investigations on a disproportionately high share of a smaller patient load."
    )
    try:
        df_inv_hr = Q.load_offpeak_investigation_pattern(filters, run_query)
        if not df_inv_hr.empty:
            for _c in ("hour_eat", "total_visits", "inv_rate_pct", "avg_inv_per_visit"):
                if _c in df_inv_hr.columns:
                    df_inv_hr[_c] = pd.to_numeric(df_inv_hr[_c], errors="coerce")

            hour_labels = [f"{int(h):02d}:00" for h in df_inv_hr["hour_eat"]]

            fig_inv_hr = go.Figure()
            # Bars: investigation rate % by hour
            fig_inv_hr.add_trace(go.Bar(
                x=hour_labels,
                y=df_inv_hr["inv_rate_pct"],
                name="Investigation rate %",
                marker_color=AFYA_BLUE,
                hovertemplate="<b>%{x}</b><br>Investigation rate: %{y:.1f}%<extra></extra>",
            ))
            # Line: visit volume — the denominator that reveals whether high rate is driven by low volume
            fig_inv_hr.add_trace(go.Scatter(
                x=hour_labels,
                y=df_inv_hr["total_visits"],
                mode="lines+markers",
                line=dict(color=CORAL, width=2),
                marker=dict(size=5),
                name="Visit volume",
                yaxis="y2",
                hovertemplate="<b>%{x}</b><br>Visits: %{y:,}<extra></extra>",
            ))
            # Night-shift shading — shade first 7 and last 4 bars
            night_indices = list(range(0, 7)) + list(range(20, 24))
            for hr_idx in night_indices:
                if hr_idx < len(hour_labels):
                    fig_inv_hr.add_vrect(
                        x0=hour_labels[hr_idx], x1=hour_labels[hr_idx],
                        fillcolor=AFYA_BLUE, opacity=0.06,
                        layer="below", line_width=0,
                    )
            fig_inv_hr.update_layout(
                height=300,
                margin=dict(l=0, r=60, t=20, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Hour of Day (EAT)", tickangle=-45),
                yaxis=dict(title="Investigation rate %", rangemode="tozero"),
                yaxis2=dict(title="Visit volume", overlaying="y", side="right",
                            rangemode="tozero", showgrid=False),
                legend=dict(orientation="h", y=1.07, xanchor="right", x=1),
            )
            _pc(fig_inv_hr)

            # Flag any off-peak hours with above-average rate
            overall_avg = float(df_inv_hr["inv_rate_pct"].mean())
            night_df = df_inv_hr[
                (df_inv_hr["hour_eat"] >= 20) | (df_inv_hr["hour_eat"] <= 6)
            ]
            if not night_df.empty:
                night_avg = float(night_df["inv_rate_pct"].mean())
                diff = night_avg - overall_avg
                if diff > 5:
                    _note(
                        f"Night-shift investigation rate ({night_avg:.1f}%) is "
                        f"{diff:.1f} percentage points above the all-hours average ({overall_avg:.1f}%). "
                        f"This pattern is worth reviewing — it may reflect shift-end ordering to clear queues.",
                        w=True,
                    )
                else:
                    _note(
                        f"Night-shift investigation rate ({night_avg:.1f}%) is close to "
                        f"the all-hours average ({overall_avg:.1f}%) — no clear over-ordering signal."
                    )
        else:
            _note("No investigation data for the selected period.", w=True)
    except Exception as e:
        st.warning(f"Off-peak investigation pattern: {e}")

    # E3 drill-down: IP vs OP split + top investigation types
    _gap(8)
    c1, c2 = st.columns([1, 1.2])

    with c1:
        st.markdown("**Where is it happening — Inpatient or Outpatient?**")
        try:
            df_ipop = Q.load_offpeak_ipop_split(filters, run_query)
            if not df_ipop.empty:
                df_ipop["inv_count"]   = pd.to_numeric(df_ipop["inv_count"],   errors="coerce")
                df_ipop["visit_count"] = pd.to_numeric(df_ipop["visit_count"], errors="coerce")
                fig_ipop = go.Figure()
                for vtype, color in [("Outpatient", TEAL), ("Inpatient", AFYA_BLUE)]:
                    sub = df_ipop[df_ipop["visit_type"] == vtype]
                    if not sub.empty:
                        fig_ipop.add_trace(go.Bar(
                            name=vtype,
                            x=sub["shift_type"],
                            y=sub["inv_count"],
                            marker_color=color,
                            text=[f"{int(float(v)):,}" if pd.notna(v) else "" for v in sub["inv_count"]],
                            textposition="outside",
                            hovertemplate=f"<b>{vtype}</b><br>%{{x}}<br>Investigations: %{{y:,}}<extra></extra>",
                        ))
                fig_ipop.update_layout(
                    barmode="group", height=300,
                    margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Investigations ordered", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                )
                _pc(fig_ipop)
            else:
                _note("No IP/OP split data available.", w=True)
        except Exception as e:
            st.warning(f"IP/OP split: {e}")

    with c2:
        st.markdown("**Which procedure types are being ordered off-peak?**")
        try:
            df_types = Q.load_offpeak_top_investigations(filters, run_query)
            if not df_types.empty:
                df_types["inv_count"] = pd.to_numeric(df_types["inv_count"], errors="coerce")
                disc_order = (
                    df_types.groupby("discipline")["inv_count"]
                    .sum().sort_values().index.tolist()
                )
                fig_types = go.Figure()
                for vtype, color in [("Outpatient", TEAL), ("Inpatient", AFYA_BLUE)]:
                    sub = df_types[df_types["visit_type"] == vtype]
                    if sub.empty:
                        continue
                    sub = sub.set_index("discipline").reindex(disc_order).reset_index()
                    fig_types.add_trace(go.Bar(
                        name=vtype,
                        y=sub["discipline"],
                        x=sub["inv_count"],
                        orientation="h",
                        marker_color=color,
                        text=[f"{int(float(v)):,}" if pd.notna(v) else "" for v in sub["inv_count"]],
                        textposition="outside",
                        hovertemplate=f"<b>%{{y}}</b><br>{vtype}: %{{x:,}} off-peak<extra></extra>",
                    ))
                fig_types.update_layout(
                    barmode="group",
                    height=320,
                    margin=dict(l=0, r=80, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Investigations ordered (off-peak)", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                )
                _pc(fig_types)
            else:
                _note("No off-peak investigation disciplines found.", w=True)
        except Exception as e:
            st.warning(f"Top investigations: {e}")

    _gap(12)

    # E4: Discharge timing vs peak admission hours
    st.markdown("**E4 — When Do Discharges Happen? Does It Coincide with Admission Peaks?**")
    _note(
        "Based on actual discharge timestamps. "
        "A discharge cluster during peak admission hours creates a bottleneck — beds "
        "are unavailable precisely when new admissions are arriving."
    )
    try:
        df_disc = Q.load_discharge_timing(filters, run_query)
        df_hm_adm = Q.load_peak_demand_heatmap(filters, run_query)

        if not df_disc.empty:
            for _c in ("discharge_count", "day_num", "hour_eat"):
                if _c in df_disc.columns:
                    df_disc[_c] = pd.to_numeric(df_disc[_c], errors="coerce")

            day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.markdown("<small>**Discharges by hour × day**</small>",
                            unsafe_allow_html=True)
                disc_pivot = df_disc.pivot_table(
                    index="day_name", columns="hour_eat",
                    values="discharge_count", aggfunc="sum", fill_value=0
                ).reindex(day_order)

                fig_disc_hm = go.Figure(go.Heatmap(
                    z=disc_pivot.values,
                    x=disc_pivot.columns.tolist(),
                    y=disc_pivot.index.tolist(),
                    colorscale=[[0, "#F0F4FF"], [1, AFYA_BLUE]],
                    hovertemplate="<b>%{y} %{x}:00</b><br>Discharge events: %{z:,}<extra></extra>",
                    colorbar=dict(thickness=12),
                    text=disc_pivot.values,
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                ))
                fig_disc_hm.update_layout(
                    height=240, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Hour of Day (EAT)"),
                    yaxis=dict(title="Day", autorange="reversed"),
                )
                _pc(fig_disc_hm)

            with c_right:
                # Hourly discharge count across all days
                disc_by_hour = (
                    df_disc.groupby("hour_eat")["discharge_count"].sum().reset_index()
                )
                fig_disc_hr = go.Figure()
                fig_disc_hr.add_trace(go.Bar(
                    x=disc_by_hour["hour_eat"],
                    y=disc_by_hour["discharge_count"],
                    name="Discharge events",
                    marker_color=TEAL,
                    hovertemplate="<b>%{x}:00</b><br>Discharges: %{y:,}<extra></extra>",
                ))

                # Overlay admission peak hours from heatmap if available
                if not df_hm_adm.empty and "hour_of_day" in df_hm_adm.columns:
                    adm_by_hour = (
                        df_hm_adm.groupby("hour_of_day")["inpatient_count"]
                        .sum().reset_index()
                    )
                    peak_adm_hr = int(adm_by_hour.loc[adm_by_hour["inpatient_count"].idxmax(),
                                                       "hour_of_day"])
                    fig_disc_hr.add_vline(
                        x=peak_adm_hr,
                        line=dict(color=CORAL, width=2, dash="dash"),
                        annotation_text=f"Peak admissions ({peak_adm_hr:02d}:00)",
                        annotation_font=dict(size=9, color=CORAL),
                        annotation_position="top right",
                    )

                fig_disc_hr.update_layout(
                    height=240,
                    margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Hour (EAT)", dtick=2),
                    yaxis=dict(title="Discharge events", rangemode="tozero"),
                    showlegend=False,
                )
                _pc(fig_disc_hr)

            # Insight: check if discharge peak overlaps with admission peak
            disc_peak_hr = int(disc_by_hour.loc[disc_by_hour["discharge_count"].idxmax(), "hour_eat"])
            _peak_adm_hr = locals().get("peak_adm_hr", None)
            _note(
                f"Discharge peak: {disc_peak_hr:02d}:00 EAT. "
                + (f"This coincides with peak admission hours ({_peak_adm_hr:02d}:00) — "
                   f"bed turnover is being compressed at the moment of highest demand."
                   if _peak_adm_hr is not None and abs(disc_peak_hr - _peak_adm_hr) <= 3 else
                   (f"Discharge activity ({disc_peak_hr:02d}:00) and admission peaks "
                    f"({_peak_adm_hr:02d}:00) appear offset — bed turnover is not a direct bottleneck."
                    if _peak_adm_hr is not None else
                    f"Discharge peak is at {disc_peak_hr:02d}:00 EAT."))
            )
        else:
            _note("No completed admission records available to estimate discharge timing.", w=True)
    except Exception as e:
        st.warning(f"Discharge timing: {e}")

    _gap(12)

    # ── SECTION F: FORECAST ───────────────────────────────────────────────
    _sh("F — Where is patient volume headed in the next 6 months?", mt=8)
    _note(
        "Base scenario is the model midline. Conservative and stretch scenarios represent "
        "the forecast range. Use the base scenario for planning and the range to "
        "stress-test capacity assumptions."
    )
    try:
        df = Q.load_encounter_forecast(filters, run_query)
        if not df.empty:
            df["visit_month"] = pd.to_datetime(df["visit_month"])
            for _c in ("actual_visits", "actual_outpatient", "actual_inpatient"):
                df[_c] = pd.to_numeric(df[_c], errors="coerce")

            last_month = df["visit_month"].max()
            mo_display = filters.get("months_back") or {
                "Last 12 months": 12, "Last 6 months": 6, "Last 90 days": 3,
            }.get(filters.get("date_range", "Last 12 months"), 12)
            cutoff = last_month - pd.DateOffset(months=mo_display - 1)
            df_act = df[df["visit_month"] >= cutoff].copy()

            median_vol = df["actual_visits"].median()
            df_valid = df[df["actual_visits"] > median_vol * 0.1]
            last3 = df_valid.tail(3) if len(df_valid) >= 3 else df.tail(3)

            op_trend = float(last3["actual_outpatient"].mean())
            ip_trend = float(last3["actual_inpatient"].mean())
            op_std   = float(df_valid["actual_outpatient"].std() or op_trend * 0.1)
            ip_std   = float(df_valid["actual_inpatient"].std()  or ip_trend * 0.1)

            n_fut = 6
            future_months = [last_month + pd.DateOffset(months=i) for i in range(1, n_fut + 1)]

            panel_defs = [
                {
                    "act_col":      "actual_outpatient",
                    "trend":        op_trend,
                    "std":          op_std,
                    "capacity":     float(df["actual_outpatient"].mean()),
                    "color":        AFYA_BLUE,
                    "title":        "Outpatient Encounter Forecast",
                    "xref":         "x",
                    "yref":         "y",
                    "ydomref":      "y domain",
                },
                {
                    "act_col":      "actual_inpatient",
                    "trend":        ip_trend,
                    "std":          ip_std,
                    "capacity":     float(df["actual_inpatient"].mean()),
                    "color":        TEAL,
                    "title":        "Inpatient Encounter Forecast",
                    "xref":         "x2",
                    "yref":         "y2",
                    "ydomref":      "y2 domain",
                },
            ]

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Outpatient Encounter Forecast",
                                 "Inpatient Encounter Forecast"],
                horizontal_spacing=0.10,
            )

            for col_idx, pd_ in enumerate(panel_defs, start=1):
                show_leg  = col_idx == 1
                act_col   = pd_["act_col"]
                color     = pd_["color"]
                xref      = pd_["xref"]
                yref      = pd_["yref"]
                ydomref   = pd_["ydomref"]
                trend     = pd_["trend"]
                std_      = pd_["std"]
                cap_val   = pd_["capacity"]
                conservative = max(0.0, trend - std_)
                stretch      = trend + std_

                last_actual_y = float(df_act[act_col].iloc[-1]) if not df_act.empty else trend

                # ── Actuals ──────────────────────────────────────────
                fig.add_trace(go.Scatter(
                    x=df_act["visit_month"], y=df_act[act_col],
                    name="Actuals (12 mo.)", mode="lines",
                    line=dict(color=TEAL, width=2.5),
                    showlegend=show_leg,
                ), row=1, col=col_idx)

                # ── Forecast-start vertical line ──────────────────────
                fig.add_shape(
                    type="line",
                    xref=xref, yref=ydomref,
                    x0=last_month, x1=last_month, y0=0, y1=1,
                    line=dict(color=PURPLE, width=1.5, dash="dash"),
                )
                fig.add_annotation(
                    xref=xref, yref=ydomref,
                    x=last_month, y=0.97,
                    text="Forecast starts here",
                    showarrow=False,
                    font=dict(size=8, color=PURPLE),
                    xanchor="right",
                )

                # ── Capacity reference line ───────────────────────────
                x_start = df_act["visit_month"].min() if not df_act.empty else last_month
                fig.add_shape(
                    type="line",
                    xref=xref, yref=yref,
                    x0=x_start, x1=future_months[-1],
                    y0=cap_val, y1=cap_val,
                    line=dict(color=GRAY, width=1.5, dash="dot"),
                )
                fig.add_annotation(
                    xref=xref, yref=yref,
                    x=x_start, y=cap_val,
                    text=f"  Capacity baseline — to be confirmed ({int(cap_val):,})",
                    xanchor="left", yanchor="bottom",
                    showarrow=False,
                    font=dict(size=8, color=GRAY),
                )

                # ── Three scenario lines ──────────────────────────────
                scenario_specs = [
                    ("Conservative", conservative, "dash",  1.5, 0.55),
                    ("Base",          trend,        "solid", 2.5, 1.0),
                    ("Stretch",       stretch,      "dash",  1.5, 0.55),
                ]
                for sc_name, sc_val, sc_dash, sc_w, sc_op in scenario_specs:
                    sc_color = color if sc_name == "Base" else MUTED
                    # Bridge from last actual to first forecast point
                    fig.add_trace(go.Scatter(
                        x=[last_month, future_months[0]],
                        y=[last_actual_y, sc_val],
                        mode="lines",
                        line=dict(color=sc_color, width=1, dash="dot"),
                        showlegend=False, hoverinfo="skip",
                    ), row=1, col=col_idx)
                    fig.add_trace(go.Scatter(
                        x=future_months,
                        y=[sc_val] * n_fut,
                        name=sc_name,
                        mode="lines",
                        line=dict(color=sc_color, width=sc_w, dash=sc_dash),
                        opacity=sc_op,
                        showlegend=show_leg,
                        hovertemplate=f"{sc_name}: %{{y:,.0f}}<extra></extra>",
                    ), row=1, col=col_idx)
                    # End-of-line label
                    fig.add_annotation(
                        xref=xref, yref=yref,
                        x=future_months[-1], y=sc_val,
                        text=f"  {sc_name} ({int(sc_val):,})",
                        xanchor="left", yanchor="middle",
                        showarrow=False,
                        font=dict(size=8,
                                  color=color if sc_name == "Base" else MUTED),
                    )

                # ── Capacity pressure callout ─────────────────────────
                if trend > cap_val * 1.05:
                    crossing_month = future_months[0].strftime("%b %Y")
                    fig.add_annotation(
                        xref=xref, yref=yref,
                        x=future_months[0], y=trend,
                        text=f"⚠ Projected capacity pressure — {crossing_month}",
                        showarrow=True, arrowhead=2, arrowcolor=CORAL,
                        font=dict(size=8, color=CORAL),
                        ax=0, ay=-35,
                    )

            fig.update_layout(
                height=400,
                margin=dict(l=0, r=130, t=55, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.06,
                    xanchor="right", x=1,
                    font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            for c in [1, 2]:
                fig.update_yaxes(
                    title_text="Visits", row=1, col=c,
                    showgrid=True, gridcolor="#EBF3FB", zeroline=False,
                    tickfont=dict(color=MUTED, size=10),
                )
            fig.update_xaxes(
                showgrid=False, tickfont=dict(color=MUTED, size=10),
                tickformat="%b %Y",
            )
            _pc(fig)
            _note(
                "Solid teal line = last 12 months actuals. "
                "Dashed lines = 6-month forecast scenarios."
            )

            # ── Four KPI cards ────────────────────────────────────────
            latest = df_act.iloc[-1] if not df_act.empty else df.iloc[-1]
            op_cons = int(max(0, op_trend - op_std))
            op_base = int(op_trend)
            op_str  = int(op_trend + op_std)
            ip_cons = int(max(0, ip_trend - ip_std))
            ip_base = int(ip_trend)
            ip_str  = int(ip_trend + ip_std)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                _kpi("Last Month Outpatient", _n(latest.get("actual_outpatient")))
            with c2:
                _kpi("Last Month Inpatient",  _n(latest.get("actual_inpatient")),
                     color=TEAL)
            with c3:
                _kpi("Next Month OP Forecast", _n(op_base),
                     s=f"Range: {op_cons:,} – {op_str:,}",
                     color=AFYA_BLUE)
            with c4:
                _kpi("Next Month IP Forecast", _n(ip_base),
                     s=f"Range: {ip_cons:,} – {ip_str:,}",
                     color=TEAL)
    except Exception as e:
        st.warning(f"Forecast: {e}")

    _gap(12)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PATIENT DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════

def render_tab2_segmentation(filters: dict, run_query):
    # ── KPI ROW ───────────────────────────────────────────────────────────
    _sh("Patient Overview")
    try:
        df = Q.load_seg_kpis(filters, run_query)
        if not df.empty:
            row = df.iloc[0]
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: _kpi("Total Patients",  _n(row.get("total_patients")))
            with c2: _kpi("Chronic",         _n(row.get("chronic_patients")),
                          str(_p(row.get("chronic_rate_pct")) or "—") + " of patients",
                          AFYA_BLUE)
            with c3: _kpi("Repeat Patients", _n(row.get("repeat_patients")),
                          str(_p(row.get("repeat_rate_pct")) or "—") + " repeat rate", TEAL)
            with c4: _kpi("Single Visit",    _n(row.get("single_visit")))
            with c5: _kpi("Avg Visits / Pt", str(row.get("avg_visits", "—")))
    except Exception as e:
        st.warning(f"Seg KPIs: {e}")

    _gap(16)

    # ── A: AGE GROUP & GENDER DISTRIBUTION ───────────────────────────────
    _sh("A — Age Group & Gender Distribution", mt=8)
    try:
        df = Q.load_demographics_age_sex(filters, run_query)
        if not df.empty:
            gender_filter = st.radio(
                "Filter by gender", ["All", "Female", "Male"],
                horizontal=True, key="demo_gender_filter",
            )
            df_f = df.copy()
            if gender_filter == "Female":
                df_f = df_f[df_f["sex"].isin(["F", "FEMALE"])]
            elif gender_filter == "Male":
                df_f = df_f[df_f["sex"].isin(["M", "MALE"])]

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Patients by Age Group**")
                age_summary = (df_f.groupby("age_group")["total"]
                               .sum().reset_index()
                               .sort_values("total", ascending=True))
                _pc(hbar_chart(
                    age_summary, x="total", y="age_group",
                    color=AFYA_BLUE, x_label="Patients", height=280,
                ))

            with c2:
                st.markdown("**Chronic vs Non-Chronic by Age**")
                chronic_summary = (df_f.groupby("age_group")
                                   .agg(chronic=("chronic", "sum"),
                                        non_chronic=("non_chronic", "sum"))
                                   .reset_index()
                                   .sort_values("chronic", ascending=True))
                _pc(hbar_chart(
                    chronic_summary, x="chronic", y="age_group",
                    color=ORANGE, x_label="Chronic Patients", height=280,
                ))

            with c3:
                st.markdown("**Gender Distribution**")
                gender_totals = (df.groupby("sex")["total"].sum().reset_index())
                gender_totals["sex_clean"] = gender_totals["sex"].map(
                    {"F": "Female", "FEMALE": "Female", "M": "Male", "MALE": "Male"}
                ).fillna("Other")
                gender_agg = (gender_totals.groupby("sex_clean")["total"].sum().reset_index())
                _pc(donut(
                    labels=gender_agg["sex_clean"].tolist(),
                    values=gender_agg["total"].tolist(),
                    color_map={"Female": PURPLE, "Male": TEAL, "Other": GRAY},
                    height=280,
                ))
    except Exception as e:
        st.warning(f"Demographics: {e}")

    _gap(12)

    # ── B: AGE COHORT GROWTH OVER TIME ───────────────────────────────────
    _sh("B — Age Cohort Growth Over Time", mt=8)
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
        st.warning(f"Cohort growth: {e}")

    _gap(12)

    # ── C: NEW VS RETURNING — VOLUMES, AGE & VISIT TYPE ──────────────────
    _sh("C — New vs Returning Patient Trends", mt=8)
    try:
        df_nvr = Q.load_new_vs_returning(filters, run_query)
        if not df_nvr.empty:
            for _c in ("total_patients", "new_patients", "returning_patients"):
                df_nvr[_c] = pd.to_numeric(df_nvr[_c], errors="coerce")
            df_nvr["new_pct"] = (
                df_nvr["new_patients"] / df_nvr["total_patients"].replace(0, float("nan")) * 100
            ).round(1)
            df_nvr["return_pct"] = (
                df_nvr["returning_patients"] / df_nvr["total_patients"].replace(0, float("nan")) * 100
            ).round(1)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**New & Returning Patient Volumes**")
                fig_nvr = go.Figure()
                fig_nvr.add_trace(go.Scatter(
                    x=df_nvr["visit_month"], y=df_nvr["new_patients"],
                    mode="lines+markers", name="New patients",
                    line=dict(color=ORANGE, width=2.5),
                    marker=dict(size=5),
                    hovertemplate="<b>%{x|%b %Y}</b><br>New: %{y:,}<extra></extra>",
                ))
                fig_nvr.add_trace(go.Scatter(
                    x=df_nvr["visit_month"], y=df_nvr["returning_patients"],
                    mode="lines+markers", name="Returning patients",
                    line=dict(color=TEAL, width=2.5),
                    marker=dict(size=5),
                    hovertemplate="<b>%{x|%b %Y}</b><br>Returning: %{y:,}<extra></extra>",
                ))
                fig_nvr.update_layout(
                    height=280, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Patients", rangemode="tozero",
                               showgrid=True, gridcolor="#EBF3FB"),
                    xaxis=dict(title="Month"),
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                )
                _pc(fig_nvr)

            with c2:
                st.markdown("**New Patient Acquisition Trend**")
                df_sorted = df_nvr.sort_values("visit_month").reset_index(drop=True)
                rolling_avg = df_sorted["new_patients"].rolling(3, min_periods=1).mean()
                fig_acq = go.Figure()
                fig_acq.add_trace(go.Bar(
                    x=df_sorted["visit_month"],
                    y=df_sorted["new_patients"],
                    name="New patients",
                    marker_color=ORANGE, opacity=0.75,
                    hovertemplate="<b>%{x|%b %Y}</b><br>New: %{y:,}<extra></extra>",
                ))
                fig_acq.add_trace(go.Scatter(
                    x=df_sorted["visit_month"],
                    y=rolling_avg,
                    name="3-month avg",
                    mode="lines",
                    line=dict(color=AFYA_BLUE, width=2.5, dash="dot"),
                    hovertemplate="<b>%{x|%b %Y}</b><br>3-mo avg: %{y:.0f}<extra></extra>",
                ))
                fig_acq.update_layout(
                    height=280, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="New Patients", rangemode="tozero",
                               showgrid=True, gridcolor="#EBF3FB"),
                    xaxis=dict(title="Month"),
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                )
                _pc(fig_acq)
    except Exception as e:
        st.warning(f"New vs returning: {e}")

    _gap(12)

    # C2: Distribution per age group and visit type
    try:
        df_dist = Q.load_visit_distribution(filters, run_query)
        if not df_dist.empty:
            for _c in ("patient_count", "visit_count"):
                df_dist[_c] = pd.to_numeric(df_dist[_c], errors="coerce")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Distribution by Age Group — New vs Returning**")
                age_dist = (
                    df_dist.groupby(["age_group", "patient_type"])["patient_count"]
                    .sum().reset_index()
                )
                age_order = ["Paediatric (<18)", "Young Adult (18–34)",
                             "Adult (35–54)", "Senior (55+)"]
                fig_age = go.Figure()
                pt_colors = {"New": ORANGE, "Returning": TEAL}
                for pt in ["New", "Returning"]:
                    sub = (age_dist[age_dist["patient_type"] == pt]
                           .set_index("age_group")
                           .reindex(age_order)
                           .reset_index())
                    fig_age.add_trace(go.Bar(
                        name=pt,
                        y=sub["age_group"],
                        x=sub["patient_count"],
                        orientation="h",
                        marker_color=pt_colors[pt],
                        hovertemplate=f"<b>%{{y}}</b><br>{pt}: %{{x:,}}<extra></extra>",
                    ))
                fig_age.update_layout(
                    barmode="group",
                    height=260, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Patients", rangemode="tozero"),
                    yaxis=dict(autorange="reversed"),
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                )
                _pc(fig_age)

            with c2:
                st.markdown("**Distribution by Visit Type — New vs Returning**")
                vtype_dist = (
                    df_dist.groupby(["visit_type", "patient_type"])["patient_count"]
                    .sum().reset_index()
                )
                total_patients = float(vtype_dist["patient_count"].sum() or 1)
                vtype_pivot = vtype_dist.pivot_table(
                    index="visit_type", columns="patient_type",
                    values="patient_count", fill_value=0
                ).reset_index()

                fig_vtype = go.Figure()
                for pt in ["New", "Returning"]:
                    if pt in vtype_pivot.columns:
                        fig_vtype.add_trace(go.Bar(
                            name=pt,
                            x=vtype_pivot["visit_type"],
                            y=vtype_pivot[pt],
                            marker_color=pt_colors[pt],
                            text=[f"{v:,}" for v in vtype_pivot[pt]],
                            textposition="outside",
                            hovertemplate=f"<b>%{{x}}</b><br>{pt}: %{{y:,}}<extra></extra>",
                        ))
                fig_vtype.update_layout(
                    barmode="group",
                    height=260, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Patients", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                )
                _pc(fig_vtype)

                # Insight: inpatient conversion rate for new vs returning
                ip_new = float(vtype_dist.loc[
                    (vtype_dist["visit_type"] == "Inpatient") &
                    (vtype_dist["patient_type"] == "New"), "patient_count"
                ].sum() or 0)
                ip_ret = float(vtype_dist.loc[
                    (vtype_dist["visit_type"] == "Inpatient") &
                    (vtype_dist["patient_type"] == "Returning"), "patient_count"
                ].sum() or 0)
                tot_new = float(vtype_dist.loc[
                    vtype_dist["patient_type"] == "New", "patient_count"].sum() or 1)
                tot_ret = float(vtype_dist.loc[
                    vtype_dist["patient_type"] == "Returning", "patient_count"].sum() or 1)
                if tot_new > 0 and tot_ret > 0:
                    _note(
                        f"Inpatient rate — new patients: {ip_new/tot_new*100:.1f}% vs "
                        f"returning: {ip_ret/tot_ret*100:.1f}%. "
                        + ("New patients are more likely to present as emergencies requiring admission."
                           if ip_new/tot_new > ip_ret/tot_ret else
                           "Returning patients have a higher inpatient conversion — likely chronic disease management.")
                    )
    except Exception as e:
        st.warning(f"Visit distribution: {e}")

    _gap(12)

    # ── D: PAYER HABIT SWITCH — NEW → RETURNING ──────────────────────────
    _sh("D — Do Payer Habits Change When New Patients Return?", mt=8)
    _note(
        "Many new patients enter as cash payers during an emergency (high-friction, low-loyalty). "
        "If they return months later on corporate insurance (Jubilee, APA, SHIF), your clinical "
        "experience converted an emergency visitor into a long-term premium client. "
        "The Sankey below maps what payer type patients used on their first visit → "
        "what they use when they come back."
    )
    try:
        df_sk = Q.load_payer_switch_sankey(filters, run_query)
        if not df_sk.empty:
            df_sk["patient_count"] = pd.to_numeric(df_sk["patient_count"], errors="coerce")

            # Build Sankey nodes: source payers (left) → target payers (right)
            payer_types = ["Cash", "NHIF / SHA", "Insurance"]
            source_labels = [f"{p} (first visit)" for p in payer_types]
            target_labels = [f"{p} (returning)" for p in payer_types]
            all_labels = source_labels + target_labels

            node_colors = {
                "Cash":       ORANGE,
                "NHIF / SHA": TEAL,
                "Insurance":  AFYA_BLUE,
            }
            node_color_list = (
                [node_colors.get(p, GRAY) for p in payer_types] +
                [node_colors.get(p, GRAY) for p in payer_types]
            )

            link_sources, link_targets, link_values, link_colors = [], [], [], []
            for _, row in df_sk.iterrows():
                src = row["source_payer"]
                tgt = row["target_payer"]
                cnt = row["patient_count"]
                if src not in payer_types or tgt not in payer_types:
                    continue
                si = payer_types.index(src)           # 0–2 (left side)
                ti = payer_types.index(tgt) + 3       # 3–5 (right side)
                link_sources.append(si)
                link_targets.append(ti)
                link_values.append(float(cnt or 0))
                # Colour by source payer, semi-transparent (rgba required by Plotly Sankey)
                base = node_colors.get(src, GRAY)
                r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
                link_colors.append(f"rgba({r},{g},{b},0.45)")

            fig_sk = go.Figure(go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20, thickness=24,
                    label=all_labels,
                    color=node_color_list,
                    hovertemplate="%{label}<br>%{value:,} patients<extra></extra>",
                ),
                link=dict(
                    source=link_sources,
                    target=link_targets,
                    value=link_values,
                    color=link_colors,
                    hovertemplate=(
                        "%{source.label} → %{target.label}<br>"
                        "%{value:,} patients<extra></extra>"
                    ),
                ),
            ))
            fig_sk.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="white",
                font=dict(size=12),
            )
            _pc(fig_sk)

            # Highlight key conversion: Cash → Insurance
            cash_upgrades = df_sk[
                (df_sk["source_payer"] == "Cash") &
                (df_sk["target_payer"].isin(["Insurance", "NHIF / SHA"]))
            ]["patient_count"].sum()
            cash_stays = df_sk[
                (df_sk["source_payer"] == "Cash") &
                (df_sk["target_payer"] == "Cash")
            ]["patient_count"].sum()
            if cash_upgrades > 0 or cash_stays > 0:
                total_cash_first = float(cash_upgrades + cash_stays or 1)
                upgrade_pct = cash_upgrades / total_cash_first * 100
                _note(
                    f"{int(cash_upgrades):,} patients who first visited as cash payers "
                    f"returned on insurance or NHIF/SHA — "
                    f"{upgrade_pct:.0f}% of all cash-first returning patients. "
                    + ("This suggests strong loyalty conversion from emergency to enrolled patients."
                       if upgrade_pct > 20 else
                       "Most cash-first patients continue paying out-of-pocket on return visits.")
                )
        else:
            _note(
                "No payer switch data available — patients may not have enough return visits "
                "in the selected period, or payer information is incomplete.",
                w=True,
            )
    except Exception as e:
        st.warning(f"Payer switch: {e}")

    _gap(12)

    # ── E: PAYER MIX BY AGE GROUP ─────────────────────────────────────────
    _sh("E — Payer Mix by Age Group", mt=8)
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

    # ── F: UNIFIED PATIENT PROFILE: CONDITION → REVENUE ──────────────────
    _sh("F — Patient Revenue Intelligence", mt=8)
    _note(
        "How revenue concentrates across clinical conditions — "
        "and what the payer and visit-type mix means for cash flow and commercial risk."
    )

    # ── F1: Revenue Concentration (Pareto Breakdown) ──────────────────────
    _gap(4)
    st.markdown("#### 📊 Revenue Concentration — Pareto Breakdown")
    try:
        df_prt = Q.load_pareto(filters, run_query)
        if not df_prt.empty:
            for _c in ("tier_revenue", "revenue_share_pct", "avg_spend",
                       "patient_count", "avg_visits"):
                df_prt[_c] = pd.to_numeric(df_prt[_c], errors="coerce")

            tier_order  = ["Top 10%", "Top 11–20%", "Middle 21–50%", "Bottom 50%"]
            tier_badges = {
                "Top 10%":       "🔵",
                "Top 11–20%":    "🟢",
                "Middle 21–50%": "🟡",
                "Bottom 50%":    "🪟",
            }
            tier_colors = {
                "Top 10%":       AFYA_BLUE,
                "Top 11–20%":    TEAL,
                "Middle 21–50%": ORANGE,
                "Bottom 50%":    GRAY,
            }

            c1, c2 = st.columns([1, 1.4])
            with c1:
                _pc(donut(
                    labels=df_prt["revenue_tier"].tolist(),
                    values=df_prt["tier_revenue"].tolist(),
                    color_map=tier_colors,
                    height=300,
                ))
            with c2:
                st.markdown("#### 📋 Tier Summary")
                df_prt_disp = df_prt.copy()
                df_prt_disp["revenue_tier"] = df_prt_disp["revenue_tier"].apply(
                    lambda t: f"{tier_badges.get(t, '')} {t}"
                )
                df_prt_disp["tier_revenue_fmt"] = df_prt_disp["tier_revenue"].apply(
                    lambda v: f"KES {v/1e6:.1f}M" if v >= 1e6 else f"KES {v:,.0f}"
                )
                df_prt_disp["revenue_share_fmt"] = df_prt_disp["revenue_share_pct"].apply(
                    lambda v: f"{v:.1f}%" if pd.notna(v) else "—"
                )
                df_prt_disp["avg_spend_fmt"] = df_prt_disp["avg_spend"].apply(
                    lambda v: f"KES {v:,.0f}" if pd.notna(v) else "—"
                )
                _pc(table_fig(
                    df_prt_disp[["revenue_tier", "patient_count", "tier_revenue_fmt",
                                  "revenue_share_fmt", "avg_spend_fmt", "avg_visits"]],
                    col_labels={
                        "revenue_tier":      "Tier",
                        "patient_count":     "Patients",
                        "tier_revenue_fmt":  "Revenue",
                        "revenue_share_fmt": "Share",
                        "avg_spend_fmt":     "Avg Spend",
                        "avg_visits":        "Avg Visits",
                    },
                    fmt={},
                    height=240,
                ))
    except Exception as e:
        st.warning(f"Pareto: {e}")

    _gap(16)

    # ── F2: Unified Patient Revenue Profile Matrix ────────────────────────
    st.markdown("#### 📋 Unified Patient Revenue Profile Matrix")
    _note(
        "Each row is a presenting condition, cross-cut by how much revenue it generates, "
        "where it sits in the revenue Pareto, how patients are admitted, "
        "and what payer mix drives its cash flow — with a commercial risk interpretation."
    )
    try:
        df_mx = Q.load_revenue_profile_matrix(filters, run_query)
        if not df_mx.empty:
            for _c in ("total_revenue", "revenue_share_pct", "ip_pct", "op_pct",
                       "cash_pct", "nhif_pct", "corp_pct", "visit_count"):
                df_mx[_c] = pd.to_numeric(df_mx[_c], errors="coerce")

            tier_badge = {
                "Top 10%":       "🔵 Top 10% (High Ticket)",
                "Top 11–20%":    "🟢 Top 11–20% (Mid-Tier Anchor)",
                "Middle 21–50%": "🟡 Middle 21–50% (Volume Driver)",
                "Bottom 50%":    "🪟 Bottom 50% (Low Ticket)",
            }

            def _risk_label(row) -> str:
                cond   = str(row.get("condition", "")).lower()
                tier   = str(row.get("pareto_tier", ""))
                ip     = float(row.get("ip_pct")   or 0)
                op     = float(row.get("op_pct")   or 0)
                cash   = float(row.get("cash_pct") or 0)
                corp   = float(row.get("corp_pct") or 0)
                nhif   = float(row.get("nhif_pct") or 0)

                is_unclass = any(x in cond for x in
                                 ["unclassif", "procedure", "unknown", "other"])
                is_trauma  = any(x in cond for x in
                                 ["trauma", "injury", "fractur", "wound", "burn"])
                is_symptoms = any(x in cond for x in
                                  ["symptom", "undiag", "sign", "ill-defined"])

                if is_unclass and corp >= 50:
                    return ("⚠️ Data Leak: High-value claims heavily exposed to insurance "
                            "audits. Needs immediate EHR clean-up.")
                if is_trauma and ip >= 70:
                    return ("💎 Premium Profit Center: Low volume but high margin. "
                            "Highly sensitive to OR throughput and bed availability.")
                if is_symptoms and op >= 80 and cash >= 60:
                    return ("🕳️ Efficiency Drain: Consumes heavy triage and doctor hours "
                            "for small, un-admitted individual invoices.")
                if ip >= 60 and corp >= 50 and nhif >= 15:
                    return ("🛡️ NHIF/SHA Exposure: Inpatient-heavy with significant public "
                            "scheme dependency — long reimbursement cycles.")
                if ip >= 60 and corp >= 50:
                    return ("⏳ Working Capital Trap: High ward bed utilisation facing "
                            "long corporate approval and payment cycles.")
                if op >= 75 and cash >= 50 and "Bottom" in tier:
                    return ("🕳️ Efficiency Drain: Consumes heavy triage and doctor hours "
                            "for small, un-admitted individual invoices.")
                if op >= 75 and cash >= 45:
                    return ("⚡ Liquidity Engine: Rapid, low-friction outpatient visits "
                            "generating immediate daily cash flow.")
                if nhif >= 40:
                    return ("🛡️ NHIF/SHA Exposure: Heavy dependency on public scheme "
                            "reimbursement with associated collection delays.")
                if corp >= 60:
                    return ("📋 Insurance Cycle Risk: Corporate-dominant payer mix with "
                            "bed-day intensive utilisation — manage receivables tightly.")
                return ("📊 Mixed Profile: Balanced payer and visit-type mix — "
                        "monitor for drift toward either extreme.")

            # Build formatted display columns
            df_mx["Revenue (Share)"] = df_mx.apply(
                lambda r: (
                    f"KES {r['total_revenue']/1e6:.1f}M "
                    f"({r['revenue_share_pct']:.1f}%)"
                    if r["total_revenue"] >= 1e6
                    else f"KES {r['total_revenue']:,.0f} ({r['revenue_share_pct']:.1f}%)"
                ), axis=1
            )
            df_mx["Pareto Tier"] = df_mx["pareto_tier"].map(
                lambda t: tier_badge.get(t, t)
            )
            df_mx["Commercial Risk & Action"] = df_mx.apply(_risk_label, axis=1)

            disp = pd.DataFrame({
                "Presenting Condition": df_mx["condition"],
                "Revenue (Share)":      df_mx["Revenue (Share)"],
                "Pareto Tier":          df_mx["Pareto Tier"],
                "IP %":                 df_mx["ip_pct"].fillna(0).round(0),
                "OP %":                 df_mx["op_pct"].fillna(0).round(0),
                "Cash %":               df_mx["cash_pct"].fillna(0).round(0),
                "Corp %":               df_mx["corp_pct"].fillna(0).round(0),
                "NHIF %":               df_mx["nhif_pct"].fillna(0).round(0),
                "Commercial Risk & Action": df_mx["Commercial Risk & Action"],
            })

            st.dataframe(
                disp,
                use_container_width=True,
                hide_index=True,
                height=520,
                column_config={
                    "IP %":   st.column_config.NumberColumn("IP %",   format="%.0f%%"),
                    "OP %":   st.column_config.NumberColumn("OP %",   format="%.0f%%"),
                    "Cash %": st.column_config.NumberColumn("Cash %", format="%.0f%%"),
                    "Corp %": st.column_config.NumberColumn("Corp %", format="%.0f%%"),
                    "NHIF %": st.column_config.NumberColumn("NHIF %", format="%.0f%%"),
                },
            )

        else:
            _note("No revenue data available for the selected period.", w=True)
    except Exception as e:
        st.warning(f"Revenue profile matrix: {e}")


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
                          str(_p(rr) or "—") + " — benchmark 60%",
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

    _sh("B — Who Drops Out? LTFU Rate by Demographics & Payer", mt=8)
    _note(
        "For each demographic factor, the % of chronic patients who have not returned in over 90 days. "
        "Bars above 50% flag groups where the majority of patients are already lost to follow-up."
    )
    try:
        df_cor = Q.load_ltfu_correlation(filters, run_query)
        if not df_cor.empty:
            df_cor["ltfu_rate_pct"] = pd.to_numeric(df_cor["ltfu_rate_pct"], errors="coerce")
            df_cor["total"] = pd.to_numeric(df_cor["total"], errors="coerce")
            df_cor["ltfu"] = pd.to_numeric(df_cor["ltfu"], errors="coerce")
            df_cor["retained"] = pd.to_numeric(df_cor["retained"], errors="coerce")

            factors = df_cor["factor"].unique().tolist()
            cols = st.columns(len(factors))
            factor_colors = {"Age Group": AFYA_BLUE, "Sex": TEAL, "Payer": CORAL}

            for col, factor in zip(cols, factors):
                sub = df_cor[df_cor["factor"] == factor].sort_values("ltfu_rate_pct", ascending=False)
                with col:
                    st.markdown(f"**{factor}**")
                    fig_b = go.Figure(go.Bar(
                        x=sub["ltfu_rate_pct"],
                        y=sub["dimension"],
                        orientation="h",
                        marker_color=[CORAL if v >= 50 else ORANGE if v >= 30 else TEAL
                                      for v in sub["ltfu_rate_pct"]],
                        text=[f"{v:.0f}% ({int(n):,} pts)" for v, n in
                              zip(sub["ltfu_rate_pct"], sub["total"])],
                        textposition="outside",
                        hovertemplate=(
                            "<b>%{y}</b><br>LTFU: %{x:.1f}%<br>"
                            "<extra></extra>"
                        ),
                    ))
                    fig_b.add_vline(x=50, line=dict(color=GRAY, width=1, dash="dash"))
                    fig_b.update_layout(
                        height=max(180, len(sub) * 40 + 60),
                        margin=dict(l=0, r=80, t=10, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="LTFU %", range=[0, 115]),
                        yaxis=dict(title=""),
                        showlegend=False,
                    )
                    _pc(fig_b)

            # Highlight the highest-risk finding
            worst = df_cor.loc[df_cor["ltfu_rate_pct"].idxmax()]
            _note(
                f"Highest LTFU rate: {worst['dimension']} ({worst['factor']}) — "
                f"{worst['ltfu_rate_pct']:.0f}% of {int(worst['total'] or 0):,} chronic patients "
                f"have not returned in 90+ days.",
                w=worst["ltfu_rate_pct"] >= 50,
            )
    except Exception as e:
        st.warning(f"LTFU correlation: {e}")

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

    _gap(16)

    # ── D: DEMOGRAPHIC × DIAGNOSIS REVENUE RISK ──────────────────────────────
    _sh("D — Which Demographic-Diagnosis Group Holds the Most Revenue at Risk?", mt=8)
    _note(
        "LTFU patients × 4 expected visits/year × avg revenue per visit, grouped by "
        "age, gender, condition, and payer. The top row is your highest-priority re-engagement target."
    )
    try:
        df_ddrr = Q.load_demographic_diagnosis_revenue_risk(filters, run_query)
        if not df_ddrr.empty:
            for _c in ("ltfu_patients", "avg_rev_per_visit", "revenue_at_risk"):
                df_ddrr[_c] = pd.to_numeric(df_ddrr[_c], errors="coerce")

            top = df_ddrr.iloc[0]
            _note(
                f"Highest concentration: {top['gender']} aged {top['age_group']} "
                f"with {top['condition']} ({top['payer']}) — "
                f"{int(top['ltfu_patients']):,} LTFU patients, "
                f"est. KES {float(top['revenue_at_risk'])/1e6:.1f}M at risk annually.",
                w=True,
            )

            df_ddrr["risk_fmt"] = df_ddrr["revenue_at_risk"].apply(
                lambda v: f"KES {v/1e6:.1f}M" if v >= 1e6 else f"KES {v:,.0f}"
            )
            df_ddrr["avg_rev_fmt"] = df_ddrr["avg_rev_per_visit"].apply(
                lambda v: f"KES {v:,.0f}"
            )
            _pc(table_fig(
                df_ddrr[["age_group", "gender", "condition", "payer",
                          "ltfu_patients", "avg_rev_fmt", "risk_fmt"]].head(20),
                col_labels={
                    "age_group": "Age Group", "gender": "Gender",
                    "condition": "Condition", "payer": "Payer",
                    "ltfu_patients": "LTFU Patients",
                    "avg_rev_fmt": "Avg Rev/Visit",
                    "risk_fmt": "Annual Revenue at Risk",
                },
                height=480,
            ))
    except Exception as e:
        st.warning(f"Demographic-diagnosis revenue risk: {e}")

    _gap(12)

    # ── E: RETAINED PATIENT CLINICAL FOOTPRINT ────────────────────────────────
    _sh("E — What Are Retained Patients Actually Coming Back For?", mt=8)
    _note(
        "Each metric is independent — a single visit can include consultation, lab, and prescription. "
        "Read each cell as: 'X% of this payer's visits included this service.'"
    )
    try:
        df_fp = Q.load_retained_patient_footprint(filters, run_query)
        if not df_fp.empty:
            metric_cols = ["consult_rate_pct", "investigation_rate_pct",
                           "rx_rate_pct", "admission_rate_pct", "pharmacy_only_pct"]
            metric_labels = ["Consultation", "Lab / Imaging",
                             "Prescription", "Inpatient Admission", "Pharmacy-only"]
            for _c in ["retained_patients"] + metric_cols:
                df_fp[_c] = pd.to_numeric(df_fp[_c], errors="coerce")

            payers = df_fp["payer"].tolist()
            z = [[float(df_fp.loc[df_fp["payer"] == p, c].iloc[0])
                  if not df_fp.loc[df_fp["payer"] == p, c].empty else 0
                  for p in payers]
                 for c in metric_cols]
            text_z = [[f"{v:.0f}%" for v in row] for row in z]

            fig_fp = go.Figure(go.Heatmap(
                z=z,
                x=payers,
                y=metric_labels,
                text=text_z,
                texttemplate="%{text}",
                colorscale=[[0, "#eaf4fb"], [0.5, TEAL], [1, AFYA_BLUE]],
                zmin=0, zmax=100,
                hovertemplate="<b>%{y}</b> — %{x}<br>%{z:.1f}% of visits<extra></extra>",
                showscale=True,
                colorbar=dict(title="%", thickness=12, len=0.8),
            ))
            fig_fp.update_layout(
                height=280,
                margin=dict(l=0, r=20, t=20, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(side="top"),
                yaxis=dict(autorange="reversed"),
            )
            _pc(fig_fp)

            for _, row in df_fp.iterrows():
                pharm_pct = float(row.get("pharmacy_only_pct") or 0)
                if pharm_pct > 30:
                    _note(
                        f"{row['payer']}: {pharm_pct:.0f}% of retained visits are pharmacy-only — "
                        "patients collecting medication without clinical review."
                    )
    except Exception as e:
        st.warning(f"Retained patient footprint: {e}")

    _gap(12)

    # ── G: COST-OF-CARE CORRELATION & WAIT-TIME DROPOUT ──────────────────────
    _sh("G — Cost, Invoice Size & Wait Times: Do They Drive LTFU?", mt=8)
    _note(
        "Comparing medication costs and average invoice sizes between active, lapsing, and LTFU "
        "outpatient patients — split by payer. For insured patients, investigation wait times "
        "(hours from test order to result) are measured per lifecycle status."
    )
    try:
        df_cc = Q.load_cost_dropout_correlation(filters, run_query)
        if not df_cc.empty:
            for _c in ("patient_count", "avg_rx_cost", "avg_invoice_size", "avg_inv_wait_hrs"):
                df_cc[_c] = pd.to_numeric(df_cc[_c], errors="coerce")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Average Invoice Size by Payer × Lifecycle**")
                lifecycle_order = ["Active (≤90d)", "Lapsing (91-180d)", "LTFU (>180d)"]
                lc_colors = {"Active (≤90d)": TEAL, "Lapsing (91-180d)": ORANGE,
                             "LTFU (>180d)": CORAL}
                fig_inv = go.Figure()
                for lc in lifecycle_order:
                    sub = df_cc[df_cc["lifecycle"] == lc]
                    if not sub.empty:
                        fig_inv.add_trace(go.Bar(
                            name=lc, x=sub["payer"], y=sub["avg_invoice_size"],
                            marker_color=lc_colors.get(lc, GRAY),
                            text=[f"KES {v:,.0f}" if pd.notna(v) else "" for v in sub["avg_invoice_size"]],
                            textposition="outside",
                            hovertemplate=f"<b>%{{x}}</b><br>{lc}: KES %{{y:,.0f}}<extra></extra>",
                        ))
                fig_inv.update_layout(
                    barmode="group", height=300,
                    margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Avg Invoice (KES)", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
                )
                _pc(fig_inv)

            with c2:
                st.markdown("**Investigation Wait Time by Payer × Lifecycle (Hours)**")
                insured_df = df_cc[df_cc["payer"] == "Insurance / Corporate"]
                if not insured_df.empty and insured_df["avg_inv_wait_hrs"].notna().any():
                    fig_wt = go.Figure()
                    for lc in lifecycle_order:
                        sub = insured_df[insured_df["lifecycle"] == lc]
                        if not sub.empty:
                            fig_wt.add_trace(go.Bar(
                                name=lc, x=[lc], y=sub["avg_inv_wait_hrs"],
                                marker_color=lc_colors.get(lc, GRAY),
                                text=[f"{v:.1f}h" if pd.notna(v) else "" for v in sub["avg_inv_wait_hrs"]],
                                textposition="outside",
                            ))
                    fig_wt.update_layout(
                        barmode="group", height=300,
                        margin=dict(l=0, r=20, t=20, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(title="Avg Wait (Hours)", rangemode="tozero"),
                        showlegend=False,
                    )
                    _pc(fig_wt)
                else:
                    st.info("Investigation wait time data not available for insured patients.")

            # Narrative: cost correlation insight
            cash_df = df_cc[df_cc["payer"] == "Cash"].set_index("lifecycle")
            if "LTFU (>180d)" in cash_df.index and "Active (≤90d)" in cash_df.index:
                ltfu_inv = float(cash_df.loc["LTFU (>180d)", "avg_invoice_size"] or 0)
                active_inv = float(cash_df.loc["Active (≤90d)", "avg_invoice_size"] or 0)
                if ltfu_inv > 0 and active_inv > 0:
                    diff_pct = (ltfu_inv - active_inv) / active_inv * 100
                    _note(
                        f"Cash patients who hit 180-day LTFU had avg invoices of KES {ltfu_inv:,.0f} "
                        f"vs KES {active_inv:,.0f} for retained cash patients "
                        f"({'higher' if diff_pct > 0 else 'lower'} by {abs(diff_pct):.0f}%). "
                        + ("Higher bills appear to correlate with permanent dropout among cash-paying chronic patients."
                           if diff_pct > 20 else
                           "Invoice size alone does not appear to be the primary dropout driver for cash patients.")
                    )
    except Exception as e:
        st.warning(f"Cost dropout correlation: {e}")

    _gap(12)

    # ── H: PEAK HOUR LTFU — DID CHAOS BREAK INSURED LOYALTY? ────────────────
    _sh("I — Did Peak-Hour Visits Permanently Break Insured Patient Loyalty?", mt=8)
    _note(
        "For insured chronic patients who crossed 180-day LTFU, this checks whether their "
        "final recorded visit fell on a statistically peak day (top quartile of daily volume). "
        "A high % suggests that chaotic, high-volume days — with long queues and slow corporate "
        "clearance — may have been the breaking point."
    )
    try:
        df_ph = Q.load_ltfu_peak_hour_analysis(filters, run_query)
        if not df_ph.empty:
            row = df_ph.iloc[0]
            total_ltfu = int(row.get("total_ltfu_insured") or 0)
            pct_peak   = float(row.get("pct_on_peak_day") or 0)
            morning    = int(row.get("morning_rush_patients") or 0)
            mon_fri    = int(row.get("mon_fri_patients") or 0)
            avg_hour   = float(row.get("avg_final_visit_hour") or 0)

            c1, c2, c3, c4 = st.columns(4)
            with c1: _kpi("180d LTFU Insured", _n(total_ltfu), color=CORAL)
            with c2: _kpi("Last Visit on Peak Day", f"{pct_peak:.0f}%",
                          "of LTFU insured patients",
                          CORAL if pct_peak > 50 else ORANGE)
            with c3: _kpi("Morning Rush (7–10am)", _n(morning),
                          f"{morning/max(total_ltfu,1)*100:.0f}% of LTFU insured")
            with c4: _kpi("Mon or Fri Final Visit", _n(mon_fri),
                          f"avg final visit at {avg_hour:.0f}:00")

            if pct_peak > 50:
                _note(
                    f"{pct_peak:.0f}% of insured LTFU patients' last visits occurred on "
                    f"statistically high-volume days. This is above 50% — strongly suggesting "
                    f"that operational congestion (long waits, slow authorisations) during peak "
                    f"days contributed to these patients choosing not to return.",
                    w=True,
                )
            else:
                _note(
                    f"{pct_peak:.0f}% of insured LTFU patients' last visits were on peak days — "
                    f"below 50%, suggesting peak-day congestion alone is not the primary driver. "
                    f"Check dropout causes (Section B) for other signals."
                )
    except Exception as e:
        st.warning(f"Peak hour LTFU analysis: {e}")

    _gap(12)

    # ── J: INSURED FOLLOW-UP DURING SURGE MONTHS ─────────────────────────────
    _sh("J — Do Insured Patients Return Less During High-Volume Months?", mt=8)
    _note(
        "Each bar shows the average days between insured patient visits, overlaid with whether "
        "that month was a volume surge (z-score > 1). Wider gaps during surge months confirm "
        "that operational pressure is pushing scheduled follow-ups out."
    )
    try:
        df_sf = Q.load_insured_surge_followup(filters, run_query)
        if not df_sf.empty:
            df_sf["avg_days_to_next_visit"] = pd.to_numeric(df_sf["avg_days_to_next_visit"], errors="coerce")
            df_sf["is_surge_month"] = pd.to_numeric(df_sf["is_surge_month"], errors="coerce")

            fig_sf = go.Figure()
            colors = [CORAL if s == 1 else TEAL for s in df_sf["is_surge_month"]]
            fig_sf.add_trace(go.Bar(
                x=df_sf["visit_month"],
                y=df_sf["avg_days_to_next_visit"],
                marker_color=colors,
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Avg days to next visit: %{y:.0f}<br>"
                    "<extra></extra>"
                ),
            ))
            # Surge month annotations
            for _, row in df_sf[df_sf["is_surge_month"] == 1].iterrows():
                fig_sf.add_annotation(
                    x=row["visit_month"], y=float(row["avg_days_to_next_visit"] or 0) + 3,
                    text="SURGE", showarrow=False,
                    font=dict(size=8, color=CORAL),
                )
            fig_sf.update_layout(
                height=280, margin=dict(l=0, r=20, t=20, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Avg Days to Next Visit", rangemode="tozero"),
                xaxis=dict(title="Month"),
                showlegend=False,
            )
            _pc(fig_sf)

            surge_months = df_sf[df_sf["is_surge_month"] == 1]
            normal_months = df_sf[df_sf["is_surge_month"] == 0]
            if not surge_months.empty and not normal_months.empty:
                surge_avg = float(surge_months["avg_days_to_next_visit"].mean() or 0)
                normal_avg = float(normal_months["avg_days_to_next_visit"].mean() or 0)
                _note(
                    f"Average insured patient return gap: {surge_avg:.0f} days during surge months "
                    f"vs {normal_avg:.0f} days in normal months "
                    f"({'longer' if surge_avg > normal_avg else 'shorter'} by "
                    f"{abs(surge_avg - normal_avg):.0f} days). "
                    + ("Surge periods appear to delay or disrupt scheduled insured follow-ups — "
                       "investigate whether clinics are rebooking or patients are walking away."
                       if surge_avg > normal_avg * 1.15 else
                       "No significant pattern detected between surge months and insured follow-up frequency.")
                )
    except Exception as e:
        st.warning(f"Surge follow-up: {e}")

    _gap(12)

    # ── K: LOW-ENGAGEMENT REVENUE RISK ───────────────────────────────────────
    _sh("K — How Much Revenue at Risk Comes from Patients Who Only Visited 1–2 Times?", mt=8)
    _note(
        "Patients who visited only once or twice before crossing the 180-day mark may represent "
        "first-impression failures — a bad experience, confusing process, or unresolved clinical "
        "concern that prevented a second visit. These cases are structurally different from "
        "established chronic patients who gradually lapsed."
    )
    try:
        df_le = Q.load_low_engagement_revenue_risk(filters, run_query)
        if not df_le.empty:
            for _c in ("ltfu_patients", "avg_rev_per_visit", "revenue_at_risk"):
                df_le[_c] = pd.to_numeric(df_le[_c], errors="coerce")

            total_risk = float(df_le["revenue_at_risk"].sum() or 1)
            df_le["risk_share_pct"] = (df_le["revenue_at_risk"] / total_risk * 100).round(1)

            c1, c2 = st.columns(2)
            with c1:
                low_eng = df_le[df_le["engagement_tier"] == "1–2 Visits"]
                if not low_eng.empty:
                    _kpi(
                        "1–2 Visit LTFU Patients",
                        _n(low_eng["ltfu_patients"].iloc[0]),
                        f"{low_eng['risk_share_pct'].iloc[0]:.0f}% of total revenue at risk",
                        CORAL,
                    )
                    _gap(8)
                    _kpi(
                        "Revenue at Risk (1–2 Visits)",
                        _k(low_eng["revenue_at_risk"].iloc[0]),
                        "from low-engagement chronic LTFU",
                        CORAL,
                    )
                    _gap(8)
                    _note(
                        f"{low_eng['risk_share_pct'].iloc[0]:.0f}% of total 180-day LTFU revenue risk "
                        f"comes from patients who visited only 1–2 times. "
                        + ("A significant share — investigate whether intake quality, "
                           "cost transparency, or wait times at first visits deterred return."
                           if float(low_eng["risk_share_pct"].iloc[0] or 0) > 30 else
                           "Most revenue risk is from established patients who lapsed, not first-time visitors.")
                    )
            with c2:
                _pc(table_fig(
                    df_le[["engagement_tier", "ltfu_patients", "avg_rev_per_visit",
                            "revenue_at_risk", "risk_share_pct"]],
                    col_labels={
                        "engagement_tier": "Visit Tier",
                        "ltfu_patients": "LTFU Patients",
                        "avg_rev_per_visit": "Avg Rev/Visit",
                        "revenue_at_risk": "Revenue at Risk",
                        "risk_share_pct": "% of Total Risk",
                    },
                    fmt={"avg_rev_per_visit": "num", "revenue_at_risk": "num",
                         "risk_share_pct": "pct"},
                    height=200,
                ))
    except Exception as e:
        st.warning(f"Low engagement revenue risk: {e}")




# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DISEASE BURDEN
# ══════════════════════════════════════════════════════════════════════════════

def render_tab4_disease_burden(filters: dict, run_query):
    st_a, st_b, st_c, st_d, st_e = st.tabs([
        "Overview", "NCD & Chronic", "RMNCH",
        "Communicable & HIV", "Mental Health & Psychiatric",
    ])

    # ── OVERVIEW TAB ─────────────────────────────────────────────────────────
    with st_a:
        _sh("A — Disease Burden Overview")

        # KPI row — Undetected NCD replaced with financial leakage signal
        try:
            c1, c2, c3, c4, c5 = st.columns(5)
            df_kpi = Q.load_burden_kpis(filters, run_query)
            if not df_kpi.empty:
                row = df_kpi.iloc[0]
                with c1: _kpi("Diagnosed Visits", _n(row.get("total_diagnosed")))
                with c2: _kpi("Comorbidity Rate", _p(row.get("comorbidity_rate_pct")), color=ORANGE)
                with c3: _kpi("NCD Share", _p(row.get("ncd_share_pct")), color=AFYA_BLUE)
                with c4: _kpi("Communicable Share", _p(row.get("communicable_share_pct")), color=TEAL)
            try:
                df_leak = Q.load_ncd_leakage_kpi(filters, run_query)
                if not df_leak.empty:
                    leak_row = df_leak.iloc[0]
                    leak_kes = float(leak_row.get("estimated_leakage_kes") or 0)
                    undetected = int(leak_row.get("undetected_ncd_patients") or 0)
                    with c5:
                        _kpi(
                            "NCD Billing Leakage",
                            f"KES {leak_kes/1e6:.1f}M" if leak_kes >= 1e6 else f"KES {leak_kes:,.0f}",
                            f"{undetected:,} patients with elevated vitals & no NCD code",
                            CORAL,
                        )
            except Exception:
                with c5:
                    _kpi("Undetected NCD", _n(df_kpi.iloc[0].get("undetected_ncd")),
                         "elevated vitals, no NCD code", CORAL)
        except Exception as e:
            st.warning(f"A1: {e}")

        _gap(12)

        # Top 5 burden groups — actual visit counts over time
        _sh("Top 5 Disease Groups — Visit Growth Over Time", mt=8)
        _note("Actual visit counts (not normalised) for the five highest-volume disease groups. Steeper upward slopes indicate faster growth.")
        try:
            df_bt = Q.load_burden_trend(filters, run_query)
            if not df_bt.empty:
                df_bt["visit_count"] = pd.to_numeric(df_bt["visit_count"], errors="coerce")
                df_bt["visit_month"] = pd.to_datetime(df_bt["visit_month"])
                top5_groups = (
                    df_bt.groupby("burden_group")["visit_count"]
                    .sum().nlargest(5).index.tolist()
                )
                df_top5 = df_bt[df_bt["burden_group"].isin(top5_groups)].copy()
                line_colors = [AFYA_BLUE, CORAL, TEAL, PURPLE, ORANGE]
                fig_top5 = go.Figure()
                for i, grp in enumerate(top5_groups):
                    sub = df_top5[df_top5["burden_group"] == grp].sort_values("visit_month")
                    fig_top5.add_trace(go.Scatter(
                        x=sub["visit_month"], y=sub["visit_count"],
                        name=grp, mode="lines+markers",
                        line=dict(color=line_colors[i % len(line_colors)], width=2),
                        marker=dict(size=5),
                        hovertemplate=f"<b>{grp}</b><br>%{{x|%b %Y}}: %{{y:,}} visits<extra></extra>",
                    ))
                fig_top5.update_layout(
                    height=340, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Visits", rangemode="tozero"),
                    xaxis=dict(title=""),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                _pc(fig_top5)
        except Exception as e:
            st.warning(f"A2: {e}")

        _gap(12)

        # Top 10 diagnoses with IP/OP split
        _sh("Top Diagnoses — Inpatient vs Outpatient Split", mt=8)
        _note("A surge in a communicable outpatient condition is low-margin. A surge in a chronic cardiovascular condition means long-term retention value.")
        try:
            df_td = Q.load_top_diagnoses_ip_op(filters, run_query)
            if not df_td.empty:
                for _c in ("total_visits", "inpatient_visits", "outpatient_visits", "ip_pct"):
                    df_td[_c] = pd.to_numeric(df_td[_c], errors="coerce")

                fig_td = go.Figure()
                fig_td.add_trace(go.Bar(
                    name="Outpatient",
                    x=df_td["outpatient_visits"],
                    y=df_td["burden_group"],
                    orientation="h",
                    marker_color=TEAL,
                    hovertemplate="<b>%{y}</b><br>OP: %{x:,}<extra></extra>",
                ))
                fig_td.add_trace(go.Bar(
                    name="Inpatient",
                    x=df_td["inpatient_visits"],
                    y=df_td["burden_group"],
                    orientation="h",
                    marker_color=CORAL,
                    hovertemplate="<b>%{y}</b><br>IP: %{x:,}<extra></extra>",
                ))
                fig_td.update_layout(
                    barmode="stack", height=420,
                    margin=dict(l=0, r=60, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Visits"),
                    yaxis=dict(title=""),
                    legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                )
                _pc(fig_td)
            else:
                df_td2 = Q.load_top_diagnoses(filters, run_query)
                if not df_td2.empty:
                    _pc(hbar_chart(df_td2, x="visit_count", y="disease_group",
                                   color=AFYA_BLUE, x_label="Visits", height=320))
        except Exception as e:
            st.warning(f"A3: {e}")

        _gap(12)

        # Emerging diagnoses
        _sh("Emerging Mid-Tier Diagnoses — 90-Day Growth Rate", mt=8)
        _note("Conditions outside the top 5 showing the highest month-over-month percentage growth. These are the next wave before they dominate the top 10.")
        try:
            df_em = Q.load_emerging_diagnoses_90d(filters, run_query)
            if not df_em.empty:
                for _c in ("recent_90d_visits", "prior_90d_visits", "mom_growth_pct", "inpatient_pct"):
                    df_em[_c] = pd.to_numeric(df_em[_c], errors="coerce")

                c1, c2 = st.columns([1.3, 0.7])
                with c1:
                    fig_em = go.Figure(go.Bar(
                        x=df_em["mom_growth_pct"].head(12),
                        y=df_em["condition"].head(12),
                        orientation="h",
                        marker_color=[CORAL if v > 0 else TEAL for v in df_em["mom_growth_pct"].head(12)],
                        text=[f"+{v:.0f}%" if v > 0 else f"{v:.0f}%" for v in df_em["mom_growth_pct"].head(12)],
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>MoM Growth: %{x:.1f}%<extra></extra>",
                    ))
                    fig_em.update_layout(
                        height=360, margin=dict(l=0, r=60, t=20, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="% Growth (Recent 90d vs Prior 90d)"),
                        yaxis=dict(title=""),
                        showlegend=False,
                    )
                    _pc(fig_em)
                with c2:
                    _pc(table_fig(
                        df_em[["condition", "recent_90d_visits", "prior_90d_visits",
                               "mom_growth_pct", "inpatient_pct"]].head(12),
                        col_labels={
                            "condition": "Condition", "recent_90d_visits": "Recent 90d",
                            "prior_90d_visits": "Prior 90d", "mom_growth_pct": "MoM %",
                            "inpatient_pct": "IP %",
                        },
                        fmt={"mom_growth_pct": "pct", "inpatient_pct": "pct"},
                        height=380,
                    ))
        except Exception as e:
            st.warning(f"A5: {e}")

        _gap(12)

        # Disease Intelligence Matrix
        _sh("Disease Intelligence Matrix — 90-Day Operational View", mt=8)
        _note("Combines volume trend, demographics, visit type split, payer mix, and risk signal into one scannable operational table.")
        try:
            df_dim = Q.load_disease_intelligence_matrix(filters, run_query)
            if not df_dim.empty:
                for _c in ("total_visits", "ip_pct", "op_pct", "trend_pct"):
                    if _c in df_dim.columns:
                        df_dim[_c] = pd.to_numeric(df_dim[_c], errors="coerce")

                def _trend_arrow(v):
                    if pd.isna(v): return "→ Stable"
                    return f"📈 +{v:.0f}%" if v > 10 else f"📉 {v:.0f}%" if v < -10 else f"→ {v:+.0f}%"

                df_dim["trend_fmt"] = df_dim.get("trend_pct", pd.Series([0]*len(df_dim))).apply(_trend_arrow)
                df_dim["visit_split"] = df_dim.apply(
                    lambda r: f"🔄 {r.get('op_pct', 0):.0f}% OP  🛌 {r.get('ip_pct', 0):.0f}% IP", axis=1
                )

                rows_html = ""
                for _, r in df_dim.iterrows():
                    trend_raw = float(r.get("trend_pct") or 0) if "trend_pct" in r else 0
                    bg = "#FFF5F5" if trend_raw > 15 else "#F0FFF4" if trend_raw < -5 else "white"
                    rows_html += (
                        f'<tr style="background:{bg}">'
                        f'<td style="padding:6px 8px;font-weight:600;font-size:12px">{r.get("condition","")}</td>'
                        f'<td style="padding:6px 8px;font-size:11px">{r.get("trend_fmt","→")}</td>'
                        f'<td style="padding:6px 8px;font-size:11px">{r.get("primary_age_group","")}'
                        f' · {r.get("primary_gender","")}</td>'
                        f'<td style="padding:6px 8px;font-size:11px">{r.get("visit_split","")}</td>'
                        f'<td style="padding:6px 8px;font-size:11px">{r.get("primary_payer","")}</td>'
                        f'<td style="padding:6px 8px;font-size:11px;text-align:right">{int(r.get("total_visits",0)):,}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;'
                    'font-family:sans-serif">'
                    '<thead><tr style="background:#0072CE;color:white">'
                    '<th style="padding:8px;text-align:left">Condition</th>'
                    '<th style="padding:8px;text-align:left">90d Trend</th>'
                    '<th style="padding:8px;text-align:left">Primary Demographic</th>'
                    '<th style="padding:8px;text-align:left">Visit Type Split</th>'
                    '<th style="padding:8px;text-align:left">Primary Payer</th>'
                    '<th style="padding:8px;text-align:right">90d Visits</th>'
                    '</tr></thead><tbody>' + rows_html + '</tbody></table></div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(f"A6: {e}")

    # ── NCD & CHRONIC TAB ────────────────────────────────────────────────────
    with st_b:
        _sh("B — NCD & Chronic Disease")

        # ── B1: KPI ROW ──────────────────────────────────────────────────────
        try:
            df_bkpi = Q.load_ncd_kpis(filters, run_query)
            if not df_bkpi.empty:
                row = df_bkpi.iloc[0]
                comorb_pct = float(row.get("comorbidity_rate_pct") or 0)
                htn_pct    = float(row.get("controlled_htn_pct") or 0)
                c1, c2, c3, c4 = st.columns(4)
                with c1: _kpi("NCD Patients",    _n(row.get("ncd_patients")), color=AFYA_BLUE)
                with c2: _kpi("Comorbidity Rate", _p(comorb_pct),
                               "patients with 2+ NCDs",
                               ORANGE if comorb_pct >= 20 else AFYA_BLUE)
                with c3: _kpi("Controlled HTN",   _p(htn_pct),
                               "avg BP <140/90 — benchmark 60%",
                               TEAL if htn_pct >= 60 else CORAL)
                with c4: _kpi("Undetected NCD",  _n(row.get("undetected_ncd_patients")),
                               "elevated vitals, no NCD code", CORAL)
        except Exception as e:
            st.warning(f"B1: {e}")

        _gap(12)

        # ── B2: TOP NCDs RANKED + GENDER ─────────────────────────────────────
        _sh("Top NCDs — Patient Volume by Condition & Gender", mt=8)
        _note("Diabetes and Endocrine & Metabolic are consolidated — they overlap heavily in ICD10 coding and represent the same patient risk profile.")
        try:
            df_rk = Q.load_ncd_ranked_with_gender(filters, run_query)
            if not df_rk.empty:
                df_rk["patient_count"] = pd.to_numeric(df_rk["patient_count"], errors="coerce")
                cond_totals = (df_rk.groupby("ncd_group")["patient_count"]
                               .sum().sort_values(ascending=False))
                top_conds = cond_totals.head(10).index.tolist()
                df_rk_top = df_rk[df_rk["ncd_group"].isin(top_conds)].copy()
                c1, c2 = st.columns([1.4, 0.6])
                with c1:
                    st.markdown("**Patient count by NCD group and gender:**")
                    pivot_g = (df_rk_top.pivot_table(
                        index="ncd_group", columns="gender",
                        values="patient_count", aggfunc="sum", fill_value=0
                    ).reindex(top_conds))
                    color_map_g = {"FEMALE": PURPLE, "MALE": AFYA_BLUE, "F": PURPLE,
                                   "M": AFYA_BLUE, "Unknown": GRAY}
                    fig_rk = go.Figure()
                    for g in pivot_g.columns.tolist():
                        fig_rk.add_trace(go.Bar(
                            x=pivot_g[g].values, y=pivot_g.index.tolist(),
                            name=g, orientation="h",
                            marker_color=color_map_g.get(g, TEAL),
                            hovertemplate=f"<b>%{{y}}</b><br>{g}: %{{x:,}}<extra></extra>",
                        ))
                    fig_rk.update_layout(
                        barmode="stack", height=340,
                        margin=dict(l=0, r=20, t=10, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Patients", rangemode="tozero"),
                        yaxis=dict(title="", autorange="reversed"),
                        legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                    )
                    _pc(fig_rk)
                with c2:
                    st.markdown("**Totals:**")
                    _pc(table_fig(
                        cond_totals.head(10).reset_index().rename(columns={
                            "ncd_group": "Condition", "patient_count": "Patients"
                        }),
                        col_labels={}, height=340,
                    ))
        except Exception as e:
            st.warning(f"B2: {e}")

        _gap(12)

        # ── B3: NCD COMPLEXITY ────────────────────────────────────────────────
        _sh("NCD Complexity — Simple vs Multi-Morbidity Cases", mt=8)
        _note("Patients with 2+ NCDs are 3-4x more expensive to manage. The share of complex cases drives case management staffing and chronic care protocol design.")
        try:
            df_cx = Q.load_ncd_complexity_distribution(filters, run_query)
            if not df_cx.empty:
                df_cx["patient_count"]      = pd.to_numeric(df_cx["patient_count"],      errors="coerce")
                df_cx["pct_of_ncd_patients"] = pd.to_numeric(df_cx["pct_of_ncd_patients"], errors="coerce")
                c1, c2 = st.columns(2)
                with c1:
                    _pc(donut(
                        labels=df_cx["ncd_complexity"].tolist(),
                        values=df_cx["patient_count"].tolist(),
                        color_map={
                            "1 NCD": TEAL, "2 NCDs": ORANGE,
                            "3 NCDs": CORAL, "4+ NCDs (Complex)": PURPLE,
                        },
                        height=280,
                    ))
                with c2:
                    st.dataframe(
                        df_cx[["ncd_complexity", "patient_count", "pct_of_ncd_patients"]].rename(
                            columns={"ncd_complexity": "Complexity",
                                     "patient_count": "Patients",
                                     "pct_of_ncd_patients": "% of NCD Patients"}
                        ),
                        use_container_width=True, hide_index=True, height=220,
                        column_config={
                            "% of NCD Patients": st.column_config.NumberColumn(format="%.1f%%")
                        },
                    )
                    complex_pct = float(
                        df_cx.loc[df_cx["ncd_complexity"] != "1 NCD",
                                  "pct_of_ncd_patients"].sum()
                    )
                    _note(
                        f"{complex_pct:.0f}% of NCD patients carry 2+ conditions. "
                        "These patients need integrated management protocols.",
                        w=(complex_pct > 30),
                    )
        except Exception as e:
            st.warning(f"B3: {e}")

        _gap(12)

        # ── B4: NCD BY AGE GROUP HEATMAP ─────────────────────────────────────
        _sh("NCD Distribution by Age Group", mt=8)
        _note("Dark cells = high patient concentration. Use this to design age-targeted care pathways.")
        try:
            df_hm = Q.load_ncd_age_heatmap(filters, run_query)
            if not df_hm.empty:
                df_hm["patient_count"] = pd.to_numeric(df_hm["patient_count"], errors="coerce")
                pivot_hm = df_hm.pivot_table(
                    index="age_group", columns="chronic_condition",
                    values="patient_count", aggfunc="sum", fill_value=0
                )
                age_order = ["Under 18", "18-34", "35-49", "50-64", "65+"]
                pivot_hm = pivot_hm.reindex([a for a in age_order if a in pivot_hm.index])
                fig_hm = go.Figure(go.Heatmap(
                    z=pivot_hm.values,
                    x=pivot_hm.columns.tolist(),
                    y=pivot_hm.index.tolist(),
                    colorscale=[[0, "#EBF3FB"], [0.5, "#0072CE"], [1, "#003F88"]],
                    hovertemplate="<b>%{y} — %{x}</b><br>Patients: %{z:,}<extra></extra>",
                    text=pivot_hm.values, texttemplate="%{text}",
                    textfont=dict(size=10, color="white"),
                ))
                fig_hm.update_layout(
                    height=300, margin=dict(l=0, r=20, t=20, b=60),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(tickangle=-30, title=""),
                    yaxis=dict(title="Age Group"),
                )
                _pc(fig_hm)
        except Exception as e:
            st.warning(f"B4: {e}")

        _gap(12)

        # ── B5: COMORBIDITY PAIRS ─────────────────────────────────────────────
        _sh("Comorbidity Pipeline — Top Condition Pairs & Progression Speed", mt=8)
        _note("How quickly do patients develop a second NCD? Diabetes & Endocrine/Metabolic consolidated into one group.")
        try:
            df_cp = Q.load_chronic_comorbidity_pairs(filters, run_query)
            if not df_cp.empty:
                df_cp["avg_days_between_diagnoses"] = pd.to_numeric(
                    df_cp["avg_days_between_diagnoses"], errors="coerce")
                df_cp["patient_count"] = pd.to_numeric(df_cp["patient_count"], errors="coerce")
                c1, c2 = st.columns([1.3, 0.7])
                with c1:
                    _pc(hbar_chart(
                        df_cp.head(10), x="patient_count", y="condition_pair",
                        x_label="Patients", color=AFYA_BLUE, height=320, show_text=True,
                    ))
                with c2:
                    _pc(table_fig(
                        df_cp[["condition_pair", "patient_count",
                               "avg_days_between_diagnoses"]].head(10),
                        col_labels={"condition_pair": "Pair",
                                    "patient_count": "Patients",
                                    "avg_days_between_diagnoses": "Avg Days to 2nd Dx"},
                        height=320,
                    ))
                fastest = df_cp.dropna(subset=["avg_days_between_diagnoses"])
                if not fastest.empty:
                    f_row = fastest.loc[fastest["avg_days_between_diagnoses"].idxmin()]
                    _note(
                        f"Fastest progression: {f_row['condition_pair']} — avg "
                        f"{int(f_row['avg_days_between_diagnoses'])} days to second diagnosis. "
                        "This pair needs a co-management protocol."
                    )
        except Exception as e:
            st.warning(f"B5: {e}")

        _gap(12)

        # ── B6: HTN CONTROLLED vs UNCONTROLLED ───────────────────────────────
        _sh("HTN: Controlled vs Uncontrolled — Annual Visits & Doctors Seen", mt=8)
        _note("Controlled patients follow a consistent single-doctor cadence. Uncontrolled patients exhibit doctor-shopping and erratic visit gaps.")
        try:
            df_sc = Q.load_htn_scatter_data(filters, run_query)
            if not df_sc.empty:
                for _c in ("avg_annual_visits", "unique_doctors"):
                    df_sc[_c] = pd.to_numeric(df_sc[_c], errors="coerce")
                controlled   = df_sc[df_sc["htn_status"] == "Controlled"]
                uncontrolled = df_sc[df_sc["htn_status"] == "Uncontrolled"]
                fig_sc = go.Figure()
                for grp, color, name in [
                    (controlled,   TEAL,  "Controlled (BP <140/90)"),
                    (uncontrolled, CORAL, "Uncontrolled (BP >=140/90)"),
                ]:
                    if not grp.empty:
                        fig_sc.add_trace(go.Scatter(
                            x=grp["avg_annual_visits"], y=grp["unique_doctors"],
                            mode="markers", name=name,
                            marker=dict(color=color, size=8, opacity=0.7),
                            hovertemplate=(f"<b>{name}</b><br>Annual visits: %{{x:.1f}}<br>"
                                           "Unique doctors: %{y}<extra></extra>"),
                        ))
                fig_sc.update_layout(
                    height=300, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Avg Annual Visits", rangemode="tozero"),
                    yaxis=dict(title="Unique Doctors Seen", rangemode="tozero"),
                    legend=dict(orientation="h", y=1.08, xanchor="right", x=1),
                )
                _pc(fig_sc)
                if not controlled.empty and not uncontrolled.empty:
                    ctrl_docs   = float(controlled["unique_doctors"].median() or 0)
                    unctrl_docs = float(uncontrolled["unique_doctors"].median() or 0)
                    _note(
                        f"Median unique doctors: {ctrl_docs:.0f} (Controlled) vs "
                        f"{unctrl_docs:.0f} (Uncontrolled). "
                        + ("Uncontrolled patients see significantly more doctors — "
                           "care fragmentation is likely driving poor BP management."
                           if unctrl_docs > ctrl_docs * 1.3
                           else "Doctor frequency is similar; investigate medication adherence.")
                    )
            else:
                df_htn = Q.load_htn_controlled(filters, run_query)
                if not df_htn.empty:
                    c1, c2 = st.columns(2)
                    with c1:
                        _pc(donut(labels=df_htn["htn_status"].tolist(),
                                  values=df_htn["patient_count"].tolist(),
                                  color_map={"Controlled": TEAL, "Uncontrolled": CORAL,
                                             "No BP Recorded": GRAY},
                                  height=260))
                    with c2:
                        _pc(table_fig(df_htn,
                                      col_labels={"htn_status": "Status",
                                                  "patient_count": "Patients",
                                                  "avg_systolic": "Avg Systolic"},
                                      height=180))
        except Exception as e:
            st.warning(f"B6: {e}")

        _gap(12)

        # ── B7: UNCONTROLLED HTN PROFILE ──────────────────────────────────────
        _sh("Uncontrolled HTN — Who Are They & Why?", mt=8)
        _note("Age, payer type, comorbidity burden, medication use, and investigation rate for HTN patients. Reveals whether uncontrolled status is linked to demographics, multi-morbidity, or medication gaps.")
        try:
            df_hp = Q.load_htn_uncontrolled_profile(filters, run_query)
            if not df_hp.empty:
                for _c in ("patient_count", "avg_investigations", "avg_visits", "avg_systolic"):
                    if _c in df_hp.columns:
                        df_hp[_c] = pd.to_numeric(df_hp[_c], errors="coerce")

                uc = df_hp[df_hp["htn_status"] == "Uncontrolled"]
                if not uc.empty:
                    total_uc = int(uc["patient_count"].sum())
                    on_rx_n  = df_hp[
                        (df_hp["htn_status"] == "Uncontrolled") &
                        (pd.to_numeric(df_hp["on_antihypertensive"], errors="coerce") == 1)
                    ]["patient_count"].sum()
                    on_rx_pct = on_rx_n / max(total_uc, 1) * 100
                    avg_inv = ((uc["avg_investigations"] * uc["patient_count"]).sum()
                               / max(total_uc, 1))
                    k1, k2, k3 = st.columns(3)
                    with k1: _kpi("Uncontrolled HTN",       _n(total_uc), color=CORAL)
                    with k2: _kpi("On Antihypertensive Rx", f"{on_rx_pct:.0f}%",
                                  "prescribed despite uncontrolled BP",
                                  ORANGE if on_rx_pct > 40 else CORAL)
                    with k3: _kpi("Avg Investigations",      f"{avg_inv:.1f}",
                                  "investigations per patient", AFYA_BLUE)
                    _gap(8)

                c1, c2 = st.columns(2)
                with c1:
                    age_grp = (df_hp.groupby(["htn_status", "age_group"])["patient_count"]
                               .sum().reset_index())
                    fig_age = go.Figure()
                    for status, color in [("Controlled", TEAL), ("Uncontrolled", CORAL),
                                          ("No BP Recorded", GRAY)]:
                        sub = age_grp[age_grp["htn_status"] == status]
                        if not sub.empty:
                            fig_age.add_trace(go.Bar(
                                x=sub["age_group"], y=sub["patient_count"],
                                name=status, marker_color=color,
                                hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{y:,}}<extra></extra>",
                            ))
                    fig_age.update_layout(
                        barmode="group", height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Age Group"),
                        yaxis=dict(title="Patients", rangemode="tozero"),
                        legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
                        title=dict(text="By Age Group", font=dict(size=12)),
                    )
                    _pc(fig_age)

                with c2:
                    pay_grp = (df_hp.groupby(["htn_status", "payer"])["patient_count"]
                               .sum().reset_index())
                    fig_pay = go.Figure()
                    for status, color in [("Controlled", TEAL), ("Uncontrolled", CORAL),
                                          ("No BP Recorded", GRAY)]:
                        sub = pay_grp[pay_grp["htn_status"] == status]
                        if not sub.empty:
                            fig_pay.add_trace(go.Bar(
                                x=sub["payer"], y=sub["patient_count"],
                                name=status, marker_color=color,
                                hovertemplate=f"<b>%{{x}}</b><br>{status}: %{{y:,}}<extra></extra>",
                            ))
                    fig_pay.update_layout(
                        barmode="group", height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="Payer"),
                        yaxis=dict(title="Patients", rangemode="tozero"),
                        legend=dict(orientation="h", y=1.1, xanchor="right", x=1),
                        title=dict(text="By Payer Type", font=dict(size=12)),
                    )
                    _pc(fig_pay)

                _gap(8)
                comorb_grp = (
                    df_hp.groupby(["htn_status", "comorbidity_group"])
                    .apply(lambda d: pd.Series({
                        "patients": d["patient_count"].sum(),
                        "avg_inv":  ((d["avg_investigations"] * d["patient_count"]).sum()
                                    / max(d["patient_count"].sum(), 1)),
                        "on_rx_pct": (d.loc[
                            pd.to_numeric(d["on_antihypertensive"], errors="coerce") == 1,
                            "patient_count"
                        ].sum() / max(d["patient_count"].sum(), 1) * 100),
                    }), include_groups=False)
                    .reset_index()
                )
                st.dataframe(
                    comorb_grp.rename(columns={
                        "htn_status": "HTN Status", "comorbidity_group": "Comorbidity",
                        "patients": "Patients", "avg_inv": "Avg Investigations",
                        "on_rx_pct": "On Antihypertensive %",
                    }),
                    use_container_width=True, hide_index=True,
                    height=min(400, len(comorb_grp) * 35 + 50),
                    column_config={
                        "Avg Investigations":    st.column_config.NumberColumn(format="%.1f"),
                        "On Antihypertensive %": st.column_config.NumberColumn(format="%.0f%%"),
                    },
                )
        except Exception as e:
            st.warning(f"B7: {e}")

        _gap(12)

        # ── B8: PRESCRIPTION WITHOUT CLINICAL ASSESSMENT ──────────────────────
        _sh("Prescription Without Clinical Assessment — Documentation Gap", mt=8)
        _note(
            "Chronic visits where a prescription was issued but NO vitals were recorded AND "
            "NO clinical note exists. May indicate a documentation gap or a true pharmacy-only "
            "refill without clinical review. Either way it is a governance risk."
        )
        try:
            df_ph = Q.load_chronic_pharmacy_only(filters, run_query)
            if not df_ph.empty:
                for _c in ("patient_count", "pharmacy_only_pct", "avg_annual_revenue"):
                    if _c in df_ph.columns:
                        df_ph[_c] = pd.to_numeric(df_ph[_c], errors="coerce")
                if "pharmacy_only_visits" in df_ph.columns:
                    total_gap = int(pd.to_numeric(
                        df_ph["pharmacy_only_visits"], errors="coerce"
                    ).sum())
                    if total_gap > 0:
                        _note(
                            f"{total_gap:,} visits have a prescription but no recorded vitals or "
                            "clinical note. Investigate whether these are genuine assessments with "
                            "missing documentation or true unreviewed refills.",
                            w=True,
                        )
                _pc(table_fig(
                    df_ph[["payer", "condition", "patient_count",
                            "pharmacy_only_pct", "avg_annual_revenue"]].head(20),
                    col_labels={
                        "payer": "Payer", "condition": "Condition",
                        "patient_count": "Patients",
                        "pharmacy_only_pct": "Gap Visit %",
                        "avg_annual_revenue": "Avg Annual Rev (KES)",
                    },
                    fmt={"pharmacy_only_pct": "pct", "avg_annual_revenue": "num"},
                    height=380,
                ))
        except Exception as e:
            st.warning(f"B8: {e}")

        _gap(12)

        # ── B9: CHRONIC CARE QUALITY & REVENUE MATRIX — DATA DRIVEN ──────────
        _sh("Chronic Care Quality & Revenue Matrix", mt=8)
        _note("Data-driven view of actual top NCD conditions. All metrics from live data. Trend = % change last 6 months vs prior 6 months.")
        try:
            df_qmx = Q.load_chronic_care_matrix(filters, run_query)
            if not df_qmx.empty:
                for _c in ("patient_count", "trend_pct", "ip_rate_pct",
                           "avg_visits_per_patient", "investigations_per_visit",
                           "avg_revenue_per_patient", "controlled_pct"):
                    if _c in df_qmx.columns:
                        df_qmx[_c] = pd.to_numeric(df_qmx[_c], errors="coerce")

                def _trend_icon(v):
                    if pd.isna(v): return "—"
                    return f"↑ {abs(v):.0f}%" if v > 5 else f"↓ {abs(v):.0f}%" if v < -5 else f"→ {v:.0f}%"

                disp_qmx = pd.DataFrame({
                    "Condition":       df_qmx["condition"],
                    "Patients":        df_qmx["patient_count"].fillna(0).astype(int),
                    "6-Mo Trend":      df_qmx["trend_pct"].apply(_trend_icon),
                    "IP Rate %":       df_qmx["ip_rate_pct"].fillna(0).round(1),
                    "Top Payer":       df_qmx.get("top_payer",
                                           pd.Series(["-"] * len(df_qmx))).fillna("-"),
                    "Visits/Patient":  df_qmx["avg_visits_per_patient"].fillna(0).round(1),
                    "Inv/Visit":       df_qmx["investigations_per_visit"].fillna(0).round(2),
                    "Avg Rev/Patient": df_qmx["avg_revenue_per_patient"].fillna(0).round(0),
                    "Controlled %":    df_qmx["controlled_pct"].apply(
                                           lambda v: f"{v:.0f}%" if pd.notna(v) else "-"),
                })
                st.dataframe(
                    disp_qmx, use_container_width=True, hide_index=True,
                    height=min(560, len(disp_qmx) * 38 + 50),
                    column_config={
                        "IP Rate %":      st.column_config.NumberColumn(format="%.1f%%"),
                        "Visits/Patient": st.column_config.NumberColumn(format="%.1f"),
                        "Inv/Visit":      st.column_config.NumberColumn(format="%.2f"),
                        "Avg Rev/Patient": st.column_config.NumberColumn(format="KES %,.0f"),
                    },
                )
        except Exception as e:
            st.warning(f"B9: {e}")

        _gap(12)

        # ── B10: UNDETECTED NCD RISK PATIENTS ────────────────────────────────
        _sh("Undetected NCD Risk — Elevated Vitals, No Chronic Diagnosis", mt=8)
        _note(
            "Patients with systolic BP >= 140 or blood sugar >= 10 on at least 2 separate visits, "
            "with no NCD diagnosis code. Highest clinical and billing risk."
        )
        try:
            df_ev = Q.load_elevated_vitals_no_ncd_patients(filters, run_query)
            if df_ev.empty:
                st.success("No undetected NCD risk patients found in this period.")
            else:
                for _c in ("visit_count", "latest_systolic",
                           "latest_blood_sugar", "days_since_last_visit"):
                    if _c in df_ev.columns:
                        df_ev[_c] = pd.to_numeric(df_ev[_c], errors="coerce")
                _note(
                    f"{len(df_ev):,} patients flagged. Assign to a chronic disease nurse for "
                    "screening within 30 days.",
                    w=True,
                )
                _pc(table_fig(
                    df_ev[["patient", "visit_count", "latest_systolic",
                           "latest_blood_sugar", "days_since_last_visit", "payer"]].head(40),
                    col_labels={
                        "patient": "Patient", "visit_count": "Flagged Visits",
                        "latest_systolic": "Latest Systolic",
                        "latest_blood_sugar": "Latest Blood Sugar",
                        "days_since_last_visit": "Days Since Last Visit",
                        "payer": "Payer",
                    },
                    height=min(520, len(df_ev.head(40)) * 30 + 60),
                ))
        except Exception as e:
            st.warning(f"B10: {e}")


    # ── RMNCH TAB ─────────────────────────────────────────────────────────────
    with st_c:
        _sh("C — RMNCH: Maternal, Child & Reproductive Health")

        # ── C0: Facility Profile Overview ─────────────────────────────────────
        _sh("Facility Profile — What Does This Hospital Do?", mt=8)
        _note("How visits are distributed across maternal care categories. Reveals whether the facility is primarily delivery-focused, ANC-focused, or balanced.")
        try:
            df_fac = Q.load_anc_vs_delivery_pnc(filters, run_query)
            if not df_fac.empty:
                total_visits = int(df_fac["visit_count"].sum() or 1)
                c1, c2, c3, c4 = st.columns(4)
                for col, cat, color in [
                    (c1, "Antenatal Care (ANC)", AFYA_BLUE),
                    (c2, "Delivery",              PURPLE),
                    (c3, "Postnatal Care (PNC)",  TEAL),
                    (c4, "Family Planning",        ORANGE),
                ]:
                    row_f = df_fac[df_fac["care_category"] == cat]
                    v = int(row_f["visit_count"].sum()) if not row_f.empty else 0
                    pct = v / total_visits * 100
                    with col:
                        _kpi(cat, _n(v), f"{pct:.1f}% of RMNCH visits", color)
                _gap(8)
                c1, c2 = st.columns(2)
                with c1:
                    _pc(bar_chart(df_fac, x="care_category", y="visit_count",
                                  color=PURPLE, y_label="Visits", height=260, show_text=True))
                with c2:
                    _pc(donut(labels=df_fac["care_category"].tolist(),
                              values=df_fac["visit_count"].tolist(),
                              height=260))
        except Exception as e:
            st.warning(f"Facility profile: {e}")

        _gap(12)

        # ── C1: ANC Retention Curve ────────────────────────────────────────────
        _sh("ANC Pathway — Retention Curve", mt=8)
        _note("How many women are retained through each ANC visit stage. Steep drops indicate the exact stage where patients defect.")
        try:
            df_anc = Q.load_anc_funnel(filters, run_query)
            if not df_anc.empty:
                row = df_anc.iloc[0]
                anc_vals = [int(row.get("anc1") or 0), int(row.get("anc2") or 0),
                            int(row.get("anc3") or 0), int(row.get("anc4") or 0)]
                stages = ["ANC 1", "ANC 2", "ANC 3", "ANC 4"]

                c1, c2 = st.columns([1.3, 0.7])
                with c1:
                    retention_pcts = [100.0] + [
                        anc_vals[i] / anc_vals[0] * 100 if anc_vals[0] > 0 else 0
                        for i in range(1, 4)
                    ]
                    drop_pcts = [
                        (anc_vals[i-1] - anc_vals[i]) / anc_vals[i-1] * 100
                        if anc_vals[i-1] > 0 else 0
                        for i in range(1, 4)
                    ]
                    fig_anc = go.Figure()
                    fig_anc.add_trace(go.Scatter(
                        x=stages, y=retention_pcts, mode="lines+markers+text",
                        line=dict(color=AFYA_BLUE, width=3),
                        marker=dict(size=12, color=AFYA_BLUE),
                        text=[f"{v:.0f}%" for v in retention_pcts],
                        textposition="top center",
                        fill="tozeroy", fillcolor="rgba(0,114,206,0.13)",
                        hovertemplate="<b>%{x}</b><br>Retention: %{y:.1f}%<extra></extra>",
                    ))
                    fig_anc.update_layout(
                        height=280, margin=dict(l=0, r=20, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(title="% Retained from ANC1", range=[0, 120]),
                        xaxis=dict(title=""), showlegend=False,
                    )
                    _pc(fig_anc)
                with c2:
                    worst_drop_idx = drop_pcts.index(max(drop_pcts)) if drop_pcts else 0
                    _kpi("ANC4 Completion", _p(row.get("anc4_completion_pct")),
                         "completing all 4 visits",
                         TEAL if float(row.get("anc4_completion_pct") or 0) >= 50 else CORAL)
                    _gap(8)
                    _kpi("Biggest Drop-off",
                         f"ANC{worst_drop_idx+1} → ANC{worst_drop_idx+2}",
                         f"{max(drop_pcts):.0f}% patient loss at this stage", CORAL)
        except Exception as e:
            st.warning(f"ANC funnel: {e}")

        _gap(12)

        # ── C2: ANC Dropout by Payer ───────────────────────────────────────────
        _sh("Why Is ANC Completion Low? — Dropout by Payer", mt=8)
        _note("ANC4 completion rates split by how patients pay. Cash patients typically drop out earlier due to unexpected billing costs at each visit.")
        try:
            df_pay = Q.load_anc_dropout_by_payer(filters, run_query)
            if not df_pay.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**ANC4 Completion Rate by Payer**")
                    _pc(bar_chart(df_pay, x="payer_type", y="anc4_completion_pct",
                                  color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                             "Insurance / Corporate": AFYA_BLUE},
                                  y_label="ANC4 Completion %", y_format="pct",
                                  height=260, show_text=True))
                with c2:
                    st.markdown("**Funnel — Patients Reaching Each Stage**")
                    fig_pay = go.Figure()
                    colors = {"Cash": CORAL, "NHIF / SHA": TEAL, "Insurance / Corporate": AFYA_BLUE}
                    for _, pr in df_pay.iterrows():
                        clr = colors.get(pr["payer_type"], AFYA_BLUE)
                        fig_pay.add_trace(go.Bar(
                            name=pr["payer_type"],
                            x=["ANC1", "ANC2", "ANC3", "ANC4"],
                            y=[pr["total_anc1_patients"], pr["reached_anc2"],
                               pr["reached_anc3"], pr["reached_anc4"]],
                            marker_color=clr,
                        ))
                    fig_pay.update_layout(
                        barmode="group", height=260,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        yaxis_title="Patients",
                    )
                    _pc(fig_pay)
        except Exception as e:
            st.warning(f"ANC dropout by payer: {e}")

        _gap(12)

        # ── C3: Same vs Different Patients at Each Stage ───────────────────────
        _sh("Are the Same Patients Coming Back for ANC?", mt=8)
        _note("Patient cohort tracking: how many of the women who came for ANC1 are the same ones seen at ANC2, 3, 4. Breaks down by payer type and age group.")
        try:
            df_coh = Q.load_anc_patient_cohort_profile(filters, run_query)
            if not df_coh.empty:
                c1, c2 = st.columns(2)
                with c1:
                    journey_summary = df_coh.groupby("anc_journey")["patient_count"].sum().reset_index()
                    journey_summary = journey_summary.sort_values("anc_journey")
                    _pc(bar_chart(journey_summary, x="anc_journey", y="patient_count",
                                  color=AFYA_BLUE, y_label="Patients", height=280, show_text=True))
                with c2:
                    payer_summary = df_coh.groupby("payer_type")["patient_count"].sum().reset_index()
                    completed = df_coh[df_coh["stages_completed"] >= 4].groupby(
                        "payer_type")["patient_count"].sum().reset_index()
                    completed.columns = ["payer_type", "completed"]
                    merged = payer_summary.merge(completed, on="payer_type", how="left").fillna(0)
                    merged["completion_pct"] = (merged["completed"] / merged["patient_count"] * 100).round(1)
                    _pc(bar_chart(merged, x="payer_type", y="completion_pct",
                                  color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                             "Insurance / Corporate": AFYA_BLUE},
                                  y_label="ANC4 Completion %", y_format="pct",
                                  height=280, show_text=True))
        except Exception as e:
            st.warning(f"ANC patient cohort: {e}")

        _gap(12)

        # ── C4: High-Risk Pregnancy ────────────────────────────────────────────
        _sh("High-Risk Pregnancies — Detection & Profile", mt=8)
        _note("Three risk detection methods: clinical ICD10 diagnosis flags, maternal age (adolescent <18 or AME 35+), and repeated elevated BP in vitals. A patient may have multiple risk factors.")
        try:
            df_hr = Q.load_high_risk_pregnancy_profile(filters, run_query)
            if not df_hr.empty:
                total_hr = int(df_hr["patient_count"].sum() or 0)
                _kpi("High-Risk ANC Patients Detected", _n(total_hr),
                     "across all risk categories", CORAL)
                _gap(8)
                _pc(hbar_chart(df_hr.head(12), x="patient_count", y="risk_type",
                               x_label="Patients", color=CORAL, height=320, show_text=True))
        except Exception as e:
            st.warning(f"High-risk pregnancy profile: {e}")

        _gap(8)
        _note("Patient-level high-risk list — sorted by clinical severity, then age risk, then days since last ANC visit.")
        try:
            df_hrp = Q.load_high_risk_pregnancy_patients(filters, run_query)
            if not df_hrp.empty:
                show_cols = ["patient", "age_group", "payer_type", "anc_visits",
                             "days_since_last_anc", "risk_flags"]
                col_labels = {"patient": "Patient", "age_group": "Age",
                              "payer_type": "Payer", "anc_visits": "ANC Visits",
                              "days_since_last_anc": "Days Since ANC", "risk_flags": "Risk Flags"}
                _pc(table_fig(df_hrp[show_cols].head(30), col_labels=col_labels,
                              height=min(600, len(df_hrp.head(30)) * 28 + 60)))
        except Exception as e:
            st.warning(f"High-risk patient list: {e}")

        _gap(12)

        # ── C5: Illnesses in Pregnant Women ───────────────────────────────────
        _sh("Comorbidities in Pregnant Women", mt=8)
        _note("Other conditions diagnosed in patients who also had ANC visits. Identifies illnesses co-occurring with pregnancy that may compound risk.")
        try:
            df_com = Q.load_pregnancy_comorbidities(filters, run_query)
            if not df_com.empty:
                _pc(hbar_chart(df_com.head(15), x="patient_count", y="condition_group",
                               x_label="Pregnant Patients Affected", color=ORANGE,
                               height=min(400, len(df_com.head(15)) * 26 + 60), show_text=True))
        except Exception as e:
            st.warning(f"Pregnancy comorbidities: {e}")

        _gap(12)

        # ── C6: PNC Profile ────────────────────────────────────────────────────
        _sh("Postnatal Care (PNC) — Who Is Coming Back?", mt=8)
        _note("PNC visits by payer and age group. Low PNC volume relative to deliveries signals a retention gap after birth.")
        try:
            df_pnc = Q.load_pnc_profile(filters, run_query)
            if not df_pnc.empty:
                total_pnc = int(df_pnc["visit_count"].sum() or 0)
                total_pnc_pts = int(df_pnc["patient_count"].sum() or 0)
                c1, c2, c3 = st.columns(3)
                with c1: _kpi("PNC Visits", _n(total_pnc), color=TEAL)
                with c2: _kpi("PNC Patients", _n(total_pnc_pts), color=TEAL)
                _gap(8)
                c1, c2 = st.columns(2)
                with c1:
                    pnc_age = df_pnc.groupby("age_group")["visit_count"].sum().reset_index()
                    _pc(bar_chart(pnc_age, x="age_group", y="visit_count",
                                  color=TEAL, y_label="PNC Visits", height=240, show_text=True))
                with c2:
                    pnc_pay = df_pnc.groupby("payer_type")["visit_count"].sum().reset_index()
                    _pc(donut(labels=pnc_pay["payer_type"].tolist(),
                              values=pnc_pay["visit_count"].tolist(),
                              color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                         "Insurance / Corporate": AFYA_BLUE},
                              height=240))
        except Exception as e:
            st.warning(f"PNC profile: {e}")

        _gap(12)

        # ── C7: Deliveries by Age ──────────────────────────────────────────────
        _sh("Deliveries by Maternal Age Group", mt=8)
        try:
            df_del = Q.load_deliveries_by_age(filters, run_query)
            if not df_del.empty:
                _pc(bar_chart(df_del, x="maternal_age_group", y="delivery_count",
                              color=PURPLE, y_label="Deliveries", height=260, show_text=True))
        except Exception as e:
            st.warning(f"Deliveries by age: {e}")

        _gap(12)

        # ── C8: Under-5 Profile ────────────────────────────────────────────────
        _sh("Under-5 Children — What Brought Them?", mt=8)
        _note("Visit category breakdown for children under 5, by age bucket. Shows whether admissions cluster in a specific age group or illness category.")
        try:
            df_u5 = Q.load_under5_profile(filters, run_query)
            if not df_u5.empty:
                total_u5 = int(df_u5["visit_count"].sum() or 0)
                total_admitted = int(df_u5["admitted_count"].sum() or 0)
                c1, c2, c3 = st.columns(3)
                with c1: _kpi("Under-5 Visits", _n(total_u5), color=AFYA_BLUE)
                with c2: _kpi("Admissions", _n(total_admitted),
                              f"{total_admitted / total_u5 * 100:.1f}% admission rate" if total_u5 else "",
                              CORAL if total_admitted / max(total_u5, 1) > 0.1 else TEAL)
                _gap(8)
                cat_summary = df_u5.groupby("visit_category").agg(
                    visit_count=("visit_count", "sum"),
                    admitted_count=("admitted_count", "sum"),
                ).reset_index().sort_values("visit_count", ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Visits by Reason**")
                    _pc(bar_chart(cat_summary, x="visit_category", y="visit_count",
                                  color=AFYA_BLUE, y_label="Visits", height=300, show_text=True))
                with c2:
                    st.markdown("**Admission Rate by Category**")
                    cat_summary["admission_rate_pct"] = (
                        cat_summary["admitted_count"] / cat_summary["visit_count"].replace(0, 1) * 100
                    ).round(1)
                    _pc(hbar_chart(cat_summary.sort_values("admission_rate_pct"),
                                   x="admission_rate_pct", y="visit_category",
                                   x_label="Admission Rate %", y_format="pct",
                                   color=CORAL, height=300, show_text=True))
                _gap(8)
                age_summary = df_u5.groupby("age_bucket")["visit_count"].sum().reset_index()
                st.markdown("**Visits by Age Bucket**")
                _pc(bar_chart(age_summary, x="age_bucket", y="visit_count",
                              color=TEAL, y_label="Visits", height=220, show_text=True))
        except Exception as e:
            st.warning(f"Under-5 profile: {e}")

        _gap(12)

        # ── C9: Adolescent Reproductive Health ────────────────────────────────
        _sh("Adolescent Reproductive Health (Age 10–19)", mt=8)
        _note("What brings adolescents to the hospital in reproductive health terms: family planning, ANC/pregnancy, abortion, PID, STI, PNC, and delivery. Separated by sex and payer.")
        try:
            df_adol = Q.load_adolescent_rh_profile(filters, run_query)
            if not df_adol.empty:
                cat_tot = df_adol.groupby("rh_category").agg(
                    visit_count=("visit_count", "sum"),
                    patient_count=("patient_count", "sum"),
                ).reset_index().sort_values("patient_count", ascending=False)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Adolescent RH — Visit Category**")
                    _pc(hbar_chart(cat_tot, x="patient_count", y="rh_category",
                                   x_label="Patients", color=PURPLE,
                                   height=min(380, len(cat_tot) * 34 + 60), show_text=True))
                with c2:
                    st.markdown("**By Payer Type**")
                    pay_tot = df_adol.groupby("payer_type")["patient_count"].sum().reset_index()
                    _pc(donut(labels=pay_tot["payer_type"].tolist(),
                              values=pay_tot["patient_count"].tolist(),
                              color_map={"Cash": CORAL, "NHIF / SHA": TEAL,
                                         "Insurance / Corporate": AFYA_BLUE},
                              height=280))
        except Exception as e:
            st.warning(f"Adolescent RH: {e}")

        _gap(12)

        # ── C10: RMNCH Revenue — Segment Share & Trend ────────────────────────
        _sh("RMNCH Revenue — Segment Share & Monthly Trend", mt=8)
        _note("Revenue from Maternal, Paediatric, and Adolescent RH segments. The RMNCH Care & Capital Matrix below is built from actual query data.")
        try:
            df_rev = Q.load_rmnch_revenue_trend(filters, run_query)
            if not df_rev.empty:
                seg_summary = df_rev.groupby("rmnch_segment").agg(
                    visit_count=("visit_count", "sum"),
                    patient_count=("patient_count", "sum"),
                    revenue=("revenue", "sum"),
                ).reset_index()
                total_rev = float(seg_summary["revenue"].sum() or 1)
                seg_summary["rev_share_pct"] = (seg_summary["revenue"] / total_rev * 100).round(1)

                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                with c1: _kpi("Total RMNCH Revenue", _k(total_rev), color=PURPLE)
                for col, seg, color in [(c2, "Maternal", AFYA_BLUE),
                                        (c3, "Paediatric (<12y)", TEAL),
                                        (c4, "Adolescent RH", ORANGE)]:
                    row_s = seg_summary[seg_summary["rmnch_segment"] == seg]
                    rev_s = float(row_s["revenue"].sum()) if not row_s.empty else 0
                    pct_s = float(row_s["rev_share_pct"].sum()) if not row_s.empty else 0
                    with col:
                        _kpi(seg, _k(rev_s), f"{pct_s:.1f}% of RMNCH revenue", color)

                _gap(12)

                # RMNCH Care & Capital Matrix — from live data
                st.markdown("**RMNCH Care & Capital Matrix**")
                matrix_rows = []
                for seg in ["Maternal", "Paediatric (<12y)", "Adolescent RH"]:
                    row_s = seg_summary[seg_summary["rmnch_segment"] == seg]
                    if row_s.empty:
                        continue
                    r = row_s.iloc[0]
                    matrix_rows.append({
                        "Segment": seg,
                        "Patients": _n(r["patient_count"]),
                        "Visits": _n(r["visit_count"]),
                        "Revenue": _k(r["revenue"]),
                        "% Share": f"{r['rev_share_pct']:.1f}%",
                    })
                if matrix_rows:
                    mdf = pd.DataFrame(matrix_rows)
                    _pc(table_fig(mdf, height=min(240, len(mdf) * 40 + 60)))

                _gap(12)

                # Monthly revenue trend
                st.markdown("**Monthly Revenue Trend by RMNCH Segment**")
                if "visit_month" in df_rev.columns:
                    df_rev["visit_month"] = pd.to_datetime(df_rev["visit_month"])
                    monthly = df_rev.sort_values("visit_month")
                    seg_colors = {"Maternal": AFYA_BLUE, "Paediatric (<12y)": TEAL,
                                  "Adolescent RH": ORANGE}
                    fig_trend = go.Figure()
                    for seg, grp in monthly.groupby("rmnch_segment"):
                        fig_trend.add_trace(go.Scatter(
                            x=grp["visit_month"], y=grp["revenue"],
                            mode="lines+markers", name=seg,
                            line=dict(color=seg_colors.get(seg, PURPLE), width=2),
                            hovertemplate=f"<b>{seg}</b><br>%{{x|%b %Y}}<br>KES %{{y:,.0f}}<extra></extra>",
                        ))
                    fig_trend.update_layout(
                        height=300, margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_title="Revenue (KES)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    _pc(fig_trend)
        except Exception as e:
            st.warning(f"RMNCH revenue: {e}")

    # ── COMMUNICABLE & HIV TAB ────────────────────────────────────────────────
    with st_d:
        _sh("D — Communicable Disease & HIV")

        # ── D0: Disease KPI Snapshot ───────────────────────────────────────────
        _sh("Disease Case Counts — TB, Malaria, URTI, Typhoid, Enteric, HIV", mt=8)
        _note("Total patients and visits per disease in the selected period, with how many required inpatient admission.")
        try:
            df_kpi = Q.load_disease_kpi_snapshot(filters, run_query)
            if not df_kpi.empty:
                for col_set in [df_kpi.iloc[i:i+3] for i in range(0, len(df_kpi), 3)]:
                    cols = st.columns(3)
                    for col, (_, row_k) in zip(cols, col_set.iterrows()):
                        adm_pct = float(row_k.get("admission_rate_pct") or 0)
                        with col:
                            _kpi(
                                str(row_k["disease_label"]),
                                f"{_n(row_k['patient_count'])} patients",
                                f"{_n(row_k['visit_count'])} visits · {adm_pct:.0f}% admitted",
                                CORAL if adm_pct > 15 else AFYA_BLUE,
                            )
                    _gap(4)
        except Exception as e:
            st.warning(f"Disease KPIs: {e}")

        _gap(12)

        # ── D1: Who Does Each Disease Affect? ─────────────────────────────────
        _sh("Who Does Each Disease Affect? — Age & Sex Breakdown", mt=8)
        _note("Each bar shows patients by age group. The two bars side-by-side compare Female (F) vs Male (M). Paediatric = Under 5 + 5-17 combined. Use this to identify whether a disease disproportionately hits a specific demographic.")
        try:
            df_dem = Q.load_disease_demographics(filters, run_query)
            if not df_dem.empty:
                diseases = df_dem["disease_label"].unique().tolist()
                n_cols = min(3, len(diseases))
                rows_d = [diseases[i:i+n_cols] for i in range(0, len(diseases), n_cols)]
                for row_diseases in rows_d:
                    cols_d = st.columns(n_cols)
                    for col_d, dis in zip(cols_d, row_diseases):
                        sub_d = df_dem[df_dem["disease_label"] == dis].copy()
                        with col_d:
                            st.markdown(f"**{dis}**")
                            age_sex = sub_d.groupby(["age_group", "sex"])["patient_count"].sum().reset_index()
                            fig_ds = go.Figure()
                            sex_colors = {"F": PURPLE, "FEMALE": PURPLE,
                                          "M": AFYA_BLUE, "MALE": AFYA_BLUE,
                                          "Unknown": GRAY}
                            for sx in age_sex["sex"].unique():
                                sub_sx = age_sex[age_sex["sex"] == sx]
                                fig_ds.add_trace(go.Bar(
                                    name=sx,
                                    x=sub_sx["age_group"],
                                    y=sub_sx["patient_count"],
                                    marker_color=sex_colors.get(sx, GRAY),
                                    showlegend=(dis == diseases[0]),
                                ))
                            fig_ds.update_layout(
                                barmode="group", height=200,
                                margin=dict(l=0, r=0, t=10, b=30),
                                plot_bgcolor="white", paper_bgcolor="white",
                                yaxis_title="Patients",
                                xaxis=dict(tickangle=-20),
                                legend=dict(orientation="h", y=-0.35),
                            )
                            _pc(fig_ds)
        except Exception as e:
            st.warning(f"Disease demographics: {e}")

        _gap(12)

        # ── D2: Monthly Trend — Spike vs Sustained ─────────────────────────────
        _sh("Monthly Case Trend — Spike or Sustained? (Typhoid Focus)", mt=8)
        _note("Use this to determine whether the high typhoid inpatient count was a one-time outbreak spike or a persistent pattern. A single tall bar indicates a spike; multiple high months indicate endemic spread.")
        try:
            df_trend = Q.load_disease_monthly_trend(filters, run_query)
            if not df_trend.empty:
                df_trend["visit_month"] = pd.to_datetime(df_trend["visit_month"])
                selected_diseases = ["Typhoid", "Malaria", "URTI", "TB", "Enteric / GI"]
                disease_colors = {
                    "Typhoid": "#D97706", "Malaria": TEAL, "URTI": AFYA_BLUE,
                    "TB": CORAL, "Enteric / GI": PURPLE, "HIV": GRAY,
                }
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_tr = go.Figure()
                    for dis in selected_diseases:
                        sub_tr = df_trend[df_trend["disease_label"] == dis].sort_values("visit_month")
                        if not sub_tr.empty:
                            fig_tr.add_trace(go.Scatter(
                                x=sub_tr["visit_month"], y=sub_tr["visit_count"],
                                name=dis, mode="lines+markers",
                                line=dict(color=disease_colors.get(dis, GRAY), width=2),
                                hovertemplate=f"<b>{dis}</b> — %{{x|%b %Y}}: %{{y:,}} visits<extra></extra>",
                            ))
                    fig_tr.update_layout(
                        height=320, margin=dict(l=0, r=0, t=20, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_title="Visits",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    _pc(fig_tr)
                with c2:
                    typhoid_monthly = df_trend[df_trend["disease_label"] == "Typhoid"].copy()
                    if not typhoid_monthly.empty:
                        mean_v = typhoid_monthly["visit_count"].mean()
                        max_v = typhoid_monthly["visit_count"].max()
                        spike_months = typhoid_monthly[
                            typhoid_monthly["visit_count"] > mean_v * 1.5
                        ]
                        _kpi("Typhoid Monthly Avg", f"{mean_v:.0f} visits", color=ORANGE)
                        _gap(8)
                        _kpi("Peak Month", f"{max_v:.0f} visits",
                             f"{len(spike_months)} month(s) above 1.5× average",
                             CORAL if len(spike_months) <= 2 else ORANGE)
                        _gap(8)
                        verdict = "Spike (1–2 high months)" if len(spike_months) <= 2 else "Sustained / Endemic"
                        st.markdown(
                            f'<div style="background:#FFF8E7;border-left:4px solid #E69A00;'
                            f'padding:10px;border-radius:6px;font-size:12px">'
                            f'<b>Pattern:</b> {verdict}</div>',
                            unsafe_allow_html=True,
                        )
        except Exception as e:
            st.warning(f"Monthly trend: {e}")

        _gap(12)

        # ── D3: TB-HIV Co-infection ────────────────────────────────────────────
        _sh("TB & HIV — Co-infection Analysis", mt=8)
        _note("The female/male counts below are HIV patients (B20 diagnosis) by sex — they show who has HIV, not who has TB. The co-infection section below shows specifically which TB patients also have an HIV diagnosis, and whether HIV was tested or detected via clinical markers.")
        try:
            df_coinfect = Q.load_tb_hiv_coinfection(filters, run_query)
            if not df_coinfect.empty:
                row_c = df_coinfect.iloc[0]
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1: _kpi("TB Patients", _n(row_c.get("tb_patients")), color=CORAL)
                with c2: _kpi("HIV Patients", _n(row_c.get("hiv_patients")), color=ORANGE)
                with c3: _kpi("TB+HIV Co-infected", _n(row_c.get("tb_hiv_coinfected")),
                               f"{float(row_c.get('coinfection_rate_pct') or 0):.1f}% of TB patients",
                               CORAL if float(row_c.get("coinfection_rate_pct") or 0) > 10 else TEAL)
                with c4: _kpi("HIV Test Coverage",
                               _p(row_c.get("hiv_test_coverage_pct")),
                               "of TB patients had HIV test done",
                               TEAL if float(row_c.get("hiv_test_coverage_pct") or 0) >= 80 else CORAL)
                with c5: _kpi("TB with Fever Signal", _n(row_c.get("tb_with_fever")),
                               "elevated temp (>37.5°C) in vitals", ORANGE)
        except Exception as e:
            st.warning(f"TB-HIV co-infection: {e}")

        _gap(8)
        try:
            df_hiv = Q.load_hiv_profile(filters, run_query)
            if not df_hiv.empty:
                row_h = df_hiv.iloc[0]
                _note(f"HIV patient profile: {_n(row_h.get('hiv_patients'))} total — "
                      f"{_n(row_h.get('female'))} Female · {_n(row_h.get('male'))} Male · "
                      f"{_n(row_h.get('paediatric'))} Paediatric")
        except Exception as e:
            st.warning(f"HIV profile: {e}")

        _gap(12)

        # ── D4: Malaria — Who It Affects + Lab Accuracy ────────────────────────
        _sh("Malaria — Who Is Most Affected & Test Accuracy", mt=8)
        _note("Left: malaria patients by age group and sex. Right: how often is malaria tested vs clinically diagnosed without a test (false positive risk). Diagnosis without a test = clinical-only.")
        try:
            df_dem2 = Q.load_disease_demographics(filters, run_query)
            df_mal_lab = Q.load_malaria_lab_accuracy(filters, run_query)
            c1, c2 = st.columns(2)
            with c1:
                if not df_dem2.empty:
                    mal_dem = df_dem2[df_dem2["disease_label"] == "Malaria"].copy()
                    if not mal_dem.empty:
                        st.markdown("**Malaria Patients by Age Group**")
                        age_mal = mal_dem.groupby("age_group")["patient_count"].sum().reset_index().sort_values("patient_count", ascending=False)
                        _pc(bar_chart(age_mal, x="age_group", y="patient_count",
                                      color=TEAL, y_label="Patients", height=260, show_text=True))
            with c2:
                if not df_mal_lab.empty:
                    row_ml = df_mal_lab.iloc[0]
                    tested = int(row_ml.get("visits_with_test") or 0)
                    not_tested = int(row_ml.get("no_test_done") or 0)
                    resulted = int(row_ml.get("test_resulted") or 0)
                    ordered_only = int(row_ml.get("test_ordered_only") or 0)
                    st.markdown("**Malaria Test Coverage**")
                    _kpi("Test Rate", _p(row_ml.get("test_rate_pct")),
                         f"{tested:,} tested · {not_tested:,} no test (clinical-only dx)",
                         TEAL if float(row_ml.get("test_rate_pct") or 0) >= 70 else CORAL)
                    _gap(8)
                    if tested > 0:
                        _kpi("Result Rate", _p(row_ml.get("result_rate_pct")),
                             f"{resulted:,} resulted · {ordered_only:,} ordered but not resulted",
                             TEAL if float(row_ml.get("result_rate_pct") or 0) >= 80 else ORANGE)
                        _gap(8)
                        _pc(donut(
                            labels=["Test Resulted", "Test Ordered Only", "No Test Done"],
                            values=[resulted, ordered_only, not_tested],
                            color_map={"Test Resulted": TEAL,
                                       "Test Ordered Only": ORANGE,
                                       "No Test Done": CORAL},
                            height=200,
                        ))
        except Exception as e:
            st.warning(f"Malaria: {e}")

        _gap(12)

        # ── D5: Admissions per Disease ─────────────────────────────────────────
        _sh("Which Diseases Lead to Admission?", mt=8)
        _note("Admission rate per disease. TB and Typhoid typically have higher admission rates than URTI. High admission rates in diseases that are normally outpatient indicate severity or delayed presentation.")
        try:
            df_kpi2 = Q.load_disease_kpi_snapshot(filters, run_query)
            if not df_kpi2.empty:
                df_kpi2["admission_rate_pct"] = pd.to_numeric(df_kpi2["admission_rate_pct"], errors="coerce")
                c1, c2 = st.columns(2)
                with c1:
                    _pc(hbar_chart(
                        df_kpi2.sort_values("admission_rate_pct"),
                        x="admission_rate_pct", y="disease_label",
                        x_label="Admission Rate %", y_format="pct",
                        color=CORAL, height=280, show_text=True,
                    ))
                with c2:
                    _pc(bar_chart(
                        df_kpi2.sort_values("admitted_count", ascending=False),
                        x="disease_label", y="admitted_count",
                        color=ORANGE, y_label="Admitted Patients", height=280, show_text=True,
                    ))
        except Exception as e:
            st.warning(f"Disease admissions: {e}")

        _gap(12)

        # ── D6: Comorbidities per Communicable Disease ─────────────────────────
        _sh("What Other Conditions Co-occur with Communicable Diseases?", mt=8)
        _note("Top 5 comorbidities found in patients who also have each communicable diagnosis. Helps identify compound risk and treatment complexity.")
        try:
            df_comorb = Q.load_communicable_comorbidities(filters, run_query)
            if not df_comorb.empty:
                diseases_c = df_comorb["disease_label"].unique().tolist()
                cols_c = st.columns(min(3, len(diseases_c)))
                for col_c, dis_c in zip(cols_c, diseases_c[:3]):
                    sub_c = df_comorb[df_comorb["disease_label"] == dis_c]
                    with col_c:
                        st.markdown(f"**{dis_c} — Top Comorbidities**")
                        _pc(hbar_chart(sub_c, x="patient_count", y="comorbidity",
                                       x_label="Patients", color=PURPLE,
                                       height=min(240, len(sub_c) * 36 + 60), show_text=True))
                if len(diseases_c) > 3:
                    cols_c2 = st.columns(min(3, len(diseases_c) - 3))
                    for col_c2, dis_c2 in zip(cols_c2, diseases_c[3:6]):
                        sub_c2 = df_comorb[df_comorb["disease_label"] == dis_c2]
                        with col_c2:
                            st.markdown(f"**{dis_c2} — Top Comorbidities**")
                            _pc(hbar_chart(sub_c2, x="patient_count", y="comorbidity",
                                           x_label="Patients", color=PURPLE,
                                           height=min(240, len(sub_c2) * 36 + 60), show_text=True))
        except Exception as e:
            st.warning(f"Communicable comorbidities: {e}")

        _gap(12)

        # ── D7: Surge Pattern Indicator ────────────────────────────────────────
        _sh("Seasonal Surge Pattern — Malaria, URTI & Typhoid", mt=8)
        _note("Each disease keeps its own color. A red triangle marker (▲) appears on top of a bar when that disease exceeded 1.5× its period average that month — the label shows which disease surged.")
        try:
            df_surge = Q.load_disease_monthly_trend(filters, run_query)
            if not df_surge.empty:
                df_surge["visit_month"] = pd.to_datetime(df_surge["visit_month"])
                surge_diseases = ["Malaria", "URTI", "Typhoid"]
                surge_df = df_surge[df_surge["disease_label"].isin(surge_diseases)].copy()
                if not surge_df.empty:
                    means = surge_df.groupby("disease_label")["visit_count"].mean()
                    surge_df["is_surge"] = surge_df.apply(
                        lambda r: r["visit_count"] > means.get(r["disease_label"], 0) * 1.5, axis=1
                    )
                    surge_clrs = {"Malaria": TEAL, "URTI": AFYA_BLUE, "Typhoid": ORANGE}
                    fig_sg = go.Figure()

                    # bars — each disease keeps its own colour always
                    for dis_s in surge_diseases:
                        sub_s = surge_df[surge_df["disease_label"] == dis_s].sort_values("visit_month")
                        if sub_s.empty:
                            continue
                        fig_sg.add_trace(go.Bar(
                            name=dis_s,
                            x=sub_s["visit_month"],
                            y=sub_s["visit_count"],
                            marker_color=surge_clrs.get(dis_s, GRAY),
                            hovertemplate=f"<b>{dis_s}</b> — %{{x|%b %Y}}: %{{y:,}} visits<extra></extra>",
                        ))

                    # surge markers — red ▲ per disease, offset so they don't overlap
                    offsets = {"Malaria": -0.27, "URTI": 0, "Typhoid": 0.27}
                    first_surge = True
                    for dis_s in surge_diseases:
                        sub_s = surge_df[
                            (surge_df["disease_label"] == dis_s) & surge_df["is_surge"]
                        ].sort_values("visit_month")
                        if sub_s.empty:
                            continue
                        fig_sg.add_trace(go.Scatter(
                            name="Surge flag",
                            x=sub_s["visit_month"],
                            y=sub_s["visit_count"],
                            mode="markers+text",
                            marker=dict(symbol="triangle-up", size=12,
                                        color="#DC2626", line=dict(color="white", width=1)),
                            text=[dis_s] * len(sub_s),
                            textposition="top center",
                            textfont=dict(size=9, color="#DC2626"),
                            showlegend=first_surge,
                            legendgroup="surge",
                            hovertemplate=f"<b>SURGE: {dis_s}</b><br>%{{x|%b %Y}} — %{{y:,}} visits<br>(>1.5× average)<extra></extra>",
                        ))
                        first_surge = False

                    fig_sg.update_layout(
                        barmode="group", height=320,
                        margin=dict(l=0, r=0, t=30, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis_title="Visits", xaxis_title="",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    _pc(fig_sg)

                    # surge summary table
                    surge_rows = surge_df[surge_df["is_surge"]].copy()
                    if not surge_rows.empty:
                        surge_rows["Month"] = surge_rows["visit_month"].dt.strftime("%b %Y")
                        surge_rows["vs Average"] = surge_rows.apply(
                            lambda r: f"{r['visit_count'] / max(means.get(r['disease_label'], 1), 1):.1f}×", axis=1
                        )
                        surge_rows = surge_rows.rename(columns={
                            "disease_label": "Disease", "visit_count": "Visits"
                        })[["Disease", "Month", "Visits", "vs Average"]].sort_values("Month")
                        st.markdown("**Surge months (>1.5× average):**")
                        _pc(table_fig(surge_rows, height=min(240, len(surge_rows) * 32 + 50)))
        except Exception as e:
            st.warning(f"Surge pattern: {e}")

        _gap(12)

        # ── D8: Unified Pipeline Matrix (from live data) ───────────────────────
        _sh("Unified Acute & Communicable Pipeline Matrix", mt=8)
        _note("Built from actual Snowflake data: lab confirmation rate, inpatient admission rate, primary demographic, top comorbidity, and primary payer per disease.")
        try:
            df_cpm = Q.load_communicable_pipeline_matrix(filters, run_query)
            if not df_cpm.empty:
                for _c in ("quarterly_visits", "lab_confirmation_pct", "inpatient_admission_pct"):
                    if _c in df_cpm.columns:
                        df_cpm[_c] = pd.to_numeric(df_cpm[_c], errors="coerce")

                rows_html = ""
                for _, r in df_cpm.iterrows():
                    ip_pct = float(r.get("inpatient_admission_pct") or 0)
                    ip_color = "#DC2626" if ip_pct > 15 else "#16A34A" if ip_pct < 5 else "#D97706"
                    rows_html += (
                        f'<tr>'
                        f'<td style="padding:7px 8px;font-weight:600;font-size:12px">{r.get("disease_group","")}</td>'
                        f'<td style="padding:7px 8px;font-size:11px;text-align:right">{int(r.get("quarterly_visits", 0) or 0):,}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{r.get("primary_age_sex","")}</td>'
                        f'<td style="padding:7px 8px;font-size:11px;text-align:right">'
                        f'{float(r.get("lab_confirmation_pct", 0) or 0):.0f}%</td>'
                        f'<td style="padding:7px 8px;font-size:11px;text-align:right;color:{ip_color};font-weight:700">'
                        f'{ip_pct:.0f}%</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{r.get("primary_comorbidity","—")}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{r.get("primary_payer","")}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-family:sans-serif">'
                    '<thead><tr style="background:#E05C2D;color:white">'
                    '<th style="padding:8px;text-align:left">Disease</th>'
                    '<th style="padding:8px;text-align:right">90d Visits</th>'
                    '<th style="padding:8px;text-align:left">Primary Demographic</th>'
                    '<th style="padding:8px;text-align:right">Lab Confirm %</th>'
                    '<th style="padding:8px;text-align:right">IP Admission %</th>'
                    '<th style="padding:8px;text-align:left">Top Comorbidity</th>'
                    '<th style="padding:8px;text-align:left">Primary Payer</th>'
                    '</tr></thead><tbody>' + rows_html + '</tbody></table></div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(f"Pipeline matrix: {e}")

    # ── MENTAL HEALTH TAB ─────────────────────────────────────────────────────
    with st_e:
        _sh("E — Mental Health & Psychiatric Analytics")

        try:
            df_mhk = Q.load_mh_kpis(filters, run_query)
            if not df_mhk.empty:
                row = df_mhk.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1: _kpi("MH Patients", _n(row.get("total_mh_patients")))
                with c2: _kpi("Inpatient Share", _p(row.get("inpatient_share_pct")),
                               "admitted for MH", ORANGE)
                with c3: _kpi("MH Visits", _n(row.get("total_mh_visits")))
        except Exception as e:
            st.warning(f"E1: {e}")

        _gap(12)

        # Grouped Demographic Illness Matrix
        _sh("Mental Health — Diagnostic Breakdown by Age & Condition", mt=8)
        _note("Each age group is broken into specific diagnostic sub-segments: Depression/Anxiety vs Substance Abuse vs Psychotic Disorders vs Dementia.")
        try:
            df_mhd = Q.load_mh_diagnostic_breakdown(filters, run_query)
            if not df_mhd.empty:
                df_mhd["patient_count"] = pd.to_numeric(df_mhd["patient_count"], errors="coerce")
                fig_mhd = go.Figure()
                mh_cat_colors = {
                    "Depression & Anxiety": AFYA_BLUE,
                    "Substance & Alcohol": CORAL,
                    "Psychotic Disorders": PURPLE,
                    "Dementia / Organic Brain": ORANGE,
                    "Other Mental Health": MUTED,
                }
                for cat, color in mh_cat_colors.items():
                    sub = df_mhd[df_mhd["mh_category"] == cat]
                    if not sub.empty:
                        age_grp = sub.groupby("age_group")["patient_count"].sum().reset_index()
                        fig_mhd.add_trace(go.Bar(
                            name=cat, x=age_grp["age_group"], y=age_grp["patient_count"],
                            marker_color=color,
                            hovertemplate=f"<b>%{{x}}</b><br>{cat}: %{{y:,}}<extra></extra>",
                        ))
                fig_mhd.update_layout(
                    barmode="stack", height=300,
                    margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Patients", rangemode="tozero"),
                    legend=dict(orientation="h", y=-0.25, xanchor="center", x=0.5),
                )
                _pc(fig_mhd)
            else:
                df_mhas = Q.load_mh_by_age_sex(filters, run_query)
                if not df_mhas.empty:
                    pivot = df_mhas.pivot_table(index="age_group", columns="sex",
                                                values="patient_count", aggfunc="sum", fill_value=0)
                    _pc(bar_chart(pivot.reset_index(), x="age_group", y=list(pivot.columns),
                                  color_map={"F": "#7b5ea7", "FEMALE": "#7b5ea7",
                                             "M": TEAL, "MALE": TEAL},
                                  y_label="Patients", height=280))
        except Exception as e:
            st.warning(f"E2: {e}")

        _gap(12)

        # Comorbidity profile: standalone vs comorbid
        _sh("Standalone vs Comorbid Mental Health — Who Has a Hidden Primary Condition?", mt=8)
        _note("A patient with Depression secondary to Hypertension is a chronic patient who needs integrated care, not just psychiatric medication.")
        try:
            df_mhc = Q.load_mh_comorbidity_profile(filters, run_query)
            if not df_mhc.empty:
                for _c in ("standalone_patients", "comorbid_patients", "standalone_pct"):
                    df_mhc[_c] = pd.to_numeric(df_mhc[_c], errors="coerce")

                c1, c2 = st.columns([1.2, 0.8])
                with c1:
                    fig_mhc = go.Figure()
                    fig_mhc.add_trace(go.Bar(
                        name="Standalone", x=df_mhc["mh_category"],
                        y=df_mhc["standalone_patients"], marker_color=TEAL,
                        hovertemplate="<b>%{x}</b><br>Standalone: %{y:,}<extra></extra>",
                    ))
                    fig_mhc.add_trace(go.Bar(
                        name="Comorbid (has primary NCD/other)", x=df_mhc["mh_category"],
                        y=df_mhc["comorbid_patients"], marker_color=CORAL,
                        hovertemplate="<b>%{x}</b><br>Comorbid: %{y:,}<extra></extra>",
                    ))
                    fig_mhc.update_layout(
                        barmode="group", height=280,
                        margin=dict(l=0, r=20, t=20, b=60),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(tickangle=-20),
                        yaxis=dict(title="Patients", rangemode="tozero"),
                        legend=dict(orientation="h", y=1.05, xanchor="right", x=1),
                    )
                    _pc(fig_mhc)
                with c2:
                    _pc(table_fig(
                        df_mhc[["mh_category", "standalone_patients", "comorbid_patients",
                                "standalone_pct", "top_comorbidity"]],
                        col_labels={"mh_category": "Condition", "standalone_patients": "Standalone",
                                    "comorbid_patients": "Comorbid",
                                    "standalone_pct": "Standalone %",
                                    "top_comorbidity": "Top Co-Condition"},
                        fmt={"standalone_pct": "pct"},
                        height=280,
                    ))
        except Exception as e:
            st.warning(f"E3: {e}")

        _gap(12)

        # Monthly trend by MH category
        _sh("Mental Health Visit Growth — Monthly Trend by Category", mt=8)
        try:
            df_mht = Q.load_mh_monthly_trend(filters, run_query)
            if not df_mht.empty:
                df_mht["visit_count"] = pd.to_numeric(df_mht["visit_count"], errors="coerce")
                pivot_mht = df_mht.pivot_table(
                    index="visit_month", columns="mh_category",
                    values="visit_count", aggfunc="sum", fill_value=0
                ).reset_index()
                fig_mht = go.Figure()
                for col in [c for c in pivot_mht.columns if c != "visit_month"]:
                    fig_mht.add_trace(go.Scatter(
                        x=pivot_mht["visit_month"], y=pivot_mht[col],
                        name=col, mode="lines+markers",
                        hovertemplate=f"<b>{col}</b> — %{{x|%b %Y}}: %{{y:,}}<extra></extra>",
                    ))
                fig_mht.update_layout(
                    height=280, margin=dict(l=0, r=20, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Visits", rangemode="tozero"),
                    legend=dict(orientation="h", y=-0.25, xanchor="center", x=0.5),
                )
                _pc(fig_mht)
        except Exception as e:
            st.warning(f"E4: {e}")

        _gap(12)

        # Mental Health Care & Synergy Matrix — built from live query data
        _sh("Mental Health Care & Synergy Matrix", mt=8)
        _note("Built from actual data: comorbidity split from load_mh_comorbidity_profile, trend direction from load_mh_monthly_trend.")
        try:
            df_mhc2 = Q.load_mh_comorbidity_profile(filters, run_query)
            df_mht2 = Q.load_mh_monthly_trend(filters, run_query)
            if not df_mhc2.empty:
                # compute trend direction per category from monthly data
                trend_map = {}
                if not df_mht2.empty:
                    df_mht2["visit_month"] = pd.to_datetime(df_mht2["visit_month"])
                    for cat_t, grp_t in df_mht2.groupby("mh_category"):
                        grp_t = grp_t.sort_values("visit_month")
                        if len(grp_t) >= 2:
                            first_h = grp_t["visit_count"].iloc[:len(grp_t)//2].mean()
                            last_h = grp_t["visit_count"].iloc[len(grp_t)//2:].mean()
                            trend_map[cat_t] = (
                                "Rising" if last_h > first_h * 1.15
                                else "Declining" if last_h < first_h * 0.85
                                else "Stable"
                            )

                rows_html_mh = ""
                for _, r_mh in df_mhc2.iterrows():
                    cat_mh = str(r_mh.get("mh_category", ""))
                    standalone = int(r_mh.get("standalone_patients") or 0)
                    comorbid   = int(r_mh.get("comorbid_patients") or 0)
                    total_mh   = standalone + comorbid or 1
                    sa_pct = round(standalone / total_mh * 100)
                    co_pct = 100 - sa_pct
                    trend_label = trend_map.get(cat_mh, "—")
                    trend_color = "#16A34A" if trend_label == "Rising" else "#DC2626" if trend_label == "Declining" else "#D97706"
                    top_comorbid = str(r_mh.get("top_comorbidity") or "—")
                    rows_html_mh += (
                        f'<tr>'
                        f'<td style="padding:7px 8px;font-weight:600;font-size:12px">{cat_mh}</td>'
                        f'<td style="padding:7px 8px;font-size:11px;color:{trend_color};font-weight:700">{trend_label}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{sa_pct}% Standalone · {co_pct}% Comorbid</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{standalone:,} / {comorbid:,}</td>'
                        f'<td style="padding:7px 8px;font-size:11px">{top_comorbid}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    '<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-family:sans-serif">'
                    '<thead><tr style="background:#7b5ea7;color:white">'
                    '<th style="padding:8px;text-align:left">Condition</th>'
                    '<th style="padding:8px;text-align:left">Trend</th>'
                    '<th style="padding:8px;text-align:left">Standalone vs Comorbid %</th>'
                    '<th style="padding:8px;text-align:right">Standalone / Comorbid Count</th>'
                    '<th style="padding:8px;text-align:left">Top Co-Condition</th>'
                    '</tr></thead><tbody>' + rows_html_mh + '</tbody></table></div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.warning(f"MH Synergy Matrix: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CLINICAL WORKLOAD & QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def render_tab5_workload(filters: dict, run_query):
    _sh("Clinical Workload & Quality")
    _note("Scoped for Head of Clinical. "
          "Shortcut rate, BP omission, unplanned 72h returns.")

    # ── QUALITY SIGNALS SCORECARD ─────────────────────────────────────────────
    _sh("Clinical Quality Signals — Bubble View", mt=12)
    _note(
        "Bubble size = patient volume. Position = rate. "
        "Top-right bubbles are highest priority: high volume AND high risk rate."
    )
    try:
        df_sc  = Q.load_shortcut_rate(filters, run_query)
        df_bp  = Q.load_bp_omission_rate(filters, run_query)
        df_72  = Q.load_return_72h(filters, run_query)

        panels = [
            (df_sc,  "shortcut_rate_pct",  "chronic_visits",  "Shortcut Rate", ORANGE,
             "Single Dx on chronic patients", "Shortcut Rate %"),
            (df_bp,  "omission_pct",       "htn_visits",      "BP Omission",   CORAL,
             "HTN visits without BP recorded", "Omission %"),
            (df_72,  "return_72h_pct",     "total_visits",    "72h Return",    PURPLE,
             "Unplanned return within 72h", "Return Rate %"),
        ]

        cols3 = st.columns(3)
        for col, (df_p, rate_col, vol_col, title, color, subtitle, y_label) in zip(cols3, panels):
            with col:
                if df_p.empty:
                    st.info(f"No data: {title}")
                    continue
                df_p = df_p.copy()
                df_p[rate_col] = pd.to_numeric(df_p[rate_col], errors="coerce")
                df_p[vol_col]  = pd.to_numeric(df_p[vol_col],  errors="coerce")
                df_p = df_p.reset_index(drop=True)
                df_p["label"] = [f"C{i+1}" for i in range(len(df_p))]
                median_rate = df_p[rate_col].median()

                fig_q = go.Figure()
                fig_q.add_trace(go.Scatter(
                    x=df_p[vol_col],
                    y=df_p[rate_col],
                    mode="markers+text",
                    text=df_p["label"],
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(
                        size=df_p[vol_col].apply(
                            lambda v: max(10, min(40, v / df_p[vol_col].max() * 36 + 8))
                            if df_p[vol_col].max() > 0 else 12
                        ),
                        color=[CORAL if v > median_rate * 1.5
                               else ORANGE if v > median_rate
                               else TEAL
                               for v in df_p[rate_col]],
                        opacity=0.85,
                        line=dict(width=1, color="white"),
                    ),
                    customdata=df_p[["clinician", vol_col, rate_col]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"Volume: %{{customdata[1]:.0f}}<br>"
                        f"{y_label}: %{{customdata[2]:.1f}}%<extra></extra>"
                    ),
                ))
                fig_q.add_hline(y=median_rate,
                                line=dict(color=GRAY, width=1, dash="dot"),
                                annotation_text=f"median {median_rate:.0f}%",
                                annotation_font=dict(size=8, color=GRAY))
                fig_q.update_layout(
                    height=280,
                    margin=dict(l=0, r=10, t=30, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    title=dict(text=f"<b>{title}</b><br><sup>{subtitle}</sup>",
                               font=dict(size=11), x=0),
                    xaxis=dict(title="Patient Volume", rangemode="tozero"),
                    yaxis=dict(title=y_label, rangemode="tozero"),
                    showlegend=False,
                )
                _pc(fig_q)

                # Top offender note
                worst_row = df_p.loc[df_p[rate_col].idxmax()]
                if float(worst_row[rate_col] or 0) > median_rate * 1.5:
                    _note(
                        f"{worst_row['clinician']}: {worst_row[rate_col]:.0f}% {y_label.lower()} "
                        f"({int(worst_row[vol_col] or 0):,} patients)."
                    )
    except Exception as e:
        st.warning(f"Quality signals: {e}")

    _gap(12)

    # ── SECTION F: CLINICIAN DOCUMENTATION RATES ─────────────────────────────
    _sh("F — Documentation Rates by Clinician — Ordered by Patient Load", mt=8)
    _note(
        "Clinicians with the highest load do not always have the lowest documentation rates "
        "— but the relationship is worth watching."
    )
    try:
        df = Q.load_clinician_load(filters, run_query)
        if not df.empty:
            for _c in ("vitals_rate_pct", "notes_rate_pct", "vitals_rate_new_pct",
                       "vitals_rate_returning_pct", "new_visit_pct",
                       "new_visits", "returning_visits", "total_visits",
                       "avg_daily_patients", "days_worked"):
                if _c in df.columns:
                    df[_c] = pd.to_numeric(df[_c], errors="coerce")

            df = df.reset_index(drop=True)
            df["label"] = ["Clinician " + str(i + 1) for i in range(len(df))]
            df15 = df.head(15).copy()

            zero_mask = (df15["vitals_rate_pct"].fillna(0) == 0) & (df15["notes_rate_pct"].fillna(0) == 0)
            has_zero = bool(zero_mask.any())

            non_zero = df15[~zero_mask]
            med_vitals = non_zero["vitals_rate_pct"].median() if not non_zero.empty else None
            med_notes  = non_zero["notes_rate_pct"].median()  if not non_zero.empty else None

            hover_new = [
                f"<b>{row.label}</b><br>"
                f"Vitals (new patients): {row.vitals_rate_new_pct:.0f}%<br>"
                f"New patients: {int(row.new_visits or 0):,}<extra></extra>"
                for _, row in df15.iterrows()
            ]
            hover_ret = [
                f"<b>{row.label}</b><br>"
                f"Vitals (returning/follow-up): {row.vitals_rate_returning_pct:.0f}%<br>"
                f"Returning patients: {int(row.returning_visits or 0):,}<extra></extra>"
                for _, row in df15.iterrows()
            ]
            hover_notes = [
                f"<b>{row.label}</b><br>"
                f"Notes documented: {row.notes_rate_pct:.0f}%<br>"
                f"Total visits: {int(row.total_visits or 0):,}<extra></extra>"
                for _, row in df15.iterrows()
            ]

            fig_doc = go.Figure()
            fig_doc.add_trace(go.Bar(
                y=df15["label"], x=df15["vitals_rate_new_pct"],
                name="Vitals — new patients", orientation="h",
                marker_color=TEAL,
                text=[f"{v:.0f}%" if pd.notna(v) else "—" for v in df15["vitals_rate_new_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_new,
            ))
            fig_doc.add_trace(go.Bar(
                y=df15["label"], x=df15["vitals_rate_returning_pct"],
                name="Vitals — returning/follow-up", orientation="h",
                marker_color=COOL_BLUE,
                text=[f"{v:.0f}%" if pd.notna(v) else "—" for v in df15["vitals_rate_returning_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_ret,
            ))
            fig_doc.add_trace(go.Bar(
                y=df15["label"], x=df15["notes_rate_pct"],
                name="Notes documented", orientation="h",
                marker_color=AFYA_BLUE,
                text=[f"{v:.0f}%" if pd.notna(v) else "—" for v in df15["notes_rate_pct"]],
                textposition="outside", cliponaxis=False,
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_notes,
            ))

            if med_vitals is not None:
                fig_doc.add_vline(
                    x=med_vitals,
                    line=dict(color=TEAL, width=1.5, dash="dot"),
                    annotation_text=f"Vitals median ({med_vitals:.0f}%)",
                    annotation_position="top right",
                    annotation_font=dict(size=9, color=TEAL),
                )
            if med_notes is not None:
                fig_doc.add_vline(
                    x=med_notes,
                    line=dict(color=AFYA_BLUE, width=1.5, dash="dot"),
                    annotation_text=f"Notes median ({med_notes:.0f}%)",
                    annotation_position="top left",
                    annotation_font=dict(size=9, color=AFYA_BLUE),
                )

            fig_doc.update_layout(
                barmode="group",
                height=max(420, len(df15) * 36),
                margin=dict(l=0, r=90, t=30, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Rate %", range=[0, 130],
                           showgrid=True, gridcolor="#EBF3FB"),
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            _pc(fig_doc)
            _note(
                "Vitals rates are split by visit type. A low rate on returning/follow-up patients "
                "is often expected — clinicians managing chronic patients may not re-record vitals "
                "every visit. The new-patient vitals rate is the more meaningful quality signal.",
            )

            if has_zero:
                _note(
                    "Some clinicians show 0% on all rates — this likely reflects a data capture gap "
                    "rather than clinical behaviour.",
                    w=True,
                )

            _gap(8)
            st.markdown("**Full Load Summary**")
            df15_disp = df15[["label", "days_worked", "total_visits",
                               "avg_daily_patients", "new_visit_pct",
                               "vitals_rate_new_pct", "vitals_rate_returning_pct",
                               "notes_rate_pct"]].copy()
            df15_disp["visit_mix"] = df15["new_visit_pct"].apply(
                lambda x: f"{x:.0f}% new · {100 - x:.0f}% returning" if pd.notna(x) else "—"
            )
            _pc(table_fig(
                df15_disp[["label", "days_worked", "total_visits",
                            "avg_daily_patients", "visit_mix",
                            "vitals_rate_new_pct", "vitals_rate_returning_pct",
                            "notes_rate_pct"]],
                col_labels={
                    "label": "Clinician", "days_worked": "Days",
                    "total_visits": "Visits", "avg_daily_patients": "Avg/Day",
                    "visit_mix": "Visit Mix",
                    "vitals_rate_new_pct": "Vitals % (New)",
                    "vitals_rate_returning_pct": "Vitals % (Return)",
                    "notes_rate_pct": "Notes %",
                },
                fmt={"vitals_rate_new_pct": "pct", "vitals_rate_returning_pct": "pct",
                     "notes_rate_pct": "pct"},
                height=360,
            ))
    except Exception as e:
        st.warning(f"Clinician documentation rates: {e}")

    _gap(12)

    # ── CHRONIC LTFU RATES BY CLINICIAN ───────────────────────────────────────
    _sh("Chronic Patient LTFU Rate by Clinician", mt=8)
    _note(
        "% of each clinician's chronic patients who crossed 180 days without returning. "
        "High rates may reflect gaps in follow-up scheduling or patient satisfaction. Min 5 chronic patients."
    )
    try:
        df_cl = Q.load_clinician_ltfu_rate(filters, run_query)
        if not df_cl.empty:
            df_cl["ltfu_rate_pct"] = pd.to_numeric(df_cl["ltfu_rate_pct"], errors="coerce")
            df_cl["ltfu_count"] = pd.to_numeric(df_cl["ltfu_count"], errors="coerce")
            df_cl = df_cl.reset_index(drop=True)
            df_cl["clinician_label"] = [f"Clinician {i+1}" for i in range(len(df_cl))]

            c1, c2 = st.columns([1.4, 0.6])
            with c1:
                fig_cl = go.Figure(go.Bar(
                    x=df_cl["ltfu_rate_pct"],
                    y=df_cl["clinician_label"],
                    orientation="h",
                    marker_color=[CORAL if v > 60 else ORANGE if v > 40 else TEAL
                                  for v in df_cl["ltfu_rate_pct"]],
                    text=[f"{v:.0f}%" for v in df_cl["ltfu_rate_pct"]],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>LTFU Rate: %{x:.1f}%<extra></extra>",
                ))
                fig_cl.add_vline(x=50, line=dict(color=GRAY, width=1.5, dash="dash"),
                                 annotation_text="50% reference",
                                 annotation_font=dict(size=9))
                fig_cl.update_layout(
                    height=max(300, len(df_cl) * 28),
                    margin=dict(l=0, r=60, t=20, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="% of Chronic Patients → 180d LTFU", range=[0, 115]),
                    yaxis=dict(title=""),
                )
                _pc(fig_cl)
            with c2:
                _pc(table_fig(
                    df_cl[["clinician_label", "chronic_patients_seen",
                            "ltfu_count", "ltfu_rate_pct"]].head(20),
                    col_labels={
                        "clinician_label": "Clinician",
                        "chronic_patients_seen": "Chronic Seen",
                        "ltfu_count": "LTFU",
                        "ltfu_rate_pct": "LTFU %",
                    },
                    fmt={"ltfu_rate_pct": "pct"},
                    height=420,
                ))
            worst = df_cl.iloc[0]
            _note(
                f"Highest LTFU rate: {worst['clinician_label']} — "
                f"{worst['ltfu_rate_pct']:.0f}% of their {int(worst['chronic_patients_seen']):,} "
                "chronic patients crossed 180 days. Review follow-up scheduling practices."
            )
    except Exception as e:
        st.warning(f"Clinician LTFU rate: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CLINICIAN VIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_clinician_view(filters: dict, run_query):
    # ── PAGE HEADER ───────────────────────────────────────────────────────────
    st.markdown(
        '<div style="border-bottom:2px solid #E8F0FA;padding-bottom:10px;margin-bottom:16px">'
        '<p style="font-size:11px;font-weight:800;letter-spacing:3px;text-transform:uppercase;'
        'color:#0072CE;margin:0 0 2px 0">Clinician Priority View</p>'
        '<p style="font-size:12px;color:#6B8CAE;margin:0">'
        'Priority patients &nbsp;·&nbsp; Risk signals &nbsp;·&nbsp; Visit gaps &nbsp;·&nbsp; Patient card</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not filters.get("schema") and not filters.get("clinician_id"):
        st.markdown(
            '<div style="background:#FFF8E7;border-left:4px solid #F59E0B;border-radius:4px;'
            'padding:10px 14px;font-size:12px;color:#92400E;margin-bottom:12px">'
            '⚠ Select a hospital schema and optionally your clinician ID in the sidebar.</div>',
            unsafe_allow_html=True,
        )

    try:
        df_pr = Q.load_priority_patients(filters, run_query)
    except Exception as e:
        st.warning(f"Priority patients: {e}")
        return

    if df_pr.empty:
        st.info("No patients found for the selected filters.")
        return

    for _c in ("days_since_last_visit", "is_chronic", "has_undetected_ncd",
               "had_op_to_ip", "unique_clinicians"):
        if _c in df_pr.columns:
            df_pr[_c] = pd.to_numeric(df_pr[_c], errors="coerce").fillna(0)

    # ── KPI SUMMARY ──────────────────────────────────────────────────────────
    n_high    = int((df_pr["priority_flag"] == "HIGH").sum())
    n_medium  = int((df_pr["priority_flag"] == "MEDIUM").sum())
    n_monitor = int((df_pr["priority_flag"] == "MONITOR").sum())

    kpi_cards = [
        ("Total Patients", len(df_pr), "#0072CE", "👤"),
        ("HIGH Priority",  n_high,     CORAL,     "🔴"),
        ("MEDIUM",         n_medium,   ORANGE,    "🟡"),
        ("MONITOR",        n_monitor,  TEAL,      "🟢"),
    ]
    k_cols = st.columns(4)
    for col, (label, val, color, icon) in zip(k_cols, kpi_cards):
        with col:
            st.markdown(
                f'<div style="background:white;border:1px solid #E8F0FA;border-top:3px solid {color};'
                f'border-radius:8px;padding:12px 16px">'
                f'<div style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;'
                f'color:#6B8CAE;margin-bottom:6px">{icon} {label}</div>'
                f'<div style="font-size:26px;font-weight:800;color:{color};line-height:1">{val:,}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _gap(16)

    # ── FILTER + SEARCH ───────────────────────────────────────────────────────
    fc1, fc2 = st.columns([3, 1])
    with fc1:
        search_val = st.text_input("Search patient ID or condition",
                                   placeholder="Type to filter…",
                                   label_visibility="collapsed")
    with fc2:
        pf_filter = st.selectbox("Priority filter",
                                 ["All", "HIGH only", "MEDIUM only"],
                                 label_visibility="collapsed")

    if pf_filter == "HIGH only":
        df_show = df_pr[df_pr["priority_flag"] == "HIGH"].copy()
    elif pf_filter == "MEDIUM only":
        df_show = df_pr[df_pr["priority_flag"] == "MEDIUM"].copy()
    else:
        df_show = df_pr.copy()

    if search_val.strip():
        q = search_val.strip().lower()
        mask = (
            df_show["patient"].astype(str).str.lower().str.contains(q) |
            df_show["primary_condition"].fillna("").str.lower().str.contains(q)
        )
        df_show = df_show[mask]

    # ── PRIORITY LIST — interactive row selection ─────────────────────────────
    priority_colors = {"HIGH": CORAL, "MEDIUM": ORANGE, "MONITOR": TEAL}

    def _signals_text(row):
        parts = []
        if row.get("had_op_to_ip"):            parts.append("OP→IP")
        if row.get("has_undetected_ncd"):      parts.append("NCD undetected")
        if row.get("days_since_last_visit", 0) >= 90: parts.append("Long gap")
        if row.get("unique_clinicians", 1) >= 3:      parts.append("Fragmented")
        return " · ".join(parts) if parts else "—"

    def _signals_html(row):
        chip_defs = []
        if row.get("had_op_to_ip"):            chip_defs.append(("OP→IP",           "#7C3AED", "#EDE9FE"))
        if row.get("has_undetected_ncd"):      chip_defs.append(("NCD undetected",  "#0072CE", "#EBF5FB"))
        if row.get("days_since_last_visit", 0) >= 90: chip_defs.append(("Long gap", CORAL, "#FEE2E2"))
        if row.get("unique_clinicians", 1) >= 3:      chip_defs.append(("Fragmented", ORANGE, "#FEF3C7"))
        return "".join(
            f'<span style="background:{bg};color:{fg};padding:1px 6px;border-radius:10px;'
            f'font-size:10px;font-weight:600;margin-right:3px;white-space:nowrap">{t}</span>'
            for t, fg, bg in chip_defs
        ) or '<span style="color:#9CA3AF;font-size:11px">—</span>'

    if df_show.empty:
        st.info("No patients to display with the current filter.")
        return

    st.markdown(
        '<p style="font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;'
        'color:#6B8CAE;margin-bottom:4px">Click a row to open that patient\'s card ↓</p>',
        unsafe_allow_html=True,
    )

    disp = pd.DataFrame({
        "Priority":          df_show["priority_flag"].map(
                                 {"HIGH": "🔴 HIGH", "MEDIUM": "🟡 MEDIUM"}
                             ).fillna("🟢 MONITOR"),
        "Patient ID":        df_show["patient"].astype(str),
        "Gender":            df_show["gender"].fillna("—"),
        "Age Group":         df_show["age_group"].fillna("—"),
        "Primary Condition": df_show["primary_condition"].fillna("Not recorded"),
        "Days Since Visit":  df_show["days_since_last_visit"].fillna(0).astype(int),
        "Risk Signals":      df_show.apply(_signals_text, axis=1),
        "Clinician":         df_show["current_clinician"].fillna("—").astype(str),
    }).reset_index(drop=True)

    event = st.dataframe(
        disp,
        selection_mode="single-row",
        on_select="rerun",
        use_container_width=True,
        hide_index=True,
        height=min(540, len(disp) * 35 + 50),
        column_config={
            "Priority":          st.column_config.TextColumn("Priority",          width="small"),
            "Patient ID":        st.column_config.TextColumn("Patient ID",        width="small"),
            "Gender":            st.column_config.TextColumn("Gender",            width="small"),
            "Age Group":         st.column_config.TextColumn("Age Group",         width="small"),
            "Primary Condition": st.column_config.TextColumn("Primary Condition"),
            "Days Since Visit":  st.column_config.NumberColumn("Days Since Visit", format="%d d"),
            "Risk Signals":      st.column_config.TextColumn("Risk Signals",      width="large"),
            "Clinician":         st.column_config.TextColumn("Clinician",         width="small"),
        },
    )

    # ── PATIENT DETAIL — opens when row is selected ───────────────────────────
    selected      = None
    sel_row_data  = None
    if event.selection and event.selection.rows:
        idx = event.selection.rows[0]
        if idx < len(df_show):
            selected     = str(disp.iloc[idx]["Patient ID"])
            sel_row_data = df_show.iloc[idx]

    if selected is None:
        st.markdown(
            '<div style="text-align:center;padding:24px 0;color:#9CA3AF;font-size:12px">'
            '↑ Select a patient from the list above to view their full clinical record</div>',
            unsafe_allow_html=True,
        )
        return

    sel_schema = str(sel_row_data.get("source_schema") or filters.get("schema") or "")

    st.markdown(
        '<div style="border-top:2px solid #E8F0FA;margin:16px 0 12px 0;display:flex;'
        'align-items:center;gap:8px;padding-top:12px">'
        '<span style="font-size:18px;color:#0072CE">↓</span>'
        '<span style="font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;'
        'color:#0072CE">Patient Detail</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    flag  = str(sel_row_data.get("priority_flag", ""))
    pc    = priority_colors.get(flag, TEAL)
    days  = int(float(sel_row_data.get("days_since_last_visit") or 0))
    clin  = str(sel_row_data.get("current_clinician") or "Unknown")
    cond  = str(sel_row_data.get("primary_condition") or "Not recorded")

    st.markdown(
        f'<div style="background:linear-gradient(135deg,{pc}18 0%,#F8FAFF 100%);'
        f'border:1px solid {pc}55;border-left:5px solid {pc};'
        f'border-radius:10px;padding:14px 20px;margin-bottom:16px">'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px">'
        f'<span style="background:{pc};color:white;padding:3px 10px;border-radius:5px;'
        f'font-size:11px;font-weight:800;letter-spacing:0.5px">{flag}</span>'
        f'<span style="font-size:16px;font-weight:800;color:#111827">Patient {selected}</span>'
        f'<span style="font-size:12px;color:#374151;background:white;border:1px solid #E5E7EB;'
        f'padding:2px 10px;border-radius:4px">{cond}</span>'
        f'</div>'
        f'<div style="display:flex;gap:20px;flex-wrap:wrap">'
        f'<span style="font-size:11px;color:#6B7280">'
        f'👤 {sel_row_data.get("gender","—")} · {sel_row_data.get("age_group","—")}</span>'
        f'<span style="font-size:11px;color:#6B7280">'
        f'📅 Last seen <b style="color:{CORAL if days>=90 else "#374151"}">{days} days ago</b></span>'
        f'<span style="font-size:11px;color:#6B7280">🩺 Clinician {clin}</span>'
        f'</div>'
        f'<div style="margin-top:8px">{_signals_html(sel_row_data)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if sel_schema:
        _render_patient_card(selected, sel_schema, run_query)


def _sec_header(icon: str, title: str, subtitle: str = ""):
    sub = f'<span style="font-size:11px;color:#6B8CAE;font-weight:400;margin-left:8px">{subtitle}</span>' if subtitle else ""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:16px 0 10px 0;'
        f'padding-bottom:8px;border-bottom:1px solid #E8F0FA">'
        f'<span style="font-size:16px">{icon}</span>'
        f'<span style="font-size:13px;font-weight:700;color:#111827">{title}</span>'
        f'{sub}</div>',
        unsafe_allow_html=True,
    )


def _stat_card(label: str, value: str, note: str = "", color: str = "#0072CE",
               full_width: bool = False):
    w = "100%" if full_width else "auto"
    st.markdown(
        f'<div style="background:#F8FAFC;border:1px solid #E8F0FA;border-radius:8px;'
        f'padding:12px 16px;min-width:100px;width:{w}">'
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:1px;color:#6B8CAE;margin-bottom:4px">{label}</div>'
        f'<div style="font-size:22px;font-weight:800;color:{color};line-height:1">{value}</div>'
        + (f'<div style="font-size:10px;color:#9CA3AF;margin-top:3px">{note}</div>' if note else "")
        + f'</div>',
        unsafe_allow_html=True,
    )


def _render_patient_card(patient_id: str, source_schema: str, run_query):
    _gap(4)

    tab_cadence, tab_illness, tab_vitals, tab_labs, tab_meds = st.tabs([
        "📅  Visit Cadence", "🗂  Illness History", "❤️  Vitals", "🔬  Lab Tests", "💊  Medications"
    ])

    # ── TAB 1: VISIT CADENCE & RETURN PATTERN ────────────────────────────────
    with tab_cadence:
        _sec_header("📅", "Visit Cadence & Return Pattern",
                    "Gap = days since previous visit")
        try:
            cl_cad = Q.load_patient_visit_cadence(patient_id, source_schema, run_query)
            if not cl_cad.empty:
                cl_cad["gap_days"]   = pd.to_numeric(cl_cad["gap_days"],   errors="coerce")
                cl_cad["visit_date"] = pd.to_datetime(cl_cad["visit_date"], errors="coerce")
                # Floor to date so the timeline shows dates not intra-second timestamps
                cl_cad["visit_date"] = cl_cad["visit_date"].dt.floor("D")

                total_visits = len(cl_cad)
                avg_gap = cl_cad["gap_days"].dropna().mean()
                max_gap = cl_cad["gap_days"].dropna().max()
                last_dt = cl_cad["visit_date"].max()

                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    _stat_card("Total Visits", str(total_visits))
                with s2:
                    _stat_card("Avg Gap",
                               f"{avg_gap:.0f}d" if pd.notna(avg_gap) else "—",
                               color=CORAL if avg_gap and avg_gap > 90 else TEAL)
                with s3:
                    _stat_card("Longest Gap",
                               f"{max_gap:.0f}d" if pd.notna(max_gap) else "—",
                               color=CORAL if max_gap and max_gap > 180 else AFYA_BLUE)
                with s4:
                    _stat_card("Last Visit",
                               last_dt.strftime("%d %b %Y") if pd.notna(last_dt) else "—")

                _gap(12)
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("**Visit timeline**")
                    fig_cad = go.Figure()
                    ip_mask = cl_cad["visit_type"] == "Inpatient"
                    fig_cad.add_trace(go.Scatter(
                        x=cl_cad.loc[~ip_mask, "visit_date"],
                        y=[1] * (~ip_mask).sum(),
                        mode="markers", name="Outpatient",
                        marker=dict(color=TEAL, size=12, symbol="circle"),
                        hovertemplate="<b>%{x|%d %b %Y}</b><br>Outpatient<extra></extra>",
                    ))
                    fig_cad.add_trace(go.Scatter(
                        x=cl_cad.loc[ip_mask, "visit_date"],
                        y=[1] * ip_mask.sum(),
                    mode="markers", name="Inpatient",
                    marker=dict(color=CORAL, size=14, symbol="diamond"),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Inpatient<extra></extra>",
                ))
                fig_cad.update_layout(
                    height=100, margin=dict(l=0, r=0, t=10, b=20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=True, xaxis=dict(title="", tickformat="%b %Y"),
                    yaxis=dict(visible=False),
                    legend=dict(orientation="h", y=1.4, xanchor="right", x=1),
                )
                _pc(fig_cad)

                with c_r:
                    st.markdown("**Gap between visits (days)**")
                    gap_df = cl_cad.dropna(subset=["gap_days"]).copy()
                    if not gap_df.empty:
                        gap_colors = [CORAL if g > 90 else ORANGE if g > 60 else TEAL
                                      for g in gap_df["gap_days"]]
                        fig_gap = go.Figure(go.Bar(
                            x=gap_df["visit_date"],
                            y=gap_df["gap_days"],
                            marker_color=gap_colors,
                            text=[f"{int(g)}d" for g in gap_df["gap_days"]],
                            textposition="outside",
                            hovertemplate="<b>%{x|%d %b %Y}</b><br>Gap: %{y:.0f} days<extra></extra>",
                        ))
                        fig_gap.update_layout(
                            height=220, margin=dict(l=0, r=0, t=20, b=0),
                            plot_bgcolor="white", paper_bgcolor="white",
                            yaxis=dict(title="Days since previous visit", rangemode="tozero"),
                            xaxis=dict(title="", tickformat="%b %Y"),
                            showlegend=False,
                        )
                        _pc(fig_gap)
                    else:
                        st.info("Only one visit recorded — no gaps to show.")
            else:
                st.info("No visit cadence data found for this patient.")
        except Exception as e:
            st.warning(f"Visit cadence: {e}")

    # ── TAB 2: ILLNESS HISTORY ────────────────────────────────────────────────
    with tab_illness:
        _sec_header("🗂", "Illness History",
                    "Most recent first · Red = inpatient · Extended LOS flagged")
        try:
            cl4 = Q.load_patient_illness_history(patient_id, source_schema, run_query)
            if not cl4.empty:
                cl4["los_days"]   = pd.to_numeric(cl4.get("los_days"), errors="coerce")
                cl4["visit_date"] = pd.to_datetime(cl4["visit_date"], errors="coerce")

                _EXP_LOS = {
                    "cardiovascular": 3, "diabetes": 4, "neurolog": 5,
                    "mental": 7, "respiratory": 3, "renal": 5,
                }
                def _exp_los(burden):
                    b = str(burden).lower()
                    for k, v in _EXP_LOS.items():
                        if k in b:
                            return v
                    return 3

                cl4["exp_los"] = cl4.get("disease_burden_group_1", pd.Series()).apply(_exp_los)
                cl4["los_status"] = cl4.apply(
                    lambda r: (
                        "Extended" if pd.notna(r["los_days"]) and r["los_days"] > r["exp_los"] + 1
                        else "Normal" if pd.notna(r["los_days"]) else "—"
                    ), axis=1
                )

                if "disease_group" in cl4.columns:
                    dx_counts = cl4["disease_group"].dropna().value_counts()
                    recurring = dx_counts[dx_counts >= 3].index.tolist()
                    if recurring:
                        st.markdown(
                            f'<div style="background:#FEF3C7;border-left:4px solid {ORANGE};'
                            f'border-radius:4px;padding:8px 14px;font-size:12px;margin-bottom:10px">'
                            f'⚠ <b>Recurring diagnoses (3+ visits):</b> '
                            f'{", ".join(str(r) for r in recurring[:4])} — '
                            f'may indicate an unresolved underlying risk factor.</div>',
                            unsafe_allow_html=True,
                        )

                if "disease_group" in cl4.columns and "visit_date" in cl4.columns:
                    first_last = (
                        cl4.dropna(subset=["disease_group"])
                        .groupby("disease_group")["visit_date"]
                        .agg(first_seen="min", last_seen="max", times="count")
                        .reset_index().sort_values("times", ascending=False).head(8)
                    )
                    first_last["first_seen"] = first_last["first_seen"].dt.strftime("%d %b %Y")
                    first_last["last_seen"]  = first_last["last_seen"].dt.strftime("%d %b %Y")
                    st.markdown(
                        '<p style="font-size:11px;font-weight:700;letter-spacing:1px;'
                        'text-transform:uppercase;color:#6B8CAE;margin-bottom:6px">'
                        'Condition history</p>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        first_last.rename(columns={
                            "disease_group": "Diagnosis", "first_seen": "First Seen",
                            "last_seen": "Last Seen", "times": "Visits",
                        }),
                        use_container_width=True, hide_index=True, height=260,
                        column_config={
                            "Visits": st.column_config.NumberColumn("Visits", format="%d"),
                        },
                    )
                    _gap(12)

                st.markdown(
                    '<p style="font-size:11px;font-weight:700;letter-spacing:1px;'
                    'text-transform:uppercase;color:#6B8CAE;margin-bottom:6px">'
                    'Full visit timeline</p>',
                    unsafe_allow_html=True,
                )
                disp4 = cl4.copy()
                disp4["Visit Date"] = disp4["visit_date"].dt.strftime("%d %b %Y")
                disp4["Type"] = disp4["visit_type"].apply(
                    lambda v: "🏥 Inpatient" if str(v).lower() == "inpatient" else "Outpatient"
                )
                disp4["LOS"] = disp4.apply(
                    lambda r: (f"{int(r['los_days'])}d — {r['los_status']}"
                               if pd.notna(r["los_days"]) else "—"), axis=1
                )
                cols4 = [c for c in ["Visit Date", "disease_group", "disease_burden_group_1",
                                     "Type", "LOS", "payer"] if c in disp4.columns]
                st.dataframe(
                    disp4[cols4].rename(columns={
                        "disease_group": "Diagnosis", "disease_burden_group_1": "Burden Group",
                        "payer": "Payer",
                    }).head(40),
                    use_container_width=True, hide_index=True,
                    height=min(500, len(cl4.head(40)) * 35 + 40),
                )
            else:
                st.info("No illness history found for this patient.")
        except Exception as e:
            st.warning(f"Illness history: {e}")

    # ── TAB 3: VITALS ─────────────────────────────────────────────────────────
    with tab_vitals:
        _sec_header("❤️", "Vitals Trend",
                    "Last 6 readings · Trajectory over consecutive visits")
        try:
            cl2 = Q.load_patient_vitals_trend(patient_id, source_schema, run_query)
            if not cl2.empty:
                row0   = cl2.iloc[0]
                signal = str(row0.get("clinical_signal", ""))
                sig_color = (CORAL if "elevated" in signal.lower() or "rising" in signal.lower()
                             else TEAL if "expected" in signal.lower() else ORANGE)
                st.markdown(
                    f'<div style="background:{sig_color};color:white;border-radius:8px;'
                    f'padding:10px 16px;font-size:13px;font-weight:700;margin-bottom:16px;'
                    f'letter-spacing:0.3px">🩺 {signal}</div>',
                    unsafe_allow_html=True,
                )
                cv1, cv2, cv3 = st.columns(3)
                for col, label, val_col, trend_col, vals_col in [
                    (cv1, "BP Systolic",  "recent_sys",   "systolic_trend",  "bp_systolic"),
                    (cv2, "BP Diastolic", "recent_dia",   "diastolic_trend", "bp_diastolic"),
                    (cv3, "Blood Sugar",  "recent_sugar",  "sugar_trend",     "blood_sugar"),
                ]:
                    trend = str(row0.get(trend_col, ""))
                    val   = row0.get(val_col)
                    tc    = (TEAL if trend == "Improving" else CORAL if trend == "Worsening"
                             else AFYA_BLUE)
                    arrow = "↑" if trend == "Worsening" else "↓" if trend == "Improving" else "→"
                    val_str = f"{float(val):.0f}" if val and str(val) not in ("nan","None","") else "—"
                    with col:
                        st.markdown(
                            f'<div style="background:white;border:1px solid #E8F0FA;'
                            f'border-top:3px solid {tc};border-radius:8px;padding:14px 16px">'
                            f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
                            f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">'
                            f'{label}</div>'
                            f'<div style="font-size:28px;font-weight:800;color:{tc};line-height:1">'
                            f'{val_str}</div>'
                            f'<div style="font-size:12px;color:{tc};margin-top:4px;font-weight:600">'
                            f'{arrow} {trend if trend else "Insufficient data"}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        spark_vals = cl2[vals_col].dropna().tolist()
                        if len(spark_vals) >= 2:
                            _pc(sparkline(spark_vals[::-1], trend=trend, height=60))
                        else:
                            st.markdown(
                                '<p style="font-size:10px;color:#9CA3AF;margin:4px 0 0 0">'
                                '→ Insufficient data</p>',
                                unsafe_allow_html=True,
                            )
            else:
                st.info("No vitals recorded for this patient.")
        except Exception as e:
            st.warning(f"Vitals: {e}")

    # ── TAB 4: LAB TESTS ──────────────────────────────────────────────────────
    with tab_labs:
        _sec_header("🔬", "Lab Tests & Investigations",
                    "Abnormal flags highlighted · TAT = minutes order → result")
        try:
            cl_lab = Q.load_patient_lab_tests(patient_id, source_schema, run_query)
            if not cl_lab.empty:
                cl_lab["turnaround_mins"] = pd.to_numeric(cl_lab["turnaround_mins"], errors="coerce")
                cl_lab["test_date"]       = pd.to_datetime(cl_lab["test_date"], errors="coerce")

                abnormal_count = cl_lab["flag"].notna().sum() if "flag" in cl_lab.columns else 0
                if abnormal_count > 0:
                    st.markdown(
                        f'<div style="background:#FEE2E2;border-left:4px solid {CORAL};'
                        f'border-radius:4px;padding:8px 14px;font-size:12px;font-weight:600;'
                        f'color:#991B1B;margin-bottom:12px">'
                        f'⚠ {int(abnormal_count)} flagged result{"s" if abnormal_count > 1 else ""}. '
                        f'Review alert level for critical values.</div>',
                        unsafe_allow_html=True,
                    )

                type_summary = (
                    cl_lab.groupby("investigation_type")
                    .agg(count=("procedure_name", "count"),
                         flagged=("flag", lambda x: x.notna().sum()))
                    .reset_index().sort_values("count", ascending=False)
                )
                if not type_summary.empty:
                    st.markdown(
                        '<p style="font-size:11px;font-weight:700;letter-spacing:1px;'
                        'text-transform:uppercase;color:#6B8CAE;margin-bottom:6px">'
                        'Tests by category</p>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        type_summary.rename(columns={
                            "investigation_type": "Type",
                            "count": "Tests Done", "flagged": "Flagged",
                        }),
                        use_container_width=True, hide_index=True, height=160,
                    )
                    _gap(12)

                disp_lab = cl_lab.copy()
                disp_lab["Date"] = disp_lab["test_date"].dt.strftime("%d %b %Y")
                disp_lab["TAT"]  = disp_lab["turnaround_mins"].apply(
                    lambda v: f"{int(v)}m" if pd.notna(v) else "—"
                )
                disp_lab["Flag"] = disp_lab.apply(
                    lambda r: (r.get("alert_level") or r.get("flag") or ""), axis=1
                )
                st.markdown(
                    '<p style="font-size:11px;font-weight:700;letter-spacing:1px;'
                    'text-transform:uppercase;color:#6B8CAE;margin-bottom:6px">'
                    'All tests (most recent first)</p>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    disp_lab[["Date", "investigation_type", "procedure_name", "TAT", "Flag"]].rename(
                        columns={"investigation_type": "Type", "procedure_name": "Test"}
                    ).head(50),
                    use_container_width=True, hide_index=True,
                    height=min(480, len(cl_lab.head(50)) * 35 + 40),
                )
            else:
                st.info("No lab tests found for this patient.")
        except Exception as e:
            st.warning(f"Lab tests: {e}")

    # ── TAB 5: MEDICATIONS ────────────────────────────────────────────────────
    with tab_meds:
        _sec_header("💊", "Medication History & Changes",
                    "'Changed' = different drug from prior event")
        try:
            cl5 = Q.load_patient_medication_change_timeline(patient_id, source_schema, run_query)
            if not cl5.empty:
                cl5["is_new_drug"]        = pd.to_numeric(cl5.get("is_new_drug", 0), errors="coerce")
                cl5["prescription_date"]  = pd.to_datetime(cl5["prescription_date"], errors="coerce")

                change_count = int(cl5["is_new_drug"].sum())
                if change_count > 0:
                    st.markdown(
                        f'<div style="background:#EFF6FF;border-left:4px solid {AFYA_BLUE};'
                        f'border-radius:4px;padding:10px 14px;font-size:12px;margin-bottom:12px">'
                        f'💊 <b>{change_count} medication change{"s" if change_count > 1 else ""} detected.</b> '
                        f'Cross-check with the Vitals tab to see if values stabilised after each switch.'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                cl5["Date"]   = cl5["prescription_date"].dt.strftime("%d %b %Y")
                cl5["Status"] = cl5.apply(
                    lambda r: (
                        f"Changed from: {r.get('prev_drug','?')}"
                        if r["is_new_drug"] == 1 and pd.notna(r.get("prev_drug"))
                        else "New" if r["is_new_drug"] == 1
                        else "Continued"
                    ), axis=1
                )
                cl5["Gap"] = cl5["days_since_last_prescription"].apply(
                    lambda v: f"{int(float(v))}d" if pd.notna(v) else "—"
                )
                st.markdown(
                    '<p style="font-size:11px;font-weight:700;letter-spacing:1px;'
                    'text-transform:uppercase;color:#6B8CAE;margin-bottom:6px">'
                    'Prescription timeline</p>',
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    cl5[["Date", "drug_name", "Status", "Gap"]].rename(
                        columns={"drug_name": "Drug", "Gap": "Days Since Last Rx"}
                    ).head(30),
                    use_container_width=True, hide_index=True,
                    height=min(480, len(cl5.head(30)) * 35 + 40),
                    column_config={
                        "Status": st.column_config.TextColumn("Status", width="large"),
                    },
                )

                _gap(12)
                st.markdown(
                    '<p style="font-size:11px;font-weight:700;letter-spacing:1px;'
                    'text-transform:uppercase;color:#6B8CAE;margin-bottom:6px">'
                    'Current active medications</p>',
                    unsafe_allow_html=True,
                )
                try:
                    cl3 = Q.load_medication_continuity(patient_id, source_schema, run_query)
                    if not cl3.empty:
                        gaps = int(cl3["is_gap"].sum())
                        if gaps > 0:
                            st.markdown(
                                f'<div style="background:#FEE2E2;border-left:4px solid {CORAL};'
                                f'border-radius:4px;padding:8px 14px;font-size:12px;'
                                f'font-weight:600;color:#991B1B;margin-bottom:8px">'
                                f'⚠ {gaps} medication gap{"s" if gaps > 1 else ""} — '
                                f'expected drug class not recently prescribed</div>',
                                unsafe_allow_html=True,
                            )
                        st.dataframe(
                            cl3[["condition", "expected_drug_class", "active_drug",
                                  "days_since_prescribed", "continuity_status"]].rename(columns={
                                "condition": "Condition",
                                "expected_drug_class": "Expected Class",
                                "active_drug": "Active Drug",
                                "days_since_prescribed": "Days Since Rx",
                                "continuity_status": "Status",
                            }),
                            use_container_width=True, hide_index=True,
                            height=min(300, len(cl3) * 35 + 40),
                        )
                    else:
                        st.info("No current medication continuity data.")
                except Exception as e_cl3:
                    st.warning(f"Medication continuity: {e_cl3}")
            else:
                st.info("No prescription history found for this patient.")
        except Exception as e:
            st.warning(f"Medication history: {e}")