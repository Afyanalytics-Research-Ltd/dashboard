"""
Diagnostics — pages/2_diagnostics.py
Single-layout, no tabs. 5-section chain.
Question: Is diagnostics supporting patient flow, or becoming a bottleneck?

Imaging TAT: V2 + V1 2024+ only (same stage: order→scan completion, Inv 137).
Pre-2024 V1 imaging = physician review, incompatible stage.
Lab TAT: V1+V2 unified (consistent definition both systems).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Diagnostics · SPH Ortho",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import (
    apply_theme, render_sidebar, kpi_card,
    page_header, COLORS,
)

from facility_operations.dashboard.queries import (
    q_lab_summary, q_imaging_summary,
    q_lab_mom, q_imaging_mom,
    q_imaging_modality_completion,
    q_diag_demand_monthly,
    q_imaging_modality_tat,
    q_lab_tat_by_test, q_lab_chain_tat,
    q_lab_collect_wait_by_test,
    q_imaging_tat_by_hour,
)

apply_theme()
render_sidebar("diagnostics")

page_header(
    "Diagnostics",
    subtitle="Is diagnostics supporting patient flow, or becoming a bottleneck?",
)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _section(label):
    st.markdown(
        f'<div style="border-top:1.5px solid #D6E4F0;margin:40px 0 18px 0;padding-top:14px">'
        f'<span style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:2.5px">{label}</span></div>',
        unsafe_allow_html=True,
    )

def _caption(text):
    st.markdown(
        f'<div style="font-size:9px;color:#9BAEC8;margin-top:-6px;margin-bottom:4px">{text}</div>',
        unsafe_allow_html=True,
    )

def _note(text):
    st.markdown(
        f'<div style="font-size:9px;color:#9BAEC8;font-style:italic;'
        f'margin-top:4px;margin-bottom:8px">{text}</div>',
        unsafe_allow_html=True,
    )

def _metric_card(headline, value, sub, color=None):
    c = color or COLORS["primary"]
    st.markdown(
        f'<div style="background:#F0F5FA;border-left:4px solid {c};border-radius:6px;'
        f'padding:14px 18px;margin-bottom:12px">'
        f'<div style="font-size:10px;font-weight:600;color:{COLORS["muted"]};'
        f'text-transform:uppercase;letter-spacing:1px">{headline}</div>'
        f'<div style="font-size:26px;font-weight:800;color:{c};margin:4px 0 2px 0">{value}</div>'
        f'<div style="font-size:11px;color:{COLORS["dark"]}">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── load all data up front ─────────────────────────────────────────────────────

demand_df    = q_diag_demand_monthly()
lab_kpis     = q_lab_summary().iloc[0]
img_kpis     = q_imaging_summary().iloc[0]
_lab_mom_df  = q_lab_mom()
_img_mom_df  = q_imaging_mom()
_lab_mom     = _lab_mom_df.iloc[0] if not _lab_mom_df.empty else None
_img_mom     = _img_mom_df.iloc[0] if not _img_mom_df.empty else None
mod_compl_df = q_imaging_modality_completion()
mod_tat_df   = q_imaging_modality_tat()
lab_test_df  = q_lab_tat_by_test()
chain_kpis   = q_lab_chain_tat().iloc[0]
collect_wait_df = q_lab_collect_wait_by_test()
hour_df      = q_imaging_tat_by_hour()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DEMAND
# ══════════════════════════════════════════════════════════════════════════════

_section("1 · Diagnostic Load")

# Summary KPIs
tot_lab     = int(demand_df["LAB_ORDERS"].sum())
tot_img     = int(demand_df["IMAGING_ORDERS"].sum())
valid_months = demand_df[demand_df["OPD_VISITS"] > 0]
avg_lab_100 = round(float(valid_months["LAB_PER_100"].mean()), 1) if not valid_months.empty else 0.0
avg_img_100 = round(float(valid_months["IMAGING_PER_100"].mean()), 1) if not valid_months.empty else 0.0

def _mom_card(col, title, mom_row, fallback_val):
    if mom_row is None:
        with col:
            kpi_card(title, f"{fallback_val:,}", "last complete month")
        return
    month_name = pd.to_datetime(mom_row["LAST_MONTH"]).strftime("%b %Y")
    val        = f"{int(mom_row['LAST_MONTH_ORDERS']):,}"
    pct        = float(mom_row["MOM_PCT"]) if mom_row["MOM_PCT"] is not None else None
    if pct is None:
        sub   = month_name
        color = COLORS["primary"]
    elif pct >= 0:
        sub   = (f'<span style="color:{COLORS["success"]};font-weight:700">▲ {abs(pct):.1f}%</span>'
                 f' vs prior month · {month_name}')
        color = COLORS["success"]
    else:
        sub   = (f'<span style="color:{COLORS["danger"]};font-weight:700">▼ {abs(pct):.1f}%</span>'
                 f' vs prior month · {month_name}')
        color = COLORS["danger"]
    with col:
        kpi_card(title, val, sub, color=color)

k1, k2, k3, k4 = st.columns(4)
_mom_card(k1, "Lab Orders", _lab_mom, tot_lab)
_mom_card(k2, "Imaging Orders", _img_mom, tot_img)
with k3:
    kpi_card("Lab Orders / 100 OPD", str(avg_lab_100), "Monthly average")
with k4:
    kpi_card("Imaging / 100 OPD", str(avg_img_100), "Monthly average")

# Monthly trend — stacked bars (lab + imaging) + OPD line on secondary axis
fig_demand = go.Figure()
fig_demand.add_trace(go.Bar(
    x=demand_df["MONTH"], y=demand_df["LAB_ORDERS"],
    name="Lab orders",
    marker_color=COLORS["primary"],
    opacity=0.85,
    yaxis="y",
))
fig_demand.add_trace(go.Bar(
    x=demand_df["MONTH"], y=demand_df["IMAGING_ORDERS"],
    name="Imaging orders",
    marker_color=COLORS["coral"],
    opacity=0.85,
    yaxis="y",
))
fig_demand.add_trace(go.Scatter(
    x=demand_df["MONTH"], y=demand_df["OPD_VISITS"],
    name="OPD visits",
    mode="lines+markers",
    line=dict(color=COLORS["muted"], width=2, dash="dot"),
    marker=dict(size=4, color=COLORS["muted"]),
    yaxis="y2",
))
fig_demand.update_layout(
    height=280,
    barmode="stack",
    paper_bgcolor="#fff", plot_bgcolor="#fff",
    font=dict(family="Montserrat", color=COLORS["dark"]),
    margin=dict(l=0, r=60, t=10, b=30),
    legend=dict(orientation="h", y=1.10, font=dict(size=9)),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=9, color=COLORS["muted"])),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=9, color=COLORS["muted"]),
               title="Diagnostic orders"),
    yaxis2=dict(overlaying="y", side="right",
                tickfont=dict(size=9, color=COLORS["muted"]),
                title="OPD visits", showgrid=False),
)
fig_demand.add_shape(
    type="line",
    x0=pd.Timestamp("2025-02-01"), x1=pd.Timestamp("2025-02-01"),
    y0=0, y1=1, yref="paper",
    line=dict(color=COLORS["muted"], width=1, dash="dot"),
)
fig_demand.add_annotation(
    x=pd.Timestamp("2025-02-01"), y=0.97, yref="paper",
    text="V2 start", showarrow=False,
    font=dict(size=8, color=COLORS["muted"]), xanchor="left",
)
st.plotly_chart(fig_demand, use_container_width=True)
_caption(
    "Monthly orders by type (stacked bars, left axis) vs OPD visit volume (dotted line, right axis) · V1+V2"
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMPLETION
# ══════════════════════════════════════════════════════════════════════════════

_section("2 · Completion — Can clinicians rely on diagnostics?")

col_lab_compl, col_img_compl = st.columns(2)

with col_lab_compl:
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;color:{COLORS["dark"]};'
        f'margin-bottom:10px">Lab Results</div>',
        unsafe_allow_html=True,
    )
    lc1, lc2 = st.columns(2)
    _pending_pct = round(100 - float(lab_kpis['COMPLETION_PCT']), 1)
    with lc1:
        kpi_card("Result Completion Rate", f"{lab_kpis['COMPLETION_PCT']}%",
                 f"{int(lab_kpis['RESULTED']):,} orders with a result on file")
    with lc2:
        kpi_card("Orders Without Result", f"{_pending_pct}%",
                 f"{int(lab_kpis['UNRESULTED']):,} orders with no result recorded")
    _note(
        "Completion uses order status flag (consistent V1+V2). "
        "Result timestamp coverage is lower in V2 (~52%) — not a completion issue (Inv 140)."
    )

with col_img_compl:
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;color:{COLORS["dark"]};'
        f'margin-bottom:10px">Imaging Results by Modality</div>',
        unsafe_allow_html=True,
    )
    df_mc = mod_compl_df[mod_compl_df["MODALITY_GROUP"].notna()].sort_values("COMPLETION_PCT")
    bar_colors_mc = [
        COLORS["danger"] if r < 30 else COLORS["warning"] if r < 60 else COLORS["success"]
        for r in df_mc["COMPLETION_PCT"]
    ]
    fig_compl = go.Figure(go.Bar(
        x=df_mc["COMPLETION_PCT"],
        y=df_mc["MODALITY_GROUP"],
        orientation="h",
        marker_color=bar_colors_mc,
        text=[f"{r}%  ({n:,})" for r, n in
              zip(df_mc["COMPLETION_PCT"], df_mc["ORDERS"])],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig_compl.update_layout(
        height=210,
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Montserrat", color=COLORS["dark"]),
        margin=dict(l=0, r=110, t=10, b=10),
        xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=9, color=COLORS["muted"]),
                   range=[0, 130], title="Completion %"),
        yaxis=dict(tickfont=dict(size=10, color=COLORS["dark"])),
        showlegend=False,
    )
    st.plotly_chart(fig_compl, use_container_width=True)
    _caption("V1+V2 · green ≥80% · amber 30–79% · red <30%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TIMELINESS
# ══════════════════════════════════════════════════════════════════════════════

_section("3 · Timeliness — Where are orders delayed?")

# Imaging modality table
st.markdown(
    f'<div style="font-size:11px;font-weight:700;color:{COLORS["dark"]};'
    f'margin-bottom:8px">Imaging TAT by Modality</div>',
    unsafe_allow_html=True,
)

if not mod_tat_df.empty:
    _max_idx = mod_tat_df["P50_MINS"].idxmax()

    def _row_color(row):
        bg = "#FFF3CD" if row.name == _max_idx else "#FFFFFF"
        return [f"background-color: {bg}"] * len(row)

    display_tat = mod_tat_df.rename(columns={
        "MODALITY_GROUP": "Modality",
        "ORDERS": "Orders",
        "P50_MINS": "Median (min)",
        "WITHIN_60": "Within 60 min",
        "PCT_WITHIN_60": "% Within 60 min",
    })[["Modality", "Orders", "Median (min)", "Within 60 min", "% Within 60 min"]]
    styled_tat = display_tat.style.apply(_row_color, axis=1).format({
        "Orders":        "{:,.0f}",
        "Median (min)":  "{:.0f}",
        "Within 60 min": "{:,.0f}",
        "% Within 60 min": "{:.1f}%",
    })
    st.dataframe(styled_tat, use_container_width=True, hide_index=True)
    _caption(
        "V2 + V1 2024+ only (order→scan completion, Inv 137) · capped at 1440 min (Issue 97) · "
        "highlighted row = slowest modality by median TAT"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Lab chain breakdown — chart full width, cards below
st.markdown(
    f'<div style="font-size:11px;font-weight:700;color:{COLORS["dark"]};'
    f'margin-bottom:4px">Lab Delay — Queue or Lab? (V2)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="font-size:11px;color:{COLORS["dark"]};line-height:1.6;margin-bottom:8px">'
    f'Median minutes from order to result, split into two segments: '
    f'waiting for specimen collection (amber) vs lab processing (blue). '
    f'Sorted by collection wait — longest queue at top.'
    f'</div>',
    unsafe_allow_html=True,
)
if not collect_wait_df.empty:
    _cw = collect_wait_df.copy()
    _cw = _cw.sort_values("P50_ORDER_TO_COLLECT", ascending=True)
    _cw["TOTAL"] = _cw["P50_ORDER_TO_COLLECT"] + _cw["P50_COLLECT_TO_RESULT"]

    fig_cw = go.Figure()
    fig_cw.add_trace(go.Bar(
        name="Collection wait (queue)",
        x=_cw["P50_ORDER_TO_COLLECT"],
        y=_cw["TEST_NAME"],
        orientation="h",
        marker_color=COLORS["warning"],
        text=[f"{v} min" for v in _cw["P50_ORDER_TO_COLLECT"]],
        textposition="inside",
        textfont=dict(size=9, color="#fff"),
        hovertemplate="<b>%{y}</b><br>Collection wait: %{x} min<extra></extra>",
    ))
    fig_cw.add_trace(go.Bar(
        name="Lab processing",
        x=_cw["P50_COLLECT_TO_RESULT"],
        y=_cw["TEST_NAME"],
        orientation="h",
        marker_color=COLORS["primary"],
        text=[f"{v} min" for v in _cw["P50_COLLECT_TO_RESULT"]],
        textposition="inside",
        textfont=dict(size=9, color="#fff"),
        hovertemplate="<b>%{y}</b><br>Lab processing: %{x} min<extra></extra>",
    ))
    # Invisible trace — anchors total label outside end of each bar
    fig_cw.add_trace(go.Bar(
        name="",
        x=[0.01] * len(_cw),
        y=_cw["TEST_NAME"],
        orientation="h",
        base=_cw["TOTAL"].tolist(),
        marker_color="rgba(0,0,0,0)",
        marker_line_width=0,
        text=[f"Total TAT: {t} min" for t in _cw["TOTAL"]],
        textposition="outside",
        textfont=dict(size=9, color=COLORS["dark"]),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig_cw.update_layout(
        barmode="stack",
        height=max(300, len(_cw) * 30),
        paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(family="Montserrat", color=COLORS["dark"]),
        margin=dict(l=0, r=120, t=24, b=10),
        xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=9, color=COLORS["muted"]),
                   title="Median (minutes)"),
        yaxis=dict(tickfont=dict(size=9, color=COLORS["dark"])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=9)),
    )
    st.plotly_chart(fig_cw, use_container_width=True)
    _caption(
        "V2 only · Median minutes · amber = collection queue wait · "
        "blue = lab processing · ≥ 20 timed orders per test"
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div style="font-size:11px;font-weight:700;color:{COLORS["dark"]};'
    f'margin-bottom:8px">V2 Lab: Median TAT by Stage</div>',
    unsafe_allow_html=True,
)
col_chain_a, col_chain_b = st.columns(2)
with col_chain_a:
    _metric_card(
        "Order → Specimen Collected",
        f"{int(chain_kpis['P50_ORDER_TO_COLLECT'])} min",
        f"Median TAT · 1 in 10 orders waits >{int(chain_kpis['P90_ORDER_TO_COLLECT'])} min",
        COLORS["warning"],
    )
with col_chain_b:
    _metric_card(
        "Specimen Collected → Result",
        f"{int(chain_kpis['P50_COLLECT_TO_RESULT'])} min",
        f"Median TAT · 1 in 10 takes >{int(chain_kpis['P90_COLLECT_TO_RESULT'])} min",
        COLORS["primary"],
    )
_note(
    f"V2 only · {int(chain_kpis['FULL_CHAIN']):,} orders with full 3-stage chain "
    f"(of {int(chain_kpis['WITH_COLLECTION']):,} with collection timestamp)"
)



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OPERATIONAL IMPACT
# ══════════════════════════════════════════════════════════════════════════════

_section("4 · Operational Impact")

col_narr, col_before_after = st.columns([3, 2])

with col_narr:
    st.markdown(
        f'<div style="background:#F0F9FF;border-left:4px solid {COLORS["primary"]};'
        f'border-radius:6px;padding:16px 20px;margin-bottom:16px">'
        f'<div style="font-size:12px;font-weight:700;color:{COLORS["dark"]};margin-bottom:8px">'
        f'Same-Day Imaging Reduced Avoidable Return Visits</div>'
        f'<div style="font-size:12px;color:{COLORS["dark"]};line-height:1.7">'
        f'<b>Before 2024</b>, imaging results were reported the following day — median 42 hours '
        f'from order to result. Patients whose scans were not ready left and returned. '
        f'<b>From 2024</b>, same-day reporting was introduced and median TAT dropped to 3 hours.'
        f'<br><br>'
        f'Comparing both groups: the rate of patients returning within 7 days fell from '
        f'<b>22.4% → 19.6%</b> — a 2.8 percentage point reduction. '
        f'Fewer patients needed a return visit to collect results they had not yet received.'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

with col_before_after:
    st.markdown(
        f'<div style="font-size:11px;font-weight:700;color:{COLORS["dark"]};'
        f'margin-bottom:8px">7-Day Return Rate</div>',
        unsafe_allow_html=True,
    )
    ba1, ba2 = st.columns(2)
    with ba1:
        kpi_card("Before 2024", "22.4%", "Next-day results · return within 7 days")
    with ba2:
        kpi_card("From 2024", "19.6%", "Same-day results · −2.8pp")



# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — VARIATION (conditional — only rendered if meaningful pattern found)
# ══════════════════════════════════════════════════════════════════════════════

if not hour_df.empty:
    variation = (
        hour_df.groupby("MODALITY_GROUP")["P50_MINS"]
        .agg(p50_max="max", p50_min="min", hour_count="count")
        .reset_index()
    )
    variation["ratio"] = (
        variation["p50_max"] / variation["p50_min"].replace(0, 1)
    )
    meaningful_mods = variation[
        (variation["hour_count"] >= 3) & (variation["ratio"] >= 1.5)
    ].sort_values("ratio", ascending=False)

    if not meaningful_mods.empty:
        _section("5 · Variation — Does Time of Day Explain the Tail?")

        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["dark"]};margin-bottom:12px;line-height:1.6">'
            f'Each line shows median TAT for orders placed at that hour. A rising line means '
            f'orders placed later in the day take longer — pointing to afternoon staffing or '
            f'capacity constraints rather than a systemic workflow issue.'
            f'</div>',
            unsafe_allow_html=True,
        )

        show_mods   = meaningful_mods["MODALITY_GROUP"].tolist()[:3]
        hour_subset = hour_df[hour_df["MODALITY_GROUP"].isin(show_mods)]
        palette     = [COLORS["primary"], COLORS["coral"], COLORS["warning"]]

        fig_hour = go.Figure()
        for i, mod in enumerate(show_mods):
            df_m = hour_subset[hour_subset["MODALITY_GROUP"] == mod].sort_values("REQUEST_HOUR")
            if df_m.empty:
                continue
            fig_hour.add_trace(go.Scatter(
                x=df_m["REQUEST_HOUR"],
                y=df_m["P50_MINS"],
                mode="lines+markers",
                name=mod,
                line=dict(color=palette[i % len(palette)], width=2),
                marker=dict(size=6),
            ))

        fig_hour.update_layout(
            height=280,
            paper_bgcolor="#fff", plot_bgcolor="#fff",
            font=dict(family="Montserrat", color=COLORS["dark"]),
            margin=dict(l=0, r=20, t=10, b=30),
            xaxis=dict(
                gridcolor="#EBF3FB",
                tickfont=dict(size=9, color=COLORS["muted"]),
                title="Hour of day (order placed)",
                dtick=2,
            ),
            yaxis=dict(
                gridcolor="#EBF3FB",
                tickfont=dict(size=9, color=COLORS["muted"]),
                title="Median TAT (min)",
            ),
            legend=dict(orientation="h", y=1.10, font=dict(size=9)),
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        _caption(
            "Median imaging TAT by hour of order · modalities with ≥1.5× intraday variation shown · "
            "V2 + V1 2024+ · hours with <10 orders excluded"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLOSE — INVESTIGATION PRIORITIES
# ══════════════════════════════════════════════════════════════════════════════

_section("Open Investigation Priorities")

st.markdown(
    f'<div style="font-size:12px;color:{COLORS["dark"]};margin-bottom:16px;line-height:1.6">'
    f'The bottlenecks identified are process-specific — concentrated in Cardiac imaging workflow '
    f'and lab specimen collection — not a systemic demand problem across diagnostics as a whole.'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <table style="width:100%;border-collapse:collapse;font-size:11px;
                  font-family:Montserrat,sans-serif;color:{COLORS['dark']}">
      <thead>
        <tr style="border-bottom:2px solid #D6E4F0">
          <th style="text-align:left;padding:8px 12px;color:{COLORS['muted']};font-weight:600;
                     text-transform:uppercase;font-size:9px;letter-spacing:1px">Priority</th>
          <th style="text-align:left;padding:8px 12px;color:{COLORS['muted']};font-weight:600;
                     text-transform:uppercase;font-size:9px;letter-spacing:1px">Evidence</th>
          <th style="text-align:left;padding:8px 12px;color:{COLORS['muted']};font-weight:600;
                     text-transform:uppercase;font-size:9px;letter-spacing:1px">Investigate</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #EBF3FB">
          <td style="padding:8px 12px;font-weight:700">1 — Cardiac Imaging</td>
          <td style="padding:8px 12px">Median = 87 min · only 44% scans complete within 1 hour</td>
          <td style="padding:8px 12px">Cardiac imaging workflow — booking and scheduling process</td>
        </tr>
        <tr style="border-bottom:1px solid #EBF3FB">
          <td style="padding:8px 12px;font-weight:700">2 — Lab Specimen Collection</td>
          <td style="padding:8px 12px">V2 3-stage chain shows order→collection delay vs processing time</td>
          <td style="padding:8px 12px">Specimen collection process — phlebotomy queue and turnaround</td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:700;color:{COLORS['muted']}">3 — X-Ray</td>
          <td style="padding:8px 12px">Median = 30 min · 78% complete within 1 hour — performing well</td>
          <td style="padding:8px 12px">Monitor only — no immediate action indicated</td>
        </tr>
      </tbody>
    </table>
    """,
    unsafe_allow_html=True,
)
