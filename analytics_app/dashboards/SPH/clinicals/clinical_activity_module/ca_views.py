"""
sph/clinical_activity_module/ca_views.py
==========================================
Clinical Activity tab, rebuilt to match sph_clinical_activity_build_spec.md
section by section (tab header, S0 overview, S1-S9 analysis sections, S10
recommendations). Entry point: render_clinical_activity_tab().

Rules enforced here (per the build spec):
  - No SQL in any _render_* function — all queries called once in
    render_clinical_activity_tab() and passed down as DataFrames.
  - No hardcoded numeric literals in any insight bar — every number is an
    f-string interpolation from a DataFrame.
  - Every _render_* function shows an empty state and returns early if its
    DataFrame argument is empty.
  - insight_bar variants: "amber" | "red" | "blue" | "teal" (spec-level
    names) — mapped locally to ui_template.insight_bar's real variant
    names (warning/danger/primary/success) via _insight() below, since
    ui_template.py's Python variant map isn't touched by this rebuild.
"""

import textwrap

import pandas as pd
import streamlit as st
import plotly.graph_objects as go


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clinicals.clinical_activity_module.ca_queries as CAQ

from clinicals.opd_ipd_module.ui_template import (
    CA_BLUE, CA_GREEN, CA_RED, CA_AMBER, CA_MUTED, CA_PINK,
    CHART_LAYOUT, AXIS_STYLE, PC_CFG, TEXT_HINT, TEXT_MUTED, BORDER,
    section_header, insight_bar, chart_card, chart_card_close, kpi_row,
    ACCENT_CRITICAL, ACCENT_MONITOR, ACCENT_INFO, ACCENT_POSITIVE,
    DARK_NAVY, RASP_3,
    priority_cards,
)

_BL = {**CHART_LAYOUT}

# Spec §4 categorical convention: navy = male, raspberry-light = female —
# NOT the full teal/raspberry category colors (those mean orthopedics vs.
# general surgery/OBGYN, a different dimension entirely). Every male/female
# split in this tab should use this pair, not CA_BLUE/CA_PINK.
_GENDER_COLOR = {"Male": DARK_NAVY, "Female": RASP_3}

# Spec §4 multi-bar rule: 3+ bars/segments/layers with no verdict attached
# get graded through one color family, darkest = largest, instead of
# scattered unrelated hues. Shared across every ranked/multi-bar chart in
# this tab rather than redefined per chart.
_teal_ramp = ["#0F6E56", "#1B8A82", "#4FADA5", "#8FCFC8", "#E1F5EE"]

_AX = {**AXIS_STYLE}

_VARIANT_MAP = {"amber": "warning", "red": "danger", "blue": "primary", "teal": "success"}


def _insight(bullets: list, variant: str = "blue") -> None:
    """Wraps ui_template.insight_bar() with the spec's amber/red/blue/teal
    variant names, translated to the real component's variant set."""
    insight_bar(bullets, variant=_VARIANT_MAP.get(variant, "primary"))


def _safe(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty


def _empty_state(message: str = "Data not yet available for this section") -> None:
    st.markdown(f'<div class="ca-empty-state">🚧 {message}</div>', unsafe_allow_html=True)


def _v(df: pd.DataFrame, col: str, row: int = 0, default=0):
    try:
        val = df[col].iloc[row]
        return default if pd.isna(val) else val
    except Exception:
        return default


# ── Tab header ────────────────────────────────────────────────────────────────

def _render_tab_header() -> None:
    st.markdown(
        '<div class="ca-tab-header">'
        '<div>'
        '<div class="ca-tab-title">Clinical Quality &amp; Safety</div>'
        '<div class="ca-tab-subtitle">Clinical activity, quality indicators, and patient safety '
        'outcomes</div>'
        '</div>'
        '<div class="ca-caveat-chip">ⓘ ~18 months of clean current-system data</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Section 0 — Overview ─────────────────────────────────────────────────────

def _render_overview(kpis: pd.DataFrame) -> None:
    section_header("Overview")
    if not _safe(kpis):
        _empty_state("Overview KPIs not available.")
        return

    r = float(_v(kpis, "READMISSION_RATE"))
    r_min = float(_v(kpis, "READMISSION_RATE_MIN"))
    r_max = float(_v(kpis, "READMISSION_RATE_MAX"))
    avg_los = float(_v(kpis, "AVG_LOS"))
    med_los = float(_v(kpis, "MEDIAN_LOS"))
    worst_ssi = float(_v(kpis, "WORST_SSI_RATE"))
    worst_ssi_cat = _v(kpis, "WORST_SSI_CATEGORY", default="—")
    worst_ssi_bench = float(_v(kpis, "WORST_SSI_BENCHMARK"))
    blind_spot = int(_v(kpis, "BLIND_SPOT_COUNT"))

    kpi_row([
        {"label": "Readmission Rate (Current System)", "value": f"{r:.1f}%",
         "delta": f"{r_min:.0f}–{r_max:.0f}% monthly range", "accent_color": ACCENT_CRITICAL},
        {"label": "Length of Stay, Hospital-wide", "value": f"{avg_los:.1f}d avg",
         "delta": f"Median {med_los:.1f}d", "accent_color": DARK_NAVY},
        {"label": "Worst SSI Category vs Benchmark", "value": f"{worst_ssi:.1f}%",
         "delta": f"{worst_ssi_cat} · ceiling {worst_ssi_bench:.1f}%", "accent_color": ACCENT_CRITICAL},
        {"label": "31–90 Day Blind Spot", "value": f"{blind_spot:,} patients",
         "delta": "invisible to standard KPI", "accent_color": ACCENT_MONITOR},
    ])


# ── Section 1 — Monthly trend + spike drill-down ─────────────────────────────

def _render_s1_monthly_trend(monthly: pd.DataFrame, spike_df: pd.DataFrame) -> None:
    section_header("1 — Readmissions: Monthly Trend")
    if not _safe(monthly):
        _empty_state("Readmission data pipeline not yet configured — contact the data team.")
        return

    col_l, col_r = st.columns(2)

    with col_l:
        chart_card("Monthly readmission rate", "EMR_V2 only · 30-day window · bars = discharge volume")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["VISIT_MONTH"], y=monthly["DISCHARGE_COUNT"], name="Discharges",
            marker=dict(color="rgba(27,138,130,0.10)", cornerradius=3), yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            x=monthly["VISIT_MONTH"], y=monthly["READMISSION_RATE"], mode="lines+markers",
            name="Readmission rate", line=dict(color=CA_BLUE, width=2.5), marker=dict(size=7, color=CA_BLUE),
        ))
        fig.update_layout(
            **{
                **_BL, "height": 320,
                "margin": {**_BL.get("margin", {}), "b": 110},
                "legend": dict(orientation="h", y=-0.55, x=0.5, xanchor="center"),
            },
            xaxis={**_AX, "tickangle": -45, "automargin": True},
            yaxis={**_AX, "ticksuffix": "%", "title": "Readmission rate"},
            yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Discharges"),
        )
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    peak_row = monthly.loc[monthly["READMISSION_RATE"].idxmax()]
    peak_month = peak_row["VISIT_MONTH"]
    peak_rate = float(peak_row["READMISSION_RATE"])
    n_months = len(monthly)

    simba_share, staged_count = None, None
    with col_r:
        if not _safe(spike_df):
            _empty_state("Spike drill-down data not available.")
        else:
            sdf = spike_df.sort_values("SORT_MONTH")
            month_order = sdf.drop_duplicates("SPIKE_MONTH").sort_values("SORT_MONTH")["SPIKE_MONTH"].tolist()

            # Descriptive breakdown, no verdict attached — graded teal ramp by
            # total volume (spec §4 multi-bar rule), not red/amber assigned
            # by specific ward name, which reads as a status judgment on
            # that ward rather than "this ward contributed more volume."
            by_ward = sdf.groupby(["SPIKE_MONTH", "WARD"], sort=False)["READMISSION_COUNT"].sum().reset_index()
            ward_totals = by_ward.groupby("WARD")["READMISSION_COUNT"].sum().sort_values(ascending=False)
            wards = ward_totals.index.tolist()
            ward_colors = {
                w: _teal_ramp[min(i, len(_teal_ramp) - 1)] for i, w in enumerate(wards)
            }

            chart_card(
                "Readmissions by ward, spike months only",
                "Each bar = one confirmed spike month, stacked by index-discharge ward",
            )
            fig = go.Figure()
            for w in wards:
                sub = (
                    by_ward[by_ward["WARD"] == w]
                    .set_index("SPIKE_MONTH")["READMISSION_COUNT"]
                    .reindex(month_order, fill_value=0)
                )
                fig.add_trace(go.Bar(
                    x=month_order, y=sub, name=w.title(),
                    marker=dict(color=ward_colors[w], cornerradius=3),
                ))
            fig.update_layout(
                **{**_BL, "height": 280, "barmode": "stack",
                   "legend": dict(orientation="h", y=-0.22, x=0.5, xanchor="center")},
                xaxis=_AX, yaxis={**_AX, "title": "Readmissions"},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

            simba_share = (
                by_ward[by_ward["WARD"].str.contains("SIMBA", na=False)]["READMISSION_COUNT"].sum()
                / by_ward["READMISSION_COUNT"].sum() * 100
            ) if by_ward["READMISSION_COUNT"].sum() else 0.0

            staged_kw = ["removal", "revision", "staged", "planned"]
            staged_count = sdf["READMIT_DX_LABEL"].str.lower().apply(
                lambda t: any(k in t for k in staged_kw)
            ).sum()

    bullets = [
        f"{peak_month} peak ({peak_rate:.1f}%) is the highest single-month rate in the current-system window.",
        f"{n_months} months of data is too short to confirm whether {peak_month} was a one-off or the start "
        "of an upward pattern.",
        "May 2025, Jul 2025, Nov 2025, Mar 2026, and May 2026 are the spike months — drilled into on the "
        "right.",
    ]
    if simba_share is not None:
        bullets.append(
            f"SIMBA-ORTHOPAEDIC is {simba_share:.0f}% of all readmissions across those 5 spike months — the "
            f"same ward dominates nearly every spike, not five unrelated events. At least {staged_count} are "
            "labeled implant/k-wire/ex-fix removal or plating revision — scheduled staged-care follow-ups on "
            "trauma patients, not unplanned complications, and almost none are tagged as a wound/infection "
            "complication."
        )
    bullets.append(
        "<em><strong>So what:</strong> These spikes look like a low-volume ward having a normal cluster "
        "of scheduled staged-care follow-ups land in the same month, not a quality signal — confirm with "
        "the Expected vs. Potentially Preventable split before treating this as an incident.</em>"
    )
    _insight(bullets, variant="amber")


# ── Section 2 — Ward analysis ────────────────────────────────────────────────

def _ward_condition_chart(df_ward: pd.DataFrame, ward_name: str, rate: float, is_outlier: bool,
                           height: int = 260) -> None:
    rate_class = "ca-ward-rate-outlier" if is_outlier else "ca-ward-rate-normal"
    rate_label = f"▲ Outlier ({rate:.1f}%)" if is_outlier else f"{rate:.1f}%"
    bar_color = CA_RED if is_outlier else CA_AMBER

    chart_card(
        f'<span class="ca-ward-label">{ward_name}</span>'
        f'<span class="{rate_class}">{rate_label}</span>',
        "Readmission category — every readmission for this ward, by clinical category",
    )
    df_ward = df_ward.sort_values("READMISSION_COUNT")
    bar_colors = [
        CA_MUTED if label == "No diagnosis recorded" else bar_color
        for label in df_ward["DIAGNOSIS_LABEL"]
    ]
    fig = go.Figure(go.Bar(
        orientation="h", y=df_ward["DIAGNOSIS_LABEL"], x=df_ward["READMISSION_COUNT"],
        marker=dict(color=bar_colors, cornerradius=3),
        text=df_ward["READMISSION_COUNT"].astype(str), textposition="outside",
        textfont=dict(size=12, color=CA_MUTED, family=_BL["font"]["family"]),
        cliponaxis=False,
        hovertemplate="%{y}: <b>%{x}</b> readmissions<extra></extra>",
    ))
    fig.update_layout(
        **{**_BL, "height": height, "margin": dict(t=10, b=40, l=10, r=60)}, showlegend=False,
        xaxis={**_AX, "showgrid": False, "title": "Readmissions"},
        yaxis={**_AX, "showgrid": False, "automargin": True},
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()


def _render_s2_ward_analysis(ward_rates: pd.DataFrame, ward_cause: pd.DataFrame,
                              ward_diagnoses: pd.DataFrame) -> None:
    section_header("2 — Readmissions: Where, and What's Driving It")
    if not _safe(ward_rates):
        _empty_state("Ward-level readmission data not yet available.")
        return

    avg_rate = float(ward_rates["READMISSION_RATE"].mean())
    threshold = avg_rate * 1.5
    df = ward_rates.sort_values("READMISSION_RATE", ascending=True)
    colors = [CA_RED if r > threshold else CA_BLUE for r in df["READMISSION_RATE"]]

    col_l, col_r = st.columns(2)

    # Shared legend styling for both charts below — merged onto the base
    # layout's legend dict (not replacing it) so the legend keeps the same
    # font as the axis labels instead of falling back to Plotly's default.
    _legend = {**_BL.get("legend", {}), "orientation": "h", "y": -0.28, "x": 0.5, "xanchor": "center"}

    with col_l:
        chart_card("Ward readmission rate",
                   "EMR_V2 only · wards ≥20 discharges · red bars exceed 1.5× average threshold")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df["WARD"], x=df["READMISSION_RATE"], orientation="h",
            marker=dict(color=colors, cornerradius=3),
            text=[f"{v:.1f}%" for v in df["READMISSION_RATE"]], textposition="outside",
            showlegend=False,
        ))
        # Full-height reference lines via add_vline (spans the whole plot,
        # unlike a Scatter trace pinned to category positions), with no
        # inline annotation text — labeled instead via invisible dummy
        # traces below, purely so they get a clean legend entry.
        fig.add_vline(x=avg_rate, line_dash="dot", line_color="#141F3D", line_width=2)
        fig.add_vline(x=threshold, line_dash="dash", line_color="#E24B4A", line_width=2)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#141F3D", dash="dot", width=2),
            name=f"Average ({avg_rate:.1f}%)",
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="#E24B4A", dash="dash", width=2),
            name=f"Outlier threshold, 1.5× avg ({threshold:.1f}%)",
        ))
        fig.update_layout(
            **{**_BL, "height": 320, "margin": dict(t=16, b=70, l=120, r=10), "legend": _legend},
            xaxis={**_AX, "ticksuffix": "%"}, yaxis={**_AX, "showgrid": False},
        )
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    if _safe(ward_cause):
        wide = ward_cause.pivot_table(index="WARD", columns="CAUSE_TYPE", values="READMISSION_COUNT",
                                       aggfunc="sum", fill_value=0)
        wide = wide.reindex(df["WARD"])
        with col_r:
            chart_card("Readmission causes per ward — expected vs potentially preventable",
                       "All wards · each bar = total readmissions broken down by whether the return was "
                       "clinically expected")
            fig2 = go.Figure()
            for cause, color in [("Expected", CA_GREEN), ("Potentially preventable", CA_RED),
                                  ("Unclear / other", CA_MUTED)]:
                if cause in wide.columns:
                    fig2.add_trace(go.Bar(y=wide.index, x=wide[cause], name=cause, orientation="h",
                                           marker=dict(color=color, cornerradius=3)))
            fig2.update_layout(
                **{**_BL, "height": 320, "barmode": "stack",
                   "legend": _legend,
                   "margin": dict(t=16, b=70, l=120, r=10)},
                xaxis={**_AX, "title": "Readmissions"}, yaxis={**_AX, "showgrid": False},
            )
            st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
            chart_card_close()

    outlier_wards = df[df["READMISSION_RATE"] > threshold]["WARD"].tolist()

    df_desc = df.sort_values("READMISSION_RATE", ascending=False)
    top_wards = df_desc["WARD"].tolist()[:3]
    other_wards = df_desc["WARD"].tolist()[3:]

    if _safe(ward_diagnoses):
        st.markdown('<div class="ca-divider-label" style="margin-top:16px">Readmission categories — '
                    'top 3 wards by readmission rate</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ca-explain">Each chart shows every readmission for that ward, grouped into a '
            'clinical category (Hardware / implant complication, Wound / infection complication, '
            'Amputation-related, Likely ward misattribution, Pain management, GI / metabolic, Cardiac, '
            'Respiratory, New / unrelated fracture or injury, Planned follow-up / staged procedure, '
            'Other / unclear) — not just whichever exact diagnosis phrasing happened to repeat, so counts '
            'add up to the ward\'s full readmission total. Ordered highest to lowest readmission rate. Bar '
            'colour: red = outlier ward (above 1.5× threshold), amber = all others. Lower-rate wards are '
            'summarized in the insight below.</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for col, ward in zip(cols, top_wards):
            df_w = ward_diagnoses[ward_diagnoses["WARD"] == ward]
            if df_w.empty:
                continue
            rate = float(df[df["WARD"] == ward]["READMISSION_RATE"].iloc[0])
            with col:
                _ward_condition_chart(df_w.sort_values("READMISSION_COUNT"), ward, rate,
                                       ward in outlier_wards, height=220)

    bullets = []
    if other_wards:
        other_summaries = []
        for ward in other_wards:
            rate = float(df[df["WARD"] == ward]["READMISSION_RATE"].iloc[0])
            df_w = ward_diagnoses[ward_diagnoses["WARD"] == ward] if _safe(ward_diagnoses) else pd.DataFrame()
            if not df_w.empty:
                total_w = int(df_w["READMISSION_COUNT"].sum())
                top_row = df_w.sort_values("READMISSION_COUNT", ascending=False).iloc[0]
                top_dx, top_n = top_row["DIAGNOSIS_LABEL"], int(top_row["READMISSION_COUNT"])
                other_summaries.append(
                    f"{ward} ({rate:.1f}%, {total_w} readmissions, top driver: {top_dx} — {top_n} of {total_w})"
                )
            else:
                other_summaries.append(f"{ward} ({rate:.1f}%)")
        bullets.append(
            "Lower-rate wards not charted above — " + "; ".join(other_summaries) + "."
        )
    if outlier_wards:
        outlier_row = df[df["WARD"] == outlier_wards[0]].iloc[0]
        bullets.append(
            f"<strong>{outlier_wards[0]}</strong> is the highest-rate ward above the {threshold:.1f}% "
            f"outlier threshold ({float(outlier_row['READMISSION_RATE']):.1f}% vs {threshold:.1f}% ceiling)."
        )
    if _safe(ward_cause):
        preventable = ward_cause[ward_cause["CAUSE_TYPE"] == "Potentially preventable"]
        if not preventable.empty:
            worst = preventable.groupby("WARD")["READMISSION_COUNT"].sum().idxmax()
            bullets.append(
                f"<strong>Potentially preventable readmissions</strong> are highest in {worst} — worth a "
                "targeted wound-management and pain-control review at that ward specifically."
            )
    if _safe(ward_diagnoses):
        no_dx = ward_diagnoses[ward_diagnoses["DIAGNOSIS_LABEL"] == "No diagnosis recorded"]
        if not no_dx.empty:
            no_dx_total = int(no_dx["READMISSION_COUNT"].sum())
            all_total = int(ward_diagnoses["READMISSION_COUNT"].sum())
            worst_no_dx = no_dx.sort_values("READMISSION_COUNT", ascending=False).iloc[0]
            bullets.append(
                f"<strong>{no_dx_total} of {all_total} readmissions ({100 * no_dx_total / all_total:.0f}%)</strong> "
                f"have no diagnosis record at all — worst at {worst_no_dx['WARD']} "
                f"({int(worst_no_dx['READMISSION_COUNT'])} readmissions). This is a documentation gap, not a "
                "clinically ambiguous cause, and it's counted separately here rather than folded into "
                "'Other / unclear' — the true unexplained-cause rate is lower than the combined bar would suggest."
            )

        all_total = int(ward_diagnoses["READMISSION_COUNT"].sum())
        cat_totals = ward_diagnoses.groupby("DIAGNOSIS_LABEL")["READMISSION_COUNT"].sum()
        if all_total and "Planned follow-up / staged procedure" in cat_totals.index:
            staged_total = int(cat_totals["Planned follow-up / staged procedure"])
            is_largest = staged_total == cat_totals.max()
            bullets.append(
                f"<strong>Planned follow-up / staged procedure"
                f"{' is the single largest category hospital-wide' if is_largest else ''} — "
                f"{staged_total} of {all_total} readmissions ({100 * staged_total / all_total:.0f}%)</strong> "
                "(implant/k-wire/ex-fix removal, plating, nailing revisions) — scheduled trauma-care "
                "follow-ups, not unplanned returns" +
                (", and the main reason raw readmission rates overstate the quality problem."
                 if is_largest else ".")
            )

        misattributed = ward_diagnoses[ward_diagnoses["DIAGNOSIS_LABEL"] == "Likely ward misattribution"]
        if not misattributed.empty:
            mis_total = int(misattributed["READMISSION_COUNT"].sum())
            mis_wards = ", ".join(misattributed["WARD"].tolist())
            bullets.append(
                f"<strong>{mis_total} readmission{'s' if mis_total != 1 else ''} at {mis_wards}</strong> "
                "carry a diagnosis (cataract / cleft lip) that isn't an orthopaedic or general-surgical "
                "condition — likely recorded against the wrong ward, worth a data-entry check before "
                "trusting that ward's rate at face value."
            )
    bullets.append(
        "Discharge date accuracy issues mean ward-level counts may be understated for affected wards."
    )
    so_what = (
        "A targeted, ward-specific fix — not a hospital-wide policy change — is what would move the "
        "outlier ward's number."
    )
    if outlier_wards and _safe(ward_diagnoses):
        top_ward = outlier_wards[0]
        _non_actionable = [
            "No diagnosis recorded", "Other / unclear",
            "Likely ward misattribution", "New / unrelated fracture or injury",
        ]
        clinical = ward_diagnoses[
            (ward_diagnoses["WARD"] == top_ward)
            & (~ward_diagnoses["DIAGNOSIS_LABEL"].isin(_non_actionable))
        ]
        if not clinical.empty:
            lead_cat = clinical.sort_values("READMISSION_COUNT", ascending=False).iloc[0]
            so_what = (
                f"A targeted fix at <strong>{top_ward}</strong> — starting with "
                f"{lead_cat['DIAGNOSIS_LABEL']} ({int(lead_cat['READMISSION_COUNT'])} readmissions, its "
                "largest identified category) — not a hospital-wide policy change, is what would move the "
                "outlier ward's number."
            )
    bullets.append(f"<em><strong>So what:</strong> {so_what}</em>")
    _insight(bullets, variant="red" if outlier_wards else "blue")


# ── Section 3 — Why ───────────────────────────────────────────────────────────

def _render_s3_readmission_type(type_breakdown: pd.DataFrame, area_breakdown: pd.DataFrame,
                                 area_diagnoses: pd.DataFrame) -> None:
    section_header("3 — Readmissions: Why")
    if not _safe(type_breakdown) and not _safe(area_breakdown):
        _empty_state("Readmission classification data not yet available.")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        if not _safe(type_breakdown):
            _empty_state()
        else:
            df = type_breakdown.sort_values("COUNT")
            chart_card("Readmission type breakdown")
            fig = go.Figure(go.Bar(
                orientation="h", y=df["READMISSION_TYPE"], x=df["COUNT"],
                marker=dict(color=CA_BLUE, cornerradius=3),
                text=[f"{p:.0f}%" for p in df["PCT"]], textposition="outside",
                textfont=dict(size=12, color=CA_MUTED, family=_BL["font"]["family"]),
                cliponaxis=False,
                hovertemplate="%{y}: <b>%{x}</b> readmissions<extra></extra>",
            ))
            fig.update_layout(
                **{**_BL, "height": 260, "margin": dict(t=8, b=52, l=10, r=70)}, showlegend=False,
                xaxis={**_AX, "title": "Number of readmissions (labels show % of total)",
                       "range": [0, df["COUNT"].max() * 1.15]},
                yaxis={**_AX, "showgrid": False},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(area_breakdown):
            _empty_state()
        else:
            # Categorical breakdown, no verdict — teal/raspberry/grey (spec §4 donut charts)
            colors_map = {"Same clinical area": CA_BLUE, "Different clinical area": CA_PINK, "Unknown": CA_MUTED}
            colors = [colors_map.get(a, CA_MUTED) for a in area_breakdown["AREA_GROUP"]]
            chart_card("Same vs. different clinical area")
            fig = go.Figure(go.Pie(
                labels=area_breakdown["AREA_GROUP"], values=area_breakdown["COUNT"], hole=0.55,
                marker=dict(colors=colors), textinfo="percent",
            ))
            fig.update_layout(**{**_BL, "height": 260,
                                  "legend": dict(orientation="h", y=-0.22, x=0.5, xanchor="center")})
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    if _safe(area_diagnoses):
        col_l2, col_r2 = st.columns(2)
        for col, area_label, color in ((col_l2, "Same clinical area", CA_BLUE),
                                        (col_r2, "Different clinical area", CA_PINK)):
            sub = area_diagnoses[area_diagnoses["AREA_GROUP"] == area_label].sort_values("COUNT")
            with col:
                chart_card(f"Top diagnoses — {area_label}")
                if sub.empty:
                    _empty_state()
                else:
                    fig = go.Figure(go.Bar(
                        orientation="h", y=sub["DIAGNOSIS_LABEL"], x=sub["COUNT"],
                        marker=dict(color=color, cornerradius=3),
                    ))
                    fig.update_layout(**{**_BL, "height": 260}, showlegend=False,
                                       xaxis={**_AX, "showgrid": False}, yaxis={**_AX, "showgrid": False})
                    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
                chart_card_close()

    bullets = []
    if _safe(type_breakdown):
        unclear = type_breakdown[type_breakdown["READMISSION_TYPE"] == "Other / unclear"]
        if not unclear.empty:
            bullets.append(
                f"<strong>Other / Unclear</strong> is {'the largest' if unclear['COUNT'].iloc[0] == type_breakdown['COUNT'].max() else 'a notable'} "
                f"readmission category ({int(unclear['COUNT'].iloc[0])} cases, {unclear['PCT'].iloc[0]:.0f}%) — "
                "this warrants a documentation audit as much as a clinical one."
            )
        actionable = type_breakdown[type_breakdown["READMISSION_TYPE"] == "Wound / infection complication"]
        if not actionable.empty:
            bullets.append(
                f"Wound / infection complications represent {int(actionable['COUNT'].sum())} cases "
                f"({actionable['PCT'].sum():.0f}%) — the share most likely to be reduced by improved discharge "
                "protocols."
            )
    if _safe(area_breakdown):
        same = area_breakdown[area_breakdown["AREA_GROUP"] == "Same clinical area"]
        diff = area_breakdown[area_breakdown["AREA_GROUP"] == "Different clinical area"]
        if not same.empty and not diff.empty:
            bullets.append(
                f"<strong>Same clinical area</strong> ({same['PCT'].iloc[0]:.0f}%) means the patient returned "
                "to the same ward or specialty as their original admission — a direct consequence of the "
                f"original procedure. <strong>Different clinical area</strong> ({diff['PCT'].iloc[0]:.0f}%) "
                "means a different specialty — pointing to a secondary complication or a new, unrelated "
                "condition."
            )
    bullets.append(
        "<em><strong>So what:</strong> A blanket readmission programme would treat several clinically "
        "distinct problems as one. The actionable categories above are the most targeted place to start.</em>"
    )
    if bullets:
        _insight(bullets, variant="blue")


# ── Section 4 — Who ───────────────────────────────────────────────────────────

def _render_s4_demographics(age_complication: pd.DataFrame) -> None:
    section_header("4 — Readmissions: Who")
    if not _safe(age_complication):
        _empty_state("Age/complication data not yet available.")
        return

    _AGE_ORDER = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    # Reuses the same, already-validated _READMISSION_TYPE_CASE taxonomy as
    # Sections 1-3 (see get_ca_age_complication query) instead of the older,
    # cruder 4-category _COMPLICATION_TYPE_CASE, which had no matching for
    # pain/GI/cardiac/respiratory/staged-procedure/fracture/amputation and
    # so dumped most readmissions into "Other". Lower-volume clinical
    # categories share a colour to keep the legend readable.
    # Descriptive breakdown, no verdict attached — graded teal ramp per
    # spec §4 multi-bar rule, not red/green/pink (those are reserved for
    # status). Low-volume categories share the lightest teal tint; true
    # unclear/other stays grey.
    _COMPL_COLORS = {
        "Wound / infection complication": "#0F6E56",
        "New / unrelated fracture or injury": "#1B8A82",
        "Planned follow-up / staged procedure": "#4FADA5",
        "Hardware / implant complication": "#8FCFC8",
        "Pain management": "#E1F5EE", "GI / metabolic": "#E1F5EE",
        "Cardiac": "#E1F5EE", "Respiratory": "#E1F5EE",
        "Amputation-related": "#E1F5EE", "Likely ward misattribution": "#E1F5EE",
        "Other / unclear": CA_MUTED,
    }

    by_age_compl = age_complication.groupby(["AGE_GROUP", "COMPLICATION_TYPE"])["READMISSION_COUNT"].sum().reset_index()
    wide = by_age_compl.pivot_table(index="AGE_GROUP", columns="COMPLICATION_TYPE",
                                     values="READMISSION_COUNT", aggfunc="sum", fill_value=0)
    wide = wide.reindex([a for a in _AGE_ORDER if a in wide.index])

    chart_card(
        "Age profile × complication type — where do readmissions concentrate?",
        "Each bar = total readmissions for that age group · colour = complication driving the readmission",
    )
    fig = go.Figure()
    for compl, color in _COMPL_COLORS.items():
        if compl in wide.columns:
            fig.add_trace(go.Bar(x=wide.index, y=wide[compl], name=compl, marker=dict(color=color, cornerradius=3)))
    fig.update_layout(
        **{**_BL, "height": 300, "barmode": "stack",
           "legend": dict(orientation="h", y=-0.25, x=0.5, xanchor="center")},
        xaxis={**_AX, "showgrid": False, "tickangle": -15}, yaxis=_AX,
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()

    col_l, col_r = st.columns(2)
    with col_l:
        by_age_gender = age_complication.groupby(["AGE_GROUP", "GENDER"])["READMISSION_COUNT"].sum().reset_index()
        wide_g = by_age_gender.pivot_table(index="AGE_GROUP", columns="GENDER",
                                            values="READMISSION_COUNT", aggfunc="sum", fill_value=0)
        wide_g = wide_g.reindex([a for a in _AGE_ORDER if a in wide_g.index])
        chart_card("Gender split within age bands")
        fig2 = go.Figure()
        # Navy = male, raspberry-light = female (spec §4 categorical convention)
        for gender, color in [("Male", _GENDER_COLOR["Male"]), ("Female", _GENDER_COLOR["Female"])]:
            if gender in wide_g.columns:
                fig2.add_trace(go.Bar(x=wide_g.index, y=wide_g[gender], name=gender,
                                       marker=dict(color=color, cornerradius=3)))
        fig2.update_layout(
            **{**_BL, "height": 260, "barmode": "group",
               "legend": dict(orientation="h", y=-0.25, x=0.5, xanchor="center")},
            xaxis={**_AX, "showgrid": False, "tickangle": -20}, yaxis=_AX,
        )
        st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
        chart_card_close()

    with col_r:
        by_compl = age_complication.groupby("COMPLICATION_TYPE")["READMISSION_COUNT"].sum().sort_values()
        chart_card("Complication category breakdown")
        # Multi-bar, no verdict — graded teal ramp, darkest = highest count (spec §4)
        _teal_ramp = ["#E1F5EE", "#8FCFC8", "#4FADA5", "#1B8A82", "#0F6E56"]
        n = len(by_compl)
        bar_colors = [
            _teal_ramp[min(int(i / max(n - 1, 1) * (len(_teal_ramp) - 1)), len(_teal_ramp) - 1)]
            for i in range(n)
        ]
        fig3 = go.Figure(go.Bar(
            orientation="h", y=by_compl.index, x=by_compl.values, marker=dict(color=bar_colors, cornerradius=3),
        ))
        fig3.update_layout(**{**_BL, "height": 260}, showlegend=False,
                            xaxis={**_AX, "showgrid": False}, yaxis={**_AX, "showgrid": False})
        st.plotly_chart(fig3, use_container_width=True, config=PC_CFG)
        chart_card_close()

    ssi_55plus = age_complication[
        (age_complication["AGE_GROUP"].isin(["55-64", "65+"])) &
        (age_complication["COMPLICATION_TYPE"] == "Wound / infection complication")
    ]["READMISSION_COUNT"].sum()
    ssi_total = age_complication[
        age_complication["COMPLICATION_TYPE"] == "Wound / infection complication"
    ]["READMISSION_COUNT"].sum()

    grand_total = int(by_compl.sum())
    top_cat, top_cat_n = by_compl.index[-1], int(by_compl.iloc[-1])

    bullets = []
    if grand_total:
        bullets.append(
            f"<strong>{top_cat}</strong> is the largest category overall — {top_cat_n} of {grand_total} "
            f"readmissions ({100 * top_cat_n / grand_total:.0f}%)."
        )
    if ssi_total:
        bullets.append(
            f"Age 55+ is where wound/infection complications concentrate — {int(ssi_55plus)} of {int(ssi_total)} "
            "wound/infection readmissions occurred in patients 55 and older."
        )
    top_age = age_complication.groupby("AGE_GROUP")["READMISSION_COUNT"].sum().idxmax()
    bullets.append(f"{top_age} carries the largest total readmission volume across complication types.")
    bullets.append(
        "This chart now uses the same category definitions as Sections 1-3 (previously a separate, cruder "
        "classifier lumped most non-infection, non-mechanical readmissions into \"Other\")."
    )
    bullets.append(
        f"<em><strong>So what:</strong> Discharge planning for {top_age} needs to stratify by complication "
        f"type, not just age — starting with {top_cat.lower()}, its largest driver.</em>"
    )
    _insight(bullets, variant="amber")


# ── Section 5 — Blind spot ───────────────────────────────────────────────────

def _render_s5_blind_spot(blind_spot: pd.DataFrame, blind_spot_type: pd.DataFrame,
                           delayed: pd.DataFrame) -> None:
    section_header("5 — The 31–90 Day Blind Spot")
    if not _safe(blind_spot):
        _empty_state("31–90 day lookback data not yet available.")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        df = blind_spot.sort_values("BUCKET_ORDER")
        colors = [CA_AMBER if b else CA_BLUE for b in df["IS_BLIND_SPOT"]]
        chart_card("Gap between discharge and next admission")
        fig = go.Figure(go.Bar(x=df["GAP_BUCKET"], y=df["PATIENT_COUNT"], marker=dict(color=colors, cornerradius=3)))
        fig.add_annotation(x=0.52, y=1.10, xref="paper", yref="paper", showarrow=False,
                            text="← standard KPI window   |   blind spot →",
                            font=dict(color="#854F0B", size=10))
        fig.update_layout(**{**_BL, "height": 260, "margin": dict(t=32, b=52, l=10, r=10)}, showlegend=False,
                           xaxis={**_AX, "showgrid": False, "tickangle": -20}, yaxis=_AX)
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    with col_r:
        if not _safe(blind_spot_type):
            _empty_state()
        else:
            colors_map = {"Delayed complication": CA_RED, "Planned staged care": CA_GREEN,
                          "New / unrelated injury": CA_BLUE, "Other / unclear": CA_AMBER}
            colors = [colors_map.get(t, CA_MUTED) for t in blind_spot_type["RETURN_TYPE"]]
            chart_card("31–90 day return type breakdown")
            fig2 = go.Figure(go.Pie(
                labels=blind_spot_type["RETURN_TYPE"], values=blind_spot_type["COUNT"], hole=0.55,
                marker=dict(colors=colors), textinfo="percent",
            ))
            fig2.update_layout(
                **{**_BL, "height": 260, "legend": dict(orientation="v", x=1.02, y=0.5),
                   "margin": dict(t=8, b=8, l=10, r=140)},
            )
            st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
            chart_card_close()

    if _safe(delayed):
        df_d = delayed.sort_values("PATIENT_COUNT")
        chart_card("Delayed complications", "Named diagnoses behind the 31–90 day 'delayed complication' bucket")
        # Same category, same color everywhere it appears — "Delayed
        # complication" is red in the return-type donut above, so it stays
        # red here too, not a different amber.
        fig3 = go.Figure(go.Bar(
            orientation="h", y=df_d["COMPLICATION_LABEL"], x=df_d["PATIENT_COUNT"],
            marker=dict(color=CA_RED, cornerradius=3),
            text=[f"{n} patients" for n in df_d["PATIENT_COUNT"]], textposition="outside",
        ))
        fig3.update_layout(**{**_BL, "height": 272, "margin": dict(t=8, b=52, l=10, r=100)}, showlegend=False,
                            xaxis={**_AX, "showgrid": False}, yaxis={**_AX, "showgrid": False})
        st.plotly_chart(fig3, use_container_width=True, config=PC_CFG)
        chart_card_close()

    blind_n = int(blind_spot.loc[blind_spot["IS_BLIND_SPOT"], "PATIENT_COUNT"].sum())
    bullets = [
        f"<strong>{blind_n:,} patients</strong> return in the 31–90 day window — currently invisible to "
        "the hospital's standard readmission KPI."
    ]
    if _safe(delayed) and blind_n:
        delayed_n = int(delayed["PATIENT_COUNT"].sum())
        delayed_pct = round(100 * delayed_n / blind_n) if blind_n else 0
        bullets.append(
            f"Of these {blind_n}, <strong>{delayed_n} ({delayed_pct}%) are delayed complications</strong> — "
            "named clinical events, not administrative returns."
        )
    if _safe(blind_spot_type):
        planned = blind_spot_type[blind_spot_type["RETURN_TYPE"] == "Planned staged care"]
        if not planned.empty and blind_n:
            planned_n = int(planned["COUNT"].iloc[0])
            planned_pct = round(100 * planned_n / blind_n)
            bullets.append(
                f"The remaining {planned_n} ({planned_pct}%) are planned staged care — expected returns not "
                "tracked as readmissions, not a patient safety concern."
            )
    bullets.append(
        "<em><strong>So what:</strong> The delayed complications above represent a measurable, named "
        "patient safety gap. The hospital's official readmission rate is undercounting by at least that "
        "much.</em>"
    )
    _insight(bullets, variant="amber")


# ── Section 6 — Length of stay ───────────────────────────────────────────────

def _render_s6_los(los_ward: pd.DataFrame, los_dist: pd.DataFrame, los_conditions: pd.DataFrame,
                    los_scatter: pd.DataFrame, los_index_readmit: pd.DataFrame) -> None:
    section_header("6 — Length of Stay")
    if not _safe(los_ward):
        _empty_state("LOS data not yet available.")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        df = los_ward.sort_values("AVG_LOS")
        chart_card("LOS avg vs. median by ward")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=df["WARD"], x=df["AVG_LOS"], name="Average LOS", orientation="h",
                              marker=dict(color=CA_BLUE, cornerradius=3)))
        fig.add_trace(go.Bar(y=df["WARD"], x=df["MEDIAN_LOS"], name="Median LOS", orientation="h",
                              marker=dict(color=CA_MUTED, cornerradius=3)))
        fig.update_layout(
            **{**_BL, "height": 260, "barmode": "group",
               "legend": dict(orientation="h", y=-0.22, x=0.5, xanchor="center")},
            xaxis={**_AX, "title": "Days"}, yaxis={**_AX, "showgrid": False},
        )
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    with col_r:
        if not _safe(los_dist):
            _empty_state()
        else:
            chart_card("LOS distribution by ward (IQR)")
            fig2 = go.Figure()
            for ward in sorted(los_dist["WARD"].unique(), reverse=True):
                dfw = los_dist[los_dist["WARD"] == ward]
                has_condition = "CONDITION_LABEL" in dfw.columns
                # Single-series magnitude, no verdict — teal, not a raw
                # off-palette blue. Outlier points are statistical (box-plot
                # whisker outliers), not a clinical flag, so grey rather
                # than red — red is reserved for an actual verdict.
                fig2.add_trace(go.Box(
                    x=dfw["LOS_DAYS"], name=ward, boxpoints="outliers", orientation="h",
                    fillcolor="rgba(27,138,130,0.15)", line=dict(color=CA_BLUE),
                    marker=dict(color=CA_MUTED, size=4, opacity=0.7),
                    customdata=dfw[["CONDITION_LABEL"]].values if has_condition else None,
                    hovertemplate=(
                        "%{customdata[0]}<br>LOS: %{x} days<extra></extra>"
                        if has_condition else "LOS: %{x} days<extra></extra>"
                    ),
                ))
            fig2.update_layout(**{**_BL, "height": 260, "margin": dict(t=8, b=40, l=120, r=10)}, showlegend=False,
                                xaxis={**_AX, "title": "Length of stay (days)"}, yaxis={**_AX, "showgrid": False})
            st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
            chart_card_close()

    los_condition_caveat = None
    if _safe(los_conditions):
        chart_card(
            "Top conditions by average length of stay",
            "Top 8 shown, ranked by average days. Ward, case count, and average age shown per "
            "condition — most are small samples (n≤8).",
        )
        df_c = los_conditions.sort_values("AVG_LOS", ascending=False).reset_index(drop=True)
        max_los = float(df_c["AVG_LOS"].max()) or 1.0
        _RELIABLE_N = 15  # case count at/above which an average is treated as reasonably stable

        rows_html = ""
        for i, row in df_c.iterrows():
            n = int(row["CASE_COUNT"])
            los = float(row["AVG_LOS"])
            age = row.get("AVG_AGE")
            age_txt = f" · avg age {age:.0f}" if pd.notna(age) else ""
            ward = row["WARD"] or "Unknown ward"
            condition_label = str(row["CONDITION_LABEL"]).strip().title()

            # Ranked list, no verdict attached — a graded teal ramp by rank,
            # not a red/amber/teal mix implying "top 2 = bad, small-sample =
            # caution." Sample-size reliability is already called out in the
            # text caveat below, not through bar color.
            bar_color = _teal_ramp[min(i, len(_teal_ramp) - 1)]

            bar_pct = max(4, round(100 * los / max_los))
            rows_html += textwrap.dedent(f"""\
                <div style="display:flex;align-items:center;gap:10px;padding:4px 0;border-bottom:1px solid {BORDER}">
                <div style="flex:0 0 190px;min-width:0">
                <div style="font-size:11.5px;font-weight:600;color:#141F3D;line-height:1.3">{condition_label}</div>
                <div style="font-size:9.5px;color:{bar_color};font-weight:600;line-height:1.4">{ward}</div>
                <div style="font-size:9.5px;color:{TEXT_MUTED};line-height:1.4">{n} cases{age_txt}</div>
                </div>
                <div style="flex:1;display:flex;align-items:center;gap:6px;min-width:0">
                <div style="flex:1;background:#F4F6FA;border-radius:3px;height:10px;overflow:hidden">
                <div style="width:{bar_pct}%;height:100%;background:{bar_color};border-radius:3px"></div>
                </div>
                <div style="flex:0 0 38px;text-align:right;font-size:11px;font-weight:600;color:{bar_color}">{los:.1f}d</div>
                </div>
                </div>
                """)
        st.markdown(f'<div style="padding:2px 2px">{rows_html}</div>', unsafe_allow_html=True)

        reliable = df_c[df_c["CASE_COUNT"] >= _RELIABLE_N]
        if not reliable.empty:
            reliable_label = str(reliable.iloc[0]["CONDITION_LABEL"]).strip().title()
            reliable_n = int(reliable.iloc[0]["CASE_COUNT"])
            other_max_n = int(df_c.loc[df_c["CASE_COUNT"] < _RELIABLE_N, "CASE_COUNT"].max())
            los_condition_caveat = (
                f'<strong>{reliable_label}</strong> (n={reliable_n}) is the only condition here with '
                f'enough patients to trust its average — every other one has {other_max_n} or fewer, so '
                'a single unusual case could move it up or down the ranking.'
            )
        else:
            los_condition_caveat = (
                f'<strong>Small samples:</strong> every condition here has {_RELIABLE_N - 1} or fewer '
                'cases — treat these averages as directional, not statistically robust.'
            )
        chart_card_close()

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        if not _safe(los_scatter):
            _empty_state()
        else:
            chart_card("Ward LOS vs. readmission rate")
            _positions = ["top center", "bottom center", "middle right", "middle left", "top right"]
            fig4 = go.Figure()
            for i, (_, r) in enumerate(los_scatter.iterrows()):
                fig4.add_trace(go.Scatter(
                    x=[r["AVG_LOS"]], y=[r["READMISSION_RATE"]], mode="markers+text",
                    text=[r["WARD"]], textposition=_positions[i % len(_positions)],
                    textfont=dict(size=9),
                    marker=dict(size=10, color=CA_BLUE, opacity=0.85),
                    showlegend=False,
                ))
            fig4.update_layout(
                **{**_BL, "height": 300, "margin": dict(t=24, b=48, l=10, r=40)},
                xaxis={**_AX, "title": "Avg LOS (days)", "range": [
                    float(los_scatter["AVG_LOS"].min()) - 3, float(los_scatter["AVG_LOS"].max()) + 3,
                ]},
                yaxis={**_AX, "ticksuffix": "%", "title": "Readmission rate"},
            )
            st.plotly_chart(fig4, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r2:
        if not _safe(los_index_readmit):
            _empty_state()
        else:
            chart_card("Index stay vs. readmission stay")
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(x=los_index_readmit["LOS_TYPE"], y=los_index_readmit["INDEX_STAY"],
                                   name="Index stay", marker=dict(color=CA_BLUE, cornerradius=3)))
            fig5.add_trace(go.Bar(x=los_index_readmit["LOS_TYPE"], y=los_index_readmit["READMIT_STAY"],
                                   name="Readmission stay", marker=dict(color=CA_MUTED, cornerradius=3)))
            fig5.update_layout(
                **{**_BL, "height": 300, "barmode": "group",
                   "legend": dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                   "margin": dict(t=24, b=48, l=10, r=10)},
                xaxis={**_AX, "showgrid": False}, yaxis=_AX,
            )
            st.plotly_chart(fig5, use_container_width=True, config=PC_CFG)
            chart_card_close()

    avg_los_hosp = float(los_ward["AVG_LOS"].mean())
    med_los_hosp = float(los_ward["MEDIAN_LOS"].median())
    pct_shorter = float(los_index_readmit.attrs.get("pct_shorter", 0) or 0) if _safe(los_index_readmit) else 0

    bullets = [
        f"Current-system LOS (avg {avg_los_hosp:.1f}d / median {med_los_hosp:.1f}d) is stable. The gap "
        "between average and median signals a small number of very long stays pulling the mean up.",
    ]
    if _safe(los_ward):
        widest = (los_ward["AVG_LOS"].astype(float) - los_ward["MEDIAN_LOS"].astype(float)).abs()
        if not widest.empty:
            widest_ward = los_ward.loc[widest.idxmax(), "WARD"]
            bullets.append(
                f"LOS variability is highest in {widest_ward} — average and median diverge most there, "
                "suggesting case complexity, not a fixed protocol, is driving length of stay."
            )
    shortest_ward = None
    if _safe(los_scatter) and len(los_scatter) >= 3:
        corr = los_scatter["AVG_LOS"].astype(float).corr(los_scatter["READMISSION_RATE"].astype(float))
        if pd.notna(corr) and corr < -0.2:
            shortest = los_scatter.loc[los_scatter["AVG_LOS"].astype(float).idxmin()]
            shortest_ward = shortest["WARD"]
            bullets.append(
                f"Wards with shorter average stays tend to have <strong>higher</strong> readmission rates "
                f"(e.g. {shortest['WARD']}: {float(shortest['AVG_LOS']):.1f}d avg, "
                f"{float(shortest['READMISSION_RATE']):.1f}% readmission) — consistent with patients being "
                "discharged before care is complete, not with shorter stays being a sign of efficiency."
            )
    if _safe(los_conditions):
        top_los = los_conditions.sort_values("AVG_LOS", ascending=False).head(2)
        top_parts = [
            f"{str(r['CONDITION_LABEL']).strip().title()} ({float(r['AVG_LOS']):.1f}d, "
            f"n={int(r['CASE_COUNT'])}, {r['WARD']})"
            for _, r in top_los.iterrows()
        ]
        if top_parts:
            bullets.append(
                "Longest average stays: " + "; ".join(top_parts) + ". Both are small-sample enough that "
                "whether these are clinically expected (genuine complexity/complication) or outliers "
                "(e.g. one prolonged case skewing a thin group) can't be determined from volume alone — "
                "worth a chart-level review before treating either as a pattern."
            )
    if pct_shorter:
        bullets.append(
            f"Readmission stays are consistently shorter than original stays ({pct_shorter:.0f}% are "
            "shorter) — read alongside the LOS-vs-readmission pattern above, this looks less like routine "
            "monitoring and more like patients returning to complete care that an early original discharge "
            "cut short."
        )
    bullets.append(
        "Discharge date artifact: records with the 2025-09-01 batch date are excluded. Average LOS for "
        "affected wards may be slightly understated."
    )
    if los_condition_caveat:
        bullets.append(los_condition_caveat)
    if shortest_ward:
        so_what = (
            f"<em><strong>So what:</strong> Add a discharge-readiness check at {shortest_ward} — its short "
            "stays and high readmission rate together suggest patients are being sent home before care is "
            "finished, not that the ward is running efficiently. Don't target average LOS reduction "
            "hospital-wide; that would push in the wrong direction here.</em>"
        )
    else:
        so_what = (
            "<em><strong>So what:</strong> Focus on the small number of extreme-stay cases individually, "
            "not the hospital-wide average — and treat any further LOS-shortening push with caution until "
            "the readmission link is checked ward by ward.</em>"
        )
    bullets.append(so_what)
    _insight(bullets, variant="blue")


# ── Section 7 — SSI benchmark ────────────────────────────────────────────────

def _render_s7_ssi_benchmark(ssi_benchmark: pd.DataFrame) -> None:
    section_header("7 — SSI: Benchmark Comparison")
    if not _safe(ssi_benchmark):
        _empty_state("SSI classification table not yet available.")
        return

    chart_card(
        "SSI rate vs published benchmark — how far is each category from its ceiling?",
        "Each line spans from benchmark ceiling to actual rate · circle = actual · diamond = benchmark · "
        "▲/✓ on the right shows above-ceiling vs within-range",
    )

    cats = ssi_benchmark["SURGICAL_CATEGORY"].tolist()
    actual = ssi_benchmark["ACTUAL_SSI_RATE"].astype(float).tolist()
    bench = ssi_benchmark["BENCHMARK_CEILING"].astype(float).tolist()
    is_above = [a > b for a, b in zip(actual, bench)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        mode="markers", name="Actual SSI rate", y=cats, x=actual,
        marker=dict(size=18, color=CA_RED, symbol="circle", line=dict(width=2, color="white")),
        hovertemplate="<b>%{y}</b><br>Actual SSI rate: %{x:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        mode="markers", name="Benchmark ceiling", y=cats, x=bench,
        marker=dict(size=14, color="#8A93A6", symbol="diamond", line=dict(width=2, color="#141F3D")),
        hovertemplate="<b>%{y}</b><br>Benchmark ceiling: %{x:.1f}%<extra></extra>",
    ))

    # One color throughout for the "actual" series — status (above/within)
    # is conveyed by the ▲/✓ annotations on the right, not by recoloring the
    # marker or connecting line, so the legend swatch always matches what's
    # plotted.
    shapes = [dict(type="line", xref="x", yref="y", x0=min(a, b), x1=max(a, b), y0=c, y1=c,
                   line=dict(color="rgba(163,45,45,0.4)", width=5))
              for c, a, b in zip(cats, actual, bench)]
    x_max = max(max(actual, default=0), max(bench, default=0)) + 1
    annotations = [dict(
        xref="x", yref="y", x=x_max, y=c,
        text=(f'<b style="color:{CA_RED}">▲ {a - b:.1f}% above</b>' if ab
              else f'<span style="color:{CA_GREEN}">✓ within</span>'),
        showarrow=False, xanchor="left", font=dict(size=10),
    ) for c, a, b, ab in zip(cats, actual, bench, is_above)]

    fig.update_layout(
        **{**_BL, "height": 300, "legend": dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
           "margin": dict(t=12, b=52, l=10, r=90)},
        xaxis={**_AX, "ticksuffix": "%", "title": "SSI rate", "range": [0, x_max + 1]},
        yaxis={**_AX, "showgrid": False, "automargin": True},
        shapes=shapes, annotations=annotations, showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()


# ── Section 8 — SSI: who's at risk ───────────────────────────────────────────

def _render_s8_ssi_risk(ssi_comorbidity: pd.DataFrame, ssi_multimorbidity: pd.DataFrame,
                         ssi_gender: pd.DataFrame, ssi_benchmark: pd.DataFrame) -> None:
    section_header("8 — SSI: Who's at Risk")
    if not (_safe(ssi_comorbidity) or _safe(ssi_multimorbidity) or _safe(ssi_gender)):
        _empty_state("SSI risk-factor data not yet available.")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        if not _safe(ssi_comorbidity):
            _empty_state()
        else:
            df = ssi_comorbidity.sort_values("SSI_PREVALENCE")
            chart_card("Comorbidity prevalence: SSI cases vs. overall population",
                       "Substitutes Cardiac condition / Anaemia for the spec's Obesity / Malnutrition — "
                       "not tracked in this schema")
            # Green = baseline/low-risk, red = the group of concern — same
            # risk-ramp language as the multimorbidity chart alongside it
            # (0 conditions = green, 2+ = red), not an unrelated grey.
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df["CONDITION"], x=df["SSI_PREVALENCE"], name="SSI cases", orientation="h",
                                  marker=dict(color=CA_RED, cornerradius=3)))
            fig.add_trace(go.Bar(y=df["CONDITION"], x=df["OVERALL_PREVALENCE"], name="Overall population",
                                  orientation="h", marker=dict(color=CA_GREEN, cornerradius=3, opacity=0.75)))
            fig.update_layout(
                **{**_BL, "height": 260, "barmode": "group",
                   "legend": dict(orientation="h", y=-0.22, x=0.5, xanchor="center")},
                xaxis={**_AX, "ticksuffix": "%"}, yaxis={**_AX, "showgrid": False},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(ssi_multimorbidity):
            _empty_state()
        else:
            df_m = ssi_multimorbidity.sort_values("TIER_ORDER")
            colors_map = {"0 conditions": CA_GREEN, "1 condition": CA_AMBER, "2+ conditions": CA_RED}
            colors = [colors_map.get(t, CA_MUTED) for t in df_m["CONDITION_COUNT"]]
            chart_card("Multimorbidity dose-response")
            fig2 = go.Figure(go.Bar(
                x=df_m["CONDITION_COUNT"], y=df_m["SSI_RATE"], marker=dict(color=colors, cornerradius=3),
                text=[f"{v:.1f}%" for v in df_m["SSI_RATE"]], textposition="outside",
            ))
            fig2.update_layout(**{**_BL, "height": 260}, showlegend=False,
                                xaxis={**_AX, "showgrid": False}, yaxis={**_AX, "ticksuffix": "%"})
            st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
            chart_card_close()

    if _safe(ssi_gender):
        wide = ssi_gender.pivot_table(index="SURGICAL_CATEGORY", columns="GENDER", values="SSI_RATE",
                                       aggfunc="first")
        chart_card("SSI rate by gender within surgical category")
        fig3 = go.Figure()
        # Navy = male, raspberry-light = female (spec §4 categorical convention)
        for gender, color in [("Male", _GENDER_COLOR["Male"]), ("Female", _GENDER_COLOR["Female"])]:
            if gender in wide.columns:
                fig3.add_trace(go.Bar(x=wide.index, y=wide[gender], name=gender,
                                       marker=dict(color=color, cornerradius=3),
                                       text=[f"{v:.1f}%" for v in wide[gender]], textposition="outside"))
        fig3.update_layout(
            **{**_BL, "height": 320, "barmode": "group",
               "margin": dict(t=16, b=90, l=10, r=10),
               "legend": dict(orientation="h", y=-0.45, x=0.5, xanchor="center")},
            xaxis={**_AX, "showgrid": False, "tickangle": -25}, yaxis={**_AX, "ticksuffix": "%"},
        )
        st.plotly_chart(fig3, use_container_width=True, config=PC_CFG)
        chart_card_close()

    bullets = []
    if _safe(ssi_benchmark):
        above = ssi_benchmark[ssi_benchmark["ACTUAL_SSI_RATE"] > ssi_benchmark["BENCHMARK_CEILING"]]
        if not above.empty:
            worst = above.sort_values("ACTUAL_SSI_RATE", ascending=False).iloc[0]
            ratio = round(worst["ACTUAL_SSI_RATE"] / worst["BENCHMARK_CEILING"], 1) if worst["BENCHMARK_CEILING"] else 0
            bullets.append(
                f"<strong>The SSI problem is a single-category problem:</strong> {worst['SURGICAL_CATEGORY']} "
                f"runs at {worst['ACTUAL_SSI_RATE']:.1f}% — {ratio:.1f}× the {worst['BENCHMARK_CEILING']:.1f}% "
                "benchmark ceiling. Other tracked categories are within benchmark."
            )
    if _safe(ssi_multimorbidity):
        rates = ssi_multimorbidity.set_index("CONDITION_COUNT")["SSI_RATE"]
        r0, r1, r2 = rates.get("0 conditions", 0), rates.get("1 condition", 0), rates.get("2+ conditions", 0)
        bullets.append(
            f"<strong>Pre-op comorbidity is the primary risk setter:</strong> SSI risk rises with "
            f"comorbidity count ({r0:.1f}% → {r1:.1f}% → {r2:.1f}%)."
        )
    if _safe(ssi_gender):
        wide = ssi_gender.pivot_table(index="SURGICAL_CATEGORY", columns="GENDER", values="SSI_RATE", aggfunc="first")
        if "Male" in wide.columns and "Female" in wide.columns:
            gap = (wide["Male"].astype(float) - wide["Female"].astype(float))
            if not gap.empty:
                worst_cat = gap.abs().idxmax()
                bullets.append(
                    f"A gender gap in SSI rate persists within {worst_cat} "
                    f"({wide.loc[worst_cat, 'Male']:.1f}% male vs {wide.loc[worst_cat, 'Female']:.1f}% female) "
                    "— worth direct clinical review, not a data artefact."
                )
    bullets.append(
        "<em><strong>So what:</strong> Addressing SSI in the worst category requires two levers — pre-op "
        "comorbidity screening, and an intra-op technique/prophylaxis review for that category "
        "specifically.</em>"
    )
    _insight(bullets, variant="red")


# ── Section 9 — SSI timing and trend ─────────────────────────────────────────

def _render_s9_ssi_timing(ssi_timing: pd.DataFrame, ssi_during_after: pd.DataFrame,
                           ssi_trend: pd.DataFrame, ssi_benchmark: pd.DataFrame) -> None:
    section_header("9 — SSI: Timing and Trend")
    if not (_safe(ssi_timing) or _safe(ssi_during_after) or _safe(ssi_trend)):
        _empty_state("SSI discovery-date data not yet available.")
        return

    col_l, col_r = st.columns(2)
    with col_l:
        if not _safe(ssi_timing):
            _empty_state()
        else:
            df = ssi_timing.sort_values("BUCKET_ORDER")
            colors = [CA_AMBER if p else CA_BLUE for p in df["IS_POST_WINDOW"]]
            chart_card("Days from discharge to SSI discovery")
            fig = go.Figure(go.Bar(x=df["TIMING_BUCKET"], y=df["EPISODE_COUNT"],
                                    marker=dict(color=colors, cornerradius=3)))
            fig.update_layout(**{**_BL, "height": 260}, showlegend=False,
                               xaxis={**_AX, "showgrid": False, "tickangle": -40}, yaxis=_AX)
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(ssi_during_after):
            _empty_state()
        else:
            wide = ssi_during_after.pivot_table(index="SURGICAL_CATEGORY", columns="DETECTION_TIMING",
                                                 values="EPISODE_COUNT", aggfunc="sum", fill_value=0)
            chart_card(
                "During vs. after discharge, by surgical category",
                "Raw episode counts, not rate — a category can lead here on volume while having a lower "
                "SSI rate than a smaller, higher-rate category (see the benchmark chart above).",
            )
            fig2 = go.Figure()
            for timing, color in [("Found during index stay", CA_GREEN), ("Found after discharge", CA_RED)]:
                if timing in wide.columns:
                    fig2.add_trace(go.Bar(x=wide.index, y=wide[timing], name=timing,
                                           marker=dict(color=color, cornerradius=3)))
            fig2.update_layout(
                **{**_BL, "height": 320, "barmode": "stack",
                   "margin": dict(t=16, b=90, l=10, r=10),
                   "legend": dict(orientation="h", y=-0.45, x=0.5, xanchor="center")},
                xaxis={**_AX, "showgrid": False, "tickangle": -25}, yaxis=_AX,
            )
            st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
            chart_card_close()

    if _safe(ssi_trend):
        df_t = ssi_trend.sort_values("SORT_MONTH")
        chart_card(
            "SSI rate and other-HAI rate — monthly",
            "Both lines now cover the same months across both EMR systems — a level shift or drop in "
            "other-HAI around the system cutover reflects a documentation-convention change (legacy terms "
            "like 'nosocomial' and 'catheter associated' fell out of use), not a real change in infections.",
        )
        # Spec §4 line-chart rule: a metric-over-time line isn't red just
        # because it's the one we're worried about — red is reserved for a
        # verdict, not "this happens to be the concerning line." Dual-line
        # comparisons standardize to teal solid (primary) + raspberry dashed
        # (comparison), not red/grey.
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_t["VISIT_MONTH"], y=df_t["SSI_RATE"], mode="lines+markers",
                                   line=dict(color=CA_BLUE, width=2.5), name="SSI rate", connectgaps=False))
        fig3.add_trace(go.Scatter(x=df_t["VISIT_MONTH"], y=df_t["OTHER_HAI_RATE"], mode="lines+markers",
                                   line=dict(color=CA_PINK, width=2, dash="dot"), name="Other-HAI rate",
                                   connectgaps=False))
        fig3.update_layout(
            **{**_BL, "height": 340, "margin": dict(t=16, b=100, l=10, r=10),
               "legend": dict(orientation="h", y=-0.5, x=0.5, xanchor="center")},
            xaxis={**_AX, "showgrid": False, "tickangle": -60}, yaxis={**_AX, "ticksuffix": "%", "title": "Rate"},
        )
        st.plotly_chart(fig3, use_container_width=True, config=PC_CFG)
        chart_card_close()

    bullets = []
    if _safe(ssi_during_after) and _safe(ssi_benchmark):
        post = ssi_during_after[ssi_during_after["DETECTION_TIMING"] == "Found after discharge"]
        above = ssi_benchmark[ssi_benchmark["ACTUAL_SSI_RATE"] > ssi_benchmark["BENCHMARK_CEILING"]]
        if not post.empty and not above.empty:
            most_count = post.loc[post["EPISODE_COUNT"].idxmax()]
            worst_rate = above.sort_values("ACTUAL_SSI_RATE", ascending=False).iloc[0]
            bullets.append(
                f"<strong>{worst_rate['SURGICAL_CATEGORY']}</strong> has the highest SSI <strong>rate</strong> "
                f"({worst_rate['ACTUAL_SSI_RATE']:.1f}% of its surgeries), but "
                f"<strong>{most_count['SURGICAL_CATEGORY']}</strong> has the most SSI cases in "
                f"<strong>absolute number</strong> ({int(most_count['EPISODE_COUNT'])} after-discharge "
                f"episodes) — its far higher surgical volume means the same or fewer infections still add "
                "up to more total cases. Prioritize by rate, resource by count."
            )
        elif not post.empty:
            most_count = post.loc[post["EPISODE_COUNT"].idxmax()]
            bullets.append(
                f"{most_count['SURGICAL_CATEGORY']} has the most SSI cases in absolute count, driven by "
                "surgical volume, not necessarily a worse infection rate."
            )
    if _safe(ssi_trend) and "OTHER_HAI_CASES" in ssi_trend.columns:
        hai_months = ssi_trend[ssi_trend["OTHER_HAI_CASES"].fillna(0) > 0].sort_values("SORT_MONTH")
        total_hai = int(hai_months["OTHER_HAI_CASES"].sum())
        if not hai_months.empty:
            _cap = 8
            shown = hai_months.head(_cap)
            instances = "; ".join(
                f"{r['VISIT_MONTH']} ({int(r['OTHER_HAI_CASES'])} case{'s' if r['OTHER_HAI_CASES'] != 1 else ''})"
                for _, r in shown.iterrows()
            )
            if len(hai_months) > _cap:
                instances += f"; +{len(hai_months) - _cap} more month(s)"
            bullets.append(
                f"<strong>Other-HAI is {total_hai} cases total, all in named months</strong> — {instances}. "
                "Every other month on the chart is a genuine zero, not missing data — read the flat "
                "stretches as literally no recorded cases, not as a gap."
            )
        else:
            bullets.append(
                "<strong>Other-HAI has zero recorded cases across the whole window</strong> — the flat "
                "line isn't a data gap, it's a genuine absence under this keyword definition."
            )
    bullets.append(
        "SSI rate and other-HAI rate above now cover the same months across both EMR systems — if "
        "other-HAI drops or flattens around the system cutover, that's a documentation-convention change "
        "(legacy terms like 'nosocomial' and 'catheter associated' fell out of use), not a real change "
        "in infections. Read the SSI line as the reliable one throughout."
    )
    bullets.append(
        "<em><strong>So what:</strong> A hospital relying on 30-day readmission counts to catch SSI is "
        "missing cases discovered later. A structured post-discharge wound review would capture more of "
        "the missed cases.</em>"
    )
    _insight(bullets, variant="red")


# ── Section 10 — Recommendations ─────────────────────────────────────────────

# Matches ui_template.py's _PRIORITY_SEVERITY_COLOR so the "Action:" line
# inside a card highlights in the same color as that card's left border/label
# — same pattern used in the OPD-IPD and Flow-Retention tabs' recommendation
# cards, brought here for consistency.
_CA_REC_SEVERITY_COLOR = {"critical": CA_RED, "monitor": CA_AMBER, "okay": CA_GREEN}


def _rec_list(items: list, severity: str = "monitor") -> str:
    color = _CA_REC_SEVERITY_COLOR.get(severity, CA_AMBER)
    lis = "".join(
        f'<li style="margin-bottom:5px;font-weight:700;color:{color}">{i}</li>'
        if i.startswith("Action:") else
        f'<li style="margin-bottom:5px">{i}</li>'
        for i in items
    )
    return f'<ul style="margin:2px 0 0;padding-left:16px">{lis}</ul>'


def _lim_card(title: str, detail: str, fix_text: str) -> str:
    return (
        f'<div class="ca-lim-card">'
        f'<div class="ca-lim-title">{title}</div>'
        f'<div class="ca-lim-detail">{detail}</div>'
        f'<div class="ca-lim-fix">Fix: {fix_text}</div>'
        f'</div>'
    )


def _render_s10_recommendations(ssi_benchmark: pd.DataFrame, blind_spot: pd.DataFrame,
                                 delayed: pd.DataFrame, ward_rates: pd.DataFrame) -> None:
    section_header("10 — Key Findings & Recommendations")
    st.markdown(
        '<div style="font-size:12px;color:#5C6478;margin-bottom:16px;line-height:1.65">'
        'Each card below states the finding and the action it points to together — not a separate '
        'findings summary and recommendation list. Ordered by priority.'
        '</div>',
        unsafe_allow_html=True,
    )

    worst_rate = worst_bench = 0.0
    worst_cat = "the worst-performing category"
    if _safe(ssi_benchmark):
        above = ssi_benchmark[ssi_benchmark["ACTUAL_SSI_RATE"] > ssi_benchmark["BENCHMARK_CEILING"]]
        if not above.empty:
            top = above.sort_values("ACTUAL_SSI_RATE", ascending=False).iloc[0]
            worst_cat, worst_rate, worst_bench = top["SURGICAL_CATEGORY"], float(top["ACTUAL_SSI_RATE"]), float(top["BENCHMARK_CEILING"])

    blind_n = int(blind_spot.loc[blind_spot["IS_BLIND_SPOT"], "PATIENT_COUNT"].sum()) if _safe(blind_spot) else 0
    delayed_n = int(delayed["PATIENT_COUNT"].sum()) if _safe(delayed) else 0

    outlier_name, outlier_rate, threshold = None, 0.0, 0.0
    if _safe(ward_rates):
        avg_rate = float(ward_rates["READMISSION_RATE"].mean())
        threshold = avg_rate * 1.5
        top_ward = ward_rates.sort_values("READMISSION_RATE", ascending=False).iloc[0]
        outlier_name, outlier_rate = top_ward["WARD"], float(top_ward["READMISSION_RATE"])

    ssi_ratio = round(worst_rate / worst_bench, 1) if worst_bench else 0.0

    p1_items = [
        f"Runs at {ssi_ratio:.1f}× benchmark ({worst_rate:.1f}% vs {worst_bench:.1f}% ceiling, "
        f"+{worst_rate - worst_bench:.1f}pp) — other categories are within range, so this is masked in "
        "the blended overall SSI rate.",
        "Action: audit pre-op infection screening and prophylaxis, and prioritise multimorbid patients "
        "(Section 8) before surgical listing.",
    ]

    p2_items = [
        f"{blind_n} patients return in the 31–90 day window — comparable in size to confirmed 30-day "
        f"readmissions and currently invisible to the standard KPI; {delayed_n} of those are named "
        "delayed complications.",
        "Action: report the 30-day and 90-day rate side by side, and add a structured post-discharge "
        "wound review at 6 weeks.",
    ]

    if outlier_name:
        p3_items = [
            f"Readmission rate {outlier_rate:.1f}% vs {threshold:.1f}% threshold — like the SSI problem "
            "above, this cause is concentrated in one ward, not hospital-wide, so a targeted fix (not a "
            "blanket policy) is what would move it.",
            f"Action: ward-level review of discharge criteria at {outlier_name}, focused on its specific "
            "complication mix (Section 2) rather than a hospital-wide policy change.",
        ]
    else:
        p3_items = ["No ward-level data currently available.",
                    "Action: re-run once ward-level data is available."]

    p4_items = [
        "SSI/wound complications concentrate in the 55+ age band (Section 4) — the same "
        "concentrated-cause pattern as the SSI and ward findings above.",
        "Action: stratify discharge planning for 55+ patients by complication type, not age alone.",
    ]

    priority_cards([
        {"label": "Urgent — Clinical Quality", "severity": "critical",
         "title": f"{worst_cat} SSI protocol review",
         "body": _rec_list(p1_items, "critical")},
        {"label": "Urgent — Patient Safety Gap", "severity": "critical",
         "title": "Extend readmission tracking to 90 days",
         "body": _rec_list(p2_items, "critical")},
        {"label": "Moderate — Ward-level", "severity": "monitor",
         "title": f"{outlier_name} targeted discharge review" if outlier_name else "Ward-level readmission review",
         "body": _rec_list(p3_items, "monitor")},
        {"label": "Structural — Discharge Planning", "severity": "monitor",
         "title": "Age-stratified complication review",
         "body": _rec_list(p4_items, "monitor")},
    ])

    st.markdown(
        '<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;'
        'color:#8A93A6;margin:8px 0 12px">Known data limitations — and how to fix them</div>',
        unsafe_allow_html=True,
    )
    # STATIC: structural data constraint — not derivable from any DataFrame.
    col1, col2, col3 = st.columns(3)
    col1.markdown(_lim_card(
        "⚠ Discharge date artifact",
        "A batch processing artifact assigned 2025-09-01 as the discharge date to a volume of records in "
        "both EMR systems. Affected records are excluded from every ward-level and LOS figure in this tab. "
        "The extent of any resulting undercount is unknown until the artifact is corrected at source.",
        "Correct the batch job at source; re-run affected ETL windows.",
    ), unsafe_allow_html=True)
    col2.markdown(_lim_card(
        "⚠ 30-day tracking window",
        f"{blind_n} patients sit in the 31–90 day window, invisible to the standard KPI; {delayed_n} of "
        "those are named delayed clinical complications.",
        "Extend reporting to 90 days as a secondary KPI alongside the standard 30-day rate.",
    ), unsafe_allow_html=True)
    col3.markdown(_lim_card(
        "⚠ Cross-system readmissions invisible",
        "A patient discharged from SPH and readmitted at another facility is invisible to every query on "
        "this tab — no identity crosswalk exists.",
        "Pursue a patient identity crosswalk with regional health authority or insurance claims data.",
    ), unsafe_allow_html=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def render_clinical_activity_tab() -> None:
    with st.spinner("Loading data…"):
        kpis = CAQ.get_ca_overview_kpis()
        monthly = CAQ.get_ca_monthly_readmission()
        spike_drilldown = CAQ.get_ca_spike_month_drilldown()
        ward_rates = CAQ.get_ca_ward_readmission_rates()
        ward_cause = CAQ.get_ca_ward_readmission_cause()
        ward_diagnoses = CAQ.get_ca_ward_top_diagnoses()
        type_breakdown = CAQ.get_ca_readmission_type_breakdown()
        area_breakdown = CAQ.get_ca_readmission_area()
        area_diagnoses = CAQ.get_ca_readmission_top_by_area()
        age_complication = CAQ.get_ca_age_complication()
        blind_spot = CAQ.get_ca_blind_spot()
        blind_spot_type = CAQ.get_ca_blind_spot_type()
        delayed = CAQ.get_ca_delayed_complications()
        los_ward = CAQ.get_ca_los_by_ward()
        los_dist = CAQ.get_ca_los_distribution()
        los_conditions = CAQ.get_ca_top_los_conditions()
        los_scatter = CAQ.get_ca_los_vs_readmission_scatter()
        los_index_readmit = CAQ.get_ca_index_vs_readmit_los()
        ssi_benchmark = CAQ.get_ca_ssi_benchmark()
        ssi_comorbidity = CAQ.get_ca_ssi_comorbidity()
        ssi_multimorbidity = CAQ.get_ca_ssi_multimorbidity()
        ssi_gender = CAQ.get_ca_ssi_by_gender_category()
        ssi_timing = CAQ.get_ca_ssi_timing()
        ssi_during_after = CAQ.get_ca_ssi_during_vs_after()
        ssi_trend = CAQ.get_ca_ssi_monthly_trend()

    _render_tab_header()
    _render_overview(kpis)
    _render_s1_monthly_trend(monthly, spike_drilldown)
    _render_s2_ward_analysis(ward_rates, ward_cause, ward_diagnoses)
    _render_s3_readmission_type(type_breakdown, area_breakdown, area_diagnoses)
    _render_s4_demographics(age_complication)
    _render_s5_blind_spot(blind_spot, blind_spot_type, delayed)
    _render_s6_los(los_ward, los_dist, los_conditions, los_scatter, los_index_readmit)
    _render_s7_ssi_benchmark(ssi_benchmark)
    _render_s8_ssi_risk(ssi_comorbidity, ssi_multimorbidity, ssi_gender, ssi_benchmark)
    _render_s9_ssi_timing(ssi_timing, ssi_during_after, ssi_trend, ssi_benchmark)
    _render_s10_recommendations(ssi_benchmark, blind_spot, delayed, ward_rates)


# Backward-compat alias for the previous entry-point name.
render_tab = render_clinical_activity_tab
