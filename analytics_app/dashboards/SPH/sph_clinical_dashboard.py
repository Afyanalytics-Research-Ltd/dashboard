"""
sph/opd_ipd_dashboard.py
=========================
Main entry point for the SPH Clinical Operations dashboard.

Run with:
    streamlit run sph/opd_ipd_dashboard.py

Tab routing is handled via st.session_state["active_tab"].
Each sidebar nav item sets the active tab; the main area renders accordingly.

Tabs:
  Overview                    — Template A page: KPIs, stage row, issues table, synthesis
  OPD → IPD conversion        — Template B page: full 10-section clinical analysis
  Clinical activity           — readmissions, length of stay, and SSI (12 sections)
  Flow and retention          — placeholder (not yet built)
  Disease burden              — placeholder (not yet built)
  Case mix                    — segment composition, growth, comorbidity (7 sections)
  Data quality                — placeholder (not yet built)
"""

import sys
import os
import base64
from urllib.parse import quote

# Parent of sph/ must be on the path for sph.clinicals.opd_ipd_module.* to resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also insert sph/ itself so intra-module imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

import sph.clinicals.opd_ipd_module.queries as Q
import sph.clinicals.opd_ipd_module.views   as V
from sph.clinicals.opd_ipd_module.ui_template import (
    inject_css,
    page_header,
    section_header,
    kpi_row,
    stage_row,
    issues_table,
    overview_synthesis,
    tab_synthesis,
    fmt_num,
    fmt_pct,
    ACCENT_INFO, ACCENT_CRITICAL, ACCENT_MONITOR, ACCENT_POSITIVE, ACCENT_NEUTRAL,
    PRIMARY, DANGER, WARNING, SUCCESS, NEUTRAL,
    BORDER, TEXT_PRI, TEXT_SEC, SURFACE_1,
    overview_tab_section, overview_key_findings,
    CHART_LAYOUT, AXIS_X, AXIS_Y, PC_CFG,
)
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPH — Clinical Operations",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
    '&display=swap" rel="stylesheet">'
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.47.0/'
    'tabler-icons.min.css">',
    unsafe_allow_html=True,
)
inject_css()

# ── Session state — default to Overview ──────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Overview"

# ── Sidebar nav structure ────────────────────────────────────────────────────
# Grouped nav — matches afya_sidebar_redesign.html's section structure
# (Overview / Patient flow / Clinical / System). Rendered as real <a> links
# (not st.button) so Tabler <i> icon glyphs actually render — st.button's
# label is plain text and cannot hold HTML. Clicking a link navigates via
# a "tab" query param, which Streamlit picks up on rerun without any JS.
_TAB_GROUPS = [
    ("Overview", [
        ("ti-layout-dashboard", "Overview"),
    ]),
    ("Patient flow", [
        ("ti-list-check", "Case mix"),
        ("ti-transform",  "OPD → IPD conversion"),
        ("ti-repeat",     "Flow and retention"),
    ]),
    ("Clinical", [
        ("ti-activity", "Clinical activity"),
        ("ti-virus",    "Disease burden"),
    ]),
    ("System", [
        ("ti-shield-check", "Data quality"),
    ]),
]
_TABS = [tab for _, tabs in _TAB_GROUPS for tab in tabs]
_TAB_LABELS = [label for _, label in _TABS]

_qtab = st.query_params.get("tab")
if _qtab and _qtab in _TAB_LABELS and _qtab != st.session_state["active_tab"]:
    st.session_state["active_tab"] = _qtab

# Disease burden sub-tab — same query-param pattern as the top-level tabs,
# since plain divs with cursor:pointer have no click handler in Streamlit.
_DISEASE_BURDEN_SUBTABS = ["Orthopedics", "Maternal health"]
if "disease_burden_sub_tab" not in st.session_state:
    st.session_state["disease_burden_sub_tab"] = "Orthopedics"
_qsub = st.query_params.get("sub")
if _qsub and _qsub in _DISEASE_BURDEN_SUBTABS and _qsub != st.session_state["disease_burden_sub_tab"]:
    st.session_state["disease_burden_sub_tab"] = _qsub

with st.sidebar:
    # Logo — main sidebar header, replaces the old brand block entirely
    _logo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "logo", "st_peters_logo.png",
    )
    _logo_html = ""
    if os.path.exists(_logo_path):
        with open(_logo_path, "rb") as _f:
            _logo_b64 = base64.b64encode(_f.read()).decode()
        _logo_html = (
            f'<img src="data:image/png;base64,{_logo_b64}" '
            f'style="max-width:100%;max-height:48px;object-fit:contain" />'
        )
    st.markdown(
        f"""
        <div style="background:#FFFFFF;border:0.5px solid {BORDER};
                    border-radius:8px;padding:10px 12px;margin-bottom:14px">
          <div style="display:flex;align-items:center;justify-content:center">
            {_logo_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tab navigation — icon-tile rows (rounded colored icon box + label),
    # active state as a solid full-width teal pill with a translucent white
    # icon tile inside it. Per-group tile tint keeps the same visual
    # grouping as the section headers without repeating a border every row.
    _active_label = st.session_state["active_tab"]
    _GROUP_TILE = {
        "Overview":     {"bg": "#DCEFE9", "fg": PRIMARY},
        "Patient flow": {"bg": "#DCEFE9", "fg": PRIMARY},
        "Clinical":     {"bg": "#FBE2E9", "fg": "#C13868"},
        "System":       {"bg": "#FBE2E9", "fg": "#C13868"},
    }

    groups_html = ""
    for group_name, tabs in _TAB_GROUPS:
        tile = _GROUP_TILE.get(group_name, {"bg": "#DCEFE9", "fg": PRIMARY})
        groups_html += (
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:#8A93A6;'
            f'margin:14px 4px 6px">{group_name}</div>'
        )
        for icon_class, label in tabs:
            is_active = label == _active_label
            row_bg = PRIMARY if is_active else "transparent"
            text_color = "#FFFFFF" if is_active else TEXT_PRI
            weight = "600" if is_active else "500"
            icon_bg = "rgba(255,255,255,0.22)" if is_active else tile["bg"]
            icon_fg = "#FFFFFF" if is_active else tile["fg"]
            href = f"?tab={quote(label)}"
            groups_html += (
                f'<a href="{href}" target="_self" style="text-decoration:none;display:flex;'
                f'align-items:center;gap:10px;padding:7px 8px;border-radius:10px;'
                f'background:{row_bg};color:{text_color};font-weight:{weight};'
                f'font-size:13px;margin-bottom:3px">'
                f'<span style="display:flex;align-items:center;justify-content:center;'
                f'width:30px;height:30px;min-width:30px;border-radius:8px;background:{icon_bg}">'
                f'<i class="ti {icon_class}" style="font-size:15px;color:{icon_fg}"></i></span>'
                f'<span>{label}</span></a>'
            )

    st.markdown(f'<div>{groups_html}</div>', unsafe_allow_html=True)

    # Footer — user block + refresh action, per redesign
    st.markdown(f"<hr style='border:none;border-top:0.5px solid {BORDER};margin:10px 0 8px'>",
                unsafe_allow_html=True)
    if st.button("↻  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    import sph.clinicals.email_digest as EmailDigest
    EmailDigest.render_sidebar_control()


# ── Route to the active tab ───────────────────────────────────────────────────
active = st.session_state["active_tab"]


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW TAB  (v2 — per build spec: KPI row + 6 tab sections + key findings)
# ─────────────────────────────────────────────────────────────────────────────
if active == "Overview":
    page_header(
        "Hospital at a Glance",
        subtitle="Performance highlights across all service areas",
    )

    # Pull real numbers from every other tab's own query module wherever the
    # live Snowflake connection is available; fall back to the spec's
    # confirmed figures (below) when a query returns empty (e.g. no
    # credentials in this environment) so the page still renders correctly.
    import sph.clinicals.case_mix_module.cm_queries as CMQ
    import sph.clinicals.flow_retention_module.fr_queries as FRQ
    import sph.clinicals.clinical_activity_module.ca_queries as CAQ
    import sph.clinicals.disease_burden_module.orthopedics.orth_queries as ORQ
    import sph.clinicals.disease_burden_module.maternal.mat_queries as MAQ
    from sph.clinicals.case_mix_module.cm_views import _seasonal_trend_forecast

    def _safe(fn, *a):
        try:
            df = fn(*a)
            return df if df is not None and not df.empty else None
        except Exception:
            return None

    def _empty_chart():
        st.markdown(
            f'<div style="padding:24px;text-align:center;color:{NEUTRAL};'
            f'font-size:11px;font-style:italic">No data available</div>',
            unsafe_allow_html=True,
        )

    def _fmt(v, suffix="%", digits=1, default="—"):
        return f"{v:.{digits}f}{suffix}" if v is not None else default

    # Short display labels reused across every chart on this page — same
    # segment must read the same everywhere, per the color/naming standard.
    _SEG_SHORT = {
        "Core Orthopedics: General": "Ortho General",
        "Core Orthopedics: Spine and Back Pain Care": "Spine & Back Pain",
        "Core General Surgery": "Gen Surgery",
        "Standalone Specialty: Obstetrics & Gynaecology": "Obs & Gynae",
        "Standalone Specialty: Urology": "Urology",
        "Standalone Specialty: ENT": "ENT",
        "Standalone Specialty: Neurosurgery (structural/acute)": "Neurosurgery",
        "Standalone Specialty: Plastic Surgery": "Plastic Surgery",
        "Standalone Specialty: Dental": "Dental",
        "Standalone Specialty: Eye/Ophthalmology": "Eye/Ophthalmology",
        "Standalone Specialty: Maxillofacial": "Maxillofacial",
        "Standalone Medical: Sepsis/Infection": "Sepsis/Infection",
        "Standalone Medical: Cardiovascular": "Cardiovascular",
        "Standalone Medical: Endocrine/Metabolic": "Endocrine/Metabolic",
        "Standalone Medical: Neurology (chronic/medical)": "Neurology",
        "Other General Outpatient": "Other OPD",
    }

    # ── Live data — every number on this page traces to a query that also
    # backs its own sub-tab, so nothing here can drift out of sync with it.
    df_cm_kpis      = _safe(CMQ.get_cm_headline_kpis)
    df_cm_yearly    = _safe(CMQ.get_cm_yearly_trend)
    df_cm_other_trend   = _safe(CMQ.get_cm_other_opd_trend)
    df_cm_other_monthly = _safe(CMQ.get_cm_other_opd_monthly)
    df_fr_status    = _safe(FRQ.get_fr_status_overall)
    df_fr_retention_trend = _safe(FRQ.get_fr_retention_trend)
    df_fr_gs        = _safe(FRQ.get_fr_general_surgery_counterexample)
    df_fr_visit_seg = _safe(FRQ.get_fr_ltfu_by_segment_and_visit_number)
    df_fr_outreach  = _safe(FRQ.get_fr_spine_structural_outreach_list)
    df_opd_kpis     = _safe(Q.get_headline_kpis)
    df_opd_segment  = _safe(Q.get_segment_conversion)
    df_opd_workload = _safe(Q.get_workload_vs_conversion)
    df_opd_comorbid = _safe(Q.get_comorbidity_conversion)
    df_ca_kpis      = _safe(CAQ.get_ca_overview_kpis)
    df_ca_ward_rate = _safe(CAQ.get_ca_ward_readmission_rates)
    df_ca_los_ward  = _safe(CAQ.get_ca_los_by_ward)
    df_orth_spine   = _safe(ORQ.get_orth_spine_casetype_by_year)
    df_orth_vte     = _safe(ORQ.get_orth_vte_compliance)
    df_orth_compl   = _safe(ORQ.get_orth_complications)
    df_mat_anc      = _safe(MAQ.get_mat_anc_visit_distribution)
    df_mat_qual_a   = _safe(MAQ.get_mat_anc_quality_part_a)
    df_mat_qual_b   = _safe(MAQ.get_mat_anc_quality_part_b)
    df_mat_workup   = _safe(MAQ.get_mat_haemorrhage_workup)

    total_visits = core_ortho_pct = blended_conv = None
    if df_cm_kpis is not None:
        total_visits = int(df_cm_kpis.iloc[0]["TOTAL_VISITS"])
        core_ortho_pct = float(df_cm_kpis.iloc[0]["CORE_ORTHO_SHARE_PCT"])
    if df_opd_kpis is not None:
        blended_conv = float(df_opd_kpis.iloc[0]["OVERALL_CONVERSION_PCT"])

    readmission_rate = r_min = r_max = worst_ssi = worst_ssi_bench = blind_spot = None
    worst_ssi_cat = "—"
    if df_ca_kpis is not None:
        r = df_ca_kpis.iloc[0]
        readmission_rate = float(r.get("READMISSION_RATE", 0) or 0)
        r_min = float(r.get("READMISSION_RATE_MIN", 0) or 0)
        r_max = float(r.get("READMISSION_RATE_MAX", 0) or 0)
        worst_ssi = float(r.get("WORST_SSI_RATE", 0) or 0)
        worst_ssi_cat = r.get("WORST_SSI_CATEGORY", "—")
        worst_ssi_bench = float(r.get("WORST_SSI_BENCHMARK", 0) or 0)
        blind_spot = int(r.get("BLIND_SPOT_COUNT", 0) or 0)
    ssi_ratio = round(worst_ssi / worst_ssi_bench, 1) if worst_ssi and worst_ssi_bench else None

    active_pct = lapsing_pct = ltfu_pct = retention_pct = None
    if df_fr_status is not None:
        s = df_fr_status.set_index("STATUS")["PCT_OF_CLASSIFIABLE_PATIENTS"]
        active_pct = float(s.get("Active", 0))
        lapsing_pct = float(s.get("Lapsing", 0))
        ltfu_pct = float(s.get("LTFU", 0))
        retention_pct = active_pct + lapsing_pct

    anc_single_pct = anc_4plus_pct = None
    if df_mat_anc is not None:
        single = df_mat_anc[df_mat_anc["VISIT_COUNT_BUCKET"].str.startswith("1 visit")]
        four_p = df_mat_anc[df_mat_anc["VISIT_COUNT_BUCKET"].str.startswith("4+")]
        anc_single_pct = float(single.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not single.empty else None
        anc_4plus_pct = float(four_p.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not four_p.empty else None

    # Severity → accent-color mapping shared by both strips, matching the
    # top-border-accent kpi_row() style used on every other tab (not a full
    # tinted background, which is a different, Overview-only pattern).
    _SEV_ACCENT = {"neutral": ACCENT_NEUTRAL, "success": ACCENT_POSITIVE,
                   "warning": ACCENT_MONITOR, "danger": ACCENT_CRITICAL}

    def _kpi_spacer():
        st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # Three equal rows of 3 — same column count every row so every card is
    # the same size, with a spacer between rows so they don't sit flush
    # against each other.
    kpi_row([
        {"label": "Total visits", "value": f"{total_visits:,}" if total_visits else "—",
         "delta": "Jun 2022–Jun 2026", "accent_color": _SEV_ACCENT["neutral"]},
        {"label": "Core orthopedics share", "value": _fmt(core_ortho_pct),
         "delta": "expected core identity", "accent_color": _SEV_ACCENT["success"]},
        {"label": "Blended conversion", "value": _fmt(blended_conv),
         "delta": "dominated by ortho volume", "accent_color": _SEV_ACCENT["warning"]},
    ])
    _kpi_spacer()
    kpi_row([
        {"label": "Readmission rate", "value": _fmt(readmission_rate),
         "delta": f"Lowest month {r_min:.0f}%, highest {r_max:.0f}%" if r_min is not None else "",
         "accent_color": _SEV_ACCENT["danger"]},
        {"label": "Retention rate", "value": _fmt(retention_pct),
         "delta": f"Lapsing {_fmt(lapsing_pct)}" if lapsing_pct is not None else "",
         "accent_color": _SEV_ACCENT["warning" if (lapsing_pct or 0) >= 15 else "success"]},
        {"label": "Loss to follow-up", "value": _fmt(ltfu_pct),
         "delta": "rising every month", "accent_color": _SEV_ACCENT["danger"]},
    ])
    _kpi_spacer()
    kpi_row([
        {"label": "Worst SSI vs benchmark", "value": _fmt(worst_ssi),
         "delta": f"{worst_ssi_cat}, {ssi_ratio}x ceiling" if ssi_ratio else str(worst_ssi_cat),
         "accent_color": _SEV_ACCENT["danger"]},
        {"label": "ANC single-visit rate", "value": _fmt(anc_single_pct),
         "delta": "continuity failure", "accent_color": _SEV_ACCENT["danger"]},
        {"label": "ANC reaches quality threshold", "value": _fmt(anc_4plus_pct),
         "delta": "4+ visits — published outcome threshold", "accent_color": _SEV_ACCENT["danger"]},
    ])
    _kpi_spacer()

    # ── Section 1 — Case mix: growth & demand ────────────────────────────────
    # Cumulative first-year-vs-last-year growth — the exact same metric as
    # the Case Mix tab's own "Growth, {year_min}→{year_max}" card (render_s4
    # in cm_views.py), not year-over-year. Year-over-year against a partial
    # final year (2026 is only ~6 months of data) made almost every segment
    # look like it was declining, which didn't match the real tab at all.
    _cm_year_min = _cm_year_max = None
    _cm_growth = None
    if df_cm_yearly is not None:
        years = sorted(df_cm_yearly["VISIT_YEAR"].unique())
        _cm_year_min, _cm_year_max = int(years[0]), int(years[-1])
        wide = df_cm_yearly.pivot_table(index="PRIMARY_VISIT_SEGMENT", columns="VISIT_YEAR",
                                         values="TOTAL_VISITS", fill_value=0)
        if _cm_year_min in wide.columns and _cm_year_max in wide.columns:
            v_start, v_end = wide[_cm_year_min], wide[_cm_year_max]
            valid = v_start > 0
            _cm_growth = (100.0 * (v_end - v_start) / v_start)[valid].sort_values()

    def _chart_case_mix():
        if _cm_growth is None or _cm_growth.empty:
            _empty_chart()
            return
        # Both ends of the distribution, not just the top — a couple of tiny-
        # base segments growing 1000%+ (e.g. Dental) were drowning out every
        # declining segment entirely. Bar height is capped at 100% so one
        # outlier doesn't compress everything else to a sliver; the label
        # shows the same capped value as the bar height for consistency.
        d = pd.concat([_cm_growth.head(4), _cm_growth.tail(4)]).drop_duplicates().sort_values()
        labels = [_SEG_SHORT.get(s, s) for s in d.index]
        vals = d.tolist()
        capped = [max(min(v, 100), -100) for v in vals]
        colors = [DANGER if v < 0 else (SUCCESS if v > 20 else WARNING) for v in vals]
        fig = go.Figure(go.Bar(
            x=labels, y=capped, marker_color=colors,
            text=[f"{v:+.0f}%" for v in capped], textposition="outside",
            textfont=dict(size=11),
            cliponaxis=False,
        ))
        # Extra height + generous top/bottom margin so the outside labels on
        # capped bars (sitting right at the ±100 edge) don't get clipped by
        # the plot boundary — that's what made them unreadable before.
        fig.update_layout(**{**CHART_LAYOUT, "height": 280, "margin": {**CHART_LAYOUT.get("margin", {}), "t": 40, "b": 90}})
        fig.update_xaxes(**AXIS_X, tickangle=-25)
        fig.update_yaxes(**AXIS_Y, ticksuffix="%", range=[-135, 135], title="Growth %")
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    cm_issues = []
    if _cm_growth is not None and not _cm_growth.empty:
        if "Core General Surgery" in _cm_growth.index:
            v = float(_cm_growth["Core General Surgery"])
            cm_issues.append({
                "severity": "danger", "tag": "Declining",
                "body": f"Core General Surgery volume {'fell' if v < 0 else 'grew'} {abs(v):.1f}% from "
                        f"{_cm_year_min} to {_cm_year_max} — this is the highest-converting segment on "
                        "this page, so the change matters more than its share of total volume suggests.",
            })
        if "Standalone Medical: Sepsis/Infection" in _cm_growth.index:
            v = float(_cm_growth["Standalone Medical: Sepsis/Infection"])
            cm_issues.append({
                "severity": "danger", "tag": "Growing",
                "body": f"Sepsis/Infection cases grew {v:.1f}% from {_cm_year_min} to {_cm_year_max} — a "
                        "clinical-risk category, not just a volume trend to note.",
            })
        opp_segs = ["Standalone Specialty: Urology", "Standalone Specialty: Obstetrics & Gynaecology",
                    "Standalone Specialty: ENT"]
        opp = _cm_growth[_cm_growth.index.isin(opp_segs)]
        if not opp.empty:
            parts = ", ".join(f"{_SEG_SHORT.get(s, s)} ({v:+.0f}%)" for s, v in opp.items())
            cm_issues.append({"severity": "success", "tag": "Opportunity",
                               "body": f"{parts} are all growing from a small base — worth investment "
                                       f"attention as genuine opportunities, not just noise in the totals."})
    if core_ortho_pct is not None:
        cm_issues.append({
            "severity": "warning", "tag": "Concentrated",
            "body": f"Spine & Back Pain and Ortho General together account for {core_ortho_pct:.1f}% of "
                    "all visits. Spine's own growth is led by recurring patients, especially from Feb 2025 "
                    "onward — read alongside the known Feb 2025 diagnosis-coding-change artifact, which "
                    "inflates part of that same period.",
        })

    overview_tab_section(
        tab_name="Case mix",
        tab_tag=f"Core orthopedics {core_ortho_pct:.1f}% of volume" if core_ortho_pct is not None else "",
        chart_title=f"Segment volume growth, {_cm_year_min}→{_cm_year_max}" if _cm_year_min else "Segment volume growth",
        chart_note="Bars capped at ±100% so outlier growth doesn't compress the rest.",
        chart_fn=_chart_case_mix,
        issues=cm_issues or [{"severity": "neutral", "tag": "No data", "body": "Case mix growth data not available."}],
    )

    # ── Case mix, continued — Other General Outpatient volume & projection ──
    # Same chart and forecast method as Case Mix Section 7's right column
    # (_seasonal_trend_forecast) — this bucket is where the respiratory/
    # gastric growth mentioned in the brainstorm actually lives, and it
    # wasn't represented anywhere on this page yet.
    _other_proj_years, _other_proj_vals, _other_df_full = [], [], None
    if df_cm_other_trend is not None:
        _other_df_full = df_cm_other_trend.sort_values("VISIT_YEAR")
        _other_current_year = int(_other_df_full["VISIT_YEAR"].max())
        if df_cm_other_monthly is not None:
            _other_proj_years = [_other_current_year, _other_current_year + 1, _other_current_year + 2]
            _forecast, _reliable = _seasonal_trend_forecast(
                df_cm_other_monthly, through=pd.Timestamp(f"{_other_proj_years[-1]}-12-01"),
            )
            for y in _other_proj_years:
                actual_part = _reliable[_reliable.index.year == y].sum()
                forecast_part = _forecast.loc[_forecast.index.year == y, "forecast"].sum()
                _other_proj_vals.append(actual_part + forecast_part)

    def _chart_other_opd():
        if _other_df_full is None:
            _empty_chart()
            return
        df_hist = _other_df_full[_other_df_full["VISIT_YEAR"] != int(_other_df_full["VISIT_YEAR"].max())]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_hist["VISIT_YEAR"].astype(str), y=df_hist["TOTAL_VISITS"], name="Actual",
            marker_color=PRIMARY, marker_cornerradius=3,
            text=df_hist["TOTAL_VISITS"].apply(lambda v: f"{v:,.0f}"), textposition="outside",
        ))
        if _other_proj_years:
            fig.add_trace(go.Bar(
                x=[str(y) for y in _other_proj_years], y=_other_proj_vals, name="Projected",
                marker_color="rgba(27,138,130,0.35)", marker_pattern_shape="/", marker_cornerradius=3,
                text=[f"{v:,.0f}" for v in _other_proj_vals], textposition="outside",
            ))
        fig.update_layout(**{**CHART_LAYOUT, "height": 230}, showlegend=bool(_other_proj_years))
        fig.update_xaxes(**AXIS_X)
        fig.update_yaxes(**AXIS_Y)
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    other_opd_signal = "Other General Outpatient projection data not available."
    if _other_df_full is not None and len(_other_df_full) >= 2:
        first_full, last_full = _other_df_full.iloc[0], _other_df_full[
            _other_df_full["VISIT_YEAR"] != int(_other_df_full["VISIT_YEAR"].max())
        ].iloc[-1]
        y0, v0 = int(first_full["VISIT_YEAR"]), int(first_full["TOTAL_VISITS"])
        y1, v1 = int(last_full["VISIT_YEAR"]), int(last_full["TOTAL_VISITS"])
        other_opd_signal = (
            f"Other General Outpatient (largely respiratory and gastric cases) grew from {v0:,} visits "
            f"in {y0} to {v1:,} in {y1} — a genuine, sustained general-medicine layer alongside the "
            "orthopaedic core, not classifier noise."
        )
        if _other_proj_years:
            watch_year, watch_val = _other_proj_years[1], _other_proj_vals[1]
            other_opd_signal += (
                f" Projected to reach ~{watch_val:,.0f} visits in {watch_year} if the trend and seasonal "
                "pattern hold."
            )

    st.markdown(
        '<div style="display:flex;align-items:baseline;justify-content:space-between;'
        'margin:20px 0 8px;font-family:Inter,sans-serif">'
        '<span style="font-size:13px;font-weight:600;color:' + TEXT_PRI + '">Case mix — Other General '
        'Outpatient</span></div>',
        unsafe_allow_html=True,
    )
    col_other_chart, col_other_signal = st.columns([1, 1.3])
    with col_other_chart:
        st.markdown(
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};border-radius:10px;'
            f'padding:14px 16px 8px;font-family:Inter,sans-serif">'
            f'<div style="font-size:12px;font-weight:600;color:{TEXT_SEC};margin-bottom:6px">'
            'Other General Outpatient — volume by year and projection</div>',
            unsafe_allow_html=True,
        )
        _chart_other_opd()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_other_signal:
        st.markdown(
            f'<div style="display:flex;flex-direction:column;justify-content:center;height:100%">'
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};border-top:3px solid {WARNING};'
            f'border-radius:10px;padding:10px 12px;font-family:Inter,sans-serif">'
            f'<span style="display:block;font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:{WARNING};margin-bottom:4px">Signal</span>'
            f'<span style="font-size:12px;color:{TEXT_SEC};line-height:1.5">{other_opd_signal}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    _kpi_spacer()

    # ── Section 2 — OPD → IPD conversion ────────────────────────────────────
    # Line-over-time for the top 5 segments by volume, not a single-period
    # bar snapshot — shows whether a segment's conversion rate is stable,
    # declining, or improving, which a bar chart of one period can't.
    df_opd_segment_by_year = _safe(Q.get_segment_conversion_by_year)
    _top_vol_segs = []
    if df_opd_segment is not None:
        _top_vol_segs = df_opd_segment.sort_values("TOTAL_VISITS", ascending=False).head(5)["PRIMARY_VISIT_SEGMENT"].tolist()
    _conv_trend = None
    if df_opd_segment_by_year is not None and _top_vol_segs:
        _conv_trend = df_opd_segment_by_year[df_opd_segment_by_year["PRIMARY_VISIT_SEGMENT"].isin(_top_vol_segs)]

    _conv_line_colors = [PRIMARY, "#C13868", NEUTRAL, WARNING, "#5C6478"]

    def _chart_conversion():
        if _conv_trend is None or _conv_trend.empty:
            _empty_chart()
            return
        fig = go.Figure()
        for i, seg in enumerate(_top_vol_segs):
            sub = _conv_trend[_conv_trend["PRIMARY_VISIT_SEGMENT"] == seg].sort_values("VISIT_YEAR")
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["VISIT_YEAR"], y=sub["CONVERSION_RATE_PCT"], mode="lines+markers",
                name=_SEG_SHORT.get(seg, seg),
                line=dict(color=_conv_line_colors[i % len(_conv_line_colors)], width=2.5),
                marker=dict(size=6, color=_conv_line_colors[i % len(_conv_line_colors)]),
            ))
        fig.update_layout(**{**CHART_LAYOUT, "height": 230, "showlegend": True})
        fig.update_xaxes(**AXIS_X, dtick=1)
        fig.update_yaxes(**AXIS_Y, ticksuffix="%", title="Conversion rate")
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    conv_issues = []
    spine_conv = None
    if df_opd_segment is not None:
        spine_row = df_opd_segment[df_opd_segment["PRIMARY_VISIT_SEGMENT"] == "Core Orthopedics: Spine and Back Pain Care"]
        if not spine_row.empty:
            spine_conv = float(spine_row.iloc[0]["CONVERSION_RATE_PCT"])
    if _conv_trend is not None and len(_top_vol_segs) >= 1:
        seg0 = _top_vol_segs[0]
        sub0 = _conv_trend[_conv_trend["PRIMARY_VISIT_SEGMENT"] == seg0].sort_values("VISIT_YEAR")
        if len(sub0) >= 2:
            y0, y1 = sub0.iloc[0], sub0.iloc[-1]
            conv_issues.append({
                "severity": "warning", "tag": "Trend",
                "body": f"{_SEG_SHORT.get(seg0, seg0)}, the highest-volume segment, moved from "
                        f"{float(y0['CONVERSION_RATE_PCT']):.1f}% in {int(y0['VISIT_YEAR'])} to "
                        f"{float(y1['CONVERSION_RATE_PCT']):.1f}% in {int(y1['VISIT_YEAR'])}.",
            })
    # Hospital-wide conversion trend, computed by rolling every segment's
    # admissions/visits up per year — ties the declining-conversion pattern
    # directly to Spine's growing, largely non-surgical OPD volume, instead
    # of leaving "conversion is falling" and "Spine walk-ins aren't
    # surgical" as two separate, unconnected findings.
    if df_opd_segment_by_year is not None:
        yearly_all = df_opd_segment_by_year.groupby("VISIT_YEAR").agg(
            TOTAL_VISITS=("TOTAL_VISITS", "sum"), INPATIENT_ADMISSIONS=("INPATIENT_ADMISSIONS", "sum"),
        ).sort_index()
        if len(yearly_all) >= 2:
            yearly_all["CONV_PCT"] = 100.0 * yearly_all["INPATIENT_ADMISSIONS"] / yearly_all["TOTAL_VISITS"]
            y0, y1 = yearly_all.iloc[0], yearly_all.iloc[-1]
            yr0, yr1 = int(yearly_all.index[0]), int(yearly_all.index[-1])
            visits_up = y1["TOTAL_VISITS"] > y0["TOTAL_VISITS"]
            conv_down = y1["CONV_PCT"] < y0["CONV_PCT"]
            spine_row_yr = df_opd_segment_by_year[
                df_opd_segment_by_year["PRIMARY_VISIT_SEGMENT"] == "Core Orthopedics: Spine and Back Pain Care"
            ]
            spine_growth_pts = []
            if len(spine_row_yr) >= 2:
                sp = spine_row_yr.sort_values("VISIT_YEAR")
                sp0, sp1 = sp.iloc[0], sp.iloc[-1]
                if sp1["TOTAL_VISITS"] > sp0["TOTAL_VISITS"]:
                    spine_growth_pts.append(
                        f"Spine visits grew {int(sp0['TOTAL_VISITS']):,} → {int(sp1['TOTAL_VISITS']):,} "
                        "over the same period — most of that added volume is non-surgical."
                    )
            if visits_up and conv_down:
                pts = [
                    f"Visits grew {int(y0['TOTAL_VISITS']):,} → {int(y1['TOTAL_VISITS']):,} ({yr0}–{yr1}) "
                    "while admissions didn't keep pace.",
                    f"Blended conversion fell {y0['CONV_PCT']:.1f}% → {y1['CONV_PCT']:.1f}%.",
                ] + spine_growth_pts
                if spine_conv is not None:
                    pts.append(
                        f"Spine converts at only {spine_conv:.1f}% — most walk-ins are chronic back pain, "
                        "not surgical candidates."
                    )
                body_html = "<ul style='margin:0;padding-left:16px'>" + "".join(
                    f"<li style='margin-bottom:3px'>{p}</li>" for p in pts
                ) + "</ul>"
                conv_issues.append({"severity": "danger", "tag": "Critical", "body": body_html})
    if df_opd_workload is not None and len(df_opd_workload) >= 2:
        wl = df_opd_workload.sort_values("AVG_WORKLOAD")
        lo, hi = wl.iloc[0], wl.iloc[-1]
        conv_issues.append({
            "severity": "warning", "tag": "Confirmed driver",
            "body": f"Conversion drops from {float(lo['AVG_CONVERSION_RATE_PCT']):.1f}% at "
                    f"~{int(lo['AVG_WORKLOAD'])} visits/month to {float(hi['AVG_CONVERSION_RATE_PCT']):.1f}% "
                    f"at ~{int(hi['AVG_WORKLOAD'])}/month — a staffing/caseload effect, not a gradual "
                    "quality decline.",
        })
    if df_opd_comorbid is not None:
        surg = df_opd_comorbid[df_opd_comorbid["SEGMENT_TYPE"] == "Surgical"]
        c_true = surg[surg["HAS_CHRONIC_CONDITION"].isin([True, 1])]
        c_false = surg[surg["HAS_CHRONIC_CONDITION"].isin([False, 0])]
        if not c_true.empty and not c_false.empty:
            rate_true = float(c_true.iloc[0]["CONVERSION_RATE_PCT"])
            rate_false = float(c_false.iloc[0]["CONVERSION_RATE_PCT"])
            mult = round(rate_true / rate_false, 1) if rate_false else None
            conv_issues.append({
                "severity": "warning", "tag": "Confirmed driver",
                "body": f"Comorbid surgical patients convert at {rate_true:.1f}% vs {rate_false:.1f}% for "
                        f"non-comorbid ({mult}× effect)" + (f" — not currently flagged at triage." if mult else "."),
            })

    overview_tab_section(
        tab_name="OPD → IPD conversion",
        tab_tag=f"Spine converts at {spine_conv:.1f}% — walk-ins are pain management, not surgical" if spine_conv is not None else "",
        chart_title="Conversion rate over time — top 5 segments by volume",
        chart_fn=_chart_conversion,
        issues=conv_issues or [{"severity": "neutral", "tag": "No data", "body": "Conversion data not available."}],
    )

    # ── Section 3 — Flow and retention ──────────────────────────────────────
    # Same chart as Flow & Retention Section 1 — real data (get_fr_retention_trend),
    # same three-line Active/Lapsing/LTFU treatment with the exact status-system
    # hex values, not the fabricated two-series area-fill this used to be.
    def _chart_flow_retention():
        if df_fr_retention_trend is None:
            st.markdown(
                f'<div style="padding:16px;text-align:center;color:{NEUTRAL};'
                f'font-size:11px;font-style:italic">No data available</div>',
                unsafe_allow_html=True,
            )
            return
        wide = df_fr_retention_trend.pivot_table(
            index="AS_OF_MONTH", columns="STATUS", values="PCT_OF_CLASSIFIABLE_PATIENTS", aggfunc="first"
        )
        wide = wide.sort_index()
        fig = go.Figure()
        for status, color in [("Active", "#639922"), ("Lapsing", "#EF9F27"), ("LTFU", "#E24B4A")]:
            if status in wide.columns:
                fig.add_trace(go.Scatter(
                    x=wide.index, y=wide[status], mode="lines+markers", name=status,
                    line=dict(color=color, width=2.5), marker=dict(size=6, color=color),
                ))
        fig.update_layout(**CHART_LAYOUT, height=230)
        fig.update_xaxes(**AXIS_X)
        fig.update_yaxes(**AXIS_Y, ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    fr_issues = []
    if df_fr_retention_trend is not None:
        wide_t = df_fr_retention_trend.pivot_table(
            index="AS_OF_MONTH", columns="STATUS", values="PCT_OF_CLASSIFIABLE_PATIENTS", aggfunc="first"
        ).sort_index()
        if "LTFU" in wide_t.columns and len(wide_t) > 1:
            ltfu_first, ltfu_last = float(wide_t["LTFU"].iloc[0]), float(wide_t["LTFU"].iloc[-1])
            fr_issues.append({
                "severity": "danger", "tag": "Critical",
                "body": f"LTFU moved from {ltfu_first:.1f}% to {ltfu_last:.1f}% over the trailing "
                        f"{len(wide_t)} months — a sustained rise, not a snapshot artifact.",
            })
    if df_fr_gs is not None:
        gs_row = df_fr_gs.iloc[0]
        gs_scheduled = float(gs_row["PCT_SCHEDULED"])
        gs_returned = float(gs_row["PCT_UNSCHEDULED_RETURNED_ANYWAY"] or 0)
        fr_issues.append({
            "severity": "warning", "tag": "Scheduling gap",
            "body": f"Only {gs_scheduled:.1f}% of General Surgery visits get a scheduled follow-up, yet "
                    f"{gs_returned:.1f}% of unscheduled patients return anyway — patients are largely "
                    "self-initiating continuity, and the scheduling data itself is incomplete, so this is "
                    "as much a measurement gap as a process one.",
        })
    if df_fr_visit_seg is not None:
        v1 = df_fr_visit_seg[df_fr_visit_seg["LTFU_AT_VISIT_NUMBER"] == "1"]
        if not v1.empty:
            worst = v1.sort_values("PCT_WITHIN_SEGMENT", ascending=False).iloc[0]
            fr_issues.append({
                "severity": "danger", "tag": "Factor",
                "body": f"{worst['SEGMENT']} loses {float(worst['PCT_WITHIN_SEGMENT']):.0f}% of its own "
                        "LTFU patients right after visit 1 — LTFU here is front-loaded, not a slow drift, "
                        "and much of it has no recorded explanation (no follow-up, no later visit anywhere).",
            })
    if df_fr_outreach is not None:
        fr_issues.append({
            "severity": "success", "tag": "Actionable",
            "body": f"{len(df_fr_outreach)} verified post-spine-surgery patients identified for direct "
                    "outreach — the clearest, smallest, most tractable list in this dataset.",
        })

    overview_tab_section(
        tab_name="Flow and retention",
        tab_tag=f"{ltfu_pct:.1f}% LTFU, rising every month" if ltfu_pct is not None else "",
        chart_title="Active vs. LTFU share — 12-month trend",
        chart_fn=_chart_flow_retention,
        issues=fr_issues or [{"severity": "neutral", "tag": "No data", "body": "Retention data not available."}],
    )

    # ── Section 4 — Clinical activity ───────────────────────────────────────
    _ca_ward = None
    if df_ca_los_ward is not None and df_ca_ward_rate is not None:
        _ca_ward = df_ca_los_ward.merge(df_ca_ward_rate[["WARD", "READMISSION_RATE"]], on="WARD", how="inner")

    def _chart_clinical_activity():
        if _ca_ward is None or _ca_ward.empty:
            _empty_chart()
            return
        d = _ca_ward.sort_values("AVG_LOS")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["WARD"], y=d["AVG_LOS"], name="Avg LOS (days)", marker_color=WARNING, yaxis="y"))
        fig.add_trace(go.Bar(x=d["WARD"], y=d["READMISSION_RATE"], name="Readmission %", marker_color=DANGER, yaxis="y2"))
        fig.update_layout(
            **{**CHART_LAYOUT, "showlegend": True,
               # Legend pinned above the plot, ward labels rotated below it —
               # with no explicit position they landed on top of each other.
               "legend": dict(orientation="h", y=1.18, x=0.5, xanchor="center", font=dict(size=10)),
               "margin": {**CHART_LAYOUT.get("margin", {}), "t": 40, "b": 100}},
            height=310,
            barmode="group",
            yaxis=dict(title="LOS (days)", **{k: v for k, v in AXIS_Y.items() if k != "title_font"}),
            yaxis2=dict(title="Readmission %", overlaying="y", side="right",
                        showgrid=False, tickfont=AXIS_Y["tickfont"]),
        )
        fig.update_xaxes(**AXIS_X, tickangle=-30, automargin=True)
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    ca_issues = []
    if ssi_ratio is not None:
        ca_issues.append({
            "severity": "danger", "tag": "Critical",
            "body": f"{worst_ssi_cat} SSI runs {ssi_ratio}× its benchmark ({worst_ssi:.1f}% vs "
                    f"{worst_ssi_bench:.1f}% ceiling) — masked by strong performance elsewhere in the "
                    "blended rate.",
        })
    if _ca_ward is not None and len(_ca_ward) >= 3:
        d = _ca_ward.sort_values("AVG_LOS")
        shortest, longest = d.iloc[0], d.iloc[-1]
        corr = d["AVG_LOS"].astype(float).corr(d["READMISSION_RATE"].astype(float))
        if pd.notna(corr) and corr < -0.2:
            ca_issues.append({
                "severity": "danger", "tag": "Critical",
                "body": f"{shortest['WARD']} has the shortest average stay ({float(shortest['AVG_LOS']):.1f}d) "
                        f"and a higher readmission rate ({float(shortest['READMISSION_RATE']):.1f}%) than "
                        f"{longest['WARD']}'s longer stay ({float(longest['AVG_LOS']):.1f}d, "
                        f"{float(longest['READMISSION_RATE']):.1f}%) — consistent with early discharge "
                        "before care is complete.",
            })
    if blind_spot:
        ca_issues.append({
            "severity": "warning", "tag": "Monitor",
            "body": f"{blind_spot} patients return in the untracked 31–90 day window — comparable in size "
                    "to tracked readmissions and currently invisible to the standard KPI.",
        })

    overview_tab_section(
        tab_name="Clinical activity",
        tab_tag=f"{worst_ssi_cat} SSI {ssi_ratio}× benchmark" if ssi_ratio is not None else "",
        chart_title="Length of stay vs. readmission rate, by ward",
        chart_note="Shorter stay + higher readmission may mean patients are discharged before care is complete",
        chart_fn=_chart_clinical_activity,
        issues=ca_issues or [{"severity": "neutral", "tag": "No data", "body": "Clinical activity data not available."}],
    )

    # ── Section 5 — Disease burden — Orthopedics ────────────────────────────
    def _chart_orth_vte():
        if df_orth_vte is None:
            _empty_chart()
            return
        d = df_orth_vte.sort_values("PCT_PROPHYLAXIS_COMPLIANCE")
        colors = [DANGER if p < 70 else (WARNING if p < 85 else NEUTRAL) for p in d["PCT_PROPHYLAXIS_COMPLIANCE"]]
        fig = go.Figure(go.Bar(x=d["PCT_PROPHYLAXIS_COMPLIANCE"], y=d["MAJOR_PROCEDURE_CATEGORY"],
                                orientation="h", marker_color=colors))
        fig.add_vline(x=90, line_dash="dash", line_color="#8A93A6")
        fig.update_layout(**CHART_LAYOUT, height=230)
        fig.update_xaxes(**AXIS_X, range=[0, 100], title="Compliance %")
        fig.update_yaxes(**{**AXIS_Y, "showgrid": False})
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    orth_issues = []
    spine_tag = ""
    if df_orth_spine is not None:
        years = sorted(df_orth_spine["YEAR"].unique())
        wide_share = df_orth_spine.pivot_table(index="YEAR", columns="SPINE_CASE_TYPE",
                                                values="PCT_OF_YEAR_SPINE_VOLUME", fill_value=0)
        gp = "General pain / likely conservative management"
        if years and gp in wide_share.columns:
            g0, g1 = float(wide_share.loc[years[0]][gp]), float(wide_share.loc[years[-1]][gp])
            orth_issues.append({
                "severity": "danger", "tag": "Critical",
                "body": f"Spine and Back Pain Care has become dominated by conservative pain management — "
                        f"general pain grew from {g0:.1f}% to {g1:.1f}% of spine volume ({years[0]}–{years[-1]}).",
            })
            spine_tag = f"Spine general-pain share: {g0:.1f}% → {g1:.1f}%"
    if df_orth_vte is not None:
        worst = df_orth_vte.sort_values("PCT_PROPHYLAXIS_COMPLIANCE").iloc[0]
        orth_issues.append({
            "severity": "danger", "tag": "Critical",
            "body": f"{worst['MAJOR_PROCEDURE_CATEGORY']} VTE prophylaxis compliance is "
                    f"{float(worst['PCT_PROPHYLAXIS_COMPLIANCE']):.1f}% — the lowest of any tracked "
                    "procedure, against a near-universal clinical standard.",
        })
    if df_orth_compl is not None:
        top = df_orth_compl.sort_values("DISTINCT_VISITS", ascending=False).iloc[0]
        orth_issues.append({
            "severity": "warning", "tag": "Monitor",
            "body": f"{top['COMPLICATION_TYPE']} ({int(top['DISTINCT_VISITS'])} cases) is the dominant "
                    "orthopaedic complication, with no dedicated investigation yet — unlike SSI.",
        })

    overview_tab_section(
        tab_name="Disease burden — Orthopedics",
        tab_tag=spine_tag,
        chart_title="VTE compliance vs. 90% standard",
        chart_fn=_chart_orth_vte,
        issues=orth_issues or [{"severity": "neutral", "tag": "No data", "body": "Orthopedics data not available."}],
    )

    # ── Section 6 — Disease burden — Maternal health ────────────────────────
    _mat_bucket_order = ["1 visit", "2 visits", "3 visits", "4+ visits"]
    _mat_bucket_colors = {"1 visit": DANGER, "2 visits": WARNING, "3 visits": NEUTRAL, "4+ visits": SUCCESS}

    def _chart_maternal():
        if df_mat_anc is None:
            _empty_chart()
            return
        d = df_mat_anc.copy()
        d["BUCKET"] = d["VISIT_COUNT_BUCKET"].map(
            lambda x: next((b for b in _mat_bucket_order if x.startswith(b.split(" ")[0])), x)
        )
        d = d.set_index("BUCKET").reindex(_mat_bucket_order)
        colors = [_mat_bucket_colors[b] for b in _mat_bucket_order]
        fig = go.Figure(go.Bar(x=_mat_bucket_order, y=d["PCT_OF_ANC_PATIENTS"], marker_color=colors))
        fig.update_layout(**CHART_LAYOUT, height=230)
        fig.update_xaxes(**AXIS_X)
        fig.update_yaxes(**AXIS_Y, ticksuffix="%", title="% of patients")
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    mat_issues = []
    if anc_4plus_pct is not None:
        mat_issues.append({
            "severity": "danger", "tag": "Critical",
            "body": f"Only {anc_4plus_pct:.1f}% of pregnant patients reach the 4-visit quality threshold — "
                    "confirmed robust across independent checks, not a data artifact.",
        })
    if df_mat_qual_b is not None:
        zero = df_mat_qual_b[df_mat_qual_b["ANC_QUALITY_SCORE_OUT_OF_5"] == 0]
        if not zero.empty:
            zero_pct = float(zero.iloc[0]["PCT_OF_ANC_VISITS"])
            mat_issues.append({
                "severity": "danger", "tag": "Critical",
                "body": f"{zero_pct:.1f}% of ANC visits have zero quality indicators recorded — no BP, "
                        "iron, blood, or ultrasound.",
            })
    if df_mat_workup is not None:
        w = df_mat_workup.iloc[0]
        total_haem = int(w.get("TOTAL_HAEMORRHAGE_VISITS", 0) or 0)
        hgb_n = int(w.get("WITH_HEMOGLOBIN_CHECK", 0) or 0)
        if total_haem:
            mat_issues.append({
                "severity": "danger", "tag": "Critical",
                "body": f"Haemoglobin — the most basic test for haemorrhage — is on record for only "
                        f"{hgb_n} of {total_haem} haemorrhage visits.",
            })

    overview_tab_section(
        tab_name="Disease burden — Maternal health",
        tab_tag=f"{anc_single_pct:.1f}% single-visit rate" if anc_single_pct is not None else "",
        chart_title="ANC visits per patient",
        chart_fn=_chart_maternal,
        issues=mat_issues or [{"severity": "neutral", "tag": "No data", "body": "Maternal health data not available."}],
    )

    # ── Maternal health, continued — ANC quality indicators ──────────────────
    # Same coverage-rate chart as Maternal Section 4 — continuity (visit
    # count) and quality (what's actually done at each visit) are two
    # different failure modes and neither was represented alone before.
    _anc_indicators = []
    if df_mat_qual_a is not None:
        row_a = df_mat_qual_a.iloc[0]
        _anc_indicators = [
            ("Obstetric ultrasound", float(row_a["PCT_ULTRASOUND_FETAL_PROXY"]), PRIMARY),
            ("Urine sample", float(row_a["PCT_URINE_SAMPLE"]), WARNING),
            ("Blood sample", float(row_a["PCT_BLOOD_SAMPLE"]), WARNING),
            ("Blood pressure taken", float(row_a["PCT_BP_TAKEN"]), DANGER),
            ("Iron supplementation", float(row_a["PCT_IRON_GIVEN"]), DANGER),
        ]

    def _chart_anc_quality():
        if not _anc_indicators:
            _empty_chart()
            return
        d = sorted(_anc_indicators, key=lambda t: t[1])
        fig = go.Figure(go.Bar(
            y=[n for n, _, _ in d], x=[v for _, v, _ in d], orientation="h",
            marker_color=[c for _, _, c in d],
            text=[f"{v:.1f}%" for _, v, _ in d], textposition="outside",
        ))
        fig.update_layout(**{**CHART_LAYOUT, "height": 230, "margin": {**CHART_LAYOUT.get("margin", {}), "l": 10, "r": 40}},
                           showlegend=False)
        fig.update_xaxes(**AXIS_X, ticksuffix="%", range=[0, 100])
        fig.update_yaxes(**{**AXIS_Y, "showgrid": False}, automargin=True)
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)

    anc_quality_items = []
    if _anc_indicators:
        worst_name, worst_pct, _ = min(_anc_indicators, key=lambda t: t[1])
        anc_quality_items.append(
            f"{worst_name} is the lowest-coverage indicator at {worst_pct:.1f}% — a basic, low-cost "
            "intervention that should be near-universal."
        )
        if df_mat_qual_b is not None:
            zero = df_mat_qual_b[df_mat_qual_b["ANC_QUALITY_SCORE_OUT_OF_5"] == 0]
            five = df_mat_qual_b[df_mat_qual_b["ANC_QUALITY_SCORE_OUT_OF_5"] == 5]
            zero_pct = float(zero.iloc[0]["PCT_OF_ANC_VISITS"]) if not zero.empty else None
            five_n = int(five.iloc[0]["TOTAL_VISITS"]) if not five.empty else 0
            if zero_pct is not None:
                anc_quality_items.append(
                    f"{zero_pct:.1f}% of ANC visits record none of these 5 indicators at all, and only "
                    f"{five_n} visits achieve all 5."
                )
                anc_quality_items.append(
                    "Continuity (how often patients return) and quality (what happens when they do) are "
                    "two separate gaps, not one."
                )
    anc_quality_signal = (
        "<ul style='margin:0;padding-left:16px'>" +
        "".join(f"<li style='margin-bottom:4px'>{item}</li>" for item in anc_quality_items) +
        "</ul>"
        if anc_quality_items else "ANC quality-indicator data not available."
    )

    st.markdown(
        '<div style="display:flex;align-items:baseline;justify-content:space-between;'
        'margin:20px 0 8px;font-family:Inter,sans-serif">'
        '<span style="font-size:13px;font-weight:600;color:' + TEXT_PRI + '">Disease burden — Maternal '
        'health, ANC quality</span></div>',
        unsafe_allow_html=True,
    )
    col_anc_chart, col_anc_signal = st.columns([1, 1.3])
    with col_anc_chart:
        st.markdown(
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};border-radius:10px;'
            f'padding:14px 16px 8px;font-family:Inter,sans-serif">'
            f'<div style="font-size:12px;font-weight:600;color:{TEXT_SEC};margin-bottom:6px">'
            'Coverage rate per quality indicator — ANC visits</div>',
            unsafe_allow_html=True,
        )
        _chart_anc_quality()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_anc_signal:
        st.markdown(
            f'<div style="display:flex;flex-direction:column;justify-content:center;height:100%">'
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};border-top:3px solid {DANGER};'
            f'border-radius:10px;padding:10px 12px;font-family:Inter,sans-serif">'
            f'<span style="display:block;font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:{DANGER};margin-bottom:4px">Signal</span>'
            f'<span style="font-size:12px;color:{TEXT_SEC};line-height:1.5">{anc_quality_signal}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    _kpi_spacer()


# ─────────────────────────────────────────────────────────────────────────────
# OPD → IPD CONVERSION TAB  (Template B — full 10-section analysis)
# ─────────────────────────────────────────────────────────────────────────────
elif active == "OPD → IPD conversion":
    page_header(
        "Outpatient to Admission Pathway",
        subtitle="Conversion performance from outpatient visits to admissions",
    )

    with st.spinner("Loading data…"):
        df_kpis        = Q.get_headline_kpis()
        df_trend       = Q.get_monthly_trend()
        df_segments    = Q.get_segment_conversion()
        df_ortho       = Q.get_ortho_burden_breakdown()
        df_spine_vol   = Q.get_spine_volume_trend()
        df_non_ortho   = Q.get_non_ortho_case_mix()
        df_workload    = Q.get_workload_vs_conversion()
        df_staffing    = Q.get_staffing_trend()
        df_comorbidity = Q.get_comorbidity_conversion()
        df_escalation  = Q.get_escalation_trend()
        # get_escalation_investigation_coverage() (blended OPD/IPD join, no
        # before/after split) is retained in queries.py for documentation of
        # the join fix, but no longer feeds this section directly — see
        # OPD_IPD_Escalation_Section_Rebuild.md §4.
        df_ortho_conversion_by_year  = Q.get_ortho_general_conversion_by_year()
        df_escalation_timing         = Q.get_escalation_investigation_timing()

    V.render_s1_kpis(df_kpis)
    V.render_s2_trend(df_trend)
    V.render_s3_treemap(df_segments)
    V.render_s4_segment_bar(df_segments)
    V.render_s5_ortho_deep_dive(df_ortho, df_spine_vol)
    V.render_s6_non_ortho(df_non_ortho)
    V.render_s7_factors(df_workload, df_staffing, df_comorbidity)
    V.render_s8_escalation(df_escalation, df_ortho_conversion_by_year, df_escalation_timing)
    V.render_s10_recommendations(
        df_segments, df_ortho, df_spine_vol, df_staffing, df_comorbidity, df_escalation_timing,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL ACTIVITY TAB  (Template B — readmissions, LOS, SSI)
# ─────────────────────────────────────────────────────────────────────────────
elif active == "Clinical activity":
    with st.sidebar:
        st.markdown(
            f"""
            <div style="background:#FFFFFF;border:0.5px solid {BORDER};
                        border-radius:8px;padding:10px 12px;margin:14px 0">
              <div style="color:{TEXT_PRI};font-weight:500;font-size:12px;margin-bottom:6px">
                Readmission cause definitions
              </div>
              <div style="color:#5C6478;font-size:11px;line-height:1.6">
                <strong style="color:{TEXT_PRI}">Expected:</strong> planned staged care, scheduled
                follow-up procedures.<br/>
                <strong style="color:{TEXT_PRI}">Potentially preventable:</strong> wound complications,
                uncontrolled pain, early surgical infections.<br/>
                <strong style="color:{TEXT_PRI}">Unclear / other:</strong> insufficient documentation
                to classify.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    import sph.clinicals.clinical_activity_module.ca_views as CAV
    CAV.render_clinical_activity_tab()


# ─────────────────────────────────────────────────────────────────────────────
# CASE MIX TAB  (segment composition, growth, comorbidity)
# ─────────────────────────────────────────────────────────────────────────────
elif active == "Case mix":
    page_header(
        "Service Mix & Growth",
        subtitle="Caseload composition, trends, and emerging shifts",
    )

    import sph.clinicals.case_mix_module.cm_views as CMV
    CMV.render_tab()


# ─────────────────────────────────────────────────────────────────────────────
# FLOW AND RETENTION TAB  (continuity of care, scheduled follow-up, LTFU)
# ─────────────────────────────────────────────────────────────────────────────
elif active == "Flow and retention":
    import sph.clinicals.flow_retention_module.fr_views as FRV
    FRV.render_tab()


# ─────────────────────────────────────────────────────────────────────────────
# DATA QUALITY TAB
# ─────────────────────────────────────────────────────────────────────────────
elif active == "Data quality":
    import sph.clinicals.data_quality_module.dq_views as DQV
    DQV.render_tab()


# ─────────────────────────────────────────────────────────────────────────────
# DISEASE BURDEN TAB  (sub-tab bar: Orthopedics / Maternal health)
# ─────────────────────────────────────────────────────────────────────────────
elif active == "Disease burden":
    def _render_subtab_bar(active_sub: str) -> None:
        items = ""
        for t in _DISEASE_BURDEN_SUBTABS:
            is_active = t == active_sub
            color  = PRIMARY if is_active else "#8A93A6"
            border = f"2px solid {PRIMARY}" if is_active else "2px solid transparent"
            weight = "600" if is_active else "500"
            href = f"?tab={quote(active)}&sub={quote(t)}"
            items += (
                f'<a href="{href}" target="_self" style="text-decoration:none;font-size:14px;'
                f'font-weight:{weight};padding:8px 14px;color:{color};border-bottom:{border};'
                f'cursor:pointer">{t}</a>'
            )
        st.markdown(
            f'<div style="display:flex;gap:0;border-bottom:0.5px solid #E4E7ED;'
            f'background:#FFFFFF;padding:0 16px;margin:-13px -16px 13px">'
            f'{items}</div>',
            unsafe_allow_html=True,
        )

    sub = st.session_state["disease_burden_sub_tab"]

    if sub == "Orthopedics":
        page_header(
            "Orthopedic Case Profile",
            subtitle="Patient profiles, procedures, and alignment with care standards",
        )
        _render_subtab_bar(sub)

        import sph.clinicals.disease_burden_module.orthopedics.orth_views as ORV
        ORV.render_tab()
    elif sub == "Maternal health":
        page_header(
            "Maternal & Women's Health",
            subtitle="Service utilization, care outcomes, and maternal health trends",
        )
        _render_subtab_bar(sub)

        import sph.clinicals.disease_burden_module.maternal.mat_views as MAV
        MAV.render_tab()
    else:
        page_header(sub, subtitle="This sub-tab is not yet built.")
        _render_subtab_bar(sub)
        st.markdown(
            f"""
            <div style="margin-top:40px;padding:40px;text-align:center;
                        background:#FFFFFF;border:1px solid #E4E7ED;border-radius:10px">
              <div style="font-size:32px;margin-bottom:12px">🚧</div>
              <div style="font-size:15px;font-weight:600;color:#141F3D;margin-bottom:6px">
                {sub}
              </div>
              <div style="font-size:13px;color:#8A93A6">
                This sub-tab is on the build roadmap.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PLACEHOLDER TABS — not yet built
# ─────────────────────────────────────────────────────────────────────────────
else:
    page_header(
        active,
        subtitle="This tab is not yet built.",
    )
    st.markdown(
        f"""
        <div style="margin-top:40px;padding:40px;text-align:center;
                    background:#FFFFFF;border:1px solid #E4E7ED;border-radius:10px">
          <div style="font-size:32px;margin-bottom:12px">🚧</div>
          <div style="font-size:15px;font-weight:600;color:#141F3D;margin-bottom:6px">
            {active}
          </div>
          <div style="font-size:13px;color:#8A93A6">
            This tab is on the build roadmap. Check back once the next workstream is complete.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )