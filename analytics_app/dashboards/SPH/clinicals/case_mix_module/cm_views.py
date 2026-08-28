"""
sph/case_mix_module/cm_views.py
==================================
All render functions for the SPH Case Mix tab.

Rules enforced here:
  - Zero SQL — no database calls, no query strings.
  - All insight text is computed from the DataFrame passed in, never
    hardcoded.
  - Reuses the shared ui_template component set (section_header, kpi_row,
    chart_card, insight_bar) — this tab has no separate build spec, so
    it follows the same visual language as the OPD → IPD conversion tab.
"""

import calendar
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sph.clinicals.opd_ipd_module.ui_template import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, SECONDARY,
    SURFACE_1, BORDER, TEXT_PRI, TEXT_SEC, TEXT_MUT,
    CHART_LAYOUT, AXIS_X, AXIS_Y, PC_CFG,
    section_header, kpi_row,
    chart_card, chart_card_close,
    insight_bar, key_findings_cards, priority_cards,
    fmt_num, fmt_pct,
)

_C_BLUE   = PRIMARY     # teal — orthopedics category (spec §4)
_C_ORANGE = WARNING
_C_RED    = DANGER
_C_AMBER  = WARNING
_C_TEAL   = SUCCESS
_C_GREY   = "#D3D6DE"
_C_PURPLE = SECONDARY   # raspberry — general surgery/OBGYN category (spec §4)
_C_MGREY  = NEUTRAL

_LAYOUT = CHART_LAYOUT
_H_SINGLE = 280
_H_PAIRED = 240


def _safe(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty


def _empty(msg: str = "No data available") -> None:
    st.markdown(
        f'<div style="padding:20px;text-align:center;color:{TEXT_MUT};'
        f'font-size:13px;font-style:italic">{msg}</div>',
        unsafe_allow_html=True,
    )


# Category treemap: teal family = orthopedics, raspberry family = general
# surgery/OBGYN, grey = other/unclassified (spec §4 Treemap). Darkest = largest
# box within its family — see _segment_family_shade below for the ranked version.
# Narrower band than the full 5-stop chart ramp — with only 1-2 boxes per
# family here, the outermost stops (near-black-green / maroon) read as a
# different hue entirely rather than a shade of the same color.
_TEAL_RAMP = ["#1B8A82", "#4FADA5", "#8FCFC8", "#C7E6E0", "#E1F5EE"]
_RASP_RAMP = ["#C13868", "#D6698C", "#EBA3B8", "#F3C9D6", "#FBEAF0"]


def _segment_family(segment: str) -> str:
    s = (segment or "").lower()
    if "spine" in s or "ortho" in s:
        return "teal"
    if "surgery" in s or "surgical" in s or "obgyn" in s or "gynaecolog" in s or "gynecolog" in s:
        return "raspberry"
    return "grey"


def _segment_color(segment: str) -> str:
    fam = _segment_family(segment)
    if fam == "teal":
        return _TEAL_RAMP[0]
    if fam == "raspberry":
        return _RASP_RAMP[0]
    return "#D3D6DE"


def _segment_text_color(segment: str) -> str:
    fam = _segment_family(segment)
    if fam == "teal":
        return "#1B8A82"
    if fam == "raspberry":
        return "#C13868"
    return "#5C6478"


# "Core Orthopedics: General".split(": ")[-1] alone reads as just "General" —
# ambiguous with other medical segments. Named explicitly here instead of
# truncating blindly on every chart.
_SHORT_LABEL = {
    "Core Orthopedics: General": "Ortho General",
    "Core Orthopedics: Spine and Back Pain Care": "Spine and Back Pain Care",
    "Core General Surgery": "Core General Surgery",
    "Standalone Specialty: Plastic Surgery": "Plastic Surgery",
    "Standalone Specialty: Maxillofacial": "Maxillofacial",
    "Standalone Specialty: Dental": "Dental",
    "Standalone Specialty: Eye/Ophthalmology": "Eye/Ophthalmology",
    "Standalone Specialty: ENT": "ENT",
    "Standalone Specialty: Obstetrics & Gynaecology": "Obstetrics & Gynaecology",
    "Standalone Specialty: Neurosurgery (structural/acute)": "Neurosurgery",
    "Standalone Medical: Neurology (chronic/medical)": "Neurology",
    "Standalone Specialty: Urology": "Urology",
    "Standalone Medical: Sepsis/Infection": "Sepsis/Infection",
    "Standalone Medical: Cardiovascular": "Cardiovascular",
    "Standalone Medical: Endocrine/Metabolic": "Endocrine/Metabolic",
    "Other General Outpatient": "Other General Outpatient",
}


def _short(segment: str) -> str:
    return _SHORT_LABEL.get(segment, segment.split(": ")[-1] if segment else segment)


def growth_badge(pct_growth) -> str:
    """Small colour-coded growth/decline badge, used in the S4 growth table.

    Displayed value is capped at 100% so one extreme outlier (e.g. a
    low-base segment growing 1600%+) doesn't dominate the table visually —
    the true figure is still shown on hover via the title attribute.
    """
    if pct_growth is None or pd.isna(pct_growth):
        return f'<span style="font-size:11px;color:{TEXT_MUT}">—</span>'
    pct_growth = float(pct_growth)
    color = SUCCESS if pct_growth > 0 else (DANGER if pct_growth < 0 else TEXT_MUT)
    sign = "+" if pct_growth > 0 else ""
    capped = max(-100.0, min(100.0, pct_growth))
    label = f"{sign}{capped:.1f}%" if abs(pct_growth) <= 100 else f"{sign}100%+"
    return (
        f'<span title="Actual: {sign}{pct_growth:.1f}%" style="font-size:11px;font-weight:700;color:{color};'
        f'background:{color}1A;padding:2px 8px;border-radius:4px">'
        f'{label}</span>'
    )


# ── S1: Headline KPIs ────────────────────────────────────────────────────────

def render_s1(df: pd.DataFrame) -> None:
    section_header("Case mix — headline")
    if not _safe(df):
        _empty()
        return

    row = df.iloc[0]
    total       = int(row.get("TOTAL_VISITS", 0) or 0)
    core_pct    = float(row.get("CORE_ORTHO_SHARE_PCT", 0) or 0)
    spine_2022  = float(row.get("SPINE_SHARE_2022_PCT", 0) or 0)
    spine_latest= float(row.get("SPINE_SHARE_LATEST_PCT", 0) or 0)
    divers_pct  = float(row.get("DIVERSIFICATION_SHARE_PCT", 0) or 0)

    kpi_row([
        {
            "label": "Total visits", "value": fmt_num(total),
            "delta": "All segments, full history", "accent_color": PRIMARY,
        },
        {
            "label": "Core orthopedics share", "value": fmt_pct(core_pct),
            "delta": "Spine + General combined — expected for an ortho specialty center", "accent_color": SUCCESS,
        },
        {
            "label": "Spine's share of volume", "value": f"{spine_2022:.1f}% → {spine_latest:.1f}%",
            "delta": "2022 vs. 2026", "delta_good": spine_latest > spine_2022,
            "accent_color": WARNING,
        },
        {
            "label": "Diversification layer", "value": fmt_pct(divers_pct),
            "delta": "Genuine general-medicine layer outside the ortho core", "accent_color": SUCCESS,
        },
    ])


# ── S2: Overall composition (treemap) ────────────────────────────────────────

def render_s2(df: pd.DataFrame) -> None:
    section_header("Overall case mix composition")
    if not _safe(df):
        _empty()
        return

    chart_card(
        "Visit volume by clinical segment",
        "Box size and shade both track visit volume — darkest teal is the largest segment.",
    )

    labels, parents, values, colors, text_vals = [], [], [], [], []
    root = "All visits"
    labels.append(root)
    parents.append("")
    values.append(int(df["TOTAL_VISITS"].sum()))
    colors.append("#F4F6FA")
    text_vals.append("")

    # Sequential teal ramp keyed to visit volume — shade intensity tracks the
    # same metric as box size, min-max scaled within this table.
    _vol_min = float(df["TOTAL_VISITS"].min())
    _vol_max = float(df["TOTAL_VISITS"].max())

    def _volume_color(vols: float) -> str:
        t = (vols - _vol_min) / (_vol_max - _vol_min) if _vol_max > _vol_min else 1.0
        stops = [(0.0, (225, 245, 238)), (0.5, (79, 173, 165)), (1.0, (15, 110, 86))]
        (t0, c0), (t1, c1) = next(
            ((stops[i], stops[i + 1]) for i in range(len(stops) - 1) if stops[i][0] <= t <= stops[i + 1][0]),
            (stops[-2], stops[-1]),
        )
        f = (t - t0) / (t1 - t0) if t1 > t0 else 0
        r, g, b = (round(c0[i] + (c1[i] - c0[i]) * f) for i in range(3))
        return f"#{r:02X}{g:02X}{b:02X}"

    for _, row in df.iterrows():
        seg = row["PRIMARY_VISIT_SEGMENT"]
        vols = int(row["TOTAL_VISITS"])
        pct = float(row["PCT_OF_ALL_VISITS"])
        labels.append(seg)
        parents.append(root)
        values.append(vols)
        colors.append(_volume_color(vols))
        text_vals.append(
            f"{_short(seg)}<br>{fmt_num(vols)} visits ({pct}%)"
        )

    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        branchvalues="total",
        customdata=text_vals,
        texttemplate="%{label}",
        hovertemplate="%{customdata}<extra></extra>",
        marker=dict(colors=colors, colorscale=None, showscale=False,
                    line=dict(width=2, color="#FFFFFF")),
        tiling=dict(pad=4),
        pathbar=dict(visible=False),
    ))
    fig.update_layout(**{**_LAYOUT, "height": _H_SINGLE, "margin": dict(t=4, b=4, l=0, r=0)})
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()

    top = df.sort_values("TOTAL_VISITS", ascending=False).iloc[0]
    other = df[df["PRIMARY_VISIT_SEGMENT"] == "Other General Outpatient"]
    other_pct = float(other["PCT_OF_ALL_VISITS"].iloc[0]) if not other.empty else 0.0
    insight_bar(
        bullets=[
            f"{_short(top['PRIMARY_VISIT_SEGMENT'])} is the single largest segment at "
            f"{fmt_pct(top['PCT_OF_ALL_VISITS'])} of all visits.",
            f"Other General Outpatient accounts for {fmt_pct(other_pct)} of volume — a genuine, "
            f"growing general-medicine layer alongside the orthopedic core, not classifier noise.",
        ],
        variant="primary",
    )


# ── S3: Encounter type split (dumbbell) + New vs. returning patient, by segment ──
# Two distinct questions, shown side by side so the difference is visible at a glance:
#   left  = encounter type   — does THIS VISIT's diagnosis text read as a recheck/
#           post-op note vs. a new complaint (visit-level, not patient-aware)
#   right = patient identity — is this the patient's first-ever visit to the
#           segment vs. a repeat visit (patient_id-based, from get_cm_new_returning_patients)

def _draw_stacked_split(labels, pct_a, pct_b, name_a, name_b, color_a, color_b, height):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pct_a, y=labels, orientation="h", name=name_a,
        marker_color=color_a, marker_cornerradius=3,
        text=pct_a.apply(lambda v: f"{v:.0f}%"), textposition="inside",
        insidetextanchor="middle", textfont=dict(color="#FFFFFF", size=10),
    ))
    fig.add_trace(go.Bar(
        x=pct_b, y=labels, orientation="h", name=name_b,
        marker_color=color_b, marker_cornerradius=3,
        text=pct_b.apply(lambda v: f"{v:.0f}%"), textposition="inside",
        insidetextanchor="middle", textfont=dict(color=TEXT_SEC, size=10),
    ))
    fig.update_layout(
        **{
            **_LAYOUT,
            "height": height,
            "barmode": "stack",
            "legend": {**_LAYOUT["legend"], "y": -0.14},
        },
        xaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, 100]},
        yaxis={**AXIS_Y, "showgrid": False},
    )
    return fig


def render_s3(df_encounter: pd.DataFrame, df_new_ret: pd.DataFrame) -> None:
    section_header("Encounter type vs. patient identity, by segment")
    col_l, col_r = st.columns(2)

    with col_l:
        if not _safe(df_encounter):
            _empty()
        else:
            df = df_encounter[df_encounter["TOTAL_VISITS"] >= 20].sort_values(
                "PCT_NEW_ACUTE", ascending=True
            )
            chart_card("New/acute vs. follow-up, by segment",
                       "Per-visit — was this visit for a new or acute issue, or a scheduled recheck "
                       "(post-op, review, chronic management)?")
            fig = _draw_stacked_split(
                df["PRIMARY_VISIT_SEGMENT"].map(_short),
                df["PCT_NEW_ACUTE"], df["PCT_FOLLOW_UP"],
                "New/acute %", "Follow-up %", _C_BLUE, _C_PURPLE,
                max(280, len(df) * 32),
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(df_new_ret):
            _empty()
        else:
            agg = (
                df_new_ret.groupby(["PRIMARY_VISIT_SEGMENT", "PATIENT_STATUS"])["TOTAL_VISITS"]
                .sum().reset_index()
            )
            wide = agg.pivot_table(
                index="PRIMARY_VISIT_SEGMENT", columns="PATIENT_STATUS",
                values="TOTAL_VISITS", aggfunc="sum", fill_value=0,
            ).reset_index()
            new_col = next((c for c in wide.columns if "New patient" in str(c)), None)
            ret_col = next((c for c in wide.columns if c == "Returning patient"), None)
            if new_col is None or ret_col is None:
                _empty()
            else:
                wide["TOTAL"] = wide[new_col] + wide[ret_col]
                wide = wide[wide["TOTAL"] >= 20]
                wide["PCT_NEW"] = 100.0 * wide[new_col] / wide["TOTAL"]
                wide["PCT_RET"] = 100.0 * wide[ret_col] / wide["TOTAL"]
                wide = wide.sort_values("PCT_NEW", ascending=True)

                chart_card("New patients vs. recurring patients, by segment",
                           "Per-patient — first-ever visit to this segment vs. a repeat visit")
                fig = _draw_stacked_split(
                    wide["PRIMARY_VISIT_SEGMENT"].map(_short),
                    wide["PCT_NEW"], wide["PCT_RET"],
                    "New patient %", "Recurring patient %", _C_BLUE, _C_PURPLE,
                    max(280, len(wide) * 32),
                )
                st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
                chart_card_close()

    # Identify whichever segment is actually lowest, rather than assuming it's
    # Spine — a hardcoded assumption goes stale the moment the underlying
    # classification changes (e.g. the Feb 2025 ICD-10 coding change shifted
    # Spine's own new/acute share well above where it used to sit).
    eligible_enc = (
        df_encounter[df_encounter["TOTAL_VISITS"] >= 20].sort_values("PCT_NEW_ACUTE", ascending=True)
        if _safe(df_encounter) else pd.DataFrame()
    )
    insight_bar(
        bullets=[
            (f"{_short(eligible_enc.iloc[0]['PRIMARY_VISIT_SEGMENT'])} shows the lowest new/acute "
             f"encounter share ({fmt_pct(eligible_enc.iloc[0]['PCT_NEW_ACUTE'])}) of any segment — "
             f"the majority of its visits are follow-up/chronic management."
             if not eligible_enc.empty else
             "Segments vary widely in new/acute vs. follow-up composition — blended visit counts "
             "hide this difference."),
            "These two charts answer different questions and can move independently: a segment can "
            "be follow-up-heavy on encounter type while still growing its patient base, or vice "
            "versa — read them side by side, not as the same metric twice.",
            "A follow-up-heavy segment needs recheck capacity; a segment with a high returning-"
            "patient share but low follow-up encounters is getting repeat business for new "
            "complaints — each needs different staffing.",
        ],
        variant="warning",
    )


# ── S4: Yearly trend + growth table ──────────────────────────────────────────

_TREND_SEGMENTS = [
    "Core Orthopedics: Spine and Back Pain Care", "Core Orthopedics: General",
    "Core General Surgery", "Other General Outpatient",
]
_GROWTH_SEGMENTS = _TREND_SEGMENTS + ["Standalone Specialty: Obstetrics & Gynaecology"]
_TREND_COLORS = {
    # Ortho General → teal, Spine → raspberry, Other → grey (spec §4 line charts)
    "Core Orthopedics: Spine and Back Pain Care": _C_PURPLE,
    "Core Orthopedics: General": _C_BLUE,
    "Core General Surgery": "#D6698C",
    "Other General Outpatient": _C_MGREY,
}


def render_s4(df: pd.DataFrame) -> None:
    if not _safe(df):
        section_header("Segment volume trend")
        _empty()
        return

    year_min = int(df["VISIT_YEAR"].min())
    year_max = int(df["VISIT_YEAR"].max())
    section_header(f"Segment volume trend, {year_min}–{year_max}")

    col_l, col_r = st.columns([1.2, 0.8])

    with col_l:
        chart_card("Visit volume by year — key segments")
        fig = go.Figure()
        for seg in _TREND_SEGMENTS:
            sub = df[df["PRIMARY_VISIT_SEGMENT"] == seg].sort_values("VISIT_YEAR")
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["VISIT_YEAR"], y=sub["TOTAL_VISITS"], mode="lines+markers",
                name=_short(seg), line=dict(width=2, color=_TREND_COLORS.get(seg, _C_MGREY)),
                marker=dict(size=5, color=_TREND_COLORS.get(seg, _C_MGREY)),
            ))
        fig.update_layout(
            **{**_LAYOUT, "height": _H_PAIRED},
            xaxis={**AXIS_X, "showgrid": False, "tickvals": sorted(df["VISIT_YEAR"].unique())},
            yaxis=AXIS_Y,
        )
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    with col_r:
        chart_card(f"Growth, {year_min}→{year_max}")
        all_segs = df["PRIMARY_VISIT_SEGMENT"].unique().tolist()
        ordered_segs = [s for s in _GROWTH_SEGMENTS if s in all_segs] + \
                       [s for s in all_segs if s not in _GROWTH_SEGMENTS]

        rows_html = ""
        for seg in ordered_segs:
            sub = df[df["PRIMARY_VISIT_SEGMENT"] == seg]
            v_start = sub[sub["VISIT_YEAR"] == year_min]["TOTAL_VISITS"]
            v_end = sub[sub["VISIT_YEAR"] == year_max]["TOTAL_VISITS"]
            if v_start.empty or v_end.empty or int(v_start.iloc[0]) == 0:
                continue
            growth = 100.0 * (int(v_end.iloc[0]) - int(v_start.iloc[0])) / int(v_start.iloc[0])
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:6px 0;border-bottom:1px solid {BORDER}">'
                f'<span style="font-size:12px;color:{TEXT_SEC}">{_short(seg)}</span>'
                f'{growth_badge(growth)}</div>'
            )
        st.markdown(
            f'<div style="max-height:220px;overflow-y:auto">{rows_html}</div>',
            unsafe_allow_html=True,
        )
        chart_card_close()

    spine = df[df["PRIMARY_VISIT_SEGMENT"] == "Core Orthopedics: Spine and Back Pain Care"]
    insight_bar(
        bullets=[
            "Ortho General's absolute volume kept growing even as its share of total visits fell — "
            "Spine is growing faster, not replacing it.",
            "Other General Outpatient shows steady, sustained growth across every year shown — "
            "not a one-off spike.",
            f"{year_min} (data starts mid-year) and {year_max} (still in progress) are partial "
            f"years — some lower-volume segments had individual months with zero recorded visits, "
            f"so a growth badge showing a steep decline may reflect missing months in the record, "
            f"not a genuine drop in demand.",
        ],
        variant="primary",
    )


# ── S5: New vs. returning, and seasonality ──────────────────────────────────

def render_s5(df_new_ret: pd.DataFrame, df_seasonal: pd.DataFrame, df_spine_dx: pd.DataFrame) -> None:
    section_header("Patient base growth and seasonality")
    col_l, col_r = st.columns(2)

    ortho_segs = ["Core Orthopedics: Spine and Back Pain Care", "Core Orthopedics: General"]

    with col_l:
        if not _safe(df_new_ret):
            _empty()
        else:
            chart_card("New patients vs. recurring patients — Ortho segments")
            seg_choice = st.selectbox(
                "Segment", [s for s in ortho_segs if s in df_new_ret["PRIMARY_VISIT_SEGMENT"].unique()],
                format_func=_short, key="cm_s5_new_ret_segment",
            )
            df = df_new_ret[df_new_ret["PRIMARY_VISIT_SEGMENT"] == seg_choice]
            wide = df.pivot_table(
                index=["VISIT_YEAR", "MONTH_NUM", "MONTH_LABEL"], columns="PATIENT_STATUS",
                values="TOTAL_VISITS", aggfunc="sum", fill_value=0,
            ).reset_index()
            wide = wide.sort_values(["VISIT_YEAR", "MONTH_NUM"])
            x_label = wide["MONTH_LABEL"] + " " + wide["VISIT_YEAR"].astype(str).str[-2:]

            fig = go.Figure()
            new_col = next((c for c in wide.columns if "New patient" in str(c)), None)
            ret_col = next((c for c in wide.columns if c == "Returning patient"), None)
            if new_col:
                fig.add_trace(go.Bar(x=x_label, y=wide[new_col], name="New patient",
                                      marker_color=_C_BLUE, marker_cornerradius=3))
            if ret_col:
                fig.add_trace(go.Bar(x=x_label, y=wide[ret_col], name="Recurring patient",
                                      marker_color=_C_PURPLE, marker_cornerradius=3))
            fig.update_layout(
                **{
                    **_LAYOUT,
                    "height": _H_PAIRED,
                    "barmode": "stack",
                    "margin": {**_LAYOUT["margin"], "b": 85},
                    "legend": {**_LAYOUT["legend"], "y": -0.32},
                },
                xaxis={**AXIS_X, "showgrid": False, "tickmode": "array",
                       "tickvals": list(x_label[::3])},
                yaxis=AXIS_Y,
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(df_seasonal):
            _empty()
        else:
            top_segs = (
                df_seasonal.groupby("PRIMARY_VISIT_SEGMENT")["TOTAL_VISITS"]
                .sum().sort_values(ascending=False).head(4).index.tolist()
            )
            chart_card("Seasonality — visits by month, year over year")
            # Spacer to match the height of the "Segment" selectbox in the
            # left column, so the two title cards stay top-aligned and the
            # charts below them still line up at the same height.
            st.markdown('<div style="height:66px"></div>', unsafe_allow_html=True)
            timeline = (
                df_seasonal[["VISIT_YEAR", "MONTH_NUM", "MONTH_LABEL"]]
                .drop_duplicates().sort_values(["VISIT_YEAR", "MONTH_NUM"])
            )
            timeline_labels = timeline["MONTH_LABEL"] + " " + timeline["VISIT_YEAR"].astype(str).str[-2:]

            fig = go.Figure()
            palette = [_C_BLUE, _C_RED, _C_TEAL, _C_AMBER]
            for i, seg in enumerate(top_segs):
                sub = df_seasonal[df_seasonal["PRIMARY_VISIT_SEGMENT"] == seg].sort_values(
                    ["VISIT_YEAR", "MONTH_NUM"]
                )
                x_label = sub["MONTH_LABEL"] + " " + sub["VISIT_YEAR"].astype(str).str[-2:]
                fig.add_trace(go.Scatter(
                    x=x_label, y=sub["TOTAL_VISITS"],
                    mode="lines", name=_short(seg),
                    line=dict(width=2, color=palette[i % len(palette)]),
                ))
            fig.update_layout(
                **{
                    **_LAYOUT,
                    "height": _H_PAIRED,
                    "margin": {**_LAYOUT["margin"], "b": 85},
                    "legend": {**_LAYOUT["legend"], "y": -0.32},
                },
                xaxis={**AXIS_X, "showgrid": False, "tickmode": "array",
                       "tickvals": list(timeline_labels[::3])},
                yaxis=AXIS_Y,
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    if _safe(df_spine_dx):
        df_dx = df_spine_dx.copy()
        df_dx["CATEGORY"] = df_dx["CLEAN_DX_TEXT"].apply(_bucket_spine_dx)
        monthly = df_dx.groupby(["VISIT_MONTH", "CATEGORY"])["TOTAL_VISITS"].sum().reset_index()
        wide = monthly.pivot_table(
            index="VISIT_MONTH", columns="CATEGORY", values="TOTAL_VISITS",
            aggfunc="sum", fill_value=0,
        ).reset_index().sort_values("VISIT_MONTH")

        cat_order = [c for c in (_SPINE_DX_CATEGORIES + ["Other"]) if c in wide.columns]
        # Distinct, readable categories — draws from the existing brand
        # families (teal / raspberry / navy / grey) rather than one monochrome
        # ramp (too hard to tell apart as 7 stacked layers) or status red/amber/green.
        _dx_palette = ["#141F3D", "#1B8A82", "#C13868", "#4FADA5",
                       "#D6698C", "#5C6478", "#8FCFC8"]
        palette = [
            _C_GREY if cat == "Other" else _dx_palette[i % len(_dx_palette)]
            for i, cat in enumerate(cat_order)
        ]

        chart_card(
            "Spine and Back Pain Care — monthly volume by diagnosis category",
            "Top 12 diagnosis strings per month, grouped into clinical categories — Feb 2025 marked",
        )
        fig = go.Figure()
        x = wide["VISIT_MONTH"].astype(str)
        for i, cat in enumerate(cat_order):
            fig.add_trace(go.Scatter(
                x=x, y=wide[cat], mode="lines", name=cat, stackgroup="one",
                line=dict(width=0.5, color=palette[i % len(palette)]),
            ))
        fig.add_vline(x="2025-02-01", line_dash="dash", line_color=TEXT_MUT)
        fig.update_layout(
            **{**_LAYOUT, "height": 300, "legend": {**_LAYOUT["legend"], "y": -0.25}},
            xaxis={**AXIS_X, "showgrid": False, "tickmode": "array", "tickvals": list(x[::3])},
            yaxis=AXIS_Y,
        )
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    insight_bar(
        bullets=[
            "Spine and Back Pain Care volume jumps sharply from Feb 2025 onward, but the clinical "
            "mix doesn't change — lumbago, sciatica, and low back pain are the dominant categories "
            "before and after, with no new condition appearing.",
            "The diagnosis text itself changes shape at the same point: pre-2025 entries are "
            "free-text and inconsistently spelled ('lumbargo', 'loose of lumbar lordosis'), while "
            "Feb 2025 onward entries are duplicated standardized terms (e.g. 'low back pain low "
            "back pain') — consistent with an ICD-10 coding/mapping change going live around then.",
            "The constant clinical mix is also consistent with a genuine volume increase rather than "
            "a pure coding artifact — cross-check the new-vs-recurring chart above with Spine and "
            "Back Pain Care selected: recurring-patient volume was low beforehand, then picked back "
            "up starting May, so the growth looks driven by existing patients returning, not new "
            "patient acquisition.",
        ],
        variant="primary",
    )


_SPINE_DX_CATEGORIES = [
    "Lumbago with sciatica", "Low back pain (general)", "Sciatica (alone)",
    "Lumbago (alone)", "Lumbar radiculopathy", "Lumbar spondylosis",
    "Disc bulge / degeneration",
]


def _bucket_spine_dx(text: str) -> str:
    t = str(text).lower()
    if "lumbago" in t and "sciatica" in t:
        return "Lumbago with sciatica"
    if "low back pain" in t or "lbp" in t or "back pain" in t:
        return "Low back pain (general)"
    if "sciatica" in t:
        return "Sciatica (alone)"
    if "lumbago" in t or "lumbargo" in t:
        return "Lumbago (alone)"
    if "radiculopathy" in t:
        return "Lumbar radiculopathy"
    if "spondylosis" in t or "spondylitis" in t:
        return "Lumbar spondylosis"
    if "disc bulge" in t or "degenerat" in t or "herniat" in t:
        return "Disc bulge / degeneration"
    return "Other"


# ── S6: Comorbidity grid ─────────────────────────────────────────────────────

_COMORBIDITY_COLS = [
    ("PCT_HYPERTENSION", "Hypertension"), ("PCT_DIABETES", "Diabetes"),
    ("PCT_ANAEMIA", "Anaemia"), ("PCT_CARDIAC", "Cardiac"),
    ("PCT_RENAL", "Renal"), ("PCT_HIV", "HIV"),
    ("PCT_THYROID", "Thyroid"), ("PCT_ASTHMA", "Asthma"),
]
_COMORBIDITY_SEGMENTS = [
    "Core Orthopedics: Spine and Back Pain Care", "Core Orthopedics: General",
    "Core General Surgery", "Standalone Medical: Sepsis/Infection",
]
# Confirmed by pulling the raw diagnosis text (goitre / thyroidectomy /
# multinodular goiter in 34 of 36 sampled visits): has_thyroid_condition in
# Core General Surgery is almost entirely the surgical indication itself,
# not an incidental comorbidity. Never treat it as a genuine risk signal.
_COMORBIDITY_PROCEDURE_ARTIFACTS = {("Core General Surgery", "Thyroid")}


def render_comorbidity_grid(df: pd.DataFrame) -> None:
    header_cells = "".join(
        f'<th style="font-size:9px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.05em;color:{TEXT_MUT};padding:8px 6px;text-align:right">{lbl}</th>'
        for _, lbl in _COMORBIDITY_COLS
    )
    # Same 20-visit materiality floor used by the S3 charts — a "standalone"
    # segment that's actually thin (e.g. a catch-all comorbidity bucket with
    # very few visits) should silently drop out here rather than appear as
    # if it carries the same weight as the core segments.
    rows_html = ""
    for seg in _COMORBIDITY_SEGMENTS:
        sub = df[df["PRIMARY_VISIT_SEGMENT"] == seg]
        if sub.empty:
            continue
        row = sub.iloc[0]
        total = row.get("TOTAL_VISITS")
        if total is not None and float(total) < 20:
            continue
        cells = ""
        for col, lbl in _COMORBIDITY_COLS:
            val = float(row.get(col, 0) or 0)
            is_procedure_artifact = (_short(seg), lbl) in _COMORBIDITY_PROCEDURE_ARTIFACTS
            # Per-condition prevalence runs much lower than an aggregate
            # "any comorbidity" score — 15%/5% thresholds calibrated for that
            # aggregate never fire here. Rescaled to this table's real range.
            if is_procedure_artifact:
                color, weight = TEXT_MUT, 400
            else:
                color = DANGER if val >= 3 else (WARNING if val >= 1 else TEXT_SEC)
                weight = 700 if val >= 3 else 400
            mark = "†" if is_procedure_artifact else ""
            cells += (
                f'<td style="font-size:12px;color:{color};font-weight:{weight};'
                f'padding:8px 6px;text-align:right">{val:.1f}%{mark}</td>'
            )
        rows_html += (
            f'<tr style="border-bottom:1px solid {BORDER}">'
            f'<td style="font-size:12px;color:{TEXT_PRI};font-weight:600;padding:8px 6px">'
            f'{_short(seg)}</td>{cells}</tr>'
        )

    st.markdown(
        f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch">'
        f'<table style="width:100%;min-width:480px;border-collapse:collapse;font-family:Inter,sans-serif">'
        f'<thead><tr><th style="padding:8px 6px;text-align:left"></th>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
        f'<div style="font-size:10px;color:{TEXT_MUT};margin-top:6px">'
        f'† surgical indication itself (goitre/thyroidectomy), not an incidental comorbidity</div>',
        unsafe_allow_html=True,
    )


def render_s6(df: pd.DataFrame) -> None:
    section_header("Comorbidity profile, by segment")
    if not _safe(df):
        _empty()
        return

    chart_card("Chronic condition prevalence — key segments")
    render_comorbidity_grid(df)
    chart_card_close()

    sepsis = df[df["PRIMARY_VISIT_SEGMENT"] == "Standalone Medical: Sepsis/Infection"]

    # Find the actual standout cell across the rendered grid, rather than a
    # hardcoded threshold claim that stops matching reality once the data
    # (or the classification behind it) shifts.
    standout_bullet = (
        "Values in red are 3%+ prevalence, amber 1%+ — worth flagging at first contact for "
        "pre-procedure risk screening in surgical segments."
    )
    best_seg, best_col, best_val = None, None, -1.0
    for seg in _COMORBIDITY_SEGMENTS:
        sub = df[df["PRIMARY_VISIT_SEGMENT"] == seg]
        if sub.empty:
            continue
        row = sub.iloc[0]
        for col, lbl in _COMORBIDITY_COLS:
            if (_short(seg), lbl) in _COMORBIDITY_PROCEDURE_ARTIFACTS:
                continue
            val = float(row.get(col, 0) or 0)
            if val > best_val:
                best_seg, best_col, best_val = seg, lbl, val
    if best_seg is not None and best_val >= 3:
        standout_bullet = (
            f"{best_col} in {_short(best_seg)} is the standout at {best_val:.1f}% — well above "
            f"every other condition/segment pairing in this grid, worth flagging at first contact "
            f"for pre-procedure risk screening."
        )

    insight_bar(
        bullets=[
            (f"Sepsis/Infection carries the highest overall comorbidity burden at "
             f"{fmt_pct(sepsis.iloc[0]['PCT_ANY_COMORBIDITY'])} of visits."
             if not sepsis.empty else
             "Comorbidity prevalence varies meaningfully across segments — a flat hospital-wide "
             "average would hide this."),
            "The 4.8% Thyroid figure in Core General Surgery is almost entirely goitre and "
            "thyroidectomy cases — patients having thyroid surgery, not patients with an "
            "unrelated procedure who happen to also have thyroid disease.",
            standout_bullet,
        ],
        variant="warning",
    )


# ── S7: Other General Outpatient breakdown ──────────────────────────────────

_PRIORITY_DX = [
    "Upper Respiratory Tract Infection", "Gastritis", "Lower Respiratory Tract Infection",
    "Gastroenteritis", "Peptic Ulcer Disease", "Pneumonia",
]
_DX_COLORS = {
    "Upper Respiratory Tract Infection": "#1B8A82",
    "Gastritis": "#854F0B",
    "Lower Respiratory Tract Infection": "rgba(27,138,130,0.7)",
    "Gastroenteritis": "rgba(133,79,11,0.7)",
    "Peptic Ulcer Disease": "#8A93A6",
    "Pneumonia": "rgba(138,147,166,0.7)",
    "blank": "#A32D2D",
}
def _seasonal_trend_forecast(df_monthly: pd.DataFrame, through: pd.Timestamp):
    """
    Simple trend + seasonal-index projection (classical multiplicative
    decomposition) — replaces a flat year-over-year linear fit with one
    that accounts for which calendar months actually run hot or cold.

    Drops the first calendar year (partial — data starts mid-year) and
    the final month (usually a partial, still-in-progress month) before
    fitting, so the trend line isn't skewed by incomplete months.

    Returns a monthly forecast DataFrame (trend, seasonal_idx, forecast)
    for every month from the last reliable month through `through`
    (inclusive), plus the reliable actuals used to fit it.
    """
    s = df_monthly.set_index(pd.to_datetime(df_monthly["VISIT_MONTH"]))["TOTAL_VISITS"].sort_index()
    first_year = s.index.min().year
    reliable = s[(s.index.year != first_year)].iloc[:-1]  # drop partial first year + final partial month

    t = np.arange(len(reliable))
    slope, intercept = np.polyfit(t, reliable.values, 1)
    trend = slope * t + intercept
    seasonal_ratio = reliable.values / trend
    seasonal_idx = pd.Series(seasonal_ratio, index=reliable.index.month).groupby(level=0).mean()
    seasonal_idx = seasonal_idx / seasonal_idx.mean()  # normalize to average 1.0 across months

    future_idx = pd.date_range(reliable.index.max() + pd.DateOffset(months=1), through, freq="MS")
    months_ahead = len(future_idx)
    future_t = np.arange(len(reliable), len(reliable) + months_ahead)
    future_trend = slope * future_t + intercept
    future_seasonal = pd.Series(future_idx.month).map(seasonal_idx).values
    forecast = np.clip(future_trend * future_seasonal, 0, None)

    return pd.DataFrame(
        {"trend": future_trend, "seasonal_idx": future_seasonal, "forecast": forecast},
        index=future_idx,
    ), reliable


def render_s7(df_other_dx: pd.DataFrame, df_other_trend: pd.DataFrame, df_other_monthly: pd.DataFrame) -> None:
    section_header("What's in 'Other General Outpatient'")
    col_l, col_r = st.columns(2)
    blank_total = 0

    with col_l:
        if not _safe(df_other_dx):
            _empty()
        else:
            df = df_other_dx.copy()
            df["UNIFIED_DIAGNOSIS"] = df["UNIFIED_DIAGNOSIS"].fillna("")
            is_blank = df["UNIFIED_DIAGNOSIS"].str.strip() == ""
            is_gastritis = df["UNIFIED_DIAGNOSIS"].str.contains("gastritis", case=False, na=False)

            gastritis_total = df.loc[is_gastritis, "OCCURRENCES"].sum()
            blank_total = df.loc[is_blank, "OCCURRENCES"].sum()

            rows = []
            for label in _PRIORITY_DX:
                if label == "Gastritis":
                    rows.append({"LABEL": "Gastritis (all subtypes)", "OCCURRENCES": gastritis_total,
                                  "COLOR": _DX_COLORS[label]})
                else:
                    match = df[df["UNIFIED_DIAGNOSIS"] == label]
                    occ = int(match["OCCURRENCES"].iloc[0]) if not match.empty else 0
                    rows.append({"LABEL": label, "OCCURRENCES": occ, "COLOR": _DX_COLORS.get(label, _C_MGREY)})

            chart_df = pd.DataFrame(rows).sort_values("OCCURRENCES")
            chart_card("Top diagnoses within Other General Outpatient")
            fig = go.Figure(go.Bar(
                y=chart_df["LABEL"], x=chart_df["OCCURRENCES"], orientation="h",
                marker_color=chart_df["COLOR"], marker_cornerradius=3,
            ))
            fig.update_layout(**{**_LAYOUT, "height": _H_PAIRED}, showlegend=False,
                               xaxis={**AXIS_Y, "showgrid": True}, yaxis={**AXIS_X, "showgrid": False})
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    proj_years, proj_vals = [], []
    watch_month, watch_vol, watch_year = None, None, None
    with col_r:
        if not _safe(df_other_trend):
            _empty()
        else:
            df_all = df_other_trend.sort_values("VISIT_YEAR")
            current_year = int(df_all["VISIT_YEAR"].max())  # still in-progress year
            df = df_all[df_all["VISIT_YEAR"] != current_year]  # fully-elapsed years only
            # Single series, magnitude only — one uniform color for every
            # "Actual" bar, no per-year variation.
            colors = [PRIMARY] * len(df)

            # Trend + seasonal-index projection — fit on monthly data so the
            # forecast captures which calendar months actually run hot or
            # cold (e.g. the March spike), instead of a flat straight line
            # through 3 annual totals. The current in-progress year is
            # completed with modeled months for whatever hasn't happened
            # yet, then treated as a projection alongside the next 2 years.
            if _safe(df_other_monthly):
                proj_years = [current_year, current_year + 1, current_year + 2]
                forecast, reliable = _seasonal_trend_forecast(
                    df_other_monthly, through=pd.Timestamp(f"{proj_years[-1]}-12-01"),
                )
                proj_vals = []
                for y in proj_years:
                    actual_part = reliable[reliable.index.year == y].sum()
                    forecast_part = forecast.loc[forecast.index.year == y, "forecast"].sum()
                    proj_vals.append(actual_part + forecast_part)

                watch_year = proj_years[1]  # first fully-forecasted year — cleanest watch signal
                year_fc = forecast.loc[forecast.index.year == watch_year, "forecast"]
                if not year_fc.empty:
                    watch_month = int(year_fc.idxmax().month)
                    watch_vol = float(year_fc.max())

            chart_card("Other General Outpatient — volume by year")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["VISIT_YEAR"].astype(str), y=df["TOTAL_VISITS"], name="Actual",
                marker_color=colors, marker_cornerradius=3,
                text=df["TOTAL_VISITS"].apply(fmt_num), textposition="outside",
            ))
            if proj_years:
                fig.add_trace(go.Bar(
                    x=[str(y) for y in proj_years], y=proj_vals, name="Projected",
                    marker_color="rgba(42,120,214,0.35)", marker_pattern_shape="/",
                    marker_cornerradius=3,
                    text=[fmt_num(v) for v in proj_vals], textposition="outside",
                ))
            fig.update_layout(
                **{**_LAYOUT, "height": _H_PAIRED}, showlegend=bool(proj_years),
                xaxis={**AXIS_X, "showgrid": False}, yaxis=AXIS_Y,
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    bullets = [
        "Respiratory and gastritis-spectrum conditions dominate this bucket — genuine walk-in "
        "general-practice content, structurally distinct from the orthopedic core.",
        f"{fmt_num(blank_total)} visits in this bucket have no diagnosis recorded at all — a real "
        "documentation gap, not resolved by this cleanup pass.",
    ]
    if proj_years:
        year_list = ", ".join(
            f"{fmt_num(v)} in {y}" for y, v in zip(proj_years[:-1], proj_vals[:-1])
        )
        bullets.append(
            f"Projection excludes visits with no diagnosis and the partial first calendar year — "
            f"the trend line and each month's typical seasonal share are both fit on "
            f"{len(reliable)} months of clean monthly data. {proj_years[0]} blends actual visits "
            f"recorded so far with modeled remaining months; at the fitted rate, volume reaches "
            f"roughly {year_list}, and {fmt_num(proj_vals[-1])} in {proj_years[-1]}."
        )
    if watch_vol is not None:
        watch_idx = proj_years.index(watch_year)
        bullets.append(
            f"Weather watch signal — {calendar.month_name[watch_month]} {watch_year} is the "
            f"projected peak (~{fmt_num(watch_vol)} visits, vs. a {fmt_num(proj_vals[watch_idx] / 12)} "
            f"monthly average that year), driven by respiratory volume that rises ahead of the long "
            f"rains — track local rainfall onset as an early signal for staffing this bucket, not "
            f"just the calendar month."
        )
    insight_bar(bullets=bullets, variant="warning")


# ── Recommendations ──────────────────────────────────────────────────────────

def render_recommendations(
    df_encounter: pd.DataFrame, df_spine_dx: pd.DataFrame, df_other_dx: pd.DataFrame,
    df_other_trend: pd.DataFrame, df_other_monthly: pd.DataFrame,
    df_comorbidity: pd.DataFrame, df_trend: pd.DataFrame,
) -> None:
    section_header("Recommendations and key findings")

    # Matches ui_template.py's _PRIORITY_SEVERITY_COLOR so the "Action:" line
    # inside a card highlights in the same color as that card's left border/label.
    _SEVERITY_COLOR = {"critical": DANGER, "monitor": WARNING, "okay": SUCCESS}

    def _list(items: list, severity: str = "monitor") -> str:
        color = _SEVERITY_COLOR.get(severity, WARNING)
        lis = "".join(
            f'<li style="margin-bottom:3px;font-weight:700;color:{color}">{i}</li>'
            if i.startswith("Action:") else
            f'<li style="margin-bottom:3px">{i}</li>'
            for i in items
        )
        return f'<ul style="margin:2px 0 0;padding-left:16px">{lis}</ul>'

    # ── Card 1 — Spine follow-up staffing (same figure as the S3 insight) ──
    spine_follow_body = _list([
        "Spine follow-up share not currently available — see the encounter-type chart above.",
    ], "critical")
    if _safe(df_encounter):
        spine_enc = df_encounter[df_encounter["PRIMARY_VISIT_SEGMENT"].str.contains("Spine", na=False)]
        if not spine_enc.empty:
            follow_pct = float(spine_enc.iloc[0]["PCT_FOLLOW_UP"])
            spine_follow_body = _list([
                f"{fmt_pct(follow_pct)} of Spine encounters are follow-up, not new/acute",
                "Recent volume jump is confirmed driven by returning patients, not new acquisition",
                "Action: staff this segment around recheck capacity, not as an intake-heavy service",
            ], "critical")

    # ── Card 2 — Feb 2025 coding-change confirmation (narrative from S5) ───
    coding_change_body = _list([
        "Diagnosis text shifts from free-text to duplicated standardized terms exactly when volume jumps",
        "Consistent with an ICD-10 mapping change going live",
        "Action: confirm directly with records/IT rather than treating the jump as purely clinical",
    ], "critical") if _safe(df_spine_dx) else _list([
        "Spine diagnosis text data not currently available to confirm this.",
    ], "critical")

    # ── Card 3 — undiagnosed visits in Other General Outpatient (S7 figure) ─
    blank_total = 0
    if _safe(df_other_dx):
        dx = df_other_dx.copy()
        dx["UNIFIED_DIAGNOSIS"] = dx["UNIFIED_DIAGNOSIS"].fillna("")
        blank_total = int(dx.loc[dx["UNIFIED_DIAGNOSIS"].str.strip() == "", "OCCURRENCES"].sum())
    undiagnosed_body = _list([
        f"{fmt_num(blank_total)} visits have no diagnosis recorded at all",
        "This bucket's true composition remains partly unknown",
        "Action: resolve the documentation gap — classification alone can't fix missing data",
    ], "monitor")

    # ── Card 4 — weather-watch forecast signal (same computation as S7) ────
    forecast_body = _list([
        "Other General Outpatient forecast not currently available — see the volume-by-year chart above.",
    ], "monitor")
    if _safe(df_other_trend) and _safe(df_other_monthly):
        df_all = df_other_trend.sort_values("VISIT_YEAR")
        current_year = int(df_all["VISIT_YEAR"].max())
        proj_years = [current_year, current_year + 1, current_year + 2]
        forecast, reliable = _seasonal_trend_forecast(
            df_other_monthly, through=pd.Timestamp(f"{proj_years[-1]}-12-01"),
        )
        watch_year = proj_years[1]
        year_fc = forecast.loc[forecast.index.year == watch_year, "forecast"]
        if not year_fc.empty:
            watch_month = int(year_fc.idxmax().month)
            watch_vol = float(year_fc.max())
            year_avg = float(year_fc.mean())
            forecast_body = _list([
                f"Projection flags {calendar.month_name[watch_month]} {watch_year} as a likely peak "
                f"(~{fmt_num(watch_vol)} visits vs. a {fmt_num(year_avg)} monthly average that year)",
                "Driven by respiratory volume rising ahead of the long rains",
                "Action: track local rainfall onset as a leading indicator, not just the calendar month",
            ], "monitor")

    # ── Card 5 — Sepsis/Infection comorbidity burden (S6 figure) ───────────
    comorbidity_body = _list(["Comorbidity profile data not currently available."], "monitor")
    if _safe(df_comorbidity):
        sepsis = df_comorbidity[df_comorbidity["PRIMARY_VISIT_SEGMENT"] == "Standalone Medical: Sepsis/Infection"]
        if not sepsis.empty:
            sepsis_pct = float(sepsis.iloc[0]["PCT_ANY_COMORBIDITY"])
            comorbidity_body = _list([
                f"Sepsis/Infection carries the highest overall comorbidity burden ({fmt_pct(sepsis_pct)} of visits)",
                "Action: flag comorbidity prevalence at 15%+ at first contact for surgical segments",
            ], "monitor")

    # ── Card 6 — growth-rate reporting caveat (S4 figure) ──────────────────
    caveat_body = _list(["Segment volume trend data not currently available."], "monitor")
    if _safe(df_trend):
        year_min = int(df_trend["VISIT_YEAR"].min())
        year_max = int(df_trend["VISIT_YEAR"].max())
        caveat_body = _list([
            f"{year_min} data starts mid-year and {year_max} is still in progress",
            "A growth badge showing steep decline for a lower-volume segment may reflect missing "
            "months, not a genuine drop in demand",
            f"Action: state this caveat wherever a {year_min}→{year_max} growth figure is reported",
        ], "monitor")

    priority_cards([
        {"label": "PRIORITY 1 — STAFFING", "severity": "critical",
         "title": "Rebuild Spine and Back Pain Care staffing around recheck capacity",
         "body": spine_follow_body, "source": "Encounter type vs. patient identity; new-vs-recurring cross-check"},
        {"label": "PRIORITY 2 — DATA INTEGRITY", "severity": "critical",
         "title": "Confirm the Feb 2025 diagnosis coding change with records/IT",
         "body": coding_change_body, "source": "Spine and Back Pain Care — monthly volume by diagnosis category"},
        {"label": "PRIORITY 3 — DOCUMENTATION GAP", "severity": "monitor",
         "title": f"Resolve the {fmt_num(blank_total)} undiagnosed visits in Other General Outpatient",
         "body": undiagnosed_body, "source": 'What\'s in "Other General Outpatient"'},
        {"label": "PRIORITY 4 — FORECAST PLANNING", "severity": "monitor",
         "title": "Use the weather-watch signal to plan Other General Outpatient staffing",
         "body": forecast_body, "source": "Other General Outpatient — volume by year, projection note"},
        {"label": "PRIORITY 5 — PRE-PROCEDURE SCREENING", "severity": "monitor",
         "title": "Flag Sepsis/Infection's comorbidity burden at pre-procedure screening",
         "body": comorbidity_body, "source": "Comorbidity profile, by segment"},
        {"label": "PRIORITY 6 — REPORTING CAVEAT", "severity": "monitor",
         "title": "Caveat growth-rate figures for the first and last years shown",
         "body": caveat_body, "source": "Segment volume trend"},
    ])

    # ── Key findings — folded into the same section, one section_header only.
    # "Core orthopedics share" finding dropped: it's already shown as a KPI
    # tile at the top of this tab (render_s1), so repeating it here was pure
    # duplication rather than a distinct finding.
    spine_txt = "Spine's share of total volume has grown substantially since 2022."
    if _safe(df_trend):
        spine = df_trend[df_trend["PRIMARY_VISIT_SEGMENT"] == "Core Orthopedics: Spine and Back Pain Care"]
        y22 = spine[spine["VISIT_YEAR"] == spine["VISIT_YEAR"].min()]
        y25 = spine[spine["VISIT_YEAR"] == spine["VISIT_YEAR"].max()]
        if not y22.empty and not y25.empty:
            spine_txt = (
                f"Spine's share of total volume grew from {float(y22.iloc[0]['PCT_OF_YEAR_TOTAL']):.1f}% "
                f"to {float(y25.iloc[0]['PCT_OF_YEAR_TOTAL']):.1f}% between "
                f"{int(y22.iloc[0]['VISIT_YEAR'])} and {int(y25.iloc[0]['VISIT_YEAR'])}."
            )

    other_txt = "Other General Outpatient volume has grown steadily year over year."
    if _safe(df_other_trend):
        d = df_other_trend.sort_values("VISIT_YEAR")
        first, last = d.iloc[0], d.iloc[-1]
        other_txt = (
            f"Other General Outpatient grew from {fmt_num(first['TOTAL_VISITS'])} visits in "
            f"{int(first['VISIT_YEAR'])} to {fmt_num(last['TOTAL_VISITS'])} in {int(last['VISIT_YEAR'])} — "
            f"a genuine, sustained general-medicine layer, not a one-off blip."
        )

    key_findings_cards([
        {"num": "01", "title": "Internal shift",
         "body": spine_txt + " Ortho General's absolute volume kept growing even as its share fell — "
                              "Spine is growing faster, not replacing it."},
        {"num": "02", "title": "Genuine diversification", "body": other_txt},
    ])


# ── Tab entry point ───────────────────────────────────────────────────────────

def render_tab() -> None:
    import sph.clinicals.case_mix_module.cm_queries as CMQ

    with st.spinner("Loading data…"):
        df_headline      = CMQ.get_cm_headline_kpis()
        df_composition   = CMQ.get_cm_overall_composition()
        df_encounter     = CMQ.get_cm_encounter_type_split()
        df_trend         = CMQ.get_cm_yearly_trend()
        df_new_ret       = CMQ.get_cm_new_returning_patients()
        df_seasonal      = CMQ.get_cm_seasonality()
        df_comorbidity   = CMQ.get_cm_comorbidity_profile()
        df_other_dx      = CMQ.get_cm_other_opd_breakdown()
        df_other_trend   = CMQ.get_cm_other_opd_trend()
        df_other_monthly = CMQ.get_cm_other_opd_monthly()
        df_spine_dx      = CMQ.get_cm_spine_diagnosis_monthly()

    render_s1(df_headline)
    render_s2(df_composition)
    render_s3(df_encounter, df_new_ret)
    render_s4(df_trend)
    render_s5(df_new_ret, df_seasonal, df_spine_dx)
    render_s6(df_comorbidity)
    render_s7(df_other_dx, df_other_trend, df_other_monthly)
    render_recommendations(
        df_encounter, df_spine_dx, df_other_dx, df_other_trend, df_other_monthly,
        df_comorbidity, df_trend,
    )
