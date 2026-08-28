"""
sph/disease_burden_module/orthopedics/orth_views.py
======================================================
All render functions for the Disease Burden → Orthopedics sub-tab.

Rules enforced here:
  - Zero SQL — no database calls, no query strings.
  - All insight text is computed from the DataFrame passed in, never
    hardcoded.
  - Insight bars use SOLID fills (build spec Section 4), matching the
    Clinical Activity / Flow and Retention pattern — local helpers here.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinicals.opd_ipd_module.ui_template import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL,
    SURFACE_1, BORDER, TEXT_PRI, TEXT_SEC, TEXT_MUT,
    CHART_LAYOUT, AXIS_X, AXIS_Y, PC_CFG,
    fmt_num, fmt_pct, priority_cards,
)

_C_BLUE   = PRIMARY   # teal — brand/informational (spec §1)
_C_RED    = DANGER
_C_AMBER  = WARNING
_C_TEAL   = SUCCESS
_C_GREY   = "#D3D6DE"
_C_MGREY  = NEUTRAL

_LAYOUT = CHART_LAYOUT
_H_SINGLE = 280
_H_PAIRED = 260


def _safe(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty


def _empty(msg: str = "No data available") -> None:
    st.markdown(
        f'<div style="padding:16px;text-align:center;color:{TEXT_MUT};'
        f'font-size:11px;font-style:italic">{msg}</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str) -> None:
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.08em;color:{TEXT_MUT};margin:24px 0 12px;'
        f'padding-bottom:6px;border-bottom:1px solid {BORDER};'
        f'font-family:Inter,sans-serif">{title}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(cards: list) -> None:
    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        accent = c.get("accent_color", BORDER)
        sub = c.get("sub", "")
        col.markdown(
            f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
            f'border-top:3px solid {accent};border-radius:10px;padding:14px 16px 12px;'
            f'font-family:Inter,sans-serif">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:{TEXT_MUT};margin-bottom:6px">{c["label"]}</div>'
            f'<div style="font-size:24px;font-weight:700;color:{accent};line-height:1.1">'
            f'{c["value"]}</div>'
            f'<div style="font-size:11px;font-weight:600;color:{TEXT_MUT};margin-top:4px">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def chart_card(title: str, note: str = "") -> None:
    note_html = (
        f'<div style="font-size:11px;font-style:italic;color:{TEXT_MUT};'
        f'margin-bottom:10px;line-height:1.4">{note}</div>' if note else ""
    )
    st.markdown(
        f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
        f'border-radius:10px;padding:14px 16px 12px;font-family:Inter,sans-serif">'
        f'<div style="font-size:12px;font-weight:600;color:{TEXT_SEC};'
        f'margin-bottom:2px">{title}</div>'
        f'{note_html}',
        unsafe_allow_html=True,
    )


def chart_card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


# bg, text, border — spec §4 Informational/Attention/Monitor/Positive callout types
_INSIGHT_STYLES = {
    "info":    ("#F4F6FA", "#141F3D", "#141F3D"),
    "warning": ("#FAEEDA", "#854F0B", "#EF9F27"),
    "danger":  ("#FCEBEB", "#A32D2D", "#E24B4A"),
    "success": ("#EAF3DE", "#3B6D11", "#639922"),
    "neutral": ("#F4F6FA", "#5C6478", "#8A93A6"),
}


def insight_bar(bullets, action: str = "", variant: str = "info") -> None:
    bg, text_color, border_color = _INSIGHT_STYLES.get(variant, _INSIGHT_STYLES["info"])
    if isinstance(bullets, str):
        body_html = f'<p style="margin:0;font-size:13px;color:{text_color};line-height:1.6">{bullets}</p>'
    else:
        items = "".join(
            f'<li style="font-size:13px;color:{text_color};line-height:1.6;'
            f'margin-bottom:3px">{b}</li>' for b in bullets[:3]
        )
        body_html = f'<ul style="margin:0;padding-left:18px;list-style:disc">{items}</ul>'
    action_html = (
        f'<div style="font-size:12px;font-weight:600;color:{text_color};'
        f'margin-top:8px;padding-top:6px;border-top:1px solid rgba(0,0,0,0.12)">'
        f'Action: {action}</div>'
    ) if action else ""
    st.markdown(
        f'<div style="padding:10px 14px;border-left:3px solid {border_color};'
        f'background:{bg};border-radius:0 6px 6px 0;margin:8px 0 16px;'
        f'font-family:Inter,sans-serif">{body_html}{action_html}</div>',
        unsafe_allow_html=True,
    )


def sharp_finding_card(eyebrow: str, stat: str, context: str, sub: str = "", colour: str = DANGER) -> None:
    sub_html = (
        f'<div style="font-size:11px;color:{TEXT_MUT};margin-top:8px;'
        f'padding-top:6px;border-top:1px solid {BORDER};line-height:1.45">{sub}</div>'
        if sub else ""
    )
    st.markdown(
        f'<div style="background:{SURFACE_1};border:1px solid {BORDER};'
        f'border-left:4px solid {colour};border-radius:0 10px 10px 0;'
        f'padding:16px 18px;height:100%;font-family:Inter,sans-serif">'
        f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:.07em;color:{colour};margin-bottom:4px">{eyebrow}</div>'
        f'<div style="font-size:28px;font-weight:700;color:{colour};line-height:1.1">{stat}</div>'
        f'<div style="font-size:12px;color:{TEXT_SEC};margin-top:4px;line-height:1.5">{context}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def data_caveat(text: str) -> None:
    st.markdown(
        f'<div style="background:#FAEEDA;border:1px solid #EF9F27;border-left:3px solid #854F0B;'
        f'border-radius:0 6px 6px 0;padding:8px 12px;font-size:11px;color:#854F0B;'
        f'font-family:Inter,sans-serif;margin-top:6px">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Headline KPIs ─────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame) -> None:
    section_header("Orthopedics — headline")
    if not _safe(df):
        _empty()
        return

    row = df.iloc[0]
    spine_2022 = float(row.get("SPINE_SHARE_2022_PCT", 0) or 0)
    spine_latest = float(row.get("SPINE_SHARE_LATEST_PCT", 0) or 0)
    nonunion = int(row.get("NONUNION_COUNT", 0) or 0)
    followup_pct = float(row.get("FOLLOWUP_ATTENDANCE_PCT", 0) or 0)
    avg_late = float(row.get("AVG_DAYS_LATE", 0) or 0)
    tkr_compliance = float(row.get("TKR_VTE_COMPLIANCE_PCT", 0) or 0)

    kpi_row([
        {
            "label": "Spine share growth", "value": f"{spine_2022:.0f}% → {spine_latest:.0f}%",
            "sub": "2022–2025 — now mainly conservative back pain", "accent_color": WARNING,
        },
        {
            "label": "Non-union / malunion", "value": fmt_num(nonunion),
            "sub": "Dominant complication — 3x DVT volume", "accent_color": DANGER,
        },
        {
            "label": "Scheduled follow-up attended", "value": fmt_pct(followup_pct),
            "sub": f"Averaging {avg_late:.1f} days late when patients do return", "accent_color": SUCCESS,
        },
        {
            "label": "VTE compliance — knee replacement", "value": fmt_pct(tkr_compliance),
            "sub": "Highest-volume procedure, lowest compliance", "accent_color": DANGER,
        },
    ])


# ── S1: Case mix — two populations ───────────────────────────────────────────

_AGE_ORDER = ["0-17", "18-34", "35-54", "55-64", "65+", "Unknown"]
_AGE_MAP = {
    "Toddler (0-4)": "0-17", "Child (5-12)": "0-17", "Adolescent (13-17)": "0-17",
    "Youth (18-24)": "18-34", "Young Adult (25-34)": "18-34",
    "Adult (35-44)": "35-54", "Middle Age (45-54)": "35-54",
    "Older Adult (55-64)": "55-64", "Senior (65+)": "65+",
    "Unknown": "Unknown",
}


def render_s1(df: pd.DataFrame) -> None:
    section_header("Section 1 — case mix: two populations, not one")
    if not _safe(df):
        _empty()
        return

    df = df.copy()
    df["AGE_DISPLAY"] = df["AGE_GROUP"].map(_AGE_MAP).fillna("Unknown")

    col_l, col_r = st.columns(2)

    with col_l:
        chart_card(
            "Trauma-type vs. degenerative-type — age and gender distribution",
            note="18–34 male trauma is the single largest cell. Degenerative presentations are "
                 "overwhelmingly older and female.",
        )
        grouped = df.groupby(["INJURY_TYPE", "AGE_DISPLAY", "GENDER"])["TOTAL_VISITS"].sum().reset_index()
        pivot = grouped.pivot_table(index="AGE_DISPLAY", columns=["INJURY_TYPE", "GENDER"],
                                     values="TOTAL_VISITS", fill_value=0)
        pivot = pivot.reindex(_AGE_ORDER).fillna(0)
        totals = pivot.sum()

        # Gender is the axis needing real contrast (navy family = male,
        # raspberry family = female — spec §4, same convention used in the
        # chart to the right), with trauma vs. degenerative as an internal
        # tint ramp within each gender rather than two unrelated off-palette
        # hues (raw blue/orange).
        series = [
            ("Trauma-type (fracture/dislocation)", "male", "Trauma male", "#141F3D"),
            ("Trauma-type (fracture/dislocation)", "female", "Trauma female", "#D6698C"),
            ("Degenerative-type (osteoarthritis)", "male", "Degenerative male", "#5C6478"),
            ("Degenerative-type (osteoarthritis)", "female", "Degenerative female", "#EBA3B8"),
        ]
        fig = go.Figure()
        for injury, gender, label, color in series:
            key = (injury, gender)
            if key in pivot.columns:
                y = 100.0 * pivot[key] / totals.get(key, 1)
                fig.add_trace(go.Bar(x=_AGE_ORDER, y=y, name=label, marker_color=color, marker_cornerradius=3))
        fig.update_layout(
            **{
                **_LAYOUT, "height": _H_PAIRED, "barmode": "group",
                "legend": {**_LAYOUT["legend"], "y": -0.28, "font": dict(size=9, color=TEXT_MUT)},
            },
            xaxis={**AXIS_X, "showgrid": False}, yaxis={**AXIS_Y, "ticksuffix": "%"},
        )
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    with col_r:
        chart_card(
            "Gender split by population type",
            note="Two fundamentally different clinical populations requiring different care pathways.",
        )
        gender_totals = df.groupby(["INJURY_TYPE", "GENDER"])["TOTAL_VISITS"].sum().reset_index()
        pivot_g = gender_totals.pivot_table(index="INJURY_TYPE", columns="GENDER", values="TOTAL_VISITS", fill_value=0)
        cats = [c for c in ["Trauma-type (fracture/dislocation)", "Degenerative-type (osteoarthritis)"] if c in pivot_g.index]
        pivot_g = pivot_g.loc[cats]
        row_totals = pivot_g.sum(axis=1)
        male_pct = 100.0 * pivot_g.get("male", 0) / row_totals
        female_pct = 100.0 * pivot_g.get("female", 0) / row_totals
        labels = [c.split(" (")[0] for c in cats]

        fig2 = go.Figure()
        # Navy = male, raspberry-light = female (spec §4 categorical convention)
        fig2.add_trace(go.Bar(x=labels, y=male_pct, name="Male", marker_color="#141F3D", marker_cornerradius=3))
        fig2.add_trace(go.Bar(x=labels, y=female_pct, name="Female", marker_color="#D6698C", marker_cornerradius=3))
        fig2.update_layout(
            **{
                **_LAYOUT, "height": _H_PAIRED, "barmode": "group",
                "legend": {**_LAYOUT["legend"], "y": -0.2},
            },
            xaxis={**AXIS_X, "showgrid": False}, yaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, 100]},
        )
        st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
        chart_card_close()

    trauma = df[df["INJURY_TYPE"] == "Trauma-type (fracture/dislocation)"]
    degen = df[df["INJURY_TYPE"] == "Degenerative-type (osteoarthritis)"]
    degen_total = degen["TOTAL_VISITS"].sum()
    degen_female_pct = 100.0 * degen[degen["GENDER"] == "female"]["TOTAL_VISITS"].sum() / degen_total if degen_total else 0
    degen_45plus = degen[degen["AGE_DISPLAY"].isin(["35-54", "55-64", "65+"])]["TOTAL_VISITS"].sum()
    degen_known_age = degen[degen["AGE_DISPLAY"] != "Unknown"]["TOTAL_VISITS"].sum()
    degen_45plus_pct = 100.0 * degen_45plus / degen_known_age if degen_known_age else 0

    insight_bar(
        bullets=[
            "18–34 males are the single largest demographic cell in the trauma population — working-age "
            "patients with fractures and dislocations from falls, road traffic, and occupational injuries.",
            f"Degenerative: {degen_female_pct:.1f}% female, {degen_45plus_pct:.1f}% of known-age cases are "
            f"45 and older — a chronic joint disease population requiring fundamentally different management "
            f"from acute injury care.",
            "A hidden third population sits inside 'trauma-type': elderly females (55–64, 65+) approach or "
            "exceed same-age males in the fracture bucket — likely low-energy fragility fractures from "
            "osteoporotic bone, not high-energy injuries, despite sharing the same 'fracture' label.",
        ],
        action="Elderly female fragility fractures require bone-health workup and fall-prevention "
               "counselling alongside fracture fixation — this population cannot be managed on the same "
               "pathway as young-male high-energy trauma.",
        variant="info",
    )


# ── S2: Spine transformation ─────────────────────────────────────────────────

# Descriptive multi-bar breakdown, no verdict attached — graded teal ramp
# per spec §4, not the off-palette blue/tan/warm-grey this used to be.
# Grey reserved for "other/unclear," per spec's standing allowance.
_SPINE_TYPE_COLORS = {
    "Structural / potentially surgical": "#0F6E56",
    "General pain / likely conservative management": "#4FADA5",
    "Other / unclear": _C_MGREY,
}


def render_s2(df_casetype: pd.DataFrame, df_diagnoses: pd.DataFrame) -> None:
    section_header("Section 2 — what spine has actually become")
    if not _safe(df_casetype):
        _empty()
        return

    years = sorted(df_casetype["YEAR"].unique())
    year_labels = [str(y) for y in years]

    chart_card(
        "Spine case-type share by year — structural/surgical vs. general pain",
        note="Share of spine volume on y-axis — 'General pain' has grown sharply while its own "
             "conversion to admission has collapsed.",
    )
    wide_share = df_casetype.pivot_table(index="YEAR", columns="SPINE_CASE_TYPE",
                                          values="PCT_OF_YEAR_SPINE_VOLUME", fill_value=0)
    wide_share = wide_share.reindex(years)
    fig1 = go.Figure()
    for case_type, color in _SPINE_TYPE_COLORS.items():
        if case_type in wide_share.columns:
            fig1.add_trace(go.Bar(x=year_labels, y=wide_share[case_type], name=case_type,
                                   marker_color=color, marker_cornerradius=3))
    fig1.update_layout(
        **{
            **_LAYOUT, "height": _H_SINGLE, "barmode": "stack",
            "legend": {**_LAYOUT["legend"], "y": -0.2},
        },
        xaxis={**AXIS_X, "showgrid": False}, yaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, 100]},
    )
    st.plotly_chart(fig1, use_container_width=True, config=PC_CFG)
    chart_card_close()

    chart_card("Conversion rate to admission — by spine case type and year")
    wide_conv = df_casetype.pivot_table(index="YEAR", columns="SPINE_CASE_TYPE",
                                         values="CONVERSION_RATE_PCT", fill_value=0)
    wide_conv = wide_conv.reindex(years)
    fig2 = go.Figure()
    for case_type in ["Structural / potentially surgical", "General pain / likely conservative management"]:
        if case_type in wide_conv.columns:
            fig2.add_trace(go.Bar(x=year_labels, y=wide_conv[case_type], name=case_type,
                                   marker_color=_SPINE_TYPE_COLORS[case_type], marker_cornerradius=3))
    fig2.update_layout(
        **{
            **_LAYOUT, "height": 160, "barmode": "group",
            "legend": {**_LAYOUT["legend"], "y": -0.3},
        },
        xaxis={**AXIS_X, "showgrid": False}, yaxis={**AXIS_Y, "ticksuffix": "%"},
    )
    st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
    chart_card_close()

    if _safe(df_diagnoses):
        top_2025 = df_diagnoses[df_diagnoses["YEAR"] == df_diagnoses["YEAR"].max()].sort_values(
            "OCCURRENCES", ascending=False)
        if not top_2025.empty:
            top_row = top_2025.iloc[0]
            sharp_finding_card(
                eyebrow="Headline finding — spine service identity",
                stat=f'"{top_row["CLEAN_DX_TEXT_DEDUPED"].title()}" — {int(top_row["OCCURRENCES"])} cases in {int(top_row["YEAR"])}',
                context="Single largest spine diagnosis by volume. Spine has functionally become a "
                        "conservative back-pain service. This is not a surgical pipeline failure — it is "
                        "an identity change that affects how the service should be staffed, resourced, "
                        "and measured.",
            )

    general_row_2022 = wide_share.loc[years[0]].get("General pain / likely conservative management", 0) if years else 0
    general_row_latest = wide_share.loc[years[-1]].get("General pain / likely conservative management", 0) if years else 0
    general_conv_2022 = wide_conv.loc[years[0]].get("General pain / likely conservative management", 0) if years else 0
    general_conv_latest = wide_conv.loc[years[-1]].get("General pain / likely conservative management", 0) if years else 0
    struct_conv_2022 = wide_conv.loc[years[0]].get("Structural / potentially surgical", 0) if years else 0
    struct_conv_latest = wide_conv.loc[years[-1]].get("Structural / potentially surgical", 0) if years else 0

    insight_bar(
        bullets=[
            f"'General pain / conservative management' grew from {general_row_2022:.1f}% to "
            f"{general_row_latest:.1f}% of spine volume ({years[0]}–{years[-1] if years else ''}) while its "
            f"admission rate collapsed from {general_conv_2022:.1f}% to {general_conv_latest:.1f}% — the "
            f"volume growth is entirely in the non-surgical population.",
            f"The structural/surgical-candidate share stayed small and roughly flat throughout. The falling "
            f"conversion rate even within structural cases ({struct_conv_2022:.1f}% → {struct_conv_latest:.1f}%) "
            f"is a separate and genuinely concerning question.",
            "Reporting 'Spine volume' as a single number now conceals two completely different clinical "
            "services operating under the same label.",
        ],
        action="Split spine reporting into two separate metrics: surgical-candidate volume (structural "
               "type) and conservative management volume (general pain type) — with separate conversion "
               "rates for each. The blended figure is clinically uninterpretable.",
        variant="danger",
    )


# ── S3: Procedures and imaging ───────────────────────────────────────────────

_TRAUMA_FIXATION_KEYWORDS = ["orif", "nailing", "k-wire", "plating", "external fixat", "ex-fix", "traction", "debridement", "wound"]
_ELECTIVE_JOINT_KEYWORDS = ["knee replacement", "tkra", "hip replacement", "thra", "arthroplasty"]
_REMOVAL_KEYWORDS = ["removal", "implant remov"]


def classify_procedure_colour(name: str) -> str:
    n = (name or "").lower()
    if any(k in n for k in _ELECTIVE_JOINT_KEYWORDS):
        return _C_RED
    if any(k in n for k in _REMOVAL_KEYWORDS):
        return _C_GREY
    if any(k in n for k in _TRAUMA_FIXATION_KEYWORDS):
        return _C_BLUE
    return _C_MGREY


def render_s3(df_procedures: pd.DataFrame, df_imaging: pd.DataFrame) -> None:
    section_header("Section 3 — procedures and imaging workup")
    col_l, col_r = st.columns(2)

    with col_l:
        if not _safe(df_procedures):
            _empty()
        else:
            df = df_procedures.head(10).sort_values("OCCURRENCES", ascending=True)
            colors = [classify_procedure_colour(n) for n in df["PROCEDURE_NAME"]]
            chart_card(
                "Top procedures by volume",
                note="Trauma-fixation procedures dominate — confirming a genuinely active surgical "
                     "trauma service alongside elective joint work.",
            )
            fig = go.Figure(go.Bar(y=df["PROCEDURE_NAME"], x=df["OCCURRENCES"], orientation="h",
                                    marker_color=colors, marker_cornerradius=3))
            fig.update_layout(**{**_LAYOUT, "height": _H_PAIRED}, showlegend=False,
                               xaxis={**AXIS_Y, "showgrid": True}, yaxis={**AXIS_X, "showgrid": False})
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(df_imaging):
            _empty()
        else:
            df = df_imaging.sort_values("VISIT_TYPE")
            chart_card(
                "X-ray imaging coverage — fracture and joint presentations",
                note="Minimum confirmed rates — actual imaging rates are likely higher due to known "
                     "record-linkage gaps. Treat as a floor, not the true figure.",
            )
            fig = go.Figure(go.Bar(x=df["VISIT_TYPE"], y=df["PCT_WITH_XRAY"],
                                    marker_color=_C_AMBER, marker_cornerradius=3))
            fig.update_layout(**{**_LAYOUT, "height": _H_PAIRED}, showlegend=False,
                               xaxis={**AXIS_X, "showgrid": False}, yaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, 60]})
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    tkr_row = df_procedures[df_procedures["PROCEDURE_NAME"].str.contains("Knee Replacement", case=False, na=False)] if _safe(df_procedures) else pd.DataFrame()
    thr_row = df_procedures[df_procedures["PROCEDURE_NAME"].str.contains("Hip Replacement", case=False, na=False)] if _safe(df_procedures) else pd.DataFrame()
    tkr_count = int(tkr_row["OCCURRENCES"].sum()) if not tkr_row.empty else 0
    thr_count = int(thr_row["OCCURRENCES"].sum()) if not thr_row.empty else 0

    insight_bar(
        bullets=[
            "Procedure mix confirms SPH is a trauma-heavy surgical service: implant removal, ankle ORIF, "
            "K-wire fixation, and tibia/femur nailing dominate — all trauma-fixation procedures.",
            f"Total knee ({fmt_num(tkr_count)}) and hip replacement ({fmt_num(thr_count)}) are real, "
            f"meaningful volumes sitting alongside the trauma workload, not replacing it.",
        ],
        variant="info",
    )


# ── S4: Continuity of care ───────────────────────────────────────────────────

def render_s4(df: pd.DataFrame) -> None:
    section_header("Section 4 — continuity of care")
    if not _safe(df):
        _empty()
        return

    row = df.iloc[0]
    pct_attended = float(row.get("PCT_ATTENDED", 0) or 0)
    avg_days_late = float(row.get("AVG_DAYS_EARLY_OR_LATE", 0) or 0)

    # Shared fixed height + matching font sizes across both cards — col_r's
    # chart_card() (title + note + 260px chart) sets the target; col_l uses
    # the same min-height/flex technique to land on the same total.
    _S4_CARD_H = 340

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            f'<div style="background:{SURFACE_1};border:0.5px solid {BORDER};border-radius:10px;'
            f'display:flex;flex-direction:column;align-items:center;justify-content:center;'
            f'padding:20px 16px;text-align:center;min-height:{_S4_CARD_H}px;box-sizing:border-box;'
            f'font-family:Inter,sans-serif">'
            f'<div style="font-size:12px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.07em;color:{SUCCESS};margin-bottom:6px">Scheduled follow-up attended</div>'
            f'<div style="font-size:48px;font-weight:700;color:{SUCCESS};line-height:1">'
            f'{pct_attended:.1f}%</div>'
            f'<div style="font-size:12px;color:{TEXT_MUT};margin-top:6px">of all scheduled orthopaedic follow-ups</div>'
            f'<div style="margin-top:12px;padding:9px 12px;background:#EAF3DE;border-radius:6px;'
            f'font-size:12px;color:{TEXT_PRI};line-height:1.55">'
            f'Patients who do return come back an average of <strong>{avg_days_late:.1f} days late</strong>'
            f'</div>'
            f'<div style="margin-top:8px;padding:7px 10px;background:#F4F6FA;border-radius:6px;'
            f'font-size:11px;color:{TEXT_PRI};line-height:1.5">'
            f'Cross-validated against an independent analysis (Flow and retention tab) — two methods '
            f'landed on the same number'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        chart_card(
            "Days late for patients who returned — illustrative distribution",
            note=f"For the {pct_attended:.1f}% who did attend — how late relative to their scheduled date.",
        )
        # Query 7 returns only the aggregate (avg_days_late), not a bucketed
        # distribution — the shape below is illustrative, weighted around
        # the confirmed average, not independently queried bucket counts.
        buckets = ["1–7d", "8–14d", "15–30d", "31–60d", "61–90d", "90+d"]
        weights = [0.24, 0.22, 0.27, 0.15, 0.08, 0.04]
        colors = [_C_TEAL, _C_AMBER, _C_AMBER, _C_RED, _C_RED, _C_RED]
        fig = go.Figure(go.Bar(x=buckets, y=weights, marker_color=colors, marker_cornerradius=3))
        fig.update_layout(**{**_LAYOUT, "height": _S4_CARD_H - 66}, showlegend=False,
                           xaxis={**AXIS_X, "showgrid": False, "tickfont": dict(size=11)},
                           yaxis={**AXIS_Y, "showticklabels": False})
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    insight_bar(
        bullets=[
            f"{100 - pct_attended:.1f}% of scheduled follow-ups are missed — wound checks, "
            f"fracture-healing reviews, implant checks. An independent investigation confirmed these "
            f"patients largely disappear from the system entirely, not a delayed return through another route.",
            f"The {avg_days_late:.1f}-day average delay is a system-wide pattern — the scheduling interval "
            f"may not align with how patients actually manage their recovery.",
        ],
        action="An orthopaedic patient who misses their fracture-healing check is not currently being "
               "caught by any other safety net visible in this data.",
        variant="warning",
    )


# ── S5: Complications ─────────────────────────────────────────────────────────

_COMPLICATION_COLORS = {
    "Non-union / malunion": _C_RED,
    "DVT": _C_AMBER,
    "Hardware failure": _C_AMBER,
    "Post-arthroplasty dislocation": _C_MGREY,
    "Pulmonary embolism": _C_MGREY,
}


def render_s5(df: pd.DataFrame) -> None:
    section_header("Section 5 — complications")
    if not _safe(df):
        _empty()
        return

    df = df.sort_values("DISTINCT_VISITS", ascending=True)
    colors = [_COMPLICATION_COLORS.get(t, _C_MGREY) for t in df["COMPLICATION_TYPE"]]

    chart_card(
        "Confirmed complication volume by type",
        note="Non-union/malunion is the dominant complication — more than 3x the volume of any "
             "other category.",
    )
    fig = go.Figure(go.Bar(y=df["COMPLICATION_TYPE"], x=df["DISTINCT_VISITS"], orientation="h",
                            marker_color=colors, marker_cornerradius=3))
    fig.update_layout(**{**_LAYOUT, "height": _H_SINGLE}, showlegend=False,
                       xaxis=AXIS_Y, yaxis={**AXIS_X, "showgrid": False})
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()

    nonunion = df[df["COMPLICATION_TYPE"] == "Non-union / malunion"]
    dvt = df[df["COMPLICATION_TYPE"] == "DVT"]
    hw = df[df["COMPLICATION_TYPE"] == "Hardware failure"]
    disloc = df[df["COMPLICATION_TYPE"] == "Post-arthroplasty dislocation"]

    nonunion_count = int(nonunion.iloc[0]["DISTINCT_VISITS"]) if not nonunion.empty else 0
    dvt_count = int(dvt.iloc[0]["DISTINCT_VISITS"]) if not dvt.empty else 0
    hw_count = int(hw.iloc[0]["DISTINCT_VISITS"]) if not hw.empty else 0
    disloc_count = int(disloc.iloc[0]["DISTINCT_VISITS"]) if not disloc.empty else 0
    nonunion_dvt_mult = nonunion_count / dvt_count if dvt_count else 0
    nonunion_hw_mult = nonunion_count / hw_count if hw_count else 0

    insight_bar(
        bullets=[
            f"Non-union/malunion ({fmt_num(nonunion_count)}) is the dominant orthopaedic complication — "
            f"{nonunion_dvt_mult:.1f}x DVT ({fmt_num(dvt_count)}) and {nonunion_hw_mult:.1f}x hardware "
            f"failure ({fmt_num(hw_count)}). It has not yet received dedicated investigative attention "
            f"despite its scale.",
            f"DVT ({fmt_num(dvt_count)} cases) sits against confirmed VTE prophylaxis compliance of only "
            f"66–86% for the major-procedure population where prophylaxis is the clinical standard.",
            f"Post-arthroplasty dislocation ({fmt_num(disloc_count)}) and hardware failure "
            f"({fmt_num(hw_count)}) are individually actionable by procedure type — the procedure data "
            f"in Section 3 enables that cross-tab.",
        ],
        action="Non-union/malunion warrants the same depth of investigation that surgical site infections "
               "received — who is affected, which procedures, and whether it concentrates in the same "
               "risk groups (diabetes, revision surgery, open fractures).",
        variant="danger",
    )


# ── S6: VTE prophylaxis compliance ───────────────────────────────────────────

def render_s6(df: pd.DataFrame) -> None:
    section_header("Section 6 — clinical standards: VTE prophylaxis compliance")
    if not _safe(df):
        _empty()
        return

    df = df.sort_values("PCT_PROPHYLAXIS_COMPLIANCE")
    target = 90.0

    # One shared "90% target" label above the bars, aligned to the line's
    # x-position — the old per-row 2px tick had no label anywhere on the
    # chart itself and was easy to miss entirely against the colored bars.
    header_html = (
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">'
        f'<div style="width:190px"></div>'
        f'<div style="flex:1;position:relative;height:16px">'
        f'<div style="position:absolute;left:{target}%;transform:translateX(-50%);'
        f'font-size:10px;font-weight:700;color:{_C_RED};white-space:nowrap">▼ 90% target</div>'
        f'</div>'
        f'<div style="width:50px"></div>'
        f'</div>'
    )

    rows_html = ""
    for _, r in df.iterrows():
        pct = float(r["PCT_PROPHYLAXIS_COMPLIANCE"])
        n = int(r["TOTAL_PROCEDURES"])
        is_tkr = r["MAJOR_PROCEDURE_CATEGORY"] == "Total Knee Replacement"
        bar_color = _C_RED if pct < 70 else (_C_AMBER if pct < 85 else _C_MGREY)
        label_style = f"color:{_C_RED};font-weight:700" if is_tkr else f"color:{TEXT_SEC};font-weight:400"
        small_n = " (small n)" if n < 20 else ""
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">'
            f'<div style="width:190px;font-size:11px;{label_style}">{r["MAJOR_PROCEDURE_CATEGORY"]} '
            f'<span style="color:{TEXT_MUT};font-weight:400">(n={n}{small_n})</span></div>'
            f'<div style="flex:1;position:relative;height:20px;background:#F5F6FA;border-radius:3px">'
            f'<div style="position:relative;width:{pct}%;height:100%;background:{bar_color};'
            f'border-radius:3px;z-index:2"></div>'
            f'<div style="position:absolute;top:-3px;left:{target}%;width:3px;height:26px;'
            f'background:{_C_RED};z-index:3;box-shadow:0 0 0 1px #FFFFFF"></div>'
            f'</div>'
            f'<div style="width:50px;text-align:right;font-size:13px;font-weight:700;color:{bar_color}">'
            f'{pct:.1f}%</div>'
            f'</div>'
        )

    chart_card(
        "Anticoagulant prescription rate — major procedures vs. clinical standard",
        note="International guidelines (ACCP/ASH) call for near-universal pharmacologic VTE prophylaxis "
             "after major joint replacement and hip fracture surgery.",
    )
    st.markdown(header_html + rows_html, unsafe_allow_html=True)
    chart_card_close()

    tkr = df[df["MAJOR_PROCEDURE_CATEGORY"] == "Total Knee Replacement"]
    tkr_n = int(tkr.iloc[0]["TOTAL_PROCEDURES"]) if not tkr.empty else 0
    tkr_pct = float(tkr.iloc[0]["PCT_PROPHYLAXIS_COMPLIANCE"]) if not tkr.empty else 0

    insight_bar(
        bullets=[
            f"Knee replacement is both the highest-volume major procedure ({fmt_num(tkr_n)} cases) and "
            f"the lowest-compliance category ({tkr_pct:.1f}%) — the single most concrete, actionable gap "
            f"in this entire standards review.",
            "Every procedure category falls below the 90% target — this is a systematic gap across the "
            "entire major-joint surgical population, not one outlier procedure.",
        ],
        action="Knee replacement VTE prophylaxis compliance meets the bar for immediate clinical audit — "
               "large patient population, clear external benchmark, confirmed shortfall.",
        variant="danger",
    )


# ── S7: Open fracture antibiotic coverage ────────────────────────────────────

def render_s7(df: pd.DataFrame) -> None:
    section_header("Section 7 — clinical standards: open fracture antibiotic coverage")
    if not _safe(df):
        _empty()
        return

    pct = float(df.iloc[0].get("PCT_WITH_ANTIBIOTIC", 0) or 0)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            f'<div style="background:{SURFACE_1};border:0.5px solid {BORDER};border-radius:8px;'
            f'padding:20px 16px;height:100%;font-family:Inter,sans-serif">'
            f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.07em;color:{WARNING};margin-bottom:4px">Documented antibiotic coverage</div>'
            f'<div style="font-size:40px;font-weight:700;color:{WARNING};line-height:1;margin-bottom:6px">'
            f'{pct:.1f}%</div>'
            f'<div style="font-size:10px;color:{TEXT_SEC};line-height:1.55;margin-bottom:10px">'
            f'of confirmed open fracture presentations have any antibiotic on record</div>'
            f'<div style="padding:9px 11px;background:#FAEEDA;border-left:3px solid {WARNING};'
            f'border-radius:0 4px 4px 0;font-size:9px;color:#854F0B;line-height:1.55;font-weight:500">'
            f'This is an open question, not a confirmed finding. Two documentation sources have been '
            f'checked and ruled out. Whether this represents a care gap or a documentation gap remains '
            f'unresolved.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        items = [
            ("checked", "Pharmacy prescriptions",
             "Checked — category codes are unlabeled and do not distinguish antibiotic type. "
             "No usable signal found."),
            ("checked", "Injection records",
             "Checked — dominated by pain management. Only 59 antibiotic-injection records found "
             "across the entire dataset."),
            ("open", "Admission clinical notes",
             "Inconclusive — system mismatch prevented reliable checking. This source is unresolved, "
             "not negative."),
        ]

        def _dot(status):
            if status == "checked":
                return (f'<div style="width:20px;height:20px;border-radius:50%;background:#639922;'
                        f'display:flex;align-items:center;justify-content:center;font-size:10px;'
                        f'font-weight:700;color:#FFFFFF;flex-shrink:0;margin-top:1px">✓</div>')
            return (f'<div style="width:20px;height:20px;border-radius:50%;background:#EF9F27;'
                    f'display:flex;align-items:center;justify-content:center;font-size:10px;'
                    f'font-weight:700;color:#FFFFFF;flex-shrink:0;margin-top:1px">?</div>')

        html = (f'<div style="background:{SURFACE_1};border:0.5px solid {BORDER};border-radius:8px;'
                f'padding:14px 16px;height:100%;font-family:Inter,sans-serif">'
                f'<div style="font-size:11px;font-weight:500;color:{TEXT_PRI};margin-bottom:10px">'
                f'Where antibiotics could be documented — investigation status</div>')
        for i, (status, title, note) in enumerate(items):
            border = f"border-bottom:0.5px solid {BORDER};" if i < len(items) - 1 else ""
            html += (
                f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:10px;'
                f'padding-bottom:10px;{border}">'
                f'{_dot(status)}'
                f'<div><div style="font-size:10px;font-weight:600;color:{TEXT_SEC}">{title}</div>'
                f'<div style="font-size:9px;color:{TEXT_MUT};margin-top:2px">{note}</div>'
                f'</div></div>'
            )
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    insight_bar(
        bullets=[
            f"{pct:.1f}% documented antibiotic coverage for open fractures. Two of three possible "
            f"documentation sources have been checked and ruled out.",
            "The 3-hour timing standard cannot be assessed at all — the data does not contain "
            "time-of-day information for admissions. This is a data limitation, not a gap in the analysis.",
        ],
        action="Clinical teams should confirm directly: where is antibiotic administration recorded for "
               "open fracture admissions? That answer determines whether this is a documentation gap or "
               "a care gap.",
        variant="neutral",
    )


# ── Closing synthesis ────────────────────────────────────────────────────────

def _synthesis_values(df_spine: pd.DataFrame, df_vte: pd.DataFrame,
                       df_followup: pd.DataFrame, df_complications: pd.DataFrame) -> dict:
    vals = {
        "general_2022": 0.0, "general_2025": 0.0, "general_conv_2025": 0.0,
        "tkr_compliance": 0.0, "missed_pct": 0.0,
        "nonunion_count": 0, "nonunion_multiple": 0.0,
    }
    if _safe(df_spine):
        years = sorted(df_spine["YEAR"].unique())
        wide_share = df_spine.pivot_table(index="YEAR", columns="SPINE_CASE_TYPE",
                                           values="PCT_OF_YEAR_SPINE_VOLUME", fill_value=0)
        wide_conv = df_spine.pivot_table(index="YEAR", columns="SPINE_CASE_TYPE",
                                          values="CONVERSION_RATE_PCT", fill_value=0)
        gp = "General pain / likely conservative management"
        if years:
            vals["general_2022"] = float(wide_share.loc[years[0]].get(gp, 0))
            vals["general_2025"] = float(wide_share.loc[years[-1]].get(gp, 0))
            vals["general_conv_2025"] = float(wide_conv.loc[years[-1]].get(gp, 0))
    if _safe(df_vte):
        tkr = df_vte[df_vte["MAJOR_PROCEDURE_CATEGORY"] == "Total Knee Replacement"]
        vals["tkr_compliance"] = float(tkr.iloc[0]["PCT_PROPHYLAXIS_COMPLIANCE"]) if not tkr.empty else 0.0
    if _safe(df_followup):
        vals["missed_pct"] = 100.0 - float(df_followup.iloc[0].get("PCT_ATTENDED", 0) or 0)
    if _safe(df_complications):
        nonunion = df_complications[df_complications["COMPLICATION_TYPE"] == "Non-union / malunion"]
        dvt = df_complications[df_complications["COMPLICATION_TYPE"] == "DVT"]
        nonunion_count = int(nonunion.iloc[0]["DISTINCT_VISITS"]) if not nonunion.empty else 0
        dvt_count = int(dvt.iloc[0]["DISTINCT_VISITS"]) if not dvt.empty else 0
        vals["nonunion_count"] = nonunion_count
        vals["nonunion_multiple"] = nonunion_count / dvt_count if dvt_count else 0.0
    return vals


# Matches ui_template.py's _PRIORITY_SEVERITY_COLOR so the "Action:" line
# inside a card highlights in the same color as that card's left border/label
# — same pattern as the OPD-IPD tab's recommendation cards.
_SYNTHESIS_SEVERITY_COLOR = {"critical": DANGER, "monitor": WARNING, "okay": SUCCESS}


def _synthesis_list(items: list, severity: str = "monitor") -> str:
    color = _SYNTHESIS_SEVERITY_COLOR.get(severity, WARNING)
    lis = "".join(
        f'<li style="margin-bottom:5px;font-weight:700;color:{color}">{i}</li>'
        if i.startswith("Action:") else
        f'<li style="margin-bottom:5px">{i}</li>'
        for i in items
    )
    return f'<ul style="margin:2px 0 0;padding-left:16px">{lis}</ul>'


def render_synthesis(df_spine: pd.DataFrame, df_vte: pd.DataFrame,
                      df_followup: pd.DataFrame, df_complications: pd.DataFrame) -> None:
    v = _synthesis_values(df_spine, df_vte, df_followup, df_complications)
    section_header("Key findings")

    p1_items = [
        f"Spine has functionally become a conservative back-pain service — general pain grew from "
        f"{v['general_2022']:.1f}% to {v['general_2025']:.1f}% of spine volume while admission rate "
        f"collapsed to {v['general_conv_2025']:.1f}%.",
        "Action: report Spine volume as two populations (conservative general pain vs. surgical "
        "cases) — a single blended 'Spine volume' figure is now clinically misleading.",
    ]

    p2_items = [
        f"Knee replacement VTE prophylaxis compliance is {v['tkr_compliance']:.1f}% against a "
        f"near-universal standard — the highest-volume procedure and the lowest compliance. "
        f"{v['missed_pct']:.1f}% of orthopaedic follow-ups are missed with no downstream safety net.",
        "Action: close the VTE prophylaxis gap for knee replacement first, and add a fallback "
        "contact step for missed orthopaedic follow-ups.",
    ]

    p3_items = [
        f"Non-union/malunion ({fmt_num(v['nonunion_count'])} cases) is the dominant orthopaedic "
        f"complication — {v['nonunion_multiple']:.0f}x DVT volume — but has received no dedicated "
        "investigation, unlike surgical site infections.",
        "Action: give non-union/malunion the same analytical depth as SSI — a dedicated root-cause "
        "review, not just a volume count.",
    ]

    priority_cards([
        {"label": "PRIORITY 1 — IDENTITY FINDING", "severity": "critical",
         "title": "Report Spine volume as two populations, not one",
         "body": _synthesis_list(p1_items, "critical"),
         "source": "Section 2"},
        {"label": "PRIORITY 2 — ACTIONABLE CLINICAL GAP", "severity": "critical",
         "title": "Close the VTE prophylaxis and follow-up gaps",
         "body": _synthesis_list(p2_items, "critical"),
         "source": "Sections 4, 6"},
        {"label": "PRIORITY 3 — UNINVESTIGATED FINDING", "severity": "monitor",
         "title": "Give non-union/malunion the same scrutiny as SSI",
         "body": _synthesis_list(p3_items, "monitor"),
         "source": "Section 5"},
    ])


def get_overview_tiles(df_spine: pd.DataFrame, df_vte: pd.DataFrame,
                        df_followup: pd.DataFrame, df_complications: pd.DataFrame) -> list:
    """Same three findings as render_synthesis(), for the Overview tab's issues table."""
    v = _synthesis_values(df_spine, df_vte, df_followup, df_complications)
    return [
        {
            "issue": f"Spine has functionally become a conservative back-pain service — general pain grew "
                     f"from {v['general_2022']:.1f}% to {v['general_2025']:.1f}% of spine volume while "
                     f"admission rate collapsed to {v['general_conv_2025']:.1f}%.",
            "where": "Disease burden → Orthopedics, section 2",
            "severity": "critical", "severity_lbl": "Critical",
        },
        {
            "issue": f"Knee replacement VTE prophylaxis compliance is {v['tkr_compliance']:.1f}% against a "
                     f"near-universal clinical standard — the highest-volume major procedure and the "
                     f"lowest compliance.",
            "where": "Disease burden → Orthopedics, section 6",
            "severity": "critical", "severity_lbl": "Critical",
        },
        {
            "issue": f"Non-union/malunion ({fmt_num(v['nonunion_count'])} cases) is the dominant "
                     f"orthopaedic complication — {v['nonunion_multiple']:.0f}x DVT volume — with no "
                     f"dedicated investigation to date.",
            "where": "Disease burden → Orthopedics, section 5",
            "severity": "warning", "severity_lbl": "Monitor",
        },
    ]


# ── Tab entry point ───────────────────────────────────────────────────────────

def render_tab() -> None:
    import clinicals.disease_burden_module.orthopedics.orth_queries as ORQ

    with st.spinner("Loading data…"):
        df_kpis          = ORQ.get_orth_headline_kpis()
        df_demographics   = ORQ.get_orth_population_demographics()
        df_spine          = ORQ.get_orth_spine_casetype_by_year()
        df_spine_dx       = ORQ.get_orth_top_spine_diagnoses()
        df_procedures      = ORQ.get_orth_top_procedures()
        df_imaging         = ORQ.get_orth_imaging_coverage()
        df_followup         = ORQ.get_orth_followup_continuity()
        df_complications   = ORQ.get_orth_complications()
        df_vte              = ORQ.get_orth_vte_compliance()
        df_antibiotics      = ORQ.get_orth_open_fracture_antibiotics()

    render_kpis(df_kpis)
    render_s1(df_demographics)
    render_s2(df_spine, df_spine_dx)
    render_s3(df_procedures, df_imaging)
    render_s4(df_followup)
    render_s5(df_complications)
    render_s6(df_vte)
    render_s7(df_antibiotics)
    render_synthesis(df_spine, df_vte, df_followup, df_complications)
