"""
sph/flow_retention_module/fr_views.py
========================================
Flow and Retention tab, rebuilt to match Flow_Retention_Template_B_Build_Spec.md
against the verified query set in fr_queries.py (6-segment taxonomy:
Core Orthopedics: General, Spine-conservative, Spine-structural,
ANC / Routine Pregnancy, High-Risk Pregnancy, Fibroids-conservative).

Rules enforced here:
  - Zero SQL — every DataFrame is fetched once in render_tab() and passed
    down to section renderers.
  - No hardcoded numbers in any insight bar — every number is an
    f-string interpolation from a DataFrame.
  - Insight bar variants: "danger" | "warning" | "info" | "success"
    (build-spec names), mapped locally to ui_template.insight_bar's real
    variant set (danger/warning/primary/success) via _insight().
  - Pattern D (sharp-finding card) used exactly once — Section 5.
  - Every paired chart row (Pattern B) uses identical fixed heights on
    both sides; the insight bar always sits below the full row, never
    inside a column.
"""

import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sph.clinicals.opd_ipd_module.ui_template import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL,
    SURFACE_1, BORDER, TEXT_PRI, TEXT_SEC, TEXT_MUT,
    CHART_LAYOUT, AXIS_X, AXIS_Y, PC_CFG,
    fmt_num, fmt_pct, kpi_row, insight_bar as _ui_insight_bar,
    sharp_finding_card, priority_cards,
)

_C_TEAL   = PRIMARY
_C_NAVY   = "#141F3D"
_C_CORAL  = DANGER
_C_AMBER  = WARNING
_C_NEUTRAL = NEUTRAL
_C_NEUTRAL2 = "#D3D6DE"
_C_LIGHT_BLUE = "#8FCFC8"
_C_RASPBERRY = "#C13868"

# Distinct, on-palette colors for a single donut's slices — a same-hue
# gradient ramp reads as "which slice is biggest" but not "which slice is
# which" at a glance, so these are picked for contrast against each other,
# not for belonging to one category.
_CATEGORICAL_PALETTE = [
    _C_CORAL, _C_TEAL, _C_AMBER, _C_NAVY, _C_RASPBERRY, "#5DCAA5", _C_NEUTRAL2, _C_NEUTRAL,
]

_LAYOUT = CHART_LAYOUT
_H_SINGLE = 280
_H_PAIRED = 260

_SEG_SHORT = {
    "Core Orthopedics: General": "Ortho General",
    "Core Orthopedics: Spine and Back Pain Care": "Spine and Back Pain Care",
    "Spine-conservative": "Spine (conservative)",
    "Spine-structural": "Spine (structural)",
    "ANC / Routine Pregnancy": "ANC",
    "High-Risk Pregnancy": "High-Risk Pregnancy",
    "Fibroids-conservative": "Fibroids",
    "Core General Surgery": "Gen Surgery",
    "ANC": "ANC",
    "Maternal Health": "Maternal Health",
}

_VARIANT_MAP = {"danger": "danger", "warning": "warning", "info": "primary", "success": "success"}


def _short(seg) -> str:
    if not seg:
        return seg
    return _SEG_SHORT.get(seg, seg)


def _age_sort_key(age_group: str):
    """Sorts age-group labels youngest-to-oldest regardless of the exact
    bucket naming convention used by the source query, with 'Unknown'
    always last."""
    if not age_group or str(age_group).strip().lower() == "unknown":
        return (1, 0)
    import re
    m = re.search(r"\d+", str(age_group))
    return (0, int(m.group()) if m else 999)


def _safe(df: pd.DataFrame) -> bool:
    return df is not None and not df.empty


def _text_color_for(hex_color: str) -> str:
    """White text on a dark fill, navy text on a light one — perceived
    luminance, not a fixed threshold, so it works across very different hues
    (a mid-tone amber and a mid-tone navy don't have the same brightness)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#141F3D" if luminance > 150 else "#FFFFFF"


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


def _insight(bullets, variant: str = "info", action: str = "") -> None:
    """Local wrapper: maps the build spec's danger/warning/info/success
    variant names onto ui_template.insight_bar()'s real variant set."""
    _ui_insight_bar(bullets, action=action, variant=_VARIANT_MAP.get(variant, "primary"))


# ── Tab header ────────────────────────────────────────────────────────────────

def render_tab_header() -> None:
    st.markdown(
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'margin-bottom:24px;padding-bottom:16px;border-bottom:2px solid #E4E7ED;'
        'font-family:Inter,sans-serif">'
        '<div>'
        f'<div style="font-size:20px;font-weight:700;color:{_C_NAVY};letter-spacing:-0.3px">'
        'Follow-Up &amp; Continuity</div>'
        f'<div style="font-size:12px;color:{TEXT_MUT};margin-top:3px">'
        "Patient retention, follow-up patterns, and points of attrition</div>"
        '</div>'
        '<div style="display:inline-flex;align-items:center;gap:5px;flex-shrink:0;'
        'background:#F4F6FA;color:#5C6478;padding:5px 12px;border-radius:20px;'
        'font-size:11px;font-weight:500;white-space:nowrap;margin-top:2px">'
        'ⓘ Scoped to patients with an ongoing-care signal in the trailing 365-day window — '
        'not total hospital volume</div>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── KPI row ───────────────────────────────────────────────────────────────────

def render_kpis(df_status: pd.DataFrame, df_visit_number: pd.DataFrame) -> None:
    ltfu_pct = active_pct = lapsing_pct = 0.0
    ltfu_n = active_n = lapsing_n = total_n = 0
    if _safe(df_status):
        total_n = int(df_status["TOTAL_PATIENTS"].sum())
        for _, row in df_status.iterrows():
            if row["STATUS"] == "LTFU":
                ltfu_pct, ltfu_n = float(row["PCT_OF_CLASSIFIABLE_PATIENTS"]), int(row["TOTAL_PATIENTS"])
            elif row["STATUS"] == "Active":
                active_pct, active_n = float(row["PCT_OF_CLASSIFIABLE_PATIENTS"]), int(row["TOTAL_PATIENTS"])
            elif row["STATUS"] == "Lapsing":
                lapsing_pct, lapsing_n = float(row["PCT_OF_CLASSIFIABLE_PATIENTS"]), int(row["TOTAL_PATIENTS"])

    lost_v1_pct = 0.0
    if _safe(df_visit_number):
        total_lost = int(df_visit_number["TOTAL_PATIENTS"].sum())
        v1 = df_visit_number[df_visit_number["LTFU_AT_VISIT_NUMBER"] == "1"]
        v1_n = int(v1["TOTAL_PATIENTS"].iloc[0]) if not v1.empty else 0
        lost_v1_pct = round(100.0 * v1_n / total_lost, 1) if total_lost else 0.0
    else:
        v1_n = total_lost = 0

    retention_pct = active_pct + lapsing_pct
    # Lapsing patients are still counted as "retained" today, but they're the
    # group actively sliding toward LTFU — call them out as at-risk rather
    # than folding them silently into a reassuring green headline number.
    retention_color = WARNING if lapsing_pct >= 15 else SUCCESS

    kpi_row([
        {"label": "Retention rate", "value": fmt_pct(retention_pct, 1),
         "delta": f"{fmt_num(active_n + lapsing_n)} of {fmt_num(total_n)} not LTFU — but "
                  f"{fmt_pct(lapsing_pct, 1)} ({fmt_num(lapsing_n)}) are lapsing and at risk of "
                  f"becoming LTFU", "accent_color": retention_color},
        {"label": "LTFU, current", "value": fmt_pct(ltfu_pct, 1),
         "delta": f"{fmt_num(ltfu_n)} of {fmt_num(total_n)} classifiable patients", "accent_color": DANGER},
        {"label": "Active", "value": fmt_pct(active_pct, 1),
         "delta": f"{fmt_num(active_n)} patients", "accent_color": SUCCESS},
        {"label": "Lapsing", "value": fmt_pct(lapsing_pct, 1),
         "delta": f"{fmt_num(lapsing_n)} patients", "accent_color": WARNING},
        {"label": "LTFU after first visit", "value": fmt_pct(lost_v1_pct, 1),
         "delta": "Patients who did not return after Visit 1", "accent_color": DANGER},
    ])


# ── Section 1 — Retention trend, last 12 months (Pattern A) ─────────────────

def render_s1(df_trend: pd.DataFrame) -> None:
    section_header("1 — Retention Trend, Last 12 Months")
    if not _safe(df_trend):
        _empty()
        return

    wide = df_trend.pivot_table(index="AS_OF_MONTH", columns="STATUS",
                                 values="PCT_OF_CLASSIFIABLE_PATIENTS", aggfunc="first")
    wide = wide.sort_index()

    chart_card("Active / Lapsing / LTFU share, trailing 12 months",
               "Rolling 365-day window, re-classified as of each month's end")
    fig = go.Figure()
    # Exact status-system hex per spec §3/§4 (border tones, not text tones)
    for status, color in [("Active", "#639922"), ("Lapsing", "#EF9F27"), ("LTFU", "#E24B4A")]:
        if status in wide.columns:
            fig.add_trace(go.Scatter(x=wide.index, y=wide[status], mode="lines+markers", name=status,
                                      line=dict(color=color, width=2.5), marker=dict(size=6, color=color)))
    fig.update_layout(**{**_LAYOUT, "height": 140}, xaxis={**AXIS_X, "showgrid": False},
                       yaxis={**AXIS_Y, "ticksuffix": "%"})
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()

    bullets = []
    if "Active" in wide.columns and "LTFU" in wide.columns and len(wide) > 1:
        active_first, active_last = float(wide["Active"].iloc[0]), float(wide["Active"].iloc[-1])
        ltfu_first, ltfu_last = float(wide["LTFU"].iloc[0]), float(wide["LTFU"].iloc[-1])
        active_declining = sum(
            1 for a, b in zip(wide["Active"], wide["Active"].iloc[1:]) if b <= a
        )
        n_months = len(wide)
        bullets.append(
            f"Active share moved from {active_first:.1f}% to {active_last:.1f}% and LTFU moved from "
            f"{ltfu_first:.1f}% to {ltfu_last:.1f}% across the {n_months} months shown — "
            f"{'a sustained decline' if active_last < active_first and ltfu_last > ltfu_first else 'a mixed pattern'}, "
            "not a single-month artifact."
        )
    _insight(bullets or ["Retention trend data is available but incomplete for a monthly comparison."],
             variant="danger")


# ── Section 2 — Who is leaving, and why (Pattern B) ──────────────────────────

# Descriptive breakdown, no verdict — blended-family rule (spec §4 stacked bars)
_PATHWAY_COLORS = {
    "HAD_THEATRE_PROCEDURE": ("Procedure", "#1B8A82"),
    "HAD_MEDICATION_PICKUP": ("Medication", "#C13868"),
    "HAD_INVESTIGATION_OR_LAB": ("Investigation", "#8FCFC8"),
    "HAD_IMAGING": ("Imaging", "#4FADA5"),
    "CONSULTATION_ONLY_NO_OTHER_RECORD": ("Consultation-only", "#8A93A6"),
}


def render_s2(df_ltfu_share: pd.DataFrame, df_pathway: pd.DataFrame) -> None:
    section_header("3 — Who Is Leaving, and Why")

    if not _safe(df_ltfu_share):
        _empty()
    else:
        chart_card(
            "LTFU share by segment, age, and gender",
            "Each cell: LTFU patients in that age/gender slice as a % of the segment's TOTAL "
            "population (within the 365-day window) — how much of the whole segment this specific "
            "group's loss accounts for. Unknown age/gender is shown as its own row/column rather "
            "than dropped.",
        )
        ages = sorted(df_ltfu_share["AGE_GROUP"].unique(), key=_age_sort_key)

        def _cell_color(pct):
            # Fixed 5-stop teal ramp, magnitude only — spec §4 Heatmaps.
            # Scaled 0-100% since this is now a share, not a raw count.
            if pct is None or pd.isna(pct):
                return "#F4F6FA", TEXT_MUT
            if pct > 60:
                return "#1B8A82", "#FFFFFF"
            if pct > 40:
                return "#5DCAA5", "#FFFFFF"
            if pct > 20:
                return "#9FE1CB", "#141F3D"
            return "#E1F5EE", "#141F3D"

        _SEG_COL_PX, _GENDER_COL_PX = 130, 60
        header = "".join(
            f'<th style="font-size:9px;color:{TEXT_MUT};padding:4px 5px;text-align:center;'
            f'white-space:nowrap">{age}</th>' for age in ages
        )
        rows_html = ""
        for seg in sorted(df_ltfu_share["SEGMENT"].unique()):
            seg_df = df_ltfu_share[df_ltfu_share["SEGMENT"] == seg]
            # Every gender value present in the data, including "unknown" —
            # not filtered down to a fixed male/female list.
            present = seg_df["GENDER"].str.lower().unique().tolist()
            genders = [g for g in ["male", "female"] if g in present] + \
                      [g for g in present if g not in ("male", "female")]
            n_genders = len(genders) or 1
            for gi, gender in enumerate(genders or ["unknown"]):
                cells = ""
                for age in ages:
                    m = seg_df[(seg_df["AGE_GROUP"] == age) & (seg_df["GENDER"].str.lower() == gender)]
                    if m.empty:
                        pct, n_total, n_ltfu, n_seg_total = None, None, None, None
                    else:
                        pct = float(m.iloc[0]["LTFU_SHARE_PCT"])
                        n_total = int(m.iloc[0]["TOTAL_PATIENTS"])
                        n_ltfu = int(m.iloc[0]["TOTAL_LTFU_PATIENTS"])
                        n_seg_total = int(m.iloc[0]["SEGMENT_TOTAL_PATIENTS"])
                    bg, fg = _cell_color(pct)
                    # 0% LTFU isn't worth calling out visually — same treatment
                    # as no data. Checked on the rounded display value, not
                    # raw pct == 0, so a share that rounds down (e.g. 0.4%)
                    # is also left blank instead of misleadingly showing "0%".
                    label = "—" if pct is None or round(pct) == 0 else f"{pct:.0f}%"
                    # CSS-anchored tooltip, not the native `title` attribute —
                    # a browser title tooltip positions off the cursor, not
                    # the cell, so in a dense grid it can visually land over
                    # the wrong row and look like it belongs to a neighbor.
                    # This one is pinned directly above the exact cell hovered.
                    tip_html = f'<div class="ltfu-tip">{n_ltfu} out of {n_seg_total} patients LTFU</div>' if pct is not None else ""
                    cells += (f'<td class="ltfu-cell" style="background:{bg};padding:8px 4px;'
                              f'text-align:center;border-radius:4px;position:relative">'
                              f'<div style="font-size:11px;font-weight:600;color:{fg}">{label}</div>'
                              f'{tip_html}</td>')
                seg_cell = (
                    f'<td rowspan="{n_genders}" style="font-size:11px;font-weight:600;color:{TEXT_PRI};'
                    f'padding:6px;white-space:nowrap;vertical-align:middle;width:{_SEG_COL_PX}px">'
                    f'{_short(seg)}</td>'
                ) if gi == 0 else ""
                rows_html += (
                    f'<tr>{seg_cell}'
                    f'<td style="font-size:10px;color:{TEXT_MUT};padding:6px;white-space:nowrap;'
                    f'text-transform:capitalize;width:{_GENDER_COL_PX}px">{gender}</td>{cells}</tr>'
                )
        st.markdown(
            '<style>\n'
            '.ltfu-cell .ltfu-tip {\n'
            'visibility: hidden; opacity: 0; transition: opacity .1s;\n'
            'position: absolute; bottom: 105%; left: 50%; transform: translateX(-50%);\n'
            'background: #141F3D; color: #FFFFFF; font-size: 11px; font-weight: 600;\n'
            'padding: 4px 8px; border-radius: 4px; white-space: nowrap;\n'
            'z-index: 50; pointer-events: none;\n'
            '}\n'
            '.ltfu-cell:hover .ltfu-tip { visibility: visible; opacity: 1; }\n'
            '</style>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch">'
            f'<table style="border-collapse:separate;border-spacing:3px;'
            f'font-family:Inter,sans-serif;width:100%;min-width:520px;table-layout:auto">'
            f'<thead><tr>'
            f'<th style="width:{_SEG_COL_PX}px"></th><th style="width:{_GENDER_COL_PX}px"></th>'
            f'{header}</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )
        _heatmap_total = int(df_ltfu_share["TOTAL_LTFU_PATIENTS"].sum())
        st.markdown(
            f'<div style="font-size:10px;color:{TEXT_MUT};margin-top:6px;font-style:italic">'
            'Each cell: LTFU share % of that segment/age/gender group. Hover a cell for the '
            f'underlying patient counts. Sum of LTFU patients across every cell above: '
            f'<strong>{fmt_num(_heatmap_total)}</strong> — compare against the LTFU KPI at the top of '
            'this tab; both are segment-level counts, so a patient LTFU in two segments counts twice '
            'in each.</div>',
            unsafe_allow_html=True,
        )
        chart_card_close()

    if not _safe(df_pathway):
        _empty()
    else:
        col_l, col_r = st.columns([3, 2])

        with col_l:
            df = df_pathway.copy()
            for col in _PATHWAY_COLORS:
                df[col] = 100.0 * df[col] / df["TOTAL_LTFU_PATIENTS"]
            df = df.sort_values("CONSULTATION_ONLY_NO_OTHER_RECORD")
            labels = [_short(s) for s in df["SEGMENT"]]
            chart_card("What was recorded at final visit",
                       "% of each segment's own LTFU population — not mutually exclusive, bars won't sum to 100%")
            fig = go.Figure()
            for col, (name, color) in _PATHWAY_COLORS.items():
                fig.add_trace(go.Bar(y=labels, x=df[col], name=name, orientation="h",
                                      marker=dict(color=color, cornerradius=3)))
            fig.update_layout(
                **{**_LAYOUT, "height": max(340, 30 * len(labels) * len(_PATHWAY_COLORS) // 2 + 100),
                   "barmode": "group",
                   "legend": dict(orientation="h", y=-0.14, x=0.5, xanchor="center", font=dict(size=10)),
                   "margin": dict(t=8, b=60, l=150, r=40)},
                xaxis={**AXIS_Y, "ticksuffix": "%"}, yaxis={**AXIS_X, "showgrid": False, "automargin": True},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

        with col_r:
            dfp = df_pathway.copy()
            dfp["consult_only_pct"] = 100.0 * dfp["CONSULTATION_ONLY_NO_OTHER_RECORD"] / dfp["TOTAL_LTFU_PATIENTS"]
            worst = dfp.sort_values("consult_only_pct", ascending=False).iloc[0]
            ortho = dfp[dfp["SEGMENT"] == "Core Orthopedics: General"]
            if worst["SEGMENT"] != "Core Orthopedics: General" and not ortho.empty:
                ortho_pct = float(ortho.iloc[0]["consult_only_pct"])
                worst_pct = float(worst["consult_only_pct"])
                ratio = worst_pct / ortho_pct if ortho_pct else 0
                chart_card(f"{_short(worst['SEGMENT'])} vs. Ortho General",
                           "Share of LTFU patients whose final visit was consultation-only")
                fig2 = go.Figure(go.Bar(
                    x=[_short(worst["SEGMENT"]), "Ortho General"], y=[worst_pct, ortho_pct],
                    marker=dict(color=[_C_CORAL, _C_TEAL], cornerradius=3),
                    text=[f"{worst_pct:.1f}%", f"{ortho_pct:.1f}%"], textposition="outside",
                ))
                fig2.update_layout(
                    **{**_LAYOUT, "height": max(340, 30 * len(labels) * len(_PATHWAY_COLORS) // 2 + 100),
                       "margin": dict(t=20, b=40, l=10, r=10)},
                    showlegend=False,
                    xaxis={**AXIS_X, "showgrid": False}, yaxis={**AXIS_Y, "ticksuffix": "%"},
                )
                st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
                st.markdown(
                    f'<div style="text-align:center;font-size:13px;font-weight:700;color:{_C_CORAL};'
                    f'margin-top:4px">{ratio:.1f}x as likely</div>',
                    unsafe_allow_html=True,
                )
                chart_card_close()

    # Combined insight — synthesizes the age/gender concentration (who) with
    # the final-visit pathway pattern (why) into one narrative.
    bullets = []
    has_who = has_why = False
    if _safe(df_ltfu_share):
        by_gender = df_ltfu_share.groupby("GENDER")["TOTAL_LTFU_PATIENTS"].sum()
        if "female" in by_gender.index and "male" in by_gender.index:
            older = df_ltfu_share[df_ltfu_share["AGE_GROUP"].map(lambda a: _age_sort_key(a)[1] >= 55)]
            older_female = older[older["GENDER"] == "female"]["TOTAL_LTFU_PATIENTS"].sum()
            older_female_pop = older[older["GENDER"] == "female"]["TOTAL_PATIENTS"].sum()
            total = df_ltfu_share["TOTAL_LTFU_PATIENTS"].sum()
            if older_female:
                share_txt = ""
                if older_female_pop:
                    share_pct = older_female / older_female_pop * 100
                    share_txt = f" — a {share_pct:.0f}% LTFU share within that group specifically"
                bullets.append(
                    f"<strong>Who:</strong> women 55+ account for {fmt_num(int(older_female))} of "
                    f"{fmt_num(int(total))} LTFU patients across segments — the single largest demographic "
                    f"slice of the population leaving{share_txt}."
                )
                has_who = True

    if _safe(df_pathway):
        dfp = df_pathway.copy()
        dfp["consult_only_pct"] = 100.0 * dfp["CONSULTATION_ONLY_NO_OTHER_RECORD"] / dfp["TOTAL_LTFU_PATIENTS"]
        worst = dfp.sort_values("consult_only_pct", ascending=False).iloc[0]
        ortho = dfp[dfp["SEGMENT"] == "Core Orthopedics: General"]
        if worst["SEGMENT"] != "Core Orthopedics: General" and not ortho.empty:
            ortho_pct = float(ortho.iloc[0]["consult_only_pct"])
            ratio = worst["consult_only_pct"] / ortho_pct if ortho_pct else 0
            bullets.append(
                f"<strong>Why:</strong> {_short(worst['SEGMENT'])} patients are {ratio:.1f}x as likely as "
                f"Ortho General patients ({worst['consult_only_pct']:.1f}% vs {ortho_pct:.1f}%) to disappear "
                "after a visit where nothing beyond a conversation was recorded — no medication, no imaging, "
                "no investigation. That's a structurally weaker touchpoint to begin with, independent of the "
                "patient's intent to return."
            )
        else:
            bullets.append(
                f"<strong>Why:</strong> {_short(worst['SEGMENT'])} has the highest consultation-only "
                f"final-visit rate ({worst['consult_only_pct']:.1f}%) of any segment — the weakest touchpoint "
                "before going LTFU."
            )
        has_why = True

        # Draws on the FULL breakdown chart (all segments), not just the
        # two-segment comparison — shows the worst segment isn't an outlier
        # from one benchmark, it's the extreme of a pattern across the board.
        if len(dfp) > 2:
            best = dfp.sort_values("consult_only_pct", ascending=True).iloc[0]
            best_pct = float(best["consult_only_pct"])
            worst_pct = float(worst["consult_only_pct"])
            spread = worst_pct - best_pct
            evidence_cols = ["HAD_THEATRE_PROCEDURE", "HAD_MEDICATION_PICKUP", "HAD_INVESTIGATION_OR_LAB", "HAD_IMAGING"]
            best_evidence_pct = 100.0 * dfp.loc[dfp["SEGMENT"] == best["SEGMENT"], evidence_cols].sum(axis=1).iloc[0] / \
                best["TOTAL_LTFU_PATIENTS"]
            bullets.append(
                f"<strong>Full picture:</strong> across all {len(dfp)} segments, consultation-only final "
                f"visits range from {best_pct:.1f}% ({_short(best['SEGMENT'])}) to {worst_pct:.1f}% "
                f"({_short(worst['SEGMENT'])}) — a {spread:.0f}-point spread. {_short(best['SEGMENT'])}'s "
                f"final visits carry {best_evidence_pct:.0f}% combined procedure/medication/investigation/"
                f"imaging coverage; {_short(worst['SEGMENT'])} sits at the opposite end. This isn't one "
                "segment underperforming one benchmark — it's a hospital-wide gradient in how much clinical "
                "trail a segment leaves before a patient disappears."
            )

    # "So what" is only earned if the two findings above are actually about
    # the same patients — check whether women 55+ are over-represented
    # specifically within the weak-touchpoint segments (worst + Ortho
    # General), not just assumed to be, before claiming a compounding risk.
    if has_who and has_why and _safe(df_ltfu_share) and _safe(df_pathway):
        def _older_female_share_of_segment(segment: str):
            seg_df = df_ltfu_share[df_ltfu_share["SEGMENT"] == segment]
            seg_total_ltfu = seg_df["TOTAL_LTFU_PATIENTS"].sum()
            if not seg_total_ltfu:
                return None
            older_f = seg_df[
                (seg_df["GENDER"] == "female")
                & (seg_df["AGE_GROUP"].map(lambda a: _age_sort_key(a)[1] >= 55))
            ]["TOTAL_LTFU_PATIENTS"].sum()
            return float(older_f) / float(seg_total_ltfu) * 100

        baseline_pct = (float(older_female) / float(total) * 100) if _safe(df_ltfu_share) and total else None
        worst_seg_pct = _older_female_share_of_segment(worst["SEGMENT"])
        ortho_seg_pct = _older_female_share_of_segment("Core Orthopedics: General")

        if baseline_pct is not None and worst_seg_pct is not None and ortho_seg_pct is not None:
            concentrated = worst_seg_pct > baseline_pct and ortho_seg_pct > baseline_pct
            if concentrated:
                bullets.append(
                    f"<em><strong>So what:</strong> women 55+ make up {worst_seg_pct:.0f}% of "
                    f"{_short(worst['SEGMENT'])}'s own LTFU patients and {ortho_seg_pct:.0f}% of Ortho "
                    f"General's, both above the {baseline_pct:.0f}% hospital-wide baseline — the population "
                    "most likely to disappear is genuinely concentrated in the segments whose final visit "
                    "leaves the weakest clinical trail, a compounding risk rather than two unrelated "
                    "findings.</em>"
                )
            else:
                bullets.append(
                    f"<em><strong>So what:</strong> women 55+ make up {worst_seg_pct:.0f}% of "
                    f"{_short(worst['SEGMENT'])}'s LTFU patients and {ortho_seg_pct:.0f}% of Ortho General's, "
                    f"against a {baseline_pct:.0f}% hospital-wide baseline — not clearly concentrated in "
                    "these weak-touchpoint segments, so the 'who' and 'why' findings above should be treated "
                    "as two separate risks rather than one compounding one.</em>"
                )

    _insight(bullets or ["LTFU demographic and final-visit pathway data is limited."], variant="danger")


# ── Section 3 — Where the problem concentrates (Pattern B) ──────────────────

_S3_RELIABLE = ["Spine-conservative", "Spine-structural", "ANC / Routine Pregnancy",
                "Core Orthopedics: General", "High-Risk Pregnancy", "Fibroids-conservative"]
_S3_SMALL_N = {"Fibroids-conservative", "High-Risk Pregnancy"}


# Sequential by visit number, not a verdict — teal ramp, darkest = visit 1
# (the largest loss point), per spec §4 donut charts.
_VISIT_NUMBER_COLORS = ["#0F6E56", "#1B8A82", "#4FADA5", "#8FCFC8", "#B8E0DA", "#E1F5EE", _C_NEUTRAL2]


_RELATED_DESTINATIONS = {
    "Spine-structural": {"Spine-conservative"},
    "Spine-conservative": {"Spine-structural", "Core Orthopedics: General"},
    "Core Orthopedics: General": {"Spine-structural", "Spine-conservative"},
    "ANC / Routine Pregnancy": {"High-Risk Pregnancy"},
    "High-Risk Pregnancy": {"ANC / Routine Pregnancy"},
}


def render_s3(
    df_status_seg: pd.DataFrame, df_visit_number: pd.DataFrame,
    df_visit_number_by_segment: pd.DataFrame = None,
    df_patient_signals: pd.DataFrame = None,
) -> None:
    section_header("2 — Where the Problem Concentrates")

    # Only two charts, deliberately: the point of this section is "a lot of
    # patients leave after one visit — here's where that concentrates by
    # clinical area," not a general segment-LTFU-rate comparison (that's
    # covered elsewhere). A per-segment LTFU-rate bar chart was removed from
    # here for that reason.
    col_l, col_r = st.columns(2)

    with col_l:
        if not _safe(df_visit_number):
            _empty()
        else:
            df = df_visit_number.copy()
            order = ["1", "2", "3", "4", "5", "6", "7+"]
            df["_order"] = df["LTFU_AT_VISIT_NUMBER"].map(lambda v: order.index(v) if v in order else 99)
            df = df.sort_values("_order")
            colors = _VISIT_NUMBER_COLORS[: len(df)]
            chart_card("LTFU by visit number", "At which visit did the care relationship effectively end")
            fig = go.Figure(go.Pie(
                labels=df["LTFU_AT_VISIT_NUMBER"], values=df["TOTAL_PATIENTS"], hole=0.55,
                marker=dict(colors=colors), textinfo="percent",
            ))
            fig.update_layout(
                **{**_LAYOUT, "height": _H_PAIRED,
                   "legend": dict(orientation="v", x=1.02, y=0.5, font=dict(size=10))},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(df_visit_number_by_segment):
            _empty()
        else:
            chart_card(
                "LTFU by visit number, per segment",
                "Of each segment's own LTFU patients, at which visit did the care relationship end?",
            )
            dvs = df_visit_number_by_segment.copy()
            order = ["1", "2", "3", "4", "5", "6", "7+"]
            dvs["_order"] = dvs["LTFU_AT_VISIT_NUMBER"].map(lambda v: order.index(v) if v in order else 99)
            seg_order = (
                dvs[dvs["LTFU_AT_VISIT_NUMBER"] == "1"]
                .sort_values("PCT_WITHIN_SEGMENT", ascending=True)["SEGMENT"].tolist()
            )
            seg_labels = [_short(s) for s in seg_order]
            fig_seg = go.Figure()
            for vn in order:
                sub = dvs[dvs["LTFU_AT_VISIT_NUMBER"] == vn].set_index("SEGMENT")
                y_vals = [float(sub.loc[s, "PCT_WITHIN_SEGMENT"]) if s in sub.index else 0.0 for s in seg_order]
                color = _VISIT_NUMBER_COLORS[order.index(vn) % len(_VISIT_NUMBER_COLORS)]
                fig_seg.add_trace(go.Bar(
                    y=seg_labels, x=y_vals, orientation="h", name=vn,
                    marker=dict(color=color, cornerradius=2),
                    text=[f"{v:.0f}%" if v >= 6 else "" for v in y_vals], textposition="inside",
                    textfont=dict(size=9, color="#FFFFFF" if order.index(vn) <= 2 else _C_NAVY),
                ))
            fig_seg.update_layout(
                **{**_LAYOUT, "height": _H_PAIRED, "barmode": "stack",
                   "legend": dict(orientation="h", y=-0.22, x=0.5, xanchor="center", title="Visit #",
                                   font=dict(size=10))},
                xaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, 100]},
                yaxis={**AXIS_X, "showgrid": False},
            )
            st.plotly_chart(fig_seg, use_container_width=True, config=PC_CFG)
            chart_card_close()

    # ── Outcome classification, from patient-level signals ──────────────────
    # A missing follow-up date means "unknown," not "no follow-up needed" —
    # so silence on every signal falls to Unresolved, never to a clean bucket.
    def _classify_outcome(row) -> str:
        related = _RELATED_DESTINATIONS.get(row["SEGMENT"], set())
        if row["HAS_LATER_VISIT_ELSEWHERE"] == 1 and row.get("NEXT_VISIT_ELSEWHERE_SEGMENT") in related:
            return "Probable pathway transfer"
        if row["HAD_SCHEDULED_FOLLOWUP"] == 1:
            return "Probable true LTFU"
        if row["HAD_PROCEDURE"] == 1 or row["HAD_MEDICATION"] == 1 or row["HAD_INVESTIGATION"] == 1 or row["HAD_IMAGING"] == 1:
            return "Possible completed episode"
        return "Unresolved"

    df_outcomes = None
    if _safe(df_patient_signals):
        df_outcomes = df_patient_signals.copy()
        df_outcomes["OUTCOME"] = df_outcomes.apply(_classify_outcome, axis=1)

    # ── LTFU pathway sankey: total → when care ended → what happened ────────
    # Replaces the separate "outcome classification by area" and "where
    # patients went next" charts with one flow that answers both at once.
    # Tier 3 buckets reuse the same classification as everywhere else in
    # this section (_classify_outcome) — the sankey doesn't introduce a
    # new definition, just a new way of showing the existing one.
    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"

    _RED_FILL, _RED_STROKE = "#FCEBEB", "#E24B4A"
    _AMBER_FILL, _AMBER_STROKE = "#FAEEDA", "#EF9F27"
    _GREEN_FILL, _GREEN_STROKE = "#EAF3DE", "#639922"
    _NEUTRAL_FILL, _NEUTRAL_STROKE = _C_NEUTRAL2, _C_NEUTRAL

    _TIER3_ORDER = ["No continuity", "Care delivered, outcome unclear", "Pathway found", "No return visit"]
    _OUTCOME_TO_TIER3 = {
        "Unresolved": "No continuity",
        "Possible completed episode": "Care delivered, outcome unclear",
        "Probable pathway transfer": "Pathway found",
        "Probable true LTFU": "No return visit",
    }
    _TIER3_FILL = {"No continuity": _RED_FILL, "Care delivered, outcome unclear": _AMBER_FILL,
                   "Pathway found": _GREEN_FILL, "No return visit": _RED_FILL}
    _TIER3_STROKE = {"No continuity": _RED_STROKE, "Care delivered, outcome unclear": _AMBER_STROKE,
                      "Pathway found": _GREEN_STROKE, "No return visit": _RED_STROKE}

    if _safe(df_outcomes):
        df_outcomes["TIER3"] = df_outcomes["OUTCOME"].map(_OUTCOME_TO_TIER3)
        total = len(df_outcomes)
        is_v1 = df_outcomes["VISIT_NUMBER_AT_LTFU"] == "1"
        n_v1, n_v2 = int(is_v1.sum()), int((~is_v1).sum())

        node_labels = [
            f"All LTFU<br>{total:,} episodes",
            f"Visit 1 loss<br>{n_v1:,} ({n_v1 / total * 100:.0f}%)" if total else "Visit 1 loss",
            f"Visit 2+ loss<br>{n_v2:,} ({n_v2 / total * 100:.0f}%)" if total else "Visit 2+ loss",
        ] + _TIER3_ORDER
        node_fill = [_NEUTRAL_FILL, _NEUTRAL_FILL, _NEUTRAL_FILL] + [_TIER3_FILL[t] for t in _TIER3_ORDER]
        node_stroke = [_NEUTRAL_STROKE, _NEUTRAL_STROKE, _NEUTRAL_STROKE] + [_TIER3_STROKE[t] for t in _TIER3_ORDER]

        link_source, link_target, link_value, link_color = [0, 0], [1, 2], [n_v1, n_v2], \
            [_hex_to_rgba(_NEUTRAL_STROKE, 0.35)] * 2
        for tier2_idx, mask in ((1, is_v1), (2, ~is_v1)):
            for j, tier3 in enumerate(_TIER3_ORDER):
                v = int((mask & (df_outcomes["TIER3"] == tier3)).sum())
                link_source.append(tier2_idx)
                link_target.append(3 + j)
                link_value.append(v)
                link_color.append(_hex_to_rgba(_TIER3_STROKE[tier3], 0.4))

        chart_card(
            "LTFU pathway: when care ended, and what happened next",
            "Of everyone lost to follow-up — when they were lost, and whether that loss was explained",
        )
        fig_sankey = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                label=node_labels, pad=18, thickness=16,
                color=node_fill, line=dict(color=node_stroke, width=1),
                hovertemplate="%{label}<extra></extra>",
            ),
            link=dict(
                source=link_source, target=link_target, value=link_value, color=link_color,
                hovertemplate="%{value} patients<extra></extra>",
            ),
        ))
        fig_sankey.update_layout(**{**_LAYOUT, "height": 420, "font": dict(size=11)})
        st.plotly_chart(fig_sankey, use_container_width=True, config=PC_CFG)
        chart_card_close()

    # ── First-visit loss requiring review — patient-level download only ─────
    v1 = df_outcomes[df_outcomes["VISIT_NUMBER_AT_LTFU"] == "1"] if _safe(df_outcomes) else None
    if v1 is not None and not v1.empty:
        def _v1_bucket(outcome: str) -> str:
            if outcome == "Probable pathway transfer":
                return "Had a later related visit"
            if outcome == "Probable true LTFU":
                return "Documented follow-up, no return"
            return "No subsequent visit at all"

        v1 = v1.copy()
        v1["bucket"] = v1["OUTCOME"].map(_v1_bucket)
        _bucket_rank = {"Documented follow-up, no return": 0, "No subsequent visit at all": 1, "Had a later related visit": 2}
        export_cols = [
            "SEGMENT", "PATIENT_ID", "bucket", "LAST_VISIT_DATE", "DAYS_SINCE_LAST_VISIT",
            "DIAGNOSIS_TEXT", "SCHEDULED_FOLLOWUP_DATE", "PROCEDURE_NAMES",
            "NEXT_VISIT_ELSEWHERE_DATE", "NEXT_VISIT_ELSEWHERE_SEGMENT",
        ]
        v1_export = (
            v1[[c for c in export_cols if c in v1.columns]]
            .rename(columns={"bucket": "REVIEW_CATEGORY"})
            .sort_values(
                by=["REVIEW_CATEGORY", "DAYS_SINCE_LAST_VISIT"],
                key=lambda col: col.map(_bucket_rank) if col.name == "REVIEW_CATEGORY" else col,
                ascending=[True, False],
            )
        )
        st.download_button(
            "Download first-visit loss list (CSV)",
            data=v1_export.to_csv(index=False).encode("utf-8"),
            file_name="ltfu_first_visit_loss_for_review.csv",
            mime="text/csv",
            key="fr_s3_v1_download",
        )

    # ── D. Data-quality callout ──────────────────────────────────────────────
    st.markdown(
        f'<div style="padding:10px 14px;border-left:3px solid {WARNING};'
        f'background:#FAEEDA;border-radius:0 6px 6px 0;margin:8px 0 16px;'
        f'font-family:Inter,sans-serif;font-size:12px;color:{TEXT_SEC}">'
        f'<strong>Follow-up schedule unavailable:</strong> Scheduled follow-up dates are missing for '
        f'many patients, so Unresolved cases cannot yet be separated reliably into expected care '
        f'completion and true LTFU.</div>',
        unsafe_allow_html=True,
    )

    def _v1_and_v2_pct(segment: str):
        if not _safe(df_visit_number_by_segment):
            return None, None
        sub = df_visit_number_by_segment[df_visit_number_by_segment["SEGMENT"] == segment]
        v1_row = sub[sub["LTFU_AT_VISIT_NUMBER"] == "1"]
        v2_row = sub[sub["LTFU_AT_VISIT_NUMBER"] == "2"]
        v1 = float(v1_row.iloc[0]["PCT_WITHIN_SEGMENT"]) if not v1_row.empty else None
        v2 = float(v2_row.iloc[0]["PCT_WITHIN_SEGMENT"]) if not v2_row.empty else None
        return v1, v2

    # Patient-level signals for visit-1 losses only, per segment — used to
    # separate "moved into a related pathway" from "documented follow-up but
    # never returned" from "no trace at all," instead of treating every
    # visit-1 loss as undifferentiated attrition.
    def _v1_signals(segment: str):
        if not _safe(df_patient_signals):
            return None
        sub = df_patient_signals[
            (df_patient_signals["SEGMENT"] == segment)
            & (df_patient_signals["VISIT_NUMBER_AT_LTFU"] == "1")
        ]
        n = len(sub)
        if n == 0:
            return None
        related = _RELATED_DESTINATIONS.get(segment, set())
        is_related_transfer = (
            (sub["HAS_LATER_VISIT_ELSEWHERE"] == 1)
            & (sub["NEXT_VISIT_ELSEWHERE_SEGMENT"].isin(related))
        )
        n_transfer = int(is_related_transfer.sum())
        n_documented_no_return = int(
            ((sub["HAD_SCHEDULED_FOLLOWUP"] == 1) & ~is_related_transfer).sum()
        )
        n_no_trace = int(
            ((sub["HAS_LATER_VISIT_ELSEWHERE"] == 0) & (sub["HAD_SCHEDULED_FOLLOWUP"] == 0)).sum()
        )
        return {"n": n, "transfer": n_transfer, "documented_no_return": n_documented_no_return, "no_trace": n_no_trace}

    # Four conclusions, each computed live, then a short imperative action
    # list — same structure as the OPD-IPD tab's insight_bar(bullets=, action=).
    spine_struct_v1, _ = _v1_and_v2_pct("Spine-structural")
    anc_v1, _ = _v1_and_v2_pct("ANC / Routine Pregnancy")
    fibroids_v1, _ = _v1_and_v2_pct("Fibroids-conservative")
    spine_cons_v1, _ = _v1_and_v2_pct("Spine-conservative")
    hrp_v1, hrp_v2 = _v1_and_v2_pct("High-Risk Pregnancy")
    hrp_flagged = hrp_v1 is not None and hrp_v2 is not None and round(hrp_v1 + hrp_v2) >= 99

    sig_spine_struct = _v1_signals("Spine-structural")
    sig_spine_cons = _v1_signals("Spine-conservative")

    n_no_trace_all = None
    if _safe(df_outcomes):
        v1_all = df_outcomes[df_outcomes["VISIT_NUMBER_AT_LTFU"] == "1"]
        n_no_trace_all = int((v1_all["HAS_LATER_VISIT_ELSEWHERE"] == 0).sum())

    high_conc = [
        (n, v) for n, v in
        [("Spine (structural)", spine_struct_v1), ("ANC", anc_v1), ("Fibroids", fibroids_v1)]
        if v is not None
    ]

    bullets = []

    if high_conc:
        one_and_done = "<strong>One-and-done attendance, not gradual drop-off, drives most loss.</strong> " + \
            ", ".join(f"{n} {v:.0f}%" for n, v in high_conc) + " of losses happen right after visit 1"
        if sig_spine_cons and sig_spine_cons["n"]:
            sc_pct = round(sig_spine_cons["no_trace"] / sig_spine_cons["n"] * 100)
            one_and_done += (
                f" — even Spine (conservative), the most spread-out segment, still loses "
                f"{sig_spine_cons['no_trace']}/{sig_spine_cons['n']} (~{sc_pct}%) after visit 1 alone."
            )
        else:
            one_and_done += "."
        bullets.append(one_and_done)

    if sig_spine_struct and sig_spine_struct["n"]:
        transfer_pct = round(sig_spine_struct["transfer"] / sig_spine_struct["n"] * 100)
        remainder = sig_spine_struct["n"] - sig_spine_struct["transfer"]
        bullets.append(
            "<strong>Legitimate pathway transfer explains only a minority of checkable cases.</strong> "
            f"For Spine (structural), the only segment with a clear related destination, just "
            f"{sig_spine_struct['transfer']}/{sig_spine_struct['n']} ({transfer_pct}%) moved to "
            f"Spine-conservative — the remaining {remainder} ({sig_spine_struct['no_trace']} with no trace "
            "plus documented-follow-up-no-return) can't be assumed as continued care."
        )

    if n_no_trace_all:
        bullets.append(
            "<strong>The unexplained volume is far larger than any segment shows.</strong> Hospital-wide, "
            f"<strong>{n_no_trace_all:,} visit-1 losses</strong> have no later visit anywhere — dwarfing the "
            "segment counts above, meaning most of this gap sits outside the highlighted segments (e.g. "
            "Core Orthopedics: General)."
        )

    if anc_v1 is not None or hrp_flagged:
        bullets.append(
            "<strong>Clinical risk compounds the retention risk.</strong> ANC and High-Risk Pregnancy carry "
            "the highest first-visit concentration and the highest clinical stakes — a silent loss there is "
            "a safety exposure, not just a revenue one."
        )

    actions = []
    if spine_struct_v1 is not None:
        actions.append(("Spine (structural)", "confirm referrals; reclassify verified Spine-conservative transfers instead of counting them as LTFU."))
    if anc_v1 is not None:
        actions.append(("ANC", "review patients with no later attendance to confirm whether care continued elsewhere."))
    if spine_cons_v1 is not None:
        actions.append(("Spine (conservative)", "identify patients told to return and prioritize them for follow-up outreach."))
    if fibroids_v1 is not None:
        actions.append(("Fibroids", "review each case individually; too few to trend as a segment."))
    if hrp_flagged:
        actions.append(("High-Risk Pregnancy", "review each loss individually to confirm outcome and continuity of care."))
    if n_no_trace_all:
        actions.append(("Hospital-wide", "pull the downloadable list above and work through it, starting with 'No subsequent visit at all.'"))

    # Rendered inline (via _insight's action= slot) so it stays part of the
    # same box as the conclusions above, but with its own styling — numbered
    # teal badges and normal-weight text — that overrides the bar's default
    # bold-on-red action treatment, which reads poorly for a multi-line list.
    actions_html = ""
    if actions:
        header_html = (
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:.06em;color:{TEXT_MUT};margin-bottom:8px">Recommended actions</div>'
        )
        rows_html = "".join(
            f'<div style="display:flex;gap:10px;align-items:flex-start;'
            f'{"margin-bottom:8px;" if i < len(actions) - 1 else ""}">'
            f'<div style="flex:0 0 20px;height:20px;border-radius:50%;background:{PRIMARY};'
            f'color:#FFFFFF;font-size:11px;font-weight:700;display:flex;align-items:center;'
            f'justify-content:center">{i + 1}</div>'
            f'<div style="font-size:13px;font-weight:400;color:{TEXT_SEC};line-height:1.5;padding-top:1px">'
            f'<strong style="color:{TEXT_PRI}">{seg}</strong> — {act}</div></div>'
            for i, (seg, act) in enumerate(actions)
        )
        actions_html = header_html + rows_html

    _insight(
        bullets or ["Visit-number breakdown by segment is not currently available."],
        variant="danger",
        action=actions_html,
    )


# ── Section 4 — Two segments, two different stories (Pattern B) ─────────────

def render_s4(df_condition: pd.DataFrame) -> None:
    section_header("4 — Two Segments, Two Different Stories")
    col_l, col_r = st.columns(2)

    with col_l:
        if not _safe(df_condition):
            _empty()
        else:
            df = df_condition[
                (df_condition["SEGMENT"] == "Core Orthopedics: General")
                & (df_condition["CONDITION_CATEGORY"] != "Other / Unclassified")
            ].sort_values("DISTINCT_LTFU_PATIENTS").tail(5)
            chart_card("Core Orthopedics: General — condition breakdown", "Top 5 named conditions")
            if df.empty:
                _empty()
            else:
                fig = go.Figure(go.Bar(
                    y=df["CONDITION_CATEGORY"], x=df["PCT_WITHIN_SEGMENT"], orientation="h",
                    marker=dict(color=_C_NAVY, cornerradius=3),
                    text=[f"{v:.1f}%" for v in df["PCT_WITHIN_SEGMENT"]], textposition="outside",
                    textfont=dict(size=11, color=TEXT_SEC),
                ))
                fig.update_layout(
                    **{**_LAYOUT, "height": _H_PAIRED, "margin": dict(t=8, b=40, l=10, r=50)}, showlegend=False,
                    xaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, df["PCT_WITHIN_SEGMENT"].max() * 1.25]},
                    yaxis={**AXIS_X, "showgrid": False, "automargin": True},
                )
                st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col_r:
        if not _safe(df_condition):
            _empty()
        else:
            df = df_condition[
                (df_condition["SEGMENT"] == "Spine-conservative")
                & (df_condition["CONDITION_CATEGORY"] != "Other / Unclassified")
            ].sort_values("DISTINCT_LTFU_PATIENTS").tail(5)
            chart_card("Spine-conservative — condition breakdown", "Top 5 named conditions")
            if df.empty:
                _empty()
            else:
                fig = go.Figure(go.Bar(
                    y=df["CONDITION_CATEGORY"], x=df["PCT_WITHIN_SEGMENT"], orientation="h",
                    marker=dict(color=_C_CORAL, cornerradius=3),
                    text=[f"{v:.1f}%" for v in df["PCT_WITHIN_SEGMENT"]], textposition="outside",
                    textfont=dict(size=11, color=TEXT_SEC),
                ))
                fig.update_layout(
                    **{**_LAYOUT, "height": _H_PAIRED, "margin": dict(t=8, b=40, l=10, r=50)}, showlegend=False,
                    xaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, df["PCT_WITHIN_SEGMENT"].max() * 1.25]},
                    yaxis={**AXIS_X, "showgrid": False, "automargin": True},
                )
                st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    bullets = []
    if _safe(df_condition):
        ortho_named = df_condition[
            (df_condition["SEGMENT"] == "Core Orthopedics: General")
            & (df_condition["CONDITION_CATEGORY"] != "Other / Unclassified")
        ]
        spine_named = df_condition[
            (df_condition["SEGMENT"] == "Spine-conservative")
            & (df_condition["CONDITION_CATEGORY"] != "Other / Unclassified")
        ]
        if not ortho_named.empty:
            top_ortho = ortho_named.sort_values("PCT_WITHIN_SEGMENT", ascending=False).iloc[0]
            n_ortho = ortho_named["CONDITION_CATEGORY"].nunique()
            bullets.append(
                f"<strong>Ortho General:</strong> loss is broad and scattered — {n_ortho} distinct named "
                f"conditions each carry meaningful share, led by {top_ortho['CONDITION_CATEGORY']} at only "
                f"{top_ortho['PCT_WITHIN_SEGMENT']:.1f}%. No single condition explains the loss, which "
                "points to a general re-engagement problem across the whole segment, not a condition-"
                "specific one."
            )
        if not spine_named.empty:
            top_spine = spine_named.sort_values("PCT_WITHIN_SEGMENT", ascending=False).iloc[0]
            bullets.append(
                f"<strong>Spine-conservative:</strong> loss concentrates in one condition — "
                f"{top_spine['CONDITION_CATEGORY']} alone accounts for {top_spine['PCT_WITHIN_SEGMENT']:.1f}% "
                "of this segment's named LTFU cases. This segment functions largely as a conservative "
                "back-pain clinic, and that population needs its own tailored retention approach rather "
                "than the broad-based fix Ortho General needs."
            )
        if len(bullets) == 2:
            bullets.append(
                "<em><strong>So what:</strong> a single retention programme can't fix both — Ortho General "
                "needs a general re-engagement effort, Spine-conservative needs one targeted intervention "
                "aimed at its dominant condition.</em>"
            )
    _insight(bullets or ["Condition-level breakdown is limited for this comparison."], variant="info")


# ── Section 5 — Priority outreach (Pattern D, used once) ────────────────────

def render_s5(df_outreach: pd.DataFrame, df_lost_v1: pd.DataFrame, df_condition: pd.DataFrame) -> None:
    section_header("5 — Priority Outreach")
    if not _safe(df_outreach):
        _empty()
        return

    n = len(df_outreach)
    n_55plus = int((df_outreach["AGE_GROUP"].isin(["55-64", "65+"])).sum())
    n_65f = int(((df_outreach["AGE_GROUP"] == "65+") & (df_outreach["GENDER"] == "female")).sum())
    min_days = int(df_outreach["DAYS_SINCE_VISIT"].min())
    max_days = int(df_outreach["DAYS_SINCE_VISIT"].max())

    context = (
        f"Manually verifiable post-spine-surgery patients, {min_days}–{max_days} days since last visit — "
        f"{n_55plus} of {n} are 55+, {n_65f} of {n} are specifically 65+ and female."
    )

    sub = ""
    if _safe(df_lost_v1) and _safe(df_condition):
        v1_spine = df_lost_v1[df_lost_v1["SEGMENT"] == "Spine-structural"]
        v1_total = int(v1_spine["TOTAL_PATIENTS"].sum())
        v1_pss = v1_spine[v1_spine["CONDITION_CATEGORY"] == "Post-Spine-Surgery Follow-up"]
        v1_pss_n = int(v1_pss["TOTAL_PATIENTS"].sum())
        v1_pct = round(100.0 * v1_pss_n / v1_total, 1) if v1_total else 0.0

        overall_spine = df_condition[df_condition["SEGMENT"] == "Spine-structural"]
        overall_pss = overall_spine[overall_spine["CONDITION_CATEGORY"] == "Post-Spine-Surgery Follow-up"]
        overall_pct = float(overall_pss.iloc[0]["PCT_WITHIN_SEGMENT"]) if not overall_pss.empty else 0.0

        sub = (
            f"Post-Spine-Surgery Follow-up is {v1_pct:.1f}% of Spine-structural's first-visit losses alone, "
            f"vs. {overall_pct:.1f}% of the segment's overall LTFU — the problem concentrates further when "
            "isolated to visit-1 losses."
        )

    sharp_finding_card(
        eyebrow="Verified post-spine-surgery outreach list",
        stat=f"{n} patients",
        context=context,
        sub=sub,
    )

    with st.expander("Named patient list (operational worklist export)"):
        st.dataframe(
            df_outreach[["PATIENT_ID", "LAST_VISIT_DATE", "DAYS_SINCE_VISIT", "AGE_GROUP", "GENDER"]],
            use_container_width=True, hide_index=True,
        )


# ── Section B header — scheduling KPI row ────────────────────────────────────

def render_section_b_header(df_scheduled: pd.DataFrame) -> None:
    st.markdown(
        '<div style="background:#FAEEDA;border:1px solid #EF9F27;border-radius:8px;'
        'padding:10px 14px;margin:8px 0 16px;font-size:12px;color:#854F0B;'
        'font-family:Inter,sans-serif">'
        '⚠ <strong>Scheduling data cutoff:</strong> no "Schedule Follow Up" records exist past a fixed '
        'point in the source data. Section B figures for the most recent period should be read as '
        'incomplete, not as evidence that scheduling stopped happening.'
        '</div>',
        unsafe_allow_html=True,
    )
    section_header("Section B — Is the Scheduling System Working?")
    if not _safe(df_scheduled):
        _empty()
        return

    cards = []
    for seg in ["Core Orthopedics: Spine and Back Pain Care", "Core Orthopedics: General",
                "ANC", "Maternal Health", "Core General Surgery"]:
        row = df_scheduled[df_scheduled["SEGMENT"] == seg]
        if row.empty:
            continue
        r = row.iloc[0]
        cards.append({
            "label": f"{_short(seg)} scheduled rate",
            "value": fmt_pct(r["PCT_SCHEDULED"], 1),
            "delta": f"{fmt_num(int(r['MATCHED_A_SCHEDULE']))} of {fmt_num(int(r['TOTAL_RETURN_VISITS']))} returns",
            "accent_color": DANGER if r["PCT_SCHEDULED"] < 10 else WARNING,
        })
    if cards:
        kpi_row(cards)


# ── Section 6 — Scheduled vs. self-initiated returns (Pattern A) ────────────

def render_s6(df_scheduled: pd.DataFrame, df_scheduled_age: pd.DataFrame) -> None:
    section_header("6 — Scheduled vs. Self-Initiated Returns")
    if not _safe(df_scheduled):
        _empty()
        return

    df = df_scheduled.sort_values("PCT_SCHEDULED")
    chart_card("Scheduled vs. self-initiated returns, by segment",
               "Self-initiated = the patient came back on their own, with no scheduling action behind it")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[_short(s) for s in df["SEGMENT"]], x=df["PCT_SCHEDULED"], name="Scheduled",
                          orientation="h", marker=dict(color=_C_TEAL, cornerradius=3)))
    fig.add_trace(go.Bar(y=[_short(s) for s in df["SEGMENT"]], x=df["PCT_ORGANIC"], name="Self-initiated",
                          orientation="h", marker=dict(color="#D3D6DE", cornerradius=3)))
    fig.update_layout(
        **{**_LAYOUT, "height": _H_SINGLE, "barmode": "stack",
           "legend": dict(orientation="h", y=-0.18, x=0.5, xanchor="center")},
        xaxis={**AXIS_Y, "ticksuffix": "%"}, yaxis={**AXIS_X, "showgrid": False},
    )
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()

    worst = df.sort_values("PCT_ORGANIC", ascending=False).iloc[0]
    _insight(
        [f"Every segment is overwhelmingly self-initiated. {_short(worst['SEGMENT'])} is the extreme case "
         f"at {worst['PCT_ORGANIC']:.1f}% self-initiated, meaning almost none of its return visits trace to "
         "a deliberate scheduling action."],
        variant="danger",
    )

    if not _safe(df_scheduled_age):
        _empty()
        return

    chart_card("Scheduled-returns volume by age and gender", "Count of scheduled returns per segment x age x gender")
    ages = sorted(df_scheduled_age["AGE_GROUP"].unique(), key=_age_sort_key)
    max_v = df_scheduled_age["TOTAL_SCHEDULED_RETURNS"].max()

    def _cell_color(v):
        # Fixed 5-stop teal ramp, magnitude only — spec §4 Heatmaps.
        # Min-max scaled within this table, not shared across tables.
        if v is None or pd.isna(v) or v == 0:
            return "#F4F6FA", TEXT_MUT
        ratio = v / max_v if max_v else 0
        if ratio > 0.75:
            return "#1B8A82", "#FFFFFF"
        if ratio > 0.5:
            return "#5DCAA5", "#FFFFFF"
        if ratio > 0.25:
            return "#9FE1CB", "#141F3D"
        return "#E1F5EE", "#141F3D"

    _SEG_COL_PX, _GENDER_COL_PX = 130, 60
    header = "".join(
        f'<th style="font-size:9px;color:{TEXT_MUT};padding:4px 5px;text-align:center;'
        f'white-space:nowrap">{age}</th>' for age in ages
    )
    rows_html = ""
    for seg in sorted(df_scheduled_age["SEGMENT"].unique()):
        seg_df = df_scheduled_age[df_scheduled_age["SEGMENT"] == seg]
        genders = [g for g in ["male", "female"] if g in seg_df["GENDER"].str.lower().values]
        n_genders = len(genders) or 1
        for gi, gender in enumerate(genders or ["—"]):
            cells = ""
            for age in ages:
                m = seg_df[(seg_df["AGE_GROUP"] == age) & (seg_df["GENDER"].str.lower() == gender)]
                v = float(m.iloc[0]["TOTAL_SCHEDULED_RETURNS"]) if not m.empty else None
                bg, fg = _cell_color(v)
                label = "—" if v is None else f"{int(v)}"
                cells += (f'<td style="background:{bg};padding:8px 4px;text-align:center;'
                          f'border-radius:4px"><div style="font-size:11px;font-weight:600;'
                          f'color:{fg}">{label}</div></td>')
            seg_cell = (
                f'<td rowspan="{n_genders}" style="font-size:11px;font-weight:600;color:{TEXT_PRI};'
                f'padding:6px;white-space:nowrap;vertical-align:middle;width:{_SEG_COL_PX}px">'
                f'{_short(seg)}</td>'
            ) if gi == 0 else ""
            rows_html += (
                f'<tr>{seg_cell}'
                f'<td style="font-size:10px;color:{TEXT_MUT};padding:6px;white-space:nowrap;'
                f'text-transform:capitalize;width:{_GENDER_COL_PX}px">{gender}</td>{cells}</tr>'
            )
    st.markdown(
        f'<div style="overflow-x:auto;-webkit-overflow-scrolling:touch">'
        f'<table style="border-collapse:separate;border-spacing:3px;'
        f'font-family:Inter,sans-serif;width:100%;min-width:520px;table-layout:auto">'
        f'<thead><tr>'
        f'<th style="width:{_SEG_COL_PX}px"></th><th style="width:{_GENDER_COL_PX}px"></th>'
        f'{header}</tr></thead><tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    chart_card_close()

    if "Unknown" in df_scheduled_age["AGE_GROUP"].values:
        unknown_share = df_scheduled_age[df_scheduled_age["AGE_GROUP"] == "Unknown"]["TOTAL_SCHEDULED_RETURNS"].sum()
        total_share = df_scheduled_age["TOTAL_SCHEDULED_RETURNS"].sum()
        _insight(
            [f"'Unknown' age accounts for {fmt_num(int(unknown_share))} of {fmt_num(int(total_share))} "
             "scheduled returns — any age-based reading of this table should account for that gap before "
             "drawing conclusions."],
            variant="warning",
        )


# ── Section 7 — Attendance outcome by segment, then by condition ────────────

_S7_AREA_COLOR = {
    "Core Orthopedics: General": _C_NAVY, "Core Orthopedics: Spine and Back Pain Care": _C_CORAL,
    "ANC": _C_AMBER, "Maternal Health": _C_LIGHT_BLUE, "Core General Surgery": _C_TEAL,
}

_S7_OUTCOME_ORDER = [
    "Showed EARLY",
    "Showed ON TIME / mildly LATE (within 30 days)",
    "Returned, but well beyond scheduled date (30+ days late)",
    "Never returned",
]
_S7_OUTCOME_LABEL = {
    "Showed EARLY": "Early",
    "Showed ON TIME / mildly LATE (within 30 days)": "On time / mildly late",
    "Returned, but well beyond scheduled date (30+ days late)": "Late (30+ days)",
    "Never returned": "Never returned",
}
_S7_OUTCOME_COLOR = {
    "Showed EARLY": _C_TEAL,
    "Showed ON TIME / mildly LATE (within 30 days)": "#8FCFC8",
    "Returned, but well beyond scheduled date (30+ days late)": _C_AMBER,
    "Never returned": _C_CORAL,
}


_S7_ROUTE_ORDER = [
    "Obstetric complication or risk flag",
    "Routine obstetric / pregnancy staging",
    "Orthopedics / trauma",
    "General medicine (unrelated to pregnancy)",
]
_S7_ROUTE_COLOR = {
    "Obstetric complication or risk flag": DANGER,
    "Routine obstetric / pregnancy staging": _C_AMBER,
    "Orthopedics / trauma": _C_NAVY,
    "General medicine (unrelated to pregnancy)": NEUTRAL,
}
# Keyword rules, checked in order — a diagnosis matching an earlier rule
# never falls through to a later one. Complication/risk terms are checked
# first because a label like "Hyperemesis Gravidarum" also contains a
# generic pregnancy word and must not be demoted to "routine staging".
_S7_COMPLICATION_KEYWORDS = [
    "hyperemesis", "ectopic", "pre-eclamp", "preeclamp", "eclamp", "pprom",
    "antepartum haemorrhage", "antepartum hemorrhage", "gestational diabetes",
    "miscarriage", "abortion", "stillbirth", "iugr", "intrauterine growth",
    "hypotension in pregnancy", "headache secondary to pregnancy",
    "gu tract infection", "urinary tract infection in pregnancy",
    "subchorionic haematoma", "subchorionic hematoma", "skin infection",
    "threatened", "placenta praevia", "placenta previa",
]
_S7_ROUTINE_OBSTETRIC_KEYWORDS = [
    "primigravida", "para ", "gravida", "pregnant", "antenatal", "anc ",
    "booking", "gestation",
]
_S7_ORTHO_KEYWORDS = [
    "fracture", "lisfranc", "radius", "ulna", "spine", "sciatica", "injury",
    "dislocation", "sprain", "ligament", "hardware", "orthopedic",
]


def _route_category(diagnosis: str) -> str:
    """Where a diagnosis text would clinically route, independent of the
    ANC-segment text-match that pulled the visit into this extract in the
    first place — used to show that a "never returned" ANC patient isn't
    always a lost pregnancy case."""
    text = str(diagnosis or "").lower()
    if any(k in text for k in _S7_COMPLICATION_KEYWORDS):
        return "Obstetric complication or risk flag"
    if any(k in text for k in _S7_ORTHO_KEYWORDS):
        return "Orthopedics / trauma"
    if any(k in text for k in _S7_ROUTINE_OBSTETRIC_KEYWORDS):
        return "Routine obstetric / pregnancy staging"
    return "General medicine (unrelated to pregnancy)"


# Known free-text misspellings in the raw EMR diagnosis field — corrected
# only for grouping/display purposes here, not in the source data.
_DX_TYPO_FIXES = {
    "ectpic": "ectopic",
    "heamtoma": "haematoma",
    "hemtoma": "haematoma",
    "resistancy": "hesitancy",
    "hesistancy": "hesitancy",
}


def _normalize_dx_key(diagnosis: str) -> str:
    """Collapses cosmetic variants of the same raw diagnosis text (typos,
    a leading Right/Left laterality prefix, a truncated '+ second dx' tail,
    stray punctuation/whitespace) down to one grouping key, so "Ruptured
    Ectpic Pregnancy" and "Right Ruptured Ectopic Preg..." count as the
    same bar instead of two near-duplicates."""
    text = str(diagnosis or "").lower()
    text = text.split(" + ")[0].split("+")[0]
    text = re.sub(r"^\s*(right|left|bilateral)\s+", "", text)
    for wrong, right in _DX_TYPO_FIXES.items():
        text = text.replace(wrong, right)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_dx_display(diagnosis: str) -> str:
    """Same cleanup as _normalize_dx_key (drop the '+ second dx' tail, drop
    a leading laterality prefix, fix known typos) but preserving real
    words/casing for display — _normalize_dx_key strips all punctuation,
    which is fine for grouping but mangles things like '22/40' into
    '2240' if shown as-is."""
    text = str(diagnosis or "").strip()
    text = re.split(r"\s*\+\s*", text)[0].strip()
    text = re.sub(r"(?i)^\s*(right|left|bilateral)\s+", "", text)
    for wrong, right in _DX_TYPO_FIXES.items():
        text = re.sub(re.escape(wrong), right, text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def render_s7(df_outcome: pd.DataFrame, df_anc_never_return: pd.DataFrame = None) -> None:
    section_header("7 — Attendance Outcome by Segment, Then by Condition")
    if not _safe(df_outcome):
        _empty()
        return

    # Part 1 — early / on-time / late / never, by clinical segment. Built
    # from the query's '__SEGMENT_TOTAL__' rows, which are thresholded only
    # per (segment, outcome) — NOT by re-summing the condition-level rows
    # below, which are separately thresholded per (segment, condition,
    # outcome) and would silently drop any 1-2-patient combination before
    # it reached a segment total, understating "showed up" volume for
    # segments split across many condition labels (e.g. ANC).
    seg_rows = df_outcome[df_outcome["CONDITION_GROUP"] == "__SEGMENT_TOTAL__"]
    by_seg = (
        seg_rows.groupby(["PRIMARY_VISIT_SEGMENT", "ATTENDANCE_OUTCOME"])["TOTAL_PATIENTS"]
        .sum().reset_index()
    )
    seg_totals = by_seg.groupby("PRIMARY_VISIT_SEGMENT")["TOTAL_PATIENTS"].transform("sum")
    by_seg["PCT"] = 100.0 * by_seg["TOTAL_PATIENTS"] / seg_totals
    seg_order = by_seg.groupby("PRIMARY_VISIT_SEGMENT")["TOTAL_PATIENTS"].sum().sort_values().index.tolist()

    # Part 2 — never-return rate by condition, scoped to the segment each
    # condition actually occurred in (condition names can repeat across
    # segments — e.g. Hip Replacement can occur under more than one
    # classification, since segment and condition are two independent
    # text-match classifiers reading the same diagnosis field — so each
    # segment gets its own bar, side by side, per condition). Excludes the
    # '__SEGMENT_TOTAL__' rows used for Part 1 above.
    df_condition_detail = df_outcome[df_outcome["CONDITION_GROUP"] != "__SEGMENT_TOTAL__"]
    grp = df_condition_detail.groupby(["PRIMARY_VISIT_SEGMENT", "CONDITION_GROUP"])

    def _never_stats(g: pd.DataFrame) -> pd.Series:
        never_n = int(g.loc[g["ATTENDANCE_OUTCOME"] == "Never returned", "TOTAL_PATIENTS"].sum())
        total_n = int(g["TOTAL_PATIENTS"].sum())
        return pd.Series({
            "NEVER_RETURN_PCT": 100.0 * never_n / total_n if total_n else 0.0,
            "NEVER_N": never_n, "TOTAL_N": total_n,
        })

    never_rate = grp.apply(_never_stats).reset_index()
    never_rate = never_rate[never_rate["NEVER_RETURN_PCT"] > 0].copy()
    # Segment leads the label, not the condition name — a condition that
    # shares a text string across segments (e.g. "Long Bone Fracture
    # Fixation" tagged under both Ortho General and Spine and Back Pain
    # Care) is not the same clinical population in each: the Spine and
    # Back Pain Care cases carry a spine-related diagnosis alongside the
    # fixation, which is exactly what put them in that segment. Naming
    # the segment first makes that a genuinely distinct row, not a
    # repeated label with a footnote.
    never_rate["LABEL"] = never_rate.apply(
        lambda r: f"{_short(r['PRIMARY_VISIT_SEGMENT'])} · {r['CONDITION_GROUP']}", axis=1
    )
    never_rate = never_rate.sort_values("NEVER_RETURN_PCT")
    colors = [_S7_AREA_COLOR.get(s, NEUTRAL) for s in never_rate["PRIMARY_VISIT_SEGMENT"]]

    # Both charts in this Pattern B row share one height, sized to whichever
    # has more rows — a fixed 260px was squeezing the condition chart's
    # bars down to unreadable slivers.
    n_rows = max(len(seg_order), len(never_rate), 1)
    paired_h = max(_H_PAIRED, 32 * n_rows + 90)

    col_l, col_r = st.columns(2)

    with col_l:
        chart_card("Attendance outcome by segment",
                   "Share of scheduled follow-ups, by when the patient actually returned")
        seg_n = {s: int(seg_totals[by_seg["PRIMARY_VISIT_SEGMENT"] == s].iloc[0]) for s in seg_order}
        fig1 = go.Figure()
        for outcome in _S7_OUTCOME_ORDER:
            sub = by_seg[by_seg["ATTENDANCE_OUTCOME"] == outcome].set_index("PRIMARY_VISIT_SEGMENT")
            y = [float(sub.loc[s, "PCT"]) if s in sub.index else 0.0 for s in seg_order]
            counts = [int(sub.loc[s, "TOTAL_PATIENTS"]) if s in sub.index else 0 for s in seg_order]
            customdata = list(zip(counts, [seg_n[s] for s in seg_order]))
            fig1.add_trace(go.Bar(
                y=[_short(s) for s in seg_order], x=y, name=_S7_OUTCOME_LABEL[outcome], orientation="h",
                marker=dict(color=_S7_OUTCOME_COLOR[outcome], cornerradius=3),
                text=[f"{v:.0f}%" if v >= 8 else "" for v in y], textposition="inside",
                textfont=dict(size=9, color="#FFFFFF"),
                customdata=customdata,
                hovertemplate=(
                    f"{_S7_OUTCOME_LABEL[outcome]}<br>%{{customdata[0]}} of %{{customdata[1]}} patients "
                    "(%{x:.0f}%)<extra></extra>"
                ),
            ))
        fig1.update_layout(
            **{**_LAYOUT, "height": paired_h, "barmode": "stack", "bargap": 0.35,
               "margin": dict(t=8, b=60, l=110, r=20),
               "legend": dict(orientation="h", y=-0.16, x=0.5, xanchor="center", font=dict(size=9))},
            xaxis={**AXIS_Y, "ticksuffix": "%", "range": [0, 100]},
            yaxis={**AXIS_X, "showgrid": False, "automargin": True},
        )
        st.plotly_chart(fig1, use_container_width=True, config=PC_CFG)
        chart_card_close()

    with col_r:
        chart_card("Never-return rate by condition")
        rows_html = ""
        for _, r in never_rate.sort_values("NEVER_RETURN_PCT", ascending=False).iterrows():
            seg_color = _S7_AREA_COLOR.get(r["PRIMARY_VISIT_SEGMENT"], NEUTRAL)
            pct = float(r["NEVER_RETURN_PCT"])
            never_n, total_n = int(r["NEVER_N"]), int(r["TOTAL_N"])
            tooltip = f'{never_n} of {total_n} patients never returned'
            rows_html += (
                f'<div title="{tooltip}" style="display:flex;align-items:center;gap:12px;margin-bottom:12px">'
                f'<div style="width:170px;flex-shrink:0;text-align:right">'
                f'<div style="font-size:9px;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:.03em;color:{seg_color}">{_short(r["PRIMARY_VISIT_SEGMENT"])}</div>'
                f'<div style="font-size:12px;font-weight:700;color:{TEXT_PRI};line-height:1.3">'
                f'{r["CONDITION_GROUP"]}</div>'
                f'</div>'
                f'<div style="flex:1;background:#F4F6FA;border-radius:8px;height:14px;overflow:hidden">'
                f'<div style="background:{seg_color};height:14px;border-radius:8px;width:{pct}%"></div>'
                f'</div>'
                f'<div style="width:64px;flex-shrink:0;font-size:12px;font-weight:700;color:{TEXT_PRI}">'
                f'{pct:.0f}% <span style="font-size:10px;font-weight:400;color:{TEXT_MUT}">'
                f'({never_n}/{total_n})</span></div>'
                f'</div>'
            )
        legend_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:10px;'
            f'color:{TEXT_MUT};margin-right:14px"><span style="width:10px;height:10px;border-radius:2px;'
            f'background:{_S7_AREA_COLOR[s]};display:inline-block"></span>{_short(s)}</span>'
            for s in _S7_AREA_COLOR if s in never_rate["PRIMARY_VISIT_SEGMENT"].values
        )
        st.markdown(
            f'<div style="padding-top:4px">{rows_html}</div>'
            f'<div style="margin-top:8px">{legend_html}</div>',
            unsafe_allow_html=True,
        )
        chart_card_close()

    bullets = []
    never_by_seg = by_seg[by_seg["ATTENDANCE_OUTCOME"] == "Never returned"].sort_values("PCT", ascending=False)
    if not never_by_seg.empty:
        w = never_by_seg.iloc[0]
        bullets.append(
            f"{_short(w['PRIMARY_VISIT_SEGMENT'])} has the highest never-return rate ({w['PCT']:.1f}%) of "
            "any segment tracked."
        )
    if not never_rate.empty:
        worst = never_rate.sort_values("NEVER_RETURN_PCT", ascending=False).iloc[0]
        bullets.append(
            f"{worst['CONDITION_GROUP']} has the highest never-return rate ({worst['NEVER_RETURN_PCT']:.1f}%) "
            f"of any condition tracked — in {_short(worst['PRIMARY_VISIT_SEGMENT'])}, where a missed "
            "follow-up carries real clinical stakes if it involves surgical hardware or an infection risk."
        )
    _insight(bullets or ["Attendance outcome data is limited."], variant="danger")

    if _safe(df_anc_never_return):
        n = len(df_anc_never_return)
        df_anc_never_return = df_anc_never_return.copy()
        df_anc_never_return["ROUTE_CATEGORY"] = df_anc_never_return["INDEX_DIAGNOSIS"].apply(_route_category)

        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{TEXT_PRI};margin-top:20px">'
            f'Who these {n} "lost" ANC patients actually were, by diagnosis</div>'
            f'<div style="font-size:11px;font-style:italic;color:{TEXT_MUT};margin-bottom:14px">'
            'Grouped by where the diagnosis would clinically route — not all of these are '
            'obstetric losses</div>',
            unsafe_allow_html=True,
        )

        def _render_cat_chart(cat: str) -> None:
            sub = df_anc_never_return[df_anc_never_return["ROUTE_CATEGORY"] == cat]
            cat_n = len(sub)
            # Raw EMR diagnosis text carries free-text noise (misspellings,
            # a "Right"/"Left" laterality prefix, a truncated "+ ..." second
            # diagnosis) that splits what is really one condition into
            # several near-duplicate bars — e.g. "Ruptured Ectpic Pregnancy"
            # vs "Right Ruptured Ectopic Preg...". Group on a fully-stripped
            # key, display a lightly-cleaned (not fully stripped) label so
            # real content like "22/40" survives.
            norm_dx = sub["INDEX_DIAGNOSIS"].apply(_normalize_dx_key)
            display_label = sub["INDEX_DIAGNOSIS"].apply(_clean_dx_display)
            group_label = display_label.groupby(norm_dx).agg(lambda s: min(s, key=len))
            dx_counts = norm_dx.value_counts()
            dx_counts.index = dx_counts.index.map(group_label)
            # Largest-first for the donut (Plotly draws pie slices in trace
            # order starting at 12 o'clock). Distinct on-palette colors, not
            # a same-hue ramp — a gradient reads as size, not identity.
            dx_counts = dx_counts.sort_values(ascending=False)
            colors = [_CATEGORICAL_PALETTE[i % len(_CATEGORICAL_PALETTE)] for i in range(len(dx_counts))]
            text_colors = [_text_color_for(c) for c in colors]

            chart_card(cat, f"{cat_n} of {n} patients")
            fig = go.Figure(go.Pie(
                labels=dx_counts.index, values=dx_counts.values, hole=0.55,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=1)),
                textinfo="value", textfont=dict(size=9, color=text_colors),
                hovertemplate="%{label}: %{value} patients<extra></extra>",
            ))
            # Fixed height for every category, regardless of slice count —
            # so the four cards in this grid all read as the same size. Kept
            # small since this is a supporting detail, not the section's
            # focal point.
            fig.update_layout(
                **{**_LAYOUT, "height": 130,
                   "margin": dict(t=4, b=4, l=4, r=4),
                   "legend": dict(orientation="v", x=1.0, y=0.5, font=dict(size=11))},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

        present_cats = [c for c in _S7_ROUTE_ORDER if not df_anc_never_return[
            df_anc_never_return["ROUTE_CATEGORY"] == c].empty]
        for row_start in range(0, len(present_cats), 2):
            row_cats = present_cats[row_start:row_start + 2]
            cols = st.columns(2, gap="medium")
            for col, cat in zip(cols, row_cats):
                with col:
                    _render_cat_chart(cat)
                    # Row gap — without this, two stacked chart_card() rows
                    # sit border-to-border with no visual separation.
                    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        complication_flag_n = int(
            (df_anc_never_return["ROUTE_CATEGORY"] == "Obstetric complication or risk flag").sum()
        )
        non_complication_n = n - complication_flag_n
        complication_share_pct = round(100.0 * complication_flag_n / n, 0) if n else 0.0
        _insight(
            [f"Does this mean the hospital is losing patients who were at risk? Yes — partly.",
             f"{complication_flag_n} of {n} ({complication_share_pct:.0f}%) had a genuine pregnancy "
             "complication or risk flag — that's the group that matters clinically.",
             f"The other {non_complication_n} are pregnant women (or non-pregnancy cases pulled in by "
             "the same text-match) who came in for something else entirely and were swept into this "
             "ANC extract by classification, not by their actual reason for visiting — treat those as "
             "a labeling issue, not a retention failure."],
            variant="danger",
        )


# ── Section 9 — Clinician scheduling-rate spread (Pattern A) ────────────────

def render_s8(df_clinician: pd.DataFrame) -> None:
    section_header("9 — Clinician Scheduling-Rate Spread")
    if not _safe(df_clinician):
        _empty()
        return

    df = df_clinician.sort_values("TOTAL_CONSULTATIONS", ascending=False).reset_index(drop=True)
    df["ANON_LABEL"] = [f"Clinician {i+1}" for i in range(len(df))]

    chart_card(
        "Scheduling rate vs. consultation volume",
        "One point per clinician (≥20 consultations) · Dashed line = 80% target",
    )
    st.markdown(
        f'<div style="background:#FAEEDA;border:1px solid #EF9F27;border-radius:6px;'
        f'padding:8px 12px;margin-bottom:12px;font-size:11px;line-height:1.5;color:{WARNING}">'
        "No external benchmark exists for this specific metric (confirmed by search — published "
        "thresholds cover appointment fill rate, no-show rate, and capacity utilization, not "
        "follow-up scheduling behavior). These three tiers are a pragmatic internal split based on "
        "distance from the 80% target, not an industry standard.</div>",
        unsafe_allow_html=True,
    )
    # Exact status-system hex per spec §3/§4 (scatter markers use border tones)
    _tiers = [
        ("Meets target", "≥80%", "#639922", lambda p: p >= 80),
        ("Developing", "40–79%", "#EF9F27", lambda p: 40 <= p < 80),
        ("Needs review", "<40%", "#E24B4A", lambda p: p < 40),
    ]
    st.markdown(
        '<div style="display:flex;gap:18px;margin-bottom:10px">' + "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:{TEXT_SEC}">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:{color};display:inline-block">'
            f'</span>{label} · {rng}</span>'
            for label, rng, color, _ in _tiers
        ) + '</div>',
        unsafe_allow_html=True,
    )
    fig = go.Figure()
    fig.add_hline(y=80, line=dict(color=TEXT_MUT, width=1.3, dash="dash"))
    for label, _, color, cond in _tiers:
        sub = df[df["PCT_SCHEDULED"].apply(cond)]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["TOTAL_CONSULTATIONS"], y=sub["PCT_SCHEDULED"], mode="markers",
            name=label, text=sub["ANON_LABEL"],
            marker=dict(size=11, color=color, opacity=0.9, line=dict(width=1, color="#FFFFFF")),
            hovertemplate="%{text}<br>Consultations: %{x}<br>Scheduled: %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(**{**_LAYOUT, "height": _H_SINGLE}, showlegend=False,
                       xaxis={**AXIS_X, "title": "Number of consultations"},
                       yaxis={**AXIS_Y, "ticksuffix": "%", "title": "% scheduled", "range": [0, 100]})
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
    chart_card_close()

    # Converts the scatter into the number that actually maps to revenue —
    # consultations × the gap to the 80% target — so "this clinician is at
    # 45%" and "this clinician is at 45% of 6,000 consultations" don't read
    # as the same-size problem when they aren't.
    df["MISSED_FOLLOWUPS"] = (df["TOTAL_CONSULTATIONS"] * (80 - df["PCT_SCHEDULED"]).clip(lower=0) / 100).round(0)
    total_missed = df["MISSED_FOLLOWUPS"].sum()
    if total_missed > 0:
        top_missed = df.sort_values("MISSED_FOLLOWUPS", ascending=False).head(10)
        top_missed = top_missed[top_missed["MISSED_FOLLOWUPS"] > 0].sort_values("MISSED_FOLLOWUPS", ascending=True)
        chart_card(
            "Estimated missed follow-up encounters, by clinician",
            "Consultations × gap to the 80% target — the count that actually maps to lost follow-up revenue, "
            "not just the % gap",
        )
        tier_color = {t[0]: t[2] for t in _tiers}
        bar_colors = [
            tier_color["Meets target"] if p >= 80 else tier_color["Developing"] if p >= 40 else tier_color["Needs review"]
            for p in top_missed["PCT_SCHEDULED"]
        ]
        fig_missed = go.Figure(go.Bar(
            y=top_missed["ANON_LABEL"], x=top_missed["MISSED_FOLLOWUPS"], orientation="h",
            marker=dict(color=bar_colors, cornerradius=3),
            text=[f"{int(v):,}" for v in top_missed["MISSED_FOLLOWUPS"]], textposition="outside",
            customdata=top_missed[["TOTAL_CONSULTATIONS", "PCT_SCHEDULED"]].values,
            hovertemplate="%{y}<br>%{customdata[0]:,.0f} consultations, %{customdata[1]:.1f}% scheduled"
                          "<br>%{x:,.0f} missed follow-ups<extra></extra>",
        ))
        fig_missed.update_layout(
            **{**_LAYOUT, "height": max(220, 30 * len(top_missed) + 60), "showlegend": False},
            xaxis={**AXIS_Y, "showgrid": True, "title": "Estimated missed follow-ups"},
            yaxis={**AXIS_X, "showgrid": False, "automargin": True},
        )
        st.plotly_chart(fig_missed, use_container_width=True, config=PC_CFG)
        chart_card_close()

    bullets = []
    if total_missed > 0:
        top4_missed_share = 100.0 * df.sort_values("TOTAL_CONSULTATIONS", ascending=False).head(4)["MISSED_FOLLOWUPS"].sum() / total_missed
        bullets.append(
            f"<strong>Estimated {int(total_missed):,} missed follow-up encounters</strong> across all "
            f"clinicians below the 80% target — {top4_missed_share:.0f}% of that gap sits with just the "
            "4 highest-volume clinicians, so that's where fixing scheduling recovers the most volume, "
            "even though their % gap looks similar to everyone else's."
        )
    top4 = df.head(4)
    if not top4.empty:
        bullets.append(
            f"Among the {len(top4)} highest-volume clinicians alone ({fmt_num(int(top4['TOTAL_CONSULTATIONS'].min()))}+ "
            f"consultations each), scheduling rates range from {top4['PCT_SCHEDULED'].min():.1f}% to "
            f"{top4['PCT_SCHEDULED'].max():.1f}%. This is an individual-practice pattern, not a "
            "hospital-wide standard — closing this gap is a process fix, not a staffing fix."
        )
    outliers = df[df["PCT_SCHEDULED"] >= df["PCT_SCHEDULED"].quantile(0.95)]
    if not outliers.empty:
        top_outlier = outliers.sort_values("PCT_SCHEDULED", ascending=False).iloc[0]
        bullets.append(
            f"{top_outlier['ANON_LABEL']} ({fmt_num(int(top_outlier['TOTAL_CONSULTATIONS']))} consultations, "
            f"{top_outlier['PCT_SCHEDULED']:.1f}% scheduled) is worth a direct follow-up conversation — "
            "their practice may represent what's achievable elsewhere."
        )
    needs_review = df[df["PCT_SCHEDULED"] < 40]
    if not needs_review.empty:
        bullets.append(
            f"{len(needs_review)} clinician(s) fall in the 'needs review' tier (<40% scheduled) — "
            "worth confirming whether that's a workflow gap or a data-entry gap before assuming it's "
            "a scheduling problem."
        )
    _insight(bullets, variant="info")

    if not needs_review.empty:
        # Real clinician ID here, not the anonymized ANON_LABEL used on-screen
        # — this list exists to be acted on, so it needs to identify who to
        # follow up with, not just how many.
        export_cols = ["FILLED_BY_USER_ID", "TOTAL_CONSULTATIONS", "PCT_SCHEDULED", "MISSED_FOLLOWUPS"]
        needs_review_export = (
            needs_review[[c for c in export_cols if c in needs_review.columns]]
            .rename(columns={"FILLED_BY_USER_ID": "CLINICIAN_ID"})
            .sort_values("MISSED_FOLLOWUPS", ascending=False)
        )
        st.download_button(
            "Download 'needs review' clinician list (CSV)",
            data=needs_review_export.to_csv(index=False).encode("utf-8"),
            file_name="clinicians_needing_scheduling_review.csv",
            mime="text/csv",
            key="fr_s8_needs_review_download",
        )


# ── Section 8 — The counter-example (Pattern A / stat-only) ─────────────────

def render_s9(df_gs: pd.DataFrame) -> None:
    section_header("8 — The Counter-Example")
    if not _safe(df_gs):
        _empty()
        return

    row = df_gs.iloc[0]
    pct_scheduled = float(row["PCT_SCHEDULED"])
    pct_returned_anyway = float(row["PCT_UNSCHEDULED_RETURNED_ANYWAY"] or 0)
    total_visits = int(row["TOTAL_GS_VISITS"])

    col_a, col_b = st.columns(2)
    with col_a:
        chart_card("Scheduled follow-up rate")
        st.markdown(
            f'<div style="text-align:center;padding:20px 0">'
            f'<div style="font-size:44px;font-weight:700;color:{_C_CORAL}">{pct_scheduled:.1f}%</div>'
            f'<div style="font-size:12px;color:{TEXT_MUT};margin-top:4px">of {fmt_num(total_visits)} '
            'General Surgery visits had a scheduled follow-up</div></div>',
            unsafe_allow_html=True,
        )
        chart_card_close()
    with col_b:
        chart_card("Unscheduled patients who returned anyway")
        st.markdown(
            f'<div style="text-align:center;padding:20px 0">'
            f'<div style="font-size:44px;font-weight:700;color:{_C_TEAL}">{pct_returned_anyway:.1f}%</div>'
            f'<div style="font-size:12px;color:{TEXT_MUT};margin-top:4px">of unscheduled patients '
            'returned anyway</div></div>',
            unsafe_allow_html=True,
        )
        chart_card_close()

    _insight(
        [f"Most General Surgery patients return on their own even without a scheduled prompt "
         f"({pct_returned_anyway:.1f}% of the {100-pct_scheduled:.1f}% who weren't scheduled came back "
         "anyway) — the gap here is that the hospital isn't scheduling them, not that patients won't "
         "come back. This is a system gap, not a patient-willingness gap."],
        variant="success",
    )


# ── Closing synthesis ────────────────────────────────────────────────────────

# Matches ui_template.py's _PRIORITY_SEVERITY_COLOR so the "Action:" line
# inside a card highlights in the same color as that card's left border/label.
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


def render_synthesis(df_outreach: pd.DataFrame, df_outcome: pd.DataFrame,
                      df_clinician: pd.DataFrame, df_status_overall: pd.DataFrame = None,
                      df_visit_number: pd.DataFrame = None, df_gs: pd.DataFrame = None,
                      df_visit_number_by_segment: pd.DataFrame = None,
                      df_patient_signals: pd.DataFrame = None) -> None:
    n_outreach = len(df_outreach) if _safe(df_outreach) else 0

    hip_pct, hip_never, hip_total = None, None, None
    if _safe(df_outcome):
        hip = df_outcome[df_outcome["CONDITION_GROUP"] == "Hip Replacement"]
        if not hip.empty:
            hip_total = int(hip["TOTAL_PATIENTS"].sum())
            hip_never = int(hip.loc[hip["ATTENDANCE_OUTCOME"] == "Never returned", "TOTAL_PATIENTS"].sum())
            hip_pct = round(100.0 * hip_never / hip_total, 1) if hip_total else 0.0

    meets_target_n = needs_review_n = total_clinicians = None
    total_missed_hospitalwide = None
    if _safe(df_clinician):
        total_clinicians = len(df_clinician)
        meets_target_n = int((df_clinician["PCT_SCHEDULED"] >= 80).sum())
        needs_review_n = int((df_clinician["PCT_SCHEDULED"] < 40).sum())
        gap = (80 - df_clinician["PCT_SCHEDULED"]).clip(lower=0) / 100
        total_missed_hospitalwide = int((df_clinician["TOTAL_CONSULTATIONS"] * gap).sum())

    p1_items = [
        f"{n_outreach} patients confirmed via chart review to be post-spine-surgery and currently LTFU.",
        "The clearest, smallest, most tractable list in this dataset — a direct outreach exercise, not a "
        "system-level fix.",
        "Action: assign this list to case management this week; expect measurable closure within days.",
    ]

    p2_items = []
    if hip_pct is not None:
        p2_items.append(f"{hip_pct:.1f}% of Hip Replacement patients never return for a scheduled "
                         f"follow-up ({hip_never} of {hip_total}).")
    p2_items.append("A missed follow-up here carries real infection and implant-failure risk — this is a "
                     "clinical safety gap, not just a retention metric.")
    p2_items.append("Action: audit Hip Replacement discharge protocol to confirm every patient leaves "
                     "with an actual scheduled date, not just a verbal instruction.")

    p3_items = []
    if total_clinicians:
        p3_items.append(f"Only {meets_target_n} of {total_clinicians} clinicians (≥20 consultations) "
                         f"meet the 80% scheduling target; {needs_review_n} fall below 40%.")
    if total_missed_hospitalwide:
        p3_items.append(f"Estimated {fmt_num(total_missed_hospitalwide)} missed follow-up encounters "
                         "hospital-wide from clinicians below the 80% target.")
    p3_items.append("Action: make scheduling a required checklist step at visit close, not a "
                     "memory-dependent habit — closes most of this gap before it starts.")

    p4_items = [
        "No \"Schedule Follow Up\" record exists past a fixed point in the source data — this affects "
        "every \"scheduled\" calculation on this tab for the most recent period.",
        "The most recent months should not be read as evidence scheduling stopped happening — a "
        "data-capture gap is the more likely explanation.",
        "Action: confirm directly with clinical ops whether scheduling is still being recorded "
        "consistently in the source system.",
    ]

    # LTFU rate/flow + the General Surgery counter-example — ties together
    # "how bad is it and where does it concentrate" with "scheduling isn't
    # the same fix everywhere," which the first four cards don't otherwise
    # connect.
    ltfu_pct = None
    if _safe(df_status_overall):
        ltfu_row = df_status_overall[df_status_overall["STATUS"] == "LTFU"]
        if not ltfu_row.empty:
            ltfu_pct = float(ltfu_row.iloc[0]["PCT_OF_CLASSIFIABLE_PATIENTS"])

    visit1_pct = None
    if _safe(df_visit_number):
        v_total = df_visit_number["TOTAL_PATIENTS"].sum()
        v1_row = df_visit_number[df_visit_number["LTFU_AT_VISIT_NUMBER"] == "1"]
        if v_total and not v1_row.empty:
            visit1_pct = round(100.0 * float(v1_row.iloc[0]["TOTAL_PATIENTS"]) / v_total, 1)

    gs_scheduled_pct = gs_returned_pct = None
    if _safe(df_gs):
        gs_row = df_gs.iloc[0]
        gs_scheduled_pct = float(gs_row["PCT_SCHEDULED"])
        gs_returned_pct = float(gs_row["PCT_UNSCHEDULED_RETURNED_ANYWAY"] or 0)

    p5_items = []
    if ltfu_pct is not None or visit1_pct is not None:
        parts = []
        if ltfu_pct is not None:
            parts.append(f"{ltfu_pct:.1f}% of tracked patients are currently LTFU hospital-wide")
        if visit1_pct is not None:
            parts.append(f"{visit1_pct:.1f}% of all LTFU losses happen after just one visit")
        p5_items.append(", ".join(parts).capitalize() + " — concentrated, front-loaded attrition, not a "
                         "slow gradual drift.")
    if gs_scheduled_pct is not None and gs_returned_pct is not None:
        p5_items.append(
            f"General Surgery: only {gs_scheduled_pct:.1f}% of visits get a scheduled follow-up, yet "
            f"{gs_returned_pct:.1f}% of unscheduled patients return anyway — patients are largely bringing "
            "themselves back without the hospital's help."
        )
    p5_items.append(
        "Action: don't apply one scheduling fix everywhere — prioritize segments where patients do NOT "
        "self-return (Hip Replacement, ANC) over General Surgery, where they already do."
    )

    # Same figures as Section 2's "Where the Problem Concentrates" insight
    # bar — the single largest, least-explained volume in this tab, and it
    # was missing entirely from this closing summary.
    def _v1_pct(segment: str):
        if not _safe(df_visit_number_by_segment):
            return None
        row = df_visit_number_by_segment[
            (df_visit_number_by_segment["SEGMENT"] == segment)
            & (df_visit_number_by_segment["LTFU_AT_VISIT_NUMBER"] == "1")
        ]
        return float(row.iloc[0]["PCT_WITHIN_SEGMENT"]) if not row.empty else None

    n_no_trace_all = None
    if _safe(df_patient_signals):
        v1_all = df_patient_signals[df_patient_signals["VISIT_NUMBER_AT_LTFU"] == "1"]
        n_no_trace_all = int((v1_all["HAS_LATER_VISIT_ELSEWHERE"] == 0).sum())

    high_conc = [
        (n, v) for n, v in
        [("Spine (structural)", _v1_pct("Spine-structural")),
         ("ANC", _v1_pct("ANC / Routine Pregnancy")),
         ("Fibroids", _v1_pct("Fibroids-conservative"))]
        if v is not None
    ]

    p6_items = []
    if high_conc:
        p6_items.append(
            "One-and-done attendance, not gradual drop-off, drives most loss: " +
            ", ".join(f"{n} {v:.0f}%" for n, v in high_conc) +
            " of losses happen right after visit 1."
        )
    if n_no_trace_all:
        p6_items.append(
            f"Hospital-wide, {fmt_num(n_no_trace_all)} visit-1 losses have no later visit anywhere — "
            "an order of magnitude larger than any single segment shown elsewhere on this tab, and mostly "
            "outside the highlighted segments (e.g. Core Orthopedics: General)."
        )
    p6_items.append(
        "Action: pull the visit-1 patient-level export from Section 2 and audit it segment by segment, "
        "starting with 'No subsequent visit at all' — this dwarfs every other list on this tab."
    )

    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;'
        f'color:{TEXT_MUT};margin:16px 0 10px;padding-bottom:10px;border-bottom:1px solid {BORDER}">'
        'What this tab found, and what to do next</div>',
        unsafe_allow_html=True,
    )
    priority_cards([
        {"label": "PRIORITY 1", "severity": "critical",
         "title": f"Contact the {n_outreach} verified post-spine-surgery patients",
         "body": _synthesis_list(p1_items, "critical")},
        {"label": "PRIORITY 2", "severity": "critical",
         "title": "Fix Hip Replacement follow-up specifically",
         "body": _synthesis_list(p2_items, "critical")},
        {"label": "PRIORITY 3", "severity": "monitor",
         "title": "Standardize scheduling at the point of care",
         "body": _synthesis_list(p3_items, "monitor")},
        {"label": "PRIORITY 4", "severity": "monitor",
         "title": "Confirm current scheduling practice directly",
         "body": _synthesis_list(p4_items, "monitor")},
        {"label": "PRIORITY 5", "severity": "monitor",
         "title": "LTFU is front-loaded, and scheduling isn't the same fix everywhere",
         "body": _synthesis_list(p5_items, "monitor")},
        {"label": "PRIORITY 6", "severity": "critical",
         "title": "Audit the largest, least-explained LTFU volume on this tab",
         "body": _synthesis_list(p6_items, "critical")},
    ])


# ── Tab entry point ───────────────────────────────────────────────────────────

def render_tab() -> None:
    import sph.clinicals.flow_retention_module.fr_queries as FRQ

    with st.spinner("Loading data…"):
        df_status_overall = FRQ.get_fr_status_overall()
        df_status_by_segment = FRQ.get_fr_status_by_segment()
        df_trend = FRQ.get_fr_retention_trend()
        df_visit_number = FRQ.get_fr_ltfu_by_visit_number()
        df_visit_number_by_segment = FRQ.get_fr_ltfu_by_segment_and_visit_number()
        df_patient_signals = FRQ.get_fr_ltfu_patient_level_signals()
        df_pathway = FRQ.get_fr_ltfu_last_pathway()
        df_ltfu_share = FRQ.get_fr_ltfu_share_by_segment_age_gender()
        df_scheduled_age = FRQ.get_fr_scheduled_returns_by_age()
        df_condition = FRQ.get_fr_ltfu_condition_breakdown()
        df_lost_v1 = FRQ.get_fr_lost_after_visit1()
        df_outreach = FRQ.get_fr_spine_structural_outreach_list()
        df_scheduled = FRQ.get_fr_scheduled_vs_organic()
        df_outcome = FRQ.get_fr_attendance_outcome()
        df_anc_never_return = FRQ.get_fr_anc_never_return_profile()
        df_clinician = FRQ.get_fr_clinician_scheduling_rate()
        df_gs = FRQ.get_fr_general_surgery_counterexample()

    render_tab_header()
    render_kpis(df_status_overall, df_visit_number)
    render_s1(df_trend)
    render_s3(df_status_by_segment, df_visit_number, df_visit_number_by_segment, df_patient_signals)
    render_s2(df_ltfu_share, df_pathway)
    render_s4(df_condition)
    render_s5(df_outreach, df_lost_v1, df_condition)
    render_section_b_header(df_scheduled)
    render_s6(df_scheduled, df_scheduled_age)
    render_s7(df_outcome, df_anc_never_return)
    render_s9(df_gs)
    render_s8(df_clinician)
    render_synthesis(
        df_outreach, df_outcome, df_clinician, df_status_overall, df_visit_number, df_gs,
        df_visit_number_by_segment, df_patient_signals,
    )
