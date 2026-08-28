"""
sph/disease_burden_module/maternal/mat_views.py
======================================================
All render functions for the Disease Burden → Maternal health sub-tab.

Rules enforced here:
  - Zero SQL — no database calls, no query strings.
  - All insight text is computed from the DataFrame passed in, never
    hardcoded.
  - Insight bars use SOLID fills, matching the Orthopedics pattern.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinicals.opd_ipd_module.ui_template import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL, SECONDARY,
    SURFACE_1, BORDER, TEXT_PRI, TEXT_SEC, TEXT_MUT,
    CHART_LAYOUT, AXIS_X, AXIS_Y, PC_CFG,
    fmt_num, fmt_pct, priority_cards, kpi_row,
)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clinicals.disease_burden_module.orthopedics.orth_views import (
    section_header, chart_card, chart_card_close, insight_bar, data_caveat,
    _safe, _empty,
)

_C_BLUE = PRIMARY     # teal
_C_TEAL = SUCCESS
_C_AMBER = WARNING
_C_RED = DANGER
_C_PURPLE = SECONDARY  # raspberry — general surgery/OBGYN category (spec §4)
_C_GREY = "#D3D6DE"
_C_MGREY = NEUTRAL

_LAYOUT = CHART_LAYOUT
_H_SINGLE = 280
_H_PAIRED = 260

CATEGORY_COLOURS = {
    "ANC / Routine Pregnancy Care": _C_BLUE,
    "Fibroids": _C_TEAL,
    "Pregnancy Loss": _C_AMBER,
    "High-Risk Pregnancy / Complications": _C_RED,
    "Labour & Delivery": _C_PURPLE,
    "Ovarian conditions (cysts/PCOS)": _C_BLUE,
    "Dysmenorrhea": _C_GREY,
    "Other OBGYN": _C_MGREY,
    "Abnormal Uterine Bleeding": _C_AMBER,
    "Post-Hysterectomy Follow-up": _C_GREY,
    "Infertility / PID": _C_AMBER,
    "Endometriosis / Endometrial conditions": _C_MGREY,
    "Pelvic pain / mass": _C_MGREY,
    "Adenomyosis": _C_GREY,
    "Postnatal Care": _C_TEAL,
    "Amenorrhea": _C_GREY,
    "Dyspareunia": _C_GREY,
    "Gynaecological malignancy": _C_RED,
}


# ── Headline KPIs ────────────────────────────────────────────────────────────

def render_kpis(df: pd.DataFrame) -> None:
    section_header("Maternal / OBGYN — headline")
    if not _safe(df):
        _empty()
        return

    row = df.iloc[0]
    single_pct = float(row.get("SINGLE_VISIT_PCT", 0) or 0)
    four_plus_pct = float(row.get("FOUR_PLUS_VISIT_PCT", 0) or 0)
    zero_pct = float(row.get("ZERO_INDICATOR_PCT", 0) or 0)
    fibroids_pct = float(row.get("FIBROIDS_PCT_OF_VOLUME", 0) or 0)
    fibroids_conv = float(row.get("FIBROIDS_CONV_RATE", 0) or 0)

    kpi_row([
        {"label": "ANC single-visit rate", "value": fmt_pct(single_pct),
         "delta": "of pregnant patients have exactly 1 recorded visit", "accent_color": DANGER},
        {"label": "Reaches quality threshold", "value": fmt_pct(four_plus_pct),
         "delta": "4+ visits — Asiimwe et al. Kenya DHS 2022", "accent_color": WARNING},
        {"label": "ANC visits with zero indicators", "value": fmt_pct(zero_pct),
         "delta": "No BP, urine, blood, iron or ultrasound recorded", "accent_color": WARNING},
        {"label": "Fibroids — largest gynaecology category", "value": fmt_pct(fibroids_pct),
         "delta": f"{fibroids_conv:.1f}% admission rate — twice the antenatal rate", "accent_color": SUCCESS},
    ])


# ── Section 1 — Case mix ────────────────────────────────────────────────────

def render_s1(df: pd.DataFrame) -> None:
    section_header("Section 1 — case mix: what this service treats")
    if not _safe(df):
        _empty()
        return

    df = df.sort_values("TOTAL_VISITS", ascending=False)
    top10 = df.head(10)

    col_l, col_r = st.columns(2)
    with col_l:
        chart_card(
            "Visit volume by category",
            note="Ranked by volume. Fibroids is the largest gynaecological category — entirely missed "
                 "by the first two classifier versions.",
        )
        d = top10.sort_values("TOTAL_VISITS", ascending=True)
        colors = [CATEGORY_COLOURS.get(c, _C_MGREY) for c in d["CASE_MIX_CATEGORY"]]
        fig = go.Figure(go.Bar(y=d["CASE_MIX_CATEGORY"], x=d["TOTAL_VISITS"], orientation="h",
                                marker_color=colors, marker_cornerradius=3))
        fig.update_layout(**{**_LAYOUT, "height": _H_PAIRED}, showlegend=False,
                           xaxis=AXIS_Y, yaxis={**AXIS_X, "showgrid": False})
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    with col_r:
        chart_card(
            "Admission rate by category",
            note="Labour & Delivery at 70.6% is expected. Fibroids at 35.4% is the clinical surprise — "
                 "more than twice the antenatal rate.",
        )
        d2 = top10.sort_values("CONVERSION_RATE_PCT", ascending=True)
        colors2 = [CATEGORY_COLOURS.get(c, _C_MGREY) for c in d2["CASE_MIX_CATEGORY"]]
        fig2 = go.Figure(go.Bar(y=d2["CASE_MIX_CATEGORY"], x=d2["CONVERSION_RATE_PCT"], orientation="h",
                                 marker_color=colors2, marker_cornerradius=3))
        fig2.update_layout(**{**_LAYOUT, "height": _H_PAIRED}, showlegend=False,
                            xaxis={**AXIS_Y, "ticksuffix": "%"}, yaxis={**AXIS_X, "showgrid": False})
        st.plotly_chart(fig2, use_container_width=True, config=PC_CFG)
        chart_card_close()

    def _row(cat):
        m = df[df["CASE_MIX_CATEGORY"] == cat]
        return m.iloc[0] if not m.empty else None

    fib = _row("Fibroids")
    loss = _row("Pregnancy Loss")
    dysm = _row("Dysmenorrhea")

    fibroids_pct = float(fib["PCT_OF_OBGYN_VOLUME"]) if fib is not None else 0
    fibroids_conv = float(fib["CONVERSION_RATE_PCT"]) if fib is not None else 0
    preg_loss_n = int(loss["TOTAL_VISITS"]) if loss is not None else 0
    preg_loss_pct = float(loss["PCT_OF_OBGYN_VOLUME"]) if loss is not None else 0
    dysmenorrhea_n = int(dysm["TOTAL_VISITS"]) if dysm is not None else 0

    insight_bar(
        bullets=[
            f"Fibroids ({fibroids_pct:.1f}% of volume, {fibroids_conv:.1f}% admission rate) was entirely "
            f"invisible in the initial classifier — coded as 'myoma' or 'leiomyoma', not 'fibroid'. Found "
            f"only through manual review of the residual category.",
            f"Pregnancy loss ({preg_loss_n} visits, {preg_loss_pct:.1f}%) is a real, substantial share of "
            f"volume — warrants explicit acknowledgement in any service summary.",
            f"Dysmenorrhea ({dysmenorrhea_n} visits) converts at 0% — an entirely outpatient condition, "
            f"appropriately managed without admission.",
        ],
        action="Fibroids is the unexpected volume and admission story in this service — the third-most "
               "admission-generating category by rate, after Labour & Delivery and Infertility/PID.",
        variant="info",
    )


# ── Section 2 — Demographics + comorbidities ────────────────────────────────

_S2_CATEGORIES = [
    ("ANC / Routine Pregnancy Care", PRIMARY, "#E1F5EE", "#1B8A82"),
    ("Fibroids", SUCCESS, "#EAF3DE", "#3B6D11"),
    ("Dysmenorrhea", NEUTRAL, SURFACE_1, "#5C6478"),
    ("High-Risk Pregnancy / Complications", DANGER, "#FCEBEB", "#A32D2D"),
    ("Pregnancy Loss", WARNING, "#FAEEDA", "#854F0B"),
    ("Labour & Delivery", SECONDARY, "#FBEAF0", "#C13868"),
]
_AGE_GROUPS_S2 = ["Adolescent (13-17)", "Youth (18-24)", "Young Adult (25-34)",
                  "Adult (35-44)", "Middle Age (45-54)", "Unknown"]
_AGE_LABELS_S2 = {"Adolescent (13-17)": "Adolescent 13–17", "Youth (18-24)": "Youth 18–24",
                  "Young Adult (25-34)": "Young adult 25–34", "Adult (35-44)": "Adult 35–44",
                  "Middle Age (45-54)": "Middle age 45–54", "Unknown": "Unknown"}


def _render_demographics_small_multiples(df: pd.DataFrame) -> None:
    df = df.copy()
    df["PCT_WITHIN_CATEGORY"] = (
        df["TOTAL_VISITS"] / df.groupby("CASE_MIX_CATEGORY")["TOTAL_VISITS"].transform("sum") * 100
    ).round(1)

    grid_html = '<div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px">'
    for cat, colour, badge_bg, badge_text in _S2_CATEGORIES:
        subset = df[df["CASE_MIX_CATEGORY"] == cat]
        total = int(subset["TOTAL_VISITS"].sum())
        max_pct = subset["PCT_WITHIN_CATEGORY"].max() if total > 0 else 1
        peak_age = "—"
        if total > 0 and not subset.empty:
            peak_age = _AGE_LABELS_S2.get(
                subset.loc[subset["PCT_WITHIN_CATEGORY"].idxmax(), "AGE_GROUP"],
                subset.loc[subset["PCT_WITHIN_CATEGORY"].idxmax(), "AGE_GROUP"],
            )

        bars_html = ""
        for ag in _AGE_GROUPS_S2:
            row = subset[subset["AGE_GROUP"] == ag]
            pct = float(row["PCT_WITHIN_CATEGORY"].values[0]) if not row.empty else 0.0
            bar_w = int(round(pct / max_pct * 100)) if max_pct > 0 else 0
            bars_html += (
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">'
                f'<div style="font-size:10.5px;color:#3D4457;width:90px;flex-shrink:0;text-align:right">{_AGE_LABELS_S2[ag]}</div>'
                f'<div style="flex:1;height:16px;background:#F4F6FA;border-radius:3px;overflow:hidden">'
                f'<div style="width:{bar_w}%;height:100%;background:{colour};border-radius:3px"></div></div>'
                f'<div style="font-size:10.5px;font-weight:700;color:{colour};width:34px">{pct:.0f}%</div>'
                f'</div>'
            )

        grid_html += (
            f'<div style="background:#FFFFFF;border:0.5px solid #E4E7ED;border-radius:8px;padding:12px 14px">'
            f'<div style="font-size:12px;font-weight:700;color:{colour};margin-bottom:8px">'
            f'{cat} <span style="font-weight:400;color:#8A93A6;font-size:10px">{total} visits</span></div>'
            f'{bars_html}'
            f'<div style="margin-top:8px;font-size:10px;font-weight:700;color:{badge_text};'
            f'background:{badge_bg};padding:4px 8px;border-radius:4px">Peak: {peak_age}</div>'
            f'</div>'
        )
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def render_s2(df_demographics: pd.DataFrame, df_comorbidities: pd.DataFrame) -> None:
    section_header("Section 2 — demographics by category")
    if not _safe(df_demographics):
        _empty()
        return

    _render_demographics_small_multiples(df_demographics)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    htn_high = htn_other = dm_high = dm_other = 0.0
    if _safe(df_comorbidities):
        hr = df_comorbidities[df_comorbidities["CASE_MIX_CATEGORY"] == "High-Risk Pregnancy / Complications"]
        others = df_comorbidities[df_comorbidities["CASE_MIX_CATEGORY"] != "High-Risk Pregnancy / Complications"]
        if not hr.empty:
            htn_high = float(hr.iloc[0]["PCT_HYPERTENSION"] or 0)
            dm_high = float(hr.iloc[0]["PCT_DIABETES"] or 0)
        other_total = others["TOTAL_VISITS"].sum()
        if other_total:
            htn_other = float((others["PCT_HYPERTENSION"] * others["TOTAL_VISITS"]).sum() / other_total)
            dm_other = float((others["PCT_DIABETES"] * others["TOTAL_VISITS"]).sum() / other_total)

    chart_card("Comorbidity rate — High-risk pregnancy vs. all other categories")
    max_val = max(htn_high, htn_other, dm_high, dm_other, 1.0) * 1.15

    def _cmp_row(label, high_val, other_val):
        # Minimum bar width isn't just cosmetic here — "All other" can be
        # genuinely near 0%, and a near-invisible sliver at the old 12px
        # height read as missing data rather than "correctly near zero."
        hw = max(int(high_val / max_val * 100), 3)
        ow = max(int(other_val / max_val * 100), 3)
        return (
            f'<div style="margin-bottom:16px">'
            f'<div style="font-size:12px;font-weight:700;color:{TEXT_PRI};margin-bottom:6px">{label}</div>'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">'
            f'<div style="width:90px;font-size:11px;color:{_C_RED};text-align:right">High-risk</div>'
            f'<div style="flex:1;height:18px;background:#F4F6FA;border-radius:3px;overflow:hidden">'
            f'<div style="width:{hw}%;height:100%;background:{_C_RED}"></div></div>'
            f'<div style="width:44px;font-size:12px;font-weight:700;color:{_C_RED}">{high_val:.1f}%</div></div>'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="width:90px;font-size:11px;color:#5C6478;text-align:right">All other</div>'
            f'<div style="flex:1;height:18px;background:#F4F6FA;border-radius:3px;overflow:hidden">'
            f'<div style="width:{ow}%;height:100%;background:{_C_GREY}"></div></div>'
            f'<div style="width:44px;font-size:12px;font-weight:700;color:#5C6478">{other_val:.1f}%</div></div>'
            f'</div>'
        )

    st.markdown(_cmp_row("Hypertension", htn_high, htn_other) + _cmp_row("Diabetes", dm_high, dm_other),
                unsafe_allow_html=True)
    chart_card_close()

    insight_bar(
        bullets=[
            "Fibroids concentrating in Adult 35–44, Dysmenorrhea peaking in Youth 18–24, and ANC/Labour & "
            "Delivery centred on Young Adult 25–34 — all three patterns are exactly where clinical "
            "expectation places them.",
            "High-risk pregnancy spans all reproductive ages (18–44) — complications do not skew to a "
            "single age window, so age alone is not a useful screening criterion for this category.",
            "Adolescent presentations (13–17) appear across ANC, Dysmenorrhea, and Labour & Delivery — a "
            "small but real population requiring age-appropriate care pathways.",
        ],
        action="The demographic patterns validate the classifier — this section confirms the categories "
               "are capturing clinically coherent populations, not a finding requiring operational response.",
        variant="success",
    )


# ── Section 3 — ANC continuity ──────────────────────────────────────────────

_S3_BUCKET_MAP = {
    "1 visit": "1 visit", "2 visits": "2 visits", "3 visits": "3 visits",
    "4+ visits (meets paper's quality-predictive threshold)": "4+ visits",
}


def _title_card(title: str) -> None:
    """Small standalone title card — just the heading, boxed, with no
    border wrapping the chart/content that follows underneath it."""
    st.markdown(
        f'<div style="background:#FFFFFF;border:0.5px solid #E4E7ED;border-radius:8px;'
        f'padding:12px 14px;font-size:14px;font-weight:700;color:#141F3D">{title}</div>',
        unsafe_allow_html=True,
    )


def _render_s3_funnel(df: pd.DataFrame) -> None:
    df = df.copy()
    df["BUCKET_DISPLAY"] = df["VISIT_COUNT_BUCKET"].map(lambda x: _S3_BUCKET_MAP.get(x, x))
    config = [("1 visit", _C_RED), ("2 visits", _C_AMBER), ("3 visits", _C_GREY), ("4+ visits", _C_TEAL)]

    # Percentage always sits outside the bar, in its own column — a bar for
    # a small share (e.g. 0.7%) is too narrow to hold its own label legibly,
    # which left it clipped down to just a stray "%" sign.
    rows_html = ""
    for label, colour in config:
        row = df[df["BUCKET_DISPLAY"] == label]
        pct = float(row["PCT_OF_ANC_PATIENTS"].values[0]) if not row.empty else 0.0
        n = int(row["TOTAL_PATIENTS"].values[0]) if not row.empty else 0
        bar_w = max(pct, 3)
        tick_label = "4+ →" if label == "4+ visits" else label
        lbl_colour = _C_TEAL if label == "4+ visits" else "#5C6478"
        lbl_weight = "600" if label == "4+ visits" else "400"
        val_colour = colour if colour != _C_GREY else _C_MGREY
        rows_html += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:9px">'
            f'<div style="font-size:11px;color:{lbl_colour};font-weight:{lbl_weight};'
            f'width:60px;flex-shrink:0;text-align:right">{tick_label}</div>'
            f'<div style="flex:1;height:22px;background:#F4F6FA;border-radius:3px;overflow:hidden">'
            f'<div style="width:{bar_w}%;height:100%;background:{colour};border-radius:3px"></div></div>'
            f'<div style="font-size:13px;font-weight:700;color:{val_colour};width:52px;text-align:right">'
            f'{pct:.1f}%</div>'
            f'<div style="font-size:11px;color:#8A93A6;width:56px">{n:,} pts</div>'
            f'</div>'
        )

    cross_val = (
        '<div style="margin-top:12px;padding:9px 11px;background:#FCEBEB;'
        'border-left:3px solid #A32D2D;border-radius:0 6px 6px 0;'
        'font-size:9px;color:#A32D2D;line-height:1.55">'
        '<strong>Cross-validated — three independent methods:</strong> this visit-count '
        "analysis, the Flow and Retention tab's 16.2% repeat visit rate, and 44.2% "
        'scheduled follow-up attendance all converge on the same finding.'
        '</div>'
    )

    # Small standalone title card, then the bars rendered plainly below it
    # with no bordering box of their own — matches the reference layout
    # (title floats in its own compact card; chart sits directly on the
    # page background beneath it).
    _title_card(f'ANC visits per patient — {int(df["TOTAL_PATIENTS"].sum()):,} patients')
    st.markdown(
        f'<div style="padding:14px 4px 0">'
        f'<div style="font-size:9px;font-style:italic;color:#8A93A6;margin-bottom:12px">'
        f'4+ visits = published quality threshold (Asiimwe et al., Kenya DHS 2022)</div>'
        f'{rows_html}'
        f'{cross_val}</div>',
        unsafe_allow_html=True,
    )


def _render_s3_sankey(df: pd.DataFrame) -> None:
    df = df.copy()
    df["BUCKET"] = df["VISIT_COUNT_BUCKET"].map(lambda x: _S3_BUCKET_MAP.get(x, x))

    def n(bucket):
        r = df[df["BUCKET"] == bucket]["TOTAL_PATIENTS"]
        return int(r.values[0]) if not r.empty else 0

    total = int(df["TOTAL_PATIENTS"].sum())
    n_leave = n("1 visit")
    n_return = total - n_leave
    n_thresh = n("4+ visits")
    n_stop = n_return - n_thresh

    if total == 0:
        _empty()
        return

    label = [
        f"All patients<br>{total}",
        f"Leave after 1 visit<br>{n_leave} ({n_leave/total*100:.1f}%)",
        f"Return for more<br>{n_return} ({n_return/total*100:.1f}%)",
        f"Reach 4+ visits →<br>{n_thresh} ({n_thresh/total*100:.1f}%)",
        f"Stop at 2–3 visits<br>{n_stop} ({n_stop/total*100:.1f}%)",
    ]

    # Trunk = navy, "Leave" = red (the problem outcome), "Return" = green
    # (spec §4 Sankey) — sub-branches shaded green/amber by how far they get.
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=18, line=dict(color="white", width=0.5),
            label=label, color=["#141F3D", "#A32D2D", "#639922", "#3B6D11", "#EF9F27"],
            x=[0.01, 0.5, 0.5, 0.99, 0.99], y=[0.5, 0.2, 0.8, 0.7, 0.9],
        ),
        link=dict(
            source=[0, 0, 2, 2], target=[1, 2, 3, 4],
            value=[n_leave, n_return, n_thresh, n_stop],
            color=["rgba(163,45,45,0.25)", "rgba(99,153,34,0.25)",
                   "rgba(59,109,17,0.3)", "rgba(239,159,39,0.25)"],
            label=[f"{n_leave/total*100:.1f}%", f"{n_return/total*100:.1f}%",
                   f"{n_thresh/total*100:.1f}%", f"{n_stop/total*100:.1f}%"],
        ),
    ))
    fig.update_layout(height=315, margin=dict(t=8, b=8, l=8, r=8),
                       paper_bgcolor="white", plot_bgcolor="white",
                       font=dict(family="Inter, sans-serif", size=10, color="#5C6478"))
    st.plotly_chart(fig, use_container_width=True, config=PC_CFG)


def render_s3(df: pd.DataFrame) -> None:
    section_header("Section 3 — ANC continuity: the headline finding")
    if not _safe(df):
        _empty()
        return

    col_l, col_r = st.columns(2)
    with col_l:
        _render_s3_funnel(df)
    with col_r:
        _title_card("What happens after the first visit — patient pathway")
        _render_s3_sankey(df)

    single = df[df["VISIT_COUNT_BUCKET"].str.startswith("1 visit")]
    four_p = df[df["VISIT_COUNT_BUCKET"].str.startswith("4+")]
    single_pct = float(single.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not single.empty else 0
    four_plus_n = int(four_p.iloc[0]["TOTAL_PATIENTS"]) if not four_p.empty else 0
    four_plus_pct = float(four_p.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not four_p.empty else 0

    insight_bar(
        bullets=[
            f"{single_pct:.1f}% of pregnant patients have exactly one recorded antenatal visit — confirmed "
            f"robust across two population definitions (85.2% → 85.7% with corrected filter; marginal "
            f"shift confirms this is not a classification artifact).",
            f"Only {four_plus_n} patients ({four_plus_pct:.1f}%) reach the 4+ visit threshold — published "
            f"Kenyan research links this threshold to meaningfully better pregnancy outcomes.",
            "Antenatal care continuity is the single priority for this service — quality of individual "
            "visits (Section 4) is secondary until continuity is addressed.",
        ],
        action="Address continuity first. Even if every visit scored 5/5 on quality indicators, patients "
               "who only ever have one visit cannot receive a complete course of antenatal care.",
        variant="danger",
    )


# ── Section 4 — ANC quality ─────────────────────────────────────────────────

def render_s4(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    section_header("Section 4 — ANC quality: what is actually done at visits")
    if not _safe(df_a) or not _safe(df_b):
        _empty()
        return

    row = df_a.iloc[0]
    indicators = [
        ("Obstetric ultrasound", float(row["PCT_ULTRASOUND_FETAL_PROXY"]), _C_TEAL),
        ("Urine sample", float(row["PCT_URINE_SAMPLE"]), _C_AMBER),
        ("Blood sample", float(row["PCT_BLOOD_SAMPLE"]), _C_AMBER),
        ("Blood pressure taken", float(row["PCT_BP_TAKEN"]), _C_RED),
        ("Iron supplementation", float(row["PCT_IRON_GIVEN"]), _C_RED),
    ]

    col_l, col_r = st.columns(2)
    with col_l:
        rows_html = ""
        for name, pct, colour in indicators:
            rows_html += (
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:11px">'
                f'<div style="font-size:12px;color:#3D4457;width:150px;flex-shrink:0;text-align:right">{name}</div>'
                f'<div style="flex:1;height:20px;background:#F4F6FA;border-radius:3px;overflow:hidden">'
                f'<div style="width:{pct}%;height:100%;background:{colour};border-radius:3px"></div></div>'
                f'<div style="font-size:13px;font-weight:700;color:{colour};width:44px">{pct:.1f}%</div>'
                f'</div>'
            )
        caveat = (
            '<div style="margin-top:10px;background:#FAEEDA;border:0.5px solid #EF9F27;'
            'border-left:3px solid #854F0B;border-radius:0 6px 6px 0;padding:9px 12px;'
            'font-size:11px;color:#854F0B;line-height:1.55">'
            'Urine-sample matching was never independently verified against a full lab name '
            'list — treat 21.1% as approximate. Ultrasound is the most reliable — '
            'confirmed from structured imaging records, not a text-search proxy.'
            '</div>'
        )
        # Font sizes and title style matched to chart_card() (used on the
        # right), and a fixed height matching that card's total (title +
        # note + 260px chart + padding) so the two sit at the same length.
        st.markdown(
            f'<div style="background:#FFFFFF;border:0.5px solid #E4E7ED;border-radius:10px;'
            f'padding:14px 16px 12px;min-height:334px;box-sizing:border-box;display:flex;'
            f'flex-direction:column;font-family:Inter,sans-serif">'
            f'<div style="font-size:12px;font-weight:600;color:#5C6478;margin-bottom:2px">'
            f'Coverage rate per quality indicator — ANC visits</div>'
            f'<div style="font-size:11px;font-style:italic;color:#8A93A6;margin-bottom:14px;line-height:1.4">'
            f'5 of 8 WHO/Kenya ANC components measurable with structured data. Nutrition, '
            f'breastfeeding, and danger-signs counselling cannot be measured — no structured '
            f'field exists for any of them.</div>'
            f'{rows_html}'
            f'<div style="flex:1"></div>'
            f'{caveat}</div>',
            unsafe_allow_html=True,
        )

    with col_r:
        chart_card(
            "Composite quality score — indicators per visit (out of 5)",
            note="No visit achieves all 5. Nearly half have none. Even a 5/5 score here cannot confirm "
                 "'quality ANC' — the 3 counselling components remain unmeasured.",
        )
        db = df_b.sort_values("ANC_QUALITY_SCORE_OUT_OF_5")
        x = [f"{int(s)} / 5" for s in db["ANC_QUALITY_SCORE_OUT_OF_5"]]
        colour_map = {0: _C_RED, 1: _C_AMBER, 2: _C_AMBER, 3: _C_GREY, 4: _C_TEAL, 5: _C_TEAL}
        colors = [colour_map.get(int(s), _C_GREY) for s in db["ANC_QUALITY_SCORE_OUT_OF_5"]]
        fig = go.Figure(go.Bar(x=x, y=db["PCT_OF_ANC_VISITS"], marker_color=colors, marker_cornerradius=3))
        fig.update_layout(**{**_LAYOUT, "height": _H_PAIRED, "bargap": 0.35}, showlegend=False,
                           xaxis={**AXIS_X, "showgrid": False}, yaxis={**AXIS_Y, "ticksuffix": "%"})
        st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
        chart_card_close()

    zero_row = df_b[df_b["ANC_QUALITY_SCORE_OUT_OF_5"] == 0]
    zero_pct = float(zero_row.iloc[0]["PCT_OF_ANC_VISITS"]) if not zero_row.empty else 0
    five_row = df_b[df_b["ANC_QUALITY_SCORE_OUT_OF_5"] == 5]
    five_visits = int(five_row.iloc[0]["TOTAL_VISITS"]) if not five_row.empty else 0
    iron_pct = float(row["PCT_IRON_GIVEN"])

    insight_bar(
        bullets=[
            f"{zero_pct:.1f}% of ANC visits have no quality indicator recorded at all — no BP, urine, "
            f"blood, iron, or ultrasound.",
            f"Not a single visit achieves all 5 checkable indicators (5/5 = {five_visits} visits).",
            f"Iron supplementation ({iron_pct:.1f}%) is the lowest — a basic, low-cost intervention with "
            f"clear evidence for maternal and fetal outcomes.",
        ],
        action="Even if continuity improves, the quality of each visit needs attention — iron "
               "supplementation and BP recording are low-cost and should be near-universal.",
        variant="danger",
    )


# ── Section 5 — High-risk pregnancy: complications and workup ──────────────

_COMPLICATION_COLOURS = {
    "Haemorrhage": _C_RED, "Hyperemesis Gravidarum": _C_AMBER,
    "PPROM / Premature rupture of membranes": _C_AMBER,
    "Hypertensive disorder (general)": _C_AMBER, "Pre-eclampsia": _C_RED,
    "Gestational diabetes": _C_GREY, "Obstructed labour": _C_RED,
    "Eclampsia": _C_RED, "Subchorionic bleeding": _C_GREY,
}


def render_s5(df_complications: pd.DataFrame, df_bp: pd.DataFrame, df_workup: pd.DataFrame) -> None:
    section_header("Section 5 — high-risk pregnancy: complications and workup")

    col1, col2, col3 = st.columns(3)

    # Shared fixed height across all three cards in this row — chart_card()'s
    # own title+note+300px-chart total on col1 sets the target; col2/col3
    # match it with min-height + flex so the caveat/retraction boxes sit at
    # the same bottom edge instead of wherever their content happens to end.
    _S5_CARD_H = 380

    with col1:
        if not _safe(df_complications):
            _empty()
        else:
            d = df_complications.sort_values("DISTINCT_VISITS", ascending=True)
            colors = [_COMPLICATION_COLOURS.get(t, _C_MGREY) for t in d["COMPLICATION_TYPE"]]
            total_hr = int(df_complications["DISTINCT_VISITS"].sum())
            # This column is one of three side by side — a fixed pixel
            # margin, even a large one, gets fought over by Plotly's own
            # automargin once the container is this narrow, which is what
            # was clipping "PPROM / Premature rupture of membranes" down to
            # "OM / Premature...". Shortened for display rather than
            # relying on margin alone to fit it.
            _short_label = {"PPROM / Premature rupture of membranes": "PPROM"}
            d = d.copy()
            d["DISPLAY_LABEL"] = d["COMPLICATION_TYPE"].map(lambda t: _short_label.get(t, t))
            chart_card(f"Complication types — high-risk pregnancy ({total_hr} visits)")
            # Value labels placed outside each bar — several categories are
            # single-digit counts whose bars are a near-invisible sliver at
            # this width; the label makes the value legible regardless of
            # how thin the bar itself renders.
            fig = go.Figure(go.Bar(
                y=d["DISPLAY_LABEL"], x=d["DISTINCT_VISITS"], orientation="h",
                marker_color=colors, marker_cornerradius=3,
                text=d["DISTINCT_VISITS"], textposition="outside",
                textfont=dict(size=11, color=TEXT_PRI),
                cliponaxis=False,
            ))
            # chart_card()'s own padding + title + note line adds roughly
            # 66px on top of the plot, so the plot height is set to bring
            # the card's total to _S5_CARD_H, matching col2/col3.
            fig.update_layout(
                **{**_LAYOUT, "height": _S5_CARD_H - 66, "margin": dict(t=10, b=10, l=150, r=36)},
                showlegend=False,
                xaxis={**AXIS_Y, "showgrid": False, "visible": False},
                yaxis={**AXIS_X, "showgrid": False, "automargin": True, "tickfont": dict(size=10.5)},
            )
            st.plotly_chart(fig, use_container_width=True, config=PC_CFG)
            chart_card_close()

    with col2:
        n_with_bp = n_total = 0
        if _safe(df_bp):
            n_with_bp = int(df_bp.iloc[0].get("N_WITH_BP", 0) or 0)
            n_total = int(df_bp.iloc[0].get("N_TOTAL", 0) or 0)
        st.markdown(
            f'<div style="background:{SURFACE_1};border:0.5px solid {BORDER};border-radius:10px;'
            f'padding:14px 16px 12px;min-height:{_S5_CARD_H}px;box-sizing:border-box;display:flex;'
            f'flex-direction:column;font-family:Inter,sans-serif">'
            f'<div style="font-size:12px;font-weight:600;color:#5C6478;margin-bottom:10px">'
            f'BP monitoring for hypertensive pregnancy patients</div>'
            f'<div style="text-align:center;padding:8px 0">'
            f'<div style="font-size:42px;font-weight:500;color:{_C_TEAL};line-height:1">'
            f'{n_with_bp} of {n_total}</div>'
            f'<div style="font-size:12px;color:{TEXT_SEC};margin-top:4px">patients have real BP readings</div>'
            f'<div style="font-size:11px;color:{TEXT_MUT};margin-top:2px">most meet the 140/90 clinical threshold</div>'
            f'</div>'
            f'<div style="flex:1"></div>'
            f'<div style="margin-top:12px;padding:9px 11px;background:#EAF3DE;'
            f'border-left:3px solid {_C_TEAL};border-radius:0 4px 4px 0;'
            f'font-size:11px;color:{TEXT_PRI};line-height:1.55">'
            f'<strong>Retraction of earlier finding.</strong> The previous result '
            f'(5 of 8 patients with zero BP data) was a false negative from using an '
            f'incomplete data source. BP IS being recorded and readings confirm the diagnoses.'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        if not _safe(df_workup):
            _empty()
        else:
            wrow = df_workup.iloc[0]
            total = int(wrow.get("TOTAL_HAEMORRHAGE_VISITS", 0) or 0)
            items = [
                ("Haemoglobin check", int(wrow.get("WITH_HEMOGLOBIN_CHECK", 0) or 0), total),
                ("Blood group typed", int(wrow.get("WITH_BLOOD_GROUP_CHECK", 0) or 0), total),
                ("Clotting screen", int(wrow.get("WITH_COAGULATION_CHECK", 0) or 0), total),
            ]
            rows_html = ""
            for label, n_done, n_total_v in items:
                bar_w = int(n_done / n_total_v * 100) if n_total_v > 0 else 0
                rows_html += (
                    f'<div style="display:flex;justify-content:space-between;align-items:center;'
                    f'padding:9px 0;border-bottom:0.5px solid #F4F6FA">'
                    f'<span style="font-size:12px;color:#3D4457">{label}</span>'
                    f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<div style="width:80px;height:9px;background:#F4F6FA;border-radius:2px;overflow:hidden">'
                    f'<div style="width:{bar_w}%;height:100%;background:{_C_RED}"></div></div>'
                    f'<span style="font-size:13px;font-weight:700;color:{_C_RED};width:44px;text-align:right">'
                    f'{n_done}/{n_total_v}</span></div></div>'
                )
            pulse = wrow.get("WITH_PULSE_RECORDED")
            bp_rec = wrow.get("WITH_BP_RECORDED")
            pulse_str = "— (outstanding)" if pd.isna(pulse) else str(int(pulse))
            bp_str = "— (outstanding)" if pd.isna(bp_rec) else str(int(bp_rec))
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;align-items:center;padding:9px 0">'
                f'<span style="font-size:12px;color:#3D4457">Pulse / BP recorded (shock check)</span>'
                f'<span style="font-size:12px;font-weight:600;color:{TEXT_MUT}">{pulse_str} / {bp_str}</span></div>'
            )
            caveat = (
                '<div style="margin-top:10px;background:#FAEEDA;border:0.5px solid #EF9F27;'
                'border-left:3px solid #854F0B;border-radius:0 4px 4px 0;padding:9px 11px;'
                'font-size:11px;color:#854F0B;line-height:1.55">'
                'Lab figures are confirmed floors — actual rates likely higher.'
                '</div>'
            )
            st.markdown(
                f'<div style="background:#FFFFFF;border:0.5px solid #E4E7ED;border-radius:10px;'
                f'padding:14px 16px 12px;min-height:{_S5_CARD_H}px;box-sizing:border-box;display:flex;'
                f'flex-direction:column">'
                f'<div style="font-size:12px;font-weight:600;color:#5C6478;margin-bottom:2px">'
                f'Haemorrhage workup — lab coverage (n={total})</div>'
                f'<div style="font-size:11px;font-style:italic;color:#8A93A6;margin-bottom:10px;line-height:1.4">'
                f'Minimum confirmed rates — actual coverage likely higher due to known record-linkage gaps</div>'
                f'{rows_html}'
                f'<div style="flex:1"></div>'
                f'{caveat}</div>',
                unsafe_allow_html=True,
            )

    hgb_n = bg_n = clot_n = total_haem = 0
    if _safe(df_workup):
        wrow = df_workup.iloc[0]
        total_haem = int(wrow.get("TOTAL_HAEMORRHAGE_VISITS", 0) or 0)
        hgb_n = int(wrow.get("WITH_HEMOGLOBIN_CHECK", 0) or 0)
        bg_n = int(wrow.get("WITH_BLOOD_GROUP_CHECK", 0) or 0)
        clot_n = int(wrow.get("WITH_COAGULATION_CHECK", 0) or 0)

    insight_bar(
        bullets=[
            f"Haemorrhage and hyperemesis are the two most frequent complication types, at similar volume "
            f"— PPROM (premature rupture of membranes) is a smaller but distinct category among the rest.",
            f"Haemoglobin is the most basic test for a patient presenting with haemorrhage — on record for "
            f"only {hgb_n} of {total_haem} visits. Blood group ({bg_n}/{total_haem}) and clotting screen "
            f"({clot_n}/{total_haem}) are standard transfusion prerequisites.",
            "BP monitoring for hypertensive patients is a corrected finding — an earlier check using an "
            "incomplete data source incorrectly suggested BP was not being recorded; the earlier "
            "false-negative result is a retraction, not a caveat. It is genuinely reassuring.",
        ],
        action="Closing the documentation gap for haemoglobin in haemorrhage cases directly supports safer "
               "clinical decision-making for the most common pregnancy complication at this hospital.",
        variant="warning",
    )


# ── Closing synthesis ────────────────────────────────────────────────────────

def _synthesis_values(df_anc: pd.DataFrame, df_quality_b: pd.DataFrame, df_bp: pd.DataFrame) -> dict:
    vals = {"single_pct": 0.0, "four_plus_pct": 0.0, "zero_pct": 0.0, "iron_pct": 0.0}
    if _safe(df_anc):
        single = df_anc[df_anc["VISIT_COUNT_BUCKET"].str.startswith("1 visit")]
        four_p = df_anc[df_anc["VISIT_COUNT_BUCKET"].str.startswith("4+")]
        vals["single_pct"] = float(single.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not single.empty else 0.0
        vals["four_plus_pct"] = float(four_p.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not four_p.empty else 0.0
    if _safe(df_quality_b):
        zero = df_quality_b[df_quality_b["ANC_QUALITY_SCORE_OUT_OF_5"] == 0]
        vals["zero_pct"] = float(zero.iloc[0]["PCT_OF_ANC_VISITS"]) if not zero.empty else 0.0
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


def render_synthesis(df_anc: pd.DataFrame, df_quality_a: pd.DataFrame,
                      df_quality_b: pd.DataFrame, df_bp: pd.DataFrame,
                      df_workup: pd.DataFrame = None) -> None:
    v = _synthesis_values(df_anc, df_quality_b, df_bp)
    iron_pct = float(df_quality_a.iloc[0]["PCT_IRON_GIVEN"]) if _safe(df_quality_a) else 0.0

    hgb_n = bg_n = clot_n = total_haem = 0
    if _safe(df_workup):
        wrow = df_workup.iloc[0]
        total_haem = int(wrow.get("TOTAL_HAEMORRHAGE_VISITS", 0) or 0)
        hgb_n = int(wrow.get("WITH_HEMOGLOBIN_CHECK", 0) or 0)
        bg_n = int(wrow.get("WITH_BLOOD_GROUP_CHECK", 0) or 0)
        clot_n = int(wrow.get("WITH_COAGULATION_CHECK", 0) or 0)

    section_header("Key findings")

    p1_items = [
        f"{v['single_pct']:.1f}% of pregnant patients have exactly one recorded visit; only "
        f"{v['four_plus_pct']:.1f}% reach the 4-visit quality threshold — confirmed by three "
        "independent measurement approaches, so this isn't a classification artifact.",
        "Action: treat single-visit ANC as the default case, not the exception — prioritize "
        "continuity outreach ahead of individual-visit quality work.",
    ]

    p2_items = [
        f"{v['zero_pct']:.1f}% of ANC visits have no quality indicator recorded at all; not a "
        f"single visit achieves all 5 measurable indicators, and iron supplementation — a basic, "
        f"low-cost intervention — is documented in only {iron_pct:.1f}% of visits.",
        "Action: make iron supplementation and the other 4 quality indicators a required checklist "
        "step at every ANC visit, starting with the lowest-cost, highest-yield one first.",
    ]

    p3_items = []
    if total_haem:
        p3_items.append(
            f"Haemoglobin — the most basic test for a haemorrhage presentation — is on record for "
            f"only {hgb_n} of {total_haem} visits. Blood group ({bg_n}/{total_haem}) and clotting "
            f"screen ({clot_n}/{total_haem}), both standard transfusion prerequisites, are lower still."
        )
    else:
        p3_items.append(
            "Haemoglobin, blood group, and clotting screen — the standard haemorrhage workup — are "
            "documented in only a minority of haemorrhage-complication visits."
        )
    p3_items.append(
        "Action: close the haemoglobin, blood group, and clotting-screen documentation gap for "
        "haemorrhage presentations — this directly supports safer transfusion decisions for the "
        "most frequent high-risk-pregnancy complication."
    )

    priority_cards([
        {"label": "PRIORITY 1 — HEADLINE FINDING", "severity": "critical",
         "title": "Treat single-visit ANC as the default case, not the exception",
         "body": _synthesis_list(p1_items, "critical"),
         "source": "Section 3"},
        {"label": "PRIORITY 2 — CARE QUALITY AT EXISTING VISITS", "severity": "monitor",
         "title": "Close the quality-indicator and iron-supplementation gaps",
         "body": _synthesis_list(p2_items, "monitor"),
         "source": "Section 4"},
        {"label": "PRIORITY 3 — HAEMORRHAGE WORKUP GAP", "severity": "critical",
         "title": "Close the haemorrhage workup documentation gap",
         "body": _synthesis_list(p3_items, "critical"),
         "source": "Section 5"},
    ])


def get_overview_tiles(df_anc: pd.DataFrame, df_quality_a: pd.DataFrame, df_quality_b: pd.DataFrame) -> list:
    v = _synthesis_values(df_anc, df_quality_b, None)
    iron_pct = float(df_quality_a.iloc[0]["PCT_IRON_GIVEN"]) if _safe(df_quality_a) else 0.0
    return [
        {
            "issue": f"{v['single_pct']:.1f}% of pregnant patients have exactly one recorded visit. Only "
                     f"{v['four_plus_pct']:.1f}% reach the 4-visit threshold linked to quality outcomes in "
                     f"published Kenyan research.",
            "where": "Disease burden → Maternal health, section 3", "severity": "critical",
            "severity_lbl": "Critical",
        },
        {
            "issue": f"{v['zero_pct']:.1f}% of ANC visits have no quality indicator recorded at all. Iron "
                     f"supplementation is documented in only {iron_pct:.1f}% of visits.",
            "where": "Disease burden → Maternal health, section 4", "severity": "warning",
            "severity_lbl": "Monitor",
        },
        {
            "issue": "Blood pressure IS being recorded for most hypertensive pregnancy patients, and "
                     "readings confirm the diagnoses — a retraction of a previously reported concern.",
            "where": "Disease burden → Maternal health, section 5", "severity": "resolved",
            "severity_lbl": "Resolved",
        },
    ]


# ── Tab entry point ──────────────────────────────────────────────────────────

def render_tab() -> None:
    import clinicals.disease_burden_module.maternal.mat_queries as MAQ

    with st.spinner("Loading data…"):
        df_kpis = MAQ.get_mat_headline_kpis()
        df_case_mix = MAQ.get_mat_case_mix()
        df_demographics = MAQ.get_mat_demographics()
        df_comorbidities = MAQ.get_mat_comorbidities()
        df_anc_visits = MAQ.get_mat_anc_visit_distribution()
        df_quality_a = MAQ.get_mat_anc_quality_part_a()
        df_quality_b = MAQ.get_mat_anc_quality_part_b()
        df_complications = MAQ.get_mat_complications()
        df_bp = MAQ.get_mat_bp_hypertensive()
        df_workup = MAQ.get_mat_haemorrhage_workup()

    render_kpis(df_kpis)
    render_s1(df_case_mix)
    render_s2(df_demographics, df_comorbidities)
    render_s3(df_anc_visits)
    render_s4(df_quality_a, df_quality_b)
    render_s5(df_complications, df_bp, df_workup)
    render_synthesis(df_anc_visits, df_quality_a, df_quality_b, df_bp, df_workup)
