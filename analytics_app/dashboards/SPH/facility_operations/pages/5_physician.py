"""
Physician Dependence — pages/5_physician.py
Central question: Is inpatient care overly dependent on a small number of physicians?
V2 only (Feb 2025 – present) — doctor_hash covers 100% of inpatient admissions.
V1 physician attribution was 28.7% — excluded from workload analysis (Issue 72).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Physician Dependence · SPH Ortho",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.theme import (
    apply_theme, render_sidebar, COLORS, cl,
    kpi_card, section_header, page_header,
)
from dashboard.queries import (
    q_physician_kpis, q_physician_workload, q_physician_efficiency,
    q_physician_rank_trend, q_physician_continuity,
)

apply_theme()
render_sidebar("physician")

# ── Data loading ───────────────────────────────────────────────────────────────
_kpis  = pd.DataFrame()
_wkld  = pd.DataFrame()
_eff   = pd.DataFrame()
_rank  = pd.DataFrame()
_cont  = pd.DataFrame()

try: _kpis  = q_physician_kpis()
except Exception: pass

try: _wkld  = q_physician_workload()
except Exception: pass

try: _eff   = q_physician_efficiency()
except Exception: pass

try: _rank  = q_physician_rank_trend()
except Exception: pass

try: _cont  = q_physician_continuity()
except Exception: pass


# ── KPI helpers ────────────────────────────────────────────────────────────────
def _safe(df, col, default=0):
    if df.empty or col not in df.columns:
        return default
    v = df.iloc[0][col]
    return default if pd.isna(v) else v


_total_phys = int(_safe(_kpis, "TOTAL_PHYSICIANS"))
_top1_share = float(_safe(_kpis, "TOP1_SHARE"))
_top3_share = float(_safe(_kpis, "TOP3_SHARE"))
_hhi        = float(_safe(_kpis, "HHI", 0.0))


def _conc(top3):
    if top3 >= 65:
        return "High",     COLORS["danger"]
    if top3 >= 40:
        return "Moderate", COLORS["warning"]
    return "Low",          COLORS["primary"]


_conc_label, _conc_color = _conc(_top3_share)

# Label physicians by workload rank (anonymised — Issue 72)
if not _wkld.empty:
    _wkld    = _wkld.copy()
    _wkld["LABEL"] = [f"Physician {i + 1}" for i in range(len(_wkld))]
    _rank_map      = dict(zip(_wkld["DOCTOR_HASH"], _wkld["LABEL"]))
    _top5_share    = float(_wkld.head(5)["SHARE_PCT"].sum())
else:
    _rank_map   = {}
    _top5_share = 0.0

# Efficiency labels
if not _eff.empty:
    _eff = _eff.copy()
    _eff["LABEL"] = _eff["DOCTOR_HASH"].map(_rank_map).fillna("Other")

# Continuity values
_both_n    = int(_safe(_cont, "BOTH_RECORDED"))
_same_n    = int(_safe(_cont, "SAME_PHYSICIAN"))
_xfer_n    = int(_safe(_cont, "TRANSFERRED"))
_match_pct = float(_safe(_cont, "MATCH_PCT"))
_xfer_pct  = round(100.0 - _match_pct, 1)          # no conditional — avoids zero-bug when match=0

# Bump chart colours (one per physician rank 1–5)
_BUMP_COLORS = [
    COLORS["danger"],
    COLORS["warning"],
    COLORS["primary"],
    "#7C3AED",
    "#0891B2",
]


# ── Header ─────────────────────────────────────────────────────────────────────
page_header(
    "Physician Dependence",
    subtitle="Is inpatient care overly dependent on a small number of physicians?",
    period="Feb 2025 – present · V2 inpatient admissions",
)

st.markdown(
    '<div style="font-size:12px;color:#6B8CAE;margin-bottom:28px">'
    'Analysis uses V2 data only. Historical V1 physician identifiers had 28.7% coverage '
    'and are excluded from workload analysis. V2 <code>doctor_hash</code> covers 100% '
    'of inpatient admissions.'
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# §1 — DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────
section_header("Dependency — How concentrated is physician workload?")

k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card(
        "Admitting Physicians",
        str(_total_phys),
        "distinct physicians · V2 inpatients",
        color=COLORS["primary"],
    )
with k2:
    kpi_card(
        "Top Physician Share",
        f"{_top1_share:.1f}%",
        "of all inpatient admissions",
        color=COLORS["primary"],
    )
with k3:
    kpi_card(
        "Top 3 Share",
        f"{_top3_share:.1f}%",
        "combined share · top 3 physicians",
        color=_conc_color,
    )
with k4:
    kpi_card(
        "Concentration",
        _conc_label,
        f"HHI {_hhi:.3f} · top 3 = {_top3_share:.1f}%",
        color=_conc_color,
    )

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# §2 — WORKLOAD DISTRIBUTION  (top 5 only)
# ─────────────────────────────────────────────────────────────────────────────
section_header("Workload Distribution — Who carries inpatient admissions?")

if not _wkld.empty:
    _display = _wkld.head(5).copy()

    st.markdown(
        f'<div style="background:#F4F8FC;border-radius:8px;padding:16px 22px;'
        f'margin-bottom:22px;display:flex;align-items:baseline;gap:8px">'
        f'<span style="font-size:13px;color:#6B8CAE">Top 5 physicians account for</span>'
        f'<span style="font-size:26px;font-weight:800;color:#003467">{_top5_share:.1f}%</span>'
        f'<span style="font-size:13px;color:#6B8CAE">of V2 inpatient admissions.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _bar_colors = [
        COLORS["danger"]  if i == 0 else
        COLORS["warning"] if i < 3  else
        COLORS["primary"]
        for i in range(len(_display))
    ]

    fig_wk = go.Figure(go.Bar(
        x=_display["ADMISSIONS"].tolist(),
        y=_display["LABEL"].tolist(),
        orientation="h",
        marker_color=_bar_colors,
        text=[f"{s:.1f}%" for s in _display["SHARE_PCT"].tolist()],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:,} admissions (%{text})<extra></extra>",
    ))
    fig_wk.update_layout(**cl(
        height=220,
        showlegend=False,
        yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Inpatient admissions", gridcolor="#EBF3FB"),
        margin=dict(l=0, r=80, t=10, b=30),
    ))
    st.plotly_chart(fig_wk, use_container_width=True)

    st.markdown(
        '<div style="font-size:11px;color:#9DB4CC;margin-top:-12px">'
        'Physician identifiers are anonymised hashes — labels are rank-ordered for readability.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Workload data unavailable.")

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# §3 — WORKLOAD CONCENTRATION  (donut + LOS table)
# ─────────────────────────────────────────────────────────────────────────────
section_header("Workload Concentration — Top 5 vs the rest of the team")

if not _wkld.empty:
    _d5     = _wkld.head(5).copy()
    _others = int(_wkld.iloc[5:]["ADMISSIONS"].sum()) if len(_wkld) > 5 else 0

    _labels = _d5["LABEL"].tolist()
    _values = _d5["ADMISSIONS"].tolist()
    _dcolors = _BUMP_COLORS[:len(_labels)]

    if _others > 0:
        _labels.append("Others")
        _values.append(_others)
        _dcolors.append(COLORS["muted"])

    col_donut, col_los = st.columns([1, 1])

    with col_donut:
        fig_dn = go.Figure(go.Pie(
            labels=_labels,
            values=_values,
            hole=0.58,
            marker=dict(colors=_dcolors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b>: %{value:,} admissions (%{percent})<extra></extra>",
        ))
        _total_v2 = sum(_values)
        fig_dn.update_layout(**cl(
            height=300,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=10),
            annotations=[dict(
                text=f"<b>{_total_v2:,}</b><br><span style='font-size:10px'>admissions</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False,
            )],
        ))
        st.plotly_chart(fig_dn, use_container_width=True)

    with col_los:
        if not _eff.empty:
            _los_top5 = (
                _eff[_eff["LABEL"].isin(_d5["LABEL"])]
                .sort_values("ADMISSIONS", ascending=False)
                [["LABEL", "ADMISSIONS", "MEDIAN_LOS"]]
                .rename(columns={"LABEL": "Physician", "ADMISSIONS": "Admissions", "MEDIAN_LOS": "Median LOS (days)"})
            )
            st.markdown(
                '<div style="font-size:12px;color:#6B8CAE;margin-bottom:8px;margin-top:24px">'
                'Median length of stay — top 5 physicians<br>'
                '<span style="font-size:10px;color:#9DB4CC">'
                'Descriptive only — LOS reflects case mix, not performance.</span></div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                _los_top5,
                use_container_width=True,
                hide_index=True,
                height=min(200, (len(_los_top5) + 1) * 35 + 10),
            )
        else:
            st.markdown(
                '<div style="font-size:12px;color:#9DB4CC;margin-top:24px">'
                'LOS data requires ≥ 20 admissions per physician with discharge dates recorded.</div>',
                unsafe_allow_html=True,
            )
else:
    st.info("Concentration data unavailable.")

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# §4 — DEPENDENCY TREND  (bump / rank chart)
# ─────────────────────────────────────────────────────────────────────────────
section_header("Dependency Trend — Is concentration changing over time?")

if not _rank.empty:
    _rank2 = _rank.copy()
    _rank2["MONTH"] = pd.to_datetime(_rank2["ADMISSION_MONTH"])

    fig_bump = go.Figure()

    for rnk in range(1, 6):
        _sub = _rank2[_rank2["OVERALL_RANK"] == rnk].sort_values("MONTH")
        if _sub.empty:
            continue
        _col   = _BUMP_COLORS[rnk - 1]
        _lbl   = f"Physician {rnk}"
        _last  = _sub.iloc[-1]

        fig_bump.add_scatter(
            x=_sub["MONTH"].tolist(),
            y=_sub["MONTHLY_RANK"].tolist(),
            mode="lines+markers",
            name=_lbl,
            line=dict(color=_col, width=2.5),
            marker=dict(size=7, color=_col),
            customdata=_sub["ADMISSIONS"].tolist(),
            hovertemplate=(
                f"<b>{_lbl}</b><br>"
                "%{x|%b %Y} — Rank #%{y}<br>"
                "Admissions: %{customdata:,}<extra></extra>"
            ),
        )

        # End label to the right of the last point
        fig_bump.add_annotation(
            x=_last["MONTH"],
            y=int(_last["MONTHLY_RANK"]),
            text=f"  {_lbl}",
            showarrow=False,
            xanchor="left",
            font=dict(size=9, color=_col),
        )

    fig_bump.update_layout(**cl(
        height=300,
        showlegend=False,
        yaxis=dict(
            title="Monthly rank",
            autorange="reversed",
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["#1", "#2", "#3", "#4", "#5"],
            gridcolor="#EBF3FB",
            dtick=1,
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=110, t=20, b=30),
    ))
    st.plotly_chart(fig_bump, use_container_width=True)

    st.markdown(
        '<div style="font-size:11px;color:#9DB4CC;margin-top:-12px">'
        'Each line tracks one of the top 5 overall physicians by monthly rank. '
        'A line crossing upward means another physician overtook them that month. '
        'Months with fewer than 20 admissions excluded.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Rank trend data unavailable.")

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# §5 — CONTINUITY OF CARE
# ─────────────────────────────────────────────────────────────────────────────
section_header("Continuity of Care — Are admissions handed over?")

if not _cont.empty and _both_n > 0:
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        kpi_card(
            "Both Physicians Recorded",
            f"{_both_n:,}",
            "admissions with both admitting and treating physician",
            color=COLORS["primary"],
        )
    with cc2:
        kpi_card(
            "Same Physician",
            f"{_match_pct:.1f}%",
            "admitted and managed by the same doctor",
            color=COLORS["primary"] if _match_pct > 50 else COLORS["muted"],
        )
    with cc3:
        kpi_card(
            "Care Transfer",
            f"{_xfer_pct:.1f}%",
            "admitted by one physician, managed by another",
            color=COLORS["warning"] if _xfer_pct > 50 else COLORS["muted"],
        )

    if _both_n > 0:
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        fig_cc = go.Figure()
        if _same_n > 0:
            fig_cc.add_trace(go.Bar(
                name="Same physician",
                x=[_same_n], y=[""],
                orientation="h",
                marker_color=COLORS["primary"],
                text=[f"Same · {_match_pct:.0f}%"],
                textposition="inside",
                textfont=dict(size=11, color="white"),
                hovertemplate="Same physician: %{x:,}<extra></extra>",
            ))
        if _xfer_n > 0:
            fig_cc.add_trace(go.Bar(
                name="Care transfer",
                x=[_xfer_n], y=[""],
                orientation="h",
                marker_color=COLORS["warning"],
                text=[f"Transfer · {_xfer_pct:.0f}%"],
                textposition="inside",
                textfont=dict(size=11, color="white"),
                hovertemplate="Care transfer: %{x:,}<extra></extra>",
            ))
        fig_cc.update_layout(**cl(
            barmode="stack",
            height=90,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="left", x=0),
            xaxis=dict(title="Admissions", gridcolor="#EBF3FB", title_standoff=8),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=0, r=20, t=40, b=30),
        ))
        st.plotly_chart(fig_cc, use_container_width=True)

    st.markdown(
        '<div style="font-size:11px;color:#9DB4CC;margin-top:-12px">'
        'V2 only. Compares <code>admitted_by_hash</code> (physician who admitted) '
        'vs <code>doctor_hash</code> (physician on discharge record). '
        'A high transfer rate may indicate a care model where admission decisions and '
        'ongoing clinical management are handled by different roles — not necessarily a quality gap.'
        '</div>',
        unsafe_allow_html=True,
    )
elif _both_n == 0:
    st.markdown(
        '<div style="background:#FFF8F0;border-left:4px solid #F59E0B;'
        'border-radius:0 8px 8px 0;padding:14px 18px;font-size:13px;color:#1A3C5E">'
        'Continuity data unavailable — neither <code>admitted_by_hash</code> nor both physician '
        'fields are populated in V2 admissions. This field may require a reporting view update.'
        '</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Continuity data unavailable.")

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# §6 — OPERATIONAL CONCLUSION
# ─────────────────────────────────────────────────────────────────────────────
section_header("Operational Conclusion")

_risk_stmt = {
    "High":     (
        f"Physician workload is <b>highly concentrated</b>: the top 3 clinicians account for "
        f"<b>{_top3_share:.1f}%</b> of inpatient admissions across {_total_phys} active physicians. "
        f"The lead physician alone carries <b>{_top1_share:.1f}%</b>. "
        f"Service continuity is exposed if any of the top clinicians become unavailable."
    ),
    "Moderate": (
        f"Physician workload shows <b>moderate concentration</b>: the top 3 clinicians account for "
        f"<b>{_top3_share:.1f}%</b> of inpatient admissions across {_total_phys} active physicians. "
        f"Manageable, but warrants monitoring — particularly for leave planning."
    ),
    "Low":      (
        f"Physician workload is <b>broadly distributed</b>: the top 3 clinicians account for "
        f"<b>{_top3_share:.1f}%</b> of inpatient admissions across {_total_phys} active physicians. "
        f"Concentration risk is low under current operating conditions."
    ),
}[_conc_label]

_priority = {
    "High":     (
        "Ensure formal leave planning, cross-cover arrangements, and succession plans are in place "
        "for the top 3–4 clinicians. Consider whether referral patterns can be more equitably "
        "distributed without compromising clinical outcomes."
    ),
    "Moderate": (
        "Monitor concentration trend. Ensure the top 3–4 clinicians have documented cross-cover "
        "and that planned leave does not create inpatient capacity gaps."
    ),
    "Low":      (
        "No immediate action required on physician concentration. Continue monitoring trend. "
        "Ensure cross-cover remains in place as standard practice."
    ),
}[_conc_label]

st.markdown(
    f'<div style="background:#F0FAF9;border-left:4px solid #0BB99F;'
    f'border-radius:0 8px 8px 0;padding:20px 24px;margin-top:8px">'
    f'<div style="font-size:15px;font-weight:700;color:#003467;margin-bottom:10px">'
    f'Dependency risk: <span style="color:{_conc_color}">{_conc_label}</span></div>'
    f'<ul style="font-size:13px;color:#1A3C5E;line-height:1.9;margin:0;padding-left:20px">'
    f'<li>{_risk_stmt}</li>'
    f'<li>Top 5 physicians combined: <b>{_top5_share:.1f}%</b> of all V2 inpatient admissions.</li>'
    f'<li>Continuity: <b>{_match_pct:.1f}%</b> same physician throughout · '
    f'<b>{_xfer_pct:.1f}%</b> involve a care transfer.</li>'
    f'</ul>'
    f'<div style="font-size:12px;color:#2D6A4F;background:#D1FAE5;border-radius:6px;'
    f'padding:8px 14px;margin-top:12px"><b>Management priority:</b> {_priority}</div>'
    f'</div>',
    unsafe_allow_html=True,
)
