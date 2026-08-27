import streamlit as st

st.set_page_config(page_title="Camp Response · MF", layout="wide")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.theme import apply_theme, render_sidebar, section_header, cl, COLORS
from dashboard.queries import get_encounters, get_referrals, get_condition_profile

apply_theme()
render_sidebar(active="response")

try:
    enc  = get_encounters()
    ref  = get_referrals()
    cond = get_condition_profile()
except Exception as e:
    st.error(f"Database connection failed.\n\n{e}")
    st.stop()

if enc.empty:
    st.warning("No data returned.")
    st.stop()

# ── Group label map ────────────────────────────────────────────────────────────
_GROUP_LABELS = {
    "NCD_MUSCULOSKELETAL":        "Musculoskeletal",
    "NCD_RESPIRATORY":            "Respiratory",
    "NCD_CARDIOVASCULAR":         "Cardiovascular",
    "NCD_RENAL":                  "Renal",
    "NCD_ENDOCRINE":              "Endocrine / Metabolic",
    "GASTROINTESTINAL":           "Gastrointestinal",
    "OPHTHALMOLOGY":              "Ophthalmology",
    "NEUROLOGICAL":               "Neurological",
    "DERMATOLOGICAL":             "Dermatological",
    "REPRODUCTIVE_GYNAECOLOGICAL":"Reproductive / Gynaecological",
    "INFECTIOUS_PARASITIC":       "Infectious & Parasitic",
    "HAEMATOLOGICAL":             "Haematological",
    "ENT":                        "ENT",
    "DENTAL":                     "Dental",
    "MENTAL_HEALTH":              "Mental Health",
    "ALLERGY_IMMUNOLOGY":         "Allergy / Immunology",
    "OTHER":                      "Other / Unclassified",
}

def _grp_label(grp):
    return _GROUP_LABELS.get(grp, grp.replace("_", " ").title())

# ── Pull encounter-level values ────────────────────────────────────────────────
r         = enc.iloc[0]
total_enc = int(r["TOTAL_ENCOUNTERS"])
enc_ref   = int(r["ENCOUNTERS_WITH_REFERRAL"])
enc_med   = int(r["ENCOUNTERS_WITH_MEDICATION"])
enc_inv   = int(r["ENCOUNTERS_WITH_INVESTIGATION"])
pct_ref   = float(r["PCT_WITH_REFERRAL"])
pct_med   = float(r["PCT_WITH_MEDICATION"])
pct_inv   = float(r["PCT_WITH_INVESTIGATION"])

# Top referral specialty
top_spec     = "—"
top_spec_pct = 0.0
if not ref.empty and "REFERRAL_SPECIALTY" in ref.columns:
    top_spec     = str(ref.iloc[0]["REFERRAL_SPECIALTY"]).title()
    top_spec_pct = 100 * int(ref.iloc[0]["ENCOUNTER_COUNT"]) / max(int(ref["ENCOUNTER_COUNT"].sum()), 1)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:9px;font-weight:700;color:#8BAAC5;text-transform:uppercase;'
    'letter-spacing:2.5px;margin-bottom:10px">'
    'Mother Francisca Mission &nbsp;·&nbsp; Nandi County</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="font-size:24px;font-weight:800;color:#003467;margin-bottom:4px">'
    f'What did the camp do?</div>'
    f'<div style="font-size:13px;color:#6B8CAE;margin-bottom:22px">'
    f'Clinical actions documented across <b style="color:#003467">{total_enc:,}</b> encounter records</div>',
    unsafe_allow_html=True,
)

# ── P3.1 Key finding panel ─────────────────────────────────────────────────────
section_header("What the Camp Delivered")

_s1, _s2, _s3 = st.columns(3)
for _col, _n, _pct, _label, _color in [
    (_s1, enc_ref, pct_ref, "Referrals",      COLORS["primary"]),
    (_s2, enc_med, pct_med, "Medications",     COLORS["success"]),
    (_s3, enc_inv, pct_inv, "Investigations",  COLORS["purple"]),
]:
    with _col:
        st.markdown(
            f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;'
            f'padding:24px 20px;text-align:center">'
            f'<div style="font-size:36px;font-weight:800;color:{_color};line-height:1">'
            f'{_n:,}</div>'
            f'<div style="font-size:12px;font-weight:700;color:#003467;margin:6px 0 4px;'
            f'text-transform:uppercase;letter-spacing:1px">{_label}</div>'
            f'<div style="font-size:13px;color:#6B8CAE">'
            f'<b style="color:{_color}">{_pct:.0f}%</b> of all encounters</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)

_spec_note = (
    f" &nbsp;·&nbsp; <b>{top_spec}</b> received {top_spec_pct:.0f}% of referrals"
    if top_spec != "—" else ""
)
st.markdown(
    f'<div style="background:#EBF3FB;border-left:4px solid {COLORS["primary"]};'
    f'border-radius:0 6px 6px 0;padding:14px 20px">'
    f'<div style="font-size:9px;font-weight:700;color:{COLORS["primary"]};'
    f'text-transform:uppercase;letter-spacing:1.8px;margin-bottom:6px">Key Finding</div>'
    f'<div style="font-size:14px;font-weight:600;color:#003467">'
    f'<b>{pct_ref:.0f}%</b> of encounters generated a referral &nbsp;·&nbsp; '
    f'<b>{pct_med:.0f}%</b> received medication &nbsp;·&nbsp; '
    f'<b>{pct_inv:.0f}%</b> had an investigation ordered{_spec_note}'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── P3.2 Referral rate by condition group ──────────────────────────────────────
if not cond.empty:
    section_header("Referral Rate by Condition Group")

    _ref_df = (
        cond[cond["DIAGNOSIS_GROUP"] != "OTHER"]
        .assign(label=lambda d: d["DIAGNOSIS_GROUP"].map(_grp_label))
        .sort_values("REFERRAL_RATE_PCT", ascending=True)
    )

    _ref_fig = go.Figure()
    _ref_fig.add_bar(
        x=_ref_df["REFERRAL_RATE_PCT"],
        y=_ref_df["label"],
        orientation="h",
        marker_color=COLORS["primary"],
        marker_opacity=0.85,
        text=[
            f'{r:.0f}%  (n={int(n)})'
            for r, n in zip(_ref_df["REFERRAL_RATE_PCT"], _ref_df["PATIENTS_WITH_DX"])
        ],
        textposition="outside",
        textfont=dict(size=10, color="#003467"),
        hovertemplate="<b>%{y}</b><br>Referral rate: %{x:.1f}%<extra></extra>",
    )
    _ref_fig.update_layout(
        **cl(
            height=420,
            margin=dict(l=0, r=80, t=10, b=30),
            xaxis=dict(
                title="% of patients with this condition referred",
                range=[0, max(_ref_df["REFERRAL_RATE_PCT"].max() * 1.35, 10)],
                ticksuffix="%",
                gridcolor="#EBF3FB",
                tickfont=dict(size=10, color="#6B8CAE"),
            ),
            yaxis=dict(tickfont=dict(size=11, color="#003467"), gridcolor="#EBF3FB"),
        )
    )
    st.plotly_chart(_ref_fig, use_container_width=True)

    st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── P3.3 + P3.4 Slopegraph: Investigation → Medication (2 subplots) ───────────
if not cond.empty:
    section_header("Investigation vs Medication Rate — Response Divergence")

    _sg_all = (
        cond[cond["DIAGNOSIS_GROUP"] != "OTHER"]
        .assign(label=lambda d: d["DIAGNOSIS_GROUP"].map(_grp_label))
        .sort_values("INVESTIGATION_RATE_PCT", ascending=False)
        .reset_index(drop=True)
    )

    _sg_ncd   = _sg_all[_sg_all["DIAGNOSIS_GROUP"].str.startswith("NCD_")]
    _sg_other = _sg_all[~_sg_all["DIAGNOSIS_GROUP"].str.startswith("NCD_")]

    def _add_slope_traces(fig, df, col):
        for _, row in df.iterrows():
            _inv = float(row["INVESTIGATION_RATE_PCT"])
            _med = float(row["MEDICATION_RATE_PCT"])
            _lbl = row["label"]
            _lc  = COLORS["success"] if _med >= _inv else COLORS["purple"]
            fig.add_scatter(
                x=[0, 1], y=[_inv, _med],
                mode="lines+markers",
                line=dict(color=_lc, width=2),
                marker=dict(size=7, color=_lc),
                hovertemplate=(
                    f"<b>{_lbl}</b><br>"
                    f"Investigated: {_inv:.0f}%<br>"
                    f"Medicated: {_med:.0f}%<extra></extra>"
                ),
                showlegend=False,
                row=1, col=col,
            )
            fig.add_annotation(
                x=0, y=_inv,
                text=f"<b>{_inv:.0f}%</b> {_lbl}",
                xanchor="right", xshift=-8,
                showarrow=False,
                font=dict(size=9.5, color="#003467"),
                row=1, col=col,
            )
            fig.add_annotation(
                x=1, y=_med,
                text=f"{_med:.0f}%",
                xanchor="left", xshift=8,
                showarrow=False,
                font=dict(size=9.5, color="#003467"),
                row=1, col=col,
            )

    _sg_fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Non-Communicable Diseases", "Other Condition Groups"],
        horizontal_spacing=0.22,
    )

    _add_slope_traces(_sg_fig, _sg_ncd,   col=1)
    _add_slope_traces(_sg_fig, _sg_other, col=2)

    _axis_common = dict(
        tickvals=[0, 1],
        ticktext=["Investigated", "Medicated"],
        tickfont=dict(size=11, color="#6B8CAE"),
        range=[-0.05, 1.05],
        showgrid=False,
        zeroline=False,
    )
    _yaxis_common = dict(
        ticksuffix="%",
        range=[-5, 105],
        gridcolor="#EBF3FB",
        tickfont=dict(size=10, color="#6B8CAE"),
        zeroline=False,
    )

    _sg_fig.update_layout(
        paper_bgcolor="#fff",
        plot_bgcolor="#fff",
        font=dict(family="Montserrat", color="#003467"),
        height=480,
        margin=dict(l=160, r=70, t=50, b=30),
        showlegend=False,
    )
    _sg_fig.update_xaxes(**_axis_common)
    _sg_fig.update_yaxes(**_yaxis_common)

    st.markdown(
        f'<div style="display:flex;gap:24px;margin-bottom:10px;font-size:11px;color:#003467">'
        f'<span><span style="display:inline-block;width:18px;height:3px;'
        f'background:{COLORS["success"]};vertical-align:middle;margin-right:5px"></span>'
        f'Medicated &gt; investigated</span>'
        f'<span><span style="display:inline-block;width:18px;height:3px;'
        f'background:{COLORS["purple"]};vertical-align:middle;margin-right:5px"></span>'
        f'Investigated &gt; medicated</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(_sg_fig, use_container_width=True)
    st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── P3.5 Referral by specialty — donut ────────────────────────────────────────
if not ref.empty and "REFERRAL_SPECIALTY" in ref.columns:
    section_header("Downstream Referral Workload by Specialty")

    _spec_df    = ref.sort_values("ENCOUNTER_COUNT", ascending=False)
    _spec_total = int(_spec_df["ENCOUNTER_COUNT"].sum())
    _spec_labels = _spec_df["REFERRAL_SPECIALTY"].str.title().tolist()
    _spec_values = _spec_df["ENCOUNTER_COUNT"].tolist()

    _DONUT_COLORS = [
        COLORS["primary"], COLORS["success"], COLORS["purple"],
        COLORS["warning"], COLORS["coral"], COLORS["green"],
        COLORS["muted"],   "#42A5F5",        "#EC407A",
    ]

    _donut_col, _legend_col = st.columns([1, 1])

    with _donut_col:
        _donut_fig = go.Figure()
        _donut_fig.add_pie(
            labels=_spec_labels,
            values=_spec_values,
            hole=0.62,
            marker=dict(
                colors=_DONUT_COLORS[:len(_spec_labels)],
                line=dict(color="#fff", width=2),
            ),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value} referrals (%{percent})<extra></extra>",
        )
        _donut_fig.update_layout(
            **cl(
                height=300,
                margin=dict(l=0, r=0, t=10, b=10),
                showlegend=False,
                annotations=[dict(
                    text=f'<b style="font-size:18px">{_spec_total}</b><br>'
                         f'<span style="font-size:11px;color:#6B8CAE">referrals</span>',
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="#003467"),
                )],
            )
        )
        st.plotly_chart(_donut_fig, use_container_width=True)

    with _legend_col:
        st.markdown('<div style="margin-top:24px"></div>', unsafe_allow_html=True)
        for i, (lbl, val) in enumerate(zip(_spec_labels, _spec_values)):
            _pct  = 100 * val / max(_spec_total, 1)
            _col  = _DONUT_COLORS[i % len(_DONUT_COLORS)]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'padding:7px 0;border-bottom:1px solid #EBF3FB">'
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                f'background:{_col};flex-shrink:0"></span>'
                f'<span style="font-size:12px;font-weight:600;color:#003467;flex:1">{lbl}</span>'
                f'<span style="font-size:12px;font-weight:700;color:{_col}">{int(val)}</span>'
                f'<span style="font-size:11px;color:#9BAEC8;min-width:36px;text-align:right">'
                f'{_pct:.0f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="margin-bottom:16px"></div>', unsafe_allow_html=True)

# ── Persistent caveat ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #EBF3FB;margin-top:16px;padding:12px 0 4px;'
    'font-size:11px;color:#9BAEC8;text-align:center;font-style:italic">'
    'Figures describe OCR-extracted documentation, not confirmed care delivery or clinical outcomes. '
    'Missing fields do not indicate absence of service.'
    '</div>',
    unsafe_allow_html=True,
)
