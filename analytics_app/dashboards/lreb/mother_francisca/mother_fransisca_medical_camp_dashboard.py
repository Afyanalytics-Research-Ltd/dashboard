import streamlit as st

st.set_page_config(page_title="Mother Francisca Medical Camp Dashboard", layout="wide")

import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import apply_theme, render_sidebar, section_header, kpi_card, cl, COLORS
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.queries import get_encounters, get_demographics, get_diagnoses, get_referrals, get_age_diagnosis

apply_theme()
render_sidebar(active="overview")

try:
    enc    = get_encounters()
    demo   = get_demographics()
    diag   = get_diagnoses()
    ref    = get_referrals()
    age_dx = get_age_diagnosis()
except Exception as e:
    st.error(f"Database connection failed. Check TKLK/.env credentials.\n\n{e}")
    st.stop()

if enc.empty or demo.empty:
    st.warning("No data returned.")
    st.stop()

# ── Pull values ────────────────────────────────────────────────────────────────
r  = enc.iloc[0]
rd = demo.iloc[0]

total_enc = int(r["TOTAL_ENCOUNTERS"])
enc_diag  = int(r["ENCOUNTERS_WITH_DIAGNOSIS"])
enc_inv   = int(r["ENCOUNTERS_WITH_INVESTIGATION"])
enc_med   = int(r["ENCOUNTERS_WITH_MEDICATION"])
enc_ref   = int(r["ENCOUNTERS_WITH_REFERRAL"])
pct_diag  = float(r["PCT_WITH_DIAGNOSIS"])
pct_inv   = float(r["PCT_WITH_INVESTIGATION"])
pct_med   = float(r["PCT_WITH_MEDICATION"])
pct_ref   = float(r["PCT_WITH_REFERRAL"])
q_high    = int(r["QUALITY_HIGH"])
q_med     = int(r["QUALITY_MEDIUM"])
q_low     = int(r["QUALITY_LOW"])
q_total   = q_high + q_med + q_low or 1

female_n  = int(rd["FEMALE_N"])
male_n    = int(rd["MALE_N"])
unk_n     = int(rd["SEX_UNKNOWN_N"])
linked_n  = int(rd["UNIQUE_PATIENTS"])

# ── Compute 3 findings ────────────────────────────────────────────────────────
# Finding 1 — Population
_AGE_BANDS_F = ["<1","1–4","5–14","15–24","25–34","35–44","45–59","60–74","75+"]
_AGE_COLS_F  = ["AGE_UNDER1_N","AGE_1_4_N","AGE_5_14_N","AGE_15_24_N","AGE_25_34_N",
                "AGE_35_44_N","AGE_45_59_N","AGE_60_74_N","AGE_75PLUS_N"]
_age_vals_f  = [int(rd.get(c, 0)) for c in _AGE_COLS_F]
top_age_band = _AGE_BANDS_F[_age_vals_f.index(max(_age_vals_f))] if max(_age_vals_f) > 0 else "older"
female_pct   = 100 * female_n / max(female_n + male_n, 1)

# Finding 2 — Burden
_ncd_enc   = int(diag[diag["BURDEN_CLASS"] == "NCD"]["ENCOUNTER_COUNT"].sum()) if not diag.empty else 0
_comm_enc  = int(diag[diag["BURDEN_CLASS"] == "COMMUNICABLE"]["ENCOUNTER_COUNT"].sum()) if not diag.empty else 0
_other_enc = int(diag[diag["BURDEN_CLASS"].isin(["INJURY","MATERNAL"])]["ENCOUNTER_COUNT"].sum()) if not diag.empty else 0
_total_cls = max(_ncd_enc + _comm_enc + _other_enc, 1)
ncd_pct_f  = 100 * _ncd_enc / _total_cls

_GRP_LABELS_F = {
    "NCD_MUSCULOSKELETAL": "Musculoskeletal", "OPHTHALMOLOGY": "Ophthalmology",
    "ENT": "ENT", "NCD_CARDIOVASCULAR": "Cardiovascular",
    "GASTROINTESTINAL": "Gastrointestinal", "NCD_RESPIRATORY": "Respiratory",
    "REPRODUCTIVE_GYNAE": "Reproductive / Gynaecology", "ENDOCRINE": "Endocrine",
    "NCD_RENAL": "Renal", "INFECTIOUS": "Infectious", "NEUROLOGICAL": "Neurological",
    "DERMATOLOGICAL": "Dermatological", "SURGICAL": "Surgical",
    "NUTRITIONAL": "Nutritional", "MENTAL_HEALTH": "Mental Health",
}
if not diag.empty and "DIAGNOSIS_GROUP" in diag.columns:
    _grp_tots      = diag[diag["DIAGNOSIS_GROUP"] != "OTHER"].groupby("DIAGNOSIS_GROUP")["ENCOUNTER_COUNT"].sum()
    _top_grp       = _grp_tots.idxmax() if not _grp_tots.empty else ""
    top_group_label = _GRP_LABELS_F.get(_top_grp, _top_grp.replace("_", " ").title())
else:
    top_group_label = "musculoskeletal conditions"

# Finding 3 — Service
if not ref.empty and "REFERRAL_SPECIALTY" in ref.columns and "ENCOUNTER_COUNT" in ref.columns:
    _ref_total = int(ref["ENCOUNTER_COUNT"].sum()) or 1
    top_spec   = str(ref.iloc[0]["REFERRAL_SPECIALTY"]).title()
    top_spec_pct = 100 * int(ref.iloc[0]["ENCOUNTER_COUNT"]) / _ref_total
else:
    top_spec, top_spec_pct = "Physiotherapy", 0.0

# ── Breadcrumb ─────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:9px;font-weight:700;color:#8BAAC5;'
    'text-transform:uppercase;letter-spacing:2.5px;margin-bottom:20px">'
    'Mother Francisca Mission &nbsp;·&nbsp; Nandi County &nbsp;·&nbsp; Jul – Sep 2026'
    '</div>',
    unsafe_allow_html=True,
)

# ── Section 1: KPI cards ───────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card(
        "Encounters",
        f"{total_enc:,}",
        sub="Encounter records extracted from scanned documents",
    )
with k2:
    kpi_card(
        "Patients",
        f"{linked_n:,}",
        sub="After name-based deduplication",
    )
with k3:
    kpi_card(
        "Classified",
        f"{pct_diag:.0f}%",
        sub=f"{enc_diag:,} encounters with ≥1 classified diagnosis",
    )
with k4:
    kpi_card(
        "Referrals",
        f"{enc_ref:,}",
        sub=f"{pct_ref:.0f}% of all encounters",
    )

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Section 2: The Camp in 3 Findings ────────────────────────────────────────
_f1 = (
    f"The camp predominantly served older adults — <b>{top_age_band}</b> was the peak "
    f"attendance group — with <b>{female_pct:.0f}%</b> of patients female."
)
_f2 = (
    f"Non-communicable conditions accounted for <b>{ncd_pct_f:.0f}%</b> of classified "
    f"diagnoses, led by <b>{top_group_label}</b> — consistent with the older adult profile."
)
_f3 = (
    f"<b>{enc_ref:,}</b> referrals documented — <b>{top_spec_pct:.0f}%</b> directed to "
    f"{top_spec}, directly aligned with the {top_group_label.lower()} burden."
)

section_header("The Camp in 3 Findings")

_fc1, _fc2, _fc3 = st.columns(3)
for _col, _num, _nc, _cat, _txt in zip(
    [_fc1, _fc2, _fc3],
    ["01", "02", "03"],
    [COLORS["primary"], COLORS["warning"], COLORS["success"]],
    ["Population", "Burden", "Service"],
    [_f1, _f2, _f3],
):
    with _col:
        st.markdown(
            f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:8px;'
            f'padding:22px 20px">'
            f'<div style="font-size:22px;font-weight:800;color:{_nc};line-height:1;margin-bottom:4px">'
            f'{_num}</div>'
            f'<div style="font-size:9px;font-weight:700;color:#8BAAC5;text-transform:uppercase;'
            f'letter-spacing:1.8px;margin-bottom:12px">{_cat}</div>'
            f'<div style="font-size:13px;color:#003467;line-height:1.65">{_txt}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Section 3: Coverage ────────────────────────────────────────────────────────
section_header("Documentation Coverage")

_COVERAGE = [
    ("Diagnosis",     enc_diag, pct_diag, COLORS["primary"]),
    ("Medication",    enc_med,  pct_med,  COLORS["success"]),
    ("Investigation", enc_inv,  pct_inv,  COLORS["purple"]),
    ("Referral",      enc_ref,  pct_ref,  COLORS["warning"]),
]

for _lbl, _n, _pct, _col in _COVERAGE:
    st.markdown(
        f'<div style="margin-bottom:14px">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
        f'<span style="font-size:12px;font-weight:700;color:#003467">{_lbl}</span>'
        f'<span style="font-size:12px;color:#6B8CAE">{_n:,} of {total_enc:,} encounters &nbsp;·&nbsp; '
        f'<b style="color:{_col}">{_pct:.0f}%</b></span>'
        f'</div>'
        f'<div style="background:#EBF3FB;border-radius:4px;height:10px;width:100%">'
        f'<div style="background:{_col};border-radius:4px;height:10px;width:{_pct:.1f}%"></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Section 4: Who came ────────────────────────────────────────────────────────
section_header("Who Came")

col_sex, col_age = st.columns([1, 2])

with col_sex:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:12px">'
        'Encounters by documented sex</div>',
        unsafe_allow_html=True,
    )
    for _s_lbl, _s_n, _s_col in [
        ("Female",  female_n, COLORS["primary"]),
        ("Male",    male_n,   COLORS["success"]),
        ("Unknown", unk_n,    COLORS["muted"]),
    ]:
        _bar_pct = 100 * _s_n / total_enc if total_enc else 0
        st.markdown(
            f'<div style="margin-bottom:10px">'
            f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px">'
            f'<span style="color:#003467;font-weight:600">{_s_lbl}</span>'
            f'<span style="color:#6B8CAE">{_s_n:,}</span>'
            f'</div>'
            f'<div style="background:#EBF3FB;border-radius:3px;height:6px">'
            f'<div style="background:{_s_col};border-radius:3px;height:6px;'
            f'width:{_bar_pct:.1f}%"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with col_age:
    st.markdown(
        '<div style="font-size:11px;font-weight:600;color:#003467;margin-bottom:12px">'
        'Encounters by age band</div>',
        unsafe_allow_html=True,
    )
    _AGE_BANDS = ["<1", "1–4", "5–14", "15–24", "25–34", "35–44", "45–59", "60–74", "75+"]
    _AGE_COLS  = [
        "AGE_UNDER1_N", "AGE_1_4_N", "AGE_5_14_N", "AGE_15_24_N", "AGE_25_34_N",
        "AGE_35_44_N", "AGE_45_59_N", "AGE_60_74_N", "AGE_75PLUS_N",
    ]
    age_vals = [int(rd[c]) for c in _AGE_COLS]

    fig_age = go.Figure(go.Bar(
        x=_AGE_BANDS,
        y=age_vals,
        marker_color=COLORS["primary"],
        text=age_vals,
        textposition="outside",
        hovertemplate="<b>%{x}</b>: %{y:,} encounters<extra></extra>",
    ))
    fig_age.update_layout(**cl(
        height=240,
        xaxis=dict(title=None, tickfont=dict(size=11, color="#003467")),
        yaxis=dict(title=None, gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
        margin=dict(l=0, r=10, t=20, b=10),
    ))
    st.plotly_chart(fig_age, use_container_width=True)

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Section 5: Why They Came ──────────────────────────────────────────────────
section_header("Why They Came — Top 10 Diagnoses")

if not diag.empty and "DIAGNOSIS_GROUP" in diag.columns:
    _grp_df = (
        diag[diag["DIAGNOSIS_GROUP"] != "OTHER"]
        .groupby("DIAGNOSIS_GROUP", as_index=False)["ENCOUNTER_COUNT"]
        .sum()
        .sort_values("ENCOUNTER_COUNT", ascending=False)
        .head(10)
    )
    _grp_df["label"] = _grp_df["DIAGNOSIS_GROUP"].map(_GRP_LABELS_F).fillna(
        _grp_df["DIAGNOSIS_GROUP"].str.replace("_", " ").str.title()
    )
    _grp_df = _grp_df.sort_values("ENCOUNTER_COUNT", ascending=True)

    fig_why = go.Figure(go.Bar(
        x=_grp_df["ENCOUNTER_COUNT"],
        y=_grp_df["label"],
        orientation="h",
        marker_color=COLORS["primary"],
        text=_grp_df["ENCOUNTER_COUNT"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:,} encounters<extra></extra>",
    ))
    fig_why.update_layout(**cl(
        height=360,
        xaxis=dict(title=None, gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
        yaxis=dict(title=None, tickfont=dict(size=11, color="#003467")),
        margin=dict(l=0, r=70, t=10, b=10),
    ))
    st.plotly_chart(fig_why, use_container_width=True)

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Section 6: Who Had What ───────────────────────────────────────────────────
section_header("Who Had What — Top 5 Conditions by Age Group")

st.markdown(
    '<div style="font-size:11px;color:#6B8CAE;margin-bottom:14px">'
    '<b style="color:#003467">Paediatric</b> &nbsp;·&nbsp; under 15 yrs &nbsp;&nbsp;|&nbsp;&nbsp;'
    '<b style="color:#003467">Adult</b> &nbsp;·&nbsp; 15 – 59 yrs &nbsp;&nbsp;|&nbsp;&nbsp;'
    '<b style="color:#003467">Older Adult</b> &nbsp;·&nbsp; 60 yrs and above'
    '</div>',
    unsafe_allow_html=True,
)

if not age_dx.empty:
    _TIER_MAP = {
        "<1": "Paediatric", "1-4": "Paediatric", "5-14": "Paediatric",
        "15-24": "Adult", "25-34": "Adult", "35-44": "Adult", "45-59": "Adult",
        "60-74": "Older Adult", "75+": "Older Adult",
    }
    _adf = age_dx.copy()
    _adf["TIER"] = _adf["AGE_BAND"].map(_TIER_MAP).fillna("Other")
    _adf = _adf[_adf["TIER"] != "Other"]

    _top5 = (
        _adf.groupby("DIAGNOSIS_GROUP")["ENCOUNTER_COUNT"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )
    _adf = _adf[_adf["DIAGNOSIS_GROUP"].isin(_top5)]
    _adf["label"] = _adf["DIAGNOSIS_GROUP"].map(_GRP_LABELS_F).fillna(
        _adf["DIAGNOSIS_GROUP"].str.replace("_", " ").str.title()
    )

    _pivot = _adf.groupby(["label", "TIER"])["ENCOUNTER_COUNT"].sum().reset_index()
    _x_order = (
        _pivot.groupby("label")["ENCOUNTER_COUNT"].sum()
        .sort_values(ascending=False).index.tolist()
    )

    _tier_colors = {
        "Paediatric":  COLORS["success"],
        "Adult":       COLORS["primary"],
        "Older Adult": COLORS["warning"],
    }
    fig_who = go.Figure()
    for _tier in ["Paediatric", "Adult", "Older Adult"]:
        _sub = _pivot[_pivot["TIER"] == _tier].set_index("label")
        _y = [int(_sub.loc[lbl, "ENCOUNTER_COUNT"]) if lbl in _sub.index else 0 for lbl in _x_order]
        fig_who.add_trace(go.Bar(
            name=_tier,
            x=_x_order,
            y=_y,
            marker_color=_tier_colors[_tier],
            hovertemplate=f"<b>%{{x}}</b> · {_tier}: %{{y:,}}<extra></extra>",
        ))

    fig_who.update_layout(**cl(
        height=320,
        barmode="group",
        xaxis=dict(title=None, tickfont=dict(size=11, color="#003467")),
        yaxis=dict(title=None, gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
        margin=dict(l=0, r=10, t=10, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#003467"),
        ),
    ))
    st.plotly_chart(fig_who, use_container_width=True)

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Section 7: Data quality ────────────────────────────────────────────────────
section_header("Data Quality")

dq_col1, dq_col2, dq_col3, dq_col4 = st.columns(4)
_dq_box = (
    "padding:14px 16px;background:#F4F8FC;border:1px solid #D6E4F0;"
    "border-radius:6px;height:100%"
)
with dq_col1:
    st.markdown(
        f'<div style="{_dq_box}">'
        f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">High quality</div>'
        f'<div style="font-size:28px;font-weight:800;color:{COLORS["success"]};line-height:1">'
        f'{q_high:,}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:5px">'
        f'{100*q_high/q_total:.0f}% of encounters</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with dq_col2:
    st.markdown(
        f'<div style="{_dq_box}">'
        f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">Medium quality</div>'
        f'<div style="font-size:28px;font-weight:800;color:{COLORS["warning"]};line-height:1">'
        f'{q_med:,}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:5px">'
        f'{100*q_med/q_total:.0f}% of encounters</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with dq_col3:
    st.markdown(
        f'<div style="{_dq_box}">'
        f'<div style="font-size:9px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">Low quality</div>'
        f'<div style="font-size:28px;font-weight:800;color:{COLORS["danger"]};line-height:1">'
        f'{q_low:,}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:5px">'
        f'{100*q_low/q_total:.0f}% of encounters</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with dq_col4:
    st.markdown(
        f'<div style="{_dq_box};font-size:11px;color:#6B8CAE;line-height:1.75">'
        f'<b style="color:#003467">High:</b> name + date + ≥1 clinical field<br>'
        f'<b style="color:#003467">Medium:</b> name + ≥1 clinical field<br>'
        f'<b style="color:#003467">Low:</b> minimal fields extracted'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom:20px"></div>', unsafe_allow_html=True)

# ── Consolidated data notes ────────────────────────────────────────────────────
with st.expander("About the data"):
    st.markdown(
        f"""
**Source:** Mother Francisca Mission Maternity & Health Centre, Nandi County — July 2026 health camp.

**How records were created:** Handwritten clinical forms were scanned and processed through an AI/OCR extraction pipeline. Each row in the underlying dataset represents one page from one scanned PDF. Multiple pages may relate to the same patient visit.

**Patient linkage:** "Linked patient records" are derived by matching patient names after normalisation. This is probabilistic — same name is assumed to be the same person. It is not validated record linkage. The figure {linked_n:,} is a lower bound; the true count of unique individuals may differ.

**Sex ({int(r['PCT_SEX_DOCUMENTED']):.0f}% documented):** Sex is not captured on all document types. Prescription forms and lab result pages typically omit demographics — these were extracted without a corresponding registration page.

**Age ({int(r['PCT_AGE_DOCUMENTED']):.0f}% documented):** Same constraint as sex. Age is derived from free-text OCR and may contain parsing artefacts.

**Quality tiers** reflect OCR extraction confidence, not clinical accuracy or completeness of care.

**Coverage figures** (Diagnosis, Medication, Investigation, Referral) describe whether a field was *documented* in the scanned record — not whether the service was delivered. An encounter with no diagnosis documented may still have received care.
        """,
        unsafe_allow_html=False,
    )

# ── Persistent caveat ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #EBF3FB;margin-top:16px;padding:12px 0 4px;'
    'font-size:11px;color:#9BAEC8;text-align:center;font-style:italic">'
    'Figures describe OCR-extracted documentation, not confirmed attendance, '
    'care delivery, or clinical outcomes. Missing fields do not indicate absence of care.'
    '</div>',
    unsafe_allow_html=True,
)
