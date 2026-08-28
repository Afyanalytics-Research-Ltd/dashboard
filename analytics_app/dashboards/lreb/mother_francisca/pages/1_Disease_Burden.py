import streamlit as st

st.set_page_config(page_title="Disease Burden · MF Camp", layout="wide")

import pandas as pd
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import (
    apply_theme, render_sidebar, section_header, cl, COLORS,
)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.queries import get_diagnoses, get_encounters, get_patient_diagnoses, get_patient_spine, get_age_diagnosis

apply_theme()
render_sidebar(active="conditions")

try:
    diag     = get_diagnoses()
    enc      = get_encounters()
    pt_diag  = get_patient_diagnoses()
    pt_spine = get_patient_spine()
    age_dx   = get_age_diagnosis()
except Exception as e:
    st.error(f"Database connection failed.\n\n{e}")
    st.stop()

if diag.empty:
    st.warning("No data returned.")
    st.stop()

re            = enc.iloc[0]
enc_with_diag = int(re["ENCOUNTERS_WITH_DIAGNOSIS"])
pct_diag      = float(re["PCT_WITH_DIAGNOSIS"])
total_enc     = int(re["TOTAL_ENCOUNTERS"])

# ── Patient-level join for demographics ───────────────────────────────────────
_has_demo = False
pt_merged = pd.DataFrame()
if not pt_spine.empty and not pt_diag.empty:
    if "SEX" in pt_spine.columns and "AGE_BAND" in pt_spine.columns:
        _spine_slim = pt_spine[["PATIENT_KEY", "SEX", "AGE_BAND"]].drop_duplicates("PATIENT_KEY")
        pt_merged   = pt_diag.merge(_spine_slim, on="PATIENT_KEY", how="left")
        _has_demo   = True

# ── IDSR signal ───────────────────────────────────────────────────────────────
_IDSR_MAP = {
    "TB": "TB", "TUBERCULOSIS": "TB", "PULMONARY TUBERCULOSIS": "TB",
    "MALARIA": "Malaria", "PLASMODIUM FALCIPARUM MALARIA": "Malaria",
    "TYPHOID": "Typhoid", "TYPHOID FEVER": "Typhoid",
    "BRUCELLOSIS": "Brucellosis",
    "SYPHILIS": "Syphilis",
    "HEPATITIS B": "Hepatitis B", "HEP B": "Hepatitis B",
    "HEPATITIS C": "Hepatitis C", "HEP C": "Hepatitis C",
}
idsr_rows = []
if "TERM_CANONICAL" in diag.columns:
    _idsr_df = diag[diag["TERM_CANONICAL"].str.upper().isin(_IDSR_MAP)].copy()
    if not _idsr_df.empty:
        _idsr_df["idsr_label"] = _idsr_df["TERM_CANONICAL"].str.upper().map(_IDSR_MAP)
        _g = _idsr_df.groupby("idsr_label", as_index=False)["ENCOUNTER_COUNT"].sum()
        idsr_rows = _g[_g["ENCOUNTER_COUNT"] > 0].to_dict("records")
_has_idsr = len(idsr_rows) > 0

# ── Shared display maps ───────────────────────────────────────────────────────
_GROUP_LABELS = {
    "NCD_MUSCULOSKELETAL": "Musculoskeletal",
    "OPHTHALMOLOGY":       "Ophthalmology",
    "ENT":                 "ENT",
    "NCD_CARDIOVASCULAR":  "Cardiovascular",
    "NCD_DIABETES":        "Diabetes",
    "GASTROINTESTINAL":    "Gastrointestinal",
    "NCD_RESPIRATORY":     "Respiratory",
    "REPRODUCTIVE_GYNAE":  "Reproductive / Gynaecology",
    "ENDOCRINE":           "Endocrine",
    "NCD_RENAL":           "Renal",
    "INFECTIOUS":          "Infectious",
    "NEUROLOGICAL":        "Neurological",
    "DERMATOLOGICAL":      "Dermatological",
    "SURGICAL":            "Surgical",
    "NUTRITIONAL":         "Nutritional / Haematological",
    "MENTAL_HEALTH":       "Mental Health",
    "ALLERGY_IMMUNOLOGY":  "Allergy / Immunology",
    "ONCOLOGY":            "Oncology",
    "DENTAL":              "Dental",
}
_GROUP_COLORS = {
    "NCD_MUSCULOSKELETAL": COLORS["primary"],
    "OPHTHALMOLOGY":       COLORS["purple"],
    "ENT":                 COLORS["coral"],
    "NCD_CARDIOVASCULAR":  COLORS["danger"],
    "NCD_DIABETES":        "#FB8C00",
    "GASTROINTESTINAL":    COLORS["warning"],
    "NCD_RESPIRATORY":     COLORS["success"],
    "REPRODUCTIVE_GYNAE":  "#E91E8C",
    "ENDOCRINE":           "#F5A623",
    "NCD_RENAL":           "#5C6BC0",
    "INFECTIOUS":          "#26A69A",
    "NEUROLOGICAL":        "#AB47BC",
    "DERMATOLOGICAL":      "#8D6E63",
    "NUTRITIONAL":         "#66BB6A",
    "SURGICAL":            "#EF5350",
    "MENTAL_HEALTH":       "#7E57C2",
    "ALLERGY_IMMUNOLOGY":  "#42A5F5",
    "ONCOLOGY":            "#EC407A",
    "DENTAL":              "#FFA726",
}
_BURDEN_DISPLAY = {
    "NCD":             ("Non-Communicable Disease",        COLORS["primary"]),
    "COMMUNICABLE":    ("Communicable / Infectious",       COLORS["danger"]),
    "INJURY":          ("Injury / Trauma",                 COLORS["warning"]),
    "MATERNAL":        ("Maternal",                        "#E91E8C"),
    "UNDIFFERENTIATED":("Undifferentiated / Unclassified", "#C8D8E8"),
}
_SEX_COLORS = {
    "F": COLORS["primary"], "Female": COLORS["primary"],
    "M": COLORS["success"], "Male":   COLORS["success"],
    "Unknown": COLORS["muted"], "U": COLORS["muted"],
}
_SEX_LABELS = {"F": "Female", "M": "Male"}
_AGE_ORDER  = ["<1","1-4","5-14","15-24","25-34","35-44","45-59","60-74","75+"]

def _collapse_burden(bc):
    if bc in ("SYMPTOM", "UNKNOWN"):
        return "UNDIFFERENTIATED"
    return bc

# ── Pre-compute burden metrics ────────────────────────────────────────────────
burden_collapsed = diag.copy()
burden_collapsed["_bc"] = burden_collapsed["BURDEN_CLASS"].map(_collapse_burden)
burden_by_class  = burden_collapsed.groupby("_bc")["ENCOUNTER_COUNT"].sum()

ncd_enc    = int(diag[diag["BURDEN_CLASS"] == "NCD"]["ENCOUNTER_COUNT"].sum())
comm_enc   = int(diag[diag["BURDEN_CLASS"] == "COMMUNICABLE"]["ENCOUNTER_COUNT"].sum())
undiff_enc = int(burden_by_class.get("UNDIFFERENTIATED", 0))
other_enc  = int(diag[diag["BURDEN_CLASS"].isin(["INJURY","MATERNAL"])]["ENCOUNTER_COUNT"].sum())
total_classified = max(ncd_enc + comm_enc + other_enc, 1)
total_inc_undiff = max(total_classified + undiff_enc, 1)

# ── Patient-level metrics by burden class ─────────────────────────────────────
def _pt_sex(burden_class):
    if not _has_demo or pt_merged.empty:
        return 0, 0
    sub = pt_merged[pt_merged["BURDEN_CLASS"] == burden_class]
    f   = sub[sub["SEX"].isin(["F","Female"])]["PATIENT_KEY"].nunique()
    m   = sub[sub["SEX"].isin(["M","Male"])  ]["PATIENT_KEY"].nunique()
    return f, m

def _pt_top_age(burden_class):
    if not _has_demo or pt_merged.empty:
        return None
    sub = pt_merged[pt_merged["BURDEN_CLASS"] == burden_class]
    if sub.empty:
        return None
    counts = sub.groupby("AGE_BAND")["PATIENT_KEY"].nunique()
    return counts.idxmax() if not counts.empty else None

_ncd_f,  _ncd_m  = _pt_sex("NCD")
_comm_f, _comm_m = _pt_sex("COMMUNICABLE")
_ncd_top_age     = _pt_top_age("NCD")
_comm_top_age    = _pt_top_age("COMMUNICABLE")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:9px;font-weight:700;color:#8BAAC5;text-transform:uppercase;'
    'letter-spacing:2.5px;margin-bottom:10px">Mother Francisca Mission &nbsp;·&nbsp; Nandi County</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="font-size:24px;font-weight:800;color:#003467;margin-bottom:4px">'
    f'What conditions were documented?</div>'
    f'<div style="font-size:13px;color:#6B8CAE;margin-bottom:22px">'
    f'<b style="color:#003467">{enc_with_diag:,}</b> encounters with ≥1 classified diagnosis '
    f'&nbsp;·&nbsp; {pct_diag:.1f}% of {total_enc:,} encounter records</div>',
    unsafe_allow_html=True,
)

# ── Key finding card ─────────────────────────────────────────────────────────
_ncd_pct     = 100 * ncd_enc / total_classified
_top_ncd_row = diag[diag["BURDEN_CLASS"] == "NCD"].groupby("DIAGNOSIS_GROUP")["ENCOUNTER_COUNT"].sum()
_top_ncd_key = _top_ncd_row.idxmax() if not _top_ncd_row.empty else ""
_top_ncd_lbl = _GROUP_LABELS.get(_top_ncd_key, _top_ncd_key.replace("_", " ").title())
_age_note    = f" &nbsp;·&nbsp; Peak age group: <b>{_ncd_top_age}</b>" if _ncd_top_age else ""

st.markdown(
    f'<div style="background:#EBF3FB;border-left:4px solid {COLORS["primary"]};'
    f'border-radius:0 6px 6px 0;padding:14px 20px;margin-bottom:18px">'
    f'<div style="font-size:9px;font-weight:700;color:{COLORS["primary"]};'
    f'text-transform:uppercase;letter-spacing:1.8px;margin-bottom:6px">Key Finding</div>'
    f'<div style="font-size:14px;font-weight:600;color:#003467">'
    f'<b>{_ncd_pct:.0f}%</b> of classified diagnoses were non-communicable '
    f'&nbsp;·&nbsp; Led by <b>{_top_ncd_lbl}</b>{_age_note}'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── IDSR banner ───────────────────────────────────────────────────────────────
if _has_idsr:
    _pills = "  &nbsp;·&nbsp;  ".join(
        f'<b>{r["idsr_label"]}</b> ({r["ENCOUNTER_COUNT"]} enc.)'
        for r in sorted(idsr_rows, key=lambda x: -x["ENCOUNTER_COUNT"])
    )
    st.markdown(
        f'<div style="background:#FFF1F3;border:1px solid {COLORS["danger"]}40;'
        f'border-left:4px solid {COLORS["danger"]};border-radius:6px;'
        f'padding:14px 18px;margin-bottom:22px">'
        f'<div style="font-size:10px;font-weight:800;color:{COLORS["danger"]};'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Public Health Signal</div>'
        f'<div style="font-size:13px;font-weight:600;color:#003467;margin-bottom:6px">'
        f'Nationally notifiable conditions documented: &nbsp;{_pills}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;font-style:italic">'
        f'Documentation does not confirm laboratory notification or case confirmation.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ── Communicable vs NCD — side-by-side bars ──────────────────────────────────
section_header("Communicable vs Non-Communicable")

def _side_bar(col, burden_class, title, color, top_n=8):
    _df = (
        diag[diag["BURDEN_CLASS"] == burden_class]
        .groupby("DIAGNOSIS_GROUP", as_index=False)["ENCOUNTER_COUNT"]
        .sum()
        .sort_values("ENCOUNTER_COUNT", ascending=False)
        .head(top_n)
    )
    if _df.empty:
        return
    _df["label"] = _df["DIAGNOSIS_GROUP"].map(_GROUP_LABELS).fillna(
        _df["DIAGNOSIS_GROUP"].str.replace("_", " ").str.title()
    )
    _df = _df.sort_values("ENCOUNTER_COUNT", ascending=True)
    _fig = go.Figure(go.Bar(
        x=_df["ENCOUNTER_COUNT"],
        y=_df["label"],
        orientation="h",
        marker_color=color,
        text=_df["ENCOUNTER_COUNT"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:,} encounters<extra></extra>",
    ))
    _fig.update_layout(**cl(
        height=320,
        xaxis=dict(title=None, gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
        yaxis=dict(title=None, tickfont=dict(size=11, color="#003467")),
        margin=dict(l=0, r=60, t=10, b=10),
    ))
    with col:
        st.markdown(
            f'<div style="font-size:12px;font-weight:700;color:#003467;margin-bottom:8px">'
            f'{title}</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_fig, use_container_width=True)

_c_comm, _c_ncd = st.columns(2)
_side_bar(_c_comm, "COMMUNICABLE", f"Communicable &nbsp;·&nbsp; {comm_enc:,} encounters", COLORS["danger"])
_side_bar(_c_ncd,  "NCD",          f"Non-Communicable &nbsp;·&nbsp; {ncd_enc:,} encounters", COLORS["primary"])

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Burden by age + sex — comparison ─────────────────────────────────────────
section_header("Burden by Age & Sex")

# Age comparison — grouped bar
if not age_dx.empty:
    _age_comp = (
        age_dx[age_dx["BURDEN_CLASS"].isin(["NCD", "COMMUNICABLE"])]
        .groupby(["AGE_BAND", "BURDEN_CLASS"], as_index=False)["ENCOUNTER_COUNT"]
        .sum()
    )
    _age_pivot = (
        _age_comp
        .pivot(index="AGE_BAND", columns="BURDEN_CLASS", values="ENCOUNTER_COUNT")
        .fillna(0)
        .reindex([b for b in _AGE_ORDER if b in _age_comp["AGE_BAND"].values])
    )
    fig_age = go.Figure()
    for _bc, _col, _name in [
        ("NCD",          COLORS["primary"], "NCD"),
        ("COMMUNICABLE", COLORS["danger"],  "Communicable"),
    ]:
        _y = _age_pivot[_bc].tolist() if _bc in _age_pivot.columns else [0] * len(_age_pivot)
        fig_age.add_trace(go.Bar(
            name=_name,
            x=_age_pivot.index.tolist(),
            y=_y,
            marker_color=_col,
            hovertemplate=f"<b>%{{x}}</b> · {_name}: %{{y:,}}<extra></extra>",
        ))
    fig_age.update_layout(**cl(
        height=300,
        barmode="group",
        xaxis=dict(title=None, tickfont=dict(size=11, color="#003467")),
        yaxis=dict(title=None, gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
        margin=dict(l=0, r=10, t=10, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#003467"),
        ),
    ))
    st.plotly_chart(fig_age, use_container_width=True)

# Sex breakdown strip
if _ncd_f + _ncd_m > 0 or _comm_f + _comm_m > 0:
    _sx1, _sx2 = st.columns(2)
    for _col, _f, _m, _label, _accent in [
        (_sx1, _ncd_f,  _ncd_m,  "NCD",          COLORS["primary"]),
        (_sx2, _comm_f, _comm_m, "Communicable",  COLORS["danger"]),
    ]:
        _tot = max(_f + _m, 1)
        _fp  = 100 * _f / _tot
        _mp  = 100 * _m / _tot
        with _col:
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;color:#003467;margin-bottom:6px">'
                f'{_label} — sex of patients with documented sex</div>'
                f'<div style="display:flex;gap:24px;font-size:12px;color:#6B8CAE">'
                f'<span><b style="color:{COLORS["primary"]}">{_fp:.0f}%</b> Female ({_f:,})</span>'
                f'<span><b style="color:{COLORS["success"]}">{_mp:.0f}%</b> Male ({_m:,})</span>'
                f'</div>'
                f'<div style="display:flex;margin-top:6px;border-radius:4px;overflow:hidden;height:6px">'
                f'<div style="width:{_fp:.1f}%;background:{COLORS["primary"]}"></div>'
                f'<div style="width:{_mp:.1f}%;background:{COLORS["success"]}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Top 20 conditions table ───────────────────────────────────────────────────
section_header("Top 20 Conditions")

_top20 = (
    diag
    .groupby(["TERM_CANONICAL", "DIAGNOSIS_GROUP", "BURDEN_CLASS"], as_index=False)["ENCOUNTER_COUNT"]
    .sum()
    .sort_values("ENCOUNTER_COUNT", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

if not _top20.empty:
    _max_enc = int(_top20["ENCOUNTER_COUNT"].max())

    _CHIP = {
        "NCD":          ("#E8F0FE", COLORS["primary"]),
        "COMMUNICABLE": ("#FEE8E8", COLORS["danger"]),
        "INJURY":       ("#FFF3E0", COLORS["warning"]),
        "MATERNAL":     ("#FCE4F0", "#E91E8C"),
        "SYMPTOM":      ("#F0F4F8", "#6B8CAE"),
        "UNKNOWN":      ("#F0F4F8", "#6B8CAE"),
    }
    _BC_LABEL = {
        "NCD": "NCD", "COMMUNICABLE": "Comm.", "INJURY": "Injury",
        "MATERNAL": "Maternal", "SYMPTOM": "Symptom", "UNKNOWN": "—",
    }

    rows_html = ""
    for i, row in _top20.iterrows():
        rank     = i + 1
        cond     = str(row["TERM_CANONICAL"]).title()
        grp_key  = str(row["DIAGNOSIS_GROUP"])
        grp      = _GROUP_LABELS.get(grp_key, "—") if grp_key != "OTHER" else "—"
        bc       = str(row["BURDEN_CLASS"])
        enc      = int(row["ENCOUNTER_COUNT"])
        bar_pct  = 100 * enc / _max_enc
        chip_bg, chip_fg = _CHIP.get(bc, ("#F0F4F8", "#6B8CAE"))
        bc_lbl   = _BC_LABEL.get(bc, bc.title())
        l_border = f"border-left:3px solid {COLORS['warning']};" if rank <= 3 else "border-left:3px solid transparent;"
        row_bg   = "#FAFCFF" if rank % 2 == 0 else "#FFFFFF"

        rows_html += (
            f'<tr style="background:{row_bg};{l_border}">'
            f'<td style="padding:8px 10px;font-size:11px;color:#8BAAC5;font-weight:700;width:32px">{rank}</td>'
            f'<td style="padding:8px 10px;font-size:12px;color:#003467;font-weight:600">{cond}</td>'
            f'<td style="padding:8px 10px;font-size:11px;color:#6B8CAE">{grp}</td>'
            f'<td style="padding:8px 10px">'
            f'<span style="background:{chip_bg};color:{chip_fg};font-size:10px;font-weight:700;'
            f'padding:2px 8px;border-radius:10px;white-space:nowrap">{bc_lbl}</span>'
            f'</td>'
            f'<td style="padding:8px 12px;width:220px">'
            f'<div style="display:flex;align-items:center;gap:8px">'
            f'<div style="flex:1;background:#EBF3FB;border-radius:3px;height:7px">'
            f'<div style="width:{bar_pct:.1f}%;background:{chip_fg};border-radius:3px;height:7px"></div>'
            f'</div>'
            f'<span style="font-size:12px;font-weight:700;color:#003467;min-width:30px;text-align:right">{enc:,}</span>'
            f'</div>'
            f'</td>'
            f'</tr>'
        )

    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-family:inherit">'
        f'<thead><tr style="border-bottom:2px solid #EBF3FB">'
        f'<th style="padding:8px 10px;font-size:10px;font-weight:700;color:#8BAAC5;text-transform:uppercase;letter-spacing:1px;text-align:left">#</th>'
        f'<th style="padding:8px 10px;font-size:10px;font-weight:700;color:#8BAAC5;text-transform:uppercase;letter-spacing:1px;text-align:left">Condition</th>'
        f'<th style="padding:8px 10px;font-size:10px;font-weight:700;color:#8BAAC5;text-transform:uppercase;letter-spacing:1px;text-align:left">Group</th>'
        f'<th style="padding:8px 10px;font-size:10px;font-weight:700;color:#8BAAC5;text-transform:uppercase;letter-spacing:1px;text-align:left">Class</th>'
        f'<th style="padding:8px 12px;font-size:10px;font-weight:700;color:#8BAAC5;text-transform:uppercase;letter-spacing:1px;text-align:left">Encounters</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom:28px"></div>', unsafe_allow_html=True)

# ── Persistent caveat ─────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #EBF3FB;margin-top:16px;padding:12px 0 4px;'
    'font-size:11px;color:#9BAEC8;text-align:center;font-style:italic">'
    'Figures describe OCR-extracted documentation, not confirmed diagnoses, attendance, or outcomes.'
    '</div>',
    unsafe_allow_html=True,
)
