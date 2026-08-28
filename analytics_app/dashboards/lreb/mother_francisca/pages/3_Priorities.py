import streamlit as st

st.set_page_config(page_title="Priorities · MF", layout="wide")

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.theme import apply_theme, render_sidebar, section_header, cl, COLORS
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.queries import get_encounters, get_condition_profile, get_diagnoses, get_vitals_signals

apply_theme()
render_sidebar(active="priorities")

try:
    enc    = get_encounters()
    cond   = get_condition_profile()
    diag   = get_diagnoses()
    vitals = get_vitals_signals()
except Exception as e:
    st.error(f"Database connection failed.\n\n{e}")
    st.stop()

if enc.empty:
    st.warning("No data returned.")
    st.stop()

total_enc = int(enc.iloc[0]["TOTAL_ENCOUNTERS"])

# ── Derive priority evidence ───────────────────────────────────────────────────

# P1 — NCD continuity gap: top NCD groups by medication-referral gap
_ncd = cond[
    cond["DIAGNOSIS_GROUP"].str.startswith("NCD_") &
    (cond["DIAGNOSIS_GROUP"] != "NCD_RENAL")
].copy()
_ncd["gap"] = _ncd["MEDICATION_RATE_PCT"] - _ncd["REFERRAL_RATE_PCT"]
_ncd_top = _ncd.sort_values("gap", ascending=False).head(3)

# P2 — Renal care gap
_renal_rows = cond[cond["DIAGNOSIS_GROUP"] == "NCD_RENAL"]
_renal_n       = int(_renal_rows["PATIENTS_WITH_DX"].iloc[0]) if not _renal_rows.empty else 0
_renal_inv_pct = float(_renal_rows["INVESTIGATION_RATE_PCT"].iloc[0]) if not _renal_rows.empty else 0
_renal_ref_pct = float(_renal_rows["REFERRAL_RATE_PCT"].iloc[0]) if not _renal_rows.empty else 0

# P3 — Documentation fill rates (from raw data profiling — fixed)
_doc_fields = [
    ("Patient sex",   45),
    ("Patient age",   44),
    ("Diagnoses",     40),
    ("Vitals",        29),
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="font-size:9px;font-weight:700;color:#8BAAC5;text-transform:uppercase;'
    'letter-spacing:2.5px;margin-bottom:10px">'
    'Mother Francisca Mission &nbsp;·&nbsp; Nandi County</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:24px;font-weight:800;color:#003467;margin-bottom:4px">'
    'What should matter next?</div>'
    '<div style="font-size:13px;color:#6B8CAE;margin-bottom:28px">'
    'Three priorities derived directly from camp data</div>',
    unsafe_allow_html=True,
)

# ── Priority cards ─────────────────────────────────────────────────────────────
_c1, _c2, _c3 = st.columns(3)

# ── Priority 01 — NCD continuity ───────────────────────────────────────────────
_p1_rows = ""
for _, row in _ncd_top.iterrows():
    grp   = row["DIAGNOSIS_GROUP"].replace("NCD_", "").replace("_", " ").title()
    med   = row["MEDICATION_RATE_PCT"]
    ref   = row["REFERRAL_RATE_PCT"]
    n     = int(row["PATIENTS_WITH_DX"])
    _p1_rows += (
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
        f'padding:6px 0;border-bottom:1px solid #D6E4F0;font-size:12px">'
        f'<span style="color:#003467;font-weight:600">{grp} <span style="font-weight:400;'
        f'color:#9BAEC8">(n={n})</span></span>'
        f'<span>'
        f'<span style="color:{COLORS["success"]};font-weight:700">{med:.0f}% med</span>'
        f'&nbsp;·&nbsp;'
        f'<span style="color:{COLORS["primary"]};font-weight:700">{ref:.0f}% ref</span>'
        f'</span></div>'
    )

with _c1:
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:10px;'
        f'padding:28px 24px;min-height:380px">'
        f'<div style="font-size:10px;font-weight:800;color:{COLORS["primary"]};'
        f'letter-spacing:2.5px;margin-bottom:14px">PRIORITY 01</div>'
        f'<div style="font-size:16px;font-weight:800;color:#003467;margin-bottom:6px;'
        f'line-height:1.35">Continuity of Care for NCDs</div>'
        f'<div style="font-size:12px;color:#6B8CAE;margin-bottom:18px">'
        f'Treated at camp, not followed up</div>'
        f'<div style="background:#EBF3FB;border-radius:6px;padding:14px 16px;margin-bottom:20px">'
        f'<div style="font-size:9px;font-weight:700;color:{COLORS["primary"]};'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px">Evidence</div>'
        f'{_p1_rows}'
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:8px;font-style:italic">'
        f'High medication, low referral = managed in-camp with no pathway out</div>'
        f'</div>'
        f'<div style="border-top:2px solid {COLORS["primary"]};padding-top:14px">'
        f'<div style="font-size:9px;font-weight:700;color:{COLORS["primary"]};'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Action</div>'
        f'<div style="font-size:12px;color:#003467;line-height:1.6">'
        f'Build a post-camp NCD referral register. Every medicated NCD patient '
        f'needs a named facility for follow-up before the camp closes.'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Priority 02 — Renal care gap ───────────────────────────────────────────────
with _c2:
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:10px;'
        f'padding:28px 24px;min-height:380px">'
        f'<div style="font-size:10px;font-weight:800;color:{COLORS["danger"]};'
        f'letter-spacing:2.5px;margin-bottom:14px">PRIORITY 02</div>'
        f'<div style="font-size:16px;font-weight:800;color:#003467;margin-bottom:6px;'
        f'line-height:1.35">Renal Care Gap</div>'
        f'<div style="font-size:12px;color:#6B8CAE;margin-bottom:18px">'
        f'Tested but not acted on</div>'
        f'<div style="background:#FEF2F2;border-radius:6px;padding:14px 16px;margin-bottom:20px">'
        f'<div style="font-size:9px;font-weight:700;color:{COLORS["danger"]};'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Evidence</div>'
        f'<div style="display:flex;gap:12px;margin-bottom:10px">'
        f'<div style="flex:1;text-align:center">'
        f'<div style="font-size:28px;font-weight:800;color:{COLORS["purple"]}">'
        f'{_renal_inv_pct:.0f}%</div>'
        f'<div style="font-size:10px;color:#6B8CAE;margin-top:2px">investigated</div>'
        f'</div>'
        f'<div style="flex:1;text-align:center">'
        f'<div style="font-size:28px;font-weight:800;color:{COLORS["danger"]}">'
        f'{_renal_ref_pct:.0f}%</div>'
        f'<div style="font-size:10px;color:#6B8CAE;margin-top:2px">referred</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:11px;color:#9BAEC8;font-style:italic">'
        f'{_renal_n} patients · investigations ordered, results documented, no referral generated'
        f'</div>'
        f'</div>'
        f'<div style="border-top:2px solid {COLORS["danger"]};padding-top:14px">'
        f'<div style="font-size:9px;font-weight:700;color:{COLORS["danger"]};'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Action</div>'
        f'<div style="font-size:12px;color:#003467;line-height:1.6">'
        f'Individually review all {_renal_n} renal patients. '
        f'Any abnormal investigation result without a referral is a documented care gap '
        f'that requires immediate follow-up.'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Priority 03 — Documentation quality ───────────────────────────────────────
_p3_bars = ""
for field, pct in _doc_fields:
    _fill_color = COLORS["success"] if pct >= 60 else COLORS["warning"] if pct >= 40 else COLORS["danger"]
    _p3_bars += (
        f'<div style="margin-bottom:10px">'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;'
        f'color:#003467;font-weight:600;margin-bottom:4px">'
        f'<span>{field}</span><span style="color:{_fill_color}">{pct}%</span></div>'
        f'<div style="background:#E5EEF7;border-radius:4px;height:6px">'
        f'<div style="width:{pct}%;background:{_fill_color};border-radius:4px;height:6px"></div>'
        f'</div>'
        f'</div>'
    )

with _c3:
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:10px;'
        f'padding:28px 24px;min-height:380px">'
        f'<div style="font-size:10px;font-weight:800;color:{COLORS["warning"]};'
        f'letter-spacing:2.5px;margin-bottom:14px">PRIORITY 03</div>'
        f'<div style="font-size:16px;font-weight:800;color:#003467;margin-bottom:6px;'
        f'line-height:1.35">Documentation Quality</div>'
        f'<div style="font-size:12px;color:#6B8CAE;margin-bottom:18px">'
        f'Critical fields missing in most records</div>'
        f'<div style="background:#FFFBEB;border-radius:6px;padding:14px 16px;margin-bottom:20px">'
        f'<div style="font-size:9px;font-weight:700;color:{COLORS["warning"]};'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Field fill rates</div>'
        f'{_p3_bars}'
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:6px;font-style:italic">'
        f'Sex and age missing in &gt;55% of records — no reliable stratified analysis possible'
        f'</div>'
        f'</div>'
        f'<div style="border-top:2px solid {COLORS["warning"]};padding-top:14px">'
        f'<div style="font-size:9px;font-weight:700;color:{COLORS["warning"]};'
        f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Action</div>'
        f'<div style="font-size:12px;color:#003467;line-height:1.6">'
        f'Replace free-form intake sheets with a structured form: mandatory sex, age, '
        f'and chief complaint fields before the next camp runs.'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── IDSR signals ───────────────────────────────────────────────────────────────
section_header("Public Health Signals — IDSR")

_IDSR_MAP = {
    "TB": ("TB / Tuberculosis",              "Immediate"),
    "TUBERCULOSIS": ("TB / Tuberculosis",    "Immediate"),
    "PULMONARY TUBERCULOSIS": ("TB / Tuberculosis", "Immediate"),
    "MALARIA": ("Malaria",                   "Weekly"),
    "PLASMODIUM FALCIPARUM MALARIA": ("Malaria", "Weekly"),
    "TYPHOID": ("Typhoid Fever",             "Immediate"),
    "TYPHOID FEVER": ("Typhoid Fever",       "Immediate"),
    "BRUCELLOSIS": ("Brucellosis",           "Immediate"),
    "SYPHILIS": ("Syphilis",                 "Weekly"),
    "HEPATITIS B": ("Hepatitis B",           "Weekly"),
    "HEP B": ("Hepatitis B",                 "Weekly"),
    "HEPATITIS C": ("Hepatitis C",           "Weekly"),
    "HEP C": ("Hepatitis C",                 "Weekly"),
    "MENINGITIS": ("Meningitis",             "Immediate"),
    "MEASLES": ("Measles",                   "Immediate"),
    "CHOLERA": ("Cholera",                   "Immediate"),
    "DYSENTERY": ("Dysentery",               "Weekly"),
    "PERTUSSIS": ("Pertussis",               "Immediate"),
    "WHOOPING COUGH": ("Pertussis",          "Immediate"),
    "KALA AZAR": ("Kala-azar",               "Weekly"),
    "LEISHMANIASIS": ("Kala-azar",           "Weekly"),
}

_idsr_detected = []
if not diag.empty and "TERM_CANONICAL" in diag.columns:
    _idsr_df = diag[diag["TERM_CANONICAL"].str.upper().isin(_IDSR_MAP)].copy()
    if not _idsr_df.empty:
        _idsr_df["_key"]   = _idsr_df["TERM_CANONICAL"].str.upper()
        _idsr_df["_label"] = _idsr_df["_key"].map(lambda k: _IDSR_MAP[k][0])
        _idsr_df["_freq"]  = _idsr_df["_key"].map(lambda k: _IDSR_MAP[k][1])
        _g = (
            _idsr_df.groupby(["_label", "_freq"], as_index=False)["ENCOUNTER_COUNT"]
            .sum()
            .sort_values("ENCOUNTER_COUNT", ascending=False)
        )
        _idsr_detected = _g[_g["ENCOUNTER_COUNT"] > 0].to_dict("records")

if _idsr_detected:
    _freq_color = {"Immediate": COLORS["danger"], "Weekly": COLORS["warning"]}

    _rows_html = ""
    for row in _idsr_detected:
        _fc  = _freq_color.get(row["_freq"], COLORS["muted"])
        _rows_html += (
            f'<tr>'
            f'<td style="padding:10px 14px;font-size:13px;font-weight:600;color:#003467">'
            f'{row["_label"]}</td>'
            f'<td style="padding:10px 14px;text-align:center;font-size:13px;'
            f'font-weight:700;color:{COLORS["primary"]}">{int(row["ENCOUNTER_COUNT"])}</td>'
            f'<td style="padding:10px 14px;text-align:center">'
            f'<span style="background:{_fc}18;color:{_fc};font-size:10px;font-weight:700;'
            f'border-radius:4px;padding:3px 9px;text-transform:uppercase;letter-spacing:1px">'
            f'{row["_freq"]}</span></td>'
            f'<td style="padding:10px 14px;font-size:11px;color:#6B8CAE">'
            f'Clinical review required before IDSR notification</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="background:#FFF1F3;border:1px solid {COLORS["danger"]}40;'
        f'border-left:4px solid {COLORS["danger"]};border-radius:8px;padding:20px 24px;'
        f'margin-bottom:20px">'
        f'<div style="font-size:9px;font-weight:700;color:{COLORS["danger"]};'
        f'text-transform:uppercase;letter-spacing:1.8px;margin-bottom:6px">Alert</div>'
        f'<div style="font-size:13px;font-weight:600;color:#003467">'
        f'{len(_idsr_detected)} notifiable condition{"s" if len(_idsr_detected) > 1 else ""} '
        f'detected in camp records — clinical review and sub-county notification required</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<table style="width:100%;border-collapse:collapse;font-family:Montserrat,sans-serif">'
        f'<thead><tr style="background:#F0F5FA;border-bottom:2px solid #D6E4F0">'
        f'<th style="padding:10px 14px;text-align:left;font-size:10px;font-weight:700;'
        f'color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px">Condition</th>'
        f'<th style="padding:10px 14px;text-align:center;font-size:10px;font-weight:700;'
        f'color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px">Encounters</th>'
        f'<th style="padding:10px 14px;text-align:center;font-size:10px;font-weight:700;'
        f'color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px">Reporting</th>'
        f'<th style="padding:10px 14px;text-align:left;font-size:10px;font-weight:700;'
        f'color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px">Action</th>'
        f'</tr></thead>'
        f'<tbody>{_rows_html}</tbody>'
        f'</table>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:10px;font-style:italic;'
        f'padding:0 4px">Conditions are OCR-extracted and not clinically confirmed. '
        f'Do not submit IDSR notifications based on this output alone — '
        f'review individual records first.</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        '<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
        'padding:20px 24px;font-size:13px;color:#6B8CAE;font-style:italic">'
        'No Kenya IDSR notifiable conditions detected in extracted diagnoses. '
        'Manual clinical review of individual records is still recommended.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── Vitals signals ─────────────────────────────────────────────────────────────
section_header("Vitals Signals — Clinical Alerts")

if not vitals.empty:
    _BP_CRITICAL  = {"HYPERTENSIVE_EMERGENCY", "HYPERTENSIVE_URGENCY"}
    _PL_CRITICAL  = {"SEVERE_TACHYCARDIA", "TACHYCARDIA"}

    _total_vitals  = int((vitals["BP_SIGNAL"].notna() | vitals["PULSE_SIGNAL"].notna()).sum())
    _escalated     = int(vitals["NEEDS_ESCALATION"].sum())
    _bp_emergency  = int((vitals["BP_SIGNAL"] == "HYPERTENSIVE_EMERGENCY").sum())
    _bp_urgency    = int((vitals["BP_SIGNAL"] == "HYPERTENSIVE_URGENCY").sum())
    _pl_critical   = int(vitals["PULSE_SIGNAL"].isin(_PL_CRITICAL).sum())
    _both_critical = int(
        (vitals["BP_SIGNAL"].isin(_BP_CRITICAL) & vitals["PULSE_SIGNAL"].isin(_PL_CRITICAL)).sum()
    )
    _bradycardia   = int(vitals["BRADYCARDIA_FLAG"].sum()) if "BRADYCARDIA_FLAG" in vitals.columns else 0

    if _escalated > 0:
        st.markdown(
            f'<div style="background:#FFF1F3;border:1px solid {COLORS["danger"]}40;'
            f'border-left:4px solid {COLORS["danger"]};border-radius:8px;padding:20px 24px;'
            f'margin-bottom:20px">'
            f'<div style="font-size:9px;font-weight:700;color:{COLORS["danger"]};'
            f'text-transform:uppercase;letter-spacing:1.8px;margin-bottom:6px">Clinical Alert</div>'
            f'<div style="font-size:13px;font-weight:600;color:#003467">'
            f'{_escalated} patient{"s" if _escalated > 1 else ""} with escalation-level vitals — '
            f'hypertensive urgency/emergency or significant tachycardia'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    _va, _vb = st.columns([1, 2])

    with _va:
        st.markdown(
            f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:10px;'
            f'padding:24px;text-align:center">'
            f'<div style="font-size:38px;font-weight:800;color:{COLORS["danger"]};line-height:1">'
            f'{_escalated}</div>'
            f'<div style="font-size:12px;font-weight:600;color:#003467;margin:6px 0 4px">'
            f'require escalation</div>'
            f'<div style="font-size:11px;color:#6B8CAE">'
            f'of {_total_vitals} patients with readable vitals</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with _vb:
        _sig_rows = [
            ("BP Emergency  (>180 systolic or >120 diastolic)", _bp_emergency,  COLORS["danger"]),
            ("BP Urgency    (>160 systolic or >100 diastolic)", _bp_urgency,    COLORS["warning"]),
            ("Tachycardia   (>100 bpm)",                        _pl_critical,   COLORS["purple"]),
            ("Both BP critical + tachycardia",                  _both_critical, "#8B0000"),
            ("Bradycardia   (<50 bpm) — for review",            _bradycardia,   COLORS["muted"]),
        ]
        _sig_html = "".join(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:8px 0;border-bottom:1px solid #EBF3FB">'
            f'<span style="font-size:12px;color:#003467">{lbl}</span>'
            f'<span style="font-size:16px;font-weight:800;color:{col}">{n}</span>'
            f'</div>'
            for lbl, n, col in _sig_rows
        )
        st.markdown(
            f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;border-radius:10px;'
            f'padding:20px 24px">'
            f'<div style="font-size:9px;font-weight:700;color:{COLORS["primary"]};'
            f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">Signal Breakdown</div>'
            f'{_sig_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="font-size:11px;color:#9BAEC8;margin-top:10px;font-style:italic;padding:0 4px">'
        'Vitals coverage: 75% of encounter rows yield readable BP · 43% yield readable pulse. '
        'Patients with no parseable vitals are excluded. Readings are OCR-extracted — '
        'clinical verification required before acting on individual flags.'
        '</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        '<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
        'padding:20px 24px;font-size:13px;color:#6B8CAE;font-style:italic">'
        'No vitals signal data available.'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── Continuity-of-care conditions ─────────────────────────────────────────────
section_header("Conditions Requiring Ongoing Care")

_CHRONIC_GROUPS = {
    "NCD_CARDIOVASCULAR":  "Cardiovascular",
    "NCD_RENAL":           "Renal",
    "NCD_ENDOCRINE":       "Endocrine / Metabolic",
    "NCD_MUSCULOSKELETAL": "Musculoskeletal",
    "NCD_RESPIRATORY":     "Respiratory",
    "NEUROLOGICAL":        "Neurological",
    "MENTAL_HEALTH":       "Mental Health",
}

_chronic_df = cond[cond["DIAGNOSIS_GROUP"].isin(_CHRONIC_GROUPS)].copy()
_chronic_df["_label"]    = _chronic_df["DIAGNOSIS_GROUP"].map(_CHRONIC_GROUPS)
_chronic_df["_gap_n"]    = (
    (_chronic_df["MEDICATION_RATE_PCT"] / 100 * _chronic_df["PATIENTS_WITH_DX"])
    - (_chronic_df["REFERRAL_RATE_PCT"] / 100 * _chronic_df["PATIENTS_WITH_DX"])
).round().astype(int)
_chronic_df = _chronic_df.sort_values("_gap_n", ascending=False)

if not _chronic_df.empty:
    _total_gap_n = int(_chronic_df["_gap_n"].sum())
    _total_med_n = int(
        (_chronic_df["MEDICATION_RATE_PCT"] / 100 * _chronic_df["PATIENTS_WITH_DX"]).sum()
    )

    st.markdown(
        f'<div style="background:#FFF1F3;border-left:4px solid {COLORS["danger"]};'
        f'border-radius:0 8px 8px 0;padding:22px 28px;margin-bottom:20px">'
        f'<div style="font-size:38px;font-weight:800;color:{COLORS["danger"]};line-height:1">'
        f'{_total_gap_n}</div>'
        f'<div style="font-size:13px;font-weight:600;color:#003467;margin:6px 0 4px">'
        f'patients medicated for a chronic condition without a referral</div>'
        f'<div style="font-size:12px;color:#6B8CAE">'
        f'Out of {_total_med_n} medicated across {len(_chronic_df)} chronic condition groups — '
        f'these patients left camp with drugs but no follow-up pathway.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    _pills = "  &nbsp;·&nbsp;  ".join(
        f'<b style="color:#003467">{int(row["_gap_n"])}</b>'
        f' <span style="color:#6B8CAE">{row["_label"]}</span>'
        for _, row in _chronic_df.iterrows()
        if int(row["_gap_n"]) > 0
    )
    st.markdown(
        f'<div style="font-size:12px;padding:4px 2px;line-height:2">{_pills}</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── Data quality (subordinate) ─────────────────────────────────────────────────
section_header("Data Quality — Field Completeness")

_DQ_FIELDS = [
    ("Investigations",     64.0, "Usable for service analysis"),
    ("Treatments",         51.5, "Usable for service analysis"),
    ("Patient sex",        45.0, "Too low for sex-disaggregated reporting"),
    ("Patient age",        44.0, "Too low for age-stratified reporting"),
    ("Diagnoses",          40.0, "Limits burden analysis coverage"),
    ("Clinical summary",   37.1, "Qualitative context only"),
    ("Vitals",             29.3, "Insufficient for clinical audit"),
]

_dq_rows = ""
for field, pct, note in _DQ_FIELDS:
    _fc = COLORS["success"] if pct >= 60 else COLORS["warning"] if pct >= 40 else COLORS["danger"]
    _bar = (
        f'<div style="background:#E5EEF7;border-radius:3px;height:6px;width:120px;display:inline-block;vertical-align:middle">'
        f'<div style="width:{min(pct,100):.0f}%;background:{_fc};border-radius:3px;height:6px"></div>'
        f'</div>'
    )
    _dq_rows += (
        f'<tr style="border-bottom:1px solid #EBF3FB">'
        f'<td style="padding:8px 14px;font-size:12px;font-weight:600;color:#003467">{field}</td>'
        f'<td style="padding:8px 14px;text-align:center;font-size:12px;font-weight:700;color:{_fc}">{pct:.0f}%</td>'
        f'<td style="padding:8px 14px">{_bar}</td>'
        f'<td style="padding:8px 14px;font-size:11px;color:#6B8CAE">{note}</td>'
        f'</tr>'
    )

st.markdown(
    f'<table style="width:100%;border-collapse:collapse;font-family:Montserrat,sans-serif">'
    f'<thead><tr style="background:#F0F5FA;border-bottom:2px solid #D6E4F0">'
    f'<th style="padding:8px 14px;text-align:left;font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px">Field</th>'
    f'<th style="padding:8px 14px;text-align:center;font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px">Fill rate</th>'
    f'<th style="padding:8px 14px;font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px"></th>'
    f'<th style="padding:8px 14px;text-align:left;font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;letter-spacing:1.2px">Implication</th>'
    f'</tr></thead>'
    f'<tbody>{_dq_rows}</tbody>'
    f'</table>',
    unsafe_allow_html=True,
)

st.markdown('<div style="margin-bottom:32px"></div>', unsafe_allow_html=True)

# ── Camp conclusion ────────────────────────────────────────────────────────────
_chronic_count  = int(_chronic_df["PATIENTS_WITH_DX"].sum()) if not _chronic_df.empty else 0
_idsr_count     = len(_idsr_detected)
_idsr_phrase    = (
    f'{_idsr_count} IDSR-notifiable condition{"s" if _idsr_count > 1 else ""} flagged'
    if _idsr_count > 0 else "no IDSR-notifiable conditions flagged"
)

st.markdown(
    f'<div style="background:#F0F5FA;border-radius:10px;padding:28px 32px;'
    f'border:1px solid #D6E4F0;margin-bottom:8px">'
    f'<div style="font-size:9px;font-weight:700;color:{COLORS["primary"]};'
    f'text-transform:uppercase;letter-spacing:2px;margin-bottom:12px">Camp Summary</div>'
    f'<div style="font-size:15px;font-weight:600;color:#003467;line-height:1.75">'
    f'The Mother Francisca Mission health camp generated <b>{total_enc:,} encounter records</b> '
    f'across a musculoskeletal-dominant NCD burden, with '
    f'<b>{_chronic_count} patients</b> requiring ongoing care, '
    f'<b>{_idsr_phrase}</b>, and critical documentation gaps in sex and age fields '
    f'that limit the reliability of all stratified findings.'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<div style="margin-bottom:16px"></div>', unsafe_allow_html=True)

# ── Persistent caveat ──────────────────────────────────────────────────────────
st.markdown(
    '<div style="border-top:1px solid #EBF3FB;margin-top:16px;padding:12px 0 4px;'
    'font-size:11px;color:#9BAEC8;text-align:center;font-style:italic">'
    'Priorities are derived from OCR-extracted documentation. '
    'Clinical review of individual records is required before action is taken.'
    '</div>',
    unsafe_allow_html=True,
)
