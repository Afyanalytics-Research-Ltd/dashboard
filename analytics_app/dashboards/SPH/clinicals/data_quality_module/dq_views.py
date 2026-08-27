"""
sph/clinicals/data_quality_module/dq_views.py
================================================
Data Quality tab — five named dimensions (Consistency, Reliability,
Validity, Timeliness, Uniqueness), each scored 0-100 from a real query in
dq_queries.py, rolled up into one overall Data Quality Score, plus the
specific checks behind each score (e.g. male-gendered ANC visits,
toddler-age-group ANC visits, gender/age_group completeness).
"""

import streamlit as st

import sph.clinicals.data_quality_module.dq_queries as DQ
from sph.clinicals.opd_ipd_module.ui_template import (
    PRIMARY, SUCCESS, DANGER, WARNING, NEUTRAL,
    SURFACE_1, BORDER, TEXT_PRI, TEXT_SEC, TEXT_MUT,
    page_header, section_header, kpi_row, priority_cards,
)

_ACCENT_GOOD = SUCCESS
_ACCENT_WARN = WARNING
_ACCENT_BAD = DANGER
_ACCENT_NEUTRAL = NEUTRAL


def _safe(fn, *args):
    try:
        df = fn(*args)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _accent_for(score):
    if score is None:
        return _ACCENT_NEUTRAL
    if score >= 90:
        return _ACCENT_GOOD
    if score >= 75:
        return _ACCENT_WARN
    return _ACCENT_BAD


def _severity_for(score):
    if score is None:
        return "monitor"
    if score >= 90:
        return "okay"
    if score >= 75:
        return "monitor"
    return "critical"


def compute_scores() -> dict:
    """
    Runs all 5 dimension checks and returns every raw figure + derived
    score in one dict. Shared by render_tab() and the email digest so the
    two never compute this differently.
    """
    df_cons = _safe(DQ.get_dq_consistency)
    df_rel = _safe(DQ.get_dq_reliability_anc_anomalies)
    df_val = _safe(DQ.get_dq_validity)
    df_time = _safe(DQ.get_dq_timeliness)
    df_uniq = _safe(DQ.get_dq_uniqueness)

    consistency_score = reliability_score = validity_score = None
    timeliness_score = uniqueness_score = None

    inconsistent_pct = inconsistent_n = total_patients = None
    if df_cons is not None:
        r = df_cons.iloc[0]
        inconsistent_pct = float(r.get("INCONSISTENT_PCT", 0) or 0)
        inconsistent_n = int(r.get("INCONSISTENT_PATIENTS", 0) or 0)
        total_patients = int(r.get("TOTAL_PATIENTS", 0) or 0)
        consistency_score = round(100 - inconsistent_pct, 1)

    anomalous_pct = male_anc_n = young_anc_n = total_anc = None
    if df_rel is not None:
        r = df_rel.iloc[0]
        anomalous_pct = float(r.get("ANOMALOUS_PCT", 0) or 0)
        male_anc_n = int(r.get("MALE_ANC_VISITS", 0) or 0)
        young_anc_n = int(r.get("TOO_YOUNG_ANC_VISITS", 0) or 0)
        total_anc = int(r.get("TOTAL_ANC_VISITS", 0) or 0)
        reliability_score = round(100 - anomalous_pct, 1)

    missing_gender_pct = missing_age_pct = invalid_gender_n = total_visits_val = None
    if df_val is not None:
        r = df_val.iloc[0]
        missing_gender_pct = float(r.get("MISSING_GENDER_PCT", 0) or 0)
        missing_age_pct = float(r.get("MISSING_AGE_GROUP_PCT", 0) or 0)
        invalid_gender_n = int(r.get("INVALID_GENDER_VALUE", 0) or 0)
        total_visits_val = int(r.get("TOTAL_VISITS", 0) or 0)
        invalid_gender_pct = 100.0 * invalid_gender_n / total_visits_val if total_visits_val else 0
        validity_score = round(100 - ((missing_gender_pct + missing_age_pct + invalid_gender_pct) / 3), 1)

    days_since_last = blind_spot_visits = None
    if df_time is not None:
        r = df_time.iloc[0]
        days_since_last = int(r.get("DAYS_SINCE_LAST_VISIT", 0) or 0)
        blind_spot_visits = int(r.get("BLIND_SPOT_VISITS", 0) or 0)
        timeliness_score = round(max(0, 100 - days_since_last * (100 / 90)), 1)

    dup_pct = dup_keys = excess_rows = total_keys = None
    if df_uniq is not None:
        r = df_uniq.iloc[0]
        dup_pct = float(r.get("DUPLICATE_KEY_PCT", 0) or 0)
        dup_keys = int(r.get("DUPLICATE_KEYS", 0) or 0)
        excess_rows = int(r.get("EXCESS_DUPLICATE_ROWS", 0) or 0)
        total_keys = int(r.get("TOTAL_KEYS", 0) or 0)
        uniqueness_score = round(100 - dup_pct, 1)

    _scores = [s for s in (consistency_score, reliability_score, validity_score,
                            timeliness_score, uniqueness_score) if s is not None]
    overall_score = round(sum(_scores) / len(_scores), 1) if _scores else None

    return dict(
        consistency_score=consistency_score, reliability_score=reliability_score,
        validity_score=validity_score, timeliness_score=timeliness_score,
        uniqueness_score=uniqueness_score, overall_score=overall_score,
        inconsistent_pct=inconsistent_pct, inconsistent_n=inconsistent_n, total_patients=total_patients,
        anomalous_pct=anomalous_pct, male_anc_n=male_anc_n, young_anc_n=young_anc_n, total_anc=total_anc,
        missing_gender_pct=missing_gender_pct, missing_age_pct=missing_age_pct,
        invalid_gender_n=invalid_gender_n, total_visits_val=total_visits_val,
        days_since_last=days_since_last, blind_spot_visits=blind_spot_visits,
        dup_pct=dup_pct, dup_keys=dup_keys, excess_rows=excess_rows, total_keys=total_keys,
    )


def render_tab() -> None:
    page_header(
        "Data Quality",
        subtitle="St. Peter's Orthopaedic Hospital — how much the underlying visit data can be trusted, "
                  "scored across five dimensions.",
    )

    with st.spinner("Running data quality checks…"):
        s = compute_scores()

    consistency_score = s["consistency_score"]; reliability_score = s["reliability_score"]
    validity_score = s["validity_score"]; timeliness_score = s["timeliness_score"]
    uniqueness_score = s["uniqueness_score"]; overall_score = s["overall_score"]
    inconsistent_pct = s["inconsistent_pct"]; inconsistent_n = s["inconsistent_n"]; total_patients = s["total_patients"]
    anomalous_pct = s["anomalous_pct"]; male_anc_n = s["male_anc_n"]; young_anc_n = s["young_anc_n"]; total_anc = s["total_anc"]
    missing_gender_pct = s["missing_gender_pct"]; missing_age_pct = s["missing_age_pct"]
    invalid_gender_n = s["invalid_gender_n"]; total_visits_val = s["total_visits_val"]
    days_since_last = s["days_since_last"]; blind_spot_visits = s["blind_spot_visits"]
    dup_pct = s["dup_pct"]; dup_keys = s["dup_keys"]; excess_rows = s["excess_rows"]; total_keys = s["total_keys"]

    # ── Overall score ────────────────────────────────────────────────────────
    section_header("Overall data quality score")
    kpi_row([
        {"label": "Data quality score", "value": f"{overall_score:.0f} / 100" if overall_score is not None else "—",
         "delta": "Average across all 5 dimensions", "accent_color": _accent_for(overall_score)},
        {"label": "Consistency", "value": f"{consistency_score:.0f}" if consistency_score is not None else "—",
         "delta": f"{inconsistent_n:,} of {total_patients:,} patients have a gender that changes across visits"
                  if inconsistent_n is not None else "", "accent_color": _accent_for(consistency_score)},
        {"label": "Reliability", "value": f"{reliability_score:.0f}" if reliability_score is not None else "—",
         "delta": f"{male_anc_n + young_anc_n:,} of {total_anc:,} ANC visits carry an implausible profile"
                  if male_anc_n is not None else "", "accent_color": _accent_for(reliability_score)},
    ])
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    kpi_row([
        {"label": "Validity", "value": f"{validity_score:.0f}" if validity_score is not None else "—",
         "delta": f"{missing_gender_pct:.1f}% missing gender, {missing_age_pct:.1f}% missing age group"
                  if missing_gender_pct is not None else "", "accent_color": _accent_for(validity_score)},
        {"label": "Timeliness", "value": f"{timeliness_score:.0f}" if timeliness_score is not None else "—",
         "delta": f"Most recent visit is {days_since_last:,} days old" if days_since_last is not None else "",
         "accent_color": _accent_for(timeliness_score)},
        {"label": "Uniqueness", "value": f"{uniqueness_score:.0f}" if uniqueness_score is not None else "—",
         "delta": f"{dup_keys:,} duplicate visit keys ({excess_rows:,} excess rows)"
                  if dup_keys is not None else "", "accent_color": _accent_for(uniqueness_score)},
    ])

    # ── What each dimension is checking ─────────────────────────────────────
    section_header("What we're checking, and what we found")

    cards = []

    cards.append({
        "label": "Consistency",
        "title": "Does a patient's gender ever change across their own visits?",
        "body": (f"{inconsistent_n:,} of {total_patients:,} patients ({inconsistent_pct:.2f}%) have more than "
                 "one distinct gender value recorded across their visit history — the same person shouldn't "
                 "flip gender between records logged in different systems or on different dates."
                 if inconsistent_n is not None else
                 "Checks whether a patient's recorded gender is uniform across every visit they have — a "
                 "changing value points to a data-entry or system-merge error, not a real-world change."),
        "severity": _severity_for(consistency_score),
        "source": "STG_VISITS, grouped by patient_id",
    })

    cards.append({
        "label": "Reliability",
        "title": "Male patients and toddlers recorded as ANC visits",
        "body": (f"{male_anc_n:,} ANC-flagged visits are recorded against a male patient, and {young_anc_n:,} "
                  "against a Toddler (0-4) or Child (5-12) age group — clinically implausible combinations "
                  f"that make up {anomalous_pct:.2f}% of all {total_anc:,} ANC visits. These are the "
                  "clearest sign that a segment/department flag can't always be trusted at face value."
                 if male_anc_n is not None else
                 "Checks ANC-flagged visits for gender or age-group values that are clinically implausible "
                 "for antenatal care — e.g. a male patient or a toddler-age patient."),
        "severity": _severity_for(reliability_score),
        "source": "STG_VISITS joined to ANC burden-group flag",
    })

    cards.append({
        "label": "Validity",
        "title": "Gender and age-group field completeness and format",
        "body": (f"{missing_gender_pct:.1f}% of visits are missing gender and {missing_age_pct:.1f}% are "
                  f"missing age group entirely; a further {invalid_gender_n:,} visits carry a gender value "
                  "outside the expected Male/Female set."
                 if missing_gender_pct is not None else
                 "Checks that gender and age_group are populated on every visit, and that populated values "
                 "fall inside the expected value set."),
        "severity": _severity_for(validity_score),
        "source": "STG_VISITS",
    })

    cards.append({
        "label": "Timeliness",
        "title": "How fresh is the most recent visit record?",
        "body": (f"The latest visit on file is {days_since_last:,} days old. {blind_spot_visits:,} visits sit "
                  "in the untracked 31-90 day window between the standard readmission and long-term "
                  "follow-up checks, where activity happens but isn't captured by either KPI."
                 if days_since_last is not None else
                 "Checks the gap between today and the most recent visit date on file, plus how many visits "
                 "fall into the untracked 31-90 day follow-up window."),
        "severity": _severity_for(timeliness_score),
        "source": "STG_VISITS, MAX(visit_date)",
    })

    cards.append({
        "label": "Uniqueness",
        "title": "Duplicate (visit_id, source_system) records",
        "body": (f"{dup_keys:,} of {total_keys:,} visit keys ({dup_pct:.2f}%) appear more than once in "
                  f"STG_VISITS, contributing {excess_rows:,} excess rows — each (visit_id, source_system) "
                  "pair should be unique."
                 if dup_keys is not None else
                 "Checks that every (visit_id, source_system) pair — the composite key every other module "
                 "on this dashboard joins on — appears exactly once in STG_VISITS."),
        "severity": _severity_for(uniqueness_score),
        "source": "STG_VISITS, grouped by (visit_id, source_system)",
    })

    priority_cards(cards)
