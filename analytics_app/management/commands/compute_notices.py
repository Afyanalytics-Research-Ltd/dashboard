"""
Compute KSH active alert notices and send the executive digest email.

Usage:
    python manage.py compute_notices
    python manage.py compute_notices --dry-run   # print notices, no email

Schedule (C2 — pending): weekly Monday 07:00 EAT
    0 7 * * 1   /path/venv/bin/python /path/dashboard/manage.py compute_notices

Rules implemented: 3, 4, 10–35 (Rules 1/2/5–9 are hidden readmissions, suppressed).
Oct 2025 excluded globally (_OCT_2025_GAP) — pipeline gap (Inv 32).
"""

import logging
import os
import sys

import pandas as pd
from django.core.management.base import BaseCommand

# Resolve ksh/ directory so 'from facility_utilization...' imports work.
# This file: analytics_app/management/commands/compute_notices.py
# Target:    analytics_app/dashboards/ksh/facility_utilization/
_KSH_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dashboards', 'ksh')
)
if _KSH_DIR not in sys.path:
    sys.path.insert(0, _KSH_DIR)

from facility_utilization.queries import (  # noqa: E402
    q_theatre_trend,
    q_dialysis_trend,
    q_ward_admissions_monthly,
    q_ward_los_monthly,
    q_ward_discharge_monthly,
    q_doctor_workload_monthly,
    q_lab_monthly,
    q_btr_bti_monthly,
    q_admission_tat_monthly,
    q_revpab_private_monthly,
    q_cd12_monthly_rate,
    q_imaging_trend,
)
from facility_utilization.notifier import send_digest, get_recipients, write_current_notices  # noqa: E402

logger = logging.getLogger(__name__)

# ── Shared constants (mirrors dashboard.py) ───────────────────────────────

_OCT_2025_GAP = "2025-10-01"
_FACILITY     = "KSH"

# Rule 3 — Theatre completion
_THEATRE_WATCH, _THEATRE_CRIT = 85.0, 75.0

# Rules 10–14 — Ward traffic volume (ward_category keys after normalisation)
_TRAFFIC_WATCH = {
    "Medical Female": 40,  "Medical Male": 25, "Maternity": 20,
    "Paediatric": 32,      "Private/Amenity": 18,
}
_TRAFFIC_CRIT = {
    "Medical Female": 45,  "Medical Male": None, "Maternity": 25,
    "Paediatric": 37,      "Private/Amenity": 22,
}

# Rules 15–19 — Median LOS
_LOS_WATCH = {
    "Medical Female": 5.0, "Medical Male": 5.0, "Maternity": 4.0,
    "Paediatric": 3.5,     "Private/Amenity": 5.5,
}
_LOS_CRIT = {
    "Medical Female": 7.0, "Medical Male": 7.0, "Maternity": 6.0,
    "Paediatric": 5.0,     "Private/Amenity": 8.0,
}

# Rules 20–24 — Patient Request discharge rate
_PR_WATCH = {
    "Medical Female": 68, "Medical Male": 62, "Maternity": 82,
    "Paediatric": 68,     "Private/Amenity": 75,
}
_PR_CRIT = {
    "Medical Female": 78, "Medical Male": 72, "Maternity": None,
    "Paediatric": 78,     "Private/Amenity": None,
}

# Rules 25, 26 — Doctor concentration + burnout
_DOC_CONC_WATCH, _DOC_CONC_CRIT = 40.0, 50.0

# Rules 27, 28 — Lab volume + abnormal rate
_LAB_VOL_WATCH, _LAB_VOL_CRIT       = 430,  350
_LAB_ABNORM_WATCH, _LAB_ABNORM_CRIT = 9.0, 11.0

# Rule 29 — Ward Idle BTR+BTI (per-ward P25/P75 floors from Inv 46)
_BTR_P25 = {
    "General Female": 0.62, "General Male": 0.69, "General Maternity": 0.42,
    "Pediatric General": 0.53, "Private Female": 0.45,
    "Private Male": 0.25, "Private Maternity": 0.20,
}
_BTI_P75 = {
    "General Female": 46.6,  "General Male": 41.7,  "General Maternity": 71.7,
    "Pediatric General": 58.2, "Private Female": 66.8,
    "Private Male": 118.6, "Private Maternity": 147.0,
}

# Rule 30 — Admission TAT fast-track rate
_TAT_WATCH, _TAT_CRIT = 45.0, 35.0

# Rule 31 — BOR low occupancy (per-ward P25 from Inv 48)
_BOR_P25 = {
    "General Female": 6.5, "General Male": 7.3, "General Maternity": 3.2,
    "Pediatric General": 4.0, "Private Female": 3.7,
    "Private Male": 1.8, "Private Maternity": 3.2,
}

# Rule 32 — Private ward revenue drop
_REVPAB_WATCH_DROP = 25.0

# Rule 33 — Physician workload absolute P90 (Inv 50; makinyi excluded — departed Dec 2025)
_DOC_WL_TRACKED = frozenset({"eawando", "lowino", "jogutu"})
_DOC_WL_P90     = {"eawando": 795, "lowino": 595, "jogutu": 378}

# Rule 34 — CD12 creatinine non-admission
_CD12_WATCH, _CD12_CRIT, _CD12_MIN_EVTS = 50.0, 65.0, 8

# Rule 35 — CT imaging volume drop
_IMAGING_WATCH_PCT, _IMAGING_CRIT_PCT = 80.0, 65.0


# ── Helpers ───────────────────────────────────────────────────────────────

def _two_consec(vals, threshold, direction="above"):
    if len(vals) < 2:
        return False
    if direction == "above":
        return vals[-2] > threshold and vals[-1] > threshold
    return vals[-2] < threshold and vals[-1] < threshold


def _notice(level, title, metric, action):
    return {"level": level, "title": title, "metric": metric, "action": action}


def _ward_key(ward_category):
    """Map rpt_* ward_category string to threshold dict key (keyword match)."""
    if not ward_category:
        return None
    cat = str(ward_category).lower()
    if "medical" in cat and "female" in cat:
        return "Medical Female"
    if "medical" in cat and "male" in cat:
        return "Medical Male"
    if "maternity" in cat and "private" not in cat:
        return "Maternity"
    if "paediatric" in cat or "pediatric" in cat:
        return "Paediatric"
    if "private" in cat or "amenity" in cat:
        return "Private/Amenity"
    return None


def _safe_load(fn, *args, **kwargs):
    """Call a query function, normalise column names to lowercase, return empty df on error."""
    try:
        df = fn(*args, **kwargs)
        if df is None or df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        return df
    except Exception as exc:
        logger.warning("Query %s failed: %s", fn.__name__, exc)
        return pd.DataFrame()


def _consec_adjacent(df, month_col):
    """Return True if the last 2 rows in df are adjacent calendar months."""
    if len(df) < 2:
        return False
    m = pd.to_datetime(df[month_col]).tolist()
    return (m[-1].year * 12 + m[-1].month) - (m[-2].year * 12 + m[-2].month) == 1


# ── Rule evaluators ───────────────────────────────────────────────────────

def _rule3_theatre(df):
    """Theatre completion — trailing 3-month average below target."""
    if df.empty or "completion_rate_pct" not in df.columns:
        return []
    d = df[df["session_month"].astype(str) != _OCT_2025_GAP].sort_values("session_month")
    if len(d) < 3:
        return []
    avg   = round(float(d.tail(3)["completion_rate_pct"].mean()), 1)
    mo    = pd.to_datetime(d["session_month"].iloc[-1]).strftime("%b %Y")
    gap   = round(85.0 - avg, 1)
    if avg < _THEATRE_CRIT:
        return [_notice("CRITICAL", "Theatre: Completion Below Target",
                        f"{avg}% 3-mo avg · target ≥85% · latest {mo}",
                        f"Ops lead: review cancellations and no-shows — {gap}pp below WATCH")]
    if avg < _THEATRE_WATCH:
        return [_notice("WATCH", "Theatre: Completion Below Target",
                        f"{avg}% 3-mo avg · {mo}",
                        f"Ops lead: monitor — {gap}pp below 85% threshold")]
    return []


def _rule4_dialysis(df):
    """Dialysis idle >= 6 months."""
    if df.empty or "session_month" not in df.columns:
        return []
    ksh = df[df["facility"].str.upper() == "KISUMU_CLEAN"].sort_values("session_month")
    active = ksh[ksh["total_sessions"] > 0] if not ksh.empty else pd.DataFrame()
    if active.empty:
        return []
    last        = pd.to_datetime(active["session_month"].iloc[-1])
    today       = pd.Timestamp.today()
    months_idle = (today.year * 12 + today.month) - (last.year * 12 + last.month)
    if months_idle >= 6:
        return [_notice("WATCH", "Dialysis Equipment Idle",
                        f"{months_idle} months idle · last session {last.strftime('%b %Y')}",
                        "Ops lead: referral pipeline needed — critical renal patients confirmed (see CD12)")]
    return []


def _rules10_14_traffic(df):
    """Ward traffic volume — 2 consecutive months above WATCH/CRITICAL."""
    notices = []
    if df.empty or "ward_category" not in df.columns:
        return notices
    d = df[
        (df["facility"].str.upper() == "KISUMU_CLEAN") &
        (df["admission_month"].astype(str) != _OCT_2025_GAP)
    ].sort_values("admission_month")
    for cat in d["ward_category"].dropna().unique():
        key   = _ward_key(cat)
        w_thr = _TRAFFIC_WATCH.get(key)
        c_thr = _TRAFFIC_CRIT.get(key)
        if w_thr is None:
            continue
        wd  = d[d["ward_category"] == cat].tail(2)
        if len(wd) < 2 or not _consec_adjacent(wd, "admission_month"):
            continue
        vals = wd["admissions"].tolist()
        mo   = pd.to_datetime(wd["admission_month"].iloc[-1]).strftime("%b %Y")
        if c_thr and _two_consec(vals, c_thr, "above"):
            notices.append(_notice("CRITICAL", f"Ward Traffic — {cat}",
                f"{int(vals[-1])} admissions · {mo} · CRITICAL >{c_thr}/mo",
                "Ward manager: sustained high volume — review bed allocation"))
        elif _two_consec(vals, w_thr, "above"):
            notices.append(_notice("WATCH", f"Ward Traffic — {cat}",
                f"{int(vals[-1])} admissions · {mo} · above {w_thr}/mo 2 months",
                "Ward manager: monitor — approaching capacity threshold"))
    return notices


def _rules15_19_los(df):
    """Median LOS deviation — 2 consecutive months above WATCH/CRITICAL."""
    notices = []
    if df.empty or "ward_category" not in df.columns:
        return notices
    d = df[
        (df["facility"].str.upper() == "KISUMU_CLEAN") &
        (df["admission_month"].astype(str) != _OCT_2025_GAP)
    ].sort_values("admission_month")
    for cat in d["ward_category"].dropna().unique():
        key   = _ward_key(cat)
        w_thr = _LOS_WATCH.get(key)
        c_thr = _LOS_CRIT.get(key)
        if w_thr is None:
            continue
        wd  = d[d["ward_category"] == cat].tail(2)
        if len(wd) < 2 or not _consec_adjacent(wd, "admission_month"):
            continue
        vals = wd["median_los_days"].tolist()
        mo   = pd.to_datetime(wd["admission_month"].iloc[-1]).strftime("%b %Y")
        if c_thr and _two_consec(vals, c_thr, "above"):
            notices.append(_notice("CRITICAL", f"Ward LOS — {cat}",
                f"{vals[-1]:.1f}d median · {mo} · CRITICAL >{c_thr}d",
                "Clinical + ward manager: investigate discharge delays — sustained above critical"))
        elif _two_consec(vals, w_thr, "above"):
            notices.append(_notice("WATCH", f"Ward LOS — {cat}",
                f"{vals[-1]:.1f}d median · {mo} · above {w_thr}d 2 months",
                "Clinical + ward manager: review discharge pathway — extended stays consuming beds"))
    return notices


def _rules20_24_patient_request(df):
    """Patient Request discharge rate — 2 consecutive months above WATCH/CRITICAL."""
    notices = []
    if df.empty or "ward_category" not in df.columns:
        return notices
    d = df[
        (df["facility"].str.upper() == "KISUMU_CLEAN") &
        (df["admission_month"].astype(str) != _OCT_2025_GAP)
    ].sort_values("admission_month")
    for cat in d["ward_category"].dropna().unique():
        key   = _ward_key(cat)
        w_thr = _PR_WATCH.get(key)
        c_thr = _PR_CRIT.get(key)
        if w_thr is None:
            continue
        wd  = d[d["ward_category"] == cat].tail(2)
        if len(wd) < 2 or not _consec_adjacent(wd, "admission_month"):
            continue
        if wd["total_admissions"].min() < 10:
            continue
        vals = wd["patient_request_pct"].tolist()
        mo   = pd.to_datetime(wd["admission_month"].iloc[-1]).strftime("%b %Y")
        if c_thr and _two_consec(vals, c_thr, "above"):
            notices.append(_notice("CRITICAL", f"Patient Request Discharge — {cat}",
                f"{vals[-1]:.1f}% self-discharge · {mo} · CRITICAL >{c_thr}%",
                "Clinical lead: patient request rate is leading indicator for 30-day readmissions"))
        elif _two_consec(vals, w_thr, "above"):
            notices.append(_notice("WATCH", f"Patient Request Discharge — {cat}",
                f"{vals[-1]:.1f}% self-discharge · {mo} · above {w_thr}% 2 months",
                "Clinical lead: sustained self-discharge — review patient request drivers"))
    return notices


def _rule25_doctor_concentration(df):
    """Single doctor > 40%/50% share of all evaluation visits this month."""
    if df.empty or "username" not in df.columns:
        return []
    d = df[df["visit_month"].astype(str) != _OCT_2025_GAP].sort_values("visit_month")
    if d.empty:
        return []
    latest = d["visit_month"].max()
    ld     = d[d["visit_month"] == latest]
    total  = float(ld["monthly_visits"].sum())
    if total == 0:
        return []
    top   = ld.nlargest(1, "monthly_visits").iloc[0]
    share = round(100.0 * float(top["monthly_visits"]) / total, 1)
    mo    = pd.to_datetime(latest).strftime("%b %Y")
    if share > _DOC_CONC_CRIT:
        return [_notice("CRITICAL", "Doctor Concentration Risk",
                        f"{share}% of all visits · {mo}",
                        "Ops lead: single-doctor dependency — any absence disrupts patient intake pathway")]
    if share > _DOC_CONC_WATCH:
        return [_notice("WATCH", "Doctor Concentration Risk",
                        f"{share}% of all visits · {mo}",
                        "Ops lead: flag concentration risk — monitor for sustained pattern")]
    return []


def _rule26_doctor_burnout(df):
    """Individual doctor > 150% of personal 3-month avg for 2 consecutive months."""
    notices = []
    if df.empty or "username" not in df.columns:
        return notices
    d = df[df["visit_month"].astype(str) != _OCT_2025_GAP].sort_values("visit_month")
    for doc in d["username"].unique():
        dd = d[d["username"] == doc].tail(5)
        if len(dd) < 4:
            continue
        base_avg = float(dd.iloc[:-2]["monthly_visits"].mean())
        if base_avg == 0:
            continue
        tail2 = dd.tail(2)
        if not _consec_adjacent(tail2, "visit_month"):
            continue
        vals = tail2["monthly_visits"].tolist()
        pcts = [v / base_avg * 100 for v in vals]
        if _two_consec(pcts, 150, "above"):
            mo = pd.to_datetime(tail2["visit_month"].iloc[-1]).strftime("%b %Y")
            notices.append(_notice("WATCH", f"Doctor Burnout Signal — {doc}",
                f"{int(vals[-1])} visits · {mo} · {round(pcts[-1])}% of personal avg",
                "Ops lead: volume unsustainable — 2 consecutive months above 150% of baseline"))
    return notices


def _rules27_28_lab(df):
    """Lab volume drop (Rule 27) and abnormal rate spike (Rule 28)."""
    notices = []
    if df.empty:
        return notices
    d = df[df["lab_month"].astype(str) != _OCT_2025_GAP].sort_values("lab_month")
    # Rule 27 — volume
    if "distinct_visits" in d.columns:
        tail2 = d.tail(2)
        latest_vol = int(tail2["distinct_visits"].iloc[-1])
        mo = pd.to_datetime(tail2["lab_month"].iloc[-1]).strftime("%b %Y")
        if latest_vol < _LAB_VOL_CRIT:
            notices.append(_notice("CRITICAL", "Lab Volume Drop",
                f"{latest_vol} visits · {mo} · CRITICAL <{_LAB_VOL_CRIT}/mo",
                "Lab/ops lead: severe drop — confirm equipment and staffing status immediately"))
        elif len(tail2) == 2 and _two_consec(tail2["distinct_visits"].tolist(), _LAB_VOL_WATCH, "below"):
            notices.append(_notice("WATCH", "Lab Volume Drop",
                f"{latest_vol} visits · {mo} · below {_LAB_VOL_WATCH}/mo 2 months",
                "Lab/ops lead: confirm lab capacity — sustained below WATCH threshold"))
    # Rule 28 — abnormal rate
    if "abnormal_pct" in d.columns:
        tail2 = d.tail(2)
        latest_ab = float(tail2["abnormal_pct"].iloc[-1])
        mo = pd.to_datetime(tail2["lab_month"].iloc[-1]).strftime("%b %Y")
        if latest_ab > _LAB_ABNORM_CRIT:
            notices.append(_notice("CRITICAL", "Lab Abnormal Rate Spike",
                f"{latest_ab:.1f}% abnormal · {mo} · CRITICAL >{_LAB_ABNORM_CRIT}%",
                "Clinical lead: high abnormal rate — cross-reference with ward admissions"))
        elif len(tail2) == 2 and _two_consec(tail2["abnormal_pct"].tolist(), _LAB_ABNORM_WATCH, "above"):
            notices.append(_notice("WATCH", "Lab Abnormal Rate Spike",
                f"{latest_ab:.1f}% abnormal · {mo} · above {_LAB_ABNORM_WATCH}% 2 months",
                "Clinical lead: rising abnormal rate — predict higher admission demand"))
    return notices


def _rule29_ward_idle(df):
    """BTR < P25 AND BTI > P75 latest month per ward. Returns (notices, fired_wards)."""
    notices     = []
    fired_wards = set()
    if df.empty or "ward_name" not in df.columns:
        return notices, fired_wards
    d = df[df["month"].astype(str) != _OCT_2025_GAP].sort_values("month")
    for ward in d["ward_name"].unique():
        p25_btr = _BTR_P25.get(ward)
        p75_bti = _BTI_P75.get(ward)
        if p25_btr is None or p75_bti is None:
            continue
        wd = d[d["ward_name"] == ward].tail(1)
        if wd.empty:
            continue
        btr = float(wd["btr"].iloc[0])
        bti = float(wd["bti_days"].iloc[0])
        mo  = pd.to_datetime(wd["month"].iloc[0]).strftime("%b %Y")
        if btr < p25_btr and bti > p75_bti:
            fired_wards.add(ward)
            notices.append(_notice("WATCH", f"Ward Idle — {ward}",
                f"BTR {btr:.2f} (floor {p25_btr}) · BTI {bti:.1f}d (ceiling {p75_bti}d) · {mo}",
                f"Ward manager: low admissions + long bed idle time in {mo}"))
    return notices, fired_wards


def _rule30_admission_tat(df):
    """TAT fast-track — WATCH 2-consec <45%, CRITICAL single <35%."""
    if df.empty or "fast_pct" not in df.columns:
        return []
    d = df[df["tat_month"].astype(str) != _OCT_2025_GAP].sort_values("tat_month")
    if d.empty:
        return []
    latest = float(d["fast_pct"].iloc[-1])
    p50    = float(d["p50_tat_min"].iloc[-1]) if "p50_tat_min" in d.columns else None
    mo     = pd.to_datetime(d["tat_month"].iloc[-1]).strftime("%b %Y")
    p50s   = f" · p50 TAT {int(p50)} min" if p50 else ""
    if latest < _TAT_CRIT:
        return [_notice("CRITICAL", "Admission TAT Deterioration",
                        f"{latest:.1f}% fast-track (<60 min) · {mo}{p50s}",
                        f"Ops lead: only {latest:.1f}% admitted within 60 min — review ED-to-ward handoff immediately")]
    tail2 = d.tail(2)
    if len(tail2) == 2 and _consec_adjacent(tail2, "tat_month"):
        if _two_consec(tail2["fast_pct"].tolist(), _TAT_WATCH, "below"):
            return [_notice("WATCH", "Admission TAT Deterioration",
                            f"{latest:.1f}% fast-track · {mo}{p50s} · below {_TAT_WATCH}% 2 months",
                            "Ops lead: sustained TAT pressure — review ED-to-ward handoff process")]
    return []


def _rule31_bor_low(df, fired_wards):
    """BOR < ward P25 for 2 consecutive months. Suppressed if Rule 29 fired same ward."""
    notices = []
    if df.empty or "bor_pct" not in df.columns:
        return notices
    d = df[df["month"].astype(str) != _OCT_2025_GAP].sort_values("month")
    for ward in d["ward_name"].unique():
        if ward in fired_wards:
            continue
        p25 = _BOR_P25.get(ward)
        if p25 is None:
            continue
        wd  = d[d["ward_name"] == ward].tail(2)
        if len(wd) < 2 or not _consec_adjacent(wd, "month"):
            continue
        vals = wd["bor_pct"].tolist()
        if _two_consec(vals, p25, "below"):
            mo = pd.to_datetime(wd["month"].iloc[-1]).strftime("%b %Y")
            notices.append(_notice("WATCH", f"Low BOR — {ward}",
                f"{vals[-1]:.1f}% occupancy · floor {p25}% · {mo} · 2 consecutive months",
                "Ward manager: sustained low occupancy — review admission referral patterns"))
    return notices


def _rule32_private_revenue(df):
    """Private ward revenue > 25% below 3-month rolling average."""
    if df.empty or "total_revenue" not in df.columns:
        return []
    d = df.sort_values("admission_month")
    if len(d) < 4:
        return []
    tail4       = d.tail(4)
    rolling_avg = float(tail4.head(3)["total_revenue"].mean())
    latest      = float(tail4.iloc[-1]["total_revenue"])
    mo          = pd.to_datetime(tail4.iloc[-1]["admission_month"]).strftime("%b %Y")
    if rolling_avg <= 0:
        return []
    drop_pct = round(100.0 * (1 - latest / rolling_avg), 1)
    if drop_pct > _REVPAB_WATCH_DROP:
        return [_notice("WATCH", "Private Ward Revenue Drop",
                        f"KES {int(latest):,} · {mo} · {drop_pct}% below 3-mo avg (KES {int(rolling_avg):,})",
                        "Finance lead: review private ward admission volume — revenue below rolling baseline")]
    return []


def _rule33_physician_workload(df):
    """eawando/lowino/jogutu: monthly visits > personal P90 for 2 consecutive months."""
    notices = []
    if df.empty or "username" not in df.columns:
        return notices
    d = df[df["visit_month"].astype(str) != _OCT_2025_GAP].sort_values("visit_month")
    for doc in _DOC_WL_TRACKED:
        p90 = _DOC_WL_P90.get(doc)
        if p90 is None:
            continue
        dd = d[d["username"] == doc].tail(2)
        if len(dd) < 2 or not _consec_adjacent(dd, "visit_month"):
            continue
        vals = dd["monthly_visits"].tolist()
        if _two_consec(vals, p90, "above"):
            mo = pd.to_datetime(dd["visit_month"].iloc[-1]).strftime("%b %Y")
            notices.append(_notice("WATCH", f"Physician Workload — {doc}",
                f"{int(vals[-1])} visits · {mo} · P90 {p90} · 2 consecutive months",
                f"Ops lead: {doc} sustained above P90 — volume may affect evaluation quality"))
    return notices


def _rule34_cd12(df):
    """CD12 creatinine non-admission. Returns (notices, clinical_notes)."""
    notices        = []
    clinical_notes = []
    if df.empty or "non_admission_rate_pct" not in df.columns:
        return notices, clinical_notes
    d    = df[df["critical_month"].astype(str) != _OCT_2025_GAP].sort_values("critical_month")
    qual = d[d["total_critical"] >= _CD12_MIN_EVTS]
    if qual.empty:
        return notices, clinical_notes
    row    = qual.iloc[-1]
    mo_ts  = pd.to_datetime(row["critical_month"])
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=3)
    if mo_ts < cutoff:
        return notices, clinical_notes
    rate    = float(row["non_admission_rate_pct"])
    total   = int(row["total_critical"])
    not_adm = int(row["not_admitted"])
    mo      = mo_ts.strftime("%b %Y")
    # Always surface as clinical note (routine monitoring regardless of threshold)
    if len(d) >= 3:
        _3mo     = d.tail(3)
        avg_rate = round(_3mo["not_admitted"].sum() / max(_3mo["total_critical"].sum(), 1) * 100, 1)
        clinical_notes.append({
            "title":  "Renal Pathway — Critical Creatinine",
            "metric": f"{avg_rate}% non-admission rate (3-mo avg) · {total} patients {mo}",
            "note":   "For review: Clinical / Medical Lead",
        })
    if rate > _CD12_CRIT:
        notices.append(_notice("CRITICAL", "CD12 — Creatinine Non-Admission",
                               f"{rate:.1f}% not admitted · {not_adm}/{total} patients · {mo}",
                               "Clinical lead: review creatinine-flagged patient admission decisions — rate above 65%"))
    elif rate > _CD12_WATCH:
        notices.append(_notice("WATCH", "CD12 — Creatinine Non-Admission",
                               f"{rate:.1f}% not admitted · {not_adm}/{total} patients · {mo}",
                               "Clinical lead: review creatinine-flagged patient admission decisions"))
    return notices, clinical_notes


def _rule35_ct_imaging(df):
    """CT/Angio sessions < 80% (WATCH) / < 65% (CRITICAL) of 3-month rolling average."""
    if df.empty or "modality" not in df.columns:
        return []
    ct = df[df["modality"] == "CT / Angio"].copy()
    current_mo = pd.Timestamp.today().replace(day=1)
    ct = ct[
        (pd.to_datetime(ct["revenue_month"]) < current_mo) &
        (ct["revenue_month"].astype(str) != _OCT_2025_GAP)
    ].sort_values("revenue_month")
    if len(ct) < 4:
        return []
    tail4       = ct.tail(4)
    rolling_avg = float(tail4.head(3)["sessions"].mean())
    latest      = int(tail4.iloc[-1]["sessions"])
    mo          = pd.to_datetime(tail4.iloc[-1]["revenue_month"]).strftime("%b %Y")
    if rolling_avg <= 0:
        return []
    pct  = round(100.0 * latest / rolling_avg, 1)
    drop = round(100.0 - pct, 1)
    if pct < _IMAGING_CRIT_PCT:
        return [_notice("CRITICAL", "CT Imaging Volume Drop",
                        f"{latest} sessions · {mo} · {pct}% of 3-mo avg ({int(rolling_avg)} sessions)",
                        f"Imaging lead: {drop:.1f}% below average — review CT scheduling and equipment availability")]
    if pct < _IMAGING_WATCH_PCT:
        return [_notice("WATCH", "CT Imaging Volume Drop",
                        f"{latest} sessions · {mo} · {pct}% of 3-mo avg",
                        f"Imaging lead: {drop:.1f}% below average — monitor CT scheduling")]
    return []


# ── Management command ────────────────────────────────────────────────────

class Command(BaseCommand):
    help = "Compute KSH active alert notices and send the executive digest email."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run", action="store_true",
            help="Print active notices to stdout without sending email.",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        self.stdout.write("Loading data from Snowflake...")

        th_df  = _safe_load(q_theatre_trend)
        di_df  = _safe_load(q_dialysis_trend, "KISUMU_CLEAN")
        wa_df  = _safe_load(q_ward_admissions_monthly, "KISUMU_CLEAN")
        wl_df  = _safe_load(q_ward_los_monthly, "KISUMU_CLEAN")
        wd_df  = _safe_load(q_ward_discharge_monthly, "KISUMU_CLEAN")
        dw_df  = _safe_load(q_doctor_workload_monthly)
        lab_df = _safe_load(q_lab_monthly)
        btr_df = _safe_load(q_btr_bti_monthly)
        tat_df = _safe_load(q_admission_tat_monthly)
        rv_df  = _safe_load(q_revpab_private_monthly)
        cd_df  = _safe_load(q_cd12_monthly_rate)
        img_df = _safe_load(q_imaging_trend, "KISUMU_CLEAN")

        self.stdout.write("Running alert rules...")

        notices        = []
        clinical_notes = []

        notices += _rule3_theatre(th_df)
        notices += _rule4_dialysis(di_df)
        notices += _rules10_14_traffic(wa_df)
        notices += _rules15_19_los(wl_df)
        notices += _rules20_24_patient_request(wd_df)
        notices += _rule25_doctor_concentration(dw_df)
        notices += _rule26_doctor_burnout(dw_df)
        notices += _rules27_28_lab(lab_df)

        r29, fired_wards = _rule29_ward_idle(btr_df)
        notices += r29
        notices += _rule30_admission_tat(tat_df)
        notices += _rule31_bor_low(btr_df, fired_wards)
        notices += _rule32_private_revenue(rv_df)
        notices += _rule33_physician_workload(dw_df)

        r34, r34_clinical = _rule34_cd12(cd_df)
        notices        += r34
        clinical_notes += r34_clinical

        notices += _rule35_ct_imaging(img_df)

        n = len(notices)
        self.stdout.write(f"Active alerts: {n}")

        if dry_run:
            self.stdout.write("\n── DRY RUN — no email sent ──────────────")
            for notice in notices:
                self.stdout.write(f"  [{notice['level']}] {notice['title']}: {notice['metric']}")
            if clinical_notes:
                self.stdout.write("── Clinical notes ────────────────────────")
                for cn in clinical_notes:
                    self.stdout.write(f"  [CLINICAL] {cn['title']}: {cn['metric']}")
            return

        today = pd.Timestamp.today().strftime("%d %b %Y %H:%M")
        stats = f"KSH · {n} active alert(s) · {today}"

        ok, msg = send_digest(
            facility_name=_FACILITY,
            notices=notices,
            stats=stats,
            clinical_notes=clinical_notes or None,
        )

        if ok:
            self.stdout.write(self.style.SUCCESS(f"Digest sent — {n} notice(s)."))
            logger.info("compute_notices: digest sent, %d notice(s).", n)
        else:
            self.stderr.write(f"Digest send failed: {msg}")
            logger.error("compute_notices: send_digest failed — %s", msg)
