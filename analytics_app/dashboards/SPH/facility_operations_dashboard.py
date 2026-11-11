"""
St. Peter's Orthopedic Hospital — Operations Command Center (Home)
Sections: Priority → Hospital Flow → Domain Grid (2 rows) → Today's Watchlist
V2 operational data (Feb 2025 – present).
"""

import sys
import os
import json
import threading
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here if os.path.exists(os.path.join(_here, 'dashboard')) else os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(sys.path)
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="SPH · Home",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

from facility_operations.dashboard.theme import (
    apply_theme, render_sidebar, nav_url, COLORS, STATUS_BG, STATUS_BORDER, STATUS_LABEL, STATUS_EMOJI,
    notice_card, section_header, page_header, info_card,
)
from facility_operations.dashboard.queries import (
    q_waiting_rbi_summary, q_dropoff_kpis, q_dropoff_stage_responsibility,
    q_leakage_summary, q_leakage_by_procedure,
    q_cc_pipeline, q_cc_freshness, q_cc_pharm_dispensing, q_cc_pharm_fulfillment,
    q_cc_lab_completion,
    q_imaging_modality_tat, q_lab_chain_tat,
    q_diag_demand_monthly, q_conv_v2_monthly,
    preload_all,
)
from facility_operations.dashboard.notifier import write_current_notices

apply_theme()

if not st.session_state.get("_preload_started"):
    st.session_state["_preload_started"] = True
    threading.Thread(target=preload_all, daemon=True).start()

# ── Data load ──────────────────────────────────────────────────────────────────
def _load_df(fn):
    try:
        df = fn()
        if not df.empty:
            df.columns = df.columns.str.upper()
        return df
    except Exception:
        return pd.DataFrame()

def _load_row(fn):
    df = _load_df(fn)
    return df.iloc[0] if not df.empty else None

_rbi_df        = _load_df(q_waiting_rbi_summary)
_dropoff_row   = _load_row(q_dropoff_kpis)
_dropoff_stage = _load_df(q_dropoff_stage_responsibility)
_leakage_row   = _load_row(q_leakage_summary)
_leakage_proc  = _load_df(q_leakage_by_procedure)
_pipeline_row  = _load_row(q_cc_pipeline)
_pharm_disp    = _load_row(q_cc_pharm_dispensing)
_pharm_fulfill = _load_row(q_cc_pharm_fulfillment)
_lab_comp_row  = _load_row(q_cc_lab_completion)
_diag_mod_tat  = _load_df(q_imaging_modality_tat)
_lab_chain_row   = _load_row(q_lab_chain_tat)
_diag_demand_df  = _load_df(q_diag_demand_monthly)
_conv_monthly_df = _load_df(q_conv_v2_monthly)
_freshness       = None
try:
    r = q_cc_freshness()
    if not r.empty:
        r.columns = r.columns.str.upper()
        _freshness = pd.to_datetime(r.iloc[0]["V2_LATEST_DATE"])
except Exception:
    pass

_intel = None
try:
    _intel_path = os.path.join(
        _here,
        "facility_operations","ai_foundation", "latest_run.json",
    )
    with open(_intel_path, encoding="utf-8") as _f:
        _intel = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError):
    pass

# ── Utility ────────────────────────────────────────────────────────────────────
def _safe(v, default=None):
    try:
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default

def _delta_pct(current, baseline):
    c, b = _safe(current), _safe(baseline)
    if c is None or b is None or b == 0:
        return None
    return round((c - b) / b * 100, 1)

def _alert_score(rbi_label, pct_change, coverage_pct):
    sev  = {"Bottleneck": 3, "Elevated": 2, "Normal": 1}.get(str(rbi_label or "Normal"), 1)
    pct  = _safe(pct_change, 0.0)
    pers = 1.3 if pct > 10 else (1.1 if pct > 0 else 1.0)
    conf = _safe(coverage_pct, 0.0) / 100.0
    return round(sev * pers * conf, 2)

def _get_rbi_row(stage_name):
    if _rbi_df.empty:
        return None
    match = _rbi_df[_rbi_df["STAGE"].str.lower() == stage_name.lower()]
    return match.iloc[0] if not match.empty else None

# ── Prescriptive decisions per stage ──────────────────────────────────────────
_DECISIONS = {
    "Consult":  "Review department TAT breakdown and staffing. See Patient Waiting page for detail.",
    "Pharmacy": "Review dispensing queue and pending prescriptions. Escalate to Pharmacy supervisor.",
    "Lab":      "Escalate to Laboratory head — review specimen collection and processing queue.",
    "Imaging":  "Escalate to Radiology head — review scheduling queue and slot availability.",
}
_DEFAULT_DECISION = "Review the relevant department page for TAT breakdown. Escalate to department head if bottleneck persists."

# ── Alert Score / top incident ─────────────────────────────────────────────────
_top_incident = None
_incident_sev = None

if not _rbi_df.empty:
    _scored = _rbi_df.copy()
    _scored["_s"] = _scored.apply(
        lambda r: _alert_score(r.get("RBI_LABEL"), r.get("PCT_CHANGE_28D"), r.get("COVERAGE_PCT")), axis=1
    )
    _scored = _scored.sort_values("_s", ascending=False).reset_index(drop=True)
    _top = _scored.iloc[0]
    _top_lbl = str(_top.get("RBI_LABEL") or "")
    if _top_lbl == "Bottleneck" or float(_top["_s"]) >= 2.0:
        _incident_sev = "CRITICAL"
        _top_incident = _top
    elif _top_lbl == "Elevated" or float(_top["_s"]) >= 1.2:
        _incident_sev = "WATCH"
        _top_incident = _top

# ── Domain insights — Row 1 (validated scoring models) ────────────────────────

# Patient Waiting
_waiting_status  = "GREEN"
_waiting_urgency = ("🟢 Monitor only", COLORS["success"])
_waiting_story   = "All stages flowing within normal range."
_waiting_metrics = []

if not _rbi_df.empty:
    _w = _rbi_df.sort_values("RBI_SCORE", ascending=False).reset_index(drop=True).iloc[0]
    _wl = str(_w.get("RBI_LABEL") or "Normal")
    _ws = str(_w.get("STAGE") or "")
    _wp = _safe(_w.get("CURRENT_P50_MINS"))
    _wd = _safe(_w.get("PCT_CHANGE_28D"))
    _wc = _safe(_w.get("COVERAGE_PCT"))
    _wn = _safe(_w.get("COVERAGE_N"))
    _p50s  = f"Median {int(_wp)} min" if _wp else "Median —"
    _delts = (f"{'+' if _wd >= 0 else ''}{_wd:.0f}% vs 28-day avg" if _wd is not None else "")
    _covs  = f"{_wc:.0f}% coverage" if _wc else ""
    _ns    = f"n={int(_wn):,}" if _wn else ""
    _waiting_metrics = []
    if _wl == "Bottleneck":
        _waiting_status  = "RED"
        _waiting_urgency = ("🔴 Action required today", COLORS["danger"])
        _waiting_story   = f"{_ws} is constraining patient flow."
    elif _wl == "Elevated":
        _waiting_status  = "AMBER"
        _waiting_urgency = ("🟡 Monitor closely", COLORS["warning"])
        _waiting_story   = f"{_ws} TAT is elevated and rising."
    else:
        _waiting_story   = f"Top stage: {_ws} — within normal range."

# Patient Drop-off
_dropoff_status  = "GREEN"
_dropoff_urgency = ("🟢 Monitor only", COLORS["success"])
_dropoff_story   = "Patient pathway completion within expected range."
_dropoff_metrics = []
_primary_exit    = ""
_exit_pct        = 0.0

if _dropoff_row is not None:
    _ac = _safe(_dropoff_row.get("ARRIVAL_TO_CONSULT_PCT"), 0)
    _oi = _safe(_dropoff_row.get("OPD_INCOMPLETE_PCT"), 0)
    _cv = _safe(_dropoff_row.get("OPD_ADMISSION_CONVERSION_PCT"), 0)
    if _ac < 40:
        _dropoff_status  = "RED"
        _dropoff_urgency = ("🔴 Investigation required", COLORS["danger"])
    elif _ac < 50:
        _dropoff_status  = "AMBER"
        _dropoff_urgency = ("🟡 Monitor closely", COLORS["warning"])

    if not _dropoff_stage.empty:
        _top_s        = _dropoff_stage.iloc[0]
        _stage_display = {
            "post-registration": "Post-Registration",
            "post-triage":       "Post-Triage",
            "consult":           "Consult",
            "ancillary":         "Ancillary",
            "admission":         "Admission",
            "theatre":           "Theatre",
        }
        _stage_raw    = str(_top_s.get("DROP_OFF_STAGE") or "").lower()
        _primary_exit = _stage_display.get(_stage_raw, _stage_raw.replace("-", " ").title())
        _exit_pct     = _safe(_top_s.get("DROP_OFF_PCT"), 0)

    # Override to AMBER if primary exit stage captures ≥30% of incomplete visits —
    # that's a material operational signal even if consult rate is just above threshold.
    if _dropoff_status == "GREEN" and _exit_pct >= 30:
        _dropoff_status  = "AMBER"
        _dropoff_urgency = ("🟡 Monitor closely", COLORS["warning"])

    _dropoff_story = (
        f"OPD ghost rate: {_oi:.1f}% — {_primary_exit} is the primary exit stage."
        if _primary_exit else f"OPD incomplete: {_oi:.1f}%"
    )
    _dropoff_metrics = [
    ]

# Diagnostics — derived from confirmed investigation findings
_diag_status   = "GREEN"
_diag_urgency  = ("🟢 Monitor only", COLORS["success"])
_diag_story    = "No active diagnostic bottlenecks detected."
_diag_findings = []

if not _diag_mod_tat.empty:
    _cardiac = _diag_mod_tat[_diag_mod_tat["MODALITY_GROUP"].str.upper() == "CARDIAC"]
    if not _cardiac.empty:
        _c_p50 = _safe(_cardiac.iloc[0].get("P50_MINS"))
        _c_pct = _safe(_cardiac.iloc[0].get("PCT_WITHIN_60"))
        if _c_p50 is not None and _c_p50 > 60:
            _diag_status  = "RED" if _c_p50 > 90 else "AMBER"
            _diag_urgency = (
                ("🔴 Action required", COLORS["danger"]) if _c_p50 > 90
                else ("🟡 Investigation required", COLORS["warning"])
            )
            _pct_s = f" · {_c_pct:.0f}% within 1 hr" if _c_pct is not None else ""
            _diag_story = f"Cardiac imaging slow — {int(_c_p50)} min median{_pct_s}."
            _diag_findings.append(f"Cardiac: investigate scheduling and booking.")

if _lab_chain_row is not None:
    _lc_p90 = _safe(_lab_chain_row.get("P90_ORDER_TO_COLLECT"))
    _lc_p50 = _safe(_lab_chain_row.get("P50_ORDER_TO_COLLECT"))
    if _lc_p90 is not None and _lc_p90 > 60:
        if _diag_status == "GREEN":
            _diag_status  = "AMBER"
            _diag_urgency = ("🟡 Investigation required", COLORS["warning"])
            _diag_story   = f"Lab delays in collection queue — slowest 10% wait over {int(_lc_p90)} min."
        else:
            _diag_story += f" Lab delays in collection queue — slowest 10% wait over {int(_lc_p90)} min."
        _diag_findings.append(
            f"Lab: slowest 10% collection waits {int(_lc_p90)} min · median processing {int(_lc_p50)} min."
        )

# Revenue Leakage
_leakage_status  = "GREEN"
_leakage_urgency = ("🟢 Monitor only", COLORS["success"])
_leakage_story   = "Collection rate within acceptable range."
_leakage_metrics = []
_top_proc_name   = ""
_top_proc_kes    = 0.0
_top_proc_lr     = 0.0
_top_proc_shr    = 0.0

if _leakage_row is not None:
    _cr  = _safe(_leakage_row.get("COLLECTION_RATE_PCT"), 0)
    _kes = _safe(_leakage_row.get("TOTAL_UNCOLLECTED_KES"), 0)
    _kes_s = f"KES {_kes/1e6:.1f}M" if _kes >= 1e6 else f"KES {_kes:,.0f}"
    if _cr < 75:
        _leakage_status  = "RED"
        _leakage_urgency = ("📋 Finance review this week", COLORS["warning"])
    elif _cr < 85:
        _leakage_status  = "AMBER"
        _leakage_urgency = ("📋 Finance review this week", COLORS["warning"])

    if not _leakage_proc.empty:
        _tp = _leakage_proc.iloc[0]
        _top_proc_name = str(_tp.get("REQUEST_NAME") or "")
        _top_proc_kes  = _safe(_tp.get("UNCOLLECTED_KES"), 0)
        _top_proc_lr   = _safe(_tp.get("LEAKAGE_PCT"), 0)
        _top_proc_shr  = _safe(_tp.get("SHARE_OF_TOTAL_PCT"), 0)

    if _top_proc_name:
        _tpk_s = f"KES {_top_proc_kes/1e3:.0f}K" if _top_proc_kes >= 1000 else f"KES {_top_proc_kes:.0f}"
        _leakage_story = f"{_top_proc_name} drives the most uncollected revenue."
        _leakage_metrics = []
    else:
        _leakage_story = f"Collection rate: {_cr:.1f}%"
        _leakage_metrics = []

# ── Domain insights — Row 2 (TAT only, no badge) ──────────────────────────────
def _tat_narrative(stage_name, p50, min_coverage_pct=10):
    """Descriptive text for pulse cards — no status claim, no badge.
    Suppresses pct_change when coverage is below min_coverage_pct (noise dominates)."""
    if p50 is None:
        return "TAT data unavailable"
    row = _get_rbi_row(stage_name)
    if row is None:
        return "Within recent operating range"
    cov = _safe(row.get("COVERAGE_PCT"), 0)
    if cov < min_coverage_pct:
        return f"Coverage {cov:.0f}% — trend suppressed (insufficient sample)"
    pct = _safe(row.get("PCT_CHANGE_28D"))
    if pct is None or abs(pct) <= 5:
        return "Within recent operating range"
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.0f}% vs prior 28 days"

_lab_row    = _get_rbi_row("Lab")
_img_row    = _get_rbi_row("Imaging")
# Lab completion rate from rpt_ortho_lab (has_result = status='4' for V2 — Issue 89 fix).
# TAT suppressed — result_stamp broken for V2 since Feb 2026; completion % is the correct metric.
_lab_comp_pct = _safe(_lab_comp_row.get("COMPLETION_PCT")) if _lab_comp_row is not None else None
_img_p50      = _safe(_img_row.get("CURRENT_P50_MINS"))    if _img_row is not None else None
# Pharmacy: use real dispensing P50 from rpt_ortho_pharmacy V2 (Issue 91 — mart column = queue arrival)
_pharm_p50  = _safe(_pharm_disp.get("P50_DISPENSING_MINS")) if _pharm_disp is not None else None
_pharm_cov  = _safe(_pharm_disp.get("COVERAGE_PCT"))        if _pharm_disp is not None else None

# ── Operational Risks / Watchlist ──────────────────────────────────────────────
_risks = []

# Immediate — RBI bottleneck / elevated
if not _rbi_df.empty:
    for _, row in _rbi_df.iterrows():
        lbl = str(row.get("RBI_LABEL") or "Normal")
        if lbl in ("Bottleneck", "Elevated"):
            sname = str(row.get("STAGE") or "")
            p50   = _safe(row.get("CURRENT_P50_MINS"))
            pct   = _safe(row.get("PCT_CHANGE_28D"))
            owner = str(row.get("OPERATIONAL_OWNER") or "—")
            ev    = " · ".join(x for x in [
                f"Median {int(p50)} min" if p50 else "",
                (f"{'+' if pct >= 0 else ''}{pct:.0f}% TAT vs 28d" if pct is not None else ""),
            ] if x)
            _risks.append({
                "risk":  f"{sname} {lbl.lower()}",
                "ev":    ev,
                "owner": owner,
                "tier":  "Immediate",
                "sev":   "RED" if lbl == "Bottleneck" else "AMBER",
            })

# This week — top leakage procedure
if _top_proc_name:
    _tpk_s3 = f"KES {_top_proc_kes/1e3:.0f}K" if _top_proc_kes >= 1000 else f"KES {_top_proc_kes:.0f}"
    _risks.append({
        "risk":  f"{_top_proc_name} billing leakage",
        "ev":    f"{_tpk_s3} uncollected · {_top_proc_lr:.0f}% leakage · {_top_proc_shr:.0f}% of total",
        "owner": "Finance",
        "tier":  "This week",
        "sev":   "AMBER",
    })

# Structural — Cardiac imaging delay (threshold: P50 > 60 min)
if not _diag_mod_tat.empty:
    _cardiac = _diag_mod_tat[
        _diag_mod_tat["MODALITY_GROUP"].str.upper() == "CARDIAC"
    ]
    if not _cardiac.empty:
        _c_p50  = _safe(_cardiac.iloc[0].get("P50_MINS"))
        _c_pct  = _safe(_cardiac.iloc[0].get("PCT_WITHIN_60"))
        if _c_p50 is not None and _c_p50 > 60:
            _c_sev = "RED" if _c_p50 > 90 else "AMBER"
            _c_ev  = (
                f"Median {int(_c_p50)} min · "
                + (f"{_c_pct:.0f}% within 1 hr" if _c_pct is not None else "")
            ).rstrip(" ·")
            _risks.append({
                "risk":  "Cardiac imaging delay",
                "ev":    _c_ev,
                "owner": "Radiology",
                "tier":  "Structural",
                "sev":   _c_sev,
            })

# Structural — Lab specimen collection queue (threshold: P90 order→collection > 60 min)
if _lab_chain_row is not None:
    _lc_p50 = _safe(_lab_chain_row.get("P50_ORDER_TO_COLLECT"))
    _lc_p90 = _safe(_lab_chain_row.get("P90_ORDER_TO_COLLECT"))
    if _lc_p90 is not None and _lc_p90 > 60:
        _lc_ev = (
            f"Slowest 10% wait {int(_lc_p90)} min order→collection"
            + (f" · median {int(_lc_p50)} min" if _lc_p50 is not None else "")
            + " · collection queue exceeds processing time"
        )
        _risks.append({
            "risk":  "Lab specimen collection queue",
            "ev":    _lc_ev,
            "owner": "Laboratory",
            "tier":  "Structural",
            "sev":   "AMBER",
        })

# Structural — primary drop-off stage (only if ≥10%)
if _primary_exit and _exit_pct >= 10:
    _exit_owner_map = {
        "post-registration": "Triage / Front Desk",
        "post-triage":       "Nursing",
        "consult":           "Clinical",
        "ancillary":         "Lab / Radiology",
        "admission":         "Admissions",
        "theatre":           "Theatre",
        "discharged":        "Ward",
    }
    _exit_owner = next(
        (v for k, v in _exit_owner_map.items() if k in _primary_exit.lower()), "Operations"
    )
    _risks.append({
        "risk":  f"{_primary_exit} exits",
        "ev":    f"{_exit_pct:.0f}% of incomplete OPD visits exit here",
        "owner": _exit_owner,
        "tier":  "Structural",
        "sev":   "MUTED",
    })

# ── Active Notices — digest feed ───────────────────────────────────────────────
# Maps domain statuses (computed above) to notices for the sidebar + email digest.
# Variable safety: RED/AMBER statuses are only set inside non-empty data blocks,
# so all referenced variables (_ws, _p50s, _kes_s, etc.) are defined when used.
_notices = []

if _waiting_status in ("RED", "AMBER"):
    _w_action = _DECISIONS.get(_ws, _DEFAULT_DECISION)
    _notices.append({
        "level":  "CRITICAL" if _waiting_status == "RED" else "WATCH",
        "title":  f"Patient Waiting — {_ws}",
        "metric": _p50s,
        "action": _w_action,
    })

if _dropoff_status in ("RED", "AMBER"):
    _d_action = (
        f"Investigate {_primary_exit} exit stage — review staffing and workflow."
        if _primary_exit else "Review patient pathway completion."
    )
    _notices.append({
        "level":  "CRITICAL" if _dropoff_status == "RED" else "WATCH",
        "title":  "Patient Drop-off" + (" — Critical" if _dropoff_status == "RED" else ""),
        "metric": _dropoff_story,
        "action": _d_action,
    })

if _leakage_status in ("RED", "AMBER"):
    _l_action = (
        f"Finance review: {_top_proc_name} — {_top_proc_lr:.0f}% leakage rate."
        if _top_proc_name else "Finance review required."
    )
    _notices.append({
        "level":  "CRITICAL" if _leakage_status == "RED" else "WATCH",
        "title":  "Revenue Leakage",
        "metric": _leakage_story,
        "action": _l_action,
    })

if _diag_status in ("RED", "AMBER"):
    _dg_action = (
        " ".join(_diag_findings) if _diag_findings
        else "Escalate to department head."
    )
    _notices.append({
        "level":  "CRITICAL" if _diag_status == "RED" else "WATCH",
        "title":  "Diagnostics — Delay Detected",
        "metric": _diag_story,
        "action": _dg_action,
    })

st.session_state["active_notices"] = _notices
write_current_notices("SPH", _notices)

# ── HTML renderers ─────────────────────────────────────────────────────────────

def _flow_node_html(title, metric_value, metric_label,
                    badge_text=None, badge_color=None, is_pending=False):
    if badge_text and badge_color:
        badge_html = (
            f'<div style="margin-top:10px">'
            f'<span style="background:{badge_color}18;border:1px solid {badge_color}50;'
            f'color:{badge_color};font-size:10px;font-weight:700;padding:3px 10px;'
            f'border-radius:12px">{badge_text}</span></div>'
        )
        accent = f"border-left:4px solid {badge_color}"
    elif is_pending:
        badge_html = (
            f'<div style="margin-top:10px">'
            f'<span style="background:#F4F8FC;border:1px solid #C5D8EC;color:#9BAEC8;'
            f'font-size:10px;font-weight:600;padding:3px 10px;border-radius:12px">'
            f'Data pending</span></div>'
        )
        accent = "border-left:4px solid #D6E4F0"
    else:
        badge_html = ""
        accent = "border-left:4px solid #D6E4F0"

    return (
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;{accent};'
        f'border-radius:8px;padding:14px 18px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">{title}</div>'
        f'<div style="font-size:24px;font-weight:800;color:#003467;line-height:1.1">{metric_value}</div>'
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:2px">{metric_label}</div>'
        f'{badge_html}'
        f'</div>'
    )

_arrow_html = (
    '<div style="padding:6px 0 6px 20px;color:#C5D8EC;font-size:18px;line-height:1">↓</div>'
)

_h_arrow_html = (
    '<div style="padding-top:28px;text-align:center;color:#C5D8EC;'
    'font-size:28px;font-weight:300;line-height:1">›</div>'
)


def _flow_card_html(title, metric_value, metric_label,
                    badge_text=None, badge_color=None, is_pending=False):
    """Horizontal grid flow card — center-aligned, top-border accent."""
    if badge_text and badge_color:
        badge_html = (
            f'<div style="margin-top:10px">'
            f'<span style="background:{badge_color}18;border:1px solid {badge_color}50;'
            f'color:{badge_color};font-size:10px;font-weight:700;padding:3px 10px;'
            f'border-radius:12px">{badge_text}</span></div>'
        )
        top_border = f"border-top:4px solid {badge_color}"
    elif is_pending:
        badge_html = (
            '<div style="margin-top:10px">'
            '<span style="background:#F4F8FC;border:1px solid #C5D8EC;color:#9BAEC8;'
            'font-size:10px;font-weight:600;padding:3px 10px;border-radius:12px">'
            'Data pending</span></div>'
        )
        top_border = "border-top:4px solid #D6E4F0"
    else:
        badge_html = ""
        top_border = "border-top:4px solid #D6E4F0"

    return (
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;{top_border};'
        f'border-radius:10px;padding:24px 14px 20px;text-align:center;min-height:180px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">{title}</div>'
        f'<div style="font-size:30px;font-weight:800;color:#003467;line-height:1.1">{metric_value}</div>'
        f'<div style="font-size:10px;color:#9BAEC8;margin-top:6px">{metric_label}</div>'
        f'{badge_html}'
        f'</div>'
    )


def _domain_card(domain, icon, status, urgency_label, urgency_color, story, metrics, href, href_label):
    dc   = STATUS_BORDER[status]
    dbg  = STATUS_BG[status]
    dlbl = STATUS_LABEL[status]
    de   = STATUS_EMOJI[status]
    mhtml = "".join(
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">{m}</div>'
        for m in metrics if m
    )
    return (
        f'<div style="background:{dbg};border:1px solid {dc}40;border-top:4px solid {dc};'
        f'border-radius:10px;padding:20px 18px 16px;min-height:220px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">'
        f'<div>'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">'
        f'<i class="{icon}" style="margin-right:5px"></i>{domain}</div>'
        f'<span style="background:{dc};color:#fff;font-size:10px;font-weight:800;'
        f'letter-spacing:1.2px;padding:3px 9px;border-radius:4px">{dlbl}</span>'
        f'</div>'
        f'<span style="font-size:22px;line-height:1">{de}</span>'
        f'</div>'
        f'<div style="margin:10px 0 4px">'
        f'<span style="background:{urgency_color}15;border:1px solid {urgency_color}40;'
        f'color:{urgency_color};font-size:10px;font-weight:700;padding:3px 10px;border-radius:12px">'
        f'{urgency_label}</span>'
        f'</div>'
        f'<div style="font-size:14px;font-weight:700;color:#003467;line-height:1.5;margin:10px 0 6px">'
        f'{story}</div>'
        f'{mhtml}'
        f'<div style="margin-top:14px;border-top:1px solid {dc}30;padding-top:10px">'
        f'<a href="{href}" target="_self" style="font-size:12px;color:#0072CE;font-weight:700;'
        f'text-decoration:none">→ {href_label}</a>'
        f'</div>'
        f'</div>'
    )


def _pulse_card(domain, icon, tat_value, tat_label, narrative, href, href_label, unit="min"):
    """Row 2 — service metric only. No RAG badge: no validated scoring model for this domain."""
    if tat_value is not None:
        val_s = f"{int(tat_value)}{' ' + unit if unit else ''}"
    else:
        val_s = "—"
    return (
        f'<div style="background:#FAFCFF;border:1px solid #D6E4F0;border-top:4px solid #D6E4F0;'
        f'border-radius:10px;padding:20px 18px 16px;min-height:200px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">'
        f'<i class="{icon}" style="margin-right:5px"></i>{domain}</div>'
        f'<div style="font-size:32px;font-weight:800;color:#003467;line-height:1">{val_s}</div>'
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:2px">{tat_label}</div>'
        f'<div style="font-size:13px;color:#6B8CAE;margin-top:10px;line-height:1.4">{narrative}</div>'
        f'<div style="margin-top:14px;border-top:1px solid #D6E4F080;padding-top:10px">'
        f'<a href="{href}" target="_self" style="font-size:12px;color:#0072CE;font-weight:700;'
        f'text-decoration:none">→ {href_label}</a>'
        f'</div>'
        f'</div>'
    )


def _signal_card(domain, icon, value, value_label, narrative, href, href_label, decimals=1):
    """Operational signal card — shows float value, no RAG badge, informational only."""
    val_s = f"{value:.{decimals}f}" if value is not None else "—"
    return (
        f'<div style="background:#FAFCFF;border:1px solid #D6E4F0;border-top:4px solid #D6E4F0;'
        f'border-radius:10px;padding:20px 18px 16px;min-height:200px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">'
        f'<i class="{icon}" style="margin-right:5px"></i>{domain}</div>'
        f'<div style="font-size:32px;font-weight:800;color:#003467;line-height:1">{val_s}</div>'
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:2px">{value_label}</div>'
        f'<div style="font-size:13px;color:#6B8CAE;margin-top:10px;line-height:1.4">{narrative}</div>'
        f'<div style="margin-top:14px;border-top:1px solid #D6E4F080;padding-top:10px">'
        f'<a href="{href}" target="_self" style="font-size:12px;color:#0072CE;font-weight:700;'
        f'text-decoration:none">→ {href_label}</a>'
        f'</div>'
        f'</div>'
    )


_STATUS_MAP = {"Immediate": "Open", "This week": "Open", "Structural": "Monitoring"}

_INTEL_SEV_COLOR = {
    "Critical": COLORS["danger"],
    "Warning":  COLORS["warning"],
    "Info":     COLORS["muted"],
}
_INTEL_SEV_EMOJI = {"Critical": "🔴", "Warning": "⚠", "Info": "ℹ"}


def _render_intelligence(intel):
    """Render the Operational Intelligence section — compact ranked feed.

    Reads only from the persistence JSON dict — no ai_foundation imports.
    Domain-agnostic: maps OperationalBriefing semantic fields to display slots.
    Each problem renders as a compact summary row; full briefing + evidence are in an expander.
    """
    from datetime import datetime as _dt

    status = intel.get("status", "pipeline_failed") if intel else "pipeline_failed"
    run_ts_raw = (intel.get("run_ts") or "") if intel else ""

    try:
        _parsed = _dt.fromisoformat(run_ts_raw.replace("+00:00", "").replace("Z", ""))
        analysed_str = _parsed.strftime("%d %b %H:%M")
    except Exception:
        analysed_str = run_ts_raw[:16] if run_ts_raw else ""

    ts_suffix = (
        f'&nbsp;<span style="font-size:10px;font-weight:400;color:#9BAEC8">'
        f'· Analysed: {analysed_str}</span>'
        if analysed_str else ""
    )
    section_header(f"Operational Intelligence{ts_suffix}", margin_top=36)

    if intel is None or status == "no_trigger":
        st.markdown(
            '<div style="background:#F0FBF8;border:1px solid #0BB99F40;border-radius:8px;'
            'padding:14px 18px;font-size:13px;color:#003467">'
            '<b style="color:#0BB99F">No active intelligence findings</b> — '
            'no configured operational triggers fired in the latest analysis.</div>',
            unsafe_allow_html=True,
        )
        return

    if status == "pipeline_failed":
        _last = f"Last attempted: {analysed_str}." if analysed_str else ""
        st.markdown(
            f'<div style="background:#FFF5F5;border:1px solid #E11D4840;border-radius:8px;'
            f'padding:14px 18px;font-size:13px;color:#003467">'
            f'<b style="color:#E11D48">Operational intelligence unavailable</b> — '
            f'the latest analysis could not be completed. {_last}</div>',
            unsafe_allow_html=True,
        )
        return

    problems = intel.get("problems", [])
    if not problems:
        return

    for p in problems:
        severity         = p.get("severity") or "Warning"
        sev_color        = _INTEL_SEV_COLOR.get(severity, COLORS["warning"])
        sev_emoji        = _INTEL_SEV_EMOJI.get(severity, "⚠")
        metric_id        = p.get("metric_id", "—")
        priority_score   = float(p.get("priority_score") or 0)
        evidence_payload = p.get("evidence_payload") or ""
        briefing         = p.get("briefing")
        sig              = p.get("signature") or {}
        synthesis_failed = briefing is None

        # Compact one-line summary derived from deterministic signature
        attribution = sig.get("attribution", "—").title()
        _tp = sig.get("temporal_pattern", "")
        _tp_parts = _tp.split("/") if "/" in _tp else [_tp, ""]
        temporal = f"{_tp_parts[0]} at {_tp_parts[1]}" if _tp_parts[1] else _tp_parts[0]
        mechanism_tag = sig.get("mechanism", "—").title()
        summary = f"{mechanism_tag} demand spike — peak {temporal}"

        # Compact row
        st.markdown(
            f'<div style="background:#FAFCFF;border:1px solid {sev_color}40;'
            f'border-left:4px solid {sev_color};border-radius:10px;'
            f'padding:14px 20px;margin-bottom:4px">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
            f'<div style="flex:1;padding-right:16px">'
            f'<div style="margin-bottom:5px">'
            f'<span style="font-size:9px;font-weight:700;color:{sev_color};'
            f'text-transform:uppercase;letter-spacing:2px">{sev_emoji} {severity}</span>'
            f'&nbsp;&nbsp;'
            f'<span style="font-size:13px;font-weight:700;color:#003467">{attribution}</span>'
            f'<span style="font-size:11px;color:#9BAEC8;font-weight:400"> · {metric_id}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:#6B8CAE;line-height:1.4">{summary}</div>'
            f'</div>'
            f'<div style="text-align:right;flex-shrink:0">'
            f'<div style="font-size:9px;color:#9BAEC8;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:2px">Priority</div>'
            f'<div style="font-size:20px;font-weight:800;color:{sev_color};line-height:1">'
            f'{priority_score:.2f}</div>'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Full briefing + evidence"):
            if synthesis_failed:
                st.markdown(
                    '<div style="font-size:13px;color:#6B8CAE;font-style:italic;margin-bottom:12px">'
                    'Intelligence narrative unavailable — verified evidence below.</div>',
                    unsafe_allow_html=True,
                )
            else:
                what       = briefing.get("what", "—")
                where_txt  = briefing.get("where", "—")
                when_txt   = briefing.get("when", "—")
                mech_txt   = briefing.get("mechanism", "—")
                downstream = briefing.get("downstream", "—")
                unknowns   = briefing.get("unknowns", "—")
                action     = briefing.get("action", "—")

                st.markdown(
                    f'<div style="padding:4px 0 16px">'

                    f'<div style="margin-bottom:12px">'
                    f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Signal</div>'
                    f'<div style="font-size:13px;color:#003467;line-height:1.6">{what}</div>'
                    f'</div>'

                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px">'
                    f'<div><div style="font-size:9px;font-weight:700;color:#9BAEC8;'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Where</div>'
                    f'<div style="font-size:13px;color:#003467;line-height:1.6">{where_txt}</div></div>'
                    f'<div><div style="font-size:9px;font-weight:700;color:#9BAEC8;'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">When</div>'
                    f'<div style="font-size:13px;color:#003467;line-height:1.6">{when_txt}</div></div>'
                    f'</div>'

                    f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px">'
                    f'<div><div style="font-size:9px;font-weight:700;color:#9BAEC8;'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Mechanism</div>'
                    f'<div style="font-size:13px;color:#003467;line-height:1.6">{mech_txt}</div></div>'
                    f'<div><div style="font-size:9px;font-weight:700;color:#9BAEC8;'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">Downstream impact</div>'
                    f'<div style="font-size:13px;color:#003467;line-height:1.6">{downstream}</div></div>'
                    f'</div>'

                    f'<div style="margin-bottom:12px">'
                    f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px">What cannot be determined</div>'
                    f'<div style="font-size:12px;color:#6B8CAE;font-style:italic;line-height:1.6">{unknowns}</div>'
                    f'</div>'

                    f'<div style="background:{sev_color}0C;border:1px solid {sev_color}30;'
                    f'border-radius:8px;padding:14px;margin-bottom:16px">'
                    f'<div style="font-size:9px;font-weight:700;color:{sev_color};'
                    f'text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px">Evidence-backed action</div>'
                    f'<div style="font-size:14px;font-weight:700;color:#003467;line-height:1.5">{action}</div>'
                    f'</div>'

                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
                'letter-spacing:1.5px;margin:0 0 6px">Evidence</div>',
                unsafe_allow_html=True,
            )
            st.code(evidence_payload or "No evidence payload.", language=None)


def _watchlist_table(risks):
    if not risks:
        return (
            '<div style="background:#F0FBF8;border:1px solid #0BB99F40;border-radius:8px;'
            'padding:14px 18px;font-size:12px;color:#003467">'
            '<b style="color:#0BB99F">No risks above threshold</b> — all monitored indicators within normal range.</div>'
        )
    rows = ""
    for i, r in enumerate(risks, 1):
        sev_col = {"RED": COLORS["danger"], "AMBER": COLORS["warning"], "MUTED": COLORS["muted"]}.get(
            r["sev"], COLORS["muted"]
        )
        status = _STATUS_MAP.get(r["tier"], "Open")
        st_bg  = "#FFF5F5" if status == "Open" else "#F4F8FC"
        st_col = COLORS["danger"] if status == "Open" else COLORS["muted"]
        rows += (
            f'<tr style="border-bottom:1px solid #EBF3FB;vertical-align:top">'
            f'<td style="padding:12px 14px;white-space:nowrap">'
            f'<span style="display:inline-flex;align-items:center;gap:6px">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{sev_col};flex-shrink:0"></span>'
            f'<span style="font-size:13px;font-weight:700;color:#003467">{i}</span>'
            f'</span></td>'
            f'<td style="padding:12px 14px">'
            f'<span style="font-size:13px;font-weight:600;color:#003467">{r["risk"]}</span><br>'
            f'<span style="font-size:11px;color:#9BAEC8">{r["ev"]}</span>'
            f'</td>'
            f'<td style="padding:12px 14px;font-size:12px;font-weight:600;color:#003467;white-space:nowrap">{r["owner"]}</td>'
            f'<td style="padding:12px 14px">'
            f'<span style="background:{st_bg};border:1px solid {st_col}40;color:{st_col};'
            f'font-size:10px;font-weight:700;padding:3px 9px;border-radius:12px">{status}</span>'
            f'</td>'
            f'</tr>'
        )
    return (
        f'<div style="background:#fff;border:1px solid #D6E4F0;border-radius:8px;overflow:hidden">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<thead>'
        f'<tr style="background:#F4F8FC;border-bottom:2px solid #D6E4F0">'
        f'<th style="padding:10px 14px;font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;text-align:left">Priority</th>'
        f'<th style="padding:10px 14px;font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;text-align:left">Issue</th>'
        f'<th style="padding:10px 14px;font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;text-align:left">Owner</th>'
        f'<th style="padding:10px 14px;font-size:10px;font-weight:700;color:#6B8CAE;'
        f'text-transform:uppercase;letter-spacing:1.5px;text-align:left">Status</th>'
        f'</tr>'
        f'</thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# RENDER
# ══════════════════════════════════════════════════════════════════════════════
render_sidebar("overview", show_notify=True)

_fresh_str = _freshness.strftime("%d %b %Y") if _freshness is not None else "—"
# ── Section 2: Hospital Flow ──────────────────────────────────────────────────
section_header("Hospital Flow · Today's Status", margin_top=8)
st.markdown(
    f'<div style="font-size:12px;color:#6B8CAE;margin-bottom:16px">'
    f'Each stage shows the strongest validated metric available. V2 data to {_fresh_str}.</div>',
    unsafe_allow_html=True,
)

# Derive flow node values
_cons_rbi_row  = _get_rbi_row("Consult")
_cons_lbl      = str(_cons_rbi_row.get("RBI_LABEL") or "Normal") if _cons_rbi_row is not None else "Normal"
_cons_p50      = _safe(_cons_rbi_row.get("CURRENT_P50_MINS")) if _cons_rbi_row is not None else None
_cons_p50s     = f"{int(_cons_p50)} min" if _cons_p50 else "—"
_cons_badge_col = (
    COLORS["danger"]  if _cons_lbl == "Bottleneck" else
    COLORS["warning"] if _cons_lbl == "Elevated"   else
    COLORS["success"]
)

_arr_vol  = _safe(_pipeline_row.get("ARR_7D")) if _pipeline_row is not None else None
_arr_s    = f"{int(_arr_vol)}/day" if _arr_vol else "—"

_lab_comp_s = f"{_lab_comp_pct:.0f}%" if _lab_comp_pct is not None else "—"
_img_tat_s  = f"{int(_img_p50)} min"  if _img_p50 is not None else "—"
# Lab TAT — total chain (order → result), V2
_lab_tat_total = None
if _lab_chain_row is not None:
    _lc_p50c = _safe(_lab_chain_row.get("P50_ORDER_TO_COLLECT"))
    _lc_p50r = _safe(_lab_chain_row.get("P50_COLLECT_TO_RESULT"))
    if _lc_p50c is not None and _lc_p50r is not None:
        _lab_tat_total = int(_lc_p50c + _lc_p50r)
_lab_tat_s = f"{_lab_tat_total} min" if _lab_tat_total is not None else _lab_comp_s

# ── Operational signals ───────────────────────────────────────────────────────
_lab_per100_cur   = None
_img_per100_cur   = None
_conv_rate_cur    = None
_lab_per100_trend = ""
_img_per100_trend = ""
_conv_rate_trend  = ""
_lab_per100_lbl   = "—"
_img_per100_lbl   = "—"
_conv_rate_lbl    = "—"

if not _diag_demand_df.empty:
    _dd = _diag_demand_df.copy()
    _dd["MONTH"] = pd.to_datetime(_dd["MONTH"])
    _dd = _dd[_dd["MONTH"] >= pd.Timestamp("2025-02-01")]  # V2 only
    _dd = _dd.sort_values("MONTH").reset_index(drop=True)
    if len(_dd) >= 2 and "OPD_VISITS" in _dd.columns:
        if _safe(_dd.iloc[-1]["OPD_VISITS"], 0) < _safe(_dd.iloc[-2]["OPD_VISITS"], 0) * 0.5:
            _dd = _dd.iloc[:-1].reset_index(drop=True)
    if not _dd.empty:
        _lab_per100_cur = _safe(_dd.iloc[-1]["LAB_PER_100"])
        _img_per100_cur = _safe(_dd.iloc[-1]["IMAGING_PER_100"])
        _lab_per100_lbl = _dd.iloc[-1]["MONTH"].strftime("%b %Y")
        _img_per100_lbl = _lab_per100_lbl
        if len(_dd) >= 2:
            _lab_old = _safe(_dd.iloc[-2]["LAB_PER_100"])
            _img_old = _safe(_dd.iloc[-2]["IMAGING_PER_100"])
            _prior_lbl = _dd.iloc[-2]["MONTH"].strftime("%b")
            if _lab_per100_cur is not None and _lab_old is not None:
                _ld = _lab_per100_cur - _lab_old
                _la = "↑" if _ld > 0 else ("↓" if _ld < 0 else "→")
                _lab_per100_trend = f"{_la} {abs(_ld):.1f} vs {_prior_lbl}"
            if _img_per100_cur is not None and _img_old is not None:
                _id = _img_per100_cur - _img_old
                _ia = "↑" if _id > 0 else ("↓" if _id < 0 else "→")
                _img_per100_trend = f"{_ia} {abs(_id):.1f} vs {_prior_lbl}"

if not _conv_monthly_df.empty:
    _cm = _conv_monthly_df.copy()
    _cm["MONTH"] = pd.to_datetime(_cm["ADMISSION_MONTH"])
    _cm = _cm.sort_values("MONTH").reset_index(drop=True)
    if len(_cm) >= 2 and "OPD_VISITS" in _cm.columns:
        if _safe(_cm.iloc[-1]["OPD_VISITS"], 0) < _safe(_cm.iloc[-2]["OPD_VISITS"], 0) * 0.5:
            _cm = _cm.iloc[:-1].reset_index(drop=True)
    if not _cm.empty:
        _conv_rate_cur = _safe(_cm.iloc[-1]["CONVERSION_RATE"])
        _conv_rate_lbl = _cm.iloc[-1]["MONTH"].strftime("%b %Y")
        if len(_cm) >= 2:
            _cv_old = _safe(_cm.iloc[-2]["CONVERSION_RATE"])
            _conv_prior_lbl = _cm.iloc[-2]["MONTH"].strftime("%b")
            if _conv_rate_cur is not None and _cv_old is not None:
                _cd = _conv_rate_cur - _cv_old
                _ca = "↑" if _cd > 0 else ("↓" if _cd < 0 else "→")
                _conv_rate_trend = f"{_ca} {abs(_cd):.2f}pp vs {_conv_prior_lbl}"

_pharm_fulfill_rate = _safe(_pharm_fulfill.get("FULFILLMENT_RATE")) if _pharm_fulfill is not None else None
_pharm_fulfill_s = f"{_pharm_fulfill_rate:.1f}%" if _pharm_fulfill_rate is not None else "—"
_pharm_dispensed = _safe(_pharm_fulfill.get("DISPENSED_ORDERS")) if _pharm_fulfill is not None else None
_pharm_total = _safe(_pharm_fulfill.get("TOTAL_ORDERS")) if _pharm_fulfill is not None else None
_pharm_mom = _safe(_pharm_fulfill.get("MOM_DELTA_PP")) if _pharm_fulfill is not None else None
_pharm_flow_label = "V2 fulfillment · status=2"
_pharm_flow_badge = None
_pharm_flow_badge_color = None
if _pharm_mom is not None:
    _pharm_arrow = "↑" if _pharm_mom > 0 else "↓" if _pharm_mom < 0 else "→"
    _pharm_flow_badge = f"{_pharm_arrow} {abs(_pharm_mom):.1f}pp vs prior month"
    _pharm_flow_badge_color = COLORS["success"] if _pharm_mom > 0 else COLORS["danger"] if _pharm_mom < 0 else COLORS["muted"]
if _pharm_dispensed is not None and _pharm_total is not None:
    _pharm_flow_label += f" · {int(_pharm_dispensed):,} of {int(_pharm_total):,}"

s1, a1, s2, a2, s3, a3, s4, a4, s5 = st.columns([2.5, 0.4, 2.5, 0.4, 2.5, 0.4, 2.5, 0.4, 2.5])
with s1:
    st.markdown(
        _flow_card_html("Registration", _arr_s, "Arrivals · 7-day avg"),
        unsafe_allow_html=True,
    )
with a1:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with s2:
    st.markdown(
        _flow_card_html(
            "Consult", _cons_p50s, "Arrival → doctor · median wait",
            badge_text=_cons_lbl, badge_color=_cons_badge_col,
        ),
        unsafe_allow_html=True,
    )
with a2:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with s3:
    st.markdown(
        _flow_card_html("Lab", _lab_tat_s, "Order → result · median · V2"),
        unsafe_allow_html=True,
    )
with a3:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with s4:
    st.markdown(
        _flow_card_html("Imaging", _img_tat_s, "Order → arrival · median"),
        unsafe_allow_html=True,
    )
with a4:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with s5:
    st.markdown(
        _flow_card_html(
            "Pharmacy", _pharm_fulfill_s, _pharm_flow_label,
            badge_text=_pharm_flow_badge, badge_color=_pharm_flow_badge_color,
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)

# ── Section 3: Alerts ─────────────────────────────────────────────────────────
section_header("Alerts")

_alert_domains = [
    {
        "title": "Patient Waiting",    "icon": "fa-solid fa-hourglass-half",
        "status": _waiting_status,     "urgency": _waiting_urgency,
        "story": _waiting_story,       "metrics": _waiting_metrics,
        # No dedicated "waiting" page exists — Patient Flow (opd) is where
        # waiting-time/bottleneck analysis actually lives (see pages/1_opd.py).
        "href": nav_url("opd"),        "href_label": "Patient Waiting",
    },
    {
        "title": "Patient Drop-off",   "icon": "fa-solid fa-route",
        "status": _dropoff_status,     "urgency": _dropoff_urgency,
        "story": _dropoff_story,       "metrics": _dropoff_metrics,
        "href": nav_url("dropoff"),    "href_label": "Patient Drop-off",
    },
    {
        "title": "Revenue Leakage",    "icon": "fa-solid fa-file-invoice-dollar",
        "status": _leakage_status,     "urgency": _leakage_urgency,
        "story": _leakage_story,       "metrics": _leakage_metrics,
        "href": nav_url("leakage"),    "href_label": "Revenue Leakage",
    },
    {
        "title": "Diagnostics",        "icon": "fa-solid fa-vials",
        "status": _diag_status,        "urgency": _diag_urgency,
        "story": _diag_story,          "metrics": _diag_findings,
        "href": nav_url("diagnostics"), "href_label": "Diagnostics",
    },
]

_critical_domains = [d for d in _alert_domains if d["status"] == "RED"]
_watch_domains    = [d for d in _alert_domains if d["status"] in ("AMBER",)]
_stable_domains   = [d for d in _alert_domains if d["status"] == "GREEN"]

def _render_alert_row(domains):
    if not domains:
        return
    cols = st.columns(3, gap="medium")
    for col, d in zip(cols, domains):
        with col:
            st.markdown(
                _domain_card(
                    d["title"], d["icon"],
                    d["status"], d["urgency"][0], d["urgency"][1],
                    d["story"], d["metrics"], d["href"], d["href_label"],
                ),
                unsafe_allow_html=True,
            )

if _critical_domains:
    st.markdown(
        '<div style="font-size:9px;font-weight:700;color:#C0392B;text-transform:uppercase;'
        'letter-spacing:2px;margin:0 0 10px 2px">🔴 Critical</div>',
        unsafe_allow_html=True,
    )
    _render_alert_row(_critical_domains)
    st.markdown('<div style="margin-bottom:16px"></div>', unsafe_allow_html=True)

if _watch_domains:
    st.markdown(
        '<div style="font-size:9px;font-weight:700;color:#D68910;text-transform:uppercase;'
        'letter-spacing:2px;margin:0 0 10px 2px">🟡 Watch</div>',
        unsafe_allow_html=True,
    )
    _render_alert_row(_watch_domains)
    st.markdown('<div style="margin-bottom:16px"></div>', unsafe_allow_html=True)

if _stable_domains:
    st.markdown(
        '<div style="font-size:9px;font-weight:700;color:#1E8449;text-transform:uppercase;'
        'letter-spacing:2px;margin:0 0 10px 2px">🟢 Stable</div>',
        unsafe_allow_html=True,
    )
    _render_alert_row(_stable_domains)

# ── Section: Operational Intelligence ─────────────────────────────────────────
_render_intelligence(_intel)

st.markdown(
    '<div style="margin:20px 0 12px;border-top:1px solid #EBF3FB;padding-top:12px;'
    'font-size:10px;font-weight:700;color:#9BAEC8;text-transform:uppercase;letter-spacing:1.5px">'
    'Service Metrics</div>',
    unsafe_allow_html=True,
)

# Row 2 — service metrics (no badge — no validated scoring model)
p1, p2, p3, p4 = st.columns(4, gap="medium")
with p1:
    _lab_svc_val  = _lab_tat_total if _lab_tat_total is not None else None
    _lab_svc_lbl  = "Order → result · median · V2" if _lab_svc_val is not None else "Completion rate · 28-day avg · V2"
    _lab_svc_narr = (
        "V2 collection + processing chain · see Diagnostics for breakdown"
        if _lab_svc_val is not None
        else "28-day avg · V2 orders resulted · see Diagnostics for full breakdown"
    )
    st.markdown(
        _pulse_card(
            "Lab", "fa-solid fa-flask",
            _lab_svc_val if _lab_svc_val is not None else _lab_comp_pct,
            _lab_svc_lbl,
            _lab_svc_narr,
            nav_url("diagnostics"), "Diagnostics",
            unit="min" if _lab_svc_val is not None else "%",
        ),
        unsafe_allow_html=True,
    )
with p2:
    st.markdown(
        _pulse_card(
            "Imaging", "fa-solid fa-x-ray",
            _img_p50, "Order → radiology arrival · 28-day avg",
            _tat_narrative("Imaging", _img_p50),
            nav_url("diagnostics"), "Diagnostics",
        ),
        unsafe_allow_html=True,
    )
with p3:
    _pharm_narrative = (
        f"V2 coverage {_pharm_cov:.0f}% · request → dispensed" if _pharm_cov else "V2 · request → dispensed"
    )
    st.markdown(
        _pulse_card(
            "Pharmacy", "fa-solid fa-pills",
            _pharm_p50, "Dispensing interval · V2",
            _pharm_narrative,
            nav_url("pharmacy"), "Pharmacy",
        ),
        unsafe_allow_html=True,
    )
with p4:
    st.markdown(
        _pulse_card(
            "Theatre", "fa-solid fa-scalpel", 91.2,
            "Illustrative utilisation · May 2023 · V1",
            "Hypothetical only · 167.8 recorded hours / 184 assumed hours",
            nav_url("admissions"), "Theatre",
            unit="%",
        ),
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="margin:20px 0 12px;'
    'font-size:10px;font-weight:700;color:#9BAEC8;text-transform:uppercase;letter-spacing:1.5px">'
    'Operational Signals &nbsp;·&nbsp; Informational</div>',
    unsafe_allow_html=True,
)

o1, o2, o3 = st.columns(3, gap="medium")
with o1:
    st.markdown(
        _signal_card(
            "Lab Orders", "fa-solid fa-vial",
            _lab_per100_cur,
            f"Orders per 100 OPD visits · {_lab_per100_lbl}",
            _lab_per100_trend or "Insufficient history",
            nav_url("diagnostics"), "Diagnostics",
        ),
        unsafe_allow_html=True,
    )
with o2:
    st.markdown(
        _signal_card(
            "Imaging Orders", "fa-solid fa-x-ray",
            _img_per100_cur,
            f"Orders per 100 OPD visits · {_img_per100_lbl}",
            _img_per100_trend or "Insufficient history",
            nav_url("diagnostics"), "Diagnostics",
        ),
        unsafe_allow_html=True,
    )
with o3:
    st.markdown(
        _signal_card(
            "OPD → Admission", "fa-solid fa-bed",
            _conv_rate_cur,
            f"Conversion rate · {_conv_rate_lbl}",
            _conv_rate_trend or "Insufficient history",
            nav_url("admissions"), "Admissions",
            decimals=2,
        ),
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:36px'></div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
_v2_to = _freshness.strftime("%b %Y") if _freshness is not None else "—"
st.markdown(
    f'<div style="border-top:1px solid #EBF3FB;padding-top:14px;'
    f'font-size:11px;color:#9BAEC8;line-height:1.9">'
    f'<b style="color:#6B8CAE">V2 · Operational:</b> Feb 2025 – {_v2_to} &nbsp;·&nbsp; '
    f'RBI labelled "Relative" until SLA thresholds defined by governance &nbsp;·&nbsp; '
    f'Operational signals are informational — no alert thresholds set.</div>',
    unsafe_allow_html=True,
)
