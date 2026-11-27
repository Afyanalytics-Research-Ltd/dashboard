"""
KSH V3 â€” Operations Dashboard Â· Homepage
Sections:
  1. Patient Journey Chain (flow chain, always visible)
  2. Operational Status (unified â€” ðŸ”´ Alert / ðŸŸ¡ Watch / ðŸŸ¢ Stable / â¬œ No Status)
"""

import streamlit as st
import pandas as pd

from ksh_operations.dashboard.theme import apply_theme, render_sidebar, section_header, page_header
from ksh_operations.dashboard.notifier import write_current_notices
from ksh_operations.dashboard.queries import (
    _load_row,
    q_opd_homepage,
    q_lab_homepage,
    q_pharmacy_homepage,
    q_inpatient_homepage,
    q_inpatient_bor,
    q_inpatient_bor_monthly,
    q_inpatient_breach,
    q_inpatient_flow,
    q_opd_lwbs_homepage,
    q_theatre_homepage,
    q_clinician_homepage,
)
from ksh_operations.dashboard.components import (
    COLORS,
    _safe,
    _h_arrow_html,
    _flow_card_html,
    _domain_card,
)

apply_theme()

# â”€â”€ Data load (all queries run once; _load_row returns a dict or {}) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_opd       = _load_row(q_opd_homepage)
_lab       = _load_row(q_lab_homepage)
_pharmacy  = _load_row(q_pharmacy_homepage)
_inpatient = _load_row(q_inpatient_homepage)
_bor       = _load_row(q_inpatient_bor)
_borm      = _load_row(q_inpatient_bor_monthly)
_breach    = _load_row(q_inpatient_breach)
_flow      = _load_row(q_inpatient_flow)
_lwbs_hp   = _load_row(q_opd_lwbs_homepage)
_theatre   = _load_row(q_theatre_homepage)
_clinician = _load_row(q_clinician_homepage)

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _fmt_date(v):
    try:
        return pd.to_datetime(v).strftime('%d %b')
    except Exception:
        return "â€”"


def _fmt_month(v):
    try:
        return pd.to_datetime(v).strftime('%b')
    except Exception:
        return "prior month"


def _dir_html(val, prior, higher_is_bad=True):
    """Colored â†‘/â†“ + prior value. Empty string when no prior or change < 1 unit."""
    if val is None or prior is None:
        return ""
    delta = val - prior
    if abs(delta) < 1:
        return f' <span style="font-size:12px;color:#9BAEC8">â†’ {int(prior)} last month</span>'
    better = (delta < 0) == higher_is_bad
    color  = "#0F6E56" if better else "#E11D48"
    arrow  = "â†“" if delta < 0 else "â†‘"
    return f' <span style="font-size:14px;font-weight:700;color:{color}">{arrow} {int(prior)} last month</span>'


def _big(val_s):
    return f'<span style="font-size:30px;font-weight:900;color:#003467;line-height:1.1">{val_s}</span>'


def _pct_badge(pct_change, amber_thresh=10, red_thresh=20, label="vs prior"):
    """(badge_text, badge_color) for TAT-type metrics where up = bad. None, None if in range."""
    v = _safe(pct_change)
    if v is None:
        return None, None
    arrow = "â†‘" if v > 0 else "â†“"
    text = f"{arrow} {abs(v):.0f}% {label}"
    if v > red_thresh:
        return text, COLORS["danger"]
    if v > amber_thresh:
        return text, COLORS["warning"]
    return None, None


# â”€â”€ Page header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

page_header("Operations Dashboard", subtitle="Patient journey Â· real-time operational status")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 1 â€” Patient Journey Chain
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section_header("Patient Journey")

_opd_from = _fmt_date(_opd.get("PERIOD_FROM")) if _opd else "â€”"
_opd_to   = _fmt_date(_opd.get("PERIOD_TO"))   if _opd else "â€”"
st.markdown(
    f'<div style="font-size:12px;color:#6B8CAE;margin-bottom:16px">'
    f'Each module shows its freshest reliable window â€” see card labels for period. '
    f'OPD data {_opd_from} â€“ {_opd_to}.</div>',
    unsafe_allow_html=True,
)

# â”€â”€ Node values â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# 1. OPD Arrivals â€” no badge (no baseline for visit volume)
_opd_vol   = _safe(_opd.get("AVG_DAILY_VISITS")) if _opd else None
_opd_vol_s = f"{int(_opd_vol)}/day" if _opd_vol is not None else "â€”"
_opd_lbl   = f"{_opd_from} â€“ {_opd_to}"

# 2. Door-to-Doctor â€” badge on wait_pct_change vs 28-day rolling avg (up = bad)
_wait_p50  = _safe(_opd.get("WAIT_P50")) if _opd else None
_wait_s    = f"{int(_wait_p50)} min" if _wait_p50 is not None else "â€”"
_timed_n   = _safe(_opd.get("TIMED_N")) if _opd else None
_wait_lbl  = f"n={int(_timed_n):,} timed visits Â· 7-day" if _timed_n else "coverage data unavailable"
_wait_badge, _wait_color = _pct_badge(
    _opd.get("WAIT_PCT_CHANGE") if _opd else None, label="vs 28-day avg"
)

# 3. Lab TAT â€” badge on MoM delta vs prior complete month (up = bad)
_lab_p50   = _safe(_lab.get("TAT_P50")) if _lab else None
_lab_s     = f"{int(_lab_p50)} min" if _lab_p50 is not None else "â€”"
_lab_ord   = _safe(_lab.get("ORDER_COUNT")) if _lab else None
_lab_cov   = _safe(_lab.get("COVERAGE_PCT")) if _lab else None
_lab_mon   = _fmt_month(_lab.get("MONTH")) if _lab else "â€”"
_lab_lbl   = (f"{int(_lab_ord):,} orders Â· {_lab_cov:.0f}% timed Â· {_lab_mon}"
              if _lab_ord and _lab_cov else "â€”")
_lab_prior = _fmt_month(_lab.get("PRIOR_MONTH")) if _lab else "prior"
_lab_badge, _lab_color = _pct_badge(
    _lab.get("TAT_MOM_PCT") if _lab else None, label=f"vs {_lab_prior}"
)

# 4. Pharmacy TAT â€” badge on MoM delta vs prior complete month (up = bad)
_rx_p50    = _safe(_pharmacy.get("TAT_P50")) if _pharmacy else None
_rx_s      = f"{int(_rx_p50)} min" if _rx_p50 is not None else "â€”"
_rx_count  = _safe(_pharmacy.get("PRESCRIPTION_COUNT")) if _pharmacy else None
_rx_mon    = _fmt_month(_pharmacy.get("MONTH")) if _pharmacy else "â€”"
_rx_lbl    = f"{int(_rx_count):,} Rx Â· {_rx_mon}" if _rx_count else "â€”"
_rx_prior  = _fmt_month(_pharmacy.get("PRIOR_MONTH")) if _pharmacy else "prior"
_rx_badge, _rx_color = _pct_badge(
    _pharmacy.get("TAT_MOM_PCT") if _pharmacy else None, label=f"vs {_rx_prior}"
)

# 5. Inpatient â€” badge on alos_vs_baseline_days (positive = worse)
_open_adm  = _safe(_inpatient.get("OPEN_ADMISSION_COUNT")) if _inpatient else None
_open_s    = f"{int(_open_adm)} open" if _open_adm is not None else "â€”"
_adm_cnt   = _safe(_inpatient.get("ADMISSION_COUNT")) if _inpatient else None
_inp_week  = _fmt_date(_inpatient.get("WEEK_START")) if _inpatient else "â€”"
_inp_lbl   = f"{int(_adm_cnt)} admissions Â· {_inp_week}" if _adm_cnt else "â€”"
_alos_d    = _safe(_inpatient.get("ALOS_VS_BASELINE_DAYS")) if _inpatient else None
if _alos_d is not None and _alos_d > 0.5:
    _inp_badge  = f"+{_alos_d:.1f}d vs 3wk avg"
    _inp_color  = COLORS["danger"] if _alos_d > 1.0 else COLORS["warning"]
else:
    _inp_badge, _inp_color = None, None

# 6. Theatre â€” badge on MoM completion rate delta (negative = worse)
_comp_pct  = _safe(_theatre.get("COMPLETION_RATE_PCT")) if _theatre else None
_comp_s    = f"{_comp_pct:.0f}%" if _comp_pct is not None else "â€”"
_due_cnt   = _safe(_theatre.get("DUE_COUNT")) if _theatre else None
_th_mon    = _fmt_month(_theatre.get("MONTH")) if _theatre else "â€”"
_th_lbl    = f"{int(_due_cnt)} cases Â· {_th_mon}" if _due_cnt else "â€”"
_comp_d    = _safe(_theatre.get("COMPLETION_RATE_MOM_PP")) if _theatre else None
_th_prior  = _fmt_month(_theatre.get("PRIOR_MONTH")) if _theatre else "prior"
# suppress badge when due_count < 10 â€” too few cases for a meaningful MoM signal
if _comp_d is not None and _comp_d < -5 and (_due_cnt is not None and _due_cnt >= 10):
    _th_badge  = f"â†“ {abs(_comp_d):.0f}pp vs {_th_prior}"
    _th_color  = COLORS["danger"] if _comp_d < -10 else COLORS["warning"]
else:
    _th_badge, _th_color = None, None

# â”€â”€ Render â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
c1, a1, c2, a2, c3, a3, c4, a4, c5, a5, c6 = st.columns(
    [2.5, 0.4, 2.5, 0.4, 2.5, 0.4, 2.5, 0.4, 2.5, 0.4, 2.5]
)
with c1:
    st.markdown(_flow_card_html("OPD Arrivals", _opd_vol_s, _opd_lbl), unsafe_allow_html=True)
with a1:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with c2:
    st.markdown(_flow_card_html("Door to Doctor", _wait_s, _wait_lbl,
                                badge_text=_wait_badge, badge_color=_wait_color), unsafe_allow_html=True)
with a2:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with c3:
    st.markdown(_flow_card_html("Lab TAT", _lab_s, _lab_lbl,
                                badge_text=_lab_badge, badge_color=_lab_color), unsafe_allow_html=True)
with a3:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with c4:
    st.markdown(_flow_card_html("Pharmacy TAT", _rx_s, _rx_lbl,
                                badge_text=_rx_badge, badge_color=_rx_color), unsafe_allow_html=True)
with a4:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with c5:
    st.markdown(_flow_card_html("Inpatient", _open_s, _inp_lbl,
                                badge_text=_inp_badge, badge_color=_inp_color), unsafe_allow_html=True)
with a5:
    st.markdown(_h_arrow_html, unsafe_allow_html=True)
with c6:
    st.markdown(_flow_card_html("Theatre", _comp_s, _th_lbl,
                                badge_text=_th_badge, badge_color=_th_color), unsafe_allow_html=True)

st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 2 â€” Operational Status (unified â€” green stays, amber/red promote)
# shortcut: demotion to Stable requires two consecutive clean periods â€” needs
# persistent state; real-time status used here, upgrade when state store added
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section_header("Alerts", margin_top=8)

_data_line_parts = [
    f"OPD to {_opd_to}" if _opd and _opd_to != "â€”" else None,
    f"Lab/Pharmacy {_lab_mon}" if _lab else None,
    f"Inpatient wk {_inp_week}" if _inpatient and _inp_week != "â€”" else None,
    f"Theatre {_th_mon}" if _theatre else None,
]
st.markdown(
    f'<div style="font-size:11px;color:#9BAEC8;margin:-4px 0 14px">'
    f'Data up to: {" Â· ".join(p for p in _data_line_parts if p)}</div>',
    unsafe_allow_html=True,
)

# â”€â”€ Additional extracts (beyond Section 1 variables) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_bor_pct       = _safe(_bor.get("BOR_PCT"))                  if _bor else None
_bor_ward      = str(_bor.get("WARD_NAME", ""))              if _bor else "â€”"
_lab_covn      = _safe(_lab.get("COVERAGE_N"))               if _lab else None
_lab_o1d       = _safe(_lab.get("OVER_1D_COUNT"))            if _lab else None
_lab_prior_p50 = _safe(_lab.get("PRIOR_TAT_P50"))            if _lab else None
_rx_covn       = _safe(_pharmacy.get("COVERAGE_N"))          if _pharmacy else None
_rx_prior_p50  = _safe(_pharmacy.get("PRIOR_TAT_P50"))       if _pharmacy else None
_th_prior_pct  = _safe(_theatre.get("PRIOR_COMPLETION_RATE_PCT")) if _theatre else None
_th_emrg       = _safe(_theatre.get("EMERGENCY_RATE_PCT"))   if _theatre else None
_th_completed  = _safe(_theatre.get("COMPLETED_COUNT"))      if _theatre else None

_disch_timed  = _safe(_breach.get("DISCHARGE_TIMED_N"))      if _breach else None
_disch_4h     = _safe(_breach.get("DISCHARGE_OVER_4H_N"))    if _breach else None
_c2b_timed    = _safe(_breach.get("C2B_TIMED_N"))            if _breach else None
_c2b_4h       = _safe(_breach.get("C2B_OVER_4H_N"))          if _breach else None
_disch_bpct   = round(_disch_4h / _disch_timed * 100, 1) if (_disch_timed and _disch_4h is not None) else None
_c2b_bpct     = round(_c2b_4h / _c2b_timed * 100, 1)     if (_c2b_timed and _c2b_4h is not None)   else None
_breach_from  = _fmt_date(_breach.get("PERIOD_FROM"))        if _breach else "â€”"
_breach_to    = _fmt_date(_breach.get("PERIOD_TO"))          if _breach else "â€”"

# â”€â”€ Status (GREEN / AMBER / RED / GREY) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# OPD: share over 40 min â€” 7-day pooled counts (n â‰¥ 30, bands <20% / 20â€“23% / â‰¥23%)
# 26-week history: p75=18.2%, p90=20.5% â†’ amber at 20% fires ~10% of weeks; red at 23% fires rarely
_opd_over_a   = _safe(_opd.get("WAIT_OVER_A_N")) if _opd else None
_opd_over_a_p = round(_opd_over_a / _timed_n * 100, 1) if (_opd_over_a is not None and _timed_n) else None
if _timed_n is None or _timed_n < 30:         _opd_st = "GREY"
elif _opd_over_a_p is not None and _opd_over_a_p >= 23: _opd_st = "RED"
elif _opd_over_a_p is not None and _opd_over_a_p >= 20: _opd_st = "AMBER"
else:                                          _opd_st = "GREEN"

# Lab: last complete month P50 (n â‰¥ 200, bands â‰¤80 / >80â€“90 / >90 min)
# 60-min clinical target never met in 25 months; amber at 80 fires 9/25, red at 90 is the ceiling
if _lab_covn is None or _lab_covn < 200:      _lab_st = "GREY"
elif _lab_p50 and _lab_p50 > 90:              _lab_st = "RED"
elif _lab_p50 and _lab_p50 > 80:              _lab_st = "AMBER"
else:                                          _lab_st = "GREEN"

# Pharmacy: last complete month P50 (n â‰¥ 100, bands â‰¤35 / >35â€“45 / >45 min)
# 30-min amber fired 16/25 months (median); 35-min fires 6/25, 2 of last 9 â€” calibrated against history
if _rx_covn is None or _rx_covn < 100:        _rx_st = "GREY"
elif _rx_p50 and _rx_p50 > 45:                _rx_st = "RED"
elif _rx_p50 and _rx_p50 > 35:                _rx_st = "AMBER"
else:                                          _rx_st = "GREEN"

# Inpatient: discharge breach rate trailing 4 weeks (n â‰¥ 50, bands <20% / 20â€“30% / â‰¥30%)
# 20-week history: range 2.9â€“30.4%; red at 30% fires ~15% of weeks â€” calibrated 2026-11-17
if _disch_timed is None or _disch_timed < 50: _inp_st = "GREY"
elif _disch_bpct >= 30:                        _inp_st = "RED"
elif _disch_bpct >= 20:                        _inp_st = "AMBER"
else:                                          _inp_st = "GREEN"

# Theatre: elective completion rate last complete month (n â‰¥ 20, bands â‰¥85% / 80â€“85% / <80%)
# 24-month history: floor is 78â€“80%; red at 70% has never fired â€” recalibrated 2026-11-17
if _due_cnt is None or _due_cnt < 20:          _th_st = "GREY"
elif _comp_pct is not None and _comp_pct < 80: _th_st = "RED"
elif _comp_pct is not None and _comp_pct < 85: _th_st = "AMBER"
else:                                           _th_st = "GREEN"

# Near-limit: GREEN within 10% of amber threshold â€” signals likely flip on next period
_lab_near = _lab_st == "GREEN" and _lab_p50 is not None and _lab_p50 >= 72    # 80 Ã— 0.9
_inp_near = _inp_st == "GREEN" and _disch_bpct is not None and _disch_bpct >= 18  # 20 Ã— 0.9

# â”€â”€ Signal metrics (now threshold-monitored, fold into _domains) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# OPD conversion â€” flag HIGH (unusual admission pressure)
# 26-week distribution: p50=4.88%, p90=6.15% â†’ amber â‰¥6.5%, red â‰¥8.0%
_conv_pct       = _safe(_flow.get("ANY_CONVERSION_PCT"))       if _flow else None
_prior_conv_pct = _safe(_flow.get("PRIOR_ANY_CONVERSION_PCT")) if _flow else None
_conv_adm       = _safe(_flow.get("CONVERTED_ANY"))            if _flow else None
_conv_opd       = _safe(_flow.get("OPD_VISIT_COUNT"))          if _flow else None
_conv_mon       = _fmt_month(_flow.get("VISIT_MONTH"))         if _flow else "â€”"
if _conv_pct is None:            _conv_st = "GREY"
elif _conv_pct >= 8.0:           _conv_st = "RED"
elif _conv_pct >= 6.5:           _conv_st = "AMBER"
else:                            _conv_st = "GREEN"

# LWBS â€” flag HIGH (patients left before seeing doctor)
# 21-week distribution: p50=17.2%, p90=19.4% â†’ amber â‰¥20%, red â‰¥23%
_lwbs_pct    = _safe(_lwbs_hp.get("LWBS_PCT"))       if _lwbs_hp else None
_lwbs_prior  = _safe(_lwbs_hp.get("PRIOR_LWBS_PCT")) if _lwbs_hp else None
_lwbs_cnt    = _safe(_lwbs_hp.get("LWBS_COUNT"))      if _lwbs_hp else None
_lwbs_tot    = _safe(_lwbs_hp.get("TOTAL_VISITS"))    if _lwbs_hp else None
_lwbs_from   = _fmt_date(_lwbs_hp.get("PERIOD_FROM")) if _lwbs_hp else "â€”"
_lwbs_to     = _fmt_date(_lwbs_hp.get("PERIOD_TO"))   if _lwbs_hp else "â€”"
if _lwbs_pct is None:    _lwbs_st = "GREY"
elif _lwbs_pct >= 23:    _lwbs_st = "RED"
elif _lwbs_pct >= 20:    _lwbs_st = "AMBER"
else:                    _lwbs_st = "GREEN"

# Pharmacy fulfillment â€” flag LOW (dispensing shortfall)
# 22-month distribution: p10=79.4%, min=75.3% â†’ amber <80%, red <77%
_fulfil_pct     = _safe(_pharmacy.get("FULFILLMENT_PCT"))       if _pharmacy else None
_fulfil_prior   = _safe(_pharmacy.get("PRIOR_FULFILLMENT_PCT")) if _pharmacy else None
_fulfil_rx      = _safe(_pharmacy.get("PRESCRIPTION_COUNT"))    if _pharmacy else None
_fulfil_disp    = _safe(_pharmacy.get("DISPENSED_COUNT"))       if _pharmacy else None
_fulfil_mon     = _fmt_month(_pharmacy.get("MONTH"))            if _pharmacy else "â€”"
if _fulfil_pct is None:          _fulfil_st = "GREY"
elif _fulfil_pct < 77:           _fulfil_st = "RED"
elif _fulfil_pct < 80:           _fulfil_st = "AMBER"
else:                            _fulfil_st = "GREEN"

# BOR: facility-level last complete month (bands <50=RED, 50â€“60=AMBER, â‰¥60=GREEN)
# 24-month facility range 28â€“69%; standard 80â€“85% thresholds not applicable â€” calibrated 2026-11-17
_borm_pct = _safe(_borm.get("BOR_PCT"))      if _borm else None
_borm_mon = _fmt_month(_borm.get("MONTH"))   if _borm else "â€”"
_borm_mom = _safe(_borm.get("MOM_DELTA_PP")) if _borm else None
if _borm_pct is None:        _bor_m_st = "GREY"
elif _borm_pct < 50:         _bor_m_st = "RED"
elif _borm_pct < 60:         _bor_m_st = "AMBER"
else:                        _bor_m_st = "GREEN"
_bor_drop_flag = _borm_mom is not None and _borm_mom <= -15

# â”€â”€ Build domain cards â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_STATUS_COLOR = {
    "GREEN": COLORS["success"], "AMBER": COLORS["warning"],
    "RED":   COLORS["danger"],  "GREY":  COLORS["muted"],
}

_domains = []  # (status, card_html) â€” split into bands below

# OPD
_opd_p50s   = f"{int(_wait_p50)} min" if _wait_p50 else "â€”"
_opd_a_pcts = f"{_opd_over_a_p:.0f}%" if _opd_over_a_p is not None else "â€”"
_domains.append((_opd_st, _domain_card(
    "OPD â€” time to see a doctor", "fas fa-clock",
    _opd_st,
    "Waited over 40 min",
    _STATUS_COLOR[_opd_st],
    _big(_opd_a_pcts) if _opd_st != "GREY" else _big("â€”"),
    [f"20% limit Â· {_opd_from}â€“{_opd_to}",
     f"Typical wait: {_opd_p50s} Â· {int(_timed_n):,} visits" if _timed_n else ""],
    "/1_OPD_Flow", "OPD detail", show_label=False,
)))

# Lab
_lab_p50s = f"{int(_lab_p50)} min" if _lab_p50 else "â€”"
_lab_uc   = COLORS["warning"] if _lab_near else _STATUS_COLOR[_lab_st]
_lab_o1d_s = f" Â· {int(_lab_o1d)} over 24 hrs" if _lab_o1d is not None else ""
_domains.append((_lab_st, _domain_card(
    "Lab results", "fas fa-flask",
    _lab_st,
    "Result time (median)",
    _lab_uc,
    _big(_lab_p50s) + _dir_html(_lab_p50, _lab_prior_p50) if _lab_st != "GREY" else _big("â€”"),
    [f"80 min limit Â· {_lab_mon}",
     f"{int(_lab_covn):,} results{_lab_o1d_s} Â· Lab only, radiology excluded" if _lab_covn else ""],
    "/2_Lab_TAT", "Lab detail", show_label=False,
)))

# Pharmacy
_rx_p50s = f"{int(_rx_p50)} min" if _rx_p50 else "â€”"
_domains.append((_rx_st, _domain_card(
    "Pharmacy", "fas fa-pills",
    _rx_st,
    "Dispensing time (median)",
    _STATUS_COLOR[_rx_st],
    _big(_rx_p50s) + _dir_html(_rx_p50, _rx_prior_p50) if _rx_st != "GREY" else _big("â€”"),
    [f"35 min limit Â· {_rx_mon}",
     f"{int(_rx_count):,} drug lines Â· Main pharmacy only" if _rx_count else "Main pharmacy only"],
    "/3_Pharmacy_TAT", "Pharmacy detail", show_label=False,
)))

# Inpatient
_disch_4h_i = int(_disch_4h) if _disch_4h is not None else None
_disch_t_i  = int(_disch_timed) if _disch_timed else None
_dbpct_s    = f"{_disch_bpct:.0f}%" if _disch_bpct is not None else "â€”"
_inp_count_s = (f' <span style="font-size:13px;color:#6B8CAE;margin-left:6px">'
                f'{_disch_4h_i} of {_disch_t_i} patients</span>'
                if _disch_4h_i is not None and _disch_t_i else "")
_bor_line   = f"Highest bed occupancy: {_bor_pct:.0f}% ({_bor_ward})" if _bor_pct else ""
_inp_uc     = COLORS["warning"] if _inp_near else _STATUS_COLOR[_inp_st]
_domains.append((_inp_st, _domain_card(
    "Patient discharge", "fas fa-bed",
    _inp_st,
    "Delayed over 4 hours",
    _inp_uc,
    _big(_dbpct_s) + _inp_count_s if _inp_st != "GREY" else _big("â€”"),
    [m for m in [f"20% limit Â· {_breach_from}â€“{_breach_to} Â· After doctor's order",
                 _bor_line] if m],
    "/4_Inpatient", "Inpatient detail", show_label=False,
)))

# OPD conversion
_conv_pct_s  = f"{_conv_pct:.1f}%" if _conv_pct is not None else "â€”"
_conv_cnt_s  = (f"{int(_conv_adm):,} of {int(_conv_opd):,} visits"
                if _conv_adm is not None and _conv_opd is not None else "")
_conv_prior_s = f"Prior month: {_prior_conv_pct:.1f}%" if _prior_conv_pct is not None else ""
_domains.append((_conv_st, _domain_card(
    "OPD conversion", "fas fa-procedures",
    _conv_st,
    "% of OPD visits admitted",
    _STATUS_COLOR[_conv_st],
    _big(_conv_pct_s) if _conv_st != "GREY" else _big("â€”"),
    [m for m in [f"â‰¥6.5% limit Â· {_conv_mon} Â· Overnight + day case",
                 _conv_cnt_s, _conv_prior_s] if m],
    "/1_OPD_Flow", "OPD detail", show_label=False,
)))

# LWBS
_lwbs_pct_s   = f"{_lwbs_pct:.1f}%" if _lwbs_pct is not None else "â€”"
_lwbs_prior_s = f"Prev week: {_lwbs_prior:.1f}%" if _lwbs_prior is not None else ""
_lwbs_cnt_s   = (f"{int(_lwbs_cnt):,} of {int(_lwbs_tot):,} visits"
                 if _lwbs_cnt is not None and _lwbs_tot is not None else "")
_domains.append((_lwbs_st, _domain_card(
    "Left without being seen", "fas fa-person-walking-arrow-right",
    _lwbs_st,
    "LWBS rate",
    _STATUS_COLOR[_lwbs_st],
    _big(_lwbs_pct_s) if _lwbs_st != "GREY" else _big("â€”"),
    [m for m in [f"20% limit Â· {_lwbs_from}â€“{_lwbs_to}",
                 _lwbs_cnt_s, _lwbs_prior_s] if m],
    "/1_OPD_Flow", "OPD detail", show_label=False,
)))

# Pharmacy fulfillment
_fulfil_pct_s  = f"{_fulfil_pct:.1f}%" if _fulfil_pct is not None else "â€”"
_fulfil_cnt_s  = (f"{int(_fulfil_disp):,} of {int(_fulfil_rx):,} drug lines"
                  if _fulfil_disp is not None and _fulfil_rx is not None else "")
_fulfil_prior_s = f"Prior month: {_fulfil_prior:.1f}%" if _fulfil_prior is not None else ""
_domains.append((_fulfil_st, _domain_card(
    "Pharmacy fulfillment", "fas fa-pills",
    _fulfil_st,
    "Drug lines dispensed",
    _STATUS_COLOR[_fulfil_st],
    _big(_fulfil_pct_s) if _fulfil_st != "GREY" else _big("â€”"),
    [m for m in [f"<80% limit Â· {_fulfil_mon} Â· Main pharmacy",
                 _fulfil_cnt_s, _fulfil_prior_s] if m],
    "/3_Pharmacy_TAT", "Pharmacy detail", show_label=False,
)))

# BOR
_borm_pct_s = f"{_borm_pct:.0f}%" if _borm_pct is not None else "â€”"
_bor_lines  = [f"60% green Â· {_borm_mon} Â· Facility-wide"]
if _borm_mom is not None:
    _bor_lines.append(f"{'âš  ' if _bor_drop_flag else ''}{_borm_mom:+.0f}pp vs prior month")
_domains.append((_bor_m_st, _domain_card(
    "Bed occupancy (BOR)", "fas fa-procedures",
    _bor_m_st,
    "Facility occupancy rate",
    _STATUS_COLOR[_bor_m_st],
    _big(_borm_pct_s) if _bor_m_st != "GREY" else _big("â€”"),
    _bor_lines,
    "/4_Inpatient", "Inpatient detail", show_label=False,
)))

# Theatre
_comp_s2   = f"{int(_comp_pct)}%" if _comp_pct else "â€”"
_th_comp_i = int(_th_completed) if _th_completed is not None else None
_th_due_i  = int(_due_cnt) if _due_cnt is not None else None
_emrg_line = f"Emergency: {int(_th_emrg)}% of scheduled" if _th_emrg is not None else ""
_domains.append((_th_st, _domain_card(
    "Operating theatre", "fas fa-syringe",
    _th_st,
    "Operations completed",
    _STATUS_COLOR[_th_st],
    _big(_comp_s2) + _dir_html(_comp_pct, _th_prior_pct, higher_is_bad=False) if _th_st != "GREY" else _big("â€”"),
    [m for m in [f"{_th_comp_i} of {_th_due_i} cases Â· {_th_mon} Â· Target â‰¥ 85%" if _th_comp_i else f"{_th_mon} Â· Target â‰¥ 85%",
                 _emrg_line] if m],
    "/5_Theatre", "Theatre detail", show_label=False,
)))

# â”€â”€ Split into bands and render â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# ── Notices (computed from domain statuses; persisted to JSON + session state) ──
_notices = []
_notice_map = [
    (_opd_st,    "OPD — Patient Wait Time",                   _opd_a_pcts,    "Review staffing at registration and triage stations."),
    (_lab_st,    "Lab Results — TAT Elevated",                _lab_p50s,      "Check lab processing backlog and staffing on current shift."),
    (_rx_st,     "Pharmacy — Dispensing Time",                _rx_p50s,       "Review pharmacy counter queue and dispensing staff allocation."),
    (_inp_st,    "Inpatient — Discharge Delays",              _dbpct_s,       "Escalate delayed discharge cases to ward coordinator."),
    (_th_st,     "Operating Theatre — Completion Rate",       _comp_s2,       "Review cancelled and postponed cases; check scheduling backlog."),
    (_conv_st,   "OPD Conversion — Admission Pressure",       _conv_pct_s,    "Check bed availability and admission bottlenecks."),
    (_lwbs_st,   "LWBS — Patients Left Without Being Seen",   _lwbs_pct_s,    "Investigate triage wait and peak-hour staffing gaps."),
    (_fulfil_st, "Pharmacy Fulfillment — Dispensing Shortfall", _fulfil_pct_s, "Check drug stock levels and procurement orders."),
    (_bor_m_st,  "Bed Occupancy Rate",                        _borm_pct_s,    "Review ward capacity and admission planning."),
]
for _st, _title, _metric, _action in _notice_map:
    if _st in ("RED", "AMBER"):
        _notices.append({"level": "CRITICAL" if _st == "RED" else "WATCH",
                         "title": _title, "metric": _metric, "action": _action})

st.session_state["active_notices"] = _notices
write_current_notices("KSH", _notices)
render_sidebar("overview", show_notify=True)

_band_red   = [(s, h) for s, h in _domains if s == "RED"]
_band_amber = [(s, h) for s, h in _domains if s == "AMBER"]
_band_green = [(s, h) for s, h in _domains if s == "GREEN"]
_band_grey  = [(s, h) for s, h in _domains if s == "GREY"]


def _band_header(label):
    st.markdown(
        f'<div style="font-size:11px;font-weight:800;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin:16px 0 10px">{label}</div>',
        unsafe_allow_html=True,
    )


def _band_render(items):
    rows = [items[i:i + 3] for i in range(0, len(items), 3)]
    for row in rows:
        cols = st.columns([1] * len(row) + [1] * (3 - len(row)))
        for i, (_, html) in enumerate(row):
            with cols[i]:
                st.markdown(html, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)


if not _band_red and not _band_amber:
    st.markdown(
        '<div style="background:#F0FBF8;border:1px solid #0F6E5640;border-left:4px solid #0F6E56;'
        'border-radius:8px;padding:14px 18px;font-size:13px;color:#0F6E56;font-weight:600;margin-bottom:16px">'
        'âœ… No alerts this period</div>',
        unsafe_allow_html=True,
    )

if _band_red:
    _band_header("ðŸ”´ Needs urgent attention")
    _band_render(_band_red)
if _band_amber:
    _band_header("ðŸŸ¡ Needs attention")
    _band_render(_band_amber)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SECTION 4 â€” Operational Signals (all threshold-monitored, bands from _domains)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

section_header("Operational Signals", margin_top=8)

if _band_green:
    _band_render(_band_green)
if _band_grey:
    _band_header("â¬œ No recent data")
    _band_render(_band_grey)

st.markdown("<div style='margin-top:32px'></div>", unsafe_allow_html=True)

# â”€â”€ Footer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown(
    '<div style="margin-top:40px;border-top:1px solid #EBF3FB;padding-top:14px;'
    'font-size:11px;color:#9BAEC8;line-height:1.9">'
    'Gold layer: <b style="color:#6B8CAE">HOSPITALS.REPORTING</b> Â· '
    'Queries cached 1 hour Â· Signal strength declared on every card</div>',
    unsafe_allow_html=True,
)

