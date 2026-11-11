"""
KSH Revenue Intelligence 
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

ROOT = Path(os.path.abspath(__file__)).parent / "revenue_intelligence"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import views
from queries import revenue as R, receivables as AR, leakage as L, analytics as A

views.setup()


@st.cache_data(ttl=3600, show_spinner=False)
def load_all(_v=5):
    dm = A.prepare_monthly(R.get_monthly_summary())
    snap = AR.get_ar_snapshot()
    leak = L.get_leakage_summary()
    exp = A.exposure_bridge(snap, leak)
    conc = R.get_insurer_concentration()
    theatre = L.get_theatre_summary()
    pharm_fulfil = L.get_pharmacy_fulfillment()
    unk = AR.get_unknown_insurer_trend()
    cutoff = AR.get_data_cutoff()
    aging = AR.get_undispatched_aging()
    return dict(
        dm=dm, snap=snap, leak=leak, exp=exp, conc=conc, theatre=theatre,
        pharm_fulfil=pharm_fulfil, unk=unk, cutoff=cutoff, aging=aging,
        signals=A.build_signals(dm, exp, unk, conc),
        actions=A.action_center(exp, snap, theatre, pharm_fulfil, conc),
        svc=R.get_service_line_monthly(), dow=R.get_timing_dow(), top_items=R.get_top_items(),
        dispatch=AR.get_dispatch_trend(), collections=AR.get_collections_monthly(),
        pharm_trend=L.get_pharmacy_trend(), top_drugs=L.get_top_drugs(),
        doctors=L.get_doctor_leakage(), outlier=L.get_pharmacy_outlier(),
        theatre_rej=L.get_theatre_rejections(), dq=L.get_data_quality(),
    )


with st.spinner("Loading revenue intelligence…"):
    D = load_all()

dm = D["dm"]
cur, prior = A.latest_complete(dm)
exp = D["exp"]
cutoff = D["cutoff"]
lag = A.data_lag_days(D["snap"], cutoff)
aging_dist = A.aging_distribution(D["aging"], exp.undispatched)
ninety_plus = float(aging_dist.loc[aging_dist["AGING_BUCKET"] == "90+ days", "KES"].sum())

_snap = D["snap"].copy()
_snap["OUTSTANDING_KES"] = pd.to_numeric(_snap["OUTSTANDING_KES"], errors="coerce").fillna(0)
_snap["AVG_DAYS_OUTSTANDING"] = pd.to_numeric(_snap["AVG_DAYS_OUTSTANDING"], errors="coerce")
_disp = _snap[_snap["AR_STATE"].str.contains("Dispatched", na=False)].dropna(subset=["AVG_DAYS_OUTSTANDING"])
if _disp["OUTSTANDING_KES"].sum() > 0:
    _w = (_disp["AVG_DAYS_OUTSTANDING"] * _disp["OUTSTANDING_KES"]).sum() / _disp["OUTSTANDING_KES"].sum()
    dispatched_sitting_days = max(0.0, _w - lag)
else:
    dispatched_sitting_days = 0.0

ctx = SimpleNamespace(D=D, dm=dm, cur=cur, prior=prior, exp=exp, cutoff=cutoff, lag=lag,
                      aging_dist=aging_dist, ninety_plus=ninety_plus,
                      dispatched_sitting_days=dispatched_sitting_days,
                      signals=D["signals"], actions=D["actions"])

with st.sidebar:
    # ── Brand (matches facility_operations theme.render_sidebar) ─────────────
    logo = str(ROOT.parent / "logo" / "logo.png")
    logo_col = st.columns([1, 2, 1])
    with logo_col[1]:
        if os.path.exists(logo):
            st.image(logo, use_container_width=True)
    st.markdown(
        '<div style="text-align:center;padding:10px 0 16px">'
        '<div style="font-size:9px;font-weight:700;color:#8BAAC5;'
        'text-transform:uppercase;letter-spacing:2px;margin-bottom:5px">'
        'AFYA</div>'
        '<div style="font-size:17px;font-weight:800;color:#003467;line-height:1.2">'
        'Revenue Intelligence</div>'
        '</div>',
        unsafe_allow_html=True)
    # Button nav (styled by .st-key-side_nav in utils/components.py) — unlike
    # option_menu's iframe it inherits the page's Montserrat font.
    _NAV = [
        ("Executive Brief",    "speed"),
        ("Revenue",            "trending_up"),
        ("Receivables & Cash", "hourglass_top"),
        ("Revenue Leakage",    "water_drop"),
        ("Action Center",      "bolt"),
    ]
    st.session_state.setdefault("rev_page", _NAV[0][0])
    with st.container(key="side_nav"):
        for _label, _icon in _NAV:
            _active = st.session_state["rev_page"] == _label
            if st.button(_label, icon=f":material/{_icon}:", key=f"rev_nav_{_label}",
                         type="primary" if _active else "secondary", width="stretch"):
                st.session_state["rev_page"] = _label
                st.rerun()
    page = st.session_state["rev_page"]
    if A.has_inflight_month(dm) is not None:
        st.markdown(
            f"<div style='font-size:11px;color:#D97706;background:#FFFBEB;border:1px solid #F0C580;"
            f"border-radius:6px;padding:7px 9px;margin-top:12px;line-height:1.4'>"
            f"⚠ {A.mon_label(A.has_inflight_month(dm))} is a partial month — headline figures use "
            f"{A.mon_label(cur['REV_MONTH'])}.</div>", unsafe_allow_html=True)
    views.data_notes_expander(ctx)
    if st.button("↺  Refresh data", use_container_width=True, type="secondary"):
        st.cache_data.clear()
        st.rerun()

_override = st.query_params.get("view")
if _override in views.VIEWS:
    page = _override
views.VIEWS[page](ctx)
