"""sidebar.py: dark navigation. Routing uses ?tab=<key> plus session_state."""
from __future__ import annotations

import streamlit as st

SECTIONS = [
    ("OVERVIEW", [("tab_summary", "Summary"), ("tab_geo", "Population & Geography")]),
    ("DISEASE BURDEN", [("tab_infectious", "Infectious disease"), ("tab_ncds", "NCDs"), ("tab_mental", "Psychiatric Conditions"),
                        ("tab_resp", "Respiratory"), ("tab_maternal", "Maternal & Obstetric"), ("tab_injury", "Injury & Trauma")]),
    ("WORKFORCE", [("tab_workload", "Clinician Workload")]),
    ("SPECIAL", [("tab_dataquality", "Data Quality")]),
]
ICON_OF = {"tab_summary": "summary", "tab_demo": "demo", "tab_geo": "geo", "tab_infectious": "infectious", "tab_ncds": "ncd",
           "tab_mental": "mental", "tab_resp": "respiratory", "tab_maternal": "maternal", "tab_injury": "injury",
           "tab_workload": "workload", "tab_dataquality": "dataquality"}
LABELS = {k: v for _, items in SECTIONS for k, v in items}
DEFAULT = "tab_summary"

CSS = """
<style>
section[data-testid="stSidebar"]{background:#402818;}
section[data-testid="stSidebar"] *{color:#EDE4DC;}
section[data-testid="stSidebar"] .nav-sec{font-size:11px;letter-spacing:.1em;color:#BFA88F;margin:18px 0 6px;font-weight:700}
section[data-testid="stSidebar"] .stButton>button{width:100%;text-align:left;justify-content:flex-start;background:transparent;border:0;border-radius:8px;padding:6px 10px;font-size:14px}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.09)}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{background:#43A94B;box-shadow:inset 3px 0 0 #F8C818}
section[data-testid="stSidebar"] .stButton>button[kind="primary"] *{color:#fff !important;font-weight:600}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]::before{filter:brightness(0) invert(1)}
section[data-testid="stSidebar"] a.nav-it{display:flex;align-items:center;gap:10px;padding:8px 10px;margin-bottom:2px;border-radius:8px;font-size:14px;text-decoration:none !important;color:#EDE4DC !important}
section[data-testid="stSidebar"] a.nav-it:hover{background:rgba(255,255,255,.09)}
section[data-testid="stSidebar"] a.nav-it img{width:18px;height:18px;flex:none}
section[data-testid="stSidebar"] a.nav-it.on{background:#43A94B;box-shadow:inset 3px 0 0 #F8C818;color:#fff !important;font-weight:600}
section[data-testid="stSidebar"] a.nav-it.on img{filter:brightness(0) invert(1)}
section[data-testid="stSidebar"] hr{border-color:rgba(255,255,255,.14)}
.crh{padding:6px 4px 14px;margin:0 0 4px;border-bottom:1px solid rgba(255,255,255,.14)}
.crh-k{font-size:10.5px;letter-spacing:.18em;font-weight:700;color:#F8C818 !important}
.crh-n{font-size:22px;font-weight:800;color:#fff !important;margin-top:4px;line-height:1.15}
.crh-s{font-size:11.5px;color:#BFA88F !important;margin-top:4px}
section[data-testid="stSidebar"] .st-key-nav-refresh, section[data-testid="stSidebar"] .st-key-nav-refresh .stButton, section[data-testid="stSidebar"] [data-testid="stEmpty"], section[data-testid="stSidebar"] .st-key-nav-refresh button{width:100% !important}
section[data-testid="stSidebar"] .st-key-nav-refresh button{justify-content:center;text-align:center;background:#F8C818;border-radius:10px;padding:9px 10px;font-weight:700}
section[data-testid="stSidebar"] .st-key-nav-refresh button *{color:#402818 !important;font-weight:700}
section[data-testid="stSidebar"] .st-key-nav-refresh button:hover{background:#FFD83A}
.refresh-pill{display:flex;align-items:center;justify-content:center;gap:10px;padding:9px 10px;font-size:14px;font-weight:700;color:#402818 !important;background:#F8C818;border-radius:10px;opacity:.9}
.refresh-pill .spin{width:14px;height:14px;border:2px solid rgba(64,40,24,.25);border-top-color:#402818;border-radius:50%;animation:afspin .8s linear infinite}
@keyframes afspin{to{transform:rotate(360deg)}}
</style>
"""


def _nav_html(cur: str) -> str:
    """Real links, so a middle-click, Ctrl/Cmd-click or right-click 'Open in new tab' opens that tab in its own browser tab."""
    import nandi_ui_template as ui
    out = []
    for title, items in SECTIONS:
        out.append(f'<div class="nav-sec">{title}</div>')
        for key, label in items:
            on = " on" if key == cur else ""
            out.append(f'<a class="nav-it{on}" href="?tab={key}" target="_self"><img src="{ui.icon_uri(ICON_OF[key])}" alt="">{label}</a>')
    return "".join(out)


def _brand_html() -> str:
    return ('<div class="crh"><div class="crh-k">NANDI COUNTY</div><div class="crh-n">Kapsabet CRH</div>'
            '<div class="crh-s">Referral hospital · Afyanalytics</div></div>')


def current_tab() -> str:
    tab = st.query_params.get("tab") or st.session_state.get("tab") or DEFAULT
    return tab if tab in LABELS else DEFAULT


def render() -> str:
    cur = current_tab()
    st.session_state["tab"] = cur
    with st.sidebar:
        st.markdown(CSS, unsafe_allow_html=True)
        st.markdown(_brand_html() + _nav_html(cur), unsafe_allow_html=True)
        st.markdown("---")
        _slot = st.empty()
        if st.session_state.get("_refreshing"):
            _slot.markdown('<div class="refresh-pill"><span class="spin"></span>Refreshing data…</div>', unsafe_allow_html=True)
        else:
            if _slot.button("↻  Refresh data", key="nav-refresh"):
                st.cache_data.clear()
                st.session_state["_refreshing"] = True
                st.rerun()
        st.session_state["_slot"] = _slot
    return cur


def finish_refresh() -> None:
    """Call after the page has rendered: ends the 'Refreshing' state and restores the button."""
    if st.session_state.pop("_refreshing", False):
        import time; time.sleep(0.9)  # keeps the spinner visible when data loads instantly
        slot = st.session_state.get("_slot")
        if slot is not None:
            slot.button("↻  Refresh data", key="nav-refresh")
        st.toast("Data refreshed", icon="✅")
