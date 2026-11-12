"""Afyanalytics internal metrics dashboard: Overview, Patient counts and Discovery tabs."""

import os
import sys

import streamlit as st

try:
    st.set_page_config(page_title="Afyanalytics · Metrics", layout="wide",
                       initial_sidebar_state="collapsed")
except st.errors.StreamlitAPIException:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The platform loader re-runs this file in a long-lived process; drop cached modules so edits load.
for _m in [m for m in sys.modules if m == "afya_metrics" or m.startswith("afya_metrics.")]:
    del sys.modules[_m]

from afya_metrics import data as D, discovery, hmis, overview, patients, theme as T, update

T.inject_css()

try:
    rows, _ = D.read_rows()
    load_error = None
except Exception as exc:
    rows, load_error = [], exc

try:
    pc_rows, pc_updated = D.read_patient_counts()
    pc_receipts = D.read_receipts()
    pc_error = None
except Exception as exc:
    pc_rows, pc_updated, pc_receipts, pc_error = [], None, [], exc

# HMIS counts. The environment switch (Patient counts tab) is read here, before the totals, so both tabs follow it:
# the chosen environment's facilities join the sources table and the totals (patients into UPDs).
try:
    hmis_envs = D.read_hmis_environments()
    # first visit: production if it has runs, else test; a click on the selected option (which clears it) keeps it
    last = st.session_state.get("_hmis_last")
    if st.session_state.get(hmis.SWITCH_KEY) not in hmis.CHOICES:
        default = hmis.pick_environment(hmis_envs)
        st.session_state[hmis.SWITCH_KEY] = last or (str(default).lower() if default else hmis.PRODUCTION)
    hmis_choice = st.session_state["_hmis_last"] = st.session_state[hmis.SWITCH_KEY]
    hmis_env = hmis.resolve(hmis_choice, hmis_envs)
    hmis_view = hmis.build(D.read_hmis(hmis_env), hmis_env, [r["SOURCE"] for r in pc_rows]) if hmis_env else None
    hmis_error = None
except Exception as exc:
    hmis_choice, hmis_view, hmis_error = hmis.PRODUCTION, None, exc
if pc_error is None:
    pc_rows = pc_rows + hmis.counted_rows(hmis_view)
upds = sum(r["UPDS"] for r in pc_rows) if pc_rows else None
patient_records = sum(r["PATIENT_RECORDS"] for r in pc_rows) if pc_rows else None

weeks = [(r["PERIOD"], r["WEEK_ENDING"]) for r in reversed(rows)]
goto = st.session_state.pop("x_goto_week", None)
if goto:
    st.session_state["x_week"] = next((k for k in weeks if k[1] == goto), weeks[0] if weeks else None)

with st.container(key="x_head"):
    c_brand, c_btn = st.columns([8, 0.9], vertical_alignment="center")
    c_brand.markdown(
        f'<div class="x-brand">{T.logo_tag(os.path.join(_HERE, "logo.png"))}<div>'
        f'<div class="x-title">Metrics Overview</div>'
        f'<div class="x-org">Afyanalytics Research Limited</div></div></div>',
        unsafe_allow_html=True)
    if c_btn.button("Update", icon=":material/add:", key="x_update", width="stretch",
                    disabled=load_error is not None):
        update.open_dialog(rows, patient_records)


def _unavailable(exc):
    T.html(f'<div class="x-empty"><b>Metrics are unavailable</b>Couldn\'t reach the metrics store '
           f'({exc.__class__.__name__}).</div>')


tab_overview, tab_patients, tab_discovery = st.tabs(["Overview", "Patient counts", "Discovery"])
with tab_overview:
    if load_error is not None:
        _unavailable(load_error)
    elif not weeks:
        T.html('<div class="x-empty"><b>No figures recorded yet</b>Use Update to add the first week.</div>')
    else:
        with st.container(key="x_ctx"):
            c_week, c_strip = st.columns([1.3, 3], vertical_alignment="bottom", gap="medium")
            sel = c_week.selectbox("Week ending", weeks, key="x_week", format_func=lambda k: f"{k[1]:%d %b %Y}")
            with c_strip:
                overview.strip(sel[0], sel[1])
        overview.render(rows, sel[0], sel[1], upds, patient_records)
with tab_patients:
    if pc_error is not None:
        _unavailable(pc_error)
    else:
        patients.render(pc_rows, pc_updated, pc_receipts, hmis_view, hmis_error, hmis_choice)
with tab_discovery:
    discovery.render()
