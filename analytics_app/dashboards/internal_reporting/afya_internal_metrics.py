"""Afyanalytics internal metrics dashboard: Overview and Patient counts tabs."""

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

from afya_metrics import data as D, overview, patients, theme as T, update

T.inject_css()

try:
    rows, _ = D.read_rows()
    load_error = None
except Exception as exc:
    rows, load_error = [], exc

try:
    pc_rows, pc_updated = D.read_patient_counts()
    pc_error = None
except Exception as exc:
    pc_rows, pc_updated, pc_error = [], None, exc
unique_patients = sum(r["UNIQUE_PATIENTS"] for r in pc_rows) if pc_rows else None
patient_records = sum(r["PATIENTS"] for r in pc_rows) if pc_rows else None

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


tab_overview, tab_patients = st.tabs(["Overview", "Patient counts"])
with tab_overview:
    if load_error is not None:
        _unavailable(load_error)
    elif not weeks:
        T.html('<div class="x-empty"><b>No figures recorded yet</b>Use Update to add the first week.</div>')
    else:
        with st.container(key="x_ctx"):
            c_strip, c_week = st.columns([5, 1.35], vertical_alignment="center", gap="medium")
            sel = c_week.selectbox("Week", weeks, key="x_week", label_visibility="collapsed",
                                   format_func=lambda k: f"Week ending {k[1]:%d %b %Y}")
            with c_strip:
                overview.strip(sel[0], sel[1])
        overview.render(rows, sel[0], sel[1], unique_patients, patient_records)
with tab_patients:
    if pc_error is not None:
        _unavailable(pc_error)
    else:
        patients.render(pc_rows, pc_updated)
