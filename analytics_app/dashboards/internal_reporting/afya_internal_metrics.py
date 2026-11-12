"""Afyanalytics internal metrics dashboard: Overview, Patient counts and Discovery tabs."""

import os
import sys
import time
from datetime import timedelta

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

# Automatic refresh: the HMIS pipeline posts every 15 minutes, so the whole page reruns every 15 minutes and picks up
# the latest run (the Snowflake reads are cached for 10, so they are fresh by then). It waits for up to two refreshes
# after the Update dialog was last used, since a rerun would close it and lose what was typed (Streamlit gives no
# signal when a dialog is dismissed; saving clears the wait).
REFRESH = int(os.environ.get("AFYA_METRICS_REFRESH", str(15 * 60)))      # seconds
st.session_state["x_full_run_at"] = time.time()


@st.fragment(run_every=timedelta(seconds=min(60, REFRESH)))     # a cheap check each minute; it draws nothing
def _auto_refresh():
    now = time.time()
    if now - st.session_state.get("x_full_run_at", now) < REFRESH:
        return                              # part of a full run, or one happened less than REFRESH ago
    if now - st.session_state.get(update.IN_USE, 0) < 2 * REFRESH:
        return                              # the Update dialog is in use: try again next time
    st.rerun(scope="app")


_auto_refresh()

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

# HMIS counts: every HMIS environment with runs is loaded; the Environment filter on the Patient counts tab picks
# which of Warehouse / Production / Migrate feed the table and the totals. It is read here, before the totals, so
# the Overview follows it too.
hmis_parts = {}                          # {environment label: (view, dated arrivals)}
try:
    hmis_envs = D.read_hmis_environments()
    for choice, label in hmis.CHOICES.items():
        env = hmis.resolve(choice, hmis_envs)
        view = hmis.build(D.read_hmis(env), env, [r["SOURCE"] for r in pc_rows]) if env else None
        if view:
            hmis_parts[label] = (view, hmis.receipts(view, D.read_hmis_daily(env)))
    hmis_error = None
except Exception as exc:
    hmis_parts, hmis_error = {}, exc
env_options = hmis.env_options(hmis_parts)
# first visit, or nothing (or something no longer available) selected: Warehouse and Production
if not set(st.session_state.get(hmis.ENV_KEY) or []) & set(env_options) or \
        not set(st.session_state.get(hmis.ENV_KEY) or []) <= set(env_options):
    st.session_state[hmis.ENV_KEY] = hmis.default_envs(hmis_parts)
env_selected = list(st.session_state[hmis.ENV_KEY])
env_overlap = []
if pc_error is None:
    pc_rows, pc_receipts, env_overlap = hmis.combine(pc_rows, pc_receipts, hmis_parts, env_selected)
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
        patients.render(pc_rows, pc_updated, pc_receipts, hmis_parts, hmis_error, env_options, env_overlap)
with tab_discovery:
    discovery.render()
