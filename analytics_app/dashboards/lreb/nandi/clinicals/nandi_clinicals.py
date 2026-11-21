"""
Kapsabet CRH dashboard (Nandi / LREB). Entry point.
Run:  streamlit run nandi_clinicals.py          (live Snowflake)
      NANDI_DEMO=1 streamlit run nandi_clinicals.py   (synthetic data, no Snowflake)
Modules load under unique nandi_* names so generic folder names (queries,
template, views) never clash with other dashboards.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
# Make snowflake_service importable when the repo layout is intact (lreb/../snowflake_service).
for p in HERE.parents:
    if (p / "snowflake_service").exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

MODULES = [
    ("nandi_demo_data", "queries/demo_data.py"),
    ("nandi_queries", "queries/queries.py"),
    ("nandi_ui_template", "template/ui_template.py"),
    ("nandi_geo_map", "template/geo_map.py"),
    ("nandi_views_geo", "views/views_geo.py"),
    ("nandi_views_infectious", "views/views_infectious.py"),
    ("nandi_views_ncd", "views/views_ncd.py"),
    ("nandi_views_mental", "views/views_mental.py"),
    ("nandi_views_respiratory", "views/views_respiratory.py"),
    ("nandi_views_maternal", "views/views_maternal.py"),
    ("nandi_views_injury", "views/views_injury.py"),
    ("nandi_views_workload", "views/views_workload.py"),
    ("nandi_views_dataquality", "views/views_dataquality.py"),
    ("nandi_views_summary", "views/views_summary.py"),
    ("nandi_sidebar", "template/sidebar.py"),
]


_MTIME: dict[str, float] = {}


def _load(name: str, rel: str):
    """Import a file under a unique name. Re-imports when the file changed on disk, so edits show up
    on the next rerun without restarting Streamlit."""
    path = HERE / rel
    mt = path.stat().st_mtime
    if name in sys.modules and _MTIME.get(name) == mt:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    _MTIME[name] = mt
    spec.loader.exec_module(mod)
    return mod


st.set_page_config(page_title="Kapsabet CRH · Afyanalytics", layout="wide", initial_sidebar_state="expanded")
_stamp = max((HERE / r).stat().st_mtime for _, r in MODULES) + sum((HERE / "template" / f).stat().st_mtime for f in ("geo_engine.js", "leaflet.js"))
if st.session_state.get("_nandi_stamp") != _stamp:
    for _n, _ in MODULES:
        sys.modules.pop(_n, None)
    _MTIME.clear()
    st.session_state["_nandi_stamp"] = _stamp
mods = {n: _load(n, r) for n, r in MODULES}
ui, sidebar, geo_view = mods["nandi_ui_template"], mods["nandi_sidebar"], mods["nandi_views_geo"]
infectious_view = mods["nandi_views_infectious"]
ncd_view = mods["nandi_views_ncd"]
mental_view = mods["nandi_views_mental"]
resp_view = mods["nandi_views_respiratory"]
maternal_view = mods["nandi_views_maternal"]
injury_view = mods["nandi_views_injury"]
workload_view = mods["nandi_views_workload"]
dq_view = mods["nandi_views_dataquality"]
summary_view = mods["nandi_views_summary"]

ui.inject_css()
tab = sidebar.render()
if tab == "tab_summary":
    summary_view.render()
elif tab == "tab_geo":
    geo_view.render()
elif tab == "tab_infectious":
    infectious_view.render()
elif tab == "tab_ncds":
    ncd_view.render()
elif tab == "tab_mental":
    mental_view.render()
elif tab == "tab_resp":
    resp_view.render()
elif tab == "tab_maternal":
    maternal_view.render()
elif tab == "tab_injury":
    injury_view.render()
elif tab == "tab_workload":
    workload_view.render()
elif tab == "tab_dataquality":
    dq_view.render()
else:
    ui.placeholder_page(sidebar.LABELS[tab], "This view will be built on the same ui_template.")
sidebar.finish_refresh()
