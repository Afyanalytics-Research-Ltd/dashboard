"""
KSH V3 — Capacity and Operations Dashboard · Entry point
Registers all pages via st.navigation. Add new pages here as they are built.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="Capacity and Operations Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("ksh_operations/pages/0_Home.py",          title="Home",      url_path="",             default=True),
        st.Page("ksh_operations/pages/1_OPD_Flow.py",      title="OPD Flow",  url_path="1_OPD_Flow"),
        st.Page("ksh_operations/pages/2_Lab_TAT.py",       title="LAB",       url_path="2_Lab_TAT"),
        st.Page("ksh_operations/pages/3_Pharmacy_TAT.py",  title="Pharmacy",  url_path="3_Pharmacy_TAT"),
        st.Page("ksh_operations/pages/4_Inpatient.py",     title="Inpatient", url_path="4_Inpatient"),
        st.Page("ksh_operations/pages/5_Theatre.py",       title="Theatre",   url_path="5_Theatre"),
        st.Page("ksh_operations/pages/6_Clinician.py",     title="Clinician", url_path="6_Clinician"),
    ],
    position="hidden",
)
pg.run()
