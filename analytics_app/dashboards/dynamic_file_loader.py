import streamlit as st
import os
import glob

# get dashboard from URL
params = st.query_params
dashboard = params.get("dashboard")

st.set_page_config(layout="wide")

if not dashboard:
    st.title("No dashboard selected")
else:
    pattern = f"analytics_app/dashboards/*/{dashboard}.py"

    matches = glob.glob(pattern)
    file_path = None
    if matches:
        file_path = matches[0]  # first match
    else:
        file_path = None
    print(file_path)
    if os.path.exists(file_path):
        with open(file_path) as f:
            code = f.read()

        # 🔥 dynamically run the dashboard
        exec(code, {"st": st, "__name__": "__main__"})
    else:
        st.error("Dashboard not found")