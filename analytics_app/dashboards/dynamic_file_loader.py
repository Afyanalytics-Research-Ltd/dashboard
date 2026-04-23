import streamlit as st
import os
import glob
import traceback
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
        
        try:
            compiled_code = compile(code, file_path, "exec")
            exec(compiled_code, {"st": st, "__name__": "__main__"})
        except Exception as e:
            st.error(f"Error: {e}")
            st.code(traceback.format_exc())
    else:
        st.error("Dashboard not found")