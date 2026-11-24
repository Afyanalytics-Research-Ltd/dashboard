import streamlit as st
import os
import glob
import traceback
# get dashboard (and optional sub-page) from the URL's query string.
# Streamlit has no native URL-path routing here — this script IS the
# actual entrypoint Streamlit was launched with (see
# docker-compose.streamlit.yaml), and every dashboard's own code is
# exec()'d inline as a string rather than run as its own Streamlit
# process, so a path segment like /opd is never seen by anything: only
# ?dashboard=... (and now ?page=...) query parameters are read.
params = st.query_params
dashboard = params.get("dashboard")
page = params.get("page")

st.set_page_config(layout="wide")


@st.cache_resource(show_spinner=False)
def _warehouse_seen() -> dict:
    return {}


def _follow_warehouse_switch():
    """System Settings -> Data warehouse: every dashboard caches one warehouse
    connection (st.cache_resource) and its query results (st.cache_data) for
    the life of this container. When the switch flips, drop both so the next
    query reconnects to the new warehouse (afya_backends.toggles asks Django
    over HTTP - see _http_source there)."""
    try:
        # The dashboards load .env themselves (load_dotenv in their connection helpers) - after this runs.
        # Load it first: the switch is read over HTTP with AFYA_INTERNAL_API_TOKEN from .env, and an answer
        # read without it (the deploy default) would be cached for the dashboard's first connection.
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))
        from afya_backends import toggles
        now = toggles.warehouse_backend()
    except Exception:  # noqa: BLE001 - a dashboard must still render
        return
    seen = _warehouse_seen()
    before = seen.get("backend")
    if before and before != now:
        st.cache_resource.clear()
        st.cache_data.clear()
        seen = _warehouse_seen()
    seen["backend"] = now


_follow_warehouse_switch()


def _run_file(file_path):
    with open(file_path, encoding='utf-8') as f:
        code = f.read()
    try:
        compiled_code = compile(code, file_path, "exec")
        exec(compiled_code, {
            "st": st,
            "__name__": "__main__",
            "__file__": os.path.abspath(file_path),
        })
    except Exception as e:
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())


if not dashboard:
    st.title("No dashboard selected")
else:
    matches = glob.glob(f"analytics_app/dashboards/*/{dashboard}.py")
    file_path = matches[0] if matches else None

    if file_path is None:
        st.error("Dashboard not found")
    elif page:
        # Classic Streamlit multi-page sub-scripts live in a `pages/`
        # folder nested under the dashboard's own package (e.g.
        # SPH/facility_operations/pages/1_opd.py) — NOT adjacent to this
        # loader script, so Streamlit's built-in pages/ auto-discovery can
        # never find them no matter how this app is launched. Each of
        # those page files is fully self-contained (own imports, own
        # st.set_page_config, own sidebar render) — confirmed by reading
        # them — so exec'ing the matched one directly, exactly like the
        # main dashboard file, is sufficient; nothing from the main
        # dashboard file needs to run first.
        dashboard_dir = os.path.dirname(file_path)
        page_matches = (
            glob.glob(os.path.join(dashboard_dir, "**", "pages", f"*_{page}.py"), recursive=True)
            or glob.glob(os.path.join(dashboard_dir, "**", "pages", f"{page}.py"), recursive=True)
        )
        if not page_matches:
            st.error(f"Page '{page}' not found for dashboard '{dashboard}'.")
        else:
            _run_file(page_matches[0])
    else:
        _run_file(file_path)
