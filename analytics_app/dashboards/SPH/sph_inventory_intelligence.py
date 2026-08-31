"""SPH Inventory Intelligence — dashboard entry point.

Launched by ``analytics_app/dashboards/dynamic_file_loader.py`` via
``?dashboard=sph_inventory_intelligence`` (the loader ``exec()``s this file with
``st`` injected and the repository root as the working directory).

The dashboard reads pre-computed analytics tables shipped as parquet under
``SPH/inventory_intelligence/output/`` and, where available, live Snowflake for
names, procurement and supplier data. Navigation is an ``option_menu`` sidebar;
each page is an existing script under ``dashboard/pages`` run in place.
"""
import base64
import os
import runpy
import sys
from pathlib import Path

# The package root — add it so ``inventory_intelligence.*`` imports resolve.
# ``__file__`` is unavailable under the loader's exec(), so anchor on the CWD
# (the repo root), matching the sibling KSH dashboard.
_ROOT = Path(os.path.abspath("analytics_app/dashboards/SPH"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st  # noqa: E402
from streamlit_option_menu import option_menu  # noqa: E402

from inventory_intelligence.dashboard import theme  # noqa: E402

theme.inject_css()
st.session_state.setdefault("facility", "SPH")

_PAGES_DIR = _ROOT / "inventory_intelligence" / "dashboard" / "pages"
_PAGES = {
    "Overview": ("overview.py", "speedometer2"),
    "Demand & availability": ("demand_availability.py", "graph-up-arrow"),
    "Movement & capital": ("movement_capital.py", "box-seam"),
    "Ordering plan": ("replenishment.py", "list-check"),
    "Spending & suppliers": ("procurement.py", "truck"),
    "Data quality": ("data_quality.py", "patch-check"),
}

_ACCENT = theme.accent()
_CH = theme.chrome()

_LOGO_B64 = base64.b64encode(
    (_ROOT / "inventory_intelligence" / "utils" / "logo.png").read_bytes()
).decode()

with st.sidebar:
    st.markdown(
        f'<div class="sph-brand">'
        f'<div class="sph-logo"><img src="data:image/png;base64,{_LOGO_B64}" '
        f'alt="St. Peter\'s Orthopedic and Surgical Speciality Center"></div>'
        f'<div class="sph-brand-sub">Inventory Intelligence</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    choice = option_menu(
        menu_title=None,
        options=list(_PAGES),
        icons=[icon for _, icon in _PAGES.values()],
        default_index=0,
        styles={
            "container": {"padding": "4px 0", "background-color": "transparent"},
            "icon": {"font-size": "16px"},
            "nav-link": {
                "font-size": "13.5px", "font-weight": "500", "padding": "8px 12px",
                "margin": "1px 0", "border-radius": "9px",
                "color": _CH["ink_secondary"],
            },
            "nav-link-selected": {
                "background-color": theme.rgba(_ACCENT, 0.13), "color": _ACCENT,
                "font-weight": "700",
            },
        },
    )
    st.markdown('<div class="sph-sidebar-gap"></div>', unsafe_allow_html=True)
    if st.button(
        "Email digest",
        icon=":material/mail:",
        use_container_width=True,
        help="Send today's stock-availability digest to the configured recipients.",
    ):
        with st.spinner("Sending digest…"):
            try:
                from inventory_intelligence.utils import notifier
                if notifier.send_daily_digest(force=True):
                    st.success("Digest sent.")
                else:
                    st.warning("No recipients configured — set NOTIFY_EMAIL_TO.")
            except Exception as exc:
                st.error(f"Couldn't send: {exc}")

runpy.run_path(str(_PAGES_DIR / _PAGES[choice][0]), run_name="__main__")
