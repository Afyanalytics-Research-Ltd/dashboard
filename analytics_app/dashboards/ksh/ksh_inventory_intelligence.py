"""KSH Inventory Intelligence — entry point.

Run directly:   streamlit run analytics_app/dashboards/ksh/ksh_inventory_intelligence.py
Via the dashboard loader: exec'd with __file__ set (slug = ksh_inventory_intelligence).

The dashboard logic lives in the inventory_intelligence_v3 package alongside this file;
this module is the thin, conventionally-named entry the loader discovers and runs.
"""
from __future__ import annotations

import base64
import sys
import traceback
from pathlib import Path

import streamlit as st

_KSH_DIR = str(Path(__file__).resolve().parent)              # .../dashboards/ksh  (package parent)
if _KSH_DIR not in sys.path:
    sys.path.insert(0, _KSH_DIR)

try:  # the dashboard loader may already have called set_page_config
    st.set_page_config(page_title="KSH Inventory Intelligence", page_icon="💊", layout="wide",
                       initial_sidebar_state="expanded")
except Exception:  # noqa: BLE001
    pass

from inventory_intelligence_v3 import context, data, ui  # noqa: E402
from inventory_intelligence_v3.tabs import ask, briefing, control, dead_stock, demand, orders, stockouts  # noqa: E402

TABS = {
    "Briefing": briefing, "Order Workbench": orders, "Stockout Watch": stockouts, "Demand Insights": demand,
    "Dead & Excess": dead_stock, "Inventory Control": control, "Ask the data": ask,
}
LOGO = Path(__file__).parent / "inventory_intelligence_v3" / "ksh_logo.png"
LOGO_SRC = f"data:image/png;base64,{base64.b64encode(LOGO.read_bytes()).decode()}" if LOGO.exists() else ""



def _sidebar(d: dict) -> data.Filters:
    stores = d["stores"][d["stores"]["is_active_store"]].sort_values("store_name")
    products = d["products"]
    sb = st.sidebar
    sb.markdown("### Filters")
    if sb.button("Reset filters", icon=":material/restart_alt:", use_container_width=True):
        for k in list(st.session_state):
            if k.startswith("flt_"):
                del st.session_state[k]
    store_ids = sb.multiselect("Store", stores["store_id"].tolist(), key="flt_stores",
                               format_func=dict(zip(stores["store_id"], stores["store_name"])).get,
                               placeholder="All active stores")
    classes = sb.multiselect("Product class", [c for c in ui.C.CLASS_ORDER if c in set(products["product_class"])],
                             key="flt_classes", placeholder="All classes")
    pool = products[products["product_class"].isin(classes)] if classes else products
    ther = sorted(c for c in pool["therapeutic_class"].dropna().unique() if c != "Unclassified")
    ther_sel = sb.multiselect("Therapeutic class", ther, key="flt_ther", placeholder="All therapeutic classes")
    tiers = sb.multiselect("Clinical priority", ["CRITICAL", "HIGH", "STANDARD"], key="flt_tiers",
                           placeholder="All tiers",
                           help="CRITICAL: opioids, carbapenems, glycopeptides, oxazolidinones, uterotonics, "
                                "antiretrovirals. HIGH: cardiovascular, endocrine, antimicrobials, "
                                "neurology & psychiatry, oncology.")
    search = sb.text_input("Find a product", key="flt_search", placeholder="brand or generic, e.g. paracetamol")
    stream = sb.radio("Demand view", ["All consumption", "Dispensing only", "Ward use only"], key="flt_stream",
                      help="Applies to the Demand Insights tab only. 'Dispensing only' matches the previous dashboard. "
                           "Stock levels, orders and stockouts always count all usage.")
    return data.Filters(stores=store_ids, classes=classes, ther_classes=ther_sel, tiers=tiers,
                        search=search.strip(), stream=stream)


def main() -> None:
    ui.inject_css()
    with st.spinner("Loading inventory data…"):
        try:
            d = data.load_all()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load data from Snowflake: {str(exc).splitlines()[0][:300]}")
            return
    f = _sidebar(d)
    as_of = data.as_of(d)
    facility = d["stores"]["facility_name"].dropna().iloc[0] if d["stores"]["facility_name"].notna().any() else ""

    logo = f'<img class="ksh-logo" src="{LOGO_SRC}" alt="{ui.esc(facility)}"><div class="ksh-rule"></div>'         if LOGO_SRC else ""
    sub = "Pharmacy, wards, theatre and stores" if LOGO_SRC else f"{ui.esc(facility)} · pharmacy, wards, theatre and stores"
    st.markdown(
        f'<div class="ksh-header"><div class="ksh-brand">{logo}<div><div class="ksh-title">Inventory Intelligence'
        f'</div><div class="ksh-sub">{sub}</div></div></div>'
        f'<div class="ksh-asof">Stock as of {as_of:%d %b %Y} · usage to {data.usage_end(d):%d %b}</div></div>',
        unsafe_allow_html=True)

    names = list(TABS)
    current = st.query_params.get("tab", names[0])
    tab = st.segmented_control("View", names, default=current if current in names else names[0],
                               key="nav", label_visibility="collapsed") or names[0]
    st.query_params["tab"] = tab
    ui.chips(f.describe(dict(zip(d["stores"]["store_id"], d["stores"]["store_name"]))))

    key = (tuple(f.stores), tuple(f.classes), tuple(f.ther_classes), tuple(f.tiers), f.search, f.stream)
    with st.spinner("Analysing…"):
        ctx = context.build(key)
    try:
        TABS[tab].render(ctx)
    except Exception:  # noqa: BLE001  — one tab failing must not take the dashboard down
        st.error(f"The {tab} view hit an error. Other views still work.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


main()
