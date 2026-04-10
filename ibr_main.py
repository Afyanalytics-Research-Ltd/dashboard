"""
app.py — PharmaPlus IBR Engine
Dead Stock Recovery · Inter-Branch Rebalancing · Markdown Intelligence
"""

import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

from data.loader import build_multi_branch_data, BRANCH_NAMES
from engine.dead_stock import flag_dead_stock, dead_stock_summary
from engine.velocity_scorer import compute_velocity, compute_trend
from engine.ibr_recommender import build_recommendations, recommendations_summary
from engine.predictive import flag_proactive_transfers
from engine.price_signal import load_competitor_data, build_price_signal

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PharmaPlus · IBR Engine",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Montserrat',sans-serif;background:#fff;color:#003467}
.stApp{background:#fff}
[data-testid="stSidebar"]{background:#F4F8FC;border-right:1px solid #D6E4F0}
[data-testid="stSidebar"] *{color:#003467!important;font-family:'Montserrat',sans-serif!important}
.sh{font-size:10px;font-weight:800;color:#0072CE;text-transform:uppercase;
    letter-spacing:2.5px;padding:8px 0;border-bottom:2px solid #EBF3FB;margin-bottom:16px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;font-weight:700}
.badge-DEAD{background:#FFF1F2;color:#E11D48;border:1px solid #FDA4AF}
.badge-ALERT{background:#FFFBEB;color:#D97706;border:1px solid #FCD34D}
.badge-WATCH{background:#EBF5FF;color:#0072CE;border:1px solid #93C5FD}
.stButton button{background:#0072CE!important;color:#fff!important;border:none!important;
  font-family:'Montserrat',sans-serif!important;font-size:11px!important;font-weight:700!important;
  letter-spacing:1px!important;padding:8px 18px!important;border-radius:6px!important}
.stButton button:hover{background:#003467!important}
[data-baseweb="tab"]{font-family:'Montserrat',sans-serif!important;font-weight:600!important;
  color:#6B8CAE!important;font-size:12px!important}
[aria-selected="true"]{color:#0072CE!important;border-bottom-color:#0072CE!important}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-thumb{background:#B0C8E0;border-radius:10px}
[data-testid="stSidebarCollapseButton"] *,
.material-symbols-rounded{font-family:'Material Symbols Rounded',sans-serif!important}
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def fmt_ksh(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"KSh {v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"KSh {v/1_000:.1f}K"
    return f"KSh {v:.0f}"

CAT_COLORS = {
    "Pharma":                 "#0072CE",
    "Beauty & Cosmetics":     "#D4537E",
    "Vitamins & Supplements": "#1D9E75",
    "Body Building":          "#D85A30",
    "Non-Pharma":             "#7F77DD",
}

REC_CONFIG = {
    "TRANSFER": {"color": "#0BB99F", "label": "Transfer",      "icon": "⇄"},
    "MARKDOWN": {"color": "#D97706", "label": "Mark down",     "icon": "▼"},
    "BUNDLE":   {"color": "#0072CE", "label": "Split transfer","icon": "⇉"},
    "REVIEW":   {"color": "#888780", "label": "Flag for review","icon": "⚑"},
}

def _kpi(label, value, sub, color="#003467"):
    return (
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;padding:18px 16px">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:8px">{label}</div>'
        f'<div style="font-size:28px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:6px">{sub}</div>'
        f'</div>'
    )

def _rec_badge(rec_type):
    cfg = REC_CONFIG.get(rec_type, REC_CONFIG["REVIEW"])
    c   = cfg["color"]
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 8px;'
        f'border-radius:4px;font-size:10px;font-weight:700;color:{c};'
        f'background:{c}15;border:1px solid {c}44">'
        f'{cfg["icon"]} {cfg["label"]}</span>'
    )

def _cat_html(cat):
    c = CAT_COLORS.get(cat, "#888780")
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
        f'font-size:10px;font-weight:700;color:{c};background:{c}15;'
        f'border:1px solid {c}33">{cat}</span>'
    )

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

PHARMAPLUS_CSV = "Pricing and Product Listings/pharmaplus_product_list.csv"
MYDAWA_CSV     = "Pricing and Product Listings/mydawa_product_list.csv"
GOODLIFE_CSV   = "Pricing and Product Listings/goodlife_price_list.csv"
LINTON_CSV     = "Pricing and Product Listings/linton.csv"

with st.sidebar:
    try:
        st.image("assets/pharmaplus_logo.png", width=160)
    except:
        st.markdown(
            '<div style="font-size:16px;font-weight:800;color:#0072CE;padding:8px 0 16px">PharmaPlus</div>',
            unsafe_allow_html=True,
        )

    build_btn = st.button("Run Analysis", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="sh">Filters</div>', unsafe_allow_html=True)
    branch_filter = st.multiselect(
        "Branches", options=list(BRANCH_NAMES.values()),
        default=list(BRANCH_NAMES.values()),
    )
    tier_filter = st.multiselect(
        "Tier", options=["DEAD", "ALERT", "WATCH"],
        default=["DEAD", "ALERT", "WATCH"],
    )
    cat_filter = st.multiselect(
        "Category",
        options=["Pharma","Beauty & Cosmetics","Vitamins & Supplements","Body Building","Non-Pharma"],
        default=["Pharma","Beauty & Cosmetics","Vitamins & Supplements","Body Building","Non-Pharma"],
    )
    rec_type_filter = st.multiselect(
        "Action type",
        options=["TRANSFER","MARKDOWN","BUNDLE","REVIEW"],
        default=["TRANSFER","MARKDOWN","BUNDLE"],
    )
    min_ksh = st.slider("Min KSh at risk", 0, 100_000, 500, 500, format="KSh %d")

    st.markdown("---")
    st.markdown('<div class="sh">IBR settings</div>', unsafe_allow_html=True)
    min_velocity = st.slider("Min destination velocity", 0, 80, 40, 5)


# ─── SESSION STATE ────────────────────────────────────────────────────────────

_state_keys = {
    "data_loaded":        False,
    "approved_transfers": set(),
    "approved_markdowns": set(),
    "dispensing":         pd.DataFrame(),
    "inventory":          pd.DataFrame(),
    "products":           pd.DataFrame(),
    "dead_stock":         pd.DataFrame(),
    "velocity":           pd.DataFrame(),
    "trend":              pd.DataFrame(),
    "proactive":          pd.DataFrame(),
    "match_table":        pd.DataFrame(),
    "price_signal_df":    pd.DataFrame(),
    "recs":               pd.DataFrame(),
}
for k, v in _state_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── SIMULATION ──────────────────────────────────────────────────────────────

def run_simulation():
    disp, inv, products, match_table = build_multi_branch_data(
        pharmaplus_csv_path=PHARMAPLUS_CSV,
        goodlife_csv_path=GOODLIFE_CSV if os.path.exists(GOODLIFE_CSV) else None,
        linton_csv_path=LINTON_CSV     if os.path.exists(LINTON_CSV)   else None,
    )
    if disp.empty:
        return False

    dead_stock = flag_dead_stock(disp, inv, products)
    velocity   = compute_velocity(disp)
    trend      = compute_trend(disp)
    proactive  = flag_proactive_transfers(
        trend, velocity,
        dead_stock_product_ids=set(dead_stock["product_id"].unique()),
        inventory=inv,
    )

    competitor_sources = load_competitor_data(
        mydawa_path=MYDAWA_CSV   if os.path.exists(MYDAWA_CSV)   else None,
        goodlife_path=GOODLIFE_CSV if os.path.exists(GOODLIFE_CSV) else None,
        linton_path=LINTON_CSV   if os.path.exists(LINTON_CSV)   else None,
    )
    price_signal_df = build_price_signal(match_table, competitor_sources, dead_stock=dead_stock)

    recs = build_recommendations(
        dead_stock, velocity, trend,
        price_signal=price_signal_df if not price_signal_df.empty else None,
        min_velocity_score=min_velocity,
    )

    st.session_state.update({
        "dispensing": disp, "inventory": inv, "products": products,
        "dead_stock": dead_stock, "velocity": velocity, "trend": trend,
        "proactive": proactive, "match_table": match_table,
        "price_signal_df": price_signal_df, "recs": recs,
        "data_loaded": True,
    })
    return True


if build_btn:
    if run_simulation():
        st.sidebar.success("Analysis complete.")

if not st.session_state.get("data_loaded", False):
    st.markdown(
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'justify-content:center;height:65vh;gap:14px">'
        '<div style="font-size:52px;color:#0072CE">⚕</div>'
        '<div style="font-size:13px;color:#003467;font-weight:800;'
        'letter-spacing:4px;text-transform:uppercase">IBR Engine</div>'
        '<div style="font-size:12px;color:#6B8CAE;text-align:center;max-width:320px">'
        'Click <b>Run Analysis</b> in the sidebar to build the simulation and generate recommendations.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ─── FILTERED DATA ────────────────────────────────────────────────────────────

dead_stock = st.session_state.dead_stock.copy()
dead_stock = dead_stock[~dead_stock["product_name"].isin(
    ["PLUSMED TEST STRIPS","GLOVES LATEX","JIK 5 LITRES","ISOFLURANE"]
)]
if "internal_category" not in dead_stock.columns:
    dead_stock["internal_category"] = "Pharma"
dead_stock["internal_category"] = dead_stock["internal_category"].fillna("Pharma")

branch_ids = [k for k, v in BRANCH_NAMES.items() if v in branch_filter]
dead_stock = dead_stock[dead_stock["store_id"].isin(branch_ids)]
dead_stock = dead_stock[dead_stock["tier"].isin(tier_filter)]
dead_stock = dead_stock[dead_stock["internal_category"].isin(cat_filter)]
dead_stock = dead_stock[dead_stock["ksh_at_risk"] >= min_ksh]

recs = st.session_state.recs.copy()
if not recs.empty:
    recs = recs[recs["source_store_id"].isin(branch_ids)]
    recs = recs[recs["tier"].isin(tier_filter)]
    recs = recs[recs["internal_category"].isin(cat_filter)]
    recs = recs[recs["recommendation_type"].isin(rec_type_filter)]
    recs = recs[recs["ksh_at_risk"] >= min_ksh]

price_sig_df = st.session_state.price_signal_df
proactive    = st.session_state.proactive

ds_sum  = dead_stock_summary(dead_stock)
rec_sum = recommendations_summary(recs) if not recs.empty else {
    "total_recoverable_ksh": 0, "transfer_ksh": 0, "transfer_count": 0,
    "markdown_ksh": 0, "markdown_count": 0,
    "bundle_ksh": 0, "bundle_count": 0, "review_count": 0,
}


# ─── HEADER ──────────────────────────────────────────────────────────────────

st.markdown(
    '<p style="font-size:11px;font-weight:800;letter-spacing:3px;'
    'text-transform:uppercase;color:#0072CE;margin-bottom:16px">PharmaPlus · Dead Stock Recovery & Inter Branch Rebalancing Engine</p>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(_kpi("Frozen capital", fmt_ksh(ds_sum["total_frozen_ksh"]),
        f"{ds_sum['total_skus']} SKUs · {ds_sum['branches_affected']} branches", "#E11D48"),
        unsafe_allow_html=True)
with k2:
    transfer_sub = f"{rec_sum['transfer_count']} transfers · {rec_sum['markdown_count']} markdowns"
    st.markdown(_kpi("Est. recoverable", fmt_ksh(rec_sum["total_recoverable_ksh"]),
        transfer_sub, "#0BB99F"), unsafe_allow_html=True)
with k3:
    dead_count  = ds_sum["by_tier"].get("DEAD",  {}).get("sku_count", 0)
    alert_count = ds_sum["by_tier"].get("ALERT", {}).get("sku_count", 0)
    st.markdown(_kpi("Needs action now", str(dead_count),
        f"DEAD · + {alert_count} on ALERT", "#E11D48"), unsafe_allow_html=True)
with k4:
    pro_count = len(proactive) if not proactive.empty else 0
    st.markdown(_kpi("Early warnings", str(pro_count),
        "decelerating · act before threshold", "#D97706"), unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)


# ─── TABS ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "⇄  Actions",
    "△  Early Warnings",
    "∑  Analytics",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if recs.empty:
        st.info("No recommendations with current filters.")
    else:
        display_recs = recs.head(50)

        st.markdown(
            f'<div class="sh">{len(display_recs)} recommendations · '
            f'ranked by estimated recovery · approve to add to waybill</div>',
            unsafe_allow_html=True,
        )

        for _, rec in display_recs.iterrows():
            rec_type  = str(rec["recommendation_type"])
            cfg       = REC_CONFIG.get(rec_type, REC_CONFIG["REVIEW"])
            pid       = rec["product_id"]
            src_id    = int(rec["source_store_id"])
            dest_id   = int(rec.get("dest_store_id", src_id))
            item_key  = f"{src_id}_{str(pid)}_{dest_id}"
            approved  = item_key in st.session_state.approved_transfers

            tier      = str(rec["tier"])
            cat       = str(rec.get("internal_category", "—"))
            recovery  = float(rec["estimated_recovery_ksh"])
            ksh_risk  = float(rec["ksh_at_risk"])
            rationale = str(rec["recovery_rationale"])
            days_frz  = int(rec["days_frozen"])

            border_col = cfg["color"]

            # Pre-build fragments
            cat_span  = _cat_html(cat)
            tier_span = f'<span class="badge badge-{tier}">{tier}</span>'
            rec_span  = _rec_badge(rec_type)
            rec_color = cfg["color"]

            # Action-specific detail block
            if rec_type == "TRANSFER" or rec_type == "BUNDLE":
                dest_name  = str(rec.get("dest_branch_name","—")).replace("PharmaPlus ","")
                from_name  = str(rec["source_branch_name"]).replace("PharmaPlus ","")
                t_cost     = float(rec.get("transit_cost_ksh", 0))
                t_days     = int(rec.get("transit_days", 0))
                shelf      = str(rec.get("shelf_viability","—"))
                trend_lbl  = str(rec.get("dest_trend_label","—"))
                vel_score  = float(rec.get("dest_velocity_score", 0))
                t_icon     = {"accelerating":"▲","stable":"→"}.get(trend_lbl,"→")
                t_col      = "#0BB99F" if trend_lbl == "accelerating" else "#6B8CAE"
                shelf_col  = "#059669" if shelf == "safe" else "#D97706"

                detail_html = (
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                    f'gap:10px;margin-top:12px">'
                    f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:.8px">From</div>'
                    f'<div style="font-size:12px;color:#003467;font-weight:600;margin-top:2px">'
                    f'{from_name}</div></div>'
                    f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:.8px">To</div>'
                    f'<div style="font-size:12px;color:#003467;font-weight:600;margin-top:2px">'
                    f'{dest_name}</div></div>'
                    f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:.8px">Transit</div>'
                    f'<div style="font-size:12px;color:#003467;margin-top:2px">'
                    f'{fmt_ksh(t_cost)} · {t_days}d</div></div>'
                    f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:.8px">Dest demand</div>'
                    f'<div style="font-size:12px;margin-top:2px;color:{t_col};font-weight:600">'
                    f'{t_icon} {trend_lbl} · {vel_score:.0f}/100</div></div>'
                    f'</div>'
                    f'<div style="margin-top:8px;font-size:11px;color:#6B8CAE">'
                    f'Shelf: <span style="color:{shelf_col};font-weight:600">{shelf}</span>'
                    f'</div>'
                )

            elif rec_type == "MARKDOWN":
                target     = float(rec.get("markdown_target_price") or 0)
                pct_above  = float(rec.get("pct_above_market", 0))
                competitor = str(rec.get("primary_competitor", ""))
                from_name  = str(rec["source_branch_name"]).replace("PharmaPlus ", "")
                match_meth = str(rec.get("match_method", ""))
                conf_label = "Direct match" if match_meth == "sku_match" else "Est."
                tier_r     = str(rec["tier"])
                urgency_col = {"DEAD": "#E11D48", "ALERT": "#D97706"}.get(tier_r, "#6B8CAE")

                detail_html = (
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;'
                    f'gap:10px;margin-top:12px">'
                    f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:.8px">Branch</div>'
                    f'<div style="font-size:12px;color:#003467;font-weight:600;margin-top:2px">'
                    f'{from_name}</div></div>'
                    f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:.8px">Suggested price</div>'
                    f'<div style="font-size:16px;color:#D97706;font-weight:800;margin-top:2px">'
                    f'{fmt_ksh(target)}</div>'
                    f'<div style="font-size:10px;color:#6B8CAE;margin-top:1px">'
                    f'{conf_label} · {competitor}</div></div>'
                    f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase;'
                    f'letter-spacing:.8px">Above market</div>'
                    f'<div style="font-size:14px;color:{urgency_col};font-weight:700;margin-top:2px">'
                    f'{pct_above:.1f}%</div></div>'
                    f'</div>'
                )

            else:  # REVIEW
                from_name  = str(rec["source_branch_name"]).replace("PharmaPlus ","")
                detail_html = (
                    f'<div style="margin-top:12px;padding:10px 12px;background:#F4F8FC;'
                    f'border-radius:6px;font-size:12px;color:#6B8CAE">'
                    f'No branch has sufficient demand · no price signal available · '
                    f'consider supplier return or promotional bundle with a fast-moving SKU'
                    f'</div>'
                )

            # Rationale strip
            rationale_html = (
                f'<div style="margin-top:10px;padding:8px 10px;background:#F4F8FC;'
                f'border-radius:4px;font-size:11px;color:#6B8CAE;border-left:3px solid {rec_color}">'
                f'{rationale}</div>'
            )

            recovery_pct = round((recovery / ksh_risk * 100)) if ksh_risk > 0 else 0

            st.markdown(
                f'<div style="background:#fff;border:1px solid #D6E4F0;'
                f'border-left:4px solid {border_col};border-radius:0;'
                f'padding:14px 18px;margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'<div style="flex:1">'
                f'<div style="font-size:13px;font-weight:700;color:#003467;margin-bottom:6px">'
                f'{str(rec["product_name"])[:60]}</div>'
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">'
                f'{cat_span}{tier_span}'
                f'<span style="color:#6B8CAE;font-size:11px">{days_frz}d frozen</span>'
                f'{rec_span}'
                f'</div>'
                f'</div>'
                f'<div style="text-align:right;margin-left:16px">'
                f'<div style="font-size:18px;font-weight:800;color:{rec_color}">'
                f'{fmt_ksh(recovery)}</div>'
                f'<div style="font-size:10px;color:#6B8CAE">est. recovery · {recovery_pct}% of at-risk</div>'
                f'<div style="font-size:10px;color:#B0C8E0;margin-top:2px">'
                f'at risk: {fmt_ksh(ksh_risk)}</div>'
                f'</div>'
                f'</div>'
                f'{detail_html}'
                f'{rationale_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Approve / Undo — only for TRANSFER and BUNDLE
            if rec_type in ("TRANSFER", "BUNDLE"):
                col_a, col_b, _ = st.columns([1, 1, 6])
                with col_a:
                    if not approved:
                        if st.button("Approve", key=f"app_{item_key}"):
                            st.session_state.approved_transfers.add(item_key)
                            st.rerun()
                    else:
                        st.markdown(
                            '<div style="font-size:11px;color:#0BB99F;padding:8px 0;'
                            'font-weight:700;letter-spacing:1px">✓ APPROVED</div>',
                            unsafe_allow_html=True,
                        )
                with col_b:
                    if approved:
                        if st.button("Undo", key=f"undo_{item_key}"):
                            st.session_state.approved_transfers.discard(item_key)
                            st.rerun()
            elif rec_type == "MARKDOWN":
                md_approved = item_key in st.session_state.approved_markdowns
                col_a, col_b, _ = st.columns([1, 1, 6])
                with col_a:
                    if not md_approved:
                        if st.button("Approve", key=f"mdapp_{item_key}"):
                            st.session_state.approved_markdowns.add(item_key)
                            st.rerun()
                    else:
                        st.markdown(
                            '<div style="font-size:11px;color:#D97706;padding:8px 0;'
                            'font-weight:700;letter-spacing:1px">✓ QUEUED</div>',
                            unsafe_allow_html=True,
                        )
                with col_b:
                    if md_approved:
                        if st.button("Undo", key=f"mdundo_{item_key}"):
                            st.session_state.approved_markdowns.discard(item_key)
                            st.rerun()

            st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

        # ── WAYBILL ───────────────────────────────────────────────────────────
        if st.session_state.approved_transfers and not recs.empty:
            transfer_recs = recs[recs["recommendation_type"].isin(["TRANSFER","BUNDLE"])].copy()

            def _make_key(row):
                return f"{int(row['source_store_id'])}_{str(row['product_id'])}_{int(row['dest_store_id'])}"

            approved_recs = transfer_recs[
                transfer_recs.apply(_make_key, axis=1).isin(st.session_state.approved_transfers)
            ].copy()

            if not approved_recs.empty:
                st.markdown(
                    '<div class="sh" style="margin-top:32px">Consolidated waybills</div>',
                    unsafe_allow_html=True,
                )
                routes = approved_recs.groupby(["source_branch_name","dest_branch_name"])
                total_gross, total_transit, total_recovery = 0.0, 0.0, 0.0

                for (src, dest), rdf in routes:
                    rg = float(rdf["ksh_at_risk"].sum())
                    rt = float(rdf["transit_cost_ksh"].max())
                    rv = float(rdf["estimated_recovery_ksh"].sum())
                    total_gross    += rg
                    total_transit  += rt
                    total_recovery += rv
                    st.markdown(
                        f'<div style="background:#F4F8FC;border:1px dashed #B0C8E0;'
                        f'border-radius:6px;padding:14px;margin-bottom:10px">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:10px">'
                        f'<span style="font-size:12px;font-weight:700;color:#003467">'
                        f'{src.replace("PharmaPlus ","")} '
                        f'<span style="color:#0072CE">→</span> '
                        f'{dest.replace("PharmaPlus ","")}</span>'
                        f'<span style="font-size:11px;color:#6B8CAE">{len(rdf)} SKUs</span>'
                        f'</div>'
                        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px">'
                        f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase">'
                        f'Stock value</div>'
                        f'<div style="font-size:13px;font-weight:600;color:#003467">'
                        f'{fmt_ksh(rg)}</div></div>'
                        f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase">'
                        f'Transit cost</div>'
                        f'<div style="font-size:13px;font-weight:600;color:#E11D48">'
                        f'- {fmt_ksh(rt)}</div></div>'
                        f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase">'
                        f'Est. recovery</div>'
                        f'<div style="font-size:13px;font-weight:700;color:#0BB99F">'
                        f'{fmt_ksh(rv)}</div></div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f'<div style="border:1.5px solid #0BB99F;border-radius:8px;padding:16px 20px;margin-bottom:16px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div>'
                    f'<div style="font-size:12px;font-weight:800;color:#0BB99F;letter-spacing:1px">'
                    f'✓ WAYBILLS READY</div>'
                    f'<div style="font-size:11px;color:#6B8CAE;margin-top:3px">'
                    f'{len(approved_recs)} SKUs · {len(routes)} shipments · '
                    f'total transit: {fmt_ksh(total_transit)}</div>'
                    f'</div>'
                    f'<div style="text-align:right">'
                    f'<div style="font-size:22px;font-weight:800;color:#0BB99F">'
                    f'{fmt_ksh(total_recovery)}</div>'
                    f'<div style="font-size:11px;color:#6B8CAE">estimated recovery</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )

                # Export
                export_rows = []
                for _, row in approved_recs.iterrows():
                    export_rows.append({
                        "From Branch":        row["source_branch_name"],
                        "To Branch":          row["dest_branch_name"],
                        "Product":            row["product_name"],
                        "Category":           row.get("internal_category","—"),
                        "Tier":               row["tier"],
                        "Days Frozen":        int(row["days_frozen"]),
                        "Qty":                int(row.get("qty_on_hand", 0)),
                        "Stock Value (KSh)":  round(float(row["ksh_at_risk"]),2),
                        "Transit Cost (KSh)": round(float(row["transit_cost_ksh"]),2),
                        "Transit Days":       int(row["transit_days"]),
                        "Est. Recovery (KSh)":round(float(row["estimated_recovery_ksh"]),2),
                        "Shelf Viability":    str(row.get("shelf_viability","—")).replace("_"," ").title(),
                        "Dest Demand Trend":  str(row.get("dest_trend_label","—")).title(),
                        "Days to Clear":      row.get("days_to_clear_at_dest") or "—",
                    })

                st.download_button(
                    "⬇  Export Waybill (Excel-ready CSV)",
                    data=pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
                    file_name="pharmaplus_waybill.csv",
                    mime="text/csv",
                )

        # ── MARKDOWN INSTRUCTION SHEET ────────────────────────────────────────
        if st.session_state.approved_markdowns and not recs.empty:
            md_recs = recs[recs["recommendation_type"] == "MARKDOWN"].copy()

            def _make_md_key(row):
                return f"{int(row['source_store_id'])}_{str(row['product_id'])}_{int(row['source_store_id'])}"

            approved_mds = md_recs[
                md_recs.apply(_make_md_key, axis=1).isin(st.session_state.approved_markdowns)
            ].copy()

            if not approved_mds.empty:
                st.markdown(
                    '<div class="sh" style="margin-top:24px">Markdown instruction sheet</div>',
                    unsafe_allow_html=True,
                )

                # Summary card
                total_at_risk = float(approved_mds["ksh_at_risk"].sum())
                total_recovery = float(approved_mds["estimated_recovery_ksh"].sum())
                st.markdown(
                    f'<div style="border:1.5px solid #D97706;border-radius:8px;'
                    f'padding:16px 20px;margin-bottom:16px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center">'
                    f'<div>'
                    f'<div style="font-size:12px;font-weight:800;color:#D97706;letter-spacing:1px">'
                    f'▼ MARKDOWN SHEET READY</div>'
                    f'<div style="font-size:11px;color:#6B8CAE;margin-top:3px">'
                    f'{len(approved_mds)} SKUs · '
                    f'{approved_mds["source_branch_name"].nunique()} branches</div>'
                    f'</div>'
                    f'<div style="text-align:right">'
                    f'<div style="font-size:20px;font-weight:800;color:#D97706">'
                    f'{fmt_ksh(total_recovery)}</div>'
                    f'<div style="font-size:11px;color:#6B8CAE">est. recovery from {fmt_ksh(total_at_risk)} at risk</div>'
                    f'</div></div></div>',
                    unsafe_allow_html=True,
                )

                md_export = []
                for _, row in approved_mds.iterrows():
                    md_export.append({
                        "Branch":              row["source_branch_name"],
                        "Product":             row["product_name"],
                        "Category":            row.get("internal_category", "—"),
                        "Tier":                row["tier"],
                        "Days Frozen":         int(row["days_frozen"]),
                        "Current Price (KSh)": round(float(row.get("pct_above_market", 0) and
                            row.get("markdown_target_price", 0) / (1 - row.get("pct_above_market", 0)/100)
                            if row.get("markdown_target_price") else 0), 2),
                        "Suggested Price (KSh)": round(float(row.get("markdown_target_price") or 0), 2),
                        "% Above Market":      f"{float(row.get('pct_above_market', 0)):.1f}%",
                        "Competitor Source":   row.get("primary_competitor", "—"),
                        "Est. Recovery (KSh)": round(float(row["estimated_recovery_ksh"]), 2),
                        "Stock Value (KSh)":   round(float(row["ksh_at_risk"]), 2),
                        "Action":              f"Update POS price to KSh {row.get('markdown_target_price', 0):,.0f} and place shelf label",
                    })

                st.download_button(
                    "⬇  Export Markdown Sheet (Excel-ready CSV)",
                    data=pd.DataFrame(md_export).to_csv(index=False).encode("utf-8"),
                    file_name="pharmaplus_markdown_instructions.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EARLY WARNINGS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div class="sh">Early warnings — high-value SKUs decelerating before dead stock threshold</div>',
        unsafe_allow_html=True,
    )
    if proactive is None or proactive.empty:
        st.info("No early warnings. All high-value SKUs are selling at expected velocity.")
    else:
        prod_map = st.session_state.products.set_index("product_id")["product_name"].to_dict()
        pro      = proactive.copy()
        pro["product_name"] = pro["product_id"].map(prod_map).fillna("Unknown")

        st.caption(
            f"{len(pro)} high-value SKUs showing deceleration · "
            f"act now to avoid dead stock threshold"
        )

        for _, flag in pro.iterrows():
            uc         = "#E11D48" if flag["urgency"] == "high" else "#D97706"
            ubg        = "#FFF8F8" if flag["urgency"] == "high" else "#FFFBEB"
            src_short  = str(flag["source_branch_name"]).replace("PharmaPlus ","")
            dest_short = str(flag["dest_branch_name"]).replace("PharmaPlus ","")
            src_trend  = float(flag["source_trend_pct"])
            dest_trend = float(flag["dest_trend_pct"])
            src_vel    = float(flag["source_velocity_score"])
            dest_vel   = float(flag["dest_velocity_score"])
            inv_ksh    = float(flag.get("inv_ksh", 0))

            st.markdown(
                f'<div style="background:{ubg};border:1px solid {uc}33;'
                f'border-left:4px solid {uc};border-radius:0;'
                f'padding:14px 16px;margin-bottom:8px">'
                f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                f'<div>'
                f'<div style="font-size:13px;font-weight:700;color:#003467">'
                f'{str(flag["product_name"])[:55]}</div>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">'
                f'Slowing at <b style="color:#003467">{src_short}</b>'
                f' · growing at <b style="color:#0BB99F">{dest_short}</b></div>'
                f'</div>'
                f'<div style="text-align:right">'
                f'<span style="font-size:10px;color:{uc};font-weight:800;'
                f'text-transform:uppercase;letter-spacing:1px">{flag["urgency"]} urgency</span>'
                f'<div style="font-size:11px;color:#6B8CAE;margin-top:3px">'
                f'{fmt_ksh(inv_ksh)} at risk</div>'
                f'</div>'
                f'</div>'
                f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                f'gap:10px;margin-top:12px">'
                f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase">'
                f'Source trend</div>'
                f'<div style="font-size:13px;font-weight:700;color:#E11D48;margin-top:2px">'
                f'▼ {src_trend:+.1f}%</div></div>'
                f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase">'
                f'Dest trend</div>'
                f'<div style="font-size:13px;font-weight:700;color:#0BB99F;margin-top:2px">'
                f'▲ {dest_trend:+.1f}%</div></div>'
                f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase">'
                f'Source velocity</div>'
                f'<div style="font-size:13px;font-weight:600;color:#003467;margin-top:2px">'
                f'{src_vel:.0f}/100</div></div>'
                f'<div><div style="font-size:10px;color:#6B8CAE;text-transform:uppercase">'
                f'Dest velocity</div>'
                f'<div style="font-size:13px;font-weight:600;color:#0BB99F;margin-top:2px">'
                f'{dest_vel:.0f}/100</div></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    # Recovery breakdown by action type
    st.markdown('<div class="sh">Recovery breakdown by action type</div>', unsafe_allow_html=True)

    if not recs.empty:
        rc1, rc2, rc3, rc4 = st.columns(4)
        cols_map = [
            (rc1, "TRANSFER", "#0BB99F"),
            (rc2, "MARKDOWN", "#D97706"),
            (rc3, "BUNDLE",   "#0072CE"),
            (rc4, "REVIEW",   "#888780"),
        ]
        for col, rtype, color in cols_map:
            subset = recs[recs["recommendation_type"] == rtype]
            with col:
                st.markdown(
                    f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;'
                    f'border-top:3px solid {color};border-radius:8px;padding:14px">'
                    f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;'
                    f'text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">'
                    f'{REC_CONFIG[rtype]["label"]}</div>'
                    f'<div style="font-size:20px;font-weight:800;color:{color}">'
                    f'{fmt_ksh(float(subset["estimated_recovery_ksh"].sum()))}</div>'
                    f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">'
                    f'{len(subset)} SKUs</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="sh">Frozen capital by category per branch</div>', unsafe_allow_html=True)
        if not dead_stock.empty and "internal_category" in dead_stock.columns:
            cb = dead_stock.groupby(["store_id","internal_category"])["ksh_at_risk"].sum().reset_index()
            cb["branch"] = cb["store_id"].map(BRANCH_NAMES)
            fig = go.Figure()
            for cat in cb["internal_category"].unique():
                sub = cb[cb["internal_category"] == cat]
                fig.add_trace(go.Bar(
                    name=cat, x=sub["branch"], y=sub["ksh_at_risk"],
                    marker_color=CAT_COLORS.get(cat,"#888780"),
                    hovertemplate=f"{cat}: KSh %{{y:,.0f}}<extra></extra>",
                ))
            fig.update_layout(
                barmode="stack", paper_bgcolor="#fff", plot_bgcolor="#fff",
                height=320, margin=dict(l=0,r=0,t=10,b=30),
                legend=dict(font=dict(size=10,family="Montserrat"),bgcolor="#fff",bordercolor="#D6E4F0"),
                xaxis=dict(tickfont=dict(color="#003467",size=10,family="Montserrat")),
                yaxis=dict(tickfont=dict(color="#6B8CAE",size=9, family="Montserrat"),gridcolor="#EBF3FB"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sh">Recommendations by action type</div>', unsafe_allow_html=True)
        if not recs.empty:
            rc_cnt = recs["recommendation_type"].value_counts().reset_index()
            rc_cnt.columns = ["Type","Count"]
            colors = [REC_CONFIG.get(t,{}).get("color","#888780") for t in rc_cnt["Type"]]
            fig2 = go.Figure(go.Bar(
                x=rc_cnt["Count"], y=rc_cnt["Type"],
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}: %{x} SKUs<extra></extra>",
            ))
            fig2.update_layout(
                paper_bgcolor="#fff", plot_bgcolor="#fff",
                height=200, margin=dict(l=0,r=20,t=10,b=10),
                xaxis=dict(tickfont=dict(color="#6B8CAE",size=10,family="Montserrat"),gridcolor="#EBF3FB"),
                yaxis=dict(tickfont=dict(color="#003467",size=11,family="Montserrat")),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="sh" style="margin-top:8px">Top frozen SKUs</div>', unsafe_allow_html=True)
        if not dead_stock.empty:
            top_ds = dead_stock.nlargest(10,"ksh_at_risk")[
                ["product_name","internal_category","tier","days_frozen","ksh_at_risk","store_id"]
            ].copy()
            top_ds["Branch"]     = top_ds["store_id"].map(BRANCH_NAMES)
            top_ds["KSh at risk"] = top_ds["ksh_at_risk"].apply(fmt_ksh)
            top_ds["Days"]        = top_ds["days_frozen"].astype(int)
            st.dataframe(
                top_ds[["product_name","internal_category","tier","Days","KSh at risk","Branch"]]
                .rename(columns={"product_name":"Product","internal_category":"Category","tier":"Tier"}),
                hide_index=True, use_container_width=True, height=260,
            )