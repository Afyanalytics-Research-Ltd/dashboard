"""Palette, CSS and HTML building blocks.

Colour carries meaning: navy neutral, teal on track / increase / headline,
amber needs attention, red off track (text) / decrease.
"""

import base64
import os

import streamlit as st

NAVY = "#003467"
TEAL = "#0BB99F"
TEAL_D = "#08907C"
AMBER = "#C26A05"
RED = "#C0392B"
INK = "#1B3350"
MUTED = "#4C627A"
LABEL = "#556C85"
BORDER = "#DCE4ED"
DIVIDE = "#E8EEF4"
TRACK = "#E6ECF2"
SOFT = "#B7CADB"
PAGE = "#E9EEF5"
ON_NAVY = "#A8C3DE"
FONT = "Montserrat, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
SHADOW = "0 1px 2px rgba(16,42,74,.05), 0 2px 8px rgba(16,42,74,.04)"

STATUS_COLOUR = {"met": TEAL, "on": TEAL, "risk": AMBER, "off": AMBER, None: NAVY}
STATUS_PILL = {"met": (TEAL_D, "#E3F5F1"), "on": (TEAL_D, "#E3F5F1"),
               "risk": (AMBER, "#FBF0E1"), "off": (RED, "#FBE7E5")}

_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');

html, body, .stApp, button, input, label, p, li, td, th,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-baseweb="select"] *, [data-baseweb="popover"] li {{ font-family:{FONT} !important; }}
[data-testid="stIconMaterial"], [data-testid="stExpanderIcon"] {{ font-family:'Material Symbols Rounded' !important; }}
.stApp {{ background:{PAGE} !important; }}
[data-testid="stMainBlockContainer"] {{ max-width:1280px; padding:1.1rem 2.25rem 1.25rem; }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], #MainMenu, footer {{ display:none !important; }}

/* header (sticky: Streamlit's wrapper is the element that can stick) */
.st-key-x_head {{ background:{NAVY}; border-radius:14px; padding:18px 28px;
                 box-shadow:0 2px 10px rgba(0,30,64,.18); position:relative; }}
[data-testid="stLayoutWrapper"]:has(> .st-key-x_head) {{ position:sticky; top:12px; z-index:999; }}
.st-key-x_head::before {{ content:""; position:absolute; z-index:-1; left:-4px; right:-4px;
                         top:-12px; height:calc(100% + 14px); background:{PAGE}; }}
.x-brand {{ display:flex; align-items:center; gap:16px; }}
.x-brand img {{ height:40px; display:block; filter:brightness(0) invert(1); }}
.x-title {{ font-size:22px; font-weight:800; color:#fff; letter-spacing:-.3px; line-height:1.1; }}
.x-org {{ font-size:9.5px; font-weight:700; color:{ON_NAVY}; text-transform:uppercase; letter-spacing:2px;
         margin-top:5px; }}

/* week picker and Update button */
[data-testid="stMain"] [data-baseweb="select"] > div:first-child {{
    background:#fff !important; border:1px solid {BORDER} !important; border-radius:8px !important;
    min-height:36px !important; box-shadow:{SHADOW} !important;
}}
[data-testid="stMain"] [data-baseweb="select"] div {{ color:{NAVY} !important; font-weight:600 !important; font-size:12.5px !important; }}
[data-testid="stMain"] [data-baseweb="select"] svg {{ color:{MUTED} !important; }}
.st-key-x_update button {{
    min-height:36px !important; border-radius:8px !important; border:none !important;
    background:#fff !important; box-shadow:0 1px 2px rgba(0,0,0,.15) !important; transition:background .15s ease;
}}
.st-key-x_update button p, .st-key-x_update button [data-testid="stIconMaterial"] {{
    color:{NAVY} !important; font-size:12.5px !important; font-weight:700 !important;
}}
.st-key-x_update button:hover {{ background:#E8F1FA !important; }}
.st-key-x_update button:focus:not(:active) {{ border:none !important; box-shadow:0 0 0 3px rgba(11,185,159,.30) !important; }}

/* tabs */
[data-testid="stTabs"] {{ margin-top:12px; }}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:26px; border-bottom:1px solid {BORDER}; }}
[data-testid="stTabs"] button[data-baseweb="tab"] {{ padding:0 0 10px !important; }}
[data-testid="stTabs"] button[data-baseweb="tab"] p {{ font-size:13px !important; font-weight:600 !important; color:{MUTED} !important; }}
[data-testid="stTabs"] button[aria-selected="true"] p {{ color:{NAVY} !important; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background:{TEAL} !important; height:2px !important; }}
[data-testid="stTabs"] [data-baseweb="tab-border"] {{ display:none; }}

/* quarter strip */
.x-strip {{ display:flex; align-items:center; gap:14px; font-size:12.5px; color:{MUTED}; min-height:36px; }}
.st-key-x_ctx {{ margin:2px 0 8px; }}
.x-ctx {{ margin:8px 0 14px; }}
.x-strip b {{ color:{NAVY}; font-weight:700; }}
.x-qbar {{ flex:1; height:3px; background:{TRACK}; border-radius:2px; overflow:hidden; }}
.x-qbar i {{ display:block; height:100%; background:{NAVY}; border-radius:2px; }}

/* panels; cell heights share what's left after ~390px of fixed chrome */
.x-panel {{ background:#fff; border:1px solid {BORDER}; border-radius:12px; box-shadow:{SHADOW};
           overflow:hidden; margin-bottom:12px; }}
.x-panel-hd {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px;
              padding:10px 24px 9px; border-bottom:1px solid {DIVIDE}; background:#F6F9FC; }}
.x-panel-t {{ font-size:10.5px; font-weight:700; color:{NAVY}; text-transform:uppercase; letter-spacing:.1em; }}
.x-panel-h {{ font-size:11.5px; color:{MUTED}; }}
.x-cells {{ display:grid; gap:1px; background:{DIVIDE}; }}
.x-cell {{ background:#fff; padding:18px 24px 16px; min-width:0; display:flex; flex-direction:column;
          min-height:max(112px, min(280px, calc((100vh - 390px) * .70))); }}
.x-lbl {{ font-size:10.5px; font-weight:700; color:{LABEL}; text-transform:uppercase; letter-spacing:.08em;
         white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.x-valrow {{ display:flex; align-items:baseline; gap:8px; margin-top:10px; flex-wrap:wrap; }}
.x-val {{ font-size:30px; font-weight:700; color:{NAVY}; line-height:1.1; letter-spacing:-.5px;
         font-variant-numeric:tabular-nums; }}
.x-unit {{ font-size:13px; font-weight:600; color:{MUTED}; }}
.x-chg {{ font-size:12px; font-weight:700; white-space:nowrap; font-variant-numeric:tabular-nums; }}
.x-chg.up {{ color:{TEAL_D}; }}
.x-chg.down {{ color:{RED}; }}
.x-bar {{ height:4px; background:{TRACK}; border-radius:2px; overflow:hidden; display:flex; gap:2px; }}
.x-bar i {{ display:block; height:100%; border-radius:2px; }}
.x-detail {{ margin-top:auto; padding-top:16px; }}
.x-foot {{ display:flex; justify-content:space-between; align-items:baseline; gap:10px; margin-top:9px;
          font-size:12px; color:{MUTED}; white-space:nowrap; }}
.x-foot b {{ color:{INK}; font-weight:700; }}
.x-foot .dot {{ display:inline-block; width:7px; height:7px; border-radius:2px; margin-right:6px; }}
.x-st {{ font-size:11.5px; font-weight:700; }}

/* target cell: centred stack of ring, value, target and status */
.x-cell.target {{ align-items:center; text-align:center; }}
.x-cell.target .x-lbl {{ align-self:stretch; }}
.x-ringrow {{ display:flex; flex-direction:column; align-items:center; gap:12px; margin:auto 0; padding-top:10px; }}
.x-ring {{ position:relative; flex-shrink:0; width:var(--sz); height:var(--sz); }}
.x-ring svg {{ display:block; transform:rotate(-90deg); }}
.x-ring b {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
            font-size:15px; font-weight:800; color:{NAVY}; font-variant-numeric:tabular-nums; }}
.x-ringtxt {{ min-width:0; }}
.x-ringtxt .x-valrow {{ margin-top:0; justify-content:center; }}
.x-ringtxt .x-val {{ font-size:26px; }}
.x-sub {{ font-size:12px; color:{MUTED}; margin-top:4px; white-space:nowrap; }}
.x-sub b {{ color:{INK}; font-weight:700; }}
.x-ringtxt .x-st {{ display:block; margin-top:4px; }}
.x-card.primary {{ box-shadow:inset 0 3px 0 {TEAL}, {SHADOW}; }}
.x-card.primary .x-lbl {{ color:{NAVY}; }}

/* wide cell: value on the left, bar and detail on the right */
.x-cell.wide {{ flex-direction:row; align-items:center; gap:40px; padding:16px 24px;
               min-height:max(68px, min(124px, calc((100vh - 390px) * .30))); }}
.x-cell.wide .x-main {{ min-width:190px; }}
.x-cell.wide .x-detail {{ flex:1; min-width:0; padding:0 0 3px; margin-top:0; }}
.x-cell.wide .x-val {{ font-size:26px; }}

/* summary tiles (patients tab) */
.x-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-bottom:14px; }}
.x-card {{ background:#fff; border:1px solid {BORDER}; border-radius:12px; padding:18px 22px;
          box-shadow:{SHADOW}; min-width:0; }}
.x-card .x-meta {{ font-size:12px; color:{MUTED}; margin-top:6px; }}

/* table */
.x-table {{ width:100%; border-collapse:separate !important; border-spacing:0; background:#fff;
           border:1px solid {BORDER} !important; border-radius:12px; overflow:hidden; font-size:12.5px;
           box-shadow:{SHADOW}; }}
.x-table th, .x-table td {{ border:none !important; }}
.x-table th {{ font-size:10.5px; font-weight:700; color:{LABEL}; text-transform:uppercase; letter-spacing:.08em;
              text-align:left; padding:11px 20px; background:#FAFCFE; border-bottom:1px solid {BORDER} !important; }}
.x-table td {{ padding:10px 20px; color:{INK}; font-weight:500; border-bottom:1px solid {DIVIDE} !important; }}
.x-table tbody tr:hover td {{ background:#F8FBFD; }}
.x-table tbody tr:last-child td {{ border-bottom:none !important; }}
.x-table tfoot td {{ font-weight:800; color:{NAVY}; background:#FAFCFE; border-top:1px solid {BORDER} !important; }}
.x-table .x-num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.x-table .x-dim {{ color:{MUTED}; }}
.x-table .x-flag {{ color:{AMBER}; font-weight:700; cursor:help; }}
.x-table .x-barcell {{ width:30%; }}
.x-hbar {{ position:relative; height:4px; background:{TRACK}; border-radius:2px; overflow:hidden; }}
.x-hbar i {{ position:absolute; left:0; top:0; bottom:0; border-radius:2px; }}
.x-key {{ text-transform:none; letter-spacing:0; font-weight:600; color:{MUTED}; margin-right:16px; }}
.x-key i {{ display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:6px; }}

/* empty / error state */
.x-empty {{ background:#fff; border:1px dashed {BORDER}; border-radius:12px; padding:64px 24px;
           text-align:center; color:{MUTED}; font-size:14px; margin-top:24px; }}
.x-empty b {{ display:block; color:{NAVY}; font-size:18px; margin-bottom:6px; }}

/* update dialog */
.x-sec {{ font-size:11px; font-weight:700; color:{LABEL}; text-transform:uppercase; letter-spacing:.08em;
         margin:16px 0 4px; padding-bottom:6px; border-bottom:1px solid {BORDER}; }}
[data-testid="stDialog"] [data-testid="stBaseButton-primary"] {{ background:{NAVY} !important;
    border-color:{NAVY} !important; border-radius:10px !important; min-height:42px; }}
[data-testid="stDialog"] [data-testid="stCaptionContainer"] {{ margin-top:-12px; }}
[data-testid="stDialog"] [data-testid="stCaptionContainer"] p {{ font-size:11.5px !important; }}
[data-testid="stDialog"] [data-testid="stBaseButton-segmented_controlActive"] {{
    background:{NAVY} !important; border-color:{NAVY} !important; }}
[data-testid="stDialog"] [data-testid="stBaseButton-segmented_controlActive"] p {{ color:#fff !important; }}
[data-testid="stDialog"] [data-testid="stBaseButton-segmented_control"] p {{ color:{NAVY} !important; }}

/* short windows: tighter gaps so both panels fit */
@media (max-height:620px) {{
    [data-testid="stMainBlockContainer"] {{ padding-top:.6rem; }}
    .st-key-x_head {{ padding:12px 26px; }}
    [data-testid="stTabs"] {{ margin-top:6px; }}
    .st-key-x_ctx {{ margin:0 0 4px; }}
    .x-panel {{ margin-bottom:8px; }}
    .x-panel-hd {{ padding:7px 24px 6px; }}
    .x-cell {{ padding-top:14px; padding-bottom:12px; }}
    .x-cell.wide {{ padding-top:12px; padding-bottom:12px; }}
    .x-ring {{ --sz:56px !important; }}
}}
@media (max-width:1100px) {{
    .x-cells {{ grid-template-columns:repeat(2,minmax(0,1fr)) !important; }}
    .x-cell.wide {{ flex-direction:column; align-items:stretch; gap:14px; }}
}}
@media (max-width:640px) {{
    [data-testid="stMainBlockContainer"] {{ padding:1.4rem 1rem 2rem; }}
    .x-cells, .x-grid {{ grid-template-columns:1fr !important; }}
    .x-strip .x-qbar {{ display:none; }}
}}
"""


def inject_css():
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def logo_tag(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return f'<img src="data:image/png;base64,{base64.b64encode(f.read()).decode()}" alt="AFYA">'


def bar(pct, colour):
    return f'<div class="x-bar"><i style="width:{min(max(pct, 0), 100):.1f}%;background:{colour}"></i></div>'


def split(parts):
    """Composition bar from [(value, colour), ...]."""
    segs = "".join(f'<i style="flex:{v or .001};background:{c}"></i>' for v, c in parts)
    return f'<div class="x-bar" style="background:none">{segs}</div>'


def ring(pct, colour, size=76, stroke=7):
    r = (size - stroke) / 2
    c = 2 * 3.14159265 * r
    dash = c * min(max(pct, 0), 100) / 100
    return (f'<div class="x-ring" style="--sz:{size}px"><svg width="100%" height="100%" viewBox="0 0 {size} {size}">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{TRACK}" stroke-width="{stroke}"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{colour}" stroke-width="{stroke}" '
            f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {c:.1f}"/></svg><b>{pct:.0f}%</b></div>')


def ring_cell(label, value, ring_html, change="", sub="", status=""):
    """Target cell: label, then a centred ring, value, target and status."""
    return (f'<div class="x-cell target"><div class="x-lbl">{label}</div><div class="x-ringrow">{ring_html}'
            f'<div class="x-ringtxt"><div class="x-valrow"><span class="x-val">{value}</span>{change}</div>'
            f'<div class="x-sub">{sub}</div>{status or '<span class="x-st">&nbsp;</span>'}</div></div></div>')


def cell(label, value, unit="", change="", bar="", left="", right="", primary=False, wide=False):
    """Metric cell: label, value (+ unit, change), bar, detail line."""
    unit_html = f'<span class="x-unit">{unit}</span>' if unit else ""
    main = (f'<div class="x-main"><div class="x-lbl">{label}</div>'
            f'<div class="x-valrow"><span class="x-val">{value}</span>{unit_html}{change}</div></div>')
    detail = (f'<div class="x-detail">{bar}'
              f'<div class="x-foot"><span>{left}</span><span>{right}</span></div></div>')
    cls = "x-cell" + (" primary" if primary else "") + (" wide" if wide else "")
    return f'<div class="{cls}">{main}{detail}</div>'


def panel(title, hint, cells):
    hint_html = f'<span class="x-panel-h">{hint}</span>' if hint else ""
    return html(f'<div class="x-panel"><div class="x-panel-hd"><span class="x-panel-t">{title}</span>{hint_html}</div>'
                f'<div class="x-cells" style="grid-template-columns:repeat({len(cells)},minmax(0,1fr))">'
                f'{"".join(cells)}</div></div>')


def tile(label, value, meta="", primary=False):
    meta_html = f'<div class="x-meta">{meta}</div>' if meta else ""
    return (f'<div class="x-card{" primary" if primary else ""}"><div class="x-lbl">{label}</div>'
            f'<div class="x-valrow"><span class="x-val">{value}</span></div>{meta_html}</div>')
