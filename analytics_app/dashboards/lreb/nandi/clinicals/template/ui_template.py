"""
ui_template.py: shared UI for every tab of the Nandi County OPD dashboard.

Every view imports from here; no view carries its own CSS or colours.

    page_header, section_header, kpi_row, chart_card, insight_bar, why_cards,
    synthesis, html_table, bar_chart, stacked_bar_chart, grouped_dumbbell,
    placeholder_page, loading, plus the colour tokens and CHART_LAYOUT.

Maps live in template/geo_map.py (render_geo).
"""
from __future__ import annotations

import re
from contextlib import contextmanager

import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------- tokens
PRIMARY = "#0F6E56"      # teal: Nandi / positive / "more"
BLUE = "#0C447C"         # main data colour
AMBER = "#D97706"        # warning / "fewer"
CRITICAL = "#A32D2D"     # critical / injury
NEUTRAL = "#888780"
BG = "#F6F5F0"
SURFACE = "#FFFFFF"
SURFACE_2 = "#F0F1F5"
BORDER = "#E5E7EB"
TEXT = "#111827"
TEXT_MUTED = "#6B7280"
GREEN = "#2E8B3A"        # Nandi green (text and links); fills use #43A94B
GREEN_FILL = "#43A94B"
GOLD = "#F8C818"
BROWN = "#402818"
SIDEBAR_BG = BROWN
SIDEBAR_ACTIVE = GREEN_FILL
FONT_FAMILY = "Inter, 'Segoe UI', system-ui, sans-serif"

ACCENTS = {"blue": BLUE, "teal": PRIMARY, "amber": AMBER, "crit": CRITICAL, "neutral": NEUTRAL}
TONES = {  # insight bar tone -> (background, left border)
    "teal": ("#EAF6EB", GREEN),
    "amber": ("#FEF3E0", AMBER),
    "red": ("#FFF5F5", CRITICAL),
    "blue": ("#EEF4FB", BLUE),
}

CHART_LAYOUT = dict(
    font=dict(family=FONT_FAMILY, size=12, color=TEXT),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=20, t=10, b=10),
    hoverlabel=dict(font_family=FONT_FAMILY),
    legend=dict(orientation="h", y=-0.15),
)
PC_CFG = {"displayModeBar": False, "responsive": True}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT_FAMILY}; }}
.stApp {{ background: {BG}; }}
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] * {{ font-family: {FONT_FAMILY}; }}
header[data-testid="stHeader"] {{ background: transparent; }}  /* keeps the sidebar toggle reachable */
/* while a tab loads, hide the previous tab's leftover (stale) elements so only the loading card shows */
[data-testid="stMain"] [data-stale="true"], .stMain [data-stale="true"] {{ display:none !important; }}
[data-testid="stSkeleton"], [class*="stSkeleton"], [class*="Skeleton"], [class*="skeleton"] {{ display:none !important; }}
[data-testid="stElementContainer"]:has([data-testid="stSkeleton"]), .element-container:has(.stSkeleton) {{ display:none !important; }}
/* the loading card sits in a full-page layer, so nothing from the old tab or Streamlit's placeholders can show through */
.afya-load-ov {{ position:fixed; top:0; bottom:0; left:0; right:0; z-index:1000; background:{BG}; padding:96px 28px 28px; overflow:hidden; }}
.stApp:has(section[data-testid="stSidebar"][aria-expanded="true"]) .afya-load-ov {{ left:300px; }}
.afya-load-ov .afya-load {{ max-width:1100px; margin:0 auto; }}  /* Streamlit's grey placeholder boxes while a tab loads */
/* any bar or card that holds a link to another tab is clickable all over, not just on the link text */
.afya-link, .afya-insight, .afya-synth, .afya-limit {{ position:relative; }}
.afya-link a[href*="?tab="]::after, .afya-insight a[href*="?tab="]::after, .afya-synth a[href*="?tab="]::after, .afya-limit a[href*="?tab="]::after {{ content:""; position:absolute; inset:0; }}
.afya-link:has(a[href*="?tab="]), .afya-insight:has(a[href*="?tab="]), .afya-synth:has(a[href*="?tab="]), .afya-limit:has(a[href*="?tab="]) {{ cursor:pointer; transition:box-shadow .15s; }}
.afya-link:has(a[href*="?tab="]):hover, .afya-insight:has(a[href*="?tab="]):hover, .afya-synth:has(a[href*="?tab="]):hover, .afya-limit:has(a[href*="?tab="]):hover {{ box-shadow:0 2px 10px rgba(0,0,0,.10); }}
footer {{ visibility: hidden; }}  /* keep the ⋮ menu: Clear cache, Print, Rerun */
.block-container {{ padding: 0 28px 48px 28px; max-width: 100%; }}
[data-testid="stSidebar"] {{ background: {SIDEBAR_BG}; }}
[data-testid="stSidebar"] * {{ color: #EDE4DC; }}
.afya-page-header {{ background:#fff; border-bottom:1px solid {BORDER}; margin:0 -28px 12px -28px; padding:20px 28px 14px; }}
.afya-page-title {{ font-size:18px; font-weight:500; color:{TEXT}; }}
.afya-page-sub {{ font-size:12px; color:{TEXT_MUTED}; margin-top:2px; }}
.afya-ph {{ background:#fff; border-bottom:1px solid {BORDER}; margin:0 -28px 12px -28px; padding:16px 28px; display:grid; grid-template-columns:48px 1fr auto; gap:16px; align-items:center; }}
.afya-ph-ic {{ width:48px; height:48px; border-radius:12px; display:grid; place-items:center; }}
.afya-ph-ic svg {{ width:24px; height:24px; fill:none; stroke:currentColor; stroke-width:1.8; stroke-linecap:round; stroke-linejoin:round; }}
.afya-ph-t {{ font-size:20px; font-weight:650; color:{TEXT}; line-height:1.2; }}
.afya-ph-s {{ font-size:12.5px; color:{TEXT_MUTED}; margin-top:3px; line-height:1.4; max-width:760px; }}
.afya-ph-k {{ text-align:right; border-left:1px solid {BORDER}; padding-left:20px; min-width:120px; }}
.afya-ph-n {{ font-size:22px; font-weight:700; line-height:1.1; }}
.afya-ph-l {{ font-size:11px; color:{TEXT_MUTED}; margin-top:2px; }}
@media(max-width:760px) {{ .afya-ph {{ grid-template-columns:40px 1fr; padding:14px 16px; gap:12px; }} .afya-ph-ic {{ width:40px; height:40px; }} .afya-ph-k {{ display:none; }} }}
.afya-section {{ margin:28px 0 12px; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:{BROWN}; padding-left:10px; border-left:3px solid {GOLD}; }}
.afya-kpis {{ display:grid; grid-template-columns:repeat(6,1fr); gap:12px; margin:4px 0 0; }}
@media(max-width:1100px) {{ .afya-kpis {{ grid-template-columns:repeat(3,1fr); }} }}
.afya-kpi {{ background:#fff; border:1px solid {BORDER}; border-top:3px solid {BLUE}; border-radius:8px; padding:14px 16px 12px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.afya-kpi-label {{ font-size:11px; font-weight:600; color:{TEXT_MUTED}; letter-spacing:.04em; text-transform:uppercase; margin-bottom:4px; }}
.afya-kpi-value {{ font-size:22px; font-weight:700; line-height:1.1; color:{TEXT}; }}
.afya-kpi-sub {{ font-size:11px; color:{TEXT_MUTED}; margin-top:3px; }}
div[class*="st-key-card-"] {{ background:#fff; border:1px solid {BORDER}; border-radius:8px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.afya-card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-bottom:6px; }}
.afya-card-title {{ font-size:13px; font-weight:600; color:{TEXT}; }}
.afya-src {{ font-size:10px; font-weight:600; padding:2px 7px; border-radius:99px; background:{SURFACE_2}; color:{TEXT_MUTED}; white-space:nowrap; }}
.afya-src.new {{ background:#FEF3E0; color:#B45309; }}
.afya-note {{ font-size:11px; font-style:italic; color:{TEXT_MUTED}; margin-bottom:6px; }}
.afya-insight {{ border-radius:0 6px 6px 0; padding:12px 16px; margin:14px 0 12px; font-size:13px; line-height:1.6; color:{TEXT}; }}
.afya-why {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:12px; }}
@media(max-width:900px) {{ .afya-why {{ grid-template-columns:1fr; }} }}
.afya-why div {{ border:1px solid {BORDER}; border-left:3px solid {BLUE}; border-radius:0 6px 6px 0; padding:10px 12px; font-size:12px; background:#fff; }}
.afya-why b {{ display:block; margin-bottom:2px; }}
.afya-synth {{ margin-top:28px; border-left:4px solid {GOLD}; background:#FFF8DB; border-radius:0 6px 6px 0; padding:16px 20px; font-size:13px; line-height:1.7; }}
.afya-table {{ border-collapse:collapse; width:100%; font-size:12px; }}
.afya-table th {{ padding:6px 8px; font-size:10px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:{TEXT_MUTED}; border-bottom:1px solid {BORDER}; text-align:left; }}
.afya-table td {{ padding:7px 8px; border-bottom:1px solid {BORDER}; }}
.afya-table .r {{ text-align:right; }}
.afya-table .hl {{ font-weight:700; }}
.afya-demo {{ background:#FEF3E0; border:1px dashed {AMBER}; color:#92400E; border-radius:6px; padding:8px 12px; font-size:12px; margin-bottom:12px; }}
[data-testid="stStatusWidget"] {{ visibility:hidden; }}  /* no "Running q_xxx()" in the corner */
div[class*="st-key-card-"] {{ min-width:0; overflow:hidden; box-sizing:border-box; }}
/* cards in the same row share one height; at most two per row */
div[data-testid="stHorizontalBlock"] {{ align-items:stretch; }}
div[data-testid="stColumn"] {{ display:flex; flex-direction:column; min-width:0; }}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {{ flex:1 1 auto; }}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div:has(> div[class*="st-key-card-"]),
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] > div[class*="st-key-card-"] {{ flex:1 1 auto; }}
div[data-testid="stColumn"] div[class*="st-key-card-"] {{ height:100%; }}
@media(max-width:1000px) {{
  div[data-testid="stHorizontalBlock"] {{ flex-wrap:wrap; }}
  div[data-testid="stColumn"] {{ flex:1 1 100% !important; min-width:100% !important; width:100% !important; }}
}}
.afya-table-wrap {{ overflow-x:auto; width:100%; }}
.afya-table.compact th, .afya-table.compact td {{ padding:5px 4px; font-size:11px; }}
.afya-table.compact th {{ font-size:9px; letter-spacing:.03em; }}
.afya-load {{ background:#fff; border:1px solid {BORDER}; border-radius:10px; padding:44px 24px; margin:18px 0; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,.04); }}
.afya-load-title {{ font-size:15px; font-weight:600; color:{TEXT}; margin-top:6px; }}
.afya-load-quip {{ position:relative; height:22px; margin-top:8px; font-size:12.5px; color:{TEXT_MUTED}; font-style:italic; }}
.afya-load-quip span {{ position:absolute; left:0; right:0; opacity:0; }}
.afya-ecg {{ width:220px; max-width:70%; height:56px; }}
.afya-ecg path {{ fill:none; stroke:{PRIMARY}; stroke-width:2.5; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:420; stroke-dashoffset:420; animation:afya-beat 1.8s linear infinite; }}
@keyframes afya-beat {{ 0% {{ stroke-dashoffset:420; }} 70% {{ stroke-dashoffset:0; }} 100% {{ stroke-dashoffset:-420; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- page chrome
_ICONS = {
    "injury": '<path d="M12 3v18M3 12h18"/><rect x="5" y="5" width="14" height="14" rx="3"/>',
    "workload": '<circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.4"/><path d="M16.5 14.2c2.7.2 4.5 2.4 4.5 5.3"/>',
    "dataquality": '<path d="M12 3l7 3v5.5c0 4.4-2.9 7.9-7 9.5-4.1-1.6-7-5.1-7-9.5V6z"/><path d="M8.8 12l2.4 2.4 4.2-4.6"/>',
    "maternal": '<circle cx="12.500" cy="6.500" r="3"/><circle cx="8.800" cy="3.400" r="1.300"/><path d="M4.500 21c-.8-4.500.2-8.500 3-10 1.200-.7 2.800-1 5-1s3.800.3 5 1c2.800 1.500 3.800 5.500 3 10"/><circle cx="15.500" cy="15.500" r="1.900"/><path d="M8 18.500c2.800.9 5.500.9 8-.3"/>',
    "mental": '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 6 1V4a3 3 0 0 0-3 0z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-6 1"/>',
    "infectious": '<circle cx="12" cy="12" r="4"/><path d="M16.0 12.0L19.0 12.0"/><circle cx="20.3" cy="12.0" r="1.1"/><path d="M14.8 14.8L16.9 16.9"/><circle cx="17.9" cy="17.9" r="1.1"/><path d="M12.0 16.0L12.0 19.0"/><circle cx="12.0" cy="20.3" r="1.1"/><path d="M9.2 14.8L7.1 16.9"/><circle cx="6.1" cy="17.9" r="1.1"/><path d="M8.0 12.0L5.0 12.0"/><circle cx="3.7" cy="12.0" r="1.1"/><path d="M9.2 9.2L7.1 7.1"/><circle cx="6.1" cy="6.1" r="1.1"/><path d="M12.0 8.0L12.0 5.0"/><circle cx="12.0" cy="3.7" r="1.1"/><path d="M14.8 9.2L16.9 7.1"/><circle cx="17.9" cy="6.1" r="1.1"/><circle cx="10.800" cy="11" r=".5"/><circle cx="13.200" cy="13" r=".5"/>',
    "ncd": '<path d="M5 3v5a5 5 0 0 0 10 0V3"/><path d="M10 13v3a5 5 0 0 0 10 0v-2"/><circle cx="20" cy="12" r="2"/>',
    "respiratory": '<path d="M12 4v8"/><path d="M12 8C9 8 6 11 5.5 16c-.3 2.5 1.5 4 3.5 3.5 1.7-.4 3-2 3-4.5"/><path d="M12 8c3 0 6 3 6.5 8 .3 2.5-1.5 4-3.5 3.5-1.7-.4-3-2-3-4.5"/>',
    "summary": '<rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/>',
    "demo": '<path d="M12 3a9 9 0 1 0 9 9h-9z"/><path d="M15 3.5A9 9 0 0 1 20.500 9H15z"/>',
    "geo": '<path d="M12 21s-7-6.2-7-11.5a7 7 0 0 1 14 0C19 14.800 12 21 12 21z"/><circle cx="12" cy="9.5" r="2.5"/>',
}
_PH_COLORS = {"injury": "#A32D2D", "workload": "#DB2777", "dataquality": "#4F46E5", "summary": "#2563EB", "demo": "#65A30D", "maternal": "#0F6E56", "mental": "#6D28D9",
              "infectious": "#D97706", "ncd": "#1D6FA8", "respiratory": "#0E7490", "geo": "#0C447C"}


def _lighten(hexc: str, t: float = 0.45) -> str:
    r, g, b = (int(hexc[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02X%02X%02X" % tuple(round(v + (255 - v) * t) for v in (r, g, b))


def icon_uri(icon: str, light: bool = True) -> str:
    """data: URI of a tab icon in its tab colour (lightened for the dark sidebar)."""
    from urllib.parse import quote
    c = _lighten(_PH_COLORS[icon]) if light else _PH_COLORS[icon]
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="1.8" '
           f'stroke-linecap="round" stroke-linejoin="round">{_ICONS[icon]}</svg>')
    return "data:image/svg+xml," + quote(svg)


def page_header(title: str, subtitle: str = "", icon: str | None = None, stat: tuple | None = None) -> None:
    """Icon tile + title + subtitle, and one headline number on the right.
    icon: key of _ICONS (also sets the tab colour). stat: (value, label)."""
    if icon not in _ICONS:
        st.markdown(
            f'<div class="afya-page-header"><div class="afya-page-title">{title}</div>'
            f'<div class="afya-page-sub">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )
        return
    c = GREEN  # headers carry the Nandi green on every tab; _PH_COLORS only colours the sidebar icons
    k = (f'<div class="afya-ph-k"><div class="afya-ph-n" style="color:{c}">{stat[0]}</div>'
         f'<div class="afya-ph-l">{stat[1]}</div></div>') if stat else "<div></div>"
    st.markdown(
        f'<div class="afya-ph"><div class="afya-ph-ic" style="background:#43A94B24;color:{c}">'
        f'<svg viewBox="0 0 24 24">{_ICONS[icon]}</svg></div>'
        f'<div><div class="afya-ph-t">{title}</div><div class="afya-ph-s">{subtitle}</div></div>{k}</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    st.markdown(f'<div class="afya-section">{text}</div>', unsafe_allow_html=True)


def kpi_row(cards: list[dict]) -> None:
    """cards: [{label, value, sub, accent('blue'|'teal'|'amber'|'crit')}]"""
    html = "".join(
        f'<div class="afya-kpi" style="border-top-color:{ACCENTS.get(c.get("accent", "blue"), BLUE)}">'
        f'<div class="afya-kpi-label">{c["label"]}</div><div class="afya-kpi-value">{c["value"]}</div>'
        f'<div class="afya-kpi-sub">{c.get("sub", "")}</div></div>'
        for c in cards
    )
    st.markdown(f'<div class="afya-kpis">{html}</div>', unsafe_allow_html=True)


def insight_bar(html: str, tone: str = "teal") -> None:
    bg, border = TONES.get(tone, TONES["teal"])
    st.markdown(
        f'<div class="afya-insight" style="background:{bg};border-left:3px solid {border}">{html}</div>',
        unsafe_allow_html=True,
    )


def why_cards(items: list[tuple[str, str]]) -> None:
    html = "".join(f"<div><b>{t}</b>{b}</div>" for t, b in items)
    st.markdown(f'<div class="afya-why">{html}</div>', unsafe_allow_html=True)


def synthesis(html: str) -> None:
    st.markdown(f'<div class="afya-synth"><strong>Synthesis.</strong> {html}</div>', unsafe_allow_html=True)


def demo_banner() -> None:
    st.markdown(
        '<div class="afya-demo"><b>Demo data.</b> NANDI_DEMO is on, so every number on this page is '
        "illustrative. Unset NANDI_DEMO to read from Snowflake.</div>",
        unsafe_allow_html=True,
    )


@contextmanager
def chart_card(title: str, src: str | None = None, new: bool = False, note: str | None = None):
    """White card with a title and a small source badge. Use as a context manager."""
    key = "card-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    with st.container(key=key):
        badge = f'<span class="afya-src{" new" if new else ""}">{src}</span>' if src else ""
        st.markdown(
            f'<div class="afya-card-head"><div class="afya-card-title">{title}</div>{badge}</div>',
            unsafe_allow_html=True,
        )
        if note:
            st.markdown(f'<div class="afya-note">{note}</div>', unsafe_allow_html=True)
        yield


def _mix(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:.2f})"


def heat(value: float, lo: float, hi: float, colour: str = BLUE) -> str:
    """Inline style that shades a table cell by where value sits between lo and hi (white text once dark)."""
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (value - lo) / (hi - lo)))
    a = 0.08 + 0.72 * t
    return f"background:{_mix(colour, a)};color:{'#fff' if a > 0.6 else TEXT};font-weight:600"


def diverge(value: float, base: float, span: float, above: str = PRIMARY, below: str = AMBER) -> str:
    """Shade a cell teal above `base` and amber below it, stronger the further away it is."""
    t = max(0.0, min(1.0, abs(value - base) / span)) if span else 0.0
    a = 0.10 + 0.70 * t
    return f"background:{_mix(above if value >= base else below, a)};color:{'#fff' if a > 0.6 else TEXT};font-weight:600"


def html_table(columns: list[tuple[str, str]], rows: list[list], right_from: int = 1, compact: bool = False,
               widths: list[str] | None = None, min_width: int | None = None) -> None:
    """columns: [(header, key)]; rows: lists of strings, (text, css_class) or (text, css_class, inline_style) tuples.
    The table scrolls sideways inside its card instead of being clipped; compact=True tightens it for narrow cards.
    widths (e.g. ["20%", "10%", ...]) fixes the column widths so numeric columns come out the same size."""
    head = "".join(f'<th class="{"r" if i >= right_from else ""}">{h}</th>' for i, (h, _) in enumerate(columns))
    body = ""
    for r in rows:
        tds = ""
        for i, cell in enumerate(r):
            text, cls, style = (tuple(cell) + ("", ""))[:3] if isinstance(cell, tuple) else (cell, "", "")
            tds += f'<td class="{"r" if i >= right_from else ""} {cls}" style="{style}">{text}</td>'
        body += f"<tr>{tds}</tr>"
    col_tags = "".join('<col style="width:' + w + '">' for w in (widths or []))
    cols = f"<colgroup>{col_tags}</colgroup>" if widths else ""
    style = ("table-layout:fixed;" if widths else "") + (f"min-width:{min_width}px;" if min_width else "")
    st.markdown(f'<div class="afya-table-wrap"><table class="afya-table{" compact" if compact else ""}" style="{style}">{cols}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>',
                unsafe_allow_html=True)


def placeholder_page(title: str, subtitle: str = "") -> None:
    page_header(title, subtitle)
    insight_bar("This tab has not been built yet. It will use the same <b>ui_template</b> as the rest of the dashboard.", "blue")



# ----------------------------------------------------------------- loading screen
_QUIPS = {
    "Infectious disease": ["Counting coughs, fevers and febrile toddlers", "Asking the mosquitoes for their side of the story",
                           "Checking if the rain had anything to do with it", "Teaching the sepsis numbers to line up",
                           "Convincing 107,616 patients to hold still"],
    "Population & Geography": ["Finding Kapsabet on the map, again", "Measuring how far people really travel",
                               "Asking the wards where everyone lives", "Untangling counties from sub-counties"],
}
_QUIPS_DEFAULT = ["Warming up the stethoscope", "Politely asking Snowflake for numbers", "Counting the beds nobody counted",
                  "Untangling duplicate patient IDs", "Reading the doctors' handwriting"]


@contextmanager
def loading(page: str, quips: list[str] | None = None):
    """Show the page name with an animated heartbeat and rotating one-liners while a view loads its data.
    Use around the query calls:  with ui.loading("Infectious disease"):  D = _load()"""
    lines = quips or _QUIPS.get(page, _QUIPS_DEFAULT)
    n, step = len(lines), 2.6
    total = n * step
    on, off = 100 / n, 100 / n
    kf = (f"@keyframes afya-quip{{0%{{opacity:0;transform:translateY(6px)}}4%{{opacity:1;transform:none}}"
          f"{max(off - 4, 5):.1f}%{{opacity:1;transform:none}}{off:.1f}%{{opacity:0}}100%{{opacity:0}}}}")
    spans = "".join(f'<span style="animation:afya-quip {total:.1f}s linear infinite;animation-delay:{i * step:.1f}s;animation-fill-mode:backwards">{t}…</span>' for i, t in enumerate(lines))
    html = (f"<style>{kf}</style><div class='afya-load-ov'><div class='afya-load'>"
            "<svg class='afya-ecg' viewBox='0 0 220 56'><path d='M0 30 H62 L72 30 L80 8 L92 50 L102 22 L108 30 H220'/></svg>"
            f"<div class='afya-load-title'>Loading {page}</div><div class='afya-load-quip'>{spans}</div></div></div>")
    slot = st.empty()
    slot.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        slot.empty()

# ----------------------------------------------------------------- trend arrows
def trend_color(r: float, up_is: str = "bad") -> str:
    """Arrow colour from the direction and what that direction means.
    up_is: 'bad' (more disease or more problems: up red, down green), 'good' (up green, down red), 'neutral' (always amber).
    No change is amber."""
    if not r or up_is == "neutral":
        return AMBER
    up = r > 0
    return CRITICAL if (up and up_is == "bad") or (not up and up_is == "good") else GREEN


def arrow_html(r: float, up_is: str = "bad", flat: str = "no change", pts: bool = False, dec: int = 0) -> str:
    """Bold coloured arrow with the change, e.g. ▲ +5%. r is the change in percent (or points when pts=True)."""
    c = trend_color(r, up_is)
    if not r:
        return f'<b style="color:{c}">&#9654; {flat}</b>'
    unit = " pts" if pts else "%"
    return f'<b style="color:{c}">{"&#9650;" if r > 0 else "&#9660;"} {r:+.{dec}f}{unit}</b>'


# ----------------------------------------------------------------- charts (Plotly)
def _finish(fig: go.Figure, height: int, **layout) -> go.Figure:
    fig.update_layout(**{**CHART_LAYOUT, "height": height, **layout})
    fig.update_xaxes(showgrid=True, gridcolor=SURFACE_2, zeroline=False, tickfont=dict(size=11, color=TEXT_MUTED))
    fig.update_yaxes(showgrid=False, tickfont=dict(size=11, color=TEXT))
    return fig


def bar_chart(labels, values, colors=None, texts=None, height: int = 220, xsuffix: str = "") -> go.Figure:
    """Horizontal bars, first label on top."""
    colors = colors or [BLUE] * len(labels)
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors, text=texts,
                           textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:,}<extra></extra>"))
    _finish(fig, height, showlegend=False, margin=dict(l=10, r=60, t=6, b=6))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(ticksuffix=xsuffix, showgrid=False, visible=False)
    return fig


def stacked_bar_chart(labels, series: list[tuple[str, list, str]], height: int = 320, xsuffix: str = "") -> go.Figure:
    """series: [(name, values, colour)]. Horizontal stacked bars, first label on top."""
    fig = go.Figure()
    for name, vals, col in series:
        fig.add_trace(go.Bar(x=vals, y=labels, orientation="h", name=name, marker_color=col,
                             hovertemplate=f"%{{y}}<br>{name}: %{{x:.1f}}{xsuffix}<extra></extra>"))
    _finish(fig, height, barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(ticksuffix=xsuffix)
    return fig


def pyramid_chart(groups, female, male, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=groups, x=[-v for v in female], orientation="h", name="Female", marker_color=PRIMARY, customdata=female,
                         hovertemplate="%{y} · Female: %{customdata:,}<extra></extra>"))
    fig.add_trace(go.Bar(y=groups, x=male, orientation="h", name="Male", marker_color=BLUE,
                         hovertemplate="%{y} · Male: %{x:,}<extra></extra>"))
    _finish(fig, height, barmode="overlay")
    mx = max(list(female) + list(male) + [1])
    step = max(1, round(mx / 3, -3) or 1000)
    ticks = [-step * 2, -step, 0, step, step * 2]
    fig.update_xaxes(tickvals=ticks, ticktext=[f"{abs(t) // 1000}k" if t else "0" for t in ticks])
    return fig
