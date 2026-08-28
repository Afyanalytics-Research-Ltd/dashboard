import os
import numpy as np
import pandas as pd
import streamlit as st
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.notifier import send_digest, get_recipients

_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "st_peters_logo.png")

COLORS = {
    "primary": "#0072CE",
    "success": "#0BB99F",
    "warning": "#D97706",
    "danger":  "#E11D48",
    "muted":   "#6B8CAE",
    "dark":    "#003467",
    "purple":  "#7F77DD",
    "coral":   "#D85A30",
    "green":   "#1D9E75",
}

CHART_LAYOUT = dict(
    paper_bgcolor="#fff",
    plot_bgcolor="#fff",
    font=dict(family="Montserrat", color="#003467"),
    margin=dict(l=0, r=0, t=10, b=30),
    xaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
    yaxis=dict(gridcolor="#EBF3FB", tickfont=dict(size=10, color="#6B8CAE")),
)

_CSS = """
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
    background: #fff;
    color: #003467;
}
.stApp { background: #fff; }

/* ── Sidebar shell ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F0F5FA !important;
    border-right: 1px solid #D6E4F0 !important;
}
[data-testid="stSidebar"] *:not(i) {
    font-family: 'Montserrat', sans-serif !important;
}

/* Hide Streamlit's auto-generated page nav */
[data-testid="stSidebarNav"] { display: none !important; }

/* Restore Material Symbols font for Streamlit's sidebar collapse/expand button */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {
    font-family: 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga';
}

/* ── Sidebar nav anchor hover ────────────────────────────────────────────── */
[data-testid="stSidebar"] a:hover {
    background: #DDE9F5 !important;
    text-decoration: none !important;
}

/* ── Section labels in sidebar ──────────────────────────────────────────── */
.sb-label {
    font-size: 9px;
    font-weight: 700;
    color: #8BAAC5;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    padding: 0 4px;
    margin-bottom: 4px;
}

/* ── Main content ───────────────────────────────────────────────────────── */
.sh {
    font-size: 10px;
    font-weight: 800;
    color: #0072CE;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    padding: 8px 0;
    border-bottom: 2px solid #EBF3FB;
    margin-bottom: 16px;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
}
.stButton button {
    background: #0072CE !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Montserrat', sans-serif !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 8px 18px !important;
    border-radius: 6px !important;
}
.stButton button:hover { background: #003467 !important; }

[data-baseweb="tab"] {
    font-family: 'Montserrat', sans-serif !important;
    font-weight: 600 !important;
    color: #6B8CAE !important;
    font-size: 12px !important;
}
[aria-selected="true"] {
    color: #0072CE !important;
    border-bottom-color: #0072CE !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #B0C8E0; border-radius: 10px; }
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


# dynamic_file_loader.py (the actual running Streamlit entrypoint — see
# analytics_app/dashboards/dynamic_file_loader.py) reads ?dashboard=/&page=
# query params, not URL paths — Streamlit's own pages/ auto-discovery can
# never see this app's sub-pages, since they're exec()'d inline rather
# than run as Streamlit's real multi-page app. Plain "/opd"-style hrefs
# used to be dead links for exactly that reason; every nav URL here must
# go through nav_url() instead.
DASHBOARD_ID = "facility_operations_dashboard"


def nav_url(page_id: str | None) -> str:
    """Build a loader-compatible URL. `page_id=None`/"overview" -> the
    dashboard's own home page; anything else -> that sub-page via ?page=."""
    if not page_id or page_id == "overview":
        return f"?dashboard={DASHBOARD_ID}"
    return f"?dashboard={DASHBOARD_ID}&page={page_id}"


_NAV = [
    # ── Home ────────────────────────────────────────────────────────────────
    ("overview",    "fa-solid fa-house",                  "Home",                   nav_url("overview")),
    # ── V2 Operational ──────────────────────────────────────────────────────
    (None,          None,                                 "V2 · Operational",       None),
    ("opd",         "fa-solid fa-user-clock",             "Patient Flow",           nav_url("opd")),
    ("dropoff",     "fa-solid fa-route",                  "Patient Drop-off",       nav_url("dropoff")),
    # ("capacity", "fa-solid fa-gauge-high", "Capacity Pressure", nav_url("capacity")),  # DEFERRED — awaiting denominators + service-level detail
    ("diagnostics", "fa-solid fa-microscope",             "Diagnostics",            nav_url("diagnostics")),
    ("pharmacy",    "fa-solid fa-pills",                  "Pharmacy",               nav_url("pharmacy")),
    ("admissions",  "fa-solid fa-bed-pulse",              "Admissions & Theatre",   nav_url("admissions")),
    ("physician",   "fa-solid fa-user-doctor",            "Physician Attribution",  nav_url("physician")),
    ("leakage",     "fa-solid fa-file-invoice-dollar",    "Ops Revenue",            nav_url("leakage")),
]


def render_sidebar(active: str = "overview", show_notify: bool = False):
    with st.sidebar:
        # ── Brand ────────────────────────────────────────────────────────────
        logo_col = st.columns([1, 2, 1])
        with logo_col[1]:
            if os.path.exists(_LOGO_PATH):
                st.image(_LOGO_PATH, use_container_width=True)

        st.markdown(
            '<div style="text-align:center;padding:10px 0 16px">'
            '<div style="font-size:9px;font-weight:700;color:#8BAAC5;'
            'text-transform:uppercase;letter-spacing:2px;margin-bottom:5px">'
            'St. Peter\'s Orthopedic Hospital</div>'
            '<div style="font-size:17px;font-weight:800;color:#003467;line-height:1.2">'
            'Ops Dashboard</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Notify ───────────────────────────────────────────────────────────
        if show_notify:
            _notices    = st.session_state.get("active_notices", [])
            _recipients = get_recipients()
            _n_count    = len(_notices)
            _dot_col    = "#E11D48" if _n_count else "#0BB99F"
            _status_txt = (
                f"{_n_count} notice{'s' if _n_count != 1 else ''} firing"
                if _n_count else "All clear"
            )
            st.markdown(
                f'<div class="sb-label" style="margin-bottom:4px">Notify</div>'
                f'<div style="display:flex;align-items:center;gap:6px;'
                f'padding:2px 4px 8px;font-size:11px;color:#003467">'
                f'<span style="color:{_dot_col};font-size:8px;line-height:1">&#9679;</span>'
                f'<span>{_status_txt}</span></div>',
                unsafe_allow_html=True,
            )
            if _n_count:
                if st.button("Send Executive Digest", use_container_width=True, key="send_digest_btn"):
                    if not _recipients:
                        st.error("Set DIGEST_RECIPIENTS in .env")
                    else:
                        _ok, _msg = send_digest("SPH", _notices)
                        st.success("Sent ✓") if _ok else st.error(f"Failed: {_msg}")
            st.markdown('<div style="margin-bottom:4px"></div>', unsafe_allow_html=True)

        # ── Navigation ───────────────────────────────────────────────────────
        st.markdown('<div class="sb-label" style="margin-bottom:6px">Navigate</div>',
                    unsafe_allow_html=True)

        items_html = '<div style="display:flex;flex-direction:column;gap:2px;margin-bottom:4px">'
        for page_id, icon, label, href in _NAV:
            if page_id is None:
                items_html += (
                    f'<div style="margin:10px 0 4px;padding:0 4px;font-size:9px;font-weight:700;'
                    f'color:#8BAAC5;text-transform:uppercase;letter-spacing:1.8px;'
                    f'border-top:1px solid #D6E4F0;padding-top:10px">{label}</div>'
                )
                continue
            is_active  = page_id == active
            is_disabled = href is None
            icon_color  = "#fff" if is_active else ("#8BAAC5" if is_disabled else "#0072CE")
            bg          = "background:#003467;" if is_active else ""
            txt_color   = "#fff" if is_active else ("#8BAAC5" if is_disabled else "#1E3A55")
            weight      = "800" if is_active else "600"
            opacity     = "opacity:0.45;" if is_disabled else ""
            cursor      = "cursor:default;" if (is_active or is_disabled) else "cursor:pointer;"

            inner = (
                f'<i class="{icon}" style="width:18px;text-align:center;'
                f'font-size:14px;color:{icon_color};flex-shrink:0"></i>'
                f'<span style="font-size:13px;font-weight:{weight};color:{txt_color}">{label}</span>'
            )
            if is_disabled:
                inner += (
                    '<span style="margin-left:auto;font-size:9px;background:#DDE9F5;'
                    'color:#8BAAC5;padding:2px 6px;border-radius:4px;font-weight:700">SOON</span>'
                )

            row_style = (
                f'display:flex;align-items:center;gap:12px;padding:10px 14px;'
                f'border-radius:8px;text-decoration:none;{bg}{opacity}{cursor}'
            )

            if is_active or is_disabled:
                items_html += f'<div style="{row_style}">{inner}</div>'
            else:
                items_html += (
                    f'<a href="{href}" target="_self" style="{row_style}">'
                    f'{inner}</a>'
                )

        items_html += '</div>'
        st.markdown(items_html, unsafe_allow_html=True)

        # ── Data scope ───────────────────────────────────────────────────────
        st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)
        st.divider()
        st.markdown(
            '<div class="sb-label">Data Scope</div>'
            '<div style="font-size:10px;color:#6B8CAE;line-height:1.85;padding:2px 4px 0">'
            '<b style="color:#003467">V1 · Jun 2022 – Jan 2025</b><br>'
            '32 months &nbsp;·&nbsp; 35,000+ patients<br>'
            '<b style="color:#0072CE">V2 · Feb 2025 – present</b><br>'
            '14 months &nbsp;·&nbsp; operational'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Abbreviations ────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            '<div class="sb-label">Abbreviations</div>'
            '<div style="font-size:10px;color:#6B8CAE;line-height:1.85;padding:2px 4px 0">'
            '<b style="color:#003467">OPD</b> — Outpatient Department<br>'
            '<b style="color:#003467">IPD / ADM</b> — Inpatient Admission<br>'
            '<b style="color:#003467">TAT</b> — Turnaround Time<br>'
            '<b style="color:#003467">LOS</b> — Length of Stay<br>'
            '<b style="color:#003467">USS</b> — Ultrasound<br>'
            '<b style="color:#003467">Median</b> — Middle value; half of patients wait less than this<br>'
            '<b style="color:#003467">V1 / V2</b> — EMR system generation<br>'
            '<b style="color:#003467">pp</b> — Percentage points<br>'
            '<b style="color:#003467">MoM / YoY</b> — Month-on-Month / Year-on-Year'
            '</div>',
            unsafe_allow_html=True,
        )


def cl(**kw):
    return {**CHART_LAYOUT, **kw}


def kpi_card(label, value, sub="", color="#003467", icon=""):
    _accent = {COLORS["danger"], COLORS["warning"], COLORS["success"]}
    bl = f"border-left:4px solid {color};" if color in _accent else ""
    icon_html = f'<span style="font-size:13px;margin-right:5px">{icon}</span>' if icon else ""
    st.markdown(
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;border-radius:8px;'
        f'padding:24px 20px;{bl}">'
        f'<div style="font-size:10px;font-weight:700;color:#6B8CAE;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">{icon_html}{label}</div>'
        f'<div style="font-size:42px;font-weight:800;color:{color};line-height:1">{value}</div>'
        f'<div style="font-size:12px;color:#6B8CAE;margin-top:8px">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)


def info_card(text, border_color="#0072CE"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;border-left:3px solid {border_color};'
        f'border-radius:4px;font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
        unsafe_allow_html=True,
    )


_MODE_META = {
    "historical_baseline": {
        "chip_bg":    "#FFF7ED",
        "chip_border": "#D97706",
        "chip_color":  "#D97706",
        "chip_label":  "Historical Baseline",
        "notice":      "Not live operational monitoring",
    },
    "live": {
        "chip_bg":    "#F0FBF8",
        "chip_border": "#0BB99F",
        "chip_color":  "#0BB99F",
        "chip_label":  "Live Operations",
        "notice":      "Operational · V2 data",
    },
    "snapshot": {
        "chip_bg":    "#EFF6FF",
        "chip_border": "#0072CE",
        "chip_color":  "#0072CE",
        "chip_label":  "Monthly Snapshot",
        "notice":      "Point-in-time view",
    },
}


def page_header(title, subtitle=None, period=None, mode=None, center=False):
    """Standard page identity block — title, subtitle, separator."""
    _align = "center" if center else "left"
    sub_html = (
        f'<div style="font-size:12px;color:#6B8CAE;margin-top:4px;text-align:{_align}">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:8px;text-align:{_align}">'
        f'  <div style="font-size:9px;font-weight:700;color:#6B8CAE;'
        f'       text-transform:uppercase;letter-spacing:2px">'
        f'    St. Peter\'s Orthopedic Hospital</div>'
        f'  <div style="font-size:24px;font-weight:800;color:#003467;margin-top:4px">{title}</div>'
        f'  {sub_html}'
        f'</div>'
        f'<div style="border-bottom:1px solid #EBF3FB;margin:16px 0 24px"></div>',
        unsafe_allow_html=True,
    )


def insight_panel(title, items, footer=None, border_color=None):
    """Titled interpretation box — executive synthesis, analytical boundaries, forward questions.
    items: list of str  OR  list of (text, source) tuples.
    Tuple form renders source attribution right-aligned as '→ Source'.
    """
    if border_color is None:
        border_color = COLORS["primary"]
    _items_html = ""
    for item in items:
        if isinstance(item, tuple):
            text, source = item
            _items_html += (
                f'<li style="margin-bottom:10px;display:flex;justify-content:space-between;'
                f'align-items:flex-start;gap:12px">'
                f'  <span style="color:#003467">{text}</span>'
                f'  <span style="font-size:10px;color:#9BAEC8;white-space:nowrap;'
                f'       margin-top:2px;flex-shrink:0">→ {source}</span>'
                f'</li>'
            )
        else:
            _items_html += (
                f'<li style="margin-bottom:10px;color:#003467">{item}</li>'
            )
    _footer_html = (
        f'<div style="font-size:11px;color:#6B8CAE;border-top:1px solid #EBF3FB;'
        f'padding-top:10px;margin-top:4px;font-style:italic">{footer}</div>'
        if footer else ""
    )
    st.markdown(
        f'<div style="background:#F8FBFE;border:1px solid #D6E4F0;'
        f'border-left:3px solid {border_color};border-radius:6px;'
        f'padding:18px 20px;margin:16px 0">'
        f'  <div style="font-size:10px;font-weight:800;color:{border_color};'
        f'       text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px">'
        f'    {title}</div>'
        f'  <ul style="margin:0;padding-left:16px;font-size:13px;line-height:1.7">'
        f'  {_items_html}</ul>'
        f'  {_footer_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def dq_note(text):
    st.markdown(
        f'<div style="background:#F4F8FC;border-left:3px solid #B0C8E0;border-radius:4px;'
        f'padding:8px 12px;margin:10px 0;font-size:12px;color:#003467;line-height:1.5">'
        f'<span style="font-weight:700;color:#6B8CAE">Data note · </span>{text}</div>',
        unsafe_allow_html=True,
    )


def _dot(series, higher_is_good=True, n=3, label="vs prior period"):
    if series is None:
        return ""
    vals = pd.Series(series).dropna().values
    if len(vals) < n + 1:
        return ""
    recent = vals[-n:].mean()
    prior  = vals[-n * 2:-n].mean() if len(vals) >= n * 2 else vals[: len(vals) - n].mean()
    if abs(prior) < 1e-10:
        return ""
    pct   = (recent - prior) / abs(prior) * 100
    is_up = pct >= 0
    is_good = is_up == higher_is_good
    clr   = COLORS["success"] if is_good else COLORS["danger"]
    arrow = "▲" if is_up else "▼"
    return f'<span style="color:{clr};font-size:10px">{arrow} {abs(pct):.1f}% {label}</span>'


STATUS_BG     = {"GREEN": "#F0FBF8", "AMBER": "#FFFBEB", "RED": "#FFF1F3"}
STATUS_BORDER = {"GREEN": "#0BB99F", "AMBER": "#D97706", "RED": "#E11D48"}
STATUS_LABEL  = {"GREEN": "ALL CLEAR", "AMBER": "WATCH", "RED": "ALERT"}
STATUS_EMOJI  = {"GREEN": "✅", "AMBER": "⚠️", "RED": "🔴"}


def pulse_card(icon, domain, status, message):
    """Operational Pulse domain card — GREEN / AMBER / RED."""
    _dc  = STATUS_BORDER[status]
    _dbg = STATUS_BG[status]
    _dlbl = STATUS_LABEL[status]
    _de   = STATUS_EMOJI[status]
    st.markdown(
        f'<div style="background:{_dbg};border:1px solid {_dc}40;border-top:4px solid {_dc};'
        f'border-radius:10px;padding:20px 18px 16px;min-height:130px;'
        f'box-shadow:0 2px 8px {_dc}18">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">'
        f'<i class="{icon}" style="font-size:26px;color:{_dc};opacity:0.9"></i>'
        f'<span style="font-size:22px;line-height:1">{_de}</span></div>'
        f'<div style="font-size:13px;font-weight:800;color:#003467;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:8px">{domain}</div>'
        f'<div style="margin-bottom:10px">'
        f'<span style="background:{_dc};color:#fff;font-size:10px;font-weight:800;'
        f'letter-spacing:1.2px;padding:3px 9px;border-radius:4px">{_dlbl}</span></div>'
        f'<div style="font-size:12px;color:#003467;line-height:1.55;opacity:0.85">{message}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def notice_card(severity, title, value, delta_line, implication, color):
    """Active alert card — fires when a threshold rule is breached."""
    _badge_bg = COLORS["danger"] if severity == "CRITICAL" else COLORS["warning"]
    st.markdown(
        f'<div style="background:#fff;border:1px solid #D6E4F0;border-left:4px solid {color};'
        f'border-radius:8px;padding:14px 16px;margin-bottom:12px">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
        f'<span style="background:{_badge_bg};color:#fff;font-size:9px;font-weight:800;'
        f'letter-spacing:1.5px;padding:2px 7px;border-radius:3px">{severity}</span>'
        f'<span style="font-size:11px;font-weight:700;color:#003467;text-transform:uppercase;'
        f'letter-spacing:0.8px">{title}</span></div>'
        f'<div style="font-size:22px;font-weight:800;color:{color};line-height:1.2">{value}</div>'
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">{delta_line}</div>'
        f'<div style="font-size:11px;color:#003467;margin-top:6px;line-height:1.5;'
        f'border-top:1px solid #EBF3FB;padding-top:6px">{implication}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _add_rolling_mean(fig, x_series, y_series, n=3, name="3-mo avg", color=None, dash="dot"):
    if color is None:
        color = COLORS["muted"]
    roll = pd.Series(
        y_series.values if hasattr(y_series, "values") else y_series
    ).rolling(n, min_periods=2).mean()
    fig.add_scatter(
        x=x_series, y=roll,
        mode="lines", name=name,
        line=dict(color=color, width=2, dash=dash),
        hovertemplate=f"<b>{name}</b>: %{{y:.1f}}<extra></extra>",
    )
