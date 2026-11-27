"""
KSH V3 Operations Dashboard — theme, sidebar, layout helpers.
Adapted from SPH (Orthodata/dashboard/theme.py). KSH teal: #0F6E56.
"""

from pathlib import Path

import pandas as pd
import streamlit as st
from ksh_operations.dashboard.notifier import send_digest, get_recipients

_LOGO = Path(__file__).parent / "ksh_logo.png"

COLORS = {
    "primary":   "#0072CE",   # Afya Blue
    "secondary": "#0BB99F",   # Insight Teal
    "dark":      "#003467",   # Cool Blue — text & data viz
    "muted":     "#6B8CAE",   # neutral grey-blue
    "warning":   "#D97706",   # functional only — alert states
    "danger":    "#E11D48",   # functional only — alert states
    "success":   "#0BB99F",   # maps to Insight Teal
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

[data-testid="stSidebar"] {
    background: #F0F5FA !important;
    border-right: 1px solid #D6E4F0 !important;
}
[data-testid="stSidebar"] *:not(i) {
    font-family: 'Montserrat', sans-serif !important;
}

/* Hide Streamlit's auto-generated page nav */
[data-testid="stSidebarNav"] { display: none !important; }

[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {
    font-family: 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga';
}

[data-testid="stSidebar"] a:hover {
    background: #DDE9F5 !important;
    text-decoration: none !important;
}

.sb-label {
    font-size: 9px;
    font-weight: 700;
    color: #8BAAC5;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    padding: 0 4px;
    margin-bottom: 4px;
}

.sh {
    font-size: 10px;
    font-weight: 800;
    color: #0F6E56;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    padding: 8px 0;
    border-bottom: 2px solid #EBF3FB;
    margin-bottom: 16px;
}

.stButton button {
    background: #0F6E56 !important;
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
    color: #0F6E56 !important;
    border-bottom-color: #0F6E56 !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #B0C8E0; border-radius: 10px; }
</style>
"""

_NAV = [
    ("overview",  "fa-solid fa-house",           "Home",          "/"),
    (None,        None,                           "Modules",       None),
    ("opd",       "fa-solid fa-user-clock",       "OPD Flow",      "/1_OPD_Flow"),
    ("lab",       "fa-solid fa-flask",            "LAB",           "/2_Lab_TAT"),
    ("pharmacy",  "fa-solid fa-pills",            "Pharmacy TAT",  "/3_Pharmacy_TAT"),
    ("inpatient", "fa-solid fa-bed-pulse",        "Inpatient",     "/4_Inpatient"),
    ("theatre",   "fa-solid fa-scalpel",          "Theatre",       "/5_Theatre"),
    ("clinician", "fa-solid fa-user-doctor",      "Clinician",     "/6_Clinician"),
]


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar(active: str = "overview", show_notify: bool = False):
    with st.sidebar:
        if _LOGO.exists():
            st.image(str(_LOGO), width=140)
        st.markdown(
            '<div style="text-align:center;padding:16px 0 20px">'
            '<div style="font-size:9px;font-weight:700;color:#8BAAC5;'
            'text-transform:uppercase;letter-spacing:2px;margin-bottom:5px">'
            'Kisumu Specialist Hospital</div>'
            '<div style="font-size:17px;font-weight:800;color:#0F6E56;line-height:1.2">'
            'Ops Dashboard</div>'
            '<div style="font-size:10px;color:#9BAEC8;margin-top:3px">V3 · Operations</div>'
            '</div>',
            unsafe_allow_html=True,
        )

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
            is_active   = page_id == active
            is_disabled = href is None
            icon_color  = "#fff"    if is_active else ("#8BAAC5" if is_disabled else "#0F6E56")
            bg          = "background:#0F6E56;" if is_active else ""
            txt_color   = "#fff"    if is_active else ("#8BAAC5" if is_disabled else "#1E3A55")
            weight      = "800"     if is_active else "600"
            opacity     = "opacity:0.45;" if is_disabled else ""
            cursor      = "cursor:default;" if (is_active or is_disabled) else "cursor:pointer;"

            inner = (
                f'<i class="{icon}" style="width:18px;text-align:center;'
                f'font-size:14px;color:{icon_color};flex-shrink:0"></i>'
                f'<span style="font-size:13px;font-weight:{weight};color:{txt_color}">{label}</span>'
            )

            row_style = (
                f'display:flex;align-items:center;gap:12px;padding:10px 14px;'
                f'border-radius:8px;text-decoration:none;{bg}{opacity}{cursor}'
            )

            if is_active or is_disabled:
                items_html += f'<div style="{row_style}">{inner}</div>'
            else:
                items_html += f'<a href="{href}" target="_self" style="{row_style}">{inner}</a>'

        items_html += '</div>'
        st.markdown(items_html, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:8px"></div>', unsafe_allow_html=True)

        if show_notify:
            _notices    = st.session_state.get("active_notices", [])
            _recipients = get_recipients()
            _n_count    = len(_notices)
            _dot_col    = "#E11D48" if _n_count else "#0BB99F"
            _status_txt = f"{_n_count} active notice{'s' if _n_count != 1 else ''}" if _n_count else "All clear"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:6px 4px 8px">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{_dot_col};'
                f'display:inline-block;flex-shrink:0"></span>'
                f'<span style="font-size:11px;color:#6B8CAE">{_status_txt}</span></div>',
                unsafe_allow_html=True,
            )
            if _n_count:
                if st.button("Send Executive Digest", use_container_width=True, key="send_digest_btn"):
                    if not _recipients:
                        st.error("Set DIGEST_RECIPIENTS in .env")
                    else:
                        _ok, _msg = send_digest("KSH", _notices)
                        st.success("Sent ✓") if _ok else st.error(f"Failed: {_msg}")
            st.markdown('<div style="margin-bottom:4px"></div>', unsafe_allow_html=True)

        st.divider()
        st.markdown(
            '<div class="sb-label">Abbreviations</div>'
            '<div style="font-size:10px;color:#6B8CAE;line-height:1.85;padding:2px 4px 0">'
            '<b style="color:#003467">OPD</b> — Outpatient Department<br>'
            '<b style="color:#003467">IPD</b> — Inpatient Admission<br>'
            '<b style="color:#003467">TAT</b> — Turnaround Time<br>'
            '<b style="color:#003467">LOS</b> — Length of Stay<br>'
            '<b style="color:#003467">ALOS</b> — Average Length of Stay<br>'
            '<b style="color:#003467">BOR</b> — Bed Occupancy Rate<br>'
            '<b style="color:#003467">P50 / P90</b> — Median / 90th percentile wait<br>'
            '<b style="color:#003467">pp</b> — Percentage points<br>'
            '<b style="color:#003467">Signal</b> — Strong / Indicative / Descriptive'
            '</div>',
            unsafe_allow_html=True,
        )


def section_header(text, margin_top=0):
    style = f"margin-top:{margin_top}px" if margin_top else ""
    st.markdown(f'<div class="sh" style="{style}">{text}</div>', unsafe_allow_html=True)


def page_header(title, subtitle=None):
    sub_html = (
        f'<div style="font-size:12px;color:#6B8CAE;margin-top:4px">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="margin-bottom:8px">'
        f'  <div style="font-size:9px;font-weight:700;color:#6B8CAE;'
        f'       text-transform:uppercase;letter-spacing:2px">'
        f'    Kisumu Specialist Hospital</div>'
        f'  <div style="font-size:24px;font-weight:800;color:#003467;margin-top:4px">{title}</div>'
        f'  {sub_html}'
        f'</div>'
        f'<div style="border-bottom:1px solid #EBF3FB;margin:16px 0 24px"></div>',
        unsafe_allow_html=True,
    )


def info_card(text, border_color="#0F6E56"):
    st.markdown(
        f'<div style="padding:10px 14px;background:#F4F8FC;border-left:3px solid {border_color};'
        f'border-radius:4px;font-size:12px;color:#003467;margin-bottom:10px">{text}</div>',
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
    """Trend indicator — returns coloured HTML arrow + pct string."""
    vals = pd.Series(series).dropna().values
    if len(vals) < n + 1:
        return ""
    recent = vals[-n:].mean()
    prior  = vals[-n * 2:-n].mean() if len(vals) >= n * 2 else vals[: len(vals) - n].mean()
    if abs(prior) < 1e-10:
        return ""
    pct    = (recent - prior) / abs(prior) * 100
    is_up  = pct >= 0
    is_good = is_up == higher_is_good
    clr    = COLORS["success"] if is_good else COLORS["danger"]
    arrow  = "▲" if is_up else "▼"
    return f'<span style="color:{clr};font-size:10px">{arrow} {abs(pct):.1f}% {label}</span>'


def cl(**kw):
    """Merge extra kwargs into CHART_LAYOUT for plotly figures."""
    return {**CHART_LAYOUT, **kw}
