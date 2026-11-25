"""
Rendering functions for KSH V3 Operations Dashboard.

Pure HTML string builders — no Streamlit calls, no data loading.
Every function returns a string; the caller does st.markdown(..., unsafe_allow_html=True).
"""

import pandas as pd

# ── Brand & Status ─────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#0F6E56",
    "success": "#0F6E56",
    "warning": "#D97706",
    "danger":  "#E11D48",
    "muted":   "#6B8CAE",
    "dark":    "#003467",
}

STATUS_BG     = {"GREEN": "#F0FBF8", "AMBER": "#FFFBEB", "RED": "#FFF1F3", "GREY": "#F4F8FC"}
STATUS_BORDER = {"GREEN": "#0F6E56", "AMBER": "#D97706", "RED": "#E11D48", "GREY": "#9BAEC8"}
STATUS_LABEL  = {"GREEN": "ALL CLEAR", "AMBER": "WATCH", "RED": "ALERT",  "GREY": "NO STATUS"}
STATUS_EMOJI  = {"GREEN": "✅", "AMBER": "⚠️", "RED": "🔴", "GREY": "⬜"}

# ── Utility ────────────────────────────────────────────────────────────────────

def _safe(v, default=None):
    try:
        f = float(v)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default

# ── Alert Scoring ──────────────────────────────────────────────────────────────

def _ksh_severity(module: str, value: float) -> int:
    """Map module-specific delta to severity tier 1/2/3."""
    if module in ("opd", "lab", "pharmacy"):
        # value = pct_change (%)
        if value > 20:  return 3
        if value > 10:  return 2
        return 1
    if module == "inpatient":
        # value = alos_vs_baseline_days (positive = longer than baseline)
        if value > 1.0:  return 3
        if value > 0.5:  return 2
        return 1
    if module == "theatre":
        # value = completion_rate_vs_baseline_pp (negative = rate dropped; only score drops)
        v = abs(value) if value < 0 else 0
        if v > 10:  return 3
        if v > 5:   return 2
        return 1
    return 1


def _alert_score(severity: int, pct_change_magnitude: float, coverage_pct: float) -> float:
    """
    score = severity × persistence × confidence
    severity            : 1/2/3 from _ksh_severity()
    pct_change_magnitude: absolute value of the change
    coverage_pct        : 0–100
    CRITICAL >= 2.0, WATCH >= 1.2 (see DASHBOARD_BUILD.md §4)
    """
    pers = 1.3 if pct_change_magnitude > 10 else (1.1 if pct_change_magnitude > 0 else 1.0)
    conf = _safe(coverage_pct, 0.0) / 100.0
    return round(severity * pers * conf, 2)

# ── Flow Chain ─────────────────────────────────────────────────────────────────

_h_arrow_html = (
    '<div style="padding-top:28px;text-align:center;color:#C5D8EC;'
    'font-size:28px;font-weight:300;line-height:1">›</div>'
)


def _flow_card_html(title, metric_value, metric_label,
                    badge_text=None, badge_color=None, is_pending=False):
    """Horizontal grid flow card — center-aligned, top-border accent."""
    if badge_text and badge_color:
        badge_html = (
            f'<div style="margin-top:10px">'
            f'<span style="background:{badge_color}18;border:1px solid {badge_color}50;'
            f'color:{badge_color};font-size:10px;font-weight:700;padding:3px 10px;'
            f'border-radius:12px">{badge_text}</span></div>'
        )
        top_border = f"border-top:4px solid {badge_color}"
    elif is_pending:
        badge_html = (
            '<div style="margin-top:10px">'
            '<span style="background:#F4F8FC;border:1px solid #C5D8EC;color:#9BAEC8;'
            'font-size:10px;font-weight:600;padding:3px 10px;border-radius:12px">'
            'Data pending</span></div>'
        )
        top_border = "border-top:4px solid #D6E4F0"
    else:
        badge_html = ""
        top_border = "border-top:4px solid #D6E4F0"

    return (
        f'<div style="background:#F4F8FC;border:1px solid #D6E4F0;{top_border};'
        f'border-radius:10px;padding:24px 14px 20px;text-align:center;min-height:180px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">{title}</div>'
        f'<div style="font-size:30px;font-weight:800;color:#003467;line-height:1.1">{metric_value}</div>'
        f'<div style="font-size:10px;color:#9BAEC8;margin-top:6px">{metric_label}</div>'
        f'{badge_html}'
        f'</div>'
    )

# ── Domain / Alert Cards ───────────────────────────────────────────────────────

def _domain_card(domain, icon, status, urgency_label, urgency_color, story, metrics, href, href_label, show_label=True):
    """Alert card with RAG badge. status: 'RED' | 'AMBER' | 'GREEN' | 'GREY'."""
    dc   = STATUS_BORDER[status]
    dbg  = STATUS_BG[status]
    dlbl = STATUS_LABEL[status]
    mhtml = "".join(
        f'<div style="font-size:11px;color:#6B8CAE;margin-top:4px">{m}</div>'
        for m in metrics if m
    )
    _lbl_html = (
        f'<span style="background:{dc};color:#fff;font-size:10px;font-weight:800;'
        f'letter-spacing:1.2px;padding:3px 9px;border-radius:4px">{dlbl}</span>'
    ) if show_label else ''
    return (
        f'<div style="background:{dbg};border:1px solid {dc}40;border-top:4px solid {dc};'
        f'border-radius:10px;padding:20px 18px 16px;min-height:220px">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">'
        f'<div>'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:6px">'
        f'<i class="{icon}" style="margin-right:5px"></i>{domain}</div>'
        f'{_lbl_html}'
        f'</div>'
        f'</div>'
        f'<div style="margin:10px 0 4px">'
        f'<span style="background:{urgency_color}15;border:1px solid {urgency_color}40;'
        f'color:{urgency_color};font-size:10px;font-weight:700;padding:3px 10px;border-radius:12px">'
        f'{urgency_label}</span>'
        f'</div>'
        f'<div style="font-size:14px;font-weight:700;color:#003467;line-height:1.5;margin:10px 0 6px">'
        f'{story}</div>'
        f'{mhtml}'
        f'<div style="margin-top:14px;border-top:1px solid {dc}30;padding-top:10px">'
        f'<a href="{href}" target="_self" style="font-size:12px;color:#0F6E56;font-weight:700;'
        f'text-decoration:none">→ {href_label}</a>'
        f'</div>'
        f'</div>'
    )


def _pulse_card(domain, icon, tat_value, tat_label, narrative, href, href_label, unit="min"):
    """Service metric card — no RAG badge, no validated scoring model for this domain."""
    val_s = f"{int(tat_value)}{' ' + unit if unit else ''}" if tat_value is not None else "—"
    return (
        f'<div style="background:#FAFCFF;border:1px solid #D6E4F0;border-top:4px solid #D6E4F0;'
        f'border-radius:10px;padding:20px 18px 16px;min-height:200px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">'
        f'<i class="{icon}" style="margin-right:5px"></i>{domain}</div>'
        f'<div style="font-size:32px;font-weight:800;color:#003467;line-height:1">{val_s}</div>'
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:2px">{tat_label}</div>'
        f'<div style="font-size:13px;color:#6B8CAE;margin-top:10px;line-height:1.4">{narrative}</div>'
        f'<div style="margin-top:14px;border-top:1px solid #D6E4F080;padding-top:10px">'
        f'<a href="{href}" target="_self" style="font-size:12px;color:#0F6E56;font-weight:700;'
        f'text-decoration:none">→ {href_label}</a>'
        f'</div>'
        f'</div>'
    )


def _monitor_arrow(current, prior, flat_band):
    """(arrow_char, arrow_color) for a higher_is_bad metric. '' if no prior."""
    if prior is None or current is None:
        return "", "#9BAEC8"
    delta = current - prior
    if abs(delta) <= flat_band:
        return "→", "#9BAEC8"
    return ("↑", "#B45309") if delta > 0 else ("↓", "#0F6E56")


def _monitor_card(domain, icon, value_str, arrow, arrow_color, delta_str,
                  period_str, threshold_ref, href, href_label):
    """Always-on operational monitor card — no RAG badge, shows proximity to threshold."""
    trend_html = (
        f'<span style="font-size:13px;font-weight:700;color:{arrow_color};margin-left:8px">'
        f'{arrow} {delta_str}</span>'
    ) if arrow else ""
    return (
        f'<div style="background:#FAFCFF;border:1px solid #D6E4F0;border-top:4px solid #D6E4F0;'
        f'border-radius:10px;padding:20px 18px 16px;min-height:190px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">'
        f'<i class="{icon}" style="margin-right:5px"></i>{domain}</div>'
        f'<div style="display:flex;align-items:baseline">'
        f'<div style="font-size:30px;font-weight:800;color:#003467;line-height:1">{value_str}</div>'
        f'{trend_html}</div>'
        f'<div style="font-size:10px;color:#9BAEC8;margin-top:4px">{period_str}</div>'
        f'<div style="font-size:11px;color:#C5D8EC;margin-top:8px">{threshold_ref}</div>'
        f'<div style="margin-top:14px;border-top:1px solid #D6E4F080;padding-top:10px">'
        f'<a href="{href}" target="_self" style="font-size:12px;color:#0F6E56;font-weight:700;'
        f'text-decoration:none">→ {href_label}</a>'
        f'</div>'
        f'</div>'
    )


def _signal_card(domain, icon, value, value_label, narrative, href, href_label, decimals=1):
    """Operational signal card — float value, no RAG badge, informational only."""
    val_s = f"{value:.{decimals}f}" if value is not None else "—"
    return (
        f'<div style="background:#FAFCFF;border:1px solid #D6E4F0;border-top:4px solid #D6E4F0;'
        f'border-radius:10px;padding:20px 18px 16px;min-height:200px">'
        f'<div style="font-size:9px;font-weight:700;color:#9BAEC8;text-transform:uppercase;'
        f'letter-spacing:1.5px;margin-bottom:10px">'
        f'<i class="{icon}" style="margin-right:5px"></i>{domain}</div>'
        f'<div style="font-size:32px;font-weight:800;color:#003467;line-height:1">{val_s}</div>'
        f'<div style="font-size:11px;color:#9BAEC8;margin-top:2px">{value_label}</div>'
        f'<div style="font-size:13px;color:#6B8CAE;margin-top:10px;line-height:1.4">{narrative}</div>'
        f'<div style="margin-top:14px;border-top:1px solid #D6E4F080;padding-top:10px">'
        f'<a href="{href}" target="_self" style="font-size:12px;color:#0F6E56;font-weight:700;'
        f'text-decoration:none">→ {href_label}</a>'
        f'</div>'
        f'</div>'
    )
