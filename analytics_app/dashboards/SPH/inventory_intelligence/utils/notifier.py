"""Daily digest notifier — emails the day's stock-availability signals.

Reads the pre-computed analytics tables and sends an HTML + plaintext email
summarising what is stocked out, what is about to run out, and what to order —
mirroring the sibling KSH inventory digest. Delivery is Django email; all
config comes from the environment.

Trigger it from a scheduler (cron / management command) or run this file
directly:  ``python inventory_intelligence/utils/notifier.py``.

Environment:
  EMAIL_HOST / EMAIL_PORT / EMAIL_USE_TLS / EMAIL_HOST_USER / EMAIL_HOST_PASSWORD
  NOTIFY_EMAIL_FROM   sender address (default noreply@afyaanalytics.com)
  NOTIFY_EMAIL_TO     comma-separated recipients (required, else no-op)
  DASHBOARD_URL       optional link shown as the call-to-action
"""
from __future__ import annotations

import html
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from django.conf import settings as dj_settings
from django.core.mail import EmailMultiAlternatives

FACILITY_NAME = "St. Peter's Orthopaedic"
FACILITY_SLUG = "sph"
_ACCENT = "#0F6E56"


# ── Config ────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _ensure_email_configured() -> None:
    if dj_settings.configured:
        return
    if os.environ.get("DJANGO_SETTINGS_MODULE"):
        import django
        django.setup()
        return
    dj_settings.configure(
        INSTALLED_APPS=[],
        EMAIL_BACKEND="django.core.mail.backends.smtp.EmailBackend",
        EMAIL_HOST=_env("EMAIL_HOST", "smtp.gmail.com"),
        EMAIL_PORT=int(_env("EMAIL_PORT", "587")),
        EMAIL_USE_TLS=_env("EMAIL_USE_TLS", "true").lower() != "false",
        EMAIL_HOST_USER=_env("EMAIL_HOST_USER"),
        EMAIL_HOST_PASSWORD=_env("EMAIL_HOST_PASSWORD"),
        DEFAULT_FROM_EMAIL=_env("NOTIFY_EMAIL_FROM", "noreply@afyaanalytics.com"),
    )


def _already_sent_today() -> bool:
    stamp = Path(tempfile.gettempdir()) / f"afya_digest_{FACILITY_SLUG}_{date.today()}.sent"
    return stamp.exists()


def _mark_sent_today() -> None:
    (Path(tempfile.gettempdir()) / f"afya_digest_{FACILITY_SLUG}_{date.today()}.sent").touch()


# ── Signals ───────────────────────────────────────────────────────────────────

def gather_signals() -> dict:
    """Compute the availability signals from the pre-computed tables.

    Returns counts (stocked_out / critical / low / order_now) and a ranked list
    of the most urgent items, each with a plain-language headline and action.
    """
    from inventory_intelligence.dashboard import data_access

    stockout = data_access.load_table("stockout_risk")
    if stockout is None or stockout.empty:
        return {}
    stockout = stockout.copy()
    stockout["item_key"] = stockout["item_key"].astype(str)
    for c in ("p_stockout_30", "days_to_stockout_med", "soh"):
        if c in stockout.columns:
            stockout[c] = pd.to_numeric(stockout[c], errors="coerce")

    try:
        names = data_access.item_lookup("SPH")[["item_key", "display_name"]]
        df = stockout.merge(names, on="item_key", how="left")
        df["display_name"] = df["display_name"].fillna(df["item_key"])
    except Exception:
        df = stockout.assign(display_name=stockout["item_key"])

    soh = df.get("soh")
    p30 = df.get("p_stockout_30")
    days = df.get("days_to_stockout_med")

    stocked_out = df[(soh <= 0) | (p30 >= 0.99)]
    live = df[soh > 0] if soh is not None else df
    critical = live[live["days_to_stockout_med"] <= 7] if days is not None else live.iloc[0:0]
    low = live[(live["days_to_stockout_med"] > 7) & (live["days_to_stockout_med"] <= 30)] \
        if days is not None else live.iloc[0:0]
    order_now = int((p30 > 0.5).sum()) if p30 is not None else 0

    ranked = df.sort_values("p_stockout_30", ascending=False)
    signals = []
    for _, r in ranked.head(6).iterrows():
        p = r.get("p_stockout_30")
        d = r.get("days_to_stockout_med")
        s = r.get("soh")
        if pd.isna(p) or p < 0.5:
            continue
        if pd.notna(s) and s <= 0:
            sev, when = "CRITICAL", "already out of stock"
        elif pd.notna(d) and d <= 7:
            sev, when = "CRITICAL", f"~{int(d)} day(s) of stock left"
        elif pd.notna(d) and d <= 30:
            sev, when = "HIGH", f"~{int(d)} days of stock left"
        else:
            sev, when = "HIGH", "running low"
        signals.append({
            "name": str(r["display_name"]),
            "severity": sev,
            "headline": f"{r['display_name']} — {when}",
            "fact": f"{p * 100:.0f}% modelled chance of running out within a month",
            "action": "Order now" if sev == "CRITICAL" else "Order this week",
        })

    return {
        "stocked_out": int(len(stocked_out)),
        "critical": int(len(critical)),
        "low": int(len(low)),
        "order_now": order_now,
        "signals": signals,
    }


# ── Email bodies ──────────────────────────────────────────────────────────────

_SEV = {"CRITICAL": ("#FEE2E2", "#DC2626", "#991B1B"),
        "HIGH": ("#FEF3C7", "#D97706", "#92400E")}


def _signal_card(s: dict) -> str:
    bg, border, text = _SEV.get(s["severity"], ("#F9FAFB", "#9CA3AF", "#374151"))
    return (
        f'<div style="border-left:4px solid {border};border-radius:0 8px 8px 0;'
        f'padding:12px 16px;margin:8px 0;background:{bg}">'
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:{text};margin-bottom:4px">{s["severity"]}</div>'
        f'<div style="font-weight:700;font-size:13px;color:#111827;margin-bottom:4px">'
        f'{html.escape(s["headline"])}</div>'
        f'<div style="font-size:12px;color:#6B7280;margin-bottom:8px">{html.escape(s["fact"])}</div>'
        f'<span style="display:inline-block;font-size:10px;font-weight:700;padding:3px 10px;'
        f'border-radius:4px;background:#F0FDF4;color:#166534;border:1px solid #86EFAC;'
        f'text-transform:uppercase;letter-spacing:0.05em">{html.escape(s["action"])}</span>'
        f'</div>'
    )


def _kpi_cell(value: int, label: str, alarm: bool = False) -> str:
    color = "#DC2626" if alarm and value else "#111827"
    return (
        f'<td style="text-align:center;padding:10px;border:1px solid #E5E7EB;border-radius:6px">'
        f'<div style="font-size:22px;font-weight:700;color:{color}">{value}</div>'
        f'<div style="font-size:10px;color:#9CA3AF;text-transform:uppercase">{label}</div></td>'
    )


def build_html(sig: dict, dashboard_url: str, today: date) -> str:
    cards = "".join(_signal_card(s) for s in sig.get("signals", [])) or (
        '<p style="color:#6B7280;font-size:13px">No items at risk today — stock cover looks healthy.</p>')
    kpis = (
        '<table style="width:100%;border-collapse:collapse;margin:12px 0"><tr>'
        + _kpi_cell(sig.get("stocked_out", 0), "Stocked out", alarm=True)
        + _kpi_cell(sig.get("critical", 0), "Critical &lt;7d", alarm=True)
        + _kpi_cell(sig.get("low", 0), "Low 7–30d")
        + _kpi_cell(sig.get("order_now", 0), "Order now")
        + '</tr></table>'
    )
    cta = (
        f'<div style="text-align:center;margin:20px 0"><a href="{html.escape(dashboard_url)}" '
        f'style="display:inline-block;background:{_ACCENT};color:#FFF;font-weight:700;font-size:13px;'
        f'padding:12px 28px;border-radius:6px;text-decoration:none">Open dashboard →</a></div>'
        if dashboard_url else "")
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;background:#F3F4F6;font-family:'Helvetica Neue',Arial,sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" bgcolor="#F3F4F6"><tr>
<td align="center" style="padding:32px 16px">
<table width="600" cellpadding="0" cellspacing="0"
       style="background:#FFF;border-radius:12px;border:1px solid #E5E7EB;overflow:hidden">
<tr><td style="background:{_ACCENT};padding:20px 28px">
  <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;color:#A7F3D0">
    SPH Inventory Intelligence</div>
  <div style="font-size:20px;font-weight:800;color:#FFF;margin:4px 0 2px">Daily digest</div>
  <div style="font-size:12px;color:#6EE7B7">{FACILITY_NAME} · {today.strftime('%A, %d %b %Y')}</div>
</td></tr>
<tr><td style="padding:20px 28px 0">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;
              color:#9CA3AF;margin-bottom:6px">Today's snapshot</div>{kpis}</td></tr>
<tr><td style="padding:16px 28px 0">
  <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;
              color:#9CA3AF;margin-bottom:4px">Most urgent</div>{cards}</td></tr>
<tr><td style="padding:8px 28px 24px">{cta}</td></tr>
<tr><td style="background:#F9FAFB;padding:14px 28px;border-top:1px solid #E5E7EB">
  <div style="font-size:10px;color:#9CA3AF">Sent by SPH Inventory Intelligence · {today.strftime('%d %b %Y')}
  · Modelled from pre-computed data. Do not reply.</div></td></tr>
</table></td></tr></table></body></html>"""


def build_plaintext(sig: dict, dashboard_url: str, today: date) -> str:
    lines = [
        "SPH INVENTORY — DAILY DIGEST",
        f"{FACILITY_NAME}  |  {today.strftime('%d %b %Y')}", "",
        "SNAPSHOT",
        f"  Stocked out : {sig.get('stocked_out', 0)}",
        f"  Critical <7d: {sig.get('critical', 0)}",
        f"  Low 7-30d   : {sig.get('low', 0)}",
        f"  Order now   : {sig.get('order_now', 0)}", "",
        "MOST URGENT",
    ]
    for i, s in enumerate(sig.get("signals", []), 1):
        lines.append(f"  {i}. [{s['severity']}] {s['headline']}")
        lines.append(f"     {s['fact']} -> {s['action']}")
    if not sig.get("signals"):
        lines.append("  No items at risk today.")
    if dashboard_url:
        lines += ["", f"Open dashboard: {dashboard_url}"]
    lines += ["", "-- SPH Inventory Intelligence · modelled from pre-computed data."]
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def send_daily_digest(force: bool = False) -> bool:
    """Send the digest email. Returns True if sent, False if skipped.

    Skips when already sent today (unless ``force``) or when NOTIFY_EMAIL_TO is
    unset. Raises on SMTP delivery errors.
    """
    _ensure_email_configured()
    if not force and _already_sent_today():
        return False

    recipients = [r.strip() for r in _env("NOTIFY_EMAIL_TO").split(",") if r.strip()]
    if not recipients:
        return False

    sig = gather_signals()
    if not sig:
        return False

    today = date.today()
    dashboard_url = _env("DASHBOARD_URL")
    subject = (f"[SPH] Daily digest — {sig.get('stocked_out', 0)} out, "
               f"{sig.get('critical', 0)} critical — {today.strftime('%d %b %Y')}")

    msg = EmailMultiAlternatives(
        subject, build_plaintext(sig, dashboard_url, today),
        getattr(dj_settings, "DEFAULT_FROM_EMAIL", "noreply@afyaanalytics.com"), recipients)
    msg.attach_alternative(build_html(sig, dashboard_url, today), "text/html")
    msg.send()

    _mark_sent_today()
    return True


if __name__ == "__main__":
    sent = send_daily_digest(force=True)
    print("Digest sent." if sent else "Digest skipped (no recipients set, or nothing to send).")
