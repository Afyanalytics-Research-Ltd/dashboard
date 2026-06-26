"""
Phase 2.5 — Daily Digest delivery.

Sends a daily email digest (top insights + key KPIs + dashboard link) via SMTP.
WhatsApp delivery is deferred pending Meta Business API approval.

Configuration (add to .env):
    NOTIFY_EMAIL_TO      — recipient address(es), comma-separated
    NOTIFY_EMAIL_FROM    — sender address (e.g. alerts@afya.ai)
    SMTP_HOST            — SMTP server hostname
    SMTP_PORT            — SMTP port (default 587)
    SMTP_USER            — SMTP login username
    SMTP_PASSWORD        — SMTP login password
    DASHBOARD_URL        — public URL of the dashboard (for the CTA button)

Usage (standalone script or cron):
    from utils.notifier import send_daily_digest
    send_daily_digest(
        facility_name="Facility Name",
        insights=insight_rows,          # List[InsightRow] from insight_engine.detect_all()
        kpi=kpi_dict,                   # from get_kpi_summary()
        order_count=12,                 # ORDER_NOW count from score_all()
        patient_risk_count=8,           # total_patients_at_risk from get_patient_risk_totals()
    )

Design principles (from ROADMAP):
  - SMTP credentials loaded from .env — never hard-coded
  - LLM never touches raw data — all numbers pre-computed before this call
  - Sends at most once per facility per calendar day (idempotency guard via a local
    date-stamp file under /tmp — lightweight, no database required)
"""

from __future__ import annotations

import html
import os
import smtplib
import tempfile
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from intelligence.insight_engine import InsightRow


# ── Config from environment ───────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# ── Idempotency guard ─────────────────────────────────────────────────────────

def _already_sent_today(facility_slug: str) -> bool:
    """Return True if digest was already sent for this facility today."""
    stamp_path = Path(tempfile.gettempdir()) / f"afya_digest_{facility_slug}_{date.today()}.sent"
    return stamp_path.exists()


def _mark_sent_today(facility_slug: str) -> None:
    stamp_path = Path(tempfile.gettempdir()) / f"afya_digest_{facility_slug}_{date.today()}.sent"
    stamp_path.touch()


# ── HTML template ─────────────────────────────────────────────────────────────

_SEV_COLORS = {
    "CRITICAL": ("#FEE2E2", "#DC2626", "#991B1B"),
    "HIGH":     ("#FEF3C7", "#D97706", "#92400E"),
    "MEDIUM":   ("#F0FDF4", "#0F6E56", "#065F46"),
}


def _insight_block_html(row: "InsightRow") -> str:
    sev = str(getattr(row, "severity", "MEDIUM")).upper()
    bg, border, text = _SEV_COLORS.get(sev, ("#F9FAFB", "#9CA3AF", "#374151"))
    drug     = html.escape(str(getattr(row, "drug", "")))
    headline = html.escape(str(getattr(row, "headline", "")))
    action   = html.escape(str(getattr(row, "recommended_action", "")))
    facts    = [html.escape(f) for f in getattr(row, "supporting_facts", [])]
    facts_li = "".join(f"<li style='margin:2px 0;color:#6B7280'>{f}</li>" for f in facts[:2])

    return (
        f'<div style="border:1px solid {border};border-left:4px solid {border};'
        f'border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0;background:{bg}">'
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;color:{text};margin-bottom:4px">{sev}</div>'
        f'<div style="font-weight:700;font-size:13px;color:#111827;margin-bottom:6px">'
        f'{headline}</div>'
        f'<ul style="margin:0 0 8px;padding-left:16px">{facts_li}</ul>'
        f'<span style="display:inline-block;font-size:10px;font-weight:700;'
        f'padding:3px 10px;border-radius:4px;background:#F0FDF4;color:#166534;'
        f'border:1px solid #86EFAC;text-transform:uppercase;letter-spacing:0.05em">'
        f'{action}</span>'
        f'</div>'
    )


def _build_html(
    facility_name: str,
    insights: List["InsightRow"],
    kpi: dict,
    order_count: int,
    patient_risk_count: int,
    dashboard_url: str,
    digest_date: date,
    clinical_alerts: Optional[List["InsightRow"]] = None,
) -> str:
    insight_blocks = "".join(_insight_block_html(r) for r in insights[:3])
    if not insight_blocks:
        insight_blocks = (
            '<p style="color:#6B7280;font-size:13px">'
            'No critical insights detected today. Stock levels are healthy.</p>'
        )

    _kpi = {k.lower(): v for k, v in kpi.items()}
    stockouts    = int(_kpi.get("active_stockouts", 0) or 0)
    critical     = int(_kpi.get("critical_count", 0) or 0)
    low          = int(_kpi.get("low_count", 0) or 0)
    chronic      = int(_kpi.get("chronic_patients_active", 0) or 0)

    kpi_row = (
        f'<table style="width:100%;border-collapse:collapse;margin:12px 0">'
        f'<tr>'
        f'<td style="text-align:center;padding:10px;border:1px solid #E5E7EB;border-radius:6px">'
        f'<div style="font-size:22px;font-weight:700;color:{"#DC2626" if stockouts else "#111827"}">'
        f'{stockouts}</div><div style="font-size:10px;color:#9CA3AF;text-transform:uppercase">'
        f'Stocked out</div></td>'
        f'<td style="text-align:center;padding:10px;border:1px solid #E5E7EB;border-radius:6px">'
        f'<div style="font-size:22px;font-weight:700;color:{"#D97706" if critical else "#111827"}">'
        f'{critical}</div><div style="font-size:10px;color:#9CA3AF;text-transform:uppercase">'
        f'Critical &lt;7d</div></td>'
        f'<td style="text-align:center;padding:10px;border:1px solid #E5E7EB;border-radius:6px">'
        f'<div style="font-size:22px;font-weight:700;color:#111827">{low}</div>'
        f'<div style="font-size:10px;color:#9CA3AF;text-transform:uppercase">Low 7–30d</div></td>'
        f'<td style="text-align:center;padding:10px;border:1px solid #E5E7EB;border-radius:6px">'
        f'<div style="font-size:22px;font-weight:700;color:#111827">{order_count}</div>'
        f'<div style="font-size:10px;color:#9CA3AF;text-transform:uppercase">Order now</div></td>'
        f'<td style="text-align:center;padding:10px;border:1px solid #E5E7EB;border-radius:6px">'
        f'<div style="font-size:22px;font-weight:700;color:#111827">{patient_risk_count}</div>'
        f'<div style="font-size:10px;color:#9CA3AF;text-transform:uppercase">Patients at risk</div></td>'
        f'</tr></table>'
    )

    # Clinical alerts section (R3 dead stock + R5 refill overdue)
    if clinical_alerts:
        alert_blocks = "".join(_insight_block_html(r) for r in clinical_alerts[:5])
        clinical_section = (
            f'<tr><td style="padding:16px 28px 0">'
            f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.07em;color:#9CA3AF;margin-bottom:4px">'
            f'Clinical alerts</div>'
            f'{alert_blocks}'
            f'</td></tr>'
        )
    else:
        clinical_section = ""

    cta = (
        f'<div style="text-align:center;margin:20px 0">'
        f'<a href="{html.escape(dashboard_url)}" style="display:inline-block;'
        f'background:#0F6E56;color:#FFFFFF;font-weight:700;font-size:13px;'
        f'padding:12px 28px;border-radius:6px;text-decoration:none;'
        f'letter-spacing:0.03em">Open Full Dashboard →</a></div>'
    ) if dashboard_url else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Afya Inventory — Daily Digest</title>
</head>
<body style="margin:0;padding:0;background:#F3F4F6;font-family:'Helvetica Neue',Arial,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0" bgcolor="#F3F4F6">
    <tr><td align="center" style="padding:32px 16px">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#FFFFFF;border-radius:12px;border:1px solid #E5E7EB;overflow:hidden">

        <!-- Header -->
        <tr>
          <td style="background:#0F6E56;padding:20px 28px">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em;color:#A7F3D0">Afya Inventory Intelligence</div>
            <div style="font-size:20px;font-weight:800;color:#FFFFFF;margin:4px 0 2px">
              Daily Digest</div>
            <div style="font-size:12px;color:#6EE7B7">
              {facility_name} &nbsp;·&nbsp; {digest_date.strftime('%A, %d %b %Y')}</div>
          </td>
        </tr>

        <!-- KPI strip -->
        <tr><td style="padding:20px 28px 0">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.07em;color:#9CA3AF;margin-bottom:6px">
            Today's snapshot</div>
          {kpi_row}
        </td></tr>

        <!-- Insight cards -->
        <tr><td style="padding:16px 28px 0">
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.07em;color:#9CA3AF;margin-bottom:4px">
            Priority insights</div>
          {insight_blocks}
        </td></tr>

        <!-- Clinical alerts -->
        {clinical_section}

        <!-- CTA -->
        <tr><td style="padding:8px 28px 24px">{cta}</td></tr>

        <!-- Footer -->
        <tr>
          <td style="background:#F9FAFB;padding:14px 28px;border-top:1px solid #E5E7EB">
            <div style="font-size:10px;color:#9CA3AF">
              Sent by Afya Inventory Intelligence · {digest_date.strftime('%d %b %Y')} ·
              Numbers pre-computed from facility dispensing data.
              Do not reply to this email.
            </div>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def _build_plaintext(
    facility_name: str,
    insights: List["InsightRow"],
    kpi: dict,
    order_count: int,
    patient_risk_count: int,
    dashboard_url: str,
    digest_date: date,
    clinical_alerts: Optional[List["InsightRow"]] = None,
) -> str:
    _kpi = {k.lower(): v for k, v in kpi.items()}
    stockouts = int(_kpi.get("active_stockouts", 0) or 0)
    critical  = int(_kpi.get("critical_count", 0) or 0)
    low       = int(_kpi.get("low_count", 0) or 0)

    lines = [
        f"AFYA INVENTORY — DAILY DIGEST",
        f"{facility_name}  |  {digest_date.strftime('%d %b %Y')}",
        "",
        "SNAPSHOT",
        f"  Stocked out : {stockouts}",
        f"  Critical <7d: {critical}",
        f"  Low 7-30d   : {low}",
        f"  Order now   : {order_count}",
        f"  Patients at risk: {patient_risk_count}",
        "",
        "PRIORITY INSIGHTS",
    ]
    for i, row in enumerate(insights[:3], 1):
        lines.append(f"  {i}. [{getattr(row, 'severity', '')}] {getattr(row, 'headline', '')}")
        lines.append(f"     → {getattr(row, 'recommended_action', '')}")
    if not insights:
        lines.append("  No critical insights today. Stock levels are healthy.")

    if clinical_alerts:
        lines += ["", "CLINICAL ALERTS"]
        for i, row in enumerate(clinical_alerts[:5], 1):
            lines.append(f"  {i}. [{getattr(row, 'severity', '')}] {getattr(row, 'headline', '')}")
            lines.append(f"     → {getattr(row, 'recommended_action', '')}")

    if dashboard_url:
        lines += ["", f"Open dashboard: {dashboard_url}"]

    lines += ["", "—", "Afya Inventory Intelligence · Numbers pre-computed from facility data."]
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def send_daily_digest(
    facility_name: str,
    insights: List["InsightRow"],
    kpi: dict,
    order_count: int,
    patient_risk_count: int,
    facility_slug: Optional[str] = None,
    force: bool = False,
    clinical_alerts: Optional[List["InsightRow"]] = None,
) -> bool:
    """
    Send the daily digest email.

    Args:
        facility_name:       Human-readable facility label (shown in email).
        insights:            Top insights from insight_engine.detect_all() (stockout/demand items).
        kpi:                 KPI dict from get_kpi_summary() (keys case-insensitive).
        order_count:         Number of ORDER_NOW items from score_all().
        patient_risk_count:  Total patients at risk from get_patient_risk_totals().
        facility_slug:       Short identifier for idempotency stamp (default: lowered facility_name).
        force:               If True, bypass the once-per-day idempotency guard.
        clinical_alerts:     R3 dead stock + R5 refill overdue InsightRows. Rendered as a
                             dedicated "Clinical alerts" section below priority insights.

    Returns:
        True if email was sent, False if skipped (already sent today or config missing).

    Raises:
        smtplib.SMTPException on SMTP-level errors.
    """
    slug = (facility_slug or facility_name.lower().replace(" ", "_"))

    if not force and _already_sent_today(slug):
        return False

    # Load SMTP config from environment
    recipients_raw = _env("NOTIFY_EMAIL_TO")
    sender         = _env("NOTIFY_EMAIL_FROM")
    smtp_host      = _env("SMTP_HOST")
    smtp_port      = int(_env("SMTP_PORT", "587"))
    smtp_user      = _env("SMTP_USER")
    smtp_password  = _env("SMTP_PASSWORD")
    dashboard_url  = _env("DASHBOARD_URL")

    if not (recipients_raw and sender and smtp_host and smtp_user and smtp_password):
        # Config not set — silently skip (digest is optional)
        return False

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    if not recipients:
        return False

    today = date.today()
    subject = (
        f"[Afya] Daily digest — {facility_name} — "
        f"{len(insights)} insight{'s' if len(insights) != 1 else ''} · "
        f"{today.strftime('%d %b %Y')}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = ", ".join(recipients)

    plain     = _build_plaintext(facility_name, insights, kpi, order_count, patient_risk_count, dashboard_url, today, clinical_alerts=clinical_alerts)
    html_body = _build_html(facility_name, insights, kpi, order_count, patient_risk_count, dashboard_url, today, clinical_alerts=clinical_alerts)

    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, recipients, msg.as_string())

    _mark_sent_today(slug)
    return True
