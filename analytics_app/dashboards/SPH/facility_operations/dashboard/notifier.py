"""
Executive digest email via smtplib (Gmail SMTP).
Usage:
    import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from facility_operations.dashboard.notifier import send_digest
    ok, msg = send_digest("SPH", notices, stats)

Django migration: replace the smtplib block (_smtp_send function) with:
    from django.core.mail import send_mail
and replace the _smtp_send() call in send_digest() with:
    send_mail(subject, "", from_email, to_list, html_message=html, fail_silently=False)
All other code — _build_html, send_digest, write_current_notices — stays identical.
"""

import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def _smtp_send(subject: str, html: str, to_list: list) -> None:
    host_user = os.getenv("EMAIL_HOST_USER", "")
    host_pass = os.getenv("EMAIL_HOST_PASSWORD", "")
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = host_user
    msg["To"]      = ", ".join(to_list)
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.ehlo()
        s.starttls()
        s.login(host_user, host_pass)
        s.sendmail(host_user, to_list, msg.as_string())


def _build_html(facility_name: str, notices: list, stats: str,
                clinical_notes: list = None) -> str:
    today = datetime.now().strftime("%d %b %Y")
    _notices_json = json.dumps(
        {"facility": facility_name, "date": today, "count": len(notices), "notices": notices},
        separators=(",", ":"),
    )

    notice_rows = ""
    for n in notices:
        badge_bg = "#E11D48" if n["level"] == "CRITICAL" else "#D97706"
        notice_rows += f"""
        <tr>
          <td style="padding:14px 24px;border-bottom:1px solid #EBF3FB">
            <span style="display:inline-block;padding:2px 8px;border-radius:3px;
              font-size:9px;font-weight:800;letter-spacing:1.5px;color:#fff;
              background:{badge_bg};margin-bottom:6px">{n["level"]}</span><br>
            <span style="font-size:14px;font-weight:700;color:#003467">{n["title"]}</span>
            &nbsp;&nbsp;<span style="font-size:18px;font-weight:800;color:{badge_bg}">{n["metric"]}</span><br>
            <span style="font-size:11px;color:#003467;margin-top:6px;display:block">
              &#8594; {n["action"]}
            </span>
          </td>
        </tr>"""

    if not notice_rows:
        notice_rows = """
        <tr>
          <td style="padding:20px 24px;text-align:center;color:#0BB99F;
            font-size:13px;font-weight:700">
            &#10003; All Clear &mdash; no active notices
          </td>
        </tr>"""

    stats_row = ""
    if stats:
        stats_row = f"""
        <tr>
          <td style="padding:14px 24px;background:#F4F8FC;
            font-size:11px;color:#6B8CAE;border-top:2px solid #EBF3FB">
            {stats}
          </td>
        </tr>"""

    clinical_rows = ""
    if clinical_notes:
        clinical_rows = """
        <tr>
          <td style="padding:14px 24px 6px;font-size:9px;font-weight:800;
            color:#6B8CAE;text-transform:uppercase;letter-spacing:2px;
            border-top:2px solid #EBF3FB">
            Clinical Safety Monitor
          </td>
        </tr>"""
        for c in clinical_notes:
            clinical_rows += f"""
        <tr>
          <td style="padding:10px 24px 14px;border-bottom:1px solid #EBF3FB">
            <span style="font-size:13px;font-weight:700;color:#003467">{c["title"]}</span><br>
            <span style="font-size:12px;color:#003467;margin-top:4px;display:block">
              {c["metric"]}
            </span>
            <span style="font-size:11px;color:#6B8CAE;margin-top:4px;display:block">
              &#8594; {c["note"]}
            </span>
          </td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#F4F8FC;
  font-family:'Helvetica Neue',Helvetica,Arial,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0"
    style="padding:32px 12px">
    <tr><td>
      <table width="600" cellpadding="0" cellspacing="0"
        style="max-width:600px;margin:0 auto;background:#fff;
        border-radius:8px;border:1px solid #D6E4F0;overflow:hidden">

        <!-- Header -->
        <tr>
          <td style="padding:20px 24px;background:#003467">
            <div style="font-size:9px;font-weight:700;color:#7FB3E0;
              text-transform:uppercase;letter-spacing:2px">
              {facility_name} &middot; Operational Intelligence
            </div>
            <div style="font-size:20px;font-weight:800;color:#fff;margin-top:4px">
              Executive Digest
            </div>
            <div style="font-size:11px;color:#7FB3E0;margin-top:2px">{today}</div>
          </td>
        </tr>

        <!-- Label -->
        <tr>
          <td style="padding:14px 24px 6px;font-size:9px;font-weight:800;
            color:#0072CE;text-transform:uppercase;letter-spacing:2px">
            Active Notices
          </td>
        </tr>

        <!-- Notices -->
        {notice_rows}

        {stats_row}

        {clinical_rows}

        <!-- Footer -->
        <tr>
          <td style="padding:14px 24px;border-top:1px solid #EBF3FB;
            font-size:10px;color:#B0C8E0;text-align:center">
            Afya Analytics &middot; St. Peter's Orthopedic Hospital
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
<!-- NOTICES_JSON: {_notices_json} -->
</body>
</html>"""


def get_recipients() -> list:
    """Load recipient list from DIGEST_RECIPIENTS env var (comma-separated)."""
    raw = os.getenv("DIGEST_RECIPIENTS", "")
    return [e.strip() for e in raw.split(",") if e.strip()]


def send_digest(facility_name: str, notices: list, stats: str = "",
                recipients: list = None, clinical_notes: list = None) -> tuple:
    """
    Send an executive digest email via Gmail SMTP.

    notices:        list of dicts — {level, title, metric, action}
    stats:          optional one-line plain text footer
    clinical_notes: optional list of dicts — {title, metric, note}
    recipients:     list of email addresses; defaults to DIGEST_RECIPIENTS env var
    Returns:        (success: bool, message: str)
    """
    host_user = os.getenv("EMAIL_HOST_USER", "")
    if not host_user:
        return False, "EMAIL_HOST_USER not set in .env"

    to_list = recipients if recipients is not None else get_recipients()
    if not to_list:
        return False, "No recipients configured (set DIGEST_RECIPIENTS in .env)"

    n     = len(notices)
    tag   = f"{n} Active Notice{'s' if n != 1 else ''}" if n else "All Clear"
    today = datetime.now().strftime("%d %b %Y")

    try:
        _smtp_send(
            subject=f"{facility_name} — {tag} | {today}",
            html=_build_html(facility_name, notices, stats, clinical_notes),
            to_list=to_list,
        )
        write_current_notices(facility_name, notices)
        return True, "sent"
    except Exception as e:
        return False, str(e)


def write_current_notices(facility_name: str, notices: list) -> None:
    """Write per-facility notices JSON so the file always reflects current state."""
    slug = facility_name.replace(" ", "_").upper()
    path = os.path.join(os.path.dirname(__file__), f"current_notices_{slug}.json")
    today = datetime.now().strftime("%d %b %Y")
    with open(path, "w") as f:
        json.dump(
            {"facility": facility_name, "date": today, "count": len(notices), "notices": notices},
            f, indent=2,
        )
