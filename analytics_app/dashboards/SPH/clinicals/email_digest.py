"""
sph/clinicals/email_digest.py
================================
Sidebar control that emails the Overview ("Hospital at a Glance") page's
KPI strip and top signals as an HTML digest.

Reads the same .env values (EMAIL_HOST, EMAIL_PORT, EMAIL_HOST_USER,
EMAIL_HOST_PASSWORD, DEFAULT_FROM_EMAIL, DIGEST_RECIPIENTS) already used by
analytics_app/dashboards/ksh/facility_utilization/notifier.py — but sends
with plain smtplib instead of routing through django.setup(). Bootstrapping
the full Django app registry (as notifier.py does) pulls in every
dependency the whole Django project needs at import time (Celery, Redis,
etc. via airflow_dashboard/__init__.py's `from .celery import app`), which
isn't installed in this Streamlit app's environment — so this reuses the
*values*, not the Django plumbing.

Only sends on explicit user action ("Send now") — there is no background
scheduler here. Streamlit's process model (one script re-run per browser
interaction) can't reliably fire a digest on a wall-clock schedule by
itself; a recurring send needs an external trigger (cron / cloud scheduler)
calling `send_overview_digest(recipients)` from outside the app process.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_env():
    """
    Loads the same .env the Django settings module reads — explicit path
    computed from this file's own location (dashboard root is 4 levels up:
    clinicals -> sph -> dashboards -> analytics_app -> dashboard), so this
    works regardless of the process's current working directory.
    """
    if load_dotenv is None:
        return
    dashboard_root = Path(__file__).resolve().parents[4]
    load_dotenv(dashboard_root / ".env")


def _email_config() -> dict:
    _load_env()
    return dict(
        host=os.getenv("EMAIL_HOST", "smtp.gmail.com").strip(),
        port=int(os.getenv("EMAIL_PORT", "587").strip()),
        use_tls=os.getenv("EMAIL_USE_TLS", "True").strip().lower() in ("true", "1", "yes"),
        user=os.getenv("EMAIL_HOST_USER", "").strip(),
        password=os.getenv("EMAIL_HOST_PASSWORD", "").strip(),
        sender=os.getenv("DEFAULT_FROM_EMAIL", "").strip(),
    )


def get_recipients() -> list:
    """Matches notifier.py's get_recipients() — DIGEST_RECIPIENTS env var,
    comma-separated — falling back to a single known recipient if unset."""
    _load_env()
    raw = os.getenv("DIGEST_RECIPIENTS", "")
    recipients = [e.strip() for e in raw.split(",") if e.strip()]
    return recipients or [DEFAULT_DIGEST_RECIPIENT]


def _safe(fn, *a):
    try:
        df = fn(*a)
        return df if df is not None and not df.empty else None
    except Exception:
        return None


def _fmt(v, suffix="%", digits=1, default="—"):
    return f"{v:.{digits}f}{suffix}" if v is not None else default


def build_overview_snapshot() -> dict:
    """
    Re-runs the same headline queries the Overview tab uses for its KPI
    strip, plus the Data Quality overall score, and returns one flat dict.
    Kept intentionally to headline numbers only — an email digest is a
    KPI summary, not a re-rendering of the page's interactive charts.
    """
    import clinicals.case_mix_module.cm_queries as CMQ
    import clinicals.flow_retention_module.fr_queries as FRQ
    import clinicals.clinical_activity_module.ca_queries as CAQ
    import clinicals.disease_burden_module.maternal.mat_queries as MAQ
    import clinicals.opd_ipd_module.queries as Q
    import clinicals.data_quality_module.dq_views as DQV

    df_cm_kpis  = _safe(CMQ.get_cm_headline_kpis)
    df_opd_kpis = _safe(Q.get_headline_kpis)
    df_ca_kpis  = _safe(CAQ.get_ca_overview_kpis)
    df_fr_status = _safe(FRQ.get_fr_status_overall)
    df_mat_anc  = _safe(MAQ.get_mat_anc_visit_distribution)

    total_visits = core_ortho_pct = blended_conv = None
    if df_cm_kpis is not None:
        total_visits = int(df_cm_kpis.iloc[0]["TOTAL_VISITS"])
        core_ortho_pct = float(df_cm_kpis.iloc[0]["CORE_ORTHO_SHARE_PCT"])
    if df_opd_kpis is not None:
        blended_conv = float(df_opd_kpis.iloc[0]["OVERALL_CONVERSION_PCT"])

    readmission_rate = worst_ssi = worst_ssi_bench = None
    worst_ssi_cat = "—"
    if df_ca_kpis is not None:
        r = df_ca_kpis.iloc[0]
        readmission_rate = float(r.get("READMISSION_RATE", 0) or 0)
        worst_ssi = float(r.get("WORST_SSI_RATE", 0) or 0)
        worst_ssi_cat = r.get("WORST_SSI_CATEGORY", "—")
        worst_ssi_bench = float(r.get("WORST_SSI_BENCHMARK", 0) or 0)
    ssi_ratio = round(worst_ssi / worst_ssi_bench, 1) if worst_ssi and worst_ssi_bench else None

    retention_pct = ltfu_pct = lapsing_pct = None
    if df_fr_status is not None:
        s = df_fr_status.set_index("STATUS")["PCT_OF_CLASSIFIABLE_PATIENTS"]
        active_pct = float(s.get("Active", 0))
        lapsing_pct = float(s.get("Lapsing", 0))
        ltfu_pct = float(s.get("LTFU", 0))
        retention_pct = active_pct + lapsing_pct

    anc_single_pct = None
    if df_mat_anc is not None:
        single = df_mat_anc[df_mat_anc["VISIT_COUNT_BUCKET"].str.startswith("1 visit")]
        anc_single_pct = float(single.iloc[0]["PCT_OF_ANC_PATIENTS"]) if not single.empty else None

    dq_overall = None
    try:
        dq_overall = DQV.compute_scores().get("overall_score")
    except Exception:
        pass

    return dict(
        total_visits=total_visits, core_ortho_pct=core_ortho_pct, blended_conv=blended_conv,
        readmission_rate=readmission_rate, worst_ssi_cat=worst_ssi_cat, ssi_ratio=ssi_ratio,
        retention_pct=retention_pct, ltfu_pct=ltfu_pct, lapsing_pct=lapsing_pct,
        anc_single_pct=anc_single_pct, dq_overall=dq_overall,
    )


def build_digest_html(snap: dict, today: str) -> str:
    """
    Same branded card structure as notifier.py's _build_html() (navy header,
    uppercase eyebrow label, bordered content card, muted footer) so the two
    digests this codebase sends look like they came from the same product,
    adapted to a KPI table instead of a notices list.
    """
    rows = [
        ("Total visits", f"{snap['total_visits']:,}" if snap["total_visits"] else "—"),
        ("Core orthopedics share", _fmt(snap["core_ortho_pct"])),
        ("Blended conversion", _fmt(snap["blended_conv"])),
        ("Readmission rate", _fmt(snap["readmission_rate"])),
        ("Retention rate", _fmt(snap["retention_pct"])),
        ("Loss to follow-up", _fmt(snap["ltfu_pct"])),
        ("Worst SSI vs benchmark",
         f"{snap['worst_ssi_cat']}, {snap['ssi_ratio']}x ceiling" if snap["ssi_ratio"] else str(snap["worst_ssi_cat"])),
        ("ANC single-visit rate", _fmt(snap["anc_single_pct"])),
        ("Data quality score", f"{snap['dq_overall']:.0f} / 100" if snap["dq_overall"] is not None else "—"),
    ]
    rows_html = "".join(
        f"""
        <tr>
          <td style="padding:14px 24px;border-bottom:1px solid #EBF3FB;font-size:13px;color:#003467">{label}</td>
          <td style="padding:14px 24px;border-bottom:1px solid #EBF3FB;font-size:15px;font-weight:800;
            color:#003467;text-align:right">{value}</td>
        </tr>"""
        for label, value in rows
    )
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="margin:0;padding:0;background:#F4F8FC;
  font-family:'Helvetica Neue',Helvetica,Arial,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0" style="padding:32px 12px">
    <tr><td>
      <table width="600" cellpadding="0" cellspacing="0"
        style="max-width:600px;margin:0 auto;background:#fff;
        border-radius:8px;border:1px solid #D6E4F0;overflow:hidden">

        <!-- Header -->
        <tr>
          <td style="padding:20px 24px;background:#003467">
            <div style="font-size:9px;font-weight:700;color:#7FB3E0;
              text-transform:uppercase;letter-spacing:2px">
              St. Peter's Orthopaedic Hospital &middot; Clinical Operations
            </div>
            <div style="font-size:20px;font-weight:800;color:#fff;margin-top:4px">
              Hospital at a Glance — Digest
            </div>
            <div style="font-size:11px;color:#7FB3E0;margin-top:2px">{today}</div>
          </td>
        </tr>

        <!-- Label -->
        <tr>
          <td style="padding:14px 24px 6px;font-size:9px;font-weight:800;
            color:#0072CE;text-transform:uppercase;letter-spacing:2px">
            Headline KPIs
          </td>
        </tr>

        <!-- KPI rows -->
        <table width="100%" cellpadding="0" cellspacing="0">{rows_html}</table>

        <!-- Footer -->
        <tr>
          <td style="padding:14px 24px;border-top:1px solid #EBF3FB;
            font-size:10px;color:#B0C8E0;text-align:center">
            Afya Analytics &middot; Private Hospitals Dashboard
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


DEFAULT_DIGEST_RECIPIENT = "mkubania@afya.ai"


def send_overview_digest(recipients: list = None) -> tuple:
    """Returns (success: bool, message: str)."""
    to_list = recipients if recipients else get_recipients()
    if not to_list:
        return False, "No recipients configured (set DIGEST_RECIPIENTS in .env)."

    cfg = _email_config()
    if not cfg["user"] or not cfg["password"]:
        return False, "EMAIL_HOST_USER/EMAIL_HOST_PASSWORD not set in .env — same config notifier.py uses for KSH digests."

    from datetime import datetime
    today = datetime.now().strftime("%d %b %Y")

    try:
        snap = build_overview_snapshot()
        html = build_digest_html(snap, today)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Hospital at a Glance — Digest | {today}"
        msg["From"] = cfg["sender"] or cfg["user"]
        msg["To"] = ", ".join(to_list)
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as server:
            if cfg["use_tls"]:
                server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["sender"] or cfg["user"], to_list, msg.as_string())
        return True, "Digest sent."
    except Exception as exc:
        return False, f"Send failed: {exc}"


def render_sidebar_control() -> None:
    """
    One-click icon-tile button — no toggle state, no visible sending
    process. Click fires the send immediately; only the final result
    (sent / failed) is shown. No recipient is ever shown in the UI.
    """
    with st.sidebar.container(border=True):
        col_icon, col_btn = st.columns([5, 1], vertical_alignment="center")
        col_icon.markdown(
            '<div style="display:flex;align-items:center;gap:10px">'
            '<div style="background:#DCEFE9;border-radius:8px;width:34px;height:34px;min-width:34px;'
            'display:flex;align-items:center;justify-content:center">'
            '<i class="ti ti-mail" style="font-size:16px;color:#1B8A82"></i></div>'
            '<div><div style="font-size:13px;font-weight:600;color:#141F3D;line-height:1.3">Email digest</div>'
            '<div style="font-size:11px;color:#8A93A6;font-weight:500">Click to send</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        clicked = col_btn.button("➤", key="dq_email_digest_send", help="Send email digest",
                                  use_container_width=True)

    if clicked:
        ok, message = send_overview_digest()
        if ok:
            st.sidebar.success("Digest sent.")
        else:
            st.sidebar.error(message)
