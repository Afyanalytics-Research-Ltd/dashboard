"""
sph/email_digest.py
=====================
Sidebar control that emails the Overview ("Hospital at a Glance") page's
KPI strip and top signals as a plain HTML digest.

Sends via SMTP using credentials from Streamlit secrets or environment
variables (never hardcoded):
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM

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

import streamlit as st


def _smtp_config():
    def _get(key, default=None):
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        return os.environ.get(key, default)

    return dict(
        host=_get("SMTP_HOST"),
        port=int(_get("SMTP_PORT", 587) or 587),
        user=_get("SMTP_USER"),
        password=_get("SMTP_PASSWORD"),
        sender=_get("SMTP_FROM"),
    )


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
    import sph.clinicals.case_mix_module.cm_queries as CMQ
    import sph.clinicals.flow_retention_module.fr_queries as FRQ
    import sph.clinicals.clinical_activity_module.ca_queries as CAQ
    import sph.clinicals.disease_burden_module.maternal.mat_queries as MAQ
    import sph.clinicals.opd_ipd_module.queries as Q
    import sph.clinicals.data_quality_module.dq_views as DQV

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


def build_digest_html(snap: dict) -> str:
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
        f'<tr><td style="padding:8px 12px;border-bottom:1px solid #E4E7ED;color:#4B5468">{label}</td>'
        f'<td style="padding:8px 12px;border-bottom:1px solid #E4E7ED;font-weight:700;color:#141F3D;'
        f'text-align:right">{value}</td></tr>'
        for label, value in rows
    )
    return f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto">
      <div style="background:#1B8A82;padding:16px 20px;border-radius:8px 8px 0 0">
        <span style="color:#FFFFFF;font-size:16px;font-weight:700">Hospital at a Glance — Digest</span>
      </div>
      <div style="border:1px solid #E4E7ED;border-top:0;border-radius:0 0 8px 8px;padding:4px 0">
        <table style="width:100%;border-collapse:collapse;font-size:13px">{rows_html}</table>
      </div>
      <p style="font-size:11px;color:#8A93A6;margin-top:12px">
        Generated from live St. Peter's Orthopaedic Hospital data. See the full Overview page in the
        dashboard for charts, trends, and section-level detail behind each figure above.
      </p>
    </div>
    """


def send_overview_digest(recipients: list) -> tuple:
    """Returns (success: bool, message: str)."""
    cfg = _smtp_config()
    missing = [k for k in ("host", "user", "password", "sender") if not cfg[k]]
    if missing:
        return False, f"SMTP not configured — missing: {', '.join(missing)}."
    if not recipients:
        return False, "No recipients provided."

    snap = build_overview_snapshot()
    html = build_digest_html(snap)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Hospital at a Glance — Digest"
    msg["From"] = cfg["sender"]
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=20) as server:
            server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["sender"], recipients, msg.as_string())
        return True, "Digest sent."
    except Exception as exc:
        return False, f"Send failed: {exc}"


DEFAULT_DIGEST_RECIPIENT = "mkubania@afya.ai"


def render_sidebar_control() -> None:
    """
    Icon-tile toggle (matches the app's other "Option D" tile pattern) —
    flipping it on fires the send once; flipping it off then on again
    re-sends. No recipient is ever shown in the UI.
    """
    prev_on = st.session_state.get("dq_email_digest_prev", False)

    with st.sidebar.container(border=True):
        col_icon, col_toggle = st.columns([5, 1], vertical_alignment="center")
        status_placeholder = col_icon.empty()
        with col_toggle:
            is_on = st.toggle("Email digest", key="dq_email_digest_toggle", label_visibility="collapsed")

    status_text = "Active" if is_on else "Idle"
    status_color = "#1B8A82" if is_on else "#8A93A6"
    status_placeholder.markdown(
        '<div style="display:flex;align-items:center;gap:10px">'
        '<div style="background:#DCEFE9;border-radius:8px;width:34px;height:34px;min-width:34px;'
        'display:flex;align-items:center;justify-content:center">'
        '<i class="ti ti-mail" style="font-size:16px;color:#1B8A82"></i></div>'
        '<div><div style="font-size:13px;font-weight:600;color:#141F3D;line-height:1.3">Email digest</div>'
        f'<div style="font-size:11px;color:{status_color};font-weight:500">{status_text}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if is_on and not prev_on:
        with st.spinner("Sending digest…"):
            ok, message = send_overview_digest([DEFAULT_DIGEST_RECIPIENT])
        if ok:
            st.sidebar.success("Digest sent.")
        else:
            st.sidebar.error(message)

    st.session_state["dq_email_digest_prev"] = is_on
