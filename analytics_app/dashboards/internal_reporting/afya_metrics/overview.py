"""Overview tab: quarter strip, targets panel, delivery & team panel."""

from . import analysis as A
from . import theme as T
from .analysis import fmt, pct


def strip(period, week):
    clock = A.clock_for(period, week)
    if not clock.start:
        T.html(f'<div class="x-strip"><span><b>{period}</b></span></div>')
        return
    T.html(f'<div class="x-strip"><span><b>{period}</b> &nbsp;·&nbsp; Week {clock.week_no} of {clock.week_count}'
           f'</span><div class="x-qbar"><i style="width:{clock.elapsed_pct:.1f}%"></i></div>'
           f'<span><b>{clock.days_left}</b> days left</span></div>')


def _change(d, kind="num"):
    if not d:
        return ""
    arrow, cls = ("▲", "up") if d > 0 else ("▼", "down")
    return f'<span class="x-chg {cls}">{arrow} {fmt(abs(d), kind, compact=True)}</span>'


def _target_cell(t, unit="target"):
    status = (f'<span class="x-st" style="color:{T.STATUS_PILL[t.status][0]}">{A.STATUS_LABEL[t.status]}</span>'
              if t.status else "")
    return T.ring_cell(t.label, fmt(t.value, t.kind), T.ring(t.pct, T.STATUS_COLOUR[t.status]),
                       change=_change(t.delta, t.kind),
                       sub=f"of <b>{fmt(t.target, t.kind, compact=True)}</b> {unit}", status=status)


def _upds_cell(t, unique_patients):
    """UPDs = unique patients from Patient counts (a snapshot, so no weekly change or status)."""
    if unique_patients is None:
        return T.ring_cell(t.label, "—", T.ring(0, T.NAVY), sub="Patient counts unavailable")
    p = pct(unique_patients, t.target)
    return T.ring_cell(t.label, fmt(unique_patients), T.ring(p, T.NAVY),
                       sub=f"of <b>{fmt(t.target, compact=True)}</b> target")


def _data_cell(series, patient_records):
    """Records processed against total patient records from Patient counts."""
    proc = series[-1]["DATA_PROCESSED"]
    d = proc - series[-2]["DATA_PROCESSED"] if len(series) > 1 else 0
    if not patient_records:
        return T.cell("Data processed", fmt(proc), change=_change(d), bar=T.bar(0, T.NAVY),
                      left="Patient counts unavailable", wide=True)
    return T.cell("Data processed", fmt(proc), change=_change(d),
                  bar=T.bar(pct(proc, patient_records), T.NAVY),
                  left=f"<b>{pct(proc, patient_records):.0f}%</b> of {fmt(patient_records, compact=True)} patient records",
                  right=f"{fmt(max(patient_records - proc, 0), compact=True)} backlog", wide=True)


def _team_cell(series):
    cur = series[-1]
    ft, co = cur["TEAM_FULL_TIME"], cur["TEAM_CONSULTANTS"]
    tot = lambda r: r["TEAM_FULL_TIME"] + r["TEAM_CONSULTANTS"]
    d = tot(cur) - tot(series[-2]) if len(series) > 1 else 0
    return T.cell("Team", f"{ft + co}", unit="people", change=_change(d),
                  bar=T.split([(ft, T.NAVY), (co, T.SOFT)]),
                  left=f'<i class="dot" style="background:{T.NAVY}"></i><b>{ft}</b> full-time',
                  right=f'<i class="dot" style="background:{T.SOFT}"></i><b>{co}</b> consultants', wide=True)


def render(rows, period, week, unique_patients, patient_records):
    series = [r for r in rows if r["PERIOD"] == period and r["WEEK_ENDING"] <= week]
    trk = {t.key: t for t in A.tracks(series, A.clock_for(period, week))}
    hint = "▲ ▼ change vs previous week" if len(series) > 1 else ""

    T.panel(f"{period} targets", hint, [
        _upds_cell(trk["upds"], unique_patients), _target_cell(trk["revenue"]),
        _target_cell(trk["mou"], "counties"), _target_cell(trk["facilities"]), _target_cell(trk["associates"]),
    ])
    T.panel("Delivery & team", "", [_data_cell(series, patient_records), _team_cell(series)])
