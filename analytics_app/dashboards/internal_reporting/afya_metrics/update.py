"""Update dialog: weekly figures (new week, new quarter, or correction) and patient counts."""

import pandas as pd
import streamlit as st

from . import data as D
from . import theme as T
from .analysis import fmt

# (column, label, step) by section
SECTIONS = [
    ("Delivery", [("DATA_PROCESSED", "Records processed", 1_000)]),
    ("Reach", [("MOU_SIGNED", "County MoUs signed", 1),
               ("FACILITIES_PRIVATE", "Facilities on HMIS", 1),
               ("APPRENTICES_ONBOARDED", "Associates onboarded", 1)]),
    ("Revenue & team", [("REVENUE_BOOKED", "Revenue booked (USD)", 1_000),
                        ("TEAM_FULL_TIME", "Full-time staff", 1),
                        ("TEAM_CONSULTANTS", "Consultants", 1)]),
]
TARGETS = [("UPDS_TARGET", "UPDs", 100_000), ("REVENUE_TARGET", "Revenue (USD)", 1_000),
           ("MOU_TOTAL_COUNTIES", "Counties", 1), ("FACILITIES_TARGET", "Facilities", 1),
           ("APPRENTICES_TARGET", "Associates", 1)]
CUMULATIVE = {"DATA_PROCESSED": "Records processed",
              "MOU_SIGNED": "MoUs signed", "FACILITIES_PRIVATE": "Facilities on HMIS",
              "REVENUE_BOOKED": "Revenue booked", "APPRENTICES_ONBOARDED": "Associates onboarded"}


def _checks(row, prev, patient_records):
    errors, warnings = [], []
    if row["MOU_SIGNED"] > row["MOU_TOTAL_COUNTIES"]:
        errors.append("MoUs signed is higher than the number of counties.")
    if patient_records and row["DATA_PROCESSED"] > patient_records:
        warnings.append(f"Records processed is higher than total patient records ({fmt(patient_records)}).")
    if prev:
        warnings += [f"{label} is lower than the previous week ({fmt(prev[c])} → {fmt(row[c])}) — "
                     f"these are running totals, so check this is a correction."
                     for c, label in CUMULATIVE.items() if row[c] < prev[c]]
    return errors, warnings


def _number(col, c, label, step, value, ref, ref_name, key):
    """Number input with the reference value and change shown underneath."""
    v = int(col.number_input(label, min_value=0, value=int(value), step=step, key=key))
    if ref is not None:
        d = v - ref[c]
        change = f" · :{'green' if d > 0 else 'red'}[{'▲' if d > 0 else '▼'} {fmt(abs(d))}]" if d else ""
        col.caption(f"{ref_name}: {fmt(ref[c])}{change}")
    return v


def _pick_week(rows):
    """Returns (week, period, saved row or None). Period comes from the date, or the saved row."""
    c_mode, c_week, c_period = st.columns([1.5, 1, 1], vertical_alignment="bottom")
    mode = c_mode.segmented_control("Entry", ["New week", "Correct a saved week"], default="New week",
                                    key="up_mode") if rows else "New week"
    mode = mode or "New week"
    if mode == "Correct a saved week":
        saved = [r["WEEK_ENDING"] for r in reversed(rows)]
        week = c_week.selectbox("Week ending", saved, key="up_week_saved",
                                format_func=lambda d: f"{d:%d %b %Y}")
    else:
        week = c_week.date_input("Week ending", value=D.next_week_ending(rows), format="DD/MM/YYYY",
                                 key="up_week_new")
    stored = next((r for r in rows if r["WEEK_ENDING"] == week), None)
    period = stored["PERIOD"] if stored else D.period_for(week)
    c_period.text_input("Period", value=period, disabled=True, key=f"up_period_{week}")
    return week, period, stored


def _weekly(rows, patient_records):
    week, period, existing = _pick_week(rows)
    prev = next((r for r in reversed(rows) if r["WEEK_ENDING"] < week), None)
    new_quarter = existing is None and (prev is None or prev["PERIOD"] != period)
    base = existing or prev or {c: 0 for c in D.NUM_COLS}
    ref, ref_name = (existing, "Saved") if existing else (prev, "Last week")

    if existing:
        st.info(f"Editing the saved figures for **{week:%d %b %Y}** ({period}).", icon=":material/edit:")
    elif new_quarter and prev:
        st.info(f"**Starting {period}.** Running totals carry forward from {prev['PERIOD']} — "
                f"confirm the {period} targets first.", icon=":material/flag:")
    elif prev:
        st.caption(f"Prefilled from {prev['WEEK_ENDING']:%d %b %Y} — update what changed.")

    sfx = f"{period}_{week}"
    # UPDs and records received come from Patient counts; carry the stored columns forward
    row = {"PERIOD": period, "WEEK_ENDING": week, "UPDS_PROCESSED": int(base["UPDS_PROCESSED"]),
           "DATA_RECEIVED": int(base["DATA_RECEIVED"])}

    def targets():
        cols = st.columns(len(TARGETS))
        for col, (c, label, step) in zip(cols, TARGETS):
            row[c] = _number(col, c, label, step, base[c], prev if new_quarter else ref,
                             prev["PERIOD"] if new_quarter and prev else ref_name, f"up_{c}_{sfx}")

    confirmed = True
    if new_quarter:
        T.html(f'<div class="x-sec">{period} targets</div>')
        targets()
        confirmed = st.checkbox(f"These are the {period} targets", key=f"up_confirm_{sfx}")

    for title, fields in SECTIONS:
        T.html(f'<div class="x-sec">{title}</div>')
        cols = st.columns(3)
        for col, (c, label, step) in zip(cols, fields):
            row[c] = _number(col, c, label, step, base[c], ref, ref_name, f"up_{c}_{sfx}")

    if not new_quarter:
        with st.expander(f"{period} targets"):
            targets()

    errors, warnings = _checks(row, prev, patient_records)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    n = sum(row[c] != ref[c] for c in D.NUM_COLS) if ref is not None else None
    if ref is not None:
        summary = [f"**{n} change{'s' if n != 1 else ''}**" if n else
                   ("No changes" if existing else "Same figures as last week")]
        if warnings:
            summary.append(f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''}")
        st.caption(" · ".join(summary))
    if not confirmed:
        st.caption(f"Confirm the {period} targets to save.")

    label = f"Save changes to {week:%d %b %Y}" if existing else f"Save week ending {week:%d %b %Y}"
    nothing_to_save = bool(existing) and n == 0
    if st.button(label, type="primary", disabled=bool(errors) or not confirmed or nothing_to_save,
                 width="stretch"):
        if _saved(D.upsert, row):
            st.session_state["x_goto_week"] = week
            st.rerun()


def _patients():
    rows, _ = D.read_patient_counts()
    st.caption("Edit counts in place, add a source in the empty last row, or select rows to delete them.")
    edited = st.data_editor(
        pd.DataFrame(rows, columns=["SOURCE", "PATIENTS", "UNIQUE_PATIENTS"]),
        num_rows="dynamic", hide_index=True, width="stretch", key="up_patients",
        column_config={
            "SOURCE": st.column_config.TextColumn("Source", required=True, width="large"),
            "PATIENTS": st.column_config.NumberColumn("Patients", min_value=0, step=1, format="localized"),
            "UNIQUE_PATIENTS": st.column_config.NumberColumn("Unique patients", min_value=0, step=1,
                                                             format="localized"),
        })

    after = {}
    for rec in edited.to_dict("records"):
        src = str(rec.get("SOURCE") or "").strip()
        if src and src.lower() != "nan":
            after[src] = {"SOURCE": src, "PATIENTS": D._num(rec.get("PATIENTS")),
                          "UNIQUE_PATIENTS": D._num(rec.get("UNIQUE_PATIENTS"))}
    before = {r["SOURCE"]: r for r in rows}
    upserts = [r for k, r in after.items() if before.get(k) != r]
    deletes = [k for k in before if k not in after]

    bad = [r["SOURCE"] for r in after.values() if r["UNIQUE_PATIENTS"] > r["PATIENTS"]]
    if bad:
        st.error("Unique patients can't exceed patients: " + ", ".join(bad))
    changes = len(upserts) + len(deletes)
    if st.button(f"Save {changes} change{'s' if changes != 1 else ''}" if changes else "No changes",
                 type="primary", disabled=bool(bad) or not changes, width="stretch", key="up_pc_save"):
        if _saved(D.save_patient_counts, upserts, deletes):
            st.rerun()


def _saved(fn, *args):
    try:
        fn(*args)
    except Exception as exc:
        st.error(f"Couldn't save: {exc}")
        return False
    return True


@st.dialog("Update", width="large")
def open_dialog(rows, patient_records=None):
    t1, t2 = st.tabs(["Weekly figures", "Patient counts"])
    with t1:
        _weekly(rows, patient_records)
    with t2:
        _patients()
