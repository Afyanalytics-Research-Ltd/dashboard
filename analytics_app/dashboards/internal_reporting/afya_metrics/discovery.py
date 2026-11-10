"""Discovery tab: the questions patient discovery needs a person to answer, and the answers recorded so far.

Questions come from AFYA_PATIENT_DISCOVERY_QUESTIONS (replaced by every discovery run). Answers are appended to
AFYA_PATIENT_DISCOVERY_STATEMENTS with who, when and why. Each meaning question carries what answering "patients"
does (computed by the discovery run); once every answer it needs is in, the accepted tables are registered in
AFYA_PATIENT_COUNT_SOURCES at once (insert-only, with provenance) and counted on the next refresh. Existing registry
rows and counts are never changed here.
"""

from html import escape

import streamlit as st

from patient_discovery.model import questions as Q
from patient_discovery.model import store as STORE

from . import data as D
from . import theme as T

LABELS = {"person": "Individual patients", "household": "Households", "insurance_member": "Insurance members",
          "file": "Files or documents", "visit": "Visits", "other": "Something else", "not_sure": "Not sure",
          "yes": "Yes, in scope", "no": "No, not in scope", "not_a_source": "Not a patient source",
          "keep_asking": "Keep asking"}
DECISIONS = ("meaning", "scope")       # questions that decide what is counted; every other type is a data check
# Plain-language heading and one-line explanation per data-check type (display only; the run's own wording stays
# available under "Technical details").
CHECKS = {
    "GRAIN": ("Rows need checking", "Rows in this table can't be counted reliably yet, for example because "
              "they repeat or come from documents."),
    "KIND_UNKNOWN": ("Table type not recognised", "Discovery couldn't tell what one row of this table represents."),
    "UNLINKED": ("Separate group of patients", "None of its patient IDs appear in this facility's counted data."),
    "DERIVED_VIEW": ("Built from other data", "This view reads other tables; those are usually what should be "
                     "counted."),
    "DERIVED_LAYER": ("Reporting or staging layer", "These tables are built from other tables."),
    "VALUE_RELATION": ("Overlaps counted data", "Some of its patient IDs also appear in a source that is already "
                       "counted."),
    "DOUBLE_COUNT": ("Could be counted twice", "It uses the same patient IDs as a source that is already counted."),
    "REKEYED": ("Same records, different IDs", "It holds records of a counted source under other patient IDs."),
    "SUPERSET_LIVE": ("Contains a counted source", "It holds every record of a counted source, plus more."),
    "PARTIAL_RECORDS": ("Shares records with a counted source", "Part of its records are already counted "
                        "elsewhere."),
    "HISTORY_CHANGE": ("Possible replacement", "It looks like a newer version of a counted source, but its history "
                       "differs."),
    "REPRESENTATIONS": ("Same patients, several tables", "One group of patients appears in several tables."),
    "SCHEMA_DRIFT": ("Counted column has changed", "The column a counted source relies on no longer exists."),
    "LINEAGE_UNREADABLE": ("Origin unknown", "Discovery couldn't read which tables this view is built from."),
    "COVER_PREMISE": ("Coverage rule doesn't hold", "A rule says this data is covered by a source that isn't "
                      "counted."),
    "BEHAVIOUR": ("Confirmed ID behaves unexpectedly", "A confirmed patient ID doesn't behave like one in this "
                  "table."),
    "TRUNCATED": ("Too many ID-like columns", "Not every candidate column could be measured."),
}


@st.cache_data(ttl=D.SF_TTL, show_spinner=False)
def _questions():
    return STORE.read_questions(D._fetch)


@st.cache_data(ttl=D.SF_TTL, show_spinner=False)
def _answers():
    return STORE.answered_questions(D._fetch), STORE.read_statements(D._fetch)


def _short(fqn):
    return fqn.split(".", 1)[1] if fqn.count(".") == 2 else fqn


def _table(fqn):
    """Table name alone, for headings; the full name stays in the technical details."""
    return fqn.rsplit(".", 1)[-1]


def _facility(f):
    return (f or "").replace("_", " ").title()


def _chip(text, kind):
    return f"<span class='x-disc-chip {kind}'>{escape(text)}</span>"


def _rows(q):
    return q.get("members") or [[f, None] for f in q.get("objects", [])]


def _heading(q):
    """(title, plain one-line explanation) for a question card."""
    fac = _facility(q.get("facility"))
    if q["type"] == "meaning":
        n = len(_rows(q))
        tables = f"{n} table{'s' if n != 1 else ''} at {fac}"
        if q.get("column"):
            return (f"What does {q['column']} identify?",
                    f"Used in {tables}. Choose what one {q['column']} value stands for.")
        return ("Which column identifies patients?",
                f"No likely patient ID column was found in {tables}. Name the column, then choose what it stands for.")
    if q["type"] == "scope":
        return (f"Should {fac} be counted?",
                f"Patients from {fac} appear on the dashboard only if the facility is in scope.")
    title, plain = CHECKS.get(q["type"], (q["type"].replace("_", " ").capitalize(), ""))
    return title, plain


def _details(q):
    """The run's own wording and evidence, per table (shown under 'Technical details')."""
    items = []
    for f, col in _rows(q):
        note = (q.get("evidence") or {}).get(f, "")
        items.append(f"<li><code>{escape(_short(f))}</code>{' · <code>' + escape(col) + '</code>' if col else ''}"
                     f"{'<div>' + escape(note[:600]) + ('…' if len(note) > 600 else '') + '</div>' if note else ''}"
                     f"</li>")
    outcome = [f"<li><code>{escape(_short(f))}</code><div>{escape('; '.join(o.get('reasons', []))[:600])}</div></li>"
               for f, o in sorted((q.get("on_person") or {}).items()) if o.get("reasons")]
    after = (f"<div class='x-disc-tech-q'>If answered “Individual patients”, why a table stays open:</div>"
             f"<ul>{''.join(outcome)}</ul>") if outcome else ""
    return (f"<div class='x-disc-tech'><div class='x-disc-tech-q'>{escape(q['ask'][:600])}</div>"
            f"<ul>{''.join(items)}</ul>{after}</div>")


def _write(pairs):
    STORE.check_writes(pairs)          # the same guard as the command line: discovery tables + insert-if-absent
    D._write(pairs)


def _register_due():
    """Register every accepted table whose required answers are now all in. Returns what was registered."""
    questions, _ = STORE.read_questions(D._fetch)
    registry = D._fetch(f"SELECT SOURCE, SOURCE_OBJECT FROM {STORE.REGISTRY}")
    due = Q.due_registrations(questions, STORE.read_statements(D._fetch), registry)
    if due:
        _write([STORE.register_sql(r) for r in due] +
               STORE.record_statements([STORE.registered_statement(r) for r in due]))
        D.read_patient_counts.clear()
    return due


def _record(q, answer, name, note, column=None, dashboard_name=None):
    q = dict(q, column=column or q.get("column"))
    extra = {"dashboard_name": dashboard_name.strip()} if dashboard_name and dashboard_name.strip() else None
    new = Q.to_statements(q, answer, name, note, extra)
    existing = {s["id"]: s for s in STORE.read_statements(D._fetch)}
    new = STORE.merge(new, existing)
    if new:
        _write(STORE.record_statements(new))
    due = _register_due()
    _answers.clear()
    return new, due


def _outcome(q):
    """What answering 'Individual patients' does, per table (computed by the discovery run)."""
    out = q.get("on_person") or {}
    if not out:
        return ""
    rows = []
    for f, o in sorted(out.items()):
        if o["stage"] == "auto":
            chip, what = _chip("Will be counted", "ok"), f"by {escape(o['config']['key'])}, from the next refresh"
        elif o["stage"] == "counted":
            chip, what = _chip("Already counted", "done"), "the answer confirms its patient ID"
        elif o["stage"] == "replacement":
            chip, what = _chip("Needs sign-off", "warn"), "would replace a source that is already counted"
        else:
            chip = _chip("Stays open", "open")
            what = escape(" · ".join(CHECKS.get(c, (c.replace("_", " ").capitalize(),))[0].lower()
                                     for c in o.get("open", [])) or "see technical details")
        rows.append(f"<div class='x-disc-out-row'><span class='x-disc-tbl'>{escape(_table(f))}</span>{chip}"
                    f"<span class='x-disc-out-what'>{what}</span></div>")
    needs = (f"<div class='x-disc-needs'>Also needs the question <b>Should {escape(_facility(q.get('facility')))} "
             f"be counted?</b> answered.</div>") if len(q.get("requires", [])) > 1 else ""
    return (f"<div class='x-disc-out'><div class='x-disc-out-hd'>If you answer “Individual patients”</div>"
            f"{''.join(rows)}{needs}</div>")


def _question(q, name):
    with st.container(border=True, key=f"disc_{q['id']}"):
        title, plain = _heading(q)
        kind = "Decision" if q["type"] in DECISIONS else "Data check"
        tables = "" if q.get("on_person") else (
            "<div class='x-disc-tables'>" + "".join(f"<span>{escape(_table(f))}</span>" for f, _ in _rows(q)) + "</div>")
        T.html(f"<div class='x-disc-tags'><span class='x-disc-fac'>{escape(_facility(q.get('facility')))}</span>"
               f"<span class='x-disc-kind'>{kind}</span></div>"
               f"<div class='x-disc-title'>{escape(title)}</div>"
               f"{'<div class=x-disc-plain>' + escape(plain) + '</div>' if plain else ''}{tables}{_outcome(q)}")
        options = q.get("options", [])
        answer = st.radio("Your answer", options, index=None, horizontal=True, key=f"a_{q['id']}",
                          format_func=lambda o: LABELS.get(o, o))
        column = None
        need_column = q["type"] == "meaning" and not q.get("column")
        has_auto = any(o["stage"] == "auto" for o in (q.get("on_person") or {}).values())
        cols = st.columns([1] * (need_column + has_auto) + [2], vertical_alignment="bottom")
        i = 0
        if need_column:
            column = cols[i].text_input("Identifier column", key=f"c_{q['id']}",
                                        help="No candidate column was found; name the column that identifies the "
                                             "subject.")
            i += 1
        dash = None
        if has_auto:
            dash = cols[i].text_input("Name on the dashboard", key=f"d_{q['id']}",
                                      value=(q.get("facility") or "").replace("_", " ").title(),
                                      help="Used if the answer registers a source.")
            i += 1
        note = cols[-1].text_input("Note (optional)", key=f"n_{q['id']}",
                                   placeholder="How do you know? e.g. confirmed with the facility")
        with st.expander("Technical details"):
            T.html(_details(q))
        ready = bool(name and answer and (q["type"] != "meaning" or q.get("column") or column))
        if answer and not name:
            st.caption("Add your name at the top of the page to record this answer.")
        if st.button("Record answer", key=f"r_{q['id']}", disabled=not ready, type="primary"):
            try:
                new, due = _record(q, answer, name, note, column, dash)
                msg = f"Recorded {len(new)} answer(s)."
                if due:
                    msg += " Registered: " + ", ".join(r["name"] for r in due) + " - counted on the next refresh."
                st.toast(msg)
                st.rerun()
            except Exception as exc:
                st.error(f"Couldn't record the answer ({exc.__class__.__name__}: {exc})")


def render():
    try:
        questions, generated = _questions()
        answered, statements = _answers()
    except Exception as exc:
        T.html(f'<div class="x-empty"><b>Discovery is unavailable</b>Couldn\'t reach the discovery store '
               f'({exc.__class__.__name__}).</div>')
        return
    if not questions and not statements:
        T.html('<div class="x-empty"><b>No discovery questions yet</b>They appear here after the next discovery '
               'run (python -m patient_discovery.run plan).</div>')
        return
    # an answer hides its question only until the next discovery run regenerates the pack
    answered = {k: v for k, v in answered.items() if generated is None or v[1] >= generated}
    # a question about tables that have all been answered 'not a patient source' has nothing left to decide
    settled = {q["id"] for q in questions if q["id"] not in answered and Q.settled(q, statements)}
    open_q = [q for q in questions if q["id"] not in answered and q["id"] not in settled]
    waiting = [q for q in questions if q["id"] in answered]
    decisions = [q for q in open_q if q["type"] in DECISIONS]
    checks = [q for q in open_q if q["type"] not in DECISIONS]

    with st.container(key="disc_top"):
        c1, c2 = st.columns([3, 1.3], vertical_alignment="bottom")
        when = f"{generated:%d %b %Y, %H:%M} EAT" if generated else "not run yet"
        c1.markdown(
            "<div class='x-disc-stats'>"
            f"<div class='x-disc-stat{' hot' if decisions else ''}'><span class='x-lbl'>Decisions waiting</span>"
            f"<b>{len(decisions)}</b></div>"
            f"<div class='x-disc-stat'><span class='x-lbl'>Data checks</span><b>{len(checks)}</b></div>"
            f"<div class='x-disc-stat'><span class='x-lbl'>Answered</span><b>{len(waiting)}</b></div>"
            f"<div class='x-disc-stat'><span class='x-lbl'>Last run</span><span class='x-disc-when'>"
            f"{escape(when)}</span></div></div>", unsafe_allow_html=True)
        name = c2.text_input("Your name", key="disc_name", placeholder="Recorded with each answer",
                             help="Answers are recorded with your name and take effect at once.")
    notes = []
    if waiting:
        by = sorted({answered[q["id"]][0] for q in waiting})
        facs = sorted({_facility(q.get("facility")) for q in waiting})
        notes.append(f"<b>Answered since the last run</b> {escape(', '.join(facs))} · by {escape(', '.join(by))}")
    if settled:
        facs = sorted({_facility(q.get("facility")) for q in questions if q["id"] in settled})
        notes.append(f"<b>Closed: tables set aside as not patient data</b> {len(settled)} question"
                     f"{'s' if len(settled) != 1 else ''} · {escape(', '.join(facs))}")
    if notes:
        T.html("<div class='x-disc-notes'>" + "".join(f"<div>{n}</div>" for n in notes) + "</div>")

    if not open_q:
        T.html("<div class='x-disc-none'><b>Nothing waiting</b>New questions appear here after the next discovery "
               "run.</div>")
    if decisions:
        T.html("<div class='x-disc-sec'><span>Decisions</span><span class='x-disc-sec-h'>Decide what is counted on "
               "the dashboard</span></div>")
        for q in decisions:
            _question(q, name.strip())
    if checks:
        T.html("<div class='x-disc-sec'><span>Data checks</span><span class='x-disc-sec-h'>For the data team: these "
               "never change a count on their own</span></div>")
        for q in checks:
            _question(q, name.strip())

    with st.expander(f"Answer history ({len(statements)})"):
        rows = []
        for s in sorted(statements, key=lambda s: (s["assertion"].get("provenance") or {}).get("at", ""), reverse=True):
            p = s["assertion"].get("provenance") or {}
            a = s["assertion"]
            what = (LABELS.get(a.get("meaning"), a.get("meaning"))
                    or (("In scope" if a.get("onboarded") else "Not in scope") if s["kind"] == "facility" else None)
                    or (f"Registered as {a.get('name')}" if s["kind"] == "registered" else None)
                    or LABELS.get(s["kind"], s["kind"].replace("_", " ").capitalize()))
            at = str(p.get("at") or "")
            rows.append({"When": at.replace("T", " ")[:16], "Decision": what, "About": s.get("subject"),
                         "By": p.get("by"), "Note": p.get("basis"), "Reference": s["id"]})
        st.dataframe(rows, hide_index=True, width="stretch")
