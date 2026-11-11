"""Questions for people, grouped so each fact is asked once, and answers recorded as statements.

Groups:
  meaning  one question per (facility, identifier column) - every object whose identifier meaning is unresolved,
           including counted sources whose key meaning was never confirmed. The answer names the members explicitly,
           so it never reaches an object the person did not see; derivation then carries it to versions and views.
  scope    one question per facility not yet onboarded.
  other    one question per (code, facility): grain, kind of table, value relations, derived views, ...
Answer in the dashboard's Discovery tab (or fill in "answer" and run `python -m patient_discovery.run answer`).

Each meaning question carries `on_person`: what answering "person" (with the facility in scope) does to each member,
computed by the same interpreter during the discovery run. `due_registrations` turns recorded answers plus those
outcomes into registry rows, so an answer takes effect immediately - without re-measuring anything.
"""
import hashlib
import json
from pathlib import Path

from . import statements as S

MEANINGS = ["person", "household", "insurance_member", "file", "visit", "other", "not_sure"]
MEANING_CODES = ("MEANING_UNRESOLVED", "NO_USABLE_IDENTIFIER", "NO_CANDIDATE")


def _qid(*parts):
    return "Q-" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:8]


def build(beliefs):
    A, groups = beliefs.assessments, {}

    def group(key, make):
        if key not in groups:
            groups[key] = make()
        return groups[key]

    def meaning(facility, column):
        what = column or "(no candidate column: name one in 'column')"
        return group(("meaning", facility, (column or "?").upper()), lambda: dict(
            id=_qid("meaning", facility, (column or "?").upper()), type="meaning", facility=facility, column=column,
            ask=f"In {facility}, what does {what} identify?",
            options=MEANINGS, answer=None, note="", members=[], evidence={}))

    for f, info in sorted(beliefs.counted.items()):
        a = A[f]
        if info["meaning"] is None and a.selected is not None:
            g = meaning(a.facility, a.selected.column)
            g["members"].append([f, a.selected.column])
            g["evidence"][f] = f"counted today as '{a.counted.get('source')}'; meaning never confirmed"
    for a in sorted(A.values(), key=lambda a: a.fqn):
        if a.stage != "question":
            continue
        for q in a.questions:
            if q.code in MEANING_CODES:
                col = a.selected.column if a.selected else None
                g = meaning(a.facility, col)
                if [a.fqn, col] not in g["members"]:
                    g["members"].append([a.fqn, col])
                g["evidence"][a.fqn] = q.text
            elif q.code == "SCOPE_UNKNOWN":
                g = group(("scope", a.facility), lambda: dict(
                    id=_qid("scope", a.facility), type="scope", facility=a.facility,
                    ask=f"Is {a.facility} in scope for patient counting?", options=["yes", "no"], answer=None,
                    note="", objects=[]))
                g["objects"].append(a.fqn)
            else:
                g = group((q.code, a.facility), lambda: dict(
                    id=_qid(q.code, a.facility), type=q.code, facility=a.facility, ask=q.text,
                    options=["not_a_source", "keep_asking"], answer=None, note="", objects=[], evidence={}))
                g["objects"].append(a.fqn)
                g["evidence"][a.fqn] = q.text
    return sorted(groups.values(), key=lambda g: (g["type"] != "meaning", g["type"], -len(g.get("members", g.get(
        "objects", []))), g["facility"]))


def write(groups, path):
    Path(path).write_text(json.dumps({"questions": groups}, indent=2) + "\n", encoding="utf-8")


def to_statements(q, answer, by, note="", extra=None):
    """The statements one answered question becomes (with provenance). [] for not_sure / keep_asking."""
    ans = (answer or "").strip()
    if not ans or ans in ("not_sure", "keep_asking"):
        return []
    prov = S.provenance(q["id"], by, note or "answered from the question pack")
    if q["type"] == "meaning":
        if ans not in MEANINGS:
            raise ValueError(f"{q['id']}: answer must be one of {MEANINGS}")
        col = q.get("column")
        members = [[f, c or col] for f, c in q["members"]]
        if not col or any(c is None for _, c in members):
            raise ValueError(f"{q['id']}: name the identifier column in 'column'")
        return [dict(id=f"N:{q['facility']}:{col.upper()}", kind="namespace", subject=f"{q['facility']} {col}",
                     origin="human", assertion=dict(meaning=ans, members=members, scope=dict(facility=q["facility"]),
                                                    provenance=prov, **(extra or {})))]
    if q["type"] == "scope":
        out = [dict(id=f"F:{q['facility']}", kind="facility", subject=q["facility"], origin="human",
                    assertion=dict(onboarded=ans == "yes", provenance=prov))]
        if ans == "no":
            out += [dict(id=f"X:{f}", kind="not_a_source", subject=f, origin="human",
                         assertion=dict(provenance=prov)) for f in q["objects"]]
        return out
    if ans == "not_a_source":
        return [dict(id=f"X:{f}", kind="not_a_source", subject=f, origin="human", assertion=dict(provenance=prov))
                for f in q["objects"]]
    raise ValueError(f"{q['id']}: unexpected answer {ans!r}")


def record(pack_path, by, path=S.FILE):
    """Answered questions from a pack file -> statements appended to statements.json. Returns them."""
    pack = json.loads(Path(pack_path).read_text(encoding="utf-8"))["questions"]
    new = [s for q in pack for s in to_statements(q, q.get("answer"), by, q.get("note"))]
    if new:
        S.append(new, path)
    return new


def hypothesis(g, statements):
    """The statements answering 'person' (and 'in scope') to meaning question g would add, or None."""
    from .concepts import Statement
    if g["type"] != "meaning" or not g.get("column"):
        return None
    fac, col = g["facility"], g["column"]
    onboarded = {s.subject for s in statements if s.kind == "facility" and s.assertion.get("onboarded")}
    hyp = [Statement(f"N:{fac}:{col.upper()}", "namespace", "if answered", dict(
        meaning="person", members=[[f, c or col] for f, c in g["members"]], scope=dict(facility=fac)))]
    if fac not in onboarded:
        hyp.append(Statement(f"F:{fac}", "facility", fac, {"onboarded": True}))
    ids = {h.id for h in hyp}
    return [s for s in statements if s.id not in ids] + hyp


def with_outcomes(groups, catalogue, observations, statements, now):
    """Add to each meaning question what answering 'person' would do: for every member, the stage and (if accepted)
    the counting configuration, interpreted with that answer (and the facility in scope) added. Also `requires`:
    the questions that must be answered for those outcomes to hold. A 'pending' outcome means a measurement is
    still needed; `profile` takes those measurements so outcomes are known before anyone answers."""
    from .interpret import interpret
    onboarded = {s.subject for s in statements if s.kind == "facility" and s.assertion.get("onboarded")}
    scope_q = {g["facility"]: g["id"] for g in groups if g["type"] == "scope"}
    for g in groups:
        hyp = hypothesis(g, statements)
        if hyp is None:
            continue
        fac = g["facility"]
        b = interpret(catalogue, observations, hyp, now)
        g["on_person"] = {}
        for f, _ in g["members"]:
            a = b.assessments.get(f)
            if a is not None:
                g["on_person"][f] = dict(stage=a.stage, config=dict(a.config) if a.stage == "auto" else None,
                                         open=sorted(a.codes()), reasons=a.reasons[:2])
        g["requires"] = [g["id"]] + ([scope_q[fac]] if fac not in onboarded and fac in scope_q else [])
    return groups


def _answers(statements):
    """{question id: (answer, statement)} from recorded statements (latest per question)."""
    out = {}
    for s in statements:
        p = (s.get("assertion") or {}).get("provenance") or {}
        q = p.get("question")
        if not q:
            continue
        if s["kind"] == "namespace":
            out[q] = (s["assertion"]["meaning"], s)
        elif s["kind"] == "facility":
            out[q] = ("yes" if s["assertion"].get("onboarded") else "no", s)
    return out


def tables_of(q):
    """The tables a question is about."""
    return [m[0] for m in q.get("members") or []] or list(q.get("objects") or [])


def settled(q, statements):
    """True when every table the question is about has been answered 'not a patient source' (directly, or through
    a facility answered 'not in scope'): nothing the question could still decide. A scope question closes the same
    way: with every table of the facility excluded, there is nothing left to count."""
    excluded = {s["subject"] for s in statements if s["kind"] == "not_a_source"}
    tables = tables_of(q)
    return bool(tables) and all(f in excluded for f in tables)


def due_registrations(questions, statements, registry_rows):
    """Registry rows that recorded answers now justify: members whose precomputed outcome is 'auto', for a meaning
    question answered 'person' whose required questions are all answered positively, not already registered.
    Pure: the caller writes them."""
    answers = _answers(statements)
    registered = {(r.get("SOURCE_OBJECT") or "").strip().upper() for r in registry_rows}
    names = {(r.get("SOURCE") or "").strip() for r in registry_rows}
    out = []
    for q in questions:
        if not q.get("on_person") or answers.get(q["id"], (None,))[0] != "person":
            continue
        if any(answers.get(r, (None,))[0] not in ("person", "yes") for r in q.get("requires", [])):
            continue
        s = answers[q["id"]][1]
        name = (s["assertion"].get("dashboard_name") or "").strip() or q["facility"].replace("_", " ").title()
        autos = [(f, o) for f, o in sorted(q["on_person"].items())
                 if o["stage"] == "auto" and o.get("config") and f.upper() not in registered]
        for f, o in autos:
            nm = name if len(autos) == 1 else f"{name} ({f.split('.')[-1]})"
            if nm in names:
                continue
            names.add(nm)
            out.append(dict(object=f, name=nm, config=o["config"], provenance=dict(
                question=q["id"], answers=[answers[r][1]["id"] for r in q.get("requires", [])],
                by=(s["assertion"].get("provenance") or {}).get("by"), basis="discovery: confirmed and accepted")))
    return out
