"""Critical-rule mutation gate. Each mutation re-introduces one flaw (most found in the 2026-11-09 adversarial review);
the model suite (Scenarios, Replay and the operational tests) must fail under every one of them.

Run:  cd analytics_app/dashboards/internal_reporting && python -m patient_discovery.tests.mutation_gate
Exit code 1 if any mutation survives.
"""
import contextlib
import sys
import time
import unittest

from patient_discovery import config as C
from patient_discovery.classify import family
from patient_discovery.model import interpret as I
from patient_discovery.model.concepts import Candidate
from patient_discovery.tests import model_world as W
from patient_discovery.tests import test_model, test_operations, test_rules

MUTATIONS = {}


def mutation(fn):
    MUTATIONS[fn.__name__] = contextlib.contextmanager(fn)
    return fn


@contextlib.contextmanager
def patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def thresholds(**kw):
    old = dict(I.T)
    I.T.update(kw)
    try:
        yield
    finally:
        I.T.clear()
        I.T.update(old)


# ------------------------------------------------------------------ meaning
@mutation
def meaning_from_containment():
    """Values contained in a namespace member give meaning (no derivation)."""
    orig = I._Run.derivation

    def derivation(self, O, oc, X, xc):
        step = orig(self, O, oc, X, xc)
        if step:
            return step
        c = self.contain(O, (O.fqn, oc.column), (X.fqn, xc.column))
        return "contained" if c and c["da"] and c["a_in_b"] >= I.T["contained"] else None
    with patched(I._Run, "derivation", derivation):
        yield


@mutation
def meaning_from_registers():
    """The old rule: values found in a unique column of a table named like a register give meaning."""
    orig = I._Run.meanings

    def meanings(self):
        orig(self)
        for O in self.A.values():
            if O.stage != "pending" or not O.candidates:
                continue
            c = O.candidates[0]
            for R in self.A.values():
                if R.grain.event_class != "registration" or R.facility != O.facility or c.meaning:
                    continue
                for rc in R.candidates:
                    v = self.contain(O, (O.fqn, c.column), (R.fqn, rc.column))
                    if v and v["da"] and v["a_in_b"] >= I.T["contained"]:
                        c.meaning, c.namespace, c.provenance = "person", "register", ["register"]
    with patched(I._Run, "meanings", meanings):
        yield


@mutation
def meaning_from_layout():
    """The old rule: the column layout of two counted sources that count this column gives meaning."""
    orig = I._Run.meanings

    def meanings(self):
        orig(self)
        for O in self.A.values():
            mine = {c.upper() for c, _ in self.cat[O.fqn]["columns"]}
            for c in O.candidates:
                if O.stage != "pending" or c.meaning:
                    continue
                n = sum(1 for f, s in self.pol.sources.items() if f != O.fqn and f in self.cat
                        and s["key"].upper() == c.column.upper()
                        and len(mine & {x.upper() for x, _ in self.cat[f]["columns"]}) >= 6)
                if n >= 2:
                    c.meaning, c.namespace, c.provenance = "person", "layout", ["layout"]
    with patched(I._Run, "meanings", meanings):
        yield


@mutation
def scope_ignored():
    with patched(I._Run, "in_scope", lambda self, O, ns: True):
        yield


@mutation
def alignment_ignores_agreement():
    """Shared record IDs alone count as derivation (identifier values not compared)."""
    orig = I._Run.align

    def align(self, a, x, y, why):
        v = orig(self, a, x, y, why)
        return v and dict(v, agree=v["shared"], disagree=0)
    with patched(I._Run, "align", align):
        yield


@mutation
def formats_ignored():
    with patched(I._Run, "formats_ok", lambda self, O, oc, X, xc: "formats not checked"):
        yield


@mutation
def rekeying_ignored():
    orig = I._Run.align

    def align(self, a, x, y, why):
        v = orig(self, a, x, y, why)
        return v and dict(v, disagree=0)
    with patched(I._Run, "align", align):
        yield


@mutation
def containment_means_same_people():
    """The old rule: values inside a counted source's values mean the same people (subset / double count)."""
    with patched(I._Run, "same_namespace", lambda self, S, X: True):
        yield


# ------------------------------------------------------------------ versions and relations
@mutation
def any_window_reconciles():
    with thresholds(reconcile=1.0):
        yield


@mutation
def everything_stopped():
    with patched(I._Run, "stopped", lambda self, f: True):
        yield


@mutation
def view_of_any_counted_ref():
    """The old rule: a view reading any counted source is that source's view."""
    orig = I._Run.relate

    def relate(self, a):
        refs = self.cat[a.fqn].get("refs") or []
        if a.object_type != "BASE TABLE" and any(r in self.pol.sources for r in refs):
            return self.terminal(a, "not_a_source", "view of counted")
        return orig(self, a)
    with patched(I._Run, "relate", relate):
        yield


@mutation
def facility_policy_global():
    orig = I.Policy.__init__

    class Everywhere(dict):
        def __contains__(self, k):
            return bool(len(self)) or dict.__contains__(self, k)

        def __getitem__(self, k):
            return next(iter(self.values()))

    def init(self, statements):
        orig(self, statements)
        self.separate = Everywhere(self.separate)
    with patched(I.Policy, "__init__", init):
        yield


@mutation
def held_out_statements_leak():
    """The old replay builder: onboarding and namespaces from every source, the Kakamega policy kept."""
    def leaky(held_out=None):
        onboard = {family(s.subject.split(".")[1]) for s in W.SOURCES.values()}
        return [s for n, s in W.SOURCES.items() if n != held_out] + W.REJECTED + list(W.NAMESPACES.values()) + \
            W.policies(onboard, set(W.SEPARATE))
    with patched(W, "replica_statements", leaky):
        yield


@mutation
def noise_filter_off():
    """Unrecognised tables with only generic-named candidates are asked about again."""
    with patched(I._Run, "nothing_points_to_patients", lambda self, a: False):
        yield


@mutation
def noise_filter_ignores_value_links():
    """The filter sets aside generic-named columns even when their values sit in a counted patient identifier."""
    with patched(I._Run, "nothing_points_to_patients",
                 lambda self, a: not any(c.meaning or c.hint in ("stated", "registered", "listed", "person")
                                         for c in a.candidates)):
        yield


@mutation
def link_checks_after_meaning():
    """Derivation is still measured on tables whose meaning a statement already set (wasted statements)."""
    def meanings(self):
        for a in self.A.values():
            for c in a.candidates:
                st = self.pol.members.get((a.fqn, c.column.upper()))
                if st:
                    c.meaning, c.namespace = st.assertion["meaning"], st.id
                    c.provenance = [st.id]
                    self.holders.append((a, c))
        for X, xc in list(self.holders):
            for O in self.A.values():
                if O is not X and self.in_scope(O, xc.namespace) and O.stage in ("pending", "counted"):
                    for oc in O.candidates[:I.T["profiled"]]:
                        if oc.meaning is None:
                            self.derivation(O, oc, X, xc)
    with patched(I._Run, "meanings", meanings):
        yield


@mutation
def id_endings_as_letters():
    """The old rule: any name whose letters end in ID / NO / KEY… is an identifier candidate (AMOUNT_PAID)."""
    with patched(I, "id_named", lambda up: up == "ID" or up.replace("_", "").endswith(I.ID_ENDINGS)):
        yield


# ------------------------------------------------------------------ registration
@mutation
def registration_ignores_required_answers():
    from patient_discovery.model import questions as Qm
    orig = Qm.due_registrations

    def due(questions, statements, registry_rows):
        return orig([dict(q, requires=[q["id"]]) for q in questions], statements, registry_rows)
    with patched(Qm, "due_registrations", due):
        yield


@mutation
def registration_of_any_outcome():
    from patient_discovery.model import questions as Qm
    orig = Qm.due_registrations

    def due(questions, statements, registry_rows):
        qs = [dict(q, on_person={f: dict(o, stage="auto", config=o.get("config") or {"key": "X"})
                                 for f, o in q.get("on_person", {}).items()}) for q in questions]
        return orig(qs, statements, registry_rows)
    with patched(Qm, "due_registrations", due):
        yield


@mutation
def registry_write_guard_open():
    from patient_discovery.model import store
    with patched(store, "check_writes", lambda pairs: None):
        yield


# ------------------------------------------------------------------ candidates and behaviour
@mutation
def record_level_candidates_dropped():
    """The old rule: a once-per-row candidate finer than a repeating one is silently dismissed."""
    orig = I._Run.select

    def select(self, a):
        if any(c.verdict == "consistent" for c in a.candidates):
            a.candidates = [c for c in a.candidates if c.verdict != "insufficient"]
        return orig(self, a)
    with patched(I._Run, "select", select):
        yield


@mutation
def contradicting_candidates_hidden():
    orig = I._Run.select

    def select(self, a):
        if a.candidates and not any(c.verdict == "screened" for c in a.candidates):
            a.candidates = [c for c in a.candidates if c.verdict != "contradicts" or c.meaning]
        return orig(self, a)
    with patched(I._Run, "select", select):
        yield


@mutation
def single_visit_check_off():
    with thresholds(single_rpv=0.0, single_share=2.0):
        yield


@mutation
def equals_record_check_off():
    with thresholds(eq_record=10.0):
        yield


@mutation
def duplicate_rows_check_off():
    with thresholds(dup_rows=0.0):
        yield


@mutation
def several_values_per_record_check_off():
    with thresholds(stable=100.0):
        yield


@mutation
def small_table_check_off():
    with thresholds(small=0):
        yield


@mutation
def dominant_value_check_off():
    with patched(C, "DOMINANT_KEY_SHARE", 10.0):
        yield


@mutation
def deleted_share_check_off():
    with patched(C, "DELETED_SHARE", 2.0):
        yield


@mutation
def future_dates_check_off():
    with patched(I._Run, "today", lambda self: "9999-12-31"):
        yield


@mutation
def file_rows_check_off():
    with patched(C, "FILE_COLUMNS", []):
        yield


def suite():
    loader = unittest.TestLoader()
    return unittest.TestSuite([loader.loadTestsFromTestCase(test_model.Scenarios),
                               loader.loadTestsFromTestCase(test_model.Replay),
                               loader.loadTestsFromModule(test_operations),
                               loader.loadTestsFromTestCase(test_rules.Rules)])


def main():
    base = unittest.TextTestRunner(stream=open("nul" if sys.platform == "win32" else "/dev/null", "w"),
                                   verbosity=0).run(suite())
    if base.failures or base.errors:
        print("baseline suite fails; fix it before running the gate")
        return 2
    survived = []
    for name, ctx in MUTATIONS.items():
        t0 = time.time()
        with ctx():
            r = unittest.TextTestRunner(stream=open("nul" if sys.platform == "win32" else "/dev/null", "w"),
                                        verbosity=0).run(suite())
        failed = [t.id().split(".")[-1] + (f" {t._subDescription()}" if hasattr(t, "_subDescription") else "")
                  for t, _ in r.failures + r.errors]
        status = "killed" if failed else "SURVIVED"
        if not failed:
            survived.append(name)
        print(f"{status:9} {name:38} {len(failed):3} failing  {time.time() - t0:5.1f}s  {'; '.join(failed[:3])[:120]}")
    print(f"\n{len(MUTATIONS) - len(survived)} of {len(MUTATIONS)} mutations killed")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
