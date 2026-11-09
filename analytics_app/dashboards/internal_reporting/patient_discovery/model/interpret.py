"""Interpreter: catalogue + current observations + statements -> beliefs.

A pure function: no I/O and no clock (`now` is passed in). Machine conclusions are never stored; they are re-derived
on every call. A missing measurement comes back as a Gap for the planner. An observation is used only while its input
fingerprint matches the catalogue.

The semantic boundary (see concepts.py):
- statistics (behaviour) nominate and evaluate candidates; they never give a column a meaning;
- meaning comes from a namespace statement, directly or by derivation inside the statement's scope;
- derivation = record alignment (shared records carry the same identifier value, values keep the namespace's
  formats) or direct lineage with the same value set. Containment, names, layouts and policies relate or rank,
  never mean;
- a dataset is 'auto' only when its selected identifier means 'person' and it is eligible.
"""
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta

from .. import config as C
from ..classify import copy_schema, family, role, tokens
from .concepts import Assessment, Candidate, Gap, Question, Relation
from .measure import MEASURE_VERSION

# Thresholds: behaviour checks (existing values from config.py) and one tolerance, `contained`, used for every
# 'almost all' comparison (containment, alignment, formats). No threshold decides meaning.
T = dict(complete=1 - C.MISSING_KEY_SHARE, stable=1.01, eq_record=C.KEY_EQUALS_RECORD_SHARE, single_rpv=1.10,
         single_share=C.ONE_ROW_PER_KEY_SHARE, many_rpv=C.MANY_RECORDS_PER_PERSON, variant=C.KEY_VARIANT_SHARE,
         small=C.SMALL_TABLE_ROWS, dup_rows=0.99, overlap=0.05, contained=0.95, conditional_pure=0.95,
         reconcile=0.01, stopped_days=30, max_candidates=12, profiled=4, pair_others=3, default_rows=20000)

# Nomination hints. Names nominate and rank; they never decide.
ID_ENDINGS = ("ID", "NO", "NUM", "NUMBER", "CODE", "UUID", "KEY", "NBR", "HASH")
# words that merely end in an identifier ending (AMOUNT_PAID, IS_VALID, THYROID): never nominated by name
NOT_ID_WORDS = {"PAID", "UNPAID", "PREPAID", "POSTPAID", "OVERPAID", "UNDERPAID", "VALID", "INVALID", "VOID", "AVOID",
                "FLUID", "FLUIDS", "LIQUID", "ACID", "RAPID", "SOLID", "HYBRID", "MID", "HUMID", "LUCID", "MORBID",
                "PYRAMID", "THYROID", "STEROID", "OPIOID", "ALKALOID", "ANDROID", "CANDID", "AID", "LID", "GRID",
                "BID", "SAID", "RAID", "AMID", "CASINO", "PIANO", "VOLCANO", "MONKEY", "TURKEY", "DONKEY", "JOCKEY",
                "HOCKEY"}


def id_named(up):
    """True when an (upper-case) column name ends in an identifier ending as a word, not just as letters:
    PATIENT_ID, PATIENTID, PHONE_NO yes; AMOUNT_PAID, IS_VALID, THYROID_LEVEL no."""
    last = [t for t in up.split("_") if t][-1:] or [""]
    last = last[0]
    if last in ID_ENDINGS:
        return True
    return last.endswith(ID_ENDINGS) and last not in NOT_ID_WORDS
PERSON_STEMS = ("PATIENT", "CLIENT", "PERSON", "MEMBER", "MRN", "MRD", "REGID", "REGNO", "REG_NO", "UPI", "CCC",
                "OPNO", "OP_NO", "IPNO", "IP_NO", "PNO")
PRIVACY = ("NAME", "PHONE", "MOBILE", "TELNO", "TEL_NO", "TELEPHONE", "EMAIL", "ADDRESS", "BIRTH", "DOB", "NATIONAL",
           "PASSPORT", "KRA")                  # never profiled, whatever the evidence
ENTITY = ("VISIT", "BILL", "ITEM", "INVOICE", "RECEIPT", "TRANSACTION", "ACCOUNT", "DOCTOR", "USER", "STAFF",
          "EMPLOYEE", "FACILITY", "WARD", "BED", "DEPOSIT", "ORDER", "PRESC", "LAB", "TEST", "SAMPLE", "STOCK",
          "PAYMENT", "POLICY", "GROUP", "FILE", "DOCUMENT", "PAGE", "ADMIT", "ADMISSION", "APT", "BENEFIT",
          "DEPARTMENT", "CLINIC", "SERVICE")    # lowers the rank only
FREE_TEXT = ("NOTE", "COMMENT", "DIAGNOS", "DESCRIPTION", "HISTORY", "COMPLAINT", "REMARK", "FINDING", "NARRATIVE",
             "TEXT", "REASON", "INSTRUCTION", "PRESCRIPTION", "RESULT")   # never grouped or counted by value
EVENT_FROM_ROLE = {"encounter": "encounter", "register": "registration", "satellite": "supporting",
                   "scheduling": "scheduling"}
PRIORITY = {"encounter": 0, "registration": 0, "admission": 1}


def short(f):
    return f.split(".", 1)[1]


def nominate(columns, exclude, stated=()):
    """(candidates, truncated column names). Ranked: named in a statement, listed name, person stem, generic."""
    out = []
    for i, (col, t) in enumerate(columns):
        up = col.upper()
        if up in exclude:
            continue
        if up in stated:
            out.append((-1.0, i, Candidate(col, t, "stated", -1.0)))
            continue
        if t not in C.KEY_TYPES or any(p in up for p in PRIVACY):
            continue
        tier = C.PERSON_KEYS.get(up)
        if tier is None and not id_named(up):
            continue
        if tier in (1, 2):
            hint, rank = "listed", 0.0
        elif tier == 3 or any(s in up for s in PERSON_STEMS):
            hint, rank = "person", 1.0
        else:
            hint, rank = "generic", 2.0
        if hint != "listed" and any(e in up for e in ENTITY):
            rank += 1.0
        out.append((rank, i, Candidate(col, t, hint, rank)))
    out.sort(key=lambda x: (x[0], x[1]))
    cands = [c for _, _, c in out]
    return cands[:T["max_candidates"]], [c.column for c in cands[T["max_candidates"]:]]


def fingerprints(catalogue):
    """Input fingerprint per object: base tables by last-altered, row count and columns; views by their columns,
    last-altered and the fingerprints of what they read (so a view changes when its tables change)."""
    cat = {o["fqn"]: o for o in catalogue}
    memo = {}

    def fp(f, depth=0):
        if f in memo:
            return memo[f]
        o = cat.get(f)
        if o is None:
            return f"?{f}"
        sig = ",".join(f"{c}:{t}" for c, t in sorted(o["columns"]))
        if o["type"] == "BASE TABLE" or o.get("store"):
            raw = f"{MEASURE_VERSION}|T|{o['last_altered']}|{o['row_count']}|{sig}"
        elif o.get("refs") is None or depth > 5:
            raw = f"{MEASURE_VERSION}|V|{o['last_altered']}|{sig}|?"
        else:
            raw = f"{MEASURE_VERSION}|V|{o['last_altered']}|{sig}|" + ";".join(fp(r, depth + 1) for r in sorted(o["refs"]))
        memo[f] = hashlib.sha1(raw.encode()).hexdigest()[:16]
        return memo[f]

    return {f: fp(f) for f in cat}


def fp_of(fps, subject):
    return "|".join(fps.get(f, "?") for f in subject)


class Policy:
    """Statements by kind: source, rejected, not_a_source, exclude_schemas, hold_schemas, covered_schemas,
    derived_schemas, facility, event_class, unlinked_populations (per facility), namespace."""

    def __init__(self, statements):
        self.sources, self.rejected, self.ignored = {}, {}, {}
        self.excluded, self.held, self.covered, self.derived = {}, {}, {}, {}
        self.onboarded, self.facility_map, self.events = {}, {}, {}
        self.namespaces, self.members, self.separate = {}, {}, {}
        for s in statements:
            k, a = s.kind, s.assertion
            if k == "source":
                self.sources[s.subject] = dict(a, statement=s.id)
            elif k == "rejected":
                self.rejected[s.subject] = s
            elif k == "not_a_source":
                self.ignored[s.subject] = s
            elif k in ("exclude_schemas", "hold_schemas", "derived_schemas"):
                target = {"exclude_schemas": self.excluded, "hold_schemas": self.held, "derived_schemas": self.derived}[k]
                target.update({x: s for x in a["schemas"]})
            elif k == "covered_schemas":
                self.covered.update({x: (src, s) for x, src in a["schemas"].items()})
            elif k == "facility":
                if a.get("onboarded"):
                    self.onboarded[s.subject] = s
                self.facility_map.update({x: s.subject for x in a.get("schemas", ())})
            elif k == "event_class":
                self.events[s.subject] = s
            elif k == "unlinked_populations":
                if a.get("action") == "count_separately":
                    self.separate[s.subject] = s
            elif k == "registered":
                pass                      # an audit entry; the registry itself says what is counted
            elif k == "namespace":
                self.namespaces[s.id] = s
                for f, col in a["members"]:
                    self.members[(f, col.upper())] = s
            else:
                raise ValueError(f"unknown statement kind {k}")

    def facility_of(self, schema):
        return self.facility_map.get(schema) or family(schema)

    def scope(self, ns):
        return self.namespaces[ns].assertion["scope"]["facility"]


@dataclass
class Beliefs:
    assessments: dict
    namespaces: dict       # namespace id -> [(fqn, column, provenance)]
    questions: list        # (Question, [fqns]) most datasets first
    gaps: list
    counted: dict          # counted fqn -> meaning of its key (None = never established) and provenance


def interpret(catalogue, observations, statements, now=None):
    return _Run(catalogue, observations, statements, now).run()


class _Run:
    def __init__(self, catalogue, observations, statements, now):
        self.pol, self.now = Policy(statements), now
        self.cat = {o["fqn"]: o for o in catalogue}
        self.fps = fingerprints(catalogue)
        self.obs, self.screen_rows = {}, {}
        for o in observations:
            if o.fp == fp_of(self.fps, o.subject) and not self.expired(o):
                self.obs[(o.kind, o.subject, o.args)] = o.values
                if o.kind == "screen":
                    self.screen_rows[o.subject[0]] = o.values["rows"]
        self.A, self.holders = {}, []

    # ------------------------------------------------------------------ helpers
    def expired(self, o):
        """Views carry no reliable change marker for the data they read (raw stores, hidden lineage): an observation
        involving a view is used for at most VIEW_PROFILE_MAX_AGE_DAYS."""
        if not (self.now and getattr(o, "at", None)):
            return False
        if not any(self.cat.get(f, {}).get("type") not in ("BASE TABLE", None) for f in o.subject):
            return False
        try:
            return datetime.fromisoformat(o.at) < self.now - timedelta(days=C.VIEW_PROFILE_MAX_AGE_DAYS)
        except ValueError:
            return True

    def type_of(self, f, col):
        return dict(self.cat[f]["columns"]).get(col, "TEXT")

    def rows_of(self, f):
        if f in self.screen_rows:
            return self.screen_rows[f]
        rc = self.cat[f]["row_count"]
        return rc if rc is not None else self.cat[f].get("row_estimate", T["default_rows"])

    def cost(self, kind, subject, args):
        r = [self.rows_of(f) for f in subject]
        return {"screen": r[0] * (1 + 0.5 * len(args[0])), "profile": 2 * r[0], "pair": 3 * r[0],
                "shape": 2 * r[0], "containment": 2 * r[0] + (r[1] if len(r) > 1 else 0), "window": r[0],
                "categories": r[0] * (1 + 0.2 * len(args)), "conditional": r[0],
                "align": 2 * sum(r), "formats": sum(r)}[kind]

    def get(self, a, kind, subject, args, blocking=True, why=""):
        v = self.obs.get((kind, subject, args))
        if v is None:
            a.gaps.append(Gap(kind, subject, args, self.cost(kind, subject, args), blocking, why,
                              PRIORITY.get(a.grain.event_class, 2)))
        return v

    def stopped(self, f):
        try:
            la = datetime.fromisoformat(str(self.cat[f]["last_altered"])[:19])
        except (TypeError, ValueError):
            return None
        return None if self.now is None else la < self.now - timedelta(days=T["stopped_days"])

    def terminal(self, a, stage, reason):
        a.stage = stage
        a.reasons.append(reason)
        return a

    def ask(self, a, code, text, level="object", subject=None):
        a.questions.append(Question(code, level, subject or a.fqn, text))

    def screen_args(self, a):
        g = a.grain
        rk = (g.record_key, self.type_of(a.fqn, g.record_key)) if g.record_key else None
        return (tuple((c.column, c.data_type) for c in a.candidates), rk, g.date)

    def screen(self, a, of=None):
        of = of or a
        return self.get(a, "screen", (of.fqn,), self.screen_args(of), why="screen candidates")

    def record_key(self, x):
        rk = (x.counted or {}).get("record") or x.grain.record_key
        return rk if rk and rk.upper() in {c.upper() for c, _ in self.cat[x.fqn]["columns"]} else None

    def compatible(self, a, x, y, blocking=True):
        """Value shapes of two identifier columns. None = not measured yet; False = clearly different namespaces."""
        sx = self.get(a, "shape", (x[0],), ((x[1], self.type_of(*x)),), blocking, "value shapes")
        sy = self.get(a, "shape", (y[0],), ((y[1], self.type_of(*y)),), blocking, "value shapes")
        if sx is None or sy is None:
            return None
        dx = sx["top"][0][0] if sx["top"] and sx["top"][0][1] >= .5 else None
        dy = sy["top"][0][0] if sy["top"] and sy["top"][0][1] >= .5 else None
        if dx and dy and dx != dy and dx not in {s for s, _ in sy["top"]} and dy not in {s for s, _ in sx["top"]}:
            return False
        return True

    def contain(self, a, x, y, why="containment", blocking=True):
        """Containment between (fqn, column) x and y, measured once per unordered pair; oriented as x within y."""
        first, second = sorted([x, y])
        v = self.get(a, "containment", (first[0], second[0]),
                     ((first[1], self.type_of(*first)), (second[1], self.type_of(*second))), blocking, why)
        if v is None or first == x:
            return v
        return dict(da=v["db"], db=v["da"], inter=v["inter"], a_in_b=v["b_in_a"], b_in_a=v["a_in_b"])

    def align(self, a, x, y, why):
        """x, y = (fqn, record key, identifier column): records shared by record ID and whether they carry the same
        identifier value. Measured once per unordered pair; oriented so ra counts x's records."""
        first, second = sorted([x, y])
        args = tuple((c, self.type_of(f, c)) for f, rk, col in (first, second) for c in (rk, col))
        v = self.get(a, "align", (first[0], second[0]), args, why=why)
        if v is None or first == x:
            return v
        return dict(v, ra=v["rb"], rb=v["ra"])

    # ------------------------------------------------------------------ steps
    def run(self):
        for f, o in self.cat.items():
            self.A[f] = self.base(o)
        for a in self.A.values():
            self.early(a)
        live = [a for a in self.A.values() if a.stage == "pending"]
        for a in live:
            self.measure_object(a)
        self.meanings()
        for a in live:
            if a.stage == "pending":
                self.select(a)
        for a in live:
            if a.stage == "pending":
                self.relate(a)
        self.representations([a for a in live if a.stage == "pending"])
        for a in live:
            if a.stage in ("pending", "replacement"):
                self.decide(a)
        questions = defaultdict(list)
        for a in self.A.values():
            if a.stage in ("question", "counted"):
                for q in a.questions:
                    questions[q].append(a.fqn)
        ranked = sorted(((q, sorted(set(fs))) for q, fs in questions.items()),
                        key=lambda x: (-len(x[1]), x[0].level, x[0].code, x[0].subject))
        gaps, seen = [], set()
        for a in self.A.values():
            for g in a.gaps:
                if (g.kind, g.subject, g.args) not in seen:
                    seen.add((g.kind, g.subject, g.args))
                    gaps.append(g)
        namespaces = defaultdict(list)
        for a in self.A.values():
            for c in a.candidates:
                if c.namespace:
                    namespaces[c.namespace].append((a.fqn, c.column, c.provenance))
        counted = {f: dict(meaning=a.meaning, namespace=a.selected.namespace if a.selected else None,
                           provenance=a.selected.provenance if a.selected else [])
                   for f, a in self.A.items() if a.counted}
        return Beliefs(self.A, dict(namespaces), ranked, gaps, counted)

    def base(self, o):
        f, schema, name, cols = o["fqn"], o["schema"], o["name"], o["columns"]
        a = Assessment(f, schema, name, o["type"], self.pol.facility_of(schema))
        up = {c.upper(): c for c, _ in cols}
        first = lambda opts: next((up[n] for n in opts if n in up), None)
        g = a.grain
        g.record_key = first(C.RECORD_KEYS)
        g.deleted = first([d for d in C.DELETED_COLUMNS if d not in C.INVERTED_DELETED_COLUMNS])
        g.version = first(C.VERSION_COLUMNS) if g.record_key else None
        g.received = first(C.RECEIVED_COLUMNS)
        g.file_column = first(C.FILE_COLUMNS)
        g.date = next((c for c, t in cols if t in C.DATE_TYPES and not set(tokens(c)) & C.NOT_VISIT_DATE_TOKENS
                       and not any(p in c.upper() for p in PRIVACY)), None)
        r, subset = role(name)
        uname = name.upper()
        if r in ("copy", "identity"):
            g.event_class = r
        elif "ADMISSION" in uname or "IPVISIT" in uname:
            g.event_class = "admission"
        else:
            g.event_class = EVENT_FROM_ROLE.get(r, "unknown")
        if subset and g.event_class == "encounter":
            a.notes.append("name suggests one service point")
        stated = {c for (m, c) in self.pol.members if m == f}
        exclude = {x.upper() for x in (g.deleted, g.version, g.received, g.file_column) if x} - stated
        if g.record_key and g.event_class != "registration":
            exclude.add(g.record_key.upper())
        a.candidates, a.truncated = nominate(cols, exclude, stated)
        if f in self.pol.sources:
            s = self.pol.sources[f]
            a.counted = s
            key = next((c for c in a.candidates if c.column.upper() == s["key"].upper()), None)
            if key is None:
                real = next((c for c, _ in cols if c.upper() == s["key"].upper()), None)
                key = Candidate(real or s["key"], self.type_of(f, real) if real else "TEXT", "registered", -1.0)
                a.candidates.insert(0, key)
                if real is None:
                    self.ask(a, "SCHEMA_DRIFT", f"configured key {s['key']} no longer exists")
            a.selected = key
        return a

    def early(self, a):
        """Decisions that need no measurement: counted, policy exclusions, holds, copies, kinds not counted."""
        pol, f, ec = self.pol, a.fqn, a.grain.event_class
        if self.cat[f].get("store"):
            self.terminal(a, "not_a_source", "raw store (read only for lineage fingerprints)")
        elif a.counted:
            a.stage, a.eligibility = "counted", "eligible"
            a.config = {k: a.counted.get(k) for k in ("key", "record", "deleted", "version", "received")}
        elif f in pol.ignored:
            self.terminal(a, "not_a_source", f"a person said: not a source ({pol.ignored[f].id})")
        elif a.schema in pol.excluded:
            self.terminal(a, "not_a_source", f"schema excluded ({pol.excluded[a.schema].id})")
        elif f in pol.rejected:
            self.terminal(a, "not_a_source", f"rejected in the registry ({pol.rejected[f].id})")
        elif copy_schema(a.schema):
            self.terminal(a, "not_a_source", "test or backup schema")
        elif ec in ("copy", "identity"):
            self.terminal(a, "not_a_source", f"{ec} object by name")
        elif a.schema in pol.held:
            self.terminal(a, "held", f"schema on hold ({pol.held[a.schema].id})")
        elif ec in pol.events and pol.events[ec].assertion["action"] == "exclude" and not pol.events[ec].premise:
            self.terminal(a, "not_a_source", f"{ec} tables are not counted ({pol.events[ec].id})")
        elif not a.candidates and ec != "encounter":
            self.terminal(a, "not_a_source", "no identifier candidate")
        if a.stage not in ("pending", "counted"):
            a.eligibility = "excluded"

    def measure_object(self, a):
        """Behaviour only: grain and candidate verdicts from the screen and exact profiles."""
        if not a.candidates:
            self.ask(a, "NO_CANDIDATE", "visit table with no identifier candidate: which column identifies patients? "
                                        "(a namespace statement naming it makes the next run measure it)")
            return
        s = self.screen(a)
        if s is None:
            return
        rows, g = s["rows"], a.grain
        if rows == 0:
            return self.terminal(a, "not_a_source", "empty")
        bad, weak = [], []
        if rows < T["small"]:
            weak.append(f"only {rows} rows")
        if g.record_key:
            if s["rk_rows"] < rows * T["complete"]:
                bad.append("record identifier missing on some rows")
            elif s["rk_distinct"] < s["rk_rows"]:
                if g.version:
                    a.notes.append("records have versions; the latest is kept")
                else:
                    bad.append("record identifiers repeat with no version column")
        else:
            weak.append("no record identifier: duplicates and versions cannot be measured")
            if s["distinct_rows"] is not None and s["distinct_rows"] < rows * T["dup_rows"]:
                bad.append(f"{rows - s['distinct_rows']} exact duplicate rows")
        if g.file_column:
            weak.append("rows come from documents or files")
        for c in a.candidates:
            m = s["cands"].get(c.column)
            if m is None:
                continue
            c.measures.update(m)
            x, i = [], []
            comp = m["nonnull"] / rows
            c.measures["completeness"] = comp
            if comp < T["complete"]:
                x.append(f"present on {comp:.0%} of rows")
            recs = m["nonnull"]
            if g.record_key and s["rk_distinct"]:
                recs = m["rk_pairs"]
                if m["rk_pairs"] > s["rk_distinct"] * T["stable"]:
                    x.append("several values per record")
            if m["eq_rk"] > T["eq_record"] * rows:
                x.append("equals the record identifier")
            rpv = recs / m["distinct"] if m["distinct"] else 0.0
            c.measures["records_per_value"] = rpv
            if m["distinct"] and rpv > T["many_rpv"]:
                x.append(f"{rpv:.0f} records per value")
            if m["distinct"] and rpv < T["single_rpv"]:
                i.append("about one record per value")
            c.reasons, c.verdict = x + i, "contradicts" if x else "screened"
            c.measures["_weak"] = i
        screened = [c for c in a.candidates if c.verdict == "screened"]
        for c in screened[T["profiled"]:]:
            c.verdict, c.reasons = "unprofiled", c.reasons + ["not profiled (lower-ranked candidate)"]
        for c in screened[:T["profiled"]]:
            p = self.get(a, "profile", (a.fqn,), ((c.column, c.data_type), g.record_key, g.deleted, g.version, g.date),
                         why="exact profile")
            if p is None:
                continue
            x, i = [], list(c.measures.pop("_weak", []))
            keys, recs = p["keys"], p["key_records"]
            c.measures.update({k: p[k] for k in ("keys", "singleton_keys", "top_key_records", "key_records")})
            if keys and p["singleton_keys"] >= T["single_share"] * keys and "about one record per value" not in i:
                i.append("most values appear once")
            if keys and p["raw_key_values"] > (1 + T["variant"]) * keys:
                x.append("format variants of the same value")
            if keys and p["top_key_records"] > C.DOMINANT_KEY_SHARE * recs and \
                    p["top_key_records"] > C.DOMINANT_KEY_VS_MEAN * recs / keys:
                x.append("one value dominates")
            if p.get("records_with_many_keys", 0) > C.RECORD_KEY_SHARED_SHARE * max(p.get("distinct_records", 0), 1):
                x.append("record identifiers shared by several values")
            if p["rows_deleted"] > C.DELETED_SHARE * p["rows"]:
                bad.append("most rows deleted")
            if self.today() and p["date_min"] and p["date_min"] > self.today():
                bad.append("only future dates")
            c.reasons = x + i
            c.verdict = "contradicts" if x else "insufficient" if i else "consistent"
        g.reasons = list(dict.fromkeys(bad + weak))
        g.verdict = "contradicts" if bad else "insufficient" if weak else "known"
        self.explain_missing(a)

    def explain_missing(self, a):
        """Non-blocking: is a partly-present candidate present only for some kinds of row (a conditional attribute)?"""
        partial = [c for c in a.candidates if 0.02 < c.measures.get("completeness", 1) < T["complete"]][:2]
        if not partial:
            return
        names = {c.column for c in a.candidates} | {x for x in (a.grain.record_key, a.grain.date) if x}
        texts = tuple(c for c, t in self.cat[a.fqn]["columns"]
                      if t == "TEXT" and c not in names and not any(p in c.upper() for p in PRIVACY + FREE_TEXT))[:12]
        if not texts:
            return
        cats = self.get(a, "categories", (a.fqn,), texts, blocking=False, why="find grouping columns")
        if cats is None:
            return
        groupers = [g for g in texts if 2 <= cats["distinct"][g] <= 20][:3]
        for c in partial:
            for gcol in groupers:
                v = self.get(a, "conditional", (a.fqn,), ((c.column, c.data_type), gcol), blocking=False,
                             why="explain missing values")
                # conditional = present in some kinds of row and absent in others, with little in between
                if v and v["rows"] and v["present_rows"] and v["absent_rows"] and \
                        v["present_rows"] + v["absent_rows"] >= T["conditional_pure"] * v["rows"]:
                    c.reasons.append(f"presence follows {gcol}: a conditional attribute, not a population identifier")
                    break

    # ------------------------------------------------------------------ meaning
    def meanings(self):
        """Meaning from namespace statements, then transferred along derivation links inside each namespace's scope.
        Every step is written to the candidate's provenance."""
        for a in self.A.values():
            for c in a.candidates:
                st = self.pol.members.get((a.fqn, c.column.upper()))
                if st:
                    c.meaning, c.namespace = st.assertion["meaning"], st.id
                    c.provenance = [f"{st.id}: {short(a.fqn)}.{c.column} is a {c.meaning} identifier "
                                    f"(scope {self.pol.scope(st.id)})"]
                    self.holders.append((a, c))
        frontier = list(self.holders)
        while frontier:
            new = []
            for X, xc in frontier:
                for O in self.A.values():
                    if O is X or not self.in_scope(O, xc.namespace) or O.stage not in ("pending", "counted"):
                        continue
                    if any(c.meaning is not None for c in O.candidates):
                        continue        # meaning already set: further link checks would only spend statements
                    for oc in [c for c in O.candidates if c.verdict != "unprofiled"][:T["profiled"]]:
                        if oc.meaning is not None:
                            continue
                        step = self.derivation(O, oc, X, xc)
                        if step:
                            oc.meaning, oc.namespace, oc.provenance = xc.meaning, xc.namespace, xc.provenance + [step]
                            new.append((O, oc))
                            break
            self.holders += new
            frontier = new

    def nothing_points_to_patients(self, a):
        """An unrecognised table is worth a person's time only if a candidate has a person-like or stated name, or
        has a meaning, or its values relate to a counted source. Visit tables never reach this check."""
        if any(c.meaning or c.hint in ("stated", "registered", "listed", "person") for c in a.candidates):
            return False
        related = {"values_overlap", "complements", "subset_of", "superset_of", "same_records", "shares_records",
                   "supersedes", "same_population", "derived_from"}
        return not any(r.kind in related for r in a.relations)

    def in_scope(self, O, ns):
        return O.facility == self.pol.scope(ns)

    def same_namespace(self, S, X):
        return S.meaning == "person" and S.namespace is not None and S.namespace == X.selected.namespace

    def today(self):
        return self.now.date().isoformat() if self.now else None

    def formats_ok(self, O, oc, X, xc):
        """None while unmeasured; else whether (almost) all of oc's values have formats seen among xc's values."""
        fm = self.get(O, "formats", (O.fqn, X.fqn), ((oc.column, oc.data_type), (xc.column, xc.data_type)),
                      why="namespace formats")
        if fm is None:
            return None
        if fm["covered"] < T["contained"] * fm["distinct"]:
            O.notes.append(f"{oc.column}: {fm['distinct'] - fm['covered']} of {fm['distinct']} values have formats "
                           f"not seen in {xc.namespace}; meaning not transferred")
            return False
        return f"{fm['covered']} of {fm['distinct']} values in the namespace's formats"

    def derivation(self, O, oc, X, xc):
        """A derivation link from member column xc of X to column oc of O, or None. Never by name or containment."""
        ct = T["contained"]
        refs = self.cat[O.fqn].get("refs") or []
        if O.object_type != "BASE TABLE" and X.fqn in refs:
            c = self.contain(O, (O.fqn, oc.column), (X.fqn, xc.column), "lineage values")
            if c and c["da"] and c["a_in_b"] >= ct and c["b_in_a"] >= ct:
                return (f"lineage: {short(O.fqn)} reads {short(X.fqn)} directly and {oc.column} holds the same "
                        f"{c['da']} values as {xc.column}")
        rk_o, rk_x = O.grain.record_key, self.record_key(X)
        if not (rk_o and rk_x):
            return None
        al = self.align(O, (O.fqn, rk_o, oc.column), (X.fqn, rk_x, xc.column), "derivation by records")
        if al is None:
            return None
        keyed = al["agree"] + al["disagree"]
        if not (al["agree"] and al["agree"] >= ct * min(al["ra"], al["rb"]) and al["agree"] >= ct * keyed):
            return None
        formats = self.formats_ok(O, oc, X, xc)
        if not formats:
            return None
        return (f"aligned with {short(X.fqn)}: {al['agree']} of {keyed} shared records carry the same {xc.column} "
                f"value; {formats}")

    def select(self, a):
        """The identifier to count: a column whose meaning is person; otherwise ask, listing every candidate."""
        cands = a.candidates
        if not cands or any(c.verdict == "screened" for c in cands):
            return
        person = sorted((c for c in cands if c.meaning == "person"), key=lambda c: (len(c.provenance), c.rank))
        if person:
            a.selected = person[0]
            a.equivalents = [c for c in person[1:] if c.namespace == a.selected.namespace]
            return
        plaus = sorted((c for c in cands if c.verdict in ("consistent", "insufficient") and c.meaning is None),
                       key=lambda c: (c.verdict != "consistent", c.rank))
        a.selected = plaus[0] if plaus else None
        rel = {}
        top = a.selected
        for x in plaus[1:1 + T["pair_others"]]:
            first, second = sorted([(top.column, top.data_type), (x.column, x.data_type)])
            p = self.get(a, "pair", (a.fqn,), (first, second), blocking=False, why="relation between candidates")
            if p is None:
                continue
            r = pair_relation(p, top.column == first[0])
            if r == "equivalent" and top.verdict == "consistent":
                a.equivalents.append(x)
            rel[x.column] = describe(r, top, x)
        listing = []
        for c in cands:
            if c.meaning is not None:
                listing.append(f"{c.column}: a {c.meaning} identifier ({c.namespace})")
                continue
            detail = ", ".join(c.reasons) or c.verdict
            listing.append(f"{c.column} [{c.hint}; {c.verdict}" + (f": {detail}" if c.reasons else "") +
                           (f"; {rel[c.column]}" if c.column in rel else "") + "]")
        known = self.namespace_relations(a, top) if top else []
        if not plaus:
            self.ask(a, "NO_USABLE_IDENTIFIER", "no candidate behaves like an identifier: " + "; ".join(listing))
        else:
            self.ask(a, "MEANING_UNRESOLVED", "which column, if any, identifies patients? " + "; ".join(listing) +
                     ("; values: " + "; ".join(known) if known else ""))

    def namespace_relations(self, a, top):
        """Information for the person: value containment against established namespaces of this facility."""
        out = []
        for X, xc in self.holders:
            if X.facility != a.facility or X is a:
                continue
            ok = self.compatible(a, (a.fqn, top.column), (X.fqn, xc.column), blocking=False)
            if not ok:
                continue
            c = self.contain(a, (a.fqn, top.column), (X.fqn, xc.column), "relation to a namespace", blocking=False)
            if c and c["inter"]:
                out.append(f"{c['a_in_b']:.0%} of {top.column} found in {xc.namespace} ({short(X.fqn)}.{xc.column}) - "
                           f"a value relationship, not a meaning")
        return out

    # ------------------------------------------------------------------ relations
    def relate(self, a):
        pol, o = self.pol, self.cat[a.fqn]
        if a.object_type != "BASE TABLE":
            refs = o.get("refs")
            if refs is None:
                self.ask(a, "LINEAGE_UNREADABLE", "lineage could not be read")
            else:
                keyed = [r for r in refs if r in self.A and self.A[r].candidates]
                counted = [r for r in keyed if r in pol.sources]
                if counted and len(counted) == len(keyed):
                    a.relations += [Relation("derived_from", r, "lineage") for r in counted]
                    return self.terminal(a, "not_a_source", f"view of counted {', '.join(counted)}")
                if keyed:
                    a.relations += [Relation("derived_from", r, "lineage") for r in keyed]
                    self.ask(a, "DERIVED_VIEW", f"reads {', '.join(sorted(keyed))}: count this view, or what it reads?")
        if a.schema in pol.derived:
            self.ask(a, "DERIVED_LAYER", f"in a derived layer ({pol.derived[a.schema].id}): count it?")
        S = a.selected
        if S is None:
            return
        ct, me, ec = T["contained"], (a.fqn, S.column), a.grain.event_class
        counted_here = [X for X in self.A.values() if X.counted and X is not a and X.facility == a.facility
                        and X.selected is not None]
        # 1. record-level relations (derivation): same records, subset, a successor, or shared records
        rk, waiting = a.grain.record_key, False
        for X in counted_here:
            rk_x = self.record_key(X)
            if not (rk and rk_x):
                continue
            al = self.align(a, (a.fqn, rk, S.column), (X.fqn, rk_x, X.selected.column), "records shared with counted")
            if al is None:
                waiting = True
                continue
            agree, keyed = al["agree"], al["agree"] + al["disagree"]
            if not agree or agree < ct * keyed:
                continue                   # record IDs coincide but carry other identifiers: no record relation
            m = dict(agree=agree, disagree=al["disagree"], records=al["ra"], counted_records=al["rb"])
            mine, theirs = agree / max(al["ra"], 1), agree / max(al["rb"], 1)
            if mine >= ct and theirs >= ct:
                a.relations.append(Relation("same_records", X.fqn, "record-aligned", m))
                if al["disagree"]:
                    self.ask(a, "REKEYED", f"same records as counted {X.fqn}, but {al['disagree']} carry another "
                                           f"identifier value (merges or corrections?)")
                    return
                return self.terminal(a, "not_a_source", f"same records as counted {X.fqn} (record-aligned)")
            if mine >= ct:
                a.relations.append(Relation("subset_of", X.fqn, "record-aligned", m))
                return self.terminal(a, "not_a_source", f"its records are already counted in {X.fqn} (record-aligned)")
            if theirs >= ct:
                v = self.version(a, X, al)
                if v is None:
                    return
                if not v:
                    a.relations.append(Relation("superset_of", X.fqn, "record-aligned", m))
                    self.ask(a, "SUPERSET_LIVE", f"holds every record of counted {X.fqn} plus {al['ra'] - agree} more, "
                                                 f"and {short(X.fqn)} is still live: a new version or a wider extract?")
                return
            a.relations.append(Relation("shares_records", X.fqn, "record-aligned", m))
            self.ask(a, "PARTIAL_RECORDS", f"shares {agree} records with counted {X.fqn} ({theirs:.0%} of its records; "
                                           f"{mine:.0%} of these): dropped records, or a partial copy?")
            return
        if waiting:
            return
        # 2. value relations: an overlap of values is only a relationship unless both columns share a namespace
        others = counted_here + ([X for X in self.A.values() if X is not a and X.facility == a.facility
                                  and X.grain.event_class == "encounter" and X.selected is not None
                                  and not X.counted and X.stage == "pending"] if ec == "admission" else [])
        linked = []
        for X in others:
            ok = self.compatible(a, me, (X.fqn, X.selected.column))
            if ok is None:
                waiting = True
                continue
            if not ok:
                continue
            c = self.contain(a, me, (X.fqn, X.selected.column), f"compare with {X.fqn}")
            if c is None:
                waiting = True
            elif c["inter"] >= T["overlap"] * max(min(c["da"], c["db"]), 1):
                linked.append((X, c))
        if waiting:
            return
        same_ns = lambda X: self.same_namespace(S, X)
        if ec == "admission":
            ref = [(X, c) for X, c in linked if same_ns(X)]
            if ref:
                X, c = max(ref, key=lambda t: t[1]["db"])
                new = c["da"] - c["inter"]
                a.contribution = dict(status="measured", base=X.fqn, new=new, namespace=S.namespace,
                                      share=new / c["db"] if c["db"] else None)
            else:
                a.contribution = dict(status="unmeasurable", why="no visit population in the same established namespace")
            return
        for X, c in [(X, c) for X, c in linked if X.counted]:
            m = dict(this=c["da"], counted=c["db"], shared=c["inter"])
            if same_ns(X):
                if c["a_in_b"] >= ct:
                    a.relations.append(Relation("subset_of", X.fqn, f"namespace {S.namespace}", m))
                    return self.terminal(a, "not_a_source", f"adds no subjects to counted {X.fqn} ({S.namespace})")
                new = c["da"] - c["inter"]
                a.relations.append(Relation("complements", X.fqn, f"namespace {S.namespace}", m))
                a.contribution = dict(status="measured", base=X.fqn, new=new, overlap=c["inter"],
                                      namespace=S.namespace)
                self.ask(a, "DOUBLE_COUNT", f"same namespace {S.namespace} as counted {X.fqn}: counting both per source "
                                            f"counts {c['inter']} subjects twice")
            else:
                a.relations.append(Relation("values_overlap", X.fqn, "value relationship only", m))
                self.ask(a, "VALUE_RELATION", f"{c['a_in_b']:.0%} of {S.column} values are in counted {X.fqn} and "
                                              f"{c['b_in_a']:.0%} of its values are here: the same people, or a "
                                              f"coincidental numbering? (meaning not inferred)")
            return
        if counted_here:
            a.relations += [Relation("unlinked", X.fqn, "no value overlap") for X in counted_here]
            a.contribution = dict(status="unmeasurable", why="no value overlap with the facility's counted sources")
            if a.facility in pol.separate:
                a.notes.append(f"counted separately by policy ({pol.separate[a.facility].id})")
            else:
                self.ask(a, "UNLINKED", "a population with no value overlap with this facility's counted sources: "
                                        "count it separately?", "facility", a.facility)
        else:
            a.contribution = dict(status="new population", base=None)

    def version(self, a, X, al):
        """True if `a` supersedes counted X: X stopped and `a` extends its dates (records already shown to continue).
        None while measurements are missing."""
        xs, own = self.screen(a, of=X), self.obs.get(("screen", (a.fqn,), self.screen_args(a)))
        if xs is None or own is None:
            return None
        if not (self.stopped(X.fqn) and own["date_max"] and xs["date_max"] and own["date_max"] > xs["date_max"]):
            return False
        S = a.selected
        a.relations.append(Relation("supersedes", X.fqn, "stopped predecessor, extended dates, records continue",
                                    dict(agree=al["agree"], disagree=al["disagree"])))
        a.stage = "replacement"
        cols = {c.upper() for c, _ in self.cat[a.fqn]["columns"]}
        inherited = {k: X.counted.get(k) for k in ("key", "record", "deleted", "version", "received")}
        if all(v is None or v.upper() in cols for v in inherited.values()):
            a.config = dict(inherited, inherited_from=X.fqn)
        else:
            a.config = dict(key=S.column, record=a.grain.record_key, deleted=a.grain.deleted,
                            version=a.grain.version, received=a.grain.received)
        if al["disagree"]:
            self.ask(a, "REKEYED", f"{al['disagree']} records carried over from {X.fqn} have another identifier "
                                   f"value (merges or corrections?)")
        if not a.grain.date:
            self.ask(a, "HISTORY_CHANGE", f"replaces {X.fqn}; no visit date to reconcile the overlap")
            return True
        w = self.get(a, "window", (a.fqn,), ((S.column, S.data_type), a.grain.date, xs["date_max"]),
                     why="reconcile with predecessor")
        if w is None:
            return True
        prev = self.distinct_of(a, X)
        if prev is None:
            return True
        a.contribution = dict(status="replacement", base=X.fqn, overlap_window_subjects=w["distinct"],
                              predecessor_subjects=prev)
        if abs(w["distinct"] - prev) <= T["reconcile"] * max(prev, 1):
            a.reasons.append(f"replaces {X.fqn}; reconciled over the overlap window ({w['distinct']} vs {prev})")
        else:
            self.ask(a, "HISTORY_CHANGE", f"replaces {X.fqn} but changes its history ({w['distinct']} vs {prev} subjects "
                                          f"up to {xs['date_max']}): sign-off needed")
        return True

    def distinct_of(self, a, X):
        """Distinct identifier values of counted X (from its screen)."""
        xs = self.screen(a, of=X)
        if xs is None:
            return None
        return xs["cands"].get(X.selected.column, {}).get("distinct")

    def representations(self, live):
        """Uncounted objects of one facility holding the same records (record-aligned): one population, several
        representations. Exactly one clean representation is canonical; otherwise a person picks."""
        cands = [a for a in live if a.selected is not None and a.grain.record_key and not a.counted
                 and a.grain.event_class in ("encounter", "unknown") and a.stage == "pending"]
        parent = {a.fqn: a.fqn for a in cands}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        ct = T["contained"]
        for i, a in enumerate(cands):
            for b in cands[i + 1:]:
                if a.facility != b.facility:
                    continue
                before = len(a.gaps)
                al = self.align(a, (a.fqn, a.grain.record_key, a.selected.column),
                                (b.fqn, b.grain.record_key, b.selected.column), "same records?")
                b.gaps += a.gaps[before:]          # the decision for both waits on this measurement
                if al and al["agree"] >= ct * al["ra"] and al["agree"] >= ct * al["rb"]:
                    parent[find(a.fqn)] = find(b.fqn)
        groups = defaultdict(list)
        for a in cands:
            groups[find(a.fqn)].append(a)
        for group in groups.values():
            if len(group) < 2:
                continue
            clean = [a for a in group if a.grain.verdict == "known" and a.selected.verdict == "consistent"]
            names = ", ".join(sorted(a.fqn for a in group))
            if len(clean) == 1:
                for a in group:
                    if a is not clean[0]:
                        a.relations.append(Relation("same_records", clean[0].fqn, "record-aligned"))
                        self.terminal(a, "not_a_source", f"another representation of {clean[0].fqn} (record-aligned)")
                clean[0].notes.append(f"canonical among {names}")
            else:
                for a in group:
                    a.relations += [Relation("same_records", b.fqn, "record-aligned") for b in group if b is not a]
                    self.ask(a, "REPRESENTATIONS", f"one population, several representations: {names}",
                             "facility", a.facility)

    # ------------------------------------------------------------------ decision
    def decide(self, a):
        if any(g.blocking for g in a.gaps):
            a.stage = "pending"
            return
        pol, ec, S = self.pol, a.grain.event_class, a.selected
        stmt = pol.events.get(ec)
        if ec == "admission" and stmt:
            share, premise = a.contribution.get("share"), stmt.premise or {}
            if a.contribution.get("status") != "measured" or share is None:
                return self.terminal(a, "not_a_source", f"admissions not counted ({stmt.id}); premise not checkable: "
                                                       f"{a.contribution.get('why', 'not measured')}")
            if share <= premise.get("max", 1.0):
                return self.terminal(a, "not_a_source", f"admissions not counted ({stmt.id}); premise holds: adds "
                                                       f"{share:.1%} ({a.contribution['new']} subjects)")
            a.questions = [Question("POLICY_PREMISE", "policy", stmt.id,
                                    f"premise no longer holds at {a.facility}: admissions add {share:.1%} "
                                    f"({a.contribution['new']} subjects) in {a.contribution['namespace']}")]
            a.stage = "question"
            return
        if stmt and stmt.assertion["action"] == "exclude":
            return self.terminal(a, "not_a_source", f"{ec} tables are not counted ({stmt.id})")
        if ec == "unknown" and self.nothing_points_to_patients(a):
            return self.terminal(a, "not_a_source", "unrecognised table: its identifier-like columns have generic "
                                                    "names and their values relate to no counted or confirmed "
                                                    "patient identifier")
        if ec == "unknown" or (stmt and stmt.assertion["action"] == "ask"):
            self.ask(a, "KIND_UNKNOWN", "what does one row represent? (kind of table not recognised)")
        if a.schema in pol.covered:
            src, st = pol.covered[a.schema]
            if src in pol.sources:
                return self.terminal(a, "not_a_source", f"covered by counted {src} ({st.id})")
            self.ask(a, "COVER_PREMISE", f"says {a.schema} is covered by {src}, which is not counted", "policy", st.id)
        if a.facility not in pol.onboarded:
            self.ask(a, "SCOPE_UNKNOWN", "is this facility in scope?", "facility", a.facility)
        if a.grain.verdict in ("insufficient", "contradicts"):
            self.ask(a, "GRAIN", "row grain: " + "; ".join(a.grain.reasons))
        if S is not None and S.meaning == "person" and S.verdict == "contradicts":
            self.ask(a, "BEHAVIOUR", f"{S.column} is a confirmed person identifier ({S.namespace}) but here: "
                                     + ", ".join(S.reasons))
        if a.truncated:
            self.ask(a, "TRUNCATED", f"more than {T['max_candidates']} candidate columns; not measured: "
                                     f"{', '.join(a.truncated)}")
        if not a.questions and (S is None or S.meaning != "person"):
            self.ask(a, "MEANING_UNRESOLVED", "no identifier with an established person meaning")
        if a.questions:
            a.stage = "question"
            return
        if a.stage != "replacement":
            a.stage = "auto"
            a.config = dict(key=S.column, record=a.grain.record_key, deleted=a.grain.deleted,
                            version=a.grain.version, received=a.grain.received)
        a.eligibility = "eligible"


def pair_relation(p, top_is_a):
    """equivalent | top_groups_x | x_groups_top | crossing - a measured relation; no meaning is inferred."""
    a_many = p["a_with_many_b"] > 0.01 * max(p["da"], 1)
    b_many = p["b_with_many_a"] > 0.01 * max(p["db"], 1)
    if not a_many and not b_many:
        return "equivalent"
    if a_many and b_many:
        return "crossing"
    if top_is_a:
        return "top_groups_x" if a_many else "x_groups_top"
    return "x_groups_top" if a_many else "top_groups_x"


def describe(rel, top, x):
    return {"equivalent": f"one-to-one with {top.column}",
            "top_groups_x": f"each {top.column} covers several {x.column} values",
            "x_groups_top": f"each {x.column} covers several {top.column} values",
            "crossing": f"crosses {top.column} in both directions"}.get(rel, rel) + " (meaning not inferred)"
