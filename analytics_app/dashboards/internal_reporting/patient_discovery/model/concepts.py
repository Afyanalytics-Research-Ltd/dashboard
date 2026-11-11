"""Concepts of the warehouse model. Each one exists because a counting decision is wrong without it.

Object       What physically exists: a table or view, its columns, and what it reads (lineage).

Grain        What one row is: record identifier (if any), versions, a coarse event class from the name.
             verdict: known | insufficient | contradicts.

Candidate    A column that might identify subjects. Names only nominate and rank.

Behaviour    What the data says about a candidate - statistics only, never meaning:
               consistent    complete, one value per record, values repeat across records
               insufficient  repetition cannot be shown (about one record per value, too few rows)
               contradicts   incomplete, several values per record, equal to the record ID, one value dominating,
                             far too many records per value, format variants
             A household number, an insurance member number and a patient number can all be 'consistent'.

Namespace    A set of identifier values with a stated meaning (person, household, insurance member, file ...) and an
             explicit scope (a facility). Created only by a statement: a person (or a counted source whose key a
             person confirmed) names its member columns. Nothing else creates one.

Meaning      What a candidate's values denote. Only two routes, both recorded as provenance:
               stated       the column is a member of a namespace statement
               transferred  the column derives from a member of a namespace, inside its scope:
                              aligned  both objects have record identifiers, and records shared with the member
                                       carry the same identifier value; the values keep the namespace's formats
                              lineage  the object reads the member directly and its column holds the member's
                                       values (both directions)
             Never meaning: names, column layouts, a table called PATIENTS, value containment (it only relates
             values), chains of containment, a policy, a counted source whose key meaning was never confirmed.

Eligibility  Whether our objective counts the dataset: scope (onboarding), event classes, decisions D2/D3/D6/D7,
             per-facility policies. Separate from meaning; a premise is re-checked on every run.

Population   The subjects a dataset observes: distinct values in one namespace, one facility, a date window.
Contribution What a population adds to a counted one. Measured only when both columns are in the same established
             namespace; otherwise overlap is a value relationship, reported, never interpreted.

Stage        counted       a registered source (a statement)
             auto          eligible, grain known, behaviour not contradicting, meaning = person (with provenance),
                           no unresolved relation
             replacement   supersedes a counted source whose meaning is established; reconciled
             question      a person must answer something specific; `questions` carry a code
             not_a_source  derived, duplicate or subset (by derivation), excluded by policy, empty
             held          a hold statement (D3)
             pending       measurements still needed
"""
from dataclasses import dataclass, field

STAGES = ("counted", "auto", "replacement", "question", "not_a_source", "held", "pending")

# question codes (what a person is asked; tests assert these, not wording)
CODES = ("MEANING_UNRESOLVED", "NO_CANDIDATE", "NO_USABLE_IDENTIFIER", "BEHAVIOUR", "GRAIN", "SCOPE_UNKNOWN",
         "KIND_UNKNOWN", "DERIVED_VIEW", "DERIVED_LAYER", "LINEAGE_UNREADABLE", "VALUE_RELATION", "DOUBLE_COUNT",
         "UNLINKED", "PARTIAL_RECORDS", "SUPERSET_LIVE", "REKEYED", "HISTORY_CHANGE", "REPRESENTATIONS", "TRUNCATED",
         "POLICY_PREMISE", "COVER_PREMISE", "SCHEMA_DRIFT")


@dataclass(frozen=True)
class Observation:
    """One aggregate measurement. `fp` is the input fingerprint of every object involved; the observation is
    current only while it matches. Values are counts, shares, dates or value shapes - never identifier values."""
    kind: str
    subject: tuple
    args: tuple
    fp: str
    values: dict
    at: str = None              # when it was measured (ISO); views expire after VIEW_PROFILE_MAX_AGE_DAYS


@dataclass(frozen=True)
class Statement:
    """A human decision or policy. `premise` (optional) is re-checked against measurements on every run."""
    id: str
    kind: str
    subject: str
    assertion: dict
    premise: dict = None
    origin: str = "human"


@dataclass(frozen=True)
class Question:
    code: str
    level: str          # object | facility | namespace | policy
    subject: str
    text: str


@dataclass(frozen=True)
class Gap:
    """A measurement the interpreter needs. `blocking` gaps keep a dataset pending; others only add explanation."""
    kind: str
    subject: tuple
    args: tuple = ()
    cost: float = 0.0
    blocking: bool = True
    why: str = ""
    priority: int = 0           # 0 visits and the evidence they need, 1 admissions, 2 unknown kinds of table


@dataclass
class Candidate:
    column: str
    data_type: str
    hint: str                   # stated | listed | person | generic
    rank: float
    verdict: str = "unmeasured"  # behaviour: consistent | insufficient | contradicts | screened | unprofiled
    reasons: list = field(default_factory=list)
    measures: dict = field(default_factory=dict)
    meaning: str = None          # None = unresolved; else the namespace's meaning (person, household, file ...)
    namespace: str = None        # statement id of the namespace
    provenance: list = field(default_factory=list)   # how the meaning reached this column, step by step


@dataclass
class Grain:
    record_key: object = None
    deleted: object = None
    version: object = None
    received: object = None
    date: object = None
    file_column: object = None
    event_class: str = "unknown"
    verdict: str = "unmeasured"
    reasons: list = field(default_factory=list)


@dataclass
class Relation:
    kind: str
    other: str
    detail: str = ""
    measures: dict = field(default_factory=dict)


@dataclass
class Assessment:
    fqn: str
    schema: str
    name: str
    object_type: str
    facility: str
    grain: Grain = field(default_factory=Grain)
    candidates: list = field(default_factory=list)
    truncated: list = field(default_factory=list)
    selected: object = None          # Candidate
    equivalents: list = field(default_factory=list)
    relations: list = field(default_factory=list)
    contribution: dict = field(default_factory=dict)
    eligibility: str = "undetermined"
    stage: str = "pending"
    reasons: list = field(default_factory=list)     # why the stage (not_a_source / held / replacement)
    questions: list = field(default_factory=list)   # Question
    notes: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    gaps: list = field(default_factory=list)
    counted: dict = None             # the source statement's assertion, if counted

    @property
    def meaning(self):
        s = self.selected
        return None if s is None else s.meaning

    def codes(self):
        return {q.code for q in self.questions}
