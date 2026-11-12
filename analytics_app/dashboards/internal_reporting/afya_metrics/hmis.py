"""Patient counts posted by the HMIS: HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS (one row per facility per run).

The HMIS posts PATIENT_COUNT, the patients registered at each facility, every hour or so, from one of its
environments (migrate now, production soon). This module decides which run feeds the dashboard and how its facilities
join the warehouse patient counts. It is pure logic; the Snowflake reads are data.read_hmis_environments/read_hmis.

- Environment: every HMIS environment with runs is loaded; the dashboard's Environment filter (session key ENV_KEY)
  picks which of Warehouse / Production / Migrate feed the table and the totals (combine). Default: Warehouse and
  Production (else Migrate). Selecting two HMIS environments together is flagged: they can hold the same facilities.
- Rows: the latest whole run; facility rows with a facility ID only (organization rows are their totals, and
  unassigned rows have no facility); the agreed counting method. Facilities with no patients yet are posted with 0
  and listed, so every facility on the HMIS appears.
- Facilities: test organisations (IGNORE_ORGS) are dropped entirely. Facilities the warehouse already counts
  (WAREHOUSE, or an exact name match with a warehouse source) are not added again: their HMIS figure is shown on the
  warehouse row instead, so nobody is counted twice. Every other facility is a row of its own.
- HMIS facilities add PATIENT_COUNT to UPDs (patients) and VISIT_COUNT to Patient Records (visits), once posted.
- HMIS_WINS: warehouse sources the HMIS replaces (the HMIS row keeps the HMIS facility name; the warehouse row
  goes). DUPLICATES: HMIS facilities that are a second copy of another one, never counted.
- Dated arrivals (for the period filters): each day's change in a facility's counts, from the run history, with
  the whole count on the day it first appears. They add up to the latest counts.
"""

import os
import re
from datetime import datetime, timedelta, timezone

PRODUCTION = "prod"
MIGRATE = "migrate"
CHOICES = {PRODUCTION: "Production", MIGRATE: "Migrate"}
ALIASES = {PRODUCTION: {"prod", "production"}, MIGRATE: {"migrate"}}     # ENVIRONMENT values per environment
WAREHOUSE_ENV = "Warehouse"                     # the Environment value of the warehouse sources
ENV_KEY = "pc_envs"                             # session key of the dashboard's Environment filter
METHODS = {m.strip() for m in os.environ.get("AFYA_HMIS_METHODS", "gateway_count").split(",") if m.strip()}
STALE_HOURS = 26          # runs are hourly; a day without one is worth a look
JUMP = 0.20               # flag a facility whose count moved more than this since the previous day
# CAPTURED_AT is stored in UTC; the dashboard shows HMIS times and dates in East Africa Time (data.py converts).
TZ = "Africa/Nairobi"
EAT = timezone(timedelta(hours=3), "EAT")

# Test organisations: never shown or counted (matched on the organisation name, any environment).
IGNORE_ORGS = {"STAGING", "LIVE DB MIGRATE", "DEMO FACILITY 6", "DRY RUN HOSPITAL"}
# (organisation, facility) names -> the warehouse source that already counts it (migrate posts names).
WAREHOUSE = {
    ("KISUMU SPECIALIST", "KSH"): "Kisumu Specialist Hospital (KSH)",
    ("WEMA MEDICAL CENTER", "WEMA MEDICAL CENTER"): "Wema Medical",
    ("MEDHANIT", "MEDHANIT"): "Medhanit",
    ("PORTBELL", "PORTBELL"): "Portbell",
    ("SILVERWOOD MEDICAL HOSPITAL", "SILVERWOOD HOSPITAL"): "Silverwood",
}
# Production names differ from the warehouse's, so its matches go by ID:
# environment -> {(organisation id, facility id): warehouse source}.
WAREHOUSE_IDS = {PRODUCTION: {
    (284, 283): "Kisumu Specialist Hospital (KSH)",
    (258, 256): "Wema Medical",
    (71, 69): "Medhanit",
    (263, 261): "Portbell",
    (82, 80): "Silverwood",
}}
# Warehouse sources the HMIS replaces: when the chosen environment has the facility, its HMIS figures (under the
# HMIS facility name) are used and the warehouse source is dropped from the table, the totals and the dated arrivals.
HMIS_WINS = {"Kisumu Specialist Hospital (KSH)", "Medhanit", "Portbell", "Silverwood", "Wema Medical"}
# environment -> HMIS facilities that duplicate another one and are never counted. KSH sits under two organisations
# with the same May 2024 load and near-identical months since; organisation 284 is the live one (new patients in
# November 2026), 250 stopped in October.
DUPLICATES = {PRODUCTION: {(250, 247)}}
# environment -> {(organisation id, facility id): name}, for facilities posted without one. The Eagle branches
# carry the same IDs' patient counts as the named migrate rows (65,849 / 46,970 / 18,817).
NAMES_IDS = {PRODUCTION: {
    (199, 329): "Kangemi · Eagle Nursing Home",
    (199, 328): "Kawangware · Eagle Nursing Home",
    (199, 330): "Kibra · Eagle Nursing Home",
}}

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def now_eat():
    """The current time in East Africa Time, without a time zone, like the converted CAPTURED_AT values."""
    return datetime.now(EAT).replace(tzinfo=None)


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip()).upper()


def choice_of(environment):
    """The environment ('prod' / 'migrate') an ENVIRONMENT value belongs to, or None."""
    e = str(environment or "").strip().lower()
    return next((c for c, names in ALIASES.items() if e in names), None)


def resolve(choice, environments):
    """The ENVIRONMENT value as stored for 'prod' / 'migrate', or None if that environment has no rows."""
    for e in environments:
        if e is not None and choice_of(e) == choice and _SAFE.match(str(e)):
            return e
    return None


def _nice(s):
    s = re.sub(r"\s+", " ", str(s or "").strip())
    return s.title() if s.isupper() and len(s) > 4 else s


def label(org, facility, scope):
    o, f = _nice(org), _nice(facility)
    if scope == "unassigned" or not f or f == "(unassigned)":
        return f"{o} (unassigned)"
    return f if not o or norm(f) == norm(o) else f"{f} · {o}"


def build(rows, environment, warehouse_sources, now=None):
    """rows: the latest run and the previous day's run (WHICH = 'latest' | 'previous'). Returns a view dict:
    facilities (the rows added to the table), on_warehouse ({warehouse source: HMIS patients} for facilities the
    warehouse already counts) and the run's details. None when the environment has no run."""
    now = now or now_eat()
    latest = [r for r in rows if r["WHICH"] == "latest"]
    if not latest:
        return None
    key = lambda r: (r["ORGANIZATION_ID"], r["FACILITY_ID"] if r["SCOPE"] == "facility" else "unassigned")
    prev = {key(r): r["PATIENT_COUNT"] for r in rows if r["WHICH"] == "previous"
            and r["COUNT_METHOD"] in METHODS}
    prev_at = max((r["CAPTURED_AT"] for r in rows if r["WHICH"] == "previous"), default=None)
    captured = max(r["CAPTURED_AT"] for r in latest)
    sources = {norm(s): s for s in warehouse_sources}
    choice = choice_of(environment)
    by_id, names_by_id = WAREHOUSE_IDS.get(choice, {}), NAMES_IDS.get(choice, {})

    facilities, on_warehouse, dropped, seen, replaced = [], {}, 0, set(), set()
    for r in latest:
        n = int(r["PATIENT_COUNT"] or 0)
        if r["COUNT_METHOD"] not in METHODS:
            dropped += 1
            continue
        if r["FACILITY_ID"] is None:                       # no facility (the unassigned rows): not shown or counted
            continue
        org, fac = norm(r["ORGANIZATION_NAME"]), norm(r["FACILITY_NAME"])
        if org in IGNORE_ORGS:                             # test organisations never reach the dashboard
            continue
        ids = (r["ORGANIZATION_ID"], r["FACILITY_ID"])
        if ids in DUPLICATES.get(choice, set()):           # a second copy of another facility: never counted
            continue
        if ids in names_by_id:
            name = names_by_id[ids]
        elif fac:
            name = label(r["ORGANIZATION_NAME"], r["FACILITY_NAME"], r["SCOPE"])
        else:                                              # no name posted: identify it by its HMIS IDs
            name = f"HMIS facility {r['FACILITY_ID']} (organisation {r['ORGANIZATION_ID']})"
        source = by_id.get(ids) or WAREHOUSE.get((org, fac)) or sources.get(fac) or sources.get(norm(name))
        if source in HMIS_WINS:                            # the HMIS replaces this warehouse source, under its own name
            name = _nice(r["FACILITY_NAME"]) or name
            replaced.add(source)
        elif source:                                       # already counted from the warehouse: not added again
            on_warehouse[source] = on_warehouse.get(source, 0) + n
            continue
        if name in seen:                                   # two facilities with the same names: keep both apart
            name = f"{name} (#{r['FACILITY_ID']})"
        seen.add(name)
        before = prev.get(key(r))
        change = (n - before) / before if before else None
        flags = [f"{change:+.0%} since {prev_at:%d %b}"] if change is not None and abs(change) > JUMP else []
        visits = r.get("VISIT_COUNT")
        facilities.append(dict(name=name, patients=n, visits=None if visits is None else int(visits), flags=flags,
                               ids=ids))
    return dict(environment=environment, captured=captured, previous_at=prev_at,
                stale=captured < now - timedelta(hours=STALE_HOURS), facilities=facilities,
                on_warehouse=on_warehouse, replaced=replaced, dropped=dropped)


def receipts(view, daily):
    """Dated arrivals for the HMIS facilities in the table, in the shape of the warehouse receipts: the patients
    each facility gained on each day it was posted (the whole count on its first day). Facilities left to the
    warehouse, test organisations and rows without a facility never appear, as in the table."""
    if not view:
        return []
    names = {f["ids"]: f["name"] for f in view["facilities"]}
    return [{"SOURCE": names[(d["ORGANIZATION_ID"], d["FACILITY_ID"])], "DATE": d["DATE"],
             "RECORDS": d.get("VISITS_ADDED", 0), "UPDS": d["ADDED"]} for d in daily
            if (d["ORGANIZATION_ID"], d["FACILITY_ID"]) in names and (d["ADDED"] or d.get("VISITS_ADDED"))]


def counted_rows(view):
    """The HMIS facilities as patient-count rows: patients into UPDs, no Patient Records."""
    if not view:
        return []
    day = view["captured"].date() if hasattr(view["captured"], "date") else view["captured"]
    env = CHOICES.get(choice_of(view["environment"]), str(view["environment"]))
    return [{"SOURCE": f["name"], "PATIENT_RECORDS": f.get("visits") or 0, "VISITS_POSTED": f.get("visits") is not None,
             "UPDS": f["patients"], "COUNTED_AT": day, "HMIS": True, "ENV": env, "FLAGS": f["flags"]}
            for f in view["facilities"]]


def env_options(parts):
    """The Environment filter's options: the warehouse, then each HMIS environment that has a run."""
    return [WAREHOUSE_ENV] + [label for label in CHOICES.values() if label in parts]


def default_envs(parts):
    """Warehouse plus production when it has runs, else plus the first HMIS environment that does."""
    hmis = [label for label in CHOICES.values() if label in parts]
    return [WAREHOUSE_ENV] + hmis[:1]


def combine(warehouse_rows, warehouse_receipts, parts, selected):
    """The sources table and dated arrivals for the selected environments. parts: {environment label: (view,
    receipts)}. Warehouse sources the HMIS replaces (HMIS_WINS) drop out whenever an HMIS environment has them, even
    with only the warehouse selected: the HMIS is their only source. Returns (rows, receipts, overlap): overlap lists HMIS environments selected together, whose facilities
    can be the same ones (Production and Migrate)."""
    hmis = [label for label in CHOICES.values() if label in parts and label in selected]
    replaced = set().union(*(view["replaced"] for view, _ in parts.values() if view))
    rows, rec = [], []
    if WAREHOUSE_ENV in selected:
        rows += [{**r, "ENV": WAREHOUSE_ENV} for r in warehouse_rows if r["SOURCE"] not in replaced]
        rec += [x for x in warehouse_receipts if x["SOURCE"] not in replaced]
    for i, label in enumerate(hmis):
        view, receipts_ = parts[label]
        tag = f" ({label})" if i else ""               # two HMIS environments: keep same-named facilities apart
        rows += [{**r, "SOURCE": r["SOURCE"] + tag} for r in counted_rows(view)]
        rec += [{**x, "SOURCE": x["SOURCE"] + tag} for x in receipts_]
    return rows, rec, hmis if len(hmis) > 1 else []
