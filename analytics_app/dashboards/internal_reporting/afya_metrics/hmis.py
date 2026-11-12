"""Patient counts posted by the HMIS: HOSPITALS.ANALYTICS.AFYA_HMIS_PATIENT_COUNTS (one row per facility per run).

The HMIS posts PATIENT_COUNT, the patients registered at each facility, every hour or so, from one of its
environments (migrate now, production soon). This module decides which run feeds the dashboard and how its facilities
join the warehouse patient counts. It is pure logic; the Snowflake reads are data.read_hmis_environments/read_hmis.

- Environment: chosen on the dashboard (session key SWITCH_KEY); by default production once it has runs, else
  migrate. AFYA_HMIS_ENVIRONMENT sets the default. The chosen environment feeds the table and the totals.
- Rows: the latest whole run; facility and unassigned rows only (organization rows are their totals); the agreed
  counting method; counts above zero.
- Facilities: test organisations (IGNORE_ORGS) are dropped entirely. Facilities the warehouse already counts
  (WAREHOUSE, or an exact name match with a warehouse source) are not added again: their HMIS figure is shown on the
  warehouse row instead, so nobody is counted twice. Every other facility is a row of its own.
- HMIS facilities add PATIENT_COUNT to UPDs (patients). They have no Patient Records (visits).
"""

import os
import re
from datetime import datetime, timedelta

PRODUCTION = "production"
MIGRATE = "migrate"
CHOICES = {PRODUCTION: "Production", MIGRATE: "Migrate"}
SWITCH_KEY = "hmis_env"
OVERRIDE = os.environ.get("AFYA_HMIS_ENVIRONMENT", "").strip().lower()
METHODS = {m.strip() for m in os.environ.get("AFYA_HMIS_METHODS", "gateway_count").split(",") if m.strip()}
STALE_HOURS = 26          # runs are hourly; a day without one is worth a look
JUMP = 0.20               # flag a facility whose count moved more than this since the previous day

# Test organisations: never shown or counted (matched on the organisation name, any environment).
IGNORE_ORGS = {"STAGING", "LIVE DB MIGRATE", "DEMO FACILITY 6", "DRY RUN HOSPITAL"}
# (organisation, facility) -> the warehouse source that already counts it.
WAREHOUSE = {
    ("KISUMU SPECIALIST", "KSH"): "Kisumu Specialist Hospital (KSH)",
    ("WEMA MEDICAL CENTER", "WEMA MEDICAL CENTER"): "Wema Medical",
    ("MEDHANIT", "MEDHANIT"): "Medhanit",
    ("PORTBELL", "PORTBELL"): "Portbell",
    ("SILVERWOOD MEDICAL HOSPITAL", "SILVERWOOD HOSPITAL"): "Silverwood",
}

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip()).upper()


def pick_environment(environments):
    """The default environment, from {name: last captured}: the override, else production, else migrate."""
    names = {str(e).strip().lower(): e for e in environments if e is not None}
    for want in ([OVERRIDE] if OVERRIDE else []) + [PRODUCTION, MIGRATE]:
        if want in names and _SAFE.match(str(names[want])):
            return names[want]
    return None


def resolve(choice, environments):
    """The environment name as stored for a switch choice ('production' / 'migrate'), or None if it has no rows."""
    for e in environments:
        if e is not None and str(e).strip().lower() == choice and _SAFE.match(str(e)):
            return e
    return None


def _nice(s):
    s = re.sub(r"\s+", " ", str(s or "").strip())
    return s.title() if s.isupper() and len(s) > 4 else s


def label(org, facility, scope):
    o, f = _nice(org), _nice(facility)
    if scope == "unassigned" or not f or f == "(unassigned)":
        return f"{o} (unassigned)"
    return f if norm(f) == norm(o) else f"{f} · {o}"


def build(rows, environment, warehouse_sources, now=None):
    """rows: the latest run and the previous day's run (WHICH = 'latest' | 'previous'). Returns a view dict:
    facilities (the rows added to the table), on_warehouse ({warehouse source: HMIS patients} for facilities the
    warehouse already counts) and the run's details. None when the environment has no run."""
    now = now or datetime.now()
    latest = [r for r in rows if r["WHICH"] == "latest"]
    if not latest:
        return None
    key = lambda r: (r["ORGANIZATION_ID"], r["FACILITY_ID"] if r["SCOPE"] == "facility" else "unassigned")
    prev = {key(r): r["PATIENT_COUNT"] for r in rows if r["WHICH"] == "previous"
            and r["COUNT_METHOD"] in METHODS}
    prev_at = max((r["CAPTURED_AT"] for r in rows if r["WHICH"] == "previous"), default=None)
    captured = max(r["CAPTURED_AT"] for r in latest)
    sources = {norm(s): s for s in warehouse_sources}

    facilities, on_warehouse, dropped, seen = [], {}, 0, set()
    for r in latest:
        n = int(r["PATIENT_COUNT"] or 0)
        if r["COUNT_METHOD"] not in METHODS or n <= 0:
            dropped += 1
            continue
        org, fac = norm(r["ORGANIZATION_NAME"]), norm(r["FACILITY_NAME"])
        if org in IGNORE_ORGS:                             # test organisations never reach the dashboard
            continue
        name = label(r["ORGANIZATION_NAME"], r["FACILITY_NAME"], r["SCOPE"])
        if name in seen:                                   # two facilities with the same names: keep both apart
            name = f"{name} (#{r['FACILITY_ID']})"
        seen.add(name)
        source = WAREHOUSE.get((org, fac)) or sources.get(fac) or sources.get(norm(name))
        if source:                                         # already counted from the warehouse: shown on its row
            on_warehouse[source] = on_warehouse.get(source, 0) + n
            continue
        before = prev.get(key(r))
        change = (n - before) / before if before else None
        flags = [f"{change:+.0%} since {prev_at:%d %b}"] if change is not None and abs(change) > JUMP else []
        facilities.append(dict(name=name, patients=n, flags=flags))
    return dict(environment=environment, captured=captured, previous_at=prev_at,
                stale=captured < now - timedelta(hours=STALE_HOURS), facilities=facilities,
                on_warehouse=on_warehouse, dropped=dropped)


def counted_rows(view):
    """The HMIS facilities as patient-count rows: patients into UPDs, no Patient Records."""
    if not view:
        return []
    day = view["captured"].date() if hasattr(view["captured"], "date") else view["captured"]
    return [{"SOURCE": f["name"], "PATIENT_RECORDS": 0, "UPDS": f["patients"], "COUNTED_AT": day, "HMIS": True,
             "FLAGS": f["flags"]} for f in view["facilities"]]
