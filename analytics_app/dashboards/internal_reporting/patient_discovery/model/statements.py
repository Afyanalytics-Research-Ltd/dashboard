"""Where statements come from, and where answers go.

- the registry (AFYA_PATIENT_COUNT_SOURCES): 'active' rows are counted sources, 'rejected' rows are rejections;
  'proposed' rows come from the retired in-database discovery and are never used (reported as legacy);
- config.py: decisions D2, D3, D6, D7 and the event-class policies;
- statements.json (next to this file, human-authored, versioned): identifier namespaces, onboarded facilities,
  per-facility policies, 'not a source' answers. Every entry records who, when and on what basis.
Onboarding: a facility with an active source is onboarded (origin 'registry'); others only by statement.
"""
import json
from datetime import datetime
from pathlib import Path

from .. import config as C
from ..classify import family
from .concepts import Statement

FILE = Path(__file__).with_name("statements.json")


def policies():
    return [
        Statement("D2", "exclude_schemas", "schemas", {"schemas": sorted(C.EXCLUDED_SCHEMAS)}, origin="config"),
        Statement("D3", "hold_schemas", "schemas", {"schemas": sorted(C.HELD_SCHEMAS)}, origin="config"),
        Statement("D7", "covered_schemas", "schemas", {"schemas": dict(C.COVERED_SCHEMAS)},
                  premise={"metric": "covering_source_counted"}, origin="config"),
        Statement("D6", "event_class", "admission", {"action": "exclude"},
                  premise={"metric": "contribution_share", "max": 0.01}, origin="config"),
        Statement("P-REG", "event_class", "registration", {"action": "exclude"}, origin="config"),
        Statement("P-SUP", "event_class", "supporting", {"action": "exclude"}, origin="config"),
        Statement("P-SCH", "event_class", "scheduling", {"action": "exclude"}, origin="config"),
        Statement("P-DER", "derived_schemas", "schemas", {"schemas": sorted(C.DERIVED_SCHEMAS)}, origin="config"),
    ]


def read_file(path=FILE):
    data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {"statements": []}
    return [Statement(s["id"], s["kind"], s["subject"], s["assertion"], s.get("premise"), s.get("origin", "human"))
            for s in data["statements"]]


def from_registry(rows):
    """(statements, legacy proposals). Keys are upper-cased like the refresh does."""
    st, legacy, onboard = [], [], set()
    for r in rows:
        f, status = r["SOURCE_OBJECT"], (r["STATUS"] or "").lower()
        if status == "active":
            st.append(Statement(f"S:{r['SOURCE']}", "source", f, dict(
                source=r["SOURCE"], key=r["UPD_KEY"], record=r["RECORD_KEY"], deleted=r["DELETED_COLUMN"],
                version=r["VERSION_COLUMN"], received=r["RECEIVED_COLUMN"]), origin="registry"))
            onboard.add(family(f.split(".")[1]))
        elif status == "rejected":
            st.append(Statement(f"R:{f}", "rejected", f, {"source": r["SOURCE"]}, origin="registry"))
        else:
            legacy.append(dict(object=f, status=status, source=r["SOURCE"]))
    st += [Statement(f"F:{fac}", "facility", fac, {"onboarded": True}, origin="registry") for fac in sorted(onboard)]
    return st, legacy


def from_dicts(dicts):
    return [Statement(s["id"], s["kind"], s["subject"], s["assertion"], s.get("premise"), s.get("origin", "human"))
            for s in dicts]


def load(registry_rows, path=FILE, stored=()):
    """Registry + config + statements.json + answers stored in Snowflake (a stored answer replaces the file's
    statement with the same id)."""
    st, legacy = from_registry(registry_rows)
    people = {s.id: s for s in read_file(path)}
    people.update({s.id: s for s in from_dicts(stored)})
    return st + policies() + list(people.values()), legacy


def append(new, path=FILE):
    """Add statements (dicts) to the file; a namespace with an existing id gains members instead of duplicating."""
    data = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {"statements": []}
    by_id = {s["id"]: s for s in data["statements"]}
    for s in new:
        old = by_id.get(s["id"])
        if old and s["kind"] == "namespace":
            members = old["assertion"]["members"]
            members += [m for m in s["assertion"]["members"] if m not in members]
            old["assertion"].setdefault("history", []).append(s["assertion"].get("provenance"))
        elif old:
            continue                                            # the same fact is never recorded twice
        else:
            data["statements"].append(s)
            by_id[s["id"]] = s
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def provenance(question, by, basis):
    return dict(question=question, by=by, at=datetime.now().isoformat(timespec="seconds"), basis=basis)
