"""Facility names behind stable codes on the Patient counts tab.

Names are hidden unless the deployment sets AFYA_HIDE_FACILITY_NAMES=false (an unset or unrecognised value keeps
them hidden). There is no switch on the dashboard. A code is turned back into a name with a lookup in Snowflake:
the HMIS code is the facility ID in AFYA_HMIS_PATIENT_COUNTS; a warehouse code is
UPPER(LEFT(SHA1(TRIM(SOURCE)), 4)) of a source in AFYA_PATIENT_COUNTS.

- HMIS facilities: "Facility F-0283" from the HMIS facility ID, so anyone with HMIS access can look it up.
- Warehouse sources (no HMIS ID): "Source S-7C2A" from the source name, the same code on every load.
- A code shared by two rows (a facility ID posted under two organisations, or the same ID in two HMIS environments)
  carries the organisation ID or the environment to keep them apart.
"""

import hashlib
import os

SWITCH = "AFYA_HIDE_FACILITY_NAMES"
_OFF = {"false", "0", "no", "off"}


def hidden(value=None):
    """Names are hidden unless the switch (default: the deployment's AFYA_HIDE_FACILITY_NAMES) says off."""
    value = os.environ.get(SWITCH) if value is None else value
    return value is None or str(value).strip().lower() not in _OFF


def code(row):
    fid = row.get("FACILITY_ID")
    if row.get("HMIS") and fid is not None:
        return f"Facility F-{int(fid):04d}"
    return "Source S-" + hashlib.sha1(str(row["SOURCE"]).encode("utf-8")).hexdigest()[:4].upper()


def codes(rows):
    """{source name: code} for the rows of the sources table."""
    first = {}
    for r in rows:
        first.setdefault(code(r), []).append(r)
    out = {}
    for c, group in first.items():
        for r in group:
            if len(group) == 1:
                out[r["SOURCE"]] = c
            elif len({g.get("ORGANIZATION_ID") for g in group}) == len(group):
                out[r["SOURCE"]] = f"{c}/O-{int(r['ORGANIZATION_ID']):04d}"
            else:
                out[r["SOURCE"]] = f"{c} ({r.get('ENV') or 'Warehouse'})"
    return out


def apply(rows, receipts):
    """The sources table and dated arrivals with every name replaced by its code."""
    names = codes(rows)
    swap = lambda s: names.get(s) or code({"SOURCE": s})
    return ([{**r, "SOURCE": names[r["SOURCE"]]} for r in rows],
            [{**x, "SOURCE": swap(x["SOURCE"])} for x in receipts])
