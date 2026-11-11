"""Metadata-only classification of one warehouse object: role, person key, grain columns, facility family.

Pure functions of names and types; no warehouse access.
"""
import re
from dataclasses import dataclass, field

from . import config as C


def tokens(name):
    return [t for t in name.upper().split("_") if t]


def family(schema):
    toks = tokens(schema)
    while len(toks) > 1 and (toks[-1] in C.FAMILY_TRAILING_TOKENS or re.fullmatch(r"V\d+", toks[-1])):
        toks.pop()
    return "_".join(toks) or schema.upper()


def key_tier(column, data_type):
    """Tier 1-3 if the column can identify a person, else None."""
    if data_type not in C.KEY_TYPES:
        return None
    name = column.upper()
    if name in C.PERSON_KEYS:
        return C.PERSON_KEYS[name]
    toks = tokens(name)
    if (set(toks) & C.KEY_VARIANT_TOKENS and toks[-1] in C.KEY_VARIANT_LAST
            and not set(toks) & C.NOT_PERSON_KEY_TOKENS):
        return 3
    return None


def role(object_name):
    """copy | identity | encounter | scheduling | satellite | register | other, plus a 'service subset' flag."""
    name = object_name.upper()
    toks = tokens(name)
    if set(toks) & C.COPY_TOKENS or re.search(C.DATED_COPY_SUFFIX, name):
        return "copy", False
    if set(toks) & C.IDENTITY_TOKENS or (toks and toks[-1] in C.IDENTITY_LAST_TOKENS):
        return "identity", False
    core = [t for t in toks if t not in C.ROLE_MODIFIER_TOKENS and not re.fullmatch(r"V?\d+", t)]
    head = core[-1] if core else ""
    for r in ("encounter", "scheduling", "satellite", "register"):
        if (head in C.ROLE_TOKENS[r]
                or (r == "encounter" and head.endswith(C.ENCOUNTER_SUFFIXES))
                or (r == "register" and head.endswith(C.REGISTER_SUFFIXES))):
            subset = r == "encounter" and any(x in C.ROLE_TOKENS["satellite"] for x in core[:-1])
            return r, subset          # e.g. LAB_VISITS: encounter rows for one service point only
    return "other", False


def names_an_encounter(object_name):
    """True if any token is an encounter word (a copy of a visit table stays visible for review)."""
    return any(t in C.ROLE_TOKENS["encounter"] or t.endswith(C.ENCOUNTER_SUFFIXES) for t in tokens(object_name))


def _first(names, columns):
    upper = {c.upper(): c for c in columns}
    return next((upper[n] for n in names if n in upper), None)


@dataclass
class Classified:
    fqn: str
    schema: str
    name: str
    object_type: str
    row_count: object
    last_altered: object
    columns: list                      # [(name, data_type)]
    role: str = "other"
    service_subset: bool = False
    family: str = ""
    key: object = None
    key_type: object = None
    tier: object = None
    keys: list = field(default_factory=list)
    record_key: object = None
    deleted_column: object = None
    inverted_deleted: object = None
    version_column: object = None
    received_column: object = None
    date_column: object = None
    file_column: object = None
    household_column: object = None
    identifier_ok: bool = True
    person_key_count: int = 0          # columns that look like tier 1-2 person keys
    deletion_column_count: int = 0


def classify(obj):
    """obj: dict(schema, name, type, row_count, last_altered, columns=[(name, type)])."""
    cols = obj["columns"]
    names = [c for c, _ in cols]
    c = Classified(fqn=f"{C.DATABASE}.{obj['schema']}.{obj['name']}", schema=obj["schema"], name=obj["name"],
                   object_type=obj["type"], row_count=obj.get("row_count"), last_altered=obj.get("last_altered"),
                   columns=cols)
    c.role, c.service_subset = role(obj["name"])
    c.family = family(obj["schema"])
    c.keys = sorted(((n, key_tier(n, t), t) for n, t in cols if key_tier(n, t)), key=lambda k: k[1])
    if c.keys:
        c.key, c.tier, c.key_type = c.keys[0]
    c.record_key = _first(C.RECORD_KEYS, names)
    c.deleted_column = _first(C.DELETED_COLUMNS, names)
    c.person_key_count = sum(1 for k in c.keys if k[1] <= 2)
    c.deletion_column_count = sum(1 for n in names if n.upper() in C.DELETED_COLUMNS)
    c.inverted_deleted = _first(C.INVERTED_DELETED_COLUMNS, names)
    c.version_column = _first(C.VERSION_COLUMNS, names) if c.record_key else None
    c.received_column = _first(C.RECEIVED_COLUMNS, names)
    c.file_column = _first(C.FILE_COLUMNS, names)
    c.household_column = _first(C.HOUSEHOLD_COLUMNS, names)
    c.date_column = next((n for n, t in cols if t in C.DATE_TYPES
                          and not set(tokens(n)) & C.NOT_VISIT_DATE_TOKENS), None)
    config_names = [obj["schema"], obj["name"]] + [x for x in (c.key, c.record_key, c.deleted_column,
                                                                c.version_column, c.received_column) if x]
    c.identifier_ok = all(re.fullmatch(C.REFRESH_IDENTIFIER, n) for n in config_names)
    return c


def copy_schema(schema):
    """True when the schema's last meaningful token marks a test / backup copy (X_TEST_CLEAN, X_CLEAN_BAK)."""
    toks = [t for t in schema.upper().split("_") if t and t not in ("CLEAN", "RAW") and not t.startswith("V")]
    return bool(toks) and toks[-1] in C.COPY_SCHEMA_TOKENS
