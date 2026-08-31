"""Inventory scope — which items are MEDICAL inventory the engine should manage.

St. Peter's source catalogue (SALEITEMS) is a point-of-sale list that mixes
clinical stock (drugs, implants, sutures, dressings, lab) with genuinely
non-medical supplies the store also issues: office stationery, cleaning /
janitorial chemicals, furniture, batteries, and patient-hotel items. Left in,
the non-medical lines distort every clinical view (turnover, ABC, ordering) —
one bulk-issued stationery line (Printing Papers) alone dwarfed real drug value.

There is no clean medical/non-medical flag in the source and the taxonomy
``product_category`` is unreliable ('other' is a catch-all that holds both
Branula cannulas and box files), so scope is decided here by a documented,
reviewed name classifier. It is deliberately CONSERVATIVE: any clinical marker
(``KEEP``) wins, so a borderline name is kept in scope rather than risk dropping
a real clinical item (e.g. a surgical "skin stapler" is kept; a "pen torch"
exam light is kept; dental "in-office" procedures are kept).

This is an explicit, auditable scope definition — not a hidden category filter
(implants live in 'other'/'equipment', so a category filter would drop them).
Export the full per-item classification with ``classify_catalog`` for review;
maintainers can extend the two pattern lists below as new lines appear.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# Non-medical: office / stationery, cleaning & janitorial, furniture, batteries,
# and patient-hotel (non-clinical) supplies.
_NON_MEDICAL = re.compile(
    r"\bpens?\b|biro|felt pen|marker pen|staedler|treasury tag|office tray|"
    r"office chair|visitor chair|visitors chair|visitor book|box file|"
    r"spring file|self ink|stamp ink|paper punch|appointment card|conquerer|"
    r"computer paper|laser film|thermal paper|thermo paper|\blabels?\b|"
    r"\bstaplers?\b|stapple|\bstaples\b|\benvelopes?\b|\benvelops?\b|cheque|"
    r"cash ?book|counter ?book|note ?book|foolscap|letterhead|printing paper|"
    r"receipt paper|paper towel|hand paper|kitchen paper|"
    r"toilet (paper|tissue|ball|bowl)|\bmops?\b|jumbo mop|floor brite|"
    r"scouring pad|micro ?fibre|dusting|descaler|ungerol|industrial salt|"
    r"\bnpq\b|ufacid|fridge guard|spray bottle|bin ?liner|pedal bin|"
    r"packing bag|packaging bag|masking tape|duct ?tape|hair net|\bcurtains?\b|"
    r"stepping stool|\bbasin\b|air fresh|freshn?er|moth ball|pine perfume|"
    r"\bperfume\b|neetkleen|neet ceramic|mortein|\bdoom\b|\bbatter|eveready|"
    r"bathing soap|toothbrush|toothpaste|sensodyne|colgate|\bslippers?\b|"
    r"tumblers?|\bspoons?\b",
    re.I,
)

# Clinical markers that KEEP an item in scope even if it brushes a pattern above
# (surgical/skin stapler, pen torch, in-office dental, medicated soaps, etc.).
_KEEP = re.compile(
    r"\bskin\b|medical|surgical|suture|catheter|dressing|gauze|syringe|needle|"
    r"cannula|splint|brace|crutch|sling|insole|collar|orthosis|implant|cement|"
    r"vacutainer|reagent|antigen|\bkit\b|strip|drops?|inject|tablet|capsule|"
    r"syrup|cream|ointment|infusion|swab|povidone|dettol|savlon|antiseptic|"
    r"chlorhex|formalin|torch|in-office|in office",
    re.I,
)


def is_non_medical(name: Optional[str]) -> bool:
    """True when a product name is a genuinely non-medical supply (out of scope).
    Conservative: any clinical marker keeps the item in scope."""
    s = "" if name is None else str(name)
    if _KEEP.search(s):
        return False
    return bool(_NON_MEDICAL.search(s))


def non_medical_item_keys(catalog: pd.DataFrame) -> set:
    """Item keys in ``catalog`` classified non-medical. Uses product_name, then
    canonical_name. Empty when the catalog lacks names/keys."""
    if catalog is None or "item_key" not in getattr(catalog, "columns", []):
        return set()
    name = catalog.get("product_name")
    canon = catalog.get("canonical_name")
    names = (canon if name is None else
             (name.fillna(canon) if canon is not None else name))
    if names is None:
        return set()
    mask = names.fillna("").map(is_non_medical)
    return set(catalog.loc[mask, "item_key"].astype(str))


def excluded_item_keys(catalog: pd.DataFrame,
                       review_csv: Optional[str] = None) -> set:
    """Item keys to exclude from clinical inventory.

    If ``review_csv`` exists it is AUTHORITATIVE — a hospital-owned, editable
    drop-table: any row whose ``scope`` is not 'medical' (e.g. 'non_medical',
    'exclude', 'data_error', 'test') is dropped, and any row flipped back to
    'medical' is kept. This is the "edit a table to drop it" mechanism, for both
    non-medical items and one-off data-entry/test errors the hospital confirms.
    When the file is absent, fall back to the automatic classifier.
    """
    import os
    if review_csv and os.path.isfile(review_csv):
        try:
            r = pd.read_csv(review_csv, dtype={"item_key": str})
            if {"item_key", "scope"} <= set(r.columns):
                drop = r[r["scope"].astype(str).str.strip().str.lower() != "medical"]
                return set(drop["item_key"].astype(str))
        except Exception:
            pass
    return non_medical_item_keys(catalog)


def classify_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Full per-item scope classification for review/audit.
    Columns: item_key, name, product_category, scope ('medical'|'non_medical')."""
    if catalog is None or catalog.empty:
        return pd.DataFrame(columns=["item_key", "name", "product_category", "scope"])
    df = catalog.copy()
    name = df.get("product_name")
    canon = df.get("canonical_name")
    df["name"] = (canon if name is None else
                  (name.fillna(canon) if canon is not None else name)).fillna("")
    df["scope"] = df["name"].map(
        lambda n: "non_medical" if is_non_medical(n) else "medical")
    cols = ["item_key", "name", "product_category", "scope"]
    return df[[c for c in cols if c in df.columns]]
