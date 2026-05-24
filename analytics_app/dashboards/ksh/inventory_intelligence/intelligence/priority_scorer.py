"""
Clinical and operational urgency scoring.
Produces a ranked action type and clinical priority for each at-risk product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from intelligence.config import (
    CRITICAL_THERAPEUTIC_SUBCLASSES,
    HIGH_THERAPEUTIC_CLASSES,
    DOS_CRITICAL,
    DOS_LOW,
)

# Action types in descending urgency
ORDER_NOW = "ORDER NOW"
ORDER_THIS_WEEK = "ORDER THIS WEEK"
MONITOR = "MONITOR"
REVIEW = "REVIEW"          # dead/slow stock — not a procurement action

# Clinical priority tiers
PRIORITY_CRITICAL = "CRITICAL"
PRIORITY_HIGH = "HIGH"
PRIORITY_STANDARD = "STANDARD"


@dataclass
class PriorityScore:
    product_id: str
    canonical_name: str
    action: str               # ORDER_NOW | ORDER_THIS_WEEK | MONITOR | REVIEW
    clinical_priority: str    # CRITICAL | HIGH | STANDARD
    urgency_score: int        # 0–100, higher = more urgent
    reason: str


def clinical_priority(
    canonical_name: str,
    therapeutic_class: str,
    therapeutic_subclass: str,
) -> str:
    """Classify a product's clinical criticality tier."""
    subclass = (therapeutic_subclass or "").strip()
    t_class = (therapeutic_class or "").strip()

    if subclass in CRITICAL_THERAPEUTIC_SUBCLASSES:
        return PRIORITY_CRITICAL
    if t_class in HIGH_THERAPEUTIC_CLASSES:
        return PRIORITY_HIGH
    return PRIORITY_STANDARD


def score(
    product_id: str,
    canonical_name: str,
    therapeutic_class: str,
    therapeutic_subclass: str,
    days_of_stock: Optional[float],
    current_soh: float,
) -> PriorityScore:
    """
    Compute action type and urgency score for a single product.

    Urgency = base score from DOS tier (0–60)
              + clinical priority bonus (0–40)
    """
    cp = clinical_priority(canonical_name, therapeutic_class, therapeutic_subclass)
    cp_bonus = {"CRITICAL": 40, "HIGH": 25, "STANDARD": 0}[cp]

    if current_soh <= 0:
        base = 60
        action = ORDER_NOW
        reason = "Currently stocked out"
    elif days_of_stock is not None and days_of_stock <= DOS_CRITICAL:
        base = 55
        action = ORDER_NOW
        reason = f"Stockout predicted in ~{int(days_of_stock)}d"
    elif days_of_stock is not None and days_of_stock <= DOS_LOW:
        base = 35
        action = ORDER_THIS_WEEK
        reason = f"{int(days_of_stock)} days of stock remaining"
    elif days_of_stock is None:
        base = 10
        action = MONITOR
        reason = "Insufficient consumption data for forecast"
    else:
        base = 5
        action = MONITOR
        reason = f"{int(days_of_stock)} days of stock — adequate"

    # Escalate action for critical drugs even at moderate DOS
    if cp == PRIORITY_CRITICAL and action == ORDER_THIS_WEEK:
        action = ORDER_NOW
        reason += " (critical drug — escalated)"

    return PriorityScore(
        product_id=product_id,
        canonical_name=canonical_name,
        action=action,
        clinical_priority=cp,
        urgency_score=min(100, base + cp_bonus),
        reason=reason,
    )


def score_all(stock_df: pd.DataFrame) -> "pd.DataFrame":
    """
    Score all products in a stock status DataFrame.
    Expected columns: product_id, canonical_name, therapeutic_class,
                      therapeutic_subclass, days_of_stock, current_soh (or soh_after_raw)
    Accepts both uppercase and lowercase column names.
    """
    import pandas as pd  # local import to keep module lightweight

    df = stock_df.copy()
    df.columns = df.columns.str.lower()

    soh_col = "current_soh" if "current_soh" in df.columns else "soh_after_raw"
    dos_col = "days_of_stock" if "days_of_stock" in df.columns else "days_of_stock_p50"

    rows = []
    for _, r in df.iterrows():
        # Resolve canonical name — fall back to product_id if null/empty so
        # action cards always have an identifiable label.
        raw_name = r.get("canonical_name")
        name = (
            str(raw_name).strip()
            if raw_name is not None and pd.notna(raw_name) and str(raw_name).strip()
            else str(r.get("product_id", "Unknown"))
        )
        s = score(
            product_id=r.get("product_id", ""),
            canonical_name=name,
            therapeutic_class=r.get("therapeutic_class", ""),
            therapeutic_subclass=r.get("therapeutic_subclass", ""),
            days_of_stock=r.get(dos_col),
            current_soh=float(r.get(soh_col, 0) or 0),
        )
        rows.append(s.__dict__)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("urgency_score", ascending=False)
    return result
