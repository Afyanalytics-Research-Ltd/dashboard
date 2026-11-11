"""
Phase 2 — Insight Engine.
Converts raw data signals into ranked, structured InsightRows for Today's Briefing.

Pull → Push: the system surfaces the 5 most important things to act on each morning,
rather than waiting for the pharmacist to hunt through individual modules.

Detection rules implemented:
  R1_STOCKOUT       — Critical stockout approaching (DOS < 7d AND clinical priority = CRITICAL/HIGH)
  R2_DEMAND_SPIKE   — Demand 2.5x above 90-day baseline (wraps AnomalyEngine output)
  R3_DEAD_STOCK     — 0 dispenses in 60+ days AND SOH > 0 (capital locked up)
  R5_REFILL_OVERDUE — Patient last dispense > (mean interval × 1.4) AND drug < 7d cover

  R4_LEAD_TIME_MISS — DEFERRED pending MS Dynamics procurement data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from intelligence.config import (
    CRITICAL_THERAPEUTIC_SUBCLASSES,
    HIGH_THERAPEUTIC_CLASSES,
)
from intelligence.priority_scorer import ORDER_NOW, PRIORITY_CRITICAL, PRIORITY_HIGH

# ── Severity constants ────────────────────────────────────────────────────────

SEV_CRITICAL = "CRITICAL"
SEV_HIGH     = "HIGH"
SEV_MEDIUM   = "MEDIUM"

# ── Rule identifiers ──────────────────────────────────────────────────────────

RULE_STOCKOUT       = "R1_STOCKOUT"
RULE_DEMAND_SPIKE   = "R2_DEMAND_SPIKE"
RULE_DEAD_STOCK     = "R3_DEAD_STOCK"
RULE_REFILL_OVERDUE = "R5_REFILL_OVERDUE"
RULE_SEASONAL       = "R6_SEASONAL"

# Minimum dead/idle days to surface on briefing (lower than DEAD_STOCK_DAYS to
# catch 60-day idle items before they reach the 90-day threshold)
BRIEFING_IDLE_THRESHOLD_DAYS: int = 60


@dataclass
class InsightRow:
    """
    A single structured insight surfaced by the Insight Engine.
    Produced by detection rules, classified by severity, narrated by LLM or fallback template.
    """
    rule_id: str                               # R1_STOCKOUT | R2_DEMAND_SPIKE | R3_DEAD_STOCK | R5_REFILL_OVERDUE
    drug: str                                  # canonical_name
    severity: str                              # CRITICAL | HIGH | MEDIUM
    urgency_score: int                         # 0–100, higher = surfaces first
    headline: str                              # 1 sentence, drug name first
    supporting_facts: List[str]                # 2–3 pre-computed data bullets
    recommended_action: str                    # Action chip label
    module: str                                # Destination module for deep-link
    metadata: dict = field(default_factory=dict)  # Raw values for narrator


# ── Urgency scoring ───────────────────────────────────────────────────────────

def _urgency(severity: str, dos: Optional[float] = None,
             magnitude_pct: Optional[float] = None,
             patient_count: int = 0) -> int:
    base = {SEV_CRITICAL: 75, SEV_HIGH: 55, SEV_MEDIUM: 35}.get(severity, 35)

    # Days-of-stock bonus
    if dos is not None:
        if dos <= 0:
            base += 20
        elif dos < 1:
            base += 20
        elif dos < 3:
            base += 15
        elif dos < 7:
            base += 10

    # Demand spike magnitude bonus
    if magnitude_pct is not None:
        if abs(magnitude_pct) > 500:
            base += 15
        elif abs(magnitude_pct) > 200:
            base += 10

    # Patient exposure bonus
    if patient_count > 10:
        base += 10
    elif patient_count > 5:
        base += 5

    return min(100, base)


def _is_critical_class(therapeutic_class: str, therapeutic_subclass: str) -> bool:
    sub = (therapeutic_subclass or "").strip()
    cls = (therapeutic_class or "").strip()
    return sub in CRITICAL_THERAPEUTIC_SUBCLASSES


def _is_high_class(therapeutic_class: str) -> bool:
    return (therapeutic_class or "").strip() in HIGH_THERAPEUTIC_CLASSES


# ── Rule 1: Stockout / near-stockout ─────────────────────────────────────────

def detect_stockout_critical(
    actions_df: pd.DataFrame,
    soh_df: pd.DataFrame,
) -> List[InsightRow]:
    """
    R1: Surface ORDER_NOW items to the briefing as structured InsightRows.
    Severity: CRITICAL if clinical priority CRITICAL; HIGH if HIGH priority or DOS < 3d;
              MEDIUM otherwise.
    Only surfaces items with clinical_priority CRITICAL or HIGH, or DOS < 7d,
    to keep the briefing focused on genuinely urgent items.
    """
    if actions_df.empty:
        return []

    rows: List[InsightRow] = []
    df = actions_df[actions_df["action"] == ORDER_NOW].copy()
    if df.empty:
        return []

    # Merge DOS from soh_df
    _soh = soh_df.copy()
    _soh.columns = _soh.columns.str.lower()
    _dos_col = "days_of_stock_p50" if "days_of_stock_p50" in _soh.columns else "days_of_stock"
    _merge_cols = ["canonical_name", "current_soh", _dos_col]
    if "therapeutic_class" in _soh.columns:
        _merge_cols.append("therapeutic_class")
    if "therapeutic_subclass" in _soh.columns:
        _merge_cols.append("therapeutic_subclass")

    df = df.merge(
        _soh[[c for c in _merge_cols if c in _soh.columns]].rename(
            columns={_dos_col: "dos_remaining"}
        ),
        on="canonical_name",
        how="left",
    )

    for _, r in df.iterrows():
        cp     = str(r.get("clinical_priority", "STANDARD"))
        dos    = r.get("dos_remaining")
        soh    = float(r.get("current_soh", 0) or 0)
        drug   = str(r.get("canonical_name", "Unknown"))
        reason = str(r.get("reason", ""))
        t_sub  = str(r.get("therapeutic_subclass", "") or "")
        t_cls  = str(r.get("therapeutic_class", "") or "")

        # Filter: only CRITICAL/HIGH priority drugs, or anything truly stocked out
        if cp not in (PRIORITY_CRITICAL, PRIORITY_HIGH) and (dos is None or dos >= 7) and soh > 0:
            continue

        if cp == PRIORITY_CRITICAL:
            severity = SEV_CRITICAL
        elif cp == PRIORITY_HIGH or (dos is not None and dos < 3):
            severity = SEV_HIGH
        else:
            severity = SEV_MEDIUM

        dos_str = "Stocked out" if soh <= 0 else (f"{dos:.0f}d cover" if dos is not None else "No data")
        facts = [f"Days of cover: {dos_str}"]
        if cp in (PRIORITY_CRITICAL, PRIORITY_HIGH):
            facts.append(f"Clinical priority: {cp.title()} — patient therapy at risk")
        if soh > 0 and dos is not None:
            facts.append(f"Current SOH: {soh:.0f} units")

        rows.append(InsightRow(
            rule_id=RULE_STOCKOUT,
            drug=drug,
            severity=severity,
            urgency_score=_urgency(severity, dos=float(dos) if dos is not None else None),
            headline=f"{drug} has {dos_str} of stock — immediate order required.",
            supporting_facts=facts[:3],
            recommended_action="Go to Order Workbench",
            module="Order Workbench",
            metadata={
                "dos": float(dos) if dos is not None else None,
                "current_soh": soh,
                "clinical_priority": cp,
                "reason": reason,
            },
        ))

    return rows


# ── Rule 2: Demand spike ──────────────────────────────────────────────────────

def detect_demand_spikes(
    anomalies_df: pd.DataFrame,
    soh_df: pd.DataFrame,
) -> List[InsightRow]:
    """
    R2: Consumption anomaly UP direction only, magnitude ≥ 150% above baseline.
    Severity: CRITICAL if drug is critical class; HIGH if high class; MEDIUM otherwise.
    """
    if anomalies_df.empty:
        return []

    rows: List[InsightRow] = []
    up_spikes = anomalies_df[
        (anomalies_df["is_anomaly"] == True) &
        (anomalies_df["direction"] == "UP") &
        (anomalies_df["magnitude_pct"] >= 150)
    ].copy()

    if up_spikes.empty:
        return []

    # Enrich with therapeutic class from soh_df
    _soh = soh_df.copy()
    _soh.columns = _soh.columns.str.lower()
    _tax_cols = ["canonical_name"]
    for c in ["therapeutic_class", "therapeutic_subclass", "days_of_stock_p50", "days_of_stock"]:
        if c in _soh.columns:
            _tax_cols.append(c)
    up_spikes = up_spikes.merge(_soh[_tax_cols].drop_duplicates("canonical_name"), on="canonical_name", how="left")

    for _, r in up_spikes.iterrows():
        drug      = str(r.get("canonical_name", "Unknown"))
        mag       = float(r.get("magnitude_pct", 0))
        baseline  = float(r.get("baseline_avg_daily", 0))
        recent    = float(r.get("recent_avg_daily", 0))
        t_cls     = str(r.get("therapeutic_class", "") or "")
        t_sub     = str(r.get("therapeutic_subclass", "") or "")
        _dos_col  = "days_of_stock_p50" if "days_of_stock_p50" in r.index else "days_of_stock"
        dos       = r.get(_dos_col)

        if _is_critical_class(t_cls, t_sub):
            severity = SEV_CRITICAL
        elif _is_high_class(t_cls):
            severity = SEV_HIGH
        else:
            severity = SEV_HIGH if mag >= 300 else SEV_MEDIUM

        facts = [
            f"Demand +{mag:.0f}% above 90-day baseline (recent: {recent:.1f} vs baseline: {baseline:.1f} units/day)",
            f"z-score: {r.get('z_score', 0):.1f} — statistically significant",
        ]
        if dos is not None:
            facts.append(f"At current rate: {dos:.0f}d cover — stock may deplete faster than planned")

        rows.append(InsightRow(
            rule_id=RULE_DEMAND_SPIKE,
            drug=drug,
            severity=severity,
            urgency_score=_urgency(severity, dos=float(dos) if dos is not None else None, magnitude_pct=mag),
            headline=f"{drug} demand is +{mag:.0f}% above baseline — review order quantity.",
            supporting_facts=facts[:3],
            recommended_action="Analyse in Stockout Watch",
            module="Stockout Watch",
            metadata={
                "magnitude_pct": mag,
                "baseline_avg_daily": baseline,
                "recent_avg_daily": recent,
                "z_score": float(r.get("z_score", 0)),
                "dos": float(dos) if dos is not None else None,
            },
        ))

    return rows


# ── Rule 3: Dead stock on briefing ────────────────────────────────────────────

def detect_dead_stock_briefing(dead_stock_df: pd.DataFrame) -> List[InsightRow]:
    """
    R3: Surface idle/dead stock items worth most capital to the briefing.
    Only shows top 3 by total_historical_value × current_soh proxy.
    Severity: always MEDIUM (capital tied up, not a patient safety issue).
    """
    if dead_stock_df.empty:
        return []

    df = dead_stock_df.copy()
    df.columns = df.columns.str.lower()

    # Filter to items idle for 60+ days
    if "days_idle" not in df.columns:
        return []
    df = df[df["days_idle"].fillna(0) >= BRIEFING_IDLE_THRESHOLD_DAYS]
    if df.empty:
        return []

    # Sort by historical value descending, take top 3
    if "total_historical_value" in df.columns:
        df = df.sort_values("total_historical_value", ascending=False)
    df = df.head(3)

    rows: List[InsightRow] = []
    for _, r in df.iterrows():
        drug      = str(r.get("canonical_name", "Unknown"))
        days_idle = int(r.get("days_idle", 0) or 0)
        soh       = float(r.get("current_soh", 0) or 0)
        value     = float(r.get("total_historical_value", 0) or 0)
        category  = str(r.get("idle_category", "slow") or "slow")

        facts = [
            f"No dispenses in {days_idle} days — "
            f"{'dead' if category == 'dead' else 'slow-moving'} stock",
            f"Current SOH: {soh:.0f} units — capital locked up",
        ]
        if value > 0:
            facts.append(f"Historical dispensing value: KES {value:,.0f}")

        rows.append(InsightRow(
            rule_id=RULE_DEAD_STOCK,
            drug=drug,
            severity=SEV_MEDIUM,
            urgency_score=_urgency(SEV_MEDIUM),
            headline=f"{drug} has had no movement for {days_idle} days — review for redistribution.",
            supporting_facts=facts[:3],
            recommended_action="Review in Dead Stock Actions",
            module="Dead Stock Actions",
            metadata={
                "days_idle": days_idle,
                "current_soh": soh,
                "idle_category": category,
                "total_historical_value": value,
            },
        ))

    return rows


# ── Rule 5: Patient refill overdue ────────────────────────────────────────────

def detect_patient_refill_overdue(patient_refill_df: pd.DataFrame) -> List[InsightRow]:
    """
    R5: Patients whose estimated supply has run out AND the drug has < 7 days of cover.
    Supply is estimated as: last_qty_dispensed / facility_avg_daily_units (capped at 180d).
    Overdue when: days_since_last_visit > estimated_supply × 1.2.
    Severity: HIGH (patient therapy disruption, but not confirmed stockout).
    """
    if patient_refill_df.empty:
        return []

    df = patient_refill_df.copy()
    df.columns = df.columns.str.lower()

    if "overdue_patient_count" not in df.columns:
        return []

    rows: List[InsightRow] = []
    for _, r in df.iterrows():
        drug          = str(r.get("canonical_name", "Unknown"))
        patient_count = int(r.get("overdue_patient_count", 0) or 0)
        avg_overdue   = float(r.get("avg_days_overdue", 0) or 0)
        dos           = r.get("days_of_cover")
        t_sub         = str(r.get("therapeutic_subclass", "") or "")
        t_cls         = str(r.get("therapeutic_class", "") or "")

        if patient_count <= 0:
            continue

        severity = SEV_CRITICAL if _is_critical_class(t_cls, t_sub) else SEV_HIGH

        dos_str = f"{dos:.0f}d cover" if dos is not None else "low cover"
        facts = [
            f"{patient_count} patient{'s' if patient_count > 1 else ''} past estimated supply "
            f"(avg {avg_overdue:.0f} days beyond expected return)",
            f"Drug cover: {dos_str} — refill may not be dispensable",
        ]
        if t_cls:
            facts.append(f"Class: {t_cls} — therapy continuity at risk")

        rows.append(InsightRow(
            rule_id=RULE_REFILL_OVERDUE,
            drug=drug,
            severity=severity,
            urgency_score=_urgency(
                severity,
                dos=float(dos) if dos is not None else None,
                patient_count=patient_count,
            ),
            headline=(
                f"{patient_count} patient{'s' if patient_count > 1 else ''} "
                f"overdue refill on {drug} — contact and check cover."
            ),
            supporting_facts=facts[:3],
            recommended_action="Review in Patient Risk",
            module="Patient Risk",
            metadata={
                "patient_count": patient_count,
                "avg_days_overdue": avg_overdue,
                "dos": float(dos) if dos is not None else None,
                "therapeutic_class": t_cls,
            },
        ))

    return rows


# ── Deduplication & classification ───────────────────────────────────────────

def _deduplicate(insight_rows: List[InsightRow]) -> List[InsightRow]:
    """
    Merge multiple InsightRows for the same drug into a single compound insight.
    Keeps the highest severity headline; combines supporting facts (max 3).
    Urgency_score = max of individual scores + 5 per additional rule (capped 100).
    """
    # Group by drug name
    by_drug: dict[str, List[InsightRow]] = {}
    for row in insight_rows:
        by_drug.setdefault(row.drug, []).append(row)

    merged: List[InsightRow] = []
    for drug, group in by_drug.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Sort by severity then urgency_score
        _sev_rank = {SEV_CRITICAL: 0, SEV_HIGH: 1, SEV_MEDIUM: 2}
        group.sort(key=lambda r: (_sev_rank.get(r.severity, 3), -r.urgency_score))
        primary = group[0]

        # Merge supporting facts, deduplicate identical strings
        seen: set[str] = set()
        combined_facts: List[str] = []
        for r in group:
            for f in r.supporting_facts:
                if f not in seen:
                    seen.add(f)
                    combined_facts.append(f)

        extra_boost = min(5 * (len(group) - 1), 15)
        merged.append(InsightRow(
            rule_id=primary.rule_id,
            drug=drug,
            severity=primary.severity,
            urgency_score=min(100, primary.urgency_score + extra_boost),
            headline=primary.headline,
            supporting_facts=combined_facts[:3],
            recommended_action=primary.recommended_action,
            module=primary.module,
            metadata={**primary.metadata, "_compound_rules": [r.rule_id for r in group]},
        ))

    return merged


# ── Rule 6: Seasonal demand alert ────────────────────────────────────────────

def detect_seasonal_demand(disease_summaries: list) -> List[InsightRow]:
    """
    R6: Surface approaching seasonal disease peaks as InsightRows.

    One InsightRow per disease (not per drug) — the headline describes the
    disease-level risk; the supporting facts quantify how many drugs are
    affected and the expected demand uplift.

    disease_summaries: output of SeasonalEngine.get_disease_summary()
    """
    if not disease_summaries:
        return []

    rows: List[InsightRow] = []
    for s in disease_summaries:
        disease   = s["disease"]
        weeks     = s["weeks_to_peak"]
        mult      = s["demand_multiplier"]
        at_risk   = s["drugs_at_risk"]
        severity  = s["severity"]
        boosted   = s["climate_boosted"]
        uplift_pct = round((mult - 1) * 100)

        if weeks == 0:
            timing = "currently at peak"
        elif weeks == 1:
            timing = "1 week away"
        else:
            timing = f"{weeks} weeks away"

        headline = (
            f"{disease} season {timing} — "
            f"{at_risk} drug{'s' if at_risk != 1 else ''} may be "
            f"under-stocked at peak demand (+{uplift_pct}% expected)."
        )

        facts = [
            f"Historical demand uplift at peak: +{uplift_pct}% above baseline",
            f"{at_risk} drug{'s' if at_risk != 1 else ''} will not cover "
            f"60 days under seasonal demand rate",
        ]
        if boosted:
            facts.append(
                "Kisumu rainfall is above seasonal average — elevated transmission risk"
            )

        base_score = {SEV_CRITICAL: 70, SEV_HIGH: 50, SEV_MEDIUM: 35}.get(severity, 35)
        urgency = min(100, base_score + (10 if at_risk > 5 else 5 if at_risk > 2 else 0))

        rows.append(InsightRow(
            rule_id=RULE_SEASONAL,
            drug=disease,            # disease name as the "drug" field — won't conflict with R1-R5
            severity=severity,
            urgency_score=urgency,
            headline=headline,
            supporting_facts=facts[:3],
            recommended_action="Build stock in Order Workbench",
            module="Order Workbench",
            metadata={
                "disease": disease,
                "weeks_to_peak": weeks,
                "demand_multiplier": mult,
                "drugs_at_risk": at_risk,
                "climate_boosted": boosted,
            },
        ))

    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def detect_all(
    soh_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    dead_stock_df: pd.DataFrame,
    patient_refill_df: pd.DataFrame,
    top_n: int = 5,
    seasonal_summaries: Optional[list] = None,
) -> List[InsightRow]:
    """
    Run all detection rules, classify, deduplicate, and return the top N insights
    sorted by urgency_score descending.

    Args:
        soh_df:              Current SOH snapshot
        actions_df:          Priority-scored actions DataFrame
        anomalies_df:        Anomaly detection results
        dead_stock_df:       Dead/slow stock candidates
        patient_refill_df:   Overdue refill summary
        top_n:               Maximum number of insights to return (default 5)
        seasonal_summaries:  Output of SeasonalEngine.get_disease_summary() (optional)

    Returns:
        List[InsightRow] sorted by urgency_score descending, max top_n items.
    """
    all_rows: List[InsightRow] = []

    all_rows.extend(detect_stockout_critical(actions_df, soh_df))
    all_rows.extend(detect_demand_spikes(anomalies_df, soh_df))
    all_rows.extend(detect_dead_stock_briefing(dead_stock_df))
    all_rows.extend(detect_patient_refill_overdue(patient_refill_df))
    if seasonal_summaries:
        all_rows.extend(detect_seasonal_demand(seasonal_summaries))

    # Deduplicate same drug across rules (R6 uses disease name, won't merge with drug rules)
    all_rows = _deduplicate(all_rows)

    # Sort by urgency_score descending
    all_rows.sort(key=lambda r: r.urgency_score, reverse=True)

    return all_rows[:top_n]
