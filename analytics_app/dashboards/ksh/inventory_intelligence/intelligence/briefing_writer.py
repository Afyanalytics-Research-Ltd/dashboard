"""AI-powered daily briefing generator with rule-based fallback."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from intelligence import ai_client
from intelligence.priority_scorer import ORDER_NOW, ORDER_THIS_WEEK


# ── Public entry point ────────────────────────────────────────────────────────

def generate(
    facility_name: str,
    kpi_row: dict,
    actions_df: pd.DataFrame,
    anomalies_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Return a 2–4 sentence briefing for the Today's Briefing page.
    Tries the configured LLM first; falls back to rule-based if unavailable.
    """
    if ai_client.get_provider() != "none":
        result = _generate_llm(facility_name, kpi_row, actions_df, anomalies_df)
        if result:
            return result
    return _generate_rule_based(facility_name, kpi_row, actions_df, anomalies_df)


# ── LLM path ──────────────────────────────────────────────────────────────────

def _generate_llm(
    facility_name: str,
    kpi_row: dict,
    actions_df: pd.DataFrame,
    anomalies_df: Optional[pd.DataFrame],
) -> Optional[str]:
    kpi = {k.lower(): v for k, v in kpi_row.items()}

    # ── Pre-compute every figure that should appear in the output ─────────────
    stockouts   = int(kpi.get("active_stockouts", 0) or 0)
    critical    = int(kpi.get("critical_count", 0) or 0)
    low         = int(kpi.get("low_count", 0) or 0)
    total       = int(kpi.get("total_products", 0) or 0)
    chronic     = int(kpi.get("chronic_patients_active", 0) or 0)
    opioid      = int(kpi.get("opioid_patients_active", 0) or 0)
    value_30d   = float(kpi.get("total_dispensing_value_30d") or 0)

    at_risk_count = stockouts + critical
    at_risk_pct   = round(at_risk_count / total * 100, 1) if total > 0 else 0.0
    patients_total = chronic + opioid

    urgent_drugs: list[str] = []
    urgent_count = 0
    if not actions_df.empty:
        _now = actions_df[actions_df["action"] == ORDER_NOW].sort_values(
            "urgency_score", ascending=False
        )
        urgent_count = len(_now)
        urgent_drugs = _now.head(4)["canonical_name"].tolist()

    anomaly_lines: list[str] = []
    if anomalies_df is not None and not anomalies_df.empty:
        for _, r in anomalies_df.head(2).iterrows():
            verb = "above" if r["direction"] == "UP" else "below"
            anomaly_lines.append(
                f"  - {r['canonical_name']}: {abs(r['magnitude_pct']):.0f}% {verb} baseline"
            )
    anom_block = "\n".join(anomaly_lines) if anomaly_lines else "  None detected"

    prompt = f"""Write a 2–4 sentence daily stock alert addressed directly to the {facility_name} pharmacy team.
Speak to them directly — use "you have", "your portfolio", "order now" — never "the pharmacy team is facing" or third-person.
Plain prose — no bullet points, no headers.
IMPORTANT: Use ONLY the numbers below. Do not calculate, derive, or invent any figures.

Pre-computed facts (use these exact numbers):
- Total SKUs tracked: {total}
- Stocked out now: {stockouts}
- Critical (< 7 days): {critical}
- At immediate risk (stocked out + critical): {at_risk_count} ({at_risk_pct}% of portfolio)
- Low stock (7–30 days): {low}
- Drugs needing immediate orders: {urgent_count} ({', '.join(urgent_drugs) if urgent_drugs else 'none'})
- 30-day dispensing value: KES {value_30d:,.0f}
- Patients potentially affected: {patients_total:,} ({chronic:,} chronic, {opioid:,} opioid therapy)
- Consumption anomalies flagged:
{anom_block}

Write the alert now:"""

    return ai_client.complete(
        prompt,
        system_prompt=(
            "You are a pharmacy intelligence system sending a direct daily alert to the pharmacist-in-charge. "
            "Write as if this will be forwarded as a WhatsApp message — direct, specific, actionable. "
            "Always address the team directly: 'You have X stockouts' not 'The pharmacy has X stockouts'. "
            "Plain prose — no bullet points, no headers. "
            "Never invent numbers — use only the pre-computed figures provided."
        ),
        max_tokens=220,
    )


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _generate_rule_based(
    facility_name: str,
    kpi_row: dict,
    actions_df: pd.DataFrame,
    anomalies_df: Optional[pd.DataFrame],
) -> str:
    kpi = {k.lower(): v for k, v in kpi_row.items()}

    stockouts = int(kpi.get("active_stockouts", 0) or 0)
    critical  = int(kpi.get("critical_count", 0) or 0)
    low       = int(kpi.get("low_count", 0) or 0)
    chronic   = int(kpi.get("chronic_patients_active", 0) or 0)
    opioid    = int(kpi.get("opioid_patients_active", 0) or 0)

    sentences: list[str] = []
    urgent_actions = (
        len(actions_df[actions_df["action"] == ORDER_NOW])
        if not actions_df.empty else 0
    )

    if stockouts == 0 and critical == 0:
        sentences.append(
            f"Your stock levels are healthy with no immediate shortfalls today."
        )
    elif urgent_actions >= 5:
        sentences.append(
            f"You have significant procurement pressure right now — "
            f"{urgent_actions} products need immediate orders."
        )
    else:
        total_at_risk = stockouts + critical
        sentences.append(
            f"You have {total_at_risk} product{'s' if total_at_risk != 1 else ''} "
            f"at immediate stockout risk and {low} running low."
        )

    if not actions_df.empty:
        top_now = (
            actions_df[actions_df["action"] == ORDER_NOW]
            .sort_values("urgency_score", ascending=False)
            .head(3)["canonical_name"]
            .tolist()
        )
        if top_now:
            sentences.append(f"Order now: {_oxford_list(top_now)}.")

    if chronic > 0 or opioid > 0:
        parts = []
        if chronic > 0:
            parts.append(f"{chronic:,} chronic disease patients")
        if opioid > 0:
            parts.append(f"{opioid:,} opioid therapy patients")
        sentences.append(
            f"Current shortfalls put {' and '.join(parts)} at risk — escalate if these drugs are out."
        )

    if anomalies_df is not None and not anomalies_df.empty:
        top = anomalies_df.nlargest(1, "magnitude_pct").iloc[0]
        verb = "above" if top["direction"] == "UP" else "below"
        sentences.append(
            f"Flag: {top['canonical_name']} is {abs(top['magnitude_pct']):.0f}% {verb} "
            f"its recent baseline — review before placing a standard order."
        )

    if stockouts == 0 and critical == 0 and (anomalies_df is None or anomalies_df.empty):
        week_orders = (
            len(actions_df[actions_df["action"] == ORDER_THIS_WEEK])
            if not actions_df.empty else 0
        )
        if week_orders > 0:
            sentences.append(
                f"You have {week_orders} product{'s' if week_orders != 1 else ''} to "
                f"order this week to maintain cover."
            )

    return " ".join(sentences)


def _oxford_list(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
