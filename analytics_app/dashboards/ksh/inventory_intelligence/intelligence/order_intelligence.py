"""
AI-powered order recommendation intelligence.

The quantity math lives in safety_stock.py — this module wraps those computed
numbers with AI-generated clinical reasoning:
  - Why this quantity and why now
  - Stockout gap during lead time (days with no stock while waiting for delivery)
  - Patient impact context
  - Cost estimate from dispensing value data
  - Rule-based fallback if no LLM is configured

Designed to be called on-demand (not bulk on page load) to keep API costs minimal.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from intelligence import ai_client
from intelligence.config import DEFAULT_LEAD_TIME_DAYS


@dataclass
class OrderBrief:
    canonical_name: str
    recommended_qty: int
    cover_days: int
    lead_time_days: int
    stockout_gap_days: int          # days with no stock while waiting for delivery
    cost_estimate_kes: Optional[float]
    narrative: str                   # 2-3 sentence reasoning
    is_ai: bool                      # True = LLM generated, False = rule-based


_SYSTEM = (
    "You are a pharmacy procurement assistant in a Kenyan hospital. "
    "Write exactly 2 sentences. Sentence 1: the specific stock position and what it means for procurement timing. "
    "Sentence 2: one non-obvious consideration — lead time risk, demand trend implication, or ordering strategy. "
    "Rules: reference the exact numbers given. Never write generic filler like 'it is essential', "
    "'critical nature', 'patient care may be compromised', 'it is crucial', or 'vulnerable to stockouts' — "
    "these are obvious from the data and add no value. If you cannot say something specific and useful, "
    "describe the stockout gap and ordering window precisely."
)


def generate(
    canonical_name: str,
    dos_remaining: float,
    avg_daily_units: float,
    current_soh: float,
    order_qty: int,
    target_cover_days: int = 30,
    clinical_priority: str = "STANDARD",
    therapeutic_class: str = "",
    patients_at_risk: int = 0,
    trend_direction: str = "STABLE",
    confidence: str = "MEDIUM",
    lead_time_days: int = DEFAULT_LEAD_TIME_DAYS,
    avg_unit_value_kes: Optional[float] = None,
) -> OrderBrief:
    """
    Generate an AI order brief for a single drug.
    Falls back to rule-based narrative if no LLM is configured or the call fails.
    """
    cost_est: Optional[float] = None
    if avg_unit_value_kes and avg_unit_value_kes > 0 and order_qty > 0:
        cost_est = round(order_qty * avg_unit_value_kes)

    gap = max(0, math.ceil(lead_time_days - max(dos_remaining, 0)))

    prompt = _build_prompt(
        canonical_name, dos_remaining, avg_daily_units, current_soh,
        order_qty, target_cover_days, clinical_priority, therapeutic_class,
        patients_at_risk, trend_direction, confidence, lead_time_days, cost_est, gap,
    )
    text = ai_client.complete(prompt, system_prompt=_SYSTEM, max_tokens=160)

    if text:
        return OrderBrief(
            canonical_name=canonical_name,
            recommended_qty=order_qty,
            cover_days=target_cover_days,
            lead_time_days=lead_time_days,
            stockout_gap_days=gap,
            cost_estimate_kes=cost_est,
            narrative=text,
            is_ai=True,
        )

    return OrderBrief(
        canonical_name=canonical_name,
        recommended_qty=order_qty,
        cover_days=target_cover_days,
        lead_time_days=lead_time_days,
        stockout_gap_days=gap,
        cost_estimate_kes=cost_est,
        narrative=_rule_based(
            canonical_name, dos_remaining, order_qty, lead_time_days,
            gap, patients_at_risk, clinical_priority, cost_est, trend_direction,
        ),
        is_ai=False,
    )


def _build_prompt(
    name: str, dos: float, adc: float, soh: float, qty: int,
    cover: int, priority: str, t_class: str, patients: int,
    trend: str, confidence: str, lead_time: int,
    cost: Optional[float], gap: int,
) -> str:
    cost_str = f"~KES {cost:,.0f}" if cost else "cost data unavailable"
    gap_str = (
        f"a {gap}-day stockout gap is expected before the order arrives"
        if gap > 0 else "stock will hold through the lead time"
    )
    trend_str = {
        "UP": "consumption trending upward",
        "DOWN": "trending downward",
        "STABLE": "demand stable",
    }.get(trend, "demand stable")

    qty_line = (
        f"Recommended order: {qty:,} units to cover {cover} days at {adc:.1f} u/day | {cost_str}"
        if qty > 0 and adc > 0
        else f"Recommended order: UNKNOWN — no dispensing history available; pharmacist must estimate from clinical need | {cost_str}"
    )

    patients_line = (
        f"Patients on this drug at risk of stockout: {patients:,}"
        if patients > 0 else ""
    )

    return f"""Drug: {name}
Priority: {priority} | Class: {t_class or "unclassified"}
Stock: {soh:.0f} units | Days of stock: {dos:.1f}d | Lead time: {lead_time}d
Stockout gap if ordered today: {gap} days ({gap_str})
{qty_line}
Demand trend: {trend_str} | Confidence: {confidence}
{patients_line}

Write 2 sentences about this specific drug's procurement situation. Be precise about timing and risk."""


def analyse_anomaly(ctx: dict) -> tuple[str, bool]:
    """
    AI action recommendation for a consumption anomaly, grounded in real data.
    ctx is the dict returned by anomaly_engine.build_anomaly_context().
    Returns (one_sentence_action, is_ai).

    The spike_type drives both the prompt framing and the rule-based fallback:
      TRANSIENT  → investigate the event date; do NOT order based on spike rate
      DECLINING  → hold order decision; monitor for 5 more days
      SUSTAINED  → act on new demand rate; adjust standing order
    """
    _SYSTEM = (
        "You are a hospital pharmacy analyst. Given specific consumption data "
        "and a spike classification, write exactly ONE sentence telling the pharmacist "
        "the most important action right now. Be specific to the numbers. No preamble."
    )

    name             = ctx["name"]
    direction        = ctx["direction"]
    base_avg         = ctx["baseline_avg"]
    recent_avg       = ctx["recent_avg"]
    safe_adc         = ctx.get("safe_order_adc", base_avg)
    proper_order_qty = ctx.get("proper_order_qty", 0)
    magnitude        = abs(ctx["magnitude_pct"])
    soh              = ctx["current_soh"]
    days_curr        = ctx["days_at_current_rate"]
    days_norm        = ctx["days_at_normal_rate"]
    correlated       = ctx["correlated_drugs"]
    spike_start      = ctx.get("spike_start")
    spike_type       = ctx.get("spike_type", "SUSTAINED")
    t_class          = ctx.get("therapeutic_class", "")

    verb          = "above" if direction == "UP" else "below"
    direction_w   = "spike" if direction == "UP" else "drop"

    spike_line = (
        f"Spike started: ~{spike_start.strftime('%d %b')} "
        f"({(ctx['ref_date'] - spike_start).days} days ago)"
        if spike_start is not None else "Onset: within the last 14 days"
    )
    stock_line = (
        f"Current SOH: {soh:.0f} units "
        f"(target after order: 30-day cover + lead time = {int((30 + DEFAULT_LEAD_TIME_DAYS) * safe_adc)} units total)"
    )
    corr_line = (
        f"Also {direction_w}ing (same class): {', '.join(correlated)}"
        if correlated else ""
    )

    # Determine whether an order is needed regardless of spike type
    _stocked_out    = soh <= 0
    _low_stock      = days_norm is not None and days_norm <= 7
    _needs_order    = _stocked_out or _low_stock
    _order_qty_str  = f"{proper_order_qty:,} units" if proper_order_qty > 0 else "quantity per Workbench"

    type_guidance = {
        "TRANSIENT": (
            f"Classification: TRANSIENT — spike concentrated in 1-3 days, now resolved. "
            f"Do NOT use the spike rate ({recent_avg:.1f}/day) for ordering. "
            f"Safe rate is {safe_adc:.1f} u/day (baseline). "
            + (f"HOWEVER: current SOH is critically low ({soh:.0f} units). "
               f"Order {_order_qty_str} at baseline rate to restore 30-day cover, then investigate the spike event."
               if _needs_order else
               f"Current SOH is adequate at baseline rate. Investigate the spike event only.")
        ),
        "DECLINING": (
            f"Classification: DECLINING — spike is actively reversing. "
            f"Safe rate is {safe_adc:.1f} u/day (baseline). "
            + (f"Current SOH is critically low ({soh:.0f} units) — order {_order_qty_str} at baseline rate despite the declining spike."
               if _needs_order else
               f"Hold order decision and monitor for 5 more days.")
        ),
        "SUSTAINED": (
            f"Classification: SUSTAINED — consistently elevated across 14 days. Genuine demand shift. "
            f"Use safe rate {safe_adc:.1f} u/day for ordering. "
            f"Recommended order: {_order_qty_str} (30d cover + lead time at safe rate, minus SOH {soh:.0f})."
        ),
    }.get(spike_type, "")

    prompt = f"""Drug: {name} ({t_class or "unclassified"})
Consumption {direction_w}: {magnitude:.0f}% {verb} 90-day baseline
Normal rate: {base_avg:.1f} u/day | Spike rate: {recent_avg:.1f} u/day | Safe order rate: {safe_adc:.1f} u/day
{spike_line}
{stock_line}
{corr_line}
{type_guidance}

Write ONE sentence — the most important action. Use the pre-calculated qty ({_order_qty_str}) and safe rate. The ORDER TARGET is always 30-day cover plus lead time — never use the current days-in-stock as the target."""

    text = ai_client.complete(prompt, system_prompt=_SYSTEM, max_tokens=90)
    if text:
        return text.strip().lstrip("•-– "), True

    # ── Rule-based fallback — spike-type and stock-aware ───────────────────────
    spike_date_str = spike_start.strftime("%d %b") if spike_start else "recently"

    if direction == "UP":
        if spike_type == "TRANSIENT":
            if _needs_order:
                action = (
                    f"Order {_order_qty_str} of {name} at the baseline rate of {safe_adc:.1f} u/day "
                    f"to cover low stock, then investigate the {spike_date_str} bulk dispensing event "
                    f"— do not use the spike rate of {recent_avg:.1f} u/day for ordering."
                )
            else:
                action = (
                    f"Investigate the {name} dispensing spike on {spike_date_str} "
                    f"— it has already resolved; the standing order should stay at "
                    f"{safe_adc:.1f} u/day (baseline), not the spike rate."
                )
        elif spike_type == "DECLINING":
            if _needs_order:
                action = (
                    f"Order {_order_qty_str} of {name} at baseline rate {safe_adc:.1f} u/day "
                    f"— stock is critically low even as the spike declines."
                )
            else:
                action = (
                    f"Hold the {name} order — the spike is declining and SOH covers "
                    f"{days_norm} days at baseline; reassess in 5 days."
                )
        else:  # SUSTAINED
            action = (
                f"Order {_order_qty_str} of {name} immediately — sustained consumption "
                f"of {safe_adc:.1f} u/day ({magnitude:.0f}% above baseline) "
                + (f"means stock runs out in {days_curr} days." if days_curr else ".")
            )
    else:  # DOWN
        if spike_type == "DECLINING":
            action = (
                f"Monitor {name} for 5 more days — consumption dropped "
                f"{magnitude:.0f}% below baseline but is recovering."
            )
        else:
            action = (
                f"Verify {name} is not being substituted before ordering — "
                f"consumption is {magnitude:.0f}% below baseline "
                f"({recent_avg:.1f} vs {base_avg:.1f} u/day); current SOH may exceed actual demand."
            )
    return action, False


def _rule_based(
    name: str, dos: float, qty: int, lead_time: int, gap: int,
    patients: int, priority: str, cost: Optional[float], trend: str,
) -> str:
    cost_str = f" (~KES {cost:,.0f})" if cost else ""
    if dos <= 0:
        s1 = f"{name} is currently stocked out."
    elif gap > 0:
        s1 = (
            f"{name} has {dos:.0f}d of stock against an estimated {lead_time}-day lead time "
            f"— a {gap}-day gap is expected if ordered today."
        )
    else:
        s1 = (
            f"{name} has {dos:.0f}d of stock remaining and should be ordered "
            f"this week to maintain cover."
        )

    s2 = f"Order {qty:,} units{cost_str} to cover the next {30 + lead_time} days including lead time."

    extras = []
    if patients > 0:
        extras.append(
            f"{patients:,} patient{'s' if patients != 1 else ''} at the facility "
            f"may be affected by a stockout."
        )
    if trend == "UP":
        extras.append("Upward demand trend — consider a larger buffer if budget allows.")
    if priority == "CRITICAL":
        extras.append("This is a critical-priority drug — escalate procurement immediately.")

    return " ".join(filter(None, [s1, s2] + extras[:1]))
