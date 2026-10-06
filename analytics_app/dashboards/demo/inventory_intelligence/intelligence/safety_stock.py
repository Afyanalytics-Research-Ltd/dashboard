"""
Safety stock and reorder calculations.
Pure functions — no side effects, no Streamlit dependencies.
"""

from __future__ import annotations

import math

from intelligence.config import (
    SERVICE_LEVEL_Z,
    DEFAULT_SERVICE_LEVEL,
    DEFAULT_LEAD_TIME_DAYS,
    DEFAULT_ORDER_COST_KES,
    DEFAULT_HOLDING_RATE,
)


def safety_stock(
    demand_std: float,
    lead_time_mean: float,
    lead_time_std: float = 0.0,
    avg_daily_demand: float = 0.0,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """
    Safety stock using the combined demand + lead-time variability formula.

    SS = Z * sqrt(LT_mean * σ_demand² + avg_demand² * σ_LT²)

    Falls back to simpler SS = Z * σ_demand * sqrt(LT_mean) when lead-time
    variability is unknown (lt_std = 0).
    """
    z = SERVICE_LEVEL_Z.get(service_level, 1.645)
    lt = max(1.0, lead_time_mean)

    if lead_time_std > 0 and avg_daily_demand > 0:
        combined_var = lt * demand_std**2 + avg_daily_demand**2 * lead_time_std**2
        return max(0.0, z * math.sqrt(combined_var))

    return max(0.0, z * demand_std * math.sqrt(lt))


def reorder_point(
    avg_daily_demand: float,
    lead_time_mean: float,
    demand_std: float,
    lead_time_std: float = 0.0,
    service_level: float = DEFAULT_SERVICE_LEVEL,
) -> float:
    """
    Reorder point = demand during lead time + safety stock.
    ROP = avg_demand * LT_mean + SS
    """
    ss = safety_stock(demand_std, lead_time_mean, lead_time_std, avg_daily_demand, service_level)
    return max(0.0, avg_daily_demand * max(1.0, lead_time_mean) + ss)


def eoq(
    avg_daily_demand: float,
    unit_cost: float = 100.0,
    order_cost: float = DEFAULT_ORDER_COST_KES,
    holding_rate: float = DEFAULT_HOLDING_RATE,
) -> float:
    """
    Economic Order Quantity: sqrt(2 * D * S / H)
    D = annual demand, S = order cost, H = annual holding cost per unit.
    Returns 0 if inputs are invalid.
    """
    annual_demand = avg_daily_demand * 365
    holding_cost = unit_cost * holding_rate
    if annual_demand <= 0 or holding_cost <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * order_cost / holding_cost)


def recommended_order_quantity(
    current_soh: float,
    rop: float,
    avg_daily_demand: float,
    lead_time_mean: float = DEFAULT_LEAD_TIME_DAYS,
    target_cover_days: int = 30,
    eoq_qty: float = 0.0,
) -> float:
    """
    How much to order.

    Quantity = units needed to reach (target_cover_days + lead_time) days of cover
               minus current SOH, rounded up.
    If EOQ is provided and larger, use EOQ (avoids under-ordering for economics).
    """
    if avg_daily_demand <= 0:
        return 0.0

    cover_needed = (target_cover_days + lead_time_mean) * avg_daily_demand
    base_qty = max(0.0, cover_needed - max(0.0, current_soh))

    if eoq_qty > 0:
        return max(base_qty, eoq_qty)
    return math.ceil(base_qty)


def days_of_stock(current_soh: float, avg_daily_demand: float) -> float | None:
    """Return remaining days of stock; None if demand is zero."""
    if avg_daily_demand <= 0:
        return None
    return max(0.0, current_soh / avg_daily_demand)
