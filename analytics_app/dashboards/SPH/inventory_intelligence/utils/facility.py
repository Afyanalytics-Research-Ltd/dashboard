"""Facility registry and data-era configuration.

A facility's data arrives in eras. St. Peter's has two: the v1 EMR ledger
(2022-06 → 2025-01-23) carries stock-on-hand, procurement and dispensing; the
v2 system (2025-02 onward) currently carries dispensing only. Forecasting and
analytics run on ``training_eras``; the remaining ``validation_eras`` are held
out to measure forecast accuracy against later real consumption.

When v2 grows a stock and procurement feed, move ``"v2"`` into
``training_eras``, point ``stock_era`` at it, set ``is_live=True`` and advance
``as_of`` — no other code changes are required.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class FacilityMeta:
    """Immutable facility descriptor.

    ``schema`` is the ``source_schema`` discriminator used across the
    ``HOSPITALS`` gold layer. ``as_of`` is the analysis date: the last day of
    training data and the date the stock snapshot is valid for.
    """

    schema: str
    as_of: date
    go_live_date: date
    training_eras: tuple[str, ...] = ("v1",)
    validation_eras: tuple[str, ...] = ("v2",)
    stock_era: str = "v1"
    is_live: bool = False

    @property
    def stock_as_of(self) -> date:
        return self.as_of


FACILITIES: dict[str, FacilityMeta] = {
    "SPH": FacilityMeta(
        schema="SPH",
        as_of=date(2025, 1, 23),
        go_live_date=date(2022, 6, 2),
    ),
}


def sql_ref_date(fac: FacilityMeta) -> str:
    """SQL date expression anchoring time-windowed queries — the analysis date
    for a historical facility, ``CURRENT_DATE`` once a live feed is in scope."""
    if fac.is_live:
        return "CURRENT_DATE"
    return f"'{fac.as_of.isoformat()}'::DATE"


def sql_go_live_filter(fac: FacilityMeta, date_col: str = "dispensed_at") -> str:
    """AND-clause excluding pre-go-live test records."""
    return f"AND {date_col} >= '{fac.go_live_date.isoformat()}'"
