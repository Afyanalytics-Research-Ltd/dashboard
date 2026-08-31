"""Record-level data-quality gate for the consumption stream.

Some dispensing rows are not consumption at all — they are inventory
adjustments / write-offs that zero out an (often erroneous) stock figure. The
clearest case at SPH: a single "dispense" of 994,742 units of Cebactum, flagged
a stockout dispense, that took soh_before 994,742 → soh_after 0. Left in, one
such row dominates the item's fitted demand and inflates its order quantity by
orders of magnitude.

These are identified by their own transaction signature — a stockout-flagged
dispense that empties the stock (``soh_after <= 0``), is essentially the whole
``soh_before``, and is large — not by a statistical outlier cut (which would
also drop legitimate large bulk store-issues, the real v1 measurement grain).
Dropping is RECORD-level, so the item's genuine dispenses are kept.

Legitimate bulk store-issues (e.g. Metronidazole issued 6,000 with stock left
over) do NOT match and are retained — that inflation, where present, is a
measurement-grain matter for v2 recalibration, not a data error.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd

#: A write-off is "large": below this a zeroing dispense is an ordinary
#: end-of-stock issue, not a data error. Documented operational threshold.
_MIN_WRITEOFF_QTY = 2000.0


def drop_adjustments(consumption: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split consumption into (kept, dropped_adjustments).

    Needs ``is_stockout_dispense``, ``soh_before``, ``soh_after`` and
    ``quantity``; rows lacking them (e.g. the v2 per-patient stream) are never
    dropped. Returns the cleaned frame and the dropped rows (for the audit log).
    """
    empty = consumption.iloc[0:0]
    need = {"is_stockout_dispense", "soh_before", "soh_after", "quantity"}
    if consumption is None or consumption.empty or not need <= set(consumption.columns):
        return consumption, empty

    q = pd.to_numeric(consumption["quantity"], errors="coerce")
    sb = pd.to_numeric(consumption["soh_before"], errors="coerce")
    sa = pd.to_numeric(consumption["soh_after"], errors="coerce")
    stockout = consumption["is_stockout_dispense"].fillna(False).astype(bool)

    # write-off signature: stockout dispense that empties the stock, ≈ the whole
    # soh_before, and large.
    mask = stockout & (sa <= 0) & (q >= 0.9 * sb) & (q > _MIN_WRITEOFF_QTY)
    mask = mask.fillna(False)
    return consumption[~mask].copy(), consumption[mask].copy()
