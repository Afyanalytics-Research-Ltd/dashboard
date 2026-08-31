"""Stitched v1+v2 ingestion.

WHY this layer exists: SPH consumption lives in two systems with a verified
27-day seam between them (``utils.facility.SEAM_GAP``). v1 rows arrive on the
master-product spine already; v2 rows arrive keyed by ``(itcode, itname)``
and are mapped onto the same spine here — in Python, never via a SQL UNION
(handoff gotcha #1: TRY_CAST/UNION type reconciliation).

Item spine: ``item_key`` = master_product_id (v1
ObjectId, or the v2 code itself for post-migration products), resolved
through the bridge file with per-row ``bridge_method`` provenance:

- ``bridge_itcode`` / ``bridge_name`` — resolved via ``sph_source_bridge.csv``
- ``name_exact``  — exact normalized-name match against the catalog
  (fallback when the bridge is absent or misses the row)
- ``unbridged``   — kept on its own key, visible for the mapping worklist

Every loader accepts either a live Snowflake connection or ``offline_dir``
pointing at cached parquet extracts — offline mode requires no warehouse, no
connector, no network.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..utils.facility import FACILITIES
from ..utils.snowflake_conn import run_query
from . import queries

#: Bridge produced by the Canonical Product Harmonisation work:
#: columns source_system, source_catalog, source_product_id, source_name,
#: name_normalised, master_product_id, link_type.
DEFAULT_BRIDGE_CSV = Path(
    r"C:\Users\TechConvenience\OneDrive - Refrontier Group\Desktop\AFYA"
    r"\Cannonical Product Harmonisation\sph_source_bridge.csv"
)

#: Offline parquet filenames each loader looks for inside ``offline_dir``.
OFFLINE_FILES = {
    "consumption": "consumption.parquet",
    "receipts": "receipts.parquet",
    "stock": "stock.parquet",
    "catalog": "catalog.parquet",
    "procedures": "procedures.parquet",
}

#: Column contract of :func:`load_consumption`.
CONSUMPTION_COLUMNS = [
    "item_key", "canonical_name", "inn_name", "therapeutic_class",
    "product_category", "dispensed_at", "quantity", "unit_price",
    "source_system", "soh_before", "patient_id", "store", "batch",
]

_WS_RE = re.compile(r"\s+")


def normalize_name(name: Any) -> Optional[str]:
    """UPPER + trim + collapse internal whitespace — the exact-name matching
    key used by both the bridge's ``name_normalised`` column and the
    ``name_exact`` fallback."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = _WS_RE.sub(" ", str(name)).strip().upper()
    return text or None


def _require_source(conn: Any, offline_dir: Optional[Path | str]) -> None:
    if conn is None and offline_dir is None:
        raise ValueError("Provide either a live Snowflake connection or offline_dir.")


def _read_offline(offline_dir: Path | str, kind: str) -> pd.DataFrame:
    path = Path(offline_dir) / OFFLINE_FILES[kind]
    if not path.is_file():
        raise FileNotFoundError(f"Offline extract missing: {path}")
    return pd.read_parquet(path)


# ── Bridge handling ───────────────────────────────────────────────────────────

def _load_bridge(bridge_csv: Optional[Path | str]) -> Optional[pd.DataFrame]:
    """Load the id bridge, filtered to PHARMREQUESTS rows. None when absent
    (callers then fall back to name matching — degraded but honest)."""
    if bridge_csv is None:
        return None
    path = Path(bridge_csv)
    if not path.is_file():
        return None
    bridge = pd.read_csv(path, dtype=str)
    bridge.columns = [c.strip().lower() for c in bridge.columns]
    if "source_catalog" in bridge.columns:
        bridge = bridge[bridge["source_catalog"].str.strip().str.upper() == "PHARMREQUESTS"]
    return bridge.reset_index(drop=True)


def _bridge_v2_items(
    v2: pd.DataFrame,
    bridge: Optional[pd.DataFrame],
    catalog: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach ``item_key`` + ``bridge_method`` to v2 PHARMREQUESTS rows.

    Resolution order: bridge-by-itcode → bridge-by-normalized-name →
    exact normalized-name match against the catalog → own key (unbridged).
    Unbridged rows keep ``itcode`` (matching the bridge's ``new_master``
    convention where a v2 code becomes its own master id) or, lacking one,
    their normalized name — so nothing is silently dropped.
    """
    v2 = v2.copy()
    v2["_norm_name"] = v2["itname"].map(normalize_name)
    v2["_itcode"] = v2["itcode"].map(lambda x: str(x).strip() if pd.notna(x) else None)

    code_map: dict[str, str] = {}
    name_map: dict[str, str] = {}
    if bridge is not None and not bridge.empty:
        for _, row in bridge.iterrows():
            master = str(row.get("master_product_id") or "").strip()
            if not master:
                continue
            code = str(row.get("source_product_id") or "").strip()
            if code and code.lower() != "nan":
                code_map.setdefault(code, master)
            norm = normalize_name(row.get("name_normalised") or row.get("source_name"))
            if norm:
                name_map.setdefault(norm, master)

    catalog_name_map: dict[str, str] = {}
    if catalog is not None and not catalog.empty:
        for name_col in ("product_name", "canonical_name"):
            if name_col in catalog.columns:
                for _, row in catalog.iterrows():
                    norm = normalize_name(row[name_col])
                    if norm and pd.notna(row["item_key"]):
                        catalog_name_map.setdefault(norm, str(row["item_key"]))

    def resolve(itcode: Optional[str], norm: Optional[str]) -> tuple[str, str]:
        if itcode and itcode in code_map:
            return code_map[itcode], "bridge_itcode"
        if norm and norm in name_map:
            return name_map[norm], "bridge_name"
        if norm and norm in catalog_name_map:
            return catalog_name_map[norm], "name_exact"
        # Own key: v2 code (new_master convention) else normalized name.
        return (itcode or norm or "UNKNOWN"), "unbridged"

    resolved = [resolve(c, n) for c, n in zip(v2["_itcode"], v2["_norm_name"])]
    v2["item_key"] = [r[0] for r in resolved]
    v2["bridge_method"] = [r[1] for r in resolved]
    return v2.drop(columns=["_norm_name", "_itcode"])


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_era(conn, era: str, facility: str, bridge_csv) -> pd.DataFrame:
    keep = CONSUMPTION_COLUMNS + ["bridge_method"]
    if era == "v1":
        v1 = run_query(conn, queries.consumption_v1_sql(), {"facility": facility})
        v1["item_key"] = v1["item_key"].astype(str)
        v1["bridge_method"] = "native"
        # Drop stock-adjustment / write-off records (not consumption) before the
        # column prune removes the signals they are detected by.
        from .quality import drop_adjustments
        v1, dropped = drop_adjustments(v1)
        if len(dropped):
            import logging
            logging.getLogger(__name__).info(
                "data-quality: dropped %d write-off/adjustment record(s) from v1 "
                "consumption (not real demand)", len(dropped))
        return v1.reindex(columns=keep)
    if era == "v2":
        catalog = load_catalog(conn, facility=facility)
        v2 = _bridge_v2_items(
            run_query(conn, queries.consumption_v2_sql()), _load_bridge(bridge_csv), catalog
        )
        attrs = catalog.drop_duplicates("item_key")[
            ["item_key", "canonical_name", "inn_name", "therapeutic_class", "product_category"]
        ]
        v2 = v2.merge(attrs, on="item_key", how="left")
        v2["canonical_name"] = v2["canonical_name"].fillna(v2["itname"])
        v2["product_category"] = v2["product_category"].fillna("uncategorized")
        v2["source_system"] = "v2"
        v2["soh_before"] = pd.NA
        return v2.reindex(columns=keep)
    raise ValueError(f"Unknown data era {era!r}")


def _finalize_consumption(frames: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True)
    out["dispensed_at"] = pd.to_datetime(out["dispensed_at"])
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")
    out["soh_before"] = pd.to_numeric(out["soh_before"], errors="coerce")
    return out.sort_values(["item_key", "dispensed_at"]).reset_index(drop=True)


def load_consumption(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
    bridge_csv: Optional[Path | str] = DEFAULT_BRIDGE_CSV,
    eras: Optional[tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Consumption on the master item spine for the facility's training eras.

    ``eras`` defaults to ``FacilityMeta.training_eras``. v1 rows are native to
    the spine; v2 rows are bridged and carry no ``soh_before``.
    """
    if offline_dir is not None:
        return _read_offline(offline_dir, "consumption")
    _require_source(conn, offline_dir)
    eras = eras or FACILITIES[facility].training_eras
    return _finalize_consumption([_load_era(conn, e, facility, bridge_csv) for e in eras])


def load_validation_actuals(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
    bridge_csv: Optional[Path | str] = DEFAULT_BRIDGE_CSV,
) -> pd.DataFrame:
    """Held-out consumption (the validation eras) used to score forecasts made
    from the training eras. Same schema as :func:`load_consumption`."""
    if offline_dir is not None:
        return _read_offline(offline_dir, "validation")
    _require_source(conn, offline_dir)
    eras = FACILITIES[facility].validation_eras
    if not eras:
        return pd.DataFrame(columns=CONSUMPTION_COLUMNS + ["bridge_method"])
    return _finalize_consumption([_load_era(conn, e, facility, bridge_csv) for e in eras])


def load_receipts(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Receipt events: procurement RECEIPT lines + v1 ledger IN movements.

    Both streams are fetched as separate single-source statements and
    concatenated here (never a SQL UNION — gotcha #1). Feeds the
    replenishment-interval model; ``receipt_source`` lets it weigh
    invoice receipts and ledger movements separately if they disagree.
    Columns: item_key, received_at, quantity, supplier, unit_cost,
    batch_expiry, product_category, receipt_source.
    """
    if offline_dir is not None:
        return _read_offline(offline_dir, "receipts")
    _require_source(conn, offline_dir)

    proc = run_query(conn, queries.receipts_procurement_sql(), {"facility": facility})
    ledger = run_query(conn, queries.receipts_ledger_in_sql())
    out = pd.concat([proc, ledger], ignore_index=True)
    out["item_key"] = out["item_key"].astype(str)
    out["received_at"] = pd.to_datetime(out["received_at"])
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["unit_cost"] = pd.to_numeric(out["unit_cost"], errors="coerce")
    return out.sort_values(["item_key", "received_at"]).reset_index(drop=True)


def load_stock_snapshot(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Latest SOH per item, stamped with its validity date.

    ``soh_as_of`` comes from ``FacilityMeta.stock_as_of`` — at SPH the
    snapshot froze with the v1 ledger, so every SOH-dependent output must
    surface this date. ``store`` is NULL (the reporting view carries
    no store grain). Columns: item_key, soh, soh_raw, soh_as_of, store,
    plus taxonomy attrs.
    """
    if offline_dir is not None:
        return _read_offline(offline_dir, "stock")
    _require_source(conn, offline_dir)

    out = run_query(conn, queries.stock_status_sql(), {"facility": facility})
    out["item_key"] = out["item_key"].astype(str)
    out["soh"] = pd.to_numeric(out["soh"], errors="coerce")
    out["soh_raw"] = pd.to_numeric(out["soh_raw"], errors="coerce")
    if "days_since_last_dispense" in out.columns:
        out["days_since_last_dispense"] = pd.to_numeric(
            out["days_since_last_dispense"], errors="coerce"
        )
    fac = FACILITIES.get(facility)
    out["soh_as_of"] = pd.Timestamp(fac.stock_as_of) if fac and fac.stock_as_of else pd.NaT
    out["store"] = pd.NA
    return out.reset_index(drop=True)


def load_batches(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Batch/expiry lines for the expiry-risk watchlist (expiry_risk).

    Procurement-grain and sparse at SPH (no batch-level stock ledger); returns
    an empty frame rather than raising when the source or expiry data is
    absent. Columns: item_key, batch, qty, expiry.
    """
    cols = ["item_key", "batch", "qty", "expiry"]
    if offline_dir is not None:
        try:
            return _read_offline(offline_dir, "batches")
        except Exception:
            return pd.DataFrame(columns=cols)
    if conn is None:
        return pd.DataFrame(columns=cols)
    try:
        out = run_query(conn, queries.batches_sql(), {"facility": facility})
    except Exception:
        return pd.DataFrame(columns=cols)
    if out is None or out.empty:
        return pd.DataFrame(columns=cols)
    out["item_key"] = out["item_key"].astype(str)
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce")
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    return out.dropna(subset=["expiry"]).reset_index(drop=True)


def load_item_costs(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Per-item acquisition cost (median procurement unit price) — the COST
    basis for inventory valuation. Columns: item_key, unit_cost. Degrades to an
    empty frame when the source is unavailable."""
    cols = ["item_key", "unit_cost"]
    if offline_dir is not None:
        try:
            return _read_offline(offline_dir, "item_costs")
        except Exception:
            return pd.DataFrame(columns=cols)
    if conn is None:
        return pd.DataFrame(columns=cols)
    try:
        out = run_query(conn, queries.item_cost_sql(), {"facility": facility})
    except Exception:
        return pd.DataFrame(columns=cols)
    if out is None or out.empty:
        return pd.DataFrame(columns=cols)
    out["item_key"] = out["item_key"].astype(str)
    out["unit_cost"] = pd.to_numeric(out["unit_cost"], errors="coerce")
    return out.dropna(subset=["unit_cost"]).reset_index(drop=True)


def load_catalog(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
    bridge_csv: Optional[Path | str] = DEFAULT_BRIDGE_CSV,
) -> pd.DataFrame:
    """Product master: taxonomy attributes + raw names from all sources.

    ``source_names`` aggregates every raw source string the bridge knows for
    the item. ``canonical_product_id`` is carried
    for reference only and must never be used as a join key (gotcha #4).
    """
    if offline_dir is not None:
        return _read_offline(offline_dir, "catalog")
    _require_source(conn, offline_dir)

    out = run_query(conn, queries.catalog_sql(), {"facility": facility})
    out["item_key"] = out["item_key"].astype(str)

    # Sentinel→NULL (cleaning contract): the taxonomy writes the literal
    # string 'unspecified' where strength/form are unknown (2,092/2,631 SPH
    # rows for form). Left as-is it masquerades as a real value. Missing means
    # missing.
    for col in ("strength_canonical", "form_canonical"):
        if col in out.columns:
            sentinel = out[col].astype("string").str.strip().str.lower() == "unspecified"
            out.loc[sentinel, col] = pd.NA

    bridge = _load_bridge(bridge_csv)
    if bridge is not None and not bridge.empty and "source_name" in bridge.columns:
        names = (
            bridge.assign(master_product_id=bridge["master_product_id"].astype(str).str.strip())
            .dropna(subset=["source_name"])
            .groupby("master_product_id")["source_name"]
            .agg(lambda s: sorted(set(s.astype(str).str.strip())))
            .rename("source_names")
        )
        out = out.merge(names, left_on="item_key", right_index=True, how="left")
    if "source_names" not in out.columns:
        out["source_names"] = None
    return out.reset_index(drop=True)


def load_procedures(
    conn: Any = None,
    facility: str = "SPH",
    offline_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Theatre requests (row level) with an ISO-week stamp.

    Weekly counts (see ``features.weekly_procedure_counts``) feed
    procedure-driven demand covariates.
    """
    if offline_dir is not None:
        return _read_offline(offline_dir, "procedures")
    _require_source(conn, offline_dir)

    out = run_query(conn, queries.theatre_procedures_sql())
    out["requested_at"] = pd.to_datetime(out["requested_at"])
    out["week_start"] = out["requested_at"].dt.to_period("W-SUN").dt.start_time
    return out.reset_index(drop=True)
