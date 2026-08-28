"""All data loading for the SPH dashboard.

Three sources, in order of preference:

1. **Engine analytics tables** — parquet (CSV fallback) written by
   ``pipeline/run_analytics.py`` into ``output/analytics/``. These are the
   only source of model numbers; the dashboard never recomputes them.
2. **Fitted-artifact metadata** — ``output/artifacts/*.meta.json``.
3. **Live Snowflake** (read-only, optional) — item catalog for display
   names/categories, procurement spend views, and the raw consumption stream
   for the demand-history explorer. Every query goes through the engine's
   ``data.queries`` / ``utils.snowflake_conn`` modules and anchors lookbacks
   with ``utils.facility.sql_ref_date`` (never a raw ``CURRENT_DATE`` literal
   in this module).

Degraded modes are first-class: every loader returns ``None`` (or an empty
frame) instead of raising, and pages render an honest banner for whichever
source is missing.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

from inventory_intelligence.config import BusinessInputs
from inventory_intelligence.data import queries
from inventory_intelligence.utils.facility import FACILITIES, FacilityMeta, sql_ref_date

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PACKAGE_ROOT / "output"
ANALYTICS_DIR = OUTPUT_DIR / "analytics"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

#: The analytics tables the engine emits.
ENGINE_TABLES = [
    "demand_forecast", "abc_xyz", "burn_rate", "replenishment",
    "inventory_policy", "stockout_risk", "cost_risk", "anomalies",
    "size_mix", "expiry_risk", "inventory_health", "forecast_validation",
    "run_metadata",
]

#: Optional directory of offline parquet extracts (consumption.parquet, ...)
#: for running the demand explorer without Snowflake.
OFFLINE_DIR_ENV = "SPH_DASHBOARD_OFFLINE_DIR"


# ── Facility registry ─────────────────────────────────────────────────────────

def facility_codes() -> list[str]:
    return list(FACILITIES.keys())


def facility(code: str = "SPH") -> FacilityMeta:
    return FACILITIES[code]


def selected_facility() -> str:
    """Facility chosen in the sidebar (the entry sets this; SPH default)."""
    return st.session_state.get("facility", "SPH")


# ── Engine analytics tables ───────────────────────────────────────────────────

def _table_paths(name: str) -> tuple[Path, Path]:
    return ANALYTICS_DIR / f"{name}.parquet", ANALYTICS_DIR / f"{name}.csv"


def _table_mtime(name: str) -> float:
    """Latest mtime across the table's files — cache key so a fresh pipeline
    run invalidates the cached frame file-for-file."""
    return max((p.stat().st_mtime for p in _table_paths(name) if p.is_file()), default=0.0)


@st.cache_data(show_spinner=False)
def _load_table_cached(name: str, _mtime: float) -> Optional[pd.DataFrame]:
    parquet, csv = _table_paths(name)
    if parquet.is_file():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if csv.is_file():
        try:
            return pd.read_csv(csv)
        except Exception:
            return None
    return None


def load_table(name: str) -> Optional[pd.DataFrame]:
    """One engine analytics table, or None when it hasn't been produced.

    Degrades gracefully for an unrecognised name (returns None) rather than
    raising — consistent with this module's contract that every loader returns
    None/empty so a page shows an honest empty state instead of crashing.
    """
    if name not in ENGINE_TABLES:
        return None
    return _load_table_cached(name, _table_mtime(name))


def analytics_available() -> bool:
    """True when the core tables exist (pipeline has produced output)."""
    core = ("stockout_risk", "demand_forecast", "cost_risk")
    return all(load_table(t) is not None for t in core)


def analytics_written_at() -> Optional[datetime]:
    """Timestamp of the most recent analytics file — the run's freshness."""
    mtimes = [_table_mtime(t) for t in ENGINE_TABLES]
    newest = max(mtimes, default=0.0)
    return datetime.fromtimestamp(newest) if newest > 0 else None


def run_metadata() -> dict[str, str]:
    """``run_metadata`` table as a {key: value} dict (strings)."""
    df = load_table("run_metadata")
    if df is None or df.empty or not {"key", "value"} <= set(df.columns):
        return {}
    return {str(k): str(v) for k, v in zip(df["key"], df["value"])}


def run_metadata_json(key: str) -> Any:
    """Parse a JSON-valued run_metadata entry (returns None when absent)."""
    raw = run_metadata().get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def placeholder_inputs() -> list[str]:
    """Business inputs still on placeholder provenance for the *current run*.

    Prefers the run's own record (run_metadata) so the banner describes the
    numbers actually on screen; falls back to the live config defaults.
    """
    recorded = run_metadata_json("business_input_placeholders")
    if isinstance(recorded, list):
        return [str(x) for x in recorded]
    return BusinessInputs().placeholders()


def pretty_date(value) -> str:
    """ISO date string → '23 Jan 2025'."""
    try:
        return pd.to_datetime(value).strftime("%d %b %Y")
    except Exception:
        return str(value)


def soh_as_of() -> Optional[str]:
    """SOH snapshot validity date — from the stockout table itself when
    present (authoritative: it is what the numbers were computed with),
    else the facility registry."""
    df = load_table("stockout_risk")
    if df is not None and "soh_as_of" in df.columns and not df.empty:
        vals = df["soh_as_of"].dropna().astype(str).unique()
        if len(vals):
            return sorted(vals)[-1][:10]
    meta = facility(selected_facility())
    return meta.stock_as_of.isoformat() if meta.stock_as_of else None


def artifact_metadata() -> dict[str, dict]:
    """{artifact_name: meta dict} from output/artifacts/*.meta.json."""
    out: dict[str, dict] = {}
    if not ARTIFACTS_DIR.is_dir():
        return out
    for path in sorted(ARTIFACTS_DIR.glob("*.meta.json")):
        try:
            out[path.name.replace(".meta.json", "")] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


# ── Live Snowflake (optional) ─────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _connection_state() -> dict:
    """Singleton connection attempt. {'conn': conn|None, 'error': str|None}.

    A failed attempt is cached too (no reconnect storm on every rerun);
    clear via st.cache_resource.clear() from the sidebar retry button.
    """
    try:
        from inventory_intelligence.utils.snowflake_conn import get_connection

        return {"conn": get_connection(), "error": None}
    except Exception as exc:  # missing key file, network, package, MFA...
        return {"conn": None, "error": f"{type(exc).__name__}: {exc}"}


def snowflake_connection():
    return _connection_state()["conn"]


def snowflake_error() -> Optional[str]:
    return _connection_state()["error"]


def snowflake_available() -> bool:
    return snowflake_connection() is not None


def retry_snowflake() -> None:
    _connection_state.clear()


def _run_query(sql: str, params: Optional[dict] = None) -> Optional[pd.DataFrame]:
    conn = snowflake_connection()
    if conn is None:
        return None
    try:
        from inventory_intelligence.utils.snowflake_conn import run_query

        return run_query(conn, sql, params)
    except Exception:
        return None


def offline_dir() -> Optional[Path]:
    raw = os.getenv(OFFLINE_DIR_ENV, "").strip()
    if raw and Path(raw).is_dir():
        return Path(raw)
    return None


# ── Item catalog / display names ──────────────────────────────────────────────

def _denull_names(series: pd.Series) -> pd.Series:
    """Strip and treat null-like tokens ('na', 'n/a', 'n-a', 'none', 'null',
    'nan', 'nil', 'n', '-', …) as missing, so they fall through to the next
    name source instead of surfacing as a literal 'na' on screen."""
    s = series.astype("string").str.strip()
    norm = s.str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    nullish = {"", "na", "n", "none", "null", "nan", "nil", "x", "unspecified", "unknown"}
    return s.where(~norm.isin(nullish), pd.NA)


def _analytics_item_keys() -> set[str]:
    """Every item_key present across the engine's analytics tables — the set
    the dashboard must be able to name."""
    keys: set[str] = set()
    for table in ("stockout_risk", "abc_xyz", "demand_forecast",
                  "inventory_policy", "cost_risk", "anomalies", "replenishment"):
        frame = load_table(table)
        if frame is not None and "item_key" in frame.columns:
            keys.update(frame["item_key"].dropna().astype(str))
    return keys


@st.cache_data(ttl=3600, show_spinner=False)
def _bridge_name_map() -> dict[str, str]:
    """item_key (master_product_id) → raw source name from the product
    harmonisation bridge.

    The canonical taxonomy only names ~91% of items; orthopedic consumables
    (Knee Support, Axillary Crutches, …) live in SALEITEMS / INVENTORYITEMS
    and never entered the pharma-centric taxonomy, but the bridge carries a
    ``source_name`` for them keyed by ``master_product_id`` (the analytics
    item_key). Returns {} when the bridge file isn't reachable — degraded but
    honest. Override the path with ``SPH_BRIDGE_CSV``.
    """
    try:
        from inventory_intelligence.data.ingestion import DEFAULT_BRIDGE_CSV

        path = Path(os.getenv("SPH_BRIDGE_CSV", str(DEFAULT_BRIDGE_CSV)))
        if not path.is_file():
            return {}
        b = pd.read_csv(path, dtype=str)
        b.columns = [c.strip().lower() for c in b.columns]
        if not {"master_product_id", "source_name"} <= set(b.columns):
            return {}
        b["item_key"] = b["master_product_id"].astype("string").str.strip()
        b["name"] = b["source_name"].astype("string").str.strip().replace("", pd.NA)
        b = b.dropna(subset=["item_key", "name"]).drop_duplicates("item_key")
        return dict(zip(b["item_key"], b["name"].astype(str)))
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _saleitems_name_map() -> dict[str, str]:
    """item_key → sales-item ``service_name`` from the v1 product master.

    The analytics item_key (``FACT_DISPENSING.product_id``) matches
    ``ORTHO_SALEITEMS.inventory_item_id`` (NOT the table's own ``id``), and
    ``service_name`` is the product name staff actually use. This is the most
    complete single name source (~99% of items, including the orthopedic
    consumables the pharma-centric taxonomy omits). {} when Snowflake is off.
    """
    sql = """
        SELECT CAST(inventory_item_id AS VARCHAR)  AS item_key,
               COALESCE(service_name, description)  AS name
        FROM HOSPITALS.ORTHOPEDIC_CLEAN.ORTHO_SALEITEMS
        WHERE inventory_item_id IS NOT NULL
    """
    df = _run_query(sql)
    if df is None or df.empty or "item_key" not in df.columns:
        return {}
    df = df.dropna(subset=["item_key"]).copy()
    df["item_key"] = df["item_key"].astype(str)
    df["name"] = df["name"].astype("string").str.strip().replace("", pd.NA)
    df = df.dropna(subset=["name"]).drop_duplicates("item_key")
    return dict(zip(df["item_key"], df["name"].astype(str)))


@st.cache_data(ttl=3600, show_spinner=False)
def _procurement_name_map(facility_code: str) -> dict[str, str]:
    """item_key → procurement item name, for items a catalog/master misses.
    Keyed by ``item_id`` (= the analytics item_key). {} when Snowflake is off."""
    sql = f"""
        SELECT CAST(p.item_id AS VARCHAR)            AS item_key,
               COALESCE(p.canonical_name, p.item_name) AS name
        FROM {queries.FACT_PROCUREMENT} p
        WHERE p.source_schema = %(facility)s
    """
    df = _run_query(sql, {"facility": facility_code})
    if df is None or df.empty or "item_key" not in df.columns:
        return {}
    df = df.dropna(subset=["item_key"]).copy()
    df["item_key"] = df["item_key"].astype(str)
    df["name"] = df["name"].astype("string").str.strip().replace("", pd.NA)
    df = df.dropna(subset=["name"]).drop_duplicates("item_key")
    return dict(zip(df["item_key"], df["name"].astype(str)))


@st.cache_data(ttl=3600, show_spinner=False)
def item_lookup(facility_code: str) -> pd.DataFrame:
    """item_key → display_name, product_category, therapeutic_class.

    Names are resolved from layered, correctly-keyed sources so an item without
    a canonical-taxonomy entry still shows a real name instead of its ObjectId:

        canonical taxonomy  →  harmonisation bridge  →  procurement  →  id

    The taxonomy also supplies the ``product_category`` dimension used by the
    sidebar (never a query filter). Every source degrades
    to empty independently, so the dashboard still names as many items as it
    can when Snowflake or the bridge file is unavailable.
    """
    # 1) Canonical taxonomy — best-quality names + category / therapeutic class.
    tax = _run_query(queries.catalog_sql(), {"facility": facility_code})
    have_tax = tax is not None and not tax.empty and "item_key" in tax.columns
    if have_tax:
        tax = tax.drop_duplicates("item_key").copy()
        tax["item_key"] = tax["item_key"].astype(str)
        tname = None
        for col in ("canonical_name", "product_name", "inn_name"):
            if col in tax.columns:
                cand = _denull_names(tax[col])
                tname = cand if tname is None else tname.fillna(cand)
        tax["_tax_name"] = tname if tname is not None else pd.Series(pd.NA, index=tax.index, dtype="string")
        tax["product_category"] = (
            tax["product_category"].astype("string").str.strip().replace("", pd.NA)
            if "product_category" in tax.columns else pd.Series(pd.NA, index=tax.index, dtype="string")
        )
        if "therapeutic_class" not in tax.columns:
            tax["therapeutic_class"] = pd.NA
        tax = tax[["item_key", "_tax_name", "product_category", "therapeutic_class"]]
    else:
        tax = pd.DataFrame(columns=["item_key", "_tax_name", "product_category", "therapeutic_class"])

    # 2) Fallback name sources for items outside the taxonomy, most complete
    #    first: the sales-item master, then the harmonisation bridge, then
    #    procurement. Each degrades to {} independently.
    saleitems = _saleitems_name_map()
    bridge = _bridge_name_map()
    procurement = _procurement_name_map(facility_code)

    # 3) Universe = taxonomy keys ∪ every item the analytics tables reference.
    keys = set(tax["item_key"]) | _analytics_item_keys()
    out = pd.DataFrame({"item_key": sorted(keys)}).merge(tax, on="item_key", how="left")

    # 4) Coalesce the display name across sources; id only as the last resort.
    #    Each source is de-nulled so a literal 'na'/'n'/'none' never wins.
    sale_name = _denull_names(out["item_key"].map(saleitems))
    bridge_name = _denull_names(out["item_key"].map(bridge))
    proc_name = _denull_names(out["item_key"].map(procurement))
    out["display_name"] = (
        _denull_names(out["_tax_name"])
        .fillna(sale_name).fillna(bridge_name).fillna(proc_name)
        .fillna(out["item_key"].astype("string")).astype(str)
    )
    offline_cat = "uncatalogued (catalog offline)" if not have_tax else "uncatalogued"
    out["product_category"] = out["product_category"].astype("string").fillna(offline_cat).astype(str)

    # 5) Disambiguate collided display names. Distinct SKUs — different brands of
    #    one molecule (e.g. Augmentin / Bactoclav / Amoxiclav → "AMOXICILLIN
    #    625MG TABLET") — would otherwise read as duplicates. There are no true
    #    duplicates (verified), so we make each distinguishable by appending its
    #    own raw/brand name, never by merging (a specific brand's stock is what
    #    goes dead or expires).
    dup_mask = out["display_name"].duplicated(keep=False)
    if bool(dup_mask.any()):
        brand = sale_name.fillna(bridge_name).fillna(proc_name)
        disambiguated = []
        for disp, b, key in zip(out.loc[dup_mask, "display_name"],
                                brand.loc[dup_mask], out.loc[dup_mask, "item_key"]):
            bb = "" if pd.isna(b) else str(b).strip()
            if bb and bb.lower() != str(disp).lower():
                disambiguated.append(f"{disp} · {bb}")
            else:
                disambiguated.append(f"{disp} · #{str(key)[-4:]}")
        out.loc[dup_mask, "display_name"] = disambiguated

    return out[["item_key", "display_name", "product_category", "therapeutic_class"]]


def with_names(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Left-join display names + category dimension onto an engine table."""
    if df is None or "item_key" not in getattr(df, "columns", []):
        return df
    lookup = item_lookup(selected_facility())
    merged = df.copy()
    merged["item_key"] = merged["item_key"].astype(str)
    return merged.merge(lookup, on="item_key", how="left").assign(
        display_name=lambda d: d["display_name"].fillna(d["item_key"]),
        product_category=lambda d: d["product_category"].fillna("uncatalogued (catalog offline)"),
    )


def category_options(df: Optional[pd.DataFrame] = None) -> list[str]:
    """Distinct categories present in the data — the sidebar dimension."""
    source = df if df is not None and "product_category" in getattr(df, "columns", []) \
        else item_lookup(selected_facility())
    return sorted(source["product_category"].dropna().astype(str).unique())


def apply_category_filter(df: Optional[pd.DataFrame], selected: list[str]) -> Optional[pd.DataFrame]:
    """Filter by the category *dimension* chosen in the UI (empty = all)."""
    if df is None or not selected or "product_category" not in df.columns:
        return df
    return df[df["product_category"].isin(selected)]


# ── Demand history (heavy; live or offline extracts) ──────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading consumption history…")
def demand_panel(facility_code: str) -> Optional[dict]:
    """Daily demand panel {'daily': df, 'meta': df} for the explorer page.

    Offline extracts dir (``SPH_DASHBOARD_OFFLINE_DIR``) wins; else live
    Snowflake through the engine's own ingestion + features modules (same
    censoring/masking semantics as the pipeline). None → degraded page.
    """
    try:
        from inventory_intelligence.data.features import build_demand_panel
        from inventory_intelligence.data.ingestion import load_consumption, load_stock_snapshot

        off = offline_dir()
        if off is not None:
            consumption = load_consumption(None, facility=facility_code, offline_dir=off)
            try:
                stock = load_stock_snapshot(None, facility=facility_code, offline_dir=off)
            except Exception:
                stock = None
        else:
            conn = snowflake_connection()
            if conn is None:
                return None
            consumption = load_consumption(conn, facility=facility_code)
            stock = load_stock_snapshot(conn, facility=facility_code)
        panel = build_demand_panel(consumption, stock=stock)
        return {"daily": panel.daily, "meta": panel.meta}
    except Exception:
        return None


# ── Procurement (live Snowflake views) ────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _excluded_scope(facility_code: str) -> set[str]:
    """Item keys that are OUT of clinical scope (non-medical: stationery,
    cleaning, furniture, patient-hotel). Same authority the analytics pipeline
    uses — the hospital-editable ``item_scope_review.csv`` drop-table when
    present, else the ``data.scope`` name classifier — so the Spending page
    excludes exactly what every other page already does."""
    from inventory_intelligence.data.scope import excluded_item_keys
    catalog = _run_query(queries.catalog_sql(), {"facility": facility_code})
    review = OUTPUT_DIR / "item_scope_review.csv"
    try:
        return {str(k) for k in excluded_item_keys(
            catalog, review_csv=str(review) if review.is_file() else None)}
    except Exception:
        return set()


def _drop_non_medical(df: Optional[pd.DataFrame], facility_code: str) -> Optional[pd.DataFrame]:
    """Remove non-medical rows from an item-grain procurement frame: by item key
    (the reviewed drop-table) and, defensively, by the name classifier so a
    procurement item absent from the catalog is still caught.

    Non-medical items have no clinical ``canonical_name`` (they're outside the
    taxonomy), so the classifier runs on the populated ``item_name`` — that's
    what catches cleaning/janitorial lines (UNGEROL, bin liners, perfume) the
    catalog drop-table doesn't list."""
    if df is None or df.empty or "item_key" not in df.columns:
        return df
    keep = ~df["item_key"].astype(str).isin(_excluded_scope(facility_code))
    canon = df.get("canonical_name")
    name = df.get("item_name")
    names = None
    if canon is not None and name is not None:
        names = canon.fillna(name)
    elif canon is not None:
        names = canon
    elif name is not None:
        names = name
    if names is not None:
        from inventory_intelligence.data.scope import is_non_medical
        keep &= ~names.fillna("").map(is_non_medical)
    return df[keep]


@st.cache_data(ttl=3600, show_spinner="Computing lead times…")
def item_lead_times(facility_code: str) -> dict:
    """Per-item median **real** lead time (days) — the gap from a purchase order
    to the matching goods-receipt, from paired ORDER/RECEIPT records. Blank for
    items with no matched pair (most purchasing bypasses POs, so coverage is
    partial). This is the honest order-to-delivery time, NOT the reorder interval.
    """
    sql = f"""
        SELECT item_id AS item_key, doc_type, supplier_id,
               CAST(doc_at AS TIMESTAMP_NTZ) AS doc_at
        FROM {queries.FACT_PROCUREMENT}
        WHERE source_schema = %(facility)s AND doc_type IN ('ORDER', 'RECEIPT')
          AND doc_at IS NOT NULL
    """
    df = _run_query(sql, {"facility": facility_code})
    if df is None or df.empty:
        return {}
    df.columns = [c.lower() for c in df.columns]
    df["doc_at"] = pd.to_datetime(df["doc_at"], errors="coerce")
    df = df.dropna(subset=["doc_at"])
    orders = df[df["doc_type"] == "ORDER"].rename(columns={"doc_at": "order_at"})
    recs = df[df["doc_type"] == "RECEIPT"]
    if orders.empty or recs.empty:
        return {}
    try:
        m = pd.merge_asof(
            recs.sort_values("doc_at"),
            orders[["item_key", "supplier_id", "order_at"]].sort_values("order_at"),
            left_on="doc_at", right_on="order_at",
            by=["item_key", "supplier_id"], direction="backward",
        )
    except Exception:
        return {}
    m["lead"] = (m["doc_at"] - m["order_at"]).dt.days
    m = m[(m["lead"] >= 0) & (m["lead"] <= 180)]   # sane order→delivery window
    if m.empty:
        return {}
    return {str(k): float(v) for k, v in m.groupby("item_key")["lead"].median().items()}


@st.cache_data(ttl=3600, show_spinner="Reading delivery sizes…")
def item_max_receipt(facility_code: str) -> dict:
    """Per-item largest single goods-receipt (units), summed per delivery
    document then maxed across deliveries. Lets the surplus view tell apart
    stock that piled up from one oversized delivery (a lot-sizing choice) from
    stock that accumulated across many normal-sized deliveries. Returns {} when
    Snowflake is off or no receipts exist — callers then simply don't claim a
    lot-size cause rather than guessing one.
    """
    sql = f"""
        SELECT item_id AS item_key, doc_id,
               CAST(SUM(quantity) AS FLOAT) AS doc_qty
        FROM {queries.FACT_PROCUREMENT}
        WHERE source_schema = %(facility)s AND doc_type = 'RECEIPT'
          AND quantity IS NOT NULL
        GROUP BY item_id, doc_id
    """
    df = _run_query(sql, {"facility": facility_code})
    if df is None or df.empty:
        return {}
    df.columns = [c.lower() for c in df.columns]
    df["doc_qty"] = pd.to_numeric(df["doc_qty"], errors="coerce")
    df = df.dropna(subset=["doc_qty"])
    if df.empty:
        return {}
    return {str(k): float(v)
            for k, v in df.groupby("item_key")["doc_qty"].max().items()}


#: Root-cause attribution for over-top-up surplus. Each cause is a single,
#: action-bearing label derived only from a model output (the demand trend) and
#: one observed procurement fact (the largest past delivery) — never a hand-set
#: threshold — so the "why" behind released capital stays defensible.
EXCESS_CAUSE_LABELS = {
    "demand_fell": "Demand is falling",
    "over_bought": "Bought too much at once",
    "steady_overstock": "Above plan at steady use",
}
EXCESS_CAUSE_ACTIONS = {
    "demand_fell": "Taper the next order — this stock is on a path to dead.",
    "over_bought": "Order smaller and more often — right-size the lot.",
    "steady_overstock": "Skip reorders and let it draw down.",
}


def enrich_surplus(health: Optional[pd.DataFrame],
                   facility_code: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Add the **consumption-based** surplus measure to an inventory_health frame.

    Surplus is defined against what an item actually *consumes*, not the Monte
    Carlo order-up-to level (which collapses to ~0 on weak/censored forecasts and
    then flags every stocked unit as excess). An item's **working requirement**
    is the stock needed to bridge to its next delivery —

        requirement = consumption/day × (reorder interval + lead time) + safety

    all per-item and data-derived: consumption from the trailing-year units,
    interval from the replenishment fit, lead time from paired PO→receipt
    records, and the fitted safety buffer. Anything held above that, for an item
    that genuinely moves, is releasable surplus.

    Scope guards keep the number honest and non-overlapping with other views:
    non-moving stock (dead/slow) is capital handled in its own section, and
    probable miscounts (>2-year supply, ``stock_suspect``) stay quarantined —
    neither is surplus.

    Overwrites ``excess_units``/``excess_value`` with the consumption-based
    figures and adds ``working_requirement``, ``months_of_supply`` and
    ``releasable`` so every capital view (Overview split, waterfall, the surplus
    section) reads one consistent number.
    """
    if health is None or getattr(health, "empty", True) or "item_key" not in health.columns:
        return health
    fac = facility_code or selected_facility()
    out = health.copy()
    out["item_key"] = out["item_key"].astype(str)

    def _map(table: str, col: str) -> pd.Series:
        frame = load_table(table)
        if frame is None or not {"item_key", col} <= set(frame.columns):
            return pd.Series(np.nan, index=out.index, dtype=float)
        f = frame.drop_duplicates("item_key").copy()
        f["item_key"] = f["item_key"].astype(str)
        s = pd.to_numeric(f.set_index("item_key")[col], errors="coerce")
        return out["item_key"].map(s)

    safety = _map("inventory_policy", "safety_stock").fillna(0.0)
    interval = _map("replenishment", "interval_mean")

    leads = item_lead_times(fac)
    lead_med = float(pd.Series(list(leads.values())).median()) if leads else 30.0
    lead = pd.to_numeric(out["item_key"].map(leads), errors="coerce").fillna(lead_med)

    soh = pd.to_numeric(out.get("soh"), errors="coerce")
    annual = pd.to_numeric(out.get("annual_units"), errors="coerce").fillna(0.0)
    price = pd.to_numeric(out.get("unit_price"), errors="coerce")
    mc = out.get("movement_class", pd.Series("", index=out.index)).astype(str)
    susp = out.get("stock_suspect")
    susp = susp.fillna(False).astype(bool) if susp is not None else pd.Series(False, index=out.index)

    daily = annual / 365.0
    coverage_days = interval.fillna(0.0) + lead
    requirement = daily * coverage_days + safety
    months = np.where(daily > 0, soh / (daily * 30.0), np.nan)

    scope = (annual > 0) & (mc == "active") & (~susp)
    excess_units = (soh - requirement).clip(lower=0).where(scope, 0.0)
    excess_value = excess_units * price

    out["working_requirement"] = requirement
    out["months_of_supply"] = months
    out["excess_units"] = excess_units
    out["excess_value"] = excess_value
    out["releasable"] = excess_units > 0
    return out


def classify_excess_cause(release: pd.DataFrame,
                          max_receipt: Optional[dict] = None) -> pd.DataFrame:
    """Attribute each over-top-up item to one root cause, in priority order:

    1. ``demand_fell`` — the item's demand trend is DOWN (the same falling-use
       signal behind freezing): the surplus won't clear on its own, so the
       action is to taper, regardless of how the stock arrived.
    2. ``over_bought`` — a single past delivery was, by itself, at least as
       large as the entire current surplus: the excess is explained by lot size,
       so buy smaller and more often. Needs ``max_receipt`` (delivery sizes).
    3. ``steady_overstock`` — none of the above: steady use but holding above
       plan, so simply stop reordering until it draws down.

    Returns ``release`` with ``cause``/``cause_label``/``cause_action`` columns.
    """
    out = release.copy()
    if out.empty:
        for col in ("cause", "cause_label", "cause_action"):
            out[col] = pd.Series(dtype=object)
        return out
    trend = out.get("trend_direction", pd.Series(index=out.index, dtype=object)).astype("string")
    excess_u = pd.to_numeric(out.get("excess_units"), errors="coerce")
    mr = pd.to_numeric(out["item_key"].astype(str).map(max_receipt or {}), errors="coerce")

    cause = pd.Series("steady_overstock", index=out.index, dtype=object)
    over_bought = mr.notna() & (mr > 0) & excess_u.notna() & (mr >= excess_u)
    cause[over_bought] = "over_bought"
    cause[trend == "DOWN"] = "demand_fell"   # highest priority — assigned last
    out["cause"] = cause
    out["cause_label"] = cause.map(EXCESS_CAUSE_LABELS)
    out["cause_action"] = cause.map(EXCESS_CAUSE_ACTIONS)
    return out


def procurement_spend(facility_code: str, months: int = 36) -> Optional[pd.DataFrame]:
    """Monthly spend/volume by doc_type × category × product, lookback anchored
    to the facility registry's reference date.

    Built from item-grain ``FACT_PROCUREMENT`` (not the pre-aggregated RPT view)
    so non-medical items are removed at source — the clinical-scope rule applied
    everywhere else on the dashboard."""
    ref = sql_ref_date(facility(facility_code))
    sql = f"""
        SELECT p.item_id AS item_key, p.doc_type, p.doc_month, p.doc_id,
               COALESCE(p.product_category, 'uncategorized') AS product_category,
               p.therapeutic_class, p.canonical_name, p.item_name,
               CAST(p.quantity AS FLOAT)   AS quantity,
               CAST(p.line_value AS FLOAT) AS line_value
        FROM {queries.FACT_PROCUREMENT} p
        WHERE p.source_schema = %(facility)s
          AND p.doc_at >= DATEADD(month, -%(months)s, {ref})
    """
    df = _run_query(sql, {"facility": facility_code, "months": months})
    if df is None or df.empty:
        return df
    df = _drop_non_medical(df, facility_code)
    df["doc_month"] = pd.to_datetime(df["doc_month"])
    agg = (df.groupby(["doc_type", "doc_month", "product_category",
                       "therapeutic_class", "canonical_name"], dropna=False)
             .agg(documents=("doc_id", "nunique"), lines=("doc_id", "size"),
                  total_quantity=("quantity", "sum"),
                  total_value=("line_value", "sum"))
             .reset_index())
    agg.insert(0, "facility", facility_code)
    return agg


@st.cache_data(ttl=3600, show_spinner="Querying supplier summary…")
def supplier_summary(facility_code: str) -> Optional[pd.DataFrame]:
    """Per-supplier ordered/received totals, all-time, rebuilt from item-grain
    ``FACT_PROCUREMENT`` with non-medical items removed — so pure-stationery
    suppliers (e.g. TONA STATIONERS) drop out and mixed suppliers show only
    their medical spend."""
    sql = f"""
        SELECT p.item_id AS item_key, p.supplier_id, p.supplier_name,
               p.doc_type, p.doc_id, p.doc_at, p.canonical_name, p.item_name,
               CAST(p.line_value AS FLOAT) AS line_value
        FROM {queries.FACT_PROCUREMENT} p
        WHERE p.source_schema = %(facility)s
    """
    df = _run_query(sql, {"facility": facility_code})
    if df is None or df.empty:
        return df
    df = _drop_non_medical(df, facility_code)
    df["doc_at"] = pd.to_datetime(df["doc_at"], errors="coerce")
    dt = df["doc_type"].astype(str).str.upper()
    base = (df.groupby(["supplier_id", "supplier_name"], dropna=False)
              .agg(distinct_items=("item_key", "nunique"),
                   first_activity=("doc_at", "min"),
                   last_activity=("doc_at", "max")).reset_index())
    orders = (df[dt == "ORDER"].groupby(["supplier_id", "supplier_name"], dropna=False)
              .agg(purchase_orders=("doc_id", "nunique"),
                   ordered_value=("line_value", "sum")).reset_index())
    recv = (df[dt == "RECEIPT"].groupby(["supplier_id", "supplier_name"], dropna=False)
            .agg(goods_received_docs=("doc_id", "nunique"),
                 received_value=("line_value", "sum")).reset_index())
    out = (base.merge(orders, on=["supplier_id", "supplier_name"], how="left")
               .merge(recv, on=["supplier_id", "supplier_name"], how="left"))
    for c in ["purchase_orders", "ordered_value", "goods_received_docs", "received_value"]:
        out[c] = out[c].fillna(0)
    return out


@st.cache_data(ttl=3600, show_spinner="Assessing supplier dependency…")
def supplier_dependency(facility_code: str) -> Optional[pd.DataFrame]:
    """Per-item **observed** supplier profile from goods-receipt records
    (non-medical dropped), one row per ``item_key``:

    - ``observed_suppliers`` — distinct supplier_ids seen receiving this item
    - ``receipt_occasions`` — number of receipt documents (evidence depth)
    - ``received_value`` — KES received, all-time in the records
    - ``span_days`` — first→last receipt span
    - ``supplier_name`` — the one observed supplier's name (only when
      ``observed_suppliers == 1``; else blank)

    Sourcing is only judgeable where an item was received on **2+ occasions**;
    callers MUST restrict dependency conclusions to that repeat-purchased subset
    — a single receipt is not evidence of single-sourcing. ``supplier_id`` is
    100% populated and clean (validated), so item→supplier is reliable. This is
    SKU/brand-level: a molecule split across brand IDs may be multi-sourced even
    where one brand looks dependent. None when Snowflake is unavailable.
    """
    sql = f"""
        SELECT CAST(p.item_id AS VARCHAR) AS item_key, p.canonical_name, p.item_name,
               p.supplier_id, p.supplier_name, p.doc_id,
               CAST(p.doc_at AS TIMESTAMP_NTZ) AS doc_at,
               CAST(p.line_value AS FLOAT)     AS line_value
        FROM {queries.FACT_PROCUREMENT} p
        WHERE p.source_schema = %(facility)s AND p.doc_type = 'RECEIPT'
    """
    df = _run_query(sql, {"facility": facility_code})
    if df is None or df.empty:
        return None
    df = _drop_non_medical(df, facility_code)
    if df is None or df.empty:
        return None
    df = df.copy()
    df["item_key"] = df["item_key"].astype(str)
    df["doc_at"] = pd.to_datetime(df["doc_at"], errors="coerce")
    g = (df.groupby("item_key")
           .agg(observed_suppliers=("supplier_id", "nunique"),
                receipt_occasions=("doc_id", "nunique"),
                received_value=("line_value", "sum"),
                first_receipt=("doc_at", "min"),
                last_receipt=("doc_at", "max"))
           .reset_index())
    g["span_days"] = (g["last_receipt"] - g["first_receipt"]).dt.days
    first_name = (df.dropna(subset=["supplier_name"]).sort_values("doc_at")
                    .groupby("item_key")["supplier_name"].first())
    g["supplier_name"] = g["item_key"].map(first_name)
    g.loc[g["observed_suppliers"] != 1, "supplier_name"] = pd.NA
    return g

