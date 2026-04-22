"""
PharmaPlus Kenya — Recommendation Engine v2
============================================
Value-optimised argmax decision engine.

Decision waterfall (priority order):
    1. CRITICAL  — days_to_expiry < CRITICAL_DAYS (override all)
    2. TRANSFER  — transfer_value beats discount AND bundle
    3. BUNDLE    — bundle_value beats discount
    4. DISCOUNT  — passes recovery floor
    5. MONITOR   — no viable action

All Polars when/then chains use pl.lit() for string literals.
Compatible with both demo CSV loader and production MySQL loader.
"""

import logging
from typing import Optional

import polars as pl


# ── Config ─────────────────────────────────────────────────────────────────────
class EngineConfig:
    """
    Centralized constants for the PharmaPlus recommendation engine.
    All business logic thresholds live here — do not hardcode elsewhere.
    """

    # Discount
    MAX_DISCOUNT      = 0.60    # Maximum pharmacist discount (60%)
    RECOVERY_FLOOR    = 0.40    # Min revenue recovery to recommend discount
    DISCOUNT_FLOOR    = 0.05    # Suppress signals below this threshold
    DISCOUNT_EXPONENT = 2       # Urgency curve steepness
    DISCOUNT_WINDOW   = 180     # Fixed window for continuous urgency curve

    # Elasticity
    ELASTICITY_FACTOR = 1.5     # Demand uplift per unit of discount percentage

    # Transfer
    TRANSIT_DAYS         = 2    # Assumed transit days (demo hardcoded)
    TRANSFER_BASE_COST   = 200  # KSH flat cost per transfer
    TRANSFER_COST_PER_KM = 5    # KSH per km × qty transferred

    # Bundle
    BUNDLE_DISCOUNT = 0.10      # Margin surrendered on the expiring product in bundle

    # Critical
    CRITICAL_DAYS = 15          # Days to expiry below which action is CRITICAL

    # Branch geography
    BRANCH_DISTANCES_KM: dict = {
        (1, 2): 4.2, (2, 1): 4.2,
        (1, 3): 8.7, (3, 1): 8.7,
        (2, 3): 7.1, (3, 2): 7.1,
    }
    BRANCH_FACILITY_MAP: dict = {
        "Branch Yaya":       1,
        "Branch Westlands":  2,
        "Branch Imara Mall": 3,
    }


# ── Transfer value helper ──────────────────────────────────────────────────────
def _compute_transfer_targets(
    df: pl.DataFrame,
    config: EngineConfig,
) -> pl.DataFrame:
    """
    Split-transfer: allocate qty_at_risk across ALL viable targets in velocity order.
    Each target absorbs min(qty_remaining, tgt_vel × days_usable).

    Aggregated per product × source:
        transfer_value        : sum of value across all destinations
        target_facility_id    : primary destination (highest allocation)
        transfer_destinations : pipe-delimited branch names  e.g. "Branch Westlands | Branch Yaya"
        transfer_quantities   : pipe-delimited qty per dest  e.g. "312 | 229"
        transfer_km / cost    : from primary destination
    """
    FACILITY_NAME = config.BRANCH_FACILITY_MAP  # {name: id} — invert below
    FAC_ID_TO_NAME = {v: k for k, v in FACILITY_NAME.items()}

    records = []

    for row in df.iter_rows(named=True):
        src     = row["facility_id"]
        qty_rem = float(row["qty_at_risk"])

        # Collect all viable targets with their capacity, sorted best velocity first
        candidates = []
        for (s, t), km in config.BRANCH_DISTANCES_KM.items():
            if s != src:
                continue
            matches = df.filter(
                (pl.col("product_id") == row["product_id"]) &
                (pl.col("facility_id") == t)
            )
            if matches.shape[0] == 0:
                continue

            tgt_vel = float(matches["daily_velocity"][0])
            tgt_dte = float(matches["days_to_expiry"][0])

            if tgt_vel <= row["daily_velocity"]:
                continue  # no velocity advantage

            days_usable = tgt_dte - config.TRANSIT_DAYS
            if days_usable <= 0:
                continue  # target expires before transfer lands

            capacity = tgt_vel * days_usable
            if capacity <= 0:
                continue

            candidates.append((tgt_vel, t, km, capacity))

        if not candidates:
            continue

        # Sort by velocity descending — allocate to best movers first
        candidates.sort(key=lambda x: x[0], reverse=True)

        dest_names  = []
        dest_qtys   = []
        total_value = 0.0
        total_cost  = 0.0
        primary_t   = candidates[0][1]
        primary_km  = candidates[0][2]

        # Base cost charged once per product transfer, not per destination
        base_cost_remaining = float(config.TRANSFER_BASE_COST)

        for _, t, km, capacity in candidates:
            if qty_rem <= 0:
                break
            alloc     = min(qty_rem, capacity)
            var_cost  = km * config.TRANSFER_COST_PER_KM * alloc
            logistics = base_cost_remaining + var_cost
            value     = alloc * row["selling_price"] - logistics

            # Fix 4: skip allocations that cost more than they recover
            if value <= 0:
                continue

            dest_names.append(FAC_ID_TO_NAME.get(t, str(t)))
            dest_qtys.append(str(round(alloc)))
            total_value          += value
            total_cost           += logistics
            base_cost_remaining   = 0.0   # base cost absorbed after first viable destination
            qty_rem              -= alloc

        if not dest_names:
            continue

        records.append({
            "product_id":             row["product_id"],
            "facility_id":            src,
            "target_facility_id":     primary_t,
            "transfer_km":            primary_km,
            "transfer_cost":          round(total_cost, 2),
            "transfer_value":         round(total_value, 2),
            "transfer_destinations":  " | ".join(dest_names),
            "transfer_quantities":    " | ".join(dest_qtys),
        })

    empty_schema = {
        "product_id":            pl.Utf8,
        "facility_id":           pl.Int64,
        "target_facility_id":    pl.Int64,
        "transfer_km":           pl.Float64,
        "transfer_cost":         pl.Float64,
        "transfer_value":        pl.Float64,
        "transfer_destinations": pl.Utf8,
        "transfer_quantities":   pl.Utf8,
    }
    if not records:
        return pl.DataFrame(schema=empty_schema)

    return pl.DataFrame(records)


# ── Bundle logic ───────────────────────────────────────────────────────────────
def apply_bundle_logic(
    df_expiry: pl.DataFrame,
    config: Optional[EngineConfig] = None,
) -> pl.DataFrame:
    """
    PharmaPlus Kenya — Dynamic Bundle Engine.

    Identifies at-risk products (clearance-based) and pairs them with
    high-velocity companions within the same facility and product segment.

    At-risk gate   : qty_at_risk / daily_velocity > days_to_expiry
    Companion gate : same facility_id + segment, companion_dte > at_risk_dte,
                     companion_days_of_stock >= at_risk_dte (viability gate)

    Leader Bundle model:
        Companion sells at full price (drives footfall).
        Expiring product sells at BUNDLE_DISCOUNT (10% reduction).

    Returns top-3 companion pairs per at_risk_product × facility,
    ranked by companion daily velocity (descending).
    """
    if config is None:
        config = EngineConfig()

    # At-risk products
    at_risk = (
        df_expiry
        .filter(
            pl.col("qty_at_risk") / (pl.col("daily_velocity") + 1e-9)
            > pl.col("days_to_expiry")
        )
        .select([
            "product_id", "product_name", "facility_id", "store_name",
            "segment", "days_to_expiry", "expiry_tier",
            "qty_at_risk", "daily_velocity", "selling_price",
            "unit_cost", "ksh_value_at_risk",
        ])
        .rename({
            "product_id":       "at_risk_product_id",
            "product_name":     "at_risk_product_name",
            "days_to_expiry":   "at_risk_dte",
            "qty_at_risk":      "at_risk_qty",
            "daily_velocity":   "at_risk_velocity",
            "selling_price":    "at_risk_price",
            "unit_cost":        "at_risk_cost",
            "ksh_value_at_risk":"at_risk_ksh",
        })
    )

    if at_risk.shape[0] == 0:
        logging.warning("apply_bundle_logic: no at-risk products found")
        return pl.DataFrame()

    # Companion pool: NOT at-risk
    companions = (
        df_expiry
        .filter(
            pl.col("qty_at_risk") / (pl.col("daily_velocity") + 1e-9)
            <= pl.col("days_to_expiry")
        )
        .select([
            "product_id", "product_name", "facility_id",
            "segment", "days_to_expiry", "qty_at_risk",
            "daily_velocity", "selling_price",
        ])
        .rename({
            "product_id":   "companion_product_id",
            "product_name": "companion_product_name",
            "days_to_expiry": "companion_dte",
            "qty_at_risk":  "companion_qty",
            "daily_velocity": "companion_daily_velocity",
            "selling_price":  "companion_price",
        })
    )

    # Cross-join on facility + segment, apply all filters
    df_pairs = (
        at_risk.join(companions, on=["facility_id", "segment"])
        .filter(
            (pl.col("companion_dte") > pl.col("at_risk_dte")) &
            (pl.col("at_risk_product_id") != pl.col("companion_product_id"))
        )
        .with_columns([
            (pl.col("companion_qty") / (pl.col("companion_daily_velocity") + 1e-9))
            .alias("companion_days_of_stock")
        ])
        .filter(
            pl.col("companion_days_of_stock") >= pl.col("at_risk_dte")
        )
    )

    if df_pairs.shape[0] == 0:
        logging.info("apply_bundle_logic: no viable companion pairs found")
        return pl.DataFrame()

    df_pairs = df_pairs.with_columns([
        # Bundle unit value: full-price companion + discounted expiring product
        (
            pl.col("companion_price") +
            pl.col("at_risk_price") * (1 - pl.lit(config.BUNDLE_DISCOUNT))
        ).alias("bundle_unit_value"),

        # Projected bundle revenue over full at-risk period
        (
            pl.col("companion_price") * pl.col("at_risk_dte") * pl.col("companion_daily_velocity")
            + pl.col("at_risk_ksh") * (1 - pl.lit(config.BUNDLE_DISCOUNT))
        ).alias("projected_bundle_revenue"),

        # Value recovered for the expiring product
        (pl.col("at_risk_ksh") * (1 - pl.lit(config.BUNDLE_DISCOUNT)))
        .alias("bundle_recovered_value"),

        # Cost of the bundle discount
        (pl.col("at_risk_ksh") * pl.lit(config.BUNDLE_DISCOUNT))
        .alias("bundle_discount_cost"),

        # Rank companions by velocity (best companion = rank 1)
        (-pl.col("companion_daily_velocity"))
        .rank(method="ordinal")
        .over(["at_risk_product_id", "facility_id"])
        .alias("companion_rank"),
    ]).filter(
        pl.col("companion_rank") <= 3
    ).sort(["at_risk_dte", "facility_id", "companion_rank"])

    desired = [
        "at_risk_product_id", "at_risk_product_name",
        "facility_id", "store_name",
        "at_risk_dte", "expiry_tier",
        "at_risk_qty", "at_risk_price", "at_risk_ksh",
        "companion_rank",
        "companion_product_id", "companion_product_name",
        "companion_daily_velocity", "companion_qty",
        "companion_days_of_stock", "companion_dte",
        "bundle_unit_value", "projected_bundle_revenue",
        "bundle_recovered_value", "bundle_discount_cost",
    ]
    available = [c for c in desired if c in df_pairs.columns]
    return df_pairs.select(available)


# ── Core recommendation engine ─────────────────────────────────────────────────
def apply_recommendation_logic(
    df_expiry:  pl.DataFrame,
    df_bundle:  Optional[pl.DataFrame] = None,   # pre-computed bundle output (optional)
    df_geo:     Optional[pl.DataFrame] = None,   # branch_market_dna.csv (optional)
    df_serp:    Optional[pl.DataFrame] = None,   # competitor pricing (optional)
    config:     Optional[EngineConfig] = None,
) -> pl.DataFrame:
    """
    PharmaPlus Kenya — Core Recommendation Engine v2.

    Input:
        df_expiry  : expiry_stock_sim.csv (or production v_expiry_stock view)
                     Required columns: product_id, product_name, facility_id,
                     store_name, segment, days_to_expiry, expiry_tier,
                     qty_at_risk, selling_price, unit_cost, ksh_value_at_risk,
                     daily_velocity, seasonal_boost

        df_bundle  : pre-computed bundle output from apply_bundle_logic()
                     If None, bundle logic is run internally.

        df_geo     : branch_market_dna.csv — geo signals (optional enrichment)
        df_serp    : competitor pricing (optional enrichment)
        config     : EngineConfig instance (defaults to EngineConfig())

    Output:
        60-row (or production-sized) DataFrame with action + value columns.
    """
    if config is None:
        config = EngineConfig()

    df = df_expiry.clone()

    # ── Guard: required columns ────────────────────────────────────────────────
    required = [
        "product_id", "facility_id", "days_to_expiry",
        "qty_at_risk", "daily_velocity", "selling_price",
        "unit_cost", "ksh_value_at_risk",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df_expiry missing required columns: {missing}")

    # ── Ensure float types ─────────────────────────────────────────────────────
    df = df.with_columns([
        pl.col("daily_velocity").cast(pl.Float64),
        pl.col("qty_at_risk").cast(pl.Float64),
        pl.col("selling_price").cast(pl.Float64),
        pl.col("unit_cost").cast(pl.Float64),
        pl.col("ksh_value_at_risk").cast(pl.Float64),
        pl.col("days_to_expiry").cast(pl.Float64),
    ])

    # Apply seasonal boost if column present (filled with 1.0 if missing)
    if "seasonal_boost" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("seasonal_boost"))
    df = df.with_columns([
        (pl.col("daily_velocity") * pl.col("seasonal_boost"))
        .alias("adj_daily_velocity")
    ])

    # ── STEP 1: Clearance probability ─────────────────────────────────────────
    df = df.with_columns([
        (
            (pl.col("days_to_expiry") * pl.col("adj_daily_velocity"))
            / (pl.col("qty_at_risk") + 1e-9)
        ).clip(0.0, 1.0).alias("clearance_prob"),

        (
            pl.col("qty_at_risk") / (pl.col("adj_daily_velocity") + 1e-9)
            > pl.col("days_to_expiry")
        ).alias("is_at_risk_flag"),
    ])

    # ── STEP 2: Urgency-scaled discount ───────────────────────────────────────
    df = df.with_columns([
        (config.MAX_DISCOUNT * (1 - pl.col("clearance_prob")))
        .alias("effective_M")
    ]).with_columns([
        (
            pl.col("effective_M") *
            (
                (pl.lit(float(config.DISCOUNT_WINDOW)) -
                 pl.col("days_to_expiry").clip(0.0, float(config.DISCOUNT_WINDOW)))
                / pl.lit(float(config.DISCOUNT_WINDOW))
            ).pow(config.DISCOUNT_EXPONENT)
        ).clip(0.0, config.MAX_DISCOUNT).alias("raw_discount_pct")
    ]).with_columns([
        pl.when(pl.col("raw_discount_pct") < config.DISCOUNT_FLOOR)
        .then(pl.lit(0.0))
        .otherwise(pl.col("raw_discount_pct"))
        .alias("discount_pct")
    ])

    # ── STEP 3: Elasticity-adjusted discount value ────────────────────────────
    # expected_units = adj_velocity * (1 + ELASTICITY * discount_pct) * days_to_expiry
    # capped at qty_at_risk
    df = df.with_columns([
        (
            pl.col("adj_daily_velocity") *
            (1 + pl.lit(config.ELASTICITY_FACTOR) * pl.col("discount_pct")) *
            pl.col("days_to_expiry")
        ).clip(0.0, pl.col("qty_at_risk")).alias("expected_units_sold")
    ]).with_columns([
        (
            pl.col("expected_units_sold") *
            pl.col("selling_price") *
            (1 - pl.col("discount_pct"))
        ).alias("discount_value")
    ])

    # ── STEP 4: Transfer value ─────────────────────────────────────────────────
    df_transfers = _compute_transfer_targets(df, config)

    if df_transfers.shape[0] > 0:
        df = df.join(
            df_transfers.select([
                "product_id", "facility_id",
                "target_facility_id", "transfer_km",
                "transfer_cost", "transfer_value",
                "transfer_destinations", "transfer_quantities",
            ]),
            on=["product_id", "facility_id"],
            how="left",
        )
    else:
        df = df.with_columns([
            pl.lit(None).cast(pl.Int64).alias("target_facility_id"),
            pl.lit(0.0).alias("transfer_km"),
            pl.lit(0.0).alias("transfer_cost"),
            pl.lit(0.0).alias("transfer_value"),
            pl.lit(None).cast(pl.Utf8).alias("transfer_destinations"),
            pl.lit(None).cast(pl.Utf8).alias("transfer_quantities"),
        ])

    df = df.with_columns([
        pl.col("target_facility_id").fill_null(-1),
        pl.col("transfer_value").fill_null(0.0),
        pl.col("transfer_cost").fill_null(0.0),
        pl.col("transfer_km").fill_null(0.0),
        pl.col("transfer_destinations").fill_null(""),
        pl.col("transfer_quantities").fill_null(""),
    ])

    # ── STEP 5: Bundle value ───────────────────────────────────────────────────
    if df_bundle is None:
        df_bundle = apply_bundle_logic(df_expiry, config)

    if df_bundle is not None and df_bundle.shape[0] > 0:
        # Take best companion per at-risk product × facility
        bundle_best = (
            df_bundle
            .sort("companion_rank")
            .unique(subset=["at_risk_product_id", "facility_id"], keep="first")
            .select([
                pl.col("at_risk_product_id").alias("product_id"),
                "facility_id",
                pl.col("companion_product_id").alias("bundle_companion_id"),
                pl.col("companion_daily_velocity").alias("bundle_companion_velocity"),
                pl.col("bundle_recovered_value").alias("bundle_value"),
            ])
        )
        df = df.join(bundle_best, on=["product_id", "facility_id"], how="left")
    else:
        df = df.with_columns([
            pl.lit(None).cast(pl.Utf8).alias("bundle_companion_id"),
            pl.lit(0.0).alias("bundle_companion_velocity"),
            pl.lit(0.0).alias("bundle_value"),
        ])

    df = df.with_columns([
        pl.col("bundle_companion_id").fill_null(""),
        pl.col("bundle_value").fill_null(0.0),
    ])

    # ── STEP 6: Geo enrichment + value adjustments (optional) ────────────────
    if df_geo is not None and "facility_id" in df_geo.columns:
        geo_cols = [c for c in [
            "facility_id", "Competition", "Lifestyle_Drivers",
            "Social_Drivers", "High_Pressure_Comp"
        ] if c in df_geo.columns]
        if len(geo_cols) > 1:
            df = df.join(df_geo.select(geo_cols), on="facility_id", how="left")
            logging.info("Geo DNA signals joined successfully")

        # Geo adjustments — applied before argmax
        # High local competition → stronger demand response to discounts
        # Correct application: boost ELASTICITY on expected_units_sold (physical units),
        # NOT on final revenue — you cannot recover more than qty × selling_price.
        if "High_Pressure_Comp" in df.columns:
            _geo_elasticity = pl.lit(config.ELASTICITY_FACTOR * 1.10)
            df = df.with_columns([
                pl.when(pl.col("High_Pressure_Comp") > pl.lit(3))
                .then(
                    (
                        pl.col("adj_daily_velocity") *
                        (pl.lit(1.0) + _geo_elasticity * pl.col("discount_pct")) *
                        pl.col("days_to_expiry")
                    ).clip(0.0, pl.col("qty_at_risk"))
                )
                .otherwise(pl.col("expected_units_sold"))
                .alias("expected_units_sold")
            ]).with_columns([
                # Recompute discount_value from updated expected_units_sold
                (
                    pl.col("expected_units_sold") *
                    pl.col("selling_price") *
                    (pl.lit(1.0) - pl.col("discount_pct"))
                ).alias("discount_value")
            ])

        # Lifestyle-driven branch → less discount needed to move bundle stock
        # Applied as a reduction in the effective bundle discount rate,
        # keeping bundle_value ≤ at_risk_ksh (no phantom revenue).
        if "Lifestyle_Drivers" in df.columns:
            _lifestyle_discount = pl.lit(config.BUNDLE_DISCOUNT * 0.90)   # 9% instead of 10%
            df = df.with_columns([
                pl.when(pl.col("Lifestyle_Drivers") > pl.lit(10))
                .then(pl.col("ksh_value_at_risk") * (pl.lit(1.0) - _lifestyle_discount))
                .otherwise(pl.col("bundle_value"))
                .alias("bundle_value")
            ])

    # ── STEP 7: Competitor pricing annotation (optional) ──────────────────────
    if df_serp is not None and "product_id" in df_serp.columns:
        comp_cols = [c for c in [
            "product_id", "goodlife", "lintons", "mydawa",
            "market_low", "price_gap_pct", "price_position",
            "match_score", "competitor_match_count",
        ] if c in df_serp.columns]
        if len(comp_cols) > 1:
            df = df.join(df_serp.select(comp_cols), on="product_id", how="left")
            logging.info("Competitor prices joined successfully")

    # ── STEP 8: Argmax decision waterfall ─────────────────────────────────────
    #
    #  Priority 1 — CRITICAL  : imminent expiry, override everything
    #  Priority 2 — TRANSFER  : transfer_value >= discount AND bundle, > 0
    #  Priority 3 — BUNDLE    : bundle_value >= discount, companion viable
    #  Priority 4 — DISCOUNT  : passes recovery floor
    #  Priority 5 — MONITOR   : no viable action
    #
    df = df.with_columns([
        pl.when(
            pl.col("days_to_expiry") <= pl.lit(float(config.CRITICAL_DAYS))
        )
        .then(pl.lit("CRITICAL"))

        .when(
            (pl.col("target_facility_id") != -1) &
            (pl.col("transfer_value") >= pl.col("discount_value")) &
            (pl.col("transfer_value") >= pl.col("bundle_value")) &
            (pl.col("transfer_value") > 0)
        )
        .then(pl.lit("TRANSFER"))

        .when(
            (pl.col("bundle_companion_id") != pl.lit("")) &
            (pl.col("bundle_value") >= pl.col("discount_value")) &
            (pl.col("bundle_value") > 0)
        )
        .then(pl.lit("BUNDLE"))

        .when(
            (pl.col("discount_pct") > pl.lit(0.0)) &          # must be a real discount
            (pl.col("discount_value") > 0) &
            (
                pl.col("discount_value") >=
                pl.lit(config.RECOVERY_FLOOR) * pl.col("ksh_value_at_risk")
            )
        )
        .then(pl.lit("DISCOUNT"))

        .otherwise(pl.lit("MONITOR"))
        .alias("action")
    ])

    # ── STEP 9: Recovered value projection ────────────────────────────────────
    df = df.with_columns([
        pl.when(pl.col("action") == pl.lit("CRITICAL"))
        .then(pl.lit(0.0))

        .when(pl.col("action") == pl.lit("TRANSFER"))
        .then(pl.col("transfer_value"))

        .when(pl.col("action") == pl.lit("BUNDLE"))
        .then(pl.col("bundle_value"))

        .when(pl.col("action") == pl.lit("DISCOUNT"))
        .then(pl.col("discount_value"))

        .otherwise(pl.lit(0.0))
        .alias("recovered_value")
    ])

    # ── STEP 10: Output selection ──────────────────────────────────────────────
    desired_cols = [
        # Identity
        "product_id", "product_name", "brand",
        "category", "segment",
        "facility_id", "store_name",
        # Expiry
        "expiry_date", "days_to_expiry", "expiry_tier",
        # Inventory
        "qty_at_risk", "selling_price", "unit_cost", "ksh_value_at_risk",
        # Velocity
        "daily_velocity", "adj_daily_velocity", "seasonal_boost", "seasonal_flag",
        # Clearance
        "clearance_prob", "is_at_risk_flag",
        # Discount
        "effective_M", "raw_discount_pct", "discount_pct",
        "expected_units_sold", "discount_value",
        # Transfer
        "target_facility_id", "transfer_km", "transfer_cost", "transfer_value",
        "transfer_destinations", "transfer_quantities",
        # Bundle
        "bundle_companion_id", "bundle_companion_velocity", "bundle_value",
        # Geo (optional)
        "Competition", "Lifestyle_Drivers", "Social_Drivers", "High_Pressure_Comp",
        # Competitor pricing (optional)
        "goodlife", "lintons", "mydawa", "market_low",
        "price_gap_pct", "price_position", "match_score", "competitor_match_count",
        # Decision
        "action", "recovered_value",
    ]
    available = [c for c in desired_cols if c in df.columns]
    return df.select(available)
