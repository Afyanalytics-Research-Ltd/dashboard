"""
Seasonal demand intelligence for Kisumu Specialist Hospital.

Two layers:
  1. Curated disease calendar — Kisumu-specific epidemiological patterns mapped to
     drug classes. Based on KEMRI Disease Surveillance Reports, WHO Kenya Country
     Office data, and Kenya DHIS2 Kisumu County historical incidence records.
     Updated annually.
  2. Live climate signal — Open-Meteo rainfall data for Kisumu (lat -0.1022, lon 34.7617).
     Rainfall anomaly modulates alert severity for climate-driven diseases (malaria,
     diarrhoeal disease, typhoid).

Usage (dashboard level):
    from intelligence.seasonal_engine import SeasonalEngine, get_climate_signal, parse_ref_date

    ref = parse_ref_date(_ref_date)
    climate = get_climate_signal(ref)             # cached at 86400s TTL in dashboard
    eng = SeasonalEngine()
    alerts  = eng.match_products(soh_df, ref, climate)
    mults   = eng.get_seasonal_multipliers(soh_df, ref, climate)
    outlook = eng.get_outlook(ref)
"""

from __future__ import annotations

import calendar as _cal
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import pandas as pd

# ── Kisumu GPS coordinates ────────────────────────────────────────────────────

KISUMU_LAT: float = -0.1022
KISUMU_LON: float = 34.7617

# ── Tuning constants ──────────────────────────────────────────────────────────

# Rainfall anomaly thresholds for severity modulation
RAINFALL_HIGH_PCT: float    = 20.0   # >20% above avg → moderate uplift (+8%)
RAINFALL_EXTREME_PCT: float = 50.0   # >50% above avg → strong uplift (+15%)

# Minimum adjusted DOS (after applying multiplier) to bother alerting
MIN_ALERTABLE_ADJ_DOS: float = 60.0  # don't alert if adjusted DOS still > 60d

# ── Phase 3.5: Facility-Specific Seasonal Index constants ────────────────────

# Minimum total months of dispensing data for ANY facility index to be computed
MIN_MONTHS_FOR_FACILITY_INDEX: int = 6
# Minimum months a subclass must appear in for its index to be used
MIN_MONTHS_PER_SUBCLASS: int = 3
# Stockout rate threshold: months where >30% of dispenses were from zero-stock
# are flagged as demand-suppressed (true demand was higher than observed)
STOCKOUT_RATE_THRESHOLD: float = 0.30
# SOH_BEFORE ratio: if median(soh_before) in a month is >2.5x the annual median,
# a dispensing spike in that month may be supply-driven (large delivery → more dispensed),
# not a true demand peak
SUPPLY_SIGNAL_SOH_RATIO: float = 2.5
MIN_INDEX_FOR_SUPPLY_CHECK: float = 1.15  # only check supply signal when index is elevated

# Blending alpha: weight of facility index vs calendar prior
# Based on n_years of observed data for a given therapeutic subclass
# n=1→0.20  (sparse — calendar still dominates)
# n=2→0.40  (reasonable — lean toward calendar)
# n=3→0.60  (good — lean toward facility data)
# n≥4→0.75  (strong — facility data is the primary signal)
_ALPHA_BY_YEARS: Dict[int, float] = {1: 0.20, 2: 0.40, 3: 0.60}
_ALPHA_MAX: float = 0.75


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SeasonalDisease:
    """One entry in the curated disease calendar."""
    name: str
    peak_months: List[int]           # 1=Jan … 12=Dec
    subclass_keywords: List[str]     # case-insensitive substrings vs THERAPEUTIC_SUBCLASS
    class_keywords: List[str]        # fallback vs THERAPEUTIC_CLASS
    demand_multiplier: float         # expected demand uplift at peak (1.40 = +40%)
    climate_driven: bool             # True = rainfall anomaly modulates severity/multiplier
    warning_weeks: int               # weeks ahead to start warning before peak month
    description: str                 # shown in Demand Insights seasonal outlook


@dataclass
class SeasonalAlert:
    """A fired seasonal alert for one drug approaching a disease season peak."""
    disease: str
    drug_name: str
    product_id: str
    therapeutic_subclass: str
    weeks_to_peak: int               # 0 = currently in peak, >0 = approaching
    demand_multiplier: float         # effective multiplier (may be climate-boosted)
    current_dos: Optional[float]
    adjusted_dos: Optional[float]    # current_dos / demand_multiplier
    current_soh: Optional[float]
    severity: str                    # CRITICAL | HIGH | MEDIUM
    climate_boosted: bool
    rainfall_anomaly_pct: Optional[float]


@dataclass
class ClimateSignal:
    """Current month rainfall vs historical average for Kisumu."""
    current_month_mm: float
    historical_avg_mm: float
    anomaly_pct: float               # (current - avg) / avg * 100
    anomaly_label: str               # "above average" | "near average" | "below average"
    current_month_name: str
    data_source: str
    fetched_at: str


@dataclass
class FacilitySeasonalProfile:
    """
    Per-facility seasonal demand indices computed from actual dispensing history.

    Index = monthly_avg_adc / annual_avg_adc per therapeutic subclass.
    1.0 = baseline month; 1.4 = 40% above annual average = seasonal peak.

    These replace (or blend with) the hardcoded Kisumu disease calendar for
    facilities with sufficient dispensing history.
    """
    # Core indices: subclass → {month_int → index}
    index_by_subclass_month: Dict[str, Dict[int, float]]
    # n_years of data backing each (subclass, month) — determines blending alpha
    confidence_by_subclass_month: Dict[str, Dict[int, int]]
    # Months where stockout rate > threshold: observed ADC is suppressed/unreliable
    stockout_flagged_months: Dict[str, Set[int]]
    # Months where elevated SOH_BEFORE suggests supply-driven dispensing spike
    supply_signal_months: Dict[str, Set[int]]
    # Annual average ADC per subclass (baseline for index computation)
    annual_adc_by_subclass: Dict[str, float]
    # Blending alpha per subclass (facility weight vs calendar prior)
    alpha_by_subclass: Dict[str, float]
    # Total months of dispensing data across all products
    n_months_history: int
    facility_schema: str
    computed_at: datetime.date

    def get_effective_index(self, subclass: str, month: int) -> Optional[float]:
        """
        Facility index for (subclass, month), or None if flagged unreliable.

        Returns None for:
        - Subclass not in profile (cold start — caller falls back to calendar)
        - Month not observed
        - Stockout-suppressed month (demand was suppressed; index biased downward)
        """
        if subclass not in self.index_by_subclass_month:
            return None
        idx = self.index_by_subclass_month[subclass].get(month)
        if idx is None:
            return None
        if month in self.stockout_flagged_months.get(subclass, set()):
            return None
        return idx

    def get_blending_alpha(self, subclass: str) -> float:
        """Weight of facility index in the blend: facility*alpha + calendar*(1-alpha)."""
        return self.alpha_by_subclass.get(subclass, 0.0)

    def is_supply_signal(self, subclass: str, month: int) -> bool:
        """True if a dispensing peak in this month may be supply-driven, not demand-driven."""
        return month in self.supply_signal_months.get(subclass, set())


# ── Curated disease calendar for Kisumu County ────────────────────────────────
# Sources:
#   KEMRI Annual Disease Surveillance Report (Western Region)
#   WHO Kenya Country Office — Disease Burden Profiles
#   Kenya National DHIS2 — Kisumu County Monthly Morbidity Reports
#   Ministry of Health Kenya — Integrated Disease Surveillance and Response (IDSR)

DISEASE_CALENDAR: List[SeasonalDisease] = [
    SeasonalDisease(
        name="Malaria",
        peak_months=[3, 4, 5, 10, 11],
        subclass_keywords=["antimalarial", "malaria", "artemether", "quinine", "lumefantrine"],
        class_keywords=["antiparasitic", "antimalarials"],
        demand_multiplier=1.45,
        climate_driven=True,
        warning_weeks=8,
        description=(
            "Kisumu sits on Lake Victoria — Kenya's highest malaria burden zone. "
            "Two peaks: long rains (March–May) and short rains (October–November). "
            "Rainfall above seasonal average significantly amplifies mosquito breeding "
            "and transmission. Artemether-lumefantrine, quinine, and antipyretics "
            "are the primary affected drug classes."
        ),
    ),
    SeasonalDisease(
        name="Diarrhoeal Disease & Cholera",
        peak_months=[5, 6, 11, 12],
        subclass_keywords=[
            "antidiarrhoeal", "antidiarrheal", "oral rehydration", "ors",
            "metronidazole", "zinc sulphate", "zinc sulfate",
        ],
        class_keywords=["oral rehydration"],
        demand_multiplier=1.35,
        climate_driven=True,
        warning_weeks=6,
        description=(
            "Waterborne disease peaks 4–6 weeks after heavy rains as floodwater "
            "contaminates water sources around Lake Victoria. Kisumu's low-lying "
            "terrain increases flood risk significantly. Cholera outbreaks historically "
            "follow long rains in informal settlements. ORS, zinc, metronidazole, "
            "ciprofloxacin, and IV fluids are primary affected classes."
        ),
    ),
    SeasonalDisease(
        name="Respiratory Infections",
        peak_months=[7, 8, 9],
        subclass_keywords=[
            "bronchodilator", "salbutamol", "ipratropium",
            "corticosteroid inhaler", "budesonide", "beclomethasone",
            "prednisolone", "macrolide", "azithromycin",
            "leukotriene",  # Confirmed at KSH: 86 units/day — dominant respiratory class
        ],
        class_keywords=["respiratory"],
        demand_multiplier=1.25,
        climate_driven=False,
        warning_weeks=6,
        description=(
            "Dry season (July–September) brings cold mornings and dust around "
            "Lake Victoria. Pneumonia and URTI admissions historically peak "
            "in August–September, particularly in children under 5 and elderly "
            "patients. Amoxicillin, azithromycin, salbutamol, and prednisolone "
            "see the largest volume increases."
        ),
    ),
    SeasonalDisease(
        name="Typhoid",
        peak_months=[6, 7],
        subclass_keywords=[
            "fluoroquinolone", "ciprofloxacin", "ceftriaxone", "cefixime",
        ],
        class_keywords=[],
        demand_multiplier=1.30,
        climate_driven=True,
        warning_weeks=6,
        description=(
            "Typhoid surges 4–6 weeks post long rains (June–July) as floodwater "
            "infiltrates water supplies in peri-urban Kisumu. Ciprofloxacin, "
            "ceftriaxone, and cefixime demand increases alongside enteric "
            "fever admissions at referral facilities."
        ),
    ),
    SeasonalDisease(
        name="Malaria-Associated Anaemia",
        peak_months=[4, 5, 6, 11, 12],
        subclass_keywords=[
            # Haematinics excluded: KSH dispensing data shows no seasonal pattern
            # (flat index year-round); not a reliable signal at this facility
            "blood transfusion",
        ],
        class_keywords=["haematology"],
        demand_multiplier=1.30,
        climate_driven=True,
        warning_weeks=6,
        description=(
            "Severe malaria causes haemolytic anaemia — disproportionately in "
            "children under 5. Ferrous sulphate, folic acid, and transfusion "
            "supplies peak in the weeks following malaria season onset, lagging "
            "the malaria peak by approximately 2–4 weeks."
        ),
    ),
    SeasonalDisease(
        name="Skin & Wound Infections",
        peak_months=[4, 5, 10, 11],
        subclass_keywords=[
            "flucloxacillin", "cloxacillin", "dicloxacillin",
            "wound care", "antiseptic", "povidone", "chlorhexidine",
            "gentamicin cream", "topical antibiotic",
        ],
        class_keywords=[],
        demand_multiplier=1.20,
        climate_driven=True,
        warning_weeks=4,
        description=(
            "Rainy season flooding increases skin abrasions, wound infections, "
            "and cellulitis admissions at Kisumu facilities. Penicillinase-resistant "
            "penicillins and wound care supplies see moderate demand uplift "
            "during and immediately following peak flood periods."
        ),
    ),
]


# ── Phase 3.5: Facility seasonal index computation ────────────────────────────

def compute_facility_seasonal_index(
    dispensing_df: pd.DataFrame,
    ref_date: datetime.date,
    facility_schema: str = "",
) -> Optional[FacilitySeasonalProfile]:
    """
    Compute per-subclass seasonal demand indices from facility dispensing history.

    Returns None if the facility has fewer than MIN_MONTHS_FOR_FACILITY_INDEX
    months of clean dispensing data (cold-start — caller uses calendar only).

    Methodology
    -----------
    1. Exclude stockout-suppressed days: soh_before > 0, is_stockout_dispense = False,
       dispensed_from_negative_stock = False.
    2. Aggregate to monthly ADC per (subclass, year, month).
    3. Average ADC across years for each calendar month.
    4. Divide by annual average ADC → seasonal index.
    5. Flag months where stockout rate > STOCKOUT_RATE_THRESHOLD (demand suppressed).
    6. Flag months where elevated median soh_before suggests supply-driven spike.
    7. Blend alpha based on years of data: more data → higher facility index weight.

    Parameters
    ----------
    dispensing_df : DataFrame returned by get_dispensing_history() — lowercase columns
    ref_date      : The dashboard ref date (used as metadata only; lookback is in df)
    facility_schema : e.g. "KSH" — stored for traceability
    """
    if dispensing_df is None or dispensing_df.empty:
        return None

    df = dispensing_df.copy()
    df.columns = df.columns.str.lower()

    required_cols = {"quantity_dispensed", "dispensed_at", "therapeutic_subclass"}
    if not required_cols.issubset(df.columns):
        return None

    df["dispensed_at"] = pd.to_datetime(df["dispensed_at"], errors="coerce")
    df = df.dropna(subset=["dispensed_at"])
    if df.empty:
        return None

    df["year"]  = df["dispensed_at"].dt.year
    df["month"] = df["dispensed_at"].dt.month
    df["quantity_dispensed"] = pd.to_numeric(df["quantity_dispensed"], errors="coerce").fillna(0.0)

    # Numeric soh_before — missing treated as zero (conservative: counts as stockout)
    has_soh = "soh_before" in df.columns
    if has_soh:
        df["soh_before"] = pd.to_numeric(df["soh_before"], errors="coerce").fillna(0.0)

    # Mark each row as a stockout-day or not (used for rate calc before filtering)
    def _stockout_mask(frame: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=frame.index)
        if "soh_before" in frame.columns:
            mask |= frame["soh_before"] <= 0
        for flag_col in ("is_stockout_dispense", "dispensed_from_negative_stock"):
            if flag_col in frame.columns:
                mask |= frame[flag_col].fillna(False).astype(bool)
        return mask

    df["_stockout"] = _stockout_mask(df)

    # ── Stockout rate per (subclass, year, month) — computed before filtering ─
    stockout_agg = (
        df.groupby(["therapeutic_subclass", "year", "month"])
        .agg(_n=("quantity_dispensed", "count"), _s=("_stockout", "sum"))
        .reset_index()
    )
    stockout_agg["stockout_rate"] = stockout_agg["_s"] / stockout_agg["_n"].clip(lower=1)

    # ── Calendar-month stockout flag (any year that month was suppressed counts) ─
    bad_months_agg = stockout_agg[stockout_agg["stockout_rate"] > STOCKOUT_RATE_THRESHOLD]
    stockout_flagged_months: Dict[str, Set[int]] = {}
    for _, r in bad_months_agg.iterrows():
        sc = r["therapeutic_subclass"]
        stockout_flagged_months.setdefault(sc, set()).add(int(r["month"]))

    # ── Clean dispensing: exclude stockout-suppressed rows ────────────────────
    clean = df[~df["_stockout"]].copy()
    clean = clean[
        clean["therapeutic_subclass"].notna()
        & (clean["therapeutic_subclass"].str.strip() != "")
    ]

    # Check minimum data threshold
    total_months = clean.groupby(["year", "month"]).ngroups
    if total_months < MIN_MONTHS_FOR_FACILITY_INDEX:
        return None

    # ── Monthly ADC per (subclass, year, month) ───────────────────────────────
    monthly = (
        clean.groupby(["therapeutic_subclass", "year", "month"])["quantity_dispensed"]
        .sum()
        .reset_index()
        .rename(columns={"quantity_dispensed": "monthly_qty"})
    )
    monthly["days_in_month"] = monthly.apply(
        lambda r: _cal.monthrange(int(r["year"]), int(r["month"]))[1], axis=1
    )
    monthly["adc"] = monthly["monthly_qty"] / monthly["days_in_month"].clip(lower=1)

    # ── Average across years for same calendar month ──────────────────────────
    calendar_avg = (
        monthly.groupby(["therapeutic_subclass", "month"])
        .agg(avg_adc=("adc", "mean"), n_years=("adc", "count"))
        .reset_index()
    )

    # ── Annual baseline per subclass ──────────────────────────────────────────
    annual = (
        calendar_avg.groupby("therapeutic_subclass")
        .agg(annual_avg_adc=("avg_adc", "mean"), months_covered=("month", "count"))
        .reset_index()
    )

    # ── Index = monthly_avg_adc / annual_avg_adc ──────────────────────────────
    indexed = calendar_avg.merge(annual, on="therapeutic_subclass")
    indexed = indexed[indexed["annual_avg_adc"] > 0].copy()
    indexed["idx"] = (indexed["avg_adc"] / indexed["annual_avg_adc"]).round(3)

    # ── SOH_BEFORE elevation check ─────────────────────────────────────────────
    # If median(soh_before) in a peak month is SUPPLY_SIGNAL_SOH_RATIO× the annual
    # median, a dispensing spike may be supply-driven (large delivery → more dispensed),
    # not a genuine demand increase.
    supply_signal_months: Dict[str, Set[int]] = {}
    if has_soh and "soh_before" in clean.columns:
        soh_monthly = (
            clean.groupby(["therapeutic_subclass", "month"])["soh_before"]
            .median()
            .reset_index()
            .rename(columns={"soh_before": "med_soh"})
        )
        soh_annual = (
            clean.groupby("therapeutic_subclass")["soh_before"]
            .median()
            .reset_index()
            .rename(columns={"soh_before": "annual_med_soh"})
        )
        soh_check = soh_monthly.merge(soh_annual, on="therapeutic_subclass")
        soh_check["soh_ratio"] = soh_check["med_soh"] / soh_check["annual_med_soh"].clip(lower=0.1)
        soh_check = soh_check.merge(
            indexed[["therapeutic_subclass", "month", "idx"]],
            on=["therapeutic_subclass", "month"],
            how="left",
        )
        flagged_supply = soh_check[
            (soh_check["idx"] >= MIN_INDEX_FOR_SUPPLY_CHECK)
            & (soh_check["soh_ratio"] >= SUPPLY_SIGNAL_SOH_RATIO)
        ]
        for _, r in flagged_supply.iterrows():
            sc = r["therapeutic_subclass"]
            supply_signal_months.setdefault(sc, set()).add(int(r["month"]))

    # ── Assemble index dicts (filter out subclasses with too little data) ─────
    index_by_subclass_month: Dict[str, Dict[int, float]] = {}
    confidence_by_subclass_month: Dict[str, Dict[int, int]] = {}

    for _, row in indexed.iterrows():
        sc = row["therapeutic_subclass"]
        if int(row["months_covered"]) < MIN_MONTHS_PER_SUBCLASS:
            continue
        m = int(row["month"])
        index_by_subclass_month.setdefault(sc, {})[m] = float(row["idx"])
        confidence_by_subclass_month.setdefault(sc, {})[m] = int(row["n_years"])

    if not index_by_subclass_month:
        return None

    # ── Blending alpha: determined by n_distinct_years per subclass ───────────
    alpha_by_subclass: Dict[str, float] = {}
    for sc in index_by_subclass_month:
        n_years = int(monthly[monthly["therapeutic_subclass"] == sc]["year"].nunique())
        alpha_by_subclass[sc] = _ALPHA_BY_YEARS.get(n_years, _ALPHA_MAX)

    annual_adc_by_subclass: Dict[str, float] = {
        row["therapeutic_subclass"]: float(row["annual_avg_adc"])
        for _, row in annual.iterrows()
        if row["therapeutic_subclass"] in index_by_subclass_month
    }

    return FacilitySeasonalProfile(
        index_by_subclass_month=index_by_subclass_month,
        confidence_by_subclass_month=confidence_by_subclass_month,
        stockout_flagged_months=stockout_flagged_months,
        supply_signal_months=supply_signal_months,
        annual_adc_by_subclass=annual_adc_by_subclass,
        alpha_by_subclass=alpha_by_subclass,
        n_months_history=total_months,
        facility_schema=facility_schema,
        computed_at=ref_date,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_ref_date(ref_date: str) -> datetime.date:
    """Convert SQL ref_date expression to datetime.date."""
    if ref_date == "CURRENT_DATE":
        return datetime.date.today()
    return datetime.date.fromisoformat(ref_date.strip("'"))


def _add_months(d: datetime.date, months: int) -> datetime.date:
    """Add calendar months to a date without dateutil dependency."""
    month = d.month - 1 + months
    year  = d.year + month // 12
    month = month % 12 + 1
    return datetime.date(year, month, 1)


def _weeks_to_next_peak(peak_months: List[int], ref_date: datetime.date) -> int:
    """
    Calendar weeks until the next peak month starts.
    Returns 0 if currently inside a peak month.
    Returns the smallest positive week count to any upcoming peak month.
    """
    if not peak_months:
        return 999  # no peak defined — never fires

    if ref_date.month in peak_months:
        return 0

    min_days = None
    for pm in peak_months:
        # Next occurrence of peak month pm after ref_date
        year = ref_date.year
        if pm > ref_date.month:
            target = datetime.date(year, pm, 1)
        else:
            target = datetime.date(year + 1, pm, 1)

        days = (target - ref_date).days
        if min_days is None or days < min_days:
            min_days = days

    return max(0, (min_days or 0) // 7)


# ── Open-Meteo climate signal ─────────────────────────────────────────────────

def get_climate_signal(ref_date: datetime.date) -> Optional[ClimateSignal]:
    """
    Fetch monthly rainfall for Kisumu from Open-Meteo historical archive API.
    Compares current month total to 5-year historical average for that calendar month.

    Returns None on any API or parsing failure — seasonal alerts continue to work
    via the calendar alone (without climate modulation).

    Cache this call at TTL=86400 in the dashboard layer.
    """
    try:
        import requests

        history_start = datetime.date(max(2019, ref_date.year - 5), 1, 1)
        # Archive API lags by 1-2 days; cap end at yesterday
        history_end = min(ref_date, datetime.date.today() - datetime.timedelta(days=1))
        if history_start >= history_end:
            return None

        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   KISUMU_LAT,
                "longitude":  KISUMU_LON,
                "start_date": history_start.isoformat(),
                "end_date":   history_end.isoformat(),
                "daily":      "precipitation_sum",
                "timezone":   "Africa/Nairobi",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame({
            "date":   pd.to_datetime(data["daily"]["time"]),
            "precip": pd.to_numeric(data["daily"]["precipitation_sum"], errors="coerce").fillna(0),
        })
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month

        monthly = df.groupby(["year", "month"])["precip"].sum().reset_index()

        # Historical baseline: average of same calendar month across prior years
        current_month = ref_date.month
        current_year  = ref_date.year
        hist_data = monthly[(monthly["year"] < current_year) & (monthly["month"] == current_month)]
        hist_avg  = float(hist_data["precip"].mean()) if not hist_data.empty else None

        # Current month total — scale up if month is still in progress
        today = datetime.date.today()
        cur_data = monthly[(monthly["year"] == current_year) & (monthly["month"] == current_month)]
        if cur_data.empty:
            # No data for the current month yet (e.g. called on the 1st before archive updates).
            # Comparing last available month's rainfall against this month's historical average
            # produces a meaningless anomaly — return None and fall back to calendar-only mode.
            return None

        current_mm = float(cur_data["precip"].iloc[0])

        # Project incomplete month to full-month estimate, but only once we have
        # at least 7 days of data — fewer days amplify noise by 4-30×.
        if ref_date.year == today.year and ref_date.month == today.month and today.day < 28:
            if today.day >= 7:
                if today.month == 12:
                    days_in_month = 31
                else:
                    days_in_month = (datetime.date(today.year, today.month + 1, 1) - datetime.timedelta(days=1)).day
                current_mm = current_mm * days_in_month / today.day

        if hist_avg is None or hist_avg == 0:
            anomaly_pct = 0.0
        else:
            anomaly_pct = (current_mm - hist_avg) / hist_avg * 100.0

        if anomaly_pct > RAINFALL_HIGH_PCT:
            label = "above average"
        elif anomaly_pct < -RAINFALL_HIGH_PCT:
            label = "below average"
        else:
            label = "near average"

        return ClimateSignal(
            current_month_mm=round(current_mm, 1),
            historical_avg_mm=round(hist_avg or 0.0, 1),
            anomaly_pct=round(anomaly_pct, 1),
            anomaly_label=label,
            current_month_name=ref_date.strftime("%B %Y"),
            data_source="Open-Meteo Historical Archive (archive-api.open-meteo.com)",
            fetched_at=today.isoformat(),
        )

    except Exception:
        return None


# ── Seasonal Engine ───────────────────────────────────────────────────────────

class SeasonalEngine:
    """
    Matches facility SOH products to the Kisumu disease calendar and returns
    structured alerts + demand multipliers for integration into the Order Workbench
    and Insight Engine.
    """

    def _effective_multiplier(
        self,
        disease: SeasonalDisease,
        climate: Optional[ClimateSignal],
    ) -> float:
        mult = disease.demand_multiplier
        if not disease.climate_driven or climate is None:
            return mult
        anomaly = climate.anomaly_pct
        if anomaly > RAINFALL_EXTREME_PCT:
            mult += 0.15
        elif anomaly > RAINFALL_HIGH_PCT:
            mult += 0.08
        return round(mult, 2)

    def _blend_multiplier(
        self,
        calendar_mult: float,
        subclass: str,
        month: int,
        profile: Optional[FacilitySeasonalProfile],
    ) -> float:
        """
        Blend the calendar-based multiplier with the facility's own seasonal index.

        If the profile has a reliable index for (subclass, month):
          blended = alpha * facility_index + (1 - alpha) * calendar_mult
        Otherwise falls back to calendar_mult unchanged.

        Supply-signal months are NOT excluded here — a supply-driven spike still
        represents elevated dispensing; we leave the pharmacist to validate.
        """
        if profile is None:
            return calendar_mult
        facility_idx = profile.get_effective_index(subclass, month)
        if facility_idx is None:
            return calendar_mult
        alpha = profile.get_blending_alpha(subclass)
        blended = alpha * facility_idx + (1.0 - alpha) * calendar_mult
        return round(blended, 3)

    def _find_matching_products(
        self,
        soh_df: pd.DataFrame,
        disease: SeasonalDisease,
    ) -> pd.DataFrame:
        """Return rows in soh_df whose therapeutic class/subclass matches this disease."""
        df = soh_df.copy()

        subclass_mask = pd.Series(False, index=df.index)
        if "THERAPEUTIC_SUBCLASS" in df.columns:
            for kw in disease.subclass_keywords:
                subclass_mask |= (
                    df["THERAPEUTIC_SUBCLASS"].fillna("").str.lower()
                    .str.contains(kw.lower(), regex=False)
                )

        class_mask = pd.Series(False, index=df.index)
        if "THERAPEUTIC_CLASS" in df.columns:
            for kw in disease.class_keywords:
                class_mask |= (
                    df["THERAPEUTIC_CLASS"].fillna("").str.lower()
                    .str.contains(kw.lower(), regex=False)
                )

        return df[subclass_mask | class_mask].copy()

    def match_products(
        self,
        soh_df: pd.DataFrame,
        ref_date: datetime.date,
        climate: Optional[ClimateSignal] = None,
    ) -> List[SeasonalAlert]:
        """
        For each approaching disease season, find products at risk and return
        SeasonalAlert objects. Only fires when:
          - Peak is within disease.warning_weeks ahead, OR currently in peak
          - Adjusted DOS (current_dos / multiplier) is below MIN_ALERTABLE_ADJ_DOS
        """
        df = soh_df.copy()
        df.columns = df.columns.str.upper()
        for col in ["CURRENT_SOH", "CURRENT_SOH_DISPLAY", "AVG_DAILY_UNITS", "DAYS_OF_STOCK"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        alerts: List[SeasonalAlert] = []

        for disease in DISEASE_CALENDAR:
            weeks = _weeks_to_next_peak(disease.peak_months, ref_date)
            if weeks > disease.warning_weeks:
                continue

            matches = self._find_matching_products(df, disease)
            if matches.empty:
                continue

            mult = self._effective_multiplier(disease, climate)
            climate_boosted = mult > disease.demand_multiplier
            rainfall_anomaly = climate.anomaly_pct if climate else None

            for _, row in matches.iterrows():
                dos = row.get("DAYS_OF_STOCK")
                soh = row.get("CURRENT_SOH")
                subclass = str(row.get("THERAPEUTIC_SUBCLASS") or "")
                drug = str(row.get("CANONICAL_NAME") or row.get("PRODUCT_ID") or "")
                pid  = str(row.get("PRODUCT_ID") or "")

                # Compute adjusted DOS under peak demand
                adj_dos = (float(dos) / mult) if (dos is not None and mult > 0) else None

                # Skip if adjusted DOS is still very comfortable
                if adj_dos is not None and adj_dos > MIN_ALERTABLE_ADJ_DOS:
                    continue

                # Severity
                if weeks == 0 and (adj_dos is None or adj_dos < 7):
                    severity = "CRITICAL"
                elif weeks <= 2 or (adj_dos is not None and adj_dos < 14):
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                if climate_boosted and disease.climate_driven and severity == "MEDIUM":
                    severity = "HIGH"

                alerts.append(SeasonalAlert(
                    disease=disease.name,
                    drug_name=drug,
                    product_id=pid,
                    therapeutic_subclass=subclass,
                    weeks_to_peak=weeks,
                    demand_multiplier=mult,
                    current_dos=float(dos) if dos is not None else None,
                    adjusted_dos=round(adj_dos, 1) if adj_dos is not None else None,
                    current_soh=float(soh) if soh is not None else None,
                    severity=severity,
                    climate_boosted=climate_boosted,
                    rainfall_anomaly_pct=rainfall_anomaly,
                ))

        return alerts

    def get_seasonal_multipliers(
        self,
        soh_df: pd.DataFrame,
        ref_date: datetime.date,
        climate: Optional[ClimateSignal] = None,
        facility_profile: Optional[FacilitySeasonalProfile] = None,
    ) -> Dict[str, float]:
        """
        Returns {product_id: multiplier} for products inside an approaching season.
        Products not in any active season are absent (caller should default to 1.0).
        If multiple diseases affect the same product, the highest multiplier wins.

        When facility_profile is provided, the per-subclass facility index is blended
        with the calendar multiplier using the profile's alpha weighting.
        """
        df = soh_df.copy()
        df.columns = df.columns.str.upper()

        if "DAYS_OF_STOCK" in df.columns:
            df["DAYS_OF_STOCK"] = pd.to_numeric(df["DAYS_OF_STOCK"], errors="coerce")

        result: Dict[str, float] = {}
        current_month = ref_date.month

        for disease in DISEASE_CALENDAR:
            weeks = _weeks_to_next_peak(disease.peak_months, ref_date)
            if weeks > disease.warning_weeks:
                continue

            matches = self._find_matching_products(df, disease)
            if matches.empty:
                continue

            calendar_mult = self._effective_multiplier(disease, climate)

            for _, row in matches.iterrows():
                pid = str(row.get("PRODUCT_ID") or "").upper()
                if not pid:
                    continue

                # Blend facility index with calendar multiplier if profile is available
                subclass = str(row.get("THERAPEUTIC_SUBCLASS") or "")
                mult = self._blend_multiplier(
                    calendar_mult=calendar_mult,
                    subclass=subclass,
                    month=current_month,
                    profile=facility_profile,
                )

                dos = row.get("DAYS_OF_STOCK")
                adj_dos = (float(dos) / mult) if (dos is not None and mult > 0) else None
                if adj_dos is not None and adj_dos > MIN_ALERTABLE_ADJ_DOS:
                    continue

                result[pid] = max(result.get(pid, 1.0), mult)

        return result

    def get_disease_summary(
        self,
        soh_df: pd.DataFrame,
        ref_date: datetime.date,
        climate: Optional[ClimateSignal] = None,
        facility_profile: Optional[FacilitySeasonalProfile] = None,
    ) -> List[dict]:
        """
        Aggregate view: one row per approaching disease showing drug count and
        max severity — for surfacing in Briefing insight cards.

        When facility_profile is provided, multipliers are blended per-product and
        a source attribution string is included: "facility data (Xy)" or "calendar".
        """
        df = soh_df.copy()
        df.columns = df.columns.str.upper()
        for col in ["DAYS_OF_STOCK"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        current_month = ref_date.month
        summaries = []

        for disease in DISEASE_CALENDAR:
            weeks = _weeks_to_next_peak(disease.peak_months, ref_date)
            if weeks > disease.warning_weeks:
                continue

            matches = self._find_matching_products(df, disease)
            if matches.empty:
                continue

            calendar_mult = self._effective_multiplier(disease, climate)
            climate_boosted = calendar_mult > disease.demand_multiplier

            # Count drugs at risk; track how many used facility vs calendar signal
            at_risk = 0
            facility_backed_count = 0
            effective_mults = []

            for _, row in matches.iterrows():
                subclass = str(row.get("THERAPEUTIC_SUBCLASS") or "")
                mult = self._blend_multiplier(
                    calendar_mult=calendar_mult,
                    subclass=subclass,
                    month=current_month,
                    profile=facility_profile,
                )
                dos = row.get("DAYS_OF_STOCK")
                adj_dos = (float(dos) / mult) if (dos is not None and mult > 0) else None
                if adj_dos is None or adj_dos < MIN_ALERTABLE_ADJ_DOS:
                    at_risk += 1
                    effective_mults.append(mult)
                    # Check if this product had a facility index (not just calendar)
                    if (
                        facility_profile is not None
                        and facility_profile.get_effective_index(subclass, current_month) is not None
                    ):
                        facility_backed_count += 1

            if at_risk == 0:
                continue

            # Representative multiplier: median across at-risk products
            rep_mult = (
                sorted(effective_mults)[len(effective_mults) // 2]
                if effective_mults
                else calendar_mult
            )

            if weeks == 0:
                severity = "CRITICAL"
            elif weeks <= 3:
                severity = "HIGH"
            else:
                severity = "MEDIUM"

            if climate_boosted and disease.climate_driven and severity == "MEDIUM":
                severity = "HIGH"

            # Source attribution for Briefing chip tooltip
            if facility_profile is not None and facility_backed_count > 0:
                # Round to nearest year; show at least 1yr. n_months covers calendar
                # months with data — 730 days typically yields ~20 distinct months.
                n_yrs = max(1, round(facility_profile.n_months_history / 12))
                source = f"facility data ({n_yrs}yr)"
            else:
                source = "calendar estimate"

            summaries.append({
                "disease": disease.name,
                "weeks_to_peak": weeks,
                "demand_multiplier": rep_mult,
                "drugs_at_risk": at_risk,
                "total_drugs_matched": len(matches),
                "severity": severity,
                "climate_boosted": climate_boosted,
                "description": disease.description,
                "peak_months": disease.peak_months,
                "signal_source": source,
            })

        return summaries

    def get_seasonal_context_map(
        self,
        soh_df: pd.DataFrame,
        ref_date: datetime.date,
        climate: Optional[ClimateSignal] = None,
        facility_profile: Optional[FacilitySeasonalProfile] = None,
    ) -> Dict[str, dict]:
        """
        {PRODUCT_ID: {"disease": str, "weeks_to_peak": int, "demand_mult": float, "climate_boosted": bool}}

        Same filtering logic as get_seasonal_multipliers(). Used to pass full disease
        context into the AI order narrative — the LLM needs to know WHY the quantity
        is larger than baseline, not just that the multiplier was applied.

        When multiple diseases match the same product, the highest-multiplier disease wins.
        Only includes products where the blended multiplier > 1.0 (no calendar-vs-data contradictions).
        """
        df = soh_df.copy()
        df.columns = df.columns.str.upper()
        if "DAYS_OF_STOCK" in df.columns:
            df["DAYS_OF_STOCK"] = pd.to_numeric(df["DAYS_OF_STOCK"], errors="coerce")

        current_month = ref_date.month
        result: Dict[str, dict] = {}

        for disease in DISEASE_CALENDAR:
            weeks = _weeks_to_next_peak(disease.peak_months, ref_date)
            if weeks > disease.warning_weeks:
                continue

            matches = self._find_matching_products(df, disease)
            if matches.empty:
                continue

            calendar_mult = self._effective_multiplier(disease, climate)
            climate_boosted = calendar_mult > disease.demand_multiplier

            for _, row in matches.iterrows():
                pid = str(row.get("PRODUCT_ID") or "").upper()
                if not pid:
                    continue

                subclass = str(row.get("THERAPEUTIC_SUBCLASS") or "")
                mult = self._blend_multiplier(
                    calendar_mult=calendar_mult,
                    subclass=subclass,
                    month=current_month,
                    profile=facility_profile,
                )

                # Only include genuine demand increases — exclude contradictions
                if mult <= 1.0:
                    continue

                dos = row.get("DAYS_OF_STOCK")
                adj_dos = (float(dos) / mult) if (dos is not None and mult > 0) else None
                if adj_dos is not None and adj_dos > MIN_ALERTABLE_ADJ_DOS:
                    continue

                # Highest-multiplier disease wins when multiple seasons overlap
                if pid not in result or mult > result[pid]["demand_mult"]:
                    result[pid] = {
                        "disease":       disease.name,
                        "weeks_to_peak": weeks,
                        "demand_mult":   round(mult, 3),
                        "climate_boosted": climate_boosted,
                    }

        return result

    def get_outlook(
        self,
        ref_date: datetime.date,
        months_ahead: int = 6,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame suitable for the Demand Insights seasonal outlook table.

        Columns: disease, month_label, is_peak, risk_level, demand_multiplier
        Rows cover months_ahead calendar months starting from ref_date's month.
        """
        rows = []
        month_labels = []
        for i in range(months_ahead):
            target = _add_months(ref_date, i)
            month_labels.append(target.strftime("%b %Y"))

        for disease in DISEASE_CALENDAR:
            row_data: dict = {"Disease": disease.name}
            for i, label in enumerate(month_labels):
                target = _add_months(ref_date, i)
                is_peak = target.month in disease.peak_months
                if is_peak:
                    row_data[label] = f"+{round((disease.demand_multiplier - 1) * 100)}%"
                else:
                    row_data[label] = "—"
            rows.append(row_data)

        return pd.DataFrame(rows).set_index("Disease")
